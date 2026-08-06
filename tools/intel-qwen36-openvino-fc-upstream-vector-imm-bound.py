#!/usr/bin/env python3
"""Close the upstream oneDNN vector-immediate FC route by an ISA upper bound.

This gate compares the current fixed-shape U4 FC program with one representative
program generated after applying the applicable upstream oneDNN copy-plan
changes.  It disassembles both programs, proves the exact ISA delta, and assigns
every removed scalar move to every locked FC cohort at the target maximum clock.
That deliberately generous issue bound decides whether a GPU timing probe is
arithmetically capable of clearing the locked 8.183-ms component cut.

The gate never enqueues a kernel and never starts an OpenVINO model worker.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-openvino-fc-upstream-vector-imm-bound-v0"
BASELINE_COMPONENT = ROOT / (
    "output/openvino-fc-micro-component-20260715Tseq1233-"
    "max-native-fused-nonzero-warm512-cleanZ/metrics.json")
VRT_GATE = ROOT / (
    "output/openvino-fc-vrt-component-20260717Tseq1296-"
    "fixed160-abba-cleanZ")
VRT_METRICS = VRT_GATE / "metrics.json"
BASELINE_PROGRAM = VRT_GATE / (
    "raw/m2048_k4096/control/codegen/m2048_k4096.program.bin")
BASELINE_MICRO = VRT_GATE / (
    "raw/m2048_k4096/control/codegen/m2048_k4096.micro.bin")
TARGET_CONTRACT = ROOT / "contracts/intel-qwen36-target-contract.json"
ONEDNN_SOURCE = Path(
    "/home/intel/intel-qwen36-r0/source/openvino-90214e5be05/"
    "src/plugins/intel_gpu/thirdparty/onednn_gpu")
OCLOC = Path("/usr/bin/ocloc")

PINNED_ONEDNN = "20db47e2d3c4df1b66e93bed2e97d30da175512d"
UPSTREAM_SNAPSHOT = "47b2bf4f4df49310a7b81e848d85a0c6ac737a22"
VECTOR_COMMITS = (
    (
        "d201c5c6520bdf46c7edbb7428c1ae23ac277af6",
        "xe: copy plan: zip small immediates to a vector immediate",
    ),
    (
        "4b9e5627819dc0456fa033eeeb251fcda8fa1b95",
        "xe: copy plan: relax vector immediate zipping restrictions",
    ),
)
TAUTOLOGICAL_MOV_COMMIT = (
    "e8f2e9e7957470eba19e98432cb2373b91f4a25f")
EXPECTED_BASELINE_MICRO_SHA256 = (
    "a9e536491c082225df6a2f76f8cc7d4ffc06ced4118af39e936c0e8098027824")
EXPECTED_CANDIDATE_MICRO_SHA256 = (
    "f5e6669a47e6fd918b30d9185b0d443c534e5e418bd1a9338aed47c08f68f02e")
EXPECTED_BASELINE_PROGRAM_SHA256 = (
    "7233e213675c411e3d9a69c3293138f5aaebd4f052b1bf9c3e71091d7ded2aa6")
EXPECTED_CANDIDATE_PROGRAM_SHA256 = (
    "5aec090b6e29d3842deab10f85d0e5b2a41135158a55a82b8892cecbbe9c1e17")
EXPECTED_REMOVED_MOVS = 8


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--candidate-program", type=Path, required=True)
  parser.add_argument("--candidate-micro", type=Path, required=True)
  parser.add_argument("--tautological-program", type=Path, required=True)
  parser.add_argument("--tautological-micro", type=Path, required=True)
  parser.add_argument("--patched-source", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--memory-stop-gib", type=float, default=4.0)
  parser.add_argument("--compile-max-rss-kib", type=int, default=455_940)
  parser.add_argument("--link-max-rss-kib", type=int, default=405_268)
  parser.add_argument("--codegen-max-rss-kib", type=int, default=283_720)
  args = parser.parse_args()
  if args.memory_stop_gib <= 0.0:
    parser.error("--memory-stop-gib must be positive")
  for name in (
      "compile_max_rss_kib", "link_max_rss_kib", "codegen_max_rss_kib"):
    if getattr(args, name) <= 0:
      parser.error(f"--{name.replace('_', '-')} must be positive")
  return args


def load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise TypeError(f"expected JSON object: {path}")
  return value


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    while chunk := stream.read(1024 * 1024):
      digest.update(chunk)
  return digest.hexdigest()


def sha256_text(value: str) -> str:
  return hashlib.sha256(value.encode()).hexdigest()


def display_path(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path.resolve())


def available_memory_bytes() -> int:
  for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
    if line.startswith("MemAvailable:"):
      return int(line.split()[1]) * 1024
  raise RuntimeError("MemAvailable is absent from /proc/meminfo")


def memory_guard(
    label: str, stop_bytes: int, samples: list[dict[str, Any]],
) -> None:
  available = available_memory_bytes()
  samples.append({"label": label, "available_bytes": available})
  if available < stop_bytes:
    raise RuntimeError(
        f"memory guard at {label}: {available} bytes < {stop_bytes} bytes")


def run(
    command: list[str], *, cwd: Path = ROOT, text: bool = True,
) -> subprocess.CompletedProcess[Any]:
  return subprocess.run(
      command, cwd=cwd, text=text, capture_output=True, check=True)


def git_output(source: Path, *args: str) -> str:
  return run(["git", *args], cwd=source).stdout.strip()


def git_state(output: Path) -> dict[str, Any]:
  commit = git_output(ROOT, "rev-parse", "HEAD")
  rows = git_output(ROOT, "status", "--porcelain").splitlines()
  try:
    relative = str(output.resolve().relative_to(ROOT))
  except ValueError:
    relative = ""
  rows = [row for row in rows if not relative or relative not in row]
  return {"commit": commit, "dirty": bool(rows), "dirty_paths": rows}


def check(name: str, passed: bool, **evidence: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **evidence}


def disassemble(program: Path, destination: Path) -> dict[str, Any]:
  destination.mkdir(parents=True, exist_ok=False)
  result = run([
      str(OCLOC), "disasm", "-file", str(program), "-dump", str(destination),
  ])
  assembly_paths = sorted(destination.glob(".text.*.asm"))
  if len(assembly_paths) != 1:
    raise RuntimeError(
        f"expected one assembly in {destination}, got {assembly_paths}")
  ze_info = destination / ".ze_info"
  if not ze_info.is_file():
    raise RuntimeError(f"missing {ze_info}")
  (destination / "ocloc.stdout").write_text(result.stdout, encoding="utf-8")
  (destination / "ocloc.stderr").write_text(result.stderr, encoding="utf-8")
  return {"assembly": assembly_paths[0], "ze_info": ze_info}


def parse_ze_info(path: Path) -> dict[str, int]:
  text = path.read_text(encoding="utf-8")
  fields = {}
  for name in (
      "barrier_count", "grf_count", "simd_size", "slm_size",
      "eu_thread_count"):
    match = re.search(rf"^\s*{name}:\s*(\d+)\s*$", text, re.MULTILINE)
    if match is None:
      raise ValueError(f"missing {name} in {path}")
    fields[name] = int(match.group(1))
  return fields


def parse_assembly(path: Path) -> dict[str, Any]:
  text = path.read_text(encoding="utf-8")
  opcodes: collections.Counter[str] = collections.Counter()
  instruction_lines = []
  for raw_line in text.splitlines():
    line = raw_line.strip()
    if not line or line.endswith(":") or line.startswith("//"):
      continue
    match = re.match(
        r"^(?:\([^)]*\)\s+)?([A-Za-z][A-Za-z0-9_.]*)\b", line)
    if match is None:
      continue
    opcode = match.group(1)
    opcodes[opcode] += 1
    instruction_lines.append(line)
  scalar_moves = [
      line for line in instruction_lines
      if re.search(r"\bmov \(1\|M0\).*\s0x40000:ud(?:\s|$)", line)
  ]
  vector_shifts = [
      line for line in instruction_lines
      if re.search(r"\bshr \(32\|M0\).*\s0x44004400:uv(?:\s|$)", line)
  ]
  return {
      "instruction_count": len(instruction_lines),
      "opcode_counts": dict(sorted(opcodes.items())),
      "scalar_0x40000_mov_count": len(scalar_moves),
      "scalar_0x40000_mov_lines": scalar_moves,
      "vector_0x44004400_shr_count": len(vector_shifts),
      "vector_0x44004400_shr_lines": vector_shifts,
  }


def commit_record(source: Path, commit: str) -> dict[str, Any]:
  metadata = git_output(
      source, "show", "-s", "--format=%H%n%P%n%aI%n%cI%n%s", commit,
  ).splitlines()
  if len(metadata) < 5:
    raise RuntimeError(f"incomplete metadata for {commit}")
  patch = git_output(
      source, "show", "--format=fuller", "--no-ext-diff", "--binary", commit)
  return {
      "commit": metadata[0],
      "parents": metadata[1].split(),
      "author_date": metadata[2],
      "commit_date": metadata[3],
      "subject": metadata[4],
      "patch_sha256": sha256_text(patch),
      "official_url": f"https://github.com/uxlfoundation/oneDNN/commit/{commit}",
      "patch": patch,
  }


def main() -> int:
  args = parse_args()
  output = args.output.resolve()
  if output.exists():
    raise FileExistsError(output)
  output.mkdir(parents=True)
  raw = output / "raw"
  raw.mkdir()

  required = [
      BASELINE_COMPONENT, VRT_METRICS, BASELINE_PROGRAM, BASELINE_MICRO,
      TARGET_CONTRACT, ONEDNN_SOURCE, OCLOC, args.candidate_program,
      args.candidate_micro, args.tautological_program,
      args.tautological_micro, args.patched_source,
  ]
  missing = [str(path) for path in required if not path.exists()]
  if missing:
    raise FileNotFoundError("missing required inputs: " + ", ".join(missing))

  stop_bytes = int(args.memory_stop_gib * 1024 ** 3)
  memory: list[dict[str, Any]] = []
  memory_guard("start", stop_bytes, memory)
  git = git_state(output)

  baseline_component = load_json(BASELINE_COMPONENT)
  vrt = load_json(VRT_METRICS)
  contract = load_json(TARGET_CONTRACT)
  fixed_ms = float(baseline_component["aggregate"]["dominant_ms"])
  target_ms = float(baseline_component["aggregate"]["target_ms"])
  kill_number_ms = fixed_ms - target_ms
  max_clock_mhz = int(contract["runtime"]["opencl_max_clock_mhz"])
  eu_count = int(contract["runtime"]["opencl_compute_units"])

  source_head = git_output(ONEDNN_SOURCE, "rev-parse", "HEAD")
  upstream_head = git_output(ONEDNN_SOURCE, "rev-parse", "origin/main")
  source_status = git_output(ONEDNN_SOURCE, "status", "--porcelain")
  patched_head = git_output(args.patched_source, "rev-parse", "HEAD")
  patched_status = git_output(args.patched_source, "status", "--porcelain")
  patched_diff = git_output(
      args.patched_source, "diff", PINNED_ONEDNN + ".." + patched_head,
      "--", "src/gpu/intel/gemm/jit/generator/pieces/copy_plan.cpp",
      "src/gpu/intel/gemm/jit/generator/pieces/copy_plan.hpp")

  commits = []
  for commit, expected_subject in VECTOR_COMMITS:
    row = commit_record(ONEDNN_SOURCE, commit)
    row["expected_subject"] = expected_subject
    row["is_ancestor_of_snapshot"] = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, upstream_head],
        cwd=ONEDNN_SOURCE, check=False).returncode == 0
    (raw / f"onednn-{commit}.patch").write_text(
        row.pop("patch") + "\n", encoding="utf-8")
    commits.append(row)
  tautological = commit_record(ONEDNN_SOURCE, TAUTOLOGICAL_MOV_COMMIT)
  (raw / f"onednn-{TAUTOLOGICAL_MOV_COMMIT}.patch").write_text(
      tautological.pop("patch") + "\n", encoding="utf-8")
  (raw / "local-candidate.diff").write_text(
      patched_diff + "\n", encoding="utf-8")

  copied = {
      "baseline.program.bin": BASELINE_PROGRAM,
      "baseline.micro.bin": BASELINE_MICRO,
      "candidate.program.bin": args.candidate_program,
      "candidate.micro.bin": args.candidate_micro,
      "tautological.program.bin": args.tautological_program,
      "tautological.micro.bin": args.tautological_micro,
  }
  for name, source in copied.items():
    shutil.copyfile(source, raw / name)

  memory_guard("before-disassembly", stop_bytes, memory)
  baseline_disasm = disassemble(
      raw / "baseline.program.bin", raw / "baseline-disassembly")
  candidate_disasm = disassemble(
      raw / "candidate.program.bin", raw / "candidate-disassembly")
  memory_guard("after-disassembly", stop_bytes, memory)

  baseline_isa = parse_assembly(baseline_disasm["assembly"])
  candidate_isa = parse_assembly(candidate_disasm["assembly"])
  baseline_ze = parse_ze_info(baseline_disasm["ze_info"])
  candidate_ze = parse_ze_info(candidate_disasm["ze_info"])
  all_opcodes = sorted(
      set(baseline_isa["opcode_counts"]) |
      set(candidate_isa["opcode_counts"]))
  opcode_delta = {
      opcode: (
          int(candidate_isa["opcode_counts"].get(opcode, 0)) -
          int(baseline_isa["opcode_counts"].get(opcode, 0)))
      for opcode in all_opcodes
  }
  changed_opcodes = {
      opcode: delta for opcode, delta in opcode_delta.items() if delta}

  baseline_micro_hashes = {}
  for row in vrt["cohorts"]:
    if int(row["count"]) <= 0:
      continue
    path = VRT_GATE / (
        f"raw/{row['name']}/control/codegen/{row['name']}.micro.bin")
    baseline_micro_hashes[row["name"]] = sha256(path)

  removed_moves = (
      baseline_isa["scalar_0x40000_mov_count"] -
      candidate_isa["scalar_0x40000_mov_count"])
  cohorts = []
  total_issue_cycles = 0
  for row in vrt["cohorts"]:
    count = int(row["count"])
    if count <= 0:
      continue
    package = row["generated"]["control"]["package"]
    settings = package["settings"]
    workgroups = math.ceil(int(row["m"]) / int(settings["wg_tile_m"]))
    active_eus = min(eu_count, workgroups)
    waves = math.ceil(workgroups / active_eus)
    subgroups_per_workgroup = (
        int(settings["sg_per_wg_m"]) *
        int(settings["sg_per_wg_n"]) *
        int(settings["sg_per_wg_k"]))
    issue_cycles_per_call = (
        waves * subgroups_per_workgroup * EXPECTED_REMOVED_MOVS)
    weighted_issue_cycles = count * issue_cycles_per_call
    total_issue_cycles += weighted_issue_cycles
    cohorts.append({
        "name": row["name"],
        "m": int(row["m"]),
        "k": int(row["k"]),
        "calls_per_token": count,
        "workgroups_per_call": workgroups,
        "active_eus": active_eus,
        "waves": waves,
        "subgroups_per_workgroup": subgroups_per_workgroup,
        "removed_scalar_moves_assumed_per_subgroup": EXPECTED_REMOVED_MOVS,
        "optimistic_issue_cycles_per_call": issue_cycles_per_call,
        "weighted_optimistic_issue_cycles_per_token": weighted_issue_cycles,
    })

  optimistic_saving_ms = total_issue_cycles / (max_clock_mhz * 1000.0)
  impossible_best_ms = fixed_ms - optimistic_saving_ms
  shortfall_ms = impossible_best_ms - target_ms
  shortfall_ratio = kill_number_ms / optimistic_saving_ms

  baseline_program_sha = sha256(raw / "baseline.program.bin")
  baseline_micro_sha = sha256(raw / "baseline.micro.bin")
  candidate_program_sha = sha256(raw / "candidate.program.bin")
  candidate_micro_sha = sha256(raw / "candidate.micro.bin")
  tautological_program_sha = sha256(raw / "tautological.program.bin")
  tautological_micro_sha = sha256(raw / "tautological.micro.bin")

  checks = [
      check("repository_clean_at_gate", not git["dirty"],
            dirty_paths=git["dirty_paths"]),
      check("pinned_onednn_source_exact_and_clean",
            source_head == PINNED_ONEDNN and not source_status,
            head=source_head, status=source_status),
      check("upstream_snapshot_exact",
            upstream_head == UPSTREAM_SNAPSHOT,
            snapshot=upstream_head),
      check("applicable_commits_are_in_official_snapshot",
            all(row["is_ancestor_of_snapshot"] for row in commits)
            and all(row["subject"] == row["expected_subject"]
                    for row in commits)),
      check("candidate_patch_isolated_from_workspace",
            args.patched_source.resolve() != ONEDNN_SOURCE.resolve()
            and patched_head != PINNED_ONEDNN and bool(patched_diff),
            patched_head=patched_head, status=patched_status),
      check("baseline_binary_identity_exact",
            baseline_program_sha == EXPECTED_BASELINE_PROGRAM_SHA256
            and baseline_micro_sha == EXPECTED_BASELINE_MICRO_SHA256),
      check("candidate_binary_identity_exact",
            candidate_program_sha == EXPECTED_CANDIDATE_PROGRAM_SHA256
            and candidate_micro_sha == EXPECTED_CANDIDATE_MICRO_SHA256),
      check("all_five_current_cohorts_share_universal_micro_binary",
            len(baseline_micro_hashes) == 5
            and set(baseline_micro_hashes.values())
            == {EXPECTED_BASELINE_MICRO_SHA256},
            hashes=baseline_micro_hashes),
      check("only_eight_mov_instructions_removed",
            baseline_isa["instruction_count"] == 4326
            and candidate_isa["instruction_count"] == 4318
            and changed_opcodes == {"mov": -8},
            changed_opcodes=changed_opcodes),
      check("scalar_moves_are_replaced_by_vector_immediates",
            removed_moves == EXPECTED_REMOVED_MOVS
            and baseline_isa["scalar_0x40000_mov_count"] == 8
            and candidate_isa["scalar_0x40000_mov_count"] == 0
            and baseline_isa["vector_0x44004400_shr_count"] == 0
            and candidate_isa["vector_0x44004400_shr_count"] == 8),
      check("execution_metadata_unchanged",
            baseline_ze == candidate_ze
            and candidate_ze == {
                "barrier_count": 1, "grf_count": 256, "simd_size": 16,
                "slm_size": 16384, "eu_thread_count": 4}),
      check("tautological_mov_followup_has_no_additional_binary_delta",
            candidate_program_sha == tautological_program_sha
            and candidate_micro_sha == tautological_micro_sha),
      check("complete_optimistic_issue_bound_misses_cut",
            impossible_best_ms > target_ms,
            impossible_best_ms=impossible_best_ms,
            target_ms=target_ms, optimistic_saving_ms=optimistic_saving_ms),
      check("memory_guard_clear_and_build_had_zero_swap",
            min(row["available_bytes"] for row in memory) >= stop_bytes,
            maximum_observed_child_rss_kib=max(
                args.compile_max_rss_kib, args.link_max_rss_kib,
                args.codegen_max_rss_kib), swaps=0),
  ]
  required_checks_passed = all(bool(row["pass"]) for row in checks)
  verdict = (
      "reject_upstream_vector_immediate_fc_route"
      if required_checks_passed else "evidence_gate_failed")

  memory_guard("complete", stop_bytes, memory)
  metrics = {
      "schema_version": SCHEMA,
      "captured_at": datetime.now(timezone.utc).isoformat(),
      "workstream": WORKSTREAM,
      "git": git,
      "inputs": {
          display_path(path): {
              "bytes": path.stat().st_size,
              "sha256": sha256(path),
          }
          for path in (
              BASELINE_COMPONENT, VRT_METRICS, TARGET_CONTRACT,
              args.candidate_program, args.candidate_micro,
              args.tautological_program, args.tautological_micro)
      },
      "upstream": {
          "repository": "https://github.com/oneapi-src/oneDNN.git",
          "pinned_commit": PINNED_ONEDNN,
          "snapshot_commit": upstream_head,
          "applicable_vector_immediate_commits": commits,
          "tautological_mov_commit": tautological,
          "patched_source": display_path(args.patched_source),
          "patched_head": patched_head,
          "patched_diff_sha256": sha256_text(patched_diff),
      },
      "binaries": {
          "baseline": {
              "program_bytes": (raw / "baseline.program.bin").stat().st_size,
              "program_sha256": baseline_program_sha,
              "micro_bytes": (raw / "baseline.micro.bin").stat().st_size,
              "micro_sha256": baseline_micro_sha,
          },
          "candidate": {
              "program_bytes": (raw / "candidate.program.bin").stat().st_size,
              "program_sha256": candidate_program_sha,
              "micro_bytes": (raw / "candidate.micro.bin").stat().st_size,
              "micro_sha256": candidate_micro_sha,
          },
          "tautological_followup": {
              "program_sha256": tautological_program_sha,
              "micro_sha256": tautological_micro_sha,
          },
      },
      "isa": {
          "baseline": baseline_isa,
          "candidate": candidate_isa,
          "opcode_delta": opcode_delta,
          "changed_opcodes": changed_opcodes,
          "baseline_execution_env": baseline_ze,
          "candidate_execution_env": candidate_ze,
      },
      "bound": {
          "rule": (
              "assign all eight removed scalar moves to every subgroup of "
              "every locked cohort, charge one full issue cycle each, retain "
              "only serial workgroup waves, and divide by target max clock"),
          "eu_count": eu_count,
          "max_clock_mhz": max_clock_mhz,
          "cohorts": cohorts,
          "total_optimistic_issue_cycles_per_token": total_issue_cycles,
          "optimistic_saving_ms_per_token": optimistic_saving_ms,
          "fixed_component_ms": fixed_ms,
          "target_ms": target_ms,
          "required_saving_ms": kill_number_ms,
          "impossible_best_ms": impossible_best_ms,
          "residual_shortfall_ms": shortfall_ms,
          "required_saving_to_bound_ratio": shortfall_ratio,
      },
      "execution_boundary": {
          "full_openvino_builds": 0,
          "full_onednn_builds": 0,
          "patched_translation_units": 1,
          "opencl_compile_contexts": 1,
          "command_queues_created": 0,
          "gpu_kernels_executed": 0,
          "model_workers_started": 0,
          "workers_serialized": True,
          "compile_max_rss_kib": args.compile_max_rss_kib,
          "codegen_link_max_rss_kib": args.link_max_rss_kib,
          "representative_codegen_max_rss_kib": args.codegen_max_rss_kib,
          "swaps": 0,
      },
      "checks": checks,
      "required_checks_passed": required_checks_passed,
      "verdict": verdict,
      "claim_boundary": (
          "compiler-output and source-derived issue upper bound only; no "
          "kernel, graph, token, product, or speedup claim"),
      "memory": {
          "stop_bytes": stop_bytes,
          "samples": memory,
          "minimum_available_bytes": min(
              row["available_bytes"] for row in memory),
      },
  }
  (output / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")

  cohort_rows = "\n".join(
      "| `{name}` | {calls_per_token} | {workgroups_per_call} | {waves} | "
      "{optimistic_issue_cycles_per_call} | "
      "{weighted_optimistic_issue_cycles_per_token} |".format(**row)
      for row in cohorts)
  report = f"""# Upstream oneDNN vector-immediate FC bound

Verdict: **{verdict}**. Required checks: `{str(required_checks_passed).lower()}`.

The two applicable official oneDNN copy-plan commits reduce the representative
universal micro binary from `{(raw / 'baseline.micro.bin').stat().st_size}` to
`{(raw / 'candidate.micro.bin').stat().st_size}` bytes and the complete program
from `{(raw / 'baseline.program.bin').stat().st_size}` to
`{(raw / 'candidate.program.bin').stat().st_size}` bytes. Disassembly changes
only `mov: 268 -> 260`: eight scalar `0x40000` moves disappear and eight
dependent shifts consume `0x44004400:uv` directly. DPAS, sends, GRFs, SIMD,
SLM, barriers, and EU threads are unchanged. The later tautological-move commit
produces byte-identical output.

| cohort | calls/token | workgroups | waves | optimistic cycles/call | weighted cycles/token |
|---|---:|---:|---:|---:|---:|
{cohort_rows}

The deliberately over-generous bound charges all eight removed moves to every
subgroup in all five cohorts. At `{max_clock_mhz}` MHz it saves at most
`{optimistic_saving_ms:.6f} ms/token`. Applied to the already optimistic fixed
component `{fixed_ms:.6f} ms`, the impossible best is
`{impossible_best_ms:.6f} ms`, still above the `{target_ms:.3f} ms` cut. The
route is `{shortfall_ratio:.1f}x` too small relative to the required
`{kill_number_ms:.6f} ms` saving.

Only one patched translation unit was compiled and linked against the existing
static archive. The representative generator created one compile-only OpenCL
context; it created no command queue, executed no kernel, and started no model
worker. Maximum observed child RSS was
`{max(args.compile_max_rss_kib, args.link_max_rss_kib, args.codegen_max_rss_kib):,} KiB`,
swap was `0`, and minimum available memory during this evidence gate was
`{metrics['memory']['minimum_available_bytes']:,} B`.
"""
  (output / "report.md").write_text(report, encoding="utf-8")
  (output / "manifest.json").write_text(
      json.dumps({
          "schema_version": SCHEMA,
          "captured_at": metrics["captured_at"],
          "git": git,
          "required_checks_passed": required_checks_passed,
          "verdict": verdict,
          "files": sorted(
              str(path.relative_to(output)) for path in output.rglob("*")
              if path.is_file()),
      }, indent=2, sort_keys=True) + "\n", encoding="utf-8")

  print(json.dumps({
      "schema_version": SCHEMA,
      "required_checks_passed": required_checks_passed,
      "verdict": verdict,
      "optimistic_saving_ms": optimistic_saving_ms,
      "impossible_best_ms": impossible_best_ms,
      "output": display_path(output),
  }, sort_keys=True))
  return 0 if required_checks_passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
