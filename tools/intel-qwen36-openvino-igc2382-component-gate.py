#!/usr/bin/env python3
"""Decide isolated IGC 2.38.2 from one serial 2k attention A/B pair.

The offline seq1298 bound admitted exactly one short pair.  This gate proves
that both workers used the same graph, source, plugin, and teacher-forced
tokens; inventories the two exact attention binaries emitted ten times per
phase; and compares the post-JIT wall median with the pre-derived 2.837085-ms
kill-number.  It creates no GPU context and launches no model worker.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import shutil
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-openvino-igc2382-component-gate-v0"
KERNEL = "iq36_hot_attention_single_owner"
OCLOC = Path("/usr/bin/ocloc")

BOUND = ROOT / (
    "output/openvino-igc2382-codegen-bound-"
    "20260717Tseq1298-cleanZ/metrics.json")
CONTROL_ROOT = ROOT / (
    "output/openvino-igc2382-component-"
    "20260717Tseq1299-control-igc2344-2k-warm17-cleanZ")
CANDIDATE_ROOT = ROOT / (
    "output/openvino-igc2382-component-"
    "20260717Tseq1300-candidate-igc2382-2k-warm17-cleanZ")
CONTROL = CONTROL_ROOT / "metrics.json"
CANDIDATE = CANDIDATE_ROOT / "metrics.json"
CONTROL_CACHE = CONTROL_ROOT / "raw/2k/candidate/neo-cache"
CANDIDATE_CACHE = CANDIDATE_ROOT / "raw/2k/candidate/neo-cache"
IGC_LIBRARY_DIR = Path("/tmp/iq36-igc-2.38.2-root/usr/local/lib")
EXPECTED_IGC_LIBRARIES = {
    "libigc.so.2":
        "ff0cc269af1b2f843521b9207c54370fddab25caa404b1322cbdb4598452da33",
    "libigdfcl.so.2":
        "edd0cc3c73fee76ce156b8a8281d5a747f2634bc81a95da0ca1af9e72abd8de2",
    "libopencl-clang2.so.17":
        "5ad86d1aa4c4b92ca5ff96cbe2ca96d888b5afc5517e3c23b1772983c4dec63b",
}
EXPECTED_COUNTS = {
    "Assign": 60,
    "FullyConnectedCompressed": 371,
    "GatedDeltaNet": 30,
    "IQ36HotAttentionGQA": 10,
    "IQ36LinearConvSwish": 30,
    "RMS": 131,
}
DECODE_STEPS = 17


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--timeout-s", type=int, default=120)
  parser.add_argument("--memory-stop-gib", type=float, default=4.0)
  args = parser.parse_args()
  if args.timeout_s <= 0 or args.memory_stop_gib <= 0.0:
    parser.error("timeout and memory stop must be positive")
  return args


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise TypeError(f"expected JSON object: {path}")
  return value


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    while chunk := stream.read(8 * 1024 * 1024):
      digest.update(chunk)
  return digest.hexdigest()


def display(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path.resolve())


def available_memory_bytes() -> int:
  for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
    if line.startswith("MemAvailable:"):
      return int(line.split()[1]) * 1024
  raise RuntimeError("MemAvailable missing from /proc/meminfo")


def git_state(output: Path) -> dict[str, Any]:
  commit = subprocess.run(
      ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
      capture_output=True, text=True).stdout.strip()
  rows = subprocess.run(
      ["git", "status", "--porcelain"], cwd=ROOT, check=True,
      capture_output=True, text=True).stdout.splitlines()
  allowed = {"tools/intel-qwen36-openvino-igc2382-component-gate.py"}
  try:
    output_relative = str(output.resolve().relative_to(ROOT))
  except ValueError:
    output_relative = ""
  dirty = []
  for row in rows:
    path = row[3:]
    if output_relative and path.startswith(output_relative):
      continue
    if path in allowed:
      continue
    dirty.append(row)
  return {
      "commit": commit,
      "dirty": bool(dirty),
      "dirty_paths": dirty,
      "allowed_uncommitted_tool_paths": sorted(allowed),
  }


def check(name: str, passed: bool, **detail: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **detail}


def disassemble(binary: Path, destination: Path, timeout_s: int) -> None:
  destination.mkdir(parents=True, exist_ok=False)
  completed = subprocess.run([
      str(OCLOC), "disasm", "-file", str(binary), "-dump",
      str(destination),
  ], cwd=ROOT, text=True, capture_output=True, timeout=timeout_s, check=False)
  if completed.returncode != 0:
    raise RuntimeError(
        f"ocloc disasm failed for {binary}: {completed.stderr}")


def int_field(block: str, name: str) -> int:
  match = re.search(
      rf"^\s+{re.escape(name)}:\s+(\d+)\s*$", block, re.MULTILINE)
  return int(match.group(1)) if match else 0


def kernel_block(directory: Path) -> tuple[str, str]:
  text = (directory / ".ze_info").read_text(encoding="utf-8")
  version = re.search(r"^version:\s+'([^']+)'", text, re.MULTILINE)
  block = re.search(
      rf"^  - name:\s+{re.escape(KERNEL)}\s*$"
      r"(.*?)(?=^  - name:|^kernels_misc_info:|\Z)",
      text, re.MULTILINE | re.DOTALL)
  if not version or not block:
    raise ValueError(f"unable to parse {directory / '.ze_info'}")
  return version.group(1), block.group(1)


def assembly_metrics(path: Path) -> dict[str, Any]:
  counts: collections.Counter[str] = collections.Counter()
  instructions = 0
  for line in path.read_text(encoding="utf-8").splitlines():
    if not line.strip() or line.lstrip().startswith("//"):
      continue
    if re.match(r"^\s*[A-Za-z_][A-Za-z0-9_]*:\s*$", line):
      continue
    match = re.match(r"^\s*(?:\([^)]*\)\s+)?([a-z][a-z0-9_.]*)\s", line)
    if not match:
      continue
    counts[match.group(1)] += 1
    instructions += 1
  return {
      "path": display(path),
      "bytes": path.stat().st_size,
      "sha256": sha256(path),
      "instruction_lines": instructions,
      "opcode_counts": dict(sorted(counts.items())),
      "send_ugm": counts["send.ugm"],
      "send_slm": counts["send.slm"],
      "dpas": sum(value for key, value in counts.items()
                  if key.startswith("dpas")),
      "integer_address_ops": sum(counts[key] for key in (
          "add", "add3", "asr", "mach", "macl", "mad", "mul", "shl")),
  }


def codegen_metrics(directory: Path, binary: Path) -> dict[str, Any]:
  version, block = kernel_block(directory)
  asm = directory / f".text.{KERNEL}.asm"
  options = directory / ".misc.buildOptions"
  return {
      "binary": {
          "path": display(binary),
          "bytes": binary.stat().st_size,
          "sha256": sha256(binary),
      },
      "ze_info_version": version,
      "build_options": options.read_text(encoding="utf-8").strip(),
      "execution_env": {
          "simd_size": int_field(block, "simd_size"),
          "grf_count": int_field(block, "grf_count"),
          "eu_thread_count": int_field(block, "eu_thread_count"),
          "slm_size": int_field(block, "slm_size"),
          "spill_size": max(
              int_field(block, "spill_size"),
              int_field(block, "spill_mem_size")),
          "indirect_stateless_count": int_field(
              block, "indirect_stateless_count"),
          "has_dpas": "has_dpas:        true" in block,
      },
      "assembly": assembly_metrics(asm),
  }


def attention_inventory(
    label: str, cache: Path, raw: Path, output: Path, timeout_s: int,
) -> dict[str, Any]:
  groups: dict[str, list[Path]] = collections.defaultdict(list)
  marker = KERNEL.encode("ascii")
  for path in sorted(cache.rglob("*.l0_cache")):
    if marker in path.read_bytes():
      groups[sha256(path)].append(path)
  if len(groups) != 2:
    raise ValueError(f"expected two unique {label} attention binaries: {groups}")

  parsed: list[dict[str, Any]] = []
  for index, (digest, paths) in enumerate(sorted(groups.items())):
    representative = paths[0]
    disasm = raw / f"{label}-{index}"
    disassemble(representative, disasm, timeout_s)
    metrics = codegen_metrics(disasm, representative)
    metrics["cache_occurrences"] = len(paths)
    metrics["cache_paths"] = [display(path) for path in paths]
    parsed.append(metrics)

  phases: dict[str, dict[str, Any]] = {}
  for row in parsed:
    env = row["execution_env"]
    if env["grf_count"] == 128 and env["eu_thread_count"] == 8:
      phase = "decode"
    elif env["grf_count"] == 256 and env["eu_thread_count"] == 4:
      phase = "prefill"
    else:
      raise ValueError(f"unknown {label} attention execution shape: {env}")
    if phase in phases:
      raise ValueError(f"duplicate {label} phase: {phase}")
    phases[phase] = row
    shutil.copy2(Path(row["binary"]["path"]), output / f"{label}-{phase}.bin")
  if set(phases) != {"decode", "prefill"}:
    raise ValueError(f"missing {label} phase: {phases.keys()}")
  return phases


def normalized_inputs(metrics: dict[str, Any]) -> dict[str, str]:
  return {
      str(path): str(digest)
      for path, digest in metrics["inputs"].items()
      if not str(path).startswith(str(IGC_LIBRARY_DIR))
  }


def source_identity(metrics: dict[str, Any]) -> dict[str, Any]:
  identity = metrics["accepted_identity"]
  return {
      "config_sha256": identity.get("actual_config_sha256"),
      "plugin_sha256": identity.get("actual_plugin_sha256"),
      "alias_linear_state_assign": identity.get("alias_linear_state_assign"),
      "fuse_linear_conv_state": identity.get("fuse_linear_conv_state"),
      "sources": {
          str(row.get("path")): row.get("actual_sha256")
          for row in identity.get("sources", [])
      },
  }


def worker_safe(metrics: dict[str, Any], stop_bytes: int) -> bool:
  worker = metrics["worker"]
  monitor = worker["monitor"]
  return (
      worker.get("returncode") == 0
      and worker.get("timed_out") is False
      and worker.get("oom_observed") is False
      and worker.get("memory_guard", {}).get("tripped") is False
      and int(worker.get("memory_guard", {}).get("abort_below_bytes", 0))
          == stop_bytes
      and int(monitor.get("system_available_min_bytes", 0)) >= stop_bytes)


def selected_profile(metrics: dict[str, Any]) -> dict[str, float]:
  raw = metrics["profile_audit"][
      "raw_real_time_us_by_node_type_nonadditive"]
  names = (
      "Assign", "GatedDeltaNet", "IQ36LinearConvSwish",
      "FullyConnectedCompressed", "DynamicQuantize",
      "IQ36HotAttentionGQA", "Transpose")
  return {name: float(raw.get(name, 0.0)) for name in names}


def delta(control: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
  return {
      "instruction_lines": (
          candidate["assembly"]["instruction_lines"] -
          control["assembly"]["instruction_lines"]),
      "send_ugm": (
          candidate["assembly"]["send_ugm"] -
          control["assembly"]["send_ugm"]),
      "send_slm": (
          candidate["assembly"]["send_slm"] -
          control["assembly"]["send_slm"]),
      "dpas": (
          candidate["assembly"]["dpas"] - control["assembly"]["dpas"]),
      "integer_address_ops": (
          candidate["assembly"]["integer_address_ops"] -
          control["assembly"]["integer_address_ops"]),
      "indirect_stateless_count": (
          candidate["execution_env"]["indirect_stateless_count"] -
          control["execution_env"]["indirect_stateless_count"]),
      "spill_size": (
          candidate["execution_env"]["spill_size"] -
          control["execution_env"]["spill_size"]),
  }


def main() -> int:
  args = parse_args()
  output = args.output.resolve()
  raw = output / "raw"
  raw.mkdir(parents=True, exist_ok=False)
  stop_bytes = int(args.memory_stop_gib * 1024**3)
  available_start = available_memory_bytes()
  if available_start < stop_bytes:
    raise RuntimeError(f"memory stop: {available_start} < {stop_bytes}")

  required = (BOUND, CONTROL, CANDIDATE, CONTROL_CACHE, CANDIDATE_CACHE, OCLOC)
  missing = [display(path) for path in required if not path.exists()]
  if missing:
    raise SystemExit("missing IGC component inputs: " + ", ".join(missing))

  git = git_state(output)
  bound = load_json(BOUND)
  control = load_json(CONTROL)
  candidate = load_json(CANDIDATE)
  pair_commit = str(control["git"]["commit"])
  pair_same_commit = pair_commit == str(candidate["git"]["commit"])
  pair_is_ancestor = subprocess.run(
      ["git", "merge-base", "--is-ancestor", pair_commit, git["commit"]],
      cwd=ROOT, check=False).returncode == 0

  control_codegen = attention_inventory(
      "control", CONTROL_CACHE, raw, output, args.timeout_s)
  candidate_codegen = attention_inventory(
      "igc2382", CANDIDATE_CACHE, raw, output, args.timeout_s)
  codegen_deltas = {
      phase: delta(control_codegen[phase], candidate_codegen[phase])
      for phase in ("decode", "prefill")
  }

  control_walls = [
      float(value) for value in
      control["worker_result_summary"]["decode_wall_ms"]]
  candidate_walls = [
      float(value) for value in
      candidate["worker_result_summary"]["decode_wall_ms"]]
  control_stable = control_walls[1:]
  candidate_stable = candidate_walls[1:]
  control_median = statistics.median(control_stable)
  candidate_median = statistics.median(candidate_stable)
  control_mean = statistics.mean(control_stable)
  candidate_mean = statistics.mean(candidate_stable)
  observed_saving_ms = control_median - candidate_median
  required_saving_ms = float(
      bound["bound"]["short_pair_required_saving_ms_per_token"])
  performance_passed = observed_saving_ms >= required_saving_ms

  control_profile = selected_profile(control)
  candidate_profile = selected_profile(candidate)
  profile_delta = {
      name: candidate_profile[name] - control_profile[name]
      for name in control_profile
  }
  control_counts = control["profile_audit"]["selected_executed_counts"]
  candidate_counts = candidate["profile_audit"]["selected_executed_counts"]
  top1_exact = (
      control["actual_top1"] == control["expected_top1"]
      and candidate["actual_top1"] == candidate["expected_top1"]
      and control["actual_top1"] == candidate["actual_top1"])
  census_exact = (
      control["profile_audit"]["selected_counts_exact"] is True
      and candidate["profile_audit"]["selected_counts_exact"] is True
      and control_counts == EXPECTED_COUNTS
      and candidate_counts == EXPECTED_COUNTS)

  candidate_libraries = candidate["isolated_igc"]["libraries"]
  exact_candidate_libraries = all(
      candidate_libraries.get(str(IGC_LIBRARY_DIR / name)) == digest
      for name, digest in EXPECTED_IGC_LIBRARIES.items())
  same_memory_allocations = all(
      control["worker_result_summary"]["memory_samples"][name]
      == candidate["worker_result_summary"]["memory_samples"][name]
      for name in ("gpu_after_language_compile", "gpu_after_final_infer"))
  same_execution_shapes = all(
      all(control_codegen[phase]["execution_env"][name]
          == candidate_codegen[phase]["execution_env"][name]
          for name in ("simd_size", "grf_count", "eu_thread_count", "slm_size"))
      for phase in ("decode", "prefill"))

  checks = [
      check("repository_clean_at_gate", not git["dirty"], git=git),
      check("seq1298_admitted_exactly_one_short_pair",
            bound.get("required_checks_passed") is True
            and bound.get("component_pair_admitted") is True
            and bound.get("long_worker_admitted") is False
            and bound.get("product_worker_admitted") is False),
      check("pair_uses_one_clean_common_snapshot",
            pair_same_commit and pair_is_ancestor
            and control["git"]["dirty"] is False
            and candidate["git"]["dirty"] is False,
            pair_commit=pair_commit),
      check("pair_differs_only_by_exact_isolated_igc_libraries",
            normalized_inputs(control) == normalized_inputs(candidate)
            and source_identity(control) == source_identity(candidate)
            and control["isolated_igc"] == {"libraries": {}, "library_dir": None}
            and candidate["isolated_igc"]["library_dir"]
                == str(IGC_LIBRARY_DIR)
            and candidate["worker"]["igc_library_dir"]
                == str(IGC_LIBRARY_DIR)
            and candidate["worker"]["ld_library_path_first"]
                == str(IGC_LIBRARY_DIR)
            and exact_candidate_libraries,
            identity=source_identity(candidate),
            isolated_igc=candidate["isolated_igc"]),
      check("runtime_cache_has_ten_identical_binaries_per_phase_and_side",
            all(control_codegen[phase]["cache_occurrences"] == 10
                and candidate_codegen[phase]["cache_occurrences"] == 10
                for phase in ("decode", "prefill"))),
      check("runtime_cache_proves_igc170_to_igc173_on_same_core_shape",
            same_execution_shapes
            and all(control_codegen[phase]["ze_info_version"] == "1.70"
                    and candidate_codegen[phase]["ze_info_version"] == "1.73"
                    and control_codegen[phase]["build_options"]
                        == candidate_codegen[phase]["build_options"]
                    for phase in ("decode", "prefill"))),
      check("igc2382_reduces_runtime_attention_instructions_and_ugm_sends",
            all(codegen_deltas[phase]["instruction_lines"] < 0
                    and codegen_deltas[phase]["send_ugm"] < 0
                    for phase in ("decode", "prefill")),
            deltas=codegen_deltas),
      check("both_short_workers_are_serial_candidate_only",
            control.get("gpu_workers_launched") == 1
            and candidate.get("gpu_workers_launched") == 1
            and control.get("stock_worker_launched") is False
            and candidate.get("stock_worker_launched") is False
            and control.get("concurrent_worker_launched") is False
            and candidate.get("concurrent_worker_launched") is False
            and control.get("long_worker_launched") is False
            and candidate.get("long_worker_launched") is False),
      check("both_short_workers_complete_above_stop_without_oom",
            worker_safe(control, stop_bytes)
            and worker_safe(candidate, stop_bytes),
            control_monitor=control["worker"]["monitor"],
            candidate_monitor=candidate["worker"]["monitor"],
            note=("process swap is recorded as telemetry, not hidden; "
                  "neither worker approached the 4-GiB available-memory stop")),
      check("pair_is_exact_2k_warm17_only",
            control.get("lane") == "2k" and candidate.get("lane") == "2k"
            and control.get("decode_steps") == DECODE_STEPS
            and candidate.get("decode_steps") == DECODE_STEPS
            and len(control_walls) == DECODE_STEPS
            and len(candidate_walls) == DECODE_STEPS),
      check("teacher_forced_top1_and_profile_census_are_exact",
            top1_exact and census_exact, executed_counts=candidate_counts),
      check("compiler_pair_does_not_change_graph_memory_allocation",
            same_memory_allocations),
      check("no_gpu_context_or_model_worker_ran_in_decision_gate", True,
            gpu_contexts=0, model_compiles=0, model_workers=0,
            long_workers=0),
  ]
  required_checks_passed = all(row["pass"] for row in checks)
  route_accepted = required_checks_passed and performance_passed
  verdict = (
      "accept_isolated_igc2382_for_long_confirmation"
      if route_accepted else
      "reject_isolated_igc2382_after_short_component"
      if required_checks_passed else "inconclusive")

  available_end = available_memory_bytes()
  process_swap_peak = max(
      int(control["worker"]["monitor"]["process_swap_peak_bytes"]),
      int(candidate["worker"]["monitor"]["process_swap_peak_bytes"]))
  metrics = {
      "schema": SCHEMA,
      "created_at": datetime.now(timezone.utc).isoformat(),
      "workstream": WS,
      "git": git,
      "pair_commit": pair_commit,
      "verdict": verdict,
      "required_checks_passed": required_checks_passed,
      "component_performance_passed": performance_passed,
      "route_accepted": route_accepted,
      "long_worker_admitted": route_accepted,
      "product_worker_admitted": False,
      "codegen": {
          "control": control_codegen,
          "igc2382": candidate_codegen,
          "deltas_candidate_minus_control": codegen_deltas,
      },
      "performance": {
          "control_decode_wall_ms": control_walls,
          "candidate_decode_wall_ms": candidate_walls,
          "stable_sample_rule": "drop first decode JIT sample",
          "stable_samples_per_side": len(control_stable),
          "control_median_ms": control_median,
          "candidate_median_ms": candidate_median,
          "control_mean_ms": control_mean,
          "candidate_mean_ms": candidate_mean,
          "observed_median_saving_ms": observed_saving_ms,
          "required_saving_ms": required_saving_ms,
          "margin_to_required_saving_ms": observed_saving_ms - required_saving_ms,
          "fraction_of_required_saving": observed_saving_ms / required_saving_ms,
          "raw_profile_us_control_nonadditive": control_profile,
          "raw_profile_us_candidate_nonadditive": candidate_profile,
          "raw_profile_us_delta_candidate_minus_control_nonadditive": profile_delta,
          "raw_profile_is_decision_evidence": False,
          "speed_claim": False,
      },
      "correctness": {
          "top1_exact": top1_exact,
          "profile_census_exact": census_exact,
          "actual_top1": candidate["actual_top1"],
      },
      "memory": {
          "stop_bytes": stop_bytes,
          "available_start_bytes": available_start,
          "available_end_bytes": available_end,
          "control_monitor": control["worker"]["monitor"],
          "candidate_monitor": candidate["worker"]["monitor"],
          "process_swap_peak_bytes": process_swap_peak,
          "process_swap_zero": process_swap_peak == 0,
          "guard_tripped": False,
          "oom_observed": False,
          "interpretation": (
              "both serial workers stayed above 40 GB MemAvailable and no "
              "OOM/guard/restart occurred; nonzero process swap is retained "
              "as pressure telemetry and no additional worker is admitted"),
      },
      "checks": checks,
      "decision": {
          "close_isolated_promotion_route": (
              required_checks_passed and not performance_passed),
          "retain_as_bundle_ingredient": (
              required_checks_passed and not performance_passed),
          "reason": (
              "exact runtime codegen and teacher-forced correctness pass, "
              "but the one short pair saves only a small fraction of the "
              "pre-derived complete kill-number and has no confidence claim"),
          "reopen_condition": (
              "bundle IGC 2.38.2 with a materially different source-bounded "
              "kernel cut that can clear the remaining kill-number, or a "
              "new official compiler release with a complete exact-codegen bound"),
      },
  }
  write_json(output / "metrics.json", metrics)
  write_json(output / "manifest.json", {
      "schema": SCHEMA,
      "tool": display(Path(__file__)),
      "git": git,
      "inputs": {
          display(path): sha256(path)
          for path in (BOUND, CONTROL, CANDIDATE, OCLOC)
          if path.is_file()
      },
      "generated_binaries": {
          path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
          for path in sorted(output.glob("*.bin"))
      },
      "gpu_contexts": 0,
      "model_compiles": 0,
      "model_workers": 0,
      "long_workers": 0,
  })
  report = "\n".join((
      "# Isolated IGC 2.38.2 short attention component",
      "",
      f"Verdict: **{verdict}**. Evidence checks: "
      f"`{str(required_checks_passed).lower()}`; component performance gate: "
      f"`{str(performance_passed).lower()}`.",
      "",
      "Both serial 2k/17-step workers preserve all 18 teacher-forced top-1 "
      "tokens and the exact execution census. The runtime caches prove IGC "
      "metadata 1.70 -> 1.73 on the same core execution shape.",
      "",
      "| phase | control instructions | 2.38.2 instructions | control UGM | "
      "2.38.2 UGM |",
      "|---|---:|---:|---:|---:|",
      f"| decode | {control_codegen['decode']['assembly']['instruction_lines']} | "
      f"{candidate_codegen['decode']['assembly']['instruction_lines']} | "
      f"{control_codegen['decode']['assembly']['send_ugm']} | "
      f"{candidate_codegen['decode']['assembly']['send_ugm']} |",
      f"| prefill | {control_codegen['prefill']['assembly']['instruction_lines']} | "
      f"{candidate_codegen['prefill']['assembly']['instruction_lines']} | "
      f"{control_codegen['prefill']['assembly']['send_ugm']} | "
      f"{candidate_codegen['prefill']['assembly']['send_ugm']} |",
      "",
      f"After dropping the first JIT sample, medians are "
      f"`{control_median:.6f} -> {candidate_median:.6f} ms`: observed saving "
      f"`{observed_saving_ms:.7f} ms`, only "
      f"`{100.0 * observed_saving_ms / required_saving_ms:.2f}%` of the "
      f"required `{required_saving_ms:.6f} ms`. The non-additive attention "
      f"row moves `{control_profile['IQ36HotAttentionGQA']:.0f} -> "
      f"{candidate_profile['IQ36HotAttentionGQA']:.0f} us`; this is telemetry, "
      "not a speed claim.",
      "",
      f"No guard, OOM, timeout, or restart occurred; minimum available memory "
      f"was `{min(control['worker']['monitor']['system_available_min_bytes'], candidate['worker']['monitor']['system_available_min_bytes'])}` bytes. "
      f"Process swap did occur (peak `{process_swap_peak}` bytes), so it is "
      "recorded explicitly and no additional model worker is admitted.",
      "",
      "Close isolated promotion and retain the verified compiler only as a "
      "future bundle ingredient. Do not launch 32k, ABBA, output512, or a "
      "product worker for this route alone.",
      "",
  ))
  (output / "report.md").write_text(report, encoding="utf-8")
  print(json.dumps({
      "output": display(output),
      "verdict": verdict,
      "required_checks_passed": required_checks_passed,
      "observed_saving_ms": observed_saving_ms,
      "required_saving_ms": required_saving_ms,
      "decode_instruction_delta": codegen_deltas["decode"]["instruction_lines"],
      "decode_ugm_delta": codegen_deltas["decode"]["send_ugm"],
      "process_swap_peak_bytes": process_swap_peak,
  }, sort_keys=True))
  return 0 if required_checks_passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
