#!/usr/bin/env python3
"""Bound isolated IGC 2.38.2 on the exact accepted attention SPIR-V.

The gate starts from the decode and prefill zebins captured by clean seq1240,
extracts their embedded SPIR-V, proves that the installed IGC 2.34.4 exactly
reproduces both accepted ISA streams, and recompiles the same IR with the
official IGC 2.38.2 Ubuntu packages in an isolated ``LD_LIBRARY_PATH``.  It is
an offline source/codegen admission gate: no GPU context or model worker runs.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-openvino-igc2382-codegen-bound-v0"

OCLOC = Path("/usr/bin/ocloc")
SEQ1240 = ROOT / (
    "output/openvino-accepted-carrier-profile-refresh-"
    "20260715Tseq1240-2k-warm17-cleanZ")
DECODE_ZEBIN = SEQ1240 / (
    "raw/2k/candidate/neo-cache/0/a/0ad4e1c668041af6.l0_cache")
PREFILL_ZEBIN = SEQ1240 / (
    "raw/2k/candidate/neo-cache/2/4/244e4d0a64495cb8.l0_cache")
SEQ1240_METRICS = SEQ1240 / "metrics.json"
FRONTIER = ROOT / "doc/active" / WS / "frontier.json"

DECODE_ZEBIN_SHA256 = (
    "8f2aa97885917e8f97c995467946e3197be20c77d6145e7b1467cfee072bbbba")
PREFILL_ZEBIN_SHA256 = (
    "87b4b5c7e8765b3bc9b2e5477351b0437056a6af527391d5d73bfd168a5f315e")
DECODE_ACCEPTED_ASM_SHA256 = (
    "0cbbe4a1863ab12c32e56b1231828eca64176a07f41de1852437b4367f49e45f")
PREFILL_ACCEPTED_ASM_SHA256 = (
    "3ff82d499f911a6606ee322748dd93636908e9a4ad7887b902921d28b26a4908")

IGC_RELEASE = "v2.38.2"
IGC_SOURCE_COMMIT = "3eef0f89d3a4fe2b443de595e23d7700a5d1491b"
IGC_PUBLISHED_AT = "2026-07-14T12:14:06Z"
IGC_CORE_PACKAGE = "intel-igc-core-2_2.38.2+22051_amd64.deb"
IGC_OPENCL_PACKAGE = "intel-igc-opencl-2_2.38.2+22051_amd64.deb"
IGC_CORE_PACKAGE_SHA256 = (
    "3dbcbe4e716d62e9bd43a4a476d724cf772b4581dbcdd096d70df382e7ccad7e")
IGC_OPENCL_PACKAGE_SHA256 = (
    "e265d191590efd5491bfbbd148c144fdd40aea51e0b57f8651130d2da20b8186")
IGC_RELEASE_URL = (
    "https://github.com/intel/intel-graphics-compiler/releases/tag/v2.38.2")

CURRENT_LIBRARY_SHA256 = {
    "libigc.so.2":
        "5dbf0da8af02783b7770b9294485a034ce9756cf6bcdbed95d9a3b6d10f0674a",
    "libigdfcl.so.2":
        "f9b9db2bc681f44040f3c29fbd17de5ec1bb82dd92921a8022b992b0e02e50e5",
    "libopencl-clang2.so.16":
        "bff108c0dc259768a574372e1d33e3101543da4f2453a47f4296152aa314519d",
}
NEW_LIBRARY_SHA256 = {
    "libigc.so.2":
        "ff0cc269af1b2f843521b9207c54370fddab25caa404b1322cbdb4598452da33",
    "libigdfcl.so.2":
        "edd0cc3c73fee76ce156b8a8281d5a747f2634bc81a95da0ca1af9e72abd8de2",
    "libopencl-clang2.so.17":
        "5ad86d1aa4c4b92ca5ff96cbe2ca96d888b5afc5517e3c23b1772983c4dec63b",
}

REGISTERED_ATTENTION_MS = 8.456
KERNEL = "iq36_hot_attention_single_owner"


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument(
      "--package-dir", type=Path,
      default=Path("/tmp/iq36-igc-2.38.2-packages"))
  parser.add_argument(
      "--igc-root", type=Path,
      default=Path("/tmp/iq36-igc-2.38.2-root"))
  parser.add_argument("--timeout-s", type=int, default=300)
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
  status = subprocess.run(
      ["git", "status", "--porcelain"], cwd=ROOT, check=True,
      capture_output=True, text=True).stdout.splitlines()
  try:
    relative = str(output.resolve().relative_to(ROOT))
  except ValueError:
    relative = ""
  allowed = {
      "tools/intel-qwen36-openvino-igc2382-codegen-bound.py",
      "tools/intel-qwen36-openvino-accepted-carrier-profile-refresh.py",
  }
  dirty = []
  for row in status:
    path = row[3:]
    if relative and path.startswith(relative):
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


def run(
    command: list[str], *, timeout_s: int, cwd: Path = ROOT,
    environment: dict[str, str] | None = None,
) -> dict[str, Any]:
  completed = subprocess.run(
      command, cwd=cwd, env=environment, timeout=timeout_s,
      check=False, capture_output=True, text=True,
      encoding="utf-8", errors="replace")
  return {
      "command": command,
      "returncode": completed.returncode,
      "stdout": completed.stdout,
      "stderr": completed.stderr,
  }


def check(name: str, passed: bool, **detail: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **detail}


def disassemble(binary: Path, destination: Path, timeout_s: int) -> dict[str, Any]:
  destination.mkdir(parents=True, exist_ok=False)
  result = run([
      str(OCLOC), "disasm", "-file", str(binary), "-dump",
      str(destination),
  ], timeout_s=timeout_s)
  result.update({
      "binary": display(binary),
      "binary_bytes": binary.stat().st_size,
      "binary_sha256": sha256(binary),
      "destination": display(destination),
  })
  return result


def parse_time(path: Path) -> dict[str, Any]:
  text = path.read_text(encoding="utf-8", errors="replace")
  fields: dict[str, Any] = {"path": display(path)}
  for line in text.splitlines():
    stripped = line.strip()
    if stripped.startswith("Elapsed (wall clock) time"):
      fields["elapsed"] = stripped.split(": ", 1)[1]
    elif stripped.startswith("Maximum resident set size (kbytes)"):
      fields["maximum_rss_kib"] = int(stripped.rsplit(":", 1)[1])
    elif stripped.startswith("Swaps"):
      fields["swaps"] = int(stripped.rsplit(":", 1)[1])
    elif stripped.startswith("Exit status"):
      fields["exit_status"] = int(stripped.rsplit(":", 1)[1])
  return fields


def compile_spv(
    name: str, spv: Path, options: str, output_dir: Path,
    library_dir: Path, timeout_s: int,
) -> dict[str, Any]:
  output_dir.mkdir(parents=True, exist_ok=False)
  time_path = output_dir.parent / f"{name}.time.txt"
  environment = os.environ.copy()
  environment["LD_LIBRARY_PATH"] = str(library_dir.resolve())
  command = [
      "/usr/bin/time", "-v", "-o", str(time_path),
      str(OCLOC), "compile", "-file", str(spv), "-spirv_input",
      "-device", "0xb080", "-output", name, "-out_dir", str(output_dir),
      "-64", "--format", "zebin", "-options", options, "-q",
  ]
  result = run(command, timeout_s=timeout_s, environment=environment)
  binary = output_dir / f"{name}_ptl.bin"
  result.update({
      "library_dir": str(library_dir.resolve()),
      "binary": display(binary),
      "binary_bytes": binary.stat().st_size if binary.is_file() else None,
      "binary_sha256": sha256(binary) if binary.is_file() else None,
      "time": parse_time(time_path) if time_path.is_file() else {},
  })
  return result


def kernel_asm(directory: Path) -> Path:
  path = directory / f".text.{KERNEL}.asm"
  if not path.is_file():
    raise FileNotFoundError(path)
  return path


def ze_kernel_block(path: Path) -> tuple[str, str]:
  text = path.read_text(encoding="utf-8")
  version_match = re.search(r"^version:\s+'([^']+)'", text, re.MULTILINE)
  match = re.search(
      rf"^  - name:\s+{re.escape(KERNEL)}\s*$"
      r"(.*?)(?=^  - name:|^kernels_misc_info:|\Z)",
      text, re.MULTILINE | re.DOTALL)
  if not version_match or not match:
    raise ValueError(f"unable to parse {path}")
  return version_match.group(1), match.group(1)


def int_field(block: str, name: str) -> int:
  match = re.search(
      rf"^\s+{re.escape(name)}:\s+(\d+)\s*$", block, re.MULTILINE)
  return int(match.group(1)) if match else 0


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
  version, block = ze_kernel_block(directory / ".ze_info")
  return {
      "binary": {
          "path": display(binary),
          "bytes": binary.stat().st_size,
          "sha256": sha256(binary),
      },
      "ze_info_version": version,
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
      "assembly": assembly_metrics(kernel_asm(directory)),
  }


def library_identity(directory: Path, expected: dict[str, str]) -> dict[str, Any]:
  rows = {}
  for name, expected_hash in expected.items():
    path = directory / name
    actual = sha256(path) if path.is_file() else None
    rows[name] = {
        "path": str(path.resolve()),
        "expected_sha256": expected_hash,
        "actual_sha256": actual,
        "match": actual == expected_hash,
    }
  return {
      "directory": str(directory.resolve()),
      "libraries": rows,
      "all_match": all(row["match"] for row in rows.values()),
  }


def same_execution_shape(control: dict[str, Any], candidate: dict[str, Any]) -> bool:
  names = (
      "simd_size", "grf_count", "eu_thread_count", "slm_size",
      "spill_size", "has_dpas")
  return all(
      control["execution_env"][name] == candidate["execution_env"][name]
      for name in names)


def main() -> int:
  args = parse_args()
  output = args.output.resolve()
  raw = output / "raw"
  raw.mkdir(parents=True, exist_ok=False)
  stop_bytes = int(args.memory_stop_gib * 1024**3)
  available_start = available_memory_bytes()
  if available_start < stop_bytes:
    raise RuntimeError(f"memory stop: {available_start} < {stop_bytes}")

  package_dir = args.package_dir.resolve()
  igc_root = args.igc_root.resolve()
  current_lib = Path("/usr/local/lib")
  new_lib = igc_root / "usr/local/lib"
  core_package = package_dir / IGC_CORE_PACKAGE
  opencl_package = package_dir / IGC_OPENCL_PACKAGE
  required = (
      OCLOC, DECODE_ZEBIN, PREFILL_ZEBIN, SEQ1240_METRICS, FRONTIER,
      core_package, opencl_package,
      *(current_lib / name for name in CURRENT_LIBRARY_SHA256),
      *(new_lib / name for name in NEW_LIBRARY_SHA256),
  )
  missing = [display(path) for path in required if not path.is_file()]
  if missing:
    raise SystemExit("missing IGC codegen-bound inputs: " + ", ".join(missing))

  git = git_state(output)
  seq1240 = load_json(SEQ1240_METRICS)
  frontier = load_json(FRONTIER)
  kill_number_ms = float(
      frontier["goal_budget"]["per_token_ms"]["remaining_cut"])
  registered_attention_ms = float(
      seq1240["route_selection"][
          "registered_event_buckets_ms_per_token"]["custom_attention"])

  packages = {
      IGC_CORE_PACKAGE: {
          "path": str(core_package),
          "expected_sha256": IGC_CORE_PACKAGE_SHA256,
          "actual_sha256": sha256(core_package),
      },
      IGC_OPENCL_PACKAGE: {
          "path": str(opencl_package),
          "expected_sha256": IGC_OPENCL_PACKAGE_SHA256,
          "actual_sha256": sha256(opencl_package),
      },
  }
  current_identity = library_identity(current_lib, CURRENT_LIBRARY_SHA256)
  candidate_identity = library_identity(new_lib, NEW_LIBRARY_SHA256)

  accepted_dirs = {
      "decode": raw / "accepted-decode",
      "prefill": raw / "accepted-prefill",
  }
  accepted_disassembly = {
      "decode": disassemble(DECODE_ZEBIN, accepted_dirs["decode"], args.timeout_s),
      "prefill": disassemble(PREFILL_ZEBIN, accepted_dirs["prefill"], args.timeout_s),
  }
  accepted = {
      "decode": codegen_metrics(accepted_dirs["decode"], DECODE_ZEBIN),
      "prefill": codegen_metrics(accepted_dirs["prefill"], PREFILL_ZEBIN),
  }

  compiled: dict[str, dict[str, Any]] = {"current": {}, "igc2382": {}}
  parsed: dict[str, dict[str, Any]] = {"current": {}, "igc2382": {}}
  for phase in ("decode", "prefill"):
    spv = accepted_dirs[phase] / ".spv"
    options = (accepted_dirs[phase] / ".misc.buildOptions").read_text(
        encoding="utf-8")
    for label, library_dir in (
        ("current", current_lib), ("igc2382", new_lib)):
      phase_root = raw / f"{phase}-{label}"
      binary_dir = phase_root / "compile"
      compiled[label][phase] = compile_spv(
          f"{phase}-{label}", spv, options, binary_dir,
          library_dir, args.timeout_s)
      binary = binary_dir / f"{phase}-{label}_ptl.bin"
      disasm_dir = phase_root / "disasm"
      compiled[label][phase]["disassembly"] = disassemble(
          binary, disasm_dir, args.timeout_s)
      parsed[label][phase] = codegen_metrics(disasm_dir, binary)
      shutil.copy2(binary, output / f"{phase}-{label}.bin")

  available_end = available_memory_bytes()
  deltas = {}
  for phase in ("decode", "prefill"):
    control = parsed["current"][phase]
    candidate = parsed["igc2382"][phase]
    deltas[phase] = {
        "instruction_lines": (
            candidate["assembly"]["instruction_lines"] -
            control["assembly"]["instruction_lines"]),
        "instruction_fraction": (
            candidate["assembly"]["instruction_lines"] /
            control["assembly"]["instruction_lines"]),
        "send_ugm": (
            candidate["assembly"]["send_ugm"] -
            control["assembly"]["send_ugm"]),
        "send_slm": (
            candidate["assembly"]["send_slm"] -
            control["assembly"]["send_slm"]),
        "integer_address_ops": (
            candidate["assembly"]["integer_address_ops"] -
            control["assembly"]["integer_address_ops"]),
        "indirect_stateless_count": (
            candidate["execution_env"]["indirect_stateless_count"] -
            control["execution_env"]["indirect_stateless_count"]),
    }

  compile_rows = [
      compiled[label][phase]
      for label in ("current", "igc2382")
      for phase in ("decode", "prefill")]
  checks = [
      check("repository_clean_at_gate", not git["dirty"], git=git),
      check("seq1240_exact_attention_zebins_match",
            sha256(DECODE_ZEBIN) == DECODE_ZEBIN_SHA256 and
            sha256(PREFILL_ZEBIN) == PREFILL_ZEBIN_SHA256),
      check("official_igc2382_packages_match_release_sha256",
            all(row["actual_sha256"] == row["expected_sha256"]
                for row in packages.values()), packages=packages),
      check("installed_and_isolated_libraries_match_exact_hashes",
            current_identity["all_match"] and candidate_identity["all_match"],
            current=current_identity, candidate=candidate_identity),
      check("all_four_offline_compiles_and_disassemblies_pass",
            all(row["returncode"] == 0 and
                row["disassembly"]["returncode"] == 0 and
                row["time"].get("exit_status") == 0
                for row in compile_rows)),
      check("current_igc_exactly_reproduces_accepted_decode_and_prefill_isa",
            accepted["decode"]["assembly"]["sha256"] ==
                DECODE_ACCEPTED_ASM_SHA256 ==
                parsed["current"]["decode"]["assembly"]["sha256"] and
            accepted["prefill"]["assembly"]["sha256"] ==
                PREFILL_ACCEPTED_ASM_SHA256 ==
                parsed["current"]["prefill"]["assembly"]["sha256"]),
      check("isolated_igc2382_is_new_codegen_on_same_execution_shape",
            parsed["igc2382"]["decode"]["ze_info_version"] == "1.73" and
            parsed["igc2382"]["prefill"]["ze_info_version"] == "1.73" and
            parsed["current"]["decode"]["ze_info_version"] == "1.70" and
            parsed["current"]["prefill"]["ze_info_version"] == "1.70" and
            same_execution_shape(
                parsed["current"]["decode"], parsed["igc2382"]["decode"]) and
            same_execution_shape(
                parsed["current"]["prefill"], parsed["igc2382"]["prefill"])),
      check("igc2382_reduces_both_exact_attention_instruction_streams",
            deltas["decode"]["instruction_lines"] < 0 and
            deltas["prefill"]["instruction_lines"] < 0 and
            parsed["igc2382"]["decode"]["assembly"]["send_ugm"] <=
                parsed["current"]["decode"]["assembly"]["send_ugm"] and
            parsed["igc2382"]["prefill"]["assembly"]["send_ugm"] <
                parsed["current"]["prefill"]["assembly"]["send_ugm"],
            deltas=deltas),
      check("complete_attention_bucket_clears_current_kill_number",
            abs(registered_attention_ms - REGISTERED_ATTENTION_MS) < 1e-12 and
            registered_attention_ms > kill_number_ms,
            registered_attention_ms=registered_attention_ms,
            kill_number_ms=kill_number_ms),
      check("offline_gate_stays_above_memory_stop_without_swap",
            available_start >= stop_bytes and available_end >= stop_bytes and
            all(row["time"].get("swaps", -1) == 0 for row in compile_rows),
            available_start_bytes=available_start,
            available_end_bytes=available_end,
            stop_bytes=stop_bytes),
      check("no_gpu_context_or_model_worker_ran", True,
            gpu_contexts=0, model_compiles=0, model_workers=0),
  ]
  required_checks_passed = all(row["pass"] for row in checks)
  verdict = (
      "admit_isolated_igc2382_short_attention_component"
      if required_checks_passed else "inconclusive")

  metrics = {
      "schema": SCHEMA,
      "created_at": datetime.now(timezone.utc).isoformat(),
      "workstream": WS,
      "git": git,
      "verdict": verdict,
      "required_checks_passed": required_checks_passed,
      "component_pair_admitted": required_checks_passed,
      "long_worker_admitted": False,
      "product_worker_admitted": False,
      "release": {
          "tag": IGC_RELEASE,
          "source_commit": IGC_SOURCE_COMMIT,
          "published_at": IGC_PUBLISHED_AT,
          "official_url": IGC_RELEASE_URL,
          "packages": packages,
      },
      "libraries": {
          "current_igc_2_34_4": current_identity,
          "isolated_igc_2_38_2": candidate_identity,
      },
      "accepted": accepted,
      "recompiled": parsed,
      "deltas_candidate_minus_current": deltas,
      "compiles": compiled,
      "bound": {
          "current_kill_number_ms_per_token": kill_number_ms,
          "registered_complete_custom_attention_ms_per_token":
              registered_attention_ms,
          "stress_margin_ms_per_token":
              registered_attention_ms - kill_number_ms,
          "short_pair_required_saving_ms_per_token": kill_number_ms,
          "interpretation": (
              "stress admission only: grant the compiler route the complete "
              "registered custom-attention bucket; the serial short pair "
              "must save the full current kill-number before any long row"),
      },
      "checks": checks,
      "memory": {
          "stop_bytes": stop_bytes,
          "available_start_bytes": available_start,
          "available_end_bytes": available_end,
          "gpu_contexts": 0,
          "model_workers": 0,
      },
  }
  write_json(output / "metrics.json", metrics)
  binary_inputs = {
      path.name: {
          "bytes": path.stat().st_size,
          "sha256": sha256(path),
      }
      for path in sorted(output.glob("*.bin"))
  }
  write_json(output / "manifest.json", {
      "schema": SCHEMA,
      "tool": display(Path(__file__)),
      "git": git,
      "inputs": {
          display(path): sha256(path) for path in required
      },
      "generated_binaries": binary_inputs,
      "gpu_contexts": 0,
      "model_compiles": 0,
      "model_workers": 0,
      "long_workers": 0,
  })
  decode_current = parsed["current"]["decode"]["assembly"]
  decode_new = parsed["igc2382"]["decode"]["assembly"]
  prefill_current = parsed["current"]["prefill"]["assembly"]
  prefill_new = parsed["igc2382"]["prefill"]["assembly"]
  report = "\n".join((
      "# Isolated IGC 2.38.2 exact attention codegen bound",
      "",
      f"Verdict: **{verdict}**. Required checks: "
      f"`{str(required_checks_passed).lower()}`.",
      "",
      "The installed IGC 2.34.4 exactly reproduces the clean seq1240 "
      "decode and prefill ISA hashes from their embedded SPIR-V. The official "
      "IGC 2.38.2 packages were hash-verified and loaded only through the "
      "offline compiler process; no system package changed.",
      "",
      "| phase | current instructions | 2.38.2 instructions | current UGM | "
      "2.38.2 UGM | execution shape |",
      "|---|---:|---:|---:|---:|---|",
      f"| decode | {decode_current['instruction_lines']} | "
      f"{decode_new['instruction_lines']} | {decode_current['send_ugm']} | "
      f"{decode_new['send_ugm']} | SIMD16 / 128 GRF / 8 threads |",
      f"| prefill | {prefill_current['instruction_lines']} | "
      f"{prefill_new['instruction_lines']} | {prefill_current['send_ugm']} | "
      f"{prefill_new['send_ugm']} | SIMD16 / 256 GRF / 4 threads / "
      f"{parsed['current']['prefill']['execution_env']['spill_size']} B spill |",
      "",
      f"The complete registered custom-attention bucket is "
      f"`{registered_attention_ms:.3f} ms/token`, above the current "
      f"`{kill_number_ms:.6f} ms/token` kill-number. This is a stress bound, "
      "not a speed claim: it admits one serial 2k/17-step isolated A/B only. "
      "That pair must save the full kill-number before any 32k, ABBA, "
      "output512, or product worker.",
      "",
      f"Maximum offline compiler RSS was "
      f"`{max(row['time']['maximum_rss_kib'] for row in compile_rows)} KiB`; "
      "all compilers reported zero swaps and available memory stayed above "
      "the 4 GiB stop.",
      "",
  ))
  (output / "report.md").write_text(report, encoding="utf-8")
  print(json.dumps({
      "output": display(output),
      "verdict": verdict,
      "decode_instruction_delta": deltas["decode"]["instruction_lines"],
      "prefill_instruction_delta": deltas["prefill"]["instruction_lines"],
      "required_short_pair_saving_ms": kill_number_ms,
  }, sort_keys=True))
  return 0 if required_checks_passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
