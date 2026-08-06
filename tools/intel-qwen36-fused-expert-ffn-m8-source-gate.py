#!/usr/bin/env python3
"""Compile and measure the sole admitted M8 expert-major FFN source."""

from __future__ import annotations

import argparse
import json
import math
import re
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-fused-expert-ffn-m8-source-gate-v0"
MODEL = Path("/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf")
ENV_SCRIPT = Path(
    "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh")
CMAKE = Path("/home/intel/intel-box-env/conda/bin/cmake")
CXX = Path("/home/intel/intel-box-env/conda/bin/g++")
ONEDNN_SOURCE = Path(
    "/home/intel/intel-qwen36-r0/source/"
    "oneDNN-01b479323f794da1a7a41a6fc084c7e11ccc2c3b")
ONEDNN_BUILD = Path(
    "/home/intel/intel-qwen36-r0/build/onednn-01b479-ocl-grouped")
ONEDNN_COMMIT = "01b479323f794da1a7a41a6fc084c7e11ccc2c3b"
ONEDNN_PATHS = [
    "src/gpu/intel/matmul/grouped_micro_gemm.cl",
    "src/gpu/intel/matmul/grouped_micro_gemm.cpp",
    "src/gpu/intel/ocl/engine.cpp",
    "src/gpu/intel/ocl/kernel.cpp",
]
PATCH = ROOT / "engine/gpu/opencl/onednn-grouped-s8-u4-fused.patch"
CODEGEN_SOURCE = ROOT / "engine/tools/onednn_grouped_q4k_moe_component.cpp"
SUPPORT_KERNEL = (
    ROOT / "engine/gpu/opencl/grouped_s8_u4_f16_contribution_moe.cl")
PREPACKED = (
    ROOT / "output/grouped-s8-u4-prefill-gate-20260711Tseq673cleanZ/raw/"
    "prepacked")
CAPTURE = (
    ROOT / "output/onednn-q4k-routed-moe-component-gate-"
    "20260711Tseq646cleanZ/raw/capture/payloads")
BUILD_DIR = ROOT / "build/engine"
RUNTIME_TARGET = "iq36-grouped-s8-u4-prefill-runtime"
MATRIX_MACS = 28_689_039_360
MATRIX_CAP_US = 5312.785
MATRIX_RATE_TMAC_S = 5.4
NOISE_FRACTION = 0.005


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--model", type=Path, default=MODEL)
  parser.add_argument("--env-script", type=Path, default=ENV_SCRIPT)
  parser.add_argument("--cmake", type=Path, default=CMAKE)
  parser.add_argument("--cxx", type=Path, default=CXX)
  parser.add_argument("--onednn-source", type=Path, default=ONEDNN_SOURCE)
  parser.add_argument("--onednn-build", type=Path, default=ONEDNN_BUILD)
  parser.add_argument("--build-dir", type=Path, default=BUILD_DIR)
  parser.add_argument("--warmup", type=int, default=8)
  parser.add_argument("--repeat", type=int, default=21)
  parser.add_argument("--timeout-s", type=int, default=600)
  parser.add_argument("--out-dir", type=Path)
  args = parser.parse_args()
  if args.warmup < 1 or args.repeat < 5 or args.timeout_s <= 0:
    parser.error("warmup/repeat/timeout arguments are invalid")
  if args.out_dir is None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.out_dir = ROOT / f"output/fused-expert-ffn-m8-source-gate-{stamp}"
  return args


def run(command: list[str], timeout_s: int, cwd: Path = ROOT,
        env: dict[str, str] | None = None) -> dict[str, Any]:
  try:
    completed = subprocess.run(
        command, cwd=cwd, env=env, text=True, capture_output=True,
        timeout=timeout_s, check=False, encoding="utf-8", errors="replace")
    return {"command": command, "returncode": completed.returncode,
            "stdout": completed.stdout, "stderr": completed.stderr,
            "timed_out": False}
  except subprocess.TimeoutExpired as error:
    return {"command": command, "returncode": 124,
            "stdout": error.stdout or "", "stderr": error.stderr or "",
            "timed_out": True}


def run_intel(command: list[str], args: argparse.Namespace,
              extra_env: dict[str, str] | None = None,
              cwd: Path = ROOT) -> dict[str, Any]:
  exports = ["export INTEL_FORCE_PROBE=b080", "export DNNL_VERBOSE=0"]
  for key, value in sorted((extra_env or {}).items()):
    exports.append(f"export {key}={shlex.quote(value)}")
  shell = (
      f"source {shlex.quote(str(args.env_script))} >/dev/null 2>&1 && "
      + " && ".join(exports) + " && " + shlex.join(command))
  return run(["bash", "-lc", shell], args.timeout_s, cwd=cwd)


def write_run(raw: Path, label: str, row: dict[str, Any]) -> None:
  (raw / f"{label}.command.json").write_text(
      json.dumps(row.get("command", []), indent=2) + "\n", encoding="utf-8")
  (raw / f"{label}.stdout").write_text(
      str(row.get("stdout", "")), encoding="utf-8")
  (raw / f"{label}.stderr").write_text(
      str(row.get("stderr", "")), encoding="utf-8")


def parse_json_line(row: dict[str, Any]) -> dict[str, Any]:
  for line in reversed(str(row.get("stdout", "")).splitlines()):
    try:
      value = json.loads(line)
    except json.JSONDecodeError:
      continue
    if isinstance(value, dict):
      return value
  return {}


def git_output(*parts: str, cwd: Path = ROOT) -> str:
  completed = subprocess.run(
      ["git", *parts], cwd=cwd, text=True, capture_output=True, check=True)
  return completed.stdout.strip()


def check(name: str, passed: bool, **evidence: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **evidence}


def relative(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path.resolve())


def number(text: str, key: str) -> int | None:
  match = re.search(rf"\b{re.escape(key)}:\s*(\d+)", text)
  return int(match.group(1)) if match else None


def runtime_command(binary: Path, gateup: Path, down: Path,
                    args: argparse.Namespace) -> list[str]:
  return [
      str(binary), "--prep-dir", str(PREPACKED),
      "--gateup-binary", str(gateup), "--down-binary", str(down),
      "--kernels", str(SUPPORT_KERNEL), "--input",
      str(CAPTURE / "attn_post_norm-27__tok1023__ord0.bin"),
      "--topk", str(CAPTURE / "ffn_moe_topk-27__tok1023__ord1.bin"),
      "--topk-stride", "1024", "--oracle",
      str(CAPTURE / "ffn_moe_swiglu-27__tok1023__ord3.bin"),
      "--router-weights",
      str(CAPTURE / "ffn_moe_weights_norm-27__tok1023__ord2.bin"),
      "--down-oracle",
      str(CAPTURE / "ffn_moe_down-27__tok1023__ord4.bin"),
      "--moe-oracle", str(CAPTURE / "ffn_moe_out-27__tok1023__ord5.bin"),
      "--warmup", str(args.warmup), "--repeat", str(args.repeat),
      "--kernel-cap-us", str(MATRIX_CAP_US), "--m8-source-preflight",
  ]


def main() -> int:
  args = parse_args()
  out = args.out_dir.resolve()
  raw = out / "raw"
  raw.mkdir(parents=True, exist_ok=False)
  payloads = [
      CAPTURE / "attn_post_norm-27__tok1023__ord0.bin",
      CAPTURE / "ffn_moe_topk-27__tok1023__ord1.bin",
      CAPTURE / "ffn_moe_swiglu-27__tok1023__ord3.bin",
      CAPTURE / "ffn_moe_weights_norm-27__tok1023__ord2.bin",
      CAPTURE / "ffn_moe_down-27__tok1023__ord4.bin",
      CAPTURE / "ffn_moe_out-27__tok1023__ord5.bin",
  ]
  required = [
      args.model, args.env_script, args.cmake, args.cxx,
      args.onednn_source, args.onednn_build, PATCH, CODEGEN_SOURCE,
      SUPPORT_KERNEL, PREPACKED, ROOT / "engine/CMakeLists.txt",
      args.onednn_build / "src/libdnnl.so",
      args.onednn_build / "include/oneapi/dnnl/dnnl_config.h",
      args.onednn_source / "include/oneapi/dnnl/dnnl.hpp", *payloads,
  ]
  missing = [str(path) for path in required if not path.exists()]
  if missing:
    raise SystemExit("missing inputs: " + ", ".join(missing))

  created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
  commit = git_output("rev-parse", "HEAD")
  dirty = git_output("status", "--porcelain")
  onednn_commit = git_output("rev-parse", "HEAD", cwd=args.onednn_source)
  onednn_diff = subprocess.run(
      ["git", "diff", "--unified=0", "--", *ONEDNN_PATHS],
      cwd=args.onednn_source, capture_output=True, check=False).stdout
  onednn_status_run = subprocess.run(
      ["git", "status", "--short"], cwd=args.onednn_source,
      text=True, capture_output=True, check=True)
  onednn_status_lines = onednn_status_run.stdout.splitlines()
  expected_status = sorted(f" M {path}" for path in ONEDNN_PATHS)
  patch_exact = (
      onednn_diff == PATCH.read_bytes() and
      sorted(onednn_status_lines) == expected_status)

  onednn_build = run_intel([
      str(args.cmake), "--build", str(args.onednn_build), "--target", "dnnl",
      "-j16"], args)
  write_run(raw, "onednn-build", onednn_build)
  configure = run([
      str(args.cmake), "-S", str(ROOT / "engine"), "-B",
      str(args.build_dir)], args.timeout_s)
  write_run(raw, "configure", configure)
  engine_build = run([
      str(args.cmake), "--build", str(args.build_dir), "--target",
      RUNTIME_TARGET, "-j16"], args.timeout_s)
  write_run(raw, "engine-build", engine_build)
  runtime = args.build_dir / RUNTIME_TARGET

  codegen = raw / "onednn-grouped-m8-source-codegen"
  codegen_build_command = [
      str(args.cxx), "-std=gnu++17", "-O3", "-DNDEBUG",
      "-DCL_TARGET_OPENCL_VERSION=300",
      f"-I{args.onednn_build / 'include'}",
      f"-I{args.onednn_source / 'include'}", str(CODEGEN_SOURCE),
      f"-L{args.onednn_build / 'src'}",
      f"-Wl,-rpath,{args.onednn_build / 'src'}", "-ldnnl", "-lOpenCL",
      "-o", str(codegen),
  ]
  codegen_build = (
      run_intel(codegen_build_command, args)
      if onednn_build["returncode"] == 0 else
      {"command": codegen_build_command, "returncode": 125, "stdout": "",
       "stderr": "oneDNN build failed", "timed_out": False})
  write_run(raw, "codegen-build", codegen_build)

  codegen_command = [
      str(codegen), "--model", str(args.model),
      "--weight-offset", "14585674336", "--weight-bytes", "301989888",
      "--input", str(payloads[0]), "--topk", str(payloads[1]),
      "--topk-stride", "1024", "--oracle", str(payloads[2]),
      "--down-weight-offset", "14431394400", "--down-weight-bytes",
      "150994944", "--router-weights", str(payloads[3]),
      "--down-oracle", str(payloads[4]), "--moe-oracle", str(payloads[5]),
      "--warmup", "1", "--repeat", "1", "--kernel-cap-us",
      str(MATRIX_CAP_US),
  ]
  binaries: dict[str, Path] = {}
  disassembly: dict[str, dict[str, Any]] = {}
  generation: dict[str, dict[str, Any]] = {}
  for kind in ("gateup", "down"):
    prefix = raw / f"m8-{kind}"
    generation[kind] = (
        run_intel(codegen_command, args, extra_env={
            "DNNL_PRIMITIVE_CACHE_CAPACITY": "0",
            "IQ36_DUMP_FUSED_PROGRAM_PREFIX": str(prefix),
            "IQ36_EXIT_AFTER_FUSED_DUMP": "1",
            "IQ36_GENERATE_S8_GROUPED": "1",
            "IQ36_GROUPED_FUSED_KIND": kind,
            "IQ36_GROUPED_M8_SOURCE_PREFLIGHT": "1",
            "IQ36_GROUPED_PERSISTENT_DISPATCH": "1",
        }) if codegen_build["returncode"] == 0 else
        {"command": codegen_command, "returncode": 125, "stdout": "",
         "stderr": "codegen build failed", "timed_out": False})
    write_run(raw, f"{kind}-generate", generation[kind])
    binary = raw / f"m8-{kind}.0.bin"
    binaries[kind] = binary
    dump = raw / f"m8-{kind}-disasm"
    disasm_command = [
        "ocloc", "disasm", "-file", str(binary), "-dump", str(dump),
        "-device", "0xb080"]
    disasm_run = (
        run_intel(disasm_command, args) if binary.exists() else
        {"command": disasm_command, "returncode": 125, "stdout": "",
         "stderr": "binary missing", "timed_out": False})
    write_run(raw, f"{kind}-disasm", disasm_run)
    ze_path = dump / ".ze_info"
    asm_paths = sorted(set(dump.glob("*.asm")) | set(dump.glob(".*.asm")))
    disassembly[kind] = {
        "run": disasm_run,
        "ze_info": (ze_path.read_text(encoding="utf-8", errors="replace")
                    if ze_path.exists() else ""),
        "assembly": "\n".join(path.read_text(
            encoding="utf-8", errors="replace") for path in asm_paths),
    }

  probe_rows: list[dict[str, Any]] = []
  probe_command = runtime_command(
      runtime, binaries["gateup"], binaries["down"], args)
  for label in ("repeat", "confirm"):
    prerequisites = (
        engine_build["returncode"] == 0 and runtime.exists() and
        all(path.exists() for path in binaries.values()))
    probe_run = (
        run_intel(probe_command, args) if prerequisites else
        {"command": probe_command, "returncode": 125, "stdout": "",
         "stderr": "runtime prerequisite failed", "timed_out": False})
    write_run(raw, label, probe_run)
    probe_rows.append({
        "label": label, "returncode": probe_run["returncode"],
        "probe": parse_json_line(probe_run),
    })

  resources: dict[str, dict[str, Any]] = {}
  expected_workgroups = {"gateup": "[ 16, 8, 1 ]", "down": "[ 32, 4, 1 ]"}
  for kind, value in disassembly.items():
    ze_info = value["ze_info"]
    assembly = value["assembly"]
    resources[kind] = {
        "grf_count": number(ze_info, "grf_count"),
        "eu_thread_count": number(ze_info, "eu_thread_count"),
        "static_dpas": len(re.findall(r"\bdpas(?:\.|\b)", assembly)),
        "workgroup": expected_workgroups[kind],
        "passes": (
            value["run"]["returncode"] == 0 and
            number(ze_info, "grf_count") is not None and
            int(number(ze_info, "grf_count") or 999) <= 128 and
            int(number(ze_info, "eu_thread_count") or 0) >= 8 and
            "has_dpas:        true" in ze_info and
            f"required_work_group_size: {expected_workgroups[kind]}" in ze_info and
            "scratch" not in ze_info.lower() and "spill" not in ze_info.lower() and
            len(re.findall(r"\bdpas(?:\.|\b)", assembly)) > 0),
    }

  rates = [float(row["probe"].get("matrix_rate_tmac_s", 0.0))
           for row in probe_rows]
  matrix_minima = [float(row["probe"].get("matrix_minimum_us", math.inf))
                    for row in probe_rows]
  pair_spread = (
      abs(matrix_minima[0] - matrix_minima[1]) / min(matrix_minima)
      if len(matrix_minima) == 2 and
      all(math.isfinite(value) and value > 0 for value in matrix_minima)
      else math.inf)
  row_evaluated = [
      row["returncode"] in (0, 2) and bool(row["probe"])
      for row in probe_rows]
  row_correct = [
      row["probe"].get("correctness_pass") is True and
      row["probe"].get("maps_native_only") is True and
      row["probe"].get("m8_padded_rows") == 9120 and
      row["probe"].get("preflight_logical_tasks") == 36480 and
      row["probe"].get("matrix_padded_macs") == MATRIX_MACS
      for row in probe_rows]
  row_rate_pass = [
      rate >= MATRIX_RATE_TMAC_S and minimum <= MATRIX_CAP_US
      for rate, minimum in zip(rates, matrix_minima)]

  checks = [
      check("repository_clean_at_gate", dirty == "",
            dirty_paths=dirty.splitlines()),
      check("pinned_onednn_source_commit", onednn_commit == ONEDNN_COMMIT,
            observed=onednn_commit, expected=ONEDNN_COMMIT),
      check("onednn_source_diff_exactly_matches_repo_patch", patch_exact,
            dirty_paths=onednn_status_lines),
      check("onednn_codegen_library_builds", onednn_build["returncode"] == 0),
      check("engine_fixed_m8_runtime_builds",
            configure["returncode"] == 0 and engine_build["returncode"] == 0),
      check("fixed_m8_codegen_driver_builds", codegen_build["returncode"] == 0),
      check("fixed_m8_gateup_and_down_programs_generated",
            all(generation[kind]["returncode"] == 0 and
                binaries[kind].exists() for kind in binaries)),
      check("fixed_m8_compiler_resources_pass",
            all(value["passes"] for value in resources.values()),
            resources=resources),
      check("repeat_and_confirm_complete", all(row_evaluated),
            returncodes=[row["returncode"] for row in probe_rows]),
      check("repeat_and_confirm_real_oracle_correct", all(row_correct),
            row_correctness=row_correct),
      check("repeat_and_confirm_clear_5p4_tmac_s_and_5312p785_us",
            all(row_rate_pass), rates_tmac_s=rates,
            matrix_minima_us=matrix_minima, rate_floor_tmac_s=MATRIX_RATE_TMAC_S,
            cap_us=MATRIX_CAP_US),
      check("repeat_confirm_spread_inside_noise_band",
            pair_spread <= NOISE_FRACTION, spread_fraction=pair_spread,
            noise_fraction=NOISE_FRACTION),
  ]
  required_checks_passed = all(item["pass"] for item in checks)
  evaluation_completed = (
      patch_exact and all(row_evaluated) and all(row_correct) and
      all(value["passes"] for value in resources.values()))
  disposition = (
      "accept_fixed_m8_expert_major_source_implement_full_ffn"
      if required_checks_passed else
      ("reject_fixed_m8_expert_major_source_below_matrix_rate"
       if evaluation_completed else "incomplete_fixed_m8_source_gate"))
  selected_next_route = (
      "native_prefill_fused_expert_major_m8_full_ffn_gate"
      if required_checks_passed else "native_prefill_product_route_reflection_gate")
  reason = (
      "The fixed source is correct, resource-safe, stable, and clears the "
      "registered matrix kill number twice; implement the full FFN boundary."
      if required_checks_passed else
      "The fixed M8 source is numerically correct and compiler-resource safe, "
      "but its real gate/up row-reuse cost leaves the matrix rate below the "
      "5.4 TMAC/s kill number. Close this source without a tile/workgroup sweep "
      "and return to product-route reflection.")
  result = {
      "schema_version": SCHEMA, "workstream": WORKSTREAM,
      "created_at": created_at, "commit": commit,
      "inputs": {"model": str(args.model), "prepacked": relative(PREPACKED),
                 "capture": relative(CAPTURE), "patch": relative(PATCH)},
      "fixed_design": {
          "m_tile": 8, "gateup_n_tile": 32, "down_n_tile": 64,
          "persistent_workgroups": 96, "padded_rows": 9120,
          "padded_matrix_macs": MATRIX_MACS,
          "matrix_cap_us": MATRIX_CAP_US,
          "matrix_rate_floor_tmac_s": MATRIX_RATE_TMAC_S,
      },
      "compiler_resources": resources, "rows": probe_rows,
      "matrix_rates_tmac_s": rates,
      "matrix_minima_us": matrix_minima,
      "paired_spread_fraction": pair_spread, "checks": checks,
      "evaluation_completed": evaluation_completed,
      "required_checks_passed": required_checks_passed,
      "disposition": disposition, "selected_next_route": selected_next_route,
      "next_route_reason": reason,
  }
  (out / "result.json").write_text(
      json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  (out / "manifest.json").write_text(json.dumps({
      "schema_version": SCHEMA, "created_at": created_at, "commit": commit,
      "git_dirty": bool(dirty), "required_checks_passed": required_checks_passed,
  }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  failed = [item["name"] for item in checks if not item["pass"]]
  (out / "summary.md").write_text("\n".join([
      "# Fixed M8 expert-major FFN source gate", "",
      f"- required_checks_passed: `{str(required_checks_passed).lower()}`",
      f"- disposition: `{disposition}`",
      f"- compiler resources: `{resources}`",
      f"- repeat/confirm matrix minimum: `{matrix_minima} us`",
      f"- repeat/confirm padded rate: `{rates} TMAC/s`",
      f"- paired spread: `{pair_spread:.6%}`",
      f"- failed checks: `{failed}`", "", reason, ""]), encoding="utf-8")
  print(json.dumps({
      "required_checks_passed": required_checks_passed,
      "disposition": disposition, "matrix_rates_tmac_s": rates,
      "matrix_minima_us": matrix_minima,
      "paired_spread_fraction": pair_spread,
      "selected_next_route": selected_next_route,
      "out_dir": relative(out)}, sort_keys=True))
  return 0 if required_checks_passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
