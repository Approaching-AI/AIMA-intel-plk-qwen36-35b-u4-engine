#!/usr/bin/env python3
"""Gate one fixed real-input F16-F16-F32 XMX prefill tile."""

from __future__ import annotations

import argparse
import json
import math
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "intel-qwen36-f16-dpas-prefill-feasibility-gate-v0"
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
KERNEL = ROOT / "engine/gpu/opencl/f16_dpas_prefill_preflight.cl"
CAPTURE = (
    ROOT / "output/linear-attention-prefill-boundary-"
    "20260712Tseq750cleanZ/raw/capture-output")
ENV_SCRIPT = Path(
    "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh")
CMAKE = Path("/home/intel/intel-box-env/conda/bin/cmake")
TARGET = "iq36-f16-dpas-prefill-preflight"
TILE_COUNT = 65_536
M = 8
N = 16
K = 128
MACS = TILE_COUNT * M * N * K
MINIMUM_TMAC_PER_SECOND = 4.0
NOISE_FRACTION = 0.005
COSINE_MINIMUM = 0.999
RELATIVE_L2_MAXIMUM = 0.002
STATE_CORE_CAP_US = 2_184.0
# Fixed 64x64 DPAS tiles compute full KKT/QK matrices before masking.  The
# complete chunk-64 mapping therefore exposes this many matrix MACs.
MATRIX_ADDRESSABLE_MACS = 2_952_790_016
MATRIX_BUDGET_US = MATRIX_ADDRESSABLE_MACS / (
    MINIMUM_TMAC_PER_SECOND * 1.0e6)
NON_MATRIX_BUDGET_US = STATE_CORE_CAP_US - MATRIX_BUDGET_US


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--kernel", type=Path, default=KERNEL)
  parser.add_argument("--capture", type=Path, default=CAPTURE)
  parser.add_argument("--env-script", type=Path, default=ENV_SCRIPT)
  parser.add_argument("--cmake", type=Path, default=CMAKE)
  parser.add_argument("--jobs", type=int, default=16)
  parser.add_argument("--warmup", type=int, default=20)
  parser.add_argument("--repeat", type=int, default=21)
  parser.add_argument("--timeout-s", type=int, default=1800)
  parser.add_argument("--out-dir", type=Path)
  args = parser.parse_args()
  if min(args.jobs, args.warmup, args.repeat, args.timeout_s) <= 0:
    parser.error("jobs, warmup, repeat, and timeout must be positive")
  if args.out_dir is None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.out_dir = ROOT / f"output/f16-dpas-prefill-feasibility-{stamp}"
  return args


def rel(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path.resolve())


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git_output(*args: str) -> str:
  result = subprocess.run(
      ["git", *args], cwd=ROOT, check=False, capture_output=True, text=True)
  return result.stdout.strip() if result.returncode == 0 else ""


def run(command: list[str], timeout_s: int) -> dict[str, Any]:
  try:
    result = subprocess.run(
        command, cwd=ROOT, check=False, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=timeout_s)
    return {"command": command, "returncode": result.returncode,
            "stdout": result.stdout, "stderr": result.stderr,
            "timed_out": False}
  except subprocess.TimeoutExpired as error:
    return {"command": command, "returncode": 124,
            "stdout": error.stdout if isinstance(error.stdout, str) else "",
            "stderr": error.stderr if isinstance(error.stderr, str) else "",
            "timed_out": True}


def run_env(command: list[str], args: argparse.Namespace) -> dict[str, Any]:
  shell = (
      f"source {shlex.quote(str(args.env_script))} >/dev/null 2>&1 && "
      "export INTEL_FORCE_PROBE=b080 DNNL_VERBOSE=0 && "
      f"{shlex.join(command)}")
  return run(["bash", "-lc", shell], args.timeout_s)


def write_run(raw: Path, name: str, result: dict[str, Any]) -> None:
  write_json(raw / f"{name}.command.json", {
      "command": result["command"], "returncode": result["returncode"],
      "timed_out": result["timed_out"]})
  (raw / f"{name}.stdout").write_text(
      str(result["stdout"]), encoding="utf-8")
  (raw / f"{name}.stderr").write_text(
      str(result["stderr"]), encoding="utf-8")


def parse_json_line(result: dict[str, Any]) -> dict[str, Any]:
  for line in reversed(str(result.get("stdout", "")).splitlines()):
    try:
      value = json.loads(line)
    except json.JSONDecodeError:
      continue
    if isinstance(value, dict):
      return value
  return {}


def check(name: str, passed: bool, **evidence: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **evidence}


def main() -> int:
  args = parse_args()
  out = args.out_dir.resolve()
  raw = out / "raw"
  raw.mkdir(parents=True, exist_ok=False)
  required_paths = [
      args.kernel, args.capture / "capture-summary.json",
      args.capture / "tensor-dumps.jsonl", args.env_script, args.cmake,
      ROOT / "engine/boundaries.json"]
  missing = [str(path) for path in required_paths if not path.exists()]
  if missing:
    raise SystemExit("missing required paths: " + ", ".join(missing))

  dirty = git_output("status", "--porcelain")
  commit = git_output("rev-parse", "HEAD")
  created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
  capture_summary = json.loads(
      (args.capture / "capture-summary.json").read_text(encoding="utf-8"))
  source = args.kernel.read_text(encoding="utf-8")

  build_dir = raw / "build"
  configure = run_env([
      str(args.cmake), "-S", str(ROOT / "engine"), "-B", str(build_dir),
      "-DCMAKE_BUILD_TYPE=Release"], args)
  write_run(raw, "configure", configure)
  build = run_env([
      str(args.cmake), "--build", str(build_dir), f"-j{args.jobs}",
      "--target", TARGET], args) if configure["returncode"] == 0 else {
          "command": [], "returncode": 125, "stdout": "",
          "stderr": "configure failed", "timed_out": False}
  write_run(raw, "build", build)

  binary = build_dir / TARGET
  program_binary = raw / "f16-dpas-preflight.bin"
  command = [
      str(binary), "--kernel", str(args.kernel), "--capture",
      str(args.capture), "--dump-binary", str(program_binary),
      "--warmup", str(args.warmup), "--repeat", str(args.repeat),
      "--minimum-tmac-per-second", str(MINIMUM_TMAC_PER_SECOND)]
  rows = []
  for label in ("repeat", "confirm"):
    result = run_env(command, args) if build["returncode"] == 0 else {
        "command": command, "returncode": 125, "stdout": "",
        "stderr": "build failed", "timed_out": False}
    write_run(raw, label, result)
    rows.append({"label": label, "returncode": result["returncode"],
                 "probe": parse_json_line(result)})

  disassembly_dir = raw / "disassembly"
  disassembly_dir.mkdir(exist_ok=True)
  disassembly = run_env([
      "ocloc", "disasm", "-file", str(program_binary), "-dump",
      str(disassembly_dir)], args) if program_binary.is_file() else {
          "command": [], "returncode": 125, "stdout": "",
          "stderr": "program binary missing", "timed_out": False}
  write_run(raw, "ocloc-disasm", disassembly)
  ze_info_path = disassembly_dir / ".ze_info"
  asm_path = disassembly_dir / ".text.iq36_f16_dpas_8x16x128_preflight.asm"
  ze_info = ze_info_path.read_text(
      encoding="utf-8", errors="replace") if ze_info_path.is_file() else ""
  assembly = asm_path.read_text(
      encoding="utf-8", errors="replace") if asm_path.is_file() else ""

  times_us = [float(row["probe"].get("median_us", math.inf)) for row in rows]
  rates = [
      float(row["probe"].get("tmac_per_second", -math.inf)) for row in rows]
  spread_fraction = (
      abs(times_us[0] - times_us[1]) / min(times_us)
      if len(times_us) == 2 and
      all(math.isfinite(value) and value > 0 for value in times_us)
      else math.inf)
  row_numeric = []
  row_rate = []
  for row in rows:
    probe = row["probe"]
    comparison = probe.get("comparison", {})
    row_numeric.append(
        row["returncode"] in (0, 2) and
        probe.get("tile_count") == TILE_COUNT and probe.get("m") == M and
        probe.get("n") == N and probe.get("k") == K and
        probe.get("macs") == MACS and
        comparison.get("count") == M * N and
        comparison.get("passes") is True and
        float(comparison.get("relative_l2", math.inf)) <=
        RELATIVE_L2_MAXIMUM and
        float(comparison.get("cosine", -math.inf)) >= COSINE_MINIMUM and
        probe.get("timed_host_upload_bytes") == 0 and
        probe.get("timed_host_read_bytes") == 0 and
        probe.get("forbidden_runtime_mapped") is False)
    row_rate.append(
        float(probe.get("tmac_per_second", -math.inf)) >=
        MINIMUM_TMAC_PER_SECOND)

  compiler_passes = (
      disassembly["returncode"] == 0 and
      "name:            iq36_f16_dpas_8x16x128_preflight" in ze_info and
      "has_dpas:        true" in ze_info and
      "grf_count:       64" in ze_info and
      "simd_size:       16" in ze_info and
      "eu_thread_count: 10" in ze_info and
      "spill" not in ze_info.lower() and "scratch" not in ze_info.lower() and
      assembly.count("dpas.8x8") == 8)
  fixed_source = (
      "#define IQ36_DPAS_TILE_COUNT 65536U" in source and
      "#define IQ36_DPAS_M 8U" in source and
      "#define IQ36_DPAS_N 16U" in source and
      "#define IQ36_DPAS_K 128U" in source and
      "intel_sub_group_f16_f16_matrix_mad_k16" in source)
  checks = [
      check("repository_clean_at_gate", dirty == "",
            dirty_paths=dirty.splitlines()),
      check("real_layer0_capture_locked",
            capture_summary.get("token_count") == 1024 and
            capture_summary.get("batch_all") is True and
            capture_summary.get("linear_component_layer") == 0,
            capture=rel(args.capture)),
      check("fixed_65536x8x16x128_shape", fixed_source,
            tile_count=TILE_COUNT, macs=MACS),
      check("matrix_kill_number_preregistered",
            MINIMUM_TMAC_PER_SECOND == 4.0 and
            MATRIX_BUDGET_US < STATE_CORE_CAP_US and
            NON_MATRIX_BUDGET_US > 0.0,
            matrix_addressable_macs=MATRIX_ADDRESSABLE_MACS,
            minimum_tmac_per_second=MINIMUM_TMAC_PER_SECOND,
            matrix_budget_us=MATRIX_BUDGET_US,
            state_core_cap_us=STATE_CORE_CAP_US,
            non_matrix_budget_us=NON_MATRIX_BUDGET_US),
      check("compiler_confirms_f16_dpas_without_spill", compiler_passes,
            ze_info=rel(ze_info_path), assembly=rel(asm_path)),
      check("repeat_and_confirm_pass_real_fp16_numeric_contract",
            all(row_numeric), row_numeric=row_numeric),
      check("repeat_and_confirm_clear_4_tmac_per_second",
            all(row_rate), row_rate=row_rate, rates=rates,
            minimum_tmac_per_second=MINIMUM_TMAC_PER_SECOND),
      check("repeat_confirm_spread_inside_noise_band",
            spread_fraction <= NOISE_FRACTION,
            spread_fraction=spread_fraction,
            noise_fraction=NOISE_FRACTION),
  ]
  required = all(bool(item["pass"]) for item in checks)
  evaluation_completed = (
      dirty == "" and build["returncode"] == 0 and
      disassembly["returncode"] == 0 and
      all(row["returncode"] in (0, 2) and bool(row["probe"])
          for row in rows))
  disposition = (
      "accept_f16_dpas_admit_xmx_chunk64_gdn_design"
      if required else
      ("reject_f16_dpas_close_xmx_chunked_gdn"
       if evaluation_completed else "incomplete_f16_dpas_feasibility_gate"))
  selected_next_route = (
      "native_linear_prefill_xmx_chunk64_gdn_design_gate"
      if required else "native_prefill_route_reflection_gate")
  reason = (
      "The fixed real-input tile clears the XMX kill-number with stable F16 "
      "numeric and compiler evidence; admit one matrix-engine chunk-64 GDN."
      if required else
      "The fixed real-input XMX tile does not clear every pre-registered "
      "numeric, compiler, rate, and noise check. Close XMX chunked GDN rather "
      "than lowering the rate floor or sweeping tile shapes.")
  metrics = {
      "schema_version": SCHEMA, "workstream": WORKSTREAM,
      "created_at": created_at, "commit": commit,
      "inputs": {"kernel": rel(args.kernel), "capture": rel(args.capture)},
      "budget": {
          "state_core_cap_us": STATE_CORE_CAP_US,
          "matrix_addressable_macs": MATRIX_ADDRESSABLE_MACS,
          "minimum_tmac_per_second": MINIMUM_TMAC_PER_SECOND,
          "matrix_budget_us": MATRIX_BUDGET_US,
          "non_matrix_budget_us": NON_MATRIX_BUDGET_US},
      "rows": rows, "checks": checks,
      "times_us": times_us, "rates_tmac_per_second": rates,
      "spread_fraction": spread_fraction,
      "required_checks_passed": required,
      "evaluation_completed": evaluation_completed,
      "disposition": disposition,
      "selected_next_route": selected_next_route,
      "next_route_reason": reason,
  }
  write_json(out / "result.json", metrics)
  write_json(out / "manifest.json", {
      "schema_version": SCHEMA, "created_at": created_at,
      "commit": commit, "git_dirty": bool(dirty),
      "required_checks_passed": required})
  failed = [item["name"] for item in checks if not item["pass"]]
  (out / "summary.md").write_text("\n".join([
      "# F16 DPAS prefill feasibility gate", "",
      f"- required_checks_passed: `{str(required).lower()}`",
      f"- disposition: `{disposition}`",
      f"- repeat/confirm TMAC/s: `{rates}`",
      f"- repeat/confirm time us: `{times_us}`",
      f"- spread: `{spread_fraction:.6%}`",
      f"- failed checks: `{failed}`", "", reason, ""]),
      encoding="utf-8")
  print(json.dumps({
      "required_checks_passed": required,
      "disposition": disposition,
      "rates_tmac_per_second": rates,
      "spread_fraction": spread_fraction,
      "selected_next_route": selected_next_route,
      "out_dir": rel(out)}, sort_keys=True))
  return 0 if required else 2


if __name__ == "__main__":
  raise SystemExit(main())
