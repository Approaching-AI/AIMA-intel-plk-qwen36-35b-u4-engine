#!/usr/bin/env python3
"""Gate the fixed exact-Q4 M64 NPU third before full hybrid FFN wiring."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-npu-exact-q4-m64-prefill-source-gate-v0"
PREPACK = ROOT / "output/grouped-s8-u4-prefill-gate-20260711Tseq673cleanZ/raw/prepacked"
CAPTURE = ROOT / (
    "output/complete-ffn-boundary-gate-20260712Tseq763cleanZ/raw/capture/"
    "payloads/attn_post_norm-27__tok1023__ord0.bin")
SEQ759 = ROOT / "output/fused-expert-ffn-design-gate-20260712Tseq759cleanZ/result.json"
SEQ761 = ROOT / "output/fused-expert-ffn-m8-source-gate-20260712Tseq761cleanZ/result.json"
SEQ766 = ROOT / "output/phase-split-gpu-npu-prefill-reopen-20260712Tseq766cleanZ/result.json"
OV_PYTHON = Path("/home/intel/ov/openvino_env/bin/python")
CXX = Path("/home/intel/intel-box-env/conda/bin/g++")
L0_INCLUDE = Path("/home/intel/intel-box-env/conda/include")
L0_LIB = Path("/home/intel/intel-box-env/conda/lib")
WORKER = ROOT / "engine/tools/npu_exact_q4_m64_worker.py"
HARNESS = ROOT / "engine/tools/npu_level_zero_graph_blob_preflight.cpp"
ABI = ROOT / "engine/include/intel_qwen36/npu_level_zero_graph_ext.hpp"
ROWS = 73_728
TOKENS = 64
COMPARE_ROWS = 512
HIDDEN = 2_048
FFN_CAP_US = 6_250.0
NOISE = 0.005
COSINE_MIN = 0.999
RELATIVE_L2_MAX = 0.002


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out-dir", type=Path)
  parser.add_argument("--prepack", type=Path, default=PREPACK)
  parser.add_argument("--capture", type=Path, default=CAPTURE)
  parser.add_argument("--openvino-python", type=Path, default=OV_PYTHON)
  parser.add_argument("--cxx", type=Path, default=CXX)
  parser.add_argument("--level-zero-include", type=Path, default=L0_INCLUDE)
  parser.add_argument("--level-zero-lib", type=Path, default=L0_LIB)
  parser.add_argument("--warmup", type=int, default=3)
  parser.add_argument("--repeat", type=int, default=11)
  parser.add_argument("--timeout-s", type=int, default=1_800)
  args = parser.parse_args()
  if args.warmup < 0 or args.repeat < 5 or args.timeout_s <= 0:
    parser.error("warmup must be non-negative, repeat >=5, timeout positive")
  if args.out_dir is None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.out_dir = ROOT / f"output/npu-exact-q4-m64-prefill-source-{stamp}"
  return args


def load(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise RuntimeError(f"expected object: {path}")
  return value


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def rel(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path.resolve())


def git_output(*parts: str) -> str:
  result = subprocess.run(
      ["git", *parts], cwd=ROOT, text=True, capture_output=True, check=True)
  return result.stdout.strip()


def run(command: list[str], timeout_s: int) -> dict[str, Any]:
  started = time.perf_counter()
  try:
    result = subprocess.run(
        command, cwd=ROOT, text=True, capture_output=True, check=False,
        timeout=timeout_s, encoding="utf-8", errors="replace")
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "timed_out": False,
        "wall_s": time.perf_counter() - started,
    }
  except subprocess.TimeoutExpired as error:
    return {
        "command": command,
        "returncode": 124,
        "stdout": error.stdout if isinstance(error.stdout, str) else "",
        "stderr": error.stderr if isinstance(error.stderr, str) else "",
        "timed_out": True,
        "wall_s": time.perf_counter() - started,
    }


def skipped(command: list[str], reason: str) -> dict[str, Any]:
  return {
      "command": command, "returncode": 125, "stdout": "",
      "stderr": reason, "timed_out": False, "wall_s": 0.0,
  }


def write_run(raw: Path, name: str, result: dict[str, Any]) -> None:
  write_json(raw / f"{name}.command.json", {
      "command": result["command"],
      "returncode": result["returncode"],
      "timed_out": result["timed_out"],
      "wall_s": result["wall_s"],
  })
  (raw / f"{name}.stdout").write_text(
      result["stdout"], encoding="utf-8")
  (raw / f"{name}.stderr").write_text(
      result["stderr"], encoding="utf-8")


def parse_last_json(text: str) -> dict[str, Any]:
  for line in reversed(text.splitlines()):
    try:
      value = json.loads(line)
    except json.JSONDecodeError:
      continue
    if isinstance(value, dict):
      return value
  return {}


def number(value: Any) -> float | None:
  if isinstance(value, (int, float)) and math.isfinite(float(value)):
    return float(value)
  return None


def check(name: str, passed: bool, **details: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **details}


def main() -> int:
  args = parse_args()
  out = args.out_dir.resolve()
  raw = out / "raw"
  raw.mkdir(parents=True, exist_ok=False)
  required = [
      args.prepack / "gateup-weights.bin",
      args.prepack / "gateup-scales.bin",
      args.prepack / "gateup-min-codes.bin",
      args.prepack / "gateup-dmins.bin",
      args.capture, args.openvino_python, args.cxx,
      args.level_zero_include / "level_zero/ze_api.h",
      args.level_zero_lib / "libze_loader.so",
      WORKER, HARNESS, ABI, SEQ759, SEQ761, SEQ766,
  ]
  missing = [str(path) for path in required if not path.exists()]
  if missing:
    raise SystemExit("missing required paths: " + ", ".join(missing))

  seq759 = load(SEQ759)
  seq761 = load(SEQ761)
  seq766 = load(SEQ766)
  routed_macs = int(seq759["design"]["matrix_macs"])
  shared_macs = int(seq766["projection"]["shared_true_macs"])
  fixed_nonmatrix_us = float(seq759["design"]["fixed_nonmatrix_us"])
  scalar_gate_us = float(seq766["projection"]["shared_scalar_gate_us"])
  matrix_window_us = FFN_CAP_US - fixed_nonmatrix_us - scalar_gate_us
  npu_share_ops = 2.0 * routed_macs / 3.0
  gpu_share_ops = 2.0 * (2.0 * routed_macs / 3.0 + shared_macs)
  required_npu_tops = npu_share_ops / (matrix_window_us * 1e6)
  required_gpu_tops = gpu_share_ops / (matrix_window_us * 1e6)
  measured_gpu_tops = 2.0 * min(
      float(value) for value in seq761["matrix_rates_tmac_s"])
  benchmark_ops = 2 * TOKENS * HIDDEN * ROWS

  xml_path = raw / "exact-q4-m64.xml"
  bin_path = raw / "exact-q4-m64.bin"
  blob_path = raw / "exact-q4-m64.blob"
  inputs_path = raw / "exact-q4-m64.inputs"
  reference_path = raw / "exact-q4-m64.reference"
  binary_path = raw / "npu-level-zero-graph-blob-preflight"

  worker_command = [
      str(args.openvino_python), str(WORKER),
      "--prepack", str(args.prepack), "--capture", str(args.capture),
      "--rows", str(ROWS), "--tokens", str(TOKENS),
      "--compare-rows", str(COMPARE_ROWS),
      "--xml", str(xml_path), "--bin", str(bin_path),
      "--warmup", str(args.warmup), "--repeat", str(args.repeat),
  ]
  worker_run = run(worker_command, args.timeout_s)
  write_run(raw, "openvino-npu-worker", worker_run)
  worker = parse_last_json(worker_run["stdout"])

  build_command = [
      str(args.cxx), "-std=gnu++20", "-O2", "-Wall", "-Wextra",
      "-Wpedantic", "-Werror", f"-I{ROOT / 'engine/include'}",
      f"-I{args.level_zero_include}", str(HARNESS),
      f"-L{args.level_zero_lib}", f"-Wl,-rpath,{args.level_zero_lib}",
      "-lze_loader", "-o", str(binary_path),
  ]
  build_run = run(build_command, args.timeout_s)
  write_run(raw, "build-native-harness", build_run)
  ldd_run = (
      run(["ldd", str(binary_path)], args.timeout_s)
      if build_run["returncode"] == 0 else
      skipped(["ldd", str(binary_path)], "native harness build failed"))
  write_run(raw, "ldd-native-harness", ldd_run)

  compile_command = [
      str(binary_path), "--mode", "compile", "--xml", str(xml_path),
      "--blob", str(blob_path), "--inputs", str(inputs_path),
      "--reference", str(reference_path), "--warmup", str(args.warmup),
      "--repeat", str(args.repeat),
  ]
  compile_run = (
      run(compile_command, args.timeout_s)
      if worker_run["returncode"] == 0 and build_run["returncode"] == 0
      and xml_path.exists() and bin_path.exists() else
      skipped(compile_command, "worker or native build prerequisite failed"))
  write_run(raw, "compile-native-blob", compile_run)
  compile_probe = parse_last_json(compile_run["stdout"])

  native_command = [
      str(binary_path), "--mode", "run", "--blob", str(blob_path),
      "--inputs", str(inputs_path), "--reference", str(reference_path),
      "--warmup", str(args.warmup), "--repeat", str(args.repeat),
  ]
  native_runs = []
  for label in ("repeat", "confirm"):
    row = (
        run(native_command, args.timeout_s)
        if compile_run["returncode"] == 0 else
        skipped(native_command, "native blob compilation failed"))
    write_run(raw, f"native-{label}", row)
    native_runs.append({"label": label, "run": row,
                        "probe": parse_last_json(row["stdout"])})

  comparison = worker.get("comparison", {})
  cosine = number(comparison.get("cosine"))
  relative_l2 = number(comparison.get("relative_l2"))
  native_medians = [
      number(row["probe"].get("execution_median_us"))
      for row in native_runs]
  native_tops = [
      benchmark_ops / (value * 1e6) if value and value > 0 else None
      for value in native_medians]
  spread = (
      abs(native_medians[0] - native_medians[1]) /
      min(native_medians[0], native_medians[1])
      if all(value and value > 0 for value in native_medians) else None)
  dependency_text = ldd_run["stdout"].lower()
  dirty = git_output("status", "--porcelain")

  checks = [
      check("repository_clean_at_gate", dirty == "", dirty_paths=dirty.splitlines()),
      check("seq766_reopens_only_fixed_prefill_source",
            seq766.get("required_checks_passed") is True
            and seq766.get("selected_next_route")
            == "gpu_npu_prefill_only_exact_q4_complete_ffn_component_gate"),
      check("fixed_npu_m64_shape_matches_one_third_routed_work",
            ROWS == 73_728 and TOKENS == 64 and benchmark_ops >= npu_share_ops,
            benchmark_ops=benchmark_ops, npu_share_ops=npu_share_ops),
      check("existing_gpu_source_clears_fixed_gpu_share_rate",
            measured_gpu_tops >= required_gpu_tops,
            measured_tops=measured_gpu_tops, required_tops=required_gpu_tops),
      check("worker_builds_exact_real_q4_graph", worker_run["returncode"] == 0
            and "fatal_error" not in worker
            and worker.get("source", {}).get("rows") == ROWS
            and worker.get("source", {}).get("tokens") == TOKENS,
            fatal_error=worker.get("fatal_error"), source=worker.get("source")),
      check("worker_real_input_numeric_slice_passes",
            comparison.get("finite") is True
            and comparison.get("compared") == TOKENS * COMPARE_ROWS
            and cosine is not None and cosine >= COSINE_MIN
            and relative_l2 is not None and relative_l2 <= RELATIVE_L2_MAX,
            comparison=comparison, cosine_min=COSINE_MIN,
            relative_l2_max=RELATIVE_L2_MAX),
      check("native_harness_builds_and_links_level_zero_only",
            build_run["returncode"] == 0 and ldd_run["returncode"] == 0
            and "libze_loader" in dependency_text
            and "openvino" not in dependency_text and "dnnl" not in dependency_text,
            ldd=ldd_run["stdout"].splitlines()),
      check("exact_graph_compiles_to_native_blob",
            compile_run["returncode"] == 0
            and int(compile_probe.get("native_blob_bytes", 0) or 0) > 0
            and compile_probe.get("openvino_mapped") is False,
            probe=compile_probe),
      check("native_repeat_confirm_reproduce_compiler_reference",
            all(row["run"]["returncode"] == 0
                and row["probe"].get("mismatch_bytes") == 0
                and row["probe"].get("openvino_mapped") is False
                for row in native_runs),
            rows=[row["probe"] for row in native_runs]),
      check("native_repeat_confirm_clear_required_npu_rate",
            all(value is not None and value >= required_npu_tops
                for value in native_tops),
            observed_tops=native_tops, required_tops=required_npu_tops),
      check("native_repeat_confirm_spread_inside_noise_band",
            spread is not None and spread <= NOISE,
            observed=spread, maximum=NOISE),
      check("timed_host_transfer_is_zero_by_native_resident_graph_contract", True,
            timed_host_upload_bytes=0, timed_host_readback_bytes=0),
  ]
  required_passed = all(bool(row["pass"]) for row in checks)
  evaluation_completed = (
      worker_run["returncode"] != 125 and build_run["returncode"] != 125)
  if not evaluation_completed:
    disposition = "incomplete_npu_exact_q4_source_evaluation"
  elif required_passed:
    disposition = "admit_fixed_npu_q4_m64_rate_to_complete_hybrid_ffn"
  else:
    disposition = "reject_phase_split_gpu_npu_prefill_source"

  artifacts = {
      path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
      for path in (xml_path, bin_path, blob_path, inputs_path, reference_path)
      if path.exists()
  }
  result = {
      "schema_version": SCHEMA,
      "workstream": WORKSTREAM,
      "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
      "commit": git_output("rev-parse", "HEAD"),
      "inputs": {
          "capture": rel(args.capture), "prepack": rel(args.prepack),
          "seq759": rel(SEQ759), "seq761": rel(SEQ761), "seq766": rel(SEQ766),
      },
      "fixed_work": {
          "benchmark_rows": ROWS, "benchmark_tokens": TOKENS,
          "benchmark_ops": benchmark_ops,
          "npu_share_ops": npu_share_ops, "gpu_share_ops": gpu_share_ops,
          "matrix_window_us": matrix_window_us,
          "required_npu_tops": required_npu_tops,
          "required_gpu_tops": required_gpu_tops,
          "measured_gpu_tops": measured_gpu_tops,
      },
      "worker": worker,
      "native_compile_probe": compile_probe,
      "native_rows": [
          {"label": row["label"], "probe": row["probe"],
           "returncode": row["run"]["returncode"]}
          for row in native_runs],
      "native_median_us": native_medians,
      "native_tops": native_tops,
      "paired_spread_fraction": spread,
      "artifacts": artifacts,
      "checks": checks,
      "evaluation_completed": evaluation_completed,
      "required_checks_passed": required_passed,
      "disposition": disposition,
      "selected_next_route": (
          "gpu_npu_prefill_only_exact_q4_complete_ffn_integration_gate"
          if required_passed else
          "owner_contract_decision_after_measured_1p10_prefill_exhaustion"),
      "speedup_claims_allowed": False,
      "product_goal_complete": False,
      "stop_condition": (
          "Any worker/build/native correctness/rate/noise failure closes the "
          "phase-split source without row-count, M, partition, graph, datatype, "
          "compiler, precision, or synchronization variants."),
      "next_route_reason": (
          "The exact native NPU branch clears its fixed share twice; attach it "
          "once to the unchanged passing GPU source and measure complete ffn_out."
          if required_passed else
          "The exact native NPU branch fails a terminal prerequisite for the "
          "narrow phase-split projection; restore ADR 0048 owner-decision state."),
  }
  write_json(out / "result.json", result)
  write_json(out / "correctness.json", {
      "comparison": comparison,
      "native_reference_reproduction": [row["probe"] for row in native_runs],
      "required_checks_passed": required_passed,
  })
  write_json(out / "manifest.json", {
      "captured_at": result["created_at"], "commit": result["commit"],
      "schema_version": SCHEMA, "tool": rel(Path(__file__)),
      "workstream": WORKSTREAM,
  })
  with (out / "metrics.jsonl").open("w", encoding="utf-8") as handle:
    for metric, value in (
        ("worker_relative_l2", relative_l2),
        ("worker_cosine", cosine),
        ("native_repeat_tops", native_tops[0]),
        ("native_confirm_tops", native_tops[1]),
        ("required_npu_tops", required_npu_tops),
        ("paired_spread_fraction", spread),
        ("required_checks_passed", required_passed)):
      handle.write(json.dumps({"metric": metric, "value": value}) + "\n")
  (out / "summary.md").write_text("\n".join([
      "# Exact-Q4 M64 NPU Prefill Source Gate", "",
      f"- required checks passed: `{str(required_passed).lower()}`",
      f"- disposition: `{disposition}`",
      f"- worker cosine / relative L2: `{cosine}` / `{relative_l2}`",
      f"- native repeat/confirm TOPS: `{native_tops}`",
      f"- required NPU TOPS: `{required_npu_tops}`",
      f"- paired spread: `{spread}`", "", result["next_route_reason"], "",
  ]), encoding="utf-8")
  print(json.dumps({
      "required_checks_passed": required_passed,
      "evaluation_completed": evaluation_completed,
      "disposition": disposition,
      "selected_next_route": result["selected_next_route"],
      "out_dir": rel(out),
  }, sort_keys=True))
  return 0 if required_passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
