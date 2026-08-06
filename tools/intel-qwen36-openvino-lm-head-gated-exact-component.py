#!/usr/bin/env python3
"""Profile the accepted LM-head gated-exact fallback by kernel stage.

This is a component-only source-bound gate.  It compiles a forced-execution
mirror of the accepted fallback, uploads the locked LM-head I8 weight once,
and runs captured real hidden rows through Q8, full-I8 matvec, block top-8,
global merge, and exact F16-hidden correction.  Repeat and confirm are serial
and memory guarded.  No OpenVINO model graph or stock worker is launched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-openvino-lm-head-gated-exact-component-v1"
MODEL_BIN = Path("/home/intel/Qwen3.6-35B-A3B-ov/openvino_language_model.bin")
CMAKE = Path("/home/intel/intel-box-env/conda/bin/cmake")
BUILD_DIR = REPO / "build/engine"
TARGET = "iq36-openvino-lm-head-gated-exact-component"
BINARY = BUILD_DIR / TARGET
MODULE_SOURCE = (
    REPO / "engine/gpu/opencl/iq36_lm_head_gated_exact_component.cl")
CPP_SOURCE = (
    REPO / "engine/tools/openvino_lm_head_gated_exact_component.cpp")
BOUNDARIES = REPO / "engine/boundaries.json"
PRODUCT_PATCH = REPO / "engine/openvino/iq36-lm-head-i8q1-gated-exact.patch"
BOUND = REPO / (
    "output/openvino-lm-head-gated-exact-fallback-bound-"
    "20260731Tseq2186-clean/result.json")
CAPTURE_DIR = REPO / (
    "output/openvino-lm-head-gated-exact-live-hidden-"
    "20260723Tseq2116-all10-128k-o130-correctness/raw/"
    "sentinel_128k/correctness/candidate")
REFERENCE_DIR = REPO / (
    "output/openvino-lm-head-gated-exact-count25-"
    "20260723Tseq2120-all10-128k-o130-correctness/raw/"
    "sentinel_128k/correctness/candidate")
KLD_LIMIT = 0.005
# Seq2118's three count25 anchors are the rows on which the accepted product
# actually dispatches this fallback.  Forced fallback output is not expected
# to match the accepted fast Q1 path on gate-negative rows.
FALLBACK_REFERENCE_PHASES = (63, 96, 129)
WEIGHT_BYTES = 508_559_360
SCALE_BYTES = 496_640
F16_OUTPUT_BYTES = 496_640
MANDATORY_MATVEC_BYTES = WEIGHT_BYTES + SCALE_BYTES + F16_OUTPUT_BYTES
STAGES = (
    "q8", "matvec", "block_topk", "merge", "correction",
    "fallback_shell", "wall")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--bound", type=Path, default=BOUND)
  parser.add_argument("--capture-dir", type=Path, default=CAPTURE_DIR)
  parser.add_argument("--reference-dir", type=Path, default=REFERENCE_DIR)
  parser.add_argument("--cmake", type=Path, default=CMAKE)
  parser.add_argument("--build-dir", type=Path, default=BUILD_DIR)
  parser.add_argument("--warmup", type=int, default=3)
  parser.add_argument("--samples", type=int, default=20)
  parser.add_argument("--first-phase", type=int, default=0)
  parser.add_argument("--last-phase", type=int, default=129)
  parser.add_argument("--timeout-s", type=int, default=600)
  parser.add_argument("--min-available-gib", type=float, default=8.0)
  parser.add_argument("--abort-below-available-gib", type=float, default=4.0)
  parser.add_argument("--poll-interval-s", type=float, default=0.1)
  args = parser.parse_args()
  if args.warmup < 0 or args.samples < 20:
    parser.error("warmup must be nonnegative and samples at least 20")
  if (args.first_phase < 0 or args.last_phase < args.first_phase or
      args.last_phase > 4096):
    parser.error("phase range must satisfy 0 <= first <= last <= 4096")
  if args.timeout_s <= 0 or args.poll_interval_s <= 0.0:
    parser.error("timeout and poll interval must be positive")
  if (args.min_available_gib < 0.0 or
      args.abort_below_available_gib < 0.0 or
      args.abort_below_available_gib > args.min_available_gib):
    parser.error("memory thresholds are invalid")
  return args


def load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise TypeError(f"expected JSON object: {path}")
  return value


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    while chunk := stream.read(8 * 1024 * 1024):
      digest.update(chunk)
  return digest.hexdigest()


def display_path(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(REPO))
  except ValueError:
    return str(path.resolve())


def run(command: list[str], timeout_s: int = 300) -> dict[str, Any]:
  try:
    result = subprocess.run(
        command, cwd=REPO, capture_output=True, text=True, check=False,
        timeout=timeout_s, encoding="utf-8", errors="replace")
    return {
        "command": command, "returncode": result.returncode,
        "stdout": result.stdout, "stderr": result.stderr,
        "timed_out": False,
    }
  except subprocess.TimeoutExpired as error:
    return {
        "command": command, "returncode": 124,
        "stdout": str(error.stdout or ""),
        "stderr": str(error.stderr or "") + "\ntimeout",
        "timed_out": True,
    }


def check(name: str, passed: bool, **details: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **details}


def git_state(output: Path) -> dict[str, Any]:
  commit = run(["git", "rev-parse", "HEAD"], 30)["stdout"].strip()
  rows = run(["git", "status", "--porcelain"], 30)["stdout"].splitlines()
  try:
    output_relative = str(output.resolve().relative_to(REPO))
  except ValueError:
    output_relative = ""
  rows = [
      row for row in rows
      if not output_relative or output_relative not in row
  ]
  return {"commit": commit, "dirty": bool(rows), "status": rows}


def meminfo() -> dict[str, int]:
  rows: dict[str, int] = {}
  for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
    key, value = line.split(":", 1)
    fields = value.split()
    rows[key] = int(fields[0]) * 1024 if fields else 0
  return rows


def process_memory(pid: int) -> dict[str, int]:
  path = Path(f"/proc/{pid}/status")
  if not path.is_file():
    return {"VmRSS": 0, "VmSwap": 0}
  selected: dict[str, int] = {}
  try:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
  except (FileNotFoundError, ProcessLookupError):
    return {"VmRSS": 0, "VmSwap": 0}
  for line in lines:
    if ":" not in line:
      continue
    key, value = line.split(":", 1)
    fields = value.split()
    if key in ("VmRSS", "VmSwap") and fields:
      selected[key] = int(fields[0]) * 1024
  return {
      "VmRSS": selected.get("VmRSS", 0),
      "VmSwap": selected.get("VmSwap", 0),
  }


def wait_for_memory(required_bytes: int) -> dict[str, Any]:
  started = time.monotonic()
  while True:
    available = int(meminfo().get("MemAvailable", 0))
    if available >= required_bytes:
      return {
          "available_bytes": available,
          "required_bytes": required_bytes,
          "waited_seconds": time.monotonic() - started,
      }
    if time.monotonic() - started > 60.0:
      raise RuntimeError(
          f"available memory {available} remains below {required_bytes}")
    time.sleep(2.0)


def stop_process_group(
    process: subprocess.Popen[Any], first_signal: int,
) -> None:
  try:
    os.killpg(process.pid, first_signal)
  except ProcessLookupError:
    return
  try:
    process.wait(timeout=10)
    return
  except subprocess.TimeoutExpired:
    pass
  try:
    os.killpg(process.pid, signal.SIGKILL)
  except ProcessLookupError:
    pass
  process.wait()


def other_gpu_workers() -> list[dict[str, Any]]:
  rows = []
  for path in Path("/proc").iterdir():
    if not path.name.isdigit() or int(path.name) == os.getpid():
      continue
    try:
      command = (path / "cmdline").read_bytes().replace(
          b"\0", b" ").decode("utf-8", errors="replace")
    except (FileNotFoundError, PermissionError, ProcessLookupError):
      continue
    is_model_worker = (
        "intel-qwen36-openvino-hot-cold-attention-gate.py" in command and
        "--worker-config" in command)
    is_component = (
        ("/" + TARGET + " ") in command or command.startswith(TARGET + " "))
    if is_model_worker or is_component:
      rows.append({"pid": int(path.name), "command": command.strip()})
  return rows


def parse_worker_stdout(path: Path) -> dict[str, Any]:
  if not path.is_file():
    return {}
  for line in reversed(path.read_text(
      encoding="utf-8", errors="replace").splitlines()):
    try:
      value = json.loads(line)
    except json.JSONDecodeError:
      continue
    if isinstance(value, dict):
      return value
  return {}


def launch_component(
    args: argparse.Namespace, directory: Path, module: Path,
) -> dict[str, Any]:
  directory.mkdir(parents=True)
  stdout_path = directory / "component.stdout"
  stderr_path = directory / "component.stderr"
  command = [
      str(BINARY), str(MODEL_BIN), str(module),
      str(args.capture_dir), str(directory),
      str(args.warmup), str(args.samples),
      str(args.first_phase), str(args.last_phase),
  ]
  preflight = wait_for_memory(int(args.min_available_gib * 1024**3))
  abort_bytes = int(args.abort_below_available_gib * 1024**3)
  started = time.monotonic()
  monitor: dict[str, Any] = {
      "process_rss_peak_bytes": 0,
      "process_swap_peak_bytes": 0,
      "system_available_min_bytes": None,
      "system_swap_used_peak_bytes": 0,
      "sample_count": 0,
  }
  timed_out = False
  guard_tripped = False
  with stdout_path.open("w", encoding="utf-8") as stdout_handle, \
       stderr_path.open("w", encoding="utf-8") as stderr_handle:
    process = subprocess.Popen(
        command, cwd=REPO, stdout=stdout_handle, stderr=stderr_handle,
        text=True, start_new_session=True)
    while process.poll() is None:
      current = meminfo()
      process_row = process_memory(process.pid)
      available = int(current.get("MemAvailable", 0))
      swap_used = max(
          0, int(current.get("SwapTotal", 0)) -
          int(current.get("SwapFree", 0)))
      monitor["process_rss_peak_bytes"] = max(
          int(monitor["process_rss_peak_bytes"]),
          int(process_row["VmRSS"]))
      monitor["process_swap_peak_bytes"] = max(
          int(monitor["process_swap_peak_bytes"]),
          int(process_row["VmSwap"]))
      minimum = monitor["system_available_min_bytes"]
      monitor["system_available_min_bytes"] = (
          available if minimum is None else min(int(minimum), available))
      monitor["system_swap_used_peak_bytes"] = max(
          int(monitor["system_swap_used_peak_bytes"]), swap_used)
      monitor["sample_count"] = int(monitor["sample_count"]) + 1
      if available < abort_bytes:
        guard_tripped = True
        stop_process_group(process, signal.SIGTERM)
        break
      if time.monotonic() - started > args.timeout_s:
        timed_out = True
        stop_process_group(process, signal.SIGTERM)
        break
      time.sleep(args.poll_interval_s)
    returncode = process.wait()
  stderr = (
      stderr_path.read_text(encoding="utf-8", errors="replace")
      if stderr_path.is_file() else "")
  lower_stderr = stderr.lower()
  oom = (
      returncode in (-9, 137) or
      "out of memory" in lower_stderr or
      "ze_result_error_out_of_device_memory" in lower_stderr or
      "ze_result_error_out_of_host_memory" in lower_stderr)
  return {
      "command": command,
      "returncode": returncode,
      "timed_out": timed_out,
      "memory_preflight": preflight,
      "memory_guard": {
          "abort_below_bytes": abort_bytes,
          "tripped": guard_tripped,
      },
      "monitor": monitor,
      "oom_observed": oom,
      "elapsed_seconds": time.monotonic() - started,
      "stdout": display_path(stdout_path),
      "stderr": display_path(stderr_path),
      "result": parse_worker_stdout(stdout_path),
  }


def distribution_metrics(reference: Any, candidate: Any, np: Any) -> dict[str, Any]:
  ref = np.asarray(reference, dtype=np.float64).reshape(-1)
  cand = np.asarray(candidate, dtype=np.float64).reshape(-1)
  if ref.shape != cand.shape or ref.size == 0:
    return {
        "shape_match": False, "finite": False,
        "kld_reference_to_candidate": math.inf,
        "top1_match": False, "max_abs": math.inf,
        "relative_l2": math.inf,
    }
  finite = bool(np.isfinite(ref).all() and np.isfinite(cand).all())
  if not finite:
    return {
        "shape_match": True, "finite": False,
        "kld_reference_to_candidate": math.inf,
        "top1_match": False, "max_abs": math.inf,
        "relative_l2": math.inf,
    }
  difference = cand - ref
  ref_probability = np.exp(ref - float(np.max(ref)))
  cand_probability = np.exp(cand - float(np.max(cand)))
  ref_probability /= float(ref_probability.sum())
  cand_probability /= float(cand_probability.sum())
  epsilon = np.finfo(np.float64).tiny
  reference_top1 = int(np.argmax(ref))
  candidate_top1 = int(np.argmax(cand))
  return {
      "shape_match": True,
      "finite": True,
      "max_abs": float(np.max(np.abs(difference))),
      "relative_l2": float(np.linalg.norm(difference) / np.linalg.norm(ref)),
      "kld_reference_to_candidate": float(np.sum(
          ref_probability * (
              np.log(np.maximum(ref_probability, epsilon)) -
              np.log(np.maximum(cand_probability, epsilon))))),
      "reference_top1": reference_top1,
      "candidate_top1": candidate_top1,
      "top1_match": reference_top1 == candidate_top1,
  }


def stage_profile(
    phase_rows: list[dict[str, Any]], np: Any,
) -> dict[str, Any]:
  metrics: dict[str, Any] = {}
  for stage in STAGES:
    key = f"{stage}_samples_us"
    values = [
        float(value)
        for row in phase_rows
        for value in row.get(key, [])
        if math.isfinite(float(value))
    ]
    if not values:
      metrics[stage] = {
          "sample_count": 0, "min_us": math.inf,
          "median_us": math.inf, "mean_us": math.inf,
          "p95_us": math.inf, "max_us": math.inf,
      }
      continue
    array = np.asarray(values, dtype=np.float64)
    metrics[stage] = {
        "sample_count": len(values),
        "min_us": float(np.min(array)),
        "median_us": float(np.median(array)),
        "mean_us": float(np.mean(array)),
        "p95_us": float(np.percentile(array, 95.0)),
        "max_us": float(np.max(array)),
    }
  nontraffic_values = []
  for row in phase_rows:
    vectors = [
        row.get("block_topk_samples_us", []),
        row.get("merge_samples_us", []),
        row.get("correction_samples_us", []),
    ]
    if not vectors or len({len(values) for values in vectors}) != 1:
      continue
    nontraffic_values.extend(
        sum(float(values[index]) for values in vectors)
        for index in range(len(vectors[0])))
  if nontraffic_values:
    array = np.asarray(nontraffic_values, dtype=np.float64)
    metrics["nontraffic"] = {
        "sample_count": len(nontraffic_values),
        "min_us": float(np.min(array)),
        "median_us": float(np.median(array)),
        "mean_us": float(np.mean(array)),
        "p95_us": float(np.percentile(array, 95.0)),
        "max_us": float(np.max(array)),
    }
  else:
    metrics["nontraffic"] = {
        "sample_count": 0, "min_us": math.inf,
        "median_us": math.inf, "mean_us": math.inf,
        "p95_us": math.inf, "max_us": math.inf,
    }
  matvec_median_us = float(metrics["matvec"]["median_us"])
  metrics["matvec"]["mandatory_bytes"] = MANDATORY_MATVEC_BYTES
  metrics["matvec"]["median_gb_s"] = (
      MANDATORY_MATVEC_BYTES / (matvec_median_us * 1000.0)
      if math.isfinite(matvec_median_us) and matvec_median_us > 0.0
      else 0.0)
  return metrics


def audit_component(
    launched: dict[str, Any], phases: tuple[int, ...], np: Any,
) -> dict[str, Any]:
  result = launched.get("result", {})
  phase_by_id = {
      int(row["phase"]): row
      for row in result.get("phases", [])
      if isinstance(row, dict) and "phase" in row
  }
  comparisons = []
  output_hashes = []
  reference_hash_matches = []
  phase_rows = []
  for phase in phases:
    row = phase_by_id.get(phase, {})
    phase_rows.append(row)
    raw_output = str(row.get("output", ""))
    output = Path(raw_output) if raw_output else Path("/nonexistent")
    if raw_output and not output.is_absolute():
      output = REPO / output
    if phase in FALLBACK_REFERENCE_PHASES:
      reference_path = (
          audit_component.reference_dir /
          f"step{phase:04d}-logits.f32")
      reference = np.fromfile(reference_path, dtype="<f4")
      candidate = (
          np.fromfile(output, dtype="<f4")
          if output.is_file() else np.asarray([], dtype=np.float32))
      comparisons.append({
          "phase": phase,
          **distribution_metrics(reference, candidate, np),
      })
      reference_hash_matches.append({
          "phase": phase,
          "reference_sha256": sha256(reference_path),
          "candidate_sha256": sha256(output) if output.is_file() else None,
          "match": (
              output.is_file() and sha256(output) == sha256(reference_path)),
      })
    output_hashes.append({
        "phase": phase,
        "path": display_path(output) if raw_output else None,
        "sha256": sha256(output) if output.is_file() else None,
    })
  profile = stage_profile(phase_rows, np)
  return {
      "phase_count": len(phase_by_id),
      "expected_phase_count": len(phases),
      "comparisons": comparisons,
      "reference_phase_count": len(comparisons),
      "expected_reference_phase_count": sum(
          phase in FALLBACK_REFERENCE_PHASES for phase in phases),
      "reference_hash_matches": reference_hash_matches,
      "bitwise_reference_matches": sum(
          row["match"] for row in reference_hash_matches),
      "output_hashes": output_hashes,
      "top1_matches": sum(
          row.get("top1_match") is True for row in comparisons),
      "finite_comparisons": sum(
          row.get("finite") is True for row in comparisons),
      "max_kld": max(
          (float(row["kld_reference_to_candidate"])
           for row in comparisons), default=math.inf),
      "max_relative_l2": max(
          (float(row["relative_l2"]) for row in comparisons),
          default=math.inf),
      "max_abs": max(
          (float(row["max_abs"]) for row in comparisons),
          default=math.inf),
      "device_name": result.get("device_name"),
      "worker_required_checks_passed":
          result.get("required_checks_passed"),
      "worker_contract": {
          key: result.get(key) for key in (
              "weight_bytes", "scale_bytes", "f16_output_bytes",
              "mandatory_matvec_bytes", "rows", "columns",
              "matvec_workgroups", "block_count", "topk",
              "all_finite", "all_stage_timestamps_positive",
              "all_selected_ids_valid")
      },
      "stage_profile": profile,
      "phase_timing": phase_rows,
  }


def source_contract() -> dict[str, Any]:
  module = MODULE_SOURCE.read_text(encoding="utf-8")
  cpp = CPP_SOURCE.read_text(encoding="utf-8")
  kernel_names = (
      "iq36_lm_head_gated_exact_component_q8_f16",
      "iq36_lm_head_gated_exact_component_matvec_f16",
      "iq36_lm_head_gated_exact_component_block_topk8_f16",
      "iq36_lm_head_gated_exact_component_topk8_merge_f32",
      "iq36_lm_head_gated_exact_component_correction_f16",
  )
  occurrences = {
      name: module.count("__kernel void " + name) for name in kernel_names
  }
  return {
      "kernel_occurrences": occurrences,
      "exactly_five_stages": all(value == 1 for value in occurrences.values()),
      "forced_fallback_has_no_gate_state":
          "__global const uint* state" not in module,
      "locked_shape_present":
          "#define IQ36_ROWS 248320U" in module and
          "#define IQ36_COLUMNS 2048U" in module,
      "product_geometry_present":
          "kMatvecWorkgroups = 384U" in cpp and
          "kBlockCount = (kRows + kBlockRows - 1U) / kBlockRows" in cpp,
      "f16_hidden_and_output_present":
          "__global const half* input" in module and
          "__global half* output" in module,
  }


def write_summary(output: Path, result: dict[str, Any]) -> None:
  repeat = result["repeat_audit"]
  confirm = result["confirm_audit"]
  rp = repeat["stage_profile"]
  cp = confirm["stage_profile"]
  direction = result["profile_decision"]
  lines = [
      "# OpenVINO LM-head gated-exact component profile",
      "",
      f"- verdict: `{result['verdict']}`",
      f"- required checks passed: "
      f"`{str(result['required_checks_passed']).lower()}`",
      f"- repeat/confirm exact top-1: "
      f"`{repeat['top1_matches']}/"
      f"{repeat['expected_reference_phase_count']}` / "
      f"`{confirm['top1_matches']}/"
      f"{confirm['expected_reference_phase_count']}`",
      f"- repeat/confirm max KLD: "
      f"`{repeat['max_kld']}` / `{confirm['max_kld']}`",
      f"- repeat/confirm matvec median: "
      f"`{rp['matvec']['median_us']:.3f}` / "
      f"`{cp['matvec']['median_us']:.3f} us`",
      f"- repeat/confirm physical rate: "
      f"`{rp['matvec']['median_gb_s']:.3f}` / "
      f"`{cp['matvec']['median_gb_s']:.3f} GB/s`",
      f"- repeat/confirm block top-8 median: "
      f"`{rp['block_topk']['median_us']:.3f}` / "
      f"`{cp['block_topk']['median_us']:.3f} us`",
      f"- repeat/confirm merge median: "
      f"`{rp['merge']['median_us']:.3f}` / "
      f"`{cp['merge']['median_us']:.3f} us`",
      f"- repeat/confirm correction median: "
      f"`{rp['correction']['median_us']:.3f}` / "
      f"`{cp['correction']['median_us']:.3f} us`",
      f"- repeat/confirm fallback shell median: "
      f"`{rp['fallback_shell']['median_us']:.3f}` / "
      f"`{cp['fallback_shell']['median_us']:.3f} us`",
      f"- product jitter cut required: "
      f"`{result['bound']['required_saving_us']:.3f} us`",
      f"- next source-bound stage: "
      f"`{direction['next_stage_to_source_bound']}` "
      f"(`{direction['largest_nonmatvec_stage_median_us']:.3f} us` median)",
      f"- OOM/guard repeat/confirm: "
      f"`{result['repeat_worker']['oom_observed']}` / "
      f"`{result['repeat_worker']['memory_guard']['tripped']}`, "
      f"`{result['confirm_worker']['oom_observed']}` / "
      f"`{result['confirm_worker']['memory_guard']['tripped']}`",
      "",
      "This is a component profile, not a product speed claim. It launches no "
      "OpenVINO model worker and does not promote a kernel variant.",
      "",
  ]
  (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  args = parse_args()
  import numpy as np

  output = args.output.resolve()
  generated = output / "generated"
  generated.mkdir(parents=True, exist_ok=False)
  args.bound = args.bound.resolve()
  args.capture_dir = args.capture_dir.resolve()
  args.reference_dir = args.reference_dir.resolve()
  args.build_dir = args.build_dir.resolve()
  phases = tuple(range(args.first_phase, args.last_phase + 1))
  audit_component.reference_dir = args.reference_dir
  capture_worker = args.capture_dir / "worker-result.json"
  reference_worker = args.reference_dir / "worker-result.json"
  required = [
      MODEL_BIN, args.cmake, MODULE_SOURCE, CPP_SOURCE, BOUNDARIES,
      PRODUCT_PATCH, args.bound, capture_worker, reference_worker,
      *[
          args.capture_dir / f"step{phase:04d}-lm-head-input.f32"
          for phase in phases
      ],
      *[
          args.reference_dir / f"step{phase:04d}-logits.f32"
          for phase in phases
      ],
  ]
  missing = [str(path) for path in required if not path.is_file()]
  if missing:
    raise SystemExit("missing component inputs: " + ", ".join(missing))

  state = git_state(output)
  concurrent = other_gpu_workers()
  if concurrent:
    raise RuntimeError(f"concurrent GPU worker detected: {concurrent}")
  bound = load_json(args.bound)
  capture_worker_row = load_json(capture_worker)
  reference_worker_row = load_json(reference_worker)
  contract = source_contract()

  module_compile = run([
      "ocloc", "compile", "-file", str(MODULE_SOURCE),
      "-device", "0xb080", "-output", "iq36_gated_exact_component",
      "-out_dir", str(generated), "-output_no_suffix", "--format", "zebin",
      "-options", "-cl-std=CL3.0", "-q",
  ], 120)
  module = generated / "iq36_gated_exact_component.bin"
  module_validate = (
      run(["ocloc", "validate", "-file", str(module)], 60)
      if module.is_file() else {
          "command": [], "returncode": 125, "stdout": "",
          "stderr": "module missing", "timed_out": False,
      })
  configure = run([
      str(args.cmake), "-S", str(REPO / "engine"), "-B",
      str(args.build_dir), "-DCMAKE_BUILD_TYPE=Release",
  ])
  build = (
      run([
          str(args.cmake), "--build", str(args.build_dir),
          "--target", TARGET, "-j", "1",
      ], 300)
      if configure["returncode"] == 0 else {
          "command": [], "returncode": 125, "stdout": "",
          "stderr": "configure failed", "timed_out": False,
      })
  link_map = (
      run(["ldd", str(BINARY)], 30) if BINARY.is_file() else {
          "command": [], "returncode": 125, "stdout": "",
          "stderr": "binary missing", "timed_out": False,
      })

  runnable = (
      module_compile["returncode"] == 0 and
      module_validate["returncode"] == 0 and
      build["returncode"] == 0 and module.is_file() and BINARY.is_file())
  repeat = (
      launch_component(args, output / "repeat", module)
      if runnable else {
          "returncode": 125, "timed_out": False,
          "memory_guard": {"tripped": False},
          "oom_observed": False, "result": {},
      })
  repeat_audit = audit_component(repeat, phases, np)
  confirm = (
      launch_component(args, output / "confirm", module)
      if repeat["returncode"] == 0 else {
          "returncode": 125, "timed_out": False,
          "memory_guard": {"tripped": False},
          "oom_observed": False, "result": {},
      })
  confirm_audit = audit_component(confirm, phases, np)

  output_hash_match = all(
      lhs["sha256"] is not None and lhs["sha256"] == rhs["sha256"]
      for lhs, rhs in zip(
          repeat_audit["output_hashes"], confirm_audit["output_hashes"]))
  required_sample_count = len(phases) * args.samples
  repeat_profile = repeat_audit["stage_profile"]
  confirm_profile = confirm_audit["stage_profile"]
  repeat_shell = float(repeat_profile["fallback_shell"]["median_us"])
  confirm_shell = float(confirm_profile["fallback_shell"]["median_us"])
  shell_relative_delta = (
      abs(repeat_shell - confirm_shell) / min(repeat_shell, confirm_shell)
      if min(repeat_shell, confirm_shell) > 0.0 else math.inf)
  required_saving_us = float(bound["required_saving_ms"]) * 1000.0
  traffic_floor_us = (
      float(bound["traffic_bound"]["traffic_floor_ms"]) * 1000.0)
  bandwidth_lcb = float(bound["traffic_bound"]["bandwidth_lcb_gb_s"])
  observed_slow_increment_us = (
      float(bound["traffic_bound"]["minimum_observed_slow_increment_ms"]) *
      1000.0)
  nonmatvec_stage_medians = {
      stage: statistics.median([
          float(repeat_profile[stage]["median_us"]),
          float(confirm_profile[stage]["median_us"]),
      ])
      for stage in ("block_topk", "merge", "correction")
  }
  next_stage = max(nonmatvec_stage_medians, key=nonmatvec_stage_medians.get)
  largest_nonmatvec_us = nonmatvec_stage_medians[next_stage]
  profile_decision = {
      "next_stage_to_source_bound": next_stage,
      "nonmatvec_stage_medians_us": nonmatvec_stage_medians,
      "largest_nonmatvec_stage_median_us": largest_nonmatvec_us,
      "required_product_saving_us": required_saving_us,
      "stage_gross_time_multiple_over_required_saving":
          largest_nonmatvec_us / required_saving_us,
      "source_candidate_admitted": False,
      "reason": (
          "baseline only; source-bound the largest non-matvec stage before "
          "choosing one implementation cut"),
  }

  lower_links = (
      link_map.get("stdout", "") + link_map.get("stderr", "")).lower()
  worker_contract_expected = {
      "weight_bytes": WEIGHT_BYTES,
      "scale_bytes": SCALE_BYTES,
      "f16_output_bytes": F16_OUTPUT_BYTES,
      "mandatory_matvec_bytes": MANDATORY_MATVEC_BYTES,
      "rows": 248320,
      "columns": 2048,
      "matvec_workgroups": 384,
      "block_count": 970,
      "topk": 8,
      "all_finite": True,
      "all_stage_timestamps_positive": True,
      "all_selected_ids_valid": True,
  }
  checks = [
      check("repository_clean_at_gate", not state["dirty"], git=state),
      check("no_concurrent_gpu_worker_at_launch", not concurrent,
            concurrent=concurrent),
      check("seq2186_admits_this_exact_component_profile",
            bound.get("required_checks_passed") is True and
            bound.get("verdict") ==
                "admit_stage_separated_gated_exact_component_profile" and
            int(bound["component_profile_contract"][
                "minimum_profile_samples_per_hidden"]) <= args.samples),
      check("captured_hidden_and_reference_have_identical_token_trajectory",
            capture_worker_row.get("generated_token_ids_sha256") ==
                reference_worker_row.get("generated_token_ids_sha256") and
            capture_worker_row.get("generated_token_count") == len(phases) and
            reference_worker_row.get("generated_token_count") == len(phases),
            capture_sha256=capture_worker_row.get(
                "generated_token_ids_sha256"),
            reference_sha256=reference_worker_row.get(
                "generated_token_ids_sha256")),
      check("forced_component_mirrors_five_accepted_data_plane_stages",
            contract["exactly_five_stages"] and
            contract["forced_fallback_has_no_gate_state"] and
            contract["locked_shape_present"] and
            contract["product_geometry_present"] and
            contract["f16_hidden_and_output_present"],
            source_contract=contract),
      check("module_compiles_and_validates_for_ptl",
            module_compile["returncode"] == 0 and
            module_validate["returncode"] == 0 and module.is_file()),
      check("component_builds_serially",
            configure["returncode"] == 0 and build["returncode"] == 0 and
            BINARY.is_file()),
      check("level_zero_only_runtime_boundary",
            link_map["returncode"] == 0 and "libze_loader" in lower_links and
            "openvino" not in lower_links and "libdnnl" not in lower_links),
      check("repeat_and_confirm_complete_without_oom",
            all(
                worker["returncode"] == 0 and
                not worker["timed_out"] and
                not worker["memory_guard"]["tripped"] and
                not worker["oom_observed"]
                for worker in (repeat, confirm)),
            repeat={
                key: repeat.get(key) for key in (
                    "returncode", "timed_out", "memory_guard",
                    "oom_observed", "monitor")
            },
            confirm={
                key: confirm.get(key) for key in (
                    "returncode", "timed_out", "memory_guard",
                    "oom_observed", "monitor")
            }),
      check("worker_shape_geometry_and_stage_contract_are_exact",
            all(
                audit["worker_required_checks_passed"] is True and
                audit["worker_contract"] == worker_contract_expected
                for audit in (repeat_audit, confirm_audit)),
            expected=worker_contract_expected,
            repeat=repeat_audit["worker_contract"],
            confirm=confirm_audit["worker_contract"]),
      check("accepted_fallback_anchor_outputs_are_bitwise_exact",
            all(
                audit["phase_count"] == len(phases) and
                audit["reference_phase_count"] ==
                    len(FALLBACK_REFERENCE_PHASES) and
                audit["finite_comparisons"] ==
                    len(FALLBACK_REFERENCE_PHASES) and
                audit["top1_matches"] ==
                    len(FALLBACK_REFERENCE_PHASES) and
                audit["bitwise_reference_matches"] ==
                    len(FALLBACK_REFERENCE_PHASES) and
                audit["max_kld"] <= KLD_LIMIT
                for audit in (repeat_audit, confirm_audit)),
            kld_limit=KLD_LIMIT,
            reference_phases=list(FALLBACK_REFERENCE_PHASES),
            repeat={
                key: repeat_audit[key] for key in (
                    "phase_count", "reference_phase_count",
                    "finite_comparisons", "top1_matches",
                    "bitwise_reference_matches", "max_kld",
                    "max_relative_l2", "max_abs")
            },
            confirm={
                key: confirm_audit[key] for key in (
                    "phase_count", "reference_phase_count",
                    "finite_comparisons", "top1_matches",
                    "bitwise_reference_matches", "max_kld",
                    "max_relative_l2", "max_abs")
            }),
      check("repeat_and_confirm_outputs_are_bitwise_deterministic",
            output_hash_match),
      check("every_stage_has_twenty_samples_per_hidden",
            all(
                audit["stage_profile"][stage]["sample_count"] ==
                    required_sample_count
                for audit in (repeat_audit, confirm_audit)
                for stage in STAGES),
            required_sample_count=required_sample_count,
            repeat={
                stage: repeat_profile[stage]["sample_count"]
                for stage in STAGES
            },
            confirm={
                stage: confirm_profile[stage]["sample_count"]
                for stage in STAGES
            }),
      check("repeat_confirm_fallback_shell_is_profile_stable",
            shell_relative_delta <= 0.03,
            relative_delta=shell_relative_delta,
            repeat_median_us=repeat_shell,
            confirm_median_us=confirm_shell),
  ]
  passed = all(row["pass"] for row in checks)
  verdict = (
      "admit_stage_separated_gated_exact_component_baseline"
      if passed else
      "reject_stage_separated_gated_exact_component_baseline")
  result = {
      "schema": SCHEMA,
      "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
      "workstream": WS,
      "git": state,
      "verdict": verdict,
      "required_checks_passed": passed,
      "checks": checks,
      "bound": {
          "path": display_path(args.bound),
          "sha256": sha256(args.bound),
          "required_saving_us": required_saving_us,
          "traffic_floor_us": traffic_floor_us,
          "accepted_bandwidth_lcb_gb_s": bandwidth_lcb,
          "observed_minimum_slow_increment_us": observed_slow_increment_us,
      },
      "profile_decision": profile_decision,
      "repeat_audit": repeat_audit,
      "confirm_audit": confirm_audit,
      "repeat_worker": {
          key: value for key, value in repeat.items() if key != "result"
      },
      "confirm_worker": {
          key: value for key, value in confirm.items() if key != "result"
      },
      "build": {
          "module_compile": module_compile,
          "module_validate": module_validate,
          "configure": configure,
          "build": build,
          "link_map": link_map,
      },
      "inputs": {
          "capture_dir": display_path(args.capture_dir),
          "reference_dir": display_path(args.reference_dir),
          "capture_worker_sha256": sha256(capture_worker),
          "reference_worker_sha256": sha256(reference_worker),
          "model_bin": str(MODEL_BIN),
          "model_bin_size": MODEL_BIN.stat().st_size,
          "phase_range": [args.first_phase, args.last_phase],
          "warmup": args.warmup,
          "samples_per_hidden": args.samples,
          "sources": {
              display_path(path): sha256(path)
              for path in (
                  MODULE_SOURCE, CPP_SOURCE, BOUNDARIES, PRODUCT_PATCH)
          },
          "source_contract": contract,
          "module": display_path(module),
          "module_sha256": sha256(module) if module.is_file() else None,
      },
      "gpu_component_workers_launched": 2,
      "model_workers_launched": 0,
      "stock_workers_launched": 0,
      "workers_concurrent": False,
      "product_promotion_ready": False,
      "speedup_claims_allowed": False,
  }
  write_json(output / "result.json", result)
  write_json(output / "manifest.json", {
      "schema": SCHEMA,
      "git": state,
      "verdict": verdict,
      "required_checks_passed": passed,
      "tool": display_path(Path(__file__)),
      "module_source": display_path(MODULE_SOURCE),
      "cpp_source": display_path(CPP_SOURCE),
      "phase_range": [args.first_phase, args.last_phase],
      "samples_per_hidden": args.samples,
      "memory_preflight_gib": args.min_available_gib,
      "memory_abort_gib": args.abort_below_available_gib,
      "gpu_component_workers_launched": 2,
      "model_workers_launched": 0,
      "product_promotion_ready": False,
      "speedup_claims_allowed": False,
  })
  write_summary(output, result)
  print(json.dumps({
      "output": display_path(output),
      "verdict": verdict,
      "required_checks_passed": passed,
      "repeat_max_kld": repeat_audit["max_kld"],
      "confirm_max_kld": confirm_audit["max_kld"],
      "repeat_matvec_median_us":
          repeat_profile["matvec"]["median_us"],
      "confirm_matvec_median_us":
          confirm_profile["matvec"]["median_us"],
      "repeat_matvec_gb_s": repeat_profile["matvec"]["median_gb_s"],
      "confirm_matvec_gb_s": confirm_profile["matvec"]["median_gb_s"],
      "repeat_block_topk_median_us":
          repeat_profile["block_topk"]["median_us"],
      "confirm_block_topk_median_us":
          confirm_profile["block_topk"]["median_us"],
      "repeat_merge_median_us": repeat_profile["merge"]["median_us"],
      "confirm_merge_median_us": confirm_profile["merge"]["median_us"],
      "repeat_correction_median_us":
          repeat_profile["correction"]["median_us"],
      "confirm_correction_median_us":
          confirm_profile["correction"]["median_us"],
      "next_stage_to_source_bound": next_stage,
  }, sort_keys=True))
  return 0 if passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
