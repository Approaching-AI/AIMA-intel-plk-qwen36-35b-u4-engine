#!/usr/bin/env python3
"""Gate the seq2289 affine-Q4/group128 certificate on the real PTL GPU.

The tool builds one standalone Level Zero component, runs repeat and confirm
workers strictly serially, verifies all 2000 fixed population rows, forces the
capacity-overflow fallback, and measures interleaved ABBA full-I8/certificate
latency.  It does not compile an OpenVINO model or edit/build a product plugin.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import signal
import statistics
import struct
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = (
    "intel-qwen36-openvino-lm-head-affine-q4-group128-component-v1")
MODEL_BIN = Path(
    "/home/intel/Qwen3.6-35B-A3B-ov/openvino_language_model.bin")
MODULE_SOURCE = ROOT / (
    "engine/gpu/opencl/"
    "iq36_lm_head_affine_q4_group128_component.cl")
CPP_SOURCE = ROOT / (
    "engine/tools/"
    "openvino_lm_head_affine_q4_group128_component.cpp")
BOUNDARIES = ROOT / "engine/boundaries.json"
CMAKE = Path("/home/intel/intel-box-env/conda/bin/cmake")
BUILD_DIR = ROOT / "build/engine"
TARGET = "iq36-openvino-lm-head-affine-q4-group128-component"
POPULATION_ROOT = ROOT / (
    "output/openvino-lm-head-slow-event-population-bound-"
    "20260801Tseq2287-clean")
POPULATION_METRICS = POPULATION_ROOT / "metrics.json"
POPULATION_MANIFEST = POPULATION_ROOT / "manifest.json"
EVENTS = POPULATION_ROOT / "slow-events.json"
HIDDEN = POPULATION_ROOT / "slow-lm-head-inputs.f16"
CERTIFICATE_ROOT = ROOT / (
    "output/openvino-lm-head-affine-q4-group128-"
    "population-certificate-20260801Tseq2289-clean")
CERTIFICATE_METRICS = CERTIFICATE_ROOT / "metrics.json"
CERTIFICATE_MANIFEST = CERTIFICATE_ROOT / "manifest.json"
CPU_CANDIDATE_COUNTS = CERTIFICATE_ROOT / "cpu-candidate-counts.i32"

ROWS = 248_320
COLUMNS = 2_048
EVENT_COUNT = 2_000
CAPACITY = 16_812
FULL_I8_BYTES = 509_552_640
FIXED_ACTIVE_BYTES = 271_165_440
PER_CANDIDATE_BYTES = 2_056
TRAFFIC_CAP = 0.60
REQUIRED_PRODUCT_SAVING_US = 11.20375
FALLBACK_RATE = 50.0 / 495.0
REQUIRED_FALLBACK_SAVING_US = (
    REQUIRED_PRODUCT_SAVING_US / FALLBACK_RATE)
BOOTSTRAP_SEED = 2289
BOOTSTRAP_RESAMPLES = 20_000
TIMING_ORDINALS = (
    0, 125, 250, 375, 500, 625, 750, 875, 923,
    1000, 1125, 1250, 1375, 1500, 1625, 1750, 1875, 1999,
)
EXPECTED_KERNELS = (
    "iq36_affine_q4_q8_f16",
    "iq36_affine_q4_hidden_group_norms_f16",
    "iq36_affine_q4_reset",
    "iq36_affine_q4_bound_select_f16",
    "iq36_affine_q4_exact_candidates_f16",
    "iq36_affine_q4_candidate_top1_f16",
    "iq36_affine_q4_full_i8_q8_matvec_f16",
    "iq36_affine_q4_reference_matvec_f16",
    "iq36_affine_q4_violation_reset",
    "iq36_affine_q4_bound_violations_f16",
    "iq36_affine_q4_full_top1_f16",
)
EXPECTED_SHA256 = {
    MODEL_BIN:
        "46140b595760e891d9626c5bfaffc2c998cce176d0de7f6c290af5ae1f2393a4",
    POPULATION_METRICS:
        "82f53963168367227ee9e621fe6e8f64b8609f7b408e20e2c950c2226fde6fb8",
    POPULATION_MANIFEST:
        "99c21f0c52d9c3946ee5c75d1ccf48c19e8d5fd77dcaf19c06ad90c0370883cd",
    EVENTS:
        "5a1b7aa5954c7fd5b0f25032f7c9cf0a43e7cf3e7cc331bad87ad1e721f9caf9",
    HIDDEN:
        "48e8bd79c03f1adcbc1fbb5d784aa306c859272e46f186de2ccf0c0ce999f836",
    CERTIFICATE_METRICS:
        "64f601b79db8c7bd2fcdba4172e23f930ae8d4e0de80289cc47492c1744af5f1",
    CERTIFICATE_MANIFEST:
        "93092a6ac2cbff239168948dc0b7f3c5690c4fa08266ab459f08856b37b93508",
    CPU_CANDIDATE_COUNTS:
        "96a24c80c7fe4c4c0fc427a86698b49601ef51e1733171688c2c56fd155c641b",
}


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--cmake", type=Path, default=CMAKE)
  parser.add_argument("--build-dir", type=Path, default=BUILD_DIR)
  parser.add_argument("--warmup", type=int, default=3)
  parser.add_argument("--blocks", type=int, default=8)
  parser.add_argument("--correctness-events", type=int, default=EVENT_COUNT)
  parser.add_argument("--timeout-s", type=int, default=300)
  parser.add_argument("--min-available-gib", type=float, default=8.0)
  parser.add_argument(
      "--abort-below-available-gib", type=float, default=4.0)
  parser.add_argument("--poll-interval-s", type=float, default=0.1)
  args = parser.parse_args()
  if args.warmup < 0 or args.blocks < 8:
    parser.error("warmup must be nonnegative and blocks at least 8")
  if not 1 <= args.correctness_events <= EVENT_COUNT:
    parser.error("correctness events must be in [1, 2000]")
  if args.timeout_s <= 0 or args.poll_interval_s <= 0.0:
    parser.error("timeout and poll interval must be positive")
  if (
      args.min_available_gib < 0.0 or
      args.abort_below_available_gib < 0.0 or
      args.abort_below_available_gib > args.min_available_gib
  ):
    parser.error("memory thresholds are invalid")
  return args


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


def load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise TypeError(f"expected JSON object: {path}")
  return value


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True) + "\n",
      encoding="utf-8")


def read_i32(path: Path, count: int) -> list[int]:
  payload = path.read_bytes()
  if len(payload) != count * 4:
    raise ValueError(
        f"expected {count * 4} bytes in {path}, observed {len(payload)}")
  return list(struct.unpack(f"<{count}i", payload))


def run(command: list[str], timeout_s: int = 300) -> dict[str, Any]:
  try:
    result = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, check=False,
        timeout=timeout_s, encoding="utf-8", errors="replace")
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "timed_out": False,
    }
  except subprocess.TimeoutExpired as error:
    return {
        "command": command,
        "returncode": 124,
        "stdout": str(error.stdout or ""),
        "stderr": str(error.stderr or "") + "\ntimeout",
        "timed_out": True,
    }


def check(name: str, passed: bool, **details: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **details}


def git_state(output: Path) -> dict[str, Any]:
  commit = run(["git", "rev-parse", "HEAD"], 30)["stdout"].strip()
  upstream = run(["git", "rev-parse", "@{upstream}"], 30)["stdout"].strip()
  rows = run([
      "git", "status", "--porcelain=v1", "--untracked-files=all",
  ], 30)["stdout"].splitlines()
  try:
    output_relative = str(output.resolve().relative_to(ROOT))
  except ValueError:
    output_relative = ""
  rows = [
      row for row in rows
      if not output_relative or output_relative not in row
  ]
  return {
      "branch": run(["git", "branch", "--show-current"], 30)[
          "stdout"].strip(),
      "commit": commit,
      "dirty": bool(rows),
      "status": rows,
      "pushed": bool(commit) and commit == upstream,
      "upstream_commit": upstream,
  }


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
  try:
    lines = path.read_text(
        encoding="utf-8", errors="replace").splitlines()
  except (FileNotFoundError, ProcessLookupError):
    return {"VmRSS": 0, "VmSwap": 0}
  selected: dict[str, int] = {}
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
  needles = (
      TARGET,
      "intel-qwen36-openvino-hot-cold-attention-gate.py --worker-config",
      "openvino_lm_head_gated_exact_component",
  )
  for path in Path("/proc").iterdir():
    if not path.name.isdigit() or int(path.name) == os.getpid():
      continue
    try:
      command = (path / "cmdline").read_bytes().replace(
          b"\0", b" ").decode("utf-8", errors="replace")
    except (FileNotFoundError, PermissionError, ProcessLookupError):
      continue
    if any(needle in command for needle in needles):
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
    args: argparse.Namespace, binary: Path, directory: Path, module: Path,
    seed_ids: Path, timing_ids: Path,
) -> dict[str, Any]:
  directory.mkdir(parents=True)
  stdout_path = directory / "component.stdout"
  stderr_path = directory / "component.stderr"
  command = [
      str(binary), str(MODEL_BIN), str(module), str(HIDDEN),
      str(seed_ids), str(timing_ids), str(args.warmup),
      str(args.blocks), str(args.correctness_events),
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
        command, cwd=ROOT, stdout=stdout_handle, stderr=stderr_handle,
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
      "stdout": display(stdout_path),
      "stderr": display(stderr_path),
      "result": parse_worker_stdout(stdout_path),
  }


def lower_bootstrap_median(values: list[float]) -> float:
  if not values or any(not math.isfinite(value) for value in values):
    return -math.inf
  rng = random.Random(BOOTSTRAP_SEED)
  medians = sorted(
      statistics.median(rng.choices(values, k=len(values)))
      for _ in range(BOOTSTRAP_RESAMPLES))
  rank = max(1, math.ceil(0.05 * len(medians)))
  return float(medians[rank - 1])


def distribution(values: list[float]) -> dict[str, Any]:
  if not values or any(not math.isfinite(value) for value in values):
    return {
        "sample_count": len(values),
        "minimum_us": math.nan,
        "median_us": math.nan,
        "mean_us": math.nan,
        "maximum_us": math.nan,
    }
  return {
      "sample_count": len(values),
      "minimum_us": min(values),
      "median_us": statistics.median(values),
      "mean_us": statistics.mean(values),
      "maximum_us": max(values),
  }


def candidate_count_delta(
    observed: list[int], reference: list[int],
) -> dict[str, Any]:
  if len(observed) != len(reference):
    return {
        "geometry_exact": False,
        "observed_events": len(observed),
        "reference_events": len(reference),
    }
  deltas = [
      observed_value - reference_value
      for observed_value, reference_value in zip(observed, reference)
  ]
  absolute = [abs(value) for value in deltas]
  return {
      "geometry_exact": True,
      "observed_events": len(observed),
      "reference_events": len(reference),
      "exact_event_count": sum(value == 0 for value in deltas),
      "different_event_count": sum(value != 0 for value in deltas),
      "observed_lower_event_count": sum(value < 0 for value in deltas),
      "observed_higher_event_count": sum(value > 0 for value in deltas),
      "signed_candidate_delta": sum(deltas),
      "maximum_absolute_event_delta": max(absolute, default=0),
      "minimum_event_delta": min(deltas, default=0),
      "maximum_event_delta": max(deltas, default=0),
      "reference_minimum": min(reference, default=None),
      "reference_maximum": max(reference, default=None),
      "observed_minimum": min(observed, default=None),
      "observed_maximum": max(observed, default=None),
  }


def audit_worker(worker: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
  result = worker.get("result") or {}
  correctness = result.get("correctness") or {}
  forced = result.get("forced_overflow") or {}
  samples = (result.get("timing") or {}).get("samples") or []
  expected_samples = len(TIMING_ORDINALS) * args.blocks
  baseline_kernel = [
      float(row["baseline_kernel_us"]) for row in samples
      if isinstance(row, dict) and "baseline_kernel_us" in row
  ]
  candidate_kernel = [
      float(row["candidate_kernel_us"]) for row in samples
      if isinstance(row, dict) and "candidate_kernel_us" in row
  ]
  saving_kernel = [
      float(row["saving_kernel_us"]) for row in samples
      if isinstance(row, dict) and "saving_kernel_us" in row
  ]
  baseline_wall = [
      float(row["baseline_wall_us"]) for row in samples
      if isinstance(row, dict) and "baseline_wall_us" in row
  ]
  candidate_wall = [
      float(row["candidate_wall_us"]) for row in samples
      if isinstance(row, dict) and "candidate_wall_us" in row
  ]
  saving_wall = [
      float(row["saving_wall_us"]) for row in samples
      if isinstance(row, dict) and "saving_wall_us" in row
  ]
  arithmetic_exact = (
      len(samples) == expected_samples and
      len(baseline_kernel) == len(candidate_kernel) ==
          len(saving_kernel) == expected_samples and
      len(baseline_wall) == len(candidate_wall) ==
          len(saving_wall) == expected_samples
  )
  if arithmetic_exact:
    for row in samples:
      if (
          abs(
              (float(row["baseline_kernel_us"]) -
               float(row["candidate_kernel_us"])) -
              float(row["saving_kernel_us"])) > 1e-5 or
          abs(
              (float(row["baseline_wall_us"]) -
               float(row["candidate_wall_us"])) -
              float(row["saving_wall_us"])) > 1e-5
      ):
        arithmetic_exact = False
        break
  by_event_kernel: dict[int, list[float]] = {}
  by_event_wall: dict[int, list[float]] = {}
  for row in samples:
    event = int(row.get("event", -1))
    by_event_kernel.setdefault(event, []).append(
        float(row.get("saving_kernel_us", math.nan)))
    by_event_wall.setdefault(event, []).append(
        float(row.get("saving_wall_us", math.nan)))
  event_kernel_medians = [
      statistics.median(by_event_kernel[event])
      for event in TIMING_ORDINALS
      if len(by_event_kernel.get(event, [])) == args.blocks
  ]
  event_wall_medians = [
      statistics.median(by_event_wall[event])
      for event in TIMING_ORDINALS
      if len(by_event_wall.get(event, [])) == args.blocks
  ]
  candidate_counts = [
      int(value) for value in correctness.get("candidate_counts", [])
  ]
  maximum_count = (
      max(candidate_counts) if candidate_counts else math.inf)
  active_bytes = (
      FIXED_ACTIVE_BYTES + int(maximum_count) * PER_CANDIDATE_BYTES
      if math.isfinite(maximum_count) else math.inf)
  active_ratio = (
      active_bytes / FULL_I8_BYTES
      if math.isfinite(active_bytes) else math.inf)
  return {
      "result_schema": result.get("schema"),
      "required_checks_passed": result.get("required_checks_passed"),
      "correctness": {
          "events": int(correctness.get("events", -1)),
          "candidate_counts": candidate_counts,
          "maximum_candidate_count": maximum_count,
          "overflow_count": int(correctness.get("overflow_count", -1)),
          "bound_violation_count": int(
              correctness.get("bound_violation_count", -1)),
          "token_mismatch_count": int(
              correctness.get("token_mismatch_count", -1)),
          "reference_mismatch_count": int(
              correctness.get("reference_mismatch_count", -1)),
          "maximum_active_bytes": active_bytes,
          "maximum_active_ratio": active_ratio,
      },
      "forced_overflow": forced,
      "resources": result.get("resources") or {},
      "packing_seconds": result.get("packing_seconds"),
      "packed_bytes": result.get("packed_bytes") or {},
      "timing": {
          "sample_geometry_exact": arithmetic_exact,
          "sample_count": len(samples),
          "baseline_kernel": distribution(baseline_kernel),
          "candidate_kernel": distribution(candidate_kernel),
          "saving_kernel": distribution(saving_kernel),
          "baseline_wall": distribution(baseline_wall),
          "candidate_wall": distribution(candidate_wall),
          "saving_wall": distribution(saving_wall),
          "event_kernel_medians_us": event_kernel_medians,
          "event_wall_medians_us": event_wall_medians,
          "kernel_saving_lcb_us":
              lower_bootstrap_median(event_kernel_medians),
          "wall_saving_lcb_us":
              lower_bootstrap_median(event_wall_medians),
          "candidate_stage_medians_us": {
              name: statistics.median([
                  float(row["candidate_stages_us"][index])
                  for row in samples
              ]) if samples else math.nan
              for index, name in enumerate((
                  "reset", "hidden_norms", "bound_select",
                  "exact_candidates", "top1"))
          },
      },
  }


def source_contract() -> dict[str, Any]:
  module = MODULE_SOURCE.read_text(encoding="utf-8")
  cpp = CPP_SOURCE.read_text(encoding="utf-8")
  boundaries = load_json(BOUNDARIES)
  kernels = re.findall(
      r"__kernel\s+void\s+([A-Za-z0-9_]+)", module)
  targets = [
      row for row in boundaries.get("infra_targets", [])
      if isinstance(row, dict) and row.get("target") == TARGET
  ]
  return {
      "kernels": kernels,
      "kernels_exact": tuple(kernels) == EXPECTED_KERNELS,
      "group128_locked": (
          "#define IQ36_GROUP128 128U" in module and
          "#define IQ36_GROUPS128 16U" in module),
      "capacity_locked": "#define IQ36_CAPACITY 16812U" in module,
      "full_i8_fallback_present": (
          "iq36_affine_q4_full_i8_q8_matvec_f16" in module and
          "CheckForcedOverflow" in cpp and
          "candidate_count > kCapacity" in cpp),
      "serial_abba_present": (
          "RunTimed(baseline_list_, 2U)" in cpp and
          cpp.count("RunTimed(candidate_list_, 5U)") >= 3),
      "target_registration": targets,
      "target_registered_once": len(targets) == 1,
  }


def main() -> int:
  args = parse_args()
  output = args.output.resolve()
  if output.exists():
    raise SystemExit(f"output already exists: {output}")
  generated = output / "generated"
  generated.mkdir(parents=True)
  args.build_dir = args.build_dir.resolve()
  binary = args.build_dir / TARGET
  state = git_state(output)
  if (
      state["branch"] != "main" or state["dirty"] or
      not state["pushed"]
  ):
    raise SystemExit("component gate requires clean pushed main")
  concurrent = other_gpu_workers()
  if concurrent:
    raise SystemExit(
        "concurrent GPU worker detected: " +
        json.dumps(concurrent, sort_keys=True))

  missing = [
      str(path) for path in (
          MODEL_BIN, MODULE_SOURCE, CPP_SOURCE, BOUNDARIES,
          POPULATION_METRICS, POPULATION_MANIFEST, EVENTS, HIDDEN,
          CERTIFICATE_METRICS, CERTIFICATE_MANIFEST,
          CPU_CANDIDATE_COUNTS, args.cmake,
      )
      if not path.is_file()
  ]
  if missing:
    raise SystemExit("missing component inputs: " + ", ".join(missing))
  observed_hashes = {path: sha256(path) for path in EXPECTED_SHA256}
  mismatches = {
      display(path): {
          "expected": expected,
          "observed": observed_hashes[path],
      }
      for path, expected in EXPECTED_SHA256.items()
      if observed_hashes[path] != expected
  }
  if mismatches:
    raise SystemExit(
        "registered input hash mismatch: " +
        json.dumps(mismatches, sort_keys=True))

  events_payload = load_json(EVENTS)
  events = events_payload.get("events") or []
  if (
      len(events) != EVENT_COUNT or
      [int(row["ordinal"]) for row in events] != list(range(EVENT_COUNT))
  ):
    raise SystemExit("population event ordinals are not exact")
  cpu_candidate_counts = read_i32(CPU_CANDIDATE_COUNTS, EVENT_COUNT)
  seed_values = [int(row["generated_token_id"]) for row in events]
  seed_ids = generated / "seed-ids.i32"
  timing_ids = generated / "timing-ids.i32"
  seed_ids.write_bytes(b"".join(
      struct.pack("<i", value) for value in seed_values))
  timing_ids.write_bytes(b"".join(
      struct.pack("<i", value) for value in TIMING_ORDINALS))

  contract = source_contract()
  module_compile = run([
      "ocloc", "compile", "-file", str(MODULE_SOURCE),
      "-device", "0xb080",
      "-output", "iq36_affine_q4_group128_component",
      "-out_dir", str(generated), "-output_no_suffix",
      "--format", "zebin", "-options", "-cl-std=CL3.0", "-q",
  ], 120)
  module = generated / "iq36_affine_q4_group128_component.bin"
  module_validate = (
      run(["ocloc", "validate", "-file", str(module)], 60)
      if module.is_file() else {
          "command": [], "returncode": 125,
          "stdout": "", "stderr": "module missing", "timed_out": False,
      })
  configure = run([
      str(args.cmake), "-S", str(ROOT / "engine"), "-B",
      str(args.build_dir), "-DCMAKE_BUILD_TYPE=Release",
  ])
  build = (
      run([
          str(args.cmake), "--build", str(args.build_dir),
          "--target", TARGET, "-j", "1",
      ], 300)
      if configure["returncode"] == 0 else {
          "command": [], "returncode": 125,
          "stdout": "", "stderr": "configure failed", "timed_out": False,
      })
  link_map = (
      run(["ldd", str(binary)], 30)
      if binary.is_file() else {
          "command": [], "returncode": 125,
          "stdout": "", "stderr": "binary missing", "timed_out": False,
      })
  runnable = (
      module_compile["returncode"] == 0 and
      module_validate["returncode"] == 0 and
      configure["returncode"] == 0 and
      build["returncode"] == 0 and
      module.is_file() and binary.is_file()
  )
  repeat = (
      launch_component(
          args, binary, output / "repeat", module, seed_ids, timing_ids)
      if runnable else {
          "returncode": 125, "timed_out": False,
          "memory_guard": {"tripped": False},
          "oom_observed": False, "result": {}, "monitor": {},
      })
  repeat_audit = audit_worker(repeat, args)
  confirm = (
      launch_component(
          args, binary, output / "confirm", module, seed_ids, timing_ids)
      if repeat["returncode"] == 0 else {
          "returncode": 125, "timed_out": False,
          "memory_guard": {"tripped": False},
          "oom_observed": False, "result": {}, "monitor": {},
      })
  confirm_audit = audit_worker(confirm, args)

  repeat_correctness = repeat_audit["correctness"]
  confirm_correctness = confirm_audit["correctness"]
  observed_max = max(
      int(repeat_correctness["maximum_candidate_count"]),
      int(confirm_correctness["maximum_candidate_count"]))
  observed_active_bytes = (
      FIXED_ACTIVE_BYTES + observed_max * PER_CANDIDATE_BYTES)
  observed_ratio = observed_active_bytes / FULL_I8_BYTES
  repeat_resources = repeat_audit["resources"]
  confirm_resources = confirm_audit["resources"]
  repeat_cpu_delta = candidate_count_delta(
      repeat_correctness["candidate_counts"], cpu_candidate_counts)
  confirm_cpu_delta = candidate_count_delta(
      confirm_correctness["candidate_counts"], cpu_candidate_counts)
  checks = [
      check(
          "repository_clean_pushed_main_at_gate",
          state["branch"] == "main" and not state["dirty"] and
          state["pushed"], git=state),
      check(
          "registered_inputs_match_exact_hashes",
          not mismatches,
          observed={
              display(path): observed_hashes[path]
              for path in observed_hashes
          }),
      check(
          "source_contract_is_exact_and_bounded",
          contract["kernels_exact"] and
          contract["group128_locked"] and
          contract["capacity_locked"] and
          contract["full_i8_fallback_present"] and
          contract["serial_abba_present"] and
          contract["target_registered_once"],
          contract=contract),
      check(
          "module_and_serial_component_build_pass",
          runnable and
          "-j" in build.get("command", []) and
          build.get("command", [])[-1] == "1" and
          "ze_loader" in link_map.get("stdout", ""),
          module_compile_returncode=module_compile["returncode"],
          module_validate_returncode=module_validate["returncode"],
          configure_returncode=configure["returncode"],
          build_returncode=build["returncode"],
          build_command=build.get("command"),
          link_map=link_map.get("stdout", "")),
      check(
          "repeat_and_confirm_workers_complete_serially",
          repeat["returncode"] == 0 and
          confirm["returncode"] == 0 and
          not repeat.get("timed_out") and not confirm.get("timed_out") and
          repeat_audit["required_checks_passed"] is True and
          confirm_audit["required_checks_passed"] is True,
          repeat_returncode=repeat["returncode"],
          confirm_returncode=confirm["returncode"]),
      check(
          "all_2000_population_tokens_and_references_are_exact",
          all(
              row["events"] == EVENT_COUNT and
              row["token_mismatch_count"] == 0 and
              row["reference_mismatch_count"] == 0
              for row in (repeat_correctness, confirm_correctness)),
          repeat=repeat_correctness,
          confirm=confirm_correctness),
      check(
          "gpu_upper_bound_has_zero_496m64_pair_violations",
          repeat_correctness["bound_violation_count"] == 0 and
          confirm_correctness["bound_violation_count"] == 0,
          evaluated_pairs_per_worker=ROWS * EVENT_COUNT,
          repeat_violations=repeat_correctness["bound_violation_count"],
          confirm_violations=confirm_correctness["bound_violation_count"]),
      check(
          "gpu_native_candidate_counts_are_reproducible_and_fit_capacity",
          repeat_correctness["candidate_counts"] ==
              confirm_correctness["candidate_counts"] and
          repeat_correctness["overflow_count"] == 0 and
          confirm_correctness["overflow_count"] == 0 and
          observed_max <= CAPACITY,
          capacity=CAPACITY,
          observed_maximum=observed_max,
          cpu_certificate_maximum=max(cpu_candidate_counts),
          repeat_vs_cpu_certificate=repeat_cpu_delta,
          confirm_vs_cpu_certificate=confirm_cpu_delta,
          note=(
              "CPU and GPU reductions need not select identical conservative "
              "upper bounds; zero GPU bound violations, reproducibility, "
              "capacity, and measured traffic are the gate invariants")),
      check(
          "forced_capacity_overflow_selects_full_i8_fallback",
          all(
              row.get("candidate_count") == ROWS and
              row.get("fallback_selected") is True and
              row.get("candidate_token") == row.get("reference_token")
              for row in (
                  repeat_audit["forced_overflow"],
                  confirm_audit["forced_overflow"])),
          repeat=repeat_audit["forced_overflow"],
          confirm=confirm_audit["forced_overflow"]),
      check(
          "observed_worst_active_traffic_clears_0p60_cap",
          observed_ratio <= TRAFFIC_CAP,
          observed_maximum_candidate_rows=observed_max,
          maximum_active_bytes=observed_active_bytes,
          full_i8_bytes=FULL_I8_BYTES,
          maximum_ratio=observed_ratio,
          cap=TRAFFIC_CAP),
      check(
          "paired_abba_sample_geometry_and_arithmetic_are_exact",
          repeat_audit["timing"]["sample_geometry_exact"] and
          confirm_audit["timing"]["sample_geometry_exact"] and
          repeat_audit["timing"]["sample_count"] ==
              len(TIMING_ORDINALS) * args.blocks and
          confirm_audit["timing"]["sample_count"] ==
              len(TIMING_ORDINALS) * args.blocks,
          repeat_samples=repeat_audit["timing"]["sample_count"],
          confirm_samples=confirm_audit["timing"]["sample_count"],
          timing_ordinals=list(TIMING_ORDINALS),
          blocks=args.blocks),
      check(
          "paired_one_sided_95pct_wall_saving_lcb_clears_product_remainder",
          repeat_audit["timing"]["wall_saving_lcb_us"] >=
              REQUIRED_FALLBACK_SAVING_US and
          confirm_audit["timing"]["wall_saving_lcb_us"] >=
              REQUIRED_FALLBACK_SAVING_US,
          method="paired_one_sided_percentile_bootstrap_median",
          bootstrap_resamples=BOOTSTRAP_RESAMPLES,
          bootstrap_seed=BOOTSTRAP_SEED,
          required_fallback_saving_us=REQUIRED_FALLBACK_SAVING_US,
          repeat_lcb_us=repeat_audit["timing"]["wall_saving_lcb_us"],
          confirm_lcb_us=confirm_audit["timing"]["wall_saving_lcb_us"]),
      check(
          "selection_and_exact_kernels_are_spill_free",
          all(
              int(resources.get(name, {}).get("spill_mem_bytes", -1)) == 0
              for resources in (repeat_resources, confirm_resources)
              for name in ("select", "exact")),
          repeat=repeat_resources,
          confirm=confirm_resources),
      check(
          "memory_guards_hold_without_oom",
          not repeat.get("oom_observed") and
          not confirm.get("oom_observed") and
          not repeat.get("memory_guard", {}).get("tripped") and
          not confirm.get("memory_guard", {}).get("tripped"),
          repeat=repeat.get("monitor"),
          confirm=confirm.get("monitor"),
          repeat_oom=repeat.get("oom_observed"),
          confirm_oom=confirm.get("oom_observed")),
      check(
          "no_openvino_model_or_product_worker_ran",
          not concurrent,
          model_workers=0, product_workers=0,
          component_workers=2, workers_concurrent=False),
  ]
  required = all(row["pass"] for row in checks)
  verdict = (
      "admit_affine_q4_group128_isolated_product_integration"
      if required else
      "reject_affine_q4_group128_gpu_component")
  result = {
      "schema": SCHEMA,
      "workstream": WS,
      "created_at": datetime.now(timezone.utc).isoformat(),
      "git": state,
      "verdict": verdict,
      "required_checks_passed": required,
      "component_implementation_admitted": required,
      "isolated_product_integration_admitted": required,
      "product_promotion_admitted": False,
      "performance_claim_admitted": False,
      "checks": checks,
      "inputs": {
          "registered_sha256": {
              display(path): observed_hashes[path]
              for path in observed_hashes
          },
          "source_sha256": {
              display(MODULE_SOURCE): sha256(MODULE_SOURCE),
              display(CPP_SOURCE): sha256(CPP_SOURCE),
              display(BOUNDARIES): sha256(BOUNDARIES),
          },
          "generated_sha256": {
              display(seed_ids): sha256(seed_ids),
              display(timing_ids): sha256(timing_ids),
              display(module): sha256(module) if module.is_file() else None,
              display(binary): sha256(binary) if binary.is_file() else None,
          },
          "timing_ordinals": list(TIMING_ORDINALS),
      },
      "build": {
          "module_compile": module_compile,
          "module_validate": module_validate,
          "configure": configure,
          "component": build,
          "link_map": link_map,
      },
      "repeat": repeat,
      "confirm": confirm,
      "audit": {
          "repeat": repeat_audit,
          "confirm": confirm_audit,
          "candidate_counts_vs_cpu_certificate": {
              "repeat": repeat_cpu_delta,
              "confirm": confirm_cpu_delta,
          },
      },
      "traffic": {
          "fixed_active_bytes": FIXED_ACTIVE_BYTES,
          "per_candidate_bytes": PER_CANDIDATE_BYTES,
          "observed_maximum_candidate_rows": observed_max,
          "maximum_active_bytes": observed_active_bytes,
          "full_i8_bytes": FULL_I8_BYTES,
          "maximum_ratio": observed_ratio,
          "cap": TRAFFIC_CAP,
      },
      "inference": {
          "method": "paired_one_sided_percentile_bootstrap_median",
          "required_product_saving_us": REQUIRED_PRODUCT_SAVING_US,
          "observed_fallback_rate": FALLBACK_RATE,
          "required_fallback_saving_us": REQUIRED_FALLBACK_SAVING_US,
          "repeat_wall_saving_lcb_us":
              repeat_audit["timing"]["wall_saving_lcb_us"],
          "confirm_wall_saving_lcb_us":
              confirm_audit["timing"]["wall_saving_lcb_us"],
          "repeat_kernel_saving_lcb_us":
              repeat_audit["timing"]["kernel_saving_lcb_us"],
          "confirm_kernel_saving_lcb_us":
              confirm_audit["timing"]["kernel_saving_lcb_us"],
      },
      "workers": {
          "component_workers": 2,
          "model_workers": 0,
          "product_workers": 0,
          "workers_concurrent": False,
      },
      "next_gate": {
          "route": (
              "isolated_affine_q4_group128_product_source_and_compile_gate"
              if required else "close_affine_q4_group128_route"),
          "requirements": [
              "bind only the count25 exact fallback branch",
              "prepack one shared affine-Q4/group128 LM-head tensor",
              "retain full-I8 fallback on capacity overflow",
              "preserve all accepted carrier providers and state",
              "run source and compile census before first inference",
              "make no product speed claim before exact correctness and ABBA",
          ],
      },
  }
  write_json(output / "result.json", result)
  write_json(output / "manifest.json", {
      "schema": SCHEMA,
      "tool": display(Path(__file__)),
      "tool_sha256": sha256(Path(__file__)),
      "git": state,
      "inputs": result["inputs"],
      "workers": result["workers"],
  })
  summary = f"""# Affine-Q4/group128 GPU component

Verdict: **{verdict}**. Required checks: `{str(required).lower()}`.

Both workers recover `2000/2000` fixed-population tokens with zero GPU upper
bound violations across `496,640,000` pairs per worker. The observed maximum
is `{observed_max}` candidates, producing `{observed_active_bytes}` active
bytes or `{observed_ratio:.6f}x` full I8. Forced overflow selects the full-I8
fallback. GPU-native reduction rounding differs from the CPU certificate on
`{repeat_cpu_delta['different_event_count']}/2000` count rows, with maximum
absolute delta `{repeat_cpu_delta['maximum_absolute_event_delta']}` and
maximum counts `{observed_max}` / `{max(cpu_candidate_counts)}`; repeat and
confirm GPU counts are identical.

Repeat/confirm one-sided 95% wall-saving LCBs are
`{repeat_audit['timing']['wall_saving_lcb_us']:.3f}` /
`{confirm_audit['timing']['wall_saving_lcb_us']:.3f} us` per fallback versus
the `{REQUIRED_FALLBACK_SAVING_US:.3f}-us` product remainder. This admits only
isolated product source/compile work, not a product speedup claim.
"""
  (output / "summary.md").write_text(summary, encoding="utf-8")
  print(json.dumps({
      "artifact": display(output),
      "verdict": verdict,
      "required_checks_passed": required,
      "maximum_candidate_rows": observed_max,
      "maximum_active_ratio": observed_ratio,
      "repeat_wall_saving_lcb_us":
          repeat_audit["timing"]["wall_saving_lcb_us"],
      "confirm_wall_saving_lcb_us":
          confirm_audit["timing"]["wall_saving_lcb_us"],
      "repeat_peak_rss_bytes":
          repeat.get("monitor", {}).get("process_rss_peak_bytes"),
      "confirm_peak_rss_bytes":
          confirm.get("monitor", {}).get("process_rss_peak_bytes"),
  }, separators=(",", ":")), flush=True)
  return 0 if required else 2


if __name__ == "__main__":
  raise SystemExit(main())
