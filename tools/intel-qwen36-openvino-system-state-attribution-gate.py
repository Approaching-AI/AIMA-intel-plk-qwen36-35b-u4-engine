#!/usr/bin/env python3
"""Attribute repeat-to-repeat 128k movement to read-only system state.

The gate clones one completed candidate timing worker, shortens it to a
bounded token prefix, and runs exactly two copies serially.  It changes no
runtime, graph, plugin, kernel, queue, frequency, power, or thermal setting.
While each worker is alive it samples Xe GT frequency, VM page activity,
cgroup pressure, descendant memory/fault counters, and CPU thermal state.
The result is attribution telemetry, not a speed claim.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import statistics
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "intel-qwen36-openvino-system-state-attribution-v3"
PRODUCT_TOOL = ROOT / "tools/intel-qwen36-openvino-hot-cold-product-gate.py"
DEFAULT_SOURCE = ROOT / (
    "output/openvino-exact-attention-dual-cohort-long-nonsentinel-abba1-"
    "20260724Tseq2163-clean/raw/filler_128k/block00/candidate-b1")
VMSTAT_KEYS = (
    "pgpgin", "pgpgout", "pswpin", "pswpout", "pgfault", "pgmajfault",
    "workingset_refault_anon", "workingset_refault_file",
    "workingset_activate_anon", "workingset_activate_file",
    "allocstall_dma", "allocstall_dma32", "allocstall_normal",
    "allocstall_movable", "allocstall_device", "compact_stall",
)
MEMINFO_KEYS = (
    "MemAvailable", "SwapTotal", "SwapFree", "Dirty", "Writeback",
    "AnonPages", "Mapped", "Shmem", "Slab", "SReclaimable", "SUnreclaim",
)
CGROUP_MEMORY_COUNTER_KEYS = (
    "workingset_refault_anon", "workingset_refault_file",
    "workingset_activate_anon", "workingset_activate_file",
    "pgscan", "pgsteal", "pswpin", "pswpout", "pgscan_kswapd",
    "pgscan_direct", "pgfault", "pgmajfault", "pgrefill", "pgactivate",
    "swpin_zero", "swpout_zero",
)


def load_module(name: str, path: Path) -> Any:
  spec = importlib.util.spec_from_file_location(name, path)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import {path}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


PRODUCT = load_module("iq36_gt_frequency_product", PRODUCT_TOOL)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out-dir", type=Path, required=True)
  parser.add_argument("--source-worker-dir", type=Path, default=DEFAULT_SOURCE)
  parser.add_argument("--output-tokens", type=int, default=128)
  parser.add_argument("--sample-interval-s", type=float, default=0.25)
  parser.add_argument("--timeout-s", type=int, default=1800)
  parser.add_argument("--poll-interval-s", type=float, default=0.5)
  parser.add_argument("--min-available-gib", type=float, default=8.0)
  parser.add_argument("--abort-below-available-gib", type=float, default=4.0)
  parser.add_argument(
      "--worker-transient-scope", action="store_true",
      help=("place each of the two workers in its own fresh transient scope; "
            "no resource limit or runtime setting is changed"))
  parser.add_argument("--plan-only", action="store_true")
  args = parser.parse_args()
  if args.output_tokens < 16:
    parser.error("output-tokens must be at least sixteen")
  if args.sample_interval_s <= 0 or args.poll_interval_s <= 0:
    parser.error("sample and poll intervals must be positive")
  if args.timeout_s <= 0:
    parser.error("timeout must be positive")
  if (args.abort_below_available_gib < 0 or args.min_available_gib < 0 or
      args.abort_below_available_gib > args.min_available_gib):
    parser.error("invalid memory thresholds")
  return args


def load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise TypeError(f"expected JSON object: {path}")
  return value


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git_state() -> dict[str, Any]:
  commit = subprocess.run(
      ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
      capture_output=True, check=True).stdout.strip()
  status = subprocess.run(
      ["git", "status", "--porcelain"], cwd=ROOT, text=True,
      capture_output=True, check=True).stdout.splitlines()
  return {"commit": commit, "dirty": bool(status), "status": status}


def other_product_workers() -> list[dict[str, Any]]:
  rows = []
  for proc in Path("/proc").iterdir():
    if not proc.name.isdigit() or int(proc.name) == os.getpid():
      continue
    try:
      command = (proc / "cmdline").read_bytes().replace(
          b"\0", b" ").decode("utf-8", errors="replace")
    except (FileNotFoundError, PermissionError, ProcessLookupError):
      continue
    if PRODUCT_TOOL.name in command and "--worker-config" in command:
      rows.append({"pid": int(proc.name), "command": command.strip()})
  return rows


def find_frequency_root() -> Path:
  candidates = sorted(Path("/sys/class/drm").glob(
      "card*/device/tile0/gt0/freq0"))
  exact = [
      path for path in candidates
      if (path / "act_freq").is_file() and
      (path / "throttle/status").is_file()]
  if len(exact) != 1:
    raise RuntimeError(f"expected one Xe GT frequency root, observed {exact}")
  return exact[0].resolve()


def read_int(path: Path) -> int | None:
  try:
    return int(path.read_text(encoding="utf-8").strip())
  except (FileNotFoundError, PermissionError, ValueError):
    return None


def read_named_ints(path: Path) -> dict[str, int]:
  rows: dict[str, int] = {}
  try:
    lines = path.read_text(encoding="utf-8").splitlines()
  except (FileNotFoundError, PermissionError):
    return rows
  for line in lines:
    fields = line.split()
    if len(fields) < 2:
      continue
    try:
      rows[fields[0].rstrip(":")] = int(fields[1])
    except ValueError:
      continue
  return rows


def read_meminfo_bytes() -> dict[str, int]:
  raw = read_named_ints(Path("/proc/meminfo"))
  return {
      key: raw[key] * 1024
      for key in MEMINFO_KEYS
      if key in raw
  }


def read_pressure(path: Path) -> dict[str, dict[str, float | int]]:
  rows: dict[str, dict[str, float | int]] = {}
  try:
    lines = path.read_text(encoding="utf-8").splitlines()
  except (FileNotFoundError, PermissionError):
    return rows
  for line in lines:
    fields = line.split()
    if not fields:
      continue
    values: dict[str, float | int] = {}
    for field in fields[1:]:
      if "=" not in field:
        continue
      key, raw = field.split("=", 1)
      try:
        values[key] = int(raw) if key == "total" else float(raw)
      except ValueError:
        continue
    rows[fields[0]] = values
  return rows


def find_cgroup_root() -> Path:
  for line in Path("/proc/self/cgroup").read_text(
      encoding="utf-8").splitlines():
    if line.startswith("0::"):
      root = Path("/sys/fs/cgroup") / line[3:].lstrip("/")
      if root.is_dir():
        return root.resolve()
  raise RuntimeError("current cgroup-v2 root is unavailable")


def find_coretemp_root() -> Path | None:
  for path in sorted(Path("/sys/class/hwmon").glob("hwmon*")):
    try:
      if (path / "name").read_text(encoding="utf-8").strip() == "coretemp":
        return path.resolve()
    except (FileNotFoundError, PermissionError):
      continue
  return None


def proc_rows() -> dict[int, dict[str, int]]:
  rows: dict[int, dict[str, int]] = {}
  for proc in Path("/proc").iterdir():
    if not proc.name.isdigit():
      continue
    try:
      raw = (proc / "stat").read_text(encoding="utf-8")
      fields = raw[raw.rfind(")") + 2:].split()
      if len(fields) < 10:
        continue
      rows[int(proc.name)] = {
          "ppid": int(fields[1]),
          "minor_faults": int(fields[7]),
          "major_faults": int(fields[9]),
      }
    except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
      continue
  return rows


def sample_descendants() -> dict[str, Any]:
  rows = proc_rows()
  parents = {os.getpid()}
  descendants: set[int] = set()
  while True:
    added = {
        pid for pid, row in rows.items()
        if row["ppid"] in parents and pid not in descendants
    }
    if not added:
      break
    descendants.update(added)
    parents.update(added)
  total_rss = 0
  total_swap = 0
  total_read = 0
  total_write = 0
  present = []
  minor_faults = 0
  major_faults = 0
  for pid in sorted(descendants):
    proc = Path("/proc") / str(pid)
    status = read_named_ints(proc / "status")
    io = read_named_ints(proc / "io")
    if not status and not io and pid not in rows:
      continue
    present.append(pid)
    total_rss += status.get("VmRSS", 0) * 1024
    total_swap += status.get("VmSwap", 0) * 1024
    total_read += io.get("read_bytes", 0)
    total_write += io.get("write_bytes", 0)
    minor_faults += rows.get(pid, {}).get("minor_faults", 0)
    major_faults += rows.get(pid, {}).get("major_faults", 0)
  return {
      "count": len(present),
      "pids": present,
      "rss_bytes": total_rss,
      "swap_bytes": total_swap,
      "minor_faults": minor_faults,
      "major_faults": major_faults,
      "io_read_bytes": total_read,
      "io_write_bytes": total_write,
  }


def sample_cpu(coretemp_root: Path | None) -> dict[str, Any]:
  frequencies = [
      value for value in (
          read_int(path) for path in sorted(Path(
              "/sys/devices/system/cpu/cpufreq").glob(
                  "policy*/scaling_cur_freq")))
      if value is not None
  ]
  temperatures = []
  if coretemp_root is not None:
    temperatures = [
        value for value in (
            read_int(path) for path in sorted(coretemp_root.glob("temp*_input")))
        if value is not None
    ]
  throttle_root = Path(
      "/sys/devices/system/cpu/cpu0/thermal_throttle")
  return {
      "frequency_khz": {
          "min": min(frequencies) if frequencies else None,
          "median": statistics.median(frequencies) if frequencies else None,
          "max": max(frequencies) if frequencies else None,
      },
      "coretemp_millic": {
          "min": min(temperatures) if temperatures else None,
          "max": max(temperatures) if temperatures else None,
      },
      "thermal_throttle": {
          path.name: read_int(path)
          for path in sorted(throttle_root.glob("*"))
          if path.is_file()
      },
  }


def sample_system_state(
    frequency_root: Path,
    cgroup_root: Path,
    coretemp_root: Path | None,
    repeat: int,
    started: float,
) -> dict[str, Any]:
  throttle_root = frequency_root / "throttle"
  meminfo = read_meminfo_bytes()
  vmstat = read_named_ints(Path("/proc/vmstat"))
  return {
      "repeat": repeat,
      "elapsed_seconds": time.monotonic() - started,
      "timestamp_utc": datetime.now(timezone.utc).isoformat(),
      "act_freq_mhz": read_int(frequency_root / "act_freq"),
      "cur_freq_mhz": read_int(frequency_root / "cur_freq"),
      "min_freq_mhz": read_int(frequency_root / "min_freq"),
      "max_freq_mhz": read_int(frequency_root / "max_freq"),
      "throttle_status": read_int(throttle_root / "status"),
      "throttle_reasons": {
          path.name.removeprefix("reason_"): read_int(path)
          for path in sorted(throttle_root.glob("reason_*"))
      },
      "vmstat": {key: vmstat.get(key) for key in VMSTAT_KEYS},
      "meminfo_bytes": meminfo,
      "global_pressure": {
          name: read_pressure(Path("/proc/pressure") / name)
          for name in ("memory", "io", "cpu")
      },
      "cgroup": {
          "memory_current_bytes": read_int(cgroup_root / "memory.current"),
          "memory_swap_current_bytes": read_int(
              cgroup_root / "memory.swap.current"),
          "memory_events": read_named_ints(cgroup_root / "memory.events"),
          "memory_stat": read_named_ints(cgroup_root / "memory.stat"),
          "memory_pressure": read_pressure(cgroup_root / "memory.pressure"),
          "io_pressure": read_pressure(cgroup_root / "io.pressure"),
          "cpu_pressure": read_pressure(cgroup_root / "cpu.pressure"),
      },
      "descendants": sample_descendants(),
      "cpu": sample_cpu(coretemp_root),
  }


def percentile(values: list[float], probability: float) -> float | None:
  if not values:
    return None
  ordered = sorted(values)
  position = probability * (len(ordered) - 1)
  lower = math.floor(position)
  upper = math.ceil(position)
  if lower == upper:
    return ordered[lower]
  weight = position - lower
  return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_frequency(samples: list[dict[str, Any]]) -> dict[str, Any]:
  active = [
      float(row["act_freq_mhz"]) for row in samples
      if isinstance(row.get("act_freq_mhz"), int) and
      int(row["act_freq_mhz"]) > 0]
  reasons: dict[str, int] = {}
  for row in samples:
    for name, value in row.get("throttle_reasons", {}).items():
      reasons[name] = reasons.get(name, 0) + int(value == 1)
  return {
      "sample_count": len(samples),
      "active_sample_count": len(active),
      "active_freq_mhz": {
          "min": min(active) if active else None,
          "p10": percentile(active, 0.10),
          "median": statistics.median(active) if active else None,
          "p90": percentile(active, 0.90),
          "max": max(active) if active else None,
      },
      "throttle_status_sample_count": sum(
          row.get("throttle_status") == 1 for row in samples),
      "throttle_reason_sample_counts": reasons,
  }


def nested(row: dict[str, Any], *path: str) -> Any:
  value: Any = row
  for key in path:
    if not isinstance(value, dict):
      return None
    value = value.get(key)
  return value


def numeric_values(
    samples: list[dict[str, Any]], *path: str,
) -> list[float]:
  return [
      float(value) for row in samples
      if finite(value := nested(row, *path))
  ]


def positive_counter_delta(
    samples: list[dict[str, Any]], *path: str,
) -> float | None:
  values = [
      (float(row["elapsed_seconds"]), float(value))
      for row in samples
      if finite(value := nested(row, *path))
  ]
  if len(values) < 2:
    return None
  total = 0.0
  for (_, before), (_, after) in zip(values, values[1:]):
    total += max(0.0, after - before)
  return total


def positive_counter_peak_rate(
    samples: list[dict[str, Any]], *path: str,
) -> float | None:
  values = [
      (float(row["elapsed_seconds"]), float(value))
      for row in samples
      if finite(value := nested(row, *path))
  ]
  rates = []
  for (before_t, before), (after_t, after) in zip(values, values[1:]):
    elapsed = after_t - before_t
    if elapsed > 0 and after >= before:
      rates.append((after - before) / elapsed)
  return max(rates) if rates else None


def maximum(
    samples: list[dict[str, Any]], *path: str,
) -> float | None:
  values = numeric_values(samples, *path)
  return max(values) if values else None


def minimum(
    samples: list[dict[str, Any]], *path: str,
) -> float | None:
  values = numeric_values(samples, *path)
  return min(values) if values else None


def summarize_system_state(samples: list[dict[str, Any]]) -> dict[str, Any]:
  vmstat_delta = {
      key: positive_counter_delta(samples, "vmstat", key)
      for key in VMSTAT_KEYS
  }
  vmstat_peak_rate = {
      key: positive_counter_peak_rate(samples, "vmstat", key)
      for key in ("pswpin", "pswpout", "pgmajfault", "pgpgin", "pgpgout")
  }
  cgroup_memory_counter_delta = {
      key: positive_counter_delta(
          samples, "cgroup", "memory_stat", key)
      for key in CGROUP_MEMORY_COUNTER_KEYS
  }
  cgroup_memory_counter_peak_rate = {
      key: positive_counter_peak_rate(
          samples, "cgroup", "memory_stat", key)
      for key in ("pswpin", "pswpout", "pgscan", "pgmajfault")
  }
  swap_total = maximum(samples, "meminfo_bytes", "SwapTotal")
  swap_free_min = minimum(samples, "meminfo_bytes", "SwapFree")
  system_swap_used_peak = (
      swap_total - swap_free_min
      if swap_total is not None and swap_free_min is not None else None)
  return {
      "sample_count": len(samples),
      "elapsed_seconds": (
          float(samples[-1]["elapsed_seconds"]) if samples else None),
      "mem_available_min_bytes": minimum(
          samples, "meminfo_bytes", "MemAvailable"),
      "system_swap_used_peak_bytes": system_swap_used_peak,
      "vmstat_delta": vmstat_delta,
      "vmstat_peak_rate_per_s": vmstat_peak_rate,
      "global_pressure_delta_seconds": {
          f"{resource}_{level}": (
              value / 1e6 if value is not None else None)
          for resource in ("memory", "io", "cpu")
          for level in ("some", "full")
          if (value := positive_counter_delta(
              samples, "global_pressure", resource, level, "total")) is not None
      },
      "cgroup": {
          "memory_peak_bytes": maximum(
              samples, "cgroup", "memory_current_bytes"),
          "swap_peak_bytes": maximum(
              samples, "cgroup", "memory_swap_current_bytes"),
          "memory_events_delta": {
              key: positive_counter_delta(
                  samples, "cgroup", "memory_events", key)
              for key in ("low", "high", "max", "oom", "oom_kill",
                          "oom_group_kill")
          },
          "memory_counter_delta": cgroup_memory_counter_delta,
          "memory_counter_peak_rate_per_s":
              cgroup_memory_counter_peak_rate,
          "pressure_delta_seconds": {
              f"{resource}_{level}": (
                  value / 1e6 if value is not None else None)
              for resource in ("memory", "io", "cpu")
              for level in ("some", "full")
              if (value := positive_counter_delta(
                  samples, "cgroup", f"{resource}_pressure",
                  level, "total")) is not None
          },
      },
      "descendants": {
          "count_peak": maximum(samples, "descendants", "count"),
          "rss_peak_bytes": maximum(samples, "descendants", "rss_bytes"),
          "swap_peak_bytes": maximum(samples, "descendants", "swap_bytes"),
          "minor_fault_positive_delta": positive_counter_delta(
              samples, "descendants", "minor_faults"),
          "major_fault_positive_delta": positive_counter_delta(
              samples, "descendants", "major_faults"),
          "io_read_positive_delta_bytes": positive_counter_delta(
              samples, "descendants", "io_read_bytes"),
          "io_write_positive_delta_bytes": positive_counter_delta(
              samples, "descendants", "io_write_bytes"),
      },
      "cpu": {
          "frequency_min_khz": minimum(
              samples, "cpu", "frequency_khz", "min"),
          "frequency_median_of_policy_medians_khz": (
              statistics.median(values)
              if (values := numeric_values(
                  samples, "cpu", "frequency_khz", "median")) else None),
          "frequency_max_khz": maximum(
              samples, "cpu", "frequency_khz", "max"),
          "coretemp_max_millic": maximum(
              samples, "cpu", "coretemp_millic", "max"),
          "package_throttle_count_delta": positive_counter_delta(
              samples, "cpu", "thermal_throttle",
              "package_throttle_count"),
          "package_throttle_total_time_ms_delta": positive_counter_delta(
              samples, "cpu", "thermal_throttle",
              "package_throttle_total_time_ms"),
      },
  }


def check(name: str, passed: bool, **details: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **details}


def finite(value: Any) -> bool:
  return isinstance(value, (int, float)) and math.isfinite(float(value))


def ratio(numerator: Any, denominator: Any) -> float | None:
  if not finite(numerator) or not finite(denominator) or denominator == 0:
    return None
  return float(numerator) / float(denominator)


def normalized_worker_config(path: Path) -> dict[str, Any]:
  value = load_json(path)
  value.pop("raw", None)
  value.pop("result", None)
  return value


def main() -> int:
  args = parse_args()
  out_dir = args.out_dir.resolve()
  source_dir = args.source_worker_dir.resolve()
  source_config_path = source_dir / "worker-config.json"
  source_result_path = source_dir / "worker-result.json"
  source_run_path = source_dir / "run.json"
  for path in (source_config_path, source_result_path, source_run_path):
    if not path.is_file():
      raise SystemExit(f"missing source worker evidence: {path}")
  source_config = load_json(source_config_path)
  source_result = load_json(source_result_path)
  source_run = load_json(source_run_path)
  plugin = Path(str(source_config["candidate_gpu_plugin"])).resolve()
  if not plugin.is_file():
    raise SystemExit(f"missing candidate plugin: {plugin}")
  expected_tokens = [
      int(value) for value in source_result["generated_token_ids"][
          :args.output_tokens]]
  if len(expected_tokens) != args.output_tokens:
    raise SystemExit("source worker does not cover requested output tokens")
  frequency_root = find_frequency_root()
  orchestrator_cgroup_root = find_cgroup_root()
  coretemp_root = find_coretemp_root()

  config = dict(source_config)
  config.update({
      "capture_execution_census": False,
      "capture_lm_head_hidden": False,
      "capture_logits": False,
      "capture_prefill_profiles": False,
      "checkpoint_steps": list(range(args.output_tokens)),
      "host_time_profiling": 0,
      "output_tokens": args.output_tokens,
      "reference_result": str((out_dir / "teacher-reference.json").resolve()),
  })
  plan = {
      "source_worker": str(source_dir),
      "out_dir": str(out_dir),
      "candidate_gpu_plugin": str(plugin),
      "candidate_gpu_plugin_sha256": PRODUCT.sha256_file(plugin),
      "case_id": config.get("case_id"),
      "bucket": config.get("bucket"),
      "purpose": config.get("purpose"),
      "output_tokens": args.output_tokens,
      "expected_token_sha256": hashlib.sha256(
          json.dumps(expected_tokens, separators=(",", ":")).encode(
              "utf-8")).hexdigest(),
      "frequency_root": str(frequency_root),
      "orchestrator_cgroup_root": str(orchestrator_cgroup_root),
      "worker_cgroup_mode": (
          "fresh_transient_scope_per_worker"
          if args.worker_transient_scope else "orchestrator_scope"),
      "coretemp_root": str(coretemp_root) if coretemp_root else None,
      "telemetry": [
          "xe_gt_frequency_and_throttle",
          "proc_vmstat_meminfo_and_psi",
          "cgroup_memory_swap_events_stat_and_psi",
          "descendant_rss_swap_fault_and_io",
          "cpu_frequency_coretemp_and_thermal_throttle",
      ],
      "root_only_telemetry_omitted": [
          "/sys/devices/system/cpu/intel_uncore_frequency/*/current_freq_khz",
          "/sys/devices/virtual/powercap/*/energy_uj",
      ],
      "sample_interval_s": args.sample_interval_s,
      "worker_count": 2,
      "stock_worker_count": 0,
      "workers_are_identical_and_serial": True,
      "worker_transient_scope": args.worker_transient_scope,
      "worker_scope_resource_limits_changed": False,
      "runtime_or_power_settings_changed": False,
      "memory_preflight_gib": args.min_available_gib,
      "memory_abort_gib": args.abort_below_available_gib,
  }
  if args.plan_only:
    print(json.dumps(plan, indent=2, sort_keys=True))
    return 0

  out_dir.mkdir(parents=True, exist_ok=False)
  write_json(out_dir / "teacher-reference.json", {
      "generated_token_ids": expected_tokens})
  git = git_state()
  concurrent = other_product_workers()
  if concurrent:
    raise RuntimeError(f"concurrent product worker detected: {concurrent}")

  worker_args = SimpleNamespace(
      abort_below_available_gib=args.abort_below_available_gib,
      candidate_gpu_plugin=plugin,
      candidate_impls_cache_capacity=source_config.get(
          "candidate_impls_cache_capacity"),
      custom_config=Path(str(source_config["custom_config"])).resolve(),
      device=str(source_config.get("device", "GPU")),
      min_available_gib=args.min_available_gib,
      model_dir=Path(str(source_config["model_dir"])).resolve(),
      openvino_python=PRODUCT.OV_PYTHON,
      pack_gdn_state=bool(source_config.get("pack_gdn_state", False)),
      poll_interval_s=args.poll_interval_s,
      prime_candidate_exact_decode_shape=bool(
          source_config.get("prime_candidate_exact_decode_shape", False)),
      resume=False,
      timeout_s=args.timeout_s,
      worker_transient_scope=args.worker_transient_scope,
  )
  workers = []
  all_samples = []
  summaries = []
  trace_path = out_dir / "system-state.jsonl"
  for repeat in range(2):
    worker_dir = out_dir / "raw" / f"repeat{repeat}"
    telemetry_cgroup_root = (
        PRODUCT.worker_scope_cgroup_root(worker_dir)
        if args.worker_transient_scope else orchestrator_cgroup_root)
    box: dict[str, Any] = {}
    failure: list[BaseException] = []

    def launch() -> None:
      try:
        box["worker"] = PRODUCT.run_worker(worker_args, worker_dir, config)
      except BaseException as exc:  # Preserve worker-thread failure.
        failure.append(exc)

    started = time.monotonic()
    thread = threading.Thread(target=launch, daemon=False)
    thread.start()
    repeat_samples = []
    with trace_path.open("a", encoding="utf-8") as trace:
      while thread.is_alive():
        row = sample_system_state(
            frequency_root, telemetry_cgroup_root, coretemp_root,
            repeat, started)
        repeat_samples.append(row)
        trace.write(json.dumps(row, sort_keys=True) + "\n")
        trace.flush()
        thread.join(timeout=args.sample_interval_s)
    thread.join()
    if failure:
      raise failure[0]
    worker = box["worker"]
    workers.append(worker)
    all_samples.extend(repeat_samples)
    result = worker.get("result") or {}
    summaries.append({
        "repeat": repeat,
        "prefill_tokens_s": result.get("prefill_tokens_s"),
        "decode_tokens_s": result.get("decode_tokens_s"),
        "tpot_ms": result.get("tpot_ms"),
        "decode_first_ms": (
            result.get("decode_wall_ms", [None])[0]
            if result.get("decode_wall_ms") else None),
        "decode_post_first_median_ms": (
            statistics.median(result["decode_wall_ms"][1:])
            if len(result.get("decode_wall_ms", [])) > 1 else None),
        "generated_token_ids_sha256": result.get(
            "generated_token_ids_sha256"),
        "frequency": summarize_frequency(repeat_samples),
        "system_state": summarize_system_state(repeat_samples),
    })

  performance_ratio = {
      "repeat1_over_repeat0_prefill": ratio(
          summaries[1]["prefill_tokens_s"], summaries[0]["prefill_tokens_s"]),
      "repeat1_over_repeat0_decode": ratio(
          summaries[1]["decode_tokens_s"], summaries[0]["decode_tokens_s"]),
      "repeat1_over_repeat0_active_frequency_median": ratio(
          summaries[1]["frequency"]["active_freq_mhz"]["median"],
          summaries[0]["frequency"]["active_freq_mhz"]["median"]),
      "repeat1_over_repeat0_cgroup_pswpin": ratio(
          summaries[1]["system_state"]["cgroup"]["memory_counter_delta"][
              "pswpin"],
          summaries[0]["system_state"]["cgroup"]["memory_counter_delta"][
              "pswpin"]),
      "repeat1_over_repeat0_cgroup_pgmajfault": ratio(
          summaries[1]["system_state"]["cgroup"]["memory_counter_delta"][
              "pgmajfault"],
          summaries[0]["system_state"]["cgroup"]["memory_counter_delta"][
              "pgmajfault"]),
      "repeat1_over_repeat0_cgroup_memory_full_pressure_seconds": ratio(
          summaries[1]["system_state"]["cgroup"][
              "pressure_delta_seconds"].get("memory_full"),
          summaries[0]["system_state"]["cgroup"][
              "pressure_delta_seconds"].get("memory_full")),
  }
  worker_pass = all(
      worker.get("returncode") == 0 and
      not worker.get("timed_out") and
      not worker.get("oom_observed") and
      not (worker.get("memory_guard") or {}).get("tripped")
      for worker in workers)
  token_pass = all(
      (worker.get("result") or {}).get("teacher_forced_from_stock") is True and
      (worker.get("result") or {}).get("generated_token_ids") == expected_tokens
      for worker in workers)
  plugin_pass = all(
      (worker.get("result") or {}).get("candidate_gpu_plugin_sha256") ==
      source_result.get("candidate_gpu_plugin_sha256")
      for worker in workers)
  provider_pass = all(
      (worker.get("result") or {}).get("lm_head_i8q1_gated_exact") is False and
      (worker.get("result") or {}).get("lm_head_i8q1_greedy_local2") is True and
      (worker.get("result") or {}).get("lm_head_token_only_feedback") is True and
      (worker.get("result") or {}).get("timing_token_output") is True
      for worker in workers)
  config_pass = (
      normalized_worker_config(
          out_dir / "raw/repeat0/worker-config.json") ==
      normalized_worker_config(
          out_dir / "raw/repeat1/worker-config.json"))
  scope_pass = all(
      bool((worker.get("worker_transient_scope") or {}).get("enabled")) ==
          args.worker_transient_scope and
      (not args.worker_transient_scope or
       (worker.get("worker_transient_scope") or {}).get("cgroup_root") ==
       str(PRODUCT.worker_scope_cgroup_root(
           out_dir / "raw" / f"repeat{repeat}")))
      for repeat, worker in enumerate(workers))
  frequency_telemetry_pass = all(
      summary["frequency"]["active_sample_count"] >= 4
      for summary in summaries)
  system_telemetry_pass = all(
      summary["system_state"]["sample_count"] >= 4 and
      summary["system_state"]["cgroup"]["memory_counter_delta"][
          "pswpin"] is not None and
      summary["system_state"]["cgroup"]["pressure_delta_seconds"].get(
          "memory_full") is not None and
      summary["system_state"]["descendants"]["rss_peak_bytes"] is not None
      for summary in summaries)
  checks = [
      check("repository_clean_at_gate", not git["dirty"], git=git),
      check("source_worker_completed", source_run.get("returncode") == 0),
      check("source_is_128k_candidate_timing_bundle",
            source_config.get("bucket") == 131072 and
            source_config.get("mode") == "candidate" and
            source_config.get("purpose") == "paired_product_timing"),
      check("exactly_two_identical_serial_candidate_workers",
            len(workers) == 2 and not concurrent and config_pass and
            scope_pass),
      check("workers_complete_without_oom", worker_pass),
      check("workers_preserve_teacher_forced_prefix", token_pass),
      check("workers_use_exact_source_plugin", plugin_pass),
      check("workers_preserve_compact_token_only_provider", provider_pass),
      check("read_only_gt_frequency_telemetry_present",
            frequency_telemetry_pass),
      check("read_only_vm_cgroup_descendant_thermal_telemetry_present",
            system_telemetry_pass),
      check("attribution_only_no_speed_claim", True),
  ]
  passed = all(row["pass"] for row in checks)
  metrics = {
      "schema_version": SCHEMA,
      "required_checks_passed": passed,
      "git": git,
      "plan": plan,
      "source_worker": str(source_dir),
      "source_worker_result_sha256": PRODUCT.sha256_file(source_result_path),
      "candidate_gpu_plugin": str(plugin),
      "candidate_gpu_plugin_sha256": PRODUCT.sha256_file(plugin),
      "gpu_workers_launched": 2,
      "stock_workers_launched": 0,
      "concurrent_workers_at_launch": concurrent,
      "workers": [
          {key: value for key, value in worker.items() if key != "result"}
          for worker in workers],
      "summaries": summaries,
      "performance_ratio": performance_ratio,
      "system_state_samples": len(all_samples),
      "runtime_or_power_settings_changed": False,
      "attribution_is_speed_claim": False,
      "checks": checks,
  }
  write_json(out_dir / "metrics.json", metrics)
  write_json(out_dir / "manifest.json", {
      "schema_version": SCHEMA,
      "git": git,
      **plan,
  })
  (out_dir / "summary.md").write_text(
      "# Read-only system-state attribution\n\n"
      f"Required checks passed: `{str(passed).lower()}`. Two identical "
      "candidate workers ran strictly serially with read-only GT, VM, cgroup, "
      "descendant, and thermal sampling. No runtime or system setting "
      "changed.\n\n"
      f"Repeat-1 / repeat-0 prefill ratio: "
      f"`{performance_ratio['repeat1_over_repeat0_prefill']}`; decode ratio: "
      f"`{performance_ratio['repeat1_over_repeat0_decode']}`; active-frequency "
      f"median ratio: "
      f"`{performance_ratio['repeat1_over_repeat0_active_frequency_median']}`."
      f" Cgroup swap-in ratio: "
      f"`{performance_ratio['repeat1_over_repeat0_cgroup_pswpin']}`; "
      f"major-fault ratio: "
      f"`{performance_ratio['repeat1_over_repeat0_cgroup_pgmajfault']}`. "
      "This artifact is attribution telemetry, not a speed claim.\n",
      encoding="utf-8")
  print(json.dumps({
      "artifact": str(out_dir),
      "required_checks_passed": passed,
      "performance_ratio": performance_ratio,
      "oom_observed": any(worker.get("oom_observed") for worker in workers),
      "memory_guard_tripped": any(
          (worker.get("memory_guard") or {}).get("tripped")
          for worker in workers),
  }, sort_keys=True), flush=True)
  return 0 if passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
