#!/usr/bin/env python3
"""Measure the retained PR5059 GMLP binary at the two exact product shapes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-onednn-gmlp-exact-component-run-v0"
ADMISSION = ROOT / (
    "output/onednn-gmlp-exact-component-admission-"
    "20260731Tseq2223-clean/plan.json")
ADMISSION_SHA256 = (
    "f21ff2d8bffb326f5cc0bd7cfceb8544b5369bde2a2508914254b35961cb7c33")
BUILD_AUDIT = ROOT / (
    "output/onednn-gmlp-exact-component-build-audit-"
    "20260731Tseq2224a-clean/metrics.json")
BUILD_AUDIT_SHA256 = (
    "25880917fec7c95ac2096cecaf994bee6851b247b7b751f4ee770f53ac0ac2b7")
SOURCE_WORKTREE = Path(
    "/home/intel/intel-qwen36-r0/source/oneDNN-862174-gmlp-exact")
ONEDNN_HEAD = "8621740ea5e600468c76a11a3c0c1616977f978d"
BUILD_DIR = Path(
    "/home/intel/intel-qwen36-r0/build/onednn-862174-gmlp-exact")
TEST_BINARY = BUILD_DIR / "tests/gtests/internals/test_internals_gmlp"
TEST_BINARY_SHA256 = (
    "61a749e95f6ead521ce17e13d50ea9f377e12a5121036ec5764bc3d8b24c8747")
SYSTEMD_RUN = Path("/usr/bin/systemd-run")
SYSTEMCTL = Path("/usr/bin/systemctl")
TIME = Path("/usr/bin/time")
OCL_ICD_VENDOR_DIR = Path("/etc/OpenCL/vendors")
INTEL_ICD = OCL_ICD_VENDOR_DIR / "intel.icd"
EXPECTED_INTEL_ICD = "/usr/lib/x86_64-linux-gnu/intel-opencl/libigdrcl.so"
PREFLIGHT_BYTES = 8 * 1024**3
ABORT_BYTES = 4 * 1024**3
PAIRS_PER_SHAPE = 8
BOOTSTRAP_RESAMPLES = 20_000
BOOTSTRAP_SEED = 22_230
REQUIRED_PROVIDER = "ocl:micro_horz:any"
FLOAT = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
SHAPES = {
    "decode": {
        "gmlp_test": "1 2048 512 16 4 1 64 16 8",
        "mb": 1,
        "ic": 2048,
        "oc": 512,
        "delta_ucb_cap_ms": 0.0,
    },
    "prefill": {
        "gmlp_test": "2048 2048 512 16 4 1 64 16 8",
        "mb": 2048,
        "ic": 2048,
        "oc": 512,
        "delta_ucb_cap_ms": -0.001209,
    },
}


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", required=True, type=Path)
  parser.add_argument("--worker-timeout-s", default=600.0, type=float)
  parser.add_argument("--poll-interval-s", default=0.1, type=float)
  args = parser.parse_args()
  if args.worker_timeout_s <= 0 or args.poll_interval_s <= 0:
    parser.error("timeout and poll interval must be positive")
  return args


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    while chunk := stream.read(1024 * 1024):
      digest.update(chunk)
  return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise TypeError(f"expected JSON object: {path}")
  return value


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True) + "\n",
      encoding="utf-8")


def run(command: list[str], cwd: Path = ROOT) -> str:
  result = subprocess.run(
      command, cwd=cwd, check=False, capture_output=True, text=True,
      encoding="utf-8", errors="replace")
  if result.returncode != 0:
    raise RuntimeError(
        f"command failed ({result.returncode}): {command}\n"
        f"{result.stderr}")
  return result.stdout


def git(cwd: Path, *args: str) -> str:
  return run(["git", *args], cwd=cwd).strip()


def relative(path: Path) -> str:
  try:
    return path.resolve().relative_to(ROOT).as_posix()
  except ValueError:
    return str(path.resolve())


def proc_meminfo() -> dict[str, int]:
  result = {}
  for line in Path("/proc/meminfo").read_text(
      encoding="utf-8").splitlines():
    key, value = line.split(":", 1)
    fields = value.strip().split()
    if fields and fields[0].isdigit():
      result[key] = int(fields[0]) * 1024
  return result


def read_int(path: Path) -> int:
  try:
    return int(path.read_text(encoding="utf-8").strip())
  except (FileNotFoundError, PermissionError, ValueError):
    return 0


def read_named_ints(path: Path) -> dict[str, int]:
  try:
    rows = path.read_text(encoding="utf-8").splitlines()
  except (FileNotFoundError, PermissionError):
    return {}
  result = {}
  for row in rows:
    fields = row.split()
    if len(fields) == 2 and fields[1].isdigit():
      result[fields[0]] = int(fields[1])
  return result


def process_memory(pid: int) -> dict[str, int]:
  result = {"rss_bytes": 0, "swap_bytes": 0}
  try:
    rows = Path(f"/proc/{pid}/status").read_text(
        encoding="utf-8").splitlines()
  except (FileNotFoundError, PermissionError):
    return result
  for row in rows:
    if row.startswith("VmRSS:"):
      result["rss_bytes"] = int(row.split()[1]) * 1024
    elif row.startswith("VmSwap:"):
      result["swap_bytes"] = int(row.split()[1]) * 1024
  return result


def parse_time_max_rss(path: Path) -> int:
  if not path.is_file():
    return 0
  match = re.search(
      r"Maximum resident set size \(kbytes\):\s*(\d+)",
      path.read_text(encoding="utf-8", errors="replace"))
  return int(match.group(1)) * 1024 if match else 0


def repository_state(output: Path) -> dict[str, Any]:
  head = git(ROOT, "rev-parse", "HEAD")
  upstream = git(ROOT, "rev-parse", "@{u}")
  output_rel = relative(output)
  dirty = []
  for row in git(
      ROOT, "status", "--porcelain", "--untracked-files=all").splitlines():
    path = row[3:]
    if path == output_rel or path.startswith(output_rel + "/"):
      continue
    dirty.append(row)
  return {
      "branch": git(ROOT, "branch", "--show-current"),
      "commit": head,
      "upstream_commit": upstream,
      "pushed": head == upstream,
      "dirty": bool(dirty),
      "dirty_paths": dirty,
  }


def check(name: str, passed: bool, **evidence: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **evidence}


def scope_unit(output: Path, block: int, shape: str) -> str:
  digest = hashlib.sha256(
      f"{output.resolve()}:{block}:{shape}".encode("utf-8")
  ).hexdigest()[:12]
  return f"iq36-gmlp-b{block:02d}-{shape}-{digest}"


def scope_cgroup(unit: str) -> Path:
  uid = os.getuid()
  return Path(
      f"/sys/fs/cgroup/user.slice/user-{uid}.slice/user@{uid}.service/"
      f"app.slice/{unit}.scope")


def stop_scope(unit: str, process: subprocess.Popen[Any]) -> None:
  subprocess.run(
      [str(SYSTEMCTL), "--user", "kill", "--signal=SIGINT",
       f"{unit}.scope"],
      check=False, capture_output=True, text=True)
  try:
    process.wait(timeout=10.0)
    return
  except subprocess.TimeoutExpired:
    pass
  subprocess.run(
      [str(SYSTEMCTL), "--user", "kill", "--signal=SIGKILL",
       f"{unit}.scope"],
      check=False, capture_output=True, text=True)
  try:
    os.killpg(process.pid, signal.SIGKILL)
  except ProcessLookupError:
    pass
  process.wait()


def parse_worker_output(text: str, shape: dict[str, Any]) -> dict[str, Any]:
  mismatch_matches = re.findall(
      r"total mismatches:\s*(\d+),\s*allowed:\s*(\d+)", text)
  timing_matches = re.findall(
      rf"avg time internal vs primitive:\s*({FLOAT})\s*vs\s*({FLOAT}),"
      rf"\s*w/speedup of\s*({FLOAT})",
      text)
  shape_match = re.search(
      r"GMLP_TEST: \((\d+) x (\d+) x (\d+), "
      r"(\w+) x (\w+), q = (\d+), gs = (\d+), "
      r"s = (\w+), zp = (\w+)\)",
      text)
  mismatches = int(mismatch_matches[0][0]) if len(
      mismatch_matches) == 1 else None
  allowed = int(mismatch_matches[0][1]) if len(
      mismatch_matches) == 1 else None
  internal_ms = float(timing_matches[0][0]) if len(
      timing_matches) == 1 else None
  primitive_ms = float(timing_matches[0][1]) if len(
      timing_matches) == 1 else None
  reported_speedup = float(timing_matches[0][2]) if len(
      timing_matches) == 1 else None
  parsed_shape = None
  if shape_match:
    parsed_shape = {
        "mb": int(shape_match.group(1)),
        "ic": int(shape_match.group(2)),
        "oc": int(shape_match.group(3)),
        "src_dt": shape_match.group(4),
        "weights_dt": shape_match.group(5),
        "quantized": int(shape_match.group(6)),
        "group_size": int(shape_match.group(7)),
        "scale_dt": shape_match.group(8),
        "zero_point_dt": shape_match.group(9),
    }
  expected_shape = {
      "mb": shape["mb"],
      "ic": shape["ic"],
      "oc": shape["oc"],
      "src_dt": "f16",
      "weights_dt": "u4",
      "quantized": 1,
      "group_size": 64,
      "scale_dt": "f16",
      "zero_point_dt": "u8",
  }
  expected_allowed = int(shape["mb"] * shape["ic"] * 0.0006)
  finite_timings = bool(
      internal_ms is not None and primitive_ms is not None
      and math.isfinite(internal_ms) and math.isfinite(primitive_ms)
      and internal_ms > 0 and primitive_ms > 0)
  computed_speedup = (
      primitive_ms / internal_ms if finite_timings else None)
  verbose_lines = text.splitlines()
  successful_operations = (
      ",primitive,exec,",
      ",primitive,create:cache_hit,",
      ",primitive,create:cache_miss,",
      ",primitive,create:kernel_cache_hit,",
      ",primitive,create:persistent_cache_hit,",
      ",primitive,create:nested_cache_hit,",
  )
  successful_gmlp_lines = [
      line for line in verbose_lines
      if ",gated_mlp," in line
      and any(marker in line for marker in successful_operations)]
  provider_success_lines = [
      line for line in successful_gmlp_lines
      if REQUIRED_PROVIDER in line]
  fallback_success_lines = [
      line for line in successful_gmlp_lines
      if "ocl:ref:any" in line]
  return {
      "mismatch_rows": len(mismatch_matches),
      "timing_rows": len(timing_matches),
      "mismatches": mismatches,
      "allowed": allowed,
      "expected_allowed": expected_allowed,
      "internal_ms": internal_ms,
      "primitive_ms": primitive_ms,
      "delta_ms": (
          internal_ms - primitive_ms if finite_timings else None),
      "reported_speedup": reported_speedup,
      "computed_speedup": computed_speedup,
      "shape": parsed_shape,
      "expected_shape": expected_shape,
      "shape_exact": parsed_shape == expected_shape,
      "provider": REQUIRED_PROVIDER,
      "provider_count": text.count(REQUIRED_PROVIDER),
      "provider_success_count": len(provider_success_lines),
      "provider_success_lines": provider_success_lines,
      "fallback_provider_success_lines": fallback_success_lines,
      "provider_present": bool(
          provider_success_lines and not fallback_success_lines),
      "gtest_one_test_passed": "[  PASSED  ] 1 test." in text,
      "correctness_passed": bool(
          mismatches is not None and allowed is not None
          and allowed == expected_allowed and mismatches <= allowed),
      "timing_parsed": finite_timings,
  }


def run_worker(
    output: Path, block: int, shape_name: str, shape: dict[str, Any],
    timeout_s: float, poll_interval_s: float,
) -> dict[str, Any]:
  worker_dir = output / "raw" / f"block{block:02d}" / shape_name
  worker_dir.mkdir(parents=True, exist_ok=False)
  stdout_path = worker_dir / "stdout.log"
  stderr_path = worker_dir / "stderr.log"
  time_path = worker_dir / "time.txt"
  unit = scope_unit(output, block, shape_name)
  cgroup = scope_cgroup(unit)
  command = [
      str(TEST_BINARY),
      "--gtest_color=no",
      "--gtest_filter=VEC/mlp_test_t.compare/*",
  ]
  timed_command = [
      str(TIME), "-v", "-o", str(time_path), *command]
  scoped_command = [
      str(SYSTEMD_RUN), "--user", "--scope", "--quiet", "--collect",
      f"--unit={unit}", *timed_command]
  environment = os.environ.copy()
  environment.update({
      "GMLP_TEST": str(shape["gmlp_test"]),
      "DNNL_VERBOSE": "all",
      "ONEDNN_VERBOSE": "all",
      "ONEDNN_VERBOSE_TIMESTAMP": "0",
      "OCL_ICD_VENDORS": str(OCL_ICD_VENDOR_DIR),
      "LD_LIBRARY_PATH": (
          str(BUILD_DIR / "src") + ":"
          "/home/intel/intel-box-env/conda/lib:"
          + environment.get("LD_LIBRARY_PATH", "")),
  })
  start_memory = proc_meminfo()
  if int(start_memory.get("MemAvailable", 0)) < PREFLIGHT_BYTES:
    raise RuntimeError(
        f"{shape_name} block {block} preflight below 8 GiB: "
        f"{start_memory.get('MemAvailable', 0)}")
  monitor: dict[str, Any] = {
      "system_available_min_bytes": int(start_memory["MemAvailable"]),
      "system_swap_used_start_bytes": (
          int(start_memory.get("SwapTotal", 0))
          - int(start_memory.get("SwapFree", 0))),
      "system_swap_used_peak_bytes": (
          int(start_memory.get("SwapTotal", 0))
          - int(start_memory.get("SwapFree", 0))),
      "cgroup_memory_peak_bytes": 0,
      "cgroup_swap_peak_bytes": 0,
      "wrapper_rss_peak_bytes": 0,
      "wrapper_swap_peak_bytes": 0,
      "process_count_peak": 0,
      "memory_events_max": {},
      "samples": 0,
  }
  started_monotonic = time.monotonic()
  started_at = datetime.now(timezone.utc).isoformat()
  timed_out = False
  guard_tripped = False
  with stdout_path.open("w", encoding="utf-8") as stdout_handle, \
       stderr_path.open("w", encoding="utf-8") as stderr_handle:
    process = subprocess.Popen(
        scoped_command, cwd=ROOT, env=environment,
        stdout=stdout_handle, stderr=stderr_handle, text=True,
        start_new_session=True)
    while process.poll() is None:
      elapsed = time.monotonic() - started_monotonic
      if elapsed > timeout_s:
        timed_out = True
        stop_scope(unit, process)
        break
      system = proc_meminfo()
      available = int(system.get("MemAvailable", 0))
      swap_used = (
          int(system.get("SwapTotal", 0))
          - int(system.get("SwapFree", 0)))
      wrapper = process_memory(process.pid)
      cgroup_current = read_int(cgroup / "memory.current")
      cgroup_swap = read_int(cgroup / "memory.swap.current")
      cgroup_peak = read_int(cgroup / "memory.peak")
      events = read_named_ints(cgroup / "memory.events")
      process_count = len(
          (cgroup / "cgroup.procs").read_text(
              encoding="utf-8").splitlines()
          ) if (cgroup / "cgroup.procs").is_file() else 0
      monitor["samples"] += 1
      monitor["system_available_min_bytes"] = min(
          int(monitor["system_available_min_bytes"]), available)
      monitor["system_swap_used_peak_bytes"] = max(
          int(monitor["system_swap_used_peak_bytes"]), swap_used)
      monitor["cgroup_memory_peak_bytes"] = max(
          int(monitor["cgroup_memory_peak_bytes"]),
          cgroup_current, cgroup_peak)
      monitor["cgroup_swap_peak_bytes"] = max(
          int(monitor["cgroup_swap_peak_bytes"]), cgroup_swap)
      monitor["wrapper_rss_peak_bytes"] = max(
          int(monitor["wrapper_rss_peak_bytes"]), wrapper["rss_bytes"])
      monitor["wrapper_swap_peak_bytes"] = max(
          int(monitor["wrapper_swap_peak_bytes"]), wrapper["swap_bytes"])
      monitor["process_count_peak"] = max(
          int(monitor["process_count_peak"]), process_count)
      previous_events = monitor["memory_events_max"]
      monitor["memory_events_max"] = {
          key: max(int(previous_events.get(key, 0)), value)
          for key, value in events.items()}
      if available < ABORT_BYTES:
        guard_tripped = True
        stop_scope(unit, process)
        break
      time.sleep(poll_interval_s)
    returncode = process.wait()
  finished_monotonic = time.monotonic()
  finished_at = datetime.now(timezone.utc).isoformat()
  stdout = stdout_path.read_text(encoding="utf-8", errors="replace")
  stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
  events = monitor["memory_events_max"]
  oom_observed = bool(
      not guard_tripped
      and (returncode in (-9, 137)
           or int(events.get("oom_kill", 0)) > 0
           or "out of memory" in (stdout + stderr).lower()))
  monitor["time_max_rss_bytes"] = parse_time_max_rss(time_path)
  monitor["process_rss_peak_bytes"] = max(
      int(monitor["cgroup_memory_peak_bytes"]),
      int(monitor["time_max_rss_bytes"]),
      int(monitor["wrapper_rss_peak_bytes"]))
  parsed = parse_worker_output(stdout + "\n" + stderr, shape)
  return {
      "block": block,
      "shape": shape_name,
      "gmlp_test": shape["gmlp_test"],
      "command": command,
      "environment": {
          "GMLP_TEST": shape["gmlp_test"],
          "DNNL_VERBOSE": "all",
          "ONEDNN_VERBOSE": "all",
          "ONEDNN_VERBOSE_TIMESTAMP": "0",
          "OCL_ICD_VENDORS": str(OCL_ICD_VENDOR_DIR),
      },
      "scope": {
          "enabled": True,
          "unit": unit,
          "cgroup_root": str(cgroup),
          "resource_limits_changed": False,
      },
      "returncode": returncode,
      "elapsed_seconds": finished_monotonic - started_monotonic,
      "started_at": started_at,
      "finished_at": finished_at,
      "started_monotonic": started_monotonic,
      "finished_monotonic": finished_monotonic,
      "timed_out": timed_out,
      "memory_guard_tripped": guard_tripped,
      "oom_observed": oom_observed,
      "monitor": monitor,
      "stdout": relative(stdout_path),
      "stderr": relative(stderr_path),
      "time": relative(time_path),
      "parsed": parsed,
  }


def bootstrap_median_delta(
    deltas: list[float], rng: np.random.Generator,
) -> dict[str, Any]:
  values = np.asarray(deltas, dtype=np.float64)
  indices = rng.integers(
      0, len(values), size=(BOOTSTRAP_RESAMPLES, len(values)))
  samples = np.median(values[indices], axis=1)
  return {
      "n": len(deltas),
      "delta_definition": "internal_micro_horz_ms - primitive_ms",
      "point_median_delta_ms": float(np.median(values)),
      "one_sided_95pct_ucb_delta_ms": float(np.quantile(
          samples, 0.95, method="linear")),
      "confidence": 0.95,
      "method": "paired_one_sided_percentile_bootstrap_median_delta",
      "quantile_method": "linear",
      "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
      "bootstrap_seed": BOOTSTRAP_SEED,
  }


def worker_valid(row: dict[str, Any]) -> bool:
  parsed = row["parsed"]
  events = row["monitor"]["memory_events_max"]
  return bool(
      row["returncode"] == 0
      and not row["timed_out"]
      and not row["memory_guard_tripped"]
      and not row["oom_observed"]
      and int(events.get("oom", 0)) == 0
      and int(events.get("oom_kill", 0)) == 0
      and int(events.get("oom_group_kill", 0)) == 0
      and parsed["gtest_one_test_passed"]
      and parsed["shape_exact"]
      and parsed["provider_present"]
      and parsed["correctness_passed"]
      and parsed["timing_parsed"])


def main() -> int:
  args = parse_args()
  output = args.output.resolve()
  output.mkdir(parents=True, exist_ok=False)
  required = (
      ADMISSION, BUILD_AUDIT, SOURCE_WORKTREE, TEST_BINARY,
      SYSTEMD_RUN, SYSTEMCTL, TIME, INTEL_ICD)
  missing = [str(path) for path in required if not path.exists()]
  if missing:
    raise SystemExit("missing GMLP component inputs: " + ", ".join(missing))

  repo = repository_state(output)
  admission = load_json(ADMISSION)
  build_audit = load_json(BUILD_AUDIT)
  source_commit = git(SOURCE_WORKTREE, "rev-parse", "HEAD")
  source_dirty = git(
      SOURCE_WORKTREE, "status", "--short", "--untracked-files=all")
  initial_memory = proc_meminfo()
  identity_ok = bool(
      repo["branch"] == "main" and repo["pushed"] and not repo["dirty"]
      and sha256(ADMISSION) == ADMISSION_SHA256
      and admission["verdict"]["component_build_admitted"] is True
      and sha256(BUILD_AUDIT) == BUILD_AUDIT_SHA256
      and build_audit["verdict"]["component_runs_admitted"] is True
      and source_commit == ONEDNN_HEAD and source_dirty == ""
      and sha256(TEST_BINARY) == TEST_BINARY_SHA256
      and INTEL_ICD.read_text(encoding="utf-8").strip()
      == EXPECTED_INTEL_ICD
      and int(initial_memory.get("MemAvailable", 0)) >= PREFLIGHT_BYTES)
  if not identity_ok:
    raise SystemExit(
        "component identity/repository/memory preflight failed before GPU")

  workers = []
  stop_reason = None
  for block in range(1, PAIRS_PER_SHAPE + 1):
    for shape_name in ("decode", "prefill"):
      row = run_worker(
          output, block, shape_name, SHAPES[shape_name],
          args.worker_timeout_s, args.poll_interval_s)
      workers.append(row)
      print(json.dumps({
          "block": block,
          "shape": shape_name,
          "returncode": row["returncode"],
          "elapsed_seconds": row["elapsed_seconds"],
          "provider_count": row["parsed"]["provider_count"],
          "mismatches": row["parsed"]["mismatches"],
          "allowed": row["parsed"]["allowed"],
          "internal_ms": row["parsed"]["internal_ms"],
          "primitive_ms": row["parsed"]["primitive_ms"],
          "delta_ms": row["parsed"]["delta_ms"],
          "memory_available_min_bytes": (
              row["monitor"]["system_available_min_bytes"]),
          "worker_valid": worker_valid(row),
      }, sort_keys=True), flush=True)
      if not worker_valid(row):
        stop_reason = (
            f"block {block} {shape_name} failed provider, correctness, "
            "timing, process, or memory checks")
        break
    if stop_reason:
      break

  by_shape = {
      shape_name: [
          row for row in workers if row["shape"] == shape_name]
      for shape_name in SHAPES}
  complete = all(
      len(rows) == PAIRS_PER_SHAPE
      for rows in by_shape.values())
  inference: dict[str, Any] = {}
  if complete and all(worker_valid(row) for row in workers):
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for shape_name in ("decode", "prefill"):
      deltas = [
          float(row["parsed"]["delta_ms"])
          for row in by_shape[shape_name]]
      result = bootstrap_median_delta(deltas, rng)
      result["delta_ucb_cap_ms"] = SHAPES[
          shape_name]["delta_ucb_cap_ms"]
      result["pass"] = (
          result["one_sided_95pct_ucb_delta_ms"]
          <= result["delta_ucb_cap_ms"])
      result["primitive_ms"] = [
          row["parsed"]["primitive_ms"]
          for row in by_shape[shape_name]]
      result["internal_ms"] = [
          row["parsed"]["internal_ms"]
          for row in by_shape[shape_name]]
      result["delta_ms"] = deltas
      inference[shape_name] = result

  units = [row["scope"]["unit"] for row in workers]
  intervals = [
      (float(row["started_monotonic"]), float(row["finished_monotonic"]))
      for row in workers]
  strict_serial = all(
      intervals[index][0] >= intervals[index - 1][1]
      for index in range(1, len(intervals)))
  all_workers_valid = bool(
      complete and all(worker_valid(row) for row in workers))
  performance_pass = bool(
      all_workers_valid
      and all(inference[shape]["pass"] for shape in SHAPES))
  min_available = min(
      [int(initial_memory["MemAvailable"])]
      + [int(row["monitor"]["system_available_min_bytes"])
         for row in workers])
  peak_rss = max(
      [0] + [int(row["monitor"]["process_rss_peak_bytes"])
             for row in workers])
  peak_cgroup_swap = max(
      [0] + [int(row["monitor"]["cgroup_swap_peak_bytes"])
             for row in workers])
  checks = [
      check(
          "repository_clean_and_pushed_at_gate",
          repo["branch"] == "main" and repo["pushed"] and not repo["dirty"],
          **repo),
      check(
          "admission_and_build_audit_identity_exact",
          sha256(ADMISSION) == ADMISSION_SHA256
          and admission["verdict"]["component_build_admitted"] is True
          and sha256(BUILD_AUDIT) == BUILD_AUDIT_SHA256
          and build_audit["verdict"]["component_runs_admitted"] is True,
          admission_sha256=sha256(ADMISSION),
          build_audit_sha256=sha256(BUILD_AUDIT)),
      check(
          "source_and_binary_identity_exact",
          source_commit == ONEDNN_HEAD and source_dirty == ""
          and sha256(TEST_BINARY) == TEST_BINARY_SHA256,
          source_commit=source_commit, source_dirty=source_dirty,
          binary_sha256=sha256(TEST_BINARY)),
      check(
          "opencl_icd_discovery_path_exact",
          INTEL_ICD.read_text(encoding="utf-8").strip()
          == EXPECTED_INTEL_ICD,
          vendor_directory=str(OCL_ICD_VENDOR_DIR),
          icd_file=str(INTEL_ICD),
          icd_library=INTEL_ICD.read_text(encoding="utf-8").strip()),
      check(
          "exact_eight_pairs_per_shape_completed",
          complete and len(workers) == 2 * PAIRS_PER_SHAPE,
          counts={key: len(value) for key, value in by_shape.items()},
          stop_reason=stop_reason),
      check(
          "workers_strictly_serial_and_unique_scoped",
          strict_serial and len(units) == len(set(units)),
          maximum_concurrent_workers=1, units=units),
      check(
          "all_workers_process_and_memory_safe",
          all_workers_valid
          and min_available >= ABORT_BYTES,
          minimum_available_bytes=min_available,
          maximum_process_rss_bytes=peak_rss,
          maximum_cgroup_swap_bytes=peak_cgroup_swap),
      check(
          "all_workers_select_exact_provider",
          all_workers_valid and all(
              row["parsed"]["provider_present"] for row in workers),
          required_provider=REQUIRED_PROVIDER,
          provider_counts=[
              row["parsed"]["provider_count"] for row in workers]),
      check(
          "all_component_correctness_rows_pass",
          all_workers_valid and all(
              row["parsed"]["correctness_passed"] for row in workers),
          mismatch_rows=[{
              "block": row["block"],
              "shape": row["shape"],
              "mismatches": row["parsed"]["mismatches"],
              "allowed": row["parsed"]["allowed"],
          } for row in workers]),
      check(
          "prefill_delta_ucb_clears_registered_funding",
          bool(inference.get("prefill", {}).get("pass", False)),
          inference=inference.get("prefill")),
      check(
          "decode_delta_ucb_is_nonregressive",
          bool(inference.get("decode", {}).get("pass", False)),
          inference=inference.get("decode")),
      check(
          "no_product_model_or_infer_request_ran",
          True, model_workers_started=0, infer_requests_created=0,
          openvino_product_builds=0),
  ]
  passed = all(row["pass"] for row in checks)
  verdict = {
      "required_checks_passed": passed,
      "component_promotable": passed,
      "product_integration_design_admitted": passed,
      "product_build_admitted": False,
      "verdict": (
          "admit_exact_gmlp_product_integration_design"
          if passed else
          "reject_current_pr5059_exact_gmlp_component"),
      "next_if_pass": (
          "perform a zero-GPU exact OpenVINO integration source and version "
          "binding audit; do not build a product plugin yet"),
      "next_if_fail": (
          "close this PR5059 body and switch to a distinct profile-backed "
          "kernel/provider/layout route"),
  }
  metrics = {
      "schema": SCHEMA,
      "workstream": WS,
      "created_at": datetime.now(timezone.utc).isoformat(),
      "git": repo,
      "inputs": {
          "admission": {
              "path": relative(ADMISSION), "sha256": sha256(ADMISSION)},
          "build_audit": {
              "path": relative(BUILD_AUDIT), "sha256": sha256(BUILD_AUDIT)},
          "source_commit": source_commit,
          "binary": str(TEST_BINARY),
          "binary_sha256": sha256(TEST_BINARY),
          "opencl_icd_vendor_dir": str(OCL_ICD_VENDOR_DIR),
          "intel_icd": str(INTEL_ICD),
          "intel_icd_library": INTEL_ICD.read_text(
              encoding="utf-8").strip(),
      },
      "protocol": {
          "shape_order_per_block": ["decode", "prefill"],
          "pairs_per_shape": PAIRS_PER_SHAPE,
          "maximum_concurrent_workers": 1,
          "worker_timeout_s": args.worker_timeout_s,
          "environment": {
              "DNNL_VERBOSE": "all",
              "ONEDNN_VERBOSE": "all",
              "ONEDNN_VERBOSE_TIMESTAMP": "0",
              "OCL_ICD_VENDORS": str(OCL_ICD_VENDOR_DIR),
          },
          "required_provider": REQUIRED_PROVIDER,
          "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
          "bootstrap_seed": BOOTSTRAP_SEED,
          "shapes": SHAPES,
      },
      "workers": workers,
      "inference": inference,
      "memory": {
          "initial": initial_memory,
          "preflight_bytes": PREFLIGHT_BYTES,
          "abort_below_bytes": ABORT_BYTES,
          "minimum_available_bytes": min_available,
          "maximum_process_rss_bytes": peak_rss,
          "maximum_cgroup_swap_bytes": peak_cgroup_swap,
      },
      "process_census": {
          "workers_started": len(workers),
          "maximum_concurrent_workers": 1,
          "gpu_contexts_expected": len(workers),
          "model_workers_started": 0,
          "infer_requests_created": 0,
          "openvino_product_builds": 0,
      },
      "checks": checks,
      "verdict": verdict,
  }
  write_json(output / "metrics.json", metrics)
  prefill = inference.get("prefill", {})
  decode = inference.get("decode", {})
  (output / "report.md").write_text(
      "# oneDNN PR5059 exact GMLP component\n\n"
      f"- Required checks: `{passed}`\n"
      f"- Verdict: `{verdict['verdict']}`\n"
      f"- Workers: `{len(workers)}` strictly serial\n"
      f"- Provider: `{REQUIRED_PROVIDER}`\n"
      f"- Prefill delta median/UCB/cap ms: "
      f"`{prefill.get('point_median_delta_ms')}/"
      f"{prefill.get('one_sided_95pct_ucb_delta_ms')}/-0.001209`\n"
      f"- Decode delta median/UCB/cap ms: "
      f"`{decode.get('point_median_delta_ms')}/"
      f"{decode.get('one_sided_95pct_ucb_delta_ms')}/0`\n"
      f"- Minimum available / peak RSS / peak cgroup swap B: "
      f"`{min_available}/{peak_rss}/{peak_cgroup_swap}`\n"
      "- Product model/InferRequest/build: `0/0/0`\n",
      encoding="utf-8")
  print(json.dumps({
      "output": relative(output),
      "required_checks_passed": passed,
      "verdict": verdict["verdict"],
      "workers_started": len(workers),
      "prefill_delta_ucb_ms": prefill.get(
          "one_sided_95pct_ucb_delta_ms"),
      "decode_delta_ucb_ms": decode.get(
          "one_sided_95pct_ucb_delta_ms"),
      "minimum_available_bytes": min_available,
      "maximum_process_rss_bytes": peak_rss,
  }, sort_keys=True), flush=True)
  return 0 if passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
