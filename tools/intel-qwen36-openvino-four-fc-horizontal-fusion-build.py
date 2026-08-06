#!/usr/bin/env python3
"""Run the single admitted four-FC incremental GPU-plugin build."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-openvino-four-fc-horizontal-fusion-build-v0"
R0 = Path("/home/intel/intel-qwen36-r0")
SOURCE_TREE = R0 / "source/openvino-90214e5be05"
BUILD_TREE = R0 / "build/openvino-90214e-l0-gpu"
PLUGIN = R0 / (
    "output/openvino-90214e-l0-gpu/bin/intel64/Release/"
    "libopenvino_intel_gpu_plugin.so")
CONTROL_PLUGIN = R0 / (
    "output/openvino-90214e-l0-gpu-dynamic-split-control-seq1304/"
    "bin/intel64/Release/libopenvino_intel_gpu_plugin.so")
PATCH = ROOT / "engine/openvino/iq36-four-fc-horizontal-fusion.patch"
SOURCE_GATE = ROOT / (
    "output/openvino-four-fc-horizontal-fusion-source-gate-"
    "20260718Tseq1329-cleanZ/metrics.json")
TARGET_SOURCE = (
    "src/plugins/intel_gpu/src/plugin/transformations/"
    "fc_horizontal_fusion.cpp")
TARGET_TEST = (
    "src/plugins/intel_gpu/tests/unit/transformations/"
    "horizontal_fc_fusion_test.cpp")
EXPECTED_CONTROL_SHA256 = (
    "20ea71d72b12c0a2428bdbfd125c2e232b418bd8aa84e4dd7f29d73b8aa1e06a")
EXPECTED_PREVIOUS_CANDIDATE_SHA256 = (
    "1c96cbac5d0f7f6edc9fe8a55ba0340f189600df1aea594d8cdbc5beeeb5944f")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--memory-stop-gib", type=float, default=4.0)
  parser.add_argument("--parallel", type=int, default=4)
  parser.add_argument("--timeout-s", type=float, default=300.0)
  parser.add_argument("--poll-interval-s", type=float, default=0.05)
  args = parser.parse_args()
  if (args.memory_stop_gib <= 0.0 or args.parallel <= 0
      or args.timeout_s <= 0.0 or args.poll_interval_s <= 0.0):
    parser.error("memory stop, parallelism, timeout, and poll interval must be positive")
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


def display(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path.resolve())


def proc_meminfo() -> dict[str, int]:
  result = {}
  for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
    key, value = line.split(":", 1)
    fields = value.strip().split()
    if fields and fields[0].isdigit():
      result[key] = int(fields[0]) * 1024
  return result


def process_group_memory(pgrp: int) -> dict[str, int]:
  rss = 0
  swap = 0
  processes = 0
  for stat_path in Path("/proc").glob("[0-9]*/stat"):
    try:
      stat = stat_path.read_text(encoding="utf-8")
      tail = stat[stat.rfind(")") + 2:].split()
      if len(tail) < 3 or int(tail[2]) != pgrp:
        continue
      status_path = stat_path.with_name("status")
      values = status_path.read_text(
          encoding="utf-8", errors="replace").splitlines()
    except (FileNotFoundError, ProcessLookupError, PermissionError,
            ValueError):
      continue
    processes += 1
    for line in values:
      if line.startswith("VmRSS:"):
        rss += int(line.split()[1]) * 1024
      elif line.startswith("VmSwap:"):
        swap += int(line.split()[1]) * 1024
  return {"rss_bytes": rss, "swap_bytes": swap, "processes": processes}


def stop_group(process: subprocess.Popen[Any], first_signal: int) -> None:
  try:
    os.killpg(process.pid, first_signal)
  except ProcessLookupError:
    return
  try:
    process.wait(timeout=10.0)
    return
  except subprocess.TimeoutExpired:
    pass
  try:
    os.killpg(process.pid, signal.SIGKILL)
  except ProcessLookupError:
    pass
  process.wait()


def git_state(output: Path) -> dict[str, Any]:
  commit = subprocess.run(
      ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
      capture_output=True, text=True).stdout.strip()
  rows = subprocess.run(
      ["git", "status", "--porcelain"], cwd=ROOT, check=True,
      capture_output=True, text=True).stdout.splitlines()
  allowed = {
      "engine/openvino/iq36-four-fc-horizontal-fusion.patch",
      "tools/intel-qwen36-openvino-fc-rms-igc-qk-rope-bundle-bound.py",
      "tools/intel-qwen36-openvino-four-fc-horizontal-fusion-source-gate.py",
      "tools/intel-qwen36-openvino-four-fc-horizontal-fusion-build.py",
  }
  relative_output = str(output.resolve().relative_to(ROOT))
  dirty = []
  for row in rows:
    path = row[3:]
    if path in allowed or path.startswith(relative_output):
      continue
    dirty.append(row)
  return {
      "commit": commit,
      "dirty": bool(dirty),
      "dirty_paths": dirty,
      "allowed_uncommitted_paths": sorted(allowed),
  }


def check(name: str, passed: bool, **detail: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **detail}


def parse_time_max_rss(path: Path) -> int:
  if not path.is_file():
    return 0
  match = re.search(
      r"Maximum resident set size \(kbytes\):\s*(\d+)",
      path.read_text(encoding="utf-8", errors="replace"))
  return int(match.group(1)) * 1024 if match else 0


def main() -> int:
  args = parse_args()
  output = args.output.resolve()
  raw = output / "raw"
  raw.mkdir(parents=True, exist_ok=False)
  required = (SOURCE_TREE, BUILD_TREE, PLUGIN, CONTROL_PLUGIN, PATCH, SOURCE_GATE)
  missing = [str(path) for path in required if not path.exists()]
  if missing:
    raise SystemExit("missing incremental-build inputs: " + ", ".join(missing))
  stop_bytes = int(args.memory_stop_gib * 1024**3)
  start_memory = proc_meminfo()
  if int(start_memory.get("MemAvailable", 0)) < stop_bytes:
    raise RuntimeError("memory stop tripped before incremental build")

  git = git_state(output)
  gate = load_json(SOURCE_GATE)
  plugin_before = {
      "path": str(PLUGIN), "sha256": sha256(PLUGIN),
      "size_bytes": PLUGIN.stat().st_size}
  control = {
      "path": str(CONTROL_PLUGIN), "sha256": sha256(CONTROL_PLUGIN),
      "size_bytes": CONTROL_PLUGIN.stat().st_size}
  target_diff = subprocess.run([
      "git", "diff", "--", TARGET_SOURCE, TARGET_TEST,
  ], cwd=SOURCE_TREE, check=True, capture_output=True, text=True).stdout
  patch_text = PATCH.read_text(encoding="utf-8")

  stdout_path = raw / "build.stdout"
  stderr_path = raw / "build.stderr"
  time_path = raw / "build.time"
  build_command = [
      "/home/intel/intel-box-env/conda/bin/cmake", "--build",
      str(BUILD_TREE), "--target", "openvino_intel_gpu_plugin",
      "--parallel", str(args.parallel)]
  command = ["/usr/bin/time", "-v", "-o", str(time_path), *build_command]
  monitor = {
      "process_group_rss_peak_bytes": 0,
      "process_group_swap_peak_bytes": 0,
      "process_count_peak": 0,
      "system_available_min_bytes": int(start_memory["MemAvailable"]),
      "system_swap_used_peak_bytes": (
          int(start_memory.get("SwapTotal", 0))
          - int(start_memory.get("SwapFree", 0))),
      "samples": 0,
  }
  started = time.monotonic()
  timed_out = False
  guard_tripped = False
  with stdout_path.open("w", encoding="utf-8") as stdout_handle, \
       stderr_path.open("w", encoding="utf-8") as stderr_handle:
    process = subprocess.Popen(
        command, cwd=ROOT, stdout=stdout_handle, stderr=stderr_handle,
        text=True, start_new_session=True)
    while process.poll() is None:
      elapsed = time.monotonic() - started
      if elapsed > args.timeout_s:
        timed_out = True
        stop_group(process, signal.SIGTERM)
        break
      system = proc_meminfo()
      group = process_group_memory(process.pid)
      available = int(system.get("MemAvailable", 0))
      swap_used = (int(system.get("SwapTotal", 0))
                   - int(system.get("SwapFree", 0)))
      monitor["samples"] += 1
      monitor["process_group_rss_peak_bytes"] = max(
          int(monitor["process_group_rss_peak_bytes"]), group["rss_bytes"])
      monitor["process_group_swap_peak_bytes"] = max(
          int(monitor["process_group_swap_peak_bytes"]), group["swap_bytes"])
      monitor["process_count_peak"] = max(
          int(monitor["process_count_peak"]), group["processes"])
      monitor["system_available_min_bytes"] = min(
          int(monitor["system_available_min_bytes"]), available)
      monitor["system_swap_used_peak_bytes"] = max(
          int(monitor["system_swap_used_peak_bytes"]), swap_used)
      if available < stop_bytes:
        guard_tripped = True
        stop_group(process, signal.SIGINT)
        break
      time.sleep(args.poll_interval_s)
    returncode = process.wait()
  elapsed_seconds = time.monotonic() - started
  time_max_rss = parse_time_max_rss(time_path)
  monitor["time_max_rss_bytes"] = time_max_rss
  monitor["process_rss_peak_bytes"] = max(
      int(monitor["process_group_rss_peak_bytes"]), time_max_rss)
  stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
  oom_observed = (
      not guard_tripped
      and (returncode in (-9, 137) or "out of memory" in stderr.lower()))
  plugin_after = {
      "path": str(PLUGIN), "sha256": sha256(PLUGIN),
      "size_bytes": PLUGIN.stat().st_size}

  checks = [
      check("repository_clean_at_gate", not git["dirty"], git=git),
      check("source_gate_admits_exactly_one_plugin_build",
            gate.get("required_checks_passed") is True
            and gate.get("plugin_build_admitted") is True
            and gate.get("gpu_context_admitted") is False),
      check("durable_patch_matches_applied_target_diff",
            target_diff == patch_text, patch_sha256=sha256(PATCH)),
      check("control_and_previous_candidate_identities_exact",
            control["sha256"] == EXPECTED_CONTROL_SHA256
            and plugin_before["sha256"] == EXPECTED_PREVIOUS_CANDIDATE_SHA256,
            control=control, previous_candidate=plugin_before),
      check("incremental_plugin_build_succeeded",
            returncode == 0 and not timed_out and not guard_tripped,
            returncode=returncode, timed_out=timed_out,
            memory_guard_tripped=guard_tripped),
      check("candidate_plugin_changed_and_is_nonempty",
            plugin_after["sha256"] != plugin_before["sha256"]
            and plugin_after["size_bytes"] > 0,
            candidate_plugin=plugin_after),
      check("build_process_used_zero_swap_and_no_oom",
            int(monitor["process_group_swap_peak_bytes"]) == 0
            and not oom_observed,
            monitor=monitor, oom_observed=oom_observed),
      check("four_gib_available_memory_stop_held",
            int(monitor["system_available_min_bytes"]) >= stop_bytes,
            stop_bytes=stop_bytes,
            available_min_bytes=monitor["system_available_min_bytes"]),
      check("no_gpu_context_or_model_worker_ran", True,
            gpu_contexts=0, model_workers=0),
  ]
  required_checks_passed = all(row["pass"] for row in checks)
  verdict = (
      "retain_four_fc_candidate_plugin_for_no_gpu_activation_preflight"
      if required_checks_passed else "incremental_build_failed")
  metrics = {
      "schema": SCHEMA,
      "created_at": datetime.now(timezone.utc).isoformat(),
      "workstream": WS,
      "git": git,
      "verdict": verdict,
      "required_checks_passed": required_checks_passed,
      "candidate_plugin_retained": required_checks_passed,
      "gpu_context_admitted": False,
      "model_worker_admitted": False,
      "build": {
          "command": build_command,
          "returncode": returncode,
          "elapsed_seconds": elapsed_seconds,
          "parallel": args.parallel,
          "timed_out": timed_out,
          "memory_guard_tripped": guard_tripped,
          "oom_observed": oom_observed,
          "monitor": monitor,
      },
      "control_plugin": control,
      "candidate_plugin_before": plugin_before,
      "candidate_plugin_after": plugin_after,
      "source_diff_sha256": hashlib.sha256(target_diff.encode()).hexdigest(),
      "next_action": {
          "route": "openvino_four_fc_horizontal_fusion_activation_preflight",
          "requirements": [
              "verify candidate plugin identity and source patch",
              "derive exact 371-to-161 runtime census and split expectations",
              "admit at most one guarded candidate-only 2k/17-step worker",
          ],
      },
      "checks": checks,
  }
  write_json(output / "metrics.json", metrics)
  write_json(output / "manifest.json", {
      "schema": SCHEMA,
      "tool": display(Path(__file__)),
      "git": git,
      "inputs": {display(path): sha256(path)
                 for path in (PATCH, SOURCE_GATE, CONTROL_PLUGIN)},
      "candidate_plugin": plugin_after,
      "compiler_builds": 1,
      "gpu_contexts": 0,
      "model_workers": 0,
  })
  report = f"""# Four-way compressed-FC incremental plugin build

Verdict: **{verdict}**. Required checks:
`{str(required_checks_passed).lower()}`.

The single admitted incremental GPU-plugin build completed in
`{elapsed_seconds:.2f} s`. Peak monitored RSS was
`{int(monitor['process_rss_peak_bytes']) / 1024:.0f} KiB`, process-group swap
was `{monitor['process_group_swap_peak_bytes']} B`, and minimum available
memory was `{monitor['system_available_min_bytes']} B`; the 4-GiB stop did not
trip and no OOM was observed.

The candidate plugin changed from `{plugin_before['sha256']}` to
`{plugin_after['sha256']}`. No GPU context or model worker ran. Retain only for
an exact activation preflight; there is no correctness or performance claim.
"""
  (output / "report.md").write_text(report, encoding="utf-8")
  print(json.dumps({
      "artifact": display(output),
      "verdict": verdict,
      "returncode": returncode,
      "elapsed_seconds": elapsed_seconds,
      "peak_rss_bytes": monitor["process_rss_peak_bytes"],
      "candidate_plugin_sha256": plugin_after["sha256"],
      "oom_observed": oom_observed,
  }, separators=(",", ":")), flush=True)
  return 0 if required_checks_passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
