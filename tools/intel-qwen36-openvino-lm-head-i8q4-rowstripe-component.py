#!/usr/bin/env python3
"""Build and gate low-bit OpenVINO LM-head decode rowstripe kernels.

The component consumes seq1454's 17 real decode hidden vectors, packs the
locked product IR's per-row I8 LM-head weight once, and executes a provider-
matching half/group-256 Q8 activation kernel plus either signed-Q4 or binary
two-centroid rowstripe matvec on the local PTL Level Zero device.  Q4 top-8
uses a global merge; binary top-12 corrects twelve rows per 256-row block.
Both recompute selected rows from the original I8 weight.  Repeat and confirm
are serial.  Existing carrier logits are read as the correctness oracle; no
stock, long-context, or concurrent GPU worker is launched.
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
SCHEMA = "intel-qwen36-openvino-lm-head-lowbit-rowstripe-component-v2"
MODEL_BIN = Path("/home/intel/Qwen3.6-35B-A3B-ov/openvino_language_model.bin")
CMAKE = Path("/home/intel/intel-box-env/conda/bin/cmake")
BUILD_DIR = REPO / "build/engine"
TARGET = "iq36-openvino-lm-head-i8q4-component"
BINARY = BUILD_DIR / TARGET
MODULE_SOURCE = REPO / "engine/gpu/opencl/q4x8_matvec.cl"
CPP_SOURCE = REPO / "engine/tools/openvino_lm_head_i8q4_component.cpp"
BOUNDARIES = REPO / "engine/boundaries.json"
BOUND = REPO / (
    "output/openvino-lm-head-codec-bound-20260718Tseq1454-"
    "decode-q8-real-hidden-clean/metrics.json")
CAPTURE_DIR = REPO / (
    "output/openvino-lm-head-codec-bound-20260718Tseq1454-"
    "decode-q8-real-hidden-clean/raw/2k/candidate-pruned-head")
PROFILE_WORKER = REPO / (
    "output/openvino-accepted-carrier-profile-refresh-20260718Tseq1452-"
    "clean-all-alias-2k-warm17/raw/2k/candidate/worker-result.json")
KLD_LIMIT = 0.005
TOPK_KLD_LIMIT = 0.001
Q4_PACKED_BYTES = 255_272_960
BINARY_PACKED_BYTES = 66_053_120
RATE_NOISE_FRACTION = 0.005


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--bound", type=Path, default=BOUND)
  parser.add_argument("--capture-dir", type=Path, default=CAPTURE_DIR)
  parser.add_argument("--profile-worker", type=Path, default=PROFILE_WORKER)
  parser.add_argument(
      "--reference-dir", type=Path,
      help="directory containing stepNNNN-logits.f32 correctness oracles")
  parser.add_argument("--cmake", type=Path, default=CMAKE)
  parser.add_argument("--build-dir", type=Path, default=BUILD_DIR)
  parser.add_argument("--warmup", type=int, default=7)
  parser.add_argument("--samples", type=int, default=11)
  parser.add_argument("--topk", type=int, choices=(0, 2, 8, 12), default=0)
  parser.add_argument("--binary", action="store_true")
  parser.add_argument("--last-phase", type=int, default=17)
  parser.add_argument("--timeout-s", type=int, default=300)
  parser.add_argument("--min-available-gib", type=float, default=8.0)
  parser.add_argument("--abort-below-available-gib", type=float, default=4.0)
  parser.add_argument("--poll-interval-s", type=float, default=0.1)
  args = parser.parse_args()
  if args.warmup < 0 or args.samples <= 0 or args.timeout_s <= 0:
    parser.error("warmup must be nonnegative; samples and timeout positive")
  if args.min_available_gib < 0.0 or args.abort_below_available_gib < 0.0:
    parser.error("memory thresholds must be nonnegative")
  if args.abort_below_available_gib > args.min_available_gib:
    parser.error("abort threshold must not exceed preflight threshold")
  if args.poll_interval_s <= 0.0:
    parser.error("poll interval must be positive")
  if args.last_phase < 1 or args.last_phase > 4096:
    parser.error("last-phase must be in [1,4096]")
  if args.binary and args.topk not in (0, 2, 12):
    parser.error("binary mode supports topk 0 or block-local topk 2/12")
  if not args.binary and args.topk not in (0, 8):
    parser.error("Q4 mode supports topk 0 or global topk 8")
  return args


def hidden_path(directory: Path, phase: int) -> Path:
  legacy = directory / f"phase{phase}-lm-head-input.f32"
  return legacy if legacy.is_file() else directory / (
      f"step{phase:04d}-lm-head-input.f32")


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
    return {"command": command, "returncode": result.returncode,
            "stdout": result.stdout, "stderr": result.stderr,
            "timed_out": False}
  except subprocess.TimeoutExpired as error:
    return {"command": command, "returncode": 124,
            "stdout": str(error.stdout or ""),
            "stderr": str(error.stderr or "") + "\ntimeout",
            "timed_out": True}


def check(name: str, passed: bool, **details: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **details}


def git_state(output: Path) -> dict[str, Any]:
  commit = run(["git", "rev-parse", "HEAD"], 30)["stdout"].strip()
  rows = run(["git", "status", "--porcelain"], 30)["stdout"].splitlines()
  try:
    output_relative = str(output.resolve().relative_to(REPO))
  except ValueError:
    output_relative = ""
  rows = [row for row in rows
          if not output_relative or output_relative not in row]
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
  for line in path.read_text(
      encoding="utf-8", errors="replace").splitlines():
    if ":" not in line:
      continue
    key, value = line.split(":", 1)
    fields = value.split()
    if key in ("VmRSS", "VmSwap") and fields:
      selected[key] = int(fields[0]) * 1024
  return {"VmRSS": selected.get("VmRSS", 0),
          "VmSwap": selected.get("VmSwap", 0)}


def wait_for_memory(required_bytes: int) -> dict[str, Any]:
  started = time.monotonic()
  while True:
    available = int(meminfo().get("MemAvailable", 0))
    if available >= required_bytes:
      return {"available_bytes": available, "required_bytes": required_bytes,
              "waited_seconds": time.monotonic() - started}
    if time.monotonic() - started > 60.0:
      raise RuntimeError(
          f"available memory {available} remains below {required_bytes}")
    time.sleep(2.0)


def stop_process_group(process: subprocess.Popen[Any], first_signal: int) -> None:
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
    if (("intel-qwen36-openvino-hot-cold-attention-gate.py" in command and
         "--worker-config" in command) or TARGET in command):
      rows.append({"pid": int(path.name), "command": command.strip()})
  return rows


def launch_component(
    args: argparse.Namespace, label: str, directory: Path, module: Path,
) -> dict[str, Any]:
  directory.mkdir(parents=True)
  stdout_path = directory / "component.stdout"
  stderr_path = directory / "component.stderr"
  command = [
      str(BINARY), str(MODEL_BIN), str(module), str(args.capture_dir),
      str(directory), str(args.warmup), str(args.samples),
  ]
  if args.topk:
    command.append(str(args.topk))
  preflight = wait_for_memory(int(args.min_available_gib * 1024**3))
  started = time.monotonic()
  abort_bytes = int(args.abort_below_available_gib * 1024**3)
  monitor: dict[str, Any] = {
      "process_rss_peak_bytes": 0, "process_swap_peak_bytes": 0,
      "system_available_min_bytes": None,
      "system_swap_used_peak_bytes": 0, "sample_count": 0,
  }
  timed_out = False
  guard_tripped = False
  environment = os.environ.copy()
  environment["IQ36_LM_HEAD_LAST_PHASE"] = str(args.last_phase)
  if args.binary:
    environment["IQ36_LM_HEAD_BINARY"] = "1"
  with stdout_path.open("w", encoding="utf-8") as stdout_handle, \
       stderr_path.open("w", encoding="utf-8") as stderr_handle:
    process = subprocess.Popen(
        command, cwd=REPO, stdout=stdout_handle, stderr=stderr_handle,
        text=True, start_new_session=True, env=environment)
    while process.poll() is None:
      if time.monotonic() - started > args.timeout_s:
        timed_out = True
        stop_process_group(process, signal.SIGTERM)
        break
      system = meminfo()
      process_row = process_memory(process.pid)
      available = int(system.get("MemAvailable", 0))
      swap_used = int(system.get("SwapTotal", 0)) - int(
          system.get("SwapFree", 0))
      monitor["sample_count"] = int(monitor["sample_count"]) + 1
      monitor["process_rss_peak_bytes"] = max(
          int(monitor["process_rss_peak_bytes"]), process_row["VmRSS"])
      monitor["process_swap_peak_bytes"] = max(
          int(monitor["process_swap_peak_bytes"]), process_row["VmSwap"])
      current_min = monitor["system_available_min_bytes"]
      monitor["system_available_min_bytes"] = (
          available if current_min is None else min(int(current_min), available))
      monitor["system_swap_used_peak_bytes"] = max(
          int(monitor["system_swap_used_peak_bytes"]), swap_used)
      if abort_bytes and available < abort_bytes:
        guard_tripped = True
        stop_process_group(process, signal.SIGINT)
        break
      time.sleep(args.poll_interval_s)
    returncode = process.wait()
  stdout = stdout_path.read_text(encoding="utf-8", errors="replace")
  stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
  try:
    parsed = json.loads(stdout.strip())
  except json.JSONDecodeError:
    parsed = {}
  oom = (not guard_tripped and
         (returncode in (-9, 137) or "out of memory" in stderr.lower() or
          "ze_result_error_out_of" in stderr.lower()))
  return {
      "label": label, "command": command, "returncode": returncode,
      "environment": {
          "IQ36_LM_HEAD_BINARY": "1" if args.binary else None,
          "IQ36_LM_HEAD_LAST_PHASE": str(args.last_phase),
      },
      "timed_out": timed_out,
      "memory_guard": {"abort_below_bytes": abort_bytes,
                       "tripped": guard_tripped},
      "memory_preflight": preflight, "monitor": monitor,
      "oom_observed": oom, "elapsed_seconds": time.monotonic() - started,
      "result": parsed if isinstance(parsed, dict) else {},
  }


def distribution_metrics(reference: Any, candidate: Any, np: Any) -> dict[str, Any]:
  ref = np.asarray(reference, dtype=np.float64).reshape(-1)
  cand = np.asarray(candidate, dtype=np.float64).reshape(-1)
  if ref.shape != cand.shape:
    return {"shape_match": False, "finite": False,
            "kld_reference_to_candidate": math.inf,
            "top1_match": False}
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
      "finite": bool(np.isfinite(ref).all() and np.isfinite(cand).all()),
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


def audit_component(
    launched: dict[str, Any], phases: tuple[int, ...],
    profile_worker: dict[str, Any] | None, reference_dir: Path | None,
    np: Any,
) -> dict[str, Any]:
  result = launched["result"]
  phase_by_id = {int(row["phase"]): row for row in result.get("phases", [])}
  comparisons = []
  output_hashes = []
  for phase in phases:
    row = phase_by_id.get(phase, {})
    output = Path(str(row.get("output", "")))
    if output and not output.is_absolute():
      output = REPO / output
    reference_path = (
        reference_dir / f"step{phase:04d}-logits.f32"
        if reference_dir is not None else
        REPO / profile_worker["phases"][phase]["logits_path"])
    reference = np.fromfile(reference_path, dtype="<f4")
    candidate = (
        np.fromfile(output, dtype="<f4") if output.is_file() else
        np.asarray([], dtype=np.float32))
    comparisons.append({
        "phase": phase,
        **distribution_metrics(reference, candidate, np),
    })
    output_hashes.append({
        "phase": phase, "path": display_path(output),
        "sha256": sha256(output) if output.is_file() else None,
    })
  timing_rows = [phase_by_id.get(phase, {}) for phase in phases]
  return {
      "comparisons": comparisons,
      "output_hashes": output_hashes,
      "top1_matches": sum(row.get("top1_match") is True for row in comparisons),
      "max_kld": max(
          float(row.get("kld_reference_to_candidate", math.inf))
          for row in comparisons),
      "max_relative_l2": max(
          float(row.get("relative_l2", math.inf)) for row in comparisons),
      "max_abs": max(float(row.get("max_abs", math.inf))
                     for row in comparisons),
      "phase_count": len(timing_rows),
      "packed_bytes": result.get("packed_bytes"),
      "pack_ms": result.get("pack_ms"),
      "device_name": result.get("device_name"),
      "topk": result.get("topk"),
      "source_checks_passed": result.get("required_checks_passed"),
      "min_effective_packed_gb_s": min(
          float(row.get("effective_packed_gb_s", -math.inf))
          for row in timing_rows),
      "median_effective_packed_gb_s": statistics.median(
          float(row.get("effective_packed_gb_s", math.nan))
          for row in timing_rows),
      "max_shell_min_us": max(
          float(row.get("shell_min_us", math.inf)) for row in timing_rows),
      "median_shell_min_us": statistics.median(
          float(row.get("shell_min_us", math.nan)) for row in timing_rows),
      "max_q8_min_us": max(
          float(row.get("q8_min_us", math.inf)) for row in timing_rows),
      "max_topk_merge_min_us": max(
          float(row.get("topk_merge_min_us", math.inf))
          for row in timing_rows),
      "max_correction_min_us": max(
          float(row.get("correction_min_us", math.inf))
          for row in timing_rows),
      "max_wall_min_us": max(
          float(row.get("wall_min_us", math.inf)) for row in timing_rows),
      "phase_timing": timing_rows,
  }


def write_summary(output: Path, result: dict[str, Any]) -> None:
  repeat = result["repeat_audit"]
  confirm = result["confirm_audit"]
  phase_count = int(repeat["phase_count"])
  codec = "binary two-centroid" if result["binary"] else "signed-Q4"
  correction_scope = (
      f"{repeat['topk']} per 256-row block"
      if result["binary"] and repeat["topk"] else str(repeat["topk"]))
  lines = [
      f"# OpenVINO LM-head {codec} rowstripe component",
      "",
      f"- verdict: `{result['verdict']}`",
      f"- required checks passed: `{str(result['required_checks_passed']).lower()}`",
      f"- packed weight: `{repeat['packed_bytes']}` bytes",
      f"- dynamic exact correction rows: `{correction_scope}`",
      f"- repeat/confirm max KLD: `{repeat['max_kld']}` / `{confirm['max_kld']}`",
      f"- repeat/confirm top-1: `{repeat['top1_matches']}/{phase_count}` / `{confirm['top1_matches']}/{phase_count}`",
      f"- repeat/confirm minimum effective rate: `{repeat['min_effective_packed_gb_s']:.3f}` / `{confirm['min_effective_packed_gb_s']:.3f} GB/s`",
      f"- repeat/confirm worst phase kernel shell: `{repeat['max_shell_min_us']:.3f}` / `{confirm['max_shell_min_us']:.3f} us`",
      f"- repeat/confirm worst top-K merge: `{repeat['max_topk_merge_min_us']:.3f}` / `{confirm['max_topk_merge_min_us']:.3f} us`",
      f"- repeat/confirm worst I8 correction: `{repeat['max_correction_min_us']:.3f}` / `{confirm['max_correction_min_us']:.3f} us`",
      f"- component shell limit from the complete kill-number: `{result['admission']['shell_max_us']:.3f} us`",
      f"- OOM/guard: `{result['repeat_worker']['oom_observed']}` / `{result['repeat_worker']['memory_guard']['tripped']}`, `{result['confirm_worker']['oom_observed']}` / `{result['confirm_worker']['memory_guard']['tripped']}`",
      "",
      "This admits a full-vocabulary T=1 kernel component only. Prefill remains "
      "on the existing I8 provider; no 32k/output512 or product speedup is "
      "claimed.",
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
  args.profile_worker = args.profile_worker.resolve()
  if args.reference_dir is not None:
    args.reference_dir = args.reference_dir.resolve()
  args.build_dir = args.build_dir.resolve()
  decode_phases = tuple(range(1, args.last_phase + 1))
  required = [
      MODEL_BIN, args.cmake, MODULE_SOURCE, CPP_SOURCE, BOUNDARIES,
      args.bound,
      *[hidden_path(args.capture_dir, phase) for phase in decode_phases],
  ]
  if args.reference_dir is None:
    required.append(args.profile_worker)
  else:
    required.extend(
        args.reference_dir / f"step{phase:04d}-logits.f32"
        for phase in decode_phases)
  missing = [str(path) for path in required if not path.is_file()]
  if missing:
    raise SystemExit("missing component inputs: " + ", ".join(missing))

  state = git_state(output)
  concurrent = other_gpu_workers()
  if concurrent:
    raise RuntimeError(f"concurrent GPU worker detected: {concurrent}")
  bound = load_json(args.bound)
  profile_worker = (
      load_json(args.profile_worker) if args.reference_dir is None else None)
  profile_ms = float(
      bound["bandwidth_bound"]["profile_ms_nonadditive_source_bound"])
  kill_number_ms = float(
      bound["bandwidth_bound"]["kill_number_ms_per_token"])
  shell_max_us = (profile_ms - kill_number_ms) * 1000.0
  conservative_gb_s = float(
      bound["bandwidth_bound"]["conservative_bandwidth_gbps"])
  effective_rate_min = conservative_gb_s * (1.0 - RATE_NOISE_FRACTION)
  expected_packed_bytes = (
      BINARY_PACKED_BYTES if args.binary else Q4_PACKED_BYTES)
  component_kld_limit = (
      TOPK_KLD_LIMIT if not args.binary and args.topk == 8 else KLD_LIMIT)

  module_compile = run([
      "ocloc", "compile", "-file", str(MODULE_SOURCE), "-device", "0xb080",
      "-output", "iq36_q4x8_all", "-out_dir", str(generated),
      "-output_no_suffix", "--format", "zebin", "-options",
      "-cl-std=CL3.0 -D IQ36_USE_INTEGER_DOT=1", "-q",
  ])
  module = generated / "iq36_q4x8_all.bin"
  module_validate = (
      run(["ocloc", "validate", "-file", str(module)], 60)
      if module.is_file() else
      {"command": [], "returncode": 125, "stdout": "",
       "stderr": "module missing", "timed_out": False})
  configure = run([
      str(args.cmake), "-S", str(REPO / "engine"), "-B",
      str(args.build_dir), "-DCMAKE_BUILD_TYPE=Release",
  ])
  build = (
      run([str(args.cmake), "--build", str(args.build_dir),
           "--target", TARGET, "-j", "1"], 300)
      if configure["returncode"] == 0 else
      {"command": [], "returncode": 125, "stdout": "",
       "stderr": "configure failed", "timed_out": False})
  link_map = (
      run(["ldd", str(BINARY)], 30) if BINARY.is_file() else
      {"command": [], "returncode": 125, "stdout": "",
       "stderr": "binary missing", "timed_out": False})

  repeat = (
      launch_component(args, "repeat", output / "repeat", module)
      if module_validate["returncode"] == 0 and build["returncode"] == 0 else
      {"returncode": 125, "timed_out": False,
       "memory_guard": {"tripped": False}, "oom_observed": False,
       "result": {}})
  repeat_audit = audit_component(
      repeat, decode_phases, profile_worker, args.reference_dir, np)
  confirm = (
      launch_component(args, "confirm", output / "confirm", module)
      if repeat["returncode"] == 0 else
      {"returncode": 125, "timed_out": False,
       "memory_guard": {"tripped": False}, "oom_observed": False,
       "result": {}})
  confirm_audit = audit_component(
      confirm, decode_phases, profile_worker, args.reference_dir, np)

  output_hash_match = all(
      lhs["sha256"] is not None and lhs["sha256"] == rhs["sha256"]
      for lhs, rhs in zip(
          repeat_audit["output_hashes"], confirm_audit["output_hashes"]))
  median_shell_delta = abs(
      repeat_audit["median_shell_min_us"] -
      confirm_audit["median_shell_min_us"])
  median_shell_relative_delta = median_shell_delta / min(
      repeat_audit["median_shell_min_us"],
      confirm_audit["median_shell_min_us"])
  lower_links = (link_map.get("stdout", "") +
                 link_map.get("stderr", "")).lower()
  checks = [
      check("repository_clean_at_gate", not state["dirty"], git=state),
      check("no_concurrent_gpu_worker_at_launch", not concurrent,
            concurrent=concurrent),
      check("codec_bound_funds_decode_only_lm_head",
            bound.get("required_checks_passed") is True and
            bound.get("selected_codec") == 4 and
            bound.get("provider_scope") ==
                "decode T=1 only; retain existing I8 prefill provider"),
      check("module_compiles_and_validates_with_integer_dot",
            module_compile["returncode"] == 0 and
            module_validate["returncode"] == 0 and module.is_file()),
      check("component_builds_serially",
            configure["returncode"] == 0 and build["returncode"] == 0 and
            BINARY.is_file()),
      check("repeat_and_confirm_complete_without_oom",
            all(worker["returncode"] == 0 and not worker["timed_out"] and
                not worker["memory_guard"]["tripped"] and
                not worker["oom_observed"] for worker in (repeat, confirm)),
            repeat={key: repeat.get(key) for key in (
                "returncode", "timed_out", "memory_guard", "oom_observed",
                "monitor")},
            confirm={key: confirm.get(key) for key in (
                "returncode", "timed_out", "memory_guard", "oom_observed",
                "monitor")}),
      check("full_vocab_outputs_are_deterministic",
            output_hash_match,
            repeat=repeat_audit["output_hashes"],
            confirm=confirm_audit["output_hashes"]),
      check("repeat_and_confirm_real_decode_distributions_pass",
            all(audit["phase_count"] == len(decode_phases) and
                audit["source_checks_passed"] is True and
                audit["top1_matches"] == len(decode_phases) and
                audit["max_kld"] <= component_kld_limit
                for audit in (repeat_audit, confirm_audit)),
            repeat={key: repeat_audit[key] for key in (
                "phase_count", "top1_matches", "max_kld",
                "max_relative_l2", "max_abs")},
            confirm={key: confirm_audit[key] for key in (
                "phase_count", "top1_matches", "max_kld",
                "max_relative_l2", "max_abs")}),
      check("requested_dynamic_correction_scope_is_exact",
            all(audit["topk"] == args.topk
                for audit in (repeat_audit, confirm_audit)),
            requested_topk=args.topk,
            repeat_topk=repeat_audit["topk"],
            confirm_topk=confirm_audit["topk"]),
      check("packed_layout_and_plain_rate_clear_bound",
            all(audit["packed_bytes"] == expected_packed_bytes and
                (args.topk != 0 or
                 audit["min_effective_packed_gb_s"] >= effective_rate_min)
                for audit in (repeat_audit, confirm_audit)),
            required_gb_s=effective_rate_min,
            repeat_min_gb_s=repeat_audit["min_effective_packed_gb_s"],
            confirm_min_gb_s=confirm_audit["min_effective_packed_gb_s"]),
      check("every_phase_kernel_shell_covers_complete_kill_number",
            all(audit["max_shell_min_us"] <= shell_max_us
                for audit in (repeat_audit, confirm_audit)),
            shell_max_us=shell_max_us,
            repeat_max_shell_us=repeat_audit["max_shell_min_us"],
            confirm_max_shell_us=confirm_audit["max_shell_min_us"],
            kill_number_ms=kill_number_ms),
      check("activation_quantize_is_not_the_new_bottleneck",
            all(audit["max_q8_min_us"] <= 50.0
                for audit in (repeat_audit, confirm_audit)),
            repeat_max_q8_us=repeat_audit["max_q8_min_us"],
            confirm_max_q8_us=confirm_audit["max_q8_min_us"]),
      check("repeat_confirm_kernel_shell_is_stable",
            median_shell_relative_delta <= 0.02,
            median_shell_relative_delta=median_shell_relative_delta),
      check("level_zero_dependency_boundary",
            link_map["returncode"] == 0 and "libze_loader" in lower_links and
            "openvino" not in lower_links and "libdnnl" not in lower_links),
  ]
  passed = all(row["pass"] for row in checks)
  codec_label = "binary_two_centroid" if args.binary else "signed4"
  correction_label = (
      f"_local_top{args.topk}_exact" if args.binary and args.topk else
      f"_top{args.topk}_exact" if args.topk else "")
  verdict = (
      "admit" if passed else "reject") + (
          f"_{codec_label}{correction_label}_decode_lm_head_component")
  result = {
      "schema": SCHEMA,
      "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
      "workstream": WS, "git": state, "verdict": verdict,
      "required_checks_passed": passed,
      "binary": args.binary,
      "topk_exact_correction": args.topk,
      "provider_scope": "decode T=1 full vocabulary; existing I8 prefill",
      "gpu_workers_launched": 2,
      "stock_worker_launched": False,
      "concurrent_worker_launched": False,
      "long_worker_launched": False,
      "admission": {
          "kld_max": KLD_LIMIT,
          "component_kld_max": component_kld_limit,
          "top1_required": len(decode_phases),
          "conservative_rate_gb_s": conservative_gb_s,
          "effective_rate_noise_floor_gb_s": effective_rate_min,
          "profile_lm_head_ms_nonadditive_source_bound": profile_ms,
          "kill_number_ms_per_token": kill_number_ms,
          "shell_max_us": shell_max_us,
      },
      "repeat_audit": repeat_audit,
      "confirm_audit": confirm_audit,
      "repeat_worker": {key: value for key, value in repeat.items()
                        if key != "result"},
      "confirm_worker": {key: value for key, value in confirm.items()
                         if key != "result"},
      "build": {"module_compile": module_compile,
                "module_validate": module_validate,
                "configure": configure, "build": build,
                "link_map": link_map},
      "inputs": {
          "bound": display_path(args.bound),
          "bound_sha256": sha256(args.bound),
          "capture_dir": display_path(args.capture_dir),
          "last_phase": args.last_phase,
          "profile_worker": (
              display_path(args.profile_worker)
              if args.reference_dir is None else None),
          "profile_worker_sha256": (
              sha256(args.profile_worker)
              if args.reference_dir is None else None),
          "reference_dir": (
              display_path(args.reference_dir)
              if args.reference_dir is not None else None),
          "reference_hashes": (
              {f"step{phase:04d}": sha256(
                  args.reference_dir / f"step{phase:04d}-logits.f32")
               for phase in decode_phases}
              if args.reference_dir is not None else None),
          "model_bin": str(MODEL_BIN),
          "sources": {
              display_path(path): sha256(path)
              for path in (MODULE_SOURCE, CPP_SOURCE, BOUNDARIES)},
          "module": display_path(module),
          "module_sha256": sha256(module) if module.is_file() else None,
      },
      "checks": checks,
      "product_promotion_ready": False,
      "speedup_claims_allowed": False,
  }
  write_json(output / "metrics.json", result)
  write_json(output / "manifest.json", {
      "schema": SCHEMA, "git": state, "verdict": verdict,
      "required_checks_passed": passed,
      "tool": display_path(Path(__file__)),
      "module_source": display_path(MODULE_SOURCE),
      "cpp_source": display_path(CPP_SOURCE),
      "packed_bytes": repeat_audit["packed_bytes"],
      "binary": args.binary,
      "topk_exact_correction": args.topk,
      "last_phase": args.last_phase,
      "gpu_workers_launched": 2,
      "memory_preflight_gib": args.min_available_gib,
      "memory_abort_gib": args.abort_below_available_gib,
      "product_promotion_ready": False,
      "speedup_claims_allowed": False,
  })
  write_summary(output, result)
  print(json.dumps({
      "output": display_path(output), "verdict": verdict,
      "required_checks_passed": passed,
      "topk_exact_correction": args.topk,
      "repeat_max_kld": repeat_audit["max_kld"],
      "confirm_max_kld": confirm_audit["max_kld"],
      "repeat_min_gb_s": repeat_audit["min_effective_packed_gb_s"],
      "confirm_min_gb_s": confirm_audit["min_effective_packed_gb_s"],
      "repeat_max_shell_us": repeat_audit["max_shell_min_us"],
      "confirm_max_shell_us": confirm_audit["max_shell_min_us"],
      "repeat_max_topk_merge_us": repeat_audit["max_topk_merge_min_us"],
      "confirm_max_topk_merge_us": confirm_audit["max_topk_merge_min_us"],
      "repeat_max_correction_us": repeat_audit["max_correction_min_us"],
      "confirm_max_correction_us": confirm_audit["max_correction_min_us"],
  }, sort_keys=True))
  return 0 if passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
