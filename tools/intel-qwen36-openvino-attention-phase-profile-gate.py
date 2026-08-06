#!/usr/bin/env python3
"""Bind exact phase timing and live OpenCL codegen for stock/custom attention.

The gate runs isolated stock and candidate workers on selected exact context-
ladder sentinel prompts. OpenCL event timing is enabled only while each
InferRequest is active, and every dispatch carries a phase marker. This is an
attribution and codegen gate: it records, but does not waive, the tiled-carrier
performance admission rule.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import statistics
import subprocess
from pathlib import Path
from typing import Any

from iq36_perf_inference import latency_cap_inference


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "intel-qwen36-openvino-attention-phase-profile-gate-v0"
WORKER = ROOT / "tools/intel-qwen36-openvino-hot-cold-attention-gate.py"
OV_PYTHON = Path("/home/intel/ov/openvino_env/bin/python")
MODEL_DIR = Path("/home/intel/Qwen3.6-35B-A3B-ov")
PROMPT_DIR = (
    ROOT / "output/r0-oracle-prompt-materialization-20260626T082201Z/"
    "prompts")
PROMPTS = {
    "2k": PROMPT_DIR / "sentinel_002k.txt",
    "4k": PROMPT_DIR / "sentinel_004k.txt",
    "8k": PROMPT_DIR / "sentinel_008k.txt",
    "16k": PROMPT_DIR / "sentinel_016k.txt",
    "32k": PROMPT_DIR / "sentinel_032k.txt",
    "64k": PROMPT_DIR / "sentinel_064k.txt",
    "128k": PROMPT_DIR / "sentinel_128k.txt",
}
DECODE_STEPS = {lane: 1 for lane in PROMPTS}
DECODE_STEPS["8k"] = 2
PREFILL_CHUNK_TOKENS = 8192
CUSTOM_CONFIG = ROOT / "engine/openvino/custom/iq36_hot_attention_gqa.xml"
TARGET_LAYERS = tuple(range(3, 40, 4))
CMAKE = Path("/home/intel/intel-box-env/conda/bin/cmake")
BUILD_DIR = ROOT / "build/engine"
TRACE_TARGET = "iq36-opencl-dispatch-trace"
TRACE_LIBRARY = BUILD_DIR / "iq36-opencl-dispatch-trace.so"
DIRECT_I8_FIXED_COLD_CAPACITY = 32768
DIRECT_I8_PREFILL_HISTORY_CAPACITY = 16384
FINE_CODEC_COMPONENT_CAP_MS = 0.5618915
FINE_CODEC_PROFILE_SKIP_STEPS = 5
FINE_CODEC_PROFILE_MINIMUM_SAMPLES = 20


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
  parser.add_argument("--custom-config", type=Path, default=CUSTOM_CONFIG)
  parser.add_argument("--device", default="GPU")
  parser.add_argument("--openvino-python", type=Path, default=OV_PYTHON)
  parser.add_argument(
      "--candidate-gpu-plugin", type=Path,
      help=("candidate-only OpenVINO GPU plugin; an explicit plugin registry "
            "keeps it isolated from the stock worker"))
  parser.add_argument(
      "--candidate-gpu-usm-policy", type=int, choices=(0, 1, 2),
      help=("candidate-only OV_GPU_USM_POLICY value; requires "
            "--candidate-gpu-plugin"))
  parser.add_argument(
      "--graph-initialized-hot-states", action="store_true",
      help=("initialize custom hot ReadValue storage in the graph and skip "
            "the diagnostic request-state self-assignment"))
  parser.add_argument(
      "--dump-runtime-graph", action="store_true",
      help="serialize each compiled GPU execution graph")
  parser.add_argument(
      "--capture-full-profile", action="store_true",
      help=("save all OpenVINO profiling rows for prefill and the final "
            "decode step"))
  parser.add_argument(
      "--fuse-linear-conv-state", action="store_true",
      help=("profile the accepted 30-layer fused linear conv/state/SiLU "
            "candidate boundary"))
  parser.add_argument(
      "--pack-gdn-state", action="store_true",
      help="profile the candidate-only coalesced [V,K] GDN state carrier")
  parser.add_argument(
      "--decode-steps", type=int,
      help=("override the lane-default number of teacher-forced decode steps; "
            "useful for capturing a warmed final decode profile"))
  parser.add_argument(
      "--lanes", default="2k,8k",
      help=("comma-separated subset of 2k,4k,8k,16k,32k,64k,128k "
            "(default: 2k,8k)"))
  parser.add_argument(
      "--target-layers", default=",".join(map(str, TARGET_LAYERS)),
      help="comma-separated full-attention layer subset")
  parser.add_argument(
      "--phase-branch-prefill", action="store_true",
      help="profile the experimental prefill/decode If carrier")
  parser.add_argument(
      "--stock-prefill-custom-decode", action="store_true",
      help="profile stock SDPA prefill plus state-only/custom decode")
  parser.add_argument(
      "--static-phase-separated", action="store_true",
      help=("compile stock-prefill/state-update and custom-decode graphs "
            "separately, then transfer request state once"))
  parser.add_argument(
      "--stock-prefill-sliced-decode", action="store_true",
      help=("run stock prefill SDPA and reduce its unused decode history to "
            "one token beside the custom decode"))
  parser.add_argument(
      "--direct-i8-group4-full-cold", action="store_true",
      help=("profile the correctness-promoted one-layer fixed direct-I8 "
            "group-4/full-cold integration"))
  parser.add_argument(
      "--direct-i8-hybrid-k2-v4", action="store_true",
      help=("profile the correctness-promoted one-layer fixed direct-I8 "
            "K2/V4 full-cold integration"))
  parser.add_argument("--out-dir", type=Path)
  parser.add_argument("--timeout-s", type=int, default=1800)
  parser.add_argument("--memory-stop-gib", type=float, default=4.0)
  args = parser.parse_args()
  if args.timeout_s <= 0:
    parser.error("timeout-s must be positive")
  if args.memory_stop_gib <= 0.0:
    parser.error("memory-stop-gib must be positive")
  if args.decode_steps is not None and args.decode_steps <= 0:
    parser.error("decode-steps must be positive")
  if sum((args.phase_branch_prefill,
          args.stock_prefill_custom_decode,
          args.static_phase_separated,
          args.stock_prefill_sliced_decode)) > 1:
    parser.error("phase composition modes are mutually exclusive")
  if (args.candidate_gpu_usm_policy is not None and
      args.candidate_gpu_plugin is None):
    parser.error(
        "candidate-gpu-usm-policy requires candidate-gpu-plugin")
  if (args.candidate_gpu_plugin is not None and
      not args.candidate_gpu_plugin.is_file()):
    parser.error(
        f"candidate GPU plugin does not exist: {args.candidate_gpu_plugin}")
  if args.pack_gdn_state and args.candidate_gpu_plugin is None:
    parser.error("pack-gdn-state requires candidate-gpu-plugin")
  args.lanes = tuple(item.strip() for item in args.lanes.split(",")
                     if item.strip())
  if not args.lanes or any(item not in PROMPTS for item in args.lanes):
    parser.error(
        "lanes must be a comma-separated subset of "
        "2k,4k,8k,16k,32k,64k,128k")
  try:
    args.target_layers = tuple(
        int(item.strip()) for item in args.target_layers.split(",")
        if item.strip())
  except ValueError:
    parser.error("target-layers must contain comma-separated integers")
  if (not args.target_layers or
      len(set(args.target_layers)) != len(args.target_layers) or
      any(layer not in TARGET_LAYERS for layer in args.target_layers)):
    parser.error(f"target-layers must be a unique subset of {TARGET_LAYERS}")
  if args.direct_i8_group4_full_cold and args.direct_i8_hybrid_k2_v4:
    parser.error("direct-I8 fine-codec profiles are mutually exclusive")
  if args.direct_i8_group4_full_cold or args.direct_i8_hybrid_k2_v4:
    if args.lanes != ("32k",) or len(args.target_layers) != 1:
      parser.error(
          "direct-I8 fine-codec profile requires one layer and 32k")
    if (args.decode_steps is None or
        args.decode_steps <
            FINE_CODEC_PROFILE_SKIP_STEPS +
            FINE_CODEC_PROFILE_MINIMUM_SAMPLES):
      parser.error(
          "direct-I8 fine-codec profile requires at least 25 decode steps")
    if any((args.phase_branch_prefill,
            args.stock_prefill_custom_decode,
            args.static_phase_separated,
            args.stock_prefill_sliced_decode)):
      parser.error(
          "direct-I8 fine-codec profile requires unified phase composition")
  if args.out_dir is None:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.out_dir = ROOT / f"output/openvino-attention-phase-profile-{stamp}"
  return args


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise ValueError(f"{path}: expected JSON object")
  return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
  if not path.is_file():
    return []
  rows = []
  for number, line in enumerate(
      path.read_text(encoding="utf-8").splitlines(), start=1):
    if not line.strip():
      continue
    value = json.loads(line)
    if not isinstance(value, dict):
      raise ValueError(f"{path}:{number}: expected JSON object")
    rows.append(value)
  return rows


def sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def relative(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path.resolve())


def git_state(out_dir: Path) -> dict[str, Any]:
  def git(*items: str) -> str:
    run = subprocess.run(
        ["git", *items], cwd=ROOT, check=False, capture_output=True,
        text=True, encoding="utf-8", errors="replace")
    return run.stdout.strip() if run.returncode == 0 else ""

  dirty = git("status", "--porcelain").splitlines()
  try:
    out_relative = str(out_dir.relative_to(ROOT))
  except ValueError:
    out_relative = ""
  dirty = [row for row in dirty
           if not out_relative or out_relative not in row]
  return {
      "commit": git("rev-parse", "HEAD"),
      "dirty": bool(dirty),
      "dirty_paths": dirty,
  }


def available_memory_bytes() -> int:
  for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
    if line.startswith("MemAvailable:"):
      return int(line.split()[1]) * 1024
  raise RuntimeError("/proc/meminfo does not contain MemAvailable")


def command(
    items: list[str], *, timeout_s: int,
    environment: dict[str, str] | None = None,
) -> dict[str, Any]:
  try:
    run = subprocess.run(
        items, cwd=ROOT, env=environment, check=False, capture_output=True,
        text=True, encoding="utf-8", errors="replace", timeout=timeout_s)
    return {
        "command": items,
        "returncode": run.returncode,
        "stdout": run.stdout,
        "stderr": run.stderr,
    }
  except subprocess.TimeoutExpired as exc:
    return {
        "command": items,
        "returncode": 124,
        "stdout": str(exc.stdout or ""),
        "stderr": str(exc.stderr or exc),
    }


def build_trace(raw: Path, timeout_s: int) -> dict[str, Any]:
  configure = command([
      str(CMAKE), "-S", str(ROOT / "engine"), "-B", str(BUILD_DIR),
      "-DCMAKE_BUILD_TYPE=Release",
  ], timeout_s=min(timeout_s, 600))
  build = command([
      str(CMAKE), "--build", str(BUILD_DIR), "--target", TRACE_TARGET,
      "-j1",
  ], timeout_s=min(timeout_s, 600))
  result = {
      "configure": configure,
      "build": build,
      "library": relative(TRACE_LIBRARY),
      "library_sha256": (
          sha256_file(TRACE_LIBRARY) if TRACE_LIBRARY.is_file() else None),
      "pass": (
          configure["returncode"] == 0 and build["returncode"] == 0 and
          TRACE_LIBRARY.is_file()),
  }
  write_json(raw / "trace-build.json", result)
  return result


def run_worker(
    args: argparse.Namespace, raw: Path, lane: str, mode: str,
    decode_tokens: list[int], decode_steps: int,
) -> dict[str, Any]:
  fine_codec = (
      args.direct_i8_group4_full_cold or args.direct_i8_hybrid_k2_v4)
  memory_stop_bytes = int(args.memory_stop_gib * 1024**3)
  available_before_worker = available_memory_bytes()
  if available_before_worker < memory_stop_bytes:
    raise RuntimeError(
        f"{lane} {mode} worker skipped to avoid host OOM: "
        f"available={available_before_worker} stop={memory_stop_bytes}")
  worker_dir = raw / lane / mode
  programs = worker_dir / "programs"
  cache = worker_dir / "neo-cache"
  worker_dir.mkdir(parents=True)
  programs.mkdir()
  cache.mkdir()
  result_path = worker_dir / "worker-result.json"
  marker_path = worker_dir / "trace-active"
  config_path = worker_dir / "worker-config.json"
  write_json(config_path, {
      "collect_states": False,
      "custom_config": str(args.custom_config.resolve()),
      "candidate_gpu_plugin": (
          str(args.candidate_gpu_plugin.resolve())
          if mode == "candidate" and args.candidate_gpu_plugin is not None
          else None),
      "decode_steps": decode_steps,
      "decode_tokens": decode_tokens,
      "device": args.device,
      "lane": lane,
      "mode": mode,
      "model_dir": str(args.model_dir.resolve()),
      "prompt": str(PROMPTS[lane].resolve()),
      "prefill_chunk_tokens": PREFILL_CHUNK_TOKENS,
      # The exact lane owns fixed request-local storage.  Besides removing the
      # O(history) graph append, this exposes fresh/continuation lengths to IGC
      # through static tensor dimensions instead of runtime state metadata.
      "fixed_cold_capacity": (
          DIRECT_I8_FIXED_COLD_CAPACITY
          if mode == "candidate" and fine_codec else
          int(lane.removesuffix("k")) * 1024),
      "initialize_hot_states": args.graph_initialized_hot_states,
      "skip_hot_state_self_bind": args.graph_initialized_hot_states,
      "dump_runtime_graph": args.dump_runtime_graph,
      "capture_full_profile": args.capture_full_profile,
      "fuse_linear_conv_state": args.fuse_linear_conv_state,
      "pack_gdn_state": mode == "candidate" and args.pack_gdn_state,
      "direct_i8_fixed_layout": (
          mode == "candidate" and fine_codec),
      "direct_i8_group4_full_cold": (
          mode == "candidate" and args.direct_i8_group4_full_cold),
      "direct_i8_hybrid_k2_v4": (
          mode == "candidate" and args.direct_i8_hybrid_k2_v4),
      "memory_stop_bytes": memory_stop_bytes,
      "prefill_history_capacity": (
          DIRECT_I8_PREFILL_HISTORY_CAPACITY
          if mode == "candidate" and fine_codec else
          max(2 * PREFILL_CHUNK_TOKENS,
              int(lane.removesuffix("k")) * 1024)),
      "phase_branch_prefill": args.phase_branch_prefill,
      "stock_prefill_custom_decode": args.stock_prefill_custom_decode,
      "stock_prefill_sliced_decode": args.stock_prefill_sliced_decode,
      "static_phase_separated": (
          mode == "candidate" and args.static_phase_separated),
      "raw": str(worker_dir.resolve()),
      "result": str(result_path.resolve()),
      "target_layers": list(args.target_layers),
      "trace_marker": str(marker_path.resolve()),
  })
  trace_path = worker_dir / "opencl-trace.jsonl"
  trace_filter = (
      "sdpa_micro__" if mode == "stock" else
      ("iq36_,sdpa_micro__" if (
          args.stock_prefill_custom_decode or
          args.stock_prefill_sliced_decode or
          args.static_phase_separated) else
       ("iq36_" if args.phase_branch_prefill else
        "iq36_hot_attention_single_owner")))
  environment = os.environ.copy()
  environment.pop("OV_GPU_CONFIG_FILE", None)
  environment.pop("OV_GPU_USM_POLICY", None)
  environment.pop("IQ36_GDN_TRANSPOSED_STATE", None)
  environment.update({
      "IQ36_OPENCL_CUSTOM_SDPA_DUMP_LIMIT": "64",
      "IQ36_OPENCL_PROGRAM_DUMP_DIR": str(programs.resolve()),
      "IQ36_OPENCL_STOCK_SDPA_DUMP_LIMIT": "64",
      "IQ36_OPENCL_TRACE_FILTER": trace_filter,
      "IQ36_OPENCL_TRACE_MARKER": str(marker_path.resolve()),
      "IQ36_OPENCL_TRACE_PATH": str(trace_path.resolve()),
      "IQ36_OPENCL_TRACE_TIMING": "1",
      "LD_AUDIT": str(TRACE_LIBRARY.resolve()),
      "NEO_CACHE_DIR": str(cache.resolve()),
      "NEO_CACHE_MAX_SIZE": str(4 * 1024 * 1024 * 1024),
      "NEO_CACHE_PERSISTENT": "1",
  })
  if mode == "candidate" and args.candidate_gpu_usm_policy is not None:
    environment["OV_GPU_USM_POLICY"] = str(args.candidate_gpu_usm_policy)
  items = [
      str(args.openvino_python), str(WORKER),
      "--worker-config", str(config_path),
  ]
  run = command(items, timeout_s=args.timeout_s, environment=environment)
  (worker_dir / "worker.stdout").write_text(
      run["stdout"], encoding="utf-8")
  (worker_dir / "worker.stderr").write_text(
      run["stderr"], encoding="utf-8")
  captured_environment = [
      "IQ36_OPENCL_CUSTOM_SDPA_DUMP_LIMIT",
      "IQ36_OPENCL_PROGRAM_DUMP_DIR",
      "IQ36_OPENCL_STOCK_SDPA_DUMP_LIMIT",
      "IQ36_OPENCL_TRACE_FILTER",
      "IQ36_OPENCL_TRACE_MARKER", "IQ36_OPENCL_TRACE_PATH",
      "IQ36_OPENCL_TRACE_TIMING", "LD_AUDIT", "NEO_CACHE_DIR",
      "NEO_CACHE_MAX_SIZE", "NEO_CACHE_PERSISTENT",
  ]
  if "OV_GPU_USM_POLICY" in environment:
    captured_environment.append("OV_GPU_USM_POLICY")
  write_json(worker_dir / "worker-command.json", {
      "command": items,
      "environment": {key: environment[key]
                      for key in captured_environment},
      "returncode": run["returncode"],
  })
  return {
      "returncode": run["returncode"],
      "result": load_json(result_path) if result_path.is_file() else {},
      "trace": load_jsonl(trace_path),
      "available_before_worker_bytes": available_before_worker,
  }


def phase_summary(
    rows: list[dict[str, Any]], expected_marker: str,
) -> dict[str, Any]:
  selected = [
      row for row in rows
      if row.get("event") == "ndrange" and
      (row.get("marker") == expected_marker or
       str(row.get("marker", "")).startswith(expected_marker + "-chunk"))]
  durations = [int(row.get("duration_ns", 0)) for row in selected]
  return {
      "dispatch_count": len(selected),
      "dispatch_markers": sorted({str(row.get("marker", ""))
                                    for row in selected}),
      "duration_total_ms": sum(durations) / 1_000_000.0,
      "duration_median_us": (
          statistics.median(durations) / 1000.0 if durations else None),
      "duration_min_us": min(durations) / 1000.0 if durations else None,
      "duration_max_us": max(durations) / 1000.0 if durations else None,
      "durations_positive": bool(durations) and all(value > 0
                                                    for value in durations),
      "timing_statuses": sorted({row.get("timing_status")
                                  for row in selected}),
      "kernels": sorted({str(row.get("kernel")) for row in selected}),
      "global_sizes": sorted({tuple(row.get("global_size") or [])
                              for row in selected}),
      "local_sizes": sorted({tuple(row.get("local_size") or [])
                             for row in selected}),
  }


def distribution_comparison(
    reference_path: Path, candidate_path: Path,
) -> dict[str, Any]:
  import numpy as np

  reference = np.fromfile(reference_path, dtype="<f4").astype(np.float64)
  candidate = np.fromfile(candidate_path, dtype="<f4").astype(np.float64)
  shape_match = reference.shape == candidate.shape
  finite = bool(
      shape_match and np.isfinite(reference).all() and
      np.isfinite(candidate).all())
  if not finite:
    return {
        "shape_match": shape_match, "finite": False,
        "reference_top1": None, "candidate_top1": None,
        "top1_match": False, "kld_reference_to_candidate": float("inf"),
    }
  reference_probability = np.exp(reference - float(np.max(reference)))
  candidate_probability = np.exp(candidate - float(np.max(candidate)))
  reference_probability /= float(reference_probability.sum())
  candidate_probability /= float(candidate_probability.sum())
  epsilon = np.finfo(np.float64).tiny
  reference_top1 = int(np.argmax(reference))
  candidate_top1 = int(np.argmax(candidate))
  return {
      "shape_match": True,
      "finite": True,
      "reference_top1": reference_top1,
      "candidate_top1": candidate_top1,
      "top1_match": reference_top1 == candidate_top1,
      "kld_reference_to_candidate": float(np.sum(
          reference_probability * (
              np.log(np.maximum(reference_probability, epsilon)) -
              np.log(np.maximum(candidate_probability, epsilon))))),
  }


def check(name: str, passed: bool, **evidence: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **evidence}


def main() -> int:
  args = parse_args()
  fine_codec_name = (
      "hybrid_k2_v4" if args.direct_i8_hybrid_k2_v4 else
      "group4" if args.direct_i8_group4_full_cold else None)

  def expected_dispatch_count(mode: str, phase_index: int) -> int:
    if mode == "stock":
      return len(TARGET_LAYERS)
    if args.static_phase_separated:
      return (len(TARGET_LAYERS) + len(args.target_layers)
              if phase_index == 0 else len(TARGET_LAYERS))
    if args.stock_prefill_custom_decode:
      return ((len(TARGET_LAYERS) - len(args.target_layers)) +
              len(args.target_layers) *
              (3 if phase_index == 0 else 2))
    if args.stock_prefill_sliced_decode:
      return len(TARGET_LAYERS) + len(args.target_layers)
    return len(args.target_layers)

  out = args.out_dir.resolve()
  raw = out / "raw"
  raw.mkdir(parents=True, exist_ok=False)
  git = git_state(out)
  memory_stop_bytes = int(args.memory_stop_gib * 1024**3)
  initial_available_bytes = available_memory_bytes()
  if initial_available_bytes < memory_stop_bytes:
    raise RuntimeError(
        "trace build skipped to avoid host OOM: "
        f"available={initial_available_bytes} stop={memory_stop_bytes}")
  trace_build = build_trace(raw, args.timeout_s)
  runs: dict[str, dict[str, dict[str, Any]]] = {}
  lane_decode_steps = {
      lane: (args.decode_steps
             if args.decode_steps is not None else DECODE_STEPS[lane])
      for lane in args.lanes
  }
  for lane in args.lanes:
    decode_steps = lane_decode_steps[lane]
    stock = run_worker(args, raw, lane, "stock", [], decode_steps)
    stock_phases = stock["result"].get("phases", [])
    teacher = [int(row["top1"])
               for row in stock_phases[:decode_steps]]
    candidate = run_worker(
        args, raw, lane, "candidate", teacher, decode_steps)
    runs[lane] = {"stock": stock, "candidate": candidate}

  lanes: dict[str, Any] = {}
  checks = [
      check("repository_clean_at_gate", not git["dirty"], git=git),
      check("trace_library_builds", trace_build["pass"],
            trace_build=trace_build),
  ]
  for lane, modes in runs.items():
    lane_row: dict[str, Any] = {"modes": {}}
    phase_count = lane_decode_steps[lane] + 1
    for mode, run in modes.items():
      result = run["result"]
      phase_rows = []
      result_phases = result.get("phases", [])
      for index in range(phase_count):
        phase = result_phases[index] if index < len(result_phases) else {}
        marker = str(phase.get(
            "trace_marker",
            f"{lane}-{mode}-phase{index}-missing"))
        prefill_chunks = list(phase.get("prefill_chunks", []))
        phase_rows.append({
            "index": index,
            "input_tokens": phase.get("input_tokens"),
            "total_tokens": phase.get("total_tokens"),
            "top1": phase.get("top1"),
            "logits_finite": phase.get("logits_finite"),
            "marker": marker,
            "prefill_chunks": prefill_chunks,
            "expected_dispatch_multiplier": max(1, len(prefill_chunks)),
            **phase_summary(run["trace"], marker),
        })
      dumps = [row for row in run["trace"]
               if row.get("event") == "program_dump"]
      lane_row["modes"][mode] = {
          "returncode": run["returncode"],
          "program_dumps": dumps,
          "phases": phase_rows,
          "top1": [phase.get("top1") for phase in result.get("phases", [])],
          "openvino_version": result.get("openvino_version"),
          "pack_gdn_state": result.get("pack_gdn_state"),
          "direct_i8_group4_full_cold": result.get(
              "direct_i8_group4_full_cold"),
          "direct_i8_hybrid_k2_v4": result.get(
              "direct_i8_hybrid_k2_v4"),
          "memory_samples": result.get("memory_samples"),
          "available_before_worker_bytes": run.get(
              "available_before_worker_bytes"),
      }
      checks.extend([
          check(f"{lane}_{mode}_worker_completes",
                run["returncode"] == 0),
          check(f"{lane}_{mode}_has_live_program_dump", bool(dumps),
                program_dumps=dumps),
          check(f"{lane}_{mode}_all_phase_logits_finite",
                len(phase_rows) == phase_count and all(
                    row["logits_finite"] is True for row in phase_rows),
                phases=phase_rows),
          check(f"{lane}_{mode}_has_ten_timed_dispatches_per_phase",
                len(phase_rows) == phase_count and all(
                    row["dispatch_count"] == expected_dispatch_count(
                        mode, row["index"]) *
                    row["expected_dispatch_multiplier"] and
                    row["durations_positive"] and
                    row["timing_statuses"] == [0]
                    for row in phase_rows), phases=phase_rows),
      ])
    stock_rows = lane_row["modes"]["stock"]["phases"]
    candidate_rows = lane_row["modes"]["candidate"]["phases"]
    comparisons = []
    for stock, candidate in zip(stock_rows, candidate_rows, strict=True):
      stock_ms = float(stock["duration_total_ms"])
      candidate_ms = float(candidate["duration_total_ms"])
      comparable = (
          stock["dispatch_count"] > 0 and candidate["dispatch_count"] > 0 and
          stock["durations_positive"] and candidate["durations_positive"])
      comparisons.append({
          "index": stock["index"],
          "stock_ms": stock_ms,
          "candidate_ms": candidate_ms,
          "candidate_over_stock": (
              candidate_ms / stock_ms if comparable and stock_ms > 0.0
              else None),
          "comparable": comparable,
          "carrier_admission_pass": (
              comparable and candidate_ms <= stock_ms),
      })
    lane_row["comparisons"] = comparisons
    stock_top1 = lane_row["modes"]["stock"]["top1"]
    candidate_top1 = lane_row["modes"]["candidate"]["top1"]
    checks.append(check(
        f"{lane}_packed_gdn_state_is_candidate_only",
        lane_row["modes"]["stock"]["pack_gdn_state"] is False and
        lane_row["modes"]["candidate"]["pack_gdn_state"] is
            args.pack_gdn_state))
    checks.append(check(
        f"{lane}_greedy_path_matches_during_attribution",
        stock_top1 == candidate_top1,
        stock_top1=stock_top1, candidate_top1=candidate_top1))
    lanes[lane] = lane_row

  fine_codec_integration_inference = None
  fine_codec_distribution_rows = None
  fine_codec_max_kld = None
  if fine_codec_name is not None:
    candidate_decode_rows = lanes["32k"]["modes"]["candidate"]["phases"]
    fine_codec_samples = [
        float(row["duration_total_ms"])
        for row in candidate_decode_rows
        if int(row["index"]) > FINE_CODEC_PROFILE_SKIP_STEPS]
    fine_codec_integration_inference = latency_cap_inference(
        fine_codec_samples, cap=FINE_CODEC_COMPONENT_CAP_MS,
        min_samples=FINE_CODEC_PROFILE_MINIMUM_SAMPLES)
    stock_result_phases = runs["32k"]["stock"]["result"].get("phases", [])
    candidate_result_phases = runs["32k"]["candidate"]["result"].get(
        "phases", [])
    fine_codec_distribution_rows = []
    for stock_phase, candidate_phase in zip(
        stock_result_phases, candidate_result_phases, strict=True):
      stock_path = Path(stock_phase["logits_path"])
      candidate_path = Path(candidate_phase["logits_path"])
      if not stock_path.is_absolute():
        stock_path = ROOT / stock_path
      if not candidate_path.is_absolute():
        candidate_path = ROOT / candidate_path
      fine_codec_distribution_rows.append({
          "index": int(stock_phase["index"]),
          **distribution_comparison(stock_path, candidate_path),
      })
    fine_codec_max_kld = max(
        float(row["kld_reference_to_candidate"])
        for row in fine_codec_distribution_rows)
    memory_stop_bytes = int(args.memory_stop_gib * 1024**3)
    checks.extend([
        check(
            f"{fine_codec_name}_is_isolated_to_candidate",
            lanes["32k"]["modes"]["stock"].get(
                "direct_i8_group4_full_cold") is False and
            lanes["32k"]["modes"]["stock"].get(
                "direct_i8_hybrid_k2_v4") is False and
            lanes["32k"]["modes"]["candidate"].get(
                "direct_i8_group4_full_cold") is
                    args.direct_i8_group4_full_cold and
            lanes["32k"]["modes"]["candidate"].get(
                "direct_i8_hybrid_k2_v4") is args.direct_i8_hybrid_k2_v4),
        check(
            f"{fine_codec_name}_integrated_decode_ucb_clears_component_cap",
            fine_codec_integration_inference.get("rate_pass") is True,
            performance_inference=fine_codec_integration_inference,
            skipped_decode_steps=FINE_CODEC_PROFILE_SKIP_STEPS),
        check(
            f"{fine_codec_name}_all_profile_distributions_pass",
            len(fine_codec_distribution_rows) == args.decode_steps + 1 and
            all(
                row["finite"] is True and row["top1_match"] is True and
                float(row["kld_reference_to_candidate"]) <= 0.005
                for row in fine_codec_distribution_rows),
            max_kld=fine_codec_max_kld,
            distributions=fine_codec_distribution_rows),
        check(
            f"{fine_codec_name}_memory_stop_never_trips",
            all(
                int(mode["available_before_worker_bytes"]) >=
                    memory_stop_bytes and
                int((mode.get("memory_samples") or {}).get(
                    "after_language_compile", 0)) >= memory_stop_bytes
                for mode in lanes["32k"]["modes"].values()),
            memory_stop_bytes=memory_stop_bytes),
    ])

  attribution_passed = all(row["pass"] for row in checks)
  carrier_admission_passed = all(
      comparison["carrier_admission_pass"]
      for lane in lanes.values() for comparison in lane["comparisons"])
  metrics = {
      "schema": SCHEMA,
      "git": git,
      "host": platform.node(),
      "kernel": platform.release(),
      "target_layers": list(args.target_layers),
      "phase_branch_prefill": args.phase_branch_prefill,
      "stock_prefill_custom_decode": args.stock_prefill_custom_decode,
      "stock_prefill_sliced_decode": args.stock_prefill_sliced_decode,
      "static_phase_separated": args.static_phase_separated,
      "candidate_gpu_plugin": (
          relative(args.candidate_gpu_plugin)
          if args.candidate_gpu_plugin is not None else None),
      "candidate_gpu_plugin_sha256": (
          sha256_file(args.candidate_gpu_plugin)
          if args.candidate_gpu_plugin is not None else None),
      "candidate_gpu_usm_policy": args.candidate_gpu_usm_policy,
      "graph_initialized_hot_states": args.graph_initialized_hot_states,
      "dump_runtime_graph": args.dump_runtime_graph,
      "capture_full_profile": args.capture_full_profile,
      "fuse_linear_conv_state": args.fuse_linear_conv_state,
      "pack_gdn_state": args.pack_gdn_state,
      "direct_i8_group4_full_cold": args.direct_i8_group4_full_cold,
      "direct_i8_hybrid_k2_v4": args.direct_i8_hybrid_k2_v4,
      "fine_codec_name": fine_codec_name,
      "memory_stop_bytes": int(args.memory_stop_gib * 1024**3),
      "initial_available_bytes": initial_available_bytes,
      "decode_steps": lane_decode_steps,
      "fine_codec_integration_inference": fine_codec_integration_inference,
      "fine_codec_distribution_rows": fine_codec_distribution_rows,
      "fine_codec_max_kld": fine_codec_max_kld,
      "group4_integration_inference": (
          fine_codec_integration_inference
          if args.direct_i8_group4_full_cold else None),
      "group4_distribution_rows": (
          fine_codec_distribution_rows
          if args.direct_i8_group4_full_cold else None),
      "group4_max_kld": (
          fine_codec_max_kld if args.direct_i8_group4_full_cold else None),
      "hybrid_k2_v4_integration_inference": (
          fine_codec_integration_inference
          if args.direct_i8_hybrid_k2_v4 else None),
      "hybrid_k2_v4_distribution_rows": (
          fine_codec_distribution_rows
          if args.direct_i8_hybrid_k2_v4 else None),
      "hybrid_k2_v4_max_kld": (
          fine_codec_max_kld if args.direct_i8_hybrid_k2_v4 else None),
      "attribution_checks_passed": attribution_passed,
      "carrier_admission_passed": carrier_admission_passed,
      "checks": checks,
      "lanes": lanes,
  }
  write_json(out / "metrics.json", metrics)
  write_json(out / "manifest.json", {
      "schema": SCHEMA,
      "git": git,
      "model_dir": str(args.model_dir.resolve()),
      "custom_config": relative(args.custom_config),
      "phase_branch_prefill": args.phase_branch_prefill,
      "stock_prefill_custom_decode": args.stock_prefill_custom_decode,
      "stock_prefill_sliced_decode": args.stock_prefill_sliced_decode,
      "static_phase_separated": args.static_phase_separated,
      "candidate_gpu_plugin": (
          relative(args.candidate_gpu_plugin)
          if args.candidate_gpu_plugin is not None else None),
      "candidate_gpu_plugin_sha256": (
          sha256_file(args.candidate_gpu_plugin)
          if args.candidate_gpu_plugin is not None else None),
      "candidate_gpu_usm_policy": args.candidate_gpu_usm_policy,
      "graph_initialized_hot_states": args.graph_initialized_hot_states,
      "dump_runtime_graph": args.dump_runtime_graph,
      "capture_full_profile": args.capture_full_profile,
      "fuse_linear_conv_state": args.fuse_linear_conv_state,
      "pack_gdn_state": args.pack_gdn_state,
      "direct_i8_group4_full_cold": args.direct_i8_group4_full_cold,
      "direct_i8_hybrid_k2_v4": args.direct_i8_hybrid_k2_v4,
      "fine_codec_name": fine_codec_name,
      "memory_stop_bytes": int(args.memory_stop_gib * 1024**3),
      "initial_available_bytes": initial_available_bytes,
      "decode_steps": lane_decode_steps,
      "prefill_chunk_tokens": PREFILL_CHUNK_TOKENS,
      "prompts": {lane: relative(path) for lane, path in PROMPTS.items()},
      "trace_library": relative(TRACE_LIBRARY),
  })
  summary = [
      "# OpenVINO attention phase profile gate",
      "",
      f"- attribution checks passed: `{str(attribution_passed).lower()}`",
      f"- tiled-carrier admission passed: `{str(carrier_admission_passed).lower()}`",
      f"- commit: `{git['commit']}`",
      "",
      "| lane | phase | stock ten-layer ms | candidate ten-layer ms | ratio |",
      "|---|---:|---:|---:|---:|",
  ]
  for lane, row in lanes.items():
    for comparison in row["comparisons"]:
      ratio = comparison["candidate_over_stock"]
      ratio_text = f"{ratio:.3f}x" if ratio is not None else "n/a"
      summary.append(
          f"| {lane} | {comparison['index']} | "
          f"{comparison['stock_ms']:.6f} | "
          f"{comparison['candidate_ms']:.6f} | "
          f"{ratio_text} |")
  summary.extend([
      "",
      "This artifact closes phase attribution and live-program capture only.",
      "The candidate remains blocked until every comparison is no slower",
      "than same-run stock after the tiled/XMX carrier is installed.",
      "",
  ])
  if fine_codec_integration_inference is not None:
    codec_heading = (
        "Hybrid K2/V4" if args.direct_i8_hybrid_k2_v4 else "Group-4")
    summary.extend([
        f"## {codec_heading} integrated decode",
        "",
        ("- median / one-sided 95% UCB / cap: "
         f"`{fine_codec_integration_inference['point_estimate_ms']} / "
         f"{fine_codec_integration_inference['upper_confidence_bound_ms']} / "
         f"{FINE_CODEC_COMPONENT_CAP_MS} ms`"),
        ("- samples after skip: "
         f"`{fine_codec_integration_inference['sample_count']}`"),
        ("- maximum teacher-forced KLD: "
         f"`{fine_codec_max_kld}`"),
        "",
    ])
  (out / "summary.md").write_text("\n".join(summary), encoding="utf-8")
  print(json.dumps({
      "event": "complete",
      "out_dir": relative(out),
      "attribution_checks_passed": attribution_passed,
      "carrier_admission_passed": carrier_admission_passed,
      "fine_codec_integration_admitted": (
          fine_codec_integration_inference is not None and
          fine_codec_integration_inference.get("rate_pass") is True and
          attribution_passed and carrier_admission_passed),
      "group4_integration_admitted": (
          args.direct_i8_group4_full_cold and
          fine_codec_integration_inference is not None and
          fine_codec_integration_inference.get("rate_pass") is True and
          attribution_passed and carrier_admission_passed),
      "hybrid_k2_v4_integration_admitted": (
          args.direct_i8_hybrid_k2_v4 and
          fine_codec_integration_inference is not None and
          fine_codec_integration_inference.get("rate_pass") is True and
          attribution_passed and carrier_admission_passed),
  }, sort_keys=True))
  return 0 if (
      attribution_passed and
      (fine_codec_name is None or
       (carrier_admission_passed and
        fine_codec_integration_inference is not None and
        fine_codec_integration_inference.get("rate_pass") is True))) else 1


if __name__ == "__main__":
  raise SystemExit(main())
