#!/usr/bin/env python3
"""Capture a context-spanning population of reproducible LM-head slow events.

This gate launches two strictly serial teacher-forced candidate workers from
the accepted seq2189 plugin.  The locked code corpus prevents greedy-loop
collapse while retaining real model hidden states.  The first worker records
every F16 LM-head input in a single memmap; the second repeats the same input
stream without writing the matrix.  A classifier registered against the
accepted 18 short rows removes smooth context drift, and only slow indices
reproduced by both long workers are eligible for the exactly-2000-row offline
certificate population.

This is a capture gate, not a source/plugin integration or speedup gate.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import math
import statistics
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-openvino-lm-head-slow-event-capture-v1"
PRODUCT_TOOL = ROOT / "tools/intel-qwen36-openvino-hot-cold-product-gate.py"
BASE_ROOT = ROOT / (
    "output/openvino-lm-head-parallel-block-topk-2k-abba8-"
    "20260731Tseq2193-clean")
BASE_CONFIG = BASE_ROOT / (
    "raw/prefill_shape_002k/block01/candidate-b1/worker-config.json")
BASE_GATE = BASE_ROOT / "gate.json"
CALIBRATION_GLOB = (
    "raw/prefill_shape_002k/block*/candidate-b*/worker-result.json")
FALLBACK_BOUND = ROOT / (
    "output/openvino-lm-head-gated-exact-fallback-bound-"
    "20260731Tseq2186-clean/result.json")
OLD_CALIBRATION_RESULTS = (
    ROOT / (
        "output/openvino-short-nonsentinel-auto-abba1-"
        "20260731Tseq2185-clean/raw/prefill_shape_002k/block00/"
        "candidate-b1/worker-result.json"),
    ROOT / (
        "output/openvino-short-nonsentinel-auto-abba1-"
        "20260731Tseq2185-clean/raw/prefill_shape_002k/block00/"
        "candidate-b2/worker-result.json"),
)
CERTIFICATE_BOUND = ROOT / (
    "output/openvino-lm-head-exact-token-certificate-bound-"
    "20260801Tseq2280-lloyd-clean/metrics.json")
CORPUS_SMOKE_ROOT = ROOT / (
    "output/openvino-lm-head-teacher-forced-corpus-smoke-"
    "20260801Tseq2284-explore")
CORPUS_SMOKE_REFERENCE = CORPUS_SMOKE_ROOT / "teacher-reference.json"
CORPUS_SMOKE_RESULT = CORPUS_SMOKE_ROOT / "raw/candidate/worker-result.json"
TEACHER_FORCED_CORPUS = ROOT / "engine/src/gpu_q4x8_matvec.cpp"
PLUGIN = Path(
    "/home/intel/intel-qwen36-r0/output/"
    "openvino-90214e-l0-gpu-seq2189/bin/intel64/Release/"
    "libopenvino_intel_gpu_plugin.so")

EXPECTED_BASE_CONFIG_SHA256 = (
    "393ccb8ef91b923b77a34f77757e5a9ebe8d63e07743837f10c94b86effe9eea")
EXPECTED_BASE_GATE_SHA256 = (
    "c125f51dde39d6080ed1b4a8698cb3864874fcf31e3acb5a38fffbae9c86ceee")
EXPECTED_FALLBACK_BOUND_SHA256 = (
    "6bed3a9f24917433d51559c62d3dec222abaa9654c16ad1443b96b98b0936be7")
EXPECTED_OLD_CALIBRATION_SHA256 = (
    "dcf157a8efded91ee122a1e837668d8b9282a8747d7d190c11a2dce0781aa4c7",
    "eea8d0be34e51a595a4e533cf2f0f5577ea4fcbb17df583aa349e17c942af36b",
)
EXPECTED_CURRENT_CALIBRATION_SET_SHA256 = (
    "2e85b793f5de61ca11f8cd6ffe74a0f80a958f15202508634aff93fd725fcd48")
EXPECTED_CERTIFICATE_BOUND_SHA256 = (
    "a2003ff9cfdd401a6f5f4915193738e48c66068e3a226bf1dd3076d52d919d8a")
EXPECTED_CORPUS_SMOKE_REFERENCE_SHA256 = (
    "87a720996bf0c4b3d0f31269fdf82951645073b655280986ca3b22416089174b")
EXPECTED_CORPUS_SMOKE_RESULT_SHA256 = (
    "06b1cf8f960683f29654af00e01da11adee01172dc18ad5caae4b5c44e50f02c")
EXPECTED_TEACHER_FORCED_CORPUS_SHA256 = (
    "6427111ee1566da61b83fb729d9ec848d3f5a643c16e495fe9c9dd6428564eea")
EXPECTED_PLUGIN_SHA256 = (
    "9832adffa8bf3fd7b013c47d9d2abefa66987c5fbde350c9669fce58c697b985")
EXPECTED_PROVIDER = "+".join((
    "iq36_lm_head_q8_group256_f16_sums",
    "iq36_lm_head_i8q1_rowstripe8_matvec_local_top12_f16",
    "iq36_lm_head_i8_exact_local_top12_correction_f16",
    "iq36_lm_head_output_topk8_f16",
    "iq36_lm_head_topk8_merge_f32",
    "iq36_lm_head_i8_direct_topk8_correction_f16",
    "iq36_lm_head_i8q1_gated_exact_reset_f16",
    "iq36_lm_head_i8q1_gated_exact_collect_f16",
    "iq36_lm_head_i8_gated_exact_matvec_f16",
    "iq36_lm_head_i8q1_gated_exact_output_topk8_f16",
    "iq36_lm_head_i8q1_gated_exact_topk8_merge_f32",
    "iq36_lm_head_i8_gated_exact_topk8_correction_f16",
))

OUTPUT_TOKENS = 32_768
PROMPT_TOKENS = 2_048
EXACT_HISTORY_CAPACITY = PROMPT_TOKENS + OUTPUT_TOKENS
PREFILL_HISTORY_CAPACITY = 16_384
HIDDEN_COLUMNS = 2_048
TARGET_EVENTS = 2_000
SKIP_INTERVALS = 16
ROLLING_WINDOW = 201
EXPECTED_INCREMENT_MS = 4.767555999999999
RESIDUAL_LOW_MS = -0.5 * EXPECTED_INCREMENT_MS
RESIDUAL_HIGH_MS = 2.0 * EXPECTED_INCREMENT_MS
SPLIT_MIDPOINT_LOW_MS = 0.35 * EXPECTED_INCREMENT_MS
SPLIT_MIDPOINT_HIGH_MS = 0.85 * EXPECTED_INCREMENT_MS
MIN_SPLIT_GAP_MS = 0.25 * EXPECTED_INCREMENT_MS
SLOW_MEDIAN_LOW_MS = 0.75 * EXPECTED_INCREMENT_MS
SLOW_MEDIAN_HIGH_MS = 1.25 * EXPECTED_INCREMENT_MS
SLOW_RATE_LOW = 0.05
SLOW_RATE_HIGH = 0.20
PREFLIGHT_GIB = 8.0
MEMORY_STOP_GIB = 4.0


def load_module(name: str, path: Path) -> Any:
  spec = importlib.util.spec_from_file_location(name, path)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import {path}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


PRODUCT = load_module("iq36_lm_head_slow_capture_product", PRODUCT_TOOL)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out-dir", type=Path, required=True)
  parser.add_argument("--resume", action="store_true")
  parser.add_argument("--timeout-s", type=int, default=1_800)
  args = parser.parse_args()
  if args.timeout_s <= 0:
    parser.error("timeout must be positive")
  return args


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    while chunk := stream.read(8 * 1024 * 1024):
      digest.update(chunk)
  return digest.hexdigest()


def calibration_set_sha256(paths: list[Path]) -> str:
  """Match sha256sum's stable text representation for the registered set."""
  digest = hashlib.sha256()
  for path in paths:
    row = f"{sha256(path)}  {PRODUCT.relative(path)}\n"
    digest.update(row.encode("utf-8"))
  return digest.hexdigest()


def check(name: str, passed: bool, **details: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **details}


def git_state(out: Path) -> dict[str, Any]:
  state = PRODUCT.BOOT.git_state(out)

  def git(*parts: str) -> str:
    run = subprocess.run(
        ["git", *parts], cwd=ROOT, check=False, capture_output=True,
        text=True, encoding="utf-8", errors="replace")
    return run.stdout.strip() if run.returncode == 0 else ""

  branch = git("branch", "--show-current")
  upstream = git("rev-parse", "--verify", "@{upstream}")
  state.update({
      "branch": branch,
      "upstream_commit": upstream or None,
      "pushed": bool(upstream) and upstream == state["commit"],
  })
  return state


def percentile(values: list[float], probability: float) -> float:
  ordered = sorted(values)
  if not ordered:
    raise ValueError("empty percentile input")
  index = min(
      len(ordered) - 1, int(round(probability * (len(ordered) - 1))))
  return ordered[index]


def detrended_slow_events(
    decode_wall_ms: list[Any],
) -> tuple[dict[str, Any], list[float], list[float]]:
  values = [float(value) for value in decode_wall_ms]
  if len(values) <= SKIP_INTERVALS + ROLLING_WINDOW:
    raise ValueError("not enough decode intervals for registered classifier")
  measured = values[SKIP_INTERVALS:]
  half = ROLLING_WINDOW // 2
  baselines = [
      float(statistics.median(
          measured[max(0, index - half):
                   min(len(measured), index + half + 1)]))
      for index in range(len(measured))
  ]
  residuals = [
      value - baseline for value, baseline in zip(measured, baselines)]
  bounded = sorted(
      value for value in residuals
      if RESIDUAL_LOW_MS <= value <= RESIDUAL_HIGH_MS)
  if len(bounded) < 2:
    raise ValueError("not enough bounded residuals")
  candidates = []
  for index, (lower, upper) in enumerate(zip(bounded, bounded[1:])):
    midpoint = 0.5 * (lower + upper)
    if SPLIT_MIDPOINT_LOW_MS <= midpoint <= SPLIT_MIDPOINT_HIGH_MS:
      candidates.append((upper - lower, index, midpoint))
  if not candidates:
    raise ValueError("no registered slow-mode split candidate")
  largest_gap, split_index, threshold = max(candidates)
  slow_after_skip = [
      index for index, value in enumerate(residuals)
      if threshold <= value <= RESIDUAL_HIGH_MS]
  slow_values = [residuals[index] for index in slow_after_skip]
  if not slow_values:
    raise ValueError("registered classifier returned no slow events")
  slow_decode_indices = [
      index + SKIP_INTERVALS for index in slow_after_skip]
  summary = {
      "algorithm": "centered_rolling_median_then_registered_largest_gap",
      "decode_interval_count": len(values),
      "measured_interval_count": len(measured),
      "skip_intervals": SKIP_INTERVALS,
      "rolling_window": ROLLING_WINDOW,
      "residual_bound_ms": [RESIDUAL_LOW_MS, RESIDUAL_HIGH_MS],
      "split_midpoint_bound_ms": [
          SPLIT_MIDPOINT_LOW_MS, SPLIT_MIDPOINT_HIGH_MS],
      "largest_gap_ms": largest_gap,
      "split_index_in_bounded_order": split_index,
      "split_threshold_ms": threshold,
      "slow_count": len(slow_after_skip),
      "slow_rate": len(slow_after_skip) / len(measured),
      "slow_indices_after_skip": slow_after_skip,
      "slow_decode_indices": slow_decode_indices,
      "slow_steps": [index + 1 for index in slow_decode_indices],
      "slow_residual_min_ms": min(slow_values),
      "slow_residual_median_ms": float(statistics.median(slow_values)),
      "slow_residual_p95_ms": percentile(slow_values, 0.95),
      "outlier_count_above_registered_residual": sum(
          value > RESIDUAL_HIGH_MS for value in residuals),
  }
  return summary, baselines, residuals


def classifier_calibration(
    fallback: dict[str, Any], old_paths: tuple[Path, ...],
    current_paths: list[Path],
) -> dict[str, Any]:
  expected = [
      int(value) for value in
      fallback["worker_rows"][0]["mode"]["slow_indices_after_skip"]
  ]
  expected_set = set(expected)
  rows = []
  for cohort, paths in (("seq2185", old_paths), ("seq2193", current_paths)):
    for path in paths:
      result = PRODUCT.load_json(path)
      summary, _, _ = detrended_slow_events(result["decode_wall_ms"])
      observed = set(summary["slow_indices_after_skip"])
      rows.append({
          "cohort": cohort,
          "path": PRODUCT.relative(path),
          "sha256": sha256(path),
          "slow_count": summary["slow_count"],
          "largest_gap_ms": summary["largest_gap_ms"],
          "split_threshold_ms": summary["split_threshold_ms"],
          "slow_residual_median_ms": summary["slow_residual_median_ms"],
          "expected_missing": sorted(expected_set - observed),
          "unexpected_extra": sorted(observed - expected_set),
      })
  old_rows = [row for row in rows if row["cohort"] == "seq2185"]
  current_rows = [row for row in rows if row["cohort"] == "seq2193"]
  current_sets = []
  for path in current_paths:
    result = PRODUCT.load_json(path)
    summary, _, _ = detrended_slow_events(result["decode_wall_ms"])
    current_sets.append(set(summary["slow_indices_after_skip"]))
  current_intersection = sorted(set.intersection(*current_sets))
  return {
      "expected_slow_indices_after_skip": expected,
      "expected_slow_count": len(expected),
      "rows": rows,
      "seq2185_exact": all(
          not row["expected_missing"] and not row["unexpected_extra"]
          for row in old_rows),
      "seq2193_all_contain_expected": all(
          not row["expected_missing"] for row in current_rows),
      "seq2193_max_extra_count": max(
          len(row["unexpected_extra"]) for row in current_rows),
      "seq2193_intersection": current_intersection,
      "seq2193_intersection_exact": current_intersection == expected,
      "registered_checks_pass": (
          len(old_rows) == 2 and len(current_rows) == 16 and
          all(
              not row["expected_missing"] and not row["unexpected_extra"]
              for row in old_rows) and
          all(not row["expected_missing"] for row in current_rows) and
          max(len(row["unexpected_extra"]) for row in current_rows) <= 1 and
          current_intersection == expected and
          all(
              float(row["largest_gap_ms"]) >= MIN_SPLIT_GAP_MS and
              SLOW_MEDIAN_LOW_MS <=
                  float(row["slow_residual_median_ms"]) <=
                  SLOW_MEDIAN_HIGH_MS
              for row in rows)),
  }


def provider_exact(result: dict[str, Any]) -> bool:
  trace = result.get("lm_head_i8q1_trace") or {}
  selections = trace.get("selection_rows") or []
  prepacks = trace.get("weight_prepack_rows") or []
  return (
      len(selections) == 2 and len(prepacks) == 2 and
      prepacks[0].get("process_cache_hit") is False and
      prepacks[1].get("process_cache_hit") is True and
      all(
          row.get("provider") == EXPECTED_PROVIDER and
          row.get("global") == [248320, 1, 1] and
          row.get("local") == [256, 1, 1] and
          row.get("correction_passes") == 2
          for row in selections))


def stratified_positions(count: int, target: int) -> list[int]:
  if count < target:
    return []
  if target == 1:
    return [0]
  positions = [
      index * (count - 1) // (target - 1)
      for index in range(target)
  ]
  if len(set(positions)) != target:
    raise RuntimeError("stratified selection is not unique")
  return positions


def main() -> int:
  args = parse_args()
  out = args.out_dir.resolve()
  if out.exists() and not args.resume:
    raise SystemExit(f"output already exists: {out}")
  raw = out / "raw"
  raw.mkdir(parents=True, exist_ok=args.resume)

  current_calibration = sorted(BASE_ROOT.glob(CALIBRATION_GLOB))
  required_paths = [
      PRODUCT_TOOL, BASE_CONFIG, BASE_GATE, FALLBACK_BOUND,
      CERTIFICATE_BOUND, CORPUS_SMOKE_REFERENCE, CORPUS_SMOKE_RESULT,
      TEACHER_FORCED_CORPUS, PLUGIN, *OLD_CALIBRATION_RESULTS,
      *current_calibration,
  ]
  missing = [str(path) for path in required_paths if not path.is_file()]
  if missing:
    raise SystemExit("missing slow-event capture inputs: " + ", ".join(missing))

  input_hashes = {
      PRODUCT.relative(path): sha256(path) for path in required_paths
  }
  registered_hashes_match = (
      len(current_calibration) == 16 and
      sha256(BASE_CONFIG) == EXPECTED_BASE_CONFIG_SHA256 and
      sha256(BASE_GATE) == EXPECTED_BASE_GATE_SHA256 and
      sha256(FALLBACK_BOUND) == EXPECTED_FALLBACK_BOUND_SHA256 and
      tuple(sha256(path) for path in OLD_CALIBRATION_RESULTS) ==
          EXPECTED_OLD_CALIBRATION_SHA256 and
      calibration_set_sha256(current_calibration) ==
          EXPECTED_CURRENT_CALIBRATION_SET_SHA256 and
      sha256(CERTIFICATE_BOUND) == EXPECTED_CERTIFICATE_BOUND_SHA256 and
      sha256(CORPUS_SMOKE_REFERENCE) ==
          EXPECTED_CORPUS_SMOKE_REFERENCE_SHA256 and
      sha256(CORPUS_SMOKE_RESULT) == EXPECTED_CORPUS_SMOKE_RESULT_SHA256 and
      sha256(TEACHER_FORCED_CORPUS) ==
          EXPECTED_TEACHER_FORCED_CORPUS_SHA256 and
      sha256(PLUGIN) == EXPECTED_PLUGIN_SHA256)
  if not registered_hashes_match:
    raise SystemExit("registered slow-event capture input hash mismatch")

  git = git_state(out)
  if git["dirty"] or not git["pushed"]:
    raise SystemExit("slow-event capture requires a clean pushed commit")

  fallback = PRODUCT.load_json(FALLBACK_BOUND)
  certificate = PRODUCT.load_json(CERTIFICATE_BOUND)
  base_gate = PRODUCT.load_json(BASE_GATE)
  calibration = classifier_calibration(
      fallback, OLD_CALIBRATION_RESULTS, current_calibration)
  corpus_smoke_reference = PRODUCT.load_json(CORPUS_SMOKE_REFERENCE)
  corpus_smoke_result = PRODUCT.load_json(CORPUS_SMOKE_RESULT)
  corpus_smoke_classification, _, _ = detrended_slow_events(
      corpus_smoke_result["decode_wall_ms"])
  corpus_smoke_admits_long_capture = (
      corpus_smoke_reference.get("source_sha256") ==
          EXPECTED_TEACHER_FORCED_CORPUS_SHA256 and
      corpus_smoke_reference.get("selected_token_count") == 2_048 and
      corpus_smoke_result.get("generated_token_count") == 2_048 and
      float(corpus_smoke_classification["largest_gap_ms"]) >=
          MIN_SPLIT_GAP_MS and
      SLOW_MEDIAN_LOW_MS <=
          float(corpus_smoke_classification["slow_residual_median_ms"]) <=
          SLOW_MEDIAN_HIGH_MS and
      SLOW_RATE_LOW <= float(corpus_smoke_classification["slow_rate"]) <=
          SLOW_RATE_HIGH)
  prior_admission = (
      fallback.get("required_checks_passed") is True and
      certificate.get("required_checks_passed") is True and
      certificate.get("verdict") ==
          "reject_q1_and_fixed_q4_fund_lloyd_q4_slow_event_capture" and
      base_gate.get("run_checks_passed") is True and
      calibration["registered_checks_pass"] is True and
      corpus_smoke_admits_long_capture)
  if not prior_admission:
    raise SystemExit("registered predecessor or classifier gate is not admitted")

  worker_args = SimpleNamespace(
      abort_below_available_gib=MEMORY_STOP_GIB,
      candidate_gpu_plugin=PLUGIN,
      candidate_impls_cache_capacity=None,
      custom_config=PRODUCT.CUSTOM_CONFIG,
      device="GPU",
      min_available_gib=PREFLIGHT_GIB,
      model_dir=PRODUCT.MODEL_DIR,
      openvino_python=PRODUCT.OV_PYTHON,
      pack_gdn_state=False,
      poll_interval_s=1.0,
      prime_candidate_exact_decode_shape=False,
      resume=args.resume,
      timeout_s=args.timeout_s,
      worker_transient_scope=True,
  )
  base_config = PRODUCT.load_json(BASE_CONFIG)
  runs: dict[str, dict[str, Any]] = {}
  for label, matrix_capture in (("capture", True), ("confirm", False)):
    config = {
        **base_config,
        "capture_attention_history_layers": [],
        "capture_attention_history_steps": [],
        "capture_attention_layers": [],
        "capture_attention_steps": [],
        "capture_execution_census": False,
        "capture_lm_head_hidden": True,
        "capture_lm_head_hidden_matrix": matrix_capture,
        "capture_logits": False,
        "capture_prefill_profiles": False,
        "case_id": f"prefill_shape_002k_slow_event_{label}",
        "checkpoint_steps": [],
        "exact_history_capacity": EXACT_HISTORY_CAPACITY,
        "expected_answer": None,
        "host_time_profiling": 0,
        "output_tokens": OUTPUT_TOKENS,
        "prefill_history_capacity": PREFILL_HISTORY_CAPACITY,
        "purpose": "slow_event_population_capture",
        "reference_result": None,
        "teacher_forced_prompt": str(TEACHER_FORCED_CORPUS.resolve()),
        "teacher_forced_prompt_offset": 0,
        "timing_token_output": False,
        "warmup": True,
    }
    runs[label] = PRODUCT.run_worker(
        worker_args, raw / label, config)

  results = {
      label: run.get("result") or {} for label, run in runs.items()}
  workers_ok = all(
      run.get("returncode") == 0 and
      run.get("timed_out") is False and
      run.get("oom_observed") is False and
      (run.get("memory_guard") or {}).get("tripped") is False and
      (run.get("reused") is not True or args.resume)
      for run in runs.values())
  results_complete = all(
      result.get("output_tokens") == OUTPUT_TOKENS and
      result.get("input_token_count") == PROMPT_TOKENS and
      len(result.get("decode_wall_ms") or []) == OUTPUT_TOKENS - 1 and
      len(result.get("generated_token_ids") or []) == OUTPUT_TOKENS
      for result in results.values())

  classifications: dict[str, dict[str, Any]] = {}
  classifier_arrays: dict[str, tuple[list[float], list[float]]] = {}
  if workers_ok and results_complete:
    for label, result in results.items():
      summary, baselines, residuals = detrended_slow_events(
          result["decode_wall_ms"])
      classifications[label] = summary
      classifier_arrays[label] = (baselines, residuals)

  capture_steps = set(
      classifications.get("capture", {}).get("slow_steps", []))
  confirm_steps = set(
      classifications.get("confirm", {}).get("slow_steps", []))
  repeated_steps = sorted(capture_steps & confirm_steps)
  symmetric_difference = sorted(capture_steps ^ confirm_steps)
  selected_positions = stratified_positions(
      len(repeated_steps), TARGET_EVENTS)
  selected_steps = [
      repeated_steps[position] for position in selected_positions]

  selected_matrix_path = out / "slow-lm-head-inputs.f16"
  events_path = out / "slow-events.json"
  selected_matrix = None
  events: list[dict[str, Any]] = []
  matrix_descriptor = results.get("capture", {}).get(
      "lm_head_hidden_matrix") or {}
  matrix_path = raw / "capture" / "lm-head-inputs.f16"
  matrix_input_valid = (
      matrix_descriptor.get("shape") == [OUTPUT_TOKENS, HIDDEN_COLUMNS] and
      matrix_descriptor.get("dtype") == "float16-little-endian" and
      matrix_descriptor.get("byte_count") ==
          OUTPUT_TOKENS * HIDDEN_COLUMNS * 2 and
      matrix_path.is_file() and
      matrix_path.stat().st_size == OUTPUT_TOKENS * HIDDEN_COLUMNS * 2 and
      matrix_descriptor.get("sha256") == sha256(matrix_path))
  if len(selected_steps) == TARGET_EVENTS and matrix_input_valid:
    source = np.memmap(
        matrix_path, dtype="<f2", mode="r",
        shape=(OUTPUT_TOKENS, HIDDEN_COLUMNS))
    selected_matrix = np.memmap(
        selected_matrix_path, dtype="<f2", mode="w+",
        shape=(TARGET_EVENTS, HIDDEN_COLUMNS))
    capture_baselines, capture_residuals = classifier_arrays["capture"]
    confirm_baselines, confirm_residuals = classifier_arrays["confirm"]
    capture_walls = [
        float(value) for value in results["capture"]["decode_wall_ms"]]
    confirm_walls = [
        float(value) for value in results["confirm"]["decode_wall_ms"]]
    tokens = [
        int(value) for value in results["capture"]["generated_token_ids"]]
    for ordinal, step in enumerate(selected_steps):
      decode_index = step - 1
      after_skip_index = decode_index - SKIP_INTERVALS
      selected_matrix[ordinal] = source[step]
      events.append({
          "ordinal": ordinal,
          "step": step,
          "decode_index": decode_index,
          "after_skip_index": after_skip_index,
          "generated_token_id": tokens[step],
          "capture_wall_ms": capture_walls[decode_index],
          "capture_baseline_ms": capture_baselines[after_skip_index],
          "capture_residual_ms": capture_residuals[after_skip_index],
          "confirm_wall_ms": confirm_walls[decode_index],
          "confirm_baseline_ms": confirm_baselines[after_skip_index],
          "confirm_residual_ms": confirm_residuals[after_skip_index],
      })
    selected_matrix.flush()
    PRODUCT.write_json(events_path, {
        "schema": SCHEMA,
        "selection": "chronological_equal-index_stratification",
        "source_repeated_slow_count": len(repeated_steps),
        "target_event_count": TARGET_EVENTS,
        "hidden_columns": HIDDEN_COLUMNS,
        "matrix_file": PRODUCT.relative(selected_matrix_path),
        "events": events,
    })

  scopes = [
      (run.get("worker_transient_scope") or {}).get("unit")
      for run in runs.values()
  ]
  monitors = [run.get("monitor") or {} for run in runs.values()]
  peak_rss = max(
      (int(row.get("process_rss_peak_bytes") or 0) for row in monitors),
      default=0)
  peak_swap = max(
      (int(row.get("process_swap_peak_bytes") or 0) for row in monitors),
      default=0)
  minimum_available = min(
      (int(row.get("system_available_min_bytes") or 0) for row in monitors),
      default=0)
  classification_quality = (
      set(classifications) == {"capture", "confirm"} and
      all(
          float(row["largest_gap_ms"]) >= MIN_SPLIT_GAP_MS and
          SLOW_MEDIAN_LOW_MS <=
              float(row["slow_residual_median_ms"]) <=
              SLOW_MEDIAN_HIGH_MS and
          SLOW_RATE_LOW <= float(row["slow_rate"]) <= SLOW_RATE_HIGH
          for row in classifications.values()))
  max_symmetric_difference = max(32, math.ceil(
      0.02 * max(1, len(repeated_steps))))
  selected_matrix_valid = (
      selected_matrix_path.is_file() and
      selected_matrix_path.stat().st_size ==
          TARGET_EVENTS * HIDDEN_COLUMNS * 2 and
      events_path.is_file() and len(events) == TARGET_EVENTS and
      len(selected_steps) == len(set(selected_steps)) == TARGET_EVENTS)

  checks = [
      check("repository_clean_and_pushed_at_gate",
            not git["dirty"] and git["pushed"], git=git),
      check("registered_inputs_match_exact_hashes",
            registered_hashes_match, input_hashes=input_hashes),
      check("predecessor_gates_admit_only_population_capture",
            prior_admission,
            corpus_smoke_classification=corpus_smoke_classification),
      check("detrended_classifier_reproduces_registered_18_rows",
            calibration["registered_checks_pass"],
            calibration=calibration),
      check("two_fresh_candidate_workers_complete_strictly_serially",
            workers_ok and list(runs) == ["capture", "confirm"] and
            len(scopes) == len(set(scopes)) == 2 and
            all(scope for scope in scopes),
            scopes=scopes, workers_concurrent=False),
      check("long_teacher_forced_configs_match_registered_geometry",
            results_complete and all(
                result.get("teacher_forced") is True and
                result.get("teacher_forced_from_stock") is False and
                (result.get("teacher_forced_prompt") or {}).get(
                    "file_sha256") ==
                    EXPECTED_TEACHER_FORCED_CORPUS_SHA256 and
                (result.get("teacher_forced_prompt") or {}).get(
                    "selected_token_count") == OUTPUT_TOKENS and
                result.get("exact_history_capacity") ==
                    EXACT_HISTORY_CAPACITY and
                result.get("prefill_history_capacity") ==
                    PREFILL_HISTORY_CAPACITY and
                result.get("timing_token_output") is False and
                result.get("capture_lm_head_hidden") is True
                for result in results.values())),
      check("accepted_seq2189_exact_fallback_provider_is_selected",
            sha256(PLUGIN) == EXPECTED_PLUGIN_SHA256 and
            all(
                result.get("candidate_gpu_plugin_sha256") ==
                    EXPECTED_PLUGIN_SHA256 and
                result.get("lm_head_i8q1") is True and
                result.get("lm_head_i8q1_gated_exact") is True and
                result.get("lm_head_i8q1_greedy_local2") is False and
                provider_exact(result)
                for result in results.values())),
      check("both_long_workers_generate_the_identical_candidate_stream",
            results_complete and
            results.get("capture", {}).get("generated_token_ids") ==
                results.get("confirm", {}).get("generated_token_ids") and
            results.get("capture", {}).get("generated_token_ids_sha256") ==
                results.get("confirm", {}).get("generated_token_ids_sha256"),
            token_sha256=results.get("capture", {}).get(
                "generated_token_ids_sha256")),
      check("long_slow_modes_match_registered_shape",
            classification_quality,
            classifications=classifications),
      check("repeated_slow_population_is_large_and_stable",
            len(repeated_steps) >= TARGET_EVENTS and
            len(symmetric_difference) <= max_symmetric_difference,
            repeated_count=len(repeated_steps),
            symmetric_difference_count=len(symmetric_difference),
            symmetric_difference_limit=max_symmetric_difference,
            symmetric_difference_first64=symmetric_difference[:64]),
      check("capture_matrix_is_complete_and_exactly_f16",
            matrix_input_valid, descriptor=matrix_descriptor),
      check("exactly_2000_context_spanning_rows_are_materialized",
            selected_matrix_valid,
            first_step=selected_steps[0] if selected_steps else None,
            last_step=selected_steps[-1] if selected_steps else None,
            selected_matrix_sha256=(
                sha256(selected_matrix_path)
                if selected_matrix_path.is_file() else None),
            events_sha256=(
                sha256(events_path) if events_path.is_file() else None)),
      check("memory_guards_hold_without_oom",
            workers_ok and minimum_available >=
                int(MEMORY_STOP_GIB * 1024**3),
            peak_rss_bytes=peak_rss,
            peak_swap_bytes=peak_swap,
            minimum_available_bytes=minimum_available,
            stop_bytes=int(MEMORY_STOP_GIB * 1024**3)),
      check("capture_does_not_admit_integration_or_speed_claim",
            True, plugin_builds=0, stock_workers=0,
            workers_concurrent=False),
  ]
  required = all(row["pass"] for row in checks)
  verdict = (
      "admit_2000_slow_rows_for_offline_lloyd_q4_certificate"
      if required else
      "repair_or_extend_slow_event_population_capture")
  payload = {
      "schema": SCHEMA,
      "workstream": WS,
      "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
      "git": git,
      "verdict": verdict,
      "required_checks_passed": required,
      "source_or_plugin_integration_admitted": False,
      "performance_claim_admitted": False,
      "workers_concurrent": False,
      "workers": {
          "model_workers": 2,
          "candidate_workers": 2,
          "stock_workers": 0,
          "compiler_or_plugin_builds": 0,
      },
      "registered_geometry": {
          "output_tokens": OUTPUT_TOKENS,
          "prompt_tokens": PROMPT_TOKENS,
          "exact_history_capacity": EXACT_HISTORY_CAPACITY,
          "prefill_history_capacity": PREFILL_HISTORY_CAPACITY,
          "target_events": TARGET_EVENTS,
          "hidden_columns": HIDDEN_COLUMNS,
          "teacher_forced_corpus": PRODUCT.relative(TEACHER_FORCED_CORPUS),
          "teacher_forced_corpus_sha256":
              EXPECTED_TEACHER_FORCED_CORPUS_SHA256,
      },
      "calibration": calibration,
      "classifications": classifications,
      "repeated_population": {
          "count": len(repeated_steps),
          "steps": repeated_steps,
          "symmetric_difference_count": len(symmetric_difference),
          "symmetric_difference_steps": symmetric_difference,
      },
      "selected_population": {
          "count": len(selected_steps),
          "steps": selected_steps,
          "selection": "chronological_equal-index_stratification",
          "matrix": {
              "path": PRODUCT.relative(selected_matrix_path),
              "shape": [TARGET_EVENTS, HIDDEN_COLUMNS],
              "dtype": "float16-little-endian",
              "byte_count": (
                  selected_matrix_path.stat().st_size
                  if selected_matrix_path.is_file() else None),
              "sha256": (
                  sha256(selected_matrix_path)
                  if selected_matrix_path.is_file() else None),
          },
          "events": {
              "path": PRODUCT.relative(events_path),
              "sha256": sha256(events_path) if events_path.is_file() else None,
          },
      },
      "memory": {
          "preflight_gib": PREFLIGHT_GIB,
          "abort_below_available_gib": MEMORY_STOP_GIB,
          "peak_rss_bytes": peak_rss,
          "peak_swap_bytes": peak_swap,
          "minimum_available_bytes": minimum_available,
      },
      "checks": checks,
      "runs": runs,
      "next_gate": {
          "route": "offline_lloyd_q4_global_l2_population_certificate",
          "requirements": [
              "consume exactly the selected 2000 F16 hidden rows",
              "require zero global-L2 bound violations",
              "require every accepted exact fallback token to be returned",
              "require no fixed candidate-capacity overflow",
              "require worst active traffic at or below 0.60x full I8",
              "do not integrate source/plugin code before this gate passes",
          ],
      },
  }
  PRODUCT.write_json(out / "metrics.json", payload)
  PRODUCT.write_json(out / "manifest.json", {
      "schema": SCHEMA,
      "tool": PRODUCT.relative(Path(__file__)),
      "tool_sha256": sha256(Path(__file__)),
      "product_tool_sha256": sha256(PRODUCT_TOOL),
      "git": git,
      "inputs": input_hashes,
      "plugin": {
          "path": str(PLUGIN),
          "sha256": sha256(PLUGIN),
      },
      "workers": payload["workers"],
      "workers_concurrent": False,
      "output_files": {
          PRODUCT.relative(path): {
              "bytes": path.stat().st_size,
              "sha256": sha256(path),
          }
          for path in (selected_matrix_path, events_path)
          if path.is_file()
      },
  })
  report = f"""# LM-head reproducible slow-event capture

Verdict: **{verdict}**. Required checks: `{str(required).lower()}`.

Two fresh seq2189 candidate workers ran strictly serially over
`{OUTPUT_TOKENS:,}` teacher-forced code tokens. Their candidate output streams
are identical. The registered drift-resistant classifier finds
`{classifications.get('capture', {}).get('slow_count', 0):,}` and
`{classifications.get('confirm', {}).get('slow_count', 0):,}` slow intervals;
`{len(repeated_steps):,}` indices repeat, with
`{len(symmetric_difference):,}` non-repeated classifications.

The capture materializes exactly `{len(selected_steps):,}` stratified F16
hidden rows spanning steps
`{selected_steps[0] if selected_steps else 'unavailable'}` through
`{selected_steps[-1] if selected_steps else 'unavailable'}`. Peak worker RSS
and swap are `{peak_rss:,}` and `{peak_swap:,}` B; minimum available memory is
`{minimum_available:,}` B. No worker is concurrent, and the 8/4 GiB
preflight/abort guards remain active.

This artifact admits only the offline Lloyd-Q4/global-L2 population
certificate. It does not admit source/plugin integration or a speed claim.
"""
  (out / "report.md").write_text(report, encoding="utf-8")
  print(json.dumps({
      "artifact": PRODUCT.relative(out),
      "verdict": verdict,
      "required_checks_passed": required,
      "capture_slow_count":
          classifications.get("capture", {}).get("slow_count"),
      "confirm_slow_count":
          classifications.get("confirm", {}).get("slow_count"),
      "repeated_slow_count": len(repeated_steps),
      "selected_count": len(selected_steps),
      "peak_rss_bytes": peak_rss,
      "peak_swap_bytes": peak_swap,
      "minimum_available_bytes": minimum_available,
  }, separators=(",", ":")), flush=True)
  return 0 if required else 2


if __name__ == "__main__":
  raise SystemExit(main())
