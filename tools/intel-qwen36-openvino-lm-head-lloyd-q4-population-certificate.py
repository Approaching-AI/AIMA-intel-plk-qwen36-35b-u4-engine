#!/usr/bin/env python3
"""Prove the Lloyd-Q4 LM-head certificate on 2000 slow-event rows.

This CPU-only gate consumes the admitted seq2287 population.  It evaluates the
per-row 16-centroid Lloyd codec against all 248,320 vocabulary rows and all
2000 captured hidden states.  The gate scans the mathematical F16 reference
logits as well as the conservative global-L2 upper bounds, so token recovery,
bound domination, candidate capacity, and active traffic are all measured on
the full 496.64 million row/event pairs.

No model, compiler, GPU context, or product worker is created.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import math
import resource
import subprocess
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = (
    "intel-qwen36-openvino-lm-head-lloyd-q4-"
    "population-certificate-v1")
CERTIFICATE_TOOL = ROOT / (
    "tools/intel-qwen36-openvino-lm-head-exact-token-"
    "certificate-bound.py")
MODEL_DIR = Path("/home/intel/Qwen3.6-35B-A3B-ov")
MODEL_XML = MODEL_DIR / "openvino_language_model.xml"
MODEL_BIN = MODEL_DIR / "openvino_language_model.bin"
PRIOR_CERTIFICATE = ROOT / (
    "output/openvino-lm-head-exact-token-certificate-bound-"
    "20260801Tseq2280-lloyd-clean/metrics.json")
POPULATION_ROOT = ROOT / (
    "output/openvino-lm-head-slow-event-population-bound-"
    "20260801Tseq2287-clean")
POPULATION_METRICS = POPULATION_ROOT / "metrics.json"
POPULATION_MANIFEST = POPULATION_ROOT / "manifest.json"
HIDDEN_MATRIX = POPULATION_ROOT / "slow-lm-head-inputs.f16"
EVENTS = POPULATION_ROOT / "slow-events.json"
COMPONENT_RESULT = ROOT / (
    "output/openvino-lm-head-gated-exact-component-"
    "20260731Tseq2187-clean/result.json")
OPPORTUNITY_RESULT = ROOT / (
    "output/openvino-post-pr35924-opportunity-bound-"
    "20260801Tseq2278-clean/metrics.json")

EXPECTED_SHA256 = {
    CERTIFICATE_TOOL:
        "6a4ff114ab8b2722950a8a6afd6207e5501c7d5571b7eb32621259ad531c1f5e",
    MODEL_XML:
        "fae1047f6a758ded4fab95f5faee9bf68f92b4433d778496bd9d44efa51cdbb0",
    MODEL_BIN:
        "46140b595760e891d9626c5bfaffc2c998cce176d0de7f6c290af5ae1f2393a4",
    PRIOR_CERTIFICATE:
        "a2003ff9cfdd401a6f5f4915193738e48c66068e3a226bf1dd3076d52d919d8a",
    POPULATION_METRICS:
        "82f53963168367227ee9e621fe6e8f64b8609f7b408e20e2c950c2226fde6fb8",
    POPULATION_MANIFEST:
        "99c21f0c52d9c3946ee5c75d1ccf48c19e8d5fd77dcaf19c06ad90c0370883cd",
    HIDDEN_MATRIX:
        "48e8bd79c03f1adcbc1fbb5d784aa306c859272e46f186de2ccf0c0ce999f836",
    EVENTS:
        "5a1b7aa5954c7fd5b0f25032f7c9cf0a43e7cf3e7cc331bad87ad1e721f9caf9",
    COMPONENT_RESULT:
        "caf1814a1786f74e637b5aa398455bac64a831d4dd5fa22557a7def0919d9a73",
    OPPORTUNITY_RESULT:
        "029facf058d3201613785a7aacf8a2bb7d6d6b114e3eae49bc02daa17dff5752",
}

ROWS = 248_320
COLUMNS = 2_048
EVENT_COUNT = 2_000
ACTIVE_BYTE_RATIO_CAP = 0.60
PREFLIGHT_AVAILABLE_BYTES = 8 * 1024**3
ABORT_AVAILABLE_BYTES = 4 * 1024**3


def load_module(name: str, path: Path) -> Any:
  spec = importlib.util.spec_from_file_location(name, path)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import {path}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


CERT = load_module("iq36_lm_head_population_certificate_prior", CERTIFICATE_TOOL)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out-dir", type=Path, required=True)
  parser.add_argument("--row-chunk", type=int, default=256)
  args = parser.parse_args()
  if args.row_chunk <= 0:
    parser.error("row chunk must be positive")
  return args


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    while chunk := stream.read(8 * 1024 * 1024):
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


def relative(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path.resolve())


def run(command: list[str]) -> str:
  return subprocess.run(
      command, cwd=ROOT, check=True, capture_output=True, text=True,
      encoding="utf-8", errors="replace").stdout.strip()


def git_state() -> dict[str, Any]:
  status = run(["git", "status", "--porcelain=v1", "--untracked-files=all"])
  commit = run(["git", "rev-parse", "HEAD"])
  upstream = run(["git", "rev-parse", "@{upstream}"])
  return {
      "branch": run(["git", "branch", "--show-current"]),
      "commit": commit,
      "dirty": bool(status),
      "dirty_paths": [line[3:] for line in status.splitlines() if line],
      "pushed": commit == upstream,
      "upstream_commit": upstream,
  }


def available_memory_bytes() -> int:
  for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
    if line.startswith("MemAvailable:"):
      return int(line.split()[1]) * 1024
  raise RuntimeError("MemAvailable is absent from /proc/meminfo")


def check(name: str, passed: bool, **details: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **details}


def percentile(values: Any, probability: float, np: Any) -> float:
  return float(np.percentile(
      np.asarray(values, dtype=np.float64), probability))


def f16_reference(values: Any, np: Any) -> Any:
  return values.astype(np.float16).astype(np.float32)


def main() -> int:
  args = parse_args()
  out = args.out_dir.resolve()
  if out.exists():
    raise SystemExit(f"output already exists: {out}")
  memory_start = available_memory_bytes()
  if memory_start < PREFLIGHT_AVAILABLE_BYTES:
    raise SystemExit(
        f"preflight memory below {PREFLIGHT_AVAILABLE_BYTES}: {memory_start}")

  import numpy as np

  started = time.monotonic()
  git = git_state()
  if git["dirty"] or not git["pushed"]:
    raise SystemExit("population certificate requires a clean pushed commit")
  missing = [str(path) for path in EXPECTED_SHA256 if not path.is_file()]
  if missing:
    raise SystemExit(
        "missing population-certificate inputs: " + ", ".join(missing))
  observed_hashes = {path: sha256(path) for path in EXPECTED_SHA256}
  hash_mismatches = {
      relative(path): {
          "expected": expected,
          "observed": observed_hashes[path],
      }
      for path, expected in EXPECTED_SHA256.items()
      if observed_hashes[path] != expected
  }
  if hash_mismatches:
    raise SystemExit(
        "registered population-certificate hash mismatch: " +
        json.dumps(hash_mismatches, sort_keys=True))

  prior = load_json(PRIOR_CERTIFICATE)
  population = load_json(POPULATION_METRICS)
  events_payload = load_json(EVENTS)
  events = events_payload.get("events") or []
  if len(events) != EVENT_COUNT:
    raise SystemExit(f"expected {EVENT_COUNT} population events")
  if [int(row["ordinal"]) for row in events] != list(range(EVENT_COUNT)):
    raise SystemExit("population event ordinals are not exact")
  seed_ids = np.asarray(
      [int(row["generated_token_id"]) for row in events], dtype=np.int64)
  if np.any(seed_ids < 0) or np.any(seed_ids >= ROWS):
    raise SystemExit("population contains an invalid exact token id")

  weight_fact = CERT.ir_constant(CERT.WEIGHT_NAME)
  scale_fact = CERT.ir_constant(CERT.SCALE_NAME)
  if (
      weight_fact != {
          "element_type": "i8", "offset": 18_137_149_498,
          "shape": [ROWS, COLUMNS], "size": ROWS * COLUMNS}
      or scale_fact != {
          "element_type": "f16", "offset": 18_645_708_858,
          "shape": [ROWS, 1], "size": ROWS * 2}
  ):
    raise SystemExit(
        f"LM-head IR facts drifted: {weight_fact}, {scale_fact}")

  hidden_memmap = np.memmap(
      HIDDEN_MATRIX, dtype="<f2", mode="r",
      shape=(EVENT_COUNT, COLUMNS))
  hidden = np.asarray(hidden_memmap, dtype=np.float32)
  if not np.array_equal(
      hidden, hidden.astype(np.float16).astype(np.float32)):
    raise SystemExit("population hidden matrix is not exactly F16-valued")
  quantized_hidden = CERT.q8_hidden(hidden, np)
  hidden_delta = hidden - quantized_hidden
  hidden_norm = CERT.upward_hidden_norm(hidden, np)
  hidden_delta_norm = CERT.upward_hidden_norm(hidden_delta, np)
  hidden_t = np.ascontiguousarray(hidden.T, dtype=np.float32)
  quantized_hidden_t = np.ascontiguousarray(
      quantized_hidden.T, dtype=np.float32)

  weights = np.memmap(
      MODEL_BIN, dtype=np.int8, mode="r", offset=weight_fact["offset"],
      shape=(ROWS, COLUMNS))
  scales_memmap = np.memmap(
      MODEL_BIN, dtype="<f2", mode="r", offset=scale_fact["offset"],
      shape=(ROWS,))
  scales = np.asarray(scales_memmap, dtype=np.float32)

  seed_weights = np.asarray(weights[seed_ids], dtype=np.float64)
  seed_real = np.sum(
      seed_weights * hidden.astype(np.float64),
      axis=1, dtype=np.float64)
  seed_real *= scales[seed_ids].astype(np.float64)
  seed_values = f16_reference(seed_real, np)
  del seed_weights

  selection = np.empty((EVENT_COUNT, ROWS), dtype=np.bool_)
  reference_best_values = np.full(
      EVENT_COUNT, -np.inf, dtype=np.float32)
  reference_best_ids = np.full(
      EVENT_COUNT, ROWS, dtype=np.int64)
  bound_violation_count = 0
  bound_violation_first = []
  upper_minus_reference_min = math.inf
  memory_min = memory_start
  last_progress = time.monotonic()

  for begin in range(0, ROWS, args.row_chunk):
    current_memory = available_memory_bytes()
    memory_min = min(memory_min, current_memory)
    if current_memory < ABORT_AVAILABLE_BYTES:
      raise SystemExit(
          f"runtime memory fell below {ABORT_AVAILABLE_BYTES} at row {begin}")
    end = min(ROWS, begin + args.row_chunk)
    raw = np.asarray(weights[begin:end], dtype=np.float32)
    lloyd = CERT.lloyd16_dequantize(raw, np)
    codec_norm = CERT.upward_f32_norm(lloyd, np)
    residual_norm = CERT.upward_f32_norm(raw - lloyd, np)
    approximate = (
        lloyd @ quantized_hidden_t *
        scales[begin:end, None]).astype(np.float32)
    residual_bound = np.abs(scales[begin:end, None]).astype(np.float64) * (
        residual_norm[:, None] * hidden_norm[None, :] +
        codec_norm[:, None] * hidden_delta_norm[None, :])
    round_guard = CERT.f32_dot_guard(
        codec_norm[:, None], hidden_norm[None, :],
        scales[begin:end, None], np)
    upper_output = CERT.f16_upper(
        approximate.astype(np.float64) + residual_bound + round_guard, np)

    reference_output = f16_reference(
        raw @ hidden_t * scales[begin:end, None], np)
    violation_rows, violation_events = np.nonzero(
        reference_output > upper_output)
    bound_violation_count += int(violation_rows.size)
    for row_offset, event in zip(
        violation_rows[:8 - len(bound_violation_first)],
        violation_events[:8 - len(bound_violation_first)],
    ):
      bound_violation_first.append({
          "event": int(event),
          "row": int(begin + row_offset),
          "reference": float(reference_output[row_offset, event]),
          "upper": float(upper_output[row_offset, event]),
      })
    upper_minus_reference_min = min(
        upper_minus_reference_min,
        float(np.min(
            upper_output.astype(np.float32) - reference_output)))

    selection[:, begin:end] = (
        upper_output.T >= seed_values[:, None])
    local_offsets = np.argmax(reference_output, axis=0)
    event_indices = np.arange(EVENT_COUNT)
    local_values = reference_output[local_offsets, event_indices]
    local_ids = begin + local_offsets.astype(np.int64)
    replace = (
        (local_values > reference_best_values) |
        ((local_values == reference_best_values) &
         (local_ids < reference_best_ids)))
    reference_best_values[replace] = local_values[replace]
    reference_best_ids[replace] = local_ids[replace]

    now = time.monotonic()
    if now - last_progress >= 20.0:
      print(json.dumps({
          "elapsed_seconds": round(now - started, 3),
          "event": "lloyd_q4_population_certificate_progress",
          "rows_complete": end,
          "rows_total": ROWS,
      }, sort_keys=True), flush=True)
      last_progress = now

  candidate_counts = np.sum(selection, axis=1, dtype=np.int64)
  seed_included = selection[np.arange(EVENT_COUNT), seed_ids]
  token_matches = reference_best_ids == seed_ids
  traffic_rows = [
      CERT.active_bytes(
          CERT.Q4_LLOYD_PACKED_BYTES, int(count),
          include_materialized_output=True)
      for count in candidate_counts
  ]
  traffic_totals = np.asarray(
      [row["total_bytes"] for row in traffic_rows], dtype=np.int64)
  traffic_ratios = (
      traffic_totals.astype(np.float64) /
      float(CERT.FULL_I8_SCAN_BYTES))
  fixed_traffic_bytes = CERT.active_bytes(
      CERT.Q4_LLOYD_PACKED_BYTES, 0,
      include_materialized_output=True)["total_bytes"]
  per_candidate_bytes = CERT.ROW_EXACT_BYTES + 4 + 2
  fixed_candidate_capacity = math.floor(
      (
          ACTIVE_BYTE_RATIO_CAP * CERT.FULL_I8_SCAN_BYTES -
          fixed_traffic_bytes
      ) / per_candidate_bytes)
  worst_event = int(np.argmax(traffic_totals))
  maximum_active_bytes = int(traffic_totals[worst_event])
  maximum_ratio = float(traffic_ratios[worst_event])

  component = load_json(COMPONENT_RESULT)
  opportunity = load_json(OPPORTUNITY_RESULT)
  bandwidth_lcb = float(
      component["bound"]["accepted_bandwidth_lcb_gb_s"])
  required_saving_ms = (
      float(component["bound"]["required_saving_us"]) / 1000.0)
  product_fallback_rate = float(
      opportunity["route_comparison"]["lm_head_exact_fallback"][
          "observed_fallback_rate"])
  full_floor_ms = (
      CERT.FULL_I8_SCAN_BYTES / (bandwidth_lcb * 1.0e9) * 1.0e3)
  worst_floor_ms = (
      maximum_active_bytes / (bandwidth_lcb * 1.0e9) * 1.0e3)
  optimistic_worst_saving_ms = full_floor_ms - worst_floor_ms
  optimistic_product_mean_saving_ms = (
      optimistic_worst_saving_ms * product_fallback_rate)

  rows = []
  for ordinal, event in enumerate(events):
    rows.append({
        "ordinal": ordinal,
        "step": int(event["step"]),
        "seed_id": int(seed_ids[ordinal]),
        "seed_value_f16": float(seed_values[ordinal]),
        "seed_included": bool(seed_included[ordinal]),
        "reference_top1": int(reference_best_ids[ordinal]),
        "reference_top1_value_f16":
            float(reference_best_values[ordinal]),
        "exact_token_recovered": bool(token_matches[ordinal]),
        "exact_candidate_rows": int(candidate_counts[ordinal]),
        "traffic": traffic_rows[ordinal],
        "traffic_ratio_vs_full_i8": float(traffic_ratios[ordinal]),
    })

  checks = [
      check("repository_clean_and_pushed_at_gate",
            git["branch"] == "main" and not git["dirty"] and git["pushed"],
            git=git),
      check("registered_inputs_match_exact_hashes",
            not hash_mismatches,
            inputs={
                relative(path): observed_hashes[path]
                for path in observed_hashes
            }),
      check("seq2280_admits_only_lloyd_q4_population_certificate",
            prior.get("required_checks_passed") is True and
            prior.get("verdict") ==
                "reject_q1_and_fixed_q4_fund_lloyd_q4_slow_event_capture"),
      check("seq2287_population_is_exact_and_admitted",
            population.get("required_checks_passed") is True and
            population.get("verdict") ==
                "admit_2000_high_confidence_rows_for_offline_lloyd_q4_certificate" and
            population.get("selected_population", {}).get("count") ==
                EVENT_COUNT and
            events_payload.get("target_event_count") == EVENT_COUNT),
      check("locked_ir_and_population_geometry_match",
            hidden.shape == (EVENT_COUNT, COLUMNS) and
            weight_fact["shape"] == [ROWS, COLUMNS] and
            scale_fact["shape"] == [ROWS, 1]),
      check("captured_hidden_is_exactly_f16_valued",
            bool(np.array_equal(
                hidden, hidden.astype(np.float16).astype(np.float32)))),
      check("global_l2_bound_dominates_all_496m64_reference_scores",
            bound_violation_count == 0,
            evaluated_row_event_pairs=ROWS * EVENT_COUNT,
            violation_count=bound_violation_count,
            violation_first=bound_violation_first,
            minimum_f16_upper_minus_reference=upper_minus_reference_min),
      check("every_repeated_exact_fallback_token_is_returned",
            int(np.count_nonzero(token_matches)) == EVENT_COUNT,
            recovered_count=int(np.count_nonzero(token_matches)),
            mismatch_first=[
                {
                    "ordinal": int(index),
                    "step": int(events[index]["step"]),
                    "expected": int(seed_ids[index]),
                    "observed": int(reference_best_ids[index]),
                }
                for index in np.flatnonzero(~token_matches)[:16]
            ]),
      check("every_exact_fallback_seed_is_selected",
            int(np.count_nonzero(seed_included)) == EVENT_COUNT,
            selected_seed_count=int(np.count_nonzero(seed_included))),
      check("fixed_candidate_capacity_has_no_population_overflow",
            int(np.max(candidate_counts)) <= fixed_candidate_capacity,
            capacity=fixed_candidate_capacity,
            maximum=int(np.max(candidate_counts)),
            overflow_count=int(np.count_nonzero(
                candidate_counts > fixed_candidate_capacity))),
      check("worst_active_traffic_clears_registered_0p60_cap",
            maximum_ratio <= ACTIVE_BYTE_RATIO_CAP,
            cap=ACTIVE_BYTE_RATIO_CAP,
            worst_event=worst_event,
            worst_step=int(events[worst_event]["step"]),
            maximum_active_bytes=maximum_active_bytes,
            maximum_ratio=maximum_ratio),
      check("optimistic_traffic_headroom_funds_product_kill_number",
            optimistic_worst_saving_ms > 0.0 and
            optimistic_product_mean_saving_ms >= required_saving_ms,
            optimistic_worst_event_saving_ms=
                optimistic_worst_saving_ms,
            product_fallback_rate=product_fallback_rate,
            optimistic_product_mean_saving_ms=
                optimistic_product_mean_saving_ms,
            required_product_saving_ms=required_saving_ms),
      check("memory_guards_hold_without_oom",
            memory_min >= ABORT_AVAILABLE_BYTES,
            preflight_bytes=PREFLIGHT_AVAILABLE_BYTES,
            abort_below_bytes=ABORT_AVAILABLE_BYTES,
            available_at_start_bytes=memory_start,
            available_min_bytes=memory_min),
      check("no_model_compiler_or_gpu_worker_ran",
            True, model_workers=0, product_workers=0,
            compiler_invocations=0, gpu_contexts=0),
  ]
  required = all(row["pass"] for row in checks)
  verdict = (
      "admit_lloyd_q4_certificate_component_implementation"
      if required else
      "reject_lloyd_q4_population_certificate")
  result = {
      "schema": SCHEMA,
      "workstream": WS,
      "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
      "git": git,
      "verdict": verdict,
      "required_checks_passed": required,
      "component_implementation_admitted": required,
      "product_promotion_admitted": False,
      "performance_claim_admitted": False,
      "checks": checks,
      "inputs": {
          "population_event_count": EVENT_COUNT,
          "vocabulary_rows": ROWS,
          "hidden_columns": COLUMNS,
          "evaluated_row_event_pairs": ROWS * EVENT_COUNT,
          "registered_sha256": {
              relative(path): observed_hashes[path]
              for path in observed_hashes
          },
      },
      "codec": {
          "name": "per_row_lloyd_q4_global_l2",
          "levels": CERT.Q4_LLOYD_LEVELS,
          "lloyd_iterations": 5,
          "packed_bytes": CERT.Q4_LLOYD_PACKED_BYTES,
          "fixed_candidate_capacity": fixed_candidate_capacity,
          "active_byte_ratio_cap": ACTIVE_BYTE_RATIO_CAP,
      },
      "population": {
          "certificate_pass_count": int(np.count_nonzero(
              token_matches & seed_included)),
          "zero_bound_violation_event_count": (
              EVENT_COUNT if bound_violation_count == 0 else None),
          "exact_candidate_rows": {
              "minimum": int(np.min(candidate_counts)),
              "p50": percentile(candidate_counts, 50, np),
              "p95": percentile(candidate_counts, 95, np),
              "p99": percentile(candidate_counts, 99, np),
              "maximum": int(np.max(candidate_counts)),
          },
          "traffic": {
              "full_i8_scan_bytes": CERT.FULL_I8_SCAN_BYTES,
              "maximum_active_bytes": maximum_active_bytes,
              "maximum_ratio_vs_full_i8": maximum_ratio,
              "worst_event": worst_event,
              "worst_step": int(events[worst_event]["step"]),
              "accepted_bandwidth_lcb_gb_s": bandwidth_lcb,
              "full_i8_floor_ms": full_floor_ms,
              "optimistic_worst_event_floor_ms": worst_floor_ms,
              "optimistic_worst_event_saving_ms":
                  optimistic_worst_saving_ms,
              "product_fallback_rate": product_fallback_rate,
              "optimistic_product_mean_saving_ms":
                  optimistic_product_mean_saving_ms,
              "required_product_saving_ms": required_saving_ms,
          },
          "rows": rows,
      },
      "memory": {
          "preflight_bytes": PREFLIGHT_AVAILABLE_BYTES,
          "abort_below_bytes": ABORT_AVAILABLE_BYTES,
          "available_at_start_bytes": memory_start,
          "available_min_bytes": memory_min,
          "peak_rss_bytes":
              resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
          "oom_observed": False,
      },
      "workers": {
          "model_workers": 0,
          "product_workers": 0,
          "compiler_invocations": 0,
          "gpu_contexts": 0,
          "workers_concurrent": False,
      },
      "next_gate": {
          "route": (
              "bounded_gpu_lloyd_q4_certificate_component_implementation"
              if required else "close_lloyd_q4_certificate_route"),
          "requirements": [
              "implement one parameterized GPU component, not per-row source",
              "preserve outward global-L2 bounds and exact fallback",
              "measure actual component latency and bytes",
              "retain full I8 fallback when fixed capacity is exceeded",
              "do not claim product speedup before correctness and ABBA gates",
          ],
      },
  }
  out.mkdir(parents=True)
  write_json(out / "metrics.json", result)
  write_json(out / "manifest.json", {
      "schema": SCHEMA,
      "tool": relative(Path(__file__)),
      "tool_sha256": sha256(Path(__file__)),
      "git": git,
      "inputs": result["inputs"]["registered_sha256"],
      "workers": result["workers"],
  })
  report = f"""# Lloyd-Q4 LM-head population certificate

Verdict: **{verdict}**. Required checks: `{str(required).lower()}`.

The gate evaluates `{ROWS * EVENT_COUNT:,}` full vocabulary/event pairs.
Bound violations are `{bound_violation_count:,}` and exact token recovery is
`{int(np.count_nonzero(token_matches)):,}/{EVENT_COUNT:,}`. Candidate rows
min/p50/p95/p99/max are `{int(np.min(candidate_counts)):,}` /
`{percentile(candidate_counts, 50, np):.1f}` /
`{percentile(candidate_counts, 95, np):.1f}` /
`{percentile(candidate_counts, 99, np):.1f}` /
`{int(np.max(candidate_counts)):,}`.

Worst active traffic is `{maximum_active_bytes:,}` B, or
`{maximum_ratio:.6f}x` the full-I8 scan, against the registered `0.60x` cap.
The traffic-only product mean saving is
`{optimistic_product_mean_saving_ms:.6f}` ms versus the
`{required_saving_ms:.6f}` ms kill-number. This is still not a latency or
product speedup claim.
"""
  (out / "report.md").write_text(report, encoding="utf-8")
  print(json.dumps({
      "artifact": relative(out),
      "elapsed_seconds": round(time.monotonic() - started, 3),
      "verdict": verdict,
      "required_checks_passed": required,
      "bound_violation_count": bound_violation_count,
      "exact_token_recovery_count": int(np.count_nonzero(token_matches)),
      "candidate_rows_maximum": int(np.max(candidate_counts)),
      "maximum_active_byte_ratio": maximum_ratio,
      "peak_rss_bytes": result["memory"]["peak_rss_bytes"],
      "minimum_available_bytes": memory_min,
  }, separators=(",", ":")), flush=True)
  return 0 if required else 2


if __name__ == "__main__":
  raise SystemExit(main())
