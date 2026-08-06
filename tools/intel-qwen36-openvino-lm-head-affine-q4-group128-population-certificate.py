#!/usr/bin/env python3
"""Prove an affine-Q4/group128 LM-head certificate on 2000 slow rows.

The codec stores one 4-bit code per weight and exact int8 min/max metadata for
each 128-column group.  Its scale is derived at runtime.  One outward-rounded
F16 residual norm is stored per group; the codec norm is derived while reading
the already-required Q4 codes.  The groupwise global-L2 theorem is checked
against every mathematical F16 reference score across all 496.64 million
vocabulary/event pairs.

This is CPU-only evidence.  It creates no model, compiler, GPU context, or
product worker.
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
    "intel-qwen36-openvino-lm-head-affine-q4-group128-"
    "population-certificate-v1")
PRIOR_TOOL = ROOT / (
    "tools/intel-qwen36-openvino-lm-head-lloyd-q4-"
    "population-certificate.py")
PRIOR_REJECTION_ROOT = ROOT / (
    "output/openvino-lm-head-lloyd-q4-population-certificate-"
    "20260801Tseq2288-clean")
PRIOR_REJECTION = PRIOR_REJECTION_ROOT / "metrics.json"
PRIOR_REJECTION_MANIFEST = PRIOR_REJECTION_ROOT / "manifest.json"

ROWS = 248_320
COLUMNS = 2_048
EVENT_COUNT = 2_000
GROUP_SIZE = 128
GROUPS = COLUMNS // GROUP_SIZE
LEVELS = 16
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


PRIOR = load_module("iq36_affine_q4_group128_prior", PRIOR_TOOL)
CERT = PRIOR.CERT

EXPECTED_SHA256 = {
    PRIOR_TOOL:
        "b9b84e791a1e1b438cdb162007442340249b30bee9a975479eb92f9eb9d76675",
    PRIOR_REJECTION:
        "22eee5dbcf92c3c635b6591b1c07bbf650626942573e827e6dc633dda28c8b35",
    PRIOR_REJECTION_MANIFEST:
        "95f40d575a9d5ce9409b4f54e890b0f9337ab9511c790e7dc1bc874bd04f0bee",
    **PRIOR.EXPECTED_SHA256,
}


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


def group_norm_f16(values: Any, np: Any) -> Any:
  rows = values.shape[0]
  grouped = values.reshape(rows * GROUPS, GROUP_SIZE)
  norms = CERT.upward_f32_norm(grouped, np).reshape(rows, GROUPS)
  return CERT.f16_upper(norms, np).astype(np.float64)


def affine_q4_group128(raw: Any, np: Any) -> Any:
  grouped = raw.reshape(raw.shape[0], GROUPS, GROUP_SIZE)
  minimum = np.min(grouped, axis=2)
  maximum = np.max(grouped, axis=2)
  step = (maximum - minimum) / np.float32(LEVELS - 1)
  safe_step = np.maximum(step, np.float32(2.0**-14))
  codes = np.clip(
      np.rint(
          (grouped - minimum[:, :, None]) / safe_step[:, :, None]),
      0, LEVELS - 1)
  return (
      minimum[:, :, None] + codes * step[:, :, None]
  ).astype(np.float32).reshape(raw.shape)


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
    raise SystemExit("group128 certificate requires a clean pushed commit")
  missing = [str(path) for path in EXPECTED_SHA256 if not path.is_file()]
  if missing:
    raise SystemExit(
        "missing group128-certificate inputs: " + ", ".join(missing))
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
        "registered group128-certificate hash mismatch: " +
        json.dumps(hash_mismatches, sort_keys=True))

  prior_rejection = load_json(PRIOR_REJECTION)
  prior_certificate = load_json(PRIOR.PRIOR_CERTIFICATE)
  population = load_json(PRIOR.POPULATION_METRICS)
  events_payload = load_json(PRIOR.EVENTS)
  events = events_payload.get("events") or []
  if len(events) != EVENT_COUNT:
    raise SystemExit(f"expected {EVENT_COUNT} population events")
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
      PRIOR.HIDDEN_MATRIX, dtype="<f2", mode="r",
      shape=(EVENT_COUNT, COLUMNS))
  hidden = np.asarray(hidden_memmap, dtype=np.float32)
  if not np.array_equal(
      hidden, hidden.astype(np.float16).astype(np.float32)):
    raise SystemExit("population hidden matrix is not exactly F16-valued")
  quantized_hidden = CERT.q8_hidden(hidden, np)
  hidden_delta = hidden - quantized_hidden
  hidden_group_norm = group_norm_f16(hidden, np)
  hidden_delta_group_norm = group_norm_f16(hidden_delta, np)
  hidden_t = np.ascontiguousarray(hidden.T, dtype=np.float32)
  quantized_hidden_t = np.ascontiguousarray(
      quantized_hidden.T, dtype=np.float32)

  weights = np.memmap(
      PRIOR.MODEL_BIN, dtype=np.int8, mode="r",
      offset=weight_fact["offset"], shape=(ROWS, COLUMNS))
  scales_memmap = np.memmap(
      PRIOR.MODEL_BIN, dtype="<f2", mode="r",
      offset=scale_fact["offset"], shape=(ROWS,))
  scales = np.asarray(scales_memmap, dtype=np.float32)

  seed_weights = np.asarray(weights[seed_ids], dtype=np.float64)
  seed_real = np.sum(
      seed_weights * hidden.astype(np.float64),
      axis=1, dtype=np.float64)
  seed_real *= scales[seed_ids].astype(np.float64)
  seed_values = f16_reference(seed_real, np)
  del seed_weights

  candidate_counts = np.zeros(EVENT_COUNT, dtype=np.int64)
  seed_included = np.zeros(EVENT_COUNT, dtype=np.bool_)
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
    codec = affine_q4_group128(raw, np)
    codec_group_norm = group_norm_f16(codec, np)
    residual_group_norm = group_norm_f16(raw - codec, np)
    approximate = (
        codec @ quantized_hidden_t *
        scales[begin:end, None]).astype(np.float32)
    residual_bound = np.abs(
        scales[begin:end, None]).astype(np.float64) * np.sum(
            residual_group_norm[:, None, :] *
                hidden_group_norm[None, :, :] +
            codec_group_norm[:, None, :] *
                hidden_delta_group_norm[None, :, :],
            axis=2)
    round_guard = np.sum(
        CERT.f32_dot_guard(
            codec_group_norm[:, None, :],
            hidden_group_norm[None, :, :],
            scales[begin:end, None, None], np),
        axis=2)
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

    selected = upper_output >= seed_values[None, :]
    candidate_counts += np.sum(selected, axis=0, dtype=np.int64)
    seed_events = np.flatnonzero(
        (seed_ids >= begin) & (seed_ids < end))
    seed_included[seed_events] = selected[
        seed_ids[seed_events] - begin, seed_events]
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
          "event": "affine_q4_group128_certificate_progress",
          "rows_complete": end,
          "rows_total": ROWS,
      }, sort_keys=True), flush=True)
      last_progress = now

  token_matches = reference_best_ids == seed_ids
  code_bytes = ROWS * COLUMNS // 2
  group_minmax_bytes = ROWS * GROUPS * 2
  source_scale_bytes = ROWS * 2
  packed_bytes = code_bytes + group_minmax_bytes + source_scale_bytes
  residual_norm_bytes = ROWS * GROUPS * 2
  materialized_output_bytes = ROWS * 2
  fixed_active_bytes = (
      packed_bytes + residual_norm_bytes + materialized_output_bytes)
  per_candidate_bytes = CERT.ROW_EXACT_BYTES + 4 + 2
  fixed_candidate_capacity = math.floor(
      (
          ACTIVE_BYTE_RATIO_CAP * CERT.FULL_I8_SCAN_BYTES -
          fixed_active_bytes
      ) / per_candidate_bytes)
  traffic_totals = (
      fixed_active_bytes + candidate_counts * per_candidate_bytes)
  traffic_ratios = (
      traffic_totals.astype(np.float64) /
      float(CERT.FULL_I8_SCAN_BYTES))
  worst_event = int(np.argmax(traffic_totals))
  maximum_active_bytes = int(traffic_totals[worst_event])
  maximum_ratio = float(traffic_ratios[worst_event])

  component = load_json(PRIOR.COMPONENT_RESULT)
  opportunity = load_json(PRIOR.OPPORTUNITY_RESULT)
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

  prior_failed_checks = [
      row.get("name") for row in prior_rejection.get("checks", [])
      if row.get("pass") is not True
  ]
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
      check("global_lloyd_q4_is_closed_only_on_capacity_and_traffic",
            prior_rejection.get("required_checks_passed") is False and
            prior_rejection.get("verdict") ==
                "reject_lloyd_q4_population_certificate" and
            prior_failed_checks == [
                "fixed_candidate_capacity_has_no_population_overflow",
                "worst_active_traffic_clears_registered_0p60_cap",
            ],
            prior_failed_checks=prior_failed_checks),
      check("seq2280_and_seq2287_admit_this_offline_successor",
            prior_certificate.get("required_checks_passed") is True and
            population.get("required_checks_passed") is True and
            population.get("selected_population", {}).get("count") ==
                EVENT_COUNT),
      check("locked_ir_and_population_geometry_match",
            hidden.shape == (EVENT_COUNT, COLUMNS) and
            weight_fact["shape"] == [ROWS, COLUMNS] and
            scale_fact["shape"] == [ROWS, 1]),
      check("captured_hidden_is_exactly_f16_valued",
            bool(np.array_equal(
                hidden, hidden.astype(np.float16).astype(np.float32)))),
      check("groupwise_bound_dominates_all_496m64_reference_scores",
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
      "admit_affine_q4_group128_certificate_component_implementation"
      if required else
      "reject_affine_q4_group128_population_certificate")
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
          "name": "affine_q4_group128_groupwise_global_l2",
          "group_size": GROUP_SIZE,
          "groups_per_row": GROUPS,
          "levels": LEVELS,
          "code_bytes": code_bytes,
          "group_minmax_format": "two_exact_int8_values_per_group",
          "group_minmax_bytes": group_minmax_bytes,
          "source_scale_bytes": source_scale_bytes,
          "packed_bytes": packed_bytes,
          "residual_norm_format": "outward_rounded_f16_per_group",
          "residual_norm_bytes": residual_norm_bytes,
          "codec_norm_storage_bytes": 0,
          "codec_norm_derivation": "from_already_read_q4_codes",
          "materialized_output_bytes": materialized_output_bytes,
          "fixed_active_bytes": fixed_active_bytes,
          "per_exact_candidate_bytes": per_candidate_bytes,
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
              "bounded_gpu_affine_q4_group128_component_implementation"
              if required else "close_affine_q4_group128_route"),
          "requirements": [
              "implement one parameterized GPU component",
              "derive codec norms while reading Q4 codes",
              "preserve outward F16 residual norms and exact fallback",
              "measure actual component latency and bytes",
              "retain full I8 fallback on capacity overflow",
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
  report = f"""# Affine-Q4/group128 LM-head population certificate

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
`{required_saving_ms:.6f}` ms kill-number. This is not yet a latency or
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
      "fixed_candidate_capacity": fixed_candidate_capacity,
      "maximum_active_byte_ratio": maximum_ratio,
      "peak_rss_bytes": result["memory"]["peak_rss_bytes"],
      "minimum_available_bytes": memory_min,
  }, separators=(",", ":")), flush=True)
  return 0 if required else 2


if __name__ == "__main__":
  raise SystemExit(main())
