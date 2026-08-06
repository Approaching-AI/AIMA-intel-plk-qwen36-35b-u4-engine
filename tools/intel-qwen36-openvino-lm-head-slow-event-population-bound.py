#!/usr/bin/env python3
"""Extract a high-confidence LM-head slow-event population from seq2286.

The seq2286 capture intentionally remains failed because its teacher-forced
corpus produced a higher slow-event rate and a smaller order-statistic gap than
the exploratory prior.  This evidence-only bound does not change that verdict.
Instead it uses the independently registered physical fallback increment:
an event must reproduce in both workers with a detrended residual between
0.75x and 1.25x that increment.  Exactly 2000 context-spanning rows are then
materialized for the offline Lloyd-Q4/global-L2 certificate.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-openvino-lm-head-slow-event-population-bound-v1"
CAPTURE_TOOL = ROOT / (
    "tools/intel-qwen36-openvino-lm-head-slow-event-capture.py")
CAPTURE_ROOT = ROOT / (
    "output/openvino-lm-head-slow-event-capture-"
    "20260801Tseq2286-teacher-clean")
CAPTURE_METRICS = CAPTURE_ROOT / "metrics.json"
CAPTURE_MANIFEST = CAPTURE_ROOT / "manifest.json"
CAPTURE_RESULT = CAPTURE_ROOT / "raw/capture/worker-result.json"
CONFIRM_RESULT = CAPTURE_ROOT / "raw/confirm/worker-result.json"
FULL_HIDDEN_MATRIX = CAPTURE_ROOT / "raw/capture/lm-head-inputs.f16"
FALLBACK_BOUND = ROOT / (
    "output/openvino-lm-head-gated-exact-fallback-bound-"
    "20260731Tseq2186-clean/result.json")
TEACHER_FORCED_CORPUS = ROOT / "engine/src/gpu_q4x8_matvec.cpp"

EXPECTED_SHA256 = {
    CAPTURE_TOOL:
        "54a2abd3b175cd589f2d8c6af3c7e76fb2e15d7c57bb59582821ac7784c67d8f",
    CAPTURE_METRICS:
        "eca1e2d4e3d6aef53f66df6a4de4eb86beafc7e67ac4215592df2936293071af",
    CAPTURE_MANIFEST:
        "ac34df48741899cd673fb91924c0332bd569f2a38aaf1689c2150b0c3148919d",
    CAPTURE_RESULT:
        "55fafed416d38d7b364e93a5b42901de742c645bb934523e671447a5cc2726d1",
    CONFIRM_RESULT:
        "20d819198f2e62e8b9441719ccf45e95fe151c44ee9a3f5158fd4eb01a3cd72a",
    FULL_HIDDEN_MATRIX:
        "edb8c714ee3beec77a5f5f5da512e0171bd37990d7a873f0a3ed1f260443dcc9",
    FALLBACK_BOUND:
        "6bed3a9f24917433d51559c62d3dec222abaa9654c16ad1443b96b98b0936be7",
    TEACHER_FORCED_CORPUS:
        "6427111ee1566da61b83fb729d9ec848d3f5a643c16e495fe9c9dd6428564eea",
}
OUTPUT_TOKENS = 32_768
HIDDEN_COLUMNS = 2_048
TARGET_EVENTS = 2_000


def load_module(name: str, path: Path) -> Any:
  spec = importlib.util.spec_from_file_location(name, path)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import {path}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


CAPTURE = load_module("iq36_lm_head_population_capture", CAPTURE_TOOL)
PRODUCT = CAPTURE.PRODUCT


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out-dir", type=Path, required=True)
  return parser.parse_args()


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    while chunk := stream.read(8 * 1024 * 1024):
      digest.update(chunk)
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

  upstream = git("rev-parse", "--verify", "@{upstream}")
  state.update({
      "branch": git("branch", "--show-current"),
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


def main() -> int:
  args = parse_args()
  out = args.out_dir.resolve()
  if out.exists():
    raise SystemExit(f"output already exists: {out}")
  out.mkdir(parents=True)

  missing = [str(path) for path in EXPECTED_SHA256 if not path.is_file()]
  if missing:
    raise SystemExit("missing population-bound inputs: " + ", ".join(missing))
  observed_hashes = {path: sha256(path) for path in EXPECTED_SHA256}
  hash_mismatches = {
      PRODUCT.relative(path): {
          "expected": expected,
          "observed": observed_hashes[path],
      }
      for path, expected in EXPECTED_SHA256.items()
      if observed_hashes[path] != expected
  }
  if hash_mismatches:
    raise SystemExit(
        "registered population-bound hash mismatch: " +
        json.dumps(hash_mismatches, sort_keys=True))

  git = git_state(out)
  if git["dirty"] or not git["pushed"]:
    raise SystemExit("population bound requires a clean pushed commit")

  capture_metrics = PRODUCT.load_json(CAPTURE_METRICS)
  fallback = PRODUCT.load_json(FALLBACK_BOUND)
  results = {
      "capture": PRODUCT.load_json(CAPTURE_RESULT),
      "confirm": PRODUCT.load_json(CONFIRM_RESULT),
  }
  source_failed_checks = [
      row.get("name") for row in capture_metrics.get("checks", [])
      if row.get("pass") is not True
  ]
  source_failure_is_distribution_prior_only = (
      capture_metrics.get("required_checks_passed") is False and
      capture_metrics.get("verdict") ==
          "repair_or_extend_slow_event_population_capture" and
      source_failed_checks == ["long_slow_modes_match_registered_shape"] and
      capture_metrics.get("selected_population", {}).get("count") ==
          TARGET_EVENTS)

  classifications: dict[str, dict[str, Any]] = {}
  residuals: dict[str, list[float]] = {}
  for label, result in results.items():
    summary, _, row_residuals = CAPTURE.detrended_slow_events(
        result["decode_wall_ms"])
    classifications[label] = summary
    residuals[label] = row_residuals

  residual_low_ms = CAPTURE.SLOW_MEDIAN_LOW_MS
  residual_high_ms = CAPTURE.SLOW_MEDIAN_HIGH_MS
  high_confidence_sets = {
      label: {
          index + CAPTURE.SKIP_INTERVALS + 1
          for index, residual in enumerate(row_residuals)
          if residual_low_ms <= residual <= residual_high_ms
      }
      for label, row_residuals in residuals.items()
  }
  repeated_steps = sorted(
      high_confidence_sets["capture"] & high_confidence_sets["confirm"])
  symmetric_difference = sorted(
      high_confidence_sets["capture"] ^ high_confidence_sets["confirm"])
  capture_recall = (
      len(repeated_steps) / len(high_confidence_sets["capture"]))
  confirm_recall = (
      len(repeated_steps) / len(high_confidence_sets["confirm"]))
  selected_positions = CAPTURE.stratified_positions(
      len(repeated_steps), TARGET_EVENTS)
  selected_steps = [
      repeated_steps[position] for position in selected_positions]

  selected_matrix_path = out / "slow-lm-head-inputs.f16"
  events_path = out / "slow-events.json"
  source_matrix = np.memmap(
      FULL_HIDDEN_MATRIX, dtype="<f2", mode="r",
      shape=(OUTPUT_TOKENS, HIDDEN_COLUMNS))
  selected_matrix = np.memmap(
      selected_matrix_path, dtype="<f2", mode="w+",
      shape=(TARGET_EVENTS, HIDDEN_COLUMNS))
  capture_walls = [
      float(value) for value in results["capture"]["decode_wall_ms"]]
  confirm_walls = [
      float(value) for value in results["confirm"]["decode_wall_ms"]]
  capture_tokens = [
      int(value) for value in results["capture"]["generated_token_ids"]]
  confirm_tokens = [
      int(value) for value in results["confirm"]["generated_token_ids"]]
  events = []
  for ordinal, step in enumerate(selected_steps):
    decode_index = step - 1
    after_skip_index = decode_index - CAPTURE.SKIP_INTERVALS
    selected_matrix[ordinal] = source_matrix[step]
    events.append({
        "ordinal": ordinal,
        "step": step,
        "decode_index": decode_index,
        "generated_token_id": capture_tokens[step],
        "capture_wall_ms": capture_walls[decode_index],
        "capture_residual_ms":
            residuals["capture"][after_skip_index],
        "confirm_wall_ms": confirm_walls[decode_index],
        "confirm_residual_ms":
            residuals["confirm"][after_skip_index],
    })
  selected_matrix.flush()
  PRODUCT.write_json(events_path, {
      "schema": SCHEMA,
      "selection": (
          "equal-index-stratification-over-two-worker-physical-band-"
          "intersection"),
      "registered_fallback_increment_ms": CAPTURE.EXPECTED_INCREMENT_MS,
      "registered_residual_band_ms": [residual_low_ms, residual_high_ms],
      "source_repeated_high_confidence_count": len(repeated_steps),
      "target_event_count": TARGET_EVENTS,
      "hidden_columns": HIDDEN_COLUMNS,
      "matrix_file": PRODUCT.relative(selected_matrix_path),
      "events": events,
  })

  selected_capture_residuals = [
      float(row["capture_residual_ms"]) for row in events]
  selected_confirm_residuals = [
      float(row["confirm_residual_ms"]) for row in events]
  max_symmetric_difference = max(
      128, math.ceil(0.02 * len(repeated_steps)))
  worker_runs = capture_metrics.get("runs") or {}
  source_workers_safe = (
      list(worker_runs) == ["capture", "confirm"] and
      all(
          run.get("returncode") == 0 and
          run.get("oom_observed") is False and
          run.get("timed_out") is False and
          (run.get("memory_guard") or {}).get("tripped") is False
          for run in worker_runs.values()))
  matrix_descriptor = (
      results["capture"].get("lm_head_hidden_matrix") or {})
  checks = [
      check("repository_clean_and_pushed_at_bound",
            not git["dirty"] and git["pushed"], git=git),
      check("registered_inputs_match_exact_hashes",
            not hash_mismatches,
            inputs={
                PRODUCT.relative(path): observed_hashes[path]
                for path in observed_hashes
            }),
      check("seq2286_remains_failed_only_on_exploratory_distribution_prior",
            source_failure_is_distribution_prior_only,
            failed_checks=source_failed_checks,
            source_verdict=capture_metrics.get("verdict")),
      check("registered_fallback_increment_is_unchanged",
            fallback.get("required_checks_passed") is True and
            math.isclose(
                float(fallback["traffic_bound"][
                    "minimum_observed_slow_increment_ms"]),
                CAPTURE.EXPECTED_INCREMENT_MS,
                rel_tol=0.0, abs_tol=1e-12),
            increment_ms=CAPTURE.EXPECTED_INCREMENT_MS,
            residual_band_ms=[residual_low_ms, residual_high_ms]),
      check("two_serial_source_workers_are_complete_without_oom",
            source_workers_safe,
            workers_concurrent=False),
      check("teacher_forced_inputs_and_candidate_outputs_repeat_exactly",
            results["capture"].get("generated_token_ids") ==
                results["confirm"].get("generated_token_ids") and
            results["capture"].get("generated_token_ids_sha256") ==
                results["confirm"].get("generated_token_ids_sha256") and
            results["capture"].get("teacher_forced_prompt") ==
                results["confirm"].get("teacher_forced_prompt") and
            (results["capture"].get("teacher_forced_prompt") or {}).get(
                "file_sha256") ==
                EXPECTED_SHA256[TEACHER_FORCED_CORPUS]),
      check("full_hidden_matrix_matches_worker_descriptor",
            matrix_descriptor.get("shape") ==
                [OUTPUT_TOKENS, HIDDEN_COLUMNS] and
            matrix_descriptor.get("dtype") == "float16-little-endian" and
            matrix_descriptor.get("sha256") ==
                EXPECTED_SHA256[FULL_HIDDEN_MATRIX] and
            FULL_HIDDEN_MATRIX.stat().st_size ==
                OUTPUT_TOKENS * HIDDEN_COLUMNS * 2,
            descriptor=matrix_descriptor),
      check("physical_band_population_is_large_and_reproducible",
            len(repeated_steps) >= TARGET_EVENTS and
            len(symmetric_difference) <= max_symmetric_difference and
            capture_recall >= 0.98 and confirm_recall >= 0.98,
            capture_count=len(high_confidence_sets["capture"]),
            confirm_count=len(high_confidence_sets["confirm"]),
            repeated_count=len(repeated_steps),
            capture_recall=capture_recall,
            confirm_recall=confirm_recall,
            symmetric_difference_count=len(symmetric_difference),
            symmetric_difference_limit=max_symmetric_difference),
      check("exactly_2000_context_spanning_rows_are_materialized",
            len(events) == len(selected_steps) ==
                len(set(selected_steps)) == TARGET_EVENTS and
            selected_matrix_path.stat().st_size ==
                TARGET_EVENTS * HIDDEN_COLUMNS * 2 and
            selected_steps[0] <= 1_024 and
            selected_steps[-1] >= OUTPUT_TOKENS - 1_024,
            first_step=selected_steps[0],
            last_step=selected_steps[-1],
            matrix_sha256=sha256(selected_matrix_path),
            events_sha256=sha256(events_path)),
      check("every_selected_event_is_in_the_registered_band_twice",
            all(
                residual_low_ms <= value <= residual_high_ms
                for value in (
                    selected_capture_residuals +
                    selected_confirm_residuals)),
            capture_residual_p50_ms=percentile(
                selected_capture_residuals, 0.50),
            capture_residual_p95_ms=percentile(
                selected_capture_residuals, 0.95),
            confirm_residual_p50_ms=percentile(
                selected_confirm_residuals, 0.50),
            confirm_residual_p95_ms=percentile(
                selected_confirm_residuals, 0.95)),
      check("selected_exact_fallback_tokens_repeat",
            all(
                capture_tokens[step] == confirm_tokens[step]
                for step in selected_steps)),
      check("bound_launches_no_model_compiler_or_gpu_worker",
            True, model_workers=0, product_workers=0,
            compiler_invocations=0, gpu_contexts=0),
  ]
  required = all(row["pass"] for row in checks)
  verdict = (
      "admit_2000_high_confidence_rows_for_offline_lloyd_q4_certificate"
      if required else
      "reject_or_repair_high_confidence_population_extraction")
  payload = {
      "schema": SCHEMA,
      "workstream": WS,
      "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
      "git": git,
      "verdict": verdict,
      "required_checks_passed": required,
      "source_capture_verdict_unchanged": True,
      "source_or_plugin_integration_admitted": False,
      "performance_claim_admitted": False,
      "registered_fallback_increment_ms": CAPTURE.EXPECTED_INCREMENT_MS,
      "registered_residual_band_ms": [residual_low_ms, residual_high_ms],
      "classifications": {
          label: {
              key: summary[key] for key in (
                  "decode_interval_count", "measured_interval_count",
                  "largest_gap_ms", "split_threshold_ms", "slow_count",
                  "slow_rate", "slow_residual_median_ms",
                  "slow_residual_p95_ms")
          }
          for label, summary in classifications.items()
      },
      "high_confidence_population": {
          "capture_count": len(high_confidence_sets["capture"]),
          "confirm_count": len(high_confidence_sets["confirm"]),
          "repeated_count": len(repeated_steps),
          "capture_recall": capture_recall,
          "confirm_recall": confirm_recall,
          "symmetric_difference_count": len(symmetric_difference),
          "steps": repeated_steps,
      },
      "selected_population": {
          "count": len(selected_steps),
          "steps": selected_steps,
          "matrix": {
              "path": PRODUCT.relative(selected_matrix_path),
              "dtype": "float16-little-endian",
              "shape": [TARGET_EVENTS, HIDDEN_COLUMNS],
              "byte_count": selected_matrix_path.stat().st_size,
              "sha256": sha256(selected_matrix_path),
          },
          "events": {
              "path": PRODUCT.relative(events_path),
              "sha256": sha256(events_path),
          },
      },
      "checks": checks,
      "workers": {
          "model_workers": 0,
          "product_workers": 0,
          "compiler_invocations": 0,
          "gpu_contexts": 0,
      },
      "next_gate": {
          "route": "offline_lloyd_q4_global_l2_population_certificate",
          "requirements": [
              "consume exactly this selected 2000-row matrix and event map",
              "require zero global-L2 bound violations",
              "require every repeated exact fallback token to be returned",
              "require no fixed candidate-capacity overflow",
              "require worst active traffic at or below 0.60x full I8",
          ],
      },
  }
  PRODUCT.write_json(out / "metrics.json", payload)
  PRODUCT.write_json(out / "manifest.json", {
      "schema": SCHEMA,
      "tool": PRODUCT.relative(Path(__file__)),
      "tool_sha256": sha256(Path(__file__)),
      "git": git,
      "inputs": {
          PRODUCT.relative(path): observed_hashes[path]
          for path in observed_hashes
      },
      "outputs": {
          PRODUCT.relative(path): {
              "bytes": path.stat().st_size,
              "sha256": sha256(path),
          }
          for path in (selected_matrix_path, events_path)
      },
      "workers": payload["workers"],
  })
  report = f"""# LM-head high-confidence slow-event population

Verdict: **{verdict}**. Required checks: `{str(required).lower()}`.

The seq2286 verdict remains failed on its exploratory distribution prior.
This separate zero-worker bound uses the already registered physical fallback
increment of `{CAPTURE.EXPECTED_INCREMENT_MS:.6f}` ms and retains only events
whose capture and confirm residuals both lie in
`[{residual_low_ms:.6f}, {residual_high_ms:.6f}]` ms.

The two source rows contain `{len(high_confidence_sets['capture']):,}` and
`{len(high_confidence_sets['confirm']):,}` high-confidence events;
`{len(repeated_steps):,}` repeat. Exactly `{len(selected_steps):,}` rows are
stratified across steps `{selected_steps[0]:,}` through
`{selected_steps[-1]:,}`. This admits only the offline Lloyd-Q4/global-L2
population certificate, not source/plugin integration or a speed claim.
"""
  (out / "report.md").write_text(report, encoding="utf-8")
  print(json.dumps({
      "artifact": PRODUCT.relative(out),
      "verdict": verdict,
      "required_checks_passed": required,
      "capture_high_confidence_count":
          len(high_confidence_sets["capture"]),
      "confirm_high_confidence_count":
          len(high_confidence_sets["confirm"]),
      "repeated_high_confidence_count": len(repeated_steps),
      "selected_count": len(selected_steps),
      "first_step": selected_steps[0],
      "last_step": selected_steps[-1],
      "matrix_sha256": sha256(selected_matrix_path),
  }, separators=(",", ":")), flush=True)
  return 0 if required else 2


if __name__ == "__main__":
  raise SystemExit(main())
