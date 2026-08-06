#!/usr/bin/env python3
"""Bound the deterministic count25 LM-head exact-fallback slow mode.

This is an evidence-only gate.  It consumes the two 2k prefill-shape
candidate timing rows from seq2185 plus the accepted PTL dense-traffic ceiling.
It launches no model, compiler, or GPU worker.  The result decides whether a
stage-separated component profile is arithmetically justified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRODUCT = ROOT / (
    "output/openvino-short-nonsentinel-auto-abba1-"
    "20260731Tseq2185-clean")
DEFAULT_TRAFFIC = ROOT / (
    "output/openvino-exact-attention-two-workgroup-traffic-"
    "20260724Tseq2151-clean/result.json")
SOURCE_PATCH = ROOT / "engine/openvino/iq36-lm-head-i8q1-gated-exact.patch"
CASE_ID = "prefill_shape_002k"
WORKERS = ("candidate-b1", "candidate-b2")
SKIP_INTERVALS = 16
JITTER_LIMIT = 1.25
EXPECTED_ROWS = 248_320
EXPECTED_COLUMNS = 2_048
EXPECTED_COUNT = 25
EXPECTED_DELTA = 11.0
EXPECTED_WORKGROUPS = 384
SCHEMA = "intel-qwen36-openvino-lm-head-gated-exact-fallback-bound-v1"


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--product", type=Path, default=DEFAULT_PRODUCT)
  parser.add_argument("--traffic", type=Path, default=DEFAULT_TRAFFIC)
  parser.add_argument("--output", type=Path, required=True)
  return parser.parse_args()


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
    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def relative(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path.resolve())


def git_state(output: Path) -> dict[str, Any]:
  commit = subprocess.run(
      ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
      capture_output=True, text=True).stdout.strip()
  rows = subprocess.run(
      ["git", "status", "--porcelain"], cwd=ROOT, check=True,
      capture_output=True, text=True).stdout.splitlines()
  try:
    output_relative = str(output.resolve().relative_to(ROOT))
  except ValueError:
    output_relative = ""
  rows = [
      row for row in rows
      if not output_relative or output_relative not in row]
  return {"commit": commit, "dirty": bool(rows), "status": rows}


def percentile(values: list[float], probability: float) -> float:
  if not values:
    raise ValueError("empty percentile input")
  ordered = sorted(values)
  index = min(
      len(ordered) - 1, int(round(probability * (len(ordered) - 1))))
  return ordered[index]


def split_modes(values: list[float]) -> dict[str, Any]:
  """Split the two stable modes without a hand-chosen latency threshold.

  Samples above twice the median are treated as unrelated cold/outlier rows.
  Within that broad bound, the largest adjacent order-statistic gap separates
  the normal and fallback modes.
  """
  p50 = percentile(values, 0.50)
  bounded = sorted(value for value in values if value <= 2.0 * p50)
  if len(bounded) < 2:
    raise ValueError("not enough bounded timing samples")
  gaps = [
      upper - lower for lower, upper in zip(bounded, bounded[1:])]
  split_index = max(range(len(gaps)), key=gaps.__getitem__)
  threshold = (bounded[split_index] + bounded[split_index + 1]) * 0.5
  fast = [value for value in values if value < threshold]
  slow = [
      value for value in values
      if threshold <= value <= 2.0 * p50]
  outliers = [
      {"index": index, "ms": value}
      for index, value in enumerate(values) if value > 2.0 * p50]
  slow_indices = [
      index for index, value in enumerate(values)
      if threshold <= value <= 2.0 * p50]
  return {
      "fast_count": len(fast),
      "fast_max_ms": max(fast),
      "fast_median_ms": statistics.median(fast),
      "largest_gap_ms": gaps[split_index],
      "outliers": outliers,
      "p50_ms": p50,
      "p95_ms": percentile(values, 0.95),
      "slow_count": len(slow),
      "slow_indices_after_skip": slow_indices,
      "slow_median_ms": statistics.median(slow),
      "slow_min_ms": min(slow),
      "split_threshold_ms": threshold,
  }


def check(name: str, passed: bool, **details: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **details}


def source_contract(text: str) -> dict[str, Any]:
  needles = {
      "count25": "#define IQ36_BINARY_GATED_EXACT_COUNT 25U",
      "delta11": "#define IQ36_BINARY_GATED_EXACT_DELTA 11.0f",
      "workgroups384":
          "constexpr size_t kBinaryGatedExactWorkgroups = 384;",
      "matvec": "__kernel void iq36_lm_head_i8_gated_exact_matvec_f16(",
      "block_topk":
          "__kernel void iq36_lm_head_i8q1_gated_exact_output_topk8_f16(",
      "merge":
          "__kernel void iq36_lm_head_i8q1_gated_exact_topk8_merge_f32(",
      "correction":
          "__kernel void iq36_lm_head_i8_gated_exact_topk8_correction_f16(",
  }
  return {
      "all_present": all(text.count(value) == 1 for value in needles.values()),
      "occurrences": {
          key: text.count(value) for key, value in needles.items()},
  }


def main() -> int:
  args = parse_args()
  product = args.product.resolve()
  traffic_path = args.traffic.resolve()
  output = args.output.resolve()
  if output.exists():
    raise SystemExit(f"output already exists: {output}")
  required = [
      product / "gate.json", product / "smoothness.json",
      traffic_path, SOURCE_PATCH,
  ]
  worker_paths = [
      product / "raw" / CASE_ID / "block00" / worker / "worker-result.json"
      for worker in WORKERS]
  required.extend(worker_paths)
  missing = [str(path) for path in required if not path.is_file()]
  if missing:
    raise SystemExit("missing bound inputs: " + ", ".join(missing))

  state = git_state(output)
  product_gate = load_json(product / "gate.json")
  smoothness = load_json(product / "smoothness.json")
  traffic = load_json(traffic_path)
  source_text = SOURCE_PATCH.read_text(encoding="utf-8")
  source = source_contract(source_text)
  workers = [load_json(path) for path in worker_paths]

  rows = []
  for label, worker in zip(WORKERS, workers):
    intervals = [
        float(value) for value in worker.get("decode_wall_ms", [])
    ][SKIP_INTERVALS:]
    mode = split_modes(intervals)
    p50 = float(mode["p50_ms"])
    p95 = float(mode["p95_ms"])
    allowed_p95 = JITTER_LIMIT * p50
    required_saving = max(0.0, p95 - allowed_p95)
    slow_increment = (
        float(mode["slow_median_ms"]) - float(mode["fast_median_ms"]))
    rows.append({
        "allowed_p95_ms": allowed_p95,
        "generated_token_ids_sha256":
            worker.get("generated_token_ids_sha256"),
        "interval_count_after_skip": len(intervals),
        "jitter_ratio": p95 / p50,
        "label": label,
        "mode": mode,
        "required_saving_ms": required_saving,
        "slow_increment_ms": slow_increment,
    })

  trace_rows = (
      workers[0].get("lm_head_i8q1_trace", {}).get(
          "weight_prepack_rows", []))
  trace = trace_rows[0] if trace_rows else {}
  weight_bytes = int(trace.get("source_weight_bytes", 0))
  scale_bytes = int(trace.get("source_scale_bytes", 0))
  rows_count = int(trace.get("rows", 0))
  columns_count = int(trace.get("columns", 0))
  output_bytes = rows_count * 2
  mandatory_matvec_bytes = weight_bytes + scale_bytes + output_bytes
  bandwidth_lcb = float(traffic.get("bandwidth_lcb_gb_s", math.nan))
  traffic_floor_ms = (
      mandatory_matvec_bytes / (bandwidth_lcb * 1e9) * 1e3
      if bandwidth_lcb > 0 else math.inf)
  minimum_slow_increment = min(
      float(row["slow_increment_ms"]) for row in rows)
  nontraffic_headroom_ms = minimum_slow_increment - traffic_floor_ms
  required_saving_ms = max(
      float(row["required_saving_ms"]) for row in rows)
  headroom_multiple = (
      nontraffic_headroom_ms / required_saving_ms
      if required_saving_ms > 0 else math.inf)
  slow_index_sets_exact = (
      rows[0]["mode"]["slow_indices_after_skip"] ==
      rows[1]["mode"]["slow_indices_after_skip"])

  jitter_rows = [
      row for row in smoothness.get("jitter_rows", [])
      if row.get("case_id") == CASE_ID]
  checks = [
      check("repository_clean_at_bound", not state["dirty"], git=state),
      check("seq2185_is_the_expected_single_failure",
            product_gate.get("route_label") == "rejected" and
            product_gate.get("run_checks_passed") is False and
            smoothness.get("required_checks_passed") is False and
            len([
                row for row in smoothness.get("checks", [])
                if row.get("pass") is False]) == 1 and
            len(jitter_rows) == 2 and
            all(row.get("pass") is False for row in jitter_rows)),
      check("source_contract_is_count25_delta11_six_stage",
            source["all_present"], source=source,
            count=EXPECTED_COUNT, delta=EXPECTED_DELTA,
            workgroups=EXPECTED_WORKGROUPS),
      check("locked_lm_head_trace_matches_source_contract",
            rows_count == EXPECTED_ROWS and
            columns_count == EXPECTED_COLUMNS and
            weight_bytes == EXPECTED_ROWS * EXPECTED_COLUMNS and
            scale_bytes == EXPECTED_ROWS * 2,
            rows=rows_count, columns=columns_count,
            weight_bytes=weight_bytes, scale_bytes=scale_bytes),
      check("two_candidate_rows_have_identical_fallback_indices",
            slow_index_sets_exact and
            len(rows[0]["mode"]["slow_indices_after_skip"]) == 50,
            slow_count=len(rows[0]["mode"]["slow_indices_after_skip"])),
      check("traffic_ceiling_is_accepted",
            traffic.get("required_checks_passed") is True and
            traffic.get("traffic_capacity_pass") is True and
            math.isfinite(bandwidth_lcb) and bandwidth_lcb > 0,
            bandwidth_lcb_gb_s=bandwidth_lcb),
      check("nontraffic_headroom_can_fund_required_jitter_cut",
            nontraffic_headroom_ms >= required_saving_ms > 0,
            nontraffic_headroom_ms=nontraffic_headroom_ms,
            required_saving_ms=required_saving_ms,
            headroom_multiple=headroom_multiple),
  ]
  passed = all(row["pass"] for row in checks)
  verdict = (
      "admit_stage_separated_gated_exact_component_profile"
      if passed else
      "close_or_repair_gated_exact_fallback_bound")

  result = {
      "schema": SCHEMA,
      "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
      "workstream": "intel-qwen36-35b-a3b-gguf-q4km",
      "git": state,
      "verdict": verdict,
      "required_checks_passed": passed,
      "gpu_workers_launched": 0,
      "model_workers_launched": 0,
      "case_id": CASE_ID,
      "skip_intervals": SKIP_INTERVALS,
      "jitter_limit": JITTER_LIMIT,
      "worker_rows": rows,
      "fallback_indices_exact_across_workers": slow_index_sets_exact,
      "required_saving_ms": required_saving_ms,
      "traffic_bound": {
          "bandwidth_lcb_gb_s": bandwidth_lcb,
          "mandatory_matvec_bytes": mandatory_matvec_bytes,
          "source_weight_bytes": weight_bytes,
          "source_scale_bytes": scale_bytes,
          "f16_output_bytes": output_bytes,
          "traffic_floor_ms": traffic_floor_ms,
          "minimum_observed_slow_increment_ms": minimum_slow_increment,
          "nontraffic_headroom_ms": nontraffic_headroom_ms,
          "headroom_multiple_over_required_cut": headroom_multiple,
      },
      "source_contract": source,
      "component_profile_contract": {
          "inputs": "captured real 2048-element LM-head hidden rows",
          "stages": [
              "full_i8_group256_q8_matvec",
              "block_top8",
              "global_top8_merge",
              "exact_f16_hidden_top8_correction",
          ],
          "raw_weight_bytes": weight_bytes,
          "minimum_profile_samples_per_hidden": 20,
          "required_output":
              "per-stage kernel timestamps, wall, bytes, achieved GB/s",
          "candidate_admission":
              "paired one-sided 95% saving LCB must exceed "
              f"{required_saving_ms * 1000.0:.6f} us and captured outputs "
              "must remain exact to the accepted fallback reference",
      },
      "inputs": {
          "product": relative(product),
          "product_gate_sha256": sha256(product / "gate.json"),
          "smoothness_sha256": sha256(product / "smoothness.json"),
          "traffic": relative(traffic_path),
          "traffic_sha256": sha256(traffic_path),
          "source_patch": relative(SOURCE_PATCH),
          "source_patch_sha256": sha256(SOURCE_PATCH),
          "workers": {
              label: {
                  "path": relative(path), "sha256": sha256(path)}
              for label, path in zip(WORKERS, worker_paths)},
      },
      "checks": checks,
      "product_promotion_ready": False,
      "speedup_claims_allowed": False,
  }

  output.mkdir(parents=True)
  write_json(output / "result.json", result)
  lines = [
      "# Gated-exact fallback bound",
      "",
      f"- verdict: `{verdict}`",
      f"- required checks passed: `{str(passed).lower()}`",
      "- identical slow-mode rows after the 16-interval exclusion: "
      f"`{len(rows[0]['mode']['slow_indices_after_skip'])}`",
      "- B1/B2 fallback increment: "
      f"`{rows[0]['slow_increment_ms']:.6f}/"
      f"{rows[1]['slow_increment_ms']:.6f} ms`",
      f"- required robust cut: `{required_saving_ms * 1000.0:.6f} us`",
      "- mandatory exact-matvec traffic / floor: "
      f"`{mandatory_matvec_bytes} B / {traffic_floor_ms:.6f} ms` at "
      f"`{bandwidth_lcb:.6f} GB/s`",
      "- conservative nontraffic headroom: "
      f"`{nontraffic_headroom_ms * 1000.0:.6f} us` "
      f"(`{headroom_multiple:.3f}x` the required cut)",
      "",
      "This admits only a stage-separated standalone component profile.  It "
      "does not admit a kernel edit, plugin build, model worker, formal ABBA8, "
      "or speedup claim.",
      "",
  ]
  (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
  print(json.dumps({
      "event": "gated_exact_fallback_bound_complete",
      "output": relative(output),
      "verdict": verdict,
      "required_checks_passed": passed,
      "required_saving_us": required_saving_ms * 1000.0,
      "nontraffic_headroom_us": nontraffic_headroom_ms * 1000.0,
  }, sort_keys=True))
  return 0 if passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
