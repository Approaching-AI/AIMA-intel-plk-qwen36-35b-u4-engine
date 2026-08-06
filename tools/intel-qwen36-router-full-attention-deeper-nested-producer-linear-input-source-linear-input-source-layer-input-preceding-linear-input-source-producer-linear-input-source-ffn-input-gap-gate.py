#!/usr/bin/env python3
"""Classify source FFN input drift feeding deeper nested producer linear input sources."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = (
    "intel-qwen36-router-full-attention-deeper-nested-producer-linear-"
    "input-source-linear-input-source-layer-input-preceding-linear-"
    "input-source-producer-linear-input-source-ffn-input-gap-gate-v0"
)
ROUTE_PREFIX = (
    "router_prompt_full_attention_layer_input_preceding_linear_input_source_"
    "producer_linear_input_source_linear_input_source_layer_input_preceding_"
    "linear_input_source_producer_linear_input_source_linear_input_source_"
    "layer_input_preceding_linear_input_source_producer_linear_input_source_"
    "linear_input_source_layer_input_preceding_linear_input_source"
)
CURRENT_ROUTE = (
    f"{ROUTE_PREFIX}_producer_linear_input_source_linear_input_source_layer_input_preceding_linear_input_source_producer_linear_input_source_ffn_input_gap_gate"
)
NEXT_ROUTE = (
    f"{ROUTE_PREFIX}_producer_linear_input_source_linear_input_source_layer_input_preceding_linear_input_source_producer_linear_input_source_attention_output_gap_gate"
)
MATH_ROUTE = (
    f"{ROUTE_PREFIX}_producer_linear_input_source_linear_input_source_layer_input_preceding_linear_input_source_producer_linear_input_source_ffn_norm_math_gap_gate"
)

DISPOSITION_PREFIX = (
    "source_layer_input_preceding_linear_input_source_producer_linear_input_"
    "source_linear_input_source_layer_input_preceding_linear_input_source_"
    "producer_linear_input_source_linear_input_source_layer_input_preceding_"
    "linear_input_source"
)
DIAG_PREFIX = (
    f"{DISPOSITION_PREFIX}_producer_linear_input_source_linear_input_source_"
    "layer_input_preceding_linear_input_source_producer"
)

DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_SEQ419 = (
    ROOT
    / "output/router-full-attention-deeper-nested-producer-linear-input-source-linear-input-source-layer-input-preceding-linear-input-source-producer-linear-input-source-gap-gate-20260709Tseq419Z"
    / "metrics.json"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "output/router-full-attention-deeper-nested-producer-linear-input-source-linear-input-source-layer-input-preceding-linear-input-source-producer-linear-input-source-ffn-input-gap-gate-20260709Tseq420Z"
)
COSINE_THRESHOLD = 0.9999


def _load_json(path: Path) -> Any:
  return json.loads(path.read_text(encoding="utf-8"))


def _num(value: Any, default: float = 0.0) -> float:
  return float(value) if isinstance(value, (int, float)) else default


def _rel(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path)


def _has_candidate(routes: dict[str, Any], seq: int, disposition: str) -> bool:
  return any(
      isinstance(row, dict)
      and row.get("seq") == seq
      and row.get("disposition") == disposition
      for row in routes.get("candidate_history", [])
  )


def _has_switch(routes: dict[str, Any], decision: str, seq_covered: int) -> bool:
  return any(
      isinstance(row, dict)
      and row.get("decision") == decision
      and _num(row.get("seq_covered")) >= seq_covered
      and row.get("resolved") is True
      for row in routes.get("switch_decisions", [])
  )


def _run_summaries(seq419: dict[str, Any]) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  for run in seq419.get("runs", []):
    if not isinstance(run, dict):
      continue
    summary = run.get("summary")
    if isinstance(summary, dict):
      rows.append(summary)
  return rows


def _min_source(rows: list[dict[str, Any]], metric: str) -> float:
  vals = []
  for row in rows:
    source = row.get("producer_linear_input_source", {}).get("source_ffn", {})
    vals.append(_num(source.get(metric), 1.0))
  return min(vals, default=1.0)


def _same_metric(lhs: float, rhs: float) -> bool:
  return abs(lhs - rhs) <= 1e-12


def compute(args: argparse.Namespace) -> dict[str, Any]:
  routes = _load_json(args.routes)
  seq419 = _load_json(args.seq419)
  rows = _run_summaries(seq419)
  preconditions_pass = (
      seq419.get("required_checks_passed") is True
      and seq419.get("selected_next_route") == CURRENT_ROUTE
      and seq419.get("diagnostic_classification")
      == f"{DIAG_PREFIX}_linear_input_source_ffn_input_gap"
      and _has_candidate(
          routes, 419,
          f"accept_{DISPOSITION_PREFIX}_producer_linear_input_source_linear_input_source_layer_input_preceding_linear_input_source_producer_linear_input_source_gap_classification")
      and _has_switch(routes, f"select_{CURRENT_ROUTE}", 419)
  )
  rows_available = len(rows) == 2
  min_ffn_input = _num(seq419.get("min_source_ffn_input_cosine"), 1.0)
  min_attention_output = _num(
      seq419.get("min_source_attention_output_cosine"), 1.0)
  min_cpu_ffn_norm_from_gpu_input = _num(
      seq419.get("min_source_cpu_ffn_norm_from_gpu_input_cosine"), 1.0)
  min_gpu_ffn_norm_vs_cpu = _num(
      seq419.get("min_source_gpu_ffn_norm_vs_cpu_cosine"), 1.0)
  min_cpu_ffn_from_gpu_input = _num(
      seq419.get("min_source_cpu_ffn_from_gpu_input_cosine"), 1.0)
  min_gpu_output_vs_cpu_ffn = _num(
      seq419.get("min_source_gpu_output_vs_cpu_ffn_cosine"), 1.0)
  min_cpu_delta_from_gpu_input = _num(
      seq419.get("min_source_cpu_ffn_delta_from_gpu_input_cosine"), 1.0)
  min_gpu_delta_vs_cpu = _num(
      seq419.get("min_source_gpu_ffn_delta_vs_cpu_cosine"), 1.0)
  min_source_layer_input = _num(
      seq419.get("min_source_layer_input_cosine"), 1.0)

  nested_attention = _min_source(rows, "min_attention_output_cosine")
  nested_ffn_input = _min_source(rows, "min_ffn_input_cosine")
  nested_gpu_norm = _min_source(rows, "min_gpu_ffn_norm_vs_cpu_cosine")

  ffn_input_gap = min_ffn_input < COSINE_THRESHOLD
  attention_output_gap = min_attention_output < COSINE_THRESHOLD
  ffn_norm_inherits_input = (
      min_cpu_ffn_norm_from_gpu_input < COSINE_THRESHOLD
      and min_cpu_ffn_from_gpu_input < COSINE_THRESHOLD
      and min_cpu_delta_from_gpu_input < COSINE_THRESHOLD)
  ffn_math_ok = (
      min_gpu_ffn_norm_vs_cpu >= COSINE_THRESHOLD
      and min_gpu_output_vs_cpu_ffn >= COSINE_THRESHOLD
      and min_gpu_delta_vs_cpu >= COSINE_THRESHOLD)
  nested_consistent = (
      _same_metric(nested_attention, min_attention_output)
      and _same_metric(nested_ffn_input, min_ffn_input)
      and _same_metric(nested_gpu_norm, min_gpu_ffn_norm_vs_cpu))
  diagnostic_classification = (
      f"{DIAG_PREFIX}_linear_input_source_ffn_input_attention_output_gap"
      if (ffn_input_gap and attention_output_gap and ffn_norm_inherits_input
          and ffn_math_ok) else
      f"{DIAG_PREFIX}_linear_input_source_ffn_input_norm_math_gap"
      if ffn_input_gap and not ffn_math_ok else
      f"{DIAG_PREFIX}_linear_input_source_ffn_input_gap_unclassified"
  )
  selected_next = (
      NEXT_ROUTE
      if diagnostic_classification
      == f"{DIAG_PREFIX}_linear_input_source_ffn_input_attention_output_gap"
      else MATH_ROUTE
      if diagnostic_classification
      == f"{DIAG_PREFIX}_linear_input_source_ffn_input_norm_math_gap"
      else CURRENT_ROUTE
  )
  checks = [
      {"name": "seq419_selected_deeper_nested_source_ffn_input_gap_gate",
       "pass": preconditions_pass},
      {"name": "seq419_run_summaries_available", "pass": rows_available},
      {"name": "seq419_nested_metrics_match_top_level",
       "pass": nested_consistent,
       "detail": {
           "nested_attention_output_cosine": nested_attention,
           "nested_ffn_input_cosine": nested_ffn_input,
           "nested_gpu_ffn_norm_vs_cpu_cosine": nested_gpu_norm,
       }},
      {"name": "source_ffn_input_gap_reproduced",
       "pass": ffn_input_gap,
       "detail": {"min_source_ffn_input_cosine": min_ffn_input}},
      {"name": "source_attention_output_gap_observed",
       "pass": attention_output_gap,
       "detail": {
           "min_source_attention_output_cosine": min_attention_output,
           "min_source_layer_input_cosine": min_source_layer_input,
       }},
      {"name": "source_ffn_norm_inherits_live_input",
       "pass": ffn_norm_inherits_input,
       "detail": {
           "min_source_cpu_ffn_norm_from_gpu_input_cosine": (
               min_cpu_ffn_norm_from_gpu_input),
           "min_source_cpu_ffn_from_gpu_input_cosine": (
               min_cpu_ffn_from_gpu_input),
           "min_source_cpu_ffn_delta_from_gpu_input_cosine": (
               min_cpu_delta_from_gpu_input),
       }},
      {"name": "source_ffn_math_matches_cpu_on_live_input",
       "pass": ffn_math_ok,
       "detail": {
           "min_source_gpu_ffn_norm_vs_cpu_cosine": (
               min_gpu_ffn_norm_vs_cpu),
           "min_source_gpu_output_vs_cpu_ffn_cosine": (
               min_gpu_output_vs_cpu_ffn),
           "min_source_gpu_ffn_delta_vs_cpu_cosine": (
               min_gpu_delta_vs_cpu),
       }},
  ]
  required = all(bool(row.get("pass")) for row in checks)
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "inputs": {
          "routes": _rel(args.routes),
          "seq419": _rel(args.seq419),
      },
      "checks": checks,
      "required_checks_passed": required,
      "diagnostic_classification": diagnostic_classification,
      "ffn_input_gap": ffn_input_gap,
      "attention_output_gap": attention_output_gap,
      "ffn_norm_inherits_input": ffn_norm_inherits_input,
      "ffn_math_ok": ffn_math_ok,
      "min_source_ffn_input_cosine": min_ffn_input,
      "min_source_attention_output_cosine": min_attention_output,
      "min_source_layer_input_cosine": min_source_layer_input,
      "min_source_cpu_ffn_norm_from_gpu_input_cosine": (
          min_cpu_ffn_norm_from_gpu_input),
      "min_source_gpu_ffn_norm_vs_cpu_cosine": min_gpu_ffn_norm_vs_cpu,
      "min_source_cpu_ffn_from_gpu_input_cosine": min_cpu_ffn_from_gpu_input,
      "min_source_gpu_output_vs_cpu_ffn_cosine": min_gpu_output_vs_cpu_ffn,
      "min_source_cpu_ffn_delta_from_gpu_input_cosine": (
          min_cpu_delta_from_gpu_input),
      "min_source_gpu_ffn_delta_vs_cpu_cosine": min_gpu_delta_vs_cpu,
      "speedup_claims_allowed": False,
      "disposition": (
          f"accept_{DISPOSITION_PREFIX}_producer_linear_input_source_linear_input_source_layer_input_preceding_linear_input_source_producer_linear_input_source_ffn_input_gap_classification"
          if required else
          f"block_{DISPOSITION_PREFIX}_producer_linear_input_source_linear_input_source_layer_input_preceding_linear_input_source_producer_linear_input_source_ffn_input_gap_classification"
      ),
      "selected_next_route": selected_next if required else CURRENT_ROUTE,
      "next_route_reason": (
          "The deeper nested source FFN input drift is inherited from source "
          "attention-output drift while FFN norm/output math matches CPU on "
          "live input. Root the source attention output next."
          if required and diagnostic_classification
          == f"{DIAG_PREFIX}_linear_input_source_ffn_input_attention_output_gap"
          else
          "Deeper nested source FFN math does not match CPU on live input; "
          "root FFN norm/math next."
          if required and diagnostic_classification
          == f"{DIAG_PREFIX}_linear_input_source_ffn_input_norm_math_gap"
          else
          "Deeper nested source FFN input evidence is incomplete; keep this gate open."
      ),
  }


def write_outputs(metrics: dict[str, Any], out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n",
      encoding="utf-8")
  failed = [
      row["name"] for row in metrics["checks"]
      if row.get("pass") is not True
  ]
  lines = [
      "# Router Full-Attention Deeper Nested Producer Linear Input-Source FFN Input-Gap Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- diagnostic_classification: `{metrics['diagnostic_classification']}`",
      f"- min source FFN input cosine: `{metrics['min_source_ffn_input_cosine']}`",
      f"- min source attention-output cosine: `{metrics['min_source_attention_output_cosine']}`",
      f"- min source CPU FFN-norm-from-live-input cosine: `{metrics['min_source_cpu_ffn_norm_from_gpu_input_cosine']}`",
      f"- min source GPU norm/output/delta vs CPU: `{metrics['min_source_gpu_ffn_norm_vs_cpu_cosine']}` / `{metrics['min_source_gpu_output_vs_cpu_ffn_cosine']}` / `{metrics['min_source_gpu_ffn_delta_vs_cpu_cosine']}`",
      f"- failed_checks: `{failed}`",
      f"- speedup_claims_allowed: `{str(metrics['speedup_claims_allowed']).lower()}`",
      "",
      metrics["next_route_reason"],
      "",
      "This is distribution/correctness evidence only. It is not a speed claim.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--routes", type=Path, default=DEFAULT_ROUTES)
  parser.add_argument("--seq419", type=Path, default=DEFAULT_SEQ419)
  parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
  return parser.parse_args()


def main() -> int:
  args = parse_args()
  metrics = compute(args)
  write_outputs(metrics, args.out_dir)
  print(json.dumps({
      "diagnostic_classification": metrics["diagnostic_classification"],
      "disposition": metrics["disposition"],
      "out_dir": _rel(args.out_dir),
      "required_checks_passed": metrics["required_checks_passed"],
      "selected_next_route": metrics["selected_next_route"],
  }, sort_keys=True))
  return 0 if metrics["required_checks_passed"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
