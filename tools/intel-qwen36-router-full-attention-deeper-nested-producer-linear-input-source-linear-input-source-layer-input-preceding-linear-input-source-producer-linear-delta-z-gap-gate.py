#!/usr/bin/env python3
"""Split deeper nested producer linear delta/z drift to live input."""

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
    "input-source-producer-linear-delta-z-gap-gate-v0"
)
ROUTE_PREFIX = (
    "router_prompt_full_attention_layer_input_preceding_linear_input_source_"
    "producer_linear_input_source_linear_input_source_layer_input_preceding_"
    "linear_input_source_producer_linear_input_source_linear_input_source_"
    "layer_input_preceding_linear_input_source_producer_linear_input_source_"
    "linear_input_source_layer_input_preceding_linear_input_source"
)
CURRENT_ROUTE = (
    f"{ROUTE_PREFIX}_producer_linear_input_source_linear_input_source_layer_input_preceding_linear_input_source_producer_linear_delta_z_gap_gate"
)
NEXT_ROUTE = (
    f"{ROUTE_PREFIX}_producer_linear_input_source_linear_input_source_layer_input_preceding_linear_input_source_producer_linear_input_gap_gate"
)
MATH_ROUTE = (
    f"{ROUTE_PREFIX}_producer_linear_input_source_linear_input_source_layer_input_preceding_linear_input_source_producer_linear_preconv_math_gap_gate"
)

DISPOSITION_PREFIX = (
    "source_layer_input_preceding_linear_input_source_producer_linear_input_"
    "source_linear_input_source_layer_input_preceding_linear_input_source_"
    "producer_linear_input_source_linear_input_source_layer_input_preceding_"
    "linear_input_source"
)
DIAG_PREFIX = f"{DISPOSITION_PREFIX}_producer"

DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_SEQ416 = (
    ROOT
    / "output/router-full-attention-deeper-nested-producer-linear-input-source-linear-input-source-layer-input-preceding-linear-input-source-ffn-input-gap-gate-20260709Tseq416Z"
    / "metrics.json"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "output/router-full-attention-deeper-nested-producer-linear-input-source-linear-input-source-layer-input-preceding-linear-input-source-producer-linear-delta-z-gap-gate-20260709Tseq417Z"
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


def _run_summaries(seq416: dict[str, Any]) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  for run in seq416.get("runs", []):
    if not isinstance(run, dict):
      continue
    summary = run.get("summary")
    if isinstance(summary, dict):
      rows.append(summary)
  return rows


def _min_nested(rows: list[dict[str, Any]], group: str, metric: str) -> float:
  vals = []
  for row in rows:
    group_row = row.get("producer_ffn_input", {}).get(group, {})
    vals.append(_num(group_row.get(metric), 1.0))
  return min(vals, default=1.0)


def compute(args: argparse.Namespace) -> dict[str, Any]:
  routes = _load_json(args.routes)
  seq416 = _load_json(args.seq416)
  rows = _run_summaries(seq416)
  preconditions_pass = (
      seq416.get("required_checks_passed") is True
      and seq416.get("selected_next_route") == CURRENT_ROUTE
      and seq416.get("diagnostic_classification")
      == f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_preceding_linear_input_source_producer_linear_attention_delta_z_input_gap"
      and _has_candidate(
          routes, 416,
          f"accept_{DISPOSITION_PREFIX}_producer_linear_input_source_linear_input_source_layer_input_preceding_linear_input_source_ffn_input_gap_classification")
      and _has_switch(routes, f"select_{CURRENT_ROUTE}", 416)
  )
  rows_available = len(rows) == 2

  min_final = _num(seq416.get("min_linear_final_output_cosine"), 1.0)
  min_delta_output = _num(seq416.get("min_delta_output_cosine"), 1.0)
  min_z = _num(seq416.get("min_z_cosine"), 1.0)
  min_native_recompute = _num(
      seq416.get("min_native_delta_native_z_cpu_cosine"), 1.0)
  min_attn_norm = _min_nested(
      rows, "linear_attention", "min_attn_norm_cosine")
  min_attn_norm_from_input = _min_nested(
      rows, "preconv_source", "min_attn_norm_from_gpu_input_cosine")
  min_gpu_attn_norm_vs_cpu = _min_nested(
      rows, "preconv_source", "min_gpu_attn_norm_vs_cpu_cosine")
  min_qkv_from_gpu_norm = _min_nested(
      rows, "preconv_source", "min_qkv_from_gpu_attn_norm_cosine")
  min_gpu_qkv_vs_cpu = _min_nested(
      rows, "preconv_source", "min_gpu_qkv_vs_cpu_cosine")
  min_z_from_gpu_norm = _min_nested(
      rows, "preconv_source", "min_z_from_gpu_attn_norm_cosine")
  min_gpu_z_vs_cpu = _min_nested(
      rows, "preconv_source", "min_gpu_z_vs_cpu_cosine")

  final_gap_reproduced = min_final < COSINE_THRESHOLD
  delta_z_gap_reproduced = (
      min_delta_output < COSINE_THRESHOLD and min_z < COSINE_THRESHOLD)
  native_recompute_ok = min_native_recompute >= COSINE_THRESHOLD
  live_attn_norm_drift = (
      min_attn_norm < COSINE_THRESHOLD
      and min_attn_norm_from_input < COSINE_THRESHOLD)
  preconv_math_ok = (
      min_gpu_attn_norm_vs_cpu >= COSINE_THRESHOLD
      and min_gpu_qkv_vs_cpu >= COSINE_THRESHOLD
      and min_gpu_z_vs_cpu >= COSINE_THRESHOLD)
  qkv_z_inherit_attn_norm_drift = (
      min_qkv_from_gpu_norm < COSINE_THRESHOLD
      and min_z_from_gpu_norm < COSINE_THRESHOLD)
  diagnostic_classification = (
      f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_preceding_linear_input_source_producer_linear_layer_input_attn_norm_drift"
      if (final_gap_reproduced and delta_z_gap_reproduced
          and native_recompute_ok and live_attn_norm_drift
          and preconv_math_ok and qkv_z_inherit_attn_norm_drift) else
      f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_preceding_linear_input_source_producer_linear_preconv_math_gap"
      if not preconv_math_ok else
      f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_preceding_linear_input_source_producer_linear_delta_z_gap_unclassified"
  )
  selected_next = (
      NEXT_ROUTE
      if diagnostic_classification
      == f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_preceding_linear_input_source_producer_linear_layer_input_attn_norm_drift"
      else MATH_ROUTE
      if diagnostic_classification
      == f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_preceding_linear_input_source_producer_linear_preconv_math_gap"
      else CURRENT_ROUTE
  )
  checks = [
      {"name": "seq416_selected_deeper_nested_producer_linear_delta_z_gate",
       "pass": preconditions_pass},
      {"name": "seq416_run_summaries_available", "pass": rows_available},
      {"name": "producer_linear_final_gap_reproduced",
       "pass": final_gap_reproduced,
       "detail": {"min_final_output_cosine": min_final}},
      {"name": "producer_linear_delta_and_z_gaps_reproduced",
       "pass": delta_z_gap_reproduced,
       "detail": {
           "min_delta_output_cosine": min_delta_output,
           "min_z_cosine": min_z,
       }},
      {"name": "native_linear_final_recompute_matches_native",
       "pass": native_recompute_ok,
       "detail": {
           "min_native_delta_native_z_cpu_cosine": min_native_recompute,
       }},
      {"name": "live_attn_norm_drift_observed",
       "pass": live_attn_norm_drift,
       "detail": {
           "min_attn_norm_cosine": min_attn_norm,
           "min_attn_norm_from_gpu_input_cosine": min_attn_norm_from_input,
       }},
      {"name": "preconv_math_matches_cpu_on_live_input",
       "pass": preconv_math_ok,
       "detail": {
           "min_gpu_attn_norm_vs_cpu_cosine": min_gpu_attn_norm_vs_cpu,
           "min_gpu_qkv_vs_cpu_cosine": min_gpu_qkv_vs_cpu,
           "min_gpu_z_vs_cpu_cosine": min_gpu_z_vs_cpu,
       }},
      {"name": "qkv_and_z_inherit_attn_norm_drift",
       "pass": qkv_z_inherit_attn_norm_drift,
       "detail": {
           "min_qkv_from_gpu_attn_norm_cosine": min_qkv_from_gpu_norm,
           "min_z_from_gpu_attn_norm_cosine": min_z_from_gpu_norm,
       }},
  ]
  required = all(bool(row.get("pass")) for row in checks)
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "inputs": {
          "routes": _rel(args.routes),
          "seq416": _rel(args.seq416),
      },
      "checks": checks,
      "required_checks_passed": required,
      "diagnostic_classification": diagnostic_classification,
      "final_gap_reproduced": final_gap_reproduced,
      "delta_z_gap_reproduced": delta_z_gap_reproduced,
      "native_recompute_ok": native_recompute_ok,
      "live_attn_norm_drift": live_attn_norm_drift,
      "preconv_math_ok": preconv_math_ok,
      "qkv_z_inherit_attn_norm_drift": qkv_z_inherit_attn_norm_drift,
      "min_final_output_cosine": min_final,
      "min_delta_output_cosine": min_delta_output,
      "min_z_cosine": min_z,
      "min_native_delta_native_z_cpu_cosine": min_native_recompute,
      "min_attn_norm_cosine": min_attn_norm,
      "min_attn_norm_from_gpu_input_cosine": min_attn_norm_from_input,
      "min_gpu_attn_norm_vs_cpu_cosine": min_gpu_attn_norm_vs_cpu,
      "min_gpu_qkv_vs_cpu_cosine": min_gpu_qkv_vs_cpu,
      "min_gpu_z_vs_cpu_cosine": min_gpu_z_vs_cpu,
      "min_qkv_from_gpu_attn_norm_cosine": min_qkv_from_gpu_norm,
      "min_z_from_gpu_attn_norm_cosine": min_z_from_gpu_norm,
      "speedup_claims_allowed": False,
      "disposition": (
          f"accept_{DISPOSITION_PREFIX}_producer_linear_input_source_linear_input_source_layer_input_preceding_linear_input_source_producer_linear_delta_z_gap_classification"
          if required else
          f"block_{DISPOSITION_PREFIX}_producer_linear_input_source_linear_input_source_layer_input_preceding_linear_input_source_producer_linear_delta_z_gap_classification"
      ),
      "selected_next_route": selected_next if required else CURRENT_ROUTE,
      "next_route_reason": (
          "The deeper nested producer linear delta-output and z gaps are "
          "inherited from live linear layer-input / attention-norm drift; "
          "preconv math matches CPU on that live input. Root the producer "
          "linear input next."
          if required and diagnostic_classification
          == f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_preceding_linear_input_source_producer_linear_layer_input_attn_norm_drift"
          else
          "Deeper nested producer linear preconv math does not match CPU on "
          "live input; root preconv math next."
          if required and diagnostic_classification
          == f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_preceding_linear_input_source_producer_linear_preconv_math_gap"
          else
          "Deeper nested producer linear delta/z split evidence is incomplete; "
          "keep this gate open."
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
      "# Router Full-Attention Deeper Nested Producer Linear Delta/Z Gap Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- diagnostic_classification: `{metrics['diagnostic_classification']}`",
      f"- min final-output cosine: `{metrics['min_final_output_cosine']}`",
      f"- min delta/z cosine: `{metrics['min_delta_output_cosine']}` / `{metrics['min_z_cosine']}`",
      f"- min attn-norm cosine: `{metrics['min_attn_norm_cosine']}`",
      f"- min GPU preconv math cosines: `{metrics['min_gpu_attn_norm_vs_cpu_cosine']}` / `{metrics['min_gpu_qkv_vs_cpu_cosine']}` / `{metrics['min_gpu_z_vs_cpu_cosine']}`",
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
  parser.add_argument("--seq416", type=Path, default=DEFAULT_SEQ416)
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
