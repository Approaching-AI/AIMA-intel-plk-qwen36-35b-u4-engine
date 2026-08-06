#!/usr/bin/env python3
"""Classify deeper nested producer linear input drift before preconv RMSNorm."""

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
    "input-gap-gate-v0"
)
ROUTE_PREFIX = (
    "router_prompt_full_attention_layer_input_preceding_linear_input_source_"
    "producer_linear_input_source_linear_input_source_layer_input_preceding_"
    "linear_input_source_producer_linear_input_source_linear_input_source_"
    "layer_input_preceding_linear_input_source_producer_linear_input_source_"
    "linear_input_source_layer_input_preceding_linear_input_source"
)
CURRENT_ROUTE = f"{ROUTE_PREFIX}_producer_linear_input_gap_gate"
NEXT_ROUTE = f"{ROUTE_PREFIX}_producer_linear_input_source_gap_gate"
MATH_ROUTE = f"{ROUTE_PREFIX}_producer_linear_input_preconv_math_gap_gate"

DISPOSITION_PREFIX = (
    "source_layer_input_preceding_linear_input_source_producer_linear_input_"
    "source_linear_input_source_layer_input_preceding_linear_input_source_"
    "producer_linear_input_source_linear_input_source_layer_input_preceding_"
    "linear_input_source"
)
DIAG_PREFIX = f"{DISPOSITION_PREFIX}_producer"

DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_SEQ401 = (
    ROOT
    / "output/router-full-attention-deeper-nested-preceding-linear-input-source-ffn-input-gap-gate-20260709Tseq401Z"
    / "metrics.json"
)
DEFAULT_SEQ402 = (
    ROOT
    / "output/router-full-attention-deeper-nested-producer-linear-delta-z-gap-gate-20260709Tseq402Z"
    / "metrics.json"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "output/router-full-attention-deeper-nested-producer-linear-input-gap-gate-20260709Tseq403Z"
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


def _run_summaries(seq401: dict[str, Any]) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  for run in seq401.get("runs", []):
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
  seq401 = _load_json(args.seq401)
  seq402 = _load_json(args.seq402)
  rows = _run_summaries(seq401)
  preconditions_pass = (
      seq402.get("required_checks_passed") is True
      and seq402.get("selected_next_route") == CURRENT_ROUTE
      and seq402.get("diagnostic_classification")
      == f"{DIAG_PREFIX}_linear_layer_input_attn_norm_drift"
      and _has_candidate(
          routes, 402,
          f"accept_{DISPOSITION_PREFIX}_producer_linear_delta_z_gap_classification")
      and _has_switch(routes, f"select_{CURRENT_ROUTE}", 402)
  )
  rows_available = len(rows) == 2

  min_layer_input = _min_nested(
      rows, "residual_source", "min_layer_input_cosine")
  min_attn_norm = _min_nested(
      rows, "linear_attention", "min_attn_norm_cosine")
  min_attn_norm_from_input = _min_nested(
      rows, "preconv_source", "min_attn_norm_from_gpu_input_cosine")
  min_gpu_attn_norm_vs_cpu = _min_nested(
      rows, "preconv_source", "min_gpu_attn_norm_vs_cpu_cosine")
  min_qkv_from_gpu_norm = _min_nested(
      rows, "preconv_source", "min_qkv_from_gpu_attn_norm_cosine")
  min_qkv_mixed = _min_nested(
      rows, "linear_attention", "min_qkv_mixed_cosine")
  min_gpu_qkv_vs_cpu = _min_nested(
      rows, "preconv_source", "min_gpu_qkv_vs_cpu_cosine")
  min_z_from_gpu_norm = _min_nested(
      rows, "preconv_source", "min_z_from_gpu_attn_norm_cosine")
  min_z = _min_nested(rows, "linear_attention", "min_z_cosine")
  min_gpu_z_vs_cpu = _min_nested(
      rows, "preconv_source", "min_gpu_z_vs_cpu_cosine")

  input_gap_reproduced = min_layer_input < COSINE_THRESHOLD
  attn_norm_inherits_input = (
      min_attn_norm < COSINE_THRESHOLD
      and min_attn_norm_from_input < COSINE_THRESHOLD)
  preconv_math_ok = (
      min_gpu_attn_norm_vs_cpu >= COSINE_THRESHOLD
      and min_gpu_qkv_vs_cpu >= COSINE_THRESHOLD
      and min_gpu_z_vs_cpu >= COSINE_THRESHOLD)
  qkv_z_inherit_input = (
      min_qkv_from_gpu_norm < COSINE_THRESHOLD
      and min_qkv_mixed < COSINE_THRESHOLD
      and min_z_from_gpu_norm < COSINE_THRESHOLD
      and min_z < COSINE_THRESHOLD)
  diagnostic_classification = (
      f"{DIAG_PREFIX}_linear_input_value_gap"
      if (input_gap_reproduced and attn_norm_inherits_input
          and preconv_math_ok and qkv_z_inherit_input) else
      f"{DIAG_PREFIX}_linear_input_preconv_math_gap"
      if not preconv_math_ok else
      f"{DIAG_PREFIX}_linear_input_gap_unclassified"
  )
  selected_next = (
      NEXT_ROUTE
      if diagnostic_classification == f"{DIAG_PREFIX}_linear_input_value_gap"
      else MATH_ROUTE
      if diagnostic_classification
      == f"{DIAG_PREFIX}_linear_input_preconv_math_gap"
      else CURRENT_ROUTE
  )
  checks = [
      {"name": "seq402_selected_deeper_nested_producer_linear_input_gap_gate",
       "pass": preconditions_pass},
      {"name": "seq401_run_summaries_available", "pass": rows_available},
      {"name": "producer_linear_input_gap_reproduced",
       "pass": input_gap_reproduced,
       "detail": {"min_layer_input_cosine": min_layer_input}},
      {"name": "attn_norm_inherits_input_drift",
       "pass": attn_norm_inherits_input,
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
      {"name": "qkv_and_z_inherit_input_drift",
       "pass": qkv_z_inherit_input,
       "detail": {
           "min_qkv_from_gpu_attn_norm_cosine": min_qkv_from_gpu_norm,
           "min_qkv_mixed_cosine": min_qkv_mixed,
           "min_z_from_gpu_attn_norm_cosine": min_z_from_gpu_norm,
           "min_z_cosine": min_z,
       }},
  ]
  required = all(bool(row.get("pass")) for row in checks)
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "inputs": {
          "routes": _rel(args.routes),
          "seq401": _rel(args.seq401),
          "seq402": _rel(args.seq402),
      },
      "checks": checks,
      "required_checks_passed": required,
      "diagnostic_classification": diagnostic_classification,
      "input_gap_reproduced": input_gap_reproduced,
      "attn_norm_inherits_input": attn_norm_inherits_input,
      "preconv_math_ok": preconv_math_ok,
      "qkv_z_inherit_input": qkv_z_inherit_input,
      "min_layer_input_cosine": min_layer_input,
      "min_attn_norm_cosine": min_attn_norm,
      "min_attn_norm_from_gpu_input_cosine": min_attn_norm_from_input,
      "min_gpu_attn_norm_vs_cpu_cosine": min_gpu_attn_norm_vs_cpu,
      "min_qkv_from_gpu_attn_norm_cosine": min_qkv_from_gpu_norm,
      "min_qkv_mixed_cosine": min_qkv_mixed,
      "min_gpu_qkv_vs_cpu_cosine": min_gpu_qkv_vs_cpu,
      "min_z_from_gpu_attn_norm_cosine": min_z_from_gpu_norm,
      "min_z_cosine": min_z,
      "min_gpu_z_vs_cpu_cosine": min_gpu_z_vs_cpu,
      "speedup_claims_allowed": False,
      "disposition": (
          f"accept_{DISPOSITION_PREFIX}_producer_linear_input_gap_classification"
          if required else
          f"block_{DISPOSITION_PREFIX}_producer_linear_input_gap_classification"
      ),
      "selected_next_route": selected_next if required else CURRENT_ROUTE,
      "next_route_reason": (
          "The deeper nested producer linear layer inputs already drift from "
          "native, and downstream RMSNorm/preconv math matches CPU on those "
          "live inputs. Root the value source feeding producer linear inputs next."
          if required and diagnostic_classification
          == f"{DIAG_PREFIX}_linear_input_value_gap"
          else
          "Deeper nested producer linear preconv math does not match CPU on "
          "live input; root preconv math next."
          if required and diagnostic_classification
          == f"{DIAG_PREFIX}_linear_input_preconv_math_gap"
          else
          "Deeper nested producer linear input evidence is incomplete; keep "
          "this gate open."
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
      "# Router Full-Attention Deeper Nested Producer Linear Input-Gap Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- diagnostic_classification: `{metrics['diagnostic_classification']}`",
      f"- min layer-input cosine: `{metrics['min_layer_input_cosine']}`",
      f"- min attn-norm cosine: `{metrics['min_attn_norm_cosine']}`",
      f"- min qkv/z cosines: `{metrics['min_qkv_mixed_cosine']}` / `{metrics['min_z_cosine']}`",
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
  parser.add_argument("--seq401", type=Path, default=DEFAULT_SEQ401)
  parser.add_argument("--seq402", type=Path, default=DEFAULT_SEQ402)
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
