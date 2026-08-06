#!/usr/bin/env python3
"""Classify last-layer input-delta source attribution rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = "intel-qwen36-seq529-last-layer-input-delta-source-v0"

DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_SEQ528 = (
    ROOT
    / "output/seq528-last-layer-ffn-input-sensitivity-source-gate-20260709Tseq528Z"
    / "metrics.json"
)
DEFAULT_MATH = (
    ROOT
    / "output/seq529-last-layer-input-delta-source-math-20260709Tseq529Z"
    / "result.json"
)
DEFAULT_CODE = (
    ROOT
    / "output/seq529-last-layer-input-delta-source-code-20260709Tseq529Z"
    / "result.json"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "output/seq529-last-layer-input-delta-source-gate-20260709Tseq529Z"
)

CURRENT_ROUTE = "router_prompt_distribution_last_layer_input_delta_source_gate"
NEXT_ROUTE = "router_prompt_distribution_layer38_ffn_input_source_gate"
REJECTED_ROUTE = "router_prompt_distribution_layer38_ffn_delta_dominant_source"
KLD_THRESHOLD = 0.005
TOP1_THRESHOLD = 0.99
LAST_LAYER = 39
SOURCE_LAYER = 38
MIN_FFN_INPUT_SHARE = 0.60
MAX_CLOSURE_ERROR = 1.0e-6

REQUIRED_SOURCE_FIELDS = {
    "index",
    "layer_input_available",
    "attention_output_available",
    "ffn_input_available",
    "ffn_delta_available",
    "final_residual_component",
    "layer_input_component",
    "attention_output_component",
    "ffn_input_component",
    "ffn_delta_component",
    "attention_residual_closure_error",
    "final_residual_closure_error",
}


def _load_json(path: Path) -> Any:
  return json.loads(path.read_text(encoding="utf-8"))


def _rel(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path)


def _num(value: Any, default: float = 0.0) -> float:
  return float(value) if isinstance(value, (int, float)) else default


def _has_candidate(routes: dict[str, Any], seq: int, disposition: str) -> bool:
  return any(
      isinstance(row, dict)
      and row.get("seq") == seq
      and row.get("disposition") == disposition
      for row in routes.get("candidate_history", []))


def _has_switch(routes: dict[str, Any], decision: str, seq_covered: int) -> bool:
  return any(
      isinstance(row, dict)
      and row.get("decision") == decision
      and _num(row.get("seq_covered")) >= seq_covered
      and row.get("resolved") is True
      for row in routes.get("switch_decisions", []))


def _dist(smoke: dict[str, Any]) -> dict[str, Any]:
  dist = smoke.get("distribution_ladder")
  return dist if isinstance(dist, dict) else {}


def _projection(step: dict[str, Any]) -> dict[str, Any]:
  projections = step.get("head_pair_projections")
  if not isinstance(projections, list) or not projections:
    return {}
  projection = projections[0]
  return projection if isinstance(projection, dict) else {}


def _component_sums(source_dims: list[dict[str, Any]]) -> dict[str, float]:
  def sum_abs(name: str) -> float:
    return sum(abs(_num(row.get(name))) for row in source_dims)

  ffn_input = sum_abs("ffn_input_component")
  ffn_delta = sum_abs("ffn_delta_component")
  layer_input = sum_abs("layer_input_component")
  attention_output = sum_abs("attention_output_component")
  return {
      "layer39_input_abs_sum": sum_abs("final_residual_component"),
      "layer38_layer_input_abs_sum": layer_input,
      "layer38_attention_output_abs_sum": attention_output,
      "layer38_ffn_input_abs_sum": ffn_input,
      "layer38_ffn_delta_abs_sum": ffn_delta,
      "layer38_ffn_input_direct_share": (
          ffn_input / (ffn_input + ffn_delta)
          if (ffn_input + ffn_delta) > 0.0 else 0.0),
      "layer38_ffn_delta_direct_share": (
          ffn_delta / (ffn_input + ffn_delta)
          if (ffn_input + ffn_delta) > 0.0 else 0.0),
      "layer38_layer_input_share_of_ffn_input_split": (
          layer_input / (layer_input + attention_output)
          if (layer_input + attention_output) > 0.0 else 0.0),
      "layer38_attention_output_share_of_ffn_input_split": (
          attention_output / (layer_input + attention_output)
          if (layer_input + attention_output) > 0.0 else 0.0),
  }


def _failed_step(step: dict[str, Any]) -> dict[str, Any]:
  projection = _projection(step)
  source_dims = projection.get("last_layer_input_source_dims")
  source_dims = [row for row in source_dims if isinstance(row, dict)] if (
      isinstance(source_dims, list)) else []
  top_dims = projection.get("top_pair_projection_dims")
  top_dims = top_dims if isinstance(top_dims, list) else []
  closures = [
      abs(_num(row.get("attention_residual_closure_error")))
      for row in source_dims
  ] + [
      abs(_num(row.get("final_residual_closure_error")))
      for row in source_dims
  ]
  fields_present = (
      bool(source_dims)
      and all(REQUIRED_SOURCE_FIELDS.issubset(row.keys())
              for row in source_dims))
  all_available = (
      bool(source_dims)
      and all(row.get("layer_input_available") is True
              and row.get("attention_output_available") is True
              and row.get("ffn_input_available") is True
              and row.get("ffn_delta_available") is True
              for row in source_dims))
  return {
      "token_index": step.get("token_index"),
      "token_position": step.get("token_position"),
      "kld": step.get("kld"),
      "positive_token_id": projection.get("positive_token_id"),
      "negative_token_id": projection.get("negative_token_id"),
      "last_layer": LAST_LAYER,
      "source_layer": projection.get("last_layer_input_source_layer"),
      "source_dim_count": len(source_dims),
      "top_pair_projection_dim_count": len(top_dims),
      "source_fields_present": fields_present,
      "source_components_available": all_available,
      "max_source_closure_error": max(closures) if closures else None,
      "component_sums": _component_sums(source_dims),
      "top_source_dims": source_dims[:8],
  }


def _row(path: Path, label: str) -> dict[str, Any]:
  payload = _load_json(path)
  smoke = payload.get("smoke")
  smoke = smoke if isinstance(smoke, dict) else payload
  dist = _dist(smoke)
  steps = dist.get("steps")
  steps = steps if isinstance(steps, list) else []
  failed_steps = [
      _failed_step(step) for step in steps
      if isinstance(step, dict) and _num(step.get("kld")) > KLD_THRESHOLD
  ]
  return {
      "label": label,
      "path": _rel(path),
      "case_id": smoke.get("case_id"),
      "target_returncode": payload.get("target", {}).get(
          "run", {}).get("returncode"),
      "required_checks_passed": smoke.get("required_checks_passed"),
      "distribution_required_checks_passed": dist.get("required_checks_passed"),
      "max_kld": dist.get("max_kld"),
      "mean_kld": dist.get("mean_kld"),
      "top1_rate": dist.get("top1_rate"),
      "top1_pass": dist.get("top1_pass"),
      "kld_pass": dist.get("kld_pass"),
      "full_attention_state_diff_enabled": smoke.get(
          "full_attention_state_diff_enabled"),
      "diagnostic_layer_range": payload.get("diagnostic_layer_range"),
      "step_count": len(steps),
      "failed_step_count": len(failed_steps),
      "failed_steps": failed_steps,
  }


def _row_ran(row: dict[str, Any]) -> bool:
  return (
      row.get("target_returncode") == 2
      and row.get("distribution_required_checks_passed") is False
      and row.get("kld_pass") is False
      and row.get("top1_pass") is True
      and _num(row.get("top1_rate")) >= TOP1_THRESHOLD
      and _num(row.get("max_kld")) > KLD_THRESHOLD
      and row.get("full_attention_state_diff_enabled") is True
      and row.get("diagnostic_layer_range") == "38:40")


def _source_available(row: dict[str, Any]) -> bool:
  failed = row.get("failed_steps")
  if not isinstance(failed, list) or not failed:
    return False
  for step in failed:
    if step.get("source_layer") != SOURCE_LAYER:
      return False
    if step.get("source_fields_present") is not True:
      return False
    if step.get("source_components_available") is not True:
      return False
    if step.get("source_dim_count") != step.get("top_pair_projection_dim_count"):
      return False
    if _num(step.get("max_source_closure_error")) > MAX_CLOSURE_ERROR:
      return False
  return True


def _ffn_input_dominates(row: dict[str, Any]) -> bool:
  failed = row.get("failed_steps")
  if not isinstance(failed, list) or not failed:
    return False
  for step in failed:
    sums = step.get("component_sums")
    sums = sums if isinstance(sums, dict) else {}
    if _num(sums.get("layer38_ffn_input_direct_share")) < MIN_FFN_INPUT_SHARE:
      return False
  return True


def compute(args: argparse.Namespace) -> dict[str, Any]:
  routes = _load_json(args.routes)
  seq528 = _load_json(args.seq528)
  math_row = _row(args.math, "seq529_last_layer_input_delta_source_math")
  code_row = _row(args.code, "seq529_last_layer_input_delta_source_code")

  checks = [
      {
          "name": "seq528_selected_last_layer_input_delta_route",
          "pass": (
              seq528.get("required_checks_passed") is True
              and seq528.get("selected_next_route") == CURRENT_ROUTE
              and _has_candidate(
                  routes, 528,
                  "reject_last_layer_attention_output_dominant_source_select_last_layer_input_delta_source")
              and _has_switch(
                  routes,
                  "select_router_prompt_distribution_last_layer_input_delta_source_gate",
                  528)),
      },
      {
          "name": "last_layer_input_delta_source_rows_target_ran",
          "pass": _row_ran(math_row) and _row_ran(code_row),
          "detail": {"math": math_row, "code": code_row},
      },
      {
          "name": "last_layer_input_source_dimensions_close",
          "pass": _source_available(math_row) and _source_available(code_row),
          "detail": {
              "last_layer": LAST_LAYER,
              "source_layer": SOURCE_LAYER,
              "max_source_closure_error": MAX_CLOSURE_ERROR,
              "math_failed_steps": math_row.get("failed_steps"),
              "code_failed_steps": code_row.get("failed_steps"),
          },
      },
      {
          "name": "layer38_ffn_input_dominates_direct_ffn_delta",
          "pass": _ffn_input_dominates(math_row)
          and _ffn_input_dominates(code_row),
          "detail": {
              "min_ffn_input_direct_share": MIN_FFN_INPUT_SHARE,
              "math_failed_steps": math_row.get("failed_steps"),
              "code_failed_steps": code_row.get("failed_steps"),
          },
      },
  ]
  required = all(bool(row.get("pass")) for row in checks)
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "inputs": {
          "routes": _rel(args.routes),
          "seq528": _rel(args.seq528),
          "math": _rel(args.math),
          "code": _rel(args.code),
      },
      "checks": checks,
      "required_checks_passed": required,
      "speedup_claims_allowed": False,
      "decode_probe_allowed": False,
      "router_distribution_allowed": False,
      "layer38_ffn_delta_dominant_source_allowed": False,
      "layer38_ffn_input_source_gate_allowed": required,
      "rows": {"math": math_row, "code": code_row},
      "disposition": (
          "reject_layer38_ffn_delta_dominant_source_select_layer38_ffn_input_source"
          if required else
          "block_last_layer_input_delta_source_inconsistent_evidence"),
      "rejected_route": REJECTED_ROUTE if required else None,
      "selected_next_route": NEXT_ROUTE if required else CURRENT_ROUTE,
      "next_route_reason": (
          "The layer-39 input delta is the layer-38 output delta. The traced "
          "dimensions close through the layer-38 output equation, and the "
          "layer-38 FFN-input component exceeds the direct FFN-delta component "
          "on every failed math/code step. Direct layer-38 FFN-delta dominance "
          "is closed; the next unit should attribute the layer-38 FFN-input "
          "source before any product correction, speed promotion, or "
          "long-context row."
          if required else
          "Layer-39 input-delta source evidence is inconsistent; do not switch "
          "routes or run speed/promotion/long-context rows."),
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
  rows = metrics["rows"]
  lines = [
      "# Seq529 Last-Layer Input-Delta Source Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- rejected_route: `{metrics['rejected_route']}`",
      f"- failed_checks: `{failed}`",
      f"- speedup_claims_allowed: `{str(metrics['speedup_claims_allowed']).lower()}`",
      "",
      "## Evidence",
      "",
      f"- math KLD/top1/failed steps: `{rows['math']['max_kld']}` / `{rows['math']['top1_rate']}` / `{rows['math']['failed_step_count']}`",
      f"- code KLD/top1/failed steps: `{rows['code']['max_kld']}` / `{rows['code']['top1_rate']}` / `{rows['code']['failed_step_count']}`",
      "",
      metrics["next_route_reason"],
      "",
      "This is correctness-route evidence only. It is not a speed claim.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--routes", type=Path, default=DEFAULT_ROUTES)
  parser.add_argument("--seq528", type=Path, default=DEFAULT_SEQ528)
  parser.add_argument("--math", type=Path, default=DEFAULT_MATH)
  parser.add_argument("--code", type=Path, default=DEFAULT_CODE)
  parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
  return parser.parse_args()


def main() -> int:
  args = parse_args()
  metrics = compute(args)
  write_outputs(metrics, args.out_dir)
  print(json.dumps({
      "required_checks_passed": metrics["required_checks_passed"],
      "disposition": metrics["disposition"],
      "selected_next_route": metrics["selected_next_route"],
      "out_dir": _rel(args.out_dir),
  }, sort_keys=True))
  return 0 if metrics["required_checks_passed"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
