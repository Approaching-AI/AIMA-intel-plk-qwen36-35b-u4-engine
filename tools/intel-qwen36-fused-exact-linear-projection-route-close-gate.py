#!/usr/bin/env python3
"""Close fused exact projection and select one FFN attribution boundary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = (
    "intel-qwen36-fused-exact-linear-projection-route-close-gate-v0"
)
CURRENT_ROUTE = (
    "router_prompt_distribution_fused_exact_linear_projection_route_close_gate"
)
SELECTED_NEXT_ROUTE = (
    "router_prompt_distribution_layer12_ffn_component_attribution_gate"
)
REFLECTION_ROUTE = "router_prompt_distribution_correctness_route_reflection_gate"
LINEAR_LAYERS = [
    0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18,
    20, 21, 22, 24, 25, 26, 28, 29, 30, 32, 33, 34, 36, 37, 38,
]


def _load(path: Path) -> dict[str, Any]:
  payload = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(payload, dict):
    raise TypeError(f"{path} does not contain a JSON object")
  return payload


def _rel(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path)


def _has_candidate(routes: dict[str, Any], seq: int,
                   next_route: str) -> bool:
  return any(
      isinstance(row, dict)
      and row.get("seq") == seq
      and row.get("selected_next_route") == next_route
      for row in routes.get("candidate_history", []))


def _has_switch(routes: dict[str, Any], seq: int, decision: str) -> bool:
  return any(
      isinstance(row, dict)
      and row.get("seq_covered") == seq
      and row.get("decision") == decision
      and row.get("resolved") is True
      for row in routes.get("switch_decisions", []))


def _find_layer(step: dict[str, Any], layer: int) -> dict[str, Any]:
  for row in step.get("layers", []):
    if isinstance(row, dict) and row.get("layer") == layer:
      return row
  return {}


def _find_rejection(rejected: dict[str, Any], route: str) -> dict[str, Any]:
  for row in rejected.get("rejected", []):
    if isinstance(row, dict) and row.get("route") == route:
      return row
  return {}


def compute(args: argparse.Namespace) -> dict[str, Any]:
  routes = _load(args.routes)
  rejected = _load(args.rejected_routes)
  predecessor = _load(args.predecessor)
  smoke = _load(args.smoke)
  component = _load(args.component)

  distribution = smoke.get("distribution_ladder", {})
  distribution = distribution if isinstance(distribution, dict) else {}
  steps = distribution.get("steps", [])
  token0 = steps[0] if len(steps) == 8 and isinstance(steps[0], dict) else {}
  token7 = steps[7] if len(steps) == 8 and isinstance(steps[7], dict) else {}

  boundary_steps = smoke.get("layer_boundary_diff_by_step", [])
  linear_steps = smoke.get("linear_attention_diff_by_step", [])
  norm_steps = smoke.get("ffn_norm_diff_by_step", [])
  boundary0 = (
      boundary_steps[0]
      if boundary_steps and isinstance(boundary_steps[0], dict) else {})
  linear0 = (
      linear_steps[0]
      if linear_steps and isinstance(linear_steps[0], dict) else {})
  norm0 = norm_steps[0] if norm_steps and isinstance(norm_steps[0], dict) else {}
  layer11_boundary = _find_layer(boundary0, 11)
  layer12_boundary = _find_layer(boundary0, 12)
  layer12_linear = _find_layer(linear0, 12)
  layer12_ffn_norm = _find_layer(norm0, 12)

  route_selects = (
      predecessor.get("measurement_complete") is True
      and predecessor.get("router_math_distribution_passed") is False
      and predecessor.get("required_checks_passed") is False
      and predecessor.get("selected_next_route") == CURRENT_ROUTE
      and _has_candidate(routes, 593, CURRENT_ROUTE)
      and _has_switch(
          routes, 593,
          "select_router_prompt_distribution_fused_exact_linear_projection_"
          "route_close_gate"))
  completed_failure = (
      predecessor.get("disposition")
      == "reject_fused_exact_projection_router_math_distribution"
      and distribution.get("position_count") == 8
      and distribution.get("top1_match_count") == 7
      and distribution.get("top1_pass") is False
      and float(distribution.get("max_kld", 0.0)) > 0.005
      and token7.get("token_index") == 7
      and token7.get("top1_matches") is False
      and token7.get("gpu_top1_id") == 421
      and token7.get("native_top1_id") == 25)
  selectors_locked = (
      smoke.get("linear_output_projection_rowblock16_cpuorder_finalize") is True
      and smoke.get("linear_output_projection_cpu_order_layers") == LINEAR_LAYERS
      and smoke.get("input_rmsnorm_serial_reduction_layers") == LINEAR_LAYERS
      and smoke.get("linear_final_device_q8_handoff_enabled") is True)
  component_rows = component.get("rows", [])
  component_was_exact_and_budgeted = (
      component.get("required_checks_passed") is True
      and component.get("component_passed") is True
      and len(component_rows) == 2
      and all(
          isinstance(row, dict)
          and row.get("exactness_passed") is True
          and row.get("budget_passed") is True
          for row in component_rows))
  route_is_recorded_closed = bool(_find_rejection(
      rejected,
      "router_prompt_distribution_all_linear_fused_exact_projection_and_"
      "serial_input_rmsnorm"))

  layer11_output_abs = float(
      layer11_boundary.get("output_max_abs_diff", float("inf")))
  layer12_input_abs = float(
      layer12_boundary.get("input_max_abs_diff", float("inf")))
  layer12_output_abs = float(
      layer12_boundary.get("output_max_abs_diff", 0.0))
  layer12_linear_final_abs = float(
      layer12_linear.get("final_output_max_abs_diff", float("inf")))
  layer12_ffn_norm_abs = float(
      layer12_ffn_norm.get("max_abs_diff", float("inf")))
  layer12_source_selects_ffn = (
      token0.get("token_index") == 0
      and layer11_output_abs <= 3.0e-7
      and layer12_input_abs <= 3.0e-7
      and layer12_output_abs >= 1.0e-4
      and layer12_linear_final_abs <= 1.0e-6
      and layer12_ffn_norm_abs > layer12_linear_final_abs
      and layer12_output_abs > 1000.0 * layer12_linear_final_abs)

  diagnostic_contract = {
      "case_id": "router_math_reason_001",
      "binary": predecessor.get("inputs", {}).get("binary"),
      "binary_key": predecessor.get("inputs", {}).get("binary_key"),
      "decode_tokens": 1,
      "teacher_forced": True,
      "diagnostic_layer_range": "12:13",
      "diagnostic_token_limit": 1,
      "keep_fused_exact_projection_enabled": True,
      "keep_serial_input_rmsnorm_enabled": True,
      "component_trace_only": True,
      "router_code_allowed": False,
      "speed_probe_allowed": False,
      "validation_or_test_allowed": False,
      "projection_variant_allowed": False,
      "layer_subset_sweep_allowed": False,
  }
  contract_is_bounded = (
      diagnostic_contract["decode_tokens"] == 1
      and diagnostic_contract["diagnostic_layer_range"] == "12:13"
      and diagnostic_contract["diagnostic_token_limit"] == 1
      and diagnostic_contract["component_trace_only"] is True
      and diagnostic_contract["router_code_allowed"] is False
      and diagnostic_contract["speed_probe_allowed"] is False
      and diagnostic_contract["projection_variant_allowed"] is False
      and diagnostic_contract["layer_subset_sweep_allowed"] is False)

  checks = [
      {"name": "seq593_selected_fused_route_close", "pass": route_selects},
      {"name": "seq593_is_complete_distribution_failure",
       "pass": completed_failure,
       "detail": {
           "position_count": distribution.get("position_count"),
           "top1_match_count": distribution.get("top1_match_count"),
           "max_kld": distribution.get("max_kld"),
           "token7_gpu_top1_id": token7.get("gpu_top1_id"),
           "token7_native_top1_id": token7.get("native_top1_id"),
       }},
      {"name": "fused_selectors_were_locked_to_all_30_linear_layers",
       "pass": selectors_locked},
      {"name": "component_was_exact_and_inside_floor_budget",
       "pass": component_was_exact_and_budgeted},
      {"name": "standalone_fused_route_is_in_rejected_ledger",
       "pass": route_is_recorded_closed},
      {"name": "token0_selects_layer12_ffn_not_linear_projection",
       "pass": layer12_source_selects_ffn,
       "detail": {
           "layer11_output_max_abs_diff": layer11_output_abs,
           "layer12_input_max_abs_diff": layer12_input_abs,
           "layer12_output_max_abs_diff": layer12_output_abs,
           "layer12_linear_final_max_abs_diff": layer12_linear_final_abs,
           "layer12_ffn_norm_max_abs_diff": layer12_ffn_norm_abs,
       }},
      {"name": "next_diagnostic_is_one_token_one_layer_no_speed",
       "pass": contract_is_bounded,
       "detail": diagnostic_contract},
  ]
  required = all(bool(row["pass"]) for row in checks)
  selected_next = SELECTED_NEXT_ROUTE if required else REFLECTION_ROUTE
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "sequence": args.sequence,
      "inputs": {
          "routes": _rel(args.routes),
          "rejected_routes": _rel(args.rejected_routes),
          "predecessor": _rel(args.predecessor),
          "smoke": _rel(args.smoke),
          "component": _rel(args.component),
      },
      "checks": checks,
      "required_checks_passed": required,
      "fused_exact_projection_route_closed": required,
      "one_token_diagnostic_allowed": required,
      "router_code_distribution_allowed": False,
      "speed_probe_allowed": False,
      "validation_or_test_allowed": False,
      "speedup_claims_allowed": False,
      "promotion_allowed": False,
      "diagnostic_contract": diagnostic_contract,
      "disposition": (
          "close_fused_exact_projection_select_layer12_ffn_attribution"
          if required else
          "close_fused_exact_projection_without_bounded_successor"),
      "selected_next_route": selected_next,
      "next_route_reason": (
          "The fused exact all-linear bundle fails the complete router-math "
          "ruler despite an exact, budgeted projection component. Token 0 "
          "stays sub-ULP through layer 11, and layer 12's linear-attention "
          "final remains sub-ULP while the full layer output becomes material. "
          "Run one layer-12 FFN component trace from the same binary; do not "
          "change math or run router-code/speed."
          if required else
          "Repair route closure or source-boundary evidence before another "
          "target row."),
  }


def write_outputs(metrics: dict[str, Any], out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  (out_dir / "manifest.json").write_text(
      json.dumps({
          "schema_version": metrics["schema_version"],
          "workstream": metrics["workstream"],
          "tool": _rel(Path(__file__)),
          "inputs": metrics["inputs"],
          "diagnostic_contract": metrics["diagnostic_contract"],
          "selected_next_route": metrics["selected_next_route"],
          "token_executed": False,
          "speedup_claims_allowed": False,
      }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  failed = [row["name"] for row in metrics["checks"] if not row["pass"]]
  source = next(
      row["detail"] for row in metrics["checks"]
      if row["name"] == "token0_selects_layer12_ffn_not_linear_projection")
  lines = [
      f"# Seq{metrics['sequence']} Fused Exact Projection Route Close",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- layer11 output max abs: `{source['layer11_output_max_abs_diff']}`",
      f"- layer12 input / output max abs: "
      f"`{source['layer12_input_max_abs_diff']}` / "
      f"`{source['layer12_output_max_abs_diff']}`",
      f"- layer12 linear final max abs: "
      f"`{source['layer12_linear_final_max_abs_diff']}`",
      f"- failed_checks: `{failed}`",
      "",
      metrics["next_route_reason"],
      "",
      "No token, validation case, test case, or speed row was executed.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--sequence", type=int, default=594)
  parser.add_argument("--routes", type=Path,
                      default=ACTIVE / "routes-ledger.json")
  parser.add_argument("--rejected-routes", type=Path,
                      default=ACTIVE / "rejected-routes.json")
  parser.add_argument(
      "--predecessor", type=Path,
      default=ROOT / "output/seq593-fused-exact-linear-projection-router-math-distribution-gate-20260710Tseq593Z/metrics.json")
  parser.add_argument(
      "--smoke", type=Path,
      default=ROOT / "output/seq593-fused-exact-linear-projection-router-math-distribution-gate-20260710Tseq593Z/smoke.json")
  parser.add_argument(
      "--component", type=Path,
      default=ROOT / "output/seq589-fused-exact-linear-projection-component-probe-gate-20260710Tseq589Z/metrics.json")
  parser.add_argument(
      "--out-dir", type=Path,
      default=ROOT / "output/seq594-fused-exact-linear-projection-route-close-gate-20260710Tseq594Z")
  args = parser.parse_args()
  metrics = compute(args)
  write_outputs(metrics, args.out_dir)
  print(json.dumps({
      "required_checks_passed": metrics["required_checks_passed"],
      "disposition": metrics["disposition"],
      "selected_next_route": metrics["selected_next_route"],
      "diagnostic_contract": metrics["diagnostic_contract"],
      "out_dir": _rel(args.out_dir),
  }, sort_keys=True))
  return 0 if metrics["required_checks_passed"] else 2


if __name__ == "__main__":
  raise SystemExit(main())
