#!/usr/bin/env python3
"""Trace layer-12 FFN input drift and budget one whole parity component."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = "intel-qwen36-layer12-ffn-input-source-gate-v0"
DESIGN_SCHEMA_VERSION = (
    "intel-qwen36-all-linear-preprojection-parity-budget-contract-v0"
)
CURRENT_ROUTE = "router_prompt_distribution_layer12_ffn_input_source_gate"
SELECTED_NEXT_ROUTE = (
    "router_prompt_distribution_all_linear_preprojection_parity_budget_design_gate"
)
REFLECTION_ROUTE = "router_prompt_distribution_correctness_route_reflection_gate"
SOURCE_LAYER = 0
CLOSED_STANDALONE_ROUTES = [
    "gpu_attention_front_handoff_8tok_opencl_q6_lane_sums_nofma_diagnostic",
    "gpu_attention_front_handoff_8tok_q4_cpu_order_linear_ab_diagnostic",
    "gpu_attention_front_handoff_8tok_cpu_linear_postconv_prep_diagnostic",
    "gpu_attention_front_handoff_8tok_linear_delta_cpu_shape_diagnostic",
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


def _step(smoke: dict[str, Any], table: str, token_index: int = 0) -> dict[str, Any]:
  rows = smoke.get(table, [])
  if not isinstance(rows, list):
    return {}
  for row in rows:
    if isinstance(row, dict) and row.get("token_index") == token_index:
      return row
  return {}


def _layer(step: dict[str, Any], layer: int) -> dict[str, Any]:
  rows = step.get("layers", [])
  if not isinstance(rows, list):
    return {}
  for row in rows:
    if isinstance(row, dict) and row.get("layer") == layer:
      return row
  return {}


def _rejected_map(rejected: dict[str, Any]) -> dict[str, dict[str, Any]]:
  return {
      str(row.get("route")): row
      for row in rejected.get("rejected", [])
      if isinstance(row, dict) and isinstance(row.get("route"), str)
  }


def compute(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
  routes = _load(args.routes)
  rejected = _load(args.rejected_routes)
  predecessor = _load(args.predecessor)
  smoke = _load(args.smoke)
  component = _load(args.projection_component)
  prior_budget = _load(args.prior_budget)

  boundary0 = _layer(_step(smoke, "layer_boundary_diff_by_step"), 0)
  preconv0 = _layer(_step(smoke, "linear_preconv_source_diff_by_step"), 0)
  attention0 = _layer(_step(smoke, "linear_attention_diff_by_step"), 0)
  final_mix0 = _layer(_step(smoke, "linear_final_mix_diff_by_step"), 0)
  ffn_live0 = _layer(_step(smoke, "ffn_live_math_diff_by_step"), 0)

  route_selects = (
      predecessor.get("required_checks_passed") is True
      and predecessor.get("local_ffn_math_mismatch") is False
      and predecessor.get("selected_next_route") == CURRENT_ROUTE
      and _has_candidate(routes, 595, CURRENT_ROUTE)
      and _has_switch(
          routes, 595,
          "select_router_prompt_distribution_layer12_ffn_input_source_gate"))
  first_nonzero_is_exact_input_layer0 = (
      boundary0.get("input_max_abs_diff") == 0
      and 0.0 < float(boundary0.get("output_max_abs_diff", 0.0)) <= 1.0e-6)
  layer0_source_shape = {
      "layer_input_max_abs_diff": boundary0.get("input_max_abs_diff"),
      "layer_output_max_abs_diff": boundary0.get("output_max_abs_diff"),
      "attn_norm_from_gpu_input_max_abs_diff": preconv0.get(
          "attn_norm_from_gpu_input_max_abs_diff"),
      "gpu_attn_norm_vs_cpu_max_abs_diff": preconv0.get(
          "gpu_attn_norm_vs_cpu_max_abs_diff"),
      "gpu_qkv_vs_cpu_max_abs_diff": preconv0.get(
          "gpu_qkv_vs_cpu_max_abs_diff"),
      "gpu_gate_vs_cpu_max_abs_diff": preconv0.get(
          "gpu_gate_vs_cpu_max_abs_diff"),
      "gpu_beta_vs_cpu_max_abs_diff": preconv0.get(
          "gpu_beta_vs_cpu_max_abs_diff"),
      "conv_output_raw_max_abs_diff": preconv0.get(
          "conv_output_raw_max_abs_diff"),
      "delta_output_max_abs_diff": attention0.get("delta_output_max_abs_diff"),
      "linear_final_output_max_abs_diff": attention0.get(
          "final_output_max_abs_diff"),
      "same_input_final_mix_max_abs_diff": final_mix0.get(
          "gpu_delta_gpu_z_cpu_max_abs_diff"),
      "same_input_full_ffn_max_abs_diff": ffn_live0.get(
          "gpu_output_vs_cpu_ffn_max_abs_diff"),
  }
  source_is_preprojection_bundle = (
      first_nonzero_is_exact_input_layer0
      and layer0_source_shape["attn_norm_from_gpu_input_max_abs_diff"] == 0
      and layer0_source_shape["gpu_attn_norm_vs_cpu_max_abs_diff"] == 0
      and float(layer0_source_shape["gpu_qkv_vs_cpu_max_abs_diff"] or 0) > 0
      and float(layer0_source_shape["gpu_gate_vs_cpu_max_abs_diff"] or 0) > 0
      and float(layer0_source_shape["conv_output_raw_max_abs_diff"] or 0) > 0
      and float(layer0_source_shape["delta_output_max_abs_diff"] or 0) > 0
      and float(layer0_source_shape["linear_final_output_max_abs_diff"] or 0) > 0
      and float(layer0_source_shape["same_input_final_mix_max_abs_diff"] or 0)
      <= 1.0e-6
      and float(layer0_source_shape["same_input_full_ffn_max_abs_diff"] or 0)
      <= 2.0e-5)

  rejected_by_route = _rejected_map(rejected)
  standalone_axes_closed = all(
      route in rejected_by_route for route in CLOSED_STANDALONE_ROUTES)

  rows = component.get("rows", [])
  rows = rows if isinstance(rows, list) else []
  projection_credits = [
      float(row["rowblock16_baseline_min_us"])
      - float(row["candidate_min_us"])
      for row in rows
      if isinstance(row, dict)
      and row.get("exactness_passed") is True
      and row.get("budget_passed") is True
      and isinstance(row.get("rowblock16_baseline_min_us"), (int, float))
      and isinstance(row.get("candidate_min_us"), (int, float))
  ]
  kill = prior_budget.get("kill_number", {})
  floor_headroom_per_layer = float(
      kill.get("maximum_added_us_per_layer", 0.0))
  if floor_headroom_per_layer == 0.0:
    layer_count = int(kill.get("linear_layer_count", 0))
    floor_headroom = float(kill.get("floor_headroom_us", 0.0))
    floor_headroom_per_layer = (
        floor_headroom / layer_count if layer_count else 0.0)
  conservative_projection_credit = (
      min(projection_credits) if len(projection_credits) == 2 else 0.0)
  maximum_preprojection_added_us = (
      floor_headroom_per_layer + conservative_projection_credit)
  budget_is_valid = (
      component.get("component_passed") is True
      and len(projection_credits) == 2
      and math.isclose(
          floor_headroom_per_layer, 6.841858993929781,
          rel_tol=0.0, abs_tol=1.0e-12)
      and math.isclose(
          conservative_projection_credit, 29.375000000000014,
          rel_tol=0.0, abs_tol=1.0e-12)
      and math.isclose(
          maximum_preprojection_added_us, 36.216858993929795,
          rel_tol=0.0, abs_tol=1.0e-12))

  design = {
      "schema_version": DESIGN_SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "route": SELECTED_NEXT_ROUTE,
      "purpose": (
          "Prove one whole linear preprojection/recurrent component can make "
          "CPU-equivalent `final_output` and state from an exact layer input "
          "without reopening closed arithmetic axes one at a time."
      ),
      "source_boundary": layer0_source_shape,
      "budget": {
          "floor_headroom_us_per_linear_layer": floor_headroom_per_layer,
          "conservative_fused_projection_credit_us_per_layer":
              conservative_projection_credit,
          "maximum_preprojection_added_us_per_layer":
              maximum_preprojection_added_us,
          "linear_layer_count": kill.get("linear_layer_count"),
      },
      "component_contract": {
          "one_parameterized_layer_plus_loop_required": True,
          "exact_input_and_recurrent_state_required": True,
          "q6_qkv_q4_gate_conv_postconv_delta_final_gate_in_scope": True,
          "existing_fused_exact_projection_kept_in_scope": True,
          "final_output_and_recurrent_state_bit_exact_required": True,
          "same_device_context_and_resident_buffers_required": True,
          "host_bridge_or_cpu_fallback_allowed": False,
          "standalone_axis_probe_allowed": False,
          "layer_subset_sweep_allowed": False,
          "paired_component_repeat_and_confirm_required": True,
          "paired_total_added_us_per_layer_max": maximum_preprojection_added_us,
          "token_row_allowed_before_component_pass": False,
      },
      "claim_policy": {
          "correctness_claim_allowed": False,
          "speedup_claim_allowed": False,
          "promotion_allowed": False,
      },
  }
  contract_is_bounded = (
      design["component_contract"]["host_bridge_or_cpu_fallback_allowed"]
      is False
      and design["component_contract"]["standalone_axis_probe_allowed"]
      is False
      and design["component_contract"]["layer_subset_sweep_allowed"] is False
      and design["component_contract"]["token_row_allowed_before_component_pass"]
      is False
      and design["component_contract"]["final_output_and_recurrent_state_bit_exact_required"]
      is True)

  checks = [
      {"name": "seq595_selected_layer12_ffn_input_source",
       "pass": route_selects},
      {"name": "exact_input_layer0_is_first_nonzero_source",
       "pass": first_nonzero_is_exact_input_layer0,
       "detail": {
           "layer0_input_max_abs_diff": boundary0.get("input_max_abs_diff"),
           "layer0_output_max_abs_diff": boundary0.get("output_max_abs_diff"),
       }},
      {"name": "source_is_preprojection_recurrent_bundle_not_ffn_or_projection",
       "pass": source_is_preprojection_bundle,
       "detail": layer0_source_shape},
      {"name": "standalone_arithmetic_axes_are_closed",
       "pass": standalone_axes_closed,
       "detail": {"routes": CLOSED_STANDALONE_ROUTES}},
      {"name": "whole_component_has_conservative_floor_budget",
       "pass": budget_is_valid,
       "detail": design["budget"]},
      {"name": "component_contract_blocks_axis_sweeps_and_tokens",
       "pass": contract_is_bounded},
  ]
  required = all(bool(row["pass"]) for row in checks)
  selected_next = SELECTED_NEXT_ROUTE if required else REFLECTION_ROUTE
  metrics = {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "sequence": args.sequence,
      "inputs": {
          "routes": _rel(args.routes),
          "rejected_routes": _rel(args.rejected_routes),
          "predecessor": _rel(args.predecessor),
          "smoke": _rel(args.smoke),
          "projection_component": _rel(args.projection_component),
          "prior_budget": _rel(args.prior_budget),
      },
      "checks": checks,
      "required_checks_passed": required,
      "source_control_passed": required,
      "component_design_allowed": required,
      "component_probe_allowed": False,
      "token_row_allowed": False,
      "router_code_distribution_allowed": False,
      "validation_or_test_allowed": False,
      "speedup_claims_allowed": False,
      "promotion_allowed": False,
      "design": design,
      "disposition": (
          "close_layer12_input_chase_select_whole_preprojection_budget_design"
          if required else
          "close_layer12_input_chase_without_admissible_component"),
      "selected_next_route": selected_next,
      "next_route_reason": (
          "The layer-12 input delta traces to exact-input layer 0, where input "
          "norm and the downstream projection/FFN are already controlled but "
          "Q6 QKV, gate, conv/postconv, and recurrent arithmetic introduce the "
          "first sub-ULP state. Their standalone routes are closed. Design one "
          "whole exact preprojection/recurrent component whose repeat/confirm "
          "added wall is <=36.216858994 us/layer after conservative fused-"
          "projection credit; no token or axis sweep is allowed first."
          if required else
          "Repair source attribution, closed-axis proof, or the conservative "
          "floor budget before implementation or target work."),
  }
  return metrics, design


def write_outputs(metrics: dict[str, Any], design: dict[str, Any],
                  out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  (out_dir / "design.json").write_text(
      json.dumps(design, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  failed = [row["name"] for row in metrics["checks"] if not row["pass"]]
  budget = design["budget"]
  lines = [
      f"# Seq{metrics['sequence']} Layer-12 FFN-Input Source Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- floor headroom: `{budget['floor_headroom_us_per_linear_layer']} us/layer`",
      f"- conservative projection credit: "
      f"`{budget['conservative_fused_projection_credit_us_per_layer']} us/layer`",
      f"- maximum preprojection added wall: "
      f"`{budget['maximum_preprojection_added_us_per_layer']} us/layer`",
      f"- failed_checks: `{failed}`",
      "",
      metrics["next_route_reason"],
      "",
      "No component, token, validation case, test case, or speed row was run.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--sequence", type=int, default=596)
  parser.add_argument("--routes", type=Path,
                      default=ACTIVE / "routes-ledger.json")
  parser.add_argument("--rejected-routes", type=Path,
                      default=ACTIVE / "rejected-routes.json")
  parser.add_argument(
      "--predecessor", type=Path,
      default=ROOT / "output/seq595-layer12-ffn-component-attribution-gate-20260710Tseq595Z/metrics.json")
  parser.add_argument(
      "--smoke", type=Path,
      default=ROOT / "output/seq593-fused-exact-linear-projection-router-math-distribution-gate-20260710Tseq593Z/smoke.json")
  parser.add_argument(
      "--projection-component", type=Path,
      default=ROOT / "output/seq589-fused-exact-linear-projection-component-probe-gate-20260710Tseq589Z/metrics.json")
  parser.add_argument(
      "--prior-budget", type=Path,
      default=ROOT / "output/seq563-all-linear-norm-to-projection-parity-feasibility-gate-20260710Tseq563Z/metrics.json")
  parser.add_argument(
      "--out-dir", type=Path,
      default=ROOT / "output/seq596-layer12-ffn-input-source-gate-20260710Tseq596Z")
  args = parser.parse_args()
  metrics, design = compute(args)
  write_outputs(metrics, design, args.out_dir)
  print(json.dumps({
      "required_checks_passed": metrics["required_checks_passed"],
      "disposition": metrics["disposition"],
      "selected_next_route": metrics["selected_next_route"],
      "maximum_preprojection_added_us_per_layer": design["budget"][
          "maximum_preprojection_added_us_per_layer"],
      "out_dir": _rel(args.out_dir),
  }, sort_keys=True))
  return 0 if metrics["required_checks_passed"] else 2


if __name__ == "__main__":
  raise SystemExit(main())
