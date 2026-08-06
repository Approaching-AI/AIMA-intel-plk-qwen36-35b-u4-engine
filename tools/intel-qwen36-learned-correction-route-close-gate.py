#!/usr/bin/env python3
"""Close learned correction and lock an exact-kernel kill-number gate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = "intel-qwen36-learned-correction-route-close-gate-v0"
DESIGN_SCHEMA_VERSION = "intel-qwen36-fused-exact-projection-budget-design-v0"
CURRENT_ROUTE = (
    "router_prompt_distribution_final_norm_sparse_state_feature_route_close_gate"
)
SELECTED_NEXT_ROUTE = (
    "router_prompt_distribution_fused_exact_linear_projection_budget_design_gate"
)


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


def _candidate(routes: dict[str, Any], seq: int) -> dict[str, Any]:
  for row in routes.get("candidate_history", []):
    if isinstance(row, dict) and row.get("seq") == seq:
      return row
  return {}


def compute(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
  routes = _load(args.routes)
  predecessor = _load(args.predecessor)
  prior_budget = _load(args.prior_budget)
  corpus = _load(args.corpus_contract)
  kill = prior_budget.get("kill_number", {})
  layer_count = int(kill.get("linear_layer_count", 0))
  floor_headroom = float(kill.get("floor_headroom_us", math.inf))
  current_extra = float(kill.get("per_layer_projection_extra_us", math.inf))
  maximum_added_per_layer = floor_headroom / layer_count if layer_count else 0.0
  required_reduction = (
      current_extra / maximum_added_per_layer
      if maximum_added_per_layer > 0.0 else math.inf)
  predecessor_selects = (
      predecessor.get("required_checks_passed") is True
      and predecessor.get("sparse_state_feasibility_passed") is False
      and predecessor.get("validation_allowed") is False
      and predecessor.get("runtime_selected_dimension_source_allowed") is False
      and predecessor.get("selected_next_route") == CURRENT_ROUTE
      and _has_candidate(routes, 584, CURRENT_ROUTE)
      and _has_switch(
          routes, 584,
          "select_router_prompt_distribution_final_norm_sparse_state_"
          "feature_route_close_gate"))
  learned_board_closed = (
      _candidate(routes, 568).get("disposition")
      == "reject_static_sparse_head_logit_bias_on_holdout_regression"
      and _candidate(routes, 575).get("disposition")
      == "reject_fit_model_before_validation"
      and _candidate(routes, 577).get("disposition")
      == "reject_mass_matched_feature_reframe_before_validation"
      and _candidate(routes, 579).get("disposition")
      == "reject_observable_only_head_correction_before_validation"
      and _candidate(routes, 584).get("disposition")
      == "reject_sparse_final_norm_state_model_before_validation")
  kill_number_valid = (
      prior_budget.get("required_checks_passed") is True
      and layer_count == 30
      and math.isclose(floor_headroom, 205.25576981789345,
                       rel_tol=0.0, abs_tol=1e-9)
      and math.isclose(current_extra, 17.811999999999983,
                       rel_tol=0.0, abs_tol=1e-9)
      and math.isclose(maximum_added_per_layer, 6.841858993929782,
                       rel_tol=0.0, abs_tol=1e-12)
      and math.isclose(required_reduction, 2.6033860118723733,
                       rel_tol=0.0, abs_tol=1e-12))
  blocked_ids = [
      row.get("id") for row in corpus.get("prompts", [])
      if isinstance(row, dict) and row.get("split") in ("validation", "test")
  ]
  blocked_paths = [
      str(path) for case_id in blocked_ids
      for path in args.output_root.glob(f"*{case_id}*/result.json")
      if "seq574-state-conditioned-head-fit-collection" in str(path)
      or "seq583-final-norm-sparse-state-fit-collection" in str(path)
  ]
  untouched_splits = len(blocked_ids) == 12 and not blocked_paths
  design = {
      "schema_version": DESIGN_SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "route": SELECTED_NEXT_ROUTE,
      "purpose": (
          "Reopen exact all-linear parity only through a same-context fused "
          "projection implementation that fits the current floor headroom."
      ),
      "kill_number": {
          "baseline_tok_s": kill.get("baseline_tok_s"),
          "floor_tok_s": kill.get("floor_tok_s"),
          "floor_headroom_us_per_token": floor_headroom,
          "linear_layer_count": layer_count,
          "current_separate_projection_added_us_per_layer": current_extra,
          "maximum_fused_added_us_per_layer": maximum_added_per_layer,
          "required_added_cost_reduction_ratio": required_reduction,
      },
      "implementation_contract": {
          "same_device_context_and_existing_output_buffer_required": True,
          "separate_cpu_order_projection_dispatch_allowed": False,
          "host_bridge_or_full_vector_read_allowed": False,
          "per_layer_source_files_allowed": False,
          "parameterized_one_layer_plus_loop_required": True,
          "component_exactness_required": True,
          "representative_layer_added_wall_us_max": maximum_added_per_layer,
          "component_repeat_and_confirm_required": True,
          "all_linear_token_row_allowed_before_component_pass": False,
          "subset_sweep_allowed": False,
      },
      "claim_policy": {
          "correctness_claim_allowed": False,
          "speedup_claim_allowed": False,
          "promotion_allowed": False,
      },
  }
  design_passes = (
      design["implementation_contract"][
          "same_device_context_and_existing_output_buffer_required"] is True
      and design["implementation_contract"][
          "separate_cpu_order_projection_dispatch_allowed"] is False
      and design["implementation_contract"][
          "host_bridge_or_full_vector_read_allowed"] is False
      and design["implementation_contract"][
          "all_linear_token_row_allowed_before_component_pass"] is False
      and design["implementation_contract"]["subset_sweep_allowed"] is False)
  checks = [
      {"name": "seq584_selected_learned_route_close_gate",
       "pass": predecessor_selects},
      {"name": "static_top8_and_sparse_state_learned_models_are_closed",
       "pass": learned_board_closed},
      {"name": "seq563_floor_kill_number_is_preserved",
       "pass": kill_number_valid,
       "detail": design["kill_number"]},
      {"name": "validation_and_test_splits_remain_untouched",
       "pass": untouched_splits,
       "detail": {"blocked_case_ids": blocked_ids,
                  "found_paths": blocked_paths}},
      {"name": "fused_exact_component_contract_blocks_tokens_and_sweeps",
       "pass": design_passes},
  ]
  required = all(bool(row["pass"]) for row in checks)
  metrics = {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "sequence": args.sequence,
      "inputs": {
          "routes": _rel(args.routes),
          "predecessor": _rel(args.predecessor),
          "prior_budget": _rel(args.prior_budget),
          "corpus_contract": _rel(args.corpus_contract),
      },
      "design": design,
      "checks": checks,
      "required_checks_passed": required,
      "learned_correction_routes_closed": required,
      "component_design_allowed": required,
      "component_probe_allowed": False,
      "token_row_allowed": False,
      "validation_or_test_allowed": False,
      "speedup_claims_allowed": False,
      "promotion_allowed": False,
      "disposition": (
          "close_learned_correction_select_fused_exact_projection_budget_design"
          if required else "close_learned_correction_without_successor"),
      "selected_next_route": (
          SELECTED_NEXT_ROUTE if required else
          "router_prompt_distribution_correctness_route_reflection_gate"),
      "next_route_reason": (
          "Learned correction is closed. The only exact successor must first "
          "design a same-context fused projection whose representative-layer "
          "added wall is <= 6.841858994 us, a 2.603386x reduction from the "
          "current separate implementation. No component probe or token is "
          "authorized until the design gate passes."
          if required else
          "Repair learned-route closure, untouched-split proof, or kill-number "
          "consistency before exact-kernel work."),
  }
  return metrics, design


def write_outputs(metrics: dict[str, Any], design: dict[str, Any],
                  out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  (out_dir / "design.json").write_text(
      json.dumps(design, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  (out_dir / "manifest.json").write_text(
      json.dumps({
          "schema_version": metrics["schema_version"],
          "workstream": metrics["workstream"],
          "tool": _rel(Path(__file__)),
          "inputs": metrics["inputs"],
          "design": _rel(out_dir / "design.json"),
          "selected_next_route": metrics["selected_next_route"],
          "component_probe_allowed": False,
          "token_row_allowed": False,
          "validation_or_test_allowed": False,
          "speedup_claims_allowed": False,
      }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  failed = [row["name"] for row in metrics["checks"] if not row["pass"]]
  kill = design["kill_number"]
  lines = [
      f"# Seq{metrics['sequence']} Learned-Correction Route Close",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- current / maximum added us per layer: "
      f"`{kill['current_separate_projection_added_us_per_layer']}` / "
      f"`{kill['maximum_fused_added_us_per_layer']}`",
      f"- required reduction: `{kill['required_added_cost_reduction_ratio']}x`",
      f"- failed_checks: `{failed}`",
      "",
      metrics["next_route_reason"],
      "",
      "No component probe, token, validation case, or test case was used.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--sequence", type=int, default=585)
  parser.add_argument("--routes", type=Path,
                      default=ACTIVE / "routes-ledger.json")
  parser.add_argument(
      "--predecessor", type=Path,
      default=ROOT / "output/seq584-final-norm-sparse-state-dimension-fit-gate-20260710Tseq584Z/metrics.json")
  parser.add_argument(
      "--prior-budget", type=Path,
      default=ROOT / "output/seq563-all-linear-norm-to-projection-parity-feasibility-gate-20260710Tseq563Z/metrics.json")
  parser.add_argument(
      "--corpus-contract", type=Path,
      default=ACTIVE / "intel-qwen36-35b-a3b-gguf-q4km-state-conditioned-head-correction-corpus-2026-07-10.json")
  parser.add_argument("--output-root", type=Path, default=ROOT / "output")
  parser.add_argument(
      "--out-dir", type=Path,
      default=ROOT / "output/seq585-learned-correction-route-close-gate-20260710Tseq585Z")
  args = parser.parse_args()
  metrics, design = compute(args)
  write_outputs(metrics, design, args.out_dir)
  print(json.dumps({
      "required_checks_passed": metrics["required_checks_passed"],
      "disposition": metrics["disposition"],
      "learned_correction_routes_closed": metrics[
          "learned_correction_routes_closed"],
      "maximum_fused_added_us_per_layer": design[
          "kill_number"]["maximum_fused_added_us_per_layer"],
      "required_reduction_ratio": design[
          "kill_number"]["required_added_cost_reduction_ratio"],
      "selected_next_route": metrics["selected_next_route"],
      "out_dir": _rel(args.out_dir),
  }, sort_keys=True))
  return 0 if metrics["required_checks_passed"] else 2


if __name__ == "__main__":
  raise SystemExit(main())
