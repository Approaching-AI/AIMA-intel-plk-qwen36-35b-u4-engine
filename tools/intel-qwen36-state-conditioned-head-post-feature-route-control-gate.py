#!/usr/bin/env python3
"""Lock the sole remaining whole-top8 shape model before fitting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = (
    "intel-qwen36-state-conditioned-head-post-feature-route-control-gate-v0"
)
DESIGN_SCHEMA_VERSION = (
    "intel-qwen36-state-conditioned-head-top8-shape-model-design-v0"
)
CURRENT_ROUTE = (
    "router_prompt_distribution_state_conditioned_head_correction_"
    "post_feature_reframe_route_control_gate"
)
SELECTED_NEXT_ROUTE = (
    "router_prompt_distribution_state_conditioned_head_correction_"
    "top8_shape_feature_fit_gate"
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


def compute(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
  routes = _load(args.routes)
  predecessor = _load(args.predecessor)
  oracle = _load(args.oracle_gate)
  corpus = _load(args.corpus_contract)
  model_contract = corpus.get("model_contract", {})
  fit_ids = [
      row.get("id") for row in corpus.get("prompts", [])
      if isinstance(row, dict) and row.get("split") == "fit"
  ]
  selected_token_ids = oracle.get("selected_token_ids", [])
  parameter_count = len(selected_token_ids) + 1 + 7 + 1
  predecessor_selects = (
      predecessor.get("required_checks_passed") is True
      and predecessor.get("feature_reframe_feasibility_passed") is False
      and predecessor.get("validation_allowed") is False
      and predecessor.get("selected_next_route") == CURRENT_ROUTE
      and _has_candidate(routes, 577, CURRENT_ROUTE)
      and _has_switch(
          routes, 577,
          "select_router_prompt_distribution_state_conditioned_head_"
          "correction_post_feature_reframe_route_control_gate"))
  oracle_supports_reframe = (
      oracle.get("required_checks_passed") is True
      and oracle.get("selected16_oracle_contract_passed") is True
      and oracle.get("all_top8_oracle_contract_passed") is True
      and len(selected_token_ids) == 16)
  contract_supports_design = (
      model_contract.get("maximum_parameters") == 32
      and model_contract.get("implementation_added_wall_us_per_token_max")
      == 20.525576981789345
      and model_contract.get("full_vocab_host_rescan_in_speed_lane_allowed")
      is False
      and model_contract.get("allowed_runtime_features") == [
          "gpu_top8_token_ids", "gpu_top8_logits", "top1_margin"
      ])
  design = {
      "schema_version": DESIGN_SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "fit_target": "selected16 probability-mass-matched oracle correction",
      "training_split": "fit",
      "training_case_ids": fit_ids,
      "selected_token_ids": selected_token_ids,
      "runtime_features": [
          "gpu_top8_token_ids", "gpu_top8_logits", "top1_margin"
      ],
      "model_name": "token_intercepts_shared_whole_top8_shape",
      "equation": (
          "correction(token)=token_intercept[token]+own_gap_coefficient*"
          "gpu_logit_minus_top1[token]+top1_logit_coefficient*gpu_top1_logit+"
          "sum(rank_gap_coefficient[r]*(gpu_logit[r]-gpu_top1_logit),r=1..7)"
      ),
      "parameter_layout": {
          "token_intercepts": len(selected_token_ids),
          "own_gap_coefficient": 1,
          "rank_gap_coefficients": 7,
          "top1_logit_coefficient": 1,
          "total": parameter_count,
      },
      "fit_method": (
          "ridge regression with grouped leave-one-fit-prompt-out selection "
          "over the already frozen lambda grid"
      ),
      "maximum_parameters": 32,
      "implementation_added_wall_us_per_token_max":
      20.525576981789345,
      "runtime_native_oracle_allowed": False,
      "runtime_prompt_case_or_position_features_allowed": False,
      "full_vocab_host_rescan_allowed": False,
      "validation_or_test_used_for_fit": False,
      "additional_model_variants_allowed": False,
  }
  design_is_bounded = (
      parameter_count == 25
      and parameter_count <= int(model_contract.get("maximum_parameters", 0))
      and len(fit_ids) == 12
      and design["runtime_native_oracle_allowed"] is False
      and design["runtime_prompt_case_or_position_features_allowed"] is False
      and design["full_vocab_host_rescan_allowed"] is False
      and design["additional_model_variants_allowed"] is False)
  checks = [
      {"name": "seq577_selected_post_feature_route_control",
       "pass": predecessor_selects},
      {"name": "seq576_proves_selected16_and_all_top8_oracle_ceilings",
       "pass": oracle_supports_reframe},
      {"name": "locked_corpus_contract_allows_top8_shape_features",
       "pass": contract_supports_design},
      {"name": "single_whole_top8_design_has_exactly_25_parameters",
       "pass": design_is_bounded,
       "detail": design["parameter_layout"]},
  ]
  required = all(bool(row["pass"]) for row in checks)
  metrics = {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "sequence": args.sequence,
      "inputs": {
          "routes": _rel(args.routes),
          "predecessor": _rel(args.predecessor),
          "oracle_gate": _rel(args.oracle_gate),
          "corpus_contract": _rel(args.corpus_contract),
      },
      "design": design,
      "checks": checks,
      "required_checks_passed": required,
      "fit_allowed": required,
      "validation_or_test_allowed": False,
      "runtime_source_allowed": False,
      "speedup_claims_allowed": False,
      "promotion_allowed": False,
      "disposition": (
          "accept_single_top8_shape_feature_design"
          if required else "close_observable_only_head_correction"),
      "selected_next_route": (
          SELECTED_NEXT_ROUTE if required else
          "router_prompt_distribution_post_head_correction_route_control_gate"),
      "next_route_reason": (
          "The passing selected-16 oracle needs a per-step common offset that "
          "scalar token features cannot identify. Fit exactly one 25-parameter "
          "model using the full GPU-top8 shape; grouped fit-only CV and full "
          "fit must pass before validation."
          if required else
          "No bounded observable-only design remains; close head correction "
          "without validation or target work."),
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
          "parameter_count": design["parameter_layout"]["total"],
          "selected_next_route": metrics["selected_next_route"],
          "validation_or_test_allowed": False,
          "runtime_source_allowed": False,
          "speedup_claims_allowed": False,
      }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  failed = [row["name"] for row in metrics["checks"] if not row["pass"]]
  lines = [
      f"# Seq{metrics['sequence']} Whole-Top8 Feature Route Control",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- model: `{design['model_name']}`",
      f"- parameter count: `{design['parameter_layout']['total']}`",
      f"- failed_checks: `{failed}`",
      "",
      metrics["next_route_reason"],
      "",
      "No target command, validation case, or test case was used.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--sequence", type=int, default=578)
  parser.add_argument("--routes", type=Path,
                      default=ACTIVE / "routes-ledger.json")
  parser.add_argument(
      "--predecessor", type=Path,
      default=ROOT / "output/seq577-state-conditioned-head-feature-reframe-gate-20260710Tseq577Z/metrics.json")
  parser.add_argument(
      "--oracle-gate", type=Path,
      default=ROOT / "output/seq576-state-conditioned-head-post-fit-route-control-gate-20260710Tseq576Z/metrics.json")
  parser.add_argument(
      "--corpus-contract", type=Path,
      default=ACTIVE / "intel-qwen36-35b-a3b-gguf-q4km-state-conditioned-head-correction-corpus-2026-07-10.json")
  parser.add_argument(
      "--out-dir", type=Path,
      default=ROOT / "output/seq578-state-conditioned-head-post-feature-route-control-gate-20260710Tseq578Z")
  args = parser.parse_args()
  metrics, design = compute(args)
  write_outputs(metrics, design, args.out_dir)
  print(json.dumps({
      "required_checks_passed": metrics["required_checks_passed"],
      "disposition": metrics["disposition"],
      "model_name": design["model_name"],
      "parameter_count": design["parameter_layout"]["total"],
      "fit_allowed": metrics["fit_allowed"],
      "selected_next_route": metrics["selected_next_route"],
      "out_dir": _rel(args.out_dir),
  }, sort_keys=True))
  return 0 if metrics["required_checks_passed"] else 2


if __name__ == "__main__":
  raise SystemExit(main())
