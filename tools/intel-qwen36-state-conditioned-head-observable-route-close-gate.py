#!/usr/bin/env python3
"""Close top8-observable correction and gate one sparse final-norm source."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = "intel-qwen36-head-observable-route-close-gate-v0"
CURRENT_ROUTE = (
    "router_prompt_distribution_state_conditioned_head_correction_"
    "observable_route_close_gate"
)
SELECTED_NEXT_ROUTE = (
    "router_prompt_distribution_final_norm_sparse_state_feature_source_gate"
)
EXPECTED_TOKENS = [
    25, 421, 271, 198, 8160, 7888, 248068, 90700,
    24526, 760, 47205, 16, 279, 79767, 3840, 19205,
]
FIT_IDS = [
    "fresh_arithmetic_01", "fresh_arithmetic_02",
    "fresh_code_01", "fresh_code_02",
    "fresh_instruction_01", "fresh_instruction_02",
    "fresh_factual_01", "fresh_factual_02",
    "fresh_transformation_01", "fresh_transformation_02",
    "fresh_structured_01", "fresh_structured_02",
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


def _route_candidate(routes: dict[str, Any], seq: int) -> dict[str, Any]:
  for row in routes.get("candidate_history", []):
    if isinstance(row, dict) and row.get("seq") == seq:
      return row
  return {}


def compute(args: argparse.Namespace) -> dict[str, Any]:
  routes = _load(args.routes)
  predecessor = _load(args.predecessor)
  oracle = _load(args.oracle_gate)
  contract = _load(args.contract)
  source = args.smoke_source.read_text(encoding="utf-8")
  predecessor_selects = (
      predecessor.get("required_checks_passed") is True
      and predecessor.get("top8_shape_feasibility_passed") is False
      and predecessor.get("validation_allowed") is False
      and predecessor.get("selected_next_route") == CURRENT_ROUTE
      and _has_candidate(routes, 579, CURRENT_ROUTE)
      and _has_switch(
          routes, 579,
          "select_router_prompt_distribution_state_conditioned_head_"
          "correction_observable_route_close_gate"))
  observable_board_is_closed = (
      _route_candidate(routes, 575).get("disposition")
      == "reject_fit_model_before_validation"
      and _route_candidate(routes, 577).get("disposition")
      == "reject_mass_matched_feature_reframe_before_validation"
      and _route_candidate(routes, 579).get("disposition")
      == "reject_observable_only_head_correction_before_validation"
      and oracle.get("selected16_oracle_contract_passed") is True)
  final_norm_source_exists = all(marker in source for marker in (
      "std::vector<std::vector<float>> native_final_norm_by_step;",
      '"distribution ladder native final norm missing"',
      '"distribution ladder GPU final norm missing"',
      "gpu_next.final_norm, rms_norm_epsilon, native_topk, gpu_next.topk",
  ))
  prior_attribution_exists = (
      args.seq525_gate.is_file() and args.seq526_gate.is_file()
      and _route_candidate(routes, 525).get("selected_next_route")
      == "router_prompt_distribution_final_norm_delta_dimension_source_gate"
      and _route_candidate(routes, 526).get("selected_next_route")
      == "router_prompt_distribution_final_residual_delta_dimension_source_gate")
  feature = contract.get("feature_collection_contract", {})
  selection = contract.get("feature_selection_contract", {})
  model = contract.get("model_contract", {})
  correctness = contract.get("correctness_contract", {})
  contract_passes = (
      contract.get("schema_version")
      == "intel-qwen36-final-norm-sparse-state-feature-contract-v0"
      and feature.get("hidden_size") == 2048
      and feature.get("fit_diagnostic_case_ids") == FIT_IDS
      and feature.get("distribution_only") is True
      and feature.get("runtime_full_vector_host_read_allowed") is False
      and feature.get("runtime_selected_dimension_gather_only") is True
      and feature.get("maximum_selected_dimensions") == 16
      and selection.get("maximum_selected_dimensions") == 16
      and selection.get("validation_or_test_retuning_allowed") is False
      and model.get("selected_token_ids") == EXPECTED_TOKENS
      and model.get("token_intercept_parameters") == 16
      and model.get("shared_dimension_parameters") == 16
      and model.get("maximum_parameters") == 32
      and model.get("additional_model_variants_allowed") is False
      and model.get("runtime_native_oracle_allowed") is False
      and model.get("runtime_prompt_case_or_position_features_allowed") is False
      and model.get("full_vocab_host_rescan_allowed") is False
      and model.get("implementation_added_wall_us_per_token_max")
      == 20.525576981789345
      and correctness.get("max_kld") == 0.005
      and correctness.get("top1_rate_min") == 1.0
      and correctness.get(
          "grouped_fit_cross_validation_must_pass_before_validation") is True)
  blocked_ids = [
      row.get("id") for row in _load(args.corpus_contract).get("prompts", [])
      if isinstance(row, dict) and row.get("split") in ("validation", "test")
  ]
  blocked_collection_paths = [
      str(path) for case_id in blocked_ids
      for path in args.output_root.glob(
          f"seq574-state-conditioned-head-fit-collection-{case_id}-*/result.json")
  ]
  splits_untouched = len(blocked_ids) == 12 and not blocked_collection_paths
  checks = [
      {"name": "seq579_selected_observable_route_close_gate",
       "pass": predecessor_selects},
      {"name": "native_delta_scalar_and_whole_top8_models_are_closed",
       "pass": observable_board_is_closed},
      {"name": "distribution_path_already_owns_gpu_final_norm_vector",
       "pass": final_norm_source_exists},
      {"name": "seq525_526_localize_head_gap_to_final_norm_state",
       "pass": prior_attribution_exists},
      {"name": "sparse_final_norm_contract_is_exactly_16_plus_16_parameters",
       "pass": contract_passes},
      {"name": "validation_and_test_collection_paths_remain_absent",
       "pass": splits_untouched,
       "detail": {"blocked_case_ids": blocked_ids,
                  "found_paths": blocked_collection_paths}},
  ]
  required = all(bool(row["pass"]) for row in checks)
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "sequence": args.sequence,
      "inputs": {
          "routes": _rel(args.routes),
          "predecessor": _rel(args.predecessor),
          "oracle_gate": _rel(args.oracle_gate),
          "contract": _rel(args.contract),
          "corpus_contract": _rel(args.corpus_contract),
          "smoke_source": _rel(args.smoke_source),
          "seq525_gate": _rel(args.seq525_gate),
          "seq526_gate": _rel(args.seq526_gate),
      },
      "contract": contract,
      "checks": checks,
      "required_checks_passed": required,
      "observable_only_head_correction_closed": required,
      "final_norm_feature_source_allowed": required,
      "fit_collection_allowed": False,
      "validation_or_test_allowed": False,
      "runtime_source_allowed": False,
      "speedup_claims_allowed": False,
      "promotion_allowed": False,
      "disposition": (
          "close_top8_observable_models_select_sparse_final_norm_source"
          if required else "close_head_correction_without_successor"),
      "selected_next_route": (
          SELECTED_NEXT_ROUTE if required else
          "router_prompt_distribution_post_head_correction_route_control_gate"),
      "next_route_reason": (
          "All <=32-parameter GPU-top8 models are closed, but seq525/526 and "
          "the existing distribution path expose a genuinely new GPU final-"
          "norm state source. Source-gate only the fit diagnostic next; no fit "
          "row, validation/test, runtime source, or speed work is authorized."
          if required else
          "Repair route closure, untouched-split proof, source ownership, or "
          "the sparse-state contract before any further data collection."),
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
          "contract": metrics["inputs"]["contract"],
          "observable_only_head_correction_closed": metrics[
              "observable_only_head_correction_closed"],
          "selected_next_route": metrics["selected_next_route"],
          "fit_collection_allowed": False,
          "validation_or_test_allowed": False,
          "runtime_source_allowed": False,
          "speedup_claims_allowed": False,
      }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  failed = [row["name"] for row in metrics["checks"] if not row["pass"]]
  lines = [
      f"# Seq{metrics['sequence']} Observable Head-Correction Route Close",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- observable_only_head_correction_closed: "
      f"`{str(metrics['observable_only_head_correction_closed']).lower()}`",
      f"- failed_checks: `{failed}`",
      "",
      metrics["next_route_reason"],
      "",
      "No target command, fit row, validation case, or test case was used.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--sequence", type=int, default=580)
  parser.add_argument("--routes", type=Path,
                      default=ACTIVE / "routes-ledger.json")
  parser.add_argument(
      "--predecessor", type=Path,
      default=ROOT / "output/seq579-state-conditioned-head-top8-shape-fit-gate-20260710Tseq579Z/metrics.json")
  parser.add_argument(
      "--oracle-gate", type=Path,
      default=ROOT / "output/seq576-state-conditioned-head-post-fit-route-control-gate-20260710Tseq576Z/metrics.json")
  parser.add_argument(
      "--contract", type=Path,
      default=ACTIVE / "intel-qwen36-35b-a3b-gguf-q4km-final-norm-sparse-state-feature-contract-2026-07-10.json")
  parser.add_argument(
      "--corpus-contract", type=Path,
      default=ACTIVE / "intel-qwen36-35b-a3b-gguf-q4km-state-conditioned-head-correction-corpus-2026-07-10.json")
  parser.add_argument("--smoke-source", type=Path,
                      default=ROOT / "tools/intel-qwen36-r2-gpu-decode-smoke.py")
  parser.add_argument(
      "--seq525-gate", type=Path,
      default=ROOT / "output/seq525-head-token-pair-projection-source-gate-20260709Tseq525Z/metrics.json")
  parser.add_argument(
      "--seq526-gate", type=Path,
      default=ROOT / "output/seq526-final-norm-delta-dimension-source-gate-20260709Tseq526Z/metrics.json")
  parser.add_argument("--output-root", type=Path, default=ROOT / "output")
  parser.add_argument(
      "--out-dir", type=Path,
      default=ROOT / "output/seq580-state-conditioned-head-observable-route-close-gate-20260710Tseq580Z")
  args = parser.parse_args()
  metrics = compute(args)
  write_outputs(metrics, args.out_dir)
  print(json.dumps({
      "required_checks_passed": metrics["required_checks_passed"],
      "disposition": metrics["disposition"],
      "observable_only_head_correction_closed": metrics[
          "observable_only_head_correction_closed"],
      "final_norm_feature_source_allowed": metrics[
          "final_norm_feature_source_allowed"],
      "selected_next_route": metrics["selected_next_route"],
      "out_dir": _rel(args.out_dir),
  }, sort_keys=True))
  return 0 if metrics["required_checks_passed"] else 2


if __name__ == "__main__":
  raise SystemExit(main())
