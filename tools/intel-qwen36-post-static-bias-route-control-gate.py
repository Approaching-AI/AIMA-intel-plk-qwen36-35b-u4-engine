#!/usr/bin/env python3
"""Select the route after static sparse logit bias fails held-out data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = "intel-qwen36-post-static-bias-route-control-gate-v0"
CURRENT_ROUTE = (
    "router_prompt_distribution_post_static_bias_holdout_route_control_gate"
)
SELECTED_NEXT_ROUTE = (
    "router_prompt_distribution_state_conditioned_head_correction_"
    "corpus_contract_gate"
)
TARGET_TOKEN_ID = 248068


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


def _rejected(rejected: dict[str, Any], route: str) -> bool:
  return any(
      isinstance(row, dict) and row.get("route") == route
      for row in rejected.get("rejected", []))


def _calibration_observations(route_control: dict[str, Any]) -> list[dict[str, Any]]:
  calibration = route_control.get("calibration")
  biases = calibration.get("biases") if isinstance(calibration, dict) else None
  if not isinstance(biases, list):
    return []
  for row in biases:
    if isinstance(row, dict) and row.get("token_id") == TARGET_TOKEN_ID:
      observations = row.get("observations")
      return observations if isinstance(observations, list) else []
  return []


def _distribution(row: dict[str, Any]) -> dict[str, Any]:
  smoke = row.get("smoke")
  if not isinstance(smoke, dict):
    return {}
  dist = smoke.get("distribution_ladder")
  return dist if isinstance(dist, dict) else {}


def _contributor_delta(row: dict[str, Any], token_index: int,
                       token_id: int) -> float | None:
  steps = _distribution(row).get("steps")
  if not isinstance(steps, list) or token_index >= len(steps):
    return None
  step = steps[token_index]
  if not isinstance(step, dict):
    return None
  contributors = []
  for key in ("top_kld_contributors", "top_negative_kld_contributors"):
    value = step.get(key)
    if isinstance(value, list):
      contributors.extend(value)
  for contributor in contributors:
    if not isinstance(contributor, dict) or contributor.get("token_id") != token_id:
      continue
    native = contributor.get("native_logit")
    gpu = contributor.get("gpu_logit")
    if isinstance(native, (int, float)) and isinstance(gpu, (int, float)):
      return float(native) - float(gpu)
  return None


def compute(args: argparse.Namespace) -> dict[str, Any]:
  routes = _load(args.routes)
  rejected = _load(args.rejected)
  predecessor = _load(args.predecessor)
  static_route = _load(args.static_route_control)
  transform_baseline = _load(args.transform_baseline)
  budget = _load(args.budget_gate)
  calibration = _calibration_observations(static_route)
  calibration_deltas = [
      float(row["native_minus_gpu_logit"])
      for row in calibration
      if isinstance(row, dict)
      and isinstance(row.get("native_minus_gpu_logit"), (int, float))
  ]
  transform_deltas = [
      _contributor_delta(transform_baseline, token_index, TARGET_TOKEN_ID)
      for token_index in (0, 4)
  ]
  sign_flip = (
      len(calibration_deltas) == 2
      and all(value < 0.0 for value in calibration_deltas)
      and all(isinstance(value, float) and value > 0.0
              for value in transform_deltas))
  budget_row = budget.get("kill_number")
  floor_headroom_us = (
      budget_row.get("floor_headroom_us")
      if isinstance(budget_row, dict) else None)
  routes_select = (
      predecessor.get("required_checks_passed") is True
      and predecessor.get("holdout_contract_passed") is False
      and predecessor.get("table_retuning_allowed") is False
      and predecessor.get("selected_next_route") == CURRENT_ROUTE
      and _has_candidate(routes, 568, CURRENT_ROUTE)
      and _has_switch(
          routes, 568,
          "select_router_prompt_distribution_post_static_bias_holdout_"
          "route_control_gate"))
  closed_input_independent = (
      _rejected(rejected, "router_prompt_distribution_affine_logit_calibration")
      and _rejected(
          rejected, "router_prompt_distribution_static_sparse_head_logit_bias"))
  source = args.decode_source.read_text(encoding="utf-8")
  existing_observables = (
      "DecodeTopKFromLogits" in source
      and "top_kld_contributors" in source
      and "native_logit" in source
      and "gpu_logit" in source)
  checks = [
      {"name": "seq568_selected_post_holdout_route_control",
       "pass": routes_select},
      {"name": "input_independent_logit_calibration_routes_are_closed",
       "pass": closed_input_independent},
      {"name": "same_token_correction_changes_sign_across_states",
       "pass": sign_flip,
       "detail": {
           "token_id": TARGET_TOKEN_ID,
           "calibration_native_minus_gpu": calibration_deltas,
           "transform_native_minus_gpu": transform_deltas,
       }},
      {"name": "distribution_lane_already_records_training_observables",
       "pass": existing_observables},
      {"name": "floor_headroom_is_positive_and_bounded",
       "pass": isinstance(floor_headroom_us, (int, float))
               and 0.0 < float(floor_headroom_us) < 1000.0,
       "detail": {"floor_headroom_us_per_token": floor_headroom_us}},
  ]
  required = all(bool(row["pass"]) for row in checks)
  corpus_contract = {
      "new_prompt_count": 24,
      "domains": [
          "arithmetic", "code", "instruction", "factual",
          "transformation", "structured_extraction",
      ],
      "prompts_per_domain": 4,
      "split": {"fit": 12, "validation": 6, "test": 6},
      "split_locked_before_decode": True,
      "existing_acceptance_prompts_excluded_from_fit": True,
      "prompt_case_position_features_allowed": False,
      "runtime_native_oracle_allowed": False,
      "allowed_runtime_features": [
          "gpu_top8_token_ids", "gpu_top8_logits", "top1_margin",
      ],
      "candidate_token_selection": (
          "fit-split aggregate absolute KLD contributors only, capped at 16"
      ),
      "model_class": (
          "per-token ridge affine correction over intercept and GPU logit "
          "gap-to-top1, applied only while the token is in GPU top8"
      ),
      "maximum_token_ids": 16,
      "maximum_parameters": 32,
      "validation_contract": (
          "top1 unchanged; max KLD <= 0.005; no case KLD regression beyond 1e-7"
      ),
      "acceptance_recheck_after_fresh_test": [
          "short_math_001", "short_factual_002", "short_transform_003",
          "router_math_reason_001", "router_code_reason_002",
          "router_instruction_003",
      ],
      "implementation_added_wall_us_per_token_max": (
          float(floor_headroom_us) * 0.10
          if isinstance(floor_headroom_us, (int, float)) else None
      ),
      "full_vocab_host_rescan_in_speed_lane_allowed": False,
  }
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "sequence": args.sequence,
      "inputs": {
          "routes": _rel(args.routes),
          "rejected": _rel(args.rejected),
          "predecessor": _rel(args.predecessor),
          "static_route_control": _rel(args.static_route_control),
          "transform_baseline": _rel(args.transform_baseline),
          "budget_gate": _rel(args.budget_gate),
          "decode_source": _rel(args.decode_source),
      },
      "state_dependence": {
          "token_id": TARGET_TOKEN_ID,
          "calibration_native_minus_gpu": calibration_deltas,
          "transform_native_minus_gpu": transform_deltas,
          "sign_flip": sign_flip,
      },
      "corpus_contract": corpus_contract,
      "checks": checks,
      "required_checks_passed": required,
      "corpus_contract_gate_allowed": required,
      "decode_row_allowed": False,
      "source_change_allowed": False,
      "speedup_claims_allowed": False,
      "promotion_allowed": False,
      "disposition": (
          "select_fresh_corpus_state_conditioned_head_correction_contract"
          if required else "block_post_static_bias_route_control"),
      "selected_next_route": SELECTED_NEXT_ROUTE if required else CURRENT_ROUTE,
      "next_route_reason": (
          "Token 248068 needs opposite-sign corrections across prompt states, "
          "closing input-independent offsets. The next bounded route is not a "
          "new table or target row: lock a fresh 24-prompt fit/validation/test "
          "corpus and a top8-only, at-most-32-parameter state-conditioned "
          "model before observing any new decode results."
          if required else
          "Resolve the route ledger, closure evidence, sign-flip attribution, "
          "or budget input before another correction route."),
  }


def write_outputs(metrics: dict[str, Any], out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  failed = [row["name"] for row in metrics["checks"] if not row["pass"]]
  state = metrics["state_dependence"]
  lines = [
      f"# Seq{metrics['sequence']} Post-Static-Bias Route Control",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- token {state['token_id']} calibration deltas: `{state['calibration_native_minus_gpu']}`",
      f"- token {state['token_id']} transform deltas: `{state['transform_native_minus_gpu']}`",
      f"- sign flip: `{str(state['sign_flip']).lower()}`",
      f"- failed_checks: `{failed}`",
      "",
      metrics["next_route_reason"],
      "",
      "This is no-token route-control evidence. It is not a correction or speed claim.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--sequence", type=int, default=569)
  parser.add_argument("--routes", type=Path,
                      default=ACTIVE / "routes-ledger.json")
  parser.add_argument("--rejected", type=Path,
                      default=ACTIVE / "rejected-routes.json")
  parser.add_argument(
      "--predecessor", type=Path,
      default=ROOT / "output/seq568-sparse-head-logit-bias-holdout-gate-20260710Tseq568Z/metrics.json")
  parser.add_argument(
      "--static-route-control", type=Path,
      default=ROOT / "output/seq564-non-arithmetic-product-source-route-control-gate-20260710Tseq564Z/metrics.json")
  parser.add_argument(
      "--transform-baseline", type=Path,
      default=ROOT / "output/seq568-sparse-head-logit-bias-holdout-baseline-short-transform-20260710Tseq568Z/result.json")
  parser.add_argument(
      "--budget-gate", type=Path,
      default=ROOT / "output/seq563-all-linear-norm-to-projection-parity-feasibility-gate-20260710Tseq563Z/metrics.json")
  parser.add_argument(
      "--decode-source", type=Path,
      default=ROOT / "tools/intel-qwen36-r2-gpu-decode-smoke.py")
  parser.add_argument(
      "--out-dir", type=Path,
      default=ROOT / "output/seq569-post-static-bias-route-control-gate-20260710Tseq569Z")
  args = parser.parse_args()
  metrics = compute(args)
  write_outputs(metrics, args.out_dir)
  print(json.dumps({
      "required_checks_passed": metrics["required_checks_passed"],
      "disposition": metrics["disposition"],
      "corpus_contract_gate_allowed": metrics["corpus_contract_gate_allowed"],
      "selected_next_route": metrics["selected_next_route"],
      "out_dir": _rel(args.out_dir),
  }, sort_keys=True))
  return 0 if metrics["required_checks_passed"] else 2


if __name__ == "__main__":
  raise SystemExit(main())
