#!/usr/bin/env python3
"""Classify four held-out sparse head-logit bias distribution rows."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = "intel-qwen36-sparse-head-logit-bias-holdout-gate-v0"
CURRENT_ROUTE = "router_prompt_distribution_sparse_head_logit_bias_holdout_gate"
SELECTED_NEXT_ROUTE = (
    "router_prompt_distribution_post_static_bias_holdout_route_control_gate"
)
EXPECTED_SPEC = (
    "22:0.28438186,25:0.39105892,264:0.79164315,"
    "421:-0.05538559,821:0.21105385,71093:0.01916504,"
    "248068:-0.228602415"
)
EXPECTED_BIASES = [
    {"token_id": 22, "bias": 0.28438186},
    {"token_id": 25, "bias": 0.39105892},
    {"token_id": 264, "bias": 0.79164315},
    {"token_id": 421, "bias": -0.05538559},
    {"token_id": 821, "bias": 0.21105385},
    {"token_id": 71093, "bias": 0.01916504},
    {"token_id": 248068, "bias": -0.228602415},
]
EXPECTED_MASK = (
    "0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,24,25,26,28,29,30,"
    "33,34,36,37,38"
)
CASES = (
    "router_instruction_003",
    "short_math_001",
    "short_factual_002",
    "short_transform_003",
)
KLD_MAX = 0.005
KLD_COMPARISON_EPSILON = 1e-7


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


def _distribution(row: dict[str, Any]) -> dict[str, Any]:
  smoke = row.get("smoke")
  if not isinstance(smoke, dict):
    return {}
  dist = smoke.get("distribution_ladder")
  return dist if isinstance(dist, dict) else {}


def _binary(row: dict[str, Any]) -> dict[str, Any]:
  target = row.get("target")
  cache = target.get("cache") if isinstance(target, dict) else None
  binary = cache.get("binary") if isinstance(cache, dict) else None
  return binary if isinstance(binary, dict) else {}


def _biases_match_float32(rows: Any) -> bool:
  if not isinstance(rows, list) or len(rows) != len(EXPECTED_BIASES):
    return False
  for actual, expected in zip(rows, EXPECTED_BIASES, strict=True):
    if not isinstance(actual, dict):
      return False
    if actual.get("token_id") != expected["token_id"]:
      return False
    value = actual.get("bias")
    if not isinstance(value, (int, float)) or not math.isclose(
        float(value), float(expected["bias"]), rel_tol=0.0, abs_tol=1e-7):
      return False
  return True


def _row_contract(row: dict[str, Any], case_id: str) -> bool:
  dist = _distribution(row)
  smoke = row.get("smoke") if isinstance(row.get("smoke"), dict) else {}
  return (
      row.get("case_id") == case_id
      and row.get("decode_tokens") == 8
      and row.get("distribution_ladder") is True
      and row.get("teacher_force_native_tokens") is True
      and row.get("sparse_head_logit_bias_spec") == EXPECTED_SPEC
      and row.get("sparse_head_logit_bias") == EXPECTED_BIASES
      and row.get("sparse_head_logit_bias_mode") == "static_token_id"
      and row.get("attention_front_output_projection_rowblock16_layers")
      == EXPECTED_MASK
      and row.get("cpu_layer_fallback_layers") == ""
      and row.get("router_cpu_fallback_layers") == ""
      and row.get("native_residual_realign_layers") == ""
      and row.get("input_rmsnorm_serial_reduction_layers") == []
      and row.get("linear_output_projection_cpu_order_layers") == []
      and smoke.get("sparse_head_logit_bias_enabled") is True
      and _biases_match_float32(smoke.get("sparse_head_logit_bias"))
      and _binary(row).get("ok") is True
      and dist.get("position_count") == 8)


def _baseline_contract(row: dict[str, Any], case_id: str) -> bool:
  dist = _distribution(row)
  return (
      row.get("case_id") == case_id
      and row.get("distribution_ladder") is True
      and row.get("attention_front_output_projection_rowblock16_layers")
      == EXPECTED_MASK
      and dist.get("position_count") == 8
      and dist.get("top1_rate") == 1)


def _summary(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
  candidate_dist = _distribution(candidate)
  baseline_dist = _distribution(baseline)
  candidate_kld = candidate_dist.get("max_kld")
  baseline_kld = baseline_dist.get("max_kld")
  delta = (
      float(candidate_kld) - float(baseline_kld)
      if isinstance(candidate_kld, (int, float))
      and isinstance(baseline_kld, (int, float)) else None)
  return {
      "case_id": candidate.get("case_id"),
      "binary_key": _binary(candidate).get("key"),
      "run_returncode": (
          candidate.get("target", {}).get("run", {}).get("returncode")
          if isinstance(candidate.get("target"), dict) else None),
      "position_count": candidate_dist.get("position_count"),
      "top1_rate": candidate_dist.get("top1_rate"),
      "max_kld": candidate_kld,
      "baseline_max_kld": baseline_kld,
      "kld_delta": delta,
      "no_material_regression": (
          isinstance(delta, (int, float))
          and float(delta) <= KLD_COMPARISON_EPSILON),
      "absolute_ruler_pass": (
          isinstance(candidate_kld, (int, float))
          and float(candidate_kld) <= KLD_MAX),
  }


def compute(args: argparse.Namespace) -> dict[str, Any]:
  routes = _load(args.routes)
  predecessor = _load(args.predecessor)
  candidate_paths = {
      "router_instruction_003": args.candidate_instruction,
      "short_math_001": args.candidate_short_math,
      "short_factual_002": args.candidate_short_factual,
      "short_transform_003": args.candidate_short_transform,
  }
  baseline_paths = {
      "router_instruction_003": args.baseline_instruction,
      "short_math_001": args.baseline_short_math,
      "short_factual_002": args.baseline_short_factual,
      "short_transform_003": args.baseline_short_transform,
  }
  candidates = {case_id: _load(path)
                for case_id, path in candidate_paths.items()}
  baselines = {case_id: _load(path)
               for case_id, path in baseline_paths.items()}
  summaries = {
      case_id: _summary(candidates[case_id], baselines[case_id])
      for case_id in CASES
  }
  predecessor_selects = (
      predecessor.get("required_checks_passed") is True
      and predecessor.get("holdout_allowed") is True
      and predecessor.get("selected_next_route") == CURRENT_ROUTE
      and predecessor.get("holdout_cases") == list(CASES)
      and _has_candidate(routes, 567, CURRENT_ROUTE)
      and _has_switch(
          routes, 567,
          "select_router_prompt_distribution_sparse_head_logit_bias_"
          "holdout_gate"))
  candidate_contracts = all(
      _row_contract(candidates[case_id], case_id) for case_id in CASES)
  baseline_contracts = all(
      _baseline_contract(baselines[case_id], case_id) for case_id in CASES)
  rows_valid = all(
      candidates[case_id].get("required_checks_passed") is True
      and summaries[case_id]["run_returncode"] == 0
      for case_id in CASES)
  top1_pass = all(summaries[case_id]["top1_rate"] == 1
                  for case_id in CASES)
  absolute_ruler_pass = all(summaries[case_id]["absolute_ruler_pass"]
                            for case_id in CASES)
  no_regression = all(summaries[case_id]["no_material_regression"]
                      for case_id in CASES)
  evidence_checks = [
      {"name": "seq567_selected_four_case_holdout_gate",
       "pass": predecessor_selects},
      {"name": "candidate_rows_match_exact_static_26mask_contract",
       "pass": candidate_contracts},
      {"name": "baseline_rows_match_uncorrected_26mask_contract",
       "pass": baseline_contracts},
      {"name": "all_target_builds_and_wrapper_checks_pass",
       "pass": rows_valid},
      {"name": "all_holdouts_preserve_top1", "pass": top1_pass},
      {"name": "all_holdouts_remain_under_absolute_kld_ruler",
       "pass": absolute_ruler_pass},
  ]
  required = all(bool(row["pass"]) for row in evidence_checks)
  holdout_passed = required and no_regression
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "sequence": args.sequence,
      "inputs": {
          "routes": _rel(args.routes),
          "predecessor": _rel(args.predecessor),
          "candidates": {case_id: _rel(path)
                         for case_id, path in candidate_paths.items()},
          "baselines": {case_id: _rel(path)
                        for case_id, path in baseline_paths.items()},
      },
      "thresholds": {
          "max_kld": KLD_MAX,
          "top1_rate": 1.0,
          "kld_comparison_epsilon": KLD_COMPARISON_EPSILON,
      },
      "holdouts": summaries,
      "checks": evidence_checks,
      "required_checks_passed": required,
      "no_kld_regression": no_regression,
      "holdout_contract_passed": holdout_passed,
      "failed_holdout_cases": [
          case_id for case_id in CASES
          if not summaries[case_id]["no_material_regression"]
      ],
      "speed_probe_allowed": False,
      "speedup_claims_allowed": False,
      "promotion_allowed": False,
      "table_retuning_allowed": False,
      "disposition": (
          "accept_sparse_head_logit_bias_holdouts"
          if holdout_passed else
          "reject_static_sparse_head_logit_bias_on_holdout_regression"),
      "selected_next_route": (
          "router_prompt_distribution_sparse_head_logit_bias_productization_gate"
          if holdout_passed else SELECTED_NEXT_ROUTE),
      "next_route_reason": (
          "All four unseen cases preserve top-1 and do not regress KLD; run a "
          "productization design gate before any speed row."
          if holdout_passed else
          "Short transform materially regresses from 0.003185390986 to "
          "0.004003973919 (+0.000818582933), violating the pre-registered "
          "no-regression contract. Close this exact table without tuning on "
          "holdouts and run no-token route control."),
  }


def write_outputs(metrics: dict[str, Any], out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  failed = [row["name"] for row in metrics["checks"] if not row["pass"]]
  lines = [
      f"# Seq{metrics['sequence']} Sparse Head-Logit Bias Holdout Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- holdout_contract_passed: `{str(metrics['holdout_contract_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- failed_holdout_cases: `{metrics['failed_holdout_cases']}`",
      f"- failed_evidence_checks: `{failed}`",
      "",
  ]
  for case_id in CASES:
    row = metrics["holdouts"][case_id]
    lines.append(
        f"- {case_id}: baseline `{row['baseline_max_kld']}`, candidate "
        f"`{row['max_kld']}`, delta `{row['kld_delta']}`, top1 "
        f"`{row['top1_rate']}`")
  lines.extend([
      "",
      metrics["next_route_reason"],
      "",
      "This is held-out correctness evidence. It is not speed evidence.",
      "",
  ])
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--sequence", type=int, default=568)
  parser.add_argument("--routes", type=Path,
                      default=ACTIVE / "routes-ledger.json")
  parser.add_argument(
      "--predecessor", type=Path,
      default=ROOT / "output/seq567-sparse-head-logit-bias-calibration-gate-20260710Tseq567Z/metrics.json")
  parser.add_argument(
      "--candidate-instruction", type=Path,
      default=ROOT / "output/seq568-sparse-head-logit-bias-holdout-router-instruction-20260710Tseq568Z/result.json")
  parser.add_argument(
      "--candidate-short-math", type=Path,
      default=ROOT / "output/seq568-sparse-head-logit-bias-holdout-short-math-20260710Tseq568Z/result.json")
  parser.add_argument(
      "--candidate-short-factual", type=Path,
      default=ROOT / "output/seq568-sparse-head-logit-bias-holdout-short-factual-20260710Tseq568Z/result.json")
  parser.add_argument(
      "--candidate-short-transform", type=Path,
      default=ROOT / "output/seq568-sparse-head-logit-bias-holdout-short-transform-20260710Tseq568Z/result.json")
  seq222 = ROOT / "output/r2-gpu-acceptance-matrix-router-both-rowblock16-26mask-20260708Tseq222Z/runs"
  parser.add_argument(
      "--baseline-instruction", type=Path,
      default=seq222 / "router_instruction_003-distribution/result.json")
  parser.add_argument(
      "--baseline-short-math", type=Path,
      default=ROOT / "output/r2-gpu-attention-front-rowblock16-26mask-noqueue-distribution-20260708Tseq221Z/result.json")
  parser.add_argument(
      "--baseline-short-factual", type=Path,
      default=ROOT / "output/r2-gpu-acceptance-matrix-short-factual-distribution-rowblock16-23mask-plus24-26-20260708Tseq217Z/runs/short_factual_002-distribution/result.json")
  parser.add_argument(
      "--baseline-short-transform", type=Path,
      default=ROOT / "output/seq568-sparse-head-logit-bias-holdout-baseline-short-transform-20260710Tseq568Z/result.json")
  parser.add_argument(
      "--out-dir", type=Path,
      default=ROOT / "output/seq568-sparse-head-logit-bias-holdout-gate-20260710Tseq568Z")
  args = parser.parse_args()
  metrics = compute(args)
  write_outputs(metrics, args.out_dir)
  print(json.dumps({
      "required_checks_passed": metrics["required_checks_passed"],
      "holdout_contract_passed": metrics["holdout_contract_passed"],
      "disposition": metrics["disposition"],
      "failed_holdout_cases": metrics["failed_holdout_cases"],
      "selected_next_route": metrics["selected_next_route"],
      "out_dir": _rel(args.out_dir),
  }, sort_keys=True))
  return 0 if metrics["required_checks_passed"] else 2


if __name__ == "__main__":
  raise SystemExit(main())
