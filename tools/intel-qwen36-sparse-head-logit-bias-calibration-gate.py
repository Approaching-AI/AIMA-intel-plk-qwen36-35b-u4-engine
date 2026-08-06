#!/usr/bin/env python3
"""Classify the paired math/code sparse head-logit bias calibration rows."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = "intel-qwen36-sparse-head-logit-bias-calibration-gate-v0"
CURRENT_ROUTE = (
    "router_prompt_distribution_sparse_head_logit_bias_calibration_pair_gate"
)
SELECTED_NEXT_ROUTE = (
    "router_prompt_distribution_sparse_head_logit_bias_holdout_gate"
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
CASES = ("router_math_reason_001", "router_code_reason_002")
KLD_MAX = 0.005


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


def _summary(row: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
  dist = _distribution(row)
  baseline_dist = _distribution(baseline)
  smoke = row.get("smoke") if isinstance(row.get("smoke"), dict) else {}
  max_kld = dist.get("max_kld")
  baseline_kld = baseline_dist.get("max_kld")
  return {
      "case_id": row.get("case_id"),
      "required_checks_passed": row.get("required_checks_passed"),
      "run_returncode": (
          row.get("target", {}).get("run", {}).get("returncode")
          if isinstance(row.get("target"), dict) else None),
      "binary_key": _binary(row).get("key"),
      "binary_ok": _binary(row).get("ok"),
      "position_count": dist.get("position_count"),
      "top1_rate": dist.get("top1_rate"),
      "max_kld": max_kld,
      "baseline_max_kld": baseline_kld,
      "kld_reduction": (
          float(baseline_kld) - float(max_kld)
          if isinstance(max_kld, (int, float))
          and isinstance(baseline_kld, (int, float)) else None),
      "kld_ratio_vs_baseline": (
          float(max_kld) / float(baseline_kld)
          if isinstance(max_kld, (int, float))
          and isinstance(baseline_kld, (int, float))
          and float(baseline_kld) > 0.0 else None),
      "bias_enabled": smoke.get("sparse_head_logit_bias_enabled"),
      "runtime_biases": smoke.get("sparse_head_logit_bias"),
  }


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


def compute(args: argparse.Namespace) -> dict[str, Any]:
  routes = _load(args.routes)
  predecessor = _load(args.predecessor)
  candidates = {
      "router_math_reason_001": _load(args.candidate_math),
      "router_code_reason_002": _load(args.candidate_code),
  }
  baselines = {
      "router_math_reason_001": _load(args.baseline_math),
      "router_code_reason_002": _load(args.baseline_code),
  }
  summaries = {
      case_id: _summary(candidates[case_id], baselines[case_id])
      for case_id in CASES
  }
  predecessor_selects = (
      predecessor.get("required_checks_passed") is True
      and predecessor.get("calibration_pair_allowed") is True
      and predecessor.get("selected_next_route") == CURRENT_ROUTE
      and _has_candidate(routes, 566, CURRENT_ROUTE)
      and _has_switch(
          routes, 566,
          "select_router_prompt_distribution_sparse_head_logit_bias_"
          "calibration_pair_gate"))
  contracts_pass = all(
      _row_contract(candidates[case_id], case_id) for case_id in CASES)
  top1_pass = all(
      summaries[case_id]["top1_rate"] == 1 for case_id in CASES)
  kld_pass = all(
      isinstance(summaries[case_id]["max_kld"], (int, float))
      and float(summaries[case_id]["max_kld"]) <= KLD_MAX
      for case_id in CASES)
  improves_baseline = all(
      isinstance(summaries[case_id]["kld_reduction"], (int, float))
      and float(summaries[case_id]["kld_reduction"]) > 0.0
      for case_id in CASES)
  checks = [
      {"name": "seq566_selected_calibration_pair_gate",
       "pass": predecessor_selects},
      {"name": "math_code_rows_match_exact_static_26mask_contract",
       "pass": contracts_pass},
      {"name": "math_code_target_builds_and_wrapper_checks_pass",
       "pass": all(
           candidates[case_id].get("required_checks_passed") is True
           and summaries[case_id]["run_returncode"] == 0
           for case_id in CASES)},
      {"name": "math_code_preserve_all_top1_tokens", "pass": top1_pass},
      {"name": "math_code_clear_kld_ruler", "pass": kld_pass},
      {"name": "math_code_improve_uncorrected_baselines",
       "pass": improves_baseline},
  ]
  required = all(bool(row["pass"]) for row in checks)
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "sequence": args.sequence,
      "inputs": {
          "routes": _rel(args.routes),
          "predecessor": _rel(args.predecessor),
          "candidate_math": _rel(args.candidate_math),
          "candidate_code": _rel(args.candidate_code),
          "baseline_math": _rel(args.baseline_math),
          "baseline_code": _rel(args.baseline_code),
      },
      "thresholds": {"max_kld": KLD_MAX, "top1_rate": 1.0},
      "calibration": summaries,
      "checks": checks,
      "required_checks_passed": required,
      "holdout_allowed": required,
      "holdout_cases": [
          "router_instruction_003", "short_math_001",
          "short_factual_002", "short_transform_003",
      ],
      "speedup_claims_allowed": False,
      "promotion_allowed": False,
      "disposition": (
          "accept_sparse_head_logit_bias_calibration_pair"
          if required else "reject_sparse_head_logit_bias_calibration_pair"),
      "selected_next_route": SELECTED_NEXT_ROUTE if required else CURRENT_ROUTE,
      "next_route_reason": (
          "The fixed table clears math/code at max KLD 0.002518399466 / "
          "0.003467842414 with top-1 8/8 on both, versus uncorrected "
          "0.02933664306 / 0.01663157594. Run the four pre-registered held-out "
          "distribution cases; require no top-1 change and no KLD regression."
          if required else
          "Close the static bias route; do not tune the table or run holdouts."),
  }


def write_outputs(metrics: dict[str, Any], out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  failed = [row["name"] for row in metrics["checks"] if not row["pass"]]
  math_row = metrics["calibration"]["router_math_reason_001"]
  code_row = metrics["calibration"]["router_code_reason_002"]
  lines = [
      f"# Seq{metrics['sequence']} Sparse Head-Logit Bias Calibration Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- math max KLD / top1: `{math_row['max_kld']}` / `{math_row['top1_rate']}`",
      f"- code max KLD / top1: `{code_row['max_kld']}` / `{code_row['top1_rate']}`",
      f"- holdout_allowed: `{str(metrics['holdout_allowed']).lower()}`",
      f"- failed_checks: `{failed}`",
      "",
      metrics["next_route_reason"],
      "",
      "This is calibration evidence only. It is not holdout or speed evidence.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--sequence", type=int, default=567)
  parser.add_argument("--routes", type=Path,
                      default=ACTIVE / "routes-ledger.json")
  parser.add_argument(
      "--predecessor", type=Path,
      default=ROOT / "output/seq566-sparse-head-logit-bias-target-compile-gate-20260710Tseq566Z/metrics.json")
  parser.add_argument(
      "--candidate-math", type=Path,
      default=ROOT / "output/seq567-sparse-head-logit-bias-calibration-math-20260710Tseq567Z/result.json")
  parser.add_argument(
      "--candidate-code", type=Path,
      default=ROOT / "output/seq567-sparse-head-logit-bias-calibration-code-20260710Tseq567Z/result.json")
  seq222 = ROOT / "output/r2-gpu-acceptance-matrix-router-both-rowblock16-26mask-20260708Tseq222Z/runs"
  parser.add_argument(
      "--baseline-math", type=Path,
      default=seq222 / "router_math_reason_001-distribution/result.json")
  parser.add_argument(
      "--baseline-code", type=Path,
      default=seq222 / "router_code_reason_002-distribution/result.json")
  parser.add_argument(
      "--out-dir", type=Path,
      default=ROOT / "output/seq567-sparse-head-logit-bias-calibration-gate-20260710Tseq567Z")
  args = parser.parse_args()
  metrics = compute(args)
  write_outputs(metrics, args.out_dir)
  print(json.dumps({
      "required_checks_passed": metrics["required_checks_passed"],
      "disposition": metrics["disposition"],
      "holdout_allowed": metrics["holdout_allowed"],
      "selected_next_route": metrics["selected_next_route"],
      "out_dir": _rel(args.out_dir),
  }, sort_keys=True))
  return 0 if metrics["required_checks_passed"] else 2


if __name__ == "__main__":
  raise SystemExit(main())
