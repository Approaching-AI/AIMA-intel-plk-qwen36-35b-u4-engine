#!/usr/bin/env python3
"""Classify selected-token and all-top8 oracle ceilings after fit rejection."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = (
    "intel-qwen36-state-conditioned-head-post-fit-route-control-gate-v0"
)
CURRENT_ROUTE = (
    "router_prompt_distribution_state_conditioned_head_correction_"
    "post_fit_route_control_gate"
)
FEATURE_REFRAME_ROUTE = (
    "router_prompt_distribution_state_conditioned_head_correction_"
    "feature_reframe_gate"
)
TOKEN_COVERAGE_ROUTE = (
    "router_prompt_distribution_state_conditioned_head_correction_"
    "bounded_token_coverage_feasibility_gate"
)
TAIL_ATTRIBUTION_ROUTE = (
    "router_prompt_distribution_tail_conditional_source_attribution_gate"
)
KLD_MAX = 0.005
TOP1_RATE_MIN = 1.0
KLD_REGRESSION_EPSILON = 1e-7


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


def _distribution(result: dict[str, Any]) -> dict[str, Any]:
  smoke = result.get("smoke")
  if not isinstance(smoke, dict):
    return {}
  value = smoke.get("distribution_ladder")
  return value if isinstance(value, dict) else {}


def _oracle_step(step: dict[str, Any], allowed_tokens: set[int] | None) -> dict[str, Any]:
  rows = step.get("gpu_top8_fit_observables")
  values = (
      step.get("kld"), step.get("gpu_logsumexp"),
      step.get("native_logsumexp"))
  if (not isinstance(rows, list) or len(rows) != 8
      or not all(isinstance(value, (int, float)) for value in values)):
    return {"valid": False}
  baseline_kld = float(step["kld"])
  gpu_lse = float(step["gpu_logsumexp"])
  native_lse = float(step["native_logsumexp"])
  selected = [
      row for row in rows
      if isinstance(row, dict)
      and isinstance(row.get("token_id"), int)
      and (allowed_tokens is None or int(row["token_id"]) in allowed_tokens)
  ]
  corrections: dict[int, float] = {}
  if selected:
    gpu_selected_mass = sum(
        math.exp(float(row["gpu_logit"]) - gpu_lse) for row in selected)
    native_selected_mass = sum(
        math.exp(float(row["native_logit"]) - native_lse) for row in selected)
    native_selected_scale_at_gpu_lse = sum(
        math.exp(float(row["native_logit"]) - gpu_lse) for row in selected)
    if (gpu_selected_mass >= 1.0 or native_selected_mass >= 1.0
        or native_selected_scale_at_gpu_lse <= 0.0):
      return {"valid": False}
    target_odds_mass = (
        native_selected_mass / (1.0 - native_selected_mass)
        * (1.0 - gpu_selected_mass))
    common_offset = math.log(
        target_odds_mass / native_selected_scale_at_gpu_lse)
    for row in selected:
      token_id = int(row["token_id"])
      corrections[token_id] = (
          float(row["native_logit"]) - float(row["gpu_logit"])
          + common_offset)
  mass_delta = 0.0
  native_weighted_correction = 0.0
  corrected_logits: list[tuple[int, int, float]] = []
  for row in rows:
    if not isinstance(row, dict):
      return {"valid": False}
    token_id = int(row["token_id"])
    gpu_logit = float(row["gpu_logit"])
    correction = corrections.get(token_id, 0.0)
    if token_id in corrections:
      mass_delta += math.exp(gpu_logit - gpu_lse) * math.expm1(correction)
      native_weighted_correction += (
          math.exp(float(row["native_logit"]) - native_lse) * correction)
    corrected_logits.append(
        (int(row["rank"]), token_id, gpu_logit + correction))
  if not math.isfinite(mass_delta) or mass_delta <= -1.0:
    return {"valid": False}
  corrected_kld = (
      baseline_kld + math.log1p(mass_delta) - native_weighted_correction)
  if corrected_kld < 0.0 and corrected_kld >= -1e-9:
    corrected_kld = 0.0
  winner = max(corrected_logits, key=lambda row: (row[2], -row[0]))
  outside_upper_bound = float(rows[-1]["gpu_logit"])
  top1_proven = (
      winner[1] == step.get("native_top1_id")
      and winner[2] + 1e-7 >= outside_upper_bound)
  return {
      "valid": math.isfinite(corrected_kld) and corrected_kld >= 0.0,
      "baseline_kld": baseline_kld,
      "corrected_kld": corrected_kld,
      "kld_delta": corrected_kld - baseline_kld,
      "corrected_top1_proven": top1_proven,
      "corrected_top1_id": winner[1],
      "native_top1_id": step.get("native_top1_id"),
      "corrected_token_count": len(corrections),
  }


def _evaluate_oracle(
    results: dict[str, dict[str, Any]], allowed_tokens: set[int] | None,
) -> dict[str, Any]:
  cases: dict[str, dict[str, Any]] = {}
  all_steps: list[dict[str, Any]] = []
  for case_id, result in results.items():
    steps = [
        _oracle_step(step, allowed_tokens)
        for step in _distribution(result).get("steps", [])
        if isinstance(step, dict)
    ]
    valid = len(steps) == 8 and all(row.get("valid") for row in steps)
    baseline_max = max(
        (float(row["baseline_kld"]) for row in steps), default=math.inf)
    corrected_max = max(
        (float(row["corrected_kld"]) for row in steps), default=math.inf)
    corrected_mean = (
        sum(float(row["corrected_kld"]) for row in steps) / len(steps)
        if steps else math.inf)
    top1_count = sum(bool(row.get("corrected_top1_proven")) for row in steps)
    top1_rate = top1_count / len(steps) if steps else 0.0
    no_regression = corrected_max <= baseline_max + KLD_REGRESSION_EPSILON
    contract_passed = (
        valid and corrected_max <= KLD_MAX
        and top1_rate >= TOP1_RATE_MIN and no_regression)
    cases[case_id] = {
        "valid": valid,
        "position_count": len(steps),
        "baseline_max_kld": baseline_max,
        "corrected_max_kld": corrected_max,
        "corrected_mean_kld": corrected_mean,
        "max_kld_delta": corrected_max - baseline_max,
        "corrected_top1_match_count": top1_count,
        "corrected_top1_rate": top1_rate,
        "no_kld_regression": no_regression,
        "contract_passed": contract_passed,
    }
    all_steps.extend(steps)
  valid_steps = [row for row in all_steps if row.get("valid")]
  return {
      "case_count": len(cases),
      "position_count": len(all_steps),
      "valid_position_count": len(valid_steps),
      "corrected_max_kld": max(
          (float(row["corrected_kld"]) for row in valid_steps),
          default=math.inf),
      "corrected_mean_kld": (
          sum(float(row["corrected_kld"]) for row in valid_steps)
          / len(valid_steps) if valid_steps else math.inf),
      "corrected_top1_rate": (
          sum(bool(row["corrected_top1_proven"]) for row in valid_steps)
          / len(valid_steps) if valid_steps else 0.0),
      "failed_case_count": sum(
          not row["contract_passed"] for row in cases.values()),
      "failed_cases": [
          case_id for case_id, row in cases.items()
          if not row["contract_passed"]
      ],
      "contract_passed": bool(cases) and all(
          row["contract_passed"] for row in cases.values()),
      "cases": cases,
  }


def compute(args: argparse.Namespace) -> dict[str, Any]:
  routes = _load(args.routes)
  predecessor = _load(args.predecessor)
  corpus = _load(args.corpus_contract)
  fit_ids = [
      str(row["id"]) for row in corpus.get("prompts", [])
      if isinstance(row, dict) and row.get("split") == "fit"
  ]
  result_paths = predecessor.get("inputs", {}).get("fit_result_paths", {})
  loaded = {
      case_id: _load(ROOT / path)
      for case_id, path in result_paths.items()
      if isinstance(case_id, str) and isinstance(path, str)
  }
  results = {case_id: loaded[case_id] for case_id in fit_ids
             if case_id in loaded}
  parameter_rows = predecessor.get("model", {}).get("parameters", [])
  selected_tokens = {
      int(row["token_id"]) for row in parameter_rows
      if isinstance(row, dict) and isinstance(row.get("token_id"), int)
  }
  all_top8_tokens = {
      int(row["token_id"])
      for result in results.values()
      for step in _distribution(result).get("steps", [])
      if isinstance(step, dict)
      for row in step.get("gpu_top8_fit_observables", [])
      if isinstance(row, dict) and isinstance(row.get("token_id"), int)
  }
  predecessor_selects = (
      predecessor.get("required_checks_passed") is True
      and predecessor.get("fit_feasibility_passed") is False
      and predecessor.get("validation_allowed") is False
      and predecessor.get("selected_next_route") == CURRENT_ROUTE
      and _has_candidate(routes, 575, CURRENT_ROUTE)
      and _has_switch(
          routes, 575,
          "select_router_prompt_distribution_state_conditioned_head_"
          "correction_post_fit_route_control_gate"))
  data_is_fit_only = (
      len(results) == 12 and list(results) == fit_ids
      and set(loaded) == set(fit_ids))
  selected_oracle = _evaluate_oracle(results, selected_tokens)
  all_top8_oracle = _evaluate_oracle(results, None)
  ceiling_math_passes = (
      selected_oracle["case_count"] == 12
      and selected_oracle["position_count"] == 96
      and selected_oracle["valid_position_count"] == 96
      and all_top8_oracle["case_count"] == 12
      and all_top8_oracle["position_count"] == 96
      and all_top8_oracle["valid_position_count"] == 96)
  shape_passes = len(selected_tokens) == 16 and len(all_top8_tokens) >= 16
  checks = [
      {"name": "seq575_selected_no_target_post_fit_route_control",
       "pass": predecessor_selects},
      {"name": "only_the_12_locked_fit_cases_are_loaded",
       "pass": data_is_fit_only,
       "detail": {"fit_ids": fit_ids, "loaded_ids": list(results)}},
      {"name": "selected16_and_all_top8_oracle_shapes_are_bounded",
       "pass": shape_passes,
       "detail": {"selected_token_count": len(selected_tokens),
                  "unique_fit_top8_token_count": len(all_top8_tokens)}},
      {"name": "mass_matched_oracle_math_covers_all_96_steps",
       "pass": ceiling_math_passes},
  ]
  required = all(bool(row["pass"]) for row in checks)
  selected_passes = required and selected_oracle["contract_passed"]
  all_top8_passes = required and all_top8_oracle["contract_passed"]
  if selected_passes:
    disposition = "close_ridge_affine_select_feature_reframe"
    selected_next_route = FEATURE_REFRAME_ROUTE
    reason = (
        "The exact selected-16 oracle clears the fit corpus, so token coverage "
        "is sufficient and the frozen affine feature class is the blocker. "
        "Reframe features under a new no-target contract before validation.")
  elif all_top8_passes:
    disposition = "close_selected16_select_token_coverage_feasibility"
    selected_next_route = TOKEN_COVERAGE_ROUTE
    reason = (
        "The selected-16 oracle fails while the exact all-top8 oracle clears "
        "the fit corpus. Quantify whether broader token coverage can remain "
        "inside the 32-parameter and implementation budgets before new data.")
  else:
    disposition = "close_top8_head_correction_select_tail_attribution"
    selected_next_route = TAIL_ATTRIBUTION_ROUTE
    reason = (
        "Even the per-step probability-mass-matched all-top8 native-logit "
        "oracle fails the fit contract. Close top8-only head correction and "
        "attribute the irreducible tail-conditional distribution error without "
        "opening validation or runtime source work.")
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "sequence": args.sequence,
      "inputs": {
          "routes": _rel(args.routes),
          "predecessor": _rel(args.predecessor),
          "corpus_contract": _rel(args.corpus_contract),
          "fit_result_paths": result_paths,
      },
      "thresholds": {
          "max_kld": KLD_MAX,
          "top1_rate_min": TOP1_RATE_MIN,
          "per_case_kld_regression_epsilon": KLD_REGRESSION_EPSILON,
      },
      "selected_token_ids": sorted(selected_tokens),
      "unique_fit_top8_token_count": len(all_top8_tokens),
      "selected16_mass_matched_oracle": selected_oracle,
      "all_top8_mass_matched_oracle": all_top8_oracle,
      "checks": checks,
      "required_checks_passed": required,
      "selected16_oracle_contract_passed": selected_passes,
      "all_top8_oracle_contract_passed": all_top8_passes,
      "validation_or_test_allowed": False,
      "runtime_source_allowed": False,
      "speedup_claims_allowed": False,
      "promotion_allowed": False,
      "disposition": disposition,
      "selected_next_route": selected_next_route,
      "next_route_reason": reason,
  }


def write_outputs(metrics: dict[str, Any], out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  manifest = {
      "schema_version": metrics["schema_version"],
      "workstream": metrics["workstream"],
      "tool": _rel(Path(__file__)),
      "inputs": metrics["inputs"],
      "selected_token_ids": metrics["selected_token_ids"],
      "unique_fit_top8_token_count": metrics["unique_fit_top8_token_count"],
      "selected16_oracle_contract_passed": metrics[
          "selected16_oracle_contract_passed"],
      "all_top8_oracle_contract_passed": metrics[
          "all_top8_oracle_contract_passed"],
      "selected_next_route": metrics["selected_next_route"],
      "validation_or_test_allowed": False,
      "runtime_source_allowed": False,
      "speedup_claims_allowed": False,
  }
  (out_dir / "manifest.json").write_text(
      json.dumps(manifest, indent=2, sort_keys=True) + "\n",
      encoding="utf-8")
  failed = [row["name"] for row in metrics["checks"] if not row["pass"]]
  selected = metrics["selected16_mass_matched_oracle"]
  all_top8 = metrics["all_top8_mass_matched_oracle"]
  lines = [
      f"# Seq{metrics['sequence']} Post-Fit Head-Correction Route Control",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- selected16 max KLD / top1 / failed cases: "
      f"`{selected['corrected_max_kld']}` / "
      f"`{selected['corrected_top1_rate']}` / `{selected['failed_case_count']}`",
      f"- all-top8 max KLD / top1 / failed cases: "
      f"`{all_top8['corrected_max_kld']}` / "
      f"`{all_top8['corrected_top1_rate']}` / `{all_top8['failed_case_count']}`",
      f"- unique fit top8 token IDs: `{metrics['unique_fit_top8_token_count']}`",
      f"- failed evidence checks: `{failed}`",
      "",
      metrics["next_route_reason"],
      "",
      "No target command, validation case, or test case was used.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--sequence", type=int, default=576)
  parser.add_argument("--routes", type=Path,
                      default=ACTIVE / "routes-ledger.json")
  parser.add_argument(
      "--predecessor", type=Path,
      default=ROOT / "output/seq575-state-conditioned-head-fit-model-gate-20260710Tseq575Z/metrics.json")
  parser.add_argument(
      "--corpus-contract", type=Path,
      default=ACTIVE / "intel-qwen36-35b-a3b-gguf-q4km-state-conditioned-head-correction-corpus-2026-07-10.json")
  parser.add_argument(
      "--out-dir", type=Path,
      default=ROOT / "output/seq576-state-conditioned-head-post-fit-route-control-gate-20260710Tseq576Z")
  args = parser.parse_args()
  metrics = compute(args)
  write_outputs(metrics, args.out_dir)
  print(json.dumps({
      "required_checks_passed": metrics["required_checks_passed"],
      "disposition": metrics["disposition"],
      "selected16_oracle": {
          "max_kld": metrics[
              "selected16_mass_matched_oracle"]["corrected_max_kld"],
          "top1_rate": metrics[
              "selected16_mass_matched_oracle"]["corrected_top1_rate"],
          "failed_cases": metrics[
              "selected16_mass_matched_oracle"]["failed_cases"],
      },
      "all_top8_oracle": {
          "max_kld": metrics[
              "all_top8_mass_matched_oracle"]["corrected_max_kld"],
          "top1_rate": metrics[
              "all_top8_mass_matched_oracle"]["corrected_top1_rate"],
          "failed_cases": metrics[
              "all_top8_mass_matched_oracle"]["failed_cases"],
      },
      "selected_next_route": metrics["selected_next_route"],
      "out_dir": _rel(args.out_dir),
  }, sort_keys=True))
  return 0 if metrics["required_checks_passed"] else 2


if __name__ == "__main__":
  raise SystemExit(main())
