#!/usr/bin/env python3
"""Fit and classify the locked state-conditioned head correction offline."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = "intel-qwen36-state-conditioned-head-fit-model-gate-v0"
MODEL_SCHEMA_VERSION = "intel-qwen36-state-conditioned-head-correction-v0"
CURRENT_ROUTE = (
    "router_prompt_distribution_state_conditioned_head_correction_"
    "fit_model_gate"
)
VALIDATION_ROUTE = (
    "router_prompt_distribution_state_conditioned_head_correction_"
    "validation_split_gate"
)
REJECTION_ROUTE = (
    "router_prompt_distribution_state_conditioned_head_correction_"
    "post_fit_route_control_gate"
)
RIDGE_LAMBDAS = (0.0001, 0.001, 0.01, 0.1, 1.0, 10.0)
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


def _iter_token_rows(
    results: dict[str, dict[str, Any]], excluded_case: str | None = None,
) -> Iterable[tuple[str, int, float, float]]:
  for case_id, result in results.items():
    if case_id == excluded_case:
      continue
    for step in _distribution(result).get("steps", []):
      if not isinstance(step, dict):
        continue
      for row in step.get("gpu_top8_fit_observables", []):
        if not isinstance(row, dict):
          continue
        token_id = row.get("token_id")
        gap = row.get("gpu_logit_minus_top1")
        target = row.get("native_minus_gpu_logit")
        if (isinstance(token_id, int)
            and isinstance(gap, (int, float))
            and isinstance(target, (int, float))):
          yield case_id, token_id, float(gap), float(target)


def _fit_parameters(
    results: dict[str, dict[str, Any]], token_ids: list[int], ridge_lambda: float,
    excluded_case: str | None = None,
) -> list[dict[str, Any]]:
  observations: dict[int, list[tuple[float, float]]] = {
      token_id: [] for token_id in token_ids
  }
  for _, token_id, gap, target in _iter_token_rows(results, excluded_case):
    if token_id in observations:
      observations[token_id].append((gap, target))
  parameters = []
  for token_id in token_ids:
    rows = observations[token_id]
    count = len(rows)
    sum_x = sum(row[0] for row in rows)
    sum_y = sum(row[1] for row in rows)
    sum_xx = sum(row[0] * row[0] for row in rows)
    sum_xy = sum(row[0] * row[1] for row in rows)
    a00 = count + ridge_lambda
    a01 = sum_x
    a11 = sum_xx + ridge_lambda
    determinant = a00 * a11 - a01 * a01
    if determinant <= 0.0 or not math.isfinite(determinant):
      intercept = 0.0
      slope = 0.0
    else:
      intercept = (sum_y * a11 - a01 * sum_xy) / determinant
      slope = (a00 * sum_xy - a01 * sum_y) / determinant
    squared_error = sum(
        (intercept + slope * gap - target) ** 2 for gap, target in rows)
    parameters.append({
        "token_id": token_id,
        "intercept": intercept,
        "gpu_logit_minus_top1_coefficient": slope,
        "fit_observation_count": count,
        "fit_rmse": math.sqrt(squared_error / count) if count else None,
    })
  return parameters


def _safe_expm1(value: float) -> float:
  if value > 700.0:
    return math.inf
  return math.expm1(value)


def _evaluate_step(step: dict[str, Any], parameters: dict[int, tuple[float, float]]) -> dict[str, Any]:
  rows = step.get("gpu_top8_fit_observables")
  if not isinstance(rows, list) or len(rows) != 8:
    return {"valid": False}
  baseline_kld = step.get("kld")
  gpu_lse = step.get("gpu_logsumexp")
  native_lse = step.get("native_logsumexp")
  native_top1_id = step.get("native_top1_id")
  if not all(isinstance(value, (int, float))
             for value in (baseline_kld, gpu_lse, native_lse)):
    return {"valid": False}
  correction_rows: list[dict[str, Any]] = []
  mass_delta = 0.0
  native_weighted_correction = 0.0
  corrected_logits: list[tuple[int, int, float]] = []
  for row in rows:
    if not isinstance(row, dict):
      return {"valid": False}
    token_id = row.get("token_id")
    rank = row.get("rank")
    gpu_logit = row.get("gpu_logit")
    native_logit = row.get("native_logit")
    if (not isinstance(token_id, int) or not isinstance(rank, int)
        or not isinstance(gpu_logit, (int, float))
        or not isinstance(native_logit, (int, float))):
      return {"valid": False}
    intercept, slope = parameters.get(token_id, (0.0, 0.0))
    gap = float(row["gpu_logit_minus_top1"])
    correction = intercept + slope * gap
    if not math.isfinite(correction):
      return {"valid": False}
    if token_id in parameters:
      gpu_probability = math.exp(float(gpu_logit) - float(gpu_lse))
      native_probability = math.exp(float(native_logit) - float(native_lse))
      mass_delta += gpu_probability * _safe_expm1(correction)
      native_weighted_correction += native_probability * correction
      correction_rows.append({
          "token_id": token_id,
          "correction": correction,
      })
    corrected_logits.append((rank, token_id, float(gpu_logit) + correction))
  if not math.isfinite(mass_delta) or mass_delta <= -1.0:
    return {"valid": False}
  logsumexp_delta = math.log1p(mass_delta)
  corrected_kld = (
      float(baseline_kld) + logsumexp_delta - native_weighted_correction)
  if corrected_kld < 0.0 and corrected_kld >= -1e-9:
    corrected_kld = 0.0
  winner = max(corrected_logits, key=lambda row: (row[2], -row[0]))
  outside_upper_bound = float(rows[-1]["gpu_logit"])
  top1_proven = (
      winner[1] == native_top1_id
      and winner[2] + 1e-7 >= outside_upper_bound)
  return {
      "valid": math.isfinite(corrected_kld) and corrected_kld >= 0.0,
      "baseline_kld": float(baseline_kld),
      "corrected_kld": corrected_kld,
      "kld_delta": corrected_kld - float(baseline_kld),
      "native_top1_id": native_top1_id,
      "corrected_top1_id": winner[1],
      "corrected_top1_proven": top1_proven,
      "logsumexp_delta": logsumexp_delta,
      "corrected_token_count": len(correction_rows),
  }


def _evaluate(
    results: dict[str, dict[str, Any]], parameter_rows: list[dict[str, Any]],
) -> dict[str, Any]:
  parameters = {
      int(row["token_id"]): (
          float(row["intercept"]),
          float(row["gpu_logit_minus_top1_coefficient"]),
      )
      for row in parameter_rows
  }
  case_summaries: dict[str, dict[str, Any]] = {}
  all_steps: list[dict[str, Any]] = []
  for case_id, result in results.items():
    evaluations = [
        _evaluate_step(step, parameters)
        for step in _distribution(result).get("steps", [])
        if isinstance(step, dict)
    ]
    valid = len(evaluations) == 8 and all(row.get("valid") for row in evaluations)
    baseline_max = max(
        (float(row["baseline_kld"]) for row in evaluations), default=math.inf)
    corrected_max = max(
        (float(row["corrected_kld"]) for row in evaluations), default=math.inf)
    corrected_mean = (
        sum(float(row["corrected_kld"]) for row in evaluations)
        / len(evaluations) if evaluations else math.inf)
    top1_count = sum(bool(row.get("corrected_top1_proven"))
                     for row in evaluations)
    top1_rate = top1_count / len(evaluations) if evaluations else 0.0
    no_regression = corrected_max <= baseline_max + KLD_REGRESSION_EPSILON
    contract_passed = (
        valid and corrected_max <= KLD_MAX and top1_rate >= TOP1_RATE_MIN
        and no_regression)
    case_summaries[case_id] = {
        "valid": valid,
        "position_count": len(evaluations),
        "baseline_max_kld": baseline_max,
        "corrected_max_kld": corrected_max,
        "corrected_mean_kld": corrected_mean,
        "max_kld_delta": corrected_max - baseline_max,
        "corrected_top1_match_count": top1_count,
        "corrected_top1_rate": top1_rate,
        "no_kld_regression": no_regression,
        "contract_passed": contract_passed,
    }
    all_steps.extend(evaluations)
  valid_steps = [row for row in all_steps if row.get("valid")]
  return {
      "case_count": len(case_summaries),
      "position_count": len(all_steps),
      "valid_position_count": len(valid_steps),
      "baseline_max_kld": max(
          (float(row["baseline_kld"]) for row in valid_steps),
          default=math.inf),
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
          not row["contract_passed"] for row in case_summaries.values()),
      "failed_cases": [
          case_id for case_id, row in case_summaries.items()
          if not row["contract_passed"]
      ],
      "cases": case_summaries,
  }


def _cross_validate(
    results: dict[str, dict[str, Any]], token_ids: list[int], ridge_lambda: float,
) -> dict[str, Any]:
  cases: dict[str, dict[str, Any]] = {}
  for held_case in results:
    parameters = _fit_parameters(
        results, token_ids, ridge_lambda, excluded_case=held_case)
    evaluation = _evaluate({held_case: results[held_case]}, parameters)
    cases[held_case] = evaluation["cases"][held_case]
  corrected_max = max(
      (float(row["corrected_max_kld"]) for row in cases.values()),
      default=math.inf)
  mean_numerator = sum(
      float(row["corrected_mean_kld"]) * int(row["position_count"])
      for row in cases.values())
  positions = sum(int(row["position_count"]) for row in cases.values())
  top1_matches = sum(
      int(row["corrected_top1_match_count"]) for row in cases.values())
  return {
      "ridge_lambda": ridge_lambda,
      "case_count": len(cases),
      "position_count": positions,
      "corrected_max_kld": corrected_max,
      "corrected_mean_kld": mean_numerator / positions if positions else math.inf,
      "corrected_top1_rate": top1_matches / positions if positions else 0.0,
      "failed_case_count": sum(
          not row["contract_passed"] for row in cases.values()),
      "failed_cases": [
          case_id for case_id, row in cases.items()
          if not row["contract_passed"]
      ],
      "cases": cases,
  }


def _cv_objective(row: dict[str, Any]) -> tuple[Any, ...]:
  top1_mismatches = round(
      (1.0 - float(row["corrected_top1_rate"]))
      * int(row["position_count"]))
  return (
      int(row["failed_case_count"]),
      top1_mismatches,
      float(row["corrected_max_kld"]),
      float(row["corrected_mean_kld"]),
      -float(row["ridge_lambda"]),
  )


def compute(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
  routes = _load(args.routes)
  predecessor = _load(args.predecessor)
  corpus = _load(args.corpus_contract)
  selected_rows = predecessor.get(
      "candidate_token_selection", {}).get("selected_tokens", [])
  token_ids = [
      int(row["token_id"]) for row in selected_rows
      if isinstance(row, dict) and isinstance(row.get("token_id"), int)
  ]
  result_paths = predecessor.get("inputs", {}).get("result_paths", {})
  loaded_results = {
      case_id: _load(ROOT / path)
      for case_id, path in result_paths.items()
      if isinstance(case_id, str) and isinstance(path, str)
  }
  fit_ids = [
      str(row["id"]) for row in corpus.get("prompts", [])
      if isinstance(row, dict) and row.get("split") == "fit"
  ]
  results = {
      case_id: loaded_results[case_id]
      for case_id in fit_ids if case_id in loaded_results
  }
  predecessor_selects = (
      predecessor.get("required_checks_passed") is True
      and predecessor.get("model_fit_allowed") is True
      and predecessor.get("validation_or_test_allowed") is False
      and predecessor.get("selected_next_route") == CURRENT_ROUTE
      and _has_candidate(routes, 574, CURRENT_ROUTE)
      and _has_switch(
          routes, 574,
          "select_router_prompt_distribution_state_conditioned_head_"
          "correction_fit_model_gate"))
  data_is_fit_only = (
      len(results) == 12
      and list(results) == fit_ids
      and set(loaded_results) == set(fit_ids)
      and set(result_paths) == set(fit_ids))
  model_contract = corpus.get("model_contract", {})
  contract_locked = (
      model_contract.get("maximum_token_ids") == 16
      and model_contract.get("parameters_per_token") == 2
      and model_contract.get("maximum_parameters") == 32
      and model_contract.get("candidate_token_selection")
      == "fit-split aggregate absolute KLD contributors only"
      and model_contract.get("model_class")
      == "per-token ridge affine correction over intercept and "
      "gpu_logit_minus_top1_logit, applied only when the token is in GPU top8"
      and len(token_ids) == 16
      and len(set(token_ids)) == 16)
  cv_rows = [
      _cross_validate(results, token_ids, ridge_lambda)
      for ridge_lambda in RIDGE_LAMBDAS
  ]
  selected_cv = min(cv_rows, key=_cv_objective)
  selected_lambda = float(selected_cv["ridge_lambda"])
  parameter_rows = _fit_parameters(results, token_ids, selected_lambda)
  full_fit = _evaluate(results, parameter_rows)
  parameter_shape_passes = (
      len(parameter_rows) == 16
      and len(parameter_rows) * 2 <= 32
      and [row["token_id"] for row in parameter_rows] == token_ids
      and all(
          math.isfinite(float(row["intercept"]))
          and math.isfinite(float(row[
              "gpu_logit_minus_top1_coefficient"]))
          and int(row["fit_observation_count"]) >= 1
          for row in parameter_rows))
  simulation_passes = (
      all(row["case_count"] == 12 and row["position_count"] == 96
          for row in cv_rows)
      and full_fit["case_count"] == 12
      and full_fit["position_count"] == 96
      and full_fit["valid_position_count"] == 96)
  checks = [
      {"name": "seq574_selected_fit_only_model_gate",
       "pass": predecessor_selects},
      {"name": "loaded_results_are_exactly_the_12_locked_fit_cases",
       "pass": data_is_fit_only,
       "detail": {"loaded_case_ids": list(results), "fit_ids": fit_ids}},
      {"name": "frozen_model_contract_and_16_token_order_match",
       "pass": contract_locked,
       "detail": {"token_ids": token_ids}},
      {"name": "ridge_grid_and_exact_kld_simulation_are_complete",
       "pass": simulation_passes,
       "detail": {"ridge_lambdas": list(RIDGE_LAMBDAS)}},
      {"name": "final_model_has_32_finite_parameters_with_fit_support",
       "pass": parameter_shape_passes},
  ]
  required = all(bool(row["pass"]) for row in checks)
  feasibility_passed = (
      required
      and selected_cv["failed_case_count"] == 0
      and selected_cv["corrected_top1_rate"] >= TOP1_RATE_MIN
      and selected_cv["corrected_max_kld"] <= KLD_MAX
      and full_fit["failed_case_count"] == 0
      and full_fit["corrected_top1_rate"] >= TOP1_RATE_MIN
      and full_fit["corrected_max_kld"] <= KLD_MAX)
  model = {
      "schema_version": MODEL_SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "fit_sequence": args.sequence,
      "training_split": "fit",
      "training_case_ids": list(results),
      "candidate_token_selection_source": _rel(args.predecessor),
      "candidate_token_selection":
      "fit-split aggregate absolute KLD contributors only",
      "runtime_features": [
          "gpu_top8_token_ids", "gpu_top8_logits", "top1_margin"
      ],
      "model_class": (
          "per-token ridge affine correction over intercept and "
          "gpu_logit_minus_top1_logit, applied only when the token is in GPU top8"
      ),
      "ridge_lambda": selected_lambda,
      "parameters_per_token": 2,
      "parameter_count": len(parameter_rows) * 2,
      "parameters": parameter_rows,
      "runtime_native_oracle_required": False,
      "runtime_prompt_case_or_position_features_required": False,
      "full_vocab_host_rescan_required": False,
      "validation_or_test_used_for_fit": False,
  }
  metrics = {
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
      "ridge_selection": {
          "method": (
              "grouped leave-one-fit-prompt-out; minimize failed cases, top1 "
              "mismatches, max KLD, mean KLD, then prefer stronger ridge"
          ),
          "grid": cv_rows,
          "selected_lambda": selected_lambda,
          "selected_cross_validation": selected_cv,
      },
      "full_fit": full_fit,
      "model": model,
      "checks": checks,
      "required_checks_passed": required,
      "fit_feasibility_passed": feasibility_passed,
      "validation_allowed": feasibility_passed,
      "test_allowed": False,
      "runtime_source_allowed": False,
      "speedup_claims_allowed": False,
      "promotion_allowed": False,
      "disposition": (
          "accept_fit_model_feasibility"
          if feasibility_passed else
          "reject_fit_model_before_validation"),
      "selected_next_route": (
          VALIDATION_ROUTE if feasibility_passed else REJECTION_ROUTE),
      "next_route_reason": (
          "Grouped fit-only cross-validation and the full fit both clear every "
          "case at KLD <= 0.005, top1 1.0, and no per-case regression. Run "
          "the six locked validation cases without changing the model."
          if feasibility_passed else
          "The frozen ridge-affine class does not clear the fit-only "
          "cross-validation and full-fit feasibility contract. Do not expose "
          "validation/test or add runtime source; run no-token route control."),
  }
  return metrics, model


def write_outputs(metrics: dict[str, Any], model: dict[str, Any],
                  out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  (out_dir / "model.json").write_text(
      json.dumps(model, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  manifest = {
      "schema_version": metrics["schema_version"],
      "workstream": metrics["workstream"],
      "tool": _rel(Path(__file__)),
      "inputs": metrics["inputs"],
      "model": _rel(out_dir / "model.json"),
      "selected_lambda": metrics["ridge_selection"]["selected_lambda"],
      "fit_feasibility_passed": metrics["fit_feasibility_passed"],
      "selected_next_route": metrics["selected_next_route"],
      "validation_allowed": metrics["validation_allowed"],
      "test_allowed": False,
      "runtime_source_allowed": False,
      "speedup_claims_allowed": False,
  }
  (out_dir / "manifest.json").write_text(
      json.dumps(manifest, indent=2, sort_keys=True) + "\n",
      encoding="utf-8")
  failed = [row["name"] for row in metrics["checks"] if not row["pass"]]
  cv = metrics["ridge_selection"]["selected_cross_validation"]
  full = metrics["full_fit"]
  lines = [
      f"# Seq{metrics['sequence']} State-Conditioned Fit Model Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- fit_feasibility_passed: `{str(metrics['fit_feasibility_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- selected ridge lambda: `{metrics['ridge_selection']['selected_lambda']}`",
      f"- CV max KLD / top1 / failed cases: `{cv['corrected_max_kld']}` / "
      f"`{cv['corrected_top1_rate']}` / `{cv['failed_case_count']}`",
      f"- full-fit max KLD / top1 / failed cases: "
      f"`{full['corrected_max_kld']}` / `{full['corrected_top1_rate']}` / "
      f"`{full['failed_case_count']}`",
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
  parser.add_argument("--sequence", type=int, default=575)
  parser.add_argument("--routes", type=Path,
                      default=ACTIVE / "routes-ledger.json")
  parser.add_argument(
      "--predecessor", type=Path,
      default=ROOT / "output/seq574-state-conditioned-head-fit-collection-gate-20260710Tseq574Z/metrics.json")
  parser.add_argument(
      "--corpus-contract", type=Path,
      default=ACTIVE / "intel-qwen36-35b-a3b-gguf-q4km-state-conditioned-head-correction-corpus-2026-07-10.json")
  parser.add_argument(
      "--out-dir", type=Path,
      default=ROOT / "output/seq575-state-conditioned-head-fit-model-gate-20260710Tseq575Z")
  args = parser.parse_args()
  metrics, model = compute(args)
  write_outputs(metrics, model, args.out_dir)
  print(json.dumps({
      "required_checks_passed": metrics["required_checks_passed"],
      "fit_feasibility_passed": metrics["fit_feasibility_passed"],
      "disposition": metrics["disposition"],
      "selected_lambda": metrics["ridge_selection"]["selected_lambda"],
      "cv_failed_cases": metrics["ridge_selection"][
          "selected_cross_validation"]["failed_cases"],
      "full_fit_failed_cases": metrics["full_fit"]["failed_cases"],
      "validation_allowed": metrics["validation_allowed"],
      "selected_next_route": metrics["selected_next_route"],
      "out_dir": _rel(args.out_dir),
  }, sort_keys=True))
  return 0 if metrics["required_checks_passed"] else 2


if __name__ == "__main__":
  raise SystemExit(main())
