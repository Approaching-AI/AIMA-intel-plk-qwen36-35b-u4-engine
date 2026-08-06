#!/usr/bin/env python3
"""Fit bounded GPU-feature models against mass-matched fit targets."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = "intel-qwen36-state-conditioned-head-feature-reframe-gate-v0"
MODEL_SCHEMA_VERSION = (
    "intel-qwen36-state-conditioned-head-mass-matched-correction-v0"
)
CURRENT_ROUTE = (
    "router_prompt_distribution_state_conditioned_head_correction_"
    "feature_reframe_gate"
)
VALIDATION_ROUTE = (
    "router_prompt_distribution_state_conditioned_head_correction_"
    "validation_split_gate"
)
REJECTION_ROUTE = (
    "router_prompt_distribution_state_conditioned_head_correction_"
    "post_feature_reframe_route_control_gate"
)
RIDGE_LAMBDAS = (0.0001, 0.001, 0.01, 0.1, 1.0, 10.0)
MODEL_NAMES = (
    "per_token_intercept_gap",
    "per_token_intercept_top1_margin",
    "per_token_intercept_shared_gap_margin",
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


def _mass_matched_targets(
    step: dict[str, Any], selected_tokens: set[int],
) -> dict[int, float] | None:
  rows = step.get("gpu_top8_fit_observables")
  gpu_lse = step.get("gpu_logsumexp")
  native_lse = step.get("native_logsumexp")
  if (not isinstance(rows, list) or len(rows) != 8
      or not isinstance(gpu_lse, (int, float))
      or not isinstance(native_lse, (int, float))):
    return None
  selected = [
      row for row in rows
      if isinstance(row, dict)
      and isinstance(row.get("token_id"), int)
      and int(row["token_id"]) in selected_tokens
  ]
  if not selected:
    return {}
  gpu_mass = sum(
      math.exp(float(row["gpu_logit"]) - float(gpu_lse))
      for row in selected)
  native_mass = sum(
      math.exp(float(row["native_logit"]) - float(native_lse))
      for row in selected)
  native_scale_at_gpu_lse = sum(
      math.exp(float(row["native_logit"]) - float(gpu_lse))
      for row in selected)
  if gpu_mass >= 1.0 or native_mass >= 1.0 or native_scale_at_gpu_lse <= 0.0:
    return None
  target_mass = native_mass / (1.0 - native_mass) * (1.0 - gpu_mass)
  common_offset = math.log(target_mass / native_scale_at_gpu_lse)
  return {
      int(row["token_id"]): (
          float(row["native_logit"]) - float(row["gpu_logit"])
          + common_offset)
      for row in selected
  }


def _parameter_count(model_name: str, token_count: int) -> int:
  if model_name in (
      "per_token_intercept_gap", "per_token_intercept_top1_margin"):
    return token_count * 2
  if model_name == "per_token_intercept_shared_gap_margin":
    return token_count + 2
  raise ValueError(f"unknown model: {model_name}")


def _features(
    model_name: str, token_index: dict[int, int], token_id: int,
    gap: float, top1_margin: float,
) -> list[float]:
  count = len(token_index)
  values = [0.0] * _parameter_count(model_name, count)
  index = token_index[token_id]
  if model_name == "per_token_intercept_gap":
    values[index * 2] = 1.0
    values[index * 2 + 1] = gap
  elif model_name == "per_token_intercept_top1_margin":
    values[index * 2] = 1.0
    values[index * 2 + 1] = top1_margin
  elif model_name == "per_token_intercept_shared_gap_margin":
    values[index] = 1.0
    values[count] = gap
    values[count + 1] = top1_margin
  else:
    raise ValueError(f"unknown model: {model_name}")
  return values


def _training_rows(
    results: dict[str, dict[str, Any]], token_ids: list[int],
    excluded_case: str | None = None,
) -> Iterable[tuple[str, int, float, float, float]]:
  selected = set(token_ids)
  for case_id, result in results.items():
    if case_id == excluded_case:
      continue
    for step in _distribution(result).get("steps", []):
      if not isinstance(step, dict):
        continue
      targets = _mass_matched_targets(step, selected)
      if targets is None:
        continue
      margin = step.get("gpu_top1_margin")
      if not isinstance(margin, (int, float)):
        continue
      for row in step.get("gpu_top8_fit_observables", []):
        if not isinstance(row, dict):
          continue
        token_id = row.get("token_id")
        gap = row.get("gpu_logit_minus_top1")
        if (isinstance(token_id, int) and token_id in targets
            and isinstance(gap, (int, float))):
          yield (
              case_id, token_id, float(gap), float(margin),
              float(targets[token_id]))


def _solve_ridge(
    features: list[list[float]], targets: list[float], ridge_lambda: float,
) -> list[float] | None:
  if not features:
    return None
  size = len(features[0])
  matrix = [[0.0] * (size + 1) for _ in range(size)]
  for row, target in zip(features, targets, strict=True):
    for i, value_i in enumerate(row):
      if value_i == 0.0:
        continue
      matrix[i][-1] += value_i * target
      for j, value_j in enumerate(row):
        if value_j != 0.0:
          matrix[i][j] += value_i * value_j
  for i in range(size):
    matrix[i][i] += ridge_lambda
  for column in range(size):
    pivot = max(range(column, size), key=lambda row: abs(matrix[row][column]))
    if abs(matrix[pivot][column]) < 1e-14:
      return None
    if pivot != column:
      matrix[column], matrix[pivot] = matrix[pivot], matrix[column]
    pivot_value = matrix[column][column]
    for j in range(column, size + 1):
      matrix[column][j] /= pivot_value
    for row in range(size):
      if row == column:
        continue
      factor = matrix[row][column]
      if factor == 0.0:
        continue
      for j in range(column, size + 1):
        matrix[row][j] -= factor * matrix[column][j]
  solution = [matrix[row][-1] for row in range(size)]
  return solution if all(math.isfinite(value) for value in solution) else None


def _fit(
    results: dict[str, dict[str, Any]], token_ids: list[int], model_name: str,
    ridge_lambda: float, excluded_case: str | None = None,
) -> dict[str, Any]:
  token_index = {token_id: index for index, token_id in enumerate(token_ids)}
  rows = list(_training_rows(results, token_ids, excluded_case))
  features = [
      _features(model_name, token_index, token_id, gap, margin)
      for _, token_id, gap, margin, _ in rows
  ]
  targets = [target for *_, target in rows]
  coefficients = _solve_ridge(features, targets, ridge_lambda)
  if coefficients is None:
    coefficients = [0.0] * _parameter_count(model_name, len(token_ids))
  squared_error = sum(
      (sum(value * coefficient
           for value, coefficient in zip(row, coefficients, strict=True))
       - target) ** 2
      for row, target in zip(features, targets, strict=True))
  return {
      "model_name": model_name,
      "ridge_lambda": ridge_lambda,
      "parameter_count": len(coefficients),
      "training_observation_count": len(rows),
      "training_rmse": math.sqrt(squared_error / len(rows)) if rows else None,
      "coefficients": coefficients,
  }


def _predictor(
    model: dict[str, Any], token_ids: list[int],
) -> Callable[[int, float, float], float]:
  token_index = {token_id: index for index, token_id in enumerate(token_ids)}
  coefficients = [float(value) for value in model["coefficients"]]

  def predict(token_id: int, gap: float, margin: float) -> float:
    if token_id not in token_index:
      return 0.0
    row = _features(model["model_name"], token_index, token_id, gap, margin)
    return sum(value * coefficient
               for value, coefficient in zip(row, coefficients, strict=True))

  return predict


def _evaluate_step(
    step: dict[str, Any], selected_tokens: set[int],
    predict: Callable[[int, float, float], float],
) -> dict[str, Any]:
  rows = step.get("gpu_top8_fit_observables")
  values = (
      step.get("kld"), step.get("gpu_logsumexp"),
      step.get("native_logsumexp"), step.get("gpu_top1_margin"))
  if (not isinstance(rows, list) or len(rows) != 8
      or not all(isinstance(value, (int, float)) for value in values)):
    return {"valid": False}
  baseline_kld = float(step["kld"])
  gpu_lse = float(step["gpu_logsumexp"])
  native_lse = float(step["native_logsumexp"])
  margin = float(step["gpu_top1_margin"])
  mass_delta = 0.0
  native_weighted = 0.0
  corrected_logits: list[tuple[int, int, float]] = []
  corrected_count = 0
  for row in rows:
    if not isinstance(row, dict):
      return {"valid": False}
    token_id = int(row["token_id"])
    gpu_logit = float(row["gpu_logit"])
    gap = float(row["gpu_logit_minus_top1"])
    correction = predict(token_id, gap, margin) if token_id in selected_tokens else 0.0
    if not math.isfinite(correction):
      return {"valid": False}
    if token_id in selected_tokens:
      mass_delta += math.exp(gpu_logit - gpu_lse) * math.expm1(correction)
      native_weighted += (
          math.exp(float(row["native_logit"]) - native_lse) * correction)
      corrected_count += 1
    corrected_logits.append(
        (int(row["rank"]), token_id, gpu_logit + correction))
  if not math.isfinite(mass_delta) or mass_delta <= -1.0:
    return {"valid": False}
  corrected_kld = baseline_kld + math.log1p(mass_delta) - native_weighted
  if corrected_kld < 0.0 and corrected_kld >= -1e-9:
    corrected_kld = 0.0
  winner = max(corrected_logits, key=lambda row: (row[2], -row[0]))
  top1_proven = (
      winner[1] == step.get("native_top1_id")
      and winner[2] + 1e-7 >= float(rows[-1]["gpu_logit"]))
  return {
      "valid": math.isfinite(corrected_kld) and corrected_kld >= 0.0,
      "baseline_kld": baseline_kld,
      "corrected_kld": corrected_kld,
      "corrected_top1_proven": top1_proven,
      "corrected_token_count": corrected_count,
  }


def _evaluate(
    results: dict[str, dict[str, Any]], token_ids: list[int],
    model: dict[str, Any],
) -> dict[str, Any]:
  selected = set(token_ids)
  predict = _predictor(model, token_ids)
  cases: dict[str, dict[str, Any]] = {}
  all_steps: list[dict[str, Any]] = []
  for case_id, result in results.items():
    steps = [
        _evaluate_step(step, selected, predict)
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
      "cases": cases,
  }


def _cross_validate(
    results: dict[str, dict[str, Any]], token_ids: list[int], model_name: str,
    ridge_lambda: float,
) -> dict[str, Any]:
  cases: dict[str, dict[str, Any]] = {}
  observations = 0
  for held_case in results:
    model = _fit(
        results, token_ids, model_name, ridge_lambda,
        excluded_case=held_case)
    observations += int(model["training_observation_count"])
    evaluation = _evaluate({held_case: results[held_case]}, token_ids, model)
    cases[held_case] = evaluation["cases"][held_case]
  positions = sum(int(row["position_count"]) for row in cases.values())
  top1_matches = sum(
      int(row["corrected_top1_match_count"]) for row in cases.values())
  return {
      "model_name": model_name,
      "ridge_lambda": ridge_lambda,
      "parameter_count": _parameter_count(model_name, len(token_ids)),
      "fold_training_observation_count_sum": observations,
      "case_count": len(cases),
      "position_count": positions,
      "corrected_max_kld": max(
          (float(row["corrected_max_kld"]) for row in cases.values()),
          default=math.inf),
      "corrected_mean_kld": (
          sum(float(row["corrected_mean_kld"]) * int(row["position_count"])
              for row in cases.values()) / positions
          if positions else math.inf),
      "corrected_top1_rate": top1_matches / positions if positions else 0.0,
      "failed_case_count": sum(
          not row["contract_passed"] for row in cases.values()),
      "failed_cases": [
          case_id for case_id, row in cases.items()
          if not row["contract_passed"]
      ],
      "cases": cases,
  }


def _objective(row: dict[str, Any]) -> tuple[Any, ...]:
  top1_mismatches = round(
      (1.0 - float(row["corrected_top1_rate"]))
      * int(row["position_count"]))
  return (
      int(row["failed_case_count"]),
      top1_mismatches,
      float(row["corrected_max_kld"]),
      float(row["corrected_mean_kld"]),
      int(row["parameter_count"]),
      MODEL_NAMES.index(str(row["model_name"])),
      -float(row["ridge_lambda"]),
  )


def _structured_model(
    model: dict[str, Any], token_ids: list[int], sequence: int,
    training_case_ids: list[str], source: str,
) -> dict[str, Any]:
  coefficients = [float(value) for value in model["coefficients"]]
  name = str(model["model_name"])
  if name in ("per_token_intercept_gap",
              "per_token_intercept_top1_margin"):
    feature_name = (
        "gpu_logit_minus_top1" if name.endswith("_gap") else "top1_margin")
    parameters = [
        {"token_id": token_id,
         "intercept": coefficients[index * 2],
         f"{feature_name}_coefficient": coefficients[index * 2 + 1]}
        for index, token_id in enumerate(token_ids)
    ]
  else:
    parameters = {
        "token_intercepts": [
            {"token_id": token_id, "intercept": coefficients[index]}
            for index, token_id in enumerate(token_ids)
        ],
        "shared_gpu_logit_minus_top1_coefficient": coefficients[len(token_ids)],
        "shared_top1_margin_coefficient": coefficients[len(token_ids) + 1],
    }
  return {
      "schema_version": MODEL_SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "fit_sequence": sequence,
      "training_split": "fit",
      "training_case_ids": training_case_ids,
      "fit_target": "selected16 probability-mass-matched oracle correction",
      "fit_target_source": source,
      "runtime_native_oracle_required": False,
      "runtime_prompt_case_or_position_features_required": False,
      "runtime_features": [
          "gpu_top8_token_ids", "gpu_top8_logits", "top1_margin"
      ],
      "model_name": name,
      "ridge_lambda": model["ridge_lambda"],
      "parameter_count": model["parameter_count"],
      "parameters": parameters,
      "full_vocab_host_rescan_required": False,
      "validation_or_test_used_for_fit": False,
  }


def compute(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
  routes = _load(args.routes)
  predecessor = _load(args.predecessor)
  prior_fit = _load(args.prior_fit)
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
  token_ids = [
      int(row["token_id"])
      for row in prior_fit.get("model", {}).get("parameters", [])
      if isinstance(row, dict) and isinstance(row.get("token_id"), int)
  ]
  predecessor_selects = (
      predecessor.get("required_checks_passed") is True
      and predecessor.get("selected16_oracle_contract_passed") is True
      and predecessor.get("validation_or_test_allowed") is False
      and predecessor.get("selected_next_route") == CURRENT_ROUTE
      and prior_fit.get("fit_feasibility_passed") is False
      and _has_candidate(routes, 576, CURRENT_ROUTE)
      and _has_switch(
          routes, 576,
          "select_router_prompt_distribution_state_conditioned_head_"
          "correction_feature_reframe_gate"))
  data_and_shape_pass = (
      len(results) == 12 and list(results) == fit_ids
      and set(loaded) == set(fit_ids)
      and len(token_ids) == 16 and len(set(token_ids)) == 16)
  cv_rows = [
      _cross_validate(results, token_ids, model_name, ridge_lambda)
      for model_name in MODEL_NAMES
      for ridge_lambda in RIDGE_LAMBDAS
  ]
  selected_cv = min(cv_rows, key=_objective)
  selected_model = _fit(
      results, token_ids, str(selected_cv["model_name"]),
      float(selected_cv["ridge_lambda"]))
  full_fit = _evaluate(results, token_ids, selected_model)
  selected_model_shape = (
      selected_model["parameter_count"] <= 32
      and selected_model["training_observation_count"] > 0
      and len(selected_model["coefficients"])
      == selected_model["parameter_count"]
      and all(math.isfinite(float(value))
              for value in selected_model["coefficients"]))
  grid_complete = (
      len(cv_rows) == len(MODEL_NAMES) * len(RIDGE_LAMBDAS)
      and all(row["case_count"] == 12 and row["position_count"] == 96
              and row["parameter_count"] <= 32 for row in cv_rows)
      and full_fit["case_count"] == 12
      and full_fit["position_count"] == 96
      and full_fit["valid_position_count"] == 96)
  checks = [
      {"name": "seq576_selected_mass_matched_feature_reframe",
       "pass": predecessor_selects},
      {"name": "fit_only_data_and_frozen_16_token_shape_match",
       "pass": data_and_shape_pass,
       "detail": {"fit_ids": fit_ids, "loaded_ids": list(results),
                  "token_ids": token_ids}},
      {"name": "three_pre_registered_models_complete_grouped_cv",
       "pass": grid_complete,
       "detail": {"model_names": list(MODEL_NAMES),
                  "ridge_lambdas": list(RIDGE_LAMBDAS)}},
      {"name": "selected_model_is_finite_and_within_32_parameters",
       "pass": selected_model_shape,
       "detail": {"model_name": selected_model["model_name"],
                  "parameter_count": selected_model["parameter_count"]}},
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
  model = _structured_model(
      selected_model, token_ids, args.sequence, list(results),
      _rel(args.predecessor))
  metrics = {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "sequence": args.sequence,
      "inputs": {
          "routes": _rel(args.routes),
          "predecessor": _rel(args.predecessor),
          "prior_fit": _rel(args.prior_fit),
          "corpus_contract": _rel(args.corpus_contract),
          "fit_result_paths": result_paths,
      },
      "thresholds": {
          "max_kld": KLD_MAX,
          "top1_rate_min": TOP1_RATE_MIN,
          "per_case_kld_regression_epsilon": KLD_REGRESSION_EPSILON,
          "maximum_parameters": 32,
      },
      "feature_reframe": {
          "fit_target": "selected16 probability-mass-matched oracle correction",
          "selection_method": (
              "grouped leave-one-fit-prompt-out; minimize failed cases, top1 "
              "mismatches, max KLD, mean KLD, parameter count, fixed model "
              "order, then prefer stronger ridge"
          ),
          "grid": cv_rows,
          "selected_cross_validation": selected_cv,
      },
      "full_fit": full_fit,
      "model": model,
      "checks": checks,
      "required_checks_passed": required,
      "feature_reframe_feasibility_passed": feasibility_passed,
      "validation_allowed": feasibility_passed,
      "test_allowed": False,
      "runtime_source_allowed": False,
      "speedup_claims_allowed": False,
      "promotion_allowed": False,
      "disposition": (
          "accept_mass_matched_feature_reframe"
          if feasibility_passed else
          "reject_mass_matched_feature_reframe_before_validation"),
      "selected_next_route": (
          VALIDATION_ROUTE if feasibility_passed else REJECTION_ROUTE),
      "next_route_reason": (
          "The selected mass-matched GPU-feature model clears grouped fit-only "
          "cross-validation and full fit. Freeze it and run exactly the six "
          "locked validation cases without model changes."
          if feasibility_passed else
          "All three bounded mass-matched GPU-feature reframes fail before "
          "validation. Keep validation/test sealed and run no-target route "
          "control; do not add feature variants or target source."),
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
      "selected_model_name": model["model_name"],
      "selected_ridge_lambda": model["ridge_lambda"],
      "parameter_count": model["parameter_count"],
      "feature_reframe_feasibility_passed": metrics[
          "feature_reframe_feasibility_passed"],
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
  cv = metrics["feature_reframe"]["selected_cross_validation"]
  full = metrics["full_fit"]
  lines = [
      f"# Seq{metrics['sequence']} Mass-Matched Feature Reframe Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- feasibility_passed: `{str(metrics['feature_reframe_feasibility_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- selected model / lambda / parameters: `{model['model_name']}` / "
      f"`{model['ridge_lambda']}` / `{model['parameter_count']}`",
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
  parser.add_argument("--sequence", type=int, default=577)
  parser.add_argument("--routes", type=Path,
                      default=ACTIVE / "routes-ledger.json")
  parser.add_argument(
      "--predecessor", type=Path,
      default=ROOT / "output/seq576-state-conditioned-head-post-fit-route-control-gate-20260710Tseq576Z/metrics.json")
  parser.add_argument(
      "--prior-fit", type=Path,
      default=ROOT / "output/seq575-state-conditioned-head-fit-model-gate-20260710Tseq575Z/metrics.json")
  parser.add_argument(
      "--corpus-contract", type=Path,
      default=ACTIVE / "intel-qwen36-35b-a3b-gguf-q4km-state-conditioned-head-correction-corpus-2026-07-10.json")
  parser.add_argument(
      "--out-dir", type=Path,
      default=ROOT / "output/seq577-state-conditioned-head-feature-reframe-gate-20260710Tseq577Z")
  args = parser.parse_args()
  metrics, model = compute(args)
  write_outputs(metrics, model, args.out_dir)
  cv = metrics["feature_reframe"]["selected_cross_validation"]
  print(json.dumps({
      "required_checks_passed": metrics["required_checks_passed"],
      "feature_reframe_feasibility_passed": metrics[
          "feature_reframe_feasibility_passed"],
      "disposition": metrics["disposition"],
      "selected_model_name": model["model_name"],
      "selected_ridge_lambda": model["ridge_lambda"],
      "parameter_count": model["parameter_count"],
      "cv_max_kld": cv["corrected_max_kld"],
      "cv_top1_rate": cv["corrected_top1_rate"],
      "cv_failed_cases": cv["failed_cases"],
      "full_fit_max_kld": metrics["full_fit"]["corrected_max_kld"],
      "full_fit_top1_rate": metrics["full_fit"]["corrected_top1_rate"],
      "full_fit_failed_cases": metrics["full_fit"]["failed_cases"],
      "validation_allowed": metrics["validation_allowed"],
      "selected_next_route": metrics["selected_next_route"],
      "out_dir": _rel(args.out_dir),
  }, sort_keys=True))
  return 0 if metrics["required_checks_passed"] else 2


if __name__ == "__main__":
  raise SystemExit(main())
