#!/usr/bin/env python3
"""Select sparse GPU final-norm dimensions and fit the locked model."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
FEATURE_TOOL = ROOT / "tools/intel-qwen36-state-conditioned-head-feature-reframe-gate.py"
SCHEMA_VERSION = "intel-qwen36-final-norm-sparse-state-dimension-fit-gate-v0"
MODEL_SCHEMA_VERSION = "intel-qwen36-final-norm-sparse-state-model-v0"
CURRENT_ROUTE = (
    "router_prompt_distribution_final_norm_sparse_state_feature_dimension_fit_gate"
)
VALIDATION_ROUTE = (
    "router_prompt_distribution_final_norm_sparse_state_feature_validation_gate"
)
REJECTION_ROUTE = (
    "router_prompt_distribution_final_norm_sparse_state_feature_route_close_gate"
)
KLD_MAX = 0.005
TOP1_RATE_MIN = 1.0
KLD_REGRESSION_EPSILON = 1e-7


def _load_module(path: Path, name: str) -> Any:
  spec = importlib.util.spec_from_file_location(name, path)
  if spec is None or spec.loader is None:
    raise SystemExit(f"cannot load module: {path}")
  module = importlib.util.module_from_spec(spec)
  sys.modules[name] = module
  spec.loader.exec_module(module)
  return module


FEATURE = _load_module(FEATURE_TOOL, "iq36_sparse_state_feature_helpers")


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


def _common_offset(step: dict[str, Any], selected_tokens: set[int]) -> float | None:
  targets = FEATURE._mass_matched_targets(step, selected_tokens)  # noqa: SLF001
  rows = step.get("gpu_top8_fit_observables")
  if not targets or not isinstance(rows, list):
    return None
  offsets = []
  for row in rows:
    if not isinstance(row, dict):
      continue
    token_id = row.get("token_id")
    if isinstance(token_id, int) and token_id in targets:
      offsets.append(
          float(targets[token_id])
          - float(row["native_minus_gpu_logit"]))
  if not offsets:
    return None
  first = offsets[0]
  if any(not math.isclose(value, first, rel_tol=0.0, abs_tol=1e-9)
         for value in offsets[1:]):
    return None
  return sum(offsets) / len(offsets)


def _select_dimensions(
    results: dict[str, dict[str, Any]], selected_tokens: set[int],
    maximum: int, excluded_case: str | None = None,
) -> dict[str, Any]:
  vectors: list[list[float]] = []
  targets: list[float] = []
  source_cases: list[str] = []
  for case_id, result in results.items():
    if case_id == excluded_case:
      continue
    for step in _distribution(result).get("steps", []):
      if not isinstance(step, dict):
        continue
      vector = step.get("gpu_final_norm_fit_observables")
      target = _common_offset(step, selected_tokens)
      if (isinstance(vector, list) and len(vector) == 2048
          and target is not None):
        vectors.append([float(value) for value in vector])
        targets.append(float(target))
        source_cases.append(case_id)
  if not vectors or len(vectors) != len(targets):
    return {"valid": False, "dimensions": []}
  count = len(vectors)
  target_mean = sum(targets) / count
  target_var = sum((value - target_mean) ** 2 for value in targets)
  ranked: list[dict[str, Any]] = []
  for dimension in range(2048):
    values = [vector[dimension] for vector in vectors]
    mean = sum(values) / count
    variance = sum((value - mean) ** 2 for value in values)
    covariance = sum(
        (value - mean) * (target - target_mean)
        for value, target in zip(values, targets, strict=True))
    denominator = math.sqrt(variance * target_var)
    correlation = covariance / denominator if denominator > 0.0 else 0.0
    ranked.append({
        "dimension": dimension,
        "absolute_correlation": abs(correlation),
        "correlation": correlation,
    })
  ranked.sort(key=lambda row: (-row["absolute_correlation"], row["dimension"]))
  selected = ranked[:maximum]
  return {
      "valid": len(selected) == maximum,
      "excluded_case": excluded_case,
      "training_case_ids": sorted(set(source_cases)),
      "training_step_count": count,
      "dimensions": selected,
  }


def _training_rows(
    results: dict[str, dict[str, Any]], token_ids: list[int],
    dimensions: list[int], excluded_case: str | None = None,
) -> list[tuple[list[float], float]]:
  selected_tokens = set(token_ids)
  token_index = {token_id: index for index, token_id in enumerate(token_ids)}
  rows_out: list[tuple[list[float], float]] = []
  for case_id, result in results.items():
    if case_id == excluded_case:
      continue
    for step in _distribution(result).get("steps", []):
      if not isinstance(step, dict):
        continue
      vector = step.get("gpu_final_norm_fit_observables")
      top8 = step.get("gpu_top8_fit_observables")
      targets = FEATURE._mass_matched_targets(  # noqa: SLF001
          step, selected_tokens)
      if (not isinstance(vector, list) or len(vector) != 2048
          or not isinstance(top8, list) or targets is None):
        continue
      state_features = [float(vector[dimension]) for dimension in dimensions]
      for row in top8:
        if not isinstance(row, dict):
          continue
        token_id = row.get("token_id")
        if isinstance(token_id, int) and token_id in targets:
          features = [0.0] * 32
          features[token_index[token_id]] = 1.0
          features[16:] = state_features
          rows_out.append((features, float(targets[token_id])))
  return rows_out


def _fit(
    results: dict[str, dict[str, Any]], token_ids: list[int],
    dimensions: list[int], ridge_lambda: float,
    excluded_case: str | None = None,
) -> dict[str, Any]:
  rows = _training_rows(
      results, token_ids, dimensions, excluded_case=excluded_case)
  features = [row[0] for row in rows]
  targets = [row[1] for row in rows]
  coefficients = FEATURE._solve_ridge(  # noqa: SLF001
      features, targets, ridge_lambda)
  if coefficients is None:
    coefficients = [0.0] * 32
  squared_error = sum(
      (sum(value * coefficient
           for value, coefficient in zip(feature, coefficients, strict=True))
       - target) ** 2
      for feature, target in rows)
  return {
      "ridge_lambda": ridge_lambda,
      "parameter_count": len(coefficients),
      "dimensions": dimensions,
      "training_observation_count": len(rows),
      "training_rmse": math.sqrt(squared_error / len(rows)) if rows else None,
      "coefficients": coefficients,
  }


def _predictor(
    model: dict[str, Any], token_ids: list[int],
) -> Callable[[int, list[float]], float]:
  token_index = {token_id: index for index, token_id in enumerate(token_ids)}
  dimensions = [int(value) for value in model["dimensions"]]
  coefficients = [float(value) for value in model["coefficients"]]

  def predict(token_id: int, final_norm: list[float]) -> float:
    if token_id not in token_index:
      return 0.0
    value = coefficients[token_index[token_id]]
    value += sum(
        coefficients[16 + index] * float(final_norm[dimension])
        for index, dimension in enumerate(dimensions))
    return value

  return predict


def _evaluate_step(
    step: dict[str, Any], selected_tokens: set[int],
    predict: Callable[[int, list[float]], float],
) -> dict[str, Any]:
  rows = step.get("gpu_top8_fit_observables")
  final_norm = step.get("gpu_final_norm_fit_observables")
  values = (
      step.get("kld"), step.get("gpu_logsumexp"),
      step.get("native_logsumexp"))
  if (not isinstance(rows, list) or len(rows) != 8
      or not isinstance(final_norm, list) or len(final_norm) != 2048
      or not all(isinstance(value, (int, float)) for value in values)):
    return {"valid": False}
  baseline_kld = float(step["kld"])
  gpu_lse = float(step["gpu_logsumexp"])
  native_lse = float(step["native_logsumexp"])
  mass_delta = 0.0
  native_weighted = 0.0
  corrected_logits: list[tuple[int, int, float]] = []
  for row in rows:
    if not isinstance(row, dict):
      return {"valid": False}
    token_id = int(row["token_id"])
    gpu_logit = float(row["gpu_logit"])
    correction = predict(token_id, final_norm) if token_id in selected_tokens else 0.0
    if not math.isfinite(correction):
      return {"valid": False}
    if token_id in selected_tokens:
      mass_delta += math.exp(gpu_logit - gpu_lse) * math.expm1(correction)
      native_weighted += (
          math.exp(float(row["native_logit"]) - native_lse) * correction)
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
  }


def _evaluate(
    results: dict[str, dict[str, Any]], token_ids: list[int],
    model: dict[str, Any],
) -> dict[str, Any]:
  selected_tokens = set(token_ids)
  predict = _predictor(model, token_ids)
  cases: dict[str, dict[str, Any]] = {}
  all_steps: list[dict[str, Any]] = []
  for case_id, result in results.items():
    steps = [
        _evaluate_step(step, selected_tokens, predict)
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
    results: dict[str, dict[str, Any]], token_ids: list[int],
    ridge_lambda: float,
) -> dict[str, Any]:
  selected_tokens = set(token_ids)
  cases: dict[str, dict[str, Any]] = {}
  folds: dict[str, dict[str, Any]] = {}
  for held_case in results:
    selection = _select_dimensions(
        results, selected_tokens, 16, excluded_case=held_case)
    dimensions = [int(row["dimension"])
                  for row in selection.get("dimensions", [])]
    model = _fit(
        results, token_ids, dimensions, ridge_lambda,
        excluded_case=held_case)
    evaluation = _evaluate({held_case: results[held_case]}, token_ids, model)
    cases[held_case] = evaluation["cases"][held_case]
    folds[held_case] = {
        "training_case_ids": selection.get("training_case_ids", []),
        "dimension_selection_step_count": selection.get(
            "training_step_count"),
        "dimensions": selection.get("dimensions", []),
        "training_observation_count": model["training_observation_count"],
    }
  positions = sum(int(row["position_count"]) for row in cases.values())
  top1_matches = sum(
      int(row["corrected_top1_match_count"]) for row in cases.values())
  return {
      "ridge_lambda": ridge_lambda,
      "parameter_count": 32,
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
      "folds": folds,
      "cases": cases,
  }


def _objective(row: dict[str, Any]) -> tuple[Any, ...]:
  top1_mismatches = round(
      (1.0 - float(row["corrected_top1_rate"]))
      * int(row["position_count"]))
  return (
      int(row["failed_case_count"]), top1_mismatches,
      float(row["corrected_max_kld"]),
      float(row["corrected_mean_kld"]),
      -float(row["ridge_lambda"]),
  )


def compute(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
  routes = _load(args.routes)
  predecessor = _load(args.predecessor)
  contract = _load(args.contract)
  result_paths = predecessor.get("inputs", {}).get("result_paths", {})
  fit_ids = contract.get(
      "feature_collection_contract", {}).get("fit_diagnostic_case_ids", [])
  loaded = {case_id: _load(ROOT / path)
            for case_id, path in result_paths.items()
            if isinstance(case_id, str) and isinstance(path, str)}
  results = {case_id: loaded[case_id] for case_id in fit_ids
             if case_id in loaded}
  model_contract = contract.get("model_contract", {})
  token_ids = [int(value) for value in model_contract.get(
      "selected_token_ids", [])]
  ridge_lambdas = [float(value) for value in model_contract.get(
      "ridge_lambda_grid", [])]
  predecessor_selects = (
      predecessor.get("required_checks_passed") is True
      and predecessor.get("dimension_selection_and_fit_allowed") is True
      and predecessor.get("validation_or_test_allowed") is False
      and predecessor.get("selected_next_route") == CURRENT_ROUTE
      and _has_candidate(routes, 583, CURRENT_ROUTE)
      and _has_switch(
          routes, 583,
          "select_router_prompt_distribution_final_norm_sparse_state_"
          "feature_dimension_fit_gate"))
  contract_locked = (
      len(results) == 12 and list(results) == fit_ids
      and len(token_ids) == 16 and len(set(token_ids)) == 16
      and model_contract.get("token_intercept_parameters") == 16
      and model_contract.get("shared_dimension_parameters") == 16
      and model_contract.get("maximum_parameters") == 32
      and model_contract.get("additional_model_variants_allowed") is False
      and ridge_lambdas == [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0])
  cv_rows = [
      _cross_validate(results, token_ids, ridge_lambda)
      for ridge_lambda in ridge_lambdas
  ]
  selected_cv = min(cv_rows, key=_objective)
  final_selection = _select_dimensions(
      results, set(token_ids), 16, excluded_case=None)
  final_dimensions = [int(row["dimension"])
                      for row in final_selection.get("dimensions", [])]
  selected_model = _fit(
      results, token_ids, final_dimensions,
      float(selected_cv["ridge_lambda"]))
  full_fit = _evaluate(results, token_ids, selected_model)
  nested_selection_passes = (
      len(cv_rows) == 6
      and all(row["case_count"] == 12 and row["position_count"] == 96
              for row in cv_rows)
      and all(
          len(fold.get("dimensions", [])) == 16
          and held_case not in fold.get("training_case_ids", [])
          and len(fold.get("training_case_ids", [])) == 11
          for row in cv_rows
          for held_case, fold in row["folds"].items())
      and final_selection.get("valid") is True
      and len(final_dimensions) == 16
      and len(set(final_dimensions)) == 16)
  model_shape = (
      selected_model["parameter_count"] == 32
      and len(selected_model["coefficients"]) == 32
      and selected_model["training_observation_count"] > 0
      and all(math.isfinite(float(value))
              for value in selected_model["coefficients"])
      and full_fit["case_count"] == 12
      and full_fit["position_count"] == 96
      and full_fit["valid_position_count"] == 96)
  checks = [
      {"name": "seq583_selected_nested_dimension_fit_gate",
       "pass": predecessor_selects},
      {"name": "fit_only_data_and_exact_16_plus_16_contract_match",
       "pass": contract_locked},
      {"name": "every_cv_fold_selects_16_dimensions_without_held_case",
       "pass": nested_selection_passes},
      {"name": "final_model_has_exactly_32_finite_parameters",
       "pass": model_shape},
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
  coefficients = [float(value) for value in selected_model["coefficients"]]
  model = {
      "schema_version": MODEL_SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "fit_sequence": args.sequence,
      "training_split": "fit",
      "training_case_ids": list(results),
      "fit_target": "selected16 probability-mass-matched oracle correction",
      "dimension_selection_target": (
          "selected16 probability-mass-matched oracle common-offset target"
      ),
      "dimension_selection_method": (
          "fit-only absolute-correlation screening; nested per held prompt"
      ),
      "runtime_native_oracle_required": False,
      "runtime_prompt_case_or_position_features_required": False,
      "runtime_features": ["gpu_final_norm_selected_dimensions"],
      "model_name": "token_intercepts_plus_shared_sparse_final_norm_score",
      "ridge_lambda": selected_model["ridge_lambda"],
      "parameter_count": 32,
      "selected_dimensions": final_selection["dimensions"],
      "parameters": {
          "token_intercepts": [
              {"token_id": token_id, "intercept": coefficients[index]}
              for index, token_id in enumerate(token_ids)
          ],
          "shared_dimension_coefficients": [
              {"dimension": dimension,
               "coefficient": coefficients[16 + index]}
              for index, dimension in enumerate(final_dimensions)
          ],
      },
      "runtime_full_vector_host_read_required": False,
      "validation_or_test_used_for_fit": False,
  }
  metrics = {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "sequence": args.sequence,
      "inputs": {
          "routes": _rel(args.routes),
          "predecessor": _rel(args.predecessor),
          "contract": _rel(args.contract),
          "fit_result_paths": result_paths,
      },
      "thresholds": {
          "max_kld": KLD_MAX,
          "top1_rate_min": TOP1_RATE_MIN,
          "per_case_kld_regression_epsilon": KLD_REGRESSION_EPSILON,
      },
      "ridge_selection": {
          "method": (
              "nested grouped leave-one-fit-prompt-out; minimize failed cases, "
              "top1 mismatches, max KLD, mean KLD, then prefer stronger ridge"
          ),
          "grid": cv_rows,
          "selected_cross_validation": selected_cv,
      },
      "final_dimension_selection": final_selection,
      "full_fit": full_fit,
      "model": model,
      "checks": checks,
      "required_checks_passed": required,
      "sparse_state_feasibility_passed": feasibility_passed,
      "validation_allowed": feasibility_passed,
      "test_allowed": False,
      "runtime_selected_dimension_source_allowed": False,
      "speedup_claims_allowed": False,
      "promotion_allowed": False,
      "disposition": (
          "accept_sparse_final_norm_dimension_fit"
          if feasibility_passed else
          "reject_sparse_final_norm_state_model_before_validation"),
      "selected_next_route": (
          VALIDATION_ROUTE if feasibility_passed else REJECTION_ROUTE),
      "next_route_reason": (
          "Nested fit-only dimension selection and the full 32-parameter fit "
          "both clear every fit case. Freeze dimensions/parameters and run "
          "exactly the six locked validation cases without model changes."
          if feasibility_passed else
          "The exact sparse final-norm state model fails before validation. "
          "Keep validation/test and runtime source sealed; close this learned "
          "correction route without dimension or model variants."),
  }
  return metrics, model


def write_outputs(metrics: dict[str, Any], model: dict[str, Any],
                  out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  (out_dir / "model.json").write_text(
      json.dumps(model, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  (out_dir / "manifest.json").write_text(
      json.dumps({
          "schema_version": metrics["schema_version"],
          "workstream": metrics["workstream"],
          "tool": _rel(Path(__file__)),
          "inputs": metrics["inputs"],
          "model": _rel(out_dir / "model.json"),
          "selected_dimensions": [
              row["dimension"] for row in model["selected_dimensions"]
          ],
          "ridge_lambda": model["ridge_lambda"],
          "parameter_count": 32,
          "sparse_state_feasibility_passed": metrics[
              "sparse_state_feasibility_passed"],
          "selected_next_route": metrics["selected_next_route"],
          "validation_allowed": metrics["validation_allowed"],
          "test_allowed": False,
          "runtime_selected_dimension_source_allowed": False,
          "speedup_claims_allowed": False,
      }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  failed = [row["name"] for row in metrics["checks"] if not row["pass"]]
  cv = metrics["ridge_selection"]["selected_cross_validation"]
  full = metrics["full_fit"]
  dimensions = [row["dimension"]
                for row in metrics["final_dimension_selection"]["dimensions"]]
  lines = [
      f"# Seq{metrics['sequence']} Sparse Final-Norm Dimension Fit Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- feasibility_passed: `{str(metrics['sparse_state_feasibility_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- selected lambda / parameters: `{model['ridge_lambda']}` / `32`",
      f"- selected dimensions: `{dimensions}`",
      f"- CV max KLD / top1 / failed cases: `{cv['corrected_max_kld']}` / "
      f"`{cv['corrected_top1_rate']}` / `{cv['failed_case_count']}`",
      f"- full-fit max KLD / top1 / failed cases: "
      f"`{full['corrected_max_kld']}` / `{full['corrected_top1_rate']}` / "
      f"`{full['failed_case_count']}`",
      f"- failed evidence checks: `{failed}`",
      "",
      metrics["next_route_reason"],
      "",
      "No validation case, test case, or runtime source was used.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--sequence", type=int, default=584)
  parser.add_argument("--routes", type=Path,
                      default=ACTIVE / "routes-ledger.json")
  parser.add_argument(
      "--predecessor", type=Path,
      default=ROOT / "output/seq583-final-norm-sparse-state-fit-collection-gate-20260710Tseq583Z/metrics.json")
  parser.add_argument(
      "--contract", type=Path,
      default=ACTIVE / "intel-qwen36-35b-a3b-gguf-q4km-final-norm-sparse-state-feature-contract-2026-07-10.json")
  parser.add_argument(
      "--out-dir", type=Path,
      default=ROOT / "output/seq584-final-norm-sparse-state-dimension-fit-gate-20260710Tseq584Z")
  args = parser.parse_args()
  metrics, model = compute(args)
  write_outputs(metrics, model, args.out_dir)
  cv = metrics["ridge_selection"]["selected_cross_validation"]
  print(json.dumps({
      "required_checks_passed": metrics["required_checks_passed"],
      "sparse_state_feasibility_passed": metrics[
          "sparse_state_feasibility_passed"],
      "disposition": metrics["disposition"],
      "selected_dimensions": [
          row["dimension"] for row in model["selected_dimensions"]
      ],
      "selected_ridge_lambda": model["ridge_lambda"],
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
