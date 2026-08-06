#!/usr/bin/env python3
"""Fit the locked 25-parameter whole-top8 shape model."""

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
TOOLS = ROOT / "tools"
FEATURE_TOOL = TOOLS / "intel-qwen36-state-conditioned-head-feature-reframe-gate.py"
SCHEMA_VERSION = "intel-qwen36-state-conditioned-head-top8-shape-fit-gate-v0"
MODEL_SCHEMA_VERSION = "intel-qwen36-state-conditioned-head-top8-shape-model-v0"
CURRENT_ROUTE = (
    "router_prompt_distribution_state_conditioned_head_correction_"
    "top8_shape_feature_fit_gate"
)
VALIDATION_ROUTE = (
    "router_prompt_distribution_state_conditioned_head_correction_"
    "validation_split_gate"
)
REJECTION_ROUTE = (
    "router_prompt_distribution_state_conditioned_head_correction_"
    "observable_route_close_gate"
)
RIDGE_LAMBDAS = (0.0001, 0.001, 0.01, 0.1, 1.0, 10.0)
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


FEATURE = _load_module(FEATURE_TOOL, "iq36_head_feature_reframe")


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


def _feature_vector(
    token_index: dict[int, int], token_id: int, rows: list[dict[str, Any]],
) -> list[float]:
  values = [0.0] * 25
  values[token_index[token_id]] = 1.0
  values[16] = float(next(
      row["gpu_logit_minus_top1"] for row in rows
      if int(row["token_id"]) == token_id))
  top1_logit = float(rows[0]["gpu_logit"])
  for rank in range(1, 8):
    values[16 + rank] = float(rows[rank]["gpu_logit"]) - top1_logit
  values[24] = top1_logit
  return values


def _training_rows(
    results: dict[str, dict[str, Any]], token_ids: list[int],
    excluded_case: str | None = None,
) -> list[tuple[list[float], float]]:
  selected = set(token_ids)
  token_index = {token_id: index for index, token_id in enumerate(token_ids)}
  training: list[tuple[list[float], float]] = []
  for case_id, result in results.items():
    if case_id == excluded_case:
      continue
    for step in _distribution(result).get("steps", []):
      if not isinstance(step, dict):
        continue
      rows = step.get("gpu_top8_fit_observables")
      targets = FEATURE._mass_matched_targets(step, selected)  # noqa: SLF001
      if not isinstance(rows, list) or len(rows) != 8 or targets is None:
        continue
      for row in rows:
        if not isinstance(row, dict):
          continue
        token_id = row.get("token_id")
        if isinstance(token_id, int) and token_id in targets:
          training.append((
              _feature_vector(token_index, token_id, rows),
              float(targets[token_id]),
          ))
  return training


def _fit(
    results: dict[str, dict[str, Any]], token_ids: list[int],
    ridge_lambda: float, excluded_case: str | None = None,
) -> dict[str, Any]:
  rows = _training_rows(results, token_ids, excluded_case)
  features = [row[0] for row in rows]
  targets = [row[1] for row in rows]
  coefficients = FEATURE._solve_ridge(  # noqa: SLF001
      features, targets, ridge_lambda)
  if coefficients is None:
    coefficients = [0.0] * 25
  squared_error = sum(
      (sum(value * coefficient
           for value, coefficient in zip(feature, coefficients, strict=True))
       - target) ** 2
      for feature, target in rows)
  return {
      "model_name": "token_intercepts_shared_whole_top8_shape",
      "ridge_lambda": ridge_lambda,
      "parameter_count": len(coefficients),
      "training_observation_count": len(rows),
      "training_rmse": math.sqrt(squared_error / len(rows)) if rows else None,
      "coefficients": coefficients,
  }


def _predictor(
    model: dict[str, Any], token_ids: list[int],
) -> Callable[[int, list[dict[str, Any]]], float]:
  token_index = {token_id: index for index, token_id in enumerate(token_ids)}
  coefficients = [float(value) for value in model["coefficients"]]

  def predict(token_id: int, rows: list[dict[str, Any]]) -> float:
    if token_id not in token_index:
      return 0.0
    feature = _feature_vector(token_index, token_id, rows)
    return sum(value * coefficient
               for value, coefficient in zip(feature, coefficients, strict=True))

  return predict


def _evaluate_step(
    step: dict[str, Any], selected_tokens: set[int],
    predict: Callable[[int, list[dict[str, Any]]], float],
) -> dict[str, Any]:
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
  mass_delta = 0.0
  native_weighted = 0.0
  corrected_logits: list[tuple[int, int, float]] = []
  for row in rows:
    if not isinstance(row, dict):
      return {"valid": False}
    token_id = int(row["token_id"])
    gpu_logit = float(row["gpu_logit"])
    correction = predict(token_id, rows) if token_id in selected_tokens else 0.0
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
    results: dict[str, dict[str, Any]], token_ids: list[int],
    ridge_lambda: float,
) -> dict[str, Any]:
  cases: dict[str, dict[str, Any]] = {}
  for held_case in results:
    model = _fit(results, token_ids, ridge_lambda, excluded_case=held_case)
    evaluation = _evaluate({held_case: results[held_case]}, token_ids, model)
    cases[held_case] = evaluation["cases"][held_case]
  positions = sum(int(row["position_count"]) for row in cases.values())
  top1_matches = sum(
      int(row["corrected_top1_match_count"]) for row in cases.values())
  return {
      "ridge_lambda": ridge_lambda,
      "parameter_count": 25,
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
      int(row["failed_case_count"]), top1_mismatches,
      float(row["corrected_max_kld"]),
      float(row["corrected_mean_kld"]),
      -float(row["ridge_lambda"]),
  )


def compute(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
  routes = _load(args.routes)
  predecessor = _load(args.predecessor)
  design = _load(args.design)
  oracle = _load(args.oracle_gate)
  corpus = _load(args.corpus_contract)
  fit_ids = [
      str(row["id"]) for row in corpus.get("prompts", [])
      if isinstance(row, dict) and row.get("split") == "fit"
  ]
  result_paths = oracle.get("inputs", {}).get("fit_result_paths", {})
  loaded = {
      case_id: _load(ROOT / path)
      for case_id, path in result_paths.items()
      if isinstance(case_id, str) and isinstance(path, str)
  }
  results = {case_id: loaded[case_id] for case_id in fit_ids
             if case_id in loaded}
  token_ids = [int(value) for value in design.get("selected_token_ids", [])]
  predecessor_selects = (
      predecessor.get("required_checks_passed") is True
      and predecessor.get("fit_allowed") is True
      and predecessor.get("selected_next_route") == CURRENT_ROUTE
      and _has_candidate(routes, 578, CURRENT_ROUTE)
      and _has_switch(
          routes, 578,
          "select_router_prompt_distribution_state_conditioned_head_"
          "correction_top8_shape_feature_fit_gate"))
  design_locked = (
      design.get("model_name") == "token_intercepts_shared_whole_top8_shape"
      and design.get("parameter_layout", {}).get("total") == 25
      and design.get("additional_model_variants_allowed") is False
      and len(token_ids) == 16 and len(set(token_ids)) == 16)
  data_fit_only = (
      len(results) == 12 and list(results) == fit_ids
      and set(loaded) == set(fit_ids))
  cv_rows = [
      _cross_validate(results, token_ids, ridge_lambda)
      for ridge_lambda in RIDGE_LAMBDAS
  ]
  selected_cv = min(cv_rows, key=_objective)
  selected_model = _fit(
      results, token_ids, float(selected_cv["ridge_lambda"]))
  full_fit = _evaluate(results, token_ids, selected_model)
  model_shape = (
      selected_model["parameter_count"] == 25
      and selected_model["training_observation_count"] > 0
      and len(selected_model["coefficients"]) == 25
      and all(math.isfinite(float(value))
              for value in selected_model["coefficients"]))
  evaluation_complete = (
      len(cv_rows) == len(RIDGE_LAMBDAS)
      and all(row["case_count"] == 12 and row["position_count"] == 96
              for row in cv_rows)
      and full_fit["case_count"] == 12
      and full_fit["position_count"] == 96
      and full_fit["valid_position_count"] == 96)
  checks = [
      {"name": "seq578_selected_exact_top8_shape_fit_gate",
       "pass": predecessor_selects},
      {"name": "design_is_exactly_the_locked_25_parameter_equation",
       "pass": design_locked},
      {"name": "only_the_12_locked_fit_cases_are_loaded",
       "pass": data_fit_only,
       "detail": {"fit_ids": fit_ids, "loaded_ids": list(results)}},
      {"name": "frozen_ridge_grid_and_exact_kld_evaluation_complete",
       "pass": evaluation_complete,
       "detail": {"ridge_lambdas": list(RIDGE_LAMBDAS)}},
      {"name": "selected_model_has_25_finite_parameters",
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
      "runtime_native_oracle_required": False,
      "runtime_prompt_case_or_position_features_required": False,
      "runtime_features": [
          "gpu_top8_token_ids", "gpu_top8_logits", "top1_margin"
      ],
      "model_name": "token_intercepts_shared_whole_top8_shape",
      "ridge_lambda": selected_model["ridge_lambda"],
      "parameter_count": 25,
      "parameters": {
          "token_intercepts": [
              {"token_id": token_id, "intercept": coefficients[index]}
              for index, token_id in enumerate(token_ids)
          ],
          "own_gap_coefficient": coefficients[16],
          "rank_gap_coefficients": [
              {"rank": rank, "coefficient": coefficients[16 + rank]}
              for rank in range(1, 8)
          ],
          "top1_logit_coefficient": coefficients[24],
      },
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
          "design": _rel(args.design),
          "oracle_gate": _rel(args.oracle_gate),
          "corpus_contract": _rel(args.corpus_contract),
          "fit_result_paths": result_paths,
      },
      "thresholds": {
          "max_kld": KLD_MAX,
          "top1_rate_min": TOP1_RATE_MIN,
          "per_case_kld_regression_epsilon": KLD_REGRESSION_EPSILON,
          "maximum_parameters": 32,
      },
      "ridge_selection": {
          "method": (
              "grouped leave-one-fit-prompt-out over the frozen lambda grid; "
              "minimize failed cases, top1 mismatches, max KLD, mean KLD, "
              "then prefer stronger ridge"
          ),
          "grid": cv_rows,
          "selected_cross_validation": selected_cv,
      },
      "full_fit": full_fit,
      "model": model,
      "checks": checks,
      "required_checks_passed": required,
      "top8_shape_feasibility_passed": feasibility_passed,
      "validation_allowed": feasibility_passed,
      "test_allowed": False,
      "runtime_source_allowed": False,
      "speedup_claims_allowed": False,
      "promotion_allowed": False,
      "disposition": (
          "accept_top8_shape_fit"
          if feasibility_passed else
          "reject_observable_only_head_correction_before_validation"),
      "selected_next_route": (
          VALIDATION_ROUTE if feasibility_passed else REJECTION_ROUTE),
      "next_route_reason": (
          "The exact whole-top8 shape model clears grouped fit-only CV and "
          "full fit. Freeze it and run exactly the six validation cases."
          if feasibility_passed else
          "The sole remaining <=32-parameter observable-only model fails "
          "before validation. Close this head-correction route without target "
          "source, validation/test, or speed work."),
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
          "ridge_lambda": model["ridge_lambda"],
          "parameter_count": model["parameter_count"],
          "top8_shape_feasibility_passed": metrics[
              "top8_shape_feasibility_passed"],
          "selected_next_route": metrics["selected_next_route"],
          "validation_allowed": metrics["validation_allowed"],
          "test_allowed": False,
          "runtime_source_allowed": False,
          "speedup_claims_allowed": False,
      }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  failed = [row["name"] for row in metrics["checks"] if not row["pass"]]
  cv = metrics["ridge_selection"]["selected_cross_validation"]
  full = metrics["full_fit"]
  lines = [
      f"# Seq{metrics['sequence']} Whole-Top8 Shape Fit Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- feasibility_passed: `{str(metrics['top8_shape_feasibility_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- selected lambda / parameters: `{model['ridge_lambda']}` / `25`",
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
  parser.add_argument("--sequence", type=int, default=579)
  parser.add_argument("--routes", type=Path,
                      default=ACTIVE / "routes-ledger.json")
  parser.add_argument(
      "--predecessor", type=Path,
      default=ROOT / "output/seq578-state-conditioned-head-post-feature-route-control-gate-20260710Tseq578Z/metrics.json")
  parser.add_argument(
      "--design", type=Path,
      default=ROOT / "output/seq578-state-conditioned-head-post-feature-route-control-gate-20260710Tseq578Z/design.json")
  parser.add_argument(
      "--oracle-gate", type=Path,
      default=ROOT / "output/seq576-state-conditioned-head-post-fit-route-control-gate-20260710Tseq576Z/metrics.json")
  parser.add_argument(
      "--corpus-contract", type=Path,
      default=ACTIVE / "intel-qwen36-35b-a3b-gguf-q4km-state-conditioned-head-correction-corpus-2026-07-10.json")
  parser.add_argument(
      "--out-dir", type=Path,
      default=ROOT / "output/seq579-state-conditioned-head-top8-shape-fit-gate-20260710Tseq579Z")
  args = parser.parse_args()
  metrics, model = compute(args)
  write_outputs(metrics, model, args.out_dir)
  cv = metrics["ridge_selection"]["selected_cross_validation"]
  print(json.dumps({
      "required_checks_passed": metrics["required_checks_passed"],
      "top8_shape_feasibility_passed": metrics[
          "top8_shape_feasibility_passed"],
      "disposition": metrics["disposition"],
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
