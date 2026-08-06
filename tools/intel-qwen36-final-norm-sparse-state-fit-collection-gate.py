#!/usr/bin/env python3
"""Classify the locked GPU final-norm fit collection."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = "intel-qwen36-final-norm-sparse-state-fit-collection-gate-v0"
CURRENT_ROUTE = (
    "router_prompt_distribution_final_norm_sparse_state_feature_fit_collection_gate"
)
SELECTED_NEXT_ROUTE = (
    "router_prompt_distribution_final_norm_sparse_state_feature_dimension_fit_gate"
)
EXPECTED_MASK = (
    "0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,24,25,26,28,29,30,"
    "33,34,36,37,38"
)
EXPECTED_TRUE_FLAGS = (
    "opencl_no_queue_profiling",
    "selected_shared_q4_gateup_combined",
    "selected_shared_q4_down_combined",
    "selected_shared_q6_down_combined",
    "defer_ffn_down_finish_bundle",
    "attention_front_output_projection_rowblock16",
    "shared_q4_runner",
    "resident_q4_weights",
    "resident_selected_q4_experts",
    "resident_selected_q6_experts",
    "resident_selected_q6_sorted_cache",
    "resident_selected_q6_rowstripe",
    "resident_shared_q6_down",
    "resident_full_attention_v_q6",
    "resident_linear_q6_qkv",
    "resident_q4_cpu_order_z",
    "resident_linear_conv_weights",
    "resident_linear_state",
    "resident_postconv_delta_handoff",
    "resident_norm_weights",
    "resident_gate_up_swiglu_handoff",
    "resident_attention_front_handoff",
    "resident_full_core_attention_front_handoff",
    "gpu_router",
    "gpu_lm_head_q6",
    "distribution_ladder",
    "teacher_force_native_tokens",
    "final_norm_sparse_state_fit_observable",
)
EXPECTED_INFRA_CHECKS = (
    "target_stage_created", "source_files_transferred",
    "generated_cpp_transferred", "token_inputs_transferred",
    "target_binary_built", "target_stdout_parsed",
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


def _distribution(result: dict[str, Any]) -> dict[str, Any]:
  smoke = result.get("smoke")
  if not isinstance(smoke, dict):
    return {}
  value = smoke.get("distribution_ladder")
  return value if isinstance(value, dict) else {}


def _run_returncode(result: dict[str, Any]) -> Any:
  target = result.get("target")
  run = target.get("run") if isinstance(target, dict) else None
  return run.get("returncode") if isinstance(run, dict) else None


def _binary(result: dict[str, Any]) -> dict[str, Any]:
  target = result.get("target")
  cache = target.get("cache") if isinstance(target, dict) else None
  binary = cache.get("binary") if isinstance(cache, dict) else None
  return binary if isinstance(binary, dict) else {}


def _checks_by_name(result: dict[str, Any]) -> dict[str, Any]:
  return {
      row["name"]: row.get("pass")
      for row in result.get("checks", [])
      if isinstance(row, dict) and isinstance(row.get("name"), str)
  }


def _vector_valid(value: Any) -> bool:
  return (
      isinstance(value, list) and len(value) == 2048
      and all(isinstance(item, (int, float)) and math.isfinite(float(item))
              for item in value))


def _row_contract(result: dict[str, Any], case_id: str) -> bool:
  dist = _distribution(result)
  steps = dist.get("steps")
  checks = _checks_by_name(result)
  smoke = result.get("smoke")
  return (
      result.get("case_id") == case_id
      and isinstance(smoke, dict) and smoke.get("case_id") == case_id
      and result.get("decode_tokens") == 8
      and all(result.get(name) is True for name in EXPECTED_TRUE_FLAGS)
      and result.get("final_norm_sparse_state_fit_observable_hidden_size")
      == 2048
      and result.get(
          "final_norm_sparse_state_fit_observable_distribution_only") is True
      and result.get(
          "final_norm_sparse_state_runtime_full_vector_host_read_allowed")
      is False
      and result.get("resident_selected_cache_topk") == 16
      and result.get("resident_session_repeats") == 1
      and result.get("attention_front_output_projection_rowblock16_layers")
      == EXPECTED_MASK
      and result.get("cpu_layer_fallback_layers") == ""
      and result.get("router_cpu_fallback_layers") == ""
      and result.get("native_residual_realign_layers") == ""
      and result.get("input_rmsnorm_serial_reduction_layers") == []
      and result.get("linear_output_projection_cpu_order_layers") == []
      and result.get("sparse_head_logit_bias_spec") == ""
      and result.get("sparse_head_logit_bias") == []
      and result.get("parse_error") is None
      and _binary(result).get("ok") is True
      and _run_returncode(result) in (0, 2)
      and all(checks.get(name) is True for name in EXPECTED_INFRA_CHECKS)
      and dist.get("position_count") == 8
      and isinstance(steps, list) and len(steps) == 8
      and all(
          isinstance(step, dict)
          and _vector_valid(step.get("gpu_final_norm_fit_observables"))
          and isinstance(step.get("gpu_top8_fit_observables"), list)
          and len(step["gpu_top8_fit_observables"]) == 8
          and "native_final_norm_fit_observables" not in step
          for step in steps))


def _unchanged(candidate: dict[str, Any], baseline: dict[str, Any]) -> bool:
  candidate_dist = _distribution(candidate)
  baseline_dist = _distribution(baseline)
  candidate_steps = candidate_dist.get("steps")
  baseline_steps = baseline_dist.get("steps")
  if (not isinstance(candidate_steps, list)
      or not isinstance(baseline_steps, list)
      or len(candidate_steps) != 8 or len(baseline_steps) != 8):
    return False
  if (candidate_dist.get("max_kld") != baseline_dist.get("max_kld")
      or candidate_dist.get("top1_rate") != baseline_dist.get("top1_rate")):
    return False
  for candidate_step, baseline_step in zip(
      candidate_steps, baseline_steps, strict=True):
    if (candidate_step.get("kld") != baseline_step.get("kld")
        or candidate_step.get("native_top1_id")
        != baseline_step.get("native_top1_id")
        or candidate_step.get("gpu_top1_id")
        != baseline_step.get("gpu_top1_id")
        or candidate_step.get("gpu_top8_fit_observables")
        != baseline_step.get("gpu_top8_fit_observables")):
      return False
  return True


def compute(args: argparse.Namespace) -> dict[str, Any]:
  routes = _load(args.routes)
  predecessor = _load(args.predecessor)
  contract = _load(args.contract)
  fit_ids = contract.get(
      "feature_collection_contract", {}).get("fit_diagnostic_case_ids", [])
  result_paths = {
      case_id: args.output_root / (
          f"seq583-final-norm-sparse-state-fit-collection-{case_id}-"
          "20260710Tseq583Z/result.json")
      for case_id in fit_ids
  }
  baseline_paths = {
      case_id: args.output_root / (
          f"seq574-state-conditioned-head-fit-collection-{case_id}-"
          "20260710Tseq574Z/result.json")
      for case_id in fit_ids
  }
  discovered = sorted(args.output_root.glob(
      "seq583-final-norm-sparse-state-fit-collection-*-"
      "20260710Tseq583Z/result.json"))
  discovered_ids = {
      path.parent.name.removeprefix(
          "seq583-final-norm-sparse-state-fit-collection-").removesuffix(
              "-20260710Tseq583Z")
      for path in discovered
  }
  results = {case_id: _load(path)
             for case_id, path in result_paths.items() if path.is_file()}
  baselines = {case_id: _load(path)
               for case_id, path in baseline_paths.items() if path.is_file()}
  predecessor_selects = (
      predecessor.get("required_checks_passed") is True
      and predecessor.get("fit_collection_allowed") is True
      and predecessor.get("dimension_selection_or_fit_allowed") is False
      and predecessor.get("validation_or_test_allowed") is False
      and predecessor.get("selected_next_route") == CURRENT_ROUTE
      and predecessor.get("fit_cases") == fit_ids
      and _has_candidate(routes, 582, CURRENT_ROUTE)
      and _has_switch(
          routes, 582,
          "select_router_prompt_distribution_final_norm_sparse_state_"
          "feature_fit_collection_gate"))
  split_integrity = (
      len(fit_ids) == 12 and set(results) == set(fit_ids)
      and set(baselines) == set(fit_ids)
      and discovered_ids == set(fit_ids))
  contracts = {case_id: _row_contract(results[case_id], case_id)
               for case_id in fit_ids if case_id in results}
  unchanged = {case_id: _unchanged(results[case_id], baselines[case_id])
               for case_id in fit_ids
               if case_id in results and case_id in baselines}
  total_steps = sum(len(_distribution(result).get("steps", []))
                    for result in results.values())
  total_vector_values = sum(
      len(step.get("gpu_final_norm_fit_observables", []))
      for result in results.values()
      for step in _distribution(result).get("steps", [])
      if isinstance(step, dict))
  total_top8_rows = sum(
      len(step.get("gpu_top8_fit_observables", []))
      for result in results.values()
      for step in _distribution(result).get("steps", [])
      if isinstance(step, dict))
  checks = [
      {"name": "seq582_selected_locked_final_norm_collection_gate",
       "pass": predecessor_selects},
      {"name": "only_all_12_locked_fit_cases_are_present",
       "pass": split_integrity,
       "detail": {"fit_ids": fit_ids,
                  "discovered_ids": sorted(discovered_ids)}},
      {"name": "all_rows_match_exact_26mask_gpu_vector_contract",
       "pass": len(contracts) == 12 and all(contracts.values()),
       "detail": contracts},
      {"name": "diagnostic_changes_no_kld_top1_or_top8_row",
       "pass": len(unchanged) == 12 and all(unchanged.values()),
       "detail": unchanged},
      {"name": "collection_has_96_vectors_196608_values_and_768_top8_rows",
       "pass": (total_steps == 96 and total_vector_values == 196608
                and total_top8_rows == 768),
       "detail": {"steps": total_steps,
                  "gpu_final_norm_values": total_vector_values,
                  "top8_rows": total_top8_rows}},
  ]
  required = all(bool(row["pass"]) for row in checks)
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "sequence": args.sequence,
      "inputs": {
          "routes": _rel(args.routes),
          "predecessor": _rel(args.predecessor),
          "contract": _rel(args.contract),
          "result_paths": {case_id: _rel(path)
                           for case_id, path in result_paths.items()},
          "baseline_paths": {case_id: _rel(path)
                             for case_id, path in baseline_paths.items()},
      },
      "collection_totals": {
          "cases": len(results),
          "steps": total_steps,
          "gpu_final_norm_values": total_vector_values,
          "top8_rows": total_top8_rows,
      },
      "case_summaries": {
          case_id: {
              "target_run_returncode": _run_returncode(result),
              "wrapper_required_checks_passed": result.get(
                  "required_checks_passed"),
              "max_kld": _distribution(result).get("max_kld"),
              "top1_rate": _distribution(result).get("top1_rate"),
              "vector_value_count": sum(
                  len(step.get("gpu_final_norm_fit_observables", []))
                  for step in _distribution(result).get("steps", [])
                  if isinstance(step, dict)),
          }
          for case_id, result in results.items()
      },
      "checks": checks,
      "required_checks_passed": required,
      "dimension_selection_and_fit_allowed": required,
      "validation_or_test_allowed": False,
      "runtime_selected_dimension_source_allowed": False,
      "speedup_claims_allowed": False,
      "promotion_allowed": False,
      "disposition": (
          "accept_locked_gpu_final_norm_fit_collection"
          if required else "reject_gpu_final_norm_fit_collection"),
      "selected_next_route": SELECTED_NEXT_ROUTE if required else CURRENT_ROUTE,
      "next_route_reason": (
          "The fit collection contains exactly 96 unchanged distribution steps, "
          "196608 finite GPU final-norm values, and 768 top8 target rows. Run "
          "the locked nested dimension-selection and 32-parameter fit next; "
          "validation/test and runtime source remain blocked."
          if required else
          "Repair split leakage, target evidence, vector shape, or no-math-change "
          "evidence before dimension selection."),
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
          "collection_totals": metrics["collection_totals"],
          "selected_next_route": metrics["selected_next_route"],
          "validation_or_test_allowed": False,
          "runtime_selected_dimension_source_allowed": False,
          "speedup_claims_allowed": False,
      }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  failed = [row["name"] for row in metrics["checks"] if not row["pass"]]
  totals = metrics["collection_totals"]
  lines = [
      f"# Seq{metrics['sequence']} GPU Final-Norm Fit Collection Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- cases / steps / vector values / top8 rows: "
      f"`{totals['cases']}` / `{totals['steps']}` / "
      f"`{totals['gpu_final_norm_values']}` / `{totals['top8_rows']}`",
      f"- failed_checks: `{failed}`",
      "",
      metrics["next_route_reason"],
      "",
      "Validation, test, runtime source, speed, and promotion remain blocked.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--sequence", type=int, default=583)
  parser.add_argument("--routes", type=Path,
                      default=ACTIVE / "routes-ledger.json")
  parser.add_argument(
      "--predecessor", type=Path,
      default=ROOT / "output/seq582-final-norm-sparse-state-feature-target-compile-gate-20260710Tseq582Z/metrics.json")
  parser.add_argument(
      "--contract", type=Path,
      default=ACTIVE / "intel-qwen36-35b-a3b-gguf-q4km-final-norm-sparse-state-feature-contract-2026-07-10.json")
  parser.add_argument("--output-root", type=Path, default=ROOT / "output")
  parser.add_argument(
      "--out-dir", type=Path,
      default=ROOT / "output/seq583-final-norm-sparse-state-fit-collection-gate-20260710Tseq583Z")
  args = parser.parse_args()
  metrics = compute(args)
  write_outputs(metrics, args.out_dir)
  print(json.dumps({
      "required_checks_passed": metrics["required_checks_passed"],
      "disposition": metrics["disposition"],
      "collection_totals": metrics["collection_totals"],
      "dimension_selection_and_fit_allowed": metrics[
          "dimension_selection_and_fit_allowed"],
      "selected_next_route": metrics["selected_next_route"],
      "out_dir": _rel(args.out_dir),
  }, sort_keys=True))
  return 0 if metrics["required_checks_passed"] else 2


if __name__ == "__main__":
  raise SystemExit(main())
