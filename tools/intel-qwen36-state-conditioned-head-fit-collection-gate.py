#!/usr/bin/env python3
"""Classify the locked fit split and freeze fit-only token candidates."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = "intel-qwen36-state-conditioned-head-fit-collection-gate-v0"
CURRENT_ROUTE = (
    "router_prompt_distribution_state_conditioned_head_correction_"
    "fit_split_collection_gate"
)
SELECTED_NEXT_ROUTE = (
    "router_prompt_distribution_state_conditioned_head_correction_"
    "fit_model_gate"
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
)
EXPECTED_INFRA_CHECKS = (
    "target_stage_created",
    "source_files_transferred",
    "generated_cpp_transferred",
    "token_inputs_transferred",
    "target_binary_built",
    "target_stdout_parsed",
)
CONTRIBUTOR_FIELDS = (
    "top_kld_contributors",
    "top_negative_kld_contributors",
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


def _binary(result: dict[str, Any]) -> dict[str, Any]:
  target = result.get("target")
  cache = target.get("cache") if isinstance(target, dict) else None
  binary = cache.get("binary") if isinstance(cache, dict) else None
  return binary if isinstance(binary, dict) else {}


def _run_returncode(result: dict[str, Any]) -> Any:
  target = result.get("target")
  run = target.get("run") if isinstance(target, dict) else None
  return run.get("returncode") if isinstance(run, dict) else None


def _checks_by_name(result: dict[str, Any]) -> dict[str, Any]:
  return {
      row["name"]: row.get("pass")
      for row in result.get("checks", [])
      if isinstance(row, dict) and isinstance(row.get("name"), str)
  }


def _finite(value: Any) -> bool:
  return isinstance(value, (int, float)) and math.isfinite(float(value))


def _observable_rows_valid(rows: Any) -> bool:
  if not isinstance(rows, list) or len(rows) != 8:
    return False
  token_ids: set[int] = set()
  previous_logit = math.inf
  for expected_rank, row in enumerate(rows):
    if not isinstance(row, dict) or row.get("rank") != expected_rank:
      return False
    token_id = row.get("token_id")
    if not isinstance(token_id, int) or token_id < 0 or token_id in token_ids:
      return False
    token_ids.add(token_id)
    numeric = (
        row.get("native_logit"), row.get("gpu_logit"),
        row.get("native_minus_gpu_logit"),
        row.get("gpu_logit_minus_top1"),
    )
    if not all(_finite(value) for value in numeric):
      return False
    gpu_logit = float(row["gpu_logit"])
    gap = float(row["gpu_logit_minus_top1"])
    if gpu_logit > previous_logit + 1e-6 or gap > 1e-6:
      return False
    if expected_rank == 0 and not math.isclose(
        gap, 0.0, rel_tol=0.0, abs_tol=1e-6):
      return False
    if not math.isclose(
        float(row["native_logit"]) - gpu_logit,
        float(row["native_minus_gpu_logit"]),
        rel_tol=0.0, abs_tol=2e-5):
      return False
    previous_logit = gpu_logit
  return True


def _row_contract(result: dict[str, Any], case_id: str) -> bool:
  dist = _distribution(result)
  steps = dist.get("steps")
  smoke = result.get("smoke")
  checks = _checks_by_name(result)
  return (
      result.get("case_id") == case_id
      and isinstance(smoke, dict)
      and smoke.get("case_id") == case_id
      and result.get("decode_tokens") == 8
      and all(result.get(name) is True for name in EXPECTED_TRUE_FLAGS)
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
      and result.get("state_conditioned_head_fit_observable_source") is False
      and result.get("state_conditioned_head_fit_observable_topk") == 8
      and result.get(
          "state_conditioned_head_fit_observable_distribution_only") is True
      and result.get("parse_error") is None
      and _binary(result).get("ok") is True
      and _run_returncode(result) in (0, 2)
      and all(checks.get(name) is True for name in EXPECTED_INFRA_CHECKS)
      and dist.get("position_count") == 8
      and isinstance(steps, list)
      and len(steps) == 8
      and all(_observable_rows_valid(
          step.get("gpu_top8_fit_observables")
          if isinstance(step, dict) else None) for step in steps))


def _aggregate_candidates(
    results: dict[str, dict[str, Any]], limit: int,
) -> tuple[list[dict[str, Any]], int]:
  scores: dict[int, float] = defaultdict(float)
  contributor_hits: dict[int, int] = defaultdict(int)
  top8_hits: dict[int, int] = defaultdict(int)
  contributor_rows = 0
  for result in results.values():
    for step in _distribution(result).get("steps", []):
      if not isinstance(step, dict):
        continue
      for row in step.get("gpu_top8_fit_observables", []):
        if isinstance(row, dict) and isinstance(row.get("token_id"), int):
          top8_hits[int(row["token_id"])] += 1
      for field in CONTRIBUTOR_FIELDS:
        for row in step.get(field, []):
          if not isinstance(row, dict):
            continue
          token_id = row.get("token_id")
          contribution = row.get("contribution")
          if not isinstance(token_id, int) or not _finite(contribution):
            continue
          scores[token_id] += abs(float(contribution))
          contributor_hits[token_id] += 1
          contributor_rows += 1
  ranked = sorted(scores, key=lambda token_id: (-scores[token_id], token_id))
  selected = [
      {
          "rank": rank,
          "token_id": token_id,
          "aggregate_absolute_kld_contribution": scores[token_id],
          "contributor_hits": contributor_hits[token_id],
          "gpu_top8_fit_observation_count": top8_hits[token_id],
      }
      for rank, token_id in enumerate(ranked[:limit])
  ]
  return selected, contributor_rows


def compute(args: argparse.Namespace) -> dict[str, Any]:
  routes = _load(args.routes)
  predecessor = _load(args.predecessor)
  corpus = _load(args.corpus_contract)
  prompts = [row for row in corpus.get("prompts", [])
             if isinstance(row, dict)]
  fit_ids = [str(row["id"]) for row in prompts if row.get("split") == "fit"]
  blocked_ids = {
      str(row["id"]) for row in prompts
      if row.get("split") in ("validation", "test")
  }
  result_paths = {
      case_id: args.output_root / (
          f"seq574-state-conditioned-head-fit-collection-{case_id}-"
          "20260710Tseq574Z/result.json")
      for case_id in fit_ids
  }
  discovered = sorted(args.output_root.glob(
      "seq574-state-conditioned-head-fit-collection-*-"
      "20260710Tseq574Z/result.json"))
  discovered_ids = {
      path.parent.name.removeprefix(
          "seq574-state-conditioned-head-fit-collection-").removesuffix(
              "-20260710Tseq574Z")
      for path in discovered
  }
  results = {
      case_id: _load(path) for case_id, path in result_paths.items()
      if path.is_file()
  }
  predecessor_selects = (
      predecessor.get("required_checks_passed") is True
      and predecessor.get("fit_split_collection_allowed") is True
      and predecessor.get("validation_or_test_allowed") is False
      and predecessor.get("model_fit_allowed") is False
      and predecessor.get("selected_next_route") == CURRENT_ROUTE
      and predecessor.get("fit_cases") == fit_ids
      and _has_candidate(routes, 573, CURRENT_ROUTE)
      and _has_switch(
          routes, 573,
          "select_router_prompt_distribution_state_conditioned_head_"
          "correction_fit_split_collection_gate"))
  split_integrity = (
      len(fit_ids) == 12
      and set(results) == set(fit_ids)
      and discovered_ids == set(fit_ids)
      and not discovered_ids.intersection(blocked_ids))
  row_contracts = {
      case_id: _row_contract(results[case_id], case_id)
      for case_id in fit_ids if case_id in results
  }
  total_steps = sum(
      len(_distribution(result).get("steps", []))
      for result in results.values())
  total_observations = sum(
      len(step.get("gpu_top8_fit_observables", []))
      for result in results.values()
      for step in _distribution(result).get("steps", [])
      if isinstance(step, dict))
  selected_tokens, contributor_rows = _aggregate_candidates(
      results, args.maximum_token_ids)
  candidate_selection_passes = (
      len(selected_tokens) == args.maximum_token_ids
      and len({row["token_id"] for row in selected_tokens})
      == args.maximum_token_ids
      and all(row["aggregate_absolute_kld_contribution"] > 0.0
              for row in selected_tokens))
  checks = [
      {"name": "seq573_selected_locked_fit_collection_gate",
       "pass": predecessor_selects},
      {"name": "only_all_12_locked_fit_cases_are_present",
       "pass": split_integrity,
       "detail": {"fit_ids": fit_ids,
                  "discovered_ids": sorted(discovered_ids)}},
      {"name": "all_rows_match_exact_26mask_top8_collection_contract",
       "pass": len(row_contracts) == 12 and all(row_contracts.values()),
       "detail": row_contracts},
      {"name": "collection_has_exactly_96_steps_and_768_observations",
       "pass": total_steps == 96 and total_observations == 768,
       "detail": {"steps": total_steps,
                  "top8_observations": total_observations}},
      {"name": "fit_only_kld_contributors_freeze_16_unique_tokens",
       "pass": candidate_selection_passes,
       "detail": {"contributor_rows": contributor_rows,
                  "selected_token_count": len(selected_tokens)}},
  ]
  required = all(bool(row["pass"]) for row in checks)
  case_summaries = {
      case_id: {
          "result": _rel(result_paths[case_id]),
          "target_run_returncode": _run_returncode(result),
          "wrapper_required_checks_passed": result.get(
              "required_checks_passed"),
          "position_count": _distribution(result).get("position_count"),
          "top1_rate": _distribution(result).get("top1_rate"),
          "max_kld": _distribution(result).get("max_kld"),
          "top8_observation_count": sum(
              len(step.get("gpu_top8_fit_observables", []))
              for step in _distribution(result).get("steps", [])
              if isinstance(step, dict)),
      }
      for case_id, result in results.items()
  }
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "sequence": args.sequence,
      "inputs": {
          "routes": _rel(args.routes),
          "predecessor": _rel(args.predecessor),
          "corpus_contract": _rel(args.corpus_contract),
          "result_paths": {case_id: _rel(path)
                           for case_id, path in result_paths.items()},
      },
      "fit_cases": case_summaries,
      "collection_totals": {
          "cases": len(results),
          "steps": total_steps,
          "top8_observations": total_observations,
          "contributor_rows": contributor_rows,
      },
      "candidate_token_selection": {
          "method": "fit-split aggregate absolute KLD contributors only",
          "maximum_token_ids": args.maximum_token_ids,
          "selected_tokens": selected_tokens,
      },
      "checks": checks,
      "required_checks_passed": required,
      "model_fit_allowed": required,
      "validation_or_test_allowed": False,
      "speedup_claims_allowed": False,
      "promotion_allowed": False,
      "disposition": (
          "accept_locked_fit_collection_and_token_selection"
          if required else "reject_locked_fit_collection"),
      "selected_next_route": SELECTED_NEXT_ROUTE if required else CURRENT_ROUTE,
      "next_route_reason": (
          "The locked fit split contains exactly 12 cases, 96 steps, and 768 "
          "well-formed GPU-top8 observations. Fit the frozen top-16 token "
          "ridge-affine model using fit data only, then run a no-target "
          "feasibility gate before validation."
          if required else
          "Repair split leakage, config drift, target evidence, observable "
          "shape, or candidate selection before fitting."),
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
      "collection_totals": metrics["collection_totals"],
      "candidate_token_selection": metrics["candidate_token_selection"],
      "selected_next_route": metrics["selected_next_route"],
      "validation_or_test_allowed": False,
      "speedup_claims_allowed": False,
  }
  (out_dir / "manifest.json").write_text(
      json.dumps(manifest, indent=2, sort_keys=True) + "\n",
      encoding="utf-8")
  failed = [row["name"] for row in metrics["checks"] if not row["pass"]]
  totals = metrics["collection_totals"]
  token_ids = [
      row["token_id"]
      for row in metrics["candidate_token_selection"]["selected_tokens"]
  ]
  lines = [
      f"# Seq{metrics['sequence']} State-Conditioned Fit Collection Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- cases / steps / observations: `{totals['cases']}` / "
      f"`{totals['steps']}` / `{totals['top8_observations']}`",
      f"- selected token IDs: `{token_ids}`",
      f"- failed_checks: `{failed}`",
      "",
      metrics["next_route_reason"],
      "",
      "Validation, test, speed, and promotion remain blocked.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--sequence", type=int, default=574)
  parser.add_argument("--routes", type=Path,
                      default=ACTIVE / "routes-ledger.json")
  parser.add_argument(
      "--predecessor", type=Path,
      default=ROOT / "output/seq573-state-conditioned-head-fit-observable-target-compile-gate-20260710Tseq573Z/metrics.json")
  parser.add_argument(
      "--corpus-contract", type=Path,
      default=ACTIVE / "intel-qwen36-35b-a3b-gguf-q4km-state-conditioned-head-correction-corpus-2026-07-10.json")
  parser.add_argument("--output-root", type=Path, default=ROOT / "output")
  parser.add_argument("--maximum-token-ids", type=int, default=16)
  parser.add_argument(
      "--out-dir", type=Path,
      default=ROOT / "output/seq574-state-conditioned-head-fit-collection-gate-20260710Tseq574Z")
  args = parser.parse_args()
  metrics = compute(args)
  write_outputs(metrics, args.out_dir)
  print(json.dumps({
      "required_checks_passed": metrics["required_checks_passed"],
      "disposition": metrics["disposition"],
      "collection_totals": metrics["collection_totals"],
      "selected_token_ids": [
          row["token_id"] for row in metrics[
              "candidate_token_selection"]["selected_tokens"]
      ],
      "model_fit_allowed": metrics["model_fit_allowed"],
      "selected_next_route": metrics["selected_next_route"],
      "out_dir": _rel(args.out_dir),
  }, sort_keys=True))
  return 0 if metrics["required_checks_passed"] else 2


if __name__ == "__main__":
  raise SystemExit(main())
