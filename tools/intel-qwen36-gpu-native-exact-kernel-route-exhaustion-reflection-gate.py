#!/usr/bin/env python3
"""Audit exhausted correctness routes and select explicit route rejection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = "intel-qwen36-gpu-native-route-exhaustion-reflection-v0"
CURRENT_ROUTE = "gpu_native_exact_kernel_route_exhaustion_reflection_gate"
SELECTED_NEXT_ROUTE = "gpu_native_product_route_rejection_gate"

REQUIRED_CLOSED_ROUTES = {
    "router_prompt_distribution_static_sparse_head_logit_bias",
    "router_prompt_distribution_early_precision_island_local_correction_board",
    "router_prompt_distribution_learned_head_correction_board",
    "router_prompt_distribution_all_linear_fused_exact_projection_and_serial_input_rmsnorm",
    "router_prompt_distribution_layer12_ffn_local_math_source",
    "router_prompt_distribution_cpuorder_preprojection_bundle_v1",
    "gpu_vulkan_precise_postconv_recurrent_v1",
    "level_zero_ocloc_fused_postconv_recurrent_v1",
    "level_zero_ocloc_cr_recip_postconv_recurrent_v2",
}


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


def _candidate_selects(routes: dict[str, Any], seq: int,
                       next_route: str) -> bool:
  return any(
      isinstance(row, dict)
      and row.get("seq") == seq
      and row.get("selected_next_route") == next_route
      for row in routes.get("candidate_history", []))


def _switch_selects(routes: dict[str, Any], seq: int,
                    decision: str) -> bool:
  return any(
      isinstance(row, dict)
      and row.get("seq_covered") == seq
      and row.get("decision") == decision
      and row.get("resolved") is True
      for row in routes.get("switch_decisions", []))


def _closed_routes(rejected: dict[str, Any]) -> dict[str, dict[str, Any]]:
  return {
      str(row["route"]): row
      for row in rejected.get("rejected", [])
      if isinstance(row, dict) and isinstance(row.get("route"), str)
  }


def _distribution_rows(calibration: dict[str, Any],
                       holdout: dict[str, Any]) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  for split, payload in (
      ("calibration", calibration.get("calibration", {})),
      ("holdout", holdout.get("holdouts", {})),
  ):
    if not isinstance(payload, dict):
      continue
    for case_id, row in payload.items():
      if isinstance(row, dict):
        rows.append({
            "split": split,
            "case_id": case_id,
            "max_kld": row.get("max_kld"),
            "top1_rate": row.get("top1_rate"),
        })
  return rows


def compute(args: argparse.Namespace) -> dict[str, Any]:
  routes = _load(args.routes)
  rejected = _load(args.rejected)
  acceptance = _load(args.acceptance)
  predecessor = _load(args.predecessor)
  calibration = _load(args.calibration)
  holdout = _load(args.holdout)
  learned_close = _load(args.learned_close)
  fused_exact_close = _load(args.fused_exact_close)
  opencl_close = _load(args.opencl_close)
  vulkan_close = _load(args.vulkan_close)
  level_zero_v1_close = _load(args.level_zero_v1_close)
  closed = _closed_routes(rejected)

  accuracy = acceptance.get("accuracy", {})
  distribution = accuracy.get("teacher_forced_distribution", {})
  tokens = accuracy.get("tokens", {})
  max_kld = float(distribution.get("kl_divergence_max", -1.0))
  top1_min = float(distribution.get("top1_min", -1.0))
  rows = _distribution_rows(calibration, holdout)
  absolute_distribution_passes = (
      len(rows) == 6
      and len({row["case_id"] for row in rows}) == 6
      and all(float(row["max_kld"]) <= max_kld for row in rows)
      and all(float(row["top1_rate"]) >= top1_min for row in rows))

  static_rejection = closed.get(
      "router_prompt_distribution_static_sparse_head_logit_bias", {})
  static_route_remains_closed = (
      holdout.get("required_checks_passed") is True
      and holdout.get("holdout_contract_passed") is False
      and holdout.get("no_kld_regression") is False
      and holdout.get("failed_holdout_cases") == ["short_transform_003"]
      and holdout.get("holdouts", {}).get(
          "short_transform_003", {}).get("kld_delta", 0.0) > 1e-7
      and "state-conditioned product source" in
      static_rejection.get("reopen_condition", ""))

  exact_gate_stops_hold = (
      fused_exact_close.get("required_checks_passed") is True
      and fused_exact_close.get("fused_exact_projection_route_closed") is True
      and opencl_close.get("required_checks_passed") is True
      and opencl_close.get("bundle_closed") is True
      and vulkan_close.get("required_checks_passed") is True
      and vulkan_close.get("vulkan_component_closed") is True
      and level_zero_v1_close.get("required_checks_passed") is True
      and level_zero_v1_close.get("level_zero_component_closed") is True
      and predecessor.get("required_checks_passed") is True
      and predecessor.get("primitive_route_closed") is True)

  missing_closed_routes = sorted(REQUIRED_CLOSED_ROUTES - set(closed))
  predecessor_selects = (
      predecessor.get("selected_next_route") == CURRENT_ROUTE
      and _candidate_selects(routes, 624, CURRENT_ROUTE)
      and _switch_selects(
          routes, 624,
          "select_gpu_native_exact_kernel_route_exhaustion_reflection_gate"))
  canonical_contract_is_locked = (
      max_kld == 0.005
      and top1_min == 0.99
      and tokens.get("deterministic_greedy_exact_match_required") is True
      and tokens.get("first_divergence_blocks_promotion") is True
      and tokens.get("min_prompt_cases") == 3)
  learned_route_requires_owner = (
      learned_close.get("required_checks_passed") is True
      and learned_close.get("learned_correction_routes_closed") is True
      and "owner-approved new data/model contract" in closed.get(
          "router_prompt_distribution_learned_head_correction_board", {}
      ).get("reopen_condition", ""))
  no_runtime_evidence = not any(
      (args.out_dir / name).exists()
      for name in ("run.json", "probe.json", "tokens.jsonl"))

  checks = [
      {"name": "seq624_selected_no_target_route_exhaustion_reflection",
       "pass": predecessor_selects},
      {"name": "canonical_distribution_and_token_contract_is_locked",
       "pass": canonical_contract_is_locked},
      {"name": "static_bias_passes_absolute_distribution_ruler",
       "pass": absolute_distribution_passes,
       "detail": rows},
      {"name": "pre_registered_holdout_no_regression_still_rejects_static_bias",
       "pass": static_route_remains_closed},
      {"name": "learned_correction_requires_owner_approved_new_contract",
       "pass": learned_route_requires_owner},
      {"name": "all_recorded_exact_kernel_stop_conditions_are_terminal",
       "pass": exact_gate_stops_hold},
      {"name": "required_correction_and_exact_routes_are_rejected",
       "pass": not missing_closed_routes,
       "detail": {"missing": missing_closed_routes}},
      {"name": "reflection_created_no_runtime_evidence",
       "pass": no_runtime_evidence},
  ]
  required = all(bool(row["pass"]) for row in checks)
  alternatives = [
      {
          "rank": 1,
          "route": "owner_approved_new_correctness_data_runtime_contract",
          "status": "reopen_only",
          "reason": (
              "Requires a new independent untouched evaluation split and "
              "runtime budget; the current learned board cannot supply it."),
      },
      {
          "rank": 2,
          "route": "static_bias_absolute_threshold_exception",
          "status": "rejected_post_hoc_contract_relaxation",
          "reason": (
              "All six rows pass the absolute ruler, but waiving the observed "
              "pre-registered holdout regression would change the protocol "
              "after results are known."),
      },
      {
          "rank": 3,
          "route": "additional_exact_kernel_arithmetic_variants",
          "status": "rejected_by_recorded_stops",
          "reason": (
              "Projection, OpenCL, Vulkan, Level Zero, reciprocal, and local "
              "arithmetic variants have terminal evidence and explicit reopen "
              "conditions that are not met."),
      },
  ]
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "sequence": args.sequence,
      "inputs": {
          "routes": _rel(args.routes),
          "rejected": _rel(args.rejected),
          "acceptance": _rel(args.acceptance),
          "predecessor": _rel(args.predecessor),
          "calibration": _rel(args.calibration),
          "holdout": _rel(args.holdout),
          "learned_close": _rel(args.learned_close),
          "fused_exact_close": _rel(args.fused_exact_close),
          "opencl_close": _rel(args.opencl_close),
          "vulkan_close": _rel(args.vulkan_close),
          "level_zero_v1_close": _rel(args.level_zero_v1_close),
      },
      "alternatives": alternatives,
      "checks": checks,
      "required_checks_passed": required,
      "current_contract_has_admissible_correctness_successor": False,
      "owner_contract_change_required_for_reopen": True,
      "route_rejection_allowed": required,
      "source_repair_allowed": False,
      "target_compile_allowed": False,
      "component_probe_allowed": False,
      "decode_integration_allowed": False,
      "token_row_allowed": False,
      "validation_or_test_allowed": False,
      "speedup_claims_allowed": False,
      "promotion_allowed": False,
      "disposition": (
          "close_exact_kernel_board_select_product_route_rejection"
          if required else "repair_route_exhaustion_audit"),
      "selected_next_route": (
          SELECTED_NEXT_ROUTE if required else CURRENT_ROUTE),
      "next_route_reason": (
          "The current contract has no admissible correction successor. "
          "Record the goal's explicit route rejection without target work; "
          "reopen only through an owner-approved contract or a new boundary "
          "that satisfies a recorded reopen condition."
          if required else
          "Repair closed-route, stop-condition, or contract evidence before "
          "selecting a terminal disposition."),
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
          "selected_next_route": metrics["selected_next_route"],
          "route_rejection_allowed": metrics["route_rejection_allowed"],
          "target_compile_allowed": False,
          "token_row_allowed": False,
      }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  failed = [row["name"] for row in metrics["checks"] if not row["pass"]]
  lines = [
      f"# Seq{metrics['sequence']} Exact-Kernel Route Exhaustion Reflection",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- failed_checks: `{failed}`",
      "",
  ]
  for row in metrics["alternatives"]:
    lines.append(
        f"- rank {row['rank']} `{row['route']}`: `{row['status']}` - "
        f"{row['reason']}")
  lines += [
      "",
      metrics["next_route_reason"],
      "",
      "This gate used existing evidence only; no target command ran.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--sequence", type=int, default=625)
  parser.add_argument("--routes", type=Path,
                      default=ACTIVE / "routes-ledger.json")
  parser.add_argument("--rejected", type=Path,
                      default=ACTIVE / "rejected-routes.json")
  parser.add_argument(
      "--acceptance", type=Path,
      default=ROOT / "benchmarks" / WORKSTREAM / "acceptance-matrix.json")
  parser.add_argument(
      "--predecessor", type=Path,
      default=ROOT / (
          "output/seq624-gpu-software-correctly-rounded-qk-scale-primitive-"
          "route-close-gate-20260710Tseq624Z/metrics.json"))
  parser.add_argument(
      "--calibration", type=Path,
      default=ROOT / (
          "output/seq567-sparse-head-logit-bias-calibration-gate-"
          "20260710Tseq567Z/metrics.json"))
  parser.add_argument(
      "--holdout", type=Path,
      default=ROOT / (
          "output/seq568-sparse-head-logit-bias-holdout-gate-"
          "20260710Tseq568Z/metrics.json"))
  parser.add_argument(
      "--learned-close", type=Path,
      default=ROOT / (
          "output/seq585-learned-correction-route-close-gate-"
          "20260710Tseq585Z/metrics.json"))
  parser.add_argument(
      "--fused-exact-close", type=Path,
      default=ROOT / (
          "output/seq594-fused-exact-linear-projection-route-close-gate-"
          "20260710Tseq594Z/metrics.json"))
  parser.add_argument(
      "--opencl-close", type=Path,
      default=ROOT / (
          "output/seq605-all-linear-preprojection-parity-component-final-"
          "route-close-gate-20260710Tseq605Z/metrics.json"))
  parser.add_argument(
      "--vulkan-close", type=Path,
      default=ROOT / (
          "output/seq612-gpu-vulkan-postconv-recurrent-component-route-"
          "close-gate-20260710Tseq612Z/metrics.json"))
  parser.add_argument(
      "--level-zero-v1-close", type=Path,
      default=ROOT / (
          "output/seq618-gpu-level-zero-postconv-recurrent-component-route-"
          "close-gate-20260710Tseq618Z/metrics.json"))
  parser.add_argument(
      "--out-dir", type=Path,
      default=ROOT / (
          "output/seq625-gpu-native-exact-kernel-route-exhaustion-"
          "reflection-gate-20260710Tseq625Z"))
  args = parser.parse_args()
  metrics = compute(args)
  write_outputs(metrics, args.out_dir)
  print(json.dumps({
      "required_checks_passed": metrics["required_checks_passed"],
      "route_rejection_allowed": metrics["route_rejection_allowed"],
      "disposition": metrics["disposition"],
      "selected_next_route": metrics["selected_next_route"],
      "out_dir": _rel(args.out_dir),
  }, sort_keys=True))
  return 0 if metrics["required_checks_passed"] else 2


if __name__ == "__main__":
  raise SystemExit(main())
