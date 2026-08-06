#!/usr/bin/env python3
"""Audit the current MoE routed-down fusion board after seq115.

This is route-control evidence only. It checks whether any already-proven
selected/shared down-to-tail or routed-MoE fusion shape can justify a decode
row, or whether the next work must move to the resident hidden-state carrier.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = "intel-qwen36-moe-routed-down-fusion-route-gate-v0"

DEFAULT_FRONTIER = ACTIVE / "frontier.json"
DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_ACCEPTED = ACTIVE / "accepted-cuts.json"
DEFAULT_REJECTED = ACTIVE / "rejected-routes.json"
DEFAULT_SEQ115 = ROOT / "output/offline-repack-streaming-layout-route-gate-20260707Tseq115Z/metrics.json"
DEFAULT_SEQ54 = ROOT / "output/resident-hidden-carrier-gate-20260706Tseq54Z/metrics.json"
DEFAULT_SEQ66 = ROOT / "output/down-tail-fusion-budget-20260706Tseq66Z/metrics.json"
DEFAULT_SEQ80 = ROOT / "output/nonatomic-down-tail-gate-20260707Tseq80Z/gate.json"
DEFAULT_SEQ82 = ROOT / "output/q6-nonatomic-down-tail-decode-gate-20260707Tseq82-r2Z/metrics.json"
DEFAULT_SEQ93 = ROOT / "output/q6-rowgroup-down-tail-gate-20260707Tseq93Z/metrics.json"
DEFAULT_WEIGHTED = ROOT / "output/gpu-selected-down-weighted-probe-20260704T142326Z/probe-result.json"
DEFAULT_WEIGHTED_SUM = ROOT / "output/gpu-selected-down-weighted-sum-probe-20260704T194045Z/probe-result.json"
DEFAULT_GROUP8 = ROOT / "output/gpu-selected-down-group8-sum-confirm-20260704T204225Z/probe-result.json"
DEFAULT_OUT_DIR = ROOT / "output/moe-routed-down-fusion-route-gate-20260707Tseq116Z"


def _load_json(path: Path) -> Any:
  return json.loads(path.read_text(encoding="utf-8"))


def _num(value: Any) -> float:
  return float(value) if isinstance(value, (int, float)) else 0.0


def _rel(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path)


def _nested(obj: Any, *keys: str) -> Any:
  cur = obj
  for key in keys:
    if not isinstance(cur, dict):
      return None
    cur = cur.get(key)
  return cur


def _frontier_state(frontier: dict[str, Any]) -> dict[str, Any]:
  anchor = frontier.get("goal_anchor")
  anchor = anchor if isinstance(anchor, dict) else {}
  budget = frontier.get("goal_budget")
  budget = budget if isinstance(budget, dict) else {}
  per_token = budget.get("per_token_ms")
  per_token = per_token if isinstance(per_token, dict) else {}
  verdict = budget.get("verdict")
  verdict = verdict if isinstance(verdict, dict) else {}
  no_progress = frontier.get("no_progress")
  no_progress = no_progress if isinstance(no_progress, dict) else {}
  noise = no_progress.get("noise")
  noise = noise if isinstance(noise, dict) else {}
  wall = _num(per_token.get("wall"))
  floor_budget = _num(verdict.get("floor_budget_ms_per_token"))
  return {
      "current_best_tps": _num(anchor.get("current_best_tps")),
      "floor_tps": _num(anchor.get("same_host_vulkan_floor_tps")),
      "wall_ms_per_token": wall,
      "floor_budget_ms_per_token": floor_budget,
      "floor_gap_ms_per_token": max(0.0, wall - floor_budget),
      "noise_rel": _num(noise.get("rel")),
      "runs_since_significant_improvement": no_progress.get(
          "runs_since_significant_improvement"),
      "soft_reflection_breached": no_progress.get("soft_reflection_breached"),
      "hard_stall_breached": no_progress.get("hard_stall_breached"),
  }


def _substage_gaps(frontier: dict[str, Any]) -> dict[str, float]:
  rows = _nested(frontier, "goal_budget", "substage_gap_estimates_ms_per_token")
  out: dict[str, float] = {}
  if not isinstance(rows, list):
    return out
  for row in rows:
    if (
        isinstance(row, dict)
        and isinstance(row.get("stage"), str)
        and isinstance(row.get("substage"), str)
    ):
      out[f"{row['stage']}.{row['substage']}"] = _num(row.get("gap_ms_per_token"))
  return out


def _stage_gap(frontier: dict[str, Any], stage: str) -> float:
  rows = _nested(frontier, "goal_budget", "stage_kernel_gap_estimates_ms_per_token")
  if not isinstance(rows, list):
    return 0.0
  for row in rows:
    if isinstance(row, dict) and row.get("stage") == stage:
      return _num(row.get("gap_ms_per_token"))
  return 0.0


def _accepted_ids(accepted: dict[str, Any]) -> set[str]:
  ids: set[str] = set()
  for row in accepted.get("accepted", []):
    if isinstance(row, dict) and isinstance(row.get("id"), str):
      ids.add(row["id"])
  return ids


def _rejected_names(rejected: dict[str, Any]) -> set[str]:
  names: set[str] = set()
  for row in rejected.get("rejected", []):
    if isinstance(row, dict) and isinstance(row.get("route"), str):
      names.add(row["route"])
  return names


def _parked_route(routes: dict[str, Any], route_id: str) -> dict[str, Any] | None:
  for row in routes.get("parked_routes", []):
    if isinstance(row, dict) and row.get("id") == route_id:
      return row
  return None


def _has_switch(routes: dict[str, Any], decision: str, seq_covered: int) -> bool:
  for row in routes.get("switch_decisions", []):
    if (
        isinstance(row, dict)
        and row.get("decision") == decision
        and _num(row.get("seq_covered")) >= seq_covered
        and row.get("resolved") is True
    ):
      return True
  return False


def compute(args: argparse.Namespace) -> dict[str, Any]:
  frontier = _load_json(args.frontier)
  routes = _load_json(args.routes)
  accepted = _load_json(args.accepted)
  rejected = _load_json(args.rejected)
  seq115 = _load_json(args.seq115)
  seq54 = _load_json(args.seq54)
  seq66 = _load_json(args.seq66)
  seq80 = _load_json(args.seq80)
  seq82 = _load_json(args.seq82)
  seq93 = _load_json(args.seq93)
  weighted = _load_json(args.weighted)
  weighted_sum = _load_json(args.weighted_sum)
  group8 = _load_json(args.group8)

  frontier_state = _frontier_state(frontier)
  floor_gap = frontier_state["floor_gap_ms_per_token"]
  stage_gap = _stage_gap(frontier, "selected_ffn")
  substage_gaps = _substage_gaps(frontier)
  accepted_ids = _accepted_ids(accepted)
  rejected_names = _rejected_names(rejected)
  moe_route = _parked_route(routes, "moe_routed_down_fusion")
  resident_route = _parked_route(routes, "resident_decode_loop_streaming")

  required_accepted = {
      "selected_q4_down_tail_handle",
      "ffn_down_finish_bundle_after_tailhandle",
      "selected_shared_q4_gateup_no_concat",
      "selected_shared_q6_down_combined",
      "selected_shared_q4q6_down_combined",
      "selected_shared_q6_down_combined_per_expert_cold_cache",
  }
  required_rejected = {
      "gpu_selected_q4_down_pair2_dual_expert_component",
      "gpu_q6_expert8_pair2_row_reuse",
      "gpu_selected_q4_down_weighted_tail_decode",
      "gpu_selected_q4_down_weighted_sum_component",
      "gpu_selected_q4_down_group8_weighted_sum_component",
      "gpu_down_tail_hidden_row_serial_fusion",
      "gpu_direct_q6_down_tail_atomic_from_q6_workitems",
      "gpu_q6_nonatomic_down_tail_decode_fusion",
      "gpu_selected_shared_q6_down_tail_rowgroup_local_reduce_component",
      "gpu_selected_q6_tail_fused_expert8_rowstripe",
      "gpu_selected_q6_down_tail_fused_rowstripe_expert8",
      "gpu_selected_q4_down_swiglu_f32input_decode",
      "gpu_selected_shared_q4_gateup_to_f32input_down_component",
  }
  missing_accepted = sorted(required_accepted - accepted_ids)
  missing_rejected = sorted(required_rejected - rejected_names)

  weighted_speedup = _num(weighted.get("candidate_weighted_speedup_vs_current"))
  weighted_sum_speedup = _num(
      weighted_sum.get("candidate_weighted_sum_speedup_vs_current"))
  group8_speedup = _num(group8.get("candidate_group8_sum_speedup_vs_current"))
  seq80_metrics = seq80.get("metrics") if isinstance(seq80.get("metrics"), dict) else {}
  seq80_ratio = _num(seq80_metrics.get("ratio_vs_combined_down"))
  seq82_combined_baseline = (
      _num(seq82.get("baseline_selected_ffn_ms_per_token"))
      + _num(seq82.get("baseline_ffn_tail_ms_per_token"))
      + _num(seq82.get("baseline_shared_ffn_ms_per_token"))
  )
  seq82_combined_candidate = (
      _num(seq82.get("candidate_selected_ffn_ms_per_token"))
      + _num(seq82.get("candidate_ffn_tail_ms_per_token"))
      + _num(seq82.get("candidate_shared_ffn_ms_per_token"))
  )
  seq82_combined_delta = seq82_combined_candidate - seq82_combined_baseline

  checks = [
      {
          "name": "seq115_selected_moe_route_gate",
          "pass": _has_switch(
              routes,
              "close_current_offline_repack_board_switch_to_moe_routed_down_fusion_gate",
              115,
          ),
      },
      {
          "name": "moe_route_available_and_frontier_still_below_floor",
          "pass": (
              moe_route is not None
              and frontier_state["current_best_tps"] < frontier_state["floor_tps"]
              and stage_gap > floor_gap
          ),
          "detail": {
              "route_id": None if moe_route is None else moe_route.get("id"),
              "rank": None if moe_route is None else moe_route.get("rank"),
              "selected_ffn_gap_ms_per_token": stage_gap,
              "floor_gap_ms_per_token": floor_gap,
          },
      },
      {
          "name": "selected_down_and_wait_gaps_are_floor_sized",
          "pass": (
              substage_gaps.get("selected_ffn.down_kernel_wait", 0.0) > floor_gap
              and substage_gaps.get("selected_ffn.down", 0.0) > floor_gap
          ),
          "detail": {
              "selected_ffn.down_kernel_wait": substage_gaps.get(
                  "selected_ffn.down_kernel_wait"),
              "selected_ffn.down": substage_gaps.get("selected_ffn.down"),
              "floor_gap_ms_per_token": floor_gap,
          },
      },
      {
          "name": "accepted_stack_already_has_down_tail_and_combined_down_cuts",
          "pass": not missing_accepted,
          "detail": {"missing": missing_accepted},
      },
      {
          "name": "closed_route_board_contains_current_moe_variants",
          "pass": not missing_rejected,
          "detail": {"missing": missing_rejected},
      },
      {
          "name": "weighted_sum_and_group8_components_are_not_material",
          "pass": (
              weighted.get("required_checks_passed") is True
              and weighted_sum.get("required_checks_passed") is True
              and group8.get("required_checks_passed") is True
              and weighted_speedup < 1.02
              and weighted_sum_speedup < 1.0
              and group8_speedup < 1.0
          ),
          "detail": {
              "weighted_speedup_vs_current": weighted_speedup,
              "weighted_sum_speedup_vs_current": weighted_sum_speedup,
              "group8_confirm_speedup_vs_current": group8_speedup,
          },
      },
      {
          "name": "hidden_row_serial_fusion_is_shape_closed",
          "pass": (
              _nested(seq66, "derived", "hidden_row_serial_q6_down_tail_fusion_closed")
              is True
              and _nested(seq66, "verdict", "naive_hidden_row_serial_fusion_promotable")
              is False
          ),
          "detail": {
              "minimum_shape": _nested(
                  seq66, "derived", "minimum_admissible_fusion_shape"),
          },
      },
      {
          "name": "nonatomic_component_was_exact_but_decode_regressed",
          "pass": (
              seq80.get("required_checks_passed") is True
              and seq80_ratio <= 1.0
              and seq82.get("required_checks_passed") is True
              and seq82.get("disposition")
                  == "rejected_q6_nonatomic_down_tail_decode_regresses_8tok"
              and seq82_combined_delta > 0.0
          ),
          "detail": {
              "seq80_component_ratio_vs_combined_down": seq80_ratio,
              "seq82_candidate_tps": seq82.get("candidate_tps"),
              "seq82_same_source_baseline_tps": seq82.get("same_source_baseline_tps"),
              "seq82_delta_pct_vs_frontier": seq82.get("delta_pct_vs_frontier"),
              "selected_shared_tail_delta_ms_per_token": seq82_combined_delta,
          },
      },
      {
          "name": "rowgroup_local_reduce_component_is_regressive",
          "pass": (
              seq93.get("component_exact") is True
              and seq93.get("decode_probe_allowed") is False
              and _num(seq93.get("ratio_vs_nonatomic")) > 3.0
              and _num(seq93.get("miss_vs_target_us")) > 0.0
          ),
          "detail": {
              "rowgroup_min_us": seq93.get("rowgroup_min_us"),
              "target_us": seq93.get("target_us"),
              "miss_vs_target_us": seq93.get("miss_vs_target_us"),
              "ratio_vs_nonatomic": seq93.get("ratio_vs_nonatomic"),
          },
      },
      {
          "name": "resident_hidden_state_carrier_is_the_remaining_contract_route",
          "pass": (
              resident_route is not None
              and _nested(
                  seq54,
                  "derived",
                  "resident_hidden_state_carrier_or_down_tail_fusion_required",
              )
              is True
          ),
          "detail": {
              "resident_route_id": None if resident_route is None else resident_route.get("id"),
              "seq54_next_route": _nested(seq54, "verdict", "next_route"),
          },
      },
  ]
  required_checks_passed = all(bool(row.get("pass")) for row in checks)

  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "inputs": {
          "frontier": _rel(args.frontier),
          "routes": _rel(args.routes),
          "accepted": _rel(args.accepted),
          "rejected": _rel(args.rejected),
          "seq115_offline_repack_gate": _rel(args.seq115),
          "seq54_resident_hidden_carrier_gate": _rel(args.seq54),
          "seq66_down_tail_budget": _rel(args.seq66),
          "seq80_nonatomic_component_gate": _rel(args.seq80),
          "seq82_nonatomic_decode_gate": _rel(args.seq82),
          "seq93_rowgroup_gate": _rel(args.seq93),
          "weighted_probe": _rel(args.weighted),
          "weighted_sum_probe": _rel(args.weighted_sum),
          "group8_confirm": _rel(args.group8),
      },
      "frontier": frontier_state,
      "moe_fusion_summary": {
          "selected_ffn_gap_ms_per_token": stage_gap,
          "selected_down_gap_ms_per_token": substage_gaps.get("selected_ffn.down"),
          "selected_down_wait_gap_ms_per_token": substage_gaps.get(
              "selected_ffn.down_kernel_wait"),
          "weighted_speedup_vs_current": weighted_speedup,
          "weighted_sum_speedup_vs_current": weighted_sum_speedup,
          "group8_confirm_speedup_vs_current": group8_speedup,
          "nonatomic_component_ratio_vs_combined_down": seq80_ratio,
          "nonatomic_decode_delta_pct_vs_frontier": seq82.get("delta_pct_vs_frontier"),
          "nonatomic_selected_shared_tail_delta_ms_per_token": seq82_combined_delta,
          "rowgroup_ratio_vs_nonatomic": seq93.get("ratio_vs_nonatomic"),
          "rowgroup_miss_vs_target_us": seq93.get("miss_vs_target_us"),
      },
      "checks": checks,
      "required_checks_passed": required_checks_passed,
      "decode_probe_allowed": False,
      "component_probe_allowed": False,
      "speedup_claims_allowed": False,
      "disposition": "close_current_moe_routed_down_fusion_board",
      "selected_next_route": "resident_hidden_state_carrier_contract_gate",
      "next_route_reason": (
          "Current routed-down fusion variants are exhausted: pair2, weighted "
          "tail, serial weighted-sum, group8 same-work aggregation, hidden-row "
          "serial fusion, atomic/non-atomic down-to-tail, rowgroup-local reduce, "
          "device-Q8/f32-input handoffs, and selected-Q6 fused tail shapes are "
          "closed, exact-but-subfloor, or decode-regressive. The remaining route "
          "is the resident hidden-state carrier contract identified by seq54."
      ),
      "next_action": (
          "Build resident_hidden_state_carrier_contract_gate as source/design "
          "evidence first. Do not launch a MoE/down-tail decode row without a "
          "new component that preserves selected/shared parallelism and lowers "
          "selected-FFN plus FFN-tail together."
      ),
  }


def write_outputs(metrics: dict[str, Any], out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
  )
  summary = [
      "# MoE Routed Down Fusion Route Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- decode_probe_allowed: `{str(metrics['decode_probe_allowed']).lower()}`",
      f"- component_probe_allowed: `{str(metrics['component_probe_allowed']).lower()}`",
      "",
      "## Summary",
      "",
      metrics["next_route_reason"],
      "",
      "## Next",
      "",
      metrics["next_action"],
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(summary), encoding="utf-8")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--frontier", type=Path, default=DEFAULT_FRONTIER)
  parser.add_argument("--routes", type=Path, default=DEFAULT_ROUTES)
  parser.add_argument("--accepted", type=Path, default=DEFAULT_ACCEPTED)
  parser.add_argument("--rejected", type=Path, default=DEFAULT_REJECTED)
  parser.add_argument("--seq115", type=Path, default=DEFAULT_SEQ115)
  parser.add_argument("--seq54", type=Path, default=DEFAULT_SEQ54)
  parser.add_argument("--seq66", type=Path, default=DEFAULT_SEQ66)
  parser.add_argument("--seq80", type=Path, default=DEFAULT_SEQ80)
  parser.add_argument("--seq82", type=Path, default=DEFAULT_SEQ82)
  parser.add_argument("--seq93", type=Path, default=DEFAULT_SEQ93)
  parser.add_argument("--weighted", type=Path, default=DEFAULT_WEIGHTED)
  parser.add_argument("--weighted-sum", type=Path, default=DEFAULT_WEIGHTED_SUM)
  parser.add_argument("--group8", type=Path, default=DEFAULT_GROUP8)
  parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
  return parser.parse_args()


def main() -> int:
  args = parse_args()
  metrics = compute(args)
  write_outputs(metrics, args.out_dir)
  print(json.dumps({
      "required_checks_passed": metrics["required_checks_passed"],
      "disposition": metrics["disposition"],
      "selected_next_route": metrics["selected_next_route"],
      "out_dir": _rel(args.out_dir),
  }, sort_keys=True))
  return 0 if metrics["required_checks_passed"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
