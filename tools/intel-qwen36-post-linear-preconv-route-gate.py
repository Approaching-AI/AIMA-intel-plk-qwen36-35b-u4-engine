#!/usr/bin/env python3
"""Select the next route after closing shared-Q8 linear-preconv qkv/conv.

This is route-control evidence only. It re-ranks the remaining full-core,
attention-front, selected-FFN, and linear-delta gaps after seq90 closed the
shared-Q8 preconv qkv/conv speed route, without launching another decode row.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = "intel-qwen36-post-linear-preconv-route-gate-v0"
DEFAULT_FRONTIER = ACTIVE / "frontier.json"
DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_REJECTED = ACTIVE / "rejected-routes.json"
DEFAULT_SEQ53 = ROOT / "output/q6-defer-drain-budget-20260706Tseq53Z/metrics.json"
DEFAULT_SEQ66 = ROOT / "output/down-tail-fusion-budget-20260706Tseq66Z/metrics.json"
DEFAULT_SEQ82 = ROOT / "output/q6-nonatomic-down-tail-decode-gate-20260707Tseq82-r2Z/metrics.json"
DEFAULT_SEQ89 = ROOT / "output/post-top16-gateup-route-gate-20260707Tseq89Z/metrics.json"
DEFAULT_SEQ90_Q4 = ROOT / "output/linear-preconv-qkv-conv-root-probe-20260707Tseq90Z/metrics.json"
DEFAULT_SEQ90_Q6 = ROOT / "output/linear-preconv-qkv-conv-root-probe-20260707Tseq90-layer0Z/metrics.json"
DEFAULT_OUT_DIR = ROOT / "output/post-linear-preconv-route-gate-20260707Tseq91Z"


def _load_json(path: Path) -> Any:
  return json.loads(path.read_text(encoding="utf-8"))


def _rel(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path)


def _num(value: Any) -> float:
  return float(value) if isinstance(value, (int, float)) else 0.0


def _nested(obj: Any, *keys: str) -> Any:
  cur = obj
  for key in keys:
    if not isinstance(cur, dict):
      return None
    cur = cur.get(key)
  return cur


def _frontier_summary(frontier: dict[str, Any]) -> dict[str, Any]:
  no_progress = frontier.get("no_progress")
  no_progress = no_progress if isinstance(no_progress, dict) else {}
  budget = frontier.get("goal_budget")
  budget = budget if isinstance(budget, dict) else {}
  per_token = budget.get("per_token_ms")
  per_token = per_token if isinstance(per_token, dict) else {}
  verdict = budget.get("verdict")
  verdict = verdict if isinstance(verdict, dict) else {}
  noise = no_progress.get("noise")
  noise = noise if isinstance(noise, dict) else {}
  wall = _num(per_token.get("wall"))
  floor_budget = _num(verdict.get("floor_budget_ms_per_token"))
  return {
      "current_best_tps": _num(_nested(no_progress, "last_significant_improvement", "tps")),
      "floor_tps": _num(verdict.get("floor_tps")),
      "wall_ms_per_token": wall,
      "floor_budget_ms_per_token": floor_budget,
      "floor_gap_ms_per_token": max(0.0, wall - floor_budget),
      "noise_rel": _num(noise.get("rel")),
      "runs_since_significant_improvement": no_progress.get(
          "runs_since_significant_improvement"),
      "soft_reflection_breached": no_progress.get("soft_reflection_breached"),
      "hard_stall_breached": no_progress.get("hard_stall_breached"),
  }


def _stage_gaps(frontier: dict[str, Any]) -> dict[str, float]:
  budget = frontier.get("goal_budget")
  budget = budget if isinstance(budget, dict) else {}
  out: dict[str, float] = {}
  for row in budget.get("stage_kernel_gap_estimates_ms_per_token", []):
    if isinstance(row, dict) and isinstance(row.get("stage"), str):
      out[row["stage"]] = _num(row.get("gap_ms_per_token"))
  return out


def _substage_gaps(frontier: dict[str, Any]) -> dict[str, float]:
  budget = frontier.get("goal_budget")
  budget = budget if isinstance(budget, dict) else {}
  out: dict[str, float] = {}
  for row in budget.get("substage_gap_estimates_ms_per_token", []):
    if (
        isinstance(row, dict)
        and isinstance(row.get("stage"), str)
        and isinstance(row.get("substage"), str)
    ):
      out[f"{row['stage']}.{row['substage']}"] = _num(row.get("gap_ms_per_token"))
  return out


def _route_names(rejected: dict[str, Any]) -> set[str]:
  names: set[str] = set()
  for row in rejected.get("rejected", []):
    if isinstance(row, dict) and isinstance(row.get("route"), str):
      names.add(row["route"])
  return names


def _has_candidate(routes: dict[str, Any], seq: int, disposition: str) -> bool:
  for row in routes.get("candidate_history", []):
    if (
        isinstance(row, dict)
        and row.get("seq") == seq
        and row.get("disposition") == disposition
    ):
      return True
  return False


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


def _largest(mapping: dict[str, float]) -> tuple[str, float]:
  if not mapping:
    return "", 0.0
  key = max(mapping, key=lambda item: mapping[item])
  return key, mapping[key]


def compute(args: argparse.Namespace) -> dict[str, Any]:
  frontier = _load_json(args.frontier)
  routes = _load_json(args.routes)
  rejected = _load_json(args.rejected)
  seq53 = _load_json(args.q6_defer_drain)
  seq66 = _load_json(args.down_tail_fusion)
  seq82 = _load_json(args.q6_nonatomic_decode)
  seq89 = _load_json(args.post_top16)
  seq90_q4 = _load_json(args.seq90_q4)
  seq90_q6 = _load_json(args.seq90_q6)

  frontier_state = _frontier_summary(frontier)
  stage_gaps = _stage_gaps(frontier)
  substage_gaps = _substage_gaps(frontier)
  stage, stage_gap = _largest(stage_gaps)
  substage, substage_gap = _largest(substage_gaps)
  rejected_names = _route_names(rejected)
  floor_gap = frontier_state["floor_gap_ms_per_token"]

  q4_seq90 = seq90_q4.get("derived", {}) if isinstance(seq90_q4, dict) else {}
  q6_seq90 = seq90_q6.get("derived", {}) if isinstance(seq90_q6, dict) else {}
  seq53_derived = seq53.get("derived", {}) if isinstance(seq53, dict) else {}
  seq53_verdict = seq53.get("verdict", {}) if isinstance(seq53, dict) else {}
  seq66_verdict = seq66.get("verdict", {}) if isinstance(seq66, dict) else {}

  required_closed_routes = [
      "gpu_linear_preconv_shared_q8_qkv_conv_root_component",
      "gpu_linear_preconv_shared_q8_preconv_bundle_decode",
      "gpu_attention_linear_event_lifetime_combined_alias",
      "gpu_attention_linear_simple_final_output_handoffs",
      "gpu_selected_gateup_top16_indexed_material_component",
      "gpu_selected_shared_q6_down_occupancy4_component",
      "gpu_q6_defer_finish_without_tail_drain_elimination",
      "gpu_ffn_tail_resident_input_noqueue",
      "gpu_ffn_tail_plus_attention_residual_carrier_noqueue",
      "gpu_down_tail_hidden_row_serial_fusion",
      "gpu_ffn_tail_atomic_reduction_from_down_handles",
      "gpu_direct_q6_down_tail_atomic_from_q6_workitems",
      "gpu_q6_nonatomic_down_tail_decode_fusion",
  ]
  missing_closed_routes = [
      route for route in required_closed_routes if route not in rejected_names
  ]

  checks = [
      {
          "name": "frontier_still_below_floor",
          "pass": frontier_state["wall_ms_per_token"]
          > frontier_state["floor_budget_ms_per_token"]
          > 0.0,
      },
      {
          "name": "seq90_shared_q8_qkv_conv_closed_by_switch",
          "pass": _has_switch(
              routes,
              "close_linear_preconv_shared_q8_qkv_conv_root_as_speed_route",
              90,
          ),
      },
      {
          "name": "seq90_component_correct_but_not_floor_covering",
          "pass": (
              bool(q4_seq90.get("required_checks_passed"))
              and bool(q6_seq90.get("required_checks_passed"))
              and not bool(q4_seq90.get("component_delta_floor_covering"))
              and not bool(q6_seq90.get("component_delta_floor_covering"))
              and not bool(q4_seq90.get("component_qkv_conv_non_growth_or_bounded"))
              and not bool(q6_seq90.get("component_qkv_conv_non_growth_or_bounded"))
          ),
      },
      {
          "name": "seq89_linear_preconv_root_consumed",
          "pass": seq89.get("required_checks_passed") is True
          and seq89.get("disposition") == "select_linear_preconv_qkv_conv_root_component_gate",
      },
      {
          "name": "selected_ffn_is_largest_remaining_stage_gap",
          "pass": stage == "selected_ffn" and stage_gap > floor_gap,
          "detail": {
              "largest_stage": stage,
              "largest_stage_gap_ms_per_token": stage_gap,
              "floor_gap_ms_per_token": floor_gap,
          },
      },
      {
          "name": "selected_down_wait_is_largest_substage_gap",
          "pass": substage == "selected_ffn.down_kernel_wait" and substage_gap > floor_gap,
          "detail": {
              "largest_substage": substage,
              "largest_substage_gap_ms_per_token": substage_gap,
              "floor_gap_ms_per_token": floor_gap,
          },
      },
      {
          "name": "defer_finish_signal_was_drain_shift_not_speed",
          "pass": bool(seq53_verdict.get("tail_drain_shift_confirmed"))
          and _num(seq53_derived.get("selected_down_wait_saved_ms_per_token")) > floor_gap
          and _num(seq53_derived.get("ffn_tail_growth_ms_per_token")) > floor_gap
          and bool(seq53_derived.get("tail_drain_elimination_clears_floor"))
          and not bool(seq53_derived.get("promotion_outside_noise")),
          "detail": {
              "selected_down_wait_saved_ms_per_token": _num(
                  seq53_derived.get("selected_down_wait_saved_ms_per_token")),
              "ffn_tail_growth_ms_per_token": _num(
                  seq53_derived.get("ffn_tail_growth_ms_per_token")),
              "tail_drain_elimination_clears_floor": bool(
                  seq53_derived.get("tail_drain_elimination_clears_floor")),
              "promotion_outside_noise": bool(
                  seq53_derived.get("promotion_outside_noise")),
          },
      },
      {
          "name": "hidden_row_serial_fusion_closed",
          "pass": seq66_verdict.get("naive_hidden_row_serial_fusion_promotable")
          is False,
      },
      {
          "name": "q6_nonatomic_decode_closed_as_speed",
          "pass": seq82.get("required_checks_passed") is True
          and seq82.get("disposition")
          == "rejected_q6_nonatomic_down_tail_decode_regresses_8tok",
      },
      {
          "name": "required_route_closures_recorded",
          "pass": not missing_closed_routes,
          "detail": {"missing_closed_routes": missing_closed_routes},
      },
      {
          "name": "seq81_down_tail_family_switch_recorded",
          "pass": _has_switch(routes, "close_q6_down_tail_decode_fusion_variants", 81),
      },
      {
          "name": "seq90_candidate_recorded",
          "pass": _has_candidate(
              routes, 90, "rejected_shared_q8_qkv_conv_component_speed_proof"),
      },
  ]
  required = all(check["pass"] for check in checks)
  selected_next_route = (
      "selected_ffn_down_wait_drain_removal_source_component_gate"
      if required
      else "route_selection_needs_manual_review"
  )
  disposition = (
      "select_selected_ffn_down_wait_drain_removal_source_component_gate"
      if required
      else "post_linear_preconv_route_gate_failed"
  )
  next_action = (
      "Run a source/component gate for selected-FFN down-wait root cause that "
      "proves a design can lower selected-FFN plus FFN-tail together. The proof "
      "must eliminate the tail drain instead of moving the selected-down wait, "
      "preserve selected/shared Q6 per-expert parallelism, and avoid the closed "
      "top16 gate/up, Q6 down-only, down-tail atomic/non-atomic, simple "
      "attention-front device-Q8, shared-Q8 preconv, and combined attention/"
      "linear alias routes. Do not launch a decode row until that source/"
      "component gate passes."
      if required
      else "Fix failed route-gate checks before launching another target probe."
  )

  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "required_checks_passed": required,
      "speedup_claims_allowed": False,
      "disposition": disposition,
      "selected_next_route": selected_next_route,
      "next_action": next_action,
      "inputs": {
          "frontier": _rel(args.frontier),
          "routes": _rel(args.routes),
          "rejected": _rel(args.rejected),
          "q6_defer_drain": _rel(args.q6_defer_drain),
          "down_tail_fusion": _rel(args.down_tail_fusion),
          "q6_nonatomic_decode": _rel(args.q6_nonatomic_decode),
          "post_top16": _rel(args.post_top16),
          "seq90_q4": _rel(args.seq90_q4),
          "seq90_q6": _rel(args.seq90_q6),
      },
      "frontier": frontier_state,
      "stage_gap_ms_per_token": stage_gaps,
      "substage_gap_ms_per_token": substage_gaps,
      "dominant_gap": {
          "stage": stage,
          "stage_gap_ms_per_token": stage_gap,
          "substage": substage,
          "substage_gap_ms_per_token": substage_gap,
      },
      "seq53_drain_shift": {
          "selected_down_wait_saved_ms_per_token": _num(
              seq53_derived.get("selected_down_wait_saved_ms_per_token")),
          "ffn_tail_growth_ms_per_token": _num(
              seq53_derived.get("ffn_tail_growth_ms_per_token")),
          "net_selected_plus_tail_delta_ms_per_token": _num(
              seq53_derived.get("net_selected_plus_tail_delta_ms_per_token")),
          "tail_drain_elimination_clears_floor": bool(
              seq53_derived.get("tail_drain_elimination_clears_floor")),
      },
      "seq90_linear_preconv": {
          "q4_estimated_delta_ms_per_token": _num(
              q4_seq90.get("component_estimated_delta_ms_per_token")),
          "q6_estimated_delta_ms_per_token": _num(
              q6_seq90.get("component_estimated_delta_ms_per_token")),
          "q4_component_wall_speedup_host_over_device": _num(
              q4_seq90.get("component_wall_speedup_host_over_device")),
          "q6_component_wall_speedup_host_over_device": _num(
              q6_seq90.get("component_wall_speedup_host_over_device")),
          "qkv_conv_rebased_growth_ms_per_token": _num(_nested(
              seq90_q4,
              "seq77_rebased_profile_ms_per_token",
              "rebased_qkv_conv_growth_after_moved_stages")),
      },
      "closed_route_requirements": {
          "required": required_closed_routes,
          "missing": missing_closed_routes,
      },
      "checks": checks,
  }


def write_outputs(out_dir: Path, payload: dict[str, Any]) -> None:
  out_dir.mkdir(parents=True, exist_ok=False)
  (out_dir / "metrics.json").write_text(
      json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  failed = [check["name"] for check in payload["checks"] if not check["pass"]]
  frontier = payload["frontier"]
  dominant = payload["dominant_gap"]
  seq53 = payload["seq53_drain_shift"]
  lines = [
      "# Post Linear-Preconv Route Gate",
      "",
      f"- required checks passed: `{str(payload['required_checks_passed']).lower()}`",
      f"- disposition: `{payload['disposition']}`",
      f"- selected next route: `{payload['selected_next_route']}`",
      f"- frontier gap: `{frontier['floor_gap_ms_per_token']:.3f}` ms/token",
      f"- largest stage gap: `{dominant['stage']}` "
      f"({dominant['stage_gap_ms_per_token']:.3f} ms/token)",
      f"- largest substage gap: `{dominant['substage']}` "
      f"({dominant['substage_gap_ms_per_token']:.3f} ms/token)",
      f"- seq53 selected-down wait saved: "
      f"`{seq53['selected_down_wait_saved_ms_per_token']:.3f}` ms/token",
      f"- seq53 FFN-tail growth: "
      f"`{seq53['ffn_tail_growth_ms_per_token']:.3f}` ms/token",
      f"- failed checks: `{failed}`",
      "",
      payload["next_action"],
      "",
      "This is route-control evidence only. It does not claim speed.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--frontier", type=Path, default=DEFAULT_FRONTIER)
  parser.add_argument("--routes", type=Path, default=DEFAULT_ROUTES)
  parser.add_argument("--rejected", type=Path, default=DEFAULT_REJECTED)
  parser.add_argument("--q6-defer-drain", type=Path, default=DEFAULT_SEQ53)
  parser.add_argument("--down-tail-fusion", type=Path, default=DEFAULT_SEQ66)
  parser.add_argument("--q6-nonatomic-decode", type=Path, default=DEFAULT_SEQ82)
  parser.add_argument("--post-top16", type=Path, default=DEFAULT_SEQ89)
  parser.add_argument("--seq90-q4", type=Path, default=DEFAULT_SEQ90_Q4)
  parser.add_argument("--seq90-q6", type=Path, default=DEFAULT_SEQ90_Q6)
  parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
  return parser.parse_args()


def main() -> int:
  args = parse_args()
  payload = compute(args)
  write_outputs(args.out_dir, payload)
  print(json.dumps(payload, indent=2, sort_keys=True))
  return 0 if payload["required_checks_passed"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
