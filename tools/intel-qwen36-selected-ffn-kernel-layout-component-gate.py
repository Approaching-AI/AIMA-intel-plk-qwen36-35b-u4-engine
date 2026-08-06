#!/usr/bin/env python3
"""Audit the selected-FFN kernel/layout component route after seq108.

This is route-control evidence only. It consumes the accepted selected+shared
Q4/Q6 down stack, the current frontier budget, and the selected-FFN component
closures. It either authorizes a concrete selected-FFN decode probe or closes
the current selected-FFN component route before any token-emitting row.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = "intel-qwen36-selected-ffn-kernel-layout-component-gate-v0"
DEFAULT_FRONTIER = ACTIVE / "frontier.json"
DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_REJECTED = ACTIVE / "rejected-routes.json"
DEFAULT_BEST = (
    ROOT
    / "output/r2-gpu-selected-shared-q4q6-down-cold-q6-experts-noqueue-speed-20260705T143006Z/result.json"
)
DEFAULT_POST_SETUP = (
    ROOT / "output/post-setup-hoist-route-gate-20260707Tseq108Z/metrics.json"
)
DEFAULT_MATERIAL_GATE = (
    ROOT / "output/selected-ffn-material-layout-gate-20260707Tseq87Z/metrics.json"
)
DEFAULT_TOP16_CONFIRM = (
    ROOT
    / "output/gpu-q4x8-selected-gateup-top16-indexed-probe-confirm-20260707Tseq88Z/probe-result.json"
)
DEFAULT_Q4_LOCAL64 = (
    ROOT / "output/gpu-selected-down-q4-selected-shared-local64-baseline-20260705T141004Z/probe-result.json"
)
DEFAULT_Q4_LOCAL128 = (
    ROOT / "output/gpu-selected-down-q4-selected-shared-local128-probe-20260705T141044Z/probe-result.json"
)
DEFAULT_Q6_LOCAL64 = (
    ROOT / "output/gpu-selected-down-q6-selected-shared-local64-baseline-20260705T145000Z/probe-result.json"
)
DEFAULT_Q6_LOCAL128 = (
    ROOT / "output/gpu-selected-down-q6-selected-shared-local128-probe-20260705T145100Z/probe-result.json"
)
DEFAULT_Q6_OCCUPANCY = (
    ROOT / "output/gpu-selected-down-q6-selected-shared-occupancy4-probe-20260705T220711Z/probe-result.json"
)
DEFAULT_FUSED_SWIGLU = (
    ROOT / "output/gpu-q4x8-selected-shared-gate-up-fused-swiglu-probe-20260705T212105Z/probe-result.json"
)
DEFAULT_ROWGROUP = (
    ROOT / "output/q6-rowgroup-down-tail-gate-20260707Tseq93Z/metrics.json"
)
DEFAULT_HOIST = (
    ROOT / "output/linear-setup-specialized-hoist-gate-20260707Tseq105-107Z/metrics.json"
)
DEFAULT_OUT_DIR = (
    ROOT / "output/selected-ffn-kernel-layout-component-gate-20260707Tseq109Z"
)

Q6_LAYER_INVOCATIONS_PER_TOKEN = 20.0
Q4_LAYER_INVOCATIONS_PER_TOKEN = 20.0


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


def _stage_walls(frontier: dict[str, Any]) -> dict[str, float]:
  budget = frontier.get("goal_budget")
  budget = budget if isinstance(budget, dict) else {}
  out: dict[str, float] = {}
  for row in budget.get("top_stage_walls_ms_per_token", []):
    if isinstance(row, dict) and isinstance(row.get("stage"), str):
      out[row["stage"]] = _num(row.get("ms_per_token"))
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


def _timing(obj: dict[str, Any], name: str) -> float:
  return _num(_nested(obj, "timings", name))


def _ms_per_token_from_us_delta(
    before_us: float, after_us: float, invocations_per_token: float
) -> float:
  return max(0.0, before_us - after_us) * invocations_per_token / 1000.0


def compute(args: argparse.Namespace) -> dict[str, Any]:
  frontier = _load_json(args.frontier)
  routes = _load_json(args.routes)
  rejected = _load_json(args.rejected)
  best = _load_json(args.best)
  post_setup = _load_json(args.post_setup)
  material_gate = _load_json(args.material_gate)
  top16_confirm = _load_json(args.top16_confirm)
  q4_local64 = _load_json(args.q4_local64)
  q4_local128 = _load_json(args.q4_local128)
  q6_local64 = _load_json(args.q6_local64)
  q6_local128 = _load_json(args.q6_local128)
  q6_occupancy = _load_json(args.q6_occupancy)
  fused_swiglu = _load_json(args.fused_swiglu)
  rowgroup = _load_json(args.rowgroup)
  hoist = _load_json(args.hoist)

  frontier_state = _frontier_summary(frontier)
  stage_gaps = _stage_gaps(frontier)
  stage_walls = _stage_walls(frontier)
  substage_gaps = _substage_gaps(frontier)
  floor_gap = frontier_state["floor_gap_ms_per_token"]
  rejected_names = _route_names(rejected)
  smoke = best.get("smoke") if isinstance(best.get("smoke"), dict) else best

  top16_ratio = _timing(
      top16_confirm, "top16_indexed_vs_no_concat_shell_speedup")
  top16_required = _timing(top16_confirm, "top16_floor_covering_required_ratio")
  q6_required = _num(
      _nested(material_gate, "derived", "q6_down_required_speedup_to_cover_floor"))
  q6_occupancy_ratio = _num(
      q6_occupancy.get(
          "candidate_q6_selected_shared_occupancy_scaled_speedup_vs_combined"))
  fused_swiglu_ratio = _timing(fused_swiglu, "fused_vs_noconcat_shell_speedup")

  q4_local64_us = _timing(q4_local64, "candidate_q4_selected_shared_combined_kernel_min_us")
  q4_local128_us = _timing(q4_local128, "candidate_q4_selected_shared_combined_kernel_min_us")
  q6_local64_us = _timing(q6_local64, "candidate_q6_selected_shared_combined_kernel_min_us")
  q6_local128_us = _timing(q6_local128, "candidate_q6_selected_shared_combined_kernel_min_us")
  q4_local128_ms_per_token = _ms_per_token_from_us_delta(
      q4_local64_us, q4_local128_us, Q4_LAYER_INVOCATIONS_PER_TOKEN)
  q6_local128_ms_per_token = _ms_per_token_from_us_delta(
      q6_local64_us, q6_local128_us, Q6_LAYER_INVOCATIONS_PER_TOKEN)

  remaining_setup_ms = _num(
      _nested(hoist, "specialized_hoist_source_profile",
              "dispatch_profiled_after_ms_per_token"))
  remaining_layer_dispatch_ms = _num(
      _nested(hoist, "specialized_hoist_source_profile",
              "layer_dispatch_gap_after_ms_per_token"))

  required_closed_routes = [
      "gpu_selected_shared_q4_gateup_combined_concat_decode",
      "gpu_selected_all_expert_residency",
      "gpu_selected_shared_q4_gateup_group8_component",
      "gpu_selected_shared_q4_gateup_no_concat_local128_noqueue",
      "gpu_selected_shared_q4_gateup_fused_swiglu_component",
      "gpu_selected_gateup_top16_indexed_material_component",
      "gpu_selected_shared_q4_down_combined_local128_component",
      "gpu_selected_shared_q6_down_combined_local128_component",
      "gpu_selected_shared_q6_down_occupancy4_component",
      "gpu_selected_q4_down_pair2_dual_expert_component",
      "gpu_selected_q4_down_weighted_tail_decode",
      "gpu_selected_q4_down_weighted_sum_component",
      "gpu_selected_q4_down_group8_weighted_sum_component",
      "gpu_selected_q4_down_device_q8_component",
      "gpu_selected_q4_down_swiglu_device_q8_component",
      "gpu_selected_q4_down_swiglu_f32input_decode",
      "gpu_q6_expert8_pair2_row_reuse",
      "gpu_selected_down_q6_plane_layout_probe",
      "gpu_q6_bpr2_selected_down_specialization",
      "gpu_q6_nonatomic_down_tail_decode_fusion",
      "gpu_direct_q6_down_tail_atomic_from_q6_workitems",
      "gpu_ffn_tail_atomic_reduction_from_down_handles",
      "gpu_selected_shared_q6_down_tail_rowgroup_local_reduce_component",
      "generic_token_setup_cache",
      "linear_setup_specialized_hoist",
  ]
  missing_closed_routes = [
      route for route in required_closed_routes if route not in rejected_names
  ]

  largest_stage = max(stage_gaps, key=stage_gaps.get) if stage_gaps else ""
  non_selected = {
      stage: gap for stage, gap in stage_gaps.items()
      if stage not in {"selected_ffn", "shared_ffn", "lm_head_gpu"}
  }
  next_stage = max(non_selected, key=non_selected.get) if non_selected else ""
  next_stage_gap = non_selected.get(next_stage, 0.0)

  checks = [
      {
          "name": "seq108_selected_ffn_component_gate_selected",
          "pass": post_setup.get("required_checks_passed") is True
          and post_setup.get("selected_next_route")
          == "selected_ffn_kernel_layout_component_gate"
          and _has_switch(
              routes, "switch_to_selected_ffn_kernel_layout_component_gate", 108),
      },
      {
          "name": "frontier_still_below_floor",
          "pass": frontier_state["wall_ms_per_token"]
          > frontier_state["floor_budget_ms_per_token"]
          > 0.0,
      },
      {
          "name": "selected_ffn_is_largest_gap_but_component_gate_only",
          "pass": largest_stage == "selected_ffn"
          and stage_gaps.get("selected_ffn", 0.0) > floor_gap,
          "detail": {
              "largest_stage": largest_stage,
              "selected_ffn_gap_ms_per_token": stage_gaps.get("selected_ffn", 0.0),
              "selected_ffn_wall_ms_per_token": stage_walls.get("selected_ffn", 0.0),
              "floor_gap_ms_per_token": floor_gap,
          },
      },
      {
          "name": "accepted_stack_is_selected_shared_q4q6_with_zero_selected_misses",
          "pass": (
              smoke.get("top1_match_count") == 8
              and smoke.get("selected_shared_q4_gateup_combined") is True
              and smoke.get("selected_shared_q4_down_combined") is True
              and smoke.get("selected_shared_q6_down_combined") is True
              and smoke.get("resident_selected_gate_up_misses") == 0
              and smoke.get("resident_selected_down_misses") == 0
              and smoke.get("resident_selected_q6_down_misses") == 0
          ),
      },
      {
          "name": "selected_ffn_closed_route_board_complete",
          "pass": not missing_closed_routes,
          "detail": {"missing": missing_closed_routes},
      },
      {
          "name": "top16_gateup_exact_but_not_floor_covering",
          "pass": bool(
              _nested(top16_confirm, "checks",
                      "top16_indexed_gateup_swiglu_matches_references"))
          and 0.0 < top16_ratio < top16_required,
          "detail": {
              "observed_shell_speedup": top16_ratio,
              "required_shell_speedup": top16_required,
          },
      },
      {
          "name": "q6_down_occupancy4_is_synthetic_and_not_floor_covering",
          "pass": 0.0 < q6_occupancy_ratio < q6_required,
          "detail": {
              "observed_occupancy4_scaled_speedup": q6_occupancy_ratio,
              "required_speedup": q6_required,
          },
      },
      {
          "name": "local_size_component_deltas_are_below_floor_gap",
          "pass": q4_local128_ms_per_token < floor_gap
          and q6_local128_ms_per_token < floor_gap,
          "detail": {
              "q4_local128_projected_ms_per_token": q4_local128_ms_per_token,
              "q6_local128_projected_ms_per_token": q6_local128_ms_per_token,
              "floor_gap_ms_per_token": floor_gap,
          },
      },
      {
          "name": "fused_swiglu_component_regresses",
          "pass": 0.0 < fused_swiglu_ratio < 1.0,
          "detail": {"fused_vs_noconcat_shell_speedup": fused_swiglu_ratio},
      },
      {
          "name": "down_tail_rowgroup_component_not_decode_allowed",
          "pass": rowgroup.get("decode_probe_allowed") is False
          and rowgroup.get("target_cleared") is False
          and _num(rowgroup.get("miss_vs_target_us")) > 0.0,
          "detail": {
              "rowgroup_min_us": rowgroup.get("rowgroup_min_us"),
              "target_us": rowgroup.get("target_us"),
              "miss_vs_target_us": rowgroup.get("miss_vs_target_us"),
          },
      },
      {
          "name": "remaining_setup_after_hoist_is_smaller_than_floor_gap",
          "pass": remaining_setup_ms < floor_gap
          and remaining_layer_dispatch_ms < floor_gap,
          "detail": {
              "dispatch_profiled_after_ms_per_token": remaining_setup_ms,
              "layer_dispatch_gap_after_ms_per_token": remaining_layer_dispatch_ms,
              "floor_gap_ms_per_token": floor_gap,
          },
      },
      {
          "name": "next_nonselected_stage_gap_can_cover_floor",
          "pass": next_stage_gap > floor_gap,
          "detail": {
              "next_stage": next_stage,
              "next_stage_gap_ms_per_token": next_stage_gap,
              "floor_gap_ms_per_token": floor_gap,
          },
      },
  ]
  required = all(check["pass"] for check in checks)
  selected_next_route = (
      "full_core_attention_front_kernel_algorithm_gate"
      if required else "manual_review_selected_ffn_component_gate"
  )
  next_action = (
      "Close the current selected-FFN kernel/layout component route as a decode "
      "probe source. The next unit is a full-core/attention-front kernel "
      "algorithm gate, still component-first: it must not repeat full-core "
      "local-size, score-local32, softmax-cache, finish-coalesce, attention "
      "residual/RMSNorm fusion, simple linear-final device-Q8 handoff, shared-Q8 "
      "preconv, or the combined attention/linear alias."
      if required else "Review selected-FFN component evidence before selecting a route."
  )

  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "required_checks_passed": required,
      "speedup_claims_allowed": False,
      "decode_probe_allowed": False,
      "disposition": (
          "close_current_selected_ffn_kernel_layout_component_route"
          if required else "selected_ffn_kernel_layout_gate_failed"
      ),
      "selected_next_route": selected_next_route,
      "next_action": next_action,
      "inputs": {
          "frontier": _rel(args.frontier),
          "routes": _rel(args.routes),
          "rejected": _rel(args.rejected),
          "best": _rel(args.best),
          "post_setup": _rel(args.post_setup),
          "material_gate": _rel(args.material_gate),
          "top16_confirm": _rel(args.top16_confirm),
          "q4_local64": _rel(args.q4_local64),
          "q4_local128": _rel(args.q4_local128),
          "q6_local64": _rel(args.q6_local64),
          "q6_local128": _rel(args.q6_local128),
          "q6_occupancy": _rel(args.q6_occupancy),
          "fused_swiglu": _rel(args.fused_swiglu),
          "rowgroup": _rel(args.rowgroup),
          "hoist": _rel(args.hoist),
      },
      "frontier": frontier_state,
      "stage_gap_ms_per_token": stage_gaps,
      "stage_wall_ms_per_token": stage_walls,
      "selected_substage_gap_ms_per_token": {
          key: gap for key, gap in substage_gaps.items()
          if key.startswith("selected_ffn.")
      },
      "selected_ffn_component_summary": {
          "top16_shell_speedup": top16_ratio,
          "top16_required_speedup": top16_required,
          "q6_occupancy4_scaled_speedup": q6_occupancy_ratio,
          "q6_required_speedup": q6_required,
          "q4_local128_projected_ms_per_token": q4_local128_ms_per_token,
          "q6_local128_projected_ms_per_token": q6_local128_ms_per_token,
          "fused_swiglu_vs_noconcat_speedup": fused_swiglu_ratio,
          "rowgroup_miss_vs_target_us": rowgroup.get("miss_vs_target_us"),
          "remaining_setup_ms_per_token": remaining_setup_ms,
          "remaining_layer_dispatch_ms_per_token": remaining_layer_dispatch_ms,
      },
      "next_route_reason": {
          "next_stage": next_stage,
          "next_stage_gap_ms_per_token": next_stage_gap,
          "full_core_gap_ms_per_token": stage_gaps.get("full_core", 0.0),
          "attention_front_gap_ms_per_token": stage_gaps.get("attention_front", 0.0),
          "linear_preconv_gap_ms_per_token": stage_gaps.get("linear_preconv", 0.0),
          "linear_delta_gap_ms_per_token": stage_gaps.get("linear_delta", 0.0),
      },
      "checks": checks,
  }


def write_outputs(out_dir: Path, payload: dict[str, Any]) -> None:
  out_dir.mkdir(parents=True, exist_ok=False)
  (out_dir / "metrics.json").write_text(
      json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  failed = [check["name"] for check in payload["checks"] if not check["pass"]]
  frontier = payload["frontier"]
  selected = payload["selected_ffn_component_summary"]
  next_reason = payload["next_route_reason"]
  lines = [
      "# Selected-FFN Kernel/Layout Component Gate",
      "",
      f"- required checks passed: `{str(payload['required_checks_passed']).lower()}`",
      f"- disposition: `{payload['disposition']}`",
      f"- decode probe allowed: `{str(payload['decode_probe_allowed']).lower()}`",
      f"- selected next route: `{payload['selected_next_route']}`",
      f"- frontier gap: `{frontier['floor_gap_ms_per_token']:.3f}` ms/token",
      f"- top16 shell speedup / required: "
      f"`{selected['top16_shell_speedup']:.3f}` / "
      f"`{selected['top16_required_speedup']:.3f}`",
      f"- Q6 occupancy4 speedup / required: "
      f"`{selected['q6_occupancy4_scaled_speedup']:.3f}` / "
      f"`{selected['q6_required_speedup']:.3f}`",
      f"- Q4/Q6 local128 projected cuts: "
      f"`{selected['q4_local128_projected_ms_per_token']:.3f}` / "
      f"`{selected['q6_local128_projected_ms_per_token']:.3f}` ms/token",
      f"- rowgroup miss vs target: "
      f"`{selected['rowgroup_miss_vs_target_us']:.3f}` us",
      f"- next non-selected stage: `{next_reason['next_stage']}` "
      f"(`{next_reason['next_stage_gap_ms_per_token']:.3f}` ms/token)",
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
  parser.add_argument("--best", type=Path, default=DEFAULT_BEST)
  parser.add_argument("--post-setup", type=Path, default=DEFAULT_POST_SETUP)
  parser.add_argument("--material-gate", type=Path, default=DEFAULT_MATERIAL_GATE)
  parser.add_argument("--top16-confirm", type=Path, default=DEFAULT_TOP16_CONFIRM)
  parser.add_argument("--q4-local64", type=Path, default=DEFAULT_Q4_LOCAL64)
  parser.add_argument("--q4-local128", type=Path, default=DEFAULT_Q4_LOCAL128)
  parser.add_argument("--q6-local64", type=Path, default=DEFAULT_Q6_LOCAL64)
  parser.add_argument("--q6-local128", type=Path, default=DEFAULT_Q6_LOCAL128)
  parser.add_argument("--q6-occupancy", type=Path, default=DEFAULT_Q6_OCCUPANCY)
  parser.add_argument("--fused-swiglu", type=Path, default=DEFAULT_FUSED_SWIGLU)
  parser.add_argument("--rowgroup", type=Path, default=DEFAULT_ROWGROUP)
  parser.add_argument("--hoist", type=Path, default=DEFAULT_HOIST)
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
