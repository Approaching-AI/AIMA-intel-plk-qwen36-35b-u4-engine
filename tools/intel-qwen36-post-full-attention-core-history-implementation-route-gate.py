#!/usr/bin/env python3
"""Select the next route after the full-attention core/history speed rejection.

This is route-control evidence only. It consumes the correctness-valid but
speed-regressive resident core/history handle stack and selects the next
profile/root-cause unit without launching another token row.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = (
    "intel-qwen36-post-full-attention-core-history-implementation-route-gate-v0"
)

DEFAULT_FRONTIER = ACTIVE / "frontier.json"
DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_REJECTED = ACTIVE / "rejected-routes.json"
DEFAULT_SEQ124 = (
    ROOT / "output/resident-hidden-state-carrier-full-boundary-decode-gate-20260707Tseq124Z"
    / "metrics.json"
)
DEFAULT_SEQ125 = (
    ROOT / "output/resident-hidden-state-carrier-tail-growth-root-gate-20260707Tseq125Z"
    / "metrics.json"
)
DEFAULT_SEQ134 = (
    ROOT
    / "output/resident-hidden-state-carrier-layer-output-handle-loop-full-core-parity-speed-gate-20260707Tseq134Z"
    / "metrics.json"
)
DEFAULT_SEQ142 = (
    ROOT
    / "output/resident-hidden-state-carrier-full-attention-qkv-handle-defer-binaryhit-speed-gate-20260707Tseq142Z"
    / "metrics.json"
)
DEFAULT_SEQ152 = (
    ROOT
    / "output/resident-hidden-state-carrier-full-attention-core-history-implementation-decode-gate-20260707Tseq152Z"
    / "metrics.json"
)
DEFAULT_SEQ153 = (
    ROOT
    / "output/resident-hidden-state-carrier-full-attention-core-history-implementation-speed-gate-20260707Tseq153Z"
    / "metrics.json"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "output/post-full-attention-core-history-implementation-route-gate-20260708Tseq154Z"
)


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
  glide = no_progress.get("glide_slope")
  glide = glide if isinstance(glide, dict) else {}
  wall = _num(per_token.get("wall"))
  floor_budget = _num(verdict.get("floor_budget_ms_per_token"))
  overhead = _num(per_token.get("non_kernel_overhead"))
  floor_gap = max(0.0, wall - floor_budget)
  return {
      "current_best_tps": _num(anchor.get("current_best_tps")),
      "floor_tps": _num(anchor.get("same_host_vulkan_floor_tps")),
      "wall_ms_per_token": wall,
      "floor_budget_ms_per_token": floor_budget,
      "floor_gap_ms_per_token": floor_gap,
      "gpu_kernel_busy_floor_ms_per_token": _num(
          per_token.get("gpu_kernel_busy_floor")),
      "non_kernel_overhead_ms_per_token": overhead,
      "overhead_cut_fraction_needed": (
          floor_gap / overhead if overhead > 0.0 else 0.0
      ),
      "overhead_only_ceiling_tok_s": _num(
          verdict.get("overhead_only_ceiling_tok_s")),
      "can_reach_floor_without_kernel_work": bool(
          verdict.get("can_reach_floor_without_kernel_work")),
      "noise_rel": _num(noise.get("rel")),
      "runs_since_significant_improvement": no_progress.get(
          "runs_since_significant_improvement"),
      "hard_stall_breached": no_progress.get("hard_stall_breached"),
      "soft_reflection_breached": no_progress.get("soft_reflection_breached"),
      "review_recorded_for_current_best": no_progress.get(
          "review_recorded_for_current_best"),
      "glide_projected_runs_to_floor": glide.get("projected_runs_to_floor"),
      "glide_breached": glide.get("breached"),
  }


def _stage_gaps(frontier: dict[str, Any]) -> dict[str, float]:
  rows = _nested(frontier, "goal_budget", "stage_kernel_gap_estimates_ms_per_token")
  out: dict[str, float] = {}
  if not isinstance(rows, list):
    return out
  for row in rows:
    if isinstance(row, dict) and isinstance(row.get("stage"), str):
      out[row["stage"]] = _num(row.get("gap_ms_per_token"))
  return out


def _stage_walls(frontier: dict[str, Any]) -> dict[str, float]:
  rows = _nested(frontier, "goal_budget", "top_stage_walls_ms_per_token")
  out: dict[str, float] = {}
  if not isinstance(rows, list):
    return out
  for row in rows:
    if isinstance(row, dict) and isinstance(row.get("stage"), str):
      out[row["stage"]] = _num(row.get("ms_per_token"))
  return out


def _rejected_names(rejected: dict[str, Any]) -> set[str]:
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


def _speed(metrics: dict[str, Any]) -> dict[str, Any]:
  speed = metrics.get("speed")
  return speed if isinstance(speed, dict) else {}


def _check_detail(metrics: dict[str, Any], name: str) -> dict[str, Any]:
  checks = metrics.get("checks")
  if not isinstance(checks, list):
    return {}
  for row in checks:
    if isinstance(row, dict) and row.get("name") == name:
      detail = row.get("detail")
      return detail if isinstance(detail, dict) else {}
  return {}


def _regresses_outside_noise(speed: dict[str, Any], noise_rel: float) -> bool:
  return (
      _num(speed.get("explore_tps")) < _num(speed.get("best_tps"))
      and _num(speed.get("relative_delta")) <= -noise_rel
  )


def compute(args: argparse.Namespace) -> dict[str, Any]:
  frontier = _load_json(args.frontier)
  routes = _load_json(args.routes)
  rejected = _load_json(args.rejected)
  seq124 = _load_json(args.seq124)
  seq125 = _load_json(args.seq125)
  seq134 = _load_json(args.seq134)
  seq142 = _load_json(args.seq142)
  seq152 = _load_json(args.seq152)
  seq153 = _load_json(args.seq153)

  frontier_state = _frontier_state(frontier)
  floor_gap = frontier_state["floor_gap_ms_per_token"]
  stage_gaps = _stage_gaps(frontier)
  stage_walls = _stage_walls(frontier)
  rejected_names = _rejected_names(rejected)
  seq134_speed = _speed(seq134)
  seq142_speed = _speed(seq142)
  seq153_speed = _speed(seq153)
  seq152_stack = _check_detail(seq152, "distribution_uses_core_history_handle_stack")
  seq153_growth = _check_detail(
      seq153, "full_attention_boundary_growth_explains_regression")

  required_rejected = {
      "current_resident_hidden_state_carrier_full_boundary_speed_shape",
      "current_resident_hidden_state_carrier_tail_readback_loop_shape",
      "current_resident_hidden_state_carrier_layer_output_loop_full_core_parity_speed_shape",
      "current_resident_hidden_state_carrier_full_attention_qkv_handle_speed_shape",
      "current_resident_hidden_state_carrier_full_attention_core_history_handle_speed_shape",
      "current_selected_ffn_kernel_layout_component_board",
      "current_full_core_attention_front_kernel_algorithm_board",
      "current_linear_preconv_alpha_beta_algorithm_board",
      "current_linear_delta_algorithm_board",
      "current_offline_repack_streaming_layout_board",
      "current_moe_routed_down_fusion_board",
      "engine_resident_gpu_hot_loop_api_shell",
      "generic_token_setup_cache",
      "linear_setup_specialized_hoist",
      "gpu_full_attention_state_resident_history",
      "gpu_full_attention_flat_history_cache",
  }
  missing_rejected = sorted(required_rejected - rejected_names)

  checks = [
      {
          "name": "seq153_valid_core_history_speed_rejection",
          "pass": (
              seq153.get("required_checks_passed") is True
              and seq153.get("disposition")
              == "reject_full_attention_core_history_implementation_as_speed_cut"
              and seq153.get("selected_next_route")
              == "post_full_attention_core_history_implementation_route_gate"
              and _regresses_outside_noise(
                  seq153_speed, frontier_state["noise_rel"])
          ),
          "detail": {"speed_gate": _rel(args.seq153), "speed": seq153_speed},
      },
      {
          "name": "seq152_correctness_validates_not_ruler_failure",
          "pass": (
              seq152.get("required_checks_passed") is True
              and seq152.get("disposition")
              == "accept_full_attention_core_history_implementation_correctness"
              and seq152_stack.get("core_history_handle") is True
              and seq152_stack.get("qkv_handle") is True
              and _num(seq152_stack.get("full_core_handoff_layers")) >= 80
          ),
          "detail": {"decode_gate": _rel(args.seq152), "stack": seq152_stack},
      },
      {
          "name": "frontier_still_below_floor_with_review_recorded",
          "pass": (
              frontier_state["current_best_tps"] < frontier_state["floor_tps"]
              and floor_gap > 0.0
              and frontier_state["hard_stall_breached"] is True
              and frontier_state["review_recorded_for_current_best"] is True
              and frontier_state["can_reach_floor_without_kernel_work"] is True
              and frontier_state["overhead_only_ceiling_tok_s"]
              > frontier_state["floor_tps"]
          ),
          "detail": frontier_state,
      },
      {
          "name": "per_boundary_carrier_speed_shapes_are_closed",
          "pass": (
              seq124.get("required_checks_passed") is True
              and seq124.get("disposition")
              == "reject_current_carrier_full_boundary_as_speed_cut"
              and seq125.get("required_checks_passed") is True
              and seq125.get("disposition")
              == "carrier_tail_growth_bound_to_host_layer_output_drain"
              and seq134.get("required_checks_passed") is True
              and seq134.get("disposition")
              == "reject_layer_output_loop_full_core_parity_as_speed_cut"
              and seq142.get("required_checks_passed") is True
              and seq142.get("disposition")
              == "reject_full_attention_qkv_handle_as_speed_cut"
              and _num(seq134_speed.get("relative_delta"))
              <= frontier_state["noise_rel"]
              and _regresses_outside_noise(
                  seq142_speed, frontier_state["noise_rel"])
          ),
          "detail": {
              "seq124": _rel(args.seq124),
              "seq125": _rel(args.seq125),
              "seq134_speed": seq134_speed,
              "seq142_speed": seq142_speed,
          },
      },
      {
          "name": "closed_route_board_prevents_boundary_retry",
          "pass": not missing_rejected,
          "detail": {
              "missing_rejected_routes": missing_rejected,
              "required_rejected_routes": sorted(required_rejected),
          },
      },
      {
          "name": "seq153_route_switch_recorded",
          "pass": _has_switch(
              routes,
              "reject_full_attention_core_history_implementation_speed_switch_to_route_gate",
              153,
          ),
      },
      {
          "name": "next_unit_must_be_profile_not_token_row",
          "pass": (
              stage_gaps.get("selected_ffn", 0.0) > floor_gap
              and stage_gaps.get("attention_front", 0.0) > floor_gap
              and stage_gaps.get("linear_preconv", 0.0) > floor_gap
              and frontier_state["non_kernel_overhead_ms_per_token"] > floor_gap
          ),
          "detail": {
              "floor_gap_ms_per_token": floor_gap,
              "stage_gaps_ms_per_token": stage_gaps,
              "stage_walls_ms_per_token": stage_walls,
              "seq153_growth_ms_per_token": seq153_growth,
          },
      },
  ]
  required_checks_passed = all(bool(row.get("pass")) for row in checks)
  selected_next_route = (
      "resident_full_gpu_decode_loop_profile_gate"
      if required_checks_passed
      else "post_full_attention_core_history_route_review"
  )
  disposition = (
      "select_resident_full_gpu_decode_loop_profile_gate"
      if required_checks_passed
      else "post_full_attention_core_history_route_gate_incomplete"
  )

  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "inputs": {
          "frontier": _rel(args.frontier),
          "routes": _rel(args.routes),
          "rejected": _rel(args.rejected),
          "seq124_full_boundary_gate": _rel(args.seq124),
          "seq125_tail_growth_gate": _rel(args.seq125),
          "seq134_layer_output_loop_speed_gate": _rel(args.seq134),
          "seq142_qkv_speed_gate": _rel(args.seq142),
          "seq152_core_history_decode_gate": _rel(args.seq152),
          "seq153_core_history_speed_gate": _rel(args.seq153),
      },
      "frontier": frontier_state,
      "route_signal": {
          "selected_next_route": selected_next_route,
          "why_not_qkv_or_core_history_retry": (
              "Seq142 and seq153 are cache-hit, correctness-valid speed gates "
              "that regress materially versus the frontier. Another boundary "
              "retry would repeat a closed shape."
          ),
          "why_not_standalone_history": (
              "Standalone resident K/V history and flat-history cache routes "
              "are already closed. Seq153 removes the larger core/history "
              "boundary and still regresses."
          ),
          "why_not_kernel_board_default": (
              "Selected-FFN, full-core/attention-front, linear-preconv, "
              "linear-delta, offline-repack, and MoE/down-tail boards are all "
              "closed for their current shapes. The hard-stall signal asks for "
              "profile/root-cause before another token row."
          ),
          "minimum_next_proof": (
              "Run a no-speed profile/root-cause gate over the assembled "
              "resident/full-GPU decode loop. It must identify a remaining "
              "floor-sized host-boundary, dispatch, wait, or kernel bucket "
              "before any new explore row."
          ),
      },
      "closed_boundary_speed_shapes": {
          "seq124_full_boundary_relative_delta": _check_detail(
              seq124, "explore_regresses_outside_noise").get("relative_vs_best"),
          "seq134_layer_output_relative_delta": seq134_speed.get("relative_delta"),
          "seq142_qkv_relative_delta": seq142_speed.get("relative_delta"),
          "seq153_core_history_relative_delta": seq153_speed.get("relative_delta"),
          "seq153_wall_delta_ms_per_token": seq153_speed.get(
              "wall_delta_ms_per_token"),
      },
      "checks": checks,
      "required_checks_passed": required_checks_passed,
      "decode_probe_allowed": False,
      "component_probe_allowed": False,
      "source_profile_gate_allowed": required_checks_passed,
      "speedup_claims_allowed": False,
      "disposition": disposition,
      "selected_next_route": selected_next_route,
      "next_action": (
          "Build resident_full_gpu_decode_loop_profile_gate as route-control "
          "evidence first. Do not launch QK/V, core/history, standalone history, "
          "or current-kernel-board token rows until that gate identifies a "
          "floor-sized root."
          if required_checks_passed
          else "Fix failed route-gate checks before launching more token rows."
      ),
  }


def write_outputs(metrics: dict[str, Any], out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
  )
  manifest = {
      "schema_version": metrics["schema_version"],
      "workstream": metrics["workstream"],
      "tool": (
          "tools/intel-qwen36-post-full-attention-core-history-implementation-route-gate.py"
      ),
      "inputs": metrics["inputs"],
      "selected_next_route": metrics["selected_next_route"],
      "speedup_claims_allowed": False,
  }
  (out_dir / "manifest.json").write_text(
      json.dumps(manifest, indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
  )
  failed = [row["name"] for row in metrics["checks"] if not row["pass"]]
  frontier = metrics["frontier"]
  shapes = metrics["closed_boundary_speed_shapes"]
  lines = [
      "# Post Full-Attention Core/History Implementation Route Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- decode_probe_allowed: `{str(metrics['decode_probe_allowed']).lower()}`",
      f"- source_profile_gate_allowed: `{str(metrics['source_profile_gate_allowed']).lower()}`",
      f"- speedup_claims_allowed: `{str(metrics['speedup_claims_allowed']).lower()}`",
      f"- floor_gap_ms_per_token: `{frontier['floor_gap_ms_per_token']:.3f}`",
      f"- no_progress_runs: `{frontier['runs_since_significant_improvement']}`",
      f"- seq153_relative_delta: `{_num(shapes['seq153_core_history_relative_delta']):.6f}`",
      f"- failed_checks: `{failed}`",
      "",
      metrics["route_signal"]["minimum_next_proof"],
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
  parser.add_argument("--seq124", type=Path, default=DEFAULT_SEQ124)
  parser.add_argument("--seq125", type=Path, default=DEFAULT_SEQ125)
  parser.add_argument("--seq134", type=Path, default=DEFAULT_SEQ134)
  parser.add_argument("--seq142", type=Path, default=DEFAULT_SEQ142)
  parser.add_argument("--seq152", type=Path, default=DEFAULT_SEQ152)
  parser.add_argument("--seq153", type=Path, default=DEFAULT_SEQ153)
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
