#!/usr/bin/env python3
"""Select the next route after selected-down submit/wait split profiling.

This is route-control evidence only. It consumes the noqueue split-profile row
that proves selected-down submit/wait is no longer floor-sized, then selects the
next source-profile unit from the remaining wall buckets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = "intel-qwen36-post-selected-down-submit-split-route-gate-v0"

DEFAULT_FRONTIER = ACTIVE / "frontier.json"
DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_REJECTED = ACTIVE / "rejected-routes.json"
DEFAULT_SEQ162 = (
    ROOT
    / "output/resident-selected-down-submit-split-profile-explore-gate-20260708Tseq162Z"
    / "metrics.json"
)
DEFAULT_SEQ158 = (
    ROOT
    / "output/resident-queue-drain-site-profile-explore-gate-20260708Tseq158Z"
    / "metrics.json"
)
DEFAULT_DECODE = ROOT / "tools/intel-qwen36-r2-gpu-decode-smoke.py"
DEFAULT_OUT_DIR = (
    ROOT / "output/post-selected-down-submit-split-route-gate-20260708Tseq163Z"
)


def _load_json(path: Path) -> Any:
  return json.loads(path.read_text(encoding="utf-8"))


def _read(path: Path) -> str:
  return path.read_text(encoding="utf-8")


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


def _dict(value: Any) -> dict[str, Any]:
  return value if isinstance(value, dict) else {}


def _frontier_state(frontier: dict[str, Any]) -> dict[str, Any]:
  anchor = _dict(frontier.get("goal_anchor"))
  budget = _dict(frontier.get("goal_budget"))
  per_token = _dict(budget.get("per_token_ms"))
  verdict = _dict(budget.get("verdict"))
  no_progress = _dict(frontier.get("no_progress"))
  noise = _dict(no_progress.get("noise"))
  glide = _dict(no_progress.get("glide_slope"))
  wall = _num(per_token.get("wall"))
  floor_budget = _num(verdict.get("floor_budget_ms_per_token"))
  return {
      "current_best_tps": _num(anchor.get("current_best_tps")),
      "floor_tps": _num(anchor.get("same_host_vulkan_floor_tps")),
      "wall_ms_per_token": wall,
      "floor_budget_ms_per_token": floor_budget,
      "floor_gap_ms_per_token": max(0.0, wall - floor_budget),
      "non_kernel_overhead_ms_per_token": _num(
          per_token.get("non_kernel_overhead")),
      "overhead_only_ceiling_tok_s": _num(
          verdict.get("overhead_only_ceiling_tok_s")),
      "can_reach_floor_without_kernel_work": bool(
          verdict.get("can_reach_floor_without_kernel_work")),
      "noise_rel": _num(noise.get("rel")),
      "runs_since_significant_improvement": no_progress.get(
          "runs_since_significant_improvement"),
      "hard_stall_breached": no_progress.get("hard_stall_breached"),
      "review_recorded_for_current_best": no_progress.get(
          "review_recorded_for_current_best"),
      "glide_projected_runs_to_floor": glide.get("projected_runs_to_floor"),
      "glide_breached": glide.get("breached"),
  }


def _rejected_names(rejected: dict[str, Any]) -> set[str]:
  out: set[str] = set()
  for row in rejected.get("rejected", []):
    if isinstance(row, dict) and isinstance(row.get("route"), str):
      out.add(row["route"])
  return out


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


def _top_floor_buckets(wall_ms: dict[str, Any],
                       floor_gap: float) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  for key, value in wall_ms.items():
    ms = _num(value)
    if ms >= floor_gap:
      rows.append({"bucket": key, "ms_per_token": ms})
  rows.sort(key=lambda row: row["ms_per_token"], reverse=True)
  return rows


def compute(args: argparse.Namespace) -> dict[str, Any]:
  frontier = _load_json(args.frontier)
  routes = _load_json(args.routes)
  rejected = _load_json(args.rejected)
  seq162 = _load_json(args.seq162)
  seq158 = _load_json(args.seq158)
  decode_source = _read(args.decode_source)

  frontier_state = _frontier_state(frontier)
  floor_gap = frontier_state["floor_gap_ms_per_token"]
  selected_split = _dict(seq162.get("selected_ffn_wall_profile"))
  selected_wait = _dict(selected_split.get("submit_wait_ms_per_token"))
  max_submit_wait = _num(selected_split.get("max_submit_wait_ms_per_token"))
  wall_profile = _dict(seq162.get("wall_profile"))
  wall_ms = _dict(wall_profile.get("buckets_ms_per_token"))
  floor_buckets = _top_floor_buckets(wall_ms, floor_gap)
  largest_bucket = floor_buckets[0]["bucket"] if floor_buckets else ""
  rejected_names = _rejected_names(rejected)
  queue_drain_ms = _dict(
      _nested(seq158, "drain_site_profile", "buckets_ms_per_token"))

  required_rejected = {
      "current_resident_queue_drain_site_profile_explore_row",
      "current_resident_full_gpu_decode_loop_coarse_profile_bucket",
      "current_resident_hidden_state_carrier_per_boundary_handle_board",
      "current_selected_ffn_kernel_layout_component_board",
      "current_full_core_attention_front_kernel_algorithm_board",
      "current_linear_preconv_alpha_beta_algorithm_board",
      "current_linear_delta_algorithm_board",
      "current_offline_repack_streaming_layout_board",
      "current_moe_routed_down_fusion_board",
      "engine_resident_gpu_hot_loop_api_shell",
      "linear_setup_specialized_hoist",
  }
  missing_rejected = sorted(required_rejected - rejected_names)
  source_profile_markers = {
      "attention_front_noqueue_wall_split_absent": (
          "IQ36_ATTENTION_FRONT_NOQUEUE_WALL_SPLIT_PROFILE"
          not in decode_source),
      "existing_attention_front_stage_wall_present": (
          "attention_front_wall_ns" in decode_source
          and "attention_front_handoff_kernel_us" in decode_source),
      "existing_ffn_tail_stage_wall_present": (
          "ffn_tail_wall_ns" in decode_source
          and "RunGpuHybridFfnTail" in decode_source),
  }

  checks = [
      {
          "name": "seq162_selected_this_route",
          "pass": (
              seq162.get("required_checks_passed") is True
              and seq162.get("selected_next_route")
              == "post_selected_down_submit_split_route_gate"
              and _has_candidate(
                  routes, 162,
                  "accept_resident_selected_down_submit_split_profile_explore")
              and _has_switch(
                  routes, "select_post_selected_down_submit_split_route_gate",
                  162)
          ),
      },
      {
          "name": "selected_down_submit_wait_closed_as_floor_bucket",
          "pass": (
              max_submit_wait < floor_gap
              and _num(selected_wait.get("down_kernel_enqueue")) < floor_gap
              and _num(selected_wait.get("down_kernel_wait")) < floor_gap
              and seq162.get("speedup_claims_allowed") is False
          ),
          "detail": {
              "max_submit_wait_ms_per_token": max_submit_wait,
              "submit_wait_ms_per_token": selected_wait,
              "floor_gap_ms_per_token": floor_gap,
          },
      },
      {
          "name": "remaining_wall_buckets_floor_sized",
          "pass": (
              largest_bucket == "attention_front"
              and len(floor_buckets) >= 5
              and wall_ms.get("linear_preconv") is not None
              and wall_ms.get("selected_ffn") is not None
              and wall_ms.get("lm_head_gpu") is not None
              and wall_ms.get("ffn_tail") is not None
          ),
          "detail": {
              "floor_gap_ms_per_token": floor_gap,
              "largest_bucket": largest_bucket,
              "floor_buckets": floor_buckets,
          },
      },
      {
          "name": "queue_profile_bias_requires_noqueue_remaining_wall_attribution",
          "pass": (
              _num(queue_drain_ms.get("selected_down_wait_drain_site"))
              > floor_gap
              and _num(queue_drain_ms.get("attention_front_drain_site"))
              > floor_gap
              and max_submit_wait < floor_gap
              and _num(wall_ms.get("attention_front")) > floor_gap
          ),
          "detail": {
              "seq158_queue_drain_ms_per_token": queue_drain_ms,
              "seq162_noqueue_attention_front_ms_per_token": _num(
                  wall_ms.get("attention_front")),
          },
      },
      {
          "name": "closed_current_boards_recorded",
          "pass": not missing_rejected,
          "detail": {"missing": missing_rejected},
      },
      {
          "name": "source_has_no_noqueue_remaining_wall_split_yet",
          "pass": all(source_profile_markers.values()),
          "detail": source_profile_markers,
      },
      {
          "name": "frontier_still_below_floor_no_speed_claim",
          "pass": (
              frontier_state["current_best_tps"] < frontier_state["floor_tps"]
              and frontier_state["hard_stall_breached"] is True
              and frontier_state["review_recorded_for_current_best"] is True
              and frontier_state["can_reach_floor_without_kernel_work"] is True
          ),
          "detail": frontier_state,
      },
  ]
  required_checks_passed = all(bool(row.get("pass")) for row in checks)
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "inputs": {
          "frontier": _rel(args.frontier),
          "routes": _rel(args.routes),
          "rejected": _rel(args.rejected),
          "seq162_profile_gate": _rel(args.seq162),
          "seq158_queue_drain_profile": _rel(args.seq158),
          "decode_source": _rel(args.decode_source),
      },
      "frontier": frontier_state,
      "selected_down_submit_split": {
          "max_submit_wait_ms_per_token": max_submit_wait,
          "submit_wait_ms_per_token": selected_wait,
      },
      "remaining_wall": {
          "buckets_ms_per_token": wall_ms,
          "floor_buckets": floor_buckets,
          "largest_bucket": largest_bucket,
      },
      "queue_profile_context": {
          "seq158_drain_site_ms_per_token": queue_drain_ms,
      },
      "checks": checks,
      "required_checks_passed": required_checks_passed,
      "speedup_claims_allowed": False,
      "decode_probe_allowed": False,
      "component_probe_allowed": False,
      "source_profile_gate_required_before_decode": True,
      "disposition": (
          "select_resident_attention_front_noqueue_wall_split_source_gate"
          if required_checks_passed else
          "reject_post_selected_down_submit_split_route_gate"
      ),
      "selected_next_route": (
          "resident_attention_front_noqueue_wall_split_source_gate"
          if required_checks_passed else
          "post_selected_down_submit_split_route_fix_gate"
      ),
      "next_route_reason": (
          "Selected-down submit/wait is not floor-sized under noqueue profiling, "
          "but attention-front and other wall buckets remain floor-sized. The "
          "next unit is default-off noqueue wall attribution for attention-front "
          "and adjacent tail drain before any speed row."
          if required_checks_passed else
          "Route evidence is incomplete; fix the selected-down submit split "
          "profile or ledger state before selecting another source gate."
      ),
  }


def write_outputs(metrics: dict[str, Any], out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  manifest = {
      "schema_version": metrics["schema_version"],
      "workstream": metrics["workstream"],
      "tool": _rel(Path(__file__)),
      "selected_next_route": metrics["selected_next_route"],
      "speedup_claims_allowed": metrics["speedup_claims_allowed"],
      "inputs": metrics["inputs"],
  }
  (out_dir / "manifest.json").write_text(
      json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  failed = [row["name"] for row in metrics["checks"] if not row.get("pass")]
  summary = [
      "# Post Selected-Down Submit Split Route Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- largest_remaining_bucket: `{metrics['remaining_wall']['largest_bucket']}`",
      f"- decode_probe_allowed: `{str(metrics['decode_probe_allowed']).lower()}`",
      f"- failed_checks: `{failed}`",
      "",
      metrics["next_route_reason"],
      "",
      "This is route-control evidence. It does not claim speed.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(summary), encoding="utf-8")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--frontier", type=Path, default=DEFAULT_FRONTIER)
  parser.add_argument("--routes", type=Path, default=DEFAULT_ROUTES)
  parser.add_argument("--rejected", type=Path, default=DEFAULT_REJECTED)
  parser.add_argument("--seq162", type=Path, default=DEFAULT_SEQ162)
  parser.add_argument("--seq158", type=Path, default=DEFAULT_SEQ158)
  parser.add_argument("--decode-source", type=Path, default=DEFAULT_DECODE)
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
      "largest_remaining_bucket": metrics["remaining_wall"]["largest_bucket"],
      "out_dir": _rel(args.out_dir),
  }, sort_keys=True))
  return 0 if metrics["required_checks_passed"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
