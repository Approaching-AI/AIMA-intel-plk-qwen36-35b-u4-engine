#!/usr/bin/env python3
"""Select the next route after attention-front noqueue wall-split profiling.

This is route-control evidence only. It consumes the noqueue wall-split row
that proves attention-front wall is inside the call, not post-call bookkeeping,
then selects the next source-profile unit from the remaining floor-sized wall.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = "intel-qwen36-post-attention-front-noqueue-wall-split-route-gate-v0"

DEFAULT_FRONTIER = ACTIVE / "frontier.json"
DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_REJECTED = ACTIVE / "rejected-routes.json"
DEFAULT_SEQ166 = (
    ROOT
    / "output/resident-attention-front-noqueue-wall-split-profile-explore-gate-20260708Tseq166Z"
    / "metrics.json"
)
DEFAULT_DECODE = ROOT / "tools/intel-qwen36-r2-gpu-decode-smoke.py"
DEFAULT_OUT_DIR = (
    ROOT
    / "output/post-attention-front-noqueue-wall-split-route-gate-20260708Tseq167Z"
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
  seq166 = _load_json(args.seq166)
  decode_source = _read(args.decode_source)

  frontier_state = _frontier_state(frontier)
  floor_gap = frontier_state["floor_gap_ms_per_token"]
  split = _dict(seq166.get("attention_front_noqueue_wall_split_profile"))
  split_ms = _dict(split.get("buckets_ms_per_token"))
  wall = _dict(seq166.get("wall_profile"))
  wall_ms = _dict(wall.get("buckets_ms_per_token"))
  floor_buckets = _top_floor_buckets(wall_ms, floor_gap)
  largest_wall = floor_buckets[0]["bucket"] if floor_buckets else ""
  rejected_names = _rejected_names(rejected)

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
      "attention_front_call_wall_profile_absent": (
          "IQ36_ATTENTION_FRONT_CALL_WALL_PROFILE" not in decode_source),
      "existing_attention_front_call_split_present": (
          "IQ36_ATTENTION_FRONT_NOQUEUE_WALL_SPLIT_PROFILE" in decode_source
          and "attention_front_call_wall_ns" in decode_source
          and "attention_front_post_call_wall_ns" in decode_source),
      "run_gpu_attention_front_body_present": (
          "AttentionFrontRun RunGpuAttentionFront(" in decode_source
          and "RunResidentPackedQ4X8ThenResidentResidualRmsNorm" in decode_source),
  }

  checks = [
      {
          "name": "seq166_selected_this_route",
          "pass": (
              seq166.get("required_checks_passed") is True
              and seq166.get("selected_next_route")
              == "post_attention_front_noqueue_wall_split_route_gate"
              and seq166.get("speedup_claims_allowed") is False
          ),
          "detail": {
              "seq166_disposition": seq166.get("disposition"),
              "seq166_selected_next_route": seq166.get("selected_next_route"),
          },
      },
      {
          "name": "attention_front_call_not_post_call_is_floor_bucket",
          "pass": (
              _num(split_ms.get("attention_front_call")) >= floor_gap
              and _num(split_ms.get("attention_front_post_call")) < floor_gap
              and _num(split_ms.get("attention_front_call"))
              > _num(split_ms.get("ffn_tail_call"))
          ),
          "detail": {
              "floor_gap_ms_per_token": floor_gap,
              "split_ms_per_token": split_ms,
          },
      },
      {
          "name": "remaining_wall_buckets_floor_sized",
          "pass": (
              largest_wall == "attention_front"
              and len(floor_buckets) >= 5
              and _num(wall_ms.get("linear_preconv")) >= floor_gap
              and _num(wall_ms.get("selected_ffn")) >= floor_gap
              and _num(wall_ms.get("lm_head_gpu")) >= floor_gap
              and _num(wall_ms.get("ffn_tail")) >= floor_gap
          ),
          "detail": {
              "floor_gap_ms_per_token": floor_gap,
              "largest_bucket": largest_wall,
              "floor_buckets": floor_buckets,
          },
      },
      {
          "name": "closed_current_boards_recorded",
          "pass": not missing_rejected,
          "detail": {"missing": missing_rejected},
      },
      {
          "name": "source_has_no_inner_attention_front_call_profile_yet",
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
          "seq166_profile_gate": _rel(args.seq166),
          "decode_source": _rel(args.decode_source),
      },
      "frontier": frontier_state,
      "attention_front_split": {
          "buckets_ms_per_token": split_ms,
          "largest_bucket": split.get("largest_bucket"),
          "largest_ms_per_token": split.get("largest_ms_per_token"),
      },
      "remaining_wall": {
          "buckets_ms_per_token": wall_ms,
          "floor_buckets": floor_buckets,
          "largest_bucket": largest_wall,
      },
      "checks": checks,
      "required_checks_passed": required_checks_passed,
      "speedup_claims_allowed": False,
      "decode_probe_allowed": False,
      "component_probe_allowed": False,
      "source_profile_gate_required_before_decode": True,
      "disposition": (
          "select_resident_attention_front_call_wall_profile_source_gate"
          if required_checks_passed else
          "reject_post_attention_front_noqueue_wall_split_route_gate"
      ),
      "selected_next_route": (
          "resident_attention_front_call_wall_profile_source_gate"
          if required_checks_passed else
          "post_attention_front_noqueue_wall_split_route_fix_gate"
      ),
      "next_route_reason": (
          "The noqueue wall split proves the floor-sized attention-front wall "
          "is inside RunGpuAttentionFront call, not post-call bookkeeping, and "
          "it remains the largest wall bucket. The next unit is default-off "
          "inner call wall attribution for attention-front setup, output "
          "projection, residual add/RMSNorm handoff, and handle paths before "
          "any speed row."
          if required_checks_passed else
          "Route evidence is incomplete; fix the attention-front wall-split "
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
      "# Post Attention-Front Noqueue Wall-Split Route Gate",
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
  parser.add_argument("--seq166", type=Path, default=DEFAULT_SEQ166)
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
