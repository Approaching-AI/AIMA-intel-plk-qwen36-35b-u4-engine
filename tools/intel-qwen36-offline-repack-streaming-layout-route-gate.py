#!/usr/bin/env python3
"""Audit the parked offline-repack/streaming-layout route after seq114.

This is route-control evidence only. It checks whether the current pure
offline-layout board has a legal batch-1 component large enough to justify
decode, or whether the next route must move to a broader MoE/down fusion gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = "intel-qwen36-offline-repack-streaming-layout-route-gate-v0"

DEFAULT_FRONTIER = ACTIVE / "frontier.json"
DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_ACCEPTED = ACTIVE / "accepted-cuts.json"
DEFAULT_REJECTED = ACTIVE / "rejected-routes.json"
DEFAULT_SEQ87 = ROOT / "output/selected-ffn-material-layout-gate-20260707Tseq87Z/metrics.json"
DEFAULT_SEQ109 = ROOT / "output/selected-ffn-kernel-layout-component-gate-20260707Tseq109Z/metrics.json"
DEFAULT_Q4_FULL = ROOT / "output/gpu-q4x8-qmatvec-ffn-gateup-full-20260702T225500Z/probe-result.json"
DEFAULT_Q4_DOWN_FULL = ROOT / "output/gpu-q6-qmatvec-ffn-down-full-20260702T233000Z/probe-result.json"
DEFAULT_Q6_RAW_FULL = ROOT / "output/gpu-q6-qmatvec-layer7-ffn-down-full-20260702T234500Z/probe-result.json"
DEFAULT_Q6_PLANE_STREAM = ROOT / "output/gpu-repack-stream-probe-20260704T080629Z/probe-result.json"
DEFAULT_OUT_DIR = ROOT / "output/offline-repack-streaming-layout-route-gate-20260707Tseq115Z"


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


def _stage_gap(frontier: dict[str, Any], stage: str) -> float:
  rows = _nested(frontier, "goal_budget", "stage_kernel_gap_estimates_ms_per_token")
  if not isinstance(rows, list):
    return 0.0
  for row in rows:
    if isinstance(row, dict) and row.get("stage") == stage:
      return _num(row.get("gap_ms_per_token"))
  return 0.0


def _top_stage(frontier: dict[str, Any]) -> tuple[str | None, float]:
  rows = _nested(frontier, "goal_budget", "top_stage_walls_ms_per_token")
  if not isinstance(rows, list) or not rows or not isinstance(rows[0], dict):
    return None, 0.0
  return rows[0].get("stage"), _num(rows[0].get("ms_per_token"))


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


def _gb_s(obj: dict[str, Any], *keys: str) -> float:
  return _num(_nested(obj, *keys))


def compute(args: argparse.Namespace) -> dict[str, Any]:
  frontier = _load_json(args.frontier)
  routes = _load_json(args.routes)
  accepted = _load_json(args.accepted)
  rejected = _load_json(args.rejected)
  seq87 = _load_json(args.seq87)
  seq109 = _load_json(args.seq109)
  q4_full = _load_json(args.q4_full)
  q4_down_full = _load_json(args.q4_down_full)
  q6_raw_full = _load_json(args.q6_raw_full)
  q6_plane = _load_json(args.q6_plane_stream)

  frontier_state = _frontier_state(frontier)
  floor_gap = frontier_state["floor_gap_ms_per_token"]
  selected_gap = _stage_gap(frontier, "selected_ffn")
  top_stage, top_stage_wall = _top_stage(frontier)
  accepted_ids = _accepted_ids(accepted)
  rejected_names = _rejected_names(rejected)
  offline_route = _parked_route(routes, "offline_repack_streaming_layout")
  moe_route = _parked_route(routes, "moe_routed_down_fusion")

  seq87_derived = seq87.get("derived") if isinstance(seq87.get("derived"), dict) else {}
  seq109_summary = (
      seq109.get("selected_ffn_component_summary")
      if isinstance(seq109.get("selected_ffn_component_summary"), dict) else {}
  )
  q4_current_gb_s = _num(seq87_derived.get("q4_current_combined_gb_s"))
  q4_full_gb_s = _gb_s(q4_full, "gpu_effective_packed_gb_s")
  q4_down_full_gb_s = _gb_s(q4_down_full, "gpu_effective_payload_gb_s")
  q6_raw_full_gb_s = _gb_s(q6_raw_full, "gpu_effective_payload_gb_s")
  q6_plane_raw_gb_s = _gb_s(q6_plane, "aggregate", "raw_kernel_min_gb_s_mean")
  q6_plane_repacked_gb_s = _gb_s(
      q6_plane, "aggregate", "repacked_kernel_min_gb_s_mean")
  q6_plane_quant_only_gb_s = _gb_s(
      q6_plane, "aggregate", "repacked_quant_only_kernel_min_gb_s_mean")
  q6_plane_ratio = (
      q6_plane_repacked_gb_s / q6_plane_raw_gb_s
      if q6_plane_raw_gb_s > 0.0 else 0.0
  )
  q6_plane_quant_ratio = (
      q6_plane_quant_only_gb_s / q6_plane_raw_gb_s
      if q6_plane_raw_gb_s > 0.0 else 0.0
  )
  q4_full_vs_current = (
      q4_full_gb_s / q4_current_gb_s if q4_current_gb_s > 0.0 else 0.0
  )

  required_accepted = {
      "selected_shared_q4_gateup_no_concat",
      "selected_shared_q6_down_combined",
      "selected_shared_q4q6_down_combined",
      "selected_shared_q6_down_combined_per_expert_cold_cache",
  }
  required_rejected = {
      "gpu_q6_plane_v0_stream_selected_down_reopen_gate",
      "gpu_selected_down_q6_plane_layout_probe",
      "gpu_selected_gateup_top16_indexed_material_component",
      "current_selected_ffn_kernel_layout_component_board",
      "gpu_selected_all_expert_residency",
  }
  missing_accepted = sorted(required_accepted - accepted_ids)
  missing_rejected = sorted(required_rejected - rejected_names)

  top16_speedup = _num(seq109_summary.get("top16_shell_speedup"))
  top16_required = _num(seq109_summary.get("top16_required_speedup"))
  q6_occ = _num(seq109_summary.get("q6_occupancy4_scaled_speedup"))
  q6_required = _num(seq109_summary.get("q6_required_speedup"))
  q4_occ = _num(seq87_derived.get("q4_gateup_occupancy4_scaled_ratio"))

  checks = [
      {
          "name": "seq114_selected_offline_repack_gate",
          "pass": _has_switch(
              routes,
              "close_current_linear_delta_board_switch_to_offline_repack_streaming_layout_gate",
              114,
          ),
      },
      {
          "name": "offline_repack_is_rank1_parked_route",
          "pass": offline_route is not None and int(_num(offline_route.get("rank"))) == 1,
          "detail": {
              "route_id": None if offline_route is None else offline_route.get("id"),
              "rank": None if offline_route is None else offline_route.get("rank"),
          },
      },
      {
          "name": "frontier_below_floor_and_selected_ffn_still_floor_sized",
          "pass": (
              frontier_state["current_best_tps"] < frontier_state["floor_tps"]
              and top_stage == "selected_ffn"
              and selected_gap > floor_gap
          ),
          "detail": {
              "top_stage": top_stage,
              "top_stage_wall_ms_per_token": top_stage_wall,
              "selected_gap_ms_per_token": selected_gap,
              "floor_gap_ms_per_token": floor_gap,
          },
      },
      {
          "name": "accepted_stack_is_already_combined_selected_shared_q4q6",
          "pass": not missing_accepted,
          "detail": {"missing": missing_accepted},
      },
      {
          "name": "full_tensor_streaming_proofs_are_not_batch1_selected_layout",
          "pass": (
              q4_full.get("required_checks_passed") is True
              and q4_down_full.get("required_checks_passed") is True
              and q6_raw_full.get("required_checks_passed") is True
              and q4_full_vs_current > 1.5
              and q4_occ > 1.5
          ),
          "detail": {
              "q4_gateup_full_gb_s": q4_full_gb_s,
              "q4_down_full_gb_s": q4_down_full_gb_s,
              "q6_raw_full_gb_s": q6_raw_full_gb_s,
              "q4_full_vs_current_combined": q4_full_vs_current,
              "q4_occupancy4_scaled_ratio": q4_occ,
              "batch1_note": "full/occupancy4 rows repeat work and are not legal decode evidence",
          },
      },
      {
          "name": "q6_plane_v0_stream_layout_is_not_material",
          "pass": (
              q6_plane.get("aggregate", {}).get("checksum_match_all") is True
              and q6_plane_ratio < 1.2
              and q6_plane_quant_ratio < 1.0
          ),
          "detail": {
              "raw_gb_s": q6_plane_raw_gb_s,
              "repacked_gb_s": q6_plane_repacked_gb_s,
              "ratio": q6_plane_ratio,
              "quant_only_gb_s": q6_plane_quant_only_gb_s,
              "quant_only_ratio": q6_plane_quant_ratio,
          },
      },
      {
          "name": "materialized_top16_gateup_component_already_closed",
          "pass": 0.0 < top16_speedup < top16_required,
          "detail": {
              "top16_shell_speedup": top16_speedup,
              "top16_required_speedup": top16_required,
          },
      },
      {
          "name": "q6_selected_shared_occupancy_is_synthetic_and_subfloor",
          "pass": 0.0 < q6_occ < q6_required,
          "detail": {
              "q6_occupancy4_scaled_speedup": q6_occ,
              "q6_required_speedup": q6_required,
          },
      },
      {
          "name": "current_selected_ffn_layout_board_is_closed",
          "pass": (
              seq109.get("required_checks_passed") is True
              and seq109.get("decode_probe_allowed") is False
              and seq109.get("disposition")
                  == "close_current_selected_ffn_kernel_layout_component_route"
          ),
          "detail": {
              "seq109": _rel(args.seq109),
              "disposition": seq109.get("disposition"),
          },
      },
      {
          "name": "closed_route_board_blocks_current_offline_repack_variants",
          "pass": not missing_rejected,
          "detail": {"missing": missing_rejected},
      },
      {
          "name": "moe_routed_down_fusion_is_available_as_next_parked_route",
          "pass": moe_route is not None,
          "detail": {
              "route_id": None if moe_route is None else moe_route.get("id"),
              "rank": None if moe_route is None else moe_route.get("rank"),
          },
      },
  ]
  required_checks_passed = all(bool(row.get("pass")) for row in checks)
  disposition = "close_current_offline_repack_streaming_layout_board"
  selected_next_route = "moe_routed_down_fusion_route_gate"

  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "inputs": {
          "frontier": _rel(args.frontier),
          "routes": _rel(args.routes),
          "accepted": _rel(args.accepted),
          "rejected": _rel(args.rejected),
          "seq87_material_gate": _rel(args.seq87),
          "seq109_selected_ffn_gate": _rel(args.seq109),
          "q4_full": _rel(args.q4_full),
          "q4_down_full": _rel(args.q4_down_full),
          "q6_raw_full": _rel(args.q6_raw_full),
          "q6_plane_stream": _rel(args.q6_plane_stream),
      },
      "frontier": frontier_state,
      "offline_repack_summary": {
          "q4_gateup_full_gb_s": q4_full_gb_s,
          "q4_down_full_gb_s": q4_down_full_gb_s,
          "q6_raw_full_gb_s": q6_raw_full_gb_s,
          "q4_current_combined_gb_s": q4_current_gb_s,
          "q4_full_vs_current_combined": q4_full_vs_current,
          "q4_occupancy4_scaled_ratio": q4_occ,
          "q6_plane_stream_ratio": q6_plane_ratio,
          "q6_plane_quant_only_ratio": q6_plane_quant_ratio,
          "top16_shell_speedup": top16_speedup,
          "top16_required_speedup": top16_required,
          "q6_occupancy4_scaled_speedup": q6_occ,
          "q6_required_speedup": q6_required,
      },
      "checks": checks,
      "required_checks_passed": required_checks_passed,
      "decode_probe_allowed": False,
      "component_probe_allowed": False,
      "speedup_claims_allowed": False,
      "disposition": disposition,
      "selected_next_route": selected_next_route,
      "next_route_reason": (
          "Pure offline repack/layout has no current legal batch-1 component: "
          "Q4 full/occupancy rows are material but synthetic, top16 material is "
          "exact but below its floor-covering ratio, Q6 plane-v0 lacks a material "
          "streaming win, and Q6 selected/shared occupancy is below the required "
          "ratio. The next admissible work must be a broader MoE/down fusion gate."
      ),
      "next_action": (
          "Build moe_routed_down_fusion_route_gate as route/design evidence first. "
          "Do not launch a decode row or another pure repack/material component "
          "without a new legality argument and floor-sized component proof."
      ),
  }


def write_outputs(metrics: dict[str, Any], out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
  )
  summary = [
      "# Offline Repack Streaming Layout Route Gate",
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
  parser.add_argument("--seq87", type=Path, default=DEFAULT_SEQ87)
  parser.add_argument("--seq109", type=Path, default=DEFAULT_SEQ109)
  parser.add_argument("--q4-full", type=Path, default=DEFAULT_Q4_FULL)
  parser.add_argument("--q4-down-full", type=Path, default=DEFAULT_Q4_DOWN_FULL)
  parser.add_argument("--q6-raw-full", type=Path, default=DEFAULT_Q6_RAW_FULL)
  parser.add_argument("--q6-plane-stream", type=Path, default=DEFAULT_Q6_PLANE_STREAM)
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
