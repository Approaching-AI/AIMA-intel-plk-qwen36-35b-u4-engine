#!/usr/bin/env python3
"""Select the next route after fixed setup hoisting misses the speed gate.

This is route-control evidence only. It reads the live frontier, route boards,
and the specialized setup-hoist gate output, then selects the next concrete
unit without launching a decode row.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = "intel-qwen36-post-setup-hoist-route-gate-v0"
DEFAULT_FRONTIER = ACTIVE / "frontier.json"
DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_REJECTED = ACTIVE / "rejected-routes.json"
DEFAULT_HOIST_GATE = (
    ROOT / "output/linear-setup-specialized-hoist-gate-20260707Tseq105-107Z"
    / "metrics.json"
)
DEFAULT_OUT_DIR = ROOT / "output/post-setup-hoist-route-gate-20260707Tseq108Z"


def _load_json(path: Path) -> Any:
  return json.loads(path.read_text(encoding="utf-8"))


def _num(value: Any) -> float:
  return float(value) if isinstance(value, (int, float)) else 0.0


def _rel(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path)


def _route_names(rejected: dict[str, Any]) -> set[str]:
  names: set[str] = set()
  for row in rejected.get("rejected", []):
    if isinstance(row, dict) and isinstance(row.get("route"), str):
      names.add(row["route"])
  return names


def _has_switch(routes: dict[str, Any], decision: str, seq_covered: int) -> bool:
  for row in routes.get("switch_decisions", []):
    if not isinstance(row, dict):
      continue
    if (
        row.get("decision") == decision
        and _num(row.get("seq_covered")) >= seq_covered
        and row.get("resolved") is True
    ):
      return True
  return False


def _parked_route(routes: dict[str, Any], route_id: str) -> dict[str, Any] | None:
  for row in routes.get("parked_routes", []):
    if isinstance(row, dict) and row.get("id") == route_id:
      return row
  return None


def _frontier_state(frontier: dict[str, Any]) -> dict[str, Any]:
  goal_anchor = frontier.get("goal_anchor")
  goal_anchor = goal_anchor if isinstance(goal_anchor, dict) else {}
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
  floor_gap = max(0.0, wall - floor_budget)
  return {
      "current_best_tps": _num(goal_anchor.get("current_best_tps")),
      "floor_tps": _num(goal_anchor.get("same_host_vulkan_floor_tps")),
      "wall_ms_per_token": wall,
      "floor_budget_ms_per_token": floor_budget,
      "floor_gap_ms_per_token": floor_gap,
      "non_kernel_overhead_ms_per_token": _num(per_token.get("non_kernel_overhead")),
      "noise_rel": _num(noise.get("rel")),
      "runs_since_significant_improvement": no_progress.get(
          "runs_since_significant_improvement"),
      "soft_reflection_breached": no_progress.get("soft_reflection_breached"),
      "hard_stall_breached": no_progress.get("hard_stall_breached"),
  }


def _stage_gaps(frontier: dict[str, Any]) -> dict[str, dict[str, Any]]:
  budget = frontier.get("goal_budget")
  budget = budget if isinstance(budget, dict) else {}
  out: dict[str, dict[str, Any]] = {}
  for row in budget.get("stage_kernel_gap_estimates_ms_per_token", []):
    if isinstance(row, dict) and isinstance(row.get("stage"), str):
      out[row["stage"]] = row
  return out


def _stage_walls(frontier: dict[str, Any]) -> list[dict[str, Any]]:
  budget = frontier.get("goal_budget")
  budget = budget if isinstance(budget, dict) else {}
  rows = budget.get("top_stage_walls_ms_per_token")
  return rows if isinstance(rows, list) else []


def compute(args: argparse.Namespace) -> dict[str, Any]:
  frontier = _load_json(args.frontier)
  routes = _load_json(args.routes)
  rejected = _load_json(args.rejected)
  hoist_gate = _load_json(args.hoist_gate)
  frontier_state = _frontier_state(frontier)
  stage_gaps = _stage_gaps(frontier)
  stage_walls = _stage_walls(frontier)
  rejected_names = _route_names(rejected)
  offline_route = _parked_route(routes, "offline_repack_streaming_layout")

  selected_gap = stage_gaps.get("selected_ffn", {})
  selected_gap_ms = _num(selected_gap.get("gap_ms_per_token"))
  selected_wall_ms = 0.0
  if stage_walls and isinstance(stage_walls[0], dict):
    top_stage = stage_walls[0].get("stage")
  else:
    top_stage = None
  for row in stage_walls:
    if isinstance(row, dict) and row.get("stage") == "selected_ffn":
      selected_wall_ms = _num(row.get("ms_per_token"))

  closed_setup_routes = [
      "generic_token_setup_cache",
      "linear_setup_specialized_hoist",
  ]
  closed_per_boundary_examples = [
      "gpu_selected_shared_q6_down_tail_rowgroup_local_reduce_component",
      "gpu_linear_preconv_shared_q8_qkv_conv_root_component",
      "gpu_attention_linear_event_lifetime_combined_alias",
      "gpu_selected_gateup_top16_indexed_material_component",
  ]
  missing_closed = [
      name for name in closed_per_boundary_examples
      if name not in rejected_names
  ]

  hoist_profile = hoist_gate.get("specialized_hoist_source_profile")
  hoist_profile = hoist_profile if isinstance(hoist_profile, dict) else {}
  checks = [
      {
          "name": "setup_hoist_gate_passed_and_rejected_speed_cut",
          "pass": (
              hoist_gate.get("required_checks_passed") is True
              and hoist_gate.get("disposition")
                  == "reject_specialized_setup_hoist_as_speed_cut"
          ),
          "detail": {
              "hoist_gate": _rel(args.hoist_gate),
              "disposition": hoist_gate.get("disposition"),
          },
      },
      {
          "name": "remaining_setup_profile_is_smaller_than_floor_gap",
          "pass": (
              _num(hoist_profile.get("dispatch_profiled_after_ms_per_token"))
              < frontier_state["floor_gap_ms_per_token"]
              and _num(hoist_profile.get("layer_dispatch_gap_after_ms_per_token"))
              < frontier_state["floor_gap_ms_per_token"]
          ),
          "detail": {
              "floor_gap_ms_per_token": frontier_state["floor_gap_ms_per_token"],
              "dispatch_profiled_after_ms_per_token": hoist_profile.get(
                  "dispatch_profiled_after_ms_per_token"),
              "layer_dispatch_gap_after_ms_per_token": hoist_profile.get(
                  "layer_dispatch_gap_after_ms_per_token"),
          },
      },
      {
          "name": "selected_ffn_stage_gap_is_large_enough_to_cover_floor",
          "pass": (
              top_stage == "selected_ffn"
              and selected_gap_ms > frontier_state["floor_gap_ms_per_token"]
              and selected_wall_ms > 0.0
          ),
          "detail": {
              "top_stage": top_stage,
              "selected_ffn_wall_ms_per_token": selected_wall_ms,
              "selected_ffn_gap_ms_per_token": selected_gap_ms,
              "floor_gap_ms_per_token": frontier_state["floor_gap_ms_per_token"],
          },
      },
      {
          "name": "offline_repack_route_is_available_for_kernel_component_proof",
          "pass": offline_route is not None and int(_num(offline_route.get("rank"))) == 1,
          "detail": {
              "route_id": None if offline_route is None else offline_route.get("id"),
              "rank": None if offline_route is None else offline_route.get("rank"),
          },
      },
      {
          "name": "closed_micro_routes_are_not_selected_again",
          "pass": not missing_closed,
          "detail": {
              "missing_closed_route_records": missing_closed,
              "setup_routes_closed_by_this_session": closed_setup_routes,
          },
      },
      {
          "name": "prior_resident_overhead_switches_recorded",
          "pass": (
              _has_switch(
                  routes,
                  "switch_to_run_gpu_hybrid_decode_token_unprofiled_source_profile_gate",
                  97,
              )
              and _has_switch(routes, "switch_to_linear_setup_specialized_hoist_source_gate", 99)
          ),
      },
  ]
  required = all(check["pass"] for check in checks)
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "required_checks_passed": required,
      "speedup_claims_allowed": False,
      "disposition": (
          "select_selected_ffn_kernel_layout_component_gate"
          if required else "post_setup_hoist_route_gate_failed"
      ),
      "selected_next_route": (
          "selected_ffn_kernel_layout_component_gate"
          if required else "manual_review_post_setup_hoist_route"
      ),
      "next_action": (
          "Open a kernel-side selected-FFN material/layout component gate. "
          "The next unit must be a component proof, not a token-emitting "
          "decode row: use the accepted selected+shared Q4/Q6 down stack as "
          "the reference, preserve real batch-1 selected/shared work, and "
          "show at least floor-gap-sized expected movement before decode. "
          "Do not repeat setup-cache, API-shell, top16, shared-Q8 preconv, "
          "simple attention/linear handoff, or down-tail rowgroup variants."
          if required else "Fix failed route-gate checks before changing route."
      ),
      "inputs": {
          "frontier": _rel(args.frontier),
          "routes": _rel(args.routes),
          "rejected": _rel(args.rejected),
          "hoist_gate": _rel(args.hoist_gate),
      },
      "frontier": frontier_state,
      "stage_gaps": stage_gaps,
      "stage_walls": stage_walls,
      "hoist_gate_summary": {
          "disposition": hoist_gate.get("disposition"),
          "selected_next_route": hoist_gate.get("selected_next_route"),
          "hoist_tps": ((hoist_gate.get("specialized_hoist") or {}).get("tps")),
          "hoist_profile_tps": hoist_profile.get("tps"),
          "dispatch_profiled_after_ms_per_token": hoist_profile.get(
              "dispatch_profiled_after_ms_per_token"),
          "layer_dispatch_gap_after_ms_per_token": hoist_profile.get(
              "layer_dispatch_gap_after_ms_per_token"),
      },
      "selected_route_reason": {
          "route_family": "offline_repack_streaming_layout",
          "selected_stage": "selected_ffn",
          "selected_ffn_wall_ms_per_token": selected_wall_ms,
          "selected_ffn_gap_ms_per_token": selected_gap_ms,
          "floor_gap_ms_per_token": frontier_state["floor_gap_ms_per_token"],
          "parked_route_rank": None if offline_route is None else offline_route.get("rank"),
      },
      "checks": checks,
  }


def write_outputs(out_dir: Path, payload: dict[str, Any]) -> None:
  out_dir.mkdir(parents=True, exist_ok=False)
  (out_dir / "metrics.json").write_text(
      json.dumps(payload, indent=2, sort_keys=True) + "\n",
      encoding="utf-8")
  manifest = {
      "schema_version": payload["schema_version"],
      "workstream": payload["workstream"],
      "tool": "tools/intel-qwen36-post-setup-hoist-route-gate.py",
      "inputs": payload["inputs"],
      "selected_next_route": payload["selected_next_route"],
      "speedup_claims_allowed": False,
  }
  (out_dir / "manifest.json").write_text(
      json.dumps(manifest, indent=2, sort_keys=True) + "\n",
      encoding="utf-8")
  failed = [row["name"] for row in payload["checks"] if not row["pass"]]
  reason = payload["selected_route_reason"]
  lines = [
      "# Post-Setup-Hoist Route Gate",
      "",
      f"- required checks passed: `{str(payload['required_checks_passed']).lower()}`",
      f"- disposition: `{payload['disposition']}`",
      f"- selected next route: `{payload['selected_next_route']}`",
      f"- selected stage: `{reason['selected_stage']}`",
      f"- selected FFN wall/gap: `{reason['selected_ffn_wall_ms_per_token']}` / `{reason['selected_ffn_gap_ms_per_token']}` ms/token",
      f"- floor gap: `{reason['floor_gap_ms_per_token']}` ms/token",
      f"- speedup claims allowed: `{str(payload['speedup_claims_allowed']).lower()}`",
      f"- failed checks: `{failed}`",
      "",
      payload["next_action"],
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--frontier", type=Path, default=DEFAULT_FRONTIER)
  parser.add_argument("--routes", type=Path, default=DEFAULT_ROUTES)
  parser.add_argument("--rejected", type=Path, default=DEFAULT_REJECTED)
  parser.add_argument("--hoist-gate", type=Path, default=DEFAULT_HOIST_GATE)
  parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
  args = parser.parse_args()
  payload = compute(args)
  write_outputs(args.out_dir, payload)
  print(json.dumps({
      "required_checks_passed": payload["required_checks_passed"],
      "disposition": payload["disposition"],
      "selected_next_route": payload["selected_next_route"],
      "out_dir": _rel(args.out_dir),
  }, sort_keys=True))
  return 0 if payload["required_checks_passed"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
