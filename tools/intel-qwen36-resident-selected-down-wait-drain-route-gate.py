#!/usr/bin/env python3
"""Route gate after the drain-site profile identified selected-down wait.

This is route-control evidence only. It separates the queue-profiling drain
signal from the noqueue speed lane before authorizing another implementation or
token row.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = "intel-qwen36-resident-selected-down-wait-drain-route-gate-v0"

DEFAULT_FRONTIER = ACTIVE / "frontier.json"
DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_REJECTED = ACTIVE / "rejected-routes.json"
DEFAULT_SEQ158 = (
    ROOT
    / "output/resident-queue-drain-site-profile-explore-gate-20260708Tseq158Z"
    / "metrics.json"
)
DEFAULT_BEST_SMOKE = (
    ROOT
    / "output/r2-gpu-selected-shared-q4q6-down-cold-q6-experts-noqueue-speed-20260705T143006Z"
    / "smoke.json"
)
DEFAULT_ENGINE = ROOT / "engine/src/gpu_q4x8_matvec.cpp"
DEFAULT_HEADER = ROOT / "engine/include/intel_qwen36/gpu_q4x8_matvec.hpp"
DEFAULT_DECODE = ROOT / "tools/intel-qwen36-r2-gpu-decode-smoke.py"
DEFAULT_OUT_DIR = (
    ROOT
    / "output/resident-selected-down-wait-drain-route-gate-20260708Tseq159Z"
)


def _load_json(path: Path) -> Any:
  return json.loads(path.read_text(encoding="utf-8"))


def _read(path: Path) -> str:
  return path.read_text(encoding="utf-8")


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
      "hard_stall_breached": no_progress.get("hard_stall_breached"),
      "review_recorded_for_current_best": no_progress.get(
          "review_recorded_for_current_best"),
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


def _tokens(row: dict[str, Any]) -> float:
  return (
      _num(row.get("decode_continuation_output_tokens"))
      or _num(row.get("decode_tokens"))
      or 1.0
  )


def _ms_per_token(ns: Any, tokens: float) -> float:
  return _num(ns) / 1_000_000.0 / max(tokens, 1.0)


def compute(args: argparse.Namespace) -> dict[str, Any]:
  frontier = _load_json(args.frontier)
  routes = _load_json(args.routes)
  rejected = _load_json(args.rejected)
  seq158 = _load_json(args.seq158)
  best = _load_json(args.best_smoke)
  source = "\n".join([
      _read(args.engine_source),
      _read(args.engine_header),
      _read(args.decode_source),
  ])

  frontier_state = _frontier_state(frontier)
  floor_gap = frontier_state["floor_gap_ms_per_token"]
  drain_profile = seq158.get("drain_site_profile")
  drain_profile = drain_profile if isinstance(drain_profile, dict) else {}
  buckets = drain_profile.get("buckets_ms_per_token")
  buckets = buckets if isinstance(buckets, dict) else {}
  queue_profile_selected_wait = _num(buckets.get("selected_down_wait_drain_site"))

  best_tokens = _tokens(best)
  best_selected_profile = best.get("selected_ffn_wall_profile_ns")
  best_selected_profile = (
      best_selected_profile if isinstance(best_selected_profile, dict) else {})
  noqueue_selected_wait = _ms_per_token(
      best_selected_profile.get("down_kernel_wait"), best_tokens)
  noqueue_down_event_profile = _ms_per_token(
      best_selected_profile.get("down_event_profile"), best_tokens)
  queue_profile_extra = max(0.0, queue_profile_selected_wait - noqueue_selected_wait)

  required_rejected = {
      "gpu_q6_defer_finish_without_tail_drain_elimination",
      "gpu_q6_nonatomic_down_tail_decode_fusion",
      "gpu_selected_shared_q6_down_tail_rowgroup_local_reduce_component",
      "current_selected_ffn_kernel_layout_component_board",
      "current_moe_routed_down_fusion_board",
      "current_resident_hidden_state_carrier_per_boundary_handle_board",
      "current_resident_queue_drain_site_profile_explore_row",
  }
  rejected_names = _rejected_names(rejected)
  missing_rejected = sorted(required_rejected - rejected_names)

  source_semantics = {
      "defer_requires_no_event_collection": (
          "return enabled && !OpenClEventCollectionEnabled();" in source),
      "noqueue_disables_event_collection": (
          "IQ36_OPENCL_NO_QUEUE_PROFILING" in source
          and "return OpenClEventCollectionEnabled() ? event : nullptr;" in source),
      "kernel_wait_still_unsplit": (
          "kernel_wait_wall_ns" in source
          and "kernel_enqueue_wall_ns" not in source
          and "down_kernel_enqueue" not in source),
  }

  checks = [
      {
          "name": "seq158_selected_this_route",
          "pass": (
              seq158.get("required_checks_passed") is True
              and seq158.get("selected_next_route")
              == "resident_selected_down_wait_drain_route_gate"
              and _has_candidate(
                  routes, 158, "accept_resident_queue_drain_site_profile_explore")
              and _has_switch(
                  routes, "select_resident_selected_down_wait_drain_route_gate", 158)
          ),
      },
      {
          "name": "queue_profile_selected_down_bucket_floor_sized",
          "pass": (
              drain_profile.get("largest_bucket") == "selected_down_wait_drain_site"
              and queue_profile_selected_wait > floor_gap
              and seq158.get("speedup_claims_allowed") is False
          ),
          "detail": {
              "queue_profile_selected_wait_ms_per_token": queue_profile_selected_wait,
              "floor_gap_ms_per_token": floor_gap,
          },
      },
      {
          "name": "noqueue_frontier_still_has_floor_sized_selected_down_submit_wall",
          "pass": (
              best.get("opencl_no_queue_profiling") is True
              and best.get("defer_ffn_down_finish_bundle") is True
              and best.get("top1_matches_native") is True
              and noqueue_selected_wait > floor_gap
          ),
          "detail": {
              "noqueue_selected_down_kernel_wait_ms_per_token": noqueue_selected_wait,
              "noqueue_down_event_profile_ms_per_token": noqueue_down_event_profile,
              "tokens": best_tokens,
          },
      },
      {
          "name": "queue_profile_row_overstates_noqueue_selected_wait",
          "pass": queue_profile_extra > floor_gap,
          "detail": {
              "queue_profile_extra_ms_per_token": queue_profile_extra,
              "queue_profile_selected_wait_ms_per_token": queue_profile_selected_wait,
              "noqueue_selected_wait_ms_per_token": noqueue_selected_wait,
          },
      },
      {
          "name": "source_semantics_explain_finish_vs_submit_ambiguity",
          "pass": all(source_semantics.values()),
          "detail": source_semantics,
      },
      {
          "name": "closed_down_tail_and_carrier_routes_recorded",
          "pass": not missing_rejected,
          "detail": {"missing": missing_rejected},
      },
      {
          "name": "frontier_still_below_floor_no_speed_claim",
          "pass": (
              frontier_state["current_best_tps"] < frontier_state["floor_tps"]
              and frontier_state["hard_stall_breached"] is True
              and frontier_state["review_recorded_for_current_best"] is True
          ),
          "detail": frontier_state,
      },
  ]
  required = all(check["pass"] for check in checks)
  selected_next_route = (
      "resident_selected_down_submit_split_source_gate"
      if required
      else "route_selection_needs_manual_review"
  )
  disposition = (
      "select_resident_selected_down_submit_split_source_gate"
      if required
      else "resident_selected_down_wait_drain_route_gate_failed"
  )
  next_route_reason = (
      "Seq158 proves the selected-down bucket is floor-sized, but queue "
      "profiling disables the accepted defer-finish path and adds a "
      f"{queue_profile_extra:.3f} ms/token finish/profile component. The "
      "noqueue frontier still has a floor-sized selected-down submit wall "
      f"({noqueue_selected_wait:.3f} ms/token), so the next unit is a "
      "default-off source split of selected-down wait into enqueue/finish/event "
      "sites before any speed row."
      if required
      else "Fix failed route checks before selecting another token row."
  )
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "required_checks_passed": required,
      "speedup_claims_allowed": False,
      "disposition": disposition,
      "selected_next_route": selected_next_route,
      "next_route_reason": next_route_reason,
      "inputs": {
          "frontier": _rel(args.frontier),
          "routes": _rel(args.routes),
          "rejected": _rel(args.rejected),
          "seq158": _rel(args.seq158),
          "best_smoke": _rel(args.best_smoke),
          "engine_source": _rel(args.engine_source),
          "engine_header": _rel(args.engine_header),
          "decode_source": _rel(args.decode_source),
      },
      "frontier": frontier_state,
      "selected_down_wait_ms_per_token": {
          "queue_profile": queue_profile_selected_wait,
          "noqueue_frontier": noqueue_selected_wait,
          "queue_profile_extra": queue_profile_extra,
          "noqueue_event_profile": noqueue_down_event_profile,
      },
      "source_semantics": source_semantics,
      "closed_route_requirements": {
          "required": sorted(required_rejected),
          "missing": missing_rejected,
      },
      "checks": checks,
  }


def write_outputs(out_dir: Path, payload: dict[str, Any]) -> None:
  out_dir.mkdir(parents=True, exist_ok=False)
  (out_dir / "metrics.json").write_text(
      json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  failed = [check["name"] for check in payload["checks"] if not check["pass"]]
  wait = payload["selected_down_wait_ms_per_token"]
  lines = [
      "# Resident Selected-Down Wait Drain Route Gate",
      "",
      f"- required checks passed: `{str(payload['required_checks_passed']).lower()}`",
      f"- disposition: `{payload['disposition']}`",
      f"- selected next route: `{payload['selected_next_route']}`",
      f"- queue-profile selected-down wait: `{wait['queue_profile']:.3f}` ms/token",
      f"- noqueue selected-down wait: `{wait['noqueue_frontier']:.3f}` ms/token",
      f"- queue-profile extra: `{wait['queue_profile_extra']:.3f}` ms/token",
      f"- failed checks: `{failed}`",
      "",
      payload["next_route_reason"],
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
  parser.add_argument("--seq158", type=Path, default=DEFAULT_SEQ158)
  parser.add_argument("--best-smoke", type=Path, default=DEFAULT_BEST_SMOKE)
  parser.add_argument("--engine-source", type=Path, default=DEFAULT_ENGINE)
  parser.add_argument("--engine-header", type=Path, default=DEFAULT_HEADER)
  parser.add_argument("--decode-source", type=Path, default=DEFAULT_DECODE)
  parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
  return parser.parse_args()


def main() -> int:
  args = parse_args()
  payload = compute(args)
  write_outputs(args.out_dir, payload)
  print(json.dumps(
      {
          "required_checks_passed": payload["required_checks_passed"],
          "disposition": payload["disposition"],
          "selected_next_route": payload["selected_next_route"],
          "out_dir": _rel(args.out_dir),
      },
      sort_keys=True))
  return 0 if payload["required_checks_passed"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
