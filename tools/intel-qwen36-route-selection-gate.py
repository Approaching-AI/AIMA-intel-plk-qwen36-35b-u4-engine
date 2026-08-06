#!/usr/bin/env python3
"""Select the next decode route from the current budget and closed-route boards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA_VERSION = "intel-qwen36-route-selection-gate-v0"
DEFAULT_FRONTIER = ROOT / "doc/active" / WORKSTREAM / "frontier.json"
DEFAULT_ROUTES = ROOT / "doc/active" / WORKSTREAM / "routes-ledger.json"
DEFAULT_REJECTED = ROOT / "doc/active" / WORKSTREAM / "rejected-routes.json"
DEFAULT_ATTN_LINEAR = ROOT / "output/attn-linear-budget-20260707Tseq82Z/budget.json"
DEFAULT_OUT_DIR = ROOT / "output/route-selection-gate-20260707Tseq82Z"


def _load_json(path: Path) -> Any:
  return json.loads(path.read_text(encoding="utf-8"))


def _num(value: Any) -> float:
  return float(value) if isinstance(value, (int, float)) else 0.0


def _goal_budget(frontier: dict[str, Any]) -> dict[str, Any]:
  budget = frontier.get("goal_budget")
  return budget if isinstance(budget, dict) else {}


def _stage_gap(budget: dict[str, Any], stage: str) -> float:
  rows = budget.get("stage_kernel_gap_estimates_ms_per_token")
  if not isinstance(rows, list):
    return 0.0
  for row in rows:
    if isinstance(row, dict) and row.get("stage") == stage:
      return _num(row.get("gap_ms_per_token"))
  return 0.0


def _closed_routes(rejected: dict[str, Any], needles: tuple[str, ...]) -> list[str]:
  rows = rejected.get("rejected")
  if not isinstance(rows, list):
    return []
  out: list[str] = []
  lower_needles = tuple(item.lower() for item in needles)
  for row in rows:
    if not isinstance(row, dict):
      continue
    haystack = " ".join(
        str(row.get(key, ""))
        for key in ("route", "class", "reason", "runtime_cleanup")
    ).lower()
    if any(needle in haystack for needle in lower_needles):
      route = row.get("route")
      if isinstance(route, str):
        out.append(route)
  return out


def _has_switch(routes: dict[str, Any], family: str, seq_covered: int) -> bool:
  rows = routes.get("switch_decisions")
  if not isinstance(rows, list):
    return False
  for row in rows:
    if (
        isinstance(row, dict)
        and row.get("family") == family
        and _num(row.get("seq_covered")) >= seq_covered
        and row.get("resolved") is True
    ):
      return True
  return False


def _rel(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path)


def compute(args: argparse.Namespace) -> dict[str, Any]:
  frontier = _load_json(args.frontier)
  routes = _load_json(args.routes)
  rejected = _load_json(args.rejected)
  attn_linear = _load_json(args.attn_linear_budget)

  budget = _goal_budget(frontier)
  per_token = budget.get("per_token_ms") if isinstance(budget, dict) else {}
  verdict = budget.get("verdict") if isinstance(budget, dict) else {}
  wall_ms = _num(per_token.get("wall") if isinstance(per_token, dict) else None)
  floor_ms = _num(
      verdict.get("floor_budget_ms_per_token") if isinstance(verdict, dict) else None
  )
  floor_gap_ms = max(0.0, wall_ms - floor_ms)
  selected_gap = _stage_gap(budget, "selected_ffn")
  attention_gap = _stage_gap(budget, "attention_front")
  full_core_gap = _stage_gap(budget, "full_core")
  linear_gap = _stage_gap(budget, "linear_preconv")
  attn_linear_gap = attention_gap + full_core_gap + linear_gap

  no_progress = frontier.get("no_progress")
  no_progress = no_progress if isinstance(no_progress, dict) else {}
  closed_q6_down_tail = _closed_routes(
      rejected, ("q6 down-tail", "down_tail", "down-to-tail")
  )
  closed_selected_gateup = _closed_routes(
      rejected, ("selected gate", "gate/up", "gateup", "dpas local-q8")
  )
  closed_attn_linear = _closed_routes(
      rejected, ("attention", "linear", "preconv", "full_core")
  )

  attn_budget_gaps = attn_linear.get("same_source_gap_upper_bound_ms_per_token")
  attn_budget_gaps = attn_budget_gaps if isinstance(attn_budget_gaps, dict) else {}
  attn_budget_sum = _num(attn_budget_gaps.get("attention_fullcore_linear_sum"))

  selected_ffn_requires_new_proof = bool(closed_q6_down_tail) and bool(
      closed_selected_gateup
  )
  attention_linear_clears_floor = attn_linear_gap >= floor_gap_ms
  attention_linear_budget_clears_floor = attn_budget_sum >= floor_gap_ms
  queue_bundle_switch_recorded = _has_switch(
      routes, "queue_lifetime_down_tail_bundle", 81
  )

  checks = [
      {
          "name": "frontier_still_below_floor",
          "pass": wall_ms > floor_ms > 0.0,
      },
      {
          "name": "q6_down_tail_switch_recorded",
          "pass": queue_bundle_switch_recorded,
      },
      {
          "name": "selected_ffn_requires_material_new_proof",
          "pass": selected_ffn_requires_new_proof,
      },
      {
          "name": "attention_linear_gap_can_clear_floor",
          "pass": attention_linear_clears_floor and attention_linear_budget_clears_floor,
      },
      {
          "name": "attention_linear_closed_routes_acknowledged",
          "pass": len(closed_attn_linear) > 0,
      },
  ]

  selected_next_route = (
      "attention_linear_event_lifetime_proof"
      if all(item["pass"] for item in checks)
      else "route_selection_needs_manual_review"
  )
  next_action = (
      "Open a broad attention-front/full-core/linear-preconv event-lifetime proof "
      "that can remove at least the 0.45 ms/token floor gap without growing "
      "attention-front wall. Do not rerun simple linear-final device-Q8 handoff, "
      "shared-Q8 preconv, Q6 down-tail, or selected gate/up local-size/DPAS "
      "variants without a new component proof."
      if selected_next_route == "attention_linear_event_lifetime_proof"
      else "Review route boards before launching another target probe."
  )

  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "speedup_claims_allowed": False,
      "required_checks_passed": all(item["pass"] for item in checks),
      "checks": checks,
      "inputs": {
          "frontier": _rel(args.frontier),
          "routes": _rel(args.routes),
          "rejected": _rel(args.rejected),
          "attn_linear_budget": _rel(args.attn_linear_budget),
      },
      "frontier": {
          "best_tps": _num(
              (no_progress.get("last_significant_improvement") or {}).get("tps")
              if isinstance(no_progress.get("last_significant_improvement"), dict)
              else None
          ),
          "floor_tps": _num(verdict.get("floor_tps") if isinstance(verdict, dict) else None),
          "wall_ms_per_token": wall_ms,
          "floor_budget_ms_per_token": floor_ms,
          "floor_gap_ms_per_token": floor_gap_ms,
          "noise_rel": _num(
              (no_progress.get("noise") or {}).get("rel")
              if isinstance(no_progress.get("noise"), dict)
              else None
          ),
          "runs_since_significant_improvement": no_progress.get(
              "runs_since_significant_improvement"
          ),
          "soft_reflection_breached": no_progress.get("soft_reflection_breached"),
          "hard_stall_breached": no_progress.get("hard_stall_breached"),
      },
      "stage_gap_ms_per_token": {
          "selected_ffn": selected_gap,
          "attention_front": attention_gap,
          "full_core": full_core_gap,
          "linear_preconv": linear_gap,
          "attention_fullcore_linear_sum": attn_linear_gap,
      },
      "closed_route_counts": {
          "q6_down_tail": len(closed_q6_down_tail),
          "selected_gateup_or_dpas": len(closed_selected_gateup),
          "attention_or_linear": len(closed_attn_linear),
      },
      "selected_next_route": selected_next_route,
      "next_action": next_action,
      "disposition": "route_switch_to_attention_linear_event_lifetime_gate",
  }


def write_outputs(out_dir: Path, payload: dict[str, Any]) -> None:
  out_dir.mkdir(parents=True, exist_ok=False)
  (out_dir / "metrics.json").write_text(
      json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  failed = [item["name"] for item in payload["checks"] if not item["pass"]]
  frontier = payload["frontier"]
  gaps = payload["stage_gap_ms_per_token"]
  lines = [
      "# Route Selection Gate",
      "",
      f"- required checks passed: `{str(payload['required_checks_passed']).lower()}`",
      f"- selected next route: `{payload['selected_next_route']}`",
      f"- disposition: `{payload['disposition']}`",
      f"- frontier gap: `{frontier['floor_gap_ms_per_token']:.3f}` ms/token",
      f"- selected-FFN gap: `{gaps['selected_ffn']:.3f}` ms/token",
      f"- attention/full-core/linear gap sum: `{gaps['attention_fullcore_linear_sum']:.3f}` ms/token",
      f"- failed checks: `{failed}`",
      "",
      payload["next_action"],
      "",
      "This is route-selection evidence only. It does not claim speed.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--frontier", type=Path, default=DEFAULT_FRONTIER)
  parser.add_argument("--routes", type=Path, default=DEFAULT_ROUTES)
  parser.add_argument("--rejected", type=Path, default=DEFAULT_REJECTED)
  parser.add_argument("--attn-linear-budget", type=Path, default=DEFAULT_ATTN_LINEAR)
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
