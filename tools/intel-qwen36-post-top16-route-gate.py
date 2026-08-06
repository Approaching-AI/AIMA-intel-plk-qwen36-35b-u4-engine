#!/usr/bin/env python3
"""Select the next route after the top16 selected gate/up component proof.

This is route-control evidence only. It closes the just-tested top16-indexed
selected gate/up material layout as a floor-covering component proof and
selects the next source/component gate without launching another decode probe.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = "intel-qwen36-post-top16-route-gate-v0"
DEFAULT_FRONTIER = ACTIVE / "frontier.json"
DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_REJECTED = ACTIVE / "rejected-routes.json"
DEFAULT_TOP16 = (
    ROOT / "output/gpu-q4x8-selected-gateup-top16-indexed-probe-20260707Tseq88Z/probe-result.json"
)
DEFAULT_TOP16_CONFIRM = (
    ROOT
    / "output/gpu-q4x8-selected-gateup-top16-indexed-probe-confirm-20260707Tseq88Z/probe-result.json"
)
DEFAULT_OUT_DIR = ROOT / "output/post-top16-gateup-route-gate-20260707Tseq89Z"


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


def _frontier_summary(frontier: dict[str, Any]) -> dict[str, float]:
  budget = frontier.get("goal_budget")
  budget = budget if isinstance(budget, dict) else {}
  verdict = budget.get("verdict")
  verdict = verdict if isinstance(verdict, dict) else {}
  per_token = budget.get("per_token_ms")
  per_token = per_token if isinstance(per_token, dict) else {}
  wall = _num(per_token.get("wall"))
  floor_budget = _num(verdict.get("floor_budget_ms_per_token"))
  return {
      "floor_tps": _num(verdict.get("floor_tps")),
      "wall_ms_per_token": wall,
      "floor_budget_ms_per_token": floor_budget,
      "floor_gap_ms_per_token": max(0.0, wall - floor_budget),
  }


def _stage_gaps(frontier: dict[str, Any]) -> dict[str, float]:
  budget = frontier.get("goal_budget")
  budget = budget if isinstance(budget, dict) else {}
  out: dict[str, float] = {}
  for row in budget.get("stage_kernel_gap_estimates_ms_per_token", []):
    if isinstance(row, dict) and isinstance(row.get("stage"), str):
      out[row["stage"]] = _num(row.get("gap_ms_per_token"))
  return out


def _substage_gaps(frontier: dict[str, Any]) -> dict[tuple[str, str], float]:
  budget = frontier.get("goal_budget")
  budget = budget if isinstance(budget, dict) else {}
  out: dict[tuple[str, str], float] = {}
  for row in budget.get("substage_gap_estimates_ms_per_token", []):
    if (
        isinstance(row, dict)
        and isinstance(row.get("stage"), str)
        and isinstance(row.get("substage"), str)
    ):
      out[(row["stage"], row["substage"])] = _num(row.get("gap_ms_per_token"))
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


def _has_rejection(rejected: dict[str, Any], route: str) -> bool:
  return any(
      isinstance(row, dict) and row.get("route") == route
      for row in rejected.get("rejected", [])
  )


def compute(args: argparse.Namespace) -> dict[str, Any]:
  frontier = _load_json(args.frontier)
  routes = _load_json(args.routes)
  rejected = _load_json(args.rejected)
  top16 = _load_json(args.top16)
  top16_confirm = _load_json(args.top16_confirm)
  frontier_state = _frontier_summary(frontier)
  stage_gaps = _stage_gaps(frontier)
  substage_gaps = _substage_gaps(frontier)

  first_ratio = _num(_nested(
      top16, "timings", "top16_indexed_vs_no_concat_shell_speedup"))
  confirm_ratio = _num(_nested(
      top16_confirm, "timings", "top16_indexed_vs_no_concat_shell_speedup"))
  required_ratio = _num(_nested(
      top16_confirm, "timings", "top16_floor_covering_required_ratio"))
  if required_ratio == 0.0:
    required_ratio = _num(_nested(
        top16, "timings", "top16_floor_covering_required_ratio"))

  checks = [
      {
          "name": "seq87_selected_top16_contract_recorded",
          "pass": _has_candidate(
              routes,
              87,
              "selected_gateup_top16_indexed_material_component_contract_ready",
          ),
      },
      {
          "name": "top16_probe_exact_against_no_concat_and_oracle",
          "pass": bool(_nested(
              top16, "checks", "top16_indexed_gateup_swiglu_matches_references")),
      },
      {
          "name": "top16_confirm_exact_against_no_concat_and_oracle",
          "pass": bool(_nested(
              top16_confirm, "checks",
              "top16_indexed_gateup_swiglu_matches_references")),
      },
      {
          "name": "top16_probe_fails_floor_covering_ratio",
          "pass": first_ratio < required_ratio,
          "detail": {"observed": first_ratio, "required": required_ratio},
      },
      {
          "name": "top16_confirm_fails_floor_covering_ratio",
          "pass": confirm_ratio < required_ratio,
          "detail": {"observed": confirm_ratio, "required": required_ratio},
      },
      {
          "name": "q6_down_only_already_closed_as_next_gate",
          "pass": _has_rejection(
              rejected, "gpu_selected_shared_q6_down_occupancy4_component"),
      },
      {
          "name": "linear_preconv_gap_can_cover_floor",
          "pass": stage_gaps.get("linear_preconv", 0.0)
          > frontier_state["floor_gap_ms_per_token"],
          "detail": {
              "linear_preconv_gap_ms_per_token": stage_gaps.get(
                  "linear_preconv", 0.0),
              "floor_gap_ms_per_token": frontier_state[
                  "floor_gap_ms_per_token"],
          },
      },
      {
          "name": "qkv_conv_subgap_can_cover_floor",
          "pass": substage_gaps.get(("linear_preconv", "qkv_conv"), 0.0)
          > frontier_state["floor_gap_ms_per_token"],
          "detail": {
              "qkv_conv_gap_ms_per_token": substage_gaps.get(
                  ("linear_preconv", "qkv_conv"), 0.0),
              "floor_gap_ms_per_token": frontier_state[
                  "floor_gap_ms_per_token"],
          },
      },
      {
          "name": "shared_q8_preconv_decode_root_is_closed",
          "pass": _has_rejection(
              rejected, "gpu_linear_preconv_shared_q8_preconv_bundle_decode"),
      },
      {
          "name": "combined_attention_linear_alias_is_closed",
          "pass": _has_rejection(
              rejected, "gpu_attention_linear_event_lifetime_combined_alias"),
      },
  ]
  required_checks_passed = all(bool(row["pass"]) for row in checks)
  disposition = (
      "select_linear_preconv_qkv_conv_root_component_gate"
      if required_checks_passed
      else "post_top16_route_gate_failed"
  )
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "inputs": {
          "frontier": _rel(args.frontier),
          "routes": _rel(args.routes),
          "rejected": _rel(args.rejected),
          "top16": _rel(args.top16),
          "top16_confirm": _rel(args.top16_confirm),
      },
      "frontier": frontier_state,
      "stage_gap_ms_per_token": stage_gaps,
      "substage_gap_ms_per_token": {
          f"{stage}.{substage}": gap
          for (stage, substage), gap in substage_gaps.items()
      },
      "top16": {
          "first_shell_speedup": first_ratio,
          "confirm_shell_speedup": confirm_ratio,
          "required_shell_speedup": required_ratio,
          "first_required_checks_passed": bool(top16.get("required_checks_passed")),
          "confirm_required_checks_passed": bool(
              top16_confirm.get("required_checks_passed")),
          "correctness_exact": bool(_nested(
              top16_confirm, "checks",
              "top16_indexed_gateup_swiglu_matches_references")),
      },
      "checks": checks,
      "required_checks_passed": required_checks_passed,
      "disposition": disposition,
      "selected_next_route": (
          "linear_preconv_qkv_conv_regression_root_component_proof"
          if required_checks_passed
          else None
      ),
      "next_action": (
          "Run a source/component proof for the shared-Q8 linear-preconv "
          "qkv/conv regression root before any new decode probe. The proof "
          "must keep the shared-device-Q8 carrier benefit while showing qkv_conv "
          "non-growth or a floor-covering component delta; do not rerun "
          "selected gate/up top16, Q6 down-only, down-tail drain-shift, simple "
          "device-Q8 attention-front, shared-Q8 preconv decode, or the combined "
          "attention/linear alias."
          if required_checks_passed
          else "Fix the failed route-gate checks before selecting another route."
      ),
      "speedup_claims_allowed": False,
  }


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--frontier", type=Path, default=DEFAULT_FRONTIER)
  parser.add_argument("--routes", type=Path, default=DEFAULT_ROUTES)
  parser.add_argument("--rejected", type=Path, default=DEFAULT_REJECTED)
  parser.add_argument("--top16", type=Path, default=DEFAULT_TOP16)
  parser.add_argument("--top16-confirm", type=Path, default=DEFAULT_TOP16_CONFIRM)
  parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
  return parser.parse_args()


def main() -> int:
  args = parse_args()
  if args.out_dir.exists():
    raise SystemExit(f"output dir already exists: {args.out_dir}")
  args.out_dir.mkdir(parents=True)
  payload = compute(args)
  (args.out_dir / "metrics.json").write_text(
      json.dumps(payload, indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
  )
  (args.out_dir / "summary.md").write_text(
      "\n".join([
          "# Post Top16 Gateup Route Gate",
          "",
          f"- required checks passed: `{str(payload['required_checks_passed']).lower()}`",
          f"- disposition: `{payload['disposition']}`",
          f"- selected next route: `{payload['selected_next_route']}`",
          f"- next action: {payload['next_action']}",
          "",
      ]),
      encoding="utf-8",
  )
  print(args.out_dir)
  return 0 if payload["required_checks_passed"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
