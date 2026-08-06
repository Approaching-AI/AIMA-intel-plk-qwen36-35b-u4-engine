#!/usr/bin/env python3
"""Close or authorize the remaining linear-delta route.

This is route-control evidence only. It audits the current linear-delta
final-read/input-upload envelope after seq113 closed the linear-preconv
alpha/beta board. It does not launch a token-emitting decode row.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = "intel-qwen36-linear-delta-remaining-kernel-algorithm-gate-v0"
DEFAULT_FRONTIER = ACTIVE / "frontier.json"
DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_REJECTED = ACTIVE / "rejected-routes.json"
DEFAULT_ACCEPTED = ACTIVE / "accepted-cuts.json"
DEFAULT_SEQ51 = ROOT / "output/attn-linear-handoff-budget-20260706Tseq51Z/metrics.json"
DEFAULT_SEQ84 = (
    ROOT / "output/attn-linear-event-lifetime-decode-gate-20260707Tseq84Z/metrics.json"
)
DEFAULT_SEQ85 = ROOT / "output/attn-linear-regression-root-gate-20260707Tseq85Z/metrics.json"
DEFAULT_SEQ113 = (
    ROOT
    / "output/linear-preconv-alpha-beta-remaining-algorithm-gate-20260707Tseq113Z/metrics.json"
)
DEFAULT_DECODE = ROOT / "tools/intel-qwen36-r2-gpu-decode-smoke.py"
DEFAULT_ENGINE = ROOT / "engine/src/gpu_q4x8_matvec.cpp"
DEFAULT_HEADER = ROOT / "engine/include/intel_qwen36/gpu_q4x8_matvec.hpp"
DEFAULT_OUT_DIR = ROOT / "output/linear-delta-remaining-kernel-algorithm-gate-20260707Tseq114Z"


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


def _substage_rows(frontier: dict[str, Any]) -> dict[str, dict[str, Any]]:
  budget = frontier.get("goal_budget")
  budget = budget if isinstance(budget, dict) else {}
  out: dict[str, dict[str, Any]] = {}
  for row in budget.get("substage_gap_estimates_ms_per_token", []):
    if (
        isinstance(row, dict)
        and isinstance(row.get("stage"), str)
        and isinstance(row.get("substage"), str)
    ):
      out[f"{row['stage']}.{row['substage']}"] = row
  return out


def _route_names(rejected: dict[str, Any]) -> set[str]:
  names: set[str] = set()
  for row in rejected.get("rejected", []):
    if isinstance(row, dict) and isinstance(row.get("route"), str):
      names.add(row["route"])
  return names


def _accepted_ids(accepted: dict[str, Any]) -> set[str]:
  ids: set[str] = set()
  for row in accepted.get("accepted", []):
    if isinstance(row, dict) and isinstance(row.get("id"), str):
      ids.add(row["id"])
  return ids


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


def _has_parked(routes: dict[str, Any], route_id: str, rank: int) -> bool:
  for row in routes.get("parked_routes", []):
    if (
        isinstance(row, dict)
        and row.get("id") == route_id
        and int(_num(row.get("rank"))) == rank
    ):
      return True
  return False


def _has_markers(text: str, markers: list[str]) -> dict[str, Any]:
  missing = [marker for marker in markers if marker not in text]
  return {"pass": not missing, "missing": missing, "marker_count": len(markers)}


def _absent_markers(text: str, markers: list[str]) -> dict[str, Any]:
  present = [marker for marker in markers if marker in text]
  return {"pass": not present, "present": present, "marker_count": len(markers)}


def _largest(mapping: dict[str, float]) -> tuple[str, float]:
  if not mapping:
    return "", 0.0
  key = max(mapping, key=lambda item: mapping[item])
  return key, mapping[key]


def compute(args: argparse.Namespace) -> dict[str, Any]:
  frontier = _load_json(args.frontier)
  routes = _load_json(args.routes)
  rejected = _load_json(args.rejected)
  accepted = _load_json(args.accepted)
  seq51 = _load_json(args.seq51)
  seq84 = _load_json(args.seq84)
  seq85 = _load_json(args.seq85)
  seq113 = _load_json(args.seq113)
  source = "\n".join([
      _read(args.decode_source),
      _read(args.engine_source),
      _read(args.engine_header),
  ])

  frontier_state = _frontier_summary(frontier)
  floor_gap = frontier_state["floor_gap_ms_per_token"]
  stage_gaps = _stage_gaps(frontier)
  stage_walls = _stage_walls(frontier)
  largest_stage, largest_stage_gap = _largest(stage_gaps)
  substage = _substage_rows(frontier)
  rejected_names = _route_names(rejected)
  accepted = _accepted_ids(accepted)

  linear_delta = stage_gaps.get("linear_delta", 0.0)
  final_read = substage.get("linear_delta.final_read", {})
  input_upload = substage.get("linear_delta.input_upload", {})
  final_read_gap = _num(final_read.get("gap_ms_per_token"))
  input_upload_gap = _num(input_upload.get("gap_ms_per_token"))
  boundary_gap = final_read_gap + input_upload_gap

  required_accepted = [
      "resident_linear_state_pingpong",
      "linear_delta_qk_local_final_fused",
      "linear_delta_attention_readback_skip",
      "linear_delta_scratch_reuse_nonblocking",
      "linear_postconv_delta_resident_handoff",
      "linear_delta_cpu_shape_final_qk_local_fused",
  ]
  missing_accepted = [item for item in required_accepted if item not in accepted]

  required_closed_routes = [
      "gpu_linear_delta_read_drain_noqueue",
      "gpu_linear_final_device_q8_handoff_noqueue",
      "gpu_linear_final_device_q8_handoff_scratch_noqueue",
      "gpu_attention_linear_simple_final_output_handoffs",
      "gpu_attention_linear_event_lifetime_combined_alias",
      "current_linear_preconv_alpha_beta_algorithm_board",
  ]
  missing_closed_routes = [
      route for route in required_closed_routes if route not in rejected_names
  ]

  resident_delta_source = _has_markers(
      source,
      [
          "RunPostConvPrepThenLinearAttentionDeltaResidentState",
          "RunLinearAttentionDeltaResidentState",
          "RegisterF32BufferAlias",
          "linear_delta_final_alias_handle_",
          "IQ36_LINEAR_FINAL_DEVICE_Q8_HANDOFF",
      ],
  )
  upload_source = _has_markers(
      source,
      [
          "make_read(linear_delta_scratch_gate_",
          "make_read(linear_delta_scratch_beta_",
          "make_read(linear_delta_scratch_z_",
          "make_read(linear_delta_scratch_norm_",
      ],
  )
  new_device_input_absent = _absent_markers(
      source,
      [
          "IQ36_LINEAR_DELTA_DEVICE_INPUTS",
          "RunLinearAttentionDeltaResidentInputs",
          "linear_delta_gate_beta_z_device",
          "linear_delta_resident_gate_beta_z",
      ],
  )

  seq51_derived = seq51.get("derived") if isinstance(seq51.get("derived"), dict) else {}
  seq84_derived = seq84.get("derived") if isinstance(seq84.get("derived"), dict) else {}
  seq85_derived = seq85.get("derived") if isinstance(seq85.get("derived"), dict) else {}

  checks = [
      {
          "name": "seq113_selected_this_gate",
          "pass": (
              seq113.get("required_checks_passed") is True
              and seq113.get("selected_next_route")
              == "linear_delta_remaining_kernel_algorithm_gate"
              and _has_candidate(
                  routes,
                  113,
                  "close_current_linear_preconv_alpha_beta_algorithm_board",
              )
              and _has_switch(
                  routes,
                  "close_current_linear_preconv_alpha_beta_board_switch_to_linear_delta_gate",
                  113,
              )
          ),
      },
      {
          "name": "frontier_still_below_floor",
          "pass": frontier_state["wall_ms_per_token"]
          > frontier_state["floor_budget_ms_per_token"]
          > 0.0,
      },
      {
          "name": "linear_delta_gap_is_floor_sized",
          "pass": linear_delta > floor_gap,
          "detail": {
              "linear_delta_gap_ms_per_token": linear_delta,
              "floor_gap_ms_per_token": floor_gap,
          },
      },
      {
          "name": "final_read_plus_input_upload_can_cover_floor_only_as_bundle",
          "pass": boundary_gap > floor_gap and final_read_gap < floor_gap and input_upload_gap < floor_gap,
          "detail": {
              "final_read_gap_ms_per_token": final_read_gap,
              "input_upload_gap_ms_per_token": input_upload_gap,
              "combined_gap_ms_per_token": boundary_gap,
              "floor_gap_ms_per_token": floor_gap,
          },
      },
      {
          "name": "accepted_linear_delta_stack_already_contains_current_cuts",
          "pass": not missing_accepted,
          "detail": {"missing_accepted_cuts": missing_accepted},
      },
      {
          "name": "closed_routes_cover_current_boundary_variants",
          "pass": not missing_closed_routes,
          "detail": {"missing_closed_routes": missing_closed_routes},
      },
      {
          "name": "simple_final_output_handoff_is_closed_by_growth",
          "pass": (
              _num(seq51_derived.get("final_read_saved_ms_per_token")) > 0.0
              and _num(seq51_derived.get("attention_front_growth_ms_per_token")) > floor_gap
              and _num(seq51_derived.get("wall_growth_ms_per_token")) > floor_gap
              and seq51.get("verdict", {}).get("simple_final_output_handoff_closed")
              is True
          ),
          "detail": {
              "final_read_saved_ms_per_token": _num(
                  seq51_derived.get("final_read_saved_ms_per_token")),
              "attention_front_growth_ms_per_token": _num(
                  seq51_derived.get("attention_front_growth_ms_per_token")),
              "wall_growth_ms_per_token": _num(
                  seq51_derived.get("wall_growth_ms_per_token")),
          },
      },
      {
          "name": "combined_attention_linear_alias_closes_bundle_shape",
          "pass": (
              seq84.get("required_checks_passed") is True
              and _num(seq84_derived.get("linear_delta_final_read_delta_ms_per_token")) < 0.0
              and _num(seq84_derived.get("wall_delta_ms_per_token")) > floor_gap
              and seq84_derived.get("attention_front_non_growth") is False
          ),
          "detail": {
              "linear_delta_final_read_delta_ms_per_token": _num(
                  seq84_derived.get("linear_delta_final_read_delta_ms_per_token")),
              "linear_preconv_qkv_conv_delta_ms_per_token": _num(
                  seq84_derived.get("linear_preconv_qkv_conv_delta_ms_per_token")),
              "attention_front_delta_ms_per_token": _num(
                  seq84_derived.get("attention_front_delta_ms_per_token")),
              "wall_delta_ms_per_token": _num(seq84_derived.get("wall_delta_ms_per_token")),
          },
      },
      {
          "name": "regression_roots_match_closed_classes",
          "pass": (
              seq85.get("required_checks_passed") is True
              and _num(seq85_derived.get("measured_regressions_ms_per_token"))
              > _num(seq85_derived.get("measured_savings_ms_per_token"))
              and _num(seq85_derived.get("linear_delta_final_read_saved_ms_per_token"))
              > 0.0
          ),
          "detail": {
              "linear_delta_final_read_saved_ms_per_token": _num(
                  seq85_derived.get("linear_delta_final_read_saved_ms_per_token")),
              "measured_savings_ms_per_token": _num(
                  seq85_derived.get("measured_savings_ms_per_token")),
              "measured_regressions_ms_per_token": _num(
                  seq85_derived.get("measured_regressions_ms_per_token")),
          },
      },
      {
          "name": "source_has_current_resident_delta_but_no_new_device_input_algorithm",
          "pass": (
              resident_delta_source["pass"]
              and upload_source["pass"]
              and new_device_input_absent["pass"]
          ),
          "detail": {
              "resident_delta_source": resident_delta_source,
              "upload_source": upload_source,
              "new_device_input_absent": new_device_input_absent,
          },
      },
      {
          "name": "parked_offline_repack_is_next_broad_route",
          "pass": (
              largest_stage == "selected_ffn"
              and largest_stage_gap > floor_gap
              and _has_parked(routes, "offline_repack_streaming_layout", 1)
          ),
          "detail": {
              "largest_stage": largest_stage,
              "largest_stage_gap_ms_per_token": largest_stage_gap,
              "floor_gap_ms_per_token": floor_gap,
              "rank1_parked_route_present": _has_parked(
                  routes, "offline_repack_streaming_layout", 1),
          },
      },
  ]

  required = all(check["pass"] for check in checks)
  selected_next_route = (
      "offline_repack_streaming_layout_route_gate"
      if required else "manual_review_linear_delta_remaining_gate"
  )
  disposition = (
      "close_current_linear_delta_algorithm_board"
      if required else "linear_delta_remaining_kernel_algorithm_gate_failed"
  )
  next_action = (
      "Close the current linear-delta algorithm board. Final-read plus "
      "input-upload can cover the floor only as a bundle, but simple final "
      "output handoff and the combined attention/linear alias are already "
      "closed by attention-front/qkv regressions, and the source has no new "
      "device-input carrier for gate/beta/z/norm. Pop the parked rank-1 "
      "offline_repack_streaming_layout route as a design/component gate; do not "
      "launch another current-board linear-delta decode row."
      if required else "Review failed gate checks before selecting another probe."
  )

  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "required_checks_passed": required,
      "speedup_claims_allowed": False,
      "decode_probe_allowed": False,
      "component_probe_allowed": False,
      "disposition": disposition,
      "selected_next_route": selected_next_route,
      "next_action": next_action,
      "inputs": {
          "frontier": _rel(args.frontier),
          "routes": _rel(args.routes),
          "rejected": _rel(args.rejected),
          "accepted": _rel(args.accepted),
          "seq51": _rel(args.seq51),
          "seq84": _rel(args.seq84),
          "seq85": _rel(args.seq85),
          "seq113": _rel(args.seq113),
          "decode_source": _rel(args.decode_source),
          "engine_source": _rel(args.engine_source),
          "engine_header": _rel(args.engine_header),
      },
      "frontier": frontier_state,
      "stage_gap_ms_per_token": stage_gaps,
      "stage_wall_ms_per_token": stage_walls,
      "linear_delta_budget": {
          "stage_gap_ms_per_token": linear_delta,
          "final_read": final_read,
          "input_upload": input_upload,
          "final_read_plus_input_upload_gap_ms_per_token": boundary_gap,
      },
      "accepted_cut_requirements": {
          "required": required_accepted,
          "missing": missing_accepted,
      },
      "closed_route_requirements": {
          "required": required_closed_routes,
          "missing": missing_closed_routes,
      },
      "prior_closure_summary": {
          "seq51_final_read_saved_ms_per_token": _num(
              seq51_derived.get("final_read_saved_ms_per_token")),
          "seq51_attention_front_growth_ms_per_token": _num(
              seq51_derived.get("attention_front_growth_ms_per_token")),
          "seq84_linear_delta_final_read_delta_ms_per_token": _num(
              seq84_derived.get("linear_delta_final_read_delta_ms_per_token")),
          "seq84_attention_front_delta_ms_per_token": _num(
              seq84_derived.get("attention_front_delta_ms_per_token")),
          "seq84_qkv_conv_delta_ms_per_token": _num(
              seq84_derived.get("linear_preconv_qkv_conv_delta_ms_per_token")),
          "seq85_measured_savings_ms_per_token": _num(
              seq85_derived.get("measured_savings_ms_per_token")),
          "seq85_measured_regressions_ms_per_token": _num(
              seq85_derived.get("measured_regressions_ms_per_token")),
      },
      "source_shape": {
          "resident_delta_source": resident_delta_source,
          "upload_source": upload_source,
          "new_device_input_absent": new_device_input_absent,
      },
      "next_route": {
          "id": "offline_repack_streaming_layout",
          "reason": "rank-1 parked broad route after current stage boards closed",
          "largest_stage": largest_stage,
          "largest_stage_gap_ms_per_token": largest_stage_gap,
      },
      "checks": checks,
  }


def write_outputs(out_dir: Path, payload: dict[str, Any]) -> None:
  out_dir.mkdir(parents=True, exist_ok=False)
  (out_dir / "metrics.json").write_text(
      json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  failed = [check["name"] for check in payload["checks"] if check["pass"] is not True]
  frontier = payload["frontier"]
  budget = payload["linear_delta_budget"]
  closure = payload["prior_closure_summary"]
  next_route = payload["next_route"]
  lines = [
      "# Linear-Delta Remaining Kernel Algorithm Gate",
      "",
      f"- required checks passed: `{str(payload['required_checks_passed']).lower()}`",
      f"- disposition: `{payload['disposition']}`",
      f"- component probe allowed: `{str(payload['component_probe_allowed']).lower()}`",
      f"- decode probe allowed: `{str(payload['decode_probe_allowed']).lower()}`",
      f"- selected next route: `{payload['selected_next_route']}`",
      f"- frontier gap: `{frontier['floor_gap_ms_per_token']:.3f}` ms/token",
      f"- linear-delta stage gap: `{budget['stage_gap_ms_per_token']:.3f}` ms/token",
      f"- final-read + input-upload gap: "
      f"`{budget['final_read_plus_input_upload_gap_ms_per_token']:.3f}` ms/token",
      f"- seq51 final-read saved / attention-front growth: "
      f"`{closure['seq51_final_read_saved_ms_per_token']:.3f}` / "
      f"`{closure['seq51_attention_front_growth_ms_per_token']:.3f}` ms/token",
      f"- seq84 final-read delta / qkv growth / attention growth: "
      f"`{closure['seq84_linear_delta_final_read_delta_ms_per_token']:.3f}` / "
      f"`{closure['seq84_qkv_conv_delta_ms_per_token']:.3f}` / "
      f"`{closure['seq84_attention_front_delta_ms_per_token']:.3f}` ms/token",
      f"- next broad route: `{next_route['id']}` "
      f"({next_route['largest_stage']} gap "
      f"`{next_route['largest_stage_gap_ms_per_token']:.3f}` ms/token)",
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
  parser.add_argument("--accepted", type=Path, default=DEFAULT_ACCEPTED)
  parser.add_argument("--seq51", type=Path, default=DEFAULT_SEQ51)
  parser.add_argument("--seq84", type=Path, default=DEFAULT_SEQ84)
  parser.add_argument("--seq85", type=Path, default=DEFAULT_SEQ85)
  parser.add_argument("--seq113", type=Path, default=DEFAULT_SEQ113)
  parser.add_argument("--decode-source", type=Path, default=DEFAULT_DECODE)
  parser.add_argument("--engine-source", type=Path, default=DEFAULT_ENGINE)
  parser.add_argument("--engine-header", type=Path, default=DEFAULT_HEADER)
  parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
  return parser.parse_args()


def main() -> int:
  args = parse_args()
  payload = compute(args)
  write_outputs(args.out_dir, payload)
  print(json.dumps(payload, indent=2, sort_keys=True))
  return 0 if payload["required_checks_passed"] else 2


if __name__ == "__main__":
  raise SystemExit(main())
