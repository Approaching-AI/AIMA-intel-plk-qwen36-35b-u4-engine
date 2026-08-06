#!/usr/bin/env python3
"""Close or authorize the remaining linear-preconv alpha/beta route.

This is route-control evidence only. It audits the alpha/beta wall gap after
shared-Q8 qkv/conv and fused postconv-prep have both closed. It does not launch
a token-emitting decode row.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = "intel-qwen36-linear-preconv-alpha-beta-remaining-algorithm-gate-v0"
DEFAULT_FRONTIER = ACTIVE / "frontier.json"
DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_REJECTED = ACTIVE / "rejected-routes.json"
DEFAULT_PROFILE_LOG = ROOT / "output/explore-log.jsonl"
DEFAULT_SEQ109 = (
    ROOT / "output/selected-ffn-kernel-layout-component-gate-20260707Tseq109Z/metrics.json"
)
DEFAULT_SEQ110 = (
    ROOT / "output/full-core-attention-front-kernel-algorithm-gate-20260707Tseq110Z/metrics.json"
)
DEFAULT_SEQ111 = (
    ROOT / "output/linear-preconv-remaining-kernel-algorithm-gate-20260707Tseq111Z/metrics.json"
)
DEFAULT_SEQ112 = (
    ROOT / "output/linear-preconv-fused-postconv-prep-gate-20260707Tseq112Z/metrics.json"
)
DEFAULT_SEQ77 = (
    ROOT / "output/linear-preconv-shared-q8-profile-gate-20260706Tseq77Z/metrics.json"
)
DEFAULT_SEQ90_Q4 = (
    ROOT / "output/linear-preconv-qkv-conv-root-probe-20260707Tseq90Z/metrics.json"
)
DEFAULT_SEQ90_Q6 = (
    ROOT / "output/linear-preconv-qkv-conv-root-probe-20260707Tseq90-layer0Z/metrics.json"
)
DEFAULT_DECODE = ROOT / "tools/intel-qwen36-r2-gpu-decode-smoke.py"
DEFAULT_ENGINE = ROOT / "engine/src/gpu_q4x8_matvec.cpp"
DEFAULT_HEADER = ROOT / "engine/include/intel_qwen36/gpu_q4x8_matvec.hpp"
DEFAULT_OPENCL = ROOT / "engine/gpu/opencl/q4x8_matvec.cl"
DEFAULT_OUT_DIR = (
    ROOT / "output/linear-preconv-alpha-beta-remaining-algorithm-gate-20260707Tseq113Z"
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


def _frontier_summary(frontier: dict[str, Any]) -> dict[str, Any]:
  no_progress = frontier.get("no_progress")
  no_progress = no_progress if isinstance(no_progress, dict) else {}
  budget = frontier.get("goal_budget")
  budget = budget if isinstance(budget, dict) else {}
  per_token = budget.get("per_token_ms")
  per_token = per_token if isinstance(per_token, dict) else {}
  verdict = budget.get("verdict")
  verdict = verdict if isinstance(verdict, dict) else {}
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


def _has_candidate(routes: dict[str, Any], seq: int, disposition: str) -> bool:
  for row in routes.get("candidate_history", []):
    if (
        isinstance(row, dict)
        and row.get("seq") == seq
        and row.get("disposition") == disposition
    ):
      return True
  return False


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  for line in path.read_text(encoding="utf-8").splitlines():
    if not line.strip():
      continue
    row = json.loads(line)
    if isinstance(row, dict):
      rows.append(row)
  return rows


def _label_row(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
  matches = [row for row in rows if row.get("label") == label]
  return matches[-1] if matches else {}


def _profile(row: dict[str, Any]) -> dict[str, Any]:
  profile = row.get("profile_smoke")
  return profile if isinstance(profile, dict) else {}


def _wall_ms(profile: dict[str, Any], key: str, tokens: float = 8.0) -> float:
  wall = profile.get("linear_preconv_wall_profile_ns")
  wall = wall if isinstance(wall, dict) else {}
  return _num(wall.get(key)) / tokens / 1e6


def _kernel_ms(profile: dict[str, Any], *keys: str, tokens: float = 8.0) -> float:
  kernel = profile.get("linear_preconv_kernel_profile_us")
  kernel = kernel if isinstance(kernel, dict) else {}
  return sum(_num(kernel.get(key)) for key in keys) / tokens / 1000.0


def _has_markers(text: str, markers: list[str]) -> dict[str, Any]:
  missing = [marker for marker in markers if marker not in text]
  return {"pass": not missing, "missing": missing, "marker_count": len(markers)}


def _absent_markers(text: str, markers: list[str]) -> dict[str, Any]:
  present = [marker for marker in markers if marker in text]
  return {"pass": not present, "present": present, "marker_count": len(markers)}


def compute(args: argparse.Namespace) -> dict[str, Any]:
  frontier = _load_json(args.frontier)
  routes = _load_json(args.routes)
  rejected = _load_json(args.rejected)
  seq109 = _load_json(args.seq109)
  seq110 = _load_json(args.seq110)
  seq111 = _load_json(args.seq111)
  seq112 = _load_json(args.seq112)
  seq77 = _load_json(args.seq77)
  seq90_q4 = _load_json(args.seq90_q4)
  seq90_q6 = _load_json(args.seq90_q6)
  explore_rows = _read_jsonl(args.profile_log)
  source = "\n".join([
      _read(args.decode_source),
      _read(args.engine_source),
      _read(args.engine_header),
      _read(args.opencl_source),
  ])

  frontier_state = _frontier_summary(frontier)
  stage_gaps = _stage_gaps(frontier)
  stage_walls = _stage_walls(frontier)
  substage = _substage_rows(frontier)
  floor_gap = frontier_state["floor_gap_ms_per_token"]
  rejected_names = _route_names(rejected)

  baseline_row = _label_row(
      explore_rows, "selected-shared-q4q6-down-cold-q6-experts-profile")
  baseline_profile = _profile(baseline_row)

  alpha_beta = substage.get("linear_preconv.alpha_beta", {})
  qkv_conv = substage.get("linear_preconv.qkv_conv", {})
  postconv = substage.get("linear_preconv.postconv_prep", {})
  linear_delta_final = substage.get("linear_delta.final_read", {})
  linear_delta_input = substage.get("linear_delta.input_upload", {})

  alpha_beta_gap = _num(alpha_beta.get("gap_ms_per_token"))
  qkv_conv_gap = _num(qkv_conv.get("gap_ms_per_token"))
  postconv_gap = _num(postconv.get("gap_ms_per_token"))
  linear_delta_stage_gap = stage_gaps.get("linear_delta", 0.0)
  linear_delta_boundary_gap = (
      _num(linear_delta_final.get("gap_ms_per_token"))
      + _num(linear_delta_input.get("gap_ms_per_token"))
  )

  live_alpha_beta_wall = _wall_ms(baseline_profile, "alpha_beta")
  live_alpha_beta_kernel = _kernel_ms(baseline_profile, "alpha", "beta")
  live_qkv_conv_wall = _wall_ms(baseline_profile, "qkv_conv")
  live_qkv_conv_kernel = _kernel_ms(baseline_profile, "qkv", "conv")
  live_postconv_wall = _wall_ms(baseline_profile, "postconv_prep")
  live_postconv_kernel = _kernel_ms(
      baseline_profile, "postconv_silu_split", "postconv_q_l2", "postconv_k_l2")

  seq77_derived = seq77.get("derived") if isinstance(seq77.get("derived"), dict) else {}
  seq90_q4_derived = (
      seq90_q4.get("derived") if isinstance(seq90_q4.get("derived"), dict) else {})
  seq90_q6_derived = (
      seq90_q6.get("derived") if isinstance(seq90_q6.get("derived"), dict) else {})

  known_alpha_beta_shapes = _has_markers(
      source,
      [
          "g_decode_linear_abz_cpu_order_fused",
          "LinearAlphaBetaZHandle",
          "g_decode_resident_q4_cpu_order_linear_ab",
          "LinearAlphaBetaHandle",
          "RunResidentPackedQ4X8",
          "RunResidentRawQ4KCpuOrder",
          "RunF32InputHandleSharedDeviceQ8ThenResident",
      ],
  )
  new_alpha_beta_algorithm_absent = _absent_markers(
      source,
      [
          "IQ36_LINEAR_PRECONV_ALPHA_BETA_ONLY",
          "RunLinearPreconvAlphaBetaOnly",
          "linear_preconv_alpha_beta_only",
          "linear_attn_alpha_beta_fused",
          "preconv_alpha_beta_fused",
      ],
  )

  required_closed_routes = [
      "gpu_linear_ab_cpu_order_speed_lane",
      "gpu_attention_front_handoff_8tok_q4_cpu_order_linear_ab_diagnostic",
      "gpu_q4_cpu_order_auxless_nibble_loop",
      "gpu_q4_cpu_order_localq8_staging",
      "gpu_linear_qkv_alpha_beta_prefix_conv",
      "gpu_preconv_shared_q8_input_reuse",
      "gpu_linear_preconv_qkv_only_resident_input_wiring",
      "gpu_linear_preconv_shared_q8_preconv_bundle_decode",
      "gpu_linear_preconv_shared_q8_qkv_conv_root_component",
      "gpu_linear_preconv_fused_postconv_prep_component",
      "gpu_linear_final_device_q8_handoff_noqueue",
      "gpu_linear_final_device_q8_handoff_scratch_noqueue",
      "gpu_attention_linear_event_lifetime_combined_alias",
      "gpu_no_resident_q4_cpu_order_z_fallback",
      "gpu_resident_packed_q4_z_replacement",
      "current_selected_ffn_kernel_layout_component_board",
      "current_full_core_attention_front_kernel_algorithm_board",
  ]
  missing_closed_routes = [
      route for route in required_closed_routes if route not in rejected_names
  ]

  checks = [
      {
          "name": "seq112_selected_alpha_beta_gate",
          "pass": (
              seq112.get("required_checks_passed") is True
              and seq112.get("selected_next_route")
              == "linear_preconv_alpha_beta_remaining_algorithm_gate"
              and _has_candidate(
                  routes,
                  112,
                  "reject_linear_preconv_fused_postconv_prep_component",
              )
              and _has_switch(
                  routes,
                  "reject_linear_preconv_fused_postconv_prep_switch_to_alpha_beta_gate",
                  112,
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
          "name": "alpha_beta_gap_is_floor_sized",
          "pass": alpha_beta_gap > floor_gap,
          "detail": {
              "alpha_beta_gap_ms_per_token": alpha_beta_gap,
              "floor_gap_ms_per_token": floor_gap,
          },
      },
      {
          "name": "alpha_beta_profile_is_envelope_gap_not_new_component",
          "pass": live_alpha_beta_wall > live_alpha_beta_kernel > 0.0,
          "detail": {
              "live_alpha_beta_wall_ms_per_token": live_alpha_beta_wall,
              "live_alpha_beta_kernel_ms_per_token": live_alpha_beta_kernel,
              "live_alpha_beta_wall_minus_kernel_ms_per_token": (
                  live_alpha_beta_wall - live_alpha_beta_kernel),
          },
      },
      {
          "name": "known_alpha_beta_source_shapes_only",
          "pass": known_alpha_beta_shapes["pass"] and new_alpha_beta_algorithm_absent["pass"],
          "detail": {
              "known_alpha_beta_shapes": known_alpha_beta_shapes,
              "new_alpha_beta_algorithm_absent": new_alpha_beta_algorithm_absent,
          },
      },
      {
          "name": "alpha_beta_related_closed_routes_present",
          "pass": not missing_closed_routes,
          "detail": {"missing_closed_routes": missing_closed_routes},
      },
      {
          "name": "shared_q8_qkv_and_postconv_are_not_reopened",
          "pass": (
              seq111.get("required_checks_passed") is True
              and seq77.get("verdict", {}).get(
                  "shared_q8_profile_closes_speed_route") is True
              and seq90_q4_derived.get("component_delta_floor_covering") is False
              and seq90_q6_derived.get("component_delta_floor_covering") is False
              and seq112.get("component_floor_covering") is False
              and seq112.get("component_non_regressive") is False
          ),
          "detail": {
              "seq77_qkv_conv_delta_ms_per_token": _num(
                  seq77_derived.get("linear_preconv_qkv_conv_ms_per_token")),
              "seq77_alpha_beta_delta_ms_per_token": _num(
                  seq77_derived.get("linear_preconv_alpha_beta_ms_per_token")),
              "seq90_q4_component_delta_ms_per_token": _num(
                  seq90_q4_derived.get("component_estimated_delta_ms_per_token")),
              "seq90_q6_component_delta_ms_per_token": _num(
                  seq90_q6_derived.get("component_estimated_delta_ms_per_token")),
              "seq112_postconv_projected_cut_ms_per_token": _num(
                  seq112.get("projected_cut_ms_per_token")),
          },
      },
      {
          "name": "prior_stage_boards_closed_before_route_switch",
          "pass": (
              seq109.get("required_checks_passed") is True
              and seq109.get("disposition")
              == "close_current_selected_ffn_kernel_layout_component_route"
              and seq110.get("required_checks_passed") is True
              and seq110.get("disposition")
              == "close_current_full_core_attention_front_kernel_algorithm_board"
              and _has_candidate(
                  routes,
                  109,
                  "close_current_selected_ffn_kernel_layout_component_route",
              )
              and _has_candidate(
                  routes,
                  110,
                  "close_current_full_core_attention_front_kernel_algorithm_board",
              )
          ),
      },
      {
          "name": "linear_delta_is_next_floor_sized_unclosed_gap",
          "pass": (
              linear_delta_stage_gap > floor_gap
              and linear_delta_boundary_gap > floor_gap
          ),
          "detail": {
              "linear_delta_stage_gap_ms_per_token": linear_delta_stage_gap,
              "linear_delta_final_read_gap_ms_per_token": _num(
                  linear_delta_final.get("gap_ms_per_token")),
              "linear_delta_input_upload_gap_ms_per_token": _num(
                  linear_delta_input.get("gap_ms_per_token")),
              "linear_delta_boundary_gap_ms_per_token": linear_delta_boundary_gap,
              "floor_gap_ms_per_token": floor_gap,
          },
      },
  ]

  required = all(check["pass"] for check in checks)
  selected_next_route = (
      "linear_delta_remaining_kernel_algorithm_gate"
      if required else "manual_review_linear_preconv_alpha_beta_gate"
  )
  disposition = (
      "close_current_linear_preconv_alpha_beta_algorithm_board"
      if required else "linear_preconv_alpha_beta_remaining_algorithm_gate_failed"
  )
  next_action = (
      "Close the current linear-preconv alpha/beta board: its remaining "
      "floor-sized gap is covered only by already-tested packed, CPU-order, "
      "and shared-Q8 envelope shapes. The next unit is a linear-delta "
      "remaining-algorithm gate over the final-read/input-upload envelope; "
      "do not launch a decode row until that gate identifies a materially new "
      "component or source proof."
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
          "profile_log": _rel(args.profile_log),
          "seq109": _rel(args.seq109),
          "seq110": _rel(args.seq110),
          "seq111": _rel(args.seq111),
          "seq112": _rel(args.seq112),
          "seq77": _rel(args.seq77),
          "seq90_q4": _rel(args.seq90_q4),
          "seq90_q6": _rel(args.seq90_q6),
          "decode_source": _rel(args.decode_source),
          "engine_source": _rel(args.engine_source),
          "engine_header": _rel(args.engine_header),
          "opencl_source": _rel(args.opencl_source),
      },
      "frontier": frontier_state,
      "stage_gap_ms_per_token": stage_gaps,
      "stage_wall_ms_per_token": stage_walls,
      "linear_preconv_substage": {
          "alpha_beta": alpha_beta,
          "qkv_conv": qkv_conv,
          "postconv_prep": postconv,
      },
      "live_profile_ms_per_token": {
          "linear_preconv_alpha_beta_wall": live_alpha_beta_wall,
          "linear_preconv_alpha_beta_kernel": live_alpha_beta_kernel,
          "linear_preconv_qkv_conv_wall": live_qkv_conv_wall,
          "linear_preconv_qkv_conv_kernel": live_qkv_conv_kernel,
          "linear_preconv_postconv_wall": live_postconv_wall,
          "linear_preconv_postconv_kernel": live_postconv_kernel,
      },
      "closed_route_requirements": {
          "required": required_closed_routes,
          "missing": missing_closed_routes,
      },
      "source_shape": {
          "known_alpha_beta_shapes": known_alpha_beta_shapes,
          "new_alpha_beta_algorithm_absent": new_alpha_beta_algorithm_absent,
      },
      "prior_closure_summary": {
          "seq77_qkv_conv_delta_ms_per_token": _num(
              seq77_derived.get("linear_preconv_qkv_conv_ms_per_token")),
          "seq77_alpha_beta_delta_ms_per_token": _num(
              seq77_derived.get("linear_preconv_alpha_beta_ms_per_token")),
          "seq90_q4_component_delta_ms_per_token": _num(
              seq90_q4_derived.get("component_estimated_delta_ms_per_token")),
          "seq90_q6_component_delta_ms_per_token": _num(
              seq90_q6_derived.get("component_estimated_delta_ms_per_token")),
          "seq112_postconv_projected_cut_ms_per_token": _num(
              seq112.get("projected_cut_ms_per_token")),
      },
      "next_route_budget": {
          "stage": "linear_delta",
          "stage_gap_ms_per_token": linear_delta_stage_gap,
          "final_read_gap_ms_per_token": _num(linear_delta_final.get("gap_ms_per_token")),
          "input_upload_gap_ms_per_token": _num(
              linear_delta_input.get("gap_ms_per_token")),
          "final_read_plus_input_upload_gap_ms_per_token": linear_delta_boundary_gap,
          "floor_gap_ms_per_token": floor_gap,
      },
      "checks": checks,
  }


def write_outputs(out_dir: Path, payload: dict[str, Any]) -> None:
  out_dir.mkdir(parents=True, exist_ok=False)
  (out_dir / "metrics.json").write_text(
      json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  failed = [check["name"] for check in payload["checks"] if check["pass"] is not True]
  frontier = payload["frontier"]
  live = payload["live_profile_ms_per_token"]
  closure = payload["prior_closure_summary"]
  next_budget = payload["next_route_budget"]
  alpha_beta = payload["linear_preconv_substage"]["alpha_beta"]
  lines = [
      "# Linear-Preconv Alpha/Beta Remaining Algorithm Gate",
      "",
      f"- required checks passed: `{str(payload['required_checks_passed']).lower()}`",
      f"- disposition: `{payload['disposition']}`",
      f"- component probe allowed: `{str(payload['component_probe_allowed']).lower()}`",
      f"- decode probe allowed: `{str(payload['decode_probe_allowed']).lower()}`",
      f"- selected next route: `{payload['selected_next_route']}`",
      f"- frontier gap: `{frontier['floor_gap_ms_per_token']:.3f}` ms/token",
      f"- alpha/beta gap: `{_num(alpha_beta.get('gap_ms_per_token')):.3f}` ms/token",
      f"- live alpha/beta wall/kernel: "
      f"`{live['linear_preconv_alpha_beta_wall']:.3f}` / "
      f"`{live['linear_preconv_alpha_beta_kernel']:.3f}` ms/token",
      f"- seq77 alpha/beta/qkv deltas: "
      f"`{closure['seq77_alpha_beta_delta_ms_per_token']:.3f}` / "
      f"`{closure['seq77_qkv_conv_delta_ms_per_token']:.3f}` ms/token",
      f"- seq112 postconv projected cut: "
      f"`{closure['seq112_postconv_projected_cut_ms_per_token']:.3f}` ms/token",
      f"- next linear-delta stage/boundary gap: "
      f"`{next_budget['stage_gap_ms_per_token']:.3f}` / "
      f"`{next_budget['final_read_plus_input_upload_gap_ms_per_token']:.3f}` ms/token",
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
  parser.add_argument("--profile-log", type=Path, default=DEFAULT_PROFILE_LOG)
  parser.add_argument("--seq109", type=Path, default=DEFAULT_SEQ109)
  parser.add_argument("--seq110", type=Path, default=DEFAULT_SEQ110)
  parser.add_argument("--seq111", type=Path, default=DEFAULT_SEQ111)
  parser.add_argument("--seq112", type=Path, default=DEFAULT_SEQ112)
  parser.add_argument("--seq77", type=Path, default=DEFAULT_SEQ77)
  parser.add_argument("--seq90-q4", type=Path, default=DEFAULT_SEQ90_Q4)
  parser.add_argument("--seq90-q6", type=Path, default=DEFAULT_SEQ90_Q6)
  parser.add_argument("--decode-source", type=Path, default=DEFAULT_DECODE)
  parser.add_argument("--engine-source", type=Path, default=DEFAULT_ENGINE)
  parser.add_argument("--engine-header", type=Path, default=DEFAULT_HEADER)
  parser.add_argument("--opencl-source", type=Path, default=DEFAULT_OPENCL)
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
