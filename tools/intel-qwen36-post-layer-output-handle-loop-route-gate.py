#!/usr/bin/env python3
"""Select the next route after the layer-output carrier loop speed rejection.

This is route-control evidence only. It consumes seq126/127/134 evidence and
selects a source/contract gate for the remaining full-attention QK/V host
attention-norm boundary, without launching another carrier speed row.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = "intel-qwen36-post-layer-output-handle-loop-route-gate-v0"

DEFAULT_FRONTIER = ACTIVE / "frontier.json"
DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_ACCEPTED = ACTIVE / "accepted-cuts.json"
DEFAULT_REJECTED = ACTIVE / "rejected-routes.json"
DEFAULT_SEQ110 = ROOT / "output/full-core-attention-front-kernel-algorithm-gate-20260707Tseq110Z/metrics.json"
DEFAULT_SEQ126 = ROOT / "output/resident-hidden-state-carrier-layer-output-handle-loop-contract-gate-20260707Tseq126Z/metrics.json"
DEFAULT_SEQ127 = ROOT / "output/resident-hidden-state-carrier-layer-output-handle-loop-source-gate-20260707Tseq127Z/metrics.json"
DEFAULT_SEQ134 = ROOT / "output/resident-hidden-state-carrier-layer-output-handle-loop-full-core-parity-speed-gate-20260707Tseq134Z/metrics.json"
DEFAULT_OUT_DIR = ROOT / "output/post-layer-output-handle-loop-route-gate-20260707Tseq135Z"


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
      "review_recorded_for_current_best": no_progress.get(
          "review_recorded_for_current_best"),
      "overhead_only_ceiling_tok_s": _num(
          verdict.get("overhead_only_ceiling_tok_s")),
      "can_reach_floor_without_kernel_work": verdict.get(
          "can_reach_floor_without_kernel_work"),
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


def _labels(rows: Any, expected_state: str = "present") -> set[str]:
  out: set[str] = set()
  if not isinstance(rows, list):
    return out
  for row in rows:
    if not isinstance(row, dict) or not isinstance(row.get("label"), str):
      continue
    if row.get(expected_state) is True:
      out.add(row["label"])
  return out


def _speed_detail(seq134: dict[str, Any], check_name: str) -> dict[str, Any]:
  checks = seq134.get("checks")
  if not isinstance(checks, list):
    return {}
  for check in checks:
    if isinstance(check, dict) and check.get("name") == check_name:
      detail = check.get("detail")
      return detail if isinstance(detail, dict) else {}
  return {}


def compute(args: argparse.Namespace) -> dict[str, Any]:
  frontier = _load_json(args.frontier)
  routes = _load_json(args.routes)
  accepted = _load_json(args.accepted)
  rejected = _load_json(args.rejected)
  seq110 = _load_json(args.seq110)
  seq126 = _load_json(args.seq126)
  seq127 = _load_json(args.seq127)
  seq134 = _load_json(args.seq134)

  frontier_state = _frontier_state(frontier)
  gaps = _stage_gaps(frontier)
  floor_gap = frontier_state["floor_gap_ms_per_token"]
  accepted_ids = _accepted_ids(accepted)
  rejected_names = _rejected_names(rejected)

  seq126_blockers = _labels(
      _nested(seq126, "source_shape", "current_blocker_checks"))
  seq126_absent = _labels(
      _nested(seq126, "source_shape", "current_blocker_checks"), "absent")
  seq127_present = _labels(_nested(seq127, "generated", "present_checks"))
  seq127_absent = _labels(_nested(seq127, "generated", "absent_checks"), "absent")

  qkv_blockers = {
      "full_attention_qk_still_consumes_host_attention_norm",
      "full_attention_v_still_consumes_host_attention_norm",
  }
  qkv_source_markers = {
      "full_attention_uses_carrier_prev_handle_but_keeps_rmsnorm_readback",
      "full_attention_qk_v_still_host_attn_norm_boundary",
  }
  required_accepted = {
      "full_core_resident_norm_handoff",
      "attention_front_handoff_linear",
      "fused_full_core_attention_front_local_buffer_reuse",
      "r2_gpu_decode_resident_raw_q6_weight_store",
      "r2_gpu_decode_resident_linear_q6_qkv_weight_store",
  }
  required_lm_head_closed = {
      "gpu_lm_head_device_topk",
      "gpu_lm_head_q6_large_rowstripe_localq8",
      "gpu_lm_head_rmsnorm_q6_handoff",
      "lm_head_q6_pair_dot",
      "lm_head_threads=32",
      "lm_head_ordered_topk_insertion",
  }
  required_nonrepeat_closed = {
      "current_full_core_attention_front_kernel_algorithm_board",
      "current_linear_preconv_alpha_beta_algorithm_board",
      "current_linear_delta_algorithm_board",
      "current_resident_hidden_state_carrier_full_boundary_speed_shape",
      "current_resident_hidden_state_carrier_tail_readback_loop_shape",
      "current_resident_hidden_state_carrier_layer_output_loop_full_core_parity_speed_shape",
      "gpu_linear_preconv_shared_q8_qkv_conv_root_component",
      "gpu_attention_linear_event_lifetime_combined_alias",
      "gpu_linear_final_device_q8_attention_front_handoff",
  }
  missing_accepted = sorted(required_accepted - accepted_ids)
  missing_lm_head_closed = sorted(required_lm_head_closed - rejected_names)
  missing_nonrepeat_closed = sorted(required_nonrepeat_closed - rejected_names)

  speed_delta = _speed_detail(seq134, "speed_delta_is_not_progress")
  growth = _speed_detail(seq134, "selected_ffn_gain_consumed_by_tail_attention_growth")
  lm_head_gap = gaps.get("lm_head_gpu", 0.0)
  full_core_gap = gaps.get("full_core", 0.0)
  attention_front_gap = gaps.get("attention_front", 0.0)
  full_attention_gap = max(full_core_gap, attention_front_gap)

  checks = [
      {
          "name": "seq134_selected_this_route_gate",
          "pass": (
              seq134.get("required_checks_passed") is True
              and seq134.get("disposition")
              == "reject_layer_output_loop_full_core_parity_as_speed_cut"
              and seq134.get("selected_next_route")
              == "post_layer_output_handle_loop_route_gate"
              and _has_switch(
                  routes,
                  "reject_layer_output_loop_full_core_parity_speed_switch_to_route_gate",
                  134,
              )
              and _has_candidate(
                  routes,
                  134,
                  "reject_layer_output_loop_full_core_parity_as_speed_cut",
              )
          ),
      },
      {
          "name": "frontier_still_below_floor_with_reviewed_hard_stall",
          "pass": (
              frontier_state["current_best_tps"] < frontier_state["floor_tps"]
              and frontier_state["wall_ms_per_token"]
              > frontier_state["floor_budget_ms_per_token"]
              > 0.0
              and frontier_state["hard_stall_breached"] is True
              and frontier_state["review_recorded_for_current_best"] is True
              and frontier_state["overhead_only_ceiling_tok_s"]
              > frontier_state["floor_tps"]
          ),
          "detail": frontier_state,
      },
      {
          "name": "layer_output_loop_speed_shape_is_closed",
          "pass": (
              speed_delta.get("explore_tps", 0.0)
              < speed_delta.get("best_tps", 0.0)
              and abs(_num(speed_delta.get("relative_delta")))
              < frontier_state["noise_rel"]
              and _num(growth.get("ffn_tail_plus_attention_front_growth_ms_per_token"))
              > 0.9 * _num(growth.get("selected_ffn_gain_ms_per_token"))
              and "current_resident_hidden_state_carrier_layer_output_loop_full_core_parity_speed_shape"
              in rejected_names
          ),
          "detail": {
              "speed_delta": speed_delta,
              "tail_attention_growth": growth,
          },
      },
      {
          "name": "full_attention_qkv_host_boundary_remains_after_layer_output_loop",
          "pass": (
              qkv_blockers.issubset(seq126_blockers)
              and qkv_source_markers.issubset(seq127_present)
              and seq127.get("required_checks_passed") is True
          ),
          "detail": {
              "contract_blockers_present": sorted(qkv_blockers & seq126_blockers),
              "source_markers_present": sorted(qkv_source_markers & seq127_present),
          },
      },
      {
          "name": "full_attention_root_is_floor_sized_and_resident_weight_backed",
          "pass": (
              full_attention_gap > floor_gap
              and not missing_accepted
              and seq110.get("required_checks_passed") is True
              and seq110.get("disposition")
              == "close_current_full_core_attention_front_kernel_algorithm_board"
          ),
          "detail": {
              "full_core_gap_ms_per_token": full_core_gap,
              "attention_front_gap_ms_per_token": attention_front_gap,
              "floor_gap_ms_per_token": floor_gap,
              "missing_accepted": missing_accepted,
              "seq110_disposition": seq110.get("disposition"),
          },
      },
      {
          "name": "lm_head_is_not_the_next_root",
          "pass": (
              lm_head_gap < floor_gap
              and not missing_lm_head_closed
              and "lm_head_q6_input_handle_handoff_api_absent" in seq126_absent
              and "no_lm_head_input_handle_handoff_claim" in seq127_absent
          ),
          "detail": {
              "lm_head_gap_ms_per_token": lm_head_gap,
              "floor_gap_ms_per_token": floor_gap,
              "missing_closed_routes": missing_lm_head_closed,
              "input_handle_api_absent": (
                  "lm_head_q6_input_handle_handoff_api_absent" in seq126_absent),
          },
      },
      {
          "name": "route_selection_does_not_reopen_closed_boards",
          "pass": not missing_nonrepeat_closed,
          "detail": {"missing_closed_routes": missing_nonrepeat_closed},
      },
  ]
  required_checks_passed = all(bool(row.get("pass")) for row in checks)
  selected_next_route = (
      "resident_hidden_state_carrier_full_attention_qkv_handle_contract_gate"
      if required_checks_passed
      else "post_layer_output_handle_loop_route_review"
  )
  disposition = (
      "select_full_attention_qkv_resident_input_handle_contract_gate"
      if required_checks_passed
      else "post_layer_output_handle_loop_route_gate_incomplete"
  )

  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "inputs": {
          "frontier": _rel(args.frontier),
          "routes": _rel(args.routes),
          "accepted": _rel(args.accepted),
          "rejected": _rel(args.rejected),
          "seq110_full_core_board_gate": _rel(args.seq110),
          "seq126_layer_output_contract_gate": _rel(args.seq126),
          "seq127_layer_output_source_gate": _rel(args.seq127),
          "seq134_layer_output_speed_gate": _rel(args.seq134),
      },
      "frontier": frontier_state,
      "remaining_gaps_ms_per_token": {
          "full_core": full_core_gap,
          "attention_front": attention_front_gap,
          "lm_head_gpu": lm_head_gap,
          "floor_gap": floor_gap,
      },
      "route_signal": {
          "selected_root": "full_attention_qkv_host_attention_norm_boundary",
          "selected_next_route": selected_next_route,
          "why_not_carrier_speed_row": (
              "Seq134 already closed the correctness-valid layer-output "
              "handle loop as a speed shape: it was below the frontier and "
              "inside the noise band, with selected-FFN savings consumed by "
              "FFN-tail plus attention-front growth."
          ),
          "why_not_current_full_core_algorithm_board": (
              "Seq110 closes the current full-core/attention-front algorithm "
              "board. The selected next unit is only a source/contract gate "
              "for QK/V resident input handles, not a decode or local "
              "full-core micro-variant."
          ),
          "why_not_lm_head": (
              "The LM-head stage gap is below the floor miss, narrow LM-head "
              "routes are closed, and the input-handle API is explicitly "
              "absent in seq126/127 evidence."
          ),
      },
      "checks": checks,
      "required_checks_passed": required_checks_passed,
      "decode_probe_allowed": False,
      "component_probe_allowed": False,
      "source_contract_gate_allowed": required_checks_passed,
      "speedup_claims_allowed": False,
      "disposition": disposition,
      "selected_next_route": selected_next_route,
      "next_route_reason": (
          "Run a source/contract gate for full-attention QK/V resident "
          "attention-norm input handles. It must prove the API/wiring shape "
          "and no-token policy before any target compile or decode row."
          if required_checks_passed
          else "Route evidence is incomplete; do not launch another speed row."
      ),
  }


def write_outputs(metrics: dict[str, Any], out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
  )
  gaps = metrics["remaining_gaps_ms_per_token"]
  lines = [
      "# Post Layer-Output Handle Loop Route Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- speedup_claims_allowed: `{str(metrics['speedup_claims_allowed']).lower()}`",
      "",
      "## Summary",
      "",
      (
          "Seq134 closes the current carrier layer-output loop speed shape. "
          "The remaining route selected here is a no-token source/contract "
          "gate for full-attention QK/V resident input handles."
      ),
      "",
      (
          f"Full-core and attention-front gaps are `{gaps['full_core']:.3f}` "
          f"and `{gaps['attention_front']:.3f}` ms/token versus a "
          f"`{gaps['floor_gap']:.3f}` ms/token floor miss. LM-head gap is "
          f"`{gaps['lm_head_gpu']:.3f}` ms/token and is not selected."
      ),
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--frontier", type=Path, default=DEFAULT_FRONTIER)
  parser.add_argument("--routes", type=Path, default=DEFAULT_ROUTES)
  parser.add_argument("--accepted", type=Path, default=DEFAULT_ACCEPTED)
  parser.add_argument("--rejected", type=Path, default=DEFAULT_REJECTED)
  parser.add_argument("--seq110", type=Path, default=DEFAULT_SEQ110)
  parser.add_argument("--seq126", type=Path, default=DEFAULT_SEQ126)
  parser.add_argument("--seq127", type=Path, default=DEFAULT_SEQ127)
  parser.add_argument("--seq134", type=Path, default=DEFAULT_SEQ134)
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
