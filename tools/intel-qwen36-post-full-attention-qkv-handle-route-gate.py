#!/usr/bin/env python3
"""Select the next route after the full-attention QK/V handle speed rejection.

This is route-control evidence only. It consumes the valid defer-parity,
cache-hit QK/V handle speed gate and selects the next no-token source/contract
gate without launching another decode row.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = "intel-qwen36-post-full-attention-qkv-handle-route-gate-v0"

DEFAULT_FRONTIER = ACTIVE / "frontier.json"
DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_REJECTED = ACTIVE / "rejected-routes.json"
DEFAULT_SEQ110 = (
    ROOT / "output/full-core-attention-front-kernel-algorithm-gate-20260707Tseq110Z"
    / "metrics.json"
)
DEFAULT_SEQ137 = (
    ROOT / "output/resident-hidden-state-carrier-full-attention-qkv-handle-source-gate-20260707Tseq137Z"
    / "metrics.json"
)
DEFAULT_SEQ141 = (
    ROOT / "output/resident-hidden-state-carrier-full-attention-qkv-handle-defer-speed-gate-20260707Tseq141Z"
    / "metrics.json"
)
DEFAULT_SEQ142 = (
    ROOT
    / "output/resident-hidden-state-carrier-full-attention-qkv-handle-defer-binaryhit-speed-gate-20260707Tseq142Z"
    / "metrics.json"
)
DEFAULT_OUT_DIR = ROOT / "output/post-full-attention-qkv-handle-route-gate-20260707Tseq143Z"


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
      "hard_stall_breached": no_progress.get("hard_stall_breached"),
      "review_recorded_for_current_best": no_progress.get(
          "review_recorded_for_current_best"),
      "overhead_only_ceiling_tok_s": _num(
          verdict.get("overhead_only_ceiling_tok_s")),
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


def _check_detail(metrics: dict[str, Any], name: str) -> dict[str, Any]:
  checks = metrics.get("checks")
  if not isinstance(checks, list):
    return {}
  for check in checks:
    if isinstance(check, dict) and check.get("name") == name:
      detail = check.get("detail")
      return detail if isinstance(detail, dict) else {}
  return {}


def compute(args: argparse.Namespace) -> dict[str, Any]:
  frontier = _load_json(args.frontier)
  routes = _load_json(args.routes)
  rejected = _load_json(args.rejected)
  seq110 = _load_json(args.seq110)
  seq137 = _load_json(args.seq137)
  seq141 = _load_json(args.seq141)
  seq142 = _load_json(args.seq142)

  frontier_state = _frontier_state(frontier)
  floor_gap = frontier_state["floor_gap_ms_per_token"]
  gaps = _stage_gaps(frontier)
  rejected_names = _rejected_names(rejected)
  generated_present = _labels(_nested(seq137, "generated", "present_checks"))
  generated_absent = _labels(
      _nested(seq137, "generated", "absent_checks"), "absent")

  seq142_stack = _check_detail(seq142, "explore_row_uses_qkv_handle_stack")
  seq142_cache = _check_detail(seq142, "explore_row_is_artifact_free_speed_probe")
  seq142_speed = seq142.get("speed")
  seq142_speed = seq142_speed if isinstance(seq142_speed, dict) else {}
  seq141_cache = _check_detail(seq141, "explore_row_is_artifact_free_speed_probe")

  closed_required = {
      "current_selected_ffn_kernel_layout_component_board",
      "current_offline_repack_streaming_layout_board",
      "current_moe_routed_down_fusion_board",
      "current_full_core_attention_front_kernel_algorithm_board",
      "gpu_full_attention_state_resident_history",
      "gpu_full_attention_flat_history_cache",
  }
  missing_closed = sorted(closed_required - rejected_names)
  qkv_host_output_markers = {
      "qk_helper_keeps_host_q_and_k_outputs",
      "v_helpers_keep_host_v_output",
      "host_qk_fallback_preserved",
      "host_v_fallback_preserved",
  }

  checks = [
      {
          "name": "seq142_valid_qkv_handle_speed_rejection",
          "pass": (
              seq142.get("required_checks_passed") is True
              and seq142.get("disposition")
              == "reject_full_attention_qkv_handle_as_speed_cut"
              and seq142.get("selected_next_route")
              == "post_full_attention_qkv_handle_route_gate"
              and _num(seq142_speed.get("explore_tps"))
              < _num(seq142_speed.get("best_tps"))
              and _num(seq142_speed.get("relative_delta"))
              <= -frontier_state["noise_rel"]
          ),
          "detail": {
              "speed_gate": _rel(args.seq142),
              "speed": seq142_speed,
          },
      },
      {
          "name": "seq142_preserves_frontier_defer_and_cache_parity",
          "pass": (
              _nested(seq142_stack, "candidate_flags",
                      "defer_ffn_down_finish_bundle") is True
              and _nested(seq142_stack, "candidate_flags",
                          "resident_hidden_state_carrier_full_attention_qkv_handle")
              is True
              and _nested(seq142_cache, "candidate_cache", "binary_hit") is True
              and _nested(seq142_cache, "candidate_cache", "tokens_hit") is True
          ),
          "detail": {
              "stack": seq142_stack,
              "cache": seq142_cache,
          },
      },
      {
          "name": "seq141_cold_binary_probe_not_route_closure",
          "pass": (
              seq141.get("required_checks_passed") is False
              and _nested(seq141_cache, "candidate_cache", "binary_hit") is False
          ),
          "detail": {
              "seq141_gate": _rel(args.seq141),
              "cache": seq141_cache,
          },
      },
      {
          "name": "frontier_still_below_floor_after_qkv_rejection",
          "pass": (
              frontier_state["current_best_tps"] < frontier_state["floor_tps"]
              and floor_gap > 0.0
              and frontier_state["hard_stall_breached"] is True
              and frontier_state["review_recorded_for_current_best"] is True
              and frontier_state["overhead_only_ceiling_tok_s"]
              > frontier_state["floor_tps"]
          ),
          "detail": frontier_state,
      },
      {
          "name": "qkv_source_keeps_host_outputs_and_has_no_core_handle_claim",
          "pass": (
              seq137.get("required_checks_passed") is True
              and qkv_host_output_markers.issubset(generated_present)
              and "no_direct_full_attention_core_handle_claim" in generated_absent
          ),
          "detail": {
              "present": sorted(qkv_host_output_markers & generated_present),
              "absent": sorted(generated_absent),
          },
      },
      {
          "name": "full_attention_core_history_boundary_is_floor_sized",
          "pass": (
              gaps.get("full_core", 0.0) > floor_gap
              and gaps.get("attention_front", 0.0) > floor_gap
              and seq110.get("required_checks_passed") is True
              and seq110.get("disposition")
              == "close_current_full_core_attention_front_kernel_algorithm_board"
          ),
          "detail": {
              "full_core_gap_ms_per_token": gaps.get("full_core"),
              "attention_front_gap_ms_per_token": gaps.get("attention_front"),
              "floor_gap_ms_per_token": floor_gap,
              "seq110_disposition": seq110.get("disposition"),
          },
      },
      {
          "name": "next_route_does_not_reopen_closed_boards",
          "pass": not missing_closed,
          "detail": {
              "missing_closed_routes": missing_closed,
              "closed_routes_checked": sorted(closed_required),
          },
      },
      {
          "name": "prior_qkv_correctness_switch_recorded",
          "pass": _has_switch(
              routes,
              "accept_full_attention_qkv_handle_decode_switch_to_speed_explore_gate",
              139,
          ),
      },
  ]
  required_checks_passed = all(bool(row.get("pass")) for row in checks)
  selected_next_route = (
      "resident_hidden_state_carrier_full_attention_core_history_contract_gate"
      if required_checks_passed
      else "post_full_attention_qkv_handle_route_review"
  )
  disposition = (
      "select_full_attention_core_history_contract_gate"
      if required_checks_passed
      else "post_full_attention_qkv_handle_route_gate_incomplete"
  )

  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "inputs": {
          "frontier": _rel(args.frontier),
          "routes": _rel(args.routes),
          "rejected": _rel(args.rejected),
          "seq110_full_core_board_gate": _rel(args.seq110),
          "seq137_qkv_source_gate": _rel(args.seq137),
          "seq141_cold_binary_speed_gate": _rel(args.seq141),
          "seq142_valid_speed_gate": _rel(args.seq142),
      },
      "frontier": frontier_state,
      "remaining_gaps_ms_per_token": {
          "full_core": gaps.get("full_core", 0.0),
          "attention_front": gaps.get("attention_front", 0.0),
          "selected_ffn": gaps.get("selected_ffn", 0.0),
          "floor_gap": floor_gap,
      },
      "route_signal": {
          "selected_root": "full_attention_core_history_resident_boundary",
          "selected_next_route": selected_next_route,
          "why_not_qkv_handle_retry": (
              "Seq142 is a defer-parity, cache-hit speed row and still "
              "regresses materially versus the frontier. Seq141 is retained "
              "only as a cold-binary mismatch probe."
          ),
          "why_not_current_full_core_algorithm_board": (
              "Seq110 closes the current full-core/attention-front algorithm "
              "board. The selected unit is a no-token contract for resident "
              "Q/K normalization, RoPE, K/V history, and full-core input "
              "ownership, not another local full-core algorithm variant."
          ),
          "why_not_kv_history_only": (
              "Standalone resident K/V history and flat-history cache routes "
              "are closed. The next contract must not re-run those shapes "
              "without also removing the host Q/K/V output and q/k RoPE/core "
              "input boundaries."
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
          "Run a no-token contract gate for a full-attention core/history "
          "resident boundary. It must define Q/K normalization, RoPE, K/V "
          "history update, q_full/core input ownership, and fallback/correctness "
          "guards before any source edit or token row."
          if required_checks_passed
          else "Route evidence is incomplete; fix the failed checks before "
               "launching more token rows."
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
      "# Post Full-Attention QK/V Handle Route Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- speedup_claims_allowed: `{str(metrics['speedup_claims_allowed']).lower()}`",
      "",
      "## Summary",
      "",
      (
          "Seq142 closes the defer-parity, cache-hit full-attention QK/V "
          "input-handle speed shape. The next route is a no-token contract "
          "gate for the larger full-attention core/history resident boundary."
      ),
      "",
      (
          f"Full-core and attention-front gaps are `{gaps['full_core']:.3f}` "
          f"and `{gaps['attention_front']:.3f}` ms/token versus a "
          f"`{gaps['floor_gap']:.3f}` ms/token floor miss."
      ),
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--frontier", type=Path, default=DEFAULT_FRONTIER)
  parser.add_argument("--routes", type=Path, default=DEFAULT_ROUTES)
  parser.add_argument("--rejected", type=Path, default=DEFAULT_REJECTED)
  parser.add_argument("--seq110", type=Path, default=DEFAULT_SEQ110)
  parser.add_argument("--seq137", type=Path, default=DEFAULT_SEQ137)
  parser.add_argument("--seq141", type=Path, default=DEFAULT_SEQ141)
  parser.add_argument("--seq142", type=Path, default=DEFAULT_SEQ142)
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
