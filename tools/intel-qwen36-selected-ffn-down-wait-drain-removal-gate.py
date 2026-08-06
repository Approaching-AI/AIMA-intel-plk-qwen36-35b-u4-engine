#!/usr/bin/env python3
"""Gate the selected-FFN down-wait drain-removal continuation.

This is source/component route-control evidence only. It turns seq91's selected
down-wait route into one concrete component proof, with a hard arithmetic bar
before any decode row is admissible.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = "intel-qwen36-selected-ffn-down-wait-drain-removal-gate-v0"
DEFAULT_FRONTIER = ACTIVE / "frontier.json"
DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_REJECTED = ACTIVE / "rejected-routes.json"
DEFAULT_SEQ53 = ROOT / "output/q6-defer-drain-budget-20260706Tseq53Z/metrics.json"
DEFAULT_SEQ66 = ROOT / "output/down-tail-fusion-budget-20260706Tseq66Z/metrics.json"
DEFAULT_SEQ80 = ROOT / "output/nonatomic-down-tail-gate-20260707Tseq80Z/gate.json"
DEFAULT_SEQ82 = ROOT / "output/q6-nonatomic-down-tail-decode-gate-20260707Tseq82-r2Z/metrics.json"
DEFAULT_SEQ91 = ROOT / "output/post-linear-preconv-route-gate-20260707Tseq91Z/metrics.json"
DEFAULT_ENGINE = ROOT / "engine/src/gpu_q4x8_matvec.cpp"
DEFAULT_HEADER = ROOT / "engine/include/intel_qwen36/gpu_q4x8_matvec.hpp"
DEFAULT_OPENCL = ROOT / "engine/gpu/opencl/q4x8_matvec.cl"
DEFAULT_DECODE = ROOT / "tools/intel-qwen36-r2-gpu-decode-smoke.py"
DEFAULT_OUT_DIR = ROOT / "output/selected-ffn-down-wait-drain-removal-gate-20260707Tseq92Z"
Q6_LAYER_INVOCATIONS_PER_TOKEN = 20.0


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
  wall = _num(per_token.get("wall"))
  floor_budget = _num(verdict.get("floor_budget_ms_per_token"))
  return {
      "current_best_tps": _num(_nested(no_progress, "last_significant_improvement", "tps")),
      "floor_tps": _num(verdict.get("floor_tps")),
      "wall_ms_per_token": wall,
      "floor_budget_ms_per_token": floor_budget,
      "floor_gap_ms_per_token": max(0.0, wall - floor_budget),
      "runs_since_significant_improvement": no_progress.get(
          "runs_since_significant_improvement"),
      "noise_rel": _num(_nested(no_progress, "noise", "rel")),
  }


def _stage_gaps(frontier: dict[str, Any]) -> dict[str, float]:
  budget = frontier.get("goal_budget")
  budget = budget if isinstance(budget, dict) else {}
  out: dict[str, float] = {}
  for row in budget.get("stage_kernel_gap_estimates_ms_per_token", []):
    if isinstance(row, dict) and isinstance(row.get("stage"), str):
      out[row["stage"]] = _num(row.get("gap_ms_per_token"))
  return out


def _substage_gaps(frontier: dict[str, Any]) -> dict[str, float]:
  budget = frontier.get("goal_budget")
  budget = budget if isinstance(budget, dict) else {}
  out: dict[str, float] = {}
  for row in budget.get("substage_gap_estimates_ms_per_token", []):
    if (
        isinstance(row, dict)
        and isinstance(row.get("stage"), str)
        and isinstance(row.get("substage"), str)
    ):
      out[f"{row['stage']}.{row['substage']}"] = _num(row.get("gap_ms_per_token"))
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


def _has_markers(text: str, markers: list[str]) -> dict[str, Any]:
  missing = [marker for marker in markers if marker not in text]
  return {"pass": not missing, "missing": missing, "marker_count": len(markers)}


def _lacks_markers(text: str, markers: list[str]) -> dict[str, Any]:
  present = [marker for marker in markers if marker in text]
  return {"pass": not present, "present": present, "marker_count": len(markers)}


def compute(args: argparse.Namespace) -> dict[str, Any]:
  frontier = _load_json(args.frontier)
  routes = _load_json(args.routes)
  rejected = _load_json(args.rejected)
  seq53 = _load_json(args.q6_defer_drain)
  seq66 = _load_json(args.down_tail_fusion)
  seq80 = _load_json(args.nonatomic_component)
  seq82 = _load_json(args.nonatomic_decode)
  seq91 = _load_json(args.post_linear_preconv)
  source = "\n".join([
      _read(args.decode_source),
      _read(args.engine_source),
      _read(args.engine_header),
      _read(args.opencl_source),
  ])

  frontier_state = _frontier_summary(frontier)
  floor_gap = frontier_state["floor_gap_ms_per_token"]
  stage_gaps = _stage_gaps(frontier)
  substage_gaps = _substage_gaps(frontier)
  rejected_names = _route_names(rejected)
  selected_gap = stage_gaps.get("selected_ffn", 0.0)
  down_wait_gap = substage_gaps.get("selected_ffn.down_kernel_wait", 0.0)
  seq53_derived = seq53.get("derived", {}) if isinstance(seq53, dict) else {}
  seq53_verdict = seq53.get("verdict", {}) if isinstance(seq53, dict) else {}
  seq66_derived = seq66.get("derived", {}) if isinstance(seq66, dict) else {}
  seq66_verdict = seq66.get("verdict", {}) if isinstance(seq66, dict) else {}
  seq80_metrics = seq80.get("metrics", {}) if isinstance(seq80, dict) else {}

  baseline_selected_tail = (
      _num(seq82.get("baseline_selected_ffn_ms_per_token"))
      + _num(seq82.get("baseline_ffn_tail_ms_per_token"))
  )
  candidate_selected_tail = (
      _num(seq82.get("candidate_selected_ffn_ms_per_token"))
      + _num(seq82.get("candidate_ffn_tail_ms_per_token"))
  )
  nonatomic_delta_vs_baseline = candidate_selected_tail - baseline_selected_tail
  required_cut_vs_nonatomic_ms = nonatomic_delta_vs_baseline + floor_gap
  required_cut_vs_nonatomic_us_per_layer = (
      required_cut_vs_nonatomic_ms * 1000.0 / Q6_LAYER_INVOCATIONS_PER_TOKEN
      if Q6_LAYER_INVOCATIONS_PER_TOKEN > 0
      else 0.0
  )
  nonatomic_shell_us_per_layer = (
      _num(seq82.get("nonatomic_shell_kernel_us_profile"))
      / (8.0 * Q6_LAYER_INVOCATIONS_PER_TOKEN)
  )
  target_shell_us_per_layer = max(
      0.0, nonatomic_shell_us_per_layer - required_cut_vs_nonatomic_us_per_layer
  )
  component_shell_us = _num(seq80_metrics.get("shell_sum_min_us"))
  component_combined_down_us = _num(seq80_metrics.get("combined_down_min_us"))
  component_required_target_us = max(
      0.0, component_shell_us - required_cut_vs_nonatomic_us_per_layer
  )

  required_closed_routes = [
      "gpu_q6_defer_finish_without_tail_drain_elimination",
      "gpu_q6_defer_tail_read_drain_noqueue",
      "gpu_q6_defer_tail_rmsnorm_input_noqueue",
      "gpu_ffn_tail_resident_input_noqueue",
      "gpu_ffn_tail_plus_attention_residual_carrier_noqueue",
      "gpu_down_tail_hidden_row_serial_fusion",
      "gpu_ffn_tail_atomic_reduction_from_down_handles",
      "gpu_direct_q6_down_tail_atomic_from_q6_workitems",
      "gpu_q6_nonatomic_down_tail_decode_fusion",
      "gpu_selected_gateup_top16_indexed_material_component",
      "gpu_selected_shared_q6_down_occupancy4_component",
  ]
  missing_closed_routes = [
      route for route in required_closed_routes if route not in rejected_names
  ]
  closed_path_markers = _has_markers(
      source,
      [
          "q6k_selected_down_matvec_rowstripe_expert8_plus_shared_tail_contrib_raw",
          "ffn_tail_reduce9_contrib_f32",
          "q6k_selected_down_matvec_rowstripe_expert8_plus_shared_tail_atomic_raw",
          "RunResidentRawQ6KExpert8PlusSharedToFfnTailNonAtomic(",
          "RunResidentRawQ6KExpert8PlusSharedToFfnTailAtomic(",
          "IQ36_SELECTED_SHARED_Q6_DOWN_TAIL_NONATOMIC",
          "IQ36_SELECTED_SHARED_Q6_DOWN_TAIL_DIRECT",
      ],
  )
  proposed_source_gap = _lacks_markers(
      source,
      [
          "q6k_selected_down_matvec_rowstripe_expert8_plus_shared_tail_rowgroup_reduce_raw",
          "RunResidentRawQ6KExpert8PlusSharedToFfnTailRowgroupReduce",
          "IQ36_SELECTED_SHARED_Q6_DOWN_TAIL_ROWGROUP_REDUCE",
      ],
  )

  checks = [
      {
          "name": "frontier_still_below_floor",
          "pass": frontier_state["wall_ms_per_token"]
          > frontier_state["floor_budget_ms_per_token"]
          > 0.0,
      },
      {
          "name": "seq91_selected_drain_route_selected",
          "pass": seq91.get("required_checks_passed") is True
          and seq91.get("selected_next_route")
          == "selected_ffn_down_wait_drain_removal_source_component_gate",
      },
      {
          "name": "seq91_switch_recorded",
          "pass": _has_switch(
              routes,
              "switch_to_selected_ffn_down_wait_drain_removal_source_component_gate",
              91,
          ),
      },
      {
          "name": "selected_ffn_and_down_wait_cover_floor",
          "pass": selected_gap > floor_gap and down_wait_gap > floor_gap,
          "detail": {
              "selected_ffn_gap_ms_per_token": selected_gap,
              "selected_down_wait_gap_ms_per_token": down_wait_gap,
              "floor_gap_ms_per_token": floor_gap,
          },
      },
      {
          "name": "seq53_proves_drain_shift_and_tail_elimination_budget",
          "pass": bool(seq53_verdict.get("tail_drain_shift_confirmed"))
          and _num(seq53_derived.get("selected_down_wait_saved_ms_per_token")) > floor_gap
          and _num(seq53_derived.get("ffn_tail_growth_ms_per_token")) > floor_gap
          and bool(seq53_derived.get("tail_drain_elimination_clears_floor")),
      },
      {
          "name": "serial_hidden_row_fusion_closed",
          "pass": bool(seq66_derived.get("hidden_row_serial_q6_down_tail_fusion_closed"))
          and seq66_verdict.get("naive_hidden_row_serial_fusion_promotable") is False,
      },
      {
          "name": "nonatomic_component_was_exact_but_decode_rejected",
          "pass": seq80.get("required_checks_passed") is True
          and seq82.get("required_checks_passed") is True
          and seq82.get("disposition")
          == "rejected_q6_nonatomic_down_tail_decode_regresses_8tok",
      },
      {
          "name": "nonatomic_decode_delta_sets_material_component_bar",
          "pass": required_cut_vs_nonatomic_ms > floor_gap
          and target_shell_us_per_layer > 0.0
          and component_required_target_us > 0.0,
          "detail": {
              "baseline_selected_plus_tail_ms_per_token": baseline_selected_tail,
              "candidate_selected_plus_tail_ms_per_token": candidate_selected_tail,
              "nonatomic_delta_vs_baseline_ms_per_token": nonatomic_delta_vs_baseline,
              "required_cut_vs_nonatomic_ms_per_token": required_cut_vs_nonatomic_ms,
              "required_cut_vs_nonatomic_us_per_layer": required_cut_vs_nonatomic_us_per_layer,
          },
      },
      {
          "name": "closed_down_tail_routes_recorded",
          "pass": not missing_closed_routes,
          "detail": {"missing_closed_routes": missing_closed_routes},
      },
      {
          "name": "closed_atomic_and_nonatomic_source_paths_present",
          "pass": closed_path_markers["pass"],
          "detail": closed_path_markers,
      },
      {
          "name": "rowgroup_reduce_component_path_not_yet_present",
          "pass": proposed_source_gap["pass"],
          "detail": proposed_source_gap,
      },
      {
          "name": "seq91_candidate_recorded",
          "pass": _has_candidate(
              routes,
              91,
              "select_selected_ffn_down_wait_drain_removal_source_component_gate",
          ),
      },
  ]
  required = all(check["pass"] for check in checks)
  selected_next_route = (
      "selected_shared_q6_down_tail_rowgroup_reduce_component_proof"
      if required
      else "route_selection_needs_manual_review"
  )
  disposition = (
      "rowgroup_reduce_component_contract_ready"
      if required
      else "selected_ffn_drain_removal_gate_failed"
  )
  next_action = (
      "Implement a component-only selected/shared Q6 down-to-tail rowgroup-local "
      "reduce proof. The candidate must keep one contributor work item per "
      "selected/shared expert for each hidden row, reduce within the rowgroup, "
      "avoid global contribution scratch and atomics, and beat the hard bar "
      f"of {component_required_target_us:.3f} us per layer component shell "
      f"or {target_shell_us_per_layer:.3f} us per layer in the seq82 profile "
      "projection before any decode row is allowed."
      if required
      else "Fix failed source/component-gate checks before selecting another route."
  )

  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "required_checks_passed": required,
      "speedup_claims_allowed": False,
      "disposition": disposition,
      "selected_next_route": selected_next_route,
      "next_action": next_action,
      "inputs": {
          "frontier": _rel(args.frontier),
          "routes": _rel(args.routes),
          "rejected": _rel(args.rejected),
          "q6_defer_drain": _rel(args.q6_defer_drain),
          "down_tail_fusion": _rel(args.down_tail_fusion),
          "nonatomic_component": _rel(args.nonatomic_component),
          "nonatomic_decode": _rel(args.nonatomic_decode),
          "post_linear_preconv": _rel(args.post_linear_preconv),
          "decode_source": _rel(args.decode_source),
          "engine_source": _rel(args.engine_source),
          "engine_header": _rel(args.engine_header),
          "opencl_source": _rel(args.opencl_source),
      },
      "frontier": frontier_state,
      "stage_gap_ms_per_token": stage_gaps,
      "substage_gap_ms_per_token": substage_gaps,
      "drain_shift_evidence": {
          "selected_down_wait_saved_ms_per_token": _num(
              seq53_derived.get("selected_down_wait_saved_ms_per_token")),
          "ffn_tail_growth_ms_per_token": _num(
              seq53_derived.get("ffn_tail_growth_ms_per_token")),
          "tail_drain_elimination_clears_floor": bool(
              seq53_derived.get("tail_drain_elimination_clears_floor")),
      },
      "nonatomic_decode_arithmetic": {
          "baseline_selected_plus_tail_ms_per_token": baseline_selected_tail,
          "candidate_selected_plus_tail_ms_per_token": candidate_selected_tail,
          "nonatomic_delta_vs_baseline_ms_per_token": nonatomic_delta_vs_baseline,
          "floor_gap_ms_per_token": floor_gap,
          "required_cut_vs_nonatomic_ms_per_token": required_cut_vs_nonatomic_ms,
          "required_cut_vs_nonatomic_us_per_layer": required_cut_vs_nonatomic_us_per_layer,
          "nonatomic_shell_us_per_layer_profile": nonatomic_shell_us_per_layer,
          "target_shell_us_per_layer_profile": target_shell_us_per_layer,
      },
      "component_gate_bar": {
          "seq80_component_shell_us": component_shell_us,
          "seq80_component_combined_down_us": component_combined_down_us,
          "required_component_shell_us": component_required_target_us,
          "q6_layer_invocations_per_token": Q6_LAYER_INVOCATIONS_PER_TOKEN,
      },
      "source_contract": {
          "closed_path_markers": closed_path_markers,
          "rowgroup_reduce_missing_markers": proposed_source_gap,
      },
      "closed_route_requirements": {
          "required": required_closed_routes,
          "missing": missing_closed_routes,
      },
      "checks": checks,
  }


def write_outputs(out_dir: Path, payload: dict[str, Any]) -> None:
  out_dir.mkdir(parents=True, exist_ok=False)
  (out_dir / "metrics.json").write_text(
      json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  failed = [check["name"] for check in payload["checks"] if not check["pass"]]
  frontier = payload["frontier"]
  bar = payload["component_gate_bar"]
  arithmetic = payload["nonatomic_decode_arithmetic"]
  lines = [
      "# Selected-FFN Down-Wait Drain-Removal Gate",
      "",
      f"- required checks passed: `{str(payload['required_checks_passed']).lower()}`",
      f"- disposition: `{payload['disposition']}`",
      f"- selected next route: `{payload['selected_next_route']}`",
      f"- frontier gap: `{frontier['floor_gap_ms_per_token']:.3f}` ms/token",
      f"- seq82 non-atomic selected+tail delta: "
      f"`{arithmetic['nonatomic_delta_vs_baseline_ms_per_token']:.3f}` ms/token",
      f"- required cut vs non-atomic: "
      f"`{arithmetic['required_cut_vs_nonatomic_ms_per_token']:.3f}` ms/token",
      f"- component shell target: `{bar['required_component_shell_us']:.3f}` us/layer",
      f"- failed checks: `{failed}`",
      "",
      payload["next_action"],
      "",
      "This is source/component route-control evidence only. It does not claim speed.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--frontier", type=Path, default=DEFAULT_FRONTIER)
  parser.add_argument("--routes", type=Path, default=DEFAULT_ROUTES)
  parser.add_argument("--rejected", type=Path, default=DEFAULT_REJECTED)
  parser.add_argument("--q6-defer-drain", type=Path, default=DEFAULT_SEQ53)
  parser.add_argument("--down-tail-fusion", type=Path, default=DEFAULT_SEQ66)
  parser.add_argument("--nonatomic-component", type=Path, default=DEFAULT_SEQ80)
  parser.add_argument("--nonatomic-decode", type=Path, default=DEFAULT_SEQ82)
  parser.add_argument("--post-linear-preconv", type=Path, default=DEFAULT_SEQ91)
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
  return 0 if payload["required_checks_passed"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
