#!/usr/bin/env python3
"""Select the next route after the Q6 rowgroup down-tail component closure.

This is route-control evidence only. It reads the live frontier plus the route
boards and chooses the next non-rejected unit without launching a decode row.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = "intel-qwen36-post-rowgroup-route-gate-v0"
DEFAULT_FRONTIER = ACTIVE / "frontier.json"
DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_REJECTED = ACTIVE / "rejected-routes.json"
DEFAULT_ACCEPTED = ACTIVE / "accepted-cuts.json"
DEFAULT_SEQ93 = ROOT / "output/q6-rowgroup-down-tail-gate-20260707Tseq93Z/metrics.json"
DEFAULT_RESIDENT_HEADER = ROOT / "engine/include/intel_qwen36/resident_harness.hpp"
DEFAULT_RESIDENT_SOURCE = ROOT / "engine/src/resident_harness.cpp"
DEFAULT_DECODE_SMOKE = ROOT / "tools/intel-qwen36-r2-gpu-decode-smoke.py"
DEFAULT_OUT_DIR = ROOT / "output/post-rowgroup-route-gate-20260707Tseq94Z"


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


def _route_names(rejected: dict[str, Any]) -> set[str]:
  names: set[str] = set()
  for row in rejected.get("rejected", []):
    if isinstance(row, dict) and isinstance(row.get("route"), str):
      names.add(row["route"])
  return names


def _accepted_ids(accepted: dict[str, Any]) -> set[str]:
  ids: set[str] = set()
  rows = accepted.get("accepted")
  rows = rows if isinstance(rows, list) else accepted.get("cuts", [])
  for row in rows:
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


def _parked_route(routes: dict[str, Any], route_id: str) -> dict[str, Any] | None:
  for row in routes.get("parked_routes", []):
    if isinstance(row, dict) and row.get("id") == route_id:
      return row
  return None


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
  floor_gap = max(0.0, wall - floor_budget)
  overhead = _num(per_token.get("non_kernel_overhead"))
  return {
      "current_best_tps": _num(_nested(no_progress, "last_significant_improvement", "tps")),
      "floor_tps": _num(verdict.get("floor_tps")),
      "wall_ms_per_token": wall,
      "floor_budget_ms_per_token": floor_budget,
      "floor_gap_ms_per_token": floor_gap,
      "gpu_kernel_busy_floor_ms_per_token": _num(per_token.get("gpu_kernel_busy_floor")),
      "non_kernel_overhead_ms_per_token": overhead,
      "overhead_cut_fraction_needed": floor_gap / overhead if overhead > 0.0 else 0.0,
      "overhead_only_ceiling_tok_s": _num(verdict.get("overhead_only_ceiling_tok_s")),
      "can_reach_floor_without_kernel_work": bool(
          verdict.get("can_reach_floor_without_kernel_work")),
      "noise_rel": _num(noise.get("rel")),
      "runs_since_significant_improvement": no_progress.get(
          "runs_since_significant_improvement"),
      "soft_reflection_breached": no_progress.get("soft_reflection_breached"),
      "hard_stall_breached": no_progress.get("hard_stall_breached"),
      "glide_slope_projected_runs_to_floor": _nested(
          no_progress, "glide_slope", "projected_runs_to_floor"),
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


def _marker(text: str, pattern: str) -> bool:
  return re.search(pattern, text, flags=re.MULTILINE | re.DOTALL) is not None


def compute(args: argparse.Namespace) -> dict[str, Any]:
  frontier = _load_json(args.frontier)
  routes = _load_json(args.routes)
  rejected = _load_json(args.rejected)
  accepted = _load_json(args.accepted)
  seq93 = _load_json(args.seq93)
  resident_header = args.resident_header.read_text(encoding="utf-8")
  resident_source = args.resident_source.read_text(encoding="utf-8")
  decode_smoke = args.decode_smoke.read_text(encoding="utf-8")

  frontier_state = _frontier_summary(frontier)
  stage_gaps = _stage_gaps(frontier)
  substage_gaps = _substage_gaps(frontier)
  rejected_names = _route_names(rejected)
  accepted = _accepted_ids(accepted)

  down_tail_closed = [
      "gpu_q6_defer_finish_without_tail_drain_elimination",
      "gpu_ffn_tail_resident_input_noqueue",
      "gpu_ffn_tail_plus_attention_residual_carrier_noqueue",
      "gpu_down_tail_hidden_row_serial_fusion",
      "gpu_ffn_tail_atomic_reduction_from_down_handles",
      "gpu_direct_q6_down_tail_atomic_from_q6_workitems",
      "gpu_q6_nonatomic_down_tail_decode_fusion",
      "gpu_selected_shared_q6_down_tail_rowgroup_local_reduce_component",
  ]
  simple_boundary_closed = [
      "gpu_attention_linear_simple_final_output_handoffs",
      "gpu_attention_linear_event_lifetime_combined_alias",
      "gpu_linear_preconv_shared_q8_preconv_bundle_decode",
      "gpu_linear_preconv_shared_q8_qkv_conv_root_component",
      "gpu_linear_delta_read_drain_noqueue",
      "gpu_fullcore_apply_score_local64_noqueue",
      "gpu_attention_residual_rmsnorm_fusion",
  ]
  accepted_resident_ids = [
      "r2_gpu_decode_resident_process_multisession_reuse",
      "r2_gpu_decode_resident_device_state_handle_bank",
      "r2_gpu_decode_resident_linear_q6_qkv_weight_store",
      "r2_gpu_decode_resident_linear_conv_weight_store",
      "r2_gpu_decode_resident_q4_cpu_order_weight_store",
      "r2_gpu_decode_resident_selected_q6_weight_store",
      "r2_gpu_decode_resident_lm_head_weight_store",
  ]
  missing_down_tail = [name for name in down_tail_closed if name not in rejected_names]
  missing_simple_boundary = [
      name for name in simple_boundary_closed if name not in rejected_names
  ]
  missing_accepted_resident = [
      name for name in accepted_resident_ids if name not in accepted
  ]

  header_and_source = resident_header + "\n" + resident_source
  source_markers = [
      {
          "name": "resident_decode_loop_engine_class_present",
          "pass": _marker(resident_header, r"class\s+ResidentDecodeLoop\b"),
      },
      {
          "name": "resident_decode_loop_engine_run_present",
          "pass": _marker(resident_source, r"ResidentDecodeLoopResult\s+ResidentDecodeLoop::run"),
      },
      {
          "name": "decode_smoke_uses_resident_decode_loop_callback",
          "pass": _marker(
              decode_smoke,
              r"resident_decode_loop\.run\(\s*std::cout,\s*resident_loop_config,\s*\[&\]",
          ),
      },
      {
          "name": "decode_smoke_still_generates_target_cpp",
          "pass": "\"generated_cpp\"" in decode_smoke
          and "generated-source" in decode_smoke,
      },
      {
          "name": "engine_hot_gpu_decode_loop_not_yet_extracted",
          "pass": not _marker(
              header_and_source,
              r"(ResidentGpuDecodeLoop|ResidentHotDecodeLoop|RunResidentGpuDecodeToken)",
          ),
      },
  ]

  resident_parked = _parked_route(routes, "resident_decode_loop_streaming")
  floor_gap = frontier_state["floor_gap_ms_per_token"]
  overhead_fraction = frontier_state["overhead_cut_fraction_needed"]
  profile_gap_sum = (
      stage_gaps.get("full_core", 0.0)
      + stage_gaps.get("linear_preconv", 0.0)
      + stage_gaps.get("attention_front", 0.0)
      + stage_gaps.get("linear_delta", 0.0)
  )
  selected_gap = stage_gaps.get("selected_ffn", 0.0)
  linear_alpha_postconv_gap = (
      substage_gaps.get("linear_preconv.alpha_beta", 0.0)
      + substage_gaps.get("linear_preconv.postconv_prep", 0.0)
  )

  checks = [
      {
          "name": "frontier_still_below_floor",
          "pass": frontier_state["wall_ms_per_token"]
          > frontier_state["floor_budget_ms_per_token"]
          > 0.0,
      },
      {
          "name": "overhead_arithmetic_can_cover_floor",
          "pass": (
              frontier_state["can_reach_floor_without_kernel_work"]
              and frontier_state["overhead_only_ceiling_tok_s"]
              >= frontier_state["floor_tps"]
              and frontier_state["non_kernel_overhead_ms_per_token"] > floor_gap
              and 0.0 < overhead_fraction <= 0.05
          ),
          "detail": {
              "floor_gap_ms_per_token": floor_gap,
              "non_kernel_overhead_ms_per_token": frontier_state[
                  "non_kernel_overhead_ms_per_token"],
              "overhead_cut_fraction_needed": overhead_fraction,
          },
      },
      {
          "name": "soft_reflection_requires_direction_not_more_microcuts",
          "pass": frontier_state["soft_reflection_breached"] is True
          and frontier_state["hard_stall_breached"] is False,
      },
      {
          "name": "seq93_rowgroup_component_closed",
          "pass": (
              bool(seq93.get("component_exact"))
              and not bool(seq93.get("target_cleared"))
              and not bool(seq93.get("decode_probe_allowed"))
              and _num(seq93.get("rowgroup_min_us")) > _num(seq93.get("target_us"))
              and _num(seq93.get("ratio_vs_target")) > 1.0
          ),
          "detail": {
              "rowgroup_min_us": _num(seq93.get("rowgroup_min_us")),
              "target_us": _num(seq93.get("target_us")),
              "ratio_vs_target": _num(seq93.get("ratio_vs_target")),
          },
      },
      {
          "name": "seq93_switch_recorded",
          "pass": _has_switch(
              routes, "close_selected_shared_q6_down_tail_rowgroup_reduce_component", 93
          ),
      },
      {
          "name": "seq93_candidate_recorded",
          "pass": _has_candidate(
              routes, 93, "rejected_rowgroup_local_reduce_component_speed"
          ),
      },
      {
          "name": "down_tail_family_closures_recorded",
          "pass": not missing_down_tail,
          "detail": {"missing": missing_down_tail},
      },
      {
          "name": "simple_attention_linear_boundaries_closed",
          "pass": not missing_simple_boundary,
          "detail": {"missing": missing_simple_boundary},
      },
      {
          "name": "resident_decode_loop_parked_route_available",
          "pass": isinstance(resident_parked, dict)
          and resident_parked.get("rank") == 2,
      },
      {
          "name": "accepted_resident_ownership_prereqs_present",
          "pass": not missing_accepted_resident,
          "detail": {"missing": missing_accepted_resident},
      },
      {
          "name": "resident_loop_source_shape_selects_extraction_not_new_probe",
          "pass": all(item["pass"] for item in source_markers),
          "detail": {"markers": source_markers},
      },
      {
          "name": "selected_next_route_not_rejected",
          "pass": "gpu_resident_decode_loop_overhead_root_source_gate"
          not in rejected_names,
      },
  ]
  required = all(check["pass"] for check in checks)

  route_candidates = [
      {
          "route": "resident_decode_loop_overhead_root_source_gate",
          "admissible": required,
          "score_ms_per_token": frontier_state["non_kernel_overhead_ms_per_token"],
          "reason": (
              "The floor miss is 0.45 ms/token and requires only "
              f"{overhead_fraction:.3%} of the measured non-kernel overhead. "
              "Resident ownership prerequisites are present, but the accepted "
              "speed lane is still hosted by generated decode-smoke callback "
              "code instead of an extracted engine hot loop."
          ),
      },
      {
          "route": "selected_ffn_down_tail_continuation",
          "admissible": False,
          "score_ms_per_token": selected_gap,
          "reason": (
              "Selected-FFN remains the largest gap, but rowgroup-local Q6 "
              "down-to-tail missed the component target by "
              f"{_num(seq93.get('miss_vs_target_us')):.3f} us/layer and the "
              "down-tail family closures are recorded."
          ),
      },
      {
          "route": "linear_preconv_alpha_beta_postconv_component_gate",
          "admissible": False,
          "score_ms_per_token": linear_alpha_postconv_gap,
          "reason": (
              "Linear alpha/beta plus postconv gaps can cover the floor in "
              "arithmetic, but the accepted fused CPU-order alpha/beta/z path, "
              "shared-Q8 preconv, and isolated postconv/readback routes already "
              "closed the obvious carrier variants. Reopen only with a concrete "
              "new kernel algorithm, not a route selection default."
          ),
      },
      {
          "route": "attention_fullcore_linear_boundary_cleanup",
          "admissible": False,
          "score_ms_per_token": profile_gap_sum,
          "reason": (
              "The profile-gap sum is large, but simple device-Q8 handoff, "
              "combined attention/linear alias, shared-Q8 preconv, and "
              "linear-delta read-as-drain are closed. The remaining admissible "
              "shape is the whole resident loop overhead/root gate."
          ),
      },
  ]

  selected_next_route = (
      "resident_decode_loop_overhead_root_source_gate"
      if required
      else "route_selection_needs_manual_review"
  )
  disposition = (
      "switch_to_resident_decode_loop_overhead_root_source_gate"
      if required
      else "post_rowgroup_route_gate_failed"
  )
  next_action = (
      "Run a source/profile gate for the resident decode-loop overhead root. "
      "The gate must move the hot token loop out of the generated decode-smoke "
      "callback path into an engine-owned resident loop using the accepted "
      "resident state and weight stores, then prove at least 0.45 ms/token "
      "expected movement or a measured source-profile reduction before any "
      "decode candidate. The first token-emitting run after that gate is "
      "`decode-smoke --explore`; promotion still needs a confirm plus paired "
      "teacher-forced distribution evidence outside the 0.50% noise band."
      if required
      else "Fix failed route-gate checks before launching another target probe."
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
          "accepted": _rel(args.accepted),
          "seq93": _rel(args.seq93),
          "resident_header": _rel(args.resident_header),
          "resident_source": _rel(args.resident_source),
          "decode_smoke": _rel(args.decode_smoke),
      },
      "frontier": frontier_state,
      "stage_gap_ms_per_token": stage_gaps,
      "substage_gap_ms_per_token": substage_gaps,
      "route_candidates": route_candidates,
      "closure_requirements": {
          "down_tail_required": down_tail_closed,
          "down_tail_missing": missing_down_tail,
          "simple_boundary_required": simple_boundary_closed,
          "simple_boundary_missing": missing_simple_boundary,
          "accepted_resident_required": accepted_resident_ids,
          "accepted_resident_missing": missing_accepted_resident,
      },
      "seq93_rowgroup": {
          "component_exact": bool(seq93.get("component_exact")),
          "decode_probe_allowed": bool(seq93.get("decode_probe_allowed")),
          "rowgroup_min_us": _num(seq93.get("rowgroup_min_us")),
          "target_us": _num(seq93.get("target_us")),
          "miss_vs_target_us": _num(seq93.get("miss_vs_target_us")),
          "ratio_vs_target": _num(seq93.get("ratio_vs_target")),
          "ratio_vs_nonatomic": _num(seq93.get("ratio_vs_nonatomic")),
      },
      "source_markers": source_markers,
      "checks": checks,
  }


def write_outputs(out_dir: Path, payload: dict[str, Any]) -> None:
  out_dir.mkdir(parents=True, exist_ok=False)
  (out_dir / "metrics.json").write_text(
      json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  manifest = {
      "schema_version": payload["schema_version"],
      "workstream": payload["workstream"],
      "tool": "tools/intel-qwen36-post-rowgroup-route-gate.py",
      "inputs": payload["inputs"],
      "selected_next_route": payload["selected_next_route"],
      "speedup_claims_allowed": False,
  }
  (out_dir / "manifest.json").write_text(
      json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  failed = [check["name"] for check in payload["checks"] if not check["pass"]]
  frontier = payload["frontier"]
  rowgroup = payload["seq93_rowgroup"]
  lines = [
      "# Post Rowgroup Route Gate",
      "",
      f"- required checks passed: `{str(payload['required_checks_passed']).lower()}`",
      f"- disposition: `{payload['disposition']}`",
      f"- selected next route: `{payload['selected_next_route']}`",
      f"- frontier gap: `{frontier['floor_gap_ms_per_token']:.3f}` ms/token",
      f"- non-kernel overhead: `{frontier['non_kernel_overhead_ms_per_token']:.3f}` ms/token",
      f"- overhead fraction needed: `{frontier['overhead_cut_fraction_needed']:.3%}`",
      f"- rowgroup miss: `{rowgroup['miss_vs_target_us']:.3f}` us/layer",
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
  parser.add_argument("--seq93", type=Path, default=DEFAULT_SEQ93)
  parser.add_argument("--resident-header", type=Path, default=DEFAULT_RESIDENT_HEADER)
  parser.add_argument("--resident-source", type=Path, default=DEFAULT_RESIDENT_SOURCE)
  parser.add_argument("--decode-smoke", type=Path, default=DEFAULT_DECODE_SMOKE)
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
