#!/usr/bin/env python3
"""Select the next component source gate for attention-front handoff matvec.

This is route-control/design evidence only. It consumes seq179 and the
existing output-projection component board, then authorizes a materially new
component-first source gate. It does not launch a token row and does not claim
speed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = (
    "intel-qwen36-attention-front-handoff-matvec-kernel-algorithm-"
    "component-gate-v0"
)

DEFAULT_FRONTIER = ACTIVE / "frontier.json"
DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_REJECTED = ACTIVE / "rejected-routes.json"
DEFAULT_ACCEPTED = ACTIVE / "accepted-cuts.json"
DEFAULT_SEQ179 = (
    ROOT
    / "output/post-attention-front-handoff-matvec-submit-split-route-gate-20260708Tseq179Z"
    / "metrics.json"
)
DEFAULT_LOCAL64 = (
    ROOT
    / "output/gpu-q4x8-output-projection-local64-20260702T173315Z"
    / "probe-result.json"
)
DEFAULT_F32INPUT = (
    ROOT
    / "output/gpu-q4x8-output-projection-probe-20260706T072447Z"
    / "probe-result.json"
)
DEFAULT_ENGINE = ROOT / "engine/src/gpu_q4x8_matvec.cpp"
DEFAULT_OPENCL = ROOT / "engine/gpu/opencl/q4x8_matvec.cl"
DEFAULT_OUT_DIR = (
    ROOT
    / "output/attention-front-handoff-matvec-kernel-algorithm-component-gate-20260708Tseq180Z"
)


def _load_json(path: Path) -> Any:
  return json.loads(path.read_text(encoding="utf-8"))


def _read(path: Path) -> str:
  return path.read_text(encoding="utf-8")


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _rel(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path)


def _dict(value: Any) -> dict[str, Any]:
  return value if isinstance(value, dict) else {}


def _num(value: Any) -> float:
  return float(value) if isinstance(value, (int, float)) else 0.0


def _nested(obj: Any, *keys: str) -> Any:
  cur = obj
  for key in keys:
    if not isinstance(cur, dict):
      return None
    cur = cur.get(key)
  return cur


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


def _rejected_names(rejected: dict[str, Any]) -> set[str]:
  out: set[str] = set()
  for row in rejected.get("rejected", []):
    if isinstance(row, dict) and isinstance(row.get("route"), str):
      out.add(row["route"])
  return out


def _accepted_ids(accepted: dict[str, Any]) -> set[str]:
  out: set[str] = set()
  for row in accepted.get("accepted", []):
    if isinstance(row, dict) and isinstance(row.get("id"), str):
      out.add(row["id"])
  return out


def _frontier_state(frontier: dict[str, Any]) -> dict[str, Any]:
  anchor = _dict(frontier.get("goal_anchor"))
  budget = _dict(frontier.get("goal_budget"))
  per_token = _dict(budget.get("per_token_ms"))
  verdict = _dict(budget.get("verdict"))
  no_progress = _dict(frontier.get("no_progress"))
  noise = _dict(no_progress.get("noise"))
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


def _projection_probe_summary(probe: dict[str, Any]) -> dict[str, Any]:
  timings = _dict(probe.get("timings"))
  return {
      "required_checks_passed": probe.get("required_checks_passed") is True,
      "rows": probe.get("rows"),
      "cols": probe.get("cols"),
      "blocks_per_row": probe.get("blocks_per_row"),
      "rowlane_min_us": _num(
          timings.get("output_projection_gpu_kernel_min_us")),
      "group8_min_us": _num(
          timings.get("group8_output_projection_gpu_kernel_min_us")),
      "rowlane_effective_packed_gb_s": _num(
          timings.get("output_projection_gpu_effective_packed_gb_s")),
      "group8_effective_packed_gb_s": _num(
          timings.get("group8_output_projection_gpu_effective_packed_gb_s")),
      "gpu_vs_oracle_max_abs": _num(_nested(
          probe, "comparisons", "linear_attn_out", "gpu_vs_oracle",
          "max_abs_diff")),
      "gpu_vs_oracle_cosine": _num(_nested(
          probe, "comparisons", "linear_attn_out", "gpu_vs_oracle",
          "cosine")),
  }


def compute(args: argparse.Namespace) -> dict[str, Any]:
  frontier = _load_json(args.frontier)
  routes = _load_json(args.routes)
  rejected = _load_json(args.rejected)
  accepted = _load_json(args.accepted)
  seq179 = _load_json(args.seq179)
  local64_probe = _load_json(args.local64_probe)
  f32input_probe = _load_json(args.f32input_probe)
  engine_source = _read(args.engine_source)
  opencl_source = _read(args.opencl_source)

  frontier_state = _frontier_state(frontier)
  target = _dict(_nested(seq179, "profile", "component_target"))
  rejected_names = _rejected_names(rejected)
  accepted_ids = _accepted_ids(accepted)
  local64 = _projection_probe_summary(local64_probe)
  f32input = _projection_probe_summary(f32input_probe)

  closed_required = {
      "current_full_core_attention_front_kernel_algorithm_board",
      "gpu_attention_front_resident_residual_input_noqueue",
      "gpu_attention_q4_matvec_residual_fuse_noqueue",
      "gpu_attention_residual_rmsnorm_fusion",
      "gpu_attention_linear_simple_final_output_handoffs",
      "gpu_linear_final_f32input_output_projection_component",
      "gpu_q4_bpr16_output_projection_localq8",
      "gpu_fullcore_attention_output_q4_cpu_order_all_after_tailhandle",
  }
  missing_closed = sorted(closed_required - rejected_names)
  proposed_route = "gpu_q4_bpr16_output_projection_rowblock16_component"
  rowblock_markers = [
      "q4k_x8_matvec_rowblock16_reduce",
      "RunQ4X8Rowblock16",
      "IQ36_ATTENTION_FRONT_OUTPUT_PROJECTION_ROWBLOCK16",
  ]
  rowblock_present = [marker for marker in rowblock_markers
                      if marker in engine_source or marker in opencl_source]

  current_us = _num(target.get("kernel_wait_current_us_per_call"))
  target_us = _num(target.get("target_us_per_call"))
  required_cut = _num(target.get("required_cut_us_per_call"))
  required_ratio = _num(target.get("required_component_speedup_ratio"))
  floor_gap = frontier_state["floor_gap_ms_per_token"]

  design = {
      "candidate_id": proposed_route,
      "kernel_name": "q4k_x8_matvec_rowblock16_reduce",
      "shape_guard": {
          "rows": 2048,
          "cols": 4096,
          "blocks_per_row": 16,
          "row_groups": 256,
      },
      "work_distribution": (
          "one OpenCL workgroup per output row, 16 local work-items per row, "
          "one Q4_K block per work-item, local float reduction to one output"
      ),
      "why_materially_new": [
          "it parallelizes the 16-block inner loop instead of serial rowlane",
          "it is not the closed BPR16 local-Q8 cache variant",
          "it keeps the accepted Q8_K input and Q4_Kx8 packed layout",
          "it is not group8 serial, CPU-order, residual fusion, or f32 input",
      ],
      "component_acceptance": {
          "must_match_current_q8_rowlane": True,
          "max_abs_diff_lte": 5e-3,
          "rmse_lte": 1e-3,
          "cosine_gte": 0.99999,
          "current_us_per_call": current_us,
          "target_us_per_call": target_us,
          "required_cut_us_per_call": required_cut,
          "required_speedup_ratio": required_ratio,
      },
  }

  checks = [
      {
          "name": "seq179_selected_this_component_gate",
          "pass": (
              seq179.get("required_checks_passed") is True
              and seq179.get("selected_next_route")
              == "attention_front_handoff_matvec_kernel_algorithm_component_gate"
              and _has_candidate(
                  routes,
                  179,
                  "select_attention_front_handoff_matvec_kernel_algorithm_component_gate",
              )
              and _has_switch(
                  routes,
                  "select_attention_front_handoff_matvec_kernel_algorithm_component_gate",
                  179,
              )
          ),
          "detail": {
              "seq179_disposition": seq179.get("disposition"),
              "seq179_selected_next_route": seq179.get("selected_next_route"),
          },
      },
      {
          "name": "component_target_covers_floor_gap",
          "pass": (
              floor_gap > 0.0
              and current_us > target_us > 0.0
              and required_cut > 0.0
              and required_ratio >= 1.0
          ),
          "detail": {
              "frontier_floor_gap_ms_per_token": floor_gap,
              "component_target": target,
          },
      },
      {
          "name": "accepted_local64_output_projection_is_current_board",
          "pass": (
              "small_q4_output_projection_local64" in accepted_ids
              and local64["required_checks_passed"]
              and local64["rows"] == 2048
              and local64["cols"] == 4096
              and local64["blocks_per_row"] == 16
              and "kSmallQ4RowlaneLocalSize = 64" in engine_source
              and "blocks_per_row == 16 && global == 2048" in engine_source
          ),
          "detail": local64,
      },
      {
          "name": "existing_variants_do_not_meet_component_target",
          "pass": (
              local64["rowlane_min_us"] > target_us
              and local64["group8_min_us"] > local64["rowlane_min_us"]
              and f32input_probe.get("required_checks_passed") is False
          ),
          "detail": {
              "local64_probe": local64,
              "f32input_probe": f32input,
              "f32input_checks": f32input_probe.get("checks"),
          },
      },
      {
          "name": "closed_output_projection_variants_recorded",
          "pass": not missing_closed,
          "detail": {"missing": missing_closed},
      },
      {
          "name": "rowblock16_component_is_not_already_tried_or_present",
          "pass": proposed_route not in rejected_names and not rowblock_present,
          "detail": {
              "proposed_route": proposed_route,
              "present_markers": rowblock_present,
          },
      },
      {
          "name": "source_gate_only_no_decode_or_speed_claim",
          "pass": (
              seq179.get("decode_probe_allowed") is False
              and seq179.get("speedup_claims_allowed") is False
              and frontier_state["current_best_tps"] < frontier_state["floor_tps"]
          ),
          "detail": frontier_state,
      },
  ]
  required_checks_passed = all(row["pass"] for row in checks)
  selected_next_route = (
      "attention_front_handoff_matvec_rowblock16_component_source_gate"
  )
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "inputs": {
          "frontier": _rel(args.frontier),
          "routes": _rel(args.routes),
          "rejected": _rel(args.rejected),
          "accepted": _rel(args.accepted),
          "seq179_route_gate": _rel(args.seq179),
          "local64_probe": _rel(args.local64_probe),
          "f32input_probe": _rel(args.f32input_probe),
          "engine_source": _rel(args.engine_source),
          "engine_source_sha256": _sha256(args.engine_source),
          "opencl_source": _rel(args.opencl_source),
          "opencl_source_sha256": _sha256(args.opencl_source),
      },
      "checks": checks,
      "frontier": frontier_state,
      "component_design": design,
      "existing_component_evidence": {
          "local64_probe": local64,
          "f32input_probe": f32input,
          "closed_variants": sorted(closed_required),
      },
      "required_checks_passed": required_checks_passed,
      "disposition": (
          "select_attention_front_handoff_matvec_rowblock16_component_source_gate"
      ),
      "selected_next_route": selected_next_route,
      "component_source_allowed": required_checks_passed,
      "component_probe_allowed": False,
      "decode_probe_allowed": False,
      "speedup_claims_allowed": False,
      "next_route_reason": (
          "Existing output-projection shapes do not meet the seq179 component "
          "target, and the closed local-Q8/f32/CPU-order/group8 variants should "
          "not be repeated. The admissible next unit is source-only wiring for "
          "a rowblock16 component proof: one workgroup per output row with 16 "
          "block-parallel work-items and a local reduction. It must beat "
          f"{current_us:.3f} -> {target_us:.3f} us/call before any decode row."
      ),
  }


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--frontier", type=Path, default=DEFAULT_FRONTIER)
  parser.add_argument("--routes", type=Path, default=DEFAULT_ROUTES)
  parser.add_argument("--rejected", type=Path, default=DEFAULT_REJECTED)
  parser.add_argument("--accepted", type=Path, default=DEFAULT_ACCEPTED)
  parser.add_argument("--seq179", type=Path, default=DEFAULT_SEQ179)
  parser.add_argument("--local64-probe", type=Path, default=DEFAULT_LOCAL64)
  parser.add_argument("--f32input-probe", type=Path, default=DEFAULT_F32INPUT)
  parser.add_argument("--engine-source", type=Path, default=DEFAULT_ENGINE)
  parser.add_argument("--opencl-source", type=Path, default=DEFAULT_OPENCL)
  parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
  return parser.parse_args()


def main() -> int:
  args = parse_args()
  result = compute(args)
  args.out_dir.mkdir(parents=True, exist_ok=True)
  (args.out_dir / "metrics.json").write_text(
      json.dumps(result, indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
  )
  print(json.dumps({
      "required_checks_passed": result["required_checks_passed"],
      "disposition": result["disposition"],
      "selected_next_route": result["selected_next_route"],
      "out_dir": _rel(args.out_dir),
      "component_source_allowed": result["component_source_allowed"],
  }, sort_keys=True))
  return 0 if result["required_checks_passed"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
