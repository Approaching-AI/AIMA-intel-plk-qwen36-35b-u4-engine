#!/usr/bin/env python3
"""Select the next selected-FFN material/layout component proof.

This is source-contract and route-control evidence only. It narrows the
post-seq86 selected-FFN route to a legal batch-1 material layout proof that
does not repeat live selected-set concat, all-expert residency, local-size,
group8, down-tail, or DPAS local-Q8 variants.
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
SCHEMA_VERSION = "intel-qwen36-selected-ffn-material-layout-gate-v0"
DEFAULT_FRONTIER = ACTIVE / "frontier.json"
DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_REJECTED = ACTIVE / "rejected-routes.json"
DEFAULT_BEST = (
    ROOT
    / "output/r2-gpu-selected-shared-q4q6-down-cold-q6-experts-noqueue-speed-20260705T143006Z/result.json"
)
DEFAULT_PROFILE_LOG = ROOT / "output/explore-log.jsonl"
DEFAULT_Q4_FULL = ROOT / "output/gpu-q4x8-qmatvec-ffn-gateup-full-20260702T225500Z/probe-result.json"
DEFAULT_Q4_OCCUPANCY = (
    ROOT
    / "output/gpu-q4x8-selected-shared-gate-up-occupancy4-probe-20260705T203448Z/probe-result.json"
)
DEFAULT_Q6_OCCUPANCY = (
    ROOT
    / "output/gpu-selected-down-q6-selected-shared-occupancy4-probe-20260705T220711Z/probe-result.json"
)
DEFAULT_DECODE = ROOT / "tools/intel-qwen36-r2-gpu-decode-smoke.py"
DEFAULT_ENGINE = ROOT / "engine/src/gpu_q4x8_matvec.cpp"
DEFAULT_HEADER = ROOT / "engine/include/intel_qwen36/gpu_q4x8_matvec.hpp"
DEFAULT_OPENCL = ROOT / "engine/gpu/opencl/q4x8_matvec.cl"
DEFAULT_OUT_DIR = ROOT / "output/selected-ffn-material-layout-gate-20260707Tseq87Z"
PROFILE_LABEL = "selected-shared-q4q6-down-cold-q6-experts-profile"


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
      "current_best_tps": _num(
          (no_progress.get("last_significant_improvement") or {}).get("tps")
          if isinstance(no_progress.get("last_significant_improvement"), dict)
          else None
      ),
      "floor_tps": _num(verdict.get("floor_tps")),
      "wall_ms_per_token": wall,
      "floor_budget_ms_per_token": floor_budget,
      "floor_gap_ms_per_token": max(0.0, wall - floor_budget),
      "runs_since_significant_improvement": no_progress.get(
          "runs_since_significant_improvement"
      ),
  }


def _stage_gaps(frontier: dict[str, Any]) -> dict[str, float]:
  budget = frontier.get("goal_budget")
  budget = budget if isinstance(budget, dict) else {}
  rows = budget.get("stage_kernel_gap_estimates_ms_per_token")
  out: dict[str, float] = {}
  if isinstance(rows, list):
    for row in rows:
      if isinstance(row, dict) and isinstance(row.get("stage"), str):
        out[row["stage"]] = _num(row.get("gap_ms_per_token"))
  return out


def _substage_gaps(frontier: dict[str, Any]) -> dict[str, float]:
  budget = frontier.get("goal_budget")
  budget = budget if isinstance(budget, dict) else {}
  rows = budget.get("substage_gap_estimates_ms_per_token")
  out: dict[str, float] = {}
  if isinstance(rows, list):
    for row in rows:
      if (
          isinstance(row, dict)
          and row.get("stage") == "selected_ffn"
          and isinstance(row.get("substage"), str)
      ):
        out[row["substage"]] = _num(row.get("gap_ms_per_token"))
  return out


def _profile_row(path: Path, label: str) -> dict[str, Any]:
  for line in path.read_text(encoding="utf-8").splitlines():
    if not line.strip():
      continue
    row = json.loads(line)
    if isinstance(row, dict) and row.get("label") == label:
      return row
  raise SystemExit(f"{path}: missing explore label {label}")


def _has_switch(routes: dict[str, Any], decision: str, seq_covered: int) -> bool:
  rows = routes.get("switch_decisions")
  if not isinstance(rows, list):
    return False
  for row in rows:
    if (
        isinstance(row, dict)
        and row.get("decision") == decision
        and _num(row.get("seq_covered")) >= seq_covered
        and row.get("resolved") is True
    ):
      return True
  return False


def _rejected_route(rejected: dict[str, Any], route: str) -> dict[str, Any] | None:
  rows = rejected.get("rejected")
  if not isinstance(rows, list):
    return None
  for row in rows:
    if isinstance(row, dict) and row.get("route") == route:
      return row
  return None


def _has_markers(text: str, markers: list[str]) -> dict[str, Any]:
  missing = [marker for marker in markers if marker not in text]
  return {"pass": not missing, "missing": missing, "marker_count": len(markers)}


def _extract_before(reason: str, suffix: str) -> float:
  match = re.search(r"([0-9]+(?:\.[0-9]+)?)GB\s+" + suffix, reason)
  return float(match.group(1)) if match else 0.0


def compute(args: argparse.Namespace) -> dict[str, Any]:
  frontier = _load_json(args.frontier)
  routes = _load_json(args.routes)
  rejected = _load_json(args.rejected)
  best = _load_json(args.best)
  profile = _profile_row(args.profile_log, PROFILE_LABEL)
  q4_full = _load_json(args.q4_full)
  q4_occupancy = _load_json(args.q4_occupancy)
  q6_occupancy = _load_json(args.q6_occupancy)
  source = "\n".join([
      _read(args.decode_source),
      _read(args.engine_source),
      _read(args.engine_header),
      _read(args.opencl_source),
  ])

  frontier_state = _frontier_summary(frontier)
  stage_gaps = _stage_gaps(frontier)
  substage_gaps = _substage_gaps(frontier)
  largest_gap = max(stage_gaps.values()) if stage_gaps else 0.0
  largest_stage = next(
      (stage for stage, gap in stage_gaps.items() if gap == largest_gap), ""
  )

  smoke = best.get("smoke") if isinstance(best.get("smoke"), dict) else best
  gateup_kernel_ms_per_token = (
      _num(profile.get("selected_shared_q4_gateup_combined_kernel_us")) / 1000.0 / 8.0
  )
  q6_down_kernel_ms_per_token = (
      _num(profile.get("selected_shared_q6_down_combined_kernel_us")) / 1000.0 / 8.0
  )
  floor_gap = frontier_state["floor_gap_ms_per_token"]
  gateup_required_ratio = (
      gateup_kernel_ms_per_token / (gateup_kernel_ms_per_token - floor_gap)
      if gateup_kernel_ms_per_token > floor_gap > 0.0
      else 0.0
  )
  q6_required_ratio = (
      q6_down_kernel_ms_per_token / (q6_down_kernel_ms_per_token - floor_gap)
      if q6_down_kernel_ms_per_token > floor_gap > 0.0
      else 0.0
  )

  q4_timing = q4_occupancy.get("timings")
  q4_timing = q4_timing if isinstance(q4_timing, dict) else {}
  q6_timing = q6_occupancy.get("timings")
  q6_timing = q6_timing if isinstance(q6_timing, dict) else {}
  q4_full_gbs = _num(q4_full.get("gpu_effective_packed_gb_s"))
  q4_current_gbs = _num(q4_timing.get("combined_gate_up_gpu_effective_packed_gb_s"))
  q4_occupancy_ratio = _num(q4_timing.get("occupancy4_scaled_vs_combined_speedup"))
  q6_current_gbs = _num(
      q6_timing.get("candidate_q6_selected_shared_combined_effective_raw_gb_s")
  )
  q6_occupancy_ratio = _num(
      q6_occupancy.get("candidate_q6_selected_shared_occupancy_scaled_speedup_vs_combined")
  )

  all_expert = _rejected_route(rejected, "gpu_selected_all_expert_residency")
  all_expert_reason = str(all_expert.get("reason", "")) if all_expert else ""
  all_expert_q4_gb = _extract_before(all_expert_reason, r"selected gate/up Q4")
  all_expert_q6_gb = _extract_before(all_expert_reason, r"selected Q6")
  if all_expert_q4_gb == 0.0:
    all_expert_q4_gb = 12.08 if "12.08GB selected gate/up Q4" in all_expert_reason else 0.0
  if all_expert_q6_gb == 0.0:
    all_expert_q6_gb = 4.40 if "4.40GB selected Q6" in all_expert_reason else 0.0
  current_selected_gateup_gb = _num(smoke.get("resident_selected_gate_up_uploaded_bytes")) / 1e9
  current_selected_q6_gb = _num(smoke.get("resident_selected_q6_down_uploaded_bytes")) / 1e9

  current_source_markers = _has_markers(
      source,
      [
          "SelectedGateUpExpertHandles(",
          "RunResidentPackedQ4X8Expert8PlusSharedThenSwiGlu(",
          "q4k_x8_matvec_rowlane_expert8_plus_shared_localq8",
          "resident_selected_gate_up_expert_handles",
          "resident_selected_cache_topk",
      ],
  )
  missing_indexed_markers = _has_markers(
      source,
      [
          "RunResidentPackedQ4X8TopKIndexedExpert8PlusSharedThenSwiGlu",
          "q4k_x8_matvec_topk_indexed_expert8_plus_shared_localq8",
          "selected_gate_up_topk_indexed",
      ],
  )

  closed_routes = {
      "concat_gateup": _rejected_route(
          rejected, "gpu_selected_shared_q4_gateup_combined_concat_decode"
      ),
      "all_expert_residency": all_expert,
      "gateup_group8": _rejected_route(
          rejected, "gpu_selected_shared_q4_gateup_group8_component"
      ),
      "gateup_local128": _rejected_route(
          rejected, "gpu_selected_shared_q4_gateup_no_concat_local128_noqueue"
      ),
      "gateup_fused_swiglu": _rejected_route(
          rejected, "gpu_selected_shared_q4_gateup_fused_swiglu_component"
      ),
      "dpas_localq8": _rejected_route(
          rejected, "native_prefill_dpas_localq8_workgroup_sharing"
      ),
      "q6_down_tail": _rejected_route(
          rejected, "gpu_q6_nonatomic_down_tail_decode_fusion"
      ),
  }
  closed_missing = [
      name for name, row in closed_routes.items() if not isinstance(row, dict)
  ]

  checks = [
      {
          "name": "seq86_selected_ffn_switch_recorded",
          "pass": _has_switch(
              routes,
              "switch_from_attention_linear_to_selected_ffn_material_layout_component_proof",
              86,
          ),
      },
      {
          "name": "selected_ffn_is_largest_gap",
          "pass": largest_stage == "selected_ffn"
          and stage_gaps.get("selected_ffn", 0.0) > floor_gap,
          "detail": {
              "largest_stage": largest_stage,
              "selected_ffn_gap_ms_per_token": stage_gaps.get("selected_ffn", 0.0),
              "floor_gap_ms_per_token": floor_gap,
          },
      },
      {
          "name": "gateup_material_ratio_can_cover_floor",
          "pass": gateup_required_ratio > 1.0
          and q4_occupancy_ratio >= gateup_required_ratio
          and q4_full_gbs > q4_current_gbs > 0.0,
          "detail": {
              "gateup_kernel_ms_per_token": gateup_kernel_ms_per_token,
              "gateup_required_ratio": gateup_required_ratio,
              "q4_occupancy4_scaled_ratio": q4_occupancy_ratio,
              "q4_full_gb_s": q4_full_gbs,
              "q4_current_combined_gb_s": q4_current_gbs,
          },
      },
      {
          "name": "down_only_q6_shape_is_not_the_next_gateup_proof",
          "pass": q6_occupancy_ratio < q6_required_ratio
          or q6_required_ratio <= 1.0,
          "detail": {
              "q6_down_kernel_ms_per_token": q6_down_kernel_ms_per_token,
              "q6_required_ratio": q6_required_ratio,
              "q6_occupancy4_scaled_ratio": q6_occupancy_ratio,
              "q6_current_combined_gb_s": q6_current_gbs,
          },
      },
      {
          "name": "accepted_stack_has_top16_zero_miss_selected_gateup_cache",
          "pass": smoke.get("resident_selected_cache_topk") == 16
          and smoke.get("resident_selected_gate_up_misses") == 0
          and current_selected_gateup_gb > 0.0,
          "detail": {
              "resident_selected_cache_topk": smoke.get("resident_selected_cache_topk"),
              "resident_selected_gate_up_hits": smoke.get("resident_selected_gate_up_hits"),
              "resident_selected_gate_up_misses": smoke.get("resident_selected_gate_up_misses"),
              "resident_selected_gate_up_gb": current_selected_gateup_gb,
              "resident_selected_q6_gb": current_selected_q6_gb,
          },
      },
      {
          "name": "top16_gateup_material_is_not_all_expert_residency",
          "pass": current_selected_gateup_gb > 0.0
          and all_expert_q4_gb > current_selected_gateup_gb
          and isinstance(all_expert, dict),
          "detail": {
              "current_selected_gateup_gb": current_selected_gateup_gb,
              "rejected_all_expert_gateup_q4_gb": all_expert_q4_gb,
              "rejected_all_expert_q6_gb": all_expert_q6_gb,
          },
      },
      {
          "name": "closed_route_board_contains_forbidden_variants",
          "pass": not closed_missing,
          "detail": {"missing": closed_missing},
      },
      {
          "name": "current_source_has_no_concat_expert_handle_gateup",
          "pass": current_source_markers["pass"],
          "detail": current_source_markers,
      },
      {
          "name": "indexed_top16_material_primitive_is_not_yet_wired",
          "pass": not missing_indexed_markers["pass"],
          "detail": missing_indexed_markers,
      },
  ]
  required = all(check["pass"] for check in checks)
  selected_next_route = (
      "selected_gateup_top16_indexed_material_component_proof"
      if required
      else "selected_ffn_material_layout_needs_manual_review"
  )
  next_action = (
      "Implement and run a component proof for a top16-indexed selected gate/up "
      "material layout: one per-layer compact selected-gate/up Q4_Kx8 material "
      "buffer, live top-8 position indices, and the shared gate/up handle feeding "
      "the existing selected+shared SwiGLU output shape. Compare against the "
      "accepted no-concat path and oracle; require at least the floor-covering "
      "component ratio, and do not use live selected-set concat, all-expert Q4 "
      "residency, group8, local-size, row-pair fused SwiGLU, or live-prewarm rows."
      if required
      else "Review the selected-FFN material-layout evidence before target work."
  )

  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "required_checks_passed": required,
      "speedup_claims_allowed": False,
      "disposition": (
          "selected_gateup_top16_indexed_material_component_contract_ready"
          if required
          else "selected_ffn_material_layout_contract_incomplete"
      ),
      "selected_next_route": selected_next_route,
      "next_action": next_action,
      "inputs": {
          "frontier": _rel(args.frontier),
          "routes": _rel(args.routes),
          "rejected": _rel(args.rejected),
          "best": _rel(args.best),
          "profile_log": f"{_rel(args.profile_log)}#{PROFILE_LABEL}",
          "q4_full": _rel(args.q4_full),
          "q4_occupancy": _rel(args.q4_occupancy),
          "q6_occupancy": _rel(args.q6_occupancy),
          "decode_source": _rel(args.decode_source),
          "engine_source": _rel(args.engine_source),
          "engine_header": _rel(args.engine_header),
          "opencl_source": _rel(args.opencl_source),
      },
      "frontier": frontier_state,
      "stage_gap_ms_per_token": stage_gaps,
      "selected_substage_gap_ms_per_token": substage_gaps,
      "derived": {
          "gateup_kernel_ms_per_token": gateup_kernel_ms_per_token,
          "gateup_required_speedup_to_cover_floor": gateup_required_ratio,
          "q4_gateup_occupancy4_scaled_ratio": q4_occupancy_ratio,
          "q4_full_tensor_gb_s": q4_full_gbs,
          "q4_current_combined_gb_s": q4_current_gbs,
          "q6_down_kernel_ms_per_token": q6_down_kernel_ms_per_token,
          "q6_down_required_speedup_to_cover_floor": q6_required_ratio,
          "q6_down_occupancy4_scaled_ratio": q6_occupancy_ratio,
          "q6_current_combined_gb_s": q6_current_gbs,
          "current_selected_gateup_gb": current_selected_gateup_gb,
          "current_selected_q6_gb": current_selected_q6_gb,
          "rejected_all_expert_gateup_q4_gb": all_expert_q4_gb,
          "rejected_all_expert_q6_gb": all_expert_q6_gb,
      },
      "checks": checks,
  }


def write_outputs(out_dir: Path, payload: dict[str, Any]) -> None:
  out_dir.mkdir(parents=True, exist_ok=False)
  (out_dir / "metrics.json").write_text(
      json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  failed = [check["name"] for check in payload["checks"] if not check["pass"]]
  derived = payload["derived"]
  frontier = payload["frontier"]
  lines = [
      "# Selected-FFN Material Layout Gate",
      "",
      f"- required checks passed: `{str(payload['required_checks_passed']).lower()}`",
      f"- disposition: `{payload['disposition']}`",
      f"- selected next route: `{payload['selected_next_route']}`",
      f"- frontier gap: `{frontier['floor_gap_ms_per_token']:.3f}` ms/token",
      f"- gate/up required ratio: `{derived['gateup_required_speedup_to_cover_floor']:.3f}`",
      f"- Q4 occupancy4 ratio: `{derived['q4_gateup_occupancy4_scaled_ratio']:.3f}`",
      f"- current selected gate/up material: `{derived['current_selected_gateup_gb']:.3f}` GB",
      f"- rejected all-expert gate/up Q4: `{derived['rejected_all_expert_gateup_q4_gb']:.3f}` GB",
      f"- failed checks: `{failed}`",
      "",
      payload["next_action"],
      "",
      "This is source-contract evidence only. It does not claim speed.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--frontier", type=Path, default=DEFAULT_FRONTIER)
  parser.add_argument("--routes", type=Path, default=DEFAULT_ROUTES)
  parser.add_argument("--rejected", type=Path, default=DEFAULT_REJECTED)
  parser.add_argument("--best", type=Path, default=DEFAULT_BEST)
  parser.add_argument("--profile-log", type=Path, default=DEFAULT_PROFILE_LOG)
  parser.add_argument("--q4-full", type=Path, default=DEFAULT_Q4_FULL)
  parser.add_argument("--q4-occupancy", type=Path, default=DEFAULT_Q4_OCCUPANCY)
  parser.add_argument("--q6-occupancy", type=Path, default=DEFAULT_Q6_OCCUPANCY)
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
