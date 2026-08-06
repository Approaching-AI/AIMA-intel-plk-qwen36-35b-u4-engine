#!/usr/bin/env python3
"""Record the post-seq85 route switch from attention/linear probes.

This gate is source/route-control evidence only. It prevents a readback-only
attention-front no-op from being treated as a decode candidate when the current
accepted stack still needs host FFN inputs/residuals downstream, and it selects
the next dominant-bucket component proof.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = "intel-qwen36-post-attn-linear-route-switch-gate-v0"
DEFAULT_FRONTIER = ACTIVE / "frontier.json"
DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_REJECTED = ACTIVE / "rejected-routes.json"
DEFAULT_SEQ85 = ROOT / "output/attn-linear-regression-root-gate-20260707Tseq85Z/metrics.json"
DEFAULT_DECODE = ROOT / "tools/intel-qwen36-r2-gpu-decode-smoke.py"
DEFAULT_ENGINE = ROOT / "engine/src/gpu_q4x8_matvec.cpp"
DEFAULT_HEADER = ROOT / "engine/include/intel_qwen36/gpu_q4x8_matvec.hpp"
DEFAULT_OUT_DIR = ROOT / "output/post-attn-linear-route-switch-gate-20260707Tseq86Z"


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


def _route_names(rejected: dict[str, Any]) -> set[str]:
  rows = rejected.get("rejected")
  if not isinstance(rows, list):
    return set()
  names: set[str] = set()
  for row in rows:
    if isinstance(row, dict) and isinstance(row.get("route"), str):
      names.add(row["route"])
  return names


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


def _stage_gaps(frontier: dict[str, Any]) -> dict[str, float]:
  budget = frontier.get("goal_budget")
  budget = budget if isinstance(budget, dict) else {}
  rows = budget.get("stage_kernel_gap_estimates_ms_per_token")
  gaps: dict[str, float] = {}
  if isinstance(rows, list):
    for row in rows:
      if isinstance(row, dict) and isinstance(row.get("stage"), str):
        gaps[row["stage"]] = _num(row.get("gap_ms_per_token"))
  return gaps


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
      "current_best_tps": _num(
          (no_progress.get("last_significant_improvement") or {}).get("tps")
          if isinstance(no_progress.get("last_significant_improvement"), dict)
          else None
      ),
      "floor_tps": _num(verdict.get("floor_tps")),
      "wall_ms_per_token": wall,
      "floor_budget_ms_per_token": floor_budget,
      "floor_gap_ms_per_token": max(0.0, wall - floor_budget),
      "noise_rel": _num(noise.get("rel")),
      "runs_since_significant_improvement": no_progress.get(
          "runs_since_significant_improvement"
      ),
  }


def _has_markers(text: str, markers: list[str]) -> dict[str, Any]:
  missing = [marker for marker in markers if marker not in text]
  return {"pass": not missing, "missing": missing, "marker_count": len(markers)}


def compute(args: argparse.Namespace) -> dict[str, Any]:
  frontier = _load_json(args.frontier)
  routes = _load_json(args.routes)
  rejected = _load_json(args.rejected)
  seq85 = _load_json(args.seq85)
  source = "\n".join([
      _read(args.decode_source),
      _read(args.engine_source),
      _read(args.engine_header),
  ])

  frontier_state = _frontier_summary(frontier)
  gaps = _stage_gaps(frontier)
  selected_gap = gaps.get("selected_ffn", 0.0)
  largest_gap = max(gaps.values()) if gaps else 0.0
  largest_gap_stage = next(
      (stage for stage, gap in gaps.items() if gap == largest_gap), ""
  )

  rejected_names = _route_names(rejected)
  required_closed_routes = [
      "gpu_attention_linear_simple_final_output_handoffs",
      "gpu_linear_preconv_shared_q8_preconv_bundle_decode",
      "gpu_ffn_tail_resident_input_noqueue",
      "gpu_ffn_tail_plus_attention_residual_carrier_noqueue",
      "gpu_q6_nonatomic_down_tail_decode_fusion",
      "gpu_attention_linear_event_lifetime_combined_alias",
  ]
  missing_closed_routes = [
      route for route in required_closed_routes if route not in rejected_names
  ]

  attention_chain_markers = _has_markers(
      source,
      [
          "RunF32InputHandleDeviceQ8ThenResidentPackedQ4X8ResidualRmsNorm",
          "clFinish(device Q8 input-handle residual/norm)",
          "clEnqueueReadBuffer(device Q8 input-handle residual)",
          "clEnqueueReadBuffer(device Q8 input-handle normalized)",
      ],
  )
  downstream_host_markers = _has_markers(
      source,
      [
          "std::vector<float> RunGpuHybridFfnTail(",
          "const std::vector<float>* ffn_input_used = &ffn_input;",
          "const std::vector<float>* attention_residual_used = &attention_residual;",
          "*ffn_input_used, router.expert_ids",
          "RunFfnTailFromDownHandles(",
          "shared_gpu.down_handle, *attention_residual_used,",
      ],
  )
  resident_escape_markers = _has_markers(
      source,
      [
          "g_decode_ffn_tail_resident_input",
          "RunFfnTailFromDownHandlesResidentInputs(",
          "ffn_input_handle != 0",
          "attention_residual_handle != 0",
      ],
  )

  seq85_ok = (
      seq85.get("required_checks_passed") is True
      and seq85.get("disposition")
      == "attention_linear_regression_roots_bound_to_closed_classes"
  )
  seq82_switch_recorded = _has_switch(
      routes, "switch_decode_route_to_attention_linear_event_lifetime_proof", 82
  )
  attention_readback_only_decode_admissible = not (
      downstream_host_markers["pass"]
      and resident_escape_markers["pass"]
      and not missing_closed_routes
  )

  checks = [
      {
          "name": "frontier_still_below_floor",
          "pass": frontier_state["wall_ms_per_token"]
          > frontier_state["floor_budget_ms_per_token"]
          > 0.0,
      },
      {
          "name": "seq82_attention_linear_switch_was_recorded",
          "pass": seq82_switch_recorded,
      },
      {
          "name": "seq85_roots_bound_to_closed_classes",
          "pass": seq85_ok,
      },
      {
          "name": "attention_chain_still_readback_bound",
          "pass": attention_chain_markers["pass"],
          "detail": attention_chain_markers,
      },
      {
          "name": "accepted_stack_still_uses_host_ffn_inputs_downstream",
          "pass": downstream_host_markers["pass"],
          "detail": downstream_host_markers,
      },
      {
          "name": "resident_escape_paths_are_closed",
          "pass": not missing_closed_routes and resident_escape_markers["pass"],
          "detail": {
              "missing_closed_routes": missing_closed_routes,
              "resident_escape_markers": resident_escape_markers,
          },
      },
      {
          "name": "selected_ffn_remains_largest_gap",
          "pass": largest_gap_stage == "selected_ffn"
          and selected_gap > frontier_state["floor_gap_ms_per_token"],
          "detail": {
              "largest_gap_stage": largest_gap_stage,
              "selected_ffn_gap_ms_per_token": selected_gap,
              "floor_gap_ms_per_token": frontier_state["floor_gap_ms_per_token"],
          },
      },
      {
          "name": "attention_readback_only_decode_not_admissible",
          "pass": attention_readback_only_decode_admissible is False,
      },
  ]
  required = all(check["pass"] for check in checks)
  selected_next_route = (
      "selected_ffn_material_layout_component_proof"
      if required
      else "route_selection_needs_manual_review"
  )
  disposition = (
      "switch_from_attention_linear_to_selected_ffn_material_layout_component_proof"
      if required
      else "post_attention_linear_switch_evidence_incomplete"
  )
  next_action = (
      "Switch away from attention/linear decode probes. The next unit is a "
      "selected-FFN material/layout source or component proof that changes the "
      "largest remaining gap without repeating closed down-tail, local-size, "
      "DPAS local-Q8, group8, concat-handle, or simple device-Q8 handoff routes."
      if required
      else "Complete the missing route-control evidence before launching another target probe."
  )

  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "required_checks_passed": required,
      "speedup_claims_allowed": False,
      "disposition": disposition,
      "selected_next_route": selected_next_route,
      "next_action": next_action,
      "attention_readback_only_decode_admissible": attention_readback_only_decode_admissible,
      "inputs": {
          "frontier": _rel(args.frontier),
          "routes": _rel(args.routes),
          "rejected": _rel(args.rejected),
          "seq85": _rel(args.seq85),
          "decode_source": _rel(args.decode_source),
          "engine_source": _rel(args.engine_source),
          "engine_header": _rel(args.engine_header),
      },
      "frontier": frontier_state,
      "stage_gap_ms_per_token": gaps,
      "closed_route_requirements": {
          "required": required_closed_routes,
          "missing": missing_closed_routes,
      },
      "checks": checks,
  }


def write_outputs(out_dir: Path, payload: dict[str, Any]) -> None:
  out_dir.mkdir(parents=True, exist_ok=False)
  (out_dir / "metrics.json").write_text(
      json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  failed = [check["name"] for check in payload["checks"] if not check["pass"]]
  frontier = payload["frontier"]
  gaps = payload["stage_gap_ms_per_token"]
  lines = [
      "# Post Attention/Linear Route Switch Gate",
      "",
      f"- required checks passed: `{str(payload['required_checks_passed']).lower()}`",
      f"- disposition: `{payload['disposition']}`",
      f"- selected next route: `{payload['selected_next_route']}`",
      f"- frontier gap: `{frontier['floor_gap_ms_per_token']:.3f}` ms/token",
      f"- selected-FFN gap: `{gaps.get('selected_ffn', 0.0):.3f}` ms/token",
      f"- attention readback-only decode admissible: `{str(payload['attention_readback_only_decode_admissible']).lower()}`",
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
  parser.add_argument("--seq85", type=Path, default=DEFAULT_SEQ85)
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
  return 0 if payload["required_checks_passed"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
