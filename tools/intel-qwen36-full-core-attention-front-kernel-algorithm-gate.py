#!/usr/bin/env python3
"""Audit the full-core/attention-front kernel-algorithm route after seq109.

This is route-control evidence only. It consumes the live frontier, the
selected-FFN closure, and prior full-core/attention-front closures. It either
authorizes a component proof for a materially new kernel algorithm or closes
the current full-core/attention-front board before another token-emitting row.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = "intel-qwen36-full-core-attention-front-kernel-algorithm-gate-v0"
DEFAULT_FRONTIER = ACTIVE / "frontier.json"
DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_REJECTED = ACTIVE / "rejected-routes.json"
DEFAULT_BEST = (
    ROOT
    / "output/r2-gpu-selected-shared-q4q6-down-cold-q6-experts-noqueue-speed-20260705T143006Z/result.json"
)
DEFAULT_PROFILE_LOG = ROOT / "output/explore-log.jsonl"
DEFAULT_SEQ109 = (
    ROOT / "output/selected-ffn-kernel-layout-component-gate-20260707Tseq109Z/metrics.json"
)
DEFAULT_SEQ85 = (
    ROOT / "output/attn-linear-regression-root-gate-20260707Tseq85Z/metrics.json"
)
DEFAULT_SEQ86 = (
    ROOT / "output/post-attn-linear-route-switch-gate-20260707Tseq86Z/metrics.json"
)
DEFAULT_ATTN_BUDGET = (
    ROOT / "output/attn-linear-budget-20260707Tseq82Z/budget.json"
)
DEFAULT_RESIDUAL_CONFIRM = (
    ROOT / "output/r2-gpu-attention-front-resident-residual-noqueue-speed-20260706T101403Z/result.json"
)
DEFAULT_DECODE = ROOT / "tools/intel-qwen36-r2-gpu-decode-smoke.py"
DEFAULT_ENGINE = ROOT / "engine/src/gpu_q4x8_matvec.cpp"
DEFAULT_HEADER = ROOT / "engine/include/intel_qwen36/gpu_q4x8_matvec.hpp"
DEFAULT_OPENCL = ROOT / "engine/gpu/opencl/q4x8_matvec.cl"
DEFAULT_OUT_DIR = (
    ROOT / "output/full-core-attention-front-kernel-algorithm-gate-20260707Tseq110Z"
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
      "current_best_tps": _num(
          _nested(no_progress, "last_significant_improvement", "tps")),
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


def _absent_markers(text: str, markers: list[str]) -> dict[str, Any]:
  present = [marker for marker in markers if marker in text]
  return {"pass": not present, "present": present, "marker_count": len(markers)}


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


def _summary_for_row(row: dict[str, Any]) -> dict[str, Any]:
  profile = row.get("profile_smoke")
  profile = profile if isinstance(profile, dict) else {}
  wall = profile.get("wall_profile_ns")
  wall = wall if isinstance(wall, dict) else {}
  return {
      "ts": row.get("ts"),
      "label": row.get("label"),
      "tps": _num(row.get("tps")),
      "top1_matches_native": row.get("top1_matches_native"),
      "required_checks_passed": row.get("required_checks_passed"),
      "full_core_ms_per_token": _num(wall.get("full_core")) / 8.0 / 1e6,
      "attention_front_ms_per_token": _num(wall.get("attention_front")) / 8.0 / 1e6,
      "full_core_handoff_kernel_us": _num(
          profile.get("full_core_attention_front_handoff_kernel_us")),
      "attention_front_handoff_kernel_us": _num(
          profile.get("attention_front_handoff_kernel_us")),
  }


def _smoke(result: dict[str, Any]) -> dict[str, Any]:
  return result.get("smoke") if isinstance(result.get("smoke"), dict) else result


def _ms_per_token_kernel_cut(
    baseline_us_over_row: float, candidate_us_over_row: float, tokens: float = 8.0
) -> float:
  return max(0.0, baseline_us_over_row - candidate_us_over_row) / tokens / 1000.0


def _best_unpromoted_tps(rows: list[dict[str, Any]], residual_confirm: dict[str, Any]) -> dict[str, Any]:
  candidates = [
      _summary_for_row(_label_row(rows, label))
      for label in [
          "fullcore-score-local32-noqueue",
          "fullcore-softmax-cache-noqueue",
          "fullcore-applyscore-local64-noqueue",
          "fullcore-coregate-finish-coalesce-noqueue",
          "attention-residual-rmsnorm-fused-noqueue",
          "attn-linear-finish-bundle-noqueue",
      ]
  ]
  residual_smoke = _smoke(residual_confirm)
  candidates.append({
      "ts": "20260706T101403Z",
      "label": "attention-front-resident-residual-confirm",
      "tps": _num(residual_smoke.get("gpu_hybrid_decode_tok_s")),
      "top1_matches_native": residual_smoke.get("top1_match_count") == 8,
      "required_checks_passed": residual_smoke.get("required_checks_passed"),
      "full_core_ms_per_token": _num(
          _nested(residual_smoke, "wall_profile_ns", "full_core")) / 8.0 / 1e6,
      "attention_front_ms_per_token": _num(
          _nested(residual_smoke, "wall_profile_ns", "attention_front")) / 8.0 / 1e6,
      "full_core_handoff_kernel_us": 0.0,
      "attention_front_handoff_kernel_us": 0.0,
  })
  candidates = [row for row in candidates if row.get("tps", 0.0) > 0.0]
  best = max(candidates, key=lambda row: _num(row.get("tps"))) if candidates else {}
  return {"best": best, "candidates": candidates}


def compute(args: argparse.Namespace) -> dict[str, Any]:
  frontier = _load_json(args.frontier)
  routes = _load_json(args.routes)
  rejected = _load_json(args.rejected)
  best = _load_json(args.best)
  seq109 = _load_json(args.seq109)
  seq85 = _load_json(args.seq85)
  seq86 = _load_json(args.seq86)
  attn_budget = _load_json(args.attn_budget)
  residual_confirm = _load_json(args.residual_confirm)
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
  substage_gaps = _substage_gaps(frontier)
  floor_gap = frontier_state["floor_gap_ms_per_token"]
  noise = frontier_state["noise_rel"]
  current_best_tps = frontier_state["current_best_tps"]
  rejected_names = _route_names(rejected)

  profile_baseline = _summary_for_row(
      _label_row(explore_rows, "selected-shared-q4q6-down-cold-q6-experts-profile"))
  softmax_profile = _summary_for_row(
      _label_row(explore_rows, "fullcore-softmax-cache-profile"))
  residual_profile = _summary_for_row(
      _label_row(explore_rows, "attention-residual-rmsnorm-fused-profile"))
  best_tps = _best_unpromoted_tps(explore_rows, residual_confirm)
  best_row = best_tps["best"]
  best_rel_delta = (
      (_num(best_row.get("tps")) / current_best_tps) - 1.0
      if current_best_tps > 0.0 else 0.0
  )

  fullcore_softmax_kernel_cut = _ms_per_token_kernel_cut(
      profile_baseline["full_core_handoff_kernel_us"],
      softmax_profile["full_core_handoff_kernel_us"],
  )
  attention_residual_kernel_cut = _ms_per_token_kernel_cut(
      profile_baseline["attention_front_handoff_kernel_us"],
      residual_profile["attention_front_handoff_kernel_us"],
  )

  required_closed_routes = [
      "current_selected_ffn_kernel_layout_component_board",
      "gpu_full_attention_core_scratch_buffers",
      "gpu_full_attention_state_resident_history",
      "gpu_full_core_attention_front_handoff",
      "gpu_fullcore_attention_front_handoff_nonblocking_uploads",
      "gpu_fullcore_attention_output_q4_cpu_order_all_after_tailhandle",
      "gpu_fullcore_coregate_finish_coalesce_noqueue",
      "gpu_fullcore_apply_score_local64_noqueue",
      "gpu_fullcore_score_local32_noqueue",
      "gpu_fullcore_softmax_weight_cache_noqueue",
      "gpu_attention_front_resident_residual_input_noqueue",
      "gpu_attention_residual_rmsnorm_fusion",
      "gpu_attention_linear_finish_bundle_noqueue",
      "gpu_attention_linear_simple_final_output_handoffs",
      "gpu_linear_final_device_q8_handoff_noqueue",
      "gpu_linear_final_device_q8_handoff_scratch_noqueue",
      "gpu_linear_preconv_shared_q8_preconv_bundle_decode",
      "gpu_attention_linear_event_lifetime_combined_alias",
  ]
  missing_closed_routes = [
      route for route in required_closed_routes if route not in rejected_names
  ]

  source_contract = _has_markers(
      source,
      [
          "full_attn_score_f32",
          "full_attn_apply_score_gate_f32",
          "RunFullAttentionCoreGateThenResidentPackedQ4X8ResidualRmsNorm",
          "RunF32InputHandleDeviceQ8ThenResidentPackedQ4X8ResidualRmsNorm",
          "IQ36_ATTENTION_LINEAR_EVENT_LIFETIME",
      ],
  )
  transient_algorithm_absent = _absent_markers(
      source,
      [
          "full_attn_softmax_weight_cache",
          "full_attn_score_local32",
          "IQ36_FULLCORE_SOFTMAX_CACHE",
          "IQ36_FULLCORE_SCORE_LOCAL32",
          "IQ36_FULLCORE_APPLYSCORE_LOCAL64",
          "IQ36_FULLCORE_COREFINISH_COALESCE",
          "IQ36_ATTENTION_RESIDUAL_RMSNORM_FUSION",
          "IQ36_ATTENTION_LINEAR_FINISH_BUNDLE",
      ],
  )

  next_candidates = {
      "linear_preconv": stage_gaps.get("linear_preconv", 0.0),
      "attention_front": stage_gaps.get("attention_front", 0.0),
      "linear_delta": stage_gaps.get("linear_delta", 0.0),
  }
  next_stage = max(next_candidates, key=next_candidates.get)
  next_gap = next_candidates[next_stage]

  smoke = _smoke(best)
  checks = [
      {
          "name": "seq109_selected_this_gate",
          "pass": seq109.get("required_checks_passed") is True
          and seq109.get("selected_next_route")
          == "full_core_attention_front_kernel_algorithm_gate"
          and _has_switch(
              routes,
              "close_current_selected_ffn_kernel_layout_route_switch_to_full_core_attention_front_gate",
              109,
          ),
      },
      {
          "name": "frontier_still_below_floor",
          "pass": frontier_state["wall_ms_per_token"]
          > frontier_state["floor_budget_ms_per_token"]
          > 0.0,
      },
      {
          "name": "full_core_gap_can_cover_floor_but_requires_component_proof",
          "pass": stage_gaps.get("full_core", 0.0) > floor_gap,
          "detail": {
              "full_core_gap_ms_per_token": stage_gaps.get("full_core", 0.0),
              "full_core_wall_ms_per_token": stage_walls.get("full_core", 0.0),
              "floor_gap_ms_per_token": floor_gap,
          },
      },
      {
          "name": "accepted_stack_has_full_core_attention_handoff_and_top1",
          "pass": smoke.get("top1_match_count") == 8
          and smoke.get("resident_full_core_attention_front_handoff_enabled") is True
          and smoke.get("resident_attention_front_handoff_enabled") is True,
      },
      {
          "name": "prior_attention_linear_roots_still_closed",
          "pass": seq85.get("required_checks_passed") is True
          and seq86.get("required_checks_passed") is True
          and seq86.get("attention_readback_only_decode_admissible") is False,
      },
      {
          "name": "closed_route_board_covers_current_full_core_attention_board",
          "pass": not missing_closed_routes,
          "detail": {"missing_closed_routes": missing_closed_routes},
      },
      {
          "name": "best_prior_fullcore_attention_candidate_does_not_clear_noise_or_floor",
          "pass": _num(best_row.get("tps")) < frontier_state["floor_tps"]
          and best_rel_delta < noise,
          "detail": {
              "best_label": best_row.get("label"),
              "best_tps": best_row.get("tps"),
              "current_best_tps": current_best_tps,
              "rel_delta": best_rel_delta,
              "noise_rel": noise,
              "floor_tps": frontier_state["floor_tps"],
          },
      },
      {
          "name": "softmax_cache_profile_does_not_cut_full_core_kernel",
          "pass": fullcore_softmax_kernel_cut <= 0.0,
          "detail": {
              "baseline_full_core_handoff_kernel_us": profile_baseline[
                  "full_core_handoff_kernel_us"],
              "softmax_cache_handoff_kernel_us": softmax_profile[
                  "full_core_handoff_kernel_us"],
              "projected_cut_ms_per_token": fullcore_softmax_kernel_cut,
          },
      },
      {
          "name": "attention_residual_fusion_component_cut_below_floor_gap",
          "pass": 0.0 <= attention_residual_kernel_cut < floor_gap,
          "detail": {
              "baseline_attention_front_handoff_kernel_us": profile_baseline[
                  "attention_front_handoff_kernel_us"],
              "residual_fusion_handoff_kernel_us": residual_profile[
                  "attention_front_handoff_kernel_us"],
              "projected_cut_ms_per_token": attention_residual_kernel_cut,
              "floor_gap_ms_per_token": floor_gap,
          },
      },
      {
          "name": "current_source_has_only_default_closed_algorithm_shapes",
          "pass": source_contract["pass"] and transient_algorithm_absent["pass"],
          "detail": {
              "required_markers": source_contract,
              "transient_markers_absent": transient_algorithm_absent,
          },
      },
      {
          "name": "next_gap_after_closure_can_cover_floor",
          "pass": next_gap > floor_gap,
          "detail": {
              "next_stage": next_stage,
              "next_gap_ms_per_token": next_gap,
              "floor_gap_ms_per_token": floor_gap,
          },
      },
      {
          "name": "seq109_candidate_recorded",
          "pass": _has_candidate(
              routes,
              109,
              "close_current_selected_ffn_kernel_layout_component_route",
          ),
      },
  ]
  required = all(check["pass"] for check in checks)
  selected_next_route = (
      "linear_preconv_remaining_kernel_algorithm_gate"
      if required else "manual_review_full_core_attention_front_gate"
  )
  disposition = (
      "close_current_full_core_attention_front_kernel_algorithm_board"
      if required else "full_core_attention_front_kernel_algorithm_gate_failed"
  )
  next_action = (
      "Close the current full-core/attention-front kernel-algorithm board as a "
      "decode-probe source. The next unit is a linear-preconv remaining-kernel "
      "algorithm gate, component-first, focused on alpha/beta and postconv-prep "
      "envelope movement without reopening shared-Q8 qkv/conv, simple "
      "linear-final device-Q8 handoff, full-core score/softmax/local-size, or "
      "attention residual/RMSNorm fusion routes."
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
          "best": _rel(args.best),
          "profile_log": _rel(args.profile_log),
          "seq109": _rel(args.seq109),
          "seq85": _rel(args.seq85),
          "seq86": _rel(args.seq86),
          "attn_budget": _rel(args.attn_budget),
          "residual_confirm": _rel(args.residual_confirm),
          "decode_source": _rel(args.decode_source),
          "engine_source": _rel(args.engine_source),
          "engine_header": _rel(args.engine_header),
          "opencl_source": _rel(args.opencl_source),
      },
      "frontier": frontier_state,
      "stage_gap_ms_per_token": stage_gaps,
      "stage_wall_ms_per_token": stage_walls,
      "remaining_substage_gap_ms_per_token": {
          key: value for key, value in substage_gaps.items()
          if key.startswith("linear_preconv.") or key.startswith("linear_delta.")
      },
      "full_core_attention_summary": {
          "full_core_gap_ms_per_token": stage_gaps.get("full_core", 0.0),
          "attention_front_gap_ms_per_token": stage_gaps.get("attention_front", 0.0),
          "combined_full_core_attention_gap_ms_per_token": (
              stage_gaps.get("full_core", 0.0)
              + stage_gaps.get("attention_front", 0.0)
          ),
          "profile_baseline": profile_baseline,
          "softmax_profile": softmax_profile,
          "residual_fusion_profile": residual_profile,
          "softmax_cache_projected_cut_ms_per_token": fullcore_softmax_kernel_cut,
          "attention_residual_projected_cut_ms_per_token": attention_residual_kernel_cut,
          "best_prior_candidate": best_row,
          "best_prior_candidate_rel_delta": best_rel_delta,
          "attn_linear_gap_upper_bound_ms_per_token": attn_budget.get(
              "same_source_gap_upper_bound_ms_per_token"),
      },
      "closed_route_requirements": {
          "required": required_closed_routes,
          "missing": missing_closed_routes,
      },
      "next_route_reason": {
          "next_stage": next_stage,
          "next_gap_ms_per_token": next_gap,
          "linear_preconv_gap_ms_per_token": stage_gaps.get("linear_preconv", 0.0),
          "attention_front_gap_ms_per_token": stage_gaps.get("attention_front", 0.0),
          "linear_delta_gap_ms_per_token": stage_gaps.get("linear_delta", 0.0),
      },
      "checks": checks,
  }


def write_outputs(out_dir: Path, payload: dict[str, Any]) -> None:
  out_dir.mkdir(parents=True, exist_ok=False)
  (out_dir / "metrics.json").write_text(
      json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  failed = [check["name"] for check in payload["checks"] if check["pass"] is not True]
  frontier = payload["frontier"]
  summary = payload["full_core_attention_summary"]
  next_reason = payload["next_route_reason"]
  best = summary["best_prior_candidate"]
  lines = [
      "# Full-Core / Attention-Front Kernel Algorithm Gate",
      "",
      f"- required checks passed: `{str(payload['required_checks_passed']).lower()}`",
      f"- disposition: `{payload['disposition']}`",
      f"- decode probe allowed: `{str(payload['decode_probe_allowed']).lower()}`",
      f"- selected next route: `{payload['selected_next_route']}`",
      f"- frontier gap: `{frontier['floor_gap_ms_per_token']:.3f}` ms/token",
      f"- full-core / attention-front gaps: "
      f"`{summary['full_core_gap_ms_per_token']:.3f}` / "
      f"`{summary['attention_front_gap_ms_per_token']:.3f}` ms/token",
      f"- best prior candidate: `{best.get('label')}` at "
      f"`{_num(best.get('tps')):.6f}` tok/s",
      f"- softmax-cache projected kernel cut: "
      f"`{summary['softmax_cache_projected_cut_ms_per_token']:.3f}` ms/token",
      f"- attention residual-fusion projected kernel cut: "
      f"`{summary['attention_residual_projected_cut_ms_per_token']:.3f}` ms/token",
      f"- next stage: `{next_reason['next_stage']}` "
      f"(`{next_reason['next_gap_ms_per_token']:.3f}` ms/token)",
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
  parser.add_argument("--best", type=Path, default=DEFAULT_BEST)
  parser.add_argument("--profile-log", type=Path, default=DEFAULT_PROFILE_LOG)
  parser.add_argument("--seq109", type=Path, default=DEFAULT_SEQ109)
  parser.add_argument("--seq85", type=Path, default=DEFAULT_SEQ85)
  parser.add_argument("--seq86", type=Path, default=DEFAULT_SEQ86)
  parser.add_argument("--attn-budget", type=Path, default=DEFAULT_ATTN_BUDGET)
  parser.add_argument("--residual-confirm", type=Path, default=DEFAULT_RESIDUAL_CONFIRM)
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
