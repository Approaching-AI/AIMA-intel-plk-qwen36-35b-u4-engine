#!/usr/bin/env python3
"""Select the next route after handoff matvec submit-split profiling."""

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
    "intel-qwen36-post-attention-front-handoff-matvec-submit-split-route-gate-v0"
)

DEFAULT_FRONTIER = ACTIVE / "frontier.json"
DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_REJECTED = ACTIVE / "rejected-routes.json"
DEFAULT_ACCEPTED = ACTIVE / "accepted-cuts.json"
DEFAULT_ENGINE = ROOT / "engine/src/gpu_q4x8_matvec.cpp"
DEFAULT_DECODE = ROOT / "tools/intel-qwen36-r2-gpu-decode-smoke.py"
DEFAULT_SEQ178 = (
    ROOT
    / "output/resident-attention-front-handoff-matvec-submit-split-explore-gate-20260708Tseq178Z"
    / "metrics.json"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "output/post-attention-front-handoff-matvec-submit-split-route-gate-20260708Tseq179Z"
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


def _num(value: Any) -> float:
  return float(value) if isinstance(value, (int, float)) else 0.0


def _dict(value: Any) -> dict[str, Any]:
  return value if isinstance(value, dict) else {}


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
  glide = _dict(no_progress.get("glide_slope"))
  wall = _num(per_token.get("wall"))
  floor_budget = _num(verdict.get("floor_budget_ms_per_token"))
  return {
      "current_best_tps": _num(anchor.get("current_best_tps")),
      "floor_tps": _num(anchor.get("same_host_vulkan_floor_tps")),
      "wall_ms_per_token": wall,
      "floor_budget_ms_per_token": floor_budget,
      "floor_gap_ms_per_token": max(0.0, wall - floor_budget),
      "kernel_busy_floor_ms_per_token": _num(per_token.get("kernel_busy_floor")),
      "non_kernel_overhead_ms_per_token": _num(
          per_token.get("non_kernel_overhead")),
      "overhead_only_ceiling_tok_s": _num(
          verdict.get("overhead_only_ceiling_tok_s")),
      "can_reach_floor_without_kernel_work": bool(
          verdict.get("can_reach_floor_without_kernel_work")),
      "noise_rel": _num(noise.get("rel")),
      "runs_since_significant_improvement": no_progress.get(
          "runs_since_significant_improvement"),
      "hard_stall_breached": no_progress.get("hard_stall_breached"),
      "review_recorded_for_current_best": no_progress.get(
          "review_recorded_for_current_best"),
      "glide_projected_runs_to_floor": glide.get("projected_runs_to_floor"),
      "glide_breached": glide.get("breached"),
  }


def _stage_walls(frontier: dict[str, Any]) -> dict[str, float]:
  budget = _dict(frontier.get("goal_budget"))
  out: dict[str, float] = {}
  for row in budget.get("top_stage_walls_ms_per_token", []):
    if isinstance(row, dict) and isinstance(row.get("stage"), str):
      out[row["stage"]] = _num(row.get("ms_per_token"))
  return out


def _ms_per_token_map(values: dict[str, Any]) -> dict[str, float]:
  return {key: _num(value) for key, value in values.items()}


def _component_target(profile: dict[str, Any],
                      floor_gap_ms_per_token: float) -> dict[str, Any]:
  profile_ns = _dict(profile.get("profile_ns"))
  tokens = _num(_dict(profile.get("explore_row")).get("decode_tokens")) or 8.0
  layer_count = 40.0
  calls = tokens * layer_count
  kernel_wait_ns = _num(profile_ns.get("kernel_wait"))
  current_us_per_call = kernel_wait_ns / calls / 1000.0 if calls else 0.0
  required_cut_us_per_call = floor_gap_ms_per_token * 1000.0 / layer_count
  required_ratio = (
      required_cut_us_per_call / current_us_per_call
      if current_us_per_call > 0.0 else 0.0
  )
  return {
      "assumed_layers": int(layer_count),
      "assumed_decode_tokens": int(tokens),
      "kernel_wait_current_us_per_call": current_us_per_call,
      "required_cut_us_per_call": required_cut_us_per_call,
      "required_component_speedup_ratio": (
          1.0 / (1.0 - required_ratio)
          if 0.0 < required_ratio < 1.0 else 0.0
      ),
      "target_us_per_call": max(0.0, current_us_per_call - required_cut_us_per_call),
  }


def compute(args: argparse.Namespace) -> dict[str, Any]:
  frontier = _load_json(args.frontier)
  routes = _load_json(args.routes)
  rejected = _load_json(args.rejected)
  accepted = _load_json(args.accepted)
  seq178 = _load_json(args.seq178)
  engine_source = _read(args.engine_source)
  decode_source = _read(args.decode_source)

  frontier_state = _frontier_state(frontier)
  floor_gap = frontier_state["floor_gap_ms_per_token"]
  profile = _dict(seq178.get(
      "attention_front_handoff_matvec_submit_split_profile"))
  buckets = _ms_per_token_map(_dict(profile.get("buckets_ms_per_token")))
  wall_profile = _ms_per_token_map(
      _dict(_dict(seq178.get("wall_profile")).get("buckets_ms_per_token")))
  explore_row = _dict(seq178.get("explore_row"))
  profile["explore_row"] = explore_row
  target = _component_target(profile, floor_gap)
  rejected_names = _rejected_names(rejected)
  accepted_ids = _accepted_ids(accepted)

  required_rejected = {
      "current_full_core_attention_front_kernel_algorithm_board",
      "gpu_attention_front_resident_residual_input_noqueue",
      "gpu_attention_q4_matvec_residual_fuse_noqueue",
      "gpu_attention_residual_rmsnorm_fusion",
      "gpu_attention_linear_simple_final_output_handoffs",
      "gpu_linear_final_f32input_output_projection_component",
      "gpu_q4_bpr16_output_projection_localq8",
      "gpu_fullcore_attention_output_q4_cpu_order_all_after_tailhandle",
  }
  missing_rejected = sorted(required_rejected - rejected_names)
  local64_present = (
      "small_q4_output_projection_local64" in accepted_ids
      and "kSmallQ4RowlaneLocalSize = 64" in engine_source
      and "blocks_per_row == 16 && global == 2048" in engine_source
  )
  submit_split_instrumented = (
      "IQ36_ATTENTION_FRONT_HANDOFF_MATVEC_SUBMIT_SPLIT_PROFILE" in decode_source
      and "AttentionFrontHandoffMatvecSubmitSplitProfile" in engine_source
      and "queue_drain_cleanup_wall_ns" in engine_source
  )

  checks = [
      {
          "name": "seq178_selected_this_route_gate",
          "pass": (
              seq178.get("required_checks_passed") is True
              and seq178.get("selected_next_route")
              == "post_attention_front_handoff_matvec_submit_split_route_gate"
              and _has_candidate(
                  routes,
                  178,
                  "accept_resident_attention_front_handoff_matvec_submit_split_profile_explore",
              )
              and _has_switch(
                  routes,
                  "select_post_attention_front_handoff_matvec_submit_split_route_gate",
                  178,
              )
          ),
          "detail": {
              "seq178_disposition": seq178.get("disposition"),
              "seq178_selected_next_route": seq178.get("selected_next_route"),
          },
      },
      {
          "name": "profile_row_preserved_top1_and_cannot_promote",
          "pass": (
              explore_row.get("top1_matches_native") is True
              and explore_row.get("speedup_claims_allowed") is False
              and _num(explore_row.get("tps"))
              < frontier_state["current_best_tps"]
          ),
          "detail": explore_row,
      },
      {
          "name": "handoff_matvec_is_kernel_wait_finish_not_submit_overhead",
          "pass": (
              buckets.get("kernel_wait", 0.0) >= floor_gap
              and buckets.get("kernel_finish", 0.0) >= floor_gap
              and buckets.get("kernel_enqueue", 0.0) < floor_gap
              and buckets.get("kernel_setup", 0.0) < floor_gap
              and buckets.get("queue_drain_cleanup", 0.0) < floor_gap
              and buckets.get("kernel_finish", 0.0)
              >= 0.95 * buckets.get("kernel_wait", 0.0)
          ),
          "detail": {
              "buckets_ms_per_token": buckets,
              "floor_gap_ms_per_token": floor_gap,
          },
      },
      {
          "name": "attention_front_remains_floor_sized",
          "pass": (
              wall_profile.get("attention_front", 0.0) >= floor_gap
              and wall_profile.get("linear_preconv", 0.0) >= floor_gap
              and wall_profile.get("selected_ffn", 0.0) >= floor_gap
              and wall_profile.get("lm_head_gpu", 0.0) >= floor_gap
          ),
          "detail": {
              "wall_profile_ms_per_token": wall_profile,
              "frontier_stage_walls_ms_per_token": _stage_walls(frontier),
          },
      },
      {
          "name": "accepted_current_q4_output_projection_local64_is_active",
          "pass": local64_present,
          "detail": {
              "accepted_id_present": (
                  "small_q4_output_projection_local64" in accepted_ids),
              "source_local64_branch_present": (
                  "kSmallQ4RowlaneLocalSize = 64" in engine_source),
              "source_shape_guard_present": (
                  "blocks_per_row == 16 && global == 2048" in engine_source),
          },
      },
      {
          "name": "closed_attention_projection_variants_recorded",
          "pass": not missing_rejected,
          "detail": {"missing": missing_rejected},
      },
      {
          "name": "submit_split_source_is_default_off_and_present",
          "pass": submit_split_instrumented,
      },
      {
          "name": "component_gate_has_floor_covering_target",
          "pass": (
              target["required_component_speedup_ratio"] > 1.0
              and target["target_us_per_call"] > 0.0
          ),
          "detail": target,
      },
      {
          "name": "frontier_still_below_floor_no_speed_claim",
          "pass": (
              frontier_state["current_best_tps"] < frontier_state["floor_tps"]
              and frontier_state["hard_stall_breached"] is True
              and frontier_state["review_recorded_for_current_best"] is True
          ),
          "detail": frontier_state,
      },
  ]
  required = all(bool(row.get("pass")) for row in checks)
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "inputs": {
          "frontier": _rel(args.frontier),
          "routes": _rel(args.routes),
          "rejected": _rel(args.rejected),
          "accepted": _rel(args.accepted),
          "seq178_profile_gate": _rel(args.seq178),
          "engine_source": _rel(args.engine_source),
          "engine_source_sha256": _sha256(args.engine_source),
          "decode_source": _rel(args.decode_source),
          "decode_source_sha256": _sha256(args.decode_source),
      },
      "frontier": frontier_state,
      "profile": {
          "explore_row": explore_row,
          "buckets_ms_per_token": buckets,
          "floor_gap_ms_per_token": floor_gap,
          "component_target": target,
      },
      "closed_route_context": {
          "required_rejected": sorted(required_rejected),
          "missing_rejected": missing_rejected,
          "accepted_ids_used": ["small_q4_output_projection_local64"],
      },
      "checks": checks,
      "required_checks_passed": required,
      "decode_probe_allowed": False,
      "component_probe_allowed": required,
      "speedup_claims_allowed": False,
      "disposition": (
          "select_attention_front_handoff_matvec_kernel_algorithm_component_gate"
          if required else
          "reject_attention_front_handoff_matvec_submit_split_route_selection"
      ),
      "selected_next_route": (
          "attention_front_handoff_matvec_kernel_algorithm_component_gate"
          if required else
          "post_attention_front_handoff_matvec_submit_split_route_fix_gate"
      ),
      "next_route_reason": (
          "Seq178 proves the attention-front handoff matvec bucket is kernel "
          "wait/finish dominated, not submit/setup/queue cleanup. Because the "
          "accepted local64 BPR16 output-projection shape is active and the "
          "simple attention-front variants are closed, the next admissible "
          "unit is a component/design gate for a materially new Q4 "
          "output-projection kernel algorithm. It must beat the current "
          f"{target['kernel_wait_current_us_per_call']:.3f} us/call by at "
          f"least {target['required_cut_us_per_call']:.3f} us/call "
          f"({target['required_component_speedup_ratio']:.3f}x) before any "
          "decode row."
          if required else
          "Route selection checks failed; fix evidence before selecting a new "
          "probe."
      ),
  }


def write_outputs(payload: dict[str, Any], out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=False)
  (out_dir / "metrics.json").write_text(
      json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  manifest = {
      "schema_version": payload["schema_version"],
      "workstream": payload["workstream"],
      "tool": _rel(Path(__file__)),
      "selected_next_route": payload["selected_next_route"],
      "speedup_claims_allowed": payload["speedup_claims_allowed"],
      "inputs": payload["inputs"],
  }
  (out_dir / "manifest.json").write_text(
      json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  failed = [row["name"] for row in payload["checks"] if not row.get("pass")]
  target = payload["profile"]["component_target"]
  lines = [
      "# Post Attention-Front Handoff Matvec Submit-Split Route Gate",
      "",
      f"- required_checks_passed: `{str(payload['required_checks_passed']).lower()}`",
      f"- disposition: `{payload['disposition']}`",
      f"- selected_next_route: `{payload['selected_next_route']}`",
      f"- decode_probe_allowed: `{str(payload['decode_probe_allowed']).lower()}`",
      f"- component_probe_allowed: `{str(payload['component_probe_allowed']).lower()}`",
      f"- current_us_per_call: `{target['kernel_wait_current_us_per_call']:.3f}`",
      f"- target_us_per_call: `{target['target_us_per_call']:.3f}`",
      f"- failed_checks: `{failed}`",
      "",
      payload["next_route_reason"],
      "",
      "This is route-control evidence. It does not claim speed.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--frontier", type=Path, default=DEFAULT_FRONTIER)
  parser.add_argument("--routes", type=Path, default=DEFAULT_ROUTES)
  parser.add_argument("--rejected", type=Path, default=DEFAULT_REJECTED)
  parser.add_argument("--accepted", type=Path, default=DEFAULT_ACCEPTED)
  parser.add_argument("--engine-source", type=Path, default=DEFAULT_ENGINE)
  parser.add_argument("--decode-source", type=Path, default=DEFAULT_DECODE)
  parser.add_argument("--seq178", type=Path, default=DEFAULT_SEQ178)
  parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
  return parser.parse_args()


def main() -> int:
  args = parse_args()
  payload = compute(args)
  write_outputs(payload, args.out_dir)
  print(json.dumps({
      "required_checks_passed": payload["required_checks_passed"],
      "disposition": payload["disposition"],
      "selected_next_route": payload["selected_next_route"],
      "component_probe_allowed": payload["component_probe_allowed"],
      "out_dir": _rel(args.out_dir),
  }, sort_keys=True))
  return 0 if payload["required_checks_passed"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
