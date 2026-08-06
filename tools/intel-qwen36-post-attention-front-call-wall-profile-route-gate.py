#!/usr/bin/env python3
"""Select the next route after attention-front call wall-profile evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = (
    "intel-qwen36-post-attention-front-call-wall-profile-route-gate-v0"
)

DEFAULT_FRONTIER = ACTIVE / "frontier.json"
DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_REJECTED = ACTIVE / "rejected-routes.json"
DEFAULT_DECODE = ROOT / "tools/intel-qwen36-r2-gpu-decode-smoke.py"
DEFAULT_ENGINE = ROOT / "engine/src/gpu_q4x8_matvec.cpp"
DEFAULT_SEQ170 = (
    ROOT
    / "output/resident-attention-front-call-wall-profile-explore-gate-20260708Tseq170Z"
    / "metrics.json"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "output/post-attention-front-call-wall-profile-route-gate-20260708Tseq171Z"
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
      "noise_rel": _num(noise.get("rel")),
      "wall_ms_per_token": wall,
      "floor_budget_ms_per_token": floor_budget,
      "floor_gap_ms_per_token": max(0.0, wall - floor_budget),
      "runs_since_significant_improvement": no_progress.get(
          "runs_since_significant_improvement"),
      "hard_stall_breached": no_progress.get("hard_stall_breached"),
      "review_recorded_for_current_best": no_progress.get(
          "review_recorded_for_current_best"),
      "can_reach_floor_without_kernel_work": bool(
          verdict.get("can_reach_floor_without_kernel_work")),
      "overhead_only_ceiling_tok_s": _num(
          verdict.get("overhead_only_ceiling_tok_s")),
  }


def _floor_buckets(values: dict[str, Any], floor_gap: float) -> list[dict[str, Any]]:
  rows = [
      {"bucket": key, "ms_per_token": _num(value)}
      for key, value in values.items()
      if _num(value) >= floor_gap
  ]
  return sorted(rows, key=lambda row: row["ms_per_token"], reverse=True)


def compute(args: argparse.Namespace) -> dict[str, Any]:
  frontier = _load_json(args.frontier)
  routes = _load_json(args.routes)
  rejected = _load_json(args.rejected)
  seq170 = _load_json(args.seq170)
  decode_source = _read(args.decode_source)
  engine_source = _read(args.engine_source)
  frontier_state = _frontier_state(frontier)
  profile = seq170.get("attention_front_call_wall_profile")
  profile = profile if isinstance(profile, dict) else {}
  buckets = profile.get("buckets_ms_per_token")
  buckets = buckets if isinstance(buckets, dict) else {}
  wall = seq170.get("wall_profile")
  wall = wall if isinstance(wall, dict) else {}
  wall_buckets = wall.get("buckets_ms_per_token")
  wall_buckets = wall_buckets if isinstance(wall_buckets, dict) else {}
  floor_gap = frontier_state["floor_gap_ms_per_token"]
  rejected_count = len(rejected.get("rejected", []))

  handoff_ms = _num(buckets.get("handoff"))
  q8_bridge_ms = _num(buckets.get("q8_bridge"))
  largest_bucket = profile.get("largest_bucket")
  non_handoff_floor_sized = {
      key: _num(value)
      for key, value in buckets.items()
      if key not in {"handoff", "profiled"} and _num(value) >= floor_gap
  }
  source_has_no_split = (
      "IQ36_ATTENTION_FRONT_HANDOFF_WALL_SPLIT_PROFILE" not in decode_source
      and "attention_front_handoff_wall_split_profile" not in decode_source
      and "attention_front_handoff_wall_split_profile" not in engine_source
  )
  handoff_function_present = (
      "RunResidentPackedQ4X8ThenResidentResidualRmsNorm" in engine_source
      and "clEnqueueReadBuffer(Q4 resident RMSNorm normalized)" in engine_source
      and "clFinish(Q4 resident RMSNorm)" in engine_source
  )

  checks = [
      {
          "name": "seq170_selected_this_route_gate",
          "pass": (
              seq170.get("required_checks_passed") is True
              and seq170.get("selected_next_route")
              == "post_attention_front_call_wall_profile_route_gate"
              and _has_candidate(
                  routes,
                  170,
                  "accept_resident_attention_front_call_wall_profile_explore",
              )
              and _has_switch(
                  routes,
                  "accept_resident_attention_front_call_wall_profile_compile_switch_to_profile_explore_gate",
                  169,
              )
          ),
          "detail": {
              "seq170_disposition": seq170.get("disposition"),
              "seq170_selected_next_route": seq170.get("selected_next_route"),
          },
      },
      {
          "name": "attention_front_handoff_is_dominant_floor_sized_inner_bucket",
          "pass": (
              largest_bucket == "handoff"
              and handoff_ms >= floor_gap
              and not non_handoff_floor_sized
          ),
          "detail": {
              "largest_bucket": largest_bucket,
              "handoff_ms_per_token": handoff_ms,
              "q8_bridge_ms_per_token": q8_bridge_ms,
              "floor_gap_ms_per_token": floor_gap,
              "non_handoff_floor_sized": non_handoff_floor_sized,
          },
      },
      {
          "name": "remaining_wall_buckets_floor_sized",
          "pass": (
              wall_buckets.get("attention_front", 0.0) >= floor_gap
              and wall_buckets.get("linear_preconv", 0.0) >= floor_gap
              and wall_buckets.get("selected_ffn", 0.0) >= floor_gap
              and wall_buckets.get("lm_head_gpu", 0.0) >= floor_gap
              and wall_buckets.get("ffn_tail", 0.0) >= floor_gap
          ),
          "detail": {
              "floor_gap_ms_per_token": floor_gap,
              "floor_buckets": _floor_buckets(wall_buckets, floor_gap),
          },
      },
      {
          "name": "source_has_handoff_target_but_no_inner_split_yet",
          "pass": source_has_no_split and handoff_function_present,
          "detail": {
              "source_has_no_split": source_has_no_split,
              "handoff_function_present": handoff_function_present,
          },
      },
      {
          "name": "frontier_still_below_floor_no_speed_claim",
          "pass": (
              frontier_state["current_best_tps"] < frontier_state["floor_tps"]
              and frontier_state["hard_stall_breached"] is True
              and frontier_state["review_recorded_for_current_best"] is True
              and frontier_state["can_reach_floor_without_kernel_work"] is True
          ),
          "detail": frontier_state,
      },
  ]
  required_checks_passed = all(bool(row.get("pass")) for row in checks)
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "inputs": {
          "frontier": _rel(args.frontier),
          "routes": _rel(args.routes),
          "rejected": _rel(args.rejected),
          "rejected_route_count": rejected_count,
          "seq170_profile_gate": _rel(args.seq170),
          "decode_source": _rel(args.decode_source),
          "engine_source": _rel(args.engine_source),
      },
      "frontier": frontier_state,
      "attention_front_call_wall_profile": {
          "buckets_ms_per_token": buckets,
          "largest_bucket": largest_bucket,
          "largest_ms_per_token": _num(profile.get("largest_ms_per_token")),
      },
      "remaining_wall": {
          "buckets_ms_per_token": wall_buckets,
          "floor_buckets": _floor_buckets(wall_buckets, floor_gap),
          "largest_bucket": (
              _floor_buckets(wall_buckets, floor_gap)[0]["bucket"]
              if _floor_buckets(wall_buckets, floor_gap) else ""),
      },
      "checks": checks,
      "required_checks_passed": required_checks_passed,
      "decode_probe_allowed": False,
      "component_probe_allowed": False,
      "source_profile_gate_required_before_decode": True,
      "speedup_claims_allowed": False,
      "disposition": (
          "select_resident_attention_front_handoff_wall_split_source_gate"
          if required_checks_passed else
          "reject_post_attention_front_call_wall_profile_route_gate"
      ),
      "selected_next_route": (
          "resident_attention_front_handoff_wall_split_source_gate"
          if required_checks_passed else
          "attention_front_call_wall_profile_route_fix_gate"
      ),
      "next_route_reason": (
          "The call wall profile shows the floor-sized attention-front call "
          "wall is dominated by the resident attention-front handoff call "
          "while setup, handle, Q8 bridge, runner setup, projection, residual "
          "norm, and diagnostic override are below the floor gap. The next "
          "unit is default-off handoff inner wall attribution in "
          "RunResidentPackedQ4X8ThenResidentResidualRmsNorm before any speed "
          "row."
          if required_checks_passed else
          "The attention-front call profile does not justify a clean route "
          "switch. Fix route evidence before opening another profile source."
      ),
  }


def write_outputs(metrics: dict[str, Any], out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  manifest = {
      "schema_version": metrics["schema_version"],
      "workstream": metrics["workstream"],
      "tool": _rel(Path(__file__)),
      "selected_next_route": metrics["selected_next_route"],
      "speedup_claims_allowed": metrics["speedup_claims_allowed"],
      "inputs": metrics["inputs"],
  }
  (out_dir / "manifest.json").write_text(
      json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  failed = [row["name"] for row in metrics["checks"] if not row.get("pass")]
  profile = metrics["attention_front_call_wall_profile"]
  summary = [
      "# Post Attention-Front Call Wall-Profile Route Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- largest_inner_bucket: `{profile['largest_bucket']}`",
      f"- largest_inner_ms_per_token: `{profile['largest_ms_per_token']:.3f}`",
      f"- decode_probe_allowed: `{str(metrics['decode_probe_allowed']).lower()}`",
      f"- failed_checks: `{failed}`",
      "",
      metrics["next_route_reason"],
      "",
      "This is route-control evidence. It does not claim speed.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(summary), encoding="utf-8")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--frontier", type=Path, default=DEFAULT_FRONTIER)
  parser.add_argument("--routes", type=Path, default=DEFAULT_ROUTES)
  parser.add_argument("--rejected", type=Path, default=DEFAULT_REJECTED)
  parser.add_argument("--decode-source", type=Path, default=DEFAULT_DECODE)
  parser.add_argument("--engine-source", type=Path, default=DEFAULT_ENGINE)
  parser.add_argument("--seq170", type=Path, default=DEFAULT_SEQ170)
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
