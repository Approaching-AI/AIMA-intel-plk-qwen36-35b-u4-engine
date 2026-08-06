#!/usr/bin/env python3
"""Select the next route after attention-front handoff wall-split profiling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = (
    "intel-qwen36-post-attention-front-handoff-wall-split-route-gate-v0"
)

DEFAULT_FRONTIER = ACTIVE / "frontier.json"
DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_REJECTED = ACTIVE / "rejected-routes.json"
DEFAULT_ENGINE = ROOT / "engine/src/gpu_q4x8_matvec.cpp"
DEFAULT_DECODE = ROOT / "tools/intel-qwen36-r2-gpu-decode-smoke.py"
DEFAULT_SEQ174 = (
    ROOT
    / "output/resident-attention-front-handoff-wall-split-explore-gate-20260708Tseq174Z"
    / "metrics.json"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "output/post-attention-front-handoff-wall-split-route-gate-20260708Tseq175Z"
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


def _floor_buckets(values: dict[str, Any],
                   floor_gap: float) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  for key, value in values.items():
    ms = _num(value)
    if ms >= floor_gap:
      rows.append({"bucket": key, "ms_per_token": ms})
  rows.sort(key=lambda row: row["ms_per_token"], reverse=True)
  return rows


def compute(args: argparse.Namespace) -> dict[str, Any]:
  frontier = _load_json(args.frontier)
  routes = _load_json(args.routes)
  rejected = _load_json(args.rejected)
  seq174 = _load_json(args.seq174)
  engine_source = _read(args.engine_source)
  decode_source = _read(args.decode_source)

  frontier_state = _frontier_state(frontier)
  floor_gap = frontier_state["floor_gap_ms_per_token"]
  profile = _dict(seq174.get("attention_front_handoff_wall_split_profile"))
  buckets = _dict(profile.get("buckets_ms_per_token"))
  wall = _dict(seq174.get("wall_profile"))
  wall_buckets = _dict(wall.get("buckets_ms_per_token"))
  rejected_names = _rejected_names(rejected)

  required_rejected = {
      "gpu_attention_front_resident_output_handoff",
      "gpu_attention_front_rmsnorm_read_as_completion_drain",
      "gpu_attention_front_skip_linear_out_materialization",
      "gpu_attention_q4_matvec_residual_fuse_noqueue",
      "gpu_attention_linear_simple_final_output_handoffs",
      "current_full_core_attention_front_kernel_algorithm_board",
  }
  missing_rejected = sorted(required_rejected - rejected_names)
  floor_sized = {
      key: value for key, value in buckets.items() if _num(value) >= floor_gap
  }
  matvec_split_absent = (
      "IQ36_ATTENTION_FRONT_HANDOFF_MATVEC_SUBMIT_SPLIT_PROFILE"
      not in decode_source
      and "attention_front_handoff_matvec_submit_split_profile"
      not in decode_source
      and "handoff_matvec_enqueue_wall_ns" not in engine_source
      and "handoff_matvec_finish_wall_ns" not in engine_source
  )
  matvec_target_present = (
      "RunResidentPackedQ4X8ThenResidentResidualRmsNorm" in engine_source
      and "run.timing.matvec = RunKernel(" in engine_source
      and "clFinish(kernel)" in engine_source
      and "ClearPendingHostUploadsAfterQueueDrain" in engine_source
  )

  checks = [
      {
          "name": "seq174_selected_this_route_gate",
          "pass": (
              seq174.get("required_checks_passed") is True
              and seq174.get("selected_next_route")
              == "post_attention_front_handoff_wall_split_route_gate"
              and _has_candidate(
                  routes,
                  174,
                  "accept_resident_attention_front_handoff_wall_split_profile_explore",
              )
              and _has_switch(
                  routes,
                  "accept_resident_attention_front_handoff_wall_split_compile_switch_to_profile_explore_gate",
                  173,
              )
          ),
          "detail": {
              "seq174_disposition": seq174.get("disposition"),
              "seq174_selected_next_route": seq174.get("selected_next_route"),
          },
      },
      {
          "name": "handoff_matvec_is_dominant_floor_sized_bucket",
          "pass": (
              profile.get("largest_bucket") == "matvec"
              and _num(buckets.get("matvec")) >= floor_gap
              and _num(buckets.get("matvec"))
              > _num(buckets.get("residual_rmsnorm_enqueue_finish"))
          ),
          "detail": {
              "largest_bucket": profile.get("largest_bucket"),
              "matvec_ms_per_token": _num(buckets.get("matvec")),
              "residual_rmsnorm_enqueue_finish_ms_per_token": _num(
                  buckets.get("residual_rmsnorm_enqueue_finish")),
              "setup_ms_per_token": _num(buckets.get("setup")),
              "floor_gap_ms_per_token": floor_gap,
              "floor_sized": floor_sized,
          },
      },
      {
          "name": "readbacks_are_not_dominant",
          "pass": (
              _num(buckets.get("residual_read")) < floor_gap
              and _num(buckets.get("normalized_read")) < floor_gap
              and _num(buckets.get("residual_read"))
              < _num(buckets.get("matvec"))
              and _num(buckets.get("normalized_read"))
              < _num(buckets.get("matvec"))
          ),
          "detail": {
              "residual_read_ms_per_token": _num(buckets.get("residual_read")),
              "normalized_read_ms_per_token": _num(
                  buckets.get("normalized_read")),
              "floor_gap_ms_per_token": floor_gap,
          },
      },
      {
          "name": "remaining_wall_buckets_floor_sized",
          "pass": (
              _num(wall_buckets.get("attention_front")) >= floor_gap
              and _num(wall_buckets.get("linear_preconv")) >= floor_gap
              and _num(wall_buckets.get("selected_ffn")) >= floor_gap
              and _num(wall_buckets.get("lm_head_gpu")) >= floor_gap
              and _num(wall_buckets.get("ffn_tail")) >= floor_gap
          ),
          "detail": {
              "floor_gap_ms_per_token": floor_gap,
              "floor_buckets": _floor_buckets(wall_buckets, floor_gap),
          },
      },
      {
          "name": "closed_simple_attention_handoffs_recorded",
          "pass": not missing_rejected,
          "detail": {"missing": missing_rejected},
      },
      {
          "name": "source_has_matvec_target_but_no_submit_split_yet",
          "pass": matvec_target_present and matvec_split_absent,
          "detail": {
              "matvec_target_present": matvec_target_present,
              "matvec_split_absent": matvec_split_absent,
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
          "seq174_profile_gate": _rel(args.seq174),
          "engine_source": _rel(args.engine_source),
          "decode_source": _rel(args.decode_source),
      },
      "frontier": frontier_state,
      "attention_front_handoff_wall_split_profile": {
          "buckets_ms_per_token": buckets,
          "floor_buckets": _floor_buckets(buckets, floor_gap),
          "largest_bucket": profile.get("largest_bucket"),
          "largest_ms_per_token": _num(profile.get("largest_ms_per_token")),
      },
      "remaining_wall": {
          "buckets_ms_per_token": wall_buckets,
          "floor_buckets": _floor_buckets(wall_buckets, floor_gap),
      },
      "checks": checks,
      "required_checks_passed": required_checks_passed,
      "speedup_claims_allowed": False,
      "decode_probe_allowed": False,
      "component_probe_allowed": False,
      "source_profile_gate_required_before_decode": True,
      "disposition": (
          "select_resident_attention_front_handoff_matvec_submit_split_source_gate"
          if required_checks_passed else
          "reject_post_attention_front_handoff_wall_split_route_gate"
      ),
      "selected_next_route": (
          "resident_attention_front_handoff_matvec_submit_split_source_gate"
          if required_checks_passed else
          "post_attention_front_handoff_wall_split_route_fix_gate"
      ),
      "next_route_reason": (
          "The handoff split shows the floor-sized attention-front handoff wall "
          "is dominated by the resident Q4 output-projection matvec call. "
          "Readbacks are not dominant, while residual/RMSNorm enqueue+finish "
          "and setup are secondary floor-sized buckets. The next unit is a "
          "default-off submit/finish split inside the handoff matvec RunKernel "
          "path before any speed row or algorithm claim."
          if required_checks_passed else
          "Route evidence is incomplete; fix the handoff wall-split profile or "
          "ledger state before selecting another source gate."
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
  profile = metrics["attention_front_handoff_wall_split_profile"]
  summary = [
      "# Post Attention-Front Handoff Wall-Split Route Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- largest_handoff_bucket: `{profile['largest_bucket']}`",
      f"- largest_handoff_ms_per_token: `{profile['largest_ms_per_token']:.3f}`",
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
  parser.add_argument("--seq174", type=Path, default=DEFAULT_SEQ174)
  parser.add_argument("--engine-source", type=Path, default=DEFAULT_ENGINE)
  parser.add_argument("--decode-source", type=Path, default=DEFAULT_DECODE)
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
      "largest_handoff_bucket": (
          metrics["attention_front_handoff_wall_split_profile"]["largest_bucket"]),
      "out_dir": _rel(args.out_dir),
  }, sort_keys=True))
  return 0 if metrics["required_checks_passed"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
