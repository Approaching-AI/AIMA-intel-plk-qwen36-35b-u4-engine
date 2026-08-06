#!/usr/bin/env python3
"""Gate the resident decode-loop overhead root before the next source cut.

This is source/profile route-control evidence only. It proves that the next
admissible unit is engine-owned resident GPU hot-loop extraction, and that the
work is bounded by the live frontier plus the best paired correctness artifact.
It does not launch a token-emitting decode row and does not claim speed.
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
SCHEMA_VERSION = "intel-qwen36-resident-decode-loop-overhead-root-gate-v0"
DEFAULT_FRONTIER = ACTIVE / "frontier.json"
DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_REJECTED = ACTIVE / "rejected-routes.json"
DEFAULT_ACCEPTED = ACTIVE / "accepted-cuts.json"
DEFAULT_SEQ94 = ROOT / "output/post-rowgroup-route-gate-20260707Tseq94Z/metrics.json"
DEFAULT_SPEED = (
    ROOT
    / "output/r2-gpu-selected-shared-q4q6-down-cold-q6-experts-noqueue-speed-20260705T143006Z/result.json"
)
DEFAULT_DISTRIBUTION = (
    ROOT
    / "output/r2-gpu-selected-shared-q4q6-down-cold-q6-experts-noqueue-distribution-20260705T143408Z/result.json"
)
DEFAULT_RESIDENT_HEADER = ROOT / "engine/include/intel_qwen36/resident_harness.hpp"
DEFAULT_RESIDENT_SOURCE = ROOT / "engine/src/resident_harness.cpp"
DEFAULT_DECODE_SMOKE = ROOT / "tools/intel-qwen36-r2-gpu-decode-smoke.py"
DEFAULT_OUT_DIR = ROOT / "output/resident-decode-loop-overhead-root-gate-20260707Tseq95Z"


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


def _marker(text: str, pattern: str) -> bool:
  return re.search(pattern, text, flags=re.MULTILINE | re.DOTALL) is not None


def _route_names(rejected: dict[str, Any]) -> set[str]:
  names: set[str] = set()
  for row in rejected.get("rejected", []):
    if isinstance(row, dict) and isinstance(row.get("route"), str):
      names.add(row["route"])
  return names


def _accepted_ids(accepted: dict[str, Any]) -> set[str]:
  rows = accepted.get("accepted")
  rows = rows if isinstance(rows, list) else accepted.get("cuts", [])
  ids: set[str] = set()
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


def _frontier_summary(frontier: dict[str, Any]) -> dict[str, Any]:
  budget = frontier.get("goal_budget")
  budget = budget if isinstance(budget, dict) else {}
  per_token = budget.get("per_token_ms")
  per_token = per_token if isinstance(per_token, dict) else {}
  verdict = budget.get("verdict")
  verdict = verdict if isinstance(verdict, dict) else {}
  no_progress = frontier.get("no_progress")
  no_progress = no_progress if isinstance(no_progress, dict) else {}
  goal_anchor = frontier.get("goal_anchor")
  goal_anchor = goal_anchor if isinstance(goal_anchor, dict) else {}
  noise = no_progress.get("noise")
  noise = noise if isinstance(noise, dict) else {}
  wall = _num(per_token.get("wall"))
  floor_budget = _num(verdict.get("floor_budget_ms_per_token"))
  floor_gap = max(0.0, wall - floor_budget)
  overhead = _num(per_token.get("non_kernel_overhead"))
  return {
      "current_best_tps": _num(goal_anchor.get("current_best_tps")),
      "best_artifact": goal_anchor.get("best_artifact"),
      "best_correctness_artifact": goal_anchor.get("best_correctness_artifact"),
      "floor_tps": _num(verdict.get("floor_tps")),
      "wall_ms_per_token": wall,
      "floor_budget_ms_per_token": floor_budget,
      "floor_gap_ms_per_token": floor_gap,
      "gpu_kernel_busy_floor_ms_per_token": _num(
          per_token.get("gpu_kernel_busy_floor")),
      "non_kernel_overhead_ms_per_token": overhead,
      "overhead_cut_fraction_needed": floor_gap / overhead if overhead > 0 else 0.0,
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


def _collect_key_numbers(obj: Any, key_pattern: re.Pattern[str]) -> list[float]:
  values: list[float] = []
  if isinstance(obj, dict):
    for key, value in obj.items():
      if key_pattern.search(str(key)) and isinstance(value, (int, float)):
        values.append(float(value))
      values.extend(_collect_key_numbers(value, key_pattern))
  elif isinstance(obj, list):
    for item in obj:
      values.extend(_collect_key_numbers(item, key_pattern))
  return values


def _speed_profile(speed: dict[str, Any], frontier_state: dict[str, Any]) -> dict[str, Any]:
  smoke = speed.get("smoke")
  smoke = smoke if isinstance(smoke, dict) else {}
  tokens = int(_num(smoke.get("decode_continuation_output_tokens")))
  unprofiled_ns = _num(smoke.get("unprofiled_wall_ns"))
  profiled_ns = _num(smoke.get("profiled_wall_ns"))
  bookkeeping_ns = _num(_nested(smoke, "gpu_loop_bookkeeping_wall_profile_ns", "profiled"))
  floor_gap = frontier_state["floor_gap_ms_per_token"]
  unprofiled_ms = unprofiled_ns / 1_000_000.0
  unprofiled_ms_per_token = unprofiled_ms / tokens if tokens > 0 else 0.0
  bookkeeping_ms_per_token = (
      (bookkeeping_ns / 1_000_000.0) / tokens if tokens > 0 else 0.0
  )
  target_reduction_ms_total = floor_gap * tokens
  return {
      "required_checks_passed": speed.get("required_checks_passed") is True,
      "diagnostic_required_checks_passed": speed.get("required_checks_passed") is True,
      "failing_checks": [
          row.get("name")
          for row in speed.get("checks", [])
          if isinstance(row, dict) and row.get("pass") is False
      ],
      "decode_continuation_output_tokens": tokens,
      "gpu_hybrid_decode_tok_s": _num(smoke.get("gpu_hybrid_decode_tok_s")),
      "top1_matches_native": smoke.get("top1_matches_native") is True,
      "top1_match_count": int(_num(smoke.get("top1_match_count"))),
      "distribution_ladder_enabled": smoke.get("distribution_ladder_enabled") is True,
      "profiled_wall_ns": int(profiled_ns),
      "unprofiled_wall_ns": int(unprofiled_ns),
      "unprofiled_ms_total": unprofiled_ms,
      "unprofiled_ms_per_token": unprofiled_ms_per_token,
      "gpu_loop_bookkeeping_profiled_ns": int(bookkeeping_ns),
      "gpu_loop_bookkeeping_ms_per_token": bookkeeping_ms_per_token,
      "floor_gap_ms_per_token": floor_gap,
      "target_reduction_ms_total": target_reduction_ms_total,
      "required_fraction_of_unprofiled": (
          target_reduction_ms_total / unprofiled_ms if unprofiled_ms > 0.0 else 0.0
      ),
      "matches_frontier_best_tps": abs(
          _num(smoke.get("gpu_hybrid_decode_tok_s"))
          - frontier_state["current_best_tps"]
      ) < 0.000001,
  }


def _distribution_profile(distribution: dict[str, Any]) -> dict[str, Any]:
  smoke = distribution.get("smoke")
  smoke = smoke if isinstance(smoke, dict) else {}
  ladder = smoke.get("distribution_ladder")
  ladder = ladder if isinstance(ladder, dict) else {}
  max_kld = _num(ladder.get("max_kld"))
  if max_kld == 0.0:
    kld_values = _collect_key_numbers(ladder, re.compile(r"(^|_)kld$|kl", re.I))
    max_kld = max(kld_values) if kld_values else 0.0
  steps = ladder.get("steps")
  return {
      "required_checks_passed": distribution.get("required_checks_passed") is True,
      "distribution_ladder_enabled": smoke.get("distribution_ladder_enabled") is True,
      "distribution_ladder_required_checks_passed": (
          ladder.get("required_checks_passed") is True
      ),
      "decode_continuation_output_tokens": int(
          _num(smoke.get("decode_continuation_output_tokens"))),
      "top1_matches_native": smoke.get("top1_matches_native") is True,
      "top1_match_count": int(_num(smoke.get("top1_match_count"))),
      "top1_rate": _num(ladder.get("top1_rate")),
      "max_kld": max_kld,
      "kld_pass": ladder.get("kld_pass") is True,
      "top1_pass": ladder.get("top1_pass") is True,
      "step_count": len(steps) if isinstance(steps, list) else 0,
  }


def _source_profile(
    resident_header: str, resident_source: str, decode_smoke: str) -> dict[str, Any]:
  header_and_source = resident_header + "\n" + resident_source
  engine_loop_markers = [
      {
          "name": "resident_decode_loop_engine_class_present",
          "pass": _marker(resident_header, r"class\s+ResidentDecodeLoop\b"),
      },
      {
          "name": "resident_decode_loop_engine_run_present",
          "pass": _marker(
              resident_source,
              r"ResidentDecodeLoopResult\s+ResidentDecodeLoop::run",
          ),
      },
      {
          "name": "engine_hot_gpu_decode_loop_not_yet_extracted",
          "pass": not _marker(
              header_and_source,
              r"(ResidentGpuDecodeLoop|ResidentHotDecodeLoop|RunResidentGpuDecodeToken)",
          ),
      },
  ]
  generated_callback_markers = [
      {
          "name": "decode_smoke_uses_resident_loop_lambda_callback",
          "pass": _marker(
              decode_smoke,
              r"resident_decode_loop\.run\(\s*std::cout,\s*resident_loop_config,\s*\[&\]",
          ),
      },
      {
          "name": "decode_smoke_callback_calls_gpu_token_function",
          "pass": "RunGpuHybridDecodeToken(" in decode_smoke,
      },
      {
          "name": "decode_smoke_still_generates_target_cpp",
          "pass": "\"generated_cpp\"" in decode_smoke
          and "generated-source" in decode_smoke,
      },
  ]
  profiling_markers = [
      {"name": "captures_resident_counters", "pass": "DecodeCaptureResidentCounters(" in decode_smoke},
      {"name": "diffs_decode_stats", "pass": "DecodeStatsDelta(" in decode_smoke},
      {
          "name": "diffs_resident_counters",
          "pass": "DecodeResidentCounterDelta(" in decode_smoke,
      },
      {
          "name": "compares_distribution_step",
          "pass": "DecodeCompareDistributionStep(" in decode_smoke,
      },
      {"name": "pushes_token_profiles", "pass": "gpu_token_profiles.push_back" in decode_smoke},
      {"name": "pushes_step_topks", "pass": "gpu_topks_by_step.push_back" in decode_smoke},
      {"name": "serializes_topk_rows", "pass": "DecodeResidentTopKRows" in decode_smoke},
      {"name": "profiles_loop_bookkeeping", "pass": "loop_bookkeeping_profile" in decode_smoke},
  ]
  return {
      "engine_loop_markers": engine_loop_markers,
      "generated_callback_markers": generated_callback_markers,
      "profiling_overhead_markers": profiling_markers,
      "engine_markers_passed": all(row["pass"] for row in engine_loop_markers),
      "generated_callback_markers_passed": all(
          row["pass"] for row in generated_callback_markers),
      "profiling_markers_passed": all(row["pass"] for row in profiling_markers),
  }


def compute(args: argparse.Namespace) -> dict[str, Any]:
  frontier = _load_json(args.frontier)
  routes = _load_json(args.routes)
  rejected = _load_json(args.rejected)
  accepted = _load_json(args.accepted)
  seq94 = _load_json(args.seq94)
  speed = _load_json(args.speed)
  distribution = _load_json(args.distribution)
  resident_header = args.resident_header.read_text(encoding="utf-8")
  resident_source = args.resident_source.read_text(encoding="utf-8")
  decode_smoke = args.decode_smoke.read_text(encoding="utf-8")

  frontier_state = _frontier_summary(frontier)
  speed_state = _speed_profile(speed, frontier_state)
  distribution_state = _distribution_profile(distribution)
  source_state = _source_profile(resident_header, resident_source, decode_smoke)
  rejected_names = _route_names(rejected)
  accepted = _accepted_ids(accepted)

  accepted_resident_ids = [
      "r2_gpu_decode_resident_process_multisession_reuse",
      "r2_gpu_decode_resident_device_state_handle_bank",
      "r2_gpu_decode_resident_linear_q6_qkv_weight_store",
      "r2_gpu_decode_resident_linear_conv_weight_store",
      "r2_gpu_decode_resident_q4_cpu_order_weight_store",
      "r2_gpu_decode_resident_selected_q6_weight_store",
      "r2_gpu_decode_resident_lm_head_weight_store",
  ]
  optional_resident_ids = [
      "r2_gpu_decode_resident_packed_q4_weight_store",
      "r2_gpu_decode_resident_raw_q6_weight_store",
      "r2_gpu_decode_resident_f32_weight_store",
      "r2_gpu_decode_resident_norm_weight_store",
  ]
  missing_accepted = [row for row in accepted_resident_ids if row not in accepted]
  missing_optional = [row for row in optional_resident_ids if row not in accepted]

  checks = [
      {
          "name": "seq94_selected_resident_overhead_root_gate",
          "pass": seq94.get("required_checks_passed") is True
          and seq94.get("selected_next_route")
          == "resident_decode_loop_overhead_root_source_gate",
      },
      {
          "name": "seq94_candidate_recorded",
          "pass": _has_candidate(
              routes, 94, "switch_to_resident_decode_loop_overhead_root_source_gate"
          ),
      },
      {
          "name": "seq94_switch_recorded",
          "pass": _has_switch(
              routes, "switch_to_resident_decode_loop_overhead_root_source_gate", 94
          ),
      },
      {
          "name": "frontier_overhead_arithmetic_still_valid",
          "pass": (
              frontier_state["floor_tps"] == 19.5
              and frontier_state["current_best_tps"] > 0.0
              and frontier_state["current_best_tps"] < frontier_state["floor_tps"]
              and frontier_state["can_reach_floor_without_kernel_work"]
              and frontier_state["overhead_only_ceiling_tok_s"]
              >= frontier_state["floor_tps"]
              and frontier_state["non_kernel_overhead_ms_per_token"]
              > frontier_state["floor_gap_ms_per_token"]
              and 0.0 < frontier_state["overhead_cut_fraction_needed"] <= 0.05
          ),
          "detail": {
              "floor_gap_ms_per_token": frontier_state["floor_gap_ms_per_token"],
              "non_kernel_overhead_ms_per_token": frontier_state[
                  "non_kernel_overhead_ms_per_token"],
              "overhead_cut_fraction_needed": frontier_state[
                  "overhead_cut_fraction_needed"],
          },
      },
      {
          "name": "speed_artifact_matches_frontier_best_diagnostic_row",
          "pass": (
              speed_state["matches_frontier_best_tps"]
              and speed_state["decode_continuation_output_tokens"] == 8
              and speed_state["top1_matches_native"]
              and speed_state["top1_match_count"] == 8
          ),
          "detail": {
              "required_checks_passed": speed_state["required_checks_passed"],
              "failing_checks": speed_state["failing_checks"],
              "note": (
                  "The speed row is diagnostic because its wrapper checks are "
                  "false; paired distribution carries correctness evidence."
              ),
          },
      },
      {
          "name": "paired_distribution_ladder_passed",
          "pass": (
              distribution_state["required_checks_passed"]
              and distribution_state["distribution_ladder_enabled"]
              and distribution_state["distribution_ladder_required_checks_passed"]
              and distribution_state["decode_continuation_output_tokens"] == 8
              and distribution_state["top1_matches_native"]
              and distribution_state["top1_match_count"] == 8
              and distribution_state["top1_rate"] == 1.0
              and 0.0 < distribution_state["max_kld"] <= 0.005
              and distribution_state["step_count"] == 8
          ),
      },
      {
          "name": "unprofiled_source_target_can_cover_floor",
          "pass": (
              speed_state["unprofiled_ms_per_token"]
              >= frontier_state["floor_gap_ms_per_token"]
              and 0.0 < speed_state["required_fraction_of_unprofiled"] <= 1.0
          ),
          "detail": {
              "unprofiled_ms_per_token": speed_state["unprofiled_ms_per_token"],
              "target_reduction_ms_total": speed_state["target_reduction_ms_total"],
              "required_fraction_of_unprofiled": speed_state[
                  "required_fraction_of_unprofiled"],
          },
      },
      {
          "name": "bookkeeping_profile_alone_not_enough",
          "pass": (
              0.0 < speed_state["gpu_loop_bookkeeping_ms_per_token"]
              < frontier_state["floor_gap_ms_per_token"]
          ),
          "detail": {
              "gpu_loop_bookkeeping_ms_per_token": speed_state[
                  "gpu_loop_bookkeeping_ms_per_token"],
              "floor_gap_ms_per_token": frontier_state["floor_gap_ms_per_token"],
          },
      },
      {
          "name": "resident_loop_engine_shell_present",
          "pass": source_state["engine_markers_passed"],
          "detail": source_state["engine_loop_markers"],
      },
      {
          "name": "generated_callback_still_owns_hot_token_work",
          "pass": source_state["generated_callback_markers_passed"],
          "detail": source_state["generated_callback_markers"],
      },
      {
          "name": "generated_callback_owns_profile_and_correctness_overhead",
          "pass": source_state["profiling_markers_passed"],
          "detail": source_state["profiling_overhead_markers"],
      },
      {
          "name": "accepted_resident_ownership_prereqs_present",
          "pass": not missing_accepted,
          "detail": {
              "required": accepted_resident_ids,
              "missing": missing_accepted,
              "optional_missing": missing_optional,
          },
      },
      {
          "name": "engine_hot_loop_extraction_not_rejected",
          "pass": "engine_resident_gpu_hot_loop_extraction" not in rejected_names
          and "gpu_resident_decode_loop_overhead_root_source_gate" not in rejected_names,
      },
  ]
  required = all(row["pass"] for row in checks)
  selected_next_route = (
      "engine_resident_gpu_hot_loop_extraction"
      if required
      else "resident_decode_loop_overhead_root_gate_failed"
  )
  disposition = (
      "resident_decode_loop_overhead_root_contract_ready"
      if required
      else "resident_decode_loop_overhead_root_gate_failed"
  )
  next_action = (
      "Implement a default-off engine-owned resident GPU hot decode loop that "
      "uses the accepted resident state and weight stores, preserving the "
      "paired teacher-forced correctness path. The source cut must target the "
      "generated-callback unprofiled wall bucket, not the tiny loop-bookkeeping "
      "profile bucket; first token-emitting validation is `decode-smoke "
      "--explore`, and promotion still needs confirm plus paired distribution "
      "outside the 0.50% noise band."
      if required
      else "Fix failed gate checks before writing the resident hot-loop source cut."
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
          "seq94": _rel(args.seq94),
          "speed": _rel(args.speed),
          "distribution": _rel(args.distribution),
          "resident_header": _rel(args.resident_header),
          "resident_source": _rel(args.resident_source),
          "decode_smoke": _rel(args.decode_smoke),
      },
      "frontier": frontier_state,
      "speed_profile": speed_state,
      "distribution_profile": distribution_state,
      "source_profile": source_state,
      "accepted_resident_required": accepted_resident_ids,
      "accepted_resident_missing": missing_accepted,
      "accepted_resident_optional_missing": missing_optional,
      "checks": checks,
  }


def write_outputs(out_dir: Path, payload: dict[str, Any]) -> None:
  out_dir.mkdir(parents=True, exist_ok=False)
  (out_dir / "metrics.json").write_text(
      json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  manifest = {
      "schema_version": payload["schema_version"],
      "workstream": payload["workstream"],
      "tool": "tools/intel-qwen36-resident-decode-loop-overhead-root-gate.py",
      "inputs": payload["inputs"],
      "selected_next_route": payload["selected_next_route"],
      "speedup_claims_allowed": False,
  }
  (out_dir / "manifest.json").write_text(
      json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  failed = [row["name"] for row in payload["checks"] if not row["pass"]]
  frontier = payload["frontier"]
  speed = payload["speed_profile"]
  distribution = payload["distribution_profile"]
  lines = [
      "# Resident Decode-Loop Overhead Root Gate",
      "",
      f"- required checks passed: `{str(payload['required_checks_passed']).lower()}`",
      f"- disposition: `{payload['disposition']}`",
      f"- selected next route: `{payload['selected_next_route']}`",
      f"- frontier gap: `{frontier['floor_gap_ms_per_token']:.3f}` ms/token",
      f"- generated-callback unprofiled wall: `{speed['unprofiled_ms_per_token']:.6f}` ms/token",
      f"- reduction required from unprofiled bucket: `{speed['required_fraction_of_unprofiled']:.3%}`",
      f"- loop-bookkeeping profile bucket: `{speed['gpu_loop_bookkeeping_ms_per_token']:.6f}` ms/token",
      f"- paired distribution: max KLD `{distribution['max_kld']:.12g}`, top-1 `{distribution['top1_rate']:.3f}`",
      f"- failed checks: `{failed}`",
      "",
      payload["next_action"],
      "",
      "This is source/profile route-control evidence only. It does not claim speed.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--frontier", type=Path, default=DEFAULT_FRONTIER)
  parser.add_argument("--routes", type=Path, default=DEFAULT_ROUTES)
  parser.add_argument("--rejected", type=Path, default=DEFAULT_REJECTED)
  parser.add_argument("--accepted", type=Path, default=DEFAULT_ACCEPTED)
  parser.add_argument("--seq94", type=Path, default=DEFAULT_SEQ94)
  parser.add_argument("--speed", type=Path, default=DEFAULT_SPEED)
  parser.add_argument("--distribution", type=Path, default=DEFAULT_DISTRIBUTION)
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
