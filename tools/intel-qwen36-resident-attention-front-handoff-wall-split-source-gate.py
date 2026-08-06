#!/usr/bin/env python3
"""Audit default-off attention-front handoff wall-split source wiring.

This is source/generate-only evidence. It verifies that seq171's selected
resident attention-front handoff route now has source attribution inside
RunResidentPackedQ4X8ThenResidentResidualRmsNorm before target compile or any
token row is allowed.
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
    "intel-qwen36-resident-attention-front-handoff-wall-split-source-gate-v0"
)

DEFAULT_FRONTIER = ACTIVE / "frontier.json"
DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_REJECTED = ACTIVE / "rejected-routes.json"
DEFAULT_DECODE = ROOT / "tools/intel-qwen36-r2-gpu-decode-smoke.py"
DEFAULT_ENGINE = ROOT / "engine/src/gpu_q4x8_matvec.cpp"
DEFAULT_HEADER = ROOT / "engine/include/intel_qwen36/gpu_q4x8_matvec.hpp"
DEFAULT_SEQ171 = (
    ROOT
    / "output/post-attention-front-call-wall-profile-route-gate-20260708Tseq171Z"
    / "metrics.json"
)
DEFAULT_GENERATE_DIR = (
    ROOT
    / "output/resident-attention-front-handoff-wall-split-generate-only-20260708Tseq172Z"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "output/resident-attention-front-handoff-wall-split-source-gate-20260708Tseq172Z"
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
  wall = _num(per_token.get("wall"))
  floor_budget = _num(verdict.get("floor_budget_ms_per_token"))
  return {
      "current_best_tps": _num(anchor.get("current_best_tps")),
      "floor_tps": _num(anchor.get("same_host_vulkan_floor_tps")),
      "floor_gap_ms_per_token": max(0.0, wall - floor_budget),
      "runs_since_significant_improvement": no_progress.get(
          "runs_since_significant_improvement"),
      "hard_stall_breached": no_progress.get("hard_stall_breached"),
      "review_recorded_for_current_best": no_progress.get(
          "review_recorded_for_current_best"),
      "can_reach_floor_without_kernel_work": bool(
          verdict.get("can_reach_floor_without_kernel_work")),
  }


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


def _marker_state(text: str, required: list[str],
                  forbidden: list[str] | None = None) -> dict[str, Any]:
  forbidden = forbidden or []
  missing = [marker for marker in required if marker not in text]
  present_forbidden = [marker for marker in forbidden if marker in text]
  return {
      "required": required,
      "missing": missing,
      "forbidden": forbidden,
      "present_forbidden": present_forbidden,
      "pass": not missing and not present_forbidden,
  }


def _decode_markers(text: str) -> dict[str, Any]:
  return _marker_state(
      text,
      [
          "IQ36_ATTENTION_FRONT_HANDOFF_WALL_SPLIT_PROFILE",
          "attention_front_handoff_wall_split_profile",
          "g_decode_attention_front_handoff_wall_split_profile",
          "attention_front_handoff_setup_wall_ns",
          "attention_front_handoff_residual_input_write_wall_ns",
          "attention_front_handoff_matvec_wall_ns",
          "attention_front_handoff_residual_rmsnorm_args_wall_ns",
          "attention_front_handoff_residual_rmsnorm_enqueue_finish_wall_ns",
          "attention_front_handoff_event_profile_wall_ns",
          "attention_front_handoff_residual_read_wall_ns",
          "attention_front_handoff_normalized_read_wall_ns",
          "attention_front_handoff_alias_wall_ns",
          "attention_front_handoff_release_wall_ns",
          "DecodeAttentionFrontHandoffWallSnapshotNow",
          "DecodeAddAttentionFrontHandoffWallDelta",
          "WriteAttentionFrontHandoffWallSplitProfile",
          "attention_front_handoff_wall_split_profile_ns",
          "attention_front_handoff_wall_split_profile_enabled",
      ],
      [
          "--attention-front-handoff-wall-split-profile",
          "speedup_claims_allowed\":true",
      ],
  )


def _engine_markers(engine_text: str, header_text: str) -> dict[str, Any]:
  combined = engine_text + "\n" + header_text
  return _marker_state(
      combined,
      [
          "IQ36_ATTENTION_FRONT_HANDOFF_WALL_SPLIT_PROFILE",
          "RunResidentPackedQ4X8ThenResidentResidualRmsNorm",
          "handoff_setup_wall_ns",
          "handoff_residual_input_write_wall_ns",
          "handoff_matvec_wall_ns",
          "handoff_residual_rmsnorm_args_wall_ns",
          "handoff_residual_rmsnorm_enqueue_finish_wall_ns",
          "handoff_event_profile_wall_ns",
          "handoff_residual_read_wall_ns",
          "handoff_normalized_read_wall_ns",
          "handoff_alias_wall_ns",
          "handoff_release_wall_ns",
          "clEnqueueWriteBuffer(Q4 resident RMSNorm residual input)",
          "clFinish(Q4 resident RMSNorm)",
          "clEnqueueReadBuffer(Q4 resident RMSNorm residual)",
          "clEnqueueReadBuffer(Q4 resident RMSNorm normalized)",
      ],
  )


def _manifest_checks(result: dict[str, Any], generate_dir: Path) -> dict[str, bool]:
  return {
      "generate_only": result.get("generate_only") is True,
      "attention_front_handoff_wall_split_profile": (
          result.get("attention_front_handoff_wall_split_profile") is True),
      "opencl_no_queue_profiling": (
          result.get("opencl_no_queue_profiling") is True),
      "decode_tokens_eight": result.get("decode_tokens") == 8,
      "accepted_frontier_stack_present": (
          result.get("shared_q4_runner") is True
          and result.get("resident_q4_weights") is True
          and result.get("resident_selected_q4_experts") is True
          and result.get("resident_selected_q6_experts") is True
          and result.get("resident_selected_q6_sorted_cache") is True
          and result.get("resident_selected_q6_rowstripe") is True
          and result.get("resident_selected_cache_topk") == 16
          and result.get("resident_shared_q6_down") is True
          and result.get("resident_full_attention_v_q6") is True
          and result.get("resident_linear_q6_qkv") is True
          and result.get("resident_q4_cpu_order_z") is True
          and result.get("resident_linear_conv_weights") is True
          and result.get("resident_linear_state") is True
          and result.get("resident_postconv_delta_handoff") is True
          and result.get("resident_norm_weights") is True
          and result.get("resident_gate_up_swiglu_handoff") is True
          and result.get("resident_attention_front_handoff") is True
          and result.get("resident_full_core_attention_front_handoff") is True
          and result.get("gpu_router") is True
          and result.get("gpu_lm_head_q6") is True
      ),
      "no_smoke_json": not (generate_dir / "smoke.json").exists(),
  }


def compute(args: argparse.Namespace) -> dict[str, Any]:
  frontier = _load_json(args.frontier)
  routes = _load_json(args.routes)
  rejected = _load_json(args.rejected)
  seq171 = _load_json(args.seq171)
  result_path = args.generate_dir / "result.json"
  cpp_path = args.generate_dir / "r2_gpu_decode_smoke.cpp"
  result = _load_json(result_path)
  decode_source = _read(args.decode_source)
  engine_source = _read(args.engine_source)
  header_source = _read(args.header_source)
  generated_cpp = _read(cpp_path)
  rejected_rows = rejected.get("rejected")
  rejected_count = len(rejected_rows) if isinstance(rejected_rows, list) else 0

  frontier_state = _frontier_state(frontier)
  decode_checks = _decode_markers(decode_source)
  engine_checks = _engine_markers(engine_source, header_source)
  generated_checks = _decode_markers(generated_cpp)
  manifest_checks = _manifest_checks(result, args.generate_dir)

  checks = [
      {
          "name": "seq171_selected_this_source_gate",
          "pass": (
              seq171.get("required_checks_passed") is True
              and seq171.get("selected_next_route")
              == "resident_attention_front_handoff_wall_split_source_gate"
              and _has_candidate(
                  routes,
                  171,
                  "select_resident_attention_front_handoff_wall_split_source_gate",
              )
              and _has_switch(
                  routes,
                  "select_resident_attention_front_handoff_wall_split_source_gate",
                  171,
              )
          ),
          "detail": {
              "seq171_disposition": seq171.get("disposition"),
              "seq171_selected_next_route": seq171.get("selected_next_route"),
          },
      },
      {
          "name": "decode_source_handoff_wall_split_markers_present_default_off",
          "pass": decode_checks["pass"],
          "detail": decode_checks,
      },
      {
          "name": "engine_source_handoff_wall_split_markers_present",
          "pass": engine_checks["pass"],
          "detail": engine_checks,
      },
      {
          "name": "generated_cpp_handoff_wall_split_markers_present_default_off",
          "pass": generated_checks["pass"],
          "detail": generated_checks,
      },
      {
          "name": "generate_only_manifest_is_not_token_row",
          "pass": all(manifest_checks.values()),
          "detail": manifest_checks,
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
  required = all(bool(row.get("pass")) for row in checks)
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "inputs": {
          "frontier": _rel(args.frontier),
          "routes": _rel(args.routes),
          "rejected": _rel(args.rejected),
          "rejected_route_count": rejected_count,
          "seq171_route_gate": _rel(args.seq171),
          "decode_source": _rel(args.decode_source),
          "decode_source_sha256": _sha256(args.decode_source),
          "engine_source": _rel(args.engine_source),
          "engine_source_sha256": _sha256(args.engine_source),
          "header_source": _rel(args.header_source),
          "header_source_sha256": _sha256(args.header_source),
          "generate_only_result": _rel(result_path),
          "generated_cpp": _rel(cpp_path),
          "generated_cpp_sha256": _sha256(cpp_path),
      },
      "frontier": frontier_state,
      "decode_source": decode_checks,
      "engine_source": engine_checks,
      "generated": generated_checks,
      "generate_manifest_checks": manifest_checks,
      "checks": checks,
      "required_checks_passed": required,
      "decode_probe_allowed": False,
      "component_probe_allowed": False,
      "target_compile_required_before_decode": True,
      "speedup_claims_allowed": False,
      "disposition": (
          "accept_resident_attention_front_handoff_wall_split_source_wiring"
          if required else
          "reject_resident_attention_front_handoff_wall_split_source_wiring"
      ),
      "selected_next_route": (
          "resident_attention_front_handoff_wall_split_target_compile_gate"
          if required else
          "resident_attention_front_handoff_wall_split_source_fix_gate"
      ),
      "next_route_reason": (
          "Default-off attention-front handoff wall-split source wiring now "
          "records setup, residual input write, matvec, residual/RMSNorm args, "
          "residual/RMSNorm enqueue+finish, event profile, residual read, "
          "normalized read, alias, and release wall buckets inside "
          "RunResidentPackedQ4X8ThenResidentResidualRmsNorm. Target compile "
          "is required before any profile explore."
          if required else
          "The attention-front handoff wall-split source contract is incomplete; "
          "fix source and generate-only evidence before compile or decode."
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
  lines = [
      "# Resident Attention-Front Handoff Wall-Split Source Gate",
      "",
      f"- required_checks_passed: `{str(payload['required_checks_passed']).lower()}`",
      f"- disposition: `{payload['disposition']}`",
      f"- selected_next_route: `{payload['selected_next_route']}`",
      f"- decode_probe_allowed: `{str(payload['decode_probe_allowed']).lower()}`",
      f"- target_compile_required_before_decode: `{str(payload['target_compile_required_before_decode']).lower()}`",
      f"- failed_checks: `{failed}`",
      "",
      payload["next_route_reason"],
      "",
      "This is source/generate-only evidence. It does not claim speed.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--frontier", type=Path, default=DEFAULT_FRONTIER)
  parser.add_argument("--routes", type=Path, default=DEFAULT_ROUTES)
  parser.add_argument("--rejected", type=Path, default=DEFAULT_REJECTED)
  parser.add_argument("--decode-source", type=Path, default=DEFAULT_DECODE)
  parser.add_argument("--engine-source", type=Path, default=DEFAULT_ENGINE)
  parser.add_argument("--header-source", type=Path, default=DEFAULT_HEADER)
  parser.add_argument("--seq171", type=Path, default=DEFAULT_SEQ171)
  parser.add_argument("--generate-dir", type=Path, default=DEFAULT_GENERATE_DIR)
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
      "out_dir": _rel(args.out_dir),
  }, sort_keys=True))
  return 0 if payload["required_checks_passed"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
