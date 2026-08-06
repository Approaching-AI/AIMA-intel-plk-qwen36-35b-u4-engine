#!/usr/bin/env python3
"""Audit real full-attention core/history resident-boundary source wiring.

This is source/generate-only evidence. It verifies that the earlier source-only
guard has been replaced by a default-off implementation path for device-owned
Q/K norm+RoPE, K/V history append, and full-core input handles without
launching a token row.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = (
    "intel-qwen36-resident-hidden-state-carrier-full-attention-core-history-"
    "implementation-source-gate-v0"
)

DEFAULT_FRONTIER = ACTIVE / "frontier.json"
DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_REJECTED = ACTIVE / "rejected-routes.json"
DEFAULT_DECODE_SOURCE = ROOT / "tools/intel-qwen36-r2-gpu-decode-smoke.py"
DEFAULT_ENGINE_HEADER = ROOT / "engine/include/intel_qwen36/gpu_q4x8_matvec.hpp"
DEFAULT_ENGINE_SOURCE = ROOT / "engine/src/gpu_q4x8_matvec.cpp"
DEFAULT_OPENCL_SOURCE = ROOT / "engine/gpu/opencl/q4x8_matvec.cl"
DEFAULT_SEQ146 = (
    ROOT
    / "output/resident-hidden-state-carrier-full-attention-core-history-target-compile-gate-20260707Tseq146Z"
    / "metrics.json"
)
DEFAULT_GENERATE_DIR = (
    ROOT
    / "output/resident-hidden-state-carrier-full-attention-core-history-implementation-generate-only-20260707Tseq147Z"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "output/resident-hidden-state-carrier-full-attention-core-history-implementation-source-gate-20260707Tseq147Z"
)


def _load_json(path: Path) -> Any:
  return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _rel(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path)


def _num(value: Any) -> float:
  return float(value) if isinstance(value, (int, float)) else 0.0


def _line_of(text: str, pattern: str, *, regex: bool = True) -> int | None:
  if regex:
    match = re.search(pattern, text, flags=re.S | re.M)
    if match is None:
      return None
    return text.count("\n", 0, match.start()) + 1
  index = text.find(pattern)
  if index < 0:
    return None
  return text.count("\n", 0, index) + 1


def _present(text: str, label: str, pattern: str, *,
             regex: bool = True) -> dict[str, Any]:
  line = _line_of(text, pattern, regex=regex)
  return {"label": label, "present": line is not None, "line": line}


def _absent(text: str, label: str, pattern: str, *,
            regex: bool = True) -> dict[str, Any]:
  line = _line_of(text, pattern, regex=regex)
  return {"label": label, "absent": line is None, "line": line}


def _all_present(rows: list[dict[str, Any]]) -> bool:
  return all(row.get("present") is True for row in rows)


def _all_absent(rows: list[dict[str, Any]]) -> bool:
  return all(row.get("absent") is True for row in rows)


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
  no_progress = frontier.get("no_progress")
  no_progress = no_progress if isinstance(no_progress, dict) else {}
  noise = no_progress.get("noise")
  noise = noise if isinstance(noise, dict) else {}
  return {
      "current_best_tps": _num(anchor.get("current_best_tps")),
      "floor_tps": _num(anchor.get("same_host_vulkan_floor_tps")),
      "noise_rel": _num(noise.get("rel")),
      "runs_since_significant_improvement": no_progress.get(
          "runs_since_significant_improvement"),
      "hard_stall_breached": no_progress.get("hard_stall_breached"),
      "review_recorded_for_current_best": no_progress.get(
          "review_recorded_for_current_best"),
  }


def _rejected_names(rejected: dict[str, Any]) -> set[str]:
  names: set[str] = set()
  for row in rejected.get("rejected", []):
    if isinstance(row, dict) and isinstance(row.get("route"), str):
      names.add(row["route"])
  return names


def _source_checks(text: str) -> dict[str, Any]:
  present = [
      _present(text, "env_gate_present",
               "IQ36_RESIDENT_HIDDEN_STATE_CARRIER_FULL_ATTENTION_CORE_HISTORY_HANDLE",
               regex=False),
      _present(text, "runtime_requires_qkv_carrier",
               "full-attention core/history carrier requires QKV handle carrier",
               regex=False),
      _present(text, "runtime_requires_full_core_handoff",
               "full-attention core/history carrier requires full-core handoff",
               regex=False),
      _present(text, "rope_cache_helper_present",
               "DecodeFullAttentionRopeCache", regex=False),
      _present(text, "previous_history_flatten_helper_present",
               "DecodeFlattenFullAttentionPreviousHistory", regex=False),
      _present(text, "qk_norm_rope_runner_call_present",
               "RunFullAttentionQkNormRopeFromHandles", regex=False),
      _present(text, "history_append_runner_call_present",
               "BuildFullAttentionHistoryFromHandle", regex=False),
      _present(text, "full_core_from_handles_runner_call_present",
               "RunFullAttentionCoreGateThenResidentPackedQ4X8ResidualRmsNormFromHandles",
               regex=False),
      _present(text, "stable_resident_projection_clone_present",
               "CloneResidentF32Buffer", regex=False),
      _present(text, "qk_run_carries_q_handle",
               r"struct FullAttentionQkRun[\s\S]*?q_full_handle"),
      _present(text, "qk_run_carries_k_handle",
               r"struct FullAttentionQkRun[\s\S]*?k_raw_handle"),
      _present(text, "v_run_carries_v_handle",
               r"struct FullAttentionVQ6Run[\s\S]*?v_handle"),
  ]
  absent = [
      _absent(text, "source_only_runtime_guard_removed",
              "full-attention core/history resident boundary is source-gate only",
              regex=False),
  ]
  return {
      "present": _all_present(present),
      "absent": _all_absent(absent),
      "present_checks": present,
      "absent_checks": absent,
  }


def _generated_checks(text: str) -> dict[str, Any]:
  present = [
      _present(text, "contract_ready_helper_present",
               "DecodeFullAttentionCoreHistoryResidentBoundaryReady",
               regex=False),
      _present(text, "rope_cache_helper_present",
               "DecodeFullAttentionRopeCache", regex=False),
      _present(text, "qk_norm_rope_runner_call_present",
               "RunFullAttentionQkNormRopeFromHandles", regex=False),
      _present(text, "history_append_runner_call_present",
               "BuildFullAttentionHistoryFromHandle", regex=False),
      _present(text, "full_core_from_handles_runner_call_present",
               "RunFullAttentionCoreGateThenResidentPackedQ4X8ResidualRmsNormFromHandles",
               regex=False),
      _present(text, "stable_resident_projection_clone_present",
               "CloneResidentF32Buffer", regex=False),
      _present(text, "core_history_contract_populated",
               r"core_history_contract\.q_rope_handle[\s\S]*?"
               r"core_history_contract\.k_history_handle[\s\S]*?"
               r"core_history_contract\.v_history_handle[\s\S]*?"
               r"core_history_contract\.q_full_handle"),
      _present(text, "host_full_core_fallback_still_present",
               r"RunFullAttentionCoreGateThenResidentPackedQ4X8ResidualRmsNorm\(\s*"
               r"rope\.q_rope,\s*k_history_flat,\s*v_history_flat,\s*qk_gpu\.q_full"),
      _present(text, "generated_stdout_field_present",
               "resident_hidden_state_carrier_full_attention_core_history_handle_enabled",
               regex=False),
  ]
  absent = [
      _absent(text, "source_only_runtime_guard_removed",
              "full-attention core/history resident boundary is source-gate only",
              regex=False),
  ]
  return {
      "present": _all_present(present),
      "absent": _all_absent(absent),
      "present_checks": present,
      "absent_checks": absent,
  }


def _engine_checks(header: str, source: str, opencl: str) -> dict[str, Any]:
  header_present = [
      _present(header, "header_qk_norm_rope_run",
               "struct GpuFullAttentionQkNormRopeRun", regex=False),
      _present(header, "header_history_append_run",
               "struct GpuFullAttentionHistoryAppendRun", regex=False),
      _present(header, "header_qk_norm_rope_api",
               "RunFullAttentionQkNormRopeFromHandles", regex=False),
      _present(header, "header_history_append_api",
               "BuildFullAttentionHistoryFromHandle", regex=False),
      _present(header, "header_full_core_from_handles_api",
               "RunFullAttentionCoreGateThenResidentPackedQ4X8ResidualRmsNormFromHandles",
               regex=False),
      _present(header, "header_clone_resident_f32_api",
               "CloneResidentF32Buffer", regex=False),
  ]
  source_present = [
      _present(source, "source_kernel_create",
               'CreateNamedKernel("full_attn_qk_norm_rope_f32")',
               regex=False),
      _present(source, "source_qk_norm_rope_impl",
               "GpuFullAttentionQkNormRopeRun RunFullAttentionQkNormRopeFromHandles",
               regex=False),
      _present(source, "source_history_append_impl",
               "GpuFullAttentionHistoryAppendRun BuildFullAttentionHistoryFromHandle",
               regex=False),
      _present(source, "source_resident_core_inputs",
               r"q_rope_handle[\s\S]*?k_history_handle[\s\S]*?"
               r"v_history_handle[\s\S]*?q_full_handle"),
      _present(source, "source_public_full_core_from_handles",
               r"GpuQ4X8MatvecRunner::\s*"
               r"RunFullAttentionCoreGateThenResidentPackedQ4X8ResidualRmsNormFromHandles"),
      _present(source, "source_clone_resident_f32_impl",
               "CloneResidentF32Buffer", regex=False),
  ]
  opencl_present = [
      _present(opencl, "opencl_qk_norm_rope_kernel",
               "__kernel void full_attn_qk_norm_rope_f32", regex=False),
  ]
  return {
      "header_present": _all_present(header_present),
      "source_present": _all_present(source_present),
      "opencl_present": _all_present(opencl_present),
      "header_checks": header_present,
      "source_checks": source_present,
      "opencl_checks": opencl_present,
  }


def compute(args: argparse.Namespace) -> dict[str, Any]:
  frontier = _load_json(args.frontier)
  routes = _load_json(args.routes)
  rejected = _load_json(args.rejected)
  seq146 = _load_json(args.seq146)
  decode_source = args.decode_source.read_text(encoding="utf-8")
  engine_header = args.engine_header.read_text(encoding="utf-8")
  engine_source = args.engine_source.read_text(encoding="utf-8")
  opencl_source = args.opencl_source.read_text(encoding="utf-8")
  result_path = args.generate_dir / "result.json"
  cpp_path = args.generate_dir / "r2_gpu_decode_smoke.cpp"
  result = _load_json(result_path)
  generated_cpp = cpp_path.read_text(encoding="utf-8")
  smoke_path = args.generate_dir / "smoke.json"

  source = _source_checks(decode_source)
  generated = _generated_checks(generated_cpp)
  engine = _engine_checks(engine_header, engine_source, opencl_source)
  frontier_state = _frontier_state(frontier)
  rejected_names = _rejected_names(rejected)
  required_closed = {
      "current_resident_hidden_state_carrier_full_attention_qkv_handle_speed_shape",
      "gpu_full_attention_state_resident_history",
      "gpu_full_attention_flat_history_cache",
      "current_full_core_attention_front_kernel_algorithm_board",
  }
  missing_closed = sorted(required_closed - rejected_names)
  manifest_checks = {
      "generate_only": result.get("generate_only") is True,
      "resident_hidden_state_carrier": (
          result.get("resident_hidden_state_carrier") is True),
      "resident_hidden_state_carrier_preconv_bundle": (
          result.get("resident_hidden_state_carrier_preconv_bundle") is True),
      "resident_hidden_state_carrier_selected_shared_tail": (
          result.get("resident_hidden_state_carrier_selected_shared_tail") is True),
      "resident_hidden_state_carrier_layer_output_handle_loop": (
          result.get("resident_hidden_state_carrier_layer_output_handle_loop")
          is True),
      "resident_hidden_state_carrier_full_attention_qkv_handle": (
          result.get("resident_hidden_state_carrier_full_attention_qkv_handle")
          is True),
      "resident_hidden_state_carrier_full_attention_core_history_handle": (
          result.get(
              "resident_hidden_state_carrier_full_attention_core_history_handle")
          is True),
      "resident_full_core_attention_front_handoff": (
          result.get("resident_full_core_attention_front_handoff") is True),
      "resident_full_attention_v_q6": (
          result.get("resident_full_attention_v_q6") is True),
      "ffn_tail_resident_input_false": (
          result.get("ffn_tail_resident_input") is False),
      "attention_front_resident_residual_input_false": (
          result.get("attention_front_resident_residual_input") is False),
      "linear_final_device_q8_handoff_false": (
          result.get("linear_final_device_q8_handoff") is False),
      "no_smoke_json": not smoke_path.exists(),
  }
  checks = [
      {
          "name": "seq146_selected_implementation_source_gate",
          "pass": (
              seq146.get("required_checks_passed") is True
              and seq146.get("selected_next_route")
              == "resident_hidden_state_carrier_full_attention_core_history_implementation_source_gate"
              and _has_switch(
                  routes,
                  "accept_full_attention_core_history_compile_switch_to_implementation_source_gate",
                  146,
              )
          ),
      },
      {
          "name": "decode_source_implementation_present_guard_removed",
          "pass": source["present"] and source["absent"],
          "detail": source,
      },
      {
          "name": "engine_implementation_apis_present",
          "pass": (
              engine["header_present"]
              and engine["source_present"]
              and engine["opencl_present"]
          ),
          "detail": engine,
      },
      {
          "name": "generated_cpp_implementation_present_guard_removed",
          "pass": generated["present"] and generated["absent"],
          "detail": generated,
      },
      {
          "name": "generate_only_manifest_is_implementation_not_decode_row",
          "pass": all(manifest_checks.values()),
          "detail": manifest_checks,
      },
      {
          "name": "closed_boards_not_reopened",
          "pass": not missing_closed,
          "detail": {"missing_closed_routes": missing_closed},
      },
      {
          "name": "frontier_still_below_floor_no_speed_claim",
          "pass": frontier_state["current_best_tps"] < frontier_state["floor_tps"],
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
          "decode_source": _rel(args.decode_source),
          "decode_source_sha256": _sha256(args.decode_source),
          "engine_header": _rel(args.engine_header),
          "engine_header_sha256": _sha256(args.engine_header),
          "engine_source": _rel(args.engine_source),
          "engine_source_sha256": _sha256(args.engine_source),
          "opencl_source": _rel(args.opencl_source),
          "opencl_source_sha256": _sha256(args.opencl_source),
          "seq146_target_compile_gate": _rel(args.seq146),
          "generate_only_result": _rel(result_path),
          "generated_cpp": _rel(cpp_path),
          "generated_cpp_sha256": _sha256(cpp_path),
      },
      "frontier": frontier_state,
      "source": source,
      "engine": engine,
      "generated": generated,
      "generate_manifest_checks": manifest_checks,
      "checks": checks,
      "required_checks_passed": required_checks_passed,
      "decode_probe_allowed": False,
      "component_probe_allowed": False,
      "target_compile_required_before_decode": True,
      "speedup_claims_allowed": False,
      "disposition": (
          "accept_full_attention_core_history_implementation_source_wiring"
          if required_checks_passed
          else "reject_full_attention_core_history_implementation_source_wiring"
      ),
      "selected_next_route": (
          "resident_hidden_state_carrier_full_attention_core_history_implementation_target_compile_gate"
          if required_checks_passed
          else "resident_hidden_state_carrier_full_attention_core_history_implementation_source_fix_gate"
      ),
      "next_route_reason": (
          "The source-only guard is gone and the generated path now carries "
          "device-owned Q/K norm+RoPE, K/V history, and full-core handles. "
          "The next admissible unit is a target compile gate before any "
          "correctness or speed decode row."
          if required_checks_passed
          else "The implementation source evidence is incomplete; fix source "
               "and regenerate before target compile or decode."
      ),
  }


def write_outputs(metrics: dict[str, Any], out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
  )
  lines = [
      "# Resident Hidden-State Carrier Full-Attention Core/History Implementation Source Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- decode_probe_allowed: `{str(metrics['decode_probe_allowed']).lower()}`",
      f"- target_compile_required_before_decode: `{str(metrics['target_compile_required_before_decode']).lower()}`",
      "",
      metrics["next_route_reason"],
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--frontier", type=Path, default=DEFAULT_FRONTIER)
  parser.add_argument("--routes", type=Path, default=DEFAULT_ROUTES)
  parser.add_argument("--rejected", type=Path, default=DEFAULT_REJECTED)
  parser.add_argument("--decode-source", type=Path, default=DEFAULT_DECODE_SOURCE)
  parser.add_argument("--engine-header", type=Path, default=DEFAULT_ENGINE_HEADER)
  parser.add_argument("--engine-source", type=Path, default=DEFAULT_ENGINE_SOURCE)
  parser.add_argument("--opencl-source", type=Path, default=DEFAULT_OPENCL_SOURCE)
  parser.add_argument("--seq146", type=Path, default=DEFAULT_SEQ146)
  parser.add_argument("--generate-dir", type=Path, default=DEFAULT_GENERATE_DIR)
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
