#!/usr/bin/env python3
"""Audit full-attention QK/V resident input-handle source wiring.

This is source/generate-only evidence only. It verifies the default-off carrier
path can feed the resident attention-norm handle into full-attention Q, K, and V
projection Q8 quantization, while keeping Q/K/V host outputs visible for the
existing q/k RMSNorm, rope, K/V history, diagnostics, and full-core path.
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
    "intel-qwen36-resident-hidden-state-carrier-full-attention-qkv-handle-"
    "source-gate-v0"
)

DEFAULT_FRONTIER = ACTIVE / "frontier.json"
DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_DECODE_SOURCE = ROOT / "tools/intel-qwen36-r2-gpu-decode-smoke.py"
DEFAULT_SEQ136 = (
    ROOT
    / "output/resident-hidden-state-carrier-full-attention-qkv-handle-contract-gate-20260707Tseq136Z/metrics.json"
)
DEFAULT_GENERATE_DIR = (
    ROOT
    / "output/resident-hidden-state-carrier-full-attention-qkv-handle-generate-only-20260707Tseq137Z"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "output/resident-hidden-state-carrier-full-attention-qkv-handle-source-gate-20260707Tseq137Z"
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


def _source_checks(text: str) -> dict[str, Any]:
  rows = [
      _present(
          text,
          "python_env_gate_present",
          "IQ36_RESIDENT_HIDDEN_STATE_CARRIER_FULL_ATTENTION_QKV_HANDLE",
          regex=False,
      ),
      _present(
          text,
          "python_args_flag_present",
          "resident_hidden_state_carrier_full_attention_qkv_handle",
          regex=False,
      ),
      _present(
          text,
          "python_validation_requires_layer_output_loop",
          r"FULL_ATTENTION_QKV_HANDLE requires\s*\"\s*"
          r"\"IQ36_RESIDENT_HIDDEN_STATE_CARRIER_LAYER_OUTPUT_HANDLE_LOOP",
      ),
      _present(
          text,
          "python_validation_requires_full_core_handoff",
          r"FULL_ATTENTION_QKV_HANDLE requires\s*\"\s*"
          r"\"--resident-full-core-attention-front-handoff",
      ),
      _present(
          text,
          "python_validation_requires_resident_v_q6",
          r"FULL_ATTENTION_QKV_HANDLE requires\s*\"\s*"
          r"\"--resident-full-attention-v-q6",
      ),
      _present(
          text,
          "manifest_records_qkv_handle_flag",
          '"resident_hidden_state_carrier_full_attention_qkv_handle"',
          regex=False,
      ),
      _present(
          text,
          "run_env_propagates_qkv_handle_flag",
          r"env_parts[\s\S]*?"
          r"IQ36_RESIDENT_HIDDEN_STATE_CARRIER_FULL_ATTENTION_QKV_HANDLE",
      ),
      _present(
          text,
          "qk_input_handle_helper_source_present",
          "RunGpuFullAttentionQkFrontFromInputHandle",
          regex=False,
      ),
      _present(
          text,
          "v_input_handle_helper_source_present",
          "RunGpuFullAttentionVAnyFromInputHandle",
          regex=False,
      ),
      _present(
          text,
          "live_full_attention_uses_carrier_attention_norm_handle",
          r"use_carrier_full_attention_qkv_handle.*?"
          r"g_decode_resident_hidden_state_carrier\.attention_norm_handle",
      ),
  ]
  return {
      "present_checks": rows,
      "present": _all_present(rows),
  }


def _generated_checks(text: str) -> dict[str, Any]:
  present = [
      _present(
          text,
          "global_bool_present",
          "bool g_decode_resident_hidden_state_carrier_full_attention_qkv_handle = false;",
          regex=False,
      ),
      _present(
          text,
          "env_parse_present",
          "IQ36_RESIDENT_HIDDEN_STATE_CARRIER_FULL_ATTENTION_QKV_HANDLE",
          regex=False,
      ),
      _present(
          text,
          "stdout_field_present",
          "resident_hidden_state_carrier_full_attention_qkv_handle_enabled",
          regex=False,
      ),
      _present(
          text,
          "qk_input_handle_helper_present",
          r"FullAttentionQkRun RunGpuFullAttentionQkFrontFromInputHandle",
      ),
      _present(
          text,
          "qk_helper_uses_device_q8_q4_input_handle",
          r"RunGpuFullAttentionQkFrontFromInputHandle[\s\S]*?"
          r"RunF32InputHandleDeviceQ8ThenResidentPackedQ4X8",
      ),
      _present(
          text,
          "qk_helper_keeps_host_q_and_k_outputs",
          r"RunGpuFullAttentionQkFrontFromInputHandle[\s\S]*?"
          r"run\.q_full = q\.output;[\s\S]*?run\.k_raw = k\.output;",
      ),
      _present(
          text,
          "v_q6_input_handle_helper_present",
          r"FullAttentionVQ6Run RunGpuFullAttentionVQ6ResidentInputHandle",
      ),
      _present(
          text,
          "v_q6_helper_uses_device_q8_q6_input_handle",
          r"RunGpuFullAttentionVQ6ResidentInputHandle[\s\S]*?"
          r"RunF32InputHandleDeviceQ8ThenResidentRawQ6K",
      ),
      _present(
          text,
          "v_any_input_handle_helper_present",
          r"FullAttentionVQ6Run RunGpuFullAttentionVAnyFromInputHandle",
      ),
      _present(
          text,
          "v_q4_helper_uses_device_q8_q4_input_handle",
          r"RunGpuFullAttentionVAnyFromInputHandle[\s\S]*?"
          r"RunF32InputHandleDeviceQ8ThenResidentPackedQ4X8",
      ),
      _present(
          text,
          "v_helpers_keep_host_v_output",
          r"RunGpuFullAttentionVQ6ResidentInputHandle[\s\S]*?"
          r"run\.v = projected\.output;[\s\S]*?"
          r"RunGpuFullAttentionVAnyFromInputHandle[\s\S]*?"
          r"run\.v = projected\.output;",
      ),
      _present(
          text,
          "live_qk_consumes_carrier_attention_norm_handle",
          r"use_carrier_full_attention_qkv_handle[\s\S]*?"
          r"RunGpuFullAttentionQkFrontFromInputHandle\([\s\S]*?"
          r"g_decode_resident_hidden_state_carrier\.attention_norm_handle",
      ),
      _present(
          text,
          "live_v_consumes_carrier_attention_norm_handle",
          r"use_carrier_full_attention_qkv_handle[\s\S]*?"
          r"RunGpuFullAttentionVAnyFromInputHandle\([\s\S]*?"
          r"g_decode_resident_hidden_state_carrier\.attention_norm_handle",
      ),
      _present(
          text,
          "host_qk_fallback_preserved",
          r": RunGpuFullAttentionQkFront\([\s\S]*?rms_gpu\.attn_norm",
      ),
      _present(
          text,
          "host_v_fallback_preserved",
          r": RunGpuFullAttentionVAny\([\s\S]*?rms_gpu\.attn_norm",
      ),
  ]
  absent = [
      _absent(
          text,
          "no_direct_full_attention_core_handle_claim",
          "RunFullAttentionCoreGateFromResidentHandles",
          regex=False,
      ),
  ]
  return {
      "present_checks": present,
      "absent_checks": absent,
      "present": _all_present(present),
      "absent": _all_absent(absent),
  }


def compute(args: argparse.Namespace) -> dict[str, Any]:
  frontier = _load_json(args.frontier)
  routes = _load_json(args.routes)
  seq136 = _load_json(args.seq136)
  decode_source = args.decode_source.read_text(encoding="utf-8")
  result_path = args.generate_dir / "result.json"
  cpp_path = args.generate_dir / "r2_gpu_decode_smoke.cpp"
  result = _load_json(result_path)
  generated_cpp = cpp_path.read_text(encoding="utf-8")
  smoke_path = args.generate_dir / "smoke.json"

  frontier_state = _frontier_state(frontier)
  source = _source_checks(decode_source)
  generated = _generated_checks(generated_cpp)
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
          "name": "seq136_selected_full_attention_qkv_source_gate",
          "pass": (
              seq136.get("required_checks_passed") is True
              and seq136.get("selected_next_route")
              == "resident_hidden_state_carrier_full_attention_qkv_handle_source_gate"
              and _has_switch(
                  routes,
                  "accept_full_attention_qkv_handle_contract_switch_to_source_gate",
                  136,
              )
          ),
          "detail": {
              "seq136_disposition": seq136.get("disposition"),
              "seq136_selected_next_route": seq136.get("selected_next_route"),
          },
      },
      {
          "name": "source_full_attention_qkv_handle_contract_present",
          "pass": source["present"],
          "detail": source,
      },
      {
          "name": "generated_cpp_full_attention_qkv_handle_contract_present",
          "pass": generated["present"] and generated["absent"],
          "detail": generated,
      },
      {
          "name": "generate_only_manifest_is_source_gate_not_decode_row",
          "pass": all(manifest_checks.values()),
          "detail": manifest_checks,
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
          "decode_source": _rel(args.decode_source),
          "decode_source_sha256": _sha256(args.decode_source),
          "seq136_contract_gate": _rel(args.seq136),
          "generate_only_result": _rel(result_path),
          "generated_cpp": _rel(cpp_path),
          "generated_cpp_sha256": _sha256(cpp_path),
      },
      "frontier": frontier_state,
      "source": source,
      "generated": generated,
      "generate_manifest_checks": manifest_checks,
      "checks": checks,
      "required_checks_passed": required_checks_passed,
      "decode_probe_allowed": False,
      "component_probe_allowed": False,
      "target_compile_required_before_decode": True,
      "speedup_claims_allowed": False,
      "disposition": (
          "accept_full_attention_qkv_handle_source_wiring"
          if required_checks_passed
          else "reject_full_attention_qkv_handle_source_wiring"
      ),
      "selected_next_route": (
          "resident_hidden_state_carrier_full_attention_qkv_handle_target_compile_gate"
          if required_checks_passed
          else "resident_hidden_state_carrier_full_attention_qkv_handle_source_fix_gate"
      ),
      "next_route_reason": (
          "Default-off full-attention QK/V source wiring now consumes the "
          "carrier attention-norm handle through device-Q8 input-handle "
          "projection helpers, while preserving Q/K/V host outputs and host "
          "fallback. The next admissible unit is target compile only."
          if required_checks_passed
          else "The full-attention QK/V handle source contract is incomplete; "
               "fix source/generate-only evidence before compile or decode."
      ),
  }


def write_outputs(metrics: dict[str, Any], out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
  )
  summary = [
      "# Resident Hidden-State Carrier Full-Attention QK/V Handle Source Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- decode_probe_allowed: `{str(metrics['decode_probe_allowed']).lower()}`",
      f"- target_compile_required_before_decode: `{str(metrics['target_compile_required_before_decode']).lower()}`",
      "",
      "## Summary",
      "",
      metrics["next_route_reason"],
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(summary), encoding="utf-8")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--frontier", type=Path, default=DEFAULT_FRONTIER)
  parser.add_argument("--routes", type=Path, default=DEFAULT_ROUTES)
  parser.add_argument("--decode-source", type=Path, default=DEFAULT_DECODE_SOURCE)
  parser.add_argument("--seq136", type=Path, default=DEFAULT_SEQ136)
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
