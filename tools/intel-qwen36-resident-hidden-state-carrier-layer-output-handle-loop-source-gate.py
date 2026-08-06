#!/usr/bin/env python3
"""Audit the carrier layer-output handle-loop source wiring.

This is source/generate-only evidence. It verifies the default-off carrier path
can carry intermediate layer outputs by resident handle, while preserving host
readback at full-attention QK/V and final LM-head boundaries. It does not run a
token row and does not create speed evidence.
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
    "intel-qwen36-resident-hidden-state-carrier-layer-output-handle-loop-"
    "source-gate-v0"
)

DEFAULT_FRONTIER = ACTIVE / "frontier.json"
DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_DECODE_SOURCE = ROOT / "tools/intel-qwen36-r2-gpu-decode-smoke.py"
DEFAULT_SEQ126 = (
    ROOT
    / "output/resident-hidden-state-carrier-layer-output-handle-loop-contract-gate-20260707Tseq126Z/metrics.json"
)
DEFAULT_GENERATE_DIR = (
    ROOT
    / "output/resident-hidden-state-carrier-layer-output-handle-loop-generate-only-20260707Tseq127Z"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "output/resident-hidden-state-carrier-layer-output-handle-loop-source-gate-20260707Tseq127Z"
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
      "soft_reflection_breached": no_progress.get("soft_reflection_breached"),
      "hard_stall_breached": no_progress.get("hard_stall_breached"),
  }


def _source_checks(text: str, *, generated: bool) -> dict[str, Any]:
  rows = [
      _present(
          text,
          "layer_output_loop_flag_present",
          "resident_hidden_state_carrier_layer_output_handle_loop",
          regex=False,
      ),
      _present(
          text,
          "layer_output_loop_env_present",
          "IQ36_RESIDENT_HIDDEN_STATE_CARRIER_LAYER_OUTPUT_HANDLE_LOOP",
          regex=False,
      ),
      _present(
          text,
          "keep_prev_handle_includes_loop",
          r"DecodeKeepPrevLayerOutputHandle\(\).*?"
          r"g_decode_resident_hidden_state_carrier_layer_output_handle_loop",
      ),
      _present(
          text,
          "carrier_loop_active_helper_present",
          r"bool DecodeCarrierLayerOutputHandleLoopActive\(\) \{\s*"
          r"return g_decode_resident_hidden_state_carrier_layer_output_handle_loop &&\s*"
          r"g_decode_resident_hidden_state_carrier_enabled;\s*\}",
      ),
      _present(
          text,
          "final_layer_readback_preserved",
          r"DecodeCarrierLayerOutputReadbackRequired\(int layer\).*?"
          r"layer \+ 1 >= kLayerCount",
      ),
      _present(
          text,
          "rmsnorm_wrapper_has_readback_flag",
          r"RunGpuLayerInputRmsNorm\([^)]*bool readback_output = true",
      ),
      _present(
          text,
          "rmsnorm_resident_path_passes_readback",
          r"RunRmsNormHiddenResidentInputResidentWeight\([^;]*readback_output",
      ),
      _present(
          text,
          "rmsnorm_accepts_resident_input_without_host_residual",
          r"residual_input\.size\(\) == kHiddenSize \|\| resident_input_handle != 0",
      ),
      _present(
          text,
          "preconv_accepts_resident_attn_norm_handle",
          r"attn_norm\.size\(\) == kHiddenSize \|\|\s*"
          r"\(use_shared_q8_preconv && attn_norm_handle != 0\)",
      ),
      _present(
          text,
          "attention_front_accepts_resident_residual_handle",
          r"residual_input\.size\(\) == kHiddenSize \|\|\s*"
          r"resident_residual_input_handle != 0",
      ),
      _present(
          text,
          "attention_front_skips_host_diff_when_resident_only",
          r"if \(residual_input\.size\(\) == run\.attn_residual\.size\(\)\).*?"
          r"run\.linear_attn_out\.clear\(\)",
      ),
      _present(
          text,
          "tail_wrapper_has_readback_flag",
          r"RunGpuHybridFfnTail\([^)]*bool readback_layer_output = true",
      ),
      _present(
          text,
          "tail_passes_readback_to_resident_input_primitive",
          r"RunFfnTailFromDownHandlesResidentInputs\([^;]*"
          r"readback_layer_output",
      ),
      _present(
          text,
          "linear_layer_uses_carrier_prev_handle_for_rmsnorm_and_attention",
          r"RunGpuHybridLinearLayerLive\(.*?use_carrier_prev_layer_input.*?"
          r"attention_residual_input_handle.*?RunGpuLayerInputRmsNorm",
      ),
      _present(
          text,
          "linear_layer_can_suppress_intermediate_rmsnorm_readback",
          r"readback_layer_input_norm\s*=.*?!\(use_carrier_prev_layer_input",
      ),
      _present(
          text,
          "linear_z_fallback_skips_carrier_preconv",
          r"!g_decode_linear_preconv_shared_q8 &&\s*!use_carrier_preconv_bundle",
      ),
      _present(
          text,
          "full_attention_uses_carrier_prev_handle_but_keeps_rmsnorm_readback",
          r"RunGpuHybridFullAttentionLayerLive\(.*?use_carrier_prev_layer_input.*?"
          r"RunGpuLayerInputRmsNorm\([^;]*attn_norm_weight_handle, true\)",
      ),
      _present(
          text,
          "full_attention_qk_v_still_host_attn_norm_boundary",
          r"RunGpuFullAttentionQkFront\([^;]*rms_gpu\.attn_norm.*?"
          r"RunGpuFullAttentionVAny\([^;]*rms_gpu\.attn_norm",
      ),
      _present(
          text,
          "carrier_loop_enables_effective_attention_front_handoff",
          r"g_decode_resident_attention_front_handoff\s*=.*?"
          r"resident_hidden_state_carrier_layer_output_handle_loop_enabled",
      ),
      _present(
          text,
          "tail_calls_preserve_final_readback_policy",
          r"DecodeCarrierLayerOutputReadbackRequired\(layer\)",
      ),
  ]
  if not generated:
    rows = rows[:2]
  if not generated:
    rows.extend([
        _present(
            text,
            "python_manifest_records_loop_flag",
            '"resident_hidden_state_carrier_layer_output_handle_loop"',
            regex=False,
        ),
        _present(
            text,
            "python_validation_requires_selected_tail",
            "IQ36_RESIDENT_HIDDEN_STATE_CARRIER_LAYER_OUTPUT_HANDLE_LOOP requires",
            regex=False,
        ),
        _present(
            text,
            "python_validation_rejects_closed_standalone_flags",
            "separate from IQ36_ATTENTION_FRONT_RESIDENT_RESIDUAL_INPUT",
            regex=False,
        ),
    ])
  blockers_absent = [
      _absent(
          text,
          "no_lm_head_input_handle_handoff_claim",
          "RunRmsNormThenResidentRawQ6KFromInputHandle",
          regex=False,
      ),
  ]
  return {
      "present_checks": rows,
      "absent_checks": blockers_absent,
      "present": _all_present(rows),
      "absent": _all_absent(blockers_absent),
  }


def compute(args: argparse.Namespace) -> dict[str, Any]:
  frontier = _load_json(args.frontier)
  routes = _load_json(args.routes)
  seq126 = _load_json(args.seq126)
  decode_source = args.decode_source.read_text(encoding="utf-8")
  result_path = args.generate_dir / "result.json"
  cpp_path = args.generate_dir / "r2_gpu_decode_smoke.cpp"
  result = _load_json(result_path)
  generated_cpp = cpp_path.read_text(encoding="utf-8")
  smoke_path = args.generate_dir / "smoke.json"

  frontier_state = _frontier_state(frontier)
  source = _source_checks(decode_source, generated=False)
  generated = _source_checks(generated_cpp, generated=True)
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
      "ffn_tail_resident_input_false": (
          result.get("ffn_tail_resident_input") is False),
      "attention_front_resident_residual_input_false": (
          result.get("attention_front_resident_residual_input") is False),
      "linear_final_device_q8_handoff_false": (
          result.get("linear_final_device_q8_handoff") is False),
      "resident_full_core_attention_front_handoff_false": (
          result.get("resident_full_core_attention_front_handoff") is False),
      "resident_attention_front_handoff_arg_false_internal_effective": (
          result.get("resident_attention_front_handoff") is False),
      "no_smoke_json": not smoke_path.exists(),
  }

  checks = [
      {
          "name": "seq126_selected_layer_output_loop_source_gate",
          "pass": (
              seq126.get("required_checks_passed") is True
              and seq126.get("selected_next_route")
              == "resident_hidden_state_carrier_layer_output_handle_loop_source_gate"
              and _has_switch(
                  routes,
                  "accept_layer_output_handle_loop_contract_switch_to_source_gate",
                  126,
              )
          ),
          "detail": {
              "seq126_disposition": seq126.get("disposition"),
              "seq126_selected_next_route": seq126.get("selected_next_route"),
          },
      },
      {
          "name": "source_layer_output_loop_contract_present",
          "pass": source["present"] and source["absent"],
          "detail": source,
      },
      {
          "name": "generated_cpp_layer_output_loop_contract_present",
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
          "seq126_contract_gate": _rel(args.seq126),
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
          "accept_resident_hidden_state_carrier_layer_output_handle_loop_source_wiring"
          if required_checks_passed
          else "reject_resident_hidden_state_carrier_layer_output_handle_loop_source_wiring"
      ),
      "selected_next_route": (
          "resident_hidden_state_carrier_layer_output_handle_loop_target_compile_gate"
          if required_checks_passed
          else "resident_hidden_state_carrier_layer_output_handle_loop_source_fix_gate"
      ),
      "next_route_reason": (
          "The default-off carrier loop now wires resident layer-output handles "
          "through next-layer RMSNorm, attention-front residual add/RMSNorm, "
          "preconv, router/FFN input, and selected/shared tail readback control. "
          "Full-attention QK/V and final LM-head still preserve host readback, "
          "so the next admissible unit is target compile only."
          if required_checks_passed
          else "The layer-output handle-loop source contract is incomplete; fix "
               "source/generate-only evidence before compile or decode."
      ),
  }


def write_outputs(metrics: dict[str, Any], out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
  )
  summary = [
      "# Resident Hidden-State Carrier Layer-Output Handle Loop Source Gate",
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
  parser.add_argument("--seq126", type=Path, default=DEFAULT_SEQ126)
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
