#!/usr/bin/env python3
"""Audit full-attention core/history resident-boundary source scaffolding.

This is source/generate-only evidence. It verifies a default-off contract shape
for replacing the host Q/K norm, RoPE, K/V history, and full-core input-vector
boundary without launching a token row.
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
    "source-gate-v0"
)

DEFAULT_FRONTIER = ACTIVE / "frontier.json"
DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_REJECTED = ACTIVE / "rejected-routes.json"
DEFAULT_DECODE_SOURCE = ROOT / "tools/intel-qwen36-r2-gpu-decode-smoke.py"
DEFAULT_SEQ144 = (
    ROOT
    / "output/resident-hidden-state-carrier-full-attention-core-history-contract-gate-20260707Tseq144Z"
    / "metrics.json"
)
DEFAULT_GENERATE_DIR = (
    ROOT
    / "output/resident-hidden-state-carrier-full-attention-core-history-generate-only-20260707Tseq145Z"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "output/resident-hidden-state-carrier-full-attention-core-history-source-gate-20260707Tseq145Z"
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
  return {
      "current_best_tps": _num(anchor.get("current_best_tps")),
      "floor_tps": _num(anchor.get("same_host_vulkan_floor_tps")),
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
      _present(
          text,
          "python_env_gate_present",
          "IQ36_RESIDENT_HIDDEN_STATE_CARRIER_FULL_ATTENTION_CORE_HISTORY_HANDLE",
          regex=False,
      ),
      _present(
          text,
          "python_args_flag_present",
          "resident_hidden_state_carrier_full_attention_core_history_handle",
          regex=False,
      ),
      _present(
          text,
          "run_env_propagates_core_history_flag",
          r"env_parts[\s\S]*?"
          r"IQ36_RESIDENT_HIDDEN_STATE_CARRIER_FULL_ATTENTION_CORE_HISTORY_HANDLE",
      ),
      _present(
          text,
          "validation_requires_qkv_handle",
          r"FULL_ATTENTION_CORE_HISTORY_HANDLE[\s\S]*?"
          r"requires IQ36_RESIDENT_HIDDEN_STATE_CARRIER_FULL_ATTENTION_QKV_HANDLE",
      ),
      _present(
          text,
          "validation_requires_full_core_handoff",
          r"FULL_ATTENTION_CORE_HISTORY_HANDLE[\s\S]*?"
          r"requires --resident-full-core-attention-front-handoff",
      ),
      _present(
          text,
          "manifest_records_core_history_flag",
          '"resident_hidden_state_carrier_full_attention_core_history_handle"',
          regex=False,
      ),
      _present(
          text,
          "source_only_guard_present",
          "full-attention core/history resident boundary is source-gate only",
          regex=False,
      ),
  ]
  return {"present": _all_present(present), "present_checks": present}


def _generated_checks(text: str) -> dict[str, Any]:
  present = [
      _present(
          text,
          "global_bool_present",
          "bool g_decode_resident_hidden_state_carrier_full_attention_core_history_handle = false;",
          regex=False,
      ),
      _present(
          text,
          "contract_struct_present",
          r"struct DecodeFullAttentionCoreHistoryResidentBoundaryContract",
      ),
      _present(
          text,
          "contract_requires_q_rope_handle",
          r"DecodeFullAttentionCoreHistoryResidentBoundaryReady[\s\S]*?"
          r"contract\.q_rope_handle != 0",
      ),
      _present(
          text,
          "contract_requires_kv_history_handles",
          r"DecodeFullAttentionCoreHistoryResidentBoundaryReady[\s\S]*?"
          r"contract\.k_history_handle != 0[\s\S]*?"
          r"contract\.v_history_handle != 0",
      ),
      _present(
          text,
          "contract_requires_q_full_handle",
          r"DecodeFullAttentionCoreHistoryResidentBoundaryReady[\s\S]*?"
          r"contract\.q_full_handle != 0",
      ),
      _present(
          text,
          "contract_requires_device_ownership_flags",
          r"DecodeFullAttentionCoreHistoryResidentBoundaryReady[\s\S]*?"
          r"contract\.qk_norm_rope_device_owned[\s\S]*?"
          r"contract\.kv_history_device_owned[\s\S]*?"
          r"contract\.full_core_inputs_device_owned",
      ),
      _present(
          text,
          "env_parse_present",
          "IQ36_RESIDENT_HIDDEN_STATE_CARRIER_FULL_ATTENTION_CORE_HISTORY_HANDLE",
          regex=False,
      ),
      _present(
          text,
          "stdout_field_present",
          "resident_hidden_state_carrier_full_attention_core_history_handle_enabled",
          regex=False,
      ),
      _present(
          text,
          "source_only_runtime_guard_present",
          "full-attention core/history resident boundary is source-gate only",
          regex=False,
      ),
      _present(
          text,
          "host_core_handoff_fallback_still_present",
          r"RunFullAttentionCoreGateThenResidentPackedQ4X8ResidualRmsNorm\(\s*"
          r"rope\.q_rope,\s*k_history_flat,\s*v_history_flat,\s*qk_gpu\.q_full",
      ),
  ]
  absent = [
      _absent(
          text,
          "no_token_decode_execution_claim",
          "RunFullAttentionCoreGateFromResidentHandles",
          regex=False,
      ),
  ]
  return {
      "present": _all_present(present),
      "absent": _all_absent(absent),
      "present_checks": present,
      "absent_checks": absent,
  }


def compute(args: argparse.Namespace) -> dict[str, Any]:
  frontier = _load_json(args.frontier)
  routes = _load_json(args.routes)
  rejected = _load_json(args.rejected)
  seq144 = _load_json(args.seq144)
  decode_source = args.decode_source.read_text(encoding="utf-8")
  result_path = args.generate_dir / "result.json"
  cpp_path = args.generate_dir / "r2_gpu_decode_smoke.cpp"
  result = _load_json(result_path)
  generated_cpp = cpp_path.read_text(encoding="utf-8")
  smoke_path = args.generate_dir / "smoke.json"

  source = _source_checks(decode_source)
  generated = _generated_checks(generated_cpp)
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
          "name": "seq144_selected_core_history_source_gate",
          "pass": (
              seq144.get("required_checks_passed") is True
              and seq144.get("selected_next_route")
              == "resident_hidden_state_carrier_full_attention_core_history_source_gate"
              and _has_switch(
                  routes,
                  "accept_full_attention_core_history_contract_switch_to_source_gate",
                  144,
              )
          ),
      },
      {
          "name": "source_core_history_contract_present",
          "pass": source["present"],
          "detail": source,
      },
      {
          "name": "generated_cpp_core_history_contract_present",
          "pass": generated["present"] and generated["absent"],
          "detail": generated,
      },
      {
          "name": "generate_only_manifest_is_source_gate_not_decode_row",
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
          "seq144_contract_gate": _rel(args.seq144),
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
          "accept_full_attention_core_history_source_wiring"
          if required_checks_passed
          else "reject_full_attention_core_history_source_wiring"
      ),
      "selected_next_route": (
          "resident_hidden_state_carrier_full_attention_core_history_target_compile_gate"
          if required_checks_passed
          else "resident_hidden_state_carrier_full_attention_core_history_source_fix_gate"
      ),
      "next_route_reason": (
          "Default-off core/history source scaffolding is present, records the "
          "contract in generate-only evidence, preserves the host full-core "
          "handoff fallback, and blocks token execution until a real source "
          "implementation is added. The next admissible unit is target compile."
          if required_checks_passed
          else "The core/history source contract is incomplete; fix source and "
               "generate-only evidence before compile or decode."
      ),
  }


def write_outputs(metrics: dict[str, Any], out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
  )
  lines = [
      "# Resident Hidden-State Carrier Full-Attention Core/History Source Gate",
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
  parser.add_argument("--seq144", type=Path, default=DEFAULT_SEQ144)
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
