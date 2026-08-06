#!/usr/bin/env python3
"""Audit full-core handoff parity for the layer-output handle loop."""

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
    "full-core-parity-source-gate-v0"
)

DEFAULT_FRONTIER = ACTIVE / "frontier.json"
DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_DECODE_SOURCE = ROOT / "tools/intel-qwen36-r2-gpu-decode-smoke.py"
DEFAULT_SEQ130 = (
    ROOT
    / "output/resident-hidden-state-carrier-layer-output-handle-loop-correctness-root-gate-20260707Tseq130Z/metrics.json"
)
DEFAULT_GENERATE_DIR = (
    ROOT
    / "output/resident-hidden-state-carrier-layer-output-handle-loop-full-core-parity-generate-only-20260707Tseq131Z"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "output/resident-hidden-state-carrier-layer-output-handle-loop-full-core-parity-source-gate-20260707Tseq131Z"
)


def _load_json(path: Path) -> Any:
  return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _num(value: Any) -> float:
  return float(value) if isinstance(value, (int, float)) else 0.0


def _rel(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path)


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


def _source_contract(text: str) -> dict[str, Any]:
  present = [
      _present(
          text,
          "loop_flag_present",
          "IQ36_RESIDENT_HIDDEN_STATE_CARRIER_LAYER_OUTPUT_HANDLE_LOOP",
          regex=False,
      ),
      _present(
          text,
          "full_core_handoff_arg_present",
          "--resident-full-core-attention-front-handoff",
          regex=False,
      ),
      _present(
          text,
          "full_core_handoff_receives_resident_residual_handle",
          r"RunFullAttentionCoreGateThenResidentPackedQ4X8ResidualRmsNorm"
          r"\([^;]*attention_residual_input_handle",
      ),
      _present(
          text,
          "full_core_tail_uses_carrier_readback_policy",
          "DecodeCarrierLayerOutputReadbackRequired(layer)",
          regex=False,
      ),
  ]
  absent = [
      _absent(
          text,
          "loop_validation_no_longer_blocks_full_core_handoff",
          r"if \(?args\.resident_full_core_attention_front_handoff\)?[^\n]*\n"
          r"\s*raise SystemExit\([^)]*?IQ36_RESIDENT_HIDDEN_STATE_CARRIER_LAYER_OUTPUT_HANDLE_LOOP",
      ),
  ]
  return {
      "present_checks": present,
      "absent_checks": absent,
      "present": all(row.get("present") is True for row in present),
      "absent": all(row.get("absent") is True for row in absent),
  }


def compute(args: argparse.Namespace) -> dict[str, Any]:
  frontier = _load_json(args.frontier)
  routes = _load_json(args.routes)
  seq130 = _load_json(args.seq130)
  decode_source = args.decode_source.read_text(encoding="utf-8")
  result_path = args.generate_dir / "result.json"
  cpp_path = args.generate_dir / "r2_gpu_decode_smoke.cpp"
  result = _load_json(result_path)
  generated_cpp = cpp_path.read_text(encoding="utf-8")
  smoke_path = args.generate_dir / "smoke.json"

  frontier_state = _frontier_state(frontier)
  source = _source_contract(decode_source)
  generated = _source_contract(generated_cpp)
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
      "resident_full_core_attention_front_handoff": (
          result.get("resident_full_core_attention_front_handoff") is True),
      "resident_attention_front_handoff_arg_false_internal_effective": (
          result.get("resident_attention_front_handoff") is False),
      "ffn_tail_resident_input_false": (
          result.get("ffn_tail_resident_input") is False),
      "linear_final_device_q8_handoff_false": (
          result.get("linear_final_device_q8_handoff") is False),
      "no_smoke_json": not smoke_path.exists(),
  }

  checks = [
      {
          "name": "seq130_selected_full_core_parity_source_gate",
          "pass": (
              seq130.get("required_checks_passed") is True
              and seq130.get("selected_next_route")
              == "resident_hidden_state_carrier_layer_output_handle_loop_full_core_parity_source_gate"
              and _has_switch(
                  routes,
                  "bind_layer_output_loop_root_switch_to_full_core_parity_source_gate",
                  130,
              )
          ),
      },
      {
          "name": "source_full_core_parity_contract_present",
          "pass": source["present"] and source["absent"],
          "detail": source,
      },
      {
          "name": "generated_cpp_full_core_parity_contract_present",
          "pass": generated["present"] and generated["absent"],
          "detail": generated,
      },
      {
          "name": "generate_only_manifest_is_parity_source_gate_not_decode_row",
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
          "seq130_correctness_root_gate": _rel(args.seq130),
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
          "accept_layer_output_loop_full_core_parity_source_wiring"
          if required_checks_passed
          else "reject_layer_output_loop_full_core_parity_source_wiring"
      ),
      "selected_next_route": (
          "resident_hidden_state_carrier_layer_output_handle_loop_full_core_parity_target_compile_gate"
          if required_checks_passed
          else "resident_hidden_state_carrier_layer_output_handle_loop_full_core_parity_source_fix_gate"
      ),
      "next_route_reason": (
          "The layer-output handle-loop source now allows the accepted full-core "
          "attention-front handoff path, keeps carrier readback policy in that "
          "branch, and generated manifest parity is source-only with no token "
          "row. Target compile is required before another decode row."
          if required_checks_passed
          else "The full-core parity source contract is incomplete; fix source "
               "or generate-only evidence before compile or decode."
      ),
  }


def write_outputs(metrics: dict[str, Any], out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
  )
  lines = [
      "# Resident Hidden-State Carrier Layer-Output Handle Loop Full-Core Parity Source Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- target_compile_required_before_decode: `{str(metrics['target_compile_required_before_decode']).lower()}`",
      "",
      "## Summary",
      "",
      metrics["next_route_reason"],
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--frontier", type=Path, default=DEFAULT_FRONTIER)
  parser.add_argument("--routes", type=Path, default=DEFAULT_ROUTES)
  parser.add_argument("--decode-source", type=Path, default=DEFAULT_DECODE_SOURCE)
  parser.add_argument("--seq130", type=Path, default=DEFAULT_SEQ130)
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
