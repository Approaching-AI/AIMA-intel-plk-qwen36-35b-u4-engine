#!/usr/bin/env python3
"""Gate the carrier layer-output handle-loop contract before source edits.

This is source/design route-control evidence only. It consumes seq125, the
existing no-readback primitive gates, and the generated carrier source. It does
not run decode and does not create speed evidence.
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
    "contract-gate-v0"
)

DEFAULT_FRONTIER = ACTIVE / "frontier.json"
DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_SEQ125 = (
    ROOT
    / "output/resident-hidden-state-carrier-tail-growth-root-gate-20260707Tseq125Z/metrics.json"
)
DEFAULT_SEQ55 = ROOT / "output/tail-no-readback-contract-gate-20260706Tseq55Z/metrics.json"
DEFAULT_SEQ56 = ROOT / "output/rmsnorm-no-readback-contract-gate-20260706Tseq56Z/metrics.json"
DEFAULT_SEQ57 = ROOT / "output/f32matvec-no-readback-contract-gate-20260706Tseq57Z/metrics.json"
DEFAULT_SEQ59 = ROOT / "output/device-q8-q4-no-readback-contract-gate-20260706Tseq59Z/metrics.json"
DEFAULT_SEQ60 = ROOT / "output/device-q8-q6-no-readback-contract-gate-20260706Tseq60Z/metrics.json"
DEFAULT_SEQ61 = ROOT / "output/ffn-tail-resident-input-contract-gate-20260706Tseq61Z/metrics.json"
DEFAULT_SEQ64 = ROOT / "output/attention-front-resident-residual-gate-20260706Tseq64Z/metrics.json"
DEFAULT_SEQ119 = (
    ROOT
    / "output/resident-hidden-state-carrier-preconv-bundle-source-gate-20260707Tseq119Z/metrics.json"
)
DEFAULT_SEQ121 = (
    ROOT
    / "output/resident-hidden-state-carrier-selected-shared-tail-source-gate-20260707Tseq121Z/metrics.json"
)
DEFAULT_DECODE_SOURCE = ROOT / "tools/intel-qwen36-r2-gpu-decode-smoke.py"
DEFAULT_GENERATED_CPP = (
    ROOT
    / "output/resident-hidden-state-carrier-selected-shared-tail-generate-only-20260707Tseq121Z/r2_gpu_decode_smoke.cpp"
)
DEFAULT_ENGINE_SOURCE = ROOT / "engine/src/gpu_q4x8_matvec.cpp"
DEFAULT_ENGINE_HEADER = ROOT / "engine/include/intel_qwen36/gpu_q4x8_matvec.hpp"
DEFAULT_OUT_DIR = (
    ROOT
    / "output/resident-hidden-state-carrier-layer-output-handle-loop-contract-gate-20260707Tseq126Z"
)


def _load_json(path: Path) -> Any:
  return json.loads(path.read_text(encoding="utf-8"))


def _load_text(path: Path) -> str:
  return path.read_text(encoding="utf-8")


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
    match = re.search(pattern, text, flags=re.S)
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
      "soft_reflection_breached": no_progress.get("soft_reflection_breached"),
      "hard_stall_breached": no_progress.get("hard_stall_breached"),
  }


def _metric_ready(payload: dict[str, Any], key: str) -> bool:
  derived = payload.get("derived")
  derived = derived if isinstance(derived, dict) else {}
  if key in derived:
    return derived.get(key) is True
  return False


def _source_shape(decode_source: str, generated_cpp: str, engine_text: str) -> dict[str, Any]:
  combined = "\n".join([decode_source, generated_cpp, engine_text])
  checks_present = [
      _present(
          generated_cpp,
          "carrier_struct_tracks_layer_input_and_output_handles",
          r"struct ResidentHiddenStateCarrier\s*\{[^}]*layer_input_handle"
          r"[^}]*layer_output_handle",
      ),
      _present(
          generated_cpp,
          "decode_keeps_prev_layer_output_for_carrier_tail",
          r"DecodeKeepPrevLayerOutputHandle\(\)\s*\{[^}]*"
          r"g_decode_resident_hidden_state_carrier_selected_shared_tail",
      ),
      _present(
          generated_cpp,
          "linear_layer_begins_carrier_from_prev_handle",
          r"g_decode_resident_hidden_state_carrier\.BeginLayer\(\s*"
          r"layer,\s*g_decode_prev_layer_output_handle\)",
      ),
      _present(
          generated_cpp,
          "linear_preconv_consumes_carrier_attention_norm_handle",
          r"carrier_preconv_attn_norm_handle\s*=\s*"
          r"use_carrier_preconv_bundle\s*\?\s*"
          r"g_decode_resident_hidden_state_carrier\.attention_norm_handle",
      ),
      _present(
          generated_cpp,
          "router_can_consume_ffn_norm_handle",
          r"RunResidentF32MatvecFromInputHandle\(\s*router_handle,\s*"
          r"ffn_input_handle",
      ),
      _present(
          generated_cpp,
          "carrier_tail_uses_resident_input_primitive",
          r"RunFfnTailFromDownHandlesResidentInputs\(\s*"
          r"shared_gate_handle,\s*ffn_input_handle,\s*"
          r"selected_gpu\.down_handle",
      ),
      _present(
          generated_cpp,
          "carrier_tail_captures_layer_output_handle",
          r"CaptureLayerOutput\(\s*resident_tail\.layer_output_handle\s*\)",
      ),
      _present(
          engine_text,
          "tail_primitive_can_skip_layer_output_readback",
          r"RunFfnTailFromDownHandlesResidentInputs\([^)]*"
          r"bool readback_layer_output = true\)",
      ),
      _present(
          engine_text,
          "rmsnorm_primitive_can_skip_output_readback",
          r"RunRmsNormHiddenResidentInputResidentWeight\([^)]*"
          r"bool readback_output = true\)",
      ),
  ]
  blockers_present = [
      _present(
          generated_cpp,
          "rmsnorm_wrapper_still_requires_host_residual_input_size",
          r"RunGpuLayerInputRmsNorm\([^}]*"
          r"Require\(residual_input\.size\(\) == kHiddenSize",
      ),
      _present(
          generated_cpp,
          "rmsnorm_wrapper_signature_lacks_readback_output_argument",
          r"RunGpuLayerInputRmsNorm\([^)]*"
          r"std::uint64_t resident_norm_weight_handle\)\s*\{",
      ),
      _present(
          generated_cpp,
          "rmsnorm_resident_input_call_keeps_default_readback",
          r"RunRmsNormHiddenResidentInputResidentWeight\(\s*"
          r"resident_input_handle,\s*norm_handle,\s*norm_value_count,\s*"
          r"rms_norm_epsilon,\s*repeat\)",
      ),
      _present(
          generated_cpp,
          "linear_next_rmsnorm_handle_is_closed_flag_only",
          r"g_decode_resident_tail_output_rmsnorm_input\s*\?\s*"
          r"g_decode_prev_layer_output_handle\s*:\s*0",
      ),
      _present(
          generated_cpp,
          "attention_residual_handle_is_closed_flag_only",
          r"g_decode_attention_front_resident_residual_input\s*\?\s*"
          r"g_decode_prev_layer_output_handle\s*:\s*0",
      ),
      _present(
          generated_cpp,
          "preconv_still_requires_host_attention_norm_vector",
          r"RunGpuPreConvFront\([^}]*"
          r"Require\(attn_norm\.size\(\) == kHiddenSize",
      ),
      _present(
          generated_cpp,
          "full_attention_qk_still_consumes_host_attention_norm",
          r"RunGpuFullAttentionQkFront\(\s*model_path,\s*\*t\.q_tensor,"
          r"\s*\*t\.k_tensor,\s*rms_gpu\.attn_norm",
      ),
      _present(
          generated_cpp,
          "full_attention_v_still_consumes_host_attention_norm",
          r"RunGpuFullAttentionVAny\(\s*model_path,\s*\*t\.v_tensor,"
          r"\s*rms_gpu\.attn_norm",
      ),
      _present(
          generated_cpp,
          "decode_loop_still_owns_host_residual_vector",
          r"std::vector<float> residual\s*=\s*"
          r"iq36::decode_tensor_row\([^;]*token_embd\.weight",
      ),
      _present(
          generated_cpp,
          "final_lm_head_call_still_consumes_host_residual",
          r"DecodeGpuLmHeadQ6TopK\(\s*args\.model_path,\s*index,\s*"
          r"residual,\s*rms_norm_epsilon",
      ),
      _present(
          generated_cpp,
          "lm_head_q6_handoff_uploads_host_residual",
          r"RunRmsNormThenResidentRawQ6K\(\s*residual,\s*"
          r"lm_head_norm_handle",
      ),
      _absent(
          combined,
          "lm_head_q6_input_handle_handoff_api_absent",
          r"RunRmsNormThenResidentRawQ6KFromInputHandle",
      ),
      _absent(
          generated_cpp,
          "carrier_tail_call_does_not_pass_no_readback_false_yet",
          r"RunFfnTailFromDownHandlesResidentInputs\([^;]*,\s*false\)",
      ),
      _absent(
          generated_cpp,
          "no_new_layer_output_loop_env_gate_yet",
          r"IQ36_RESIDENT_HIDDEN_STATE_CARRIER_LAYER_OUTPUT_HANDLE_LOOP",
      ),
  ]
  return {
      "covered_consumer_checks": checks_present,
      "current_blocker_checks": blockers_present,
      "covered_consumer_chain_present": all(
          row.get("present") is True for row in checks_present),
      "current_source_requires_new_loop_contract": all(
          (
              row.get("present") is True
              if "present" in row else row.get("absent") is True
          )
          for row in blockers_present
      ),
      "blocker_labels": [row["label"] for row in blockers_present],
  }


def compute(args: argparse.Namespace) -> dict[str, Any]:
  frontier = _frontier_state(_load_json(args.frontier))
  routes = _load_json(args.routes)
  seq125 = _load_json(args.seq125)
  seq55 = _load_json(args.seq55)
  seq56 = _load_json(args.seq56)
  seq57 = _load_json(args.seq57)
  seq59 = _load_json(args.seq59)
  seq60 = _load_json(args.seq60)
  seq61 = _load_json(args.seq61)
  seq64 = _load_json(args.seq64)
  seq119 = _load_json(args.seq119)
  seq121 = _load_json(args.seq121)
  decode_source = _load_text(args.decode_source)
  generated_cpp = _load_text(args.generated_cpp)
  engine_text = "\n".join([
      _load_text(args.engine_source),
      _load_text(args.engine_header),
  ])
  source = _source_shape(decode_source, generated_cpp, engine_text)

  prerequisite_ready = {
      "seq55_tail_no_readback": _metric_ready(
          seq55, "primitive_ready_for_carrier_wiring"),
      "seq56_rmsnorm_no_readback": _metric_ready(
          seq56, "primitive_ready_for_carrier_wiring"),
      "seq57_f32matvec_no_readback": _metric_ready(
          seq57, "primitive_ready_for_carrier_wiring"),
      "seq59_device_q8_q4_no_readback": _metric_ready(
          seq59, "primitive_ready_for_carrier_wiring"),
      "seq60_device_q8_q6_no_readback": _metric_ready(
          seq60, "primitive_ready_for_carrier_wiring"),
      "seq61_ffn_tail_resident_input": _metric_ready(
          seq61, "primitive_ready_for_carrier_wiring"),
      "seq64_attention_front_residual_handle": _metric_ready(
          seq64, "decode_path_ready_for_explore"),
      "seq119_carrier_preconv_bundle_source": (
          seq119.get("required_checks_passed") is True),
      "seq121_carrier_selected_shared_tail_source": (
          seq121.get("required_checks_passed") is True),
  }

  contract = {
      "decode_probe_allowed": False,
      "token_row_allowed": False,
      "source_cut_allowed": True,
      "target_compile_required_before_decode": True,
      "initial_enablement_scope": (
          "non-final carrier layer-output handle loop; final layer keeps "
          "readback until an LM-head input-handle handoff exists; full-attention "
          "RMSNorm readback remains required until QK/V consume resident handles"
      ),
      "source_gate_must_add": [
          "a default-off carrier layer-output handle-loop gate",
          "carrier OR conditions for next-layer RMSNorm and attention residual handles",
          "a readback_output argument through RunGpuLayerInputRmsNorm",
          "relaxed host-attn-norm validation for shared-Q8 carrier preconv",
          "tail readback disabled only where the next consumer chain is handle-backed",
          "host residual validity guards for diagnostics and final LM-head readback",
      ],
      "source_gate_must_not_do": [
          "launch a token row",
          "reuse the closed standalone FFN-tail or attention-residual env routes as the speed route",
          "skip final readback before an LM-head input-handle API exists",
          "skip full-attention RMSNorm readback before QK/V input-handle consumers exist",
      ],
  }

  checks = [
      {
          "name": "seq125_selected_this_contract_gate",
          "pass": (
              seq125.get("required_checks_passed") is True
              and seq125.get("selected_next_route")
              == "resident_hidden_state_carrier_layer_output_handle_loop_contract_gate"
              and _has_switch(
                  routes,
                  "bind_tail_growth_switch_to_layer_output_handle_loop_contract",
                  125,
              )
          ),
          "detail": {
              "seq125_disposition": seq125.get("disposition"),
              "seq125_selected_next_route": seq125.get("selected_next_route"),
          },
      },
      {
          "name": "primitive_and_carrier_source_prerequisites_ready",
          "pass": all(prerequisite_ready.values()),
          "detail": prerequisite_ready,
      },
      {
          "name": "existing_carrier_handle_consumers_are_present",
          "pass": source["covered_consumer_chain_present"],
          "detail": source["covered_consumer_checks"],
      },
      {
          "name": "current_source_still_needs_layer_output_handle_loop_contract",
          "pass": source["current_source_requires_new_loop_contract"],
          "detail": source["current_blocker_checks"],
      },
      {
          "name": "contract_forbids_token_row_until_source_and_compile_gates",
          "pass": (
              contract["decode_probe_allowed"] is False
              and contract["token_row_allowed"] is False
              and contract["target_compile_required_before_decode"] is True
          ),
          "detail": contract,
      },
      {
          "name": "frontier_still_below_floor_no_speed_claim",
          "pass": (
              frontier["current_best_tps"] < frontier["floor_tps"]
              and frontier["floor_gap_ms_per_token"] > 0.0
          ),
          "detail": frontier,
      },
  ]
  required_checks_passed = all(bool(row.get("pass")) for row in checks)

  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "required_checks_passed": required_checks_passed,
      "speedup_claims_allowed": False,
      "disposition": (
          "accept_layer_output_handle_loop_contract_select_source_gate"
          if required_checks_passed
          else "layer_output_handle_loop_contract_evidence_incomplete"
      ),
      "selected_next_route": (
          "resident_hidden_state_carrier_layer_output_handle_loop_source_gate"
          if required_checks_passed
          else "resident_hidden_state_carrier_layer_output_handle_loop_manual_review"
      ),
      "next_action": (
          "Do not launch another carrier token row yet. The source gate must "
          "add a default-off carrier layer-output handle loop and prove that "
          "tail readback is disabled only where the next consumer chain is "
          "handle-backed. Initial scope is non-final carrier loop wiring with "
          "final readback retained; full-attention and LM-head host consumers "
          "remain explicit blockers until their input-handle APIs exist."
          if required_checks_passed
          else "Fix the failed contract evidence before any source or decode work."
      ),
      "inputs": {
          "frontier": _rel(args.frontier),
          "routes": _rel(args.routes),
          "seq125": _rel(args.seq125),
          "seq55": _rel(args.seq55),
          "seq56": _rel(args.seq56),
          "seq57": _rel(args.seq57),
          "seq59": _rel(args.seq59),
          "seq60": _rel(args.seq60),
          "seq61": _rel(args.seq61),
          "seq64": _rel(args.seq64),
          "seq119": _rel(args.seq119),
          "seq121": _rel(args.seq121),
          "decode_source": {
              "path": _rel(args.decode_source),
              "sha256": _sha256(args.decode_source),
          },
          "generated_cpp": {
              "path": _rel(args.generated_cpp),
              "sha256": _sha256(args.generated_cpp),
          },
          "engine_source": {
              "path": _rel(args.engine_source),
              "sha256": _sha256(args.engine_source),
          },
          "engine_header": {
              "path": _rel(args.engine_header),
              "sha256": _sha256(args.engine_header),
          },
      },
      "frontier": frontier,
      "prerequisite_ready": prerequisite_ready,
      "source_shape": source,
      "contract": contract,
      "checks": checks,
      "decode_probe_allowed": False,
      "target_compile_required_before_decode": True,
  }


def write_outputs(payload: dict[str, Any], out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "metrics.json").write_text(
      json.dumps(payload, indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
  )
  failed = [row["name"] for row in payload["checks"] if not row["pass"]]
  lines = [
      "# Resident Hidden-State Carrier Layer-Output Handle-Loop Contract Gate",
      "",
      f"- required checks passed: `{str(payload['required_checks_passed']).lower()}`",
      f"- disposition: `{payload['disposition']}`",
      f"- selected next route: `{payload['selected_next_route']}`",
      f"- decode probe allowed: `{str(payload['decode_probe_allowed']).lower()}`",
      f"- target compile required before decode: `{str(payload['target_compile_required_before_decode']).lower()}`",
      f"- failed checks: `{failed}`",
      f"- speedup claims allowed: `{str(payload['speedup_claims_allowed']).lower()}`",
      "",
      "## Contract",
      "",
      payload["contract"]["initial_enablement_scope"],
      "",
      payload["next_action"],
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--frontier", type=Path, default=DEFAULT_FRONTIER)
  parser.add_argument("--routes", type=Path, default=DEFAULT_ROUTES)
  parser.add_argument("--seq125", type=Path, default=DEFAULT_SEQ125)
  parser.add_argument("--seq55", type=Path, default=DEFAULT_SEQ55)
  parser.add_argument("--seq56", type=Path, default=DEFAULT_SEQ56)
  parser.add_argument("--seq57", type=Path, default=DEFAULT_SEQ57)
  parser.add_argument("--seq59", type=Path, default=DEFAULT_SEQ59)
  parser.add_argument("--seq60", type=Path, default=DEFAULT_SEQ60)
  parser.add_argument("--seq61", type=Path, default=DEFAULT_SEQ61)
  parser.add_argument("--seq64", type=Path, default=DEFAULT_SEQ64)
  parser.add_argument("--seq119", type=Path, default=DEFAULT_SEQ119)
  parser.add_argument("--seq121", type=Path, default=DEFAULT_SEQ121)
  parser.add_argument("--decode-source", type=Path, default=DEFAULT_DECODE_SOURCE)
  parser.add_argument("--generated-cpp", type=Path, default=DEFAULT_GENERATED_CPP)
  parser.add_argument("--engine-source", type=Path, default=DEFAULT_ENGINE_SOURCE)
  parser.add_argument("--engine-header", type=Path, default=DEFAULT_ENGINE_HEADER)
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
