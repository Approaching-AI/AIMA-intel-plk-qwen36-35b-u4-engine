#!/usr/bin/env python3
"""Audit router qkv-delta layer-input producer source wiring.

This is source/generate-only evidence. It verifies the default-off
full-attention residual layer-input producer contract before target compile or
any token row is allowed.
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
    "intel-qwen36-router-qkv-delta-layer-input-producer-source-gate-v0"
)

DEFAULT_FRONTIER = ACTIVE / "frontier.json"
DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_DECODE_SOURCE = ROOT / "tools/intel-qwen36-r2-gpu-decode-smoke.py"
DEFAULT_SEQ298 = (
    ROOT
    / "output/router-qkv-delta-layer-input-producer-root-gate-20260708Tseq298Z"
    / "metrics.json"
)
DEFAULT_GENERATE_DIR = (
    ROOT
    / "output/router-qkv-delta-layer-input-producer-generate-only-20260708Tseq299Z"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "output/router-qkv-delta-layer-input-producer-source-gate-20260708Tseq299Z"
)

ALL_LINEAR_LAYERS = [
    0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18,
    20, 21, 22, 24, 25, 26, 28, 29, 30, 32, 33, 34, 36, 37, 38,
]
PRODUCER_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31, 35]
DECODE_TOKENS = 8
HIDDEN_SIZE = 2048
ROOT_VALUES = len(PRODUCER_LAYERS) * DECODE_TOKENS * HIDDEN_SIZE


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


def _has_candidate(routes: dict[str, Any], seq: int, disposition: str) -> bool:
  return any(
      isinstance(row, dict)
      and row.get("seq") == seq
      and row.get("disposition") == disposition
      for row in routes.get("candidate_history", [])
  )


def _has_switch(routes: dict[str, Any], decision: str, seq_covered: int) -> bool:
  return any(
      isinstance(row, dict)
      and row.get("decision") == decision
      and _num(row.get("seq_covered")) >= seq_covered
      and row.get("resolved") is True
      for row in routes.get("switch_decisions", [])
  )


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
  }


def _source_markers(text: str) -> dict[str, Any]:
  present = [
      _present(text, "env_gate_present",
               "IQ36_ROUTER_QKV_DELTA_LAYER_INPUT_PRODUCER_SOURCE",
               regex=False),
      _present(text, "python_arg_present",
               "router_qkv_delta_layer_input_producer_source", regex=False),
      _present(text, "run_env_propagates_gate",
               r"env_parts[\s\S]*?IQ36_ROUTER_QKV_DELTA_LAYER_INPUT_PRODUCER_SOURCE"),
      _present(text, "manifest_records_gate",
               '"router_qkv_delta_layer_input_producer_source"', regex=False),
      _present(text, "cxx_global_present",
               "bool g_decode_router_qkv_delta_layer_input_producer_source = false;",
               regex=False),
      _present(text, "producer_layer_constant",
               "kDecodeRouterQkvDeltaFullAttentionProducerLayers", regex=False),
      _present(text, "producer_contract_struct",
               "DecodeRouterQkvDeltaFullAttentionResidualSource", regex=False),
      _present(text, "producer_ready_function",
               "DecodeRouterQkvDeltaFullAttentionResidualSourceReady",
               regex=False),
      _present(text, "contract_product_source_field_false",
               "bool product_owned_source = false;", regex=False),
      _present(text, "contract_shadow_free_field_false",
               "bool cpu_shadow_free = false;", regex=False),
      _present(text, "contract_host_sync_free_field_false",
               "bool host_sync_free = false;", regex=False),
      _present(text, "resident_residual_handle_source_field_false",
               "bool resident_attention_residual_handle_source = false;",
               regex=False),
      _present(text, "layer_input_values_selected_field_false",
               "bool layer_input_values_selected = false;", regex=False),
      _present(text, "source_only_guard",
               "IQ36_ROUTER_QKV_DELTA_LAYER_INPUT_PRODUCER_SOURCE is source-gate only",
               regex=False),
      _present(text, "stdout_field_present",
               "router_qkv_delta_layer_input_producer_source_enabled",
               regex=False),
      _present(text, "prior_full_attention_handle_primitive",
               "attention_gpu.attn_residual_handle", regex=False),
      _present(text, "carrier_loop_primitive",
               "DecodeCarrierLayerOutputHandleLoopActive", regex=False),
  ]
  absent = [
      _absent(text, "no_product_source_bit_flip",
              "product_owned_source = true", regex=False),
      _absent(text, "no_cpu_shadow_free_bit_flip",
              "cpu_shadow_free = true", regex=False),
      _absent(text, "no_host_sync_free_bit_flip",
              "host_sync_free = true", regex=False),
      _absent(text, "no_runtime_selected_values_buffer",
              "router_qkv_delta_product_values", regex=False),
  ]
  return {
      "present": _all_present(present),
      "fake_product_source_absent": _all_absent(absent),
      "present_checks": present,
      "absent_checks": absent,
  }


def _generated_markers(text: str) -> dict[str, Any]:
  present = [
      _present(text, "env_gate_present",
               "IQ36_ROUTER_QKV_DELTA_LAYER_INPUT_PRODUCER_SOURCE",
               regex=False),
      _present(text, "cxx_arg_present",
               "router_qkv_delta_layer_input_producer_source", regex=False),
      _present(text, "cxx_global_present",
               "bool g_decode_router_qkv_delta_layer_input_producer_source = false;",
               regex=False),
      _present(text, "producer_layer_constant",
               "kDecodeRouterQkvDeltaFullAttentionProducerLayers", regex=False),
      _present(text, "producer_contract_struct",
               "DecodeRouterQkvDeltaFullAttentionResidualSource", regex=False),
      _present(text, "producer_ready_function",
               "DecodeRouterQkvDeltaFullAttentionResidualSourceReady",
               regex=False),
      _present(text, "resident_residual_handle_source_field_false",
               "bool resident_attention_residual_handle_source = false;",
               regex=False),
      _present(text, "layer_input_values_selected_field_false",
               "bool layer_input_values_selected = false;", regex=False),
      _present(text, "source_only_guard",
               "IQ36_ROUTER_QKV_DELTA_LAYER_INPUT_PRODUCER_SOURCE is source-gate only",
               regex=False),
      _present(text, "stdout_field_present",
               "router_qkv_delta_layer_input_producer_source_enabled",
               regex=False),
      _present(text, "prior_full_attention_handle_primitive",
               "attention_gpu.attn_residual_handle", regex=False),
      _present(text, "carrier_loop_primitive",
               "DecodeCarrierLayerOutputHandleLoopActive", regex=False),
  ]
  absent = [
      _absent(text, "no_product_source_bit_flip",
              "product_owned_source = true", regex=False),
      _absent(text, "no_cpu_shadow_free_bit_flip",
              "cpu_shadow_free = true", regex=False),
      _absent(text, "no_host_sync_free_bit_flip",
              "host_sync_free = true", regex=False),
      _absent(text, "no_runtime_selected_values_buffer",
              "router_qkv_delta_product_values", regex=False),
  ]
  return {
      "present": _all_present(present),
      "fake_product_source_absent": _all_absent(absent),
      "present_checks": present,
      "absent_checks": absent,
  }


def _manifest_checks(result: dict[str, Any], generate_dir: Path) -> dict[str, bool]:
  return {
      "generate_only": result.get("generate_only") is True,
      "component_source_still_enabled": (
          result.get("router_qkv_delta_component_source") is True),
      "producer_source_enabled": (
          result.get("router_qkv_delta_layer_input_producer_source") is True),
      "producer_layers": (
          result.get("router_qkv_delta_layer_input_producer_layers")
          == PRODUCER_LAYERS),
      "producer_decode_tokens": (
          result.get("router_qkv_delta_layer_input_producer_decode_tokens")
          == DECODE_TOKENS),
      "producer_hidden_size": (
          result.get("router_qkv_delta_layer_input_producer_hidden_size")
          == HIDDEN_SIZE),
      "producer_root_values": (
          result.get("router_qkv_delta_layer_input_producer_root_values")
          == ROOT_VALUES),
      "component_top512_layers_preserved": (
          result.get("router_qkv_delta_component_layers") == ALL_LINEAR_LAYERS
          and result.get("router_qkv_delta_component_topk") == 512),
      "opencl_double_swiglu": result.get("opencl_double_swiglu") is True,
      "decode_tokens_eight": result.get("decode_tokens") == DECODE_TOKENS,
      "frontier_stack_present": (
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
          and result.get("gpu_lm_head_q6") is True),
      "speedup_claims_forbidden": (
          result.get("speedup_claims_allowed") is False),
      "no_smoke_json": not (generate_dir / "smoke.json").exists(),
  }


def compute(args: argparse.Namespace) -> dict[str, Any]:
  frontier = _load_json(args.frontier)
  routes = _load_json(args.routes)
  seq298 = _load_json(args.seq298)
  decode_source = _read(args.decode_source)
  result_path = args.generate_dir / "result.json"
  cpp_path = args.generate_dir / "r2_gpu_decode_smoke.cpp"
  result = _load_json(result_path)
  generated_cpp = _read(cpp_path)

  source = _source_markers(decode_source)
  generated = _generated_markers(generated_cpp)
  manifest_checks = _manifest_checks(result, args.generate_dir)
  checks = [
      {
          "name": "seq298_selected_layer_input_producer_source_gate",
          "pass": (
              seq298.get("required_checks_passed") is True
              and seq298.get("selected_next_route")
              == "router_prompt_full_attention_residual_layer_input_producer_source_gate"
              and seq298.get("decode_probe_allowed") is False
              and _has_candidate(
                  routes, 298,
                  "select_full_attention_residual_layer_input_producer_source_gate")
              and _has_switch(
                  routes,
                  "select_router_prompt_full_attention_residual_layer_input_producer_source_gate",
                  298)
          ),
      },
      {
          "name": "source_layer_input_producer_contract_present_default_off",
          "pass": (
              source["present"] and source["fake_product_source_absent"]),
          "detail": source,
      },
      {
          "name": "generated_cpp_layer_input_producer_contract_present_default_off",
          "pass": (
              generated["present"] and generated["fake_product_source_absent"]),
          "detail": generated,
      },
      {
          "name": "generate_only_manifest_is_source_not_token_row",
          "pass": all(manifest_checks.values()),
          "detail": manifest_checks,
      },
      {
          "name": "producer_root_shape_preserved",
          "pass": ROOT_VALUES == 147456,
          "detail": {
              "producer_layers": PRODUCER_LAYERS,
              "decode_tokens": DECODE_TOKENS,
              "hidden_size": HIDDEN_SIZE,
              "root_values": ROOT_VALUES,
          },
      },
  ]
  required = all(bool(row.get("pass")) for row in checks)
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "inputs": {
          "frontier": _rel(args.frontier),
          "routes": _rel(args.routes),
          "decode_source": _rel(args.decode_source),
          "decode_source_sha256": _sha256(args.decode_source),
          "seq298_root_gate": _rel(args.seq298),
          "generate_only_result": _rel(result_path),
          "generated_cpp": _rel(cpp_path),
          "generated_cpp_sha256": _sha256(cpp_path),
      },
      "frontier": _frontier_state(frontier),
      "source": source,
      "generated": generated,
      "manifest_checks": manifest_checks,
      "producer_root": {
          "root": "prior_full_attention_ffn_residual_input",
          "producer_layers": PRODUCER_LAYERS,
          "producer_layer_count": len(PRODUCER_LAYERS),
          "decode_tokens": DECODE_TOKENS,
          "hidden_size": HIDDEN_SIZE,
          "root_values": ROOT_VALUES,
      },
      "checks": checks,
      "required_checks_passed": required,
      "target_compile_allowed": required,
      "component_probe_allowed": False,
      "decode_probe_allowed": False,
      "router_distribution_allowed": False,
      "speedup_claims_allowed": False,
      "disposition": (
          "accept_full_attention_residual_layer_input_producer_source_wiring"
          if required else
          "reject_full_attention_residual_layer_input_producer_source_wiring"
      ),
      "selected_next_route": (
          "router_prompt_full_attention_residual_layer_input_producer_target_compile_gate"
          if required else
          "router_prompt_full_attention_residual_layer_input_producer_source_fix_gate"
      ),
      "next_route_reason": (
          "Default-off source/generate-only wiring now records the prior "
          "full-attention FFN residual layer-input producer root. Target compile "
          "is required before any component probe, decode row, router "
          "distribution row, selector/value approximation rerun, or long-context "
          "expansion."
          if required else
          "The layer-input producer source wiring is incomplete. Fix the "
          "source/generate-only contract before target compile or any token row."
      ),
  }


def write_outputs(metrics: dict[str, Any], out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
  )
  failed = [
      row["name"] for row in metrics["checks"]
      if not bool(row.get("pass"))
  ]
  lines = [
      "# Router QKV Delta Layer-Input Producer Source Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- target_compile_allowed: `{str(metrics['target_compile_allowed']).lower()}`",
      f"- decode_probe_allowed: `{str(metrics['decode_probe_allowed']).lower()}`",
      f"- router_distribution_allowed: `{str(metrics['router_distribution_allowed']).lower()}`",
      f"- producer root: `{metrics['producer_root']['root']}`",
      f"- producer root values: `{metrics['producer_root']['root_values']}`",
      f"- failed_checks: `{failed}`",
      "",
      metrics["next_route_reason"],
      "",
      "This is source/generate-only evidence. It does not claim speed.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--frontier", type=Path, default=DEFAULT_FRONTIER)
  parser.add_argument("--routes", type=Path, default=DEFAULT_ROUTES)
  parser.add_argument("--decode-source", type=Path, default=DEFAULT_DECODE_SOURCE)
  parser.add_argument("--seq298", type=Path, default=DEFAULT_SEQ298)
  parser.add_argument("--generate-dir", type=Path, default=DEFAULT_GENERATE_DIR)
  parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
  return parser.parse_args()


def main() -> int:
  args = parse_args()
  metrics = compute(args)
  write_outputs(metrics, args.out_dir)
  print(json.dumps({
      "disposition": metrics["disposition"],
      "out_dir": _rel(args.out_dir),
      "required_checks_passed": metrics["required_checks_passed"],
      "selected_next_route": metrics["selected_next_route"],
  }, sort_keys=True))
  return 0 if metrics["required_checks_passed"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
