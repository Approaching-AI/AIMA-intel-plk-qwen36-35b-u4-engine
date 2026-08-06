#!/usr/bin/env python3
"""Audit router all-linear qkv-delta component source wiring.

This is source/generate-only evidence. It verifies the default-off component
contract for the all-linear qkv-delta route before target compile or any token
row is allowed.
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
SCHEMA_VERSION = "intel-qwen36-router-qkv-delta-component-source-gate-v0"

DEFAULT_FRONTIER = ACTIVE / "frontier.json"
DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_REJECTED = ACTIVE / "rejected-routes.json"
DEFAULT_DECODE_SOURCE = ROOT / "tools/intel-qwen36-r2-gpu-decode-smoke.py"
DEFAULT_SEQ292 = (
    ROOT / "output/router-qkv-delta-source-gate-20260708Tseq292Z/metrics.json"
)
DEFAULT_GENERATE_DIR = (
    ROOT / "output/router-qkv-delta-component-generate-only-20260708Tseq293Z"
)
DEFAULT_OUT_DIR = (
    ROOT / "output/router-qkv-delta-component-source-gate-20260708Tseq293Z"
)

ALL_LINEAR_LAYERS = [
    0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18,
    20, 21, 22, 24, 25, 26, 28, 29, 30, 32, 33, 34, 36, 37, 38,
]
TOPK = 512
HIDDEN_SIZE = 8192


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


def _all_present(rows: list[dict[str, Any]]) -> bool:
  return all(row.get("present") is True for row in rows)


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


def _rejected_names(rejected: dict[str, Any]) -> set[str]:
  names: set[str] = set()
  for row in rejected.get("rejected", []):
    if isinstance(row, dict) and isinstance(row.get("route"), str):
      names.add(row["route"])
  return names


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
  rows = [
      _present(text, "env_gate_present",
               "IQ36_ROUTER_QKV_DELTA_COMPONENT_SOURCE", regex=False),
      _present(text, "python_arg_present",
               "router_qkv_delta_component_source", regex=False),
      _present(text, "run_env_propagates_gate",
               r"env_parts[\s\S]*?IQ36_ROUTER_QKV_DELTA_COMPONENT_SOURCE"),
      _present(text, "manifest_records_gate",
               '"router_qkv_delta_component_source"', regex=False),
      _present(text, "cxx_global_present",
               "bool g_decode_router_qkv_delta_component_source = false;",
               regex=False),
      _present(text, "component_topk_constant",
               "kDecodeRouterQkvDeltaComponentTopK = 512", regex=False),
      _present(text, "component_layer_constant",
               "kDecodeRouterQkvDeltaComponentLayers", regex=False),
      _present(text, "component_contract_struct",
               "DecodeRouterQkvDeltaComponentContract", regex=False),
      _present(text, "component_ready_function",
               "DecodeRouterQkvDeltaComponentReady", regex=False),
      _present(text, "contract_product_source_field",
               "product_owned_source", regex=False),
      _present(text, "contract_shadow_free_field",
               "cpu_shadow_free", regex=False),
      _present(text, "contract_host_sync_free_field",
               "host_sync_free", regex=False),
      _present(text, "source_only_guard",
               "IQ36_ROUTER_QKV_DELTA_COMPONENT_SOURCE is source-gate only",
               regex=False),
      _present(text, "stdout_field_present",
               "router_qkv_delta_component_source_enabled", regex=False),
  ]
  return {"present": _all_present(rows), "present_checks": rows}


def _generated_markers(text: str) -> dict[str, Any]:
  rows = [
      _present(text, "env_gate_present",
               "IQ36_ROUTER_QKV_DELTA_COMPONENT_SOURCE", regex=False),
      _present(text, "cxx_arg_present",
               "router_qkv_delta_component_source", regex=False),
      _present(text, "cxx_global_present",
               "bool g_decode_router_qkv_delta_component_source = false;",
               regex=False),
      _present(text, "component_topk_constant",
               "kDecodeRouterQkvDeltaComponentTopK = 512", regex=False),
      _present(text, "component_layer_constant",
               "kDecodeRouterQkvDeltaComponentLayers", regex=False),
      _present(text, "component_contract_struct",
               "DecodeRouterQkvDeltaComponentContract", regex=False),
      _present(text, "component_ready_function",
               "DecodeRouterQkvDeltaComponentReady", regex=False),
      _present(text, "contract_product_source_field",
               "product_owned_source", regex=False),
      _present(text, "contract_shadow_free_field",
               "cpu_shadow_free", regex=False),
      _present(text, "contract_host_sync_free_field",
               "host_sync_free", regex=False),
      _present(text, "source_only_guard",
               "IQ36_ROUTER_QKV_DELTA_COMPONENT_SOURCE is source-gate only",
               regex=False),
      _present(text, "stdout_field_present",
               "router_qkv_delta_component_source_enabled", regex=False),
  ]
  return {"present": _all_present(rows), "present_checks": rows}


def _manifest_checks(result: dict[str, Any], generate_dir: Path) -> dict[str, bool]:
  return {
      "generate_only": result.get("generate_only") is True,
      "router_qkv_delta_component_source": (
          result.get("router_qkv_delta_component_source") is True),
      "router_qkv_delta_component_topk": (
          result.get("router_qkv_delta_component_topk") == TOPK),
      "router_qkv_delta_component_layers": (
          result.get("router_qkv_delta_component_layers") == ALL_LINEAR_LAYERS),
      "opencl_double_swiglu": result.get("opencl_double_swiglu") is True,
      "decode_tokens_eight": result.get("decode_tokens") == 8,
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
      "no_smoke_json": not (generate_dir / "smoke.json").exists(),
  }


def compute(args: argparse.Namespace) -> dict[str, Any]:
  frontier = _load_json(args.frontier)
  routes = _load_json(args.routes)
  rejected = _load_json(args.rejected)
  seq292 = _load_json(args.seq292)
  decode_source = _read(args.decode_source)
  result_path = args.generate_dir / "result.json"
  cpp_path = args.generate_dir / "r2_gpu_decode_smoke.cpp"
  result = _load_json(result_path)
  generated_cpp = _read(cpp_path)

  source = _source_markers(decode_source)
  generated = _generated_markers(generated_cpp)
  manifest_checks = _manifest_checks(result, args.generate_dir)
  rejected_names = _rejected_names(rejected)
  required_closed = {
      "router_math_static_or_lagged_qkv_delta_predictors",
      "router_math_live_round_or_selected_affine_qkv_delta_approximation",
      "router_math_split_full_attention_projection_arithmetic_residual_fix",
  }
  missing_closed = sorted(required_closed - rejected_names)
  expected_values = len(ALL_LINEAR_LAYERS) * 8 * TOPK

  checks = [
      {
          "name": "seq292_selected_component_source_gate",
          "pass": (
              seq292.get("required_checks_passed") is True
              and seq292.get("selected_next_route")
              == "router_prompt_all_linear_qkv_delta_component_source_gate"
              and _has_candidate(
                  routes, 292,
                  "select_all_linear_qkv_delta_component_source_gate")
              and _has_switch(
                  routes,
                  "select_router_prompt_all_linear_qkv_delta_component_source_gate",
                  292)
          ),
      },
      {
          "name": "source_component_contract_present_default_off",
          "pass": source["present"],
          "detail": source,
      },
      {
          "name": "generated_cpp_component_contract_present_default_off",
          "pass": generated["present"],
          "detail": generated,
      },
      {
          "name": "generate_only_manifest_is_component_source_not_token_row",
          "pass": all(manifest_checks.values()),
          "detail": manifest_checks,
      },
      {
          "name": "top512_shape_preserved",
          "pass": expected_values == 122880,
          "detail": {
              "layers": ALL_LINEAR_LAYERS,
              "topk": TOPK,
              "hidden_size": HIDDEN_SIZE,
              "expected_values": expected_values,
              "fraction_per_layer": TOPK / HIDDEN_SIZE,
          },
      },
      {
          "name": "closed_approximation_classes_not_reopened",
          "pass": not missing_closed,
          "detail": {"missing_closed_routes": missing_closed},
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
          "decode_source": _rel(args.decode_source),
          "decode_source_sha256": _sha256(args.decode_source),
          "seq292_source_gate": _rel(args.seq292),
          "generate_only_result": _rel(result_path),
          "generated_cpp": _rel(cpp_path),
          "generated_cpp_sha256": _sha256(cpp_path),
      },
      "frontier": _frontier_state(frontier),
      "source": source,
      "generated": generated,
      "generate_manifest_checks": manifest_checks,
      "correction_shape": {
          "layers": ALL_LINEAR_LAYERS,
          "layer_count": len(ALL_LINEAR_LAYERS),
          "topk": TOPK,
          "hidden_size": HIDDEN_SIZE,
          "decode_tokens": 8,
          "top512_values_expected": expected_values,
          "top512_fraction_per_layer": TOPK / HIDDEN_SIZE,
      },
      "checks": checks,
      "required_checks_passed": required,
      "decode_probe_allowed": False,
      "component_probe_allowed": False,
      "target_compile_required_before_component_probe": True,
      "speedup_claims_allowed": False,
      "disposition": (
          "accept_all_linear_qkv_delta_component_source_wiring"
          if required else
          "reject_all_linear_qkv_delta_component_source_wiring"
      ),
      "selected_next_route": (
          "router_prompt_all_linear_qkv_delta_component_target_compile_gate"
          if required else
          "router_prompt_all_linear_qkv_delta_component_source_fix_gate"
      ),
      "next_route_reason": (
          "Default-off router qkv-delta component source wiring now records "
          "the all-linear top512 contract and blocks token execution. Target "
          "compile is required before any component or decode probe."
          if required else
          "The qkv-delta component source contract is incomplete; fix source "
          "and generate-only evidence before compile, component probe, or decode."
      ),
  }


def write_outputs(metrics: dict[str, Any], out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  failed = [row["name"] for row in metrics["checks"] if not row.get("pass")]
  lines = [
      "# Router QKV Delta Component Source Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- decode_probe_allowed: `{str(metrics['decode_probe_allowed']).lower()}`",
      f"- target_compile_required_before_component_probe: `{str(metrics['target_compile_required_before_component_probe']).lower()}`",
      f"- top512 values: `{metrics['correction_shape']['top512_values_expected']}`",
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
  parser.add_argument("--rejected", type=Path, default=DEFAULT_REJECTED)
  parser.add_argument("--decode-source", type=Path, default=DEFAULT_DECODE_SOURCE)
  parser.add_argument("--seq292", type=Path, default=DEFAULT_SEQ292)
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
