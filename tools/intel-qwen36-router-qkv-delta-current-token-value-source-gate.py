#!/usr/bin/env python3
"""Gate the qkv-delta current-token value-source route.

This is route-control evidence. It verifies that the current source still lacks
a product-owned value source, that closed value approximations stay closed, and
that the next source unit must root at the layer-input producer rather than at a
selector/mask variant.
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
SCHEMA_VERSION = "intel-qwen36-router-qkv-delta-current-token-value-source-gate-v0"

DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_REJECTED = ACTIVE / "rejected-routes.json"
DEFAULT_DECODE_SOURCE = ROOT / "tools/intel-qwen36-r2-gpu-decode-smoke.py"
DEFAULT_ENGINE_HEADER = ROOT / "engine/include/intel_qwen36/gpu_q4x8_matvec.hpp"
DEFAULT_ENGINE_SOURCE = ROOT / "engine/src/gpu_q4x8_matvec.cpp"
DEFAULT_SEQ296 = (
    ROOT
    / "output/router-qkv-delta-component-implementation-source-gate-20260708Tseq296Z"
    / "metrics.json"
)
DEFAULT_OUT_DIR = (
    ROOT / "output/router-qkv-delta-current-token-value-source-gate-20260708Tseq297Z"
)

ALL_LINEAR_LAYERS = [
    0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18,
    20, 21, 22, 24, 25, 26, 28, 29, 30, 32, 33, 34, 36, 37, 38,
]
TOPK = 512
HIDDEN_SIZE = 8192
DECODE_TOKENS = 8


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


def _rejected_names(rejected: dict[str, Any]) -> set[str]:
  names: set[str] = set()
  for row in rejected.get("rejected", []):
    if isinstance(row, dict) and isinstance(row.get("route"), str):
      names.add(row["route"])
  return names


def _source_state(text: str) -> dict[str, Any]:
  absent_product = [
      _absent(text, "no_product_value_source_helper",
              "DecodeRouterQkvDeltaProductValueSource", regex=False),
      _absent(text, "no_runtime_product_values_buffer",
              "router_qkv_delta_product_values", regex=False),
      _absent(text, "no_contract_product_source_true",
              "product_owned_source = true", regex=False),
      _absent(text, "no_contract_cpu_shadow_free_true",
              "cpu_shadow_free = true", regex=False),
      _absent(text, "no_contract_host_sync_free_true",
              "host_sync_free = true", regex=False),
  ]
  present_shadow_only = [
      _present(text, "source_only_guard_present",
               "IQ36_ROUTER_QKV_DELTA_COMPONENT_SOURCE is source-gate only",
               regex=False),
      _present(text, "cpu_shadow_trace_global_present",
               "g_decode_cpu_shadow_trace", regex=False),
      _present(text, "shadow_delta_applier_present",
               "DecodeApplyTopKShadowDelta", regex=False),
      _present(text, "linear_qkv_selector_present",
               "linear_qkv_col_abs", regex=False),
  ]
  layer_input_target = [
      _present(text, "correction_targets_residual_layer_input",
               r"residual\s*=\s*DecodeApplyTopKShadowDelta\("),
      _present(text, "layer_input_trace_source",
               "layer_input_by_layer", regex=False),
      _present(text, "correction_invalidates_layer_output_handle",
               "stats->cpu_shadow_layer_input_delta_values += applied_values",
               regex=False),
  ]
  device_producer_primitives = [
      _present(text, "prev_layer_output_handle_present",
               "g_decode_prev_layer_output_handle", regex=False),
      _present(text, "layer_output_handle_loop_present",
               "DecodeCarrierLayerOutputHandleLoopActive", regex=False),
      _present(text, "resident_device_state_bank_present",
               "ResidentDeviceStateHandleBank", regex=False),
      _present(text, "full_attention_core_history_handle_present",
               "resident_hidden_state_carrier_full_attention_core_history_handle",
               regex=False),
      _present(text, "clone_resident_f32_buffer_used",
               "CloneResidentF32Buffer", regex=False),
  ]
  return {
      "product_value_source_absent": _all_absent(absent_product),
      "shadow_only_diagnostic_present": _all_present(present_shadow_only),
      "correction_targets_layer_input": _all_present(layer_input_target),
      "device_producer_primitives_present": _all_present(device_producer_primitives),
      "absent_product_checks": absent_product,
      "shadow_only_checks": present_shadow_only,
      "layer_input_target_checks": layer_input_target,
      "device_producer_primitive_checks": device_producer_primitives,
  }


def _engine_state(header: str, source: str) -> dict[str, Any]:
  rows = [
      _present(header, "clone_resident_f32_api",
               "CloneResidentF32Buffer", regex=False),
      _present(header, "full_attention_qk_from_handles_api",
               "RunFullAttentionQkNormRopeFromHandles", regex=False),
      _present(header, "full_attention_history_from_handle_api",
               "BuildFullAttentionHistoryFromHandle", regex=False),
      _present(header, "full_attention_core_from_handles_api",
               "RunFullAttentionCoreGateThenResidentPackedQ4X8ResidualRmsNormFromHandles",
               regex=False),
      _present(source, "clone_resident_f32_impl",
               "CloneResidentF32Buffer", regex=False),
      _present(source, "full_attention_qk_from_handles_impl",
               "RunFullAttentionQkNormRopeFromHandles", regex=False),
      _present(source, "full_attention_history_from_handle_impl",
               "BuildFullAttentionHistoryFromHandle", regex=False),
      _present(source, "full_attention_core_from_handles_impl",
               "RunFullAttentionCoreGateThenResidentPackedQ4X8ResidualRmsNormFromHandles",
               regex=False),
  ]
  absent = [
      _absent(header + "\n" + source, "no_selected_value_overlay_api_yet",
              "RouterQkvDelta", regex=False),
  ]
  return {
      "device_handle_primitives_present": _all_present(rows),
      "selected_value_overlay_absent": _all_absent(absent),
      "present_checks": rows,
      "absent_checks": absent,
  }


def compute(args: argparse.Namespace) -> dict[str, Any]:
  routes = _load_json(args.routes)
  rejected = _load_json(args.rejected)
  seq296 = _load_json(args.seq296)
  decode_source = _read(args.decode_source)
  engine_header = _read(args.engine_header)
  engine_source = _read(args.engine_source)
  source = _source_state(decode_source)
  engine = _engine_state(engine_header, engine_source)
  rejected_names = _rejected_names(rejected)
  required_closed = {
      "router_math_static_or_lagged_qkv_delta_predictors",
      "router_math_live_round_or_selected_affine_qkv_delta_approximation",
      "router_math_split_full_attention_projection_arithmetic_residual_fix",
  }
  missing_closed = sorted(required_closed - rejected_names)
  top512_values = len(ALL_LINEAR_LAYERS) * DECODE_TOKENS * TOPK

  checks = [
      {
          "name": "seq296_selected_current_token_value_source_gate",
          "pass": (
              seq296.get("required_checks_passed") is True
              and seq296.get("selected_next_route")
              == "router_prompt_all_linear_qkv_delta_current_token_value_source_gate"
              and seq296.get("component_product_source_present") is False
              and seq296.get("decode_probe_allowed") is False
              and _has_candidate(
                  routes, 296,
                  "reject_fake_product_source_select_current_token_value_source_gate")
              and _has_switch(
                  routes,
                  "select_router_prompt_all_linear_qkv_delta_current_token_value_source_gate",
                  296)
          ),
      },
      {
          "name": "current_source_has_no_product_value_source_yet",
          "pass": (
              source["product_value_source_absent"]
              and source["shadow_only_diagnostic_present"]),
          "detail": source,
      },
      {
          "name": "correction_target_is_layer_input_producer",
          "pass": source["correction_targets_layer_input"],
          "detail": {
              "top512_values_expected": top512_values,
              "layers": ALL_LINEAR_LAYERS,
              "rule": (
                  "The qkv selector only chooses sensitive layer-input entries; "
                  "the value source must repair the current-token layer-input "
                  "producer before selector/value encodings are revisited."
              ),
          },
      },
      {
          "name": "closed_value_substitutes_not_reopened",
          "pass": not missing_closed,
          "detail": {"missing_closed_routes": missing_closed},
      },
      {
          "name": "device_current_token_producer_primitives_exist_but_overlay_absent",
          "pass": (
              source["device_producer_primitives_present"]
              and engine["device_handle_primitives_present"]
              and engine["selected_value_overlay_absent"]),
          "detail": {
              "source": source["device_producer_primitive_checks"],
              "engine": engine,
          },
      },
  ]
  required = all(bool(row.get("pass")) for row in checks)
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "inputs": {
          "routes": _rel(args.routes),
          "rejected": _rel(args.rejected),
          "decode_source": _rel(args.decode_source),
          "decode_source_sha256": _sha256(args.decode_source),
          "engine_header": _rel(args.engine_header),
          "engine_header_sha256": _sha256(args.engine_header),
          "engine_source": _rel(args.engine_source),
          "engine_source_sha256": _sha256(args.engine_source),
          "seq296_gate": _rel(args.seq296),
      },
      "source": source,
      "engine": engine,
      "correction_shape": {
          "layers": ALL_LINEAR_LAYERS,
          "layer_count": len(ALL_LINEAR_LAYERS),
          "topk": TOPK,
          "hidden_size": HIDDEN_SIZE,
          "decode_tokens": DECODE_TOKENS,
          "top512_values_expected": top512_values,
          "top512_fraction_per_layer": TOPK / HIDDEN_SIZE,
      },
      "checks": checks,
      "required_checks_passed": required,
      "component_product_source_present": False,
      "decode_probe_allowed": False,
      "router_distribution_allowed": False,
      "speedup_claims_allowed": False,
      "disposition": (
          "reject_missing_product_value_source_select_layer_input_producer_root_gate"
          if required else
          "block_before_layer_input_producer_root_gate"
      ),
      "selected_next_route": (
          "router_prompt_all_linear_qkv_delta_layer_input_producer_root_gate"
          if required else
          "router_prompt_all_linear_qkv_delta_current_token_value_source_fix_gate"
      ),
      "next_route_reason": (
          "The current source has no product-owned value source; the only passing "
          "path still reads CPU-shadow layer inputs. Since the correction target "
          "is the current-token layer input, the next unit must root the producer "
          "of those values on the device/resident path before any selector, "
          "encoding, decode, or distribution row."
          if required else
          "Current-token value-source evidence is inconsistent; fix this gate "
          "before adding source or running decode."
      ),
  }


def write_outputs(metrics: dict[str, Any], out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  failed = [row["name"] for row in metrics["checks"] if not row.get("pass")]
  lines = [
      "# Router QKV Delta Current-Token Value Source Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- component_product_source_present: `{str(metrics['component_product_source_present']).lower()}`",
      f"- decode_probe_allowed: `{str(metrics['decode_probe_allowed']).lower()}`",
      f"- router_distribution_allowed: `{str(metrics['router_distribution_allowed']).lower()}`",
      f"- top512 values: `{metrics['correction_shape']['top512_values_expected']}`",
      f"- failed_checks: `{failed}`",
      "",
      metrics["next_route_reason"],
      "",
      "This is route-control evidence. It does not claim speed.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--routes", type=Path, default=DEFAULT_ROUTES)
  parser.add_argument("--rejected", type=Path, default=DEFAULT_REJECTED)
  parser.add_argument("--decode-source", type=Path, default=DEFAULT_DECODE_SOURCE)
  parser.add_argument("--engine-header", type=Path, default=DEFAULT_ENGINE_HEADER)
  parser.add_argument("--engine-source", type=Path, default=DEFAULT_ENGINE_SOURCE)
  parser.add_argument("--seq296", type=Path, default=DEFAULT_SEQ296)
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
