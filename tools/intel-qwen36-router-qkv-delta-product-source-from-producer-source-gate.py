#!/usr/bin/env python3
"""Gate the qkv-delta consumer source after producer decode evidence.

This is route-control/source evidence. It consumes the product-owned producer
decode gate and checks whether the all-linear qkv-delta consumer can honestly
be promoted from the existing source. If the consumer is still absent, it
selects the next source unit for a device sparse-overlay primitive rather than
launching a decode or distribution row.
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
    "intel-qwen36-router-qkv-delta-product-source-from-producer-source-gate-v0"
)

DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_REJECTED = ACTIVE / "rejected-routes.json"
DEFAULT_DECODE_SOURCE = ROOT / "tools/intel-qwen36-r2-gpu-decode-smoke.py"
DEFAULT_ENGINE_HEADER = ROOT / "engine/include/intel_qwen36/gpu_q4x8_matvec.hpp"
DEFAULT_ENGINE_SOURCE = ROOT / "engine/src/gpu_q4x8_matvec.cpp"
DEFAULT_SEQ305 = (
    ROOT
    / "output/router-qkv-delta-layer-input-producer-decode-gate-20260708Tseq305Z"
    / "metrics.json"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "output/router-qkv-delta-product-source-from-producer-source-gate-20260708Tseq306Z"
)

ALL_LINEAR_LAYERS = [
    0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18,
    20, 21, 22, 24, 25, 26, 28, 29, 30, 32, 33, 34, 36, 37, 38,
]
PRODUCER_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31, 35]
TOPK = 512
HIDDEN_SIZE = 2048
DECODE_TOKENS = 8
TOP512_VALUES = len(ALL_LINEAR_LAYERS) * DECODE_TOKENS * TOPK
PRODUCER_VALUES = len(PRODUCER_LAYERS) * DECODE_TOKENS * HIDDEN_SIZE


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
  producer_present = [
      _present(text, "producer_product_contract",
               "DecodeRouterQkvDeltaFullAttentionResidualProductSourceContract",
               regex=False),
      _present(text, "producer_handle_vector",
               "g_decode_router_qkv_delta_full_attention_residual_source_handles",
               regex=False),
      _present(text, "producer_capture_function",
               "DecodeCaptureRouterQkvDeltaFullAttentionResidualSource",
               regex=False),
      _present(text, "producer_ready_stdout",
               "router_qkv_delta_layer_input_producer_source_ready",
               regex=False),
  ]
  component_still_source_only = [
      _present(text, "component_source_env",
               "IQ36_ROUTER_QKV_DELTA_COMPONENT_SOURCE", regex=False),
      _present(text, "component_source_only_guard",
               "IQ36_ROUTER_QKV_DELTA_COMPONENT_SOURCE is source-gate only",
               regex=False),
      _present(text, "component_contract_still_false",
               "bool product_owned_source = false;", regex=False),
  ]
  consumer_absent = [
      _absent(text, "no_product_consumer_helper",
              "DecodeRouterQkvDeltaProductSourceFromProducer", regex=False),
      _absent(text, "no_product_consumer_env",
              "IQ36_ROUTER_QKV_DELTA_PRODUCT_SOURCE_FROM_PRODUCER", regex=False),
      _absent(text, "no_runtime_product_values_buffer",
              "router_qkv_delta_product_values", regex=False),
      _absent(text, "no_component_product_contract_helper",
              "DecodeRouterQkvDeltaComponentProductSourceContract", regex=False),
  ]
  return {
      "producer_source_present": _all_present(producer_present),
      "component_source_still_source_only": _all_present(component_still_source_only),
      "product_consumer_absent": _all_absent(consumer_absent),
      "producer_checks": producer_present,
      "component_source_only_checks": component_still_source_only,
      "consumer_absent_checks": consumer_absent,
  }


def _engine_state(header: str, source: str) -> dict[str, Any]:
  combined = header + "\n" + source
  present = [
      _present(header, "resident_f32_handle_api",
               "RunResidentF32MatvecFromInputHandle", regex=False),
      _present(header, "resident_buffer_clone_api",
               "CloneResidentF32Buffer", regex=False),
      _present(source, "resident_f32_handle_impl",
               "RunResidentF32MatvecFromInputHandle", regex=False),
      _present(source, "resident_buffer_clone_impl",
               "CloneResidentF32Buffer", regex=False),
  ]
  absent = [
      _absent(combined, "no_router_qkv_delta_overlay_api",
              "RouterQkvDelta", regex=False),
      _absent(combined, "no_sparse_overlay_kernel",
              "qkv_delta_sparse_overlay", regex=False),
      _absent(combined, "no_selected_value_overlay",
              "SelectedValueOverlay", regex=False),
  ]
  return {
      "resident_handle_primitives_present": _all_present(present),
      "device_sparse_overlay_absent": _all_absent(absent),
      "present_checks": present,
      "absent_checks": absent,
  }


def compute(args: argparse.Namespace) -> dict[str, Any]:
  routes = _load_json(args.routes)
  rejected = _load_json(args.rejected)
  seq305 = _load_json(args.seq305)
  source_text = _read(args.decode_source)
  engine_header = _read(args.engine_header)
  engine_source = _read(args.engine_source)
  source = _source_state(source_text)
  engine = _engine_state(engine_header, engine_source)
  rejected_names = _rejected_names(rejected)
  required_closed = {
      "router_math_static_or_lagged_qkv_delta_predictors",
      "router_math_live_round_or_selected_affine_qkv_delta_approximation",
      "router_math_split_full_attention_projection_arithmetic_residual_fix",
  }
  missing_closed = sorted(required_closed - rejected_names)

  checks = [
      {
          "name": "seq305_selected_product_source_consumer_gate",
          "pass": (
              seq305.get("required_checks_passed") is True
              and seq305.get("decode_correctness_passed") is True
              and seq305.get("selected_next_route")
              == "router_prompt_all_linear_qkv_delta_product_source_from_layer_input_producer_source_gate"
              and _has_candidate(
                  routes, 305,
                  "accept_full_attention_residual_layer_input_producer_decode_gate")
              and _has_switch(
                  routes,
                  "select_router_prompt_all_linear_qkv_delta_product_source_from_layer_input_producer_source_gate",
                  305)
          ),
      },
      {
          "name": "producer_decode_evidence_covers_resident_values",
          "pass": (
              seq305.get("smoke_summary", {}).get(
                  "router_qkv_delta_layer_input_producer_source_layers")
              == len(PRODUCER_LAYERS) * DECODE_TOKENS
              and seq305.get("smoke_summary", {}).get(
                  "router_qkv_delta_layer_input_producer_source_values")
              == PRODUCER_VALUES
              and seq305.get("smoke_summary", {}).get(
                  "router_qkv_delta_layer_input_producer_source_misses") == 0
          ),
          "detail": {
              "expected_producer_values": PRODUCER_VALUES,
              "smoke_summary": seq305.get("smoke_summary"),
          },
      },
      {
          "name": "top512_consumer_requirement_preserved",
          "pass": TOP512_VALUES == 122880,
          "detail": {
              "all_linear_layers": ALL_LINEAR_LAYERS,
              "topk": TOPK,
              "decode_tokens": DECODE_TOKENS,
              "hidden_size": HIDDEN_SIZE,
              "expected_top512_values": TOP512_VALUES,
          },
      },
      {
          "name": "closed_approximations_stay_closed",
          "pass": not missing_closed,
          "detail": {"missing_closed_routes": missing_closed},
      },
      {
          "name": "source_has_producer_but_no_qkv_delta_consumer",
          "pass": (
              source["producer_source_present"]
              and source["component_source_still_source_only"]
              and source["product_consumer_absent"]),
          "detail": source,
      },
      {
          "name": "engine_lacks_device_sparse_overlay_primitive",
          "pass": (
              engine["resident_handle_primitives_present"]
              and engine["device_sparse_overlay_absent"]),
          "detail": engine,
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
          "seq305_decode_gate": _rel(args.seq305),
      },
      "producer_requirement": {
          "producer_layers": PRODUCER_LAYERS,
          "decode_tokens": DECODE_TOKENS,
          "hidden_size": HIDDEN_SIZE,
          "producer_values": PRODUCER_VALUES,
      },
      "consumer_requirement": {
          "all_linear_layers": ALL_LINEAR_LAYERS,
          "topk": TOPK,
          "decode_tokens": DECODE_TOKENS,
          "top512_values": TOP512_VALUES,
      },
      "source": source,
      "engine": engine,
      "checks": checks,
      "required_checks_passed": required,
      "qkv_delta_product_consumer_present": False,
      "target_compile_allowed": False,
      "decode_probe_allowed": False,
      "router_distribution_allowed": False,
      "speedup_claims_allowed": False,
      "disposition": (
          "reject_missing_qkv_delta_product_consumer_select_device_sparse_overlay_source"
          if required else
          "block_before_qkv_delta_product_consumer_route_selection"
      ),
      "selected_next_route": (
          "router_prompt_all_linear_qkv_delta_device_sparse_overlay_source_gate"
          if required else
          "router_prompt_all_linear_qkv_delta_product_source_from_layer_input_producer_fix_gate"
      ),
      "next_route_reason": (
          "The producer is now product-owned and decode-stable, but the "
          "all-linear qkv-delta consumer is still source-only and the engine "
          "has no device sparse-overlay primitive for selected top512 values. "
          "Add that primitive/source next before target compile, decode rows, "
          "router distribution rows, or speed promotion."
          if required else
          "The consumer route selection evidence is incomplete. Fix the source "
          "gate before adding compile/decode/distribution rows."
      ),
  }


def write_outputs(metrics: dict[str, Any], out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n",
      encoding="utf-8")
  failed = [row["name"] for row in metrics["checks"] if not row.get("pass")]
  lines = [
      "# Router QKV Delta Product Source From Producer Source Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- qkv_delta_product_consumer_present: `{str(metrics['qkv_delta_product_consumer_present']).lower()}`",
      f"- target_compile_allowed: `{str(metrics['target_compile_allowed']).lower()}`",
      f"- decode_probe_allowed: `{str(metrics['decode_probe_allowed']).lower()}`",
      f"- router_distribution_allowed: `{str(metrics['router_distribution_allowed']).lower()}`",
      f"- producer values: `{metrics['producer_requirement']['producer_values']}`",
      f"- top512 consumer values: `{metrics['consumer_requirement']['top512_values']}`",
      f"- failed_checks: `{failed}`",
      "",
      metrics["next_route_reason"],
      "",
      "This is route-control/source evidence. It does not compile, decode, or claim speed.",
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
  parser.add_argument("--seq305", type=Path, default=DEFAULT_SEQ305)
  parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
  return parser.parse_args()


def main() -> int:
  args = parse_args()
  metrics = compute(args)
  write_outputs(metrics, args.out_dir)
  print(json.dumps({
      "disposition": metrics["disposition"],
      "out_dir": _rel(args.out_dir),
      "qkv_delta_product_consumer_present": metrics[
          "qkv_delta_product_consumer_present"],
      "required_checks_passed": metrics["required_checks_passed"],
      "selected_next_route": metrics["selected_next_route"],
  }, sort_keys=True))
  return 0 if metrics["required_checks_passed"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
