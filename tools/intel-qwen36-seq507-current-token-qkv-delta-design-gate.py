#!/usr/bin/env python3
"""Design gate for the current-token qkv-delta recursion break."""

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
    "intel-qwen36-seq507-current-token-qkv-delta-design-gate-v0"
)

DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_REJECTED = ACTIVE / "rejected-routes.json"
DEFAULT_SEQ316 = (
    ROOT
    / "output/router-qkv-delta-product-consumer-router-distribution-gate-20260708Tseq316Z"
    / "metrics.json"
)
DEFAULT_SEQ506 = (
    ROOT
    / "output/seq506-reentry-route-control-gate-20260709Tseq506Z"
    / "metrics.json"
)
DEFAULT_MATH_TOP512 = (
    ROOT
    / "output/r2-gpu-router-math-distribution-rowblock16-26mask-double-swiglu-shadow-linear-qkvcol-delta-blockq16-top512-20260708Tseq264Z"
    / "result.json"
)
DEFAULT_CODE_TOP512 = (
    ROOT
    / "output/r2-gpu-router-code-distribution-rowblock16-26mask-double-swiglu-shadow-all-linear-qkvcol-delta-blockq16-top512-20260708Tseq291Z"
    / "result.json"
)
DEFAULT_DECODE_SOURCE = ROOT / "tools/intel-qwen36-r2-gpu-decode-smoke.py"
DEFAULT_ENGINE_SOURCE = ROOT / "engine/src/gpu_q4x8_matvec.cpp"
DEFAULT_KERNEL_SOURCE = ROOT / "engine/gpu/opencl/q4x8_matvec.cl"
DEFAULT_OUT_DIR = (
    ROOT / "output/seq507-current-token-qkv-delta-design-gate-20260709Tseq507Z"
)

ALL_LINEAR_LAYERS = [
    0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18,
    20, 21, 22, 24, 25, 26, 28, 29, 30, 32, 33, 34, 36, 37, 38,
]
PRODUCER_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31, 35]
DECODE_TOKENS = 8
TOPK = 512
EXPECTED_LOWER_BOUND_VALUES = len(ALL_LINEAR_LAYERS) * DECODE_TOKENS * TOPK
EXPECTED_PRODUCT_VALUES = (len(ALL_LINEAR_LAYERS) - 3) * DECODE_TOKENS * TOPK
EXPECTED_PRODUCT_MISSES = 3 * DECODE_TOKENS
KLD_THRESHOLD = 0.005
TOP1_THRESHOLD = 0.99
SELECTED_NEXT_ROUTE = (
    "router_prompt_all_linear_current_token_qkv_delta_blockq16_source_contract_gate"
)


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


def _num(value: Any, default: float = 0.0) -> float:
  return float(value) if isinstance(value, (int, float)) else default


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
      for row in routes.get("candidate_history", []))


def _has_switch(routes: dict[str, Any], decision: str, seq_covered: int) -> bool:
  return any(
      isinstance(row, dict)
      and row.get("decision") == decision
      and _num(row.get("seq_covered")) >= seq_covered
      and row.get("resolved") is True
      for row in routes.get("switch_decisions", []))


def _has_rejected_route(rejected: dict[str, Any], route: str) -> bool:
  return any(
      isinstance(row, dict) and row.get("route") == route
      for row in rejected.get("rejected", []))


def _smoke(path: Path) -> dict[str, Any]:
  payload = _load_json(path)
  smoke = payload.get("smoke")
  return smoke if isinstance(smoke, dict) else payload


def _top512_row(path: Path) -> dict[str, Any]:
  smoke = _smoke(path)
  dist = smoke.get("distribution_ladder")
  dist = dist if isinstance(dist, dict) else {}
  return {
      "path": _rel(path),
      "case_id": smoke.get("case_id"),
      "required_checks_passed": smoke.get("required_checks_passed"),
      "distribution_required_checks_passed": dist.get("required_checks_passed"),
      "max_kld": dist.get("max_kld"),
      "top1_rate": dist.get("top1_rate"),
      "min_logits_cosine": dist.get("min_logits_cosine"),
      "cpu_shadow_state_each_token_enabled": smoke.get(
          "cpu_shadow_state_each_token_enabled"),
      "delta_layers": smoke.get("cpu_shadow_layer_input_delta_layer_ids"),
      "delta_values": smoke.get("cpu_shadow_layer_input_delta_values"),
      "delta_topk": smoke.get("cpu_shadow_layer_input_delta_topk"),
      "delta_selector": smoke.get("cpu_shadow_layer_input_delta_selector"),
      "delta_value_mode": smoke.get("cpu_shadow_layer_input_delta_value_mode"),
  }


def _top512_pass(row: dict[str, Any]) -> bool:
  return (
      row.get("required_checks_passed") is True
      and row.get("distribution_required_checks_passed") is True
      and _num(row.get("max_kld")) <= KLD_THRESHOLD
      and _num(row.get("top1_rate")) >= TOP1_THRESHOLD
      and row.get("cpu_shadow_state_each_token_enabled") is True
      and row.get("delta_layers") == ALL_LINEAR_LAYERS
      and row.get("delta_values") == EXPECTED_LOWER_BOUND_VALUES
      and row.get("delta_topk") == TOPK
      and row.get("delta_selector") == "linear_qkv_col_abs"
      and row.get("delta_value_mode") == "shadow_delta_block_q16")


def _seq316_rows(seq316: dict[str, Any]) -> list[dict[str, Any]]:
  rows = []
  for run in seq316.get("runs", []) or []:
    if isinstance(run, dict) and isinstance(run.get("summary"), dict):
      rows.append(run["summary"])
  return rows


def _seq316_shape(seq316_rows: list[dict[str, Any]]) -> dict[str, Any]:
  return {
      "rows": [
          {
              "case_id": row.get("case_id"),
              "product_layers": row.get("product_layers"),
              "product_values": row.get("product_values"),
              "product_misses": row.get("product_misses"),
              "producer_layers": row.get("producer_layers"),
              "producer_values": row.get("producer_values"),
              "producer_misses": row.get("producer_misses"),
              "max_kld": (row.get("distribution") or {}).get("max_kld"),
              "top1_rate": (row.get("distribution") or {}).get("top1_rate"),
          }
          for row in seq316_rows
      ],
      "product_values_expected_if_all30": EXPECTED_LOWER_BOUND_VALUES,
      "product_values_observed_expected": EXPECTED_PRODUCT_VALUES,
      "product_misses_expected": EXPECTED_PRODUCT_MISSES,
  }


def _seq316_producer_overlay_shortfall(seq316_rows: list[dict[str, Any]]) -> bool:
  return bool(seq316_rows) and all(
      row.get("product_values") == EXPECTED_PRODUCT_VALUES
      and row.get("product_misses") == EXPECTED_PRODUCT_MISSES
      and row.get("producer_layers") == len(PRODUCER_LAYERS) * DECODE_TOKENS
      and row.get("producer_misses") == 0
      and _num((row.get("distribution") or {}).get("max_kld")) > KLD_THRESHOLD
      for row in seq316_rows)


def _source_shape(decode_source: str, engine_source: str,
                  kernel_source: str) -> dict[str, Any]:
  producer_overlay_present = [
      _present(decode_source, "producer_layer_mapping",
               "DecodeRouterQkvDeltaProductProducerLayerForConsumer",
               regex=False),
      _present(decode_source, "producer_handle_vector",
               "g_decode_router_qkv_delta_full_attention_residual_source_handles",
               regex=False),
      _present(decode_source, "product_source_handle_from_producer",
               r"const std::uint64_t source_handle\s*=\s*"
               r"g_decode_router_qkv_delta_full_attention_residual_source_handles"),
      _present(decode_source, "consumer_uses_prev_layer_base",
               "g_decode_prev_layer_output_handle, residual, args.repeat, stats",
               regex=False),
  ]
  replacement_overlay_present = [
      _present(kernel_source, "sparse_overlay_kernel",
               "qkv_delta_sparse_overlay_f32", regex=False),
      _present(kernel_source, "replacement_write",
               "output[index] = source[index]", regex=False),
      _present(engine_source, "copy_base_before_overlay",
               "clEnqueueCopyBuffer(queue_, base.buffer, out_buffer", regex=False),
      _present(engine_source, "run_selected_value_overlay_api",
               "RunRouterQkvDeltaSelectedValueOverlay", regex=False),
  ]
  blockq16_product_absent = [
      _absent(decode_source + "\n" + engine_source + "\n" + kernel_source,
              "no_product_blockq16_overlay",
              "qkv_delta_blockq16", regex=False),
      _absent(decode_source + "\n" + engine_source + "\n" + kernel_source,
              "no_current_token_qkv_delta_contract",
              "CurrentTokenQkvDeltaBlockQ16", regex=False),
  ]
  diagnostic_blockq16_present = [
      _present(decode_source, "diagnostic_blockq16_value_mode",
               "shadow_delta_block_q16", regex=False),
      _present(decode_source, "diagnostic_live_plus_quantized_delta",
               "static_cast<double>(live[index]) + q * scale", regex=False),
  ]
  return {
      "producer_mapped_replacement_overlay_present": (
          _all_present(producer_overlay_present)
          and _all_present(replacement_overlay_present)),
      "blockq16_product_source_absent": _all_absent(blockq16_product_absent),
      "diagnostic_blockq16_path_present": _all_present(diagnostic_blockq16_present),
      "producer_overlay_checks": producer_overlay_present,
      "replacement_overlay_checks": replacement_overlay_present,
      "blockq16_absent_checks": blockq16_product_absent,
      "diagnostic_blockq16_checks": diagnostic_blockq16_present,
  }


def compute(args: argparse.Namespace) -> dict[str, Any]:
  routes = _load_json(args.routes)
  rejected = _load_json(args.rejected)
  seq316 = _load_json(args.seq316)
  seq506 = _load_json(args.seq506)
  math_top512 = _top512_row(args.math_top512)
  code_top512 = _top512_row(args.code_top512)
  seq316_rows = _seq316_rows(seq316)
  source = _source_shape(
      _read(args.decode_source), _read(args.engine_source),
      _read(args.kernel_source))
  checks = [
      {
          "name": "seq506_selected_recursion_break_design_gate",
          "pass": (
              seq506.get("required_checks_passed") is True
              and seq506.get("selected_next_route")
              == "router_prompt_all_linear_current_token_qkv_delta_recursion_break_design_gate"
              and _has_candidate(
                  routes, 506,
                  "accept_reentry_route_control_select_current_token_qkv_delta_recursion_break")
              and _has_switch(
                  routes,
                  "select_router_prompt_all_linear_current_token_qkv_delta_recursion_break_design_gate",
                  506)),
      },
      {
          "name": "lower_bound_requires_all30_current_token_blockq16",
          "pass": _top512_pass(math_top512) and _top512_pass(code_top512),
          "detail": {"math": math_top512, "code": code_top512},
      },
      {
          "name": "current_product_consumer_is_producer_mapped_and_short",
          "pass": _seq316_producer_overlay_shortfall(seq316_rows),
          "detail": _seq316_shape(seq316_rows),
      },
      {
          "name": "source_shape_mismatch_identified",
          "pass": (
              source["producer_mapped_replacement_overlay_present"]
              and source["blockq16_product_source_absent"]
              and source["diagnostic_blockq16_path_present"]),
          "detail": source,
      },
      {
          "name": "recursive_and_approximation_routes_closed",
          "pass": (
              _has_rejected_route(
                  rejected, "selected_layer_input_recursive_source_value_chase")
              and _has_rejected_route(
                  rejected,
                  "router_math_static_or_lagged_qkv_delta_predictors")
              and _has_rejected_route(
                  rejected,
                  "router_math_live_round_or_selected_affine_qkv_delta_approximation")),
      },
  ]
  required = all(row.get("pass") is True for row in checks)
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "inputs": {
          "routes": _rel(args.routes),
          "rejected": _rel(args.rejected),
          "seq316": _rel(args.seq316),
          "seq506": _rel(args.seq506),
          "math_top512": _rel(args.math_top512),
          "code_top512": _rel(args.code_top512),
          "decode_source": _rel(args.decode_source),
          "decode_source_sha256": _sha256(args.decode_source),
          "engine_source": _rel(args.engine_source),
          "engine_source_sha256": _sha256(args.engine_source),
          "kernel_source": _rel(args.kernel_source),
          "kernel_source_sha256": _sha256(args.kernel_source),
      },
      "checks": checks,
      "correction_shape": {
          "layers": ALL_LINEAR_LAYERS,
          "layer_count": len(ALL_LINEAR_LAYERS),
          "topk": TOPK,
          "decode_tokens": DECODE_TOKENS,
          "required_values": EXPECTED_LOWER_BOUND_VALUES,
          "value_mode": "shadow_delta_block_q16",
          "selector": "linear_qkv_col_abs",
      },
      "rejected_current_shape": {
          "producer_layers": PRODUCER_LAYERS,
          "product_values": EXPECTED_PRODUCT_VALUES,
          "product_misses": EXPECTED_PRODUCT_MISSES,
          "reason": (
              "producer-mapped replacement overlay cannot cover the first "
              "three linear consumers of each token and does not implement the "
              "block-q16 current-token delta used by the passing diagnostic"),
      },
      "required_checks_passed": required,
      "disposition": (
          "accept_current_token_blockq16_qkv_delta_design_contract"
          if required else
          "block_current_token_qkv_delta_design_gate"),
      "selected_next_route": SELECTED_NEXT_ROUTE if required else str(
          seq506.get("selected_next_route")),
      "next_contract": {
          "route": SELECTED_NEXT_ROUTE,
          "must_do": [
              "add a default-off product source contract before token rows",
              "cover all 30 linear consumer layers including 0/1/2",
              "use qkv-column top512 selection from the live current-token layer input",
              "apply a sparse block-q16 delta overlay, not producer-value replacement",
              "stay CPU-shadow-free and host-sync-free",
          ],
          "must_not_do": [
              "depend on future producer handles for the first group",
              "reopen recursive selected layer-input source-value tracing",
              "retry static, lagged, rounded, or affine qkv-delta approximations",
              "claim speed or expand to long context before router distribution passes",
          ],
      },
      "decode_probe_allowed": False,
      "router_distribution_allowed": False,
      "speedup_claims_allowed": False,
  }


def write_outputs(metrics: dict[str, Any], out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n",
      encoding="utf-8")
  failed = [
      row["name"] for row in metrics["checks"] if row.get("pass") is not True
  ]
  shape = metrics["correction_shape"]
  rejected = metrics["rejected_current_shape"]
  lines = [
      "# Seq507 Current-Token QKV Delta Design Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- failed_checks: `{failed}`",
      "",
      f"- lower-bound values: `{shape['required_values']}` "
      f"(`{shape['layer_count']}` layers x `{shape['topk']}` x `{shape['decode_tokens']}` tokens)",
      f"- lower-bound selector/value mode: `{shape['selector']}` / `{shape['value_mode']}`",
      f"- rejected product-consumer values/misses: `{rejected['product_values']}` / `{rejected['product_misses']}`",
      "",
      "The current product consumer is a producer-mapped replacement overlay. "
      "The passing lower bound is an all-30-linear current-token qkv-column "
      "block-q16 delta correction. The next source unit must implement that "
      "contract before any decode or router-distribution row.",
      "",
      "This is design/source route-control evidence only. It is not a speed claim.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--routes", type=Path, default=DEFAULT_ROUTES)
  parser.add_argument("--rejected", type=Path, default=DEFAULT_REJECTED)
  parser.add_argument("--seq316", type=Path, default=DEFAULT_SEQ316)
  parser.add_argument("--seq506", type=Path, default=DEFAULT_SEQ506)
  parser.add_argument("--math-top512", type=Path, default=DEFAULT_MATH_TOP512)
  parser.add_argument("--code-top512", type=Path, default=DEFAULT_CODE_TOP512)
  parser.add_argument("--decode-source", type=Path, default=DEFAULT_DECODE_SOURCE)
  parser.add_argument("--engine-source", type=Path, default=DEFAULT_ENGINE_SOURCE)
  parser.add_argument("--kernel-source", type=Path, default=DEFAULT_KERNEL_SOURCE)
  parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  metrics = compute(args)
  write_outputs(metrics, args.out_dir)
  print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
