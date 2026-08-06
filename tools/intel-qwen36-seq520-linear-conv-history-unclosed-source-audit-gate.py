#!/usr/bin/env python3
"""Audit remaining conv-history/upstream source routes after seq519."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = (
    "intel-qwen36-seq520-linear-conv-history-unclosed-source-audit-v0"
)

DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_REJECTED = ACTIVE / "rejected-routes.json"
DEFAULT_SEQ506 = (
    ROOT / "output/seq506-reentry-route-control-gate-20260709Tseq506Z"
    / "metrics.json"
)
DEFAULT_SEQ517 = (
    ROOT
    / "output/seq517-current-token-qkv-delta-blockq16-value-source-gate-20260709Tseq517Z"
    / "metrics.json"
)
DEFAULT_SEQ519 = (
    ROOT
    / "output/seq519-all-linear-state-product-source-feasibility-gate-20260709Tseq519Z"
    / "metrics.json"
)
DEFAULT_CONV_STATE = (
    ROOT
    / "output/r2-gpu-router-math-distribution-rowblock16-26mask-double-swiglu-cpu-shadow-state-linear-conv-20260708Tseq258Z"
    / "result.json"
)
DEFAULT_QKV_MATH = (
    ROOT
    / "output/r2-gpu-router-math-distribution-rowblock16-26mask-double-swiglu-shadow-linear-qkvcol-delta-blockq16-top512-20260708Tseq264Z"
    / "result.json"
)
DEFAULT_QKV_CODE = (
    ROOT
    / "output/r2-gpu-router-code-distribution-rowblock16-26mask-double-swiglu-shadow-all-linear-qkvcol-delta-blockq16-top512-20260708Tseq291Z"
    / "result.json"
)
DEFAULT_SEQ515 = (
    ROOT
    / "output/seq515-current-token-qkv-delta-blockq16-router-distribution-gate-20260709Tseq515Z"
    / "metrics.json"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "output/seq520-linear-conv-history-unclosed-source-audit-gate-20260709Tseq520Z"
)

KLD_THRESHOLD = 0.005
TOP1_THRESHOLD = 0.99
ALL_LINEAR_LAYER_TOKENS = 30 * 8
TOP512_VALUES = 30 * 8 * 512
SELECTED_NEXT_ROUTE = "router_prompt_distribution_correctness_route_reflection_gate"

REQUIRED_CLOSED_ROUTES = {
    "router_math_state_repair_wrong_subsets_and_qklocal_cpu_shape",
    "router_math_linear_recurrent_state_only_refresh",
    "router_math_global_opencl_no_fma_conv_history_fix",
    "router_math_resident_linear_state_host_sync_fix",
    "router_math_static_or_lagged_qkv_delta_predictors",
    "router_math_carrier_loop_or_q4_cpu_order_full_attention_projection_residual_fix",
    "router_math_split_full_attention_projection_arithmetic_residual_fix",
    "router_math_live_round_or_selected_affine_qkv_delta_approximation",
    "shared_rmsnorm_scale_kernel_serial_cpu_sqrt_product_fix",
    "selected_layer_input_recursive_source_value_chase",
    "qkv_delta_producer_mapped_replacement_overlay_as_product_fix",
    "qkv_delta_blockq16_no_product_value_source",
    "all_linear_state_direct_product_refresh",
}


def _load_json(path: Path) -> Any:
  return json.loads(path.read_text(encoding="utf-8"))


def _rel(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path)


def _num(value: Any, default: float = 0.0) -> float:
  return float(value) if isinstance(value, (int, float)) else default


def _smoke(path: Path) -> dict[str, Any]:
  payload = _load_json(path)
  smoke = payload.get("smoke")
  return smoke if isinstance(smoke, dict) else payload


def _dist(smoke: dict[str, Any]) -> dict[str, Any]:
  dist = smoke.get("distribution_ladder")
  return dist if isinstance(dist, dict) else {}


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


def _rejected_names(rejected: dict[str, Any]) -> set[str]:
  names: set[str] = set()
  for row in rejected.get("rejected", []):
    if isinstance(row, dict) and isinstance(row.get("route"), str):
      names.add(row["route"])
  return names


def _state_row(path: Path) -> dict[str, Any]:
  smoke = _smoke(path)
  dist = _dist(smoke)
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
      "cpu_shadow_state_realign_components": smoke.get(
          "cpu_shadow_state_realign_components"),
      "cpu_shadow_state_realign_layer_ids": smoke.get(
          "cpu_shadow_state_realign_layer_ids"),
      "cpu_shadow_state_tokens": smoke.get("cpu_shadow_state_tokens"),
      "cpu_shadow_state_wall_ns": smoke.get("cpu_shadow_state_wall_ns"),
      "resident_linear_state_enabled": smoke.get(
          "resident_linear_state_enabled"),
      "resident_linear_state_layers": smoke.get("resident_linear_state_layers"),
      "sync_resident_linear_state_host": smoke.get(
          "sync_resident_linear_state_host"),
  }


def _conv_partial(row: dict[str, Any]) -> bool:
  return (
      row.get("required_checks_passed") is False
      and row.get("distribution_required_checks_passed") is False
      and row.get("cpu_shadow_state_each_token_enabled") is True
      and row.get("cpu_shadow_state_realign_components") == "linear_conv"
      and _num(row.get("resident_linear_state_layers")) == ALL_LINEAR_LAYER_TOKENS
      and _num(row.get("max_kld")) > KLD_THRESHOLD
      and _num(row.get("top1_rate")) >= TOP1_THRESHOLD)


def _qkv_delta_row(path: Path) -> dict[str, Any]:
  smoke = _smoke(path)
  dist = _dist(smoke)
  return {
      "path": _rel(path),
      "case_id": smoke.get("case_id"),
      "required_checks_passed": smoke.get("required_checks_passed"),
      "distribution_required_checks_passed": dist.get("required_checks_passed"),
      "max_kld": dist.get("max_kld"),
      "top1_rate": dist.get("top1_rate"),
      "min_logits_cosine": dist.get("min_logits_cosine"),
      "position_count": dist.get("position_count"),
      "layer_tokens": smoke.get("cpu_shadow_layer_input_delta_layers"),
      "topk": smoke.get("cpu_shadow_layer_input_delta_topk"),
      "selector": smoke.get("cpu_shadow_layer_input_delta_selector"),
      "value_mode": smoke.get("cpu_shadow_layer_input_delta_value_mode"),
      "values": smoke.get("cpu_shadow_layer_input_delta_values"),
  }


def _qkv_shadow_pass(row: dict[str, Any]) -> bool:
  return (
      row.get("required_checks_passed") is True
      and row.get("distribution_required_checks_passed") is True
      and _num(row.get("max_kld")) <= KLD_THRESHOLD
      and _num(row.get("top1_rate")) >= TOP1_THRESHOLD
      and _num(row.get("layer_tokens")) == ALL_LINEAR_LAYER_TOKENS
      and _num(row.get("topk")) == 512
      and row.get("selector") == "linear_qkv_col_abs"
      and row.get("value_mode") == "shadow_delta_block_q16"
      and _num(row.get("values")) == TOP512_VALUES)


def _seq515_product_rows(path: Path) -> list[dict[str, Any]]:
  payload = _load_json(path)
  rows: list[dict[str, Any]] = []
  for row in payload.get("runs", []):
    if not isinstance(row, dict):
      continue
    summary = row.get("summary")
    if not isinstance(summary, dict):
      continue
    dist = summary.get("distribution")
    if not isinstance(dist, dict):
      dist = {}
    rows.append({
        "case_id": summary.get("case_id"),
        "required_checks_passed": summary.get("required_checks_passed"),
        "distribution_required_checks_passed": dist.get("required_checks_passed"),
        "max_kld": dist.get("max_kld"),
        "top1_rate": dist.get("top1_rate"),
        "min_logits_cosine": dist.get("min_logits_cosine"),
        "cpu_shadow_state_each_token_enabled": summary.get(
            "cpu_shadow_state_each_token_enabled"),
        "blockq16_layers": summary.get("blockq16_layers"),
        "blockq16_values": summary.get("blockq16_values"),
        "blockq16_misses": summary.get("blockq16_misses"),
    })
  return rows


def _product_rows_fail(rows: list[dict[str, Any]]) -> bool:
  expected_cases = {"router_math_reason_001", "router_code_reason_002"}
  got_cases = {str(row.get("case_id")) for row in rows}
  return (
      expected_cases.issubset(got_cases)
      and all(row.get("cpu_shadow_state_each_token_enabled") is False
              for row in rows)
      and all(row.get("distribution_required_checks_passed") is False
              for row in rows)
      and any(_num(row.get("max_kld")) > KLD_THRESHOLD for row in rows)
      and any(_num(row.get("top1_rate")) < TOP1_THRESHOLD for row in rows))


def compute(args: argparse.Namespace) -> dict[str, Any]:
  routes = _load_json(args.routes)
  rejected = _load_json(args.rejected)
  seq506 = _load_json(args.seq506)
  seq517 = _load_json(args.seq517)
  seq519 = _load_json(args.seq519)
  conv_state = _state_row(args.conv_state)
  qkv_math = _qkv_delta_row(args.qkv_math)
  qkv_code = _qkv_delta_row(args.qkv_code)
  product_rows = _seq515_product_rows(args.seq515)
  rejected_names = _rejected_names(rejected)
  missing_closed = sorted(REQUIRED_CLOSED_ROUTES - rejected_names)

  reentry_match = seq506.get("product_failure_match")
  if not isinstance(reentry_match, dict):
    reentry_match = {}

  checks = [
      {
          "name": "seq519_selected_conv_history_source_audit",
          "pass": (
              seq519.get("required_checks_passed") is True
              and seq519.get("selected_next_route")
              == "router_prompt_linear_conv_history_unclosed_source_audit_gate"
              and _has_candidate(
                  routes, 519,
                  "reject_direct_all_linear_state_product_source_feasibility_select_conv_history_reframe")
              and _has_switch(
                  routes,
                  "select_router_prompt_linear_conv_history_unclosed_source_audit_gate",
                  519)),
      },
      {
          "name": "conv_history_signal_is_shadow_partial_not_product",
          "pass": _conv_partial(conv_state),
          "detail": conv_state,
      },
      {
          "name": "qkv_delta_shadow_lower_bound_is_only_passing_signal",
          "pass": _qkv_shadow_pass(qkv_math) and _qkv_shadow_pass(qkv_code),
          "detail": {"math": qkv_math, "code": qkv_code},
      },
      {
          "name": "current_product_source_still_fails_router_distribution",
          "pass": _product_rows_fail(product_rows),
          "detail": {"seq515_product_rows": product_rows},
      },
      {
          "name": "recursive_upstream_source_chase_reentered_known_failure",
          "pass": (
              seq506.get("required_checks_passed") is True
              and seq506.get("selected_next_route")
              == "router_prompt_all_linear_current_token_qkv_delta_recursion_break_design_gate"
              and reentry_match.get("all_match") is True),
          "detail": {
              "seq506_disposition": seq506.get("disposition"),
              "product_failure_match": reentry_match,
              "source_gap": seq506.get("source_gap"),
          },
      },
      {
          "name": "blockq16_value_source_route_closed",
          "pass": (
              seq517.get("required_checks_passed") is True
              and seq517.get("disposition")
              == "reject_blockq16_value_source_no_product_source_select_route_switch"),
          "detail": {
              "seq517_disposition": seq517.get("disposition"),
              "seq517_selected_next_route": seq517.get("selected_next_route"),
          },
      },
      {
          "name": "known_conv_history_source_classes_closed",
          "pass": not missing_closed,
          "detail": {"missing_closed_routes": missing_closed},
      },
  ]
  required = all(bool(row.get("pass")) for row in checks)
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "inputs": {
          "routes": _rel(args.routes),
          "rejected": _rel(args.rejected),
          "seq506": _rel(args.seq506),
          "seq517": _rel(args.seq517),
          "seq519": _rel(args.seq519),
          "conv_state": _rel(args.conv_state),
          "qkv_math": _rel(args.qkv_math),
          "qkv_code": _rel(args.qkv_code),
          "seq515": _rel(args.seq515),
      },
      "checks": checks,
      "required_checks_passed": required,
      "speedup_claims_allowed": False,
      "decode_probe_allowed": False,
      "router_distribution_allowed": False,
      "known_product_source_classes_exhausted": required,
      "closed_source_classes": sorted(REQUIRED_CLOSED_ROUTES),
      "remaining_diagnostic_signals": {
          "conv_only_shadow_partial": conv_state,
          "all_linear_qkv_delta_top512_shadow": {
              "math": qkv_math,
              "code": qkv_code,
          },
      },
      "product_failure_rows": product_rows,
      "disposition": (
          "close_known_conv_history_product_source_classes_select_correctness_route_reflection"
          if required else
          "block_conv_history_source_audit_inconsistent_evidence"),
      "selected_next_route": (
          SELECTED_NEXT_ROUTE if required else
          "router_prompt_linear_conv_history_unclosed_source_audit_gate"),
      "next_route_reason": (
          "The remaining conv-history/qkv-history signals are diagnostic only: "
          "conv-only CPU-shadow state refresh is partial, all-linear top512 "
          "qkv-column deltas pass only from shadow_delta_block_q16 values, the "
          "current product path still fails router distribution, and the "
          "recursive upstream product-source chase re-entered the same failure. "
          "Known non-shadow substitutes are closed, so the next unit is a "
          "no-token correctness-route reflection gate. It must introduce a new "
          "non-shadow source class or change correctness strategy; decode, "
          "router-distribution reruns, speed promotion, and long-context rows "
          "remain blocked."
          if required else
          "The source audit evidence is inconsistent; do not switch routes or "
          "launch a token row."),
  }


def write_outputs(metrics: dict[str, Any], out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n",
      encoding="utf-8")
  failed = [
      row["name"] for row in metrics["checks"]
      if row.get("pass") is not True
  ]
  signals = metrics["remaining_diagnostic_signals"]
  conv = signals["conv_only_shadow_partial"]
  qkv = signals["all_linear_qkv_delta_top512_shadow"]
  lines = [
      "# Seq520 Linear Conv-History Source Audit Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- failed_checks: `{failed}`",
      f"- known_product_source_classes_exhausted: `{str(metrics['known_product_source_classes_exhausted']).lower()}`",
      f"- speedup_claims_allowed: `{str(metrics['speedup_claims_allowed']).lower()}`",
      "",
      "## Evidence",
      "",
      f"- conv-only shadow KLD/top1: `{conv['max_kld']}` / `{conv['top1_rate']}`",
      f"- top512 shadow qkv math/code KLD: `{qkv['math']['max_kld']}` / `{qkv['code']['max_kld']}`",
      "- current product rows still fail router distribution without CPU shadow",
      f"- closed source classes audited: `{len(metrics['closed_source_classes'])}`",
      "",
      metrics["next_route_reason"],
      "",
      "This is route-control/correctness evidence only. It is not a speed claim.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--routes", type=Path, default=DEFAULT_ROUTES)
  parser.add_argument("--rejected", type=Path, default=DEFAULT_REJECTED)
  parser.add_argument("--seq506", type=Path, default=DEFAULT_SEQ506)
  parser.add_argument("--seq517", type=Path, default=DEFAULT_SEQ517)
  parser.add_argument("--seq519", type=Path, default=DEFAULT_SEQ519)
  parser.add_argument("--conv-state", type=Path, default=DEFAULT_CONV_STATE)
  parser.add_argument("--qkv-math", type=Path, default=DEFAULT_QKV_MATH)
  parser.add_argument("--qkv-code", type=Path, default=DEFAULT_QKV_CODE)
  parser.add_argument("--seq515", type=Path, default=DEFAULT_SEQ515)
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
