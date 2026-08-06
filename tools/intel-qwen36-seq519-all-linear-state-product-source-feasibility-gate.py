#!/usr/bin/env python3
"""Classify direct product feasibility for the all-linear state repair signal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = (
    "intel-qwen36-seq519-all-linear-state-product-source-feasibility-v0"
)

DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_REJECTED = ACTIVE / "rejected-routes.json"
DEFAULT_SEQ518 = (
    ROOT
    / "output/seq518-router-prompt-distribution-route-switch-gate-20260709Tseq518Z"
    / "metrics.json"
)
DEFAULT_MATH_STATE = (
    ROOT
    / "output/r2-gpu-router-math-distribution-rowblock16-26mask-double-swiglu-cpu-shadow-state-linear-all-20260708Tseq251Z"
    / "result.json"
)
DEFAULT_CODE_STATE = (
    ROOT
    / "output/r2-gpu-router-code-distribution-rowblock16-26mask-double-swiglu-cpu-shadow-state-linear-all-20260708Tseq256Z"
    / "result.json"
)
DEFAULT_CONV_STATE = (
    ROOT
    / "output/r2-gpu-router-math-distribution-rowblock16-26mask-double-swiglu-cpu-shadow-state-linear-conv-20260708Tseq258Z"
    / "result.json"
)
DEFAULT_SEQ515 = (
    ROOT
    / "output/seq515-current-token-qkv-delta-blockq16-router-distribution-gate-20260709Tseq515Z"
    / "metrics.json"
)
DEFAULT_DECODE_SOURCE = ROOT / "tools/intel-qwen36-r2-gpu-decode-smoke.py"
DEFAULT_ENGINE_SOURCE = ROOT / "engine/src/gpu_q4x8_matvec.cpp"
DEFAULT_KERNEL_SOURCE = ROOT / "engine/gpu/opencl/q4x8_matvec.cl"
DEFAULT_OUT_DIR = (
    ROOT
    / "output/seq519-all-linear-state-product-source-feasibility-gate-20260709Tseq519Z"
)

KLD_THRESHOLD = 0.005
TOP1_THRESHOLD = 0.99
ALL_LINEAR_LAYER_TOKENS = 30 * 8
SELECTED_NEXT_ROUTE = "router_prompt_linear_conv_history_unclosed_source_audit_gate"

REQUIRED_CLOSED_ROUTES = {
    "router_math_linear_recurrent_state_only_refresh",
    "router_math_resident_linear_state_host_sync_fix",
    "router_math_global_opencl_no_fma_conv_history_fix",
    "router_math_static_or_lagged_qkv_delta_predictors",
    "router_math_live_round_or_selected_affine_qkv_delta_approximation",
    "selected_layer_input_recursive_source_value_chase",
    "qkv_delta_producer_mapped_replacement_overlay_as_product_fix",
    "qkv_delta_blockq16_no_product_value_source",
}


def _load_json(path: Path) -> Any:
  return json.loads(path.read_text(encoding="utf-8"))


def _read(path: Path) -> str:
  return path.read_text(encoding="utf-8")


def _rel(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path)


def _num(value: Any, default: float = 0.0) -> float:
  return float(value) if isinstance(value, (int, float)) else default


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


def _smoke(path: Path) -> dict[str, Any]:
  payload = _load_json(path)
  smoke = payload.get("smoke")
  return smoke if isinstance(smoke, dict) else payload


def _distribution_summary(path: Path) -> dict[str, Any]:
  smoke = _smoke(path)
  dist = smoke.get("distribution_ladder")
  if not isinstance(dist, dict):
    dist = {}
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
      "cpu_shadow_state_tokens": smoke.get("cpu_shadow_state_tokens"),
      "cpu_shadow_state_wall_ns": smoke.get("cpu_shadow_state_wall_ns"),
      "resident_linear_state_enabled": smoke.get("resident_linear_state_enabled"),
      "resident_linear_state_layers": smoke.get("resident_linear_state_layers"),
      "sync_resident_linear_state_host": smoke.get(
          "sync_resident_linear_state_host"),
      "resident_linear_state_host_sync_layers": smoke.get(
          "resident_linear_state_host_sync_layers"),
  }


def _state_shadow_pass(row: dict[str, Any]) -> bool:
  return (
      row.get("required_checks_passed") is True
      and row.get("distribution_required_checks_passed") is True
      and _num(row.get("max_kld")) <= KLD_THRESHOLD
      and _num(row.get("top1_rate")) >= TOP1_THRESHOLD
      and row.get("cpu_shadow_state_each_token_enabled") is True
      and _num(row.get("cpu_shadow_state_tokens")) == 8
      and _num(row.get("cpu_shadow_state_wall_ns")) > 5_000_000_000
      and row.get("resident_linear_state_enabled") is True
      and _num(row.get("resident_linear_state_layers")) == ALL_LINEAR_LAYER_TOKENS
      and row.get("sync_resident_linear_state_host") is False)


def _conv_partial(row: dict[str, Any]) -> bool:
  return (
      row.get("required_checks_passed") is False
      and row.get("distribution_required_checks_passed") is False
      and _num(row.get("max_kld")) > KLD_THRESHOLD
      and _num(row.get("top1_rate")) >= TOP1_THRESHOLD
      and row.get("cpu_shadow_state_each_token_enabled") is True)


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
      and all(row.get("cpu_shadow_state_each_token_enabled") is False for row in rows)
      and all(row.get("distribution_required_checks_passed") is False for row in rows)
      and any(_num(row.get("top1_rate")) < TOP1_THRESHOLD for row in rows)
      and any(_num(row.get("max_kld")) > KLD_THRESHOLD for row in rows))


def _source_findings(decode_source: Path,
                     engine_source: Path,
                     kernel_source: Path) -> dict[str, Any]:
  decode = _read(decode_source)
  engine = _read(engine_source)
  kernel = _read(kernel_source)
  findings = {
      "resident_device_state_path_present": all(s in decode for s in [
          "DecodeWarmResidentLinearState",
          "DecodeRefreshResidentLinearStateLayerComponents",
          "DecodeSwapResidentLinearConvStateHandles",
          "DecodeResidentLinearStateHandle(layer)",
          "DecodeResidentLinearConvStateHandle(layer)",
      ]) and all(s in engine for s in [
          "RunLinearAttentionDeltaResidentState",
          "RunPostConvPrepThenLinearAttentionDeltaResidentState",
          "resident_state.buffer",
      ]) and "linear_attn_conv_f32" in kernel,
      "cpu_shadow_refresh_path_present": all(s in decode for s in [
          "args.cpu_shadow_state_each_token",
          "RunNativeDecodeToken(",
          "apply_cpu_shadow_state",
          "DecodeRefreshResidentLinearStateLayers(",
      ]),
      "refresh_uploads_host_f32_buffers": all(s in decode for s in [
          "UploadF32Buffer(state.linear_recurrent[layer])",
          "UploadF32Buffer(state.linear_conv[layer])",
      ]),
      "direct_all_linear_product_source_contract_present": (
          "DecodeAllLinearStateProductSourceContract" in decode
          or "all_linear_state_product_source" in decode),
  }
  findings["direct_product_source_absent"] = (
      findings["resident_device_state_path_present"]
      and findings["cpu_shadow_refresh_path_present"]
      and findings["refresh_uploads_host_f32_buffers"]
      and not findings["direct_all_linear_product_source_contract_present"])
  return findings


def compute(args: argparse.Namespace) -> dict[str, Any]:
  routes = _load_json(args.routes)
  rejected = _load_json(args.rejected)
  seq518 = _load_json(args.seq518)
  rejected_names = _rejected_names(rejected)
  missing_closed = sorted(REQUIRED_CLOSED_ROUTES - rejected_names)
  math_state = _distribution_summary(args.math_state)
  code_state = _distribution_summary(args.code_state)
  conv_state = _distribution_summary(args.conv_state)
  product_rows = _seq515_product_rows(args.seq515)
  source_findings = _source_findings(
      args.decode_source, args.engine_source, args.kernel_source)

  checks = [
      {
          "name": "seq518_selected_feasibility_gate",
          "pass": (
              seq518.get("required_checks_passed") is True
              and seq518.get("selected_next_route")
              == "router_prompt_all_linear_state_product_source_feasibility_gate"
              and _has_candidate(
                  routes, 518,
                  "accept_route_switch_select_all_linear_state_product_source_feasibility")
              and _has_switch(
                  routes,
                  "select_router_prompt_all_linear_state_product_source_feasibility_gate",
                  518)),
      },
      {
          "name": "passing_all_linear_rows_are_cpu_shadow_refresh",
          "pass": _state_shadow_pass(math_state) and _state_shadow_pass(code_state),
          "detail": {"math": math_state, "code": code_state},
      },
      {
          "name": "conv_only_state_signal_is_partial",
          "pass": _conv_partial(conv_state),
          "detail": conv_state,
      },
      {
          "name": "current_product_state_path_still_fails_distribution",
          "pass": _product_rows_fail(product_rows),
          "detail": {"seq515_product_rows": product_rows},
      },
      {
          "name": "direct_product_source_not_present_in_runtime",
          "pass": source_findings["direct_product_source_absent"],
          "detail": source_findings,
      },
      {
          "name": "closed_substitutes_acknowledged",
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
          "seq518": _rel(args.seq518),
          "math_state": _rel(args.math_state),
          "code_state": _rel(args.code_state),
          "conv_state": _rel(args.conv_state),
          "seq515": _rel(args.seq515),
          "decode_source": _rel(args.decode_source),
          "engine_source": _rel(args.engine_source),
          "kernel_source": _rel(args.kernel_source),
      },
      "checks": checks,
      "required_checks_passed": required,
      "speedup_claims_allowed": False,
      "decode_probe_allowed": False,
      "router_distribution_allowed": False,
      "direct_all_linear_state_product_source_feasible": False,
      "disposition": (
          "reject_direct_all_linear_state_product_source_feasibility_select_conv_history_reframe"
          if required else
          "block_all_linear_state_product_source_feasibility_inconsistent_evidence"),
      "selected_next_route": (
          SELECTED_NEXT_ROUTE if required else
          "router_prompt_all_linear_state_product_source_feasibility_gate"),
      "next_route_reason": (
          "The passing all-linear state rows are CPU-shadow refreshes that run "
          "native decode each token and upload F32 recurrent/conv state; the "
          "existing device-resident product state path is already present and "
          "still fails router distribution without shadow values. The only "
          "unclosed state-side signal is the conv-history/qkv-history partial "
          "repair, but the current block-q16 qkv-delta value source is closed. "
          "Next audit the remaining unclosed conv-history/upstream residual "
          "source class before any decode, speed, or long-context row."
          if required else
          "Feasibility evidence is inconsistent; do not launch a token row."),
      "evidence_summary": {
          "all_linear_shadow_math": math_state,
          "all_linear_shadow_code": code_state,
          "conv_only_partial": conv_state,
          "product_distribution_rows": product_rows,
          "source_findings": source_findings,
      },
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
  summary = metrics["evidence_summary"]
  lines = [
      "# Seq519 All-Linear State Product Source Feasibility Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- failed_checks: `{failed}`",
      f"- direct_all_linear_state_product_source_feasible: `{str(metrics['direct_all_linear_state_product_source_feasible']).lower()}`",
      f"- speedup_claims_allowed: `{str(metrics['speedup_claims_allowed']).lower()}`",
      "",
      "## Evidence",
      "",
      f"- all-linear shadow math/code KLD: `{summary['all_linear_shadow_math']['max_kld']}` / `{summary['all_linear_shadow_code']['max_kld']}`",
      f"- all-linear shadow wall ns: `{summary['all_linear_shadow_math']['cpu_shadow_state_wall_ns']}` / `{summary['all_linear_shadow_code']['cpu_shadow_state_wall_ns']}`",
      f"- conv-only partial KLD/top1: `{summary['conv_only_partial']['max_kld']}` / `{summary['conv_only_partial']['top1_rate']}`",
      "- current product rows still fail router distribution without CPU shadow",
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
  parser.add_argument("--seq518", type=Path, default=DEFAULT_SEQ518)
  parser.add_argument("--math-state", type=Path, default=DEFAULT_MATH_STATE)
  parser.add_argument("--code-state", type=Path, default=DEFAULT_CODE_STATE)
  parser.add_argument("--conv-state", type=Path, default=DEFAULT_CONV_STATE)
  parser.add_argument("--seq515", type=Path, default=DEFAULT_SEQ515)
  parser.add_argument("--decode-source", type=Path, default=DEFAULT_DECODE_SOURCE)
  parser.add_argument("--engine-source", type=Path, default=DEFAULT_ENGINE_SOURCE)
  parser.add_argument("--kernel-source", type=Path, default=DEFAULT_KERNEL_SOURCE)
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
