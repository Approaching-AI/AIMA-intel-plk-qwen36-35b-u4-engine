#!/usr/bin/env python3
"""Select the next router-distribution route after qkv-delta closure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = "intel-qwen36-seq518-router-prompt-distribution-route-switch-v0"

DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_REJECTED = ACTIVE / "rejected-routes.json"
DEFAULT_SEQ517 = (
    ROOT
    / "output/seq517-current-token-qkv-delta-blockq16-value-source-gate-20260709Tseq517Z"
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
DEFAULT_OUT_DIR = (
    ROOT
    / "output/seq518-router-prompt-distribution-route-switch-gate-20260709Tseq518Z"
)

ALL_LINEAR_LAYERS = [
    0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18,
    20, 21, 22, 24, 25, 26, 28, 29, 30, 32, 33, 34, 36, 37, 38,
]
KLD_THRESHOLD = 0.005
TOP1_THRESHOLD = 0.99
SELECTED_NEXT_ROUTE = "router_prompt_all_linear_state_product_source_feasibility_gate"
REQUIRED_CLOSED_ROUTES = {
    "qkv_delta_blockq16_no_product_value_source",
    "selected_layer_input_recursive_source_value_chase",
    "qkv_delta_producer_mapped_replacement_overlay_as_product_fix",
    "router_math_static_or_lagged_qkv_delta_predictors",
    "router_math_live_round_or_selected_affine_qkv_delta_approximation",
    "shared_rmsnorm_scale_kernel_serial_cpu_sqrt_product_fix",
    "router_math_resident_linear_state_host_sync_fix",
    "router_math_global_opencl_no_fma_conv_history_fix",
    "router_math_state_repair_wrong_subsets_and_qklocal_cpu_shape",
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


def _dist(smoke: dict[str, Any]) -> dict[str, Any]:
  dist = smoke.get("distribution_ladder")
  return dist if isinstance(dist, dict) else {}


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
      "cpu_shadow_state_realign_layer_ids": smoke.get(
          "cpu_shadow_state_realign_layer_ids"),
      "cpu_shadow_state_realign_components": smoke.get(
          "cpu_shadow_state_realign_components"),
      "cpu_shadow_state_tokens": smoke.get("cpu_shadow_state_tokens"),
      "cpu_shadow_state_wall_ns": smoke.get("cpu_shadow_state_wall_ns"),
      "resident_linear_state_enabled": smoke.get("resident_linear_state_enabled"),
      "resident_linear_state_layers": smoke.get("resident_linear_state_layers"),
      "resident_linear_state_host_sync_layers": smoke.get(
          "resident_linear_state_host_sync_layers"),
      "sync_resident_linear_state_host": smoke.get(
          "sync_resident_linear_state_host"),
  }


def _state_pass(row: dict[str, Any]) -> bool:
  return (
      row.get("required_checks_passed") is True
      and row.get("distribution_required_checks_passed") is True
      and _num(row.get("max_kld")) <= KLD_THRESHOLD
      and _num(row.get("top1_rate")) >= TOP1_THRESHOLD
      and row.get("cpu_shadow_state_each_token_enabled") is True
      and row.get("cpu_shadow_state_realign_layer_ids") == ALL_LINEAR_LAYERS
      and row.get("cpu_shadow_state_realign_components") in (None, "all")
      and row.get("resident_linear_state_enabled") is True)


def _conv_partial(row: dict[str, Any]) -> bool:
  return (
      row.get("required_checks_passed") is False
      and row.get("distribution_required_checks_passed") is False
      and row.get("cpu_shadow_state_each_token_enabled") is True
      and row.get("cpu_shadow_state_realign_layer_ids") == ALL_LINEAR_LAYERS
      and row.get("cpu_shadow_state_realign_components") == "linear_conv"
      and _num(row.get("max_kld")) > KLD_THRESHOLD
      and _num(row.get("top1_rate")) >= TOP1_THRESHOLD)


def compute(args: argparse.Namespace) -> dict[str, Any]:
  routes = _load_json(args.routes)
  rejected = _load_json(args.rejected)
  seq517 = _load_json(args.seq517)
  rejected_names = _rejected_names(rejected)
  missing_closed = sorted(REQUIRED_CLOSED_ROUTES - rejected_names)
  math_state = _state_row(args.math_state)
  code_state = _state_row(args.code_state)
  conv_state = _state_row(args.conv_state)

  checks = [
      {
          "name": "seq517_selected_route_switch",
          "pass": (
              seq517.get("required_checks_passed") is True
              and seq517.get("disposition")
              == "reject_blockq16_value_source_no_product_source_select_route_switch"
              and seq517.get("selected_next_route")
              == "router_prompt_distribution_route_switch_gate"
              and _has_candidate(
                  routes, 517,
                  "reject_blockq16_value_source_no_product_source_select_route_switch")
              and _has_switch(
                  routes,
                  "select_router_prompt_distribution_route_switch_gate",
                  517)),
      },
      {
          "name": "closed_routes_acknowledged",
          "pass": not missing_closed,
          "detail": {"missing_closed_routes": missing_closed},
      },
      {
          "name": "all_linear_cpu_shadow_state_passes_math_and_code",
          "pass": _state_pass(math_state) and _state_pass(code_state),
          "detail": {"math": math_state, "code": code_state},
      },
      {
          "name": "conv_only_state_is_partial_not_promotion",
          "pass": _conv_partial(conv_state),
          "detail": conv_state,
      },
  ]
  required = all(bool(row.get("pass")) for row in checks)
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "inputs": {
          "routes": _rel(args.routes),
          "rejected": _rel(args.rejected),
          "seq517": _rel(args.seq517),
          "math_state": _rel(args.math_state),
          "code_state": _rel(args.code_state),
          "conv_state": _rel(args.conv_state),
      },
      "checks": checks,
      "required_checks_passed": required,
      "speedup_claims_allowed": False,
      "router_distribution_allowed": False,
      "decode_probe_allowed": False,
      "diagnostic_state_rows": {
          "all_linear_math": math_state,
          "all_linear_code": code_state,
          "conv_only_math": conv_state,
      },
      "disposition": (
          "accept_route_switch_select_all_linear_state_product_source_feasibility"
          if required else
          "block_before_router_distribution_route_switch"),
      "selected_next_route": (
          SELECTED_NEXT_ROUTE if required else
          "router_prompt_distribution_route_switch_gate"),
      "next_route_reason": (
          "The qkv-delta/block-q16 and recursive selected layer-input routes "
          "are closed. The remaining passing diagnostic is all-linear "
          "CPU-shadow state refresh for router math/code, while conv-only state "
          "refresh is only a partial signal. The next high-signal unit is a "
          "feasibility gate for a product-owned all-linear state source; no "
          "decode, speed, or long-context row is allowed before that source is "
          "proven or closed."
          if required else
          "Route-switch evidence is inconsistent; do not launch another target "
          "probe from this state."),
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
  diag = metrics["diagnostic_state_rows"]
  lines = [
      "# Seq518 Router Prompt Distribution Route Switch Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- failed_checks: `{failed}`",
      f"- speedup_claims_allowed: `{str(metrics['speedup_claims_allowed']).lower()}`",
      "",
      "## Diagnostic Rows",
      "",
      f"- math all-linear state: max KLD `{diag['all_linear_math']['max_kld']}`, top1 `{diag['all_linear_math']['top1_rate']}`",
      f"- code all-linear state: max KLD `{diag['all_linear_code']['max_kld']}`, top1 `{diag['all_linear_code']['top1_rate']}`",
      f"- math conv-only state: max KLD `{diag['conv_only_math']['max_kld']}`, top1 `{diag['conv_only_math']['top1_rate']}`",
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
  parser.add_argument("--seq517", type=Path, default=DEFAULT_SEQ517)
  parser.add_argument("--math-state", type=Path, default=DEFAULT_MATH_STATE)
  parser.add_argument("--code-state", type=Path, default=DEFAULT_CODE_STATE)
  parser.add_argument("--conv-state", type=Path, default=DEFAULT_CONV_STATE)
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
