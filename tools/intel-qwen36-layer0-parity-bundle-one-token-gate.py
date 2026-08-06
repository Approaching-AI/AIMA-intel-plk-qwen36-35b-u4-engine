#!/usr/bin/env python3
"""Classify the layer0 norm-to-projection parity bundle on one token."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = "intel-qwen36-layer0-parity-bundle-one-token-v0"
CURRENT_ROUTE = (
    "router_prompt_distribution_layer0_norm_to_projection_parity_bundle_"
    "one_token_gate"
)
SELECTED_NEXT_ROUTE = (
    "router_prompt_distribution_layer0_norm_to_projection_parity_bundle_"
    "router_math_distribution_gate"
)
KLD_THRESHOLD = 0.005
SUB_ULP_BOUND = 1.192092896e-7


def _load(path: Path) -> dict[str, Any]:
  payload = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(payload, dict):
    raise TypeError(f"{path} does not contain a JSON object")
  return payload


def _rel(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path)


def _num(value: Any, default: float = 0.0) -> float:
  return float(value) if isinstance(value, (int, float)) else default


def _smoke(payload: dict[str, Any]) -> dict[str, Any]:
  smoke = payload.get("smoke")
  return smoke if isinstance(smoke, dict) else payload


def _has_candidate(routes: dict[str, Any], seq: int,
                   next_route: str) -> bool:
  return any(isinstance(row, dict) and row.get("seq") == seq
             and row.get("selected_next_route") == next_route
             for row in routes.get("candidate_history", []))


def _has_switch(routes: dict[str, Any], seq: int, decision: str) -> bool:
  return any(isinstance(row, dict) and row.get("seq_covered") == seq
             and row.get("decision") == decision and row.get("resolved") is True
             for row in routes.get("switch_decisions", []))


def _layer(smoke: dict[str, Any], key: str, layer_id: int) -> dict[str, Any]:
  steps = smoke.get(key)
  if not isinstance(steps, list) or not steps:
    return {}
  layers = steps[0].get("layers") if isinstance(steps[0], dict) else []
  for row in layers if isinstance(layers, list) else []:
    if isinstance(row, dict) and row.get("layer") == layer_id:
      return row
  return {}


def compute(args: argparse.Namespace) -> dict[str, Any]:
  predecessor = _load(args.predecessor)
  routes = _load(args.routes)
  candidate_payload = _load(args.candidate)
  candidate = _smoke(candidate_payload)
  projection_only = _smoke(_load(args.projection_only))
  dist = candidate.get("distribution_ladder")
  dist = dist if isinstance(dist, dict) else {}
  preconv0 = _layer(candidate, "linear_preconv_source_diff_by_step", 0)
  projection0 = _layer(candidate, "linear_projection_source_diff_by_step", 0)
  boundary0 = _layer(candidate, "layer_boundary_diff_by_step", 0)
  boundary1 = _layer(candidate, "layer_boundary_diff_by_step", 1)
  projection_only_boundary1 = _layer(
      projection_only, "layer_boundary_diff_by_step", 1)
  checks = [
      {"name": "seq560_selected_one_token_bundle",
       "pass": (
           predecessor.get("required_checks_passed") is True
           and predecessor.get("one_token_bundle_allowed") is True
           and predecessor.get("selected_next_route") == CURRENT_ROUTE
           and _has_candidate(routes, 560, CURRENT_ROUTE)
           and _has_switch(
               routes, 560,
               "select_router_prompt_distribution_layer0_norm_to_projection_"
               "parity_bundle_one_token_gate"))},
      {"name": "bounded_combined_one_token_row_passed",
       "pass": (
           candidate_payload.get("target", {}).get("run", {}).get("returncode") == 0
           and candidate.get("required_checks_passed") is True
           and candidate.get("case_id") == "router_math_reason_001"
           and candidate.get("decode_tokens_per_session") == 1
           and candidate.get("input_rmsnorm_serial_reduction_layers") == [0]
           and candidate.get("linear_output_projection_cpu_order_layers") == [0]
           and dist.get("required_checks_passed") is True
           and _num(dist.get("max_kld")) <= KLD_THRESHOLD
           and _num(dist.get("top1_rate")) == 1.0)},
      {"name": "layer0_norm_qkv_z_and_projection_are_exact",
       "pass": (
           _num(preconv0.get("gpu_attn_norm_vs_cpu_max_abs_diff"), -1.0) == 0.0
           and _num(preconv0.get("qkv_from_gpu_attn_norm_max_abs_diff"), -1.0) == 0.0
           and _num(preconv0.get("z_from_gpu_attn_norm_max_abs_diff"), -1.0) == 0.0
           and _num(projection0.get(
               "gpu_output_vs_cpu_projection_from_gpu_input_max_abs_diff"),
                    -1.0) == 0.0)},
      {"name": "layer0_and_layer1_outputs_remain_sub_ulp",
       "pass": (
           0.0 < _num(boundary0.get("output_max_abs_diff")) <= SUB_ULP_BOUND
           and 0.0 < _num(boundary1.get("output_max_abs_diff")) <= SUB_ULP_BOUND)},
      {"name": "bundle_prevents_projection_only_layer1_amplification",
       "pass": (
           _num(boundary1.get("output_max_abs_diff"))
           < _num(projection_only_boundary1.get("output_max_abs_diff")))},
  ]
  required = all(bool(row["pass"]) for row in checks)
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "sequence": args.sequence,
      "inputs": {
          "predecessor": _rel(args.predecessor),
          "routes": _rel(args.routes),
          "candidate": _rel(args.candidate),
          "projection_only": _rel(args.projection_only),
      },
      "checks": checks,
      "required_checks_passed": required,
      "candidate_max_kld": dist.get("max_kld"),
      "layer0_output_max_abs_diff": boundary0.get("output_max_abs_diff"),
      "layer1_output_max_abs_diff": boundary1.get("output_max_abs_diff"),
      "projection_only_layer1_output_max_abs_diff": (
          projection_only_boundary1.get("output_max_abs_diff")),
      "router_math_distribution_allowed": required,
      "router_code_distribution_allowed": False,
      "speedup_claims_allowed": False,
      "promotion_allowed": False,
      "disposition": (
          "accept_bundle_one_token_select_router_math_distribution"
          if required else "reject_or_block_layer0_parity_bundle_one_token"),
      "selected_next_route": SELECTED_NEXT_ROUTE if required else CURRENT_ROUTE,
      "next_route_reason": (
          "The combined token passes KLD/top-1, closes both selected layer0 "
          "boundaries, and prevents the projection-only layer1 amplification. "
          "Run only the 8-token router-math row next; code and speed remain blocked."
          if required else
          "The combined token or structural closure failed; do not continue."
      ),
  }


def write_outputs(metrics: dict[str, Any], out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  failed = [row["name"] for row in metrics["checks"] if not row["pass"]]
  lines = [
      f"# Seq{metrics['sequence']} Layer0 Parity Bundle One-Token Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- candidate max KLD: `{metrics['candidate_max_kld']}`",
      f"- layer0/layer1 output max abs: `{metrics['layer0_output_max_abs_diff']}` / `{metrics['layer1_output_max_abs_diff']}`",
      f"- projection-only layer1 max abs: `{metrics['projection_only_layer1_output_max_abs_diff']}`",
      f"- failed_checks: `{failed}`",
      "",
      metrics["next_route_reason"],
      "",
      "This is one-token correctness evidence only. It is not a speed claim.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--sequence", type=int, default=561)
  parser.add_argument(
      "--predecessor", type=Path,
      default=ROOT / "output/seq560-post-arithmetic-parity-route-control-gate-20260710Tseq560Z/metrics.json")
  parser.add_argument("--routes", type=Path, default=ACTIVE / "routes-ledger.json")
  parser.add_argument(
      "--candidate", type=Path,
      default=ROOT / "output/seq561-layer0-norm-to-projection-parity-bundle-one-token-20260710Tseq561Z/result.json")
  parser.add_argument(
      "--projection-only", type=Path,
      default=ROOT / "output/seq558-layer0-linear-output-projection-cpuorder-one-token-20260710Tseq558Z/result.json")
  parser.add_argument(
      "--out-dir", type=Path,
      default=ROOT / "output/seq561-layer0-norm-to-projection-parity-bundle-one-token-gate-20260710Tseq561Z")
  args = parser.parse_args()
  metrics = compute(args)
  write_outputs(metrics, args.out_dir)
  print(json.dumps({
      "required_checks_passed": metrics["required_checks_passed"],
      "disposition": metrics["disposition"],
      "router_math_distribution_allowed": metrics["router_math_distribution_allowed"],
      "selected_next_route": metrics["selected_next_route"],
      "out_dir": _rel(args.out_dir),
  }, sort_keys=True))
  return 0 if metrics["required_checks_passed"] else 2


if __name__ == "__main__":
  raise SystemExit(main())
