#!/usr/bin/env python3
"""Classify the failed exact-preprojection component without target work."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = (
    "intel-qwen36-all-linear-preprojection-parity-component-route-close-v0"
)
CURRENT_ROUTE = (
    "router_prompt_distribution_all_linear_preprojection_parity_component_route_close_gate"
)
SELECTED_NEXT_ROUTE = (
    "router_prompt_distribution_all_linear_preprojection_parity_component_source_repair_gate"
)


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


def _has_candidate(routes: dict[str, Any], seq: int,
                   next_route: str) -> bool:
  return any(
      isinstance(row, dict)
      and row.get("seq") == seq
      and row.get("selected_next_route") == next_route
      for row in routes.get("candidate_history", []))


def _has_switch(routes: dict[str, Any], seq: int, decision: str) -> bool:
  return any(
      isinstance(row, dict)
      and row.get("seq_covered") == seq
      and row.get("decision") == decision
      and row.get("resolved") is True
      for row in routes.get("switch_decisions", []))


def compute(args: argparse.Namespace) -> dict[str, Any]:
  routes = _load(args.routes)
  predecessor = _load(args.predecessor)
  design = _load(args.design)
  opencl = args.opencl_source.read_text(encoding="utf-8")
  cpu = args.cpu_source.read_text(encoding="utf-8")
  rows = predecessor.get("rows", [])
  rows = rows if isinstance(rows, list) else []
  predecessor_selects = (
      predecessor.get("measurement_complete") is True
      and predecessor.get("component_passed") is False
      and predecessor.get("decode_source_allowed") is False
      and predecessor.get("selected_next_route") == CURRENT_ROUTE
      and len(rows) == 2
      and all(isinstance(row, dict) for row in rows)
      and all(row.get("budget_passed") is True for row in rows)
      and all(row.get("exactness_passed") is False for row in rows)
      and _has_candidate(routes, 600, CURRENT_ROUTE)
      and _has_switch(
          routes, 600,
          "select_router_prompt_distribution_all_linear_preprojection_"
          "parity_component_route_close_gate"))
  projection_stays_exact = all(
      row.get("exact_comparisons", {}).get("exact_projection_vs_cpu") is True
      for row in rows)
  q6_is_first_failed_boundary = all(
      row.get("exact_comparisons", {}).get("exact_qkv_vs_cpu") is False
      for row in rows)
  cpu_lane_major = all(marker in cpu for marker in [
      "float sums[8] = {};",
      "accumulate_q6_k_q8_k_block_direct(",
      "sums);",
      "for (const float lane_sum : sums)",
      "sum += lane_sum;",
  ])
  exact_kernel_begin = opencl.find(
      "__kernel void q6k_linear_qkv_cpuorder_nofma(")
  exact_kernel_end = opencl.find(
      "__kernel void linear_attn_conv_cpuorder_nofma_f32(",
      exact_kernel_begin)
  exact_q6 = (
      opencl[exact_kernel_begin:exact_kernel_end]
      if exact_kernel_begin >= 0 and exact_kernel_end > exact_kernel_begin
      else "")
  opencl_is_block_major = all(marker in exact_q6 for marker in [
      "float sum = 0.0f;",
      "for (uint block_index = 0; block_index < blocks_per_row; ++block_index)",
      "const float block_lane = combined_scale * (float)lane_sums[lane];",
      "sum = sum + block_lane;",
  ]) and "float sums[8]" not in exact_q6
  design_requires_cpu_order = (
      design.get("required_checks_passed") is True
      and design.get("design", {}).get("candidate")
      == "cpuorder_preprojection_bundle_v1"
      and design.get("design", {}).get("implementation", {}).get(
          "q6_qkv", {}).get("kernel")
      == "q6k_linear_qkv_cpuorder_nofma"
      and "block-major then lane-major float accumulation" in str(
          design.get("design", {}).get("implementation", {}).get(
              "q6_qkv", {}).get("rule", "")))
  no_runtime_evidence = not any(
      (args.out_dir / name).exists()
      for name in ("run.json", "probe.json", "tokens.jsonl"))
  repairable_contract_violation = (
      predecessor_selects and projection_stays_exact
      and q6_is_first_failed_boundary and cpu_lane_major
      and opencl_is_block_major and design_requires_cpu_order)
  repair_contract = {
      "scope": "one_source_repair_then_recompile_and_repeat_component",
      "q6_change": (
          "keep eight float lane accumulators across Q6 blocks, then reduce "
          "lanes 0..7 after the block loop"),
      "must_preserve": [
          "scoped FP_CONTRACT OFF",
          "existing raw Q6 layout and one work item per row",
          "zero new dispatches, uploads, or readbacks in decode",
          "default decode wiring unchanged",
          "97/97 flag ceiling",
      ],
      "component_rerun": {
          "required": True,
          "repeat_and_confirm": True,
          "all_exact_boundaries_bit_exact": True,
          "changed_shell_added_us_max": 6.841858993929781,
      },
      "stop_condition": (
          "If a fresh binary still fails any exact boundary, close the whole "
          "bundle; do not open another arithmetic-axis sweep."),
  }
  checks = [
      {"name": "seq600_selected_no_target_route_close_control",
       "pass": predecessor_selects},
      {"name": "timing_and_projection_pass_but_q6_is_first_exact_failure",
       "pass": projection_stays_exact and q6_is_first_failed_boundary},
      {"name": "locked_design_requires_cpu_direct_dot_lane_order",
       "pass": design_requires_cpu_order},
      {"name": "source_proves_block_major_vs_lane_major_contract_mismatch",
       "pass": cpu_lane_major and opencl_is_block_major,
       "detail": {
           "cpu_lane_major": cpu_lane_major,
           "opencl_block_major": opencl_is_block_major,
       }},
      {"name": "route_close_gate_created_no_runtime_evidence",
       "pass": no_runtime_evidence},
  ]
  required = all(bool(row["pass"]) for row in checks)
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "sequence": args.sequence,
      "inputs": {
          "routes": _rel(args.routes),
          "predecessor": _rel(args.predecessor),
          "design": _rel(args.design),
          "opencl_source": _rel(args.opencl_source),
          "cpu_source": _rel(args.cpu_source),
      },
      "repairable_contract_violation": repairable_contract_violation,
      "repair_contract": repair_contract,
      "checks": checks,
      "required_checks_passed": required,
      "source_repair_allowed": required and repairable_contract_violation,
      "target_compile_allowed": False,
      "component_probe_allowed": False,
      "decode_integration_allowed": False,
      "token_row_allowed": False,
      "validation_or_test_allowed": False,
      "speedup_claims_allowed": False,
      "promotion_allowed": False,
      "disposition": (
          "repair_cpuorder_preprojection_q6_lane_reduction_contract"
          if required and repairable_contract_violation else
          "close_cpuorder_preprojection_bundle_v1"),
      "selected_next_route": (
          SELECTED_NEXT_ROUTE
          if required and repairable_contract_violation else CURRENT_ROUTE),
      "next_route_reason": (
          "The component failure is not yet a design falsification: the new "
          "Q6 kernel violates the locked CPU direct-dot order. Apply exactly "
          "one lane-major source repair, compile locally, then target-compile "
          "a fresh component binary before one final paired component row."
          if required and repairable_contract_violation else
          "The failed component cannot be attributed to the named source "
          "contract. Close the bundle without another target row."),
  }


def write_outputs(metrics: dict[str, Any], out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  (out_dir / "manifest.json").write_text(
      json.dumps({
          "schema_version": metrics["schema_version"],
          "workstream": metrics["workstream"],
          "tool": _rel(Path(__file__)),
          "inputs": metrics["inputs"],
          "repairable_contract_violation": metrics[
              "repairable_contract_violation"],
          "selected_next_route": metrics["selected_next_route"],
          "target_compile_allowed": False,
          "component_probe_allowed": False,
          "token_row_allowed": False,
      }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  failed = [row["name"] for row in metrics["checks"] if not row["pass"]]
  lines = [
      f"# Seq{metrics['sequence']} Exact Preprojection Route Control",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- repairable_contract_violation: `{str(metrics['repairable_contract_violation']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- failed_checks: `{failed}`",
      "",
      metrics["next_route_reason"],
      "",
      "This gate used existing source and artifacts only; no target command ran.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--sequence", type=int, default=601)
  parser.add_argument("--routes", type=Path,
                      default=ACTIVE / "routes-ledger.json")
  parser.add_argument(
      "--predecessor", type=Path,
      default=ROOT / (
          "output/seq600-all-linear-preprojection-parity-component-probe-"
          "gate-20260710Tseq600Z/metrics.json"))
  parser.add_argument(
      "--design", type=Path,
      default=ROOT / (
          "output/seq597-all-linear-preprojection-parity-budget-design-"
          "gate-20260710Tseq597Z/metrics.json"))
  parser.add_argument("--opencl-source", type=Path,
                      default=ROOT / "engine/gpu/opencl/q4x8_matvec.cl")
  parser.add_argument("--cpu-source", type=Path,
                      default=ROOT / "engine/src/gguf_loader.cpp")
  parser.add_argument(
      "--out-dir", type=Path,
      default=ROOT / (
          "output/seq601-all-linear-preprojection-parity-component-route-"
          "close-gate-20260710Tseq601Z"))
  args = parser.parse_args()
  metrics = compute(args)
  write_outputs(metrics, args.out_dir)
  print(json.dumps({
      "required_checks_passed": metrics["required_checks_passed"],
      "repairable_contract_violation": metrics[
          "repairable_contract_violation"],
      "disposition": metrics["disposition"],
      "selected_next_route": metrics["selected_next_route"],
      "out_dir": _rel(args.out_dir),
  }, sort_keys=True))
  return 0 if metrics["required_checks_passed"] else 2


if __name__ == "__main__":
  raise SystemExit(main())
