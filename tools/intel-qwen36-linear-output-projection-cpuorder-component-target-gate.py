#!/usr/bin/env python3
"""Classify the layer0 CPU-order output-projection component target row."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = (
    "intel-qwen36-linear-output-projection-cpuorder-component-target-v0"
)
CURRENT_ROUTE = (
    "router_prompt_distribution_layer0_linear_output_projection_reduction_"
    "order_component_target_gate"
)
SELECTED_NEXT_ROUTE = (
    "router_prompt_distribution_layer0_linear_output_projection_cpuorder_"
    "source_gate"
)
MAX_ROWBLOCK16_COST_RATIO = 1.15


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
  predecessor = _load(args.predecessor)
  routes = _load(args.routes)
  target = _load(args.target)
  probe = target.get("probe")
  probe = probe if isinstance(probe, dict) else {}
  comparisons = probe.get("comparisons")
  comparisons = comparisons if isinstance(comparisons, dict) else {}
  cpu_order = comparisons.get("linear_attn_out_cpu_order")
  cpu_order = cpu_order if isinstance(cpu_order, dict) else {}
  gpu_vs_cpu = cpu_order.get("gpu_vs_cpu")
  gpu_vs_cpu = gpu_vs_cpu if isinstance(gpu_vs_cpu, dict) else {}
  timings = probe.get("timings")
  timings = timings if isinstance(timings, dict) else {}
  cpu_order_us = _num(
      timings.get("cpu_order_output_projection_gpu_kernel_min_us"))
  rowlane_us = _num(timings.get("output_projection_gpu_kernel_min_us"))
  rowblock16_us = _num(
      timings.get("rowblock16_output_projection_gpu_kernel_min_us"))
  rowblock16_cost_ratio = (
      cpu_order_us / rowblock16_us if rowblock16_us > 0.0 else 0.0)
  checks = [
      {
          "name": "seq554_selected_component_target_gate",
          "pass": (
              predecessor.get("required_checks_passed") is True
              and predecessor.get("target_component_allowed") is True
              and predecessor.get("selected_next_route") == CURRENT_ROUTE
              and _has_candidate(routes, 554, CURRENT_ROUTE)
              and _has_switch(
                  routes, 554,
                  "select_router_prompt_distribution_layer0_linear_output_"
                  "projection_reduction_order_component_target_gate")),
      },
      {"name": "layer0_arc_component_row_passed",
       "pass": (
           target.get("required_checks_passed") is True
           and target.get("layer") == 0
           and "Arc(TM) B390" in str(probe.get("device_name")))},
      {"name": "cpuorder_projection_is_bit_exact_vs_cpu",
       "pass": (
           _num(gpu_vs_cpu.get("max_abs_diff"), -1.0) == 0.0
           and _num(gpu_vs_cpu.get("rmse"), -1.0) == 0.0
           and _num(gpu_vs_cpu.get("mismatch_count"), -1.0) == 0.0)},
      {"name": "cpuorder_beats_rowlane_component",
       "pass": cpu_order_us > 0.0 and cpu_order_us < rowlane_us},
      {"name": "cpuorder_cost_is_close_to_rowblock16",
       "pass": (
           rowblock16_us > 0.0
           and rowblock16_cost_ratio <= MAX_ROWBLOCK16_COST_RATIO)},
  ]
  required = all(bool(row["pass"]) for row in checks)
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "sequence": args.sequence,
      "inputs": {
          "predecessor": _rel(args.predecessor),
          "routes": _rel(args.routes),
          "target": _rel(args.target),
      },
      "checks": checks,
      "required_checks_passed": required,
      "component": {
          "cpu_order_min_us": cpu_order_us,
          "rowlane_min_us": rowlane_us,
          "rowblock16_min_us": rowblock16_us,
          "cpu_order_vs_rowblock16_cost_ratio": rowblock16_cost_ratio,
          "cpu_order_gpu_vs_cpu_max_abs_diff": gpu_vs_cpu.get("max_abs_diff"),
          "cpu_order_gpu_vs_cpu_rmse": gpu_vs_cpu.get("rmse"),
      },
      "decode_source_gate_allowed": required,
      "decode_probe_allowed": False,
      "speedup_claims_allowed": False,
      "promotion_allowed": False,
      "disposition": (
          "accept_component_select_layer0_cpuorder_projection_source"
          if required else
          "reject_or_block_cpuorder_projection_component"),
      "selected_next_route": SELECTED_NEXT_ROUTE if required else CURRENT_ROUTE,
      "next_route_reason": (
          "The layer0 captured-payload CPU-order Q4 projection is bit-exact "
          "against the CPU matvec, faster than rowlane, and within 15% of "
          "rowblock16. Add one default-off layer selector and reuse the resident "
          "CPU-order runner for linear `ssm_out.weight` in source/generate-only "
          "form. Do not run a token before source and target compile gates pass."
          if required else
          "The component exactness or cost gate failed; do not wire decode."
      ),
  }


def write_outputs(metrics: dict[str, Any], out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  failed = [row["name"] for row in metrics["checks"] if not row["pass"]]
  component = metrics["component"]
  lines = [
      f"# Seq{metrics['sequence']} Linear Output CPU-Order Component Target Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- CPU-order / rowlane / rowblock16 min us: "
      f"`{component['cpu_order_min_us']}` / `{component['rowlane_min_us']}` / "
      f"`{component['rowblock16_min_us']}`",
      f"- CPU-order vs rowblock16 cost ratio: "
      f"`{component['cpu_order_vs_rowblock16_cost_ratio']}`",
      f"- CPU-order GPU vs CPU max abs: "
      f"`{component['cpu_order_gpu_vs_cpu_max_abs_diff']}`",
      f"- failed_checks: `{failed}`",
      "",
      metrics["next_route_reason"],
      "",
      "This is component correctness/timing evidence only. It is not decode speed.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--sequence", type=int, default=555)
  parser.add_argument(
      "--predecessor", type=Path,
      default=ROOT / "output/seq554-layer0-linear-output-projection-cpuorder-component-source-gate-20260710Tseq554Z/metrics.json")
  parser.add_argument("--routes", type=Path,
                      default=ACTIVE / "routes-ledger.json")
  parser.add_argument(
      "--target", type=Path,
      default=ROOT / "output/seq555-layer0-linear-output-projection-cpuorder-component-target-20260710Tseq555Z/probe.json")
  parser.add_argument(
      "--out-dir", type=Path,
      default=ROOT / "output/seq555-layer0-linear-output-projection-cpuorder-component-target-gate-20260710Tseq555Z")
  args = parser.parse_args()
  metrics = compute(args)
  write_outputs(metrics, args.out_dir)
  print(json.dumps({
      "required_checks_passed": metrics["required_checks_passed"],
      "disposition": metrics["disposition"],
      "decode_source_gate_allowed": metrics["decode_source_gate_allowed"],
      "selected_next_route": metrics["selected_next_route"],
      "out_dir": _rel(args.out_dir),
  }, sort_keys=True))
  return 0 if metrics["required_checks_passed"] else 2


if __name__ == "__main__":
  raise SystemExit(main())
