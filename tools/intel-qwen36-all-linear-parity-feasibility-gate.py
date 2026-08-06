#!/usr/bin/env python3
"""Apply a floor kill-number to the all-linear parity bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = "intel-qwen36-all-linear-parity-feasibility-v0"
CURRENT_ROUTE = (
    "router_prompt_distribution_all_linear_norm_to_projection_parity_"
    "feasibility_gate"
)
SELECTED_NEXT_ROUTE = (
    "router_prompt_distribution_non_arithmetic_product_source_route_control_gate"
)
REJECTED_ROUTE = (
    "router_prompt_distribution_all_linear_norm_to_projection_parity_bundle"
)
LINEAR_LAYERS = (
    0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18,
    20, 21, 22, 24, 25, 26, 28, 29, 30, 32, 33, 34, 36, 37, 38,
)
FLOOR_TOK_S = 19.5


def _load(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise TypeError(f"{path} does not contain a JSON object")
  return value


def _rel(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path)


def _num(value: Any, default: float = 0.0) -> float:
  return float(value) if isinstance(value, (int, float)) else default


def _has_candidate(routes: dict[str, Any], seq: int, route: str) -> bool:
  return any(isinstance(row, dict) and row.get("seq") == seq
             and row.get("selected_next_route") == route
             for row in routes.get("candidate_history", []))


def _has_switch(routes: dict[str, Any], seq: int, decision: str) -> bool:
  return any(isinstance(row, dict) and row.get("seq_covered") == seq
             and row.get("decision") == decision and row.get("resolved") is True
             for row in routes.get("switch_decisions", []))


def _candidate(routes: dict[str, Any], seq: int) -> dict[str, Any]:
  return next((row for row in routes.get("candidate_history", [])
               if isinstance(row, dict) and row.get("seq") == seq), {})


def compute(args: argparse.Namespace) -> dict[str, Any]:
  predecessor = _load(args.predecessor)
  routes = _load(args.routes)
  component = _load(args.component)
  generated = args.generated_cpp.read_text(encoding="utf-8")
  speed_row = _candidate(routes, 221)
  component_times = component.get("component", {})
  baseline_tok_s = _num(speed_row.get("decode_tok_s"))
  floor_tok_s = _num(speed_row.get("floor_tok_s"), FLOOR_TOK_S)
  cpuorder_us = _num(component_times.get("cpu_order_min_us"))
  rowblock16_us = _num(component_times.get("rowblock16_min_us"))
  per_layer_extra_us = cpuorder_us - rowblock16_us
  projection_extra_us = per_layer_extra_us * len(LINEAR_LAYERS)
  baseline_token_us = 1_000_000.0 / baseline_tok_s
  floor_token_us = 1_000_000.0 / floor_tok_s
  floor_headroom_us = floor_token_us - baseline_token_us
  lower_bound_ratio = (
      projection_extra_us / floor_headroom_us if floor_headroom_us > 0 else 0.0)
  checks = [
      {"name": "seq562_selected_all_linear_feasibility",
       "pass": (
           predecessor.get("required_checks_passed") is True
           and predecessor.get("broad_parity_feasibility_allowed") is True
           and predecessor.get("selected_next_route") == CURRENT_ROUTE
           and _has_candidate(routes, 562, CURRENT_ROUTE)
           and _has_switch(
               routes, 562,
               "select_router_prompt_distribution_all_linear_norm_to_projection_"
               "parity_feasibility_gate"))},
      {"name": "all_linear_layer_set_is_complete",
       "pass": len(LINEAR_LAYERS) == 30 and len(set(LINEAR_LAYERS)) == 30},
      {"name": "both_parameterized_selectors_exist",
       "pass": (
           "g_decode_input_rmsnorm_serial_reduction_layers" in generated
           and "g_decode_linear_output_projection_cpu_order_layers" in generated)},
      {"name": "short_floor_candidate_and_component_cost_are_recorded",
       "pass": (
           baseline_tok_s == 19.57836215 and floor_tok_s == FLOOR_TOK_S
           and cpuorder_us > rowblock16_us > 0.0)},
      {"name": "projection_only_lower_bound_exceeds_floor_headroom",
       "pass": projection_extra_us > floor_headroom_us > 0.0},
  ]
  required = all(bool(row["pass"]) for row in checks)
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "sequence": args.sequence,
      "inputs": {"predecessor": _rel(args.predecessor),
                 "routes": _rel(args.routes),
                 "component": _rel(args.component),
                 "generated_cpp": _rel(args.generated_cpp)},
      "checks": checks,
      "required_checks_passed": required,
      "kill_number": {
          "linear_layer_count": len(LINEAR_LAYERS),
          "baseline_tok_s": baseline_tok_s,
          "floor_tok_s": floor_tok_s,
          "baseline_token_us": baseline_token_us,
          "floor_token_us": floor_token_us,
          "floor_headroom_us": floor_headroom_us,
          "cpuorder_projection_min_us": cpuorder_us,
          "rowblock16_projection_min_us": rowblock16_us,
          "per_layer_projection_extra_us": per_layer_extra_us,
          "all_linear_projection_extra_lower_bound_us": projection_extra_us,
          "lower_bound_to_headroom_ratio": lower_bound_ratio,
          "serial_reduction_and_bridge_cost_included": False,
      },
      "all_linear_token_allowed": False,
      "subset_sweep_allowed": False,
      "speedup_claims_allowed": False,
      "promotion_allowed": False,
      "disposition": (
          "reject_all_linear_parity_by_floor_kill_number_select_non_arithmetic_route_control"
          if required else "block_all_linear_parity_feasibility_incomplete"),
      "rejected_route": REJECTED_ROUTE if required else None,
      "selected_next_route": SELECTED_NEXT_ROUTE if required else CURRENT_ROUTE,
      "next_route_reason": (
          "CPU-order output projection alone exceeds the entire floor headroom "
          "before serial-reduction or host-bridge cost. The all-linear parity "
          "bundle is therefore unpromotable in its current implementation and "
          "does not earn a token row. Switch to a non-arithmetic product-source "
          "route-control gate."
          if required else "Feasibility evidence is incomplete; do not run a token."),
  }


def write_outputs(metrics: dict[str, Any], out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  failed = [row["name"] for row in metrics["checks"] if not row["pass"]]
  kill = metrics["kill_number"]
  lines = [
      f"# Seq{metrics['sequence']} All-Linear Parity Feasibility Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- rejected_route: `{metrics['rejected_route']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- floor headroom us/token: `{kill['floor_headroom_us']}`",
      f"- projection-only lower bound us/token: `{kill['all_linear_projection_extra_lower_bound_us']}`",
      f"- lower-bound/headroom ratio: `{kill['lower_bound_to_headroom_ratio']}`",
      f"- failed_checks: `{failed}`",
      "",
      metrics["next_route_reason"],
      "",
      "This is source/kill-number evidence only. It is not a speed claim.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--sequence", type=int, default=563)
  parser.add_argument("--predecessor", type=Path,
      default=ROOT / "output/seq562-layer0-norm-to-projection-parity-bundle-router-math-gate-20260710Tseq562Z/metrics.json")
  parser.add_argument("--routes", type=Path, default=ACTIVE / "routes-ledger.json")
  parser.add_argument("--component", type=Path,
      default=ROOT / "output/seq555-layer0-linear-output-projection-cpuorder-component-target-gate-20260710Tseq555Z/metrics.json")
  parser.add_argument("--generated-cpp", type=Path,
      default=ROOT / "output/seq556-layer0-linear-output-projection-cpuorder-source-20260710Tseq556Z/r2_gpu_decode_smoke.cpp")
  parser.add_argument("--out-dir", type=Path,
      default=ROOT / "output/seq563-all-linear-norm-to-projection-parity-feasibility-gate-20260710Tseq563Z")
  args = parser.parse_args()
  metrics = compute(args)
  write_outputs(metrics, args.out_dir)
  print(json.dumps({"required_checks_passed": metrics["required_checks_passed"],
                    "disposition": metrics["disposition"],
                    "all_linear_token_allowed": metrics["all_linear_token_allowed"],
                    "selected_next_route": metrics["selected_next_route"],
                    "out_dir": _rel(args.out_dir)}, sort_keys=True))
  return 0 if metrics["required_checks_passed"] else 2


if __name__ == "__main__":
  raise SystemExit(main())
