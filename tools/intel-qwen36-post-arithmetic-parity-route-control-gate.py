#!/usr/bin/env python3
"""Select the only bounded post-parity bundle allowed by reopen conditions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = "intel-qwen36-post-arithmetic-parity-route-control-v0"
CURRENT_ROUTE = (
    "router_prompt_distribution_post_arithmetic_parity_route_control_gate"
)
SELECTED_NEXT_ROUTE = (
    "router_prompt_distribution_layer0_norm_to_projection_parity_bundle_"
    "one_token_gate"
)
SERIAL_ROUTE = "router_prompt_distribution_input_rmsnorm_serial_layer_subsets"
PROJECTION_ROUTE = (
    "gpu_linear_attention_output_projection_cpu_order_reduction_diagnostic"
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


def _num(value: Any, default: float = 0.0) -> float:
  return float(value) if isinstance(value, (int, float)) else default


def _smoke(payload: dict[str, Any]) -> dict[str, Any]:
  smoke = payload.get("smoke")
  return smoke if isinstance(smoke, dict) else payload


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


def _rejected(payload: dict[str, Any], route: str) -> dict[str, Any]:
  for row in payload.get("rejected", []):
    if isinstance(row, dict) and row.get("route") == route:
      return row
  return {}


def _layer0_preconv(smoke: dict[str, Any]) -> dict[str, Any]:
  steps = smoke.get("linear_preconv_source_diff_by_step")
  if not isinstance(steps, list) or not steps:
    return {}
  layers = steps[0].get("layers") if isinstance(steps[0], dict) else []
  for row in layers if isinstance(layers, list) else []:
    if isinstance(row, dict) and row.get("layer") == 0:
      return row
  return {}


def compute(args: argparse.Namespace) -> dict[str, Any]:
  predecessor = _load(args.predecessor)
  routes = _load(args.routes)
  rejected = _load(args.rejected)
  serial = _smoke(_load(args.serial_layer0))
  projection = _smoke(_load(args.projection_layer0))
  compile_gate = _load(args.compile_gate)
  generated = args.generated_cpp.read_text(encoding="utf-8")
  serial_rejection = _rejected(rejected, SERIAL_ROUTE)
  projection_rejection = _rejected(rejected, PROJECTION_ROUTE)
  serial_dist = serial.get("distribution_ladder")
  serial_dist = serial_dist if isinstance(serial_dist, dict) else {}
  projection_dist = projection.get("distribution_ladder")
  projection_dist = projection_dist if isinstance(projection_dist, dict) else {}
  serial_layer0 = _layer0_preconv(serial)
  checks = [
      {
          "name": "seq559_selected_post_parity_route_control",
          "pass": (
              predecessor.get("required_checks_passed") is True
              and predecessor.get("router_code_distribution_allowed") is False
              and predecessor.get("selected_next_route") == CURRENT_ROUTE
              and _has_candidate(routes, 559, CURRENT_ROUTE)
              and _has_switch(
                  routes, 559,
                  "select_router_prompt_distribution_post_arithmetic_parity_"
                  "route_control_gate")),
      },
      {
          "name": "serial_reopen_condition_accepts_new_upstream_state",
          "pass": (
              "new upstream-state evidence" in str(
                  serial_rejection.get("reopen_condition"))
              and "changes" in str(serial_rejection.get("reopen_condition"))),
      },
      {
          "name": "projection_reopen_condition_requires_bounded_parity_bundle",
          "pass": (
              "bounded bundle" in str(
                  projection_rejection.get("reopen_condition"))
              and "independently proven layer0 parity boundary" in str(
                  projection_rejection.get("reopen_condition"))),
      },
      {
          "name": "layer0_serial_norm_boundary_is_independently_exact",
          "pass": (
              serial.get("linear_output_projection_cpu_order_layers") in
              (None, [])
              and serial.get("input_rmsnorm_serial_reduction_layers") == [0]
              and _num(serial_dist.get("steps", [{}])[0].get("kld")) < 0.005
              and _num(serial_layer0.get("gpu_attn_norm_vs_cpu_max_abs_diff"),
                       -1.0) == 0.0
              and _num(serial_layer0.get("qkv_from_gpu_attn_norm_max_abs_diff"),
                       -1.0) == 0.0
              and _num(serial_layer0.get("z_from_gpu_attn_norm_max_abs_diff"),
                       -1.0) == 0.0),
      },
      {
          "name": "layer0_projection_boundary_is_independently_exact",
          "pass": (
              projection.get("linear_output_projection_cpu_order_layers")
              == [0]
              and projection.get("input_rmsnorm_serial_reduction_layers") == []
              and _num(projection_dist.get("steps", [{}])[0].get("kld"))
              < 0.005
              and predecessor.get("projection_rows")
              and all(_num(row.get("gpu_vs_cpu_max_abs"), -1.0) == 0.0
                      for row in predecessor.get("projection_rows", []))),
      },
      {
          "name": "same_target_binary_contains_both_default_off_selectors",
          "pass": (
              compile_gate.get("required_checks_passed") is True
              and compile_gate.get("compile_summary", {}).get("ok") is True
              and "g_decode_input_rmsnorm_serial_reduction_layers" in generated
              and "g_decode_linear_output_projection_cpu_order_layers"
              in generated),
      },
  ]
  required = all(bool(row["pass"]) for row in checks)
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "sequence": args.sequence,
      "inputs": {
          "predecessor": _rel(args.predecessor),
          "routes": _rel(args.routes),
          "rejected": _rel(args.rejected),
          "serial_layer0": _rel(args.serial_layer0),
          "projection_layer0": _rel(args.projection_layer0),
          "compile_gate": _rel(args.compile_gate),
          "generated_cpp": _rel(args.generated_cpp),
      },
      "checks": checks,
      "required_checks_passed": required,
      "bundle": {
          "input_rmsnorm_serial_reduction_layers": [0],
          "linear_output_projection_cpu_order_layers": [0],
          "new_source_required": False,
          "new_target_compile_required": False,
      },
      "one_token_bundle_allowed": required,
      "distribution_allowed": False,
      "speedup_claims_allowed": False,
      "promotion_allowed": False,
      "disposition": (
          "select_layer0_norm_to_projection_parity_bundle_one_token"
          if required else
          "close_arithmetic_parity_no_admissible_bundle"),
      "selected_next_route": SELECTED_NEXT_ROUTE if required else CURRENT_ROUTE,
      "next_route_reason": (
          "The new exact projection state satisfies both recorded reopen "
          "conditions when bundled with the independently exact layer0 serial "
          "input-norm boundary. Both selectors are already default-off in the "
          "same compiled binary. Run exactly one combined router-math token; "
          "this is the last admissible arithmetic-parity bundle before closing "
          "the board."
          if required else
          "No bundle satisfies the recorded reopen conditions; close parity work."
      ),
  }


def write_outputs(metrics: dict[str, Any], out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  failed = [row["name"] for row in metrics["checks"] if not row["pass"]]
  lines = [
      f"# Seq{metrics['sequence']} Post-Arithmetic-Parity Route Control",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- bundle: `{metrics['bundle']}`",
      f"- failed_checks: `{failed}`",
      "",
      metrics["next_route_reason"],
      "",
      "This is route-control evidence only. It is not a speed claim.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--sequence", type=int, default=560)
  parser.add_argument(
      "--predecessor", type=Path,
      default=ROOT / "output/seq559-layer0-linear-output-projection-cpuorder-router-math-gate-20260710Tseq559Z/metrics.json")
  parser.add_argument("--routes", type=Path,
                      default=ACTIVE / "routes-ledger.json")
  parser.add_argument("--rejected", type=Path,
                      default=ACTIVE / "rejected-routes.json")
  parser.add_argument(
      "--serial-layer0", type=Path,
      default=ROOT / "output/seq551-input-rmsnorm-serial-layer0-router-math-20260710Tseq551Z/result.json")
  parser.add_argument(
      "--projection-layer0", type=Path,
      default=ROOT / "output/seq559-layer0-linear-output-projection-cpuorder-router-math-20260710Tseq559Z/result.json")
  parser.add_argument(
      "--compile-gate", type=Path,
      default=ROOT / "output/seq557-layer0-linear-output-projection-cpuorder-target-compile-gate-20260710Tseq557Z/metrics.json")
  parser.add_argument(
      "--generated-cpp", type=Path,
      default=ROOT / "output/seq556-layer0-linear-output-projection-cpuorder-source-20260710Tseq556Z/r2_gpu_decode_smoke.cpp")
  parser.add_argument(
      "--out-dir", type=Path,
      default=ROOT / "output/seq560-post-arithmetic-parity-route-control-gate-20260710Tseq560Z")
  args = parser.parse_args()
  metrics = compute(args)
  write_outputs(metrics, args.out_dir)
  print(json.dumps({
      "required_checks_passed": metrics["required_checks_passed"],
      "disposition": metrics["disposition"],
      "one_token_bundle_allowed": metrics["one_token_bundle_allowed"],
      "selected_next_route": metrics["selected_next_route"],
      "out_dir": _rel(args.out_dir),
  }, sort_keys=True))
  return 0 if metrics["required_checks_passed"] else 2


if __name__ == "__main__":
  raise SystemExit(main())
