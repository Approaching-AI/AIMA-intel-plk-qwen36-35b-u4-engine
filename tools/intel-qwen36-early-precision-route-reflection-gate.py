#!/usr/bin/env python3
"""Reflect after the early-layer precision-island routes are exhausted."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = "intel-qwen36-early-precision-route-reflection-gate-v0"
CURRENT_ROUTE = (
    "router_prompt_distribution_early_precision_island_route_reflection_gate"
)
SELECTED_NEXT_ROUTE = (
    "router_prompt_distribution_gpu_cpu_arithmetic_parity_coverage_audit_gate"
)
KLD_THRESHOLD = 0.005
REQUIRED_CLOSED_ROUTES = {
    "qkv_delta_blockq16_no_product_value_source",
    "all_linear_state_direct_product_refresh",
    "linear_conv_history_known_product_source_audit",
    "router_prompt_distribution_fp64_precision_pack",
    "router_prompt_distribution_layer0_1_gpu_final_norm_precision_island",
    "router_prompt_distribution_layer0_1_input_rmsnorm_cpu_sqrt_precision_island",
    "router_prompt_distribution_input_rmsnorm_serial_layer_subsets",
    "gpu_attention_front_handoff_8tok_opencl_q6_lane_sums_nofma_diagnostic",
    "gpu_attention_front_handoff_8tok_linear_delta_cpu_shape_diagnostic",
    "gpu_attention_front_handoff_8tok_q4_cpu_order_linear_ab_diagnostic",
    "gpu_attention_front_handoff_8tok_cpu_linear_postconv_prep_diagnostic",
}


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


def _dist_summary(payload: dict[str, Any]) -> dict[str, Any]:
  smoke = _smoke(payload)
  dist = smoke.get("distribution_ladder")
  dist = dist if isinstance(dist, dict) else {}
  return {
      "case_id": smoke.get("case_id"),
      "required_checks_passed": dist.get("required_checks_passed"),
      "position_count": dist.get("position_count"),
      "max_kld": dist.get("max_kld"),
      "top1_rate": dist.get("top1_rate"),
      "top1_pass": dist.get("top1_pass"),
      "rowblock16_enabled": smoke.get(
          "attention_front_output_projection_rowblock16_enabled"),
  }


def _rejected_names(payload: dict[str, Any]) -> set[str]:
  return {
      str(row["route"])
      for row in payload.get("rejected", [])
      if isinstance(row, dict) and isinstance(row.get("route"), str)
  }


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


def _accepted_cut(payload: dict[str, Any], cut_id: str) -> dict[str, Any] | None:
  for row in payload.get("accepted", payload.get("cuts", [])):
    if isinstance(row, dict) and row.get("id") == cut_id:
      return row
  return None


def _acceptance_keeps_kld(payload: dict[str, Any]) -> bool:
  accuracy = payload.get("accuracy")
  accuracy = accuracy if isinstance(accuracy, dict) else {}
  dist = accuracy.get("teacher_forced_distribution")
  dist = dist if isinstance(dist, dict) else {}
  return abs(_num(dist.get("kl_divergence_max")) - KLD_THRESHOLD) < 1e-12


def compute(args: argparse.Namespace) -> dict[str, Any]:
  predecessor = _load(args.predecessor)
  routes = _load(args.routes)
  rejected = _load(args.rejected)
  accepted = _load(args.accepted)
  acceptance = _load(args.acceptance)
  short_baseline = _dist_summary(_load(args.short_baseline))
  no_rowblock = _dist_summary(_load(args.no_rowblock_router_math))
  seq541 = _load(args.seq541)
  rejected_names = _rejected_names(rejected)
  missing_closed = sorted(REQUIRED_CLOSED_ROUTES - rejected_names)
  accepted_frontier = _accepted_cut(
      accepted, "selected_shared_q6_down_combined_per_expert_cold_cache")
  seq541_math = seq541.get("rows", {}).get("math", {})
  checks = [
      {
          "name": "seq551_selected_route_reflection",
          "pass": (
              predecessor.get("required_checks_passed") is True
              and predecessor.get("all_serial_layer_subsets_rejected") is True
              and predecessor.get("selected_next_route") == CURRENT_ROUTE
              and _has_candidate(routes, 551, CURRENT_ROUTE)
              and _has_switch(
                  routes, 551,
                  "select_router_prompt_distribution_early_precision_island_"
                  "route_reflection_gate")),
      },
      {
          "name": "acceptance_matrix_keeps_kld_ruler",
          "pass": _acceptance_keeps_kld(acceptance),
      },
      {
          "name": "accepted_19_330_cut_exists",
          "pass": (
              isinstance(accepted_frontier, dict)
              and "19.330" in str(accepted_frontier.get("note"))
              and "0.002138777553" in str(accepted_frontier.get("note"))),
      },
      {
          "name": "accepted_19_330_distribution_is_short_only",
          "pass": (
              short_baseline["case_id"] == "short_math_001"
              and short_baseline["required_checks_passed"] is True
              and _num(short_baseline["max_kld"]) <= KLD_THRESHOLD
              and _num(short_baseline["top1_rate"]) == 1.0),
      },
      {
          "name": "no_rowblock_router_control_is_not_fallback",
          "pass": (
              no_rowblock["case_id"] == "router_math_reason_001"
              and no_rowblock["rowblock16_enabled"] is False
              and no_rowblock["required_checks_passed"] is False
              and _num(no_rowblock["max_kld"])
              > _num(seq541_math.get("max_kld"))
              and _num(no_rowblock["top1_rate"]) == 1.0),
      },
      {
          "name": "exact_input_layer0_source_is_sub_ulp_not_projection_bridge",
          "pass": (
              seq541.get("required_checks_passed") is True
              and seq541.get("disposition")
              == "reject_layer0_projection_q8_bridge_select_layer0_1_precision_island_feasibility"
              and _num(seq541_math.get("max_kld")) > KLD_THRESHOLD),
      },
      {
          "name": "known_source_and_precision_routes_are_closed",
          "pass": not missing_closed,
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
          "accepted": _rel(args.accepted),
          "acceptance": _rel(args.acceptance),
          "short_baseline": _rel(args.short_baseline),
          "no_rowblock_router_math": _rel(args.no_rowblock_router_math),
          "seq541": _rel(args.seq541),
      },
      "checks": checks,
      "required_checks_passed": required,
      "missing_closed_routes": missing_closed,
      "short_baseline": short_baseline,
      "no_rowblock_router_math": no_rowblock,
      "new_target_row_allowed": False,
      "speedup_claims_allowed": False,
      "promotion_allowed": False,
      "arithmetic_parity_coverage_audit_allowed": required,
      "disposition": (
          "accept_route_reflection_select_arithmetic_parity_coverage_audit"
          if required else
          "block_route_reflection_inconsistent_evidence"),
      "selected_next_route": SELECTED_NEXT_ROUTE if required else CURRENT_ROUTE,
      "next_route_reason": (
          "The accepted 19.330 tok/s row has only short_math distribution "
          "evidence, so it is not a router-valid fallback; the explicit "
          "no-rowblock router-math control is worse. Product value-source, "
          "global precision, local final/input norm, Q6 lane-order, recurrent "
          "CPU-shape, alpha/beta CPU-order, and postconv CPU-prep routes are "
          "closed. Audit CPU/GPU arithmetic parity coverage from the exact-input "
          "layer0 boundary and the all-linear qkv-history signal before selecting "
          "any new implementation or target row."
          if required else
          "Route-reflection evidence is incomplete; do not select a target row."
      ),
  }


def write_outputs(metrics: dict[str, Any], out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  failed = [row["name"] for row in metrics["checks"] if not row["pass"]]
  lines = [
      f"# Seq{metrics['sequence']} Early Precision Route Reflection Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- short baseline case/KLD: `{metrics['short_baseline']['case_id']}` / "
      f"`{metrics['short_baseline']['max_kld']}`",
      f"- no-rowblock router-math KLD: "
      f"`{metrics['no_rowblock_router_math']['max_kld']}`",
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
  parser.add_argument("--sequence", type=int, default=552)
  parser.add_argument(
      "--predecessor", type=Path,
      default=ROOT / "output/seq551-input-rmsnorm-serial-layer-subset-attribution-gate-20260710Tseq551Z/metrics.json")
  parser.add_argument("--routes", type=Path,
                      default=ACTIVE / "routes-ledger.json")
  parser.add_argument("--rejected", type=Path,
                      default=ACTIVE / "rejected-routes.json")
  parser.add_argument("--accepted", type=Path,
                      default=ACTIVE / "accepted-cuts.json")
  parser.add_argument(
      "--acceptance", type=Path,
      default=ROOT / "benchmarks" / WORKSTREAM / "acceptance-matrix.json")
  parser.add_argument(
      "--short-baseline", type=Path,
      default=ROOT / "output/r2-gpu-selected-shared-q4q6-down-cold-q6-experts-noqueue-distribution-20260705T143408Z/result.json")
  parser.add_argument(
      "--no-rowblock-router-math", type=Path,
      default=ROOT / "output/r2-gpu-router-math-distribution-no-rowblock16-20260708Tseq223Z/result.json")
  parser.add_argument(
      "--seq541", type=Path,
      default=ROOT / "output/seq541-layer0-exact-delta-source-gate-20260710Tseq541Z/metrics.json")
  parser.add_argument(
      "--out-dir", type=Path,
      default=ROOT / "output/seq552-early-precision-island-route-reflection-gate-20260710Tseq552Z")
  args = parser.parse_args()
  metrics = compute(args)
  write_outputs(metrics, args.out_dir)
  print(json.dumps({
      "required_checks_passed": metrics["required_checks_passed"],
      "disposition": metrics["disposition"],
      "arithmetic_parity_coverage_audit_allowed": metrics[
          "arithmetic_parity_coverage_audit_allowed"],
      "selected_next_route": metrics["selected_next_route"],
      "out_dir": _rel(args.out_dir),
  }, sort_keys=True))
  return 0 if metrics["required_checks_passed"] else 2


if __name__ == "__main__":
  raise SystemExit(main())
