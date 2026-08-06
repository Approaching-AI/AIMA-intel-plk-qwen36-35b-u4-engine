#!/usr/bin/env python3
"""Audit early-layer CPU/GPU arithmetic parity coverage without target work."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = "intel-qwen36-gpu-cpu-arithmetic-parity-coverage-audit-v0"
CURRENT_ROUTE = (
    "router_prompt_distribution_gpu_cpu_arithmetic_parity_coverage_audit_gate"
)
SELECTED_NEXT_ROUTE = (
    "router_prompt_distribution_layer0_linear_output_projection_reduction_"
    "order_component_source_gate"
)
REQUIRED_COVERED_BOUNDARIES = {
    "input_rmsnorm": {
        "router_prompt_distribution_input_rmsnorm_serial_layer_subsets",
    },
    "q6_qkv_reduction": {
        "gpu_attention_front_handoff_8tok_opencl_q6_lane_sums_nofma_diagnostic",
    },
    "q4_alpha_beta_reduction": {
        "gpu_attention_front_handoff_8tok_q4_cpu_order_linear_ab_diagnostic",
    },
    "postconv_silu_l2": {
        "gpu_attention_front_handoff_8tok_cpu_linear_postconv_prep_diagnostic",
        "gpu_attention_front_handoff_8tok_linear_l2_double_sum_diagnostic",
    },
    "linear_recurrent_update": {
        "gpu_attention_front_handoff_8tok_linear_delta_cpu_shape_diagnostic",
    },
    "linear_final_norm_gate": {
        "router_prompt_distribution_layer0_1_gpu_final_norm_precision_island",
        "gpu_attention_front_handoff_8tok_opencl_double_sigmoid_diagnostic",
    },
    "residual_and_ffn_tail": {
        "gpu_attention_front_handoff_8tok_cpu_residual_add_diagnostic",
    },
}
LINEAR_CPU_ORDER_ROUTE = (
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


def _layer0_projection_metrics(seq541: dict[str, Any]) -> dict[str, Any]:
  rows = seq541.get("rows", {}).get("math", {}).get("entries", [])
  projection_diffs: list[float] = []
  q8_qs_mismatches: list[int] = []
  q8_bsums_mismatches: list[int] = []
  for entry in rows if isinstance(rows, list) else []:
    layers = entry.get("layers") if isinstance(entry, dict) else []
    for layer in layers if isinstance(layers, list) else []:
      if not isinstance(layer, dict) or layer.get("layer") != 0:
        continue
      value = layer.get("projection_gpu_vs_cpu_max_abs_diff")
      if isinstance(value, (int, float)):
        projection_diffs.append(float(value))
      qs = layer.get("projection_q8_qs_mismatch_count")
      bsums = layer.get("projection_q8_bsums_mismatch_count")
      if isinstance(qs, int):
        q8_qs_mismatches.append(qs)
      if isinstance(bsums, int):
        q8_bsums_mismatches.append(bsums)
  return {
      "observed_rows": len(projection_diffs),
      "max_gpu_vs_cpu_abs_diff": max(projection_diffs, default=0.0),
      "all_gpu_vs_cpu_exact": bool(projection_diffs)
      and all(value == 0.0 for value in projection_diffs),
      "q8_qs_mismatch_sum": sum(q8_qs_mismatches),
      "q8_bsums_mismatch_sum": sum(q8_bsums_mismatches),
  }


def compute(args: argparse.Namespace) -> dict[str, Any]:
  predecessor = _load(args.predecessor)
  routes = _load(args.routes)
  rejected = _load(args.rejected)
  seq541 = _load(args.seq541)
  decode_source = args.decode_source.read_text(encoding="utf-8")
  header_source = args.header_source.read_text(encoding="utf-8")
  cpuorder_source = args.cpuorder_source.read_text(encoding="utf-8")
  probe_source = args.output_probe.read_text(encoding="utf-8")
  rejected_names = _rejected_names(rejected)
  coverage = {
      boundary: sorted(required)
      for boundary, required in REQUIRED_COVERED_BOUNDARIES.items()
  }
  missing_coverage = {
      boundary: sorted(required - rejected_names)
      for boundary, required in REQUIRED_COVERED_BOUNDARIES.items()
      if required - rejected_names
  }
  projection = _layer0_projection_metrics(seq541)
  current_path_markers = all(marker in decode_source for marker in (
      "Require(output_tensor.type == 12",
      "RunResidentPackedQ4X8ThenResidentResidualRmsNorm",
      "iq36::GpuQ4X8KernelVariant::kRowlaneParallel",
      "LayerTensorName(layer, \"ssm_out.weight\")",
  ))
  cpuorder_api_markers = all(marker in source for marker, source in (
      ("class GpuQ4KCpuOrderMatvecRunner", header_source),
      ("RunResidentRawQ4KCpuOrder", header_source),
      ("RunResidentRawQ4KCpuOrder", cpuorder_source),
      ("UploadRawQ4KCpuOrder", cpuorder_source),
  ))
  component_probe_markers = all(marker in probe_source for marker in (
      "ssm_out.weight",
      "final_output.bin",
      "linear_attn_out.bin",
      "GpuQ4X8KernelVariant::kRowlaneParallel",
  ))
  checks = [
      {
          "name": "seq552_selected_parity_coverage_audit",
          "pass": (
              predecessor.get("required_checks_passed") is True
              and predecessor.get("arithmetic_parity_coverage_audit_allowed")
              is True
              and predecessor.get("selected_next_route") == CURRENT_ROUTE
              and _has_candidate(routes, 552, CURRENT_ROUTE)
              and _has_switch(
                  routes, 552,
                  "select_router_prompt_distribution_gpu_cpu_arithmetic_"
                  "parity_coverage_audit_gate")),
      },
      {
          "name": "known_early_arithmetic_boundaries_have_closed_evidence",
          "pass": not missing_coverage,
      },
      {
          "name": "layer0_projection_q8_bridge_is_exact",
          "pass": (
              projection["observed_rows"] > 0
              and projection["q8_qs_mismatch_sum"] == 0
              and projection["q8_bsums_mismatch_sum"] == 0),
      },
      {
          "name": "layer0_packed_projection_reduction_is_not_exact",
          "pass": (
              projection["max_gpu_vs_cpu_abs_diff"] > 0.0
              and projection["all_gpu_vs_cpu_exact"] is False),
      },
      {
          "name": "current_linear_projection_uses_rowlane_q4",
          "pass": current_path_markers,
      },
      {
          "name": "generic_resident_cpu_order_q4_runner_exists",
          "pass": cpuorder_api_markers,
      },
      {
          "name": "captured_output_projection_component_probe_exists",
          "pass": component_probe_markers,
      },
      {
          "name": "linear_output_projection_cpu_order_route_is_unclosed",
          "pass": LINEAR_CPU_ORDER_ROUTE not in rejected_names,
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
          "seq541": _rel(args.seq541),
          "decode_source": _rel(args.decode_source),
          "header_source": _rel(args.header_source),
          "cpuorder_source": _rel(args.cpuorder_source),
          "output_probe": _rel(args.output_probe),
      },
      "checks": checks,
      "required_checks_passed": required,
      "covered_boundaries": coverage,
      "missing_coverage": missing_coverage,
      "layer0_projection": projection,
      "first_uncovered_boundary": (
          "linear_attention_output_projection_q4_reduction_order"
          if required else None),
      "component_source_gate_allowed": required,
      "new_target_row_allowed": False,
      "speedup_claims_allowed": False,
      "promotion_allowed": False,
      "disposition": (
          "select_layer0_linear_output_projection_reduction_component_source"
          if required else
          "block_arithmetic_parity_coverage_incomplete"),
      "selected_next_route": SELECTED_NEXT_ROUTE if required else CURRENT_ROUTE,
      "next_route_reason": (
          "The exact-input layer0 trace has an exact output-projection Q8 "
          "bridge but a nonzero packed-Q4 projection delta. Input norm, Q6 "
          "QKV, alpha/beta, postconv, recurrent update, final norm/gate, and "
          "residual/FFN-tail parity classes all have closed evidence. The "
          "linear `ssm_out.weight` rowlane reduction order is the first "
          "uncovered boundary. Extend the existing captured-payload component "
          "probe with the generic resident CPU-order Q4 runner in source-only "
          "form before any target compile or decode row."
          if required else
          "Parity coverage is incomplete; do not select an implementation row."
      ),
  }


def write_outputs(metrics: dict[str, Any], out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  failed = [row["name"] for row in metrics["checks"] if not row["pass"]]
  projection = metrics["layer0_projection"]
  lines = [
      f"# Seq{metrics['sequence']} GPU/CPU Arithmetic-Parity Coverage Audit",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- first_uncovered_boundary: `{metrics['first_uncovered_boundary']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- layer0 projection max GPU/CPU abs diff: "
      f"`{projection['max_gpu_vs_cpu_abs_diff']}`",
      f"- layer0 projection Q8 qs/bsums mismatches: "
      f"`{projection['q8_qs_mismatch_sum']}` / "
      f"`{projection['q8_bsums_mismatch_sum']}`",
      f"- failed_checks: `{failed}`",
      "",
      metrics["next_route_reason"],
      "",
      "This is source/evidence-only route selection. It is not a speed claim.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--sequence", type=int, default=553)
  parser.add_argument(
      "--predecessor", type=Path,
      default=ROOT / "output/seq552-early-precision-island-route-reflection-gate-20260710Tseq552Z/metrics.json")
  parser.add_argument("--routes", type=Path,
                      default=ACTIVE / "routes-ledger.json")
  parser.add_argument("--rejected", type=Path,
                      default=ACTIVE / "rejected-routes.json")
  parser.add_argument(
      "--seq541", type=Path,
      default=ROOT / "output/seq541-layer0-exact-delta-source-gate-20260710Tseq541Z/metrics.json")
  parser.add_argument(
      "--decode-source", type=Path,
      default=ROOT / "output/seq541-layer0-exact-delta-source-math-20260710Tseq541Z/r2_gpu_decode_smoke.cpp")
  parser.add_argument(
      "--header-source", type=Path,
      default=ROOT / "engine/include/intel_qwen36/gpu_q4x8_matvec.hpp")
  parser.add_argument(
      "--cpuorder-source", type=Path,
      default=ROOT / "engine/src/gpu_q4_cpu_order_matvec.cpp")
  parser.add_argument(
      "--output-probe", type=Path,
      default=ROOT / "tools/intel-qwen36-gpu-q4x8-output-projection-probe.py")
  parser.add_argument(
      "--out-dir", type=Path,
      default=ROOT / "output/seq553-gpu-cpu-arithmetic-parity-coverage-audit-gate-20260710Tseq553Z")
  args = parser.parse_args()
  metrics = compute(args)
  write_outputs(metrics, args.out_dir)
  print(json.dumps({
      "required_checks_passed": metrics["required_checks_passed"],
      "disposition": metrics["disposition"],
      "first_uncovered_boundary": metrics["first_uncovered_boundary"],
      "selected_next_route": metrics["selected_next_route"],
      "out_dir": _rel(args.out_dir),
  }, sort_keys=True))
  return 0 if metrics["required_checks_passed"] else 2


if __name__ == "__main__":
  raise SystemExit(main())
