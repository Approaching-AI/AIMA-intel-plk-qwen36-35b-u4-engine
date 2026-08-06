#!/usr/bin/env python3
"""Classify a layer-scoped input-RMSNorm reduction precision island."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = "intel-qwen36-input-rmsnorm-reduction-feasibility-gate-v0"
CURRENT_ROUTE = (
    "router_prompt_distribution_layer0_1_input_rmsnorm_reduction_"
    "precision_island_feasibility_gate"
)
SELECTED_NEXT_ROUTE = (
    "router_prompt_distribution_layer0_1_input_rmsnorm_serial_reduction_"
    "precision_island_source_gate"
)
REJECTED_ROUTE = (
    "router_prompt_distribution_layer0_1_input_rmsnorm_cpu_sqrt_precision_island"
)
GLOBAL_SERIAL_ROUTE = "shared_rmsnorm_scale_kernel_serial_cpu_sqrt_product_fix"


def _load(path: Path) -> dict[str, Any]:
  payload = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(payload, dict):
    raise TypeError(f"{path} does not contain a JSON object")
  return payload


def _read(path: Path) -> str:
  return path.read_text(encoding="utf-8")


def _rel(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path)


def _smoke(payload: dict[str, Any]) -> dict[str, Any]:
  smoke = payload.get("smoke")
  return smoke if isinstance(smoke, dict) else payload


def _layer(payload: dict[str, Any], layer: int) -> dict[str, Any]:
  steps = _smoke(payload).get("linear_preconv_source_diff_by_step")
  if not isinstance(steps, list) or not steps or not isinstance(steps[0], dict):
    return {}
  rows = steps[0].get("layers")
  rows = rows if isinstance(rows, list) else []
  return next((
      row for row in rows
      if isinstance(row, dict) and row.get("layer") == layer
  ), {})


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


def _rejected(rejected: dict[str, Any], route: str) -> bool:
  return any(
      isinstance(row, dict) and row.get("route") == route
      for row in rejected.get("rejected", []))


def _body(source: str, name: str) -> str:
  start = source.find(name)
  if start < 0:
    return ""
  brace = source.find("{", start)
  if brace < 0:
    return ""
  depth = 0
  for index in range(brace, len(source)):
    if source[index] == "{":
      depth += 1
    elif source[index] == "}":
      depth -= 1
      if depth == 0:
        return source[brace:index + 1]
  return ""


def _source_shape(args: argparse.Namespace) -> dict[str, Any]:
  kernel = _read(args.kernel_source)
  engine = _read(args.engine_source)
  decode = _read(args.decode_source)
  serial_artifact = _read(args.serial_generated_source)
  current_scale = _body(kernel, "__kernel void rms_norm_hidden_scale_f32")
  standalone = _body(kernel, "__kernel void rms_norm_hidden_f32")
  proven_serial = _body(
      serial_artifact, "__kernel void rms_norm_hidden_scale_f32")
  return {
      "current_shared_scale_is_parallel": all(marker in current_scale for marker in [
          "__local float partial[256];",
          "const uint chunk = (hidden_size + local_size - 1U) / local_size;",
          "total += partial[i];",
      ]),
      "standalone_serial_sum_exists": all(marker in standalone for marker in [
          "if ((uint)get_global_id(0) != 0U)",
          "for (uint i = 0; i < hidden_size; ++i)",
          "sum_squares += value * value;",
      ]),
      "proven_serial_shared_scale_shape": all(marker in proven_serial for marker in [
          "if ((uint)get_global_id(0) != 0U)",
          "for (uint i = 0; i < hidden_size; ++i)",
          "scale_out[0] = rsqrt(mean_square + epsilon);",
      ]) and "__local float partial[256];" not in proven_serial,
      "shared_runner_scale_apply_split": all(marker in engine for marker in [
          "kernel_rmsnorm_hidden_scale_",
          "kernel_rmsnorm_hidden_apply_scale_",
          "RunRmsNormHiddenResidentInputResidentWeight(",
      ]),
      "decode_has_layer_set_primitives": all(marker in decode for marker in [
          "DecodeParseLayerList(",
          "DecodeLayerListed(",
          "RunGpuLayerInputRmsNorm(",
      ]),
      "implementation_shape": {
          "selector": "one environment-provided layer set",
          "kernel_change": "uniform serial-reduction mode on the shared scale kernel",
          "apply_kernel": "unchanged",
          "new_kernel_families": 0,
          "per_layer_source_files": 0,
          "target_rows_before_source_gate": 0,
      },
  }


def _row_summary(row: dict[str, Any]) -> dict[str, Any]:
  return {
      "gpu_attn_norm_vs_cpu_max_abs_diff": row.get(
          "gpu_attn_norm_vs_cpu_max_abs_diff"),
      "qkv_from_gpu_attn_norm_max_abs_diff": row.get(
          "qkv_from_gpu_attn_norm_max_abs_diff"),
      "z_from_gpu_attn_norm_max_abs_diff": row.get(
          "z_from_gpu_attn_norm_max_abs_diff"),
  }


def compute(args: argparse.Namespace) -> dict[str, Any]:
  predecessor = _load(args.predecessor)
  routes = _load(args.routes)
  rejected = _load(args.rejected)
  cpu_sqrt = _load(args.cpu_sqrt_result)
  serial = _load(args.serial_result)
  serial_cpu_sqrt = _load(args.serial_cpu_sqrt_result)
  cpu0 = _layer(cpu_sqrt, 0)
  serial0 = _layer(serial, 0)
  serial1 = _layer(serial, 1)
  serial_sqrt0 = _layer(serial_cpu_sqrt, 0)
  source = _source_shape(args)
  predecessor_selects = (
      predecessor.get("required_checks_passed") is True
      and predecessor.get("selected_next_route") == CURRENT_ROUTE
      and _has_candidate(routes, 545, CURRENT_ROUTE)
      and _has_switch(
          routes, 545,
          "select_router_prompt_distribution_layer0_1_input_rmsnorm_"
          "reduction_precision_island_feasibility_gate"))
  cpu_sqrt_does_not_close = (
      _num(cpu0.get("gpu_attn_norm_vs_cpu_max_abs_diff"))
      == 3.814697266e-6)
  serial_closes_layer0 = all(
      _num(serial0.get(key)) == 0.0 for key in [
          "gpu_attn_norm_vs_cpu_max_abs_diff",
          "qkv_from_gpu_attn_norm_max_abs_diff",
          "z_from_gpu_attn_norm_max_abs_diff",
      ])
  serial_closes_layer1_local_norm = (
      _num(serial1.get("gpu_attn_norm_vs_cpu_max_abs_diff")) == 0.0)
  cpu_sqrt_adds_no_early_benefit = all(
      serial_sqrt0.get(key) == serial0.get(key) for key in [
          "gpu_attn_norm_vs_cpu_max_abs_diff",
          "qkv_from_gpu_attn_norm_max_abs_diff",
          "z_from_gpu_attn_norm_max_abs_diff",
      ])
  source_ready = all([
      source["current_shared_scale_is_parallel"],
      source["standalone_serial_sum_exists"],
      source["proven_serial_shared_scale_shape"],
      source["shared_runner_scale_apply_split"],
      source["decode_has_layer_set_primitives"],
      source["implementation_shape"]["new_kernel_families"] == 0,
      source["implementation_shape"]["per_layer_source_files"] == 0,
  ])
  checks = [
      {"name": "seq545_selected_reduction_feasibility",
       "pass": predecessor_selects},
      {"name": "global_serial_product_route_remains_closed",
       "pass": _rejected(rejected, GLOBAL_SERIAL_ROUTE)},
      {"name": "cpu_sqrt_alone_does_not_close_layer0",
       "pass": cpu_sqrt_does_not_close},
      {"name": "serial_reduction_closes_layer0_norm_qkv_z",
       "pass": serial_closes_layer0},
      {"name": "serial_reduction_closes_layer1_local_norm",
       "pass": serial_closes_layer1_local_norm},
      {"name": "cpu_sqrt_adds_no_early_layer_benefit",
       "pass": cpu_sqrt_adds_no_early_benefit},
      {"name": "o1_layer_scoped_source_shape_available",
       "pass": source_ready},
  ]
  required = all(bool(check["pass"]) for check in checks)
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "sequence": args.sequence,
      "inputs": {
          "predecessor": _rel(args.predecessor),
          "routes": _rel(args.routes),
          "rejected": _rel(args.rejected),
          "cpu_sqrt_result": _rel(args.cpu_sqrt_result),
          "serial_result": _rel(args.serial_result),
          "serial_cpu_sqrt_result": _rel(args.serial_cpu_sqrt_result),
          "serial_generated_source": _rel(args.serial_generated_source),
      },
      "checks": checks,
      "required_checks_passed": required,
      "evidence": {
          "cpu_sqrt_layer0": _row_summary(cpu0),
          "serial_layer0": _row_summary(serial0),
          "serial_layer1": _row_summary(serial1),
          "serial_cpu_sqrt_layer0": _row_summary(serial_sqrt0),
      },
      "source_findings": source,
      "target_compile_allowed": False,
      "decode_probe_allowed": False,
      "router_distribution_allowed": False,
      "speedup_claims_allowed": False,
      "disposition": (
          "reject_input_rmsnorm_cpu_sqrt_select_layer_scoped_serial_reduction_source"
          if required else
          "block_input_rmsnorm_reduction_feasibility_incomplete"),
      "rejected_route": REJECTED_ROUTE if required else None,
      "selected_next_route": SELECTED_NEXT_ROUTE if required else CURRENT_ROUTE,
      "next_route_reason": (
          "CPU-shaped sqrt alone leaves the layer0 mismatch, while the proven "
          "serial reduction with ordinary rsqrt closes layer0 norm/QKV/Z and "
          "layer1 local norm. Implement that reduction only for layers 0/1 "
          "through one parameterized layer set; keep the failed global route closed."
          if required else
          "The prior reduction evidence or source shape is incomplete; do not launch "
          "a target row."
      ),
  }


def write_outputs(metrics: dict[str, Any], out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  failed = [row["name"] for row in metrics["checks"] if not row["pass"]]
  lines = [
      f"# Seq{metrics['sequence']} Input-RMSNorm Reduction Feasibility Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- rejected_route: `{metrics['rejected_route']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- failed_checks: `{failed}`",
      "",
      metrics["next_route_reason"],
      "",
      "This is source-feasibility evidence only. It is not a target or speed claim.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--sequence", type=int, default=546)
  parser.add_argument(
      "--predecessor", type=Path,
      default=ROOT / "output/seq545-layer0-1-gpu-final-norm-precision-island-one-token-gate-20260710Tseq545Z/metrics.json")
  parser.add_argument("--routes", type=Path,
                      default=ACTIVE / "routes-ledger.json")
  parser.add_argument("--rejected", type=Path,
                      default=ACTIVE / "rejected-routes.json")
  parser.add_argument(
      "--cpu-sqrt-result", type=Path,
      default=ROOT / "output/seq500-attn-norm-scale-kernel-cpu-sqrt-target-probe-20260709Tseq500Z/cases/router_math_reason_001/result.json")
  parser.add_argument(
      "--serial-result", type=Path,
      default=ROOT / "output/seq501-attn-norm-scale-kernel-reduction-order-target-gate-20260709Tseq501Z/cases/router_math_reason_001/result.json")
  parser.add_argument(
      "--serial-cpu-sqrt-result", type=Path,
      default=ROOT / "output/seq502-attn-norm-serial-scale-cpu-sqrt-target-gate-20260709Tseq502Z/cases/router_math_reason_001/result.json")
  parser.add_argument(
      "--serial-generated-source", type=Path,
      default=ROOT / "output/seq501-attn-norm-scale-kernel-reduction-order-target-gate-20260709Tseq501Z/cases/router_math_reason_001/r2_gpu_decode_smoke.cpp")
  parser.add_argument("--kernel-source", type=Path,
                      default=ROOT / "engine/gpu/opencl/q4x8_matvec.cl")
  parser.add_argument("--engine-source", type=Path,
                      default=ROOT / "engine/src/gpu_q4x8_matvec.cpp")
  parser.add_argument("--decode-source", type=Path,
                      default=ROOT / "tools/intel-qwen36-r2-gpu-decode-smoke.py")
  parser.add_argument(
      "--out-dir", type=Path,
      default=ROOT / "output/seq546-layer0-1-input-rmsnorm-reduction-feasibility-gate-20260710Tseq546Z")
  args = parser.parse_args()
  metrics = compute(args)
  write_outputs(metrics, args.out_dir)
  print(json.dumps({
      "required_checks_passed": metrics["required_checks_passed"],
      "disposition": metrics["disposition"],
      "selected_next_route": metrics["selected_next_route"],
      "out_dir": _rel(args.out_dir),
  }, sort_keys=True))
  return 0 if metrics["required_checks_passed"] else 2


if __name__ == "__main__":
  raise SystemExit(main())
