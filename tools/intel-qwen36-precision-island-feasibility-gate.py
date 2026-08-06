#!/usr/bin/env python3
"""Classify source feasibility for a bounded early-layer precision island."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = "intel-qwen36-precision-island-feasibility-gate-v0"
CURRENT_ROUTE = (
    "router_prompt_distribution_layer0_1_precision_island_feasibility_gate"
)
SELECTED_NEXT_ROUTE = (
    "router_prompt_distribution_layer0_1_gpu_final_norm_precision_island_source_gate"
)
REJECTED_ROUTE = (
    "router_prompt_distribution_layer0_1_cpu_fallback_or_global_precision_replay"
)
REQUIRED_CLOSED_ROUTES = {
    "gpu_attention_front_handoff_8tok_cpu_linear_final_rmsnorm_isolation",
    "router_math_global_opencl_no_fma_conv_history_fix",
    "router_math_layer1_only_product_toggles",
    "router_prompt_distribution_fp64_precision_pack",
    "shared_rmsnorm_scale_kernel_serial_cpu_sqrt_product_fix",
}


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


def _line(text: str, literal: str) -> int | None:
  offset = text.find(literal)
  return None if offset < 0 else text.count("\n", 0, offset) + 1


def _present(text: str, label: str, literal: str) -> dict[str, Any]:
  line = _line(text, literal)
  return {"label": label, "present": line is not None, "line": line}


def _all_present(rows: list[dict[str, Any]]) -> bool:
  return all(row.get("present") is True for row in rows)


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


def _rejected_routes(rejected: dict[str, Any]) -> set[str]:
  return {
      str(row["route"])
      for row in rejected.get("rejected", [])
      if isinstance(row, dict) and isinstance(row.get("route"), str)
  }


def _parse_layers(text: str) -> list[int]:
  layers: list[int] = []
  for item in text.split(","):
    layer = int(item)
    if layer < 0 or layer >= 40:
      raise ValueError(f"layer out of range: {layer}")
    if layer not in layers:
      layers.append(layer)
  if not layers:
    raise ValueError("precision-island layer set is empty")
  return layers


def _source_findings(decode_path: Path, engine_path: Path,
                     kernel_path: Path, layers: list[int]) -> dict[str, Any]:
  decode = _read(decode_path)
  engine = _read(engine_path)
  kernel = _read(kernel_path)
  decode_checks = [
      _present(decode, "single_parameterized_linear_layer_function",
               "RunGpuHybridLinearLayerLive("),
      _present(decode, "layer_list_parser", "DecodeParseLayerList("),
      _present(decode, "layer_list_selector", "DecodeLayerListed("),
      _present(decode, "current_per_call_cpu_shape_selector",
               "const bool use_cpu_shape_final_norm = layer >= 4 && layer <= 10;"),
      _present(decode, "resident_delta_cpu_shape_argument",
               "use_cpu_shape_final_norm, readback_delta_attention_output"),
  ]
  engine_checks = [
      _present(engine, "cpu_shape_final_norm_parameter",
               "bool cpu_shape_final_norm"),
      _present(engine, "fast_final_norm_kernel",
               'CreateNamedKernel("linear_attn_final_norm_f32")'),
      _present(engine, "cpu_shape_final_norm_kernel",
               'CreateNamedKernel("linear_attn_final_norm_cpu_shape_f32")'),
      _present(engine, "per_call_final_norm_kernel_choice",
               "cpu_shape_final_norm ? kernel_delta_final_cpu_shape_"),
  ]
  kernel_checks = [
      _present(kernel, "cpu_shape_kernel_definition",
               "__kernel void linear_attn_final_norm_cpu_shape_f32("),
      _present(kernel, "cpu_shape_double_sqrt",
               "(float)(1.0 / sqrt((double)mean_square + (double)norm_epsilon))"),
      _present(kernel, "cpu_shape_double_silu", "const double z_double"),
  ]
  current_selector = re.search(
      r"use_cpu_shape_final_norm\s*=\s*layer\s*>=\s*4\s*&&\s*layer\s*<=\s*10",
      decode,
  )
  current_excludes_requested = current_selector is not None and all(
      layer < 4 or layer > 10 for layer in layers)
  return {
      "requested_layers": layers,
      "decode_checks": decode_checks,
      "engine_checks": engine_checks,
      "kernel_checks": kernel_checks,
      "existing_gpu_cpu_shape_path_present": (
          _all_present(decode_checks)
          and _all_present(engine_checks)
          and _all_present(kernel_checks)
      ),
      "current_selector_excludes_requested_layers": current_excludes_requested,
      "implementation_shape": {
          "selector": "one parsed layer set plus DecodeLayerListed",
          "runtime": "reuse the existing per-call GPU CPU-shaped final-norm kernel",
          "layer_source_files": 0,
          "new_opencl_kernel_families": 0,
          "target_rows_before_source_gate": 0,
      },
  }


def compute(args: argparse.Namespace) -> dict[str, Any]:
  predecessor = _load(args.predecessor)
  routes = _load(args.routes)
  rejected = _load(args.rejected)
  layers = _parse_layers(args.layers)
  source = _source_findings(
      args.decode_source, args.engine_source, args.kernel_source, layers)
  rejected_names = _rejected_routes(rejected)
  missing_closed = sorted(REQUIRED_CLOSED_ROUTES - rejected_names)
  predecessor_selects = (
      predecessor.get("required_checks_passed") is True
      and predecessor.get("selected_next_route") == CURRENT_ROUTE
      and _has_candidate(routes, 541, CURRENT_ROUTE)
      and _has_switch(
          routes, 541,
          "select_router_prompt_distribution_layer0_1_precision_island_feasibility_gate")
  )
  checks = [
      {
          "name": "seq541_selected_feasibility_gate",
          "pass": predecessor_selects,
      },
      {
          "name": "global_precision_and_cpu_fallback_replays_closed",
          "pass": not missing_closed,
          "detail": {"missing_closed_routes": missing_closed},
      },
      {
          "name": "existing_gpu_cpu_shape_final_norm_path_present",
          "pass": source["existing_gpu_cpu_shape_path_present"],
      },
      {
          "name": "current_selector_excludes_layer0_1",
          "pass": source["current_selector_excludes_requested_layers"],
      },
      {
          "name": "o1_parameterized_layer_shape_available",
          "pass": (
              source["implementation_shape"]["layer_source_files"] == 0
              and source["implementation_shape"]["new_opencl_kernel_families"] == 0
              and len(layers) <= 2
          ),
      },
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
          "decode_source": _rel(args.decode_source),
          "engine_source": _rel(args.engine_source),
          "kernel_source": _rel(args.kernel_source),
      },
      "checks": checks,
      "required_checks_passed": required,
      "source_findings": source,
      "speedup_claims_allowed": False,
      "target_compile_allowed": False,
      "decode_probe_allowed": False,
      "router_distribution_allowed": False,
      "disposition": (
          "reject_cpu_fallback_global_precision_replay_select_gpu_final_norm_precision_island_source"
          if required else
          "block_layer0_1_precision_island_infeasible_or_stale"
      ),
      "rejected_route": REJECTED_ROUTE if required else None,
      "selected_next_route": SELECTED_NEXT_ROUTE if required else CURRENT_ROUTE,
      "next_route_reason": (
          "Reuse the existing GPU CPU-shaped linear final-norm kernel behind one "
          "parsed layer selector for layers 0/1. The next gate is source/generate-only; "
          "global precision toggles, CPU fallbacks, target rows, and promotion remain closed."
          if required else
          "The source shape or route prerequisites are incomplete; keep the feasibility "
          "gate open and do not launch target rows."
      ),
  }


def write_outputs(metrics: dict[str, Any], out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  lines = [
      f"# Seq{metrics['sequence']} Precision-Island Feasibility Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- rejected_route: `{metrics['rejected_route']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      "",
      metrics["next_route_reason"],
      "",
      "This is source-feasibility evidence only. It is not a target or speed claim.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--sequence", type=int, default=542)
  parser.add_argument("--layers", default="0,1")
  parser.add_argument(
      "--predecessor", type=Path,
      default=ROOT / "output/seq541-layer0-exact-delta-source-gate-20260710Tseq541Z/metrics.json")
  parser.add_argument("--routes", type=Path,
                      default=ACTIVE / "routes-ledger.json")
  parser.add_argument("--rejected", type=Path,
                      default=ACTIVE / "rejected-routes.json")
  parser.add_argument("--decode-source", type=Path,
                      default=ROOT / "tools/intel-qwen36-r2-gpu-decode-smoke.py")
  parser.add_argument("--engine-source", type=Path,
                      default=ROOT / "engine/src/gpu_q4x8_matvec.cpp")
  parser.add_argument("--kernel-source", type=Path,
                      default=ROOT / "engine/gpu/opencl/q4x8_matvec.cl")
  parser.add_argument(
      "--out-dir", type=Path,
      default=ROOT / "output/seq542-layer0-1-precision-island-feasibility-gate-20260710Tseq542Z")
  args = parser.parse_args()
  metrics = compute(args)
  write_outputs(metrics, args.out_dir)
  print(json.dumps(metrics, indent=2, sort_keys=True))
  return 0 if metrics["required_checks_passed"] else 2


if __name__ == "__main__":
  raise SystemExit(main())
