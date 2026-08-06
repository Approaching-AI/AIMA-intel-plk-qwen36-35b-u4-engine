#!/usr/bin/env python3
"""Validate the CPU-order linear output-projection component probe source."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = (
    "intel-qwen36-linear-output-projection-cpuorder-component-source-v0"
)
CURRENT_ROUTE = (
    "router_prompt_distribution_layer0_linear_output_projection_reduction_"
    "order_component_source_gate"
)
SELECTED_NEXT_ROUTE = (
    "router_prompt_distribution_layer0_linear_output_projection_reduction_"
    "order_component_target_gate"
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


def _load_probe_module(path: Path) -> Any:
  spec = importlib.util.spec_from_file_location("iq36_output_projection_probe", path)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"could not load probe module: {path}")
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


def _compile_probe(args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
  module = _load_probe_module(args.probe_source)
  opencl = args.opencl_source.read_text(encoding="utf-8")
  generated = module.PROBE_CPP.replace(
      "@@OPENCL_SOURCE_LITERAL@@", module.cpp_raw_string_literal(opencl))
  generated_path = out_dir / "gpu_q4x8_output_projection_probe.cpp"
  generated_path.write_text(generated, encoding="utf-8")
  binary = out_dir / "gpu_q4x8_output_projection_probe"
  command = [
      "g++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Wpedantic",
      "-I", str(ROOT / "engine/include"),
      str(ROOT / "engine/src/gguf_loader.cpp"),
      str(ROOT / "engine/src/gpu_q4x8_matvec.cpp"),
      str(ROOT / "engine/src/gpu_q4_cpu_order_matvec.cpp"),
      str(generated_path), "-ldl", "-pthread", "-o", str(binary),
  ]
  completed = subprocess.run(
      command, cwd=ROOT, capture_output=True, text=True, check=False)
  return {
      "command": command,
      "returncode": completed.returncode,
      "stdout": completed.stdout,
      "stderr": completed.stderr,
      "generated_source": _rel(generated_path),
      "binary": _rel(binary),
  }


def compute(args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
  predecessor = _load(args.predecessor)
  routes = _load(args.routes)
  source = args.probe_source.read_text(encoding="utf-8")
  compile_result = _compile_probe(args, out_dir)
  source_markers = all(marker in source for marker in (
      'SCHEMA_VERSION = "intel-qwen36-gpu-q4x8-output-projection-probe-v2"',
      '"engine/src/gpu_q4_cpu_order_matvec.cpp"',
      "GpuQ4KCpuOrderMatvecRunner cpu_order_runner",
      "UploadRawQ4KCpuOrder(raw, rows, blocks_per_row)",
      "RunResidentRawQ4KCpuOrder",
      '"cpu_order_output_projection_matches_oracle"',
      '"linear_attn_out_cpu_order"',
  ))
  component_only = (
      "RunGpuHybridLinearLayer" not in source
      and "IQ36_LINEAR_OUTPUT_PROJECTION_CPU_ORDER_LAYERS" not in source
      and "speedup_claims_allowed" in source)
  checks = [
      {
          "name": "seq553_selected_component_source_gate",
          "pass": (
              predecessor.get("required_checks_passed") is True
              and predecessor.get("component_source_gate_allowed") is True
              and predecessor.get("selected_next_route") == CURRENT_ROUTE
              and _has_candidate(routes, 553, CURRENT_ROUTE)
              and _has_switch(
                  routes, 553,
                  "select_router_prompt_distribution_layer0_linear_output_"
                  "projection_reduction_order_component_source_gate")),
      },
      {"name": "cpu_order_component_source_markers_present",
       "pass": source_markers},
      {"name": "probe_remains_component_only", "pass": component_only},
      {"name": "generated_component_compiles_locally",
       "pass": compile_result["returncode"] == 0},
  ]
  required = all(bool(row["pass"]) for row in checks)
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "sequence": args.sequence,
      "inputs": {
          "predecessor": _rel(args.predecessor),
          "routes": _rel(args.routes),
          "probe_source": _rel(args.probe_source),
          "opencl_source": _rel(args.opencl_source),
      },
      "checks": checks,
      "required_checks_passed": required,
      "compile": compile_result,
      "target_component_allowed": required,
      "decode_probe_allowed": False,
      "speedup_claims_allowed": False,
      "promotion_allowed": False,
      "disposition": (
          "accept_cpuorder_output_projection_component_source"
          if required else
          "block_cpuorder_output_projection_component_source"),
      "selected_next_route": SELECTED_NEXT_ROUTE if required else CURRENT_ROUTE,
      "next_route_reason": (
          "The existing captured-payload output-projection probe now runs the "
          "generic resident CPU-order Q4 kernel alongside rowlane/group8/"
          "rowblock16, reports CPU and oracle comparisons plus event timing, "
          "and compiles locally with the production engine sources. Run only "
          "the layer0 target component probe next; decode remains blocked."
          if required else
          "The component source or local compile is incomplete; keep target work blocked."
      ),
  }


def write_outputs(metrics: dict[str, Any], out_dir: Path) -> None:
  (out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  failed = [row["name"] for row in metrics["checks"] if not row["pass"]]
  lines = [
      f"# Seq{metrics['sequence']} Linear Output CPU-Order Component Source Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- local compile returncode: `{metrics['compile']['returncode']}`",
      f"- failed_checks: `{failed}`",
      "",
      metrics["next_route_reason"],
      "",
      "This is source/local-compile evidence only. It is not a speed claim.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--sequence", type=int, default=554)
  parser.add_argument(
      "--predecessor", type=Path,
      default=ROOT / "output/seq553-gpu-cpu-arithmetic-parity-coverage-audit-gate-20260710Tseq553Z/metrics.json")
  parser.add_argument("--routes", type=Path,
                      default=ACTIVE / "routes-ledger.json")
  parser.add_argument(
      "--probe-source", type=Path,
      default=ROOT / "tools/intel-qwen36-gpu-q4x8-output-projection-probe.py")
  parser.add_argument(
      "--opencl-source", type=Path,
      default=ROOT / "engine/gpu/opencl/q4x8_matvec.cl")
  parser.add_argument(
      "--out-dir", type=Path,
      default=ROOT / "output/seq554-layer0-linear-output-projection-cpuorder-component-source-gate-20260710Tseq554Z")
  args = parser.parse_args()
  args.out_dir.mkdir(parents=True, exist_ok=True)
  metrics = compute(args, args.out_dir)
  write_outputs(metrics, args.out_dir)
  print(json.dumps({
      "required_checks_passed": metrics["required_checks_passed"],
      "disposition": metrics["disposition"],
      "target_component_allowed": metrics["target_component_allowed"],
      "selected_next_route": metrics["selected_next_route"],
      "out_dir": _rel(args.out_dir),
  }, sort_keys=True))
  return 0 if metrics["required_checks_passed"] else 2


if __name__ == "__main__":
  raise SystemExit(main())
