#!/usr/bin/env python3
"""Validate the layer0 CPU-order linear output-projection decode source."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = "intel-qwen36-linear-output-projection-cpuorder-source-v0"
CURRENT_ROUTE = (
    "router_prompt_distribution_layer0_linear_output_projection_cpuorder_"
    "source_gate"
)
SELECTED_NEXT_ROUTE = (
    "router_prompt_distribution_layer0_linear_output_projection_cpuorder_"
    "target_compile_gate"
)
EXPECTED_LAYERS = [0]


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


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _markers(source: str, wrapper: bool) -> list[dict[str, Any]]:
  markers = {
      "default_empty_layer_selector": (
          "std::vector<int> linear_output_projection_cpu_order_layers;"
          in source),
      "runtime_layer_set_env": (
          "IQ36_LINEAR_OUTPUT_PROJECTION_CPU_ORDER_LAYERS" in source),
      "layer_scoped_selection": (
          "g_decode_linear_output_projection_cpu_order_layers, layer" in source),
      "resident_cpuorder_runner": (
          "RunResidentRawQ4KCpuOrder" in source),
      "gpu_residual_rmsnorm_reuse": (
          "RunResidualRmsNormHiddenResidentWeight" in source
          and "RunResidualRmsNormHidden(" in source),
      "rowblock_default_preserved": (
          "RunGpuAttentionFrontFromInputHandle" in source
          and "RunGpuAttentionFront(" in source),
      "output_records_active_layers": (
          '"linear_output_projection_cpu_order_layers"' in source
          or '\\"linear_output_projection_cpu_order_layers\\"' in source),
  }
  if wrapper:
    markers.update({
        "source_only_guard": (
            "IQ36_LINEAR_OUTPUT_PROJECTION_CPU_ORDER_SOURCE is source-gate only"
            in source),
        "source_requires_layer0": (
            "IQ36_LINEAR_OUTPUT_PROJECTION_CPU_ORDER_LAYERS=0" in source),
        "manifest_records_source": (
            '"linear_output_projection_cpu_order_source"' in source),
    })
  return [{"name": name, "pass": passed}
          for name, passed in markers.items()]


def _compile(args: argparse.Namespace) -> dict[str, Any]:
  compile_dir = args.out_dir / "compile"
  compile_dir.mkdir(parents=True, exist_ok=True)
  generated = args.generate_dir / "r2_gpu_decode_smoke.cpp"
  commands = [
      [args.cxx, "-std=c++20", "-Iengine/include", "-O0", "-c",
       _rel(generated), "-o", _rel(compile_dir / "r2_gpu_decode_smoke.o")],
      [args.cxx, "-std=c++20", "-Iengine/include", "-O0", "-c",
       _rel(args.cpuorder_source),
       "-o", _rel(compile_dir / "gpu_q4_cpu_order_matvec.o")],
  ]
  runs = []
  for index, command in enumerate(commands):
    completed = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, check=False)
    stdout_path = compile_dir / f"compile{index}.stdout.txt"
    stderr_path = compile_dir / f"compile{index}.stderr.txt"
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    runs.append({
        "command": command,
        "returncode": completed.returncode,
        "stdout": _rel(stdout_path),
        "stderr": _rel(stderr_path),
    })
  return {"passed": all(row["returncode"] == 0 for row in runs),
          "runs": runs}


def compute(args: argparse.Namespace) -> dict[str, Any]:
  predecessor = _load(args.predecessor)
  routes = _load(args.routes)
  result_path = args.generate_dir / "result.json"
  generated_path = args.generate_dir / "r2_gpu_decode_smoke.cpp"
  manifest = _load(result_path)
  wrapper_markers = _markers(
      args.decode_source.read_text(encoding="utf-8"), wrapper=True)
  generated_markers = _markers(
      generated_path.read_text(encoding="utf-8"), wrapper=False)
  compile_result = _compile(args)
  predecessor_selects = (
      predecessor.get("required_checks_passed") is True
      and predecessor.get("decode_source_gate_allowed") is True
      and predecessor.get("selected_next_route") == CURRENT_ROUTE
      and _has_candidate(routes, 555, CURRENT_ROUTE)
      and _has_switch(
          routes, 555,
          "select_router_prompt_distribution_layer0_linear_output_projection_"
          "cpuorder_source_gate"))
  manifest_passes = (
      manifest.get("generate_only") is True
      and manifest.get("required_checks_passed") is False
      and manifest.get("linear_output_projection_cpu_order_source") is True
      and manifest.get("linear_output_projection_cpu_order_layers")
      == EXPECTED_LAYERS
      and manifest.get("speedup_claims_allowed") is False
      and not (args.generate_dir / "smoke.json").exists())
  checks = [
      {"name": "seq555_selected_decode_source_gate",
       "pass": predecessor_selects},
      {"name": "generate_only_manifest_records_layer0",
       "pass": manifest_passes},
      {"name": "decode_source_default_off_selector_and_guard",
       "pass": all(row["pass"] for row in wrapper_markers),
       "detail": wrapper_markers},
      {"name": "generated_cpp_contains_layer0_cpuorder_path",
       "pass": all(row["pass"] for row in generated_markers),
       "detail": generated_markers},
      {"name": "generated_and_cpuorder_sources_compile_locally",
       "pass": compile_result["passed"], "detail": compile_result},
  ]
  required = all(bool(row["pass"]) for row in checks)
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "sequence": args.sequence,
      "inputs": {
          "predecessor": _rel(args.predecessor),
          "routes": _rel(args.routes),
          "decode_source": _rel(args.decode_source),
          "decode_source_sha256": _sha256(args.decode_source),
          "generate_only_result": _rel(result_path),
          "generated_cpp": _rel(generated_path),
          "cpuorder_source": _rel(args.cpuorder_source),
      },
      "checks": checks,
      "required_checks_passed": required,
      "compile": compile_result,
      "target_compile_allowed": required,
      "decode_probe_allowed": False,
      "speedup_claims_allowed": False,
      "promotion_allowed": False,
      "disposition": (
          "accept_layer0_cpuorder_projection_source"
          if required else
          "block_layer0_cpuorder_projection_source"),
      "selected_next_route": SELECTED_NEXT_ROUTE if required else CURRENT_ROUTE,
      "next_route_reason": (
          "The default-empty layer selector routes only selected linear "
          "`ssm_out.weight` projections through the existing GPU CPU-order Q4 "
          "runner, then reuses the GPU residual/RMSNorm primitive. Layer0 is "
          "recorded in generate-only evidence, the default rowblock path is "
          "unchanged, and generated plus CPU-order sources compile locally. "
          "Target-compile next without launching a token."
          if required else
          "Fix the source selector, guard, manifest, or compile before target work."
      ),
  }


def write_outputs(metrics: dict[str, Any], out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  failed = [row["name"] for row in metrics["checks"] if not row["pass"]]
  lines = [
      f"# Seq{metrics['sequence']} Layer0 CPU-Order Projection Source Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- local compile passed: `{str(metrics['compile']['passed']).lower()}`",
      f"- failed_checks: `{failed}`",
      "",
      metrics["next_route_reason"],
      "",
      "This is source/generate-only evidence. It is not a decode or speed claim.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--sequence", type=int, default=556)
  parser.add_argument(
      "--predecessor", type=Path,
      default=ROOT / "output/seq555-layer0-linear-output-projection-cpuorder-component-target-gate-20260710Tseq555Z/metrics.json")
  parser.add_argument("--routes", type=Path,
                      default=ACTIVE / "routes-ledger.json")
  parser.add_argument(
      "--decode-source", type=Path,
      default=ROOT / "tools/intel-qwen36-r2-gpu-decode-smoke.py")
  parser.add_argument(
      "--generate-dir", type=Path,
      default=ROOT / "output/seq556-layer0-linear-output-projection-cpuorder-source-20260710Tseq556Z")
  parser.add_argument(
      "--cpuorder-source", type=Path,
      default=ROOT / "engine/src/gpu_q4_cpu_order_matvec.cpp")
  parser.add_argument("--cxx", default="c++")
  parser.add_argument(
      "--out-dir", type=Path,
      default=ROOT / "output/seq556-layer0-linear-output-projection-cpuorder-source-gate-20260710Tseq556Z")
  args = parser.parse_args()
  args.out_dir.mkdir(parents=True, exist_ok=True)
  metrics = compute(args)
  write_outputs(metrics, args.out_dir)
  print(json.dumps({
      "required_checks_passed": metrics["required_checks_passed"],
      "disposition": metrics["disposition"],
      "target_compile_allowed": metrics["target_compile_allowed"],
      "selected_next_route": metrics["selected_next_route"],
      "out_dir": _rel(args.out_dir),
  }, sort_keys=True))
  return 0 if metrics["required_checks_passed"] else 2


if __name__ == "__main__":
  raise SystemExit(main())
