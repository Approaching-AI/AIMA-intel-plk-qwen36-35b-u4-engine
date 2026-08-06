#!/usr/bin/env python3
"""Gate the source/generate-only layer-local GPU precision island."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = "intel-qwen36-precision-island-source-gate-v0"
CURRENT_ROUTE = (
    "router_prompt_distribution_layer0_1_gpu_final_norm_precision_island_source_gate"
)
SELECTED_NEXT_ROUTE = (
    "router_prompt_distribution_layer0_1_gpu_final_norm_precision_island_target_compile_gate"
)
EXPECTED_LAYERS = [0, 1, 4, 5, 6, 8, 9, 10]


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


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _marker(text: str, label: str, literal: str,
            *, expected: bool = True) -> dict[str, Any]:
  offset = text.find(literal)
  present = offset >= 0
  return {
      "label": label,
      "expected": expected,
      "pass": present is expected,
      "line": text.count("\n", 0, offset) + 1 if present else None,
  }


def _source_markers(text: str, *, include_wrapper: bool) -> list[dict[str, Any]]:
  rows = [
      _marker(text, "default_cpu_shape_layer_set",
              "linear_final_cpu_shape_layers = {4, 5, 6, 8, 9, 10}"),
      _marker(text, "runtime_layer_set_env",
              'std::getenv("IQ36_LINEAR_FINAL_CPU_SHAPE_LAYERS")'),
      _marker(text, "parsed_runtime_layer_set",
              "args.linear_final_cpu_shape_layers = DecodeParseLayerList(layers_env)"),
      _marker(text, "parameterized_layer_selector",
              "DecodeLayerListed(\n        g_decode_linear_final_cpu_shape_layers, layer)"),
      _marker(text, "per_session_layer_set_assignment",
              "g_decode_linear_final_cpu_shape_layers =\n        args.linear_final_cpu_shape_layers"),
      _marker(text, "per_session_layer_set_cleanup",
              "g_decode_linear_final_cpu_shape_layers.clear()"),
      _marker(text, "result_layer_set_evidence",
              '"\\\"linear_final_cpu_shape_layers\\\":"'),
      _marker(text, "old_hardcoded_selector_removed",
              "const bool use_cpu_shape_final_norm = layer >= 4 && layer <= 10",
              expected=False),
  ]
  if include_wrapper:
    rows.extend([
        _marker(text, "source_only_marker_guard",
                "IQ36_LAYER0_1_PRECISION_ISLAND_SOURCE is source-gate only"),
        _marker(text, "source_requires_layers0_1",
                "IQ36_LAYER0_1_PRECISION_ISLAND_SOURCE requires"),
        _marker(text, "remote_layer_set_forwarding",
                '"IQ36_LINEAR_FINAL_CPU_SHAPE_LAYERS="'),
    ])
  return rows


def _engine_markers(engine: str, kernel: str) -> list[dict[str, Any]]:
  return [
      _marker(engine, "per_call_cpu_shape_bool", "bool cpu_shape_final_norm"),
      _marker(engine, "per_call_kernel_choice",
              "cpu_shape_final_norm ? kernel_delta_final_cpu_shape_"),
      _marker(kernel, "cpu_shape_kernel",
              "__kernel void linear_attn_final_norm_cpu_shape_f32("),
      _marker(kernel, "cpu_shape_double_sqrt",
              "(float)(1.0 / sqrt((double)mean_square + (double)norm_epsilon))"),
  ]


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


def _compile(args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
  compile_dir = out_dir / "compile"
  compile_dir.mkdir(parents=True, exist_ok=True)
  generated_cpp = args.generate_dir / "r2_gpu_decode_smoke.cpp"
  commands = [
      [args.cxx, "-std=c++20", "-Iengine/include", "-O0", "-c",
       _rel(generated_cpp), "-o", _rel(compile_dir / "r2_gpu_decode_smoke.o")],
      [args.cxx, "-std=c++20", "-Iengine/include", "-O0", "-c",
       _rel(args.engine_source), "-o", _rel(compile_dir / "gpu_q4x8_matvec.o")],
  ]
  runs = []
  for index, command in enumerate(commands):
    proc = subprocess.run(
        command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=False)
    stdout_path = compile_dir / f"compile{index}.stdout.txt"
    stderr_path = compile_dir / f"compile{index}.stderr.txt"
    stdout_path.write_text(proc.stdout, encoding="utf-8")
    stderr_path.write_text(proc.stderr, encoding="utf-8")
    runs.append({
        "command": command,
        "returncode": proc.returncode,
        "stdout": _rel(stdout_path),
        "stderr": _rel(stderr_path),
    })
  return {"passed": all(row["returncode"] == 0 for row in runs), "runs": runs}


def compute(args: argparse.Namespace) -> dict[str, Any]:
  predecessor = _load(args.predecessor)
  routes = _load(args.routes)
  manifest_path = args.generate_dir / "result.json"
  generated_path = args.generate_dir / "r2_gpu_decode_smoke.cpp"
  manifest = _load(manifest_path)
  source_markers = _source_markers(_read(args.decode_source), include_wrapper=True)
  generated_markers = _source_markers(_read(generated_path), include_wrapper=False)
  engine_markers = _engine_markers(
      _read(args.engine_source), _read(args.kernel_source))
  compile_result = _compile(args, args.out_dir)
  predecessor_selects = (
      predecessor.get("required_checks_passed") is True
      and predecessor.get("selected_next_route") == CURRENT_ROUTE
      and _has_candidate(routes, 542, CURRENT_ROUTE)
      and _has_switch(
          routes, 542,
          "select_router_prompt_distribution_layer0_1_gpu_final_norm_precision_island_source_gate")
  )
  manifest_passes = (
      manifest.get("generate_only") is True
      and manifest.get("required_checks_passed") is False
      and manifest.get("layer0_1_precision_island_source") is True
      and manifest.get("linear_final_cpu_shape_layers") == EXPECTED_LAYERS
      and manifest.get("speedup_claims_allowed") is False
      and not (args.generate_dir / "smoke.json").exists()
  )
  checks = [
      {"name": "seq542_selected_source_gate", "pass": predecessor_selects},
      {"name": "generate_only_manifest_records_layer0_1", "pass": manifest_passes},
      {
          "name": "decode_source_has_parameterized_selector_and_guard",
          "pass": all(row["pass"] for row in source_markers),
          "detail": source_markers,
      },
      {
          "name": "generated_cpp_has_parameterized_selector",
          "pass": all(row["pass"] for row in generated_markers),
          "detail": generated_markers,
      },
      {
          "name": "existing_gpu_cpu_shape_kernel_path_preserved",
          "pass": all(row["pass"] for row in engine_markers),
          "detail": engine_markers,
      },
      {
          "name": "generated_and_engine_sources_compile_locally",
          "pass": compile_result["passed"],
          "detail": compile_result,
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
          "decode_source": _rel(args.decode_source),
          "decode_source_sha256": _sha256(args.decode_source),
          "engine_source": _rel(args.engine_source),
          "kernel_source": _rel(args.kernel_source),
          "generate_only_result": _rel(manifest_path),
          "generated_cpp": _rel(generated_path),
          "generated_cpp_sha256": _sha256(generated_path),
      },
      "checks": checks,
      "required_checks_passed": required,
      "precision_island": {
          "layers": [0, 1],
          "effective_cpu_shape_layers": EXPECTED_LAYERS,
          "kernel": "linear_attn_final_norm_cpu_shape_f32",
          "new_kernel_families": 0,
          "per_layer_source_files": 0,
          "cpu_shadow_values": False,
          "host_sync": False,
      },
      "compile": compile_result,
      "target_compile_allowed": required,
      "decode_probe_allowed": False,
      "router_distribution_allowed": False,
      "speedup_claims_allowed": False,
      "disposition": (
          "accept_layer0_1_gpu_final_norm_precision_island_source"
          if required else
          "reject_layer0_1_gpu_final_norm_precision_island_source"
      ),
      "selected_next_route": SELECTED_NEXT_ROUTE if required else CURRENT_ROUTE,
      "next_route_reason": (
          "The source is default-preserving, selects layers0/1 through one parsed "
          "layer set, reuses the existing GPU CPU-shaped final-norm kernel, and "
          "compiles locally. Target compile is required before any token or distribution row."
          if required else
          "Fix the source selector, generate-only manifest, or local compile before target use."
      ),
  }


def write_outputs(metrics: dict[str, Any], out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  failed = [row["name"] for row in metrics["checks"] if not row["pass"]]
  lines = [
      f"# Seq{metrics['sequence']} Precision-Island Source Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- target_compile_allowed: `{str(metrics['target_compile_allowed']).lower()}`",
      f"- failed_checks: `{failed}`",
      "",
      metrics["next_route_reason"],
      "",
      "This is source/generate-only evidence. It is not target, token, or speed evidence.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--sequence", type=int, default=543)
  parser.add_argument(
      "--predecessor", type=Path,
      default=ROOT / "output/seq542-layer0-1-precision-island-feasibility-gate-20260710Tseq542Z/metrics.json")
  parser.add_argument("--routes", type=Path,
                      default=ACTIVE / "routes-ledger.json")
  parser.add_argument("--decode-source", type=Path,
                      default=ROOT / "tools/intel-qwen36-r2-gpu-decode-smoke.py")
  parser.add_argument("--engine-source", type=Path,
                      default=ROOT / "engine/src/gpu_q4x8_matvec.cpp")
  parser.add_argument("--kernel-source", type=Path,
                      default=ROOT / "engine/gpu/opencl/q4x8_matvec.cl")
  parser.add_argument(
      "--generate-dir", type=Path,
      default=ROOT / "output/seq543-layer0-1-gpu-final-norm-precision-island-source-20260710Tseq543Z")
  parser.add_argument(
      "--out-dir", type=Path,
      default=ROOT / "output/seq543-layer0-1-gpu-final-norm-precision-island-source-gate-20260710Tseq543Z")
  parser.add_argument("--cxx", default="c++")
  args = parser.parse_args()
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
