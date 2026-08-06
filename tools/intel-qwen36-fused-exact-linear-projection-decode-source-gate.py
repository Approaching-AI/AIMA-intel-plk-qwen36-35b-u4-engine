#!/usr/bin/env python3
"""Gate default-off fused exact projection decode source and local compile."""

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
SCHEMA_VERSION = "intel-qwen36-fused-exact-linear-projection-decode-source-v0"
CURRENT_ROUTE = (
    "router_prompt_distribution_fused_exact_linear_projection_decode_source_gate"
)
SELECTED_NEXT_ROUTE = (
    "router_prompt_distribution_fused_exact_linear_projection_decode_target_compile_gate"
)
LINEAR_LAYERS = [
    0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18,
    20, 21, 22, 24, 25, 26, 28, 29, 30, 32, 33, 34, 36, 37, 38,
]


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


def _decode_markers(source: str, wrapper: bool) -> list[dict[str, Any]]:
  markers = {
      "default_off_runtime_selector": (
          "bool linear_output_projection_rowblock16_cpuorder_finalize = false;"
          in source
          and "g_decode_linear_output_projection_rowblock16_cpuorder_finalize = false;"
          in source),
      "runtime_selector_reuses_existing_layer_set": (
          "IQ36_LINEAR_OUTPUT_PROJECTION_ROWBLOCK16_CPUORDER_FINALIZE"
          in source
          and "g_decode_linear_output_projection_cpu_order_layers, layer"
          in source),
      "old_separate_cpuorder_path_is_preserved": (
          "use_separate_cpu_order_output_projection" in source
          and "RunResidentRawQ4KCpuOrder(" in source),
      "fused_path_removes_final_output_readback": (
          "use_separate_cpu_order_output_projection ||" in source
          and "readback_delta_final_output" in source),
      "fused_path_requires_device_q8_handoff": (
          "fused exact output projection requires device-Q8 handoff" in source
          and "use_fused_exact_output_projection" in source),
      "device_q8_handoff_receives_exact_kernel_selector": (
          "attention_residual_input_handle," in source
          and "use_fused_exact_output_projection);" in source),
      "runtime_output_records_selector": (
          "linear_output_projection_rowblock16_cpuorder_finalize" in source),
  }
  if wrapper:
    markers.update({
        "source_only_guard_exists": (
            "IQ36_FUSED_EXACT_LINEAR_OUTPUT_PROJECTION_SOURCE is source-gate only"
            in source),
        "source_guard_locks_device_q8_and_30_layers": (
            "expected_all_linear_layers" in source
            and "the exact 30-linear-layer set" in source),
        "manifest_records_no_host_bridge": (
            '"fused_exact_linear_output_projection_host_bridge_allowed": False'
            in source),
    })
  return [{"name": name, "pass": passed}
          for name, passed in markers.items()]


def _engine_markers(header: str, runner: str) -> list[dict[str, Any]]:
  markers = {
      "public_handoff_selector_defaults_off": (
          "bool use_rowblock16_cpuorder_finalize = false);" in header),
      "input_handle_path_selects_locked_kernel": (
          "if (use_rowblock16_cpuorder_finalize)" in runner
          and "RunRowblock16Kernel(" in runner
          and "row_groups, repeat, true);" in runner),
      "candidate_reuses_device_q8_and_projection_buffers": (
          "input.buffer, block_count, q8_qs_buffer, q8_bsums_buffer"
          in runner
          and "resident.buffer, q8_qs_buffer, q8_bsums_buffer, q8_d_buffer"
          in runner
          and "projection_buffer, resident.rows, resident.blocks_per_row"
          in runner),
      "candidate_has_bpr16_guard": (
          "rowblock16 CPU-order input-handle projection requires BPR16"
          in runner),
  }
  return [{"name": name, "pass": passed}
          for name, passed in markers.items()]


def _run(command: list[str], compile_dir: Path,
         stem: str) -> dict[str, Any]:
  completed = subprocess.run(
      command, cwd=ROOT, capture_output=True, text=True, check=False)
  stdout_path = compile_dir / f"{stem}.stdout.txt"
  stderr_path = compile_dir / f"{stem}.stderr.txt"
  stdout_path.write_text(completed.stdout, encoding="utf-8")
  stderr_path.write_text(completed.stderr, encoding="utf-8")
  return {
      "passed": completed.returncode == 0,
      "command": command,
      "returncode": completed.returncode,
      "stdout": _rel(stdout_path),
      "stderr": _rel(stderr_path),
  }


def _code_volume() -> dict[str, Any]:
  completed = subprocess.run(
      ["python3", "tools/intel-qwen36-code-volume-check.py"],
      cwd=ROOT, capture_output=True, text=True, check=False)
  return {
      "passed": completed.returncode == 0,
      "returncode": completed.returncode,
      "stdout": completed.stdout.strip(),
      "stderr": completed.stderr.strip(),
  }


def compute(args: argparse.Namespace) -> dict[str, Any]:
  predecessor = _load(args.predecessor)
  routes = _load(args.routes)
  result_path = args.generate_dir / "result.json"
  generated_path = args.generate_dir / "r2_gpu_decode_smoke.cpp"
  manifest = _load(result_path)
  wrapper = args.decode_source.read_text(encoding="utf-8")
  generated = generated_path.read_text(encoding="utf-8")
  header = args.header_source.read_text(encoding="utf-8")
  runner = args.runner_source.read_text(encoding="utf-8")
  wrapper_markers = _decode_markers(wrapper, wrapper=True)
  generated_markers = _decode_markers(generated, wrapper=False)
  engine_markers = _engine_markers(header, runner)
  compile_dir = args.out_dir / "compile"
  compile_dir.mkdir(parents=True, exist_ok=True)
  compiles = {
      "generated_decode": _run([
          args.cxx, "-std=c++20", "-Iengine/include", "-O0", "-c",
          _rel(generated_path), "-o", _rel(compile_dir / "r2_gpu_decode_smoke.o"),
      ], compile_dir, "generated-decode"),
      "engine_runner": _run([
          args.cxx, "-std=c++20", "-Iengine/include", "-O0", "-c",
          _rel(args.runner_source), "-o", _rel(compile_dir / "gpu_q4x8_matvec.o"),
      ], compile_dir, "engine-runner"),
      "opencl_syntax": _run([
          args.clang, "-x", "cl", "-target", "spir64", "-cl-std=CL1.2",
          "-fsyntax-only", _rel(args.opencl_source),
      ], compile_dir, "opencl-syntax"),
  }
  code_volume = _code_volume()

  predecessor_selects = (
      predecessor.get("measurement_complete") is True
      and predecessor.get("required_checks_passed") is True
      and predecessor.get("component_passed") is True
      and predecessor.get("decode_source_allowed") is True
      and predecessor.get("decode_probe_allowed") is False
      and predecessor.get("token_row_allowed") is False
      and predecessor.get("selected_next_route") == CURRENT_ROUTE
      and _has_candidate(routes, 589, CURRENT_ROUTE)
      and _has_switch(
          routes, 589,
          "select_router_prompt_distribution_fused_exact_linear_projection_"
          "decode_source_gate"))
  manifest_passes = (
      manifest.get("generate_only") is True
      and manifest.get("required_checks_passed") is False
      and manifest.get("fused_exact_linear_output_projection_source") is True
      and manifest.get(
          "linear_output_projection_rowblock16_cpuorder_finalize") is True
      and manifest.get("fused_exact_linear_output_projection_layers")
      == LINEAR_LAYERS
      and manifest.get("linear_output_projection_cpu_order_layers")
      == LINEAR_LAYERS
      and manifest.get(
          "fused_exact_linear_output_projection_device_q8_handoff") is True
      and manifest.get(
          "fused_exact_linear_output_projection_host_bridge_allowed") is False
      and not (args.generate_dir / "smoke.json").exists())
  checks = [
      {"name": "seq589_selected_decode_source_only_gate",
       "pass": predecessor_selects},
      {"name": "generate_only_manifest_locks_30_layers_device_q8_no_bridge",
       "pass": manifest_passes},
      {"name": "wrapper_default_off_source_guard_and_manifest_pass",
       "pass": all(row["pass"] for row in wrapper_markers),
       "detail": wrapper_markers},
      {"name": "generated_decode_contains_fused_and_preserved_old_paths",
       "pass": all(row["pass"] for row in generated_markers),
       "detail": generated_markers},
      {"name": "engine_handoff_reuses_existing_device_buffers",
       "pass": all(row["pass"] for row in engine_markers),
       "detail": engine_markers},
      {"name": "generated_engine_and_opencl_compile_locally",
       "pass": all(row["passed"] for row in compiles.values()),
       "detail": compiles},
      {"name": "code_volume_ceiling_is_preserved",
       "pass": code_volume["passed"], "detail": code_volume},
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
          "generated_cpp_sha256": _sha256(generated_path),
          "header_source": _rel(args.header_source),
          "header_source_sha256": _sha256(args.header_source),
          "runner_source": _rel(args.runner_source),
          "runner_source_sha256": _sha256(args.runner_source),
          "opencl_source": _rel(args.opencl_source),
          "opencl_source_sha256": _sha256(args.opencl_source),
      },
      "checks": checks,
      "required_checks_passed": required,
      "compile": compiles,
      "target_compile_allowed": required,
      "one_token_probe_allowed": False,
      "decode_probe_allowed": False,
      "token_row_allowed": False,
      "validation_or_test_allowed": False,
      "speedup_claims_allowed": False,
      "promotion_allowed": False,
      "disposition": (
          "accept_fused_exact_projection_decode_source"
          if required else "block_fused_exact_projection_decode_source"),
      "selected_next_route": (
          SELECTED_NEXT_ROUTE if required else CURRENT_ROUTE),
      "next_route_reason": (
          "The default-off 30-linear-layer selector routes only the existing "
          "device-Q8 projection matvec through the exact finalize kernel; the "
          "old separate CPU-order diagnostic remains intact, no final-output "
          "host bridge is introduced, and generated/engine/OpenCL sources "
          "compile locally. Target-compile without a token next."
          if required else
          "Repair the source selector, no-bridge handoff, manifest, or local "
          "compile before target work."),
  }


def write_outputs(metrics: dict[str, Any], out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  (out_dir / "manifest.json").write_text(
      json.dumps({
          "schema_version": metrics["schema_version"],
          "workstream": metrics["workstream"],
          "tool": _rel(Path(__file__)),
          "inputs": metrics["inputs"],
          "selected_next_route": metrics["selected_next_route"],
          "target_compile_allowed": metrics["target_compile_allowed"],
          "one_token_probe_allowed": False,
          "token_row_allowed": False,
          "speedup_claims_allowed": False,
      }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  failed = [row["name"] for row in metrics["checks"] if not row["pass"]]
  lines = [
      f"# Seq{metrics['sequence']} Fused Exact Projection Decode Source",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- target_compile_allowed: `{str(metrics['target_compile_allowed']).lower()}`",
      f"- failed_checks: `{failed}`",
      "",
      metrics["next_route_reason"],
      "",
      "This is source/generate-only evidence. No target or token was run.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--sequence", type=int, default=590)
  parser.add_argument("--predecessor", type=Path,
                      default=ROOT / "output/seq589-fused-exact-linear-projection-component-probe-gate-20260710Tseq589Z/metrics.json")
  parser.add_argument("--routes", type=Path,
                      default=ACTIVE / "routes-ledger.json")
  parser.add_argument("--decode-source", type=Path,
                      default=ROOT / "tools/intel-qwen36-r2-gpu-decode-smoke.py")
  parser.add_argument(
      "--generate-dir", type=Path,
      default=ROOT / "output/seq590-fused-exact-linear-projection-decode-source-20260710Tseq590Z")
  parser.add_argument("--header-source", type=Path,
                      default=ROOT / "engine/include/intel_qwen36/gpu_q4x8_matvec.hpp")
  parser.add_argument("--runner-source", type=Path,
                      default=ROOT / "engine/src/gpu_q4x8_matvec.cpp")
  parser.add_argument("--opencl-source", type=Path,
                      default=ROOT / "engine/gpu/opencl/q4x8_matvec.cl")
  parser.add_argument(
      "--out-dir", type=Path,
      default=ROOT / "output/seq590-fused-exact-linear-projection-decode-source-gate-20260710Tseq590Z")
  parser.add_argument("--cxx", default="c++")
  parser.add_argument("--clang", default="clang")
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
