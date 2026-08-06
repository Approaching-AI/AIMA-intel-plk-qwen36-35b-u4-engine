#!/usr/bin/env python3
"""Gate the locked rowblock16 CPU-order-finalize component source."""

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
SCHEMA_VERSION = (
    "intel-qwen36-fused-exact-linear-projection-component-source-v0"
)
CURRENT_ROUTE = (
    "router_prompt_distribution_fused_exact_linear_projection_component_source_gate"
)
SELECTED_NEXT_ROUTE = (
    "router_prompt_distribution_fused_exact_linear_projection_component_target_compile_gate"
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


def _source_markers(opencl: str, header: str, runner: str,
                    decode: str) -> list[dict[str, Any]]:
  markers = {
      "locked_kernel_name_and_bpr16_guard_exist": (
          "__kernel void q4k_x8_matvec_rowblock16_cpuorder_finalize(" in opencl
          and "if (blocks_per_row != 16U)" in opencl
          and "const uint block_index = lid;" in opencl
      ),
      "parallel_stage_emits_exact_576_byte_integer_state": (
          "__local int local_lane_sums[16 * 8];" in opencl
          and "__local int local_grouped_min_sums[16];" in opencl
          and "for (int q4_group = 0; q4_group < 4; ++q4_group)" in opencl
          and "for (int q4_lane = 0; q4_lane < 32; ++q4_lane)" in opencl
          and "local_lane_sums[lid * 8U + (uint)lane] = lane_sums[lane];"
          in opencl
      ),
      "packed_q4_byte_mapping_and_integer_order_are_locked": (
          "128 + (q4_index >> 3) * 64 + (int)j * 8 + (q4_index & 7)"
          in opencl
          and "low_scale * ((int)q8[q8_base + q4_lane] *"
          in opencl
          and "high_scale * ((int)q8[q8_base + 32 + q4_lane] *"
          in opencl
      ),
      "one_barrier_then_cpu_order_float_finalize": (
          "#pragma OPENCL FP_CONTRACT OFF" in opencl
          and "barrier(CLK_LOCAL_MEM_FENCE);" in opencl
          and "if (lid == 0U)" in opencl
          and "for (uint ordered_block = 0U; ordered_block < 16U;"
          in opencl
          and "sums[lane] +=" in opencl
          and "d * (float)local_lane_sums[ordered_block * 8U + (uint)lane]"
          in opencl
          and "min_sum -=" in opencl
          and "sum += sums[lane];" in opencl
          and "#pragma OPENCL FP_CONTRACT ON" in opencl
      ),
      "same_runner_creates_and_releases_candidate_kernel": (
          "kernel_rowblock16_cpuorder_finalize_ =" in runner
          and "CreateNamedKernel(\"q4k_x8_matvec_rowblock16_cpuorder_finalize\")"
          in runner
          and "clReleaseKernel(kernel_rowblock16_cpuorder_finalize_)" in runner
      ),
      "component_api_reuses_rowblock16_dispatch_and_buffers": (
          "GpuQ4X8MatvecRun RunRowblock16CpuOrderFinalize(" in header
          and "GpuQ4X8MatvecRunner::RunRowblock16CpuOrderFinalize(" in runner
          and "RunRowblock16Kernel(" in runner
          and "row_groups, repeat, true);" in runner
          and "? kernel_rowblock16_cpuorder_finalize_" in runner
          and ": kernel_rowblock16_;" in runner
      ),
      "decode_runtime_is_not_wired_by_source_gate": (
          "RunRowblock16CpuOrderFinalize" not in decode
          and "IQ36_FUSED_EXACT_LINEAR_PROJECTION" not in decode
          and "IQ36_ROWBLOCK16_CPUORDER_FINALIZE" not in decode
      ),
  }
  return [{"name": name, "pass": passed}
          for name, passed in markers.items()]


def _run_compile(command: list[str], out_dir: Path,
                 stem: str) -> dict[str, Any]:
  completed = subprocess.run(
      command, cwd=ROOT, capture_output=True, text=True, check=False)
  stdout_path = out_dir / f"{stem}.stdout.txt"
  stderr_path = out_dir / f"{stem}.stderr.txt"
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
  command = ["python3", "tools/intel-qwen36-code-volume-check.py"]
  completed = subprocess.run(
      command, cwd=ROOT, capture_output=True, text=True, check=False)
  return {
      "passed": completed.returncode == 0,
      "returncode": completed.returncode,
      "stdout": completed.stdout.strip(),
      "stderr": completed.stderr.strip(),
  }


def compute(args: argparse.Namespace) -> dict[str, Any]:
  routes = _load(args.routes)
  predecessor = _load(args.predecessor)
  design = predecessor.get("design", {})
  opencl = args.opencl_source.read_text(encoding="utf-8")
  header = args.header_source.read_text(encoding="utf-8")
  runner = args.runner_source.read_text(encoding="utf-8")
  decode = args.decode_source.read_text(encoding="utf-8")
  compile_dir = args.out_dir / "compile"
  compile_dir.mkdir(parents=True, exist_ok=True)
  cpp_compile = _run_compile([
      args.cxx, "-std=c++20", "-Iengine/include", "-O0", "-c",
      _rel(args.runner_source), "-o", _rel(compile_dir / "gpu_q4x8_matvec.o"),
  ], compile_dir, "cpp-compile")
  opencl_compile = _run_compile([
      args.clang, "-x", "cl", "-target", "spir64", "-cl-std=CL1.2",
      "-fsyntax-only", _rel(args.opencl_source),
  ], compile_dir, "opencl-syntax")
  code_volume = _code_volume()
  markers = _source_markers(opencl, header, runner, decode)

  predecessor_selects = (
      predecessor.get("required_checks_passed") is True
      and predecessor.get("design_passed") is True
      and predecessor.get("component_source_allowed") is True
      and predecessor.get("component_probe_allowed") is False
      and predecessor.get("token_row_allowed") is False
      and predecessor.get("selected_next_route") == CURRENT_ROUTE
      and _has_candidate(routes, 586, CURRENT_ROUTE)
      and _has_switch(
          routes, 586,
          "select_router_prompt_distribution_fused_exact_linear_projection_"
          "component_source_gate"))
  design_matches = (
      design.get("name") == "rowblock16_cpuorder_finalize"
      and design.get("kernel_name")
      == "q4k_x8_matvec_rowblock16_cpuorder_finalize"
      and design.get("algorithm", {}).get("local_state", {}).get("bytes") == 576
      and design.get("dispatch_contract", {}).get(
          "additional_dispatch_count") == 0
      and design.get("dispatch_contract", {}).get(
          "host_readback_or_bridge") is False
      and design.get("component_acceptance", {}).get(
          "candidate_gpu_vs_cpu_max_abs_diff") == 0
      and design.get("component_acceptance", {}).get(
          "candidate_us_max") == 198.1958589939298)
  no_runtime_evidence = not any(
      (args.out_dir / name).exists()
      for name in ("probe.json", "smoke.json", "result.json", "tokens.jsonl"))
  checks = [
      {"name": "seq586_selected_component_source_only_gate",
       "pass": predecessor_selects},
      {"name": "source_matches_the_single_locked_design",
       "pass": design_matches},
      {"name": "kernel_runner_and_no_decode_markers_pass",
       "pass": all(row["pass"] for row in markers),
       "detail": markers},
      {"name": "runner_compiles_locally",
       "pass": cpp_compile["passed"], "detail": cpp_compile},
      {"name": "complete_opencl_program_passes_local_spir_syntax",
       "pass": opencl_compile["passed"], "detail": opencl_compile},
      {"name": "code_volume_ceiling_is_preserved",
       "pass": code_volume["passed"], "detail": code_volume},
      {"name": "source_gate_created_no_target_or_runtime_evidence",
       "pass": no_runtime_evidence},
  ]
  required = all(bool(row["pass"]) for row in checks)
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "sequence": args.sequence,
      "inputs": {
          "routes": _rel(args.routes),
          "predecessor": _rel(args.predecessor),
          "opencl_source": _rel(args.opencl_source),
          "opencl_source_sha256": _sha256(args.opencl_source),
          "header_source": _rel(args.header_source),
          "header_source_sha256": _sha256(args.header_source),
          "runner_source": _rel(args.runner_source),
          "runner_source_sha256": _sha256(args.runner_source),
          "decode_source": _rel(args.decode_source),
          "decode_source_sha256": _sha256(args.decode_source),
      },
      "design": design,
      "checks": checks,
      "required_checks_passed": required,
      "component_source_passed": required,
      "component_target_compile_allowed": required,
      "component_probe_allowed": False,
      "decode_integration_allowed": False,
      "token_row_allowed": False,
      "validation_or_test_allowed": False,
      "speedup_claims_allowed": False,
      "promotion_allowed": False,
      "disposition": (
          "accept_rowblock16_cpuorder_finalize_component_source"
          if required else "reject_or_repair_component_source"),
      "selected_next_route": (
          SELECTED_NEXT_ROUTE if required else CURRENT_ROUTE),
      "next_route_reason": (
          "The one locked kernel and component-only same-runner API match the "
          "design, preserve the decode wrapper and flag ceiling, and compile "
          "locally as C++ plus SPIR OpenCL syntax. Target-compile the component "
          "source without executing it; no component probe or token is yet "
          "authorized."
          if required else
          "Repair the locked kernel shape, runner API, local compilation, or "
          "source-only boundary before any target command."),
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
          "component_target_compile_allowed": metrics[
              "component_target_compile_allowed"],
          "component_probe_allowed": False,
          "decode_integration_allowed": False,
          "token_row_allowed": False,
          "speedup_claims_allowed": False,
      }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  failed = [row["name"] for row in metrics["checks"] if not row["pass"]]
  lines = [
      f"# Seq{metrics['sequence']} Fused Exact Projection Component Source",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- failed_checks: `{failed}`",
      "",
      metrics["next_route_reason"],
      "",
      "No target command, component probe, decode integration, or token was used.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--sequence", type=int, default=587)
  parser.add_argument("--routes", type=Path,
                      default=ACTIVE / "routes-ledger.json")
  parser.add_argument(
      "--predecessor", type=Path,
      default=ROOT / "output/seq586-fused-exact-linear-projection-budget-design-gate-20260710Tseq586Z/metrics.json")
  parser.add_argument("--opencl-source", type=Path,
                      default=ROOT / "engine/gpu/opencl/q4x8_matvec.cl")
  parser.add_argument("--header-source", type=Path,
                      default=ROOT / "engine/include/intel_qwen36/gpu_q4x8_matvec.hpp")
  parser.add_argument("--runner-source", type=Path,
                      default=ROOT / "engine/src/gpu_q4x8_matvec.cpp")
  parser.add_argument("--decode-source", type=Path,
                      default=ROOT / "tools/intel-qwen36-r2-gpu-decode-smoke.py")
  parser.add_argument("--cxx", default="c++")
  parser.add_argument("--clang", default="clang")
  parser.add_argument(
      "--out-dir", type=Path,
      default=ROOT / "output/seq587-fused-exact-linear-projection-component-source-gate-20260710Tseq587Z")
  args = parser.parse_args()
  metrics = compute(args)
  write_outputs(metrics, args.out_dir)
  print(json.dumps({
      "required_checks_passed": metrics["required_checks_passed"],
      "disposition": metrics["disposition"],
      "component_target_compile_allowed": metrics[
          "component_target_compile_allowed"],
      "selected_next_route": metrics["selected_next_route"],
      "out_dir": _rel(args.out_dir),
  }, sort_keys=True))
  return 0 if metrics["required_checks_passed"] else 2


if __name__ == "__main__":
  raise SystemExit(main())
