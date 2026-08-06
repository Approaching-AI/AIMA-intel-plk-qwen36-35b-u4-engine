#!/usr/bin/env python3
"""Source-gate the locked exact preprojection component APIs and kernels."""

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
    "intel-qwen36-all-linear-preprojection-parity-component-source-gate-v0"
)
CURRENT_ROUTE = (
    "router_prompt_distribution_all_linear_preprojection_parity_component_source_gate"
)
SELECTED_NEXT_ROUTE = (
    "router_prompt_distribution_all_linear_preprojection_parity_component_target_compile_gate"
)
CANDIDATE = "cpuorder_preprojection_bundle_v1"


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


def _run(command: list[str], out_dir: Path, stem: str) -> dict[str, Any]:
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


def _source_markers(opencl: str, header: str, runner: str,
                    decode: str) -> list[dict[str, Any]]:
  off = opencl.rfind("#pragma OPENCL FP_CONTRACT OFF")
  on = opencl.find("#pragma OPENCL FP_CONTRACT ON", off)
  markers = {
      "exact_bundle_is_scoped_between_contract_pragmas": (
          off >= 0 and on > off
          and off < opencl.find("q6k_linear_qkv_cpuorder_nofma", off) < on
          and off < opencl.find("linear_attn_conv_cpuorder_nofma_f32", off) < on
          and off < opencl.find("linear_attn_postconv_silu_split_cpuorder_f32", off) < on
          and off < opencl.find("linear_attn_postconv_qk_l2_cpuorder_f32", off) < on
          and off < opencl.find("linear_attn_delta_recurrent_final_cpuorder_nofma_f32", off) < on
      ),
      "q6_keeps_cpu_lane_and_float_accumulation_order": (
          "__kernel void q6k_linear_qkv_cpuorder_nofma(" in opencl
          and "int lane_sums[8];" in opencl
          and "const float block_lane = combined_scale * (float)lane_sums[lane];"
          in opencl
          and "sum = sum + block_lane;" in opencl
      ),
      "conv_preserves_separate_product_add_and_state_update": (
          "__kernel void linear_attn_conv_cpuorder_nofma_f32(" in opencl
          and "const float product =" in opencl
          and "sum = sum + product;" in opencl
          and "next_conv_state[state_base + history - 1U] = qkv_mixed[channel];"
          in opencl
      ),
      "postconv_matches_cpu_double_sigmoid_and_l2_shape": (
          "const double sigmoid_double =" in opencl
          and "__kernel void linear_attn_postconv_qk_l2_cpuorder_f32(" in opencl
          and "double sum = 0.0;" in opencl
          and "const float sum_f32 = (float)sum;" in opencl
          and "fmax(sqrt(sum_f32), norm_epsilon)" in opencl
      ),
      "delta_uses_precomputed_transcendentals_and_two_state_phases": (
          "__global const float* decay" in opencl
          and "__global const float* z_silu" in opencl
          and "state_out[state_base + col] = decayed;" in opencl
          and "const float updated = state_out[state_base + col] + update;"
          in opencl
          and "final_output[v_base + row] = weighted * z_silu[v_base + row];"
          in opencl
      ),
      "runner_creates_releases_and_owns_all_exact_kernels": all(
          marker in runner for marker in [
              "kernel_q6_linear_qkv_cpuorder_ =",
              "kernel_conv_cpuorder_ =",
              "kernel_postconv_silu_split_cpuorder_ =",
              "kernel_postconv_l2_qk_cpuorder_ =",
              "kernel_delta_recurrent_final_cpuorder_ =",
              "clReleaseKernel(kernel_q6_linear_qkv_cpuorder_)",
              "clReleaseKernel(kernel_conv_cpuorder_)",
              "clReleaseKernel(kernel_postconv_silu_split_cpuorder_)",
              "clReleaseKernel(kernel_postconv_l2_qk_cpuorder_)",
              "clReleaseKernel(kernel_delta_recurrent_final_cpuorder_)",
          ]),
      "public_component_apis_select_exact_variants_only": (
          "RunResidentRawQ6KThenResidentConvStateCpuOrder(" in header
          and "RunPostConvPrepThenLinearAttentionDeltaResidentStateCpuOrder(" in header
          and "GpuQ4X8MatvecRunner::RunResidentRawQ6KThenResidentConvStateCpuOrder(" in runner
          and "readback_qkv, readback_conv_output, true);" in runner
          and "RunPostConvPrepThenLinearAttentionDeltaResidentStateCpuOrder(" in runner
          and "readback_attention_output, readback_final_output, true);" in runner
      ),
      "host_precomputes_decay_and_z_silu_without_new_api_transfer": (
          "decay[i] = std::exp(gate[i]);" in runner
          and "z_silu[i] = z[i] * sigmoid;" in runner
          and "gate_kernel_input = &decay;" in runner
          and "z_kernel_input = &z_silu;" in runner
      ),
      "default_decode_has_no_exact_preprojection_selector_or_api_call": (
          "IQ36_CPUORDER_PREPROJECTION_BUNDLE" not in decode
          and "RunResidentRawQ6KThenResidentConvStateCpuOrder" not in decode
          and "RunPostConvPrepThenLinearAttentionDeltaResidentStateCpuOrder"
          not in decode
      ),
  }
  return [{"name": name, "pass": passed}
          for name, passed in markers.items()]


def _api_harness() -> str:
  return r'''#include "intel_qwen36/gpu_q4x8_matvec.hpp"

#include <cstdint>
#include <vector>

void CompileExactPreprojectionApi(
    iq36::GpuQ4X8MatvecRunner& runner,
    const iq36::GpuQ8KInputPlanes& q8,
    const std::vector<float>& values) {
  const auto preconv =
      runner.RunResidentRawQ6KThenResidentConvStateCpuOrder(
          1, q8, 2, 3, 4, 1, true, 5, true, true);
  const auto delta =
      runner.RunPostConvPrepThenLinearAttentionDeltaResidentStateCpuOrder(
          preconv.conv_output_handle, 6, values, values, values, values,
          128, 16, 16, 1.0e-6f, 1, true, true, true);
  (void)delta;
}
'''


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
  harness = compile_dir / "exact_preprojection_api_compile.cpp"
  harness.write_text(_api_harness(), encoding="utf-8")

  cpp_compile = _run([
      args.cxx, "-std=c++20", "-Iengine/include", "-O0", "-c",
      _rel(args.runner_source), "-o", _rel(compile_dir / "gpu_q4x8_matvec.o"),
  ], compile_dir, "runner-compile")
  api_compile = _run([
      args.cxx, "-std=c++20", "-Iengine/include", "-fsyntax-only",
      _rel(harness),
  ], compile_dir, "api-compile")
  opencl_compile = _run([
      args.clang, "-x", "cl", "-target", "spir64", "-cl-std=CL1.2",
      "-fsyntax-only", _rel(args.opencl_source),
  ], compile_dir, "opencl-syntax")
  code_volume = _run([
      "python3", "tools/intel-qwen36-code-volume-check.py",
  ], compile_dir, "code-volume")
  markers = _source_markers(opencl, header, runner, decode)

  predecessor_selects = (
      predecessor.get("required_checks_passed") is True
      and predecessor.get("design_passed") is True
      and predecessor.get("component_source_allowed") is True
      and predecessor.get("target_compile_allowed") is False
      and predecessor.get("selected_next_route") == CURRENT_ROUTE
      and _has_candidate(routes, 597, CURRENT_ROUTE)
      and _has_switch(
          routes, 597,
          "select_router_prompt_distribution_all_linear_preprojection_"
          "parity_component_source_gate"))
  design_matches = (
      design.get("candidate") == CANDIDATE
      and design.get("implementation", {}).get("dispatch_contract", {}).get(
          "new_dispatches_allowed") == 0
      and design.get("implementation", {}).get("dispatch_contract", {}).get(
          "new_host_readbacks_allowed") == 0
      and design.get("component_gate", {}).get(
          "whole_shell_added_us_per_layer_max") == 6.841858993929781
      and design.get("component_gate", {}).get(
          "whole_linear_attention_output_bit_exact") is True)
  no_runtime_evidence = not any(
      (args.out_dir / name).exists()
      for name in ("probe.json", "smoke.json", "raw-run.json", "tokens.jsonl"))
  checks = [
      {"name": "seq597_selected_component_source_only",
       "pass": predecessor_selects},
      {"name": "source_preserves_the_locked_candidate_and_whole_shell_ruler",
       "pass": design_matches},
      {"name": "exact_kernel_api_and_no_decode_markers_pass",
       "pass": all(row["pass"] for row in markers),
       "detail": markers},
      {"name": "runner_compiles_locally",
       "pass": cpp_compile["passed"], "detail": cpp_compile},
      {"name": "public_component_api_harness_compiles_locally",
       "pass": api_compile["passed"], "detail": api_compile},
      {"name": "complete_opencl_program_passes_spir12_syntax",
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
          "api_harness": _rel(harness),
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
          "accept_cpuorder_preprojection_bundle_v1_component_source"
          if required else "repair_exact_preprojection_component_source"),
      "selected_next_route": (
          SELECTED_NEXT_ROUTE if required else CURRENT_ROUTE),
      "next_route_reason": (
          "The locked replacement kernels and component-only APIs compile as "
          "complete C++ and OpenCL source, preserve default decode and the "
          "flag ceiling, and add no target/runtime evidence. Generate and "
          "target-compile the captured-layer component next without executing "
          "it."
          if required else
          "Repair exact kernel semantics, API wiring, local compilation, or "
          "source isolation before any target command."),
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
      f"# Seq{metrics['sequence']} Exact Preprojection Component Source",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- failed_checks: `{failed}`",
      "",
      metrics["next_route_reason"],
      "",
      "No target command, component execution, decode integration, or token ran.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--sequence", type=int, default=598)
  parser.add_argument("--routes", type=Path,
                      default=ACTIVE / "routes-ledger.json")
  parser.add_argument(
      "--predecessor", type=Path,
      default=ROOT / "output/seq597-all-linear-preprojection-parity-budget-design-gate-20260710Tseq597Z/metrics.json")
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
      default=ROOT / "output/seq598-all-linear-preprojection-parity-component-source-gate-20260710Tseq598Z")
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
