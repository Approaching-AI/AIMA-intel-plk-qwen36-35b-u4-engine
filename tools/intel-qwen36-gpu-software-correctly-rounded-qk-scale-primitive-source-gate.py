#!/usr/bin/env python3
"""Source-gate the one correctly-rounded Q/K scale Level Zero component."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
  sys.path.insert(0, str(TOOLS))

import iq36_local  # noqa: E402


SCHEMA_VERSION = (
    "intel-qwen36-gpu-software-correctly-rounded-qk-scale-source-v0")
CURRENT_ROUTE = (
    "gpu_software_correctly_rounded_qk_scale_primitive_source_gate")
SELECTED_NEXT_ROUTE = (
    "gpu_software_correctly_rounded_qk_scale_primitive_target_compile_gate")
DEFAULT_HOST = "local"
DEFAULT_ENV_SCRIPT = (
    "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh")
DEFAULT_REMOTE_ROOT = "/home/intel/intel-qwen36-gpu"

API_HARNESS = r'''
#include "intel_qwen36/gpu_level_zero_postconv_recurrent.hpp"

#include <type_traits>

static_assert(!std::is_copy_constructible_v<
              iq36::GpuLevelZeroPostconvRecurrentRunner>);
static_assert(std::is_move_constructible_v<
              iq36::GpuLevelZeroPostconvRecurrentRunner>);

int main() {
  iq36::GpuLevelZeroPostconvRecurrentInput input;
  iq36::GpuLevelZeroPostconvRecurrentRun run;
  return static_cast<int>(input.conv_output_raw.size() +
                          run.sample_wall_us.size());
}
'''


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


def _sha256_text(value: str) -> str:
  return hashlib.sha256(value.encode("utf-8")).hexdigest()


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


def _normalize_to_v1(module_source: str) -> str:
  helper_start = module_source.index(
      "inline float iq36_cr_recip_normal_f32_u64_v1(float input) {")
  kernel_start = module_source.index(
      "__attribute__((reqd_work_group_size(128, 1, 1)))", helper_start)
  normalized = module_source[:helper_start] + module_source[kernel_start:]
  new_scale = (
      "    head_scale[0] = iq36_cr_recip_normal_f32_u64_v1(\n"
      "        fmax(sqrt(sum_f32), norm_epsilon));")
  old_scale = (
      "    head_scale[0] = 1.0f / fmax(sqrt(sum_f32), norm_epsilon);")
  if normalized.count(new_scale) != 1:
    raise ValueError("v2 Q/K scale replacement marker is not unique")
  return normalized.replace(new_scale, old_scale)


def compute(args: argparse.Namespace) -> dict[str, Any]:
  routes = _load(args.routes)
  predecessor = _load(args.predecessor)
  feasibility = _load(args.feasibility)
  v1_source = _load(args.v1_source)
  header = args.header.read_text(encoding="utf-8")
  source = args.source.read_text(encoding="utf-8")
  module_source = args.module_source.read_text(encoding="utf-8")
  cmake = args.cmake.read_text(encoding="utf-8")

  args.out_dir.mkdir(parents=True, exist_ok=True)
  raw_dir = args.out_dir / "raw"
  compile_dir = args.out_dir / "compile"
  generated_dir = args.out_dir / "generated"
  raw_dir.mkdir(parents=True, exist_ok=True)
  compile_dir.mkdir(parents=True, exist_ok=True)
  generated_dir.mkdir(parents=True, exist_ok=True)
  api_harness = compile_dir / "api_harness.cpp"
  api_harness.write_text(API_HARNESS, encoding="utf-8")

  predecessor_selects = (
      predecessor.get("required_checks_passed") is True
      and predecessor.get("design_passed") is True
      and predecessor.get("source_gate_allowed") is True
      and predecessor.get("target_command_allowed") is False
      and predecessor.get("selected_next_route") == CURRENT_ROUTE
      and predecessor.get("design", {}).get("candidate")
      == "level_zero_ocloc_cr_recip_postconv_recurrent_v2"
      and _has_candidate(routes, 620, CURRENT_ROUTE)
      and _has_switch(
          routes, 620,
          "select_gpu_software_correctly_rounded_qk_scale_primitive_"
          "source_gate"))
  api_shape_ok = all(marker in header for marker in [
      "struct GpuLevelZeroPostconvRecurrentInput",
      "struct GpuLevelZeroPostconvRecurrentRun",
      "class GpuLevelZeroPostconvRecurrentRunner",
      "const GpuLevelZeroPostconvRecurrentRunner&) = delete;",
      "std::unique_ptr<Impl> impl_;",
  ])
  runtime_shape_ok = all(marker in source for marker in [
      "class GpuLevelZeroPostconvRecurrentRunner::Impl",
      "properties.deviceId != requested_device_id",
      "ZE_MODULE_FORMAT_NATIVE",
      'CreateKernel("iq36_l0_postconv_cpuorder")',
      'CreateKernel("iq36_l0_delta_recurrent_cpuorder")',
      "zeCommandListAppendLaunchKernel(",
      "zeCommandListAppendBarrier(",
      "zeCommandQueueSynchronize(queue_, UINT64_MAX)",
      "state_in, input.recurrent_state.data()",
  ]) and source.count("Check(zeCommandListAppendLaunchKernel(") == 2

  helper_start = module_source.find(
      "inline float iq36_cr_recip_normal_f32_u64_v1(float input) {")
  helper_end = module_source.find(
      "__attribute__((reqd_work_group_size(128, 1, 1)))", helper_start)
  helper = (
      module_source[helper_start:helper_end]
      if helper_start >= 0 and helper_end > helper_start else "")
  helper_shape_ok = all(marker in helper for marker in [
      "const ulong significand = 0x800000UL | (ulong)fraction;",
      "const ulong numerator = 1UL << 47U;",
      "quotient = numerator / significand;",
      "const ulong remainder = numerator % significand;",
      "twice_remainder == significand && (quotient & 1UL) != 0UL",
      "if (quotient == 0x1000000UL)",
      "return as_float(output_bits);",
  ]) and not any(marker in helper for marker in [
      "1.0f /", "1.0 /", "sqrt(", "rsqrt(", "native_", "half_",
  ])
  module_shape_ok = (
      module_source.count("__kernel void") == 2
      and module_source.count("iq36_cr_recip_normal_f32_u64_v1(") == 2
      and all(marker in module_source for marker in [
          "#pragma OPENCL EXTENSION cl_khr_fp64 : enable",
          "#pragma OPENCL FP_CONTRACT OFF",
          "iq36_l0_postconv_cpuorder",
          "iq36_l0_delta_recurrent_cpuorder",
          "reqd_work_group_size(128, 1, 1)",
          "const double exp_value = exp(",
          "sum = sum + (double)head_value * (double)head_value;",
          "head_scale[0] = iq36_cr_recip_normal_f32_u64_v1(",
          "sum_k = sum_k + product;",
          "state_out[state_base + col] = updated;",
          "sum_q = sum_q + product;",
          "sum_squares = sum_squares + square;",
      ])
      and "fma(" not in module_source)
  normalized_v1 = _normalize_to_v1(module_source)
  expected_v1_sha = v1_source.get("inputs", {}).get("module_source_sha256")
  diff_isolated = (
      isinstance(expected_v1_sha, str)
      and _sha256_text(normalized_v1) == expected_v1_sha
      and _sha256(args.header) == v1_source.get("inputs", {}).get(
          "header_sha256")
      and _sha256(args.source) == v1_source.get("inputs", {}).get(
          "source_sha256"))
  verifier_proof = (
      feasibility.get("required_checks_passed") is True
      and feasibility.get("verifier", {}).get("result", {}).get(
          "mantissas_checked") == 8388608
      and feasibility.get("verifier", {}).get("result", {}).get(
          "mismatch_count") == 0)
  runtime_independent = not any(
      marker.lower() in (header + source).lower()
      for marker in ["opencl", "cl_mem", "llama", "openvino", "ocloc", "gguf"])
  not_in_default_build = "gpu_level_zero_postconv_recurrent.cpp" not in cmake

  api_compile = _run([
      args.cxx, "-std=c++17", "-Wall", "-Wextra", "-Wpedantic",
      "-Iengine/include", "-fsyntax-only", _rel(api_harness),
  ], compile_dir, "api-harness-syntax")
  module_syntax = _run([
      args.clang, "-x", "cl", "-cl-std=CL2.0", "-target", "spir64",
      "-Xclang", "-finclude-default-header", "-Dcl_khr_fp64",
      "-fsyntax-only", _rel(args.module_source),
  ], compile_dir, "module-opencl-syntax")
  code_volume = _run([
      "python3", "tools/intel-qwen36-code-volume-check.py",
  ], compile_dir, "code-volume")

  remote_dir = (
      f"{args.remote_root.rstrip('/')}/seq{args.sequence}-cr-recip-source")
  setup = iq36_local.run_target(
      args.host,
      " && ".join([
          "rm -rf " + shlex.quote(remote_dir),
          "mkdir -p " + shlex.quote(remote_dir + "/include/intel_qwen36"),
          "mkdir -p " + shlex.quote(remote_dir + "/src"),
          "mkdir -p " + shlex.quote(remote_dir + "/module"),
          "mkdir -p " + shlex.quote(remote_dir + "/generated"),
      ]),
      args.timeout_s)
  transfers = {}
  if setup.get("returncode") == 0:
    transfers = {
        "header": iq36_local.copy_to(
            args.host, args.header,
            remote_dir + "/include/intel_qwen36/"
            "gpu_level_zero_postconv_recurrent.hpp", args.timeout_s),
        "source": iq36_local.copy_to(
            args.host, args.source,
            remote_dir + "/src/gpu_level_zero_postconv_recurrent.cpp",
            args.timeout_s),
        "module": iq36_local.copy_to(
            args.host, args.module_source,
            remote_dir + "/module/iq36_postconv_recurrent.cl", args.timeout_s),
    }
  transfer_ok = (
      len(transfers) == 3
      and all(row.get("returncode") == 0 for row in transfers.values()))
  native_module = remote_dir + "/generated/iq36_postconv_recurrent.bin"
  module_compile_command = " ".join([
      "ocloc", "compile",
      "-file", shlex.quote(remote_dir + "/module/iq36_postconv_recurrent.cl"),
      "-device", "0xb080",
      "-options", shlex.quote("-cl-std=CL2.0"),
      "-output", "iq36_postconv_recurrent",
      "-out_dir", shlex.quote(remote_dir + "/generated"),
      "-output_no_suffix", "--format", "zebin", "-q",
  ])
  module_compile = (
      iq36_local.run_target(args.host, module_compile_command, args.timeout_s)
      if transfer_ok else {})
  module_audit = (
      iq36_local.run_target(
          args.host,
          " && ".join([
              f"test -s {shlex.quote(native_module)}",
              f"sha256sum {shlex.quote(native_module)}",
              f"ocloc validate -file {shlex.quote(native_module)}",
          ]),
          args.timeout_s)
      if module_compile.get("returncode") == 0 else {})
  audit_lines = str(module_audit.get("stdout", "")).splitlines()
  module_sha = audit_lines[0].split(maxsplit=1)[0] if audit_lines else None
  source_syntax_command = " && ".join([
      f"source {shlex.quote(args.env_script)} >/dev/null 2>&1",
      (
          "g++ -std=c++17 -O0 -Wall -Wextra -Wpedantic -fsyntax-only "
          f"-I{shlex.quote(remote_dir + '/include')} "
          f"{shlex.quote(remote_dir + '/src/gpu_level_zero_postconv_recurrent.cpp')}")
  ])
  source_syntax = (
      iq36_local.run_target(
          args.host, f"bash -lc {shlex.quote(source_syntax_command)}",
          args.timeout_s)
      if transfer_ok else {})
  fetched_module = generated_dir / "iq36_postconv_recurrent.bin"
  fetch = (
      iq36_local.copy_from(
          args.host, native_module, fetched_module, args.timeout_s)
      if module_audit.get("returncode") == 0 else {})
  fetched_sha = _sha256(fetched_module) if fetched_module.exists() else None
  cleanup = iq36_local.run_target(
      args.host, "rm -rf " + shlex.quote(remote_dir), args.timeout_s)
  iq36_local.write_json(raw_dir / "setup.json", setup)
  iq36_local.write_json(raw_dir / "transfers.json", transfers)
  iq36_local.write_json(raw_dir / "module-compile.json", module_compile)
  iq36_local.write_json(raw_dir / "module-audit.json", module_audit)
  iq36_local.write_json(raw_dir / "source-syntax.json", source_syntax)
  iq36_local.write_json(raw_dir / "fetch.json", fetch)
  iq36_local.write_json(raw_dir / "cleanup.json", cleanup)
  module_ok = (
      module_compile.get("returncode") == 0
      and module_audit.get("returncode") == 0
      and "Binary is VALID" in str(module_audit.get("stdout", ""))
      and "Kernel #0 named iq36_l0_postconv_cpuorder" in str(
          module_audit.get("stdout", ""))
      and "Kernel #1 named iq36_l0_delta_recurrent_cpuorder" in str(
          module_audit.get("stdout", ""))
      and fetch.get("returncode") == 0
      and isinstance(module_sha, str) and len(module_sha) == 64
      and fetched_sha == module_sha)

  checks = [
      {"name": "seq620_selected_one_cr_recip_source_candidate",
       "pass": predecessor_selects},
      {"name": "public_api_and_runtime_are_unchanged_from_seq615",
       "pass": api_shape_ok and runtime_shape_ok and diff_isolated
       and api_compile["passed"],
       "detail": {
           "normalized_v1_sha256": _sha256_text(normalized_v1),
           "expected_v1_sha256": expected_v1_sha,
           "api_compile": api_compile,
       }},
      {"name": "helper_is_exact_integer_quotient_remainder_ties_even_only",
       "pass": helper_shape_ok and verifier_proof},
      {"name": "module_keeps_two_locked_kernels_and_changes_only_qk_reciprocal",
       "pass": module_shape_ok and diff_isolated},
      {"name": "module_passes_local_opencl_c20_syntax",
       "pass": module_syntax["passed"], "detail": module_syntax},
      {"name": "runtime_has_no_opencl_llama_openvino_ocloc_or_gguf_dependency",
       "pass": runtime_independent},
      {"name": "component_is_not_wired_into_default_engine_build",
       "pass": not_in_default_build},
      {"name": "offline_exact_device_v2_zebin_compiles_validates_and_has_both_kernels",
       "pass": module_ok,
       "detail": {"sha256": module_sha, "fetched_sha256": fetched_sha}},
      {"name": "complete_runtime_source_passes_level_zero_header_syntax",
       "pass": source_syntax.get("returncode") == 0,
       "detail": source_syntax},
      {"name": "code_volume_ceiling_is_preserved",
       "pass": code_volume["passed"], "detail": code_volume},
      {"name": "source_gate_executed_no_kernel_model_component_or_token",
       "pass": (
           "/home/intel/models" not in module_compile_command
           and "iq36-level-zero" not in source_syntax_command)},
      {"name": "remote_source_staging_cleaned",
       "pass": cleanup.get("returncode") == 0},
  ]
  required = all(bool(row["pass"]) for row in checks)
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "sequence": args.sequence,
      "inputs": {
          "routes": _rel(args.routes),
          "predecessor": _rel(args.predecessor),
          "feasibility": _rel(args.feasibility),
          "v1_source": _rel(args.v1_source),
          "header": _rel(args.header),
          "header_sha256": _sha256(args.header),
          "source": _rel(args.source),
          "source_sha256": _sha256(args.source),
          "module_source": _rel(args.module_source),
          "module_source_sha256": _sha256(args.module_source),
          "module_source_normalized_v1_sha256": _sha256_text(normalized_v1),
          "cmake": _rel(args.cmake),
          "host": args.host,
          "env_script": args.env_script,
          "ocloc_device": "0xb080",
          "ocloc_options": "-cl-std=CL2.0 --format zebin",
      },
      "primitive": predecessor.get("design", {}).get("primitive"),
      "native_module": {
          "path": _rel(fetched_module),
          "sha256": fetched_sha,
      },
      "checks": checks,
      "required_checks_passed": required,
      "component_source_passed": required,
      "target_compile_allowed": required,
      "component_probe_allowed": False,
      "decode_integration_allowed": False,
      "token_row_allowed": False,
      "validation_or_test_allowed": False,
      "speedup_claims_allowed": False,
      "promotion_allowed": False,
      "disposition": (
          "accept_level_zero_ocloc_cr_recip_postconv_recurrent_v2_source"
          if required else
          "repair_level_zero_ocloc_cr_recip_postconv_recurrent_v2_source"),
      "selected_next_route": (
          SELECTED_NEXT_ROUTE if required else CURRENT_ROUTE),
      "next_route_reason": (
          "Target-compile one captured component harness against this exact "
          "runtime source and v2 native-module identity without execution or "
          "model access."
          if required else
          "Repair reciprocal isolation/proof, API/runtime identity, module "
          "syntax/zebin, or hygiene before any target binary or kernel creation."),
  }


def write_outputs(metrics: dict[str, Any], out_dir: Path) -> None:
  (out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  (out_dir / "manifest.json").write_text(
      json.dumps({
          "schema_version": metrics["schema_version"],
          "workstream": metrics["workstream"],
          "tool": _rel(Path(__file__)),
          "inputs": metrics["inputs"],
          "primitive": metrics["primitive"],
          "native_module": metrics["native_module"],
          "component_source_passed": metrics["component_source_passed"],
          "selected_next_route": metrics["selected_next_route"],
          "component_probe_allowed": False,
          "token_row_allowed": False,
      }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  failed = [row["name"] for row in metrics["checks"] if not row["pass"]]
  lines = [
      f"# Seq{metrics['sequence']} Correctly-Rounded Q/K Scale Source",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- native module SHA256: `{metrics['native_module']['sha256']}`",
      f"- failed_checks: `{failed}`",
      "",
      metrics["next_route_reason"],
      "",
      "Only compiler, validator, syntax, and local exhaustive-proof checks ran. No kernel, model, component, or token executed.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--sequence", type=int, default=621)
  parser.add_argument("--routes", type=Path,
                      default=ACTIVE / "routes-ledger.json")
  parser.add_argument(
      "--predecessor", type=Path,
      default=ROOT / (
          "output/seq620-gpu-software-correctly-rounded-qk-scale-primitive-"
          "design-gate-20260710Tseq620Z/metrics.json"))
  parser.add_argument(
      "--feasibility", type=Path,
      default=ROOT / (
          "output/seq619-gpu-software-correctly-rounded-qk-scale-primitive-"
          "feasibility-gate-20260710Tseq619Z/metrics.json"))
  parser.add_argument(
      "--v1-source", type=Path,
      default=ROOT / (
          "output/seq615-gpu-level-zero-postconv-recurrent-component-source-"
          "gate-20260710Tseq615Z/metrics.json"))
  parser.add_argument(
      "--header", type=Path,
      default=ROOT / (
          "engine/include/intel_qwen36/gpu_level_zero_postconv_recurrent.hpp"))
  parser.add_argument(
      "--source", type=Path,
      default=ROOT / "engine/src/gpu_level_zero_postconv_recurrent.cpp")
  parser.add_argument(
      "--module-source", type=Path,
      default=ROOT / "engine/gpu/level_zero/iq36_postconv_recurrent.cl")
  parser.add_argument("--cmake", type=Path,
                      default=ROOT / "engine/CMakeLists.txt")
  parser.add_argument("--target", "--host", dest="host", metavar="TARGET", default=DEFAULT_HOST)
  parser.add_argument("--env-script", default=DEFAULT_ENV_SCRIPT)
  parser.add_argument("--staging-root", "--remote-root", dest="remote_root", metavar="PATH", default=DEFAULT_REMOTE_ROOT)
  parser.add_argument("--timeout-s", type=int, default=600)
  parser.add_argument("--cxx", default="clang++")
  parser.add_argument("--clang", default="clang")
  parser.add_argument(
      "--out-dir", type=Path,
      default=ROOT / (
          "output/seq621-gpu-software-correctly-rounded-qk-scale-primitive-"
          "source-gate-20260710Tseq621Z"))
  args = parser.parse_args()
  metrics = compute(args)
  write_outputs(metrics, args.out_dir)
  print(json.dumps({
      "required_checks_passed": metrics["required_checks_passed"],
      "component_source_passed": metrics["component_source_passed"],
      "target_compile_allowed": metrics["target_compile_allowed"],
      "disposition": metrics["disposition"],
      "selected_next_route": metrics["selected_next_route"],
      "out_dir": _rel(args.out_dir),
  }, sort_keys=True))
  return 0 if metrics["required_checks_passed"] else 2


if __name__ == "__main__":
  raise SystemExit(main())
