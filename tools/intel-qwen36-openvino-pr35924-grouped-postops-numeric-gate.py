#!/usr/bin/env python3
"""Build and run a bounded numeric probe for PR35924 grouped post-ops.

The probe compares the fused grouped-matmul post-op output against the same
grouped matmul stored to F16 followed by the stock OpenVINO SwiGLU expression.
It preserves product K/N, U4 group64, F16, and zero-point semantics while
bounding experts and rows.  No model, InferRequest, or product worker is
created.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-openvino-pr35924-grouped-postops-numeric-gate-v0"
BASE_TOOL = ROOT / "tools/intel-qwen36-onednn-gmlp-exact-component-build.py"
SOURCE = ROOT / "engine/tools/onednn_grouped_postops_numeric_probe.cpp"
R0 = Path("/home/intel/intel-qwen36-r0")
ONEDNN = (
    R0 / "source/openvino-90214e5be05/"
    "src/plugins/intel_gpu/thirdparty/onednn_gpu")
ONEDNN_HEAD = "20db47e2d3c4df1b66e93bed2e97d30da175512d"
ENV_SCRIPT = Path(
    "/home/intel/intel-box-run/current/tools/"
    "intel/activate-intel-box-env.sh")
CMAKE = Path("/home/intel/intel-box-env/conda/bin/cmake")
CXX = Path("/home/intel/intel-box-env/conda/bin/g++")
OPENCL_INCLUDE = Path("/home/intel/intel-box-env/conda/include")
OPENCL_LIBRARY = Path("/home/intel/intel-box-env/conda/lib/libOpenCL.so")
POSTOPS_PATHS = (
    "src/gpu/intel/compute/kernel_ctx.hpp",
    "src/gpu/intel/matmul/grouped_micro_gemm.cl",
    "src/gpu/intel/matmul/grouped_micro_gemm.cpp",
    "src/gpu/intel/matmul/grouped_micro_gemm.hpp",
    "src/gpu/intel/matmul/grouped_post_ops_gen.cpp",
    "src/gpu/intel/matmul/grouped_post_ops_gen.hpp",
)


def load_base() -> Any:
  spec = importlib.util.spec_from_file_location(
      "iq36_scoped_build_base", BASE_TOOL)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import scoped build helper: {BASE_TOOL}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


BASE = load_base()


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", required=True, type=Path)
  parser.add_argument("--configure-timeout-s", default=600.0, type=float)
  parser.add_argument("--build-timeout-s", default=3600.0, type=float)
  parser.add_argument("--compile-timeout-s", default=300.0, type=float)
  parser.add_argument("--run-timeout-s", default=600.0, type=float)
  parser.add_argument("--poll-interval-s", default=0.1, type=float)
  parser.add_argument(
      "--reuse-build", action="store_true",
      help="reuse the source-hash-bound build after auditing its cache")
  parser.add_argument(
      "--incremental-rebuild", action="store_true",
      help="rerun configure and dnnl build in an existing audited build tree")
  parser.add_argument(
      "--build-dir", type=Path,
      help="explicit build tree; requires --reuse-build or "
           "--incremental-rebuild")
  parser.add_argument(
      "--expect-scalar-activation", action="store_true",
      help="require scalar-lane fused Swish generation")
  parser.add_argument(
      "--expect-identity-activation", action="store_true",
      help="require the boundary-only identity diagnostic variant")
  parser.add_argument(
      "--expect-volatile-scalar-activation", action="store_true",
      help="require scalar work-item Swish temporaries that inhibit vector "
           "recombination")
  parser.add_argument(
      "--expect-materialized-f16-activation", action="store_true",
      help="require activation to consume a volatile F16 boundary value")
  parser.add_argument(
      "--expect-standard-exp-activation", action="store_true",
      help="require standard exp for the residual activation parity case")
  parser.add_argument(
      "--expect-polynomial-exp-activation", action="store_true",
      help="require the deterministic range-reduced exp polynomial")
  parser.add_argument(
      "--expect-one-ulp-away-from-zero", action="store_true",
      help="require the residual float midpoint parity correction")
  parser.add_argument(
      "--expect-stock-syntax-activation", action="store_true",
      help="require materialized F16 input with the stock SwiGLU in-place "
           "division form")
  parser.add_argument(
      "--expect-volatile-inplace-activation", action="store_true",
      help="require a volatile materialized F16 input and in-place SwiGLU "
           "division")
  parser.add_argument(
      "--expect-finite-f16-midpoint-correction", action="store_true",
      help="require the finite-F16-census-derived stock SwiGLU midpoint "
           "correction")
  parser.add_argument(
      "--expect-stock-division-options", action="store_true",
      help="require grouped Swish to use stock OpenVINO division options")
  parser.add_argument(
      "--product-shape", action="store_true",
      help="run the locked 256-expert, 16,384-routed-row prefill shape")
  args = parser.parse_args()
  if min(
      args.configure_timeout_s, args.build_timeout_s,
      args.compile_timeout_s, args.run_timeout_s,
      args.poll_interval_s) <= 0:
    parser.error("timeouts and poll interval must be positive")
  if args.reuse_build and args.incremental_rebuild:
    parser.error("--reuse-build and --incremental-rebuild are exclusive")
  if args.build_dir is not None and not (
      args.reuse_build or args.incremental_rebuild):
    parser.error(
        "--build-dir requires --reuse-build or --incremental-rebuild")
  return args


def run(
    command: list[str], cwd: Path = ROOT,
) -> subprocess.CompletedProcess[str]:
  return subprocess.run(
      command, cwd=cwd, check=False, capture_output=True, text=True,
      encoding="utf-8", errors="replace")


def git(cwd: Path, *parts: str) -> str:
  result = run(["git", *parts], cwd=cwd)
  if result.returncode != 0:
    raise RuntimeError(
        f"git failed ({result.returncode}): {parts}\n{result.stderr}")
  return result.stdout.strip()


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    while chunk := stream.read(1024 * 1024):
      digest.update(chunk)
  return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True) + "\n",
      encoding="utf-8")


def check(name: str, passed: bool, **evidence: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **evidence}


def source_snapshot() -> dict[str, Any]:
  files = {}
  aggregate = hashlib.sha256()
  for relative in POSTOPS_PATHS:
    path = ONEDNN / relative
    record = {
        "path": str(path),
        "exists": path.is_file(),
        "sha256": sha256(path) if path.is_file() else None,
        "bytes": path.stat().st_size if path.is_file() else 0,
    }
    files[relative] = record
    aggregate.update(relative.encode("utf-8"))
    aggregate.update((record["sha256"] or "").encode("ascii"))
  generator = (
      ONEDNN / "src/gpu/intel/matmul/grouped_post_ops_gen.cpp")
  grouped_micro = (
      ONEDNN / "src/gpu/intel/matmul/grouped_micro_gemm.cpp")
  kernel_ctx = ONEDNN / "src/gpu/intel/compute/kernel_ctx.hpp"
  generator_text = (
      generator.read_text(encoding="utf-8", errors="replace")
      if generator.is_file() else "")
  grouped_micro_text = (
      grouped_micro.read_text(encoding="utf-8", errors="replace")
      if grouped_micro.is_file() else "")
  kernel_ctx_text = (
      kernel_ctx.read_text(encoding="utf-8", errors="replace")
      if kernel_ctx.is_file() else "")
  return {
      "head": git(ONEDNN, "rev-parse", "HEAD"),
      "target_status": git(
          ONEDNN, "status", "--short", "--untracked-files=all",
          "--", *POSTOPS_PATHS).splitlines(),
      "files": files,
      "aggregate_sha256": aggregate.hexdigest(),
      "postops_contract_present": (
          "generate_post_ops_microgemm_header" in generator_text
          and "po_kind_t::binary_grouped_scale" in generator_text
          and (
              "tile_elementwise((*c_tile)" in generator_text
              or "tile_elementwise_s((*c_tile)" in generator_text)
          and "tile_binary((*c_tile)" in generator_text),
      "openvino_stock_parity_variant_present": (
          "post_ops_f16_tile_type" in generator_text
          and (
              "native_exp" in generator_text
              or "1.0f + exp(-(" in generator_text
              or "iq36_grouped_postops_exp_polynomial" in generator_text)),
      "scalar_activation_variant_present": (
          "tile_elementwise_s((*c_tile)" in generator_text
          or "volatile float eltwise_gate_" in generator_text),
      "identity_activation_diagnostic_present": (
          '"#define eltwise_apply_" << i << "(v) (v)\\n"' in generator_text),
      "volatile_scalar_activation_variant_present": (
          "volatile float eltwise_gate_" in generator_text
          and "volatile float eltwise_value_" in generator_text),
      "materialized_f16_activation_variant_present": (
          (
              "volatile half eltwise_gate_f16_" in generator_text
              or '"            half eltwise_gate_f16_"' in generator_text)
          and "convert_float(eltwise_gate_f16_" in generator_text),
      "standard_exp_activation_variant_present": (
          "1.0f + exp(-(" in generator_text),
      "polynomial_exp_activation_variant_present": (
          "iq36_grouped_postops_exp_polynomial" in generator_text
          and "return ldexp(polynomial, exponent)" in generator_text),
      "one_ulp_away_from_zero_variant_present": (
          "nextafter(eltwise_value_" in generator_text
          and "copysign(INFINITY, eltwise_value_" in generator_text),
      "stock_syntax_activation_variant_present": (
          '"            float eltwise_value_"' in generator_text
          and '" /= (1.0f + native_exp(-("' in generator_text),
      "volatile_inplace_activation_variant_present": (
          '"            volatile float eltwise_value_"' in generator_text
          and '" /= (1.0f + native_exp(-("' in generator_text),
      "finite_f16_midpoint_correction_present": (
          "0xc173" in generator_text
          and "nextafter(eltwise_value_" in generator_text
          and "-INFINITY" in generator_text),
      "stock_division_options_variant_present": (
          "remove_option(const char *option)" in kernel_ctx_text
          and "find_po_in_chain(po_chain_, po_kind_t::eltwise)" in
          grouped_micro_text
          and "remove_option(" in grouped_micro_text
          and "-cl-fp32-correctly-rounded-divide-sqrt" in
          grouped_micro_text),
  }


def cache_has(cache: str, name: str, value: str) -> bool:
  return re.search(
      rf"^{re.escape(name)}:(?:BOOL|STRING|INTERNAL)="
      rf"{re.escape(value)}$",
      cache, flags=re.MULTILINE) is not None


def successful_stage(stage: dict[str, Any]) -> bool:
  return (
      stage.get("returncode") == 0
      and stage.get("timed_out") is False
      and stage.get("memory_guard_tripped") is False
      and stage.get("oom_observed") is False)


def parse_probe(stdout_path: Path) -> dict[str, Any]:
  if not stdout_path.is_file():
    return {}
  for line in reversed(
      stdout_path.read_text(
          encoding="utf-8", errors="replace").splitlines()):
    try:
      value = json.loads(line)
    except json.JSONDecodeError:
      continue
    if isinstance(value, dict):
      return value
  return {}


def classify(probe: dict[str, Any]) -> str:
  scenarios = probe.get("scenarios", [])
  if not isinstance(scenarios, list) or not scenarios:
    return "invalid_probe"
  activation_census = probe.get("exhaustive_finite_f16_activation", {})
  census_mismatches = activation_census.get(
      "swish_vs_stock_oracle", {}).get("mismatch_count")
  full = [
      row.get("swish_binary_vs_stock_pipeline", {}).get("mismatch_count")
      for row in scenarios if isinstance(row, dict)]
  swish = [
      row.get("swish_vs_stored_gate_oracle", {}).get("mismatch_count")
      for row in scenarios if isinstance(row, dict)]
  swish_f32 = [
      row.get("swish_f32_vs_stock_f32", {}).get("mismatch_count")
      for row in scenarios if isinstance(row, dict)]
  swish_f32_correctly_rounded = [
      row.get("diagnostic_swish_f32_vs_correctly_rounded", {}).get(
          "mismatch_count")
      for row in scenarios if isinstance(row, dict)]
  stock_f32_correctly_rounded = [
      row.get("diagnostic_stock_f32_vs_correctly_rounded", {}).get(
          "mismatch_count")
      for row in scenarios if isinstance(row, dict)]
  binary = [
      row.get("binary_vs_stored_gate_oracle", {}).get("mismatch_count")
      for row in scenarios if isinstance(row, dict)]
  deterministic = [
      row.get("swish_binary_repeat_determinism", {}).get("mismatch_count")
      for row in scenarios if isinstance(row, dict)]
  producer_full = [
      row.get("producer_fed_swiglu_vs_stock_pipeline", {}).get(
          "mismatch_count")
      for row in scenarios if isinstance(row, dict)]
  producer_deterministic = [
      row.get("producer_fed_swiglu_repeat_determinism", {}).get(
          "mismatch_count")
      for row in scenarios if isinstance(row, dict)]
  snapshot_transport = [
      row.get("producer_snapshot_transport", {}).get("mismatch_count")
      for row in scenarios if isinstance(row, dict)]
  snapshot_full = [
      row.get("snapshot_fed_swiglu_vs_stock_pipeline", {}).get(
          "mismatch_count")
      for row in scenarios if isinstance(row, dict)]
  producer_grf256 = [
      row.get("diagnostic_producer_fed_vs_grf256_oracle", {}).get(
          "mismatch_count")
      for row in scenarios if isinstance(row, dict)]
  stock_producer_grf256 = [
      row.get("diagnostic_stock_vs_grf256_producer_full", {}).get(
          "mismatch_count")
      for row in scenarios if isinstance(row, dict)]
  identity_swish = [
      row.get("diagnostic_swish_output_vs_stored_gate", {}).get(
          "mismatch_count")
      for row in scenarios if isinstance(row, dict)]
  identity_full = [
      row.get("diagnostic_full_output_vs_stock_binary", {}).get(
          "mismatch_count")
      for row in scenarios if isinstance(row, dict)]
  grf256_swish = [
      row.get("diagnostic_swish_vs_grf256_oracle", {}).get(
          "mismatch_count")
      for row in scenarios if isinstance(row, dict)]
  grf256_full = [
      row.get("diagnostic_full_vs_grf256_oracle", {}).get(
          "mismatch_count")
      for row in scenarios if isinstance(row, dict)]
  exp_swish = [
      row.get("diagnostic_swish_vs_exp_oracle", {}).get(
          "mismatch_count")
      for row in scenarios if isinstance(row, dict)]
  exp_full = [
      row.get("diagnostic_full_vs_exp_oracle", {}).get(
          "mismatch_count")
      for row in scenarios if isinstance(row, dict)]
  polynomial_swish = [
      row.get("diagnostic_fused_vs_polynomial_swish", {}).get(
          "mismatch_count")
      for row in scenarios if isinstance(row, dict)]
  polynomial_full = [
      row.get("diagnostic_fused_vs_polynomial_full", {}).get(
          "mismatch_count")
      for row in scenarios if isinstance(row, dict)]
  if any(value not in (0, None) for value in deterministic):
    return "nondeterministic_or_dependency_ordering_failure"
  if any(value not in (0, None) for value in producer_deterministic):
    return "producer_fed_dependency_ordering_is_nondeterministic"
  if any(value not in (0, None) for value in snapshot_transport):
    return "producer_snapshot_transport_mismatch"
  if any(isinstance(value, int) and value > 0 for value in producer_full):
    if snapshot_full and all(value == 0 for value in snapshot_full):
      return "producer_fed_memory_provenance_or_dependency_mismatch"
    if any(isinstance(value, int) and value > 0 for value in snapshot_full):
      if any(isinstance(value, int) and value > 0 for value in swish_f32):
        if (
            swish_f32_correctly_rounded
            and all(
                value == 0 for value in swish_f32_correctly_rounded)
            and any(
                isinstance(value, int) and value > 0
                for value in stock_f32_correctly_rounded)):
          return "correctly_rounded_divide_option_semantics_mismatch"
        return "fused_f32_activation_semantics_mismatch"
      if (
          producer_grf256 and all(value == 0 for value in producer_grf256)
          and any(
              isinstance(value, int) and value > 0
              for value in stock_producer_grf256)):
        return "producer_fed_256grf_native_exp_semantics_mismatch"
      return "producer_fed_arbitrary_binary_exposes_f32_activation_mismatch"
    return "producer_fed_grouped_swiglu_pipeline_mismatch"
  if full and all(value == 0 for value in full):
    return "micro_postops_match_stored_stock_pipeline"
  if (
      grf256_swish and all(value == 0 for value in grf256_swish)
      and grf256_full and all(value == 0 for value in grf256_full)):
    return "onednn_256grf_native_exp_semantics_identified"
  if (
      exp_swish and all(value == 0 for value in exp_swish)
      and exp_full and all(value == 0 for value in exp_full)):
    return "fused_activation_matches_stock_standard_exp"
  if (
      polynomial_swish and all(value == 0 for value in polynomial_swish)
      and polynomial_full and all(value == 0 for value in polynomial_full)):
    return "fused_activation_matches_deterministic_polynomial"
  if isinstance(census_mismatches, int) and census_mismatches > 0:
    return "exhaustive_f16_activation_semantics_residual"
  if (
      identity_swish and all(value == 0 for value in identity_swish)
      and identity_full and all(value == 0 for value in identity_full)):
    return "f16_boundary_and_binary_postop_exact_activation_only"
  if swish and all(value == 0 for value in swish):
    if binary and any(isinstance(value, int) and value > 0 for value in binary):
      return "binary_boundary_or_grouped_load_isolated"
    return "combined_postops_interaction_isolated"
  if swish and any(isinstance(value, int) and value > 0 for value in swish):
    return "swish_boundary_or_activation_semantics_isolated"
  return "numeric_mismatch_not_yet_isolated"


def main() -> int:
  args = parse_args()
  probe_experts = 256 if args.product_shape else 16
  probe_rows = 16384 if args.product_shape else 64
  probe_values = probe_rows * 512
  census_pages = (63488 + probe_values - 1) // probe_values
  output = args.output.resolve()
  raw = output / "raw"
  raw.mkdir(parents=True, exist_ok=False)
  required = (
      BASE_TOOL, SOURCE, ONEDNN, ENV_SCRIPT, CMAKE, CXX,
      OPENCL_INCLUDE, OPENCL_LIBRARY)
  missing = [str(path) for path in required if not path.exists()]
  if missing:
    raise SystemExit("missing numeric-gate inputs: " + ", ".join(missing))

  repo = BASE.repository_state(output)
  source = source_snapshot()
  build_contract = "shared-ocl-all-isa-all-grouped-v1"
  source_key = hashlib.sha256(
      (source["aggregate_sha256"] + ":" + build_contract).encode(
          "ascii")).hexdigest()[:12]
  build_dir = (
      args.build_dir.resolve() if args.build_dir is not None
      else R0 / f"build/onednn-20db-pr35924-postops-{source_key}")
  binary = raw / "iq36-grouped-postops-numeric-probe"
  initial_build_exists = build_dir.exists()
  environment = os.environ.copy()
  conda_bin = "/home/intel/intel-box-env/conda/bin"
  conda_lib = "/home/intel/intel-box-env/conda/lib"
  environment["PATH"] = conda_bin + ":" + environment.get("PATH", "")
  environment["LD_LIBRARY_PATH"] = (
      conda_lib + ":" + environment.get("LD_LIBRARY_PATH", ""))

  configure_command = [
      str(CMAKE), "-S", str(ONEDNN), "-B", str(build_dir), "-GNinja",
      "-DCMAKE_BUILD_TYPE=Release",
      "-DDNNL_BUILD_TESTS=OFF",
      "-DDNNL_BUILD_EXAMPLES=OFF",
      "-DDNNL_CPU_RUNTIME=NONE",
      "-DDNNL_GPU_RUNTIME=OCL",
      "-DDNNL_GPU_VENDOR=INTEL",
      "-DDNNL_LIBRARY_TYPE=SHARED",
      "-DDNNL_ENABLE_WORKLOAD=INFERENCE",
      "-DDNNL_ENABLE_PRIMITIVE=ALL",
      "-DDNNL_ENABLE_PRIMITIVE_GPU_ISA=ALL",
      "-DDNNL_EXPERIMENTAL_GROUPED_MEMORY=ON",
      "-DONEDNN_BUILD_GRAPH=OFF",
      f"-DOpenCL_INCLUDE_DIR={OPENCL_INCLUDE}",
      f"-DOpenCL_LIBRARY={OPENCL_LIBRARY}",
  ]
  build_command = [
      str(CMAKE), "--build", str(build_dir),
      "--target", "dnnl", "-j", "1"]
  configure = {
      "stage": "configure", "returncode": None, "skipped": True,
      "reason": "existing audited build requested"
      if initial_build_exists and args.reuse_build
      else "existing audited build will be incrementally rebuilt"
      if initial_build_exists and args.incremental_rebuild
      else "blocked because source-hash build already exists"
      if initial_build_exists else None,
  }
  build = {
      "stage": "build", "returncode": None, "skipped": True,
      "reason": "not reached",
  }
  if not initial_build_exists or args.incremental_rebuild:
    configure = BASE.run_scoped(
        output, raw, "configure", configure_command,
        args.configure_timeout_s, args.poll_interval_s, environment)
    configure["skipped"] = False
    if successful_stage(configure):
      build = BASE.run_scoped(
          output, raw, "build", build_command,
          args.build_timeout_s, args.poll_interval_s, environment)
      build["skipped"] = False
  elif args.reuse_build:
    configure["returncode"] = 0
    configure["timed_out"] = False
    configure["memory_guard_tripped"] = False
    configure["oom_observed"] = False
    configure["monitor"] = {
        "system_available_min_bytes":
            BASE.proc_meminfo().get("MemAvailable", 0)}
    build["returncode"] = 0
    build["timed_out"] = False
    build["memory_guard_tripped"] = False
    build["oom_observed"] = False
    build["reason"] = "source-hash-bound shared library reused"
    build["monitor"] = {
        "system_available_min_bytes":
            BASE.proc_meminfo().get("MemAvailable", 0)}

  cache_path = build_dir / "CMakeCache.txt"
  cache = (
      cache_path.read_text(encoding="utf-8", errors="replace")
      if cache_path.is_file() else "")
  library = build_dir / "src/libdnnl.so"
  compile_command = [
      str(CXX), "-std=gnu++17", "-O2", "-DNDEBUG",
      "-DCL_TARGET_OPENCL_VERSION=300",
      f"-DIQ36_PROBE_EXPERTS={probe_experts}",
      f"-DIQ36_PROBE_ROWS={probe_rows}",
      "-I", str(build_dir / "include"), "-I", str(ONEDNN / "include"),
      str(SOURCE), "-L", str(build_dir / "src"),
      f"-Wl,-rpath,{build_dir / 'src'}", "-ldnnl", "-lOpenCL",
      "-o", str(binary),
  ]
  compile_stage = {
      "stage": "compile", "returncode": None, "skipped": True,
      "reason": "oneDNN build unavailable",
  }
  if successful_stage(build) and library.is_file():
    compile_stage = BASE.run_scoped(
        output, raw, "compile", compile_command,
        args.compile_timeout_s, args.poll_interval_s, environment)
    compile_stage["skipped"] = False

  run_shell = (
      f"source {shlex.quote(str(ENV_SCRIPT))} >/dev/null 2>&1 && "
      "export INTEL_FORCE_PROBE=b080 "
      "DNNL_VERBOSE=0 DNNL_PRIMITIVE_CACHE_CAPACITY=0 && "
      f"exec {shlex.quote(str(binary))}")
  run_command = ["bash", "-lc", run_shell]
  run_stage = {
      "stage": "probe", "returncode": None, "skipped": True,
      "reason": "component compile unavailable",
  }
  if successful_stage(compile_stage) and binary.is_file():
    run_stage = BASE.run_scoped(
        output, raw, "probe", run_command,
        args.run_timeout_s, args.poll_interval_s, environment)
    run_stage["skipped"] = False

  probe = parse_probe(raw / "probe.stdout")
  diagnosis = classify(probe)
  providers = probe.get("providers", {})
  scenarios = probe.get("scenarios", [])
  activation_census = probe.get("exhaustive_finite_f16_activation", {})
  expected_cache = (
      ("CMAKE_BUILD_TYPE", "Release"),
      ("DNNL_BUILD_TESTS", "OFF"),
      ("DNNL_CPU_RUNTIME", "NONE"),
      ("DNNL_GPU_RUNTIME", "OCL"),
      ("DNNL_GPU_VENDOR", "INTEL"),
      ("DNNL_LIBRARY_TYPE", "SHARED"),
      ("DNNL_ENABLE_WORKLOAD", "INFERENCE"),
      ("DNNL_ENABLE_PRIMITIVE", "ALL"),
      ("DNNL_ENABLE_PRIMITIVE_GPU_ISA", "ALL"),
      ("DNNL_EXPERIMENTAL_GROUPED_MEMORY", "ON"),
      ("ONEDNN_BUILD_GRAPH", "OFF"),
      ("CMAKE_HOME_DIRECTORY", str(ONEDNN)),
  )
  stages = (configure, build, compile_stage, run_stage)
  stage_oom_free = all(
      stage.get("oom_observed") is False
      and stage.get("memory_guard_tripped") is False
      for stage in stages)
  checks = [
      check(
          "repository_clean_and_pushed_at_gate",
          repo["branch"] == "main" and repo["pushed"] and not repo["dirty"],
          **repo),
      check(
          "locked_patched_onednn_source_bound",
          source["head"] == ONEDNN_HEAD
          and source["postops_contract_present"]
          and (
              source["openvino_stock_parity_variant_present"]
              or source["identity_activation_diagnostic_present"])
          and (
              not args.expect_scalar_activation
              or source["scalar_activation_variant_present"])
          and (
              not args.expect_identity_activation
              or source["identity_activation_diagnostic_present"])
          and (
              not args.expect_volatile_scalar_activation
              or source["volatile_scalar_activation_variant_present"])
          and (
              not args.expect_materialized_f16_activation
              or source["materialized_f16_activation_variant_present"])
          and (
              not args.expect_standard_exp_activation
              or source["standard_exp_activation_variant_present"])
          and (
              not args.expect_polynomial_exp_activation
              or source["polynomial_exp_activation_variant_present"])
          and (
              not args.expect_one_ulp_away_from_zero
              or source["one_ulp_away_from_zero_variant_present"])
          and (
              not args.expect_stock_syntax_activation
              or source["stock_syntax_activation_variant_present"])
          and (
              not args.expect_volatile_inplace_activation
              or source["volatile_inplace_activation_variant_present"])
          and (
              not args.expect_finite_f16_midpoint_correction
              or source["finite_f16_midpoint_correction_present"])
          and (
              not args.expect_stock_division_options
              or source["stock_division_options_variant_present"]),
          source=source),
      check(
          "component_build_is_fresh_or_explicitly_audited",
          not initial_build_exists
          or args.reuse_build
          or args.incremental_rebuild,
          build_dir=str(build_dir),
          initial_build_exists=initial_build_exists,
          reuse_requested=args.reuse_build,
          incremental_rebuild_requested=args.incremental_rebuild,
          explicit_build_dir=(
              str(args.build_dir.resolve())
              if args.build_dir is not None else None)),
      check(
          "shared_component_build_cache_exact",
          cache_path.is_file()
          and all(cache_has(cache, name, value)
                  for name, value in expected_cache),
          cache_path=str(cache_path),
          expected=dict(expected_cache)),
      check(
          "configure_build_compile_and_probe_succeeded",
          all(successful_stage(stage) for stage in stages),
          stages=list(stages)),
      check(
          "all_variants_select_grouped_micro_provider",
          isinstance(providers, dict)
          and set(providers) == {
              "base", "binary", "swish", "swish_binary",
              "swish_f32",
              "producer_up", "producer_swish_binary",
              "snapshot_swish_binary"}
          and all("grouped" in str(value).lower()
                  and "micro" in str(value).lower()
                  for value in providers.values()),
          providers=providers),
      check(
          "three_product_shape_offset_scenarios_completed",
          isinstance(scenarios, list) and len(scenarios) == 3
          and probe.get("shape", {}).get("experts") == probe_experts
          and probe.get("shape", {}).get("rows") == probe_rows
          and {row.get("name") for row in scenarios
               if isinstance(row, dict)} == {
                   "dense_four_rows_per_expert",
                   "sparse_four_zero_gap",
                   "sparse_skewed_offsets"},
          scenario_count=len(scenarios) if isinstance(scenarios, list) else 0),
      check(
          "stock_grouped_prefill_swiglu_oracle_io_is_exact",
          probe.get("oracle")
          == "openvino_grouped_prefill_swiglu_subgroup32_block_io"
          and isinstance(scenarios, list) and len(scenarios) == 3
          and all(
              row.get("stock_oracle_identity_io", {}).get(
                  "mismatch_count") == 0
              for row in scenarios if isinstance(row, dict)),
          oracle=probe.get("oracle")),
      check(
          "grf256_activation_semantics_diagnostic_completed",
          isinstance(scenarios, list) and len(scenarios) == 3
          and all(
              row.get("diagnostic_swish_vs_grf256_oracle", {}).get(
                  "value_count") == probe_values
              and row.get("diagnostic_full_vs_grf256_oracle", {}).get(
                  "value_count") == probe_values
              and row.get(
                  "diagnostic_producer_fed_vs_grf256_oracle", {}).get(
                      "value_count") == probe_values
              and row.get(
                  "diagnostic_stock_vs_grf256_producer_full", {}).get(
                      "value_count") == probe_values
              for row in scenarios if isinstance(row, dict))),
      check(
          "standard_exp_activation_semantics_diagnostic_completed",
          isinstance(scenarios, list) and len(scenarios) == 3
          and all(
              row.get("diagnostic_swish_vs_exp_oracle", {}).get(
                  "value_count") == probe_values
              and row.get("diagnostic_full_vs_exp_oracle", {}).get(
                  "value_count") == probe_values
              and (
                  row.get("swish_vs_stored_gate_oracle", {}).get(
                      "mismatch_count") == 0
                  or isinstance(row.get("activation_witness"), dict))
          for row in scenarios if isinstance(row, dict))),
      check(
          "f32_activation_semantics_diagnostic_completed",
          isinstance(scenarios, list) and len(scenarios) == 3
          and all(
              row.get("swish_f32_vs_stock_f32", {}).get(
                  "value_count") == probe_values
              and row.get(
                  "diagnostic_swish_f32_vs_correctly_rounded", {}).get(
                      "value_count") == probe_values
              and row.get(
                  "diagnostic_stock_f32_vs_correctly_rounded", {}).get(
                      "value_count") == probe_values
              for row in scenarios if isinstance(row, dict))
          and "grouped" in str(
              activation_census.get(
                  "swish_f32_provider", "")).lower()
          and "micro" in str(
              activation_census.get(
                  "swish_f32_provider", "")).lower()
          and activation_census.get(
              "swish_f32_vs_stock_f32", {}).get(
                  "value_count") == 63488
          and activation_census.get(
              "swish_f32_vs_correctly_rounded", {}).get(
                  "value_count") == 63488
          and activation_census.get(
              "stock_f32_vs_correctly_rounded", {}).get(
                  "value_count") == 63488,
          activation_census=activation_census),
      check(
          "deterministic_polynomial_semantics_diagnostic_completed",
          isinstance(scenarios, list) and len(scenarios) == 3
          and all(
              row.get("diagnostic_stock_vs_polynomial_swish", {}).get(
                  "value_count") == probe_values
              and row.get("diagnostic_stock_vs_polynomial_full", {}).get(
                  "value_count") == probe_values
              and row.get("diagnostic_fused_vs_polynomial_swish", {}).get(
                  "value_count") == probe_values
              and row.get("diagnostic_fused_vs_polynomial_full", {}).get(
                  "value_count") == probe_values
              for row in scenarios if isinstance(row, dict))),
      check(
          "finite_f16_activation_census_completed",
          isinstance(activation_census, dict)
          and activation_census.get("input_value_count") == 63488
          and activation_census.get("page_count") == census_pages
          and "grouped" in str(
              activation_census.get("base_provider", "")).lower()
          and "micro" in str(
              activation_census.get("base_provider", "")).lower()
          and "grouped" in str(
              activation_census.get("swish_provider", "")).lower()
          and "micro" in str(
              activation_census.get("swish_provider", "")).lower()
          and activation_census.get(
              "identity_transport", {}).get("value_count") == 63488
          and activation_census.get(
              "swish_vs_stock_oracle", {}).get(
                  "value_count") == 63488,
          activation_census=activation_census),
      check(
          "fused_execution_is_repeat_deterministic",
          isinstance(scenarios, list) and len(scenarios) == 3
          and all(
              row.get("swish_binary_repeat_determinism", {}).get(
                  "mismatch_count") == 0
              for row in scenarios if isinstance(row, dict)),
          diagnosis=diagnosis),
      check(
          "producer_fed_grouped_swiglu_pipeline_completed",
          isinstance(scenarios, list) and len(scenarios) == 3
          and all(
              row.get("producer_fed_swiglu_vs_stock_pipeline", {}).get(
                  "value_count") == probe_values
              and row.get(
                  "producer_fed_swiglu_repeat_determinism", {}).get(
                      "mismatch_count") == 0
              for row in scenarios if isinstance(row, dict)),
          diagnosis=diagnosis),
      check(
          "producer_snapshot_isolates_memory_provenance_from_values",
          isinstance(scenarios, list) and len(scenarios) == 3
          and all(
              row.get("producer_snapshot_transport", {}).get(
                  "mismatch_count") == 0
              and row.get(
                  "snapshot_fed_swiglu_vs_stock_pipeline", {}).get(
                      "value_count") == probe_values
              for row in scenarios if isinstance(row, dict)),
          diagnosis=diagnosis),
      check(
          "memory_policy_held_without_oom",
          stage_oom_free
          and all(
              int(stage.get("monitor", {}).get(
                  "system_available_min_bytes", BASE.ABORT_BYTES))
              >= BASE.ABORT_BYTES
              for stage in stages),
          preflight_bytes=BASE.PREFLIGHT_BYTES,
          abort_bytes=BASE.ABORT_BYTES),
      check(
          "bounded_probe_only_no_model_or_infer_request",
          True, model_workers_started=0, infer_requests_created=0,
          component_experts=probe_experts, component_rows=probe_rows,
          product_shape=args.product_shape,
          product_k=2048, product_n=512,
          full_model_layers_executed=0),
  ]
  required_checks_passed = all(row["pass"] for row in checks)
  result = {
      "schema_version": SCHEMA,
      "workstream": WS,
      "created_at": datetime.now(timezone.utc).strftime(
          "%Y-%m-%dT%H:%M:%SZ"),
      "git": repo,
      "source": source,
      "build": {
          "contract": build_contract,
          "directory": str(build_dir),
          "library": {
              "path": str(library),
              "exists": library.is_file(),
              "sha256": sha256(library) if library.is_file() else None,
          },
          "configure": configure,
          "build": build,
          "compile": compile_stage,
      },
      "probe": probe,
      "diagnosis": diagnosis,
      "checks": checks,
      "required_checks_passed": required_checks_passed,
      "verdict": {
          "component_evidence_valid": required_checks_passed,
          "next_direction": diagnosis,
          "full_product_candidate_admitted": False,
          "speedup_claims_allowed": False,
      },
  }
  write_json(output / "result.json", result)
  manifest = {
      "schema_version": SCHEMA + "-manifest",
      "result_sha256": sha256(output / "result.json"),
      "source_sha256": sha256(SOURCE),
      "tool_sha256": sha256(Path(__file__)),
  }
  write_json(output / "manifest.json", manifest)
  print(json.dumps({
      "artifact": str(output),
      "required_checks_passed": required_checks_passed,
      "diagnosis": diagnosis,
      "result_sha256": manifest["result_sha256"],
  }, sort_keys=True))
  return 0 if required_checks_passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
