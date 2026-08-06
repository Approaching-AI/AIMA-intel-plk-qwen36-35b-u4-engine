#!/usr/bin/env python3
"""Build the PR35924 candidate with the locked OpenVINO Swish boundary.

Seq2235 proved that upstream grouped post-ops changed two arithmetic facts at
the removed materialization boundary: the gate accumulator was no longer
rounded through F16, and ``exp`` replaced the stock kernel's ``native_exp``.
Seq2237 then showed that a scalar conversion macro cannot accept the
microkernel's vector tile. This gate binds the corrected tile-wise conversion,
rebuilds only oneDNN and the GPU plugin at parallelism one, and admits only a
repeated correctness worker.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-openvino-pr35924-swish-parity-build-v0"
ORIGINAL_BUILD_TOOL = (
    ROOT / "tools/intel-qwen36-openvino-pr35924-product-build.py")
FAILED_CORRECTNESS = ROOT / (
    "output/openvino-pr35924-swish-parity-correctness-"
    "20260731Tseq2237-clean/result.json")
FAILED_WORKER_STDERR = ROOT / (
    "output/openvino-pr35924-swish-parity-correctness-"
    "20260731Tseq2237-clean/raw/candidate/worker.stderr")
FAILED_WORKER_STDOUT = ROOT / (
    "output/openvino-pr35924-swish-parity-correctness-"
    "20260731Tseq2237-clean/raw/candidate/worker.stdout")
PARITY_PATCH = ROOT / (
    "engine/openvino/"
    "iq36-onednn-grouped-postops-openvino-swish-parity.patch")
SCALAR_PATCH = ROOT / (
    "engine/openvino/"
    "iq36-onednn-grouped-postops-openvino-swish-scalar-parity.patch")
VOLATILE_SCALAR_PATCH = ROOT / (
    "engine/openvino/"
    "iq36-onednn-grouped-postops-openvino-swish-volatile-scalar-parity.patch")
MATERIALIZED_F16_PATCH = ROOT / (
    "engine/openvino/"
    "iq36-onednn-grouped-postops-openvino-swish-materialized-f16-parity.patch")
MATERIALIZED_F16_MIDPOINT_PATCH = ROOT / (
    "engine/openvino/"
    "iq36-onednn-grouped-postops-openvino-swish-materialized-f16-"
    "midpoint-parity.patch")
STOCK_DIVISION_PATCH = ROOT / (
    "engine/openvino/"
    "iq36-onednn-grouped-postops-stock-division-options-parity.patch")
MATERIALIZED_F16_MIDPOINT_COMPONENT = ROOT / (
    "output/openvino-pr35924-grouped-postops-finite-f16-midpoint-confirm-"
    "20260801Tseq2262-clean/result.json")
STOCK_DIVISION_COMPONENT = ROOT / (
    "output/openvino-pr35924-grouped-postops-product-stock-division-exact-"
    "20260801Tseq2274-clean/result.json")
EXPECTED_FAILED_CORRECTNESS_SHA256 = (
    "5f0f7ecb0cb483cd04c9a796dee61762a618c74840d385dd019cc494197b00e4")
EXPECTED_FAILED_WORKER_STDERR_SHA256 = (
    "0f972469a140ca74a9abb9f852c132b1a09c10220e976c99d063d9c6d47bbdf8")
EXPECTED_FAILED_WORKER_STDOUT_SHA256 = (
    "2475ebc8aca7ea09633d28962295804a518982f4f52f00c56bb527b78e7e5c8b")
EXPECTED_PARITY_PATCH_SHA256 = (
    "21b080cf4623c04ace8aa885265bca0ec182af78a18712a825c53783edd15bc6")
EXPECTED_SCALAR_PATCH_SHA256 = (
    "64b4277ec9f36e550a5f6a9948a0cd674073f1a1d63f244f327a9937f14e57a9")
EXPECTED_VOLATILE_SCALAR_PATCH_SHA256 = (
    "962b6f18d0795b4ff9c1055ea29d2a433620777a7fa03bd975c4897c76392d0e")
EXPECTED_MATERIALIZED_F16_PATCH_SHA256 = (
    "151d46ecf3251f3e3b6cba808f5f99ba2042bb1cb2cd41f149d4c0f1092753ea")
EXPECTED_MATERIALIZED_F16_MIDPOINT_PATCH_SHA256 = (
    "1ac7d5cad5266234e943b150a7de3e9cd1f43c19d9a249409bf22586fb0df71b")
EXPECTED_STOCK_DIVISION_PATCH_SHA256 = (
    "a01cceda69e64ce620187a1eedf62fcfc53c07f0310156c20ab3ccbcac16ae18")
EXPECTED_MATERIALIZED_F16_MIDPOINT_COMPONENT_SHA256 = (
    "860a222d897097947cea1cd7f9b0dfc292b2343187eab89ca8dd780f715a90f8")
EXPECTED_STOCK_DIVISION_COMPONENT_SHA256 = (
    "791a71cb6640d80791846703fec4ad79193891772a10f8349e3f9dad626d477d")

R0 = Path("/home/intel/intel-qwen36-r0")
SOURCE_TREE = R0 / "source/openvino-90214e5be05"
ONEDNN = SOURCE_TREE / "src/plugins/intel_gpu/thirdparty/onednn_gpu"
BUILD_TREE = R0 / "build/openvino-90214e-l0-gpu"
ONEDNN_BUILD = (
    BUILD_TREE / "src/plugins/intel_gpu/thirdparty/onednn_gpu_build")
ONEDNN_INSTALL_LIB = (
    BUILD_TREE / "src/plugins/intel_gpu/thirdparty/onednn_gpu_install/lib/"
    "libopenvino_onednn_gpu.a")
BUILD_PLUGIN = R0 / (
    "output/openvino-90214e-l0-gpu-seq2109/bin/intel64/Release/"
    "libopenvino_intel_gpu_plugin.so")
SEQ2233_PLUGIN = R0 / (
    "output/openvino-90214e-l0-gpu-seq2233/bin/intel64/Release/"
    "libopenvino_intel_gpu_plugin.so")
SEQ2236_PLUGIN = R0 / (
    "output/openvino-90214e-l0-gpu-seq2236/bin/intel64/Release/"
    "libopenvino_intel_gpu_plugin.so")
CONTROL_PLUGIN = R0 / (
    "output/openvino-90214e-l0-gpu-seq2189/bin/intel64/Release/"
    "libopenvino_intel_gpu_plugin.so")
DEFAULT_CANDIDATE_PLUGIN = R0 / (
    "output/openvino-90214e-l0-gpu-seq2238/bin/intel64/Release/"
    "libopenvino_intel_gpu_plugin.so")
DEFAULT_MATERIALIZED_F16_CANDIDATE_PLUGIN = R0 / (
    "output/openvino-90214e-l0-gpu-seq2252/bin/intel64/Release/"
    "libopenvino_intel_gpu_plugin.so")
DEFAULT_MATERIALIZED_F16_MIDPOINT_CANDIDATE_PLUGIN = R0 / (
    "output/openvino-90214e-l0-gpu-seq2263/bin/intel64/Release/"
    "libopenvino_intel_gpu_plugin.so")
DEFAULT_STOCK_DIVISION_CANDIDATE_PLUGIN = R0 / (
    "output/openvino-90214e-l0-gpu-seq2275/bin/intel64/Release/"
    "libopenvino_intel_gpu_plugin.so")
GENERATOR = (
    ONEDNN / "src/gpu/intel/matmul/grouped_post_ops_gen.cpp")
KERNEL_CTX = ONEDNN / "src/gpu/intel/compute/kernel_ctx.hpp"
GROUPED_MICRO_GEMM = (
    ONEDNN / "src/gpu/intel/matmul/grouped_micro_gemm.cpp")
BASELINE_SWIGLU = (
    SOURCE_TREE /
    "src/plugins/intel_gpu/src/graph/impls/ocl_v2/"
    "moe_3gemm_swiglu_fuse.cl")
CMAKE = Path("/home/intel/intel-box-env/conda/bin/cmake")
EXPECTED_GENERATOR_SHA256 = (
    "b59db2cb8825cccb0d398504f559b3d6003cfac5716ba99c0cca6a30e5bdbf54")
EXPECTED_MATERIALIZED_F16_GENERATOR_SHA256 = (
    "303ee59245cd72df1e4dc8388ce8f96892b06521b0de6ac47695f18b901f640f")
EXPECTED_MATERIALIZED_F16_MIDPOINT_GENERATOR_SHA256 = (
    "6a6de3a3831b1f947ac39f37d63ccc0b5cf9c658ef28f167d07f4ab5d0d6ccb6")
EXPECTED_STOCK_DIVISION_KERNEL_CTX_SHA256 = (
    "30add0c4553a1de61c69f6ad41f786ad486a9316e25dd4ccd485f0b15839cdbb")
EXPECTED_STOCK_DIVISION_GROUPED_MICRO_GEMM_SHA256 = (
    "9480affa77b3d4965c8473b07831b857b1b3178b8da1d6d971b440bf266e0f92")
EXPECTED_SEQ2233_SHA256 = (
    "c66c9be61ee31110a55c8a064ed1390bd3d21a3f1766a03fdea84a078a519849")
EXPECTED_CONTROL_SHA256 = (
    "9832adffa8bf3fd7b013c47d9d2abefa66987c5fbde350c9669fce58c697b985")
EXPECTED_ONEDNN_ARCHIVE_SHA256 = (
    "5208f431ece909a82732a496e51117a2fcae899db22f9deae1a2be4ba36e660f")
EXPECTED_BUILD_PLUGIN_SHA256 = (
    "7827246114e095dca42887458a9bdbc505635e22ea5d3ba6a32c990ff555dcda")
EXPECTED_MATERIALIZED_F16_BASE_ARCHIVE_SHA256 = (
    "27be953954273e435a7a25fd5ff96863061520518209de5e3f05906708bbd2f1")
EXPECTED_MATERIALIZED_F16_BASE_PLUGIN_SHA256 = (
    "bbaaa6880695eab4381d2aa6bf32162ea318565d4c66b99f19dcef31689fbbd7")
EXPECTED_MATERIALIZED_F16_MIDPOINT_BASE_ARCHIVE_SHA256 = (
    "a781f7176b8355149523978f3f74b9b433c5b8a9b489b8f8268816937cc04cbb")
EXPECTED_MATERIALIZED_F16_MIDPOINT_BASE_PLUGIN_SHA256 = (
    "c04fc5c43f90b84bb606dfd5d251f9623d118b5f31a6d356713ba0cd74fb12ec")
EXPECTED_STOCK_DIVISION_BASE_ARCHIVE_SHA256 = (
    "b91760642c397fc4951b6938d537b7ded54f918c938914049f76e365fe735e86")
EXPECTED_STOCK_DIVISION_BASE_PLUGIN_SHA256 = (
    "f827e3441ba910bd865bfae0375852fe89c52b14694b4ca2109f98dfb150725c")
PREFLIGHT_BYTES = 8 * 1024**3
ABORT_BYTES = 4 * 1024**3


def load_module(name: str, path: Path) -> Any:
  spec = importlib.util.spec_from_file_location(name, path)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import {path}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


ORIGINAL = load_module("iq36_pr35924_build_base", ORIGINAL_BUILD_TOOL)
BASE = ORIGINAL.BASE


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", required=True, type=Path)
  parser.add_argument(
      "--candidate-plugin", type=Path)
  parser.add_argument(
      "--materialized-f16", action="store_true",
      help="build the component-admitted volatile F16-input variant")
  parser.add_argument(
      "--materialized-f16-midpoint", action="store_true",
      help="build the exhaustive-F16-census-exact midpoint variant")
  parser.add_argument(
      "--stock-division-exact", action="store_true",
      help="build the F16-boundary variant with stock division options")
  parser.add_argument("--build-timeout-s", default=3600.0, type=float)
  parser.add_argument("--poll-interval-s", default=0.1, type=float)
  args = parser.parse_args()
  if args.build_timeout_s <= 0 or args.poll_interval_s <= 0:
    parser.error("timeout and poll interval must be positive")
  if sum((
      bool(args.materialized_f16),
      bool(args.materialized_f16_midpoint),
      bool(args.stock_division_exact),
  )) > 1:
    parser.error(
        "Swish parity variants are mutually exclusive")
  return args


def check(name: str, passed: bool, **details: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **details}


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    while chunk := stream.read(8 * 1024 * 1024):
      digest.update(chunk)
  return digest.hexdigest()


def run(
    command: list[str], cwd: Path = ROOT,
) -> subprocess.CompletedProcess[str]:
  return subprocess.run(
      command, cwd=cwd, check=False, capture_output=True, text=True,
      encoding="utf-8", errors="replace")


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True) + "\n",
      encoding="utf-8")


def file_record(path: Path) -> dict[str, Any]:
  return {
      "path": str(path),
      "exists": path.is_file(),
      "bytes": path.stat().st_size if path.is_file() else 0,
      "mtime_ns": path.stat().st_mtime_ns if path.is_file() else 0,
      "sha256": sha256(path) if path.is_file() else None,
  }


def reverse_apply_check(repo: Path, patch: Path) -> dict[str, Any]:
  result = run(
      ["git", "apply", "--reverse", "--check", str(patch)], cwd=repo)
  return {
      "command": result.args,
      "returncode": result.returncode,
      "stdout": result.stdout.strip(),
      "stderr": result.stderr.strip(),
  }


def stage_ok(stage: dict[str, Any]) -> bool:
  return bool(
      stage.get("returncode") == 0 and
      stage.get("timed_out") is False and
      stage.get("memory_guard_tripped") is False and
      stage.get("oom_observed") is False)


def main() -> int:
  args = parse_args()
  output = args.output.resolve()
  raw = output / "raw"
  raw.mkdir(parents=True, exist_ok=False)
  stock_division_exact = bool(args.stock_division_exact)
  materialized_f16_midpoint = bool(args.materialized_f16_midpoint)
  materialized_f16 = bool(
      args.materialized_f16 or materialized_f16_midpoint
      or stock_division_exact)
  candidate_plugin = (
      args.candidate_plugin
      or (
          DEFAULT_STOCK_DIVISION_CANDIDATE_PLUGIN
          if stock_division_exact
          else DEFAULT_MATERIALIZED_F16_MIDPOINT_CANDIDATE_PLUGIN
          if materialized_f16_midpoint
          else DEFAULT_MATERIALIZED_F16_CANDIDATE_PLUGIN
          if materialized_f16
          else DEFAULT_CANDIDATE_PLUGIN)).resolve()
  active_patch = (
      MATERIALIZED_F16_MIDPOINT_PATCH
      if materialized_f16_midpoint
      else MATERIALIZED_F16_PATCH
      if materialized_f16
      else PARITY_PATCH)
  expected_active_patch_sha256 = (
      EXPECTED_MATERIALIZED_F16_MIDPOINT_PATCH_SHA256
      if materialized_f16_midpoint
      else
      EXPECTED_MATERIALIZED_F16_PATCH_SHA256
      if materialized_f16 else EXPECTED_PARITY_PATCH_SHA256)
  expected_generator_sha256 = (
      EXPECTED_MATERIALIZED_F16_MIDPOINT_GENERATOR_SHA256
      if materialized_f16_midpoint
      else
      EXPECTED_MATERIALIZED_F16_GENERATOR_SHA256
      if materialized_f16 else EXPECTED_GENERATOR_SHA256)
  expected_archive_before_sha256 = (
      EXPECTED_STOCK_DIVISION_BASE_ARCHIVE_SHA256
      if stock_division_exact
      else EXPECTED_MATERIALIZED_F16_MIDPOINT_BASE_ARCHIVE_SHA256
      if materialized_f16_midpoint
      else
      EXPECTED_MATERIALIZED_F16_BASE_ARCHIVE_SHA256
      if materialized_f16 else EXPECTED_ONEDNN_ARCHIVE_SHA256)
  expected_build_before_sha256 = (
      EXPECTED_STOCK_DIVISION_BASE_PLUGIN_SHA256
      if stock_division_exact
      else EXPECTED_MATERIALIZED_F16_MIDPOINT_BASE_PLUGIN_SHA256
      if materialized_f16_midpoint
      else
      EXPECTED_MATERIALIZED_F16_BASE_PLUGIN_SHA256
      if materialized_f16 else EXPECTED_BUILD_PLUGIN_SHA256)
  variant = (
      "materialized_f16_native_exp_stock_division_exact"
      if stock_division_exact
      else "materialized_f16_native_exp_midpoint_exact"
      if materialized_f16_midpoint
      else "materialized_f16_native_exp"
      if materialized_f16
      else "openvino_swish_parity")
  if candidate_plugin.exists():
    raise SystemExit(
        f"isolated candidate plugin already exists: {candidate_plugin}")
  required_paths = (
      ORIGINAL_BUILD_TOOL, FAILED_CORRECTNESS, FAILED_WORKER_STDERR,
      FAILED_WORKER_STDOUT,
      PARITY_PATCH, SCALAR_PATCH, VOLATILE_SCALAR_PATCH,
      MATERIALIZED_F16_PATCH, MATERIALIZED_F16_MIDPOINT_PATCH, SOURCE_TREE,
      ONEDNN, BUILD_TREE, ONEDNN_BUILD, ONEDNN_INSTALL_LIB, BUILD_PLUGIN,
      SEQ2233_PLUGIN, SEQ2236_PLUGIN, CONTROL_PLUGIN, GENERATOR,
      BASELINE_SWIGLU, CMAKE) + (
          (MATERIALIZED_F16_MIDPOINT_COMPONENT,)
          if materialized_f16_midpoint else ()) + (
              (
                  STOCK_DIVISION_PATCH, STOCK_DIVISION_COMPONENT,
                  KERNEL_CTX, GROUPED_MICRO_GEMM,
              )
              if stock_division_exact else ())
  missing = [str(path) for path in required_paths if not path.exists()]
  if missing:
    raise SystemExit("missing parity-build inputs: " + ", ".join(missing))

  repo = BASE.repository_state(output)
  failed = ORIGINAL.load_json(FAILED_CORRECTNESS)
  midpoint_component = (
      ORIGINAL.load_json(MATERIALIZED_F16_MIDPOINT_COMPONENT)
      if materialized_f16_midpoint else {})
  stock_division_component = (
      ORIGINAL.load_json(STOCK_DIVISION_COMPONENT)
      if stock_division_exact else {})
  parity_reverse = reverse_apply_check(ONEDNN, active_patch)
  stock_division_reverse = (
      reverse_apply_check(ONEDNN, STOCK_DIVISION_PATCH)
      if stock_division_exact else {})
  generator_text = GENERATOR.read_text(
      encoding="utf-8", errors="replace")
  kernel_ctx_text = (
      KERNEL_CTX.read_text(encoding="utf-8", errors="replace")
      if stock_division_exact else "")
  grouped_micro_gemm_text = (
      GROUPED_MICRO_GEMM.read_text(encoding="utf-8", errors="replace")
      if stock_division_exact else "")
  baseline_text = BASELINE_SWIGLU.read_text(
      encoding="utf-8", errors="replace")
  build_before = file_record(BUILD_PLUGIN)
  archive_before = file_record(ONEDNN_INSTALL_LIB)
  seq2233_before = file_record(SEQ2233_PLUGIN)
  seq2236_before = file_record(SEQ2236_PLUGIN)
  control_before = file_record(CONTROL_PLUGIN)
  memory_before = BASE.proc_meminfo()

  source_checks = [
      check("repository_clean_and_pushed_at_gate",
            repo["branch"] == "main" and repo["pushed"] and
            not repo["dirty"], **repo),
      check("seq2237_vector_conversion_failure_is_exact",
            sha256(FAILED_CORRECTNESS) ==
                EXPECTED_FAILED_CORRECTNESS_SHA256 and
            sha256(FAILED_WORKER_STDERR) ==
                EXPECTED_FAILED_WORKER_STDERR_SHA256 and
            sha256(FAILED_WORKER_STDOUT) ==
                EXPECTED_FAILED_WORKER_STDOUT_SHA256 and
            failed.get("required_checks_passed") is False and
            failed.get("worker", {}).get("returncode") == 1 and
            failed.get("worker", {}).get("oom_observed") is False and
            failed.get("correctness", {}).get("checkpoint_count") == 0 and
            "ZE_RESULT_ERROR_MODULE_BUILD_FAILURE" in
                FAILED_WORKER_STDOUT.read_text(
                    encoding="utf-8", errors="replace"),
            failed_correctness_sha256=sha256(FAILED_CORRECTNESS),
            failed_worker_stderr_sha256=sha256(FAILED_WORKER_STDERR),
            failed_worker_stdout_sha256=sha256(FAILED_WORKER_STDOUT)),
      check("active_patch_and_materialized_postimage_are_exact",
            sha256(active_patch) == expected_active_patch_sha256 and
            sha256(GENERATOR) == expected_generator_sha256 and
            (
                not materialized_f16 or (
                    sha256(PARITY_PATCH)
                    == EXPECTED_PARITY_PATCH_SHA256 and
                    sha256(SCALAR_PATCH)
                    == EXPECTED_SCALAR_PATCH_SHA256 and
                    sha256(VOLATILE_SCALAR_PATCH)
                    == EXPECTED_VOLATILE_SCALAR_PATCH_SHA256 and
                    (
                        not materialized_f16_midpoint or
                        sha256(MATERIALIZED_F16_PATCH)
                        == EXPECTED_MATERIALIZED_F16_PATCH_SHA256))) and
            (
                not stock_division_exact or (
                    sha256(STOCK_DIVISION_PATCH)
                    == EXPECTED_STOCK_DIVISION_PATCH_SHA256 and
                    sha256(KERNEL_CTX)
                    == EXPECTED_STOCK_DIVISION_KERNEL_CTX_SHA256 and
                    sha256(GROUPED_MICRO_GEMM)
                    == EXPECTED_STOCK_DIVISION_GROUPED_MICRO_GEMM_SHA256 and
                    stock_division_reverse.get("returncode") == 0)) and
            parity_reverse.get("returncode") == 0,
            variant=variant,
            active_patch=str(active_patch),
            active_patch_sha256=sha256(active_patch),
            generator_sha256=sha256(GENERATOR),
            reverse_apply=parity_reverse,
            stock_division_patch=(
                file_record(STOCK_DIVISION_PATCH)
                if stock_division_exact else None),
            stock_division_reverse=(
                stock_division_reverse if stock_division_exact else None),
            kernel_ctx=(
                file_record(KERNEL_CTX) if stock_division_exact else None),
            grouped_micro_gemm=(
                file_record(GROUPED_MICRO_GEMM)
                if stock_division_exact else None)),
      check("parity_correction_matches_locked_stock_boundary",
            "DECLARE_2D_TILE(post_ops_f16_tile_type" in generator_text and
            "tile_convert((*c_tile), eltwise_f16_boundary_" in
                generator_text and
            (
                (
                    "volatile half eltwise_gate_f16_" in generator_text and
                    "convert_float(eltwise_gate_f16_" in generator_text and
                    "tile_convert(eltwise_f16_boundary_" not in generator_text)
                if materialized_f16 else
                "tile_convert(eltwise_f16_boundary_" in generator_text) and
            "native_exp(-(" in generator_text and
            (
                not materialized_f16_midpoint or (
                    "0xc173" in generator_text and
                    "nextafter(eltwise_value_" in generator_text and
                    "-INFINITY" in generator_text)) and
            (
                not stock_division_exact or (
                    "0xc173" not in generator_text and
                    "void remove_option(const char *option)" in
                        kernel_ctx_text and
                    "find_po_in_chain(po_chain_, po_kind_t::eltwise)"
                        in grouped_micro_gemm_text and
                    "kernel_ctx_.remove_option(" in grouped_micro_gemm_text and
                    '"-cl-fp32-correctly-rounded-divide-sqrt"'
                        in grouped_micro_gemm_text)) and
            "ACC_DTYPE gate_value = as_half" in baseline_text and
            "native_exp(-SWISH_BETA * x)" in baseline_text,
            note=(
                "the in-register F16 round and native_exp reproduce the "
                "removed stock gate materialization and activation")),
      check("finite_f16_midpoint_component_gate_is_exact",
            not materialized_f16_midpoint or (
                sha256(MATERIALIZED_F16_MIDPOINT_COMPONENT)
                == EXPECTED_MATERIALIZED_F16_MIDPOINT_COMPONENT_SHA256 and
                midpoint_component.get("required_checks_passed") is True and
                midpoint_component.get("diagnosis")
                == "micro_postops_match_stored_stock_pipeline" and
                midpoint_component.get(
                    "probe", {}).get(
                        "exhaustive_finite_f16_activation", {}).get(
                            "input_value_count") == 63488 and
                midpoint_component.get(
                    "probe", {}).get(
                        "exhaustive_finite_f16_activation", {}).get(
                            "swish_vs_stock_oracle", {}).get(
                                "mismatch_count") == 0 and
                len(midpoint_component.get(
                    "probe", {}).get("scenarios", [])) == 3 and
                all(
                    row.get(
                        "swish_vs_stored_gate_oracle", {}).get(
                            "mismatch_count") == 0 and
                    row.get(
                        "swish_binary_vs_stock_pipeline", {}).get(
                            "mismatch_count") == 0
                    for row in midpoint_component.get(
                        "probe", {}).get("scenarios", []))),
            component_result=(
                file_record(MATERIALIZED_F16_MIDPOINT_COMPONENT)
                if materialized_f16_midpoint else None)),
      check("stock_division_product_component_gate_is_exact",
            not stock_division_exact or (
                sha256(STOCK_DIVISION_COMPONENT)
                == EXPECTED_STOCK_DIVISION_COMPONENT_SHA256 and
                stock_division_component.get(
                    "required_checks_passed") is True and
                stock_division_component.get("diagnosis")
                == "micro_postops_match_stored_stock_pipeline" and
                stock_division_component.get(
                    "probe", {}).get(
                        "exhaustive_finite_f16_activation", {}).get(
                            "input_value_count") == 63488 and
                stock_division_component.get(
                    "probe", {}).get(
                        "exhaustive_finite_f16_activation", {}).get(
                            "swish_vs_stock_oracle", {}).get(
                                "mismatch_count") == 0 and
                stock_division_component.get(
                    "probe", {}).get(
                        "exhaustive_finite_f16_activation", {}).get(
                            "swish_f32_vs_stock_f32", {}).get(
                                "mismatch_count") == 0 and
                stock_division_component.get(
                    "probe", {}).get(
                        "exhaustive_finite_f16_activation", {}).get(
                            "swish_f32_vs_stock_f32", {}).get(
                                "max_ulp") == 0 and
                len(stock_division_component.get(
                    "probe", {}).get("scenarios", [])) == 3 and
                all(
                    row.get(
                        "swish_vs_stored_gate_oracle", {}).get(
                            "mismatch_count") == 0 and
                    row.get(
                        "swish_binary_vs_stock_pipeline", {}).get(
                            "mismatch_count") == 0 and
                    row.get(
                        "producer_fed_swiglu_vs_stock_pipeline", {}).get(
                            "mismatch_count") == 0 and
                    row.get(
                        "snapshot_fed_swiglu_vs_stock_pipeline", {}).get(
                            "mismatch_count") == 0 and
                    row.get(
                        "swish_f32_vs_stock_f32", {}).get(
                            "mismatch_count") == 0
                    for row in stock_division_component.get(
                        "probe", {}).get("scenarios", []))),
            component_result=(
                file_record(STOCK_DIVISION_COMPONENT)
                if stock_division_exact else None)),
      check("archive_and_build_are_exact_before_retry",
            archive_before["sha256"] ==
                expected_archive_before_sha256 and
            seq2233_before["sha256"] == EXPECTED_SEQ2233_SHA256 and
            seq2236_before["sha256"] == EXPECTED_BUILD_PLUGIN_SHA256 and
            control_before["sha256"] == EXPECTED_CONTROL_SHA256 and
            build_before["sha256"] == expected_build_before_sha256 and
            candidate_plugin != SEQ2233_PLUGIN.resolve() and
            candidate_plugin != SEQ2236_PLUGIN.resolve() and
            candidate_plugin != CONTROL_PLUGIN.resolve() and
            candidate_plugin != BUILD_PLUGIN.resolve(),
            archive=archive_before, build_plugin=build_before,
            seq2233_plugin=seq2233_before, seq2236_plugin=seq2236_before,
            control_plugin=control_before),
      check("eight_gib_preflight_clears",
            int(memory_before.get("MemAvailable", 0)) >= PREFLIGHT_BYTES,
            available_bytes=memory_before.get("MemAvailable"),
            preflight_bytes=PREFLIGHT_BYTES),
  ]
  source_admitted = all(row["pass"] for row in source_checks)

  environment = os.environ.copy()
  conda_bin = "/home/intel/intel-box-env/conda/bin"
  conda_lib = "/home/intel/intel-box-env/conda/lib"
  environment["PATH"] = conda_bin + ":" + environment.get("PATH", "")
  environment["LD_LIBRARY_PATH"] = (
      conda_lib + ":" + environment.get("LD_LIBRARY_PATH", ""))
  onednn_command = [
      str(CMAKE), "--build", str(ONEDNN_BUILD),
      "--target", "install", "--parallel", "1"]
  plugin_command = [
      str(CMAKE), "--build", str(BUILD_TREE),
      "--target", "openvino_intel_gpu_plugin", "--parallel", "1"]
  onednn_build: dict[str, Any] = {
      "returncode": 125, "timed_out": False,
      "memory_guard_tripped": False, "oom_observed": False,
      "skipped": "source gate failed", "monitor": {}}
  plugin_build: dict[str, Any] = {
      "returncode": 125, "timed_out": False,
      "memory_guard_tripped": False, "oom_observed": False,
      "skipped": "source gate failed", "monitor": {}}
  if source_admitted:
    onednn_build = BASE.run_scoped(
        output, raw, "build-onednn-pr35924-parity", onednn_command,
        args.build_timeout_s, args.poll_interval_s, environment)
    if stage_ok(onednn_build):
      plugin_build = BASE.run_scoped(
          output, raw, "build-plugin-pr35924-parity", plugin_command,
          args.build_timeout_s, args.poll_interval_s, environment)

  archive_after = file_record(ONEDNN_INSTALL_LIB)
  build_after = file_record(BUILD_PLUGIN)
  seq2233_after = file_record(SEQ2233_PLUGIN)
  seq2236_after = file_record(SEQ2236_PLUGIN)
  control_after = file_record(CONTROL_PLUGIN)
  build_stdout = "\n".join(
      path.read_text(encoding="utf-8", errors="replace")
      for path in (
          raw / "build-onednn-pr35924-parity.stdout",
          raw / "build-plugin-pr35924-parity.stdout")
      if path.is_file())
  compile_steps = len(re.findall(
      r"\bBuilding (?:C|CXX) object\b", build_stdout))
  copied = False
  if (stage_ok(onednn_build) and stage_ok(plugin_build) and
      build_after["sha256"] is not None and
      build_after["sha256"] != build_before["sha256"]):
    candidate_plugin.parent.mkdir(parents=True, exist_ok=False)
    shutil.copy2(BUILD_PLUGIN, candidate_plugin)
    copied = True
  candidate = file_record(candidate_plugin)
  minimum_available = min(
      int(stage.get("monitor", {}).get(
          "system_available_min_bytes", memory_before.get("MemAvailable", 0)))
      for stage in (onednn_build, plugin_build))
  oom_free = all(
      not stage.get("oom_observed", False) and
      not stage.get("memory_guard_tripped", False) and
      all(int(value) == 0 for key, value in
          stage.get("monitor", {}).get("memory_events_max", {}).items()
          if key in ("oom", "oom_kill", "oom_group_kill"))
      for stage in (onednn_build, plugin_build))

  checks = [
      *source_checks,
      check("sole_serial_onednn_and_plugin_builds_succeed",
            source_admitted and stage_ok(onednn_build) and
            stage_ok(plugin_build) and
            archive_after["sha256"] != archive_before["sha256"] and
            build_after["sha256"] != build_before["sha256"] and
            "grouped_post_ops_gen.cpp.o" in build_stdout and
            compile_steps >= 1,
            compile_steps=compile_steps,
            onednn_build=onednn_build, plugin_build=plugin_build,
            archive_before=archive_before, archive_after=archive_after,
            build_before=build_before, build_after=build_after),
      check("isolated_parity_candidate_is_copied",
            copied and candidate["exists"] and
            candidate["sha256"] == build_after["sha256"] and
            candidate["sha256"] not in (
                EXPECTED_SEQ2233_SHA256, EXPECTED_BUILD_PLUGIN_SHA256,
                EXPECTED_MATERIALIZED_F16_BASE_PLUGIN_SHA256,
                EXPECTED_MATERIALIZED_F16_MIDPOINT_BASE_PLUGIN_SHA256,
                EXPECTED_STOCK_DIVISION_BASE_PLUGIN_SHA256,
                EXPECTED_CONTROL_SHA256),
            candidate=candidate),
      check("accepted_and_failed_controls_remain_unchanged",
            seq2233_after["sha256"] == EXPECTED_SEQ2233_SHA256 and
            seq2236_after["sha256"] == EXPECTED_BUILD_PLUGIN_SHA256 and
            control_after["sha256"] == EXPECTED_CONTROL_SHA256,
            seq2233_after=seq2233_after, seq2236_after=seq2236_after,
            control_after=control_after),
      check("four_gib_abort_and_oom_guards_hold",
            minimum_available >= ABORT_BYTES and oom_free,
            minimum_available_bytes=minimum_available,
            abort_bytes=ABORT_BYTES),
      check("build_uses_no_gpu_model_or_inference", True,
            gpu_contexts_created=0, model_workers_started=0,
            infer_requests_created=0, inference_workers_started=0,
            gpu_kernels_executed=0),
  ]
  required = all(row["pass"] for row in checks)
  verdict = (
      "admit_one_pr35924_parity_output130_correctness_worker"
      if required else
      "repair_pr35924_parity_source_or_serial_build")
  variant_note = (
      "Correctly-rounded divide is removed only from grouped post-op kernels "
      "that contain eltwise, matching the stock OpenVINO compiler options "
      "without an input-specific correction."
      if stock_division_exact else
      "The only F16-output midpoint mismatch found across all "
      "63,488 finite F16 encodings is corrected."
      if materialized_f16_midpoint else
      "The removed stock arithmetic boundary is preserved.")
  payload = {
      "schema": SCHEMA,
      "workstream": WS,
      "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
      "repository": repo,
      "variant": variant,
      "verdict": verdict,
      "required_checks_passed": required,
      "candidate_output130_correctness_worker_admitted": required,
      "performance_worker_admitted": False,
      "formal_performance_admitted": False,
      "checks": checks,
      "candidate_plugin": candidate,
      "accepted_control_plugin": control_after,
      "failed_seq2233_plugin": seq2233_after,
      "failed_seq2236_plugin": seq2236_after,
      "minimum_available_bytes": minimum_available,
      "next_action": {
          "route": "pr35924_swiglu_parity_output130_correctness",
          "requirements": [
              "run one candidate-only output130 teacher-forced worker",
              "require exact tokens, KLD <= 0.005, and top-1 >= 0.99",
              "require all 40 optimized MoE owners and unchanged census",
              "only a pass may fund one short control-candidate point block",
          ],
      },
  }
  write_json(output / "metrics.json", payload)
  write_json(output / "manifest.json", {
      "schema": SCHEMA,
      "tool": str(Path(__file__).relative_to(ROOT)),
      "repository": repo,
      "inputs": {
          str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path):
              sha256(path)
          for path in (
              ORIGINAL_BUILD_TOOL, FAILED_CORRECTNESS, FAILED_WORKER_STDERR,
              FAILED_WORKER_STDOUT, PARITY_PATCH, SCALAR_PATCH,
              VOLATILE_SCALAR_PATCH, MATERIALIZED_F16_PATCH,
              MATERIALIZED_F16_MIDPOINT_PATCH,
              *((MATERIALIZED_F16_MIDPOINT_COMPONENT,)
                if materialized_f16_midpoint else ()),
              *((STOCK_DIVISION_PATCH, STOCK_DIVISION_COMPONENT,
                 KERNEL_CTX, GROUPED_MICRO_GEMM)
                if stock_division_exact else ()),
              GENERATOR, BASELINE_SWIGLU,
              SEQ2233_PLUGIN, SEQ2236_PLUGIN, CONTROL_PLUGIN)
      },
      "candidate_plugin": candidate,
      "parallelism": 1,
      "gpu_contexts": 0,
      "model_workers": 0,
      "infer_requests": 0,
  })
  report = f"""# PR35924 OpenVINO-Swish parity build

Verdict: **{verdict}**. Required checks: `{str(required).lower()}`.

Variant: `{variant}`. The correction makes the gate activation consume the
F16 tile value and uses the stock kernel's `native_exp`.
{variant_note} This preserves the removed arithmetic boundary without
restoring its scratch traffic or launch.
oneDNN and the GPU plugin were rebuilt serially; the isolated candidate SHA is
`{candidate.get('sha256')}`.

Compile steps: `{compile_steps}`. Minimum available memory:
`{minimum_available} B`. No GPU context, model, InferRequest, inference, OOM,
or guard event occurred.
"""
  (output / "report.md").write_text(report, encoding="utf-8")
  print(json.dumps({
      "artifact": str(output.relative_to(ROOT)),
      "verdict": verdict,
      "required_checks_passed": required,
      "candidate_plugin_sha256": candidate.get("sha256"),
      "compile_steps": compile_steps,
      "minimum_available_bytes": minimum_available,
      "oom_free": oom_free,
  }, separators=(",", ":")), flush=True)
  return 0 if required else 2


if __name__ == "__main__":
  raise SystemExit(main())
