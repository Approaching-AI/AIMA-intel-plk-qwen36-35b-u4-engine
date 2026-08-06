#!/usr/bin/env python3
"""Gate the standalone grouped S8-by-U4 layer-27 prefill carrier."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shlex
import struct
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "intel-qwen36-grouped-s8-u4-prefill-gate-v4"
BASE_GATE_PATH = ROOT / "tools/intel-qwen36-onednn-q4k-bucket-component-gate.py"
PREP_SOURCE = ROOT / "engine/tools/onednn_grouped_q4k_moe_component.cpp"
RUNTIME_SOURCE = ROOT / "engine/src/grouped_s8_u4_prefill_runtime.cpp"
RUNTIME_MAIN = ROOT / "engine/tools/grouped_s8_u4_prefill_runtime.cpp"
RUNTIME_API_SMOKE = (
    ROOT / "engine/tools/grouped_s8_u4_prefill_api_smoke.cpp")
RUNTIME_RESIDENT_SMOKE = (
    ROOT / "engine/tools/grouped_s8_u4_prefill_resident_smoke.cpp")
RUNTIME_HEADER = (
    ROOT / "engine/include/intel_qwen36/grouped_s8_u4_prefill_runtime.hpp")
KERNEL_SOURCE = (
    ROOT / "engine/gpu/opencl/grouped_s8_u4_f16_contribution_moe.cl")
ONEDNN_PATCH = ROOT / "engine/gpu/opencl/onednn-grouped-s8-u4-fused.patch"
DEFAULT_CAPTURE = (
    ROOT / "output/onednn-q4k-routed-moe-component-gate-20260711Tseq646cleanZ/"
    "raw/capture")
DEFAULT_ONEDNN_BUILD = Path(
    "/home/intel/intel-qwen36-r0/build/onednn-01b479-ocl-grouped")
PATCHED_ONEDNN_PATHS = [
    "src/gpu/intel/matmul/grouped_micro_gemm.cl",
    "src/gpu/intel/matmul/grouped_micro_gemm.cpp",
    "src/gpu/intel/ocl/engine.cpp",
    "src/gpu/intel/ocl/kernel.cpp",
]
PREPACK_SIZES = {
    "gateup-weights.bin": 268_435_456,
    "gateup-scales.bin": 67_108_864,
    "gateup-min-codes.bin": 16_777_216,
    "gateup-dmins.bin": 8_388_608,
    "down-weights.bin": 134_217_728,
    "down-scales.bin": 33_554_432,
    "down-min-codes.bin": 8_388_608,
    "down-dmins.bin": 4_194_304,
}
ADR_0015_KERNEL_CAP_US = 9_771.436
COMPONENT_COSINE_MIN = 0.999
COMPONENT_RELATIVE_L2_MAX = 0.002


def load_base_gate() -> Any:
  spec = importlib.util.spec_from_file_location("iq36_base_gate", BASE_GATE_PATH)
  if spec is None or spec.loader is None:
    raise SystemExit(f"could not import {BASE_GATE_PATH}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


BASE = load_base_gate()


def iso_now() -> str:
  return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--model", type=Path, default=BASE.DEFAULT_MODEL)
  parser.add_argument("--census", type=Path, default=BASE.DEFAULT_CENSUS)
  parser.add_argument("--capture", type=Path, default=DEFAULT_CAPTURE)
  parser.add_argument("--tensor-index", type=Path,
                      default=BASE.DEFAULT_TENSOR_INDEX)
  parser.add_argument("--env-script", type=Path,
                      default=BASE.DEFAULT_ENV_SCRIPT)
  parser.add_argument("--cxx", type=Path, default=BASE.DEFAULT_CXX)
  parser.add_argument("--onednn-source", type=Path,
                      default=BASE.DEFAULT_ONEDNN_SOURCE)
  parser.add_argument("--onednn-build", type=Path,
                      default=DEFAULT_ONEDNN_BUILD)
  parser.add_argument("--warmup", type=int, default=3)
  parser.add_argument("--repeat", type=int, default=11)
  parser.add_argument("--jobs", type=int, default=16)
  parser.add_argument("--timeout-s", type=int, default=1800)
  parser.add_argument("--skip-onednn-build", action="store_true")
  parser.add_argument("--out-dir", type=Path)
  args = parser.parse_args()
  if min(args.warmup, args.repeat, args.jobs, args.timeout_s) <= 0:
    parser.error("warmup, repeat, jobs, and timeout must be positive")
  if args.out_dir is None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.out_dir = ROOT / f"output/grouped-s8-u4-prefill-gate-{stamp}"
  return args


def check(name: str, passed: bool, **details: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **details}


def failed_run(command: list[str], reason: str) -> dict[str, Any]:
  return {"command": command, "returncode": 125, "stderr": reason,
          "stdout": "", "timed_out": False}


def shell_run_env(command: list[str], env_script: Path, timeout_s: int,
                  extra_env: dict[str, str] | None = None) -> dict[str, Any]:
  exports = {"INTEL_FORCE_PROBE": "b080", "DNNL_VERBOSE": "0"}
  if extra_env:
    exports.update(extra_env)
  export_text = " ".join(
      f"{key}={shlex.quote(value)}" for key, value in exports.items())
  shell = (
      f"source {shlex.quote(str(env_script))} >/dev/null 2>&1 && "
      f"export {export_text} && {shlex.join(command)}")
  return BASE.run(["bash", "-lc", shell], timeout_s)


def git_bytes(root: Path, *parts: str) -> bytes:
  result = subprocess.run(
      ["git", *parts], cwd=root, check=False, capture_output=True)
  return result.stdout if result.returncode == 0 else b""


def compare_checks(prefix: str, comparison: dict[str, Any],
                   count: int) -> list[dict[str, Any]]:
  mismatch_count = comparison.get("mismatch_count")
  max_abs_diff = float(comparison.get("max_abs_diff", float("inf")))
  cosine = float(comparison.get("cosine", float("-inf")))
  relative_l2 = float(comparison.get("relative_l2", float("inf")))
  return [
      check(f"all_{count}_{prefix}_values_compared",
            comparison.get("compared_value_count") == count),
      check(f"{prefix}_meets_component_accuracy_contract",
            comparison.get("finite") is True and
            cosine >= COMPONENT_COSINE_MIN and
            relative_l2 <= COMPONENT_RELATIVE_L2_MAX,
            cosine=cosine, cosine_min=COMPONENT_COSINE_MIN,
            relative_l2=relative_l2,
            relative_l2_max=COMPONENT_RELATIVE_L2_MAX,
            max_abs_diff=max_abs_diff, mismatch_count=mismatch_count),
  ]


def generator_command(binary: Path, args: argparse.Namespace,
                      tensors: dict[str, dict[str, Any]],
                      payloads: dict[str, Path], topk_stride: int,
                      kernel_cap_us: float) -> list[str]:
  return [
      str(binary), "--model", str(args.model),
      "--weight-offset", str(tensors["gate_up"]["absolute_offset"]),
      "--weight-bytes", str(tensors["gate_up"]["nbytes"]),
      "--input", str(payloads[f"attn_post_norm-{BASE.LAYER}"]),
      "--topk", str(payloads[f"ffn_moe_topk-{BASE.LAYER}"]),
      "--topk-stride", str(topk_stride),
      "--oracle", str(payloads[f"ffn_moe_swiglu-{BASE.LAYER}"]),
      "--down-weight-offset", str(tensors["down"]["absolute_offset"]),
      "--down-weight-bytes", str(tensors["down"]["nbytes"]),
      "--router-weights",
      str(payloads[f"ffn_moe_weights_norm-{BASE.LAYER}"]),
      "--down-oracle", str(payloads[f"ffn_moe_down-{BASE.LAYER}"]),
      "--moe-oracle", str(payloads[f"ffn_moe_out-{BASE.LAYER}"]),
      "--warmup", "1", "--repeat", "1",
      "--kernel-cap-us", str(kernel_cap_us),
  ]


def disassemble(binary: Path, directory: Path,
                args: argparse.Namespace) -> dict[str, Any]:
  directory.mkdir()
  result = shell_run_env([
      "ocloc", "disasm", "-file", str(binary), "-dump", str(directory),
      "-device", "0xb080",
  ], args.env_script, args.timeout_s)
  BASE.write_run_logs(directory.parent, f"{directory.name}-disasm", result)
  assembly = "\n".join(
      path.read_text(encoding="utf-8", errors="replace")
      for path in directory.rglob("*.asm"))
  ze_info = "\n".join(
      path.read_text(encoding="utf-8", errors="replace")
      for path in directory.rglob(".ze_info"))
  return {"assembly": assembly, "result": result, "ze_info": ze_info}


def main() -> int:
  args = parse_args()
  out_dir = args.out_dir.resolve()
  raw_dir = out_dir / "raw"
  raw_dir.mkdir(parents=True, exist_ok=False)
  prepack_dir = raw_dir / "prepacked"
  prepack_dir.mkdir()
  required = [
      args.model, args.census / "result.json",
      args.census / "layer-shapes.jsonl",
      args.census / "router-assignments.jsonl",
      args.capture / "tensor-dumps.jsonl", args.tensor_index,
      args.env_script, args.cxx, args.onednn_source, args.onednn_build,
      args.onednn_build / "src/libdnnl.so",
      args.onednn_build / "include/oneapi/dnnl/dnnl_config.h",
      BASE_GATE_PATH, PREP_SOURCE, RUNTIME_SOURCE, RUNTIME_MAIN,
      RUNTIME_API_SMOKE, RUNTIME_RESIDENT_SMOKE,
      RUNTIME_HEADER, ROOT / "engine/CMakeLists.txt",
      ROOT / "engine/boundaries.json",
      KERNEL_SOURCE, ONEDNN_PATCH,
  ]
  missing = [str(path) for path in required if not path.exists()]
  if missing:
    raise SystemExit("missing required paths: " + ", ".join(missing))
  if BASE.sha256_file(args.model) != BASE.MODEL_SHA256:
    raise SystemExit("locked model hash mismatch")

  created_at = iso_now()
  census_result, shape, assignments = BASE.selected_shape(args.census)
  budget = BASE.derive_budget(shape, True)
  legacy_cap_us = float(budget["kernel_cap_us"])
  kernel_cap_us = ADR_0015_KERNEL_CAP_US
  budget["legacy_1p25_kernel_cap_us"] = legacy_cap_us
  budget["kernel_cap_us"] = kernel_cap_us
  budget["target_contract"] = "ADR 0015 measured 1.10x"
  budget["whole_window_budget_us"] = (
      kernel_cap_us + float(budget["reserved_noncomponent_us"]))
  tensors = BASE.tensor_rows(args.tensor_index)
  metadata, payloads = BASE.captured_payloads(args.capture, True)
  topk_name = f"ffn_moe_topk-{BASE.LAYER}"
  topk_stride = int(metadata[topk_name]["nb"][1])
  router_ids_match = (
      BASE.captured_router_ids(payloads[topk_name], topk_stride) ==
      assignments["expert_ids_by_token"])

  source_commit = BASE.git_output(args.onednn_source, "rev-parse", "HEAD")
  source_diff = git_bytes(
      args.onednn_source, "diff", "--unified=0", "--", *PATCHED_ONEDNN_PATHS)
  expected_patch = ONEDNN_PATCH.read_bytes()
  source_status = git_bytes(
      args.onednn_source, "status", "--porcelain").decode("utf-8")
  expected_status = sorted(f" M {path}" for path in PATCHED_ONEDNN_PATHS)
  patch_exact = (source_diff == expected_patch and
                 sorted(source_status.splitlines()) == expected_status)

  onednn_build_command = [
      "cmake", "--build", str(args.onednn_build), "--target", "dnnl",
      "-j", str(args.jobs),
  ]
  onednn_build = (
      {"command": onednn_build_command, "returncode": 0,
       "stderr": "skipped by request", "stdout": "", "timed_out": False}
      if args.skip_onednn_build else
      shell_run_env(onednn_build_command, args.env_script, args.timeout_s))
  BASE.write_run_logs(raw_dir, "onednn-build", onednn_build)

  prep_binary = raw_dir / "offline-prepack-generator"
  prep_build_command = [
      str(args.cxx), "-std=gnu++17", "-O3", "-DNDEBUG",
      "-DCL_TARGET_OPENCL_VERSION=300",
      f"-I{args.onednn_build / 'include'}",
      f"-I{args.onednn_source / 'include'}", str(PREP_SOURCE),
      f"-L{args.onednn_build / 'src'}",
      f"-Wl,-rpath,{args.onednn_build / 'src'}", "-ldnnl", "-lOpenCL",
      "-o", str(prep_binary),
  ]
  prep_build = (
      shell_run_env(prep_build_command, args.env_script, args.timeout_s)
      if onednn_build["returncode"] == 0 else
      failed_run(prep_build_command, "oneDNN build failed"))
  BASE.write_run_logs(raw_dir, "prep-build", prep_build)

  base_generator = generator_command(
      prep_binary, args, tensors, payloads, topk_stride, kernel_cap_us)
  native_binaries: dict[str, Path] = {}
  generation_results: dict[str, dict[str, Any]] = {}
  for kind in ("gateup", "down"):
    prefix = raw_dir / kind
    extra_env = {
        "DNNL_PRIMITIVE_CACHE_CAPACITY": "0",
        "IQ36_GENERATE_S8_GROUPED": "1",
        "IQ36_GROUPED_FUSED_KIND": kind,
        "IQ36_DUMP_FUSED_PROGRAM_PREFIX": str(prefix),
        "IQ36_EXIT_AFTER_FUSED_DUMP": "1",
    }
    result = (
        shell_run_env(base_generator, args.env_script, args.timeout_s,
                      extra_env)
        if prep_build["returncode"] == 0 else
        failed_run(base_generator, "prep build failed"))
    generation_results[kind] = result
    BASE.write_run_logs(raw_dir, f"generate-{kind}", result)
    native_binaries[kind] = raw_dir / f"{kind}.0.bin"

  prep_command = [
      *base_generator,
      "--grouped-gateup-binary", str(native_binaries["gateup"]),
      "--grouped-down-binary", str(native_binaries["down"]),
      "--dump-prepacked-dir", str(prepack_dir), "--prepack-only",
  ]
  generated = all(path.is_file() for path in native_binaries.values())
  prep_result = (
      shell_run_env(prep_command, args.env_script, args.timeout_s, {
          "DNNL_PRIMITIVE_CACHE_CAPACITY": "0",
          "IQ36_GENERATE_S8_GROUPED": "1",
      })
      if generated else failed_run(prep_command, "native generation failed"))
  BASE.write_run_logs(raw_dir, "prepack", prep_result)
  prep_probe = BASE.parse_probe(prep_result)

  runtime_binary = raw_dir / "grouped-s8-u4-prefill-runtime"
  runtime_build_command = [
      str(args.cxx), "-std=gnu++17", "-O3", "-DNDEBUG",
      "-DCL_TARGET_OPENCL_VERSION=300", f"-I{ROOT / 'engine/include'}",
      str(RUNTIME_SOURCE), str(RUNTIME_MAIN), "-lOpenCL",
      "-o", str(runtime_binary),
  ]
  runtime_build = shell_run_env(
      runtime_build_command, args.env_script, args.timeout_s)
  BASE.write_run_logs(raw_dir, "runtime-build", runtime_build)
  api_smoke_binary = raw_dir / "grouped-prefill-api-smoke"
  api_smoke_build_command = [
      str(args.cxx), "-std=gnu++17", "-O3", "-DNDEBUG",
      "-DCL_TARGET_OPENCL_VERSION=300", f"-I{ROOT / 'engine/include'}",
      str(RUNTIME_SOURCE), str(RUNTIME_API_SMOKE), "-lOpenCL",
      "-o", str(api_smoke_binary),
  ]
  api_smoke_build = shell_run_env(
      api_smoke_build_command, args.env_script, args.timeout_s)
  BASE.write_run_logs(raw_dir, "api-smoke-build", api_smoke_build)
  api_smoke_command = [
      str(api_smoke_binary), str(payloads[f"ffn_moe_topk-{BASE.LAYER}"]),
      str(topk_stride),
  ]
  api_smoke_run = (
      shell_run_env(api_smoke_command, args.env_script, args.timeout_s)
      if api_smoke_build["returncode"] == 0 else
      failed_run(api_smoke_command, "API smoke build failed"))
  BASE.write_run_logs(raw_dir, "api-smoke", api_smoke_run)
  api_smoke_probe = BASE.parse_probe(api_smoke_run)
  resident_smoke_binary = raw_dir / "grouped-prefill-resident-smoke"
  resident_smoke_build_command = [
      str(args.cxx), "-std=gnu++17", "-O3", "-DNDEBUG",
      "-DCL_TARGET_OPENCL_VERSION=300", f"-I{ROOT / 'engine/include'}",
      str(RUNTIME_SOURCE), str(RUNTIME_RESIDENT_SMOKE), "-lOpenCL",
      "-o", str(resident_smoke_binary),
  ]
  resident_smoke_build = shell_run_env(
      resident_smoke_build_command, args.env_script, args.timeout_s)
  BASE.write_run_logs(raw_dir, "resident-smoke-build", resident_smoke_build)
  resident_smoke_command = [
      str(resident_smoke_binary), str(prepack_dir),
      str(native_binaries["gateup"]), str(native_binaries["down"]),
      str(KERNEL_SOURCE), str(payloads[f"attn_post_norm-{BASE.LAYER}"]),
      str(payloads[f"ffn_moe_topk-{BASE.LAYER}"]), str(topk_stride),
      str(payloads[f"ffn_moe_weights_norm-{BASE.LAYER}"]),
      str(payloads[f"ffn_moe_out-{BASE.LAYER}"]),
  ]
  resident_smoke_run = (
      shell_run_env(resident_smoke_command, args.env_script, args.timeout_s)
      if resident_smoke_build["returncode"] == 0 and generated and
      prep_result["returncode"] in (0, 2) else
      failed_run(resident_smoke_command, "resident smoke prerequisites failed"))
  BASE.write_run_logs(raw_dir, "resident-smoke", resident_smoke_run)
  resident_smoke_probe = BASE.parse_probe(resident_smoke_run)
  ldd_result = (
      BASE.run(["ldd", str(runtime_binary)], args.timeout_s)
      if runtime_build["returncode"] == 0 else
      failed_run(["ldd", str(runtime_binary)], "runtime build failed"))
  BASE.write_run_logs(raw_dir, "runtime-ldd", ldd_result)

  schedule_probe_dir = raw_dir / "schedule-probes"
  schedule_probe_dir.mkdir()
  all_shapes = {
      int(row["layer"]): row
      for row in BASE.load_jsonl(args.census / "layer-shapes.jsonl")
      if row.get("case_id") == BASE.CASE_ID
  }
  all_assignments = {
      int(row["layer"]): row
      for row in BASE.load_jsonl(args.census / "router-assignments.jsonl")
      if row.get("case_id") == BASE.CASE_ID
  }
  schedule_probes = []
  for layer in range(40):
    shape_row = all_shapes.get(layer, {})
    assignment_row = all_assignments.get(layer, {})
    expert_ids = assignment_row.get("expert_ids_by_token", [])
    payload = bytearray()
    if isinstance(expert_ids, list):
      for token_experts in expert_ids:
        if isinstance(token_experts, list):
          for expert in token_experts:
            payload.extend(struct.pack("<i", int(expert)))
    topk_probe = schedule_probe_dir / f"layer-{layer}.topk.i32"
    topk_probe.write_bytes(payload)
    command = [
        str(runtime_binary), "--schedule-probe-only",
        "--topk", str(topk_probe), "--topk-stride", "32",
    ]
    run = (
        shell_run_env(command, args.env_script, args.timeout_s)
        if runtime_build["returncode"] == 0 else
        failed_run(command, "runtime build failed"))
    probe = BASE.parse_probe(run)
    expected_max = int(shape_row.get("group_m_max", -1))
    expected_active = int(shape_row.get("active_expert_count", -1))
    passed = (
        run["returncode"] == 0 and len(payload) == 32_768 and
        probe.get("assignment_count") == 8192 and
        probe.get("active_experts") == expected_active and
        probe.get("max_group_size") == expected_max and
        probe.get("native_global_y") == ((expected_max + 31) // 32) * 4 and
        probe.get("dynamic_router_schedule") is True)
    schedule_probes.append({
        "active_experts": probe.get("active_experts"),
        "expected_active_experts": expected_active,
        "expected_max_group_size": expected_max,
        "layer": layer,
        "max_group_size": probe.get("max_group_size"),
        "native_global_y": probe.get("native_global_y"),
        "pass": passed,
        "returncode": run["returncode"],
        "schedule_prepare_us": probe.get("schedule_prepare_us"),
    })
  BASE.write_json(raw_dir / "schedule-probes.json", schedule_probes)

  runtime_command = [
      str(runtime_binary), "--prep-dir", str(prepack_dir),
      "--gateup-binary", str(native_binaries["gateup"]),
      "--down-binary", str(native_binaries["down"]),
      "--kernels", str(KERNEL_SOURCE),
      "--input", str(payloads[f"attn_post_norm-{BASE.LAYER}"]),
      "--topk", str(payloads[f"ffn_moe_topk-{BASE.LAYER}"]),
      "--topk-stride", str(topk_stride),
      "--oracle", str(payloads[f"ffn_moe_swiglu-{BASE.LAYER}"]),
      "--router-weights",
      str(payloads[f"ffn_moe_weights_norm-{BASE.LAYER}"]),
      "--down-oracle", str(payloads[f"ffn_moe_down-{BASE.LAYER}"]),
      "--moe-oracle", str(payloads[f"ffn_moe_out-{BASE.LAYER}"]),
      "--warmup", str(args.warmup), "--repeat", str(args.repeat),
      "--kernel-cap-us", str(kernel_cap_us),
  ]
  runtime_ready = runtime_build["returncode"] == 0 and prep_result["returncode"] in (0, 2)
  runtime_results = []
  probes = []
  for name in ("runtime", "runtime-confirm"):
    result = (
        shell_run_env(runtime_command, args.env_script, args.timeout_s)
        if runtime_ready else failed_run(runtime_command, "runtime inputs failed"))
    runtime_results.append(result)
    BASE.write_run_logs(raw_dir, name, result)
    probes.append(BASE.parse_probe(result))

  disassembly: dict[str, dict[str, Any]] = {}
  for kind, binary in native_binaries.items():
    disassembly[kind] = (
        disassemble(binary, raw_dir / f"disasm-{kind}", args)
        if binary.is_file() else
        {"assembly": "", "ze_info": "",
         "result": failed_run([], "native binary missing")})

  prepack_manifest: dict[str, Any] = {}
  prepack_files_ok = True
  for name, expected_size in PREPACK_SIZES.items():
    path = prepack_dir / name
    observed_size = path.stat().st_size if path.is_file() else None
    prepack_files_ok &= observed_size == expected_size
    prepack_manifest[name] = {
        "bytes": observed_size,
        "expected_bytes": expected_size,
        "sha256": BASE.sha256_file(path) if path.is_file() else None,
    }
  BASE.write_json(raw_dir / "prepack-manifest.json", prepack_manifest)
  static_prepack_contract = (
      BASE.load_json(prepack_dir / "manifest.json")
      if (prepack_dir / "manifest.json").is_file() else {})

  binary_manifest = {
      kind: {"bytes": path.stat().st_size,
             "sha256": BASE.sha256_file(path)}
      for kind, path in native_binaries.items() if path.is_file()
  }
  BASE.write_json(raw_dir / "native-binary-manifest.json", binary_manifest)
  ldd_lower = str(ldd_result.get("stdout", "")).lower()
  cmake_text = (ROOT / "engine/CMakeLists.txt").read_text(encoding="utf-8")
  boundaries = BASE.load_json(ROOT / "engine/boundaries.json")
  infra_targets = boundaries.get("infra_targets", [])
  repo_git = BASE.git_state()

  evidence_checks = [
      check("repository_clean_at_gate", repo_git["dirty"] is False),
      check("locked_census_gate_passed",
            census_result.get("required_checks_passed") is True),
      check("pinned_onednn_source_commit",
            source_commit == BASE.ONEDNN_COMMIT,
            observed=source_commit, required=BASE.ONEDNN_COMMIT),
      check("onednn_source_diff_exactly_matches_repo_patch", patch_exact,
            patch_sha256=BASE.sha256_file(ONEDNN_PATCH)),
      check("locked_capture_payload_hashes_match", True),
      check("captured_router_ids_match_seq639", router_ids_match),
      check("onednn_generator_build_passed", onednn_build["returncode"] == 0),
      check("offline_prepack_tool_build_passed", prep_build["returncode"] == 0),
      check("both_native_programs_generated",
            generated and all(result["returncode"] == 0
                              for result in generation_results.values())),
      check("offline_prepack_and_validation_passed",
            prep_result["returncode"] == 0 and
            prep_probe.get("prepack_only") is True),
      check("all_prepacked_payload_sizes_match", prepack_files_ok),
      check("static_prepack_excludes_router_schedule",
            static_prepack_contract.get("schema_version") ==
                "iq36-grouped-s8-u4-static-prepack-v2" and
            static_prepack_contract.get("router_schedule") ==
                "dynamic_runtime_input"),
      check("standalone_runtime_build_passed",
            runtime_build["returncode"] == 0),
      check("engine_grouped_prefill_typed_api_smoke_passed",
            api_smoke_build["returncode"] == 0 and
            api_smoke_run["returncode"] == 0 and
            api_smoke_probe.get("dynamic_router_schedule") is True and
            api_smoke_probe.get("active_experts") == 222 and
            api_smoke_probe.get("max_group_size") == 361),
      check("resident_runtime_reuses_context_programs_layer_and_scratch",
            resident_smoke_build["returncode"] == 0 and
            resident_smoke_run["returncode"] == 0 and
            resident_smoke_probe.get("resident_reuse_pass") is True and
            resident_smoke_probe.get("context_create_count") == 1 and
            resident_smoke_probe.get("program_load_count") == 3 and
            resident_smoke_probe.get("layer_load_count") == 1 and
            resident_smoke_probe.get("layer_count") == 1 and
            resident_smoke_probe.get("run_count") == 2 and
            resident_smoke_probe.get("resident_weight_bytes") == 541065216 and
            resident_smoke_probe.get("deterministic_mismatch_count") == 0 and
            resident_smoke_probe.get("oracle_mismatch_count") == 0 and
            resident_smoke_probe.get("steady_state_cap_pass") is True and
            resident_smoke_probe.get("maps_native_only") is True),
      check("engine_core_and_parameterized_target_registered",
            "src/grouped_s8_u4_prefill_runtime.cpp" in cmake_text and
            "OpenCL::OpenCL" in cmake_text and
            any(isinstance(row, dict) and
                row.get("target") ==
                    "iq36-grouped-s8-u4-prefill-runtime" and
                row.get("source") ==
                    "tools/grouped_s8_u4_prefill_runtime.cpp"
                for row in infra_targets) and
            any(isinstance(row, dict) and
                row.get("target") ==
                    "iq36-grouped-s8-u4-prefill-resident-smoke" and
                row.get("source") ==
                    "tools/grouped_s8_u4_prefill_resident_smoke.cpp"
                for row in infra_targets)),
      check("standalone_runtime_links_no_onednn_or_openvino",
            ldd_result["returncode"] == 0 and "dnnl" not in ldd_lower and
            "openvino" not in ldd_lower),
      check("standalone_runtime_maps_no_onednn_or_openvino",
            all(probe.get("maps_native_only") is True for probe in probes)),
      check("arc_b390_selected",
            all("B390" in str(probe.get("device_name")) for probe in probes)),
      check("real_layer_shape_preserved",
            all(probe.get("assignment_count") == 8192 and
                probe.get("active_experts") == 222 and
                probe.get("max_group_size") == 361
                for probe in probes) and
            shape.get("active_expert_count") == 222),
      check("router_schedule_is_dynamic_runtime_input",
            all(probe.get("dynamic_router_schedule") is True
                for probe in probes) and
            all(not (prepack_dir / name).exists() for name in (
                "offsets.bin", "token-map.bin", "inverse-map.bin",
                "router-weights.bin"))),
      check("all_40_layer_router_shapes_use_one_parameterized_schedule",
            len(schedule_probes) == 40 and
            all(row["pass"] for row in schedule_probes) and
            max(int(row["max_group_size"]) for row in schedule_probes) == 994 and
            max(int(row["native_global_y"]) for row in schedule_probes) == 128),
      check("runtime_payload_is_resident_all_expert_carrier",
            all(probe.get("resident_weight_bytes") == 541_065_216
                for probe in probes)),
  ]
  for kind, rows in disassembly.items():
    assembly = rows["assembly"].lower()
    ze_info = rows["ze_info"].replace(" ", "").lower()
    evidence_checks += [
        check(f"{kind}_ocloc_disassembly_passed",
              rows["result"]["returncode"] == 0),
        check(f"{kind}_native_u4_by_s8_dpas_present",
              "dpas.8x8" in assembly and ":u4" in assembly and ":b" in assembly),
        check(f"{kind}_reports_dpas_256grf_simd16_wg32x4x1",
              "has_dpas:true" in ze_info and "grf_count:256" in ze_info and
              "simd_size:16" in ze_info and
              "required_work_group_size:[32,4,1]" in ze_info),
        check(f"{kind}_reports_no_spill_or_scratch",
              "spill" not in ze_info and "scratch" not in ze_info),
    ]

  correctness_checks = []
  for index, probe in enumerate(probes):
    suffix = "" if index == 0 else "_confirm"
    correctness_checks += [
        check(f"component_correctness_passed{suffix}",
              probe.get("correctness_pass") is True),
        *[dict(row, name=row["name"] + suffix)
          for row in compare_checks(
              "swiglu", probe.get("compare", {}), 4_194_304)],
        *[dict(row, name=row["name"] + suffix)
          for row in compare_checks(
              "weighted_down", probe.get("weighted_down_compare", {}),
              16_777_216)],
        *[dict(row, name=row["name"] + suffix)
          for row in compare_checks(
              "routed_output", probe.get("moe_compare", {}), 2_097_152)],
    ]

  medians = [float(probe.get("median_us", float("inf"))) for probe in probes]
  minimums = [float(probe.get("minimum_us", float("inf"))) for probe in probes]
  complete_minimums = [
      float(probe.get("complete_minimum_us", float("inf")))
      for probe in probes]
  paired_spread = (
      abs(medians[0] - medians[1]) / min(medians)
      if len(medians) == 2 and min(medians) > 0 else float("inf"))
  performance_checks = [
      check("primary_complete_runtime_below_cap",
            probes[0].get("performance_pass") is True and
            complete_minimums[0] <= kernel_cap_us,
            kernel_minimum_us=minimums[0],
            complete_minimum_us=complete_minimums[0],
            kernel_cap_us=kernel_cap_us),
      check("confirm_complete_runtime_below_cap",
            probes[1].get("performance_pass") is True and
            complete_minimums[1] <= kernel_cap_us,
            kernel_minimum_us=minimums[1],
            complete_minimum_us=complete_minimums[1],
            kernel_cap_us=kernel_cap_us),
      check("paired_median_spread_within_frontier_p90_noise",
            paired_spread <= 0.03, paired_spread=paired_spread,
            required_max=0.03),
  ]
  evidence_passed = all(row["pass"] for row in evidence_checks)
  correctness_passed = all(row["pass"] for row in correctness_checks)
  performance_passed = all(row["pass"] for row in performance_checks)
  required_passed = evidence_passed and correctness_passed and performance_passed

  result = {
      "budget": budget,
      "case_id": BASE.CASE_ID,
      "checks": evidence_checks + correctness_checks + performance_checks,
      "correctness_checks_passed": correctness_passed,
      "created_at": created_at,
      "disposition": (
          "accept_full_tensor_grouped_s8_u4_compact_affine_f16_contribution"
          if required_passed else
          "reject_or_repair_grouped_s8_u4_prefill_carrier"),
      "evidence_checks_passed": evidence_passed,
      "git": repo_git,
      "layer": BASE.LAYER,
      "native_binaries": binary_manifest,
      "paired_median_spread": paired_spread,
      "performance_checks_passed": performance_passed,
      "prepack_manifest": prepack_manifest,
      "probe": probes[0],
      "confirm_probe": probes[1],
      "required_checks_passed": required_passed,
      "runtime_boundary": {
          "excluded": [
              "one-time model parse and all-expert resident prepack",
              "offline oneDNN native-program generation",
          ],
          "included": [
              "dynamic CPU expert-order schedule construction and OpenCL upload",
              "dynamic token Q8 quantization and expert-order gather",
              "grouped S8-by-U4 gate/up with compact exact affine repair",
              "SwiGLU and F16 intermediate carrier",
              "dynamic intermediate Q8 quantization",
              "grouped S8-by-U4 down with compact affine repair and router weight",
              "F16 contribution carrier and deterministic F32 scatter",
              "all submissions and final queue drain",
          ],
          "runtime_dependencies": ["OpenCL"],
      },
      "schema_version": SCHEMA_VERSION,
      "schedule_probes": schedule_probes,
      "source_patch": {
          "path": str(ONEDNN_PATCH.relative_to(ROOT)),
          "sha256": BASE.sha256_file(ONEDNN_PATCH),
      },
  }
  BASE.write_json(out_dir / "result.json", result)
  BASE.write_json(out_dir / "metrics.json", {
      "kernel_cap_us": kernel_cap_us,
      "primary_minimum_us": minimums[0],
      "confirm_minimum_us": minimums[1],
      "primary_complete_minimum_us": complete_minimums[0],
      "confirm_complete_minimum_us": complete_minimums[1],
      "primary_median_us": medians[0],
      "confirm_median_us": medians[1],
      "paired_median_spread": paired_spread,
      "primary_headroom_us": kernel_cap_us - complete_minimums[0],
      "confirm_headroom_us": kernel_cap_us - complete_minimums[1],
  })
  summary = [
      "# Grouped S8-by-U4 prefill component gate",
      "",
      f"- Required checks passed: `{str(required_passed).lower()}`",
      f"- Primary minimum: `{minimums[0]:.3f} us`",
      f"- Confirm minimum: `{minimums[1]:.3f} us`",
      f"- Primary complete incl. dynamic schedule: "
      f"`{complete_minimums[0]:.3f} us`",
      f"- Confirm complete incl. dynamic schedule: "
      f"`{complete_minimums[1]:.3f} us`",
      f"- Component cap: `{kernel_cap_us:.3f} us`",
      f"- Paired median spread: `{paired_spread * 100.0:.3f}%`",
      "- Runtime dependency boundary: OpenCL only; oneDNN is offline generation.",
      "- Product speedup claim: not made; this is one layer-27 prerequisite.",
      "",
  ]
  (out_dir / "summary.md").write_text("\n".join(summary), encoding="utf-8")
  print(json.dumps({
      "out_dir": str(out_dir),
      "primary_minimum_us": minimums[0],
      "confirm_minimum_us": minimums[1],
      "primary_complete_minimum_us": complete_minimums[0],
      "confirm_complete_minimum_us": complete_minimums[1],
      "kernel_cap_us": kernel_cap_us,
      "required_checks_passed": required_passed,
  }, sort_keys=True))
  return 0 if required_passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
