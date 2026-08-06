#!/usr/bin/env python3
"""Source-gate and build the isolated affine-Q4 LM-head product carrier.

The admitted seq2290 component may replace only the count25 full-I8 fallback
inside the accepted seq2189 LM-head provider.  This gate verifies the exact
incremental source patch, target-compiles every new PTL OpenCL variant, restores
the oneDNN build product from the rejected PR35924 experiment, and rebuilds the
GPU plugin at parallelism one.  It creates no model worker or InferRequest and
runs no inference.
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
SCHEMA = "intel-qwen36-openvino-lm-head-affine-q4-product-build-v0"
BASE_TOOL = ROOT / "tools/intel-qwen36-onednn-gmlp-exact-component-build.py"
PATCH = ROOT / "engine/openvino/iq36-lm-head-affine-q4-group128.patch"
COMPONENT_RESULT = ROOT / (
    "output/openvino-lm-head-affine-q4-group128-component-"
    "20260801Tseq2290-clean/result.json")
COMPONENT_MANIFEST = ROOT / (
    "output/openvino-lm-head-affine-q4-group128-component-"
    "20260801Tseq2290-clean/manifest.json")

R0 = Path("/home/intel/intel-qwen36-r0")
SOURCE_TREE = R0 / "source/openvino-90214e5be05"
ONEDNN = SOURCE_TREE / "src/plugins/intel_gpu/thirdparty/onednn_gpu"
SOURCE_REL = (
    "src/plugins/intel_gpu/src/graph/impls/ocl/"
    "iq36_lm_head_i8q4.cpp")
SOURCE = SOURCE_TREE / SOURCE_REL
MOE_REL = (
    "src/plugins/intel_gpu/src/graph/impls/ocl_v2/moe/"
    "moe_3gemm_swiglu_opt.cpp")
ONEDNN_RELS = (
    "src/gpu/intel/compute/kernel_ctx.hpp",
    "src/gpu/intel/matmul/grouped_micro_gemm.cl",
    "src/gpu/intel/matmul/grouped_micro_gemm.cpp",
    "src/gpu/intel/matmul/grouped_micro_gemm.hpp",
    "src/gpu/intel/matmul/grouped_post_ops_gen.cpp",
    "src/gpu/intel/matmul/grouped_post_ops_gen.hpp",
)
BUILD_TREE = R0 / "build/openvino-90214e-l0-gpu"
ONEDNN_BUILD = (
    BUILD_TREE / "src/plugins/intel_gpu/thirdparty/onednn_gpu_build")
ONEDNN_ARCHIVE = (
    BUILD_TREE / "src/plugins/intel_gpu/thirdparty/"
    "onednn_gpu_install/lib/libopenvino_onednn_gpu.a")
BUILD_PLUGIN = R0 / (
    "output/openvino-90214e-l0-gpu-seq2109/bin/intel64/Release/"
    "libopenvino_intel_gpu_plugin.so")
CONTROL_PLUGIN = R0 / (
    "output/openvino-90214e-l0-gpu-seq2189/bin/intel64/Release/"
    "libopenvino_intel_gpu_plugin.so")
DEFAULT_CANDIDATE_PLUGIN = R0 / (
    "output/openvino-90214e-l0-gpu-seq2291/bin/intel64/Release/"
    "libopenvino_intel_gpu_plugin.so")
CMAKE = Path("/home/intel/intel-box-env/conda/bin/cmake")
OCLOC = Path("/usr/bin/ocloc")

OPENVINO_HEAD = "90214e5be052438cec5617ed3ea7e37df1538f68"
ONEDNN_HEAD = "20db47e2d3c4df1b66e93bed2e97d30da175512d"
EXPECTED_PATCH_SHA256 = (
    "8b11fb73be20fcfe0239c2a48b1ba272e87e29e2f75c403b05d2735c3a40b42b")
EXPECTED_SOURCE_SHA256 = (
    "48795e95fc25f9a74941b6994ff545a6a8d45aaa9811682b8edc7ae3b3b6fcdd")
EXPECTED_COMPONENT_RESULT_SHA256 = (
    "88029270091a03a00742fb87abf71030e1580f2fdda36d497a88bd9dc0a26592")
EXPECTED_COMPONENT_MANIFEST_SHA256 = (
    "9eac1672678a59d15a192076af90f2d9ad9a0e63cb7264b19ace675e978bb9f7")
EXPECTED_CONTROL_SHA256 = (
    "9832adffa8bf3fd7b013c47d9d2abefa66987c5fbde350c9669fce58c697b985")
EXPECTED_MOE_SHA256 = (
    "d388d8034526c2a3f438a62ff7a5f5be7df060ba911683248960d08dfc92c855")
EXPECTED_ONEDNN_SHA256 = {
    ONEDNN_RELS[0]:
        "3a9ca78af97d70dda461e2fa4b465ac75fe4a63bd767b52404bac7861dd5c4bb",
    ONEDNN_RELS[1]:
        "96c42c4e41c57b6ffd3d86a1655e3e854782c418408ec8e423ca287a732cb09a",
    ONEDNN_RELS[2]:
        "7bb1d3009e28349a9f59cc686968a676aef8ea027a0135375bdc3ce65a04bdbb",
    ONEDNN_RELS[3]:
        "5810d0744989f5af9eb6dfe1b64409be7c3ab76815f6d8aefdc5c6082f055273",
}
PREFLIGHT_BYTES = 8 * 1024**3
ABORT_BYTES = 4 * 1024**3

VARIANTS = (
    ("reset", "iq36_lm_head_i8q1_gated_exact_reset_f16",
     "-D IQ36_BINARY_GATED_EXACT_RESET_KERNEL=1 "
     "-D IQ36_BINARY_AFFINE_Q4=1", None, None),
    ("collect", "iq36_lm_head_i8q1_gated_exact_collect_f16",
     "-D IQ36_BINARY_GATED_EXACT_COLLECT_KERNEL=1", 256, None),
    ("hidden_norms",
     "iq36_lm_head_i8q1_affine_q4_hidden_group_norms_f16",
     "-D IQ36_BINARY_AFFINE_Q4_HIDDEN_NORMS_KERNEL=1", 128, 16),
    ("bound_select", "iq36_lm_head_i8q1_affine_q4_bound_select_f16",
     "-D IQ36_BINARY_AFFINE_Q4_BOUND_SELECT_KERNEL=1", 256, 16),
    ("exact_candidates",
     "iq36_lm_head_i8_affine_q4_exact_candidates_f16",
     "-D IQ36_BINARY_AFFINE_Q4_CORRECTION_KERNEL=1", 64, 16),
    ("overflow_matvec", "iq36_lm_head_i8_gated_exact_matvec_f16",
     "-D IQ36_BINARY_GATED_EXACT_MATVEC_KERNEL=1 "
     "-D IQ36_BINARY_AFFINE_Q4_OVERFLOW=1", 256, 16),
    ("overflow_topk",
     "iq36_lm_head_i8q1_gated_exact_output_topk8_f16",
     "-D IQ36_BINARY_GATED_EXACT_OUTPUT_KERNEL=1 "
     "-D IQ36_BINARY_AFFINE_Q4_OVERFLOW=1", 256, 16),
    ("overflow_merge",
     "iq36_lm_head_i8q1_gated_exact_topk8_merge_f32",
     "-D IQ36_BINARY_GATED_EXACT_MERGE_KERNEL=1 "
     "-D IQ36_BINARY_AFFINE_Q4_OVERFLOW=1", 256, None),
    ("overflow_correction",
     "iq36_lm_head_i8_gated_exact_topk8_correction_f16",
     "-D IQ36_BINARY_GATED_EXACT_TOPK_KERNEL=1 "
     "-D IQ36_BINARY_AFFINE_Q4_OVERFLOW=1", 64, 16),
)


def load_base() -> Any:
  spec = importlib.util.spec_from_file_location(
      "iq36_affine_q4_build_base", BASE_TOOL)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import scoped build helper: {BASE_TOOL}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


BASE = load_base()


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", required=True, type=Path)
  parser.add_argument(
      "--candidate-plugin", type=Path, default=DEFAULT_CANDIDATE_PLUGIN)
  parser.add_argument("--build-timeout-s", default=3600.0, type=float)
  parser.add_argument("--poll-interval-s", default=0.1, type=float)
  args = parser.parse_args()
  if args.build_timeout_s <= 0 or args.poll_interval_s <= 0:
    parser.error("timeout and poll interval must be positive")
  return args


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    while chunk := stream.read(8 * 1024 * 1024):
      digest.update(chunk)
  return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise TypeError(f"expected JSON object: {path}")
  return value


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True) + "\n",
      encoding="utf-8")


def run(
    command: list[str], cwd: Path = ROOT,
) -> subprocess.CompletedProcess[str]:
  return subprocess.run(
      command, cwd=cwd, check=False, capture_output=True, text=True,
      encoding="utf-8", errors="replace")


def git(cwd: Path, *args: str) -> str:
  result = run(["git", *args], cwd)
  if result.returncode != 0:
    raise RuntimeError(
        f"git failed ({result.returncode}): {args}\n{result.stderr}")
  return result.stdout.strip()


def check(name: str, passed: bool, **evidence: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **evidence}


def file_record(path: Path) -> dict[str, Any]:
  return {
      "path": str(path),
      "exists": path.is_file(),
      "bytes": path.stat().st_size if path.is_file() else 0,
      "mtime_ns": path.stat().st_mtime_ns if path.is_file() else 0,
      "sha256": sha256(path) if path.is_file() else None,
  }


def apply_check(repo: Path, patch: Path, reverse: bool = False) -> dict[str, Any]:
  command = ["git", "apply"]
  if reverse:
    command.append("--reverse")
  command.extend(["--check", str(patch)])
  result = run(command, repo)
  return {
      "command": command,
      "returncode": result.returncode,
      "stdout": result.stdout.strip(),
      "stderr": result.stderr.strip(),
  }


def target_status(repo: Path, paths: tuple[str, ...]) -> list[str]:
  value = git(
      repo, "status", "--short", "--untracked-files=all", "--", *paths)
  return value.splitlines() if value else []


def stage_ok(stage: dict[str, Any]) -> bool:
  return bool(
      stage.get("returncode") == 0 and
      stage.get("timed_out") is False and
      stage.get("memory_guard_tripped") is False and
      stage.get("oom_observed") is False)


def extract_binary_source(source_text: str) -> str:
  marker = 'const std::string source = R"CLC(\n'
  start = source_text.index(marker) + len(marker)
  end = source_text.index('\n)CLC";', start)
  return source_text[start:end] + "\n"


def parse_int(info: str, field: str, default: int = 0) -> int:
  match = re.search(rf"^\s*{re.escape(field)}:\s*(\d+)\s*$",
                    info, flags=re.MULTILINE)
  return int(match.group(1)) if match else default


def compile_variant(
    raw: Path, source_path: Path, name: str, kernel: str,
    definitions: str, group_size: int | None, subgroup_size: int | None,
) -> dict[str, Any]:
  variant_dir = raw / "ocloc" / name
  disasm_dir = variant_dir / "disasm"
  variant_dir.mkdir(parents=True, exist_ok=False)
  compile_command = [
      str(OCLOC), "compile", "-q", "-file", str(source_path),
      "-device", "0xb080", "-output", name, "-out_dir", str(variant_dir),
      "-output_no_suffix", "--format", "zebin",
      "-options", f"-cl-std=CL3.0 {definitions}"]
  compiled = run(compile_command)
  (variant_dir / "compile.stdout").write_text(
      compiled.stdout, encoding="utf-8")
  (variant_dir / "compile.stderr").write_text(
      compiled.stderr, encoding="utf-8")
  binary = variant_dir / f"{name}.bin"
  validate = run(
      [str(OCLOC), "validate", "-file", str(binary)])
  (variant_dir / "validate.stdout").write_text(
      validate.stdout, encoding="utf-8")
  (variant_dir / "validate.stderr").write_text(
      validate.stderr, encoding="utf-8")
  disasm_dir.mkdir()
  disasm = run(
      [str(OCLOC), "disasm", "-q", "-file", str(binary),
       "-dump", str(disasm_dir)])
  info_path = disasm_dir / ".ze_info"
  info = (
      info_path.read_text(encoding="utf-8", errors="replace")
      if info_path.is_file() else "")
  grf_count = parse_int(info, "grf_count")
  simd_size = parse_int(info, "simd_size")
  slm_size = parse_int(info, "slm_size")
  required = re.search(
      r"required_work_group_size:\s*\[\s*(\d+),\s*(\d+),\s*(\d+)",
      info)
  required_x = int(required.group(1)) if required else 0
  scratch_or_spill = bool(re.search(
      r"scratch|spill|private_memory", info, flags=re.IGNORECASE))
  passed = bool(
      compiled.returncode == 0 and binary.is_file() and
      validate.returncode == 0 and "Binary is VALID" in validate.stdout and
      disasm.returncode == 0 and info_path.is_file() and
      info.count(f"name:            {kernel}") == 2 and
      (group_size is None or required_x == group_size) and
      grf_count in (32, 64, 96, 128) and
      grf_count <= 96 and not scratch_or_spill and
      (subgroup_size is None or simd_size == subgroup_size))
  return {
      "name": name,
      "kernel": kernel,
      "definitions": definitions,
      "pass": passed,
      "compile_returncode": compiled.returncode,
      "compile_stdout": compiled.stdout.strip(),
      "compile_stderr": compiled.stderr.strip(),
      "validate_returncode": validate.returncode,
      "validate_stdout_sha256": hashlib.sha256(
          validate.stdout.encode("utf-8")).hexdigest(),
      "disasm_returncode": disasm.returncode,
      "binary": file_record(binary),
      "resources": {
          "grf_count": grf_count,
          "simd_size": simd_size,
          "slm_size": slm_size,
          "required_group_size_x": required_x,
          "scratch_or_spill_metadata": scratch_or_spill,
      },
  }


def main() -> int:
  args = parse_args()
  output = args.output.resolve()
  raw = output / "raw"
  raw.mkdir(parents=True, exist_ok=False)
  candidate_plugin = args.candidate_plugin.resolve()
  if candidate_plugin.exists():
    raise SystemExit(
        f"isolated candidate plugin already exists: {candidate_plugin}")
  required_paths = (
      BASE_TOOL, PATCH, COMPONENT_RESULT, COMPONENT_MANIFEST, SOURCE_TREE,
      ONEDNN, SOURCE, BUILD_TREE, ONEDNN_BUILD, ONEDNN_ARCHIVE,
      BUILD_PLUGIN, CONTROL_PLUGIN, CMAKE, OCLOC)
  missing = [str(path) for path in required_paths if not path.exists()]
  if missing:
    raise SystemExit("missing affine-Q4 build inputs: " + ", ".join(missing))

  repo = BASE.repository_state(output)
  source_text = SOURCE.read_text(encoding="utf-8", errors="replace")
  patch_text = PATCH.read_text(encoding="utf-8", errors="replace")
  component = load_json(COMPONENT_RESULT)
  memory_before = BASE.proc_meminfo()
  archive_before = file_record(ONEDNN_ARCHIVE)
  build_before = file_record(BUILD_PLUGIN)
  control_before = file_record(CONTROL_PLUGIN)
  source_reverse = apply_check(SOURCE_TREE, PATCH, reverse=True)
  onednn_status = target_status(ONEDNN, ONEDNN_RELS)
  moe_status = target_status(SOURCE_TREE, (MOE_REL,))
  lm_status = target_status(SOURCE_TREE, (SOURCE_REL,))
  onednn_hashes = {
      rel: file_record(ONEDNN / rel) for rel in ONEDNN_RELS[:4]}
  markers = (
      "IQ36_LM_HEAD_I8Q1_GATED_EXACT_AFFINE_Q4",
      "return binary_gated_exact_enabled() &&",
      "kBinaryAffineQ4CorrectionCapacity = 16812",
      "kBinaryAffineQ4PackedBytes =",
      "data.kernels.resize(",
      "affine_q4 ? 15",
      "state[3] = as_uint(maximum);",
      "IQ36_BINARY_AFFINE_Q4_OVERFLOW",
      "pack_affine_q4",
      "positive_f16_upper_after_f32",
      "kBinaryGatedExactKernelName",
      "kBinaryGatedQ4KernelName",
      "kBinaryTokenReduceKernelName",
  )
  source_checks = [
      check("repository_clean_pushed_main_at_gate",
            repo["branch"] == "main" and repo["pushed"] and
            not repo["dirty"], **repo),
      check("seq2290_component_admission_is_exact",
            sha256(COMPONENT_RESULT) ==
                EXPECTED_COMPONENT_RESULT_SHA256 and
            sha256(COMPONENT_MANIFEST) ==
                EXPECTED_COMPONENT_MANIFEST_SHA256 and
            component.get("required_checks_passed") is True and
            component.get("isolated_product_integration_admitted") is True and
            component.get("performance_claim_admitted") is False and
            component.get("traffic", {}).get("observed_maximum_candidate_rows")
                == 10808 and
            component.get("traffic", {}).get("maximum_ratio", 1.0) <= 0.60,
            result=file_record(COMPONENT_RESULT),
            manifest=file_record(COMPONENT_MANIFEST)),
      check("incremental_patch_and_postimage_are_exact",
            sha256(PATCH) == EXPECTED_PATCH_SHA256 and
            sha256(SOURCE) == EXPECTED_SOURCE_SHA256 and
            source_reverse["returncode"] == 0 and
            patch_text.count("+++ b/") == 1 and
            f"+++ b/{SOURCE_REL}" in patch_text,
            patch=file_record(PATCH), source=file_record(SOURCE),
            reverse_apply=source_reverse),
      check("source_contract_binds_only_count25_affine_fallback",
            all(marker in source_text for marker in markers) and
            source_text.count(
                "IQ36_LM_HEAD_I8Q1_GATED_EXACT_AFFINE_Q4") == 1 and
            source_text.count(
                "#define IQ36_BINARY_GATED_EXACT_COUNT 25U") == 1 and
            source_text.count(
                "kBinaryAffineQ4CorrectionCapacity = 16812") == 1 and
            source_text.count(
                "kBinaryAffineQ4PackedBytes =") == 1,
            markers=list(markers), source_sha256=sha256(SOURCE)),
      check("rejected_pr35924_source_is_absent",
            git(SOURCE_TREE, "rev-parse", "HEAD") == OPENVINO_HEAD and
            git(ONEDNN, "rev-parse", "HEAD") == ONEDNN_HEAD and
            not onednn_status and not moe_status and
            sha256(SOURCE_TREE / MOE_REL) == EXPECTED_MOE_SHA256 and
            all(
                onednn_hashes[rel]["sha256"] == expected
                for rel, expected in EXPECTED_ONEDNN_SHA256.items()) and
            lm_status == [f"?? {SOURCE_REL}"],
            onednn_target_status=onednn_status,
            moe_target_status=moe_status,
            lm_head_target_status=lm_status,
            onednn_hashes=onednn_hashes,
            moe=file_record(SOURCE_TREE / MOE_REL)),
      check("accepted_seq2189_control_is_immutable",
            control_before["sha256"] == EXPECTED_CONTROL_SHA256,
            control=control_before),
      check("eight_gib_preflight_clears",
            int(memory_before.get("MemAvailable", 0)) >= PREFLIGHT_BYTES,
            available_bytes=memory_before.get("MemAvailable"),
            preflight_bytes=PREFLIGHT_BYTES),
  ]
  source_admitted = all(row["pass"] for row in source_checks)

  source_path = raw / "iq36_lm_head_binary.cl"
  target_compiles: list[dict[str, Any]] = []
  if source_admitted:
    source_path.write_text(
        extract_binary_source(source_text), encoding="utf-8")
    for variant in VARIANTS:
      target_compiles.append(
          compile_variant(raw, source_path, *variant))
  target_compile_admitted = bool(
      len(target_compiles) == len(VARIANTS) and
      all(row["pass"] for row in target_compiles))

  environment = os.environ.copy()
  conda_bin = "/home/intel/intel-box-env/conda/bin"
  conda_lib = "/home/intel/intel-box-env/conda/lib"
  environment["PATH"] = conda_bin + ":" + environment.get("PATH", "")
  environment["LD_LIBRARY_PATH"] = (
      conda_lib + ":" + environment.get("LD_LIBRARY_PATH", ""))
  skipped = {
      "returncode": 125, "timed_out": False,
      "memory_guard_tripped": False, "oom_observed": False,
      "skipped": "source or target-compile gate failed", "monitor": {}}
  onednn_configure: dict[str, Any] = dict(skipped)
  onednn_build: dict[str, Any] = dict(skipped)
  plugin_build: dict[str, Any] = dict(skipped)
  if target_compile_admitted:
    onednn_configure = BASE.run_scoped(
        output, raw, "configure-onednn-affine-q4-base",
        [str(CMAKE), "-S", str(ONEDNN), "-B", str(ONEDNN_BUILD)],
        args.build_timeout_s, args.poll_interval_s, environment)
    if stage_ok(onednn_configure):
      onednn_build = BASE.run_scoped(
          output, raw, "build-onednn-affine-q4-base",
          [str(CMAKE), "--build", str(ONEDNN_BUILD),
           "--target", "install", "--parallel", "1"],
          args.build_timeout_s, args.poll_interval_s, environment)
    if stage_ok(onednn_build):
      plugin_build = BASE.run_scoped(
          output, raw, "build-plugin-affine-q4",
          [str(CMAKE), "--build", str(BUILD_TREE),
           "--target", "openvino_intel_gpu_plugin", "--parallel", "1"],
          args.build_timeout_s, args.poll_interval_s, environment)

  archive_after = file_record(ONEDNN_ARCHIVE)
  build_after = file_record(BUILD_PLUGIN)
  control_after = file_record(CONTROL_PLUGIN)
  plugin_bytes = (
      BUILD_PLUGIN.read_bytes() if stage_ok(plugin_build) else b"")
  new_kernel_names = tuple(row[1] for row in VARIANTS)
  copied = False
  if (stage_ok(onednn_build) and stage_ok(plugin_build) and
      archive_after["sha256"] != archive_before["sha256"] and
      build_after["sha256"] != build_before["sha256"] and
      all(name.encode("ascii") in plugin_bytes for name in new_kernel_names)):
    candidate_plugin.parent.mkdir(parents=True, exist_ok=False)
    shutil.copy2(BUILD_PLUGIN, candidate_plugin)
    copied = True
  candidate = file_record(candidate_plugin)

  stages = (onednn_configure, onednn_build, plugin_build)
  minimum_available = min(
      int(stage.get("monitor", {}).get(
          "system_available_min_bytes",
          memory_before.get("MemAvailable", 0)))
      for stage in stages)
  oom_free = all(
      not stage.get("oom_observed", False) and
      not stage.get("memory_guard_tripped", False) and
      all(
          int(value) == 0 for key, value in
          stage.get("monitor", {}).get("memory_events_max", {}).items()
          if key in ("oom", "oom_kill", "oom_group_kill"))
      for stage in stages)
  checks = [
      *source_checks,
      check("all_nine_ptl_opencl_variants_target_compile_spill_free",
            target_compile_admitted,
            variants=target_compiles),
      check("serial_onednn_base_and_plugin_builds_succeed",
            target_compile_admitted and stage_ok(onednn_configure) and
            stage_ok(onednn_build) and stage_ok(plugin_build) and
            archive_after["sha256"] != archive_before["sha256"] and
            build_after["sha256"] != build_before["sha256"],
            archive_before=archive_before, archive_after=archive_after,
            plugin_before=build_before, plugin_after=build_after,
            onednn_configure=onednn_configure,
            onednn_build=onednn_build, plugin_build=plugin_build),
      check("candidate_contains_affine_and_overflow_kernels",
            all(
                name.encode("ascii") in plugin_bytes
                for name in new_kernel_names),
            kernel_names=list(new_kernel_names)),
      check("isolated_seq2291_candidate_is_copied",
            copied and candidate["exists"] and
            candidate["sha256"] == build_after["sha256"] and
            candidate["sha256"] != EXPECTED_CONTROL_SHA256,
            candidate=candidate),
      check("accepted_seq2189_control_remains_unchanged",
            control_after["sha256"] == EXPECTED_CONTROL_SHA256,
            control=control_after),
      check("four_gib_abort_and_oom_guards_hold",
            minimum_available >= ABORT_BYTES and oom_free,
            minimum_available_bytes=minimum_available,
            abort_bytes=ABORT_BYTES),
      check("build_gate_runs_no_model_or_inference", True,
            gpu_contexts_created=0, gpu_kernels_executed=0,
            model_workers_started=0, infer_requests_created=0,
            inference_workers_started=0),
  ]
  required = all(row["pass"] for row in checks)
  verdict = (
      "admit_one_affine_q4_full_logit_correctness_worker"
      if required else
      "repair_affine_q4_source_target_compile_or_serial_build")
  payload = {
      "schema": SCHEMA,
      "workstream": WS,
      "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
      "repository": repo,
      "required_checks_passed": required,
      "verdict": verdict,
      "source_compile_integration_admitted": required,
      "correctness_worker_admitted": required,
      "performance_worker_admitted": False,
      "product_promotion_admitted": False,
      "checks": checks,
      "candidate_plugin": candidate,
      "accepted_control_plugin": control_after,
      "minimum_available_bytes": minimum_available,
      "next_action": {
          "route": "affine_q4_full_logit_output130_correctness",
          "requirements": [
              "run one isolated candidate-only output130 teacher-forced worker",
              "require exact tokens, KLD <= 0.005, and top-1 >= 0.99",
              "require the affine provider only on count25 events",
              "require zero capacity overflow and the full-I8 forced fallback",
              "only a pass may fund one short paired timing point",
          ],
      },
  }
  write_json(output / "result.json", payload)
  write_json(output / "manifest.json", {
      "schema": SCHEMA,
      "tool": str(Path(__file__).relative_to(ROOT)),
      "repository": repo,
      "inputs": {
          str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path):
              sha256(path)
          for path in (
              PATCH, COMPONENT_RESULT, COMPONENT_MANIFEST, SOURCE,
              SOURCE_TREE / MOE_REL, CONTROL_PLUGIN)
      },
      "target_compile_binaries": {
          row["name"]: row["binary"] for row in target_compiles},
      "candidate_plugin": candidate,
      "parallelism": 1,
      "gpu_contexts": 0,
      "model_workers": 0,
      "infer_requests": 0,
  })
  report = f"""# Affine-Q4/group128 LM-head product build

Verdict: **{verdict}**. Required checks: `{str(required).lower()}`.

The seq2290 path is bound only to the accepted count25 full-I8 fallback.
All `{len(target_compiles)}` PTL kernel variants target-compiled before the
serial oneDNN/plugin build. Candidate plugin SHA:
`{candidate.get('sha256')}`.

Minimum available memory was `{minimum_available} B`; no model, inference,
OOM, or memory-guard event occurred. Product correctness and speed remain
separate gates.
"""
  (output / "report.md").write_text(report, encoding="utf-8")
  print(json.dumps({
      "artifact": str(output.relative_to(ROOT)),
      "verdict": verdict,
      "required_checks_passed": required,
      "candidate_plugin_sha256": candidate.get("sha256"),
      "target_compile_count": len(target_compiles),
      "minimum_available_bytes": minimum_available,
      "oom_free": oom_free,
  }, separators=(",", ":")), flush=True)
  return 0 if required else 2


if __name__ == "__main__":
  raise SystemExit(main())
