#!/usr/bin/env python3
"""Compile the sole exact-attention 48-subgroup resource probe.

The gate compiles one pinned 128-GRF OpenCL program and queries its resources.
It never enqueues the kernel, builds a plugin, or launches a model worker.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = (
    "intel-qwen36-openvino-exact-attention-triple-cohort-codegen-gate-v1")
OFFICIAL_PREFETCH_SCHEMA = (
    "intel-qwen36-openvino-exact-attention-"
    "triple-official-prefetch-codegen-gate-v1")
CODEGEN = ROOT / "engine/tools/openvino_moe_micro_codegen.cpp"
HOST_SOURCE = ROOT / "engine/gpu/opencl/exact_score_staging_component.cl"
SHIMS = ROOT / "engine/openvino/custom/iq36_decode_microkernel_shims.cl"
TARGET_CONTRACT = ROOT / "contracts/intel-qwen36-target-contract.json"
COMPONENT_BOUND = ROOT / (
    "output/openvino-exact-attention-three-stage-component-"
    "20260724Tseq2144-clean/result.json")
OFFICIAL_PREFETCH_BOUND = ROOT / (
    "output/openvino-exact-attention-k-prefetch-address-bound-"
    "20260724Tseq2152-clean/result.json")
PINNED_SOURCE = Path(
    "/home/intel/intel-qwen36-r0/source/openvino-90214e5be05/"
    "src/plugins/intel_gpu/thirdparty/onednn_gpu")
PINNED_BUILD = Path(
    "/home/intel/intel-qwen36-r0/build/onednn-20db-micro-static")
CXX = Path("/home/intel/intel-box-env/conda/bin/c++")
ENV_SCRIPT = Path(
    "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh")
PINNED_COMMIT = "20db47e2d3c4df1b66e93bed2e97d30da175512d"

KERNEL = "iq36_exact_score_triple_cohort"
HOST_DEFINE = "IQ36_COMPONENT_PROGRAM=8"
OFFICIAL_PREFETCH_HOST_DEFINE = "IQ36_COMPONENT_PROGRAM=11"
REGISTER_FILE_SIZE = 128
EXPECTED_SUBGROUPS = 48
EXPECTED_WORKGROUP_ITEMS = 768
MAX_LOCAL_MEMORY_BYTES = 65_536
MIN_LOCAL_MEMORY_BYTES = 61_441
MAX_REGISTER_COUNT = 128


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out-dir", type=Path, required=True)
  parser.add_argument("--timeout-s", type=int, default=600)
  parser.add_argument("--memory-preflight-gib", type=float, default=8.0)
  parser.add_argument("--memory-stop-gib", type=float, default=4.0)
  parser.add_argument(
      "--official-prefetch", action="store_true",
      help="compile only the seq2152 official-address prefetch correction")
  args = parser.parse_args()
  if args.timeout_s <= 0:
    parser.error("--timeout-s must be positive")
  if args.memory_preflight_gib < 8.0:
    parser.error("--memory-preflight-gib must be at least 8")
  if args.memory_stop_gib < 4.0:
    parser.error("--memory-stop-gib must be at least 4")
  if args.memory_preflight_gib <= args.memory_stop_gib:
    parser.error("--memory-preflight-gib must exceed --memory-stop-gib")
  return args


def run(
    command: list[str], timeout: int,
) -> subprocess.CompletedProcess[str]:
  try:
    return subprocess.run(
        command, cwd=ROOT, check=False, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=timeout)
  except subprocess.TimeoutExpired as error:
    stdout = (
        error.stdout.decode("utf-8", "replace")
        if isinstance(error.stdout, bytes) else error.stdout or "")
    stderr = (
        error.stderr.decode("utf-8", "replace")
        if isinstance(error.stderr, bytes) else error.stderr or "")
    return subprocess.CompletedProcess(
        command, 124, stdout, stderr + "\nworker timeout")


def activated_timed(
    command: list[str], timeout: int, time_path: Path,
) -> subprocess.CompletedProcess[str]:
  shell_command = (
      f"source {shlex.quote(str(ENV_SCRIPT))} >/dev/null 2>&1 && "
      f"/usr/bin/time -v -o {shlex.quote(str(time_path))} " +
      " ".join(shlex.quote(part) for part in command))
  return run(["bash", "-lc", shell_command], timeout)


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True) + "\n",
      encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise TypeError(f"expected JSON object: {path}")
  return value


def parse_last_json(stdout: str) -> dict[str, Any]:
  for line in reversed(stdout.splitlines()):
    try:
      value = json.loads(line)
    except json.JSONDecodeError:
      continue
    if isinstance(value, dict):
      return value
  return {}


def parse_time(path: Path) -> dict[str, Any]:
  text = path.read_text(encoding="utf-8") if path.is_file() else ""
  patterns = {
      "maximum_resident_kib": r"Maximum resident set size \(kbytes\): (\d+)",
      "major_page_faults": r"Major \(requiring I/O\) page faults: (\d+)",
      "swaps": r"Swaps: (\d+)",
      "elapsed": r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (.+)",
  }
  result: dict[str, Any] = {"raw": text}
  for key, pattern in patterns.items():
    match = re.search(pattern, text)
    if match:
      result[key] = (
          match.group(1) if key == "elapsed" else int(match.group(1)))
  return result


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    while chunk := stream.read(1024 * 1024):
      digest.update(chunk)
  return digest.hexdigest()


def display(path: Path) -> str:
  try:
    return str(path.relative_to(ROOT))
  except ValueError:
    return str(path)


def available_memory_bytes() -> int:
  for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
    if line.startswith("MemAvailable:"):
      return int(line.split()[1]) * 1024
  raise RuntimeError("MemAvailable is missing")


def sample_memory(
    label: str, minimum_bytes: int, rows: list[dict[str, Any]],
) -> None:
  available = available_memory_bytes()
  row = {
      "label": label,
      "available_bytes": available,
      "minimum_bytes": minimum_bytes,
      "pass": available >= minimum_bytes,
  }
  rows.append(row)
  if not row["pass"]:
    raise RuntimeError(
        f"memory stop at {label}: {available} < {minimum_bytes} bytes")


def git_state(out_dir: Path) -> dict[str, Any]:
  commit = run(["git", "rev-parse", "HEAD"], 30).stdout.strip()
  dirty = run(["git", "status", "--porcelain"], 30).stdout.splitlines()
  try:
    out_rel = str(out_dir.relative_to(ROOT))
  except ValueError:
    out_rel = ""
  dirty = [row for row in dirty if not out_rel or out_rel not in row]
  return {"commit": commit, "dirty": bool(dirty), "dirty_paths": dirty}


def check(name: str, passed: bool, **details: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **details}


def codegen_build_command(binary: Path) -> list[str]:
  includes = [
      PINNED_SOURCE / "src/gpu/intel/gemm/jit",
      PINNED_SOURCE / "src/gpu/intel/gemm/jit/dnnl_gpu_intel_gemm_jit",
      PINNED_SOURCE / "src/gpu/intel/gemm/jit/dnnl_gpu_intel_gemm_jit",
      PINNED_BUILD / "include",
      PINNED_SOURCE / "include",
      PINNED_SOURCE / "third_party/opencl",
      PINNED_SOURCE / "third_party",
      PINNED_SOURCE / "src",
      PINNED_SOURCE / "src/gpu/intel/jit/config",
      PINNED_SOURCE / "third_party/ngen",
      PINNED_SOURCE / "src/gpu/intel/gemm/jit/include",
  ]
  return [
      str(CXX), "-std=c++17", "-O3", "-DNDEBUG", "-fopenmp",
      "-fno-operator-names", "-DCL_TARGET_OPENCL_VERSION=120",
      "-DDNNL_X64=1", "-DGEMMSTONE_BUILD_12HP",
      "-DGEMMSTONE_BUILD_12LP", "-DGEMMSTONE_BUILD_12P7",
      "-DGEMMSTONE_BUILD_12P8", "-DGEMMSTONE_BUILD_XE2",
      "-DGEMMSTONE_BUILD_XE3", "-DGEMMSTONE_BUILD_XE3P",
      "-DGEMMSTONE_CONFIG", "-DNGEN_CONFIG",
      *[f"-I{path}" for path in includes],
      str(CODEGEN), str(PINNED_BUILD / "src/libdnnl.a"),
      "-lOpenCL", "-ldl", "-lpthread", "-o", str(binary),
  ]


def triple_kernel_source(source: str) -> str:
  start = source.find(
      "#if IQ36_COMPONENT_PROGRAM == IQ36_COMPONENT_TRIPLE_COHORT")
  if start < 0:
    return ""
  end = source.find(
      "#if IQ36_COMPONENT_PROGRAM == IQ36_COMPONENT_DUAL_COHORT", start)
  return source[start:end] if end >= 0 else ""


def summary(payload: dict[str, Any]) -> str:
  result = payload["candidate_result"]
  resources = payload["resources"]
  return "\n".join([
      "# Exact-attention triple-cohort codegen gate",
      "",
      f"Verdict: **{payload['verdict']}**. Required checks: "
      f"`{str(payload['required_checks_passed']).lower()}`.",
      "",
      f"- register count / spill: "
      f"`{result.get('kernel_register_count')} / "
      f"{result.get('kernel_spill_memory_bytes')} B`",
      f"- actual SLM / cap / two-WG use: "
      f"`{result.get('kernel_local_memory_bytes')} / "
      f"{MAX_LOCAL_MEMORY_BYTES} / "
      f"{2 * int(result.get('kernel_local_memory_bytes', 0))} B`",
      f"- maximum / required workgroup items: "
      f"`{result.get('kernel_maximum_workgroup_size')} / "
      f"{EXPECTED_WORKGROUP_ITEMS}`",
      f"- compiler peak RSS / swaps: "
      f"`{resources['candidate'].get('maximum_resident_kib')} KiB / "
      f"{resources['candidate'].get('swaps')}`",
      "",
      "No kernel was enqueued.  A pass admits one standalone bit-exact",
      "component only; graph, plugin, and model work remain closed.",
      "",
  ])


def main() -> int:
  args = parse_args()
  official_prefetch = bool(args.official_prefetch)
  schema = OFFICIAL_PREFETCH_SCHEMA if official_prefetch else SCHEMA
  component_bound = (
      OFFICIAL_PREFETCH_BOUND if official_prefetch else COMPONENT_BOUND)
  host_define = (
      OFFICIAL_PREFETCH_HOST_DEFINE
      if official_prefetch else HOST_DEFINE)
  out_dir = args.out_dir.resolve()
  raw_dir = out_dir / "raw"
  raw_dir.mkdir(parents=True, exist_ok=False)
  preflight_bytes = int(args.memory_preflight_gib * 1024**3)
  stop_bytes = int(args.memory_stop_gib * 1024**3)
  memory: list[dict[str, Any]] = []
  sample_memory("start-preflight", preflight_bytes, memory)

  required_paths = (
      CODEGEN, HOST_SOURCE, SHIMS, TARGET_CONTRACT, component_bound,
      PINNED_SOURCE, PINNED_BUILD / "src/libdnnl.a", CXX, ENV_SCRIPT)
  missing = [str(path) for path in required_paths if not path.exists()]
  if missing:
    raise SystemExit(
        "missing triple-cohort codegen inputs: " + ", ".join(missing))

  git = git_state(out_dir)
  source = HOST_SOURCE.read_text(encoding="utf-8")
  kernel_source = triple_kernel_source(source)
  shims = SHIMS.read_text(encoding="utf-8")
  codegen_source = CODEGEN.read_text(encoding="utf-8")
  bound = load_json(component_bound)
  target = load_json(TARGET_CONTRACT)
  provider_head = run(
      ["git", "-C", str(PINNED_SOURCE), "rev-parse", "HEAD"],
      30).stdout.strip()
  provider_status = run([
      "git", "-C", str(PINNED_SOURCE), "status", "--short", "--",
      "src/gpu/intel/gemm/jit", "src/gpu/intel/jit/config",
      "third_party/ngen",
  ], 30).stdout.strip()
  clinfo = activated_timed(
      ["clinfo"], 60, raw_dir / "clinfo.time.txt")
  (raw_dir / "clinfo.stdout").write_text(
      clinfo.stdout, encoding="utf-8")
  (raw_dir / "clinfo.stderr").write_text(
      clinfo.stderr, encoding="utf-8")
  expected_device = str(target.get("runtime", {}).get("opencl_device", ""))

  bound_resource = bound.get("projected_resource_contract", {})
  if official_prefetch:
    correction = bound.get("fixed_correction_contract", {})
    bound_pass = bool(
        bound.get("schema_version") ==
            "intel-qwen36-openvino-exact-attention-"
            "k-prefetch-address-bound-gate-v1"
        and bound.get("verdict") ==
            "admit_one_official_address_triple_prefetch_correction_codegen"
        and bound.get("required_checks_passed") is True
        and bound.get("compiler_resource_probe_admitted") is True
        and bound.get("standalone_component_admitted") is False
        and bound.get("kernel_enqueue_admitted") is False
        and bound.get("model_worker_admitted") is False
        and bound.get("git", {}).get("dirty") is False
        and correction.get("scope") ==
            "triple carrier K prefetch addresses only"
        and correction.get("columns") == "IQ36_D"
        and correction.get("maximum_columns") == "IQ36_D")
  else:
    bound_pass = bool(
        bound.get("schema_version") ==
            "intel-qwen36-openvino-exact-attention-"
            "three-stage-component-gate-v1"
        and bound.get("verdict") ==
            "admit_one_exact_attention_triple_cohort_codegen_gate"
        and bound.get("required_checks_passed") is True
        and bound.get("compiler_resource_probe_admitted") is True
        and bound.get("component_implementation_admitted") is False
        and bound.get("graph_integration_admitted") is False
        and bound.get("model_worker_admitted") is False
        and bound.get("git", {}).get("dirty") is False
        and bound_resource.get("subgroups") == EXPECTED_SUBGROUPS
        and bound_resource.get("workgroup_items") ==
            EXPECTED_WORKGROUP_ITEMS
        and bound_resource.get("slm_padded_ceiling_bytes") == 61_472
        and bound_resource.get("two_workgroup_slm_bytes") == 122_944)
  source_checks = {
      "one_fixed_48_subgroup_kernel":
          kernel_source.count(
              "__kernel void iq36_exact_score_triple_cohort(") == 1
          and "__attribute__((reqd_work_group_size(16, 48, 1)))"
              in kernel_source
          and "#define IQ36_TRIPLE_TOTAL_SUBGROUPS 48"
              in kernel_source,
      "three_disjoint_sixteen_subgroup_cohorts":
          "#define IQ36_TRIPLE_KQ_SUBGROUPS 16" in kernel_source
          and "#define IQ36_TRIPLE_SOFTMAX_SUBGROUPS 16"
              in kernel_source
          and "#define IQ36_TRIPLE_VS_SUBGROUPS 16" in kernel_source,
      "generated_m256_n16_kq_and_vs_retained":
          kernel_source.count(
              "iq36_component_score_tile first_score = ugemm_kq(") == 1
          and kernel_source.count(
              "iq36_component_score_tile next_score = ugemm_kq(") == 1
          and kernel_source.count(
              "iq36_component_accumulator_tile chunk_accumulator = "
              "ugemm_vs(") == 1
          and "#define ugemm_kq_wg_tile_m 256" in shims
          and "#define ugemm_kq_wg_tile_n 16" in shims
          and "#define ugemm_vs_wg_tile_m 256" in shims
          and "#define ugemm_vs_wg_tile_n 16" in shims,
      "chronological_softmax_and_accumulator_rescale":
          "for (uint block = 0U; block < block_count; ++block)"
              in kernel_source
          and "tile_vreduce_max(score, &running_max);"
              in kernel_source
          and "tile_vbroadcast_sub(&score, running_max);"
              in kernel_source
          and "tile_elementwise(score, iq36_component_scaled_exp);"
              in kernel_source
          and "block_rescale, running_max, iq36_component_rescale"
              in kernel_source
          and "tile_hbroadcast_mul(&accumulator, accumulator_scale);"
              in kernel_source
          and "tile_copy(running_max, old_running_max);"
              in kernel_source,
      "only_double_buffered_slm_intermediate_abi":
          "raw_score_double_slm" in kernel_source
          and "normalized_score_double_slm" in kernel_source
          and "rescale_double_slm" in kernel_source
          and "__global float* raw_score" not in kernel_source
          and "__global half* normalized_score" not in kernel_source
          and "(__local half*)raw_score_double_slm" in kernel_source,
      "named_subset_pipeline_and_forward_progress":
          kernel_source.count("named_barrier_init(") == 4
          and "kq_softmax_barrier" in kernel_source
          and "softmax_vs_barrier" in kernel_source
          and "softmax_internal_barrier" in kernel_source
          and "vs_internal_barrier" in kernel_source,
      "explicit_61441_byte_unpadded_slab":
          "#define IQ36_TRIPLE_PIPELINE_SLAB_UINTS" in kernel_source
          and "__local half query_slm[IQ36_D * ugemm_kq_wg_tile_n];"
              in kernel_source
          and "__local uint pipeline_slab["
              "IQ36_TRIPLE_PIPELINE_SLAB_UINTS];" in kernel_source
          and "__local char ugemm_slm[1];" in kernel_source,
      "generic_fusion_is_pinned_and_isolated":
          "void FuseExistingShim()" in codegen_source
          and "const bool native_control" in codegen_source
          and "--fuse-existing-shim" in codegen_source
          and "--host-define" in codegen_source,
  }
  if official_prefetch:
    source_checks["seq2152_official_address_contract_only"] = bool(
        "IQ36_COMPONENT_TRIPLE_OFFICIAL_PREFETCH 11" in source
        and "#if IQ36_COMPONENT_PROGRAM == "
            "IQ36_COMPONENT_TRIPLE_OFFICIAL_PREFETCH" in kernel_source
        and "key_base, IQ36_CONTEXT, IQ36_D," in kernel_source
        and "ugemm_kq_wg_tile_m, IQ36_D, IQ36_D,"
            in kernel_source
        and "const __global half* next_key =" in kernel_source
        and "(ulong)next_block * ugemm_kq_wg_tile_m * IQ36_D;"
            in kernel_source
        and "IQ36_CONTEXT - next_block * ugemm_kq_wg_tile_m,"
            in kernel_source
        and kernel_source.count("LSC_LDCC_L1C_L3C") == 4
        and kernel_source.count(
            "key_base, IQ36_D, IQ36_CONTEXT,") == 2)
  device_checks = {
      "target_device":
          bool(expected_device) and expected_device in clinfo.stdout,
      "named_subset_barrier":
          "cl_khr_subgroup_named_barrier" in clinfo.stdout,
      "independent_forward_progress":
          "Sub-group independent forward progress        Yes"
              in clinfo.stdout,
      "workgroup_1024":
          "Max work group size                             1024"
              in clinfo.stdout,
      "subgroups_64":
          "Max sub-groups per work group                   64"
              in clinfo.stdout,
      "local_memory_128k":
          "Local memory size                               131072"
              in clinfo.stdout,
  }
  sample_memory("after-source-device-audit", stop_bytes, memory)

  binary = raw_dir / "openvino-micro-codegen"
  build_command = codegen_build_command(binary)
  build = activated_timed(
      build_command, args.timeout_s, raw_dir / "codegen-build.time.txt")
  write_json(raw_dir / "codegen-build.json", {
      "command": build_command,
      "returncode": build.returncode,
      "stdout": build.stdout,
      "stderr": build.stderr,
  })
  sample_memory("after-codegen-build", stop_bytes, memory)

  default_command = [str(binary), "--provider-commit", PINNED_COMMIT]
  default = (
      activated_timed(
          default_command, args.timeout_s,
          raw_dir / "default-codegen.time.txt")
      if build.returncode == 0 else
      subprocess.CompletedProcess(
          default_command, 1, "", "codegen build failed"))
  default_result = parse_last_json(default.stdout)
  write_json(raw_dir / "default-codegen.json", {
      "command": default_command,
      "returncode": default.returncode,
      "stdout": default.stdout,
      "stderr": default.stderr,
  })
  sample_memory("after-default-codegen", stop_bytes, memory)

  candidate_dir = raw_dir / "triple-cohort"
  candidate_command = [
      str(binary),
      "--fuse-existing-shim", str(SHIMS),
      "--host-source", str(HOST_SOURCE),
      "--kernel-name", KERNEL,
      "--host-define", host_define,
      "--register-file-size", str(REGISTER_FILE_SIZE),
      "--provider-commit", PINNED_COMMIT,
      "--dump-dir", str(candidate_dir),
  ]
  sample_memory("before-sole-candidate-compile", preflight_bytes, memory)
  candidate = (
      activated_timed(
          candidate_command, args.timeout_s,
          raw_dir / "triple-cohort-codegen.time.txt")
      if build.returncode == 0 else
      subprocess.CompletedProcess(
          candidate_command, 1, "", "codegen build failed"))
  sample_memory("after-sole-candidate-compile", stop_bytes, memory)
  candidate_result = parse_last_json(candidate.stdout)
  write_json(raw_dir / "triple-cohort-codegen.json", {
      "command": candidate_command,
      "returncode": candidate.returncode,
      "stdout": candidate.stdout,
      "stderr": candidate.stderr,
  })
  fused_source_path = candidate_dir / "existing_shim.fused.cl"
  program_path = candidate_dir / "existing_shim.program.bin"
  fused_source = (
      fused_source_path.read_text(encoding="utf-8")
      if fused_source_path.is_file() else "")
  fused_kernel_source = triple_kernel_source(fused_source)
  disassembly_dir = raw_dir / "triple-cohort-disassembly"
  disassembly_dir.mkdir()
  disassembly_command = [
      "ocloc", "disasm", "-file", str(program_path),
      "-dump", str(disassembly_dir)]
  disassembly = (
      run(disassembly_command, 120)
      if program_path.is_file() else
      subprocess.CompletedProcess(
          disassembly_command, 1, "", "program is missing"))
  asm_path = (
      disassembly_dir / ".text.iq36_exact_score_triple_cohort.asm")
  asm_text = (
      asm_path.read_text(encoding="utf-8")
      if asm_path.is_file() else "")
  prefetch_send_count = sum(
      "send.ugm" in line and "load.ugm.d8u32.a64.ca.ca" in line
      for line in asm_text.splitlines())
  block2d_load_count = sum(
      "send.ugm" in line and "load_block2d.ugm" in line
      and ".ca.ca" in line
      for line in asm_text.splitlines())
  write_json(raw_dir / "triple-cohort-disassembly.json", {
      "command": disassembly_command,
      "returncode": disassembly.returncode,
      "stdout": disassembly.stdout,
      "stderr": disassembly.stderr,
      "asm_path": display(asm_path),
      "asm_sha256": sha256(asm_path) if asm_path.is_file() else None,
      "prefetch_send_count": prefetch_send_count,
      "block2d_load_count": block2d_load_count,
  })

  build_resources = parse_time(raw_dir / "codegen-build.time.txt")
  default_resources = parse_time(raw_dir / "default-codegen.time.txt")
  candidate_resources = parse_time(
      raw_dir / "triple-cohort-codegen.time.txt")
  total_swaps = sum(
      int(resource.get("swaps", 0))
      for resource in (
          parse_time(raw_dir / "clinfo.time.txt"), build_resources,
          default_resources, candidate_resources))
  actual_local = int(
      candidate_result.get("kernel_local_memory_bytes", -1))
  resource_pass = bool(
      0 < int(candidate_result.get("kernel_register_count", -1))
          <= MAX_REGISTER_COUNT
      and candidate_result.get("kernel_spill_memory_bytes") == 0
      and MIN_LOCAL_MEMORY_BYTES <= actual_local <= MAX_LOCAL_MEMORY_BYTES
      and int(candidate_result.get(
          "kernel_maximum_workgroup_size", -1)) >=
          EXPECTED_WORKGROUP_ITEMS
      and candidate_result.get(
          "kernel_preferred_workgroup_multiple") == 16
      and 2 * actual_local <= 128 * 1024)
  compile_pass = bool(
      candidate.returncode == 0
      and candidate_result.get("schema_version") ==
          "intel-qwen36-openvino-existing-micro-shim-fuse-v0"
      and candidate_result.get("mode") ==
          "fuse_existing_microkernel_shim"
      and candidate_result.get("openvino_onednn_commit") ==
          PINNED_COMMIT
      and candidate_result.get("kernel_name") == KERNEL
      and candidate_result.get("host_define") == host_define
      and candidate_result.get("register_file_size") ==
          REGISTER_FILE_SIZE
      and candidate_result.get("exact_attention_vrt160") is False
      and candidate_result.get("exact_attention_dual_cohort") is False
      and fused_source_path.is_file()
      and program_path.is_file()
      and program_path.stat().st_size > 0)

  checks = [
      check("repository_clean_at_gate", not git["dirty"], git=git),
      check(
            "clean_seq2152_address_bound_admits_only_one_compiler_probe"
            if official_prefetch else
            "clean_seq2144_bound_admits_only_one_compiler_probe",
            bound_pass),
      check("fixed_exact_triple_cohort_source_contract",
            all(source_checks.values()), source_checks=source_checks),
      check("target_device_supports_the_fixed_geometry",
            clinfo.returncode == 0 and all(device_checks.values()),
            device_checks=device_checks),
      check("pinned_provider_scope_is_clean",
            provider_head == PINNED_COMMIT and provider_status == "",
            provider_head=provider_head,
            provider_status=provider_status),
      check("codegen_build_and_default_control_pass",
            build.returncode == 0 and binary.is_file()
            and default.returncode == 0
            and default_result.get("mode") == "moe"
            and default_result.get("register_file_size") == 256
            and len(default_result.get("packages", [])) == 3,
            build_returncode=build.returncode,
            default_result=default_result),
      check("exactly_one_fixed_triple_cohort_program_compiles",
            compile_pass, candidate_result=candidate_result,
            returncode=candidate.returncode,
            stderr=candidate.stderr[-8000:]),
      check("actual_resources_are_spill_free_and_two_wg_resident",
            resource_pass, candidate_result=candidate_result,
            two_workgroup_local_memory_bytes=2 * max(actual_local, 0)),
      check("fused_source_contains_one_exact_candidate",
            fused_kernel_source.count(
                "__kernel void iq36_exact_score_triple_cohort(") == 1
            and fused_kernel_source.count(
                "iq36_component_score_tile first_score = ugemm_kq(") == 1
            and fused_kernel_source.count(
                "iq36_component_score_tile next_score = ugemm_kq(") == 1
            and fused_kernel_source.count(
                "iq36_component_accumulator_tile chunk_accumulator = "
                "ugemm_vs(") == 1),
      check(
          "candidate_isa_matches_the_fixed_prefetch_address_mode",
          disassembly.returncode == 0
          and asm_path.is_file()
          and prefetch_send_count == (16 if official_prefetch else 4)
          and block2d_load_count == 48,
          prefetch_send_count=prefetch_send_count,
          expected_prefetch_send_count=(
              16 if official_prefetch else 4),
          block2d_load_count=block2d_load_count),
      check("zero_kernel_enqueue_plugin_or_model_workers", True),
      check("memory_guards_and_zero_swap_pass",
            all(row["pass"] for row in memory)
            and total_swaps == 0
            and int(candidate_resources.get(
                "maximum_resident_kib", 1 << 62)) < 4 * 1024 * 1024,
            minimum_available_bytes=min(
                row["available_bytes"] for row in memory),
            total_swaps=total_swaps,
            build_resources=build_resources,
            default_resources=default_resources,
            candidate_resources=candidate_resources),
  ]
  required = all(row["pass"] for row in checks)
  if official_prefetch:
    verdict = (
        "admit_one_exact_attention_triple_official_prefetch_component"
        if required else
        "close_exact_attention_triple_official_prefetch_after_codegen")
  else:
    verdict = (
        "admit_one_exact_attention_triple_cohort_component"
        if required else
        "close_exact_attention_triple_cohort_after_codegen")
  sources = [
      {"path": display(path), "sha256": sha256(path)}
      for path in (
          CODEGEN, HOST_SOURCE, SHIMS, TARGET_CONTRACT, component_bound)]
  resources = {
      "build": build_resources,
      "default": default_resources,
      "candidate": candidate_resources,
      "total_swaps": total_swaps,
  }
  payload = {
      "schema_version": schema,
      "workstream": WS,
      "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
      "git": git,
      "verdict": verdict,
      "required_checks_passed": required,
      "component_admitted": required,
      "kernel_enqueue_admitted": required,
      "graph_integration_admitted": False,
      "plugin_build_admitted": False,
      "model_worker_admitted": False,
      "product_claim_allowed": False,
      "compiler_workers_launched": True,
      "kernel_worker_launched": False,
      "model_worker_launched": False,
      "official_prefetch_mode": official_prefetch,
      "host_define": host_define,
      "checks": checks,
      "candidate_result": candidate_result,
      "resource_contract": {
          "subgroups": EXPECTED_SUBGROUPS,
          "workgroup_items": EXPECTED_WORKGROUP_ITEMS,
          "register_file_size": REGISTER_FILE_SIZE,
          "maximum_register_count": MAX_REGISTER_COUNT,
          "minimum_local_memory_bytes": MIN_LOCAL_MEMORY_BYTES,
          "maximum_local_memory_bytes": MAX_LOCAL_MEMORY_BYTES,
          "two_workgroup_device_local_memory_bytes": 128 * 1024,
      },
      "resources": resources,
      "provider_head": provider_head,
      "provider_status": provider_status,
      "disassembly": {
          "returncode": disassembly.returncode,
          "asm_path": display(asm_path),
          "asm_sha256": sha256(asm_path) if asm_path.is_file() else None,
          "prefetch_send_count": prefetch_send_count,
          "block2d_load_count": block2d_load_count,
      },
      "memory_preflight_bytes": preflight_bytes,
      "memory_stop_bytes": stop_bytes,
      "memory_samples": memory,
      "sources": sources,
  }
  write_json(out_dir / "result.json", payload)
  write_json(out_dir / "resources.json", {
      "candidate_result": candidate_result,
      "resource_contract": payload["resource_contract"],
      "resources": resources,
  })
  write_json(out_dir / "manifest.json", {
      "schema_version": "intel-qwen36-artifact-manifest-v1",
      "workstream": WS,
      "git_commit": git["commit"],
      "verdict": verdict,
      "sources": sources,
      "files": [
          "result.json", "resources.json", "summary.md",
          "raw/clinfo.stdout", "raw/clinfo.stderr", "raw/clinfo.time.txt",
          "raw/codegen-build.json", "raw/codegen-build.time.txt",
          "raw/default-codegen.json", "raw/default-codegen.time.txt",
          "raw/triple-cohort-codegen.json",
          "raw/triple-cohort-codegen.time.txt",
          "raw/triple-cohort/existing_shim.fused.cl",
          "raw/triple-cohort/existing_shim.program.bin",
          "raw/triple-cohort-disassembly.json",
          "raw/triple-cohort-disassembly/"
          ".text.iq36_exact_score_triple_cohort.asm",
      ],
  })
  (out_dir / "summary.md").write_text(
      summary(payload), encoding="utf-8")
  print(json.dumps({
      "artifact": display(out_dir),
      "verdict": verdict,
      "register_count":
          candidate_result.get("kernel_register_count"),
      "spill_memory_bytes":
          candidate_result.get("kernel_spill_memory_bytes"),
      "local_memory_bytes":
          candidate_result.get("kernel_local_memory_bytes"),
      "maximum_workgroup_items":
          candidate_result.get("kernel_maximum_workgroup_size"),
      "official_prefetch_mode": official_prefetch,
      "prefetch_send_count": prefetch_send_count,
      "component_admitted": required,
      "kernel_worker_launched": False,
      "model_worker_launched": False,
  }, separators=(",", ":")))
  return 0 if required else 2


if __name__ == "__main__":
  raise SystemExit(main())
