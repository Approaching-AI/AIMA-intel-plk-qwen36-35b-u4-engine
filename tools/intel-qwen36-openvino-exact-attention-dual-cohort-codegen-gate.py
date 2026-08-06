#!/usr/bin/env python3
"""Compile-gate one exact-attention dual-cohort kernel without enqueueing it.

The sole candidate is a native-128-GRF, 512-work-item kernel containing one
16-subgroup generated-KQ producer cohort and one 16-subgroup chronological
softmax/generated-VS consumer cohort.  This gate builds the pinned shim fuser,
compiles that source, and queries actual kernel resources.  It launches no
kernel, plugin build, or model worker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = (
    "intel-qwen36-openvino-exact-attention-dual-cohort-codegen-gate-v1")
CODEGEN = ROOT / "engine/tools/openvino_moe_micro_codegen.cpp"
HOST_SOURCE = ROOT / "engine/gpu/opencl/exact_score_staging_component.cl"
SHIMS = ROOT / "engine/openvino/custom/iq36_decode_microkernel_shims.cl"
BOUND = ROOT / (
    "output/openvino-exact-attention-dual-cohort-bound-"
    "20260723Tseq2129-clean/bound.json")
STATUS = ROOT / "doc/active" / WS / "STATUS.md"
FRONTIER = ROOT / "doc/active" / WS / "frontier.json"
TARGET = ROOT / "contracts/intel-qwen36-target-contract.json"
PINNED_SOURCE = Path(
    "/home/intel/intel-qwen36-r0/source/openvino-90214e5be05/"
    "src/plugins/intel_gpu/thirdparty/onednn_gpu")
PINNED_BUILD = Path(
    "/home/intel/intel-qwen36-r0/build/onednn-20db-micro-static")
CXX = Path("/home/intel/intel-box-env/conda/bin/c++")
ENV_SCRIPT = Path(
    "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh")
PINNED_COMMIT = "20db47e2d3c4df1b66e93bed2e97d30da175512d"
EXPECTED_WORKGROUP_ITEMS = 512
MAX_EXPECTED_REGISTER_COUNT = 128
MIN_EXPECTED_LOCAL_MEMORY_BYTES = 59392
MAX_LOCAL_MEMORY_BYTES = 64 * 1024


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out-dir", type=Path, required=True)
  parser.add_argument("--timeout-s", type=int, default=600)
  parser.add_argument("--memory-preflight-gib", type=float, default=8.0)
  parser.add_argument("--memory-stop-gib", type=float, default=4.0)
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
    command: list[str], timeout: int, *, activate: bool = False,
) -> subprocess.CompletedProcess[str]:
  if activate:
    command_text = " ".join(shlex.quote(item) for item in command)
    return subprocess.run(
        [
            "bash", "-lc",
            f"source {shlex.quote(str(ENV_SCRIPT))} >/dev/null 2>&1 && "
            f"{command_text}",
        ],
        cwd=ROOT, check=False, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=timeout)
  return subprocess.run(
      command, cwd=ROOT, check=False, capture_output=True, text=True,
      encoding="utf-8", errors="replace", timeout=timeout)


def timed(
    command: list[str], timeout: int, time_path: Path,
    *, activate: bool = False,
) -> subprocess.CompletedProcess[str]:
  prefix = (
      f"source {shlex.quote(str(ENV_SCRIPT))} >/dev/null 2>&1 && "
      if activate else "")
  shell = (
      prefix + "/usr/bin/time -v -o " + shlex.quote(str(time_path)) + " "
      + " ".join(shlex.quote(item) for item in command))
  return subprocess.run(
      ["bash", "-lc", shell], cwd=ROOT, check=False, capture_output=True,
      text=True, encoding="utf-8", errors="replace", timeout=timeout)


def load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise TypeError(f"expected JSON object: {path}")
  return value


def parse_stdout_json(completed: subprocess.CompletedProcess[str]) -> dict[str, Any]:
  for line in reversed(completed.stdout.splitlines()):
    line = line.strip()
    if line.startswith("{"):
      value = json.loads(line)
      if isinstance(value, dict):
        return value
  return {}


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
  for row in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
    if row.startswith("MemAvailable:"):
      return int(row.split()[1]) * 1024
  raise RuntimeError("MemAvailable is missing")


def sample_memory(
    label: str, minimum_bytes: int, rows: list[dict[str, Any]],
) -> None:
  available = available_memory_bytes()
  rows.append({
      "label": label,
      "available_bytes": available,
      "minimum_bytes": minimum_bytes,
      "pass": available >= minimum_bytes,
  })
  if available < minimum_bytes:
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


def parse_time(path: Path) -> dict[str, Any]:
  text = path.read_text(encoding="utf-8") if path.is_file() else ""
  result: dict[str, Any] = {"raw": text}
  for key, pattern in {
      "maximum_resident_kib":
          r"Maximum resident set size \(kbytes\): (\d+)",
      "swaps": r"Swaps: (\d+)",
      "major_page_faults":
          r"Major \(requiring I/O\) page faults: (\d+)",
      "elapsed":
          r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (.+)",
  }.items():
    match = re.search(pattern, text)
    if match:
      result[key] = (
          match.group(1) if key == "elapsed" else int(match.group(1)))
  return result


def check(name: str, passed: bool, **details: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **details}


def codegen_build_command(binary: Path) -> list[str]:
  includes = [
      PINNED_SOURCE / "src/gpu/intel/gemm/jit",
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


def main() -> int:
  args = parse_args()
  out_dir = args.out_dir.resolve()
  out_dir.mkdir(parents=True, exist_ok=False)
  raw_dir = out_dir / "raw"
  raw_dir.mkdir()
  preflight_bytes = int(args.memory_preflight_gib * 1024**3)
  stop_bytes = int(args.memory_stop_gib * 1024**3)
  memory: list[dict[str, Any]] = []
  sample_memory("start-preflight", preflight_bytes, memory)

  required_paths = (
      CODEGEN, HOST_SOURCE, SHIMS, BOUND, STATUS, FRONTIER, TARGET,
      PINNED_SOURCE, PINNED_BUILD / "src/libdnnl.a", CXX, ENV_SCRIPT)
  missing = [str(path) for path in required_paths if not path.exists()]
  if missing:
    raise SystemExit("missing dual-cohort codegen inputs: " + ", ".join(missing))

  git = git_state(out_dir)
  bound = load_json(BOUND)
  source_text = HOST_SOURCE.read_text(encoding="utf-8")
  dual_begin = source_text.index(
      "#if IQ36_COMPONENT_PROGRAM == IQ36_COMPONENT_DUAL_COHORT")
  dual_end = source_text.index("\n#endif", dual_begin)
  dual_source = source_text[dual_begin:dual_end]
  shim_text = SHIMS.read_text(encoding="utf-8")
  codegen_text = CODEGEN.read_text(encoding="utf-8")
  provider_head = run(
      ["git", "-C", str(PINNED_SOURCE), "rev-parse", "HEAD"],
      30).stdout.strip()
  provider_status = run(
      ["git", "-C", str(PINNED_SOURCE), "status", "--short", "--",
       "src/gpu/intel/gemm/jit",
       "src/gpu/intel/jit/config",
       "third_party/ngen"],
      30).stdout.strip()

  source_checks = {
      "single_registered_dual_mode":
          "bool g_exact_attention_dual_cohort = false;" in codegen_text
          and 'option == "--exact-attention-dual-cohort"' in codegen_text
          and "const bool fixed_dual_cohort" in codegen_text
          and 'g_host_define == "IQ36_COMPONENT_PROGRAM=4"'
              in codegen_text,
      "fixed_512_item_two_cohort_kernel":
          "__attribute__((reqd_work_group_size(16, 32, 1)))" in source_text
          and "#define IQ36_DUAL_PRODUCER_SUBGROUPS 16" in source_text
          and "#define IQ36_DUAL_CONSUMER_SUBGROUPS 16" in source_text,
      "raw_f32_handoff_is_double_buffered_local_only":
          "__local float raw_score_double_slm[" in source_text
          and "2 * IQ36_DUAL_RAW_BUFFER_ELEMENTS" in source_text
          and "first_score, raw_score_double_slm" in source_text
          and "next_score, next_raw_score" in source_text
          and "tile_load_full(\n          &score, current_raw_score"
              in source_text,
      "consumer_and_pipeline_use_named_barriers":
          "__local NamedBarrier_t* consumer_barrier" in dual_source
          and "named_barrier_init(IQ36_DUAL_CONSUMER_SUBGROUPS)"
              in dual_source
          and "__local NamedBarrier_t* pipeline_barrier" in dual_source
          and "named_barrier_init(IQ36_DUAL_TOTAL_SUBGROUPS)"
              in dual_source
          and dual_source.count("work_group_named_barrier(") == 4
          and "barrier(CLK_LOCAL_MEM_FENCE);" in dual_source,
      "deterministic_max_reduction_replaces_atomic_and_v_prefetch":
          "float reduced_running_max = -INFINITY;" in dual_source
          and "subgroup_row < IQ36_DUAL_CONSUMER_SUBGROUPS"
              in dual_source
          and "tile_atomic_max_full(" not in dual_source
          and "value_base + (ulong)key_begin * IQ36_D" in dual_source
          and "cooperative_prefetch_2d_rem(\n"
              "          value_base + (ulong)key_begin * IQ36_D"
              not in dual_source,
      "chronological_recurrence_is_retained":
          "tile_copy(running_max, old_running_max);" in source_text
          and "tile_hbroadcast_mul(&accumulator, accumulator_scale);"
              in source_text
          and "for (uint block = 0U; block < block_count; ++block)"
              in source_text,
      "generated_m256_n16_packages_are_retained":
          "#define ugemm_kq_wg_tile_m 256" in shim_text
          and "#define ugemm_kq_wg_tile_n 16" in shim_text
          and "#define ugemm_vs_wg_tile_m 256" in shim_text
          and "#define ugemm_vs_wg_tile_n 16" in shim_text
          and "#define ugemm_kq_barrier_count  0" in shim_text
          and "#define ugemm_vs_barrier_count  0" in shim_text,
  }
  sample_memory("after-source-audit", stop_bytes, memory)

  binary = raw_dir / "openvino-micro-codegen"
  build_command = codegen_build_command(binary)
  build_time = raw_dir / "codegen-build.time.txt"
  build = timed(build_command, args.timeout_s, build_time)
  build_resources = parse_time(build_time)
  sample_memory("after-codegen-build", stop_bytes, memory)

  candidate_dir = raw_dir / "dual-cohort"
  candidate_command = [
      str(binary),
      "--exact-attention-dual-cohort",
      "--fuse-existing-shim", str(SHIMS),
      "--host-source", str(HOST_SOURCE),
      "--kernel-name", "iq36_exact_score_dual_cohort",
      "--host-define", "IQ36_COMPONENT_PROGRAM=4",
      "--register-file-size", "128",
      "--provider-commit", PINNED_COMMIT,
      "--dump-dir", str(candidate_dir),
  ]
  candidate_time = raw_dir / "dual-cohort-codegen.time.txt"
  candidate = (
      timed(
          candidate_command, args.timeout_s, candidate_time, activate=True)
      if build.returncode == 0 else
      subprocess.CompletedProcess(
          candidate_command, 127, "", "codegen build failed"))
  candidate_resources = parse_time(candidate_time)
  candidate_result = parse_stdout_json(candidate)
  sample_memory("after-dual-cohort-codegen", stop_bytes, memory)

  fused_source_path = candidate_dir / "existing_shim.fused.cl"
  program_path = candidate_dir / "existing_shim.program.bin"
  fused_source = (
      fused_source_path.read_text(encoding="utf-8")
      if fused_source_path.is_file() else "")
  resource_pass = bool(
      0 < int(candidate_result.get("kernel_register_count", -1))
          <= MAX_EXPECTED_REGISTER_COUNT
      and candidate_result.get("kernel_spill_memory_bytes") == 0
      and MIN_EXPECTED_LOCAL_MEMORY_BYTES
          <= int(candidate_result.get("kernel_local_memory_bytes", -1))
          <= MAX_LOCAL_MEMORY_BYTES
      and int(candidate_result.get(
          "kernel_maximum_workgroup_size", -1))
          >= EXPECTED_WORKGROUP_ITEMS)
  bound_pass = bool(
      bound.get("required_checks_passed") is True
      and bound.get("verdict") ==
          "admit_one_exact_attention_dual_cohort_codegen_gate"
      and bound.get("compiler_gate_admitted") is True
      and bound.get("component_admitted") is False
      and bound.get("resource_contract", {}).get("workgroup_items")
          == EXPECTED_WORKGROUP_ITEMS
      and bound.get("resource_contract", {}).get(
          "slm_bytes", {}).get("total") == 59393)

  checks = [
      check("repository_clean_at_gate", not git["dirty"], git=git),
      check("clean_source_bound_admits_only_codegen", bound_pass),
      check("pinned_provider_scope_is_clean",
            provider_head == PINNED_COMMIT and not provider_status,
            provider_head=provider_head, provider_status=provider_status),
      check("fixed_dual_cohort_source_contract",
            all(source_checks.values()), source_checks=source_checks),
      check("codegen_build_is_serial_and_passes",
            build.returncode == 0 and binary.is_file()
            and build_command[-2:] == ["-o", str(binary)],
            returncode=build.returncode,
            stderr=build.stderr[-4000:]),
      check("single_dual_cohort_program_compiles",
            candidate.returncode == 0
            and candidate_result.get("exact_attention_dual_cohort") is True
            and candidate_result.get("exact_attention_vrt160") is False
            and candidate_result.get("register_file_size") == 128
            and candidate_result.get("kernel_name") ==
                "iq36_exact_score_dual_cohort"
            and candidate_result.get("host_define") ==
                "IQ36_COMPONENT_PROGRAM=4"
            and fused_source_path.is_file() and program_path.is_file()
            and program_path.stat().st_size > 0,
            returncode=candidate.returncode,
            result=candidate_result,
            stderr=candidate.stderr[-8000:]),
      check("actual_resources_preserve_native_spill_free_two_wg_shape",
            resource_pass, result=candidate_result),
      check("fused_source_contains_exactly_one_candidate_kernel",
            fused_source.count(
                "__kernel void iq36_exact_score_dual_cohort(") == 1
            and fused_source.count(
                "iq36_component_score_tile score = ugemm_kq(") >= 2
            and fused_source.count(
                "iq36_component_accumulator_tile chunk_accumulator = "
                "ugemm_vs(") >= 2),
      check("no_kernel_plugin_or_model_worker_launched", True),
      check("memory_guards_and_zero_swap_pass",
            all(row["pass"] for row in memory)
            and build_resources.get("swaps", 0) == 0
            and candidate_resources.get("swaps", 0) == 0,
            minimum_available_bytes=min(
                row["available_bytes"] for row in memory),
            maximum_child_rss_kib=max(
                int(build_resources.get("maximum_resident_kib", 0)),
                int(candidate_resources.get("maximum_resident_kib", 0))),
            swaps=int(build_resources.get("swaps", 0))
                + int(candidate_resources.get("swaps", 0))),
  ]
  required_checks_passed = all(row["pass"] for row in checks)
  verdict = (
      "admit_one_exact_attention_dual_cohort_component"
      if required_checks_passed else
      "reject_exact_attention_dual_cohort_before_enqueue")

  inputs = {
      display(path): sha256(path)
      for path in required_paths
      if path.is_file()
  }
  payload = {
      "schema_version": SCHEMA,
      "workstream": WS,
      "created_at": datetime.now(timezone.utc).isoformat(),
      "git": git,
      "inputs": inputs,
      "source_checks": source_checks,
      "provider_head": provider_head,
      "provider_status": provider_status,
      "build": {
          "command": build_command,
          "returncode": build.returncode,
          "stdout": build.stdout,
          "stderr": build.stderr,
          "resources": build_resources,
      },
      "candidate": {
          "command": candidate_command,
          "returncode": candidate.returncode,
          "stdout": candidate.stdout,
          "stderr": candidate.stderr,
          "result": candidate_result,
          "resources": candidate_resources,
          "fused_source_path": display(fused_source_path),
          "fused_source_sha256":
              sha256(fused_source_path) if fused_source_path.is_file() else None,
          "program_path": display(program_path),
          "program_sha256":
              sha256(program_path) if program_path.is_file() else None,
      },
      "checks": checks,
      "required_checks_passed": required_checks_passed,
      "verdict": verdict,
      "component_admitted": required_checks_passed,
      "kernel_worker_launched": False,
      "plugin_build_admitted": False,
      "model_worker_admitted": False,
      "memory_preflight_bytes": preflight_bytes,
      "memory_stop_bytes": stop_bytes,
      "memory_samples": memory,
  }
  (out_dir / "result.json").write_text(
      json.dumps(payload, indent=2, sort_keys=True) + "\n",
      encoding="utf-8")
  print(json.dumps({
      "verdict": verdict,
      "register_count": candidate_result.get("kernel_register_count"),
      "spill_memory_bytes":
          candidate_result.get("kernel_spill_memory_bytes"),
      "local_memory_bytes":
          candidate_result.get("kernel_local_memory_bytes"),
      "maximum_workgroup_size":
          candidate_result.get("kernel_maximum_workgroup_size"),
  }, sort_keys=True))
  return 0 if required_checks_passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
