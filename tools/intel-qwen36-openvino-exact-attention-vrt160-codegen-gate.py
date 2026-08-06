#!/usr/bin/env python3
"""Gate one exact-attention 160-GRF compiler candidate before execution.

This gate builds the pinned Gemmstone shim fuser and compiles the same exact
M256/N16 fused-attention source in the native 128-GRF mode and the single
registered Xe3 160-GRF mode.  It enqueues no kernel and launches no model
worker.  A distinct, spill-free 160-GRF binary admits one bit-exact standalone
component; every other register size remains out of scope.
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
SCHEMA = "intel-qwen36-openvino-exact-attention-vrt160-codegen-gate-v1"
CODEGEN = ROOT / "engine/tools/openvino_moe_micro_codegen.cpp"
HOST_SOURCE = ROOT / "engine/gpu/opencl/exact_score_staging_component.cl"
SHIMS = ROOT / "engine/openvino/custom/iq36_decode_microkernel_shims.cl"
STATUS = ROOT / "doc/active" / WS / "STATUS.md"
FRONTIER = ROOT / "doc/active" / WS / "frontier.json"
STAGING_RESULT = ROOT / (
    "output/openvino-exact-score-staging-component-"
    "20260723Tseq2126-clean/result.json")
PINNED_SOURCE = Path(
    "/home/intel/intel-qwen36-r0/source/openvino-90214e5be05/"
    "src/plugins/intel_gpu/thirdparty/onednn_gpu")
PINNED_BUILD = Path(
    "/home/intel/intel-qwen36-r0/build/onednn-20db-micro-static")
COMPUTE_RUNTIME = Path(
    "/home/intel/intel-qwen36-r0/source/"
    "compute-runtime-82aab87fc932edc0558a0302d545a5bcc22edf41")
RELEASE_3004 = COMPUTE_RUNTIME / (
    "shared/source/release_helper/release_helper_3004.cpp")
XE3_CAPABILITY = COMPUTE_RUNTIME / "shared/source/xe3_core/hw_cmds_base.h"
XE3_PRODUCT = COMPUTE_RUNTIME / (
    "shared/source/xe3_core/os_agnostic_product_helper_xe3_core.inl")
XE3_OCCUPANCY = COMPUTE_RUNTIME / (
    "shared/source/helpers/gfx_core_helper_xe3_and_later.inl")
CXX = Path("/home/intel/intel-box-env/conda/bin/c++")
ENV_SCRIPT = Path(
    "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh")
PINNED_COMMIT = "20db47e2d3c4df1b66e93bed2e97d30da175512d"
COMPUTE_RUNTIME_COMMIT = "82aab87fc932edc0558a0302d545a5bcc22edf41"
CONTROL_GRFS = 128
CANDIDATE_GRFS = 160
DELTA_CAP_MS_PER_LAYER = -0.1175998


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


def run(command: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
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
        command, 124, stdout, stderr + "\ncommand timeout")


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
  values: dict[str, Any] = {"raw": text}
  for key, pattern in {
      "maximum_resident_kib":
          r"Maximum resident set size \(kbytes\): (\d+)",
      "swaps": r"Swaps: (\d+)",
      "major_page_faults":
          r"Major \(requiring I/O\) page faults: (\d+)",
  }.items():
    match = re.search(pattern, text)
    if match:
      values[key] = int(match.group(1))
  return values


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


def timed(
    command: list[str], timeout: int, time_path: Path,
    *, activate: bool = False,
) -> subprocess.CompletedProcess[str]:
  prefix = (
      f"source {shlex.quote(str(ENV_SCRIPT))} >/dev/null 2>&1 && "
      if activate else "")
  shell = (
      prefix + f"/usr/bin/time -v -o {shlex.quote(str(time_path))} " +
      " ".join(shlex.quote(part) for part in command))
  return run(["bash", "-lc", shell], timeout)


def source_has(path: Path, *needles: str) -> bool:
  text = path.read_text(encoding="utf-8")
  return all(needle in text for needle in needles)


def main() -> int:
  args = parse_args()
  out_dir = args.out_dir.resolve()
  raw_dir = out_dir / "raw"
  raw_dir.mkdir(parents=True, exist_ok=False)
  preflight_bytes = int(args.memory_preflight_gib * 1024**3)
  stop_bytes = int(args.memory_stop_gib * 1024**3)
  memory: list[dict[str, Any]] = []
  resources: list[dict[str, Any]] = []
  sample_memory("start-preflight", preflight_bytes, memory)

  required_paths = [
      CODEGEN, HOST_SOURCE, SHIMS, STATUS, FRONTIER, STAGING_RESULT,
      PINNED_SOURCE, PINNED_BUILD / "src/libdnnl.a", RELEASE_3004,
      XE3_CAPABILITY, XE3_PRODUCT, XE3_OCCUPANCY, CXX, ENV_SCRIPT]
  missing = [str(path) for path in required_paths if not path.exists()]
  if missing:
    raise SystemExit("missing VRT160 gate inputs: " + ", ".join(missing))

  git = git_state(out_dir)
  provider_head = run(
      ["git", "-C", str(PINNED_SOURCE), "rev-parse", "HEAD"], 30
  ).stdout.strip()
  provider_status = run([
      "git", "-C", str(PINNED_SOURCE), "status", "--short", "--",
      "src/gpu/intel/gemm/jit", "src/gpu/intel/jit/config",
      "third_party/ngen",
  ], 30).stdout.strip()
  runtime_head = run(
      ["git", "-C", str(COMPUTE_RUNTIME), "rev-parse", "HEAD"], 30
  ).stdout.strip()
  runtime_status = run(
      ["git", "-C", str(COMPUTE_RUNTIME), "status", "--short"], 30
  ).stdout.strip()
  staging = load_json(STAGING_RESULT)
  frontier = load_json(FRONTIER)
  status_text = STATUS.read_text(encoding="utf-8")
  source_checks = {
      "fixed_vrt_ladder":
          source_has(
              RELEASE_3004,
              "return {32u, 64u, 96u, 128u, 160u, 192u, 256u};"),
      "xe3_vrt_enabled":
          source_has(
              XE3_CAPABILITY, "enableVariableRegisterSizeAllocation = true")
          and source_has(
              XE3_PRODUCT,
              "propertiesSupport.enableVariableRegisterSizeAllocation"),
      "xe3_maps_128_to8_and_160_to6_threads":
          source_has(
              XE3_OCCUPANCY,
              "grfCount <= 128u", "maxThreadsPerEuCount = 8",
              "grfCount <= 160u", "maxThreadsPerEuCount = 6"),
      "candidate_is_hard_limited_to_exact_fused_160":
          source_has(
              CODEGEN, "const bool fixed_vrt160",
              'g_existing_kernel_name == "iq36_exact_score_fused"',
              'g_host_define == "IQ36_COMPONENT_PROGRAM=2"',
              "--exact-attention-vrt160"),
      "exact_source_uses_two_useful_groups":
          source_has(
              HOST_SOURCE,
              "#define IQ36_KV_HEADS 2",
              "__attribute__((reqd_work_group_size(16, 16, 1)))",
              "iq36_exact_score_fused"),
      "generated_m256_n16_shims":
          source_has(
              SHIMS,
              "#define ugemm_kq_wg_tile_m 256",
              "#define ugemm_kq_wg_tile_n 16",
              "#define ugemm_vs_wg_tile_m 256",
              "#define ugemm_vs_wg_tile_n 16"),
  }
  staging_samples = staging.get("result", {}).get("paired_samples", [])
  fused_values = [
      float(row["fused"]["total_ms"])
      for row in staging_samples
      if isinstance(row, dict)
      and isinstance(row.get("fused"), dict)
      and "total_ms" in row["fused"]]
  staging_baseline_ok = bool(
      staging.get("verdict") == "reject_exact_full_score_staging_component"
      and staging.get("git", {}).get("dirty") is False
      and staging.get("result", {}).get("numeric_pass") is True
      and len(fused_values) == 20)
  kill_number_ok = bool(
      abs(float(frontier["goal_budget"]["per_token_ms"]["remaining_cut"])
          - 0.345159) <= 1.0e-9
      and "1.175998 ms/token" in status_text)
  sample_memory("after-source-audit", stop_bytes, memory)

  binary = raw_dir / "openvino-micro-codegen"
  build_command = codegen_build_command(binary)
  build = timed(
      build_command, args.timeout_s, raw_dir / "codegen-build.time")
  resources.append(parse_time(raw_dir / "codegen-build.time"))
  write_json(raw_dir / "codegen-build.json", {
      "command": build_command,
      "returncode": build.returncode,
      "stdout": build.stdout,
      "stderr": build.stderr,
  })
  sample_memory("after-codegen-build", stop_bytes, memory)

  default = (
      timed([
          str(binary), "--provider-commit", PINNED_COMMIT,
      ], args.timeout_s, raw_dir / "default.time", activate=True)
      if build.returncode == 0 else
      subprocess.CompletedProcess([], 1, "", "codegen build failed"))
  resources.append(parse_time(raw_dir / "default.time"))
  default_result = parse_last_json(default.stdout)
  write_json(raw_dir / "default.json", {
      "command": list(default.args),
      "returncode": default.returncode,
      "stdout": default.stdout,
      "stderr": default.stderr,
      "result": default_result,
  })

  variants: dict[str, dict[str, Any]] = {}
  for label, grfs in (
      ("control128", CONTROL_GRFS), ("candidate160", CANDIDATE_GRFS)):
    variant_dir = raw_dir / label
    variant_dir.mkdir(parents=True, exist_ok=False)
    command = [
        str(binary),
        "--fuse-existing-shim", str(SHIMS),
        "--host-source", str(HOST_SOURCE),
        "--kernel-name", "iq36_exact_score_fused",
        "--host-define", "IQ36_COMPONENT_PROGRAM=2",
        "--register-file-size", str(grfs),
        "--provider-commit", PINNED_COMMIT,
        "--dump-dir", str(variant_dir),
    ]
    if label == "candidate160":
      command.insert(1, "--exact-attention-vrt160")
    sample_memory(f"before-{label}-codegen", stop_bytes, memory)
    completed = (
        timed(
            command, args.timeout_s, raw_dir / f"{label}.time",
            activate=True)
        if build.returncode == 0 else
        subprocess.CompletedProcess(command, 1, "", "codegen build failed"))
    resources.append(parse_time(raw_dir / f"{label}.time"))
    result = parse_last_json(completed.stdout)
    source_path = variant_dir / "existing_shim.fused.cl"
    program_path = variant_dir / "existing_shim.program.bin"
    variants[label] = {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "result": result,
        "source_path": display(source_path),
        "source_sha256": sha256(source_path) if source_path.is_file() else "",
        "program_path": display(program_path),
        "program_sha256":
            sha256(program_path) if program_path.is_file() else "",
    }
    sample_memory(f"after-{label}-codegen", stop_bytes, memory)
  write_json(raw_dir / "variants.json", variants)

  control = variants["control128"]
  candidate = variants["candidate160"]
  control_result = control["result"]
  candidate_result = candidate["result"]
  default_ok = bool(
      default.returncode == 0
      and default_result.get("mode") == "moe"
      and default_result.get("register_file_size") == 256
      and len(default_result.get("packages", [])) == 3)
  control_ok = bool(
      control["returncode"] == 0
      and control_result.get("register_file_size") == CONTROL_GRFS
      and control_result.get("exact_attention_vrt160") is False
      and control_result.get("kernel_register_count") == CONTROL_GRFS
      and control_result.get("kernel_spill_memory_bytes") == 0)
  candidate_ok = bool(
      candidate["returncode"] == 0
      and candidate_result.get("register_file_size") == CANDIDATE_GRFS
      and candidate_result.get("exact_attention_vrt160") is True
      and candidate_result.get("kernel_register_count") == CANDIDATE_GRFS
      and candidate_result.get("kernel_spill_memory_bytes") == 0)
  identical_source = bool(
      control["source_sha256"]
      and control["source_sha256"] == candidate["source_sha256"])
  distinct_program = bool(
      control["program_sha256"]
      and candidate["program_sha256"]
      and control["program_sha256"] != candidate["program_sha256"])
  max_rss = max(
      (int(row.get("maximum_resident_kib", 0)) for row in resources),
      default=0)
  swaps = sum(int(row.get("swaps", 0)) for row in resources)

  checks = [
      check("repository_clean_at_gate", not git["dirty"], git=git),
      check("pinned_provider_scope_is_clean",
            provider_head == PINNED_COMMIT and provider_status == "",
            provider_head=provider_head, provider_status=provider_status),
      check("pinned_compute_runtime_is_clean",
            runtime_head == COMPUTE_RUNTIME_COMMIT and runtime_status == "",
            runtime_head=runtime_head, runtime_status=runtime_status),
      check("fixed_source_and_runtime_contract",
            all(source_checks.values()), source_checks=source_checks),
      check("current_exact_component_and_kill_number_are_bound",
            staging_baseline_ok and kill_number_ok,
            fused_sample_count=len(fused_values),
            delta_cap_ms_per_layer=DELTA_CAP_MS_PER_LAYER),
      check("codegen_build_and_default_control_pass",
            build.returncode == 0 and default_ok),
      check("native_128_control_is_spill_free", control_ok,
            result=control_result),
      check("single_160_candidate_uses_160_grfs_without_spill",
            candidate_ok, result=candidate_result),
      check("compiler_modes_use_identical_fused_source",
            identical_source,
            control_sha256=control["source_sha256"],
            candidate_sha256=candidate["source_sha256"]),
      check("compiler_modes_emit_distinct_programs",
            distinct_program,
            control_sha256=control["program_sha256"],
            candidate_sha256=candidate["program_sha256"]),
      check("no_kernel_or_model_worker_launched", True),
      check("memory_guards_and_zero_swap_pass",
            all(row["pass"] for row in memory) and swaps == 0,
            minimum_available_bytes=min(
                row["available_bytes"] for row in memory),
            maximum_child_rss_kib=max_rss, swaps=swaps),
  ]
  required = all(row["pass"] for row in checks)
  verdict = (
      "admit_one_exact_attention_vrt160_component"
      if required else "reject_exact_attention_vrt160_before_component")
  payload = {
      "schema_version": SCHEMA,
      "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
      "workstream": WS,
      "git": git,
      "verdict": verdict,
      "required_checks_passed": required,
      "component_admitted": required,
      "graph_integration_admitted": False,
      "plugin_build_admitted": False,
      "model_worker_admitted": False,
      "kernel_worker_launched": False,
      "model_worker_launched": False,
      "control_grfs": CONTROL_GRFS,
      "candidate_grfs": CANDIDATE_GRFS,
      "component_delta_cap_ms_per_layer": DELTA_CAP_MS_PER_LAYER,
      "control_fused_samples_ms": fused_values,
      "checks": checks,
      "variants": variants,
      "memory_preflight_bytes": preflight_bytes,
      "memory_stop_bytes": stop_bytes,
      "memory_samples": memory,
      "resources": {
          "maximum_child_rss_kib": max_rss,
          "aggregate_swaps": swaps,
      },
      "inputs": {
          display(path): sha256(path)
          for path in required_paths if path.is_file()
      },
  }
  write_json(out_dir / "result.json", payload)
  (out_dir / "summary.md").write_text(
      "\n".join([
          "# Exact-attention Xe3 160-GRF codegen gate",
          "",
          f"Verdict: **{verdict}**. Required checks: "
          f"`{str(required).lower()}`.",
          "",
          f"- control/candidate register count: "
          f"`{control_result.get('kernel_register_count')} / "
          f"{candidate_result.get('kernel_register_count')}`",
          f"- control/candidate spill: "
          f"`{control_result.get('kernel_spill_memory_bytes')} / "
          f"{candidate_result.get('kernel_spill_memory_bytes')} B`",
          f"- identical source / distinct program: "
          f"`{str(identical_source).lower()} / "
          f"{str(distinct_program).lower()}`",
          f"- component paired-delta UCB cap: "
          f"`{DELTA_CAP_MS_PER_LAYER} ms/layer`",
          "",
          "No kernel was enqueued and no model worker ran. Only the adjacent",
          "160-GRF mode is admitted; 192/256-GRF variants are not authorized.",
          "",
      ]), encoding="utf-8")
  write_json(out_dir / "manifest.json", {
      "schema_version": f"{SCHEMA}-manifest-v1",
      "artifact": display(out_dir),
      "git": git,
      "verdict": verdict,
      "files": [
          "result.json", "summary.md", "raw/codegen-build.json",
          "raw/default.json", "raw/variants.json",
      ],
  })
  print(json.dumps({
      "artifact": display(out_dir),
      "verdict": verdict,
      "control_register_count":
          control_result.get("kernel_register_count"),
      "candidate_register_count":
          candidate_result.get("kernel_register_count"),
      "component_admitted": required,
      "kernel_worker_launched": False,
      "model_worker_launched": False,
  }, separators=(",", ":")))
  return 0 if required else 2


if __name__ == "__main__":
  raise SystemExit(main())
