#!/usr/bin/env python3
"""Measure the fixed two-workgroup full-KV physical traffic ceiling.

The gate compiles and runs one standalone OpenCL kernel with the accepted
triple carrier's outer geometry: two useful workgroups, 48 subgroups and 768
work-items per group.  It reads the complete 256-MiB K/V payload once per
sample.  A pass establishes physical traffic headroom only; it does not admit
a cache-policy edit, an exact-attention implementation, graph work, or a model
worker.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import re
import shlex
import statistics
import subprocess
from pathlib import Path
from typing import Any

from iq36_perf_inference import latency_cap_inference


ROOT = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = (
    "intel-qwen36-openvino-exact-attention-"
    "two-workgroup-traffic-gate-v1")
SOURCE = ROOT / "engine/gpu/opencl/exact_score_staging_component.cl"
RUNNER = ROOT / "engine/tools/exact_score_staging_component.cpp"
CODEGEN = ROOT / "engine/tools/openvino_moe_micro_codegen.cpp"
SHIMS = ROOT / "engine/openvino/custom/iq36_decode_microkernel_shims.cl"
AUDIT = ROOT / (
    "output/openvino-exact-attention-hardware-limit-opportunity-"
    "20260724Tseq2150a-clean/result.json")
CAPTURE_PROGRAM = ROOT / (
    "output/openvino-exact-attention-three-stage-component-"
    "20260724Tseq2144-clean/raw/programs/capture/"
    "existing_shim.program.bin")
PINNED_SOURCE = Path(
    "/home/intel/intel-qwen36-r0/source/openvino-90214e5be05/"
    "src/plugins/intel_gpu/thirdparty/onednn_gpu")
PINNED_BUILD = Path(
    "/home/intel/intel-qwen36-r0/build/onednn-20db-micro-static")
CXX = Path("/home/intel/intel-box-env/conda/bin/c++")
CMAKE = Path("/home/intel/intel-box-env/conda/bin/cmake")
ENV_SCRIPT = Path(
    "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh")
BUILD_DIR = ROOT / "build/engine"
TARGET = "iq36-exact-score-staging-component"
PINNED_COMMIT = "20db47e2d3c4df1b66e93bed2e97d30da175512d"

MIN_SAMPLES = 20
PAYLOAD_BYTES = 268_435_456
WORKGROUPS = 2
SUBGROUPS_PER_WORKGROUP = 48
WORKGROUP_ITEMS = 768
LATENCY_UCB_CAP_MS = 2.7375042
BANDWIDTH_LCB_FLOOR_GB_S = 98.05846361806495
CAPTURE_PROGRAM_BYTES = 307_024
CAPTURE_PROGRAM_SHA256 = (
    "06ff0dad9cdf28b05fd414058a6c496ca545af09bf013fddb2b927ece85fb748")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out-dir", type=Path, required=True)
  parser.add_argument("--timeout-s", type=int, default=300)
  parser.add_argument("--memory-preflight-gib", type=float, default=8.0)
  parser.add_argument("--memory-stop-gib", type=float, default=4.0)
  args = parser.parse_args()
  if args.timeout_s < 60:
    parser.error("--timeout-s must be at least 60")
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
      f"/usr/bin/timeout --signal=TERM --kill-after=5s {timeout}s "
      f"/usr/bin/time -v -o {shlex.quote(str(time_path))} " +
      " ".join(shlex.quote(part) for part in command))
  return run(["bash", "-lc", shell_command], timeout + 10)


def load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise TypeError(f"expected JSON object: {path}")
  return value


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True) + "\n",
      encoding="utf-8")


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
      "exit_status": r"Exit status: (\d+)",
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
    return str(path.resolve().relative_to(ROOT))
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
  rows = run(["git", "status", "--porcelain"], 30).stdout.splitlines()
  try:
    out_rel = str(out_dir.relative_to(ROOT))
  except ValueError:
    out_rel = ""
  rows = [row for row in rows if not out_rel or out_rel not in row]
  return {"commit": commit, "dirty": bool(rows), "dirty_paths": rows}


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


def fixed_result(result: dict[str, Any]) -> bool:
  return bool(
      result.get("schema_version") ==
          "intel-qwen36-exact-attention-dense-traffic-ceiling-v1"
      and result.get("algorithm") ==
          "two_workgroup_full_kv_uint8_dense_read"
      and result.get("context_tokens") == 131072
      and result.get("head_dim") == 256
      and result.get("kv_heads") == WORKGROUPS
      and result.get("traffic_workgroups") == WORKGROUPS
      and result.get("traffic_subgroups_per_workgroup") ==
          SUBGROUPS_PER_WORKGROUP
      and result.get("traffic_workitems_per_workgroup") == WORKGROUP_ITEMS
      and result.get("read_vector_bytes") == 32
      and result.get("read_vectors_per_head") == 2_097_152
      and result.get("mandatory_key_value_payload_bytes") == PAYLOAD_BYTES
      and result.get("checksum_output_bytes") == 98_304
      and result.get("checksum_compared_words") == 24_576
      and result.get("warmup_count") == 12
      and result.get("sample_count") == MIN_SAMPLES
      and result.get("schedule") ==
          "single_dense_traffic_ceiling_after_twelve_warmups")


def distribution_pass(result: dict[str, Any]) -> bool:
  rows = result.get("samples", [])
  if not isinstance(rows, list) or len(rows) != MIN_SAMPLES:
    return False
  for index, row in enumerate(rows):
    if not isinstance(row, dict) or row.get("sample") != index:
      return False
    try:
      latency = float(row["latency_ms"])
      bandwidth = float(row["bandwidth_gb_s"])
    except (KeyError, TypeError, ValueError):
      return False
    if any(
        not math.isfinite(value) or value <= 0.0
        for value in (latency, bandwidth)):
      return False
    if not math.isclose(
        bandwidth, PAYLOAD_BYTES / (latency * 1.0e6),
        rel_tol=0.0, abs_tol=5.0e-6):
      return False
  return True


def resource_pass(
    result: dict[str, Any], codegen_result: dict[str, Any],
) -> bool:
  resources = result.get("resources", {})
  expected = {
      "register_count": codegen_result.get("kernel_register_count"),
      "spill_memory_bytes":
          codegen_result.get("kernel_spill_memory_bytes"),
      "local_memory_bytes":
          codegen_result.get("kernel_local_memory_bytes"),
      "maximum_workgroup_items":
          codegen_result.get("kernel_maximum_workgroup_size"),
      "preferred_workgroup_multiple":
          codegen_result.get("kernel_preferred_workgroup_multiple"),
  }
  try:
    registers = int(resources["register_count"])
    spill = int(resources["spill_memory_bytes"])
    local = int(resources["local_memory_bytes"])
    maximum = int(resources["maximum_workgroup_items"])
    preferred = int(resources["preferred_workgroup_multiple"])
  except (KeyError, TypeError, ValueError):
    return False
  return bool(
      resources == expected
      and 0 < registers <= 128
      and spill == 0
      and local == 0
      and maximum >= WORKGROUP_ITEMS
      and preferred == 16)


def summary(payload: dict[str, Any]) -> str:
  result = payload["result"]
  inference = payload["performance_inference"]
  latencies = [
      float(row["latency_ms"])
      for row in result.get("samples", [])
      if isinstance(row, dict) and "latency_ms" in row]
  return "\n".join([
      "# Exact-attention two-workgroup traffic ceiling",
      "",
      f"Verdict: **{payload['verdict']}**. Measurement valid: "
      f"`{str(payload['measurement_valid']).lower()}`.",
      "",
      f"- latency median / 95% UCB / cap: "
      f"`{statistics.median(latencies) if latencies else None} / "
      f"{inference.get('upper_confidence_bound_ms')} / "
      f"{LATENCY_UCB_CAP_MS} ms`",
      f"- bandwidth point / LCB / floor: "
      f"`{payload.get('bandwidth_point_gb_s')} / "
      f"{payload.get('bandwidth_lcb_gb_s')} / "
      f"{BANDWIDTH_LCB_FLOOR_GB_S} GB/s`",
      f"- checksum mismatches / nonzero words: "
      f"`{result.get('checksum_mismatch_count')} / "
      f"{result.get('checksum_nonzero_words')}`",
      f"- runtime resources: `{result.get('resources')}`",
      f"- worker peak RSS / swaps: "
      f"`{payload['worker_resources'].get('maximum_resident_kib')} KiB / "
      f"{payload['worker_resources'].get('swaps')}`",
      "",
      "This result measures physical traffic capacity only. It does not admit",
      "a cache-policy edit, an exact kernel, graph/plugin work, or a model",
      "worker.",
      "",
  ])


def main() -> int:
  args = parse_args()
  out_dir = args.out_dir.resolve()
  raw_dir = out_dir / "raw"
  raw_dir.mkdir(parents=True, exist_ok=False)
  preflight_bytes = int(args.memory_preflight_gib * 1024**3)
  stop_bytes = int(args.memory_stop_gib * 1024**3)
  memory: list[dict[str, Any]] = []
  sample_memory("start-preflight", preflight_bytes, memory)

  required_paths = (
      SOURCE, RUNNER, CODEGEN, SHIMS, AUDIT, CAPTURE_PROGRAM,
      PINNED_SOURCE, PINNED_BUILD / "src/libdnnl.a",
      CXX, CMAKE, ENV_SCRIPT)
  missing = [str(path) for path in required_paths if not path.exists()]
  if missing:
    raise SystemExit(
        "missing two-workgroup traffic gate inputs: " + ", ".join(missing))

  git = git_state(out_dir)
  audit = load_json(AUDIT)
  source_text = SOURCE.read_text(encoding="utf-8")
  runner_text = RUNNER.read_text(encoding="utf-8")
  dense_marker = (
      "#if IQ36_COMPONENT_PROGRAM == IQ36_COMPONENT_DENSE_TRAFFIC")
  dense_block = (
      source_text.split(dense_marker, 1)[1].split("#endif", 1)[0]
      if source_text.count(dense_marker) == 1 else "")
  contract = audit.get("traffic_bound", {}).get(
      "traffic_ceiling_contract", {})
  audit_pass = bool(
      audit.get("verdict") ==
          "admit_one_exact_attention_two_workgroup_dense_traffic_ceiling"
      and audit.get("required_checks_passed") is True
      and audit.get("dense_traffic_ceiling_admitted") is True
      and audit.get("exact_kernel_implementation_admitted") is False
      and audit.get("gpu_worker_admitted") is True
      and audit.get("model_worker_admitted") is False
      and contract.get("workgroups") == WORKGROUPS
      and contract.get("subgroups_per_workgroup") ==
          SUBGROUPS_PER_WORKGROUP
      and contract.get("workgroup_items") == WORKGROUP_ITEMS
      and contract.get("payload_bytes") == PAYLOAD_BYTES
      and contract.get("minimum_samples") == MIN_SAMPLES
      and contract.get("latency_ucb_cap_ms") == LATENCY_UCB_CAP_MS
      and math.isclose(
          float(contract.get("bandwidth_lcb_floor_gb_s", math.nan)),
          BANDWIDTH_LCB_FLOOR_GB_S, rel_tol=0.0, abs_tol=1.0e-12))
  source_checks = {
      "one_fixed_dense_traffic_program":
          source_text.count("IQ36_COMPONENT_DENSE_TRAFFIC 10") == 1
          and source_text.count(
              "__kernel void "
              "iq36_exact_attention_dense_traffic_ceiling(") == 1
          and bool(dense_block),
      "fixed_two_workgroup_geometry":
          "__attribute__((reqd_work_group_size(16, 48, 1)))"
              in dense_block
          and "IQ36_DENSE_TRAFFIC_SUBGROUPS 48" in dense_block
          and "IQ36_DENSE_TRAFFIC_WORKITEMS" in dense_block
          and "const uint kv_head = (uint)get_group_id(1);" in dense_block,
      "full_k_and_v_payload_partition":
          "IQ36_DENSE_TRAFFIC_UINT8S_PER_HEAD" in dense_block
          and "vector_index += IQ36_DENSE_TRAFFIC_WORKITEMS" in dense_block
          and "key_accumulator ^= vload8" in dense_block
          and "value_accumulator += vload8" in dense_block
          and dense_block.count("vload8(") == 2,
      "no_local_or_synchronization_work":
          "__local" not in dense_block
          and "barrier(" not in dense_block,
      "runner_has_fixed_serial_measurement":
          "RunDenseTrafficCeiling" in runner_text
          and "--dense-traffic" in runner_text
          and "constexpr std::size_t kTrafficSubgroups = 48;"
              in runner_text
          and "constexpr int kTrafficWarmups = 12;" in runner_text
          and "constexpr int kSamples = 20;" in runner_text
          and "single_dense_traffic_ceiling_after_twelve_warmups"
              in runner_text,
      "capture_program_is_fixed":
          CAPTURE_PROGRAM.stat().st_size == CAPTURE_PROGRAM_BYTES
          and sha256(CAPTURE_PROGRAM) == CAPTURE_PROGRAM_SHA256,
  }
  provider_head = run(
      ["git", "-C", str(PINNED_SOURCE), "rev-parse", "HEAD"],
      30).stdout.strip()
  provider_status = run([
      "git", "-C", str(PINNED_SOURCE), "status", "--short", "--",
      "src/gpu/intel/gemm/jit", "src/gpu/intel/jit/config",
      "third_party/ngen",
  ], 30).stdout.strip()
  sample_memory("after-source-audit", stop_bytes, memory)

  codegen_binary = raw_dir / "openvino-micro-codegen"
  build_codegen_command = codegen_build_command(codegen_binary)
  sample_memory("before-serial-codegen-build", preflight_bytes, memory)
  codegen_build = activated_timed(
      build_codegen_command, args.timeout_s,
      raw_dir / "codegen-build.time.txt")
  write_json(raw_dir / "codegen-build.json", {
      "command": build_codegen_command,
      "returncode": codegen_build.returncode,
      "stdout": codegen_build.stdout,
      "stderr": codegen_build.stderr,
  })
  sample_memory("after-serial-codegen-build", stop_bytes, memory)

  program_dir = raw_dir / "traffic-program"
  program_dir.mkdir()
  program_command = [
      str(codegen_binary),
      "--fuse-existing-shim", str(SHIMS),
      "--host-source", str(SOURCE),
      "--kernel-name", "iq36_exact_attention_dense_traffic_ceiling",
      "--host-define", "IQ36_COMPONENT_PROGRAM=10",
      "--register-file-size", "128",
      "--provider-commit", PINNED_COMMIT,
      "--dump-dir", str(program_dir),
  ]
  sample_memory("before-sole-traffic-codegen", preflight_bytes, memory)
  program_run = (
      activated_timed(
          program_command, args.timeout_s,
          raw_dir / "traffic-codegen.time.txt")
      if codegen_build.returncode == 0 and audit_pass
      and all(source_checks.values())
      and provider_head == PINNED_COMMIT and provider_status == "" else
      subprocess.CompletedProcess(
          program_command, 1, "", "precondition failed"))
  sample_memory("after-sole-traffic-codegen", stop_bytes, memory)
  program_result = parse_last_json(program_run.stdout)
  write_json(raw_dir / "traffic-codegen.json", {
      "command": program_command,
      "returncode": program_run.returncode,
      "stdout": program_run.stdout,
      "stderr": program_run.stderr,
  })
  program_path = program_dir / "existing_shim.program.bin"
  program_ok = bool(
      program_run.returncode == 0 and program_path.is_file()
      and program_result.get("schema_version") ==
          "intel-qwen36-openvino-existing-micro-shim-fuse-v0"
      and program_result.get("openvino_onednn_commit") == PINNED_COMMIT
      and program_result.get("kernel_name") ==
          "iq36_exact_attention_dense_traffic_ceiling"
      and program_result.get("host_define") ==
          "IQ36_COMPONENT_PROGRAM=10"
      and program_result.get("register_file_size") == 128
      and program_result.get("exact_attention_vrt160") is False
      and program_result.get("exact_attention_dual_cohort") is False
      and int(program_result.get("kernel_register_count", -1)) > 0
      and int(program_result.get("kernel_register_count", 129)) <= 128
      and int(program_result.get("kernel_spill_memory_bytes", -1)) == 0
      and int(program_result.get("kernel_local_memory_bytes", -1)) == 0
      and int(program_result.get(
          "kernel_maximum_workgroup_size", -1)) >= WORKGROUP_ITEMS
      and int(program_result.get(
          "kernel_preferred_workgroup_multiple", -1)) == 16
      and program_path.stat().st_size > 0)

  configure_command = [
      str(CMAKE), "-S", str(ROOT / "engine"), "-B", str(BUILD_DIR),
      "-DCMAKE_BUILD_TYPE=Release"]
  configure = run(configure_command, 300)
  build_command = [
      str(CMAKE), "--build", str(BUILD_DIR), "--target", TARGET, "-j1"]
  sample_memory("before-serial-runner-build", preflight_bytes, memory)
  build = activated_timed(
      build_command, args.timeout_s, raw_dir / "runner-build.time.txt")
  write_json(raw_dir / "runner-build.json", {
      "configure": {
          "command": configure_command,
          "returncode": configure.returncode,
          "stdout": configure.stdout,
          "stderr": configure.stderr,
      },
      "build": {
          "command": build_command,
          "returncode": build.returncode,
          "stdout": build.stdout,
          "stderr": build.stderr,
      },
  })
  sample_memory("after-serial-runner-build", stop_bytes, memory)
  executable = BUILD_DIR / TARGET
  build_ok = bool(
      configure.returncode == 0 and build.returncode == 0
      and executable.is_file())

  worker_command = [
      str(executable), str(CAPTURE_PROGRAM), str(program_path),
      "--dense-traffic"]
  sample_memory("before-sole-traffic-worker", preflight_bytes, memory)
  worker = (
      activated_timed(
          worker_command, args.timeout_s, raw_dir / "traffic-worker.time.txt")
      if build_ok and program_ok else
      subprocess.CompletedProcess(
          worker_command, 1, "", "precondition failed"))
  sample_memory("after-sole-traffic-worker", stop_bytes, memory)
  (raw_dir / "traffic-worker.stdout").write_text(
      worker.stdout, encoding="utf-8")
  (raw_dir / "traffic-worker.stderr").write_text(
      worker.stderr, encoding="utf-8")
  write_json(raw_dir / "worker-command.json", {
      "command": worker_command,
      "returncode": worker.returncode,
  })
  result = parse_last_json(worker.stdout)
  worker_resources = parse_time(raw_dir / "traffic-worker.time.txt")
  codegen_resources = {
      "build": parse_time(raw_dir / "codegen-build.time.txt"),
      "program": parse_time(raw_dir / "traffic-codegen.time.txt"),
      "runner_build": parse_time(raw_dir / "runner-build.time.txt"),
  }
  rows = result.get("samples", [])
  latencies = [
      float(row.get("latency_ms", math.nan))
      for row in rows if isinstance(row, dict)]
  try:
    inference = latency_cap_inference(
        latencies, cap=LATENCY_UCB_CAP_MS, min_samples=MIN_SAMPLES,
        seed=215101)
  except ValueError as error:
    inference = {
        "error": str(error),
        "sample_count": len(latencies),
        "sample_count_pass": False,
        "rate_pass": False,
        "cap_ms": LATENCY_UCB_CAP_MS,
    }
  latency_point = inference.get("point_estimate_ms")
  latency_ucb = inference.get("upper_confidence_bound_ms")
  bandwidth_point = (
      PAYLOAD_BYTES / (float(latency_point) * 1.0e6)
      if isinstance(latency_point, (int, float)) and latency_point > 0
      else None)
  bandwidth_lcb = (
      PAYLOAD_BYTES / (float(latency_ucb) * 1.0e6)
      if isinstance(latency_ucb, (int, float)) and latency_ucb > 0
      else None)
  checksum_pass = bool(
      result.get("numeric_pass") is True
      and result.get("checksum_mismatch_count") == 0
      and int(result.get("checksum_nonzero_words", 0)) > 0)
  measurement_checks = [
      check("repository_clean_at_gate", not git["dirty"], git=git),
      check("seq2150a_admits_exactly_one_fixed_traffic_ceiling",
            audit_pass),
      check("fixed_source_runner_and_capture_contract",
            all(source_checks.values()), source_checks=source_checks),
      check("pinned_provider_is_clean",
            provider_head == PINNED_COMMIT and provider_status == "",
            provider_head=provider_head, provider_status=provider_status),
      check("codegen_build_is_serial", codegen_build.returncode == 0),
      check("sole_traffic_program_is_spill_free_and_geometry_valid",
            program_ok, program_result=program_result),
      check("runner_build_is_serial_j1", build_ok,
            build_command=build_command),
      check("sole_standalone_traffic_worker_executes",
            worker.returncode == 0, returncode=worker.returncode),
      check("fixed_two_workgroup_full_payload_shape",
            fixed_result(result)),
      check("twenty_sample_distribution_is_well_formed",
            distribution_pass(result)),
      check("both_payload_streams_have_stable_output_checksums",
            checksum_pass),
      check("runtime_resources_match_codegen",
            resource_pass(result, program_result),
            resources=result.get("resources", {})),
      check("worker_rss_and_swap_are_bounded",
            int(worker_resources.get("maximum_resident_kib", 1 << 62))
                < 2 * 1024 * 1024
            and int(worker_resources.get("swaps", -1)) == 0,
            worker_resources=worker_resources),
      check("memory_guards_never_tripped",
            all(row["pass"] for row in memory),
            minimum_available_bytes=min(
                row["available_bytes"] for row in memory)),
      check("no_graph_plugin_or_model_worker_launched", True),
  ]
  measurement_valid = all(row["pass"] for row in measurement_checks)
  traffic_capacity_pass = bool(
      measurement_valid and inference.get("rate_pass") is True
      and bandwidth_lcb is not None
      and bandwidth_lcb >= BANDWIDTH_LCB_FLOOR_GB_S)
  performance_check = check(
      "one_sided_95pct_latency_ucb_clears_physical_capacity_cap",
      traffic_capacity_pass, inference=inference,
      bandwidth_lcb_gb_s=bandwidth_lcb,
      bandwidth_floor_gb_s=BANDWIDTH_LCB_FLOOR_GB_S)
  checks = [*measurement_checks, performance_check]
  required = all(row["pass"] for row in checks)
  if required:
    verdict = "admit_one_source_bound_package_or_synchronization_cut"
  elif measurement_valid:
    verdict = "close_two_workgroup_exact_family_at_dense_traffic_limit"
  else:
    verdict = "inconclusive_two_workgroup_dense_traffic_measurement"
  sources = [
      {"path": display(path), "sha256": sha256(path)}
      for path in (SOURCE, RUNNER, CODEGEN, SHIMS, AUDIT, CAPTURE_PROGRAM)
  ]
  payload = {
      "schema_version": SCHEMA,
      "workstream": WS,
      "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
      "git": git,
      "verdict": verdict,
      "required_checks_passed": required,
      "measurement_valid": measurement_valid,
      "traffic_capacity_pass": traffic_capacity_pass,
      "source_bound_package_or_synchronization_cut_admitted": required,
      "cache_policy_edit_admitted": False,
      "exact_kernel_implementation_admitted": False,
      "graph_compile_admitted": False,
      "graph_integration_admitted": False,
      "plugin_build_admitted": False,
      "model_worker_admitted": False,
      "product_claim_allowed": False,
      "compiler_workers_launched": 1 if program_run.returncode in {0, 2} else 0,
      "gpu_workers_launched": 1 if worker.returncode in {0, 2} else 0,
      "model_workers_launched": 0,
      "checks": checks,
      "result": result,
      "performance_inference": inference,
      "bandwidth_point_gb_s": bandwidth_point,
      "bandwidth_lcb_gb_s": bandwidth_lcb,
      "latency_ucb_cap_ms": LATENCY_UCB_CAP_MS,
      "bandwidth_lcb_floor_gb_s": BANDWIDTH_LCB_FLOOR_GB_S,
      "worker_resources": worker_resources,
      "compiler_and_build_resources": codegen_resources,
      "memory_preflight_bytes": preflight_bytes,
      "memory_stop_bytes": stop_bytes,
      "memory_samples": memory,
      "worker_command": worker_command,
      "program_codegen": program_result,
      "program_path": display(program_path),
      "program_sha256": sha256(program_path) if program_path.is_file() else None,
      "sources": sources,
  }
  write_json(out_dir / "result.json", payload)
  write_json(out_dir / "correctness.json", {
      "numeric_pass": checksum_pass,
      "checksum_compared_words": result.get("checksum_compared_words"),
      "checksum_mismatch_count": result.get("checksum_mismatch_count"),
      "checksum_nonzero_words": result.get("checksum_nonzero_words"),
      "checksum_hash": result.get("checksum_hash"),
  })
  write_json(out_dir / "performance.json", {
      "samples": rows,
      "inference": inference,
      "bandwidth_point_gb_s": bandwidth_point,
      "bandwidth_lcb_gb_s": bandwidth_lcb,
  })
  write_json(out_dir / "resources.json", {
      "runtime_kernel_resources": result.get("resources", {}),
      "program_codegen": program_result,
      "worker_resources": worker_resources,
      "compiler_and_build_resources": codegen_resources,
  })
  write_json(out_dir / "manifest.json", {
      "schema_version": "intel-qwen36-artifact-manifest-v1",
      "workstream": WS,
      "git_commit": git["commit"],
      "verdict": verdict,
      "sources": sources,
      "files": [
          "result.json", "correctness.json", "performance.json",
          "resources.json", "summary.md", "raw/codegen-build.json",
          "raw/codegen-build.time.txt", "raw/traffic-codegen.json",
          "raw/traffic-codegen.time.txt", "raw/runner-build.json",
          "raw/runner-build.time.txt", "raw/traffic-worker.stdout",
          "raw/traffic-worker.stderr", "raw/traffic-worker.time.txt",
          "raw/worker-command.json",
      ],
  })
  (out_dir / "summary.md").write_text(
      summary(payload), encoding="utf-8")
  print(json.dumps({
      "artifact": display(out_dir),
      "verdict": verdict,
      "measurement_valid": measurement_valid,
      "traffic_capacity_pass": traffic_capacity_pass,
      "latency_median_ms": latency_point,
      "latency_ucb_ms": latency_ucb,
      "latency_cap_ms": LATENCY_UCB_CAP_MS,
      "bandwidth_point_gb_s": bandwidth_point,
      "bandwidth_lcb_gb_s": bandwidth_lcb,
      "bandwidth_floor_gb_s": BANDWIDTH_LCB_FLOOR_GB_S,
      "cache_policy_edit_admitted": False,
      "exact_kernel_implementation_admitted": False,
      "model_worker_launched": False,
  }, separators=(",", ":")))
  return 0 if required else 2


if __name__ == "__main__":
  raise SystemExit(main())
