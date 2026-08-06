#!/usr/bin/env python3
"""Gate the exact 128k KQ/softmax/VS decomposition before one compiler probe.

This is a low-memory standalone component gate.  It proves the intermediate
ABI and chronological recurrence bitwise, rejects global staging, measures a
matched softmax traffic control, and checks a conservative 48-subgroup SLM
budget.  A pass admits only one triple-cohort compiler/resource probe; it does
not admit graph integration, a plugin build, or a model worker.
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

from iq36_perf_inference import (
    bootstrap_median_bound,
    dispersion_diagnostic,
    latency_cap_inference,
)


ROOT = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = (
    "intel-qwen36-openvino-exact-attention-three-stage-component-gate-v1")
SOURCE = ROOT / "engine/gpu/opencl/exact_score_staging_component.cl"
RUNNER = ROOT / "engine/tools/exact_score_staging_component.cpp"
CODEGEN = ROOT / "engine/tools/openvino_moe_micro_codegen.cpp"
SHIMS = ROOT / "engine/openvino/custom/iq36_decode_microkernel_shims.cl"
TARGET_CONTRACT = ROOT / "contracts/intel-qwen36-target-contract.json"
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
KILL_NUMBER_MS_PER_LAYER = 0.1175998
STAGE_BALANCE_RATIO_CAP = 1.25
SUBGROUP_SIZE = 16
TRIPLE_COHORT_SUBGROUPS = 48
TRIPLE_COHORT_WORKGROUP_ITEMS = (
    SUBGROUP_SIZE * TRIPLE_COHORT_SUBGROUPS)
DEVICE_MAX_SUBGROUPS = 64
DEVICE_MAX_WORKGROUP_ITEMS = 1024
DEVICE_LOCAL_MEMORY_BYTES = 128 * 1024

# Conservative explicit local-memory slab for a future on-chip kernel.  The
# output reuses the raw-score region only after the final pipeline barrier.
SLM_BUDGET = {
    "query_f16": 256 * 16 * 2,
    "raw_score_f32_double_buffer": 2 * 256 * 16 * 4,
    "normalized_score_f16_double_buffer": 2 * 256 * 16 * 2,
    "softmax_max_and_final_sum": 2 * 256 * 4,
    "accumulator_rescale_double_buffer": 2 * 256 * 4,
    "ugemm_scratch": 1,
    "output_incremental_after_raw_score_reuse": 0,
}
SLM_UNPADDED_BYTES = sum(SLM_BUDGET.values())
SLM_PADDED_CEILING_BYTES = 61_472
TWO_WORKGROUP_SLM_BYTES = 2 * SLM_PADDED_CEILING_BYTES

EXPECTED = {
    "raw_score_bytes": 16 * 1024**2,
    "normalized_score_bytes": 8 * 1024**2,
    "accumulator_rescale_bytes": 1024 * 1024,
    "final_sum_bytes": 2048,
    "global_intermediate_round_trip_bytes": 52_432_896,
    "mandatory_key_value_payload_bytes": 256 * 1024**2,
}

PROGRAMS = {
    "capture": {
        "define": "IQ36_COMPONENT_PROGRAM=1",
        "kernel": "iq36_exact_score_serial_capture",
        "dual": False,
        "required_workgroup_items": 256,
    },
    "staged": {
        "define": "IQ36_COMPONENT_PROGRAM=3",
        "kernel": "iq36_exact_score_kq_stage",
        "dual": False,
        "required_workgroup_items": 256,
    },
    "dual": {
        "define": "IQ36_COMPONENT_PROGRAM=4",
        "kernel": "iq36_exact_score_dual_cohort",
        "dual": True,
        "required_workgroup_items": 512,
    },
    "softmax": {
        "define": "IQ36_COMPONENT_PROGRAM=5",
        "kernel": "iq36_exact_score_softmax_stage",
        "dual": False,
        "required_workgroup_items": 256,
    },
    "traffic": {
        "define": "IQ36_COMPONENT_PROGRAM=7",
        "kernel": "iq36_exact_score_softmax_traffic",
        "dual": False,
        "required_workgroup_items": 256,
    },
    "vs": {
        "define": "IQ36_COMPONENT_PROGRAM=6",
        "kernel": "iq36_exact_score_vs_stage",
        "dual": False,
        "required_workgroup_items": 256,
    },
}


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


def activated_timed(
    command: list[str], timeout: int, time_path: Path,
) -> subprocess.CompletedProcess[str]:
  shell_command = (
      f"source {shlex.quote(str(ENV_SCRIPT))} >/dev/null 2>&1 && "
      f"/usr/bin/time -v -o {shlex.quote(str(time_path))} " +
      " ".join(shlex.quote(part) for part in command))
  return run(["bash", "-lc", shell_command], timeout)


def lower_floor_inference(
    values: list[float], floor: float, seed: int,
) -> dict[str, Any]:
  if len(values) != MIN_SAMPLES or any(
      not math.isfinite(value) or value <= 0.0 for value in values):
    return {
        "error": "samples must be 20 positive finite values",
        "rate_pass": False,
        "floor_ms": floor,
    }
  lower = bootstrap_median_bound(values, side="lower", seed=seed)
  return {
      "method": "one_sided_percentile_bootstrap_median",
      "confidence": 0.95,
      "bootstrap_resamples": 20_000,
      "bootstrap_seed": seed,
      "sample_count": len(values),
      "minimum_sample_count": MIN_SAMPLES,
      "point_estimate_ms": statistics.median(values),
      "lower_confidence_bound_ms": lower,
      "floor_ms": floor,
      "sample_count_pass": True,
      "rate_pass": lower >= floor,
      "dispersion": dispersion_diagnostic(values),
  }


def fixed_result(result: dict[str, Any]) -> bool:
  return bool(
      result.get("schema_version") ==
          "intel-qwen36-exact-attention-three-stage-component-v1"
      and result.get("algorithm") ==
          "generated_m256_n16_kq_softmax_vs_decomposition"
      and result.get("context_tokens") == 131072
      and result.get("head_dim") == 256
      and result.get("query_heads") == 16
      and result.get("kv_heads") == 2
      and result.get("gqa_group") == 8
      and result.get("key_block") == 256
      and result.get("dual_useful_groups") == 2
      and result.get("kq_useful_groups") == 1024
      and result.get("softmax_useful_groups") == 2
      and result.get("vs_useful_groups") == 2
      and all(result.get(key) == value for key, value in EXPECTED.items())
      and result.get("output_compared_values") == 4096
      and result.get("sample_count") == MIN_SAMPLES
      and result.get("schedule") ==
          "interleaved_dual_three_stage_three_stage_dual")


def distribution_pass(result: dict[str, Any]) -> bool:
  rows = result.get("paired_samples", [])
  if not isinstance(rows, list) or len(rows) != MIN_SAMPLES:
    return False
  for index, row in enumerate(rows):
    if row.get("sample") != index:
      return False
    if row.get("order") != (
        "dual_three_stage" if index % 2 == 0 else "three_stage_dual"):
      return False
    staged = row.get("three_stage", {})
    expected_stage_keys = {
        "kq_ms", "softmax_ms", "vs_ms", "total_ms",
        "global_staged_bottleneck_ms", "owner_residual_proxy_ms",
        "softmax_arithmetic_ms", "projected_onchip_bottleneck_proxy_ms",
    }
    if set(staged) != expected_stage_keys:
      return False
    names = (
        "dual_ms", "owner_ms", "softmax_traffic_ms",
        "global_staging_penalty_ms",
        "projected_onchip_saving_proxy_ms")
    try:
      values = {name: float(row[name]) for name in names}
      values.update({name: float(staged[name]) for name in staged})
    except (KeyError, TypeError, ValueError):
      return False
    if any(not math.isfinite(value) for value in values.values()):
      return False
    positive = (
        "dual_ms", "owner_ms", "softmax_traffic_ms", "kq_ms",
        "softmax_ms", "vs_ms", "total_ms",
        "global_staged_bottleneck_ms", "owner_residual_proxy_ms",
        "softmax_arithmetic_ms", "projected_onchip_bottleneck_proxy_ms",
        "global_staging_penalty_ms", "projected_onchip_saving_proxy_ms")
    if any(values[name] <= 0.0 for name in positive):
      return False
    proxy = max(
        values["kq_ms"], values["softmax_ms"],
        values["owner_residual_proxy_ms"])
    identities = (
        (values["owner_residual_proxy_ms"],
         values["owner_ms"] - values["softmax_ms"]),
        (values["softmax_arithmetic_ms"],
         values["softmax_ms"] - values["softmax_traffic_ms"]),
        (values["global_staging_penalty_ms"],
         values["total_ms"] - values["dual_ms"]),
        (values["projected_onchip_bottleneck_proxy_ms"], proxy),
        (values["projected_onchip_saving_proxy_ms"],
         values["dual_ms"] - proxy),
    )
    if any(not math.isclose(
        measured, derived, rel_tol=0.0, abs_tol=3.0e-6)
        for measured, derived in identities):
      return False
  return True


def resources_pass(result: dict[str, Any]) -> bool:
  resources = result.get("resources", {})
  expected_names = {"dual", "kq", "owner", "softmax", "softmax_traffic", "vs"}
  if set(resources) != expected_names:
    return False
  required_items = {
      "dual": 512,
      "kq": 256,
      "owner": 256,
      "softmax": 256,
      "softmax_traffic": 256,
      "vs": 256,
  }
  return all(
      0 < int(resources[name].get("register_count", -1)) <= 128
      and int(resources[name].get("spill_memory_bytes", -1)) == 0
      and 0 < int(resources[name].get("local_memory_bytes", -1)) <= 65_536
      and int(resources[name].get("maximum_workgroup_items", -1))
          >= required_items[name]
      and int(resources[name].get("preferred_workgroup_multiple", -1)) == 16
      for name in expected_names)


def summary(payload: dict[str, Any]) -> str:
  result = payload["result"]
  inference = payload["performance_inference"]
  resources = result.get("resources", {})
  return "\n".join([
      "# Exact-attention three-stage component gate",
      "",
      f"Verdict: **{payload['verdict']}**. Required checks: "
      f"`{str(payload['required_checks_passed']).lower()}`.",
      "",
      f"- three-stage / owner output mismatches: "
      f"`{result.get('output_mismatch_count')} / "
      f"{result.get('owner_output_mismatch_count')}`",
      f"- global-staging penalty median / 95% LCB: "
      f"`{inference['global_staging_penalty'].get('point_estimate_ms')} / "
      f"{inference['global_staging_penalty'].get('lower_confidence_bound_ms')} "
      "ms/layer`",
      f"- softmax arithmetic median / 95% LCB: "
      f"`{inference['softmax_arithmetic'].get('point_estimate_ms')} / "
      f"{inference['softmax_arithmetic'].get('lower_confidence_bound_ms')} "
      "ms/layer`",
      f"- projected on-chip saving proxy median / 95% LCB / kill number: "
      f"`{inference['projected_onchip_saving'].get('point_estimate_ms')} / "
      f"{inference['projected_onchip_saving'].get('lower_confidence_bound_ms')} "
      f"/ {KILL_NUMBER_MS_PER_LAYER} ms/layer`",
      f"- KQ/owner-residual balance median / 95% UCB / cap: "
      f"`{inference['stage_balance'].get('point_estimate_ms')} / "
      f"{inference['stage_balance'].get('upper_confidence_bound_ms')} / "
      f"{STAGE_BALANCE_RATIO_CAP}`",
      f"- current dual / softmax / VS resources: "
      f"`{resources.get('dual')} / {resources.get('softmax')} / "
      f"{resources.get('vs')}`",
      f"- projected padded SLM / two-WG use / margin: "
      f"`{SLM_PADDED_CEILING_BYTES} / {TWO_WORKGROUP_SLM_BYTES} / "
      f"{DEVICE_LOCAL_MEMORY_BYTES - TWO_WORKGROUP_SLM_BYTES} B`",
      f"- peak RSS / swaps: "
      f"`{payload['worker_resources'].get('maximum_resident_kib')} KiB / "
      f"{payload['worker_resources'].get('swaps')}`",
      "",
      "Global intermediate staging is rejected.  A pass admits only one",
      "48-subgroup on-chip compiler/resource probe; all graph, plugin, and",
      "model work remains closed.",
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
      SOURCE, RUNNER, CODEGEN, SHIMS, TARGET_CONTRACT, PINNED_SOURCE,
      PINNED_BUILD / "src/libdnnl.a", CXX, CMAKE, ENV_SCRIPT)
  missing = [str(path) for path in required_paths if not path.exists()]
  if missing:
    raise SystemExit("missing three-stage gate inputs: " + ", ".join(missing))

  git = git_state(out_dir)
  source_text = SOURCE.read_text(encoding="utf-8")
  runner_text = RUNNER.read_text(encoding="utf-8")
  codegen_text = CODEGEN.read_text(encoding="utf-8")
  shim_text = SHIMS.read_text(encoding="utf-8")
  target = load_json(TARGET_CONTRACT)
  provider_head = run(
      ["git", "-C", str(PINNED_SOURCE), "rev-parse", "HEAD"],
      30).stdout.strip()
  provider_status = run([
      "git", "-C", str(PINNED_SOURCE), "status", "--short", "--",
      "src/gpu/intel/gemm/jit", "src/gpu/intel/jit/config",
      "third_party/ngen",
  ], 30).stdout.strip()
  clinfo_run = activated_timed(
      ["clinfo"], 60, raw_dir / "clinfo.time.txt")
  (raw_dir / "clinfo.stdout").write_text(
      clinfo_run.stdout, encoding="utf-8")
  (raw_dir / "clinfo.stderr").write_text(
      clinfo_run.stderr, encoding="utf-8")
  clinfo = clinfo_run.stdout
  expected_device = str(target.get("runtime", {}).get("opencl_device", ""))
  source_checks = {
      "three_exact_isolated_stage_programs":
          all(marker in source_text for marker in (
              "IQ36_COMPONENT_SOFTMAX_STAGE",
              "IQ36_COMPONENT_VS_STAGE",
              "IQ36_COMPONENT_SOFTMAX_TRAFFIC",
              "__kernel void iq36_exact_score_softmax_stage(",
              "__kernel void iq36_exact_score_vs_stage(",
              "__kernel void iq36_exact_score_softmax_traffic(")),
      "chronological_softmax_and_vs_recurrence":
          "old_running_max, running_max, iq36_component_rescale"
              in source_text
          and "tile_hbroadcast_mul(&accumulator, accumulator_scale);"
              in source_text
          and source_text.count(
              "iq36_component_accumulator_tile chunk_accumulator = "
              "ugemm_vs(") >= 3,
      "matched_traffic_control_retains_intermediate_abi":
          "Exact-minus-control timing isolates" in source_text
          and source_text.count(
              "__global half* normalized_score") >= 2
          and source_text.count(
              "__global float* accumulator_rescale") >= 2,
      "runner_has_fixed_interleaved_three_stage_mode":
          "constexpr int kSamples = 20;" in runner_text
          and "--three-stage" in runner_text
          and "interleaved_dual_three_stage_three_stage_dual" in runner_text
          and "projected_onchip_saving_proxy_ms" in runner_text,
      "generated_m256_n16_packages_are_retained":
          "#define ugemm_kq_wg_tile_m 256" in shim_text
          and "#define ugemm_kq_wg_tile_n 16" in shim_text
          and "#define ugemm_vs_wg_tile_m 256" in shim_text
          and "#define ugemm_vs_wg_tile_n 16" in shim_text
          and "#define ugemm_kq_sg_per_wg_m 16" in shim_text
          and "#define ugemm_vs_sg_per_wg_m 16" in shim_text,
      "codegen_retains_dual_and_native_controls":
          "--exact-attention-dual-cohort" in codegen_text
          and "const bool fixed_dual_cohort" in codegen_text
          and "const bool native_control" in codegen_text,
  }
  device_checks = {
      "target_device":
          bool(expected_device) and expected_device in clinfo,
      "named_subset_barrier":
          "cl_khr_subgroup_named_barrier" in clinfo,
      "independent_forward_progress":
          "Sub-group independent forward progress        Yes" in clinfo,
      "workgroup_1024":
          "Max work group size                             1024" in clinfo,
      "subgroups_64":
          "Max sub-groups per work group                   64" in clinfo,
      "local_memory_128k":
          "Local memory size                               131072" in clinfo,
  }
  sample_memory("after-source-device-audit", stop_bytes, memory)

  codegen_binary = raw_dir / "openvino-micro-codegen"
  codegen_build_command_value = codegen_build_command(codegen_binary)
  codegen_build = activated_timed(
      codegen_build_command_value, args.timeout_s,
      raw_dir / "codegen-build.time.txt")
  write_json(raw_dir / "codegen-build.json", {
      "command": codegen_build_command_value,
      "returncode": codegen_build.returncode,
      "stdout": codegen_build.stdout,
      "stderr": codegen_build.stderr,
  })
  sample_memory("after-codegen-build", stop_bytes, memory)

  default_probe = (
      activated_timed(
          [str(codegen_binary), "--provider-commit", PINNED_COMMIT],
          args.timeout_s, raw_dir / "default-codegen.time.txt")
      if codegen_build.returncode == 0 else
      subprocess.CompletedProcess([], 1, "", "codegen build failed"))
  default_result = parse_last_json(default_probe.stdout)
  write_json(raw_dir / "default-codegen.json", {
      "command": list(default_probe.args),
      "returncode": default_probe.returncode,
      "stdout": default_probe.stdout,
      "stderr": default_probe.stderr,
  })

  codegen_runs: dict[str, dict[str, Any]] = {}
  codegen_results: dict[str, dict[str, Any]] = {}
  program_paths: dict[str, Path] = {}
  child_resources: list[dict[str, Any]] = [
      parse_time(raw_dir / "clinfo.time.txt"),
      parse_time(raw_dir / "codegen-build.time.txt"),
      parse_time(raw_dir / "default-codegen.time.txt"),
  ]
  for label, specification in PROGRAMS.items():
    program_dir = raw_dir / "programs" / label
    program_dir.mkdir(parents=True, exist_ok=False)
    command = [str(codegen_binary)]
    if specification["dual"]:
      command.append("--exact-attention-dual-cohort")
    command.extend([
        "--fuse-existing-shim", str(SHIMS),
        "--host-source", str(SOURCE),
        "--kernel-name", str(specification["kernel"]),
        "--host-define", str(specification["define"]),
        "--register-file-size", "128",
        "--provider-commit", PINNED_COMMIT,
        "--dump-dir", str(program_dir),
    ])
    time_path = raw_dir / f"{label}-codegen.time.txt"
    completed = (
        activated_timed(command, args.timeout_s, time_path)
        if codegen_build.returncode == 0 else
        subprocess.CompletedProcess(
            command, 1, "", "codegen build failed"))
    codegen_runs[label] = {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    codegen_results[label] = parse_last_json(completed.stdout)
    program_paths[label] = program_dir / "existing_shim.program.bin"
    child_resources.append(parse_time(time_path))
    sample_memory(f"after-{label}-program", stop_bytes, memory)
  write_json(raw_dir / "program-codegen.json", codegen_runs)

  configure_command = [
      str(CMAKE), "-S", str(ROOT / "engine"), "-B", str(BUILD_DIR),
      "-DCMAKE_BUILD_TYPE=Release"]
  configure = run(configure_command, 300)
  build_command = [
      str(CMAKE), "--build", str(BUILD_DIR), "--target", TARGET, "-j1"]
  build = activated_timed(
      build_command, args.timeout_s, raw_dir / "component-build.time.txt")
  child_resources.append(parse_time(raw_dir / "component-build.time.txt"))
  write_json(raw_dir / "component-build.json", {
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
  sample_memory("after-component-build", stop_bytes, memory)

  executable = BUILD_DIR / TARGET
  build_ok = bool(
      configure.returncode == 0 and build.returncode == 0
      and executable.is_file())
  codegen_ok = bool(
      codegen_build.returncode == 0
      and provider_head == PINNED_COMMIT
      and provider_status == ""
      and default_probe.returncode == 0
      and default_result.get("mode") == "moe"
      and default_result.get("register_file_size") == 256
      and len(default_result.get("packages", [])) == 3
      and all(
          codegen_runs[label]["returncode"] == 0
          and program_paths[label].is_file()
          and codegen_results[label].get("schema_version") ==
              "intel-qwen36-openvino-existing-micro-shim-fuse-v0"
          and codegen_results[label].get("kernel_name") ==
              specification["kernel"]
          and codegen_results[label].get("host_define") ==
              specification["define"]
          and codegen_results[label].get("register_file_size") == 128
          and codegen_results[label].get(
              "exact_attention_dual_cohort") is specification["dual"]
          and int(codegen_results[label].get(
              "kernel_spill_memory_bytes", -1)) == 0
          and 0 < int(codegen_results[label].get(
              "kernel_local_memory_bytes", -1)) <= 65_536
          and int(codegen_results[label].get(
              "kernel_maximum_workgroup_size", -1)) >=
              int(specification["required_workgroup_items"])
          and program_paths[label].stat().st_size > 0
          for label, specification in PROGRAMS.items()))

  worker_command = [
      str(executable),
      str(program_paths["capture"]),
      str(program_paths["dual"]),
      str(program_paths["staged"]),
      str(program_paths["softmax"]),
      str(program_paths["traffic"]),
      str(program_paths["vs"]),
      "--three-stage",
  ]
  sample_memory("before-component-worker", preflight_bytes, memory)
  worker = (
      activated_timed(
          worker_command, args.timeout_s, raw_dir / "component.time.txt")
      if build_ok and codegen_ok else
      subprocess.CompletedProcess(
          worker_command, 1, "", "build or codegen failed"))
  sample_memory("after-component-worker", stop_bytes, memory)
  (raw_dir / "component.stdout").write_text(
      worker.stdout, encoding="utf-8")
  (raw_dir / "component.stderr").write_text(
      worker.stderr, encoding="utf-8")
  write_json(raw_dir / "worker-command.json", {
      "command": worker_command,
      "returncode": worker.returncode,
  })
  worker_resources = parse_time(raw_dir / "component.time.txt")
  child_resources.append(worker_resources)
  result = parse_last_json(worker.stdout)

  rows = result.get("paired_samples", [])
  global_penalties = [
      float(row.get("global_staging_penalty_ms", math.nan))
      for row in rows if isinstance(row, dict)]
  softmax_arithmetic = [
      float(row.get("three_stage", {}).get(
          "softmax_arithmetic_ms", math.nan))
      for row in rows if isinstance(row, dict)]
  projected_savings = [
      float(row.get("projected_onchip_saving_proxy_ms", math.nan))
      for row in rows if isinstance(row, dict)]
  balance_ratios: list[float] = []
  for row in rows:
    staged = row.get("three_stage", {}) if isinstance(row, dict) else {}
    kq = float(staged.get("kq_ms", math.nan))
    residual = float(staged.get("owner_residual_proxy_ms", math.nan))
    balance_ratios.append(
        max(kq, residual) / min(kq, residual)
        if min(kq, residual) > 0.0 else math.nan)

  inference = {
      "global_staging_penalty":
          lower_floor_inference(global_penalties, 0.0, 214401),
      "softmax_arithmetic":
          lower_floor_inference(softmax_arithmetic, 0.0, 214402),
      "projected_onchip_saving":
          lower_floor_inference(
              projected_savings, KILL_NUMBER_MS_PER_LAYER, 214403),
      "stage_balance":
          latency_cap_inference(
              balance_ratios, cap=STAGE_BALANCE_RATIO_CAP,
              min_samples=MIN_SAMPLES, seed=214404)
          if len(balance_ratios) == MIN_SAMPLES
          and all(math.isfinite(value) and value > 0.0
                  for value in balance_ratios) else
          {"error": "invalid balance ratios", "rate_pass": False},
  }
  numeric_pass = bool(
      result.get("numeric_pass") is True
      and result.get("output_mismatch_count") == 0
      and result.get("owner_output_mismatch_count") == 0)
  design_fit = bool(
      SLM_UNPADDED_BYTES <= SLM_PADDED_CEILING_BYTES
      and TWO_WORKGROUP_SLM_BYTES <= DEVICE_LOCAL_MEMORY_BYTES
      and TRIPLE_COHORT_WORKGROUP_ITEMS <= DEVICE_MAX_WORKGROUP_ITEMS
      and TRIPLE_COHORT_SUBGROUPS <= DEVICE_MAX_SUBGROUPS)
  max_child_rss = max(
      int(resource.get("maximum_resident_kib", 0))
      for resource in child_resources)
  total_child_swaps = sum(
      int(resource.get("swaps", 0)) for resource in child_resources)

  checks = [
      check("repository_clean_at_gate", not git["dirty"], git=git),
      check("fixed_exact_three_stage_source_contract",
            all(source_checks.values()), source_checks=source_checks),
      check("target_device_supports_48_subgroup_named_pipeline",
            clinfo_run.returncode == 0 and all(device_checks.values()),
            device_checks=device_checks),
      check("pinned_codegen_builds_and_all_controls_are_spill_free",
            codegen_ok, provider_head=provider_head,
            provider_status=provider_status,
            default_result=default_result,
            codegen_results=codegen_results),
      check("component_build_is_serial_j1", build_ok,
            build_command=build_command),
      check("standalone_component_worker_executes",
            worker.returncode == 0, returncode=worker.returncode),
      check("fixed_128k_decomposition_shape", fixed_result(result)),
      check("twenty_pair_interleaved_distribution",
            distribution_pass(result)),
      check("dual_owner_and_three_stage_outputs_are_bitwise_exact",
            numeric_pass),
      check("current_stage_kernels_are_spill_free_and_resource_bounded",
            resources_pass(result), resources=result.get("resources", {})),
      check("global_intermediate_staging_is_confidently_slower",
            inference["global_staging_penalty"].get("rate_pass") is True,
            inference=inference["global_staging_penalty"]),
      check("matched_control_resolves_positive_softmax_arithmetic",
            inference["softmax_arithmetic"].get("rate_pass") is True,
            inference=inference["softmax_arithmetic"]),
      check("projected_onchip_overlap_lcb_clears_layer_kill_number",
            inference["projected_onchip_saving"].get("rate_pass") is True,
            inference=inference["projected_onchip_saving"]),
      check("kq_and_owner_residual_proxy_are_balanced",
            inference["stage_balance"].get("rate_pass") is True,
            inference=inference["stage_balance"]),
      check("conservative_48_subgroup_two_workgroup_design_fits",
            design_fit, slm_budget=SLM_BUDGET,
            slm_unpadded_bytes=SLM_UNPADDED_BYTES,
            slm_padded_ceiling_bytes=SLM_PADDED_CEILING_BYTES,
            two_workgroup_slm_bytes=TWO_WORKGROUP_SLM_BYTES,
            device_local_memory_bytes=DEVICE_LOCAL_MEMORY_BYTES,
            workgroup_items=TRIPLE_COHORT_WORKGROUP_ITEMS,
            subgroups=TRIPLE_COHORT_SUBGROUPS),
      check("worker_rss_and_swap_are_bounded",
            int(worker_resources.get("maximum_resident_kib", 1 << 62))
                < 4 * 1024 * 1024
            and int(worker_resources.get("swaps", -1)) == 0
            and total_child_swaps == 0,
            worker_resources=worker_resources,
            maximum_child_rss_kib=max_child_rss,
            total_child_swaps=total_child_swaps),
      check("memory_guards_never_tripped",
            all(row["pass"] for row in memory),
            minimum_available_bytes=min(
                row["available_bytes"] for row in memory)),
      check("no_graph_plugin_or_model_worker_launched", True),
  ]
  required = all(row["pass"] for row in checks)
  verdict = (
      "admit_one_exact_attention_triple_cohort_codegen_gate"
      if required else
      "reject_exact_attention_triple_cohort_before_codegen")
  sources = [
      {"path": display(path), "sha256": sha256(path)}
      for path in (SOURCE, RUNNER, CODEGEN, SHIMS, TARGET_CONTRACT)]
  payload = {
      "schema_version": SCHEMA,
      "workstream": WS,
      "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
      "git": git,
      "verdict": verdict,
      "required_checks_passed": required,
      "global_staging_route_rejected":
          inference["global_staging_penalty"].get("rate_pass") is True,
      "compiler_resource_probe_admitted": required,
      "component_implementation_admitted": False,
      "graph_integration_admitted": False,
      "plugin_build_admitted": False,
      "model_worker_admitted": False,
      "product_claim_allowed": False,
      "standalone_component_worker_launched": worker.returncode in {0, 2},
      "model_worker_launched": False,
      "checks": checks,
      "result": result,
      "performance_inference": inference,
      "kill_number_ms_per_layer": KILL_NUMBER_MS_PER_LAYER,
      "stage_balance_ratio_cap": STAGE_BALANCE_RATIO_CAP,
      "projected_resource_contract": {
          "subgroups": TRIPLE_COHORT_SUBGROUPS,
          "workgroup_items": TRIPLE_COHORT_WORKGROUP_ITEMS,
          "slm_budget": SLM_BUDGET,
          "slm_unpadded_bytes": SLM_UNPADDED_BYTES,
          "slm_padded_ceiling_bytes": SLM_PADDED_CEILING_BYTES,
          "two_workgroup_slm_bytes": TWO_WORKGROUP_SLM_BYTES,
          "device_local_memory_bytes": DEVICE_LOCAL_MEMORY_BYTES,
          "two_workgroup_margin_bytes":
              DEVICE_LOCAL_MEMORY_BYTES - TWO_WORKGROUP_SLM_BYTES,
          "output_storage_contract":
              "reuse raw-score slab only after final pipeline barrier",
      },
      "worker_resources": worker_resources,
      "maximum_child_rss_kib": max_child_rss,
      "total_child_swaps": total_child_swaps,
      "worker_command": worker_command,
      "codegen_results": codegen_results,
      "provider_head": provider_head,
      "provider_status": provider_status,
      "memory_preflight_bytes": preflight_bytes,
      "memory_stop_bytes": stop_bytes,
      "memory_samples": memory,
      "sources": sources,
  }
  write_json(out_dir / "result.json", payload)
  write_json(out_dir / "correctness.json", {
      "numeric_pass": numeric_pass,
      "output_compared_values": result.get("output_compared_values"),
      "output_mismatch_count": result.get("output_mismatch_count"),
      "owner_output_mismatch_count":
          result.get("owner_output_mismatch_count"),
  })
  write_json(out_dir / "performance.json", {
      "paired_samples": rows,
      "inference": inference,
  })
  write_json(out_dir / "resources.json", {
      "runtime_kernel_resources": result.get("resources", {}),
      "codegen_results": codegen_results,
      "projected_resource_contract":
          payload["projected_resource_contract"],
      "worker_resources": worker_resources,
      "maximum_child_rss_kib": max_child_rss,
      "total_child_swaps": total_child_swaps,
  })
  write_json(out_dir / "manifest.json", {
      "schema_version": "intel-qwen36-artifact-manifest-v1",
      "workstream": WS,
      "git_commit": git["commit"],
      "verdict": verdict,
      "sources": sources,
      "files": [
          "result.json", "correctness.json", "performance.json",
          "resources.json", "summary.md", "raw/clinfo.stdout",
          "raw/clinfo.stderr", "raw/clinfo.time.txt",
          "raw/codegen-build.json", "raw/codegen-build.time.txt",
          "raw/default-codegen.json", "raw/default-codegen.time.txt",
          "raw/program-codegen.json", "raw/component-build.json",
          "raw/component-build.time.txt", "raw/component.stdout",
          "raw/component.stderr", "raw/component.time.txt",
          "raw/worker-command.json",
      ],
  })
  (out_dir / "summary.md").write_text(
      summary(payload), encoding="utf-8")
  print(json.dumps({
      "artifact": display(out_dir),
      "verdict": verdict,
      "numeric_pass": numeric_pass,
      "global_staging_penalty_lcb_ms":
          inference["global_staging_penalty"].get(
              "lower_confidence_bound_ms"),
      "softmax_arithmetic_lcb_ms":
          inference["softmax_arithmetic"].get(
              "lower_confidence_bound_ms"),
      "projected_onchip_saving_lcb_ms":
          inference["projected_onchip_saving"].get(
              "lower_confidence_bound_ms"),
      "stage_balance_ucb":
          inference["stage_balance"].get(
              "upper_confidence_bound_ms"),
      "compiler_resource_probe_admitted": required,
      "model_worker_launched": False,
  }, separators=(",", ":")))
  return 0 if required else 2


if __name__ == "__main__":
  raise SystemExit(main())
