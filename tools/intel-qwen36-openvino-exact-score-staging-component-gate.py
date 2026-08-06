#!/usr/bin/env python3
"""Gate the exact 128k generated-KQ full-score staging component."""

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
SOURCE = ROOT / "engine/gpu/opencl/exact_score_staging_component.cl"
RUNNER = ROOT / "engine/tools/exact_score_staging_component.cpp"
CODEGEN = ROOT / "engine/tools/openvino_moe_micro_codegen.cpp"
SHIMS = ROOT / "engine/openvino/custom/iq36_decode_microkernel_shims.cl"
BOUNDARIES = ROOT / "engine/boundaries.json"
SOURCE_BOUND = ROOT / (
    "output/openvino-exact-score-staging-bound-"
    "20260723Tseq2125c-clean/bound.json")
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
DELTA_CAP_MS = -0.1175998
MIN_SAMPLES = 20
EXPECTED_RAW_SCORE_BYTES = 16 * 1024**2
EXPECTED_RAW_SCORE_ROUND_TRIP_BYTES = 32 * 1024**2
PROGRAMS = {
    "capture": {
        "define": "IQ36_COMPONENT_PROGRAM=1",
        "kernel": "iq36_exact_score_serial_capture",
    },
    "fused": {
        "define": "IQ36_COMPONENT_PROGRAM=2",
        "kernel": "iq36_exact_score_fused",
    },
    "staged": {
        "define": "IQ36_COMPONENT_PROGRAM=3",
        "kernel": "iq36_exact_score_kq_stage",
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
    environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
  try:
    return subprocess.run(
        command, cwd=ROOT, check=False, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=timeout,
        env=environment)
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


def environment() -> dict[str, Any]:
  commands = {
      "hostname": ["hostname"],
      "kernel": ["uname", "-a"],
      "bios_version": [
          "bash", "-lc", "head -n 1 /sys/class/dmi/id/bios_version"],
      "opencl": [
          "bash", "-lc",
          f"source {shlex.quote(str(ENV_SCRIPT))} >/dev/null 2>&1 && "
          "clinfo -l"],
  }
  result: dict[str, Any] = {}
  for name, command in commands.items():
    completed = run(command, 60)
    result[name] = {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }
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


def signed_delta_cap_inference(
    values: list[float], cap: float,
) -> dict[str, Any]:
  if len(values) != MIN_SAMPLES or any(
      not math.isfinite(value) for value in values):
    return {
        "error": "signed delta samples must be 20 finite values",
        "rate_pass": False,
        "cap_ms": cap,
    }
  translation = max(1.0, 1.0 - min(values), 1.0 - cap)
  translated = latency_cap_inference(
      [value + translation for value in values],
      cap=cap + translation, min_samples=MIN_SAMPLES)
  point = float(translated["point_estimate_ms"]) - translation
  upper = float(translated["upper_confidence_bound_ms"]) - translation
  median = statistics.median(values)
  mad = statistics.median(abs(value - median) for value in values)
  return {
      "method": (
          "translation_invariant_one_sided_percentile_bootstrap_median"),
      "confidence": translated["confidence"],
      "bootstrap_resamples": translated["bootstrap_resamples"],
      "bootstrap_seed": translated["bootstrap_seed"],
      "sample_count": translated["sample_count"],
      "minimum_sample_count": translated["minimum_sample_count"],
      "point_estimate_ms": point,
      "upper_confidence_bound_ms": upper,
      "cap_ms": cap,
      "sample_count_pass": translated["sample_count_pass"],
      "rate_pass": translated["sample_count_pass"] and upper <= cap,
      "translation_ms": translation,
      "dispersion": {
          "metric": "median_absolute_deviation_ms",
          "mad_ms": mad,
          "promotion_gate": False,
      },
  }


def fixed_result(result: dict[str, Any]) -> bool:
  return bool(
      result.get("schema_version") ==
          "intel-qwen36-exact-score-staging-component-v1"
      and result.get("algorithm") ==
          "generated_m256_n16_full_raw_f32_staging"
      and result.get("context_tokens") == 131072
      and result.get("head_dim") == 256
      and result.get("query_heads") == 16
      and result.get("kv_heads") == 2
      and result.get("gqa_group") == 8
      and result.get("kq_columns_per_kv") == 16
      and result.get("key_block") == 256
      and result.get("fused_useful_groups") == 2
      and result.get("staged_kq_useful_groups") == 1024
      and result.get("staged_owner_useful_groups") == 2
      and result.get("raw_score_bytes") == EXPECTED_RAW_SCORE_BYTES
      and result.get("raw_score_round_trip_bytes") ==
          EXPECTED_RAW_SCORE_ROUND_TRIP_BYTES
      and result.get("raw_score_compared_values") == 4194304
      and result.get("output_compared_values") == 4096
      and result.get("sample_count") == MIN_SAMPLES
      and result.get("schedule") ==
          "interleaved_fused_staged_staged_fused")


def distribution_pass(result: dict[str, Any]) -> bool:
  samples = result.get("paired_samples", [])
  if not isinstance(samples, list) or len(samples) != MIN_SAMPLES:
    return False
  for index, row in enumerate(samples):
    if row.get("order") != (
        "fused_staged" if index % 2 == 0 else "staged_fused"):
      return False
    fused = row.get("fused", {})
    staged = row.get("staged", {})
    if set(fused) != {"total_ms"}:
      return False
    if set(staged) != {"kq_ms", "owner_ms", "total_ms"}:
      return False
    values = [
        fused.get("total_ms"), staged.get("kq_ms"),
        staged.get("owner_ms"), staged.get("total_ms"),
        row.get("differential_ms"),
    ]
    try:
      numbers = [float(value) for value in values]
    except (TypeError, ValueError):
      return False
    if any(not math.isfinite(value) for value in numbers):
      return False
    if any(value <= 0.0 for value in numbers[:4]):
      return False
    if not math.isclose(
        numbers[4], numbers[3] - numbers[0],
        rel_tol=0.0, abs_tol=2.0e-6):
      return False
  return True


def summary(payload: dict[str, Any]) -> str:
  result = payload["result"]
  inference = payload["performance_inference"]
  return "\n".join([
      "# Exact full-score staging component gate",
      "",
      f"Verdict: **{payload['verdict']}**. Required checks: "
      f"`{str(payload['required_checks_passed']).lower()}`.",
      "",
      f"- raw F32 score mismatch / F16 output mismatch: "
      f"`{result.get('raw_score_mismatch_count')} / "
      f"{result.get('output_mismatch_count')}`",
      f"- staged-minus-fused median / one-sided 95% UCB / cap: "
      f"`{inference.get('point_estimate_ms')} / "
      f"{inference.get('upper_confidence_bound_ms')} / "
      f"{inference.get('cap_ms')} ms/layer`",
      f"- raw-score write+read / staged KQ groups: "
      f"`{result.get('raw_score_round_trip_bytes')} B / "
      f"{result.get('staged_kq_useful_groups')}`",
      f"- peak RSS / swaps: "
      f"`{payload['worker_resources'].get('maximum_resident_kib')} KiB / "
      f"{payload['worker_resources'].get('swaps')}`",
      "",
      "The route is rejected before graph, plugin, or model work unless both",
      "bitwise equality and the negative per-layer latency-delta cap pass.",
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

  required_paths = [
      SOURCE, RUNNER, CODEGEN, SHIMS, BOUNDARIES, SOURCE_BOUND,
      PINNED_SOURCE, PINNED_BUILD / "src/libdnnl.a", CXX, CMAKE, ENV_SCRIPT]
  missing = [str(path) for path in required_paths if not path.exists()]
  if missing:
    raise SystemExit("missing component inputs: " + ", ".join(missing))

  git = git_state(out_dir)
  provider_head = run(
      ["git", "-C", str(PINNED_SOURCE), "rev-parse", "HEAD"],
      30).stdout.strip()
  provider_scope_status = run([
      "git", "-C", str(PINNED_SOURCE), "status", "--short", "--",
      "src/gpu/intel/gemm/jit", "src/gpu/intel/jit/config",
      "third_party/ngen",
  ], 30).stdout.strip()
  source_bound = load_json(SOURCE_BOUND)
  source_text = SOURCE.read_text(encoding="utf-8")
  runner_text = RUNNER.read_text(encoding="utf-8")
  codegen_text = CODEGEN.read_text(encoding="utf-8")
  boundaries = load_json(BOUNDARIES)
  registered = [
      row for row in boundaries.get("infra_targets", [])
      if row.get("target") == TARGET
      and row.get("source") == "tools/exact_score_staging_component.cpp"]
  bound_performance = source_bound.get(
      "component_contract", {}).get("performance", {})
  bound_pass = bool(
      source_bound.get("schema_version") ==
          "intel-qwen36-openvino-exact-score-staging-bound-v2"
      and source_bound.get("verdict") ==
          "admit_one_exact_raw_f32_score_staging_component"
      and source_bound.get("required_checks_passed") is True
      and source_bound.get("component_admitted") is True
      and source_bound.get("graph_integration_admitted") is False
      and source_bound.get("model_worker_admitted") is False
      and source_bound.get("git", {}).get("dirty") is False
      and source_bound.get("geometry", {}).get(
          "kq_score_columns_per_kv") == 16
      and source_bound.get("traffic_bound", {}).get(
          "raw_score_roundtrip_bytes_per_layer") ==
          EXPECTED_RAW_SCORE_ROUND_TRIP_BYTES
      and math.isclose(
          float(bound_performance.get(
              "one_sided_95pct_ucb_cap_ms_per_layer", math.nan)),
          DELTA_CAP_MS, rel_tol=0.0, abs_tol=1.0e-9))
  source_checks = {
      "full_n16_raw_f32_score_staging":
          "#define IQ36_KQ_COLUMNS 16" in source_text
          and "for (uint column = 0U; column < IQ36_KQ_COLUMNS;" in source_text
          and "__global float* raw_score" in source_text,
      "three_isolated_program_variants":
          all(marker in source_text for marker in (
              "IQ36_COMPONENT_CAPTURE", "IQ36_COMPONENT_FUSED",
              "IQ36_COMPONENT_STAGED", "IQ36_COMPONENT_PROGRAM")),
      "generated_kq_and_vs_are_retained":
          "iq36_component_score_tile score = ugemm_kq(" in source_text
          and "iq36_component_accumulator_tile chunk_accumulator = ugemm_vs("
              in source_text,
      "chronological_owner_recurrence":
          "for (uint key_begin = 0U; key_begin < IQ36_CONTEXT;" in source_text
          and "old_running_max, running_max, iq36_component_rescale"
              in source_text,
      "twenty_interleaved_bitwise_runner":
          "constexpr int kSamples = 20;" in runner_text
          and "raw_score_mismatch_count" in runner_text
          and "output_mismatch_count" in runner_text
          and "interleaved_fused_staged_staged_fused" in runner_text,
      "generic_shim_fusion_is_isolated":
          "void FuseExistingShim()" in codegen_text
          and "--fuse-existing-shim" in codegen_text
          and "--host-define" in codegen_text
          and "const bool native_control" in codegen_text
          and "const bool fixed_vrt160" in codegen_text,
      "boundary_target_registered_once": len(registered) == 1,
  }
  sample_memory("after-source-audit", stop_bytes, memory)

  codegen_binary = raw_dir / "openvino-micro-codegen"
  codegen_build = run(
      codegen_build_command(codegen_binary), args.timeout_s)
  write_json(raw_dir / "codegen-build.json", {
      "command": codegen_build_command(codegen_binary),
      "returncode": codegen_build.returncode,
      "stdout": codegen_build.stdout,
      "stderr": codegen_build.stderr,
  })
  sample_memory("after-codegen-build", stop_bytes, memory)

  default_probe = (
      run([
          str(codegen_binary), "--provider-commit", PINNED_COMMIT,
      ], args.timeout_s)
      if codegen_build.returncode == 0 else
      subprocess.CompletedProcess([], 1, "", "codegen build failed"))
  write_json(raw_dir / "default-codegen-probe.json", {
      "command": list(default_probe.args),
      "returncode": default_probe.returncode,
      "stdout": default_probe.stdout,
      "stderr": default_probe.stderr,
  })
  default_result = parse_last_json(default_probe.stdout)

  codegen_results: dict[str, dict[str, Any]] = {}
  codegen_runs: dict[str, dict[str, Any]] = {}
  program_paths: dict[str, Path] = {}
  for label, specification in PROGRAMS.items():
    program_dir = raw_dir / "programs" / label
    program_dir.mkdir(parents=True, exist_ok=False)
    command = [
        str(codegen_binary),
        "--fuse-existing-shim", str(SHIMS),
        "--host-source", str(SOURCE),
        "--kernel-name", specification["kernel"],
        "--host-define", specification["define"],
        "--register-file-size", "128",
        "--provider-commit", PINNED_COMMIT,
        "--dump-dir", str(program_dir),
    ]
    shell_command = (
        f"source {shlex.quote(str(ENV_SCRIPT))} >/dev/null 2>&1 && " +
        " ".join(shlex.quote(part) for part in command))
    completed = (
        run(["bash", "-lc", shell_command], args.timeout_s)
        if codegen_build.returncode == 0 else
        subprocess.CompletedProcess(command, 1, "", "codegen build failed"))
    codegen_runs[label] = {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    codegen_results[label] = parse_last_json(completed.stdout)
    program_paths[label] = program_dir / "existing_shim.program.bin"
    sample_memory(f"after-{label}-program", stop_bytes, memory)
  write_json(raw_dir / "program-codegen.json", codegen_runs)

  configure_command = [
      str(CMAKE), "-S", str(ROOT / "engine"), "-B", str(BUILD_DIR),
      "-DCMAKE_BUILD_TYPE=Release"]
  configure = run(configure_command, 300)
  build_command = [
      str(CMAKE), "--build", str(BUILD_DIR), "--target", TARGET, "-j1"]
  build = run(build_command, args.timeout_s)
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
  build_ok = (
      configure.returncode == 0 and build.returncode == 0
      and executable.is_file())
  codegen_ok = bool(
      codegen_build.returncode == 0
      and provider_head == PINNED_COMMIT
      and provider_scope_status == ""
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
              PROGRAMS[label]["kernel"]
          and codegen_results[label].get("host_define") ==
              PROGRAMS[label]["define"]
          and codegen_results[label].get("register_file_size") == 128
          and codegen_results[label].get("program_bytes", 0) > 0
          for label in PROGRAMS))

  time_path = raw_dir / "component.time.txt"
  worker_command = [
      str(executable),
      str(program_paths["capture"]),
      str(program_paths["fused"]),
      str(program_paths["staged"]),
  ]
  shell_command = (
      f"source {shlex.quote(str(ENV_SCRIPT))} >/dev/null 2>&1 && "
      f"/usr/bin/time -v -o {shlex.quote(str(time_path))} " +
      " ".join(shlex.quote(part) for part in worker_command))
  sample_memory("before-component-worker", preflight_bytes, memory)
  worker = (
      run(["bash", "-lc", shell_command], args.timeout_s)
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
  write_json(raw_dir / "environment.json", environment())
  result = parse_last_json(worker.stdout)
  resources = parse_time(time_path)
  samples = [
      float(row.get("differential_ms", math.nan))
      for row in result.get("paired_samples", [])
      if isinstance(row, dict)]
  inference = signed_delta_cap_inference(samples, DELTA_CAP_MS)

  numeric_pass = bool(
      result.get("numeric_pass") is True
      and result.get("raw_score_mismatch_count") == 0
      and result.get("output_mismatch_count") == 0)
  checks = [
      check("repository_clean_at_gate", not git["dirty"], git=git),
      check("clean_v2_source_bound_admits_only_component", bound_pass),
      check("fixed_source_contract", all(source_checks.values()),
            source_checks=source_checks),
      check("pinned_codegen_builds_and_default_control_is_unchanged",
            codegen_ok, default_result=default_result,
            program_results=codegen_results,
            provider_head=provider_head,
            provider_scope_status=provider_scope_status),
      check("component_build_is_serial_j1", build_ok,
            build_command=build_command),
      check("standalone_worker_executes", worker.returncode == 0,
            returncode=worker.returncode),
      check("fixed_128k_full_n16_shape", fixed_result(result)),
      check("twenty_pair_interleaved_distribution",
            distribution_pass(result)),
      check("raw_score_and_output_are_bitwise_exact", numeric_pass),
      check("one_sided_95pct_delta_ucb_clears_negative_layer_cap",
            inference.get("rate_pass") is True,
            inference=inference),
      check("worker_rss_and_swap_are_bounded",
            int(resources.get("maximum_resident_kib", 1 << 62)) <
                4 * 1024 * 1024
            and int(resources.get("swaps", -1)) == 0,
            resources=resources),
      check("memory_guards_never_tripped",
            all(row["pass"] for row in memory),
            minimum_available_bytes=min(
                row["available_bytes"] for row in memory)),
  ]
  required = all(row["pass"] for row in checks)
  verdict = (
      "promote_exact_full_score_staging_component"
      if required else "reject_exact_full_score_staging_component")
  sources = [
      {"path": display(path), "sha256": sha256(path)}
      for path in (
          SOURCE, RUNNER, CODEGEN, SHIMS, BOUNDARIES, SOURCE_BOUND)]
  payload = {
      "schema_version":
          "intel-qwen36-openvino-exact-score-staging-component-gate-v1",
      "workstream": WS,
      "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
      "git": git,
      "verdict": verdict,
      "required_checks_passed": required,
      "component_promoted": required,
      "component_rejected": not required,
      "graph_integration_admitted": required,
      "plugin_build_admitted": False,
      "model_worker_admitted": False,
      "product_worker_admitted": False,
      "product_claim_allowed": False,
      "gpu_component_worker_launched": worker.returncode in {0, 2},
      "model_worker_launched": False,
      "checks": checks,
      "result": result,
      "performance_inference": inference,
      "delta_cap_ms_per_layer": DELTA_CAP_MS,
      "worker_resources": resources,
      "worker_command": worker_command,
      "codegen_results": codegen_results,
      "provider_head": provider_head,
      "provider_scope_status": provider_scope_status,
      "memory_preflight_bytes": preflight_bytes,
      "memory_stop_bytes": stop_bytes,
      "memory_samples": memory,
      "sources": sources,
  }
  write_json(out_dir / "result.json", payload)
  write_json(out_dir / "correctness.json", {
      "numeric_pass": numeric_pass,
      "raw_score_compared_values":
          result.get("raw_score_compared_values"),
      "raw_score_mismatch_count": result.get("raw_score_mismatch_count"),
      "output_compared_values": result.get("output_compared_values"),
      "output_mismatch_count": result.get("output_mismatch_count"),
  })
  write_json(out_dir / "performance.json", {
      "paired_samples": result.get("paired_samples", []),
      "inference": inference,
  })
  write_json(out_dir / "manifest.json", {
      "schema_version": "intel-qwen36-artifact-manifest-v1",
      "workstream": WS,
      "git_commit": git["commit"],
      "verdict": verdict,
      "sources": sources,
      "files": [
          "result.json", "correctness.json", "performance.json",
          "summary.md", "raw/codegen-build.json",
          "raw/default-codegen-probe.json", "raw/program-codegen.json",
          "raw/component-build.json", "raw/component.stdout",
          "raw/component.stderr", "raw/component.time.txt",
          "raw/worker-command.json", "raw/environment.json",
      ],
  })
  (out_dir / "summary.md").write_text(
      summary(payload), encoding="utf-8")
  print(json.dumps({
      "artifact": display(out_dir),
      "verdict": verdict,
      "numeric_pass": numeric_pass,
      "delta_median_ms": inference.get("point_estimate_ms"),
      "delta_ucb_ms": inference.get("upper_confidence_bound_ms"),
      "delta_cap_ms": DELTA_CAP_MS,
      "graph_integration_admitted": required,
      "model_worker_launched": False,
  }, separators=(",", ":")))
  return 0 if required else 2


if __name__ == "__main__":
  raise SystemExit(main())
