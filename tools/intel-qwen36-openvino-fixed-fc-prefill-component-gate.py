#!/usr/bin/env python3
"""Gate one true multi-token group-64 F16/U4 fixed-FC component."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import shlex
import signal
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from iq36_perf_inference import bootstrap_median_bound, latency_cap_inference


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-openvino-fixed-fc-prefill-component-gate-v0"
ENV_SCRIPT = Path(
    "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh")
CMAKE = Path("/home/intel/intel-box-env/conda/bin/cmake")
CXX = Path("/home/intel/intel-box-env/conda/bin/c++")
OV_PYTHON = Path("/home/intel/ov/openvino_env/bin/python")
MODEL_XML = Path(
    "/home/intel/Qwen3.6-35B-A3B-ov/openvino_language_model.xml")
OV_SOURCE = Path("/home/intel/intel-qwen36-r0/source/openvino-90214e5be05")
ONEDNN_SOURCE = OV_SOURCE / "src/plugins/intel_gpu/thirdparty/onednn_gpu"
ONEDNN_BUILD = Path(
    "/home/intel/intel-qwen36-r0/build/onednn-20db-micro-static")
BUILD_DIR = ROOT / "build/engine"
CODEGEN_SOURCE = ROOT / "engine/tools/openvino_moe_micro_codegen.cpp"
SINGLE_HOST = ROOT / "engine/gpu/opencl/openvino_fc_micro_host.cl"
MULTI_HOST = ROOT / "engine/gpu/opencl/openvino_fc_multi_output_host.cl"
RUNTIME_SOURCE = ROOT / "engine/tools/openvino_fc_multi_output_runtime.cpp"
GRAPH_MODULE = ROOT / "tools/intel_qwen36_openvino_fixed_fc.py"
SEQ1354 = ROOT / (
    "output/openvino-fixed-fc-multi-output-20260718Tseq1354-cleanZ/metrics.json")

COHORTS = (
    {"name": "linear_input", "model_name": "linear_attention_input",
     "layer": 0, "widths": (8192, 32, 32, 4096), "k": 2048, "count": 30,
     "stock_dq": 4, "stock_fc": 4},
    {"name": "full_qkv", "model_name": "full_attention_qkv",
     "layer": 3, "widths": (8192, 512, 512), "k": 2048, "count": 10,
     "stock_dq": 1, "stock_fc": 1},
    {"name": "router_shared", "model_name": "router_shared_input",
     "layer": 0, "widths": (1, 512, 512, 256), "k": 2048, "count": 40,
     "stock_dq": 4, "stock_fc": 4},
    {"name": "attention_output", "model_name": "attention_output",
     "layer": 0, "widths": (2048,), "k": 4096, "count": 40,
     "stock_dq": 1, "stock_fc": 1},
    {"name": "shared_expert_down", "model_name": "shared_expert_down",
     "layer": 0, "widths": (2048,), "k": 512, "count": 40,
     "stock_dq": 1, "stock_fc": 1},
)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out-dir", type=Path)
  parser.add_argument("--tokens", type=int, default=2048)
  parser.add_argument("--numeric-tokens", type=int, default=32)
  parser.add_argument("--warmup", type=int, default=64)
  parser.add_argument("--samples", type=int, default=15)
  parser.add_argument("--repeat", type=int, default=7)
  parser.add_argument("--blocks", type=int, default=8)
  parser.add_argument("--timeout-s", type=int, default=900)
  parser.add_argument("--memory-preflight-gib", type=float, default=8.0)
  parser.add_argument("--memory-stop-gib", type=float, default=4.0)
  parser.add_argument("--worker-config", type=Path)
  args = parser.parse_args()
  if (args.tokens < 2 or args.numeric_tokens < 2 or args.warmup < 1 or
      args.samples < 5 or args.repeat < 5 or args.blocks < 8 or
      args.timeout_s <= 0 or args.memory_preflight_gib <= 0.0 or
      args.memory_stop_gib <= 0.0 or
      args.memory_preflight_gib < args.memory_stop_gib):
    parser.error("token, timing, timeout, or memory arguments are invalid")
  if args.out_dir is None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.out_dir = ROOT / f"output/openvino-fixed-fc-prefill-{stamp}"
  return args


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise TypeError(f"expected JSON object: {path}")
  return value


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def run(command: list[str], timeout_s: int,
        cwd: Path = ROOT) -> dict[str, Any]:
  try:
    result = subprocess.run(
        command, cwd=cwd, text=True, capture_output=True,
        timeout=timeout_s, check=False, encoding="utf-8", errors="replace")
    return {"command": command, "returncode": result.returncode,
            "stdout": result.stdout, "stderr": result.stderr,
            "timed_out": False}
  except subprocess.TimeoutExpired as error:
    return {"command": command, "returncode": 124,
            "stdout": error.stdout or "", "stderr": error.stderr or "",
            "timed_out": True}


def run_intel(command: list[str], timeout_s: int) -> dict[str, Any]:
  shell = (
      f"source {shlex.quote(str(ENV_SCRIPT))} >/dev/null 2>&1 && "
      "export INTEL_FORCE_PROBE=b080 && export DNNL_VERBOSE=0 && " +
      shlex.join(command))
  return run(["bash", "-lc", shell], timeout_s)


def write_run(raw: Path, label: str, row: dict[str, Any]) -> None:
  write_json(raw / f"{label}.command.json", row.get("command", []))
  (raw / f"{label}.stdout").write_text(
      str(row.get("stdout", "")), encoding="utf-8")
  (raw / f"{label}.stderr").write_text(
      str(row.get("stderr", "")), encoding="utf-8")


def parse_json_line(row: dict[str, Any]) -> dict[str, Any]:
  for line in reversed(str(row.get("stdout", "")).splitlines()):
    try:
      value = json.loads(line)
    except json.JSONDecodeError:
      continue
    if isinstance(value, dict):
      return value
  return {}


def git_output(*parts: str, cwd: Path = ROOT) -> str:
  result = subprocess.run(
      ["git", *parts], cwd=cwd, text=True, capture_output=True, check=True)
  return result.stdout.strip()


def meminfo() -> dict[str, int]:
  values: dict[str, int] = {}
  for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
    key, rest = line.split(":", 1)
    if key in {"MemAvailable", "SwapTotal", "SwapFree"}:
      values[key] = int(rest.split()[0]) * 1024
  return values


def process_group_memory(pgrp: int) -> dict[str, int]:
  rss = swap = processes = 0
  for stat_path in Path("/proc").glob("[0-9]*/stat"):
    try:
      stat = stat_path.read_text(encoding="utf-8")
      tail = stat[stat.rfind(")") + 2:].split()
      if len(tail) < 3 or int(tail[2]) != pgrp:
        continue
      status = stat_path.with_name("status").read_text(
          encoding="utf-8", errors="replace").splitlines()
    except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
      continue
    processes += 1
    for line in status:
      if line.startswith("VmRSS:"):
        rss += int(line.split()[1]) * 1024
      elif line.startswith("VmSwap:"):
        swap += int(line.split()[1]) * 1024
  return {"rss_bytes": rss, "swap_bytes": swap, "processes": processes}


def stop_group(process: subprocess.Popen[Any]) -> None:
  try:
    os.killpg(process.pid, signal.SIGTERM)
  except ProcessLookupError:
    return
  try:
    process.wait(timeout=10.0)
  except subprocess.TimeoutExpired:
    try:
      os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
      pass


def monitor_worker(command: list[str], config: dict[str, Any], raw: Path,
                   timeout_s: int, stop_bytes: int) -> dict[str, Any]:
  config_path = raw / "stock-worker-config.json"
  write_json(config_path, config)
  stdout_path = raw / "stock-worker.stdout"
  stderr_path = raw / "stock-worker.stderr"
  environment = os.environ.copy()
  environment.pop("OV_GPU_CONFIG_FILE", None)
  environment.pop("OPENVINO_GPU_CONFIG_FILE", None)
  start = meminfo()
  monitor = {
      "system_available_min_bytes": start["MemAvailable"],
      "process_group_rss_peak_bytes": 0,
      "process_group_swap_peak_bytes": 0,
      "process_count_peak": 0,
      "samples": 0,
  }
  started = time.monotonic()
  timed_out = guard_tripped = False
  with stdout_path.open("w", encoding="utf-8") as stdout, \
       stderr_path.open("w", encoding="utf-8") as stderr:
    process = subprocess.Popen(
        command, cwd=ROOT, env=environment, stdout=stdout, stderr=stderr,
        text=True, start_new_session=True)
    while process.poll() is None:
      if time.monotonic() - started > timeout_s:
        timed_out = True
        stop_group(process)
        break
      system = meminfo()
      group = process_group_memory(process.pid)
      monitor["samples"] += 1
      monitor["system_available_min_bytes"] = min(
          monitor["system_available_min_bytes"], system["MemAvailable"])
      monitor["process_group_rss_peak_bytes"] = max(
          monitor["process_group_rss_peak_bytes"], group["rss_bytes"])
      monitor["process_group_swap_peak_bytes"] = max(
          monitor["process_group_swap_peak_bytes"], group["swap_bytes"])
      monitor["process_count_peak"] = max(
          monitor["process_count_peak"], group["processes"])
      if system["MemAvailable"] < stop_bytes:
        guard_tripped = True
        stop_group(process)
        break
      time.sleep(0.05)
    returncode = process.wait()
  stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")
  return {
      "command": command, "returncode": returncode,
      "elapsed_seconds": time.monotonic() - started,
      "timed_out": timed_out, "memory_guard_tripped": guard_tripped,
      "oom_observed": (
          not guard_tripped and
          (returncode in (-9, 137) or "out of memory" in stderr_text.lower())),
      "monitor": monitor,
  }


def load_module(path: Path, name: str) -> Any:
  spec = importlib.util.spec_from_file_location(name, path)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load module from {path}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def pack_u4(values: Any, np: Any) -> Any:
  flat = np.asarray(values, dtype=np.uint8).reshape(-1)
  if flat.size % 2:
    raise ValueError("U4 stream is not byte aligned")
  return ((flat[0::2] & np.uint8(15)) |
          ((flat[1::2] & np.uint8(15)) << np.uint8(4)))


def worker_main(config_path: Path) -> int:
  config = load_json(config_path)
  import numpy as np
  import openvino as ov

  graph = load_module(GRAPH_MODULE, "iq36_fixed_fc_prefill_graph")
  core = ov.Core()
  source = core.read_model(str(MODEL_XML))
  groups = graph.discover_fixed_fc_groups(source, ov)
  data_root = Path(config["data_root"])
  stop_bytes = int(config["stop_bytes"])
  rows = []
  for cohort in COHORTS:
    if meminfo()["MemAvailable"] < stop_bytes:
      raise RuntimeError(f"memory stop before stock {cohort['name']}")
    group = next(
        row for row in groups
        if row["layer"] == cohort["layer"] and
        row["cohort"] == cohort["model_name"])
    if tuple(group["widths"]) != cohort["widths"] or group["k"] != cohort["k"]:
      raise RuntimeError(f"locked cohort shape differs: {cohort['name']}")
    directory = data_root / cohort["name"]
    directory.mkdir(parents=True, exist_ok=False)
    k = int(group["k"])
    tokens = int(config["tokens"])
    rng = np.random.default_rng(0x136100 + int(cohort["layer"]) + k)
    activation = rng.uniform(-0.25, 0.25, (1, tokens, k)).astype(np.float32)
    activation.astype(np.float16).reshape(tokens, k).tofile(
        directory / "input.f16")
    parameter = ov.opset13.parameter(
        ov.PartialShape([1, -1, k]), ov.Type.f32, name="activation")
    outputs = []
    weights = []
    scales = []
    logical_zps = []
    for index, projection in enumerate(group["projections"]):
      matmul = ov.opset13.matmul(
          parameter.output(0), projection["matmul"].input_value(1),
          False, True)
      matmul.set_friendly_name(
          f"iq36_stock_{cohort['name']}_{index}_m{projection['m']}")
      outputs.append(matmul.output(0))
      m = int(projection["m"])
      groups_k = int(projection["groups"])
      weight = np.asarray(
          projection["weight"].get_data(), dtype=np.uint8).reshape(-1).copy()
      scale = np.asarray(
          projection["scale"].get_data(), dtype=np.float16
      ).reshape(m, groups_k, 1).transpose(1, 0, 2).copy()
      packed_zp = np.asarray(
          projection["zero_point"].get_data(), dtype=np.uint8).reshape(-1)
      logical = np.empty(packed_zp.size * 2, dtype=np.uint8)
      logical[0::2] = packed_zp & np.uint8(15)
      logical[1::2] = packed_zp >> np.uint8(4)
      zp = logical[:m * groups_k].reshape(
          m, groups_k, 1).transpose(1, 0, 2).copy().reshape(groups_k, m)
      weight.tofile(directory / f"weight{index}.u4")
      scale.reshape(-1).tofile(directory / f"scale{index}.f16")
      pack_u4(zp, np).tofile(directory / f"zp{index}.u4")
      weights.append(weight)
      scales.append(scale)
      logical_zps.append(zp)
    np.concatenate(weights).tofile(directory / "fused-weights.u4")
    np.concatenate(scales, axis=1).reshape(-1).tofile(
        directory / "fused-scales.f16")
    pack_u4(np.concatenate(logical_zps, axis=1), np).tofile(
        directory / "fused-zps.u4")

    model = ov.Model(outputs, [parameter], f"iq36_stock_{cohort['name']}")
    compile_started = time.perf_counter_ns()
    compiled = core.compile_model(
        model, "GPU", {"PERFORMANCE_HINT": "LATENCY", "PERF_COUNT": True})
    compile_ms = (time.perf_counter_ns() - compile_started) / 1_000_000.0
    request = compiled.create_infer_request()
    feed = {compiled.input("activation"): activation}
    for _ in range(int(config["warmup"])):
      request.infer(feed, share_outputs=True)
    schedule_samples = []
    profile_rows = []
    for _ in range(int(config["samples"])):
      request.infer(feed, share_outputs=True)
      current = []
      for profile in request.get_profiling_info():
        if profile.node_type not in {
            "DynamicQuantize", "FullyConnectedCompressed"}:
          continue
        current.append({
            "name": profile.node_name, "type": profile.node_type,
            "exec_type": profile.exec_type,
            "real_time_us": profile.real_time.total_seconds() * 1_000_000.0,
        })
      schedule_samples.append(sum(row["real_time_us"] for row in current))
      profile_rows = current
    rows.append({
        **cohort, "compile_ms": compile_ms,
        "schedule_samples_us": schedule_samples,
        "schedule_median_us": statistics.median(schedule_samples),
        "schedule_min_us": min(schedule_samples),
        "runtime_dq_count": sum(
            row["type"] == "DynamicQuantize" for row in profile_rows),
        "runtime_fc_count": sum(
            row["type"] == "FullyConnectedCompressed" for row in profile_rows),
        "profile": profile_rows,
    })
    del request, compiled, model, activation, outputs
    gc.collect()
  write_json(Path(config["result"]), {
      "openvino_version": ov.get_version(), "rows": rows})
  return 0


def codegen_build_command(binary: Path) -> list[str]:
  includes = [
      ONEDNN_SOURCE / "src/gpu/intel/gemm/jit",
      ONEDNN_SOURCE / "src/gpu/intel/gemm/jit/dnnl_gpu_intel_gemm_jit",
      ONEDNN_BUILD / "include", ONEDNN_SOURCE / "include",
      ONEDNN_SOURCE / "third_party/opencl", ONEDNN_SOURCE / "third_party",
      ONEDNN_SOURCE / "src", ONEDNN_SOURCE / "src/gpu/intel/jit/config",
      ONEDNN_SOURCE / "third_party/ngen",
      ONEDNN_SOURCE / "src/gpu/intel/gemm/jit/include",
  ]
  return [
      str(CXX), "-std=c++17", "-O3", "-DNDEBUG", "-fopenmp",
      "-fno-operator-names", "-DCL_TARGET_OPENCL_VERSION=120",
      "-DDNNL_X64=1", "-DGEMMSTONE_BUILD_12HP", "-DGEMMSTONE_BUILD_12LP",
      "-DGEMMSTONE_BUILD_12P7", "-DGEMMSTONE_BUILD_12P8",
      "-DGEMMSTONE_BUILD_XE2", "-DGEMMSTONE_BUILD_XE3",
      "-DGEMMSTONE_BUILD_XE3P", "-DGEMMSTONE_CONFIG", "-DNGEN_CONFIG",
      *[f"-I{path}" for path in includes], str(CODEGEN_SOURCE),
      str(ONEDNN_BUILD / "src/libdnnl.a"), "-lOpenCL", "-ldl", "-lpthread",
      "-o", str(binary),
  ]


def create_router_numeric(directory: Path, widths: tuple[int, ...], k: int,
                          tokens: int) -> None:
  import numpy as np

  rng = np.random.default_rng(0x1361F16)
  groups = k // 64
  activation = rng.uniform(-0.25, 0.25, (tokens, k)).astype(np.float16)
  activation.tofile(directory / "input.f16")
  weights = []
  scales = []
  logical_zps = []
  references = []
  for index, width in enumerate(widths):
    packed = rng.integers(0, 256, (width, k // 2), dtype=np.uint8)
    logical_weight = np.empty((width, k), dtype=np.uint8)
    logical_weight[:, 0::2] = packed & np.uint8(15)
    logical_weight[:, 1::2] = packed >> np.uint8(4)
    scale = rng.uniform(0.0005, 0.003, (groups, width)).astype(np.float16)
    zp = rng.integers(0, 16, (groups, width), dtype=np.uint8)
    dequant = (
        logical_weight.reshape(width, groups, 64).astype(np.float32) -
        zp.T[:, :, None].astype(np.float32))
    dequant *= scale.T[:, :, None].astype(np.float32)
    reference = (
        activation.astype(np.float32) @ dequant.reshape(width, k).T
    ).astype(np.float16)
    packed.tofile(directory / f"weight{index}.u4")
    scale.tofile(directory / f"scale{index}.f16")
    pack_u4(zp, np).tofile(directory / f"zp{index}.u4")
    weights.append(packed.reshape(-1))
    scales.append(scale.reshape(groups, width, 1))
    logical_zps.append(zp)
    references.append(reference)
  np.concatenate(weights).tofile(directory / "fused-weights.u4")
  np.concatenate(scales, axis=1).reshape(-1).tofile(
      directory / "fused-scales.f16")
  pack_u4(np.concatenate(logical_zps, axis=1), np).tofile(
      directory / "fused-zps.u4")
  np.concatenate([value.reshape(-1) for value in references]).tofile(
      directory / "reference.f16")


def check(name: str, passed: bool, **evidence: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **evidence}


def main() -> int:
  args = parse_args()
  if args.worker_config is not None:
    return worker_main(args.worker_config)
  out = args.out_dir.resolve()
  raw = out / "raw"
  raw.mkdir(parents=True, exist_ok=False)
  preflight_bytes = int(args.memory_preflight_gib * 1024**3)
  stop_bytes = int(args.memory_stop_gib * 1024**3)
  start_memory = meminfo()
  if start_memory["MemAvailable"] < preflight_bytes:
    raise RuntimeError("eight-GiB preflight did not clear")
  required = [
      ENV_SCRIPT, CMAKE, CXX, OV_PYTHON, MODEL_XML, CODEGEN_SOURCE,
      SINGLE_HOST, MULTI_HOST, RUNTIME_SOURCE, GRAPH_MODULE, SEQ1354,
      ONEDNN_BUILD / "src/libdnnl.a",
  ]
  missing = [str(path) for path in required if not path.exists()]
  if missing:
    raise SystemExit("missing prefill component inputs: " + ", ".join(missing))
  seq1354 = load_json(SEQ1354)

  created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
  commit = git_output("rev-parse", "HEAD")
  dirty = git_output("status", "--porcelain")
  onednn_commit = git_output("rev-parse", "HEAD", cwd=ONEDNN_SOURCE)
  onednn_dirty = git_output("status", "--porcelain", cwd=ONEDNN_SOURCE)

  data_root = raw / "data"
  data_root.mkdir()
  stock_result_path = raw / "stock-result.json"
  stock_command = [
      str(OV_PYTHON), str(Path(__file__).resolve()),
      "--worker-config", str(raw / "stock-worker-config.json")]
  stock_worker = monitor_worker(
      stock_command,
      {"tokens": args.tokens, "warmup": args.warmup,
       "samples": args.samples, "stop_bytes": stop_bytes,
       "data_root": str(data_root), "result": str(stock_result_path)},
      raw, args.timeout_s, stop_bytes)
  stock = load_json(stock_result_path) if stock_result_path.exists() else {}

  configure = run([
      str(CMAKE), "-S", str(ROOT / "engine"), "-B", str(BUILD_DIR)],
      args.timeout_s)
  write_run(raw, "configure", configure)
  runtime_build = run([
      str(CMAKE), "--build", str(BUILD_DIR), "--target",
      "iq36-openvino-fc-multi-output-runtime", "--", "-j1"], args.timeout_s)
  write_run(raw, "runtime-build", runtime_build)
  runtime = BUILD_DIR / "iq36-openvino-fc-multi-output-runtime"
  codegen = raw / "openvino-fc-prefill-codegen"
  codegen_build = run(codegen_build_command(codegen), args.timeout_s)
  write_run(raw, "codegen-build", codegen_build)

  candidate_rows = []
  stock_by_name = {row["name"]: row for row in stock.get("rows", [])}
  memory_samples = [{"label": "start", **start_memory}]
  prerequisites = (
      stock_worker["returncode"] == 0 and configure["returncode"] == 0 and
      runtime_build["returncode"] == 0 and codegen_build["returncode"] == 0)
  for cohort in COHORTS:
    current = meminfo()
    memory_samples.append({"label": f"before-{cohort['name']}", **current})
    if current["MemAvailable"] < stop_bytes:
      raise RuntimeError(f"memory stop before candidate {cohort['name']}")
    package_root = raw / "packages" / cohort["name"]
    baseline_dir = package_root / "baseline"
    candidate_dir = package_root / "candidate"
    total_m = sum(cohort["widths"])
    common = [
        str(codegen), "--prefill-fc", "--shape-name", cohort["name"],
        "--m", str(total_m), "--n", str(args.tokens),
        "--k", str(cohort["k"])]
    baseline_command = [
        *common, "--dump-dir", str(baseline_dir),
        "--host-source", str(SINGLE_HOST)]
    candidate_command = [
        *common, "--projection-widths",
        ",".join(map(str, cohort["widths"])),
        "--dump-dir", str(candidate_dir), "--host-source", str(MULTI_HOST)]
    baseline_run = run_intel(baseline_command, args.timeout_s) if prerequisites else {
        "command": baseline_command, "returncode": 125, "stdout": "",
        "stderr": "prerequisite failed", "timed_out": False}
    write_run(raw, f"{cohort['name']}-baseline-codegen", baseline_run)
    candidate_run = (
        run_intel(candidate_command, args.timeout_s)
        if baseline_run["returncode"] == 0 else {
            "command": candidate_command, "returncode": 125, "stdout": "",
            "stderr": "baseline codegen failed", "timed_out": False})
    write_run(raw, f"{cohort['name']}-candidate-codegen", candidate_run)
    baseline_codegen = parse_json_line(baseline_run)
    candidate_codegen = parse_json_line(candidate_run)
    package = (candidate_codegen.get("packages") or [{}])[0]
    settings = package.get("settings", {})
    directory = data_root / cohort["name"]
    arity = len(cohort["widths"])
    runtime_command = [
        str(runtime), "--baseline-binary",
        str(baseline_dir / f"{cohort['name']}.program.bin"),
        "--candidate-binary",
        str(candidate_dir / f"{cohort['name']}.program.bin"),
        "--kernel", f"iq36_moe_micro_{cohort['name']}",
        "--input", str(directory / "input.f16"),
        "--baseline-weights", str(directory / "fused-weights.u4"),
        "--baseline-scales", str(directory / "fused-scales.f16"),
        "--baseline-zps", str(directory / "fused-zps.u4"),
        "--weights", ",".join(
            str(directory / f"weight{index}.u4") for index in range(arity)),
        "--scales", ",".join(
            str(directory / f"scale{index}.f16") for index in range(arity)),
        "--zps", ",".join(
            str(directory / f"zp{index}.u4") for index in range(arity)),
        "--widths", ",".join(map(str, cohort["widths"])),
        "--k", str(cohort["k"]), "--n", str(args.tokens),
        "--quant-group", "64",
        "--sg-per-wg-m", str(settings.get("sg_per_wg_m", 0)),
        "--sg-per-wg-n", str(settings.get("sg_per_wg_n", 0)),
        "--sg-per-wg-k", str(settings.get("sg_per_wg_k", 0)),
        "--wg-tile-m", str(settings.get("wg_tile_m", 0)),
        "--wg-tile-n", str(settings.get("wg_tile_n", 0)),
        "--warmup", str(args.warmup), "--repeat", str(args.repeat),
        "--blocks", str(args.blocks)]
    if cohort["name"] == "router_shared":
      runtime_command.append("--allow-baseline-difference")
    runtime_run = (
        run_intel(runtime_command, args.timeout_s)
        if candidate_run["returncode"] == 0 else {
            "command": runtime_command, "returncode": 125, "stdout": "",
            "stderr": "candidate codegen failed", "timed_out": False})
    write_run(raw, f"{cohort['name']}-runtime", runtime_run)
    result = parse_json_line(runtime_run)
    stock_row = stock_by_name.get(cohort["name"], {})
    candidate_rows.append({
        **cohort, "baseline_codegen": baseline_codegen,
        "candidate_codegen": candidate_codegen,
        "baseline_codegen_returncode": baseline_run["returncode"],
        "candidate_codegen_returncode": candidate_run["returncode"],
        "runtime_returncode": runtime_run["returncode"],
        "runtime": result, "stock": stock_row,
        "hashes": {
            "baseline_micro": sha256(
                baseline_dir / f"{cohort['name']}.micro.bin")
            if (baseline_dir / f"{cohort['name']}.micro.bin").exists() else None,
            "candidate_micro": sha256(
                candidate_dir / f"{cohort['name']}.micro.bin")
            if (candidate_dir / f"{cohort['name']}.micro.bin").exists() else None,
        },
    })

  router = next(row for row in candidate_rows if row["name"] == "router_shared")
  numeric_dir = raw / "router-numeric"
  numeric_dir.mkdir()
  create_router_numeric(
      numeric_dir, tuple(router["widths"]), int(router["k"]),
      args.numeric_tokens)
  router_package = (router["candidate_codegen"].get("packages") or [{}])[0]
  router_settings = router_package.get("settings", {})
  router_packages = raw / "packages" / "router_shared"
  router_numeric_command = [
      str(runtime), "--baseline-binary",
      str(router_packages / "baseline/router_shared.program.bin"),
      "--candidate-binary",
      str(router_packages / "candidate/router_shared.program.bin"),
      "--kernel", "iq36_moe_micro_router_shared",
      "--input", str(numeric_dir / "input.f16"),
      "--baseline-weights", str(numeric_dir / "fused-weights.u4"),
      "--baseline-scales", str(numeric_dir / "fused-scales.f16"),
      "--baseline-zps", str(numeric_dir / "fused-zps.u4"),
      "--weights", ",".join(str(numeric_dir / f"weight{i}.u4") for i in range(4)),
      "--scales", ",".join(str(numeric_dir / f"scale{i}.f16") for i in range(4)),
      "--zps", ",".join(str(numeric_dir / f"zp{i}.u4") for i in range(4)),
      "--widths", "1,512,512,256", "--k", "2048", "--n",
      str(args.numeric_tokens), "--quant-group", "64",
      "--sg-per-wg-m", str(router_settings.get("sg_per_wg_m", 0)),
      "--sg-per-wg-n", str(router_settings.get("sg_per_wg_n", 0)),
      "--sg-per-wg-k", str(router_settings.get("sg_per_wg_k", 0)),
      "--wg-tile-m", str(router_settings.get("wg_tile_m", 0)),
      "--wg-tile-n", str(router_settings.get("wg_tile_n", 0)),
      "--reference", str(numeric_dir / "reference.f16"),
      "--allow-baseline-difference", "--warmup", "8", "--repeat", "5",
      "--blocks", "8"]
  router_numeric_run = run_intel(
      router_numeric_command, args.timeout_s) if prerequisites else {
          "command": router_numeric_command, "returncode": 125, "stdout": "",
          "stderr": "prerequisite failed", "timed_out": False}
  write_run(raw, "router-numeric-runtime", router_numeric_run)
  router_numeric = parse_json_line(router_numeric_run)

  candidate_schedule_samples = []
  if all(len(row["runtime"].get("blocks", [])) == args.blocks
         for row in candidate_rows):
    for index in range(args.blocks):
      candidate_schedule_samples.append(sum(
          row["count"] * float(
              row["runtime"]["blocks"][index]["candidate_us"])
          for row in candidate_rows))
  stock_schedule_samples = []
  if all(len(row["stock"].get("schedule_samples_us", [])) == args.samples
         for row in candidate_rows):
    for index in range(args.samples):
      stock_schedule_samples.append(sum(
          row["count"] * float(row["stock"]["schedule_samples_us"][index])
          for row in candidate_rows))
  candidate_median = (
      statistics.median(candidate_schedule_samples)
      if candidate_schedule_samples else math.inf)
  stock_median = (
      statistics.median(stock_schedule_samples)
      if stock_schedule_samples else math.inf)
  complete_ratio = candidate_median / stock_median
  candidate_schedule_ms = [value / 1000.0
                           for value in candidate_schedule_samples]
  stock_schedule_ms = [value / 1000.0 for value in stock_schedule_samples]
  stock_lcb_ms = (
      bootstrap_median_bound(stock_schedule_ms, side="lower")
      if stock_schedule_ms else None)
  performance_inference = (
      latency_cap_inference(
          candidate_schedule_ms, cap=stock_lcb_ms, min_samples=8)
      if candidate_schedule_ms and stock_lcb_ms is not None else {})
  router_reference = router_numeric.get("candidate_reference_compare", {})
  package_rows = [
      (row["candidate_codegen"].get("packages") or [{}])[0]
      for row in candidate_rows]
  checks = [
      check("repository_clean_at_gate", dirty == "", dirty_paths=dirty.splitlines()),
      check("pinned_onednn_source_clean", onednn_dirty == "",
            commit=onednn_commit, dirty_paths=onednn_dirty.splitlines()),
      check("retained_decode_component_anchor_is_clean_and_admitted",
            seq1354.get("required_checks_passed") is True and
            seq1354.get("component_admission_passed") is True and
            seq1354.get("verdict") ==
            "admit_isolated_fixed_fc_graph_integration_source_cut",
            artifact=str(SEQ1354.relative_to(ROOT))),
      check("stock_worker_completed_serially_without_oom",
            stock_worker["returncode"] == 0 and
            not stock_worker["timed_out"] and
            not stock_worker["memory_guard_tripped"] and
            not stock_worker["oom_observed"], worker=stock_worker),
      check("serial_build_and_codegen_passed",
            configure["returncode"] == 0 and runtime_build["returncode"] == 0 and
            codegen_build["returncode"] == 0 and all(
                row["baseline_codegen_returncode"] == 0 and
                row["candidate_codegen_returncode"] == 0
                for row in candidate_rows)),
      check("all_five_locked_cohorts_executed",
            [row["name"] for row in candidate_rows] ==
            [row["name"] for row in COHORTS] and all(
                row["runtime_returncode"] == 0 for row in candidate_rows)),
      check("all_packages_are_group64_systolic_and_spill_free",
            all(package.get("quant_group_size") == 64 and
                package.get("systolic") is True and
                int(package.get("grf_min", 999)) <= 256 and
                row["runtime"].get("candidate_spill_memory_bytes") == 0
                for package, row in zip(package_rows, candidate_rows)),
            packages=package_rows),
      check("even_m_multitoken_outputs_are_bit_exact",
            all(row["runtime"].get("baseline_candidate_compare", {}).get(
                    "exact_rate") == 1.0
                for row in candidate_rows if row["name"] != "router_shared")),
      check("router_scalar_and_shared_multitoken_outputs_are_tight",
            router_numeric_run["returncode"] == 0 and
            router_reference.get("finite") is True and
            float(router_reference.get("cosine", 0.0)) >= 0.999 and
            float(router_reference.get("relative_l2", math.inf)) <= 0.002 and
            float(router_reference.get("max_abs_diff", math.inf)) <= 0.00025,
            compare=router_reference),
      check("stock_execution_census_is_exact",
            all(row["stock"].get("runtime_dq_count") == row["stock_dq"] and
                row["stock"].get("runtime_fc_count") == row["stock_fc"]
                for row in candidate_rows)),
      check("complete_schedule_inference_has_required_samples",
            performance_inference.get("sample_count_pass") is True and
            len(stock_schedule_samples) >= 8,
            candidate_samples_us=candidate_schedule_samples,
            stock_samples_us=stock_schedule_samples),
      check("memory_preflight_and_stop_guards_never_tripped",
            start_memory["MemAvailable"] >= preflight_bytes and
            stock_worker["monitor"]["system_available_min_bytes"] >= stop_bytes and
            all(row["MemAvailable"] >= stop_bytes for row in memory_samples),
            stock_monitor=stock_worker["monitor"],
            minimum_sampled_available_bytes=min(
                row["MemAvailable"] for row in memory_samples)),
  ]
  required_checks_passed = all(row["pass"] for row in checks)
  component_admission_passed = (
      required_checks_passed and
      performance_inference.get("rate_pass") is True)
  verdict = (
      "admit_phase_specialized_fixed_fc_prefill_provider_source_cut"
      if component_admission_passed else
      "close_current_group64_f16_u4_prefill_package_on_complete_schedule"
      if required_checks_passed and complete_ratio >= 1.0 else
      "inconclusive_fixed_fc_prefill_complete_schedule_confidence"
      if required_checks_passed else "inconclusive")
  next_action = (
      "Implement one bounded shape-specialized provider selection between "
      "the admitted N=1 and N>1 packages before any model worker."
      if component_admission_passed else
      "Keep the N=1 package decode-only and switch route; do not sweep "
      "neighboring prefill tiles."
      if required_checks_passed and complete_ratio >= 1.0 else
      "Profile the complete schedule or collect more confidence samples; "
      "do not sweep neighboring prefill tiles."
      if required_checks_passed else
      "Repair the incomplete evidence gate before making a route decision.")
  result = {
      "schema": SCHEMA, "created_at": created_at,
      "workstream": WORKSTREAM, "commit": commit,
      "required_checks_passed": required_checks_passed,
      "component_admission_passed": component_admission_passed,
      "verdict": verdict,
      "tokens": args.tokens, "numeric_tokens": args.numeric_tokens,
      "source": {"retained_decode_component": str(SEQ1354.relative_to(ROOT)),
                 "onednn_commit": onednn_commit},
      "stock_worker": stock_worker, "stock": stock,
      "candidate_rows": candidate_rows, "router_numeric": router_numeric,
      "complete_schedule": {
          "candidate_samples_us": candidate_schedule_samples,
          "stock_samples_us": stock_schedule_samples,
          "candidate_median_us": candidate_median,
          "stock_median_us": stock_median,
          "candidate_stock_ratio": complete_ratio,
          "candidate_ms_per_token": candidate_median / args.tokens / 1000.0,
          "stock_ms_per_token": stock_median / args.tokens / 1000.0,
          "stock_median_lower_confidence_bound_ms": stock_lcb_ms,
          "candidate_latency_inference": performance_inference,
      },
      "memory_samples": memory_samples, "checks": checks,
      "next_action": next_action,
  }
  write_json(out / "metrics.json", result)
  write_json(out / "manifest.json", {
      "schema": SCHEMA, "workstream": WORKSTREAM,
      "required_checks_passed": required_checks_passed,
      "component_admission_passed": component_admission_passed,
      "metrics": "metrics.json",
      "raw": "raw/"})
  failed = [row["name"] for row in checks if not row["pass"]]
  (out / "summary.md").write_text("\n".join([
      "# Fixed-FC multi-token prefill component gate", "",
      f"- required_checks_passed: `{str(required_checks_passed).lower()}`",
      f"- component_admission_passed: `{str(component_admission_passed).lower()}`",
      f"- verdict: `{verdict}`",
      f"- candidate/stock complete ratio: `{complete_ratio:.9f}`",
      f"- candidate complete median: `{candidate_median:.3f} us`",
      f"- stock complete median: `{stock_median:.3f} us`",
      "- candidate one-sided 95% UCB: "
      f"`{performance_inference.get('upper_confidence_bound_ms')} ms`",
      f"- stock one-sided 95% LCB: `{stock_lcb_ms} ms`",
      f"- failed checks: `{failed}`", ""]), encoding="utf-8")
  print(json.dumps({
      "required_checks_passed": required_checks_passed,
      "component_admission_passed": component_admission_passed,
      "verdict": verdict,
      "candidate_stock_ratio": complete_ratio,
      "failed_checks": failed, "out_dir": str(out.relative_to(ROOT))},
      sort_keys=True))
  return 0 if required_checks_passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
