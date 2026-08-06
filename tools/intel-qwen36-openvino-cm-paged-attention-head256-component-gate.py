#!/usr/bin/env python3
"""Gate one complete 32k CM PagedAttention head-256 layer component.

The source bound admits exactly this component: batch 1, 16 query heads,
2 KV heads, head size 256, F16 resident K/V cache, and a 32k generate step.
The worker binds every timed tensor to device-resident remote buffers.  It
uses xattention metadata only to select the CM implementation and bypasses
sparse estimation with the already-bound threshold; this is not a property,
threshold, block, tile, or subgroup sweep.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import signal
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-openvino-cm-paged-attention-head256-component-v1"
OV_PYTHON = Path("/home/intel/ov/openvino_env/bin/python")
OPENVINO_SOURCE = Path(
    "/home/intel/intel-qwen36-r0/source/openvino-90214e5be05")
PATCH = ROOT / "engine/openvino/iq36-cm-paged-attention-head256.patch"
BOUND = ROOT / (
    "output/openvino-cm-paged-attention-head256-bound-"
    "20260717Tseq1291-dependency-complete-cleanZ/metrics.json")
DEFAULT_PLUGIN = Path(
    "/home/intel/intel-qwen36-r0/output/"
    "openvino-90214e-l0-gpu-cm-pa-head256-seq1292/"
    "bin/intel64/Release/libopenvino_intel_gpu_plugin.so")
DEFAULT_CM_FE_DIR = Path(
    "/home/intel/intel-qwen36-r0/output/cmfe-1.0.410-seq1293/lib")
CM_FE_LIBRARY_NAME = "libclangFEWrapper.so.11"

CONTEXT_TOKENS = 32768
BLOCK_SIZE = 256
NUM_BLOCKS = CONTEXT_TOKENS // BLOCK_SIZE
Q_HEADS = 16
KV_HEADS = 2
HEAD_DIM = 256
GQA_RATIO = Q_HEADS // KV_HEADS
COMPLETE_UCB_CAP_MS = 0.5618915
REQUIRED_DENSE_KV_GB_S = 119.43384799378529
COSINE_MIN = 0.999
RELATIVE_L2_MAX = 0.002
MIN_SAMPLES = 20
T95_CONSERVATIVE = 1.729
EXPECTED_PROVIDER = "cm::paged_attention::opt"
DENSE_KV_BYTES = CONTEXT_TOKENS * 2 * KV_HEADS * HEAD_DIM * 2


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path)
  parser.add_argument("--candidate-plugin", type=Path,
                      default=DEFAULT_PLUGIN)
  parser.add_argument("--cm-fe-dir", type=Path, default=DEFAULT_CM_FE_DIR)
  parser.add_argument("--samples", type=int, default=24)
  parser.add_argument("--warmups", type=int, default=5)
  parser.add_argument("--timeout-s", type=float, default=300.0)
  parser.add_argument("--memory-stop-gib", type=float, default=4.0)
  parser.add_argument("--worker-config", type=Path)
  args = parser.parse_args()
  if args.worker_config is None and args.output is None:
    parser.error("--output is required")
  if args.samples < MIN_SAMPLES:
    parser.error(f"--samples must be >= {MIN_SAMPLES}")
  if args.warmups < 1:
    parser.error("--warmups must be positive")
  if args.timeout_s <= 0 or args.memory_stop_gib <= 0:
    parser.error("timeout and memory stop must be positive")
  return args


def load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise TypeError(f"expected JSON object: {path}")
  return value


def write_json(path: Path, value: Any) -> None:
  path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                  encoding="utf-8")


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    while chunk := stream.read(8 * 1024 * 1024):
      digest.update(chunk)
  return digest.hexdigest()


def relative(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path.resolve())


def available_memory_bytes() -> int:
  for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
    if line.startswith("MemAvailable:"):
      return int(line.split()[1]) * 1024
  raise RuntimeError("MemAvailable is missing")


def process_memory(pid: int) -> tuple[int, int]:
  try:
    rows = Path(f"/proc/{pid}/status").read_text(
        encoding="utf-8", errors="replace").splitlines()
  except FileNotFoundError:
    return 0, 0
  values: dict[str, int] = {}
  for row in rows:
    if row.startswith(("VmRSS:", "VmSwap:")):
      key, raw, _ = row.split(maxsplit=2)
      values[key.rstrip(":")] = int(raw) * 1024
  return values.get("VmRSS", 0), values.get("VmSwap", 0)


def git_state(output: Path) -> dict[str, Any]:
  def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, text=True, capture_output=True,
        check=True).stdout.strip()

  dirty = git("status", "--porcelain").splitlines()
  try:
    output_relative = str(output.resolve().relative_to(ROOT))
  except ValueError:
    output_relative = ""
  dirty = [row for row in dirty
           if not output_relative or output_relative not in row]
  return {
      "commit": git("rev-parse", "HEAD"),
      "dirty": bool(dirty),
      "dirty_paths": dirty,
  }


def check(name: str, passed: bool, **evidence: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **evidence}


def deterministic_array(np: Any, shape: tuple[int, ...], seed: int) -> Any:
  rng = np.random.default_rng(seed)
  return rng.standard_normal(shape, dtype=np.float32).astype(np.float16)


def make_model(ov: Any, np: Any) -> Any:
  from openvino.op import _PagedAttentionExtension

  def parameter(shape: tuple[int, ...], name: str) -> Any:
    return ov.opset13.parameter(
        ov.PartialShape(list(shape)), ov.Type.f16, name=name)

  def constant(value: Any) -> Any:
    return ov.opset13.constant(np.asarray(value))

  # CM PagedAttention sizes runtime buffers after concrete request shapes are
  # installed. Keep only the request-varying first dimension dynamic, matching
  # the production graph and the upstream CM PagedAttention tests.
  query = parameter((-1, Q_HEADS * HEAD_DIM), "query")
  key = parameter((-1, KV_HEADS * HEAD_DIM), "key")
  value = parameter((-1, KV_HEADS * HEAD_DIM), "value")
  key_cache = parameter(
      (-1, KV_HEADS, BLOCK_SIZE, HEAD_DIM), "key_cache")
  value_cache = parameter(
      (-1, KV_HEADS, BLOCK_SIZE, HEAD_DIM), "value_cache")

  inputs = [
      query,
      key,
      value,
      key_cache,
      value_cache,
      constant(np.asarray([CONTEXT_TOKENS - 1], dtype=np.int32)),
      constant(np.asarray([0, 1], dtype=np.int32)),
      constant(np.arange(NUM_BLOCKS, dtype=np.int32)),
      constant(np.asarray([0, NUM_BLOCKS], dtype=np.int32)),
      constant(np.asarray(1.0 / math.sqrt(HEAD_DIM), dtype=np.float32)),
      constant(np.asarray(0, dtype=np.int32)),
      constant(np.empty((0,), dtype=np.float32)),
      constant(np.asarray(CONTEXT_TOKENS, dtype=np.int32)),
      constant(np.asarray(0, dtype=np.int32)),
      constant(np.empty((0,), dtype=np.int32)),
      constant(np.empty((0,), dtype=np.int32)),
      constant(np.empty((0,), dtype=np.float16)),
      constant(np.asarray([100.0], dtype=np.float32)),
      constant(np.asarray(128, dtype=np.int32)),
      constant(np.asarray(8, dtype=np.int32)),
      constant(np.empty((0,), dtype=np.float16)),
      constant(np.asarray(0, dtype=np.int32)),
      constant(np.empty((0,), dtype=np.int32)),
      constant(np.empty((0,), dtype=np.int32)),
      constant(np.empty((0,), dtype=np.int32)),
      constant(np.empty((0,), dtype=np.int32)),
      constant(np.empty((0,), dtype=np.uint8)),
      constant(np.empty((0,), dtype=np.int32)),
  ]
  operation = _PagedAttentionExtension(
      [node.output(0) for node in inputs])
  operation.set_friendly_name("iq36_cm_paged_attention_head256")
  return ov.Model(
      [operation.output(0)],
      [query, key, value, key_cache, value_cache],
      "iq36_cm_paged_attention_head256_component")


def any_value(value: Any) -> Any:
  try:
    return value.value
  except Exception:
    return str(value)


def runtime_rows(compiled: Any) -> list[dict[str, Any]]:
  rows = []
  for node in compiled.get_runtime_model().get_ordered_ops():
    info = {str(key): any_value(value)
            for key, value in node.get_rt_info().items()}
    name = str(node.get_friendly_name())
    layer_type = str(info.get("layerType", ""))
    primitive_type = str(info.get("primitiveType", ""))
    if "paged" not in " ".join(
        (name, layer_type, primitive_type, str(node.get_type_name()))).lower():
      continue
    rows.append({
        "node_name": name,
        "node_type": str(node.get_type_name()),
        "layer_type": layer_type,
        "primitive_type": primitive_type,
        "runtime_precision": str(info.get("runtimePrecision", "")),
        "output_layouts": str(info.get("outputLayouts", "")),
        "output_precisions": str(info.get("outputPrecisions", "")),
    })
  return rows


def worker_main(config_path: Path) -> int:
  import numpy as np
  import openvino as ov

  cfg = load_json(config_path)
  plugin = Path(cfg["candidate_plugin"])
  cm_fe_library = Path(cfg["cm_fe_library"])
  cm_fe = ctypes.CDLL(str(cm_fe_library.resolve()))
  cm_fe.IntelCMClangFEGetInterfaceVersion.restype = ctypes.c_uint
  cm_fe_interface_version = int(
      cm_fe.IntelCMClangFEGetInterfaceVersion())
  core = ov.Core()
  core.register_plugin(str(plugin.resolve()), "GPUX")
  context = core.get_default_context("GPUX")
  model = make_model(ov, np)
  compile_config = {
      "PERFORMANCE_HINT": "LATENCY",
      "PERF_COUNT": False,
      "KV_CACHE_PRECISION": "f16",
  }
  compile_started = time.perf_counter_ns()
  compiled = core.compile_model(model, context, compile_config)
  compile_ms = (time.perf_counter_ns() - compile_started) / 1_000_000.0
  rows = runtime_rows(compiled)
  timed_request = compiled.create_infer_request()
  correctness_request = compiled.create_infer_request()

  shapes = {
      "query": (1, Q_HEADS * HEAD_DIM),
      "key": (1, KV_HEADS * HEAD_DIM),
      "value": (1, KV_HEADS * HEAD_DIM),
      "key_cache": (NUM_BLOCKS, KV_HEADS, BLOCK_SIZE, HEAD_DIM),
      "value_cache": (NUM_BLOCKS, KV_HEADS, BLOCK_SIZE, HEAD_DIM),
  }
  seeds = {
      "query": 1292001,
      "key": 1292002,
      "value": 1292003,
      "key_cache": 1292004,
      "value_cache": 1292005,
  }
  remote_tensors: dict[str, Any] = {}
  for name in ("query", "key", "value", "key_cache", "value_cache"):
    host = deterministic_array(np, shapes[name], seeds[name])
    remote = context.create_tensor(
        ov.Type.f16, ov.Shape(list(shapes[name])), {})
    remote.copy_from(ov.Tensor(host))
    timed_request.set_tensor(name, remote)
    correctness_request.set_tensor(name, remote)
    remote_tensors[name] = remote
    del host

  output_shape = (1, Q_HEADS * HEAD_DIM)
  output_remote = context.create_tensor(
      ov.Type.f16, ov.Shape(list(output_shape)), {})
  timed_request.set_tensor(compiled.output(0).get_any_name(), output_remote)

  def infer_remote() -> None:
    # infer() materializes the returned output mapping and therefore asks a
    # remote output tensor for a host data pointer. Async wait leaves the
    # output resident and measures the same submit-to-completion interval.
    timed_request.start_async()
    timed_request.wait()

  for _ in range(int(cfg["warmups"])):
    infer_remote()
  samples = []
  for _ in range(int(cfg["samples"])):
    started = time.perf_counter_ns()
    infer_remote()
    samples.append((time.perf_counter_ns() - started) / 1_000_000.0)

  numeric_started = time.perf_counter_ns()
  correctness_request.infer()
  numeric_infer_ms = (
      time.perf_counter_ns() - numeric_started) / 1_000_000.0
  output = np.array(
      correctness_request.get_output_tensor(0).data, copy=True)
  np.save(cfg["output_npy"], output, allow_pickle=False)
  properties = {
      key: str(compiled.get_property(key))
      for key in ("EXECUTION_DEVICES", "PERFORMANCE_HINT", "PERF_COUNT")
  }
  write_json(Path(cfg["result_json"]), {
      "compile_config": compile_config,
      "compile_ms": compile_ms,
      "internal_environment": {
          key: os.environ.get(key) for key in (
              "OV_GPU_USE_CM", "OV_GPU_ALLOW_BYPASS_XATTN_EXEC",
              "OV_GPU_PA_MIXED_ROUTE_MODE", "CM_FE_DIR")
      },
      "cm_frontend": {
          "interface_version": cm_fe_interface_version,
          "library": str(cm_fe_library.resolve()),
          "library_sha256": sha256(cm_fe_library),
      },
      "device": {
          key: str(core.get_property("GPUX", key))
          for key in ("FULL_DEVICE_NAME", "DEVICE_ARCHITECTURE",
                      "DEVICE_TYPE", "GPU_DEVICE_TOTAL_MEM_SIZE")
      },
      "openvino_version": ov.__version__,
      "output_shape": list(output.shape),
      "timed_output": {
          "host_transfer_in_timed_scope": False,
          "remote": True,
          "shape": list(output_shape),
      },
      "numeric_lane": {
          "host_output_after_timing": True,
          "infer_ms": numeric_infer_ms,
          "shared_remote_inputs": True,
      },
      "plugin": str(plugin.resolve()),
      "plugin_sha256": sha256(plugin),
      "properties": properties,
      "remote_inputs": sorted(remote_tensors),
      "runtime_rows": rows,
      "samples_ms": samples,
      "warmups": int(cfg["warmups"]),
  })
  print(json.dumps({
      "event": "complete",
      "median_ms": statistics.median(samples),
      "provider_rows": rows,
      "samples": len(samples),
  }, sort_keys=True), flush=True)
  return 0


def run_worker(args: argparse.Namespace, raw: Path) -> dict[str, Any]:
  cm_fe_library = args.cm_fe_dir.resolve() / CM_FE_LIBRARY_NAME
  config = {
      "candidate_plugin": str(args.candidate_plugin.resolve()),
      "cm_fe_library": str(cm_fe_library),
      "output_npy": str(raw / "candidate-output.npy"),
      "result_json": str(raw / "candidate-result.json"),
      "samples": args.samples,
      "warmups": args.warmups,
  }
  config_path = raw / "worker-config.json"
  write_json(config_path, config)
  command = [str(OV_PYTHON), str(Path(__file__).resolve()),
             "--worker-config", str(config_path)]
  environment = os.environ.copy()
  environment.pop("OV_GPU_CONFIG_FILE", None)
  environment.update({
      "NEO_CACHE_PERSISTENT": "0",
      "OMP_NUM_THREADS": "4",
      "OPENBLAS_NUM_THREADS": "4",
      "OV_GPU_USE_CM": "ON",
      "OV_GPU_ALLOW_BYPASS_XATTN_EXEC": "ON",
      "OV_GPU_PA_MIXED_ROUTE_MODE": "split",
      "CM_FE_DIR": str(args.cm_fe_dir.resolve()),
  })
  stdout_path = raw / "worker.stdout"
  stderr_path = raw / "worker.stderr"
  stop_bytes = int(args.memory_stop_gib * 1024**3)
  started = time.monotonic()
  peak_rss = 0
  peak_swap = 0
  minimum_available = available_memory_bytes()
  guard_tripped = False
  timed_out = False
  with stdout_path.open("w", encoding="utf-8") as stdout, \
       stderr_path.open("w", encoding="utf-8") as stderr:
    process = subprocess.Popen(
        command, cwd=ROOT, env=environment, stdout=stdout, stderr=stderr,
        text=True, start_new_session=True)
    while process.poll() is None:
      rss, swap = process_memory(process.pid)
      peak_rss = max(peak_rss, rss)
      peak_swap = max(peak_swap, swap)
      available = available_memory_bytes()
      minimum_available = min(minimum_available, available)
      if available < stop_bytes:
        guard_tripped = True
      if time.monotonic() - started > args.timeout_s:
        timed_out = True
      if guard_tripped or timed_out:
        os.killpg(process.pid, signal.SIGTERM)
        try:
          process.wait(timeout=5)
        except subprocess.TimeoutExpired:
          os.killpg(process.pid, signal.SIGKILL)
        break
      time.sleep(0.05)
    returncode = process.wait()
  wall_s = time.monotonic() - started
  result_path = raw / "candidate-result.json"
  result = load_json(result_path) if result_path.is_file() else {}
  return {
      "command": command,
      "environment": {key: environment[key] for key in (
          "NEO_CACHE_PERSISTENT", "OMP_NUM_THREADS",
          "OPENBLAS_NUM_THREADS", "OV_GPU_USE_CM",
          "OV_GPU_ALLOW_BYPASS_XATTN_EXEC",
          "OV_GPU_PA_MIXED_ROUTE_MODE", "CM_FE_DIR")},
      "guard_tripped": guard_tripped,
      "minimum_available_bytes": minimum_available,
      "oom_observed": returncode in (-9, 137) and not guard_tripped,
      "peak_rss_bytes": peak_rss,
      "peak_swap_bytes": peak_swap,
      "result": result,
      "returncode": returncode,
      "timed_out": timed_out,
      "wall_s": wall_s,
  }


def numpy_reference() -> Any:
  import numpy as np

  query = deterministic_array(
      np, (1, Q_HEADS * HEAD_DIM), 1292001).astype(np.float32)
  key = deterministic_array(
      np, (1, KV_HEADS * HEAD_DIM), 1292002).astype(np.float32)
  value = deterministic_array(
      np, (1, KV_HEADS * HEAD_DIM), 1292003).astype(np.float32)
  key_cache = deterministic_array(
      np, (NUM_BLOCKS, KV_HEADS, BLOCK_SIZE, HEAD_DIM), 1292004)
  value_cache = deterministic_array(
      np, (NUM_BLOCKS, KV_HEADS, BLOCK_SIZE, HEAD_DIM), 1292005)
  keys = key_cache.transpose(0, 2, 1, 3).reshape(
      CONTEXT_TOKENS, KV_HEADS, HEAD_DIM).astype(np.float32)
  values = value_cache.transpose(0, 2, 1, 3).reshape(
      CONTEXT_TOKENS, KV_HEADS, HEAD_DIM).astype(np.float32)
  keys[-1] = key.reshape(KV_HEADS, HEAD_DIM)
  values[-1] = value.reshape(KV_HEADS, HEAD_DIM)
  query = query.reshape(Q_HEADS, HEAD_DIM)
  output = np.empty((Q_HEADS, HEAD_DIM), dtype=np.float32)
  scale = np.float32(1.0 / math.sqrt(HEAD_DIM))
  for kv_head in range(KV_HEADS):
    q_start = kv_head * GQA_RATIO
    q_end = q_start + GQA_RATIO
    scores = (query[q_start:q_end] @ keys[:, kv_head].T) * scale
    scores -= scores.max(axis=1, keepdims=True)
    weights = np.exp(scores, dtype=np.float32)
    weights /= weights.sum(axis=1, keepdims=True)
    output[q_start:q_end] = weights @ values[:, kv_head]
  return output.reshape(1, Q_HEADS * HEAD_DIM)


def vector_metrics(reference: Any, candidate: Any) -> dict[str, Any]:
  import numpy as np

  ref = np.asarray(reference, dtype=np.float64).reshape(-1)
  cand = np.asarray(candidate, dtype=np.float64).reshape(-1)
  finite = bool(np.isfinite(ref).all() and np.isfinite(cand).all())
  delta = cand - ref
  ref_norm = float(np.linalg.norm(ref))
  cand_norm = float(np.linalg.norm(cand))
  return {
      "count": int(ref.size),
      "cosine": (
          float(np.dot(ref, cand) / (ref_norm * cand_norm))
          if ref_norm and cand_norm and finite else 0.0),
      "finite": finite,
      "max_abs": float(np.max(np.abs(delta))) if finite else float("inf"),
      "reference_l2": ref_norm,
      "relative_l2": (
          float(np.linalg.norm(delta) / ref_norm)
          if ref_norm and finite else float("inf")),
  }


def main() -> int:
  args = parse_args()
  if args.worker_config is not None:
    return worker_main(args.worker_config.resolve())

  output = args.output.resolve()
  output.mkdir(parents=True, exist_ok=False)
  raw = output / "raw"
  raw.mkdir()
  cm_fe_library = args.cm_fe_dir.resolve() / CM_FE_LIBRARY_NAME
  required = (BOUND, PATCH, args.candidate_plugin, OV_PYTHON,
              OPENVINO_SOURCE / ".git", cm_fe_library)
  missing = [relative(path) for path in required if not path.exists()]
  if missing:
    raise SystemExit("missing CM component inputs: " + ", ".join(missing))

  git = git_state(output)
  bound = load_json(BOUND)
  patch_reverse = subprocess.run(
      ["git", "apply", "--check", "--reverse", str(PATCH.resolve())],
      cwd=OPENVINO_SOURCE, text=True, capture_output=True).returncode == 0
  worker = run_worker(args, raw)
  worker_result = worker["result"]
  samples = [float(value) for value in worker_result.get("samples_ms", [])]
  mean_ms = statistics.mean(samples) if samples else float("inf")
  stdev_ms = statistics.stdev(samples) if len(samples) > 1 else float("inf")
  ucb_ms = (
      mean_ms + T95_CONSERVATIVE * stdev_ms / math.sqrt(len(samples))
      if len(samples) > 1 else float("inf"))
  median_ms = statistics.median(samples) if samples else float("inf")
  bandwidth = DENSE_KV_BYTES / (ucb_ms * 1_000_000.0)

  import numpy as np
  candidate_path = raw / "candidate-output.npy"
  candidate = (
      np.load(candidate_path, allow_pickle=False)
      if candidate_path.is_file() else np.empty((0,), dtype=np.float32))
  reference_started = time.perf_counter_ns()
  reference = numpy_reference() if candidate.size else np.empty((0,))
  reference_ms = (time.perf_counter_ns() - reference_started) / 1_000_000.0
  numeric = (
      vector_metrics(reference, candidate)
      if candidate.size else {
          "count": 0, "cosine": 0.0, "finite": False,
          "max_abs": float("inf"), "reference_l2": 0.0,
          "relative_l2": float("inf")})
  rows = worker_result.get("runtime_rows", [])
  raw_providers = sorted(set(
      str(row.get("primitive_type", "")) for row in rows
      if row.get("primitive_type")))
  providers = sorted(set(
      value.split("___", 1)[0] for value in raw_providers))
  bound_contract = bound["component_contract"]
  bound_budget = bound["budget"]

  checks = [
      check("repository_clean_at_gate", not git["dirty"],
            dirty_paths=git["dirty_paths"]),
      check("dependency_complete_source_bound_passed",
            bound.get("required_checks_passed") is True
            and bound.get("verdict")
            == "admit_cm_paged_attention_head256_compile_and_one_component"),
      check("exact_backport_is_present_in_staging_source", patch_reverse,
            patch_sha256=sha256(PATCH)),
      check("candidate_plugin_is_exact_and_worker_isolated",
            worker_result.get("plugin_sha256")
            == sha256(args.candidate_plugin)
            and worker_result.get("plugin")
            == str(args.candidate_plugin.resolve()),
            plugin=relative(args.candidate_plugin),
            plugin_sha256=sha256(args.candidate_plugin)),
      check("cm_frontend_abi11_is_exact_and_worker_isolated",
            worker_result.get("cm_frontend") == {
                "interface_version": 11,
                "library": str(cm_fe_library),
                "library_sha256": sha256(cm_fe_library),
            }
            and worker_result.get("internal_environment", {}).get(
                "CM_FE_DIR") == str(args.cm_fe_dir.resolve()),
            cm_fe_dir=str(args.cm_fe_dir.resolve()),
            cm_fe_library_sha256=sha256(cm_fe_library),
            worker_cm_frontend=worker_result.get("cm_frontend")),
      check("worker_completed_without_oom_timeout_or_memory_guard",
            worker["returncode"] == 0 and not worker["timed_out"]
            and not worker["guard_tripped"] and not worker["oom_observed"],
            returncode=worker["returncode"],
            timed_out=worker["timed_out"],
            guard_tripped=worker["guard_tripped"],
            oom_observed=worker["oom_observed"]),
      check("locked_abi_and_resident_f16_complete_scope_are_exact",
            (CONTEXT_TOKENS, Q_HEADS, KV_HEADS, HEAD_DIM, GQA_RATIO,
             DENSE_KV_BYTES)
            == (32768, 16, 2, 256, 8, 67108864)
            and worker_result.get("remote_inputs")
            == ["key", "key_cache", "query", "value", "value_cache"]
            and worker_result.get("timed_output") == {
                "host_transfer_in_timed_scope": False,
                "remote": True,
                "shape": [1, Q_HEADS * HEAD_DIM],
            }
            and worker_result.get("numeric_lane", {}).get(
                "host_output_after_timing") is True
            and worker_result.get("numeric_lane", {}).get(
                "shared_remote_inputs") is True,
            dense_kv_bytes=DENSE_KV_BYTES,
            remote_inputs=worker_result.get("remote_inputs"),
            timed_output=worker_result.get("timed_output"),
            numeric_lane=worker_result.get("numeric_lane")),
      check("runtime_provider_is_exact_cm_paged_attention_opt",
            EXPECTED_PROVIDER in providers,
            providers=providers, raw_providers=raw_providers,
            runtime_rows=rows),
      check("minimum_complete_sample_count", len(samples) >= MIN_SAMPLES,
            sample_count=len(samples)),
      check("complete_one_sided_95pct_ucb_clears_cap",
            ucb_ms <= COMPLETE_UCB_CAP_MS,
            mean_ms=mean_ms, stdev_ms=stdev_ms,
            conservative_t95=T95_CONSERVATIVE, ucb_ms=ucb_ms,
            cap_ms=COMPLETE_UCB_CAP_MS),
      check("effective_dense_kv_rate_clears_reopen_contract",
            bandwidth >= REQUIRED_DENSE_KV_GB_S,
            effective_dense_kv_gb_s=bandwidth,
            required_dense_kv_gb_s=REQUIRED_DENSE_KV_GB_S),
      check("independent_f32_reference_numeric_clears_contract",
            numeric["finite"] and numeric["cosine"] >= COSINE_MIN
            and numeric["relative_l2"] <= RELATIVE_L2_MAX,
            **numeric),
      check("source_bound_and_component_thresholds_are_identical",
            float(bound_budget["per_layer_complete_ucb_cap_ms"])
            == COMPLETE_UCB_CAP_MS
            and float(bound_budget["required_complete_dense_kv_gb_s"])
            == REQUIRED_DENSE_KV_GB_S
            and int(bound_contract["minimum_complete_samples"])
            == MIN_SAMPLES
            and float(bound_contract["numeric_cosine_min"]) == COSINE_MIN
            and float(bound_contract["numeric_relative_l2_max"])
            == RELATIVE_L2_MAX),
  ]
  required_passed = all(row["pass"] for row in checks)
  component_measured = all(row["pass"] for row in checks[:9])
  if required_passed:
    verdict = "promote_cm_paged_attention_head256_to_one_graph_abba"
  elif component_measured:
    verdict = "close_cm_paged_attention_head256_before_graph_integration"
  else:
    verdict = "component_not_measured_keep_route_open_for_harness_correction"
  payload = {
      "schema": SCHEMA,
      "created_at": datetime.now(timezone.utc).isoformat(),
      "workstream": WS,
      "git": git,
      "verdict": verdict,
      "required_checks_passed": required_passed,
      "speedup_claims_allowed": False,
      "graph_integration_admitted": required_passed,
      "product_worker_admitted": False,
      "long_worker_admitted": False,
      "checks": checks,
      "component": {
          "context_tokens": CONTEXT_TOKENS,
          "query_heads": Q_HEADS,
          "kv_heads": KV_HEADS,
          "head_dim": HEAD_DIM,
          "gqa_ratio": GQA_RATIO,
          "cache_precision": "F16",
          "dense_kv_bytes": DENSE_KV_BYTES,
          "samples_ms": samples,
          "mean_ms": mean_ms,
          "median_ms": median_ms,
          "stdev_ms": stdev_ms,
          "one_sided_95pct_ucb_ms": ucb_ms,
          "ucb_cap_ms": COMPLETE_UCB_CAP_MS,
          "effective_dense_kv_gb_s_at_ucb": bandwidth,
          "required_dense_kv_gb_s": REQUIRED_DENSE_KV_GB_S,
          "provider": providers,
          "provider_raw": raw_providers,
          "numeric": numeric,
          "reference_wall_ms": reference_ms,
      },
      "worker": worker,
      "memory": {
          "memory_stop_bytes": int(args.memory_stop_gib * 1024**3),
          "minimum_available_bytes": worker["minimum_available_bytes"],
          "peak_rss_bytes": worker["peak_rss_bytes"],
          "peak_swap_bytes": worker["peak_swap_bytes"],
      },
      "inputs": {
          relative(path): sha256(path) for path in required if path.is_file()
      },
  }
  write_json(output / "metrics.json", payload)
  (output / "summary.md").write_text(
      "# CM PagedAttention head-256 complete component\n\n"
      f"Verdict: **{verdict}**. Required checks: `{required_passed}`.\n\n"
      f"The device-resident 32k one-layer component measured mean / median / "
      f"one-sided 95% UCB `{mean_ms:.6f} / {median_ms:.6f} / "
      f"{ucb_ms:.6f} ms`, equivalent to `{bandwidth:.3f} GB/s` of complete "
      f"dense K+V traffic. Provider: `{', '.join(providers)}`. Numeric cosine "
      f"/ relative-L2: `{numeric['cosine']:.9f} / "
      f"{numeric['relative_l2']:.9f}`.\n\n"
      "No model, graph integration, product, long-context, output512, or ABBA "
      "worker ran in this gate.\n",
      encoding="utf-8")
  print(json.dumps({
      "effective_dense_kv_gb_s": bandwidth,
      "numeric_cosine": numeric["cosine"],
      "numeric_relative_l2": numeric["relative_l2"],
      "provider": providers,
      "required_checks_passed": required_passed,
      "ucb_ms": ucb_ms,
      "verdict": verdict,
  }, sort_keys=True))
  return 0 if required_passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
