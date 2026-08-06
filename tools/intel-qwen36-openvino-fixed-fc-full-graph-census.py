#!/usr/bin/env python3
"""Gate fixed-FC composition in the full locked OpenVINO candidate graph.

The parent launches one isolated candidate worker, monitors its entire process
group, and stops before system available memory falls below the registered
floor.  The worker composes the accepted hot/cold attention and linear-state
carrier, compiles it, and executes one non-token T=1 diagnostic input.  The
default rewrites all 160 fixed-FC groups.  ``--manager-direct`` leaves the
entire fixed-FC graph native; the plugin fuses only the 40 router/shared groups
and chooses oneDNN-T1/row-major-T>1 internally.  No stock, long-context, or
product timing worker runs.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from xml.sax.saxutils import quoteattr


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-openvino-fixed-fc-full-graph-census-v1"
OV_PYTHON = Path("/home/intel/ov/openvino_env/bin/python")
MODEL_DIR = Path("/home/intel/Qwen3.6-35B-A3B-ov")
CANDIDATE_PLUGIN = Path(
    "/home/intel/intel-qwen36-r0/output/openvino-90214e-l0-gpu/"
    "bin/intel64/Release/libopenvino_intel_gpu_plugin.so")
STOCK_PLUGIN = Path(
    "/home/intel/ov/openvino_env/lib/python3.12/site-packages/openvino/"
    "libs/libopenvino_intel_gpu_plugin.so")
OV_SOURCE = Path(
    "/home/intel/intel-qwen36-r0/source/openvino-90214e5be05")
CUSTOM_CONFIG = ROOT / "engine/openvino/custom/iq36_hot_attention_gqa.xml"
GRAPH_MODULE = ROOT / "tools/intel_qwen36_openvino_hot_cold_attention.py"
FIXED_FC_MODULE = ROOT / "tools/intel_qwen36_openvino_fixed_fc.py"
MODEL_CONTRACT = (
    ROOT / "contracts/qwen36-35b-a3b-openvino-u4-model-contract.json")
STATE_SCHEMA_REFERENCE = ROOT / (
    "output/openvino-hot-cold-product-20260715Tseq1204-"
    "alias-fused-linear-state-32k-o64-cleanZ/raw/sentinel_032k/"
    "correctness/candidate/worker-result.json")
BOUND_SOURCES = (
    CUSTOM_CONFIG,
    ROOT / "engine/openvino/custom/iq36_hot_attention_single_owner.cl",
    ROOT / "engine/openvino/custom/iq36_hot_attention_tiled_helpers.cl",
    ROOT / "engine/openvino/custom/iq36_linear_conv_swish.cl",
    ROOT / "engine/openvino/custom/iq36_fixed_fc_prefix.cl",
    ROOT / "engine/openvino/custom/iq36_fixed_fc_microkernel_shim.cl",
    ROOT / "engine/openvino/custom/iq36_fixed_fc_prefill_wide_microkernel_shim.cl",
    ROOT / "engine/openvino/custom/iq36_fixed_fc_prefill_small_microkernel_shim.cl",
    ROOT / "engine/openvino/custom/iq36_fixed_fc_multi_output.cl",
    ROOT / "engine/openvino/iq36-simplegpu-microkernel-fusion.patch",
    ROOT / "engine/openvino/iq36-fixed-fc-moe-router-compat.patch",
    ROOT / "engine/openvino/iq36-fixed-fc-phase-provider.patch",
    ROOT / "engine/openvino/iq36-level-zero-linear-state-alias.patch",
    GRAPH_MODULE,
    FIXED_FC_MODULE,
    MODEL_CONTRACT,
)
FULL_ATTENTION_LAYERS = (3, 7, 11, 15, 19, 23, 27, 31, 35, 39)
EXPECTED_FIXED_CUSTOM_COUNTS = {
    "IQ36FixedFC1": 80,
    "IQ36FixedFC3": 10,
    "IQ36FixedFC4": 70,
}
ALL_FIXED_FC_COHORTS = (
    "linear_attention_input",
    "full_attention_qkv",
    "router_shared_input",
    "attention_output",
    "shared_expert_down",
)
MANAGER_NATIVE_SUFFIXES = (
    "mlp.shared_expert_gate/ov_ext::linear/MatMul",
    "mlp.shared_expert.gate_proj/ov_ext::linear/MatMul",
    "mlp.shared_expert.up_proj/ov_ext::linear/MatMul",
    "mlp.gate/aten::linear/MatMul",
)
ALLOWED_UNCOMMITTED = {
    "tools/intel-qwen36-openvino-fixed-fc-full-graph-census.py",
}


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path)
  parser.add_argument("--openvino-python", type=Path, default=OV_PYTHON)
  parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
  parser.add_argument(
      "--candidate-plugin", type=Path, default=CANDIDATE_PLUGIN)
  parser.add_argument("--stock-plugin", type=Path, default=STOCK_PLUGIN)
  parser.add_argument("--custom-config", type=Path, default=CUSTOM_CONFIG)
  parser.add_argument("--timeout-s", type=float, default=900.0)
  parser.add_argument("--min-available-gib", type=float, default=8.0)
  parser.add_argument("--abort-below-available-gib", type=float, default=4.0)
  parser.add_argument(
      "--manager-direct", action="store_true",
      help=("leave all fixed FCs native so the candidate plugin selects "
            "oneDNN at T=1 and its row-major router provider at T>1"))
  parser.add_argument("--worker-config", type=Path, help=argparse.SUPPRESS)
  args = parser.parse_args()
  if args.worker_config is None and args.output is None:
    parser.error("--output is required")
  if args.timeout_s <= 0.0:
    parser.error("--timeout-s must be positive")
  if args.abort_below_available_gib > args.min_available_gib:
    parser.error("abort threshold must not exceed preflight threshold")
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
    while chunk := stream.read(8 * 1024 * 1024):
      digest.update(chunk)
  return digest.hexdigest()


def load_module(path: Path, name: str) -> Any:
  spec = importlib.util.spec_from_file_location(name, path)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load module from {path}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def meminfo() -> dict[str, int]:
  values = {}
  for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
    key, rest = line.split(":", 1)
    fields = rest.split()
    if fields and fields[0].isdigit():
      values[key] = int(fields[0]) * 1024
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


def rt_value(value: Any) -> str:
  try:
    return str(value.value)
  except Exception:
    return str(value)


def manager_native_names() -> list[str]:
  prefix = "__module.model.model.language_model.layers."
  return sorted(
      f"{prefix}{layer}.{suffix}"
      for layer in range(40) for suffix in MANAGER_NATIVE_SUFFIXES)


def source_census(model: Any) -> dict[str, Any]:
  counts = Counter(node.get_type_name() for node in model.get_ordered_ops())
  fixed = sorted(
      node.get_friendly_name() for node in model.get_ordered_ops()
      if node.get_type_name() in EXPECTED_FIXED_CUSTOM_COUNTS)
  selected = {
      name: int(counts.get(name, 0))
      for name in (
          "IQ36FixedFC1", "IQ36FixedFC3", "IQ36FixedFC4",
          "IQ36HotAttentionGQA", "IQ36LinearConvSwish", "MatMul",
          "ReadValue", "Assign", "Parameter", "Result")
  }
  return {
      "operation_count": len(model.get_ordered_ops()),
      "selected_type_counts": selected,
      "fixed_custom_names": fixed,
  }


def runtime_census(compiled: Any) -> dict[str, Any]:
  layer_types = Counter()
  fixed = []
  hot_attention = []
  linear_conv = []
  compressed_fc = []
  dynamic_quantize = []
  for node in compiled.get_runtime_model().get_ordered_ops():
    info = {str(key): rt_value(value)
            for key, value in node.get_rt_info().items()}
    layer_type = str(info.get("layerType", node.get_type_name()))
    name = node.get_friendly_name()
    row = {
        "layer_type": layer_type,
        "name": name,
        "primitive_type": str(info.get("primitiveType", "")),
    }
    layer_types[layer_type] += 1
    lowered_type = layer_type.lower().replace("_", "")
    if (layer_type == "CustomGPUPrimitive" and
        name.startswith("iq36_fixed_fc")):
      fixed.append(row)
    elif (layer_type == "CustomGPUPrimitive" and
          name.startswith("iq36_hot_attention_layer")):
      hot_attention.append(row)
    elif (layer_type == "CustomGPUPrimitive" and
          name.startswith("iq36_linear_conv_swish_layer")):
      linear_conv.append(row)
    if "fullyconnected" in lowered_type:
      compressed_fc.append(row)
    if "dynamicquant" in lowered_type:
      dynamic_quantize.append(row)
  return {
      "operation_count": sum(layer_types.values()),
      "layer_types": dict(layer_types),
      "fixed_custom": sorted(fixed, key=lambda row: row["name"]),
      "hot_attention_custom": sorted(
          hot_attention, key=lambda row: row["name"]),
      "linear_conv_custom": sorted(linear_conv, key=lambda row: row["name"]),
      "compressed_fc": compressed_fc,
      "dynamic_quantize": dynamic_quantize,
  }


def profile_census(request: Any) -> dict[str, Any]:
  type_counts = Counter()
  fixed = []
  compressed_fc = []
  dynamic_quantize = []
  for row in request.get_profiling_info():
    profile = {
        "exec_type": row.exec_type,
        "name": row.node_name,
        "real_time_us": row.real_time.total_seconds() * 1_000_000.0,
        "status": str(row.status),
        "type": row.node_type,
    }
    type_counts[row.node_type] += 1
    lowered_type = row.node_type.lower().replace("_", "")
    if row.node_type in EXPECTED_FIXED_CUSTOM_COUNTS:
      fixed.append(profile)
    if "fullyconnected" in lowered_type:
      compressed_fc.append(profile)
    if "dynamicquant" in lowered_type:
      dynamic_quantize.append(profile)
  return {
      "type_counts": dict(type_counts),
      "fixed_custom": sorted(fixed, key=lambda row: row["name"]),
      "compressed_fc": compressed_fc,
      "dynamic_quantize": dynamic_quantize,
  }


def state_schema(request: Any) -> list[dict[str, Any]]:
  rows = []
  for state in request.query_state():
    try:
      tensor = state.state
      rows.append({
          "bytes": int(tensor.byte_size),
          "element_type": str(tensor.element_type),
          "materialized": True,
          "name": str(state.name),
          "shape": [int(value) for value in tensor.shape],
      })
    except Exception as exc:
      rows.append({
          "materialization_error": repr(exc),
          "materialized": False,
          "name": str(state.name),
      })
  return sorted(rows, key=lambda row: row["name"])


def worker_main(config_path: Path) -> int:
  cfg = load_json(config_path)
  if Path(sys.prefix).resolve() != Path(cfg["openvino_python"]).parent.parent:
    raise RuntimeError(f"worker requires {cfg['openvino_python']}")
  if meminfo()["MemAvailable"] < int(cfg["stop_bytes"]):
    raise RuntimeError("worker memory stop at start")

  import numpy as np
  import openvino as ov

  graph = load_module(GRAPH_MODULE, "iq36_fixed_fc_full_graph")
  manager_direct = bool(cfg.get("manager_direct", False))
  raw = Path(cfg["raw"])
  plugin = Path(cfg["candidate_plugin"]).resolve()
  registry = raw / "candidate-plugins.xml"
  registry.write_text(
      "<ie><plugins><plugin name=\"GPU\" location="
      f"{quoteattr(str(plugin))}/></plugins></ie>\n",
      encoding="utf-8")
  core = ov.Core(str(registry))
  config_before = str(core.get_property("GPU", "CONFIG_FILE"))
  core.set_property("GPU", {"CONFIG_FILE": cfg["custom_config"]})
  config_after = str(core.get_property("GPU", "CONFIG_FILE"))

  build_started = time.perf_counter_ns()
  model, summary = graph.make_candidate_model(
      core, Path(cfg["model_dir"]), ov, np, FULL_ATTENTION_LAYERS,
      initialize_hot_states=True,
      fixed_cold_capacity=32768,
      prefill_history_capacity=32768,
      fuse_linear_conv_state=True,
      fuse_fixed_fc=not manager_direct)
  build_ms = (time.perf_counter_ns() - build_started) / 1_000_000.0
  source = source_census(model)
  fixed_summary = dict(summary["fixed_fc_summary"])
  fixed_rows = fixed_summary.pop("fixed_fc_rows", [])
  source["composition"] = {
      "custom_count_after": summary["custom_count_after"],
      "fixed_cold_capacity": summary["fixed_cold_capacity"],
      "fuse_fixed_fc": summary["fuse_fixed_fc"],
      "fuse_linear_conv_state": summary["fuse_linear_conv_state"],
      "initialize_hot_states": summary["initialize_hot_states"],
      "linear_conv_custom_count_after": (
          summary["linear_conv_custom_count_after"]),
      "prefill_history_capacity": summary["prefill_history_capacity"],
      "state_count_after": summary["state_count_after"],
      "target_layers": summary["target_layers"],
  }
  source["fixed_fc_summary"] = fixed_summary
  source["fixed_fc_rows"] = [{
      key: row[key] for key in (
          "arity", "cohort", "layer", "operation", "widths")
  } for row in fixed_rows]
  if manager_direct:
    nodes_by_name = {
        node.get_friendly_name(): node.get_type_name()
        for node in model.get_ordered_ops()
    }
    expected_native = manager_native_names()
    observed_native = [
        name for name in expected_native if nodes_by_name.get(name) == "MatMul"
    ]
    expected_digest = hashlib.sha256(
        "\n".join(expected_native).encode("utf-8")).hexdigest()
    observed_digest = hashlib.sha256(
        "\n".join(observed_native).encode("utf-8")).hexdigest()
    source["manager_native"] = {
        "cohort": "router_shared_input",
        "expected_name_sha256": expected_digest,
        "matmul_names": observed_native,
        "observed_name_sha256": observed_digest,
        "group_count": len(observed_native) // len(MANAGER_NATIVE_SUFFIXES),
        "projection_count": len(observed_native),
    }

  compile_config = {
      "ACTIVATIONS_SCALE_FACTOR": 0.0,
      "DYNAMIC_QUANTIZATION_GROUP_SIZE": 256,
      "PERFORMANCE_HINT": "LATENCY",
      "PERF_COUNT": True,
  }
  compile_started = time.perf_counter_ns()
  compiled = core.compile_model(model, "GPU", compile_config)
  compile_ms = (time.perf_counter_ns() - compile_started) / 1_000_000.0
  after_compile_available = meminfo()["MemAvailable"]
  if after_compile_available < int(cfg["stop_bytes"]):
    raise RuntimeError("inference skipped by worker memory stop after compile")

  runtime = runtime_census(compiled)
  request = compiled.create_infer_request()
  request.reset_state()
  phase = np.arange(2048, dtype=np.float32)
  embeds = (
      np.sin(phase * np.float32(0.03125)) * np.float32(0.125)
  ).reshape(1, 1, 2048)
  feeds = {
      compiled.input("attention_mask"): np.ones((1, 1), dtype=np.int64),
      compiled.input("inputs_embeds"): embeds,
      compiled.input("position_ids"): np.zeros((4, 1, 1), dtype=np.int64),
      compiled.input("beam_idx"): np.zeros((1,), dtype=np.int32),
  }
  infer_started = time.perf_counter_ns()
  outputs = request.infer(feeds, share_outputs=False)
  infer_ms = (time.perf_counter_ns() - infer_started) / 1_000_000.0
  logits = np.asarray(outputs[compiled.output(0)], dtype=np.float32)
  profile = profile_census(request)
  states = state_schema(request)
  result = {
      "pid": os.getpid(),
      "openvino_version": ov.get_version(),
      "candidate_plugin": str(plugin),
      "candidate_plugin_sha256": sha256(plugin),
      "custom_config": str(Path(cfg["custom_config"]).resolve()),
      "custom_config_sha256": sha256(Path(cfg["custom_config"])),
      "config_before": config_before,
      "config_after": config_after,
      "environment": {
          "IQ36_GPU_ALIAS_LINEAR_STATE_ASSIGN": os.environ.get(
              "IQ36_GPU_ALIAS_LINEAR_STATE_ASSIGN"),
          "IQ36_FIXED_FC_MANAGER_TRACE_PATH": os.environ.get(
              "IQ36_FIXED_FC_MANAGER_TRACE_PATH"),
          "IQ36_FIXED_FC_MANAGER_SCOPE": os.environ.get(
              "IQ36_FIXED_FC_MANAGER_SCOPE"),
          "NEO_CACHE_DIR": os.environ.get("NEO_CACHE_DIR"),
      },
      "compile_config": compile_config,
      "build_ms": build_ms,
      "compile_ms": compile_ms,
      "after_compile_available_bytes": after_compile_available,
      "source": source,
      "runtime": runtime,
      "profile": profile,
      "minimal_execution": {
          "input_shape": list(embeds.shape),
          "input_finite": bool(np.isfinite(embeds).all()),
          "infer_ms": infer_ms,
          "output_shape": list(logits.shape),
          "output_finite": bool(np.isfinite(logits).all()),
          "output_abs_max": float(np.max(np.abs(logits))),
          "role": "non-token T=1 full-graph execution census",
      },
      "state_schema": states,
      "state_summary": {
          "byte_count": sum(int(row.get("bytes", 0)) for row in states),
          "count": len(states),
          "materialized_count": sum(
              row.get("materialized") is True for row in states),
      },
  }
  write_json(Path(cfg["result"]), result)
  del outputs, request, compiled, model
  gc.collect()
  return 0


def run_worker(args: argparse.Namespace, raw: Path,
               stop_bytes: int) -> dict[str, Any]:
  worker_raw = raw / "candidate"
  worker_raw.mkdir()
  cache = worker_raw / "neo-cache"
  cache.mkdir()
  config = {
      "candidate_plugin": str(args.candidate_plugin.resolve()),
      "custom_config": str(args.custom_config.resolve()),
      "model_dir": str(args.model_dir.resolve()),
      "openvino_python": str(args.openvino_python.absolute()),
      "raw": str(worker_raw),
      "result": str(worker_raw / "result.json"),
      "stop_bytes": stop_bytes,
      "manager_direct": args.manager_direct,
  }
  config_path = worker_raw / "worker-config.json"
  write_json(config_path, config)
  command = [
      str(args.openvino_python), str(Path(__file__).resolve()),
      "--worker-config", str(config_path),
  ]
  environment = os.environ.copy()
  environment.pop("OV_GPU_CONFIG_FILE", None)
  environment.pop("IQ36_FIXED_FC_MANAGER_SCOPE", None)
  environment.update({
      "IQ36_GPU_ALIAS_LINEAR_STATE_ASSIGN": "1",
      "NEO_CACHE_DIR": str(cache),
      "NEO_CACHE_MAX_SIZE": str(4 * 1024**3),
      "NEO_CACHE_PERSISTENT": "1",
  })
  if args.manager_direct:
    environment["IQ36_FIXED_FC_MANAGER_TRACE_PATH"] = str(
        worker_raw / "manager-trace.jsonl")
    environment["IQ36_FIXED_FC_MANAGER_SCOPE"] = "m1024"
  monitor = {
      "system_available_min_bytes": meminfo()["MemAvailable"],
      "process_group_rss_peak_bytes": 0,
      "process_group_swap_peak_bytes": 0,
      "process_count_peak": 0,
      "samples": 0,
  }
  started = time.monotonic()
  timed_out = guard_tripped = False
  with (worker_raw / "worker.stdout").open("w", encoding="utf-8") as stdout, \
       (worker_raw / "worker.stderr").open("w", encoding="utf-8") as stderr:
    process = subprocess.Popen(
        command, cwd=ROOT, env=environment, stdout=stdout, stderr=stderr,
        text=True, start_new_session=True)
    while process.poll() is None:
      if time.monotonic() - started > args.timeout_s:
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
  stderr_text = (worker_raw / "worker.stderr").read_text(
      encoding="utf-8", errors="replace")
  oom_observed = (
      not guard_tripped and
      (returncode in (-9, 137) or
       "out of memory" in stderr_text.lower() or
       "cl_out_of_resources" in stderr_text.lower()))
  result_path = worker_raw / "result.json"
  return {
      "command": command,
      "elapsed_seconds": time.monotonic() - started,
      "environment": {key: environment[key] for key in (
          "IQ36_GPU_ALIAS_LINEAR_STATE_ASSIGN", "NEO_CACHE_DIR",
          "NEO_CACHE_MAX_SIZE", "NEO_CACHE_PERSISTENT")},
      "returncode": returncode,
      "timed_out": timed_out,
      "memory_guard_tripped": guard_tripped,
      "oom_observed": oom_observed,
      "monitor": monitor,
      "result": load_json(result_path) if result_path.is_file() else {},
  }


def git_state(output: Path) -> dict[str, Any]:
  commit = subprocess.run(
      ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
      capture_output=True, text=True).stdout.strip()
  rows = subprocess.run(
      ["git", "status", "--porcelain"], cwd=ROOT, check=True,
      capture_output=True, text=True).stdout.splitlines()
  output_relative = str(output.resolve().relative_to(ROOT))
  dirty = [
      row for row in rows
      if row[3:] not in ALLOWED_UNCOMMITTED and
      not row[3:].startswith(output_relative)
  ]
  return {
      "commit": commit,
      "dirty": bool(dirty),
      "dirty_paths": dirty,
      "allowed_uncommitted": sorted(ALLOWED_UNCOMMITTED),
  }


def normalize_state_schema(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
  return [{
      key: row.get(key) for key in (
          "bytes", "element_type", "materialized", "name", "shape")
  } for row in rows]


def check(name: str, passed: bool, **evidence: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **evidence}


def main() -> int:
  args = parse_args()
  if args.worker_config is not None:
    return worker_main(args.worker_config)

  output = args.output.resolve()
  raw = output / "raw"
  raw.mkdir(parents=True, exist_ok=False)
  required = (
      args.openvino_python, args.model_dir / "openvino_language_model.xml",
      args.model_dir / "openvino_language_model.bin",
      args.candidate_plugin, args.stock_plugin, args.custom_config,
      STATE_SCHEMA_REFERENCE, *BOUND_SOURCES)
  missing = [str(path) for path in required if not path.is_file()]
  if missing:
    raise SystemExit("missing gate inputs: " + ", ".join(missing))
  preflight_bytes = int(args.min_available_gib * 1024**3)
  stop_bytes = int(args.abort_below_available_gib * 1024**3)
  if meminfo()["MemAvailable"] < preflight_bytes:
    raise RuntimeError("eight-GiB preflight did not clear")

  git = git_state(output)
  worker = run_worker(args, raw, stop_bytes)
  result = worker["result"]
  source = result.get("source", {})
  composition = source.get("composition", {})
  fixed = source.get("fixed_fc_summary", {})
  manager_native = source.get("manager_native", {})
  runtime = result.get("runtime", {})
  profile = result.get("profile", {})
  runtime_fixed_names = [
      row["name"] for row in runtime.get("fixed_custom", [])]
  profile_fixed_names = [
      row["name"] for row in profile.get("fixed_custom", [])]
  source_fixed_names = source.get("fixed_custom_names", [])
  profile_fixed_counts = Counter(
      row["type"] for row in profile.get("fixed_custom", []))
  reference = load_json(STATE_SCHEMA_REFERENCE)
  reference_schema = normalize_state_schema(
      reference.get("state_schema_after", []))
  observed_schema = normalize_state_schema(result.get("state_schema", []))
  worker_ok = (
      worker["returncode"] == 0 and not worker["timed_out"] and
      not worker["memory_guard_tripped"] and not worker["oom_observed"])
  expected_fixed_counts = (
      {}
      if args.manager_direct else EXPECTED_FIXED_CUSTOM_COUNTS)
  expected_fixed_groups = 0 if args.manager_direct else 160
  expected_fixed_projections = 0 if args.manager_direct else 390
  expected_all_custom = 40 if args.manager_direct else 200
  expected_compressed_fc = 331 if args.manager_direct else 1
  expected_dynamic_quantize = 161 if args.manager_direct else 1
  manager_trace_path = raw / "candidate" / "manager-trace.jsonl"
  manager_trace = []
  if manager_trace_path.is_file():
    for line in manager_trace_path.read_text(encoding="utf-8").splitlines():
      try:
        manager_trace.append(json.loads(line))
      except json.JSONDecodeError:
        pass
  manager_selections = [
      row for row in manager_trace
      if row.get("provider") == "iq36_fixed_fc_row_major_u8zp"]
  manager_prepacks = [
      row for row in manager_trace if row.get("stage") == "metadata_prepack"]
  phase_provider_reverse = subprocess.run(
      ["git", "apply", "--check", "--reverse",
       str((ROOT / "engine/openvino/iq36-fixed-fc-phase-provider.patch").resolve())],
      cwd=OV_SOURCE, capture_output=True, text=True)

  checks = [
      check("repository_clean_at_gate", not git["dirty"], git=git),
      check("candidate_worker_completed_under_memory_guard", worker_ok,
            returncode=worker["returncode"],
            timed_out=worker["timed_out"],
            memory_guard_tripped=worker["memory_guard_tripped"],
            oom_observed=worker["oom_observed"]),
      check("candidate_plugin_config_and_alias_are_isolated_and_pinned",
            result.get("candidate_plugin_sha256") ==
            sha256(args.candidate_plugin) and
            sha256(args.candidate_plugin) != sha256(args.stock_plugin) and
            result.get("config_before") == "" and
            result.get("config_after") == str(args.custom_config.resolve()) and
            result.get("environment", {}).get(
                "IQ36_GPU_ALIAS_LINEAR_STATE_ASSIGN") == "1" and
            ((not args.manager_direct) or
             result.get("environment", {}).get(
                 "IQ36_FIXED_FC_MANAGER_TRACE_PATH") ==
             str(manager_trace_path) and
             result.get("environment", {}).get(
                 "IQ36_FIXED_FC_MANAGER_SCOPE") == "m1024")),
      check("accepted_full_graph_carrier_composition_is_exact",
            composition == {
                "custom_count_after": 10,
                "fixed_cold_capacity": 32768,
                "fuse_fixed_fc": not args.manager_direct,
                "fuse_linear_conv_state": True,
                "initialize_hot_states": True,
                "linear_conv_custom_count_after": 30,
                "prefill_history_capacity": 32768,
                "state_count_after": 120,
                "target_layers": list(FULL_ATTENTION_LAYERS),
            } and
            ((args.manager_direct and not fixed) or
             fixed.get("fixed_fc_selected_cohorts") ==
             list(ALL_FIXED_FC_COHORTS)),
            composition=composition),
      check("fixed_fc_source_composition_is_exact",
            ((args.manager_direct and not fixed and
              not source_fixed_names and
              manager_native.get("matmul_names") == manager_native_names() and
              manager_native.get("expected_name_sha256") ==
              manager_native.get("observed_name_sha256") and
              manager_native.get("group_count") == 40 and
              manager_native.get("projection_count") == 160 and
              source.get("selected_type_counts", {}).get("MatMul") == 512) or
             (not args.manager_direct and
              fixed.get("fixed_fc_rewrite_count") == expected_fixed_groups and
              fixed.get("fixed_fc_projection_count") ==
              expected_fixed_projections and
              fixed.get("fixed_fc_custom_counts") == expected_fixed_counts and
              fixed.get("fixed_fc_f16_to_f32_restore_count") ==
              expected_fixed_projections and
              fixed.get("fixed_fc_old_matmuls_remaining") == [] and
              len(source_fixed_names) == expected_fixed_groups)),
            selected_cohorts=fixed.get("fixed_fc_selected_cohorts"),
            fixed_custom_counts=fixed.get("fixed_fc_custom_counts"),
            manager_native={
                key: manager_native.get(key) for key in (
                    "cohort", "expected_name_sha256", "observed_name_sha256",
                    "group_count", "projection_count")
            }),
      check("compiled_runtime_contains_exact_fixed_and_carrier_custom_nodes",
            len(runtime_fixed_names) == expected_fixed_groups and
            runtime_fixed_names == source_fixed_names and
            len(runtime.get("hot_attention_custom", [])) == 10 and
            len(runtime.get("linear_conv_custom", [])) == 30 and
            runtime.get("layer_types", {}).get("CustomGPUPrimitive") ==
            expected_all_custom and
            len(runtime.get("compressed_fc", [])) == expected_compressed_fc and
            len(runtime.get("dynamic_quantize", [])) ==
            expected_dynamic_quantize,
            runtime_custom_counts={
                "fixed": len(runtime_fixed_names),
                "hot_attention": len(
                    runtime.get("hot_attention_custom", [])),
                "linear_conv": len(runtime.get("linear_conv_custom", [])),
                "all_custom_gpu": runtime.get(
                    "layer_types", {}).get("CustomGPUPrimitive"),
                "compressed_fc": len(runtime.get("compressed_fc", [])),
                "dynamic_quantize": len(
                    runtime.get("dynamic_quantize", [])),
            }),
      check("minimal_execution_runs_expected_fixed_fc_provider_set",
            profile_fixed_names == source_fixed_names and
            dict(profile_fixed_counts) == expected_fixed_counts and
            len(profile.get("compressed_fc", [])) == expected_compressed_fc and
            len(profile.get("dynamic_quantize", [])) ==
            expected_dynamic_quantize and
            ((not args.manager_direct) or
             (not manager_selections and not manager_prepacks)),
            profile_fixed_counts=dict(profile_fixed_counts),
            compressed_fc_count=len(profile.get("compressed_fc", [])),
            dynamic_quantize_count=len(profile.get("dynamic_quantize", [])),
            manager_selection_count=len(manager_selections),
            manager_prepack_count=len(manager_prepacks),
            note=("T=1 must fall through to oneDNN; the isolated dynamic "
                  "T1-T128-T1 gate covers row-major manager selection")),
      check("candidate_source_contains_durable_phase_provider_patch",
            (not args.manager_direct) or
            phase_provider_reverse.returncode == 0,
            patch="engine/openvino/iq36-fixed-fc-phase-provider.patch",
            patch_sha256=sha256(
                ROOT / "engine/openvino/iq36-fixed-fc-phase-provider.patch"),
            reverse_check_returncode=phase_provider_reverse.returncode,
            reverse_check_stderr=phase_provider_reverse.stderr),
      check("state_schema_matches_accepted_seq1204_carrier_exactly",
            observed_schema == reference_schema and
            result.get("state_summary") == {
                "byte_count": 1430508160,
                "count": 120,
                "materialized_count": 120,
            },
            observed_summary=result.get("state_summary"),
            reference=str(STATE_SCHEMA_REFERENCE.relative_to(ROOT)),
            reference_sha256=sha256(STATE_SCHEMA_REFERENCE)),
      check("non_token_t1_execution_is_finite",
            result.get("minimal_execution", {}).get("input_shape") ==
            [1, 1, 2048] and
            result.get("minimal_execution", {}).get("output_shape") ==
            [1, 1, 248320] and
            result.get("minimal_execution", {}).get("output_finite") is True,
            execution=result.get("minimal_execution")),
      check("memory_guard_held_without_oom",
            not worker["memory_guard_tripped"] and
            not worker["oom_observed"] and
            worker["monitor"]["system_available_min_bytes"] >= stop_bytes,
            monitor=worker["monitor"],
            note="process swap is pressure telemetry only"),
  ]
  passed = all(row["pass"] for row in checks)
  verdict = (
      ("admit_phase_provider_full_graph_short_correctness_preflight"
       if args.manager_direct else
       "admit_fixed_fc_full_graph_short_correctness_preflight")
      if passed else (
          "phase_provider_full_graph_census_not_admitted"
          if args.manager_direct else
          "fixed_fc_full_graph_census_not_admitted"))
  metrics = {
      "schema": SCHEMA,
      "created_at": datetime.now(timezone.utc).isoformat(),
      "workstream": WORKSTREAM,
      "verdict": verdict,
      "required_checks_passed": passed,
      "git": git,
      "inputs": {
          "manager_direct": args.manager_direct,
          "candidate_plugin": {
              "path": str(args.candidate_plugin.resolve()),
              "sha256": sha256(args.candidate_plugin),
          },
          "stock_plugin": {
              "path": str(args.stock_plugin.resolve()),
              "sha256": sha256(args.stock_plugin),
          },
          "custom_config": {
              "path": str(args.custom_config.resolve()),
              "sha256": sha256(args.custom_config),
          },
          "bound_sources": [{
              "path": str(path.relative_to(ROOT)),
              "sha256": sha256(path),
          } for path in BOUND_SOURCES],
          "state_schema_reference": {
              "path": str(STATE_SCHEMA_REFERENCE.relative_to(ROOT)),
              "sha256": sha256(STATE_SCHEMA_REFERENCE),
          },
      },
      "candidate_worker": worker,
      "manager_trace": {
          "path": (str(manager_trace_path.relative_to(output))
                   if args.manager_direct else None),
          "selection_rows": manager_selections,
          "metadata_prepack_rows": manager_prepacks,
      },
      "checks": checks,
      "next_action": (
          "Run one isolated stock/candidate 2k teacher-forced full-graph "
          "correctness and exact execution-census gate before any long, "
          "ABBA, output512, or product row." if passed else
          "Resolve the first full-graph composition, compile, runtime, state, "
          "or minimal-execution failure without launching a token worker."),
  }
  write_json(output / "metrics.json", metrics)
  write_json(output / "manifest.json", {
      "schema": SCHEMA,
      "workstream": WORKSTREAM,
      "required_checks_passed": passed,
      "metrics": "metrics.json",
      "raw": "raw/",
  })
  print(json.dumps({
      "output": str(output),
      "verdict": verdict,
      "required_checks_passed": passed,
      "failed_checks": [
          row["name"] for row in checks if not row["pass"]],
      "worker": {
          "elapsed_seconds": worker["elapsed_seconds"],
          "monitor": worker["monitor"],
          "returncode": worker["returncode"],
      },
      "execution": result.get("minimal_execution"),
      "runtime_counts": {
          "fixed": len(runtime_fixed_names),
          "hot_attention": len(runtime.get("hot_attention_custom", [])),
          "linear_conv": len(runtime.get("linear_conv_custom", [])),
          "compressed_fc_profile": len(profile.get("compressed_fc", [])),
          "dynamic_quantize_profile": len(
              profile.get("dynamic_quantize", [])),
          "manager_selections": len(manager_selections),
          "manager_prepacks": len(manager_prepacks),
      },
  }, indent=2, sort_keys=True))
  return 0 if passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
