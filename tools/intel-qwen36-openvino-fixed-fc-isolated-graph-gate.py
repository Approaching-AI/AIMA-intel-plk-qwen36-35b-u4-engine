#!/usr/bin/env python3
"""Gate phase-specialized fixed-FC providers for every locked graph family.

This gate launches one isolated candidate GPU worker.  It first audits the
complete 390-projection rewrite, then runs each of five exact product cohorts
through one dynamic compile in the T=2048, T=1, T=2048 sequence.  Provider
creation telemetry binds each concrete shape to its exact microkernel package
and work-group geometry.  It never loads the full model on GPU and never
launches a stock or product worker.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from xml.sax.saxutils import quoteattr


ROOT = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-openvino-fixed-fc-isolated-graph-gate-v1"
OV_PYTHON = Path("/home/intel/ov/openvino_env/bin/python")
OV_SOURCE = Path(
    "/home/intel/intel-qwen36-r0/source/openvino-90214e5be05")
PLUGIN = Path(
    "/home/intel/intel-qwen36-r0/output/openvino-90214e-l0-gpu/"
    "bin/intel64/Release/libopenvino_intel_gpu_plugin.so")
CONFIG = ROOT / "engine/openvino/custom/iq36_hot_attention_gqa.xml"
GRAPH = ROOT / "tools/intel_qwen36_openvino_fixed_fc.py"
PATCH = ROOT / "engine/openvino/iq36-simplegpu-microkernel-fusion.patch"
DECODE_SHIM = (
    ROOT / "engine/openvino/custom/iq36_fixed_fc_microkernel_shim.cl")
PREFILL_WIDE_SHIM = (
    ROOT /
    "engine/openvino/custom/iq36_fixed_fc_prefill_wide_microkernel_shim.cl")
PREFILL_SMALL_SHIM = (
    ROOT /
    "engine/openvino/custom/iq36_fixed_fc_prefill_small_microkernel_shim.cl")
KERNEL = ROOT / "engine/openvino/custom/iq36_fixed_fc_multi_output.cl"
SHIM_SPECS = (
    (DECODE_SHIM, "// IQ36_FIXED_FC_DECODE_BEGIN",
     "// IQ36_FIXED_FC_DECODE_END", "#define IQ36_K_PARALLEL_LOCAL 1",
     "a31a6e7ab718cd5f6df1b3b89ab496fac8315c3cf79ccae135e8fbb6d9b53e87"),
    (PREFILL_WIDE_SHIM, "// IQ36_FIXED_FC_PREFILL_WIDE_BEGIN",
     "// IQ36_FIXED_FC_PREFILL_WIDE_END",
     "#define IQ36_K_PARALLEL_LOCAL 0",
     "a267a0a0ae2f20bf52b5867f08fa2c6c505aa1b719cbc7b8dcf06144db672cc2"),
    (PREFILL_SMALL_SHIM, "// IQ36_FIXED_FC_PREFILL_SMALL_BEGIN",
     "// IQ36_FIXED_FC_PREFILL_SMALL_END",
     "#define IQ36_K_PARALLEL_LOCAL 0",
     "1d5a924fa9c84de33f4ca0ac3807e7f04910483d1de9edde8511810b71c4205b"),
)
COHORTS = (
    ("linear_attention_input", (8192, 32, 32, 4096), 2048),
    ("full_attention_qkv", (8192, 512, 512), 2048),
    ("router_shared_input", (1, 512, 512, 256), 2048),
    ("attention_output", (2048,), 4096),
    ("shared_expert_down", (2048,), 512),
)
DECODE_DISPATCH = {
    "linear_attention_input": ((6208, 1, 8), (32, 1, 8)),
    "full_attention_qkv": ((4608, 1, 8), (32, 1, 8)),
    "router_shared_input": ((672, 1, 8), (32, 1, 8)),
    "attention_output": ((1024, 1, 8), (32, 1, 8)),
    "shared_expert_down": ((1024, 1, 8), (32, 1, 8)),
}
PREFILL_2048_DISPATCH = {
    "linear_attention_input": ((6272, 64, 1), (64, 2, 1)),
    "full_attention_qkv": ((4608, 64, 1), (64, 2, 1)),
    "router_shared_input": ((768, 32, 1), (128, 4, 1)),
    "attention_output": ((1024, 32, 1), (128, 4, 1)),
    "shared_expert_down": ((1024, 32, 1), (128, 4, 1)),
}
ALLOWED_UNCOMMITTED = {
    "engine/openvino/custom/iq36_fixed_fc_microkernel_shim.cl",
    "engine/openvino/custom/iq36_fixed_fc_multi_output.cl",
    "engine/openvino/custom/iq36_fixed_fc_prefill_small_microkernel_shim.cl",
    "engine/openvino/custom/iq36_fixed_fc_prefill_wide_microkernel_shim.cl",
    "engine/openvino/custom/iq36_hot_attention_gqa.xml",
    "engine/openvino/iq36-simplegpu-microkernel-fusion.patch",
    "tools/intel-qwen36-openvino-fixed-fc-full-graph-census.py",
    "tools/intel_qwen36_openvino_fixed_fc.py",
    "tools/intel-qwen36-openvino-fixed-fc-isolated-graph-gate.py",
    "tools/intel-qwen36-openvino-hot-cold-product-gate.py",
}


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path)
  parser.add_argument("--plugin", type=Path, default=PLUGIN)
  parser.add_argument("--config", type=Path, default=CONFIG)
  parser.add_argument("--openvino-python", type=Path, default=OV_PYTHON)
  parser.add_argument("--warmups", type=int, default=512)
  parser.add_argument("--samples", type=int, default=64)
  parser.add_argument("--tokens", type=int, default=1)
  parser.add_argument(
      "--phase-specialized", action="store_true",
      help="Run one dynamic compile through tokens, 1, tokens")
  parser.add_argument(
      "--numeric-tokens", type=int, default=0,
      help="Prefix tokens checked against the CPU oracle; zero means all")
  parser.add_argument("--timeout-s", type=float, default=300.0)
  parser.add_argument("--min-available-gib", type=float, default=8.0)
  parser.add_argument("--abort-below-available-gib", type=float, default=4.0)
  parser.add_argument(
      "--queue-throttle", choices=("LOW", "MEDIUM", "HIGH"),
      default="MEDIUM")
  parser.add_argument("--worker-config", type=Path, help=argparse.SUPPRESS)
  args = parser.parse_args()
  if args.worker_config is None and args.output is None:
    parser.error("--output is required")
  if (args.warmups < 1 or args.samples < 8 or args.tokens < 1 or
      args.numeric_tokens < 0 or args.numeric_tokens > args.tokens or
      args.timeout_s <= 0):
    parser.error("warmups, samples, and timeout must be positive")
  if args.phase_specialized and args.tokens != 2048:
    parser.error("the formal phase-specialized dispatch gate requires 2048 tokens")
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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
  rows = []
  if not path.is_file():
    return rows
  for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
    try:
      row = json.loads(line)
    except json.JSONDecodeError:
      continue
    if isinstance(row, dict):
      rows.append(row)
  return rows


def shim_audit(spec: tuple[Path, str, str, str, str]) -> dict[str, Any]:
  path, begin_marker, end_marker, macro, expected_body_sha256 = spec
  lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
  begin = [index for index, line in enumerate(lines)
           if line.rstrip("\r\n") == begin_marker]
  end = [index for index, line in enumerate(lines)
         if line.rstrip("\r\n") == end_marker]
  macros = [index for index, line in enumerate(lines)
            if line.rstrip("\r\n") == macro]
  ordered = (
      len(begin) == len(end) == len(macros) == 1 and
      begin[0] + 1 == macros[0] and macros[0] < end[0])
  body = (
      "".join(lines[macros[0] + 1:end[0]]).encode("utf-8")
      if ordered else b"")
  observed_body_sha256 = hashlib.sha256(body).hexdigest()
  return {
      "path": str(path.resolve()),
      "full_sha256": sha256(path),
      "body_sha256": observed_body_sha256,
      "expected_body_sha256": expected_body_sha256,
      "begin_count": len(begin),
      "end_count": len(end),
      "macro_count": len(macros),
      "markers_ordered": ordered,
      "pass": ordered and observed_body_sha256 == expected_body_sha256,
  }


def phase_plan(tokens: int, phase_specialized: bool) -> list[tuple[str, int]]:
  if phase_specialized:
    return [("prefill_a", tokens), ("decode", 1), ("prefill_b", tokens)]
  return [("single", tokens)]


def dispatch_expectation(name: str, tokens: int) -> tuple[tuple[int, ...],
                                                            tuple[int, ...]]:
  if tokens == 1:
    return DECODE_DISPATCH[name]
  global_2048, local = PREFILL_2048_DISPATCH[name]
  tile_n = 64 if local == (64, 2, 1) else 256
  n_groups = (tokens + tile_n - 1) // tile_n
  return (global_2048[0], n_groups * local[1], 1), local


def provider_audit(cases: list[dict[str, Any]],
                   trace_rows: list[dict[str, Any]]) -> dict[str, Any]:
  expected_keys = set()
  implementations = []
  for case in cases:
    expected_entry = f"iq36_fixed_fc{case['arity']}"
    for tokens in sorted(set(case.get("phase_sequence", []))):
      key = (expected_entry, int(case["k"]), tuple(case["widths"]), tokens)
      expected_keys.add(key)
      expected_global, expected_local = dispatch_expectation(
          str(case["name"]), tokens)
      expected_provider = (
          "decode" if tokens == 1 else
          ("prefill_wide" if sum(case["widths"]) > 4096
           else "prefill_small"))
      rows = [row for row in trace_rows
              if (str(row.get("entry")), int(row.get("k", -1)),
                  tuple(row.get("widths", [])), int(row.get("tokens", -1))) ==
              key]
      implementations.append({
          "case": case["name"],
          "tokens": tokens,
          "entry": expected_entry,
          "observations": rows,
          "expected_observation_count": 1,
          "expected_provider": expected_provider,
          "expected_global_size": list(expected_global),
          "expected_local_size": list(expected_local),
          "pass": (
              len(rows) == 1 and
              rows[0].get("provider") == expected_provider and
              tuple(rows[0].get("global", [])) == expected_global and
              tuple(rows[0].get("local", [])) == expected_local),
      })
  observed_keys = {
      (str(row.get("entry")), int(row.get("k", -1)),
       tuple(row.get("widths", [])), int(row.get("tokens", -1)))
      for row in trace_rows}
  extra_keys = sorted(observed_keys - expected_keys)
  return {
      "trace_rows": len(trace_rows),
      "expected_trace_rows": len(expected_keys),
      "extra_keys": [list(value) for value in extra_keys],
      "implementations": implementations,
      "pass": (
          bool(implementations) and
          all(row["pass"] for row in implementations) and
          not extra_keys and len(trace_rows) == len(expected_keys)),
  }


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
      lines = stat_path.with_name("status").read_text(
          encoding="utf-8", errors="replace").splitlines()
    except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
      continue
    processes += 1
    for line in lines:
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
  return ((flat[0::2] & np.uint8(0x0F)) |
          ((flat[1::2] & np.uint8(0x0F)) << np.uint8(4)))


def vector_metrics(reference: Any, actual: Any, np: Any) -> dict[str, Any]:
  reference = np.asarray(reference, dtype=np.float32).reshape(-1)
  actual = np.asarray(actual, dtype=np.float32).reshape(-1)
  delta = actual - reference
  denominator = float(np.linalg.norm(reference))
  cosine_denominator = float(
      np.linalg.norm(reference) * np.linalg.norm(actual))
  return {
      "values": int(reference.size),
      "finite": bool(np.isfinite(actual).all()),
      "max_abs_diff": float(np.max(np.abs(delta))),
      "relative_l2": float(np.linalg.norm(delta) /
                           (denominator if denominator else 1.0)),
      "cosine": float(np.dot(reference, actual) /
                      (cosine_denominator if cosine_denominator else 1.0)),
      "exact_rate": float(np.mean(reference == actual)),
  }


def make_case(name: str, widths: tuple[int, ...], k: int,
              tokens: int, numeric_tokens: int, graph: Any, ov: Any,
              np: Any) -> tuple[Any, Any, list[Any]]:
  groups = k // 64
  rng = np.random.default_rng(
      0xF1C00000 ^ k ^ sum(width << (index * 3)
                          for index, width in enumerate(widths)))
  activation = rng.uniform(-0.25, 0.25, (tokens, k)).astype(np.float16)
  parameter = ov.opset13.parameter(
      ov.PartialShape([1, -1, k]), ov.Type.f16, name="activation")
  inputs = [parameter.output(0)]
  references = []
  activation_groups = activation.astype(np.float32).reshape(
      tokens, groups, 64)
  for index, width in enumerate(widths):
    weight = rng.integers(
        0, 16, (width, groups, 64), dtype=np.uint8)
    scale = rng.uniform(
        0.0005, 0.003, (groups, width, 1)).astype(np.float16)
    zero_point = rng.integers(
        0, 16, (groups, width, 1), dtype=np.uint8)
    packed_weight = pack_u4(weight, np).reshape(1, 1, 1, -1)
    packed_zp = pack_u4(zero_point, np).reshape(1, 1, 1, -1)
    inputs.extend([
        ov.opset13.constant(packed_weight, dtype=ov.Type.u8).output(0),
        ov.opset13.constant(
            scale.reshape(1, 1, groups, width),
            dtype=ov.Type.f16).output(0),
        ov.opset13.constant(packed_zp, dtype=ov.Type.u8).output(0),
    ])
    dequant = (
        weight.astype(np.float32) -
        zero_point.transpose(1, 0, 2).astype(np.float32))
    dequant *= scale.transpose(1, 0, 2).astype(np.float32)
    reference = np.einsum(
        "tgs,wgs->tw", activation_groups[:numeric_tokens], dequant,
        dtype=np.float32, optimize=True)
    references.append(reference.astype(np.float16).astype(np.float32))
  global_x = sum((width + 63) // 64 for width in widths) * 32
  carrier = ov.opset13.constant(
      np.zeros((1, 1, 1, global_x), dtype=np.uint8))
  inputs.append(carrier.output(0))
  operation = graph.fixed_fc_custom_classes(ov)[len(widths)](
      inputs, widths, k)
  operation.set_friendly_name(f"iq36_fixed_fc_{name}")
  restored_outputs = [
      ov.opset13.convert(operation.output(index), ov.Type.f32).output(0)
      for index in range(len(widths))
  ]
  model = ov.Model(
      restored_outputs, [parameter], f"iq36_fixed_fc_{name}")
  return model, activation, references


def runtime_custom_rows(compiled: Any) -> list[dict[str, Any]]:
  rows = []
  for node in compiled.get_runtime_model().get_ordered_ops():
    info = {}
    for key, value in node.get_rt_info().items():
      try:
        info[str(key)] = value.value
      except Exception:
        info[str(key)] = str(value)
    if str(info.get("layerType")) == "CustomGPUPrimitive":
      rows.append({
          "name": node.get_friendly_name(),
          "layer_type": str(info.get("layerType")),
          "primitive_type": str(info.get("primitiveType")),
      })
  return rows


def custom_profile(request: Any) -> list[dict[str, Any]]:
  return [{
      "name": row.node_name,
      "type": row.node_type,
      "exec_type": row.exec_type,
      "status": str(row.status),
      "real_time_us": row.real_time.total_seconds() * 1_000_000.0,
  } for row in request.get_profiling_info()
          if row.node_type.startswith("IQ36FixedFC")]


def full_profile(request: Any) -> list[dict[str, Any]]:
  return [{
      "name": row.node_name,
      "type": row.node_type,
      "exec_type": row.exec_type,
      "status": str(row.status),
      "real_time_us": row.real_time.total_seconds() * 1_000_000.0,
  } for row in request.get_profiling_info()]


def run_case(core: Any, case: tuple[str, tuple[int, ...], int],
             graph: Any, ov: Any, np: Any, warmups: int,
             samples: int, tokens: int, numeric_tokens: int,
             phase_specialized: bool, stop_bytes: int,
             queue_throttle: str) -> dict[str, Any]:
  name, widths, k = case
  if meminfo()["MemAvailable"] < stop_bytes:
    raise RuntimeError(f"memory stop before {name}")
  model, activation, references = make_case(
      name, widths, k, tokens, numeric_tokens, graph, ov, np)
  operation = next(
      node for node in model.get_ordered_ops()
      if node.get_type_name().startswith("IQ36FixedFC"))
  input_types = [str(operation.get_input_element_type(index))
                 for index in range(operation.get_input_size())]
  custom_output_types = [
      operation.get_output_element_type(index).get_type_name()
      for index in range(operation.get_output_size())
  ]
  compile_started = time.perf_counter_ns()
  compile_config = {
      "PERFORMANCE_HINT": "LATENCY",
      "PERF_COUNT": True,
      "GPU_QUEUE_THROTTLE": queue_throttle,
  }
  compiled = core.compile_model(model, "GPU", compile_config)
  compile_ms = (time.perf_counter_ns() - compile_started) / 1_000_000.0
  request = compiled.create_infer_request()
  runtime = runtime_custom_rows(compiled)
  phases = []
  for label, phase_tokens in phase_plan(tokens, phase_specialized):
    if meminfo()["MemAvailable"] < stop_bytes:
      raise RuntimeError(f"memory stop before {name} {label}")
    feed = {
        compiled.input("activation"):
        activation[:phase_tokens].reshape(1, phase_tokens, k),
    }
    for _ in range(warmups):
      request.infer(feed, share_outputs=False)
    wall_samples = []
    kernel_samples = []
    outputs = None
    for _ in range(samples):
      started = time.perf_counter_ns()
      outputs = request.infer(feed, share_outputs=False)
      wall_samples.append((time.perf_counter_ns() - started) / 1_000.0)
      profile = custom_profile(request)
      if len(profile) == 1:
        kernel_samples.append(float(profile[0]["real_time_us"]))
    if outputs is None:
      raise RuntimeError("no inference output")
    checked_tokens = min(numeric_tokens, phase_tokens)
    numeric = []
    for index, reference in enumerate(references):
      actual = np.asarray(
          outputs[compiled.output(index)], dtype=np.float32).reshape(
              phase_tokens, widths[index])
      checked = actual[:checked_tokens]
      numeric.append({
          "index": index,
          "width": widths[index],
          "checked_tokens": checked_tokens,
          "shape": list(checked.shape),
          "full_shape": list(actual.shape),
          "all_values_finite": bool(np.isfinite(actual).all()),
          **vector_metrics(reference[:checked_tokens], checked, np),
      })
    complete_profile = full_profile(request)
    phases.append({
        "label": label,
        "tokens": phase_tokens,
        "numeric": numeric,
        "profile": custom_profile(request),
        "noncustom_executed_profile": [
            row for row in complete_profile
            if (not row["type"].startswith("IQ36FixedFC") and
                row["real_time_us"] > 0.0)
        ],
        "timing": {
            "warmups": warmups,
            "samples": samples,
            "wall_us_median": statistics.median(wall_samples),
            "wall_us_min": min(wall_samples),
            "kernel_us_median": (
                statistics.median(kernel_samples)
                if kernel_samples else None),
            "kernel_us_min": (
                min(kernel_samples) if kernel_samples else None),
            "kernel_us_samples": kernel_samples,
        },
    })
    del outputs
  result = {
      "name": name,
      "arity": len(widths),
      "widths": list(widths),
      "k": k,
      "tokens": tokens,
      "compile_ms": compile_ms,
      "compile_config": compile_config,
      "input_types": input_types,
      "custom_output_types": custom_output_types,
      "model_output_types": [
          compiled.output(index).element_type.get_type_name()
          for index in range(len(widths))
      ],
      "phase_sequence": [phase["tokens"] for phase in phases],
      "phases": phases,
      "numeric": [row for phase in phases for row in phase["numeric"]],
      "profile": phases[-1]["profile"],
      "noncustom_executed_profile": phases[-1][
          "noncustom_executed_profile"],
      "runtime": runtime,
      "timing": phases[-1]["timing"],
  }
  del request, compiled, model, references
  gc.collect()
  return result


def worker_main(config_path: Path) -> int:
  cfg = load_json(config_path)
  if Path(sys.prefix).resolve() != Path(cfg["openvino_python"]).parent.parent:
    raise RuntimeError(f"worker requires {cfg['openvino_python']}")
  import numpy as np
  import openvino as ov

  graph = load_module(GRAPH, "iq36_fixed_fc_graph_gate")
  stop_bytes = int(cfg["stop_bytes"])
  if meminfo()["MemAvailable"] < stop_bytes:
    raise RuntimeError("worker memory stop at start")
  source_model, source = graph.read_and_rewrite_locked_model(
      ov.Core(), ov, np)
  custom_ops = [node for node in source_model.get_ordered_ops()
                if node.get_type_name().startswith("IQ36FixedFC")]
  custom_u4_inputs = sum(
      node.get_input_element_type(index) == ov.Type.u4
      for node in custom_ops for index in range(node.get_input_size()))
  packed_u8_inputs = sum(
      node.get_input_element_type(index) == ov.Type.u8
      for node in custom_ops for index in range(node.get_input_size()))
  source_audit = {
      "summary": {key: source[key] for key in (
          "fixed_fc_rewrite_count", "fixed_fc_projection_count",
          "fixed_fc_group_counts", "fixed_fc_custom_counts",
          "fixed_fc_old_matmuls_remaining",
          "fixed_fc_f16_to_f32_restore_count")},
      "custom_u4_inputs": custom_u4_inputs,
      "packed_u8_inputs": packed_u8_inputs,
  }
  del source_model, custom_ops
  gc.collect()

  plugin = Path(cfg["plugin"])
  raw = Path(cfg["raw"])
  registry = raw / "candidate-plugins.xml"
  registry.write_text(
      "<ie><plugins><plugin name=\"GPU\" location="
      f"{quoteattr(str(plugin.resolve()))}/></plugins></ie>\n",
      encoding="utf-8")
  core = ov.Core(str(registry))
  config_before = str(core.get_property("GPU", "CONFIG_FILE"))
  core.set_property("GPU", {"CONFIG_FILE": cfg["custom_config"]})
  config_after = str(core.get_property("GPU", "CONFIG_FILE"))
  cases = []
  for case in COHORTS:
    cases.append(run_case(
        core, case, graph, ov, np, int(cfg["warmups"]),
        int(cfg["samples"]), int(cfg["tokens"]),
        int(cfg["numeric_tokens"]), bool(cfg["phase_specialized"]),
        stop_bytes, str(cfg["queue_throttle"])))
  write_json(Path(cfg["result"]), {
      "openvino_version": ov.get_version(),
      "plugin": str(plugin.resolve()),
      "plugin_sha256": sha256(plugin),
      "config_before": config_before,
      "config_after": config_after,
      "source_audit": source_audit,
      "cases": cases,
  })
  return 0


def git_state(output: Path) -> dict[str, Any]:
  commit = subprocess.run(
      ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
      capture_output=True, text=True).stdout.strip()
  rows = subprocess.run(
      ["git", "status", "--porcelain"], cwd=ROOT, check=True,
      capture_output=True, text=True).stdout.splitlines()
  output_relative = str(output.resolve().relative_to(ROOT))
  dirty = []
  for row in rows:
    path = row[3:]
    if path in ALLOWED_UNCOMMITTED or path.startswith(output_relative):
      continue
    dirty.append(row)
  return {"commit": commit, "dirty": bool(dirty), "dirty_paths": dirty,
          "allowed_uncommitted": sorted(ALLOWED_UNCOMMITTED)}


def check(name: str, passed: bool, **evidence: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **evidence}


def main() -> int:
  args = parse_args()
  if args.worker_config is not None:
    return worker_main(args.worker_config)
  output = args.output.resolve()
  raw = output / "raw"
  raw.mkdir(parents=True, exist_ok=False)
  required = (args.plugin, args.config, args.openvino_python,
              GRAPH, PATCH, DECODE_SHIM, PREFILL_WIDE_SHIM,
              PREFILL_SMALL_SHIM, KERNEL)
  missing = [str(path) for path in required if not path.is_file()]
  if missing:
    raise SystemExit("missing gate inputs: " + ", ".join(missing))
  start = meminfo()
  preflight_bytes = int(args.min_available_gib * 1024**3)
  stop_bytes = int(args.abort_below_available_gib * 1024**3)
  if int(start["MemAvailable"]) < preflight_bytes:
    raise RuntimeError("eight-GiB preflight did not clear")
  result_path = raw / "worker-result.json"
  config_path = raw / "worker-config.json"
  provider_trace_path = raw / "provider-trace.jsonl"
  write_json(config_path, {
      "openvino_python": str(args.openvino_python.absolute()),
      "plugin": str(args.plugin.resolve()),
      "custom_config": str(args.config.resolve()),
      "warmups": args.warmups,
      "samples": args.samples,
      "tokens": args.tokens,
      "numeric_tokens": (
          args.numeric_tokens if args.numeric_tokens else args.tokens),
      "phase_specialized": args.phase_specialized,
      "queue_throttle": args.queue_throttle,
      "stop_bytes": stop_bytes,
      "raw": str(raw),
      "result": str(result_path),
  })
  command = [str(args.openvino_python), str(Path(__file__).resolve()),
             "--worker-config", str(config_path)]
  environment = os.environ.copy()
  for key in ("OV_GPU_CONFIG_FILE", "LD_AUDIT", "LD_PRELOAD",
              "IQ36_FIXED_FC_PROVIDER_TRACE_PATH"):
    environment.pop(key, None)
  environment.update({
      "IQ36_FIXED_FC_PROVIDER_TRACE_PATH": str(
          provider_trace_path.resolve()),
      "NEO_CACHE_DIR": str((raw / "neo-cache").resolve()),
      "NEO_CACHE_MAX_SIZE": str(2 * 1024**3),
      "NEO_CACHE_PERSISTENT": "1",
  })
  captured_environment = (
      "IQ36_FIXED_FC_PROVIDER_TRACE_PATH",
      "NEO_CACHE_DIR", "NEO_CACHE_MAX_SIZE", "NEO_CACHE_PERSISTENT")
  write_json(raw / "worker-command.json", {
      "command": command,
      "environment": {key: environment[key] for key in captured_environment},
  })
  stdout_path = raw / "worker.stdout"
  stderr_path = raw / "worker.stderr"
  monitor = {
      "system_available_min_bytes": int(start["MemAvailable"]),
      "system_swap_used_peak_bytes": (
          int(start.get("SwapTotal", 0)) - int(start.get("SwapFree", 0))),
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
      if time.monotonic() - started > args.timeout_s:
        timed_out = True
        stop_group(process)
        break
      system = meminfo()
      group = process_group_memory(process.pid)
      available = int(system["MemAvailable"])
      swap_used = (int(system.get("SwapTotal", 0)) -
                   int(system.get("SwapFree", 0)))
      monitor["samples"] += 1
      monitor["system_available_min_bytes"] = min(
          int(monitor["system_available_min_bytes"]), available)
      monitor["system_swap_used_peak_bytes"] = max(
          int(monitor["system_swap_used_peak_bytes"]), swap_used)
      monitor["process_group_rss_peak_bytes"] = max(
          int(monitor["process_group_rss_peak_bytes"]), group["rss_bytes"])
      monitor["process_group_swap_peak_bytes"] = max(
          int(monitor["process_group_swap_peak_bytes"]), group["swap_bytes"])
      monitor["process_count_peak"] = max(
          int(monitor["process_count_peak"]), group["processes"])
      if available < stop_bytes:
        guard_tripped = True
        stop_group(process)
        break
      time.sleep(0.05)
    returncode = process.wait()
  elapsed_seconds = time.monotonic() - started
  stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")
  result = load_json(result_path) if result_path.is_file() else {}
  cases = result.get("cases", [])
  numeric_rows = [row for case in cases for row in case.get("numeric", [])]
  phase_rows = [row for case in cases for row in case.get("phases", [])]
  source = result.get("source_audit", {})
  source_summary = source.get("summary", {})
  git = git_state(output)
  shim_audits = [shim_audit(spec) for spec in SHIM_SPECS]
  provider_trace_rows = load_jsonl(provider_trace_path)
  provider_trace = provider_audit(cases, provider_trace_rows)
  expected_phase_sequence = (
      [2048, 1, 2048] if args.phase_specialized else [args.tokens])
  config_text = args.config.read_text(encoding="utf-8")
  patch_text = PATCH.read_text(encoding="utf-8")
  reverse_patch = subprocess.run(
      ["git", "apply", "--reverse", "--check", str(PATCH.resolve())],
      cwd=OV_SOURCE, check=False, capture_output=True, text=True,
      encoding="utf-8", errors="replace")
  oom_observed = (
      not guard_tripped and
      (returncode in (-9, 137) or "out of memory" in stderr_text.lower()))
  checks = [
      check("repository_clean_at_gate", not git["dirty"], git=git),
      check("one_isolated_worker_completed",
            returncode == 0 and not timed_out and not guard_tripped,
            returncode=returncode, timed_out=timed_out,
            guard_tripped=guard_tripped),
      check("complete_locked_graph_rewrite_is_exact",
            source_summary.get("fixed_fc_rewrite_count") == 160 and
            source_summary.get("fixed_fc_projection_count") == 390 and
            source_summary.get("fixed_fc_f16_to_f32_restore_count") == 390 and
            source_summary.get("fixed_fc_old_matmuls_remaining") == []),
      check("custom_graph_consumes_packed_u8_not_logical_u4",
            source.get("custom_u4_inputs") == 0 and
            source.get("packed_u8_inputs") == 940,
            source_audit=source),
      check("all_five_exact_cohorts_compile_once_and_execute_all_phases",
            [case.get("name") for case in cases] ==
            [case[0] for case in COHORTS] and
            all(case.get("phase_sequence") == expected_phase_sequence
                for case in cases) and
            len(phase_rows) == len(COHORTS) * len(expected_phase_sequence) and
            all(len(case.get("runtime", [])) == 1 and
                all(len(phase.get("profile", [])) == 1
                    for phase in case.get("phases", []))
                for case in cases)),
      check("f16_internal_outputs_restore_f32_without_convert_dispatch",
            all(case.get("custom_output_types") == ["f16"] * case["arity"] and
                case.get("model_output_types") == ["f32"] * case["arity"] and
                all(all(row.get("type") == "Result" and
                        str(row.get("exec_type", "")).startswith(
                            "reorder_data")
                        for row in phase.get(
                            "noncustom_executed_profile", []))
                    for phase in case.get("phases", []))
                for case in cases)),
      check("all_outputs_are_tight_against_independent_cpu_oracle",
            len(numeric_rows) == 13 * len(expected_phase_sequence) and all(
                row.get("finite") is True and
                row.get("all_values_finite") is True and
                float(row.get("cosine", -1.0)) >= 0.999 and
                float(row.get("relative_l2", math.inf)) <= 0.001 and
                float(row.get("max_abs_diff", math.inf)) <= 0.00025
                for row in numeric_rows),
            worst={
                "max_abs_diff": max(
                    (row.get("max_abs_diff", 0.0) for row in numeric_rows),
                    default=None),
                "relative_l2": max(
                    (row.get("relative_l2", 0.0) for row in numeric_rows),
                    default=None),
                "cosine": min(
                    (row.get("cosine", 1.0) for row in numeric_rows),
                    default=None),
            }),
      check("decode_wide_and_small_microkernel_bodies_are_exact",
            all(row["pass"] for row in shim_audits),
            shims=shim_audits),
      check("phase_provider_sources_and_durable_patch_are_present",
            all(config_text.count(f'filename="{path.name}"') == 3
                for path in (
                    DECODE_SHIM, PREFILL_WIDE_SHIM, PREFILL_SMALL_SHIM)) and
            all(token in patch_text for token in (
                "phase_specialized_fixed_fc", "fixed_fc_prefill",
                "fixed_fc_wide", "IQ36_FIXED_FC_DECODE_BEGIN",
                "IQ36_FIXED_FC_PREFILL_WIDE_BEGIN",
                "IQ36_FIXED_FC_PREFILL_SMALL_BEGIN",
                "IQ36_FIXED_FC_PROVIDER_TRACE_PATH")) and
            reverse_patch.returncode == 0,
            config_source_counts={
                path.name: config_text.count(f'filename="{path.name}"')
                for path in (
                    DECODE_SHIM, PREFILL_WIDE_SHIM, PREFILL_SMALL_SHIM)},
            reverse_patch_returncode=reverse_patch.returncode,
            reverse_patch_stderr=reverse_patch.stderr),
      check("exact_decode_and_prefill_provider_implementations_observed",
            provider_trace["pass"], provider_trace=provider_trace),
      check("memory_guards_held_without_oom",
            not oom_observed and
            int(monitor["system_available_min_bytes"]) >= stop_bytes,
            monitor=monitor, oom_observed=oom_observed,
            note="swap is pressure telemetry only"),
  ]
  passed = all(row["pass"] for row in checks)
  verdict = (
      "admit_phase_specialized_fixed_fc_provider_source_cut" if passed else
      "phase_specialized_fixed_fc_provider_source_cut_not_admitted")
  metrics = {
      "schema": SCHEMA,
      "created_at": datetime.now(timezone.utc).isoformat(),
      "workstream": WS,
      "verdict": verdict,
      "required_checks_passed": passed,
      "git": git,
      "plugin": {"path": str(args.plugin.resolve()),
                 "sha256": sha256(args.plugin)},
      "custom_config": {"path": str(args.config.resolve()),
                        "sha256": sha256(args.config)},
      "custom_kernel": {"path": str(KERNEL.resolve()),
                        "sha256": sha256(KERNEL)},
      "microkernel_shims": shim_audits,
      "provider_trace": provider_trace,
      "worker": {"command": command, "returncode": returncode,
                 "elapsed_seconds": elapsed_seconds,
                 "timed_out": timed_out,
                 "memory_guard_tripped": guard_tripped,
                 "monitor": monitor, "oom_observed": oom_observed},
      "result": result,
      "checks": checks,
      "next_action": (
          "Run the locked 2k teacher-forced full-graph correctness gate."
          if passed else
          "Resolve the first failed source, dispatch, or numeric check."),
  }
  write_json(output / "metrics.json", metrics)
  write_json(output / "manifest.json", {
      "schema": SCHEMA, "workstream": WS,
      "required_checks_passed": passed, "metrics": "metrics.json",
      "raw": "raw/",
  })
  print(json.dumps({
      "output": str(output), "verdict": verdict,
      "required_checks_passed": passed,
      "failed_checks": [row["name"] for row in checks if not row["pass"]],
      "kernel_medians_us": {
          case["name"]: {
              phase["label"]: phase["timing"]["kernel_us_median"]
              for phase in case["phases"]}
          for case in cases},
  }, indent=2, sort_keys=True))
  return 0 if passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
