#!/usr/bin/env python3
"""Compare all 40 real router/shared groups with stock OpenVINO FC.

The stock and candidate models retain the locked U4 weights but cut the rest of
the language model at one dynamic activation parameter per layer.  Separate
serial GPU workers compile and execute 160 stock FC outputs versus 40 custom
four-output groups.  This gate is deliberately decode-only: it verifies that
stock DynamicQuantize is optimized out at T=1 and makes stock's static oneDNN
``jit:gemm:any`` implementation, rather than a CPU oracle or the unused OpenCL
fallback, the exact numeric target.  The optional phase-hybrid mode keeps that
stock branch for T=1 and selects the admitted custom provider only for T>1.
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
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from xml.sax.saxutils import quoteattr


ROOT = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-openvino-fixed-fc-decode-equivalence-gate-v2"
OV_PYTHON = Path("/home/intel/ov/openvino_env/bin/python")
STOCK_PLUGIN = Path(
    "/home/intel/ov/openvino_env/lib/python3.12/site-packages/openvino/"
    "libs/libopenvino_intel_gpu_plugin.so")
CANDIDATE_PLUGIN = Path(
    "/home/intel/intel-qwen36-r0/output/openvino-90214e-l0-gpu/"
    "bin/intel64/Release/libopenvino_intel_gpu_plugin.so")
CONFIG = ROOT / "engine/openvino/custom/iq36_hot_attention_gqa.xml"
GRAPH = ROOT / "tools/intel_qwen36_openvino_fixed_fc.py"
CAPTURE = ROOT / (
    "output/openvino-fc-boundary-capture-20260715Tseq1227-"
    "layer0-qkv-2k-o1-dirtyZ/raw/capture/dispatch000-arg1-before.bin")
EXPECTED_CAPTURE_SHA256 = (
    "5916d74c73811c7ad0bd54f6610842329ab570ed05e191a8875f55081be46f4c")
DQ_SOURCE = Path(
    "/home/intel/intel-qwen36-r0/source/openvino-90214e5be05/"
    "src/plugins/intel_gpu/src/graph/dynamic_quantize.cpp")
ONEDNN_FC_SOURCE = Path(
    "/home/intel/intel-qwen36-r0/source/openvino-90214e5be05/"
    "src/plugins/intel_gpu/src/graph/impls/onednn/"
    "fully_connected_onednn.cpp")
ONEDNN_JIT_SOURCE = Path(
    "/home/intel/intel-qwen36-r0/source/openvino-90214e5be05/"
    "src/plugins/intel_gpu/thirdparty/onednn_gpu/src/gpu/intel/gemm/jit.hpp")
OV_SOURCE = Path(
    "/home/intel/intel-qwen36-r0/source/openvino-90214e5be05")
PHASE_PROVIDER_PATCH = (
    ROOT / "engine/openvino/iq36-fixed-fc-phase-provider.patch")
PROJECTION_LABELS = (
    "shared_expert_gate", "shared_expert_gate_proj",
    "shared_expert_up_proj", "router")
EXPECTED_WIDTHS = (1, 512, 512, 256)
ALLOWED_UNCOMMITTED = {
    "engine/openvino/custom/iq36_fixed_fc_microkernel_shim.cl",
    "engine/openvino/custom/iq36_fixed_fc_multi_output.cl",
    "engine/openvino/custom/iq36_fixed_fc_prefill_small_microkernel_shim.cl",
    "engine/openvino/custom/iq36_fixed_fc_prefill_wide_microkernel_shim.cl",
    "engine/openvino/custom/iq36_hot_attention_gqa.xml",
    "engine/openvino/iq36-simplegpu-microkernel-fusion.patch",
    "tools/intel-qwen36-openvino-fixed-fc-decode-equivalence-gate.py",
}


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path)
  parser.add_argument("--openvino-python", type=Path, default=OV_PYTHON)
  parser.add_argument("--stock-plugin", type=Path, default=STOCK_PLUGIN)
  parser.add_argument("--candidate-plugin", type=Path,
                      default=CANDIDATE_PLUGIN)
  parser.add_argument("--config", type=Path, default=CONFIG)
  parser.add_argument("--warmups", type=int, default=4)
  parser.add_argument("--samples", type=int, default=9)
  parser.add_argument("--timeout-s", type=float, default=600.0)
  parser.add_argument("--min-available-gib", type=float, default=8.0)
  parser.add_argument("--abort-below-available-gib", type=float, default=4.0)
  parser.add_argument("--stock-jit-dump", action="store_true")
  parser.add_argument("--phase-hybrid", action="store_true")
  parser.add_argument(
      "--manager-direct", action="store_true",
      help=("run the stock FC graph on the candidate plugin, selecting "
            "oneDNN at T=1 and the plugin-internal manager at T>1"))
  parser.add_argument("--hybrid-prefill-tokens", type=int, default=128)
  parser.add_argument("--worker-config", type=Path, help=argparse.SUPPRESS)
  args = parser.parse_args()
  if args.worker_config is None and args.output is None:
    parser.error("--output is required")
  if args.warmups < 1 or args.samples < 5 or args.timeout_s <= 0:
    parser.error("warmups, samples, and timeout must be positive")
  if args.abort_below_available_gib > args.min_available_gib:
    parser.error("abort threshold must not exceed preflight threshold")
  if args.hybrid_prefill_tokens < 2:
    parser.error("hybrid prefill tokens must be at least two")
  if args.phase_hybrid and args.manager_direct:
    parser.error("--phase-hybrid and --manager-direct are mutually exclusive")
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


def source_census(model: Any) -> dict[str, int]:
  counts = Counter(node.get_type_name() for node in model.get_ordered_ops())
  return {key: int(value) for key, value in sorted(counts.items())}


def build_custom_group(group: dict[str, Any], activation: Any, graph: Any,
                       classes: dict[int, type], ov: Any, np: Any,
                       name: str) -> Any:
  activation_f16 = ov.opset13.convert(activation, ov.Type.f16)
  inputs = [activation_f16.output(0)]
  for projection in group["projections"]:
    inputs.extend([
        graph._packed_weight(ov, np, projection).output(0),
        graph._group_major_scale(ov, np, projection).output(0),
        graph._group_major_zero_point(ov, np, projection).output(0),
    ])
  work_groups = sum(
      (int(width) + graph.WG_TILE_M - 1) // graph.WG_TILE_M
      for width in group["widths"])
  carrier = ov.opset13.constant(
      np.zeros((1, 1, 1, work_groups * graph.LOCAL_X), dtype=np.uint8))
  inputs.append(carrier.output(0))
  operation = classes[len(group["projections"])](
      inputs, tuple(group["widths"]), int(group["k"]))
  operation.set_friendly_name(name)
  return operation


def build_hybrid_group(group: dict[str, Any], parameter: Any,
                       decode_condition: Any, graph: Any,
                       classes: dict[int, type], ov: Any, np: Any,
                       ) -> tuple[list[Any], dict[str, Any]]:
  layer = int(group["layer"])
  stock_parameter = ov.opset13.parameter(
      [1, -1, 2048], ov.Type.f32,
      name=f"iq36_hybrid_stock_layer{layer}_input")
  stock_results = []
  for projection in group["projections"]:
    projection["matmul"].input(0).replace_source_output(
        stock_parameter.output(0))
    stock_f16 = ov.opset13.convert(
        projection["matmul"].output(0), ov.Type.f16)
    stock_results.append(ov.opset13.result(stock_f16.output(0)))
  stock_body = ov.Model(
      stock_results, [stock_parameter],
      f"iq36_hybrid_stock_layer{layer}_body")

  custom_parameter = ov.opset13.parameter(
      [1, -1, 2048], ov.Type.f32,
      name=f"iq36_hybrid_custom_layer{layer}_input")
  custom_operation = build_custom_group(
      group, custom_parameter.output(0), graph, classes, ov, np,
      f"iq36_hybrid_custom_layer{layer}")
  custom_results = [
      ov.opset13.result(custom_operation.output(index))
      for index in range(4)
  ]
  custom_body = ov.Model(
      custom_results, [custom_parameter],
      f"iq36_hybrid_custom_layer{layer}_body")

  selector = ov.opset13.if_op(decode_condition)
  selector.set_then_body(stock_body)
  selector.set_else_body(custom_body)
  selector.set_input(
      parameter.output(0), stock_parameter, custom_parameter)
  values = [
      selector.set_output(stock_result, custom_result)
      for stock_result, custom_result in zip(stock_results, custom_results)
  ]
  selector.set_friendly_name(f"iq36_fixed_fc_phase_hybrid_layer{layer}")
  return values, {
      "layer": layer,
      "condition": "tokens == 1",
      "decode_body": stock_body.get_friendly_name(),
      "prefill_body": custom_body.get_friendly_name(),
      "selector": selector.get_friendly_name(),
  }


def build_decode_model(candidate: bool, graph: Any, ov: Any, np: Any,
                       phase_hybrid: bool = False,
                       ) -> tuple[Any, dict[str, Any]]:
  source = ov.Core().read_model(str(graph.MODEL_XML))
  groups = [group for group in graph.discover_fixed_fc_groups(source, ov)
            if group["cohort"] == "router_shared_input"]
  groups.sort(key=lambda row: int(row["layer"]))
  if (len(groups) != 40 or
      [int(group["layer"]) for group in groups] != list(range(40)) or
      any(tuple(group["widths"]) != EXPECTED_WIDTHS for group in groups)):
    raise ValueError("locked router/shared groups differ")
  rewrite = None
  rewrite_rows = {}
  if candidate and not phase_hybrid:
    rewrite = graph.rewrite_fixed_fc(
        source, ov, np, cohorts={"router_shared_input"})
    rewrite_rows = {
        (int(row["layer"]), str(row["cohort"])): row
        for row in rewrite["fixed_fc_rows"]}
  nodes = {node.get_friendly_name(): node
           for node in source.get_ordered_ops()}
  classes = graph.fixed_fc_custom_classes(ov) if phase_hybrid else {}
  parameters = []
  outputs = []
  output_rows = []
  hybrid_rows = []
  shared_decode_condition = None
  for group in groups:
    layer = int(group["layer"])
    parameter = ov.opset13.parameter(
        [1, -1, 2048],
        ov.Type.f32,
        name=f"activation_layer{layer}")
    parameters.append(parameter)
    if phase_hybrid:
      if shared_decode_condition is None:
        shape = ov.opset13.shape_of(parameter, "i64")
        tokens = ov.opset13.gather(
            shape,
            ov.opset13.constant(np.array(1, dtype=np.int64)),
            ov.opset13.constant(np.array(0, dtype=np.int64)))
        shared_decode_condition = ov.opset13.equal(
            tokens, ov.opset13.constant(np.array(1, dtype=np.int64))).output(0)
      values, hybrid_row = build_hybrid_group(
          group, parameter, shared_decode_condition, graph, classes, ov, np)
      hybrid_rows.append(hybrid_row)
    elif candidate:
      operation_name = rewrite_rows[
          (layer, "router_shared_input")]["operation"]
      operation = nodes[operation_name]
      conversion = operation.input_value(0).get_node()
      if conversion.get_type_name() != "Convert":
        raise ValueError(f"layer {layer}: candidate input Convert missing")
      conversion.input(0).replace_source_output(parameter.output(0))
      values = [operation.output(index) for index in range(4)]
    else:
      values = []
      for projection in group["projections"]:
        projection["matmul"].input(0).replace_source_output(
            parameter.output(0))
        # A direct F32 Result widens the stock FC at this artificial cut.  Keep
        # the model Result itself F16 so the plugin cannot fuse away the same
        # inference-precision boundary present at the real graph consumer.
        values.append(ov.opset13.convert(
            projection["matmul"].output(0), ov.Type.f16).output(0))
    for index, value in enumerate(values):
      outputs.append(value)
      output_rows.append({
          "index": len(outputs) - 1,
          "layer": layer,
          "projection_index": index,
          "projection": PROJECTION_LABELS[index],
          "source_name": str(group["names"][index]),
          "width": int(group["widths"][index]),
      })
  model = ov.Model(
      outputs, parameters,
      "iq36_router_shared_decode_candidate" if candidate else
      "iq36_router_shared_decode_stock")
  model.validate_nodes_and_infer_types()
  metadata = {
      "candidate": candidate,
      "phase_hybrid": phase_hybrid,
      "input_type": "f32",
      "operation_count": len(model.get_ordered_ops()),
      "operation_census": source_census(model),
      "input_count": len(model.inputs),
      "output_count": len(model.outputs),
      "output_element_types": [str(output.get_element_type())
                               for output in model.outputs],
      "output_rows": output_rows,
      "rewrite": ({key: rewrite[key] for key in (
          "fixed_fc_rewrite_count", "fixed_fc_projection_count",
          "fixed_fc_group_counts", "fixed_fc_custom_counts",
          "fixed_fc_old_matmuls_remaining",
          "fixed_fc_f16_to_f32_restore_count")} if rewrite else None),
      "hybrid_rows": hybrid_rows,
  }
  return model, metadata


def make_inputs(np: Any, prefill_tokens: int = 0,
                manager_transition: bool = False,
                ) -> tuple[list[tuple[str, dict[str, Any]]], str]:
  captured = np.fromfile(CAPTURE, dtype="<f2")
  if captured.size != 2048:
    raise ValueError("locked activation capture has wrong size")
  base = captured.astype(np.float32)
  cases = []
  for label in ("rotated_real", "signed_real"):
    feeds = {}
    digest = hashlib.sha256()
    for layer in range(40):
      values = np.roll(base, (layer * 47) % base.size).copy()
      if label == "signed_real":
        signs = np.where(
            ((np.arange(base.size) + layer) % 7) < 3, -1.0, 1.0)
        values = values * signs.astype(np.float32)
      values = np.ascontiguousarray(
          values.reshape(1, 1, 2048).astype(np.float32))
      feeds[f"activation_layer{layer}"] = values
      digest.update(values.astype("<f4", copy=False).tobytes())
    cases.append((label, feeds))
  if prefill_tokens:
    feeds = {}
    factors = (
        1.0 +
        ((np.arange(prefill_tokens, dtype=np.float32) % 17.0) - 8.0) *
        np.float32(0.001953125))
    for layer in range(40):
      row = np.roll(base, (layer * 47) % base.size)
      values = np.ascontiguousarray(
          factors[:, None] * row[None, :], dtype=np.float32).reshape(
              1, prefill_tokens, 2048)
      feeds[f"activation_layer{layer}"] = values
    cases.append((f"prefill_t{prefill_tokens}_real", feeds))
  if manager_transition and prefill_tokens:
    # Exercise the same compiled dynamic graph across both implementation
    # boundaries and then prove the oneDNN T1 path is restored.
    cases = [cases[0], cases[-1], cases[1]]
  combined = hashlib.sha256()
  for _, feeds in cases:
    for name in sorted(feeds):
      combined.update(feeds[name].astype("<f4", copy=False).tobytes())
  return cases, combined.hexdigest()


def runtime_rows(compiled: Any) -> list[dict[str, Any]]:
  rows = []
  for node in compiled.get_runtime_model().get_ordered_ops():
    info = {}
    for key, value in node.get_rt_info().items():
      try:
        info[str(key)] = value.value
      except Exception:
        info[str(key)] = str(value)
    layer_type = str(info.get("layerType", node.get_type_name()))
    if layer_type not in (
        "CustomGPUPrimitive", "DynamicQuantize", "FullyConnected"):
      continue
    rows.append({
        "name": node.get_friendly_name(),
        "layer_type": layer_type,
        "primitive_type": str(info.get("primitiveType")),
        "runtime_precision": str(info.get("runtimePrecision")),
        "output_precisions": str(info.get("outputPrecisions")),
        "rt_info": {key: str(value) for key, value in sorted(info.items())},
    })
  return rows


def profile_rows(request: Any) -> list[dict[str, Any]]:
  return [{
      "name": row.node_name,
      "type": row.node_type,
      "exec_type": row.exec_type,
      "status": str(row.status),
      "real_time_us": row.real_time.total_seconds() * 1_000_000.0,
  } for row in request.get_profiling_info()]


def worker_main(config_path: Path) -> int:
  cfg = load_json(config_path)
  if Path(sys.prefix).resolve() != Path(cfg["openvino_python"]).parent.parent:
    raise RuntimeError(f"worker requires {cfg['openvino_python']}")
  import numpy as np
  import openvino as ov

  candidate_runtime = cfg["mode"] != "stock"
  candidate_graph = cfg["mode"] not in ("stock", "manager")
  graph = load_module(GRAPH, f"iq36_decode_equivalence_{cfg['mode']}")
  if meminfo()["MemAvailable"] < int(cfg["stop_bytes"]):
    raise RuntimeError("worker memory stop at start")
  phase_hybrid_run = bool(cfg.get("include_prefill", False))
  phase_hybrid = cfg["mode"] == "hybrid"
  model, source = build_decode_model(
      candidate_graph, graph, ov, np, phase_hybrid=phase_hybrid)
  raw = Path(cfg["raw"])
  plugin = Path(cfg["plugin"]).resolve()
  registry_override = candidate_runtime or plugin != STOCK_PLUGIN.resolve()
  if registry_override:
    registry = raw / f"{cfg['mode']}-plugins.xml"
    registry.write_text(
        "<ie><plugins><plugin name=\"GPU\" location="
        f"{quoteattr(str(plugin))}/></plugins></ie>\n",
        encoding="utf-8")
    core = ov.Core(str(registry))
  else:
    core = ov.Core()
  config_before = str(core.get_property("GPU", "CONFIG_FILE"))
  if candidate_runtime:
    core.set_property("GPU", {"CONFIG_FILE": cfg["custom_config"]})
    config_after = str(core.get_property("GPU", "CONFIG_FILE"))
  else:
    config_after = config_before
  compile_config = {
      "DYNAMIC_QUANTIZATION_GROUP_SIZE": 256,
      "PERFORMANCE_HINT": "LATENCY",
      "PERF_COUNT": True,
  }
  compile_started = time.perf_counter_ns()
  compiled = core.compile_model(model, "GPU", compile_config)
  compile_ms = (time.perf_counter_ns() - compile_started) / 1_000_000.0
  request = compiled.create_infer_request()
  cases, input_sha256 = make_inputs(
      np, int(cfg.get("hybrid_prefill_tokens", 0))
      if phase_hybrid_run else 0,
      manager_transition=bool(cfg.get("manager_direct", False)))
  arrays = {}
  case_rows = []
  for case_index, (label, values) in enumerate(cases):
    feed = {compiled.input(name): value for name, value in values.items()}
    for _ in range(int(cfg["warmups"])):
      request.infer(feed, share_outputs=False)
    wall_samples = []
    outputs = None
    for _ in range(int(cfg["samples"])):
      started = time.perf_counter_ns()
      outputs = request.infer(feed, share_outputs=False)
      wall_samples.append((time.perf_counter_ns() - started) / 1_000.0)
    if outputs is None:
      raise RuntimeError("worker produced no output")
    for output_index in range(len(source["output_rows"])):
      arrays[f"case{case_index:02d}_output{output_index:03d}"] = (
          np.asarray(outputs[compiled.output(output_index)],
                     dtype=np.float32))
    profile = profile_rows(request)
    case_rows.append({
        "index": case_index,
        "label": label,
        "wall_us_samples": wall_samples,
        "wall_us_median": statistics.median(wall_samples),
        "profile": profile,
        "profile_sum_us": sum(row["real_time_us"] for row in profile),
    })
    del outputs
  np.savez(raw / "outputs.npz", **arrays)
  write_json(Path(cfg["result"]), {
      "mode": cfg["mode"],
      "openvino_version": ov.get_version(),
      "plugin": str(Path(cfg["plugin"]).resolve()),
      "plugin_sha256": sha256(Path(cfg["plugin"])),
      "registry_override": registry_override,
      "config_before": config_before,
      "config_after": config_after,
      "compile_config": compile_config,
      "compile_ms": compile_ms,
      "source": source,
      "runtime_rows": runtime_rows(compiled),
      "input_sha256": input_sha256,
      "cases": case_rows,
  })
  del request, compiled, model, arrays
  gc.collect()
  return 0


def run_worker(label: str, cfg: dict[str, Any], args: argparse.Namespace,
               raw: Path, stop_bytes: int) -> dict[str, Any]:
  worker_raw = raw / label
  worker_raw.mkdir()
  cfg = {**cfg, "raw": str(worker_raw),
         "result": str(worker_raw / "result.json")}
  config_path = worker_raw / "worker-config.json"
  write_json(config_path, cfg)
  command = [str(args.openvino_python), str(Path(__file__).resolve()),
             "--worker-config", str(config_path)]
  environment = os.environ.copy()
  environment.pop("OV_GPU_CONFIG_FILE", None)
  environment.pop("IQ36_FIXED_FC_MANAGER_SCOPE", None)
  environment.update({
      "NEO_CACHE_DIR": str(worker_raw / "neo-cache"),
      "NEO_CACHE_MAX_SIZE": str(2 * 1024**3),
      "NEO_CACHE_PERSISTENT": "1",
  })
  if cfg["mode"] == "stock" and cfg.get("stock_jit_dump", False):
    environment.update({"ONEDNN_JIT_DUMP": "1", "DNNL_JIT_DUMP": "1"})
  if cfg["mode"] == "manager":
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
  with (worker_raw / "worker.stdout").open("w", encoding="utf-8") as out, \
       (worker_raw / "worker.stderr").open("w", encoding="utf-8") as err:
    process = subprocess.Popen(
        command, cwd=worker_raw if cfg.get("stock_jit_dump", False) else ROOT,
        env=environment, stdout=out, stderr=err,
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
      (returncode in (-9, 137) or "out of memory" in stderr_text.lower()))
  result_path = worker_raw / "result.json"
  return {
      "label": label,
      "command": command,
      "returncode": returncode,
      "started_monotonic": started,
      "finished_monotonic": time.monotonic(),
      "elapsed_seconds": time.monotonic() - started,
      "timed_out": timed_out,
      "memory_guard_tripped": guard_tripped,
      "oom_observed": oom_observed,
      "monitor": monitor,
      "result": load_json(result_path) if result_path.is_file() else {},
      "outputs": str(worker_raw / "outputs.npz"),
  }


def vector_metrics(reference: Any, actual: Any, np: Any) -> dict[str, Any]:
  reference = np.asarray(reference, dtype=np.float32).reshape(-1)
  actual = np.asarray(actual, dtype=np.float32).reshape(-1)
  delta = actual - reference
  denominator = float(np.linalg.norm(reference.astype(np.float64)))
  cosine_denominator = float(
      np.linalg.norm(reference.astype(np.float64)) *
      np.linalg.norm(actual.astype(np.float64)))
  return {
      "values": int(reference.size),
      "finite": bool(np.isfinite(actual).all()),
      "max_abs_diff": float(np.max(np.abs(delta))),
      "relative_l2": float(
          np.linalg.norm(delta.astype(np.float64)) /
          (denominator if denominator else 1.0)),
      "cosine": float(
          np.dot(reference.astype(np.float64), actual.astype(np.float64)) /
          (cosine_denominator if cosine_denominator else 1.0)),
      "exact_rate": float(np.mean(reference == actual)),
  }


def git_state(output: Path) -> dict[str, Any]:
  commit = subprocess.run(
      ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
      capture_output=True, text=True).stdout.strip()
  rows = subprocess.run(
      ["git", "status", "--porcelain"], cwd=ROOT, check=True,
      capture_output=True, text=True).stdout.splitlines()
  output_relative = str(output.resolve().relative_to(ROOT))
  dirty = [row for row in rows
           if row[3:] not in ALLOWED_UNCOMMITTED and
           not row[3:].startswith(output_relative)]
  return {"commit": commit, "dirty": bool(dirty), "dirty_paths": dirty,
          "allowed_uncommitted": sorted(ALLOWED_UNCOMMITTED)}


def check(name: str, passed: bool, **evidence: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **evidence}


def main() -> int:
  args = parse_args()
  if args.worker_config is not None:
    return worker_main(args.worker_config)
  import numpy as np

  output = args.output.resolve()
  raw = output / "raw"
  raw.mkdir(parents=True, exist_ok=False)
  required = (args.openvino_python, args.stock_plugin,
              args.candidate_plugin, args.config, GRAPH, CAPTURE,
              DQ_SOURCE, ONEDNN_FC_SOURCE, ONEDNN_JIT_SOURCE,
              PHASE_PROVIDER_PATCH)
  missing = [str(path) for path in required if not path.is_file()]
  if missing:
    raise SystemExit("missing gate inputs: " + ", ".join(missing))
  if sha256(CAPTURE) != EXPECTED_CAPTURE_SHA256:
    raise RuntimeError("locked activation capture hash differs")
  start = meminfo()
  preflight_bytes = int(args.min_available_gib * 1024**3)
  stop_bytes = int(args.abort_below_available_gib * 1024**3)
  if start["MemAvailable"] < preflight_bytes:
    raise RuntimeError("eight-GiB preflight did not clear")
  common = {
      "openvino_python": str(args.openvino_python.absolute()),
      "custom_config": str(args.config.resolve()),
      "warmups": args.warmups,
      "samples": args.samples,
      "stop_bytes": stop_bytes,
      "stock_jit_dump": args.stock_jit_dump,
      "include_prefill": args.phase_hybrid or args.manager_direct,
      "hybrid_prefill_tokens": args.hybrid_prefill_tokens,
      "manager_direct": args.manager_direct,
  }
  stock = run_worker(
      "stock", {**common, "mode": "stock",
                "plugin": str(args.stock_plugin.resolve())},
      args, raw, stop_bytes)
  custom = (run_worker(
      "custom", {**common, "mode": "custom",
                 "plugin": str(args.candidate_plugin.resolve())},
      args, raw, stop_bytes)
      if args.phase_hybrid or args.manager_direct else None)
  candidate = run_worker(
      "candidate", {**common,
                    "mode": ("manager" if args.manager_direct else
                             "hybrid" if args.phase_hybrid else "candidate"),
                    "plugin": str(args.candidate_plugin.resolve())},
      args, raw, stop_bytes)
  workers = [stock] + ([custom] if custom is not None else []) + [candidate]
  worker_ok = all(
      row["returncode"] == 0 and not row["timed_out"] and
      not row["memory_guard_tripped"] and not row["oom_observed"]
      for row in workers)
  comparisons = []
  output_rows = stock["result"].get("source", {}).get("output_rows", [])
  if worker_ok:
    with np.load(stock["outputs"]) as stock_outputs, \
         np.load(candidate["outputs"]) as candidate_outputs:
      for case_index, case in enumerate(
          stock["result"].get("cases", [])):
        for row in output_rows:
          key = f"case{case_index:02d}_output{int(row['index']):03d}"
          tokens = int(stock_outputs[key].size // int(row["width"]))
          comparisons.append({
              "case": case["label"], "tokens": tokens, **row,
              **vector_metrics(stock_outputs[key], candidate_outputs[key], np),
          })
  provider_comparisons = []
  if worker_ok and custom is not None:
    with np.load(custom["outputs"]) as custom_outputs, \
         np.load(candidate["outputs"]) as candidate_outputs:
      for case_index, case in enumerate(
          custom["result"].get("cases", [])):
        for row in output_rows:
          key = f"case{case_index:02d}_output{int(row['index']):03d}"
          tokens = int(custom_outputs[key].size // int(row["width"]))
          if tokens <= 1:
            continue
          provider_comparisons.append({
              "case": case["label"], "tokens": tokens, **row,
              **vector_metrics(
                  custom_outputs[key], candidate_outputs[key], np),
          })
  by_projection = {}
  grouped = defaultdict(list)
  for row in comparisons:
    grouped[row["projection"]].append(row)
  for projection, rows in grouped.items():
    by_projection[projection] = {
        "rows": len(rows),
        "min_exact_rate": min(row["exact_rate"] for row in rows),
        "worst_relative_l2": max(row["relative_l2"] for row in rows),
        "worst_max_abs_diff": max(row["max_abs_diff"] for row in rows),
        "min_cosine": min(row["cosine"] for row in rows),
    }
  worst_rows = sorted(
      comparisons, key=lambda row: row["relative_l2"], reverse=True)[:20]
  decode_comparisons = [row for row in comparisons if row["tokens"] == 1]
  prefill_comparisons = [row for row in comparisons if row["tokens"] > 1]
  stock_runtime = stock["result"].get("runtime_rows", [])
  candidate_runtime = candidate["result"].get("runtime_rows", [])
  stock_cases = stock["result"].get("cases", [])
  candidate_cases = candidate["result"].get("cases", [])
  stock_profile = stock_cases[0].get("profile", []) if stock_cases else []
  candidate_decode_cases = [
      row for row in candidate_cases
      if not str(row.get("label", "")).startswith("prefill_t")]
  candidate_prefill_cases = [
      row for row in candidate_cases
      if str(row.get("label", "")).startswith("prefill_t")]
  candidate_decode_profile = (
      candidate_decode_cases[0].get("profile", [])
      if candidate_decode_cases else [])
  candidate_prefill_profile = (
      candidate_prefill_cases[0].get("profile", [])
      if candidate_prefill_cases else [])
  stock_dq_profile = [row for row in stock_profile
                      if row["type"] == "DynamicQuantize"]
  stock_fc_profile = [row for row in stock_profile
                      if row["type"] == "FullyConnectedCompressed"]
  candidate_custom_profile = [
      row for row in (candidate_prefill_profile if args.phase_hybrid else
                      candidate_decode_profile)
      if row["type"] == "IQ36FixedFC4"]
  candidate_decode_fc_profile = [
      row for row in candidate_decode_profile
      if row["type"] == "FullyConnectedCompressed"]
  candidate_decode_custom_profile = [
      row for row in candidate_decode_profile
      if row["type"] == "IQ36FixedFC4"]
  candidate_prefill_fc_profile = [
      row for row in candidate_prefill_profile
      if row["type"] == "FullyConnectedCompressed"]
  candidate_decode_if_profile = [
      row for row in candidate_decode_profile if row["type"] == "If"]
  candidate_prefill_if_profile = [
      row for row in candidate_prefill_profile if row["type"] == "If"]
  candidate_manager_fc_profiles = [[
      row for row in case.get("profile", [])
      if row["type"] == "FullyConnectedCompressed"]
      for case in candidate_cases]
  stock_source = stock["result"].get("source", {})
  candidate_source = candidate["result"].get("source", {})
  manager_trace_path = raw / "candidate" / "manager-trace.jsonl"
  manager_trace = []
  if args.manager_direct and manager_trace_path.is_file():
    for line in manager_trace_path.read_text(encoding="utf-8").splitlines():
      try:
        manager_trace.append(json.loads(line))
      except json.JSONDecodeError:
        pass
  manager_selections = [
      row for row in manager_trace if row.get("provider") ==
      "iq36_fixed_fc_row_major_u8zp"]
  manager_prepacks = [
      row for row in manager_trace
      if row.get("stage") == "metadata_prepack"]
  git = git_state(output)
  dq_text = DQ_SOURCE.read_text(encoding="utf-8")
  onednn_fc_text = ONEDNN_FC_SOURCE.read_text(encoding="utf-8")
  onednn_jit_text = ONEDNN_JIT_SOURCE.read_text(encoding="utf-8")
  phase_provider_patch_text = PHASE_PROVIDER_PATCH.read_text(
      encoding="utf-8")
  phase_provider_reverse = subprocess.run(
      ["git", "apply", "--check", "--reverse",
       str(PHASE_PROVIDER_PATCH.resolve())],
      cwd=OV_SOURCE, capture_output=True, text=True)
  checks = [
      check("repository_clean_at_gate", not git["dirty"], git=git),
      check("stock_and_candidate_workers_completed_serially",
            worker_ok and
            all(left["finished_monotonic"] <= right["started_monotonic"]
                for left, right in zip(workers, workers[1:]))),
      check("locked_all_layer_source_census_is_exact",
            stock_source.get("input_count") == 40 and
            stock_source.get("output_count") == 160 and
            set(stock_source.get("output_element_types", [])) == {
                "<Type: 'float16'>"} and
            stock_source.get("operation_census", {}).get("MatMul") == 160 and
            candidate_source.get("input_count") == 40 and
            candidate_source.get("output_count") == 160 and
            set(candidate_source.get("output_element_types", [])) == {
                "<Type: 'float16'>"} and
            ((args.manager_direct and
              candidate_source.get("operation_census", {}).get("MatMul") ==
              160 and
              candidate_source.get("operation_census", {}).get("If", 0) ==
              0 and
              candidate_source.get("operation_census", {}).get(
                  "IQ36FixedFC4", 0) == 0) or
             (args.phase_hybrid and
              candidate_source.get("operation_census", {}).get("If") == 40 and
              len(candidate_source.get("hybrid_rows", [])) == 40) or
             (not args.phase_hybrid and not args.manager_direct and
              candidate_source.get("operation_census", {}).get(
                  "IQ36FixedFC4") == 40 and
              candidate_source.get("operation_census", {}).get(
                  "MatMul", 0) == 0))),
      check("candidate_router_composition_is_exact",
            (args.manager_direct and
             candidate_source.get("operation_census", {}).get("MatMul") ==
             160 and candidate_source.get("rewrite") is None and
             candidate_source.get("hybrid_rows") == []) or
            (args.phase_hybrid and
             len(candidate_source.get("hybrid_rows", [])) == 40 and
             all(row.get("condition") == "tokens == 1"
                 for row in candidate_source.get("hybrid_rows", []))) or
            (not args.phase_hybrid and not args.manager_direct and
             candidate_source.get("rewrite", {}).get(
                 "fixed_fc_rewrite_count") == 40 and
             candidate_source.get("rewrite", {}).get(
                 "fixed_fc_projection_count") == 160 and
             candidate_source.get("rewrite", {}).get(
                 "fixed_fc_group_counts") == {"router_shared_input": 40})),
      check("stock_t1_dynamic_quantize_is_optimized_out",
            len(stock_dq_profile) == 160 and
            all(row["status"] == "Status.OPTIMIZED_OUT" and
                row["real_time_us"] == 0.0 for row in stock_dq_profile) and
            len(stock_fc_profile) == 160 and
            sum(row["status"] == "Status.EXECUTED"
                for row in stock_fc_profile) == 160,
            dynamic_quantize_rows=len(stock_dq_profile),
            fully_connected_rows=len(stock_fc_profile)),
      check("candidate_phase_execution_is_exact",
            (args.manager_direct and
             [row.get("label") for row in candidate_cases] == [
                 "rotated_real", f"prefill_t{args.hybrid_prefill_tokens}_real",
                 "signed_real"] and
             len(candidate_manager_fc_profiles) == 3 and
             all(len(rows) == 120 and all(
                 row["status"] == "Status.EXECUTED" for row in rows)
                 for rows in candidate_manager_fc_profiles) and
             len(manager_selections) == 1 and
             {(int(row.get("m", 0)), int(row.get("tokens", 0)))
              for row in manager_selections} == {
                  (1024, args.hybrid_prefill_tokens)} and
             len(manager_prepacks) == 40 and
             Counter(int(row.get("m", 0)) for row in manager_prepacks) ==
             {1024: 40}) or
            (args.phase_hybrid and
             len(candidate_decode_if_profile) == 40 and
             all(row["status"] == "Status.EXECUTED"
                 for row in candidate_decode_if_profile) and
             len(candidate_prefill_if_profile) == 40 and
             all(row["status"] == "Status.EXECUTED"
                 for row in candidate_prefill_if_profile)) or
            (not args.phase_hybrid and not args.manager_direct and
             len(candidate_custom_profile) == 40 and
             all(row["status"] == "Status.EXECUTED"
                 for row in candidate_custom_profile)),
            decode_fc_rows=len(candidate_decode_fc_profile),
            decode_custom_rows=len(candidate_decode_custom_profile),
            prefill_fc_rows=len(candidate_prefill_fc_profile),
            prefill_custom_rows=len(candidate_custom_profile),
            decode_if_rows=len(candidate_decode_if_profile),
            prefill_if_rows=len(candidate_prefill_if_profile),
            manager_selection_rows=len(manager_selections),
            manager_prepack_rows=len(manager_prepacks),
            manager_prepack_bytes=sum(
                int(row.get("scale_bytes", 0)) +
                int(row.get("zero_point_bytes", 0))
                for row in manager_prepacks),
            note=("If profiling is top-level; exact decode and prefill output "
                  "checks below prove the selected inner body")),
      check("source_explains_t1_stock_boundary",
            "dynamic_quantization_threshold" in dq_text and
            "set_fpmath_mode(dnnl::fpmath_mode::f16, true)" in
            onednn_fc_text and
            "get_matmul_primitive_descriptor" in onednn_fc_text and
            'DECLARE_COMMON_PD_T("jit:gemm:any", gen_t)' in
            onednn_jit_text),
      check("candidate_source_contains_durable_phase_provider_patch",
            phase_provider_reverse.returncode == 0 and
            "IQ36FixedFCImplementationManager" in
            phase_provider_patch_text and
            "iq36_fixed_fc_row_major_u8zp" in
            phase_provider_patch_text and
            "IQ36_FIXED_FC_MANAGER_SCOPE" in
            phase_provider_patch_text and
            "-cl-intel-256-GRF-per-thread" in
            phase_provider_patch_text and
            "width != 1024" in
            phase_provider_patch_text,
            patch=str(PHASE_PROVIDER_PATCH.relative_to(ROOT)),
            patch_sha256=sha256(PHASE_PROVIDER_PATCH),
            reverse_check_returncode=phase_provider_reverse.returncode,
            reverse_check_stderr=phase_provider_reverse.stderr),
      check("all_decode_outputs_are_bit_exact_to_stock",
            len(decode_comparisons) == 320 and all(
                row["finite"] and row["exact_rate"] == 1.0
                for row in decode_comparisons),
            compared_rows=len(decode_comparisons),
            by_projection=by_projection,
            worst_rows=worst_rows),
      check("hybrid_prefill_outputs_preserve_admitted_numeric_boundary",
            (not args.phase_hybrid and not args.manager_direct) or
            (args.manager_direct and len(provider_comparisons) == 160 and all(
                row["finite"] and
                ((row["projection"] in (
                    "shared_expert_gate_proj", "shared_expert_up_proj") and
                  row["exact_rate"] == 1.0) or
                 (row["projection"] == "shared_expert_gate" and
                  row["max_abs_diff"] <= 0.02 and
                  row["cosine"] >= 0.9995) or
                 (row["projection"] == "router" and
                  row["relative_l2"] <= 0.05 and
                  row["max_abs_diff"] <= 0.1 and
                  row["cosine"] >= 0.999))
                for row in provider_comparisons)) or
            (len(provider_comparisons) == 160 and all(
                row["finite"] and row["exact_rate"] == 1.0
                for row in provider_comparisons)),
            compared_rows=len(provider_comparisons),
            worst_rows=sorted(
                provider_comparisons,
                key=lambda row: row["relative_l2"], reverse=True)[:20]),
      check("memory_guards_held_without_oom",
            all(not row["oom_observed"] and
                not row["memory_guard_tripped"] and
                row["monitor"]["system_available_min_bytes"] >= stop_bytes
                for row in workers),
            workers={row["label"]: row["monitor"] for row in workers},
            note="swap is pressure telemetry only"),
  ]
  passed = all(row["pass"] for row in checks)
  stock_wall = [row["wall_us_median"]
                for row in stock["result"].get("cases", [])]
  candidate_wall = [row["wall_us_median"]
                    for row in candidate["result"].get("cases", [])]
  stock_case_map = {
      row["label"]: row for row in stock["result"].get("cases", [])}
  custom_case_map = ({
      row["label"]: row for row in custom["result"].get("cases", [])}
      if custom is not None else {})
  candidate_case_map = {
      row["label"]: row for row in candidate["result"].get("cases", [])}
  timing_by_case = []
  for label, candidate_case in candidate_case_map.items():
    stock_case = stock_case_map.get(label, {})
    custom_case = custom_case_map.get(label, {})
    stock_median = stock_case.get("wall_us_median")
    custom_median = custom_case.get("wall_us_median")
    candidate_median = candidate_case.get("wall_us_median")
    timing_by_case.append({
        "label": label,
        "tokens": (1 if not label.startswith("prefill_t") else
                   args.hybrid_prefill_tokens),
        "stock_wall_us_median": stock_median,
        "custom_wall_us_median": custom_median,
        "candidate_wall_us_median": candidate_median,
        "candidate_to_stock_ratio": (
            candidate_median / stock_median
            if stock_median and candidate_median else None),
        "candidate_to_custom_ratio": (
            candidate_median / custom_median
            if custom_median and candidate_median else None),
        "candidate_minus_stock_us": (
            candidate_median - stock_median
            if stock_median is not None and candidate_median is not None
            else None),
        "candidate_minus_custom_us": (
            candidate_median - custom_median
            if custom_median is not None and candidate_median is not None
            else None),
    })
  metrics = {
      "schema": SCHEMA,
      "created_at": datetime.now(timezone.utc).isoformat(),
      "workstream": WS,
      "verdict": (
          ("admit_plugin_internal_stock_t1_row_major_t128_phase_provider"
           if args.manager_direct else
           "admit_functional_same_graph_stock_decode_custom_prefill_phase_hybrid"
           if args.phase_hybrid else
           "admit_stock_equivalent_router_shared_decode_body") if passed else
          ("plugin_internal_phase_provider_not_admitted"
           if args.manager_direct else
           "router_shared_phase_hybrid_not_admitted" if args.phase_hybrid else
           "router_shared_decode_body_not_stock_equivalent")),
      "required_checks_passed": passed,
      "git": git,
      "inputs": {
          "activation_type": "f32",
          "phase_hybrid": args.phase_hybrid,
          "manager_direct": args.manager_direct,
          "hybrid_prefill_tokens": (
              args.hybrid_prefill_tokens
              if args.phase_hybrid or args.manager_direct else None),
          "capture": str(CAPTURE.relative_to(ROOT)),
          "capture_sha256": sha256(CAPTURE),
          "stock_plugin": {"path": str(args.stock_plugin.resolve()),
                           "sha256": sha256(args.stock_plugin)},
          "candidate_plugin": {
              "path": str(args.candidate_plugin.resolve()),
              "sha256": sha256(args.candidate_plugin)},
          "custom_config": {"path": str(args.config.resolve()),
                            "sha256": sha256(args.config)},
          "dynamic_quantize_source_sha256": sha256(DQ_SOURCE),
          "stock_onednn_fc_source_sha256": sha256(ONEDNN_FC_SOURCE),
          "stock_onednn_jit_source_sha256": sha256(ONEDNN_JIT_SOURCE),
          "phase_provider_patch": {
              "path": str(PHASE_PROVIDER_PATCH.relative_to(ROOT)),
              "sha256": sha256(PHASE_PROVIDER_PATCH),
          },
      },
      "stock_worker": stock,
      "custom_worker": custom,
      "candidate_worker": candidate,
      "comparisons": comparisons,
      "provider_comparisons": provider_comparisons,
      "rollup": {
          "by_projection": by_projection,
          "decode_compared_rows": len(decode_comparisons),
          "prefill_compared_rows": len(prefill_comparisons),
          "worst_rows": worst_rows,
          "stock_wall_us_medians": stock_wall,
          "candidate_wall_us_medians": candidate_wall,
          "timing_by_case": timing_by_case,
          "candidate_to_stock_wall_ratio": (
              statistics.median(candidate_wall) /
              statistics.median(stock_wall)
              if stock_wall and candidate_wall else None),
          "timing_role": "sequential component diagnostic only",
      },
      "runtime_summary": {
          "stock_rows": Counter(row["layer_type"] for row in stock_runtime),
          "candidate_rows": Counter(
              row["layer_type"] for row in candidate_runtime),
      },
      "manager_trace": {
          "path": (str(manager_trace_path.relative_to(output))
                   if args.manager_direct else None),
          "selection_rows": manager_selections,
          "metadata_prepack_rows": manager_prepacks,
      },
      "checks": checks,
      "next_action": (
          ("Run the bounded full-model census before any paired product row."
           if args.manager_direct else
           "Close graph-level If on measured decode overhead and build a "
           "plugin-internal stock-T1/custom-T>1 provider before another "
           "model worker."
           if args.phase_hybrid else
           "Integrate the stock-equivalent decode body.") if passed else
          "Preserve stock oneDNN jit:gemm:any for T=1 and select the admitted "
          "custom provider only for N>1; prove the phase split in this "
          "component before another model worker."),
  }
  write_json(output / "metrics.json", metrics)
  write_json(output / "manifest.json", {
      "schema": SCHEMA, "workstream": WS,
      "required_checks_passed": passed, "metrics": "metrics.json",
      "raw": "raw/",
  })
  print(json.dumps({
      "output": str(output),
      "verdict": metrics["verdict"],
      "required_checks_passed": passed,
      "failed_checks": [row["name"] for row in checks if not row["pass"]],
      "by_projection": by_projection,
      "worst_rows": worst_rows[:8],
      "candidate_to_stock_wall_ratio": metrics["rollup"][
          "candidate_to_stock_wall_ratio"],
      "timing_by_case": timing_by_case,
  }, indent=2, sort_keys=True))
  return 0 if passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
