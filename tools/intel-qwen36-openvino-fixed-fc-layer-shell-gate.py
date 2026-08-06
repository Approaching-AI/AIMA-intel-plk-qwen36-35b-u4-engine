#!/usr/bin/env python3
"""Gate the real layer-0 fixed-FC producer-consumer shell.

Stock and candidate GPU plugins execute in separate serial workers.  The shell
uses locked real weights and the captured real layer-0 FC activation, but cuts
the unrelated attention/state/MoE body at four explicit parameters.  Its eight
results retain real immediate consumers around all four layer-0 fixed-FC
families.
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
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from xml.sax.saxutils import quoteattr


ROOT = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-openvino-fixed-fc-layer-shell-gate-v0"
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

SHELL_INPUT_SHAPES = {
    "linear_attention_input": [-1, -1, 2048],
    "router_shared_input": [-1, 2048],
    "attention_output": [-1, -1, 4096],
}
RESULT_NAMES = (
    "__module.model.model.language_model.layers.0.linear_attn/"
    "aten::transpose/Transpose",
    "__module.model.model.language_model.layers.0.linear_attn/"
    "aten::softplus/SoftPlus",
    "__module.model.model.language_model.layers.0.linear_attn/"
    "aten::sigmoid/Sigmoid",
    "__module.model.model.language_model.layers.0.linear_attn/"
    "aten::reshape/Reshape_5",
    "__module.model.model.language_model.layers.0/aten::add/Add",
    "__module.model.model.language_model.layers.0.mlp.gate/"
    "aten::softmax/Softmax",
    "__module.model.model.language_model.layers.0.mlp.shared_expert/"
    "aten::mul/Multiply",
    "__module.model.model.language_model.layers.0.mlp/aten::mul/Multiply",
)
RESIDUAL_ADD = (
    "__module.model.model.language_model.layers.0/aten::add/Add")
EXPECTED_SHELL_COHORTS = {
    "linear_attention_input": (8192, 32, 32, 4096),
    "router_shared_input": (1, 512, 512, 256),
    "attention_output": (2048,),
    "shared_expert_down": (2048,),
}
ALLOWED_UNCOMMITTED = {
    "tools/intel-qwen36-openvino-fixed-fc-layer-shell-gate.py",
}


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path)
  parser.add_argument("--openvino-python", type=Path, default=OV_PYTHON)
  parser.add_argument("--stock-plugin", type=Path, default=STOCK_PLUGIN)
  parser.add_argument(
      "--candidate-plugin", type=Path, default=CANDIDATE_PLUGIN)
  parser.add_argument("--config", type=Path, default=CONFIG)
  parser.add_argument("--warmups", type=int, default=512)
  parser.add_argument("--samples", type=int, default=31)
  parser.add_argument("--blocks", type=int, default=8)
  parser.add_argument("--timeout-s", type=float, default=300.0)
  parser.add_argument("--min-available-gib", type=float, default=8.0)
  parser.add_argument("--abort-below-available-gib", type=float, default=4.0)
  parser.add_argument("--worker-config", type=Path, help=argparse.SUPPRESS)
  args = parser.parse_args()
  if args.worker_config is None and args.output is None:
    parser.error("--output is required")
  if (args.warmups < 1 or args.samples < 5 or args.blocks < 8 or
      args.timeout_s <= 0.0):
    parser.error("timing arguments are invalid")
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


def op_census(model: Any) -> dict[str, int]:
  counts = Counter(node.get_type_name() for node in model.get_ordered_ops())
  names = (
      "IQ36FixedFC1", "IQ36FixedFC3", "IQ36FixedFC4", "MatMul",
      "Convert", "Transpose", "SoftPlus", "Sigmoid", "Softmax",
      "Multiply", "Add", "Reshape")
  return {name: int(counts.get(name, 0)) for name in names}


def build_shell(candidate: bool, graph: Any, ov: Any, np: Any
                ) -> tuple[Any, dict[str, Any]]:
  source_model = ov.Core().read_model(str(graph.MODEL_XML))
  groups = graph.discover_fixed_fc_groups(source_model, ov)
  layer_groups = [group for group in groups if group["layer"] == 0]
  observed = {
      group["cohort"]: tuple(int(value) for value in group["widths"])
      for group in layer_groups
  }
  if observed != EXPECTED_SHELL_COHORTS:
    raise ValueError(f"layer-0 fixed-FC cohorts differ: {observed}")
  rewrite = None
  rows = {}
  if candidate:
    rewrite = graph.rewrite_fixed_fc(source_model, ov, np)
    rows = {(row["layer"], row["cohort"]): row
            for row in rewrite["fixed_fc_rows"]}
  nodes = {node.get_friendly_name(): node
           for node in source_model.get_ordered_ops()}
  parameters = []
  for group in layer_groups:
    cohort = group["cohort"]
    if cohort not in SHELL_INPUT_SHAPES:
      continue
    parameter = ov.opset13.parameter(
        SHELL_INPUT_SHAPES[cohort], ov.Type.f32, name=cohort)
    parameters.append(parameter)
    if candidate:
      operation = nodes[rows[(0, cohort)]["operation"]]
      conversion = operation.input_value(0).get_node()
      if conversion.get_type_name() != "Convert":
        raise ValueError(f"{cohort}: candidate F16 conversion missing")
      conversion.input(0).replace_source_output(parameter.output(0))
    else:
      for projection in group["projections"]:
        projection["matmul"].input(0).replace_source_output(
            parameter.output(0))
  residual = ov.opset13.parameter(
      [-1, -1, 2048], ov.Type.f32, name="residual")
  parameters.append(residual)
  nodes[RESIDUAL_ADD].input(0).replace_source_output(residual.output(0))
  results = [nodes[name].output(0) for name in RESULT_NAMES]
  shell = ov.Model(
      results, parameters,
      "iq36_layer0_fixed_fc_candidate" if candidate else
      "iq36_layer0_fixed_fc_stock")
  shell.validate_nodes_and_infer_types()
  metadata = {
      "layer": 0,
      "cohorts": {key: list(value) for key, value in observed.items()},
      "operation_count": len(shell.get_ordered_ops()),
      "operation_census": op_census(shell),
      "inputs": [{
          "name": value.get_any_name(),
          "shape": str(value.get_partial_shape()),
          "type": value.get_element_type().get_type_name(),
      } for value in shell.inputs],
      "outputs": [{
          "name": RESULT_NAMES[index],
          "shape": str(value.get_partial_shape()),
          "type": value.get_element_type().get_type_name(),
      } for index, value in enumerate(shell.outputs)],
      "rewrite": ({key: rewrite[key] for key in (
          "fixed_fc_rewrite_count", "fixed_fc_projection_count",
          "fixed_fc_custom_counts", "fixed_fc_old_matmuls_remaining",
          "fixed_fc_f16_to_f32_restore_count")} if rewrite else None),
  }
  return shell, metadata


def make_feeds(np: Any) -> dict[str, Any]:
  captured = np.fromfile(CAPTURE, dtype=np.float16)
  if captured.size != 2048:
    raise ValueError("captured real activation has wrong size")
  base = captured.astype(np.float32)
  attention = np.concatenate([captured, captured[::-1]])
  attention = (attention * np.float16(0.03125)).astype(
      np.float16).astype(np.float32)
  return {
      "linear_attention_input": base.reshape(1, 1, 2048),
      "router_shared_input": base.reshape(1, 2048),
      "attention_output": attention.reshape(1, 1, 4096),
      "residual": base.reshape(1, 1, 2048),
  }


def profile_rows(request: Any) -> list[dict[str, Any]]:
  return [{
      "name": row.node_name,
      "type": row.node_type,
      "exec_type": row.exec_type,
      "status": str(row.status),
      "real_time_us": row.real_time.total_seconds() * 1_000_000.0,
  } for row in request.get_profiling_info()]


def runtime_census(compiled: Any) -> dict[str, Any]:
  layer_types = Counter()
  custom = []
  for node in compiled.get_runtime_model().get_ordered_ops():
    info = {}
    for key, value in node.get_rt_info().items():
      try:
        info[str(key)] = value.value
      except Exception:
        info[str(key)] = str(value)
    layer_type = str(info.get("layerType", node.get_type_name()))
    layer_types[layer_type] += 1
    if layer_type == "CustomGPUPrimitive":
      custom.append({
          "name": node.get_friendly_name(),
          "primitive_type": str(info.get("primitiveType")),
          "output_precisions": str(info.get("outputPrecisions")),
      })
  return {"layer_types": dict(layer_types), "custom": custom}


def worker_main(config_path: Path) -> int:
  cfg = load_json(config_path)
  if Path(sys.prefix).resolve() != Path(cfg["openvino_python"]).parent.parent:
    raise RuntimeError(f"worker requires {cfg['openvino_python']}")
  import numpy as np
  import openvino as ov

  candidate = cfg["mode"] == "candidate"
  graph = load_module(GRAPH, f"iq36_fixed_fc_shell_{cfg['mode']}")
  if meminfo()["MemAvailable"] < int(cfg["stop_bytes"]):
    raise RuntimeError("worker memory stop at start")
  model, source = build_shell(candidate, graph, ov, np)
  feeds = make_feeds(np)
  raw = Path(cfg["raw"])
  if candidate:
    registry = raw / "candidate-plugins.xml"
    registry.write_text(
        "<ie><plugins><plugin name=\"GPU\" location="
        f"{quoteattr(str(Path(cfg['plugin']).resolve()))}/></plugins></ie>\n",
        encoding="utf-8")
    core = ov.Core(str(registry))
    config_before = str(core.get_property("GPU", "CONFIG_FILE"))
    core.set_property("GPU", {"CONFIG_FILE": cfg["custom_config"]})
    config_after = str(core.get_property("GPU", "CONFIG_FILE"))
  else:
    core = ov.Core()
    config_before = str(core.get_property("GPU", "CONFIG_FILE"))
    config_after = config_before
  compile_config = {"PERFORMANCE_HINT": "LATENCY", "PERF_COUNT": True}
  compile_started = time.perf_counter_ns()
  compiled = core.compile_model(model, "GPU", compile_config)
  compile_ms = (time.perf_counter_ns() - compile_started) / 1_000_000.0
  request = compiled.create_infer_request()
  feed = {compiled.input(name): value for name, value in feeds.items()}
  for _ in range(int(cfg["warmups"])):
    request.infer(feed, share_outputs=False)
  blocks = []
  outputs = None
  for block_index in range(int(cfg["blocks"])):
    wall_samples = []
    kernel_samples = []
    custom_samples = []
    for _ in range(int(cfg["samples"])):
      started = time.perf_counter_ns()
      outputs = request.infer(feed, share_outputs=False)
      wall_samples.append((time.perf_counter_ns() - started) / 1_000.0)
      profile = profile_rows(request)
      kernel_samples.append(sum(row["real_time_us"] for row in profile))
      custom_samples.append(sum(
          row["real_time_us"] for row in profile
          if row["type"].startswith("IQ36FixedFC")))
    blocks.append({
        "index": block_index,
        "wall_us_median": statistics.median(wall_samples),
        "wall_us_min": min(wall_samples),
        "profile_sum_us_median": statistics.median(kernel_samples),
        "profile_sum_us_min": min(kernel_samples),
        "custom_us_median": statistics.median(custom_samples),
    })
  if outputs is None:
    raise RuntimeError("worker produced no outputs")
  arrays = {
      f"output_{index:02d}": np.asarray(
          outputs[compiled.output(index)], dtype=np.float32)
      for index in range(len(RESULT_NAMES))
  }
  np.savez(raw / "outputs.npz", **arrays)
  final_profile = profile_rows(request)
  result = {
      "mode": cfg["mode"],
      "pid": os.getpid(),
      "openvino_version": ov.get_version(),
      "plugin": str(Path(cfg["plugin"]).resolve()),
      "plugin_sha256": sha256(Path(cfg["plugin"])),
      "config_before": config_before,
      "config_after": config_after,
      "compile_config": compile_config,
      "compile_ms": compile_ms,
      "source": source,
      "runtime": runtime_census(compiled),
      "final_profile": final_profile,
      "timing": {
          "warmups": int(cfg["warmups"]),
          "samples_per_block": int(cfg["samples"]),
          "blocks": blocks,
      },
      "outputs": [{
          "index": index,
          "name": RESULT_NAMES[index],
          "shape": list(value.shape),
          "finite": bool(np.isfinite(value).all()),
      } for index, value in enumerate(arrays.values())],
  }
  write_json(Path(cfg["result"]), result)
  del outputs, request, compiled, model
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
  environment.update({
      "NEO_CACHE_DIR": str(worker_raw / "neo-cache"),
      "NEO_CACHE_MAX_SIZE": str(2 * 1024**3),
      "NEO_CACHE_PERSISTENT": "1",
  })
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
  finished = time.monotonic()
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
      "finished_monotonic": finished,
      "elapsed_seconds": finished - started,
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
              args.candidate_plugin, args.config, GRAPH, CAPTURE)
  missing = [str(path) for path in required if not path.is_file()]
  if missing:
    raise SystemExit("missing gate inputs: " + ", ".join(missing))
  if sha256(CAPTURE) != EXPECTED_CAPTURE_SHA256:
    raise RuntimeError("locked real activation capture hash differs")
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
      "blocks": args.blocks,
      "stop_bytes": stop_bytes,
  }
  stock = run_worker(
      "stock", {**common, "mode": "stock",
                "plugin": str(args.stock_plugin.resolve())},
      args, raw, stop_bytes)
  candidate = run_worker(
      "candidate", {**common, "mode": "candidate",
                    "plugin": str(args.candidate_plugin.resolve())},
      args, raw, stop_bytes)
  worker_ok = all(
      row["returncode"] == 0 and not row["timed_out"] and
      not row["memory_guard_tripped"] and not row["oom_observed"]
      for row in (stock, candidate))
  comparisons = []
  if worker_ok:
    with np.load(stock["outputs"]) as stock_outputs, \
         np.load(candidate["outputs"]) as candidate_outputs:
      for index, name in enumerate(RESULT_NAMES):
        key = f"output_{index:02d}"
        comparisons.append({
            "index": index, "name": name,
            **vector_metrics(stock_outputs[key], candidate_outputs[key], np),
        })
  stock_source = stock["result"].get("source", {})
  candidate_source = candidate["result"].get("source", {})
  stock_blocks = stock["result"].get("timing", {}).get("blocks", [])
  candidate_blocks = candidate["result"].get(
      "timing", {}).get("blocks", [])
  stock_wall = [row["wall_us_median"] for row in stock_blocks]
  candidate_wall = [row["wall_us_median"] for row in candidate_blocks]
  stock_profile = [row["profile_sum_us_median"] for row in stock_blocks]
  candidate_profile = [
      row["profile_sum_us_median"] for row in candidate_blocks]
  index_matched_ratios = [
      candidate_wall[index] / stock_wall[index]
      for index in range(min(len(stock_wall), len(candidate_wall)))]
  timing = {
      "stock_wall_block_medians_us": stock_wall,
      "candidate_wall_block_medians_us": candidate_wall,
      "stock_profile_sum_block_medians_us": stock_profile,
      "candidate_profile_sum_block_medians_us": candidate_profile,
      "index_matched_wall_ratios": index_matched_ratios,
      "stock_wall_us_median": statistics.median(stock_wall)
      if stock_wall else None,
      "candidate_wall_us_median": statistics.median(candidate_wall)
      if candidate_wall else None,
      "candidate_to_stock_median_ratio": (
          statistics.median(candidate_wall) / statistics.median(stock_wall)
          if stock_wall and candidate_wall else None),
      "candidate_to_stock_profile_sum_ratio": (
          statistics.median(candidate_profile) /
          statistics.median(stock_profile)
          if stock_profile and candidate_profile else None),
      "strict_block_separation": (
          max(candidate_wall) < min(stock_wall)
          if stock_wall and candidate_wall else False),
      "inference_role": (
          "isolated sequential-worker route signal only; not interleaved "
          "paired product inference"),
  }
  candidate_custom = candidate["result"].get(
      "runtime", {}).get("custom", [])
  checks = [
      check("repository_clean_at_gate", not git_state(output)["dirty"],
            git=git_state(output)),
      check("stock_and_candidate_workers_completed_serially",
            worker_ok and
            stock["finished_monotonic"] <= candidate["started_monotonic"],
            stock={key: stock[key] for key in (
                "returncode", "timed_out", "memory_guard_tripped",
                "oom_observed")},
            candidate={key: candidate[key] for key in (
                "returncode", "timed_out", "memory_guard_tripped",
                "oom_observed")}),
      check("stock_and_candidate_plugins_are_distinct_and_pinned",
            stock["result"].get("plugin_sha256") == sha256(args.stock_plugin) and
            candidate["result"].get("plugin_sha256") ==
            sha256(args.candidate_plugin) and
            sha256(args.stock_plugin) != sha256(args.candidate_plugin)),
      check("locked_layer0_shell_source_census_is_exact",
            stock_source.get("operation_count") == 137 and
            stock_source.get("operation_census", {}).get("MatMul") == 10 and
            candidate_source.get("operation_count") == 79 and
            candidate_source.get("operation_census", {}).get("MatMul") == 0 and
            candidate_source.get("operation_census", {}).get(
                "IQ36FixedFC1") == 2 and
            candidate_source.get("operation_census", {}).get(
                "IQ36FixedFC4") == 2),
      check("complete_locked_rewrite_remains_exact_inside_shell_builder",
            candidate_source.get("rewrite", {}).get(
                "fixed_fc_rewrite_count") == 160 and
            candidate_source.get("rewrite", {}).get(
                "fixed_fc_projection_count") == 390 and
            candidate_source.get("rewrite", {}).get(
                "fixed_fc_f16_to_f32_restore_count") == 390 and
            candidate_source.get("rewrite", {}).get(
                "fixed_fc_old_matmuls_remaining") == []),
      check("candidate_runtime_executes_exactly_four_custom_groups",
            len(candidate_custom) == 4 and
            stock["result"].get("runtime", {}).get("custom", []) == []),
      check("all_eight_real_consumer_outputs_are_tight_to_stock",
            len(comparisons) == len(RESULT_NAMES) and all(
                row["finite"] and row["cosine"] >= 0.999 and
                row["relative_l2"] <= 0.002 and
                row["max_abs_diff"] <= 0.25
                for row in comparisons),
            worst={
                "cosine": min((row["cosine"] for row in comparisons),
                              default=None),
                "relative_l2": max(
                    (row["relative_l2"] for row in comparisons), default=None),
                "max_abs_diff": max(
                    (row["max_abs_diff"] for row in comparisons), default=None),
            }),
      check("candidate_shell_wall_has_strict_eight_block_separation",
            len(stock_wall) == args.blocks and
            len(candidate_wall) == args.blocks and
            timing["strict_block_separation"] and
            float(timing["candidate_to_stock_median_ratio"] or math.inf) <=
            0.95,
            timing=timing),
      check("memory_guards_held_without_oom",
            all(not row["oom_observed"] and
                not row["memory_guard_tripped"] and
                row["monitor"]["system_available_min_bytes"] >= stop_bytes
                for row in (stock, candidate)),
            stock=stock["monitor"], candidate=candidate["monitor"],
            note="swap is pressure telemetry only"),
  ]
  passed = all(row["pass"] for row in checks)
  verdict = (
      "admit_fixed_fc_layer0_producer_consumer_shell" if passed else
      "fixed_fc_layer0_shell_not_admitted")
  metrics = {
      "schema": SCHEMA,
      "created_at": datetime.now(timezone.utc).isoformat(),
      "workstream": WS,
      "verdict": verdict,
      "required_checks_passed": passed,
      "git": git_state(output),
      "inputs": {
          "capture": str(CAPTURE.relative_to(ROOT)),
          "capture_sha256": sha256(CAPTURE),
          "stock_plugin": {"path": str(args.stock_plugin.resolve()),
                           "sha256": sha256(args.stock_plugin)},
          "candidate_plugin": {"path": str(args.candidate_plugin.resolve()),
                               "sha256": sha256(args.candidate_plugin)},
          "custom_config": {"path": str(args.config.resolve()),
                            "sha256": sha256(args.config)},
      },
      "stock_worker": stock,
      "candidate_worker": candidate,
      "comparisons": comparisons,
      "timing": timing,
      "checks": checks,
      "next_action": (
          "Source-bound one full-graph candidate compile/runtime census before "
          "any token-bearing row." if passed else
          "Resolve the first shell numeric, runtime-census, or timing failure."),
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
      "worst_numeric": next(
          row.get("worst") for row in checks
          if row["name"] == "all_eight_real_consumer_outputs_are_tight_to_stock"),
      "timing": timing,
  }, indent=2, sort_keys=True))
  return 0 if passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
