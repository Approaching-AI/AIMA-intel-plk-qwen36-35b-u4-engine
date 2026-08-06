#!/usr/bin/env python3
"""Gate a bandwidth-reducing signed codec for the product OpenVINO LM head.

The driver launches exactly one candidate-only 2k worker.  The worker builds
the clean accepted graph, replaces the language-model result with the final
LM-head input, and therefore captures the 18 real teacher-forced hidden
vectors without compiling or executing the 508-MiB LM-head weight.

The driver then audits the locked IR constants and projects the 17 real decode
hidden vectors through the original per-row I8 weights and deterministic
signed 7/6/5/4-bit requantizations, including the GPU provider's half-precision
group-256 Q8 activation codec.  Prefill remains on the existing I8 provider.
This is a source/numeric/bandwidth funding gate, not a product correctness or
speed claim.  A decode codec may be selected only when it preserves every
real-path top-1, stays below the product KLD limit, and its conservative
packed-byte floor can cover the current complete kill-number.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import platform
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from xml.sax.saxutils import quoteattr


REPO = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-openvino-lm-head-codec-bound-v0"
MODEL_DIR = Path("/home/intel/Qwen3.6-35B-A3B-ov")
MODEL_XML = MODEL_DIR / "openvino_language_model.xml"
MODEL_BIN = MODEL_DIR / "openvino_language_model.bin"
OV_PYTHON = Path("/home/intel/ov/openvino_env/bin/python")
PLUGIN = Path(
    "/home/intel/intel-qwen36-r0/output/openvino-90214e-l0-gpu/"
    "bin/intel64/Release/libopenvino_intel_gpu_plugin.so")
CUSTOM_CONFIG = REPO / "engine/openvino/custom/iq36_hot_attention_gqa.xml"
PROMPT = REPO / (
    "output/r0-oracle-prompt-materialization-20260626T082201Z/"
    "prompts/sentinel_002k.txt")
GRAPH_MODULE = REPO / "tools/intel_qwen36_openvino_hot_cold_attention.py"
BASE_GATE = REPO / "tools/intel-qwen36-openvino-full-attention-custom-gate.py"
ACCEPTED_MANIFEST = REPO / (
    "output/openvino-linear-state-alias-validation-20260718Tseq1451-"
    "default-all-final-plugin-32k-o64-cleancommit/manifest.json")
PROFILE_METRICS = REPO / (
    "output/openvino-accepted-carrier-profile-refresh-20260718Tseq1452-"
    "clean-all-alias-2k-warm17/metrics.json")
PROFILE_WORKER = REPO / (
    "output/openvino-accepted-carrier-profile-refresh-20260718Tseq1452-"
    "clean-all-alias-2k-warm17/raw/2k/candidate/worker-result.json")
FRONTIER = REPO / "doc/active" / WS / "frontier.json"

TARGET_LAYERS = tuple(range(3, 40, 4))
LM_HEAD_NAME = "__module.model.lm_head/ov_ext::linear/MatMul"
WEIGHT_NAME = "self.model.lm_head.weight"
SCALE_NAME = "self.model.lm_head.weight/scale"
DECODE_STEPS = 17
PREFILL_CHUNK_TOKENS = 8192
KLD_LIMIT = 0.005
CODEC_BITS = (7, 6, 5, 4)
CONSERVATIVE_BANDWIDTH_GBPS = 108.0
REFERENCE_BANDWIDTHS_GBPS = (108.0, 115.0, 136.5)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path)
  parser.add_argument("--plugin", type=Path, default=PLUGIN)
  parser.add_argument("--accepted-manifest", type=Path,
                      default=ACCEPTED_MANIFEST)
  parser.add_argument("--profile-metrics", type=Path,
                      default=PROFILE_METRICS)
  parser.add_argument("--profile-worker", type=Path, default=PROFILE_WORKER)
  parser.add_argument("--timeout-s", type=int, default=1800)
  parser.add_argument("--poll-interval-s", type=float, default=1.0)
  parser.add_argument("--min-available-gib", type=float, default=8.0)
  parser.add_argument("--abort-below-available-gib", type=float, default=4.0)
  parser.add_argument("--numeric-row-chunk", type=int, default=4096)
  parser.add_argument("--worker-config", type=Path, help=argparse.SUPPRESS)
  args = parser.parse_args()
  if args.worker_config is None and args.output is None:
    parser.error("--output is required")
  if args.timeout_s <= 0 or args.poll_interval_s <= 0.0:
    parser.error("timeout and poll interval must be positive")
  if args.min_available_gib < 0.0 or args.abort_below_available_gib < 0.0:
    parser.error("memory thresholds must be nonnegative")
  if args.abort_below_available_gib > args.min_available_gib:
    parser.error("abort threshold must not exceed preflight threshold")
  if args.numeric_row_chunk <= 0:
    parser.error("numeric row chunk must be positive")
  return args


def load_module(name: str, path: Path) -> Any:
  spec = importlib.util.spec_from_file_location(name, path)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load module from {path}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise TypeError(f"expected JSON object: {path}")
  return value


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    while chunk := stream.read(8 * 1024 * 1024):
      digest.update(chunk)
  return digest.hexdigest()


def display_path(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(REPO))
  except ValueError:
    return str(path.resolve())


def check(name: str, passed: bool, **details: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **details}


def meminfo() -> dict[str, int]:
  rows: dict[str, int] = {}
  for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
    key, value = line.split(":", 1)
    fields = value.split()
    rows[key] = int(fields[0]) * 1024 if fields else 0
  return rows


def mem_available_bytes() -> int:
  return int(meminfo().get("MemAvailable", 0))


def process_memory(pid: int) -> dict[str, int]:
  path = Path(f"/proc/{pid}/status")
  if not path.is_file():
    return {"VmRSS": 0, "VmSwap": 0}
  selected: dict[str, int] = {}
  for line in path.read_text(
      encoding="utf-8", errors="replace").splitlines():
    if ":" not in line:
      continue
    key, value = line.split(":", 1)
    fields = value.split()
    if key in ("VmRSS", "VmSwap") and fields:
      selected[key] = int(fields[0]) * 1024
  return {"VmRSS": selected.get("VmRSS", 0),
          "VmSwap": selected.get("VmSwap", 0)}


def wait_for_memory(required_bytes: int) -> dict[str, Any]:
  started = time.monotonic()
  while True:
    available = mem_available_bytes()
    if available >= required_bytes:
      return {"available_bytes": available, "required_bytes": required_bytes,
              "waited_seconds": time.monotonic() - started}
    if time.monotonic() - started >= 60.0:
      raise RuntimeError(
          f"available memory {available} remains below {required_bytes}")
    time.sleep(2.0)


def git_state(output: Path) -> dict[str, Any]:
  commit = subprocess.run(
      ["git", "rev-parse", "HEAD"], cwd=REPO, check=True,
      capture_output=True, text=True).stdout.strip()
  rows = subprocess.run(
      ["git", "status", "--porcelain"], cwd=REPO, check=True,
      capture_output=True, text=True).stdout.splitlines()
  try:
    output_relative = str(output.resolve().relative_to(REPO))
  except ValueError:
    output_relative = ""
  rows = [row for row in rows
          if not output_relative or output_relative not in row]
  return {"commit": commit, "dirty": bool(rows), "status": rows}


def other_worker_pids() -> list[dict[str, Any]]:
  rows = []
  for path in Path("/proc").iterdir():
    if not path.name.isdigit() or int(path.name) == os.getpid():
      continue
    try:
      command = (path / "cmdline").read_bytes().replace(
          b"\0", b" ").decode("utf-8", errors="replace")
    except (FileNotFoundError, PermissionError, ProcessLookupError):
      continue
    if ("intel-qwen36" in command and "--worker-config" in command and
        str(Path(__file__).resolve()) not in command):
      rows.append({"pid": int(path.name), "command": command.strip()})
  return rows


def stop_process_group(process: subprocess.Popen[Any], first_signal: int) -> None:
  try:
    os.killpg(process.pid, first_signal)
  except ProcessLookupError:
    return
  try:
    process.wait(timeout=10)
    return
  except subprocess.TimeoutExpired:
    pass
  try:
    os.killpg(process.pid, signal.SIGKILL)
  except ProcessLookupError:
    pass
  process.wait()


def worker_main(config_path: Path) -> int:
  if Path(sys.prefix).resolve() != OV_PYTHON.parent.parent.resolve():
    raise RuntimeError(f"worker requires {OV_PYTHON}, got {sys.executable}")

  import numpy as np
  import openvino as ov
  import openvino_genai as ov_genai

  graph = load_module("iq36_lm_head_capture_graph", GRAPH_MODULE)
  base = load_module("iq36_lm_head_capture_base", BASE_GATE)
  cfg = load_json(config_path)
  stop_bytes = int(cfg["memory_stop_bytes"])
  bucket = int(cfg.get("bucket", 2048))
  prompt_path = Path(cfg.get("prompt", str(PROMPT)))
  if mem_available_bytes() < stop_bytes:
    raise RuntimeError("worker preflight memory stop")
  raw = Path(cfg["raw"])
  raw.mkdir(parents=True, exist_ok=True)
  plugin = Path(cfg["plugin"])
  registry = raw / "candidate-plugins.xml"
  registry.write_text(
      "<ie><plugins><plugin name=\"GPU\" location="
      f"{quoteattr(str(plugin.resolve()))}/></plugins></ie>\n",
      encoding="utf-8")
  core = ov.Core(str(registry))
  device = "GPU"
  config_before = str(core.get_property(device, "CONFIG_FILE"))
  core.set_property(device, {"CONFIG_FILE": str(CUSTOM_CONFIG.resolve())})
  config_after = str(core.get_property(device, "CONFIG_FILE"))

  source, source_summary = graph.make_candidate_model(
      core, MODEL_DIR, ov, np, TARGET_LAYERS,
      initialize_hot_states=True,
      fixed_cold_capacity=bucket,
      prefill_history_capacity=max(2 * PREFILL_CHUNK_TOKENS, bucket),
      fuse_linear_conv_state=True)
  lm_head = next(
      node for node in source.get_ordered_ops()
      if node.get_friendly_name() == LM_HEAD_NAME)
  hidden = lm_head.input_value(0)
  hidden_shape = str(hidden.get_partial_shape())
  hidden_type = str(hidden.get_element_type())
  shape = ov.opset13.shape_of(hidden, "i64")
  sequence_length = ov.opset13.gather(
      shape, ov.opset13.constant(np.array(1, dtype=np.int64)),
      ov.opset13.constant(np.array(0, dtype=np.int64)))
  last_index = ov.opset13.subtract(
      sequence_length, ov.opset13.constant(np.array(1, dtype=np.int64)))
  last_hidden = ov.opset13.gather(
      hidden, last_index,
      ov.opset13.constant(np.array(1, dtype=np.int64)))
  last_hidden.set_friendly_name("iq36_lm_head_last_query_input")
  for result in list(source.get_results()):
    source.remove_result(result)
  source.add_results([ov.opset13.result(last_hidden.output(0))])
  source.validate_nodes_and_infer_types()

  embedding = core.compile_model(
      core.read_model(str(MODEL_DIR / "openvino_text_embeddings_model.xml")),
      "CPU", {"PERFORMANCE_HINT": "LATENCY"})
  tokenizer = ov_genai.Tokenizer(str(MODEL_DIR))
  prompt_ids = np.asarray(tokenizer.encode(
      prompt_path.read_text(encoding="utf-8")).input_ids.data
  ).reshape(-1).astype(np.int64)
  np.ascontiguousarray(prompt_ids, dtype="<u4").tofile(
      raw / "prompt-token-ids.u32")

  memory_samples = {
      "memory_stop_bytes": stop_bytes,
      "before_language_compile": mem_available_bytes(),
  }
  compile_config = dict(base.COMPILE_CONFIG)
  compile_config["ACTIVATIONS_SCALE_FACTOR"] = 0.0
  started = time.perf_counter_ns()
  compiled = core.compile_model(source, device, compile_config)
  compile_ms = (time.perf_counter_ns() - started) / 1_000_000.0
  del source
  gc.collect()
  memory_samples["after_language_compile"] = mem_available_bytes()
  if memory_samples["after_language_compile"] < stop_bytes:
    raise RuntimeError("worker post-compile memory stop")
  runtime_names = [
      node.get_friendly_name()
      for node in compiled.get_runtime_model().get_ordered_ops()]
  original_lm_head_runtime_names = [
      name for name in runtime_names
      if name == LM_HEAD_NAME or WEIGHT_NAME in name or SCALE_NAME in name]
  request = compiled.create_infer_request()
  request.reset_state()

  decode_tokens = [int(value) for value in cfg["decode_tokens"]]
  if not decode_tokens:
    raise ValueError("capture worker needs at least one decode token")
  phases = []
  start = 0
  for index in range(len(decode_tokens) + 1):
    tokens: Any = prompt_ids if index == 0 else [decode_tokens[index - 1]]
    total = start + len(tokens)
    infer_started = time.perf_counter_ns()
    outputs = request.infer(
        base.make_inputs(embedding, tokens, start, total, np),
        share_outputs=False)
    infer_ms = (time.perf_counter_ns() - infer_started) / 1_000_000.0
    vector = np.array(
        np.asarray(outputs[compiled.output(0)]).reshape(-1),
        dtype="<f4", copy=True)
    path = raw / f"phase{index}-lm-head-input.f32"
    vector.tofile(path)
    phases.append({
        "index": index,
        "input_token_ids": [int(value) for value in np.asarray(tokens).reshape(-1)],
        "total_tokens": int(total),
        "shape": list(vector.shape),
        "dtype": str(vector.dtype),
        "finite": bool(np.isfinite(vector).all()),
        "minimum": float(vector.min()),
        "maximum": float(vector.max()),
        "rms": float(np.sqrt(np.mean(np.square(vector, dtype=np.float64)))),
        "wall_ms_diagnostic": infer_ms,
        "path": display_path(path),
        "sha256": sha256(path),
    })
    start = total
  memory_samples["after_final_infer"] = mem_available_bytes()

  result = {
      "schema": SCHEMA + "-worker",
      "openvino_version": ov.get_version(),
      "openvino_genai_version": ov_genai.__version__,
      "host": platform.node(),
      "plugin": str(plugin.resolve()),
      "plugin_sha256": sha256(plugin),
      "candidate_plugin_registry": display_path(registry),
      "config_before": config_before,
      "config_after": config_after,
      "compile_config": compile_config,
      "compile_ms": compile_ms,
      "prompt_tokens": int(len(prompt_ids)),
      "decode_tokens": decode_tokens,
      "same_infer_request": True,
      "reset_state_called": True,
      "hot_state_self_bind_skipped": True,
      "lm_head": {
          "name": LM_HEAD_NAME,
          "input_partial_shape": hidden_shape,
          "input_element_type": hidden_type,
          "compiled_runtime_original_lm_head_names":
              original_lm_head_runtime_names,
          "compiled_runtime_lm_head_absent":
              not original_lm_head_runtime_names,
      },
      "source_summary": source_summary,
      "memory_samples": memory_samples,
      "phases": phases,
  }
  write_json(Path(cfg["result"]), result)
  print(json.dumps({
      "event": "complete", "phases": len(phases),
      "hidden_sha256": [row["sha256"] for row in phases],
  }, sort_keys=True))
  return 0


def inspect_ir_constants() -> dict[str, Any]:
  root = ET.parse(MODEL_XML).getroot()
  layers = root.find("layers")
  if layers is None:
    raise ValueError("IR has no layers")
  by_name = {layer.attrib.get("name", ""): layer for layer in layers}

  def constant(name: str) -> dict[str, Any]:
    layer = by_name[name]
    data = layer.find("data")
    if data is None:
      raise ValueError(f"constant has no data: {name}")
    shape = [int(item.strip()) for item in data.attrib["shape"].split(",")]
    return {
        "id": int(layer.attrib["id"]), "name": name,
        "element_type": data.attrib["element_type"], "shape": shape,
        "offset": int(data.attrib["offset"]),
        "size": int(data.attrib["size"]),
    }

  head = by_name[LM_HEAD_NAME]
  data = head.find("data")
  return {
      "xml": display_path(MODEL_XML),
      "bin": display_path(MODEL_BIN),
      "weight": constant(WEIGHT_NAME),
      "scale": constant(SCALE_NAME),
      "matmul": {
          "id": int(head.attrib["id"]), "name": LM_HEAD_NAME,
          "type": head.attrib["type"],
          "transpose_a": data.attrib.get("transpose_a") if data is not None else None,
          "transpose_b": data.attrib.get("transpose_b") if data is not None else None,
      },
  }


def launch_worker(
    args: argparse.Namespace, raw: Path, decode_tokens: list[int],
    prompt: Path = PROMPT, bucket: int = 2048,
) -> dict[str, Any]:
  cache = raw / "neo-cache"
  cache.mkdir(parents=True)
  result_path = raw / "worker-result.json"
  config_path = raw / "worker-config.json"
  config = {
      "bucket": bucket,
      "decode_tokens": decode_tokens,
      "memory_stop_bytes": int(args.abort_below_available_gib * 1024**3),
      "plugin": str(args.plugin.resolve()),
      "prompt": str(prompt.resolve()),
      "raw": str(raw.resolve()),
      "result": str(result_path.resolve()),
  }
  write_json(config_path, config)
  preflight = wait_for_memory(int(args.min_available_gib * 1024**3))
  command = [str(OV_PYTHON), str(Path(__file__).resolve()),
             "--worker-config", str(config_path)]
  environment = os.environ.copy()
  for key in (
      "OV_GPU_CONFIG_FILE", "OV_GPU_USM_POLICY", "IQ36_GDN_TRANSPOSED_STATE",
      "IQ36_GPU_ALIAS_LINEAR_STATE_ASSIGN_SCOPE", "LD_AUDIT"):
    environment.pop(key, None)
  environment.update({
      "IQ36_GPU_ALIAS_LINEAR_STATE_ASSIGN": "1",
      "IQ36_GPU_ALIAS_LINEAR_STATE_ASSIGN_SCOPE": "all",
      "NEO_CACHE_DIR": str(cache.resolve()),
      "NEO_CACHE_MAX_SIZE": str(4 * 1024**3),
      "NEO_CACHE_PERSISTENT": "1",
  })
  stdout_path = raw / "worker.stdout"
  stderr_path = raw / "worker.stderr"
  started = time.monotonic()
  monitor: dict[str, Any] = {
      "process_rss_peak_bytes": 0, "process_swap_peak_bytes": 0,
      "sample_count": 0, "system_available_min_bytes": None,
      "system_swap_used_peak_bytes": 0,
  }
  timed_out = False
  guard_tripped = False
  abort_bytes = int(args.abort_below_available_gib * 1024**3)
  with stdout_path.open("w", encoding="utf-8") as stdout_handle, \
       stderr_path.open("w", encoding="utf-8") as stderr_handle:
    process = subprocess.Popen(
        command, cwd=REPO, env=environment, stdout=stdout_handle,
        stderr=stderr_handle, text=True, start_new_session=True)
    while process.poll() is None:
      if time.monotonic() - started > args.timeout_s:
        timed_out = True
        stop_process_group(process, signal.SIGTERM)
        break
      system = meminfo()
      process_row = process_memory(process.pid)
      available = int(system.get("MemAvailable", 0))
      swap_used = int(system.get("SwapTotal", 0)) - int(
          system.get("SwapFree", 0))
      monitor["sample_count"] = int(monitor["sample_count"]) + 1
      monitor["process_rss_peak_bytes"] = max(
          int(monitor["process_rss_peak_bytes"]), process_row["VmRSS"])
      monitor["process_swap_peak_bytes"] = max(
          int(monitor["process_swap_peak_bytes"]), process_row["VmSwap"])
      current_min = monitor["system_available_min_bytes"]
      monitor["system_available_min_bytes"] = (
          available if current_min is None else min(int(current_min), available))
      monitor["system_swap_used_peak_bytes"] = max(
          int(monitor["system_swap_used_peak_bytes"]), swap_used)
      if abort_bytes and available < abort_bytes:
        guard_tripped = True
        stop_process_group(process, signal.SIGINT)
        break
      time.sleep(args.poll_interval_s)
    returncode = process.wait()
  stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
  oom = (not guard_tripped and
         (returncode in (-9, 137) or "out of memory" in stderr.lower() or
          "cl_out_of_resources" in stderr.lower()))
  return {
      "command": command,
      "environment": {key: environment[key] for key in (
          "IQ36_GPU_ALIAS_LINEAR_STATE_ASSIGN",
          "IQ36_GPU_ALIAS_LINEAR_STATE_ASSIGN_SCOPE", "NEO_CACHE_DIR",
          "NEO_CACHE_MAX_SIZE", "NEO_CACHE_PERSISTENT")},
      "elapsed_seconds": time.monotonic() - started,
      "memory_preflight": preflight,
      "memory_guard": {"abort_below_bytes": abort_bytes,
                       "tripped": guard_tripped},
      "monitor": monitor,
      "oom_observed": oom,
      "returncode": returncode,
      "timed_out": timed_out,
      "result": load_json(result_path) if result_path.is_file() else {},
  }


def weight_distribution(facts: dict[str, Any], np: Any) -> dict[str, Any]:
  weight = facts["weight"]
  rows, columns = weight["shape"]
  values = np.memmap(
      MODEL_BIN, dtype=np.int8, mode="r", offset=weight["offset"],
      shape=(rows, columns))
  histogram = np.zeros(256, dtype=np.int64)
  element_fit = {bits: 0 for bits in range(4, 9)}
  row_fit = {bits: 0 for bits in range(4, 9)}
  row_absmax_quantiles: list[int] = []
  chunk_rows = 4096
  for start in range(0, rows, chunk_rows):
    block = np.asarray(values[start:min(rows, start + chunk_rows)])
    wide = block.astype(np.int16)
    histogram += np.bincount(
        (wide.reshape(-1) + 128), minlength=256).astype(np.int64)
    minimum = wide.min(axis=1)
    maximum = wide.max(axis=1)
    row_absmax_quantiles.extend(
        np.maximum(np.abs(minimum), np.abs(maximum)).tolist())
    for bits in range(4, 9):
      low = -(1 << (bits - 1))
      high = (1 << (bits - 1)) - 1
      element_fit[bits] += int(np.count_nonzero(
          np.logical_and(wide >= low, wide <= high)))
      row_fit[bits] += int(np.count_nonzero(
          np.logical_and(minimum >= low, maximum <= high)))
  total = int(rows * columns)
  probability = histogram[histogram > 0].astype(np.float64) / total
  row_abs = np.asarray(row_absmax_quantiles, dtype=np.int16)
  return {
      "value_count": total,
      "minimum": int(np.flatnonzero(histogram)[0] - 128),
      "maximum": int(np.flatnonzero(histogram)[-1] - 128),
      "nonzero_fraction": 1.0 - float(histogram[128]) / total,
      "shannon_entropy_bits_per_weight": float(
          -np.sum(probability * np.log2(probability))),
      "element_fraction_fitting_signed_width": {
          str(bits): float(element_fit[bits]) / total for bits in range(4, 9)},
      "rows_fitting_signed_width": {
          str(bits): int(row_fit[bits]) for bits in range(4, 9)},
      "row_max_abs_quantiles": {
          str(q): float(np.quantile(row_abs, q))
          for q in (0.0, 0.5, 0.9, 0.99, 1.0)},
      "raw_exact_fixed_width_below_8_feasible": any(
          row_fit[bits] == rows for bits in range(4, 8)),
  }


def distribution_metrics(reference: Any, candidate: Any, np: Any) -> dict[str, Any]:
  ref = np.asarray(reference, dtype=np.float64).reshape(-1)
  cand = np.asarray(candidate, dtype=np.float64).reshape(-1)
  difference = cand - ref
  ref_probability = np.exp(ref - float(np.max(ref)))
  cand_probability = np.exp(cand - float(np.max(cand)))
  ref_probability /= float(ref_probability.sum())
  cand_probability /= float(cand_probability.sum())
  epsilon = np.finfo(np.float64).tiny
  ref_norm = float(np.linalg.norm(ref))
  reference_top1 = int(np.argmax(ref))
  candidate_top1 = int(np.argmax(cand))
  return {
      "finite": bool(np.isfinite(ref).all() and np.isfinite(cand).all()),
      "max_abs": float(np.max(np.abs(difference))),
      "relative_l2": float(np.linalg.norm(difference) / ref_norm),
      "kld_reference_to_candidate": float(np.sum(
          ref_probability * (
              np.log(np.maximum(ref_probability, epsilon)) -
              np.log(np.maximum(cand_probability, epsilon))))),
      "reference_top1": reference_top1,
      "candidate_top1": candidate_top1,
      "top1_match": reference_top1 == candidate_top1,
  }


def numeric_projection(
    facts: dict[str, Any], worker: dict[str, Any], profile_worker: dict[str, Any],
    row_chunk: int, np: Any,
) -> dict[str, Any]:
  phases = worker["phases"]
  hidden = np.stack([
      np.fromfile(REPO / row["path"], dtype="<f4") for row in phases])
  weight_fact = facts["weight"]
  scale_fact = facts["scale"]
  rows, columns = weight_fact["shape"]
  if hidden.shape != (DECODE_STEPS + 1, columns):
    raise ValueError(f"unexpected hidden matrix shape: {hidden.shape}")
  weights = np.memmap(
      MODEL_BIN, dtype=np.int8, mode="r", offset=weight_fact["offset"],
      shape=(rows, columns))
  scales = np.memmap(
      MODEL_BIN, dtype="<f2", mode="r", offset=scale_fact["offset"],
      shape=(rows, 1))
  projected = {
      "i8": np.empty((DECODE_STEPS + 1, rows), dtype=np.float32),
      **{str(bits): np.empty((DECODE_STEPS + 1, rows), dtype=np.float32)
         for bits in CODEC_BITS},
  }
  # Match dynamic_quantize_gpu_opt.cl: half input, half group maximum,
  # half (127/max) scale, round-to-even char conversion, and half reciprocal
  # dequantization scale.  This models the activation plane a decode rowstripe
  # implementation can consume or reproduce; it is stricter than the F32
  # hidden projection used to isolate weight-codec error.
  grouped_hidden = hidden.astype(np.float16).reshape(
      DECODE_STEPS + 1, columns // 256, 256)
  group_max = np.max(np.abs(grouped_hidden), axis=2, keepdims=True).astype(
      np.float16)
  quantize_scale = (np.float16(127.0) / group_max).astype(np.float16)
  q8_codes = np.clip(
      np.rint((grouped_hidden * quantize_scale).astype(np.float16)),
      -128, 127)
  dequantize_scale = (np.float16(1.0) / quantize_scale).astype(np.float16)
  q8_hidden = (
      q8_codes.astype(np.float16) * dequantize_scale
  ).astype(np.float16).reshape(DECODE_STEPS + 1, columns).astype(np.float32)
  q8_projected = {
      str(bits): np.empty((DECODE_STEPS + 1, rows), dtype=np.float32)
      for bits in CODEC_BITS}
  hidden_t = np.ascontiguousarray(hidden.T, dtype=np.float32)
  q8_hidden_t = np.ascontiguousarray(q8_hidden.T, dtype=np.float32)
  started = time.monotonic()
  for start in range(0, rows, row_chunk):
    end = min(rows, start + row_chunk)
    raw = np.asarray(weights[start:end], dtype=np.float32)
    scale = np.asarray(scales[start:end], dtype=np.float32).reshape(-1, 1)
    projected["i8"][:, start:end] = (raw @ hidden_t * scale).T
    for bits in CODEC_BITS:
      factor = float(128 // (1 << (bits - 1)))
      codes = np.rint(raw / factor)
      np.clip(codes, -(1 << (bits - 1)), (1 << (bits - 1)) - 1,
              out=codes)
      projected[str(bits)][:, start:end] = (
          codes @ hidden_t * (scale * factor)).T
      q8_projected[str(bits)][:, start:end] = (
          codes @ q8_hidden_t * (scale * factor)).T
  projection_seconds = time.monotonic() - started

  actual = np.stack([
      np.fromfile(REPO / row["logits_path"], dtype="<f4")
      for row in profile_worker["phases"]])
  if actual.shape != (DECODE_STEPS + 1, rows):
    raise ValueError(f"unexpected actual logits shape: {actual.shape}")

  semantic_alignment = [
      distribution_metrics(actual[index], projected["i8"][index], np)
      for index in range(DECODE_STEPS + 1)]

  def decode_summary(rows_to_summarize: list[dict[str, Any]]) -> dict[str, Any]:
    decode_rows = rows_to_summarize[1:]
    return {
        "prefill_phase": rows_to_summarize[0],
        "all_phase_max_kld": max(
            row["kld_reference_to_candidate"]
            for row in rows_to_summarize),
        "all_phase_top1_matches": sum(
            row["top1_match"] for row in rows_to_summarize),
        "decode_max_kld": max(
            row["kld_reference_to_candidate"] for row in decode_rows),
        "decode_top1_matches": sum(
            row["top1_match"] for row in decode_rows),
    }

  codecs: dict[str, Any] = {}
  for bits in CODEC_BITS:
    semantic_rows = [
        distribution_metrics(
            projected["i8"][index], projected[str(bits)][index], np)
        for index in range(DECODE_STEPS + 1)]
    carrier_rows = [
        distribution_metrics(actual[index], projected[str(bits)][index], np)
        for index in range(DECODE_STEPS + 1)]
    q8_carrier_rows = [
        distribution_metrics(actual[index], q8_projected[str(bits)][index], np)
        for index in range(DECODE_STEPS + 1)]
    codecs[str(bits)] = {
        "factor": float(128 // (1 << (bits - 1))),
        "versus_i8_semantic": semantic_rows,
        "versus_accepted_carrier_logits": carrier_rows,
        "q8_group256_versus_accepted_carrier_logits": q8_carrier_rows,
        "semantic_summary": decode_summary(semantic_rows),
        "carrier_summary": decode_summary(carrier_rows),
        "q8_group256_carrier_summary": decode_summary(q8_carrier_rows),
    }
  return {
      "projection_seconds": projection_seconds,
      "hidden_shape": list(hidden.shape),
      "accepted_carrier_logits_shape": list(actual.shape),
      "i8_semantic_vs_accepted_carrier": {
          "phases": semantic_alignment,
          **decode_summary(semantic_alignment),
      },
      "activation_codec": {
          "kind": "GPU half symmetric Q8",
          "group_size": 256,
          "quantize_scale": "half(127/max_abs)",
          "conversion": "round_to_even_char",
          "dequantize_scale": "half(1/quantize_scale)",
          "decode_hidden_reconstruction_relative_l2": [
              float(np.linalg.norm(q8_hidden[index] - hidden[index]) /
                    np.linalg.norm(hidden[index]))
              for index in range(1, DECODE_STEPS + 1)],
      },
      "codecs": codecs,
  }


def bandwidth_bound(
    facts: dict[str, Any], profile_worker: dict[str, Any],
    kill_number_ms: float,
) -> dict[str, Any]:
  profile_rows = [
      row for row in profile_worker.get("full_profile", [])
      if row.get("node_name") == LM_HEAD_NAME and
      row.get("status") == "Status.EXECUTED"]
  if len(profile_rows) != 1:
    raise ValueError(f"expected one executed LM-head profile row: {profile_rows}")
  profile_ms = float(profile_rows[0]["real_time_us"]) / 1000.0
  outputs = int(facts["weight"]["shape"][0])
  inputs = int(facts["weight"]["shape"][1])
  value_count = outputs * inputs
  rows = {}
  for bits in (8, *CODEC_BITS):
    weight_bytes = math.ceil(value_count * bits / 8.0)
    total_bytes = (
        weight_bytes + int(facts["scale"]["size"]) + inputs * 4 +
        outputs * 4)
    timing = {
        str(bandwidth): total_bytes / (bandwidth * 1e9) * 1000.0
        for bandwidth in REFERENCE_BANDWIDTHS_GBPS}
    conservative_floor = timing[str(CONSERVATIVE_BANDWIDTH_GBPS)]
    savings = profile_ms - conservative_floor
    rows[str(bits)] = {
        "packed_weight_bytes": weight_bytes,
        "weight_reduction_fraction_vs_i8": 1.0 - bits / 8.0,
        "scale_hidden_and_f32_output_bytes": total_bytes - weight_bytes,
        "conservative_total_bytes": total_bytes,
        "bandwidth_floor_ms": timing,
        "conservative_savings_ceiling_ms_vs_profile_row": savings,
        "conservative_ceiling_clears_kill_number": savings > kill_number_ms,
    }
  return {
      "profile_row": profile_rows[0],
      "profile_ms_nonadditive_source_bound": profile_ms,
      "profile_rows_are_nonadditive": True,
      "conservative_bandwidth_gbps": CONSERVATIVE_BANDWIDTH_GBPS,
      "conservative_bandwidth_basis": (
          "rounded below the accepted native Q6 rowstripe effective 107.8 "
          "GB/s observation and the 115 GB/s planning line"),
      "kill_number_ms_per_token": kill_number_ms,
      "codecs": rows,
  }


def write_summary(output: Path, metrics: dict[str, Any]) -> None:
  numeric = metrics.get("numeric", {})
  bandwidth = metrics.get("bandwidth_bound", {})
  lines = [
      "# OpenVINO LM-head signed-codec bound",
      "",
      f"- verdict: `{metrics['verdict']}`",
      f"- required checks passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- selected codec: `{metrics.get('selected_codec')}`",
      f"- worker count: `{metrics['gpu_workers_launched']}` candidate-only, stock `0`",
      f"- worker OOM observed: `{str(metrics['worker']['oom_observed']).lower()}`",
      f"- real hidden matrix: `{numeric.get('hidden_shape')}`",
      f"- LM-head profile row: `{bandwidth.get('profile_ms_nonadditive_source_bound')}` ms (non-additive source bound)",
      f"- kill-number: `{bandwidth.get('kill_number_ms_per_token')}` ms/token",
      "",
      "## Real-path codec screen",
      "",
      "| signed bits | decode Q8 max KLD | decode top-1 | conservative ceiling clears gap |",
      "|---:|---:|---:|:---:|",
  ]
  for bits in CODEC_BITS:
    row = numeric.get("codecs", {}).get(str(bits), {})
    bound = bandwidth.get("codecs", {}).get(str(bits), {})
    screen = row.get("q8_group256_carrier_summary", {})
    lines.append(
        f"| {bits} | {screen.get('decode_max_kld')} | "
        f"{screen.get('decode_top1_matches')}/17 | "
        f"{bound.get('conservative_ceiling_clears_kill_number')} |")
  lines.extend([
      "",
      "Admission is decode-only: phase0 is retained as diagnostic telemetry, "
      "while prefill stays on the existing I8 provider. This is a component "
      "funding gate, not a product speedup, an output512 row, or permission "
      "to omit full logits/KLD evidence.",
      "",
  ])
  (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  args = parse_args()
  if args.worker_config is not None:
    return worker_main(args.worker_config.resolve())

  import numpy as np

  output = args.output.resolve()
  raw = output / "raw" / "2k" / "candidate-pruned-head"
  raw.mkdir(parents=True, exist_ok=False)
  args.plugin = args.plugin.resolve()
  args.accepted_manifest = args.accepted_manifest.resolve()
  args.profile_metrics = args.profile_metrics.resolve()
  args.profile_worker = args.profile_worker.resolve()
  required = [
      MODEL_XML, MODEL_BIN, OV_PYTHON, args.plugin, CUSTOM_CONFIG, PROMPT,
      GRAPH_MODULE, BASE_GATE, args.accepted_manifest, args.profile_metrics,
      args.profile_worker, FRONTIER,
  ]
  missing = [str(path) for path in required if not path.is_file()]
  if missing:
    raise SystemExit("missing LM-head bound inputs: " + ", ".join(missing))

  git = git_state(output)
  concurrent = other_worker_pids()
  if concurrent:
    raise RuntimeError(f"concurrent OpenVINO worker detected: {concurrent}")
  manifest = load_json(args.accepted_manifest)
  profile_metrics = load_json(args.profile_metrics)
  profile_worker = load_json(args.profile_worker)
  frontier = load_json(FRONTIER)
  plugin_hash = sha256(args.plugin)
  expected_top1 = [int(row["top1"]) for row in profile_worker["phases"]]
  if len(expected_top1) != DECODE_STEPS + 1:
    raise ValueError("profile worker does not contain 18 phases")
  decode_tokens = expected_top1[:DECODE_STEPS]

  facts = inspect_ir_constants()
  worker = launch_worker(args, raw, decode_tokens)
  worker_result = worker["result"]
  distribution = weight_distribution(facts, np)
  numeric = numeric_projection(
      facts, worker_result, profile_worker, args.numeric_row_chunk, np)
  kill_number = float(frontier["goal_budget"]["per_token_ms"]["remaining_cut"])
  bandwidth = bandwidth_bound(facts, profile_worker, kill_number)

  eligible = []
  for bits in CODEC_BITS:
    row = numeric["codecs"][str(bits)]
    screen = row["q8_group256_carrier_summary"]
    bound = bandwidth["codecs"][str(bits)]
    if (screen["decode_max_kld"] <= KLD_LIMIT and
        screen["decode_top1_matches"] == DECODE_STEPS and
        bound["conservative_ceiling_clears_kill_number"]):
      eligible.append(bits)
  selected = min(eligible) if eligible else None
  phases = worker_result.get("phases", [])
  identity = {
      "expected_plugin_sha256": manifest.get("candidate_gpu_plugin_sha256"),
      "actual_plugin_sha256": plugin_hash,
      "plugin_match": plugin_hash == manifest.get("candidate_gpu_plugin_sha256"),
      "accepted_alias_enabled": manifest.get("alias_linear_state_assign"),
      "accepted_alias_scope": manifest.get("linear_state_alias_scope"),
      "accepted_fuse_linear_conv_state": manifest.get("fuse_linear_conv_state"),
      "profile_git": profile_metrics.get("git"),
  }
  checks = [
      check("repository_clean_at_gate", not git["dirty"], git=git),
      check("no_concurrent_openvino_worker_at_launch", not concurrent,
            concurrent=concurrent),
      check("accepted_plugin_and_carrier_identity_exact",
            identity["plugin_match"] and
            identity["accepted_alias_enabled"] is True and
            identity["accepted_alias_scope"] == "all" and
            identity["accepted_fuse_linear_conv_state"] is True,
            identity=identity),
      check("single_pruned_candidate_worker_completes_without_oom",
            worker["returncode"] == 0 and not worker["timed_out"] and
            not worker["memory_guard"]["tripped"] and
            not worker["oom_observed"], worker={
                key: worker[key] for key in (
                    "returncode", "timed_out", "memory_guard",
                    "oom_observed", "monitor")}),
      check("capture_uses_exact_clean_graph_shape",
            worker_result.get("source_summary", {}).get("custom_count_after") == 10 and
            worker_result.get("source_summary", {}).get("stock_sdpa_count_after") == 0 and
            worker_result.get("source_summary", {}).get("linear_conv_replacement_count") == 30 and
            worker_result.get("hot_state_self_bind_skipped") is True),
      check("lm_head_is_pruned_from_capture_runtime",
            worker_result.get("lm_head", {}).get(
                "compiled_runtime_lm_head_absent") is True),
      check("exact_18_real_hidden_vectors_are_finite",
            len(phases) == DECODE_STEPS + 1 and
            all(row.get("shape") == [2048] and row.get("finite") is True
                for row in phases)),
      check("locked_ir_lm_head_facts_are_exact",
            facts["weight"]["element_type"] == "i8" and
            facts["weight"]["shape"] == [248320, 2048] and
            facts["weight"]["size"] == 508559360 and
            facts["scale"]["element_type"] == "f16" and
            facts["scale"]["shape"] == [248320, 1] and
            facts["scale"]["size"] == 496640),
      check("exact_fixed_width_reduction_below_i8_is_closed",
            distribution["raw_exact_fixed_width_below_8_feasible"] is False,
            rows_fitting=distribution["rows_fitting_signed_width"]),
      check("semantic_i8_decode_projection_aligns_with_accepted_carrier",
            numeric["i8_semantic_vs_accepted_carrier"][
                "decode_top1_matches"] == DECODE_STEPS and
            numeric["i8_semantic_vs_accepted_carrier"][
                "decode_max_kld"] <= KLD_LIMIT,
            alignment=numeric["i8_semantic_vs_accepted_carrier"]),
      check("prefill_phase_is_diagnostic_and_excluded_from_decode_admission",
            numeric["i8_semantic_vs_accepted_carrier"][
                "prefill_phase"]["top1_match"] is True,
            prefill_phase=numeric["i8_semantic_vs_accepted_carrier"][
                "prefill_phase"],
            retained_prefill_provider="existing per-row I8 compressed FC"),
      check("only_real_decode_path_accurate_and_funded_codec_is_selected",
            selected is not None, eligible_codecs=eligible,
            selected_codec=selected),
      check("profile_row_is_not_added_to_other_event_rows",
            bandwidth["profile_rows_are_nonadditive"] is True),
  ]
  required_checks_passed = all(row["pass"] for row in checks)
  verdict = (
      f"fund_signed{selected}_decode_lm_head_rowstripe_component"
      if required_checks_passed and selected is not None else
      "park_lm_head_codec_route" if selected is None else "inconclusive")
  metrics = {
      "schema": SCHEMA,
      "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
      "workstream": WS,
      "git": git,
      "verdict": verdict,
      "required_checks_passed": required_checks_passed,
      "selected_codec": selected,
      "eligible_codecs": eligible,
      "provider_scope": "decode T=1 only; retain existing I8 prefill provider",
      "gpu_workers_launched": 1,
      "stock_worker_launched": False,
      "concurrent_worker_launched": False,
      "long_worker_launched": False,
      "identity": identity,
      "ir_facts": facts,
      "weight_distribution": distribution,
      "numeric": numeric,
      "bandwidth_bound": bandwidth,
      "worker": {key: value for key, value in worker.items()
                 if key != "result"},
      "worker_result_summary": {
          "openvino_version": worker_result.get("openvino_version"),
          "compile_ms": worker_result.get("compile_ms"),
          "prompt_tokens": worker_result.get("prompt_tokens"),
          "decode_tokens": worker_result.get("decode_tokens"),
          "lm_head": worker_result.get("lm_head"),
          "memory_samples": worker_result.get("memory_samples"),
          "hidden_sha256": [row.get("sha256") for row in phases],
      },
      "inputs": {
          "accepted_manifest": display_path(args.accepted_manifest),
          "accepted_manifest_sha256": sha256(args.accepted_manifest),
          "profile_metrics": display_path(args.profile_metrics),
          "profile_metrics_sha256": sha256(args.profile_metrics),
          "profile_worker": display_path(args.profile_worker),
          "profile_worker_sha256": sha256(args.profile_worker),
          "model_xml": display_path(MODEL_XML),
          "model_xml_sha256": sha256(MODEL_XML),
          "plugin": str(args.plugin),
          "plugin_sha256": plugin_hash,
      },
      "checks": checks,
  }
  write_json(output / "metrics.json", metrics)
  write_json(output / "manifest.json", {
      "schema": SCHEMA,
      "git": git,
      "candidate_gpu_plugin": str(args.plugin),
      "candidate_gpu_plugin_sha256": plugin_hash,
      "custom_config": display_path(CUSTOM_CONFIG),
      "custom_config_sha256": sha256(CUSTOM_CONFIG),
      "alias_linear_state_assign": True,
      "linear_state_alias_scope": "all",
      "fuse_linear_conv_state": True,
      "capture_prunes_lm_head": True,
      "provider_scope": "decode T=1 only; retain existing I8 prefill provider",
      "gpu_workers_launched": 1,
      "memory_preflight_gib": args.min_available_gib,
      "memory_abort_gib": args.abort_below_available_gib,
  })
  write_summary(output, metrics)
  print(json.dumps({
      "verdict": verdict, "required_checks_passed": required_checks_passed,
      "selected_codec": selected,
      "codec_rows": {
          str(bits): {
              "decode_q8_max_kld": numeric["codecs"][str(bits)][
                  "q8_group256_carrier_summary"]["decode_max_kld"],
              "decode_top1_matches": numeric["codecs"][str(bits)][
                  "q8_group256_carrier_summary"]["decode_top1_matches"],
              "ceiling_clears_gap": bandwidth["codecs"][str(bits)][
                  "conservative_ceiling_clears_kill_number"],
          } for bits in CODEC_BITS},
      "output": display_path(output),
  }, sort_keys=True))
  return 0 if required_checks_passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
