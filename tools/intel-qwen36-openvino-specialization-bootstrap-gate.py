#!/usr/bin/env python3
"""Freeze the stock OpenVINO ruler and prove an isolated no-op GPU candidate.

The default invocation is the OV0 contract run: three deterministic short
prompts plus the seven accepted long-context sentinel prompts.  The long rows
produce exactly 512 greedy tokens.  Stock and candidate execute in separate
OpenVINO Python processes with separate NEO caches.  Candidate inserts one
repository-owned ``IQ36Identity`` operation after the real language-model
logits and loads its OpenCL implementation through the GPU ``CONFIG_FILE``
property; stock reads the locked graph unchanged and never loads that file.

``--smoke`` is a bounded mechanism check only and can never close OV0.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-openvino-specialization-bootstrap-gate-v0"
OV_PYTHON = Path("/home/intel/ov/openvino_env/bin/python")
MODEL_DIR = Path("/home/intel/Qwen3.6-35B-A3B-ov")
MODEL_CONTRACT = ROOT / "contracts/qwen36-35b-a3b-openvino-u4-model-contract.json"
ACCEPTANCE = (
    ROOT / "benchmarks/intel-qwen36-35b-a3b-gguf-q4km/acceptance-matrix.json")
SHORT_PROMPTS = (
    ROOT / "benchmarks/intel-qwen36-35b-a3b-gguf-q4km/prompts/"
    "deterministic-greedy.jsonl")
SENTINEL_PROMPTS = (
    ROOT / "benchmarks/intel-qwen36-35b-a3b-gguf-q4km/prompts/"
    "long-context-sentinels.jsonl")
MATERIALIZATION = (
    ROOT / "output/r0-oracle-prompt-materialization-20260626T082201Z")
CUSTOM_CONFIG = ROOT / "engine/openvino/custom/iq36_identity.xml"
CUSTOM_SOURCE = ROOT / "engine/openvino/custom/iq36_identity.cl"
CORE_BUCKETS = (2048, 4096, 8192, 16384, 32768, 65536, 131072)
CORE_OUTPUT_TOKENS = 512
LONG_DISTRIBUTION_STEPS = (0, 1, 7, 63, 255, 511)
KLD_MAX = 0.005
TOP1_MIN = 0.99
COSINE_MIN = 0.999
PREFILL_KILL_NUMBER_MS = 40.896


def iso_now() -> str:
  return dt.datetime.now(dt.timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
  with path.open("w", encoding="utf-8") as handle:
    for row in rows:
      handle.write(json.dumps(row, sort_keys=True) + "\n")


def load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise ValueError(f"{path}: expected a JSON object")
  return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
  rows = []
  with path.open("r", encoding="utf-8") as handle:
    for line_number, line in enumerate(handle, start=1):
      line = line.strip()
      if not line:
        continue
      value = json.loads(line)
      if not isinstance(value, dict):
        raise ValueError(f"{path}:{line_number}: expected a JSON object")
      rows.append(value)
  return rows


def sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def command_record(command: list[str], timeout_s: int = 30) -> dict[str, Any]:
  try:
    run = subprocess.run(
        command, cwd=ROOT, check=False, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=timeout_s)
    return {
        "command": command,
        "returncode": run.returncode,
        "stderr": run.stderr,
        "stdout": run.stdout,
        "timed_out": False,
    }
  except subprocess.TimeoutExpired as exc:
    return {
        "command": command,
        "returncode": None,
        "stderr": str(exc.stderr or ""),
        "stdout": str(exc.stdout or ""),
        "timed_out": True,
    }


def git_state(out_dir: Path) -> dict[str, Any]:
  def git(*parts: str) -> str:
    run = subprocess.run(
        ["git", *parts], cwd=ROOT, check=False, capture_output=True,
        text=True, encoding="utf-8", errors="replace")
    return run.stdout.strip() if run.returncode == 0 else ""

  dirty = git("status", "--porcelain").splitlines()
  try:
    relative = str(out_dir.relative_to(ROOT))
  except ValueError:
    relative = ""
  dirty = [line for line in dirty if not relative or relative not in line]
  return {
      "commit": git("rev-parse", "HEAD"),
      "dirty": bool(dirty),
      "dirty_paths": dirty,
  }


def _jsonable(value: Any) -> Any:
  if value is None or isinstance(value, (bool, int, float, str)):
    return value
  if isinstance(value, bytes):
    return value.hex()
  if isinstance(value, dict):
    return {str(key): _jsonable(item) for key, item in value.items()}
  if isinstance(value, (list, tuple, set)):
    return [_jsonable(item) for item in value]
  if hasattr(value, "tolist"):
    try:
      return _jsonable(value.tolist())
    except Exception:
      pass
  return str(value)


def _core_properties(core: Any, device: str) -> dict[str, Any]:
  result: dict[str, Any] = {}
  supported = core.get_property(device, "SUPPORTED_PROPERTIES")
  items = supported.items() if isinstance(supported, dict) else (
      (item, "unknown") for item in supported)
  for key, mutability in items:
    name = str(key)
    entry: dict[str, Any] = {"mutability": str(mutability)}
    if str(mutability) == "WO":
      entry["value"] = "write_only"
    else:
      try:
        entry["value"] = _jsonable(core.get_property(device, name))
      except Exception as exc:
        entry["error"] = repr(exc)
    result[name] = entry
  return result


def _compiled_properties(compiled: Any) -> dict[str, Any]:
  result: dict[str, Any] = {}
  supported = compiled.get_property("SUPPORTED_PROPERTIES")
  items = supported.items() if isinstance(supported, dict) else (
      (item, "unknown") for item in supported)
  for key, mutability in items:
    name = str(key)
    entry: dict[str, Any] = {"mutability": str(mutability)}
    if str(mutability) == "WO":
      entry["value"] = "write_only"
    else:
      try:
        entry["value"] = _jsonable(compiled.get_property(name))
      except Exception as exc:
        entry["error"] = repr(exc)
    result[name] = entry
  return result


def _state_schema(request: Any) -> list[dict[str, Any]]:
  rows = []
  for state in request.query_state():
    try:
      tensor = state.state
      element_type = str(tensor.element_type)
      shape = [int(value) for value in tensor.shape]
      materialized = True
      error = None
    except Exception as exc:
      # KV dimensions are dynamic before the first InferRequest.  The state
      # handles and names are nevertheless valid; exact dtypes/shapes are
      # captured again after the 2k and 128k cases materialize them.
      element_type = None
      shape = None
      materialized = False
      error = repr(exc)
    row = {
        "element_type": element_type,
        "materialized": materialized,
        "name": state.name,
        "shape": shape,
    }
    if error is not None:
      row["materialization_error"] = error
    rows.append(row)
  return sorted(rows, key=lambda row: row["name"])


def _state_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
  families: dict[str, int] = {}
  shapes: dict[str, int] = {}
  for row in rows:
    match = re.search(r"past\.(conv|ssm|key|value)\.", row["name"])
    family = match.group(1) if match else "other"
    families[family] = families.get(family, 0) + 1
    key = f"{row['element_type']}:{row['shape']}"
    shapes[key] = shapes.get(key, 0) + 1
  return {
      "count": len(rows),
      "family_counts": dict(sorted(families.items())),
      "shape_counts": dict(sorted(shapes.items())),
  }


def _profile_summary(request: Any) -> dict[str, Any]:
  groups: dict[tuple[str, str, str], dict[str, Any]] = {}
  focus = []
  focus_pattern = re.compile(
      r"iq36|gated_delta|transpose|dynamicquant|dynamic_quant|moe|"
      r"scaled_dot_product|indirectsdpa", re.IGNORECASE)
  for item in request.profiling_info:
    node_name = str(item.node_name)
    node_type = str(item.node_type)
    exec_type = str(item.exec_type)
    status = str(item.status)
    real_us = float(item.real_time.total_seconds() * 1_000_000.0)
    key = (node_type, exec_type, status)
    group = groups.setdefault(key, {
        "count": 0,
        "exec_type": exec_type,
        "node_type": node_type,
        "real_time_us": 0.0,
        "status": status,
    })
    group["count"] += 1
    group["real_time_us"] += real_us
    if focus_pattern.search(f"{node_name} {node_type} {exec_type}"):
      focus.append({
          "exec_type": exec_type,
          "node_name": node_name,
          "node_type": node_type,
          "real_time_us": real_us,
          "status": status,
      })
  rows = sorted(
      groups.values(),
      key=lambda row: (-float(row["real_time_us"]), row["node_type"],
                       row["exec_type"]),
  )
  return {
      "focus_rows": focus,
      "grouped_rows": rows,
      "profile_row_count": sum(int(row["count"]) for row in rows),
      "profile_total_real_time_us": sum(
          float(row["real_time_us"]) for row in rows),
  }


def _top8(logits: Any, np: Any) -> list[dict[str, Any]]:
  indices = np.argpartition(logits, -8)[-8:]
  indices = sorted(
      (int(index) for index in indices),
      key=lambda index: float(logits[index]), reverse=True)
  return [{"id": index, "value": float(logits[index])} for index in indices]


def _worker_custom_class(ov: Any) -> type:
  class IQ36Identity(ov.Op):
    def __init__(self, inputs: Any = None):
      super().__init__(self, inputs)
      self.constructor_validate_and_infer_types()

    def validate_and_infer_types(self) -> None:
      self.set_output_type(
          0, self.get_input_element_type(0), self.get_input_partial_shape(0))

    def clone_with_new_inputs(self, new_inputs: Any) -> Any:
      return IQ36Identity(new_inputs)

    def visit_attributes(self, visitor: Any) -> bool:
      return True

  return IQ36Identity


def _synthetic_model(ov: Any, custom: bool) -> Any:
  parameter = ov.opset13.parameter(
      [1, 1, 1, 248320], ov.Type.f32, name="logits")
  output = parameter.output(0)
  if custom:
    identity = _worker_custom_class(ov)([output])
    identity.set_friendly_name("iq36_custom_identity")
    output = identity.output(0)
  return ov.Model([output], [parameter], "iq36_identity_mechanism")


def _mechanism_worker(cfg: dict[str, Any]) -> dict[str, Any]:
  import numpy as np
  import openvino as ov

  device = cfg["device"]
  custom_config = str(Path(cfg["custom_config"]).resolve())
  stock_core = ov.Core()
  stock_before = _core_properties(stock_core, device)
  stock_compiled = stock_core.compile_model(
      _synthetic_model(ov, False), device,
      {"PERFORMANCE_HINT": "LATENCY", "PERF_COUNT": True})
  stock_request = stock_compiled.create_infer_request()

  candidate_core = ov.Core()
  candidate_core.set_property(device, {"CONFIG_FILE": custom_config})
  candidate_after = _core_properties(candidate_core, device)
  candidate_compiled = candidate_core.compile_model(
      _synthetic_model(ov, True), device,
      {"PERFORMANCE_HINT": "LATENCY", "PERF_COUNT": True})
  candidate_request = candidate_compiled.create_infer_request()

  source = np.sin(np.arange(248320, dtype=np.float32) / 101.0).reshape(
      1, 1, 1, 248320)

  def infer(request: Any, compiled: Any) -> tuple[Any, float]:
    started = time.perf_counter_ns()
    outputs = request.infer({compiled.input(0): source})
    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    return np.asarray(outputs[compiled.output(0)]), elapsed_ms

  for _ in range(int(cfg["warmup"])):
    infer(stock_request, stock_compiled)
    infer(candidate_request, candidate_compiled)

  blocks = []
  all_finite = True
  for block_index in range(int(cfg["blocks"])):
    order = (
        ("stock", stock_request, stock_compiled),
        ("candidate", candidate_request, candidate_compiled),
        ("candidate", candidate_request, candidate_compiled),
        ("stock", stock_request, stock_compiled),
    )
    rows = []
    for label, request, compiled in order:
      output, wall_ms = infer(request, compiled)
      all_finite = all_finite and bool(np.isfinite(output).all())
      rows.append({"label": label, "wall_ms": wall_ms})
    stock_times = [row["wall_ms"] for row in rows if row["label"] == "stock"]
    candidate_times = [
        row["wall_ms"] for row in rows if row["label"] == "candidate"]
    stock_ms = statistics.median(stock_times)
    candidate_ms = statistics.median(candidate_times)
    blocks.append({
        "block": block_index,
        "candidate_ms": candidate_ms,
        "overhead_ms": candidate_ms - stock_ms,
        "rows": rows,
        "stock_ms": stock_ms,
    })

  overheads = [float(row["overhead_ms"]) for row in blocks]
  mean_overhead = statistics.mean(overheads)
  # Twenty paired blocks match the component minimum in the acceptance
  # contract.  1.729133 is the one-sided 95% Student-t critical value for 19
  # degrees of freedom.
  t_critical = 1.729133
  standard_error = (
      statistics.stdev(overheads) / math.sqrt(len(overheads))
      if len(overheads) > 1 else 0.0)
  overhead_upper_95 = mean_overhead + t_critical * standard_error

  no_config_error = ""
  try:
    ov.Core().compile_model(_synthetic_model(ov, True), device)
  except Exception as exc:
    no_config_error = repr(exc)

  candidate_output, _ = infer(candidate_request, candidate_compiled)
  source64 = source.astype(np.float64).reshape(-1)
  candidate64 = candidate_output.astype(np.float64).reshape(-1)
  source_norm = float(np.linalg.norm(source64))
  candidate_norm = float(np.linalg.norm(candidate64))
  denominator = source_norm * candidate_norm
  cosine = (
      float(np.dot(source64, candidate64) / denominator)
      if denominator else 1.0)
  relative_l2 = float(
      np.linalg.norm(candidate64 - source64) / source_norm
      if source_norm else np.linalg.norm(candidate64 - source64))
  custom_profile = _profile_summary(candidate_request)
  custom_rows = [
      row for row in custom_profile["focus_rows"]
      if row["node_type"] == "IQ36Identity"
      or "iq36_custom_identity" in row["node_name"]]
  return {
      "blocks": blocks,
      "candidate_config_after": candidate_after.get("CONFIG_FILE"),
      "candidate_output_sha256": hashlib.sha256(
          np.ascontiguousarray(candidate_output).tobytes()).hexdigest(),
      "custom_profile_rows": custom_rows,
      "all_outputs_finite": all_finite,
      "candidate_cosine": cosine,
      "candidate_max_abs": float(np.max(np.abs(candidate64 - source64))),
      "candidate_relative_l2": relative_l2,
      "exact_outputs": bool(np.array_equal(candidate_output, source)),
      "no_config_compile_error": no_config_error,
      "no_config_compile_failed": bool(no_config_error),
      "openvino_runtime_version": ov.get_version(),
      "overhead_mean_ms": mean_overhead,
      "overhead_median_ms": statistics.median(overheads),
      "overhead_upper_one_sided_95_ms": overhead_upper_95,
      "paired_blocks": len(blocks),
      "stock_config_before": stock_before.get("CONFIG_FILE"),
  }


def _make_inputs(
    embedding: Any, token_ids: list[int], start: int, total: int,
    np: Any,
) -> dict[str, Any]:
  ids = np.asarray([token_ids], dtype=np.int64)
  embedded = np.asarray(
      embedding({embedding.input(0): ids})[embedding.output(0)])
  if embedded.dtype != np.float32:
    embedded = embedded.astype(np.float32)
  count = len(token_ids)
  positions = np.arange(start, start + count, dtype=np.int64)
  return {
      "attention_mask": np.ones((1, total), dtype=np.int64),
      "beam_idx": np.zeros((1,), dtype=np.int32),
      "inputs_embeds": embedded,
      "position_ids": np.tile(positions, (4, 1)).reshape(4, 1, count),
  }


def _run_case(
    case: dict[str, Any], mode: str, tokenizer: Any, embedding: Any,
    language: Any, request: Any, logits_dir: Path,
    reference_ids: list[int] | None, prefill_chunk_tokens: int, np: Any,
) -> dict[str, Any]:
  if case.get("prompt_path"):
    prompt_path = Path(case["prompt_path"])
    prompt = prompt_path.read_text(encoding="utf-8")
    prompt_sha = sha256_file(prompt_path)
  else:
    prompt = str(case["prompt"])
    prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
  encoded = tokenizer.encode(prompt).input_ids
  prompt_ids = np.asarray(encoded.data).reshape(-1).astype(np.int64)
  output_tokens = int(case["output_tokens"])
  if reference_ids is not None and len(reference_ids) != output_tokens:
    raise RuntimeError(
        f"{case['case_id']}: reference length {len(reference_ids)} != "
        f"{output_tokens}")

  request.reset_state()
  generated = []
  checkpoint_rows = []
  decode_wall_ms = []
  prefill_profile = None
  prefill_wall_ms = None
  prefill_chunk_count = 0
  for step in range(output_tokens):
    if step == 0:
      outputs = None
      prefill_wall_ms = 0.0
      for chunk_start in range(0, len(prompt_ids), prefill_chunk_tokens):
        chunk_end = min(chunk_start + prefill_chunk_tokens, len(prompt_ids))
        fed = [int(value) for value in prompt_ids[chunk_start:chunk_end]]
        started = time.perf_counter_ns()
        outputs = request.infer(_make_inputs(
            embedding, fed, chunk_start, chunk_end, np))
        prefill_wall_ms += (
            time.perf_counter_ns() - started) / 1_000_000.0
        prefill_chunk_count += 1
      if outputs is None:
        raise RuntimeError(f"{case['case_id']}: empty prompt")
      wall_ms = prefill_wall_ms
    else:
      fed_id = (
          int(reference_ids[step - 1]) if reference_ids is not None
          else int(generated[step - 1]))
      fed = [fed_id]
      start = len(prompt_ids) + step - 1
      total = len(prompt_ids) + step
      started = time.perf_counter_ns()
      outputs = request.infer(_make_inputs(
          embedding, fed, start, total, np))
      wall_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    logits = np.asarray(outputs[language.output(0)], dtype=np.float32)[0, -1]
    greedy_id = int(np.argmax(logits))
    generated.append(greedy_id)
    if step == 0:
      prefill_wall_ms = wall_ms
      if case.get("capture_profile"):
        prefill_profile = _profile_summary(request)
    else:
      decode_wall_ms.append(wall_ms)
    if step in set(int(value) for value in case["distribution_steps"]):
      path = logits_dir / f"{case['case_id']}-step{step:04d}.f32"
      contiguous = np.ascontiguousarray(logits, dtype="<f4")
      contiguous.tofile(path)
      checkpoint_rows.append({
          "byte_count": path.stat().st_size,
          "file": str(path),
          "l2_norm": float(np.linalg.norm(logits.astype(np.float64))),
          "sha256": sha256_file(path),
          "shape": [int(value) for value in logits.shape],
          "step": step,
          "top8": _top8(logits, np),
      })

  decoded = str(tokenizer.decode(generated, skip_special_tokens=False))
  expected_answer = case.get("expected_answer")
  state_rows = _state_schema(request) if case.get("capture_state") else []
  return {
      "case_id": case["case_id"],
      "decode_wall_ms_median": (
          statistics.median(decode_wall_ms) if decode_wall_ms else None),
      "decoded_text": decoded,
      "distribution_checkpoints": checkpoint_rows,
      "expected_answer": expected_answer,
      "expected_input_tokens": case.get("expected_input_tokens"),
      "generated_token_count": len(generated),
      "generated_token_ids": generated,
      "generated_token_ids_sha256": hashlib.sha256(
          np.asarray(generated, dtype="<u4").tobytes()).hexdigest(),
      "input_token_count": int(len(prompt_ids)),
      "input_token_ids_sha256": hashlib.sha256(
          np.asarray(prompt_ids, dtype="<u4").tobytes()).hexdigest(),
      "mode": mode,
      "output_tokens": output_tokens,
      "prefill_chunk_count": prefill_chunk_count,
      "prefill_chunk_tokens": prefill_chunk_tokens,
      "prefill_profile": prefill_profile,
      "prefill_wall_ms": prefill_wall_ms,
      "prompt_sha256": prompt_sha,
      "sentinel_pass": (
          expected_answer in decoded if isinstance(expected_answer, str)
          else None),
      "state_schema_after": state_rows,
      "state_summary_after": _state_summary(state_rows) if state_rows else None,
      "teacher_forced_from_stock": reference_ids is not None,
  }


def _model_worker(cfg: dict[str, Any]) -> dict[str, Any]:
  import numpy as np
  import openvino as ov
  import openvino_genai as ov_genai

  mode = cfg["mode"]
  device = cfg["device"]
  model_dir = Path(cfg["model_dir"])
  logits_dir = Path(cfg["logits_dir"])
  logits_dir.mkdir(parents=True, exist_ok=True)
  reference: dict[str, Any] = {}
  if cfg.get("reference_result"):
    stock = load_json(Path(cfg["reference_result"]))
    reference = {row["case_id"]: row for row in stock.get("cases", [])}

  core = ov.Core()
  core_before = _core_properties(core, device)
  custom_config = None
  if mode == "candidate":
    custom_config = str(Path(cfg["custom_config"]).resolve())
    core.set_property(device, {"CONFIG_FILE": custom_config})
  core_after = _core_properties(core, device)

  embedding_source = core.read_model(
      str(model_dir / "openvino_text_embeddings_model.xml"))
  embedding_started = time.perf_counter()
  embedding = core.compile_model(
      embedding_source, "CPU", {"PERFORMANCE_HINT": "LATENCY"})
  embedding_compile_ms = (time.perf_counter() - embedding_started) * 1000.0

  source = core.read_model(str(model_dir / "openvino_language_model.xml"))
  source_op_count = len(source.get_ops())
  if mode == "candidate":
    # Keep the product logits path byte-for-byte untouched.  A one-element
    # sidecar branch from beam_idx proves that this *real compiled language
    # model* can select repository OpenCL without blocking the GPU plugin's
    # LM-head/output optimizations.  OV1 will replace a real GatedDeltaNet
    # boundary only after its component oracle is available.
    beam_parameter = next(
        parameter for parameter in source.get_parameters()
        if "beam_idx" in parameter.output(0).get_names())
    beam_f32 = ov.opset13.convert(beam_parameter, ov.Type.f32)
    probe_shape = ov.opset13.constant(
        [1, 1, 1, -1], dtype=np.int64)
    probe_input = ov.opset13.reshape(beam_f32, probe_shape, False)
    identity = _worker_custom_class(ov)([probe_input.output(0)])
    identity.set_friendly_name("iq36_custom_identity")
    identity.output(0).get_tensor().set_names({"iq36_custom_probe"})
    probe_result = ov.opset13.result(identity.output(0))
    probe_result.set_friendly_name("iq36_custom_probe_result")
    source.add_results([probe_result])
    source.validate_nodes_and_infer_types()
  custom_node_count = sum(
      node.get_type_name() == "IQ36Identity" for node in source.get_ops())

  compile_config = {
      "DYNAMIC_QUANTIZATION_GROUP_SIZE": 256,
      "PERFORMANCE_HINT": "LATENCY",
      "PERF_COUNT": True,
  }
  compile_started = time.perf_counter()
  language = core.compile_model(source, device, compile_config)
  language_compile_ms = (time.perf_counter() - compile_started) * 1000.0
  request = language.create_infer_request()
  request.reset_state()
  initial_state = _state_schema(request)
  memory_before = None
  try:
    memory_before = _jsonable(core.get_property(device, "GPU_MEMORY_STATISTICS"))
  except Exception as exc:
    memory_before = {"error": repr(exc)}

  tokenizer = ov_genai.Tokenizer(str(model_dir))
  cases = []
  for index, case in enumerate(cfg["cases"], start=1):
    reference_ids = None
    if mode == "candidate":
      reference_row = reference.get(case["case_id"])
      if reference_row is None:
        raise RuntimeError(f"stock result missing {case['case_id']}")
      reference_ids = [
          int(value) for value in reference_row["generated_token_ids"]]
    case_started = time.perf_counter()
    row = _run_case(
        case, mode, tokenizer, embedding, language, request, logits_dir,
        reference_ids, int(cfg["prefill_chunk_tokens"]), np)
    row["case_wall_ms"] = (time.perf_counter() - case_started) * 1000.0
    cases.append(row)
    print(json.dumps({
        "case": row["case_id"],
        "event": "case_complete",
        "index": index,
        "input_tokens": row["input_token_count"],
        "mode": mode,
        "output_tokens": row["generated_token_count"],
        "prefill_wall_ms": row["prefill_wall_ms"],
    }), flush=True)

  memory_after = None
  try:
    memory_after = _jsonable(core.get_property(device, "GPU_MEMORY_STATISTICS"))
  except Exception as exc:
    memory_after = {"error": repr(exc)}
  return {
      "cases": cases,
      "compile_config": compile_config,
      "compiled_properties": _compiled_properties(language),
      "core_properties_after": core_after,
      "core_properties_before": core_before,
      "custom_config": custom_config,
      "custom_node_count": custom_node_count,
      "device": device,
      "embedding_compile_ms": embedding_compile_ms,
      "initial_state_schema": initial_state,
      "initial_state_summary": _state_summary(initial_state),
      "language_compile_ms": language_compile_ms,
      "memory_after": memory_after,
      "memory_before": memory_before,
      "mode": mode,
      "openvino_genai_version": ov_genai.__version__,
      "openvino_runtime_version": ov.get_version(),
      "source_op_count": source_op_count,
  }


def worker_main(config_path: Path) -> int:
  cfg = load_json(config_path)
  started = time.perf_counter()
  if cfg["worker_kind"] == "mechanism":
    payload = _mechanism_worker(cfg)
  elif cfg["worker_kind"] == "model":
    payload = _model_worker(cfg)
  else:
    raise ValueError(f"unknown worker kind: {cfg['worker_kind']}")
  payload["worker_wall_ms"] = (time.perf_counter() - started) * 1000.0
  write_json(Path(cfg["result_path"]), payload)
  print(json.dumps({
      "event": "worker_complete",
      "kind": cfg["worker_kind"],
      "mode": cfg.get("mode"),
      "result_path": cfg["result_path"],
  }), flush=True)
  return 0


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out-dir", type=Path, required=True)
  parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
  parser.add_argument("--model-contract", type=Path, default=MODEL_CONTRACT)
  parser.add_argument("--acceptance", type=Path, default=ACCEPTANCE)
  parser.add_argument("--materialization-dir", type=Path,
                      default=MATERIALIZATION)
  parser.add_argument("--short-prompts", type=Path, default=SHORT_PROMPTS)
  parser.add_argument("--sentinel-prompts", type=Path,
                      default=SENTINEL_PROMPTS)
  parser.add_argument("--custom-config", type=Path, default=CUSTOM_CONFIG)
  parser.add_argument("--custom-source", type=Path, default=CUSTOM_SOURCE)
  parser.add_argument("--openvino-python", type=Path, default=OV_PYTHON)
  parser.add_argument("--device", default="GPU")
  parser.add_argument("--timeout-s", type=int, default=2400)
  parser.add_argument("--prefill-chunk-tokens", type=int, default=1024)
  parser.add_argument("--smoke", action="store_true")
  parser.add_argument("--smoke-bucket", type=int, choices=CORE_BUCKETS,
                      default=8192)
  parser.add_argument(
      "--reanalyze-from", type=Path,
      help="Reuse a completed raw OV0 artifact; never reruns model workers.")
  args = parser.parse_args()
  if args.prefill_chunk_tokens <= 0 or args.timeout_s <= 0:
    parser.error("timeout and prefill chunk size must be positive")
  return args


def build_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
  short_rows = load_jsonl(args.short_prompts)
  sentinel_rows = {
      row["id"]: row for row in load_jsonl(args.sentinel_prompts)}
  materialized = {
      row["case_id"]: row for row in load_jsonl(
          args.materialization_dir / "materialized-prompts.jsonl")}
  cases = []
  for row in short_rows:
    output_tokens = 8 if args.smoke else int(row["max_new_tokens"])
    cases.append({
        "capture_profile": row["id"] == "short_math_001",
        "capture_state": False,
        "case_id": row["id"],
        "distribution_steps": list(range(output_tokens)),
        "expected_answer": None,
        "expected_input_tokens": None,
        "kind": "short",
        "output_tokens": output_tokens,
        "prompt": row["prompt"],
    })
  for bucket in CORE_BUCKETS:
    suffix = f"{bucket // 1024:03d}k"
    case_id = f"sentinel_{suffix}"
    spec = sentinel_rows.get(case_id)
    source = materialized.get(case_id)
    if spec is None or source is None:
      raise ValueError(f"missing sentinel definition/materialization: {case_id}")
    prompt_path = ROOT / str(source["materialized_prompt_path"])
    if not prompt_path.is_file():
      raise ValueError(f"missing materialized prompt: {prompt_path}")
    if int(source["observed_prompt_tokens"]) != bucket:
      raise ValueError(f"{case_id}: materialized token count mismatch")
    if sha256_file(prompt_path) != source["prompt_file_sha256"]:
      raise ValueError(f"{case_id}: materialized prompt digest mismatch")
    output_tokens = 64 if args.smoke else CORE_OUTPUT_TOKENS
    distribution_steps = sorted(set(
        value for value in (*LONG_DISTRIBUTION_STEPS, output_tokens - 1)
        if 0 <= value < output_tokens))
    cases.append({
        "capture_profile": bucket in (8192, 131072),
        "capture_state": bucket in (2048, 131072),
        "case_id": case_id,
        "distribution_steps": distribution_steps,
        "expected_answer": spec["expected_answer"],
        "expected_input_tokens": bucket,
        "kind": "sentinel",
        "output_tokens": output_tokens,
        "prompt_path": str(prompt_path),
    })
  if args.smoke:
    keep = {
        "short_math_001",
        f"sentinel_{args.smoke_bucket // 1024:03d}k",
    }
    cases = [case for case in cases if case["case_id"] in keep]
  return cases


def capture_model_identity(
    model_dir: Path, contract_path: Path,
) -> dict[str, Any]:
  contract = load_json(contract_path)
  locked = contract["product_model"]["locked_files"]
  rows = []
  for relative, expected in locked.items():
    path = model_dir / relative
    exists = path.is_file()
    observed_bytes = path.stat().st_size if exists else None
    observed_sha = sha256_file(path) if exists else None
    rows.append({
        "bytes_match": observed_bytes == expected["bytes"],
        "exists": exists,
        "expected_bytes": expected["bytes"],
        "expected_sha256": expected["sha256"],
        "file": relative,
        "observed_bytes": observed_bytes,
        "observed_sha256": observed_sha,
        "sha256_match": observed_sha == expected["sha256"],
    })
  return {
      "contract": str(contract_path.resolve()),
      "files": rows,
      "model_dir": str(model_dir.resolve()),
      "required_checks_passed": all(
          row["exists"] and row["bytes_match"] and row["sha256_match"]
          for row in rows),
  }


def capture_host() -> dict[str, Any]:
  commands = {
      "dpkg_gpu": [
          "dpkg-query", "-W", "-f=${Package} ${Version}\\n",
          "intel-opencl-icd", "intel-level-zero-gpu", "libze-intel-gpu1",
          "libze1"],
      "lscpu": ["lscpu", "-J"],
      "lspci": ["lspci", "-nn"],
      "uname": ["uname", "-a"],
  }
  return {
      "captured_at": iso_now(),
      "commands": {name: command_record(command) for name, command in commands.items()},
      "hostname": platform.node(),
      "platform": platform.platform(),
      "python": sys.version,
  }


def run_worker(
    args: argparse.Namespace, name: str, config: dict[str, Any], raw_dir: Path,
) -> dict[str, Any]:
  worker_dir = raw_dir / name
  worker_dir.mkdir(parents=True, exist_ok=False)
  cache_dir = worker_dir / "neo-cache"
  cache_dir.mkdir()
  result_path = worker_dir / "worker-result.json"
  config.update({
      "device": args.device,
      "result_path": str(result_path),
  })
  config_path = worker_dir / "worker-config.json"
  write_json(config_path, config)
  command = [
      str(args.openvino_python), str(Path(__file__).resolve()),
      "__worker", str(config_path)]
  environment = os.environ.copy()
  environment.update({
      "NEO_CACHE_DIR": str(cache_dir),
      "NEO_CACHE_MAX_SIZE": str(4 * 1024 * 1024 * 1024),
      "NEO_CACHE_PERSISTENT": "1",
  })
  started = time.perf_counter()
  try:
    run = subprocess.run(
        command, cwd=ROOT, check=False, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=args.timeout_s,
        env=environment)
    timed_out = False
  except subprocess.TimeoutExpired as exc:
    run = None
    timed_out = True
    stdout = str(exc.stdout or "")
    stderr = str(exc.stderr or "")
  else:
    stdout = run.stdout
    stderr = run.stderr
  record = {
      "command": command,
      "environment": {
          key: environment[key] for key in (
              "NEO_CACHE_DIR", "NEO_CACHE_MAX_SIZE", "NEO_CACHE_PERSISTENT")},
      "returncode": run.returncode if run is not None else None,
      "stderr": stderr,
      "stdout": stdout,
      "timed_out": timed_out,
      "wall_ms": (time.perf_counter() - started) * 1000.0,
  }
  write_json(worker_dir / "run.json", record)
  print(json.dumps({
      "event": "subprocess_complete",
      "name": name,
      "returncode": record["returncode"],
      "timed_out": timed_out,
      "wall_ms": record["wall_ms"],
  }), flush=True)
  if result_path.is_file():
    record["result"] = load_json(result_path)
  else:
    record["result"] = {}
  return record


def _logsumexp(values: Any, np: Any) -> float:
  maximum = float(np.max(values))
  return maximum + math.log(float(np.exp(values - maximum).sum()))


def compare_distributions(
    stock: dict[str, Any], candidate: dict[str, Any], np: Any,
) -> list[dict[str, Any]]:
  candidate_cases = {
      row["case_id"]: row for row in candidate.get("cases", [])}
  rows = []
  for stock_case in stock.get("cases", []):
    candidate_case = candidate_cases.get(stock_case["case_id"])
    if candidate_case is None:
      continue
    candidate_steps = {
        int(row["step"]): row
        for row in candidate_case["distribution_checkpoints"]}
    for stock_step in stock_case["distribution_checkpoints"]:
      step = int(stock_step["step"])
      candidate_step = candidate_steps.get(step)
      if candidate_step is None:
        continue
      stock_logits = np.fromfile(stock_step["file"], dtype="<f4")
      candidate_logits = np.fromfile(candidate_step["file"], dtype="<f4")
      same_shape = stock_logits.shape == candidate_logits.shape
      finite = bool(
          same_shape and np.isfinite(stock_logits).all()
          and np.isfinite(candidate_logits).all())
      if finite:
        stock64 = stock_logits.astype(np.float64)
        candidate64 = candidate_logits.astype(np.float64)
        stock_log_z = _logsumexp(stock64, np)
        candidate_log_z = _logsumexp(candidate64, np)
        stock_log_p = stock64 - stock_log_z
        candidate_log_p = candidate64 - candidate_log_z
        stock_p = np.exp(stock_log_p)
        kld = float(np.sum(stock_p * (stock_log_p - candidate_log_p)))
        denominator = float(
            np.linalg.norm(stock64) * np.linalg.norm(candidate64))
        cosine = (
            float(np.dot(stock64, candidate64) / denominator)
            if denominator else 1.0)
        stock_norm = float(np.linalg.norm(stock64))
        relative_l2 = float(
            np.linalg.norm(candidate64 - stock64) / stock_norm
            if stock_norm else np.linalg.norm(candidate64 - stock64))
        max_abs = float(np.max(np.abs(candidate64 - stock64)))
        stock_top1 = int(np.argmax(stock_logits))
        candidate_top1 = int(np.argmax(candidate_logits))
      else:
        kld = cosine = relative_l2 = max_abs = None
        stock_top1 = candidate_top1 = None
      rows.append({
          "candidate_top1": candidate_top1,
          "case_id": stock_case["case_id"],
          "cosine": cosine,
          "finite": finite,
          "kld_stock_to_candidate": kld,
          "max_abs": max_abs,
          "relative_l2": relative_l2,
          "same_shape": same_shape,
          "step": step,
          "stock_top1": stock_top1,
          "top1_match": stock_top1 == candidate_top1,
      })
  return rows


def _property_value(properties: dict[str, Any], name: str) -> Any:
  entry = properties.get(name)
  return entry.get("value") if isinstance(entry, dict) else None


def analyze(
    args: argparse.Namespace, git: dict[str, Any], model_identity: dict[str, Any],
    cases: list[dict[str, Any]], mechanism_run: dict[str, Any],
    stock_run: dict[str, Any], candidate_run: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
  import numpy as np

  acceptance = load_json(args.acceptance)
  stock = stock_run["result"]
  candidate = candidate_run["result"]
  mechanism = mechanism_run["result"]
  distributions = compare_distributions(stock, candidate, np)
  stock_cases = {row["case_id"]: row for row in stock.get("cases", [])}
  candidate_cases = {
      row["case_id"]: row for row in candidate.get("cases", [])}
  token_rows = []
  for case in cases:
    stock_case = stock_cases.get(case["case_id"], {})
    candidate_case = candidate_cases.get(case["case_id"], {})
    token_rows.append({
        "candidate_count": candidate_case.get("generated_token_count"),
        "case_id": case["case_id"],
        "expected_answer": case.get("expected_answer"),
        "expected_input_tokens": case.get("expected_input_tokens"),
        "input_count_exact": (
            case.get("expected_input_tokens") is None
            or (
                stock_case.get("input_token_count")
                == case.get("expected_input_tokens")
                == candidate_case.get("input_token_count"))),
        "sentinel_candidate_pass": candidate_case.get("sentinel_pass"),
        "sentinel_stock_pass": stock_case.get("sentinel_pass"),
        "stock_count": stock_case.get("generated_token_count"),
        "token_ids_exact": (
            stock_case.get("generated_token_ids")
            == candidate_case.get("generated_token_ids")
            and stock_case.get("generated_token_count") == case["output_tokens"]
            and candidate_case.get("generated_token_count")
            == case["output_tokens"]),
    })

  focus_rows = []
  for case in candidate.get("cases", []):
    profile = case.get("prefill_profile") or {}
    focus_rows.extend(profile.get("focus_rows", []))
  custom_rows = [
      row for row in focus_rows
      if row.get("node_type") == "IQ36Identity"
      or "iq36_custom_identity" in str(row.get("node_name"))]

  klds = [
      float(row["kld_stock_to_candidate"])
      for row in distributions if row["kld_stock_to_candidate"] is not None]
  cosines = [
      float(row["cosine"])
      for row in distributions if row["cosine"] is not None]
  top1_rate = (
      sum(row["top1_match"] is True for row in distributions)
      / len(distributions) if distributions else 0.0)
  best = acceptance["openvino_q4_denominator"]["best_observed_tokens_s"]
  stock_decode_tpot_8k = 1000.0 / float(best["decode"]["8192"])
  target_decode_tpot_8k = 1000.0 / float(
      acceptance["bootstrap_targets"]["decode_tokens_s"]["8192"])
  decode_cut_ms = stock_decode_tpot_8k - target_decode_tpot_8k
  mechanism_upper = mechanism.get("overhead_upper_one_sided_95_ms")
  gross_graph_rows = []
  for case in cases:
    bucket = case.get("expected_input_tokens")
    if not isinstance(bucket, int):
      continue
    stock_case = stock_cases.get(case["case_id"], {})
    candidate_case = candidate_cases.get(case["case_id"], {})
    stock_prefill = stock_case.get("prefill_wall_ms")
    candidate_prefill = candidate_case.get("prefill_wall_ms")
    stock_decode = stock_case.get("decode_wall_ms_median")
    candidate_decode = candidate_case.get("decode_wall_ms_median")
    prefill_kill = (
        bucket / float(best["prefill"][str(bucket)]) * 1000.0
        - bucket / float(
            acceptance["bootstrap_targets"]["prefill_tokens_s"][str(bucket)])
        * 1000.0)
    decode_kill = (
        1000.0 / float(best["decode"][str(bucket)])
        - 1000.0 / float(
            acceptance["bootstrap_targets"]["decode_tokens_s"][str(bucket)]))
    prefill_overhead = (
        float(candidate_prefill) - float(stock_prefill)
        if isinstance(candidate_prefill, (int, float))
        and isinstance(stock_prefill, (int, float)) else None)
    decode_overhead = (
        float(candidate_decode) - float(stock_decode)
        if isinstance(candidate_decode, (int, float))
        and isinstance(stock_decode, (int, float)) else None)
    gross_graph_rows.append({
        "bucket": bucket,
        "case_id": case["case_id"],
        "decode_kill_number_ms": decode_kill,
        "decode_overhead_ms": decode_overhead,
        "decode_within_kill_number": (
            decode_overhead is not None and decode_overhead <= decode_kill),
        "prefill_kill_number_ms": prefill_kill,
        "prefill_overhead_ms": prefill_overhead,
        "prefill_within_kill_number": (
            prefill_overhead is not None and prefill_overhead <= prefill_kill),
    })

  full_case_ids = {
      "short_math_001", "short_factual_002", "short_transform_003",
      *(f"sentinel_{bucket // 1024:03d}k" for bucket in CORE_BUCKETS),
  }
  observed_case_ids = set(stock_cases) & set(candidate_cases)
  checks = [
      {"name": "clean_commit", "pass": not git["dirty"],
       "value": git},
      {"name": "all_locked_model_files_match",
       "pass": model_identity["required_checks_passed"]},
      {"name": "mechanism_worker_passed",
       "pass": mechanism_run["returncode"] == 0 and not mechanism_run["timed_out"]},
      {"name": "stock_worker_passed",
       "pass": stock_run["returncode"] == 0 and not stock_run["timed_out"]},
      {"name": "candidate_worker_passed",
       "pass": candidate_run["returncode"] == 0 and not candidate_run["timed_out"]},
      {"name": "stock_custom_config_absent",
       "pass": not _property_value(
           stock.get("core_properties_after", {}), "CONFIG_FILE"),
       "value": _property_value(
           stock.get("core_properties_after", {}), "CONFIG_FILE")},
      {"name": "candidate_custom_config_exact",
       "pass": _property_value(
           candidate.get("core_properties_after", {}), "CONFIG_FILE")
           == str(args.custom_config.resolve()),
       "value": _property_value(
           candidate.get("core_properties_after", {}), "CONFIG_FILE")},
      {"name": "candidate_graph_has_one_custom_node",
       "pass": candidate.get("custom_node_count") == 1,
       "value": candidate.get("custom_node_count")},
      {"name": "stock_graph_has_no_custom_node",
       "pass": stock.get("custom_node_count") == 0,
       "value": stock.get("custom_node_count")},
      {"name": "custom_gpu_kernel_selected_in_real_model",
       "pass": bool(custom_rows), "value": custom_rows[:8]},
      {"name": "custom_graph_without_config_is_rejected",
       "pass": mechanism.get("no_config_compile_failed") is True,
       "value": mechanism.get("no_config_compile_error")},
      {"name": "synthetic_custom_output_numeric",
       "pass": (
           mechanism.get("all_outputs_finite") is True
           and isinstance(mechanism.get("candidate_cosine"), (int, float))
           and float(mechanism["candidate_cosine"]) >= COSINE_MIN),
       "cosine": mechanism.get("candidate_cosine"),
       "exact_bits": mechanism.get("exact_outputs"),
       "max_abs": mechanism.get("candidate_max_abs"),
       "relative_l2": mechanism.get("candidate_relative_l2")},
      {"name": "paired_mechanism_blocks_present",
       "pass": int(mechanism.get("paired_blocks") or 0) >= 20,
       "value": mechanism.get("paired_blocks")},
      {"name": "paired_noop_overhead_does_not_consume_decode_kill_number",
       "pass": (
           isinstance(mechanism_upper, (int, float))
           and float(mechanism_upper) <= decode_cut_ms),
       "upper_one_sided_95_ms": mechanism_upper,
       "decode_kill_number_ms": decode_cut_ms},
      {"name": "real_model_noop_prefill_single_run_diagnostic",
       "pass": bool(gross_graph_rows) and all(
           row["prefill_within_kill_number"] for row in gross_graph_rows),
       "required": False,
       "reason": (
           "sequential one-shot wall timing is not paired inference; retain "
           "it as a gross graph-regression diagnostic while the 20-block "
           "ABBA mechanism bound is the OV0 performance gate"),
       "rows": gross_graph_rows},
      {"name": "real_model_noop_decode_single_run_diagnostic",
       "pass": bool(gross_graph_rows) and all(
           row["decode_within_kill_number"] for row in gross_graph_rows),
       "required": False,
       "reason": (
           "sequential one-shot wall timing is not paired inference; retain "
           "it as a gross graph-regression diagnostic while the 20-block "
           "ABBA mechanism bound is the OV0 performance gate"),
       "rows": gross_graph_rows},
      {"name": "all_requested_cases_present",
       "pass": observed_case_ids == {case["case_id"] for case in cases},
       "observed": sorted(observed_case_ids)},
      {"name": "full_ov0_case_set_present",
       "pass": args.smoke or observed_case_ids == full_case_ids,
       "full_set_present": observed_case_ids == full_case_ids,
       "required_for_ov0_exit": True,
       "waived_for_smoke": args.smoke,
       "observed": sorted(observed_case_ids)},
      {"name": "all_input_counts_exact",
       "pass": bool(token_rows) and all(row["input_count_exact"] for row in token_rows)},
      {"name": "all_generated_token_ids_exact",
       "pass": bool(token_rows) and all(row["token_ids_exact"] for row in token_rows)},
      {"name": "all_stock_sentinels_pass",
       "pass": args.smoke or all(
           row["sentinel_stock_pass"] is True for row in token_rows
           if row["expected_answer"] is not None),
       "waived_for_smoke": args.smoke},
      {"name": "all_candidate_sentinels_pass",
       "pass": args.smoke or all(
           row["sentinel_candidate_pass"] is True for row in token_rows
           if row["expected_answer"] is not None),
       "waived_for_smoke": args.smoke},
      {"name": "all_distribution_rows_finite",
       "pass": bool(distributions) and all(row["finite"] for row in distributions)},
      {"name": "teacher_forced_kld",
       "pass": bool(klds) and max(klds) <= KLD_MAX,
       "max": max(klds) if klds else None, "threshold": KLD_MAX},
      {"name": "teacher_forced_top1_rate",
       "pass": top1_rate >= TOP1_MIN,
       "rate": top1_rate, "threshold": TOP1_MIN},
      {"name": "component_cosine",
       "pass": bool(cosines) and min(cosines) >= COSINE_MIN,
       "min": min(cosines) if cosines else None, "threshold": COSINE_MIN},
      {"name": "state_schema_is_80_tensors",
       "pass": (
           stock.get("initial_state_summary", {}).get("count") == 80
           and candidate.get("initial_state_summary", {}).get("count") == 80),
       "stock": stock.get("initial_state_summary"),
       "candidate": candidate.get("initial_state_summary")},
      {"name": "runtime_versions_match_contract",
       "pass": (
           stock.get("openvino_runtime_version")
           == candidate.get("openvino_runtime_version")
           == load_json(args.model_contract)["runtime_contract"]["baseline"]
           ["openvino_runtime_version"]
           and stock.get("openvino_genai_version")
           == candidate.get("openvino_genai_version")
           == load_json(args.model_contract)["runtime_contract"]["baseline"]
           ["openvino_genai_version"]),
       "stock_runtime": stock.get("openvino_runtime_version"),
       "stock_genai": stock.get("openvino_genai_version")},
  ]
  required = all(
      check["pass"] for check in checks if check.get("required", True))
  ov0_exit = required and not args.smoke and observed_case_ids == full_case_ids
  route_label = (
      "contract_promoted" if ov0_exit
      else "smoke_only" if args.smoke
      else "rejected")
  correctness = {
      "checks": checks,
      "component_cosine_min": min(cosines) if cosines else None,
      "distribution_row_count": len(distributions),
      "diagnostic_checks_passed": all(
          check["pass"] for check in checks
          if check.get("required", True) is False),
      "gate": "ov0_immutable_stock_oracle_and_noop_custom_gpu_mechanism",
      "kld_max": max(klds) if klds else None,
      "ov0_exit": ov0_exit,
      "prefill_kill_number_ms": PREFILL_KILL_NUMBER_MS,
      "required_checks_passed": required,
      "route_label": route_label,
      "token_rows": token_rows,
      "top1_rate": top1_rate,
      "workstream": WORKSTREAM,
  }
  metrics = []
  for mode, payload in (("stock", stock), ("candidate", candidate)):
    for case in payload.get("cases", []):
      metrics.append({
          "case_id": case["case_id"],
          "case_wall_ms": case["case_wall_ms"],
          "decode_wall_ms_median": case["decode_wall_ms_median"],
          "generated_token_count": case["generated_token_count"],
          "input_token_count": case["input_token_count"],
          "metric_scope": "ov0_single_run_correctness_timing_not_product_speed",
          "mode": mode,
          "prefill_wall_ms": case["prefill_wall_ms"],
          "route_label": correctness["route_label"],
      })
  metrics.extend({
      "case_id": row["case_id"],
      "cosine": row["cosine"],
      "kld_stock_to_candidate": row["kld_stock_to_candidate"],
      "metric_scope": "stock_referenced_full_vocab_distribution",
      "step": row["step"],
      "top1_match": row["top1_match"],
  } for row in distributions)
  details = {
      "custom_real_model_profile_rows": custom_rows,
      "decode_kill_number_ms": decode_cut_ms,
      "distribution_rows": distributions,
      "gross_graph_regression_rows": gross_graph_rows,
      "mechanism": mechanism,
      "stock_decode_tpot_8k_ms": stock_decode_tpot_8k,
      "target_decode_tpot_8k_ms": target_decode_tpot_8k,
  }
  return correctness, metrics, details


def summary_markdown(
    correctness: dict[str, Any], details: dict[str, Any],
) -> str:
  state = "PASS" if correctness["ov0_exit"] else "NOT CLOSED"
  lines = [
      "# OpenVINO specialization bootstrap gate",
      "",
      f"- OV0 exit: **{state}**",
      f"- route label: `{correctness['route_label']}`",
      f"- full-vocabulary comparison rows: `{correctness['distribution_row_count']}`",
      f"- maximum stock-to-candidate KLD: `{correctness['kld_max']}`",
      f"- minimum logits cosine: `{correctness['component_cosine_min']}`",
      f"- top-1 agreement: `{correctness['top1_rate']}`",
      "- product performance claim: `forbidden (OV0 correctness/mechanism only)`",
      "",
      "## Derived no-op budget",
      "",
      f"- 8k decode cut required by the accepted stock/floor pair: "
      f"`{details['decode_kill_number_ms']:.6f} ms/token`",
      f"- paired custom-shim overhead one-sided 95% upper bound: "
      f"`{details['mechanism'].get('overhead_upper_one_sided_95_ms')} ms/call`",
      f"- 8k prefill end-to-end kill-number retained for OV1: "
      f"`{PREFILL_KILL_NUMBER_MS:.3f} ms per 1024-token equivalent`",
      "",
      "## Checks",
      "",
      "| check | result |",
      "|---|:---:|",
  ]
  for check in correctness["checks"]:
    if check.get("required", True) is False and not check["pass"]:
      result = "diagnostic-fail"
    else:
      result = "pass" if check["pass"] else "fail"
    lines.append(
        f"| {check['name']} | {result} |")
  lines += [
      "",
      "Single-run case timings in this artifact exist only to detect gross",
      "mechanism regressions. They are not ABBA product evidence and cannot",
      "satisfy any speed target.",
      "",
  ]
  return "\n".join(lines)


def orchestrator_main() -> int:
  args = parse_args()
  out_dir = args.out_dir.resolve()
  raw_dir = out_dir / "raw"
  raw_dir.mkdir(parents=True, exist_ok=False)
  git = git_state(out_dir)
  cases = build_cases(args)

  model_identity = capture_model_identity(
      args.model_dir.resolve(), args.model_contract.resolve())
  write_json(out_dir / "model-identity.json", model_identity)
  write_json(out_dir / "host.json", capture_host())

  common = {
      "custom_config": str(args.custom_config.resolve()),
      "custom_source": str(args.custom_source.resolve()),
      "model_dir": str(args.model_dir.resolve()),
      "prefill_chunk_tokens": args.prefill_chunk_tokens,
  }
  mechanism_run = run_worker(args, "mechanism", {
      **common,
      "blocks": 20,
      "warmup": 8,
      "worker_kind": "mechanism",
  }, raw_dir)
  stock_result_path = raw_dir / "stock/worker-result.json"
  stock_run = run_worker(args, "stock", {
      **common,
      "cases": cases,
      "logits_dir": str(raw_dir / "stock/logits"),
      "mode": "stock",
      "reference_result": None,
      "worker_kind": "model",
  }, raw_dir)
  candidate_run = run_worker(args, "candidate", {
      **common,
      "cases": cases,
      "logits_dir": str(raw_dir / "candidate/logits"),
      "mode": "candidate",
      "reference_result": str(stock_result_path),
      "worker_kind": "model",
  }, raw_dir)

  correctness, metrics, details = analyze(
      args, git, model_identity, cases, mechanism_run, stock_run,
      candidate_run)
  write_json(out_dir / "correctness.json", correctness)
  write_json(out_dir / "comparison.json", details)
  write_jsonl(out_dir / "metrics.jsonl", metrics)
  smoothness = {
      "applicable": False,
      "notes": (
          "OV0 freezes correctness and mechanism isolation. Product context-"
          "ladder smoothness requires paired promotion timing after OV1."),
      "route_label": correctness["route_label"],
  }
  write_json(out_dir / "smoothness.json", smoothness)
  (out_dir / "summary.md").write_text(
      summary_markdown(correctness, details), encoding="utf-8")
  manifest = {
      "captured_at": iso_now(),
      "git": git,
      "model_contract": str(args.model_contract.resolve()),
      "mode": "smoke" if args.smoke else "full_ov0",
      "route_label": correctness["route_label"],
      "schema_version": SCHEMA,
      "tool": str(Path(__file__).relative_to(ROOT)),
      "workstream": WORKSTREAM,
  }
  write_json(out_dir / "manifest.json", manifest)
  print(json.dumps({
      "event": "gate_complete",
      "out_dir": str(out_dir),
      "ov0_exit": correctness["ov0_exit"],
      "required_checks_passed": correctness["required_checks_passed"],
      "route_label": correctness["route_label"],
  }), flush=True)
  return 0 if correctness["required_checks_passed"] else 2


if __name__ == "__main__":
  if len(sys.argv) == 3 and sys.argv[1] == "__worker":
    raise SystemExit(worker_main(Path(sys.argv[2])))
  try:
    import numpy  # noqa: F401
  except ModuleNotFoundError:
    # Repository tools are normally launched with the host Python, while the
    # supported NumPy/OpenVINO pair lives in the locked OpenVINO environment.
    # Re-exec before creating the artifact so the orchestrator and workers use
    # one numeric ABI without asking callers to remember a special launcher.
    os.execv(
        str(OV_PYTHON),
        [str(OV_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]],
    )
  raise SystemExit(orchestrator_main())
