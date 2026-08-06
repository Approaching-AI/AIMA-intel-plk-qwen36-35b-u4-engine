#!/usr/bin/env python3
"""Execute adaptive layer 3 with accepted real 32k query and state captures.

This component-only gate reconstructs the pre-step-178 layer-3 state from the
clean seq1668 exact-history capture and feeds the exact query/current K/V and
stock attention captured by clean seq1593.  It does not execute the model or a
token loop.  The isolated four-stage custom primitive remains the sole graph
operation under test.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any
from xml.sax.saxutils import quoteattr


ROOT = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-openvino-adaptive-attention-captured-layer3-gate-v0"
ROUTE = "openvino_attention_adaptive_layer3_captured_graph_boundary"
NEXT_ROUTE = "openvino_attention_adaptive_all10_graph_compile_boundary"

BASE_TOOL = ROOT / "tools/intel-qwen36-openvino-adaptive-attention-layer3-gate.py"
COMPILE_GATE = ROOT / (
    "output/openvino-adaptive-attention-compile-gate-"
    "20260722Tseq1927-lmhead-delta11-clean/gate.json")
SYNTHETIC_GATE = ROOT / (
    "output/openvino-adaptive-attention-layer3-gate-"
    "20260720Tseq1680-clean/gate.json")
BOUNDARY_ROOT = ROOT / (
    "output/openvino-stock-micro-stock-prefill-"
    "20260719Tseq1593-layer3-32k-o512")
HISTORY_ROOT = ROOT / (
    "output/openvino-attention-history-capture-"
    "20260720Tseq1668-all10-step178-32k-o179")
BOUNDARY_WORKER = BOUNDARY_ROOT / (
    "raw/sentinel_032k/correctness/candidate/worker-result.json")
HISTORY_WORKER = HISTORY_ROOT / (
    "raw/sentinel_032k/correctness/candidate/worker-result.json")

CAPTURE_STEP = 178
LAYER = 3
HISTORY_TOKENS = 32945
KEY_TOKENS = HISTORY_TOKENS + 1
HOT_WINDOW = 16384
COLD_TOKENS = HISTORY_TOKENS - HOT_WINDOW
COLD_CHUNKS = (COLD_TOKENS + 511) // 512
OLD_EXACT_CAPACITY = 65536
OLD_PHYSICAL_CAPACITY = OLD_EXACT_CAPACITY + 1
PREFLIGHT_BYTES = 8 * 1024 ** 3


def load_base() -> Any:
  spec = importlib.util.spec_from_file_location(
      "iq36_adaptive_layer3_base", BASE_TOOL)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load base gate: {BASE_TOOL}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  module.HISTORY_TOKENS = HISTORY_TOKENS
  module.KEY_TOKENS = KEY_TOKENS
  module.HOT_WINDOW = HOT_WINDOW
  module.COLD_TOKENS = COLD_TOKENS
  module.COLD_CHUNKS = COLD_CHUNKS
  return module


BASE = load_base()
OV_PYTHON = BASE.OV_PYTHON
DEFAULT_PLUGIN = BASE.DEFAULT_PLUGIN
PARTITION_COUNT = (
    KEY_TOKENS + BASE.PARTITION_TOKENS - 1) // BASE.PARTITION_TOKENS


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out-dir", type=Path)
  parser.add_argument("--worker-config", type=Path, help=argparse.SUPPRESS)
  parser.add_argument("--candidate-plugin", type=Path, default=DEFAULT_PLUGIN)
  parser.add_argument(
      "--packed-kv-variant", choices=("k6v7", "k7v7", "k7v8", "k8v7"))
  parser.add_argument(
      "--topk", type=int, choices=(128, 252, 256, 512), default=512)
  parser.add_argument("--timeout-s", type=int, default=1800)
  parser.add_argument("--memory-stop-gib", type=float, default=4.0)
  parser.add_argument("--allow-dirty", action="store_true")
  args = parser.parse_args()
  if args.worker_config is None and args.out_dir is None:
    parser.error("--out-dir is required outside worker mode")
  if args.timeout_s <= 0:
    parser.error("--timeout-s must be positive")
  if args.memory_stop_gib <= 0.0:
    parser.error("--memory-stop-gib must be positive")
  if args.packed_kv_variant is not None and args.topk not in (256, 512):
    parser.error("packed K/V requires --topk 256 or 512")
  if (args.packed_kv_variant is not None and args.topk == 512 and
      args.packed_kv_variant != "k7v8"):
    parser.error("only packed K7/V8 currently admits --topk 512")
  return args


def capture_path(value: str) -> Path:
  path = Path(value)
  return path if path.is_absolute() else ROOT / path


def selected_boundary_row(payload: dict[str, Any]) -> dict[str, Any]:
  rows = [
      row for row in payload.get("attention_checkpoints", [])
      if int(row.get("layer", -1)) == LAYER and
      int(row.get("step", -1)) == CAPTURE_STEP]
  if len(rows) != 1:
    raise RuntimeError(f"expected one boundary row, observed {len(rows)}")
  return rows[0]


def selected_history_rows(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
  rows = {
      str(row["role"]): row
      for row in payload.get("attention_history_checkpoints", [])
      if int(row.get("layer", -1)) == LAYER and
      int(row.get("step", -1)) == CAPTURE_STEP}
  if set(rows) != {"key", "value"}:
    raise RuntimeError(f"incomplete history rows: {sorted(rows)}")
  return rows


def captured_fixture(np: Any) -> tuple[
    dict[str, Any], dict[str, Any], dict[str, bool], dict[str, Any]]:
  boundary_payload = BASE.load_json(BOUNDARY_WORKER)
  history_payload = BASE.load_json(HISTORY_WORKER)
  boundary = selected_boundary_row(boundary_payload)
  history = selected_history_rows(history_payload)
  tensors = boundary["tensors"]

  query = np.fromfile(
      capture_path(tensors["query"]["file"]), dtype="<f4").reshape(
          tensors["query"]["shape"]).astype(np.float16).reshape(
              1, BASE.Q_HEADS, 1, BASE.HEAD_DIM)
  current_key = np.fromfile(
      capture_path(tensors["key"]["file"]), dtype="<f4").reshape(
          tensors["key"]["shape"]).astype(np.float16).reshape(
              1, BASE.KV_HEADS, 1, BASE.HEAD_DIM)
  current_value = np.fromfile(
      capture_path(tensors["value"]["file"]), dtype="<f4").reshape(
          tensors["value"]["shape"]).astype(np.float16).reshape(
              1, BASE.KV_HEADS, 1, BASE.HEAD_DIM)
  exact_attention = np.fromfile(
      capture_path(tensors["attention"]["file"]), dtype="<f4").reshape(
          tensors["attention"]["shape"]).reshape(
              1, BASE.Q_HEADS, 1, BASE.HEAD_DIM)

  old_key = np.memmap(
      capture_path(history["key"]["file"]), dtype="<i4", mode="r",
      shape=tuple(history["key"]["shape"]))
  old_value = np.memmap(
      capture_path(history["value"]["file"]), dtype="<f2", mode="r",
      shape=tuple(history["value"]["shape"]))
  old_blocks = (
      (OLD_PHYSICAL_CAPACITY + BASE.KEY_TILE_TOKENS - 1) //
      BASE.KEY_TILE_TOKENS)
  history_key = np.empty(
      (BASE.KV_HEADS, HISTORY_TOKENS, BASE.HEAD_DIM), dtype=np.float16)
  history_value = np.array(
      old_value[0, :, :HISTORY_TOKENS, :], dtype=np.float16, copy=True)
  captured_last_key = np.empty(
      (BASE.KV_HEADS, BASE.HEAD_DIM), dtype=np.float16)
  packed_last_key = np.empty_like(captured_last_key)
  for kv_head in range(BASE.KV_HEADS):
    words = old_key[0, kv_head].reshape(-1)
    dense = words[old_blocks * BASE.HOT_KEY_WORDS_PER_BLOCK:].view(
        np.float16)[:OLD_PHYSICAL_CAPACITY * BASE.HEAD_DIM].reshape(
            OLD_PHYSICAL_CAPACITY, BASE.HEAD_DIM)
    history_key[kv_head] = dense[:HISTORY_TOKENS]
    captured_last_key[kv_head] = dense[HISTORY_TOKENS]
    block, lane = divmod(HISTORY_TOKENS, BASE.KEY_TILE_TOKENS)
    pair_words = old_key[
        0, kv_head, block, lane::BASE.KEY_TILE_TOKENS].view(np.uint32)
    half_bits = np.empty((BASE.HEAD_DIM,), dtype=np.uint16)
    half_bits[0::2] = (pair_words & np.uint32(0xffff)).astype(np.uint16)
    half_bits[1::2] = (pair_words >> np.uint32(16)).astype(np.uint16)
    packed_last_key[kv_head] = half_bits.view(np.float16)
  captured_last_value = np.array(
      old_value[0, :, HISTORY_TOKENS, :], dtype=np.float16, copy=True)

  hot_key = np.zeros((
      1, BASE.KV_HEADS, BASE.HOT_KEY_STORAGE_BLOCKS,
      BASE.HOT_KEY_WORDS_PER_BLOCK), dtype=np.int32)
  hot_value = np.zeros((
      1, BASE.KV_HEADS, BASE.PHYSICAL_HOT_CAPACITY,
      BASE.HEAD_DIM), dtype=np.float16)
  cold_key = np.zeros((
      1, BASE.KV_HEADS, BASE.FIXED_COLD_CAPACITY + 1,
      BASE.HEAD_DIM), dtype=np.int8)
  cold_value = np.zeros_like(cold_key)
  cold_key_scale = np.zeros((
      1, BASE.KV_HEADS, BASE.FIXED_COLD_CAPACITY + 1,
      BASE.KEY_STATE_BYTES), dtype=np.int8)
  cold_value_scale = np.zeros((
      1, BASE.KV_HEADS, BASE.FIXED_COLD_CAPACITY + 1,
      BASE.VALUE_STATE_BYTES), dtype=np.int8)
  cold_digits = (
      COLD_TOKENS % 128,
      (COLD_TOKENS // 128) % 128,
      (COLD_TOKENS // 16384) % 128)
  cold_key[:, :, 0, :3] = np.asarray(cold_digits, dtype=np.int8)

  base_scores = np.empty(
      (BASE.KV_HEADS, BASE.GQA_GROUP, KEY_TOKENS), dtype=np.float32)
  exact_scores = np.empty_like(base_scores)
  aligned_history = (
      (HISTORY_TOKENS + BASE.KEY_TILE_TOKENS - 1) //
      BASE.KEY_TILE_TOKENS * BASE.KEY_TILE_TOKENS)
  aligned_cold = (
      (COLD_TOKENS + BASE.KEY_TILE_TOKENS - 1) //
      BASE.KEY_TILE_TOKENS * BASE.KEY_TILE_TOKENS)

  for kv_head in range(BASE.KV_HEADS):
    words = hot_key[0, kv_head].reshape(-1)
    dense_begin = BASE.HOT_KEY_BLOCKS * BASE.HOT_KEY_WORDS_PER_BLOCK
    value_begin = 2 * BASE.HOT_KEY_BLOCKS * BASE.HOT_KEY_WORDS_PER_BLOCK
    dense_key = words[dense_begin:].view(np.float16)[
        :BASE.PHYSICAL_HOT_CAPACITY * BASE.HEAD_DIM].reshape(
            BASE.PHYSICAL_HOT_CAPACITY, BASE.HEAD_DIM)
    dimension_value = words[value_begin:].view(np.float16)[
        :BASE.PHYSICAL_HOT_CAPACITY * BASE.HEAD_DIM].reshape(
            BASE.HEAD_DIM, BASE.PHYSICAL_HOT_CAPACITY)
    dense_key[:HISTORY_TOKENS] = history_key[kv_head]
    dimension_value[:, :HISTORY_TOKENS] = history_value[kv_head].T
    hot_value[0, kv_head, :HISTORY_TOKENS] = history_value[kv_head]

    padded_key = np.zeros(
        (aligned_history, BASE.HEAD_DIM), dtype=np.float16)
    padded_key[:HISTORY_TOKENS] = history_key[kv_head]
    BASE.pack_half_key(
        hot_key[0, kv_head, :aligned_history // BASE.KEY_TILE_TOKENS],
        padded_key, np)

    quantized_key, key_scales, approximate_key, _ = BASE.quantize_group(
        history_key[kv_head, :COLD_TOKENS],
        BASE.KEY_QUANT_GROUP, BASE.KEY_RESIDUAL1, np,
        BASE.KEY_QUANT_BITS)
    quantized_value, value_scales, _, _ = BASE.quantize_group(
        history_value[kv_head, :COLD_TOKENS],
        BASE.VALUE_QUANT_GROUP, BASE.VALUE_RESIDUAL1, np,
        BASE.VALUE_QUANT_BITS)
    padded_quantized_key = np.zeros(
        (aligned_cold, BASE.HEAD_DIM), dtype=np.int8)
    padded_quantized_key[:COLD_TOKENS] = quantized_key
    key_payload = cold_key[0, kv_head, 1:, :].reshape(-1)
    if BASE.PACKED_KV_VARIANT is not None:
      BASE.pack_lowbit_state(
          key_payload[:aligned_cold * BASE.KEY_PACK_WORDS * 4],
          padded_quantized_key, BASE.KEY_QUANT_BITS, np)
      if BASE.PACKED_KV_VARIANT == "k7v8":
        value_payload = cold_value[0, kv_head, 1:, :].reshape(
            BASE.HEAD_DIM, BASE.FIXED_COLD_CAPACITY)
        value_payload[:, :COLD_TOKENS] = quantized_value.T
      else:
        padded_quantized_value = np.zeros(
            (aligned_cold, BASE.HEAD_DIM), dtype=np.int8)
        padded_quantized_value[:COLD_TOKENS] = quantized_value
        BASE.pack_lowbit_state(
            cold_value[0, kv_head, 1:, :].reshape(-1)[
                :aligned_cold * BASE.VALUE_PACK_WORDS * 4],
            padded_quantized_value, BASE.VALUE_QUANT_BITS, np)
    else:
      BASE.pack_i8_key(
          key_payload[:aligned_cold * BASE.HEAD_DIM],
          padded_quantized_key, np)
      value_payload = cold_value[0, kv_head, 1:, :].reshape(
          BASE.HEAD_DIM, BASE.FIXED_COLD_CAPACITY)
      value_payload[:, :COLD_TOKENS] = quantized_value.T
    key_scale_payload = BASE.scale_plane(
        cold_key_scale, kv_head, BASE.KEY_SCALE_GROUPS, np)
    value_scale_payload = BASE.scale_plane(
        cold_value_scale, kv_head, BASE.VALUE_SCALE_GROUPS, np)
    key_scale_payload[:, :COLD_TOKENS] = key_scales.T
    value_scale_payload[:, :COLD_TOKENS] = value_scales.T

    exact_key = np.concatenate((
        history_key[kv_head], current_key[0, kv_head]), axis=0)
    q_group = query[
        0, kv_head * BASE.GQA_GROUP:(kv_head + 1) * BASE.GQA_GROUP,
        0, :].astype(np.float32)
    exact = (
        exact_key.astype(np.float32) @ q_group.T).T * np.float32(
            BASE.ATTENTION_SCALE)
    exact_scores[kv_head] = exact
    base_scores[kv_head] = exact
    approximate = (
        approximate_key.astype(np.float32) @ q_group.T).T * np.float32(
            BASE.ATTENTION_SCALE)
    base_scores[kv_head, :, 1:COLD_TOKENS] = approximate[:, 1:]

  feed = {
      "query": query,
      "hot_key_bits": hot_key,
      "hot_value": hot_value,
      "current_key": current_key,
      "current_value": current_value,
      "cold_key": cold_key,
      "cold_value": cold_value,
      "cold_key_scale": cold_key_scale,
      "cold_value_scale": cold_value_scale,
      "mask": np.zeros((1, 1, 1, 1), dtype=np.float32),
      "eviction_shape": np.zeros(
          (1, BASE.KV_HEADS, 1, BASE.HEAD_DIM), dtype=np.int8),
      "eviction_count": np.ones((1, 1, 1, 1), dtype=np.int32),
      "decode_length": np.full(
          (1, 1, 1, BASE.MAX_CHUNKS), KEY_TOKENS, dtype=np.int32),
      "dynamic_shape_infer_carrier": np.zeros(
          (1, 1, 1, 1), dtype=np.float32),
  }
  reference = {
      "base_scores": base_scores,
      "exact_attention": exact_attention,
      "exact_scores": exact_scores,
  }
  fixture_checks = {
      "boundary_and_history_token_prefix_exact": (
          history_payload.get("generated_token_ids") ==
          boundary_payload.get("generated_token_ids", [])[:
              len(history_payload.get("generated_token_ids", []))]),
      "captured_dense_key_current_row_exact": bool(np.array_equal(
          captured_last_key, current_key[0, :, 0, :])),
      "captured_packed_key_current_row_exact": bool(np.array_equal(
          packed_last_key, current_key[0, :, 0, :])),
      "captured_value_current_row_exact": bool(np.array_equal(
          captured_last_value, current_value[0, :, 0, :])),
      "logical_history_is_pre_current_row": (
          int(history["key"]["logical_tokens"]) == KEY_TOKENS and
          int(history["value"]["logical_tokens"]) == KEY_TOKENS),
  }
  provenance = {
      "boundary_tensor_sha256": {
          role: str(tensors[role]["sha256"])
          for role in ("attention", "key", "query", "value")},
      "history_sha256": {
          role: str(history[role]["sha256"])
          for role in ("key", "value")},
  }
  return feed, reference, fixture_checks, provenance


def worker_main(config_path: Path) -> int:
  started = time.perf_counter_ns()
  result_path: Path | None = None
  try:
    if Path(sys.prefix).resolve() != OV_PYTHON.parent.parent.resolve():
      raise RuntimeError(f"worker requires {OV_PYTHON}, observed {sys.executable}")
    import numpy as np
    import openvino as ov

    config = BASE.load_json(config_path)
    BASE.TOPK = int(config.get("topk", 512))
    BASE.KEY_EXACT = False
    BASE.configure_quantization(
        32, 32, False, False, config.get("packed_kv_variant"))
    raw = Path(config["raw"])
    plugin = Path(config["plugin"])
    result_path = raw / "worker-result.json"
    registry = raw / "candidate-plugins.xml"
    registry.write_text(
        "<ie><plugins><plugin name=\"GPU\" location="
        f"{quoteattr(str(plugin.resolve()))}/></plugins></ie>\n",
        encoding="utf-8")
    core = ov.Core(str(registry))
    core.set_property("GPU", {"CONFIG_FILE": str(BASE.CUSTOM_XML.resolve())})
    context = core.get_default_context("GPU")
    model = BASE.make_model(ov)
    compile_started = time.perf_counter_ns()
    compiled = core.compile_model(model, context, {
        "ACTIVATIONS_SCALE_FACTOR": 0.0,
        "PERFORMANCE_HINT": "LATENCY",
        "PERF_COUNT": True,
    })
    compile_ms = (time.perf_counter_ns() - compile_started) / 1_000_000.0
    feed, reference, fixture_checks, provenance = captured_fixture(np)
    request = compiled.create_infer_request()
    remote_inputs: dict[str, Any] = {}
    for port in compiled.inputs:
      name = port.get_any_name()
      shape = (
          port.shape if port.partial_shape.is_static else
          ov.Shape(list(feed[name].shape)))
      remote = context.create_tensor(port.element_type, shape, {})
      remote.copy_from(ov.Tensor(feed[name]))
      request.set_tensor(name, remote)
      remote_inputs[name] = remote
    remote_outputs = []
    output_shapes = []
    for port in compiled.outputs[:6]:
      shape = port.shape
      remote = context.create_tensor(port.element_type, shape, {})
      request.set_tensor(port.get_any_name(), remote)
      remote_outputs.append(remote)
      output_shapes.append(shape)

    infer_started = time.perf_counter_ns()
    request.start_async()
    request.wait()
    first_wall_ms = (time.perf_counter_ns() - infer_started) / 1_000_000.0
    first_outputs = [
        BASE.copy_remote(remote, port.element_type, shape, context, np)
        for port, shape, remote in zip(
            compiled.outputs[:6], output_shapes, remote_outputs)]
    first_outputs.append(np.array(request.get_output_tensor(6).data, copy=True))
    workspace = BASE.analyze_workspace(first_outputs[0], np)
    expected_scores = reference["base_scores"].reshape(
        BASE.Q_HEADS, KEY_TOKENS)
    observed_scores = workspace["scores"][:, :KEY_TOKENS] * np.float32(
        BASE.ATTENTION_SCALE)
    score_metrics = {
        "all": BASE.vector_metrics(expected_scores, observed_scores, np),
        "cold": BASE.vector_metrics(
            expected_scores[:, :COLD_TOKENS],
            observed_scores[:, :COLD_TOKENS], np),
        "hot": BASE.vector_metrics(
            expected_scores[:, COLD_TOKENS:],
            observed_scores[:, COLD_TOKENS:], np),
    }
    independent = BASE.adaptive_reference(feed, reference, workspace, np)
    candidate = first_outputs[1].astype(np.float32)
    independent_metrics = BASE.vector_metrics(independent, candidate, np)
    captured_stock_metrics = BASE.vector_metrics(
        reference["exact_attention"], candidate, np)
    algorithm_metrics = BASE.vector_metrics(
        reference["exact_attention"], independent, np)
    workspace_metrics = BASE.vector_metrics(
        workspace["attention"], candidate, np)

    evicted_key = np.empty(
        (BASE.KV_HEADS, BASE.HEAD_DIM), dtype=np.float16)
    for kv_head in range(BASE.KV_HEADS):
      words = feed["hot_key_bits"][0, kv_head].reshape(-1)
      dense_begin = BASE.HOT_KEY_BLOCKS * BASE.HOT_KEY_WORDS_PER_BLOCK
      dense = words[dense_begin:].view(np.float16)[
          :BASE.PHYSICAL_HOT_CAPACITY * BASE.HEAD_DIM].reshape(
              BASE.PHYSICAL_HOT_CAPACITY, BASE.HEAD_DIM)
      evicted_key[kv_head] = dense[COLD_TOKENS]
    evicted_value = feed["hot_value"][0, :, COLD_TOKENS, :]
    expected_key_q, expected_key_scale, _ = BASE.quantized_append(
        evicted_key, BASE.KEY_QUANT_GROUP, BASE.KEY_RESIDUAL1, np,
        BASE.KEY_QUANT_BITS)
    expected_value_q, expected_value_scale, _ = BASE.quantized_append(
        evicted_value, BASE.VALUE_QUANT_GROUP, BASE.VALUE_RESIDUAL1, np,
        BASE.VALUE_QUANT_BITS)
    append_checks = {
        "cold_key_append_exact": BASE.packed_append_equal(
            first_outputs[2], expected_key_q.reshape(
                1, BASE.KV_HEADS, 1, BASE.HEAD_DIM),
            BASE.KEY_QUANT_BITS, np),
        "cold_key_scale_append_exact": bool(np.array_equal(
            first_outputs[4], expected_key_scale.reshape(
                1, BASE.KV_HEADS, 1, BASE.KEY_STATE_BYTES))),
        "cold_value_append_exact": BASE.packed_append_equal(
            first_outputs[3], expected_value_q.reshape(
                1, BASE.KV_HEADS, 1, BASE.HEAD_DIM),
            BASE.VALUE_QUANT_BITS, np),
        "cold_value_scale_append_exact": bool(np.array_equal(
            first_outputs[5], expected_value_scale.reshape(
                1, BASE.KV_HEADS, 1, BASE.VALUE_STATE_BYTES))),
    }
    state_checks = BASE.state_publication_checks(
        remote_inputs, compiled, context, feed, np)

    infer_started = time.perf_counter_ns()
    request.start_async()
    request.wait()
    second_wall_ms = (time.perf_counter_ns() - infer_started) / 1_000_000.0
    second_workspace_array = BASE.copy_remote(
        remote_outputs[0], compiled.output(0).element_type,
        compiled.output(0).shape, context, np)
    second_attention = BASE.copy_remote(
        remote_outputs[1], compiled.output(1).element_type,
        compiled.output(1).shape, context, np)
    second_workspace = BASE.analyze_workspace(second_workspace_array, np)
    deterministic = {
        "attention_exact": bool(np.array_equal(first_outputs[1], second_attention)),
        "local_candidates_exact": bool(np.array_equal(
            workspace["candidates"], second_workspace["candidates"])),
        "union_exact": bool(np.array_equal(
            workspace["union"], second_workspace["union"])),
    }
    profile = BASE.profile_rows(request)
    stage_profile_path = Path(os.environ.get(
        "IQ36_ADAPTIVE_STAGE_PROFILE_PATH", raw / "stage-profile.jsonl"))
    stage_profile = BASE.stage_profile_rows(stage_profile_path)
    runtime = BASE.runtime_rows(compiled)
    observed_shapes = [list(value.shape) for value in first_outputs]
    expected_shapes = [
        [1, 1, 1, BASE.workspace_offsets()["allocated"]],
        [1, BASE.Q_HEADS, 1, BASE.HEAD_DIM],
        [1, BASE.KV_HEADS, 1, BASE.HEAD_DIM],
        [1, BASE.KV_HEADS, 1, BASE.HEAD_DIM],
        [1, BASE.KV_HEADS, 1, BASE.KEY_STATE_BYTES],
        [1, BASE.KV_HEADS, 1, BASE.VALUE_STATE_BYTES],
        [1, 1, 1, 1],
    ]
    profile_us = float(profile[0]["real_time_us"]) if len(profile) == 1 else 0.0
    required_checks = {
        "accepted_capture_rows_are_mutually_consistent": all(
            fixture_checks.values()),
        "all_four_append_outputs_exact": all(append_checks.values()),
        "attention_is_finite": (
            independent_metrics["finite"] and captured_stock_metrics["finite"]),
        "attention_matches_independent_adaptive_reference": (
            independent_metrics["relative_l2"] <= 0.002 and
            independent_metrics["max_abs"] <= 0.002),
        "attention_tracks_captured_exact_stock_boundary": (
            captured_stock_metrics["relative_l2"] <= 0.003 and
            captured_stock_metrics["max_abs"] <= 0.005),
        "completion_counters_exact": bool(np.array_equal(
            workspace["completion"],
            np.full((BASE.KV_HEADS,), PARTITION_COUNT, dtype=np.uint32))),
        "exact_output_shapes": observed_shapes == expected_shapes,
        "four_stage_profile_is_aggregated": (
            len(profile) == 1 and profile_us >= 500.0 and
            profile_us <= second_wall_ms * 1250.0),
        "layer3_top512_runtime_is_custom_gpu": (
            len(runtime) == 1 and
            runtime[0]["layer_type"] == "CustomGPUPrimitive"),
        "local_candidate_scores_match_sidecar": (
            workspace["local_candidate_score_identity_pass"]),
        "local_candidates_have_exact_shape": (
            workspace["local_candidate_shape_pass"]),
        "ordered_remote_state_publication_exact": all(state_checks.values()),
        "output_matches_f16_workspace_publication": bool(np.array_equal(
            workspace["attention"].astype(np.float16), first_outputs[1])),
        "two_inference_results_are_deterministic": all(deterministic.values()),
        "union_matches_exported_candidate_heaps": workspace["union_exact"],
        "union_rows_are_bounded": all(
            BASE.TOPK <= count <= BASE.TOPK * BASE.GQA_GROUP
            for count in workspace["union_counts"]),
    }
    payload = {
        "algorithm_metrics_against_captured_stock": algorithm_metrics,
        "append_checks": append_checks,
        "captured_stock_metrics": captured_stock_metrics,
        "compile_ms": compile_ms,
        "deterministic": deterministic,
        "elapsed_ms": (time.perf_counter_ns() - started) / 1_000_000.0,
        "fixture": {
            "capture_step": CAPTURE_STEP,
            "cold_chunks": COLD_CHUNKS,
            "cold_tokens": COLD_TOKENS,
            "history_tokens": HISTORY_TOKENS,
            "key_tokens": KEY_TOKENS,
            "layer": LAYER,
            "topk": BASE.TOPK,
        },
        "fixture_checks": fixture_checks,
        "independent_reference_metrics": independent_metrics,
        "inference": {
            "first_wall_ms_diagnostic": first_wall_ms,
            "host_input_copy_in_timed_scope": False,
            "host_output_copy_in_timed_scope": False,
            "request_count": 2,
            "second_wall_ms_diagnostic": second_wall_ms,
        },
        "openvino_version": ov.get_version(),
        "output_shapes": observed_shapes,
        "plugin": str(plugin.resolve()),
        "plugin_sha256": BASE.sha256(plugin),
        "profile": profile,
        "provenance": provenance,
        "required_checks": required_checks,
        "required_checks_passed": all(required_checks.values()),
        "runtime": runtime,
        "score_metrics": score_metrics,
        "stage_profile": stage_profile,
        "stage_profile_path": str(stage_profile_path),
        "state_checks": state_checks,
        "union_counts": workspace["union_counts"],
        "workspace_attention_metrics": workspace_metrics,
    }
    BASE.write_json(result_path, payload)
    print(json.dumps({
        "captured_stock_relative_l2": captured_stock_metrics["relative_l2"],
        "profile_us": profile_us,
        "required_checks_passed": payload["required_checks_passed"],
        "union_counts": payload["union_counts"],
    }, sort_keys=True), flush=True)
    return 0 if payload["required_checks_passed"] else 2
  except Exception as error:
    payload = {
        "elapsed_ms": (time.perf_counter_ns() - started) / 1_000_000.0,
        "error": f"{type(error).__name__}: {error}",
        "required_checks_passed": False,
        "traceback": traceback.format_exc(),
    }
    if result_path is not None:
      BASE.write_json(result_path, payload)
    print(json.dumps(payload, sort_keys=True), flush=True)
    return 2


def exact_capture_evidence() -> dict[str, Any]:
  boundary_gate = BASE.load_json(BOUNDARY_ROOT / "gate.json")
  boundary_correctness = BASE.load_json(BOUNDARY_ROOT / "correctness.json")
  history_gate = BASE.load_json(HISTORY_ROOT / "gate.json")
  history_correctness = BASE.load_json(HISTORY_ROOT / "correctness.json")
  boundary_worker = BASE.load_json(BOUNDARY_WORKER)
  history_worker = BASE.load_json(HISTORY_WORKER)
  boundary_row = selected_boundary_row(boundary_worker)
  history_rows = selected_history_rows(history_worker)
  boundary_case = boundary_correctness.get("cases", [{}])[0]
  history_case = history_correctness.get("cases", [{}])[0]
  exact_boundary = [
      row for row in boundary_case.get("attention_boundary_rows", [])
      if int(row.get("layer", -1)) == LAYER and
      int(row.get("step", -1)) == CAPTURE_STEP]
  exact_history = [
      row for row in history_case.get("attention_history_rows", [])
      if int(row.get("layer", -1)) == LAYER and
      int(row.get("step", -1)) == CAPTURE_STEP]
  files = {
      **{
          f"boundary_{role}": {
              "expected": str(row["sha256"]),
              "observed": BASE.sha256(capture_path(row["file"])),
          }
          for role, row in boundary_row["tensors"].items()},
      **{
          f"history_{role}": {
              "expected": str(row["sha256"]),
              "observed": BASE.sha256(capture_path(row["file"])),
          }
          for role, row in history_rows.items()},
  }
  return {
      "boundary_formal_pass": (
          boundary_gate.get("run_checks_passed") is True and
          boundary_correctness.get("required_checks_passed") is True and
          boundary_gate.get("git", {}).get("dirty") is False),
      "boundary_rows_exact": (
          len(exact_boundary) == 4 and
          {row.get("role") for row in exact_boundary} ==
              {"attention", "key", "query", "value"} and
          all(row.get("exact") is True and row.get("finite") is True
              for row in exact_boundary)),
      "file_hashes": files,
      "file_hashes_exact": all(
          row["expected"] == row["observed"] for row in files.values()),
      "history_formal_pass": (
          history_gate.get("run_checks_passed") is True and
          history_correctness.get("required_checks_passed") is True and
          history_gate.get("git", {}).get("dirty") is False),
      "history_rows_exact": (
          len(exact_history) == 2 and
          {row.get("role") for row in exact_history} == {"key", "value"} and
          all(row.get("exact_after_stock_f16_round") is True and
              row.get("finite") is True and
              int(row.get("logical_tokens", -1)) == KEY_TOKENS
              for row in exact_history)),
      "token_prefix_exact": (
          history_worker.get("generated_token_ids") ==
          boundary_worker.get("generated_token_ids", [])[:
              len(history_worker.get("generated_token_ids", []))]),
  }


def summary_markdown(payload: dict[str, Any]) -> str:
  worker = payload.get("worker", {})
  captured = worker.get("captured_stock_metrics", {})
  independent = worker.get("independent_reference_metrics", {})
  profile = worker.get("profile", [])
  profile_us = profile[0].get("real_time_us") if profile else None
  return "\n".join([
      "# Captured adaptive-attention layer-3 boundary gate",
      "",
      f"- Verdict: `{payload['verdict']}`",
      f"- Required checks passed: "
      f"`{str(payload['required_checks_passed']).lower()}`",
      f"- Captured-stock relative L2 / max abs: "
      f"`{captured.get('relative_l2')} / {captured.get('max_abs')}`",
      f"- Independent-reference relative L2 / max abs: "
      f"`{independent.get('relative_l2')} / {independent.get('max_abs')}`",
      f"- KV-head union rows: `{worker.get('union_counts')}`",
      f"- Aggregated four-stage device time: `{profile_us} us`",
      "",
      "This gate replays accepted real captures in one isolated component. It",
      "does not execute a model, token loop, product worker, or speed claim.",
      "",
  ])


def orchestrator_main(args: argparse.Namespace) -> int:
  assert args.out_dir is not None
  out = args.out_dir.resolve()
  raw = out / "raw"
  raw.mkdir(parents=True, exist_ok=False)
  plugin = args.candidate_plugin.resolve()
  stop_bytes = int(args.memory_stop_gib * 1024 ** 3)
  memory = [BASE.memory_sample("start")]
  if int(memory[0]["available_bytes"]) < PREFLIGHT_BYTES:
    raise RuntimeError(
        f"8-GiB preflight failed: {memory[0]['available_bytes']} bytes")
  git = BASE.git_state(out)
  compile_gate = BASE.load_json(COMPILE_GATE)
  synthetic_gate = BASE.load_json(SYNTHETIC_GATE)
  capture = exact_capture_evidence()
  expected_plugin_sha = str(compile_gate.get("plugin_sha256", ""))
  plugin_sha = BASE.sha256(plugin) if plugin.is_file() else ""

  config = raw / "worker-config.json"
  BASE.write_json(config, {
      "packed_kv_variant": args.packed_kv_variant,
      "plugin": str(plugin),
      "raw": str(raw),
      "topk": args.topk,
  })
  time_path = raw / "worker.time.txt"
  command = [
      "/usr/bin/time", "-v", "-o", str(time_path),
      str(OV_PYTHON), str(Path(__file__).resolve()),
      "--worker-config", str(config),
  ]
  environment = os.environ.copy()
  environment["NEO_CACHE_DIR"] = str((raw / "compiler-cache").resolve())
  environment["IQ36_ADAPTIVE_STAGE_PROFILE_PATH"] = str(
      (raw / "stage-profile.jsonl").resolve())
  (raw / "compiler-cache").mkdir()
  worker_run = BASE.run_monitored_worker(
      command, environment, args.timeout_s, stop_bytes, memory,
      raw / "worker.stdout", raw / "worker.stderr")
  memory.append(BASE.memory_sample("finish"))
  worker = BASE.load_json(raw / "worker-result.json") \
      if (raw / "worker-result.json").is_file() else {}
  resources = BASE.parse_time(time_path)
  BASE.write_json(raw / "worker-command.json", {
      **worker_run,
      "environment": {
          "IQ36_ADAPTIVE_STAGE_PROFILE_PATH":
              environment["IQ36_ADAPTIVE_STAGE_PROFILE_PATH"],
          "NEO_CACHE_DIR": environment["NEO_CACHE_DIR"],
      },
      "resources": resources,
  })
  profile = worker.get("profile", [])
  profile_us = float(profile[0].get("real_time_us", 0.0)) \
      if len(profile) == 1 else 0.0
  memory_pass = (
      not worker_run["guard_tripped"] and
      all(int(row["available_bytes"]) >= stop_bytes for row in memory))
  checks = [
      BASE.check("repository_clean_at_gate",
                 not git["dirty"] or args.allow_dirty,
                 git=git, allow_dirty=args.allow_dirty),
      BASE.check("compile_gate_plugin_is_exact", plugin.is_file() and
                 plugin_sha == expected_plugin_sha,
                 plugin=BASE.relative(plugin), sha256=plugin_sha),
      BASE.check("seq1680_synthetic_boundary_is_admitted",
                 synthetic_gate.get("required_checks_passed") is True and
                 synthetic_gate.get("verdict") ==
                     "admit_adaptive_attention_layer3_captured_graph_boundary"),
      BASE.check("seq1593_boundary_capture_is_formal_and_exact",
                 capture["boundary_formal_pass"] and
                 capture["boundary_rows_exact"]),
      BASE.check("seq1668_history_capture_is_formal_and_exact",
                 capture["history_formal_pass"] and
                 capture["history_rows_exact"]),
      BASE.check("capture_token_prefix_and_file_hashes_are_exact",
                 capture["token_prefix_exact"] and
                 capture["file_hashes_exact"], evidence=capture),
      BASE.check("single_serial_worker_executes_two_requests",
                 worker_run["returncode"] == 0 and
                 worker.get("inference", {}).get("request_count") == 2),
      BASE.check("captured_layer3_worker_required_checks_pass",
                 worker.get("required_checks_passed") is True,
                 worker_checks=worker.get("required_checks", {})),
      BASE.check("four_stage_profile_is_device_aggregate",
                 profile_us >= 500.0,
                 profile_us=profile_us),
      BASE.check("worker_peak_rss_is_bounded",
                 0 < int(resources.get("maximum_resident_kib", 0)) <
                     8 * 1024 * 1024,
                 maximum_resident_kib=resources.get("maximum_resident_kib")),
      BASE.check("worker_does_not_swap",
                 int(resources.get("swaps", -1)) == 0,
                 swaps=resources.get("swaps")),
      BASE.check("memory_guard_never_tripped", memory_pass,
                 stop_bytes=stop_bytes,
                 minimum_available_bytes=min(
                     int(row["available_bytes"]) for row in memory)),
  ]
  required = all(row["pass"] for row in checks)
  verdict = (
      "admit_adaptive_attention_captured_layer3_boundary"
      if required else "repair_adaptive_attention_captured_layer3_boundary")
  payload = {
      "capture_evidence": capture,
      "checks": checks,
      "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
      "git": git,
      "long_worker_admitted": False,
      "memory": memory,
      "model_worker_admitted": False,
      "next_route": NEXT_ROUTE if required else ROUTE,
      "product_worker_admitted": False,
      "required_checks_passed": required,
      "route": ROUTE,
      "schema_version": SCHEMA,
      "sources": {
          "boundary_root": BASE.relative(BOUNDARY_ROOT),
          "compile_gate": BASE.relative(COMPILE_GATE),
          "history_root": BASE.relative(HISTORY_ROOT),
          "synthetic_gate": BASE.relative(SYNTHETIC_GATE),
      },
      "verdict": verdict,
      "worker": worker,
      "worker_resources": resources,
      "worker_run": worker_run,
      "workstream": WS,
  }
  BASE.write_json(out / "gate.json", payload)
  (out / "summary.md").write_text(
      summary_markdown(payload), encoding="utf-8")
  print(json.dumps({
      "captured_stock_relative_l2": worker.get(
          "captured_stock_metrics", {}).get("relative_l2"),
      "output": BASE.relative(out),
      "profile_us": profile_us,
      "required_checks_passed": required,
      "union_counts": worker.get("union_counts"),
      "verdict": verdict,
  }, sort_keys=True), flush=True)
  return 0 if required else 2


if __name__ == "__main__":
  parsed = parse_args()
  if parsed.worker_config is not None:
    raise SystemExit(worker_main(parsed.worker_config.resolve()))
  raise SystemExit(orchestrator_main(parsed))
