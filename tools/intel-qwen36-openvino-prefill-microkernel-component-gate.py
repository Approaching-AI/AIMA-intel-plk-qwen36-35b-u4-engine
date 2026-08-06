#!/usr/bin/env python3
"""Compile and numerically gate the embedded prefill microkernel carrier.

This probe stays below full-model scale.  It validates both plugin branches:
the 32-token prefill specialization must fuse and execute the captured KQ/VS
packages, while the one-token unified specialization must strip the prefill
package and compile the ordinary decode kernel.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any
from xml.sax.saxutils import quoteattr


ROOT = Path(__file__).resolve().parents[1]
OV_PYTHON = Path("/home/intel/ov/openvino_env/bin/python")
CUSTOM_CONFIG = ROOT / "engine/openvino/custom/iq36_hot_attention_gqa.xml"
PLUGIN = Path(
    "/home/intel/intel-qwen36-r0/source/openvino-90214e5be05/"
    "bin/intel64/Release/libopenvino_intel_gpu_plugin.so")
SCHEMA = "intel-qwen36-openvino-prefill-microkernel-component-gate-v0"
DEFAULT_QUERY_TOKENS = 32
HEAD_DIM = 256
Q_HEADS = 16
KV_HEADS = 2
GQA_GROUP = 8
HOT_KEY_WORDS_PER_BLOCK = 2048


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out-dir", type=Path)
  parser.add_argument("--device", default="GPU")
  parser.add_argument("--openvino-python", type=Path, default=OV_PYTHON)
  parser.add_argument("--custom-config", type=Path, default=CUSTOM_CONFIG)
  parser.add_argument("--candidate-gpu-plugin", type=Path, default=PLUGIN)
  parser.add_argument("--query-tokens", type=int, default=DEFAULT_QUERY_TOKENS)
  parser.add_argument(
      "--past-tokens", type=int, default=0,
      help=("populate this many exact F16 history rows and compile the "
            "continuation specialization (default: initial prefill)"))
  args = parser.parse_args()
  if args.query_tokens < 1:
    parser.error("query-tokens must be positive")
  if args.past_tokens < 0:
    parser.error("past-tokens must be non-negative")
  if args.out_dir is None:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.out_dir = (
        ROOT / f"output/openvino-prefill-microkernel-component-{stamp}")
  return args


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parameter(ov: Any, shape: list[int], dtype: Any, name: str) -> Any:
  return ov.opset13.parameter(shape, dtype, name=name)


def hot_shape(query_tokens: int, past_tokens: int = 0) -> tuple[int, int]:
  total_tokens = query_tokens + past_tokens
  ring_tokens = 16384
  while ring_tokens < total_tokens:
    ring_tokens *= 2
  hot_tokens = ring_tokens + 1
  hot_key_blocks = (hot_tokens + 15) // 16
  return hot_tokens, 2 * hot_key_blocks + 1


def input_parameters(
    ov: Any, query_tokens: int, past_tokens: int = 0,
) -> list[Any]:
  # A dynamic token axis selects the plugin's multi-output custom-op path.
  # Runtime specialization still sees the concrete 32-token/one-token shape.
  token_axis = -1
  hot_tokens, hot_key_storage_blocks = hot_shape(
      query_tokens, past_tokens)
  total_tokens = query_tokens + past_tokens
  desired_cold = max(0, total_tokens - 8192)
  previous_cold = max(0, past_tokens - 8192)
  cold_append = max(1, desired_cold - previous_cold)
  cold_capacity = total_tokens + 1 if past_tokens else 1
  return [
      parameter(ov, [1, Q_HEADS, token_axis, HEAD_DIM], ov.Type.f16,
                "query"),
      parameter(ov, [1, KV_HEADS, hot_key_storage_blocks,
                     HOT_KEY_WORDS_PER_BLOCK], ov.Type.i32, "hot_key"),
      parameter(ov, [1, KV_HEADS, hot_tokens, HEAD_DIM], ov.Type.f16,
                "hot_value"),
      parameter(ov, [1, KV_HEADS, token_axis, HEAD_DIM], ov.Type.f16,
                "current_key"),
      parameter(ov, [1, KV_HEADS, token_axis, HEAD_DIM], ov.Type.f16,
                "current_value"),
      parameter(ov, [1, KV_HEADS, cold_capacity, HEAD_DIM], ov.Type.i8,
                "cold_key"),
      parameter(ov, [1, KV_HEADS, cold_capacity, HEAD_DIM], ov.Type.i8,
                "cold_value"),
      parameter(ov, [1, KV_HEADS, cold_capacity, 16], ov.Type.i8,
                "key_scale"),
      parameter(ov, [1, KV_HEADS, cold_capacity, 16], ov.Type.i8,
                "value_scale"),
      parameter(ov, [1, 1, token_axis, total_tokens], ov.Type.f16, "mask"),
      parameter(ov, [1, KV_HEADS, cold_append, HEAD_DIM], ov.Type.i8,
                "eviction_template"),
      parameter(ov, [1, 1, 1, 1], ov.Type.i32, "eviction_count"),
      parameter(ov, [1, 1, 1, 1], ov.Type.i32, "length_carrier"),
  ]


def make_model(
    ov: Any, np: Any, graph: Any, phase: str, prefill_tokens: int,
    past_tokens: int = 0,
) -> Any:
  query_tokens = prefill_tokens if phase == "prefill" else 1
  parameters = input_parameters(ov, query_tokens, past_tokens)
  operation_type = (
      graph.prefill_custom_classes(ov)[1 if past_tokens else 0]
      if phase == "prefill" else graph.custom_class(ov))
  operation = operation_type([value.output(0) for value in parameters])
  operation.set_friendly_name(f"iq36_microkernel_component_{phase}")
  if phase == "prefill":
    padded_tokens = ((prefill_tokens + 31) // 32) * 32
    expanded = ov.opset13.reshape(
        operation.output(0),
        ov.opset13.constant(np.array(
            [1, Q_HEADS, padded_tokens, HEAD_DIM], dtype=np.int64)), False)
    last = ov.opset13.gather(
        expanded,
        ov.opset13.constant(np.array(prefill_tokens - 1, dtype=np.int64)),
        ov.opset13.constant(np.array(2, dtype=np.int64)))
    outputs = [last.output(0)] + [
        operation.output(index) for index in range(1, 5)]
  else:
    outputs = [operation.output(1), operation.output(0)] + [
        operation.output(index) for index in range(2, 6)]
  return ov.Model(outputs, parameters, f"iq36_microkernel_component_{phase}")


def inputs(
    np: Any, query_tokens: int, seed: int, past_tokens: int = 0,
) -> tuple[list[Any], tuple[Any, Any, Any]]:
  rng = np.random.default_rng(seed)
  hot_tokens, hot_key_storage_blocks = hot_shape(
      query_tokens, past_tokens)
  total_tokens = query_tokens + past_tokens
  query = (rng.standard_normal(
      (1, Q_HEADS, query_tokens, HEAD_DIM)) * 0.25).astype(np.float16)
  past_key = (rng.standard_normal(
      (1, KV_HEADS, past_tokens, HEAD_DIM)) * 0.25).astype(np.float16)
  past_value = (rng.standard_normal(
      (1, KV_HEADS, past_tokens, HEAD_DIM)) * 0.25).astype(np.float16)
  current_key = (rng.standard_normal(
      (1, KV_HEADS, query_tokens, HEAD_DIM)) * 0.25).astype(np.float16)
  current_value = (rng.standard_normal(
      (1, KV_HEADS, query_tokens, HEAD_DIM)) * 0.25).astype(np.float16)
  hot_key = np.zeros((1, KV_HEADS, hot_key_storage_blocks,
                      HOT_KEY_WORDS_PER_BLOCK), dtype=np.int32)
  hot_value = np.zeros(
      (1, KV_HEADS, hot_tokens, HEAD_DIM), dtype=np.float16)
  packed_blocks = (hot_tokens + 15) // 16
  packed_pairs = hot_key[:, :, :packed_blocks].view(np.float16).reshape(
      1, KV_HEADS, packed_blocks, HEAD_DIM // 2, 16, 2)
  slots = np.arange(past_tokens, dtype=np.int64)
  slots = np.where(slots < 1, slots, 1 + (slots - 1) % (hot_tokens - 1))
  for token, slot in enumerate(slots):
    packed_pairs[:, :, slot // 16, :, slot % 16, :] = (
        past_key[:, :, token].reshape(1, KV_HEADS, HEAD_DIM // 2, 2))
  dense = np.ascontiguousarray(
      hot_key[:, :, packed_blocks:2 * packed_blocks]
  ).view(np.float16).reshape(1, KV_HEADS, -1)
  dense[:, :, :hot_tokens * HEAD_DIM] = 0
  for token, slot in enumerate(slots):
    begin = int(slot) * HEAD_DIM
    dense[:, :, begin:begin + HEAD_DIM] = past_key[:, :, token]
    hot_value[:, :, slot] = past_value[:, :, token]
  hot_key[:, :, packed_blocks:2 * packed_blocks] = dense.view(
      np.int32).reshape(1, KV_HEADS, packed_blocks, HOT_KEY_WORDS_PER_BLOCK)
  desired_cold = max(0, total_tokens - 8192)
  previous_cold = max(0, past_tokens - 8192)
  cold_append = max(1, desired_cold - previous_cold)
  cold_capacity = total_tokens + 1 if past_tokens else 1
  values = [
      query,
      hot_key,
      hot_value,
      current_key,
      current_value,
      np.zeros((1, KV_HEADS, cold_capacity, HEAD_DIM), dtype=np.int8),
      np.zeros((1, KV_HEADS, cold_capacity, HEAD_DIM), dtype=np.int8),
      np.zeros((1, KV_HEADS, cold_capacity, 16), dtype=np.int8),
      np.zeros((1, KV_HEADS, cold_capacity, 16), dtype=np.int8),
      np.zeros((1, 1, query_tokens, total_tokens), dtype=np.float16),
      np.zeros((1, KV_HEADS, cold_append, HEAD_DIM), dtype=np.int8),
      np.full((1, 1, 1, 1), desired_cold - previous_cold, dtype=np.int32),
      np.full((1, 1, 1, 1), query_tokens, dtype=np.int32),
  ]
  return values, (
      query,
      np.concatenate([past_key, current_key], axis=2),
      np.concatenate([past_value, current_value], axis=2))


def causal_reference_last(np: Any, tensors: tuple[Any, Any, Any]) -> Any:
  query, key, value = (item.astype(np.float32) for item in tensors)
  output = np.empty((1, Q_HEADS, HEAD_DIM), dtype=np.float32)
  for head in range(Q_HEADS):
    kv_head = head // GQA_GROUP
    scores = query[0, head, -1] @ key[0, kv_head].T
    scores *= np.float32(1.0 / 16.0)
    row_max = np.max(scores, keepdims=True)
    weights = np.exp(scores - row_max)
    weights /= np.sum(weights, keepdims=True)
    output[0, head] = weights @ value[0, kv_head]
  return output


def profile_rows(request: Any) -> list[dict[str, Any]]:
  return [{
      "node_name": row.node_name,
      "node_type": row.node_type,
      "exec_type": row.exec_type,
      "real_time_us": row.real_time.total_seconds() * 1_000_000.0,
  } for row in request.get_profiling_info()
          if "iq36" in row.node_name.lower()]


def infer(compiled: Any, np: Any, values: list[Any]) -> tuple[Any, Any, float]:
  request = compiled.create_infer_request()
  started = time.perf_counter_ns()
  outputs = request.infer({port: value for port, value in zip(
      compiled.inputs, values, strict=True)}, share_outputs=False)
  elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
  return request, np.asarray(outputs[compiled.output(0)]), elapsed_ms


def main() -> int:
  args = parse_args()
  if Path(sys.prefix).resolve() != args.openvino_python.parent.parent.resolve():
    raise RuntimeError(
        f"gate requires {args.openvino_python}, observed {sys.executable}")
  for path in (args.custom_config, args.candidate_gpu_plugin):
    if not path.is_file():
      raise FileNotFoundError(path)

  import numpy as np
  import openvino as ov
  import intel_qwen36_openvino_hot_cold_attention as graph

  out = args.out_dir.resolve()
  out.mkdir(parents=True, exist_ok=False)
  registry = out / "candidate-plugins.xml"
  registry.write_text(
      "<ie><plugins><plugin name=\"GPU\" location="
      f"{quoteattr(str(args.candidate_gpu_plugin.resolve()))}/></plugins></ie>\n",
      encoding="utf-8")
  core = ov.Core(str(registry))
  core.set_property(
      args.device, {"CONFIG_FILE": str(args.custom_config.resolve())})
  compile_config = {"PERFORMANCE_HINT": "LATENCY", "PERF_COUNT": True}

  prefill_started = time.perf_counter_ns()
  prefill_compiled = core.compile_model(
      make_model(
          ov, np, graph, "prefill", args.query_tokens, args.past_tokens),
      args.device, compile_config)
  prefill_compile_ms = (
      time.perf_counter_ns() - prefill_started) / 1_000_000.0
  prefill_values, tensors = inputs(
      np, args.query_tokens, 36, args.past_tokens)
  prefill_request, candidate, prefill_infer_ms = infer(
      prefill_compiled, np, prefill_values)
  reference = causal_reference_last(np, tensors)
  delta = candidate.astype(np.float32) - reference
  reference_norm = float(np.linalg.norm(reference.reshape(-1)))
  candidate_norm = float(np.linalg.norm(candidate.astype(np.float32).reshape(-1)))
  cosine_denominator = reference_norm * candidate_norm
  cosine = float(
      np.dot(reference.reshape(-1), candidate.astype(np.float32).reshape(-1)) /
      cosine_denominator) if cosine_denominator else 1.0
  relative_l2 = float(
      np.linalg.norm(delta.reshape(-1)) / reference_norm)

  decode_started = time.perf_counter_ns()
  decode_compiled = core.compile_model(
      make_model(ov, np, graph, "decode", args.query_tokens),
      args.device, compile_config)
  decode_compile_ms = (
      time.perf_counter_ns() - decode_started) / 1_000_000.0
  decode_values, _ = inputs(np, 1, 37)
  decode_request, decode_attention, decode_infer_ms = infer(
      decode_compiled, np, decode_values)

  checks = [
      {"name": "prefill_output_finite",
       "pass": bool(np.isfinite(candidate).all())},
      {"name": "prefill_relative_l2_at_most_0_01",
       "pass": relative_l2 <= 0.01},
      {"name": "prefill_cosine_at_least_0_999",
       "pass": cosine >= 0.999},
      {"name": "decode_marker_strip_compiles_and_runs",
       "pass": bool(np.isfinite(decode_attention).all())},
  ]
  passed = all(row["pass"] for row in checks)
  metrics = {
      "schema": SCHEMA,
      "passed": passed,
      "checks": checks,
      "openvino_version": ov.get_version(),
      "candidate_gpu_plugin": str(args.candidate_gpu_plugin.resolve()),
      "candidate_gpu_plugin_sha256": sha256(args.candidate_gpu_plugin),
      "custom_config": str(args.custom_config.resolve()),
      "custom_config_sha256": sha256(args.custom_config),
      "prefill": {
      "query_tokens": args.query_tokens,
      "past_tokens": args.past_tokens,
          "compile_ms": prefill_compile_ms,
          "infer_ms": prefill_infer_ms,
          "max_abs": float(np.max(np.abs(delta))),
          "relative_l2": relative_l2,
          "cosine": cosine,
          "profile": profile_rows(prefill_request),
      },
      "decode": {
          "compile_ms": decode_compile_ms,
          "infer_ms": decode_infer_ms,
          "profile": profile_rows(decode_request),
      },
  }
  write_json(out / "metrics.json", metrics)
  print(json.dumps({
      "event": "complete", "out_dir": str(out), "passed": passed,
      "prefill_relative_l2": relative_l2, "prefill_cosine": cosine,
  }, sort_keys=True))
  return 0 if passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
