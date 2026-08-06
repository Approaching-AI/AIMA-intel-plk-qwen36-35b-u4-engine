#!/usr/bin/env python3
"""Build and measure one exact low4/high2 NPU representation of real Q6_K."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import resource
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from openvino import Core, Model, Type, get_version, serialize
from openvino import opset15 as ov


QK_K = 256
Q6_BLOCK_BYTES = 210
Q6_LOW_BYTES = 128
Q6_HIGH_BYTES = 64
Q6_SCALE_BYTES = 16
GROUP = 16


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--model", type=Path, required=True)
  parser.add_argument("--tensor-name", required=True)
  parser.add_argument("--tensor-offset", type=int, required=True)
  parser.add_argument("--tensor-rows", type=int, required=True)
  parser.add_argument("--columns", type=int, required=True)
  parser.add_argument("--rows", type=int, required=True)
  parser.add_argument("--vector", type=Path, required=True)
  parser.add_argument("--vector-index", type=int, default=0)
  parser.add_argument("--xml", type=Path, required=True)
  parser.add_argument("--bin", type=Path, required=True)
  parser.add_argument("--warmup", type=int, default=3)
  parser.add_argument("--repeat", type=int, default=9)
  args = parser.parse_args()
  if args.columns <= 0 or args.columns % QK_K != 0:
    parser.error("columns must be positive and divisible by 256")
  if args.rows <= 0 or args.rows > args.tensor_rows:
    parser.error("rows must be in [1, tensor-rows]")
  if args.tensor_offset < 0 or args.vector_index < 0:
    parser.error("offsets and indices must be non-negative")
  if args.warmup < 0 or args.repeat <= 0:
    parser.error("warmup must be non-negative and repeat positive")
  return args


def sha256_bytes(value: np.ndarray) -> str:
  return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def compare(actual: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
  lhs = np.asarray(actual, dtype=np.float32).ravel()
  rhs = np.asarray(reference, dtype=np.float32).ravel()
  if lhs.shape != rhs.shape:
    raise ValueError(f"comparison shape mismatch: {lhs.shape} != {rhs.shape}")
  delta = lhs.astype(np.float64) - rhs.astype(np.float64)
  reference_sq = float(np.dot(rhs.astype(np.float64), rhs.astype(np.float64)))
  delta_sq = float(np.dot(delta, delta))
  denom = float(
      np.linalg.norm(lhs.astype(np.float64)) *
      np.linalg.norm(rhs.astype(np.float64)))
  maximum_index = int(np.argmax(np.abs(delta))) if delta.size else 0
  return {
      "compared": int(lhs.size),
      "cosine": float(
          np.dot(lhs.astype(np.float64), rhs.astype(np.float64)) / denom)
          if denom else 1.0,
      "finite": bool(np.isfinite(lhs).all() and np.isfinite(rhs).all()),
      "max_abs": float(np.max(np.abs(delta))) if delta.size else 0.0,
      "max_abs_index": maximum_index,
      "reference_abs_max": float(np.max(np.abs(rhs))) if rhs.size else 0.0,
      "relative_l2": math.sqrt(delta_sq / reference_sq)
      if reference_sq else (0.0 if delta_sq == 0.0 else math.inf),
      "rmse": math.sqrt(delta_sq / delta.size) if delta.size else 0.0,
  }


def quantize_q8_k(vector: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  source = np.asarray(vector, dtype=np.float32).reshape(-1)
  if source.size % QK_K != 0:
    raise ValueError("Q8_K vector must be 256 aligned")
  blocks = source.size // QK_K
  codes = np.zeros(source.size, dtype=np.int8)
  scales = np.zeros(blocks, dtype=np.float32)
  dequantized = np.zeros(source.size, dtype=np.float32)
  for block in range(blocks):
    begin = block * QK_K
    values = source[begin:begin + QK_K]
    maximum_index = int(np.argmax(np.abs(values)))
    maximum = np.float32(values[maximum_index])
    if maximum == 0.0:
      continue
    inverse_scale = np.float32(np.float32(-127.0) / maximum)
    scaled = np.asarray(inverse_scale * values, dtype=np.float32)
    shifted = np.asarray(scaled + np.float32(12582912.0), dtype=np.float32)
    rounded = (
        np.bitwise_and(shifted.view(np.int32), np.int32(0x007FFFFF)) -
        np.int32(0x00400000))
    rounded = np.minimum(rounded, np.int32(127))
    block_codes = rounded.astype(np.int8)
    scale = np.float32(np.float32(1.0) / inverse_scale)
    codes[begin:begin + QK_K] = block_codes
    scales[block] = scale
    dequantized[begin:begin + QK_K] = (
        block_codes.astype(np.float32) * scale)
  return codes, scales, dequantized


def unpack_q6(
    payload: np.ndarray, columns: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  rows = int(payload.shape[0])
  blocks = columns // QK_K
  low = np.empty((rows, columns), dtype=np.uint8)
  high = np.empty((rows, columns), dtype=np.uint8)
  integer_scales = np.empty((rows, columns // GROUP), dtype=np.int8)
  block_scales = np.empty((rows, blocks), dtype=np.float32)

  for block in range(blocks):
    raw = payload[:, block * Q6_BLOCK_BYTES:(block + 1) * Q6_BLOCK_BYTES]
    ql = raw[:, :Q6_LOW_BYTES]
    qh = raw[:, Q6_LOW_BYTES:Q6_LOW_BYTES + Q6_HIGH_BYTES]
    scales = raw[:, 192:192 + Q6_SCALE_BYTES].copy().view(np.int8)
    d = raw[:, 208:210].copy().view("<f2").reshape(rows).astype(np.float32)
    integer_scales[:, block * 16:(block + 1) * 16] = scales
    block_scales[:, block] = d
    base = block * QK_K
    for half in range(2):
      ql_half = ql[:, half * 64:(half + 1) * 64]
      qh_half = qh[:, half * 32:(half + 1) * 32]
      half_base = base + half * 128
      for lane in range(32):
        high_byte = qh_half[:, lane]
        low[:, half_base + lane] = ql_half[:, lane] & np.uint8(0x0F)
        high[:, half_base + lane] = high_byte & np.uint8(0x03)
        low[:, half_base + 32 + lane] = (
            ql_half[:, 32 + lane] & np.uint8(0x0F))
        high[:, half_base + 32 + lane] = (
            high_byte >> np.uint8(2)) & np.uint8(0x03)
        low[:, half_base + 64 + lane] = ql_half[:, lane] >> np.uint8(4)
        high[:, half_base + 64 + lane] = (
            high_byte >> np.uint8(4)) & np.uint8(0x03)
        low[:, half_base + 96 + lane] = ql_half[:, 32 + lane] >> np.uint8(4)
        high[:, half_base + 96 + lane] = (
            high_byte >> np.uint8(6)) & np.uint8(0x03)

  effective_scales = np.empty_like(integer_scales, dtype=np.float32)
  for block in range(blocks):
    effective_scales[:, block * 16:(block + 1) * 16] = (
        integer_scales[:, block * 16:(block + 1) * 16].astype(np.float32) *
        block_scales[:, block:block + 1])
  return low, high, integer_scales, block_scales, effective_scales


def q6_q8_oracle(
    low: np.ndarray,
    high: np.ndarray,
    integer_scales: np.ndarray,
    block_scales: np.ndarray,
    q8_codes: np.ndarray,
    q8_scales: np.ndarray,
) -> np.ndarray:
  rows, columns = low.shape
  blocks = columns // QK_K
  lane_accumulators = np.zeros((rows, 8), dtype=np.float32)
  for block in range(blocks):
    begin = block * QK_K
    codes = (
        low[:, begin:begin + QK_K].astype(np.int16) +
        np.int16(16) * high[:, begin:begin + QK_K].astype(np.int16) -
        np.int16(32)).astype(np.int32)
    activation = q8_codes[begin:begin + QK_K].astype(np.int32)
    scales = integer_scales[:, block * 16:(block + 1) * 16]
    weighted = (
        codes * activation.reshape(1, QK_K) *
        np.repeat(scales.astype(np.int32), GROUP, axis=1))
    combined = np.asarray(
        block_scales[:, block] * q8_scales[block], dtype=np.float32)
    for lane in range(8):
      lane_sum = np.sum(weighted[:, lane::8], axis=1, dtype=np.int32)
      lane_accumulators[:, lane] = np.asarray(
          lane_accumulators[:, lane] +
          combined * lane_sum.astype(np.float32), dtype=np.float32)
  return np.sum(lane_accumulators, axis=1, dtype=np.float32)


def representation_reference(
    low: np.ndarray,
    high: np.ndarray,
    effective_scales: np.ndarray,
    dequantized_input: np.ndarray,
) -> np.ndarray:
  rows, columns = low.shape
  groups = columns // GROUP
  codes = (
      low.astype(np.int16) + np.int16(16) * high.astype(np.int16) -
      np.int16(32)).reshape(rows, groups, GROUP)
  products = (
      codes.astype(np.float32) * effective_scales.reshape(rows, groups, 1) *
      dequantized_input.reshape(1, groups, GROUP))
  return np.sum(products, axis=(1, 2), dtype=np.float32)


def exact_splitplane_model(
    low: np.ndarray, high: np.ndarray, effective_scales: np.ndarray,
) -> Model:
  rows, columns = low.shape
  groups = columns // GROUP
  source = ov.parameter([1, columns], Type.f32, name="q8_dequantized_input")
  scale = ov.constant(
      effective_scales.reshape(rows, groups, 1), Type.f32,
      name="q6_group_scale_f32")

  low_codes = ov.constant(
      low.reshape(rows, groups, GROUP), Type.u4, name="q6_low4_codes")
  low_values = ov.convert(low_codes, Type.f32, name="q6_low4_to_f32")
  low_scaled = ov.multiply(low_values, scale, name="q6_low4_group_scale")
  low_weights = ov.reshape(
      low_scaled, ov.constant(np.array([rows, columns], dtype=np.int64)),
      False, name="q6_low4_matrix")
  low_result = ov.matmul(
      source, low_weights, False, True, name="q6_low4_matmul")

  high_codes = ov.constant(
      high.reshape(rows, groups, GROUP), Type.u2, name="q6_high2_codes")
  high_values = ov.convert(high_codes, Type.f32, name="q6_high2_to_f32")
  high_centered = ov.subtract(
      high_values, ov.constant(np.array(2.0, dtype=np.float32)),
      name="q6_high2_minus_two")
  high_scale = ov.multiply(
      scale, ov.constant(np.array(16.0, dtype=np.float32)),
      name="q6_high2_scale_x16")
  high_scaled = ov.multiply(
      high_centered, high_scale, name="q6_high2_group_scale")
  high_weights = ov.reshape(
      high_scaled, ov.constant(np.array([rows, columns], dtype=np.int64)),
      False, name="q6_high2_matrix")
  high_result = ov.matmul(
      source, high_weights, False, True, name="q6_high2_matmul")

  result = ov.add(low_result, high_result, name="q6_exact_splitplane_result")
  result.output(0).get_tensor().set_names({"q6_exact_splitplane_output"})
  return Model([result], [source], "iq36_exact_q6_low4_high2_m1")


def main() -> int:
  args = parse_args()
  output: dict[str, Any] = {
      "openvino_version": get_version(),
      "schema_version": "intel-qwen36-npu-exact-q6-representation-worker-v0",
  }
  try:
    row_bytes = args.columns // QK_K * Q6_BLOCK_BYTES
    payload_bytes = args.rows * row_bytes
    payload = np.memmap(
        args.model, dtype=np.uint8, mode="r", offset=args.tensor_offset,
        shape=(args.rows, row_bytes)).copy()
    vector_offset = args.vector_index * args.columns * np.dtype(np.float32).itemsize
    with args.vector.open("rb") as handle:
      handle.seek(vector_offset)
      vector = np.fromfile(handle, dtype="<f4", count=args.columns)
    if vector.size != args.columns:
      raise ValueError("real vector file is truncated")

    low, high, integer_scales, block_scales, effective_scales = unpack_q6(
        payload, args.columns)
    q8_codes, q8_scales, q8_input = quantize_q8_k(vector)
    oracle = q6_q8_oracle(
        low, high, integer_scales, block_scales, q8_codes, q8_scales)
    represented = representation_reference(low, high, effective_scales, q8_input)
    representation_comparison = compare(represented, oracle)

    source_hashes = {
        "high2_codes_sha256": sha256_bytes(high),
        "low4_codes_sha256": sha256_bytes(low),
        "q6_payload_sha256": sha256_bytes(payload),
        "q8_codes_sha256": sha256_bytes(q8_codes),
        "q8_dequantized_sha256": sha256_bytes(q8_input),
    }
    model = exact_splitplane_model(low, high, effective_scales)
    args.xml.parent.mkdir(parents=True, exist_ok=True)
    serialize(model, args.xml, args.bin)

    core = Core()
    compile_started = time.perf_counter()
    compiled = core.compile_model(model, "NPU")
    compile_s = time.perf_counter() - compile_started
    execution_devices = compiled.get_property("EXECUTION_DEVICES")
    if isinstance(execution_devices, str):
      execution_devices = [execution_devices]
    else:
      execution_devices = list(execution_devices)
    request = compiled.create_infer_request()
    input_value = q8_input.reshape(1, args.columns)
    for _ in range(args.warmup):
      request.infer({0: input_value})
    timings_us = []
    for _ in range(args.repeat):
      started = time.perf_counter()
      request.infer({0: input_value})
      timings_us.append((time.perf_counter() - started) * 1e6)
    request.infer({0: input_value})
    npu_output = np.array(
        request.get_output_tensor(0).data, dtype=np.float32, copy=True).reshape(-1)
    npu_comparison = compare(npu_output, oracle)

    output.update({
        "available_devices": list(core.available_devices),
        "compiler": {
            "compile_s": compile_s,
            "execution_devices": execution_devices,
        },
        "comparison": {
            "npu_vs_q6_q8_oracle": npu_comparison,
            "representation_vs_q6_q8_oracle": representation_comparison,
        },
        "ir": {
            "bin_bytes": args.bin.stat().st_size,
            "bin_sha256": hashlib.sha256(args.bin.read_bytes()).hexdigest(),
            "xml_bytes": args.xml.stat().st_size,
            "xml_sha256": hashlib.sha256(args.xml.read_bytes()).hexdigest(),
        },
        "npu": {
            "output_abs_sum": float(np.abs(npu_output.astype(np.float64)).sum()),
            "output_finite": bool(np.isfinite(npu_output).all()),
            "output_sha256": sha256_bytes(npu_output),
            "timings_us": timings_us,
            "timing_min_us": min(timings_us),
            "timing_median_us": float(np.median(timings_us)),
        },
        "q8": {
            "block_count": int(q8_scales.size),
            "code_max": int(q8_codes.max()),
            "code_min": int(q8_codes.min()),
            "dequantized_input_abs_max": float(np.abs(q8_input).max()),
            "source_vector_abs_max": float(np.abs(vector).max()),
            "vector_index": args.vector_index,
        },
        "representation": {
            "effective_scale_bytes": int(effective_scales.nbytes),
            "high2_logical_packed_bytes": int(high.size // 4),
            "high2_max": int(high.max()),
            "high2_min": int(high.min()),
            "low4_logical_packed_bytes": int(low.size // 2),
            "low4_max": int(low.max()),
            "low4_min": int(low.min()),
            "semantics": "low4 + 16 * (high2 - 2), group16 scale d * sc",
        },
        "source": {
            "columns": args.columns,
            "payload_bytes": payload_bytes,
            "row_bytes": row_bytes,
            "rows": args.rows,
            "tensor_name": args.tensor_name,
            "tensor_offset": args.tensor_offset,
            "tensor_rows": args.tensor_rows,
        },
        "source_hashes": source_hashes,
    })
    del payload, low, high, integer_scales, block_scales, effective_scales
    del oracle, represented, model, compiled, request
    gc.collect()
  except Exception as error:  # pylint: disable=broad-exception-caught
    output["fatal_error"] = repr(error)
    output["fatal_traceback"] = traceback.format_exc()
  output["maxrss_kb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
  print(json.dumps(output, sort_keys=True), flush=True)
  return 0 if "fatal_error" not in output else 1


if __name__ == "__main__":
  raise SystemExit(main())
