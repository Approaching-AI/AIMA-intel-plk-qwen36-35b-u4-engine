#!/usr/bin/env python3
"""Build one fixed real-Q4_K M64 NPU rate/correctness source graph."""

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
GROUP = 32
HIDDEN = 2048
EXPERTS = 256
GATEUP_ROWS_PER_EXPERT = 1024
ROW_BYTES = HIDDEN // 2
GROUPS = HIDDEN // GROUP
BLOCKS = HIDDEN // QK_K


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--prepack", type=Path, required=True)
  parser.add_argument("--capture", type=Path, required=True)
  parser.add_argument("--rows", type=int, required=True)
  parser.add_argument("--tokens", type=int, default=64)
  parser.add_argument("--compare-rows", type=int, default=512)
  parser.add_argument("--xml", type=Path, required=True)
  parser.add_argument("--bin", type=Path, required=True)
  parser.add_argument("--warmup", type=int, default=3)
  parser.add_argument("--repeat", type=int, default=9)
  args = parser.parse_args()
  maximum_rows = EXPERTS * GATEUP_ROWS_PER_EXPERT
  if args.rows <= 0 or args.rows > maximum_rows:
    parser.error(f"rows must be in [1, {maximum_rows}]")
  if args.tokens <= 0 or args.tokens > 1024:
    parser.error("tokens must be in [1, 1024]")
  if args.compare_rows <= 0 or args.compare_rows > args.rows:
    parser.error("compare-rows must be in [1, rows]")
  if args.warmup < 0 or args.repeat <= 0:
    parser.error("warmup must be non-negative and repeat positive")
  return args


def sha256_array(value: np.ndarray) -> str:
  return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def compare(actual: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
  lhs = np.asarray(actual, dtype=np.float32).ravel()
  rhs = np.asarray(reference, dtype=np.float32).ravel()
  if lhs.shape != rhs.shape:
    raise ValueError(f"comparison shape mismatch: {lhs.shape} != {rhs.shape}")
  lhs64 = lhs.astype(np.float64)
  rhs64 = rhs.astype(np.float64)
  delta = lhs64 - rhs64
  delta_sq = float(np.dot(delta, delta))
  reference_sq = float(np.dot(rhs64, rhs64))
  denom = float(np.linalg.norm(lhs64) * np.linalg.norm(rhs64))
  maximum_index = int(np.argmax(np.abs(delta))) if delta.size else 0
  return {
      "compared": int(lhs.size),
      "cosine": float(np.dot(lhs64, rhs64) / denom) if denom else 1.0,
      "finite": bool(np.isfinite(lhs).all() and np.isfinite(rhs).all()),
      "max_abs": float(np.max(np.abs(delta))) if delta.size else 0.0,
      "max_abs_index": maximum_index,
      "reference_abs_max": float(np.max(np.abs(rhs))) if rhs.size else 0.0,
      "relative_l2": math.sqrt(delta_sq / reference_sq)
      if reference_sq else (0.0 if delta_sq == 0.0 else math.inf),
      "rmse": math.sqrt(delta_sq / delta.size) if delta.size else 0.0,
  }


def quantize_q8_k(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  source = np.asarray(values, dtype=np.float32)
  tokens, columns = source.shape
  if columns % QK_K != 0:
    raise ValueError("Q8_K input width must be 256 aligned")
  dequantized = np.zeros_like(source, dtype=np.float32)
  codes = np.zeros_like(source, dtype=np.int8)
  for token in range(tokens):
    for begin in range(0, columns, QK_K):
      block = source[token, begin:begin + QK_K]
      maximum = np.float32(block[int(np.argmax(np.abs(block)))])
      if maximum == 0.0:
        continue
      inverse_scale = np.float32(np.float32(-127.0) / maximum)
      scaled = np.asarray(inverse_scale * block, dtype=np.float32)
      shifted = np.asarray(scaled + np.float32(12582912.0), dtype=np.float32)
      rounded = (
          np.bitwise_and(shifted.view(np.int32), np.int32(0x007FFFFF)) -
          np.int32(0x00400000))
      block_codes = np.minimum(rounded, np.int32(127)).astype(np.int8)
      scale = np.float32(np.float32(1.0) / inverse_scale)
      codes[token, begin:begin + QK_K] = block_codes
      dequantized[token, begin:begin + QK_K] = (
          block_codes.astype(np.float32) * scale)
  return codes, dequantized


def load_q4(prepack: Path, rows: int
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
  packed_map = np.memmap(
      prepack / "gateup-weights.bin", mode="r", dtype=np.uint8,
      shape=(EXPERTS, GATEUP_ROWS_PER_EXPERT, ROW_BYTES))
  packed = np.asarray(
      packed_map.reshape(-1, ROW_BYTES)[:rows], dtype=np.uint8).copy()
  codes = np.empty((rows, HIDDEN), dtype=np.uint8)
  codes[:, 0::2] = packed & np.uint8(15)
  codes[:, 1::2] = packed >> np.uint8(4)

  scales_map = np.memmap(
      prepack / "gateup-scales.bin", mode="r", dtype="<f4",
      shape=(EXPERTS, GROUPS, GATEUP_ROWS_PER_EXPERT))
  scales = np.asarray(
      scales_map.transpose(0, 2, 1).reshape(-1, GROUPS)[:rows],
      dtype=np.float32).copy()

  min_codes_map = np.memmap(
      prepack / "gateup-min-codes.bin", mode="r", dtype=np.uint8,
      shape=(EXPERTS, GATEUP_ROWS_PER_EXPERT, BLOCKS, 8))
  dmins_map = np.memmap(
      prepack / "gateup-dmins.bin", mode="r", dtype="<f4",
      shape=(EXPERTS, GATEUP_ROWS_PER_EXPERT, BLOCKS))
  min_codes = np.asarray(
      min_codes_map.reshape(-1, BLOCKS, 8)[:rows], dtype=np.uint8)
  dmins = np.asarray(
      dmins_map.reshape(-1, BLOCKS)[:rows], dtype=np.float32)
  mins = np.asarray(
      dmins[:, :, None] * min_codes.astype(np.float32),
      dtype=np.float32).reshape(rows, GROUPS)
  sizes = {
      "packed_weight_bytes": int(packed.nbytes),
      "logical_u4_bytes": int(codes.size // 2),
      "scale_bytes": int(scales.nbytes),
      "min_bytes": int(mins.nbytes),
  }
  return codes, scales, mins, sizes


def exact_q4_model(codes: np.ndarray, scales: np.ndarray,
                   mins: np.ndarray, tokens: int) -> Model:
  rows = int(codes.shape[0])
  source = ov.parameter([tokens, HIDDEN], Type.f32, name="q8_dequantized_input")
  code_constant = ov.constant(
      codes.reshape(rows, GROUPS, GROUP), Type.u4, name="q4_codes_u4")
  code_values = ov.convert(code_constant, Type.f32, name="q4_codes_f32")
  scale_constant = ov.constant(
      scales.reshape(rows, GROUPS, 1), Type.f32, name="q4_group_scale_f32")
  min_constant = ov.constant(
      mins.reshape(rows, GROUPS, 1), Type.f32, name="q4_group_min_f32")
  scaled = ov.multiply(code_values, scale_constant, name="q4_scaled_codes")
  affine = ov.subtract(scaled, min_constant, name="q4_exact_affine_weights")
  weights = ov.reshape(
      affine, ov.constant(np.array([rows, HIDDEN], dtype=np.int64)), False,
      name="q4_exact_weight_matrix")
  result = ov.matmul(
      source, weights, False, True, name="q4_exact_m64_matmul")
  result.output(0).get_tensor().set_names({"q4_exact_m64_output"})
  return Model([result], [source], "iq36_exact_q4_m64_prefill_rate")


def main() -> int:
  args = parse_args()
  output: dict[str, Any] = {
      "openvino_version": get_version(),
      "schema_version": "intel-qwen36-npu-exact-q4-m64-worker-v0",
  }
  try:
    required = [
        args.prepack / "gateup-weights.bin",
        args.prepack / "gateup-scales.bin",
        args.prepack / "gateup-min-codes.bin",
        args.prepack / "gateup-dmins.bin",
        args.capture,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
      raise FileNotFoundError("missing input: " + ", ".join(missing))

    captured = np.memmap(
        args.capture, mode="r", dtype="<f4", shape=(1024, HIDDEN))
    source = np.asarray(captured[:args.tokens], dtype=np.float32).copy()
    q8_codes, q8_input = quantize_q8_k(source)
    codes, scales, mins, source_sizes = load_q4(args.prepack, args.rows)

    compare_rows = args.compare_rows
    compare_weights = np.asarray(
        codes[:compare_rows].reshape(compare_rows, GROUPS, GROUP).astype(
            np.float32) * scales[:compare_rows, :, None] -
        mins[:compare_rows, :, None], dtype=np.float32).reshape(
            compare_rows, HIDDEN)
    reference = np.asarray(
        q8_input @ compare_weights.T, dtype=np.float32)

    model = exact_q4_model(codes, scales, mins, args.tokens)
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
    for _ in range(args.warmup):
      request.infer({0: q8_input})
    timings_us = []
    for _ in range(args.repeat):
      started = time.perf_counter()
      request.infer({0: q8_input})
      timings_us.append((time.perf_counter() - started) * 1e6)
    request.infer({0: q8_input})
    npu_output = np.array(
        request.get_output_tensor(0).data, dtype=np.float32, copy=True)
    comparison = compare(npu_output[:, :compare_rows], reference)
    logical_ops = 2 * args.tokens * HIDDEN * args.rows

    output.update({
        "available_devices": list(core.available_devices),
        "comparison": comparison,
        "compiler": {
            "compile_s": compile_s,
            "execution_devices": execution_devices,
        },
        "ir": {
            "bin_bytes": args.bin.stat().st_size,
            "bin_sha256": hashlib.sha256(args.bin.read_bytes()).hexdigest(),
            "xml_bytes": args.xml.stat().st_size,
            "xml_sha256": hashlib.sha256(args.xml.read_bytes()).hexdigest(),
        },
        "npu": {
            "logical_ops": logical_ops,
            "output_abs_sum": float(np.abs(npu_output.astype(np.float64)).sum()),
            "output_finite": bool(np.isfinite(npu_output).all()),
            "output_sha256": sha256_array(npu_output),
            "timing_median_us": float(np.median(timings_us)),
            "timing_min_us": min(timings_us),
            "timings_us": timings_us,
            "tops_median": logical_ops / (float(np.median(timings_us)) * 1e6),
            "tops_min_time": logical_ops / (min(timings_us) * 1e6),
        },
        "q8": {
            "codes_sha256": sha256_array(q8_codes),
            "dequantized_sha256": sha256_array(q8_input),
            "source_sha256": sha256_array(source),
        },
        "source": {
            "capture": str(args.capture),
            "columns": HIDDEN,
            "compare_rows": compare_rows,
            "logical_ops": logical_ops,
            "prepack": str(args.prepack),
            "rows": args.rows,
            "tokens": args.tokens,
            **source_sizes,
        },
        "source_hashes": {
            "codes": sha256_array(codes),
            "mins": sha256_array(mins),
            "scales": sha256_array(scales),
        },
    })
    del captured, source, q8_codes, q8_input, codes, scales, mins
    del compare_weights, reference, model, compiled, request, npu_output
    gc.collect()
  except Exception as error:  # pylint: disable=broad-exception-caught
    output["fatal_error"] = repr(error)
    output["fatal_traceback"] = traceback.format_exc()
  output["maxrss_kb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
  print(json.dumps(output, sort_keys=True), flush=True)
  return 0 if "fatal_error" not in output else 1


if __name__ == "__main__":
  raise SystemExit(main())
