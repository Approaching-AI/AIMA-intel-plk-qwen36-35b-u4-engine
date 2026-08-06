#!/usr/bin/env python3
"""Execute one deterministic layer-3 adaptive-attention boundary.

The worker reuses the seq1673 Mix/UnitValue/KeyValue/ValueValue fixture at a
64k-history plus one-token decode boundary.  It executes exactly one isolated
OpenVINO custom operation, checks the exported local candidates and KV-head
union, compares the attention result with an independent adaptive reference,
and verifies the ordered update through all four append outputs and the
request-owned remote state.  This is a component boundary, not a model or
product benchmark.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any
from xml.sax.saxutils import quoteattr


ROOT = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-openvino-adaptive-attention-layer3-gate-v0"
ROUTE = "openvino_attention_adaptive_layer3_boundary"
NEXT_ROUTE = "openvino_attention_adaptive_layer3_captured_graph_boundary"

OV_PYTHON = Path("/home/intel/ov/openvino_env/bin/python")
CUSTOM_XML = ROOT / "engine/openvino/custom/iq36_hot_attention_gqa.xml"
GRAPH_MODULE = ROOT / "tools/intel_qwen36_openvino_hot_cold_attention.py"
COMPILE_GATE = ROOT / (
    "output/openvino-adaptive-attention-compile-gate-"
    "20260722Tseq1927-lmhead-delta11-clean/gate.json")
SEQ1673 = ROOT / (
    "output/openvino-adaptive-attention-component-"
    "20260720Tseq1673-clean/result.json")
DEFAULT_PLUGIN = ROOT / (
    "output/openvino-adaptive-attention-compile-gate-"
    "20260722Tseq1927-lmhead-delta11-clean/raw/build/"
    "libopenvino_intel_gpu_plugin-adaptive.so")
SOURCE_RUNNER = ROOT / "engine/tools/adaptive_i8_hotcold_gqa_decode.cpp"
SOURCE_KERNEL = ROOT / "engine/gpu/opencl/direct_i8_hotcold_gqa_decode.cl"

TOPK = 512
KEY_EXACT = False
PACKED_KV_VARIANT: str | None = None
MAX_CHUNKS = 129
FIXED_COLD_CAPACITY = 65536
EXACT_HISTORY_CAPACITY = 66560
PHYSICAL_HOT_CAPACITY = EXACT_HISTORY_CAPACITY + 1
HISTORY_TOKENS = 65536
KEY_TOKENS = HISTORY_TOKENS + 1
HOT_WINDOW = 16384
SINK_TOKENS = 4
COLD_TOKENS = HISTORY_TOKENS - HOT_WINDOW
CHUNK_TOKENS = 512
PARTITION_TOKENS = 256
COLD_CHUNKS = COLD_TOKENS // CHUNK_TOKENS
MAX_COLD_CHUNKS = MAX_CHUNKS - HOT_WINDOW // CHUNK_TOKENS
MAX_COLD_TOKENS = MAX_COLD_CHUNKS * CHUNK_TOKENS
MAX_TOKENS = MAX_CHUNKS * CHUNK_TOKENS
MAX_PARTITIONS = MAX_CHUNKS * 2
PARTITION_COUNT = (KEY_TOKENS + PARTITION_TOKENS - 1) // PARTITION_TOKENS
LOCAL_TOPK = 64
KV_HEADS = 2
Q_HEADS = 16
GQA_GROUP = 8
HEAD_DIM = 256
KEY_QUANT_GROUP = 32
VALUE_QUANT_GROUP = 32
KEY_SCALE_GROUPS = HEAD_DIM // KEY_QUANT_GROUP
VALUE_SCALE_GROUPS = HEAD_DIM // VALUE_QUANT_GROUP
KEY_SCALE_BYTES = KEY_SCALE_GROUPS * 2
VALUE_SCALE_BYTES = VALUE_SCALE_GROUPS * 2
RESIDUAL1_BYTES = HEAD_DIM // 8
KEY_RESIDUAL1 = False
VALUE_RESIDUAL1 = False
KEY_STATE_BYTES = KEY_SCALE_BYTES
VALUE_STATE_BYTES = VALUE_SCALE_BYTES
KEY_QUANT_BITS = 8
VALUE_QUANT_BITS = 8
KEY_TILE_TOKENS = 16
KEY_PACK_WORDS = HEAD_DIM * KEY_QUANT_BITS // 32
VALUE_PACK_WORDS = HEAD_DIM * VALUE_QUANT_BITS // 32
HOT_KEY_WORDS_PER_BLOCK = 2048
HOT_KEY_BLOCKS = (
    PHYSICAL_HOT_CAPACITY + KEY_TILE_TOKENS - 1) // KEY_TILE_TOKENS
HOT_KEY_STORAGE_BLOCKS = 3 * HOT_KEY_BLOCKS + 1
ATTENTION_SCALE = 0.0625
PREFLIGHT_BYTES = 8 * 1024 ** 3


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out-dir", type=Path)
  parser.add_argument("--worker-config", type=Path, help=argparse.SUPPRESS)
  parser.add_argument("--candidate-plugin", type=Path, default=DEFAULT_PLUGIN)
  parser.add_argument(
      "--topk", type=int, choices=(128, 252, 256, 512), default=512)
  parser.add_argument("--key-quant-group", type=int, choices=(32,), default=32)
  parser.add_argument(
      "--value-quant-group", type=int, choices=(16, 32), default=32)
  parser.add_argument("--key-residual1", action="store_true")
  parser.add_argument("--value-residual1", action="store_true")
  parser.add_argument("--key-exact", action="store_true")
  parser.add_argument(
      "--packed-kv-variant", choices=("k6v7", "k7v7", "k7v8", "k8v7"))
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
  if (args.key_residual1 or args.value_residual1) and \
      args.value_quant_group != 32:
    parser.error("residual1 requires K32/V32")
  if args.key_exact and (
      args.topk != 256 or args.value_quant_group != 32 or
      args.key_residual1 or args.value_residual1):
    parser.error("key-exact requires top-256 K32/V32 without residual1")
  if args.packed_kv_variant is not None and (
      args.topk not in (256, 512) or args.key_quant_group != 32 or
      args.value_quant_group != 32 or args.key_exact or
      args.key_residual1 or args.value_residual1):
    parser.error("packed K/V requires top-256/512 K32/V32 without variants")
  if (args.packed_kv_variant is not None and args.topk == 512 and
      args.packed_kv_variant != "k7v8"):
    parser.error("only packed K7/V8 currently admits top-512 correction")
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
    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def relative(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path.resolve())


def available_memory_bytes() -> int:
  for line in Path("/proc/meminfo").read_text(
      encoding="utf-8").splitlines():
    if line.startswith("MemAvailable:"):
      return int(line.split()[1]) * 1024
  raise RuntimeError("MemAvailable is missing")


def memory_sample(label: str) -> dict[str, Any]:
  return {"available_bytes": available_memory_bytes(), "label": label}


def check(name: str, passed: bool, **evidence: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **evidence}


def git_state(out_dir: Path) -> dict[str, Any]:
  def git(*arguments: str) -> str:
    run = subprocess.run(
        ["git", *arguments], cwd=ROOT, check=False, capture_output=True,
        text=True, encoding="utf-8", errors="replace")
    return run.stdout.strip() if run.returncode == 0 else ""

  dirty = git("status", "--porcelain").splitlines()
  try:
    output_relative = str(out_dir.resolve().relative_to(ROOT))
  except ValueError:
    output_relative = ""
  dirty = [row for row in dirty
           if not output_relative or output_relative not in row]
  return {
      "commit": git("rev-parse", "HEAD"),
      "dirty": bool(dirty),
      "dirty_paths": dirty,
  }


def parse_time(path: Path) -> dict[str, Any]:
  text = path.read_text(encoding="utf-8") if path.is_file() else ""
  patterns = {
      "elapsed": r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (.+)",
      "major_page_faults": r"Major \(requiring I/O\) page faults: (\d+)",
      "maximum_resident_kib": r"Maximum resident set size \(kbytes\): (\d+)",
      "swaps": r"Swaps: (\d+)",
  }
  result: dict[str, Any] = {"raw": text}
  for key, pattern in patterns.items():
    match = re.search(pattern, text)
    if match:
      result[key] = (
          match.group(1) if key == "elapsed" else int(match.group(1)))
  return result


def run_monitored_worker(
    command: list[str], environment: dict[str, str], timeout_s: int,
    stop_bytes: int, memory: list[dict[str, Any]], stdout_path: Path,
    stderr_path: Path,
) -> dict[str, Any]:
  started = time.monotonic()
  with stdout_path.open("w", encoding="utf-8") as stdout, \
       stderr_path.open("w", encoding="utf-8") as stderr:
    process = subprocess.Popen(
        command, cwd=ROOT, env=environment, stdout=stdout, stderr=stderr,
        text=True)
    guard_tripped = False
    timed_out = False
    sample_index = 0
    while process.poll() is None:
      elapsed = time.monotonic() - started
      if elapsed > timeout_s:
        timed_out = True
        process.terminate()
        break
      sample = memory_sample(f"worker-{sample_index}")
      memory.append(sample)
      sample_index += 1
      if int(sample["available_bytes"]) < stop_bytes:
        guard_tripped = True
        process.terminate()
        break
      time.sleep(0.5)
    try:
      returncode = process.wait(timeout=10)
    except subprocess.TimeoutExpired:
      process.kill()
      returncode = process.wait()
  return {
      "command": command,
      "elapsed_s": time.monotonic() - started,
      "guard_tripped": guard_tripped,
      "returncode": 124 if timed_out else returncode,
      "timed_out": timed_out,
  }


def load_graph_module() -> Any:
  spec = importlib.util.spec_from_file_location(
      "iq36_adaptive_layer3_graph", GRAPH_MODULE)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load graph module: {GRAPH_MODULE}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def workspace_offsets() -> dict[str, int]:
  partial_meta = KV_HEADS * MAX_PARTITIONS * GQA_GROUP
  union_words = (MAX_COLD_TOKENS + 31) // 32
  offsets = {"partial_max": 0}
  offsets["partial_sum"] = offsets["partial_max"] + partial_meta
  offsets["partial_numerator"] = offsets["partial_sum"] + partial_meta
  offsets["approximate_score"] = (
      offsets["partial_numerator"] + partial_meta * HEAD_DIM)
  offsets["local_candidates"] = (
      offsets["approximate_score"] + Q_HEADS * MAX_TOKENS)
  offsets["union_bits"] = (
      offsets["local_candidates"] +
      Q_HEADS * MAX_COLD_CHUNKS * LOCAL_TOPK)
  offsets["completion"] = (
      offsets["union_bits"] + KV_HEADS * union_words)
  offsets["attention"] = offsets["completion"] + KV_HEADS
  offsets["required"] = offsets["attention"] + Q_HEADS * HEAD_DIM
  offsets["allocated"] = ((offsets["required"] + 15) // 16) * 16
  return offsets


def mix_u32(values: Any, np: Any) -> Any:
  result = np.asarray(values, dtype=np.uint32).copy()
  result ^= result >> np.uint32(16)
  result *= np.uint32(0x7feb352d)
  result ^= result >> np.uint32(15)
  result *= np.uint32(0x846ca68b)
  result ^= result >> np.uint32(16)
  return result


def unit_values(
    tokens: Any, flat_dimensions: Any, salt: int, np: Any,
) -> Any:
  token_term = (
      np.asarray(tokens, dtype=np.uint32)[:, None] * np.uint32(0x9e3779b9))
  dimension_term = (
      np.asarray(flat_dimensions, dtype=np.uint32)[None, :] *
      np.uint32(0x85ebca6b))
  values = token_term ^ dimension_term ^ np.uint32(salt)
  hashed = mix_u32(values, np)
  signed = (hashed & np.uint32(0xffff)).astype(np.int32) - 32768
  return signed.astype(np.float32) / np.float32(32768.0)


def pack_half_key(destination: Any, values: Any, np: Any) -> None:
  if values.shape[0] % KEY_TILE_TOKENS != 0:
    raise ValueError("packed half-key rows must be block16 aligned")
  bits = values.view(np.uint16)
  words = (
      bits[:, 0::2].astype(np.uint32) |
      (bits[:, 1::2].astype(np.uint32) << np.uint32(16)))
  packed = words.reshape(-1, KEY_TILE_TOKENS, HEAD_DIM // 2).transpose(
      0, 2, 1).reshape(-1, HOT_KEY_WORDS_PER_BLOCK)
  destination[:packed.shape[0], :] = packed.view(np.int32)


def pack_i8_key(destination: Any, values: Any, np: Any) -> None:
  if values.shape[0] % KEY_TILE_TOKENS != 0:
    raise ValueError("packed I8-key rows must be block16 aligned")
  packed = values.reshape(
      -1, KEY_TILE_TOKENS, KEY_PACK_WORDS, 4).transpose(
          0, 2, 1, 3).reshape(-1)
  destination[:packed.size] = packed


def packed_code_words(values: Any, quant_bits: int, np: Any) -> Any:
  """Pack signed group-32 codes into the kernel's little-endian words."""
  if values.shape[-1] != HEAD_DIM:
    raise ValueError(f"expected head dimension {HEAD_DIM}: {values.shape}")
  mask = np.uint32((1 << quant_bits) - 1)
  codes = (values.astype(np.int32).astype(np.uint32) & mask).reshape(
      -1, HEAD_DIM // 32, 32)
  words = np.zeros(
      (codes.shape[0], HEAD_DIM // 32, quant_bits), dtype=np.uint32)
  for index in range(32):
    bit = index * quant_bits
    word, shift = divmod(bit, 32)
    words[:, :, word] |= codes[:, :, index] << np.uint32(shift)
    if shift + quant_bits > 32:
      words[:, :, word + 1] |= codes[:, :, index] >> np.uint32(32 - shift)
  return words.reshape(-1, HEAD_DIM * quant_bits // 32)


def pack_lowbit_state(
    destination: Any, values: Any, quant_bits: int, np: Any,
) -> None:
  if values.shape[0] % KEY_TILE_TOKENS != 0:
    raise ValueError("packed low-bit rows must be block16 aligned")
  words_per_token = HEAD_DIM * quant_bits // 32
  words = packed_code_words(values, quant_bits, np)
  packed = words.reshape(
      -1, KEY_TILE_TOKENS, words_per_token).transpose(0, 2, 1)
  raw = np.ascontiguousarray(packed).reshape(-1).view(np.uint8)
  destination[:raw.size] = raw.view(np.int8)


def packed_state_token_bytes(
    payload: Any, token: int, words_per_token: int, np: Any,
) -> Any:
  token_block, token_lane = divmod(token, KEY_TILE_TOKENS)
  packed = payload.view(np.uint32)
  words = np.empty((words_per_token,), dtype=np.uint32)
  for word in range(words_per_token):
    words[word] = packed[
        (token_block * words_per_token + word) * KEY_TILE_TOKENS +
        token_lane]
  return words.view(np.uint8).view(np.int8)


def unpack_lowbit_state(
    payload: Any, token_count: int, quant_bits: int, np: Any,
) -> Any:
  aligned_tokens = (
      (token_count + KEY_TILE_TOKENS - 1) // KEY_TILE_TOKENS *
      KEY_TILE_TOKENS)
  words_per_token = HEAD_DIM * quant_bits // 32
  word_count = aligned_tokens * words_per_token
  block_words = payload[:word_count * 4].view(np.uint32).reshape(
      -1, words_per_token, KEY_TILE_TOKENS)
  token_words = block_words.transpose(0, 2, 1).reshape(
      aligned_tokens, HEAD_DIM // 32, quant_bits)
  values = np.empty(
      (aligned_tokens, HEAD_DIM // 32, 32), dtype=np.int8)
  mask = np.uint32((1 << quant_bits) - 1)
  sign = np.uint32(1 << (quant_bits - 1))
  for index in range(32):
    bit = index * quant_bits
    word, shift = divmod(bit, 32)
    code = token_words[:, :, word] >> np.uint32(shift)
    if shift + quant_bits > 32:
      code |= token_words[:, :, word + 1] << np.uint32(32 - shift)
    code &= mask
    values[:, :, index] = ((code ^ sign) - sign).astype(np.int32).astype(
        np.int8)
  return values.reshape(aligned_tokens, HEAD_DIM)[:token_count]


def configure_quantization(
    key_group: int, value_group: int,
    key_residual1: bool, value_residual1: bool,
    packed_kv_variant: str | None = None,
) -> None:
  global KEY_QUANT_GROUP, VALUE_QUANT_GROUP
  global KEY_SCALE_GROUPS, VALUE_SCALE_GROUPS
  global KEY_SCALE_BYTES, VALUE_SCALE_BYTES
  global KEY_RESIDUAL1, VALUE_RESIDUAL1
  global KEY_STATE_BYTES, VALUE_STATE_BYTES
  global PACKED_KV_VARIANT, KEY_QUANT_BITS, VALUE_QUANT_BITS
  global KEY_PACK_WORDS, VALUE_PACK_WORDS
  if key_group != 32 or value_group not in (16, 32):
    raise ValueError("adaptive component admits K32 with V32 or V16")
  if (key_residual1 or value_residual1) and value_group != 32:
    raise ValueError("residual1 is admitted only with K32/V32")
  if packed_kv_variant not in (None, "k6v7", "k7v7", "k7v8", "k8v7"):
    raise ValueError(f"unknown packed K/V variant: {packed_kv_variant}")
  if packed_kv_variant is not None and (
      value_group != 32 or key_residual1 or value_residual1):
    raise ValueError("packed K/V is admitted only with K32/V32")
  KEY_QUANT_GROUP = key_group
  VALUE_QUANT_GROUP = value_group
  KEY_SCALE_GROUPS = HEAD_DIM // key_group
  VALUE_SCALE_GROUPS = HEAD_DIM // value_group
  KEY_SCALE_BYTES = KEY_SCALE_GROUPS * 2
  VALUE_SCALE_BYTES = VALUE_SCALE_GROUPS * 2
  KEY_RESIDUAL1 = key_residual1
  VALUE_RESIDUAL1 = value_residual1
  PACKED_KV_VARIANT = packed_kv_variant
  KEY_QUANT_BITS = (
      int(PACKED_KV_VARIANT[1]) if PACKED_KV_VARIANT is not None else 8)
  VALUE_QUANT_BITS = (
      int(PACKED_KV_VARIANT[3]) if PACKED_KV_VARIANT is not None else 8)
  KEY_PACK_WORDS = HEAD_DIM * KEY_QUANT_BITS // 32
  VALUE_PACK_WORDS = HEAD_DIM * VALUE_QUANT_BITS // 32
  KEY_STATE_BYTES = KEY_SCALE_BYTES + (
      RESIDUAL1_BYTES if KEY_RESIDUAL1 else 0)
  VALUE_STATE_BYTES = VALUE_SCALE_BYTES + (
      RESIDUAL1_BYTES if VALUE_RESIDUAL1 else 0)


def quantize_group(
    values: Any, group: int, residual1: bool, np: Any,
    quant_bits: int = 8,
) -> tuple[Any, Any, Any, Any | None]:
  scale_groups = HEAD_DIM // group
  grouped = values.astype(np.float32).reshape(-1, scale_groups, group)
  maximum = np.max(np.abs(grouped), axis=2)
  quant_max = (1 << (quant_bits - 1)) - 1
  scales = np.where(maximum == 0.0, 1.0, maximum / float(quant_max))
  scale_half = scales.astype(np.float16)
  if residual1:
    fine = np.clip(np.rint(
        grouped * np.float32(2.0) /
        scale_half.astype(np.float32)[:, :, None]), -254, 254).astype(
            np.int16)
    base = np.floor_divide(fine + np.int16(1), np.int16(2)).astype(
        np.int16)
    residual = (fine - 2 * base + 1).astype(np.uint8)
    quantized = base.astype(np.int8)
    reconstructed = (
        quantized.astype(np.float16) +
        residual.astype(np.float16) * np.float16(0.5) - np.float16(0.5))
  else:
    quantized = np.clip(
        np.rint(grouped / scales[:, :, None]),
        -quant_max, quant_max).astype(np.int8)
    residual = None
    reconstructed = quantized.astype(np.float16)
  approximate = (
      reconstructed * scale_half[:, :, None]).reshape(
          -1, HEAD_DIM)
  residual_flat = (
      None if residual is None else residual.reshape(-1, HEAD_DIM))
  return (
      quantized.reshape(-1, HEAD_DIM), scale_half, approximate,
      residual_flat)


def scale_plane(
    state: Any, kv_head: int, scale_groups: int, np: Any,
) -> Any:
  payload = state[0, kv_head, 1:, :].reshape(-1)
  scale_bytes = scale_groups * 2 * FIXED_COLD_CAPACITY
  return payload[:scale_bytes].view(np.float16).reshape(
      scale_groups, FIXED_COLD_CAPACITY)


def key_residual1_plane(state: Any, kv_head: int, np: Any) -> Any:
  payload = state[0, kv_head, 1:, :].reshape(-1)
  begin = KEY_SCALE_BYTES * FIXED_COLD_CAPACITY
  end = begin + RESIDUAL1_BYTES * FIXED_COLD_CAPACITY
  return payload[begin:end].view(np.uint32).reshape(
      FIXED_COLD_CAPACITY // KEY_TILE_TOKENS,
      KEY_SCALE_GROUPS, KEY_TILE_TOKENS)


def value_residual1_plane(state: Any, kv_head: int, np: Any) -> Any:
  payload = state[0, kv_head, 1:, :].reshape(-1)
  begin = VALUE_SCALE_BYTES * FIXED_COLD_CAPACITY
  end = begin + RESIDUAL1_BYTES * FIXED_COLD_CAPACITY
  return payload[begin:end].view(np.uint16).reshape(
      HEAD_DIM, FIXED_COLD_CAPACITY // KEY_TILE_TOKENS)


def residual1_words32(bits: Any, np: Any) -> Any:
  weights = np.left_shift(
      np.uint32(1), np.arange(32, dtype=np.uint32))
  return np.sum(
      bits.reshape(-1, HEAD_DIM // 32, 32).astype(np.uint32) * weights,
      axis=2, dtype=np.uint32)


def residual1_append_bytes(bits: Any, np: Any) -> Any:
  return np.packbits(
      bits.astype(np.uint8), axis=1, bitorder="little").reshape(
          bits.shape[0], 1, RESIDUAL1_BYTES).view(np.int8)


def make_fixture(np: Any) -> tuple[dict[str, Any], dict[str, Any]]:
  query = np.empty((1, Q_HEADS, 1, HEAD_DIM), dtype=np.float16)
  query_tokens = np.arange(Q_HEADS, dtype=np.uint32)
  query_dims = np.arange(HEAD_DIM, dtype=np.uint32)
  query[0, :, 0, :] = unit_values(
      query_tokens, query_dims, 0x3c6ef372, np).astype(np.float16)

  hot_key = np.zeros(
      (1, KV_HEADS, HOT_KEY_STORAGE_BLOCKS,
       HOT_KEY_WORDS_PER_BLOCK), dtype=np.int32)
  hot_value = np.zeros(
      (1, KV_HEADS, PHYSICAL_HOT_CAPACITY, HEAD_DIM), dtype=np.float16)
  current_key = np.zeros((1, KV_HEADS, 1, HEAD_DIM), dtype=np.float16)
  current_value = np.zeros((1, KV_HEADS, 1, HEAD_DIM), dtype=np.float16)
  cold_key = np.zeros(
      (1, KV_HEADS, FIXED_COLD_CAPACITY + 1, HEAD_DIM), dtype=np.int8)
  cold_value = np.zeros_like(cold_key)
  cold_key_scale = np.zeros(
      (1, KV_HEADS, FIXED_COLD_CAPACITY + 1, KEY_STATE_BYTES),
      dtype=np.int8)
  cold_value_scale = np.zeros(
      (1, KV_HEADS, FIXED_COLD_CAPACITY + 1, VALUE_STATE_BYTES),
      dtype=np.int8)
  base_scores = np.empty((KV_HEADS, GQA_GROUP, KEY_TOKENS), dtype=np.float32)
  exact_scores = np.empty_like(base_scores)

  cold_digits = (
      COLD_TOKENS % 128, (COLD_TOKENS // 128) % 128,
      (COLD_TOKENS // 16384) % 128)
  cold_key[:, :, 0, :3] = np.asarray(cold_digits, dtype=np.int8)

  generation_chunk = 4096
  for kv_head in range(KV_HEADS):
    head_words = hot_key[0, kv_head].reshape(-1)
    dense_begin = HOT_KEY_BLOCKS * HOT_KEY_WORDS_PER_BLOCK
    value_begin = 2 * HOT_KEY_BLOCKS * HOT_KEY_WORDS_PER_BLOCK
    dense_key = head_words[dense_begin:].view(np.float16)[
        :PHYSICAL_HOT_CAPACITY * HEAD_DIM].reshape(
            PHYSICAL_HOT_CAPACITY, HEAD_DIM)
    dimension_value = head_words[value_begin:].view(np.float16)[
        :PHYSICAL_HOT_CAPACITY * HEAD_DIM].reshape(
            HEAD_DIM, PHYSICAL_HOT_CAPACITY)
    cold_key_payload = cold_key[0, kv_head, 1:, :].reshape(-1)
    cold_value_payload = cold_value[0, kv_head, 1:, :].reshape(
        HEAD_DIM, FIXED_COLD_CAPACITY)
    key_scale_payload = scale_plane(
        cold_key_scale, kv_head, KEY_SCALE_GROUPS, np)
    value_scale_payload = scale_plane(
        cold_value_scale, kv_head, VALUE_SCALE_GROUPS, np)
    key_residual_payload = (
        key_residual1_plane(cold_key_scale, kv_head, np)
        if KEY_RESIDUAL1 else None)
    value_residual_payload = (
        value_residual1_plane(cold_value_scale, kv_head, np)
        if VALUE_RESIDUAL1 else None)
    q_group = query[
        0, kv_head * GQA_GROUP:(kv_head + 1) * GQA_GROUP,
        0, :].astype(np.float32)
    packed_half = hot_key[0, kv_head, :HOT_KEY_BLOCKS, :]

    for begin in range(0, KEY_TOKENS, generation_chunk):
      end = min(KEY_TOKENS, begin + generation_chunk)
      tokens = np.arange(begin, end, dtype=np.uint32)
      dimensions = kv_head * HEAD_DIM + np.arange(HEAD_DIM, dtype=np.uint32)
      key_f32 = unit_values(tokens, dimensions, 0x51a7d3e1, np)
      value_f32 = (
          np.float32(0.1) + np.float32(0.5) *
          unit_values(tokens, dimensions, 0x7f4a7c15, np))
      key = key_f32.astype(np.float16)
      value = value_f32.astype(np.float16)
      exact_raw = key.astype(np.float32) @ q_group.T
      exact_scores[kv_head, :, begin:end] = (
          exact_raw.T * np.float32(ATTENTION_SCALE))
      base_scores[kv_head, :, begin:end] = exact_scores[
          kv_head, :, begin:end]

      history_end = min(end, HISTORY_TOKENS)
      if begin < history_end:
        count = history_end - begin
        dense_key[begin:history_end, :] = key[:count, :]
        dimension_value[:, begin:history_end] = value[:count, :].T
        hot_value[0, kv_head, begin:history_end, :] = value[:count, :]
        if begin % KEY_TILE_TOKENS != 0 or count % KEY_TILE_TOKENS != 0:
          raise AssertionError("history generation lost block16 alignment")
        block_begin = begin // KEY_TILE_TOKENS
        block_end = history_end // KEY_TILE_TOKENS
        pack_half_key(packed_half[block_begin:block_end], key[:count], np)
      if end > HISTORY_TOKENS:
        current_offset = HISTORY_TOKENS - begin
        current_key[0, kv_head, 0, :] = key[current_offset, :]
        current_value[0, kv_head, 0, :] = value[current_offset, :]

      cold_begin = begin
      cold_end = min(end, COLD_TOKENS)
      if cold_begin < cold_end:
        count = cold_end - cold_begin
        quantized_key, key_scales, approximate_key, key_residual = (
            quantize_group(
                key[:count, :], KEY_QUANT_GROUP, KEY_RESIDUAL1, np,
                KEY_QUANT_BITS))
        quantized_value, value_scales, _, value_residual = quantize_group(
            value[:count, :], VALUE_QUANT_GROUP, VALUE_RESIDUAL1, np,
            VALUE_QUANT_BITS)
        byte_begin = cold_begin * HEAD_DIM
        byte_end = cold_end * HEAD_DIM
        if PACKED_KV_VARIANT is not None:
          key_packed_begin = cold_begin * KEY_PACK_WORDS * 4
          key_packed_end = cold_end * KEY_PACK_WORDS * 4
          pack_lowbit_state(
              cold_key_payload[key_packed_begin:key_packed_end],
              quantized_key, KEY_QUANT_BITS, np)
          if PACKED_KV_VARIANT == "k7v8":
            cold_value_payload[:, cold_begin:cold_end] = quantized_value.T
          else:
            value_packed_begin = cold_begin * VALUE_PACK_WORDS * 4
            value_packed_end = cold_end * VALUE_PACK_WORDS * 4
            pack_lowbit_state(
                cold_value[0, kv_head, 1:, :].reshape(-1)[
                    value_packed_begin:value_packed_end],
                quantized_value, VALUE_QUANT_BITS, np)
        else:
          pack_i8_key(
              cold_key_payload[byte_begin:byte_end], quantized_key, np)
          cold_value_payload[:, cold_begin:cold_end] = quantized_value.T
        key_scale_payload[:, cold_begin:cold_end] = key_scales.T
        value_scale_payload[:, cold_begin:cold_end] = value_scales.T
        block_begin = cold_begin // KEY_TILE_TOKENS
        block_end = cold_end // KEY_TILE_TOKENS
        if key_residual_payload is not None:
          key_words = residual1_words32(key_residual, np)
          key_residual_payload[block_begin:block_end, :, :] = (
              key_words.reshape(-1, KEY_TILE_TOKENS, KEY_SCALE_GROUPS)
              .transpose(0, 2, 1))
        if value_residual_payload is not None:
          value_words = np.sum(
              value_residual.reshape(
                  -1, KEY_TILE_TOKENS, HEAD_DIM).astype(np.uint16) *
              np.left_shift(
                  np.uint16(1),
                  np.arange(KEY_TILE_TOKENS, dtype=np.uint16))[None, :, None],
              axis=1, dtype=np.uint16)
          value_residual_payload[:, block_begin:block_end] = value_words.T
        approximate_raw = approximate_key.astype(np.float32) @ q_group.T
        if not KEY_EXACT:
          base_scores[kv_head, :, cold_begin:cold_end] = (
              approximate_raw.T * np.float32(ATTENTION_SCALE))
        if cold_begin == 0:
          base_scores[kv_head, :, :SINK_TOKENS] = (
              exact_scores[kv_head, :, :SINK_TOKENS])

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
          (1, KV_HEADS, 1, HEAD_DIM), dtype=np.int8),
      "eviction_count": np.ones((1, 1, 1, 1), dtype=np.int32),
      "decode_length": np.full(
          (1, 1, 1, MAX_CHUNKS), KEY_TOKENS, dtype=np.int32),
      "dynamic_shape_infer_carrier": np.zeros(
          (1, 1, 1, 1), dtype=np.float32),
  }
  reference = {
      "base_scores": base_scores,
      "exact_scores": exact_scores,
  }
  return feed, reference


def ordered_half(bits: Any, np: Any) -> Any:
  values = bits.astype(np.uint16, copy=False)
  return np.where(
      (values & np.uint16(0x8000)) != 0,
      np.bitwise_not(values), values ^ np.uint16(0x8000)).astype(np.uint16)


def analyze_workspace(workspace: Any, np: Any) -> dict[str, Any]:
  offsets = workspace_offsets()
  words = workspace.reshape(-1).view(np.uint32)
  candidate_begin = offsets["local_candidates"]
  candidate_count = Q_HEADS * MAX_COLD_CHUNKS * LOCAL_TOPK
  candidates = words[
      candidate_begin:candidate_begin + candidate_count].reshape(
          Q_HEADS, MAX_COLD_CHUNKS, LOCAL_TOPK).copy()
  union_words = (MAX_COLD_TOKENS + 31) // 32
  union_begin = offsets["union_bits"]
  union = words[union_begin:union_begin + KV_HEADS * union_words].reshape(
      KV_HEADS, union_words).copy()
  scores = workspace.reshape(-1)[
      offsets["approximate_score"]:
      offsets["approximate_score"] + Q_HEADS * MAX_TOKENS].reshape(
          Q_HEADS, MAX_TOKENS).copy()
  completion_begin = offsets["completion"]
  completion = words[
      completion_begin:completion_begin + KV_HEADS].copy()
  attention = workspace.reshape(-1)[
      offsets["attention"]:offsets["attention"] + Q_HEADS * HEAD_DIM].reshape(
          1, Q_HEADS, 1, HEAD_DIM).copy()

  shape_pass = True
  score_identity_pass = True
  expected_union = np.zeros_like(union)
  for q_head in range(Q_HEADS):
    pool = candidates[q_head, :COLD_CHUNKS, :].reshape(-1)
    for chunk in range(COLD_CHUNKS):
      records = candidates[q_head, chunk, :]
      tokens = (records & np.uint32(0xffff)).astype(np.uint32)
      inside = np.logical_and(
          tokens >= chunk * CHUNK_TOKENS,
          tokens < (chunk + 1) * CHUNK_TOKENS)
      shape_pass = shape_pass and bool(inside.all())
      shape_pass = shape_pass and len(np.unique(tokens)) == LOCAL_TOPK
      score_bits = (records >> np.uint32(16)).astype(np.uint16)
      stored_score_bits = (
          scores[q_head, tokens] * np.float32(ATTENTION_SCALE)).astype(
              np.float16).view(np.uint16)
      score_identity_pass = score_identity_pass and bool(np.array_equal(
          score_bits, stored_score_bits))
    tokens = (pool & np.uint32(0xffff)).astype(np.uint32)
    rank = ordered_half(
        (pool >> np.uint32(16)).astype(np.uint16), np).astype(np.int64)
    order = np.lexsort((tokens.astype(np.int64), -rank))
    selected = tokens[order[:TOPK]]
    kv_head = q_head // GQA_GROUP
    for token in selected.tolist():
      expected_union[kv_head, token // 32] |= np.uint32(1 << (token & 31))
  union_counts = [
      sum(int(value).bit_count() for value in union[kv_head].tolist())
      for kv_head in range(KV_HEADS)]
  selected_tokens = []
  for kv_head in range(KV_HEADS):
    tokens = []
    for word_index, value in enumerate(union[kv_head].tolist()):
      for bit in range(32):
        token = word_index * 32 + bit
        if token < COLD_TOKENS and value & (1 << bit):
          tokens.append(token)
    selected_tokens.append(np.asarray(tokens, dtype=np.int64))
  return {
      "attention": attention,
      "candidates": candidates,
      "completion": completion,
      "expected_union": expected_union,
      "local_candidate_shape_pass": shape_pass,
      "local_candidate_score_identity_pass": score_identity_pass,
      "scores": scores,
      "selected_tokens": selected_tokens,
      "union": union,
      "union_counts": union_counts,
      "union_exact": bool(np.array_equal(union, expected_union)),
  }


def adaptive_reference(
    feed: dict[str, Any], reference: dict[str, Any], workspace: dict[str, Any],
    np: Any,
) -> Any:
  output = np.empty((1, Q_HEADS, 1, HEAD_DIM), dtype=np.float32)
  base_scores = reference["base_scores"]
  exact_scores = reference["exact_scores"]
  for kv_head in range(KV_HEADS):
    exact_value = np.concatenate((
        feed["hot_value"][0, kv_head, :HISTORY_TOKENS, :],
        feed["current_value"][0, kv_head, :, :]), axis=0)
    raw_value_payload = feed["cold_value"][0, kv_head, 1:, :].reshape(-1)
    scale_payload = scale_plane(
        feed["cold_value_scale"], kv_head, VALUE_SCALE_GROUPS, np)
    quantized_value = (
        unpack_lowbit_state(
            raw_value_payload, COLD_TOKENS, VALUE_QUANT_BITS, np)
        if PACKED_KV_VARIANT is not None and PACKED_KV_VARIANT != "k7v8" else
        raw_value_payload.reshape(
            HEAD_DIM, FIXED_COLD_CAPACITY)[:, :COLD_TOKENS].T)
    reconstructed_value = quantized_value.reshape(
        COLD_TOKENS, VALUE_SCALE_GROUPS,
        VALUE_QUANT_GROUP).astype(np.float16)
    if VALUE_RESIDUAL1:
      residual_words = value_residual1_plane(
          feed["cold_value_scale"], kv_head, np)
      tokens = np.arange(COLD_TOKENS, dtype=np.uint32)
      residual = (
          (residual_words[:, tokens // KEY_TILE_TOKENS].T >>
           (tokens % KEY_TILE_TOKENS)[:, None]) & np.uint16(1)).astype(
               np.float16)
      reconstructed_value += (
          residual * np.float16(0.5) - np.float16(0.5)).reshape(
              COLD_TOKENS, VALUE_SCALE_GROUPS, VALUE_QUANT_GROUP)
    approximate_value = (
        reconstructed_value *
        scale_payload[:, :COLD_TOKENS].T[:, :, None]).reshape(
            COLD_TOKENS, HEAD_DIM)
    base_value = exact_value.copy()
    base_value[SINK_TOKENS:COLD_TOKENS, :] = (
        approximate_value[SINK_TOKENS:, :])
    selected = workspace["selected_tokens"][kv_head]
    for head in range(GQA_GROUP):
      q_head = kv_head * GQA_GROUP + head
      scores = base_scores[kv_head, head].copy()
      values = base_value.copy()
      scores[selected] = exact_scores[kv_head, head, selected]
      values[selected, :] = exact_value[selected, :]
      partition_numerators = []
      partition_sums = []
      partition_maxima = []
      for begin in range(0, KEY_TOKENS, PARTITION_TOKENS):
        end = min(begin + PARTITION_TOKENS, KEY_TOKENS)
        partition_scores = scores[begin:end]
        maximum = float(np.max(partition_scores))
        weights = np.exp(
            partition_scores - np.float32(maximum)).astype(np.float32)
        numerator = weights.astype(np.float16).astype(np.float32) @ \
            values[begin:end, :].astype(np.float32)
        partition_numerators.append(numerator)
        partition_sums.append(float(np.sum(weights, dtype=np.float32)))
        partition_maxima.append(maximum)

      running_max = -math.inf
      numerator = np.zeros((HEAD_DIM,), dtype=np.float32)
      denominator = np.float32(0.0)
      for index, maximum in enumerate(partition_maxima):
        next_max = max(running_max, maximum)
        previous_scale = np.float32(math.exp(running_max - next_max))
        partition_scale = np.float32(math.exp(maximum - next_max))
        numerator = numerator * previous_scale + \
            partition_numerators[index] * partition_scale
        denominator = denominator * previous_scale + \
            np.float32(partition_sums[index]) * partition_scale
        running_max = next_max
      output[0, q_head, 0, :] = numerator / denominator
  return output


def vector_metrics(reference: Any, candidate: Any, np: Any) -> dict[str, Any]:
  expected = reference.astype(np.float64).reshape(-1)
  observed = candidate.astype(np.float64).reshape(-1)
  difference = observed - expected
  expected_norm = float(np.linalg.norm(expected))
  observed_norm = float(np.linalg.norm(observed))
  return {
      "cosine": float(np.dot(expected, observed) /
                      (expected_norm * observed_norm)),
      "finite": bool(np.isfinite(observed).all()),
      "max_abs": float(np.max(np.abs(difference))),
      "relative_l2": float(np.linalg.norm(difference) / expected_norm),
      "rmse": float(np.sqrt(np.mean(difference * difference))),
  }


def quantized_append(
    values: Any, group: int, residual1: bool, np: Any,
    quant_bits: int = 8,
) -> tuple[Any, Any, Any | None]:
  quantized, scales, _, residual = quantize_group(
      values.reshape(-1, HEAD_DIM), group, residual1, np, quant_bits)
  scale_bytes = scales.view(np.uint8).reshape(
      values.shape[0], 1, HEAD_DIM // group * 2).view(np.int8)
  payload = (
      np.concatenate((scale_bytes, residual1_append_bytes(residual, np)), axis=2)
      if residual is not None else scale_bytes)
  append = quantized.reshape(values.shape[0], 1, HEAD_DIM)
  if quant_bits != 8:
    compact_bytes = HEAD_DIM * quant_bits // 8
    append = np.zeros_like(append)
    append[:, 0, :compact_bytes] = packed_code_words(
        quantized, quant_bits, np).view(np.uint8).reshape(
            values.shape[0], compact_bytes).view(np.int8)
  return append, payload, residual


def packed_append_equal(
    observed: Any, expected: Any, quant_bits: int, np: Any,
) -> bool:
  payload_bytes = HEAD_DIM * quant_bits // 8
  return bool(np.array_equal(
      observed[..., :payload_bytes], expected[..., :payload_bytes]))


def copy_remote(remote: Any, element_type: Any, shape: Any, context: Any,
                np: Any) -> Any:
  host = context.create_host_tensor(element_type, shape)
  remote.copy_to(host)
  return np.array(host.data, copy=True)


def any_value(value: Any) -> Any:
  try:
    return value.value
  except Exception:
    return str(value)


def runtime_rows(compiled: Any) -> list[dict[str, Any]]:
  rows = []
  for node in compiled.get_runtime_model().get_ordered_ops():
    info = {str(key): any_value(value)
            for key, value in node.get_rt_info().items()}
    name = str(node.get_friendly_name())
    layer_type = str(info.get("layerType", ""))
    if "adaptive" not in name.lower() and layer_type != "CustomGPUPrimitive":
      continue
    rows.append({
        "layer_type": layer_type,
        "node_name": name,
        "node_type": str(node.get_type_name()),
        "output_layouts": str(info.get("outputLayouts", "")),
        "output_precisions": str(info.get("outputPrecisions", "")),
        "primitive_type": str(info.get("primitiveType", "")),
        "runtime_precision": str(info.get("runtimePrecision", "")),
    })
  return rows


def profile_rows(request: Any) -> list[dict[str, Any]]:
  rows = []
  for profile in request.get_profiling_info():
    if (profile.node_name != "iq36_adaptive_layer3_top512_boundary" and
        "IQ36Adaptive" not in profile.node_type):
      continue
    rows.append({
        "cpu_time_us": profile.cpu_time.total_seconds() * 1_000_000.0,
        "exec_type": profile.exec_type,
        "node_name": profile.node_name,
        "node_type": profile.node_type,
        "real_time_us": profile.real_time.total_seconds() * 1_000_000.0,
        "status": str(profile.status),
    })
  return rows


def stage_profile_rows(path: Path) -> list[dict[str, Any]]:
  if not path.is_file():
    return []
  rows: list[dict[str, Any]] = []
  for line_number, line in enumerate(
      path.read_text(encoding="utf-8").splitlines(), start=1):
    if not line.strip():
      continue
    payload = json.loads(line)
    stages = payload.get("stages")
    if not isinstance(stages, list):
      raise ValueError(
          f"stage profile row {line_number} has no stages list")
    rows.append({
        "request": len(rows),
        "stages": stages,
        "total_executing_ns": sum(
            int(stage.get("executing_ns", -1)) for stage in stages),
    })
  return rows


def make_model(ov: Any) -> Any:
  graph = load_graph_module()
  specifications = (
      ("query", ov.Type.f16, [1, Q_HEADS, 1, HEAD_DIM]),
      ("hot_key_bits", ov.Type.i32,
       [1, KV_HEADS, HOT_KEY_STORAGE_BLOCKS, HOT_KEY_WORDS_PER_BLOCK]),
      ("hot_value", ov.Type.f16,
       [1, KV_HEADS, PHYSICAL_HOT_CAPACITY, HEAD_DIM]),
      ("current_key", ov.Type.f16, [1, KV_HEADS, 1, HEAD_DIM]),
      ("current_value", ov.Type.f16, [1, KV_HEADS, 1, HEAD_DIM]),
      ("cold_key", ov.Type.i8,
       [1, KV_HEADS, FIXED_COLD_CAPACITY + 1, HEAD_DIM]),
      ("cold_value", ov.Type.i8,
       [1, KV_HEADS, FIXED_COLD_CAPACITY + 1, HEAD_DIM]),
      ("cold_key_scale", ov.Type.i8,
       [1, KV_HEADS, FIXED_COLD_CAPACITY + 1, KEY_STATE_BYTES]),
      ("cold_value_scale", ov.Type.i8,
       [1, KV_HEADS, FIXED_COLD_CAPACITY + 1, VALUE_STATE_BYTES]),
      ("mask", ov.Type.f32, [1, 1, 1, 1]),
      ("eviction_shape", ov.Type.i8, [1, KV_HEADS, 1, HEAD_DIM]),
      ("eviction_count", ov.Type.i32, [1, 1, 1, 1]),
      ("decode_length", ov.Type.i32, [1, 1, 1, MAX_CHUNKS]),
  )
  parameters = [
      ov.opset13.parameter(shape, element_type, name=name)
      for name, element_type, shape in specifications]
  dynamic = ov.opset13.parameter(
      ov.PartialShape([1, 1, -1, 1]), ov.Type.f32,
      name="dynamic_shape_infer_carrier")
  parameters.append(dynamic)
  operation_class = graph.adaptive_attention_custom_class(
      ov, TOPK, VALUE_QUANT_GROUP, KEY_RESIDUAL1, VALUE_RESIDUAL1,
      KEY_EXACT, PACKED_KV_VARIANT)
  operation = operation_class([
      value.output(0) for value in parameters[:len(specifications)]])
  operation.set_friendly_name("iq36_adaptive_layer3_top512_boundary")
  results = [
      ov.opset13.result(operation.output(index))
      for index in range(operation.get_output_size())]
  results.append(ov.opset13.result(dynamic.output(0)))
  model = ov.Model(results, parameters, "iq36_adaptive_layer3_boundary")
  model.validate_nodes_and_infer_types()
  return model


def state_publication_checks(
    remote_inputs: dict[str, Any], compiled: Any, context: Any,
    feed: dict[str, Any], np: Any,
) -> dict[str, Any]:
  slot = HISTORY_TOKENS
  expected_key = feed["current_key"][0, :, 0, :]
  expected_value = feed["current_value"][0, :, 0, :]
  exact_evicted_key = np.empty((KV_HEADS, HEAD_DIM), dtype=np.float16)
  for kv_head in range(KV_HEADS):
    words = feed["hot_key_bits"][0, kv_head].reshape(-1)
    dense_begin = HOT_KEY_BLOCKS * HOT_KEY_WORDS_PER_BLOCK
    dense = words[dense_begin:].view(np.float16)[
        :PHYSICAL_HOT_CAPACITY * HEAD_DIM].reshape(
            PHYSICAL_HOT_CAPACITY, HEAD_DIM)
    exact_evicted_key[kv_head] = dense[COLD_TOKENS]
  exact_evicted_value = feed["hot_value"][0, :, COLD_TOKENS, :]
  expected_key_q, expected_key_payload, expected_key_residual = (
      quantized_append(
          exact_evicted_key, KEY_QUANT_GROUP, KEY_RESIDUAL1, np,
          KEY_QUANT_BITS))
  expected_value_q, expected_value_payload, expected_value_residual = (
      quantized_append(
          exact_evicted_value, VALUE_QUANT_GROUP, VALUE_RESIDUAL1, np,
          VALUE_QUANT_BITS))

  hot_key = copy_remote(
      remote_inputs["hot_key_bits"], compiled.input("hot_key_bits").element_type,
      compiled.input("hot_key_bits").shape, context, np)
  hot_key_pass = True
  dense_key_pass = True
  dimension_value_pass = True
  for kv_head in range(KV_HEADS):
    blocks = hot_key[0, kv_head, :HOT_KEY_BLOCKS, :]
    block = slot // KEY_TILE_TOKENS
    lane = slot & (KEY_TILE_TOKENS - 1)
    expected_pairs = expected_key[kv_head].view(np.uint16).reshape(-1, 2)
    expected_words = (
        expected_pairs[:, 0].astype(np.uint32) |
        (expected_pairs[:, 1].astype(np.uint32) << np.uint32(16)))
    hot_key_pass = hot_key_pass and bool(np.array_equal(
        blocks[block, lane::KEY_TILE_TOKENS].view(np.uint32), expected_words))
    words = hot_key[0, kv_head].reshape(-1)
    dense_begin = HOT_KEY_BLOCKS * HOT_KEY_WORDS_PER_BLOCK
    value_begin = 2 * HOT_KEY_BLOCKS * HOT_KEY_WORDS_PER_BLOCK
    dense = words[dense_begin:].view(np.float16)[
        :PHYSICAL_HOT_CAPACITY * HEAD_DIM].reshape(
            PHYSICAL_HOT_CAPACITY, HEAD_DIM)
    dimension = words[value_begin:].view(np.float16)[
        :PHYSICAL_HOT_CAPACITY * HEAD_DIM].reshape(
            HEAD_DIM, PHYSICAL_HOT_CAPACITY)
    dense_key_pass = dense_key_pass and bool(np.array_equal(
        dense[slot], expected_key[kv_head]))
    dimension_value_pass = dimension_value_pass and bool(np.array_equal(
        dimension[:, slot], expected_value[kv_head]))
  del hot_key

  hot_value = copy_remote(
      remote_inputs["hot_value"], compiled.input("hot_value").element_type,
      compiled.input("hot_value").shape, context, np)
  hot_value_pass = bool(np.array_equal(
      hot_value[0, :, slot, :], expected_value))
  del hot_value

  cold_key = copy_remote(
      remote_inputs["cold_key"], compiled.input("cold_key").element_type,
      compiled.input("cold_key").shape, context, np)
  cold_value = copy_remote(
      remote_inputs["cold_value"], compiled.input("cold_value").element_type,
      compiled.input("cold_value").shape, context, np)
  cold_key_scale = copy_remote(
      remote_inputs["cold_key_scale"],
      compiled.input("cold_key_scale").element_type,
      compiled.input("cold_key_scale").shape, context, np)
  cold_value_scale = copy_remote(
      remote_inputs["cold_value_scale"],
      compiled.input("cold_value_scale").element_type,
      compiled.input("cold_value_scale").shape, context, np)
  desired_cold_tokens = COLD_TOKENS + 1
  desired_digits = np.asarray((
      desired_cold_tokens % 128,
      (desired_cold_tokens // 128) % 128,
      (desired_cold_tokens // 16384) % 128), dtype=np.int8)
  header_pass = bool(np.array_equal(
      cold_key[0, :, 0, :3], np.tile(desired_digits, (KV_HEADS, 1))))
  key_payload_pass = True
  value_payload_pass = True
  key_scale_pass = True
  value_scale_pass = True
  key_residual_pass = True
  value_residual_pass = True
  for kv_head in range(KV_HEADS):
    payload = cold_key[0, kv_head, 1:, :].reshape(-1)
    token_block = COLD_TOKENS // KEY_TILE_TOKENS
    token_lane = COLD_TOKENS & (KEY_TILE_TOKENS - 1)
    if PACKED_KV_VARIANT is not None:
      observed_key = packed_state_token_bytes(
          payload, COLD_TOKENS, KEY_PACK_WORDS, np)
      key_payload_pass = key_payload_pass and bool(np.array_equal(
          observed_key,
          expected_key_q[kv_head, 0, :KEY_PACK_WORDS * 4]))
      if PACKED_KV_VARIANT == "k7v8":
        value_payload = cold_value[0, kv_head, 1:, :].reshape(
            HEAD_DIM, FIXED_COLD_CAPACITY)
        value_payload_pass = value_payload_pass and bool(np.array_equal(
            value_payload[:, COLD_TOKENS], expected_value_q[kv_head, 0]))
      else:
        observed_value = packed_state_token_bytes(
            cold_value[0, kv_head, 1:, :].reshape(-1), COLD_TOKENS,
            VALUE_PACK_WORDS, np)
        value_payload_pass = value_payload_pass and bool(np.array_equal(
            observed_value,
            expected_value_q[kv_head, 0, :VALUE_PACK_WORDS * 4]))
    else:
      observed_key = np.empty((HEAD_DIM,), dtype=np.int8)
      for dim in range(HEAD_DIM):
        word = ((token_block * KEY_PACK_WORDS + dim // 4) *
                KEY_TILE_TOKENS + token_lane)
        observed_key[dim] = payload[word * 4 + dim % 4]
      key_payload_pass = key_payload_pass and bool(np.array_equal(
          observed_key, expected_key_q[kv_head, 0]))
      value_payload = cold_value[0, kv_head, 1:, :].reshape(
          HEAD_DIM, FIXED_COLD_CAPACITY)
      value_payload_pass = value_payload_pass and bool(np.array_equal(
          value_payload[:, COLD_TOKENS], expected_value_q[kv_head, 0]))
    key_scales = scale_plane(
        cold_key_scale, kv_head, KEY_SCALE_GROUPS, np)
    value_scales = scale_plane(
        cold_value_scale, kv_head, VALUE_SCALE_GROUPS, np)
    key_scale_pass = key_scale_pass and bool(np.array_equal(
        np.ascontiguousarray(key_scales[:, COLD_TOKENS]).view(np.uint8),
        expected_key_payload[
            kv_head, 0, :KEY_SCALE_BYTES].view(np.uint8)))
    value_scale_pass = value_scale_pass and bool(np.array_equal(
        np.ascontiguousarray(value_scales[:, COLD_TOKENS]).view(np.uint8),
        expected_value_payload[
            kv_head, 0, :VALUE_SCALE_BYTES].view(np.uint8)))
    if KEY_RESIDUAL1:
      observed_words = key_residual1_plane(
          cold_key_scale, kv_head, np)[token_block, :, token_lane]
      expected_words = residual1_words32(
          expected_key_residual[kv_head:kv_head + 1], np)[0]
      key_residual_pass = key_residual_pass and bool(np.array_equal(
          observed_words, expected_words))
    if VALUE_RESIDUAL1:
      observed_bits = (
          value_residual1_plane(
              cold_value_scale, kv_head, np)[:, token_block] >>
          np.uint16(token_lane)) & np.uint16(1)
      value_residual_pass = value_residual_pass and bool(np.array_equal(
          observed_bits.astype(np.uint8),
          expected_value_residual[kv_head]))
  return {
      "cold_header_count_exact": header_pass,
      "cold_key_payload_exact": key_payload_pass,
      "cold_key_scale_exact": key_scale_pass,
      "cold_key_residual1_exact": key_residual_pass,
      "cold_value_payload_exact": value_payload_pass,
      "cold_value_scale_exact": value_scale_pass,
      "cold_value_residual1_exact": value_residual_pass,
      "dense_key_publication_exact": dense_key_pass,
      "dimension_value_publication_exact": dimension_value_pass,
      "hot_key_publication_exact": hot_key_pass,
      "hot_value_publication_exact": hot_value_pass,
      "readback_after_inference_only": True,
  }


def worker_main(config_path: Path) -> int:
  global TOPK, KEY_EXACT
  started = time.perf_counter_ns()
  result_path: Path | None = None
  try:
    if Path(sys.prefix).resolve() != OV_PYTHON.parent.parent.resolve():
      raise RuntimeError(f"worker requires {OV_PYTHON}, observed {sys.executable}")
    import numpy as np
    import openvino as ov

    config = load_json(config_path)
    TOPK = int(config.get("topk", 512))
    KEY_EXACT = bool(config.get("key_exact", False))
    configure_quantization(
        int(config.get("key_quant_group", 32)),
        int(config.get("value_quant_group", 32)),
        bool(config.get("key_residual1", False)),
        bool(config.get("value_residual1", False)),
        config.get("packed_kv_variant"))
    raw = Path(config["raw"])
    plugin = Path(config["plugin"])
    result_path = raw / "worker-result.json"
    registry = raw / "candidate-plugins.xml"
    registry.write_text(
        "<ie><plugins><plugin name=\"GPU\" location="
        f"{quoteattr(str(plugin.resolve()))}/></plugins></ie>\n",
        encoding="utf-8")
    core = ov.Core(str(registry))
    core.set_property("GPU", {"CONFIG_FILE": str(CUSTOM_XML.resolve())})
    context = core.get_default_context("GPU")
    model = make_model(ov)
    compile_started = time.perf_counter_ns()
    compiled = core.compile_model(model, context, {
        "ACTIVATIONS_SCALE_FACTOR": 0.0,
        "PERFORMANCE_HINT": "LATENCY",
        "PERF_COUNT": True,
    })
    compile_ms = (time.perf_counter_ns() - compile_started) / 1_000_000.0
    feed, reference = make_fixture(np)
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
    remote_output_shapes = []
    # The disconnected dynamic carrier is a one-element host diagnostic.  The
    # six custom outputs stay remote through the timed request.
    for port in compiled.outputs[:6]:
      shape = port.shape
      remote = context.create_tensor(port.element_type, shape, {})
      request.set_tensor(port.get_any_name(), remote)
      remote_outputs.append(remote)
      remote_output_shapes.append(shape)

    infer_started = time.perf_counter_ns()
    request.start_async()
    request.wait()
    first_wall_ms = (time.perf_counter_ns() - infer_started) / 1_000_000.0
    first_outputs = [
        copy_remote(remote, port.element_type, shape, context, np)
        for port, shape, remote in zip(
            compiled.outputs[:6], remote_output_shapes, remote_outputs)]
    first_outputs.append(np.array(
        request.get_output_tensor(6).data, copy=True))
    first_workspace = analyze_workspace(first_outputs[0], np)
    independent_reference = adaptive_reference(
        feed, reference, first_workspace, np)
    attention_metrics = vector_metrics(
        independent_reference, first_outputs[1].astype(np.float32), np)
    workspace_attention_metrics = vector_metrics(
        first_workspace["attention"], first_outputs[1].astype(np.float32), np)

    evicted_key = np.empty((KV_HEADS, HEAD_DIM), dtype=np.float16)
    for kv_head in range(KV_HEADS):
      words = feed["hot_key_bits"][0, kv_head].reshape(-1)
      dense_begin = HOT_KEY_BLOCKS * HOT_KEY_WORDS_PER_BLOCK
      dense = words[dense_begin:].view(np.float16)[
          :PHYSICAL_HOT_CAPACITY * HEAD_DIM].reshape(
              PHYSICAL_HOT_CAPACITY, HEAD_DIM)
      evicted_key[kv_head] = dense[COLD_TOKENS]
    evicted_value = feed["hot_value"][0, :, COLD_TOKENS, :]
    expected_key_q, expected_key_payload, _ = quantized_append(
        evicted_key, KEY_QUANT_GROUP, KEY_RESIDUAL1, np, KEY_QUANT_BITS)
    expected_value_q, expected_value_payload, _ = quantized_append(
        evicted_value, VALUE_QUANT_GROUP, VALUE_RESIDUAL1, np,
        VALUE_QUANT_BITS)
    append_checks = {
        "cold_key_append_exact": packed_append_equal(
            first_outputs[2], expected_key_q.reshape(
                1, KV_HEADS, 1, HEAD_DIM), KEY_QUANT_BITS, np),
        "cold_key_scale_append_exact": bool(np.array_equal(
            first_outputs[4], expected_key_payload.reshape(
                1, KV_HEADS, 1, KEY_STATE_BYTES))),
        "cold_value_append_exact": packed_append_equal(
            first_outputs[3], expected_value_q.reshape(
                1, KV_HEADS, 1, HEAD_DIM), VALUE_QUANT_BITS, np),
        "cold_value_scale_append_exact": bool(np.array_equal(
            first_outputs[5], expected_value_payload.reshape(
                1, KV_HEADS, 1, VALUE_STATE_BYTES))),
    }
    state_checks = state_publication_checks(
        remote_inputs, compiled, context, feed, np)

    infer_started = time.perf_counter_ns()
    request.start_async()
    request.wait()
    second_wall_ms = (time.perf_counter_ns() - infer_started) / 1_000_000.0
    second_workspace_array = copy_remote(
        remote_outputs[0], compiled.output(0).element_type,
        compiled.output(0).shape, context, np)
    second_attention = copy_remote(
        remote_outputs[1], compiled.output(1).element_type,
        compiled.output(1).shape, context, np)
    second_workspace = analyze_workspace(second_workspace_array, np)
    deterministic = {
        "attention_exact": bool(np.array_equal(
            first_outputs[1], second_attention)),
        "local_candidates_exact": bool(np.array_equal(
            first_workspace["candidates"], second_workspace["candidates"])),
        "union_exact": bool(np.array_equal(
            first_workspace["union"], second_workspace["union"])),
    }
    profile = profile_rows(request)
    stage_profile_path = Path(os.environ.get(
        "IQ36_ADAPTIVE_STAGE_PROFILE_PATH", raw / "stage-profile.jsonl"))
    stage_profile = stage_profile_rows(stage_profile_path)
    expected_stage_entries = [
        "iq36_adaptive_attention_partial",
        "iq36_adaptive_attention_select_reduce_union",
        "iq36_adaptive_attention_correct_normalize",
        "iq36_adaptive_attention_ordered_update",
    ]
    stage_profile_exact = (
        len(stage_profile) == 2 and
        all(
            [stage.get("entry") for stage in row["stages"]] ==
                expected_stage_entries and
            all(int(stage.get("executing_ns", -1)) > 0
                for stage in row["stages"])
            for row in stage_profile))
    runtime = runtime_rows(compiled)
    output_shapes = [list(value.shape) for value in first_outputs]
    expected_shapes = [
        [1, 1, 1, workspace_offsets()["allocated"]],
        [1, Q_HEADS, 1, HEAD_DIM],
        [1, KV_HEADS, 1, HEAD_DIM],
        [1, KV_HEADS, 1, HEAD_DIM],
        [1, KV_HEADS, 1, KEY_STATE_BYTES],
        [1, KV_HEADS, 1, VALUE_STATE_BYTES],
        [1, 1, 1, 1],
    ]
    required_checks = {
        "all_four_append_outputs_exact": all(append_checks.values()),
        "attention_finite": attention_metrics["finite"],
        "attention_matches_seq1673_reference": (
            attention_metrics["relative_l2"] <= 0.002 and
            attention_metrics["max_abs"] <= 0.002),
        "completion_counters_exact": bool(np.array_equal(
            first_workspace["completion"],
            np.full((KV_HEADS,), PARTITION_COUNT, dtype=np.uint32))),
        "exact_output_shapes": output_shapes == expected_shapes,
        "four_stage_profile_executes": (
            len(profile) == 1 and profile[0]["real_time_us"] > 0.0),
        "four_individual_stage_intervals_execute": stage_profile_exact,
        "layer3_top512_runtime_is_custom_gpu": (
            len(runtime) == 1 and
            runtime[0]["layer_type"] == "CustomGPUPrimitive"),
        "local_candidate_scores_match_sidecar": (
            first_workspace["local_candidate_score_identity_pass"]),
        "local_candidates_have_exact_shape": (
            first_workspace["local_candidate_shape_pass"]),
        "ordered_remote_state_publication_exact": all(state_checks.values()),
        "output_matches_f16_workspace_publication": bool(np.array_equal(
            first_workspace["attention"].astype(np.float16),
            first_outputs[1])),
        "two_inference_results_are_deterministic": all(deterministic.values()),
        "union_matches_exported_candidate_heaps": first_workspace["union_exact"],
        "union_rows_are_bounded": all(
            TOPK <= count <= TOPK * GQA_GROUP
            for count in first_workspace["union_counts"]),
    }
    payload = {
        "append_checks": append_checks,
        "attention_metrics": attention_metrics,
        "compile_ms": compile_ms,
        "deterministic": deterministic,
        "elapsed_ms": (time.perf_counter_ns() - started) / 1_000_000.0,
        "fixture": {
            "cold_chunks": COLD_CHUNKS,
            "cold_tokens": COLD_TOKENS,
            "history_tokens": HISTORY_TOKENS,
            "key_quant_group": KEY_QUANT_GROUP,
            "key_quant_bits": KEY_QUANT_BITS,
            "key_exact": KEY_EXACT,
            "key_residual1": KEY_RESIDUAL1,
            "key_tokens": KEY_TOKENS,
            "layer": 3,
            "topk": TOPK,
            "packed_kv_variant": PACKED_KV_VARIANT,
            "value_quant_bits": VALUE_QUANT_BITS,
            "value_quant_group": VALUE_QUANT_GROUP,
            "value_residual1": VALUE_RESIDUAL1,
        },
        "inference": {
            "first_wall_ms_diagnostic": first_wall_ms,
            "host_input_copy_in_timed_scope": False,
            "host_output_copy_in_timed_scope": False,
            "request_count": 2,
            "second_wall_ms_diagnostic": second_wall_ms,
        },
        "openvino_version": ov.get_version(),
        "output_shapes": output_shapes,
        "plugin": str(plugin.resolve()),
        "plugin_sha256": sha256(plugin),
        "profile": profile,
        "stage_profile": stage_profile,
        "stage_profile_path": str(stage_profile_path),
        "required_checks": required_checks,
        "required_checks_passed": all(required_checks.values()),
        "runtime": runtime,
        "state_checks": state_checks,
        "union_counts": first_workspace["union_counts"],
        "workspace_attention_metrics": workspace_attention_metrics,
    }
    write_json(result_path, payload)
    print(json.dumps({
        "attention_relative_l2": attention_metrics["relative_l2"],
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
      write_json(result_path, payload)
    print(json.dumps(payload, sort_keys=True), flush=True)
    return 2


def seq1673_admitted(payload: dict[str, Any]) -> bool:
  results = payload.get("results", {})
  top512 = results.get("512", {})
  target = top512.get("target_validation", {})
  return bool(
      payload.get("required_checks_passed") is True and
      payload.get("component_promoted") is True and
      top512.get("required_checks_passed") is True and
      target.get("candidate_shape_pass") is True and
      target.get("union_exact") is True and
      target.get("union_deterministic") is True)


def summary_markdown(payload: dict[str, Any]) -> str:
  result = payload.get("worker", {})
  metrics = result.get("attention_metrics", {})
  inference = result.get("inference", {})
  profile = result.get("profile", [])
  custom_us = profile[0].get("real_time_us") if profile else None
  stage_profile = result.get("stage_profile", [])
  latest_stages = stage_profile[-1].get("stages", []) \
      if stage_profile else []
  stage_us = {
      row.get("entry"): float(row.get("executing_ns", -1)) / 1000.0
      for row in latest_stages
  }
  return "\n".join([
      "# Adaptive-attention layer-3 boundary gate",
      "",
      f"- Verdict: `{payload['verdict']}`",
      f"- Required checks passed: "
      f"`{str(payload['required_checks_passed']).lower()}`",
      f"- Attention relative L2 / max abs: "
      f"`{metrics.get('relative_l2')} / {metrics.get('max_abs')}`",
      f"- KV-head union rows: `{result.get('union_counts')}`",
      f"- First / second diagnostic wall time: "
      f"`{inference.get('first_wall_ms_diagnostic')} / "
      f"{inference.get('second_wall_ms_diagnostic')} ms`",
      f"- Custom primitive PERF_COUNT diagnostic: `{custom_us} us`",
      f"- Second-request stage device times (us): `{stage_us}`",
      "",
      "This gate executes one isolated layer-3/top-512 component twice. It",
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
  memory = [memory_sample("start")]
  if int(memory[0]["available_bytes"]) < PREFLIGHT_BYTES:
    raise RuntimeError(
        f"8-GiB preflight failed: {memory[0]['available_bytes']} bytes")
  git = git_state(out)
  compile_gate = load_json(COMPILE_GATE)
  seq1673 = load_json(SEQ1673)
  expected_plugin_sha = str(compile_gate.get("plugin_sha256", ""))
  plugin_sha = sha256(plugin) if plugin.is_file() else ""

  config = raw / "worker-config.json"
  write_json(config, {
      "key_quant_group": args.key_quant_group,
      "key_exact": args.key_exact,
      "key_residual1": args.key_residual1,
      "packed_kv_variant": args.packed_kv_variant,
      "plugin": str(plugin),
      "raw": str(raw),
      "topk": args.topk,
      "value_quant_group": args.value_quant_group,
      "value_residual1": args.value_residual1,
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
  worker_run = run_monitored_worker(
      command, environment, args.timeout_s, stop_bytes, memory,
      raw / "worker.stdout", raw / "worker.stderr")
  memory.append(memory_sample("finish"))
  worker = load_json(raw / "worker-result.json") \
      if (raw / "worker-result.json").is_file() else {}
  resources = parse_time(time_path)
  write_json(raw / "worker-command.json", {
      **worker_run,
      "environment": {
          "IQ36_ADAPTIVE_STAGE_PROFILE_PATH":
              environment["IQ36_ADAPTIVE_STAGE_PROFILE_PATH"],
          "NEO_CACHE_DIR": environment["NEO_CACHE_DIR"],
      },
      "resources": resources,
  })
  source_text = SOURCE_RUNNER.read_text(encoding="utf-8")
  seq1673_union_rows = seq1673.get(
      "results", {}).get("512", {}).get(
          "target_validation", {}).get("union_rows")
  fixture_source_pass = all(fragment in source_text for fragment in (
      "value *= 0x7feb352dU", "value *= 0x846ca68bU",
      "0x51a7d3e1U", "0x7f4a7c15U", "0x3c6ef372U"))
  memory_pass = (
      not worker_run["guard_tripped"] and
      all(int(row["available_bytes"]) >= stop_bytes for row in memory))
  checks = [
      check("repository_clean_at_gate",
            not git["dirty"] or args.allow_dirty, git=git,
            allow_dirty=args.allow_dirty),
      check("compile_gate_admits_exactly_layer3_boundary",
            compile_gate.get("required_checks_passed") is True and
            compile_gate.get("layer3_boundary_worker_admitted") is True and
            compile_gate.get("model_worker_admitted") is False and
            compile_gate.get("product_worker_admitted") is False),
      check("isolated_plugin_matches_compile_gate",
            plugin.is_file() and plugin_sha == expected_plugin_sha,
            plugin=relative(plugin), sha256=plugin_sha),
      check("seq1673_top512_reference_is_promoted", seq1673_admitted(seq1673)),
      check("fixture_literals_match_seq1673_source", fixture_source_pass,
            source=relative(SOURCE_RUNNER), source_sha256=sha256(SOURCE_RUNNER)),
      check("standalone_kernel_source_is_locked", SOURCE_KERNEL.is_file(),
            source=relative(SOURCE_KERNEL), source_sha256=sha256(SOURCE_KERNEL)),
      check("single_serial_worker_executes_two_requests",
            worker_run["returncode"] == 0 and
            worker.get("inference", {}).get("request_count") == 2),
      check("layer3_worker_required_checks_pass",
            worker.get("required_checks_passed") is True,
            worker_checks=worker.get("required_checks", {})),
      check("layer3_union_cardinality_matches_codec_contract",
            (worker.get("union_counts") == seq1673_union_rows
             if args.topk == 512 and not args.key_residual1 and
             not args.key_exact else
             isinstance(worker.get("union_counts"), list) and
             len(worker.get("union_counts")) == KV_HEADS and
             all(args.topk <= int(count) <= args.topk * GQA_GROUP
                 for count in worker.get("union_counts"))),
            observed=worker.get("union_counts"), expected=seq1673_union_rows),
      check("worker_peak_rss_is_bounded",
            int(resources.get("maximum_resident_kib", 0)) > 0 and
            int(resources.get("maximum_resident_kib", 0)) < 8 * 1024 * 1024,
            maximum_resident_kib=resources.get("maximum_resident_kib")),
      check("worker_does_not_swap", int(resources.get("swaps", -1)) == 0,
            swaps=resources.get("swaps")),
      check("memory_guard_never_tripped", memory_pass,
            stop_bytes=stop_bytes,
            minimum_available_bytes=min(
                int(row["available_bytes"]) for row in memory)),
  ]
  required = all(row["pass"] for row in checks)
  verdict = (
      "admit_adaptive_attention_layer3_captured_graph_boundary"
      if required else "repair_adaptive_attention_layer3_boundary")
  payload = {
      "checks": checks,
      "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
      "git": git,
      "long_worker_admitted": False,
      "memory": memory,
      "model_worker_admitted": False,
      "next_route": NEXT_ROUTE if required else ROUTE,
      "product_worker_admitted": False,
      "quantization": {
          "key_group": args.key_quant_group,
          "key_exact": args.key_exact,
          "key_residual1": args.key_residual1,
          "packed_kv_variant": args.packed_kv_variant,
          "topk": args.topk,
          "value_group": args.value_quant_group,
          "value_residual1": args.value_residual1,
      },
      "required_checks_passed": required,
      "route": ROUTE,
      "schema_version": SCHEMA,
      "sources": {
          "compile_gate": relative(COMPILE_GATE),
          "custom_xml": relative(CUSTOM_XML),
          "graph_module": relative(GRAPH_MODULE),
          "seq1673": relative(SEQ1673),
      },
      "verdict": verdict,
      "worker": worker,
      "worker_resources": resources,
      "worker_run": worker_run,
      "workstream": WS,
  }
  write_json(out / "gate.json", payload)
  (out / "summary.md").write_text(
      summary_markdown(payload), encoding="utf-8")
  print(json.dumps({
      "attention_relative_l2": worker.get(
          "attention_metrics", {}).get("relative_l2"),
      "output": relative(out),
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
