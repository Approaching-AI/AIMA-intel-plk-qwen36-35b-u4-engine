#!/usr/bin/env python3
"""Bound remaining decode-FC hardware-limit routes without a model worker.

This gate audits the exact five non-LM compressed-FC cohorts from the locked
OpenVINO IR.  It answers three narrow questions before any new kernel work:

* whether block-2D/DPAS or DPASW is a genuinely new data-movement route;
* whether metadata relayout alone can close the current FC residual; and
* whether exact lossless parameter coding leaves enough payload and decode
  margin to justify a software entropy-decoder implementation.

The model BIN is read in bounded chunks.  The only child programs are the
offline ``ocloc disasm`` decoder and single-threaded ``zstd`` compressors whose
stdout is counted and discarded.  No compiler, GPU context, or model worker is
started.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import re
import resource
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-openvino-fc-hardware-limit-bound-v1"

MODEL_XML = Path(
    "/home/intel/Qwen3.6-35B-A3B-ov/openvino_language_model.xml")
MODEL_BIN = MODEL_XML.with_suffix(".bin")
MODEL_CONTRACT = ROOT / "contracts/qwen36-35b-a3b-openvino-u4-model-contract.json"
TARGET_CONTRACT = ROOT / "contracts/intel-qwen36-target-contract.json"
FIXED_COMPONENT = ROOT / (
    "output/openvino-fc-micro-component-"
    "20260715Tseq1233-max-native-fused-nonzero-warm512-cleanZ/metrics.json")
REPRESENTATIVE_PROGRAM = ROOT / (
    "output/openvino-fc-micro-component-"
    "20260715Tseq1233-max-native-fused-nonzero-warm512-cleanZ/raw/"
    "m2048_k4096/codegen/m2048_k4096.program.bin")
PINNED_GEMM_STRATEGY = Path(
    "/home/intel/intel-qwen36-r0/source/openvino-90214e5be05/"
    "src/plugins/intel_gpu/thirdparty/onednn_gpu/src/gpu/intel/gemm/jit/"
    "generator/strategy.cpp")

EXPECTED_COHORTS = {
    "linear_attention_input": {
        "m": 12352, "k": 2048, "calls": 30, "tensor_count": 120},
    "full_attention_qkv": {
        "m": 9216, "k": 2048, "calls": 10, "tensor_count": 30},
    "router_shared_input": {
        "m": 1281, "k": 2048, "calls": 40, "tensor_count": 160},
    "attention_output": {
        "m": 2048, "k": 4096, "calls": 40, "tensor_count": 40},
    "shared_expert_down": {
        "m": 2048, "k": 512, "calls": 40, "tensor_count": 40},
}

COHORT_SUFFIXES = {
    "linear_attention_input": (
        ".linear_attn.in_proj_qkv.weight_compressed",
        ".linear_attn.in_proj_a.weight_compressed",
        ".linear_attn.in_proj_b.weight_compressed",
        ".linear_attn.in_proj_z.weight_compressed",
    ),
    "full_attention_qkv": (
        ".self_attn.q_proj.weight",
        ".self_attn.k_proj.weight",
        ".self_attn.v_proj.weight",
    ),
    "router_shared_input": (
        ".mlp.gate.weight_compressed",
        ".mlp.shared_expert.gate_proj.weight",
        ".mlp.shared_expert.up_proj.weight",
        ".mlp.shared_expert_gate.weight",
    ),
    "attention_output": (
        ".linear_attn.out_proj.weight_compressed",
        ".self_attn.o_proj.weight",
    ),
    "shared_expert_down": (
        ".mlp.shared_expert.down_proj.weight",
    ),
}


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--memory-stop-gib", type=float, default=4.0)
  parser.add_argument("--chunk-mib", type=int, default=8)
  parser.add_argument("--zstd", type=Path, default=Path("/usr/bin/zstd"))
  parser.add_argument("--ocloc", type=Path, default=Path("/usr/bin/ocloc"))
  parser.add_argument("--zstd-levels", default="1,3,9")
  args = parser.parse_args()
  if args.memory_stop_gib <= 0 or args.chunk_mib <= 0:
    parser.error("memory stop and chunk size must be positive")
  try:
    args.zstd_levels = tuple(int(value) for value in args.zstd_levels.split(","))
  except ValueError:
    parser.error("zstd levels must be comma-separated integers")
  if not args.zstd_levels or any(value < 1 or value > 19
                                  for value in args.zstd_levels):
    parser.error("zstd levels must be in [1, 19]")
  return args


def load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise TypeError(f"expected JSON object: {path}")
  return value


def display_path(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path)


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    while chunk := stream.read(1024 * 1024):
      digest.update(chunk)
  return digest.hexdigest()


def available_memory_bytes() -> int:
  for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
    if line.startswith("MemAvailable:"):
      return int(line.split()[1]) * 1024
  raise RuntimeError("MemAvailable is missing")


def process_swap_bytes() -> int:
  for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
    if line.startswith("VmSwap:"):
      return int(line.split()[1]) * 1024
  raise RuntimeError("VmSwap is missing")


def sample_memory(
    label: str, stop_bytes: int, rows: list[dict[str, Any]],
) -> None:
  available = available_memory_bytes()
  rows.append({
      "label": label,
      "available_bytes": available,
      "self_swap_bytes": process_swap_bytes(),
  })
  if available < stop_bytes:
    raise RuntimeError(
        f"memory stop at {label}: {available} < {stop_bytes} bytes")


def git_state(output: Path) -> dict[str, Any]:
  commit = subprocess.run(
      ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
      capture_output=True, check=True).stdout.strip()
  status = subprocess.run(
      ["git", "status", "--porcelain"], cwd=ROOT, text=True,
      capture_output=True, check=True).stdout.splitlines()
  try:
    relative = str(output.resolve().relative_to(ROOT))
  except ValueError:
    relative = ""
  status = [row for row in status if not relative or relative not in row]
  return {"commit": commit, "dirty": bool(status), "dirty_paths": status}


def check(name: str, passed: bool, **detail: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **detail}


def shape(data: ET.Element) -> tuple[int, ...]:
  text = str(data.attrib.get("shape", ""))
  if not text:
    return ()
  return tuple(int(value.strip()) for value in text.split(","))


def classify_weight(name: str) -> str | None:
  matches = [
      cohort for cohort, suffixes in COHORT_SUFFIXES.items()
      if any(name.endswith(suffix) for suffix in suffixes)]
  if len(matches) > 1:
    raise ValueError(f"ambiguous FC cohort for {name}: {matches}")
  return matches[0] if matches else None


def parse_selected_tensors(path: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
  root = ET.parse(path).getroot()
  layers = {str(layer.attrib["name"]): layer
            for layer in root.findall("./layers/layer")}
  rows: list[dict[str, Any]] = []
  all_u4_weights = 0
  excluded_grouped_weights = 0
  for name, layer in layers.items():
    data = layer.find("data")
    if (layer.attrib.get("type") != "Const" or data is None
        or data.attrib.get("element_type") != "u4"):
      continue
    tensor_shape = shape(data)
    if not tensor_shape or tensor_shape[-1] != 64:
      continue
    all_u4_weights += 1
    cohort = classify_weight(name)
    if cohort is None:
      if ("VariadicSplit." in name
          or ".mlp.experts.down_proj_compressed" in name):
        excluded_grouped_weights += 1
        continue
      raise ValueError(f"unclassified U4 weight constant: {name}")
    streams: dict[str, dict[str, Any]] = {}
    for stream_name, suffix, element_type in (
        ("weight", "", "u4"),
        ("zero_point", "/zero_point", "u4"),
        ("scale", "/scale", "f16"),
    ):
      stream_layer = layers.get(name + suffix)
      if stream_layer is None or stream_layer.attrib.get("type") != "Const":
        raise ValueError(f"missing constant {name + suffix}")
      stream_data = stream_layer.find("data")
      if (stream_data is None
          or stream_data.attrib.get("element_type") != element_type):
        raise ValueError(f"wrong stream type for {name + suffix}")
      streams[stream_name] = {
          "name": name + suffix,
          "element_type": element_type,
          "shape": list(shape(stream_data)),
          "offset": int(stream_data.attrib["offset"]),
          "bytes": int(stream_data.attrib["size"]),
      }
    codes = math.prod(tensor_shape)
    if streams["weight"]["bytes"] * 2 != codes:
      raise ValueError(f"U4 size mismatch for {name}")
    rows.append({
        "name": name,
        "cohort": cohort,
        "shape": list(tensor_shape),
        "codes": codes,
        "streams": streams,
    })
  rows.sort(key=lambda row: int(row["streams"]["weight"]["offset"]))
  return rows, {
      "all_u4_weight_count": all_u4_weights,
      "excluded_grouped_weight_count": excluded_grouped_weights,
  }


def add_u4_histograms(
    stream: BinaryIO,
    offset: int,
    size: int,
    chunk_bytes: int,
    nibble_hist: np.ndarray[Any, Any],
    byte_hist: np.ndarray[Any, Any],
    word_hist: np.ndarray[Any, Any] | None,
) -> None:
  stream.seek(offset)
  remaining = size
  while remaining:
    value = stream.read(min(remaining, chunk_bytes))
    if not value:
      raise EOFError(f"short model BIN read at {offset}, {remaining} bytes left")
    array = np.frombuffer(value, dtype=np.uint8)
    nibble_hist += np.bincount(array & 15, minlength=16).astype(np.uint64)
    nibble_hist += np.bincount(array >> 4, minlength=16).astype(np.uint64)
    byte_hist += np.bincount(array, minlength=256).astype(np.uint64)
    if word_hist is not None:
      if len(value) % 2:
        raise ValueError("selected U4 stream is not uint16 aligned")
      words = np.frombuffer(value, dtype="<u2")
      word_hist += np.bincount(words, minlength=65536).astype(np.uint64)
    remaining -= len(value)


def add_scale_histogram(
    stream: BinaryIO,
    offset: int,
    size: int,
    chunk_bytes: int,
    word_hist: np.ndarray[Any, Any],
) -> None:
  stream.seek(offset)
  remaining = size
  while remaining:
    value = stream.read(min(remaining, chunk_bytes))
    if not value:
      raise EOFError(f"short model BIN read at {offset}, {remaining} bytes left")
    if len(value) % 2:
      raise ValueError("selected F16 stream is not uint16 aligned")
    words = np.frombuffer(value, dtype="<u2")
    word_hist += np.bincount(words, minlength=65536).astype(np.uint64)
    remaining -= len(value)


def entropy_bits_per_symbol(histogram: np.ndarray[Any, Any]) -> float:
  total = int(histogram.sum())
  if total <= 0:
    raise ValueError("empty entropy histogram")
  probabilities = histogram.astype(np.float64) / total
  nonzero = probabilities[probabilities > 0]
  return float(-(nonzero * np.log2(nonzero)).sum())


def ideal_bytes(histogram: np.ndarray[Any, Any]) -> int:
  bits = int(histogram.sum()) * entropy_bits_per_symbol(histogram)
  return math.ceil(bits / 8)


def huffman_bits(histogram: np.ndarray[Any, Any]) -> int:
  heap = [int(value) for value in histogram if value]
  if len(heap) <= 1:
    return 0
  heapq.heapify(heap)
  total = 0
  while len(heap) > 1:
    merged = heapq.heappop(heap) + heapq.heappop(heap)
    total += merged
    heapq.heappush(heap, merged)
  return total


def codec_row(
    name: str,
    payload_bytes: int,
    raw_bytes: int,
    required_saving_bytes: float,
    carrier_gbps: float,
    symbol_count: int,
    issue_ceiling_symbols_s: float,
    note: str,
) -> dict[str, Any]:
  saving_bytes = raw_bytes - payload_bytes
  saving_ms = saving_bytes / (carrier_gbps * 1_000_000.0)
  margin_bytes = saving_bytes - required_saving_bytes
  margin_ms = margin_bytes / (carrier_gbps * 1_000_000.0)
  required_decode_symbols_s = (
      symbol_count / (margin_ms / 1000.0) if margin_ms > 0 else None)
  return {
      "name": name,
      "payload_bytes": payload_bytes,
      "raw_bytes": raw_bytes,
      "compression_ratio": payload_bytes / raw_bytes,
      "saving_bytes": saving_bytes,
      "saving_ms_at_registered_carrier": saving_ms,
      "payload_margin_bytes_after_required_cut": margin_bytes,
      "payload_margin_ms_after_required_cut": margin_ms,
      "decode_symbol_count": symbol_count,
      "required_decode_symbols_per_s_inside_payload_margin": (
          required_decode_symbols_s),
      "fraction_of_one_lane_op_issue_ceiling": (
          required_decode_symbols_s / issue_ceiling_symbols_s
          if required_decode_symbols_s is not None else None),
      "note": note,
  }


def selected_ranges(rows: list[dict[str, Any]]) -> list[tuple[int, int]]:
  values: list[tuple[int, int]] = []
  for row in rows:
    for stream_name in ("weight", "zero_point", "scale"):
      item = row["streams"][stream_name]
      values.append((int(item["offset"]), int(item["bytes"])))
  values.sort()
  for (left_offset, left_size), (right_offset, _) in zip(values, values[1:]):
    if left_offset + left_size > right_offset:
      raise ValueError("selected model BIN ranges overlap")
  return values


def stream_ranges(
    destination: BinaryIO,
    source: BinaryIO,
    ranges: list[tuple[int, int]],
    chunk_bytes: int,
) -> None:
  for offset, size in ranges:
    source.seek(offset)
    remaining = size
    while remaining:
      value = source.read(min(remaining, chunk_bytes))
      if not value:
        raise EOFError(f"short zstd source read at {offset}")
      destination.write(value)
      remaining -= len(value)


def zstd_size(
    executable: Path,
    level: int,
    ranges: list[tuple[int, int]],
    chunk_bytes: int,
) -> int:
  compressor = subprocess.Popen(
      [str(executable), f"-{level}", "-T1", "-q", "-c"],
      stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  if compressor.stdin is None or compressor.stdout is None:
    raise RuntimeError("failed to create zstd pipes")
  counter = subprocess.Popen(
      ["wc", "-c"], stdin=compressor.stdout, stdout=subprocess.PIPE,
      stderr=subprocess.PIPE, text=True)
  compressor.stdout.close()
  try:
    with MODEL_BIN.open("rb", buffering=0) as source:
      stream_ranges(compressor.stdin, source, ranges, chunk_bytes)
    compressor.stdin.close()
    count_stdout, count_stderr = counter.communicate()
    compressor_stderr = compressor.stderr.read().decode(
        "utf-8", errors="replace") if compressor.stderr is not None else ""
    compressor_returncode = compressor.wait()
  except BaseException:
    compressor.kill()
    counter.kill()
    raise
  if compressor_returncode != 0 or counter.returncode != 0:
    raise RuntimeError(
        f"zstd level {level} failed: {compressor_returncode}, "
        f"{counter.returncode}, {compressor_stderr}, {count_stderr}")
  return int(count_stdout.strip())


def disassemble_program(ocloc: Path, program: Path) -> dict[str, Any]:
  with tempfile.TemporaryDirectory(prefix="iq36-fc-hardware-limit-") as temp:
    result = subprocess.run(
        [str(ocloc), "disasm", "-file", str(program), "-dump", temp],
        text=True, capture_output=True, check=False)
    assembly_files = sorted(Path(temp).glob("*.asm"))
    if result.returncode != 0 or not assembly_files:
      raise RuntimeError(
          f"ocloc disasm failed: {result.returncode}: "
          f"{result.stdout}\n{result.stderr}")
    assembly = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in assembly_files)
  block_kinds = Counter(re.findall(r"\bload_block2d\.ugm\.([a-z0-9]+)", assembly))
  dpas_count = len(re.findall(r"\bdpas\.\d+x\d+", assembly))
  dpasw_count = len(re.findall(r"\bdpasw\.\d+x\d+", assembly))
  return {
      "assembly_file_count": len(assembly_files),
      "assembly_bytes": len(assembly.encode("utf-8")),
      "load_block2d_count": sum(block_kinds.values()),
      "load_block2d_kinds": dict(sorted(block_kinds.items())),
      "dpas_count": dpas_count,
      "dpasw_count": dpasw_count,
      "barrier_count_in_assembly": len(re.findall(r"^\s*barrier\b", assembly,
                                                   flags=re.MULTILINE)),
      "disassembler_stdout": result.stdout.strip(),
      "disassembler_stderr": result.stderr.strip(),
  }


def audit_dpasw_source(path: Path, target_contract: dict[str, Any]) -> dict[str, Any]:
  source = path.read_text(encoding="utf-8")
  fused_match = re.search(
      r"fused\s*=\s*one_of\(hw,\s*\{(?P<architectures>[^}]+)\}\);",
      source)
  if fused_match is None:
    raise ValueError("pinned gemmstone fused-EU architecture list is missing")
  fused_architectures = re.findall(
      r"HW::([A-Za-z0-9_]+)", fused_match.group("architectures"))
  target_label = str(target_contract["target"]["machine_label"])
  target_core = "Xe3" if "PTL" in target_label else "unknown"
  requires_fused = "dpasw &= systolic && fused;" in source
  return {
      "target_machine_label": target_label,
      "target_core": target_core,
      "fused_eu_architectures": fused_architectures,
      "target_has_gemmstone_fused_eu_flag": target_core in fused_architectures,
      "dpasw_preflight_requires_systolic_and_fused": requires_fused,
      "dpasw_eligible_on_target": (
          target_core in fused_architectures and requires_fused),
      "source_path": display_path(path),
      "source_sha256": sha256(path),
  }


def main() -> int:
  args = parse_args()
  output = args.output.resolve()
  output.mkdir(parents=True, exist_ok=False)
  stop_bytes = int(args.memory_stop_gib * 1024**3)
  chunk_bytes = args.chunk_mib * 1024**2
  memory: list[dict[str, Any]] = []
  sample_memory("start", stop_bytes, memory)

  required = (
      MODEL_XML, MODEL_BIN, MODEL_CONTRACT, TARGET_CONTRACT, FIXED_COMPONENT,
      REPRESENTATIVE_PROGRAM, PINNED_GEMM_STRATEGY, args.zstd, args.ocloc)
  missing = [display_path(path) for path in required if not path.is_file()]
  if missing:
    raise SystemExit("missing FC hardware-limit inputs: " + ", ".join(missing))

  git = git_state(output)
  if git["dirty"]:
    raise SystemExit("FC hardware-limit bound requires a clean repository: "
                     + ", ".join(git["dirty_paths"]))
  model_contract = load_json(MODEL_CONTRACT)
  target_contract = load_json(TARGET_CONTRACT)
  fixed = load_json(FIXED_COMPONENT)
  rows, graph_census = parse_selected_tensors(MODEL_XML)
  ranges = selected_ranges(rows)
  sample_memory("after-graph-census", stop_bytes, memory)

  weight_nibble = np.zeros(16, dtype=np.uint64)
  weight_byte = np.zeros(256, dtype=np.uint64)
  weight_word = np.zeros(65536, dtype=np.uint64)
  zero_nibble = np.zeros(16, dtype=np.uint64)
  zero_byte = np.zeros(256, dtype=np.uint64)
  scale_word = np.zeros(65536, dtype=np.uint64)
  with MODEL_BIN.open("rb", buffering=0) as model_stream:
    for row in rows:
      weight = row["streams"]["weight"]
      add_u4_histograms(
          model_stream, int(weight["offset"]), int(weight["bytes"]),
          chunk_bytes, weight_nibble, weight_byte, weight_word)
      zero = row["streams"]["zero_point"]
      add_u4_histograms(
          model_stream, int(zero["offset"]), int(zero["bytes"]),
          chunk_bytes, zero_nibble, zero_byte, None)
      scale = row["streams"]["scale"]
      add_scale_histogram(
          model_stream, int(scale["offset"]), int(scale["bytes"]),
          chunk_bytes, scale_word)
  sample_memory("after-exact-histograms", stop_bytes, memory)

  cohort_rows: dict[str, dict[str, Any]] = {}
  for cohort, expected in EXPECTED_COHORTS.items():
    selected = [row for row in rows if row["cohort"] == cohort]
    codes = sum(int(row["codes"]) for row in selected)
    stream_bytes = {
        stream_name: sum(int(row["streams"][stream_name]["bytes"])
                         for row in selected)
        for stream_name in ("weight", "zero_point", "scale")}
    cohort_rows[cohort] = {
        **expected,
        "observed_tensor_count": len(selected),
        "codes": codes,
        "expected_codes": int(expected["m"] * expected["k"]
                              * expected["calls"]),
        "stream_bytes": stream_bytes,
        "parameter_bytes": sum(stream_bytes.values()),
    }

  raw_bytes = sum(size for _, size in ranges)
  weight_bytes = int(weight_byte.sum())
  zero_bytes = int(zero_byte.sum())
  scale_bytes = int(scale_word.sum()) * 2
  metadata_bytes = zero_bytes + scale_bytes
  aggregate = fixed["aggregate"]
  current_ms = float(aggregate["dominant_ms"])
  target_ms = float(aggregate["target_ms"])
  required_cut_ms = current_ms - target_ms
  carrier_values = {
      float(row["runtime"]["minimum_gbps"])
      for row in fixed["cohorts"] if row.get("runtime")}
  if len(carrier_values) != 1:
    raise ValueError(f"inconsistent registered FC carriers: {carrier_values}")
  carrier_gbps = carrier_values.pop()
  required_saving_bytes = required_cut_ms * carrier_gbps * 1_000_000.0

  runtime = target_contract["runtime"]
  compute_units = int(runtime["opencl_compute_units"])
  max_clock_hz = float(runtime["opencl_max_clock_mhz"]) * 1_000_000.0
  simd_lanes = 16
  issue_ceiling_symbols_s = compute_units * simd_lanes * max_clock_hz

  ideal_payload = (
      ideal_bytes(weight_nibble) + ideal_bytes(zero_nibble)
      + ideal_bytes(scale_word))
  huffman_nibble_payload = math.ceil((
      huffman_bits(weight_nibble) + huffman_bits(zero_nibble)
      + huffman_bits(scale_word)) / 8)
  huffman_byte_payload = math.ceil((
      huffman_bits(weight_byte) + huffman_bits(zero_byte)
      + huffman_bits(scale_word)) / 8)
  huffman_word_payload = math.ceil((
      huffman_bits(weight_word) + huffman_bits(zero_byte)
      + huffman_bits(scale_word)) / 8)
  codec_candidates = [
      codec_row(
          "zero_order_entropy_nibble_u16_payload_only", ideal_payload,
          raw_bytes, required_saving_bytes, carrier_gbps,
          int(weight_nibble.sum() + zero_nibble.sum() + scale_word.sum()),
          issue_ceiling_symbols_s,
          "Non-integer information bound; no codebook, framing, random access, "
          "or decode work is charged."),
      codec_row(
          "canonical_huffman_nibble_u16_payload_only", huffman_nibble_payload,
          raw_bytes, required_saving_bytes, carrier_gbps,
          int(weight_nibble.sum() + zero_nibble.sum() + scale_word.sum()),
          issue_ceiling_symbols_s,
          "Exact global Huffman merge cost; no codebook, framing, random "
          "access, or decode work is charged."),
      codec_row(
          "canonical_huffman_byte_u16_payload_only", huffman_byte_payload,
          raw_bytes, required_saving_bytes, carrier_gbps,
          int(weight_byte.sum() + zero_byte.sum() + scale_word.sum()),
          issue_ceiling_symbols_s,
          "Weight/zero-point bytes plus F16-bit-pattern symbols; payload only."),
      codec_row(
          "canonical_huffman_u16_byte_u16_payload_only", huffman_word_payload,
          raw_bytes, required_saving_bytes, carrier_gbps,
          int(weight_word.sum() + zero_byte.sum() + scale_word.sum()),
          issue_ceiling_symbols_s,
          "Most favorable tested global software-Huffman symbolization; "
          "payload only and no random-access block restarts."),
  ]

  zstd_version = subprocess.run(
      [str(args.zstd), "--version"], text=True, capture_output=True,
      check=True).stdout.strip()
  zstd_rows: list[dict[str, Any]] = []
  for level in args.zstd_levels:
    compressed_bytes = zstd_size(
        args.zstd, level, ranges, chunk_bytes)
    zstd_rows.append(codec_row(
        f"zstd_level_{level}_single_contiguous_stream", compressed_bytes,
        raw_bytes, required_saving_bytes, carrier_gbps, 0,
        issue_ceiling_symbols_s,
        "Real single-threaded offline payload. It has no FC random access and "
        "charges zero GPU decode time."))
    sample_memory(f"after-zstd-level-{level}", stop_bytes, memory)

  isa = disassemble_program(args.ocloc, REPRESENTATIVE_PROGRAM)
  dpasw_source = audit_dpasw_source(PINNED_GEMM_STRATEGY, target_contract)
  sample_memory("after-offline-disassembly", stop_bytes, memory)
  representative = next(
      row for row in fixed["cohorts"]
      if int(row["m"]) == 2048 and int(row["k"]) == 4096
      and int(row["count"]) == 40)
  isa["package"] = representative["package"]

  input_activation_bytes = sum(
      int(row["k"] * 2 * row["calls"])
      for row in EXPECTED_COHORTS.values())
  metadata_delete_saving_ms = metadata_bytes / (carrier_gbps * 1_000_000.0)
  input_delete_saving_ms = (
      input_activation_bytes / (carrier_gbps * 1_000_000.0))
  best_zstd = min(zstd_rows, key=lambda row: int(row["payload_bytes"]))
  best_huffman = next(
      row for row in codec_candidates
      if row["name"] == "canonical_huffman_u16_byte_u16_payload_only")

  contract_files = model_contract["product_model"]["locked_files"]
  expected_xml = contract_files[MODEL_XML.name]
  expected_bin = contract_files[MODEL_BIN.name]
  checks = [
      check("repository_clean_at_gate", not git["dirty"], **git),
      check("locked_xml_identity_matches_contract",
            MODEL_XML.stat().st_size == int(expected_xml["bytes"])
            and sha256(MODEL_XML) == expected_xml["sha256"]),
      check("locked_bin_size_matches_contract_without_full_rehash",
            MODEL_BIN.stat().st_size == int(expected_bin["bytes"]),
            contract_sha256=expected_bin["sha256"]),
      check("exact_390_non_lm_fc_weight_tensors_selected",
            len(rows) == 390 and graph_census["all_u4_weight_count"] == 510
            and graph_census["excluded_grouped_weight_count"] == 120,
            selected=len(rows), **graph_census),
      check("all_five_cohort_shapes_and_counts_match",
            all(row["observed_tensor_count"] == row["tensor_count"]
                and row["codes"] == row["expected_codes"]
                for row in cohort_rows.values())),
      check("exact_parameter_bytes_match_seq1233",
            raw_bytes == int(aggregate["non_lm_fc_bytes"])
            == weight_bytes + metadata_bytes,
            observed_bytes=raw_bytes,
            seq1233_bytes=aggregate["non_lm_fc_bytes"]),
      check("u4_scale_zero_layout_is_exact_group64",
            weight_bytes == 715_038_720 and zero_bytes == 11_172_480
            and scale_bytes == 44_689_920),
      check("current_best_already_uses_block2d_and_dpas",
            isa["load_block2d_count"] > 0 and isa["dpas_count"] > 0,
            load_block2d_count=isa["load_block2d_count"],
            dpas_count=isa["dpas_count"]),
      check("current_best_has_no_dpasw",
            isa["dpasw_count"] == 0, dpasw_count=isa["dpasw_count"]),
      check("pinned_generator_disables_dpasw_on_ptl_xe3",
            dpasw_source["target_core"] == "Xe3"
            and dpasw_source["dpasw_preflight_requires_systolic_and_fused"]
            and not dpasw_source["target_has_gemmstone_fused_eu_flag"]
            and not dpasw_source["dpasw_eligible_on_target"],
            target_core=dpasw_source["target_core"],
            fused_eu_architectures=dpasw_source["fused_eu_architectures"]),
      check("metadata_relayout_cannot_close_residual_even_if_metadata_is_free",
            metadata_delete_saving_ms < required_cut_ms,
            impossible_saving_ms=metadata_delete_saving_ms,
            required_cut_ms=required_cut_ms),
      check("dpasw_input_reuse_cannot_close_residual_even_if_input_is_free",
            input_delete_saving_ms < required_cut_ms,
            impossible_saving_ms=input_delete_saving_ms,
            required_cut_ms=required_cut_ms),
      check("ideal_full_parameter_entropy_has_only_sub_tenth_ms_margin",
            0 < codec_candidates[0]["payload_margin_ms_after_required_cut"]
            < 0.1,
            margin_ms=codec_candidates[0]["payload_margin_ms_after_required_cut"]),
      check("nibble_huffman_decode_rate_exceeds_one_op_issue_ceiling",
            codec_candidates[1]["fraction_of_one_lane_op_issue_ceiling"] > 1.0,
            fraction=codec_candidates[1]["fraction_of_one_lane_op_issue_ceiling"]),
      check("byte_huffman_decode_rate_exceeds_one_op_issue_ceiling",
            codec_candidates[2]["fraction_of_one_lane_op_issue_ceiling"] > 1.0,
            fraction=codec_candidates[2]["fraction_of_one_lane_op_issue_ceiling"]),
      check("best_u16_huffman_consumes_most_one_op_issue_ceiling_before_dpas",
            best_huffman["fraction_of_one_lane_op_issue_ceiling"] > 0.8,
            fraction=best_huffman["fraction_of_one_lane_op_issue_ceiling"]),
      check("best_real_zstd_has_no_complete_decode_margin",
            best_zstd["payload_margin_ms_after_required_cut"] < 0.01,
            codec_name=best_zstd["name"],
            margin_ms=best_zstd["payload_margin_ms_after_required_cut"]),
      check("source_only_no_compiler_gpu_or_model_worker", True,
            compiler_started=False, gpu_context_created=False,
            model_worker_started=False),
      check("memory_stop_never_crossed_and_process_never_swapped",
            min(row["available_bytes"] for row in memory) >= stop_bytes
            and all(row["self_swap_bytes"] == 0 for row in memory),
            minimum_available_bytes=min(row["available_bytes"] for row in memory)),
  ]
  if not all(row["pass"] for row in checks):
    failed = [row["name"] for row in checks if not row["pass"]]
    raise RuntimeError("FC hardware-limit checks failed: " + ", ".join(failed))

  usage_self = resource.getrusage(resource.RUSAGE_SELF)
  usage_children = resource.getrusage(resource.RUSAGE_CHILDREN)
  metrics = {
      "schema_version": SCHEMA,
      "captured_at": datetime.now(timezone.utc).isoformat(),
      "workstream": WS,
      "mode": "source_only_streaming_hardware_limit_bound",
      "git": git,
      "inputs": {
          display_path(path): {"bytes": path.stat().st_size, "sha256": sha256(path)}
          for path in (Path(__file__), MODEL_CONTRACT, TARGET_CONTRACT,
                       FIXED_COMPONENT, REPRESENTATIVE_PROGRAM,
                       PINNED_GEMM_STRATEGY)
      },
      "locked_model": {
          "xml": str(MODEL_XML),
          "bin": str(MODEL_BIN),
          "xml_sha256": expected_xml["sha256"],
          "bin_contract_sha256": expected_bin["sha256"],
          "bin_full_rehashed": False,
      },
      "graph_census": graph_census,
      "cohorts": cohort_rows,
      "parameter_streams": {
          "weight_u4_bytes": weight_bytes,
          "zero_point_u4_bytes": zero_bytes,
          "scale_f16_bytes": scale_bytes,
          "metadata_bytes": metadata_bytes,
          "total_bytes": raw_bytes,
          "weight_u4_nibble_histogram": weight_nibble.tolist(),
          "zero_point_u4_nibble_histogram": zero_nibble.tolist(),
          "weight_entropy_bits_per_code": entropy_bits_per_symbol(weight_nibble),
          "zero_point_entropy_bits_per_code": entropy_bits_per_symbol(zero_nibble),
          "scale_entropy_bits_per_f16_pattern": entropy_bits_per_symbol(scale_word),
          "scale_unique_f16_patterns": int(np.count_nonzero(scale_word)),
      },
      "kill_number": {
          "current_fixed_component_ms": current_ms,
          "target_ms": target_ms,
          "required_cut_ms": required_cut_ms,
          "registered_carrier_gbps": carrier_gbps,
          "required_saving_bytes_at_registered_carrier": required_saving_bytes,
      },
      "layout_and_dpasw_bounds": {
          "metadata_delete_bytes": metadata_bytes,
          "metadata_delete_saving_ms": metadata_delete_saving_ms,
          "metadata_shortfall_ms": required_cut_ms - metadata_delete_saving_ms,
          "input_activation_bytes": input_activation_bytes,
          "input_delete_saving_ms": input_delete_saving_ms,
          "input_reuse_shortfall_ms": required_cut_ms - input_delete_saving_ms,
      },
      "isa_audit": isa,
      "dpasw_source_audit": dpasw_source,
      "issue_ceiling": {
          "opencl_compute_units": compute_units,
          "simd_lanes": simd_lanes,
          "max_clock_hz": max_clock_hz,
          "one_lane_symbol_op_per_cycle_ceiling_symbols_s": (
              issue_ceiling_symbols_s),
          "interpretation": (
              "Optimistic issue ruler only: it charges one lane operation per "
              "decoded symbol and leaves no instructions for loads, prefix "
              "lookup, bit advance, unpack, address generation, or DPAS."),
      },
      "entropy_payload_bounds": codec_candidates,
      "real_contiguous_zstd_bounds": zstd_rows,
      "zstd_version": zstd_version,
      "memory": {
          "stop_bytes": stop_bytes,
          "samples": memory,
          "minimum_available_bytes": min(row["available_bytes"] for row in memory),
          "self_peak_rss_kib": int(usage_self.ru_maxrss),
          "max_child_peak_rss_kib": int(usage_children.ru_maxrss),
          "self_swap_bytes": process_swap_bytes(),
      },
      "checks": checks,
      "required_checks_passed": True,
      "decision": (
          "reject_current_software_entropy_metadata_layout_and_dpasw_variants_"
          "park_fixed_function_random_access_weight_decode_capability"),
      "reopen_condition": (
          "A block-random-access lossless parameter codec plus fixed-function "
          "or source-derived decode schedule must preserve exact U4/F16 "
          "semantics and prove total five-cohort time below 8.183 ms including "
          "codebooks, offsets, alignment, decode, loads, DQ, DPAS, and provider "
          "work. Do not implement software Huffman, metadata relayout, DPASW, "
          "or neighboring gemmstone variants from payload-only evidence."),
  }
  (output / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")

  report = f"""# OpenVINO FC hardware-limit bound

- decision: `{metrics['decision']}`
- exact selected weights: `{len(rows)}` tensors / `{raw_bytes:,}` parameter bytes
- current / target / required cut: `{current_ms:.6f} / {target_ms:.6f} / {required_cut_ms:.6f} ms`
- registered carrier: `{carrier_gbps:.3f} GB/s`; required byte cut: `{required_saving_bytes:,.0f} B`
- representative ISA: `{isa['load_block2d_count']}` block-2D loads, `{isa['dpas_count']}` DPAS, `{isa['dpasw_count']}` DPASW
- pinned DPASW preflight: target `{dpasw_source['target_core']}`, fused-EU list `{dpasw_source['fused_eu_architectures']}`, eligible `{dpasw_source['dpasw_eligible_on_target']}`
- metadata-free impossible ceiling: `{metadata_delete_saving_ms:.6f} ms` saving, short by `{required_cut_ms - metadata_delete_saving_ms:.6f} ms`
- input-free/DPASW reuse ceiling: `{input_delete_saving_ms:.6f} ms` saving, short by `{required_cut_ms - input_delete_saving_ms:.6f} ms`

## Exact entropy

- U4 weight entropy: `{entropy_bits_per_symbol(weight_nibble):.9f}` bits/code
- U4 zero-point entropy: `{entropy_bits_per_symbol(zero_nibble):.9f}` bits/code
- F16 scale-pattern entropy: `{entropy_bits_per_symbol(scale_word):.9f}` bits/value across `{int(np.count_nonzero(scale_word))}` patterns
- ideal zero-order payload margin after the required cut: `{codec_candidates[0]['payload_margin_ms_after_required_cut']:.6f} ms`
- best tested Huffman payload margin: `{best_huffman['payload_margin_ms_after_required_cut']:.6f} ms`; its decode ruler needs `{best_huffman['required_decode_symbols_per_s_inside_payload_margin'] / 1e12:.3f} Tsymbol/s`, `{best_huffman['fraction_of_one_lane_op_issue_ceiling']:.3f}x` the optimistic one-lane-op issue ceiling before any real decode or DPAS instructions

## Real contiguous compression

| stream | payload bytes | saving ms | zero-decode margin ms |
|---|---:|---:|---:|
"""
  for row in zstd_rows:
    report += (
        f"| {row['name']} | {row['payload_bytes']:,} | "
        f"{row['saving_ms_at_registered_carrier']:.6f} | "
        f"{row['payload_margin_ms_after_required_cut']:.6f} |\n")
  report += f"""

The real zstd rows are one contiguous offline stream: they have no per-FC
random access and charge zero GPU decode work.  The best row leaves only
`{best_zstd['payload_margin_ms_after_required_cut']:.6f} ms`; it therefore does
not admit a software decoder.

No compiler, OpenCL/Level Zero context, GPU kernel, or model worker ran.  The
streaming gate used at most `{int(usage_self.ru_maxrss):,} KiB` self RSS and
`{int(usage_children.ru_maxrss):,} KiB` child RSS, with zero process swap and
minimum available memory `{min(row['available_bytes'] for row in memory):,} B`.
"""
  (output / "report.md").write_text(report, encoding="utf-8")
  print(json.dumps({
      "output": display_path(output),
      "decision": metrics["decision"],
      "required_checks_passed": True,
      "best_zstd_margin_ms": best_zstd["payload_margin_ms_after_required_cut"],
      "minimum_available_bytes": metrics["memory"]["minimum_available_bytes"],
  }, sort_keys=True))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
