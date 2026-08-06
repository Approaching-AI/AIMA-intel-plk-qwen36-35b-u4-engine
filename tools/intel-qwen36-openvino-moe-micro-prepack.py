#!/usr/bin/env python3
"""Build the one admitted group-32 F16/U4 microkernel weight package."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODEL = Path("/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf")
TENSOR_INDEX = ROOT / (
    "output/r1-native-gguf-load-map-20260705T071855Z/tensor-index.jsonl")
ROUTED = ROOT / (
    "output/grouped-s8-u4-prefill-gate-20260711Tseq673cleanZ/raw/prepacked")
MODEL_BYTES = 21_166_755_168
ROUTED_EXPERTS = 256
COMBINED_EXPERTS = 257
HIDDEN = 2048
INTERMEDIATE = 512
Q4_BLOCK_VALUES = 256
Q4_BLOCK_BYTES = 144


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--model", type=Path, default=MODEL)
  parser.add_argument("--tensor-index", type=Path, default=TENSOR_INDEX)
  parser.add_argument("--routed-prepack", type=Path, default=ROUTED)
  parser.add_argument("--out-dir", type=Path, required=True)
  return parser.parse_args()


def sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def load_tensor_rows(path: Path) -> dict[str, dict[str, Any]]:
  wanted = {
      "blk.27.ffn_gate_shexp.weight",
      "blk.27.ffn_up_shexp.weight",
      "blk.27.ffn_down_shexp.weight",
      "blk.27.ffn_gate_inp_shexp.weight",
  }
  rows: dict[str, dict[str, Any]] = {}
  for line in path.read_text(encoding="utf-8").splitlines():
    row = json.loads(line)
    name = str(row.get("name", ""))
    if name in wanted:
      rows[name] = row
  if set(rows) != wanted:
    raise RuntimeError("layer-27 shared-expert tensor index is incomplete")
  return rows


def read_model_tensor(model: Path, row: dict[str, Any]) -> bytes:
  offset = int(row["absolute_offset"])
  size = int(row["nbytes"])
  with model.open("rb") as handle:
    handle.seek(offset)
    value = handle.read(size)
  if len(value) != size:
    raise RuntimeError(f"truncated model tensor: {row['name']}")
  return value


def scale_code(scales: np.ndarray, index: int) -> np.ndarray:
  if index < 4:
    return scales[:, :, index] & np.uint8(63)
  return ((scales[:, :, index + 4] & np.uint8(15)) |
          ((scales[:, :, index - 4] >> np.uint8(6)) << np.uint8(4)))


def min_code(scales: np.ndarray, index: int) -> np.ndarray:
  if index < 4:
    return scales[:, :, index + 4] & np.uint8(63)
  return ((scales[:, :, index + 4] >> np.uint8(4)) |
          ((scales[:, :, index] >> np.uint8(6)) << np.uint8(4)))


def unpack_q4k(raw: bytes, outputs: int, input_width: int
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  blocks_per_row = input_width // Q4_BLOCK_VALUES
  expected = outputs * blocks_per_row * Q4_BLOCK_BYTES
  if len(raw) != expected:
    raise RuntimeError(
        f"Q4_K tensor has {len(raw)} bytes; expected {expected}")
  blocks = np.frombuffer(raw, dtype=np.uint8).reshape(
      outputs, blocks_per_row, Q4_BLOCK_BYTES)
  codes = np.empty((outputs, input_width), dtype=np.uint8)
  quant = blocks[:, :, 16:]
  for block in range(blocks_per_row):
    for segment in range(4):
      packed = quant[:, block, segment * 32:(segment + 1) * 32]
      base = block * Q4_BLOCK_VALUES + segment * 64
      codes[:, base:base + 32] = packed & np.uint8(15)
      codes[:, base + 32:base + 64] = packed >> np.uint8(4)
  packed_codes = (
      codes[:, 0::2] | (codes[:, 1::2] << np.uint8(4))).astype(
          np.uint8, copy=False)

  d = blocks[:, :, 0:2].copy().view("<f2").reshape(
      outputs, blocks_per_row).astype(np.float32)
  dmin = blocks[:, :, 2:4].copy().view("<f2").reshape(
      outputs, blocks_per_row).astype(np.float32)
  packed_scales = blocks[:, :, 4:16]
  scale_codes = np.stack(
      [scale_code(packed_scales, index) for index in range(8)], axis=2)
  min_codes = np.stack(
      [min_code(packed_scales, index) for index in range(8)], axis=2)
  scales = (d[:, :, None] * scale_codes.astype(np.float32)).transpose(
      1, 2, 0).reshape(blocks_per_row * 8, outputs)
  mins = (dmin[:, :, None] * min_codes.astype(np.float32)).transpose(
      1, 2, 0).reshape(blocks_per_row * 8, outputs)
  return packed_codes, scales, mins


def expected_file(path: Path, size: int) -> None:
  if not path.is_file() or path.stat().st_size != size:
    raise RuntimeError(f"prepacked input size mismatch: {path}")


def create_raw(path: Path, dtype: Any, shape: tuple[int, ...]) -> np.memmap:
  return np.memmap(path, mode="w+", dtype=dtype, shape=shape, order="C")


def write_gate_or_up(
    routed: Path, out: Path, kind: str,
    shared: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
  choose = 0 if kind == "gate" else 1
  row_bytes = HIDDEN // 2
  routed_weights = np.memmap(
      routed / "gateup-weights.bin", mode="r", dtype=np.uint8,
      shape=(ROUTED_EXPERTS, 2 * INTERMEDIATE, row_bytes))
  weights = create_raw(
      out / f"{kind}-weights.u4", np.uint8,
      (COMBINED_EXPERTS, INTERMEDIATE, row_bytes))
  weights[:ROUTED_EXPERTS] = routed_weights[:, choose::2, :]
  weights[ROUTED_EXPERTS] = shared[0]
  weights.flush()

  groups = HIDDEN // 32
  routed_scales = np.memmap(
      routed / "gateup-scales.bin", mode="r", dtype=np.float32,
      shape=(ROUTED_EXPERTS, groups, 2 * INTERMEDIATE))
  scales = create_raw(
      out / f"{kind}-scales.f16", np.float16,
      (COMBINED_EXPERTS, groups, INTERMEDIATE))
  scales[:ROUTED_EXPERTS] = routed_scales[:, :, choose::2]
  scales[ROUTED_EXPERTS] = shared[1]
  scales.flush()

  blocks = HIDDEN // Q4_BLOCK_VALUES
  routed_min_codes = np.memmap(
      routed / "gateup-min-codes.bin", mode="r", dtype=np.uint8,
      shape=(ROUTED_EXPERTS, 2 * INTERMEDIATE, blocks, 8))
  routed_dmins = np.memmap(
      routed / "gateup-dmins.bin", mode="r", dtype=np.float32,
      shape=(ROUTED_EXPERTS, 2 * INTERMEDIATE, blocks))
  mins = create_raw(
      out / f"{kind}-mins.f32", np.float32,
      (COMBINED_EXPERTS, groups, INTERMEDIATE))
  for expert in range(ROUTED_EXPERTS):
    codes = routed_min_codes[expert, choose::2].transpose(1, 2, 0)
    dmins = routed_dmins[expert, choose::2].T[:, None, :]
    mins[expert] = (dmins * codes).reshape(groups, INTERMEDIATE)
  mins[ROUTED_EXPERTS] = shared[2]
  mins.flush()

  zps = create_raw(
      out / f"{kind}-zps.u4", np.uint8,
      (COMBINED_EXPERTS * groups * INTERMEDIATE // 2,))
  zps[:] = 0
  zps.flush()


def write_down(
    routed: Path, out: Path,
    shared: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
  row_bytes = INTERMEDIATE // 2
  groups = INTERMEDIATE // 32
  blocks = INTERMEDIATE // Q4_BLOCK_VALUES
  routed_weights = np.memmap(
      routed / "down-weights.bin", mode="r", dtype=np.uint8,
      shape=(ROUTED_EXPERTS, HIDDEN, row_bytes))
  weights = create_raw(
      out / "down-weights.u4", np.uint8,
      (COMBINED_EXPERTS, HIDDEN, row_bytes))
  weights[:ROUTED_EXPERTS] = routed_weights
  weights[ROUTED_EXPERTS] = shared[0]
  weights.flush()

  routed_scales = np.memmap(
      routed / "down-scales.bin", mode="r", dtype=np.float32,
      shape=(ROUTED_EXPERTS, groups, HIDDEN))
  scales = create_raw(
      out / "down-scales.f16", np.float16,
      (COMBINED_EXPERTS, groups, HIDDEN))
  scales[:ROUTED_EXPERTS] = routed_scales
  scales[ROUTED_EXPERTS] = shared[1]
  scales.flush()

  routed_min_codes = np.memmap(
      routed / "down-min-codes.bin", mode="r", dtype=np.uint8,
      shape=(ROUTED_EXPERTS, HIDDEN, blocks, 8))
  routed_dmins = np.memmap(
      routed / "down-dmins.bin", mode="r", dtype=np.float32,
      shape=(ROUTED_EXPERTS, HIDDEN, blocks))
  mins = create_raw(
      out / "down-mins.f32", np.float32,
      (COMBINED_EXPERTS, groups, HIDDEN))
  for expert in range(ROUTED_EXPERTS):
    codes = routed_min_codes[expert].transpose(1, 2, 0)
    dmins = routed_dmins[expert].T[:, None, :]
    mins[expert] = (dmins * codes).reshape(groups, HIDDEN)
  mins[ROUTED_EXPERTS] = shared[2]
  mins.flush()

  zps = create_raw(
      out / "down-zps.u4", np.uint8,
      (COMBINED_EXPERTS * groups * HIDDEN // 2,))
  zps[:] = 0
  zps.flush()


def main() -> int:
  args = parse_args()
  out = args.out_dir.resolve()
  out.mkdir(parents=True, exist_ok=False)
  if args.model.stat().st_size != MODEL_BYTES:
    raise SystemExit("locked GGUF model size mismatch")
  rows = load_tensor_rows(args.tensor_index)
  expected = {
      "blk.27.ffn_gate_shexp.weight": ([HIDDEN, INTERMEDIATE], "Q4_K", 589824),
      "blk.27.ffn_up_shexp.weight": ([HIDDEN, INTERMEDIATE], "Q4_K", 589824),
      "blk.27.ffn_down_shexp.weight": ([INTERMEDIATE, HIDDEN], "Q4_K", 589824),
      "blk.27.ffn_gate_inp_shexp.weight": ([HIDDEN], "F32", 8192),
  }
  for name, (dims, dtype, size) in expected.items():
    row = rows[name]
    if (row.get("dims") != dims or row.get("ggml_type_name") != dtype or
        int(row.get("nbytes", -1)) != size):
      raise SystemExit(f"locked shared tensor contract mismatch: {name}")

  routed_sizes = {
      "gateup-weights.bin": 268_435_456,
      "gateup-scales.bin": 67_108_864,
      "gateup-min-codes.bin": 16_777_216,
      "gateup-dmins.bin": 8_388_608,
      "down-weights.bin": 134_217_728,
      "down-scales.bin": 33_554_432,
      "down-min-codes.bin": 8_388_608,
      "down-dmins.bin": 4_194_304,
  }
  for name, size in routed_sizes.items():
    expected_file(args.routed_prepack / name, size)

  gate_shared = unpack_q4k(
      read_model_tensor(args.model, rows[
          "blk.27.ffn_gate_shexp.weight"]), INTERMEDIATE, HIDDEN)
  up_shared = unpack_q4k(
      read_model_tensor(args.model, rows[
          "blk.27.ffn_up_shexp.weight"]), INTERMEDIATE, HIDDEN)
  down_shared = unpack_q4k(
      read_model_tensor(args.model, rows[
          "blk.27.ffn_down_shexp.weight"]), HIDDEN, INTERMEDIATE)
  write_gate_or_up(args.routed_prepack, out, "gate", gate_shared)
  write_gate_or_up(args.routed_prepack, out, "up", up_shared)
  write_down(args.routed_prepack, out, down_shared)
  scalar = np.frombuffer(read_model_tensor(
      args.model, rows["blk.27.ffn_gate_inp_shexp.weight"]),
      dtype="<f4").copy()
  if scalar.shape != (HIDDEN,) or not np.isfinite(scalar).all():
    raise SystemExit("shared scalar-gate tensor is invalid")
  scalar.tofile(out / "shared-scalar-gate.f32")

  files = sorted(path for path in out.iterdir() if path.is_file())
  manifest = {
      "schema_version": "intel-qwen36-openvino-moe-micro-prepack-v0",
      "experts": COMBINED_EXPERTS,
      "routed_experts": ROUTED_EXPERTS,
      "shared_expert_id": ROUTED_EXPERTS,
      "quant_group_size": 32,
      "activation_type": "f16",
      "weight_type": "u4",
      "scale_type": "f16",
      "zero_point_type": "u4_zero",
      "files": {
          path.name: {"bytes": path.stat().st_size,
                      "sha256": sha256_file(path)}
          for path in files
      },
      "inputs": {
          "model": str(args.model.resolve()),
          "model_bytes": args.model.stat().st_size,
          "tensor_index": str(args.tensor_index.resolve()),
          "routed_prepack": str(args.routed_prepack.resolve()),
          "routed_manifest_sha256": sha256_file(
              args.routed_prepack / "manifest.json"),
      },
  }
  (out / "manifest.json").write_text(
      json.dumps(manifest, indent=2, sort_keys=True) + "\n",
      encoding="utf-8")
  print(json.dumps(manifest, sort_keys=True))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
