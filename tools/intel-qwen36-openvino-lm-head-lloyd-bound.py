#!/usr/bin/env python3
"""Bound Lloyd-centroid LM-head codecs on captured product hidden rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path("/home/intel/Qwen3.6-35B-A3B-ov")
MODEL_XML = MODEL_DIR / "openvino_language_model.xml"
MODEL_BIN = MODEL_DIR / "openvino_language_model.bin"
LM_HEAD_NAME = "__module.model.lm_head/ov_ext::linear/MatMul"
WEIGHT_NAME = "self.model.lm_head.weight"
SCALE_NAME = "self.model.lm_head.weight/scale"
GROUP_SIZE = 256
GATED_EXACT_COUNT = 25
GATED_EXACT_DELTA = 11.0


def parse_csv_ints(value: str) -> tuple[int, ...]:
  parsed = tuple(int(item) for item in value.split(",") if item)
  if not parsed or len(parsed) != len(set(parsed)):
    raise argparse.ArgumentTypeError("expected unique comma-separated integers")
  return parsed


def parser() -> argparse.ArgumentParser:
  result = argparse.ArgumentParser(description=__doc__)
  result.add_argument("--worker", type=Path, required=True)
  result.add_argument(
      "--metric-worker", type=Path,
      help=(
          "optional worker whose captured logits are the distribution metric "
          "reference; hidden state and exact correction values still come "
          "from --worker"))
  result.add_argument(
      "--steps", type=parse_csv_ints,
      help="captured decode steps; omitted means every matched hidden/logit row")
  result.add_argument("--bits", type=parse_csv_ints, default=(1, 2, 3))
  result.add_argument(
      "--local-topks", type=parse_csv_ints, default=(4, 8, 12))
  result.add_argument("--direct-topk", type=int, default=8)
  result.add_argument("--hybrid-base-bits", type=int)
  result.add_argument("--hybrid-exact-capacity", type=int, default=4096)
  result.add_argument("--hybrid-exact-delta", type=float)
  result.add_argument("--hybrid-refine-bits", type=int)
  result.add_argument("--hybrid-selection-group-rows", type=int, default=1)
  result.add_argument("--lloyd-iterations", type=int, default=5)
  result.add_argument("--row-chunk", type=int, default=512)
  result.add_argument("--out", type=Path, required=True)
  return result


def resolve(path: str) -> Path:
  value = Path(path)
  return value if value.is_absolute() else ROOT / value


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def ir_constant(name: str) -> dict[str, Any]:
  root = ET.parse(MODEL_XML).getroot()
  layers = root.find("layers")
  if layers is None:
    raise RuntimeError("model IR has no layers")
  layer = next(
      value for value in layers if value.attrib.get("name") == name)
  data = layer.find("data")
  if data is None:
    raise RuntimeError(f"IR constant has no data: {name}")
  return {
      "element_type": data.attrib["element_type"],
      "offset": int(data.attrib["offset"]),
      "shape": [int(item) for item in data.attrib["shape"].split(",")],
      "size": int(data.attrib["size"]),
  }


def distribution_metrics(reference: Any, candidate: Any, np: Any) -> dict[str, Any]:
  reference64 = np.asarray(reference, dtype=np.float64)
  candidate64 = np.asarray(candidate, dtype=np.float64)
  reference_log_z = float(np.max(reference64)) + math.log(float(
      np.exp(reference64 - float(np.max(reference64))).sum()))
  candidate_log_z = float(np.max(candidate64)) + math.log(float(
      np.exp(candidate64 - float(np.max(candidate64))).sum()))
  reference_log_p = reference64 - reference_log_z
  candidate_log_p = candidate64 - candidate_log_z
  reference_p = np.exp(reference_log_p)
  delta = candidate64 - reference64
  reference_top1 = int(np.argmax(reference64))
  candidate_top1 = int(np.argmax(candidate64))
  denominator = float(
      np.linalg.norm(reference64) * np.linalg.norm(candidate64))
  return {
      "candidate_top1": candidate_top1,
      "cosine": (
          float(np.vdot(reference64, candidate64) / denominator)
          if denominator else 1.0),
      "kld_reference_to_candidate": float(np.sum(
          reference_p * (reference_log_p - candidate_log_p))),
      "max_abs": float(np.max(np.abs(delta))),
      "reference_top1": reference_top1,
      "relative_l2": float(
          np.linalg.norm(delta) /
          max(float(np.linalg.norm(reference64)),
              np.finfo(np.float64).tiny)),
      "top1_match": reference_top1 == candidate_top1,
  }


def q8_hidden(values: Any, np: Any) -> Any:
  rows, columns = values.shape
  grouped = values.astype(np.float16).reshape(
      rows, columns // GROUP_SIZE, GROUP_SIZE)
  maximum = np.maximum(
      np.max(np.abs(grouped), axis=2, keepdims=True),
      np.float16(0.00006103515625)).astype(np.float16)
  quantize_scale = (np.float16(127.0) / maximum).astype(np.float16)
  codes = np.clip(
      np.rint((grouped * quantize_scale).astype(np.float16)), -128, 127)
  dequantize_scale = (np.float16(1.0) / quantize_scale).astype(np.float16)
  return (
      codes.astype(np.float16) * dequantize_scale
  ).astype(np.float16).reshape(rows, columns).astype(np.float32)


def lloyd_dequantize(
    raw: Any, levels: int, iterations: int, np: Any,
) -> tuple[Any, Any]:
  values = raw.astype(np.float32)
  minimum = np.min(values, axis=1)
  maximum = np.max(values, axis=1)
  fractions = (
      (np.arange(levels, dtype=np.float32) + np.float32(0.5)) /
      np.float32(levels))
  centers = (
      minimum[:, None] +
      (maximum - minimum)[:, None] * fractions[None, :])
  assignments = None
  for _ in range(iterations):
    assignments = np.argmin(
        np.abs(values[:, :, None] - centers[:, None, :]), axis=2)
    for cluster in range(levels):
      selected = assignments == cluster
      counts = np.sum(selected, axis=1)
      sums = np.sum(values * selected, axis=1, dtype=np.float32)
      centers[:, cluster] = np.where(
          counts != 0, sums / np.maximum(counts, 1),
          centers[:, cluster])
  assignments = np.argmin(
      np.abs(values[:, :, None] - centers[:, None, :]), axis=2)
  dequantized = np.take_along_axis(
      centers[:, None, :], assignments[:, :, None], axis=2)[:, :, 0]
  return dequantized, centers


def correction_rows(
    approximate: Any, exact_q8: Any, direct_reference: Any, local_topk: int,
    direct_topk: int, np: Any,
) -> tuple[Any, dict[str, Any]]:
  vocabulary = approximate.size
  if vocabulary % 256:
    raise ValueError("LM-head rows must be divisible by 256")
  corrected = approximate.astype(np.float16).astype(np.float32)
  selected_ids = np.empty((0,), dtype=np.int64)
  if local_topk:
    blocks = corrected.reshape(vocabulary // 256, 256)
    local = np.argpartition(blocks, -local_topk, axis=1)[:, -local_topk:]
    selected_ids = (
        local + np.arange(blocks.shape[0])[:, None] * 256).reshape(-1)
    exact_rounded = exact_q8.astype(np.float16).astype(np.float32)
    corrected[selected_ids] = exact_rounded[selected_ids]
  direct_ids = np.argpartition(corrected, -direct_topk)[-direct_topk:]
  # The integrated direct correction is already proven to reproduce the
  # selected stock logits exactly.  Use the captured values here so the bound
  # isolates centroid recall/distribution error rather than re-litigating it.
  corrected[direct_ids] = direct_reference[direct_ids]
  # Mirror the integrated full-distribution fallback predicate.  The
  # production kernel only counts the product-like local correction set,
  # after the exact direct-topK overwrite.
  gate_threshold = float(np.max(corrected)) - GATED_EXACT_DELTA
  gated_exact_count = int(np.count_nonzero(
      corrected[selected_ids] >= gate_threshold)) if selected_ids.size else 0
  return corrected, {
      "direct_correction_rows": int(direct_ids.size),
      "gated_exact_count": gated_exact_count,
      "gated_exact_count_threshold": GATED_EXACT_COUNT,
      "gated_exact_delta": GATED_EXACT_DELTA,
      "gated_exact_trigger": gated_exact_count >= GATED_EXACT_COUNT,
      "local_correction_rows": int(selected_ids.size),
  }


def refinement_rows(
    approximate: Any, refinement: Any, direct_reference: Any, local_topk: int,
    direct_topk: int, exact_delta: float | None, exact_capacity: int,
    selection_group_rows: int, np: Any,
) -> tuple[Any, dict[str, Any]]:
  vocabulary = approximate.size
  if vocabulary % 256:
    raise ValueError("LM-head rows must be divisible by 256")
  corrected = approximate.astype(np.float16).astype(np.float32)
  blocks = corrected.reshape(vocabulary // 256, 256)
  if selection_group_rows == 1:
    local = np.argpartition(blocks, -local_topk, axis=1)[:, -local_topk:]
    selected_ids = (
        local + np.arange(blocks.shape[0])[:, None] * 256).reshape(-1)
  else:
    groups_per_block = 256 // selection_group_rows
    selected_group_count = local_topk // selection_group_rows
    grouped = blocks.reshape(
        blocks.shape[0], groups_per_block, selection_group_rows)
    group_scores = np.max(grouped, axis=2)
    local_groups = np.argpartition(
        group_scores, -selected_group_count, axis=1
    )[:, -selected_group_count:]
    local_rows = (
        local_groups[:, :, None] * selection_group_rows +
        np.arange(selection_group_rows)[None, None, :])
    selected_ids = (
        local_rows + np.arange(blocks.shape[0])[:, None, None] * 256
    ).reshape(-1)
  refinement_rounded = refinement.astype(np.float16).astype(np.float32)
  corrected[selected_ids] = refinement_rounded[selected_ids]
  direct_ids = np.argpartition(corrected, -direct_topk)[-direct_topk:]
  corrected[direct_ids] = direct_reference[direct_ids]
  exact_ids = np.empty((0,), dtype=np.int64)
  exact_overflow = False
  if exact_delta is not None:
    threshold = float(np.max(corrected[direct_ids])) - exact_delta
    exact_ids = selected_ids[corrected[selected_ids] >= threshold]
    exact_overflow = exact_ids.size > exact_capacity
    if exact_overflow:
      highest = np.argpartition(
          corrected[exact_ids], -exact_capacity)[-exact_capacity:]
      exact_ids = exact_ids[highest]
    corrected[exact_ids] = direct_reference[exact_ids]
  return corrected, {
      "direct_correction_rows": int(direct_ids.size),
      "exact_refinement_overflow": exact_overflow,
      "exact_refinement_rows": int(exact_ids.size),
      "refinement_rows": int(selected_ids.size),
      "selection_group_rows": selection_group_rows,
  }


def main() -> int:
  args = parser().parse_args()
  if args.out.exists():
    raise SystemExit(f"output already exists: {args.out}")
  if any(bits not in (1, 2, 3, 4) for bits in args.bits):
    raise SystemExit("bits must be selected from 1,2,3,4")
  if any(value <= 0 or value > 248 for value in args.local_topks):
    raise SystemExit("local-topks must be in 1..248")
  if not 0 < args.direct_topk <= 32:
    raise SystemExit("direct-topk must be in 1..32")
  if ((args.hybrid_base_bits is None) !=
      (args.hybrid_refine_bits is None)):
    raise SystemExit(
        "hybrid-base-bits and hybrid-refine-bits must be set together")
  if args.hybrid_base_bits is not None:
    if (args.hybrid_base_bits not in args.bits or
        args.hybrid_refine_bits not in args.bits):
      raise SystemExit("hybrid codecs must also be present in --bits")
    if args.hybrid_base_bits >= args.hybrid_refine_bits:
      raise SystemExit("hybrid refinement must use more bits than its base")
  if args.hybrid_exact_capacity <= 0:
    raise SystemExit("hybrid-exact-capacity must be positive")
  if args.hybrid_exact_delta is not None and args.hybrid_exact_delta <= 0:
    raise SystemExit("hybrid-exact-delta must be positive")
  if (args.hybrid_selection_group_rows <= 0 or
      256 % args.hybrid_selection_group_rows):
    raise SystemExit("hybrid-selection-group-rows must divide 256")
  if any(value % args.hybrid_selection_group_rows for value in args.local_topks):
    raise SystemExit("local-topks must align to hybrid selection groups")
  if args.lloyd_iterations <= 0 or args.row_chunk <= 0:
    raise SystemExit("iterations and row chunk must be positive")

  import numpy as np

  worker_path = args.worker.resolve()
  worker = json.loads(worker_path.read_text(encoding="utf-8"))
  hidden_by_step = {
      int(row["step"]): row
      for row in worker.get("lm_head_hidden_checkpoints", [])}
  logits_by_step = {
      int(row["step"]): row
      for row in worker.get("distribution_checkpoints", [])}
  available_steps = sorted(set(hidden_by_step) & set(logits_by_step))
  steps = list(args.steps) if args.steps is not None else available_steps
  if not steps or not set(steps).issubset(available_steps):
    raise SystemExit(
        f"requested steps are not completely captured: {steps} vs "
        f"{available_steps}")
  hidden = np.stack([
      np.fromfile(resolve(hidden_by_step[step]["file"]), dtype="<f4")
      for step in steps])
  worker_reference = np.stack([
      np.fromfile(resolve(logits_by_step[step]["file"]), dtype="<f4")
      for step in steps])
  metric_worker_path = (
      args.metric_worker.resolve() if args.metric_worker is not None else None)
  metric_worker = (
      json.loads(metric_worker_path.read_text(encoding="utf-8"))
      if metric_worker_path is not None else None)
  metric_logits_by_step = (
      {int(row["step"]): row
       for row in metric_worker.get("distribution_checkpoints", [])}
      if metric_worker is not None else logits_by_step)
  if not set(steps).issubset(metric_logits_by_step):
    raise SystemExit(
        "metric worker does not contain every requested logits step: "
        f"{steps} vs {sorted(metric_logits_by_step)}")
  metric_reference = np.stack([
      np.fromfile(
          resolve(metric_logits_by_step[step]["file"]), dtype="<f4")
      for step in steps])

  weight_fact = ir_constant(WEIGHT_NAME)
  scale_fact = ir_constant(SCALE_NAME)
  rows, columns = weight_fact["shape"]
  if hidden.shape != (len(steps), columns):
    raise SystemExit(f"unexpected hidden shape: {hidden.shape}")
  if worker_reference.shape != (len(steps), rows):
    raise SystemExit(
        f"unexpected worker logits shape: {worker_reference.shape}")
  if metric_reference.shape != (len(steps), rows):
    raise SystemExit(
        f"unexpected metric logits shape: {metric_reference.shape}")
  weights = np.memmap(
      MODEL_BIN, dtype=np.int8, mode="r", offset=weight_fact["offset"],
      shape=(rows, columns))
  scales = np.memmap(
      MODEL_BIN, dtype="<f2", mode="r", offset=scale_fact["offset"],
      shape=(rows, 1))
  quantized_hidden = q8_hidden(hidden, np)
  hidden_t = np.ascontiguousarray(quantized_hidden.T, dtype=np.float32)
  exact_q8 = np.empty((len(steps), rows), dtype=np.float32)
  projected = {
      bits: np.empty((len(steps), rows), dtype=np.float32)
      for bits in args.bits}
  centroid_minimum = {bits: math.inf for bits in args.bits}
  centroid_maximum = {bits: -math.inf for bits in args.bits}
  for begin in range(0, rows, args.row_chunk):
    end = min(rows, begin + args.row_chunk)
    raw = np.asarray(weights[begin:end], dtype=np.float32)
    scale = np.asarray(scales[begin:end], dtype=np.float32).reshape(-1, 1)
    exact_q8[:, begin:end] = (raw @ hidden_t * scale).T
    for bits in args.bits:
      if bits == 4:
        # Mirror the integrated signed-Q4 codec exactly: round I8/16 to
        # [-8, 7], then restore the implicit factor of 16 in the dot product.
        dequantized = np.clip(np.rint(raw / 16.0), -8, 7) * 16.0
        centers = np.asarray((-128.0, 112.0), dtype=np.float32)
      else:
        dequantized, centers = lloyd_dequantize(
            raw, 1 << bits, args.lloyd_iterations, np)
      projected[bits][:, begin:end] = (
          dequantized @ hidden_t * scale).T
      centroid_minimum[bits] = min(
          centroid_minimum[bits], float(np.min(centers)))
      centroid_maximum[bits] = max(
          centroid_maximum[bits], float(np.max(centers)))

  baseline_rows = [
      distribution_metrics(metric_reference[index], exact_q8[index], np)
      for index in range(len(steps))]
  codecs = {}
  for bits in args.bits:
    variants = {}
    for local_topk in args.local_topks:
      metric_rows = []
      correction = None
      for index, step in enumerate(steps):
        corrected, correction = correction_rows(
            projected[bits][index], exact_q8[index], worker_reference[index],
            local_topk, args.direct_topk, np)
        metric_rows.append({
            "step": step,
            **correction,
            **distribution_metrics(metric_reference[index], corrected, np),
        })
      variants[str(local_topk)] = {
          "correction": correction,
          "gated_exact_trigger_count": sum(
              row["gated_exact_trigger"] for row in metric_rows),
          "max_kld": max(
              row["kld_reference_to_candidate"] for row in metric_rows),
          "min_cosine": min(row["cosine"] for row in metric_rows),
          "rows": metric_rows,
          "top1_matches": sum(row["top1_match"] for row in metric_rows),
          "worst_kld_step": max(
              metric_rows,
              key=lambda row: row["kld_reference_to_candidate"])["step"],
      }
    levels = 1 << bits
    code_bytes = math.ceil(rows * columns * bits / 8)
    centroid_bytes = 0 if bits == 4 else rows * levels * 4
    scale_bytes = rows * 2
    rowstripe_padding_bytes = rows * 2 if bits == 4 else 0
    local_correction_bytes = rows // 256 * max(args.local_topks) * columns
    codecs[str(bits)] = {
        "centroid_range": {
            "maximum": centroid_maximum[bits],
            "minimum": centroid_minimum[bits],
        },
        "levels": levels,
        "codec": "signed_uniform_q4_div16" if bits == 4 else "lloyd",
        "packed": {
            "centroid_bytes_f32": centroid_bytes,
            "code_bytes": code_bytes,
            "rowstripe_padding_bytes": rowstripe_padding_bytes,
            "scale_bytes_f16": scale_bytes,
            "total_bytes": (
                code_bytes + centroid_bytes + scale_bytes +
                rowstripe_padding_bytes),
        },
        "product_like_variants": variants,
        "upper_local_correction_weight_bytes": local_correction_bytes,
    }

  hybrid = None
  if args.hybrid_base_bits is not None:
    base_bits = args.hybrid_base_bits
    refine_bits = args.hybrid_refine_bits
    variants = {}
    refine_row_bytes = math.ceil(columns * refine_bits / 8)
    if refine_bits != 4:
      refine_row_bytes += (1 << refine_bits) * 4
    base_packed_bytes = codecs[str(base_bits)]["packed"]["total_bytes"]
    for local_topk in args.local_topks:
      metric_rows = []
      correction = None
      for index, step in enumerate(steps):
        corrected, correction = refinement_rows(
            projected[base_bits][index], projected[refine_bits][index],
            worker_reference[index], local_topk, args.direct_topk,
            args.hybrid_exact_delta, args.hybrid_exact_capacity,
            args.hybrid_selection_group_rows, np)
        metric_rows.append({
            "step": step,
            **correction,
            **distribution_metrics(metric_reference[index], corrected, np),
        })
      refinement_bytes = correction["refinement_rows"] * refine_row_bytes
      variants[str(local_topk)] = {
          "active_codec_bytes": base_packed_bytes + refinement_bytes,
          "direct_exact_weight_bytes": args.direct_topk * columns,
          "maximum_active_bytes_with_exact": (
              base_packed_bytes + refinement_bytes +
              max(row["exact_refinement_rows"] for row in metric_rows) *
              columns),
          "maximum_exact_refinement_rows": max(
              row["exact_refinement_rows"] for row in metric_rows),
          "overflow_count": sum(
              row["exact_refinement_overflow"] for row in metric_rows),
          "max_kld": max(
              row["kld_reference_to_candidate"] for row in metric_rows),
          "min_cosine": min(row["cosine"] for row in metric_rows),
          "refinement_bytes": refinement_bytes,
          "rows": metric_rows,
          "top1_matches": sum(row["top1_match"] for row in metric_rows),
          "worst_kld_step": max(
              metric_rows,
              key=lambda row: row["kld_reference_to_candidate"])["step"],
      }
    hybrid = {
        "base_bits": base_bits,
        "base_packed_bytes": base_packed_bytes,
        "exact_correction_capacity": args.hybrid_exact_capacity,
        "exact_correction_delta": args.hybrid_exact_delta,
        "product_like_variants": variants,
        "refine_bits": refine_bits,
        "refinement_full_storage_bytes": (
            codecs[str(refine_bits)]["packed"]["total_bytes"]),
        "refinement_row_bytes": refine_row_bytes,
        "selection_group_rows": args.hybrid_selection_group_rows,
    }

  result = {
      "activation_codec": "GPU-compatible group256 Q8 from captured F16",
      "baseline_exact_q8": {
          "max_kld": max(
              row["kld_reference_to_candidate"] for row in baseline_rows),
          "rows": [
              {"step": step, **row}
              for step, row in zip(steps, baseline_rows)],
          "top1_matches": sum(row["top1_match"] for row in baseline_rows),
      },
      "codecs": codecs,
      "direct_correction_reference": "worker_native_logits",
      "hybrid": hybrid,
      "hybrid_exact_correction_reference": "worker_native_logits",
      "input": {
          "metric_worker": (
              str(metric_worker_path)
              if metric_worker_path is not None else str(worker_path)),
          "metric_worker_sha256": (
              sha256(metric_worker_path)
              if metric_worker_path is not None else sha256(worker_path)),
          "model_bin": str(MODEL_BIN),
          "model_bin_sha256": sha256(MODEL_BIN),
          "worker": str(worker_path),
          "worker_sha256": sha256(worker_path),
      },
      "lloyd_iterations": args.lloyd_iterations,
      "local_block_rows": 256,
      "gated_exact_count_threshold": GATED_EXACT_COUNT,
      "gated_exact_delta": GATED_EXACT_DELTA,
      "metric_reference": (
          "separate_worker_logits"
          if metric_worker_path is not None else "worker_native_logits"),
      "schema_version": "intel-qwen36-openvino-lm-head-lloyd-bound-v3",
      "steps": steps,
  }
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(
      json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  print(json.dumps({
      "event": "lm_head_lloyd_bound_complete",
      "out": str(args.out),
      "steps": len(steps),
  }, sort_keys=True))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
