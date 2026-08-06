#!/usr/bin/env python3
"""Bound exact-token LM-head certificates on captured product hidden rows.

This is an offline, CPU-only admission tool.  It first closes the selected
two-centroid Q1/global-L2 residual certificate on one registered hard row, then
tests the same proof topology with the already implemented signed-Q4 carrier
over every captured 128k hidden row.  If fixed signed-Q4 misses its registered
traffic cap, the tool evaluates one informed same-width successor with 16
per-row Lloyd centroids.  It does not compile source, create a GPU context,
compile a model, or launch a product worker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
MODEL_DIR = Path("/home/intel/Qwen3.6-35B-A3B-ov")
MODEL_XML = MODEL_DIR / "openvino_language_model.xml"
MODEL_BIN = MODEL_DIR / "openvino_language_model.bin"
CAPTURE_ROOT = (
    ROOT
    / "output/openvino-lm-head-gated-exact-live-hidden-"
      "20260723Tseq2116-all10-128k-o130-correctness/raw/sentinel_128k/"
      "correctness"
)
CANDIDATE_WORKER = CAPTURE_ROOT / "candidate/worker-result.json"
STOCK_WORKER = CAPTURE_ROOT / "stock/worker-result.json"
COMPONENT_RESULT = (
    ROOT
    / "output/openvino-lm-head-gated-exact-component-"
      "20260731Tseq2187-clean/result.json"
)
OPPORTUNITY_RESULT = (
    ROOT
    / "output/openvino-post-pr35924-opportunity-bound-"
      "20260801Tseq2278-clean/metrics.json"
)

EXPECTED_SHA256 = {
    str(MODEL_XML): (
        "fae1047f6a758ded4fab95f5faee9bf68f92b4433d778496bd9d44efa51cdbb0"),
    str(MODEL_BIN): (
        "46140b595760e891d9626c5bfaffc2c998cce176d0de7f6c290af5ae1f2393a4"),
    str(CANDIDATE_WORKER): (
        "34cf83a05187ec159e082b3ebd6d076f8dc072763b65aeb005dedd3909194ed9"),
    str(STOCK_WORKER): (
        "ff2d0187c03d5e74febe7cc5b94e1b68a530d07560e004725256d3358c18ceb9"),
    str(COMPONENT_RESULT): (
        "caf1814a1786f74e637b5aa398455bac64a831d4dd5fa22557a7def0919d9a73"),
    str(OPPORTUNITY_RESULT): (
        "029facf058d3201613785a7aacf8a2bb7d6d6b114e3eae49bc02daa17dff5752"),
}

WEIGHT_NAME = "self.model.lm_head.weight"
SCALE_NAME = "self.model.lm_head.weight/scale"
ROWS = 248_320
COLUMNS = 2_048
GROUP_SIZE = 256
CAPTURED_STEPS = 130
Q1_ANCHOR_STEP = 129
Q1_PACKED_BYTES = 66_053_120
Q4_PACKED_BYTES = 255_272_960
Q4_LLOYD_LEVELS = 16
Q4_LLOYD_PACKED_BYTES = (
    ROWS * COLUMNS // 2 + ROWS * Q4_LLOYD_LEVELS * 4 + ROWS * 2)
FULL_I8_SCAN_BYTES = 509_552_640
NORM_METADATA_VALUES_PER_ROW = 2
ROW_EXACT_BYTES = COLUMNS + 2
PREFLIGHT_AVAILABLE_BYTES = 8 * 1024**3
ABORT_AVAILABLE_BYTES = 4 * 1024**3
F32_UNIT_ROUNDOFF = 2.0**-24
F32_DOT_OPERATIONS = 4_096


def parser() -> argparse.ArgumentParser:
  result = argparse.ArgumentParser(description=__doc__)
  result.add_argument("--out", type=Path, required=True)
  result.add_argument("--row-chunk", type=int, default=256)
  result.add_argument(
      "--q4-active-byte-ratio-cap", type=float, default=0.60)
  return result


def run(command: list[str]) -> str:
  return subprocess.run(
      command, cwd=ROOT, check=True, text=True,
      stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.strip()


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def resolve(path: str) -> Path:
  value = Path(path)
  return value if value.is_absolute() else ROOT / value


def available_memory_bytes() -> int:
  for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
    if line.startswith("MemAvailable:"):
      return int(line.split()[1]) * 1024
  raise RuntimeError("MemAvailable is absent from /proc/meminfo")


def git_state() -> dict[str, Any]:
  status = run(["git", "status", "--porcelain=v1", "--untracked-files=all"])
  commit = run(["git", "rev-parse", "HEAD"])
  branch = run(["git", "branch", "--show-current"])
  upstream = run(["git", "rev-parse", "@{upstream}"])
  dirty_paths = [line[3:] for line in status.splitlines() if line]
  return {
      "branch": branch,
      "commit": commit,
      "dirty": bool(dirty_paths),
      "dirty_paths": dirty_paths,
      "pushed": commit == upstream,
      "upstream_commit": upstream,
  }


def ir_constant(name: str) -> dict[str, Any]:
  root = ET.parse(MODEL_XML).getroot()
  layers = root.find("layers")
  if layers is None:
    raise RuntimeError("model IR has no layers")
  layer = next(
      (value for value in layers if value.attrib.get("name") == name), None)
  if layer is None:
    raise RuntimeError(f"IR constant is absent: {name}")
  data = layer.find("data")
  if data is None:
    raise RuntimeError(f"IR constant has no data: {name}")
  return {
      "element_type": data.attrib["element_type"],
      "offset": int(data.attrib["offset"]),
      "shape": [int(item) for item in data.attrib["shape"].split(",")],
      "size": int(data.attrib["size"]),
  }


def declared_rows(worker: dict[str, Any], key: str) -> dict[int, dict[str, Any]]:
  rows = {int(row["step"]): row for row in worker.get(key, [])}
  if len(rows) != CAPTURED_STEPS or sorted(rows) != list(range(CAPTURED_STEPS)):
    raise RuntimeError(
        f"{key} must contain exact steps 0..{CAPTURED_STEPS - 1}")
  return rows


def verify_declared_file(row: dict[str, Any], expected_bytes: int) -> Path:
  path = resolve(row["file"])
  if path.stat().st_size != expected_bytes:
    raise RuntimeError(f"unexpected byte count for {path}")
  actual = sha256(path)
  if actual != row["sha256"]:
    raise RuntimeError(f"declared sha256 mismatch for {path}")
  return path


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
  dequantize_scale = (
      np.float16(1.0) / quantize_scale).astype(np.float16)
  return (
      codes.astype(np.float16) * dequantize_scale
  ).astype(np.float16).reshape(rows, columns).astype(np.float32)


def q1_dequantize_product(raw: Any, np: Any) -> Any:
  """Mirror the product prepack's signed split plus five Lloyd iterations."""
  integer = raw.astype(np.int16, copy=False)
  negative = integer < 0
  low_count = np.sum(negative, axis=1, dtype=np.int32)
  high_count = COLUMNS - low_count
  low_sum = np.sum(
      np.where(negative, integer, 0), axis=1, dtype=np.int64)
  high_sum = np.sum(
      np.where(negative, 0, integer), axis=1, dtype=np.int64)
  low = (
      low_sum.astype(np.float32) /
      np.maximum(low_count, 1).astype(np.float32)).astype(np.float32)
  high = (
      high_sum.astype(np.float32) /
      np.maximum(high_count, 1).astype(np.float32)).astype(np.float32)
  for _ in range(5):
    threshold = ((low + high) * np.float32(0.5)).astype(np.float32)
    select_low = integer.astype(np.float32) <= threshold[:, None]
    next_low_count = np.sum(select_low, axis=1, dtype=np.int32)
    next_high_count = COLUMNS - next_low_count
    next_low_sum = np.sum(
        np.where(select_low, integer, 0), axis=1, dtype=np.int64)
    next_high_sum = np.sum(
        np.where(select_low, 0, integer), axis=1, dtype=np.int64)
    has_low = next_low_count != 0
    has_high = next_high_count != 0
    low[has_low] = (
        next_low_sum[has_low].astype(np.float32) /
        next_low_count[has_low].astype(np.float32))
    high[has_high] = (
        next_high_sum[has_high].astype(np.float32) /
        next_high_count[has_high].astype(np.float32))
  threshold = ((low + high) * np.float32(0.5)).astype(np.float32)
  return np.where(
      integer.astype(np.float32) > threshold[:, None],
      high[:, None], low[:, None]).astype(np.float32)


def lloyd16_dequantize(raw: Any, np: Any) -> Any:
  """Build a per-row 16-centroid codec using five Lloyd iterations."""
  values = raw.astype(np.float32, copy=False)
  minimum = np.min(values, axis=1)
  maximum = np.max(values, axis=1)
  fractions = (
      (np.arange(Q4_LLOYD_LEVELS, dtype=np.float32) + np.float32(0.5)) /
      np.float32(Q4_LLOYD_LEVELS))
  centers = (
      minimum[:, None] +
      (maximum - minimum)[:, None] * fractions[None, :])
  for _ in range(5):
    assignments = np.argmin(
        np.abs(values[:, :, None] - centers[:, None, :]), axis=2)
    for cluster in range(Q4_LLOYD_LEVELS):
      selected = assignments == cluster
      counts = np.sum(selected, axis=1)
      sums = np.sum(values * selected, axis=1, dtype=np.float32)
      centers[:, cluster] = np.where(
          counts != 0, sums / np.maximum(counts, 1),
          centers[:, cluster])
  assignments = np.argmin(
      np.abs(values[:, :, None] - centers[:, None, :]), axis=2)
  return np.take_along_axis(
      centers[:, None, :], assignments[:, :, None], axis=2)[:, :, 0]


def upward_f32_norm(values: Any, np: Any) -> Any:
  """Return an outward-rounded F32 L2 norm for each row."""
  values64 = values.astype(np.float64, copy=False)
  square_sum = np.sum(values64 * values64, axis=1, dtype=np.float64)
  # Inflate the F64 accumulation before the F32 outward rounding.  The F64
  # dot error is tiny here, but keeping it explicit makes the proof contract
  # independent of an assumed correctly rounded vector reduction.
  gamma = COLUMNS * 2.0**-53 / (1.0 - COLUMNS * 2.0**-53)
  inflated = np.sqrt(square_sum * (1.0 + gamma))
  rounded = inflated.astype(np.float32)
  rounded = np.nextafter(
      rounded, np.full_like(rounded, np.float32(np.inf)))
  return rounded.astype(np.float64)


def upward_hidden_norm(values: Any, np: Any) -> Any:
  square_sum = np.sum(
      values.astype(np.float64) ** 2, axis=1, dtype=np.float64)
  gamma = COLUMNS * 2.0**-53 / (1.0 - COLUMNS * 2.0**-53)
  norm = np.sqrt(square_sum * (1.0 + gamma)).astype(np.float32)
  return np.nextafter(
      norm, np.full_like(norm, np.float32(np.inf))).astype(np.float64)


def f16_upper(values: Any, np: Any) -> Any:
  """Smallest representable F16 value no lower than each real-valued input."""
  rounded = values.astype(np.float16)
  rounded32 = rounded.astype(np.float32)
  below = rounded32.astype(np.float64) < values
  if np.any(below):
    rounded[below] = np.nextafter(
        rounded[below], np.full_like(rounded[below], np.float16(np.inf)))
  return rounded.astype(np.float32)


def f32_dot_guard(
    codec_norm: Any, hidden_norm: float, scales: Any, np: Any,
) -> Any:
  """Conservative error guard for an F32 implementation of the codec dot."""
  gamma = (
      F32_DOT_OPERATIONS * F32_UNIT_ROUNDOFF /
      (1.0 - F32_DOT_OPERATIONS * F32_UNIT_ROUNDOFF))
  magnitude = codec_norm * hidden_norm * np.abs(scales)
  # The gamma term covers multiply/add ordering; the extra two ulps cover the
  # final row scale and a future fused bound add before F16 outward rounding.
  return magnitude * gamma + np.maximum(
      magnitude, 1.0) * (2.0 * F32_UNIT_ROUNDOFF)


def percentile(values: list[int], value: float, np: Any) -> float:
  return float(np.percentile(np.asarray(values, dtype=np.float64), value))


def active_bytes(
    packed_bytes: int, exact_rows: int, *, include_materialized_output: bool,
) -> dict[str, int]:
  norm_metadata = ROWS * NORM_METADATA_VALUES_PER_ROW * 4
  materialized_output = ROWS * 2 if include_materialized_output else 0
  candidate_ids = exact_rows * 4
  exact_source = exact_rows * ROW_EXACT_BYTES
  exact_output = exact_rows * 2
  total = (
      packed_bytes + norm_metadata + materialized_output + candidate_ids +
      exact_source + exact_output)
  return {
      "candidate_id_bytes": candidate_ids,
      "exact_output_bytes": exact_output,
      "exact_source_bytes": exact_source,
      "materialized_f16_output_bytes": materialized_output,
      "norm_metadata_f32_bytes": norm_metadata,
      "packed_bytes": packed_bytes,
      "total_bytes": total,
  }


def load_inputs(np: Any) -> dict[str, Any]:
  candidate = json.loads(CANDIDATE_WORKER.read_text(encoding="utf-8"))
  stock = json.loads(STOCK_WORKER.read_text(encoding="utf-8"))
  hidden_rows = declared_rows(candidate, "lm_head_hidden_checkpoints")
  candidate_logits = declared_rows(candidate, "distribution_checkpoints")
  stock_logits = declared_rows(stock, "distribution_checkpoints")
  hidden = []
  stock_paths: dict[int, Path] = {}
  seeds: dict[int, dict[str, Any]] = {}
  capture_hashes = []
  for step in range(CAPTURED_STEPS):
    hidden_path = verify_declared_file(hidden_rows[step], COLUMNS * 4)
    stock_path = verify_declared_file(stock_logits[step], ROWS * 4)
    verify_declared_file(candidate_logits[step], ROWS * 4)
    values = np.fromfile(hidden_path, dtype="<f4")
    if not np.array_equal(
        values, values.astype(np.float16).astype(np.float32)):
      raise RuntimeError(f"hidden step {step} is not exactly F16-valued")
    hidden.append(values)
    stock_paths[step] = stock_path
    top8 = candidate_logits[step].get("top8", [])
    if len(top8) != 8:
      raise RuntimeError(f"candidate step {step} lacks exact top8 metadata")
    seed = top8[0]
    seed_id = int(seed["id"])
    seed_value = float(seed["value"])
    if not 0 <= seed_id < ROWS:
      raise RuntimeError(f"candidate step {step} has invalid seed id")
    seeds[step] = {"id": seed_id, "value": seed_value}
    capture_hashes.append({
        "candidate_logits_sha256": candidate_logits[step]["sha256"],
        "hidden_sha256": hidden_rows[step]["sha256"],
        "step": step,
        "stock_logits_sha256": stock_logits[step]["sha256"],
    })
  return {
      "candidate": candidate,
      "capture_hashes": capture_hashes,
      "hidden": np.stack(hidden),
      "seeds": seeds,
      "stock": stock,
      "stock_paths": stock_paths,
  }


def evaluate_step(
    *, approximate: Any, codec_norm: Any, residual_norm: Any,
    hidden_norm: float, hidden_delta_norm: float, scale: Any, seed: dict[str, Any],
    reference: Any, packed_bytes: int, np: Any,
) -> dict[str, Any]:
  scale_abs = np.abs(scale)
  residual_bound = scale_abs * (
      residual_norm * hidden_norm + codec_norm * hidden_delta_norm)
  round_guard = f32_dot_guard(codec_norm, hidden_norm, scale, np)
  upper_real = (
      approximate.astype(np.float64) + residual_bound + round_guard)
  upper_output = f16_upper(upper_real, np)
  seed_id = int(seed["id"])
  seed_value = float(seed["value"])
  reference_seed_value = float(reference[seed_id])
  if reference_seed_value != seed_value:
    raise RuntimeError(
        f"captured direct-corrected seed differs from stock at row {seed_id}: "
        f"{seed_value} vs {reference_seed_value}")
  selected = np.flatnonzero(upper_output >= np.float32(seed_value))
  reference_top1 = int(np.argmax(reference))
  candidate_values = reference[selected]
  selected_best_offset = int(np.argmax(candidate_values))
  selected_top1 = int(selected[selected_best_offset])
  selected_top1_value = float(candidate_values[selected_best_offset])
  if selected_top1_value < seed_value:
    certified_top1 = seed_id
    certified_value = seed_value
  elif selected_top1_value == seed_value:
    certified_top1 = min(seed_id, selected_top1)
    certified_value = seed_value
  else:
    certified_top1 = selected_top1
    certified_value = selected_top1_value
  violations = np.flatnonzero(
      reference.astype(np.float32) > upper_output)
  traffic = active_bytes(
      packed_bytes, int(selected.size), include_materialized_output=True)
  return {
      "bound_violation_count": int(violations.size),
      "bound_violation_first_ids": [
          int(value) for value in violations[:8]],
      "candidate_fraction": float(selected.size / ROWS),
      "certified_top1": certified_top1,
      "certified_top1_value_f16": certified_value,
      "certificate_pass": (
          violations.size == 0 and certified_top1 == reference_top1),
      "exact_candidate_rows": int(selected.size),
      "reference_top1": reference_top1,
      "reference_top1_value_f16": float(reference[reference_top1]),
      "residual_bound_max": float(np.max(residual_bound)),
      "residual_bound_median": float(np.median(residual_bound)),
      "round_guard_max": float(np.max(round_guard)),
      "seed_id": seed_id,
      "seed_value_f16": seed_value,
      "traffic": traffic,
      "traffic_ratio_vs_full_i8": (
          traffic["total_bytes"] / FULL_I8_SCAN_BYTES),
      "upper_output_max": float(np.max(upper_output)),
  }


def summary_markdown(result: dict[str, Any]) -> str:
  q1 = result["q1_anchor"]
  q4 = result["q4_summary"]
  adaptive = result["q4_lloyd_summary"]
  return "\n".join([
      "# LM-head exact-token certificate bound",
      "",
      f"- verdict: `{result['verdict']}`",
      f"- commit: `{result['git']['commit']}`",
      f"- captured hidden rows: `{result['inputs']['captured_steps']}`",
      (
          f"- Q1/global-L2 anchor: `{q1['exact_candidate_rows']}/"
          f"{ROWS}` exact rows, traffic ratio "
          f"`{q1['traffic_ratio_vs_full_i8']:.6f}`"),
      (
          f"- Q4/global-L2: `{q4['certificate_pass_count']}/"
          f"{CAPTURED_STEPS}` certificates, zero-bound-violation rows "
          f"`{q4['zero_bound_violation_step_count']}/{CAPTURED_STEPS}`"),
      (
          f"- Q4 exact rows p50/p95/max: "
          f"`{q4['exact_candidate_rows']['p50']:.1f}/"
          f"{q4['exact_candidate_rows']['p95']:.1f}/"
          f"{q4['exact_candidate_rows']['maximum']}`"),
      (
          f"- Q4 worst active bytes / full I8: "
          f"`{q4['traffic']['maximum_active_bytes']}/"
          f"{FULL_I8_SCAN_BYTES}` "
          f"(`{q4['traffic']['maximum_ratio_vs_full_i8']:.6f}`)"),
      (
          f"- Lloyd-Q4/global-L2: `{adaptive['certificate_pass_count']}/"
          f"{CAPTURED_STEPS}` certificates; exact rows p50/p95/max "
          f"`{adaptive['exact_candidate_rows']['p50']:.1f}/"
          f"{adaptive['exact_candidate_rows']['p95']:.1f}/"
          f"{adaptive['exact_candidate_rows']['maximum']}`"),
      (
          f"- Lloyd-Q4 worst active bytes / full I8: "
          f"`{adaptive['traffic']['maximum_active_bytes']}/"
          f"{FULL_I8_SCAN_BYTES}` "
          f"(`{adaptive['traffic']['maximum_ratio_vs_full_i8']:.6f}`)"),
      (
          f"- Lloyd-Q4 optimistic worst-row traffic headroom: "
          f"`{adaptive['traffic']['optimistic_worst_row_saving_ms']:.6f} ms` "
          f"per slow event; no speedup claim"),
      "",
      "Q1 is closed only for this conservative global-L2 certificate "
      "topology. Fixed signed-Q4 is exact but misses the registered worst-byte "
      "cap. Per-row Lloyd-Q4 is admitted only to a larger slow-event capture; "
      "captured agreement and traffic arithmetic do not authorize a plugin "
      "change or a product speed claim.",
      "",
  ])


def main() -> int:
  args = parser().parse_args()
  if args.out.exists():
    raise SystemExit(f"output already exists: {args.out}")
  if args.row_chunk <= 0:
    raise SystemExit("row chunk must be positive")
  if not 0.0 < args.q4_active_byte_ratio_cap < 1.0:
    raise SystemExit("Q4 active-byte ratio cap must be in (0,1)")
  memory_start = available_memory_bytes()
  if memory_start < PREFLIGHT_AVAILABLE_BYTES:
    raise SystemExit(
        f"preflight memory below {PREFLIGHT_AVAILABLE_BYTES}: {memory_start}")

  import numpy as np

  started = time.monotonic()
  git = git_state()
  registered_hashes = {path: sha256(Path(path)) for path in EXPECTED_SHA256}
  hash_mismatches = {
      path: {"actual": registered_hashes[path], "expected": expected}
      for path, expected in EXPECTED_SHA256.items()
      if registered_hashes[path] != expected}
  if hash_mismatches:
    raise SystemExit(
        "registered input hash mismatch: " +
        json.dumps(hash_mismatches, sort_keys=True))

  weight_fact = ir_constant(WEIGHT_NAME)
  scale_fact = ir_constant(SCALE_NAME)
  if (
      weight_fact != {
          "element_type": "i8", "offset": 18_137_149_498,
          "shape": [ROWS, COLUMNS], "size": ROWS * COLUMNS}
      or scale_fact != {
          "element_type": "f16", "offset": 18_645_708_858,
          "shape": [ROWS, 1], "size": ROWS * 2}
  ):
    raise SystemExit(
        f"LM-head IR facts drifted: {weight_fact}, {scale_fact}")

  inputs = load_inputs(np)
  hidden = inputs["hidden"]
  quantized_hidden = q8_hidden(hidden, np)
  hidden_delta = hidden.astype(np.float32) - quantized_hidden
  hidden_norm = upward_hidden_norm(hidden, np)
  hidden_delta_norm = upward_hidden_norm(hidden_delta, np)
  quantized_hidden_t = np.ascontiguousarray(
      quantized_hidden.T, dtype=np.float32)
  weights = np.memmap(
      MODEL_BIN, dtype=np.int8, mode="r", offset=weight_fact["offset"],
      shape=(ROWS, COLUMNS))
  scales_memmap = np.memmap(
      MODEL_BIN, dtype="<f2", mode="r", offset=scale_fact["offset"],
      shape=(ROWS,))
  scales = np.asarray(scales_memmap, dtype=np.float32)

  q4_approximate = np.empty((CAPTURED_STEPS, ROWS), dtype=np.float32)
  q4_codec_norm = np.empty(ROWS, dtype=np.float64)
  q4_residual_norm = np.empty(ROWS, dtype=np.float64)
  q4_lloyd_approximate = np.empty(
      (CAPTURED_STEPS, ROWS), dtype=np.float32)
  q4_lloyd_codec_norm = np.empty(ROWS, dtype=np.float64)
  q4_lloyd_residual_norm = np.empty(ROWS, dtype=np.float64)
  q1_approximate = np.empty(ROWS, dtype=np.float32)
  q1_codec_norm = np.empty(ROWS, dtype=np.float64)
  q1_residual_norm = np.empty(ROWS, dtype=np.float64)
  anchor_hidden = quantized_hidden[Q1_ANCHOR_STEP]
  last_progress = time.monotonic()
  memory_min = memory_start

  for begin in range(0, ROWS, args.row_chunk):
    if available_memory_bytes() < ABORT_AVAILABLE_BYTES:
      raise SystemExit(
          f"runtime memory fell below {ABORT_AVAILABLE_BYTES} at row {begin}")
    end = min(ROWS, begin + args.row_chunk)
    raw = np.asarray(weights[begin:end], dtype=np.float32)
    q4 = (
        np.clip(np.rint(raw / np.float32(16.0)), -8, 7) *
        np.float32(16.0)).astype(np.float32)
    q4_approximate[:, begin:end] = (
        q4 @ quantized_hidden_t *
        scales[begin:end, None]).T.astype(np.float32)
    q4_codec_norm[begin:end] = upward_f32_norm(q4, np)
    q4_residual_norm[begin:end] = upward_f32_norm(raw - q4, np)

    q4_lloyd = lloyd16_dequantize(raw, np)
    q4_lloyd_approximate[:, begin:end] = (
        q4_lloyd @ quantized_hidden_t *
        scales[begin:end, None]).T.astype(np.float32)
    q4_lloyd_codec_norm[begin:end] = upward_f32_norm(q4_lloyd, np)
    q4_lloyd_residual_norm[begin:end] = upward_f32_norm(
        raw - q4_lloyd, np)

    q1 = q1_dequantize_product(raw, np)
    q1_approximate[begin:end] = (
        q1 @ anchor_hidden * scales[begin:end]).astype(np.float32)
    q1_codec_norm[begin:end] = upward_f32_norm(q1, np)
    q1_residual_norm[begin:end] = upward_f32_norm(raw - q1, np)

    now = time.monotonic()
    if now - last_progress >= 20.0:
      print(json.dumps({
          "elapsed_seconds": round(now - started, 3),
          "event": "certificate_bound_progress",
          "rows_complete": end,
          "rows_total": ROWS,
      }, sort_keys=True), flush=True)
      last_progress = now
    memory_min = min(memory_min, available_memory_bytes())

  anchor_reference = np.fromfile(
      inputs["stock_paths"][Q1_ANCHOR_STEP], dtype="<f4")
  q1_anchor = evaluate_step(
      approximate=q1_approximate,
      codec_norm=q1_codec_norm,
      residual_norm=q1_residual_norm,
      hidden_norm=float(hidden_norm[Q1_ANCHOR_STEP]),
      hidden_delta_norm=float(hidden_delta_norm[Q1_ANCHOR_STEP]),
      scale=scales,
      seed=inputs["seeds"][Q1_ANCHOR_STEP],
      reference=anchor_reference,
      packed_bytes=Q1_PACKED_BYTES,
      np=np)
  q1_anchor["step"] = Q1_ANCHOR_STEP

  q4_rows = []
  q4_lloyd_rows = []
  for step in range(CAPTURED_STEPS):
    reference = np.fromfile(inputs["stock_paths"][step], dtype="<f4")
    row = evaluate_step(
        approximate=q4_approximate[step],
        codec_norm=q4_codec_norm,
        residual_norm=q4_residual_norm,
        hidden_norm=float(hidden_norm[step]),
        hidden_delta_norm=float(hidden_delta_norm[step]),
        scale=scales,
        seed=inputs["seeds"][step],
        reference=reference,
        packed_bytes=Q4_PACKED_BYTES,
        np=np)
    row["step"] = step
    q4_rows.append(row)
    adaptive_row = evaluate_step(
        approximate=q4_lloyd_approximate[step],
        codec_norm=q4_lloyd_codec_norm,
        residual_norm=q4_lloyd_residual_norm,
        hidden_norm=float(hidden_norm[step]),
        hidden_delta_norm=float(hidden_delta_norm[step]),
        scale=scales,
        seed=inputs["seeds"][step],
        reference=reference,
        packed_bytes=Q4_LLOYD_PACKED_BYTES,
        np=np)
    adaptive_row["step"] = step
    q4_lloyd_rows.append(adaptive_row)

  component = json.loads(COMPONENT_RESULT.read_text(encoding="utf-8"))
  opportunity = json.loads(OPPORTUNITY_RESULT.read_text(encoding="utf-8"))
  bandwidth_lcb = float(component["bound"]["accepted_bandwidth_lcb_gb_s"])
  required_saving_ms = float(component["bound"]["required_saving_us"]) / 1000.0
  fallback_rate = float(
      opportunity["route_comparison"]["lm_head_exact_fallback"][
          "observed_fallback_rate"])
  full_floor_ms = (
      FULL_I8_SCAN_BYTES / (bandwidth_lcb * 1.0e9) * 1.0e3)

  def codec_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    exact_counts = [row["exact_candidate_rows"] for row in rows]
    maximum_active_bytes = max(
        row["traffic"]["total_bytes"] for row in rows)
    maximum_ratio = maximum_active_bytes / FULL_I8_SCAN_BYTES
    worst_floor_ms = (
        maximum_active_bytes / (bandwidth_lcb * 1.0e9) * 1.0e3)
    optimistic_saving_ms = full_floor_ms - worst_floor_ms
    optimistic_mean_saving_ms = optimistic_saving_ms * fallback_rate
    return {
        "certificate_pass_count": sum(
            bool(row["certificate_pass"]) for row in rows),
        "exact_candidate_rows": {
            "maximum": max(exact_counts),
            "minimum": min(exact_counts),
            "p50": percentile(exact_counts, 50, np),
            "p95": percentile(exact_counts, 95, np),
            "p99": percentile(exact_counts, 99, np),
        },
        "rows": rows,
        "traffic": {
            "accepted_bandwidth_lcb_gb_s": bandwidth_lcb,
            "full_i8_floor_ms": full_floor_ms,
            "full_i8_scan_bytes": FULL_I8_SCAN_BYTES,
            "maximum_active_bytes": maximum_active_bytes,
            "maximum_ratio_vs_full_i8": maximum_ratio,
            "optimistic_mean_saving_at_observed_fallback_rate_ms": (
                optimistic_mean_saving_ms),
            "optimistic_mean_saving_multiple_over_required": (
                optimistic_mean_saving_ms / required_saving_ms),
            "optimistic_worst_row_floor_ms": worst_floor_ms,
            "optimistic_worst_row_saving_ms": optimistic_saving_ms,
            "required_product_saving_ms": required_saving_ms,
        },
        "zero_bound_violation_step_count": sum(
            row["bound_violation_count"] == 0 for row in rows),
    }

  q4_summary = codec_summary(q4_rows)
  q4_lloyd_summary = codec_summary(q4_lloyd_rows)
  maximum_ratio = q4_summary["traffic"]["maximum_ratio_vs_full_i8"]
  adaptive_maximum_ratio = (
      q4_lloyd_summary["traffic"]["maximum_ratio_vs_full_i8"])
  adaptive_optimistic_saving_ms = (
      q4_lloyd_summary["traffic"]["optimistic_worst_row_saving_ms"])
  adaptive_optimistic_mean_saving_ms = (
      q4_lloyd_summary["traffic"][
          "optimistic_mean_saving_at_observed_fallback_rate_ms"])

  checks = [
      {
          "git": git,
          "name": "repository_clean_and_pushed_at_gate",
          "pass": (
              git["branch"] == "main" and not git["dirty"] and git["pushed"]),
      },
      {
          "mismatches": hash_mismatches,
          "name": "registered_inputs_match_exact_hashes",
          "pass": not hash_mismatches,
      },
      {
          "name": "locked_ir_and_capture_geometry_match",
          "pass": (
              hidden.shape == (CAPTURED_STEPS, COLUMNS) and
              weight_fact["shape"] == [ROWS, COLUMNS] and
              scale_fact["shape"] == [ROWS, 1]),
      },
      {
          "name": "captured_hidden_is_exactly_f16_valued",
          "pass": bool(np.array_equal(
              hidden, hidden.astype(np.float16).astype(np.float32))),
      },
      {
          "name": "q1_global_l2_anchor_is_conservative_but_full_scan",
          "pass": (
              q1_anchor["bound_violation_count"] == 0 and
              q1_anchor["certificate_pass"] and
              q1_anchor["exact_candidate_rows"] == ROWS),
      },
      {
          "name": "q4_global_l2_bound_dominates_every_captured_stock_logit",
          "pass": q4_summary["zero_bound_violation_step_count"] == CAPTURED_STEPS,
      },
      {
          "name": "q4_global_l2_certificate_returns_every_exact_stock_token",
          "pass": q4_summary["certificate_pass_count"] == CAPTURED_STEPS,
      },
      {
          "cap": args.q4_active_byte_ratio_cap,
          "maximum": maximum_ratio,
          "name": "fixed_q4_is_correct_but_misses_registered_ratio_cap",
          "pass": maximum_ratio > args.q4_active_byte_ratio_cap,
      },
      {
          "name": (
              "lloyd_q4_global_l2_bound_dominates_every_captured_stock_logit"),
          "pass": (
              q4_lloyd_summary["zero_bound_violation_step_count"] ==
              CAPTURED_STEPS),
      },
      {
          "name": (
              "lloyd_q4_global_l2_certificate_returns_every_exact_stock_token"),
          "pass": (
              q4_lloyd_summary["certificate_pass_count"] == CAPTURED_STEPS),
      },
      {
          "cap": args.q4_active_byte_ratio_cap,
          "maximum": adaptive_maximum_ratio,
          "name": (
              "lloyd_q4_worst_captured_active_bytes_clear_registered_ratio_cap"),
          "pass": adaptive_maximum_ratio <= args.q4_active_byte_ratio_cap,
      },
      {
          "name": "lloyd_q4_traffic_headroom_funds_larger_slow_event_capture",
          "pass": (
              adaptive_optimistic_saving_ms > 0.0 and
              adaptive_optimistic_mean_saving_ms >= required_saving_ms),
      },
      {
          "compiler_invocations": 0,
          "gpu_contexts": 0,
          "infer_requests": 0,
          "model_compiles": 0,
          "model_workers": 0,
          "name": "no_compiler_gpu_or_model_worker_ran",
          "pass": True,
          "product_workers": 0,
      },
  ]
  required_checks_passed = all(bool(row["pass"]) for row in checks)
  verdict = (
      "reject_q1_and_fixed_q4_fund_lloyd_q4_slow_event_capture"
      if required_checks_passed else
      "reject_exact_token_certificate_bound")
  result = {
      "checks": checks,
      "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
      "git": git,
      "inputs": {
          "candidate_worker": str(CANDIDATE_WORKER),
          "capture_hashes": inputs["capture_hashes"],
          "captured_steps": CAPTURED_STEPS,
          "component_result": str(COMPONENT_RESULT),
          "model_bin": str(MODEL_BIN),
          "model_xml": str(MODEL_XML),
          "opportunity_result": str(OPPORTUNITY_RESULT),
          "registered_sha256": registered_hashes,
          "stock_worker": str(STOCK_WORKER),
      },
      "memory": {
          "abort_below_bytes": ABORT_AVAILABLE_BYTES,
          "available_at_start_bytes": memory_start,
          "available_min_bytes": memory_min,
          "oom_observed": False,
          "peak_rss_bytes": (
              resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
          "preflight_bytes": PREFLIGHT_AVAILABLE_BYTES,
      },
      "next_gate": {
          "admission": "captured_slow_events_only",
          "capture_target": 2_000,
          "forbidden_claim": (
              "the traffic floor is optimistic and is not a latency or "
              "speedup claim"),
          "route": (
              "lm_head_lloyd_q4_global_l2_exact_token_certificate_capture"),
          "source_edit_admitted": False,
      },
      "q1_anchor": q1_anchor,
      "q4_lloyd_summary": q4_lloyd_summary,
      "q4_summary": q4_summary,
      "required_checks_passed": required_checks_passed,
      "schema": (
          "intel-qwen36-openvino-lm-head-exact-token-certificate-bound-v1"),
      "speedup_claims_allowed": False,
      "verdict": verdict,
      "workers": {
          "compiler_invocations": 0,
          "gpu_contexts": 0,
          "infer_requests": 0,
          "model_compiles": 0,
          "model_workers": 0,
          "oom_observed": False,
          "product_workers": 0,
          "workers_concurrent": False,
      },
      "workstream": WORKSTREAM,
  }
  args.out.mkdir(parents=True)
  (args.out / "metrics.json").write_text(
      json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  (args.out / "summary.md").write_text(
      summary_markdown(result), encoding="utf-8")
  print(json.dumps({
      "elapsed_seconds": round(time.monotonic() - started, 3),
      "event": "lm_head_exact_token_certificate_bound_complete",
      "out": str(args.out),
      "required_checks_passed": required_checks_passed,
      "verdict": verdict,
  }, sort_keys=True))
  return 0 if required_checks_passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
