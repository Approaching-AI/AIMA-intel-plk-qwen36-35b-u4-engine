#!/usr/bin/env python3
"""Bound adaptive exact correction for grouped-quantized attention state."""

from __future__ import annotations

import argparse
import json
import math
import resource
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LAYERS = (3, 7, 11, 15, 19, 23, 27, 31, 35, 39)
HEAD_DIM = 256
QUERY_HEADS = 16
KV_HEADS = 2
GQA_GROUP = QUERY_HEADS // KV_HEADS
SINK_TOKENS = 4


def parse_csv_ints(value: str) -> tuple[int, ...]:
  parsed = tuple(int(item) for item in value.split(",") if item)
  if not parsed or len(parsed) != len(set(parsed)):
    raise argparse.ArgumentTypeError("expected unique comma-separated integers")
  return parsed


def parser() -> argparse.ArgumentParser:
  result = argparse.ArgumentParser(description=__doc__)
  result.add_argument("--history-worker", type=Path, required=True)
  result.add_argument("--boundary-worker", type=Path, required=True)
  result.add_argument(
      "--candidate-boundary-worker", type=Path,
      help=(
          "optional matched candidate worker; when present, compare the "
          "captured graph output with the runtime-shaped arithmetic model"))
  result.add_argument("--step", type=int, required=True)
  result.add_argument("--layers", type=parse_csv_ints, default=DEFAULT_LAYERS)
  result.add_argument("--hot-tokens", type=int, default=16384)
  result.add_argument(
      "--quant-group", type=int, default=32,
      help="legacy shorthand applied to both K and V unless overridden")
  result.add_argument(
      "--key-quant-group", type=int,
      help="independent K scale group; overrides --quant-group")
  result.add_argument(
      "--value-quant-group", type=int,
      help="independent V scale group; overrides --quant-group")
  result.add_argument(
      "--key-quant-bits", type=int, choices=range(4, 9), default=8,
      help="signed symmetric K payload width")
  result.add_argument(
      "--value-quant-bits", type=int, choices=range(4, 9), default=8,
      help="signed symmetric V payload width")
  result.add_argument(
      "--key-exact-layers", type=parse_csv_ints, default=(),
      help=(
          "layers that scan retained dense F16 K while keeping grouped-I8 V; "
          "the default keeps grouped-I8 K on every layer"))
  result.add_argument(
      "--value-exact-layers", type=parse_csv_ints, default=(),
      help=(
          "layers that scan retained dense F16 V while keeping grouped-I8 K; "
          "the default keeps grouped-I8 V on every layer"))
  result.add_argument(
      "--residual-bits", type=int, choices=(0, 1, 2, 3), default=0,
      help=(
          "fractional bit-planes appended to selected grouped-I8 K/V layers; "
          "the planes share the existing F16 group scale"))
  result.add_argument(
      "--key-residual-layers", type=parse_csv_ints, default=(),
      help="layers whose grouped-I8 K state carries residual bit-planes")
  result.add_argument(
      "--value-residual-layers", type=parse_csv_ints, default=(),
      help="layers whose grouped-I8 V state carries residual bit-planes")
  result.add_argument("--topk-per-query", type=int, default=256)
  result.add_argument("--high-topk-layers", type=parse_csv_ints,
                      default=(3, 7))
  result.add_argument("--high-topk-per-query", type=int, default=512)
  result.add_argument("--candidate-chunk-tokens", type=int, default=512)
  result.add_argument("--local-topk-per-query", type=int, default=64)
  result.add_argument(
      "--selection-rule", choices=("score", "oracle-impact"),
      default="score",
      help=(
          "candidate ranking rule; oracle-impact is a diagnostic upper bound "
          "that reads exact K/V and is not an implementable product route"))
  result.add_argument("--base-context", type=int, default=32768)
  result.add_argument("--target-context", type=int, default=65536)
  result.add_argument("--required-long-context-speedup", type=float,
                      default=1.423)
  result.add_argument("--max-aggregate-relative-l2", type=float,
                      default=2.0e-5)
  result.add_argument("--max-layer-relative-l2", type=float, default=1.0e-4)
  result.add_argument("--min-error-reduction", type=float, default=32.0)
  result.add_argument("--out-dir", type=Path, required=True)
  return result


def load_json(path: Path) -> dict[str, Any]:
  return json.loads(path.read_text(encoding="utf-8"))


def resolve_artifact(path: str) -> Path:
  value = Path(path)
  return value if value.is_absolute() else ROOT / value


def git_state() -> dict[str, Any]:
  commit = subprocess.check_output(
      ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
  dirty_paths = subprocess.check_output(
      ["git", "status", "--short"], cwd=ROOT, text=True).splitlines()
  return {"commit": commit, "dirty": bool(dirty_paths),
          "dirty_paths": dirty_paths}


def mem_available_bytes() -> int:
  for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
    if line.startswith("MemAvailable:"):
      return int(line.split()[1]) * 1024
  raise RuntimeError("MemAvailable missing from /proc/meminfo")


def dequantize_group(
    values: Any, np: Any, group: int, quant_bits: int,
    residual_bits: int = 0,
) -> Any:
  grouped = values.reshape(values.shape[0], HEAD_DIM // group, group)
  max_code = (1 << (quant_bits - 1)) - 1
  scale = np.max(np.abs(grouped), axis=2) / np.float32(max_code)
  scale = np.where(scale == 0, np.float32(1.0), scale)
  stored_scale = scale.astype(np.float16).astype(np.float32)
  if residual_bits == 0:
    quantized = np.rint(grouped / scale[:, :, None])
    quantized = np.clip(quantized, -max_code, max_code).astype(np.int8)
    dequantized = quantized.astype(np.float32) * stored_scale[:, :, None]
  else:
    # The extra planes are the low bits of one uniform signed code.  The high
    # eight bits remain an INT8 base and all fractional bits share the stored
    # F16 scale, so this representation needs no second scale stream.
    levels = 1 << residual_bits
    fine = np.rint(
        grouped * np.float32(levels) / stored_scale[:, :, None])
    fine = np.clip(
        fine, -max_code * levels, max_code * levels).astype(np.int16)
    base = np.floor_divide(fine + levels // 2, levels).astype(np.int8)
    residual = fine - base.astype(np.int16) * levels
    dequantized = (
        base.astype(np.float32) +
        residual.astype(np.float32) / np.float32(levels)) * (
            stored_scale[:, :, None])
  return dequantized.reshape(values.shape)


def softmax_output(scores: Any, values: Any, np: Any) -> tuple[Any, Any]:
  weights = np.exp(scores - np.max(scores), dtype=np.float32)
  weights /= np.sum(weights, dtype=np.float32)
  return weights @ values, weights


def adaptive_graph_outputs(
    approximate_scores: Any, exact_scores: Any,
    approximate_values: Any, exact_values: Any, selected: Any,
    partition_tokens: int, np: Any,
) -> tuple[Any, Any]:
  """Model the current pre-softmax replacement and partition arithmetic."""
  approximate_values = approximate_values.astype(
      np.float16).astype(np.float32)
  exact_values = exact_values.astype(np.float16).astype(np.float32)

  def partition_output(scores: Any, values: Any) -> Any:
    running_max = np.float32(-math.inf)
    running_sum = np.float32(0.0)
    numerator = np.zeros((HEAD_DIM,), dtype=np.float32)
    for begin in range(0, scores.size, partition_tokens):
      end = min(begin + partition_tokens, scores.size)
      part_scores = scores[begin:end]
      part_max = np.float32(np.max(part_scores))
      weights = np.exp(
          part_scores - part_max, dtype=np.float32)
      part_sum = np.sum(weights, dtype=np.float32)
      part_numerator = (
          weights.astype(np.float16).astype(np.float32) @
          values[begin:end])
      next_max = np.float32(max(running_max, part_max))
      previous_scale = np.float32(math.exp(float(running_max - next_max)))
      partition_scale = np.float32(math.exp(float(part_max - next_max)))
      running_sum = np.float32(
          running_sum * previous_scale + part_sum * partition_scale)
      numerator = (
          numerator * previous_scale + part_numerator * partition_scale
      ).astype(np.float32)
      running_max = next_max
    return numerator * np.float32(1.0 / running_sum)

  base_output = partition_output(
      approximate_scores, approximate_values)
  corrected_scores = approximate_scores.copy()
  corrected_values = approximate_values.copy()
  corrected_scores[selected] = exact_scores[selected]
  corrected_values[selected] = exact_values[selected]
  corrected_output = partition_output(corrected_scores, corrected_values)
  # Both stock and custom graph boundaries publish F16 attention tensors.
  return tuple(
      value.astype(np.float16).astype(np.float32)
      for value in (base_output, corrected_output))


def deterministic_topk(
    values: Any, indices: Any, count: int, np: Any,
) -> Any:
  """Return top-k indices by score descending, then token index ascending."""
  if values.ndim != 1 or indices.ndim != 1 or values.size != indices.size:
    raise ValueError("deterministic top-k expects matching one-dimensional rows")
  if not 0 < count <= values.size:
    raise ValueError("deterministic top-k count is outside the input row")
  order = np.lexsort((indices, -values.astype(np.float32)))
  return indices[order[:count]]


def replacement_correction_output(
    approximate_scores: Any,
    subtraction_scores: Any,
    exact_scores: Any,
    approximate_values: Any,
    exact_values: Any,
    selected: Any,
    np: Any,
) -> Any:
  """Apply selected-row replacement over the untouched approximate base.

  `subtraction_scores` makes the graph's current F16 score-sidecar behavior
  explicit.  Passing the original approximate scores models an exact
  compressed-K recompute in the correction stage.
  """
  selected_exact = exact_scores[selected]
  selected_subtraction = subtraction_scores[selected]
  final_max = np.max(approximate_scores)
  if selected.size:
    final_max = max(
        final_max, np.max(selected_exact), np.max(selected_subtraction))
  base_weights = np.exp(
      approximate_scores - np.float32(final_max), dtype=np.float32)
  denominator = np.sum(base_weights, dtype=np.float32)
  numerator = base_weights @ approximate_values
  if selected.size:
    exact_weights = np.exp(
        selected_exact - np.float32(final_max), dtype=np.float32)
    subtraction_weights = np.exp(
        selected_subtraction - np.float32(final_max), dtype=np.float32)
    numerator += (
        exact_weights @ exact_values[selected] -
        subtraction_weights @ approximate_values[selected])
    denominator += np.sum(
        exact_weights - subtraction_weights, dtype=np.float32)
  return numerator / denominator


def metrics(reference: Any, candidate: Any, np: Any) -> dict[str, float]:
  reference64 = reference.astype(np.float64)
  candidate64 = candidate.astype(np.float64)
  delta = candidate64 - reference64
  per_head = np.linalg.norm(delta, axis=1) / np.linalg.norm(
      reference64, axis=1)
  denominator = float(
      np.linalg.norm(reference64) * np.linalg.norm(candidate64))
  return {
      "cosine": float(np.vdot(reference64, candidate64) / denominator),
      "max_abs": float(np.max(np.abs(delta))),
      "median_head_relative_l2": float(np.median(per_head)),
      "relative_l2": float(
          np.linalg.norm(delta) / np.linalg.norm(reference64)),
      "worst_head_relative_l2": float(np.max(per_head)),
  }


def checkpoint_maps(
    history: dict[str, Any], boundary: dict[str, Any], step: int,
) -> tuple[dict[tuple[int, str], dict[str, Any]],
           dict[int, dict[str, Any]]]:
  histories = {
      (int(row["layer"]), str(row["role"])): row
      for row in history.get("attention_history_checkpoints", [])
      if int(row["step"]) == step}
  boundaries = {
      int(row["layer"]): row["tensors"]
      for row in boundary.get("attention_checkpoints", [])
      if int(row["step"]) == step}
  return histories, boundaries


def main() -> int:
  args = parser().parse_args()
  key_quant_group = args.key_quant_group or args.quant_group
  value_quant_group = args.value_quant_group or args.quant_group
  admitted_groups = (4, 8, 16, 32)
  if (key_quant_group not in admitted_groups or
      value_quant_group not in admitted_groups or
      HEAD_DIM % key_quant_group or HEAD_DIM % value_quant_group):
    raise SystemExit(
        "key/value quant groups must independently be 4, 8, 16, or 32")
  if args.target_context <= args.base_context:
    raise SystemExit("target context must exceed base context")
  if not 0 < args.hot_tokens < args.base_context:
    raise SystemExit("hot tokens must be inside the base context")
  if args.target_context - args.hot_tokens > 65536:
    raise SystemExit("U16 selection records cannot address the target cold state")
  if (args.topk_per_query <= 0 or
      args.high_topk_per_query < args.topk_per_query):
    raise SystemExit("top-k capacities must be positive and ordered")
  if not set(args.high_topk_layers).issubset(args.layers):
    raise SystemExit("high-top-k layers must be a subset of selected layers")
  if not set(args.key_exact_layers).issubset(args.layers):
    raise SystemExit("key-exact layers must be a subset of selected layers")
  if not set(args.value_exact_layers).issubset(args.layers):
    raise SystemExit("value-exact layers must be a subset of selected layers")
  if not set(args.key_residual_layers).issubset(args.layers):
    raise SystemExit("key-residual layers must be a subset of selected layers")
  if not set(args.value_residual_layers).issubset(args.layers):
    raise SystemExit("value-residual layers must be a subset of selected layers")
  if ((args.key_residual_layers or args.value_residual_layers) and
      args.residual_bits == 0):
    raise SystemExit("residual layers require --residual-bits above zero")
  if args.key_residual_layers and args.key_quant_bits != 8:
    raise SystemExit("key residual planes currently require 8-bit K")
  if args.value_residual_layers and args.value_quant_bits != 8:
    raise SystemExit("value residual planes currently require 8-bit V")
  if set(args.key_exact_layers) & set(args.key_residual_layers):
    raise SystemExit("key-exact and key-residual layers must be disjoint")
  if set(args.value_exact_layers) & set(args.value_residual_layers):
    raise SystemExit("value-exact and value-residual layers must be disjoint")
  if (args.candidate_chunk_tokens <= 0 or
      not 0 < args.local_topk_per_query <= args.candidate_chunk_tokens):
    raise SystemExit("invalid chunk/local-top-k geometry")
  if args.out_dir.exists():
    raise SystemExit(f"output already exists: {args.out_dir}")

  import numpy as np

  started_available = mem_available_bytes()
  history = load_json(args.history_worker)
  boundary = load_json(args.boundary_worker)
  candidate_boundary = (
      load_json(args.candidate_boundary_worker)
      if args.candidate_boundary_worker is not None else None)
  prefix_count = args.step + 1
  identity_checks = {
      "generated_prefix_exact": (
          history.get("generated_token_ids") ==
          boundary.get("generated_token_ids", [])[:prefix_count]),
      "input_token_digest_exact": (
          history.get("input_token_ids_sha256") ==
          boundary.get("input_token_ids_sha256")),
      "prompt_digest_exact": (
          history.get("prompt_sha256") == boundary.get("prompt_sha256")),
  }
  histories, boundaries = checkpoint_maps(history, boundary, args.step)
  candidate_boundaries = {}
  if candidate_boundary is not None:
    _, candidate_boundaries = checkpoint_maps(
        candidate_boundary, candidate_boundary, args.step)
    identity_checks.update({
        "candidate_generated_prefix_exact": (
            boundary.get("generated_token_ids", [])[:prefix_count] ==
            candidate_boundary.get("generated_token_ids", [])[:prefix_count]),
        "candidate_input_token_digest_exact": (
            boundary.get("input_token_ids_sha256") ==
            candidate_boundary.get("input_token_ids_sha256")),
        "candidate_prompt_digest_exact": (
            boundary.get("prompt_sha256") ==
            candidate_boundary.get("prompt_sha256")),
    })
  expected_history_keys = {
      (layer, role) for layer in args.layers for role in ("key", "value")}
  identity_checks["history_rows_complete"] = (
      expected_history_keys.issubset(histories))
  identity_checks["boundary_rows_complete"] = (
      set(args.layers).issubset(boundaries))
  if candidate_boundary is not None:
    identity_checks["candidate_boundary_rows_complete"] = (
        set(args.layers).issubset(candidate_boundaries))
  if not all(identity_checks.values()):
    raise SystemExit(f"input identity failure: {identity_checks}")

  logical_tokens = {
      int(row["logical_tokens"]) for row in histories.values()}
  if len(logical_tokens) != 1:
    raise SystemExit(f"history token counts disagree: {logical_tokens}")
  token_count = logical_tokens.pop()
  # Captured histories include the current decode token.  The adaptive kernel
  # only quantizes/attends the evicted prefix of the past-token history, so its
  # cold extent is one row shorter than ``logical_tokens - hot_tokens``.
  cold_tokens = token_count - args.hot_tokens - 1
  if not 0 < args.high_topk_per_query < cold_tokens:
    raise SystemExit("top-k must be below captured cold-token count")
  captured_chunk_count = math.ceil(
      cold_tokens / args.candidate_chunk_tokens)
  if (captured_chunk_count * args.local_topk_per_query <
      args.high_topk_per_query):
    raise SystemExit("local candidate pool cannot cover global top-k")

  rows = []
  union_counts = []
  selection_recalls = []
  aggregate = {
      "reference_sum_sq": 0.0,
      "stock_delta_sum_sq": 0.0,
      "block32_delta_sum_sq": 0.0,
      "corrected_delta_sum_sq": 0.0,
      "sidecar_corrected_delta_sum_sq": 0.0,
      "graph_base_delta_sum_sq": 0.0,
      "graph_corrected_delta_sum_sq": 0.0,
      "sparse_full_denominator_delta_sum_sq": 0.0,
      "sparse_renormalized_delta_sum_sq": 0.0,
      "observed_delta_sum_sq": 0.0,
      "observed_graph_delta_sum_sq": 0.0,
  }
  alignment_checks = []
  input_files = []
  key_exact_layers = set(args.key_exact_layers)
  value_exact_layers = set(args.value_exact_layers)
  key_residual_layers = set(args.key_residual_layers)
  value_residual_layers = set(args.value_residual_layers)
  for layer in args.layers:
    key_row = histories[(layer, "key")]
    value_row = histories[(layer, "value")]
    key_shape = tuple(int(value) for value in key_row["shape"])
    value_shape = tuple(int(value) for value in value_row["shape"])
    expected_shape = (1, KV_HEADS, token_count, HEAD_DIM)
    if key_shape != expected_shape or value_shape != expected_shape:
      raise SystemExit(
          f"layer {layer} history shape mismatch: {key_shape}/{value_shape}")
    key_path = resolve_artifact(key_row["file"])
    value_path = resolve_artifact(value_row["file"])
    input_files.extend(({
        "path": str(key_path.relative_to(ROOT)),
        "sha256": str(key_row["sha256"]),
    }, {
        "path": str(value_path.relative_to(ROOT)),
        "sha256": str(value_row["sha256"]),
    }))
    key = np.memmap(
        key_path, dtype=np.dtype(key_row["dtype"]), mode="r",
        shape=key_shape)[0]
    value = np.memmap(
        value_path, dtype=np.dtype(value_row["dtype"]), mode="r",
        shape=value_shape)[0]
    tensors = boundaries[layer]
    query = np.fromfile(resolve_artifact(tensors["query"]["file"]),
                        dtype="<f4").reshape(QUERY_HEADS, HEAD_DIM)
    current_key = np.fromfile(resolve_artifact(tensors["key"]["file"]),
                              dtype="<f4").reshape(QUERY_HEADS, HEAD_DIM)
    current_value = np.fromfile(resolve_artifact(tensors["value"]["file"]),
                                dtype="<f4").reshape(QUERY_HEADS, HEAD_DIM)
    stock_output = np.fromfile(
        resolve_artifact(tensors["attention"]["file"]),
        dtype="<f4").reshape(QUERY_HEADS, HEAD_DIM)
    for role, tensor in tensors.items():
      input_files.append({
          "path": str(resolve_artifact(tensor["file"]).relative_to(ROOT)),
          "sha256": str(tensor["sha256"]),
      })

    f16_exact = {
        "history_key": bool(np.array_equal(
            np.asarray(key), np.asarray(key).astype(np.float16).astype(
                np.float32))),
        "history_value": bool(np.array_equal(
            np.asarray(value), np.asarray(value).astype(np.float16).astype(
                np.float32))),
        "query": bool(np.array_equal(
            query, query.astype(np.float16).astype(np.float32))),
    }
    current_max_abs = {"key": 0.0, "value": 0.0}
    for kv_head in range(KV_HEADS):
      for query_head in range(
          kv_head * GQA_GROUP, (kv_head + 1) * GQA_GROUP):
        current_max_abs["key"] = max(
            current_max_abs["key"], float(np.max(np.abs(
                np.asarray(key[kv_head, -1], dtype=np.float32) -
                current_key[query_head]))))
        current_max_abs["value"] = max(
            current_max_abs["value"], float(np.max(np.abs(
                np.asarray(value[kv_head, -1], dtype=np.float32) -
                current_value[query_head]))))
    aligned = (
        all(f16_exact.values()) and current_max_abs["key"] == 0.0 and
        current_max_abs["value"] == 0.0)
    alignment_checks.append(aligned)

    exact_output = np.empty((QUERY_HEADS, HEAD_DIM), dtype=np.float32)
    block32_output = np.empty_like(exact_output)
    corrected_output = np.empty_like(exact_output)
    sidecar_corrected_output = np.empty_like(exact_output)
    graph_base_output = np.empty_like(exact_output)
    graph_corrected_output = np.empty_like(exact_output)
    sparse_full_denominator_output = np.empty_like(exact_output)
    sparse_renormalized_output = np.empty_like(exact_output)
    layer_selected_weight_mass = []
    observed_output = None
    if candidate_boundary is not None:
      observed_tensor = candidate_boundaries[layer]["attention"]
      observed_output = np.fromfile(
          resolve_artifact(observed_tensor["file"]),
          dtype="<f4").reshape(QUERY_HEADS, HEAD_DIM)
      input_files.append({
          "path": str(resolve_artifact(
              observed_tensor["file"]).relative_to(ROOT)),
          "sha256": str(observed_tensor["sha256"]),
      })
    layer_topk = (
        args.high_topk_per_query
        if layer in args.high_topk_layers else args.topk_per_query)
    layer_unions = []
    layer_recalls = []
    for kv_head in range(KV_HEADS):
      exact_key = np.asarray(key[kv_head], dtype=np.float32)
      exact_value = np.asarray(value[kv_head], dtype=np.float32)
      if layer in key_exact_layers:
        mixed_key = exact_key
      else:
        quant_key = dequantize_group(
            exact_key[:cold_tokens], np, key_quant_group,
            args.key_quant_bits,
            args.residual_bits if layer in key_residual_layers else 0)
        # Sink rows remain exact in the graph even after the logical cold
        # prefix grows past them. Model that mixed first block instead of
        # quantizing rows read from retained hot history.
        quant_key[:SINK_TOKENS] = exact_key[:SINK_TOKENS]
        mixed_key = np.concatenate(
            (quant_key, exact_key[cold_tokens:]), axis=0)
      if layer in value_exact_layers:
        mixed_value = exact_value
      else:
        quant_value = dequantize_group(
            exact_value[:cold_tokens], np, value_quant_group,
            args.value_quant_bits,
            args.residual_bits if layer in value_residual_layers else 0)
        quant_value[:SINK_TOKENS] = exact_value[:SINK_TOKENS]
        mixed_value = np.concatenate(
            (quant_value, exact_value[cold_tokens:]), axis=0)
      exact_scores = []
      approximate_scores = []
      for query_head in range(
          kv_head * GQA_GROUP, (kv_head + 1) * GQA_GROUP):
        exact_score = (
            exact_key @ query[query_head]) * np.float32(0.0625)
        approximate_score = (
            mixed_key @ query[query_head]) * np.float32(0.0625)
        exact_output[query_head], _ = softmax_output(
            exact_score, exact_value, np)
        block32_output[query_head], _ = softmax_output(
            approximate_score, mixed_value, np)
        exact_scores.append(exact_score)
        approximate_scores.append(approximate_score)
      selected = []
      for lane, approximate_score in enumerate(approximate_scores):
        selection_value = approximate_score
        if args.selection_rule == "oracle-impact":
          query_head = kv_head * GQA_GROUP + lane
          exact_score = exact_scores[lane]
          common_max = max(
              float(np.max(exact_score)), float(np.max(approximate_score)))
          exact_weight = np.exp(
              exact_score - np.float32(common_max), dtype=np.float32)
          approximate_weight = np.exp(
              approximate_score - np.float32(common_max), dtype=np.float32)
          # First-order normalized-output correction contribution.  This is
          # deliberately an oracle: it establishes whether a better selector
          # can help before designing a scale/query-only product proxy.
          residual = (
              exact_weight[:, None] *
                  (exact_value - exact_output[query_head]) -
              approximate_weight[:, None] *
                  (mixed_value - exact_output[query_head]))
          selection_value = np.linalg.norm(residual, axis=1)
          priority_max = float(np.max(selection_value))
          if priority_max != 0.0:
            selection_value = selection_value / np.float32(priority_max)
        stored_score = selection_value[:cold_tokens].astype(np.float16)
        local_candidates = []
        for begin in range(
            0, cold_tokens, args.candidate_chunk_tokens):
          end = min(begin + args.candidate_chunk_tokens, cold_tokens)
          local_count = min(args.local_topk_per_query, end - begin)
          local_indices = np.arange(begin, end, dtype=np.uint16)
          local_candidates.append(deterministic_topk(
              stored_score[begin:end], local_indices, local_count, np))
        candidate_pool = np.unique(np.concatenate(local_candidates))
        if len(candidate_pool) < layer_topk:
          raise RuntimeError(
              f"layer {layer} candidate pool below top-k: "
              f"{len(candidate_pool)} < {layer_topk}")
        selected_row = deterministic_topk(
            stored_score[candidate_pool], candidate_pool, layer_topk, np)
        selected.append(selected_row)
        exact_indices = np.arange(cold_tokens, dtype=np.uint16)
        exact_topk = deterministic_topk(
            selection_value[:cold_tokens], exact_indices, layer_topk, np)
        recall = len(np.intersect1d(
            selected_row, exact_topk, assume_unique=True)) / layer_topk
        layer_recalls.append(recall)
        selection_recalls.append(recall)
      union = np.unique(np.concatenate(selected))
      union = union[union >= SINK_TOKENS]
      layer_unions.append(int(len(union)))
      union_counts.append(int(len(union)))
      for lane, query_head in enumerate(range(
          kv_head * GQA_GROUP, (kv_head + 1) * GQA_GROUP)):
        corrected_score = approximate_scores[lane].copy()
        corrected_score[union] = exact_scores[lane][union]
        corrected, weights = softmax_output(
            corrected_score, mixed_value, np)
        corrected += (
            weights[union, None] *
            (exact_value[union] - mixed_value[union])).sum(
                axis=0, dtype=np.float32)
        corrected_output[query_head] = corrected
        selected_weights = weights[union]
        selected_weight_mass = float(np.sum(
            selected_weights, dtype=np.float32))
        layer_selected_weight_mass.append(selected_weight_mass)
        sparse_full_denominator = np.sum(
            selected_weights[:, None] * exact_value[union],
            axis=0, dtype=np.float32)
        sparse_full_denominator_output[query_head] = sparse_full_denominator
        sparse_renormalized_output[query_head] = (
            sparse_full_denominator / np.float32(selected_weight_mass)
            if selected_weight_mass > 0.0 else sparse_full_denominator)
        sidecar_scores = approximate_scores[lane].astype(
            np.float16).astype(np.float32)
        sidecar_corrected_output[query_head] = replacement_correction_output(
            approximate_scores[lane], sidecar_scores, exact_scores[lane],
            mixed_value, exact_value, union, np)
        graph_base, graph_corrected = adaptive_graph_outputs(
            approximate_scores[lane], exact_scores[lane],
            mixed_value, exact_value, union,
            256, np)
        graph_base_output[query_head] = graph_base
        graph_corrected_output[query_head] = graph_corrected

    exact64 = exact_output.astype(np.float64)
    aggregate["reference_sum_sq"] += float(np.sum(exact64 * exact64))
    aggregate["stock_delta_sum_sq"] += float(np.sum(
        (stock_output.astype(np.float64) - exact64) ** 2))
    aggregate["block32_delta_sum_sq"] += float(np.sum(
        (block32_output.astype(np.float64) - exact64) ** 2))
    aggregate["corrected_delta_sum_sq"] += float(np.sum(
        (corrected_output.astype(np.float64) - exact64) ** 2))
    aggregate["sidecar_corrected_delta_sum_sq"] += float(np.sum(
        (sidecar_corrected_output.astype(np.float64) - exact64) ** 2))
    aggregate["graph_base_delta_sum_sq"] += float(np.sum(
        (graph_base_output.astype(np.float64) - exact64) ** 2))
    aggregate["graph_corrected_delta_sum_sq"] += float(np.sum(
        (graph_corrected_output.astype(np.float64) - exact64) ** 2))
    aggregate["sparse_full_denominator_delta_sum_sq"] += float(np.sum(
        (sparse_full_denominator_output.astype(np.float64) - exact64) ** 2))
    aggregate["sparse_renormalized_delta_sum_sq"] += float(np.sum(
        (sparse_renormalized_output.astype(np.float64) - exact64) ** 2))
    if observed_output is not None:
      aggregate["observed_delta_sum_sq"] += float(np.sum(
          (observed_output.astype(np.float64) - exact64) ** 2))
      aggregate["observed_graph_delta_sum_sq"] += float(np.sum(
          (observed_output.astype(np.float64) -
           graph_corrected_output.astype(np.float64)) ** 2))
    row = {
        "alignment": {
            "current_history_max_abs": current_max_abs,
            "f16_exact": f16_exact,
            "pass": aligned,
        },
        "block32_vs_numpy_exact": metrics(
            exact_output, block32_output, np),
        "layer": layer,
        "selection_recall": {
            "min": min(layer_recalls),
            "median": float(np.median(layer_recalls)),
        },
        "selected_weight_mass": {
            "max": max(layer_selected_weight_mass),
            "median": float(np.median(layer_selected_weight_mass)),
            "min": min(layer_selected_weight_mass),
        },
        "topk_per_query": layer_topk,
        "f16_score_sidecar_correction_vs_numpy_exact": metrics(
            exact_output, sidecar_corrected_output, np),
        "f16_score_sidecar_correction_vs_stock": metrics(
            stock_output, sidecar_corrected_output, np),
        "f16_score_sidecar_vs_exact_score_correction": metrics(
            corrected_output, sidecar_corrected_output, np),
        "graph_arithmetic_base_vs_numpy_exact": metrics(
            exact_output, graph_base_output, np),
        "graph_arithmetic_topk_union_vs_numpy_exact": metrics(
            exact_output, graph_corrected_output, np),
        "graph_arithmetic_topk_union_vs_stock": metrics(
            stock_output, graph_corrected_output, np),
        "key_storage": (
            "dense_f16" if layer in key_exact_layers else
            f"group{key_quant_group}_i{args.key_quant_bits}" + (
                f"+r{args.residual_bits}"
                if layer in key_residual_layers else "")),
        "value_storage": (
            "dense_f16" if layer in value_exact_layers else
            f"group{value_quant_group}_i{args.value_quant_bits}" + (
                f"+r{args.residual_bits}"
                if layer in value_residual_layers else "")),
        "numpy_exact_vs_stock": metrics(exact_output, stock_output, np),
        "topk_union_fraction_by_kv_head": [
            count / cold_tokens for count in layer_unions],
        "topk_union_rows_by_kv_head": layer_unions,
        "topk_union_vs_numpy_exact": metrics(
            exact_output, corrected_output, np),
        "topk_union_vs_stock": metrics(
            stock_output, corrected_output, np),
        "topk_union_sparse_full_denominator_vs_numpy_exact": metrics(
            exact_output, sparse_full_denominator_output, np),
        "topk_union_sparse_renormalized_vs_numpy_exact": metrics(
            exact_output, sparse_renormalized_output, np),
    }
    if observed_output is not None:
      row.update({
          "observed_graph_vs_graph_arithmetic_topk_union": metrics(
              graph_corrected_output, observed_output, np),
          "observed_graph_vs_numpy_exact": metrics(
              exact_output, observed_output, np),
          "observed_graph_vs_stock": metrics(
              stock_output, observed_output, np),
      })
    rows.append(row)

  reference_sum_sq = aggregate["reference_sum_sq"]
  stock_relative_l2 = math.sqrt(
      aggregate["stock_delta_sum_sq"] / reference_sum_sq)
  block32_relative_l2 = math.sqrt(
      aggregate["block32_delta_sum_sq"] / reference_sum_sq)
  corrected_relative_l2 = math.sqrt(
      aggregate["corrected_delta_sum_sq"] / reference_sum_sq)
  sidecar_corrected_relative_l2 = math.sqrt(
      aggregate["sidecar_corrected_delta_sum_sq"] / reference_sum_sq)
  graph_base_relative_l2 = math.sqrt(
      aggregate["graph_base_delta_sum_sq"] / reference_sum_sq)
  graph_corrected_relative_l2 = math.sqrt(
      aggregate["graph_corrected_delta_sum_sq"] / reference_sum_sq)
  sparse_full_denominator_relative_l2 = math.sqrt(
      aggregate["sparse_full_denominator_delta_sum_sq"] / reference_sum_sq)
  sparse_renormalized_relative_l2 = math.sqrt(
      aggregate["sparse_renormalized_delta_sum_sq"] / reference_sum_sq)
  observed_relative_l2 = (
      math.sqrt(aggregate["observed_delta_sum_sq"] / reference_sum_sq)
      if candidate_boundary is not None else None)
  observed_graph_relative_l2 = (
      math.sqrt(
          aggregate["observed_graph_delta_sum_sq"] / reference_sum_sq)
      if candidate_boundary is not None else None)
  error_reduction = block32_relative_l2 / corrected_relative_l2
  max_layer_relative_l2 = max(
      row["topk_union_vs_numpy_exact"]["relative_l2"] for row in rows)

  increment_tokens = args.target_context - args.base_context
  elements_per_token = KV_HEADS * HEAD_DIM
  layer_count = len(args.layers)
  dense_tensor_bytes_per_layer = (
      increment_tokens * elements_per_token * 2)
  dense_bytes = dense_tensor_bytes_per_layer * 2 * layer_count
  compressed_key_bytes_per_layer = int(
      increment_tokens * elements_per_token *
      (args.key_quant_bits / 8.0 + 2.0 / key_quant_group))
  compressed_value_bytes_per_layer = int(
      increment_tokens * elements_per_token *
      (args.value_quant_bits / 8.0 + 2.0 / value_quant_group))
  residual_bytes_per_tensor_layer = int(
      increment_tokens * elements_per_token * args.residual_bits / 8.0)
  target_cold_tokens = args.target_context - args.hot_tokens
  topk_by_layer = {
      layer: (
          args.high_topk_per_query
          if layer in args.high_topk_layers else args.topk_per_query)
      for layer in args.layers}
  worst_union_rows_by_layer = {
      layer: min(GQA_GROUP * topk, target_cold_tokens)
      for layer, topk in topk_by_layer.items()}
  worst_union_fraction_by_layer = {
      layer: rows_per_head / target_cold_tokens
      for layer, rows_per_head in worst_union_rows_by_layer.items()}
  average_worst_union_fraction = sum(
      worst_union_fraction_by_layer.values()) / layer_count
  scan_bytes_by_layer = {
      layer: (
          (dense_tensor_bytes_per_layer
           if layer in key_exact_layers else
           compressed_key_bytes_per_layer + (
               residual_bytes_per_tensor_layer
               if layer in key_residual_layers else 0)) +
          (dense_tensor_bytes_per_layer
           if layer in value_exact_layers else
           compressed_value_bytes_per_layer + (
               residual_bytes_per_tensor_layer
               if layer in value_residual_layers else 0)))
      for layer in args.layers}
  codec_scan_bytes = sum(scan_bytes_by_layer.values())
  # The active four-stage graph is a partition rebuild, not the retired
  # delta-correction path: stage 1 reads K once and stage 3 reads V once.
  # Therefore it never rereads an approximate selected row.  Selection adds
  # only the retained dense K/V sources that differ from the scan source.
  approximate_correction_reread_bytes_by_layer = {
      layer: 0
      for layer in args.layers}
  exact_correction_bytes_by_layer = {
      layer: (
          dense_tensor_bytes_per_layer *
          ((0 if layer in key_exact_layers else 1) +
           (0 if layer in value_exact_layers else 1)) *
          worst_union_fraction_by_layer[layer])
      for layer in args.layers}
  approximate_correction_reread_bytes = sum(
      approximate_correction_reread_bytes_by_layer.values())
  exact_correction_bytes = sum(exact_correction_bytes_by_layer.values())
  target_chunk_count = math.ceil(
      target_cold_tokens / args.candidate_chunk_tokens)
  candidate_record_bytes = 4  # F16 score plus U16 within-context index.
  adaptive_layers = tuple(
      layer for layer in args.layers
      if not (layer in key_exact_layers and layer in value_exact_layers))
  candidate_workspace_write_bytes = (
      len(adaptive_layers) * KV_HEADS * GQA_GROUP * target_chunk_count *
      args.local_topk_per_query * candidate_record_bytes)
  candidate_workspace_read_write_bytes = (
      2 * candidate_workspace_write_bytes)
  total_bytes = (
      codec_scan_bytes + exact_correction_bytes +
      approximate_correction_reread_bytes +
      candidate_workspace_read_write_bytes)
  allowed_bytes = dense_bytes / args.required_long_context_speedup
  uniform_correction_bytes = sum(
      dense_tensor_bytes_per_layer *
      ((0 if layer in key_exact_layers else 1) +
       (0 if layer in value_exact_layers else 1))
      for layer in args.layers)
  correction_density_ceiling = (
      (allowed_bytes - codec_scan_bytes -
       candidate_workspace_read_write_bytes) /
      uniform_correction_bytes
      if uniform_correction_bytes else math.inf)
  traffic = {
      "allowed_bytes": allowed_bytes,
      "average_worst_case_union_fraction": average_worst_union_fraction,
      "approximate_correction_reread_bytes": (
          approximate_correction_reread_bytes),
      "approximate_correction_reread_bytes_by_layer": (
          approximate_correction_reread_bytes_by_layer),
      "block32_scan_bytes": codec_scan_bytes,
      "codec_scan_bytes": codec_scan_bytes,
      "codec_scan_bytes_by_layer": scan_bytes_by_layer,
      "compressed_key_scan_bytes": (
          compressed_key_bytes_per_layer *
          (layer_count - len(key_exact_layers))),
      "compressed_value_scan_bytes": (
          compressed_value_bytes_per_layer *
          (layer_count - len(value_exact_layers))),
      "candidate_workspace_read_write_bytes": (
          candidate_workspace_read_write_bytes),
      "candidate_workspace_layers": list(adaptive_layers),
      "compressed_correction_reread_bytes": (
          approximate_correction_reread_bytes),
      "correction_density_ceiling": correction_density_ceiling,
      "dense_value_scan_bytes": (
          dense_tensor_bytes_per_layer * len(value_exact_layers)),
      "dense_key_scan_bytes": (
          dense_tensor_bytes_per_layer * len(key_exact_layers)),
      "dense_f16_bytes": dense_bytes,
      "exact_correction_bytes": exact_correction_bytes,
      "exact_correction_bytes_by_layer": exact_correction_bytes_by_layer,
      "headroom_bytes_before_compute": allowed_bytes - total_bytes,
      "ideal_speedup": dense_bytes / total_bytes,
      "key_residual_scan_bytes": (
          residual_bytes_per_tensor_layer * len(key_residual_layers)),
      "pass": total_bytes <= allowed_bytes,
      "required_speedup": args.required_long_context_speedup,
      "residual_scan_bytes_per_tensor_layer": (
          residual_bytes_per_tensor_layer),
      "total_bytes_before_compute": total_bytes,
      "value_residual_scan_bytes": (
          residual_bytes_per_tensor_layer * len(value_residual_layers)),
      "worst_case_union_fraction_by_layer": (
          worst_union_fraction_by_layer),
      "worst_case_union_rows_per_kv_head_by_layer": (
          worst_union_rows_by_layer),
  }
  numeric = {
      "aggregate_block32_relative_l2": block32_relative_l2,
      "aggregate_codec_relative_l2": block32_relative_l2,
      "aggregate_stock_arithmetic_relative_l2": stock_relative_l2,
      "aggregate_topk_union_relative_l2": corrected_relative_l2,
      "aggregate_f16_score_sidecar_relative_l2": (
          sidecar_corrected_relative_l2),
      "aggregate_graph_arithmetic_base_relative_l2": (
          graph_base_relative_l2),
      "aggregate_graph_arithmetic_topk_union_relative_l2": (
          graph_corrected_relative_l2),
      "aggregate_topk_union_sparse_full_denominator_relative_l2": (
          sparse_full_denominator_relative_l2),
      "aggregate_topk_union_sparse_renormalized_relative_l2": (
          sparse_renormalized_relative_l2),
      "aggregate_observed_graph_relative_l2": observed_relative_l2,
      "aggregate_observed_vs_graph_arithmetic_relative_l2": (
          observed_graph_relative_l2),
      "error_reduction": error_reduction,
      "max_layer_topk_union_relative_l2": max_layer_relative_l2,
      "pass": (
          corrected_relative_l2 <= args.max_aggregate_relative_l2 and
          max_layer_relative_l2 <= args.max_layer_relative_l2 and
          error_reduction >= args.min_error_reduction),
      "thresholds": {
          "max_aggregate_relative_l2": args.max_aggregate_relative_l2,
          "max_layer_relative_l2": args.max_layer_relative_l2,
          "min_error_reduction": args.min_error_reduction,
      },
  }
  captured_union = {
      "max": max(union_counts),
      "max_fraction": max(union_counts) / cold_tokens,
      "median": float(np.median(union_counts)),
      "min": min(union_counts),
  }
  captured_selection_recall = {
      "mean": float(np.mean(selection_recalls)),
      "median": float(np.median(selection_recalls)),
      "min": min(selection_recalls),
  }
  checks = {
      "all_layer_alignment": all(alignment_checks),
      "input_identity": all(identity_checks.values()),
      "numeric_bound": numeric["pass"],
      "traffic_bound": traffic["pass"],
  }
  payload = {
      "all_required_checks_pass": all(checks.values()),
      "captured_selection_recall": captured_selection_recall,
      "captured_union_rows": captured_union,
      "checks": checks,
      "git": git_state(),
      "inputs": {
          "boundary_worker": str(args.boundary_worker),
          "candidate_boundary_worker": (
              str(args.candidate_boundary_worker)
              if args.candidate_boundary_worker is not None else None),
          "files": input_files,
          "history_worker": str(args.history_worker),
          "identity_checks": identity_checks,
      },
      "layers": rows,
      "memory": {
          "available_after_bytes": mem_available_bytes(),
          "available_before_bytes": started_available,
          "process_max_rss_bytes": (
              resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
      },
      "numeric": numeric,
      "rule": {
          "cold_codec": (
              "symmetric per-token per-KV-head grouped quantization with "
              "independent key/value payload widths and F16 scale groups"),
          "correction": (
              "retain local F16-score top-k candidates per cold chunk; select "
              "the layer capacity per query head, union within each KV head, "
              "then replace exact F16 K scores and V contributions"),
          "candidate_chunk_tokens": args.candidate_chunk_tokens,
          "high_topk_layers": list(args.high_topk_layers),
          "high_topk_per_query": args.high_topk_per_query,
          "hot_tokens": args.hot_tokens,
          "key_quant_group": key_quant_group,
          "key_quant_bits": args.key_quant_bits,
          "key_exact_layers": sorted(key_exact_layers),
          "key_residual_layers": sorted(key_residual_layers),
          "layers": list(args.layers),
          "local_topk_per_query": args.local_topk_per_query,
          "residual_bits": args.residual_bits,
          "residual_codec": (
              "packed fractional bit-planes sharing the base F16 group scale"),
          "selection_record": (
              "F16 priority plus U16 absolute cold-token index"),
          "selection_rule": args.selection_rule,
          "selection_tie_break": (
              "score descending, then absolute cold-token index ascending"),
          "step": args.step,
          "topk_per_query": args.topk_per_query,
          "value_quant_group": value_quant_group,
          "value_quant_bits": args.value_quant_bits,
          "value_exact_layers": sorted(value_exact_layers),
          "value_residual_layers": sorted(value_residual_layers),
      },
      "schema": "intel-qwen36-openvino-adaptive-attention-bound-v11",
      "traffic": traffic,
  }
  args.out_dir.mkdir(parents=True)
  (args.out_dir / "bound.json").write_text(
      json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  summary = [
      "# Adaptive exact grouped-quantized attention bound",
      "",
      f"- all required checks pass: `{payload['all_required_checks_pass']}`",
      f"- aggregate uncorrected codec relative L2: "
      f"`{block32_relative_l2:.12g}`",
      f"- aggregate corrected relative L2: `{corrected_relative_l2:.12g}`",
      f"- aggregate runtime-shaped corrected relative L2: "
      f"`{graph_corrected_relative_l2:.12g}`",
      f"- aggregate union-only V relative L2, full denominator / renormalized: "
      f"`{sparse_full_denominator_relative_l2:.12g} / "
      f"{sparse_renormalized_relative_l2:.12g}`",
      f"- error reduction: `{error_reduction:.6f}x`",
      f"- maximum layer corrected relative L2: "
      f"`{max_layer_relative_l2:.12g}`",
      f"- captured union rows min/median/max: "
      f"`{captured_union['min']}/{captured_union['median']}/"
      f"{captured_union['max']}`",
      f"- hierarchical selection recall min/median/mean: "
      f"`{captured_selection_recall['min']:.6f}/"
      f"{captured_selection_recall['median']:.6f}/"
      f"{captured_selection_recall['mean']:.6f}`",
      f"- average worst-case 64k union density: "
      f"`{average_worst_union_fraction:.6%}`",
      f"- ideal 64k incremental traffic speedup: "
      f"`{traffic['ideal_speedup']:.6f}x`",
      f"- traffic headroom before compute: "
      f"`{traffic['headroom_bytes_before_compute'] / 1048576:.3f} MiB`",
      "",
      "This is an offline real-boundary admission bound, not product "
      "correctness or a speed claim.",
  ]
  (args.out_dir / "summary.md").write_text(
      "\n".join(summary) + "\n", encoding="utf-8")
  print(json.dumps({
      "all_required_checks_pass": payload["all_required_checks_pass"],
      "event": "adaptive_attention_bound_complete",
      "out_dir": str(args.out_dir),
  }, sort_keys=True))
  return 0 if payload["all_required_checks_pass"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
