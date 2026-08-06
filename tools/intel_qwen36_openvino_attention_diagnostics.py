"""Artifact comparisons for OpenVINO attention product diagnostics."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any


def _path(root: Path, value: str) -> Path:
  path = Path(value)
  return path if path.is_absolute() else root / path


def _logsumexp(values: Any, np: Any) -> float:
  maximum = float(np.max(values))
  return maximum + math.log(float(np.exp(values - maximum).sum()))


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def capture_attention_history_checkpoint(
    *, step: int | None, selected_steps: set[int], layers: tuple[int, ...],
    mode: str, selected_path: str, request: Any, graph: Any, raw: Path,
    root: Path, prompt_tokens: int,
) -> list[dict[str, Any]]:
  import numpy as np

  if step is None or step not in selected_steps:
    return []
  custom_history = mode == "candidate" and selected_path == "hot_cold_custom"
  wanted = {}
  for layer in layers:
    names = (
        graph.layer_state_names(layer)[:2]
        if custom_history else graph.stock_state_names(layer))
    for role, name in zip(("key", "value"), names):
      wanted[name] = (int(layer), role)
  rows = []
  captured = set()
  for state in request.query_state():
    selected = wanted.get(str(state.name))
    if selected is None:
      continue
    layer, role = selected
    value = np.array(state.state.data, copy=True)
    path = raw / f"step{step:04d}-history-{role}-layer{layer}-{mode}.bin"
    value.tofile(path)
    storage = (
        "custom_hot_packed_key" if custom_history and role == "key" else
        "custom_hot_direct_value" if custom_history else "stock_dense")
    rows.append({
        "byte_count": path.stat().st_size,
        "dtype": value.dtype.str,
        "file": str(path.resolve().relative_to(root.resolve())),
        "finite": bool(
            np.isfinite(value).all()
            if np.issubdtype(value.dtype, np.floating) else True),
        "layer": layer,
        "logical_tokens": prompt_tokens + step,
        "name": str(state.name),
        "role": role,
        "sha256": _sha256(path),
        "shape": list(value.shape),
        "step": step,
        "storage": storage,
    })
    captured.add(str(state.name))
    del value
  missing = sorted(set(wanted) - captured)
  if missing:
    raise RuntimeError(
        f"attention history states missing at step {step}: {missing}")
  return rows


def distribution_rows(
    stock: dict[str, Any], candidate: dict[str, Any], root: Path,
) -> list[dict[str, Any]]:
  import numpy as np

  candidate_by_step = {
      int(row["step"]): row for row in candidate["distribution_checkpoints"]}
  rows = []
  for stock_row in stock["distribution_checkpoints"]:
    step = int(stock_row["step"])
    candidate_row = candidate_by_step.get(step)
    if candidate_row is None:
      continue
    stock_logits = np.fromfile(_path(root, stock_row["file"]), dtype="<f4")
    candidate_logits = np.fromfile(
        _path(root, candidate_row["file"]), dtype="<f4")
    same_shape = stock_logits.shape == candidate_logits.shape
    is_finite = bool(
        same_shape and np.isfinite(stock_logits).all()
        and np.isfinite(candidate_logits).all())
    if is_finite:
      stock64 = stock_logits.astype(np.float64)
      candidate64 = candidate_logits.astype(np.float64)
      stock_log_z = _logsumexp(stock64, np)
      candidate_log_z = _logsumexp(candidate64, np)
      stock_log_p = stock64 - stock_log_z
      candidate_log_p = candidate64 - candidate_log_z
      stock_p = np.exp(stock_log_p)
      kld = float(np.sum(stock_p * (stock_log_p - candidate_log_p)))
      denominator = float(np.linalg.norm(stock64) * np.linalg.norm(candidate64))
      cosine = (
          float(np.dot(stock64, candidate64) / denominator)
          if denominator else 1.0)
      stock_top1 = int(np.argmax(stock_logits))
      candidate_top1 = int(np.argmax(candidate_logits))
    else:
      kld = cosine = None
      stock_top1 = candidate_top1 = None
    rows.append({
        "candidate_top1": candidate_top1,
        "cosine": cosine,
        "finite": is_finite,
        "kld_stock_to_candidate": kld,
        "same_shape": same_shape,
        "step": step,
        "stock_top1": stock_top1,
        "top1_match": stock_top1 == candidate_top1,
    })
  return rows


def lm_head_hidden_rows(
    stock: dict[str, Any], candidate: dict[str, Any], root: Path,
) -> list[dict[str, Any]]:
  """Compare captured last-query LM-head inputs at matched decode steps."""
  import numpy as np

  candidate_by_step = {
      int(row["step"]): row
      for row in candidate.get("lm_head_hidden_checkpoints", [])}
  rows = []
  for stock_row in stock.get("lm_head_hidden_checkpoints", []):
    step = int(stock_row["step"])
    candidate_row = candidate_by_step.get(step)
    if candidate_row is None:
      continue
    stock_values = np.fromfile(
        _path(root, stock_row["file"]), dtype="<f4").astype(np.float64)
    candidate_values = np.fromfile(
        _path(root, candidate_row["file"]), dtype="<f4").astype(np.float64)
    stock_shape = tuple(int(value) for value in stock_row.get("shape", []))
    candidate_shape = tuple(
        int(value) for value in candidate_row.get("shape", []))
    same_shape = (
        stock_shape == candidate_shape and
        stock_values.shape == candidate_values.shape)
    is_finite = bool(
        same_shape and np.isfinite(stock_values).all() and
        np.isfinite(candidate_values).all())
    if is_finite:
      delta = candidate_values - stock_values
      stock_norm = float(np.linalg.norm(stock_values))
      candidate_norm = float(np.linalg.norm(candidate_values))
      denominator = stock_norm * candidate_norm
      cosine = (
          float(np.dot(stock_values, candidate_values) / denominator)
          if denominator else 1.0)
      max_abs = float(np.max(np.abs(delta))) if delta.size else 0.0
      relative_l2 = float(
          np.linalg.norm(delta) /
          max(stock_norm, np.finfo(np.float64).tiny))
      exact = bool(np.array_equal(stock_values, candidate_values))
    else:
      cosine = max_abs = relative_l2 = None
      exact = False
    rows.append({
        "candidate_sha256": candidate_row.get("sha256"),
        "cosine": cosine,
        "exact": exact,
        "finite": is_finite,
        "max_abs": max_abs,
        "relative_l2": relative_l2,
        "same_shape": same_shape,
        "step": step,
        "stock_sha256": stock_row.get("sha256"),
    })
  return rows


def attention_boundary_rows(
    stock: dict[str, Any], candidate: dict[str, Any], root: Path, graph: Any,
) -> list[dict[str, Any]]:
  import numpy as np

  candidate_by_key = {
      (int(row["step"]), int(row["layer"])): row
      for row in candidate.get("attention_checkpoints", [])}
  rows = []
  for stock_row in stock.get("attention_checkpoints", []):
    step = int(stock_row["step"])
    layer = int(stock_row["layer"])
    candidate_row = candidate_by_key.get((step, layer))
    if candidate_row is None:
      continue
    for role, stock_tensor in stock_row["tensors"].items():
      candidate_tensor = candidate_row["tensors"].get(role)
      if candidate_tensor is None:
        continue
      stock_values = np.fromfile(
          _path(root, stock_tensor["file"]), dtype="<f4").astype(np.float64)
      candidate_values = np.fromfile(
          _path(root, candidate_tensor["file"]),
          dtype="<f4").astype(np.float64)
      candidate_gqa_expanded = False
      stock_shape = tuple(int(value) for value in stock_tensor.get("shape", []))
      candidate_shape = tuple(
          int(value) for value in candidate_tensor.get("shape", []))
      if (role in ("key", "value") and len(stock_shape) == 3 and
          len(candidate_shape) == 3 and stock_shape[0] == candidate_shape[0] and
          stock_shape[2] == candidate_shape[2] and
          stock_shape[1] == candidate_shape[1] * graph.GQA_GROUP):
        candidate_values = np.repeat(
            candidate_values.reshape(candidate_shape),
            graph.GQA_GROUP, axis=1).reshape(-1)
        candidate_shape = stock_shape
        candidate_gqa_expanded = True
      same_shape = (
          stock_shape == candidate_shape and
          stock_values.shape == candidate_values.shape)
      is_finite = bool(
          same_shape and np.isfinite(stock_values).all() and
          np.isfinite(candidate_values).all())
      if is_finite:
        delta = candidate_values - stock_values
        stock_norm = float(np.linalg.norm(stock_values))
        candidate_norm = float(np.linalg.norm(candidate_values))
        denominator = stock_norm * candidate_norm
        cosine = (
            float(np.dot(stock_values, candidate_values) / denominator)
            if denominator else 1.0)
        max_abs = float(np.max(np.abs(delta))) if delta.size else 0.0
        relative_l2 = float(
            np.linalg.norm(delta) / max(stock_norm, np.finfo(np.float64).tiny))
      else:
        cosine = max_abs = relative_l2 = None
      rows.append({
          "candidate_sha256": candidate_tensor.get("sha256"),
          "candidate_gqa_expanded": candidate_gqa_expanded,
          "cosine": cosine,
          "exact": bool(
              is_finite and np.array_equal(stock_values, candidate_values)),
          "finite": is_finite,
          "layer": layer,
          "max_abs": max_abs,
          "relative_l2": relative_l2,
          "role": role,
          "same_shape": same_shape,
          "step": step,
          "stock_sha256": stock_tensor.get("sha256"),
      })
  return rows


def attention_history_rows(
    stock: dict[str, Any], candidate: dict[str, Any], root: Path, graph: Any,
) -> list[dict[str, Any]]:
  """Compare stock dense K/V state with the custom exact-history ring."""
  import numpy as np

  candidate_by_key = {
      (int(row["step"]), int(row["layer"]), str(row["role"])): row
      for row in candidate.get("attention_history_checkpoints", [])}
  source = candidate.get("source_summary") or {}
  capacity_by_layer = source.get("physical_ring_capacity_by_layer") or {}
  hot_key_shape_by_layer = source.get("hot_key_shape_by_layer") or {}
  rows = []
  for stock_row in stock.get("attention_history_checkpoints", []):
    step = int(stock_row["step"])
    layer = int(stock_row["layer"])
    role = str(stock_row["role"])
    candidate_row = candidate_by_key.get((step, layer, role))
    if candidate_row is None:
      continue
    stock_shape = tuple(int(value) for value in stock_row.get("shape", []))
    candidate_shape = tuple(
        int(value) for value in candidate_row.get("shape", []))
    logical_tokens = int(stock_row.get("logical_tokens", -1))
    ring_capacity = int(capacity_by_layer.get(str(layer), 0))
    recorded_hot_key_shape = tuple(
        int(value)
        for value in hot_key_shape_by_layer.get(str(layer), []))
    storage = str(candidate_row.get("storage", ""))
    same_logical_shape = (
        logical_tokens == int(candidate_row.get("logical_tokens", -2)) and
        len(stock_shape) == 4 and stock_shape[0] == 1 and
        stock_shape[1] == graph.KV_HEADS and
        stock_shape[2] == logical_tokens and
        stock_shape[3] == graph.HEAD_DIM and
        ring_capacity >= logical_tokens - graph.SINK_TOKENS and
        ((storage == "custom_hot_packed_key" and role == "key" and
          recorded_hot_key_shape and
          candidate_shape == recorded_hot_key_shape and
          candidate_shape[0] == 1 and
          candidate_shape[1] == graph.KV_HEADS and
          candidate_shape[2] >=
              math.ceil((ring_capacity + graph.SINK_TOKENS) /
                        graph.KEY_TILE_TOKENS) and
          candidate_shape[3] == graph.HOT_KEY_WORDS_PER_BLOCK) or
         (storage == "custom_hot_direct_value" and role == "value" and
          candidate_shape == (
              1, graph.KV_HEADS, ring_capacity + graph.SINK_TOKENS,
              graph.HEAD_DIM))))
    row = {
        "candidate_sha256": candidate_row.get("sha256"),
        "candidate_storage": storage,
        "exact": False,
        "exact_after_stock_f16_round": False,
        "finite": False,
        "layer": layer,
        "logical_tokens": logical_tokens,
        "role": role,
        "same_logical_shape": same_logical_shape,
        "step": step,
        "stock_sha256": stock_row.get("sha256"),
    }
    if not same_logical_shape:
      rows.append(row)
      continue
    stock_values = np.memmap(
        _path(root, stock_row["file"]), dtype=np.dtype(stock_row["dtype"]),
        mode="r", shape=stock_shape)
    candidate_values = np.memmap(
        _path(root, candidate_row["file"]),
        dtype=np.dtype(candidate_row["dtype"]), mode="r",
        shape=candidate_shape)
    element_count = mismatch_count = f16_bit_mismatch_count = 0
    stock_f16_round_mismatch_count = 0
    sum_sq_delta = sum_sq_stock = sum_sq_candidate = dot = 0.0
    max_abs = stock_f16_round_max_abs = 0.0
    first_mismatch = first_f16_bit_mismatch = max_abs_location = None
    is_finite = True
    for begin in range(0, logical_tokens, 1024):
      end = min(logical_tokens, begin + 1024)
      stock_chunk = np.asarray(
          stock_values[0, :, begin:end, :], dtype=np.float32)
      if role == "key":
        block_begin = begin // graph.KEY_TILE_TOKENS
        block_end = math.ceil(end / graph.KEY_TILE_TOKENS)
        packed = np.ascontiguousarray(
            candidate_values[0, :, block_begin:block_end, :])
        decoded = packed.view(np.float16).reshape(
            graph.KV_HEADS, block_end - block_begin, graph.HEAD_DIM // 2,
            graph.KEY_TILE_TOKENS, 2)
        decoded = decoded.transpose(0, 1, 3, 2, 4).reshape(
            graph.KV_HEADS,
            (block_end - block_begin) * graph.KEY_TILE_TOKENS,
            graph.HEAD_DIM)
        offset = begin - block_begin * graph.KEY_TILE_TOKENS
        candidate_half = np.ascontiguousarray(
            decoded[:, offset:offset + end - begin, :])
      else:
        candidate_half = np.ascontiguousarray(
            candidate_values[0, :, begin:end, :], dtype=np.float16)
      candidate_chunk = candidate_half.astype(np.float32)
      if (not np.isfinite(stock_chunk).all() or
          not np.isfinite(candidate_chunk).all()):
        is_finite = False
        break
      rounded_stock = np.ascontiguousarray(stock_chunk.astype(np.float16))
      f16_mismatch = (
          rounded_stock.view(np.uint16) != candidate_half.view(np.uint16))
      numeric_mismatch = stock_chunk != candidate_chunk
      stock_round_trip = rounded_stock.astype(np.float32)
      stock_round_mismatch = stock_chunk != stock_round_trip
      if first_mismatch is None and np.any(numeric_mismatch):
        index = np.argwhere(numeric_mismatch)[0]
        first_mismatch = {
            "dim": int(index[2]), "head": int(index[0]),
            "token": begin + int(index[1]),
        }
      if first_f16_bit_mismatch is None and np.any(f16_mismatch):
        index = np.argwhere(f16_mismatch)[0]
        first_f16_bit_mismatch = {
            "dim": int(index[2]), "head": int(index[0]),
            "token": begin + int(index[1]),
        }
      delta = candidate_chunk.astype(np.float64) - stock_chunk
      absolute = np.abs(delta)
      chunk_max = float(np.max(absolute)) if absolute.size else 0.0
      if chunk_max > max_abs:
        index = np.unravel_index(int(np.argmax(absolute)), absolute.shape)
        max_abs = chunk_max
        max_abs_location = {
            "dim": int(index[2]), "head": int(index[0]),
            "token": begin + int(index[1]),
        }
      round_absolute = np.abs(stock_round_trip - stock_chunk)
      stock_f16_round_max_abs = max(
          stock_f16_round_max_abs,
          float(np.max(round_absolute)) if round_absolute.size else 0.0)
      stock64 = stock_chunk.astype(np.float64)
      candidate64 = candidate_chunk.astype(np.float64)
      element_count += int(stock_chunk.size)
      mismatch_count += int(np.count_nonzero(numeric_mismatch))
      f16_bit_mismatch_count += int(np.count_nonzero(f16_mismatch))
      stock_f16_round_mismatch_count += int(
          np.count_nonzero(stock_round_mismatch))
      sum_sq_delta += float(np.sum(delta * delta))
      sum_sq_stock += float(np.sum(stock64 * stock64))
      sum_sq_candidate += float(np.sum(candidate64 * candidate64))
      dot += float(np.sum(stock64 * candidate64))
    if is_finite:
      denominator = math.sqrt(sum_sq_stock * sum_sq_candidate)
      row.update({
          "cosine": dot / denominator if denominator else 1.0,
          "element_count": element_count,
          "exact": mismatch_count == 0,
          "exact_after_stock_f16_round": f16_bit_mismatch_count == 0,
          "f16_bit_mismatch_count": f16_bit_mismatch_count,
          "finite": True,
          "first_f16_bit_mismatch": first_f16_bit_mismatch,
          "first_mismatch": first_mismatch,
          "max_abs": max_abs,
          "max_abs_location": max_abs_location,
          "mismatch_count": mismatch_count,
          "relative_l2": (
              math.sqrt(sum_sq_delta) /
              max(math.sqrt(sum_sq_stock), np.finfo(np.float64).tiny)),
          "rmse": math.sqrt(sum_sq_delta / max(element_count, 1)),
          "stock_f16_round_max_abs": stock_f16_round_max_abs,
          "stock_f16_round_mismatch_count":
              stock_f16_round_mismatch_count,
      })
    rows.append(row)
  return rows
