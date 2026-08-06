#!/usr/bin/env python3
"""Bound sink-plus-recent exact attention from a captured stock K/V state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
HEAD_DIM = 256
QUERY_HEADS = 16
KV_HEADS = 2
GQA_GROUP = QUERY_HEADS // KV_HEADS
ATTENTION_SCALE = 0.0625


def parse_csv_ints(value: str) -> tuple[int, ...]:
  parsed = tuple(int(item) for item in value.split(",") if item)
  if not parsed or len(parsed) != len(set(parsed)) or min(parsed) <= 0:
    raise argparse.ArgumentTypeError(
        "expected unique positive comma-separated integers")
  return parsed


def parser() -> argparse.ArgumentParser:
  result = argparse.ArgumentParser(description=__doc__)
  result.add_argument("--worker", type=Path, required=True)
  result.add_argument(
      "--boundary-worker", type=Path,
      help="matched worker carrying Q and stock attention; defaults to worker")
  result.add_argument("--layer", type=int, required=True)
  result.add_argument("--step", type=int, required=True)
  result.add_argument("--sink-tokens", type=int, default=4)
  result.add_argument(
      "--hot-tokens", type=parse_csv_ints,
      default=(4096, 8192, 12288, 16384, 24576, 32768, 49152))
  result.add_argument(
      "--page-tokens", type=parse_csv_ints, default=(256, 512, 1024),
      help="cold-page geometries for the exact-score shared-page oracle")
  result.add_argument(
      "--page-budgets", type=parse_csv_ints,
      default=(4096, 8192, 16384, 24576, 32768),
      help="maximum non-sink tokens retained by each page oracle")
  result.add_argument(
      "--page-samples", type=parse_csv_ints, default=(4, 8, 16, 32),
      help="evenly spaced K samples per page for implementable selectors")
  result.add_argument("--out", type=Path, required=True)
  return result


def resolve(path: str) -> Path:
  value = Path(path)
  return value if value.is_absolute() else ROOT / value


def metrics(reference: Any, candidate: Any, np: Any) -> dict[str, Any]:
  reference64 = reference.astype(np.float64)
  candidate64 = candidate.astype(np.float64)
  delta = candidate64 - reference64
  reference_norm = float(np.linalg.norm(reference64))
  candidate_norm = float(np.linalg.norm(candidate64))
  per_head_denominator = np.maximum(
      np.linalg.norm(reference64, axis=1), np.finfo(np.float64).tiny)
  per_head = np.linalg.norm(delta, axis=1) / per_head_denominator
  cosine_denominator = reference_norm * candidate_norm
  return {
      "cosine": (
          float(np.vdot(reference64, candidate64) / cosine_denominator)
          if cosine_denominator else 1.0),
      "max_abs": float(np.max(np.abs(delta))),
      "median_head_relative_l2": float(np.median(per_head)),
      "relative_l2": float(
          np.linalg.norm(delta) /
          max(reference_norm, np.finfo(np.float64).tiny)),
      "worst_head_relative_l2": float(np.max(per_head)),
  }


def main() -> int:
  args = parser().parse_args()
  if args.sink_tokens < 0:
    raise SystemExit("sink-tokens must be nonnegative")
  if args.out.exists():
    raise SystemExit(f"output already exists: {args.out}")

  import numpy as np

  worker_path = args.worker.resolve()
  worker = json.loads(worker_path.read_text(encoding="utf-8"))
  boundary_worker_path = (
      args.boundary_worker.resolve()
      if args.boundary_worker is not None else worker_path)
  boundary_worker = json.loads(
      boundary_worker_path.read_text(encoding="utf-8"))
  identity_checks = {
      "generated_prefix_exact": (
          worker.get("generated_token_ids", [])[:args.step + 1] ==
          boundary_worker.get("generated_token_ids", [])[:args.step + 1]),
      "input_token_digest_exact": (
          worker.get("input_token_ids_sha256") ==
          boundary_worker.get("input_token_ids_sha256")),
      "prompt_digest_exact": (
          worker.get("prompt_sha256") == boundary_worker.get("prompt_sha256")),
  }
  if not all(identity_checks.values()):
    raise SystemExit(f"worker identity mismatch: {identity_checks}")
  history_rows = {
      str(row["role"]): row
      for row in worker.get("attention_history_checkpoints", [])
      if int(row["layer"]) == args.layer and int(row["step"]) == args.step}
  if set(history_rows) != {"key", "value"}:
    raise SystemExit(
        f"missing exact K/V history for layer {args.layer}, step {args.step}")
  boundary_rows = [
      row for row in boundary_worker.get("attention_checkpoints", [])
      if int(row["layer"]) == args.layer and int(row["step"]) == args.step]
  if len(boundary_rows) != 1:
    raise SystemExit(
        f"expected one attention boundary for layer {args.layer}, "
        f"step {args.step}")
  tensors = boundary_rows[0]["tensors"]

  key_row = history_rows["key"]
  value_row = history_rows["value"]
  key_shape = tuple(int(value) for value in key_row["shape"])
  value_shape = tuple(int(value) for value in value_row["shape"])
  if key_shape != value_shape or len(key_shape) != 4:
    raise SystemExit(f"invalid K/V shapes: {key_shape}/{value_shape}")
  expected_prefix = (1, KV_HEADS)
  if key_shape[:2] != expected_prefix or key_shape[3] != HEAD_DIM:
    raise SystemExit(f"unexpected K/V shape: {key_shape}")
  token_count = key_shape[2]
  if args.sink_tokens >= token_count or max(args.hot_tokens) >= token_count:
    raise SystemExit("sink and hot windows must be smaller than the history")

  key = np.memmap(
      resolve(key_row["file"]), dtype=np.dtype(key_row["dtype"]), mode="r",
      shape=key_shape)[0]
  value = np.memmap(
      resolve(value_row["file"]), dtype=np.dtype(value_row["dtype"]),
      mode="r", shape=value_shape)[0]
  query = np.fromfile(
      resolve(tensors["query"]["file"]), dtype="<f4").reshape(
          QUERY_HEADS, HEAD_DIM)
  stock_output = np.fromfile(
      resolve(tensors["attention"]["file"]), dtype="<f4").reshape(
          QUERY_HEADS, HEAD_DIM)

  exact_output = np.empty_like(stock_output)
  full_weights = []
  for kv_head in range(KV_HEADS):
    key_rows = np.asarray(key[kv_head], dtype=np.float32)
    value_rows = np.asarray(value[kv_head], dtype=np.float32)
    for query_head in range(
        kv_head * GQA_GROUP, (kv_head + 1) * GQA_GROUP):
      scores = (key_rows @ query[query_head]) * np.float32(ATTENTION_SCALE)
      weights = np.exp(scores - np.max(scores), dtype=np.float32)
      weights /= np.sum(weights, dtype=np.float32)
      exact_output[query_head] = weights @ value_rows
      full_weights.append(weights)

  rows = []
  for hot_tokens in sorted(args.hot_tokens):
    recent_begin = token_count - hot_tokens
    retained_tokens = args.sink_tokens + hot_tokens
    candidate = np.empty_like(exact_output)
    dropped_masses = []
    retained_masses = []
    for query_head, weights in enumerate(full_weights):
      kv_head = query_head // GQA_GROUP
      value_rows = np.asarray(value[kv_head], dtype=np.float32)
      sink_mass = np.sum(weights[:args.sink_tokens], dtype=np.float32)
      recent_mass = np.sum(weights[recent_begin:], dtype=np.float32)
      retained_mass = np.float32(sink_mass + recent_mass)
      numerator = (
          weights[:args.sink_tokens] @ value_rows[:args.sink_tokens] +
          weights[recent_begin:] @ value_rows[recent_begin:])
      candidate[query_head] = numerator / retained_mass
      retained_masses.append(float(retained_mass))
      dropped_masses.append(float(np.float32(1.0) - retained_mass))
    rows.append({
        "dropped_softmax_mass": {
            "max": max(dropped_masses),
            "median": float(np.median(dropped_masses)),
            "min": min(dropped_masses),
        },
        "hot_tokens": hot_tokens,
        "kv_traffic_fraction": retained_tokens / token_count,
        "kv_traffic_reduction_fraction": 1.0 - retained_tokens / token_count,
        "recent_begin": recent_begin,
        "retained_softmax_mass": {
            "max": max(retained_masses),
            "median": float(np.median(retained_masses)),
            "min": min(retained_masses),
        },
        "retained_tokens": retained_tokens,
        "vs_numpy_exact": metrics(exact_output, candidate, np),
        "vs_stock_boundary": metrics(stock_output, candidate, np),
    })

  oracle_page_rows = []
  sampled_page_rows = []
  cold_begin = args.sink_tokens
  for page_tokens in sorted(args.page_tokens):
    page_ranges = [
        (begin, min(begin + page_tokens, token_count))
        for begin in range(cold_begin, token_count, page_tokens)]
    for page_budget in sorted(args.page_budgets):
      if page_budget % page_tokens:
        continue
      pages_to_keep = min(page_budget // page_tokens, len(page_ranges))
      candidate = np.empty_like(exact_output)
      dropped_masses = []
      retained_masses = []
      retained_tokens_by_kv = []
      selected_pages_by_kv = []
      for kv_head in range(KV_HEADS):
        query_heads = range(
            kv_head * GQA_GROUP, (kv_head + 1) * GQA_GROUP)
        page_mass = np.asarray([
            [
                np.sum(full_weights[query_head][begin:end], dtype=np.float32)
                for begin, end in page_ranges]
            for query_head in query_heads], dtype=np.float32)
        # This is deliberately an oracle upper bound.  Exact per-head mass is
        # reduced with max so one shared page list can feed all eight GQA
        # query heads without multiplying K/V traffic by eight.
        priority = np.max(page_mass, axis=0)
        page_indices = np.arange(len(page_ranges), dtype=np.int32)
        selected_pages = np.lexsort((page_indices, -priority))[:pages_to_keep]
        selected_pages.sort()
        selected_pages_by_kv.append([int(value) for value in selected_pages])
        selected_positions = np.concatenate([
            np.arange(page_ranges[index][0], page_ranges[index][1])
            for index in selected_pages])
        retained_tokens_by_kv.append(
            args.sink_tokens + int(selected_positions.size))
        value_rows = np.asarray(value[kv_head], dtype=np.float32)
        selected_values = value_rows[selected_positions]
        for query_head in query_heads:
          weights = full_weights[query_head]
          sink_mass = np.sum(
              weights[:args.sink_tokens], dtype=np.float32)
          selected_weights = weights[selected_positions]
          retained_mass = np.float32(
              sink_mass + np.sum(selected_weights, dtype=np.float32))
          numerator = (
              weights[:args.sink_tokens] @ value_rows[:args.sink_tokens] +
              selected_weights @ selected_values)
          candidate[query_head] = numerator / retained_mass
          retained_masses.append(float(retained_mass))
          dropped_masses.append(float(np.float32(1.0) - retained_mass))
      mean_retained_tokens = float(np.mean(retained_tokens_by_kv))
      oracle_page_rows.append({
          "dropped_softmax_mass": {
              "max": max(dropped_masses),
              "median": float(np.median(dropped_masses)),
              "min": min(dropped_masses),
          },
          "kv_traffic_fraction": mean_retained_tokens / token_count,
          "kv_traffic_reduction_fraction": (
              1.0 - mean_retained_tokens / token_count),
          "page_budget": page_budget,
          "page_tokens": page_tokens,
          "pages_to_keep": pages_to_keep,
          "retained_softmax_mass": {
              "max": max(retained_masses),
              "median": float(np.median(retained_masses)),
              "min": min(retained_masses),
          },
          "retained_tokens_by_kv": retained_tokens_by_kv,
          "selected_pages_by_kv": selected_pages_by_kv,
          "selector": "exact_per_head_mass_shared_max_oracle",
          "vs_numpy_exact": metrics(exact_output, candidate, np),
          "vs_stock_boundary": metrics(stock_output, candidate, np),
      })

      oracle_page_sets = [set(values) for values in selected_pages_by_kv]
      for page_samples in sorted(args.page_samples):
        if page_samples > page_tokens:
          continue
        sampled_positions_by_page = []
        for begin, end in page_ranges:
          length = end - begin
          count = min(page_samples, length)
          sampled_positions_by_page.append(
              begin + ((2 * np.arange(count) + 1) * length // (2 * count)))
        sampled_tokens_per_kv = sum(
            len(values) for values in sampled_positions_by_page)
        candidate = np.empty_like(exact_output)
        dropped_masses = []
        retained_masses = []
        retained_tokens_by_kv = []
        page_recalls = []
        for kv_head in range(KV_HEADS):
          query_heads = range(
              kv_head * GQA_GROUP, (kv_head + 1) * GQA_GROUP)
          estimated_page_mass = np.asarray([
              [
                  float(np.mean(full_weights[query_head][positions])) *
                  (page_ranges[index][1] - page_ranges[index][0])
                  for index, positions in enumerate(
                      sampled_positions_by_page)]
              for query_head in query_heads], dtype=np.float32)
          priority = np.max(estimated_page_mass, axis=0)
          page_indices = np.arange(len(page_ranges), dtype=np.int32)
          selected_pages = np.lexsort(
              (page_indices, -priority))[:pages_to_keep]
          selected_pages.sort()
          selected_set = set(int(value) for value in selected_pages)
          page_recalls.append(
              len(selected_set & oracle_page_sets[kv_head]) /
              max(1, pages_to_keep))
          selected_positions = np.concatenate([
              np.arange(page_ranges[index][0], page_ranges[index][1])
              for index in selected_pages])
          retained_tokens_by_kv.append(
              args.sink_tokens + int(selected_positions.size))
          value_rows = np.asarray(value[kv_head], dtype=np.float32)
          selected_values = value_rows[selected_positions]
          for query_head in query_heads:
            weights = full_weights[query_head]
            sink_mass = np.sum(
                weights[:args.sink_tokens], dtype=np.float32)
            selected_weights = weights[selected_positions]
            retained_mass = np.float32(
                sink_mass + np.sum(selected_weights, dtype=np.float32))
            numerator = (
                weights[:args.sink_tokens] @ value_rows[:args.sink_tokens] +
                selected_weights @ selected_values)
            candidate[query_head] = numerator / retained_mass
            retained_masses.append(float(retained_mass))
            dropped_masses.append(float(np.float32(1.0) - retained_mass))
        mean_retained_tokens = float(np.mean(retained_tokens_by_kv))
        # The selector reads only sampled K rows.  The exact phase then reads
        # both K and V for retained pages; count the two passes separately.
        estimated_traffic_fraction = (
            2.0 * mean_retained_tokens + sampled_tokens_per_kv) / (
                2.0 * token_count)
        sampled_page_rows.append({
            "dropped_softmax_mass": {
                "max": max(dropped_masses),
                "median": float(np.median(dropped_masses)),
                "min": min(dropped_masses),
            },
            "estimated_total_kv_traffic_fraction": (
                estimated_traffic_fraction),
            "estimated_total_kv_traffic_reduction_fraction": (
                1.0 - estimated_traffic_fraction),
            "oracle_page_recall": {
                "max": max(page_recalls),
                "mean": float(np.mean(page_recalls)),
                "min": min(page_recalls),
            },
            "page_budget": page_budget,
            "page_samples": page_samples,
            "page_tokens": page_tokens,
            "pages_to_keep": pages_to_keep,
            "retained_softmax_mass": {
                "max": max(retained_masses),
                "median": float(np.median(retained_masses)),
                "min": min(retained_masses),
            },
            "retained_tokens_by_kv": retained_tokens_by_kv,
            "sampled_key_tokens_per_kv": sampled_tokens_per_kv,
            "selector": "even_k_samples_page_mass_shared_max",
            "vs_numpy_exact": metrics(exact_output, candidate, np),
            "vs_stock_boundary": metrics(stock_output, candidate, np),
        })

  result = {
      "attention_scale": ATTENTION_SCALE,
      "baseline": {
          "numpy_exact_vs_stock_boundary": metrics(
              stock_output, exact_output, np),
      },
      "input": {
          "history": {
              role: {
                  "file": row["file"],
                  "sha256": row["sha256"],
              }
              for role, row in sorted(history_rows.items())
          },
          "stock_attention": {
              "file": tensors["attention"]["file"],
              "sha256": tensors["attention"]["sha256"],
          },
          "stock_query": {
              "file": tensors["query"]["file"],
              "sha256": tensors["query"]["sha256"],
          },
          "worker": str(worker_path),
          "boundary_worker": str(boundary_worker_path),
      },
      "identity_checks": identity_checks,
      "layer": args.layer,
      "oracle_page_rows": oracle_page_rows,
      "rows": rows,
      "sampled_page_rows": sampled_page_rows,
      "schema_version": "intel-qwen36-openvino-hot-only-attention-bound-v1",
      "sink_tokens": args.sink_tokens,
      "step": args.step,
      "token_count": token_count,
  }
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(
      json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  print(json.dumps({
      "event": "hot_only_attention_bound_complete",
      "out": str(args.out),
      "rows": len(rows),
  }, sort_keys=True))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
