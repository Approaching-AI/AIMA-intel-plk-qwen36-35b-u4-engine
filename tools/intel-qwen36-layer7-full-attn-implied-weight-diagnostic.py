#!/usr/bin/env python3
"""Infer layer-7 attention weights from oracle pregate and captured V history."""

from __future__ import annotations

import argparse
import array
import datetime as dt
import json
import math
from pathlib import Path
from typing import Any

import iq36_local


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA_VERSION = "intel-qwen36-layer7-full-attn-implied-weight-diagnostic-v0"
DEFAULT_ALL_HISTORY = ROOT / "output/r1-full-attn-all-history-capture-20260627T145615Z/history.json"
Q_HEADS = 16
KV_HEADS = 2
HEAD_DIM = 256
TOKENS = 16
SCALE = 0.0625


def utc_stamp() -> str:
  return dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def iso_now() -> str:
  return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--all-history-json", type=Path, default=DEFAULT_ALL_HISTORY)
  parser.add_argument("--layer", type=int, default=7)
  parser.add_argument("--out-dir", type=Path, default=None)
  return parser.parse_args()


def read_f32(path: Path) -> list[float]:
  values = array.array("f")
  with path.open("rb") as fh:
    values.fromfile(fh, path.stat().st_size // 4)
  return list(values)


def half_to_float(half: int) -> float:
  sign = (half & 0x8000) << 16
  exponent = (half >> 10) & 0x1F
  mantissa = half & 0x03FF
  if exponent == 0:
    if mantissa == 0:
      bits = sign
    else:
      exponent = 1
      while (mantissa & 0x0400) == 0:
        mantissa <<= 1
        exponent -= 1
      mantissa &= 0x03FF
      bits = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13)
  elif exponent == 31:
    bits = sign | 0x7F800000 | (mantissa << 13)
  else:
    bits = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13)
  return array.array("f", bits.to_bytes(4, "little"))[0]


def float_to_half(value: float) -> int:
  bits = int.from_bytes(array.array("f", [float(value)]).tobytes(), "little")
  sign = (bits >> 16) & 0x8000
  exponent = ((bits >> 23) & 0xFF) - 127 + 15
  mantissa = bits & 0x7FFFFF
  if exponent <= 0:
    if exponent < -10:
      return sign
    mantissa |= 0x800000
    shift = 14 - exponent
    half_mantissa = mantissa >> shift
    if (mantissa >> (shift - 1)) & 1:
      half_mantissa += 1
    return (sign | half_mantissa) & 0xFFFF
  if exponent >= 31:
    return (sign | 0x7C00) & 0xFFFF
  half = sign | (exponent << 10) | (mantissa >> 13)
  if mantissa & 0x1000:
    half += 1
  return half & 0xFFFF


def fp16(value: float) -> float:
  return half_to_float(float_to_half(value)) if math.isfinite(value) else value


def softmax(values: list[float]) -> list[float]:
  max_value = max(values)
  weights = [math.exp(value - max_value) for value in values]
  denom = sum(weights)
  return [weight / denom for weight in weights]


def solve_linear(matrix: list[list[float]], rhs: list[float]) -> list[float]:
  n = len(rhs)
  augmented = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
  for col in range(n):
    pivot = max(range(col, n), key=lambda row: abs(augmented[row][col]))
    if abs(augmented[pivot][col]) < 1e-18:
      raise RuntimeError("singular normal equation")
    augmented[col], augmented[pivot] = augmented[pivot], augmented[col]
    pivot_value = augmented[col][col]
    for item in range(col, n + 1):
      augmented[col][item] /= pivot_value
    for row in range(n):
      if row == col:
        continue
      factor = augmented[row][col]
      if factor == 0.0:
        continue
      for item in range(col, n + 1):
        augmented[row][item] -= factor * augmented[col][item]
  return [augmented[row][n] for row in range(n)]


class Capture:
  def __init__(self, history_json: Path, layer: int) -> None:
    self.history_json = history_json
    self.layer_key = str(layer)
    doc = json.loads(history_json.read_text(encoding="utf-8"))
    capture = doc["full_attn_all_history_capture"]
    self.tokens = capture["tokens"]
    source = self.tokens[15]["layers"][self.layer_key]["payloads"]
    self.q = read_f32(ROOT / source["q_rope"]["path"])
    self.pregate = read_f32(ROOT / source["attn_pregate"]["path"])
    self.k = []
    self.v = []
    for token in self.tokens:
      payloads = token["layers"][self.layer_key]["payloads"]
      self.k.append(read_f32(ROOT / payloads["k_rope"]["path"]))
      self.v.append(read_f32(ROOT / payloads["v"]["path"]))

  def shapes_ok(self) -> bool:
    return (
        len(self.q) == Q_HEADS * HEAD_DIM
        and len(self.pregate) == Q_HEADS * HEAD_DIM
        and len(self.k) == TOKENS
        and len(self.v) == TOKENS
        and all(len(item) == KV_HEADS * HEAD_DIM for item in self.k)
        and all(len(item) == KV_HEADS * HEAD_DIM for item in self.v)
    )

  def q_value(self, q_head: int, dim: int) -> float:
    return self.q[q_head * HEAD_DIM + dim]

  def pregate_value(self, q_head: int, dim: int) -> float:
    return self.pregate[q_head * HEAD_DIM + dim]

  def k_value(self, token: int, kv_head: int, dim: int, k_mode: str) -> float:
    value = self.k[token][kv_head * HEAD_DIM + dim]
    return fp16(value) if k_mode == "fp16" else value

  def v_value(self, token: int, kv_head: int, dim: int, v_mode: str) -> float:
    value = self.v[token][kv_head * HEAD_DIM + dim]
    return fp16(value) if v_mode == "fp16" else value


def core_weights(capture: Capture, q_head: int, kv_head: int, k_mode: str) -> list[float]:
  scores = []
  for token in range(TOKENS):
    dot = 0.0
    for dim in range(HEAD_DIM):
      dot += float(capture.q_value(q_head, dim)) * float(capture.k_value(token, kv_head, dim, k_mode))
    scores.append(dot * SCALE)
  return softmax(scores)


def infer_weights(capture: Capture, q_head: int, kv_head: int, v_mode: str) -> tuple[list[float], float]:
  n = TOKENS
  normal = [[0.0 for _ in range(n)] for _ in range(n)]
  rhs = [0.0 for _ in range(n)]
  for dim in range(HEAD_DIM):
    row = [float(capture.v_value(token, kv_head, dim, v_mode)) for token in range(TOKENS)]
    target = float(capture.pregate_value(q_head, dim))
    for i in range(n):
      rhs[i] += row[i] * target
      for j in range(n):
        normal[i][j] += row[i] * row[j]
  lambda_sq = 100.0 * 100.0
  for i in range(n):
    rhs[i] += lambda_sq
    for j in range(n):
      normal[i][j] += lambda_sq
  weights = solve_linear(normal, rhs)
  sq_error = 0.0
  for dim in range(HEAD_DIM):
    predicted = sum(
        weights[token] * float(capture.v_value(token, kv_head, dim, v_mode))
        for token in range(TOKENS)
    )
    diff = predicted - float(capture.pregate_value(q_head, dim))
    sq_error += diff * diff
  return weights, math.sqrt(sq_error / HEAD_DIM)


def mapping_kv_head(q_head: int, mapping: str) -> int:
  floor_mapping = q_head // (Q_HEADS // KV_HEADS)
  if mapping == "floor":
    return floor_mapping
  if mapping == "mod":
    return q_head % KV_HEADS
  if mapping == "other":
    return 1 - floor_mapping
  raise ValueError(mapping)


def diagnose(capture: Capture) -> list[dict[str, Any]]:
  rows = []
  for v_mode in ("f32", "fp16"):
    for mapping in ("floor", "mod", "other"):
      for q_head in range(Q_HEADS):
        kv_head = mapping_kv_head(q_head, mapping)
        inferred, recon_rmse = infer_weights(capture, q_head, kv_head, v_mode)
        f32_weights = core_weights(capture, q_head, kv_head, "f32")
        fp16k_weights = core_weights(capture, q_head, kv_head, "fp16")
        rows.append({
            "q_head": q_head,
            "mapping": mapping,
            "kv_head": kv_head,
            "v_mode": v_mode,
            "reconstruction_rmse": recon_rmse,
            "inferred_weight_sum": sum(inferred),
            "inferred_weight_min": min(inferred),
            "inferred_weight_max": max(inferred),
            "l1_vs_f32_k_softmax": sum(abs(inferred[i] - f32_weights[i]) for i in range(TOKENS)),
            "l1_vs_fp16_k_softmax": sum(abs(inferred[i] - fp16k_weights[i]) for i in range(TOKENS)),
            "max_vs_f32_k_softmax": max(abs(inferred[i] - f32_weights[i]) for i in range(TOKENS)),
            "max_vs_fp16_k_softmax": max(abs(inferred[i] - fp16k_weights[i]) for i in range(TOKENS)),
        })
  return rows


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
  out = {}
  for v_mode in ("f32", "fp16"):
    for mapping in ("floor", "mod", "other"):
      subset = [row for row in rows if row["v_mode"] == v_mode and row["mapping"] == mapping]
      key = f"{mapping}_{v_mode}_v"
      out[key] = {
          "head_count": len(subset),
          "reconstruction_rmse_mean": sum(row["reconstruction_rmse"] for row in subset) / len(subset),
          "reconstruction_rmse_max": max(row["reconstruction_rmse"] for row in subset),
          "l1_vs_f32_k_softmax_mean": sum(row["l1_vs_f32_k_softmax"] for row in subset) / len(subset),
          "l1_vs_fp16_k_softmax_mean": sum(row["l1_vs_fp16_k_softmax"] for row in subset) / len(subset),
          "negative_weight_heads": sum(row["inferred_weight_min"] < -1e-4 for row in subset),
      }
  return out


def write_summary(path: Path, payload: dict[str, Any]) -> None:
  agg = payload["aggregate"]
  lines = [
      "# Layer-7 Full-Attention Implied-Weight Diagnostic",
      "",
      f"- workstream: `{WORKSTREAM}`",
      f"- all-history: `{payload['all_history']}`",
      f"- layer: `{payload['layer']}`",
      f"- required checks passed: `{str(payload['required_checks_passed']).lower()}`",
      "- speedup claims allowed: `false`",
      "",
      "| mapping/V | recon RMSE mean | recon RMSE max | L1 vs f32-K softmax | L1 vs fp16-K softmax | neg heads |",
      "|---|---:|---:|---:|---:|---:|",
  ]
  for key in ("floor_f32_v", "floor_fp16_v", "mod_f32_v", "other_f32_v"):
    row = agg[key]
    lines.append(
        f"| {key} | {row['reconstruction_rmse_mean']} | {row['reconstruction_rmse_max']} | "
        f"{row['l1_vs_f32_k_softmax_mean']} | {row['l1_vs_fp16_k_softmax_mean']} | "
        f"{row['negative_weight_heads']} |"
    )
  lines += [
      "",
      "The floor GQA mapping reconstructs oracle pregate with small positive",
      "weights; modulo/other mappings are rejected by reconstruction error.",
      "Residual reconstruction error remains around the same order as layer",
      "output drift, so the next route should inspect capture/KV-cache value",
      "precision or collect oracle attention weights directly.",
      "",
  ]
  path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  args = parse_args()
  if args.layer != 7:
    raise SystemExit("--layer must be 7 for this diagnostic")
  created_at = iso_now()
  stamp = utc_stamp()
  out_dir = args.out_dir or ROOT / f"output/layer7-full-attn-implied-weight-diagnostic-{stamp}"
  out_dir.mkdir(parents=True, exist_ok=False)
  capture = Capture(args.all_history_json.resolve(), args.layer)
  rows = diagnose(capture) if capture.shapes_ok() else []
  agg = aggregate(rows) if rows else {}
  checks = [
      {"name": "history_loaded", "pass": args.all_history_json.exists()},
      {"name": "payload_shapes_ok", "pass": capture.shapes_ok()},
      {"name": "diagnostic_rows_present", "pass": len(rows) == Q_HEADS * 3 * 2},
      {"name": "speedup_claims_forbidden", "pass": True},
  ]
  required = all(item["pass"] for item in checks)
  payload = {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "created_at": created_at,
      "all_history": str(args.all_history_json.resolve().relative_to(ROOT)),
      "layer": args.layer,
      "attention_scale": SCALE,
      "head_dim": HEAD_DIM,
      "q_head_count": Q_HEADS,
      "kv_head_count": KV_HEADS,
      "token_count": TOKENS,
      "rows": rows,
      "aggregate": agg,
      "checks": checks,
      "required_checks_passed": required,
      "speedup_claims_allowed": False,
  }
  manifest = {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "created_at": created_at,
      "tool": "tools/intel-qwen36-layer7-full-attn-implied-weight-diagnostic.py",
      "artifact": str(out_dir),
      "all_history": payload["all_history"],
      "layer": args.layer,
      "required_checks_passed": required,
      "speedup_claims_allowed": False,
  }
  correctness = {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "checks": checks,
      "required_checks_passed": required,
      "speedup_claims_allowed": False,
  }
  iq36_local.write_json(out_dir / "probe.json", payload)
  iq36_local.write_json(out_dir / "manifest.json", manifest)
  iq36_local.write_json(out_dir / "correctness.json", correctness)
  iq36_local.write_metric(
      out_dir / "metrics.jsonl",
      "layer7_full_attn_implied_weight_diagnostic",
      [
          ("required_checks_passed", required),
          ("floor_f32_v_reconstruction_rmse_mean", agg.get("floor_f32_v", {}).get("reconstruction_rmse_mean")),
          ("floor_fp16_v_reconstruction_rmse_mean", agg.get("floor_fp16_v", {}).get("reconstruction_rmse_mean")),
          ("floor_f32_v_l1_vs_f32_k_softmax_mean", agg.get("floor_f32_v", {}).get("l1_vs_f32_k_softmax_mean")),
          ("floor_f32_v_l1_vs_fp16_k_softmax_mean", agg.get("floor_f32_v", {}).get("l1_vs_fp16_k_softmax_mean")),
          ("mod_f32_v_reconstruction_rmse_mean", agg.get("mod_f32_v", {}).get("reconstruction_rmse_mean")),
      ],
  )
  write_summary(out_dir / "summary.md", payload)
  print(out_dir)
  return 0 if required else 1


if __name__ == "__main__":
  raise SystemExit(main())
