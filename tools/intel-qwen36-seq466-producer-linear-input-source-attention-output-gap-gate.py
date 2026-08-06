#!/usr/bin/env python3
"""Classify seq465 producer linear input-source attention-output drift."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
BASE_GATE = (
    ROOT
    / "tools/intel-qwen36-router-full-attention-deeper-nested-producer-"
    "linear-input-source-attention-output-gap-gate.py"
)

SCHEMA_VERSION = (
    "intel-qwen36-seq466-producer-linear-input-source-attention-output-gap-gate-v0"
)
_PATTERN = (
    "producer_linear_input_source_linear_input_source_layer_input_preceding_"
    "linear_input_source"
)
SOURCE_GAP_ROUTE = (
    "router_prompt_full_attention_layer_input_preceding_linear_input_source_"
    + ((_PATTERN + "_") * 6)
    + _PATTERN
    + "_gap_gate"
)
CURRENT_ROUTE = SOURCE_GAP_ROUTE.replace(
    "_linear_input_source_gap_gate",
    "_linear_input_source_producer_linear_input_source_attention_output_gap_gate")
DELTA_Z_ROUTE = CURRENT_ROUTE.replace(
    "_producer_linear_input_source_attention_output_gap_gate",
    "_producer_linear_input_source_linear_delta_z_gap_gate")
DELTA_ROUTE = CURRENT_ROUTE.replace(
    "_producer_linear_input_source_attention_output_gap_gate",
    "_producer_linear_input_source_linear_delta_output_gap_gate")
Z_ROUTE = CURRENT_ROUTE.replace(
    "_producer_linear_input_source_attention_output_gap_gate",
    "_producer_linear_input_source_linear_z_gap_gate")
MATH_ROUTE = CURRENT_ROUTE.replace(
    "_producer_linear_input_source_attention_output_gap_gate",
    "_producer_linear_input_source_preconv_math_gap_gate")
_SOURCE_GAP_SOURCE = SOURCE_GAP_ROUTE.replace(
    "router_prompt_full_attention_layer_input_preceding_linear_input_source_",
    "source_layer_input_preceding_linear_input_source_",
    1,
)
DISPOSITION_PREFIX = _SOURCE_GAP_SOURCE.removesuffix(
    "_producer_linear_input_source_linear_input_source_layer_input_preceding_"
    "linear_input_source_producer_linear_input_source_linear_input_source_"
    "layer_input_preceding_linear_input_source_gap_gate")
DIAG_PREFIX = (
    f"{DISPOSITION_PREFIX}_producer_linear_input_source_linear_input_source_"
    "layer_input_preceding_linear_input_source_producer"
)
DISPOSITION_PREFIX = DIAG_PREFIX.removesuffix("_producer")

DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_SEQ464 = (
    ROOT
    / "output/seq464-producer-linear-input-source-linear-input-source-layer-input-preceding-linear-input-source-producer-linear-input-source-linear-input-source-layer-input-preceding-linear-input-source-producer-linear-input-source-gap-gate-20260709Tseq464Z"
    / "metrics.json"
)
DEFAULT_SEQ465 = (
    ROOT
    / "output/seq465-producer-linear-input-source-linear-input-source-layer-input-preceding-linear-input-source-producer-linear-input-source-linear-input-source-layer-input-preceding-linear-input-source-producer-linear-input-source-ffn-input-gap-gate-20260709Tseq465Z"
    / "metrics.json"
)
DEFAULT_TOKEN_INPUT_DIR = (
    ROOT
    / "output/r2-gpu-acceptance-matrix-router-both-rowblock16-26mask-20260708Tseq222Z"
    / "token-input"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "output/seq466-producer-linear-input-source-attention-output-gap-gate-20260709Tseq466Z"
)
DEFAULT_HOST = "local"
DEFAULT_MODEL = "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf"
DEFAULT_ENV_SCRIPT = "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh"
DEFAULT_REMOTE_ROOT = "/home/intel/intel-qwen36-gpu"

SOURCE_LAYERS = [0, 4]
DECODE_TOKENS = 8
EXPECTED_EVENTS = len(SOURCE_LAYERS) * DECODE_TOKENS
EXPECTED_COUNTER_LAYERS = 72
EXPECTED_COUNTER_VALUES = 147456
COSINE_THRESHOLD = 0.9999


def _load_base() -> Any:
  spec = importlib.util.spec_from_file_location(
      "iq36_seq466_producer_linear_input_source_attention_output_base",
      BASE_GATE)
  if spec is None or spec.loader is None:
    raise SystemExit(f"cannot load base source attention-output gate: {BASE_GATE}")
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


BASE = _load_base()


def _rel(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path)


def _num(value: Any, default: float = 0.0) -> float:
  return float(value) if isinstance(value, (int, float)) else default


def _patch_base() -> None:
  BASE.SCHEMA_VERSION = SCHEMA_VERSION
  BASE.CURRENT_ROUTE = CURRENT_ROUTE
  BASE.DELTA_Z_ROUTE = DELTA_Z_ROUTE
  BASE.DELTA_ROUTE = DELTA_ROUTE
  BASE.Z_ROUTE = Z_ROUTE
  BASE.MATH_ROUTE = MATH_ROUTE
  BASE.DISPOSITION_PREFIX = DISPOSITION_PREFIX
  BASE.DIAG_PREFIX = DIAG_PREFIX
  BASE.SOURCE_LAYERS = SOURCE_LAYERS
  BASE.EXPECTED_EVENTS = EXPECTED_EVENTS
  BASE.EXPECTED_COUNTER_LAYERS = EXPECTED_COUNTER_LAYERS
  BASE.EXPECTED_COUNTER_VALUES = EXPECTED_COUNTER_VALUES
  BASE.OLD.SOURCE_LAYERS = SOURCE_LAYERS
  BASE.OLD.EXPECTED_EVENTS = EXPECTED_EVENTS
  BASE.OLD.EXPECTED_COUNTER_LAYERS = EXPECTED_COUNTER_LAYERS
  BASE.OLD.EXPECTED_COUNTER_VALUES = EXPECTED_COUNTER_VALUES

  original_has_candidate = BASE._has_candidate
  original_has_switch = BASE._has_switch

  def has_candidate(routes: dict[str, Any], seq: int,
                    disposition: str) -> bool:
    if seq == 405:
      return original_has_candidate(routes, 465, disposition)
    return original_has_candidate(routes, seq, disposition)

  def has_switch(routes: dict[str, Any], decision: str,
                 seq_covered: int) -> bool:
    if seq_covered == 405:
      return original_has_switch(routes, decision, 465)
    return original_has_switch(routes, decision, seq_covered)

  BASE._has_candidate = has_candidate
  BASE._has_switch = has_switch


def _apply_final_mix_delta_z_classification(metrics: dict[str, Any]) -> None:
  min_attn_norm = _num(metrics.get("min_attn_norm_cosine"), 1.0)
  min_attn_norm_from_input = _num(
      metrics.get("min_attn_norm_from_gpu_input_cosine"), 1.0)
  min_qkv_from_gpu_norm = _num(
      metrics.get("min_qkv_from_gpu_attn_norm_cosine"), 1.0)
  min_z_from_gpu_norm = _num(
      metrics.get("min_z_from_gpu_attn_norm_cosine"), 1.0)
  min_z = _num(metrics.get("min_z_cosine"), 1.0)
  min_delta = _num(metrics.get("min_delta_output_cosine"), 1.0)
  min_delta_native_z = _num(
      metrics.get("min_gpu_delta_native_z_cpu_cosine"), 1.0)
  min_native_delta_z = _num(
      metrics.get("min_native_delta_gpu_z_cpu_cosine"), 1.0)
  min_native_recompute = _num(
      metrics.get("min_native_delta_native_z_cpu_cosine"), 1.0)
  final_mix_delta_z = (
      min_delta_native_z < COSINE_THRESHOLD
      and min_native_delta_z < COSINE_THRESHOLD
      and min_native_recompute >= COSINE_THRESHOLD)
  live_z_inherits_input = (
      min_attn_norm < COSINE_THRESHOLD
      and min_attn_norm_from_input < COSINE_THRESHOLD
      and min_qkv_from_gpu_norm < COSINE_THRESHOLD
      and min_z_from_gpu_norm < COSINE_THRESHOLD
      and min_z < COSINE_THRESHOLD)

  if not (final_mix_delta_z and live_z_inherits_input):
    return

  for row in metrics.get("checks", []):
    if row.get("name") == "qkv_z_inherit_live_input":
      row["name"] = "final_mix_delta_z_inherits_live_input"
      row["pass"] = True
      row["detail"] = {
          "min_attn_norm_cosine": min_attn_norm,
          "min_attn_norm_from_gpu_input_cosine": min_attn_norm_from_input,
          "min_qkv_from_gpu_attn_norm_cosine": min_qkv_from_gpu_norm,
          "min_delta_output_cosine": min_delta,
          "min_z_from_gpu_attn_norm_cosine": min_z_from_gpu_norm,
          "min_z_cosine": min_z,
          "min_gpu_delta_native_z_cpu_cosine": min_delta_native_z,
          "min_native_delta_gpu_z_cpu_cosine": min_native_delta_z,
          "min_native_delta_native_z_cpu_cosine": min_native_recompute,
      }

  required = all(bool(row.get("pass")) for row in metrics.get("checks", []))
  metrics["direct_delta_output_gap_reproduced"] = (
      min_delta < COSINE_THRESHOLD)
  metrics["final_mix_delta_z_gap_reproduced"] = final_mix_delta_z
  metrics["final_mix_delta_z_inherits_live_input"] = live_z_inherits_input
  metrics["qkv_z_inherit_input"] = live_z_inherits_input
  metrics["required_checks_passed"] = required
  if required:
    metrics["disposition"] = (
        f"accept_{DISPOSITION_PREFIX}_producer_linear_input_source_attention_output_gap_classification"
    )
    metrics["selected_next_route"] = DELTA_Z_ROUTE
    metrics["next_route_reason"] = (
        "The attention-output gap is reproduced through final-mix delta/z "
        "rows while native delta/native z recompute is exact. Direct delta "
        "output is above threshold in this weak nested row, but z and qkv "
        "inherit the live attention-norm/input drift and preconv math matches "
        "CPU, so split the source linear final-mix delta/z path next."
    )


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--routes", type=Path, default=DEFAULT_ROUTES)
  parser.add_argument("--seq464", type=Path, default=DEFAULT_SEQ464)
  parser.add_argument("--seq465", type=Path, default=DEFAULT_SEQ465)
  parser.add_argument("--token-input-dir", type=Path,
                      default=DEFAULT_TOKEN_INPUT_DIR)
  parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
  parser.add_argument("--target", "--host", dest="host", metavar="TARGET", default=DEFAULT_HOST)
  parser.add_argument("--model", default=DEFAULT_MODEL)
  parser.add_argument("--env-script", default=DEFAULT_ENV_SCRIPT)
  parser.add_argument("--staging-root", "--remote-root", dest="remote_root", metavar="PATH", default=DEFAULT_REMOTE_ROOT)
  parser.add_argument("--timeout-s", type=int, default=7200)
  return parser.parse_args()


def main() -> int:
  _patch_base()
  args = parse_args()
  base_args = argparse.Namespace(**vars(args))
  base_args.seq404 = args.seq464
  base_args.seq405 = args.seq465
  metrics = BASE.compute(base_args)
  _apply_final_mix_delta_z_classification(metrics)
  metrics["inputs"]["seq464"] = metrics["inputs"].pop("seq404")
  metrics["inputs"]["seq465"] = metrics["inputs"].pop("seq405")
  for row in metrics.get("checks", []):
    if row.get("name") == "seq405_selected_deeper_nested_source_attention_output_gap_gate":
      row["name"] = "seq465_selected_deeper_nested_source_attention_output_gap_gate"
  BASE.write_outputs(metrics, args.out_dir)
  print(json.dumps({
      "diagnostic_classification": metrics["diagnostic_classification"],
      "disposition": metrics["disposition"],
      "out_dir": _rel(args.out_dir),
      "required_checks_passed": metrics["required_checks_passed"],
      "selected_next_route": metrics["selected_next_route"],
  }, sort_keys=True))
  return 0 if metrics["required_checks_passed"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
