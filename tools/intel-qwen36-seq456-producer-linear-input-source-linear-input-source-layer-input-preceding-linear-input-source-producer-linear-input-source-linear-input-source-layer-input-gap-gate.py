#!/usr/bin/env python3
"""Locate seq455 upstream drift feeding source layer inputs."""

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
    "linear-input-source-linear-input-source-layer-input-gap-gate.py"
)

SCHEMA_VERSION = (
    "intel-qwen36-seq456-producer-linear-input-source-linear-input-source-"
    "layer-input-preceding-linear-input-source-producer-linear-input-source-"
    "linear-input-source-layer-input-gap-gate-v0"
)
SOURCE_GAP_ROUTE = (
    "router_prompt_full_attention_layer_input_preceding_linear_input_source_"
    "producer_linear_input_source_linear_input_source_layer_input_preceding_"
    "linear_input_source_producer_linear_input_source_linear_input_source_"
    "layer_input_preceding_linear_input_source_producer_linear_input_source_"
    "linear_input_source_layer_input_preceding_linear_input_source_"
    "producer_linear_input_source_linear_input_source_layer_input_preceding_"
    "linear_input_source_producer_linear_input_source_linear_input_source_"
    "layer_input_preceding_linear_input_source_producer_linear_input_source_"
    "linear_input_source_layer_input_preceding_linear_input_source_gap_gate"
)
CURRENT_ROUTE = SOURCE_GAP_ROUTE.replace(
    "_linear_input_source_gap_gate",
    "_linear_input_source_producer_linear_input_source_linear_input_source_layer_input_gap_gate")
OUTPUT_ROUTE = CURRENT_ROUTE.replace(
    "_layer_input_gap_gate",
    "_layer_input_preceding_linear_output_gap_gate")
MATH_ROUTE = CURRENT_ROUTE.replace(
    "_layer_input_gap_gate",
    "_layer_input_preceding_linear_math_gap_gate")
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
DEFAULT_SEQ454 = (
    ROOT
    / "output/seq454-producer-linear-input-source-linear-input-source-layer-input-preceding-linear-input-source-producer-linear-input-source-linear-input-source-gap-gate-20260709Tseq454Z"
    / "metrics.json"
)
DEFAULT_SEQ455 = (
    ROOT
    / "output/seq455-producer-linear-input-source-linear-input-source-layer-input-preceding-linear-input-source-producer-linear-input-source-linear-input-source-ffn-input-gap-gate-20260709Tseq455Z"
    / "metrics.json"
)
DEFAULT_TOKEN_INPUT_DIR = (
    ROOT
    / "output/r2-gpu-acceptance-matrix-router-both-rowblock16-26mask-20260708Tseq222Z"
    / "token-input"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "output/seq456-producer-linear-input-source-linear-input-source-layer-input-preceding-linear-input-source-producer-linear-input-source-linear-input-source-layer-input-gap-gate-20260709Tseq456Z"
)
DEFAULT_HOST = "local"
DEFAULT_MODEL = "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf"
DEFAULT_ENV_SCRIPT = "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh"
DEFAULT_REMOTE_ROOT = "/home/intel/intel-qwen36-gpu"

TARGET_LAYERS = [3, 7]
PRECEDING_LINEAR_LAYERS = [layer - 1 for layer in TARGET_LAYERS]
DECODE_TOKENS = 8
EXPECTED_EVENTS = len(TARGET_LAYERS) * DECODE_TOKENS
EXPECTED_COUNTER_LAYERS = 72
EXPECTED_COUNTER_VALUES = 147456


def _load_base() -> Any:
  spec = importlib.util.spec_from_file_location(
      "iq36_seq456_producer_linear_input_source_preceding_source_layer_input_base",
      BASE_GATE)
  if spec is None or spec.loader is None:
    raise SystemExit(f"cannot load base layer-input gate: {BASE_GATE}")
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


def _patch_base() -> None:
  BASE.SCHEMA_VERSION = SCHEMA_VERSION
  BASE.CURRENT_ROUTE = CURRENT_ROUTE
  BASE.OUTPUT_ROUTE = OUTPUT_ROUTE
  BASE.MATH_ROUTE = MATH_ROUTE
  BASE.DISPOSITION_PREFIX = DISPOSITION_PREFIX
  BASE.DIAG_PREFIX = DIAG_PREFIX
  BASE.TARGET_LAYERS = TARGET_LAYERS
  BASE.PRECEDING_LINEAR_LAYERS = PRECEDING_LINEAR_LAYERS
  BASE.EXPECTED_EVENTS = EXPECTED_EVENTS
  BASE.EXPECTED_COUNTER_LAYERS = EXPECTED_COUNTER_LAYERS
  BASE.EXPECTED_COUNTER_VALUES = EXPECTED_COUNTER_VALUES
  BASE.OLD.TARGET_LAYERS = TARGET_LAYERS
  BASE.OLD.PRECEDING_LINEAR_LAYERS = PRECEDING_LINEAR_LAYERS
  BASE.OLD.EXPECTED_EVENTS = EXPECTED_EVENTS
  BASE.OLD.EXPECTED_COUNTER_LAYERS = EXPECTED_COUNTER_LAYERS
  BASE.OLD.EXPECTED_COUNTER_VALUES = EXPECTED_COUNTER_VALUES

  original_has_candidate = BASE._has_candidate
  original_has_switch = BASE._has_switch

  def has_candidate(routes: dict[str, Any], seq: int,
                    disposition: str) -> bool:
    if seq == 410:
      return original_has_candidate(routes, 455, disposition)
    return original_has_candidate(routes, seq, disposition)

  def has_switch(routes: dict[str, Any], decision: str,
                 seq_covered: int) -> bool:
    if seq_covered == 410:
      return original_has_switch(routes, decision, 455)
    return original_has_switch(routes, decision, seq_covered)

  BASE._has_candidate = has_candidate
  BASE._has_switch = has_switch


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--routes", type=Path, default=DEFAULT_ROUTES)
  parser.add_argument("--seq454", type=Path, default=DEFAULT_SEQ454)
  parser.add_argument("--seq455", type=Path, default=DEFAULT_SEQ455)
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
  base_args.seq409 = args.seq454
  base_args.seq410 = args.seq455
  metrics = BASE.compute(base_args)
  metrics["inputs"]["seq454"] = metrics["inputs"].pop("seq409")
  metrics["inputs"]["seq455"] = metrics["inputs"].pop("seq410")
  for row in metrics.get("checks", []):
    if row.get("name") == "seq410_selected_deeper_nested_source_layer_input_gap_gate":
      row["name"] = "seq455_selected_deeper_nested_source_layer_input_gap_gate"
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
