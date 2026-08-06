#!/usr/bin/env python3
"""Classify seq438 producer linear input-source value source drift."""

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
    "linear-input-source-linear-input-source-gap-gate.py"
)

SCHEMA_VERSION = "intel-qwen36-seq439-producer-linear-input-source-linear-input-source-gap-gate-v0"
PREVIOUS_ROUTE = (
    "router_prompt_full_attention_layer_input_preceding_linear_input_source_"
    "producer_linear_input_source_linear_input_source_layer_input_preceding_"
    "linear_input_source_producer_linear_input_source_linear_input_source_"
    "layer_input_preceding_linear_input_source_producer_linear_input_source_"
    "linear_input_source_layer_input_preceding_linear_input_source_"
    "producer_linear_input_source_linear_input_source_layer_input_preceding_"
    "linear_input_source_producer_linear_input_source_linear_input_source_"
    "layer_input_preceding_linear_input_source_producer_linear_input_source_"
    "linear_input_gap_gate"
)
CURRENT_ROUTE = PREVIOUS_ROUTE.replace(
    "_linear_input_gap_gate", "_linear_input_source_gap_gate")
NEXT_ROUTE = CURRENT_ROUTE.replace(
    "_linear_input_source_gap_gate", "_linear_input_source_ffn_input_gap_gate")
MATH_ROUTE = CURRENT_ROUTE.replace(
    "_linear_input_source_gap_gate", "_linear_input_source_ffn_math_gap_gate")
_CURRENT_SOURCE = CURRENT_ROUTE.replace(
    "router_prompt_full_attention_layer_input_preceding_linear_input_source_",
    "source_layer_input_preceding_linear_input_source_",
    1,
)
DISPOSITION_PREFIX = _CURRENT_SOURCE.removesuffix(
    "_producer_linear_input_source_linear_input_source_layer_input_preceding_"
    "linear_input_source_producer_linear_input_source_linear_input_source_"
    "gap_gate")
DIAG_PREFIX = f"{DISPOSITION_PREFIX}_producer"

DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_SEQ436 = (
    ROOT
    / "output/seq436-producer-linear-input-source-attention-output-gap-gate-20260709Tseq436Z"
    / "metrics.json"
)
DEFAULT_SEQ438 = (
    ROOT
    / "output/seq438-producer-linear-input-source-linear-input-gap-gate-20260709Tseq438Z"
    / "metrics.json"
)
DEFAULT_TOKEN_INPUT_DIR = (
    ROOT
    / "output/r2-gpu-acceptance-matrix-router-both-rowblock16-26mask-20260708Tseq222Z"
    / "token-input"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "output/seq439-producer-linear-input-source-linear-input-source-gap-gate-20260709Tseq439Z"
)
DEFAULT_HOST = "local"
DEFAULT_MODEL = "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf"
DEFAULT_ENV_SCRIPT = "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh"
DEFAULT_REMOTE_ROOT = "/home/intel/intel-qwen36-gpu"

TARGET_LINEAR_LAYERS = [4, 8, 12]
SOURCE_FFN_LAYERS = [layer - 1 for layer in TARGET_LINEAR_LAYERS]
DECODE_TOKENS = 8
EXPECTED_EVENTS = len(TARGET_LINEAR_LAYERS) * DECODE_TOKENS
EXPECTED_COUNTER_LAYERS = 72
EXPECTED_COUNTER_VALUES = 147456


def _load_base() -> Any:
  spec = importlib.util.spec_from_file_location(
      "iq36_seq439_producer_linear_input_source_linear_input_source_base",
      BASE_GATE)
  if spec is None or spec.loader is None:
    raise SystemExit(f"cannot load base input-source gate: {BASE_GATE}")
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
  BASE.NEXT_ROUTE = NEXT_ROUTE
  BASE.MATH_ROUTE = MATH_ROUTE
  BASE.DISPOSITION_PREFIX = DISPOSITION_PREFIX
  BASE.DIAG_PREFIX = DIAG_PREFIX
  BASE.TARGET_LINEAR_LAYERS = TARGET_LINEAR_LAYERS
  BASE.SOURCE_FFN_LAYERS = SOURCE_FFN_LAYERS
  BASE.EXPECTED_EVENTS = EXPECTED_EVENTS
  BASE.EXPECTED_COUNTER_LAYERS = EXPECTED_COUNTER_LAYERS
  BASE.EXPECTED_COUNTER_VALUES = EXPECTED_COUNTER_VALUES
  BASE.OLD.TARGET_LINEAR_LAYERS = TARGET_LINEAR_LAYERS
  BASE.OLD.SOURCE_FFN_LAYERS = SOURCE_FFN_LAYERS
  BASE.OLD.EXPECTED_EVENTS = EXPECTED_EVENTS
  BASE.OLD.EXPECTED_COUNTER_LAYERS = EXPECTED_COUNTER_LAYERS
  BASE.OLD.EXPECTED_COUNTER_VALUES = EXPECTED_COUNTER_VALUES

  original_has_candidate = BASE._has_candidate
  original_has_switch = BASE._has_switch

  def has_candidate(routes: dict[str, Any], seq: int,
                    disposition: str) -> bool:
    if seq == 408:
      return original_has_candidate(routes, 438, disposition)
    return original_has_candidate(routes, seq, disposition)

  def has_switch(routes: dict[str, Any], decision: str,
                 seq_covered: int) -> bool:
    if seq_covered == 408:
      return original_has_switch(routes, decision, 438)
    return original_has_switch(routes, decision, seq_covered)

  BASE._has_candidate = has_candidate
  BASE._has_switch = has_switch


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--routes", type=Path, default=DEFAULT_ROUTES)
  parser.add_argument("--seq436", type=Path, default=DEFAULT_SEQ436)
  parser.add_argument("--seq438", type=Path, default=DEFAULT_SEQ438)
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
  base_args.seq406 = args.seq436
  base_args.seq408 = args.seq438
  metrics = BASE.compute(base_args)
  metrics["inputs"]["seq436"] = metrics["inputs"].pop("seq406")
  metrics["inputs"]["seq438"] = metrics["inputs"].pop("seq408")
  for row in metrics.get("checks", []):
    if row.get("name") == "seq408_selected_deeper_nested_source_linear_input_source_gate":
      row["name"] = "seq438_selected_deeper_nested_source_linear_input_source_gate"
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
