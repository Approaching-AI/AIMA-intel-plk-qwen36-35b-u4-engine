#!/usr/bin/env python3
"""Classify seq433 producer linear input value source drift."""

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
    "linear-input-source-linear-input-source-layer-input-preceding-linear-"
    "input-source-producer-linear-input-source-gap-gate.py"
)

SCHEMA_VERSION = (
    "intel-qwen36-router-full-attention-deeper-nested-producer-linear-"
    "input-source-linear-input-source-layer-input-preceding-linear-"
    "input-source-producer-linear-input-source-linear-input-source-layer-"
    "input-preceding-linear-input-source-producer-linear-input-source-gap-"
    "gate-v0"
)
ROUTE_PREFIX = (
    "router_prompt_full_attention_layer_input_preceding_linear_input_source_"
    "producer_linear_input_source_linear_input_source_layer_input_preceding_"
    "linear_input_source_producer_linear_input_source_linear_input_source_"
    "layer_input_preceding_linear_input_source_producer_linear_input_source_"
    "linear_input_source_layer_input_preceding_linear_input_source"
)
CURRENT_ROUTE = (
    f"{ROUTE_PREFIX}_producer_linear_input_source_linear_input_source_layer_input_preceding_linear_input_source_producer_linear_input_source_linear_input_source_layer_input_preceding_linear_input_source_producer_linear_input_source_gap_gate"
)
NEXT_ROUTE = (
    f"{ROUTE_PREFIX}_producer_linear_input_source_linear_input_source_layer_input_preceding_linear_input_source_producer_linear_input_source_linear_input_source_layer_input_preceding_linear_input_source_producer_linear_input_source_ffn_input_gap_gate"
)
MATH_ROUTE = (
    f"{ROUTE_PREFIX}_producer_linear_input_source_linear_input_source_layer_input_preceding_linear_input_source_producer_linear_input_source_linear_input_source_layer_input_preceding_linear_input_source_producer_linear_input_source_ffn_math_gap_gate"
)

DISPOSITION_PREFIX = (
    "source_layer_input_preceding_linear_input_source_producer_linear_input_"
    "source_linear_input_source_layer_input_preceding_linear_input_source_"
    "producer_linear_input_source_linear_input_source_layer_input_preceding_"
    "linear_input_source_producer_linear_input_source_linear_input_source_"
    "layer_input_preceding_linear_input_source"
)
DIAG_PREFIX = (
    f"{DISPOSITION_PREFIX}_producer_linear_input_source_linear_input_source_"
    "layer_input_preceding_linear_input_source_producer"
)

DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_SEQ431 = (
    ROOT
    / "output/router-full-attention-deeper-nested-producer-linear-input-source-linear-input-source-layer-input-preceding-linear-input-source-producer-linear-input-source-linear-input-source-layer-input-preceding-linear-input-source-ffn-input-gap-gate-20260709Tseq431Z"
    / "metrics.json"
)
DEFAULT_SEQ433 = (
    ROOT / "output/seq433-producer-linear-input-gap-gate-20260709Tseq433Z"
    / "metrics.json"
)
DEFAULT_TOKEN_INPUT_DIR = (
    ROOT
    / "output/r2-gpu-acceptance-matrix-router-both-rowblock16-26mask-20260708Tseq222Z"
    / "token-input"
)
DEFAULT_OUT_DIR = (
    ROOT / "output/seq434-producer-linear-input-source-gap-gate-20260709Tseq434Z"
)
DEFAULT_HOST = "local"
DEFAULT_MODEL = "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf"
DEFAULT_ENV_SCRIPT = "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh"
DEFAULT_REMOTE_ROOT = "/home/intel/intel-qwen36-gpu"

TARGET_LINEAR_LAYERS = [1, 5, 9, 13]
SOURCE_FFN_LAYERS = [layer - 1 for layer in TARGET_LINEAR_LAYERS]
DECODE_TOKENS = 8
EXPECTED_EVENTS = len(TARGET_LINEAR_LAYERS) * DECODE_TOKENS
EXPECTED_COUNTER_LAYERS = 72
EXPECTED_COUNTER_VALUES = 147456


def _load_base() -> Any:
  spec = importlib.util.spec_from_file_location(
      "iq36_seq434_producer_linear_input_source_base", BASE_GATE)
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

  inner = getattr(BASE, "BASE", None)
  if inner is not None:
    inner.TARGET_LINEAR_LAYERS = TARGET_LINEAR_LAYERS
    inner.SOURCE_FFN_LAYERS = SOURCE_FFN_LAYERS
    inner.EXPECTED_EVENTS = EXPECTED_EVENTS
    inner.EXPECTED_COUNTER_LAYERS = EXPECTED_COUNTER_LAYERS
    inner.EXPECTED_COUNTER_VALUES = EXPECTED_COUNTER_VALUES

  original_has_candidate = BASE._has_candidate
  original_has_switch = BASE._has_switch

  def has_candidate(routes: dict[str, Any], seq: int,
                    disposition: str) -> bool:
    if seq == 418:
      return original_has_candidate(routes, 433, disposition)
    return original_has_candidate(routes, seq, disposition)

  def has_switch(routes: dict[str, Any], decision: str,
                 seq_covered: int) -> bool:
    if seq_covered == 418:
      return original_has_switch(routes, decision, 433)
    return original_has_switch(routes, decision, seq_covered)

  BASE._has_candidate = has_candidate
  BASE._has_switch = has_switch


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--routes", type=Path, default=DEFAULT_ROUTES)
  parser.add_argument("--seq431", type=Path, default=DEFAULT_SEQ431)
  parser.add_argument("--seq433", type=Path, default=DEFAULT_SEQ433)
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
  base_args.seq416 = args.seq431
  base_args.seq418 = args.seq433
  metrics = BASE.compute(base_args)
  metrics["inputs"]["seq431"] = metrics["inputs"].pop("seq416")
  metrics["inputs"]["seq433"] = metrics["inputs"].pop("seq418")
  for row in metrics.get("checks", []):
    if row.get("name") == "seq418_selected_deeper_nested_producer_linear_input_source_gate":
      row["name"] = "seq433_selected_deeper_nested_producer_linear_input_source_gate"
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
