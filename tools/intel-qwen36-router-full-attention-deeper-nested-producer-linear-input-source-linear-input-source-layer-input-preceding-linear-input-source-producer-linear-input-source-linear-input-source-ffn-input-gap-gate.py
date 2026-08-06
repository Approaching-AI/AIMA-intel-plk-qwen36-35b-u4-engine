#!/usr/bin/env python3
"""Classify FFN input drift feeding deeper nested producer input-source linear input source."""

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
    / "tools/intel-qwen36-router-full-attention-deeper-nested-"
    "producer-linear-input-source-linear-input-source-ffn-input-gap-gate.py"
)

SCHEMA_VERSION = (
    "intel-qwen36-router-full-attention-deeper-nested-producer-linear-"
    "input-source-linear-input-source-layer-input-preceding-linear-"
    "input-source-producer-linear-input-source-linear-input-source-"
    "ffn-input-gap-gate-v0"
)
ROUTE_PREFIX = (
    "router_prompt_full_attention_layer_input_preceding_linear_input_source_"
    "producer_linear_input_source_linear_input_source_layer_input_preceding_"
    "linear_input_source_producer_linear_input_source_linear_input_source_"
    "layer_input_preceding_linear_input_source_producer_linear_input_source_"
    "linear_input_source_layer_input_preceding_linear_input_source"
)
CURRENT_ROUTE = (
    f"{ROUTE_PREFIX}_producer_linear_input_source_linear_input_source_layer_input_preceding_linear_input_source_producer_linear_input_source_linear_input_source_ffn_input_gap_gate"
)
LAYER_INPUT_ROUTE = (
    f"{ROUTE_PREFIX}_producer_linear_input_source_linear_input_source_layer_input_preceding_linear_input_source_producer_linear_input_source_linear_input_source_layer_input_gap_gate"
)
ATTENTION_OUTPUT_ROUTE = (
    f"{ROUTE_PREFIX}_producer_linear_input_source_linear_input_source_layer_input_preceding_linear_input_source_producer_linear_input_source_linear_input_source_attention_output_gap_gate"
)
MATH_ROUTE = (
    f"{ROUTE_PREFIX}_producer_linear_input_source_linear_input_source_layer_input_preceding_linear_input_source_producer_linear_input_source_linear_input_source_ffn_norm_math_gap_gate"
)

DISPOSITION_PREFIX = (
    "source_layer_input_preceding_linear_input_source_producer_linear_input_"
    "source_linear_input_source_layer_input_preceding_linear_input_source_"
    "producer_linear_input_source_linear_input_source_layer_input_preceding_"
    "linear_input_source_producer_linear_input_source_linear_input_source_"
    "layer_input_preceding_linear_input_source"
)
DIAG_PREFIX = f"{DISPOSITION_PREFIX}_producer"

DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_SEQ424 = (
    ROOT
    / "output/router-full-attention-deeper-nested-producer-linear-input-source-linear-input-source-layer-input-preceding-linear-input-source-producer-linear-input-source-linear-input-source-gap-gate-20260709Tseq424Z"
    / "metrics.json"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "output/router-full-attention-deeper-nested-producer-linear-input-source-linear-input-source-layer-input-preceding-linear-input-source-producer-linear-input-source-linear-input-source-ffn-input-gap-gate-20260709Tseq425Z"
)


def _load_base() -> Any:
  spec = importlib.util.spec_from_file_location(
      "iq36_deeper_nested_current_linear_input_source_ffn_base", BASE_GATE)
  if spec is None or spec.loader is None:
    raise SystemExit(f"cannot load base FFN input gate: {BASE_GATE}")
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
  BASE.LAYER_INPUT_ROUTE = LAYER_INPUT_ROUTE
  BASE.ATTENTION_OUTPUT_ROUTE = ATTENTION_OUTPUT_ROUTE
  BASE.MATH_ROUTE = MATH_ROUTE
  BASE.DISPOSITION_PREFIX = DISPOSITION_PREFIX
  BASE.DIAG_PREFIX = DIAG_PREFIX

  original_has_candidate = BASE._has_candidate
  original_has_switch = BASE._has_switch

  def has_candidate(routes: dict[str, Any], seq: int,
                    disposition: str) -> bool:
    if seq == 409:
      return original_has_candidate(routes, 424, disposition)
    return original_has_candidate(routes, seq, disposition)

  def has_switch(routes: dict[str, Any], decision: str,
                 seq_covered: int) -> bool:
    if seq_covered == 409:
      return original_has_switch(routes, decision, 424)
    return original_has_switch(routes, decision, seq_covered)

  BASE._has_candidate = has_candidate
  BASE._has_switch = has_switch


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--routes", type=Path, default=DEFAULT_ROUTES)
  parser.add_argument("--seq424", type=Path, default=DEFAULT_SEQ424)
  parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
  return parser.parse_args()


def main() -> int:
  _patch_base()
  args = parse_args()
  base_args = argparse.Namespace(
      routes=args.routes,
      seq409=args.seq424,
      out_dir=args.out_dir,
  )
  metrics = BASE.compute(base_args)
  metrics["inputs"]["seq424"] = metrics["inputs"].pop("seq409")
  for row in metrics.get("checks", []):
    if row.get("name") == "seq409_selected_deeper_nested_source_ffn_input_gap_gate":
      row["name"] = "seq424_selected_deeper_nested_source_ffn_input_gap_gate"
    if row.get("name") == "seq409_run_summaries_available":
      row["name"] = "seq424_run_summaries_available"
    if row.get("name") == "seq409_nested_metrics_match_top_level":
      row["name"] = "seq424_nested_metrics_match_top_level"
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
