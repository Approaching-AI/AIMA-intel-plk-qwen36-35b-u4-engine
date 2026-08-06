#!/usr/bin/env python3
"""Classify seq467 producer linear input-source source linear input drift."""

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
    "linear-input-source-linear-input-gap-gate.py"
)

SCHEMA_VERSION = (
    "intel-qwen36-seq468-producer-linear-input-source-linear-input-gap-gate-v0"
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
ATTENTION_ROUTE = SOURCE_GAP_ROUTE.replace(
    "_linear_input_source_gap_gate",
    "_linear_input_source_producer_linear_input_source_attention_output_gap_gate")
CURRENT_ROUTE = ATTENTION_ROUTE.replace(
    "_producer_linear_input_source_attention_output_gap_gate",
    "_producer_linear_input_source_linear_input_gap_gate")
NEXT_ROUTE = CURRENT_ROUTE.replace(
    "_linear_input_gap_gate", "_linear_input_source_gap_gate")
MATH_ROUTE = CURRENT_ROUTE.replace(
    "_linear_input_gap_gate", "_linear_input_preconv_math_gap_gate")
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
DEFAULT_SEQ466 = (
    ROOT
    / "output/seq466-producer-linear-input-source-attention-output-gap-gate-20260709Tseq466Z"
    / "metrics.json"
)
DEFAULT_SEQ467 = (
    ROOT
    / "output/seq467-producer-linear-input-source-linear-delta-z-gap-gate-20260709Tseq467Z"
    / "metrics.json"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "output/seq468-producer-linear-input-source-linear-input-gap-gate-20260709Tseq468Z"
)


def _load_base() -> Any:
  spec = importlib.util.spec_from_file_location(
      "iq36_seq468_producer_linear_input_source_source_linear_input_base",
      BASE_GATE)
  if spec is None or spec.loader is None:
    raise SystemExit(f"cannot load base source linear input gate: {BASE_GATE}")
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

  original_has_candidate = BASE._has_candidate
  original_has_switch = BASE._has_switch

  def has_candidate(routes: dict[str, Any], seq: int,
                    disposition: str) -> bool:
    if seq == 407:
      return original_has_candidate(routes, 467, disposition)
    return original_has_candidate(routes, seq, disposition)

  def has_switch(routes: dict[str, Any], decision: str,
                 seq_covered: int) -> bool:
    if seq_covered == 407:
      return original_has_switch(routes, decision, 467)
    return original_has_switch(routes, decision, seq_covered)

  BASE._has_candidate = has_candidate
  BASE._has_switch = has_switch


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--routes", type=Path, default=DEFAULT_ROUTES)
  parser.add_argument("--seq466", type=Path, default=DEFAULT_SEQ466)
  parser.add_argument("--seq467", type=Path, default=DEFAULT_SEQ467)
  parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
  return parser.parse_args()


def main() -> int:
  _patch_base()
  args = parse_args()
  base_args = argparse.Namespace(
      routes=args.routes,
      seq406=args.seq466,
      seq407=args.seq467,
      out_dir=args.out_dir,
  )
  metrics = BASE.compute(base_args)
  metrics["inputs"]["seq466"] = metrics["inputs"].pop("seq406")
  metrics["inputs"]["seq467"] = metrics["inputs"].pop("seq407")
  for row in metrics.get("checks", []):
    if row.get("name") == "seq407_selected_deeper_nested_source_linear_input_gap_gate":
      row["name"] = "seq467_selected_deeper_nested_source_linear_input_gap_gate"
    if row.get("name") == "seq406_run_summaries_available":
      row["name"] = "seq466_run_summaries_available"
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
