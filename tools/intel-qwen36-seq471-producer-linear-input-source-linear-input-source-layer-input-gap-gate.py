#!/usr/bin/env python3
"""Locate seq470 upstream drift feeding source layer inputs."""

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
    "intel-qwen36-seq471-producer-linear-input-source-linear-input-source-"
    "layer-input-gap-gate-v0"
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
    "_linear_input_source_producer_linear_input_source_linear_input_source_"
    "layer_input_gap_gate")
OUTPUT_ROUTE = CURRENT_ROUTE.replace(
    "_layer_input_gap_gate", "_layer_input_preceding_linear_output_gap_gate")
MATH_ROUTE = CURRENT_ROUTE.replace(
    "_layer_input_gap_gate", "_layer_input_preceding_linear_math_gap_gate")
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
DEFAULT_SEQ469 = (
    ROOT
    / "output/seq469-producer-linear-input-source-linear-input-source-gap-gate-20260709Tseq469Z"
    / "metrics.json"
)
DEFAULT_SEQ470 = (
    ROOT
    / "output/seq470-producer-linear-input-source-linear-input-source-ffn-input-gap-gate-20260709Tseq470Z"
    / "metrics.json"
)
DEFAULT_TOKEN_INPUT_DIR = (
    ROOT
    / "output/r2-gpu-acceptance-matrix-router-both-rowblock16-26mask-20260708Tseq222Z"
    / "token-input"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "output/seq471-producer-linear-input-source-linear-input-source-layer-input-gap-gate-20260709Tseq471Z"
)
DEFAULT_HOST = "local"
DEFAULT_MODEL = "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf"
DEFAULT_ENV_SCRIPT = "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh"
DEFAULT_REMOTE_ROOT = "/home/intel/intel-qwen36-gpu"

TARGET_LAYERS = [3]
PRECEDING_LINEAR_LAYERS = [layer - 1 for layer in TARGET_LAYERS]
DECODE_TOKENS = 8
EXPECTED_EVENTS = len(TARGET_LAYERS) * DECODE_TOKENS
EXPECTED_COUNTER_LAYERS = 72
EXPECTED_COUNTER_VALUES = 147456
COSINE_THRESHOLD = 0.9999


def _load_base() -> Any:
  spec = importlib.util.spec_from_file_location(
      "iq36_seq471_producer_linear_input_source_source_layer_input_base",
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


def _num(value: Any, default: float = 0.0) -> float:
  return float(value) if isinstance(value, (int, float)) else default


def _first_gap(current: dict[str, Any] | None,
               token_index: Any,
               layer: int,
               name: str,
               cosine: float,
               max_abs: float) -> dict[str, Any] | None:
  if current is not None or cosine >= COSINE_THRESHOLD:
    return current
  return {
      "token_index": token_index,
      "layer": layer,
      f"{name}_cosine": cosine,
      f"{name}_max_abs_diff": max_abs,
  }


def _layer_boundary_summary(smoke: dict[str, Any]) -> dict[str, Any]:
  steps = smoke.get("layer_boundary_diff_by_step")
  steps = steps if isinstance(steps, list) else []
  previous_layers = [layer - 1 for layer in TARGET_LAYERS if layer > 0]

  target_input_obs = 0
  previous_output_obs = 0
  min_target_input = 1.0
  min_previous_output = 1.0
  max_target_input_abs = 0.0
  max_previous_output_abs = 0.0
  first_target_input_gap: dict[str, Any] | None = None
  first_previous_output_gap: dict[str, Any] | None = None
  for step in steps:
    if not isinstance(step, dict):
      continue
    token_index = step.get("token_index")
    layers = step.get("layers")
    layers = layers if isinstance(layers, list) else []
    for row in layers:
      if not isinstance(row, dict):
        continue
      layer = row.get("layer")
      if not isinstance(layer, int):
        continue
      if layer in TARGET_LAYERS:
        cosine = _num(row.get("input_cosine"), 1.0)
        max_abs = _num(row.get("input_max_abs_diff"))
        target_input_obs += 1
        min_target_input = min(min_target_input, cosine)
        max_target_input_abs = max(max_target_input_abs, max_abs)
        first_target_input_gap = _first_gap(
            first_target_input_gap, token_index, layer, "target_input",
            cosine, max_abs)
      if layer in previous_layers:
        cosine = _num(row.get("output_cosine"), 1.0)
        max_abs = _num(row.get("output_max_abs_diff"))
        previous_output_obs += 1
        min_previous_output = min(min_previous_output, cosine)
        max_previous_output_abs = max(max_previous_output_abs, max_abs)
        first_previous_output_gap = _first_gap(
            first_previous_output_gap, token_index, layer, "previous_output",
            cosine, max_abs)

  return {
      "target_layers": TARGET_LAYERS,
      "previous_layers": previous_layers,
      "expected_observation_count": EXPECTED_EVENTS,
      "target_input_observation_count": target_input_obs,
      "previous_output_observation_count": previous_output_obs,
      "min_target_input_cosine": min_target_input,
      "max_target_input_abs_diff": max_target_input_abs,
      "min_previous_output_cosine": min_previous_output,
      "max_previous_output_abs_diff": max_previous_output_abs,
      "first_target_input_gap": first_target_input_gap,
      "first_previous_output_gap": first_previous_output_gap,
  }


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

  original_case_summary = BASE.OLD._case_summary

  def case_summary(case_id: str, run: dict[str, Any],
                   smoke: dict[str, Any]) -> dict[str, Any]:
    summary = original_case_summary(case_id, run, smoke)
    summary["layer_boundary"] = _layer_boundary_summary(smoke)
    return summary

  BASE.OLD._case_summary = case_summary

  original_has_candidate = BASE._has_candidate
  original_has_switch = BASE._has_switch

  def has_candidate(routes: dict[str, Any], seq: int,
                    disposition: str) -> bool:
    if seq == 410:
      return original_has_candidate(routes, 470, disposition)
    return original_has_candidate(routes, seq, disposition)

  def has_switch(routes: dict[str, Any], decision: str,
                 seq_covered: int) -> bool:
    if seq_covered == 410:
      return original_has_switch(routes, decision, 470)
    return original_has_switch(routes, decision, seq_covered)

  BASE._has_candidate = has_candidate
  BASE._has_switch = has_switch


def _postprocess_boundary_classification(metrics: dict[str, Any]) -> None:
  if metrics.get("required_checks_passed") is True:
    return
  if not str(metrics.get("diagnostic_classification", "")).endswith(
      "_layer_input_gap_unclassified"):
    return

  boundaries = [
      row.get("summary", {}).get("layer_boundary", {})
      for row in metrics.get("runs", [])
      if isinstance(row, dict)
  ]
  boundary_emitted = (
      len(boundaries) == 2
      and all(row.get("target_input_observation_count") == EXPECTED_EVENTS
              for row in boundaries)
      and all(row.get("previous_output_observation_count") == EXPECTED_EVENTS
              for row in boundaries)
  )
  min_boundary_input = min(
      (_num(row.get("min_target_input_cosine"), 1.0) for row in boundaries),
      default=1.0)
  min_previous_output = min(
      (_num(row.get("min_previous_output_cosine"), 1.0) for row in boundaries),
      default=1.0)
  boundary_input_gap = boundary_emitted and min_boundary_input < COSINE_THRESHOLD
  previous_output_gap = (
      boundary_emitted and min_previous_output < COSINE_THRESHOLD)
  preceding_output_clean = (
      metrics.get("preceding_output_gap") is False
      and _num(metrics.get("min_preceding_final_output_cosine"), 1.0)
      >= COSINE_THRESHOLD)
  preceding_norm_clean = (
      metrics.get("preceding_norm_inherits_input") is False
      and _num(metrics.get("min_preceding_attn_norm_from_gpu_input_cosine"), 1.0)
      >= COSINE_THRESHOLD)
  if not (
      metrics.get("target_layer_input_gap") is True
      and boundary_input_gap
      and preceding_output_clean
      and preceding_norm_clean
      and metrics.get("preceding_norm_math_ok") is True
  ):
    return

  if previous_output_gap:
    classification = (
        f"{DIAG_PREFIX}_linear_input_source_linear_input_source_"
        "layer_input_upstream_layer_output_gap"
    )
    selected_next = CURRENT_ROUTE.replace(
        "_layer_input_gap_gate",
        "_layer_input_upstream_layer_output_gap_gate")
    reason = (
        "Layer-input drift remains at the layer boundary while preceding "
        "linear output/norm are clean; previous layer output carries the "
        "same gap. Root previous layer output next."
    )
  else:
    classification = (
        f"{DIAG_PREFIX}_linear_input_source_linear_input_source_"
        "layer_input_boundary_handoff_gap"
    )
    selected_next = CURRENT_ROUTE.replace(
        "_layer_input_gap_gate",
        "_layer_input_boundary_handoff_gap_gate")
    reason = (
        "Layer-input drift appears at the layer boundary while both preceding "
        "linear output/norm and previous layer output are clean. Root the "
        "boundary handoff next."
    )

  checks = []
  for row in metrics.get("checks", []):
    if row.get("name") == "preceding_linear_output_gap_reproduced":
      checks.append({
          "name": "preceding_linear_output_clean_for_boundary_split",
          "pass": preceding_output_clean,
          "detail": row.get("detail"),
      })
    elif row.get("name") == "preceding_linear_norm_inherits_input":
      checks.append({
          "name": "preceding_linear_norm_clean_for_boundary_split",
          "pass": preceding_norm_clean,
          "detail": row.get("detail"),
      })
    else:
      checks.append(row)
  checks.extend([
      {
          "name": "layer_boundary_diagnostics_emitted",
          "pass": boundary_emitted,
          "detail": boundaries,
      },
      {
          "name": "layer_boundary_input_gap_reproduced",
          "pass": boundary_input_gap,
          "detail": {
              "min_boundary_target_input_cosine": min_boundary_input,
              "min_target_layer_input_cosine": metrics.get(
                  "min_target_layer_input_cosine"),
          },
      },
      {
          "name": (
              "previous_layer_output_gap_reproduced"
              if previous_output_gap else
              "previous_layer_output_clean_for_boundary_handoff"
          ),
          "pass": True,
          "detail": {
              "min_previous_layer_output_cosine": min_previous_output,
          },
      },
  ])

  metrics["checks"] = checks
  metrics["diagnostic_classification"] = classification
  metrics["disposition"] = (
      f"accept_{DISPOSITION_PREFIX}_linear_input_source_linear_input_source_"
      "layer_input_gap_classification"
  )
  metrics["selected_next_route"] = selected_next
  metrics["next_route_reason"] = reason
  metrics["min_boundary_target_input_cosine"] = min_boundary_input
  metrics["min_previous_layer_output_cosine"] = min_previous_output
  metrics["layer_boundary_input_gap"] = boundary_input_gap
  metrics["previous_layer_output_gap"] = previous_output_gap
  metrics["required_checks_passed"] = all(
      row.get("pass") is True for row in checks)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--routes", type=Path, default=DEFAULT_ROUTES)
  parser.add_argument("--seq469", type=Path, default=DEFAULT_SEQ469)
  parser.add_argument("--seq470", type=Path, default=DEFAULT_SEQ470)
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
  base_args.seq409 = args.seq469
  base_args.seq410 = args.seq470
  metrics = BASE.compute(base_args)
  _postprocess_boundary_classification(metrics)
  metrics["inputs"]["seq469"] = metrics["inputs"].pop("seq409")
  metrics["inputs"]["seq470"] = metrics["inputs"].pop("seq410")
  for row in metrics.get("checks", []):
    if row.get("name") == "seq410_selected_deeper_nested_source_layer_input_gap_gate":
      row["name"] = "seq470_selected_deeper_nested_source_layer_input_gap_gate"
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
