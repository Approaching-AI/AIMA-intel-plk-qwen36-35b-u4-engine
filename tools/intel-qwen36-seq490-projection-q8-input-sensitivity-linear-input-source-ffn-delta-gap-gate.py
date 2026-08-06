#!/usr/bin/env python3
"""Classify seq489 Q8-sensitive previous FFN-delta source material drift."""

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
SEQ489_GATE = (
    ROOT
    / "tools/intel-qwen36-seq489-projection-q8-input-sensitivity-linear-input-source-gap-gate.py"
)
SCHEMA_VERSION = (
    "intel-qwen36-seq490-projection-q8-input-sensitivity-linear-input-source-ffn-delta-gap-gate-v0"
)

DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_SEQ489 = (
    ROOT
    / "output/seq489-projection-q8-input-sensitivity-linear-input-source-gap-gate-20260709Tseq489Z"
    / "metrics.json"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "output/seq490-projection-q8-input-sensitivity-linear-input-source-ffn-delta-gap-gate-20260709Tseq490Z"
)

COSINE_THRESHOLD = 0.9999
MATERIAL_ABS_EPS = 1.0e-8
SOURCE_REPLAY_EPS = 5.0e-7


def _load_module(path: Path, name: str) -> Any:
  spec = importlib.util.spec_from_file_location(name, path)
  if spec is None or spec.loader is None:
    raise SystemExit(f"cannot load module {name}: {path}")
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


SEQ489 = _load_module(SEQ489_GATE, "iq36_seq489_gate")
CURRENT_ROUTE = SEQ489.FFN_DELTA_ROUTE
FFN_NORM_ROUTE = CURRENT_ROUTE.replace(
    "_linear_input_source_ffn_delta_gap_gate",
    "_linear_input_source_ffn_delta_ffn_norm_gap_gate")
ROUTER_ROUTE = CURRENT_ROUTE.replace(
    "_linear_input_source_ffn_delta_gap_gate",
    "_linear_input_source_ffn_delta_router_gap_gate")
SELECTED_ROUTE = CURRENT_ROUTE.replace(
    "_linear_input_source_ffn_delta_gap_gate",
    "_linear_input_source_ffn_delta_selected_gap_gate")
SHARED_ROUTE = CURRENT_ROUTE.replace(
    "_linear_input_source_ffn_delta_gap_gate",
    "_linear_input_source_ffn_delta_shared_gap_gate")
DIAG_PREFIX = SEQ489.DIAG_PREFIX
DISPOSITION_PREFIX = SEQ489.DISPOSITION_PREFIX
EXPECTED_CASES = SEQ489.EXPECTED_CASES
EXPECTED_EVENTS = SEQ489.EXPECTED_EVENTS

COMPONENT_NAMES = [
    "ffn_input",
    "ffn_norm",
    "router_logits",
    "router_weights",
    "selected_gate_up",
    "selected_swiglu",
    "selected_down",
    "weighted_selected_down",
    "moe_out",
    "shared_gate",
    "shared_gate_sigmoid",
    "shared_gate_up",
    "shared_swiglu",
    "shared_down",
    "shared_gated",
    "ffn_out",
    "residual",
]


def _load_json(path: Path) -> Any:
  return json.loads(path.read_text(encoding="utf-8"))


def _num(value: Any, default: float = 0.0) -> float:
  return float(value) if isinstance(value, (int, float)) else default


def _rel(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path)


def _has_candidate(routes: dict[str, Any], seq: int, disposition: str) -> bool:
  return any(
      isinstance(row, dict)
      and row.get("seq") == seq
      and row.get("disposition") == disposition
      for row in routes.get("candidate_history", [])
  )


def _has_switch(routes: dict[str, Any], decision: str, seq_covered: int) -> bool:
  return any(
      isinstance(row, dict)
      and row.get("decision") == decision
      and _num(row.get("seq_covered")) >= seq_covered
      and row.get("resolved") is True
      for row in routes.get("switch_decisions", [])
  )


def _load_smoke(row: dict[str, Any]) -> dict[str, Any]:
  out_dir = row.get("out_dir")
  if not isinstance(out_dir, str):
    return {}
  smoke_path = ROOT / out_dir / "smoke.json"
  if not smoke_path.exists():
    return {}
  smoke = _load_json(smoke_path)
  return smoke if isinstance(smoke, dict) else {}


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


def _first_material(current: dict[str, Any] | None,
                    token_index: Any,
                    layer: int,
                    name: str,
                    cosine: float,
                    max_abs: float) -> dict[str, Any] | None:
  if current is not None or cosine >= 1.0 or max_abs <= MATERIAL_ABS_EPS:
    return current
  return {
      "token_index": token_index,
      "layer": layer,
      f"{name}_cosine": cosine,
      f"{name}_max_abs_diff": max_abs,
  }


def _source_layers(row: dict[str, Any]) -> list[int]:
  layers = row.get("source_layers")
  layers = layers if isinstance(layers, list) else []
  return [
      int(layer) for layer in layers
      if isinstance(layer, int) and layer >= 0
  ]


def _component_summary(row: dict[str, Any]) -> dict[str, Any]:
  smoke = _load_smoke(row)
  steps = smoke.get("ffn_component_source_diff_by_step")
  steps = steps if isinstance(steps, list) else []
  mins = {name: 1.0 for name in COMPONENT_NAMES}
  max_abs = {name: 0.0 for name in COMPONENT_NAMES}
  first_gap = {name: None for name in COMPONENT_NAMES}
  first_material = {name: None for name in COMPONENT_NAMES}
  obs = 0
  router_id_mismatch_count = 0
  max_router_weight_abs_diff = 0.0
  source_layers = set(_source_layers(row))

  for step in steps:
    if not isinstance(step, dict):
      continue
    token_index = step.get("token_index")
    layers = step.get("layers")
    layers = layers if isinstance(layers, list) else []
    for component in layers:
      if not isinstance(component, dict):
        continue
      layer = component.get("layer")
      if layer not in source_layers:
        continue
      if component.get("ffn_input_available") is True:
        obs += 1
      if component.get("router_ids_match") is False:
        router_id_mismatch_count += 1
      max_router_weight_abs_diff = max(
          max_router_weight_abs_diff,
          _num(component.get("router_weight_max_abs_diff")))
      for name in COMPONENT_NAMES:
        if component.get(f"{name}_available") is not True:
          continue
        cosine = _num(component.get(f"{name}_cosine"), 1.0)
        diff = _num(component.get(f"{name}_max_abs_diff"))
        mins[name] = min(mins[name], cosine)
        max_abs[name] = max(max_abs[name], diff)
        first_gap[name] = _first_gap(first_gap[name], token_index, int(layer),
                                     name, cosine, diff)
        first_material[name] = _first_material(
            first_material[name], token_index, int(layer), name, cosine, diff)

  summary: dict[str, Any] = {
      "case_id": row.get("case_id"),
      "out_dir": row.get("out_dir"),
      "source_layers": sorted(source_layers),
      "expected_observation_count": EXPECTED_EVENTS,
      "observation_count": obs,
      "router_id_mismatch_count": router_id_mismatch_count,
      "max_router_weight_abs_diff": max_router_weight_abs_diff,
  }
  for name in COMPONENT_NAMES:
    summary[f"min_{name}_cosine"] = mins[name]
    summary[f"max_{name}_abs_diff"] = max_abs[name]
    summary[f"first_{name}_gap"] = first_gap[name]
    summary[f"first_{name}_material"] = first_material[name]
  return summary


def _min_case(rows: list[dict[str, Any]], key: str) -> float:
  return min((_num(row.get(key), 1.0) for row in rows), default=1.0)


def _max_case(rows: list[dict[str, Any]], key: str) -> float:
  return max((_num(row.get(key)) for row in rows), default=0.0)


def _material(min_cosine: float, max_abs: float) -> bool:
  return min_cosine < 1.0 and max_abs > MATERIAL_ABS_EPS


def compute(args: argparse.Namespace) -> dict[str, Any]:
  routes = _load_json(args.routes)
  seq489 = _load_json(args.seq489)
  case_rows = seq489.get("runs")
  case_rows = case_rows if isinstance(case_rows, list) else []
  rows = [_component_summary(row) for row in case_rows if isinstance(row, dict)]

  preconditions_pass = (
      seq489.get("required_checks_passed") is True
      and seq489.get("selected_next_route") == CURRENT_ROUTE
      and seq489.get("diagnostic_classification")
      == f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_input_attention_output_projection_q8_input_sensitivity_linear_input_source_ffn_delta_selected_swiglu_gate_up_input_sensitivity_ffn_norm_input_sensitivity_attention_output_projection_q8_input_sensitivity_linear_input_source_ffn_delta_gap"
      and _has_candidate(routes, 489, str(seq489.get("disposition")))
      and _has_switch(routes, f"select_{CURRENT_ROUTE}", 489)
  )
  rows_loaded = (
      len(rows) == len(EXPECTED_CASES)
      and all(row.get("observation_count") == EXPECTED_EVENTS for row in rows)
  )
  seq489_ffn_delta_source_gap = (
      seq489.get("ffn_delta_source_gap") is True
      and seq489.get("attention_output_source_gap") is False
      and seq489.get("ffn_math_gap") is False)

  min_ffn_input = _min_case(rows, "min_ffn_input_cosine")
  min_ffn_norm = _min_case(rows, "min_ffn_norm_cosine")
  min_router_logits = _min_case(rows, "min_router_logits_cosine")
  min_router_weights = _min_case(rows, "min_router_weights_cosine")
  min_selected_gate_up = _min_case(rows, "min_selected_gate_up_cosine")
  min_selected_swiglu = _min_case(rows, "min_selected_swiglu_cosine")
  min_selected_down = _min_case(rows, "min_selected_down_cosine")
  min_weighted_selected_down = _min_case(
      rows, "min_weighted_selected_down_cosine")
  min_moe_out = _min_case(rows, "min_moe_out_cosine")
  min_shared_gate = _min_case(rows, "min_shared_gate_cosine")
  min_shared_gate_sigmoid = _min_case(rows, "min_shared_gate_sigmoid_cosine")
  min_shared_gate_up = _min_case(rows, "min_shared_gate_up_cosine")
  min_shared_swiglu = _min_case(rows, "min_shared_swiglu_cosine")
  min_shared_down = _min_case(rows, "min_shared_down_cosine")
  min_shared_gated = _min_case(rows, "min_shared_gated_cosine")
  min_ffn_out = _min_case(rows, "min_ffn_out_cosine")
  min_residual = _min_case(rows, "min_residual_cosine")

  max_ffn_input = _max_case(rows, "max_ffn_input_abs_diff")
  max_ffn_norm = _max_case(rows, "max_ffn_norm_abs_diff")
  max_router_logits = _max_case(rows, "max_router_logits_abs_diff")
  max_router_weights = _max_case(rows, "max_router_weights_abs_diff")
  max_selected_gate_up = _max_case(rows, "max_selected_gate_up_abs_diff")
  max_selected_swiglu = _max_case(rows, "max_selected_swiglu_abs_diff")
  max_selected_down = _max_case(rows, "max_selected_down_abs_diff")
  max_weighted_selected_down = _max_case(
      rows, "max_weighted_selected_down_abs_diff")
  max_moe_out = _max_case(rows, "max_moe_out_abs_diff")
  max_shared_gate = _max_case(rows, "max_shared_gate_abs_diff")
  max_shared_gate_sigmoid = _max_case(rows, "max_shared_gate_sigmoid_abs_diff")
  max_shared_gate_up = _max_case(rows, "max_shared_gate_up_abs_diff")
  max_shared_swiglu = _max_case(rows, "max_shared_swiglu_abs_diff")
  max_shared_down = _max_case(rows, "max_shared_down_abs_diff")
  max_shared_gated = _max_case(rows, "max_shared_gated_abs_diff")
  max_ffn_out = _max_case(rows, "max_ffn_out_abs_diff")
  max_residual = _max_case(rows, "max_residual_abs_diff")
  router_id_mismatch_count = sum(
      int(row.get("router_id_mismatch_count", 0)) for row in rows)

  seq489_min = seq489.get("min_cosines", {})
  seq489_abs = seq489.get("max_abs_diffs", {})
  source_delta_min = _num(seq489_min.get("source_ffn_delta"), 1.0)
  source_delta_max = _num(seq489_abs.get("source_ffn_delta"))
  source_component_cos_delta = abs(source_delta_min - min_ffn_out)
  source_component_abs_delta = abs(source_delta_max - max_ffn_out)
  seq489_component_reproduced = (
      rows_loaded
      and source_component_cos_delta <= SOURCE_REPLAY_EPS
      and source_component_abs_delta <= SOURCE_REPLAY_EPS)

  ffn_delta_material = (
      rows_loaded
      and _material(min_ffn_out, max_ffn_out)
      and _material(min_residual, max_residual))
  ffn_input_clean = rows_loaded and min_ffn_input >= COSINE_THRESHOLD
  ffn_norm_gap = rows_loaded and min_ffn_norm < COSINE_THRESHOLD
  ffn_norm_clean = rows_loaded and min_ffn_norm >= COSINE_THRESHOLD
  router_gap = (
      rows_loaded
      and (router_id_mismatch_count > 0
           or min_router_logits < COSINE_THRESHOLD
           or min_router_weights < COSINE_THRESHOLD))
  router_clean = (
      rows_loaded
      and router_id_mismatch_count == 0
      and min_router_logits >= COSINE_THRESHOLD
      and min_router_weights >= COSINE_THRESHOLD)
  selected_material = (
      rows_loaded
      and any([
          _material(min_selected_gate_up, max_selected_gate_up),
          _material(min_selected_swiglu, max_selected_swiglu),
          _material(min_selected_down, max_selected_down),
          _material(min_weighted_selected_down, max_weighted_selected_down),
          _material(min_moe_out, max_moe_out),
      ]))
  selected_downstream_material = (
      rows_loaded
      and _material(min_selected_down, max_selected_down)
      and _material(min_weighted_selected_down, max_weighted_selected_down)
      and _material(min_moe_out, max_moe_out))
  shared_material = (
      rows_loaded
      and any([
          _material(min_shared_gate, max_shared_gate),
          _material(min_shared_gate_sigmoid, max_shared_gate_sigmoid),
          _material(min_shared_gate_up, max_shared_gate_up),
          _material(min_shared_swiglu, max_shared_swiglu),
          _material(min_shared_down, max_shared_down),
          _material(min_shared_gated, max_shared_gated),
      ]))

  diagnostic_classification = (
      f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_input_attention_output_projection_q8_input_sensitivity_linear_input_source_ffn_delta_selected_swiglu_gate_up_input_sensitivity_ffn_norm_input_sensitivity_attention_output_projection_q8_input_sensitivity_linear_input_source_ffn_delta_ffn_norm_gap"
      if ffn_delta_material and ffn_input_clean and ffn_norm_gap else
      f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_input_attention_output_projection_q8_input_sensitivity_linear_input_source_ffn_delta_selected_swiglu_gate_up_input_sensitivity_ffn_norm_input_sensitivity_attention_output_projection_q8_input_sensitivity_linear_input_source_ffn_delta_router_gap"
      if ffn_delta_material and ffn_norm_clean and router_gap else
      f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_input_attention_output_projection_q8_input_sensitivity_linear_input_source_ffn_delta_selected_swiglu_gate_up_input_sensitivity_ffn_norm_input_sensitivity_attention_output_projection_q8_input_sensitivity_linear_input_source_ffn_delta_selected_gap"
      if (ffn_delta_material and ffn_norm_clean and router_clean
          and selected_material and selected_downstream_material) else
      f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_input_attention_output_projection_q8_input_sensitivity_linear_input_source_ffn_delta_selected_swiglu_gate_up_input_sensitivity_ffn_norm_input_sensitivity_attention_output_projection_q8_input_sensitivity_linear_input_source_ffn_delta_shared_gap"
      if ffn_delta_material and ffn_norm_clean and router_clean and shared_material else
      f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_input_attention_output_projection_q8_input_sensitivity_linear_input_source_ffn_delta_gap_unclassified"
  )
  selected_next = (
      FFN_NORM_ROUTE
      if diagnostic_classification.endswith("_ffn_delta_ffn_norm_gap") else
      ROUTER_ROUTE
      if diagnostic_classification.endswith("_ffn_delta_router_gap") else
      SELECTED_ROUTE
      if diagnostic_classification.endswith("_ffn_delta_selected_gap") else
      SHARED_ROUTE
      if diagnostic_classification.endswith("_ffn_delta_shared_gap") else
      CURRENT_ROUTE
  )
  checks = [
      {"name": "seq489_selected_ffn_delta_gap_gate",
       "pass": preconditions_pass},
      {"name": "seq489_component_rows_loaded",
       "pass": rows_loaded,
       "detail": rows},
      {"name": "seq489_ffn_delta_source_gap_reproduced",
       "pass": seq489_ffn_delta_source_gap},
      {"name": "seq489_ffn_delta_matches_component_ffn_out",
       "pass": seq489_component_reproduced,
       "detail": {
           "source_delta_min_cosine": source_delta_min,
           "component_ffn_out_min_cosine": min_ffn_out,
           "source_component_cosine_delta": source_component_cos_delta,
           "source_delta_max_abs_diff": source_delta_max,
           "component_ffn_out_max_abs_diff": max_ffn_out,
           "source_component_abs_delta": source_component_abs_delta,
       }},
      {"name": "ffn_delta_component_material_reproduced",
       "pass": ffn_delta_material,
       "detail": {
           "min_ffn_out_cosine": min_ffn_out,
           "max_ffn_out_abs_diff": max_ffn_out,
           "min_residual_cosine": min_residual,
           "max_residual_abs_diff": max_residual,
       }},
      {"name": "ffn_prefix_clean_for_path_split",
       "pass": ffn_input_clean and ffn_norm_clean and router_clean,
       "detail": {
           "min_ffn_input_cosine": min_ffn_input,
           "min_ffn_norm_cosine": min_ffn_norm,
           "min_router_logits_cosine": min_router_logits,
           "min_router_weights_cosine": min_router_weights,
           "router_id_mismatch_count": router_id_mismatch_count,
       }},
      {"name": "ffn_delta_path_classified",
       "pass": selected_next != CURRENT_ROUTE,
       "detail": {
           "ffn_norm_gap": ffn_norm_gap,
           "router_gap": router_gap,
           "selected_material": selected_material,
           "selected_downstream_material": selected_downstream_material,
           "shared_material": shared_material,
       }},
  ]
  required = all(bool(row.get("pass")) for row in checks)
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "inputs": {
          "routes": _rel(args.routes),
          "seq489": _rel(args.seq489),
      },
      "runs": rows,
      "checks": checks,
      "required_checks_passed": required,
      "diagnostic_classification": diagnostic_classification,
      "ffn_delta_material": ffn_delta_material,
      "ffn_input_clean": ffn_input_clean,
      "ffn_norm_gap": ffn_norm_gap,
      "ffn_norm_clean": ffn_norm_clean,
      "router_gap": router_gap,
      "router_clean": router_clean,
      "selected_material": selected_material,
      "selected_downstream_material": selected_downstream_material,
      "shared_material": shared_material,
      "router_id_mismatch_count": router_id_mismatch_count,
      "source_component_match": {
          "source_component_cosine_delta": source_component_cos_delta,
          "source_component_abs_delta": source_component_abs_delta,
      },
      "min_cosines": {
          "ffn_input": min_ffn_input,
          "ffn_norm": min_ffn_norm,
          "router_logits": min_router_logits,
          "router_weights": min_router_weights,
          "selected_gate_up": min_selected_gate_up,
          "selected_swiglu": min_selected_swiglu,
          "selected_down": min_selected_down,
          "weighted_selected_down": min_weighted_selected_down,
          "moe_out": min_moe_out,
          "shared_gate": min_shared_gate,
          "shared_gate_sigmoid": min_shared_gate_sigmoid,
          "shared_gate_up": min_shared_gate_up,
          "shared_swiglu": min_shared_swiglu,
          "shared_down": min_shared_down,
          "shared_gated": min_shared_gated,
          "ffn_out": min_ffn_out,
          "residual": min_residual,
      },
      "max_abs_diffs": {
          "ffn_input": max_ffn_input,
          "ffn_norm": max_ffn_norm,
          "router_logits": max_router_logits,
          "router_weights": max_router_weights,
          "selected_gate_up": max_selected_gate_up,
          "selected_swiglu": max_selected_swiglu,
          "selected_down": max_selected_down,
          "weighted_selected_down": max_weighted_selected_down,
          "moe_out": max_moe_out,
          "shared_gate": max_shared_gate,
          "shared_gate_sigmoid": max_shared_gate_sigmoid,
          "shared_gate_up": max_shared_gate_up,
          "shared_swiglu": max_shared_swiglu,
          "shared_down": max_shared_down,
          "shared_gated": max_shared_gated,
          "ffn_out": max_ffn_out,
          "residual": max_residual,
      },
      "speedup_claims_allowed": False,
      "disposition": (
          f"accept_{DISPOSITION_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_input_attention_output_projection_q8_input_sensitivity_linear_input_source_ffn_delta_selected_swiglu_gate_up_input_sensitivity_ffn_norm_input_sensitivity_attention_output_projection_q8_input_sensitivity_linear_input_source_ffn_delta_gap_classification"
          if required else
          f"block_{DISPOSITION_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_input_attention_output_projection_q8_input_sensitivity_linear_input_source_ffn_delta_selected_swiglu_gate_up_input_sensitivity_ffn_norm_input_sensitivity_attention_output_projection_q8_input_sensitivity_linear_input_source_ffn_delta_gap_classification"
      ),
      "selected_next_route": selected_next if required else CURRENT_ROUTE,
      "next_route_reason": (
          "The Q8-sensitive previous FFN delta starts at FFN norm while FFN "
          "input is clean. Root previous FFN norm next."
          if required and selected_next == FFN_NORM_ROUTE else
          "The Q8-sensitive previous FFN delta starts in router path values. "
          "Root previous FFN router next."
          if required and selected_next == ROUTER_ROUTE else
          "The Q8-sensitive previous FFN delta keeps FFN input, FFN norm, and "
          "router values clean at the cosine gate, then carries material drift "
          "through selected expert fan-out; shared fan-out drift is recorded. "
          "Root selected FFN components first."
          if required and selected_next == SELECTED_ROUTE else
          "The Q8-sensitive previous FFN delta keeps FFN input, FFN norm, "
          "router, and selected path clean, then carries material drift through "
          "shared expert fan-out. Root shared FFN components next."
          if required and selected_next == SHARED_ROUTE else
          "Previous FFN-delta component evidence is incomplete; keep this gate open."
      ),
  }


def write_outputs(metrics: dict[str, Any], out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n",
      encoding="utf-8")
  failed = [
      row["name"] for row in metrics["checks"]
      if row.get("pass") is not True
  ]
  c = metrics["min_cosines"]
  a = metrics["max_abs_diffs"]
  lines = [
      "# Seq490 Projection Q8 Input-Sensitivity Linear Input Source FFN-Delta Gap Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- diagnostic_classification: `{metrics['diagnostic_classification']}`",
      f"- min FFN-input/norm/router-logits/router-weights cosines: `{c['ffn_input']}` / `{c['ffn_norm']}` / `{c['router_logits']}` / `{c['router_weights']}`",
      f"- max FFN-input/norm/router-logits/router-weights abs: `{a['ffn_input']}` / `{a['ffn_norm']}` / `{a['router_logits']}` / `{a['router_weights']}`",
      f"- min selected gate-up/swiglu/down/weighted/moe cosines: `{c['selected_gate_up']}` / `{c['selected_swiglu']}` / `{c['selected_down']}` / `{c['weighted_selected_down']}` / `{c['moe_out']}`",
      f"- max selected gate-up/swiglu/down/weighted/moe abs: `{a['selected_gate_up']}` / `{a['selected_swiglu']}` / `{a['selected_down']}` / `{a['weighted_selected_down']}` / `{a['moe_out']}`",
      f"- min shared gate/sigmoid/gate-up/swiglu/down/gated cosines: `{c['shared_gate']}` / `{c['shared_gate_sigmoid']}` / `{c['shared_gate_up']}` / `{c['shared_swiglu']}` / `{c['shared_down']}` / `{c['shared_gated']}`",
      f"- max shared gate/sigmoid/gate-up/swiglu/down/gated abs: `{a['shared_gate']}` / `{a['shared_gate_sigmoid']}` / `{a['shared_gate_up']}` / `{a['shared_swiglu']}` / `{a['shared_down']}` / `{a['shared_gated']}`",
      f"- min FFN-out/residual cosines: `{c['ffn_out']}` / `{c['residual']}`",
      f"- max FFN-out/residual abs: `{a['ffn_out']}` / `{a['residual']}`",
      f"- selected_material/shared_material: `{str(metrics['selected_material']).lower()}` / `{str(metrics['shared_material']).lower()}`",
      f"- failed_checks: `{failed}`",
      f"- speedup_claims_allowed: `{str(metrics['speedup_claims_allowed']).lower()}`",
      "",
      metrics["next_route_reason"],
      "",
      "This is distribution/correctness evidence only. It is not a speed claim.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--routes", type=Path, default=DEFAULT_ROUTES)
  parser.add_argument("--seq489", type=Path, default=DEFAULT_SEQ489)
  parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
  return parser.parse_args()


def main() -> int:
  args = parse_args()
  metrics = compute(args)
  write_outputs(metrics, args.out_dir)
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
