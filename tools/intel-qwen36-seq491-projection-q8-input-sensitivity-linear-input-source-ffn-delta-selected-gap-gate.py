#!/usr/bin/env python3
"""Classify seq490 selected FFN fan-out material drift."""

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
SEQ490_GATE = (
    ROOT
    / "tools/intel-qwen36-seq490-projection-q8-input-sensitivity-linear-input-source-ffn-delta-gap-gate.py"
)
SCHEMA_VERSION = (
    "intel-qwen36-seq491-projection-q8-input-sensitivity-linear-input-source-ffn-delta-selected-gap-gate-v0"
)

DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_SEQ490 = (
    ROOT
    / "output/seq490-projection-q8-input-sensitivity-linear-input-source-ffn-delta-gap-gate-20260709Tseq490Z"
    / "metrics.json"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "output/seq491-projection-q8-input-sensitivity-linear-input-source-ffn-delta-selected-gap-gate-20260709Tseq491Z"
)

COSINE_THRESHOLD = 0.9999
MATERIAL_ABS_EPS = 1.0e-8


def _load_module(path: Path, name: str) -> Any:
  spec = importlib.util.spec_from_file_location(name, path)
  if spec is None or spec.loader is None:
    raise SystemExit(f"cannot load module {name}: {path}")
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


SEQ490 = _load_module(SEQ490_GATE, "iq36_seq490_gate")
CURRENT_ROUTE = SEQ490.SELECTED_ROUTE
GATE_UP_INPUT_SENSITIVITY_ROUTE = CURRENT_ROUTE.replace(
    "_linear_input_source_ffn_delta_selected_gap_gate",
    "_linear_input_source_ffn_delta_selected_gate_up_input_sensitivity_gap_gate")
SWIGLU_ROUTE = CURRENT_ROUTE.replace(
    "_linear_input_source_ffn_delta_selected_gap_gate",
    "_linear_input_source_ffn_delta_selected_swiglu_gap_gate")
DOWN_ROUTE = CURRENT_ROUTE.replace(
    "_linear_input_source_ffn_delta_selected_gap_gate",
    "_linear_input_source_ffn_delta_selected_down_gap_gate")
DIAG_PREFIX = SEQ490.DIAG_PREFIX
DISPOSITION_PREFIX = SEQ490.DISPOSITION_PREFIX
EXPECTED_CASES = SEQ490.EXPECTED_CASES
EXPECTED_EVENTS = SEQ490.EXPECTED_EVENTS


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


def _min_case(rows: list[dict[str, Any]], key: str) -> float:
  return min((_num(row.get(key), 1.0) for row in rows), default=1.0)


def _max_case(rows: list[dict[str, Any]], key: str) -> float:
  return max((_num(row.get(key)) for row in rows), default=0.0)


def _material(min_cosine: float, max_abs: float) -> bool:
  return min_cosine < 1.0 and max_abs > MATERIAL_ABS_EPS


def _has_material(rows: list[dict[str, Any]], key: str) -> bool:
  return any(isinstance(row.get(key), dict) for row in rows)


def compute(args: argparse.Namespace) -> dict[str, Any]:
  routes = _load_json(args.routes)
  seq490 = _load_json(args.seq490)
  rows = seq490.get("runs")
  rows = rows if isinstance(rows, list) else []

  preconditions_pass = (
      seq490.get("required_checks_passed") is True
      and seq490.get("selected_next_route") == CURRENT_ROUTE
      and seq490.get("diagnostic_classification")
      == f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_input_attention_output_projection_q8_input_sensitivity_linear_input_source_ffn_delta_selected_swiglu_gate_up_input_sensitivity_ffn_norm_input_sensitivity_attention_output_projection_q8_input_sensitivity_linear_input_source_ffn_delta_selected_gap"
      and _has_candidate(routes, 490, str(seq490.get("disposition")))
      and _has_switch(routes, f"select_{CURRENT_ROUTE}", 490)
  )
  rows_loaded = (
      len(rows) == len(EXPECTED_CASES)
      and all(row.get("observation_count") == EXPECTED_EVENTS for row in rows)
  )
  seq490_selected_material = (
      seq490.get("selected_material") is True
      and seq490.get("selected_downstream_material") is True
      and seq490.get("ffn_norm_gap") is False
      and seq490.get("router_gap") is False)

  gate_up_min = _min_case(rows, "min_selected_gate_up_cosine")
  swiglu_min = _min_case(rows, "min_selected_swiglu_cosine")
  down_min = _min_case(rows, "min_selected_down_cosine")
  weighted_min = _min_case(rows, "min_weighted_selected_down_cosine")
  moe_min = _min_case(rows, "min_moe_out_cosine")
  ffn_out_min = _min_case(rows, "min_ffn_out_cosine")
  shared_down_min = _min_case(rows, "min_shared_down_cosine")
  shared_gated_min = _min_case(rows, "min_shared_gated_cosine")

  gate_up_max_abs = _max_case(rows, "max_selected_gate_up_abs_diff")
  swiglu_max_abs = _max_case(rows, "max_selected_swiglu_abs_diff")
  down_max_abs = _max_case(rows, "max_selected_down_abs_diff")
  weighted_max_abs = _max_case(rows, "max_weighted_selected_down_abs_diff")
  moe_max_abs = _max_case(rows, "max_moe_out_abs_diff")
  ffn_out_max_abs = _max_case(rows, "max_ffn_out_abs_diff")
  shared_down_max_abs = _max_case(rows, "max_shared_down_abs_diff")
  shared_gated_max_abs = _max_case(rows, "max_shared_gated_abs_diff")

  gate_up_material = rows_loaded and _material(gate_up_min, gate_up_max_abs)
  swiglu_material = rows_loaded and _material(swiglu_min, swiglu_max_abs)
  down_material = rows_loaded and _material(down_min, down_max_abs)
  weighted_material = rows_loaded and _material(weighted_min, weighted_max_abs)
  moe_material = rows_loaded and _material(moe_min, moe_max_abs)
  ffn_out_material = rows_loaded and _material(ffn_out_min, ffn_out_max_abs)
  shared_material = (
      rows_loaded
      and (_material(shared_down_min, shared_down_max_abs)
           or _material(shared_gated_min, shared_gated_max_abs)))

  gate_up_clean = rows_loaded and gate_up_min >= COSINE_THRESHOLD
  swiglu_clean = rows_loaded and swiglu_min >= COSINE_THRESHOLD
  downstream_material = (
      down_material and weighted_material and moe_material and ffn_out_material)
  first_gate_up_material = _has_material(rows, "first_selected_gate_up_material")
  first_swiglu_material = _has_material(rows, "first_selected_swiglu_material")
  first_down_material = _has_material(rows, "first_selected_down_material")

  gate_up_input_sensitivity_gap = (
      seq490_selected_material
      and gate_up_clean
      and gate_up_material
      and first_gate_up_material
      and swiglu_material
      and downstream_material)
  swiglu_material_gap = (
      seq490_selected_material
      and not gate_up_material
      and swiglu_material
      and first_swiglu_material
      and downstream_material)
  down_material_gap = (
      seq490_selected_material
      and not gate_up_material
      and swiglu_clean
      and down_material
      and first_down_material
      and weighted_material
      and moe_material
      and ffn_out_material)

  diagnostic_classification = (
      f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_input_attention_output_projection_q8_input_sensitivity_linear_input_source_ffn_delta_selected_swiglu_gate_up_input_sensitivity_ffn_norm_input_sensitivity_attention_output_projection_q8_input_sensitivity_linear_input_source_ffn_delta_selected_gate_up_input_sensitivity_gap"
      if gate_up_input_sensitivity_gap else
      f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_input_attention_output_projection_q8_input_sensitivity_linear_input_source_ffn_delta_selected_swiglu_gate_up_input_sensitivity_ffn_norm_input_sensitivity_attention_output_projection_q8_input_sensitivity_linear_input_source_ffn_delta_selected_swiglu_gap"
      if swiglu_material_gap else
      f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_input_attention_output_projection_q8_input_sensitivity_linear_input_source_ffn_delta_selected_swiglu_gate_up_input_sensitivity_ffn_norm_input_sensitivity_attention_output_projection_q8_input_sensitivity_linear_input_source_ffn_delta_selected_down_gap"
      if down_material_gap else
      f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_input_attention_output_projection_q8_input_sensitivity_linear_input_source_ffn_delta_selected_gap_unclassified"
  )
  selected_next = (
      GATE_UP_INPUT_SENSITIVITY_ROUTE
      if gate_up_input_sensitivity_gap else
      SWIGLU_ROUTE
      if swiglu_material_gap else
      DOWN_ROUTE
      if down_material_gap else
      CURRENT_ROUTE)

  checks = [
      {"name": "seq490_selected_gap_gate",
       "pass": preconditions_pass},
      {"name": "seq490_selected_rows_loaded",
       "pass": rows_loaded,
       "detail": rows},
      {"name": "seq490_selected_material_reproduced",
       "pass": seq490_selected_material},
      {"name": "selected_gate_up_clean_but_material",
       "pass": gate_up_clean and gate_up_material and first_gate_up_material,
       "detail": {
           "min_selected_gate_up_cosine": gate_up_min,
           "max_selected_gate_up_abs_diff": gate_up_max_abs,
           "first_selected_gate_up_material": first_gate_up_material,
       }},
      {"name": "selected_swiglu_and_downstream_material_carried",
       "pass": swiglu_material and downstream_material,
       "detail": {
           "min_selected_swiglu_cosine": swiglu_min,
           "max_selected_swiglu_abs_diff": swiglu_max_abs,
           "min_selected_down_cosine": down_min,
           "min_weighted_selected_down_cosine": weighted_min,
           "min_moe_out_cosine": moe_min,
           "min_ffn_out_cosine": ffn_out_min,
       }},
      {"name": "selected_fanout_path_classified",
       "pass": selected_next != CURRENT_ROUTE,
       "detail": {
           "gate_up_input_sensitivity_gap": gate_up_input_sensitivity_gap,
           "swiglu_material_gap": swiglu_material_gap,
           "down_material_gap": down_material_gap,
           "shared_material": shared_material,
       }},
  ]
  required = all(bool(row.get("pass")) for row in checks)
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "inputs": {
          "routes": _rel(args.routes),
          "seq490": _rel(args.seq490),
      },
      "runs": rows,
      "checks": checks,
      "required_checks_passed": required,
      "diagnostic_classification": diagnostic_classification,
      "gate_up_input_sensitivity_gap": gate_up_input_sensitivity_gap,
      "swiglu_material_gap": swiglu_material_gap,
      "down_material_gap": down_material_gap,
      "shared_material_recorded": shared_material,
      "downstream_material_carried": downstream_material,
      "min_cosines": {
          "selected_gate_up": gate_up_min,
          "selected_swiglu": swiglu_min,
          "selected_down": down_min,
          "weighted_selected_down": weighted_min,
          "moe_out": moe_min,
          "ffn_out": ffn_out_min,
          "shared_down": shared_down_min,
          "shared_gated": shared_gated_min,
      },
      "max_abs_diffs": {
          "selected_gate_up": gate_up_max_abs,
          "selected_swiglu": swiglu_max_abs,
          "selected_down": down_max_abs,
          "weighted_selected_down": weighted_max_abs,
          "moe_out": moe_max_abs,
          "ffn_out": ffn_out_max_abs,
          "shared_down": shared_down_max_abs,
          "shared_gated": shared_gated_max_abs,
      },
      "speedup_claims_allowed": False,
      "disposition": (
          f"accept_{DISPOSITION_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_input_attention_output_projection_q8_input_sensitivity_linear_input_source_ffn_delta_selected_gap_classification"
          if required else
          f"block_{DISPOSITION_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_input_attention_output_projection_q8_input_sensitivity_linear_input_source_ffn_delta_selected_gap_classification"
      ),
      "selected_next_route": selected_next if required else CURRENT_ROUTE,
      "next_route_reason": (
          "Selected fan-out material drift starts at selected gate/up: gate/up "
          "is still above the cosine gate but materially perturbed, and SwiGLU "
          "plus selected-downstream outputs carry the drift. Root selected "
          "gate-up input sensitivity next."
          if required and selected_next == GATE_UP_INPUT_SENSITIVITY_ROUTE else
          "Selected fan-out material drift starts at selected SwiGLU. Root "
          "selected SwiGLU next."
          if required and selected_next == SWIGLU_ROUTE else
          "Selected fan-out material drift starts at selected down. Root "
          "selected down next."
          if required and selected_next == DOWN_ROUTE else
          "Selected fan-out evidence is incomplete; keep this gate open."
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
      "# Seq491 Projection Q8 Input-Sensitivity Linear Input Source FFN-Delta Selected Gap Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- diagnostic_classification: `{metrics['diagnostic_classification']}`",
      f"- selected gate-up/swiglu/down/weighted/moe/ffn-out cosines: `{c['selected_gate_up']}` / `{c['selected_swiglu']}` / `{c['selected_down']}` / `{c['weighted_selected_down']}` / `{c['moe_out']}` / `{c['ffn_out']}`",
      f"- selected gate-up/swiglu/down/weighted/moe/ffn-out max abs: `{a['selected_gate_up']}` / `{a['selected_swiglu']}` / `{a['selected_down']}` / `{a['weighted_selected_down']}` / `{a['moe_out']}` / `{a['ffn_out']}`",
      f"- shared down/gated cosines: `{c['shared_down']}` / `{c['shared_gated']}`",
      f"- shared down/gated max abs: `{a['shared_down']}` / `{a['shared_gated']}`",
      f"- gate_up_input_sensitivity_gap: `{str(metrics['gate_up_input_sensitivity_gap']).lower()}`",
      f"- swiglu_material_gap: `{str(metrics['swiglu_material_gap']).lower()}`",
      f"- down_material_gap: `{str(metrics['down_material_gap']).lower()}`",
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
  parser.add_argument("--seq490", type=Path, default=DEFAULT_SEQ490)
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
