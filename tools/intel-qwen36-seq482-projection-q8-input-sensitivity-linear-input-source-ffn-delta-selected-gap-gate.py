#!/usr/bin/env python3
"""Classify seq481 Q8-sensitive selected FFN component drift."""

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
SEQ481_GATE = (
    ROOT
    / "tools/intel-qwen36-seq481-projection-q8-input-sensitivity-linear-input-source-ffn-delta-gap-gate.py"
)
SCHEMA_VERSION = (
    "intel-qwen36-seq482-projection-q8-input-sensitivity-linear-input-source-ffn-delta-selected-gap-gate-v0"
)

DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_SEQ481 = (
    ROOT
    / "output/seq481-projection-q8-input-sensitivity-linear-input-source-ffn-delta-gap-gate-20260709Tseq481Z"
    / "metrics.json"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "output/seq482-projection-q8-input-sensitivity-linear-input-source-ffn-delta-selected-gap-gate-20260709Tseq482Z"
)

COSINE_THRESHOLD = 0.9999


def _load_module(path: Path, name: str) -> Any:
  spec = importlib.util.spec_from_file_location(name, path)
  if spec is None or spec.loader is None:
    raise SystemExit(f"cannot load module {name}: {path}")
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


SEQ481 = _load_module(SEQ481_GATE, "iq36_seq481_gate")
CURRENT_ROUTE = SEQ481.SELECTED_ROUTE
GATE_UP_ROUTE = CURRENT_ROUTE.replace(
    "_linear_input_source_ffn_delta_selected_gap_gate",
    "_linear_input_source_ffn_delta_selected_gate_up_gap_gate")
SWIGLU_ROUTE = CURRENT_ROUTE.replace(
    "_linear_input_source_ffn_delta_selected_gap_gate",
    "_linear_input_source_ffn_delta_selected_swiglu_gap_gate")
DOWN_ROUTE = CURRENT_ROUTE.replace(
    "_linear_input_source_ffn_delta_selected_gap_gate",
    "_linear_input_source_ffn_delta_selected_down_gap_gate")
DIAG_PREFIX = SEQ481.DIAG_PREFIX
DISPOSITION_PREFIX = SEQ481.DISPOSITION_PREFIX
CASES = SEQ481.CASES
SOURCE_LAYERS = SEQ481.SOURCE_LAYERS
EXPECTED_EVENTS = SEQ481.EXPECTED_EVENTS


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


def _has_first_gap(rows: list[dict[str, Any]], key: str) -> bool:
  return any(isinstance(row.get(key), dict) for row in rows)


def compute(args: argparse.Namespace) -> dict[str, Any]:
  routes = _load_json(args.routes)
  seq481 = _load_json(args.seq481)
  rows = seq481.get("runs")
  rows = rows if isinstance(rows, list) else []

  preconditions_pass = (
      seq481.get("required_checks_passed") is True
      and seq481.get("selected_next_route") == CURRENT_ROUTE
      and seq481.get("diagnostic_classification")
      == f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_input_attention_output_projection_q8_input_sensitivity_linear_input_source_ffn_delta_selected_gap"
      and _has_candidate(routes, 481, str(seq481.get("disposition")))
      and _has_switch(routes, f"select_{CURRENT_ROUTE}", 481)
  )
  rows_loaded = (
      len(rows) == len(CASES)
      and all(row.get("observation_count") == EXPECTED_EVENTS for row in rows)
  )
  seq481_selected_gap = (
      seq481.get("selected_gap") is True
      and seq481.get("ffn_norm_gap") is False
      and seq481.get("router_gap") is False)

  gate_up_min = _min_case(rows, "min_selected_gate_up_cosine")
  swiglu_min = _min_case(rows, "min_selected_swiglu_cosine")
  down_min = _min_case(rows, "min_selected_down_cosine")
  weighted_min = _min_case(rows, "min_weighted_selected_down_cosine")
  moe_min = _min_case(rows, "min_moe_out_cosine")
  ffn_out_min = _min_case(rows, "min_ffn_out_cosine")
  shared_down_min = _min_case(rows, "min_shared_down_cosine")

  gate_up_max_abs = _max_case(rows, "max_selected_gate_up_abs_diff")
  swiglu_max_abs = _max_case(rows, "max_selected_swiglu_abs_diff")
  down_max_abs = _max_case(rows, "max_selected_down_abs_diff")
  weighted_max_abs = _max_case(rows, "max_weighted_selected_down_abs_diff")
  moe_max_abs = _max_case(rows, "max_moe_out_abs_diff")
  ffn_out_max_abs = _max_case(rows, "max_ffn_out_abs_diff")

  gate_up_clean = rows_loaded and gate_up_min >= COSINE_THRESHOLD
  swiglu_gap = rows_loaded and swiglu_min < COSINE_THRESHOLD
  down_gap = rows_loaded and down_min < COSINE_THRESHOLD
  weighted_gap = rows_loaded and weighted_min < COSINE_THRESHOLD
  moe_gap = rows_loaded and moe_min < COSINE_THRESHOLD
  ffn_out_gap = rows_loaded and ffn_out_min < COSINE_THRESHOLD
  first_gate_up_gap = _has_first_gap(rows, "first_selected_gate_up_gap")
  first_swiglu_gap = _has_first_gap(rows, "first_selected_swiglu_gap")
  first_down_gap = _has_first_gap(rows, "first_selected_down_gap")

  selected_swiglu_source_gap = (
      seq481_selected_gap
      and gate_up_clean
      and not first_gate_up_gap
      and swiglu_gap
      and first_swiglu_gap
      and down_gap
      and weighted_gap
      and moe_gap
      and ffn_out_gap)
  selected_gate_up_gap = (
      seq481_selected_gap
      and gate_up_min < COSINE_THRESHOLD
      and first_gate_up_gap)
  selected_down_gap = (
      seq481_selected_gap
      and gate_up_clean
      and swiglu_min >= COSINE_THRESHOLD
      and down_gap
      and first_down_gap)

  diagnostic_classification = (
      f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_input_attention_output_projection_q8_input_sensitivity_linear_input_source_ffn_delta_selected_swiglu_gap"
      if selected_swiglu_source_gap else
      f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_input_attention_output_projection_q8_input_sensitivity_linear_input_source_ffn_delta_selected_gate_up_gap"
      if selected_gate_up_gap else
      f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_input_attention_output_projection_q8_input_sensitivity_linear_input_source_ffn_delta_selected_down_gap"
      if selected_down_gap else
      f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_input_attention_output_projection_q8_input_sensitivity_linear_input_source_ffn_delta_selected_gap_unclassified"
  )
  selected_next = (
      SWIGLU_ROUTE
      if selected_swiglu_source_gap else
      GATE_UP_ROUTE
      if selected_gate_up_gap else
      DOWN_ROUTE
      if selected_down_gap else
      CURRENT_ROUTE)
  checks = [
      {"name": "seq481_selected_selected_gap_gate",
       "pass": preconditions_pass},
      {"name": "seq481_selected_rows_loaded",
       "pass": rows_loaded,
       "detail": rows},
      {"name": "seq481_selected_gap_reproduced",
       "pass": seq481_selected_gap},
      {"name": "selected_gate_up_clean",
       "pass": gate_up_clean and not first_gate_up_gap,
       "detail": {
           "min_selected_gate_up_cosine": gate_up_min,
           "max_selected_gate_up_abs_diff": gate_up_max_abs,
           "first_selected_gate_up_gap": first_gate_up_gap,
       }},
      {"name": "selected_swiglu_first_gap",
       "pass": swiglu_gap and first_swiglu_gap,
       "detail": {
           "min_selected_swiglu_cosine": swiglu_min,
           "max_selected_swiglu_abs_diff": swiglu_max_abs,
       }},
      {"name": "selected_downstream_gap_carried",
       "pass": down_gap and weighted_gap and moe_gap and ffn_out_gap,
       "detail": {
           "min_selected_down_cosine": down_min,
           "min_weighted_selected_down_cosine": weighted_min,
           "min_moe_out_cosine": moe_min,
           "min_ffn_out_cosine": ffn_out_min,
       }},
      {"name": "selected_component_path_classified",
       "pass": selected_next != CURRENT_ROUTE,
       "detail": {
           "selected_swiglu_source_gap": selected_swiglu_source_gap,
           "selected_gate_up_gap": selected_gate_up_gap,
           "selected_down_gap": selected_down_gap,
       }},
  ]
  required = all(bool(row.get("pass")) for row in checks)
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "inputs": {
          "routes": _rel(args.routes),
          "seq481": _rel(args.seq481),
          "source_layers": SOURCE_LAYERS,
      },
      "runs": rows,
      "checks": checks,
      "required_checks_passed": required,
      "diagnostic_classification": diagnostic_classification,
      "selected_swiglu_source_gap": selected_swiglu_source_gap,
      "selected_gate_up_gap": selected_gate_up_gap,
      "selected_down_gap": selected_down_gap,
      "downstream_gap_carried": down_gap and weighted_gap and moe_gap and ffn_out_gap,
      "shared_fanout_drift_recorded": shared_down_min < COSINE_THRESHOLD,
      "min_cosines": {
          "selected_gate_up": gate_up_min,
          "selected_swiglu": swiglu_min,
          "selected_down": down_min,
          "weighted_selected_down": weighted_min,
          "moe_out": moe_min,
          "ffn_out": ffn_out_min,
          "shared_down": shared_down_min,
      },
      "max_abs_diffs": {
          "selected_gate_up": gate_up_max_abs,
          "selected_swiglu": swiglu_max_abs,
          "selected_down": down_max_abs,
          "weighted_selected_down": weighted_max_abs,
          "moe_out": moe_max_abs,
          "ffn_out": ffn_out_max_abs,
      },
      "speedup_claims_allowed": False,
      "disposition": (
          f"accept_{DISPOSITION_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_input_attention_output_projection_q8_input_sensitivity_linear_input_source_ffn_delta_selected_gap_classification"
          if required else
          f"block_{DISPOSITION_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_input_attention_output_projection_q8_input_sensitivity_linear_input_source_ffn_delta_selected_gap_classification"
      ),
      "selected_next_route": selected_next if required else CURRENT_ROUTE,
      "next_route_reason": (
          "Selected gate/up remains clean while selected SwiGLU is the first "
          "below-threshold selected component; down, weighted down, MoE out, "
          "and FFN out carry that drift. Root selected SwiGLU next."
          if required and selected_next == SWIGLU_ROUTE else
          "Selected gate/up is already below the cosine gate. Root selected "
          "gate/up next."
          if required and selected_next == GATE_UP_ROUTE else
          "Selected gate/up and SwiGLU are clean while selected down first "
          "drops below the cosine gate. Root selected down next."
          if required and selected_next == DOWN_ROUTE else
          "Selected component evidence is incomplete; keep this gate open."
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
      "# Seq482 Projection Q8 Input-Sensitivity Linear Input Source FFN-Delta Selected Gap Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- diagnostic_classification: `{metrics['diagnostic_classification']}`",
      f"- min selected gate-up/swiglu/down/weighted/moe/ffn-out cosines: `{c['selected_gate_up']}` / `{c['selected_swiglu']}` / `{c['selected_down']}` / `{c['weighted_selected_down']}` / `{c['moe_out']}` / `{c['ffn_out']}`",
      f"- max selected gate-up/swiglu/down/weighted/moe/ffn-out abs: `{a['selected_gate_up']}` / `{a['selected_swiglu']}` / `{a['selected_down']}` / `{a['weighted_selected_down']}` / `{a['moe_out']}` / `{a['ffn_out']}`",
      f"- selected_swiglu_source_gap: `{str(metrics['selected_swiglu_source_gap']).lower()}`",
      f"- selected_gate_up_gap: `{str(metrics['selected_gate_up_gap']).lower()}`",
      f"- selected_down_gap: `{str(metrics['selected_down_gap']).lower()}`",
      f"- shared_fanout_drift_recorded: `{str(metrics['shared_fanout_drift_recorded']).lower()}`",
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
  parser.add_argument("--seq481", type=Path, default=DEFAULT_SEQ481)
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
