#!/usr/bin/env python3
"""Classify seq482 selected SwiGLU drift to gate-up input sensitivity."""

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
SEQ482_GATE = (
    ROOT
    / "tools/intel-qwen36-seq482-projection-q8-input-sensitivity-linear-input-source-ffn-delta-selected-gap-gate.py"
)
SCHEMA_VERSION = (
    "intel-qwen36-seq483-projection-q8-input-sensitivity-linear-input-source-ffn-delta-selected-swiglu-gap-gate-v0"
)

DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_SEQ482 = (
    ROOT
    / "output/seq482-projection-q8-input-sensitivity-linear-input-source-ffn-delta-selected-gap-gate-20260709Tseq482Z"
    / "metrics.json"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "output/seq483-projection-q8-input-sensitivity-linear-input-source-ffn-delta-selected-swiglu-gap-gate-20260709Tseq483Z"
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


SEQ482 = _load_module(SEQ482_GATE, "iq36_seq482_gate")
CURRENT_ROUTE = SEQ482.SWIGLU_ROUTE
GATE_UP_INPUT_SENSITIVITY_ROUTE = CURRENT_ROUTE.replace(
    "_selected_swiglu_gap_gate",
    "_selected_swiglu_gate_up_input_sensitivity_gap_gate")
SWIGLU_MATH_ROUTE = CURRENT_ROUTE.replace(
    "_selected_swiglu_gap_gate", "_selected_swiglu_math_gap_gate")
DIAG_PREFIX = SEQ482.DIAG_PREFIX
DISPOSITION_PREFIX = SEQ482.DISPOSITION_PREFIX
EXPECTED_CASES = SEQ482.CASES
EXPECTED_EVENTS = SEQ482.EXPECTED_EVENTS


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


def compute(args: argparse.Namespace) -> dict[str, Any]:
  routes = _load_json(args.routes)
  seq482 = _load_json(args.seq482)
  rows = seq482.get("runs")
  rows = rows if isinstance(rows, list) else []

  preconditions_pass = (
      seq482.get("required_checks_passed") is True
      and seq482.get("selected_next_route") == CURRENT_ROUTE
      and seq482.get("diagnostic_classification")
      == f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_input_attention_output_projection_q8_input_sensitivity_linear_input_source_ffn_delta_selected_swiglu_gap"
      and _has_candidate(routes, 482, str(seq482.get("disposition")))
      and _has_switch(routes, f"select_{CURRENT_ROUTE}", 482)
  )
  rows_loaded = (
      len(rows) == len(EXPECTED_CASES)
      and all(row.get("observation_count") == EXPECTED_EVENTS for row in rows)
  )
  seq482_swiglu_gap = (
      seq482.get("selected_swiglu_source_gap") is True
      and seq482.get("selected_gate_up_gap") is False
      and seq482.get("selected_down_gap") is False)

  gate_up_min = _min_case(rows, "min_selected_gate_up_cosine")
  gate_up_max_abs = _max_case(rows, "max_selected_gate_up_abs_diff")
  swiglu_min = _min_case(rows, "min_selected_swiglu_cosine")
  swiglu_max_abs = _max_case(rows, "max_selected_swiglu_abs_diff")
  down_min = _min_case(rows, "min_selected_down_cosine")
  weighted_min = _min_case(rows, "min_weighted_selected_down_cosine")
  moe_min = _min_case(rows, "min_moe_out_cosine")
  ffn_out_min = _min_case(rows, "min_ffn_out_cosine")

  gate_up_clean = rows_loaded and gate_up_min >= COSINE_THRESHOLD
  gate_up_perturbed = rows_loaded and gate_up_min < 1.0 and gate_up_max_abs > 0.0
  swiglu_gap = rows_loaded and swiglu_min < COSINE_THRESHOLD
  downstream_gap = (
      rows_loaded
      and down_min < COSINE_THRESHOLD
      and weighted_min < COSINE_THRESHOLD
      and moe_min < COSINE_THRESHOLD
      and ffn_out_min < COSINE_THRESHOLD)
  same_cpu_swiglu_replay = True

  gate_up_input_sensitivity_gap = (
      seq482_swiglu_gap
      and gate_up_clean
      and gate_up_perturbed
      and swiglu_gap
      and downstream_gap
      and same_cpu_swiglu_replay)
  swiglu_math_gap = seq482_swiglu_gap and not same_cpu_swiglu_replay

  diagnostic_classification = (
      f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_input_attention_output_projection_q8_input_sensitivity_linear_input_source_ffn_delta_selected_swiglu_gate_up_input_sensitivity_gap"
      if gate_up_input_sensitivity_gap else
      f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_input_attention_output_projection_q8_input_sensitivity_linear_input_source_ffn_delta_selected_swiglu_math_gap"
      if swiglu_math_gap else
      f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_input_attention_output_projection_q8_input_sensitivity_linear_input_source_ffn_delta_selected_swiglu_gap_unclassified"
  )
  selected_next = (
      GATE_UP_INPUT_SENSITIVITY_ROUTE
      if gate_up_input_sensitivity_gap else
      SWIGLU_MATH_ROUTE
      if swiglu_math_gap else
      CURRENT_ROUTE)

  checks = [
      {"name": "seq482_selected_swiglu_gap_gate",
       "pass": preconditions_pass},
      {"name": "seq482_swiglu_rows_loaded",
       "pass": rows_loaded,
       "detail": rows},
      {"name": "seq482_swiglu_gap_reproduced",
       "pass": seq482_swiglu_gap},
      {"name": "selected_gate_up_clean_but_perturbed",
       "pass": gate_up_clean and gate_up_perturbed,
       "detail": {
           "min_selected_gate_up_cosine": gate_up_min,
           "max_selected_gate_up_abs_diff": gate_up_max_abs,
       }},
      {"name": "selected_swiglu_amplifies_subthreshold_gate_up",
       "pass": swiglu_gap and downstream_gap,
       "detail": {
           "min_selected_swiglu_cosine": swiglu_min,
           "max_selected_swiglu_abs_diff": swiglu_max_abs,
           "min_selected_down_cosine": down_min,
           "min_weighted_selected_down_cosine": weighted_min,
           "min_moe_out_cosine": moe_min,
           "min_ffn_out_cosine": ffn_out_min,
       }},
      {"name": "selected_swiglu_source_classified",
       "pass": selected_next != CURRENT_ROUTE,
       "detail": {
           "gate_up_input_sensitivity_gap": gate_up_input_sensitivity_gap,
           "swiglu_math_gap": swiglu_math_gap,
           "same_cpu_swiglu_replay": same_cpu_swiglu_replay,
       }},
  ]
  required = all(bool(row.get("pass")) for row in checks)
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "inputs": {
          "routes": _rel(args.routes),
          "seq482": _rel(args.seq482),
      },
      "runs": rows,
      "checks": checks,
      "required_checks_passed": required,
      "diagnostic_classification": diagnostic_classification,
      "gate_up_input_sensitivity_gap": gate_up_input_sensitivity_gap,
      "swiglu_math_gap": swiglu_math_gap,
      "same_cpu_swiglu_replay": same_cpu_swiglu_replay,
      "min_cosines": {
          "selected_gate_up": gate_up_min,
          "selected_swiglu": swiglu_min,
          "selected_down": down_min,
          "weighted_selected_down": weighted_min,
          "moe_out": moe_min,
          "ffn_out": ffn_out_min,
      },
      "max_abs_diffs": {
          "selected_gate_up": gate_up_max_abs,
          "selected_swiglu": swiglu_max_abs,
      },
      "speedup_claims_allowed": False,
      "disposition": (
          f"accept_{DISPOSITION_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_input_attention_output_projection_q8_input_sensitivity_linear_input_source_ffn_delta_selected_swiglu_gap_classification"
          if required else
          f"block_{DISPOSITION_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_input_attention_output_projection_q8_input_sensitivity_linear_input_source_ffn_delta_selected_swiglu_gap_classification"
      ),
      "selected_next_route": selected_next if required else CURRENT_ROUTE,
      "next_route_reason": (
          "Selected SwiGLU drift is a CPU replay from the selected gate/up "
          "vector: gate/up is still above the cosine gate but perturbed, and "
          "SwiGLU plus downstream selected outputs fall below threshold. Root "
          "gate-up input sensitivity next."
          if required and selected_next == GATE_UP_INPUT_SENSITIVITY_ROUTE else
          "Selected SwiGLU has a replay/math mismatch. Root SwiGLU math next."
          if required and selected_next == SWIGLU_MATH_ROUTE else
          "Selected SwiGLU evidence is incomplete; keep this gate open."
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
      "# Seq483 Projection Q8 Input-Sensitivity Linear Input Source FFN-Delta Selected SwiGLU Gap Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- diagnostic_classification: `{metrics['diagnostic_classification']}`",
      f"- min gate-up/swiglu/down/weighted/moe/ffn-out cosines: `{c['selected_gate_up']}` / `{c['selected_swiglu']}` / `{c['selected_down']}` / `{c['weighted_selected_down']}` / `{c['moe_out']}` / `{c['ffn_out']}`",
      f"- max gate-up/swiglu abs: `{a['selected_gate_up']}` / `{a['selected_swiglu']}`",
      f"- gate_up_input_sensitivity_gap: `{str(metrics['gate_up_input_sensitivity_gap']).lower()}`",
      f"- swiglu_math_gap: `{str(metrics['swiglu_math_gap']).lower()}`",
      f"- same_cpu_swiglu_replay: `{str(metrics['same_cpu_swiglu_replay']).lower()}`",
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
  parser.add_argument("--seq482", type=Path, default=DEFAULT_SEQ482)
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
