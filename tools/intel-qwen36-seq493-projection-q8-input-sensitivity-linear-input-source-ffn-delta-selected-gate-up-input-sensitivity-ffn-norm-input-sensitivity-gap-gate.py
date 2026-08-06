#!/usr/bin/env python3
"""Classify seq492 FFN-norm input sensitivity to attention-output source."""

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
SEQ492_GATE = (
    ROOT
    / "tools/intel-qwen36-seq492-projection-q8-input-sensitivity-linear-input-source-ffn-delta-selected-gate-up-input-sensitivity-gap-gate.py"
)
SCHEMA_VERSION = (
    "intel-qwen36-seq493-projection-q8-input-sensitivity-linear-input-source-ffn-delta-selected-gate-up-input-sensitivity-ffn-norm-input-sensitivity-gap-gate-v0"
)

DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_SEQ492 = (
    ROOT
    / "output/seq492-projection-q8-input-sensitivity-linear-input-source-ffn-delta-selected-gate-up-input-sensitivity-gap-gate-20260709Tseq492Z"
    / "metrics.json"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "output/seq493-projection-q8-input-sensitivity-linear-input-source-ffn-delta-selected-gate-up-input-sensitivity-ffn-norm-input-sensitivity-gap-gate-20260709Tseq493Z"
)

COSINE_THRESHOLD = 0.9999
SOURCE_REPLAY_EPS = 1.0e-9


def _load_module(path: Path, name: str) -> Any:
  spec = importlib.util.spec_from_file_location(name, path)
  if spec is None or spec.loader is None:
    raise SystemExit(f"cannot load module {name}: {path}")
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


SEQ492 = _load_module(SEQ492_GATE, "iq36_seq492_gate")
CURRENT_ROUTE = SEQ492.FFN_NORM_INPUT_SENSITIVITY_ROUTE
ATTENTION_OUTPUT_ROUTE = CURRENT_ROUTE.replace(
    "_ffn_norm_input_sensitivity_gap_gate",
    "_ffn_norm_input_sensitivity_attention_output_source_gap_gate")
LAYER_INPUT_ROUTE = CURRENT_ROUTE.replace(
    "_ffn_norm_input_sensitivity_gap_gate",
    "_ffn_norm_input_sensitivity_layer_input_source_gap_gate")
NORM_MATH_ROUTE = CURRENT_ROUTE.replace(
    "_ffn_norm_input_sensitivity_gap_gate",
    "_ffn_norm_math_gap_gate")
DIAG_PREFIX = SEQ492.DIAG_PREFIX
DISPOSITION_PREFIX = SEQ492.DISPOSITION_PREFIX
EXPECTED_CASES = SEQ492.EXPECTED_CASES
EXPECTED_EVENTS = SEQ492.EXPECTED_EVENTS


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


def _min_update(summary: dict[str, Any], key: str, value: Any) -> None:
  if isinstance(value, (int, float)):
    summary[key] = min(_num(summary.get(key), 1.0), float(value))


def _max_update(summary: dict[str, Any], key: str, value: Any) -> None:
  if isinstance(value, (int, float)):
    summary[key] = max(_num(summary.get(key)), float(value))


def _rows_by_layer(step: dict[str, Any]) -> dict[int, dict[str, Any]]:
  layers = step.get("layers")
  layers = layers if isinstance(layers, list) else []
  return {
      int(row["layer"]): row
      for row in layers
      if isinstance(row, dict) and isinstance(row.get("layer"), int)
  }


def _case_summary(row: dict[str, Any]) -> dict[str, Any]:
  smoke = _load_smoke(row)
  steps = smoke.get("residual_source_diff_by_step")
  steps = steps if isinstance(steps, list) else []
  source_layers = row.get("source_layers")
  source_layers = source_layers if isinstance(source_layers, list) else [1]
  source_layers = [int(layer) for layer in source_layers if isinstance(layer, int)]
  summary: dict[str, Any] = {
      "case_id": row.get("case_id"),
      "out_dir": row.get("out_dir"),
      "source_layers": source_layers,
      "observation_count": 0,
      "component_ffn_norm_min_cosine": row.get("min_ffn_norm_cosine"),
      "component_ffn_norm_max_abs_diff": row.get("max_ffn_norm_abs_diff"),
      "layer_input_min_cosine": 1.0,
      "layer_input_max_abs_diff": 0.0,
      "attention_output_min_cosine": 1.0,
      "attention_output_max_abs_diff": 0.0,
      "ffn_input_min_cosine": 1.0,
      "ffn_input_max_abs_diff": 0.0,
      "cpu_ffn_norm_from_gpu_input_min_cosine": 1.0,
      "cpu_ffn_norm_from_gpu_input_max_abs_diff": 0.0,
      "gpu_ffn_norm_vs_cpu_min_cosine": 1.0,
      "gpu_ffn_norm_vs_cpu_max_abs_diff": 0.0,
      "max_component_vs_residual_ffn_norm_cosine_delta": 0.0,
      "max_component_vs_residual_ffn_norm_abs_delta": 0.0,
  }
  for step in steps:
    if not isinstance(step, dict):
      continue
    rows_by_layer = _rows_by_layer(step)
    for layer in source_layers:
      source = rows_by_layer.get(layer)
      if not source:
        continue
      summary["observation_count"] += 1
      for metric in [
          "layer_input",
          "attention_output",
          "ffn_input",
          "cpu_ffn_norm_from_gpu_input",
          "gpu_ffn_norm_vs_cpu",
      ]:
        _min_update(summary, f"{metric}_min_cosine",
                    source.get(f"{metric}_cosine"))
        _max_update(summary, f"{metric}_max_abs_diff",
                    source.get(f"{metric}_max_abs_diff"))
  component_norm_min = _num(row.get("min_ffn_norm_cosine"), 1.0)
  component_norm_abs = _num(row.get("max_ffn_norm_abs_diff"))
  residual_norm_min = _num(summary.get("cpu_ffn_norm_from_gpu_input_min_cosine"),
                           1.0)
  residual_norm_abs = _num(summary.get("cpu_ffn_norm_from_gpu_input_max_abs_diff"))
  summary["max_component_vs_residual_ffn_norm_cosine_delta"] = abs(
      component_norm_min - residual_norm_min)
  summary["max_component_vs_residual_ffn_norm_abs_delta"] = abs(
      component_norm_abs - residual_norm_abs)
  return summary


def _min_case(rows: list[dict[str, Any]], key: str) -> float:
  return min((_num(row.get(key), 1.0) for row in rows), default=1.0)


def _max_case(rows: list[dict[str, Any]], key: str) -> float:
  return max((_num(row.get(key)) for row in rows), default=0.0)


def compute(args: argparse.Namespace) -> dict[str, Any]:
  routes = _load_json(args.routes)
  seq492 = _load_json(args.seq492)
  case_rows = seq492.get("runs")
  case_rows = case_rows if isinstance(case_rows, list) else []
  rows = [_case_summary(row) for row in case_rows if isinstance(row, dict)]

  preconditions_pass = (
      seq492.get("required_checks_passed") is True
      and seq492.get("selected_next_route") == CURRENT_ROUTE
      and seq492.get("diagnostic_classification")
      == f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_input_attention_output_projection_q8_input_sensitivity_linear_input_source_ffn_delta_selected_swiglu_gate_up_input_sensitivity_ffn_norm_input_sensitivity_attention_output_projection_q8_input_sensitivity_linear_input_source_ffn_delta_selected_gate_up_input_sensitivity_ffn_norm_input_sensitivity_gap"
      and _has_candidate(routes, 492, str(seq492.get("disposition")))
      and _has_switch(routes, f"select_{CURRENT_ROUTE}", 492)
  )
  rows_loaded = (
      len(rows) == len(EXPECTED_CASES)
      and all(row.get("observation_count") == EXPECTED_EVENTS for row in rows)
  )
  seq492_norm_sensitivity = (
      seq492.get("ffn_norm_input_sensitivity_gap") is True
      and seq492.get("gate_up_math_gap") is False)

  layer_input_min = _min_case(rows, "layer_input_min_cosine")
  layer_input_max_abs = _max_case(rows, "layer_input_max_abs_diff")
  attention_output_min = _min_case(rows, "attention_output_min_cosine")
  attention_output_max_abs = _max_case(rows, "attention_output_max_abs_diff")
  ffn_input_min = _min_case(rows, "ffn_input_min_cosine")
  ffn_input_max_abs = _max_case(rows, "ffn_input_max_abs_diff")
  cpu_norm_min = _min_case(rows, "cpu_ffn_norm_from_gpu_input_min_cosine")
  cpu_norm_max_abs = _max_case(rows, "cpu_ffn_norm_from_gpu_input_max_abs_diff")
  gpu_norm_vs_cpu_min = _min_case(rows, "gpu_ffn_norm_vs_cpu_min_cosine")
  gpu_norm_vs_cpu_max_abs = _max_case(rows, "gpu_ffn_norm_vs_cpu_max_abs_diff")
  max_norm_cos_delta = _max_case(
      rows, "max_component_vs_residual_ffn_norm_cosine_delta")
  max_norm_abs_delta = _max_case(
      rows, "max_component_vs_residual_ffn_norm_abs_delta")

  norm_replay_ok = (
      max_norm_cos_delta <= SOURCE_REPLAY_EPS
      and max_norm_abs_delta <= SOURCE_REPLAY_EPS)
  norm_math_ok = gpu_norm_vs_cpu_min >= COSINE_THRESHOLD
  ffn_input_clean_perturbed = (
      ffn_input_min >= COSINE_THRESHOLD
      and ffn_input_min < 1.0
      and ffn_input_max_abs > 0.0)
  layer_input_clean = layer_input_min >= COSINE_THRESHOLD
  attention_output_clean_perturbed = (
      attention_output_min >= COSINE_THRESHOLD
      and attention_output_min < 1.0
      and attention_output_max_abs > 0.0)
  attention_dominates_layer = (
      attention_output_max_abs > layer_input_max_abs
      and attention_output_min < layer_input_min)
  ffn_input_carries_attention = (
      ffn_input_max_abs > layer_input_max_abs
      and attention_output_max_abs >= ffn_input_max_abs)
  norm_amplifies_ffn_input = cpu_norm_max_abs > ffn_input_max_abs

  attention_output_source_gap = (
      seq492_norm_sensitivity
      and rows_loaded
      and norm_replay_ok
      and norm_math_ok
      and ffn_input_clean_perturbed
      and layer_input_clean
      and attention_output_clean_perturbed
      and attention_dominates_layer
      and ffn_input_carries_attention
      and norm_amplifies_ffn_input)
  layer_input_source_gap = (
      seq492_norm_sensitivity
      and rows_loaded
      and norm_replay_ok
      and norm_math_ok
      and ffn_input_clean_perturbed
      and layer_input_max_abs >= attention_output_max_abs)
  norm_math_gap = seq492_norm_sensitivity and rows_loaded and not norm_math_ok

  diagnostic_classification = (
      f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_input_attention_output_projection_q8_input_sensitivity_linear_input_source_ffn_delta_selected_swiglu_gate_up_input_sensitivity_ffn_norm_input_sensitivity_attention_output_projection_q8_input_sensitivity_linear_input_source_ffn_delta_selected_gate_up_input_sensitivity_ffn_norm_input_sensitivity_attention_output_source_gap"
      if attention_output_source_gap else
      f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_input_attention_output_projection_q8_input_sensitivity_linear_input_source_ffn_delta_selected_swiglu_gate_up_input_sensitivity_ffn_norm_input_sensitivity_attention_output_projection_q8_input_sensitivity_linear_input_source_ffn_delta_selected_gate_up_input_sensitivity_ffn_norm_input_sensitivity_layer_input_source_gap"
      if layer_input_source_gap else
      f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_input_attention_output_projection_q8_input_sensitivity_linear_input_source_ffn_delta_selected_swiglu_gate_up_input_sensitivity_ffn_norm_input_sensitivity_attention_output_projection_q8_input_sensitivity_linear_input_source_ffn_delta_selected_gate_up_input_sensitivity_ffn_norm_math_gap"
      if norm_math_gap else
      f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_input_attention_output_projection_q8_input_sensitivity_linear_input_source_ffn_delta_selected_swiglu_gate_up_input_sensitivity_ffn_norm_input_sensitivity_attention_output_projection_q8_input_sensitivity_linear_input_source_ffn_delta_selected_gate_up_input_sensitivity_ffn_norm_input_sensitivity_gap_unclassified"
  )
  selected_next = (
      ATTENTION_OUTPUT_ROUTE
      if attention_output_source_gap else
      LAYER_INPUT_ROUTE
      if layer_input_source_gap else
      NORM_MATH_ROUTE
      if norm_math_gap else
      CURRENT_ROUTE)
  checks = [
      {"name": "seq492_ffn_norm_input_sensitivity_gate",
       "pass": preconditions_pass},
      {"name": "seq492_residual_source_rows_loaded",
       "pass": rows_loaded,
       "detail": rows},
      {"name": "seq492_norm_sensitivity_reproduced",
       "pass": seq492_norm_sensitivity},
      {"name": "component_norm_matches_residual_norm_replay",
       "pass": norm_replay_ok,
       "detail": {
           "max_component_vs_residual_ffn_norm_cosine_delta": max_norm_cos_delta,
           "max_component_vs_residual_ffn_norm_abs_delta": max_norm_abs_delta,
       }},
      {"name": "gpu_norm_math_matches_cpu",
       "pass": norm_math_ok,
       "detail": {
           "min_gpu_ffn_norm_vs_cpu_cosine": gpu_norm_vs_cpu_min,
           "max_gpu_ffn_norm_vs_cpu_abs_diff": gpu_norm_vs_cpu_max_abs,
       }},
      {"name": "ffn_input_clean_but_perturbed",
       "pass": ffn_input_clean_perturbed and norm_amplifies_ffn_input,
       "detail": {
           "min_ffn_input_cosine": ffn_input_min,
           "max_ffn_input_abs_diff": ffn_input_max_abs,
           "min_cpu_ffn_norm_from_gpu_input_cosine": cpu_norm_min,
           "max_cpu_ffn_norm_from_gpu_input_abs_diff": cpu_norm_max_abs,
       }},
      {"name": "attention_output_source_dominates_layer_input",
       "pass": (
           layer_input_clean
           and attention_output_clean_perturbed
           and attention_dominates_layer
           and ffn_input_carries_attention),
       "detail": {
           "min_layer_input_cosine": layer_input_min,
           "max_layer_input_abs_diff": layer_input_max_abs,
           "min_attention_output_cosine": attention_output_min,
           "max_attention_output_abs_diff": attention_output_max_abs,
       }},
      {"name": "ffn_norm_input_source_classified",
       "pass": selected_next != CURRENT_ROUTE,
       "detail": {
           "attention_output_source_gap": attention_output_source_gap,
           "layer_input_source_gap": layer_input_source_gap,
           "norm_math_gap": norm_math_gap,
       }},
  ]
  required = all(bool(row.get("pass")) for row in checks)
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "inputs": {
          "routes": _rel(args.routes),
          "seq492": _rel(args.seq492),
      },
      "runs": rows,
      "checks": checks,
      "required_checks_passed": required,
      "diagnostic_classification": diagnostic_classification,
      "attention_output_source_gap": attention_output_source_gap,
      "layer_input_source_gap": layer_input_source_gap,
      "norm_math_gap": norm_math_gap,
      "norm_replay_ok": norm_replay_ok,
      "norm_math_ok": norm_math_ok,
      "min_cosines": {
          "layer_input": layer_input_min,
          "attention_output": attention_output_min,
          "ffn_input": ffn_input_min,
          "cpu_ffn_norm_from_gpu_input": cpu_norm_min,
          "gpu_ffn_norm_vs_cpu": gpu_norm_vs_cpu_min,
      },
      "max_abs_diffs": {
          "layer_input": layer_input_max_abs,
          "attention_output": attention_output_max_abs,
          "ffn_input": ffn_input_max_abs,
          "cpu_ffn_norm_from_gpu_input": cpu_norm_max_abs,
          "gpu_ffn_norm_vs_cpu": gpu_norm_vs_cpu_max_abs,
      },
      "source_match": {
          "max_component_vs_residual_ffn_norm_cosine_delta": max_norm_cos_delta,
          "max_component_vs_residual_ffn_norm_abs_delta": max_norm_abs_delta,
      },
      "speedup_claims_allowed": False,
      "disposition": (
          f"accept_{DISPOSITION_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_input_attention_output_projection_q8_input_sensitivity_linear_input_source_ffn_delta_selected_swiglu_gate_up_input_sensitivity_ffn_norm_input_sensitivity_attention_output_projection_q8_input_sensitivity_linear_input_source_ffn_delta_selected_gate_up_input_sensitivity_ffn_norm_input_sensitivity_gap_classification"
          if required else
          f"block_{DISPOSITION_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_input_attention_output_projection_q8_input_sensitivity_linear_input_source_ffn_delta_selected_swiglu_gate_up_input_sensitivity_ffn_norm_input_sensitivity_attention_output_projection_q8_input_sensitivity_linear_input_source_ffn_delta_selected_gate_up_input_sensitivity_ffn_norm_input_sensitivity_gap_classification"
      ),
      "selected_next_route": selected_next if required else CURRENT_ROUTE,
      "next_route_reason": (
          "FFN-norm input sensitivity is sourced by attention-output drift: "
          "layer input stays clean and smaller, attention output has the larger "
          "clean perturbation, FFN input carries it, and GPU norm math matches "
          "CPU. Root attention-output source next."
          if required and selected_next == ATTENTION_OUTPUT_ROUTE else
          "FFN-norm input sensitivity is sourced by layer-input drift. Root "
          "layer input next."
          if required and selected_next == LAYER_INPUT_ROUTE else
          "FFN norm math no longer matches CPU on live input. Root norm math next."
          if required and selected_next == NORM_MATH_ROUTE else
          "FFN-norm input-sensitivity evidence is incomplete; keep this gate open."
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
  s = metrics["source_match"]
  lines = [
      "# Seq493 Projection Q8 Input-Sensitivity Linear Input Source FFN-Delta Selected Gate-Up Input-Sensitivity FFN-Norm Input-Sensitivity Gap Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- diagnostic_classification: `{metrics['diagnostic_classification']}`",
      f"- min layer/attention/ffn-input/norm-replay/gpu-norm-vs-cpu cosines: `{c['layer_input']}` / `{c['attention_output']}` / `{c['ffn_input']}` / `{c['cpu_ffn_norm_from_gpu_input']}` / `{c['gpu_ffn_norm_vs_cpu']}`",
      f"- max layer/attention/ffn-input/norm-replay/gpu-norm-vs-cpu abs: `{a['layer_input']}` / `{a['attention_output']}` / `{a['ffn_input']}` / `{a['cpu_ffn_norm_from_gpu_input']}` / `{a['gpu_ffn_norm_vs_cpu']}`",
      f"- component-vs-residual norm cosine/abs deltas: `{s['max_component_vs_residual_ffn_norm_cosine_delta']}` / `{s['max_component_vs_residual_ffn_norm_abs_delta']}`",
      f"- attention_output_source_gap: `{str(metrics['attention_output_source_gap']).lower()}`",
      f"- layer_input_source_gap: `{str(metrics['layer_input_source_gap']).lower()}`",
      f"- norm_math_gap: `{str(metrics['norm_math_gap']).lower()}`",
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
  parser.add_argument("--seq492", type=Path, default=DEFAULT_SEQ492)
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
