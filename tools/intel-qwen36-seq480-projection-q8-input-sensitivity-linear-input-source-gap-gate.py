#!/usr/bin/env python3
"""Classify seq479 Q8-sensitive linear input drift to previous FFN delta."""

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
SEQ479_GATE = (
    ROOT
    / "tools/intel-qwen36-seq479-projection-q8-input-sensitivity-linear-delta-z-source-gap-gate.py"
)
SCHEMA_VERSION = (
    "intel-qwen36-seq480-projection-q8-input-sensitivity-linear-input-source-gap-gate-v0"
)

DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_SEQ479 = (
    ROOT
    / "output/seq479-projection-q8-input-sensitivity-linear-delta-z-source-gap-gate-20260709Tseq479Z"
    / "metrics.json"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "output/seq480-projection-q8-input-sensitivity-linear-input-source-gap-gate-20260709Tseq480Z"
)

COSINE_THRESHOLD = 0.9999
SOURCE_MATCH_EPS = 1.0e-8
SOURCE_REPLAY_EPS = 5.0e-7


def _load_module(path: Path, name: str) -> Any:
  spec = importlib.util.spec_from_file_location(name, path)
  if spec is None or spec.loader is None:
    raise SystemExit(f"cannot load module {name}: {path}")
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


SEQ479 = _load_module(SEQ479_GATE, "iq36_seq479_gate")
CURRENT_ROUTE = SEQ479.LINEAR_INPUT_ROUTE
FFN_DELTA_ROUTE = CURRENT_ROUTE.replace(
    "_attention_output_projection_q8_input_sensitivity_linear_input_gap_gate",
    "_attention_output_projection_q8_input_sensitivity_linear_input_source_ffn_delta_gap_gate")
ATTENTION_OUTPUT_ROUTE = CURRENT_ROUTE.replace(
    "_attention_output_projection_q8_input_sensitivity_linear_input_gap_gate",
    "_attention_output_projection_q8_input_sensitivity_linear_input_source_attention_output_gap_gate")
FFN_MATH_ROUTE = CURRENT_ROUTE.replace(
    "_attention_output_projection_q8_input_sensitivity_linear_input_gap_gate",
    "_attention_output_projection_q8_input_sensitivity_linear_input_source_ffn_math_gap_gate")
DIAG_PREFIX = SEQ479.DIAG_PREFIX
DISPOSITION_PREFIX = SEQ479.DISPOSITION_PREFIX
CASES = SEQ479.CASES
TARGET_LAYERS = SEQ479.PREVIOUS_LAYERS
SOURCE_LAYERS = [layer - 1 for layer in TARGET_LAYERS if layer > 0]
EXPECTED_EVENTS = SEQ479.EXPECTED_EVENTS


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


def _rows_by_layer(step: dict[str, Any]) -> dict[int, dict[str, Any]]:
  layers = step.get("layers")
  layers = layers if isinstance(layers, list) else []
  out: dict[int, dict[str, Any]] = {}
  for row in layers:
    if isinstance(row, dict) and isinstance(row.get("layer"), int):
      out[int(row["layer"])] = row
  return out


def _min_update(summary: dict[str, Any], key: str, value: Any) -> None:
  if isinstance(value, (int, float)):
    summary[key] = min(_num(summary.get(key), 1.0), float(value))


def _max_update(summary: dict[str, Any], key: str, value: Any) -> None:
  if isinstance(value, (int, float)):
    summary[key] = max(_num(summary.get(key)), float(value))


def _case_summary(row: dict[str, Any]) -> dict[str, Any]:
  smoke = _load_smoke(row)
  boundary_steps = smoke.get("layer_boundary_diff_by_step")
  boundary_steps = boundary_steps if isinstance(boundary_steps, list) else []
  residual_steps = smoke.get("residual_source_diff_by_step")
  residual_steps = residual_steps if isinstance(residual_steps, list) else []
  residual_by_token = {
      step.get("token_index"): _rows_by_layer(step)
      for step in residual_steps
      if isinstance(step, dict)
  }
  summary: dict[str, Any] = {
      "case_id": row.get("case_id"),
      "out_dir": row.get("out_dir"),
      "observation_count": 0,
      "target_input_min_cosine": 1.0,
      "target_input_max_abs_diff": 0.0,
      "source_input_min_cosine": 1.0,
      "source_input_max_abs_diff": 0.0,
      "source_output_min_cosine": 1.0,
      "source_output_max_abs_diff": 0.0,
      "max_target_vs_source_output_cosine_delta": 0.0,
      "max_target_vs_source_output_abs_delta": 0.0,
      "source_attention_output_min_cosine": 1.0,
      "source_attention_output_max_abs_diff": 0.0,
      "source_ffn_input_min_cosine": 1.0,
      "source_ffn_input_max_abs_diff": 0.0,
      "source_ffn_delta_min_cosine": 1.0,
      "source_ffn_delta_max_abs_diff": 0.0,
      "source_cpu_ffn_from_gpu_input_min_cosine": 1.0,
      "source_cpu_ffn_from_gpu_input_max_abs_diff": 0.0,
      "source_gpu_output_vs_cpu_ffn_min_cosine": 1.0,
      "source_gpu_ffn_delta_vs_cpu_min_cosine": 1.0,
      "source_gpu_ffn_norm_vs_cpu_min_cosine": 1.0,
      "max_source_output_vs_cpu_ffn_cosine_delta": 0.0,
  }
  source_by_target = dict(zip(TARGET_LAYERS, SOURCE_LAYERS))
  for step in boundary_steps:
    if not isinstance(step, dict):
      continue
    token = step.get("token_index")
    boundary_rows = _rows_by_layer(step)
    residual_rows = residual_by_token.get(token, {})
    for target_layer, source_layer in source_by_target.items():
      target = boundary_rows.get(target_layer)
      source = boundary_rows.get(source_layer)
      residual = residual_rows.get(source_layer)
      if not target or not source or not residual:
        continue
      summary["observation_count"] += 1
      _min_update(summary, "target_input_min_cosine",
                  target.get("input_cosine"))
      _max_update(summary, "target_input_max_abs_diff",
                  target.get("input_max_abs_diff"))
      _min_update(summary, "source_input_min_cosine",
                  source.get("input_cosine"))
      _max_update(summary, "source_input_max_abs_diff",
                  source.get("input_max_abs_diff"))
      _min_update(summary, "source_output_min_cosine",
                  source.get("output_cosine"))
      _max_update(summary, "source_output_max_abs_diff",
                  source.get("output_max_abs_diff"))
      summary["max_target_vs_source_output_cosine_delta"] = max(
          summary["max_target_vs_source_output_cosine_delta"],
          abs(_num(target.get("input_cosine"), 1.0) -
              _num(source.get("output_cosine"), 1.0)))
      summary["max_target_vs_source_output_abs_delta"] = max(
          summary["max_target_vs_source_output_abs_delta"],
          abs(_num(target.get("input_max_abs_diff")) -
              _num(source.get("output_max_abs_diff"))))
      for metric in [
          "attention_output",
          "ffn_input",
          "ffn_delta",
          "cpu_ffn_from_gpu_input",
          "gpu_output_vs_cpu_ffn",
          "gpu_ffn_delta_vs_cpu",
          "gpu_ffn_norm_vs_cpu",
      ]:
        _min_update(summary, f"source_{metric}_min_cosine",
                    residual.get(f"{metric}_cosine"))
        _max_update(summary, f"source_{metric}_max_abs_diff",
                    residual.get(f"{metric}_max_abs_diff"))
      summary["max_source_output_vs_cpu_ffn_cosine_delta"] = max(
          summary["max_source_output_vs_cpu_ffn_cosine_delta"],
          abs(_num(source.get("output_cosine"), 1.0) -
              _num(residual.get("cpu_ffn_from_gpu_input_cosine"), 1.0)))
  return summary


def _min_case(rows: list[dict[str, Any]], key: str) -> float:
  return min((_num(row.get(key), 1.0) for row in rows), default=1.0)


def _max_case(rows: list[dict[str, Any]], key: str) -> float:
  return max((_num(row.get(key)) for row in rows), default=0.0)


def compute(args: argparse.Namespace) -> dict[str, Any]:
  routes = _load_json(args.routes)
  seq479 = _load_json(args.seq479)
  case_rows = seq479.get("runs")
  case_rows = case_rows if isinstance(case_rows, list) else []
  rows = [_case_summary(row) for row in case_rows if isinstance(row, dict)]
  preconditions_pass = (
      seq479.get("required_checks_passed") is True
      and seq479.get("selected_next_route") == CURRENT_ROUTE
      and seq479.get("diagnostic_classification")
      == f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_input_attention_output_projection_q8_input_sensitivity_linear_input_gap"
      and _has_candidate(routes, 479, str(seq479.get("disposition")))
      and _has_switch(routes, f"select_{CURRENT_ROUTE}", 479)
  )
  rows_loaded = (
      len(rows) == len(CASES)
      and all(row.get("observation_count") == EXPECTED_EVENTS for row in rows)
  )
  seq479_linear_input_gap = (
      seq479.get("linear_input_gap") is True
      and seq479.get("preconv_math_gap") is False
      and seq479.get("final_kernel_gap") is False)

  target_input_min = _min_case(rows, "target_input_min_cosine")
  target_input_max_abs = _max_case(rows, "target_input_max_abs_diff")
  source_output_min = _min_case(rows, "source_output_min_cosine")
  source_output_max_abs = _max_case(rows, "source_output_max_abs_diff")
  source_input_min = _min_case(rows, "source_input_min_cosine")
  source_input_max_abs = _max_case(rows, "source_input_max_abs_diff")
  max_target_source_cos_delta = _max_case(
      rows, "max_target_vs_source_output_cosine_delta")
  max_target_source_abs_delta = _max_case(
      rows, "max_target_vs_source_output_abs_delta")

  target_input_is_source_output = (
      rows_loaded
      and max_target_source_cos_delta <= SOURCE_MATCH_EPS
      and max_target_source_abs_delta <= SOURCE_MATCH_EPS)
  target_input_drift_observed = (
      rows_loaded
      and target_input_min < 1.0
      and target_input_max_abs > 0.0)
  source_layer_adds_drift = (
      rows_loaded
      and source_output_min < source_input_min
      and source_output_max_abs > source_input_max_abs)

  attention_output_min = _min_case(rows, "source_attention_output_min_cosine")
  ffn_input_min = _min_case(rows, "source_ffn_input_min_cosine")
  ffn_delta_min = _min_case(rows, "source_ffn_delta_min_cosine")
  cpu_ffn_from_input_min = _min_case(
      rows, "source_cpu_ffn_from_gpu_input_min_cosine")
  gpu_output_vs_cpu_ffn_min = _min_case(
      rows, "source_gpu_output_vs_cpu_ffn_min_cosine")
  gpu_ffn_delta_vs_cpu_min = _min_case(
      rows, "source_gpu_ffn_delta_vs_cpu_min_cosine")
  gpu_ffn_norm_vs_cpu_min = _min_case(
      rows, "source_gpu_ffn_norm_vs_cpu_min_cosine")
  max_source_cpu_ffn_delta = _max_case(
      rows, "max_source_output_vs_cpu_ffn_cosine_delta")

  attention_output_clean = attention_output_min >= COSINE_THRESHOLD
  ffn_input_clean = ffn_input_min >= COSINE_THRESHOLD
  ffn_delta_gap = ffn_delta_min < COSINE_THRESHOLD
  gpu_ffn_math_ok = (
      gpu_output_vs_cpu_ffn_min >= COSINE_THRESHOLD
      and gpu_ffn_delta_vs_cpu_min >= COSINE_THRESHOLD
      and gpu_ffn_norm_vs_cpu_min >= COSINE_THRESHOLD)
  cpu_ffn_replays_source_output = (
      max_source_cpu_ffn_delta <= SOURCE_REPLAY_EPS
      and cpu_ffn_from_input_min < 1.0)

  ffn_delta_source_gap = (
      seq479_linear_input_gap
      and target_input_is_source_output
      and target_input_drift_observed
      and source_layer_adds_drift
      and attention_output_clean
      and ffn_input_clean
      and ffn_delta_gap
      and gpu_ffn_math_ok
      and cpu_ffn_replays_source_output)
  attention_output_source_gap = (
      seq479_linear_input_gap
      and target_input_is_source_output
      and attention_output_min < COSINE_THRESHOLD
      and ffn_input_min < COSINE_THRESHOLD
      and not ffn_delta_gap)
  ffn_math_gap = (
      seq479_linear_input_gap
      and target_input_is_source_output
      and not gpu_ffn_math_ok)

  diagnostic_classification = (
      f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_input_attention_output_projection_q8_input_sensitivity_linear_input_source_ffn_delta_gap"
      if ffn_delta_source_gap else
      f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_input_attention_output_projection_q8_input_sensitivity_linear_input_source_attention_output_gap"
      if attention_output_source_gap else
      f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_input_attention_output_projection_q8_input_sensitivity_linear_input_source_ffn_math_gap"
      if ffn_math_gap else
      f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_input_attention_output_projection_q8_input_sensitivity_linear_input_gap_unclassified"
  )
  selected_next = (
      FFN_DELTA_ROUTE
      if ffn_delta_source_gap else
      ATTENTION_OUTPUT_ROUTE
      if attention_output_source_gap else
      FFN_MATH_ROUTE
      if ffn_math_gap else
      CURRENT_ROUTE)
  checks = [
      {"name": "seq479_selected_linear_input_gap_gate",
       "pass": preconditions_pass},
      {"name": "seq479_source_rows_loaded",
       "pass": rows_loaded,
       "detail": rows},
      {"name": "seq479_q8_sensitive_linear_input_gap_reproduced",
       "pass": seq479_linear_input_gap},
      {"name": "target_linear_input_is_previous_layer_output",
       "pass": target_input_is_source_output,
       "detail": {
           "max_target_vs_source_output_cosine_delta": (
               max_target_source_cos_delta),
           "max_target_vs_source_output_abs_delta": (
               max_target_source_abs_delta),
       }},
      {"name": "target_linear_input_drift_observed",
       "pass": target_input_drift_observed,
       "detail": {
           "target_input_min_cosine": target_input_min,
           "target_input_max_abs_diff": target_input_max_abs,
       }},
      {"name": "previous_layer_adds_drift",
       "pass": source_layer_adds_drift,
       "detail": {
           "source_input_min_cosine": source_input_min,
           "source_output_min_cosine": source_output_min,
           "source_input_max_abs_diff": source_input_max_abs,
           "source_output_max_abs_diff": source_output_max_abs,
       }},
      {"name": "previous_layer_ffn_delta_source_classified",
       "pass": ffn_delta_source_gap,
       "detail": {
           "attention_output_min_cosine": attention_output_min,
           "ffn_input_min_cosine": ffn_input_min,
           "ffn_delta_min_cosine": ffn_delta_min,
           "cpu_ffn_from_gpu_input_min_cosine": cpu_ffn_from_input_min,
           "gpu_output_vs_cpu_ffn_min_cosine": gpu_output_vs_cpu_ffn_min,
           "gpu_ffn_delta_vs_cpu_min_cosine": gpu_ffn_delta_vs_cpu_min,
           "gpu_ffn_norm_vs_cpu_min_cosine": gpu_ffn_norm_vs_cpu_min,
           "max_source_output_vs_cpu_ffn_cosine_delta": (
               max_source_cpu_ffn_delta),
       }},
      {"name": "linear_input_source_classified",
       "pass": selected_next != CURRENT_ROUTE,
       "detail": {
           "ffn_delta_source_gap": ffn_delta_source_gap,
           "attention_output_source_gap": attention_output_source_gap,
           "ffn_math_gap": ffn_math_gap,
       }},
  ]
  required = all(bool(row.get("pass")) for row in checks)
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "inputs": {
          "routes": _rel(args.routes),
          "seq479": _rel(args.seq479),
          "target_layers": TARGET_LAYERS,
          "source_layers": SOURCE_LAYERS,
      },
      "runs": rows,
      "checks": checks,
      "required_checks_passed": required,
      "diagnostic_classification": diagnostic_classification,
      "ffn_delta_source_gap": ffn_delta_source_gap,
      "attention_output_source_gap": attention_output_source_gap,
      "ffn_math_gap": ffn_math_gap,
      "target_input_is_source_output": target_input_is_source_output,
      "source_layer_adds_drift": source_layer_adds_drift,
      "min_cosines": {
          "target_input": target_input_min,
          "source_input": source_input_min,
          "source_output": source_output_min,
          "source_attention_output": attention_output_min,
          "source_ffn_input": ffn_input_min,
          "source_ffn_delta": ffn_delta_min,
          "source_cpu_ffn_from_gpu_input": cpu_ffn_from_input_min,
          "source_gpu_output_vs_cpu_ffn": gpu_output_vs_cpu_ffn_min,
          "source_gpu_ffn_delta_vs_cpu": gpu_ffn_delta_vs_cpu_min,
          "source_gpu_ffn_norm_vs_cpu": gpu_ffn_norm_vs_cpu_min,
      },
      "max_abs_diffs": {
          "target_input": target_input_max_abs,
          "source_input": source_input_max_abs,
          "source_output": source_output_max_abs,
      },
      "source_match": {
          "max_target_vs_source_output_cosine_delta": (
              max_target_source_cos_delta),
          "max_target_vs_source_output_abs_delta": (
              max_target_source_abs_delta),
          "max_source_output_vs_cpu_ffn_cosine_delta": (
              max_source_cpu_ffn_delta),
      },
      "speedup_claims_allowed": False,
      "disposition": (
          f"accept_{DISPOSITION_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_input_attention_output_projection_q8_input_sensitivity_linear_input_source_gap_classification"
          if required else
          f"block_{DISPOSITION_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_input_attention_output_projection_q8_input_sensitivity_linear_input_source_gap_classification"
      ),
      "selected_next_route": selected_next if required else CURRENT_ROUTE,
      "next_route_reason": (
          "The Q8-sensitive target linear input is exactly the previous layer "
          "output; previous layer input and FFN input stay clean at the cosine "
          "gate, GPU FFN math matches CPU, and the previous FFN delta is the "
          "first below-threshold source. Root previous FFN delta next."
          if required and selected_next == FFN_DELTA_ROUTE else
          "The Q8-sensitive target linear input is sourced from previous "
          "attention output. Root previous attention output next."
          if required and selected_next == ATTENTION_OUTPUT_ROUTE else
          "The Q8-sensitive target linear input source has a GPU-vs-CPU FFN "
          "math mismatch. Root FFN math next."
          if required and selected_next == FFN_MATH_ROUTE else
          "Linear input source evidence is incomplete; keep this gate open."
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
      "# Seq480 Projection Q8 Input-Sensitivity Linear Input Source Gap Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- diagnostic_classification: `{metrics['diagnostic_classification']}`",
      f"- target/source input/output cosines: `{c['target_input']}` / `{c['source_input']}` / `{c['source_output']}`",
      f"- target/source input/output max abs: `{a['target_input']}` / `{a['source_input']}` / `{a['source_output']}`",
      f"- previous attention/ffn-input/ffn-delta cosines: `{c['source_attention_output']}` / `{c['source_ffn_input']}` / `{c['source_ffn_delta']}`",
      f"- previous CPU-FFN replay/GPU-vs-CPU cosines: `{c['source_cpu_ffn_from_gpu_input']}` / `{c['source_gpu_output_vs_cpu_ffn']}`",
      f"- target-vs-source deltas cosine/abs: `{s['max_target_vs_source_output_cosine_delta']}` / `{s['max_target_vs_source_output_abs_delta']}`",
      f"- ffn_delta_source_gap: `{str(metrics['ffn_delta_source_gap']).lower()}`",
      f"- attention_output_source_gap: `{str(metrics['attention_output_source_gap']).lower()}`",
      f"- ffn_math_gap: `{str(metrics['ffn_math_gap']).lower()}`",
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
  parser.add_argument("--seq479", type=Path, default=DEFAULT_SEQ479)
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
