#!/usr/bin/env python3
"""Classify seq487 coupled linear delta/z drift to live linear input."""

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
SEQ487_GATE = (
    ROOT
    / "tools/intel-qwen36-seq487-attention-output-projection-q8-input-sensitivity-final-mix-gap-gate.py"
)
SCHEMA_VERSION = (
    "intel-qwen36-seq488-projection-q8-input-sensitivity-linear-delta-z-source-gap-gate-v0"
)

DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_SEQ487 = (
    ROOT
    / "output/seq487-attention-output-projection-q8-input-sensitivity-final-mix-gap-gate-20260709Tseq487Z"
    / "metrics.json"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "output/seq488-projection-q8-input-sensitivity-linear-delta-z-source-gap-gate-20260709Tseq488Z"
)

COSINE_THRESHOLD = 0.9999
EXPECTED_EVENTS = 8
SAME_PROJECTION_EPS = 1.0e-7

ATTENTION_METRICS = [
    "attn_norm",
    "qkv_mixed",
    "z",
    "q_predelta",
    "k_predelta",
    "v_predelta",
    "delta_output",
    "final_output",
]
PRECONV_METRICS = [
    "attn_norm_from_gpu_input",
    "gpu_attn_norm_vs_cpu",
    "qkv_from_gpu_attn_norm",
    "gpu_qkv_vs_cpu",
    "gate_from_gpu_attn_norm",
    "gpu_gate_vs_cpu",
    "beta_from_gpu_attn_norm",
    "gpu_beta_vs_cpu",
    "z_from_gpu_attn_norm",
    "gpu_z_vs_cpu",
    "conv_output_raw",
]
FINAL_MIX_METRICS = [
    "gpu_kernel_final",
    "native_delta_native_z_cpu",
    "gpu_delta_native_z_cpu",
    "native_delta_gpu_z_cpu",
    "gpu_delta_gpu_z_cpu",
]


def _load_module(path: Path, name: str) -> Any:
  spec = importlib.util.spec_from_file_location(name, path)
  if spec is None or spec.loader is None:
    raise SystemExit(f"cannot load module {name}: {path}")
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


SEQ487 = _load_module(SEQ487_GATE, "iq36_seq487_gate")
CURRENT_ROUTE = SEQ487.LINEAR_DELTA_Z_ROUTE
LINEAR_INPUT_ROUTE = CURRENT_ROUTE.replace(
    "_ffn_norm_input_sensitivity_attention_output_projection_q8_input_sensitivity_linear_delta_z_gap_gate",
    "_ffn_norm_input_sensitivity_attention_output_projection_q8_input_sensitivity_linear_input_gap_gate")
PRECONV_MATH_ROUTE = CURRENT_ROUTE.replace(
    "_ffn_norm_input_sensitivity_attention_output_projection_q8_input_sensitivity_linear_delta_z_gap_gate",
    "_ffn_norm_input_sensitivity_attention_output_projection_q8_input_sensitivity_linear_preconv_math_gap_gate")
FINAL_KERNEL_ROUTE = CURRENT_ROUTE.replace(
    "_ffn_norm_input_sensitivity_attention_output_projection_q8_input_sensitivity_linear_delta_z_gap_gate",
    "_ffn_norm_input_sensitivity_attention_output_projection_q8_input_sensitivity_linear_final_kernel_gap_gate")
DIAG_PREFIX = SEQ487.DIAG_PREFIX
DISPOSITION_PREFIX = SEQ487.DISPOSITION_PREFIX
EXPECTED_CASES = SEQ487.SEQ486.SEQ485.EXPECTED_CASES


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


def _metric_summary(steps: list[Any], source_layers: list[int],
                    metric_names: list[str]) -> dict[str, Any]:
  selected = set(source_layers)
  out: dict[str, Any] = {"observation_count": 0}
  for name in metric_names:
    out[f"min_{name}_cosine"] = 1.0
    out[f"max_{name}_abs_diff"] = 0.0
  for step in steps:
    if not isinstance(step, dict):
      continue
    layers = step.get("layers")
    layers = layers if isinstance(layers, list) else []
    for row in layers:
      if not isinstance(row, dict) or row.get("layer") not in selected:
        continue
      observed = False
      for name in metric_names:
        if row.get(f"{name}_available") is False:
          continue
        cosine = row.get(f"{name}_cosine")
        max_abs = row.get(f"{name}_max_abs_diff")
        if isinstance(cosine, (int, float)):
          observed = True
          out[f"min_{name}_cosine"] = min(
              out[f"min_{name}_cosine"], float(cosine))
        if isinstance(max_abs, (int, float)):
          out[f"max_{name}_abs_diff"] = max(
              out[f"max_{name}_abs_diff"], float(max_abs))
      if observed:
        out["observation_count"] += 1
  return out


def _boundary_summary(steps: list[Any], source_layers: list[int]) -> dict[str, Any]:
  selected = set(source_layers)
  out = {
      "observation_count": 0,
      "min_input_cosine": 1.0,
      "max_input_abs_diff": 0.0,
      "min_output_cosine": 1.0,
      "max_output_abs_diff": 0.0,
  }
  for step in steps:
    if not isinstance(step, dict):
      continue
    layers = step.get("layers")
    layers = layers if isinstance(layers, list) else []
    for row in layers:
      if not isinstance(row, dict) or row.get("layer") not in selected:
        continue
      out["observation_count"] += 1
      out["min_input_cosine"] = min(
          out["min_input_cosine"], _num(row.get("input_cosine"), 1.0))
      out["max_input_abs_diff"] = max(
          out["max_input_abs_diff"], _num(row.get("input_max_abs_diff")))
      out["min_output_cosine"] = min(
          out["min_output_cosine"], _num(row.get("output_cosine"), 1.0))
      out["max_output_abs_diff"] = max(
          out["max_output_abs_diff"], _num(row.get("output_max_abs_diff")))
  return out


def _load_smoke(row: dict[str, Any]) -> dict[str, Any]:
  out_dir = row.get("out_dir")
  if not isinstance(out_dir, str):
    return {}
  smoke_path = ROOT / out_dir / "smoke.json"
  if not smoke_path.exists():
    return {}
  smoke = _load_json(smoke_path)
  return smoke if isinstance(smoke, dict) else {}


def _case_summary(row: dict[str, Any]) -> dict[str, Any]:
  smoke = _load_smoke(row)
  source_layers = row.get("source_layers")
  source_layers = source_layers if isinstance(source_layers, list) else [1]
  source_layers = [int(layer) for layer in source_layers if isinstance(layer, int)]
  attention_steps = smoke.get("linear_attention_diff_by_step")
  attention_steps = attention_steps if isinstance(attention_steps, list) else []
  preconv_steps = smoke.get("linear_preconv_source_diff_by_step")
  preconv_steps = preconv_steps if isinstance(preconv_steps, list) else []
  final_mix_steps = smoke.get("linear_final_mix_diff_by_step")
  final_mix_steps = final_mix_steps if isinstance(final_mix_steps, list) else []
  boundary_steps = smoke.get("layer_boundary_diff_by_step")
  boundary_steps = boundary_steps if isinstance(boundary_steps, list) else []
  return {
      "case_id": row.get("case_id"),
      "out_dir": row.get("out_dir"),
      "source_layers": source_layers,
      "attention": _metric_summary(
          attention_steps, source_layers, ATTENTION_METRICS),
      "preconv": _metric_summary(
          preconv_steps, source_layers, PRECONV_METRICS),
      "final_mix": _metric_summary(
          final_mix_steps, source_layers, FINAL_MIX_METRICS),
      "boundary": _boundary_summary(boundary_steps, source_layers),
  }


def _min_case(rows: list[dict[str, Any]], group: str, key: str) -> float:
  return min(
      (_num(row.get(group, {}).get(key), 1.0) for row in rows),
      default=1.0)


def _max_case(rows: list[dict[str, Any]], group: str, key: str) -> float:
  return max(
      (_num(row.get(group, {}).get(key)) for row in rows),
      default=0.0)


def _all_observed(rows: list[dict[str, Any]], group: str) -> bool:
  return all(
      row.get(group, {}).get("observation_count") == EXPECTED_EVENTS
      for row in rows)


def _close(left: float, right: float) -> bool:
  return abs(left - right) <= SAME_PROJECTION_EPS


def compute(args: argparse.Namespace) -> dict[str, Any]:
  routes = _load_json(args.routes)
  seq487 = _load_json(args.seq487)
  case_rows = seq487.get("runs")
  case_rows = case_rows if isinstance(case_rows, list) else []
  rows = [_case_summary(row) for row in case_rows if isinstance(row, dict)]

  preconditions_pass = (
      seq487.get("required_checks_passed") is True
      and seq487.get("selected_next_route") == CURRENT_ROUTE
      and seq487.get("diagnostic_classification")
      == f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_input_attention_output_projection_q8_input_sensitivity_linear_input_source_ffn_delta_selected_swiglu_gate_up_input_sensitivity_ffn_norm_input_sensitivity_attention_output_projection_q8_input_sensitivity_linear_delta_z_gap"
      and _has_candidate(routes, 487, str(seq487.get("disposition")))
      and _has_switch(routes, f"select_{CURRENT_ROUTE}", 487)
  )
  diagnostics_loaded = (
      len(rows) == len(EXPECTED_CASES)
      and _all_observed(rows, "attention")
      and _all_observed(rows, "preconv")
      and _all_observed(rows, "final_mix")
      and _all_observed(rows, "boundary")
  )

  seq487_min = seq487.get("min_cosines", {})
  seq487_q8 = seq487.get("q8_mismatches", {})
  final_projection = _num(
      seq487_min.get("gpu_final_projection_vs_native"), 1.0)
  both_projection = _num(
      seq487_min.get("gpu_delta_gpu_z_projection_vs_native"), 1.0)
  final_qs = int(_num(seq487_q8.get("gpu_final", {})
                      .get("max_qs_mismatch_count")))
  both_qs = int(_num(seq487_q8.get("gpu_delta_gpu_z", {})
                     .get("max_qs_mismatch_count")))
  projection_delta_z_reproduced = (
      diagnostics_loaded
      and final_projection >= COSINE_THRESHOLD
      and final_projection < 1.0
      and both_projection >= COSINE_THRESHOLD
      and both_projection < 1.0
      and _close(final_projection, both_projection)
      and final_qs > 0
      and final_qs == both_qs)

  min_boundary_input = _min_case(rows, "boundary", "min_input_cosine")
  max_boundary_input_abs = _max_case(rows, "boundary", "max_input_abs_diff")
  min_boundary_output = _min_case(rows, "boundary", "min_output_cosine")
  max_boundary_output_abs = _max_case(rows, "boundary", "max_output_abs_diff")
  min_attn_norm_from_input = _min_case(
      rows, "preconv", "min_attn_norm_from_gpu_input_cosine")
  max_attn_norm_from_input_abs = _max_case(
      rows, "preconv", "max_attn_norm_from_gpu_input_abs_diff")
  live_input_drift_observed = (
      diagnostics_loaded
      and min_boundary_input < 1.0
      and max_boundary_input_abs > 0.0
      and min_attn_norm_from_input < 1.0
      and max_attn_norm_from_input_abs > 0.0)

  min_gpu_attn_norm_vs_cpu = _min_case(
      rows, "preconv", "min_gpu_attn_norm_vs_cpu_cosine")
  min_gpu_qkv_vs_cpu = _min_case(
      rows, "preconv", "min_gpu_qkv_vs_cpu_cosine")
  min_gpu_gate_vs_cpu = _min_case(
      rows, "preconv", "min_gpu_gate_vs_cpu_cosine")
  min_gpu_beta_vs_cpu = _min_case(
      rows, "preconv", "min_gpu_beta_vs_cpu_cosine")
  min_gpu_z_vs_cpu = _min_case(
      rows, "preconv", "min_gpu_z_vs_cpu_cosine")
  preconv_math_ok = (
      diagnostics_loaded
      and min_gpu_attn_norm_vs_cpu >= COSINE_THRESHOLD
      and min_gpu_qkv_vs_cpu >= COSINE_THRESHOLD
      and min_gpu_gate_vs_cpu >= COSINE_THRESHOLD
      and min_gpu_beta_vs_cpu >= COSINE_THRESHOLD
      and min_gpu_z_vs_cpu >= COSINE_THRESHOLD)

  min_qkv_from_norm = _min_case(
      rows, "preconv", "min_qkv_from_gpu_attn_norm_cosine")
  min_gate_from_norm = _min_case(
      rows, "preconv", "min_gate_from_gpu_attn_norm_cosine")
  min_beta_from_norm = _min_case(
      rows, "preconv", "min_beta_from_gpu_attn_norm_cosine")
  min_z_from_norm = _min_case(
      rows, "preconv", "min_z_from_gpu_attn_norm_cosine")
  max_qkv_from_norm_abs = _max_case(
      rows, "preconv", "max_qkv_from_gpu_attn_norm_abs_diff")
  max_z_from_norm_abs = _max_case(
      rows, "preconv", "max_z_from_gpu_attn_norm_abs_diff")
  preconv_outputs_inherit_input_drift = (
      diagnostics_loaded
      and min_qkv_from_norm < 1.0
      and min_gate_from_norm < 1.0
      and min_beta_from_norm < 1.0
      and min_z_from_norm < 1.0
      and max_qkv_from_norm_abs > 0.0
      and max_z_from_norm_abs > 0.0)

  min_gpu_kernel_final = _min_case(
      rows, "final_mix", "min_gpu_kernel_final_cosine")
  min_native_recompute = _min_case(
      rows, "final_mix", "min_native_delta_native_z_cpu_cosine")
  min_gpu_delta_native_z_cpu = _min_case(
      rows, "final_mix", "min_gpu_delta_native_z_cpu_cosine")
  min_native_delta_gpu_z_cpu = _min_case(
      rows, "final_mix", "min_native_delta_gpu_z_cpu_cosine")
  min_gpu_delta_gpu_z_cpu = _min_case(
      rows, "final_mix", "min_gpu_delta_gpu_z_cpu_cosine")
  final_mix_consistent = (
      diagnostics_loaded
      and min_native_recompute >= COSINE_THRESHOLD
      and min_gpu_kernel_final >= COSINE_THRESHOLD
      and min_gpu_delta_native_z_cpu >= COSINE_THRESHOLD
      and min_native_delta_gpu_z_cpu >= COSINE_THRESHOLD
      and min_gpu_delta_gpu_z_cpu >= COSINE_THRESHOLD)

  linear_input_gap = (
      projection_delta_z_reproduced
      and live_input_drift_observed
      and preconv_math_ok
      and preconv_outputs_inherit_input_drift
      and final_mix_consistent)
  preconv_math_gap = diagnostics_loaded and not preconv_math_ok
  final_kernel_gap = (
      diagnostics_loaded and not final_mix_consistent
      and projection_delta_z_reproduced)
  diagnostic_classification = (
      f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_input_attention_output_projection_q8_input_sensitivity_linear_input_source_ffn_delta_selected_swiglu_gate_up_input_sensitivity_ffn_norm_input_sensitivity_attention_output_projection_q8_input_sensitivity_linear_input_gap"
      if linear_input_gap else
      f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_input_attention_output_projection_q8_input_sensitivity_linear_input_source_ffn_delta_selected_swiglu_gate_up_input_sensitivity_ffn_norm_input_sensitivity_attention_output_projection_q8_input_sensitivity_linear_preconv_math_gap"
      if preconv_math_gap else
      f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_input_attention_output_projection_q8_input_sensitivity_linear_input_source_ffn_delta_selected_swiglu_gate_up_input_sensitivity_ffn_norm_input_sensitivity_attention_output_projection_q8_input_sensitivity_linear_final_kernel_gap"
      if final_kernel_gap else
      f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_input_attention_output_projection_q8_input_sensitivity_linear_delta_z_gap_unclassified"
  )
  selected_next = (
      LINEAR_INPUT_ROUTE
      if linear_input_gap else
      PRECONV_MATH_ROUTE
      if preconv_math_gap else
      FINAL_KERNEL_ROUTE
      if final_kernel_gap else
      CURRENT_ROUTE)

  checks = [
      {"name": "seq487_selected_linear_delta_z_gap_gate",
       "pass": preconditions_pass},
      {"name": "linear_delta_z_source_diagnostics_loaded",
       "pass": diagnostics_loaded,
       "detail": rows},
      {"name": "projection_delta_z_q8_sensitivity_reproduced",
       "pass": projection_delta_z_reproduced,
       "detail": {
           "seq487_gpu_final_projection_cosine": final_projection,
           "seq487_gpu_delta_gpu_z_projection_cosine": both_projection,
           "seq487_gpu_final_q8_qs_mismatch_count": final_qs,
           "seq487_gpu_delta_gpu_z_q8_qs_mismatch_count": both_qs,
       }},
      {"name": "live_linear_input_drift_observed",
       "pass": live_input_drift_observed,
       "detail": {
           "min_boundary_input_cosine": min_boundary_input,
           "max_boundary_input_abs_diff": max_boundary_input_abs,
           "min_boundary_output_cosine": min_boundary_output,
           "max_boundary_output_abs_diff": max_boundary_output_abs,
           "min_attn_norm_from_gpu_input_cosine": min_attn_norm_from_input,
           "max_attn_norm_from_gpu_input_abs_diff": (
               max_attn_norm_from_input_abs),
       }},
      {"name": "preconv_math_matches_cpu_on_live_input",
       "pass": preconv_math_ok,
       "detail": {
           "min_gpu_attn_norm_vs_cpu_cosine": min_gpu_attn_norm_vs_cpu,
           "min_gpu_qkv_vs_cpu_cosine": min_gpu_qkv_vs_cpu,
           "min_gpu_gate_vs_cpu_cosine": min_gpu_gate_vs_cpu,
           "min_gpu_beta_vs_cpu_cosine": min_gpu_beta_vs_cpu,
           "min_gpu_z_vs_cpu_cosine": min_gpu_z_vs_cpu,
       }},
      {"name": "preconv_outputs_inherit_live_input_drift",
       "pass": preconv_outputs_inherit_input_drift,
       "detail": {
           "min_qkv_from_gpu_attn_norm_cosine": min_qkv_from_norm,
           "min_gate_from_gpu_attn_norm_cosine": min_gate_from_norm,
           "min_beta_from_gpu_attn_norm_cosine": min_beta_from_norm,
           "min_z_from_gpu_attn_norm_cosine": min_z_from_norm,
           "max_qkv_from_gpu_attn_norm_abs_diff": max_qkv_from_norm_abs,
           "max_z_from_gpu_attn_norm_abs_diff": max_z_from_norm_abs,
       }},
      {"name": "final_mix_remains_vector_clean_under_cosine_gate",
       "pass": final_mix_consistent,
       "detail": {
           "min_gpu_kernel_final_cosine": min_gpu_kernel_final,
           "min_native_delta_native_z_cpu_cosine": min_native_recompute,
           "min_gpu_delta_native_z_cpu_cosine": min_gpu_delta_native_z_cpu,
           "min_native_delta_gpu_z_cpu_cosine": min_native_delta_gpu_z_cpu,
           "min_gpu_delta_gpu_z_cpu_cosine": min_gpu_delta_gpu_z_cpu,
       }},
      {"name": "linear_delta_z_source_classified",
       "pass": selected_next != CURRENT_ROUTE,
       "detail": {
           "linear_input_gap": linear_input_gap,
           "preconv_math_gap": preconv_math_gap,
           "final_kernel_gap": final_kernel_gap,
       }},
  ]
  required = all(bool(row.get("pass")) for row in checks)
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "inputs": {
          "routes": _rel(args.routes),
          "seq487": _rel(args.seq487),
      },
      "runs": rows,
      "checks": checks,
      "required_checks_passed": required,
      "diagnostic_classification": diagnostic_classification,
      "linear_input_gap": linear_input_gap,
      "preconv_math_gap": preconv_math_gap,
      "final_kernel_gap": final_kernel_gap,
      "projection_delta_z_reproduced": projection_delta_z_reproduced,
      "live_input_drift_observed": live_input_drift_observed,
      "preconv_math_ok": preconv_math_ok,
      "preconv_outputs_inherit_input_drift": (
          preconv_outputs_inherit_input_drift),
      "final_mix_consistent": final_mix_consistent,
      "min_cosines": {
          "boundary_input": min_boundary_input,
          "boundary_output": min_boundary_output,
          "attn_norm_from_gpu_input": min_attn_norm_from_input,
          "qkv_from_gpu_attn_norm": min_qkv_from_norm,
          "gate_from_gpu_attn_norm": min_gate_from_norm,
          "beta_from_gpu_attn_norm": min_beta_from_norm,
          "z_from_gpu_attn_norm": min_z_from_norm,
          "gpu_attn_norm_vs_cpu": min_gpu_attn_norm_vs_cpu,
          "gpu_qkv_vs_cpu": min_gpu_qkv_vs_cpu,
          "gpu_gate_vs_cpu": min_gpu_gate_vs_cpu,
          "gpu_beta_vs_cpu": min_gpu_beta_vs_cpu,
          "gpu_z_vs_cpu": min_gpu_z_vs_cpu,
          "gpu_kernel_final": min_gpu_kernel_final,
          "native_delta_native_z_cpu": min_native_recompute,
          "gpu_delta_native_z_cpu": min_gpu_delta_native_z_cpu,
          "native_delta_gpu_z_cpu": min_native_delta_gpu_z_cpu,
          "gpu_delta_gpu_z_cpu": min_gpu_delta_gpu_z_cpu,
      },
      "max_abs_diffs": {
          "boundary_input": max_boundary_input_abs,
          "boundary_output": max_boundary_output_abs,
          "attn_norm_from_gpu_input": max_attn_norm_from_input_abs,
          "qkv_from_gpu_attn_norm": max_qkv_from_norm_abs,
          "z_from_gpu_attn_norm": max_z_from_norm_abs,
      },
      "speedup_claims_allowed": False,
      "disposition": (
          f"accept_{DISPOSITION_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_input_attention_output_projection_q8_input_sensitivity_linear_input_source_ffn_delta_selected_swiglu_gate_up_input_sensitivity_ffn_norm_input_sensitivity_attention_output_projection_q8_input_sensitivity_linear_delta_z_source_gap_classification"
          if required else
          f"block_{DISPOSITION_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_input_attention_output_projection_q8_input_sensitivity_linear_input_source_ffn_delta_selected_swiglu_gate_up_input_sensitivity_ffn_norm_input_sensitivity_attention_output_projection_q8_input_sensitivity_linear_delta_z_source_gap_classification"
      ),
      "selected_next_route": selected_next if required else CURRENT_ROUTE,
      "next_route_reason": (
          "The coupled delta/z projection-Q8 sensitivity follows live linear "
          "input into attention-norm/qkv/gate/beta/z, while GPU preconv math "
          "matches CPU and the final-mix vectors stay above the cosine gate. "
          "Root live linear input next."
          if required and selected_next == LINEAR_INPUT_ROUTE else
          "GPU preconv math no longer matches CPU on live input. Root linear "
          "preconv math next."
          if required and selected_next == PRECONV_MATH_ROUTE else
          "Final-mix vectors no longer stay clean under the cosine gate. Root "
          "final kernel/normalization next."
          if required and selected_next == FINAL_KERNEL_ROUTE else
          "Linear delta/z source evidence is incomplete; keep this gate open."
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
      "# Seq488 Projection Q8 Input-Sensitivity Linear Delta/Z Source Gap Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- diagnostic_classification: `{metrics['diagnostic_classification']}`",
      f"- min boundary input/output/attn-norm/qkv/z cosines: `{c['boundary_input']}` / `{c['boundary_output']}` / `{c['attn_norm_from_gpu_input']}` / `{c['qkv_from_gpu_attn_norm']}` / `{c['z_from_gpu_attn_norm']}`",
      f"- max boundary input/output/attn-norm/qkv/z abs: `{a['boundary_input']}` / `{a['boundary_output']}` / `{a['attn_norm_from_gpu_input']}` / `{a['qkv_from_gpu_attn_norm']}` / `{a['z_from_gpu_attn_norm']}`",
      f"- gpu preconv math cosines attn/qkv/gate/beta/z: `{c['gpu_attn_norm_vs_cpu']}` / `{c['gpu_qkv_vs_cpu']}` / `{c['gpu_gate_vs_cpu']}` / `{c['gpu_beta_vs_cpu']}` / `{c['gpu_z_vs_cpu']}`",
      f"- final mix cosines kernel/native/delta/z/both: `{c['gpu_kernel_final']}` / `{c['native_delta_native_z_cpu']}` / `{c['gpu_delta_native_z_cpu']}` / `{c['native_delta_gpu_z_cpu']}` / `{c['gpu_delta_gpu_z_cpu']}`",
      f"- linear_input_gap: `{str(metrics['linear_input_gap']).lower()}`",
      f"- preconv_math_gap: `{str(metrics['preconv_math_gap']).lower()}`",
      f"- final_kernel_gap: `{str(metrics['final_kernel_gap']).lower()}`",
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
  parser.add_argument("--seq487", type=Path, default=DEFAULT_SEQ487)
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
