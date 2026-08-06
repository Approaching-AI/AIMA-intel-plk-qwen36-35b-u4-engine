#!/usr/bin/env python3
"""Classify seq476 attention output-projection Q8 bridge drift."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SEQ476_GATE = (
    ROOT
    / "tools/intel-qwen36-seq476-upstream-layer-output-ffn-delta-ffn-norm-input-attention-output-projection-math-gap-gate.py"
)
SMOKE_SOURCE = ROOT / "tools/intel-qwen36-r2-gpu-decode-smoke.py"
SCHEMA_VERSION = (
    "intel-qwen36-seq477-attention-output-projection-q8-bridge-gap-gate-v0"
)

DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_SEQ476 = (
    ROOT
    / "output/seq476-upstream-layer-output-ffn-delta-ffn-norm-input-attention-output-projection-math-gap-gate-20260709Tseq476Z"
    / "metrics.json"
)
DEFAULT_TOKEN_INPUT_DIR = (
    ROOT
    / "output/r2-gpu-acceptance-matrix-router-both-rowblock16-26mask-20260708Tseq222Z"
    / "token-input"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "output/seq477-attention-output-projection-q8-bridge-gap-gate-20260709Tseq477Z"
)
DEFAULT_HOST = "local"
DEFAULT_MODEL = "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf"
DEFAULT_ENV_SCRIPT = "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh"
DEFAULT_REMOTE_ROOT = "/home/intel/intel-qwen36-gpu"

COSINE_THRESHOLD = 0.9999
DECODE_TOKENS = 8
PREVIOUS_LAYERS = [2]
EXPECTED_EVENTS = len(PREVIOUS_LAYERS) * DECODE_TOKENS
EXPECTED_COUNTER_LAYERS = 72
EXPECTED_COUNTER_VALUES = 147456


def _load_module(path: Path, name: str) -> Any:
  spec = importlib.util.spec_from_file_location(name, path)
  if spec is None or spec.loader is None:
    raise SystemExit(f"cannot load module {name}: {path}")
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


SEQ476 = _load_module(SEQ476_GATE, "iq36_seq476_gate")
CURRENT_ROUTE = SEQ476.Q8_BRIDGE_ROUTE
Q8_INPUT_SENSITIVITY_ROUTE = CURRENT_ROUTE.replace(
    "_attention_output_projection_q8_bridge_gap_gate",
    "_attention_output_projection_q8_input_sensitivity_gap_gate")
GPU_KERNEL_ROUTE = CURRENT_ROUTE.replace(
    "_attention_output_projection_q8_bridge_gap_gate",
    "_attention_output_projection_gpu_kernel_gap_gate")
NATIVE_RECOMPUTE_ROUTE = CURRENT_ROUTE.replace(
    "_attention_output_projection_q8_bridge_gap_gate",
    "_attention_output_projection_native_recompute_gap_gate")
DIAG_PREFIX = SEQ476.DIAG_PREFIX
DISPOSITION_PREFIX = SEQ476.DISPOSITION_PREFIX
CASES = SEQ476.CASES


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


def _metric_summary(steps: list[Any],
                    metric_names: list[str],
                    required_metric: str) -> dict[str, Any]:
  out: dict[str, Any] = {"observation_count": 0}
  selected = set(PREVIOUS_LAYERS)
  for name in metric_names:
    out[f"min_{name}_cosine"] = 1.0
    out[f"max_{name}_abs_diff"] = 0.0
    out[f"first_{name}_gap"] = None
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
      if not isinstance(layer, int) or layer not in selected:
        continue
      if row.get(f"{required_metric}_available") is not True:
        continue
      out["observation_count"] += 1
      for name in metric_names:
        if row.get(f"{name}_available") is not True:
          continue
        cosine = _num(row.get(f"{name}_cosine"), 1.0)
        max_abs = _num(row.get(f"{name}_max_abs_diff"))
        out[f"min_{name}_cosine"] = min(
            out[f"min_{name}_cosine"], cosine)
        out[f"max_{name}_abs_diff"] = max(
            out[f"max_{name}_abs_diff"], max_abs)
        if out[f"first_{name}_gap"] is None and cosine < COSINE_THRESHOLD:
          out[f"first_{name}_gap"] = {
              "token_index": token_index,
              "layer": layer,
              f"{name}_cosine": cosine,
              f"{name}_max_abs_diff": max_abs,
          }
  return out


def _projection_source_summary(smoke: dict[str, Any]) -> dict[str, Any]:
  steps = smoke.get("linear_projection_source_diff_by_step")
  steps = steps if isinstance(steps, list) else []
  out = _metric_summary(steps, [
      "native_recompute_vs_native",
      "cpu_projection_from_gpu_input_vs_native",
      "gpu_output_vs_cpu_projection_from_gpu_input",
      "native_recompute_vs_cpu_projection_from_gpu_input",
  ], "native_recompute_vs_native")
  out.update({
      "q8_observation_count": 0,
      "max_q8_qs_mismatch_count": 0,
      "max_q8_bsums_mismatch_count": 0,
      "max_q8_d_abs_diff": 0.0,
  })
  selected = set(PREVIOUS_LAYERS)
  for step in steps:
    if not isinstance(step, dict):
      continue
    layers = step.get("layers")
    layers = layers if isinstance(layers, list) else []
    for row in layers:
      if not isinstance(row, dict) or row.get("layer") not in selected:
        continue
      if row.get("q8_bridge_available") is not True:
        continue
      out["q8_observation_count"] += 1
      out["max_q8_qs_mismatch_count"] = max(
          out["max_q8_qs_mismatch_count"],
          int(_num(row.get("q8_qs_mismatch_count"))))
      out["max_q8_bsums_mismatch_count"] = max(
          out["max_q8_bsums_mismatch_count"],
          int(_num(row.get("q8_bsums_mismatch_count"))))
      out["max_q8_d_abs_diff"] = max(
          out["max_q8_d_abs_diff"], _num(row.get("q8_d_max_abs_diff")))
  return out


def _run_case(args: argparse.Namespace, case_id: str) -> dict[str, Any]:
  case_out = args.out_dir / "cases" / case_id
  cmd = [
      sys.executable,
      str(SMOKE_SOURCE),
      "--host", args.host,
      "--model", args.model,
      "--env-script", args.env_script,
      "--remote-root", args.remote_root,
      "--token-input-dir", str(args.token_input_dir),
      "--case-id", case_id,
      "--out-dir", str(case_out),
      "--device-substring", "B390",
      "--repeat", "1",
      "--decode-tokens", str(DECODE_TOKENS),
      "--lm-head-threads", "16",
      "--shared-q4-runner",
      "--resident-q4-weights",
      "--resident-selected-q4-experts",
      "--resident-selected-q6-experts",
      "--resident-selected-q6-sorted-cache",
      "--resident-selected-q6-rowstripe",
      "--resident-selected-cache-topk", "16",
      "--resident-shared-q6-down",
      "--resident-full-attention-v-q6",
      "--resident-linear-q6-qkv",
      "--resident-q4-cpu-order-z",
      "--resident-linear-conv-weights",
      "--resident-linear-state",
      "--resident-postconv-delta-handoff",
      "--resident-norm-weights",
      "--resident-gate-up-swiglu-handoff",
      "--resident-attention-front-handoff",
      "--resident-full-core-attention-front-handoff",
      "--gpu-router",
      "--gpu-lm-head-q6",
      "--opencl-double-swiglu",
      "--distribution-ladder",
      "--full-attention-state-diff",
      "--full-core-q8-equivalence-diff",
      "--diagnostic-layer-range", "0:36",
      "--diagnostic-token-limit", str(DECODE_TOKENS),
  ]
  env = os.environ.copy()
  env.update({
      "IQ36_OPENCL_NO_QUEUE_PROFILING": "1",
      "IQ36_ATTENTION_FRONT_OUTPUT_PROJECTION_ROWBLOCK16": "1",
      "IQ36_ATTENTION_FRONT_OUTPUT_PROJECTION_ROWBLOCK16_LAYERS": (
          SEQ476.DIST_FIX.ROWBLOCK16_26MASK),
      "IQ36_SELECTED_SHARED_Q4_GATEUP_COMBINED": "1",
      "IQ36_SELECTED_SHARED_Q4_DOWN_COMBINED": "1",
      "IQ36_SELECTED_SHARED_Q6_DOWN_COMBINED": "1",
      "IQ36_DEFER_FFN_DOWN_FINISH_BUNDLE": "1",
      "IQ36_FULL_ATTENTION_LAYER_INPUT_PRODUCT_SOURCE": "1",
      "IQ36_FULL_ATTENTION_LAYER_INPUT_PRODUCT_CONSUMER_SOURCE": "1",
  })
  proc = subprocess.run(
      cmd,
      cwd=ROOT,
      env=env,
      capture_output=True,
      text=True,
      timeout=args.timeout_s,
      check=False,
  )
  result_path = case_out / "result.json"
  result = _load_json(result_path) if result_path.exists() else {}
  smoke = result.get("smoke")
  smoke = smoke if isinstance(smoke, dict) else {}
  return {
      "case_id": case_id,
      "out_dir": _rel(case_out),
      "cmd": cmd,
      "returncode": proc.returncode,
      "stdout_bytes": len(proc.stdout or ""),
      "stderr_bytes": len(proc.stderr or ""),
      "result": result,
      "smoke": smoke,
  }


def _case_summary(row: dict[str, Any]) -> dict[str, Any]:
  smoke = row.get("smoke")
  smoke = smoke if isinstance(smoke, dict) else {}
  return {
      "case_id": row.get("case_id"),
      "returncode": row.get("returncode"),
      "out_dir": row.get("out_dir"),
      "required_checks_passed": smoke.get("required_checks_passed"),
      "source_layers": smoke.get(
          "full_attention_layer_input_product_source_layers"),
      "source_values": smoke.get(
          "full_attention_layer_input_product_source_values"),
      "source_misses": smoke.get(
          "full_attention_layer_input_product_source_misses"),
      "source_ready": smoke.get(
          "full_attention_layer_input_product_source_ready"),
      "consumer_layers": smoke.get(
          "full_attention_layer_input_product_consumer_source_layers"),
      "consumer_values": smoke.get(
          "full_attention_layer_input_product_consumer_source_values"),
      "consumer_misses": smoke.get(
          "full_attention_layer_input_product_consumer_source_misses"),
      "consumer_ready": smoke.get(
          "full_attention_layer_input_product_consumer_source_ready"),
      "cpu_shadow_layer_input_layers": smoke.get("cpu_shadow_layer_input_layers"),
      "cpu_shadow_attention_output_layers": smoke.get(
          "cpu_shadow_attention_output_layers"),
      "speedup_claims_allowed": smoke.get("speedup_claims_allowed"),
      "distribution": SEQ476.UPSTREAM._distribution_summary(smoke),
      "attention_output_source": SEQ476.SEQ475._attention_output_source_summary(
          smoke),
      "projection_source": _projection_source_summary(smoke),
  }


def compute(args: argparse.Namespace) -> dict[str, Any]:
  routes = _load_json(args.routes)
  seq476 = _load_json(args.seq476)
  runs = [_run_case(args, case_id) for case_id in CASES]
  rows = [{
      "case_id": row.get("case_id"),
      "run": {
          "cmd": row.get("cmd"),
          "returncode": row.get("returncode"),
          "stdout_bytes": row.get("stdout_bytes"),
          "stderr_bytes": row.get("stderr_bytes"),
          "out_dir": row.get("out_dir"),
      },
      "summary": _case_summary(row),
  } for row in runs]

  def row_ready(row: dict[str, Any]) -> bool:
    summary = row.get("summary", {})
    return (
        bool(row.get("summary"))
        and summary.get("source_layers") == EXPECTED_COUNTER_LAYERS
        and summary.get("source_values") == EXPECTED_COUNTER_VALUES
        and summary.get("source_misses") == 0
        and summary.get("source_ready") is True
        and summary.get("consumer_layers") == EXPECTED_COUNTER_LAYERS
        and summary.get("consumer_values") == EXPECTED_COUNTER_VALUES
        and summary.get("consumer_misses") == 0
        and summary.get("consumer_ready") is True
        and summary.get("cpu_shadow_layer_input_layers") == 0
        and summary.get("cpu_shadow_attention_output_layers") == 0
        and summary.get("speedup_claims_allowed") is False
    )

  preconditions_pass = (
      seq476.get("required_checks_passed") is True
      and seq476.get("selected_next_route") == CURRENT_ROUTE
      and seq476.get("diagnostic_classification")
      == f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_input_attention_output_projection_q8_bridge_gap"
      and _has_candidate(routes, 476, str(seq476.get("disposition")))
      and _has_switch(routes, f"select_{CURRENT_ROUTE}", 476)
  )
  rows_emitted = (
      len(rows) == len(CASES)
      and all(row.get("summary", {}).get("projection_source", {})
              .get("observation_count", 0) > 0 for row in rows)
  )
  counters_ready = rows_emitted and all(row_ready(row) for row in rows)
  distribution_reproduced = rows_emitted and all(
      SEQ476.SEQ475.SEQ474.SEQ473.SEQ472._dist_fail(
          row.get("summary", {}).get("distribution", {}))
      for row in rows)
  diagnostics_emitted = counters_ready and all(
      row.get("summary", {}).get("attention_output_source", {})
      .get("attention_front", {}).get("observation_count") == EXPECTED_EVENTS
      and row.get("summary", {}).get("projection_source", {})
      .get("observation_count") == EXPECTED_EVENTS
      and row.get("summary", {}).get("projection_source", {})
      .get("q8_observation_count") == EXPECTED_EVENTS
      for row in rows)

  def min_attention(group: str, name: str) -> float:
    return min(
        (_num(row.get("summary", {}).get("attention_output_source", {})
              .get(group, {}).get(f"min_{name}_cosine"), 1.0)
         for row in rows),
        default=1.0)

  def min_projection(name: str) -> float:
    return min(
        (_num(row.get("summary", {}).get("projection_source", {})
              .get(f"min_{name}_cosine"), 1.0)
         for row in rows),
        default=1.0)

  max_q8_qs_mismatch = max(
      (int(_num(row.get("summary", {}).get("projection_source", {})
                .get("max_q8_qs_mismatch_count")))
       for row in rows),
      default=0)
  max_q8_bsums_mismatch = max(
      (int(_num(row.get("summary", {}).get("projection_source", {})
                .get("max_q8_bsums_mismatch_count")))
       for row in rows),
      default=0)
  max_q8_d_abs_diff = max(
      (_num(row.get("summary", {}).get("projection_source", {})
            .get("max_q8_d_abs_diff"))
       for row in rows),
      default=0.0)

  min_front_projection_input = min_attention(
      "attention_front", "projection_input")
  min_front_attention_output = min_attention(
      "attention_front", "attention_output")
  min_native_recompute = min_projection("native_recompute_vs_native")
  min_cpu_from_gpu_input = min_projection(
      "cpu_projection_from_gpu_input_vs_native")
  min_gpu_vs_cpu_projection = min_projection(
      "gpu_output_vs_cpu_projection_from_gpu_input")
  min_native_vs_cpu_gpu_projection = min_projection(
      "native_recompute_vs_cpu_projection_from_gpu_input")

  projection_input_clean = (
      diagnostics_emitted
      and min_front_projection_input >= COSINE_THRESHOLD)
  attention_output_gap = (
      diagnostics_emitted
      and min_front_attention_output < COSINE_THRESHOLD)
  native_recompute_ok = (
      diagnostics_emitted and min_native_recompute >= COSINE_THRESHOLD)
  cpu_from_gpu_input_gap = (
      diagnostics_emitted and min_cpu_from_gpu_input < COSINE_THRESHOLD)
  gpu_output_matches_cpu_projection = (
      diagnostics_emitted and min_gpu_vs_cpu_projection >= COSINE_THRESHOLD)
  q8_mismatch_observed = (
      diagnostics_emitted and (
          max_q8_qs_mismatch > 0
          or max_q8_bsums_mismatch > 0
          or max_q8_d_abs_diff > 0.0))
  q8_input_sensitivity_gap = (
      attention_output_gap
      and projection_input_clean
      and native_recompute_ok
      and cpu_from_gpu_input_gap
      and gpu_output_matches_cpu_projection
      and q8_mismatch_observed)
  gpu_kernel_gap = (
      attention_output_gap
      and projection_input_clean
      and native_recompute_ok
      and min_cpu_from_gpu_input >= COSINE_THRESHOLD
      and not gpu_output_matches_cpu_projection)
  native_recompute_gap = diagnostics_emitted and not native_recompute_ok

  diagnostic_classification = (
      f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_input_attention_output_projection_q8_input_sensitivity_gap"
      if q8_input_sensitivity_gap else
      f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_input_attention_output_projection_gpu_kernel_gap"
      if gpu_kernel_gap else
      f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_input_attention_output_projection_native_recompute_gap"
      if native_recompute_gap else
      f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_input_attention_output_projection_q8_bridge_gap_unclassified"
  )
  selected_next = (
      Q8_INPUT_SENSITIVITY_ROUTE
      if q8_input_sensitivity_gap else
      GPU_KERNEL_ROUTE
      if gpu_kernel_gap else
      NATIVE_RECOMPUTE_ROUTE
      if native_recompute_gap else
      CURRENT_ROUTE
  )
  checks = [
      {"name": "seq476_selected_q8_bridge_gap_gate",
       "pass": preconditions_pass},
      {"name": "q8_bridge_rows_emitted", "pass": rows_emitted},
      {"name": "source_and_consumer_counters_ready",
       "pass": counters_ready,
       "detail": [row.get("summary", {}) for row in rows]},
      {"name": "distribution_failure_reproduced",
       "pass": distribution_reproduced,
       "detail": [
           row.get("summary", {}).get("distribution", {}) for row in rows
       ]},
      {"name": "projection_source_diagnostics_emitted",
       "pass": diagnostics_emitted,
       "detail": [
           row.get("summary", {}).get("projection_source", {})
           for row in rows
       ]},
      {"name": "attention_projection_input_clean",
       "pass": projection_input_clean,
       "detail": {
           "min_front_projection_input_cosine": min_front_projection_input,
       }},
      {"name": "attention_output_gap_reproduced",
       "pass": attention_output_gap,
       "detail": {
           "min_front_attention_output_cosine": min_front_attention_output,
       }},
      {"name": "projection_source_classified",
       "pass": selected_next != CURRENT_ROUTE,
       "detail": {
           "q8_input_sensitivity_gap": q8_input_sensitivity_gap,
           "gpu_kernel_gap": gpu_kernel_gap,
           "native_recompute_gap": native_recompute_gap,
           "min_native_recompute_vs_native_cosine": min_native_recompute,
           "min_cpu_projection_from_gpu_input_vs_native_cosine": (
               min_cpu_from_gpu_input),
           "min_gpu_output_vs_cpu_projection_from_gpu_input_cosine": (
               min_gpu_vs_cpu_projection),
           "max_q8_qs_mismatch_count": max_q8_qs_mismatch,
           "max_q8_bsums_mismatch_count": max_q8_bsums_mismatch,
           "max_q8_d_abs_diff": max_q8_d_abs_diff,
       }},
  ]
  required = all(bool(row.get("pass")) for row in checks)
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "inputs": {
          "routes": _rel(args.routes),
          "seq476": _rel(args.seq476),
          "token_input_dir": _rel(args.token_input_dir),
          "host": args.host,
          "previous_layers": PREVIOUS_LAYERS,
      },
      "runs": rows,
      "checks": checks,
      "required_checks_passed": required,
      "diagnostic_classification": diagnostic_classification,
      "projection_input_clean": projection_input_clean,
      "attention_output_gap": attention_output_gap,
      "native_recompute_ok": native_recompute_ok,
      "cpu_from_gpu_input_gap": cpu_from_gpu_input_gap,
      "gpu_output_matches_cpu_projection": gpu_output_matches_cpu_projection,
      "q8_mismatch_observed": q8_mismatch_observed,
      "q8_input_sensitivity_gap": q8_input_sensitivity_gap,
      "gpu_kernel_gap": gpu_kernel_gap,
      "native_recompute_gap": native_recompute_gap,
      "min_front_projection_input_cosine": min_front_projection_input,
      "min_front_attention_output_cosine": min_front_attention_output,
      "min_native_recompute_vs_native_cosine": min_native_recompute,
      "min_cpu_projection_from_gpu_input_vs_native_cosine": (
          min_cpu_from_gpu_input),
      "min_gpu_output_vs_cpu_projection_from_gpu_input_cosine": (
          min_gpu_vs_cpu_projection),
      "min_native_recompute_vs_cpu_projection_from_gpu_input_cosine": (
          min_native_vs_cpu_gpu_projection),
      "max_q8_qs_mismatch_count": max_q8_qs_mismatch,
      "max_q8_bsums_mismatch_count": max_q8_bsums_mismatch,
      "max_q8_d_abs_diff": max_q8_d_abs_diff,
      "speedup_claims_allowed": False,
      "disposition": (
          f"accept_{DISPOSITION_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_input_attention_output_projection_q8_bridge_gap_classification"
          if required else
          f"block_{DISPOSITION_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_input_attention_output_projection_q8_bridge_gap_classification"
      ),
      "selected_next_route": selected_next if required else CURRENT_ROUTE,
      "next_route_reason": (
          "CPU projection from the live GPU projection input reproduces the "
          "attention-output drift, native recompute matches native, and GPU "
          "output matches that CPU recompute. The Q8 activation bridge is "
          "sensitive to the remaining projection-input value drift; root that "
          "input sensitivity next."
          if required and selected_next == Q8_INPUT_SENSITIVITY_ROUTE else
          "CPU projection from the live GPU input matches native while GPU "
          "projection diverges. Root the GPU projection kernel/packing path next."
          if required and selected_next == GPU_KERNEL_ROUTE else
          "Native projection recompute no longer reproduces native output. Root "
          "native recompute/trace consistency before continuing."
          if required and selected_next == NATIVE_RECOMPUTE_ROUTE else
          "Projection-source evidence is incomplete; keep this gate open."
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
  lines = [
      "# Seq477 Attention Output-Projection Q8 Bridge Gap Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- diagnostic_classification: `{metrics['diagnostic_classification']}`",
      f"- min attention front output/projection-input cosines: `{metrics['min_front_attention_output_cosine']}` / `{metrics['min_front_projection_input_cosine']}`",
      f"- min native recompute / CPU-from-GPU-input / GPU-vs-CPU projection cosines: `{metrics['min_native_recompute_vs_native_cosine']}` / `{metrics['min_cpu_projection_from_gpu_input_vs_native_cosine']}` / `{metrics['min_gpu_output_vs_cpu_projection_from_gpu_input_cosine']}`",
      f"- q8 mismatches qs/bsums/d: `{metrics['max_q8_qs_mismatch_count']}` / `{metrics['max_q8_bsums_mismatch_count']}` / `{metrics['max_q8_d_abs_diff']}`",
      f"- q8_input_sensitivity_gap: `{str(metrics['q8_input_sensitivity_gap']).lower()}`",
      f"- gpu_kernel_gap: `{str(metrics['gpu_kernel_gap']).lower()}`",
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
  parser.add_argument("--seq476", type=Path, default=DEFAULT_SEQ476)
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
