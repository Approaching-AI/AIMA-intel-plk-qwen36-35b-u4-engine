#!/usr/bin/env python3
"""Classify seq474 previous attention-output drift feeding FFN norm input."""

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
SEQ474_GATE = (
    ROOT
    / "tools/intel-qwen36-seq474-upstream-layer-output-ffn-delta-ffn-norm-gap-gate.py"
)
ATTENTION_GATE = (
    ROOT
    / "tools/intel-qwen36-router-full-attention-layer-input-preceding-linear-input-source-producer-linear-input-source-attention-output-gap-gate.py"
)
SCHEMA_VERSION = (
    "intel-qwen36-seq475-upstream-layer-output-ffn-delta-ffn-norm-input-attention-output-gap-gate-v0"
)

DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_SEQ474 = (
    ROOT
    / "output/seq474-upstream-layer-output-ffn-delta-ffn-norm-gap-gate-20260709Tseq474Z"
    / "metrics.json"
)
DEFAULT_TOKEN_INPUT_DIR = (
    ROOT
    / "output/r2-gpu-acceptance-matrix-router-both-rowblock16-26mask-20260708Tseq222Z"
    / "token-input"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "output/seq475-upstream-layer-output-ffn-delta-ffn-norm-input-attention-output-gap-gate-20260709Tseq475Z"
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


SEQ474 = _load_module(SEQ474_GATE, "iq36_seq474_gate")
ATTENTION = _load_module(ATTENTION_GATE, "iq36_seq475_attention_gate")
CURRENT_ROUTE = SEQ474.INPUT_ATTENTION_ROUTE
DELTA_Z_ROUTE = CURRENT_ROUTE.replace(
    "_ffn_norm_input_attention_output_gap_gate",
    "_ffn_norm_input_linear_delta_z_gap_gate")
DELTA_ROUTE = CURRENT_ROUTE.replace(
    "_ffn_norm_input_attention_output_gap_gate",
    "_ffn_norm_input_linear_delta_output_gap_gate")
Z_ROUTE = CURRENT_ROUTE.replace(
    "_ffn_norm_input_attention_output_gap_gate",
    "_ffn_norm_input_linear_z_gap_gate")
MATH_ROUTE = CURRENT_ROUTE.replace(
    "_ffn_norm_input_attention_output_gap_gate",
    "_ffn_norm_input_preconv_math_gap_gate")
Q_ROPE_ROUTE = CURRENT_ROUTE.replace(
    "_ffn_norm_input_attention_output_gap_gate",
    "_ffn_norm_input_full_attention_core_q_rope_gap_gate")
K_HISTORY_ROUTE = CURRENT_ROUTE.replace(
    "_ffn_norm_input_attention_output_gap_gate",
    "_ffn_norm_input_full_attention_core_k_history_gap_gate")
V_HISTORY_ROUTE = CURRENT_ROUTE.replace(
    "_ffn_norm_input_attention_output_gap_gate",
    "_ffn_norm_input_full_attention_core_v_history_gap_gate")
CORE_MATH_ROUTE = CURRENT_ROUTE.replace(
    "_ffn_norm_input_attention_output_gap_gate",
    "_ffn_norm_input_full_attention_core_math_gap_gate")
GATE_MATH_ROUTE = CURRENT_ROUTE.replace(
    "_ffn_norm_input_attention_output_gap_gate",
    "_ffn_norm_input_full_attention_gate_math_gap_gate")
OUTPUT_MATH_ROUTE = CURRENT_ROUTE.replace(
    "_ffn_norm_input_attention_output_gap_gate",
    "_ffn_norm_input_attention_output_projection_math_gap_gate")
PROJECTION_INPUT_ROUTE = CURRENT_ROUTE.replace(
    "_ffn_norm_input_attention_output_gap_gate",
    "_ffn_norm_input_attention_projection_input_gap_gate")
DIAG_PREFIX = SEQ474.DIAG_PREFIX
DISPOSITION_PREFIX = SEQ474.DISPOSITION_PREFIX


def _patch_attention_gate() -> None:
  ATTENTION.SOURCE_LAYERS = PREVIOUS_LAYERS
  ATTENTION.EXPECTED_EVENTS = EXPECTED_EVENTS
  ATTENTION.EXPECTED_COUNTER_LAYERS = EXPECTED_COUNTER_LAYERS
  ATTENTION.EXPECTED_COUNTER_VALUES = EXPECTED_COUNTER_VALUES


_patch_attention_gate()


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


def _attention_output_source_summary(smoke: dict[str, Any]) -> dict[str, Any]:
  summary = ATTENTION._source_attention_summary(smoke)
  front_steps = smoke.get("attention_front_diff_by_step")
  front_steps = front_steps if isinstance(front_steps, list) else []
  full_steps = smoke.get("full_attention_source_diff_by_step")
  full_steps = full_steps if isinstance(full_steps, list) else []
  core_steps = smoke.get("full_attention_core_pregate_source_diff_by_step")
  core_steps = core_steps if isinstance(core_steps, list) else []
  summary["attention_front"] = _metric_summary(front_steps, [
      "projection_input",
      "attention_output",
  ], "attention_output")
  summary["full_attention_source"] = _metric_summary(full_steps, [
      "attn_norm_from_gpu_input",
      "gpu_attn_norm_vs_cpu",
      "q_full_from_gpu_norm",
      "gpu_q_full_vs_cpu",
      "k_raw_from_gpu_norm",
      "gpu_k_raw_vs_cpu",
      "v_from_gpu_norm",
      "gpu_v_vs_cpu",
      "q_normed_from_gpu_q",
      "gpu_q_normed_vs_cpu",
      "k_normed_from_gpu_k",
      "gpu_k_normed_vs_cpu",
      "q_rope_from_gpu_normed",
      "gpu_q_rope_vs_cpu",
      "k_rope_from_gpu_normed",
      "gpu_k_rope_vs_cpu",
      "attn_pregate",
      "gated_from_gpu_core",
      "gpu_gated_vs_cpu",
      "output_from_gpu_gated",
      "gpu_output_vs_cpu",
  ], "output_from_gpu_gated")
  summary["full_attention_core"] = _metric_summary(core_steps, [
      "k_history_flat",
      "v_history_flat",
      "native_inputs_cpu",
      "gpu_inputs_cpu",
      "gpu_core_vs_cpu_gpu_inputs",
      "gpu_q_native_history_cpu",
      "native_q_gpu_history_cpu",
      "native_q_gpu_k_native_v_cpu",
      "native_q_native_k_gpu_v_cpu",
  ], "gpu_inputs_cpu")
  return summary


def _case_summary(case_id: str, run: dict[str, Any],
                  smoke: dict[str, Any]) -> dict[str, Any]:
  return {
      "case_id": case_id,
      "returncode": run.get("returncode"),
      "stdout_bytes": len(str(run.get("stdout") or "")),
      "stderr_bytes": len(str(run.get("stderr") or "")),
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
      "distribution": (
          SEQ474.SEQ473.SEQ472.SEQ471.BASE.OLD.BASE
          ._distribution_summary(smoke)),
      "attention_output_source": _attention_output_source_summary(smoke),
  }


def compute(args: argparse.Namespace) -> dict[str, Any]:
  routes = _load_json(args.routes)
  seq474 = _load_json(args.seq474)
  binary = seq474.get("inputs", {}).get("binary")
  if not isinstance(binary, str) or not binary:
    raise SystemExit("seq474 binary missing")
  token_cache = (
      SEQ474.SEQ473.SEQ472.SEQ471.BASE.OLD.BASE.BASE.BASE
      .iq36_local.ensure_cached_tokens(
          args.host, f"{args.remote_root}/cache", args.token_input_dir,
          args.timeout_s))

  runs: list[dict[str, Any]] = []
  if token_cache.get("ok") is True:
    for case_id in SEQ474.SEQ473.SEQ472.SEQ471.BASE.CASES:
      run = SEQ474.SEQ473.SEQ472.SEQ471.BASE.OLD.BASE._run_case(
          args, binary, str(token_cache.get("dir")), case_id)
      smoke = SEQ474.SEQ473.SEQ472.SEQ471.BASE.OLD.BASE._smoke_from_stdout(run)
      runs.append({
          "case_id": case_id,
          "run": {
              "cmd": run.get("cmd"),
              "returncode": run.get("returncode"),
              "stdout_bytes": len(str(run.get("stdout") or "")),
              "stderr_bytes": len(str(run.get("stderr") or "")),
          },
          "summary": _case_summary(case_id, run, smoke),
      })

  def row_ready(row: dict[str, Any]) -> bool:
    summary = row.get("summary", {})
    return (
        row.get("run", {}).get("returncode") in (0, 2)
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
      seq474.get("required_checks_passed") is True
      and seq474.get("selected_next_route") == CURRENT_ROUTE
      and seq474.get("diagnostic_classification")
      == f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_input_attention_output_gap"
      and _has_candidate(routes, 474, str(seq474.get("disposition")))
      and _has_switch(routes, f"select_{CURRENT_ROUTE}", 474)
  )
  rows_emitted = (
      len(runs) == len(SEQ474.SEQ473.SEQ472.SEQ471.BASE.CASES)
      and all(row.get("run", {}).get("returncode") in (0, 2)
              for row in runs)
  )
  counters_ready = rows_emitted and all(row_ready(row) for row in runs)
  distribution_reproduced = rows_emitted and all(
      SEQ474.SEQ473.SEQ472._dist_fail(
          row.get("summary", {}).get("distribution", {}))
      for row in runs)
  diagnostics_emitted = counters_ready and all(
      row.get("summary", {}).get("attention_output_source", {})
      .get("residual_source", {}).get("observation_count") == EXPECTED_EVENTS
      and row.get("summary", {}).get("attention_output_source", {})
      .get("linear_attention", {}).get("observation_count") == EXPECTED_EVENTS
      and row.get("summary", {}).get("attention_output_source", {})
      .get("preconv_source", {}).get("observation_count") == EXPECTED_EVENTS
      and row.get("summary", {}).get("attention_output_source", {})
      .get("final_mix", {}).get("observation_count") == EXPECTED_EVENTS
      and row.get("summary", {}).get("attention_output_source", {})
      .get("attention_front", {}).get("observation_count") == EXPECTED_EVENTS
      for row in runs)

  def min_metric(group: str, name: str) -> float:
    return min(
        (_num(row.get("summary", {}).get("attention_output_source", {})
              .get(group, {}).get(f"min_{name}_cosine"), 1.0)
         for row in runs),
        default=1.0)

  min_layer_input = min_metric("residual_source", "layer_input")
  min_attention_output = min_metric("residual_source", "attention_output")
  min_ffn_input = min_metric("residual_source", "ffn_input")
  min_gpu_output_vs_cpu_ffn = min_metric(
      "residual_source", "gpu_output_vs_cpu_ffn")
  min_gpu_ffn_delta_vs_cpu = min_metric(
      "residual_source", "gpu_ffn_delta_vs_cpu")
  min_gpu_ffn_norm_vs_cpu = min_metric(
      "residual_source", "gpu_ffn_norm_vs_cpu")
  min_attn_norm = min_metric("linear_attention", "attn_norm")
  min_linear_final = min_metric("linear_attention", "final_output")
  min_delta_output = min_metric("linear_attention", "delta_output")
  min_z = min_metric("linear_attention", "z")
  min_attn_norm_from_input = min_metric(
      "preconv_source", "attn_norm_from_gpu_input")
  min_gpu_attn_norm_vs_cpu = min_metric(
      "preconv_source", "gpu_attn_norm_vs_cpu")
  min_qkv_from_gpu_norm = min_metric(
      "preconv_source", "qkv_from_gpu_attn_norm")
  min_gpu_qkv_vs_cpu = min_metric("preconv_source", "gpu_qkv_vs_cpu")
  min_z_from_gpu_norm = min_metric("preconv_source", "z_from_gpu_attn_norm")
  min_gpu_z_vs_cpu = min_metric("preconv_source", "gpu_z_vs_cpu")
  min_final = min_metric("final_mix", "gpu_kernel_final")
  min_delta_native_z = min_metric("final_mix", "gpu_delta_native_z_cpu")
  min_native_delta_gpu_z = min_metric("final_mix", "native_delta_gpu_z_cpu")
  min_native_recompute = min_metric("final_mix", "native_delta_native_z_cpu")
  min_front_projection_input = min_metric("attention_front", "projection_input")
  min_front_attention_output = min_metric("attention_front", "attention_output")
  min_full_output_from_gated = min_metric(
      "full_attention_source", "output_from_gpu_gated")
  min_full_gpu_output_vs_cpu = min_metric(
      "full_attention_source", "gpu_output_vs_cpu")
  min_full_gated_from_core = min_metric(
      "full_attention_source", "gated_from_gpu_core")
  min_full_gpu_gated_vs_cpu = min_metric(
      "full_attention_source", "gpu_gated_vs_cpu")
  min_full_attn_pregate = min_metric("full_attention_source", "attn_pregate")
  min_full_q_rope_from_gpu_normed = min_metric(
      "full_attention_source", "q_rope_from_gpu_normed")
  min_full_gpu_q_rope_vs_cpu = min_metric(
      "full_attention_source", "gpu_q_rope_vs_cpu")
  min_full_k_rope_from_gpu_normed = min_metric(
      "full_attention_source", "k_rope_from_gpu_normed")
  min_full_gpu_k_rope_vs_cpu = min_metric(
      "full_attention_source", "gpu_k_rope_vs_cpu")
  min_core_k_history = min_metric("full_attention_core", "k_history_flat")
  min_core_v_history = min_metric("full_attention_core", "v_history_flat")
  min_core_native_inputs_cpu = min_metric(
      "full_attention_core", "native_inputs_cpu")
  min_core_gpu_inputs_cpu = min_metric(
      "full_attention_core", "gpu_inputs_cpu")
  min_core_gpu_core_vs_cpu = min_metric(
      "full_attention_core", "gpu_core_vs_cpu_gpu_inputs")
  min_core_gpu_q_native_history = min_metric(
      "full_attention_core", "gpu_q_native_history_cpu")
  min_core_native_q_gpu_history = min_metric(
      "full_attention_core", "native_q_gpu_history_cpu")
  min_core_native_q_gpu_k_native_v = min_metric(
      "full_attention_core", "native_q_gpu_k_native_v_cpu")
  min_core_native_q_native_k_gpu_v = min_metric(
      "full_attention_core", "native_q_native_k_gpu_v_cpu")

  layer_input_clean = diagnostics_emitted and min_layer_input >= COSINE_THRESHOLD
  attention_output_gap = (
      diagnostics_emitted and min_attention_output < COSINE_THRESHOLD)
  ffn_math_ok = (
      diagnostics_emitted
      and min_gpu_output_vs_cpu_ffn >= COSINE_THRESHOLD
      and min_gpu_ffn_delta_vs_cpu >= COSINE_THRESHOLD
      and min_gpu_ffn_norm_vs_cpu >= COSINE_THRESHOLD)
  linear_final_gap = (
      diagnostics_emitted
      and min_linear_final < COSINE_THRESHOLD
      and min_final < COSINE_THRESHOLD)
  linear_final_recompute_ok = (
      diagnostics_emitted and min_native_recompute >= COSINE_THRESHOLD)
  delta_drives_gap = (
      diagnostics_emitted and min_delta_native_z < COSINE_THRESHOLD)
  z_drives_gap = (
      diagnostics_emitted and min_native_delta_gpu_z < COSINE_THRESHOLD)
  preconv_math_ok = (
      diagnostics_emitted
      and min_gpu_attn_norm_vs_cpu >= COSINE_THRESHOLD
      and min_gpu_qkv_vs_cpu >= COSINE_THRESHOLD
      and min_gpu_z_vs_cpu >= COSINE_THRESHOLD)
  qkv_z_inherit_input = (
      diagnostics_emitted
      and min_attn_norm < COSINE_THRESHOLD
      and min_attn_norm_from_input < COSINE_THRESHOLD
      and min_qkv_from_gpu_norm < COSINE_THRESHOLD
      and min_z_from_gpu_norm < COSINE_THRESHOLD
      and min_delta_output < COSINE_THRESHOLD
      and min_z < COSINE_THRESHOLD)
  final_mix_delta_z_inherits_live_input = (
      diagnostics_emitted
      and delta_drives_gap
      and z_drives_gap
      and linear_final_recompute_ok
      and min_attn_norm < COSINE_THRESHOLD
      and min_attn_norm_from_input < COSINE_THRESHOLD
      and min_qkv_from_gpu_norm < COSINE_THRESHOLD
      and min_z_from_gpu_norm < COSINE_THRESHOLD
      and min_z < COSINE_THRESHOLD)
  attention_front_output_gap = (
      diagnostics_emitted and min_front_attention_output < COSINE_THRESHOLD)
  attention_projection_input_gap = (
      diagnostics_emitted and min_front_projection_input < COSINE_THRESHOLD)
  attention_projection_math_gap = (
      attention_front_output_gap and not attention_projection_input_gap)
  full_output_projection_math_ok = (
      diagnostics_emitted and min_full_gpu_output_vs_cpu >= COSINE_THRESHOLD)
  full_gate_math_ok = (
      diagnostics_emitted and min_full_gpu_gated_vs_cpu >= COSINE_THRESHOLD)
  full_core_math_ok = (
      diagnostics_emitted and min_core_gpu_core_vs_cpu >= COSINE_THRESHOLD)
  full_output_from_gated_gap = (
      diagnostics_emitted and min_full_output_from_gated < COSINE_THRESHOLD)
  full_gated_from_core_gap = (
      diagnostics_emitted and min_full_gated_from_core < COSINE_THRESHOLD)
  full_pregate_gap = (
      diagnostics_emitted and min_full_attn_pregate < COSINE_THRESHOLD)
  full_core_native_recompute_ok = (
      diagnostics_emitted and min_core_native_inputs_cpu >= COSINE_THRESHOLD)
  full_core_gpu_inputs_gap = (
      diagnostics_emitted and min_core_gpu_inputs_cpu < COSINE_THRESHOLD)
  full_core_q_path_gap = (
      diagnostics_emitted and min_core_gpu_q_native_history < COSINE_THRESHOLD)
  full_core_history_path_gap = (
      diagnostics_emitted and min_core_native_q_gpu_history < COSINE_THRESHOLD)
  full_core_k_history_gap = (
      diagnostics_emitted
      and min_core_k_history < COSINE_THRESHOLD
      and min_core_native_q_gpu_k_native_v < COSINE_THRESHOLD)
  full_core_v_history_gap = (
      diagnostics_emitted
      and min_core_v_history < COSINE_THRESHOLD
      and min_core_native_q_native_k_gpu_v < COSINE_THRESHOLD)
  full_q_rope_gap = (
      diagnostics_emitted
      and min_full_q_rope_from_gpu_normed < COSINE_THRESHOLD
      and min_full_gpu_q_rope_vs_cpu >= COSINE_THRESHOLD)
  full_attention_core_source_gap = (
      attention_output_gap and layer_input_clean and ffn_math_ok
      and full_output_projection_math_ok and full_gate_math_ok
      and full_output_from_gated_gap and full_gated_from_core_gap
      and full_pregate_gap and full_core_native_recompute_ok
      and full_core_gpu_inputs_gap and full_core_math_ok)

  diagnostic_classification = (
      f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_input_attention_projection_input_gap"
      if (attention_output_gap and layer_input_clean and ffn_math_ok
          and attention_front_output_gap and attention_projection_input_gap) else
      f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_input_attention_output_projection_math_gap"
      if (attention_output_gap and layer_input_clean and ffn_math_ok
          and attention_projection_math_gap) else
      f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_input_full_attention_core_k_history_gap"
      if full_attention_core_source_gap and full_core_k_history_gap else
      f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_input_full_attention_core_v_history_gap"
      if full_attention_core_source_gap and full_core_v_history_gap else
      f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_input_full_attention_core_q_rope_gap"
      if full_attention_core_source_gap and (
          full_core_q_path_gap or full_q_rope_gap) else
      f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_input_full_attention_core_math_gap"
      if (attention_output_gap and layer_input_clean and ffn_math_ok
          and not full_core_math_ok) else
      f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_input_full_attention_gate_math_gap"
      if (attention_output_gap and layer_input_clean and ffn_math_ok
          and not full_gate_math_ok) else
      f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_input_attention_output_projection_math_gap"
      if (attention_output_gap and layer_input_clean and ffn_math_ok
          and not full_output_projection_math_ok) else
      f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_input_linear_delta_output_gap"
      if (attention_output_gap and ffn_math_ok and linear_final_gap
          and delta_drives_gap and preconv_math_ok) else
      f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_input_linear_z_gap"
      if (attention_output_gap and ffn_math_ok and linear_final_gap
          and z_drives_gap and preconv_math_ok) else
      f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_input_preconv_math_gap"
      if attention_output_gap and not preconv_math_ok else
      f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_input_attention_output_gap_unclassified"
  )
  selected_next = (
      PROJECTION_INPUT_ROUTE
      if diagnostic_classification
      == f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_input_attention_projection_input_gap"
      else OUTPUT_MATH_ROUTE
      if diagnostic_classification
      == f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_input_attention_output_projection_math_gap"
      else K_HISTORY_ROUTE
      if diagnostic_classification
      == f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_input_full_attention_core_k_history_gap"
      else V_HISTORY_ROUTE
      if diagnostic_classification
      == f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_input_full_attention_core_v_history_gap"
      else Q_ROPE_ROUTE
      if diagnostic_classification
      == f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_input_full_attention_core_q_rope_gap"
      else CORE_MATH_ROUTE
      if diagnostic_classification
      == f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_input_full_attention_core_math_gap"
      else GATE_MATH_ROUTE
      if diagnostic_classification
      == f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_input_full_attention_gate_math_gap"
      else DELTA_ROUTE
      if diagnostic_classification
      == f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_input_linear_delta_output_gap"
      else Z_ROUTE
      if diagnostic_classification
      == f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_input_linear_z_gap"
      else MATH_ROUTE
      if diagnostic_classification
      == f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_input_preconv_math_gap"
      else CURRENT_ROUTE
  )
  checks = [
      {"name": "seq474_selected_attention_output_gap_gate",
       "pass": preconditions_pass},
      {"name": "attention_output_rows_emitted", "pass": rows_emitted},
      {"name": "source_and_consumer_counters_ready",
       "pass": counters_ready,
       "detail": [row.get("summary", {}) for row in runs]},
      {"name": "distribution_failure_reproduced",
       "pass": distribution_reproduced,
       "detail": [
           row.get("summary", {}).get("distribution", {}) for row in runs
       ]},
      {"name": "attention_output_diagnostics_emitted",
       "pass": diagnostics_emitted,
       "detail": [
           row.get("summary", {}).get("attention_output_source", {})
           for row in runs
       ]},
      {"name": "previous_layer_input_clean_for_attention_output_split",
       "pass": layer_input_clean,
       "detail": {"min_previous_layer_input_cosine": min_layer_input}},
      {"name": "previous_attention_output_gap_reproduced",
       "pass": attention_output_gap,
       "detail": {
           "min_previous_attention_output_cosine": min_attention_output,
           "min_previous_ffn_input_cosine": min_ffn_input,
       }},
      {"name": "previous_ffn_math_matches_cpu_on_live_input",
       "pass": ffn_math_ok,
       "detail": {
           "min_gpu_output_vs_cpu_ffn_cosine": min_gpu_output_vs_cpu_ffn,
           "min_gpu_ffn_delta_vs_cpu_cosine": min_gpu_ffn_delta_vs_cpu,
           "min_gpu_ffn_norm_vs_cpu_cosine": min_gpu_ffn_norm_vs_cpu,
       }},
      {"name": "attention_front_output_gap_reproduced",
       "pass": attention_front_output_gap,
       "detail": {
           "min_front_attention_output_cosine": min_front_attention_output,
           "min_front_projection_input_cosine": min_front_projection_input,
       }},
      {"name": "attention_projection_input_or_output_math_selected",
       "pass": attention_projection_input_gap or attention_projection_math_gap,
       "detail": {
           "attention_projection_input_gap": attention_projection_input_gap,
           "attention_projection_math_gap": attention_projection_math_gap,
           "min_front_attention_output_cosine": min_front_attention_output,
           "min_front_projection_input_cosine": min_front_projection_input,
       }},
  ]
  required = all(bool(row.get("pass")) for row in checks)
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "inputs": {
          "routes": _rel(args.routes),
          "seq474": _rel(args.seq474),
          "token_input_dir": _rel(args.token_input_dir),
          "host": args.host,
          "binary": binary,
          "previous_layers": PREVIOUS_LAYERS,
      },
      "token_cache": {
          "ok": token_cache.get("ok"),
          "hit": token_cache.get("hit"),
          "key": token_cache.get("key"),
          "dir": token_cache.get("dir"),
      },
      "runs": runs,
      "checks": checks,
      "required_checks_passed": required,
      "diagnostic_classification": diagnostic_classification,
      "attention_output_gap": attention_output_gap,
      "previous_layer_input_clean": layer_input_clean,
      "ffn_math_ok": ffn_math_ok,
      "linear_final_gap": linear_final_gap,
      "linear_final_recompute_ok": linear_final_recompute_ok,
      "delta_drives_gap": delta_drives_gap,
      "z_drives_gap": z_drives_gap,
      "preconv_math_ok": preconv_math_ok,
      "qkv_z_inherit_input": qkv_z_inherit_input,
      "final_mix_delta_z_inherits_live_input": (
          final_mix_delta_z_inherits_live_input),
      "attention_front_output_gap": attention_front_output_gap,
      "attention_projection_input_gap": attention_projection_input_gap,
      "attention_projection_math_gap": attention_projection_math_gap,
      "full_output_projection_math_ok": full_output_projection_math_ok,
      "full_gate_math_ok": full_gate_math_ok,
      "full_core_math_ok": full_core_math_ok,
      "full_output_from_gated_gap": full_output_from_gated_gap,
      "full_gated_from_core_gap": full_gated_from_core_gap,
      "full_pregate_gap": full_pregate_gap,
      "full_core_native_recompute_ok": full_core_native_recompute_ok,
      "full_core_gpu_inputs_gap": full_core_gpu_inputs_gap,
      "full_core_q_path_gap": full_core_q_path_gap,
      "full_core_history_path_gap": full_core_history_path_gap,
      "full_core_k_history_gap": full_core_k_history_gap,
      "full_core_v_history_gap": full_core_v_history_gap,
      "full_q_rope_gap": full_q_rope_gap,
      "full_attention_core_source_gap": full_attention_core_source_gap,
      "min_previous_layer_input_cosine": min_layer_input,
      "min_previous_attention_output_cosine": min_attention_output,
      "min_previous_ffn_input_cosine": min_ffn_input,
      "min_gpu_output_vs_cpu_ffn_cosine": min_gpu_output_vs_cpu_ffn,
      "min_gpu_ffn_delta_vs_cpu_cosine": min_gpu_ffn_delta_vs_cpu,
      "min_gpu_ffn_norm_vs_cpu_cosine": min_gpu_ffn_norm_vs_cpu,
      "min_attn_norm_cosine": min_attn_norm,
      "min_attn_norm_from_gpu_input_cosine": min_attn_norm_from_input,
      "min_qkv_from_gpu_attn_norm_cosine": min_qkv_from_gpu_norm,
      "min_qkv_mixed_cosine": min_metric("linear_attention", "qkv_mixed"),
      "min_delta_output_cosine": min_delta_output,
      "min_z_from_gpu_attn_norm_cosine": min_z_from_gpu_norm,
      "min_z_cosine": min_z,
      "min_linear_final_output_cosine": min_linear_final,
      "min_final_mix_gpu_kernel_final_cosine": min_final,
      "min_gpu_delta_native_z_cpu_cosine": min_delta_native_z,
      "min_native_delta_gpu_z_cpu_cosine": min_native_delta_gpu_z,
      "min_native_delta_native_z_cpu_cosine": min_native_recompute,
      "min_front_projection_input_cosine": min_front_projection_input,
      "min_front_attention_output_cosine": min_front_attention_output,
      "min_gpu_attn_norm_vs_cpu_cosine": min_gpu_attn_norm_vs_cpu,
      "min_gpu_qkv_vs_cpu_cosine": min_gpu_qkv_vs_cpu,
      "min_gpu_z_vs_cpu_cosine": min_gpu_z_vs_cpu,
      "min_full_output_from_gpu_gated_cosine": min_full_output_from_gated,
      "min_full_gpu_output_vs_cpu_cosine": min_full_gpu_output_vs_cpu,
      "min_full_gated_from_gpu_core_cosine": min_full_gated_from_core,
      "min_full_gpu_gated_vs_cpu_cosine": min_full_gpu_gated_vs_cpu,
      "min_full_attn_pregate_cosine": min_full_attn_pregate,
      "min_full_q_rope_from_gpu_normed_cosine": (
          min_full_q_rope_from_gpu_normed),
      "min_full_gpu_q_rope_vs_cpu_cosine": min_full_gpu_q_rope_vs_cpu,
      "min_full_k_rope_from_gpu_normed_cosine": (
          min_full_k_rope_from_gpu_normed),
      "min_full_gpu_k_rope_vs_cpu_cosine": min_full_gpu_k_rope_vs_cpu,
      "min_core_k_history_flat_cosine": min_core_k_history,
      "min_core_v_history_flat_cosine": min_core_v_history,
      "min_core_native_inputs_cpu_cosine": min_core_native_inputs_cpu,
      "min_core_gpu_inputs_cpu_cosine": min_core_gpu_inputs_cpu,
      "min_core_gpu_core_vs_cpu_gpu_inputs_cosine": (
          min_core_gpu_core_vs_cpu),
      "min_core_gpu_q_native_history_cpu_cosine": (
          min_core_gpu_q_native_history),
      "min_core_native_q_gpu_history_cpu_cosine": (
          min_core_native_q_gpu_history),
      "min_core_native_q_gpu_k_native_v_cpu_cosine": (
          min_core_native_q_gpu_k_native_v),
      "min_core_native_q_native_k_gpu_v_cpu_cosine": (
          min_core_native_q_native_k_gpu_v),
      "speedup_claims_allowed": False,
      "disposition": (
          f"accept_{DISPOSITION_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_input_attention_output_gap_classification"
          if required else
          f"block_{DISPOSITION_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_input_attention_output_gap_classification"
      ),
      "selected_next_route": selected_next if required else CURRENT_ROUTE,
      "next_route_reason": (
          "Previous attention-output drift is inherited from the attention "
          "projection input while FFN math matches CPU on live input. Root "
          "attention projection input next."
          if required and selected_next == PROJECTION_INPUT_ROUTE else
          "Previous attention-output projection math does not match CPU on "
          "live input. Root output projection math next."
          if required and selected_next == OUTPUT_MATH_ROUTE else
          "Previous attention-output drift is inherited from full-attention "
          "core K-history drift while output projection, gate, core, and FFN "
          "math match CPU on live inputs. Root K history next."
          if required and selected_next == K_HISTORY_ROUTE else
          "Previous attention-output drift is inherited from full-attention "
          "core V-history drift while output projection, gate, core, and FFN "
          "math match CPU on live inputs. Root V history next."
          if required and selected_next == V_HISTORY_ROUTE else
          "Previous attention-output drift is inherited from full-attention "
          "Q/RoPE input drift while core and projection math match CPU. Root "
          "Q/RoPE next."
          if required and selected_next == Q_ROPE_ROUTE else
          "Previous full-attention core math does not match CPU on live input. "
          "Root full-attention core math next."
          if required and selected_next == CORE_MATH_ROUTE else
          "Previous full-attention gate math does not match CPU on live input. "
          "Root full-attention gate math next."
          if required and selected_next == GATE_MATH_ROUTE else
          "Previous attention-output drift is inherited from linear delta "
          "output; root previous delta output next."
          if required and selected_next == DELTA_ROUTE else
          "Previous attention-output drift is inherited from linear z; root "
          "previous z next."
          if required and selected_next == Z_ROUTE else
          "Previous linear preconv math does not match CPU on live input; "
          "root preconv math next."
          if required and selected_next == MATH_ROUTE else
          "Previous attention-output evidence is incomplete; keep this gate open."
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
      "# Seq475 Upstream Layer-Output FFN-Delta FFN-Norm Input Attention-Output Gap Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- diagnostic_classification: `{metrics['diagnostic_classification']}`",
      f"- min previous attention-output cosine: `{metrics['min_previous_attention_output_cosine']}`",
      f"- min linear final/delta/z cosines: `{metrics['min_linear_final_output_cosine']}` / `{metrics['min_delta_output_cosine']}` / `{metrics['min_z_cosine']}`",
      f"- min final mix cosines: `{metrics['min_final_mix_gpu_kernel_final_cosine']}` / `{metrics['min_gpu_delta_native_z_cpu_cosine']}` / `{metrics['min_native_delta_gpu_z_cpu_cosine']}`",
      f"- min preconv math cosines: `{metrics['min_gpu_attn_norm_vs_cpu_cosine']}` / `{metrics['min_gpu_qkv_vs_cpu_cosine']}` / `{metrics['min_gpu_z_vs_cpu_cosine']}`",
      f"- min attention front output/projection-input cosines: `{metrics['min_front_attention_output_cosine']}` / `{metrics['min_front_projection_input_cosine']}`",
      f"- min full output/gate/core cosines: `{metrics['min_full_output_from_gpu_gated_cosine']}` / `{metrics['min_full_gated_from_gpu_core_cosine']}` / `{metrics['min_full_attn_pregate_cosine']}`",
      f"- min full math cosines: `{metrics['min_full_gpu_output_vs_cpu_cosine']}` / `{metrics['min_full_gpu_gated_vs_cpu_cosine']}` / `{metrics['min_core_gpu_core_vs_cpu_gpu_inputs_cosine']}`",
      f"- min core q/k/v source cosines: `{metrics['min_core_gpu_q_native_history_cpu_cosine']}` / `{metrics['min_core_native_q_gpu_k_native_v_cpu_cosine']}` / `{metrics['min_core_native_q_native_k_gpu_v_cpu_cosine']}`",
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
  parser.add_argument("--seq474", type=Path, default=DEFAULT_SEQ474)
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
