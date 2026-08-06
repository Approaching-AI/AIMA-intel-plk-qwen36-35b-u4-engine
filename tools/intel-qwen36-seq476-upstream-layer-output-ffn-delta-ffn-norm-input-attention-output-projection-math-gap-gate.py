#!/usr/bin/env python3
"""Classify seq475 previous attention output-projection math drift."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shlex
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SEQ475_GATE = (
    ROOT
    / "tools/intel-qwen36-seq475-upstream-layer-output-ffn-delta-ffn-norm-input-attention-output-gap-gate.py"
)
SCHEMA_VERSION = (
    "intel-qwen36-seq476-upstream-layer-output-ffn-delta-ffn-norm-input-"
    "attention-output-projection-math-gap-gate-v0"
)

DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_SEQ475 = (
    ROOT
    / "output/seq475-upstream-layer-output-ffn-delta-ffn-norm-input-attention-output-gap-gate-20260709Tseq475Z"
    / "metrics.json"
)
DEFAULT_TOKEN_INPUT_DIR = (
    ROOT
    / "output/r2-gpu-acceptance-matrix-router-both-rowblock16-26mask-20260708Tseq222Z"
    / "token-input"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "output/seq476-upstream-layer-output-ffn-delta-ffn-norm-input-attention-output-projection-math-gap-gate-20260709Tseq476Z"
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


SEQ475 = _load_module(SEQ475_GATE, "iq36_seq475_gate")
UPSTREAM = SEQ475.SEQ474.SEQ473.SEQ472.SEQ471.BASE.OLD.BASE
DIST_FIX = UPSTREAM.BASE.BASE
REMOTE = DIST_FIX
CASES = SEQ475.SEQ474.SEQ473.SEQ472.SEQ471.BASE.CASES
CURRENT_ROUTE = SEQ475.OUTPUT_MATH_ROUTE
Q8_BRIDGE_ROUTE = CURRENT_ROUTE.replace(
    "_attention_output_projection_math_gap_gate",
    "_attention_output_projection_q8_bridge_gap_gate")
DEVICE_HANDOFF_ROUTE = CURRENT_ROUTE.replace(
    "_attention_output_projection_math_gap_gate",
    "_attention_output_projection_device_handoff_gap_gate")
HOST_DEVICE_ROUTE = CURRENT_ROUTE.replace(
    "_attention_output_projection_math_gap_gate",
    "_attention_output_projection_host_device_mismatch_gap_gate")
HOST_MATH_ROUTE = CURRENT_ROUTE.replace(
    "_attention_output_projection_math_gap_gate",
    "_attention_output_projection_host_math_gap_gate")
DIAG_PREFIX = SEQ475.DIAG_PREFIX
DISPOSITION_PREFIX = SEQ475.DISPOSITION_PREFIX


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


def _run_case(args: argparse.Namespace, binary: str, remote_token_dir: str,
              case_id: str) -> dict[str, Any]:
  flags = [
      "--model", shlex.quote(args.model),
      "--token-dir", shlex.quote(remote_token_dir),
      "--case-id", case_id,
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
  env = [
      "IQ36_OPENCL_NO_QUEUE_PROFILING=1",
      "IQ36_ATTENTION_FRONT_OUTPUT_PROJECTION_ROWBLOCK16=1",
      f"IQ36_ATTENTION_FRONT_OUTPUT_PROJECTION_ROWBLOCK16_LAYERS={DIST_FIX.ROWBLOCK16_26MASK}",
      "IQ36_SELECTED_SHARED_Q4_GATEUP_COMBINED=1",
      "IQ36_SELECTED_SHARED_Q4_DOWN_COMBINED=1",
      "IQ36_SELECTED_SHARED_Q6_DOWN_COMBINED=1",
      "IQ36_DEFER_FFN_DOWN_FINISH_BUNDLE=1",
      "IQ36_FULL_ATTENTION_LAYER_INPUT_PRODUCT_SOURCE=1",
      "IQ36_FULL_ATTENTION_LAYER_INPUT_PRODUCT_CONSUMER_SOURCE=1",
  ]
  remote_script = " && ".join([
      f"source {shlex.quote(args.env_script)} >/dev/null 2>&1",
      " ".join([*env, shlex.quote(binary), *flags]),
  ])
  return REMOTE.iq36_local.run_target(
      args.host, f"bash -lc {shlex.quote(remote_script)}", args.timeout_s)


def _case_summary(case_id: str, run: dict[str, Any],
                  smoke: dict[str, Any]) -> dict[str, Any]:
  q8_steps = smoke.get("linear_final_q8_equivalence_diff_by_step")
  q8_steps = q8_steps if isinstance(q8_steps, list) else []
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
      "full_core_q8_equivalence_diff_enabled": smoke.get(
          "full_core_q8_equivalence_diff_enabled"),
      "distribution": UPSTREAM._distribution_summary(smoke),
      "attention_output_source": SEQ475._attention_output_source_summary(smoke),
      "linear_final_q8_equivalence": _metric_summary(q8_steps, [
          "host_projection_vs_native",
          "device_projection_vs_native",
          "host_vs_device_projection",
          "host_vs_device_residual",
          "host_vs_device_norm",
      ], "host_projection_vs_native"),
  }


def compute(args: argparse.Namespace) -> dict[str, Any]:
  routes = _load_json(args.routes)
  seq475 = _load_json(args.seq475)
  binary = seq475.get("inputs", {}).get("binary")
  if not isinstance(binary, str) or not binary:
    raise SystemExit("seq475 binary missing")
  token_cache = REMOTE.iq36_local.ensure_cached_tokens(
      args.host, f"{args.remote_root}/cache", args.token_input_dir,
      args.timeout_s)

  runs: list[dict[str, Any]] = []
  if token_cache.get("ok") is True:
    for case_id in CASES:
      run = _run_case(args, binary, str(token_cache.get("dir")), case_id)
      smoke = UPSTREAM._smoke_from_stdout(run)
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
      seq475.get("required_checks_passed") is True
      and seq475.get("selected_next_route") == CURRENT_ROUTE
      and seq475.get("diagnostic_classification")
      == f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_input_attention_output_projection_math_gap"
      and _has_candidate(routes, 475, str(seq475.get("disposition")))
      and _has_switch(routes, f"select_{CURRENT_ROUTE}", 475)
  )
  rows_emitted = (
      len(runs) == len(CASES)
      and all(row.get("run", {}).get("returncode") in (0, 2)
              for row in runs)
  )
  counters_ready = rows_emitted and all(row_ready(row) for row in runs)
  distribution_reproduced = rows_emitted and all(
      SEQ475.SEQ474.SEQ473.SEQ472._dist_fail(
          row.get("summary", {}).get("distribution", {}))
      for row in runs)
  diagnostics_emitted = counters_ready and all(
      row.get("summary", {}).get("attention_output_source", {})
      .get("attention_front", {}).get("observation_count") == EXPECTED_EVENTS
      and row.get("summary", {}).get("linear_final_q8_equivalence", {})
      .get("observation_count") == EXPECTED_EVENTS
      and row.get("summary", {}).get("full_core_q8_equivalence_diff_enabled")
      is True
      for row in runs)

  def min_attention(group: str, name: str) -> float:
    return min(
        (_num(row.get("summary", {}).get("attention_output_source", {})
              .get(group, {}).get(f"min_{name}_cosine"), 1.0)
         for row in runs),
        default=1.0)

  def min_q8(name: str) -> float:
    return min(
        (_num(row.get("summary", {}).get("linear_final_q8_equivalence", {})
              .get(f"min_{name}_cosine"), 1.0)
         for row in runs),
        default=1.0)

  min_front_projection_input = min_attention(
      "attention_front", "projection_input")
  min_front_attention_output = min_attention(
      "attention_front", "attention_output")
  min_host_projection_vs_native = min_q8("host_projection_vs_native")
  min_device_projection_vs_native = min_q8("device_projection_vs_native")
  min_host_vs_device_projection = min_q8("host_vs_device_projection")
  min_host_vs_device_residual = min_q8("host_vs_device_residual")
  min_host_vs_device_norm = min_q8("host_vs_device_norm")

  attention_projection_input_clean = (
      diagnostics_emitted
      and min_front_projection_input >= COSINE_THRESHOLD)
  attention_output_gap = (
      diagnostics_emitted
      and min_front_attention_output < COSINE_THRESHOLD)
  host_projection_gap = (
      diagnostics_emitted
      and min_host_projection_vs_native < COSINE_THRESHOLD)
  device_projection_gap = (
      diagnostics_emitted
      and min_device_projection_vs_native < COSINE_THRESHOLD)
  host_device_projection_match = (
      diagnostics_emitted
      and min_host_vs_device_projection >= COSINE_THRESHOLD)
  host_device_residual_match = (
      diagnostics_emitted
      and min_host_vs_device_residual >= COSINE_THRESHOLD)
  host_device_norm_match = (
      diagnostics_emitted
      and min_host_vs_device_norm >= COSINE_THRESHOLD)
  host_device_bridge_match = (
      host_device_projection_match
      and host_device_residual_match
      and host_device_norm_match)
  host_projection_clean = (
      diagnostics_emitted
      and min_host_projection_vs_native >= COSINE_THRESHOLD)
  device_projection_clean = (
      diagnostics_emitted
      and min_device_projection_vs_native >= COSINE_THRESHOLD)

  projection_q8_bridge_gap = (
      attention_output_gap
      and attention_projection_input_clean
      and host_projection_gap
      and device_projection_gap
      and host_device_bridge_match)
  projection_device_handoff_gap = (
      attention_output_gap
      and attention_projection_input_clean
      and host_projection_clean
      and device_projection_gap)
  projection_host_device_mismatch = (
      attention_output_gap
      and attention_projection_input_clean
      and not host_device_projection_match)
  projection_host_math_gap = (
      attention_output_gap
      and attention_projection_input_clean
      and host_projection_gap
      and device_projection_clean)

  diagnostic_classification = (
      f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_input_attention_output_projection_q8_bridge_gap"
      if projection_q8_bridge_gap else
      f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_input_attention_output_projection_device_handoff_gap"
      if projection_device_handoff_gap else
      f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_input_attention_output_projection_host_device_mismatch_gap"
      if projection_host_device_mismatch else
      f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_input_attention_output_projection_host_math_gap"
      if projection_host_math_gap else
      f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_input_attention_output_projection_math_gap_unclassified"
  )
  selected_next = (
      Q8_BRIDGE_ROUTE
      if projection_q8_bridge_gap else
      DEVICE_HANDOFF_ROUTE
      if projection_device_handoff_gap else
      HOST_DEVICE_ROUTE
      if projection_host_device_mismatch else
      HOST_MATH_ROUTE
      if projection_host_math_gap else
      CURRENT_ROUTE
  )
  projection_source_classified = selected_next != CURRENT_ROUTE
  checks = [
      {"name": "seq475_selected_projection_math_gap_gate",
       "pass": preconditions_pass},
      {"name": "projection_math_rows_emitted", "pass": rows_emitted},
      {"name": "source_and_consumer_counters_ready",
       "pass": counters_ready,
       "detail": [row.get("summary", {}) for row in runs]},
      {"name": "distribution_failure_reproduced",
       "pass": distribution_reproduced,
       "detail": [
           row.get("summary", {}).get("distribution", {}) for row in runs
       ]},
      {"name": "q8_equivalence_diagnostics_emitted",
       "pass": diagnostics_emitted,
       "detail": [
           {
               "attention_front": row.get("summary", {})
               .get("attention_output_source", {}).get("attention_front", {}),
               "linear_final_q8_equivalence": row.get("summary", {})
               .get("linear_final_q8_equivalence", {}),
           }
           for row in runs
       ]},
      {"name": "attention_projection_input_clean",
       "pass": attention_projection_input_clean,
       "detail": {
           "min_front_projection_input_cosine": min_front_projection_input,
       }},
      {"name": "attention_output_gap_reproduced",
       "pass": attention_output_gap,
       "detail": {
           "min_front_attention_output_cosine": min_front_attention_output,
       }},
      {"name": "projection_q8_source_classified",
       "pass": projection_source_classified,
       "detail": {
           "projection_q8_bridge_gap": projection_q8_bridge_gap,
           "projection_device_handoff_gap": projection_device_handoff_gap,
           "projection_host_device_mismatch": projection_host_device_mismatch,
           "projection_host_math_gap": projection_host_math_gap,
           "min_host_projection_vs_native_cosine": (
               min_host_projection_vs_native),
           "min_device_projection_vs_native_cosine": (
               min_device_projection_vs_native),
           "min_host_vs_device_projection_cosine": (
               min_host_vs_device_projection),
           "min_host_vs_device_residual_cosine": min_host_vs_device_residual,
           "min_host_vs_device_norm_cosine": min_host_vs_device_norm,
       }},
  ]
  required = all(bool(row.get("pass")) for row in checks)
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "inputs": {
          "routes": _rel(args.routes),
          "seq475": _rel(args.seq475),
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
      "attention_projection_input_clean": attention_projection_input_clean,
      "attention_output_gap": attention_output_gap,
      "host_projection_gap": host_projection_gap,
      "device_projection_gap": device_projection_gap,
      "host_device_projection_match": host_device_projection_match,
      "host_device_residual_match": host_device_residual_match,
      "host_device_norm_match": host_device_norm_match,
      "projection_q8_bridge_gap": projection_q8_bridge_gap,
      "projection_device_handoff_gap": projection_device_handoff_gap,
      "projection_host_device_mismatch": projection_host_device_mismatch,
      "projection_host_math_gap": projection_host_math_gap,
      "min_front_projection_input_cosine": min_front_projection_input,
      "min_front_attention_output_cosine": min_front_attention_output,
      "min_host_projection_vs_native_cosine": min_host_projection_vs_native,
      "min_device_projection_vs_native_cosine": min_device_projection_vs_native,
      "min_host_vs_device_projection_cosine": min_host_vs_device_projection,
      "min_host_vs_device_residual_cosine": min_host_vs_device_residual,
      "min_host_vs_device_norm_cosine": min_host_vs_device_norm,
      "speedup_claims_allowed": False,
      "disposition": (
          f"accept_{DISPOSITION_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_input_attention_output_projection_math_gap_classification"
          if required else
          f"block_{DISPOSITION_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_input_attention_output_projection_math_gap_classification"
      ),
      "selected_next_route": selected_next if required else CURRENT_ROUTE,
      "next_route_reason": (
          "Host and device output-projection replay both reproduce the native "
          "attention-output drift while host/device replay matches. Root the "
          "Q8 projection bridge next."
          if required and selected_next == Q8_BRIDGE_ROUTE else
          "Host projection replay matches native while device projection replay "
          "reproduces the drift. Root the projection device handoff next."
          if required and selected_next == DEVICE_HANDOFF_ROUTE else
          "Host and device output-projection replay diverge on the same live "
          "input. Root the host/device projection mismatch next."
          if required and selected_next == HOST_DEVICE_ROUTE else
          "Host projection replay reproduces the drift while device replay "
          "matches native. Root host projection math next."
          if required and selected_next == HOST_MATH_ROUTE else
          "Projection Q8 evidence is incomplete; keep this gate open."
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
      "# Seq476 Upstream Layer-Output Projection-Math Gap Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- diagnostic_classification: `{metrics['diagnostic_classification']}`",
      f"- min attention front output/projection-input cosines: `{metrics['min_front_attention_output_cosine']}` / `{metrics['min_front_projection_input_cosine']}`",
      f"- min host/device projection vs native cosines: `{metrics['min_host_projection_vs_native_cosine']}` / `{metrics['min_device_projection_vs_native_cosine']}`",
      f"- min host-vs-device projection/residual/norm cosines: `{metrics['min_host_vs_device_projection_cosine']}` / `{metrics['min_host_vs_device_residual_cosine']}` / `{metrics['min_host_vs_device_norm_cosine']}`",
      f"- projection_q8_bridge_gap: `{str(metrics['projection_q8_bridge_gap']).lower()}`",
      f"- projection_device_handoff_gap: `{str(metrics['projection_device_handoff_gap']).lower()}`",
      f"- projection_host_device_mismatch: `{str(metrics['projection_host_device_mismatch']).lower()}`",
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
  parser.add_argument("--seq475", type=Path, default=DEFAULT_SEQ475)
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
