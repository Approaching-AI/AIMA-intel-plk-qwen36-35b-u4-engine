#!/usr/bin/env python3
"""Locate upstream layer-output drift feeding selected full-attention inputs."""

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
BASE_GATE = (
    ROOT
    / "tools/intel-qwen36-router-full-attention-layer-input-product-source-"
    "value-gap-gate.py"
)

SCHEMA_VERSION = (
    "intel-qwen36-router-full-attention-layer-input-upstream-layer-output-"
    "value-gap-gate-v0"
)
DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_SEQ350 = (
    ROOT
    / "output/router-full-attention-layer-input-product-source-value-gap-gate-20260708Tseq350Z"
    / "metrics.json"
)
DEFAULT_TOKEN_INPUT_DIR = (
    ROOT
    / "output/r2-gpu-acceptance-matrix-router-both-rowblock16-26mask-20260708Tseq222Z"
    / "token-input"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "output/router-full-attention-layer-input-upstream-layer-output-value-gap-gate-20260708Tseq351Z"
)
DEFAULT_HOST = "local"
DEFAULT_MODEL = "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf"
DEFAULT_ENV_SCRIPT = "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh"
DEFAULT_REMOTE_ROOT = "/home/intel/intel-qwen36-gpu"

CASES = ["router_math_reason_001", "router_code_reason_002"]
COSINE_THRESHOLD = 0.9999
KLD_THRESHOLD = 0.005
TOP1_THRESHOLD = 0.99
SELECTED_FULL_ATTENTION_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31, 35]
PRECEDING_LINEAR_LAYERS = [layer - 1 for layer in SELECTED_FULL_ATTENTION_LAYERS]
DECODE_TOKENS = 8
EXPECTED_SELECTED_EVENTS = len(SELECTED_FULL_ATTENTION_LAYERS) * DECODE_TOKENS


def _load_base() -> Any:
  spec = importlib.util.spec_from_file_location("iq36_upstream_gap_base", BASE_GATE)
  if spec is None or spec.loader is None:
    raise SystemExit(f"cannot load base gate: {BASE_GATE}")
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


BASE = _load_base()


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


def _dist_fail(dist: dict[str, Any]) -> bool:
  return (
      dist.get("required_checks_passed") is False
      or _num(dist.get("max_kld")) > KLD_THRESHOLD
      or _num(dist.get("top1_rate"), 1.0) < TOP1_THRESHOLD
  )


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
      "--diagnostic-layer-range", "0:36",
      "--diagnostic-token-limit", str(DECODE_TOKENS),
  ]
  env = [
      "IQ36_OPENCL_NO_QUEUE_PROFILING=1",
      "IQ36_ATTENTION_FRONT_OUTPUT_PROJECTION_ROWBLOCK16=1",
      f"IQ36_ATTENTION_FRONT_OUTPUT_PROJECTION_ROWBLOCK16_LAYERS={BASE.BASE.ROWBLOCK16_26MASK}",
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
  return BASE.BASE.iq36_local.run_target(
      args.host, f"bash -lc {shlex.quote(remote_script)}", args.timeout_s)


def _smoke_from_stdout(run: dict[str, Any]) -> dict[str, Any]:
  return BASE.BASE._smoke_from_stdout(run)


def _distribution_summary(smoke: dict[str, Any]) -> dict[str, Any]:
  return BASE.BASE._distribution_summary(smoke)


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


def _upstream_gap_summary(smoke: dict[str, Any]) -> dict[str, Any]:
  residual_steps = smoke.get("residual_source_diff_by_step")
  residual_steps = residual_steps if isinstance(residual_steps, list) else []
  linear_steps = smoke.get("linear_attention_diff_by_step")
  linear_steps = linear_steps if isinstance(linear_steps, list) else []
  preconv_steps = smoke.get("linear_preconv_source_diff_by_step")
  preconv_steps = preconv_steps if isinstance(preconv_steps, list) else []

  selected_layer_input_obs = 0
  min_selected_layer_input = 1.0
  max_selected_layer_input_abs = 0.0
  first_selected_layer_input_gap: dict[str, Any] | None = None
  first_any_layer_input_gap: dict[str, Any] | None = None
  min_any_layer_input = 1.0
  max_any_layer_input_abs = 0.0
  for step in residual_steps:
    if not isinstance(step, dict):
      continue
    token_index = step.get("token_index")
    layers = step.get("layers")
    layers = layers if isinstance(layers, list) else []
    for row in layers:
      if not isinstance(row, dict) or row.get("layer_input_available") is not True:
        continue
      layer = row.get("layer")
      if not isinstance(layer, int):
        continue
      cosine = _num(row.get("layer_input_cosine"), 1.0)
      max_abs = _num(row.get("layer_input_max_abs_diff"))
      min_any_layer_input = min(min_any_layer_input, cosine)
      max_any_layer_input_abs = max(max_any_layer_input_abs, max_abs)
      first_any_layer_input_gap = _first_gap(
          first_any_layer_input_gap, token_index, layer, "layer_input",
          cosine, max_abs)
      if layer in SELECTED_FULL_ATTENTION_LAYERS:
        selected_layer_input_obs += 1
        min_selected_layer_input = min(min_selected_layer_input, cosine)
        max_selected_layer_input_abs = max(max_selected_layer_input_abs, max_abs)
        first_selected_layer_input_gap = _first_gap(
            first_selected_layer_input_gap, token_index, layer,
            "selected_layer_input", cosine, max_abs)

  preceding_final_obs = 0
  min_preceding_final_output = 1.0
  max_preceding_final_output_abs = 0.0
  first_preceding_final_gap: dict[str, Any] | None = None
  first_any_final_gap: dict[str, Any] | None = None
  min_any_final_output = 1.0
  max_any_final_output_abs = 0.0
  for step in linear_steps:
    if not isinstance(step, dict):
      continue
    token_index = step.get("token_index")
    layers = step.get("layers")
    layers = layers if isinstance(layers, list) else []
    for row in layers:
      if not isinstance(row, dict) or row.get("final_output_available") is not True:
        continue
      layer = row.get("layer")
      if not isinstance(layer, int):
        continue
      cosine = _num(row.get("final_output_cosine"), 1.0)
      max_abs = _num(row.get("final_output_max_abs_diff"))
      min_any_final_output = min(min_any_final_output, cosine)
      max_any_final_output_abs = max(max_any_final_output_abs, max_abs)
      first_any_final_gap = _first_gap(
          first_any_final_gap, token_index, layer, "final_output", cosine,
          max_abs)
      if layer in PRECEDING_LINEAR_LAYERS:
        preceding_final_obs += 1
        min_preceding_final_output = min(min_preceding_final_output, cosine)
        max_preceding_final_output_abs = max(
            max_preceding_final_output_abs, max_abs)
        first_preceding_final_gap = _first_gap(
            first_preceding_final_gap, token_index, layer,
            "preceding_final_output", cosine, max_abs)

  preceding_preconv_obs = 0
  min_preceding_gpu_attn_norm_vs_cpu = 1.0
  min_preceding_final_input_norm = 1.0
  first_preceding_norm_math_gap: dict[str, Any] | None = None
  for step in preconv_steps:
    if not isinstance(step, dict):
      continue
    token_index = step.get("token_index")
    layers = step.get("layers")
    layers = layers if isinstance(layers, list) else []
    for row in layers:
      if not isinstance(row, dict):
        continue
      layer = row.get("layer")
      if not isinstance(layer, int) or layer not in PRECEDING_LINEAR_LAYERS:
        continue
      if row.get("gpu_attn_norm_vs_cpu_available") is not True:
        continue
      preceding_preconv_obs += 1
      norm_math = _num(row.get("gpu_attn_norm_vs_cpu_cosine"), 1.0)
      norm_input = _num(row.get("attn_norm_from_gpu_input_cosine"), 1.0)
      norm_abs = _num(row.get("gpu_attn_norm_vs_cpu_max_abs_diff"))
      min_preceding_gpu_attn_norm_vs_cpu = min(
          min_preceding_gpu_attn_norm_vs_cpu, norm_math)
      min_preceding_final_input_norm = min(
          min_preceding_final_input_norm, norm_input)
      first_preceding_norm_math_gap = _first_gap(
          first_preceding_norm_math_gap, token_index, layer,
          "gpu_attn_norm_vs_cpu", norm_math, norm_abs)

  return {
      "selected_full_attention_layers": SELECTED_FULL_ATTENTION_LAYERS,
      "preceding_linear_layers": PRECEDING_LINEAR_LAYERS,
      "expected_selected_events": EXPECTED_SELECTED_EVENTS,
      "selected_layer_input_observation_count": selected_layer_input_obs,
      "preceding_final_output_observation_count": preceding_final_obs,
      "preceding_preconv_observation_count": preceding_preconv_obs,
      "min_selected_layer_input_cosine": min_selected_layer_input,
      "max_selected_layer_input_abs_diff": max_selected_layer_input_abs,
      "min_any_layer_input_cosine": min_any_layer_input,
      "max_any_layer_input_abs_diff": max_any_layer_input_abs,
      "min_preceding_final_output_cosine": min_preceding_final_output,
      "max_preceding_final_output_abs_diff": max_preceding_final_output_abs,
      "min_any_linear_final_output_cosine": min_any_final_output,
      "max_any_linear_final_output_abs_diff": max_any_final_output_abs,
      "min_preceding_gpu_attn_norm_vs_cpu_cosine": (
          min_preceding_gpu_attn_norm_vs_cpu),
      "min_preceding_attn_norm_from_gpu_input_cosine": (
          min_preceding_final_input_norm),
      "first_selected_layer_input_gap": first_selected_layer_input_gap,
      "first_any_layer_input_gap": first_any_layer_input_gap,
      "first_preceding_final_output_gap": first_preceding_final_gap,
      "first_any_linear_final_output_gap": first_any_final_gap,
      "first_preceding_norm_math_gap": first_preceding_norm_math_gap,
  }


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
      "speedup_claims_allowed": smoke.get("speedup_claims_allowed"),
      "distribution": _distribution_summary(smoke),
      "upstream_gap": _upstream_gap_summary(smoke),
  }


def compute(args: argparse.Namespace) -> dict[str, Any]:
  routes = _load_json(args.routes)
  seq350 = _load_json(args.seq350)
  binary = seq350.get("inputs", {}).get("binary")
  if not isinstance(binary, str) or not binary:
    raise SystemExit("seq350 binary missing")
  token_cache = BASE.BASE.iq36_local.ensure_cached_tokens(
      args.host, f"{args.remote_root}/cache", args.token_input_dir,
      args.timeout_s)

  runs: list[dict[str, Any]] = []
  if token_cache.get("ok") is True:
    for case_id in CASES:
      run = _run_case(args, binary, str(token_cache.get("dir")), case_id)
      smoke = _smoke_from_stdout(run)
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
        and summary.get("source_layers") == BASE.BASE.EXPECTED_LAYER_EVENTS
        and summary.get("source_values") == BASE.BASE.EXPECTED_VALUES
        and summary.get("source_misses") == 0
        and summary.get("source_ready") is True
        and summary.get("consumer_layers") == BASE.BASE.EXPECTED_LAYER_EVENTS
        and summary.get("consumer_values") == BASE.BASE.EXPECTED_VALUES
        and summary.get("consumer_misses") == 0
        and summary.get("consumer_ready") is True
        and summary.get("cpu_shadow_layer_input_layers") == 0
        and summary.get("speedup_claims_allowed") is False
    )

  preconditions_pass = (
      seq350.get("required_checks_passed") is True
      and seq350.get("selected_next_route")
      == "router_prompt_full_attention_layer_input_upstream_layer_output_value_gap_gate"
      and seq350.get("diagnostic_classification")
      == "upstream_layer_output_value_gap"
      and _has_candidate(
          routes, 350,
          "accept_layer_input_product_source_value_gap_classification")
      and _has_switch(
          routes,
          "select_router_prompt_full_attention_layer_input_upstream_layer_output_value_gap_gate",
          350)
  )
  rows_emitted = (
      len(runs) == len(CASES)
      and all(row.get("run", {}).get("returncode") in (0, 2) for row in runs)
  )
  counters_ready = rows_emitted and all(row_ready(row) for row in runs)
  distribution_reproduced = rows_emitted and all(
      _dist_fail(row.get("summary", {}).get("distribution", {}))
      for row in runs)
  upstream_emitted = counters_ready and all(
      row.get("summary", {}).get("upstream_gap", {})
      .get("selected_layer_input_observation_count")
      == EXPECTED_SELECTED_EVENTS
      and row.get("summary", {}).get("upstream_gap", {})
      .get("preceding_final_output_observation_count")
      == EXPECTED_SELECTED_EVENTS
      for row in runs)
  min_preceding_final = min(
      (_num(row.get("summary", {}).get("upstream_gap", {})
            .get("min_preceding_final_output_cosine"), 1.0)
       for row in runs),
      default=1.0)
  max_preceding_final_abs = max(
      (_num(row.get("summary", {}).get("upstream_gap", {})
            .get("max_preceding_final_output_abs_diff"))
       for row in runs),
      default=0.0)
  min_preceding_norm_math = min(
      (_num(row.get("summary", {}).get("upstream_gap", {})
            .get("min_preceding_gpu_attn_norm_vs_cpu_cosine"), 1.0)
       for row in runs),
      default=1.0)
  preceding_output_gap = (
      upstream_emitted and min_preceding_final < COSINE_THRESHOLD)
  preceding_norm_math_ok = (
      upstream_emitted and min_preceding_norm_math >= COSINE_THRESHOLD)
  diagnostic_classification = (
      "preceding_linear_output_value_gap"
      if preceding_output_gap and preceding_norm_math_ok else
      "preceding_linear_math_gap"
      if min_preceding_norm_math < COSINE_THRESHOLD else
      "upstream_gap_not_reproduced"
  )
  selected_next = (
      "router_prompt_full_attention_layer_input_preceding_linear_output_gap_gate"
      if diagnostic_classification == "preceding_linear_output_value_gap" else
      "router_prompt_full_attention_layer_input_preceding_linear_math_gap_gate"
      if diagnostic_classification == "preceding_linear_math_gap" else
      "router_prompt_full_attention_layer_input_upstream_layer_output_value_gap_gate"
  )
  checks = [
      {"name": "seq350_selected_upstream_gap_gate",
       "pass": preconditions_pass},
      {"name": "upstream_gap_rows_emitted", "pass": rows_emitted},
      {"name": "source_and_consumer_counters_ready",
       "pass": counters_ready,
       "detail": [row.get("summary", {}) for row in runs]},
      {"name": "upstream_gap_diagnostics_emitted",
       "pass": upstream_emitted,
       "detail": [
           row.get("summary", {}).get("upstream_gap", {}) for row in runs
       ]},
      {"name": "distribution_failure_reproduced",
       "pass": distribution_reproduced,
       "detail": [
           row.get("summary", {}).get("distribution", {}) for row in runs
       ]},
      {"name": "preceding_linear_norm_math_not_the_gap",
       "pass": preceding_norm_math_ok,
       "detail": {
           "min_preceding_gpu_attn_norm_vs_cpu_cosine": (
               min_preceding_norm_math),
           "threshold": COSINE_THRESHOLD,
       }},
  ]
  required = all(bool(row.get("pass")) for row in checks)
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "inputs": {
          "routes": _rel(args.routes),
          "seq350": _rel(args.seq350),
          "token_input_dir": _rel(args.token_input_dir),
          "host": args.host,
          "binary": binary,
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
      "preceding_output_gap": preceding_output_gap,
      "preceding_norm_math_ok": preceding_norm_math_ok,
      "min_preceding_linear_final_output_cosine": min_preceding_final,
      "max_preceding_linear_final_output_abs_diff": max_preceding_final_abs,
      "min_preceding_gpu_attn_norm_vs_cpu_cosine": min_preceding_norm_math,
      "speedup_claims_allowed": False,
      "disposition": (
          "accept_upstream_layer_output_value_gap_classification"
          if required else "block_upstream_layer_output_value_gap_classification"
      ),
      "selected_next_route": (
          selected_next if required else
          "router_prompt_full_attention_layer_input_upstream_layer_output_value_gap_gate"
      ),
      "next_route_reason": (
          "The selected full-attention layer-input gap is present in the "
          "preceding linear layer final outputs, while the preceding linear "
          "RMSNorm path matches CPU on live inputs. The next unit should isolate "
          "which preceding linear output producer or earlier input drift causes "
          "that final-output gap."
          if required else
          "Upstream layer-output evidence is incomplete; keep this gate open."
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
      "# Router Full-Attention Layer-Input Upstream Layer-Output Value-Gap Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- diagnostic_classification: `{metrics['diagnostic_classification']}`",
      f"- preceding_output_gap: `{str(metrics['preceding_output_gap']).lower()}`",
      f"- preceding_norm_math_ok: `{str(metrics['preceding_norm_math_ok']).lower()}`",
      f"- min preceding linear final-output cosine: `{metrics['min_preceding_linear_final_output_cosine']}`",
      f"- max preceding linear final-output abs diff: `{metrics['max_preceding_linear_final_output_abs_diff']}`",
      f"- min preceding GPU RMSNorm-vs-CPU cosine: `{metrics['min_preceding_gpu_attn_norm_vs_cpu_cosine']}`",
      f"- failed_checks: `{failed}`",
      f"- speedup_claims_allowed: `{str(metrics['speedup_claims_allowed']).lower()}`",
      "",
  ]
  for row in metrics["runs"]:
    summary = row["summary"]
    dist = summary["distribution"]
    gap = summary["upstream_gap"]
    lines.extend([
        f"## {summary['case_id']}",
        "",
        f"- distribution max KLD: `{dist.get('max_kld')}`",
        f"- distribution top1 rate: `{dist.get('top1_rate')}`",
        f"- source counters: `{summary.get('source_layers')}` / `{summary.get('source_values')}` / `{summary.get('source_misses')}`",
        f"- consumer counters: `{summary.get('consumer_layers')}` / `{summary.get('consumer_values')}` / `{summary.get('consumer_misses')}`",
        f"- min selected layer-input cosine: `{gap.get('min_selected_layer_input_cosine')}`",
        f"- min preceding final-output cosine: `{gap.get('min_preceding_final_output_cosine')}`",
        f"- max preceding final-output abs diff: `{gap.get('max_preceding_final_output_abs_diff')}`",
        f"- min preceding GPU RMSNorm-vs-CPU cosine: `{gap.get('min_preceding_gpu_attn_norm_vs_cpu_cosine')}`",
        f"- first preceding final-output gap: `{gap.get('first_preceding_final_output_gap')}`",
        "",
    ])
  lines.extend([
      metrics["next_route_reason"],
      "",
      "This is distribution/correctness evidence only. It is not a speed claim.",
      "",
  ])
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--routes", type=Path, default=DEFAULT_ROUTES)
  parser.add_argument("--seq350", type=Path, default=DEFAULT_SEQ350)
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
      "preceding_norm_math_ok": metrics["preceding_norm_math_ok"],
      "preceding_output_gap": metrics["preceding_output_gap"],
      "required_checks_passed": metrics["required_checks_passed"],
      "selected_next_route": metrics["selected_next_route"],
  }, sort_keys=True))
  return 0 if metrics["required_checks_passed"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
