#!/usr/bin/env python3
"""Locate upstream drift feeding preceding source layer inputs."""

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
BASE_GATE = (
    ROOT
    / "tools/intel-qwen36-router-full-attention-layer-input-upstream-"
    "layer-output-value-gap-gate.py"
)

SCHEMA_VERSION = (
    "intel-qwen36-router-full-attention-layer-input-preceding-linear-"
    "input-source-producer-linear-input-source-linear-input-source-layer-"
    "input-gap-gate-v0"
)
DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_SEQ364 = (
    ROOT
    / "output/router-full-attention-layer-input-preceding-linear-input-source-producer-linear-input-source-linear-input-source-gap-gate-20260708Tseq364Z"
    / "metrics.json"
)
DEFAULT_SEQ365 = (
    ROOT
    / "output/router-full-attention-layer-input-preceding-linear-input-source-producer-linear-input-source-linear-input-source-ffn-input-gap-gate-20260708Tseq365Z"
    / "metrics.json"
)
DEFAULT_TOKEN_INPUT_DIR = (
    ROOT
    / "output/r2-gpu-acceptance-matrix-router-both-rowblock16-26mask-20260708Tseq222Z"
    / "token-input"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "output/router-full-attention-layer-input-preceding-linear-input-source-producer-linear-input-source-linear-input-source-layer-input-gap-gate-20260708Tseq366Z"
)
DEFAULT_HOST = "local"
DEFAULT_MODEL = "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf"
DEFAULT_ENV_SCRIPT = "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh"
DEFAULT_REMOTE_ROOT = "/home/intel/intel-qwen36-gpu"

CASES = ["router_math_reason_001", "router_code_reason_002"]
TARGET_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31]
PRECEDING_LINEAR_LAYERS = [layer - 1 for layer in TARGET_LAYERS]
DECODE_TOKENS = 8
EXPECTED_EVENTS = len(TARGET_LAYERS) * DECODE_TOKENS
EXPECTED_COUNTER_LAYERS = 72
EXPECTED_COUNTER_VALUES = 147456
COSINE_THRESHOLD = 0.9999
KLD_THRESHOLD = 0.005
TOP1_THRESHOLD = 0.99


def _load_base() -> Any:
  spec = importlib.util.spec_from_file_location(
      "iq36_source_layer_input_gap_base", BASE_GATE)
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


def _source_layer_input_summary(smoke: dict[str, Any]) -> dict[str, Any]:
  residual_steps = smoke.get("residual_source_diff_by_step")
  residual_steps = residual_steps if isinstance(residual_steps, list) else []
  linear_steps = smoke.get("linear_attention_diff_by_step")
  linear_steps = linear_steps if isinstance(linear_steps, list) else []
  preconv_steps = smoke.get("linear_preconv_source_diff_by_step")
  preconv_steps = preconv_steps if isinstance(preconv_steps, list) else []

  target_obs = 0
  min_target_layer_input = 1.0
  min_target_ffn_input = 1.0
  max_target_layer_input_abs = 0.0
  first_target_gap: dict[str, Any] | None = None
  for step in residual_steps:
    if not isinstance(step, dict):
      continue
    token_index = step.get("token_index")
    layers = step.get("layers")
    layers = layers if isinstance(layers, list) else []
    for row in layers:
      if not isinstance(row, dict):
        continue
      layer = row.get("layer")
      if not isinstance(layer, int) or layer not in TARGET_LAYERS:
        continue
      if row.get("layer_input_available") is not True:
        continue
      target_obs += 1
      layer_cos = _num(row.get("layer_input_cosine"), 1.0)
      layer_abs = _num(row.get("layer_input_max_abs_diff"))
      ffn_cos = _num(row.get("ffn_input_cosine"), 1.0)
      min_target_layer_input = min(min_target_layer_input, layer_cos)
      min_target_ffn_input = min(min_target_ffn_input, ffn_cos)
      max_target_layer_input_abs = max(max_target_layer_input_abs, layer_abs)
      first_target_gap = _first_gap(
          first_target_gap, token_index, layer, "layer_input",
          layer_cos, layer_abs)

  preceding_final_obs = 0
  min_preceding_final_output = 1.0
  max_preceding_final_output_abs = 0.0
  first_preceding_final_gap: dict[str, Any] | None = None
  for step in linear_steps:
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
      if row.get("final_output_available") is not True:
        continue
      preceding_final_obs += 1
      final_cos = _num(row.get("final_output_cosine"), 1.0)
      final_abs = _num(row.get("final_output_max_abs_diff"))
      min_preceding_final_output = min(min_preceding_final_output, final_cos)
      max_preceding_final_output_abs = max(
          max_preceding_final_output_abs, final_abs)
      first_preceding_final_gap = _first_gap(
          first_preceding_final_gap, token_index, layer,
          "preceding_final_output", final_cos, final_abs)

  preceding_preconv_obs = 0
  min_preceding_attn_norm_from_input = 1.0
  min_preceding_gpu_attn_norm_vs_cpu = 1.0
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
      norm_input = _num(row.get("attn_norm_from_gpu_input_cosine"), 1.0)
      norm_math = _num(row.get("gpu_attn_norm_vs_cpu_cosine"), 1.0)
      norm_math_abs = _num(row.get("gpu_attn_norm_vs_cpu_max_abs_diff"))
      min_preceding_attn_norm_from_input = min(
          min_preceding_attn_norm_from_input, norm_input)
      min_preceding_gpu_attn_norm_vs_cpu = min(
          min_preceding_gpu_attn_norm_vs_cpu, norm_math)
      first_preceding_norm_math_gap = _first_gap(
          first_preceding_norm_math_gap, token_index, layer,
          "gpu_attn_norm_vs_cpu", norm_math, norm_math_abs)

  return {
      "target_layers": TARGET_LAYERS,
      "preceding_linear_layers": PRECEDING_LINEAR_LAYERS,
      "expected_observation_count": EXPECTED_EVENTS,
      "target_layer_input_observation_count": target_obs,
      "preceding_final_output_observation_count": preceding_final_obs,
      "preceding_preconv_observation_count": preceding_preconv_obs,
      "min_target_layer_input_cosine": min_target_layer_input,
      "max_target_layer_input_abs_diff": max_target_layer_input_abs,
      "min_target_ffn_input_cosine": min_target_ffn_input,
      "min_preceding_final_output_cosine": min_preceding_final_output,
      "max_preceding_final_output_abs_diff": max_preceding_final_output_abs,
      "min_preceding_attn_norm_from_gpu_input_cosine": (
          min_preceding_attn_norm_from_input),
      "min_preceding_gpu_attn_norm_vs_cpu_cosine": (
          min_preceding_gpu_attn_norm_vs_cpu),
      "first_target_layer_input_gap": first_target_gap,
      "first_preceding_final_output_gap": first_preceding_final_gap,
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
      "cpu_shadow_attention_output_layers": smoke.get(
          "cpu_shadow_attention_output_layers"),
      "speedup_claims_allowed": smoke.get("speedup_claims_allowed"),
      "distribution": BASE._distribution_summary(smoke),
      "source_layer_input": _source_layer_input_summary(smoke),
  }


def compute(args: argparse.Namespace) -> dict[str, Any]:
  routes = _load_json(args.routes)
  seq364 = _load_json(args.seq364)
  seq365 = _load_json(args.seq365)
  binary = seq364.get("inputs", {}).get("binary")
  if not isinstance(binary, str) or not binary:
    raise SystemExit("seq364 binary missing")
  token_cache = BASE.BASE.BASE.iq36_local.ensure_cached_tokens(
      args.host, f"{args.remote_root}/cache", args.token_input_dir,
      args.timeout_s)

  runs: list[dict[str, Any]] = []
  if token_cache.get("ok") is True:
    for case_id in CASES:
      run = BASE._run_case(args, binary, str(token_cache.get("dir")), case_id)
      smoke = BASE._smoke_from_stdout(run)
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
      seq365.get("required_checks_passed") is True
      and seq365.get("selected_next_route")
      == "router_prompt_full_attention_layer_input_preceding_linear_input_source_producer_linear_input_source_linear_input_source_layer_input_gap_gate"
      and seq365.get("diagnostic_classification")
      == "source_ffn_input_layer_input_gap"
      and _has_candidate(
          routes, 365, "accept_source_ffn_input_gap_classification")
      and _has_switch(
          routes,
          "select_router_prompt_full_attention_layer_input_preceding_linear_input_source_producer_linear_input_source_linear_input_source_layer_input_gap_gate",
          365)
  )
  rows_emitted = (
      len(runs) == len(CASES)
      and all(row.get("run", {}).get("returncode") in (0, 2) for row in runs)
  )
  counters_ready = rows_emitted and all(row_ready(row) for row in runs)
  distribution_reproduced = rows_emitted and all(
      _dist_fail(row.get("summary", {}).get("distribution", {}))
      for row in runs)
  diagnostics_emitted = counters_ready and all(
      row.get("summary", {}).get("source_layer_input", {})
      .get("target_layer_input_observation_count") == EXPECTED_EVENTS
      and row.get("summary", {}).get("source_layer_input", {})
      .get("preceding_final_output_observation_count") == EXPECTED_EVENTS
      and row.get("summary", {}).get("source_layer_input", {})
      .get("preceding_preconv_observation_count") == EXPECTED_EVENTS
      for row in runs)

  def min_metric(name: str) -> float:
    return min(
        (_num(row.get("summary", {}).get(
            "source_layer_input", {}).get(name), 1.0)
         for row in runs),
        default=1.0)

  min_target_layer_input = min_metric("min_target_layer_input_cosine")
  min_target_ffn_input = min_metric("min_target_ffn_input_cosine")
  min_preceding_final = min_metric("min_preceding_final_output_cosine")
  min_preceding_attn_norm_from_input = min_metric(
      "min_preceding_attn_norm_from_gpu_input_cosine")
  min_preceding_gpu_attn_norm_vs_cpu = min_metric(
      "min_preceding_gpu_attn_norm_vs_cpu_cosine")
  target_layer_input_gap = (
      diagnostics_emitted and min_target_layer_input < COSINE_THRESHOLD)
  preceding_output_gap = (
      diagnostics_emitted and min_preceding_final < COSINE_THRESHOLD)
  preceding_norm_inherits_input = (
      diagnostics_emitted
      and min_preceding_attn_norm_from_input < COSINE_THRESHOLD)
  preceding_norm_math_ok = (
      diagnostics_emitted
      and min_preceding_gpu_attn_norm_vs_cpu >= COSINE_THRESHOLD)
  diagnostic_classification = (
      "source_layer_input_preceding_linear_output_gap"
      if (target_layer_input_gap and preceding_output_gap
          and preceding_norm_inherits_input and preceding_norm_math_ok) else
      "source_layer_input_preceding_linear_math_gap"
      if target_layer_input_gap and not preceding_norm_math_ok else
      "source_layer_input_gap_unclassified"
  )
  selected_next = (
      "router_prompt_full_attention_layer_input_preceding_linear_input_source_producer_linear_input_source_linear_input_source_layer_input_preceding_linear_output_gap_gate"
      if diagnostic_classification
      == "source_layer_input_preceding_linear_output_gap" else
      "router_prompt_full_attention_layer_input_preceding_linear_input_source_producer_linear_input_source_linear_input_source_layer_input_preceding_linear_math_gap_gate"
      if diagnostic_classification
      == "source_layer_input_preceding_linear_math_gap" else
      "router_prompt_full_attention_layer_input_preceding_linear_input_source_producer_linear_input_source_linear_input_source_layer_input_gap_gate"
  )
  checks = [
      {"name": "seq365_selected_source_layer_input_gap_gate",
       "pass": preconditions_pass},
      {"name": "source_layer_input_rows_emitted", "pass": rows_emitted},
      {"name": "source_and_consumer_counters_ready",
       "pass": counters_ready,
       "detail": [row.get("summary", {}) for row in runs]},
      {"name": "source_layer_input_diagnostics_emitted",
       "pass": diagnostics_emitted,
       "detail": [
           row.get("summary", {}).get("source_layer_input", {})
           for row in runs
       ]},
      {"name": "distribution_failure_reproduced",
       "pass": distribution_reproduced,
       "detail": [
           row.get("summary", {}).get("distribution", {}) for row in runs
       ]},
      {"name": "target_layer_input_gap_reproduced",
       "pass": target_layer_input_gap,
       "detail": {
           "min_target_layer_input_cosine": min_target_layer_input,
           "min_target_ffn_input_cosine": min_target_ffn_input,
       }},
      {"name": "preceding_linear_output_gap_reproduced",
       "pass": preceding_output_gap,
       "detail": {
           "min_preceding_final_output_cosine": min_preceding_final,
       }},
      {"name": "preceding_linear_norm_inherits_live_input",
       "pass": preceding_norm_inherits_input,
       "detail": {
           "min_preceding_attn_norm_from_gpu_input_cosine": (
               min_preceding_attn_norm_from_input),
       }},
      {"name": "preceding_linear_norm_math_matches_cpu",
       "pass": preceding_norm_math_ok,
       "detail": {
           "min_preceding_gpu_attn_norm_vs_cpu_cosine": (
               min_preceding_gpu_attn_norm_vs_cpu),
       }},
  ]
  required = all(bool(row.get("pass")) for row in checks)
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "inputs": {
          "routes": _rel(args.routes),
          "seq364": _rel(args.seq364),
          "seq365": _rel(args.seq365),
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
      "target_layer_input_gap": target_layer_input_gap,
      "preceding_output_gap": preceding_output_gap,
      "preceding_norm_inherits_input": preceding_norm_inherits_input,
      "preceding_norm_math_ok": preceding_norm_math_ok,
      "min_target_layer_input_cosine": min_target_layer_input,
      "min_target_ffn_input_cosine": min_target_ffn_input,
      "min_preceding_final_output_cosine": min_preceding_final,
      "min_preceding_attn_norm_from_gpu_input_cosine": (
          min_preceding_attn_norm_from_input),
      "min_preceding_gpu_attn_norm_vs_cpu_cosine": (
          min_preceding_gpu_attn_norm_vs_cpu),
      "speedup_claims_allowed": False,
      "disposition": (
          "accept_source_layer_input_gap_classification"
          if required else
          "block_source_layer_input_gap_classification"
      ),
      "selected_next_route": (
          selected_next if required else
          "router_prompt_full_attention_layer_input_preceding_linear_input_source_producer_linear_input_source_linear_input_source_layer_input_gap_gate"
      ),
      "next_route_reason": (
          "The source layer-input gap is inherited from preceding linear output "
          "drift; preceding linear norm math matches CPU on live input. Root "
          "the preceding linear output next."
          if required and diagnostic_classification
          == "source_layer_input_preceding_linear_output_gap" else
          "Preceding linear norm math does not match CPU on live input; root "
          "linear math next."
          if required and diagnostic_classification
          == "source_layer_input_preceding_linear_math_gap" else
          "Source layer-input evidence is incomplete; keep this gate open."
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
      "# Router Full-Attention Source Layer-Input Gap Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- diagnostic_classification: `{metrics['diagnostic_classification']}`",
      f"- min target layer-input cosine: `{metrics['min_target_layer_input_cosine']}`",
      f"- min preceding final-output cosine: `{metrics['min_preceding_final_output_cosine']}`",
      f"- min preceding norm-from-input cosine: `{metrics['min_preceding_attn_norm_from_gpu_input_cosine']}`",
      f"- min preceding norm-vs-CPU cosine: `{metrics['min_preceding_gpu_attn_norm_vs_cpu_cosine']}`",
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
  parser.add_argument("--seq364", type=Path, default=DEFAULT_SEQ364)
  parser.add_argument("--seq365", type=Path, default=DEFAULT_SEQ365)
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
