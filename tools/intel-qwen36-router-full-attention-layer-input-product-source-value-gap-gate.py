#!/usr/bin/env python3
"""Classify the selected layer-input product source value gap."""

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
    / "tools/intel-qwen36-router-full-attention-layer-input-product-"
    "consumer-distribution-fix-gate.py"
)

SCHEMA_VERSION = (
    "intel-qwen36-router-full-attention-layer-input-product-source-"
    "value-gap-gate-v0"
)
DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_SEQ349 = (
    ROOT
    / "output/router-full-attention-layer-input-product-consumer-distribution-fix-gate-20260708Tseq349Z"
    / "metrics.json"
)
DEFAULT_TOKEN_INPUT_DIR = (
    ROOT
    / "output/r2-gpu-acceptance-matrix-router-both-rowblock16-26mask-20260708Tseq222Z"
    / "token-input"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "output/router-full-attention-layer-input-product-source-value-gap-gate-20260708Tseq350Z"
)
DEFAULT_HOST = "local"
DEFAULT_MODEL = "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf"
DEFAULT_ENV_SCRIPT = "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh"
DEFAULT_REMOTE_ROOT = "/home/intel/intel-qwen36-gpu"

CASES = ["router_math_reason_001", "router_code_reason_002"]
COSINE_THRESHOLD = 0.9999
KLD_THRESHOLD = 0.005
TOP1_THRESHOLD = 0.99


def _load_base() -> Any:
  spec = importlib.util.spec_from_file_location("iq36_layer_input_gap_base", BASE_GATE)
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


def _distribution_summary(smoke: dict[str, Any]) -> dict[str, Any]:
  return BASE._distribution_summary(smoke)


def _first_gap(current: dict[str, Any] | None,
               token_index: Any,
               layer: int,
               cosine: float,
               max_abs: float,
               prefix: str) -> dict[str, Any] | None:
  if current is not None or cosine >= COSINE_THRESHOLD:
    return current
  return {
      "token_index": token_index,
      "layer": layer,
      f"{prefix}_cosine": cosine,
      f"{prefix}_max_abs_diff": max_abs,
  }


def _source_gap_summary(smoke: dict[str, Any]) -> dict[str, Any]:
  selected = set(BASE.FULL_ATTENTION_LAYERS)
  residual_steps = smoke.get("residual_source_diff_by_step")
  residual_steps = residual_steps if isinstance(residual_steps, list) else []
  full_steps = smoke.get("full_attention_source_diff_by_step")
  full_steps = full_steps if isinstance(full_steps, list) else []

  layer_obs = 0
  min_layer_input = 1.0
  max_layer_input_abs = 0.0
  min_ffn_input = 1.0
  max_ffn_input_abs = 0.0
  first_layer_input_gap: dict[str, Any] | None = None
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
      if not isinstance(layer, int) or layer not in selected:
        continue
      if row.get("layer_input_available") is not True:
        continue
      layer_obs += 1
      layer_cos = _num(row.get("layer_input_cosine"), 1.0)
      layer_abs = _num(row.get("layer_input_max_abs_diff"))
      ffn_cos = _num(row.get("ffn_input_cosine"), 1.0)
      ffn_abs = _num(row.get("ffn_input_max_abs_diff"))
      min_layer_input = min(min_layer_input, layer_cos)
      max_layer_input_abs = max(max_layer_input_abs, layer_abs)
      min_ffn_input = min(min_ffn_input, ffn_cos)
      max_ffn_input_abs = max(max_ffn_input_abs, ffn_abs)
      first_layer_input_gap = _first_gap(
          first_layer_input_gap, token_index, layer, layer_cos, layer_abs,
          "layer_input")

  full_obs = 0
  min_attn_norm_from_gpu_input = 1.0
  max_attn_norm_from_gpu_input_abs = 0.0
  min_gpu_attn_norm_vs_cpu = 1.0
  max_gpu_attn_norm_vs_cpu_abs = 0.0
  min_q_full_from_gpu_norm = 1.0
  min_gpu_q_full_vs_cpu = 1.0
  first_attn_norm_input_gap: dict[str, Any] | None = None
  first_gpu_norm_math_gap: dict[str, Any] | None = None
  for step in full_steps:
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
      if row.get("attn_norm_from_gpu_input_available") is not True:
        continue
      full_obs += 1
      input_cos = _num(row.get("attn_norm_from_gpu_input_cosine"), 1.0)
      input_abs = _num(row.get("attn_norm_from_gpu_input_max_abs_diff"))
      gpu_norm_cos = _num(row.get("gpu_attn_norm_vs_cpu_cosine"), 1.0)
      gpu_norm_abs = _num(row.get("gpu_attn_norm_vs_cpu_max_abs_diff"))
      q_full_input_cos = _num(row.get("q_full_from_gpu_norm_cosine"), 1.0)
      gpu_q_cos = _num(row.get("gpu_q_full_vs_cpu_cosine"), 1.0)
      min_attn_norm_from_gpu_input = min(
          min_attn_norm_from_gpu_input, input_cos)
      max_attn_norm_from_gpu_input_abs = max(
          max_attn_norm_from_gpu_input_abs, input_abs)
      min_gpu_attn_norm_vs_cpu = min(min_gpu_attn_norm_vs_cpu, gpu_norm_cos)
      max_gpu_attn_norm_vs_cpu_abs = max(
          max_gpu_attn_norm_vs_cpu_abs, gpu_norm_abs)
      min_q_full_from_gpu_norm = min(min_q_full_from_gpu_norm, q_full_input_cos)
      min_gpu_q_full_vs_cpu = min(min_gpu_q_full_vs_cpu, gpu_q_cos)
      first_attn_norm_input_gap = _first_gap(
          first_attn_norm_input_gap, token_index, layer, input_cos, input_abs,
          "attn_norm_from_gpu_input")
      first_gpu_norm_math_gap = _first_gap(
          first_gpu_norm_math_gap, token_index, layer, gpu_norm_cos,
          gpu_norm_abs, "gpu_attn_norm_vs_cpu")

  return {
      "selected_layers": BASE.FULL_ATTENTION_LAYERS,
      "expected_observation_count": BASE.EXPECTED_LAYER_EVENTS,
      "layer_input_observation_count": layer_obs,
      "full_attention_source_observation_count": full_obs,
      "min_layer_input_cosine": min_layer_input,
      "max_layer_input_abs_diff": max_layer_input_abs,
      "min_ffn_input_cosine": min_ffn_input,
      "max_ffn_input_abs_diff": max_ffn_input_abs,
      "min_attn_norm_from_gpu_input_cosine": min_attn_norm_from_gpu_input,
      "max_attn_norm_from_gpu_input_abs_diff": (
          max_attn_norm_from_gpu_input_abs),
      "min_gpu_attn_norm_vs_cpu_cosine": min_gpu_attn_norm_vs_cpu,
      "max_gpu_attn_norm_vs_cpu_abs_diff": max_gpu_attn_norm_vs_cpu_abs,
      "min_q_full_from_gpu_norm_cosine": min_q_full_from_gpu_norm,
      "min_gpu_q_full_vs_cpu_cosine": min_gpu_q_full_vs_cpu,
      "first_layer_input_gap": first_layer_input_gap,
      "first_attn_norm_from_gpu_input_gap": first_attn_norm_input_gap,
      "first_gpu_attn_norm_vs_cpu_gap": first_gpu_norm_math_gap,
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
      "distribution": _distribution_summary(smoke),
      "source_gap": _source_gap_summary(smoke),
  }


def compute(args: argparse.Namespace) -> dict[str, Any]:
  routes = _load_json(args.routes)
  seq349 = _load_json(args.seq349)
  binary = seq349.get("inputs", {}).get("binary")
  if not isinstance(binary, str) or not binary:
    raise SystemExit("seq349 binary missing")
  token_cache = BASE.iq36_local.ensure_cached_tokens(
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

  def row_counters_ready(row: dict[str, Any]) -> bool:
    summary = row.get("summary", {})
    return (
        row.get("run", {}).get("returncode") in (0, 2)
        and summary.get("source_layers") == BASE.EXPECTED_LAYER_EVENTS
        and summary.get("source_values") == BASE.EXPECTED_VALUES
        and summary.get("source_misses") == 0
        and summary.get("source_ready") is True
        and summary.get("consumer_layers") == BASE.EXPECTED_LAYER_EVENTS
        and summary.get("consumer_values") == BASE.EXPECTED_VALUES
        and summary.get("consumer_misses") == 0
        and summary.get("consumer_ready") is True
        and summary.get("cpu_shadow_layer_input_layers") == 0
        and summary.get("cpu_shadow_attention_output_layers") == 0
        and summary.get("speedup_claims_allowed") is False
    )

  preconditions_pass = (
      seq349.get("required_checks_passed") is True
      and seq349.get("selected_next_route")
      == "router_prompt_full_attention_layer_input_product_source_value_gap_gate"
      and seq349.get("layer_input_gap_observed") is True
      and _has_candidate(
          routes, 349,
          "accept_layer_input_product_consumer_distribution_fix_root_diagnostic")
      and _has_switch(
          routes,
          "select_router_prompt_full_attention_layer_input_product_source_value_gap_gate",
          349)
  )
  rows_emitted = (
      len(runs) == len(CASES)
      and all(row.get("run", {}).get("returncode") in (0, 2) for row in runs)
  )
  counters_ready = rows_emitted and all(row_counters_ready(row) for row in runs)
  distribution_reproduced = rows_emitted and all(
      _dist_fail(row.get("summary", {}).get("distribution", {}))
      for row in runs)
  source_gaps_emitted = counters_ready and all(
      row.get("summary", {}).get("source_gap", {})
      .get("layer_input_observation_count") == BASE.EXPECTED_LAYER_EVENTS
      and row.get("summary", {}).get("source_gap", {})
      .get("full_attention_source_observation_count")
      == BASE.EXPECTED_LAYER_EVENTS
      for row in runs)
  min_layer_input_cosine = min(
      (_num(row.get("summary", {}).get("source_gap", {})
            .get("min_layer_input_cosine"), 1.0)
       for row in runs),
      default=1.0)
  max_layer_input_abs_diff = max(
      (_num(row.get("summary", {}).get("source_gap", {})
            .get("max_layer_input_abs_diff"))
       for row in runs),
      default=0.0)
  min_attn_norm_from_gpu_input = min(
      (_num(row.get("summary", {}).get("source_gap", {})
            .get("min_attn_norm_from_gpu_input_cosine"), 1.0)
       for row in runs),
      default=1.0)
  min_gpu_attn_norm_vs_cpu = min(
      (_num(row.get("summary", {}).get("source_gap", {})
            .get("min_gpu_attn_norm_vs_cpu_cosine"), 1.0)
       for row in runs),
      default=1.0)
  source_value_gap = (
      source_gaps_emitted and min_layer_input_cosine < COSINE_THRESHOLD)
  consumer_math_ok = (
      source_gaps_emitted and min_gpu_attn_norm_vs_cpu >= COSINE_THRESHOLD)
  diagnostic_classification = (
      "upstream_layer_output_value_gap"
      if source_value_gap and consumer_math_ok else
      "layer_input_consumer_math_gap"
      if min_gpu_attn_norm_vs_cpu < COSINE_THRESHOLD else
      "source_value_gap_not_reproduced"
  )
  selected_next = (
      "router_prompt_full_attention_layer_input_upstream_layer_output_value_gap_gate"
      if diagnostic_classification == "upstream_layer_output_value_gap" else
      "router_prompt_full_attention_layer_input_product_consumer_math_gap_gate"
      if diagnostic_classification == "layer_input_consumer_math_gap" else
      "router_prompt_full_attention_layer_input_product_source_value_gap_gate"
  )
  checks = [
      {"name": "seq349_selected_source_value_gap_gate",
       "pass": preconditions_pass},
      {"name": "source_value_gap_rows_emitted", "pass": rows_emitted},
      {"name": "source_and_consumer_counters_ready",
       "pass": counters_ready,
       "detail": [row.get("summary", {}) for row in runs]},
      {"name": "source_gap_diagnostics_emitted",
       "pass": source_gaps_emitted,
       "detail": [
           row.get("summary", {}).get("source_gap", {}) for row in runs
       ]},
      {"name": "distribution_failure_reproduced",
       "pass": distribution_reproduced,
       "detail": [
           row.get("summary", {}).get("distribution", {}) for row in runs
       ]},
      {"name": "consumer_math_not_the_gap",
       "pass": consumer_math_ok,
       "detail": {
           "min_gpu_attn_norm_vs_cpu_cosine": min_gpu_attn_norm_vs_cpu,
           "threshold": COSINE_THRESHOLD,
       }},
  ]
  required = all(bool(row.get("pass")) for row in checks)
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "inputs": {
          "routes": _rel(args.routes),
          "seq349": _rel(args.seq349),
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
      "source_value_gap": source_value_gap,
      "consumer_math_ok": consumer_math_ok,
      "min_selected_layer_input_cosine": min_layer_input_cosine,
      "max_selected_layer_input_abs_diff": max_layer_input_abs_diff,
      "min_attn_norm_from_gpu_input_cosine": min_attn_norm_from_gpu_input,
      "min_gpu_attn_norm_vs_cpu_cosine": min_gpu_attn_norm_vs_cpu,
      "speedup_claims_allowed": False,
      "disposition": (
          "accept_layer_input_product_source_value_gap_classification"
          if required else
          "block_layer_input_product_source_value_gap_classification"
      ),
      "selected_next_route": (
          selected_next if required else
          "router_prompt_full_attention_layer_input_product_source_value_gap_gate"
      ),
      "next_route_reason": (
          "The selected layer-input product source faithfully feeds the live "
          "device path, and the resident RMSNorm consumer matches CPU math on "
          "that live input. The distribution gap is upstream layer-output value "
          "drift before the selected full-attention layer input; root that "
          "producer next."
          if required else
          "Source value-gap evidence is incomplete; keep this gate open."
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
      "# Router Full-Attention Layer-Input Product Source Value-Gap Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- diagnostic_classification: `{metrics['diagnostic_classification']}`",
      f"- source_value_gap: `{str(metrics['source_value_gap']).lower()}`",
      f"- consumer_math_ok: `{str(metrics['consumer_math_ok']).lower()}`",
      f"- min selected layer-input cosine: `{metrics['min_selected_layer_input_cosine']}`",
      f"- max selected layer-input abs diff: `{metrics['max_selected_layer_input_abs_diff']}`",
      f"- min GPU RMSNorm-vs-CPU cosine on live input: `{metrics['min_gpu_attn_norm_vs_cpu_cosine']}`",
      f"- failed_checks: `{failed}`",
      f"- speedup_claims_allowed: `{str(metrics['speedup_claims_allowed']).lower()}`",
      "",
  ]
  for row in metrics["runs"]:
    summary = row["summary"]
    dist = summary["distribution"]
    gap = summary["source_gap"]
    lines.extend([
        f"## {summary['case_id']}",
        "",
        f"- distribution max KLD: `{dist.get('max_kld')}`",
        f"- distribution top1 rate: `{dist.get('top1_rate')}`",
        f"- source counters: `{summary.get('source_layers')}` / `{summary.get('source_values')}` / `{summary.get('source_misses')}`",
        f"- consumer counters: `{summary.get('consumer_layers')}` / `{summary.get('consumer_values')}` / `{summary.get('consumer_misses')}`",
        f"- min layer-input cosine: `{gap.get('min_layer_input_cosine')}`",
        f"- max layer-input abs diff: `{gap.get('max_layer_input_abs_diff')}`",
        f"- min attn-norm-from-live-input cosine: `{gap.get('min_attn_norm_from_gpu_input_cosine')}`",
        f"- min GPU RMSNorm-vs-CPU cosine: `{gap.get('min_gpu_attn_norm_vs_cpu_cosine')}`",
        f"- first layer-input gap: `{gap.get('first_layer_input_gap')}`",
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
  parser.add_argument("--seq349", type=Path, default=DEFAULT_SEQ349)
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
      "consumer_math_ok": metrics["consumer_math_ok"],
      "diagnostic_classification": metrics["diagnostic_classification"],
      "disposition": metrics["disposition"],
      "out_dir": _rel(args.out_dir),
      "required_checks_passed": metrics["required_checks_passed"],
      "selected_next_route": metrics["selected_next_route"],
      "source_value_gap": metrics["source_value_gap"],
  }, sort_keys=True))
  return 0 if metrics["required_checks_passed"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
