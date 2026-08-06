#!/usr/bin/env python3
"""Run the residual-product consumer distribution-fix root diagnostic."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
  sys.path.insert(0, str(TOOLS))

import iq36_local  # noqa: E402


SCHEMA_VERSION = (
    "intel-qwen36-router-full-attention-residual-product-consumer-"
    "distribution-fix-gate-v0"
)
DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_SEQ330 = (
    ROOT
    / "output/router-full-attention-residual-product-consumer-router-distribution-gate-20260708Tseq330Z"
    / "metrics.json"
)
DEFAULT_SEQ317 = (
    ROOT
    / "output/router-qkv-delta-product-consumer-distribution-fix-gate-20260708Tseq317Z"
    / "metrics.json"
)
DEFAULT_TOKEN_INPUT_DIR = (
    ROOT
    / "output/r2-gpu-acceptance-matrix-router-both-rowblock16-26mask-20260708Tseq222Z"
    / "token-input"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "output/router-full-attention-residual-product-consumer-distribution-fix-gate-20260708Tseq331Z"
)
DEFAULT_HOST = "local"
DEFAULT_MODEL = "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf"
DEFAULT_ENV_SCRIPT = "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh"
DEFAULT_REMOTE_ROOT = "/home/intel/intel-qwen36-gpu"

CASES = ["router_math_reason_001", "router_code_reason_002"]
DECODE_TOKENS = 8
HIDDEN_SIZE = 2048
FULL_ATTENTION_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31, 35]
EXPECTED_LAYER_EVENTS = len(FULL_ATTENTION_LAYERS) * DECODE_TOKENS
EXPECTED_VALUES = EXPECTED_LAYER_EVENTS * HIDDEN_SIZE
ROWBLOCK16_26MASK = (
    "0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,"
    "24,25,26,28,29,30,33,34,36,37,38"
)
KLD_THRESHOLD = 0.005
TOP1_THRESHOLD = 0.99
RESIDUAL_GAP_COSINE_THRESHOLD = 0.9999


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


def _smoke_from_stdout(run: dict[str, Any]) -> dict[str, Any]:
  stdout = str(run.get("stdout") or "").strip()
  if not stdout:
    return {}
  parsed = json.loads(stdout.splitlines()[0])
  return parsed if isinstance(parsed, dict) else {}


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
      "--diagnostic-layer-range", "3:36",
      "--diagnostic-token-limit", str(DECODE_TOKENS),
  ]
  env = [
      "IQ36_OPENCL_NO_QUEUE_PROFILING=1",
      "IQ36_ATTENTION_FRONT_OUTPUT_PROJECTION_ROWBLOCK16=1",
      f"IQ36_ATTENTION_FRONT_OUTPUT_PROJECTION_ROWBLOCK16_LAYERS={ROWBLOCK16_26MASK}",
      "IQ36_SELECTED_SHARED_Q4_GATEUP_COMBINED=1",
      "IQ36_SELECTED_SHARED_Q4_DOWN_COMBINED=1",
      "IQ36_SELECTED_SHARED_Q6_DOWN_COMBINED=1",
      "IQ36_DEFER_FFN_DOWN_FINISH_BUNDLE=1",
      "IQ36_FULL_ATTENTION_RESIDUAL_PRODUCT_SOURCE=1",
      "IQ36_FULL_ATTENTION_RESIDUAL_PRODUCT_CONSUMER_SOURCE=1",
  ]
  remote_script = " && ".join([
      f"source {shlex.quote(args.env_script)} >/dev/null 2>&1",
      " ".join([*env, shlex.quote(binary), *flags]),
  ])
  return iq36_local.run_target(
      args.host, f"bash -lc {shlex.quote(remote_script)}", args.timeout_s)


def _distribution_summary(smoke: dict[str, Any]) -> dict[str, Any]:
  dist = smoke.get("distribution_ladder")
  dist = dist if isinstance(dist, dict) else {}
  return {
      "position_count": dist.get("position_count"),
      "top1_rate": dist.get("top1_rate"),
      "top1_match_count": dist.get("top1_match_count"),
      "max_kld": dist.get("max_kld"),
      "mean_kld": dist.get("mean_kld"),
      "min_logits_cosine": dist.get("min_logits_cosine"),
      "required_checks_passed": dist.get("required_checks_passed"),
      "top1_pass": dist.get("top1_pass"),
      "kld_pass": dist.get("kld_pass"),
      "logits_cosine_pass": dist.get("logits_cosine_pass"),
  }


def _selected_residual_gap(smoke: dict[str, Any]) -> dict[str, Any]:
  steps = smoke.get("residual_source_diff_by_step")
  steps = steps if isinstance(steps, list) else []
  selected = set(FULL_ATTENTION_LAYERS)
  observations: list[dict[str, Any]] = []
  by_layer: dict[int, dict[str, Any]] = {
      layer: {
          "layer": layer,
          "observation_count": 0,
          "min_ffn_input_cosine": 1.0,
          "max_ffn_input_abs_diff": 0.0,
          "min_ffn_norm_cosine": 1.0,
          "max_ffn_norm_abs_diff": 0.0,
          "min_layer_input_cosine": 1.0,
          "max_layer_input_abs_diff": 0.0,
          "min_attention_output_cosine": 1.0,
          "max_attention_output_abs_diff": 0.0,
      }
      for layer in FULL_ATTENTION_LAYERS
  }
  first_gap: dict[str, Any] | None = None
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
      obs = {
          "token_index": token_index,
          "layer": layer,
          "ffn_input_cosine": row.get("ffn_input_cosine"),
          "ffn_input_max_abs_diff": row.get("ffn_input_max_abs_diff"),
          "ffn_input_rmse": row.get("ffn_input_rmse"),
          "gpu_ffn_norm_vs_cpu_cosine": row.get("gpu_ffn_norm_vs_cpu_cosine"),
          "gpu_ffn_norm_vs_cpu_max_abs_diff": row.get(
              "gpu_ffn_norm_vs_cpu_max_abs_diff"),
          "layer_input_cosine": row.get("layer_input_cosine"),
          "layer_input_max_abs_diff": row.get("layer_input_max_abs_diff"),
          "attention_output_cosine": row.get("attention_output_cosine"),
          "attention_output_max_abs_diff": row.get(
              "attention_output_max_abs_diff"),
          "cpu_ffn_from_gpu_input_cosine": row.get(
              "cpu_ffn_from_gpu_input_cosine"),
          "gpu_output_vs_cpu_ffn_cosine": row.get(
              "gpu_output_vs_cpu_ffn_cosine"),
      }
      observations.append(obs)
      layer_row = by_layer[layer]
      layer_row["observation_count"] += 1
      ffn_cos = _num(obs.get("ffn_input_cosine"), 1.0)
      ffn_abs = _num(obs.get("ffn_input_max_abs_diff"))
      norm_cos = _num(obs.get("gpu_ffn_norm_vs_cpu_cosine"), 1.0)
      norm_abs = _num(obs.get("gpu_ffn_norm_vs_cpu_max_abs_diff"))
      layer_input_cos = _num(obs.get("layer_input_cosine"), 1.0)
      layer_input_abs = _num(obs.get("layer_input_max_abs_diff"))
      attn_cos = _num(obs.get("attention_output_cosine"), 1.0)
      attn_abs = _num(obs.get("attention_output_max_abs_diff"))
      layer_row["min_ffn_input_cosine"] = min(
          layer_row["min_ffn_input_cosine"], ffn_cos)
      layer_row["max_ffn_input_abs_diff"] = max(
          layer_row["max_ffn_input_abs_diff"], ffn_abs)
      layer_row["min_ffn_norm_cosine"] = min(
          layer_row["min_ffn_norm_cosine"], norm_cos)
      layer_row["max_ffn_norm_abs_diff"] = max(
          layer_row["max_ffn_norm_abs_diff"], norm_abs)
      layer_row["min_layer_input_cosine"] = min(
          layer_row["min_layer_input_cosine"], layer_input_cos)
      layer_row["max_layer_input_abs_diff"] = max(
          layer_row["max_layer_input_abs_diff"], layer_input_abs)
      layer_row["min_attention_output_cosine"] = min(
          layer_row["min_attention_output_cosine"], attn_cos)
      layer_row["max_attention_output_abs_diff"] = max(
          layer_row["max_attention_output_abs_diff"], attn_abs)
      if first_gap is None and ffn_cos < RESIDUAL_GAP_COSINE_THRESHOLD:
        first_gap = obs

  min_ffn_cos = min(
      (_num(row.get("ffn_input_cosine"), 1.0) for row in observations),
      default=1.0)
  max_ffn_abs = max(
      (_num(row.get("ffn_input_max_abs_diff")) for row in observations),
      default=0.0)
  min_norm_cos = min(
      (_num(row.get("gpu_ffn_norm_vs_cpu_cosine"), 1.0)
       for row in observations),
      default=1.0)
  max_norm_abs = max(
      (_num(row.get("gpu_ffn_norm_vs_cpu_max_abs_diff"))
       for row in observations),
      default=0.0)
  min_layer_input_cos = min(
      (_num(row.get("layer_input_cosine"), 1.0) for row in observations),
      default=1.0)
  max_layer_input_abs = max(
      (_num(row.get("layer_input_max_abs_diff")) for row in observations),
      default=0.0)
  min_attention_cos = min(
      (_num(row.get("attention_output_cosine"), 1.0)
       for row in observations),
      default=1.0)
  max_attention_abs = max(
      (_num(row.get("attention_output_max_abs_diff")) for row in observations),
      default=0.0)
  return {
      "expected_observation_count": EXPECTED_LAYER_EVENTS,
      "observation_count": len(observations),
      "min_ffn_input_cosine": min_ffn_cos,
      "max_ffn_input_abs_diff": max_ffn_abs,
      "min_ffn_norm_cosine": min_norm_cos,
      "max_ffn_norm_abs_diff": max_norm_abs,
      "min_layer_input_cosine": min_layer_input_cos,
      "max_layer_input_abs_diff": max_layer_input_abs,
      "min_attention_output_cosine": min_attention_cos,
      "max_attention_output_abs_diff": max_attention_abs,
      "first_ffn_input_gap": first_gap,
      "full_attention_layers": FULL_ATTENTION_LAYERS,
      "by_layer": [by_layer[layer] for layer in FULL_ATTENTION_LAYERS],
  }


def _shadow_residual_context(seq317: dict[str, Any]) -> dict[str, Any]:
  rows: dict[str, Any] = {}
  for check in seq317.get("checks", []):
    if not isinstance(check, dict):
      continue
    if check.get("name") == "all_full_attention_residual_only_passes_math_and_code":
      rows["all_full_attention_residual_only"] = check.get("detail")
    elif check.get("name") == "layer35_only_is_not_enough_for_router_code":
      rows["layer35_only"] = check.get("detail")
  return rows


def _case_summary(case_id: str, run: dict[str, Any],
                  smoke: dict[str, Any]) -> dict[str, Any]:
  return {
      "case_id": case_id,
      "returncode": run.get("returncode"),
      "stdout_bytes": len(str(run.get("stdout") or "")),
      "stderr_bytes": len(str(run.get("stderr") or "")),
      "required_checks_passed": smoke.get("required_checks_passed"),
      "top1_match_count": smoke.get("top1_match_count"),
      "greedy_prefix_match_count": smoke.get("greedy_prefix_match_count"),
      "source_layers": smoke.get(
          "full_attention_residual_product_source_layers"),
      "source_values": smoke.get(
          "full_attention_residual_product_source_values"),
      "source_misses": smoke.get(
          "full_attention_residual_product_source_misses"),
      "source_ready": smoke.get(
          "full_attention_residual_product_source_ready"),
      "consumer_layers": smoke.get(
          "full_attention_residual_product_consumer_source_layers"),
      "consumer_values": smoke.get(
          "full_attention_residual_product_consumer_source_values"),
      "consumer_misses": smoke.get(
          "full_attention_residual_product_consumer_source_misses"),
      "consumer_ready": smoke.get(
          "full_attention_residual_product_consumer_source_ready"),
      "cpu_shadow_state_each_token_enabled": smoke.get(
          "cpu_shadow_state_each_token_enabled"),
      "cpu_shadow_ffn_input_each_token_enabled": smoke.get(
          "cpu_shadow_ffn_input_each_token_enabled"),
      "cpu_shadow_layer_input_layers": smoke.get("cpu_shadow_layer_input_layers"),
      "cpu_shadow_attention_output_layers": smoke.get(
          "cpu_shadow_attention_output_layers"),
      "full_attention_state_diff_enabled": smoke.get(
          "full_attention_state_diff_enabled"),
      "diagnostic_layer_start": smoke.get("diagnostic_layer_start"),
      "diagnostic_layer_end": smoke.get("diagnostic_layer_end"),
      "diagnostic_token_limit": smoke.get("diagnostic_token_limit"),
      "speedup_claims_allowed": smoke.get("speedup_claims_allowed"),
      "distribution": _distribution_summary(smoke),
      "selected_residual_gap": _selected_residual_gap(smoke),
  }


def _seq330_counter_rows(seq330: dict[str, Any]) -> list[dict[str, Any]]:
  rows = seq330.get("runs")
  rows = rows if isinstance(rows, list) else []
  return [
      row.get("summary", {}) for row in rows
      if isinstance(row, dict) and isinstance(row.get("summary"), dict)
  ]


def compute(args: argparse.Namespace) -> dict[str, Any]:
  routes = _load_json(args.routes)
  seq330 = _load_json(args.seq330)
  seq317 = _load_json(args.seq317)
  binary = seq330.get("inputs", {}).get("binary")
  if not isinstance(binary, str) or not binary:
    raise SystemExit("seq330 binary missing")
  token_cache = iq36_local.ensure_cached_tokens(
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

  def row_emitted(row: dict[str, Any]) -> bool:
    summary = row.get("summary", {})
    return (
        row.get("run", {}).get("returncode") in (0, 2)
        and summary.get("distribution", {}).get("position_count") == DECODE_TOKENS
    )

  def row_counters_ready(row: dict[str, Any]) -> bool:
    summary = row.get("summary", {})
    return (
        summary.get("source_layers") == EXPECTED_LAYER_EVENTS
        and summary.get("source_values") == EXPECTED_VALUES
        and summary.get("source_misses") == 0
        and summary.get("source_ready") is True
        and summary.get("consumer_layers") == EXPECTED_LAYER_EVENTS
        and summary.get("consumer_values") == EXPECTED_VALUES
        and summary.get("consumer_misses") == 0
        and summary.get("consumer_ready") is True
        and summary.get("cpu_shadow_state_each_token_enabled") is False
        and summary.get("cpu_shadow_ffn_input_each_token_enabled") is False
        and summary.get("cpu_shadow_layer_input_layers") == 0
        and summary.get("cpu_shadow_attention_output_layers") == 0
        and summary.get("full_attention_state_diff_enabled") is True
        and summary.get("speedup_claims_allowed") is False
    )

  def row_gap_emitted(row: dict[str, Any]) -> bool:
    gap = row.get("summary", {}).get("selected_residual_gap", {})
    return (
        gap.get("observation_count") == EXPECTED_LAYER_EVENTS
        and len(gap.get("by_layer", [])) == len(FULL_ATTENTION_LAYERS)
    )

  def row_distribution_failed(row: dict[str, Any]) -> bool:
    dist = row.get("summary", {}).get("distribution", {})
    return (
        _num(dist.get("max_kld")) > KLD_THRESHOLD
        or _num(dist.get("top1_rate"), 1.0) < TOP1_THRESHOLD
        or dist.get("required_checks_passed") is False
    )

  seq330_rows = _seq330_counter_rows(seq330)
  seq330_failed_distribution = (
      seq330.get("router_distribution_passed") is False
      and any(
          _num(row.get("distribution", {}).get("max_kld")) > KLD_THRESHOLD
          or _num(row.get("distribution", {}).get("top1_rate"), 1.0)
              < TOP1_THRESHOLD
          for row in seq330_rows
      )
  )
  seq330_counters_ready = all(
      row.get("source_layers") == EXPECTED_LAYER_EVENTS
      and row.get("source_values") == EXPECTED_VALUES
      and row.get("source_misses") == 0
      and row.get("source_ready") is True
      and row.get("consumer_layers") == EXPECTED_LAYER_EVENTS
      and row.get("consumer_values") == EXPECTED_VALUES
      and row.get("consumer_misses") == 0
      and row.get("consumer_ready") is True
      for row in seq330_rows
  )
  preconditions_pass = (
      seq330.get("required_checks_passed") is True
      and seq330.get("selected_next_route")
      == "router_prompt_full_attention_residual_product_consumer_distribution_fix_gate"
      and seq330_failed_distribution
      and seq330_counters_ready
      and _has_candidate(
          routes, 330, "reject_residual_product_consumer_router_distribution")
      and _has_switch(
          routes,
          "select_router_prompt_full_attention_residual_product_consumer_distribution_fix_gate",
          330)
  )
  rows_emitted = len(runs) == len(CASES) and all(row_emitted(row) for row in runs)
  counters_ready = rows_emitted and all(row_counters_ready(row) for row in runs)
  gaps_emitted = counters_ready and all(row_gap_emitted(row) for row in runs)
  distribution_still_fails = rows_emitted and all(
      row_distribution_failed(row) for row in runs)
  min_residual_cosine = min(
      (_num(
          row.get("summary", {})
          .get("selected_residual_gap", {})
          .get("min_ffn_input_cosine"), 1.0)
       for row in runs),
      default=1.0)
  max_residual_abs_diff = max(
      (_num(
          row.get("summary", {})
          .get("selected_residual_gap", {})
          .get("max_ffn_input_abs_diff"))
       for row in runs),
      default=0.0)
  value_gap_observed = (
      gaps_emitted and min_residual_cosine < RESIDUAL_GAP_COSINE_THRESHOLD)

  checks = [
      {"name": "seq330_selected_distribution_fix_gate",
       "pass": preconditions_pass},
      {"name": "diagnostic_distribution_rows_emitted",
       "pass": rows_emitted},
      {"name": "product_consumer_counters_ready_during_diagnostic",
       "pass": counters_ready,
       "detail": [row.get("summary", {}) for row in runs]},
      {"name": "selected_full_attention_residual_gap_emitted",
       "pass": gaps_emitted,
       "detail": [
           row.get("summary", {}).get("selected_residual_gap", {})
           for row in runs
       ]},
      {"name": "diagnostic_reproduces_distribution_failure",
       "pass": distribution_still_fails,
       "detail": [
           row.get("summary", {}).get("distribution", {}) for row in runs
       ]},
  ]
  required = all(bool(row.get("pass")) for row in checks)
  selected_next = (
      "router_prompt_full_attention_residual_value_gap_diagnostic_gate"
      if value_gap_observed else
      "router_prompt_full_attention_residual_tail_math_diagnostic_gate"
  )
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "inputs": {
          "routes": _rel(args.routes),
          "seq330": _rel(args.seq330),
          "seq317": _rel(args.seq317),
          "token_input_dir": _rel(args.token_input_dir),
          "host": args.host,
          "binary": binary,
          "rowblock16_layers": ROWBLOCK16_26MASK,
          "full_attention_layers": FULL_ATTENTION_LAYERS,
      },
      "token_cache": {
          "ok": token_cache.get("ok"),
          "hit": token_cache.get("hit"),
          "key": token_cache.get("key"),
          "dir": token_cache.get("dir"),
      },
      "seq330": {
          "failed_distribution": seq330_failed_distribution,
          "ready_counters": seq330_counters_ready,
          "runs": seq330_rows,
      },
      "cpu_shadow_residual_only_context": _shadow_residual_context(seq317),
      "runs": runs,
      "checks": checks,
      "required_checks_passed": required,
      "value_gap_observed": value_gap_observed,
      "min_selected_residual_cosine": min_residual_cosine,
      "max_selected_residual_abs_diff": max_residual_abs_diff,
      "speedup_claims_allowed": False,
      "disposition": (
          "accept_residual_product_consumer_distribution_fix_root_diagnostic"
          if required else
          "block_residual_product_consumer_distribution_fix_root_diagnostic"
      ),
      "selected_next_route": (
          selected_next if required else
          "router_prompt_full_attention_residual_product_consumer_distribution_fix_gate"
      ),
      "next_route_reason": (
          "The resident residual source/consumer remains counter-clean while "
          "router distribution still fails. The diagnostic captured the "
          "native-vs-live selected full-attention FFN residual value gap; the "
          "next unit should isolate whether the product source is exporting the "
          "wrong residual value or the resident-input tail consumes it with "
          "different math."
          if required and value_gap_observed else
          "The residual source/consumer failure reproduced, but the selected "
          "residual value gap did not cross the diagnostic cosine threshold; "
          "inspect resident-tail math before another product-source route."
          if required else
          "Distribution-fix root diagnostic evidence is incomplete; keep the "
          "current gate open."
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
      "# Router Full-Attention Residual Product Consumer Distribution Fix Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- value_gap_observed: `{str(metrics['value_gap_observed']).lower()}`",
      f"- min selected residual cosine: `{metrics['min_selected_residual_cosine']}`",
      f"- max selected residual abs diff: `{metrics['max_selected_residual_abs_diff']}`",
      f"- failed_checks: `{failed}`",
      f"- speedup_claims_allowed: `{str(metrics['speedup_claims_allowed']).lower()}`",
      "",
  ]
  for row in metrics["runs"]:
    summary = row["summary"]
    dist = summary["distribution"]
    gap = summary["selected_residual_gap"]
    lines.extend([
        f"## {summary['case_id']}",
        "",
        f"- distribution max KLD: `{dist.get('max_kld')}`",
        f"- distribution top1 rate: `{dist.get('top1_rate')}`",
        f"- source counters: `{summary.get('source_layers')}` / `{summary.get('source_values')}` / `{summary.get('source_misses')}`",
        f"- consumer counters: `{summary.get('consumer_layers')}` / `{summary.get('consumer_values')}` / `{summary.get('consumer_misses')}`",
        f"- selected residual observations: `{gap.get('observation_count')}`",
        f"- min selected residual cosine: `{gap.get('min_ffn_input_cosine')}`",
        f"- max selected residual abs diff: `{gap.get('max_ffn_input_abs_diff')}`",
        f"- min selected layer-input cosine: `{gap.get('min_layer_input_cosine')}`",
        f"- max selected layer-input abs diff: `{gap.get('max_layer_input_abs_diff')}`",
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
  parser.add_argument("--seq330", type=Path, default=DEFAULT_SEQ330)
  parser.add_argument("--seq317", type=Path, default=DEFAULT_SEQ317)
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
      "disposition": metrics["disposition"],
      "max_selected_residual_abs_diff": metrics["max_selected_residual_abs_diff"],
      "min_selected_residual_cosine": metrics["min_selected_residual_cosine"],
      "out_dir": _rel(args.out_dir),
      "required_checks_passed": metrics["required_checks_passed"],
      "selected_next_route": metrics["selected_next_route"],
      "value_gap_observed": metrics["value_gap_observed"],
  }, sort_keys=True))
  return 0 if metrics["required_checks_passed"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
