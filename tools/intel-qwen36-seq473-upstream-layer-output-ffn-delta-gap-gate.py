#!/usr/bin/env python3
"""Classify seq472 previous FFN-delta drift."""

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
SEQ472_GATE = ROOT / "tools/intel-qwen36-seq472-upstream-layer-output-gap-gate.py"
SCHEMA_VERSION = "intel-qwen36-seq473-upstream-layer-output-ffn-delta-gap-gate-v0"

DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_SEQ472 = (
    ROOT
    / "output/seq472-upstream-layer-output-gap-gate-20260709Tseq472Z"
    / "metrics.json"
)
DEFAULT_TOKEN_INPUT_DIR = (
    ROOT
    / "output/r2-gpu-acceptance-matrix-router-both-rowblock16-26mask-20260708Tseq222Z"
    / "token-input"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "output/seq473-upstream-layer-output-ffn-delta-gap-gate-20260709Tseq473Z"
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


def _load_seq472() -> Any:
  spec = importlib.util.spec_from_file_location("iq36_seq472_gate", SEQ472_GATE)
  if spec is None or spec.loader is None:
    raise SystemExit(f"cannot load seq472 gate: {SEQ472_GATE}")
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


SEQ472 = _load_seq472()
CURRENT_ROUTE = SEQ472.FFN_DELTA_ROUTE
FFN_NORM_ROUTE = CURRENT_ROUTE.replace(
    "_ffn_delta_gap_gate", "_ffn_delta_ffn_norm_gap_gate")
ROUTER_ROUTE = CURRENT_ROUTE.replace(
    "_ffn_delta_gap_gate", "_ffn_delta_router_gap_gate")
SELECTED_ROUTE = CURRENT_ROUTE.replace(
    "_ffn_delta_gap_gate", "_ffn_delta_selected_gap_gate")
SHARED_ROUTE = CURRENT_ROUTE.replace(
    "_ffn_delta_gap_gate", "_ffn_delta_shared_gap_gate")
DIAG_PREFIX = SEQ472.DIAG_PREFIX
DISPOSITION_PREFIX = SEQ472.DISPOSITION_PREFIX

COMPONENT_NAMES = [
    "ffn_input",
    "ffn_norm",
    "router_logits",
    "router_weights",
    "selected_gate_up",
    "selected_swiglu",
    "selected_down",
    "weighted_selected_down",
    "moe_out",
    "shared_gate",
    "shared_gate_sigmoid",
    "shared_gate_up",
    "shared_swiglu",
    "shared_down",
    "shared_gated",
    "ffn_out",
    "residual",
]


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


def _component_summary(smoke: dict[str, Any]) -> dict[str, Any]:
  steps = smoke.get("ffn_component_source_diff_by_step")
  steps = steps if isinstance(steps, list) else []
  mins = {name: 1.0 for name in COMPONENT_NAMES}
  max_abs = {name: 0.0 for name in COMPONENT_NAMES}
  first = {name: None for name in COMPONENT_NAMES}
  obs = 0
  router_id_mismatch_count = 0
  max_router_weight_abs_diff = 0.0
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
      if layer not in PREVIOUS_LAYERS:
        continue
      if row.get("ffn_input_available") is True:
        obs += 1
      if row.get("router_ids_match") is False:
        router_id_mismatch_count += 1
      max_router_weight_abs_diff = max(
          max_router_weight_abs_diff,
          _num(row.get("router_weight_max_abs_diff")))
      for name in COMPONENT_NAMES:
        if row.get(f"{name}_available") is not True:
          continue
        cosine = _num(row.get(f"{name}_cosine"), 1.0)
        diff = _num(row.get(f"{name}_max_abs_diff"))
        mins[name] = min(mins[name], cosine)
        max_abs[name] = max(max_abs[name], diff)
        first[name] = _first_gap(first[name], token_index, layer, name, cosine,
                                 diff)
  summary: dict[str, Any] = {
      "previous_layers": PREVIOUS_LAYERS,
      "expected_observation_count": EXPECTED_EVENTS,
      "observation_count": obs,
      "router_id_mismatch_count": router_id_mismatch_count,
      "max_router_weight_abs_diff": max_router_weight_abs_diff,
  }
  for name in COMPONENT_NAMES:
    summary[f"min_{name}_cosine"] = mins[name]
    summary[f"max_{name}_abs_diff"] = max_abs[name]
    summary[f"first_{name}_gap"] = first[name]
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
      "distribution": SEQ472.SEQ471.BASE.OLD.BASE._distribution_summary(smoke),
      "component": _component_summary(smoke),
  }


def compute(args: argparse.Namespace) -> dict[str, Any]:
  routes = _load_json(args.routes)
  seq472 = _load_json(args.seq472)
  binary = seq472.get("inputs", {}).get("binary")
  if not isinstance(binary, str) or not binary:
    raise SystemExit("seq472 binary missing")
  token_cache = SEQ472.SEQ471.BASE.OLD.BASE.BASE.BASE.iq36_local.ensure_cached_tokens(
      args.host, f"{args.remote_root}/cache", args.token_input_dir,
      args.timeout_s)

  runs: list[dict[str, Any]] = []
  if token_cache.get("ok") is True:
    for case_id in SEQ472.SEQ471.BASE.CASES:
      run = SEQ472.SEQ471.BASE.OLD.BASE._run_case(
          args, binary, str(token_cache.get("dir")), case_id)
      smoke = SEQ472.SEQ471.BASE.OLD.BASE._smoke_from_stdout(run)
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
      seq472.get("required_checks_passed") is True
      and seq472.get("selected_next_route") == CURRENT_ROUTE
      and seq472.get("diagnostic_classification")
      == f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_gap"
      and _has_candidate(routes, 472, str(seq472.get("disposition")))
      and _has_switch(routes, f"select_{CURRENT_ROUTE}", 472)
  )
  rows_emitted = (
      len(runs) == len(SEQ472.SEQ471.BASE.CASES)
      and all(row.get("run", {}).get("returncode") in (0, 2)
              for row in runs)
  )
  counters_ready = rows_emitted and all(row_ready(row) for row in runs)
  distribution_reproduced = rows_emitted and all(
      SEQ472._dist_fail(row.get("summary", {}).get("distribution", {}))
      for row in runs)
  component_emitted = counters_ready and all(
      row.get("summary", {}).get("component", {})
      .get("observation_count") == EXPECTED_EVENTS
      for row in runs)

  def min_component(name: str) -> float:
    return min(
        (_num(row.get("summary", {}).get("component", {}).get(name), 1.0)
         for row in runs),
        default=1.0)

  min_ffn_input = min_component("min_ffn_input_cosine")
  min_ffn_norm = min_component("min_ffn_norm_cosine")
  min_router_logits = min_component("min_router_logits_cosine")
  min_router_weights = min_component("min_router_weights_cosine")
  min_selected_gate_up = min_component("min_selected_gate_up_cosine")
  min_selected_swiglu = min_component("min_selected_swiglu_cosine")
  min_selected_down = min_component("min_selected_down_cosine")
  min_weighted_selected_down = min_component(
      "min_weighted_selected_down_cosine")
  min_moe_out = min_component("min_moe_out_cosine")
  min_shared_gate = min_component("min_shared_gate_cosine")
  min_shared_swiglu = min_component("min_shared_swiglu_cosine")
  min_shared_down = min_component("min_shared_down_cosine")
  min_shared_gated = min_component("min_shared_gated_cosine")
  min_ffn_out = min_component("min_ffn_out_cosine")
  min_residual = min_component("min_residual_cosine")
  router_id_mismatch_count = sum(
      int(row.get("summary", {}).get("component", {})
          .get("router_id_mismatch_count", 0))
      for row in runs)

  ffn_delta_gap = component_emitted and min_ffn_out < COSINE_THRESHOLD
  residual_gap = component_emitted and min_residual < COSINE_THRESHOLD
  ffn_input_clean = component_emitted and min_ffn_input >= COSINE_THRESHOLD
  ffn_norm_gap = component_emitted and min_ffn_norm < COSINE_THRESHOLD
  router_gap = (
      component_emitted
      and (router_id_mismatch_count > 0
           or min_router_logits < COSINE_THRESHOLD
           or min_router_weights < COSINE_THRESHOLD))
  selected_gap = (
      component_emitted
      and min(min_selected_gate_up, min_selected_swiglu, min_selected_down,
              min_weighted_selected_down, min_moe_out) < COSINE_THRESHOLD)
  shared_gap = (
      component_emitted
      and min(min_shared_gate, min_shared_swiglu, min_shared_down,
              min_shared_gated) < COSINE_THRESHOLD)
  diagnostic_classification = (
      f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_gap"
      if ffn_delta_gap and residual_gap and ffn_input_clean and ffn_norm_gap else
      f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_router_gap"
      if ffn_delta_gap and router_gap else
      f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_selected_gap"
      if ffn_delta_gap and selected_gap else
      f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_shared_gap"
      if ffn_delta_gap and shared_gap else
      f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_gap_unclassified"
  )
  selected_next = (
      FFN_NORM_ROUTE
      if diagnostic_classification
      == f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_gap"
      else ROUTER_ROUTE
      if diagnostic_classification
      == f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_router_gap"
      else SELECTED_ROUTE
      if diagnostic_classification
      == f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_selected_gap"
      else SHARED_ROUTE
      if diagnostic_classification
      == f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_shared_gap"
      else CURRENT_ROUTE
  )
  checks = [
      {"name": "seq472_selected_ffn_delta_gap_gate", "pass": preconditions_pass},
      {"name": "ffn_delta_rows_emitted", "pass": rows_emitted},
      {"name": "source_and_consumer_counters_ready",
       "pass": counters_ready,
       "detail": [row.get("summary", {}) for row in runs]},
      {"name": "distribution_failure_reproduced",
       "pass": distribution_reproduced,
       "detail": [
           row.get("summary", {}).get("distribution", {}) for row in runs
       ]},
      {"name": "ffn_component_diagnostics_emitted",
       "pass": component_emitted,
       "detail": [
           row.get("summary", {}).get("component", {}) for row in runs
       ]},
      {"name": "ffn_delta_gap_reproduced",
       "pass": ffn_delta_gap and residual_gap,
       "detail": {
           "min_ffn_out_cosine": min_ffn_out,
           "min_residual_cosine": min_residual,
       }},
      {"name": "ffn_input_clean_for_norm_split",
       "pass": ffn_input_clean,
       "detail": {"min_ffn_input_cosine": min_ffn_input}},
      {"name": "ffn_norm_gap_reproduced",
       "pass": ffn_norm_gap,
       "detail": {"min_ffn_norm_cosine": min_ffn_norm}},
  ]
  required = all(bool(row.get("pass")) for row in checks)
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "inputs": {
          "routes": _rel(args.routes),
          "seq472": _rel(args.seq472),
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
      "ffn_delta_gap": ffn_delta_gap,
      "residual_gap": residual_gap,
      "ffn_input_clean": ffn_input_clean,
      "ffn_norm_gap": ffn_norm_gap,
      "router_gap": router_gap,
      "selected_gap": selected_gap,
      "shared_gap": shared_gap,
      "router_id_mismatch_count": router_id_mismatch_count,
      "min_ffn_input_cosine": min_ffn_input,
      "min_ffn_norm_cosine": min_ffn_norm,
      "min_router_logits_cosine": min_router_logits,
      "min_router_weights_cosine": min_router_weights,
      "min_selected_gate_up_cosine": min_selected_gate_up,
      "min_selected_swiglu_cosine": min_selected_swiglu,
      "min_selected_down_cosine": min_selected_down,
      "min_weighted_selected_down_cosine": min_weighted_selected_down,
      "min_moe_out_cosine": min_moe_out,
      "min_shared_gate_cosine": min_shared_gate,
      "min_shared_swiglu_cosine": min_shared_swiglu,
      "min_shared_down_cosine": min_shared_down,
      "min_shared_gated_cosine": min_shared_gated,
      "min_ffn_out_cosine": min_ffn_out,
      "min_residual_cosine": min_residual,
      "speedup_claims_allowed": False,
      "disposition": (
          f"accept_{DISPOSITION_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_gap_classification"
          if required else
          f"block_{DISPOSITION_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_gap_classification"
      ),
      "selected_next_route": selected_next if required else CURRENT_ROUTE,
      "next_route_reason": (
          "Previous FFN delta drift starts at FFN norm while FFN input is clean. "
          "Root previous FFN norm next."
          if required and selected_next == FFN_NORM_ROUTE else
          "Previous FFN delta drift starts at the router path. Root previous "
          "FFN router next."
          if required and selected_next == ROUTER_ROUTE else
          "Previous FFN delta drift starts in the selected-expert path. Root "
          "selected FFN components next."
          if required and selected_next == SELECTED_ROUTE else
          "Previous FFN delta drift starts in the shared-expert path. Root "
          "shared FFN components next."
          if required and selected_next == SHARED_ROUTE else
          "Previous FFN-delta evidence is incomplete; keep this gate open."
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
      "# Seq473 Upstream Layer-Output FFN-Delta Gap Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- diagnostic_classification: `{metrics['diagnostic_classification']}`",
      f"- min FFN-input cosine: `{metrics['min_ffn_input_cosine']}`",
      f"- min FFN-norm cosine: `{metrics['min_ffn_norm_cosine']}`",
      f"- min FFN-out cosine: `{metrics['min_ffn_out_cosine']}`",
      f"- min residual cosine: `{metrics['min_residual_cosine']}`",
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
  parser.add_argument("--seq472", type=Path, default=DEFAULT_SEQ472)
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
