#!/usr/bin/env python3
"""Classify seq471 previous layer-output drift."""

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
SEQ471_GATE = (
    ROOT
    / "tools/intel-qwen36-seq471-producer-linear-input-source-linear-input-source-layer-input-gap-gate.py"
)
SCHEMA_VERSION = "intel-qwen36-seq472-upstream-layer-output-gap-gate-v0"

DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_SEQ471 = (
    ROOT
    / "output/seq471-producer-linear-input-source-linear-input-source-layer-input-gap-gate-20260709Tseq471Z"
    / "metrics.json"
)
DEFAULT_TOKEN_INPUT_DIR = (
    ROOT
    / "output/r2-gpu-acceptance-matrix-router-both-rowblock16-26mask-20260708Tseq222Z"
    / "token-input"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "output/seq472-upstream-layer-output-gap-gate-20260709Tseq472Z"
)
DEFAULT_HOST = "local"
DEFAULT_MODEL = "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf"
DEFAULT_ENV_SCRIPT = "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh"
DEFAULT_REMOTE_ROOT = "/home/intel/intel-qwen36-gpu"

COSINE_THRESHOLD = 0.9999
DECODE_TOKENS = 8
TARGET_LAYERS = [3]
PREVIOUS_LAYERS = [layer - 1 for layer in TARGET_LAYERS]
EXPECTED_EVENTS = len(PREVIOUS_LAYERS) * DECODE_TOKENS
EXPECTED_COUNTER_LAYERS = 72
EXPECTED_COUNTER_VALUES = 147456

_PATTERN = (
    "producer_linear_input_source_linear_input_source_layer_input_preceding_"
    "linear_input_source"
)
SOURCE_GAP_ROUTE = (
    "router_prompt_full_attention_layer_input_preceding_linear_input_source_"
    + ((_PATTERN + "_") * 6)
    + _PATTERN
    + "_gap_gate"
)
LAYER_INPUT_ROUTE = SOURCE_GAP_ROUTE.replace(
    "_linear_input_source_gap_gate",
    "_linear_input_source_producer_linear_input_source_linear_input_source_"
    "layer_input_gap_gate")
CURRENT_ROUTE = LAYER_INPUT_ROUTE.replace(
    "_layer_input_gap_gate", "_layer_input_upstream_layer_output_gap_gate")
FFN_INPUT_ROUTE = CURRENT_ROUTE.replace(
    "_layer_output_gap_gate", "_layer_output_ffn_input_gap_gate")
FFN_DELTA_ROUTE = CURRENT_ROUTE.replace(
    "_layer_output_gap_gate", "_layer_output_ffn_delta_gap_gate")
FFN_MATH_ROUTE = CURRENT_ROUTE.replace(
    "_layer_output_gap_gate", "_layer_output_ffn_math_gap_gate")
_SOURCE_GAP_SOURCE = SOURCE_GAP_ROUTE.replace(
    "router_prompt_full_attention_layer_input_preceding_linear_input_source_",
    "source_layer_input_preceding_linear_input_source_",
    1,
)
DISPOSITION_PREFIX = _SOURCE_GAP_SOURCE.removesuffix(
    "_producer_linear_input_source_linear_input_source_layer_input_preceding_"
    "linear_input_source_producer_linear_input_source_linear_input_source_"
    "layer_input_preceding_linear_input_source_gap_gate")
DIAG_PREFIX = (
    f"{DISPOSITION_PREFIX}_producer_linear_input_source_linear_input_source_"
    "layer_input_preceding_linear_input_source_producer"
)
DISPOSITION_PREFIX = DIAG_PREFIX.removesuffix("_producer")


def _load_seq471() -> Any:
  spec = importlib.util.spec_from_file_location("iq36_seq471_gate", SEQ471_GATE)
  if spec is None or spec.loader is None:
    raise SystemExit(f"cannot load seq471 gate: {SEQ471_GATE}")
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  module._patch_base()
  return module


SEQ471 = _load_seq471()


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
  return SEQ471.BASE.OLD._dist_fail(dist)


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


def _previous_ffn_summary(smoke: dict[str, Any]) -> dict[str, Any]:
  steps = smoke.get("residual_source_diff_by_step")
  steps = steps if isinstance(steps, list) else []
  names = [
      "layer_input",
      "attention_output",
      "ffn_input",
      "ffn_delta",
      "cpu_ffn_from_gpu_input",
      "gpu_output_vs_cpu_ffn",
      "cpu_ffn_delta_from_gpu_input",
      "gpu_ffn_delta_vs_cpu",
      "cpu_ffn_norm_from_gpu_input",
      "gpu_ffn_norm_vs_cpu",
  ]
  mins = {name: 1.0 for name in names}
  max_abs = {name: 0.0 for name in names}
  first = {name: None for name in names}
  obs = 0
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
      for name in names:
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
  }
  for name in names:
    summary[f"min_{name}_cosine"] = mins[name]
    summary[f"max_{name}_abs_diff"] = max_abs[name]
    summary[f"first_{name}_gap"] = first[name]
  return summary


def _case_summary(case_id: str, run: dict[str, Any],
                  smoke: dict[str, Any]) -> dict[str, Any]:
  dist = SEQ471.BASE.OLD.BASE._distribution_summary(smoke)
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
      "distribution": dist,
      "layer_boundary": SEQ471._layer_boundary_summary(smoke),
      "previous_ffn": _previous_ffn_summary(smoke),
  }


def compute(args: argparse.Namespace) -> dict[str, Any]:
  routes = _load_json(args.routes)
  seq471 = _load_json(args.seq471)
  binary = seq471.get("inputs", {}).get("binary")
  if not isinstance(binary, str) or not binary:
    raise SystemExit("seq471 binary missing")
  token_cache = SEQ471.BASE.OLD.BASE.BASE.BASE.iq36_local.ensure_cached_tokens(
      args.host, f"{args.remote_root}/cache", args.token_input_dir,
      args.timeout_s)

  runs: list[dict[str, Any]] = []
  if token_cache.get("ok") is True:
    for case_id in SEQ471.BASE.CASES:
      run = SEQ471.BASE.OLD.BASE._run_case(
          args, binary, str(token_cache.get("dir")), case_id)
      smoke = SEQ471.BASE.OLD.BASE._smoke_from_stdout(run)
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
      seq471.get("required_checks_passed") is True
      and seq471.get("selected_next_route") == CURRENT_ROUTE
      and seq471.get("diagnostic_classification")
      == f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_gap"
      and _has_candidate(routes, 471, str(seq471.get("disposition")))
      and _has_switch(routes, f"select_{CURRENT_ROUTE}", 471)
  )
  rows_emitted = (
      len(runs) == len(SEQ471.BASE.CASES)
      and all(row.get("run", {}).get("returncode") in (0, 2)
              for row in runs)
  )
  counters_ready = rows_emitted and all(row_ready(row) for row in runs)
  distribution_reproduced = rows_emitted and all(
      _dist_fail(row.get("summary", {}).get("distribution", {}))
      for row in runs)
  boundary_emitted = counters_ready and all(
      row.get("summary", {}).get("layer_boundary", {})
      .get("previous_output_observation_count") == EXPECTED_EVENTS
      for row in runs)
  previous_ffn_emitted = counters_ready and all(
      row.get("summary", {}).get("previous_ffn", {})
      .get("observation_count") == EXPECTED_EVENTS
      for row in runs)

  def min_boundary(name: str) -> float:
    return min(
        (_num(row.get("summary", {}).get("layer_boundary", {}).get(name), 1.0)
         for row in runs),
        default=1.0)

  def min_ffn(name: str) -> float:
    return min(
        (_num(row.get("summary", {}).get("previous_ffn", {}).get(name), 1.0)
         for row in runs),
        default=1.0)

  min_previous_output = min_boundary("min_previous_output_cosine")
  min_boundary_target_input = min_boundary("min_target_input_cosine")
  min_layer_input = min_ffn("min_layer_input_cosine")
  min_attention_output = min_ffn("min_attention_output_cosine")
  min_ffn_input = min_ffn("min_ffn_input_cosine")
  min_ffn_delta = min_ffn("min_ffn_delta_cosine")
  min_cpu_from_input = min_ffn("min_cpu_ffn_from_gpu_input_cosine")
  min_gpu_output_vs_cpu = min_ffn("min_gpu_output_vs_cpu_ffn_cosine")
  min_cpu_delta_from_input = min_ffn("min_cpu_ffn_delta_from_gpu_input_cosine")
  min_gpu_delta_vs_cpu = min_ffn("min_gpu_ffn_delta_vs_cpu_cosine")
  min_cpu_norm_from_input = min_ffn("min_cpu_ffn_norm_from_gpu_input_cosine")
  min_gpu_norm_vs_cpu = min_ffn("min_gpu_ffn_norm_vs_cpu_cosine")
  previous_output_gap = boundary_emitted and min_previous_output < COSINE_THRESHOLD
  previous_layer_input_clean = (
      previous_ffn_emitted and min_layer_input >= COSINE_THRESHOLD)
  previous_attention_output_gap = (
      previous_ffn_emitted and min_attention_output < COSINE_THRESHOLD)
  ffn_input_gap = previous_ffn_emitted and min_ffn_input < COSINE_THRESHOLD
  ffn_input_clean = (
      previous_ffn_emitted and min_ffn_input >= COSINE_THRESHOLD)
  ffn_delta_gap = previous_ffn_emitted and min_ffn_delta < COSINE_THRESHOLD
  output_inherits_ffn_input = (
      previous_ffn_emitted
      and min_cpu_from_input < COSINE_THRESHOLD
      and min_cpu_delta_from_input < COSINE_THRESHOLD
      and min_cpu_norm_from_input < COSINE_THRESHOLD)
  ffn_math_ok = (
      previous_ffn_emitted
      and min_gpu_output_vs_cpu >= COSINE_THRESHOLD
      and min_gpu_delta_vs_cpu >= COSINE_THRESHOLD
      and min_gpu_norm_vs_cpu >= COSINE_THRESHOLD)

  diagnostic_classification = (
      f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_input_gap"
      if (previous_output_gap and ffn_input_gap and output_inherits_ffn_input
          and ffn_math_ok) else
      f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_gap"
      if (previous_output_gap and ffn_delta_gap and ffn_input_clean
          and previous_layer_input_clean and ffn_math_ok) else
      f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_math_gap"
      if previous_output_gap and not ffn_math_ok else
      f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_gap_unclassified"
  )
  selected_next = (
      FFN_INPUT_ROUTE
      if diagnostic_classification
      == f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_input_gap"
      else FFN_DELTA_ROUTE
      if diagnostic_classification
      == f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_gap"
      else FFN_MATH_ROUTE
      if diagnostic_classification
      == f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_math_gap"
      else CURRENT_ROUTE
  )
  checks = [
      {"name": "seq471_selected_upstream_layer_output_gap_gate",
       "pass": preconditions_pass},
      {"name": "upstream_layer_output_rows_emitted", "pass": rows_emitted},
      {"name": "source_and_consumer_counters_ready",
       "pass": counters_ready,
       "detail": [row.get("summary", {}) for row in runs]},
      {"name": "distribution_failure_reproduced",
       "pass": distribution_reproduced,
       "detail": [
           row.get("summary", {}).get("distribution", {}) for row in runs
       ]},
      {"name": "layer_boundary_diagnostics_emitted",
       "pass": boundary_emitted,
       "detail": [
           row.get("summary", {}).get("layer_boundary", {}) for row in runs
       ]},
      {"name": "previous_ffn_diagnostics_emitted",
       "pass": previous_ffn_emitted,
       "detail": [
           row.get("summary", {}).get("previous_ffn", {}) for row in runs
       ]},
      {"name": "previous_layer_output_gap_reproduced",
       "pass": previous_output_gap,
       "detail": {
           "min_previous_layer_output_cosine": min_previous_output,
           "min_boundary_target_input_cosine": min_boundary_target_input,
       }},
      {"name": "previous_layer_input_clean_for_ffn_delta_split",
       "pass": previous_layer_input_clean,
       "detail": {"min_previous_layer_input_cosine": min_layer_input}},
      {"name": "previous_attention_output_gap_observed",
       "pass": previous_attention_output_gap,
       "detail": {"min_previous_attention_output_cosine": min_attention_output}},
      {"name": "previous_ffn_input_clean_for_ffn_delta_split",
       "pass": ffn_input_clean,
       "detail": {"min_previous_ffn_input_cosine": min_ffn_input}},
      {"name": "previous_ffn_delta_gap_reproduced",
       "pass": ffn_delta_gap,
       "detail": {"min_previous_ffn_delta_cosine": min_ffn_delta}},
      {"name": "previous_layer_output_inherits_ffn_input",
       "pass": output_inherits_ffn_input,
       "detail": {
           "min_cpu_ffn_from_gpu_input_cosine": min_cpu_from_input,
           "min_cpu_ffn_delta_from_gpu_input_cosine": min_cpu_delta_from_input,
           "min_cpu_ffn_norm_from_gpu_input_cosine": min_cpu_norm_from_input,
       }},
      {"name": "previous_ffn_math_matches_cpu_on_live_input",
       "pass": ffn_math_ok,
       "detail": {
           "min_gpu_output_vs_cpu_ffn_cosine": min_gpu_output_vs_cpu,
           "min_gpu_ffn_delta_vs_cpu_cosine": min_gpu_delta_vs_cpu,
           "min_gpu_ffn_norm_vs_cpu_cosine": min_gpu_norm_vs_cpu,
       }},
  ]
  required = all(bool(row.get("pass")) for row in checks)
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "inputs": {
          "routes": _rel(args.routes),
          "seq471": _rel(args.seq471),
          "token_input_dir": _rel(args.token_input_dir),
          "host": args.host,
          "binary": binary,
          "target_layers": TARGET_LAYERS,
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
      "previous_layer_output_gap": previous_output_gap,
      "previous_layer_input_clean": previous_layer_input_clean,
      "previous_attention_output_gap": previous_attention_output_gap,
      "previous_ffn_input_gap": ffn_input_gap,
      "previous_ffn_input_clean": ffn_input_clean,
      "previous_ffn_delta_gap": ffn_delta_gap,
      "previous_layer_output_inherits_ffn_input": output_inherits_ffn_input,
      "previous_ffn_math_ok": ffn_math_ok,
      "min_previous_layer_output_cosine": min_previous_output,
      "min_boundary_target_input_cosine": min_boundary_target_input,
      "min_previous_layer_input_cosine": min_layer_input,
      "min_previous_attention_output_cosine": min_attention_output,
      "min_previous_ffn_input_cosine": min_ffn_input,
      "min_previous_ffn_delta_cosine": min_ffn_delta,
      "min_previous_cpu_ffn_from_gpu_input_cosine": min_cpu_from_input,
      "min_previous_gpu_output_vs_cpu_ffn_cosine": min_gpu_output_vs_cpu,
      "min_previous_cpu_ffn_delta_from_gpu_input_cosine": (
          min_cpu_delta_from_input),
      "min_previous_gpu_ffn_delta_vs_cpu_cosine": min_gpu_delta_vs_cpu,
      "min_previous_cpu_ffn_norm_from_gpu_input_cosine": min_cpu_norm_from_input,
      "min_previous_gpu_ffn_norm_vs_cpu_cosine": min_gpu_norm_vs_cpu,
      "speedup_claims_allowed": False,
      "disposition": (
          f"accept_{DISPOSITION_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_gap_classification"
          if required else
          f"block_{DISPOSITION_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_gap_classification"
      ),
      "selected_next_route": selected_next if required else CURRENT_ROUTE,
      "next_route_reason": (
          "Previous layer output drift is inherited from previous FFN input "
          "while GPU FFN output/delta/norm math matches CPU on live input. "
          "Root previous FFN input next."
          if required and selected_next == FFN_INPUT_ROUTE else
          "Previous layer output drift is in the FFN delta while previous "
          "layer/FFN inputs are clean and GPU FFN output/delta/norm math "
          "matches CPU on live input. Root previous FFN delta next."
          if required and selected_next == FFN_DELTA_ROUTE else
          "Previous layer FFN math does not match CPU on live input. Root "
          "previous FFN math next."
          if required and selected_next == FFN_MATH_ROUTE else
          "Previous layer-output evidence is incomplete; keep this gate open."
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
      "# Seq472 Upstream Layer-Output Gap Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- diagnostic_classification: `{metrics['diagnostic_classification']}`",
      f"- min previous layer-output cosine: `{metrics['min_previous_layer_output_cosine']}`",
      f"- min previous FFN-input cosine: `{metrics['min_previous_ffn_input_cosine']}`",
      f"- min previous FFN-delta cosine: `{metrics['min_previous_ffn_delta_cosine']}`",
      f"- min previous GPU-output-vs-CPU-FFN cosine: `{metrics['min_previous_gpu_output_vs_cpu_ffn_cosine']}`",
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
  parser.add_argument("--seq471", type=Path, default=DEFAULT_SEQ471)
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
