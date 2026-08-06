#!/usr/bin/env python3
"""Classify deeper nested producer FFN input drift."""

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
    / "tools/intel-qwen36-router-full-attention-layer-input-preceding-"
    "linear-input-source-producer-linear-input-source-linear-input-source-"
    "layer-input-preceding-linear-input-source-ffn-input-gap-gate.py"
)

SCHEMA_VERSION = (
    "intel-qwen36-router-full-attention-deeper-nested-preceding-linear-"
    "input-source-ffn-input-gap-gate-v0"
)
ROUTE_PREFIX = (
    "router_prompt_full_attention_layer_input_preceding_linear_input_source_"
    "producer_linear_input_source_linear_input_source_layer_input_preceding_"
    "linear_input_source_producer_linear_input_source_linear_input_source_"
    "layer_input_preceding_linear_input_source_producer_linear_input_source_"
    "linear_input_source_layer_input_preceding_linear_input_source"
)
CURRENT_ROUTE = f"{ROUTE_PREFIX}_ffn_input_gap_gate"
DELTA_Z_ROUTE = f"{ROUTE_PREFIX}_producer_linear_delta_z_gap_gate"
DELTA_ROUTE = f"{ROUTE_PREFIX}_producer_linear_delta_output_gap_gate"
Z_ROUTE = f"{ROUTE_PREFIX}_producer_linear_z_gap_gate"
MATH_ROUTE = f"{ROUTE_PREFIX}_ffn_math_gap_gate"

DISPOSITION_PREFIX = (
    "source_layer_input_preceding_linear_input_source_producer_linear_input_"
    "source_linear_input_source_layer_input_preceding_linear_input_source_"
    "producer_linear_input_source_linear_input_source_layer_input_preceding_"
    "linear_input_source"
)
DIAG_PREFIX = (
    "source_layer_input_preceding_linear_input_source_producer_linear_input_"
    "source_linear_input_source_layer_input_preceding_linear_input_source_"
    "producer_linear_input_source_linear_input_source_layer_input_preceding_"
    "linear_input_source_producer"
)

DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_SEQ400 = (
    ROOT
    / "output/router-full-attention-nested-source-layer-input-preceding-linear-input-source-producer-linear-input-source-linear-input-source-layer-input-preceding-linear-input-source-gap-gate-20260708Tseq400Z"
    / "metrics.json"
)
DEFAULT_TOKEN_INPUT_DIR = (
    ROOT
    / "output/r2-gpu-acceptance-matrix-router-both-rowblock16-26mask-20260708Tseq222Z"
    / "token-input"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "output/router-full-attention-deeper-nested-preceding-linear-input-source-ffn-input-gap-gate-20260709Tseq401Z"
)
DEFAULT_HOST = "local"
DEFAULT_MODEL = "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf"
DEFAULT_ENV_SCRIPT = "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh"
DEFAULT_REMOTE_ROOT = "/home/intel/intel-qwen36-gpu"

PRODUCER_LAYERS = [1, 5, 9, 13, 17, 21]
DECODE_TOKENS = 8
EXPECTED_EVENTS = len(PRODUCER_LAYERS) * DECODE_TOKENS
EXPECTED_COUNTER_LAYERS = 72
EXPECTED_COUNTER_VALUES = 147456
COSINE_THRESHOLD = 0.9999


def _load_base() -> Any:
  spec = importlib.util.spec_from_file_location(
      "iq36_deeper_nested_producer_ffn_input_base", BASE_GATE)
  if spec is None or spec.loader is None:
    raise SystemExit(f"cannot load base FFN-input gate: {BASE_GATE}")
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  module.PRODUCER_LAYERS = PRODUCER_LAYERS
  module.EXPECTED_EVENTS = EXPECTED_EVENTS
  module.EXPECTED_COUNTER_LAYERS = EXPECTED_COUNTER_LAYERS
  module.EXPECTED_COUNTER_VALUES = EXPECTED_COUNTER_VALUES
  module.OLD.PRODUCER_LAYERS = PRODUCER_LAYERS
  module.OLD.EXPECTED_EVENTS = EXPECTED_EVENTS
  module.OLD.EXPECTED_COUNTER_LAYERS = EXPECTED_COUNTER_LAYERS
  module.OLD.EXPECTED_COUNTER_VALUES = EXPECTED_COUNTER_VALUES
  return module


BASE = _load_base()
CASES = BASE.CASES


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


def compute(args: argparse.Namespace) -> dict[str, Any]:
  routes = _load_json(args.routes)
  seq400 = _load_json(args.seq400)
  binary = seq400.get("inputs", {}).get("binary")
  if not isinstance(binary, str) or not binary:
    raise SystemExit("seq400 binary missing")
  token_cache = BASE.OLD.BASE.BASE.BASE.iq36_local.ensure_cached_tokens(
      args.host, f"{args.remote_root}/cache", args.token_input_dir,
      args.timeout_s)

  runs: list[dict[str, Any]] = []
  if token_cache.get("ok") is True:
    for case_id in CASES:
      run = BASE.OLD.BASE._run_case(
          args, binary, str(token_cache.get("dir")), case_id)
      smoke = BASE.OLD.BASE._smoke_from_stdout(run)
      runs.append({
          "case_id": case_id,
          "run": {
              "cmd": run.get("cmd"),
              "returncode": run.get("returncode"),
              "stdout_bytes": len(str(run.get("stdout") or "")),
              "stderr_bytes": len(str(run.get("stderr") or "")),
          },
          "summary": BASE._case_summary(case_id, run, smoke),
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
      seq400.get("required_checks_passed") is True
      and seq400.get("selected_next_route") == CURRENT_ROUTE
      and seq400.get("diagnostic_classification")
      == f"{DIAG_PREFIX}_ffn_input_gap"
      and _has_candidate(
          routes, 400,
          f"accept_{DISPOSITION_PREFIX}_gap_classification")
      and _has_switch(routes, f"select_{CURRENT_ROUTE}", 400)
  )
  rows_emitted = (
      len(runs) == len(CASES)
      and all(row.get("run", {}).get("returncode") in (0, 2)
              for row in runs)
  )
  counters_ready = rows_emitted and all(row_ready(row) for row in runs)
  distribution_reproduced = rows_emitted and all(
      BASE.OLD._dist_fail(row.get("summary", {}).get("distribution", {}))
      for row in runs)
  diagnostics_emitted = counters_ready and all(
      row.get("summary", {}).get("producer_ffn_input", {})
      .get("residual_source", {}).get("observation_count") == EXPECTED_EVENTS
      and row.get("summary", {}).get("producer_ffn_input", {})
      .get("linear_attention", {}).get("observation_count") == EXPECTED_EVENTS
      and row.get("summary", {}).get("producer_ffn_input", {})
      .get("preconv_source", {}).get("observation_count") == EXPECTED_EVENTS
      and row.get("summary", {}).get("producer_ffn_input", {})
      .get("final_mix", {}).get("observation_count") == EXPECTED_EVENTS
      for row in runs)

  def min_metric(group: str, name: str) -> float:
    return min(
        (_num(row.get("summary", {}).get("producer_ffn_input", {})
              .get(group, {}).get(f"min_{name}_cosine"), 1.0)
         for row in runs),
        default=1.0)

  min_ffn_input = min_metric("residual_source", "ffn_input")
  min_attention_output = min_metric("residual_source", "attention_output")
  min_gpu_output_vs_cpu_ffn = min_metric(
      "residual_source", "gpu_output_vs_cpu_ffn")
  min_gpu_ffn_delta_vs_cpu = min_metric(
      "residual_source", "gpu_ffn_delta_vs_cpu")
  min_gpu_ffn_norm_vs_cpu = min_metric(
      "residual_source", "gpu_ffn_norm_vs_cpu")
  min_linear_final = min_metric("linear_attention", "final_output")
  min_delta_output = min_metric("linear_attention", "delta_output")
  min_z = min_metric("linear_attention", "z")
  min_gpu_qkv_vs_cpu = min_metric("preconv_source", "gpu_qkv_vs_cpu")
  min_gpu_z_vs_cpu = min_metric("preconv_source", "gpu_z_vs_cpu")
  min_final = min_metric("final_mix", "gpu_kernel_final")
  min_delta_native_z = min_metric("final_mix", "gpu_delta_native_z_cpu")
  min_native_delta_gpu_z = min_metric("final_mix", "native_delta_gpu_z_cpu")
  min_native_recompute = min_metric("final_mix", "native_delta_native_z_cpu")

  ffn_input_gap = diagnostics_emitted and min_ffn_input < COSINE_THRESHOLD
  ffn_math_ok = (
      diagnostics_emitted
      and min_gpu_output_vs_cpu_ffn >= COSINE_THRESHOLD
      and min_gpu_ffn_delta_vs_cpu >= COSINE_THRESHOLD
      and min_gpu_ffn_norm_vs_cpu >= COSINE_THRESHOLD)
  attention_output_gap = (
      diagnostics_emitted and min_attention_output < COSINE_THRESHOLD)
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
      and min_gpu_qkv_vs_cpu >= COSINE_THRESHOLD
      and min_gpu_z_vs_cpu >= COSINE_THRESHOLD)

  diagnostic_classification = (
      f"{DIAG_PREFIX}_linear_attention_delta_z_input_gap"
      if (ffn_input_gap and ffn_math_ok and attention_output_gap
          and linear_final_gap and linear_final_recompute_ok
          and delta_drives_gap and z_drives_gap and preconv_math_ok) else
      f"{DIAG_PREFIX}_linear_attention_delta_output_gap"
      if (ffn_input_gap and ffn_math_ok and attention_output_gap
          and linear_final_gap and delta_drives_gap) else
      f"{DIAG_PREFIX}_linear_attention_z_gap"
      if (ffn_input_gap and ffn_math_ok and attention_output_gap
          and linear_final_gap and z_drives_gap) else
      f"{DIAG_PREFIX}_ffn_math_gap"
      if ffn_input_gap and not ffn_math_ok else
      f"{DIAG_PREFIX}_ffn_input_gap_unclassified"
  )
  selected_next = (
      DELTA_Z_ROUTE
      if diagnostic_classification
      == f"{DIAG_PREFIX}_linear_attention_delta_z_input_gap"
      else DELTA_ROUTE
      if diagnostic_classification
      == f"{DIAG_PREFIX}_linear_attention_delta_output_gap"
      else Z_ROUTE
      if diagnostic_classification
      == f"{DIAG_PREFIX}_linear_attention_z_gap"
      else MATH_ROUTE
      if diagnostic_classification == f"{DIAG_PREFIX}_ffn_math_gap"
      else CURRENT_ROUTE
  )
  checks = [
      {"name": "seq400_selected_deeper_nested_producer_ffn_input_gap_gate",
       "pass": preconditions_pass},
      {"name": "producer_ffn_input_rows_emitted", "pass": rows_emitted},
      {"name": "source_and_consumer_counters_ready",
       "pass": counters_ready,
       "detail": [row.get("summary", {}) for row in runs]},
      {"name": "producer_ffn_input_diagnostics_emitted",
       "pass": diagnostics_emitted,
       "detail": [
           row.get("summary", {}).get("producer_ffn_input", {})
           for row in runs
       ]},
      {"name": "distribution_failure_reproduced",
       "pass": distribution_reproduced,
       "detail": [
           row.get("summary", {}).get("distribution", {}) for row in runs
       ]},
      {"name": "producer_ffn_input_gap_reproduced",
       "pass": ffn_input_gap,
       "detail": {"min_ffn_input_cosine": min_ffn_input}},
      {"name": "producer_ffn_math_matches_cpu_on_live_input",
       "pass": ffn_math_ok,
       "detail": {
           "min_gpu_output_vs_cpu_ffn_cosine": min_gpu_output_vs_cpu_ffn,
           "min_gpu_ffn_delta_vs_cpu_cosine": min_gpu_ffn_delta_vs_cpu,
           "min_gpu_ffn_norm_vs_cpu_cosine": min_gpu_ffn_norm_vs_cpu,
       }},
      {"name": "producer_linear_final_recompute_matches_native",
       "pass": linear_final_recompute_ok,
       "detail": {
           "min_native_delta_native_z_cpu_cosine": min_native_recompute,
       }},
  ]
  required = all(bool(row.get("pass")) for row in checks)
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "inputs": {
          "routes": _rel(args.routes),
          "seq400": _rel(args.seq400),
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
      "ffn_input_gap": ffn_input_gap,
      "ffn_math_ok": ffn_math_ok,
      "attention_output_gap": attention_output_gap,
      "linear_final_gap": linear_final_gap,
      "linear_final_recompute_ok": linear_final_recompute_ok,
      "delta_drives_gap": delta_drives_gap,
      "z_drives_gap": z_drives_gap,
      "preconv_math_ok": preconv_math_ok,
      "min_ffn_input_cosine": min_ffn_input,
      "min_attention_output_cosine": min_attention_output,
      "min_gpu_output_vs_cpu_ffn_cosine": min_gpu_output_vs_cpu_ffn,
      "min_gpu_ffn_delta_vs_cpu_cosine": min_gpu_ffn_delta_vs_cpu,
      "min_gpu_ffn_norm_vs_cpu_cosine": min_gpu_ffn_norm_vs_cpu,
      "min_linear_final_output_cosine": min_linear_final,
      "min_delta_output_cosine": min_delta_output,
      "min_z_cosine": min_z,
      "min_gpu_qkv_vs_cpu_cosine": min_gpu_qkv_vs_cpu,
      "min_gpu_z_vs_cpu_cosine": min_gpu_z_vs_cpu,
      "min_gpu_kernel_final_cosine": min_final,
      "min_gpu_delta_native_z_cpu_cosine": min_delta_native_z,
      "min_native_delta_gpu_z_cpu_cosine": min_native_delta_gpu_z,
      "min_native_delta_native_z_cpu_cosine": min_native_recompute,
      "speedup_claims_allowed": False,
      "disposition": (
          f"accept_{DISPOSITION_PREFIX}_ffn_input_gap_classification"
          if required else
          f"block_{DISPOSITION_PREFIX}_ffn_input_gap_classification"
      ),
      "selected_next_route": selected_next if required else CURRENT_ROUTE,
      "next_route_reason": (
          "Deeper nested producer FFN input drift is inherited from producer "
          "linear-attention final output, and the final-output gap is driven "
          "by both delta-output and z input drift while native final recompute "
          "is sound."
          if required and diagnostic_classification
          == f"{DIAG_PREFIX}_linear_attention_delta_z_input_gap"
          else
          "Deeper nested producer FFN input drift is inherited from producer "
          "linear-attention delta-output drift."
          if required and diagnostic_classification
          == f"{DIAG_PREFIX}_linear_attention_delta_output_gap"
          else
          "Deeper nested producer FFN input drift is inherited from producer "
          "linear-attention z drift."
          if required and diagnostic_classification
          == f"{DIAG_PREFIX}_linear_attention_z_gap"
          else
          "Deeper nested producer FFN math on live input does not match CPU; "
          "root FFN math next."
          if required and diagnostic_classification == f"{DIAG_PREFIX}_ffn_math_gap"
          else
          "Deeper nested producer FFN input evidence is incomplete; keep this "
          "gate open."
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
      "# Router Full-Attention Deeper Nested Producer FFN-Input Gap Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- diagnostic_classification: `{metrics['diagnostic_classification']}`",
      f"- min FFN-input cosine: `{metrics['min_ffn_input_cosine']}`",
      f"- min attention-output cosine: `{metrics['min_attention_output_cosine']}`",
      f"- min linear final-output cosine: `{metrics['min_linear_final_output_cosine']}`",
      f"- min delta/z cosines: `{metrics['min_delta_output_cosine']}` / `{metrics['min_z_cosine']}`",
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
  parser.add_argument("--seq400", type=Path, default=DEFAULT_SEQ400)
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
