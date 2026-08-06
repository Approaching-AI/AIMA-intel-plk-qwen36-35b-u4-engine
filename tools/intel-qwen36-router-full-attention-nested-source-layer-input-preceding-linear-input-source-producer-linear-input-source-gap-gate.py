#!/usr/bin/env python3
"""Classify the value source feeding nested producer linear inputs."""

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
OLD_GATE = (
    ROOT
    / "tools/intel-qwen36-router-full-attention-layer-input-preceding-"
    "linear-input-source-producer-linear-input-source-gap-gate.py"
)

SCHEMA_VERSION = (
    "intel-qwen36-router-full-attention-nested-source-layer-input-"
    "preceding-linear-input-source-producer-linear-input-source-gap-gate-v0"
)
CURRENT_ROUTE = (
    "router_prompt_full_attention_layer_input_preceding_linear_input_source_"
    "producer_linear_input_source_linear_input_source_layer_input_preceding_"
    "linear_input_source_producer_linear_input_source_linear_input_source_layer_input_preceding_linear_input_source_producer_linear_input_source_gap_gate"
)
NEXT_ROUTE = (
    "router_prompt_full_attention_layer_input_preceding_linear_input_source_"
    "producer_linear_input_source_linear_input_source_layer_input_preceding_"
    "linear_input_source_producer_linear_input_source_linear_input_source_layer_input_preceding_linear_input_source_producer_linear_input_source_ffn_input_gap_gate"
)
MATH_ROUTE = (
    "router_prompt_full_attention_layer_input_preceding_linear_input_source_"
    "producer_linear_input_source_linear_input_source_layer_input_preceding_"
    "linear_input_source_producer_linear_input_source_linear_input_source_layer_input_preceding_linear_input_source_producer_linear_input_source_ffn_math_gap_gate"
)
DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_SEQ386 = (
    ROOT
    / "output/router-full-attention-nested-source-layer-input-preceding-linear-input-source-ffn-input-gap-gate-20260708Tseq386Z"
    / "metrics.json"
)
DEFAULT_SEQ388 = (
    ROOT
    / "output/router-full-attention-nested-source-layer-input-preceding-linear-input-source-producer-linear-input-gap-gate-20260708Tseq388Z"
    / "metrics.json"
)
DEFAULT_TOKEN_INPUT_DIR = (
    ROOT
    / "output/r2-gpu-acceptance-matrix-router-both-rowblock16-26mask-20260708Tseq222Z"
    / "token-input"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "output/router-full-attention-nested-source-layer-input-preceding-linear-input-source-producer-linear-input-source-gap-gate-20260708Tseq389Z"
)
DEFAULT_HOST = "local"
DEFAULT_MODEL = "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf"
DEFAULT_ENV_SCRIPT = "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh"
DEFAULT_REMOTE_ROOT = "/home/intel/intel-qwen36-gpu"

CASES = ["router_math_reason_001", "router_code_reason_002"]
COSINE_THRESHOLD = 0.9999
TARGET_LINEAR_LAYERS = [1, 5, 9, 13, 17, 21, 25]
SOURCE_FFN_LAYERS = [layer - 1 for layer in TARGET_LINEAR_LAYERS]
DECODE_TOKENS = 8
EXPECTED_EVENTS = len(TARGET_LINEAR_LAYERS) * DECODE_TOKENS
EXPECTED_COUNTER_LAYERS = 72
EXPECTED_COUNTER_VALUES = 147456


def _load_old() -> Any:
  spec = importlib.util.spec_from_file_location(
      "iq36_nested_producer_linear_input_source_base", OLD_GATE)
  if spec is None or spec.loader is None:
    raise SystemExit(f"cannot load old input-source gate: {OLD_GATE}")
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  module.TARGET_LINEAR_LAYERS = TARGET_LINEAR_LAYERS
  module.SOURCE_FFN_LAYERS = SOURCE_FFN_LAYERS
  module.EXPECTED_EVENTS = EXPECTED_EVENTS
  module.EXPECTED_COUNTER_LAYERS = EXPECTED_COUNTER_LAYERS
  module.EXPECTED_COUNTER_VALUES = EXPECTED_COUNTER_VALUES
  return module


OLD = _load_old()


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


def _case_summary(case_id: str, run: dict[str, Any],
                  smoke: dict[str, Any]) -> dict[str, Any]:
  summary = OLD._case_summary(case_id, run, smoke)
  summary["cpu_shadow_attention_output_layers"] = smoke.get(
      "cpu_shadow_attention_output_layers")
  return summary


def compute(args: argparse.Namespace) -> dict[str, Any]:
  routes = _load_json(args.routes)
  seq386 = _load_json(args.seq386)
  seq388 = _load_json(args.seq388)
  binary = seq386.get("inputs", {}).get("binary")
  if not isinstance(binary, str) or not binary:
    raise SystemExit("seq386 binary missing")
  token_cache = OLD.BASE.BASE.BASE.iq36_local.ensure_cached_tokens(
      args.host, f"{args.remote_root}/cache", args.token_input_dir,
      args.timeout_s)

  runs: list[dict[str, Any]] = []
  if token_cache.get("ok") is True:
    for case_id in CASES:
      run = OLD.BASE._run_case(args, binary, str(token_cache.get("dir")),
                               case_id)
      smoke = OLD.BASE._smoke_from_stdout(run)
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
      seq388.get("required_checks_passed") is True
      and seq388.get("selected_next_route") == CURRENT_ROUTE
      and seq388.get("diagnostic_classification")
      == "source_layer_input_preceding_linear_input_source_producer_linear_input_source_linear_input_source_layer_input_preceding_linear_input_source_producer_linear_input_value_gap"
      and _has_candidate(
          routes, 388,
          "accept_source_layer_input_preceding_linear_input_source_producer_linear_input_source_linear_input_source_layer_input_preceding_linear_input_source_producer_linear_input_gap_classification")
      and _has_switch(routes, f"select_{CURRENT_ROUTE}", 388)
  )
  rows_emitted = (
      len(runs) == len(CASES)
      and all(row.get("run", {}).get("returncode") in (0, 2)
              for row in runs)
  )
  counters_ready = rows_emitted and all(row_ready(row) for row in runs)
  distribution_reproduced = rows_emitted and all(
      OLD._dist_fail(row.get("summary", {}).get("distribution", {}))
      for row in runs)
  diagnostics_emitted = counters_ready and all(
      row.get("summary", {}).get("producer_linear_input_source", {})
      .get("target_input", {}).get("observation_count") == EXPECTED_EVENTS
      and row.get("summary", {}).get("producer_linear_input_source", {})
      .get("source_ffn", {}).get("observation_count") == EXPECTED_EVENTS
      for row in runs)

  def min_metric(group: str, name: str) -> float:
    return min(
        (_num(row.get("summary", {}).get(
            "producer_linear_input_source", {}).get(group, {})
              .get(f"min_{name}_cosine"), 1.0)
         for row in runs),
        default=1.0)

  min_target_layer_input = min_metric("target_input", "layer_input")
  min_source_layer_input = min_metric("source_ffn", "layer_input")
  min_source_attention_output = min_metric("source_ffn", "attention_output")
  min_source_ffn_input = min_metric("source_ffn", "ffn_input")
  min_source_cpu_ffn_from_gpu_input = min_metric(
      "source_ffn", "cpu_ffn_from_gpu_input")
  min_source_gpu_output_vs_cpu_ffn = min_metric(
      "source_ffn", "gpu_output_vs_cpu_ffn")
  min_source_cpu_ffn_delta_from_gpu_input = min_metric(
      "source_ffn", "cpu_ffn_delta_from_gpu_input")
  min_source_gpu_ffn_delta_vs_cpu = min_metric(
      "source_ffn", "gpu_ffn_delta_vs_cpu")
  min_source_cpu_ffn_norm_from_gpu_input = min_metric(
      "source_ffn", "cpu_ffn_norm_from_gpu_input")
  min_source_gpu_ffn_norm_vs_cpu = min_metric(
      "source_ffn", "gpu_ffn_norm_vs_cpu")

  target_input_gap = (
      diagnostics_emitted and min_target_layer_input < COSINE_THRESHOLD)
  source_ffn_input_gap = (
      diagnostics_emitted and min_source_ffn_input < COSINE_THRESHOLD)
  source_ffn_output_inherits_input = (
      diagnostics_emitted
      and min_source_cpu_ffn_from_gpu_input < COSINE_THRESHOLD
      and min_source_cpu_ffn_delta_from_gpu_input < COSINE_THRESHOLD
      and min_source_cpu_ffn_norm_from_gpu_input < COSINE_THRESHOLD)
  source_ffn_math_ok = (
      diagnostics_emitted
      and min_source_gpu_output_vs_cpu_ffn >= COSINE_THRESHOLD
      and min_source_gpu_ffn_delta_vs_cpu >= COSINE_THRESHOLD
      and min_source_gpu_ffn_norm_vs_cpu >= COSINE_THRESHOLD)
  diagnostic_classification = (
      "source_layer_input_preceding_linear_input_source_producer_linear_input_source_linear_input_source_layer_input_preceding_linear_input_source_producer_linear_input_source_ffn_input_gap"
      if (target_input_gap and source_ffn_input_gap
          and source_ffn_output_inherits_input and source_ffn_math_ok) else
      "source_layer_input_preceding_linear_input_source_producer_linear_input_source_linear_input_source_layer_input_preceding_linear_input_source_producer_linear_input_source_ffn_math_gap"
      if target_input_gap and not source_ffn_math_ok else
      "source_layer_input_preceding_linear_input_source_producer_linear_input_source_linear_input_source_layer_input_preceding_linear_input_source_producer_linear_input_source_gap_unclassified"
  )
  selected_next = (
      NEXT_ROUTE
      if diagnostic_classification
      == "source_layer_input_preceding_linear_input_source_producer_linear_input_source_linear_input_source_layer_input_preceding_linear_input_source_producer_linear_input_source_ffn_input_gap"
      else MATH_ROUTE
      if diagnostic_classification
      == "source_layer_input_preceding_linear_input_source_producer_linear_input_source_linear_input_source_layer_input_preceding_linear_input_source_producer_linear_input_source_ffn_math_gap"
      else CURRENT_ROUTE
  )
  checks = [
      {"name": "seq388_selected_nested_producer_linear_input_source_gate",
       "pass": preconditions_pass},
      {"name": "producer_linear_input_source_rows_emitted",
       "pass": rows_emitted},
      {"name": "source_and_consumer_counters_ready",
       "pass": counters_ready,
       "detail": [row.get("summary", {}) for row in runs]},
      {"name": "producer_linear_input_source_diagnostics_emitted",
       "pass": diagnostics_emitted,
       "detail": [
           row.get("summary", {}).get("producer_linear_input_source", {})
           for row in runs
       ]},
      {"name": "distribution_failure_reproduced",
       "pass": distribution_reproduced,
       "detail": [
           row.get("summary", {}).get("distribution", {}) for row in runs
       ]},
      {"name": "target_input_gap_reproduced",
       "pass": target_input_gap,
       "detail": {
           "min_target_layer_input_cosine": min_target_layer_input,
           "threshold": COSINE_THRESHOLD,
       }},
      {"name": "source_ffn_output_inherits_ffn_input_gap",
       "pass": source_ffn_output_inherits_input,
       "detail": {
           "min_source_ffn_input_cosine": min_source_ffn_input,
           "min_source_cpu_ffn_from_gpu_input_cosine": (
               min_source_cpu_ffn_from_gpu_input),
           "min_source_cpu_ffn_delta_from_gpu_input_cosine": (
               min_source_cpu_ffn_delta_from_gpu_input),
           "min_source_cpu_ffn_norm_from_gpu_input_cosine": (
               min_source_cpu_ffn_norm_from_gpu_input),
       }},
      {"name": "source_ffn_math_matches_cpu_on_live_input",
       "pass": source_ffn_math_ok,
       "detail": {
           "min_source_gpu_output_vs_cpu_ffn_cosine": (
               min_source_gpu_output_vs_cpu_ffn),
           "min_source_gpu_ffn_delta_vs_cpu_cosine": (
               min_source_gpu_ffn_delta_vs_cpu),
           "min_source_gpu_ffn_norm_vs_cpu_cosine": (
               min_source_gpu_ffn_norm_vs_cpu),
           "threshold": COSINE_THRESHOLD,
       }},
  ]
  required = all(bool(row.get("pass")) for row in checks)
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "inputs": {
          "routes": _rel(args.routes),
          "seq386": _rel(args.seq386),
          "seq388": _rel(args.seq388),
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
      "target_input_gap": target_input_gap,
      "source_ffn_input_gap": source_ffn_input_gap,
      "source_ffn_output_inherits_input": source_ffn_output_inherits_input,
      "source_ffn_math_ok": source_ffn_math_ok,
      "min_target_layer_input_cosine": min_target_layer_input,
      "min_source_layer_input_cosine": min_source_layer_input,
      "min_source_attention_output_cosine": min_source_attention_output,
      "min_source_ffn_input_cosine": min_source_ffn_input,
      "min_source_cpu_ffn_from_gpu_input_cosine": (
          min_source_cpu_ffn_from_gpu_input),
      "min_source_gpu_output_vs_cpu_ffn_cosine": (
          min_source_gpu_output_vs_cpu_ffn),
      "min_source_cpu_ffn_delta_from_gpu_input_cosine": (
          min_source_cpu_ffn_delta_from_gpu_input),
      "min_source_gpu_ffn_delta_vs_cpu_cosine": (
          min_source_gpu_ffn_delta_vs_cpu),
      "min_source_cpu_ffn_norm_from_gpu_input_cosine": (
          min_source_cpu_ffn_norm_from_gpu_input),
      "min_source_gpu_ffn_norm_vs_cpu_cosine": (
          min_source_gpu_ffn_norm_vs_cpu),
      "speedup_claims_allowed": False,
      "disposition": (
          "accept_source_layer_input_preceding_linear_input_source_producer_linear_input_source_linear_input_source_layer_input_preceding_linear_input_source_producer_linear_input_source_gap_classification"
          if required else
          "block_source_layer_input_preceding_linear_input_source_producer_linear_input_source_linear_input_source_layer_input_preceding_linear_input_source_producer_linear_input_source_gap_classification"
      ),
      "selected_next_route": selected_next if required else CURRENT_ROUTE,
      "next_route_reason": (
          "The nested producer linear input gap is inherited from source FFN "
          "input/norm drift; GPU FFN output math matches CPU on the live source "
          "input. Root the source FFN input next."
          if required and diagnostic_classification
          == "source_layer_input_preceding_linear_input_source_producer_linear_input_source_linear_input_source_layer_input_preceding_linear_input_source_producer_linear_input_source_ffn_input_gap"
          else
          "The nested source FFN output math does not match CPU on live input; "
          "root source FFN math next."
          if required and diagnostic_classification
          == "source_layer_input_preceding_linear_input_source_producer_linear_input_source_linear_input_source_layer_input_preceding_linear_input_source_producer_linear_input_source_ffn_math_gap"
          else
          "Nested producer linear input-source evidence is incomplete; keep this gate open."
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
      "# Router Full-Attention Nested Producer Linear Input-Source Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- diagnostic_classification: `{metrics['diagnostic_classification']}`",
      f"- min target layer-input cosine: `{metrics['min_target_layer_input_cosine']}`",
      f"- min source FFN input cosine: `{metrics['min_source_ffn_input_cosine']}`",
      f"- min source CPU FFN-from-live-input cosine: `{metrics['min_source_cpu_ffn_from_gpu_input_cosine']}`",
      f"- min source GPU-output-vs-CPU-FFN cosine: `{metrics['min_source_gpu_output_vs_cpu_ffn_cosine']}`",
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
  parser.add_argument("--seq386", type=Path, default=DEFAULT_SEQ386)
  parser.add_argument("--seq388", type=Path, default=DEFAULT_SEQ388)
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
