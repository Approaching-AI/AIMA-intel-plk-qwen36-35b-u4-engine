#!/usr/bin/env python3
"""Classify seq473 previous FFN-norm drift."""

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
SEQ473_GATE = (
    ROOT
    / "tools/intel-qwen36-seq473-upstream-layer-output-ffn-delta-gap-gate.py"
)
SCHEMA_VERSION = (
    "intel-qwen36-seq474-upstream-layer-output-ffn-delta-ffn-norm-gap-gate-v0"
)

DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_SEQ473 = (
    ROOT
    / "output/seq473-upstream-layer-output-ffn-delta-gap-gate-20260709Tseq473Z"
    / "metrics.json"
)
DEFAULT_TOKEN_INPUT_DIR = (
    ROOT
    / "output/r2-gpu-acceptance-matrix-router-both-rowblock16-26mask-20260708Tseq222Z"
    / "token-input"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "output/seq474-upstream-layer-output-ffn-delta-ffn-norm-gap-gate-20260709Tseq474Z"
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


def _load_seq473() -> Any:
  spec = importlib.util.spec_from_file_location("iq36_seq473_gate", SEQ473_GATE)
  if spec is None or spec.loader is None:
    raise SystemExit(f"cannot load seq473 gate: {SEQ473_GATE}")
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


SEQ473 = _load_seq473()
CURRENT_ROUTE = SEQ473.FFN_NORM_ROUTE
INPUT_ATTENTION_ROUTE = CURRENT_ROUTE.replace(
    "_ffn_delta_ffn_norm_gap_gate",
    "_ffn_delta_ffn_norm_input_attention_output_gap_gate")
INPUT_SENSITIVITY_ROUTE = CURRENT_ROUTE.replace(
    "_ffn_delta_ffn_norm_gap_gate",
    "_ffn_delta_ffn_norm_input_sensitivity_gap_gate")
NORM_MATH_ROUTE = CURRENT_ROUTE.replace(
    "_ffn_delta_ffn_norm_gap_gate",
    "_ffn_delta_ffn_norm_math_gap_gate")
DIAG_PREFIX = SEQ473.DIAG_PREFIX
DISPOSITION_PREFIX = SEQ473.DISPOSITION_PREFIX


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
      "distribution": SEQ473.SEQ472.SEQ471.BASE.OLD.BASE._distribution_summary(
          smoke),
      "component": SEQ473._component_summary(smoke),
      "previous_ffn": SEQ473.SEQ472._previous_ffn_summary(smoke),
  }


def compute(args: argparse.Namespace) -> dict[str, Any]:
  routes = _load_json(args.routes)
  seq473 = _load_json(args.seq473)
  binary = seq473.get("inputs", {}).get("binary")
  if not isinstance(binary, str) or not binary:
    raise SystemExit("seq473 binary missing")
  token_cache = (
      SEQ473.SEQ472.SEQ471.BASE.OLD.BASE.BASE.BASE
      .iq36_local.ensure_cached_tokens(
          args.host, f"{args.remote_root}/cache", args.token_input_dir,
          args.timeout_s))

  runs: list[dict[str, Any]] = []
  if token_cache.get("ok") is True:
    for case_id in SEQ473.SEQ472.SEQ471.BASE.CASES:
      run = SEQ473.SEQ472.SEQ471.BASE.OLD.BASE._run_case(
          args, binary, str(token_cache.get("dir")), case_id)
      smoke = SEQ473.SEQ472.SEQ471.BASE.OLD.BASE._smoke_from_stdout(run)
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
      seq473.get("required_checks_passed") is True
      and seq473.get("selected_next_route") == CURRENT_ROUTE
      and seq473.get("diagnostic_classification")
      == f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_gap"
      and _has_candidate(routes, 473, str(seq473.get("disposition")))
      and _has_switch(routes, f"select_{CURRENT_ROUTE}", 473)
  )
  rows_emitted = (
      len(runs) == len(SEQ473.SEQ472.SEQ471.BASE.CASES)
      and all(row.get("run", {}).get("returncode") in (0, 2)
              for row in runs)
  )
  counters_ready = rows_emitted and all(row_ready(row) for row in runs)
  distribution_reproduced = rows_emitted and all(
      SEQ473.SEQ472._dist_fail(row.get("summary", {}).get("distribution", {}))
      for row in runs)
  component_emitted = counters_ready and all(
      row.get("summary", {}).get("component", {})
      .get("observation_count") == EXPECTED_EVENTS
      for row in runs)
  previous_ffn_emitted = counters_ready and all(
      row.get("summary", {}).get("previous_ffn", {})
      .get("observation_count") == EXPECTED_EVENTS
      for row in runs)

  def min_summary(section: str, name: str) -> float:
    return min(
        (_num(row.get("summary", {}).get(section, {}).get(name), 1.0)
         for row in runs),
        default=1.0)

  min_ffn_input = min_summary("component", "min_ffn_input_cosine")
  min_ffn_norm = min_summary("component", "min_ffn_norm_cosine")
  min_ffn_out = min_summary("component", "min_ffn_out_cosine")
  min_cpu_norm_from_input = min_summary(
      "previous_ffn", "min_cpu_ffn_norm_from_gpu_input_cosine")
  min_gpu_norm_vs_cpu = min_summary(
      "previous_ffn", "min_gpu_ffn_norm_vs_cpu_cosine")
  min_attention_output = min_summary(
      "previous_ffn", "min_attention_output_cosine")
  min_layer_input = min_summary("previous_ffn", "min_layer_input_cosine")
  min_prev_ffn_input = min_summary("previous_ffn", "min_ffn_input_cosine")

  ffn_norm_gap = component_emitted and min_ffn_norm < COSINE_THRESHOLD
  ffn_input_clean = component_emitted and min_ffn_input >= COSINE_THRESHOLD
  ffn_out_gap = component_emitted and min_ffn_out < COSINE_THRESHOLD
  norm_input_sensitivity = (
      previous_ffn_emitted
      and min_cpu_norm_from_input < COSINE_THRESHOLD)
  gpu_norm_math_ok = (
      previous_ffn_emitted
      and min_gpu_norm_vs_cpu >= COSINE_THRESHOLD)
  attention_output_gap = (
      previous_ffn_emitted
      and min_attention_output < COSINE_THRESHOLD)
  previous_layer_input_clean = (
      previous_ffn_emitted
      and min_layer_input >= COSINE_THRESHOLD)
  previous_ffn_input_clean = (
      previous_ffn_emitted
      and min_prev_ffn_input >= COSINE_THRESHOLD)

  diagnostic_classification = (
      f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_input_attention_output_gap"
      if (ffn_norm_gap and ffn_input_clean and ffn_out_gap
          and norm_input_sensitivity and gpu_norm_math_ok
          and attention_output_gap and previous_layer_input_clean
          and previous_ffn_input_clean) else
      f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_input_sensitivity_gap"
      if (ffn_norm_gap and ffn_input_clean and ffn_out_gap
          and norm_input_sensitivity and gpu_norm_math_ok) else
      f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_math_gap"
      if ffn_norm_gap and not gpu_norm_math_ok else
      f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_gap_unclassified"
  )
  selected_next = (
      INPUT_ATTENTION_ROUTE
      if diagnostic_classification
      == f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_input_attention_output_gap"
      else INPUT_SENSITIVITY_ROUTE
      if diagnostic_classification
      == f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_input_sensitivity_gap"
      else NORM_MATH_ROUTE
      if diagnostic_classification
      == f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_math_gap"
      else CURRENT_ROUTE
  )
  checks = [
      {"name": "seq473_selected_ffn_norm_gap_gate",
       "pass": preconditions_pass},
      {"name": "ffn_norm_rows_emitted", "pass": rows_emitted},
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
      {"name": "previous_ffn_norm_diagnostics_emitted",
       "pass": previous_ffn_emitted,
       "detail": [
           row.get("summary", {}).get("previous_ffn", {}) for row in runs
       ]},
      {"name": "ffn_norm_gap_reproduced",
       "pass": ffn_norm_gap and ffn_out_gap,
       "detail": {
           "min_ffn_norm_cosine": min_ffn_norm,
           "min_ffn_out_cosine": min_ffn_out,
       }},
      {"name": "ffn_input_clean_for_norm_split",
       "pass": ffn_input_clean and previous_ffn_input_clean,
       "detail": {
           "min_component_ffn_input_cosine": min_ffn_input,
           "min_previous_ffn_input_cosine": min_prev_ffn_input,
       }},
      {"name": "gpu_ffn_norm_matches_cpu_on_live_input",
       "pass": gpu_norm_math_ok,
       "detail": {
           "min_gpu_ffn_norm_vs_cpu_cosine": min_gpu_norm_vs_cpu,
       }},
      {"name": "cpu_ffn_norm_from_live_input_reproduces_gap",
       "pass": norm_input_sensitivity,
       "detail": {
           "min_cpu_ffn_norm_from_gpu_input_cosine": min_cpu_norm_from_input,
       }},
      {"name": "attention_output_source_gap_observed",
       "pass": attention_output_gap and previous_layer_input_clean,
       "detail": {
           "min_previous_attention_output_cosine": min_attention_output,
           "min_previous_layer_input_cosine": min_layer_input,
       }},
  ]
  required = all(bool(row.get("pass")) for row in checks)
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "inputs": {
          "routes": _rel(args.routes),
          "seq473": _rel(args.seq473),
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
      "ffn_norm_gap": ffn_norm_gap,
      "ffn_input_clean": ffn_input_clean,
      "ffn_out_gap": ffn_out_gap,
      "norm_input_sensitivity": norm_input_sensitivity,
      "gpu_norm_math_ok": gpu_norm_math_ok,
      "attention_output_gap": attention_output_gap,
      "previous_layer_input_clean": previous_layer_input_clean,
      "previous_ffn_input_clean": previous_ffn_input_clean,
      "min_ffn_input_cosine": min_ffn_input,
      "min_ffn_norm_cosine": min_ffn_norm,
      "min_ffn_out_cosine": min_ffn_out,
      "min_cpu_ffn_norm_from_gpu_input_cosine": min_cpu_norm_from_input,
      "min_gpu_ffn_norm_vs_cpu_cosine": min_gpu_norm_vs_cpu,
      "min_previous_attention_output_cosine": min_attention_output,
      "min_previous_layer_input_cosine": min_layer_input,
      "min_previous_ffn_input_cosine": min_prev_ffn_input,
      "speedup_claims_allowed": False,
      "disposition": (
          f"accept_{DISPOSITION_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_gap_classification"
          if required else
          f"block_{DISPOSITION_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_gap_classification"
      ),
      "selected_next_route": selected_next if required else CURRENT_ROUTE,
      "next_route_reason": (
          "Previous FFN norm drift is RMSNorm sensitivity to clean-but-drifting "
          "FFN input; the visible source drift is previous attention output "
          "while previous layer input is clean. Root previous attention output "
          "next."
          if required and selected_next == INPUT_ATTENTION_ROUTE else
          "Previous FFN norm drift is RMSNorm sensitivity to live FFN input. "
          "Root the FFN norm input sensitivity next."
          if required and selected_next == INPUT_SENSITIVITY_ROUTE else
          "Previous GPU FFN norm math does not match CPU on live input. Root "
          "FFN norm math next."
          if required and selected_next == NORM_MATH_ROUTE else
          "Previous FFN-norm evidence is incomplete; keep this gate open."
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
      "# Seq474 Upstream Layer-Output FFN-Delta FFN-Norm Gap Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- diagnostic_classification: `{metrics['diagnostic_classification']}`",
      f"- min FFN-input cosine: `{metrics['min_ffn_input_cosine']}`",
      f"- min FFN-norm cosine: `{metrics['min_ffn_norm_cosine']}`",
      f"- min CPU-FFN-norm-from-live-input cosine: `{metrics['min_cpu_ffn_norm_from_gpu_input_cosine']}`",
      f"- min GPU-FFN-norm-vs-CPU cosine: `{metrics['min_gpu_ffn_norm_vs_cpu_cosine']}`",
      f"- min previous attention-output cosine: `{metrics['min_previous_attention_output_cosine']}`",
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
  parser.add_argument("--seq473", type=Path, default=DEFAULT_SEQ473)
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
