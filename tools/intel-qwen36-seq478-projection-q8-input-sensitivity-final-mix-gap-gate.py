#!/usr/bin/env python3
"""Classify seq477 projection Q8 input sensitivity across final-mix inputs."""

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
SEQ477_GATE = (
    ROOT
    / "tools/intel-qwen36-seq477-attention-output-projection-q8-bridge-gap-gate.py"
)
SCHEMA_VERSION = (
    "intel-qwen36-seq478-projection-q8-input-sensitivity-final-mix-gap-gate-v0"
)

DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_SEQ477 = (
    ROOT
    / "output/seq477-attention-output-projection-q8-bridge-gap-gate-20260709Tseq477Z"
    / "metrics.json"
)
DEFAULT_TOKEN_INPUT_DIR = (
    ROOT
    / "output/r2-gpu-acceptance-matrix-router-both-rowblock16-26mask-20260708Tseq222Z"
    / "token-input"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "output/seq478-projection-q8-input-sensitivity-final-mix-gap-gate-20260709Tseq478Z"
)
DEFAULT_HOST = "local"
DEFAULT_MODEL = "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf"
DEFAULT_ENV_SCRIPT = "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh"
DEFAULT_REMOTE_ROOT = "/home/intel/intel-qwen36-gpu"

COSINE_THRESHOLD = 0.9999


def _load_module(path: Path, name: str) -> Any:
  spec = importlib.util.spec_from_file_location(name, path)
  if spec is None or spec.loader is None:
    raise SystemExit(f"cannot load module {name}: {path}")
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


SEQ477 = _load_module(SEQ477_GATE, "iq36_seq477_gate")
CURRENT_ROUTE = SEQ477.Q8_INPUT_SENSITIVITY_ROUTE
LINEAR_DELTA_ROUTE = CURRENT_ROUTE.replace(
    "_attention_output_projection_q8_input_sensitivity_gap_gate",
    "_attention_output_projection_q8_input_sensitivity_linear_delta_gap_gate")
LINEAR_Z_ROUTE = CURRENT_ROUTE.replace(
    "_attention_output_projection_q8_input_sensitivity_gap_gate",
    "_attention_output_projection_q8_input_sensitivity_linear_z_gap_gate")
LINEAR_DELTA_Z_ROUTE = CURRENT_ROUTE.replace(
    "_attention_output_projection_q8_input_sensitivity_gap_gate",
    "_attention_output_projection_q8_input_sensitivity_linear_delta_z_gap_gate")
FINAL_KERNEL_ROUTE = CURRENT_ROUTE.replace(
    "_attention_output_projection_q8_input_sensitivity_gap_gate",
    "_attention_output_projection_q8_input_sensitivity_linear_final_kernel_gap_gate")
DIAG_PREFIX = SEQ477.DIAG_PREFIX
DISPOSITION_PREFIX = SEQ477.DISPOSITION_PREFIX
CASES = SEQ477.CASES
PREVIOUS_LAYERS = SEQ477.PREVIOUS_LAYERS
DECODE_TOKENS = SEQ477.DECODE_TOKENS
EXPECTED_EVENTS = SEQ477.EXPECTED_EVENTS
EXPECTED_COUNTER_LAYERS = SEQ477.EXPECTED_COUNTER_LAYERS
EXPECTED_COUNTER_VALUES = SEQ477.EXPECTED_COUNTER_VALUES

INPUT_METRICS = [
    "gpu_final_input_vs_native",
    "gpu_delta_native_z_input_vs_native",
    "native_delta_gpu_z_input_vs_native",
    "gpu_delta_gpu_z_input_vs_native",
]
PROJECTION_METRICS = [
    "gpu_final_projection_vs_native",
    "gpu_delta_native_z_projection_vs_native",
    "native_delta_gpu_z_projection_vs_native",
    "gpu_delta_gpu_z_projection_vs_native",
]
Q8_PREFIXES = [
    "gpu_final",
    "gpu_delta_native_z",
    "native_delta_gpu_z",
    "gpu_delta_gpu_z",
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


def _projection_input_sensitivity_summary(
    smoke: dict[str, Any]) -> dict[str, Any]:
  steps = smoke.get("linear_projection_input_sensitivity_diff_by_step")
  steps = steps if isinstance(steps, list) else []
  out = SEQ477._metric_summary(
      steps, INPUT_METRICS + PROJECTION_METRICS,
      "gpu_final_projection_vs_native")
  selected = set(PREVIOUS_LAYERS)
  for prefix in Q8_PREFIXES:
    out[f"{prefix}_q8_observation_count"] = 0
    out[f"{prefix}_max_q8_qs_mismatch_count"] = 0
    out[f"{prefix}_max_q8_bsums_mismatch_count"] = 0
    out[f"{prefix}_max_q8_d_abs_diff"] = 0.0
    out[f"{prefix}_max_q8_d_rmse"] = 0.0
  for step in steps:
    if not isinstance(step, dict):
      continue
    layers = step.get("layers")
    layers = layers if isinstance(layers, list) else []
    for row in layers:
      if not isinstance(row, dict) or row.get("layer") not in selected:
        continue
      for prefix in Q8_PREFIXES:
        if row.get(f"{prefix}_q8_available") is not True:
          continue
        out[f"{prefix}_q8_observation_count"] += 1
        out[f"{prefix}_max_q8_qs_mismatch_count"] = max(
            out[f"{prefix}_max_q8_qs_mismatch_count"],
            int(_num(row.get(f"{prefix}_q8_qs_mismatch_count"))))
        out[f"{prefix}_max_q8_bsums_mismatch_count"] = max(
            out[f"{prefix}_max_q8_bsums_mismatch_count"],
            int(_num(row.get(f"{prefix}_q8_bsums_mismatch_count"))))
        out[f"{prefix}_max_q8_d_abs_diff"] = max(
            out[f"{prefix}_max_q8_d_abs_diff"],
            _num(row.get(f"{prefix}_q8_d_max_abs_diff")))
        out[f"{prefix}_max_q8_d_rmse"] = max(
            out[f"{prefix}_max_q8_d_rmse"],
            _num(row.get(f"{prefix}_q8_d_rmse")))
  return out


def _case_summary(row: dict[str, Any]) -> dict[str, Any]:
  summary = SEQ477._case_summary(row)
  smoke = row.get("smoke")
  smoke = smoke if isinstance(smoke, dict) else {}
  summary["projection_input_sensitivity"] = (
      _projection_input_sensitivity_summary(smoke))
  return summary


def _row_ready(row: dict[str, Any]) -> bool:
  summary = row.get("summary", {})
  return (
      bool(summary)
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


def _min_sensitivity(rows: list[dict[str, Any]], name: str) -> float:
  return min(
      (_num(row.get("summary", {}).get("projection_input_sensitivity", {})
            .get(f"min_{name}_cosine"), 1.0)
       for row in rows),
      default=1.0)


def _max_sensitivity(rows: list[dict[str, Any]], name: str) -> float:
  return max(
      (_num(row.get("summary", {}).get("projection_input_sensitivity", {})
            .get(f"max_{name}_abs_diff"))
       for row in rows),
      default=0.0)


def _max_q8(rows: list[dict[str, Any]], prefix: str, suffix: str) -> float:
  return max(
      (_num(row.get("summary", {}).get("projection_input_sensitivity", {})
            .get(f"{prefix}_max_q8_{suffix}"))
       for row in rows),
      default=0.0)


def _q8_mismatch_observed(rows: list[dict[str, Any]], prefix: str) -> bool:
  return (
      _max_q8(rows, prefix, "qs_mismatch_count") > 0
      or _max_q8(rows, prefix, "bsums_mismatch_count") > 0
      or _max_q8(rows, prefix, "d_abs_diff") > 0.0)


def compute(args: argparse.Namespace) -> dict[str, Any]:
  routes = _load_json(args.routes)
  seq477 = _load_json(args.seq477)
  runs = [SEQ477._run_case(args, case_id) for case_id in CASES]
  rows = [{
      "case_id": row.get("case_id"),
      "run": {
          "cmd": row.get("cmd"),
          "returncode": row.get("returncode"),
          "stdout_bytes": row.get("stdout_bytes"),
          "stderr_bytes": row.get("stderr_bytes"),
          "out_dir": row.get("out_dir"),
      },
      "summary": _case_summary(row),
  } for row in runs]

  preconditions_pass = (
      seq477.get("required_checks_passed") is True
      and seq477.get("selected_next_route") == CURRENT_ROUTE
      and seq477.get("diagnostic_classification")
      == f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_input_attention_output_projection_q8_input_sensitivity_gap"
      and _has_candidate(routes, 477, str(seq477.get("disposition")))
      and _has_switch(routes, f"select_{CURRENT_ROUTE}", 477)
  )
  rows_emitted = (
      len(rows) == len(CASES)
      and all(row.get("summary", {}).get("projection_input_sensitivity", {})
              .get("observation_count", 0) > 0 for row in rows)
  )
  counters_ready = rows_emitted and all(_row_ready(row) for row in rows)
  distribution_reproduced = rows_emitted and all(
      SEQ477.SEQ476.SEQ475.SEQ474.SEQ473.SEQ472._dist_fail(
          row.get("summary", {}).get("distribution", {}))
      for row in rows)
  diagnostics_emitted = counters_ready and all(
      row.get("summary", {}).get("projection_source", {})
      .get("observation_count") == EXPECTED_EVENTS
      and row.get("summary", {}).get("projection_source", {})
      .get("q8_observation_count") == EXPECTED_EVENTS
      and row.get("summary", {}).get("projection_input_sensitivity", {})
      .get("observation_count") == EXPECTED_EVENTS
      and all(row.get("summary", {}).get("projection_input_sensitivity", {})
              .get(f"{prefix}_q8_observation_count") == EXPECTED_EVENTS
              for prefix in Q8_PREFIXES)
      for row in rows)

  min_metrics = {
      name: _min_sensitivity(rows, name)
      for name in INPUT_METRICS + PROJECTION_METRICS
  }
  max_abs_metrics = {
      name: _max_sensitivity(rows, name)
      for name in INPUT_METRICS + PROJECTION_METRICS
  }
  q8_mismatches = {
      prefix: {
          "max_qs_mismatch_count": int(_max_q8(
              rows, prefix, "qs_mismatch_count")),
          "max_bsums_mismatch_count": int(_max_q8(
              rows, prefix, "bsums_mismatch_count")),
          "max_d_abs_diff": _max_q8(rows, prefix, "d_abs_diff"),
          "max_d_rmse": _max_q8(rows, prefix, "d_rmse"),
          "mismatch_observed": _q8_mismatch_observed(rows, prefix),
      }
      for prefix in Q8_PREFIXES
  }

  def proj_gap(name: str) -> bool:
    return diagnostics_emitted and min_metrics[name] < COSINE_THRESHOLD

  gpu_final_projection_gap = proj_gap("gpu_final_projection_vs_native")
  delta_variant_gap = (
      proj_gap("gpu_delta_native_z_projection_vs_native")
      and q8_mismatches["gpu_delta_native_z"]["mismatch_observed"])
  z_variant_gap = (
      proj_gap("native_delta_gpu_z_projection_vs_native")
      and q8_mismatches["native_delta_gpu_z"]["mismatch_observed"])
  combined_variant_gap = (
      proj_gap("gpu_delta_gpu_z_projection_vs_native")
      and q8_mismatches["gpu_delta_gpu_z"]["mismatch_observed"])
  gpu_final_q8_gap = (
      gpu_final_projection_gap
      and q8_mismatches["gpu_final"]["mismatch_observed"])

  linear_delta_gap = (
      gpu_final_q8_gap and delta_variant_gap and not z_variant_gap)
  linear_z_gap = (
      gpu_final_q8_gap and z_variant_gap and not delta_variant_gap)
  linear_delta_z_gap = (
      gpu_final_q8_gap
      and ((delta_variant_gap and z_variant_gap)
           or (combined_variant_gap
               and not delta_variant_gap
               and not z_variant_gap)))
  linear_final_kernel_gap = (
      gpu_final_projection_gap
      and not delta_variant_gap
      and not z_variant_gap
      and not combined_variant_gap)

  diagnostic_classification = (
      f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_input_attention_output_projection_q8_input_sensitivity_linear_delta_gap"
      if linear_delta_gap else
      f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_input_attention_output_projection_q8_input_sensitivity_linear_z_gap"
      if linear_z_gap else
      f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_input_attention_output_projection_q8_input_sensitivity_linear_delta_z_gap"
      if linear_delta_z_gap else
      f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_input_attention_output_projection_q8_input_sensitivity_linear_final_kernel_gap"
      if linear_final_kernel_gap else
      f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_input_attention_output_projection_q8_input_sensitivity_gap_unclassified"
  )
  selected_next = (
      LINEAR_DELTA_ROUTE
      if linear_delta_gap else
      LINEAR_Z_ROUTE
      if linear_z_gap else
      LINEAR_DELTA_Z_ROUTE
      if linear_delta_z_gap else
      FINAL_KERNEL_ROUTE
      if linear_final_kernel_gap else
      CURRENT_ROUTE
  )
  checks = [
      {"name": "seq477_selected_q8_input_sensitivity_gate",
       "pass": preconditions_pass},
      {"name": "final_mix_rows_emitted", "pass": rows_emitted},
      {"name": "source_and_consumer_counters_ready",
       "pass": counters_ready,
       "detail": [row.get("summary", {}) for row in rows]},
      {"name": "distribution_failure_reproduced",
       "pass": distribution_reproduced,
       "detail": [
           row.get("summary", {}).get("distribution", {}) for row in rows
       ]},
      {"name": "projection_input_sensitivity_diagnostics_emitted",
       "pass": diagnostics_emitted,
       "detail": [
           row.get("summary", {}).get("projection_input_sensitivity", {})
           for row in rows
       ]},
      {"name": "gpu_final_input_sensitivity_reproduced",
       "pass": gpu_final_q8_gap,
       "detail": {
           "min_gpu_final_projection_vs_native_cosine": (
               min_metrics["gpu_final_projection_vs_native"]),
           "gpu_final_q8": q8_mismatches["gpu_final"],
       }},
      {"name": "projection_input_sensitivity_classified",
       "pass": selected_next != CURRENT_ROUTE,
       "detail": {
           "linear_delta_gap": linear_delta_gap,
           "linear_z_gap": linear_z_gap,
           "linear_delta_z_gap": linear_delta_z_gap,
           "linear_final_kernel_gap": linear_final_kernel_gap,
           "min_cosines": min_metrics,
           "q8_mismatches": q8_mismatches,
       }},
  ]
  required = all(bool(row.get("pass")) for row in checks)
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "inputs": {
          "routes": _rel(args.routes),
          "seq477": _rel(args.seq477),
          "token_input_dir": _rel(args.token_input_dir),
          "host": args.host,
          "previous_layers": PREVIOUS_LAYERS,
          "decode_tokens": DECODE_TOKENS,
      },
      "runs": rows,
      "checks": checks,
      "required_checks_passed": required,
      "diagnostic_classification": diagnostic_classification,
      "gpu_final_q8_gap": gpu_final_q8_gap,
      "linear_delta_gap": linear_delta_gap,
      "linear_z_gap": linear_z_gap,
      "linear_delta_z_gap": linear_delta_z_gap,
      "linear_final_kernel_gap": linear_final_kernel_gap,
      "min_cosines": min_metrics,
      "max_abs_diffs": max_abs_metrics,
      "q8_mismatches": q8_mismatches,
      "speedup_claims_allowed": False,
      "disposition": (
          f"accept_{DISPOSITION_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_input_attention_output_projection_q8_input_sensitivity_final_mix_gap_classification"
          if required else
          f"block_{DISPOSITION_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_input_attention_output_projection_q8_input_sensitivity_final_mix_gap_classification"
      ),
      "selected_next_route": selected_next if required else CURRENT_ROUTE,
      "next_route_reason": (
          "The CPU recompute with GPU delta and native z reproduces the "
          "projection Q8 input-sensitivity gap, while native delta with GPU z "
          "stays clean. Root the linear delta path next."
          if required and selected_next == LINEAR_DELTA_ROUTE else
          "The CPU recompute with native delta and GPU z reproduces the "
          "projection Q8 input-sensitivity gap, while GPU delta with native z "
          "stays clean. Root the linear z path next."
          if required and selected_next == LINEAR_Z_ROUTE else
          "Both final-mix component substitutions, or their combined CPU "
          "recompute, reproduce the projection Q8 input-sensitivity gap. Root "
          "the coupled linear delta/z path next."
          if required and selected_next == LINEAR_DELTA_Z_ROUTE else
          "The actual GPU final input reproduces the projection Q8 gap, but "
          "the CPU final-mix recomputes from GPU delta/z stay clean. Root the "
          "linear final normalization/kernel path next."
          if required and selected_next == FINAL_KERNEL_ROUTE else
          "Final-mix projection input-sensitivity evidence is incomplete; keep "
          "this gate open."
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
  c = metrics["min_cosines"]
  q8 = metrics["q8_mismatches"]
  lines = [
      "# Seq478 Projection Q8 Input-Sensitivity Final-Mix Gap Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- diagnostic_classification: `{metrics['diagnostic_classification']}`",
      f"- min projection cosines final/delta/z/both: `{c['gpu_final_projection_vs_native']}` / `{c['gpu_delta_native_z_projection_vs_native']}` / `{c['native_delta_gpu_z_projection_vs_native']}` / `{c['gpu_delta_gpu_z_projection_vs_native']}`",
      f"- min input cosines final/delta/z/both: `{c['gpu_final_input_vs_native']}` / `{c['gpu_delta_native_z_input_vs_native']}` / `{c['native_delta_gpu_z_input_vs_native']}` / `{c['gpu_delta_gpu_z_input_vs_native']}`",
      f"- q8 mismatches final qs/bsums/d: `{q8['gpu_final']['max_qs_mismatch_count']}` / `{q8['gpu_final']['max_bsums_mismatch_count']}` / `{q8['gpu_final']['max_d_abs_diff']}`",
      f"- q8 mismatches delta/z/both qs: `{q8['gpu_delta_native_z']['max_qs_mismatch_count']}` / `{q8['native_delta_gpu_z']['max_qs_mismatch_count']}` / `{q8['gpu_delta_gpu_z']['max_qs_mismatch_count']}`",
      f"- linear_delta_gap: `{str(metrics['linear_delta_gap']).lower()}`",
      f"- linear_z_gap: `{str(metrics['linear_z_gap']).lower()}`",
      f"- linear_delta_z_gap: `{str(metrics['linear_delta_z_gap']).lower()}`",
      f"- linear_final_kernel_gap: `{str(metrics['linear_final_kernel_gap']).lower()}`",
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
  parser.add_argument("--seq477", type=Path, default=DEFAULT_SEQ477)
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
