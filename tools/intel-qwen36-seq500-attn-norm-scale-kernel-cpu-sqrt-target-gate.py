#!/usr/bin/env python3
"""Run the shared RMSNorm CPU-sqrt target gate and classify the remaining drift."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SEQ499_GATE = (
    ROOT / "tools/intel-qwen36-seq499-attn-norm-shared-scale-cpu-sqrt-coverage-gate.py"
)
SEQ497_GATE = (
    ROOT
    / "tools/intel-qwen36-seq497-projection-q8-input-sensitivity-linear-input-source-ffn-delta-selected-gate-up-input-sensitivity-ffn-norm-input-sensitivity-attention-output-projection-q8-input-sensitivity-linear-z-source-gap-gate.py"
)
SEQ477_GATE = (
    ROOT / "tools/intel-qwen36-seq477-attention-output-projection-q8-bridge-gap-gate.py"
)
SCHEMA_VERSION = (
    "intel-qwen36-seq500-attn-norm-scale-kernel-cpu-sqrt-target-probe-v0"
)

DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_SEQ499 = (
    ROOT
    / "output/seq499-attn-norm-shared-scale-cpu-sqrt-coverage-gate-20260709Tseq499Z"
    / "metrics.json"
)
DEFAULT_SEQ497 = (
    ROOT
    / "output/seq497-projection-q8-input-sensitivity-linear-input-source-ffn-delta-selected-gate-up-input-sensitivity-ffn-norm-input-sensitivity-attention-output-projection-q8-input-sensitivity-linear-z-source-gap-gate-20260709Tseq497Z"
    / "metrics.json"
)
DEFAULT_TOKEN_INPUT_DIR = (
    ROOT
    / "output/r2-gpu-acceptance-matrix-router-both-rowblock16-26mask-20260708Tseq222Z"
    / "token-input"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "output/seq500-attn-norm-scale-kernel-cpu-sqrt-target-probe-20260709Tseq500Z"
)
DEFAULT_HOST = "local"
DEFAULT_MODEL = "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf"
DEFAULT_ENV_SCRIPT = "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh"
DEFAULT_REMOTE_ROOT = "/home/intel/intel-qwen36-gpu"

COSINE_THRESHOLD = 0.9999
MATERIAL_ABS_EPS = 1.0e-12
FIXED_ABS_EPS = 1.0e-7


def _load_module(path: Path, name: str) -> Any:
  spec = importlib.util.spec_from_file_location(name, path)
  if spec is None or spec.loader is None:
    raise SystemExit(f"cannot load module {name}: {path}")
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


SEQ499 = _load_module(SEQ499_GATE, "iq36_seq499_gate")
SEQ497 = _load_module(SEQ497_GATE, "iq36_seq497_for_seq500")
SEQ477 = _load_module(SEQ477_GATE, "iq36_seq477_for_seq500")
CURRENT_ROUTE = SEQ499.CPU_SQRT_PROBE_ROUTE
SQRT_SCALE_ROUTE = CURRENT_ROUTE.replace(
    "_linear_z_source_attn_norm_scale_kernel_cpu_sqrt_probe_gate",
    "_linear_z_source_attn_norm_scale_kernel_sqrt_gap_gate")
REDUCTION_ORDER_ROUTE = CURRENT_ROUTE.replace(
    "_linear_z_source_attn_norm_scale_kernel_cpu_sqrt_probe_gate",
    "_linear_z_source_attn_norm_scale_kernel_reduction_order_gap_gate")
CASES = SEQ477.CASES
SOURCE_LAYERS = [0]
DECODE_TOKENS = SEQ477.DECODE_TOKENS
EXPECTED_EVENTS = len(SOURCE_LAYERS) * DECODE_TOKENS
EXPECTED_COUNTER_LAYERS = SEQ477.EXPECTED_COUNTER_LAYERS
EXPECTED_COUNTER_VALUES = SEQ477.EXPECTED_COUNTER_VALUES


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


def _run_case(args: argparse.Namespace, case_id: str) -> dict[str, Any]:
  case_out = args.out_dir / "cases" / case_id
  result_path = case_out / "result.json"
  if result_path.exists():
    result = _load_json(result_path)
    smoke = result.get("smoke")
    smoke = smoke if isinstance(smoke, dict) else {}
    return {
        "case_id": case_id,
        "out_dir": _rel(case_out),
        "cmd": ["reused", str(result_path)],
        "returncode": 0,
        "stdout_bytes": 0,
        "stderr_bytes": 0,
        "result": result,
        "smoke": smoke,
    }
  cmd = [
      sys.executable,
      str(SEQ477.SMOKE_SOURCE),
      "--host", args.host,
      "--model", args.model,
      "--env-script", args.env_script,
      "--remote-root", args.remote_root,
      "--token-input-dir", str(args.token_input_dir),
      "--case-id", case_id,
      "--out-dir", str(case_out),
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
      "--opencl-cpu-sqrt-norm",
      "--distribution-ladder",
      "--full-attention-state-diff",
      "--full-core-q8-equivalence-diff",
      "--diagnostic-layer-range", "0:36",
      "--diagnostic-token-limit", str(DECODE_TOKENS),
  ]
  env = os.environ.copy()
  env.update({
      "IQ36_OPENCL_NO_QUEUE_PROFILING": "1",
      "IQ36_ATTENTION_FRONT_OUTPUT_PROJECTION_ROWBLOCK16": "1",
      "IQ36_ATTENTION_FRONT_OUTPUT_PROJECTION_ROWBLOCK16_LAYERS": (
          SEQ477.SEQ476.DIST_FIX.ROWBLOCK16_26MASK),
      "IQ36_SELECTED_SHARED_Q4_GATEUP_COMBINED": "1",
      "IQ36_SELECTED_SHARED_Q4_DOWN_COMBINED": "1",
      "IQ36_SELECTED_SHARED_Q6_DOWN_COMBINED": "1",
      "IQ36_DEFER_FFN_DOWN_FINISH_BUNDLE": "1",
      "IQ36_FULL_ATTENTION_LAYER_INPUT_PRODUCT_SOURCE": "1",
      "IQ36_FULL_ATTENTION_LAYER_INPUT_PRODUCT_CONSUMER_SOURCE": "1",
  })
  proc = subprocess.run(
      cmd,
      cwd=ROOT,
      env=env,
      capture_output=True,
      text=True,
      timeout=args.timeout_s,
      check=False,
  )
  result = _load_json(result_path) if result_path.exists() else {}
  smoke = result.get("smoke")
  smoke = smoke if isinstance(smoke, dict) else {}
  return {
      "case_id": case_id,
      "out_dir": _rel(case_out),
      "cmd": cmd,
      "returncode": proc.returncode,
      "stdout_bytes": len(proc.stdout or ""),
      "stderr_bytes": len(proc.stderr or ""),
      "result": result,
      "smoke": smoke,
  }


def _case_summary(row: dict[str, Any]) -> dict[str, Any]:
  base = SEQ497._case_summary({
      "case_id": row.get("case_id"),
      "out_dir": row.get("out_dir"),
      "source_layers": SOURCE_LAYERS,
  })
  smoke = row.get("smoke")
  smoke = smoke if isinstance(smoke, dict) else {}
  base["returncode"] = row.get("returncode")
  base["opencl_cpu_sqrt_norm_enabled"] = smoke.get(
      "opencl_cpu_sqrt_norm_enabled")
  base["required_checks_passed"] = smoke.get("required_checks_passed")
  base["source_layers_count"] = smoke.get(
      "full_attention_layer_input_product_source_layers")
  base["source_values"] = smoke.get(
      "full_attention_layer_input_product_source_values")
  base["source_misses"] = smoke.get(
      "full_attention_layer_input_product_source_misses")
  base["source_ready"] = smoke.get(
      "full_attention_layer_input_product_source_ready")
  base["consumer_layers_count"] = smoke.get(
      "full_attention_layer_input_product_consumer_source_layers")
  base["consumer_values"] = smoke.get(
      "full_attention_layer_input_product_consumer_source_values")
  base["consumer_misses"] = smoke.get(
      "full_attention_layer_input_product_consumer_source_misses")
  base["consumer_ready"] = smoke.get(
      "full_attention_layer_input_product_consumer_source_ready")
  base["speedup_claims_allowed"] = smoke.get("speedup_claims_allowed")
  return base


def _all_observed(rows: list[dict[str, Any]], group: str) -> bool:
  return all(
      row.get(group, {}).get("observation_count") == EXPECTED_EVENTS
      for row in rows)


def _min_case(rows: list[dict[str, Any]], group: str, key: str) -> float:
  return min((_num(row.get(group, {}).get(key), 1.0) for row in rows),
             default=1.0)


def _max_case(rows: list[dict[str, Any]], group: str, key: str) -> float:
  return max((_num(row.get(group, {}).get(key)) for row in rows), default=0.0)


def compute(args: argparse.Namespace) -> dict[str, Any]:
  routes = _load_json(args.routes)
  seq499 = _load_json(args.seq499)
  seq497 = _load_json(args.seq497)
  runs = [_run_case(args, case_id) for case_id in CASES]
  rows = [_case_summary(row) for row in runs]
  classification = str(seq499.get("diagnostic_classification"))
  preconditions_pass = (
      seq499.get("required_checks_passed") is True
      and seq499.get("selected_next_route") == CURRENT_ROUTE
      and seq499.get("cpu_sqrt_shared_scale_coverage") is True
      and classification.endswith("_attn_norm_scale_kernel_cpu_sqrt_probe_ready")
      and _has_candidate(routes, 499, str(seq499.get("disposition")))
      and _has_switch(routes, f"select_{CURRENT_ROUTE}", 499)
  )
  rows_emitted = (
      len(rows) == len(CASES)
      and all(row.get("opencl_cpu_sqrt_norm_enabled") is True for row in rows)
      and _all_observed(rows, "preconv")
      and _all_observed(rows, "projection")
  )
  counters_ready = rows_emitted and all(
      row.get("source_layers_count") == EXPECTED_COUNTER_LAYERS
      and row.get("source_values") == EXPECTED_COUNTER_VALUES
      and row.get("source_misses") == 0
      and row.get("source_ready") is True
      and row.get("consumer_layers_count") == EXPECTED_COUNTER_LAYERS
      and row.get("consumer_values") == EXPECTED_COUNTER_VALUES
      and row.get("consumer_misses") == 0
      and row.get("consumer_ready") is True
      and row.get("speedup_claims_allowed") is False
      for row in rows)

  base_abs = _num(seq497.get("max_abs_diffs", {}).get("gpu_attn_norm_vs_cpu"))
  gpu_norm_min = _min_case(rows, "preconv", "min_gpu_attn_norm_vs_cpu_cosine")
  gpu_norm_abs = _max_case(
      rows, "preconv", "max_gpu_attn_norm_vs_cpu_abs_diff")
  cpu_norm_abs = _max_case(
      rows, "preconv", "max_attn_norm_from_gpu_input_abs_diff")
  z_from_norm_abs = _max_case(
      rows, "preconv", "max_z_from_gpu_attn_norm_abs_diff")
  gpu_z_abs = _max_case(rows, "preconv", "max_gpu_z_vs_cpu_abs_diff")
  final_abs = _max_case(rows, "projection",
                        "max_gpu_final_projection_vs_native_abs_diff")
  z_projection_abs = _max_case(
      rows, "projection", "max_native_delta_gpu_z_projection_vs_native_abs_diff")
  sqrt_clears_gpu_norm = (
      rows_emitted
      and gpu_norm_min >= COSINE_THRESHOLD
      and gpu_norm_abs <= FIXED_ABS_EPS)
  reduction_order_remains = (
      rows_emitted
      and gpu_norm_min >= COSINE_THRESHOLD
      and gpu_norm_abs > MATERIAL_ABS_EPS
      and cpu_norm_abs == 0.0
      and gpu_z_abs == 0.0)
  classified = preconditions_pass and counters_ready and (
      sqrt_clears_gpu_norm or reduction_order_remains)
  selected_next = (
      SQRT_SCALE_ROUTE if sqrt_clears_gpu_norm else
      REDUCTION_ORDER_ROUTE if reduction_order_remains else
      CURRENT_ROUTE)
  checks = [
      {"name": "seq499_cpu_sqrt_probe_gate", "pass": preconditions_pass},
      {"name": "target_cpu_sqrt_rows_emitted",
       "pass": rows_emitted,
       "detail": rows},
      {"name": "product_source_consumer_counters_ready",
       "pass": counters_ready},
      {"name": "cpu_sqrt_probe_classified",
       "pass": classified,
       "detail": {
           "base_gpu_attn_norm_abs": base_abs,
           "cpu_sqrt_gpu_attn_norm_abs": gpu_norm_abs,
           "sqrt_clears_gpu_norm": sqrt_clears_gpu_norm,
           "reduction_order_remains": reduction_order_remains,
       }},
  ]
  required = all(bool(row.get("pass")) for row in checks)
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "inputs": {
          "routes": _rel(args.routes),
          "seq499": _rel(args.seq499),
          "seq497": _rel(args.seq497),
          "token_input_dir": _rel(args.token_input_dir),
      },
      "runs": [{
          "case_id": row.get("case_id"),
          "out_dir": row.get("out_dir"),
          "returncode": row.get("returncode"),
          "opencl_cpu_sqrt_norm_enabled": row.get(
              "opencl_cpu_sqrt_norm_enabled"),
      } for row in rows],
      "checks": checks,
      "required_checks_passed": required,
      "diagnostic_classification": (
          classification.replace(
              "_attn_norm_scale_kernel_cpu_sqrt_probe_ready",
              "_attn_norm_scale_kernel_sqrt_gap")
          if required and sqrt_clears_gpu_norm else
          classification.replace(
              "_attn_norm_scale_kernel_cpu_sqrt_probe_ready",
              "_attn_norm_scale_kernel_reduction_order_gap")
          if required and reduction_order_remains else
          f"{classification}_unclassified"),
      "sqrt_clears_gpu_norm": sqrt_clears_gpu_norm,
      "reduction_order_remains": reduction_order_remains,
      "base_gpu_attn_norm_abs": base_abs,
      "cpu_sqrt_gpu_attn_norm_abs": gpu_norm_abs,
      "max_abs_diffs": {
          "attn_norm_from_gpu_input": cpu_norm_abs,
          "gpu_attn_norm_vs_cpu": gpu_norm_abs,
          "z_from_gpu_attn_norm": z_from_norm_abs,
          "gpu_z_vs_cpu": gpu_z_abs,
          "final_projection": final_abs,
          "z_projection": z_projection_abs,
      },
      "speedup_claims_allowed": False,
      "disposition": (
          f"accept_{classification}_target_probe"
          if required else f"block_{classification}_target_probe"
      ),
      "selected_next_route": selected_next if required else CURRENT_ROUTE,
      "next_route_reason": (
          "The CPU-sqrt shared-scale probe clears the GPU attention-norm drift; "
          "root the shared scale sqrt expression next."
          if required and sqrt_clears_gpu_norm else
          "The CPU-sqrt shared-scale probe leaves material GPU attention-norm "
          "drift while CPU replay and z math remain clean; root the shared "
          "scale-kernel local-partial reduction order next."
          if required and reduction_order_remains else
          "CPU-sqrt shared-scale target evidence is incomplete; keep this probe open."
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
  a = metrics["max_abs_diffs"]
  lines = [
      "# Seq500 CPU-Sqrt Shared-Scale Target Probe",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- diagnostic_classification: `{metrics['diagnostic_classification']}`",
      f"- base/cpu-sqrt gpu_attn_norm max abs: `{metrics['base_gpu_attn_norm_abs']}` / `{metrics['cpu_sqrt_gpu_attn_norm_abs']}`",
      f"- cpu-norm/z-from-norm/gpu-z max abs: `{a['attn_norm_from_gpu_input']}` / `{a['z_from_gpu_attn_norm']}` / `{a['gpu_z_vs_cpu']}`",
      f"- projection final/z max abs: `{a['final_projection']}` / `{a['z_projection']}`",
      f"- sqrt_clears_gpu_norm: `{str(metrics['sqrt_clears_gpu_norm']).lower()}`",
      f"- reduction_order_remains: `{str(metrics['reduction_order_remains']).lower()}`",
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
  parser.add_argument("--seq499", type=Path, default=DEFAULT_SEQ499)
  parser.add_argument("--seq497", type=Path, default=DEFAULT_SEQ497)
  parser.add_argument("--token-input-dir", type=Path,
                      default=DEFAULT_TOKEN_INPUT_DIR)
  parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
  parser.add_argument("--target", "--host", dest="host", metavar="TARGET", default=DEFAULT_HOST)
  parser.add_argument("--model", default=DEFAULT_MODEL)
  parser.add_argument("--env-script", default=DEFAULT_ENV_SCRIPT)
  parser.add_argument("--staging-root", "--remote-root", dest="remote_root", metavar="PATH", default=DEFAULT_REMOTE_ROOT)
  parser.add_argument("--timeout-s", type=int, default=1800)
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
