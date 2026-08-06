#!/usr/bin/env python3
"""Run the serial shared RMSNorm scale-kernel target gate."""

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
SEQ500_GATE = (
    ROOT / "tools/intel-qwen36-seq500-attn-norm-scale-kernel-cpu-sqrt-target-gate.py"
)
SCHEMA_VERSION = (
    "intel-qwen36-seq501-attn-norm-scale-kernel-reduction-order-target-gate-v0"
)

DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_SEQ500 = (
    ROOT
    / "output/seq500-attn-norm-scale-kernel-cpu-sqrt-target-probe-20260709Tseq500Z"
    / "metrics.json"
)
DEFAULT_TOKEN_INPUT_DIR = (
    ROOT
    / "output/r2-gpu-acceptance-matrix-router-both-rowblock16-26mask-20260708Tseq222Z"
    / "token-input"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "output/seq501-attn-norm-scale-kernel-reduction-order-target-gate-20260709Tseq501Z"
)
DEFAULT_HOST = "local"
DEFAULT_MODEL = "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf"
DEFAULT_ENV_SCRIPT = "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh"
DEFAULT_REMOTE_ROOT = "/home/intel/intel-qwen36-gpu"

FIXED_ABS_EPS = 1.0e-7


def _load_module(path: Path, name: str) -> Any:
  spec = importlib.util.spec_from_file_location(name, path)
  if spec is None or spec.loader is None:
    raise SystemExit(f"cannot load module {name}: {path}")
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


SEQ500 = _load_module(SEQ500_GATE, "iq36_seq500_gate")
CURRENT_ROUTE = SEQ500.REDUCTION_ORDER_ROUTE
REDUCTION_FIX_ROUTE = CURRENT_ROUTE.replace(
    "_linear_z_source_attn_norm_scale_kernel_reduction_order_gap_gate",
    "_linear_z_source_attn_norm_scale_kernel_reduction_order_fix_gate")
REDUCTION_STILL_OPEN_ROUTE = CURRENT_ROUTE.replace(
    "_linear_z_source_attn_norm_scale_kernel_reduction_order_gap_gate",
    "_linear_z_source_attn_norm_scale_kernel_reduction_order_unresolved_gate")


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
    return {
        "case_id": case_id,
        "out_dir": _rel(case_out),
        "returncode": 0,
        "smoke": smoke if isinstance(smoke, dict) else {},
    }
  cmd = [
      sys.executable, str(SEQ500.SEQ477.SMOKE_SOURCE),
      "--host", args.host,
      "--model", args.model,
      "--env-script", args.env_script,
      "--remote-root", args.remote_root,
      "--token-input-dir", str(args.token_input_dir),
      "--case-id", case_id,
      "--out-dir", str(case_out),
      "--device-substring", "B390",
      "--repeat", "1",
      "--decode-tokens", str(SEQ500.DECODE_TOKENS),
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
      "--full-core-q8-equivalence-diff",
      "--diagnostic-layer-range", "0:36",
      "--diagnostic-token-limit", str(SEQ500.DECODE_TOKENS),
  ]
  env = os.environ.copy()
  env.update({
      "IQ36_OPENCL_NO_QUEUE_PROFILING": "1",
      "IQ36_ATTENTION_FRONT_OUTPUT_PROJECTION_ROWBLOCK16": "1",
      "IQ36_ATTENTION_FRONT_OUTPUT_PROJECTION_ROWBLOCK16_LAYERS": (
          SEQ500.SEQ477.SEQ476.DIST_FIX.ROWBLOCK16_26MASK),
      "IQ36_SELECTED_SHARED_Q4_GATEUP_COMBINED": "1",
      "IQ36_SELECTED_SHARED_Q4_DOWN_COMBINED": "1",
      "IQ36_SELECTED_SHARED_Q6_DOWN_COMBINED": "1",
      "IQ36_DEFER_FFN_DOWN_FINISH_BUNDLE": "1",
      "IQ36_FULL_ATTENTION_LAYER_INPUT_PRODUCT_SOURCE": "1",
      "IQ36_FULL_ATTENTION_LAYER_INPUT_PRODUCT_CONSUMER_SOURCE": "1",
  })
  proc = subprocess.run(
      cmd, cwd=ROOT, env=env, capture_output=True, text=True,
      timeout=args.timeout_s, check=False)
  result = _load_json(result_path) if result_path.exists() else {}
  smoke = result.get("smoke")
  return {
      "case_id": case_id,
      "out_dir": _rel(case_out),
      "returncode": proc.returncode,
      "smoke": smoke if isinstance(smoke, dict) else {},
  }


def _summary(row: dict[str, Any]) -> dict[str, Any]:
  base = SEQ500._case_summary({
      "case_id": row.get("case_id"),
      "out_dir": row.get("out_dir"),
      "smoke": row.get("smoke"),
      "returncode": row.get("returncode"),
  })
  return base


def compute(args: argparse.Namespace) -> dict[str, Any]:
  routes = _load_json(args.routes)
  seq500 = _load_json(args.seq500)
  runs = [_run_case(args, case_id) for case_id in SEQ500.CASES]
  rows = [_summary(row) for row in runs]
  classification = str(seq500.get("diagnostic_classification"))
  preconditions_pass = (
      seq500.get("required_checks_passed") is True
      and seq500.get("selected_next_route") == CURRENT_ROUTE
      and seq500.get("reduction_order_remains") is True
      and _has_candidate(routes, 500, str(seq500.get("disposition")))
      and _has_switch(routes, f"select_{CURRENT_ROUTE}", 500)
  )
  rows_emitted = (
      len(rows) == len(SEQ500.CASES)
      and all(row.get("opencl_cpu_sqrt_norm_enabled") is not True for row in rows)
      and SEQ500._all_observed(rows, "preconv")
      and SEQ500._all_observed(rows, "projection")
  )
  counters_ready = rows_emitted and all(
      row.get("source_layers_count") == SEQ500.EXPECTED_COUNTER_LAYERS
      and row.get("source_values") == SEQ500.EXPECTED_COUNTER_VALUES
      and row.get("source_misses") == 0
      and row.get("source_ready") is True
      and row.get("consumer_layers_count") == SEQ500.EXPECTED_COUNTER_LAYERS
      and row.get("consumer_values") == SEQ500.EXPECTED_COUNTER_VALUES
      and row.get("consumer_misses") == 0
      and row.get("consumer_ready") is True
      and row.get("speedup_claims_allowed") is False
      for row in rows)
  gpu_norm_abs = SEQ500._max_case(
      rows, "preconv", "max_gpu_attn_norm_vs_cpu_abs_diff")
  cpu_norm_abs = SEQ500._max_case(
      rows, "preconv", "max_attn_norm_from_gpu_input_abs_diff")
  z_from_norm_abs = SEQ500._max_case(
      rows, "preconv", "max_z_from_gpu_attn_norm_abs_diff")
  gpu_z_abs = SEQ500._max_case(rows, "preconv", "max_gpu_z_vs_cpu_abs_diff")
  final_abs = SEQ500._max_case(
      rows, "projection", "max_gpu_final_projection_vs_native_abs_diff")
  reduction_order_fixed = (
      rows_emitted
      and cpu_norm_abs == 0.0
      and gpu_z_abs == 0.0
      and gpu_norm_abs <= FIXED_ABS_EPS)
  still_open = (
      rows_emitted
      and cpu_norm_abs == 0.0
      and gpu_z_abs == 0.0
      and gpu_norm_abs > FIXED_ABS_EPS)
  required = preconditions_pass and counters_ready and (
      reduction_order_fixed or still_open)
  selected = (
      REDUCTION_FIX_ROUTE if reduction_order_fixed else
      REDUCTION_STILL_OPEN_ROUTE if still_open else CURRENT_ROUTE)
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "inputs": {
          "routes": _rel(args.routes),
          "seq500": _rel(args.seq500),
          "token_input_dir": _rel(args.token_input_dir),
      },
      "runs": [{
          "case_id": row.get("case_id"),
          "out_dir": row.get("out_dir"),
          "returncode": row.get("returncode"),
      } for row in rows],
      "checks": [
          {"name": "seq500_reduction_order_gate", "pass": preconditions_pass},
          {"name": "target_serial_scale_rows_emitted", "pass": rows_emitted},
          {"name": "product_source_consumer_counters_ready", "pass": counters_ready},
          {"name": "serial_scale_kernel_classified",
           "pass": reduction_order_fixed or still_open,
           "detail": {
               "gpu_attn_norm_abs": gpu_norm_abs,
               "reduction_order_fixed": reduction_order_fixed,
               "still_open": still_open,
           }},
      ],
      "required_checks_passed": required,
      "diagnostic_classification": (
          classification.replace(
              "_attn_norm_scale_kernel_reduction_order_gap",
              "_attn_norm_scale_kernel_reduction_order_fix")
          if required and reduction_order_fixed else
          classification.replace(
              "_attn_norm_scale_kernel_reduction_order_gap",
              "_attn_norm_scale_kernel_reduction_order_unresolved")
          if required and still_open else
          f"{classification}_unclassified"),
      "reduction_order_fixed": reduction_order_fixed,
      "reduction_order_still_open": still_open,
      "max_abs_diffs": {
          "attn_norm_from_gpu_input": cpu_norm_abs,
          "gpu_attn_norm_vs_cpu": gpu_norm_abs,
          "z_from_gpu_attn_norm": z_from_norm_abs,
          "gpu_z_vs_cpu": gpu_z_abs,
          "final_projection": final_abs,
      },
      "speedup_claims_allowed": False,
      "disposition": (
          f"accept_{classification}_serial_scale_target_gate"
          if required else f"block_{classification}_serial_scale_target_gate"
      ),
      "selected_next_route": selected if required else CURRENT_ROUTE,
      "next_route_reason": (
          "Serializing the shared RMSNorm scale reduction clears the selected "
          "attention-norm drift. Re-run router distribution on the product path next."
          if required and reduction_order_fixed else
          "Serializing the shared RMSNorm scale reduction does not clear the "
          "selected drift; keep this root open."
          if required and still_open else
          "Serial shared-scale target evidence is incomplete; keep this gate open."
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
      "# Seq501 Serial Shared-Scale Target Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- diagnostic_classification: `{metrics['diagnostic_classification']}`",
      f"- cpu-norm/gpu-norm/z/gpu-z max abs: `{a['attn_norm_from_gpu_input']}` / `{a['gpu_attn_norm_vs_cpu']}` / `{a['z_from_gpu_attn_norm']}` / `{a['gpu_z_vs_cpu']}`",
      f"- final projection max abs: `{a['final_projection']}`",
      f"- reduction_order_fixed: `{str(metrics['reduction_order_fixed']).lower()}`",
      f"- reduction_order_still_open: `{str(metrics['reduction_order_still_open']).lower()}`",
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
  parser.add_argument("--seq500", type=Path, default=DEFAULT_SEQ500)
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
