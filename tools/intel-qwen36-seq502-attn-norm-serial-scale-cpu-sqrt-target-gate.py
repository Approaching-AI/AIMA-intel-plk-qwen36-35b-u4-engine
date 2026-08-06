#!/usr/bin/env python3
"""Run serial shared-scale plus CPU-sqrt target gate."""

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
SEQ500_GATE = (
    ROOT / "tools/intel-qwen36-seq500-attn-norm-scale-kernel-cpu-sqrt-target-gate.py"
)
SEQ501_GATE = (
    ROOT / "tools/intel-qwen36-seq501-attn-norm-scale-kernel-reduction-order-target-gate.py"
)
SCHEMA_VERSION = (
    "intel-qwen36-seq502-attn-norm-serial-scale-cpu-sqrt-target-gate-v0"
)

DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_SEQ501 = (
    ROOT
    / "output/seq501-attn-norm-scale-kernel-reduction-order-target-gate-20260709Tseq501Z"
    / "metrics.json"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "output/seq502-attn-norm-serial-scale-cpu-sqrt-target-gate-20260709Tseq502Z"
)

FIXED_ABS_EPS = 1.0e-7


def _load_module(path: Path, name: str) -> Any:
  spec = importlib.util.spec_from_file_location(name, path)
  if spec is None or spec.loader is None:
    raise SystemExit(f"cannot load module {name}: {path}")
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


SEQ500 = _load_module(SEQ500_GATE, "iq36_seq500_for_seq502")
SEQ501 = _load_module(SEQ501_GATE, "iq36_seq501_for_seq502")
CURRENT_ROUTE = SEQ501.REDUCTION_STILL_OPEN_ROUTE
SERIAL_CPU_SQRT_FIX_ROUTE = CURRENT_ROUTE.replace(
    "_linear_z_source_attn_norm_scale_kernel_reduction_order_unresolved_gate",
    "_linear_z_source_attn_norm_scale_kernel_serial_cpu_sqrt_fix_gate")
SERIAL_CPU_SQRT_OPEN_ROUTE = CURRENT_ROUTE.replace(
    "_linear_z_source_attn_norm_scale_kernel_reduction_order_unresolved_gate",
    "_linear_z_source_attn_norm_scale_kernel_serial_cpu_sqrt_unresolved_gate")


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
  seq501 = _load_json(args.seq501)
  runs = [SEQ500._run_case(args, case_id) for case_id in SEQ500.CASES]
  rows = [SEQ500._case_summary(row) for row in runs]
  classification = str(seq501.get("diagnostic_classification"))
  preconditions_pass = (
      seq501.get("required_checks_passed") is True
      and seq501.get("selected_next_route") == CURRENT_ROUTE
      and seq501.get("reduction_order_still_open") is True
      and _has_candidate(routes, 501, str(seq501.get("disposition")))
      and _has_switch(routes, f"select_{CURRENT_ROUTE}", 501)
  )
  rows_emitted = (
      len(rows) == len(SEQ500.CASES)
      and all(row.get("opencl_cpu_sqrt_norm_enabled") is True for row in rows)
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
  serial_cpu_sqrt_fixed = (
      rows_emitted
      and cpu_norm_abs == 0.0
      and gpu_z_abs == 0.0
      and gpu_norm_abs <= FIXED_ABS_EPS
      and z_from_norm_abs <= FIXED_ABS_EPS)
  still_open = rows_emitted and not serial_cpu_sqrt_fixed
  required = preconditions_pass and counters_ready and (
      serial_cpu_sqrt_fixed or still_open)
  selected = (
      SERIAL_CPU_SQRT_FIX_ROUTE if serial_cpu_sqrt_fixed
      else SERIAL_CPU_SQRT_OPEN_ROUTE if still_open
      else CURRENT_ROUTE)
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "inputs": {
          "routes": _rel(args.routes),
          "seq501": _rel(args.seq501),
          "token_input_dir": _rel(args.token_input_dir),
      },
      "runs": [{
          "case_id": row.get("case_id"),
          "out_dir": row.get("out_dir"),
          "returncode": row.get("returncode"),
          "opencl_cpu_sqrt_norm_enabled": row.get(
              "opencl_cpu_sqrt_norm_enabled"),
      } for row in rows],
      "checks": [
          {"name": "seq501_unresolved_gate", "pass": preconditions_pass},
          {"name": "target_serial_cpu_sqrt_rows_emitted", "pass": rows_emitted},
          {"name": "product_source_consumer_counters_ready", "pass": counters_ready},
          {"name": "serial_cpu_sqrt_classified",
           "pass": serial_cpu_sqrt_fixed or still_open,
           "detail": {
               "gpu_attn_norm_abs": gpu_norm_abs,
               "serial_cpu_sqrt_fixed": serial_cpu_sqrt_fixed,
               "still_open": still_open,
           }},
      ],
      "required_checks_passed": required,
      "diagnostic_classification": (
          classification.replace(
              "_attn_norm_scale_kernel_reduction_order_unresolved",
              "_attn_norm_scale_kernel_serial_cpu_sqrt_fix")
          if required and serial_cpu_sqrt_fixed else
          classification.replace(
              "_attn_norm_scale_kernel_reduction_order_unresolved",
              "_attn_norm_scale_kernel_serial_cpu_sqrt_unresolved")
          if required and still_open else
          f"{classification}_unclassified"),
      "serial_cpu_sqrt_fixed": serial_cpu_sqrt_fixed,
      "serial_cpu_sqrt_still_open": still_open,
      "max_abs_diffs": {
          "attn_norm_from_gpu_input": cpu_norm_abs,
          "gpu_attn_norm_vs_cpu": gpu_norm_abs,
          "z_from_gpu_attn_norm": z_from_norm_abs,
          "gpu_z_vs_cpu": gpu_z_abs,
          "final_projection": final_abs,
      },
      "speedup_claims_allowed": False,
      "disposition": (
          f"accept_{classification}_serial_cpu_sqrt_target_gate"
          if required else f"block_{classification}_serial_cpu_sqrt_target_gate"
      ),
      "selected_next_route": selected if required else CURRENT_ROUTE,
      "next_route_reason": (
          "Serial shared-scale reduction plus CPU-shaped sqrt clears the selected "
          "attention-norm drift. Promote that product correction and rerun router "
          "distribution next."
          if required and serial_cpu_sqrt_fixed else
          "Serial shared-scale reduction plus CPU-shaped sqrt still leaves drift; "
          "keep the unresolved scale-kernel gate open."
          if required and still_open else
          "Serial+CPU-sqrt target evidence is incomplete; keep this gate open."
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
      "# Seq502 Serial Shared-Scale CPU-Sqrt Target Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- diagnostic_classification: `{metrics['diagnostic_classification']}`",
      f"- cpu-norm/gpu-norm/z/gpu-z max abs: `{a['attn_norm_from_gpu_input']}` / `{a['gpu_attn_norm_vs_cpu']}` / `{a['z_from_gpu_attn_norm']}` / `{a['gpu_z_vs_cpu']}`",
      f"- final projection max abs: `{a['final_projection']}`",
      f"- serial_cpu_sqrt_fixed: `{str(metrics['serial_cpu_sqrt_fixed']).lower()}`",
      f"- serial_cpu_sqrt_still_open: `{str(metrics['serial_cpu_sqrt_still_open']).lower()}`",
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
  parser.add_argument("--seq501", type=Path, default=DEFAULT_SEQ501)
  parser.add_argument("--token-input-dir", type=Path,
                      default=SEQ500.DEFAULT_TOKEN_INPUT_DIR)
  parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
  parser.add_argument("--target", "--host", dest="host", metavar="TARGET", default=SEQ500.DEFAULT_HOST)
  parser.add_argument("--model", default=SEQ500.DEFAULT_MODEL)
  parser.add_argument("--env-script", default=SEQ500.DEFAULT_ENV_SCRIPT)
  parser.add_argument("--staging-root", "--remote-root", dest="remote_root", metavar="PATH", default=SEQ500.DEFAULT_REMOTE_ROOT)
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
