#!/usr/bin/env python3
"""Confirm the CPU-sqrt norm diagnostic covers the shared RMSNorm scale kernel."""

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
SEQ498_GATE = (
    ROOT / "tools/intel-qwen36-seq498-attn-norm-math-shared-scale-kernel-gate.py"
)
SCHEMA_VERSION = (
    "intel-qwen36-seq499-attn-norm-shared-scale-cpu-sqrt-coverage-gate-v0"
)

DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_SEQ498 = (
    ROOT
    / "output/seq498-attn-norm-math-shared-scale-kernel-gate-20260709Tseq498Z"
    / "metrics.json"
)
DEFAULT_OPENCL = ROOT / "engine/gpu/opencl/q4x8_matvec.cl"
DEFAULT_SMOKE = ROOT / "tools/intel-qwen36-r2-gpu-decode-smoke.py"
DEFAULT_OUT_DIR = (
    ROOT
    / "output/seq499-attn-norm-shared-scale-cpu-sqrt-coverage-gate-20260709Tseq499Z"
)


def _load_module(path: Path, name: str) -> Any:
  spec = importlib.util.spec_from_file_location(name, path)
  if spec is None or spec.loader is None:
    raise SystemExit(f"cannot load module {name}: {path}")
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


SEQ498 = _load_module(SEQ498_GATE, "iq36_seq498_gate")
CURRENT_ROUTE = SEQ498.SCALE_KERNEL_ROUTE
CPU_SQRT_PROBE_ROUTE = CURRENT_ROUTE.replace(
    "_linear_z_source_attn_norm_scale_kernel_gap_gate",
    "_linear_z_source_attn_norm_scale_kernel_cpu_sqrt_probe_gate")


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


def _source_checks(opencl_source: str, smoke_source: str) -> dict[str, Any]:
  shared_scale_kernel_is_live = (
      all(snippet in opencl_source for snippet in [
          "__kernel void rms_norm_hidden_scale_f32",
          "__kernel void rms_norm_hidden_apply_scale_f32",
      ])
      and (
          "scale_out[0] = rsqrt(mean_square + epsilon);" in opencl_source
          or "scale_out[0] = 1.0f / sqrt(mean_square + epsilon);" in opencl_source
      ))
  cpu_sqrt_flag_exists = all(snippet in smoke_source for snippet in [
      "--opencl-cpu-sqrt-norm",
      "if opencl_cpu_sqrt_norm:",
      "shared hidden RMSNorm CPU-shaped sqrt scale",
      "hidden RMSNorm CPU-shaped sqrt scale",
  ])
  shared_scale_rewrite = all(snippet in smoke_source for snippet in [
      "scale_out[0] = rsqrt(mean_square + epsilon);",
      "scale_out[0] = 1.0f / sqrt(mean_square + epsilon);",
  ])
  standalone_rewrite = all(snippet in smoke_source for snippet in [
      "const float scale = rsqrt(mean_square + epsilon);",
      "const float scale = 1.0f / sqrt(mean_square + epsilon);",
  ])
  return {
      "shared_scale_kernel_is_live": shared_scale_kernel_is_live,
      "cpu_sqrt_flag_exists": cpu_sqrt_flag_exists,
      "shared_scale_rewrite": shared_scale_rewrite,
      "standalone_rewrite": standalone_rewrite,
      "no_new_variant_flag_required": (
          smoke_source.count("--opencl-cpu-sqrt-norm") >= 3
          and "--opencl-shared-rmsnorm" not in smoke_source),
  }


def compute(args: argparse.Namespace) -> dict[str, Any]:
  routes = _load_json(args.routes)
  seq498 = _load_json(args.seq498)
  classification = str(seq498.get("diagnostic_classification"))
  cpu_sqrt_probe_classification = classification.replace(
      "_attn_norm_scale_kernel_gap",
      "_attn_norm_scale_kernel_cpu_sqrt_probe_ready")
  preconditions_pass = (
      seq498.get("required_checks_passed") is True
      and seq498.get("selected_next_route") == CURRENT_ROUTE
      and seq498.get("attn_norm_scale_kernel_gap") is True
      and classification.endswith("_attn_norm_scale_kernel_gap")
      and _has_candidate(routes, 498, str(seq498.get("disposition")))
      and _has_switch(routes, f"select_{CURRENT_ROUTE}", 498)
  )
  source = _source_checks(
      args.opencl.read_text(encoding="utf-8"),
      args.smoke.read_text(encoding="utf-8"),
  )
  source_ready = all(source.values())
  checks = [
      {"name": "seq498_shared_scale_kernel_gate", "pass": preconditions_pass},
      {"name": "cpu_sqrt_norm_covers_shared_scale_kernel",
       "pass": source_ready,
       "detail": source},
      {"name": "target_probe_route_selected",
       "pass": preconditions_pass and source_ready,
       "detail": {"selected_next_route": CPU_SQRT_PROBE_ROUTE}},
  ]
  required = all(bool(row.get("pass")) for row in checks)
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "inputs": {
          "routes": _rel(args.routes),
          "seq498": _rel(args.seq498),
          "opencl_source": _rel(args.opencl),
          "decode_smoke_source": _rel(args.smoke),
      },
      "checks": checks,
      "required_checks_passed": required,
      "diagnostic_classification": (
          cpu_sqrt_probe_classification if required
          else f"{classification}_cpu_sqrt_probe_not_ready"),
      "cpu_sqrt_shared_scale_coverage": source_ready,
      "source_checks": source,
      "speedup_claims_allowed": False,
      "disposition": (
          f"accept_{classification}_cpu_sqrt_coverage"
          if required else
          f"block_{classification}_cpu_sqrt_coverage"
      ),
      "selected_next_route": CPU_SQRT_PROBE_ROUTE if required else CURRENT_ROUTE,
      "next_route_reason": (
          "The existing --opencl-cpu-sqrt-norm diagnostic now rewrites both the "
          "standalone hidden RMSNorm scale and the shared scale_out kernel used "
          "by the active layer-input path. The next target probe can separate "
          "scale rsqrt from the remaining local-partial reduction order."
          if required else
          "CPU-sqrt coverage for the shared RMSNorm scale kernel is incomplete; "
          "keep the shared scale-kernel gate open."
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
  s = metrics["source_checks"]
  lines = [
      "# Seq499 Shared RMSNorm CPU-Sqrt Coverage Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- diagnostic_classification: `{metrics['diagnostic_classification']}`",
      f"- shared_scale_kernel_is_live: `{str(s['shared_scale_kernel_is_live']).lower()}`",
      f"- shared_scale_rewrite: `{str(s['shared_scale_rewrite']).lower()}`",
      f"- standalone_rewrite: `{str(s['standalone_rewrite']).lower()}`",
      f"- no_new_variant_flag_required: `{str(s['no_new_variant_flag_required']).lower()}`",
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
  parser.add_argument("--seq498", type=Path, default=DEFAULT_SEQ498)
  parser.add_argument("--opencl", type=Path, default=DEFAULT_OPENCL)
  parser.add_argument("--smoke", type=Path, default=DEFAULT_SMOKE)
  parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
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
