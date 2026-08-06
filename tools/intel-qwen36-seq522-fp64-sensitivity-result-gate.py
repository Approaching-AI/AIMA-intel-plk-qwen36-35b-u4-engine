#!/usr/bin/env python3
"""Classify the router distribution FP64 sensitivity rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = "intel-qwen36-seq522-fp64-sensitivity-result-v0"

DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_SEQ521 = (
    ROOT
    / "output/seq521-distribution-correctness-route-reflection-gate-20260709Tseq521Z"
    / "metrics.json"
)
DEFAULT_BASE_MATH = (
    ROOT
    / "output/r2-gpu-acceptance-matrix-router-both-rowblock16-26mask-20260708Tseq222Z"
    / "runs/router_math_reason_001-distribution/result.json"
)
DEFAULT_BASE_CODE = (
    ROOT
    / "output/r2-gpu-acceptance-matrix-router-both-rowblock16-26mask-20260708Tseq222Z"
    / "runs/router_code_reason_002-distribution/result.json"
)
DEFAULT_FP64_MATH = (
    ROOT
    / "output/seq522-router-distribution-fp64-sensitivity-math-20260709Tseq522Z"
    / "result.json"
)
DEFAULT_FP64_CODE = (
    ROOT
    / "output/seq522-router-distribution-fp64-sensitivity-code-20260709Tseq522Z"
    / "result.json"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "output/seq522-fp64-sensitivity-result-gate-20260709Tseq522Z"
)

SELECTED_ROUTE = "router_prompt_distribution_fp64_sensitivity_gate"
NEXT_ROUTE = "router_prompt_distribution_logit_drift_anatomy_gate"
KLD_THRESHOLD = 0.005
TOP1_THRESHOLD = 0.99


def _load_json(path: Path) -> Any:
  return json.loads(path.read_text(encoding="utf-8"))


def _rel(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path)


def _num(value: Any, default: float = 0.0) -> float:
  return float(value) if isinstance(value, (int, float)) else default


def _has_candidate(routes: dict[str, Any], seq: int, disposition: str) -> bool:
  return any(
      isinstance(row, dict)
      and row.get("seq") == seq
      and row.get("disposition") == disposition
      for row in routes.get("candidate_history", []))


def _has_switch(routes: dict[str, Any], decision: str, seq_covered: int) -> bool:
  return any(
      isinstance(row, dict)
      and row.get("decision") == decision
      and _num(row.get("seq_covered")) >= seq_covered
      and row.get("resolved") is True
      for row in routes.get("switch_decisions", []))


def _smoke(path: Path) -> dict[str, Any]:
  payload = _load_json(path)
  smoke = payload.get("smoke")
  return smoke if isinstance(smoke, dict) else payload


def _dist(smoke: dict[str, Any]) -> dict[str, Any]:
  dist = smoke.get("distribution_ladder")
  return dist if isinstance(dist, dict) else {}


def _row(path: Path, label: str) -> dict[str, Any]:
  smoke = _smoke(path)
  dist = _dist(smoke)
  return {
      "label": label,
      "path": _rel(path),
      "case_id": smoke.get("case_id"),
      "required_checks_passed": smoke.get("required_checks_passed"),
      "target_returncode": _load_json(path).get("target", {}).get(
          "run", {}).get("returncode"),
      "max_kld": dist.get("max_kld"),
      "mean_kld": dist.get("mean_kld"),
      "top1_rate": dist.get("top1_rate"),
      "top1_match_count": dist.get("top1_match_count"),
      "min_logits_cosine": dist.get("min_logits_cosine"),
      "distribution_required_checks_passed": dist.get("required_checks_passed"),
      "opencl_no_fma_enabled": smoke.get("opencl_no_fma_enabled"),
      "opencl_double_sigmoid_enabled": smoke.get(
          "opencl_double_sigmoid_enabled"),
      "opencl_double_swiglu_enabled": smoke.get(
          "opencl_double_swiglu_enabled"),
      "opencl_double_softmax_enabled": smoke.get(
          "opencl_double_softmax_enabled"),
      "linear_l2_double_sum_enabled": smoke.get("linear_l2_double_sum_enabled"),
  }


def _base_still_kld_block(row: dict[str, Any]) -> bool:
  return (
      _num(row.get("max_kld")) > KLD_THRESHOLD
      and _num(row.get("top1_rate")) >= TOP1_THRESHOLD
      and row.get("distribution_required_checks_passed") is False)


def _fp64_pack_ran(row: dict[str, Any]) -> bool:
  return (
      row.get("target_returncode") == 2
      and row.get("opencl_no_fma_enabled") is True
      and row.get("opencl_double_sigmoid_enabled") is True
      and row.get("opencl_double_swiglu_enabled") is True
      and row.get("opencl_double_softmax_enabled") is False
      and row.get("linear_l2_double_sum_enabled") is False)


def _fp64_worse(base: dict[str, Any], fp64: dict[str, Any]) -> bool:
  return (
      _fp64_pack_ran(fp64)
      and _num(fp64.get("max_kld")) > _num(base.get("max_kld"))
      and _num(fp64.get("top1_rate")) < _num(base.get("top1_rate"))
      and fp64.get("distribution_required_checks_passed") is False)


def compute(args: argparse.Namespace) -> dict[str, Any]:
  routes = _load_json(args.routes)
  seq521 = _load_json(args.seq521)
  base_math = _row(args.base_math, "seq222_base_math")
  base_code = _row(args.base_code, "seq222_base_code")
  fp64_math = _row(args.fp64_math, "seq522_fp64_math")
  fp64_code = _row(args.fp64_code, "seq522_fp64_code")

  checks = [
      {
          "name": "seq521_selected_fp64_sensitivity_route",
          "pass": (
              seq521.get("required_checks_passed") is True
              and seq521.get("selected_next_route") == SELECTED_ROUTE
              and _has_candidate(
                  routes, 521,
                  "accept_correctness_route_reflection_select_fp64_sensitivity_gate")
              and _has_switch(
                  routes,
                  "select_router_prompt_distribution_fp64_sensitivity_gate",
                  521)),
      },
      {
          "name": "base_rows_are_kld_only_failures",
          "pass": _base_still_kld_block(base_math)
          and _base_still_kld_block(base_code),
          "detail": {"math": base_math, "code": base_code},
      },
      {
          "name": "fp64_precision_pack_target_rows_completed",
          "pass": _fp64_pack_ran(fp64_math) and _fp64_pack_ran(fp64_code),
          "detail": {"math": fp64_math, "code": fp64_code},
      },
      {
          "name": "fp64_precision_pack_does_not_close_distribution",
          "pass": (
              _num(fp64_math.get("max_kld")) > KLD_THRESHOLD
              and _num(fp64_code.get("max_kld")) > KLD_THRESHOLD
              and _num(fp64_math.get("top1_rate")) < TOP1_THRESHOLD
              and _num(fp64_code.get("top1_rate")) < TOP1_THRESHOLD),
          "detail": {"math": fp64_math, "code": fp64_code},
      },
      {
          "name": "fp64_precision_pack_regresses_vs_base",
          "pass": _fp64_worse(base_math, fp64_math)
          and _fp64_worse(base_code, fp64_code),
          "detail": {
              "base_math": base_math,
              "fp64_math": fp64_math,
              "base_code": base_code,
              "fp64_code": fp64_code,
          },
      },
  ]
  required = all(bool(row.get("pass")) for row in checks)
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "inputs": {
          "routes": _rel(args.routes),
          "seq521": _rel(args.seq521),
          "base_math": _rel(args.base_math),
          "base_code": _rel(args.base_code),
          "fp64_math": _rel(args.fp64_math),
          "fp64_code": _rel(args.fp64_code),
      },
      "checks": checks,
      "required_checks_passed": required,
      "speedup_claims_allowed": False,
      "decode_probe_allowed": False,
      "router_distribution_allowed": False,
      "logit_drift_anatomy_probe_allowed": required,
      "rows": {
          "base_math": base_math,
          "base_code": base_code,
          "fp64_math": fp64_math,
          "fp64_code": fp64_code,
      },
      "disposition": (
          "reject_fp64_precision_pack_sensitivity_select_logit_drift_anatomy"
          if required else
          "block_fp64_sensitivity_result_inconsistent_evidence"),
      "selected_next_route": NEXT_ROUTE if required else SELECTED_ROUTE,
      "next_route_reason": (
          "The compile-current FP64 precision pack (no-FMA plus double sigmoid "
          "and double SwiGLU) does not close the router prompt distribution "
          "block; it regresses both math and code versus the seq222 baseline "
          "and flips top-1 on both rows. This closes precision-polish as the "
          "next correctness move. The next unit should instrument full-vocab "
          "logit drift anatomy: per-step KLD contributors, native/GPU top "
          "margins, and optimal affine/temperature fit before any product "
          "calibration or source-route change."
          if required else
          "FP64 sensitivity evidence is inconsistent; do not switch routes or "
          "launch a speed/promotion row."),
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
  rows = metrics["rows"]
  lines = [
      "# Seq522 FP64 Sensitivity Result Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- failed_checks: `{failed}`",
      f"- logit_drift_anatomy_probe_allowed: `{str(metrics['logit_drift_anatomy_probe_allowed']).lower()}`",
      f"- speedup_claims_allowed: `{str(metrics['speedup_claims_allowed']).lower()}`",
      "",
      "## Evidence",
      "",
      f"- base math KLD/top1: `{rows['base_math']['max_kld']}` / `{rows['base_math']['top1_rate']}`",
      f"- FP64-pack math KLD/top1: `{rows['fp64_math']['max_kld']}` / `{rows['fp64_math']['top1_rate']}`",
      f"- base code KLD/top1: `{rows['base_code']['max_kld']}` / `{rows['base_code']['top1_rate']}`",
      f"- FP64-pack code KLD/top1: `{rows['fp64_code']['max_kld']}` / `{rows['fp64_code']['top1_rate']}`",
      "",
      metrics["next_route_reason"],
      "",
      "This is correctness-route evidence only. It is not a speed claim.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--routes", type=Path, default=DEFAULT_ROUTES)
  parser.add_argument("--seq521", type=Path, default=DEFAULT_SEQ521)
  parser.add_argument("--base-math", type=Path, default=DEFAULT_BASE_MATH)
  parser.add_argument("--base-code", type=Path, default=DEFAULT_BASE_CODE)
  parser.add_argument("--fp64-math", type=Path, default=DEFAULT_FP64_MATH)
  parser.add_argument("--fp64-code", type=Path, default=DEFAULT_FP64_CODE)
  parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
  return parser.parse_args()


def main() -> int:
  args = parse_args()
  metrics = compute(args)
  write_outputs(metrics, args.out_dir)
  print(json.dumps({
      "required_checks_passed": metrics["required_checks_passed"],
      "disposition": metrics["disposition"],
      "selected_next_route": metrics["selected_next_route"],
      "out_dir": _rel(args.out_dir),
  }, sort_keys=True))
  return 0 if metrics["required_checks_passed"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
