#!/usr/bin/env python3
"""Classify the router prompt full-vocab logit drift anatomy rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = "intel-qwen36-seq523-logit-drift-anatomy-v0"

DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_SEQ522 = (
    ROOT
    / "output/seq522-fp64-sensitivity-result-gate-20260709Tseq522Z"
    / "metrics.json"
)
DEFAULT_MATH = (
    ROOT
    / "output/seq523-logit-drift-anatomy-math-20260709Tseq523Z"
    / "result.json"
)
DEFAULT_CODE = (
    ROOT
    / "output/seq523-logit-drift-anatomy-code-20260709Tseq523Z"
    / "result.json"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "output/seq523-logit-drift-anatomy-gate-20260709Tseq523Z"
)

CURRENT_ROUTE = "router_prompt_distribution_logit_drift_anatomy_gate"
PREVIOUS_ROUTE = "router_prompt_distribution_fp64_sensitivity_gate"
REJECTED_AFFINE_ROUTE = "router_prompt_distribution_affine_logit_calibration"
NEXT_ROUTE = "router_prompt_distribution_top_kld_contributor_attribution_gate"
KLD_THRESHOLD = 0.005
TOP1_THRESHOLD = 0.99

REQUIRED_STEP_FIELDS = {
    "native_top1_margin",
    "gpu_top1_margin",
    "affine_gpu_to_native_scale",
    "affine_gpu_to_native_offset",
    "affine_gpu_to_native_rmse",
    "affine_gpu_to_native_kld",
    "affine_gpu_to_native_max_abs_diff",
    "affine_gpu_to_native_mean_abs_diff",
    "top_kld_contributors",
}


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


def _top_contributor(step: dict[str, Any]) -> dict[str, Any]:
  rows = step.get("top_kld_contributors")
  if not isinstance(rows, list) or not rows:
    return {}
  row = rows[0]
  return row if isinstance(row, dict) else {}


def _failed_step(step: dict[str, Any]) -> dict[str, Any]:
  top = _top_contributor(step)
  contribution = _num(top.get("contribution"))
  native_prob = _num(top.get("native_prob"))
  logp_delta = contribution / native_prob if native_prob > 0.0 else 0.0
  logit_delta = _num(top.get("native_logit")) - _num(top.get("gpu_logit"))
  return {
      "token_index": step.get("token_index"),
      "token_position": step.get("token_position"),
      "kld": step.get("kld"),
      "native_top1_id": step.get("native_top1_id"),
      "gpu_top1_id": step.get("gpu_top1_id"),
      "top1_matches": step.get("top1_matches"),
      "native_top1_margin": step.get("native_top1_margin"),
      "gpu_top1_margin": step.get("gpu_top1_margin"),
      "affine_gpu_to_native_kld": step.get("affine_gpu_to_native_kld"),
      "affine_gpu_to_native_scale": step.get("affine_gpu_to_native_scale"),
      "affine_gpu_to_native_rmse": step.get("affine_gpu_to_native_rmse"),
      "top_contributor": {
          "token_id": top.get("token_id"),
          "contribution": top.get("contribution"),
          "native_prob": top.get("native_prob"),
          "native_logit": top.get("native_logit"),
          "gpu_logit": top.get("gpu_logit"),
          "native_minus_gpu_logit": logit_delta,
          "native_minus_gpu_logp": logp_delta,
          "implied_native_minus_gpu_logsumexp": logit_delta - logp_delta,
      },
      "top_contributor_count": len(step.get("top_kld_contributors", []))
      if isinstance(step.get("top_kld_contributors"), list) else 0,
      "top_contributor_native_prob_sum": sum(
          _num(row.get("native_prob")) for row in step.get(
              "top_kld_contributors", [])
          if isinstance(row, dict)),
      "top_contributor_positive_kld_sum": sum(
          _num(row.get("contribution")) for row in step.get(
              "top_kld_contributors", [])
          if isinstance(row, dict)),
  }


def _row(path: Path, label: str) -> dict[str, Any]:
  payload = _load_json(path)
  smoke = payload.get("smoke")
  smoke = smoke if isinstance(smoke, dict) else payload
  dist = _dist(smoke)
  steps = dist.get("steps")
  steps = steps if isinstance(steps, list) else []
  instrumentation_fields_present = (
      len(steps) > 0
      and all(isinstance(step, dict)
              and REQUIRED_STEP_FIELDS.issubset(set(step.keys()))
              for step in steps))
  failed_steps = [
      _failed_step(step) for step in steps
      if isinstance(step, dict) and _num(step.get("kld")) > KLD_THRESHOLD
  ]
  affine_klds = [
      _num(step.get("affine_gpu_to_native_kld"))
      for step in steps if isinstance(step, dict)
  ]
  original_klds = [
      _num(step.get("kld")) for step in steps if isinstance(step, dict)
  ]
  return {
      "label": label,
      "path": _rel(path),
      "case_id": smoke.get("case_id"),
      "target_returncode": payload.get("target", {}).get(
          "run", {}).get("returncode"),
      "required_checks_passed": smoke.get("required_checks_passed"),
      "distribution_required_checks_passed": dist.get("required_checks_passed"),
      "max_kld": dist.get("max_kld"),
      "mean_kld": dist.get("mean_kld"),
      "top1_rate": dist.get("top1_rate"),
      "top1_match_count": dist.get("top1_match_count"),
      "kld_pass": dist.get("kld_pass"),
      "top1_pass": dist.get("top1_pass"),
      "thresholds": dist.get("thresholds"),
      "step_count": len(steps),
      "failed_step_count": len(failed_steps),
      "instrumentation_fields_present": instrumentation_fields_present,
      "max_affine_gpu_to_native_kld": max(affine_klds) if affine_klds else None,
      "min_affine_gpu_to_native_kld": min(affine_klds) if affine_klds else None,
      "max_original_kld": max(original_klds) if original_klds else None,
      "failed_steps": failed_steps,
      "opencl_no_fma_enabled": smoke.get("opencl_no_fma_enabled"),
      "opencl_double_sigmoid_enabled": smoke.get(
          "opencl_double_sigmoid_enabled"),
      "opencl_double_swiglu_enabled": smoke.get(
          "opencl_double_swiglu_enabled"),
      "opencl_double_softmax_enabled": smoke.get(
          "opencl_double_softmax_enabled"),
      "linear_l2_double_sum_enabled": smoke.get("linear_l2_double_sum_enabled"),
  }


def _anatomy_row_ran(row: dict[str, Any]) -> bool:
  return (
      row.get("target_returncode") == 2
      and row.get("distribution_required_checks_passed") is False
      and row.get("kld_pass") is False
      and row.get("top1_pass") is True
      and _num(row.get("top1_rate")) >= TOP1_THRESHOLD
      and _num(row.get("max_kld")) > KLD_THRESHOLD
      and row.get("instrumentation_fields_present") is True
      and row.get("opencl_no_fma_enabled") is False
      and row.get("opencl_double_sigmoid_enabled") is False
      and row.get("opencl_double_swiglu_enabled") is False
      and row.get("opencl_double_softmax_enabled") is False
      and row.get("linear_l2_double_sum_enabled") is False)


def _affine_does_not_close(row: dict[str, Any]) -> bool:
  return _num(row.get("max_affine_gpu_to_native_kld")) > KLD_THRESHOLD


def _top_contributors_present(row: dict[str, Any]) -> bool:
  failed = row.get("failed_steps")
  if not isinstance(failed, list) or not failed:
    return False
  for step in failed:
    if not isinstance(step, dict):
      return False
    top = step.get("top_contributor")
    if not isinstance(top, dict):
      return False
    if top.get("token_id") is None:
      return False
    if _num(top.get("contribution")) <= 0.0:
      return False
    if _num(top.get("native_prob")) <= 0.0:
      return False
  return True


def compute(args: argparse.Namespace) -> dict[str, Any]:
  routes = _load_json(args.routes)
  seq522 = _load_json(args.seq522)
  math_row = _row(args.math, "seq523_logit_drift_anatomy_math")
  code_row = _row(args.code, "seq523_logit_drift_anatomy_code")

  checks = [
      {
          "name": "seq522_selected_logit_drift_anatomy_route",
          "pass": (
              seq522.get("required_checks_passed") is True
              and seq522.get("selected_next_route") == CURRENT_ROUTE
              and _has_candidate(
                  routes, 522,
                  "reject_fp64_precision_pack_sensitivity_select_logit_drift_anatomy")
              and _has_switch(
                  routes,
                  "select_router_prompt_distribution_logit_drift_anatomy_gate",
                  522)),
      },
      {
          "name": "anatomy_rows_target_ran_with_new_fields",
          "pass": _anatomy_row_ran(math_row) and _anatomy_row_ran(code_row),
          "detail": {"math": math_row, "code": code_row},
      },
      {
          "name": "global_affine_fit_does_not_close_kld",
          "pass": _affine_does_not_close(math_row)
          and _affine_does_not_close(code_row),
          "detail": {
              "math_max_affine_kld": math_row.get(
                  "max_affine_gpu_to_native_kld"),
              "code_max_affine_kld": code_row.get(
                  "max_affine_gpu_to_native_kld"),
              "threshold": KLD_THRESHOLD,
          },
      },
      {
          "name": "top_kld_contributors_available_for_next_attribution",
          "pass": _top_contributors_present(math_row)
          and _top_contributors_present(code_row),
          "detail": {
              "math_failed_steps": math_row.get("failed_steps"),
              "code_failed_steps": code_row.get("failed_steps"),
          },
      },
  ]
  required = all(bool(row.get("pass")) for row in checks)
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "inputs": {
          "routes": _rel(args.routes),
          "seq522": _rel(args.seq522),
          "math": _rel(args.math),
          "code": _rel(args.code),
      },
      "checks": checks,
      "required_checks_passed": required,
      "speedup_claims_allowed": False,
      "decode_probe_allowed": False,
      "router_distribution_allowed": False,
      "affine_logit_calibration_allowed": False,
      "top_kld_contributor_attribution_allowed": required,
      "rows": {"math": math_row, "code": code_row},
      "disposition": (
          "reject_global_affine_logit_calibration_select_top_kld_contributor_attribution"
          if required else
          "block_logit_drift_anatomy_inconsistent_evidence"),
      "rejected_route": REJECTED_AFFINE_ROUTE if required else None,
      "selected_next_route": NEXT_ROUTE if required else CURRENT_ROUTE,
      "next_route_reason": (
          "The full-vocab anatomy rows reproduce the router math/code KLD "
          "block with greedy top-1 still passing. A least-squares global "
          "GPU-to-native affine fit does not close either row, so global "
          "temperature/offset calibration is not an admissible fix. The "
          "failure mass is now available as per-step positive KLD contributor "
          "tokens with native/GPU logit and log-sum-exp deltas; the next unit "
          "should attribute those specific contributor logits/normalizers "
          "before changing a product source."
          if required else
          "Logit drift anatomy evidence is inconsistent; do not switch routes "
          "or launch speed/promotion/long-context rows."),
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
      "# Seq523 Logit Drift Anatomy Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- rejected_route: `{metrics['rejected_route']}`",
      f"- failed_checks: `{failed}`",
      f"- top_kld_contributor_attribution_allowed: `{str(metrics['top_kld_contributor_attribution_allowed']).lower()}`",
      f"- speedup_claims_allowed: `{str(metrics['speedup_claims_allowed']).lower()}`",
      "",
      "## Evidence",
      "",
      f"- math KLD/top1/failed steps: `{rows['math']['max_kld']}` / `{rows['math']['top1_rate']}` / `{rows['math']['failed_step_count']}`",
      f"- math max affine KLD: `{rows['math']['max_affine_gpu_to_native_kld']}`",
      f"- code KLD/top1/failed steps: `{rows['code']['max_kld']}` / `{rows['code']['top1_rate']}` / `{rows['code']['failed_step_count']}`",
      f"- code max affine KLD: `{rows['code']['max_affine_gpu_to_native_kld']}`",
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
  parser.add_argument("--seq522", type=Path, default=DEFAULT_SEQ522)
  parser.add_argument("--math", type=Path, default=DEFAULT_MATH)
  parser.add_argument("--code", type=Path, default=DEFAULT_CODE)
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
