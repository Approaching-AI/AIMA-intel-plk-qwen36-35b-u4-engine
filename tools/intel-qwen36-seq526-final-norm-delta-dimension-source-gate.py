#!/usr/bin/env python3
"""Classify final-norm delta dimension source attribution rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = "intel-qwen36-seq526-final-norm-delta-dimension-source-v0"

DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_SEQ525 = (
    ROOT
    / "output/seq525-head-token-pair-projection-source-gate-20260709Tseq525Z"
    / "metrics.json"
)
DEFAULT_MATH = (
    ROOT
    / "output/seq526-final-norm-delta-dimension-source-math-20260709Tseq526Z"
    / "result.json"
)
DEFAULT_CODE = (
    ROOT
    / "output/seq526-final-norm-delta-dimension-source-code-20260709Tseq526Z"
    / "result.json"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "output/seq526-final-norm-delta-dimension-source-gate-20260709Tseq526Z"
)

CURRENT_ROUTE = "router_prompt_distribution_final_norm_delta_dimension_source_gate"
NEXT_ROUTE = "router_prompt_distribution_final_residual_delta_dimension_source_gate"
REJECTED_ROUTE = "router_prompt_distribution_output_norm_rms_scale_delta"
KLD_THRESHOLD = 0.005
TOP1_THRESHOLD = 0.99
RESIDUAL_DOMINANCE_MIN = 0.80

REQUIRED_DIM_FIELDS = {
    "residual_delta_component",
    "scale_delta_component",
    "native_residual",
    "gpu_residual",
    "output_norm_weight",
    "native_rms_scale",
    "gpu_rms_scale",
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


def _projection(step: dict[str, Any]) -> dict[str, Any]:
  projections = step.get("head_pair_projections")
  if not isinstance(projections, list) or not projections:
    return {}
  projection = projections[0]
  return projection if isinstance(projection, dict) else {}


def _failed_step(step: dict[str, Any]) -> dict[str, Any]:
  projection = _projection(step)
  residual_component = _num(projection.get(
      "residual_delta_pair_gap_component"))
  scale_component = _num(projection.get("scale_delta_pair_gap_component"))
  denom = abs(residual_component) + abs(scale_component)
  dims = projection.get("top_pair_projection_dims")
  top_dims = dims[:8] if isinstance(dims, list) else []
  dim_fields_present = (
      bool(top_dims)
      and all(isinstance(row, dict) and REQUIRED_DIM_FIELDS.issubset(row.keys())
              for row in top_dims))
  return {
      "token_index": step.get("token_index"),
      "token_position": step.get("token_position"),
      "kld": step.get("kld"),
      "positive_token_id": projection.get("positive_token_id"),
      "negative_token_id": projection.get("negative_token_id"),
      "observed_pair_gap_delta": projection.get("observed_pair_gap_delta"),
      "cpu_projection_pair_gap_delta": projection.get(
          "cpu_projection_pair_gap_delta"),
      "residual_delta_pair_gap_component": residual_component,
      "scale_delta_pair_gap_component": scale_component,
      "residual_dominance_ratio": (
          abs(residual_component) / denom if denom > 0.0 else 0.0),
      "dimension_fields_present": dim_fields_present,
      "top_pair_projection_dims": top_dims,
  }


def _row(path: Path, label: str) -> dict[str, Any]:
  payload = _load_json(path)
  smoke = payload.get("smoke")
  smoke = smoke if isinstance(smoke, dict) else payload
  dist = _dist(smoke)
  steps = dist.get("steps")
  steps = steps if isinstance(steps, list) else []
  failed_steps = [
      _failed_step(step) for step in steps
      if isinstance(step, dict) and _num(step.get("kld")) > KLD_THRESHOLD
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
      "top1_pass": dist.get("top1_pass"),
      "kld_pass": dist.get("kld_pass"),
      "step_count": len(steps),
      "failed_step_count": len(failed_steps),
      "failed_steps": failed_steps,
  }


def _row_ran(row: dict[str, Any]) -> bool:
  return (
      row.get("target_returncode") == 2
      and row.get("distribution_required_checks_passed") is False
      and row.get("kld_pass") is False
      and row.get("top1_pass") is True
      and _num(row.get("top1_rate")) >= TOP1_THRESHOLD
      and _num(row.get("max_kld")) > KLD_THRESHOLD)


def _residual_delta_dominates(row: dict[str, Any]) -> bool:
  failed = row.get("failed_steps")
  if not isinstance(failed, list) or not failed:
    return False
  for step in failed:
    if not isinstance(step, dict):
      return False
    if step.get("dimension_fields_present") is not True:
      return False
    if _num(step.get("residual_dominance_ratio")) < RESIDUAL_DOMINANCE_MIN:
      return False
  return True


def compute(args: argparse.Namespace) -> dict[str, Any]:
  routes = _load_json(args.routes)
  seq525 = _load_json(args.seq525)
  math_row = _row(args.math, "seq526_final_norm_delta_source_math")
  code_row = _row(args.code, "seq526_final_norm_delta_source_code")

  checks = [
      {
          "name": "seq525_selected_final_norm_delta_dimension_source_route",
          "pass": (
              seq525.get("required_checks_passed") is True
              and seq525.get("selected_next_route") == CURRENT_ROUTE
              and _has_candidate(
                  routes, 525,
                  "reject_lm_head_projection_boundary_select_final_norm_delta_dimension_source")
              and _has_switch(
                  routes,
                  "select_router_prompt_distribution_final_norm_delta_dimension_source_gate",
                  525)),
      },
      {
          "name": "final_norm_rows_target_ran_with_residual_scale_decomposition",
          "pass": _row_ran(math_row) and _row_ran(code_row),
          "detail": {"math": math_row, "code": code_row},
      },
      {
          "name": "final_norm_delta_is_residual_delta_dominated",
          "pass": _residual_delta_dominates(math_row)
          and _residual_delta_dominates(code_row),
          "detail": {
              "residual_dominance_min": RESIDUAL_DOMINANCE_MIN,
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
          "seq525": _rel(args.seq525),
          "math": _rel(args.math),
          "code": _rel(args.code),
      },
      "checks": checks,
      "required_checks_passed": required,
      "speedup_claims_allowed": False,
      "decode_probe_allowed": False,
      "router_distribution_allowed": False,
      "output_norm_rms_scale_delta_allowed": False,
      "final_residual_delta_dimension_source_gate_allowed": required,
      "rows": {"math": math_row, "code": code_row},
      "disposition": (
          "reject_output_norm_rms_scale_delta_select_final_residual_delta_dimension_source"
          if required else
          "block_final_norm_delta_dimension_source_inconsistent_evidence"),
      "rejected_route": REJECTED_ROUTE if required else None,
      "selected_next_route": NEXT_ROUTE if required else CURRENT_ROUTE,
      "next_route_reason": (
          "The final-norm dimension deltas that explain failed head-pair gaps "
          "are dominated by raw final-residual value deltas rather than output "
          "RMS scale drift. The next unit should attribute those high-weight "
          "final-residual dimensions back through the last-layer source path "
          "before any product correction."
          if required else
          "Final-norm delta dimension evidence is inconsistent; do not switch "
          "routes or run speed/promotion/long-context rows."),
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
      "# Seq526 Final-Norm Delta Dimension Source Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- rejected_route: `{metrics['rejected_route']}`",
      f"- failed_checks: `{failed}`",
      f"- speedup_claims_allowed: `{str(metrics['speedup_claims_allowed']).lower()}`",
      "",
      "## Evidence",
      "",
      f"- math KLD/top1/failed steps: `{rows['math']['max_kld']}` / `{rows['math']['top1_rate']}` / `{rows['math']['failed_step_count']}`",
      f"- code KLD/top1/failed steps: `{rows['code']['max_kld']}` / `{rows['code']['top1_rate']}` / `{rows['code']['failed_step_count']}`",
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
  parser.add_argument("--seq525", type=Path, default=DEFAULT_SEQ525)
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
