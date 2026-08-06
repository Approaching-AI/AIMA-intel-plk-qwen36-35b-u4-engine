#!/usr/bin/env python3
"""Classify head-token pair projection/source attribution rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = "intel-qwen36-seq525-head-token-pair-projection-source-v0"

DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_SEQ524 = (
    ROOT
    / "output/seq524-top-kld-contributor-attribution-gate-20260709Tseq524Z"
    / "metrics.json"
)
DEFAULT_MATH = (
    ROOT
    / "output/seq525-head-token-pair-projection-source-math-20260709Tseq525Z"
    / "result.json"
)
DEFAULT_CODE = (
    ROOT
    / "output/seq525-head-token-pair-projection-source-code-20260709Tseq525Z"
    / "result.json"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "output/seq525-head-token-pair-projection-source-gate-20260709Tseq525Z"
)

CURRENT_ROUTE = "router_prompt_distribution_head_token_pair_projection_source_gate"
NEXT_ROUTE = "router_prompt_distribution_final_norm_delta_dimension_source_gate"
REJECTED_ROUTE = "router_prompt_distribution_lm_head_projection_boundary"
KLD_THRESHOLD = 0.005
TOP1_THRESHOLD = 0.99
MAX_PROJECTION_RESIDUAL = 0.125
MIN_PROJECTION_DOMINANCE = 0.80

REQUIRED_PROJECTION_FIELDS = {
    "positive_token_id",
    "negative_token_id",
    "observed_pair_gap_delta",
    "cpu_projection_pair_gap_delta",
    "observed_minus_cpu_projection_pair_gap_delta",
    "top_pair_projection_dims",
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


def _projection(row: dict[str, Any]) -> dict[str, Any]:
  projections = row.get("head_pair_projections")
  if not isinstance(projections, list) or not projections:
    return {}
  projection = projections[0]
  return projection if isinstance(projection, dict) else {}


def _failed_step(step: dict[str, Any]) -> dict[str, Any]:
  projection = _projection(step)
  observed = _num(projection.get("observed_pair_gap_delta"))
  cpu_projection = _num(projection.get("cpu_projection_pair_gap_delta"))
  residual = _num(
      projection.get("observed_minus_cpu_projection_pair_gap_delta"))
  dims = projection.get("top_pair_projection_dims")
  return {
      "token_index": step.get("token_index"),
      "token_position": step.get("token_position"),
      "kld": step.get("kld"),
      "native_top1_id": step.get("native_top1_id"),
      "gpu_top1_id": step.get("gpu_top1_id"),
      "positive_token_id": projection.get("positive_token_id"),
      "negative_token_id": projection.get("negative_token_id"),
      "observed_pair_gap_delta": observed,
      "cpu_projection_pair_gap_delta": cpu_projection,
      "observed_minus_cpu_projection_pair_gap_delta": residual,
      "projection_dominance_ratio": (
          abs(cpu_projection) / abs(observed) if abs(observed) > 0.0 else 0.0),
      "top_pair_projection_dims": (
          dims[:8] if isinstance(dims, list) else []),
      "projection_fields_present": (
          REQUIRED_PROJECTION_FIELDS.issubset(set(projection.keys()))
          if projection else False),
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


def _projection_source_dominates(row: dict[str, Any]) -> bool:
  failed = row.get("failed_steps")
  if not isinstance(failed, list) or not failed:
    return False
  for step in failed:
    if not isinstance(step, dict):
      return False
    if step.get("projection_fields_present") is not True:
      return False
    if abs(_num(step.get("observed_minus_cpu_projection_pair_gap_delta"))) > (
        MAX_PROJECTION_RESIDUAL):
      return False
    if _num(step.get("projection_dominance_ratio")) < MIN_PROJECTION_DOMINANCE:
      return False
    dims = step.get("top_pair_projection_dims")
    if not isinstance(dims, list) or not dims:
      return False
  return True


def compute(args: argparse.Namespace) -> dict[str, Any]:
  routes = _load_json(args.routes)
  seq524 = _load_json(args.seq524)
  math_row = _row(args.math, "seq525_head_pair_projection_math")
  code_row = _row(args.code, "seq525_head_pair_projection_code")

  checks = [
      {
          "name": "seq524_selected_head_token_pair_projection_source_route",
          "pass": (
              seq524.get("required_checks_passed") is True
              and seq524.get("selected_next_route") == CURRENT_ROUTE
              and _has_candidate(
                  routes, 524,
                  "reject_tail_mass_attribution_select_head_token_pair_projection_source_gate")
              and _has_switch(
                  routes,
                  "select_router_prompt_distribution_head_token_pair_projection_source_gate",
                  524)),
      },
      {
          "name": "projection_rows_target_ran_with_head_pair_fields",
          "pass": _row_ran(math_row) and _row_ran(code_row),
          "detail": {"math": math_row, "code": code_row},
      },
      {
          "name": "final_norm_cpu_projection_explains_head_pair_gap_delta",
          "pass": _projection_source_dominates(math_row)
          and _projection_source_dominates(code_row),
          "detail": {
              "thresholds": {
                  "max_projection_residual": MAX_PROJECTION_RESIDUAL,
                  "min_projection_dominance": MIN_PROJECTION_DOMINANCE,
              },
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
          "seq524": _rel(args.seq524),
          "math": _rel(args.math),
          "code": _rel(args.code),
      },
      "checks": checks,
      "required_checks_passed": required,
      "speedup_claims_allowed": False,
      "decode_probe_allowed": False,
      "router_distribution_allowed": False,
      "lm_head_projection_boundary_allowed": False,
      "final_norm_delta_dimension_source_gate_allowed": required,
      "rows": {"math": math_row, "code": code_row},
      "disposition": (
          "reject_lm_head_projection_boundary_select_final_norm_delta_dimension_source"
          if required else
          "block_head_token_pair_projection_source_inconsistent_evidence"),
      "rejected_route": REJECTED_ROUTE if required else None,
      "selected_next_route": NEXT_ROUTE if required else CURRENT_ROUTE,
      "next_route_reason": (
          "The failed head-token pair logit gap deltas are mostly explained by "
          "CPU projection of the native-vs-GPU final-norm vector through the "
          "same output.weight rows. The remaining observed-minus-projection "
          "residual is bounded, and top final-norm weighted dimensions are now "
          "available per failed pair. This closes LM-head placement/projection "
          "as the next route and selects final-norm delta dimension source "
          "attribution before any product correction."
          if required else
          "Head-token pair projection/source evidence is inconsistent; do not "
          "switch routes or run speed/promotion/long-context rows."),
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
      "# Seq525 Head-Token Pair Projection Source Gate",
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
  parser.add_argument("--seq524", type=Path, default=DEFAULT_SEQ524)
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
