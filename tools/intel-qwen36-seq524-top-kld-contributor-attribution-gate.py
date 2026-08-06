#!/usr/bin/env python3
"""Classify top KLD contributor attribution for router prompt drift."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = "intel-qwen36-seq524-top-kld-contributor-attribution-v0"

DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_SEQ523 = (
    ROOT
    / "output/seq523-logit-drift-anatomy-gate-20260709Tseq523Z"
    / "metrics.json"
)
DEFAULT_MATH = (
    ROOT
    / "output/seq524-top-kld-contributor-attribution-math-20260709Tseq524Z"
    / "result.json"
)
DEFAULT_CODE = (
    ROOT
    / "output/seq524-top-kld-contributor-attribution-code-20260709Tseq524Z"
    / "result.json"
)
DEFAULT_CPU_LMHEAD_MATH = (
    ROOT
    / "output/r2-gpu-router-math-distribution-rowblock16-26mask-cpu-lmhead-20260708Tseq228Z"
    / "result.json"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "output/seq524-top-kld-contributor-attribution-gate-20260709Tseq524Z"
)

CURRENT_ROUTE = "router_prompt_distribution_top_kld_contributor_attribution_gate"
NEXT_ROUTE = "router_prompt_distribution_head_token_pair_projection_source_gate"
REJECTED_ROUTE = "router_prompt_distribution_tail_mass_attribution"
KLD_THRESHOLD = 0.005
TOP1_THRESHOLD = 0.99
DECOMPOSITION_TOL = 1.0e-7
TOP_POSITIVE_RATIO_MIN = 0.85
TOP_NEGATIVE_RATIO_MIN = 0.95
HEAD_NATIVE_PROB_MASS_MIN = 0.90

REQUIRED_STEP_FIELDS = {
    "native_expected_logit_delta",
    "native_minus_gpu_logsumexp",
    "positive_kld_contribution_sum",
    "negative_kld_contribution_sum",
    "top_kld_contributors",
    "top_negative_kld_contributors",
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


def _sum_rows(rows: Any, field: str) -> float:
  if not isinstance(rows, list):
    return 0.0
  return sum(_num(row.get(field)) for row in rows if isinstance(row, dict))


def _top_row(rows: Any) -> dict[str, Any]:
  if not isinstance(rows, list) or not rows:
    return {}
  row = rows[0]
  return row if isinstance(row, dict) else {}


def _failed_step(step: dict[str, Any]) -> dict[str, Any]:
  positive = step.get("top_kld_contributors")
  negative = step.get("top_negative_kld_contributors")
  positive_sum = _num(step.get("positive_kld_contribution_sum"))
  negative_sum = _num(step.get("negative_kld_contribution_sum"))
  top_positive_sum = _sum_rows(positive, "contribution")
  top_negative_sum = _sum_rows(negative, "contribution")
  head_prob_mass = (
      _sum_rows(positive, "native_prob") + _sum_rows(negative, "native_prob"))
  decomposition_kld = (
      _num(step.get("native_expected_logit_delta"))
      - _num(step.get("native_minus_gpu_logsumexp")))
  contribution_kld = positive_sum + negative_sum
  return {
      "token_index": step.get("token_index"),
      "token_position": step.get("token_position"),
      "kld": step.get("kld"),
      "native_top1_id": step.get("native_top1_id"),
      "gpu_top1_id": step.get("gpu_top1_id"),
      "top1_matches": step.get("top1_matches"),
      "native_expected_logit_delta": step.get("native_expected_logit_delta"),
      "native_minus_gpu_logsumexp": step.get(
          "native_minus_gpu_logsumexp"),
      "decomposition_kld": decomposition_kld,
      "positive_kld_contribution_sum": positive_sum,
      "negative_kld_contribution_sum": negative_sum,
      "contribution_kld": contribution_kld,
      "top_positive_coverage_ratio": (
          top_positive_sum / positive_sum if positive_sum > 0.0 else 0.0),
      "top_negative_coverage_ratio": (
          top_negative_sum / negative_sum if negative_sum < 0.0 else 0.0),
      "head_native_prob_mass": head_prob_mass,
      "top_positive": _top_row(positive),
      "top_negative": _top_row(negative),
  }


def _row(path: Path, label: str) -> dict[str, Any]:
  payload = _load_json(path)
  smoke = payload.get("smoke")
  smoke = smoke if isinstance(smoke, dict) else payload
  dist = _dist(smoke)
  steps = dist.get("steps")
  steps = steps if isinstance(steps, list) else []
  fields_present = (
      len(steps) > 0
      and all(isinstance(step, dict)
              and REQUIRED_STEP_FIELDS.issubset(set(step.keys()))
              for step in steps))
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
      "decomposition_fields_present": fields_present,
      "failed_steps": failed_steps,
  }


def _row_ran(row: dict[str, Any]) -> bool:
  return (
      row.get("target_returncode") == 2
      and row.get("distribution_required_checks_passed") is False
      and row.get("kld_pass") is False
      and row.get("top1_pass") is True
      and _num(row.get("top1_rate")) >= TOP1_THRESHOLD
      and _num(row.get("max_kld")) > KLD_THRESHOLD
      and row.get("decomposition_fields_present") is True)


def _decomposition_is_consistent(row: dict[str, Any]) -> bool:
  failed = row.get("failed_steps")
  if not isinstance(failed, list) or not failed:
    return False
  for step in failed:
    if not isinstance(step, dict):
      return False
    kld = _num(step.get("kld"))
    if abs(_num(step.get("decomposition_kld")) - kld) > DECOMPOSITION_TOL:
      return False
    if abs(_num(step.get("contribution_kld")) - kld) > DECOMPOSITION_TOL:
      return False
  return True


def _head_pair_dominates(row: dict[str, Any]) -> bool:
  failed = row.get("failed_steps")
  if not isinstance(failed, list) or not failed:
    return False
  for step in failed:
    if not isinstance(step, dict):
      return False
    if _num(step.get("top_positive_coverage_ratio")) < TOP_POSITIVE_RATIO_MIN:
      return False
    if _num(step.get("top_negative_coverage_ratio")) < TOP_NEGATIVE_RATIO_MIN:
      return False
    if _num(step.get("head_native_prob_mass")) < HEAD_NATIVE_PROB_MASS_MIN:
      return False
  return True


def _cpu_lmhead_same_failure(path: Path, math_row: dict[str, Any]) -> dict[str, Any]:
  row = _row(path, "seq228_cpu_lmhead_math")
  return {
      "path": row["path"],
      "case_id": row["case_id"],
      "max_kld": row["max_kld"],
      "top1_rate": row["top1_rate"],
      "matches_seq524_math_shape": (
          abs(_num(row.get("max_kld")) - _num(math_row.get("max_kld")))
          < 1.0e-5
          and _num(row.get("top1_rate")) >= TOP1_THRESHOLD),
  }


def compute(args: argparse.Namespace) -> dict[str, Any]:
  routes = _load_json(args.routes)
  seq523 = _load_json(args.seq523)
  math_row = _row(args.math, "seq524_top_kld_attribution_math")
  code_row = _row(args.code, "seq524_top_kld_attribution_code")
  cpu_lmhead_math = _cpu_lmhead_same_failure(args.cpu_lmhead_math, math_row)

  checks = [
      {
          "name": "seq523_selected_top_kld_contributor_attribution_route",
          "pass": (
              seq523.get("required_checks_passed") is True
              and seq523.get("selected_next_route") == CURRENT_ROUTE
              and _has_candidate(
                  routes, 523,
                  "reject_global_affine_logit_calibration_select_top_kld_contributor_attribution")
              and _has_switch(
                  routes,
                  "select_router_prompt_distribution_top_kld_contributor_attribution_gate",
                  523)),
      },
      {
          "name": "attribution_rows_target_ran_with_decomposition_fields",
          "pass": _row_ran(math_row) and _row_ran(code_row),
          "detail": {"math": math_row, "code": code_row},
      },
      {
          "name": "kld_decomposition_identity_holds",
          "pass": _decomposition_is_consistent(math_row)
          and _decomposition_is_consistent(code_row),
      },
      {
          "name": "failed_steps_are_head_token_pair_mass_not_tail_mass",
          "pass": _head_pair_dominates(math_row) and _head_pair_dominates(code_row),
          "detail": {
              "thresholds": {
                  "top_positive_ratio_min": TOP_POSITIVE_RATIO_MIN,
                  "top_negative_ratio_min": TOP_NEGATIVE_RATIO_MIN,
                  "head_native_prob_mass_min": HEAD_NATIVE_PROB_MASS_MIN,
              },
              "math_failed_steps": math_row.get("failed_steps"),
              "code_failed_steps": code_row.get("failed_steps"),
          },
      },
      {
          "name": "cpu_lmhead_does_not_move_math_failure_shape",
          "pass": cpu_lmhead_math.get("matches_seq524_math_shape") is True,
          "detail": cpu_lmhead_math,
      },
  ]
  required = all(bool(row.get("pass")) for row in checks)
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "inputs": {
          "routes": _rel(args.routes),
          "seq523": _rel(args.seq523),
          "math": _rel(args.math),
          "code": _rel(args.code),
          "cpu_lmhead_math": _rel(args.cpu_lmhead_math),
      },
      "checks": checks,
      "required_checks_passed": required,
      "speedup_claims_allowed": False,
      "decode_probe_allowed": False,
      "router_distribution_allowed": False,
      "tail_mass_attribution_allowed": False,
      "head_token_pair_projection_source_gate_allowed": required,
      "rows": {
          "math": math_row,
          "code": code_row,
          "cpu_lmhead_math": cpu_lmhead_math,
      },
      "disposition": (
          "reject_tail_mass_attribution_select_head_token_pair_projection_source_gate"
          if required else
          "block_top_kld_contributor_attribution_inconsistent_evidence"),
      "rejected_route": REJECTED_ROUTE if required else None,
      "selected_next_route": NEXT_ROUTE if required else CURRENT_ROUTE,
      "next_route_reason": (
          "The failed KLD steps are concentrated in a small high-probability "
          "head-token pair, not a tail-mass or global-normalizer-only problem. "
          "The top listed positive and negative contributors cover the KLD "
          "mass and native probability head on every failing step; CPU LM-head "
          "placement already leaves the math failure shape unchanged. The next "
          "unit should attribute the native-vs-GPU logit deltas for these head "
          "token pairs through final hidden-state projection/source, before any "
          "product correction."
          if required else
          "Top KLD contributor attribution evidence is inconsistent; do not "
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
      "# Seq524 Top KLD Contributor Attribution Gate",
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
      f"- CPU-LM-head math KLD/top1: `{rows['cpu_lmhead_math']['max_kld']}` / `{rows['cpu_lmhead_math']['top1_rate']}`",
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
  parser.add_argument("--seq523", type=Path, default=DEFAULT_SEQ523)
  parser.add_argument("--math", type=Path, default=DEFAULT_MATH)
  parser.add_argument("--code", type=Path, default=DEFAULT_CODE)
  parser.add_argument(
      "--cpu-lmhead-math", type=Path, default=DEFAULT_CPU_LMHEAD_MATH)
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
