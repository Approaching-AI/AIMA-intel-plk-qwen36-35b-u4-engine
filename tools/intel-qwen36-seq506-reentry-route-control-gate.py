#!/usr/bin/env python3
"""Select the next route after selected layer-input source-value reentry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = "intel-qwen36-seq506-reentry-route-control-gate-v0"

DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_REJECTED = ACTIVE / "rejected-routes.json"
DEFAULT_SEQ316 = (
    ROOT
    / "output/router-qkv-delta-product-consumer-router-distribution-gate-20260708Tseq316Z"
    / "metrics.json"
)
DEFAULT_SEQ317 = (
    ROOT
    / "output/router-qkv-delta-product-consumer-distribution-fix-gate-20260708Tseq317Z"
    / "metrics.json"
)
DEFAULT_SEQ349 = (
    ROOT
    / "output/router-full-attention-layer-input-product-consumer-distribution-fix-gate-20260708Tseq349Z"
    / "metrics.json"
)
DEFAULT_SEQ505 = (
    ROOT
    / "output/seq505-accepted-contiguous2-distribution-reentry-gate-20260709Tseq505Z"
    / "metrics.json"
)
DEFAULT_MATH_TOP512 = (
    ROOT
    / "output/r2-gpu-router-math-distribution-rowblock16-26mask-double-swiglu-shadow-linear-qkvcol-delta-blockq16-top512-20260708Tseq264Z"
    / "result.json"
)
DEFAULT_CODE_TOP512 = (
    ROOT
    / "output/r2-gpu-router-code-distribution-rowblock16-26mask-double-swiglu-shadow-all-linear-qkvcol-delta-blockq16-top512-20260708Tseq291Z"
    / "result.json"
)
DEFAULT_OUT_DIR = (
    ROOT / "output/seq506-reentry-route-control-gate-20260709Tseq506Z"
)

KLD_THRESHOLD = 0.005
TOP1_THRESHOLD = 0.99
FLOAT_EPS = 1.0e-12
SELECTED_NEXT_ROUTE = (
    "router_prompt_all_linear_current_token_qkv_delta_recursion_break_design_gate"
)


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
      for row in routes.get("candidate_history", []))


def _has_switch(routes: dict[str, Any], decision: str, seq_covered: int) -> bool:
  return any(
      isinstance(row, dict)
      and row.get("decision") == decision
      and _num(row.get("seq_covered")) >= seq_covered
      and row.get("resolved") is True
      for row in routes.get("switch_decisions", []))


def _has_rejected_route(rejected: dict[str, Any], route: str) -> bool:
  return any(
      isinstance(row, dict) and row.get("route") == route
      for row in rejected.get("rejected", []))


def _same(left: Any, right: Any) -> bool:
  return abs(_num(left) - _num(right)) <= FLOAT_EPS


def _smoke(path: Path) -> dict[str, Any]:
  payload = _load_json(path)
  smoke = payload.get("smoke")
  return smoke if isinstance(smoke, dict) else payload


def _dist_from_smoke(path: Path) -> dict[str, Any]:
  smoke = _smoke(path)
  dist = smoke.get("distribution_ladder")
  dist = dist if isinstance(dist, dict) else {}
  return {
      "path": _rel(path),
      "case_id": smoke.get("case_id"),
      "required_checks_passed": smoke.get("required_checks_passed"),
      "distribution_required_checks_passed": dist.get("required_checks_passed"),
      "max_kld": dist.get("max_kld"),
      "top1_rate": dist.get("top1_rate"),
      "min_logits_cosine": dist.get("min_logits_cosine"),
      "position_count": dist.get("position_count"),
      "cpu_shadow_state_each_token_enabled": smoke.get(
          "cpu_shadow_state_each_token_enabled"),
  }


def _dist_pass(row: dict[str, Any]) -> bool:
  return (
      row.get("distribution_required_checks_passed") is True
      and _num(row.get("max_kld")) <= KLD_THRESHOLD
      and _num(row.get("top1_rate")) >= TOP1_THRESHOLD)


def _seq316_rows(seq316: dict[str, Any]) -> list[dict[str, Any]]:
  rows = []
  for run in seq316.get("runs", []) or []:
    if not isinstance(run, dict):
      continue
    summary = run.get("summary")
    if isinstance(summary, dict):
      rows.append(summary)
  return rows


def _dist(row: dict[str, Any]) -> dict[str, Any]:
  dist = row.get("distribution")
  return dist if isinstance(dist, dict) else {}


def _by_case(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
  out = {}
  for row in rows:
    case_id = row.get("case_id")
    if isinstance(case_id, str):
      out[case_id] = row
  return out


def _seq505_match_seq316(seq505: dict[str, Any],
                         seq316_rows: list[dict[str, Any]]) -> dict[str, Any]:
  rows = []
  all_match = True
  seq505_rows = (
      seq505.get("distribution_match", {}).get("rows")
      if isinstance(seq505.get("distribution_match"), dict) else [])
  by_505 = _by_case([row for row in seq505_rows if isinstance(row, dict)])
  by_316 = _by_case(seq316_rows)
  for case_id in sorted(set(by_505) | set(by_316)):
    row505 = by_505.get(case_id, {})
    row316 = by_316.get(case_id, {})
    d316 = _dist(row316)
    seq504 = row505.get("seq504")
    seq504 = seq504 if isinstance(seq504, dict) else {}
    match = (
        _same(seq504.get("max_kld"), d316.get("max_kld"))
        and _same(seq504.get("top1_rate"), d316.get("top1_rate"))
        and _same(seq504.get("min_logits_cosine"),
                  d316.get("min_logits_cosine")))
    all_match = all_match and match
    rows.append({
        "case_id": case_id,
        "match": match,
        "seq505_seq504": seq504,
        "seq316": {
            "max_kld": d316.get("max_kld"),
            "top1_rate": d316.get("top1_rate"),
            "min_logits_cosine": d316.get("min_logits_cosine"),
        },
    })
  return {"all_match": bool(rows) and all_match, "rows": rows}


def _seq316_counters_ready(rows: list[dict[str, Any]]) -> bool:
  return bool(rows) and all(
      row.get("producer_layers") == 72
      and row.get("producer_values") == 147456
      and row.get("producer_misses") == 0
      and row.get("product_layers") == 216
      and row.get("product_values") == 110592
      and row.get("product_misses") == 24
      and row.get("cpu_shadow_layer_input_layers") == 0
      and row.get("cpu_shadow_attention_output_layers") == 0
      for row in rows)


def _seq316_failed(rows: list[dict[str, Any]]) -> bool:
  return bool(rows) and any(
      _num(_dist(row).get("max_kld")) > KLD_THRESHOLD
      or _num(_dist(row).get("top1_rate"), 1.0) < TOP1_THRESHOLD
      for row in rows)


def _source_gap(seq349: dict[str, Any]) -> dict[str, Any]:
  for check in seq349.get("checks", []) or []:
    if not isinstance(check, dict) or check.get("name") != "selected_layer_input_gap_emitted":
      continue
    gaps = check.get("detail")
    gaps = [gap for gap in gaps if isinstance(gap, dict)] if isinstance(gaps, list) else []
    return {
        "case_count": len(gaps),
        "observation_count": sum(int(_num(g.get("observation_count"))) for g in gaps),
        "min_layer_input_cosine": min(
            (_num(g.get("min_layer_input_cosine"), 1.0) for g in gaps),
            default=1.0),
        "max_layer_input_abs_diff": max(
            (_num(g.get("max_layer_input_abs_diff")) for g in gaps),
            default=0.0),
    }
  return {"case_count": 0, "observation_count": 0}


def compute(args: argparse.Namespace) -> dict[str, Any]:
  routes = _load_json(args.routes)
  rejected = _load_json(args.rejected)
  seq316 = _load_json(args.seq316)
  seq317 = _load_json(args.seq317)
  seq349 = _load_json(args.seq349)
  seq505 = _load_json(args.seq505)
  math_top512 = _dist_from_smoke(args.math_top512)
  code_top512 = _dist_from_smoke(args.code_top512)
  seq316_rows = _seq316_rows(seq316)
  product_failure_match = _seq505_match_seq316(seq505, seq316_rows)
  source_gap = _source_gap(seq349)
  checks = [
      {
          "name": "seq505_reentry_gate_selected",
          "pass": (
              seq505.get("required_checks_passed") is True
              and seq505.get("disposition")
              == "accept_accepted_contiguous2_distribution_reentry_to_layer_input_source_value_gap"
              and seq505.get("selected_next_route")
              == "router_prompt_full_attention_selected_layer_input_source_value_reentry_gate"
              and _has_candidate(
                  routes, 505,
                  "accept_accepted_contiguous2_distribution_reentry_to_layer_input_source_value_gap")
              and _has_switch(
                  routes,
                  "select_router_prompt_full_attention_selected_layer_input_source_value_reentry_gate",
                  505)),
      },
      {
          "name": "reentry_matches_qkv_delta_product_consumer_failure",
          "pass": product_failure_match["all_match"],
          "detail": product_failure_match["rows"],
      },
      {
          "name": "qkv_delta_product_consumer_was_product_owned_but_failed",
          "pass": (
              seq316.get("selected_next_route")
              == "router_prompt_all_linear_qkv_delta_product_consumer_distribution_fix_gate"
              and _seq316_counters_ready(seq316_rows)
              and _seq316_failed(seq316_rows)),
      },
      {
          "name": "full_attention_residual_route_has_now_reentered",
          "pass": (
              seq317.get("required_checks_passed") is True
              and seq317.get("selected_next_route")
              == "router_prompt_full_attention_residual_product_source_gate"
              and source_gap.get("case_count") == 2
              and source_gap.get("observation_count") == 144
              and _num(source_gap.get("max_layer_input_abs_diff")) > 0.0),
          "detail": source_gap,
      },
      {
          "name": "all_linear_current_token_top512_diagnostic_still_passes",
          "pass": _dist_pass(math_top512) and _dist_pass(code_top512),
          "detail": {"math_top512": math_top512, "code_top512": code_top512},
      },
      {
          "name": "closed_approximations_acknowledged",
          "pass": (
              _has_rejected_route(
                  rejected,
                  "router_math_static_or_lagged_qkv_delta_predictors")
              and _has_rejected_route(
                  rejected,
                  "router_math_live_round_or_selected_affine_qkv_delta_approximation")),
      },
  ]
  required = all(row.get("pass") is True for row in checks)
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "inputs": {
          "routes": _rel(args.routes),
          "rejected": _rel(args.rejected),
          "seq316": _rel(args.seq316),
          "seq317": _rel(args.seq317),
          "seq349": _rel(args.seq349),
          "seq505": _rel(args.seq505),
          "math_top512": _rel(args.math_top512),
          "code_top512": _rel(args.code_top512),
      },
      "checks": checks,
      "product_failure_match": product_failure_match,
      "source_gap": source_gap,
      "top512_diagnostic": {
          "math": math_top512,
          "code": code_top512,
      },
      "required_checks_passed": required,
      "diagnostic_classification": (
          "selected_layer_input_reentry_requires_current_token_qkv_delta_recursion_break"
          if required else
          "selected_layer_input_reentry_route_unclassified"),
      "disposition": (
          "accept_reentry_route_control_select_current_token_qkv_delta_recursion_break"
          if required else
          "block_reentry_route_control_gate"),
      "selected_next_route": SELECTED_NEXT_ROUTE if required else str(
          seq505.get("selected_next_route")),
      "next_route_reason": (
          "The recursive selected layer-input/product-source chain has "
          "re-entered the same failure as the qkv-delta product consumer. The "
          "only recorded lower-bound pass is still all-linear current-token "
          "qkv-column top512, while static/lagged/rounding/affine substitutes "
          "are closed. The next unit should be a design/source gate for a "
          "product current-token qkv/residual correction that breaks the "
          "recursive source chase, not another upstream value-source rerun."
          if required else
          "Reentry route-control evidence is incomplete; keep the reentry gate open."),
      "speedup_claims_allowed": False,
  }


def write_outputs(metrics: dict[str, Any], out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n",
      encoding="utf-8")
  failed = [
      row["name"] for row in metrics["checks"] if row.get("pass") is not True
  ]
  lines = [
      "# Seq506 Reentry Route-Control Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- failed_checks: `{failed}`",
      "",
      "| case | reentry KLD | qkv product KLD | match |",
      "|---|---:|---:|---|",
  ]
  for row in metrics["product_failure_match"]["rows"]:
    lines.append(
        f"| `{row['case_id']}` | `{row['seq505_seq504'].get('max_kld')}` | "
        f"`{row['seq316'].get('max_kld')}` | `{str(row['match']).lower()}` |")
  source_gap = metrics["source_gap"]
  lines.extend([
      "",
      f"- source-value min layer-input cosine: `{source_gap.get('min_layer_input_cosine')}`",
      f"- source-value max layer-input abs diff: `{source_gap.get('max_layer_input_abs_diff')}`",
      f"- top512 math/code KLD: `{metrics['top512_diagnostic']['math'].get('max_kld')}` / `{metrics['top512_diagnostic']['code'].get('max_kld')}`",
      "",
      metrics["next_route_reason"],
      "",
      "This is route-control/correctness evidence only. It is not a speed claim.",
      "",
  ])
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--routes", type=Path, default=DEFAULT_ROUTES)
  parser.add_argument("--rejected", type=Path, default=DEFAULT_REJECTED)
  parser.add_argument("--seq316", type=Path, default=DEFAULT_SEQ316)
  parser.add_argument("--seq317", type=Path, default=DEFAULT_SEQ317)
  parser.add_argument("--seq349", type=Path, default=DEFAULT_SEQ349)
  parser.add_argument("--seq505", type=Path, default=DEFAULT_SEQ505)
  parser.add_argument("--math-top512", type=Path, default=DEFAULT_MATH_TOP512)
  parser.add_argument("--code-top512", type=Path, default=DEFAULT_CODE_TOP512)
  parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  metrics = compute(args)
  write_outputs(metrics, args.out_dir)
  print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
