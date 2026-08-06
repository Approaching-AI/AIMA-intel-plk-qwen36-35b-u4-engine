#!/usr/bin/env python3
"""Classify seq504 accepted-contiguous2 distribution as a known reentry.

This gate is host-only. It compares the restored product-baseline router
distribution from seq504 with the earlier layer-input product/consumer rows, and
records whether the scale-kernel branch actually opened a new root.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = (
    "intel-qwen36-seq505-accepted-contiguous2-distribution-reentry-gate-v0"
)

DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_REJECTED = ACTIVE / "rejected-routes.json"
DEFAULT_SEQ348 = (
    ROOT
    / "output/router-full-attention-layer-input-product-consumer-router-distribution-gate-20260708Tseq348Z"
    / "metrics.json"
)
DEFAULT_SEQ349 = (
    ROOT
    / "output/router-full-attention-layer-input-product-consumer-distribution-fix-gate-20260708Tseq349Z"
    / "metrics.json"
)
DEFAULT_SEQ504 = (
    ROOT
    / "output/seq504-attn-norm-scale-kernel-contiguous2-restore-router-distribution-gate-20260709Tseq504Z"
    / "metrics.json"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "output/seq505-accepted-contiguous2-distribution-reentry-gate-20260709Tseq505Z"
)

KLD_THRESHOLD = 0.005
TOP1_THRESHOLD = 0.99
FLOAT_EPS = 1.0e-12
EXPECTED_EVENTS = 72
EXPECTED_VALUES = 147456
SERIAL_CPU_SQRT_REJECTED_ROUTE = (
    "shared_rmsnorm_scale_kernel_serial_cpu_sqrt_product_fix"
)
OLD_SOURCE_VALUE_ROUTE = (
    "router_prompt_full_attention_layer_input_product_source_value_gap_gate"
)
SELECTED_NEXT_ROUTE = (
    "router_prompt_full_attention_selected_layer_input_source_value_reentry_gate"
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


def _same_float(left: Any, right: Any) -> bool:
  return abs(_num(left) - _num(right)) <= FLOAT_EPS


def _has_candidate(routes: dict[str, Any], seq: int, disposition: str) -> bool:
  return any(
      isinstance(row, dict)
      and row.get("seq") == seq
      and row.get("disposition") == disposition
      for row in routes.get("candidate_history", []))


def _has_switch(routes: dict[str, Any], decision_suffix: str,
                seq_covered: int) -> bool:
  return any(
      isinstance(row, dict)
      and str(row.get("decision", "")).endswith(decision_suffix)
      and _num(row.get("seq_covered")) >= seq_covered
      and row.get("resolved") is True
      for row in routes.get("switch_decisions", []))


def _has_rejected_route(rejected: dict[str, Any], route: str) -> bool:
  return any(
      isinstance(row, dict) and row.get("route") == route
      for row in rejected.get("rejected", []))


def _check_detail(metrics: dict[str, Any], name: str) -> list[dict[str, Any]]:
  checks = metrics.get("checks")
  checks = checks if isinstance(checks, list) else []
  for row in checks:
    if not isinstance(row, dict) or row.get("name") != name:
      continue
    detail = row.get("detail")
    if isinstance(detail, list):
      return [item for item in detail if isinstance(item, dict)]
  return []


def _seq348_rows(seq348: dict[str, Any]) -> list[dict[str, Any]]:
  return _check_detail(seq348, "source_and_consumer_counters_ready_for_router_rows")


def _seq349_rows(seq349: dict[str, Any]) -> list[dict[str, Any]]:
  return _check_detail(seq349, "product_consumer_counters_ready_during_diagnostic")


def _seq504_rows(seq504: dict[str, Any]) -> list[dict[str, Any]]:
  rows = seq504.get("restored_contiguous2_cases")
  return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def _dist(row: dict[str, Any]) -> dict[str, Any]:
  dist = row.get("distribution")
  if isinstance(dist, dict):
    return dist
  return {
      "max_kld": row.get("max_kld"),
      "top1_rate": row.get("top1_rate"),
      "min_logits_cosine": row.get("min_logits_cosine"),
      "position_count": row.get("position_count"),
      "required_checks_passed": row.get("distribution_required_checks_passed"),
  }


def _by_case(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
  out: dict[str, dict[str, Any]] = {}
  for row in rows:
    case_id = row.get("case_id")
    if isinstance(case_id, str):
      out[case_id] = row
  return out


def _distribution_rows_match(
    seq504_rows: list[dict[str, Any]],
    seq348_rows: list[dict[str, Any]],
    seq349_rows: list[dict[str, Any]],
) -> dict[str, Any]:
  by_504 = _by_case(seq504_rows)
  by_348 = _by_case(seq348_rows)
  by_349 = _by_case(seq349_rows)
  case_ids = sorted(set(by_504) | set(by_348) | set(by_349))
  rows = []
  all_match = bool(case_ids)
  for case_id in case_ids:
    d504 = _dist(by_504.get(case_id, {}))
    d348 = _dist(by_348.get(case_id, {}))
    d349 = _dist(by_349.get(case_id, {}))
    match = (
        _same_float(d504.get("max_kld"), d348.get("max_kld"))
        and _same_float(d504.get("top1_rate"), d348.get("top1_rate"))
        and _same_float(d504.get("min_logits_cosine"),
                        d348.get("min_logits_cosine"))
        and _same_float(d504.get("max_kld"), d349.get("max_kld"))
        and _same_float(d504.get("top1_rate"), d349.get("top1_rate"))
        and _same_float(d504.get("min_logits_cosine"),
                        d349.get("min_logits_cosine")))
    all_match = all_match and match
    rows.append({
        "case_id": case_id,
        "match": match,
        "seq504": {
            "max_kld": d504.get("max_kld"),
            "top1_rate": d504.get("top1_rate"),
            "min_logits_cosine": d504.get("min_logits_cosine"),
        },
        "seq348": {
            "max_kld": d348.get("max_kld"),
            "top1_rate": d348.get("top1_rate"),
            "min_logits_cosine": d348.get("min_logits_cosine"),
        },
        "seq349": {
            "max_kld": d349.get("max_kld"),
            "top1_rate": d349.get("top1_rate"),
            "min_logits_cosine": d349.get("min_logits_cosine"),
        },
    })
  return {"all_match": all_match, "rows": rows}


def _counters_ready(rows: list[dict[str, Any]]) -> bool:
  return bool(rows) and all(
      row.get("source_ready") is True
      and row.get("source_layers") == EXPECTED_EVENTS
      and row.get("source_values") == EXPECTED_VALUES
      and row.get("source_misses") == 0
      and (
          "consumer_ready" not in row
          or (
              row.get("consumer_ready") is True
              and row.get("consumer_layers") == EXPECTED_EVENTS
              and row.get("consumer_values") == EXPECTED_VALUES
              and row.get("consumer_misses") == 0
          )
      )
      for row in rows)


def _distribution_failed(rows: list[dict[str, Any]]) -> bool:
  return bool(rows) and any(
      _num(_dist(row).get("max_kld")) > KLD_THRESHOLD
      or _num(_dist(row).get("top1_rate"), 1.0) < TOP1_THRESHOLD
      for row in rows)


def _source_gap_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
  gaps = []
  history = []
  for row in rows:
    gap = row.get("selected_layer_input_gap")
    if isinstance(gap, dict):
      gaps.append(gap)
    hist = row.get("full_attention_history_gap")
    if isinstance(hist, dict):
      history.append(hist)
  min_layer_input = min(
      (_num(gap.get("min_layer_input_cosine"), 1.0) for gap in gaps),
      default=1.0)
  max_layer_input_abs = max(
      (_num(gap.get("max_layer_input_abs_diff")) for gap in gaps),
      default=0.0)
  min_ffn_input = min(
      (_num(gap.get("min_ffn_input_cosine"), 1.0) for gap in gaps),
      default=1.0)
  max_ffn_input_abs = max(
      (_num(gap.get("max_ffn_input_abs_diff")) for gap in gaps),
      default=0.0)
  min_k_history = min(
      (_num(row.get("min_k_history_cosine"), 1.0) for row in history),
      default=1.0)
  min_v_history = min(
      (_num(row.get("min_v_history_cosine"), 1.0) for row in history),
      default=1.0)
  max_k_history_abs = max(
      (_num(row.get("max_k_history_abs_diff")) for row in history),
      default=0.0)
  max_v_history_abs = max(
      (_num(row.get("max_v_history_abs_diff")) for row in history),
      default=0.0)
  return {
      "case_count": len(gaps),
      "observation_count": sum(
          int(_num(gap.get("observation_count"))) for gap in gaps),
      "expected_observation_count_per_case": EXPECTED_EVENTS,
      "min_layer_input_cosine": min_layer_input,
      "max_layer_input_abs_diff": max_layer_input_abs,
      "min_ffn_input_cosine": min_ffn_input,
      "max_ffn_input_abs_diff": max_ffn_input_abs,
      "min_k_history_cosine": min_k_history,
      "min_v_history_cosine": min_v_history,
      "max_k_history_abs_diff": max_k_history_abs,
      "max_v_history_abs_diff": max_v_history_abs,
      "layer_input_gap_observed": max_layer_input_abs > 0.0,
  }


def compute(args: argparse.Namespace) -> dict[str, Any]:
  routes = _load_json(args.routes)
  rejected = _load_json(args.rejected)
  seq348 = _load_json(args.seq348)
  seq349 = _load_json(args.seq349)
  seq504 = _load_json(args.seq504)

  rows348 = _seq348_rows(seq348)
  rows349 = _seq349_rows(seq349)
  rows504 = _seq504_rows(seq504)
  distribution_match = _distribution_rows_match(rows504, rows348, rows349)
  source_gap = _source_gap_summary(rows349)
  seq504_selected = str(seq504.get("selected_next_route"))
  seq504_accepts_gap = (
      seq504.get("required_checks_passed") is True
      and seq504.get("disposition")
      == "accept_accepted_contiguous2_scale_kernel_router_distribution_gap"
      and seq504.get("restored_distribution_passed") is False
      and seq504_selected.endswith(
          "_attn_norm_scale_kernel_accepted_contiguous2_router_distribution_gap_gate"))
  old_source_route_selected = (
      seq349.get("required_checks_passed") is True
      and seq349.get("selected_next_route") == OLD_SOURCE_VALUE_ROUTE
  )
  checks = [
      {
          "name": "seq504_accepted_contiguous2_gap_selected",
          "pass": seq504_accepts_gap,
      },
      {
          "name": "seq504_matches_seq348_seq349_product_distribution",
          "pass": distribution_match["all_match"],
          "detail": distribution_match["rows"],
      },
      {
          "name": "product_source_consumer_counters_still_ready",
          "pass": (
              _counters_ready(rows504)
              and _counters_ready(rows348)
              and _counters_ready(rows349)),
      },
      {
          "name": "router_distribution_failure_reproduced",
          "pass": (
              _distribution_failed(rows504)
              and _distribution_failed(rows348)
              and _distribution_failed(rows349)),
      },
      {
          "name": "seq349_source_value_gap_available",
          "pass": (
              old_source_route_selected
              and source_gap["case_count"] == 2
              and source_gap["observation_count"] == EXPECTED_EVENTS * 2
              and source_gap["layer_input_gap_observed"] is True),
          "detail": source_gap,
      },
      {
          "name": "serial_cpu_sqrt_product_route_closed",
          "pass": _has_rejected_route(rejected, SERIAL_CPU_SQRT_REJECTED_ROUTE),
      },
      {
          "name": "seq504_route_recorded",
          "pass": (
              _has_candidate(
                  routes, 504,
                  "accept_accepted_contiguous2_scale_kernel_router_distribution_gap")
              and _has_switch(
                  routes,
                  "_attn_norm_scale_kernel_accepted_contiguous2_router_distribution_gap_gate",
                  504)),
      },
  ]
  required = all(row["pass"] for row in checks)
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "inputs": {
          "routes": _rel(args.routes),
          "rejected": _rel(args.rejected),
          "seq348": _rel(args.seq348),
          "seq349": _rel(args.seq349),
          "seq504": _rel(args.seq504),
      },
      "checks": checks,
      "distribution_match": distribution_match,
      "source_gap": source_gap,
      "required_checks_passed": required,
      "diagnostic_classification": (
          "accepted_contiguous2_distribution_reenters_known_layer_input_source_value_gap"
          if required else
          "accepted_contiguous2_distribution_reentry_unclassified"),
      "disposition": (
          "accept_accepted_contiguous2_distribution_reentry_to_layer_input_source_value_gap"
          if required else
          "block_accepted_contiguous2_distribution_reentry_gate"),
      "selected_next_route": (
          SELECTED_NEXT_ROUTE if required else seq504_selected),
      "next_route_reason": (
          "Seq504's restored product-baseline distribution is numerically the "
          "same math/code failure as seq348/seq349, with the same clean product "
          "source/consumer counters. The scale-kernel branch is therefore a "
          "diagnostic side branch, not a new product root; continue from the "
          "selected layer-input source-value gap using the already recorded "
          "seq349 source evidence instead of rerunning old distribution rows."
          if required else
          "The reentry comparison is incomplete; keep the accepted-contiguous2 "
          "router distribution gap open."),
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
      "# Seq505 Accepted Contiguous2 Distribution Reentry Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- diagnostic_classification: `{metrics['diagnostic_classification']}`",
      f"- failed_checks: `{failed}`",
      f"- speedup_claims_allowed: `{str(metrics['speedup_claims_allowed']).lower()}`",
      "",
      "| case | seq504 max KLD | seq348 max KLD | seq349 max KLD | match |",
      "|---|---:|---:|---:|---|",
  ]
  for row in metrics["distribution_match"]["rows"]:
    lines.append(
        f"| `{row['case_id']}` | `{row['seq504']['max_kld']}` | "
        f"`{row['seq348']['max_kld']}` | `{row['seq349']['max_kld']}` | "
        f"`{str(row['match']).lower()}` |")
  source_gap = metrics["source_gap"]
  lines.extend([
      "",
      f"- selected layer-input min cosine: `{source_gap['min_layer_input_cosine']}`",
      f"- selected layer-input max abs diff: `{source_gap['max_layer_input_abs_diff']}`",
      f"- selected K/V history min cosine: `{source_gap['min_k_history_cosine']}` / `{source_gap['min_v_history_cosine']}`",
      "",
      metrics["next_route_reason"],
      "",
      "This is route-selection/correctness evidence only. It is not a speed claim.",
      "",
  ])
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--routes", type=Path, default=DEFAULT_ROUTES)
  parser.add_argument("--rejected", type=Path, default=DEFAULT_REJECTED)
  parser.add_argument("--seq348", type=Path, default=DEFAULT_SEQ348)
  parser.add_argument("--seq349", type=Path, default=DEFAULT_SEQ349)
  parser.add_argument("--seq504", type=Path, default=DEFAULT_SEQ504)
  parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  metrics = compute(args)
  write_outputs(metrics, args.out_dir)
  print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
