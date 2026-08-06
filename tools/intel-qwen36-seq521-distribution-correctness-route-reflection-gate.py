#!/usr/bin/env python3
"""Reflect on the router distribution correctness route after seq520."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = (
    "intel-qwen36-seq521-distribution-correctness-route-reflection-v0"
)

DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_REJECTED = ACTIVE / "rejected-routes.json"
DEFAULT_ACCEPTANCE = (
    ROOT / "benchmarks" / WORKSTREAM / "acceptance-matrix.json"
)
DEFAULT_INSPIRATION = (
    ROOT / "doc/reference" / WORKSTREAM
    / "route-inspiration-from-siblings-2026-06-29.md"
)
DEFAULT_SEQ520 = (
    ROOT
    / "output/seq520-linear-conv-history-unclosed-source-audit-gate-20260709Tseq520Z"
    / "metrics.json"
)
DEFAULT_SEQ222 = (
    ROOT
    / "output/r2-gpu-acceptance-matrix-router-both-rowblock16-26mask-20260708Tseq222Z"
)
DEFAULT_SEQ223 = (
    ROOT
    / "output/r2-gpu-router-math-distribution-no-rowblock16-20260708Tseq223Z"
    / "result.json"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "output/seq521-distribution-correctness-route-reflection-gate-20260709Tseq521Z"
)

KLD_THRESHOLD = 0.005
TOP1_THRESHOLD = 0.99
TOKEN_CASES = (
    "router_math_reason_001",
    "router_code_reason_002",
    "router_instruction_003",
)
FAILING_DISTRIBUTION_CASES = {
    "router_math_reason_001",
    "router_code_reason_002",
}
PASSING_DISTRIBUTION_CASES = {"router_instruction_003"}
SELECTED_NEXT_ROUTE = "router_prompt_distribution_fp64_sensitivity_gate"
CURRENT_ROUTE = "router_prompt_distribution_correctness_route_reflection_gate"

REQUIRED_CLOSED_SOURCE_ROUTES = {
    "linear_conv_history_known_product_source_audit",
    "qkv_delta_blockq16_no_product_value_source",
    "all_linear_state_direct_product_refresh",
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


def _rejected_names(rejected: dict[str, Any]) -> set[str]:
  names: set[str] = set()
  for row in rejected.get("rejected", []):
    if isinstance(row, dict) and isinstance(row.get("route"), str):
      names.add(row["route"])
  return names


def _seen_route(routes: dict[str, Any], rejected: dict[str, Any],
                route_name: str) -> bool:
  if route_name in _rejected_names(rejected):
    return True
  for section in ("candidate_history", "switch_decisions", "parked_routes"):
    rows = routes.get(section)
    if not isinstance(rows, list):
      continue
    for row in rows:
      if not isinstance(row, dict):
        continue
      if route_name in {
          row.get("name"),
          row.get("selected_next_route"),
          row.get("decision"),
          row.get("route"),
      }:
        return True
  return False


def _smoke(path: Path) -> dict[str, Any]:
  payload = _load_json(path)
  smoke = payload.get("smoke")
  return smoke if isinstance(smoke, dict) else payload


def _dist(smoke: dict[str, Any]) -> dict[str, Any]:
  dist = smoke.get("distribution_ladder")
  return dist if isinstance(dist, dict) else {}


def _run_row(path: Path, lane: str) -> dict[str, Any]:
  smoke = _smoke(path)
  dist = _dist(smoke)
  return {
      "path": _rel(path),
      "case_id": smoke.get("case_id"),
      "lane": lane,
      "required_checks_passed": smoke.get("required_checks_passed"),
      "decode_tokens": smoke.get("decode_continuation_output_tokens"),
      "top1_matches_native": smoke.get("top1_matches_native"),
      "top1_match_count": smoke.get("top1_match_count"),
      "topk_match_count": smoke.get("topk_match_count"),
      "greedy_prefix_match_count": smoke.get("greedy_prefix_match_count"),
      "distribution_required_checks_passed": dist.get("required_checks_passed"),
      "distribution_top1_rate": dist.get("top1_rate"),
      "distribution_top1_match_count": dist.get("top1_match_count"),
      "distribution_kld_pass": dist.get("kld_pass"),
      "distribution_top1_pass": dist.get("top1_pass"),
      "max_kld": dist.get("max_kld"),
      "mean_kld": dist.get("mean_kld"),
      "min_logits_cosine": dist.get("min_logits_cosine"),
      "thresholds": dist.get("thresholds"),
      "rowblock16_enabled": smoke.get(
          "attention_front_output_projection_rowblock16_enabled"),
      "rowblock16_layers": smoke.get(
          "attention_front_output_projection_rowblock16_layers"),
  }


def _seq222_rows(seq222_dir: Path) -> dict[str, list[dict[str, Any]]]:
  rows: dict[str, list[dict[str, Any]]] = {
      "token_exact": [],
      "distribution": [],
  }
  for case_id in TOKEN_CASES:
    for lane in rows:
      path = seq222_dir / "runs" / f"{case_id}-{lane}" / "result.json"
      rows[lane].append(_run_row(path, lane))
  return rows


def _token_exact_top1_pass(rows: list[dict[str, Any]]) -> bool:
  return (
      len(rows) == len(TOKEN_CASES)
      and {str(row.get("case_id")) for row in rows} == set(TOKEN_CASES)
      and all(row.get("top1_matches_native") is True for row in rows)
      and all(_num(row.get("top1_match_count")) == _num(row.get("decode_tokens"))
              for row in rows)
      and all(_num(row.get("greedy_prefix_match_count"))
              == _num(row.get("decode_tokens")) for row in rows))


def _router_distribution_is_kld_block(
    rows: list[dict[str, Any]]) -> bool:
  by_case = {str(row.get("case_id")): row for row in rows}
  if set(TOKEN_CASES) != set(by_case):
    return False
  failing_ok = all(
      _num(by_case[case_id].get("max_kld")) > KLD_THRESHOLD
      and _num(by_case[case_id].get("distribution_top1_rate"))
      >= TOP1_THRESHOLD
      and by_case[case_id].get("distribution_kld_pass") is False
      and by_case[case_id].get("distribution_top1_pass") is True
      for case_id in FAILING_DISTRIBUTION_CASES)
  passing_ok = all(
      _num(by_case[case_id].get("max_kld")) <= KLD_THRESHOLD
      and _num(by_case[case_id].get("distribution_top1_rate"))
      >= TOP1_THRESHOLD
      and by_case[case_id].get("distribution_kld_pass") is True
      and by_case[case_id].get("distribution_top1_pass") is True
      for case_id in PASSING_DISTRIBUTION_CASES)
  return failing_ok and passing_ok


def _no_rowblock_row(path: Path) -> dict[str, Any]:
  return _run_row(path, "distribution_no_rowblock16")


def _no_rowblock_not_root(seq222_rows: list[dict[str, Any]],
                          seq223_row: dict[str, Any]) -> bool:
  math_row = next(
      (row for row in seq222_rows
       if row.get("case_id") == "router_math_reason_001"),
      None)
  return (
      isinstance(math_row, dict)
      and seq223_row.get("rowblock16_enabled") is False
      and _num(seq223_row.get("max_kld")) > _num(math_row.get("max_kld"))
      and _num(seq223_row.get("distribution_top1_rate")) >= TOP1_THRESHOLD)


def _acceptance_distribution_required(acceptance: dict[str, Any]) -> bool:
  accuracy = acceptance.get("accuracy")
  if not isinstance(accuracy, dict):
    return False
  dist = accuracy.get("teacher_forced_distribution")
  tokens = accuracy.get("tokens")
  if not isinstance(dist, dict) or not isinstance(tokens, dict):
    return False
  return (
      abs(_num(dist.get("kl_divergence_max")) - KLD_THRESHOLD) < 1e-12
      and abs(_num(dist.get("top1_min")) - TOP1_THRESHOLD) < 1e-12
      and tokens.get("deterministic_greedy_exact_match_required") is True
      and tokens.get("first_divergence_blocks_promotion") is True)


def _inspiration_supports_fp64_reflection(text: str) -> bool:
  return (
      "Keep correctness on the teacher-forced distribution gate" in text
      and "run an FP64 sensitivity check before polishing it" in text)


def compute(args: argparse.Namespace) -> dict[str, Any]:
  routes = _load_json(args.routes)
  rejected = _load_json(args.rejected)
  seq520 = _load_json(args.seq520)
  acceptance = _load_json(args.acceptance)
  inspiration = args.inspiration.read_text(encoding="utf-8")
  seq222 = _seq222_rows(args.seq222_dir)
  seq223 = _no_rowblock_row(args.seq223)
  rejected_names = _rejected_names(rejected)
  missing_closed = sorted(REQUIRED_CLOSED_SOURCE_ROUTES - rejected_names)
  fp64_route_seen = _seen_route(routes, rejected, SELECTED_NEXT_ROUTE)

  checks = [
      {
          "name": "seq520_selected_correctness_route_reflection",
          "pass": (
              seq520.get("required_checks_passed") is True
              and seq520.get("selected_next_route") == CURRENT_ROUTE
              and seq520.get("known_product_source_classes_exhausted") is True
              and _has_candidate(
                  routes, 520,
                  "close_known_conv_history_product_source_classes_select_correctness_route_reflection")
              and _has_switch(
                  routes,
                  "select_router_prompt_distribution_correctness_route_reflection_gate",
                  520)),
          "detail": {
              "seq520_disposition": seq520.get("disposition"),
              "seq520_selected_next_route": seq520.get("selected_next_route"),
          },
      },
      {
          "name": "acceptance_matrix_keeps_distribution_kld_gate",
          "pass": _acceptance_distribution_required(acceptance),
          "detail": {
              "teacher_forced_distribution":
              acceptance.get("accuracy", {}).get("teacher_forced_distribution"),
              "tokens": acceptance.get("accuracy", {}).get("tokens"),
          },
      },
      {
          "name": "router_prompt_token_exact_top1_already_passes",
          "pass": _token_exact_top1_pass(seq222["token_exact"]),
          "detail": {"token_exact_rows": seq222["token_exact"]},
      },
      {
          "name": "router_prompt_distribution_block_is_kld_not_greedy_top1",
          "pass": _router_distribution_is_kld_block(seq222["distribution"]),
          "detail": {"distribution_rows": seq222["distribution"]},
      },
      {
          "name": "rowblock16_mask_route_is_not_the_remaining_root",
          "pass": _no_rowblock_not_root(seq222["distribution"], seq223),
          "detail": {
              "rowblock16_math": next(
                  row for row in seq222["distribution"]
                  if row.get("case_id") == "router_math_reason_001"),
              "no_rowblock16_math": seq223,
          },
      },
      {
          "name": "known_source_routes_are_closed",
          "pass": not missing_closed,
          "detail": {"missing_closed_routes": missing_closed},
      },
      {
          "name": "route_inspiration_requires_kld_ruler_and_fp64_sensitivity",
          "pass": _inspiration_supports_fp64_reflection(inspiration),
          "detail": {
              "path": _rel(args.inspiration),
              "reference": (
                  "Keep teacher-forced distribution KLD/top1; after a "
                  "divergent boundary, run FP64 sensitivity before polishing."
              ),
          },
      },
      {
          "name": "current_fp64_sensitivity_route_not_yet_recorded",
          "pass": not fp64_route_seen,
          "detail": {"selected_next_route_seen": fp64_route_seen},
      },
  ]
  required = all(bool(row.get("pass")) for row in checks)
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "inputs": {
          "routes": _rel(args.routes),
          "rejected": _rel(args.rejected),
          "acceptance": _rel(args.acceptance),
          "inspiration": _rel(args.inspiration),
          "seq520": _rel(args.seq520),
          "seq222_dir": _rel(args.seq222_dir),
          "seq223": _rel(args.seq223),
      },
      "checks": checks,
      "required_checks_passed": required,
      "speedup_claims_allowed": False,
      "decode_probe_allowed": False,
      "router_distribution_allowed": False,
      "fp64_sensitivity_probe_allowed": required,
      "top1_only_override_allowed": False,
      "known_source_routes_closed": sorted(REQUIRED_CLOSED_SOURCE_ROUTES),
      "router_prompt_rows": seq222,
      "rowblock16_control": seq223,
      "disposition": (
          "accept_correctness_route_reflection_select_fp64_sensitivity_gate"
          if required else
          "block_correctness_route_reflection_inconsistent_evidence"),
      "selected_next_route": (
          SELECTED_NEXT_ROUTE if required else CURRENT_ROUTE),
      "next_route_reason": (
          "The router prompt set is not a greedy-token problem: seq222 already "
          "preserves top-1 for all token-exact rows and top-1 in distribution, "
          "but math/code exceed the accepted KLD ruler. The acceptance matrix "
          "requires teacher-forced distribution KLD <= 0.005, so a top1-only "
          "override is invalid. Seq223 shows disabling rowblock16 worsens the "
          "router_math distribution row, and seq520 closes the known non-shadow "
          "source board. The next high-signal unit is a bounded FP64/numerical "
          "sensitivity gate for the divergent router distribution boundary; no "
          "decode, router-distribution rerun, speed promotion, or long-context "
          "row is allowed before that sensitivity result selects the next "
          "route."
          if required else
          "Correctness-route reflection evidence is inconsistent; do not launch "
          "a token row or weaken the distribution ruler."),
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
  dist_rows = metrics["router_prompt_rows"]["distribution"]
  math = next(row for row in dist_rows
              if row.get("case_id") == "router_math_reason_001")
  code = next(row for row in dist_rows
              if row.get("case_id") == "router_code_reason_002")
  instruction = next(row for row in dist_rows
                     if row.get("case_id") == "router_instruction_003")
  no_rowblock = metrics["rowblock16_control"]
  lines = [
      "# Seq521 Distribution Correctness Route Reflection Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- failed_checks: `{failed}`",
      f"- fp64_sensitivity_probe_allowed: `{str(metrics['fp64_sensitivity_probe_allowed']).lower()}`",
      f"- top1_only_override_allowed: `{str(metrics['top1_only_override_allowed']).lower()}`",
      f"- speedup_claims_allowed: `{str(metrics['speedup_claims_allowed']).lower()}`",
      "",
      "## Evidence",
      "",
      f"- seq222 router math/code KLD: `{math['max_kld']}` / `{code['max_kld']}`",
      f"- seq222 router instruction KLD: `{instruction['max_kld']}`",
      "- seq222 token-exact and distribution top-1 rows preserve greedy top-1",
      f"- seq223 no-rowblock16 router_math KLD: `{no_rowblock['max_kld']}`",
      "- seq520 closed the known conv-history/upstream product-source board",
      "",
      metrics["next_route_reason"],
      "",
      "This is route-control/correctness evidence only. It is not a speed claim.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--routes", type=Path, default=DEFAULT_ROUTES)
  parser.add_argument("--rejected", type=Path, default=DEFAULT_REJECTED)
  parser.add_argument("--acceptance", type=Path, default=DEFAULT_ACCEPTANCE)
  parser.add_argument("--inspiration", type=Path, default=DEFAULT_INSPIRATION)
  parser.add_argument("--seq520", type=Path, default=DEFAULT_SEQ520)
  parser.add_argument("--seq222-dir", type=Path, default=DEFAULT_SEQ222)
  parser.add_argument("--seq223", type=Path, default=DEFAULT_SEQ223)
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
