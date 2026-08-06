#!/usr/bin/env python3
"""Classify one router-math token with layer0 CPU-order projection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = (
    "intel-qwen36-linear-output-projection-cpuorder-one-token-v0"
)
CURRENT_ROUTE = (
    "router_prompt_distribution_layer0_linear_output_projection_cpuorder_"
    "one_token_probe_gate"
)
SELECTED_NEXT_ROUTE = (
    "router_prompt_distribution_layer0_linear_output_projection_cpuorder_"
    "router_math_distribution_gate"
)
KLD_THRESHOLD = 0.005


def _load(path: Path) -> dict[str, Any]:
  payload = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(payload, dict):
    raise TypeError(f"{path} does not contain a JSON object")
  return payload


def _rel(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path)


def _num(value: Any, default: float = 0.0) -> float:
  return float(value) if isinstance(value, (int, float)) else default


def _smoke(payload: dict[str, Any]) -> dict[str, Any]:
  smoke = payload.get("smoke")
  return smoke if isinstance(smoke, dict) else payload


def _has_candidate(routes: dict[str, Any], seq: int,
                   next_route: str) -> bool:
  return any(
      isinstance(row, dict)
      and row.get("seq") == seq
      and row.get("selected_next_route") == next_route
      for row in routes.get("candidate_history", []))


def _has_switch(routes: dict[str, Any], seq: int, decision: str) -> bool:
  return any(
      isinstance(row, dict)
      and row.get("seq_covered") == seq
      and row.get("decision") == decision
      and row.get("resolved") is True
      for row in routes.get("switch_decisions", []))


def _layer(rows: Any, layer_id: int) -> dict[str, Any]:
  if not isinstance(rows, list) or not rows:
    return {}
  layers = rows[0].get("layers") if isinstance(rows[0], dict) else []
  for row in layers if isinstance(layers, list) else []:
    if isinstance(row, dict) and row.get("layer") == layer_id:
      return row
  return {}


def compute(args: argparse.Namespace) -> dict[str, Any]:
  predecessor = _load(args.predecessor)
  routes = _load(args.routes)
  baseline = _smoke(_load(args.baseline))
  candidate_payload = _load(args.candidate)
  candidate = _smoke(candidate_payload)
  baseline_dist = baseline.get("distribution_ladder")
  baseline_dist = baseline_dist if isinstance(baseline_dist, dict) else {}
  candidate_dist = candidate.get("distribution_ladder")
  candidate_dist = candidate_dist if isinstance(candidate_dist, dict) else {}
  projection = _layer(candidate.get("linear_projection_source_diff_by_step"), 0)
  boundary = _layer(candidate.get("layer_boundary_diff_by_step"), 0)
  checks = [
      {
          "name": "seq557_selected_one_token_gate",
          "pass": (
              predecessor.get("required_checks_passed") is True
              and predecessor.get("one_token_probe_allowed") is True
              and predecessor.get("selected_next_route") == CURRENT_ROUTE
              and _has_candidate(routes, 557, CURRENT_ROUTE)
              and _has_switch(
                  routes, 557,
                  "select_router_prompt_distribution_layer0_linear_output_"
                  "projection_cpuorder_one_token_probe_gate")),
      },
      {
          "name": "bounded_router_math_one_token_row_passed",
          "pass": (
              candidate_payload.get("target", {}).get("run", {}).get(
                  "returncode") == 0
              and candidate.get("required_checks_passed") is True
              and candidate.get("case_id") == "router_math_reason_001"
              and candidate.get("decode_tokens_per_session") == 1
              and candidate.get("linear_output_projection_cpu_order_layers")
              == [0]
              and candidate_dist.get("required_checks_passed") is True
              and candidate_dist.get("position_count") == 1
              and candidate_dist.get("top1_pass") is True
              and _num(candidate_dist.get("top1_rate")) == 1.0
              and _num(candidate_dist.get("max_kld")) <= KLD_THRESHOLD),
      },
      {
          "name": "one_token_kld_improves_vs_seq541",
          "pass": (
              _num(candidate_dist.get("max_kld"))
              < _num(baseline_dist.get("max_kld"))),
      },
      {
          "name": "layer0_projection_is_exact_from_live_input",
          "pass": (
              projection.get("available") is True
              and _num(projection.get(
                  "gpu_output_vs_cpu_projection_from_gpu_input_max_abs_diff"),
                  -1.0) == 0.0
              and projection.get("q8_qs_mismatch_count") == 0
              and projection.get("q8_bsums_mismatch_count") == 0),
      },
      {
          "name": "layer0_starts_exact_and_output_remains_sub_ulp",
          "pass": (
              _num(boundary.get("input_max_abs_diff"), -1.0) == 0.0
              and 0.0 < _num(boundary.get("output_max_abs_diff"))
              <= 1.192092896e-7),
      },
  ]
  required = all(bool(row["pass"]) for row in checks)
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "sequence": args.sequence,
      "inputs": {
          "predecessor": _rel(args.predecessor),
          "routes": _rel(args.routes),
          "baseline": _rel(args.baseline),
          "candidate": _rel(args.candidate),
      },
      "checks": checks,
      "required_checks_passed": required,
      "baseline_max_kld": baseline_dist.get("max_kld"),
      "candidate_max_kld": candidate_dist.get("max_kld"),
      "layer0_projection": projection,
      "layer0_boundary": boundary,
      "router_math_distribution_allowed": required,
      "router_code_distribution_allowed": False,
      "speedup_claims_allowed": False,
      "promotion_allowed": False,
      "disposition": (
          "accept_one_token_select_router_math_distribution"
          if required else
          "reject_or_block_cpuorder_projection_one_token"),
      "selected_next_route": SELECTED_NEXT_ROUTE if required else CURRENT_ROUTE,
      "next_route_reason": (
          "Layer0 CPU-order projection makes the live-input projection exact, "
          "preserves top-1, and moves token-0 KLD below the ruler. Run only the "
          "8-token router-math distribution row next. Router-code, speed, "
          "promotion, and context expansion remain blocked until math passes."
          if required else
          "The one-token row or exact projection evidence failed; do not continue."
      ),
  }


def write_outputs(metrics: dict[str, Any], out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  failed = [row["name"] for row in metrics["checks"] if not row["pass"]]
  lines = [
      f"# Seq{metrics['sequence']} Layer0 CPU-Order Projection One-Token Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- baseline -> candidate max KLD: "
      f"`{metrics['baseline_max_kld']}` -> `{metrics['candidate_max_kld']}`",
      f"- layer0 projection GPU/CPU max abs: "
      f"`{metrics['layer0_projection'].get('gpu_output_vs_cpu_projection_from_gpu_input_max_abs_diff')}`",
      f"- failed_checks: `{failed}`",
      "",
      metrics["next_route_reason"],
      "",
      "This is one-token correctness evidence only. It is not a speed claim.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--sequence", type=int, default=558)
  parser.add_argument(
      "--predecessor", type=Path,
      default=ROOT / "output/seq557-layer0-linear-output-projection-cpuorder-target-compile-gate-20260710Tseq557Z/metrics.json")
  parser.add_argument("--routes", type=Path,
                      default=ACTIVE / "routes-ledger.json")
  parser.add_argument(
      "--baseline", type=Path,
      default=ROOT / "output/seq541-layer0-exact-delta-source-math-20260710Tseq541Z/result.json")
  parser.add_argument(
      "--candidate", type=Path,
      default=ROOT / "output/seq558-layer0-linear-output-projection-cpuorder-one-token-20260710Tseq558Z/result.json")
  parser.add_argument(
      "--out-dir", type=Path,
      default=ROOT / "output/seq558-layer0-linear-output-projection-cpuorder-one-token-gate-20260710Tseq558Z")
  args = parser.parse_args()
  metrics = compute(args)
  write_outputs(metrics, args.out_dir)
  print(json.dumps({
      "required_checks_passed": metrics["required_checks_passed"],
      "disposition": metrics["disposition"],
      "router_math_distribution_allowed": metrics[
          "router_math_distribution_allowed"],
      "selected_next_route": metrics["selected_next_route"],
      "out_dir": _rel(args.out_dir),
  }, sort_keys=True))
  return 0 if metrics["required_checks_passed"] else 2


if __name__ == "__main__":
  raise SystemExit(main())
