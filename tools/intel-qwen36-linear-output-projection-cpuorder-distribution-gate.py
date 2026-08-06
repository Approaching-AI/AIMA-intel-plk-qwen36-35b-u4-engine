#!/usr/bin/env python3
"""Reject or advance layer0 CPU-order projection after router-math."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = (
    "intel-qwen36-linear-output-projection-cpuorder-distribution-v0"
)
CURRENT_ROUTE = (
    "router_prompt_distribution_layer0_linear_output_projection_cpuorder_"
    "router_math_distribution_gate"
)
SELECTED_NEXT_ROUTE = (
    "router_prompt_distribution_post_arithmetic_parity_route_control_gate"
)
REJECTED_ROUTE = (
    "gpu_linear_attention_output_projection_cpu_order_reduction_diagnostic"
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


def _projection_rows(smoke: dict[str, Any]) -> list[dict[str, Any]]:
  output = []
  steps = smoke.get("linear_projection_source_diff_by_step")
  for step in steps if isinstance(steps, list) else []:
    layers = step.get("layers") if isinstance(step, dict) else []
    for row in layers if isinstance(layers, list) else []:
      if isinstance(row, dict) and row.get("layer") == 0:
        output.append({
            "token_index": step.get("token_index"),
            "gpu_vs_cpu_max_abs": row.get(
                "gpu_output_vs_cpu_projection_from_gpu_input_max_abs_diff"),
            "q8_qs_mismatches": row.get("q8_qs_mismatch_count"),
            "q8_bsums_mismatches": row.get("q8_bsums_mismatch_count"),
        })
  return output


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
  projection_rows = _projection_rows(candidate)
  checks = [
      {
          "name": "seq558_selected_router_math_distribution",
          "pass": (
              predecessor.get("required_checks_passed") is True
              and predecessor.get("router_math_distribution_allowed") is True
              and predecessor.get("selected_next_route") == CURRENT_ROUTE
              and _has_candidate(routes, 558, CURRENT_ROUTE)
              and _has_switch(
                  routes, 558,
                  "select_router_prompt_distribution_layer0_linear_output_"
                  "projection_cpuorder_router_math_distribution_gate")),
      },
      {
          "name": "bounded_eight_token_math_row_complete",
          "pass": (
              candidate_payload.get("target", {}).get("run", {}).get(
                  "returncode") == 2
              and candidate.get("case_id") == "router_math_reason_001"
              and candidate.get("decode_tokens_per_session") == 8
              and candidate.get("linear_output_projection_cpu_order_layers")
              == [0]
              and candidate_dist.get("position_count") == 8
              and candidate_dist.get("required_checks_passed") is False),
      },
      {
          "name": "layer0_projection_exact_on_every_token",
          "pass": (
              len(projection_rows) == 8
              and all(_num(row["gpu_vs_cpu_max_abs"], -1.0) == 0.0
                      and row["q8_qs_mismatches"] == 0
                      and row["q8_bsums_mismatches"] == 0
                      for row in projection_rows)),
      },
      {
          "name": "router_math_kld_regresses_vs_baseline",
          "pass": (
              _num(candidate_dist.get("max_kld")) > KLD_THRESHOLD
              and _num(candidate_dist.get("max_kld"))
              > _num(baseline_dist.get("max_kld"))),
      },
      {
          "name": "top1_preserved_but_does_not_override_kld",
          "pass": (
              candidate_dist.get("top1_pass") is True
              and _num(candidate_dist.get("top1_rate")) == 1.0),
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
      "candidate_top1_rate": candidate_dist.get("top1_rate"),
      "projection_rows": projection_rows,
      "router_code_distribution_allowed": False,
      "speedup_claims_allowed": False,
      "promotion_allowed": False,
      "disposition": (
          "reject_cpuorder_projection_select_post_parity_route_control"
          if required else
          "block_cpuorder_projection_distribution_inconsistent_evidence"),
      "rejected_route": REJECTED_ROUTE if required else None,
      "selected_next_route": SELECTED_NEXT_ROUTE if required else CURRENT_ROUTE,
      "next_route_reason": (
          "The layer0 projection is bit-exact against CPU on all eight tokens, "
          "but router-math max KLD regresses and token7 dominates. Exacting this "
          "local reduction is therefore not a distribution correction. Close "
          "the route without router-code or speed work and run route control "
          "after the arithmetic-parity board is exhausted."
          if required else
          "The router-math evidence is incomplete; keep the gate open."
      ),
  }


def write_outputs(metrics: dict[str, Any], out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  failed = [row["name"] for row in metrics["checks"] if not row["pass"]]
  lines = [
      f"# Seq{metrics['sequence']} Layer0 CPU-Order Projection Distribution Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- rejected_route: `{metrics['rejected_route']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- baseline -> candidate max KLD: "
      f"`{metrics['baseline_max_kld']}` -> `{metrics['candidate_max_kld']}`",
      f"- candidate top-1 rate: `{metrics['candidate_top1_rate']}`",
      f"- exact projection tokens: `{len(metrics['projection_rows'])}/8`",
      f"- failed_checks: `{failed}`",
      "",
      metrics["next_route_reason"],
      "",
      "This is router-math correctness evidence only. It is not a speed claim.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--sequence", type=int, default=559)
  parser.add_argument(
      "--predecessor", type=Path,
      default=ROOT / "output/seq558-layer0-linear-output-projection-cpuorder-one-token-gate-20260710Tseq558Z/metrics.json")
  parser.add_argument("--routes", type=Path,
                      default=ACTIVE / "routes-ledger.json")
  parser.add_argument(
      "--baseline", type=Path,
      default=ROOT / "output/seq541-layer0-exact-delta-source-math-20260710Tseq541Z/result.json")
  parser.add_argument(
      "--candidate", type=Path,
      default=ROOT / "output/seq559-layer0-linear-output-projection-cpuorder-router-math-20260710Tseq559Z/result.json")
  parser.add_argument(
      "--out-dir", type=Path,
      default=ROOT / "output/seq559-layer0-linear-output-projection-cpuorder-router-math-gate-20260710Tseq559Z")
  args = parser.parse_args()
  metrics = compute(args)
  write_outputs(metrics, args.out_dir)
  print(json.dumps({
      "required_checks_passed": metrics["required_checks_passed"],
      "disposition": metrics["disposition"],
      "router_code_distribution_allowed": metrics[
          "router_code_distribution_allowed"],
      "selected_next_route": metrics["selected_next_route"],
      "out_dir": _rel(args.out_dir),
  }, sort_keys=True))
  return 0 if metrics["required_checks_passed"] else 2


if __name__ == "__main__":
  raise SystemExit(main())
