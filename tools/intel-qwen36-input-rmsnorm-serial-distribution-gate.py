#!/usr/bin/env python3
"""Classify paired 8-token distribution for the layer0/1 serial island."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = "intel-qwen36-input-rmsnorm-serial-distribution-gate-v0"
KLD_THRESHOLD = 0.005
CURRENT_ROUTE = (
    "router_prompt_distribution_layer0_1_input_rmsnorm_serial_reduction_"
    "precision_island_router_distribution_gate"
)
SELECTED_NEXT_ROUTE = (
    "router_prompt_distribution_input_rmsnorm_serial_layer_subset_attribution_gate"
)
REJECTED_ROUTE = (
    "router_prompt_distribution_layer0_1_input_rmsnorm_serial_reduction_"
    "precision_island"
)
CASES = ("router_math_reason_001", "router_code_reason_002")


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


def _summary(payload: dict[str, Any]) -> dict[str, Any]:
  smoke = payload.get("smoke")
  smoke = smoke if isinstance(smoke, dict) else payload
  dist = smoke.get("distribution_ladder")
  dist = dist if isinstance(dist, dict) else {}
  steps = dist.get("steps")
  steps = steps if isinstance(steps, list) else []
  preconv = smoke.get("linear_preconv_source_diff_by_step")
  preconv = preconv if isinstance(preconv, list) else []
  exact_counts = {0: 0, 1: 0}
  observed_counts = {0: 0, 1: 0}
  for step in preconv:
    rows = step.get("layers") if isinstance(step, dict) else []
    rows = rows if isinstance(rows, list) else []
    for row in rows:
      layer = row.get("layer") if isinstance(row, dict) else None
      if layer not in exact_counts:
        continue
      value = row.get("gpu_attn_norm_vs_cpu_max_abs_diff")
      if isinstance(value, (int, float)):
        observed_counts[layer] += 1
        if float(value) == 0.0:
          exact_counts[layer] += 1
  return {
      "case_id": smoke.get("case_id"),
      "target_returncode": payload.get("target", {}).get("run", {}).get(
          "returncode"),
      "decode_tokens": smoke.get("decode_tokens_per_session"),
      "serial_layers": smoke.get("input_rmsnorm_serial_reduction_layers"),
      "final_norm_layers": smoke.get("linear_final_cpu_shape_layers"),
      "distribution_required_checks_passed": dist.get(
          "required_checks_passed"),
      "position_count": dist.get("position_count"),
      "max_kld": dist.get("max_kld"),
      "top1_rate": dist.get("top1_rate"),
      "top1_pass": dist.get("top1_pass"),
      "token_klds": [row.get("kld") for row in steps if isinstance(row, dict)],
      "layer0_exact_norm_tokens": exact_counts[0],
      "layer0_observed_norm_tokens": observed_counts[0],
      "layer1_exact_norm_tokens": exact_counts[1],
      "layer1_observed_norm_tokens": observed_counts[1],
  }


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


def compute(args: argparse.Namespace) -> dict[str, Any]:
  predecessor = _load(args.predecessor)
  routes = _load(args.routes)
  baselines = {
      "router_math_reason_001": _summary(_load(args.baseline_math)),
      "router_code_reason_002": _summary(_load(args.baseline_code)),
  }
  candidates = {
      "router_math_reason_001": _summary(_load(args.candidate_math)),
      "router_code_reason_002": _summary(_load(args.candidate_code)),
  }
  predecessor_selects = (
      predecessor.get("required_checks_passed") is True
      and predecessor.get("eight_token_router_distribution_allowed") is True
      and predecessor.get("selected_next_route") == CURRENT_ROUTE
      and _has_candidate(routes, 549, CURRENT_ROUTE)
      and _has_switch(
          routes, 549,
          "select_router_prompt_distribution_layer0_1_input_rmsnorm_"
          "serial_reduction_precision_island_router_distribution_gate"))
  rows_complete = all(
      candidates[case]["case_id"] == case
      and candidates[case]["target_returncode"] == 2
      and candidates[case]["decode_tokens"] == 8
      and candidates[case]["position_count"] == 8
      and candidates[case]["serial_layers"] == [0, 1]
      and candidates[case]["final_norm_layers"] == [4, 5, 6, 8, 9, 10]
      and candidates[case]["top1_pass"] is True
      and _num(candidates[case]["top1_rate"]) == 1.0
      for case in CASES)
  distribution_fails = all(
      candidates[case]["distribution_required_checks_passed"] is False
      and _num(candidates[case]["max_kld"]) > KLD_THRESHOLD
      for case in CASES)
  paired_max_kld_regresses = all(
      _num(candidates[case]["max_kld"])
      > _num(baselines[case]["max_kld"])
      for case in CASES)
  token_invariant_exactness_fails = (
      candidates["router_math_reason_001"]["layer0_observed_norm_tokens"] == 8
      and candidates["router_math_reason_001"]["layer0_exact_norm_tokens"] < 8)
  checks = [
      {"name": "seq549_selected_paired_distribution",
       "pass": predecessor_selects},
      {"name": "paired_eight_token_rows_complete", "pass": rows_complete},
      {"name": "paired_distribution_still_fails", "pass": distribution_fails},
      {"name": "paired_max_kld_regresses_vs_baseline",
       "pass": paired_max_kld_regresses},
      {"name": "layer0_serial_exactness_is_not_token_invariant",
       "pass": token_invariant_exactness_fails},
  ]
  required = all(bool(check["pass"]) for check in checks)
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "sequence": args.sequence,
      "inputs": {
          "predecessor": _rel(args.predecessor),
          "routes": _rel(args.routes),
          "baseline_math": _rel(args.baseline_math),
          "baseline_code": _rel(args.baseline_code),
          "candidate_math": _rel(args.candidate_math),
          "candidate_code": _rel(args.candidate_code),
      },
      "checks": checks,
      "required_checks_passed": required,
      "baselines": baselines,
      "candidates": candidates,
      "layer_subset_attribution_allowed": required,
      "speedup_claims_allowed": False,
      "promotion_allowed": False,
      "disposition": (
          "reject_combined_serial_island_select_layer_subset_attribution"
          if required else
          "block_serial_distribution_inconsistent_evidence"),
      "rejected_route": REJECTED_ROUTE if required else None,
      "selected_next_route": SELECTED_NEXT_ROUTE if required else CURRENT_ROUTE,
      "next_route_reason": (
          "The combined layer0/1 island preserves top-1 but regresses max KLD "
          "on both router prompts, and layer0 exactness is token-dependent. Use "
          "the existing selector for singleton layer0 and layer1 router-math "
          "rows to attribute main effects and interaction; no source change is needed."
          if required else
          "The paired distribution evidence is incomplete; keep this gate open."
      ),
  }


def write_outputs(metrics: dict[str, Any], out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  failed = [row["name"] for row in metrics["checks"] if not row["pass"]]
  math = metrics["candidates"]["router_math_reason_001"]
  code = metrics["candidates"]["router_code_reason_002"]
  lines = [
      f"# Seq{metrics['sequence']} Input-RMSNorm Serial Distribution Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- rejected_route: `{metrics['rejected_route']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- candidate math/code max KLD: `{math['max_kld']}` / `{code['max_kld']}`",
      f"- layer0 exact norm tokens: `{math['layer0_exact_norm_tokens']}/{math['layer0_observed_norm_tokens']}`",
      f"- failed_checks: `{failed}`",
      "",
      metrics["next_route_reason"],
      "",
      "This is paired correctness evidence only. It is not a speed claim.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--sequence", type=int, default=550)
  parser.add_argument(
      "--predecessor", type=Path,
      default=ROOT / "output/seq549-layer0-1-input-rmsnorm-serial-reduction-one-token-gate-20260710Tseq549Z/metrics.json")
  parser.add_argument("--routes", type=Path,
                      default=ACTIVE / "routes-ledger.json")
  parser.add_argument(
      "--baseline-math", type=Path,
      default=ROOT / "output/seq541-layer0-exact-delta-source-math-20260710Tseq541Z/result.json")
  parser.add_argument(
      "--baseline-code", type=Path,
      default=ROOT / "output/seq541-layer0-exact-delta-source-code-20260710Tseq541Z/result.json")
  parser.add_argument(
      "--candidate-math", type=Path,
      default=ROOT / "output/seq550-layer0-1-input-rmsnorm-serial-reduction-router-math-20260710Tseq550Z/result.json")
  parser.add_argument(
      "--candidate-code", type=Path,
      default=ROOT / "output/seq550-layer0-1-input-rmsnorm-serial-reduction-router-code-20260710Tseq550Z/result.json")
  parser.add_argument(
      "--out-dir", type=Path,
      default=ROOT / "output/seq550-layer0-1-input-rmsnorm-serial-reduction-router-distribution-gate-20260710Tseq550Z")
  args = parser.parse_args()
  metrics = compute(args)
  write_outputs(metrics, args.out_dir)
  print(json.dumps({
      "required_checks_passed": metrics["required_checks_passed"],
      "disposition": metrics["disposition"],
      "layer_subset_attribution_allowed": metrics[
          "layer_subset_attribution_allowed"],
      "selected_next_route": metrics["selected_next_route"],
      "out_dir": _rel(args.out_dir),
  }, sort_keys=True))
  return 0 if metrics["required_checks_passed"] else 2


if __name__ == "__main__":
  raise SystemExit(main())
