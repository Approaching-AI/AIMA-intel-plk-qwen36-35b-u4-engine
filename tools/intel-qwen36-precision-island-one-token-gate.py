#!/usr/bin/env python3
"""Classify the bounded one-token GPU final-norm precision-island probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = "intel-qwen36-precision-island-one-token-gate-v0"
KLD_THRESHOLD = 0.005
CURRENT_ROUTE = (
    "router_prompt_distribution_layer0_1_gpu_final_norm_precision_island_one_token_probe_gate"
)
SELECTED_NEXT_ROUTE = (
    "router_prompt_distribution_layer0_1_input_rmsnorm_reduction_precision_island_feasibility_gate"
)
REJECTED_ROUTE = (
    "router_prompt_distribution_layer0_1_gpu_final_norm_precision_island"
)
EXPECTED_LAYERS = [0, 1, 4, 5, 6, 8, 9, 10]


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


def _layer(smoke: dict[str, Any], table: str, token: int,
           layer: int) -> dict[str, Any]:
  steps = smoke.get(table)
  steps = steps if isinstance(steps, list) else []
  for step in steps:
    if not isinstance(step, dict) or step.get("token_index") != token:
      continue
    rows = step.get("layers")
    rows = rows if isinstance(rows, list) else []
    for row in rows:
      if isinstance(row, dict) and row.get("layer") == layer:
        return row
  return {}


def _summary(payload: dict[str, Any]) -> dict[str, Any]:
  smoke = _smoke(payload)
  dist = smoke.get("distribution_ladder")
  dist = dist if isinstance(dist, dict) else {}
  dist_steps = dist.get("steps")
  dist_steps = dist_steps if isinstance(dist_steps, list) else []
  token0_dist = next((
      step for step in dist_steps
      if isinstance(step, dict) and step.get("token_index") == 0
  ), {})
  boundary0 = _layer(smoke, "layer_boundary_diff_by_step", 0, 0)
  boundary1 = _layer(smoke, "layer_boundary_diff_by_step", 0, 1)
  linear0 = _layer(smoke, "linear_attention_diff_by_step", 0, 0)
  projection0 = _layer(smoke, "linear_projection_source_diff_by_step", 0, 0)
  boundary_steps = smoke.get("layer_boundary_diff_by_step")
  boundary_step = (
      boundary_steps[0]
      if isinstance(boundary_steps, list) and boundary_steps
      and isinstance(boundary_steps[0], dict)
      else {}
  )
  return {
      "case_id": smoke.get("case_id"),
      "target_returncode": payload.get("target", {}).get("run", {}).get(
          "returncode"),
      "decode_tokens": smoke.get("decode_tokens_per_session"),
      "active_layers": smoke.get("linear_final_cpu_shape_layers"),
      "cpu_shadow_state": smoke.get("cpu_shadow_state_each_token_enabled"),
      "distribution_position_count": dist.get("position_count"),
      "distribution_required_checks_passed": dist.get("required_checks_passed"),
      "max_kld": token0_dist.get("kld"),
      "top1_rate": dist.get("top1_rate"),
      "top1_pass": dist.get("top1_pass"),
      "first_input_cosine_below_9999": boundary_step.get(
          "first_input_cosine_below_9999"),
      "first_output_cosine_below_9999": boundary_step.get(
          "first_output_cosine_below_9999"),
      "layer0_input_max_abs_diff": boundary0.get("input_max_abs_diff"),
      "layer0_output_max_abs_diff": boundary0.get("output_max_abs_diff"),
      "layer1_output_max_abs_diff": boundary1.get("output_max_abs_diff"),
      "layer0_attn_norm_max_abs_diff": linear0.get("attn_norm_max_abs_diff"),
      "layer0_delta_output_max_abs_diff": linear0.get("delta_output_max_abs_diff"),
      "layer0_final_output_max_abs_diff": linear0.get("final_output_max_abs_diff"),
      "layer0_projection_q8_qs_mismatch_count": projection0.get(
          "q8_qs_mismatch_count"),
      "layer0_projection_q8_bsums_mismatch_count": projection0.get(
          "q8_bsums_mismatch_count"),
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


def _ratio(numerator: Any, denominator: Any) -> float:
  den = _num(denominator)
  return _num(numerator) / den if den > 0.0 else 0.0


def compute(args: argparse.Namespace) -> dict[str, Any]:
  predecessor = _load(args.predecessor)
  routes = _load(args.routes)
  baseline = _summary(_load(args.baseline))
  candidate = _summary(_load(args.candidate))
  kld_ratio = _ratio(candidate["max_kld"], baseline["max_kld"])
  layer0_boundary_ratio = _ratio(
      candidate["layer0_output_max_abs_diff"],
      baseline["layer0_output_max_abs_diff"])
  layer0_final_ratio = _ratio(
      candidate["layer0_final_output_max_abs_diff"],
      baseline["layer0_final_output_max_abs_diff"])
  layer1_boundary_ratio = _ratio(
      candidate["layer1_output_max_abs_diff"],
      baseline["layer1_output_max_abs_diff"])
  predecessor_selects = (
      predecessor.get("required_checks_passed") is True
      and predecessor.get("one_token_probe_allowed") is True
      and predecessor.get("selected_next_route") == CURRENT_ROUTE
      and _has_candidate(routes, 544, CURRENT_ROUTE)
      and _has_switch(
          routes, 544,
          "select_router_prompt_distribution_layer0_1_gpu_final_norm_precision_island_one_token_probe_gate")
  )
  candidate_ran = (
      candidate["case_id"] == "router_math_reason_001"
      and candidate["target_returncode"] == 2
      and candidate["decode_tokens"] == 1
      and candidate["distribution_position_count"] == 1
      and candidate["active_layers"] == EXPECTED_LAYERS
      and candidate["cpu_shadow_state"] is False
      and candidate["top1_pass"] is True
      and _num(candidate["top1_rate"]) == 1.0
  )
  threshold_still_fails = (
      candidate["distribution_required_checks_passed"] is False
      and _num(candidate["max_kld"]) > KLD_THRESHOLD
  )
  early_root_not_closed = (
      _num(candidate["layer0_input_max_abs_diff"]) == 0.0
      and candidate["layer0_attn_norm_max_abs_diff"]
      == baseline["layer0_attn_norm_max_abs_diff"]
      and layer0_final_ratio >= 0.95
      and layer1_boundary_ratio >= 0.95
      and candidate["first_input_cosine_below_9999"]
      == baseline["first_input_cosine_below_9999"]
      and candidate["first_output_cosine_below_9999"]
      == baseline["first_output_cosine_below_9999"]
      and candidate["layer0_projection_q8_qs_mismatch_count"] == 0
      and candidate["layer0_projection_q8_bsums_mismatch_count"] == 0
  )
  checks = [
      {"name": "seq544_selected_one_token_probe", "pass": predecessor_selects},
      {"name": "bounded_candidate_row_ran", "pass": candidate_ran},
      {"name": "candidate_still_fails_kld_threshold", "pass": threshold_still_fails},
      {"name": "layer0_1_early_root_not_closed", "pass": early_root_not_closed},
  ]
  required = all(bool(check["pass"]) for check in checks)
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
      "baseline": baseline,
      "candidate": candidate,
      "metrics": {
          "kld_threshold": KLD_THRESHOLD,
          "kld_ratio_candidate_over_baseline": kld_ratio,
          "kld_relative_change": kld_ratio - 1.0,
          "layer0_boundary_abs_ratio": layer0_boundary_ratio,
          "layer0_linear_final_abs_ratio": layer0_final_ratio,
          "layer1_boundary_abs_ratio": layer1_boundary_ratio,
      },
      "eight_token_probe_allowed": False,
      "router_distribution_allowed": False,
      "speedup_claims_allowed": False,
      "disposition": (
          "reject_gpu_final_norm_precision_island_select_input_rmsnorm_reduction_feasibility"
          if required else
          "block_precision_island_one_token_inconsistent_evidence"
      ),
      "rejected_route": REJECTED_ROUTE if required else None,
      "selected_next_route": SELECTED_NEXT_ROUTE if required else CURRENT_ROUTE,
      "next_route_reason": (
          "The one-token candidate improves KLD but still fails the formal ruler, "
          "does not move the layer0 attention-norm mismatch or linear-final delta, "
          "and leaves layer1 amplification unchanged. Reject the 8-token continuation; "
          "the next source-only question is a layer0/1 input-RMSNorm reduction island."
          if required else
          "The bounded evidence is inconsistent; keep the one-token gate open."
      ),
  }


def write_outputs(metrics: dict[str, Any], out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  failed = [row["name"] for row in metrics["checks"] if not row["pass"]]
  lines = [
      f"# Seq{metrics['sequence']} Precision-Island One-Token Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- rejected_route: `{metrics['rejected_route']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- candidate max KLD: `{metrics['candidate']['max_kld']}`",
      f"- baseline token-0 KLD: `{metrics['baseline']['max_kld']}`",
      f"- eight_token_probe_allowed: `{str(metrics['eight_token_probe_allowed']).lower()}`",
      f"- failed_checks: `{failed}`",
      "",
      metrics["next_route_reason"],
      "",
      "This is bounded correctness evidence only. It is not a speed claim.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--sequence", type=int, default=545)
  parser.add_argument(
      "--predecessor", type=Path,
      default=ROOT / "output/seq544-layer0-1-gpu-final-norm-precision-island-target-compile-gate-20260710Tseq544Z/metrics.json")
  parser.add_argument("--routes", type=Path,
                      default=ACTIVE / "routes-ledger.json")
  parser.add_argument(
      "--baseline", type=Path,
      default=ROOT / "output/seq541-layer0-exact-delta-source-math-20260710Tseq541Z/result.json")
  parser.add_argument(
      "--candidate", type=Path,
      default=ROOT / "output/seq545-layer0-1-gpu-final-norm-precision-island-one-token-20260710Tseq545Z/result.json")
  parser.add_argument(
      "--out-dir", type=Path,
      default=ROOT / "output/seq545-layer0-1-gpu-final-norm-precision-island-one-token-gate-20260710Tseq545Z")
  args = parser.parse_args()
  metrics = compute(args)
  write_outputs(metrics, args.out_dir)
  print(json.dumps({
      "required_checks_passed": metrics["required_checks_passed"],
      "disposition": metrics["disposition"],
      "eight_token_probe_allowed": metrics["eight_token_probe_allowed"],
      "selected_next_route": metrics["selected_next_route"],
      "out_dir": _rel(args.out_dir),
  }, sort_keys=True))
  return 0 if metrics["required_checks_passed"] else 2


if __name__ == "__main__":
  raise SystemExit(main())
