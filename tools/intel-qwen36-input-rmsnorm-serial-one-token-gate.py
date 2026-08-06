#!/usr/bin/env python3
"""Classify the bounded layer0/1 serial-RMSNorm one-token probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = "intel-qwen36-input-rmsnorm-serial-one-token-gate-v0"
KLD_THRESHOLD = 0.005
CURRENT_ROUTE = (
    "router_prompt_distribution_layer0_1_input_rmsnorm_serial_reduction_"
    "precision_island_one_token_probe_gate"
)
SELECTED_NEXT_ROUTE = (
    "router_prompt_distribution_layer0_1_input_rmsnorm_serial_reduction_"
    "precision_island_router_distribution_gate"
)
EXPECTED_SERIAL_LAYERS = [0, 1]
EXPECTED_FINAL_NORM_LAYERS = [4, 5, 6, 8, 9, 10]


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


def _layer(smoke: dict[str, Any], table: str, layer: int) -> dict[str, Any]:
  steps = smoke.get(table)
  if not isinstance(steps, list) or not steps or not isinstance(steps[0], dict):
    return {}
  rows = steps[0].get("layers")
  rows = rows if isinstance(rows, list) else []
  return next((
      row for row in rows
      if isinstance(row, dict) and row.get("layer") == layer
  ), {})


def _summary(payload: dict[str, Any]) -> dict[str, Any]:
  smoke = _smoke(payload)
  dist = smoke.get("distribution_ladder")
  dist = dist if isinstance(dist, dict) else {}
  steps = dist.get("steps")
  steps = steps if isinstance(steps, list) else []
  token0 = next((
      row for row in steps
      if isinstance(row, dict) and row.get("token_index") == 0
  ), {})
  preconv0 = _layer(smoke, "linear_preconv_source_diff_by_step", 0)
  preconv1 = _layer(smoke, "linear_preconv_source_diff_by_step", 1)
  boundary0 = _layer(smoke, "layer_boundary_diff_by_step", 0)
  boundary1 = _layer(smoke, "layer_boundary_diff_by_step", 1)
  return {
      "case_id": smoke.get("case_id"),
      "target_returncode": payload.get("target", {}).get("run", {}).get(
          "returncode"),
      "required_checks_passed": payload.get("required_checks_passed"),
      "decode_tokens": smoke.get("decode_tokens_per_session"),
      "serial_layers": smoke.get("input_rmsnorm_serial_reduction_layers"),
      "final_norm_layers": smoke.get("linear_final_cpu_shape_layers"),
      "distribution_required_checks_passed": dist.get(
          "required_checks_passed"),
      "distribution_position_count": dist.get("position_count"),
      "max_kld": token0.get("kld"),
      "top1_rate": dist.get("top1_rate"),
      "top1_pass": dist.get("top1_pass"),
      "layer0_input_max_abs_diff": boundary0.get("input_max_abs_diff"),
      "layer0_output_max_abs_diff": boundary0.get("output_max_abs_diff"),
      "layer1_output_max_abs_diff": boundary1.get("output_max_abs_diff"),
      "layer0_gpu_norm_max_abs_diff": preconv0.get(
          "gpu_attn_norm_vs_cpu_max_abs_diff"),
      "layer0_qkv_from_norm_max_abs_diff": preconv0.get(
          "qkv_from_gpu_attn_norm_max_abs_diff"),
      "layer0_z_from_norm_max_abs_diff": preconv0.get(
          "z_from_gpu_attn_norm_max_abs_diff"),
      "layer1_gpu_norm_max_abs_diff": preconv1.get(
          "gpu_attn_norm_vs_cpu_max_abs_diff"),
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
  baseline = _summary(_load(args.baseline))
  candidate = _summary(_load(args.candidate))
  predecessor_selects = (
      predecessor.get("required_checks_passed") is True
      and predecessor.get("one_token_probe_allowed") is True
      and predecessor.get("selected_next_route") == CURRENT_ROUTE
      and _has_candidate(routes, 548, CURRENT_ROUTE)
      and _has_switch(
          routes, 548,
          "select_router_prompt_distribution_layer0_1_input_rmsnorm_"
          "serial_reduction_precision_island_one_token_probe_gate"))
  bounded_row_passes = (
      candidate["case_id"] == "router_math_reason_001"
      and candidate["target_returncode"] == 0
      and candidate["required_checks_passed"] is True
      and candidate["decode_tokens"] == 1
      and candidate["distribution_position_count"] == 1
      and candidate["serial_layers"] == EXPECTED_SERIAL_LAYERS
      and candidate["final_norm_layers"] == EXPECTED_FINAL_NORM_LAYERS
      and candidate["distribution_required_checks_passed"] is True
      and candidate["top1_pass"] is True
      and _num(candidate["top1_rate"]) == 1.0
      and _num(candidate["max_kld"]) <= KLD_THRESHOLD)
  early_root_closes = (
      _num(candidate["layer0_input_max_abs_diff"]) == 0.0
      and _num(baseline["layer0_gpu_norm_max_abs_diff"]) > 0.0
      and _num(candidate["layer0_gpu_norm_max_abs_diff"]) == 0.0
      and _num(candidate["layer0_qkv_from_norm_max_abs_diff"]) == 0.0
      and _num(candidate["layer0_z_from_norm_max_abs_diff"]) == 0.0
      and _num(candidate["layer1_gpu_norm_max_abs_diff"]) == 0.0)
  kld_materially_improves = (
      _num(baseline["max_kld"]) > KLD_THRESHOLD
      and _num(candidate["max_kld"]) <= KLD_THRESHOLD
      and _num(candidate["max_kld"]) <= 0.1 * _num(baseline["max_kld"]))
  checks = [
      {"name": "seq548_selected_one_token_probe", "pass": predecessor_selects},
      {"name": "bounded_candidate_row_passes", "pass": bounded_row_passes},
      {"name": "layer0_norm_qkv_z_root_closes", "pass": early_root_closes},
      {"name": "token0_kld_materially_clears_ruler",
       "pass": kld_materially_improves},
  ]
  required = all(bool(check["pass"]) for check in checks)
  baseline_kld = _num(baseline["max_kld"])
  candidate_kld = _num(candidate["max_kld"])
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
          "kld_ratio_candidate_over_baseline": (
              candidate_kld / baseline_kld if baseline_kld > 0.0 else 0.0),
          "kld_relative_change": (
              candidate_kld / baseline_kld - 1.0
              if baseline_kld > 0.0 else 0.0),
      },
      "eight_token_router_distribution_allowed": required,
      "speedup_claims_allowed": False,
      "promotion_allowed": False,
      "disposition": (
          "accept_serial_reduction_one_token_select_router_distribution"
          if required else
          "block_serial_reduction_one_token_inconsistent_evidence"),
      "selected_next_route": SELECTED_NEXT_ROUTE if required else CURRENT_ROUTE,
      "next_route_reason": (
          "The one-token candidate clears KLD, preserves top-1, and closes the "
          "exact-input layer0 norm/QKV/Z source. Run paired 8-token router "
          "math/code distribution next; this is not promotion or speed evidence."
          if required else
          "The bounded evidence does not close the early source; keep the one-token "
          "gate open."
      ),
  }


def write_outputs(metrics: dict[str, Any], out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  failed = [row["name"] for row in metrics["checks"] if not row["pass"]]
  lines = [
      f"# Seq{metrics['sequence']} Input-RMSNorm Serial One-Token Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- baseline token-0 KLD: `{metrics['baseline']['max_kld']}`",
      f"- candidate token-0 KLD: `{metrics['candidate']['max_kld']}`",
      f"- eight_token_router_distribution_allowed: `{str(metrics['eight_token_router_distribution_allowed']).lower()}`",
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
  parser.add_argument("--sequence", type=int, default=549)
  parser.add_argument(
      "--predecessor", type=Path,
      default=ROOT / "output/seq548-layer0-1-input-rmsnorm-serial-reduction-target-compile-gate-20260710Tseq548Z/metrics.json")
  parser.add_argument("--routes", type=Path,
                      default=ACTIVE / "routes-ledger.json")
  parser.add_argument(
      "--baseline", type=Path,
      default=ROOT / "output/seq541-layer0-exact-delta-source-math-20260710Tseq541Z/result.json")
  parser.add_argument(
      "--candidate", type=Path,
      default=ROOT / "output/seq549-layer0-1-input-rmsnorm-serial-reduction-one-token-20260710Tseq549Z/result.json")
  parser.add_argument(
      "--out-dir", type=Path,
      default=ROOT / "output/seq549-layer0-1-input-rmsnorm-serial-reduction-one-token-gate-20260710Tseq549Z")
  args = parser.parse_args()
  metrics = compute(args)
  write_outputs(metrics, args.out_dir)
  print(json.dumps({
      "required_checks_passed": metrics["required_checks_passed"],
      "disposition": metrics["disposition"],
      "eight_token_router_distribution_allowed": metrics[
          "eight_token_router_distribution_allowed"],
      "selected_next_route": metrics["selected_next_route"],
      "out_dir": _rel(args.out_dir),
  }, sort_keys=True))
  return 0 if metrics["required_checks_passed"] else 2


if __name__ == "__main__":
  raise SystemExit(main())
