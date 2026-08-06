#!/usr/bin/env python3
"""Close the local parity bundle and select broad route feasibility."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = "intel-qwen36-layer0-parity-bundle-distribution-v0"
CURRENT_ROUTE = (
    "router_prompt_distribution_layer0_norm_to_projection_parity_bundle_"
    "router_math_distribution_gate"
)
SELECTED_NEXT_ROUTE = (
    "router_prompt_distribution_all_linear_norm_to_projection_parity_"
    "feasibility_gate"
)
REJECTED_ROUTE = (
    "router_prompt_distribution_layer0_norm_to_projection_parity_bundle"
)
KLD_THRESHOLD = 0.005
SUB_ULP_BOUND = 1.192092896e-7


def _load(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise TypeError(f"{path} does not contain a JSON object")
  return value


def _rel(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path)


def _num(value: Any, default: float = 0.0) -> float:
  return float(value) if isinstance(value, (int, float)) else default


def _smoke(payload: dict[str, Any]) -> dict[str, Any]:
  return payload.get("smoke") if isinstance(payload.get("smoke"), dict) else payload


def _has_candidate(routes: dict[str, Any], seq: int, route: str) -> bool:
  return any(isinstance(row, dict) and row.get("seq") == seq
             and row.get("selected_next_route") == route
             for row in routes.get("candidate_history", []))


def _has_switch(routes: dict[str, Any], seq: int, decision: str) -> bool:
  return any(isinstance(row, dict) and row.get("seq_covered") == seq
             and row.get("decision") == decision and row.get("resolved") is True
             for row in routes.get("switch_decisions", []))


def _boundary_rows(smoke: dict[str, Any]) -> list[dict[str, Any]]:
  output = []
  for step in smoke.get("layer_boundary_diff_by_step", []):
    if not isinstance(step, dict):
      continue
    by_layer = {row.get("layer"): row for row in step.get("layers", [])
                if isinstance(row, dict)}
    output.append({
        "token_index": step.get("token_index"),
        "layer0": by_layer.get(0, {}).get("output_max_abs_diff"),
        "layer1": by_layer.get(1, {}).get("output_max_abs_diff"),
    })
  return output


def compute(args: argparse.Namespace) -> dict[str, Any]:
  predecessor = _load(args.predecessor)
  routes = _load(args.routes)
  baseline = _smoke(_load(args.baseline))
  candidate_payload = _load(args.candidate)
  candidate = _smoke(candidate_payload)
  baseline_dist = baseline.get("distribution_ladder", {})
  candidate_dist = candidate.get("distribution_ladder", {})
  boundaries = _boundary_rows(candidate)
  token7 = next((row for row in boundaries if row["token_index"] == 7), {})
  checks = [
      {"name": "seq561_selected_bundle_distribution",
       "pass": (
           predecessor.get("required_checks_passed") is True
           and predecessor.get("router_math_distribution_allowed") is True
           and predecessor.get("selected_next_route") == CURRENT_ROUTE
           and _has_candidate(routes, 561, CURRENT_ROUTE)
           and _has_switch(
               routes, 561,
               "select_router_prompt_distribution_layer0_norm_to_projection_"
               "parity_bundle_router_math_distribution_gate"))},
      {"name": "bounded_eight_token_bundle_row_complete",
       "pass": (
           candidate_payload.get("target", {}).get("run", {}).get("returncode") == 2
           and candidate.get("decode_tokens_per_session") == 8
           and candidate.get("input_rmsnorm_serial_reduction_layers") == [0]
           and candidate.get("linear_output_projection_cpu_order_layers") == [0]
           and candidate_dist.get("position_count") == 8
           and candidate_dist.get("required_checks_passed") is False)},
      {"name": "bundle_fails_and_regresses_vs_baseline",
       "pass": (
           _num(candidate_dist.get("max_kld")) > KLD_THRESHOLD
           and _num(candidate_dist.get("max_kld"))
           > _num(baseline_dist.get("max_kld")))},
      {"name": "top1_remains_exact_but_cannot_override_kld",
       "pass": (_num(candidate_dist.get("top1_rate")) == 1.0
                and candidate_dist.get("top1_pass") is True)},
      {"name": "token7_layer1_is_already_sub_ulp",
       "pass": (0.0 < _num(token7.get("layer1")) <= SUB_ULP_BOUND)},
  ]
  required = all(bool(row["pass"]) for row in checks)
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "sequence": args.sequence,
      "inputs": {"predecessor": _rel(args.predecessor),
                 "routes": _rel(args.routes),
                 "baseline": _rel(args.baseline),
                 "candidate": _rel(args.candidate)},
      "checks": checks,
      "required_checks_passed": required,
      "baseline_max_kld": baseline_dist.get("max_kld"),
      "candidate_max_kld": candidate_dist.get("max_kld"),
      "candidate_top1_rate": candidate_dist.get("top1_rate"),
      "token7_layer1_output_max_abs_diff": token7.get("layer1"),
      "incremental_layer_extension_allowed": False,
      "broad_parity_feasibility_allowed": required,
      "router_code_distribution_allowed": False,
      "speedup_claims_allowed": False,
      "promotion_allowed": False,
      "disposition": (
          "reject_local_bundle_select_all_linear_parity_feasibility"
          if required else "block_local_bundle_inconsistent_evidence"),
      "rejected_route": REJECTED_ROUTE if required else None,
      "selected_next_route": SELECTED_NEXT_ROUTE if required else CURRENT_ROUTE,
      "next_route_reason": (
          "The local bundle regresses KLD, and token7 already leaves layer1 at "
          "sub-ULP drift, so another layer-by-layer extension would repeat the "
          "closed recursive chase. Evaluate one all-linear norm-to-projection "
          "parity route in source/evidence-only form; do not sweep subsets."
          if required else "The bundle evidence is incomplete; keep the gate open."),
  }


def write_outputs(metrics: dict[str, Any], out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  failed = [row["name"] for row in metrics["checks"] if not row["pass"]]
  lines = [
      f"# Seq{metrics['sequence']} Layer0 Parity Bundle Distribution Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- rejected_route: `{metrics['rejected_route']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- baseline -> candidate max KLD: `{metrics['baseline_max_kld']}` -> `{metrics['candidate_max_kld']}`",
      f"- token7 layer1 output max abs: `{metrics['token7_layer1_output_max_abs_diff']}`",
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
  parser.add_argument("--sequence", type=int, default=562)
  parser.add_argument("--predecessor", type=Path,
      default=ROOT / "output/seq561-layer0-norm-to-projection-parity-bundle-one-token-gate-20260710Tseq561Z/metrics.json")
  parser.add_argument("--routes", type=Path, default=ACTIVE / "routes-ledger.json")
  parser.add_argument("--baseline", type=Path,
      default=ROOT / "output/seq541-layer0-exact-delta-source-math-20260710Tseq541Z/result.json")
  parser.add_argument("--candidate", type=Path,
      default=ROOT / "output/seq562-layer0-norm-to-projection-parity-bundle-router-math-20260710Tseq562Z/result.json")
  parser.add_argument("--out-dir", type=Path,
      default=ROOT / "output/seq562-layer0-norm-to-projection-parity-bundle-router-math-gate-20260710Tseq562Z")
  args = parser.parse_args()
  metrics = compute(args)
  write_outputs(metrics, args.out_dir)
  print(json.dumps({"required_checks_passed": metrics["required_checks_passed"],
                    "disposition": metrics["disposition"],
                    "broad_parity_feasibility_allowed": metrics["broad_parity_feasibility_allowed"],
                    "selected_next_route": metrics["selected_next_route"],
                    "out_dir": _rel(args.out_dir)}, sort_keys=True))
  return 0 if metrics["required_checks_passed"] else 2


if __name__ == "__main__":
  raise SystemExit(main())
