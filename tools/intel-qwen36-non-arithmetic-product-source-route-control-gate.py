#!/usr/bin/env python3
"""Select a bounded, held-out non-arithmetic correction surface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = "intel-qwen36-non-arithmetic-product-source-route-control-v0"
CURRENT_ROUTE = (
    "router_prompt_distribution_non_arithmetic_product_source_route_control_gate"
)
SELECTED_NEXT_ROUTE = (
    "router_prompt_distribution_sparse_head_logit_bias_source_gate"
)
GLOBAL_AFFINE_ROUTE = "router_prompt_distribution_affine_logit_calibration"
CALIBRATION_CASES = ("router_math_reason_001", "router_code_reason_002")
HOLDOUT_CASES = (
    "router_instruction_003",
    "short_math_001",
    "short_factual_002",
    "short_transform_003",
)


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


def _has_candidate(routes: dict[str, Any], seq: int, route: str) -> bool:
  return any(isinstance(row, dict) and row.get("seq") == seq
             and row.get("selected_next_route") == route
             for row in routes.get("candidate_history", []))


def _has_switch(routes: dict[str, Any], seq: int, decision: str) -> bool:
  return any(isinstance(row, dict) and row.get("seq_covered") == seq
             and row.get("decision") == decision and row.get("resolved") is True
             for row in routes.get("switch_decisions", []))


def _rejected(payload: dict[str, Any], route: str) -> dict[str, Any]:
  return next((row for row in payload.get("rejected", [])
               if isinstance(row, dict) and row.get("route") == route), {})


def _calibration(rows: dict[str, Any]) -> dict[str, Any]:
  observations: dict[int, list[dict[str, Any]]] = {}
  for label in ("math", "code"):
    row = rows.get(label)
    for step in row.get("failed_steps", []) if isinstance(row, dict) else []:
      for role in ("top_negative", "top_positive"):
        head = step.get(role)
        if not isinstance(head, dict):
          continue
        token_id = head.get("token_id")
        if not isinstance(token_id, int):
          continue
        delta = float(head["native_logit"]) - float(head["gpu_logit"])
        observations.setdefault(token_id, []).append({
            "case": row.get("case_id"),
            "token_index": step.get("token_index"),
            "role": role,
            "native_minus_gpu_logit": delta,
        })
  biases = []
  for token_id, values in sorted(observations.items()):
    deltas = [float(row["native_minus_gpu_logit"]) for row in values]
    biases.append({
        "token_id": token_id,
        "observation_count": len(values),
        "all_signs_consistent": all(value > 0 for value in deltas)
        or all(value < 0 for value in deltas),
        "mean_bias": sum(deltas) / len(deltas),
        "min_bias": min(deltas),
        "max_bias": max(deltas),
        "observations": values,
    })
  return {"biases": biases, "token_count": len(biases)}


def compute(args: argparse.Namespace) -> dict[str, Any]:
  predecessor = _load(args.predecessor)
  routes = _load(args.routes)
  rejected = _load(args.rejected)
  attribution = _load(args.attribution)
  source = args.decode_source.read_text(encoding="utf-8")
  rows = attribution.get("rows", {})
  calibration = _calibration(rows)
  failed_steps = []
  for label in ("math", "code"):
    row = rows.get(label, {})
    failed_steps.extend(row.get("failed_steps", []))
  repeated = [row for row in calibration["biases"]
              if row["observation_count"] > 1]
  global_affine = _rejected(rejected, GLOBAL_AFFINE_ROUTE)
  checks = [
      {"name": "seq563_selected_non_arithmetic_route_control",
       "pass": (
           predecessor.get("required_checks_passed") is True
           and predecessor.get("all_linear_token_allowed") is False
           and predecessor.get("selected_next_route") == CURRENT_ROUTE
           and _has_candidate(routes, 563, CURRENT_ROUTE)
           and _has_switch(
               routes, 563,
               "select_router_prompt_distribution_non_arithmetic_product_"
               "source_route_control_gate"))},
      {"name": "global_affine_calibration_is_closed",
       "pass": bool(global_affine)},
      {"name": "failed_kld_is_concentrated_in_head_pairs",
       "pass": (
           len(failed_steps) == 5
           and all(float(row.get("head_native_prob_mass", 0.0)) >= 0.9
                   and float(row.get("top_negative_coverage_ratio", 0.0)) >= 0.95
                   and float(row.get("top_positive_coverage_ratio", 0.0)) >= 0.85
                   and row.get("top1_matches") is True
                   for row in failed_steps))},
      {"name": "sparse_bias_union_is_bounded",
       "pass": 0 < calibration["token_count"] <= 8},
      {"name": "repeated_token_bias_signs_are_cross_prompt_consistent",
       "pass": len(repeated) >= 3
       and all(row["all_signs_consistent"] for row in repeated)},
      {"name": "full_logits_already_exist_on_host_distribution_path",
       "pass": (
           "WriteDecodeDistributionLadder" in source
           and "distribution_ladder_steps" in source
           and "gpu_logits" in source)},
  ]
  required = all(bool(row["pass"]) for row in checks)
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "sequence": args.sequence,
      "inputs": {"predecessor": _rel(args.predecessor),
                 "routes": _rel(args.routes),
                 "rejected": _rel(args.rejected),
                 "attribution": _rel(args.attribution),
                 "decode_source": _rel(args.decode_source)},
      "checks": checks,
      "required_checks_passed": required,
      "calibration": calibration,
      "contract": {
          "calibration_cases": list(CALIBRATION_CASES),
          "holdout_cases": list(HOLDOUT_CASES),
          "prompt_case_or_position_branching_allowed": False,
          "native_oracle_at_runtime_allowed": False,
          "token_id_sparse_static_bias_only": True,
          "calibration_top1_changes_allowed": False,
          "holdout_top1_changes_allowed": False,
          "holdout_kld_regression_allowed": False,
          "speed_or_promotion_before_holdout_allowed": False,
      },
      "source_gate_allowed": required,
      "target_row_allowed": False,
      "speedup_claims_allowed": False,
      "promotion_allowed": False,
      "disposition": (
          "select_sparse_head_logit_bias_source_with_holdout_contract"
          if required else "block_non_arithmetic_route_no_bounded_surface"),
      "selected_next_route": SELECTED_NEXT_ROUTE if required else CURRENT_ROUTE,
      "next_route_reason": (
          "Five failed steps preserve top-1 and concentrate KLD in at most seven "
          "head tokens; repeated token corrections have consistent signs across "
          "math/code. A sparse static token-ID bias is distinct from the closed "
          "global affine route and is speed-sized. Add only a source/generate-only "
          "diagnostic under a strict calibration/holdout contract; no prompt or "
          "position branching and no target row yet."
          if required else
          "No bounded non-arithmetic correction surface satisfies the evidence."
      ),
  }


def write_outputs(metrics: dict[str, Any], out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  failed = [row["name"] for row in metrics["checks"] if not row["pass"]]
  lines = [
      f"# Seq{metrics['sequence']} Non-Arithmetic Product-Source Route Control",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- sparse token count: `{metrics['calibration']['token_count']}`",
      f"- calibration cases: `{metrics['contract']['calibration_cases']}`",
      f"- holdout cases: `{metrics['contract']['holdout_cases']}`",
      f"- failed_checks: `{failed}`",
      "",
      metrics["next_route_reason"],
      "",
      "This is route-control evidence only. It is not calibration or speed evidence.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--sequence", type=int, default=564)
  parser.add_argument("--predecessor", type=Path,
      default=ROOT / "output/seq563-all-linear-norm-to-projection-parity-feasibility-gate-20260710Tseq563Z/metrics.json")
  parser.add_argument("--routes", type=Path, default=ACTIVE / "routes-ledger.json")
  parser.add_argument("--rejected", type=Path, default=ACTIVE / "rejected-routes.json")
  parser.add_argument("--attribution", type=Path,
      default=ROOT / "output/seq524-top-kld-contributor-attribution-gate-20260709Tseq524Z/metrics.json")
  parser.add_argument("--decode-source", type=Path,
      default=ROOT / "tools/intel-qwen36-r2-gpu-decode-smoke.py")
  parser.add_argument("--out-dir", type=Path,
      default=ROOT / "output/seq564-non-arithmetic-product-source-route-control-gate-20260710Tseq564Z")
  args = parser.parse_args()
  metrics = compute(args)
  write_outputs(metrics, args.out_dir)
  print(json.dumps({"required_checks_passed": metrics["required_checks_passed"],
                    "disposition": metrics["disposition"],
                    "source_gate_allowed": metrics["source_gate_allowed"],
                    "selected_next_route": metrics["selected_next_route"],
                    "out_dir": _rel(args.out_dir)}, sort_keys=True))
  return 0 if metrics["required_checks_passed"] else 2


if __name__ == "__main__":
  raise SystemExit(main())
