#!/usr/bin/env python3
"""Close the layer0/1 serial input-RMSNorm subset attribution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = "intel-qwen36-input-rmsnorm-serial-layer-subset-gate-v0"
KLD_THRESHOLD = 0.005
CURRENT_ROUTE = (
    "router_prompt_distribution_input_rmsnorm_serial_layer_subset_"
    "attribution_gate"
)
SELECTED_NEXT_ROUTE = (
    "router_prompt_distribution_early_precision_island_route_reflection_gate"
)
REJECTED_ROUTE = (
    "router_prompt_distribution_input_rmsnorm_serial_layer_subsets"
)


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
  serial_layers = smoke.get("input_rmsnorm_serial_reduction_layers")
  serial_layers = serial_layers if isinstance(serial_layers, list) else []
  return {
      "case_id": smoke.get("case_id"),
      "target_returncode": payload.get("target", {}).get("run", {}).get(
          "returncode"),
      "decode_tokens": smoke.get("decode_tokens_per_session"),
      "serial_layers": serial_layers,
      "final_norm_layers": smoke.get("linear_final_cpu_shape_layers"),
      "distribution_required_checks_passed": dist.get(
          "required_checks_passed"),
      "position_count": dist.get("position_count"),
      "max_kld": dist.get("max_kld"),
      "top1_rate": dist.get("top1_rate"),
      "top1_pass": dist.get("top1_pass"),
      "token_klds": [row.get("kld") for row in steps
                      if isinstance(row, dict)],
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


def _valid_row(row: dict[str, Any], expected_layers: list[int]) -> bool:
  return (
      row["case_id"] == "router_math_reason_001"
      and row["target_returncode"] == 2
      and row["decode_tokens"] == 8
      and row["position_count"] == 8
      and row["serial_layers"] == expected_layers
      and row["distribution_required_checks_passed"] is False
      and len(row["token_klds"]) == 8
  )


def compute(args: argparse.Namespace) -> dict[str, Any]:
  predecessor = _load(args.predecessor)
  routes = _load(args.routes)
  cells = {
      "empty": _summary(_load(args.baseline)),
      "layer0": _summary(_load(args.layer0)),
      "layer1": _summary(_load(args.layer1)),
      "layer0_1": _summary(_load(args.layer0_1)),
  }
  predecessor_selects = (
      predecessor.get("required_checks_passed") is True
      and predecessor.get("layer_subset_attribution_allowed") is True
      and predecessor.get("selected_next_route") == CURRENT_ROUTE
      and _has_candidate(routes, 550, CURRENT_ROUTE)
      and _has_switch(
          routes, 550,
          "select_router_prompt_distribution_input_rmsnorm_serial_"
          "layer_subset_attribution_gate"))
  rows_complete = (
      _valid_row(cells["empty"], [])
      and _valid_row(cells["layer0"], [0])
      and _valid_row(cells["layer1"], [1])
      and _valid_row(cells["layer0_1"], [0, 1])
  )
  singleton_top1_regresses = all(
      cells[name]["top1_pass"] is False
      and _num(cells[name]["top1_rate"]) == 0.875
      for name in ("layer0", "layer1"))
  singleton_kld_regresses = all(
      _num(cells[name]["max_kld"]) > _num(cells["empty"]["max_kld"])
      for name in ("layer0", "layer1"))
  combined_interaction_insufficient = (
      cells["layer0_1"]["top1_pass"] is True
      and _num(cells["layer0_1"]["top1_rate"]) == 1.0
      and _num(cells["layer0_1"]["max_kld"]) > KLD_THRESHOLD
      and _num(cells["layer0_1"]["max_kld"])
      < min(_num(cells["layer0"]["max_kld"]),
            _num(cells["layer1"]["max_kld"]))
  )
  every_nonempty_subset_fails = all(
      cells[name]["distribution_required_checks_passed"] is False
      and _num(cells[name]["max_kld"]) > KLD_THRESHOLD
      for name in ("layer0", "layer1", "layer0_1"))
  checks = [
      {"name": "seq550_selected_layer_subset_attribution",
       "pass": predecessor_selects},
      {"name": "four_cell_router_math_evidence_complete",
       "pass": rows_complete},
      {"name": "both_singletons_regress_token7_top1",
       "pass": singleton_top1_regresses},
      {"name": "both_singletons_regress_max_kld_vs_empty",
       "pass": singleton_kld_regresses},
      {"name": "combined_interaction_is_compensatory_but_insufficient",
       "pass": combined_interaction_insufficient},
      {"name": "every_nonempty_serial_subset_fails_distribution",
       "pass": every_nonempty_subset_fails},
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
          "layer0": _rel(args.layer0),
          "layer1": _rel(args.layer1),
          "layer0_1": _rel(args.layer0_1),
      },
      "checks": checks,
      "required_checks_passed": required,
      "cells": cells,
      "all_serial_layer_subsets_rejected": required,
      "new_target_row_allowed": False,
      "speedup_claims_allowed": False,
      "promotion_allowed": False,
      "disposition": (
          "reject_all_serial_layer_subsets_select_route_reflection"
          if required else
          "block_serial_layer_subset_inconsistent_evidence"),
      "rejected_route": REJECTED_ROUTE if required else None,
      "selected_next_route": SELECTED_NEXT_ROUTE if required else CURRENT_ROUTE,
      "next_route_reason": (
          "Both singleton islands lose token-7 top-1 and regress max KLD. "
          "The combined island restores top-1 through a compensatory interaction "
          "but still fails KLD. The empty, layer0, layer1, and layer0/1 cells "
          "therefore close the complete layer-subset axis. Reflect against the "
          "closed precision and source routes before authorizing another target row."
          if required else
          "The four-cell subset evidence is incomplete; keep attribution open."
      ),
  }


def write_outputs(metrics: dict[str, Any], out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  failed = [row["name"] for row in metrics["checks"] if not row["pass"]]
  cells = metrics["cells"]
  lines = [
      f"# Seq{metrics['sequence']} Input-RMSNorm Serial Layer-Subset Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- rejected_route: `{metrics['rejected_route']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- empty/layer0/layer1/layer0_1 max KLD: "
      f"`{cells['empty']['max_kld']}` / `{cells['layer0']['max_kld']}` / "
      f"`{cells['layer1']['max_kld']}` / `{cells['layer0_1']['max_kld']}`",
      f"- layer0/layer1/layer0_1 top-1 rate: "
      f"`{cells['layer0']['top1_rate']}` / `{cells['layer1']['top1_rate']}` / "
      f"`{cells['layer0_1']['top1_rate']}`",
      f"- failed_checks: `{failed}`",
      "",
      metrics["next_route_reason"],
      "",
      "This is bounded correctness attribution only. It is not a speed claim.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--sequence", type=int, default=551)
  parser.add_argument(
      "--predecessor", type=Path,
      default=ROOT / "output/seq550-layer0-1-input-rmsnorm-serial-reduction-router-distribution-gate-20260710Tseq550Z/metrics.json")
  parser.add_argument("--routes", type=Path,
                      default=ACTIVE / "routes-ledger.json")
  parser.add_argument(
      "--baseline", type=Path,
      default=ROOT / "output/seq541-layer0-exact-delta-source-math-20260710Tseq541Z/result.json")
  parser.add_argument(
      "--layer0", type=Path,
      default=ROOT / "output/seq551-input-rmsnorm-serial-layer0-router-math-20260710Tseq551Z/result.json")
  parser.add_argument(
      "--layer1", type=Path,
      default=ROOT / "output/seq551-input-rmsnorm-serial-layer1-router-math-20260710Tseq551Z/result.json")
  parser.add_argument(
      "--layer0-1", type=Path,
      default=ROOT / "output/seq550-layer0-1-input-rmsnorm-serial-reduction-router-math-20260710Tseq550Z/result.json")
  parser.add_argument(
      "--out-dir", type=Path,
      default=ROOT / "output/seq551-input-rmsnorm-serial-layer-subset-attribution-gate-20260710Tseq551Z")
  args = parser.parse_args()
  metrics = compute(args)
  write_outputs(metrics, args.out_dir)
  print(json.dumps({
      "required_checks_passed": metrics["required_checks_passed"],
      "disposition": metrics["disposition"],
      "all_serial_layer_subsets_rejected": metrics[
          "all_serial_layer_subsets_rejected"],
      "selected_next_route": metrics["selected_next_route"],
      "out_dir": _rel(args.out_dir),
  }, sort_keys=True))
  return 0 if metrics["required_checks_passed"] else 2


if __name__ == "__main__":
  raise SystemExit(main())
