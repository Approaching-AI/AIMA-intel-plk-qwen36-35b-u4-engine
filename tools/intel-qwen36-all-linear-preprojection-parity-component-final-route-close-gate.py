#!/usr/bin/env python3
"""Close the exhausted exact-preprojection bundle and select route reflection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = (
    "intel-qwen36-all-linear-preprojection-parity-component-final-route-close-v0"
)
CURRENT_ROUTE = (
    "router_prompt_distribution_all_linear_preprojection_parity_component_final_route_close_gate"
)
SELECTED_NEXT_ROUTE = "gpu_backend_runtime_route_reflection_gate"
CLOSED_ROUTE = (
    "router_prompt_distribution_cpuorder_preprojection_bundle_v1"
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
  routes = _load(args.routes)
  predecessor = _load(args.predecessor)
  first_close = _load(args.first_close)
  rows = predecessor.get("rows", [])
  rows = rows if isinstance(rows, list) else []
  route_selects = (
      predecessor.get("measurement_complete") is True
      and predecessor.get("component_passed") is False
      and predecessor.get("decode_source_allowed") is False
      and predecessor.get("selected_next_route") == CURRENT_ROUTE
      and len(rows) == 2
      and _has_candidate(routes, 604, CURRENT_ROUTE)
      and _has_switch(
          routes, 604,
          "select_router_prompt_distribution_all_linear_preprojection_"
          "parity_component_final_route_close_gate"))
  repaired_boundaries_exact = all(
      row.get("exact_comparisons", {}).get(name) is True
      for row in rows
      for name in (
          "exact_qkv_vs_cpu",
          "exact_conv_output_vs_cpu",
          "exact_conv_state_vs_cpu",
          "exact_projection_vs_cpu",
      ))
  remaining_boundaries_fail = all(
      row.get("exact_comparisons", {}).get(name) is False
      for row in rows
      for name in (
          "exact_attention_vs_cpu",
          "exact_final_vs_cpu",
          "exact_recurrent_state_vs_cpu",
      ))
  budget_passes = all(row.get("budget_passed") is True for row in rows)
  repair_exhausted = (
      first_close.get("repairable_contract_violation") is True
      and first_close.get("repair_contract", {}).get("stop_condition")
      == (
          "If a fresh binary still fails any exact boundary, close the whole "
          "bundle; do not open another arithmetic-axis sweep."))
  already_rejected = any(
      isinstance(row, dict) and row.get("route") == CLOSED_ROUTE
      for row in _load(args.rejected).get("rejected", []))
  no_runtime_evidence = not any(
      (args.out_dir / name).exists()
      for name in ("run.json", "probe.json", "tokens.jsonl"))
  checks = [
      {"name": "seq604_selected_terminal_no_target_close",
       "pass": route_selects},
      {"name": "q6_conv_projection_repair_is_exact_and_budget_passes",
       "pass": repaired_boundaries_exact and budget_passes},
      {"name": "postconv_recurrent_still_fails_both_terminal_rows",
       "pass": remaining_boundaries_fail},
      {"name": "recorded_one_repair_stop_condition_is_satisfied",
       "pass": repair_exhausted},
      {"name": "closed_route_is_not_already_in_rejected_ledger",
       "pass": not already_rejected},
      {"name": "final_close_gate_created_no_runtime_evidence",
       "pass": no_runtime_evidence},
  ]
  required = all(bool(row["pass"]) for row in checks)
  closure = {
      "route": CLOSED_ROUTE,
      "class": "correctness_route_closed",
      "reason": (
          "The locked whole bundle clears timing and makes Q6 QKV, convolution "
          "output/state, and fused projection bit-exact after one source "
          "repair, but both terminal rows retain nonzero postconv/recurrent "
          "deltas. The recorded one-repair stop condition is exhausted."),
      "evidence": _rel(args.predecessor.parent),
      "context_evidence": (
          f"{_rel(args.first_close.parent)}, "
          "output/seq602-all-linear-preprojection-parity-component-source-"
          "repair-gate-20260710Tseq602Z/"),
      "reopen_condition": (
          "an independently justified non-OpenCL or correctly-rounded "
          "postconv/recurrent primitive with a new component contract and "
          "floor kill-number; do not retry OpenCL arithmetic/order variants"),
  }
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "sequence": args.sequence,
      "inputs": {
          "routes": _rel(args.routes),
          "rejected": _rel(args.rejected),
          "predecessor": _rel(args.predecessor),
          "first_close": _rel(args.first_close),
      },
      "closure": closure,
      "checks": checks,
      "required_checks_passed": required,
      "bundle_closed": required,
      "runtime_route_reflection_allowed": required,
      "source_repair_allowed": False,
      "target_compile_allowed": False,
      "component_probe_allowed": False,
      "decode_integration_allowed": False,
      "token_row_allowed": False,
      "validation_or_test_allowed": False,
      "speedup_claims_allowed": False,
      "promotion_allowed": False,
      "disposition": (
          "close_cpuorder_preprojection_bundle_v1_select_runtime_reflection"
          if required else "repair_final_preprojection_route_close"),
      "selected_next_route": (
          SELECTED_NEXT_ROUTE if required else CURRENT_ROUTE),
      "next_route_reason": (
          "The OpenCL exact-preprojection arithmetic board is closed. Audit "
          "materially different runtime/backend routes against repo evidence "
          "and the native-dependency constraint before any new source or "
          "target row."
          if required else
          "Repair terminal evidence or rejected-ledger consistency before "
          "selecting another route."),
  }


def write_outputs(metrics: dict[str, Any], out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  (out_dir / "manifest.json").write_text(
      json.dumps({
          "schema_version": metrics["schema_version"],
          "workstream": metrics["workstream"],
          "tool": _rel(Path(__file__)),
          "inputs": metrics["inputs"],
          "bundle_closed": metrics["bundle_closed"],
          "selected_next_route": metrics["selected_next_route"],
          "source_repair_allowed": False,
          "target_compile_allowed": False,
          "token_row_allowed": False,
      }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  failed = [row["name"] for row in metrics["checks"] if not row["pass"]]
  lines = [
      f"# Seq{metrics['sequence']} Exact Preprojection Final Close",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- bundle_closed: `{str(metrics['bundle_closed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- failed_checks: `{failed}`",
      "",
      metrics["next_route_reason"],
      "",
      "This gate used existing evidence only; no target command ran.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--sequence", type=int, default=605)
  parser.add_argument("--routes", type=Path,
                      default=ACTIVE / "routes-ledger.json")
  parser.add_argument("--rejected", type=Path,
                      default=ACTIVE / "rejected-routes.json")
  parser.add_argument(
      "--predecessor", type=Path,
      default=ROOT / (
          "output/seq604-all-linear-preprojection-parity-component-final-"
          "probe-gate-20260710Tseq604Z/metrics.json"))
  parser.add_argument(
      "--first-close", type=Path,
      default=ROOT / (
          "output/seq601-all-linear-preprojection-parity-component-route-"
          "close-gate-20260710Tseq601Z/metrics.json"))
  parser.add_argument(
      "--out-dir", type=Path,
      default=ROOT / (
          "output/seq605-all-linear-preprojection-parity-component-final-"
          "route-close-gate-20260710Tseq605Z"))
  args = parser.parse_args()
  metrics = compute(args)
  write_outputs(metrics, args.out_dir)
  print(json.dumps({
      "required_checks_passed": metrics["required_checks_passed"],
      "bundle_closed": metrics["bundle_closed"],
      "disposition": metrics["disposition"],
      "selected_next_route": metrics["selected_next_route"],
      "out_dir": _rel(args.out_dir),
  }, sort_keys=True))
  return 0 if metrics["required_checks_passed"] else 2


if __name__ == "__main__":
  raise SystemExit(main())
