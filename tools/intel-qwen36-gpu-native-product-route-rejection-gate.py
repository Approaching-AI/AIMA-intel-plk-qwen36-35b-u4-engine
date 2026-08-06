#!/usr/bin/env python3
"""Record the current contract's explicit native product route rejection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = "intel-qwen36-gpu-native-product-route-rejection-v0"
CURRENT_ROUTE = "gpu_native_product_route_rejection_gate"
CLOSED_ROUTE = "gpu_arc_b390_rowblock16_26mask_product_promotion_v1"
TERMINAL_ROUTE = "none_explicit_route_rejection_recorded"


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


def _candidate(routes: dict[str, Any], seq: int) -> dict[str, Any]:
  for row in routes.get("candidate_history", []):
    if isinstance(row, dict) and row.get("seq") == seq:
      return row
  return {}


def _switch_selects(routes: dict[str, Any], seq: int,
                    decision: str) -> bool:
  return any(
      isinstance(row, dict)
      and row.get("seq_covered") == seq
      and row.get("decision") == decision
      and row.get("resolved") is True
      for row in routes.get("switch_decisions", []))


def compute(args: argparse.Namespace) -> dict[str, Any]:
  routes = _load(args.routes)
  rejected = _load(args.rejected)
  predecessor = _load(args.predecessor)
  goal_text = args.goal.read_text(encoding="utf-8")
  short_candidate = _candidate(routes, 221)
  direct_promotion = _candidate(routes, 222)

  predecessor_selects = (
      predecessor.get("required_checks_passed") is True
      and predecessor.get("route_rejection_allowed") is True
      and predecessor.get("current_contract_has_admissible_correctness_successor")
      is False
      and predecessor.get("selected_next_route") == CURRENT_ROUTE
      and _candidate(routes, 625).get("selected_next_route") == CURRENT_ROUTE
      and _switch_selects(
          routes, 625, "select_gpu_native_product_route_rejection_gate"))
  goal_allows_rejection = (
      "proof that both prefill and decode clear the accepted target, or an explicit"
      in goal_text
      and "route rejection" in goal_text)
  short_evidence_is_diagnostic_only = (
      short_candidate.get("disposition")
      == "accept_rowblock16_26mask_short_floor_clear_candidate"
      and float(short_candidate.get("decode_tok_s", 0.0)) > 19.5
      and float(short_candidate.get("confirm_tok_s", 0.0)) > 19.5
      and float(short_candidate.get("distribution_max_kld", 1.0)) <= 0.005
      and float(short_candidate.get("distribution_top1_rate", 0.0)) >= 0.99
      and direct_promotion.get("disposition")
      == "reject_direct_26mask_promotion_on_router_distribution")
  existing = next((
      row for row in rejected.get("rejected", [])
      if isinstance(row, dict) and row.get("route") == CLOSED_ROUTE), None)
  existing_is_consistent = (
      existing is None
      or (
          existing.get("class") == "explicit_product_route_rejection"
          and existing.get("evidence", "").rstrip("/")
          == _rel(args.predecessor.parent).rstrip("/")))
  no_runtime_evidence = not any(
      (args.out_dir / name).exists()
      for name in ("run.json", "probe.json", "tokens.jsonl"))

  closure = {
      "route": CLOSED_ROUTE,
      "class": "explicit_product_route_rejection",
      "reason": (
          "The rowblock16 26-mask candidate clears only the short same-host "
          "Vulkan bring-up floor at `19.57836215` tok/s with confirm "
          "`19.56354131`, but direct router math/code distribution fails. The "
          "static correction is barred by its pre-registered holdout "
          "no-regression rule, learned correction requires a new owner-approved "
          "contract, and all bounded exact-kernel successors satisfy their "
          "terminal stop conditions. The product route cannot enter token, "
          "context, prefill, or final matrix promotion under the locked contract."),
      "evidence": _rel(args.predecessor.parent),
      "context_evidence": (
          "output/r2-gpu-attention-front-rowblock16-26mask-noqueue-"
          "distribution-20260708Tseq221Z, "
          "output/r2-gpu-acceptance-matrix-router-both-rowblock16-26mask-"
          "20260708Tseq222Z, "
          "output/seq568-sparse-head-logit-bias-holdout-gate-"
          "20260710Tseq568Z, "
          "output/seq624-gpu-software-correctly-rounded-qk-scale-primitive-"
          "route-close-gate-20260710Tseq624Z"),
      "reopen_condition": (
          "an owner-approved correctness/data/runtime contract with a new "
          "independent untouched evaluation split, or a newly localized source "
          "boundary that satisfies a recorded rejected-route reopen condition "
          "and its floor kill-number; re-enter at teacher-forced distribution "
          "before deterministic tokens or the context matrix"),
  }
  checks = [
      {"name": "seq625_selected_explicit_product_route_rejection",
       "pass": predecessor_selects},
      {"name": "goal_acceptance_shape_allows_explicit_route_rejection",
       "pass": goal_allows_rejection},
      {"name": "short_floor_clear_is_preserved_as_diagnostic_only",
       "pass": short_evidence_is_diagnostic_only},
      {"name": "rejected_route_is_absent_or_matches_terminal_evidence",
       "pass": existing_is_consistent},
      {"name": "route_rejection_created_no_runtime_evidence",
       "pass": no_runtime_evidence},
  ]
  required = all(bool(row["pass"]) for row in checks)
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "sequence": args.sequence,
      "inputs": {
          "routes": _rel(args.routes),
          "rejected": _rel(args.rejected),
          "predecessor": _rel(args.predecessor),
          "goal": _rel(args.goal),
      },
      "closure": closure,
      "checks": checks,
      "required_checks_passed": required,
      "goal_route_rejection_recorded": required,
      "project_route_terminal_under_current_contract": required,
      "owner_contract_change_required_for_reopen": True,
      "source_repair_allowed": False,
      "target_compile_allowed": False,
      "component_probe_allowed": False,
      "decode_integration_allowed": False,
      "token_row_allowed": False,
      "validation_or_test_allowed": False,
      "speedup_claims_allowed": False,
      "promotion_allowed": False,
      "disposition": (
          "explicitly_reject_gpu_arc_b390_rowblock16_26mask_product_route"
          if required else "repair_product_route_rejection_evidence"),
      "selected_next_route": (
          TERMINAL_ROUTE if required else CURRENT_ROUTE),
      "next_route_reason": (
          "The current project route is terminal under the locked contract. "
          "Preserve diagnostic evidence and reopen only through the recorded "
          "owner-contract or new-boundary condition."
          if required else
          "Repair goal, route-exhaustion, or short-candidate evidence before "
          "recording an explicit route rejection."),
  }


def write_outputs(metrics: dict[str, Any], out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  (out_dir / "route-rejection.json").write_text(
      json.dumps(metrics["closure"], indent=2, sort_keys=True) + "\n",
      encoding="utf-8")
  (out_dir / "manifest.json").write_text(
      json.dumps({
          "schema_version": metrics["schema_version"],
          "workstream": metrics["workstream"],
          "tool": _rel(Path(__file__)),
          "inputs": metrics["inputs"],
          "route_rejection": _rel(out_dir / "route-rejection.json"),
          "goal_route_rejection_recorded": metrics[
              "goal_route_rejection_recorded"],
          "selected_next_route": metrics["selected_next_route"],
          "target_compile_allowed": False,
          "token_row_allowed": False,
      }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  failed = [row["name"] for row in metrics["checks"] if not row["pass"]]
  lines = [
      f"# Seq{metrics['sequence']} Native Product Route Rejection",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- goal_route_rejection_recorded: `{str(metrics['goal_route_rejection_recorded']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- failed_checks: `{failed}`",
      "",
      metrics["closure"]["reason"],
      "",
      metrics["next_route_reason"],
      "",
      "This gate used existing evidence only; no target command ran.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--sequence", type=int, default=626)
  parser.add_argument("--routes", type=Path,
                      default=ACTIVE / "routes-ledger.json")
  parser.add_argument("--rejected", type=Path,
                      default=ACTIVE / "rejected-routes.json")
  parser.add_argument(
      "--predecessor", type=Path,
      default=ROOT / (
          "output/seq625-gpu-native-exact-kernel-route-exhaustion-"
          "reflection-gate-20260710Tseq625Z/metrics.json"))
  parser.add_argument(
      "--goal", type=Path,
      default=ROOT / "goals/intel-qwen36-35b-a3b-q4km-engine.md")
  parser.add_argument(
      "--out-dir", type=Path,
      default=ROOT / (
          "output/seq626-gpu-native-product-route-rejection-gate-"
          "20260710Tseq626Z"))
  args = parser.parse_args()
  metrics = compute(args)
  write_outputs(metrics, args.out_dir)
  print(json.dumps({
      "required_checks_passed": metrics["required_checks_passed"],
      "goal_route_rejection_recorded": metrics[
          "goal_route_rejection_recorded"],
      "disposition": metrics["disposition"],
      "selected_next_route": metrics["selected_next_route"],
      "out_dir": _rel(args.out_dir),
  }, sort_keys=True))
  return 0 if metrics["required_checks_passed"] else 2


if __name__ == "__main__":
  raise SystemExit(main())
