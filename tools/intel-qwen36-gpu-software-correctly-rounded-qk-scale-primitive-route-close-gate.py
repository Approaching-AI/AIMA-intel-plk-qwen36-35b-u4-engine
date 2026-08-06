#!/usr/bin/env python3
"""Close the no-effect reciprocal primitive and select route reflection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = (
    "intel-qwen36-gpu-software-correctly-rounded-qk-scale-route-close-v0")
CURRENT_ROUTE = (
    "gpu_software_correctly_rounded_qk_scale_primitive_route_close_gate")
SELECTED_NEXT_ROUTE = "gpu_native_exact_kernel_route_exhaustion_reflection_gate"
CLOSED_ROUTE = "level_zero_ocloc_cr_recip_postconv_recurrent_v2"


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


def _comparison_signature(row: dict[str, Any]) -> dict[str, Any]:
  comparisons = row.get("comparisons", {})
  return {
      name: {
          "mismatch_count": comparisons.get(name, {}).get("mismatch_count"),
          "max_abs_diff": comparisons.get(name, {}).get("max_abs_diff"),
      }
      for name in (
          "q_conv_predelta_vs_cpu", "k_conv_predelta_vs_cpu",
          "v_conv_predelta_vs_cpu", "attention_vs_cpu", "state_vs_cpu",
          "final_vs_cpu")
  }


def compute(args: argparse.Namespace) -> dict[str, Any]:
  routes = _load(args.routes)
  rejected = _load(args.rejected)
  predecessor = _load(args.predecessor)
  v1_component = _load(args.v1_component)
  design = _load(args.design)
  source = _load(args.source)
  feasibility = _load(args.feasibility)
  rows = predecessor.get("rows", [])
  rows = rows if isinstance(rows, list) else []
  v1_rows = v1_component.get("rows", [])
  v1_rows = v1_rows if isinstance(v1_rows, list) else []

  route_selects = (
      predecessor.get("required_checks_passed") is True
      and predecessor.get("measurement_complete") is True
      and predecessor.get("component_passed") is False
      and predecessor.get("component_rejected") is True
      and predecessor.get("component_route_close_allowed") is True
      and predecessor.get("selected_next_route") == CURRENT_ROUTE
      and len(rows) == 2
      and _has_candidate(routes, 623, CURRENT_ROUTE)
      and _has_switch(
          routes, 623,
          "select_gpu_software_correctly_rounded_qk_scale_primitive_"
          "route_close_gate"))
  paired_budget_passes = (
      len(rows) == 2
      and all(row.get("budget_passed") is True for row in rows))
  deterministic = (
      len(rows) == 2
      and _comparison_signature(rows[0]) == _comparison_signature(rows[1]))
  unchanged_from_v1 = (
      len(rows) == 2 and len(v1_rows) == 2
      and all(
          _comparison_signature(rows[index])
          == _comparison_signature(v1_rows[index])
          for index in range(2)))
  primitive_executed = (
      source.get("required_checks_passed") is True
      and source.get("component_source_passed") is True
      and source.get("native_module", {}).get("sha256")
      == "1f30e0d20c058e7cb3d6eec6d378b730f0f6421af8a8e2b5c1c6aaf00d57af71"
      and feasibility.get("verifier", {}).get("result", {}).get(
          "mantissas_checked") == 8388608
      and feasibility.get("verifier", {}).get("result", {}).get(
          "mismatch_count") == 0)
  stop_condition = design.get("design", {}).get("stop_condition", "")
  stop_condition_satisfied = (
      "one source implementation and one fresh target compile" in stop_condition
      and "one paired repeat/confirm" in stop_condition
      and "do not add another reciprocal, sqrt, sigmoid" in stop_condition)
  existing_rejection = next((
      row for row in rejected.get("rejected", [])
      if isinstance(row, dict) and row.get("route") == CLOSED_ROUTE), None)
  existing_rejection_consistent = (
      existing_rejection is None
      or (
          existing_rejection.get("class") == "correctness_route_closed"
          and existing_rejection.get("evidence", "").rstrip("/")
          == _rel(args.predecessor.parent).rstrip("/")))
  no_runtime_evidence = not any(
      (args.out_dir / name).exists()
      for name in ("run.json", "probe.json", "tokens.jsonl"))

  checks = [
      {"name": "seq623_selected_no_target_primitive_close",
       "pass": route_selects},
      {"name": "v2_timing_passes_but_correctness_signature_is_deterministic",
       "pass": paired_budget_passes and deterministic},
      {"name": "v2_correctness_signature_is_identical_to_seq617_v1",
       "pass": unchanged_from_v1,
       "detail": _comparison_signature(rows[0]) if len(rows) == 2 else {}},
      {"name": "exhaustively_proven_primitive_was_present_in_executed_module",
       "pass": primitive_executed},
      {"name": "recorded_one_primitive_one_pair_stop_condition_is_satisfied",
       "pass": stop_condition_satisfied},
      {"name": "closed_route_is_absent_or_matches_terminal_evidence",
       "pass": existing_rejection_consistent},
      {"name": "route_close_created_no_runtime_evidence",
       "pass": no_runtime_evidence},
  ]
  required = all(bool(row["pass"]) for row in checks)
  closure = {
      "route": CLOSED_ROUTE,
      "class": "correctness_route_closed",
      "reason": (
          "The exhaustively proven uint64 reciprocal is present in the v2 "
          "module, but both paired rows reproduce the seq617 v1 correctness "
          "signature exactly: Q `90/2048`, K `336/2048`, V exact, state "
          "`36910/524288`, and final `1251/4096`. Timing passes at `-37.528` / "
          "`-57.926 us`, so reciprocal is falsified as the mismatch source and "
          "the seq620 one-pair stop condition is exhausted."),
      "evidence": _rel(args.predecessor.parent),
      "context_evidence": (
          f"{_rel(args.source.parent)}, {_rel(args.feasibility.parent)}, "
          f"{_rel(args.v1_component.parent)}"),
      "reopen_condition": (
          "none within reciprocal, sqrt, sigmoid, compiler-flag, workgroup, or "
          "arithmetic-order variants of this component contract; require a "
          "new source-localized boundary or owner-approved correctness/runtime "
          "contract before another exact-kernel implementation"),
  }
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "sequence": args.sequence,
      "inputs": {
          "routes": _rel(args.routes),
          "rejected": _rel(args.rejected),
          "predecessor": _rel(args.predecessor),
          "v1_component": _rel(args.v1_component),
          "design": _rel(args.design),
          "source": _rel(args.source),
          "feasibility": _rel(args.feasibility),
      },
      "closure": closure,
      "checks": checks,
      "required_checks_passed": required,
      "primitive_route_closed": required,
      "route_reflection_allowed": required,
      "source_repair_allowed": False,
      "target_compile_allowed": False,
      "component_probe_allowed": False,
      "decode_integration_allowed": False,
      "token_row_allowed": False,
      "validation_or_test_allowed": False,
      "speedup_claims_allowed": False,
      "promotion_allowed": False,
      "disposition": (
          "close_level_zero_ocloc_cr_recip_postconv_recurrent_v2_select_"
          "exact_kernel_route_reflection"
          if required else "repair_correctly_rounded_qk_route_close"),
      "selected_next_route": (
          SELECTED_NEXT_ROUTE if required else CURRENT_ROUTE),
      "next_route_reason": (
          "The reciprocal hypothesis and all variants forbidden by the seq620 "
          "stop are closed. Audit the remaining product-correctness board and "
          "select a materially different route without target execution next."
          if required else
          "Repair terminal identity, v1/v2 comparison, stop-condition, or "
          "rejected-ledger consistency before route reflection."),
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
          "closure": metrics["closure"],
          "primitive_route_closed": metrics["primitive_route_closed"],
          "selected_next_route": metrics["selected_next_route"],
          "source_repair_allowed": False,
          "target_compile_allowed": False,
          "token_row_allowed": False,
      }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  failed = [row["name"] for row in metrics["checks"] if not row["pass"]]
  lines = [
      f"# Seq{metrics['sequence']} Correctly-Rounded Q/K Scale Route Close",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- primitive_route_closed: `{str(metrics['primitive_route_closed']).lower()}`",
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
  parser.add_argument("--sequence", type=int, default=624)
  parser.add_argument("--routes", type=Path,
                      default=ACTIVE / "routes-ledger.json")
  parser.add_argument("--rejected", type=Path,
                      default=ACTIVE / "rejected-routes.json")
  parser.add_argument(
      "--predecessor", type=Path,
      default=ROOT / (
          "output/seq623-gpu-software-correctly-rounded-qk-scale-primitive-"
          "probe-gate-20260710Tseq623Z/metrics.json"))
  parser.add_argument(
      "--v1-component", type=Path,
      default=ROOT / (
          "output/seq617-gpu-level-zero-postconv-recurrent-component-probe-"
          "gate-20260710Tseq617Z/metrics.json"))
  parser.add_argument(
      "--design", type=Path,
      default=ROOT / (
          "output/seq620-gpu-software-correctly-rounded-qk-scale-primitive-"
          "design-gate-20260710Tseq620Z/metrics.json"))
  parser.add_argument(
      "--source", type=Path,
      default=ROOT / (
          "output/seq621-gpu-software-correctly-rounded-qk-scale-primitive-"
          "source-gate-20260710Tseq621Z/metrics.json"))
  parser.add_argument(
      "--feasibility", type=Path,
      default=ROOT / (
          "output/seq619-gpu-software-correctly-rounded-qk-scale-primitive-"
          "feasibility-gate-20260710Tseq619Z/metrics.json"))
  parser.add_argument(
      "--out-dir", type=Path,
      default=ROOT / (
          "output/seq624-gpu-software-correctly-rounded-qk-scale-primitive-"
          "route-close-gate-20260710Tseq624Z"))
  args = parser.parse_args()
  metrics = compute(args)
  write_outputs(metrics, args.out_dir)
  print(json.dumps({
      "required_checks_passed": metrics["required_checks_passed"],
      "primitive_route_closed": metrics["primitive_route_closed"],
      "route_reflection_allowed": metrics["route_reflection_allowed"],
      "disposition": metrics["disposition"],
      "selected_next_route": metrics["selected_next_route"],
      "out_dir": _rel(args.out_dir),
  }, sort_keys=True))
  return 0 if metrics["required_checks_passed"] else 2


if __name__ == "__main__":
  raise SystemExit(main())
