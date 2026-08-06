#!/usr/bin/env python3
"""Close the failed Level Zero component and select one primitive feasibility gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = (
    "intel-qwen36-gpu-level-zero-postconv-recurrent-component-route-close-v0")
CURRENT_ROUTE = (
    "gpu_level_zero_postconv_recurrent_component_route_close_gate")
SELECTED_NEXT_ROUTE = (
    "gpu_software_correctly_rounded_qk_scale_primitive_feasibility_gate")
CLOSED_ROUTE = "level_zero_ocloc_fused_postconv_recurrent_v1"


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


def _mismatch(row: dict[str, Any], name: str) -> int | None:
  value = row.get("comparisons", {}).get(name, {}).get("mismatch_count")
  return value if isinstance(value, int) else None


def compute(args: argparse.Namespace) -> dict[str, Any]:
  routes = _load(args.routes)
  rejected = _load(args.rejected)
  predecessor = _load(args.predecessor)
  design = _load(args.design)
  prior_close = _load(args.prior_close)
  rows = predecessor.get("rows", [])
  rows = rows if isinstance(rows, list) else []

  route_selects = (
      predecessor.get("required_checks_passed") is True
      and predecessor.get("measurement_complete") is True
      and predecessor.get("component_passed") is False
      and predecessor.get("component_rejected") is True
      and predecessor.get("component_route_close_allowed") is True
      and predecessor.get("selected_next_route") == CURRENT_ROUTE
      and len(rows) == 2
      and _has_candidate(routes, 617, CURRENT_ROUTE)
      and _has_switch(
          routes, 617,
          "select_gpu_level_zero_postconv_recurrent_component_route_close_gate"))

  exact_v = (
      len(rows) == 2
      and all(_mismatch(row, "v_conv_predelta_vs_cpu") == 0 for row in rows))
  qk_fail = (
      len(rows) == 2
      and all(
          (_mismatch(row, "q_conv_predelta_vs_cpu") or 0) > 0
          and (_mismatch(row, "k_conv_predelta_vs_cpu") or 0) > 0
          for row in rows))
  downstream_fails = (
      len(rows) == 2
      and all(
          (_mismatch(row, name) or 0) > 0
          for row in rows
          for name in ("attention_vs_cpu", "state_vs_cpu", "final_vs_cpu")))
  deterministic_failure = (
      len(rows) == 2
      and all(
          _mismatch(rows[0], name) == _mismatch(rows[1], name)
          for name in (
              "q_conv_predelta_vs_cpu", "k_conv_predelta_vs_cpu",
              "v_conv_predelta_vs_cpu", "attention_vs_cpu", "state_vs_cpu",
              "final_vs_cpu")))
  budget_passes = (
      len(rows) == 2
      and all(row.get("budget_passed") is True for row in rows))

  stop_condition = design.get("design", {}).get("stop_condition", "")
  stop_condition_satisfied = (
      "one source implementation and one fresh target compile" in stop_condition
      and "Any non-exact boundary" in stop_condition
      and "do not sweep compiler flags, workgroups, or arithmetic order"
      in stop_condition)

  prior_reopen = prior_close.get("closure", {}).get("reopen_condition", "")
  correctly_rounded_reopen_exists = (
      prior_close.get("bundle_closed") is True
      and "correctly-rounded postconv/recurrent primitive" in prior_reopen
      and "new component contract and floor kill-number" in prior_reopen)

  ceiling = design.get("design", {}).get("component_gate", {}).get(
      "whole_shell_added_us_per_layer_max")
  added_rows = [row.get("candidate_added_min_us") for row in rows]
  headrooms = [
      float(ceiling) - float(value)
      for value in added_rows
      if isinstance(ceiling, (int, float)) and isinstance(value, (int, float))
  ]
  primitive_added_us_max = min(headrooms) if len(headrooms) == 2 else None
  feasibility_headroom_exists = (
      primitive_added_us_max is not None and primitive_added_us_max > 0.0)

  existing_rejection = next((
      row for row in rejected.get("rejected", [])
      if isinstance(row, dict) and row.get("route") == CLOSED_ROUTE), None)
  existing_rejection_consistent = (
      existing_rejection is None
      or (
          existing_rejection.get("class") == "correctness_route_closed"
          and existing_rejection.get("evidence", "").rstrip("/")
          == _rel(args.predecessor.parent).rstrip("/")
          and "61.886858994 us/layer" in existing_rejection.get(
              "reopen_condition", "")))
  no_runtime_evidence = not any(
      (args.out_dir / name).exists()
      for name in ("run.json", "probe.json", "tokens.jsonl"))

  checks = [
      {"name": "seq617_selected_no_target_level_zero_close",
       "pass": route_selects},
      {"name": "failure_localizes_to_qk_after_exact_v_in_both_rows",
       "pass": exact_v and qk_fail and downstream_fails},
      {"name": "failure_shape_is_identical_across_repeat_and_confirm",
       "pass": deterministic_failure},
      {"name": "paired_level_zero_timing_passes",
       "pass": budget_passes},
      {"name": "recorded_one_source_one_pair_stop_condition_is_satisfied",
       "pass": stop_condition_satisfied},
      {"name": "prior_correctly_rounded_primitive_reopen_condition_exists",
       "pass": correctly_rounded_reopen_exists},
      {"name": "seq617_timing_leaves_positive_primitive_feasibility_headroom",
       "pass": feasibility_headroom_exists,
       "detail": {"primitive_added_us_per_layer_max": primitive_added_us_max}},
      {"name": "closed_route_is_absent_or_matches_terminal_evidence",
       "pass": existing_rejection_consistent},
      {"name": "route_close_created_no_runtime_evidence",
       "pass": no_runtime_evidence},
  ]
  required = all(bool(row["pass"]) for row in checks)
  headroom_text = (
      f"{primitive_added_us_max:.9f}" if primitive_added_us_max is not None
      else "unavailable")
  closure = {
      "route": CLOSED_ROUTE,
      "class": "correctness_route_closed",
      "reason": (
          "The one locked Level Zero component is faster than the current "
          "OpenCL shell and makes V-SiLU bit-exact, but both paired rows first "
          "fail at Q/K L2 normalization (Q `90/2048`, K `336/2048`) and "
          "reproduce downstream state/final mismatches. The seq614 one-source/"
          "one-pair stop condition therefore closes compiler-flag, workgroup, "
          "and arithmetic-order variants."),
      "evidence": _rel(args.predecessor.parent),
      "context_evidence": (
          f"{_rel(args.design.parent)}, {_rel(args.prior_close.parent)}"),
      "reopen_condition": (
          "an independently validated software correctly-rounded Q/K inverse-"
          "L2 scale primitive that proves bit-exact Q/K on captured inputs, "
          f"adds no more than {headroom_text} us/layer over the seq617 Level "
          "Zero candidate in both same-device rows, and is governed by a new "
          "component contract; do not retry compiler flags, workgroups, or "
          "arithmetic-order variants of the v1 module"),
  }
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "sequence": args.sequence,
      "inputs": {
          "routes": _rel(args.routes),
          "rejected": _rel(args.rejected),
          "predecessor": _rel(args.predecessor),
          "design": _rel(args.design),
          "prior_close": _rel(args.prior_close),
      },
      "closure": closure,
      "primitive_feasibility": {
          "whole_shell_added_us_per_layer_max": ceiling,
          "seq617_candidate_added_min_us": added_rows,
          "primitive_added_us_per_layer_max": primitive_added_us_max,
          "q_mismatch_count": _mismatch(rows[0], "q_conv_predelta_vs_cpu")
          if len(rows) == 2 else None,
          "k_mismatch_count": _mismatch(rows[0], "k_conv_predelta_vs_cpu")
          if len(rows) == 2 else None,
      },
      "checks": checks,
      "required_checks_passed": required,
      "level_zero_component_closed": required,
      "primitive_feasibility_gate_allowed": required,
      "source_repair_allowed": False,
      "target_compile_allowed": False,
      "component_probe_allowed": False,
      "decode_integration_allowed": False,
      "token_row_allowed": False,
      "validation_or_test_allowed": False,
      "speedup_claims_allowed": False,
      "promotion_allowed": False,
      "disposition": (
          "close_level_zero_ocloc_fused_postconv_recurrent_v1_select_"
          "correctly_rounded_qk_primitive_feasibility"
          if required else "repair_level_zero_component_route_close"),
      "selected_next_route": (
          SELECTED_NEXT_ROUTE if required else CURRENT_ROUTE),
      "next_route_reason": (
          "The v1 Level Zero component is closed, but its positive timing "
          "margin leaves one bounded route question: can a new software "
          "correctly-rounded Q/K scale primitive meet exactness inside the "
          f"conservative {headroom_text} us/layer incremental ceiling? Audit "
          "that feasibility without source or target execution next."
          if required else
          "Repair terminal evidence, stop-condition, timing-headroom, or "
          "rejected-ledger consistency before selecting another route."),
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
          "level_zero_component_closed": metrics["level_zero_component_closed"],
          "primitive_feasibility": metrics["primitive_feasibility"],
          "selected_next_route": metrics["selected_next_route"],
          "source_repair_allowed": False,
          "target_compile_allowed": False,
          "token_row_allowed": False,
      }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  failed = [row["name"] for row in metrics["checks"] if not row["pass"]]
  lines = [
      f"# Seq{metrics['sequence']} Native Level Zero Component Route Close",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- level_zero_component_closed: `{str(metrics['level_zero_component_closed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- primitive_added_us_per_layer_max: `"
      f"{metrics['primitive_feasibility']['primitive_added_us_per_layer_max']}`",
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
  parser.add_argument("--sequence", type=int, default=618)
  parser.add_argument("--routes", type=Path,
                      default=ACTIVE / "routes-ledger.json")
  parser.add_argument("--rejected", type=Path,
                      default=ACTIVE / "rejected-routes.json")
  parser.add_argument(
      "--predecessor", type=Path,
      default=ROOT / (
          "output/seq617-gpu-level-zero-postconv-recurrent-component-probe-"
          "gate-20260710Tseq617Z/metrics.json"))
  parser.add_argument(
      "--design", type=Path,
      default=ROOT / (
          "output/seq614-gpu-level-zero-postconv-recurrent-component-design-"
          "gate-20260710Tseq614Z/metrics.json"))
  parser.add_argument(
      "--prior-close", type=Path,
      default=ROOT / (
          "output/seq605-all-linear-preprojection-parity-component-final-"
          "route-close-gate-20260710Tseq605Z/metrics.json"))
  parser.add_argument(
      "--out-dir", type=Path,
      default=ROOT / (
          "output/seq618-gpu-level-zero-postconv-recurrent-component-route-"
          "close-gate-20260710Tseq618Z"))
  args = parser.parse_args()
  metrics = compute(args)
  write_outputs(metrics, args.out_dir)
  print(json.dumps({
      "required_checks_passed": metrics["required_checks_passed"],
      "level_zero_component_closed": metrics["level_zero_component_closed"],
      "primitive_feasibility_gate_allowed": metrics[
          "primitive_feasibility_gate_allowed"],
      "disposition": metrics["disposition"],
      "selected_next_route": metrics["selected_next_route"],
      "out_dir": _rel(args.out_dir),
  }, sort_keys=True))
  return 0 if metrics["required_checks_passed"] else 2


if __name__ == "__main__":
  raise SystemExit(main())
