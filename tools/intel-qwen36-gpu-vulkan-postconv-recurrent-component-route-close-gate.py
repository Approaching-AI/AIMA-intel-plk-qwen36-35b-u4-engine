#!/usr/bin/env python3
"""Close the failed precise Vulkan component and select Level Zero preflight."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = (
    "intel-qwen36-gpu-vulkan-postconv-recurrent-component-route-close-v0")
CURRENT_ROUTE = (
    "gpu_vulkan_postconv_recurrent_component_route_close_gate")
SELECTED_NEXT_ROUTE = (
    "gpu_level_zero_postconv_recurrent_component_preflight_gate")
CLOSED_ROUTE = "gpu_vulkan_precise_postconv_recurrent_v1"


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
  design = _load(args.design)
  source = _load(args.source)
  reflection = _load(args.reflection)
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
      and _has_candidate(routes, 611, CURRENT_ROUTE)
      and _has_switch(
          routes, 611,
          "select_gpu_vulkan_postconv_recurrent_component_route_close_gate"))
  exactness_fails_both = all(
      row.get("candidate_bit_exact") is False for row in rows)
  budget_fails_both = all(row.get("budget_passed") is False for row in rows)
  deterministic_failure = (
      len(rows) == 2
      and all(
          rows[0].get("comparisons", {}).get(name, {}).get("mismatch_count")
          == rows[1].get("comparisons", {}).get(name, {}).get("mismatch_count")
          for name in (
              "q_conv_predelta_vs_cpu", "k_conv_predelta_vs_cpu",
              "v_conv_predelta_vs_cpu", "attention_vs_cpu", "state_vs_cpu",
              "final_vs_cpu")))
  stop_condition = design.get("design", {}).get("stop_condition", "")
  stop_condition_satisfied = (
      "one source implementation and one fresh target compile" in stop_condition
      and "close this candidate without arithmetic/order variants" in stop_condition
      and source.get("component_source_passed") is True
      and source.get("spirv", {}).get(
          "iq36_postconv_cpuorder", {}).get("no_contraction_count") == 10
      and source.get("spirv", {}).get(
          "iq36_delta_recurrent_cpuorder", {}).get("no_contraction_count") == 18)
  level_zero_ranked_second = any(
      isinstance(row, dict)
      and row.get("id") == "native_level_zero_postconv_recurrent_component"
      and row.get("rank") == 2
      and row.get("status") == "parked"
      for row in reflection.get("candidates", []))
  already_rejected = any(
      isinstance(row, dict) and row.get("route") == CLOSED_ROUTE
      for row in _load(args.rejected).get("rejected", []))
  no_runtime_evidence = not any(
      (args.out_dir / name).exists()
      for name in ("run.json", "probe.json", "tokens.jsonl"))
  checks = [
      {"name": "seq611_selected_no_target_vulkan_close",
       "pass": route_selects},
      {"name": "exactness_and_floor_budget_fail_independently_in_both_rows",
       "pass": exactness_fails_both and budget_fails_both},
      {"name": "failure_shape_is_identical_across_repeat_and_confirm",
       "pass": deterministic_failure},
      {"name": "recorded_one_implementation_stop_condition_is_satisfied",
       "pass": stop_condition_satisfied},
      {"name": "level_zero_is_the_existing_ranked_material_successor",
       "pass": level_zero_ranked_second},
      {"name": "closed_route_is_not_already_in_rejected_ledger",
       "pass": not already_rejected},
      {"name": "route_close_created_no_runtime_evidence",
       "pass": no_runtime_evidence},
  ]
  required = all(bool(row["pass"]) for row in checks)
  closure = {
      "route": CLOSED_ROUTE,
      "class": "correctness_and_budget_route_closed",
      "reason": (
          "The one locked Vulkan component fails deterministically at its "
          "first Q/K/V boundary in both paired rows and also adds `207.869` / "
          "`144.851 us` versus the `6.841858994 us` ceiling. Audited SPIR-V, "
          "binary/shader identities, device selection, and cleanup pass, so "
          "exactness and budget are independent terminal failures."),
      "evidence": _rel(args.predecessor.parent),
      "context_evidence": (
          f"{_rel(args.design.parent)}, {_rel(args.source.parent)}"),
      "reopen_condition": (
          "an independently validated non-GLSL.std.450 FP64 transcendental "
          "primitive or different Vulkan kernel architecture with a new "
          "component contract that first proves an absolute <=6.841858994 us "
          "added-wall ceiling; do not retry precise/order variants of this "
          "two-dispatch design"),
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
          "source": _rel(args.source),
          "reflection": _rel(args.reflection),
      },
      "closure": closure,
      "checks": checks,
      "required_checks_passed": required,
      "vulkan_component_closed": required,
      "level_zero_preflight_allowed": required,
      "source_repair_allowed": False,
      "target_compile_allowed": False,
      "component_probe_allowed": False,
      "decode_integration_allowed": False,
      "token_row_allowed": False,
      "validation_or_test_allowed": False,
      "speedup_claims_allowed": False,
      "promotion_allowed": False,
      "disposition": (
          "close_vulkan_precise_postconv_recurrent_v1_select_level_zero_preflight"
          if required else "repair_vulkan_component_route_close"),
      "selected_next_route": (
          SELECTED_NEXT_ROUTE if required else CURRENT_ROUTE),
      "next_route_reason": (
          "The precise Vulkan component is closed on independent exactness and "
          "budget failures. Preflight only the already-ranked self-owned Level "
          "Zero path next; no model, component dispatch, or token is allowed."
          if required else
          "Repair terminal evidence, stop-condition, or rejected-ledger "
          "consistency before selecting another runtime."),
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
          "vulkan_component_closed": metrics["vulkan_component_closed"],
          "selected_next_route": metrics["selected_next_route"],
          "source_repair_allowed": False,
          "target_compile_allowed": False,
          "token_row_allowed": False,
      }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  failed = [row["name"] for row in metrics["checks"] if not row["pass"]]
  lines = [
      f"# Seq{metrics['sequence']} Native Vulkan Component Route Close",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- vulkan_component_closed: `{str(metrics['vulkan_component_closed']).lower()}`",
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
  parser.add_argument("--sequence", type=int, default=612)
  parser.add_argument("--routes", type=Path,
                      default=ACTIVE / "routes-ledger.json")
  parser.add_argument("--rejected", type=Path,
                      default=ACTIVE / "rejected-routes.json")
  parser.add_argument(
      "--predecessor", type=Path,
      default=ROOT / (
          "output/seq611-gpu-vulkan-postconv-recurrent-component-probe-gate-"
          "20260710Tseq611Z/metrics.json"))
  parser.add_argument(
      "--design", type=Path,
      default=ROOT / (
          "output/seq608-gpu-vulkan-postconv-recurrent-component-design-gate-"
          "20260710Tseq608Z/metrics.json"))
  parser.add_argument(
      "--source", type=Path,
      default=ROOT / (
          "output/seq609-gpu-vulkan-postconv-recurrent-component-source-gate-"
          "20260710Tseq609Z/metrics.json"))
  parser.add_argument(
      "--reflection", type=Path,
      default=ROOT / (
          "output/seq606-gpu-backend-runtime-route-reflection-gate-"
          "20260710Tseq606Z/metrics.json"))
  parser.add_argument(
      "--out-dir", type=Path,
      default=ROOT / (
          "output/seq612-gpu-vulkan-postconv-recurrent-component-route-close-"
          "gate-20260710Tseq612Z"))
  args = parser.parse_args()
  metrics = compute(args)
  write_outputs(metrics, args.out_dir)
  print(json.dumps({
      "required_checks_passed": metrics["required_checks_passed"],
      "vulkan_component_closed": metrics["vulkan_component_closed"],
      "level_zero_preflight_allowed": metrics["level_zero_preflight_allowed"],
      "disposition": metrics["disposition"],
      "selected_next_route": metrics["selected_next_route"],
      "out_dir": _rel(args.out_dir),
  }, sort_keys=True))
  return 0 if metrics["required_checks_passed"] else 2


if __name__ == "__main__":
  raise SystemExit(main())
