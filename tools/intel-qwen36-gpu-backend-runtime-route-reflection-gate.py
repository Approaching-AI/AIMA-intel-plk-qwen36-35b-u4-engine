#!/usr/bin/env python3
"""Select a materially different native runtime route from existing evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = "intel-qwen36-gpu-backend-runtime-route-reflection-v0"
CURRENT_ROUTE = "gpu_backend_runtime_route_reflection_gate"
SELECTED_NEXT_ROUTE = "gpu_vulkan_postconv_recurrent_component_preflight_gate"


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


def _rejected(rejected: dict[str, Any], route: str) -> bool:
  return any(
      isinstance(row, dict) and row.get("route") == route
      for row in rejected.get("rejected", []))


def compute(args: argparse.Namespace) -> dict[str, Any]:
  routes = _load(args.routes)
  predecessor = _load(args.predecessor)
  rejected = _load(args.rejected)
  target = _load(args.target_contract)
  goal_text = args.goal.read_text(encoding="utf-8")
  agents_text = args.agents.read_text(encoding="utf-8")
  inspiration = args.inspiration.read_text(encoding="utf-8")
  completed = target.get("r0_refresh", {}).get("completed_items", [])
  packages = target.get("runtime", {}).get("system_packages_present", [])

  predecessor_selects = (
      predecessor.get("required_checks_passed") is True
      and predecessor.get("bundle_closed") is True
      and predecessor.get("runtime_route_reflection_allowed") is True
      and predecessor.get("selected_next_route") == CURRENT_ROUTE
      and _has_candidate(routes, 605, CURRENT_ROUTE)
      and _has_switch(
          routes, 605, "select_gpu_backend_runtime_route_reflection_gate"))
  opencl_board_closed = _rejected(
      rejected,
      "router_prompt_distribution_cpuorder_preprojection_bundle_v1")
  learned_board_closed = _rejected(
      rejected, "router_prompt_distribution_learned_head_correction_board")
  denominator_boundary = (
      "OpenVINO and llama.cpp are denominators and correctness references, not the"
      in agents_text
      and "final native runtime dependency" in agents_text)
  vulkan_target_evidence = all(any(marker in str(item) for item in completed)
                               for marker in (
                                   "llama.cpp/Vulkan denominator preflight",
                                   "llama.cpp/Vulkan 262144 paired"))
  level_zero_target_evidence = (
      "intel-level-zero-gpu" in packages
      and any("libze1" in str(package) for package in packages))
  route_inspiration_supports_vulkan_floor = (
      "llama.cpp **Vulkan** on the identical host+model" in inspiration
      and "decode **~19.5 tok/s**" in inspiration)
  route_inspiration_supports_level_zero = (
      "Level-Zero/SYCL in-order queue" in inspiration)
  goal_requires_native = (
      "native" in goal_text.lower()
      and "Qwen3.6-35B-A3B" in goal_text)

  candidates = [
      {
          "rank": 1,
          "id": "native_vulkan_postconv_recurrent_component",
          "status": "selected",
          "reason": (
              "Same-host Vulkan already has model-correct denominator and "
              "floor evidence. A self-owned dynamic Vulkan component can "
              "test the only remaining postconv/recurrent boundary without "
              "depending on llama.cpp or OpenVINO at runtime."),
          "first_gate": SELECTED_NEXT_ROUTE,
      },
      {
          "rank": 2,
          "id": "native_level_zero_postconv_recurrent_component",
          "status": "parked",
          "reason": (
              "The Level Zero loader/GPU packages are present, but the repo "
              "has no model-correct Level Zero denominator or component "
              "evidence. Reopen only if Vulkan preflight cannot produce an "
              "independent self-owned shader path."),
      },
      {
          "rank": 3,
          "id": "openvino_or_llamacpp_runtime_embedding",
          "status": "rejected_by_mission",
          "reason": (
              "Both are references/denominators and cannot become the final "
              "native runtime dependency."),
      },
      {
          "rank": 4,
          "id": "opencl_postconv_arithmetic_retry",
          "status": "rejected_by_ledger",
          "reason": (
              "Seq605 closes the OpenCL arithmetic/order board under its "
              "one-repair stop condition."),
      },
  ]
  selected_contract = {
      "runtime": "Vulkan loaded dynamically by the native engine",
      "dependency_rule": (
          "no llama.cpp/OpenVINO library, executable, source-tree, or process "
          "dependency in the component or final runtime"),
      "scope": "captured layer0 postconv plus recurrent/final component only",
      "first_gate": {
          "type": "target preflight without model execution",
          "required": [
              "Vulkan loader and Intel GPU device visible",
              "compute queue family available",
              "SPIR-V shader compiler or checked-in generated shader route available",
              "minimal create/destroy smoke leaves no process or device resource",
          ],
      },
      "component_gate": {
          "inputs": "same seq604 captured conv output and recurrent seed",
          "repeat_and_confirm": True,
          "must_improve_all_remaining_max_abs": {
              "recurrent_state_below": 9.5367431640625e-7,
              "attention_output_below": 1.1920928955078125e-7,
              "final_output_below": 1.4901161193847656e-8,
          },
          "whole_changed_shell_added_us_max": 6.841858993929781,
          "token_row_allowed": False,
      },
  }
  no_runtime_evidence = not any(
      (args.out_dir / name).exists()
      for name in ("run.json", "probe.json", "tokens.jsonl"))
  checks = [
      {"name": "seq605_selected_backend_runtime_reflection_only",
       "pass": predecessor_selects},
      {"name": "opencl_arithmetic_and_learned_correction_boards_are_closed",
       "pass": opencl_board_closed and learned_board_closed},
      {"name": "mission_keeps_openvino_and_llamacpp_out_of_final_runtime",
       "pass": denominator_boundary and goal_requires_native},
      {"name": "same_host_vulkan_has_model_and_floor_evidence",
       "pass": vulkan_target_evidence and route_inspiration_supports_vulkan_floor},
      {"name": "level_zero_is_present_but_has_weaker_existing_evidence",
       "pass": level_zero_target_evidence and route_inspiration_supports_level_zero},
      {"name": "reflection_created_no_runtime_evidence",
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
          "target_contract": _rel(args.target_contract),
          "goal": _rel(args.goal),
          "agents": _rel(args.agents),
          "inspiration": _rel(args.inspiration),
      },
      "candidates": candidates,
      "selected_contract": selected_contract,
      "checks": checks,
      "required_checks_passed": required,
      "vulkan_preflight_allowed": required,
      "component_source_allowed": False,
      "component_probe_allowed": False,
      "decode_integration_allowed": False,
      "token_row_allowed": False,
      "validation_or_test_allowed": False,
      "speedup_claims_allowed": False,
      "promotion_allowed": False,
      "disposition": (
          "select_native_vulkan_postconv_recurrent_component_preflight"
          if required else "repair_native_runtime_route_reflection"),
      "selected_next_route": (
          SELECTED_NEXT_ROUTE if required else CURRENT_ROUTE),
      "next_route_reason": (
          "Vulkan has the strongest existing same-host model-correct evidence "
          "and can remain an independent native runtime. Preflight only its "
          "loader/device/compute/SPIR-V path next; do not execute the model or "
          "write component source yet."
          if required else
          "Repair backend evidence or mission-boundary classification before "
          "any target or source work."),
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
          "selected_next_route": metrics["selected_next_route"],
          "vulkan_preflight_allowed": metrics["vulkan_preflight_allowed"],
          "component_source_allowed": False,
          "token_row_allowed": False,
      }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  failed = [row["name"] for row in metrics["checks"] if not row["pass"]]
  lines = [
      f"# Seq{metrics['sequence']} Native Runtime Route Reflection",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- failed_checks: `{failed}`",
      "",
  ]
  for candidate in metrics["candidates"]:
    lines.append(
        f"- rank {candidate['rank']} `{candidate['id']}`: "
        f"`{candidate['status']}` - {candidate['reason']}")
  lines += [
      "",
      metrics["next_route_reason"],
      "",
      "This gate used existing evidence only; no target command ran.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--sequence", type=int, default=606)
  parser.add_argument("--routes", type=Path,
                      default=ACTIVE / "routes-ledger.json")
  parser.add_argument("--rejected", type=Path,
                      default=ACTIVE / "rejected-routes.json")
  parser.add_argument(
      "--predecessor", type=Path,
      default=ROOT / (
          "output/seq605-all-linear-preprojection-parity-component-final-"
          "route-close-gate-20260710Tseq605Z/metrics.json"))
  parser.add_argument("--target-contract", type=Path,
                      default=ROOT / "contracts/intel-qwen36-target-contract.json")
  parser.add_argument("--goal", type=Path,
                      default=ROOT / "goals/intel-qwen36-35b-a3b-q4km-engine.md")
  parser.add_argument("--agents", type=Path, default=ROOT / "AGENTS.md")
  parser.add_argument(
      "--inspiration", type=Path,
      default=ROOT / (
          "doc/reference/intel-qwen36-35b-a3b-gguf-q4km/"
          "route-inspiration-from-siblings-2026-06-29.md"))
  parser.add_argument(
      "--out-dir", type=Path,
      default=ROOT / (
          "output/seq606-gpu-backend-runtime-route-reflection-gate-"
          "20260710Tseq606Z"))
  args = parser.parse_args()
  metrics = compute(args)
  write_outputs(metrics, args.out_dir)
  print(json.dumps({
      "required_checks_passed": metrics["required_checks_passed"],
      "disposition": metrics["disposition"],
      "vulkan_preflight_allowed": metrics["vulkan_preflight_allowed"],
      "selected_next_route": metrics["selected_next_route"],
      "out_dir": _rel(args.out_dir),
  }, sort_keys=True))
  return 0 if metrics["required_checks_passed"] else 2


if __name__ == "__main__":
  raise SystemExit(main())
