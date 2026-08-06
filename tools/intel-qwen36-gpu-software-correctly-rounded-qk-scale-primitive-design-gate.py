#!/usr/bin/env python3
"""Lock one Level Zero v2 component around the proven reciprocal primitive."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = (
    "intel-qwen36-gpu-software-correctly-rounded-qk-scale-design-v0")
CURRENT_ROUTE = (
    "gpu_software_correctly_rounded_qk_scale_primitive_design_gate")
SELECTED_NEXT_ROUTE = (
    "gpu_software_correctly_rounded_qk_scale_primitive_source_gate")


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
  component = _load(args.component)
  v1_design = _load(args.v1_design)
  rejected = _load(args.rejected)
  module_source = args.module_source.read_text(encoding="utf-8")

  route_selects = (
      predecessor.get("required_checks_passed") is True
      and predecessor.get("primitive_feasible") is True
      and predecessor.get("design_gate_allowed") is True
      and predecessor.get("source_allowed") is False
      and predecessor.get("selected_next_route") == CURRENT_ROUTE
      and _has_candidate(routes, 619, CURRENT_ROUTE)
      and _has_switch(
          routes, 619,
          "select_gpu_software_correctly_rounded_qk_scale_primitive_"
          "design_gate"))
  primitive = predecessor.get("design", {})
  verifier = predecessor.get("verifier", {}).get("result", {})
  primitive_proven = (
      primitive.get("primitive_id") == "iq36_cr_recip_normal_f32_u64_v1"
      and primitive.get("floating_divide_or_rsqrt_allowed") is False
      and verifier.get("mantissas_checked") == 8388608
      and verifier.get("mismatch_count") == 0)

  rows = component.get("rows", [])
  rows = rows if isinstance(rows, list) else []
  v1_failure_shape = (
      len(rows) == 2
      and all(row.get("budget_passed") is True for row in rows)
      and all(
          row.get("comparisons", {}).get(
              "v_conv_predelta_vs_cpu", {}).get("mismatch_count") == 0
          and row.get("comparisons", {}).get(
              "q_conv_predelta_vs_cpu", {}).get("mismatch_count") == 90
          and row.get("comparisons", {}).get(
              "k_conv_predelta_vs_cpu", {}).get("mismatch_count") == 336
          for row in rows))
  v1_shape_locked = (
      v1_design.get("required_checks_passed") is True
      and v1_design.get("design", {}).get("candidate")
      == "level_zero_ocloc_fused_postconv_recurrent_v1"
      and v1_design.get("design", {}).get("native_module", {}).get(
          "kernel_count") == 2
      and v1_design.get("design", {}).get("component_gate", {}).get(
          "samples_per_row") == 11)
  headroom = primitive.get("incremental_added_us_per_layer_max")
  whole_shell = v1_design.get("design", {}).get("component_gate", {}).get(
      "whole_shell_added_us_per_layer_max")
  budgets_locked = (
      isinstance(headroom, (int, float))
      and abs(float(headroom) - 61.886858993929785) < 1e-12
      and isinstance(whole_shell, (int, float))
      and abs(float(whole_shell) - 6.841858993929781) < 1e-12)
  prior_rejection = next((
      row for row in rejected.get("rejected", [])
      if isinstance(row, dict)
      and row.get("route") == "level_zero_ocloc_fused_postconv_recurrent_v1"),
      {})
  reopen_condition_satisfied = (
      "software correctly-rounded Q/K inverse-L2 scale primitive"
      in prior_rejection.get("reopen_condition", "")
      and "61.886858994 us/layer" in prior_rejection.get(
          "reopen_condition", ""))
  source_not_prematurely_added = (
      "iq36_cr_recip_normal_f32" not in module_source
      and "1.0f / fmax(sqrt(sum_f32), norm_epsilon)" in module_source)
  no_runtime_evidence = not any(
      (args.out_dir / name).exists()
      for name in ("raw", "run.json", "probe.json", "tokens.jsonl"))

  checks = [
      {"name": "seq619_selected_design_only", "pass": route_selects},
      {"name": "reciprocal_primitive_has_exhaustive_binary32_proof",
       "pass": primitive_proven},
      {"name": "seq617_failure_shape_and_positive_timing_are_preserved",
       "pass": v1_failure_shape},
      {"name": "v1_two_kernel_component_shape_is_reused_not_expanded",
       "pass": v1_shape_locked},
      {"name": "incremental_and_whole_shell_budgets_are_locked",
       "pass": budgets_locked},
      {"name": "v2_contract_satisfies_recorded_v1_reopen_condition",
       "pass": reopen_condition_satisfied},
      {"name": "primitive_source_was_not_added_before_design",
       "pass": source_not_prematurely_added},
      {"name": "design_gate_created_no_target_or_token_evidence",
       "pass": no_runtime_evidence},
  ]
  required = all(bool(row["pass"]) for row in checks)
  design = {
      "schema_version": "intel-qwen36-level-zero-cr-recip-postconv-v2",
      "candidate": "level_zero_ocloc_cr_recip_postconv_recurrent_v2",
      "purpose": (
          "Test whether replacing only the non-correctly-rounded Q/K scale "
          "reciprocal in the timing-positive seq617 Level Zero component makes "
          "all six component boundaries bit-exact."),
      "primitive": {
          "id": "iq36_cr_recip_normal_f32_u64_v1",
          "input": "max(correctly-rounded sqrt(sum_f32), norm_epsilon)",
          "domain": "positive normal binary32 with normal reciprocal",
          "implementation": (
              "exact uint64 2^47/significand quotient and remainder, "
              "round-to-nearest ties-to-even, exponent reassembly"),
          "floating_divide_or_rsqrt_for_scale": False,
          "compiler_correct_divide_sqrt_flag": False,
          "third_party_runtime_dependency": False,
          "exhaustive_significands_checked": 8388608,
      },
      "source_delta": {
          "allowed_file": "engine/gpu/level_zero/iq36_postconv_recurrent.cl",
          "allowed_changes": [
              "add one private inline iq36_cr_recip_normal_f32_u64_v1 helper",
              "replace only the Q/K head_scale reciprocal expression",
          ],
          "forbidden_changes": [
              "sigmoid or SiLU arithmetic",
              "double sum or sqrt input construction",
              "recurrent/final kernel arithmetic",
              "kernel count, names, group sizes, dispatch count, or runtime API",
              "compiler flags, OpenCL bridge, model/decode selectors",
          ],
      },
      "native_module": v1_design.get("design", {}).get("native_module"),
      "runtime_ownership": v1_design.get("design", {}).get(
          "runtime_ownership"),
      "component_gate": {
          "same_physical_device_id": "0xb080",
          "payload": "captured layer0 token15 exact conv output and recurrent seed",
          "samples_per_row": 11,
          "repeat_and_confirm": True,
          "fresh_state_clone_per_sample": True,
          "bit_exact_boundaries": [
              "q_conv_predelta", "k_conv_predelta", "v_conv_predelta",
              "attention_output", "recurrent_state", "final_output",
          ],
          "whole_shell_added_us_per_layer_max": whole_shell,
          "incremental_over_seq617_candidate_us_per_layer_max": headroom,
          "seq617_candidate_wall_min_us": [
              row.get("candidate_wall_min_us") for row in rows],
          "speed_claim_allowed": False,
      },
      "stop_condition": (
          "After one source implementation and one fresh target compile, run "
          "one paired repeat/confirm. Any non-exact Q/K or downstream boundary, "
          "runtime failure, or whole-shell row above 6.841858993929781 us added "
          "closes the correctly-rounded primitive route; do not add another "
          "reciprocal, sqrt, sigmoid, compiler-flag, workgroup, or order variant."),
  }
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "sequence": args.sequence,
      "inputs": {
          "routes": _rel(args.routes),
          "predecessor": _rel(args.predecessor),
          "component": _rel(args.component),
          "v1_design": _rel(args.v1_design),
          "rejected": _rel(args.rejected),
          "module_source": _rel(args.module_source),
      },
      "design": design,
      "checks": checks,
      "required_checks_passed": required,
      "design_passed": required,
      "source_gate_allowed": required,
      "target_command_allowed": False,
      "component_probe_allowed": False,
      "decode_integration_allowed": False,
      "token_row_allowed": False,
      "speedup_claims_allowed": False,
      "promotion_allowed": False,
      "disposition": (
          "accept_level_zero_ocloc_cr_recip_postconv_recurrent_v2_design"
          if required else "repair_correctly_rounded_qk_design"),
      "selected_next_route": (
          SELECTED_NEXT_ROUTE if required else CURRENT_ROUTE),
      "next_route_reason": (
          "The v2 contract changes only the proven reciprocal surface and "
          "preserves the seq617 runtime/kernel shape, exactness ruler, and "
          "whole-shell budget. Add and locally source-gate that one helper "
          "next; target execution remains blocked."
          if required else
          "Repair route selection, primitive proof, reopen condition, source "
          "isolation, or budget contract before adding source."),
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
          "design": metrics["design"],
          "design_passed": metrics["design_passed"],
          "selected_next_route": metrics["selected_next_route"],
          "target_command_allowed": False,
          "token_row_allowed": False,
      }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  failed = [row["name"] for row in metrics["checks"] if not row["pass"]]
  lines = [
      f"# Seq{metrics['sequence']} Correctly-Rounded Q/K Scale Design",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- design_passed: `{str(metrics['design_passed']).lower()}`",
      f"- candidate: `{metrics['design']['candidate']}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- whole_shell_added_us_per_layer_max: `"
      f"{metrics['design']['component_gate']['whole_shell_added_us_per_layer_max']}`",
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
  parser.add_argument("--sequence", type=int, default=620)
  parser.add_argument("--routes", type=Path,
                      default=ACTIVE / "routes-ledger.json")
  parser.add_argument(
      "--predecessor", type=Path,
      default=ROOT / (
          "output/seq619-gpu-software-correctly-rounded-qk-scale-primitive-"
          "feasibility-gate-20260710Tseq619Z/metrics.json"))
  parser.add_argument(
      "--component", type=Path,
      default=ROOT / (
          "output/seq617-gpu-level-zero-postconv-recurrent-component-probe-"
          "gate-20260710Tseq617Z/metrics.json"))
  parser.add_argument(
      "--v1-design", type=Path,
      default=ROOT / (
          "output/seq614-gpu-level-zero-postconv-recurrent-component-design-"
          "gate-20260710Tseq614Z/metrics.json"))
  parser.add_argument("--rejected", type=Path,
                      default=ACTIVE / "rejected-routes.json")
  parser.add_argument("--module-source", type=Path,
                      default=ROOT / "engine/gpu/level_zero/iq36_postconv_recurrent.cl")
  parser.add_argument(
      "--out-dir", type=Path,
      default=ROOT / (
          "output/seq620-gpu-software-correctly-rounded-qk-scale-primitive-"
          "design-gate-20260710Tseq620Z"))
  args = parser.parse_args()
  metrics = compute(args)
  write_outputs(metrics, args.out_dir)
  print(json.dumps({
      "required_checks_passed": metrics["required_checks_passed"],
      "design_passed": metrics["design_passed"],
      "source_gate_allowed": metrics["source_gate_allowed"],
      "disposition": metrics["disposition"],
      "selected_next_route": metrics["selected_next_route"],
      "out_dir": _rel(args.out_dir),
  }, sort_keys=True))
  return 0 if metrics["required_checks_passed"] else 2


if __name__ == "__main__":
  raise SystemExit(main())
