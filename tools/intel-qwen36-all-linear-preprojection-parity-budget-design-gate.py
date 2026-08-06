#!/usr/bin/env python3
"""Lock one exact linear preprojection/recurrent component design."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = (
    "intel-qwen36-all-linear-preprojection-parity-budget-design-gate-v0"
)
DESIGN_SCHEMA_VERSION = (
    "intel-qwen36-exact-linear-preprojection-component-design-v0"
)
CURRENT_ROUTE = (
    "router_prompt_distribution_all_linear_preprojection_parity_budget_design_gate"
)
SELECTED_NEXT_ROUTE = (
    "router_prompt_distribution_all_linear_preprojection_parity_component_source_gate"
)
REFLECTION_ROUTE = "router_prompt_distribution_correctness_route_reflection_gate"
CANDIDATE = "cpuorder_preprojection_bundle_v1"


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


def _contains_all(source: str, markers: list[str]) -> bool:
  return all(marker in source for marker in markers)


def compute(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
  routes = _load(args.routes)
  predecessor = _load(args.predecessor)
  rejected = _load(args.rejected_routes)
  opencl = args.opencl_source.read_text(encoding="utf-8")
  cpu = args.cpu_source.read_text(encoding="utf-8")
  generator = args.generator_source.read_text(encoding="utf-8")
  current_generated = args.current_generated_source.read_text(encoding="utf-8")
  old_nofma = args.old_nofma_source.read_text(encoding="utf-8")

  prior_design = predecessor.get("design", {})
  prior_budget = prior_design.get("budget", {})
  component_contract = prior_design.get("component_contract", {})
  whole_shell_added_max = float(
      prior_budget.get("floor_headroom_us_per_linear_layer", 0.0))
  preprojection_added_max = float(
      prior_budget.get("maximum_preprojection_added_us_per_layer", 0.0))

  route_selects = (
      predecessor.get("required_checks_passed") is True
      and predecessor.get("component_design_allowed") is True
      and predecessor.get("component_probe_allowed") is False
      and predecessor.get("selected_next_route") == CURRENT_ROUTE
      and _has_candidate(routes, 596, CURRENT_ROUTE)
      and _has_switch(
          routes, 596,
          "select_router_prompt_distribution_all_linear_preprojection_"
          "parity_budget_design_gate"))
  budget_preserved = (
      abs(whole_shell_added_max - 6.841858993929781) <= 1.0e-12
      and abs(preprojection_added_max - 36.216858993929795) <= 1.0e-12
      and component_contract.get("final_output_and_recurrent_state_bit_exact_required")
      is True
      and component_contract.get("host_bridge_or_cpu_fallback_allowed") is False)

  current_source_shape = {
      "q6_lane_sums_already_present": _contains_all(opencl, [
          "__kernel void q6k_selected_down_matvec_row(",
          "int lane_sums[8];",
          "sum += combined_scale * (float)lane_sums[lane];",
      ]),
      "q6_remains_after_scoped_contract_restore": (
          opencl.find("#pragma OPENCL FP_CONTRACT OFF") >= 0
          and opencl.find("#pragma OPENCL FP_CONTRACT ON")
          > opencl.find("#pragma OPENCL FP_CONTRACT OFF")
          and opencl.find("__kernel void q6k_selected_down_matvec_row(")
          > opencl.find("#pragma OPENCL FP_CONTRACT ON")),
      "conv_is_serial_float_reduction": _contains_all(opencl, [
          "__kernel void linear_attn_conv_f32(",
          "sum += conv_state[state_base + k] * weights[weight_base + k];",
      ]),
      "postconv_uses_float_sigmoid_and_float_l2": _contains_all(opencl, [
          "__kernel void linear_attn_postconv_silu_split_f32(",
          "float sigmoid_f32(float x)",
          "__kernel void linear_attn_postconv_fused_qk_l2_f32(",
      ]),
      "delta_kernel_combines_decay_and_update": _contains_all(opencl, [
          "__kernel void linear_attn_delta_recurrent_final_qk_local_f32(",
          "state_in[state_base + col] * decay +",
          "local_k[col] * delta",
      ]),
      "cpu_postconv_uses_double_sigmoid_and_double_l2": _contains_all(cpu, [
          "float sigmoid_scalar(float value)",
          "const double x = value;",
          "double sum = 0.0;",
          "std::sqrt(static_cast<float>(sum))",
      ]),
      "cpu_delta_has_two_rounded_state_phases": _contains_all(cpu, [
          "state_row[col] *= std::exp(gate_head[col]);",
          "state_row[col] += k_head[col] * delta[row];",
      ]) or _contains_all(cpu, [
          "state_head[i] *= decay;",
          "state_row[col] += k_head[col] * delta[row];",
      ]),
      "generated_softplus_differs_from_cpu": _contains_all(current_generated, [
          "return std::log1p(std::exp(value));",
      ]) and _contains_all(cpu, [
          "return value > 20.0f ? value : std::log(1.0f + std::exp(value));",
      ]),
      "old_nofma_artifact_has_contract_off_and_lane_sums": _contains_all(
          old_nofma, [
              "#pragma OPENCL FP_CONTRACT OFF",
              "int lane_sums[8];",
          ]),
      "fused_exact_projection_exists": _contains_all(opencl, [
          "q4k_x8_matvec_rowblock16_cpuorder_finalize",
      ]),
  }
  source_shape_passes = all(current_source_shape.values())

  rejected_rows = {
      row.get("route"): row
      for row in rejected.get("rejected", [])
      if isinstance(row, dict)
  }
  prior_nofma_proof = rejected_rows.get(
      "gpu_attention_front_handoff_8tok_opencl_q6_lane_sums_nofma_diagnostic",
      {})
  old_proof_supports_localized_reuse = (
      "made early Q6 QKV same-input diffs exact"
      in str(prior_nofma_proof.get("reason", ""))
      and "do not rerun this combination without a new downstream source mismatch"
      in str(prior_nofma_proof.get("reason", "")))

  design = {
      "schema_version": DESIGN_SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "route": SELECTED_NEXT_ROUTE,
      "candidate": CANDIDATE,
      "purpose": (
          "Make the complete linear-attention `final_output`, convolution "
          "state, and recurrent state CPU-bit-exact from exact layer input, "
          "then use the accepted fused exact output projection."
      ),
      "source_basis": current_source_shape,
      "implementation": {
          "opencl_program": _rel(args.opencl_source),
          "runner_source": "engine/src/gpu_q4x8_matvec.cpp",
          "runner_header": "engine/include/intel_qwen36/gpu_q4x8_matvec.hpp",
          "generator_source": _rel(args.generator_source),
          "current_generated_source": _rel(args.current_generated_source),
          "q6_qkv": {
              "kernel": "q6k_linear_qkv_cpuorder_nofma",
              "rule": (
                  "Duplicate only the existing lane-sum row kernel under a "
                  "scoped FP_CONTRACT OFF region; retain block-major then "
                  "lane-major float accumulation and restore contraction for "
                  "unselected kernels."
              ),
          },
          "alpha_beta_z_and_host_gate": {
              "rule": (
                  "Keep resident raw-Q4 CPU-order alpha/beta/z. Replace the "
                  "diagnostic log1p/negative shortcut with the exact CPU "
                  "softplus formula; precompute decay and z_silu in the "
                  "already-existing host vectors/uploads, adding no bridge."
              ),
          },
          "conv": {
              "kernel": "linear_attn_conv_cpuorder_nofma_f32",
              "rule": (
                  "Duplicate the serial four-tap loop under scoped "
                  "FP_CONTRACT OFF and preserve a float store after every "
                  "multiply/add plus the existing resident state update."
              ),
          },
          "postconv": {
              "kernels": [
                  "linear_attn_postconv_silu_split_cpuorder_f32",
                  "linear_attn_postconv_qk_l2_cpuorder_f32",
              ],
              "rule": (
                  "Match CPU sigmoid_scalar double intermediates, accumulate "
                  "each 128-value L2 sum in double, cast to float before "
                  "sqrt/max, and preserve the CPU multiply/store order."
              ),
          },
          "delta_and_final_gate": {
              "kernel": "linear_attn_delta_recurrent_final_cpuorder_nofma_f32",
              "rule": (
                  "Consume host-precomputed decay and z_silu; preserve CPU "
                  "decay-store, delta, state-update, q-dot, float RMS sum/sqrt, "
                  "and final multiply phases with contraction disabled."
              ),
          },
          "output_projection": {
              "kernel": "q4k_x8_matvec_rowblock16_cpuorder_finalize",
              "rule": "Reuse unchanged from seq589; no second projection path.",
          },
          "dispatch_contract": {
              "replacement_kernels_only": True,
              "new_dispatches_allowed": 0,
              "new_host_readbacks_allowed": 0,
              "new_host_uploads_allowed": 0,
              "same_context_and_resident_handles_required": True,
          },
      },
      "component_gate": {
          "payload": "captured layer0 exact-input plus live convolution/recurrent state",
          "repeat_and_confirm": True,
          "qkv_conv_postconv_final_output_bit_exact": True,
          "conv_state_bit_exact": True,
          "recurrent_state_bit_exact": True,
          "whole_linear_attention_output_bit_exact": True,
          "baseline": "current preprojection/recurrent plus rowblock16 projection",
          "candidate_shell": (
              "cpuorder_preprojection_bundle_v1 plus existing fused exact "
              "projection"
          ),
          "whole_shell_added_us_per_layer_max": whole_shell_added_max,
          "preprojection_added_us_per_layer_diagnostic_max":
              preprojection_added_max,
          "both_repeat_and_confirm_must_pass": True,
      },
      "route_guards": {
          "source_only_next": True,
          "target_compile_allowed_before_source_pass": False,
          "component_probe_allowed_before_target_compile": False,
          "token_row_allowed_before_component_pass": False,
          "per_axis_probe_allowed": False,
          "layer_subset_sweep_allowed": False,
          "router_code_or_speed_allowed": False,
      },
      "claim_policy": {
          "correctness_claim_allowed": False,
          "speedup_claim_allowed": False,
          "promotion_allowed": False,
      },
  }
  design_is_single_and_bounded = (
      design["candidate"] == CANDIDATE
      and design["implementation"]["dispatch_contract"]["new_dispatches_allowed"]
      == 0
      and design["implementation"]["dispatch_contract"]["new_host_readbacks_allowed"]
      == 0
      and design["component_gate"]["whole_shell_added_us_per_layer_max"]
      == whole_shell_added_max
      and design["route_guards"]["per_axis_probe_allowed"] is False
      and design["route_guards"]["token_row_allowed_before_component_pass"]
      is False)
  candidate_names_unique = len(set(re.findall(
      r'"kernel": "([^"]+)"', json.dumps(design)))) >= 4

  checks = [
      {"name": "seq596_selected_whole_component_design_only",
       "pass": route_selects},
      {"name": "seq596_floor_and_exactness_contract_preserved",
       "pass": budget_preserved,
       "detail": {
           "whole_shell_added_us_per_layer_max": whole_shell_added_max,
           "preprojection_added_us_per_layer_max": preprojection_added_max,
       }},
      {"name": "current_source_and_cpu_formula_gaps_are_explicit",
       "pass": source_shape_passes,
       "detail": current_source_shape},
      {"name": "old_nofma_q6_exactness_is_reused_only_after_new_source",
       "pass": old_proof_supports_localized_reuse},
      {"name": "one_candidate_replaces_kernels_without_new_dispatch_or_bridge",
       "pass": design_is_single_and_bounded},
      {"name": "candidate_kernel_names_are_explicit_and_distinct",
       "pass": candidate_names_unique},
  ]
  required = all(bool(row["pass"]) for row in checks)
  selected_next = SELECTED_NEXT_ROUTE if required else REFLECTION_ROUTE
  metrics = {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "sequence": args.sequence,
      "inputs": {
          "routes": _rel(args.routes),
          "predecessor": _rel(args.predecessor),
          "rejected_routes": _rel(args.rejected_routes),
          "opencl_source": _rel(args.opencl_source),
          "cpu_source": _rel(args.cpu_source),
          "generator_source": _rel(args.generator_source),
          "old_nofma_source": _rel(args.old_nofma_source),
      },
      "checks": checks,
      "required_checks_passed": required,
      "design_passed": required,
      "component_source_allowed": required,
      "target_compile_allowed": False,
      "component_probe_allowed": False,
      "token_row_allowed": False,
      "speedup_claims_allowed": False,
      "promotion_allowed": False,
      "design": design,
      "disposition": (
          "accept_cpuorder_preprojection_bundle_v1_design_select_source"
          if required else
          "reject_whole_preprojection_design_without_source"),
      "selected_next_route": selected_next,
      "next_route_reason": (
          "One bounded design remains: scoped no-FMA/CPU-order replacement "
          "kernels plus exact existing host-vector transforms and fused exact "
          "projection. Add only component APIs and source-gate them next. The "
          "whole layer shell, not transferred microtimings, must eventually "
          "be bit-exact and add <=6.841858994 us/layer on repeat and confirm."
          if required else
          "The source proof or no-dispatch/no-bridge design is incomplete; "
          "return to correctness-route reflection without implementation."),
  }
  return metrics, design


def write_outputs(metrics: dict[str, Any], design: dict[str, Any],
                  out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  (out_dir / "design.json").write_text(
      json.dumps(design, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  failed = [row["name"] for row in metrics["checks"] if not row["pass"]]
  lines = [
      f"# Seq{metrics['sequence']} Exact Preprojection Budget Design",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- candidate: `{design['candidate']}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- whole-shell added ceiling: "
      f"`{design['component_gate']['whole_shell_added_us_per_layer_max']} us/layer`",
      f"- preprojection diagnostic ceiling: "
      f"`{design['component_gate']['preprojection_added_us_per_layer_diagnostic_max']} us/layer`",
      f"- failed_checks: `{failed}`",
      "",
      metrics["next_route_reason"],
      "",
      "No source, component, target command, or token was changed or run.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--sequence", type=int, default=597)
  parser.add_argument("--routes", type=Path,
                      default=ACTIVE / "routes-ledger.json")
  parser.add_argument(
      "--predecessor", type=Path,
      default=ROOT / "output/seq596-layer12-ffn-input-source-gate-20260710Tseq596Z/metrics.json")
  parser.add_argument("--rejected-routes", type=Path,
                      default=ACTIVE / "rejected-routes.json")
  parser.add_argument("--opencl-source", type=Path,
                      default=ROOT / "engine/gpu/opencl/q4x8_matvec.cl")
  parser.add_argument("--cpu-source", type=Path,
                      default=ROOT / "engine/src/gguf_loader.cpp")
  parser.add_argument("--generator-source", type=Path,
                      default=ROOT / "tools/intel-qwen36-r2-gpu-decode-smoke.py")
  parser.add_argument(
      "--current-generated-source", type=Path,
      default=ROOT / "output/seq590-fused-exact-linear-projection-decode-source-20260710Tseq590Z/r2_gpu_decode_smoke.cpp")
  parser.add_argument(
      "--old-nofma-source", type=Path,
      default=ROOT / "output/r2-gpu-8tok-opencl-q6-lane-sums-nofma-generate-20260701T092000Z/r2_gpu_decode_smoke.cpp")
  parser.add_argument(
      "--out-dir", type=Path,
      default=ROOT / "output/seq597-all-linear-preprojection-parity-budget-design-gate-20260710Tseq597Z")
  args = parser.parse_args()
  metrics, design = compute(args)
  write_outputs(metrics, design, args.out_dir)
  print(json.dumps({
      "required_checks_passed": metrics["required_checks_passed"],
      "candidate": design["candidate"],
      "disposition": metrics["disposition"],
      "selected_next_route": metrics["selected_next_route"],
      "whole_shell_added_us_per_layer_max": design["component_gate"][
          "whole_shell_added_us_per_layer_max"],
      "out_dir": _rel(args.out_dir),
  }, sort_keys=True))
  return 0 if metrics["required_checks_passed"] else 2


if __name__ == "__main__":
  raise SystemExit(main())
