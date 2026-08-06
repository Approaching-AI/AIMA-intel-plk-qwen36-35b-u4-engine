#!/usr/bin/env python3
"""Lock one offline-ocloc native Level Zero component design."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = (
    "intel-qwen36-gpu-level-zero-postconv-recurrent-component-design-v0")
CURRENT_ROUTE = "gpu_level_zero_postconv_recurrent_component_design_gate"
SELECTED_NEXT_ROUTE = "gpu_level_zero_postconv_recurrent_component_source_gate"


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
  opencl_probe = _load(args.opencl_probe)
  vulkan_probe = _load(args.vulkan_probe)
  old_design = _load(args.old_design)
  cpu_source = args.cpu_source.read_text(encoding="utf-8")
  opencl_source = args.opencl_source.read_text(encoding="utf-8")
  predecessor_selects = (
      predecessor.get("required_checks_passed") is True
      and predecessor.get("level_zero_preflight_passed") is True
      and predecessor.get("component_design_allowed") is True
      and predecessor.get("component_source_allowed") is False
      and predecessor.get("selected_next_route") == CURRENT_ROUTE
      and predecessor.get("smoke", {}).get("device_id") == 0xB080
      and _has_candidate(routes, 613, CURRENT_ROUTE)
      and _has_switch(
          routes, 613,
          "select_gpu_level_zero_postconv_recurrent_component_design_gate"))
  cpu_contract_ok = all(marker in cpu_source for marker in [
      "result.conv_output_silu.push_back(value * sigmoid_scalar(value));",
      "double sum = 0.0;",
      "sum += static_cast<double>(value) * static_cast<double>(value);",
      "state_head[i] *= decay;",
      "sum += state_row[col] * k_head[col];",
      "state_row[col] += k_head[col] * delta[row];",
      "sum += state_row[col] * q_head[col];",
      "sum_squares += output_head[i] * output_head[i];",
  ])
  source_basis_ok = all(marker in opencl_source for marker in [
      "#pragma OPENCL FP_CONTRACT OFF",
      "linear_attn_postconv_silu_split_cpuorder_f32",
      "linear_attn_postconv_qk_l2_cpuorder_f32",
      "linear_attn_delta_recurrent_final_cpuorder_nofma_f32",
      "const double sigmoid_double",
      "double sum = 0.0;",
  ])
  old_rows = opencl_probe.get("rows", [])
  old_opencl_attribution_ok = (
      len(old_rows) == 2
      and all(row.get("budget_passed") is True for row in old_rows)
      and all(
          row.get("exact_comparisons", {}).get("exact_qkv_vs_cpu") is True
          and row.get("exact_comparisons", {}).get(
              "exact_conv_output_vs_cpu") is True
          and row.get("exact_comparisons", {}).get(
              "exact_recurrent_state_vs_cpu") is False
          for row in old_rows))
  vulkan_rows = vulkan_probe.get("rows", [])
  same_device_baseline_ok = (
      len(vulkan_rows) == 2
      and all(row.get("device_opencl") == "Intel(R) Arc(TM) B390 GPU"
              for row in vulkan_rows)
      and predecessor.get("smoke", {}).get("device_name")
      == "Intel(R) Arc(TM) B390 GPU")
  added_us_max = old_design.get("design", {}).get(
      "component_gate", {}).get("whole_shell_added_us_per_layer_max")
  kill_number_ok = (
      isinstance(added_us_max, (int, float))
      and abs(added_us_max - 6.841858993929781) < 1.0e-12)

  design = {
      "schema_version": (
          "intel-qwen36-level-zero-ocloc-fused-postconv-recurrent-v0"),
      "candidate": "level_zero_ocloc_fused_postconv_recurrent_v1",
      "purpose": (
          "Test whether an offline-compiled native Level Zero module on the "
          "same 0xb080 device can reproduce the CPU postconv/recurrent boundary "
          "exactly within the existing floor budget."),
      "runtime_ownership": {
          "allowed_runtime_dependencies": ["libze_loader.so.1"],
          "forbidden_runtime_dependencies": [
              "OpenCL", "llama.cpp", "OpenVINO", "ocloc"],
          "build_toolchain": (
              "target /usr/bin/ocloc, exact device 0xb080, --format zebin"),
          "source_controlled_opencl_c_required": True,
          "native_module_identity_manifest_required": True,
          "native_module_shipped_without_runtime_compiler": True,
          "resident_level_zero_allocations_required": True,
          "component_probe_uploads_outside_timed_region": True,
          "opencl_level_zero_host_bridge_allowed_for_integration": False,
          "integration_after_component_pass_requires_contiguous_level_zero_island_or_external_memory_contract": True,
      },
      "shape": {
          "device_id": "0xb080",
          "head_dim": 128,
          "query_heads": 16,
          "value_heads": 32,
          "conv_output_values": 8192,
          "q_values": 2048,
          "v_values": 4096,
          "recurrent_state_values": 524288,
      },
      "native_module": {
          "kernel_count": 2,
          "kernels": [
              {
                  "name": "iq36_l0_postconv_cpuorder",
                  "group_count": [64, 1, 1],
                  "group_size": [128, 1, 1],
                  "rule": (
                      "Fuse CPU double-sigmoid/cast-to-float SiLU split and "
                      "serial double Q/K L2 reduction into one kernel."),
              },
              {
                  "name": "iq36_l0_delta_recurrent_cpuorder",
                  "group_count": [32, 1, 1],
                  "group_size": [128, 1, 1],
                  "rule": (
                      "Preserve decay/store, serial K dot, delta, serial "
                      "update/store, serial Q dot, and lane-0 serial RMS."),
              },
          ],
          "compiler_options": [
              "-cl-std=CL2.0", "--format zebin", "-device 0xb080"],
          "fp_contract_scope": "OFF around both kernels",
      },
      "arithmetic_contract": {
          "host_precomputed_existing_inputs": ["decay", "z_silu"],
          "double_sigmoid_and_double_l2_required": True,
          "serial_float_recurrent_reductions_required": True,
          "cpu_phase_stores_required": [
              "sigmoid_double_to_float",
              "l2_double_to_float_before_sqrt",
              "decayed_state",
              "updated_state",
              "attention",
              "rms_scale",
              "final_left_associative_multiplies",
          ],
          "fma_builtin_forbidden": True,
      },
      "component_gate": {
          "payload": "captured layer0 token15 exact conv output and recurrent seed",
          "fresh_state_clone_per_sample": True,
          "samples_per_row": 11,
          "repeat_and_confirm": True,
          "same_physical_device_id": "0xb080",
          "bit_exact_boundaries": [
              "q_conv_predelta", "k_conv_predelta", "v_conv_predelta",
              "attention_output", "recurrent_state", "final_output"],
          "same_host_wall_timing_for_current_opencl_and_candidate_level_zero": True,
          "uploads_module_kernel_creation_and_state_clone_outside_timed_region": True,
          "whole_shell_added_us_per_layer_max": added_us_max,
          "speed_claim_allowed": False,
      },
      "stop_condition": (
          "After one source implementation and one fresh target compile, run "
          "one paired repeat/confirm. Any non-exact boundary, runtime failure, "
          "or row above the kill-number closes the Level Zero component class; "
          "do not sweep compiler flags, workgroups, or arithmetic order."),
      "route_guards": {
          "source_only_next": True,
          "target_compile_before_source_gate": False,
          "kernel_creation_or_execution_before_target_compile": False,
          "model_access_before_component_probe": False,
          "decode_or_token_before_component_pass": False,
          "integration_before_no_bridge_contract": False,
      },
  }
  checks = [
      {"name": "seq613_selected_level_zero_component_design_only",
       "pass": predecessor_selects},
      {"name": "cpu_postconv_recurrent_operation_order_is_explicit",
       "pass": cpu_contract_ok},
      {"name": "existing_exact_opencl_source_is_reference_not_runtime_dependency",
       "pass": source_basis_ok},
      {"name": "prior_opencl_failure_is_postconv_recurrent_and_budget_positive",
       "pass": old_opencl_attribution_ok},
      {"name": "opencl_baseline_and_level_zero_preflight_are_same_device",
       "pass": same_device_baseline_ok},
      {"name": "floor_derived_whole_shell_kill_number_reused",
       "pass": kill_number_ok,
       "detail": {"whole_shell_added_us_per_layer_max": added_us_max}},
      {"name": "one_two_kernel_native_module_and_no_bridge_are_locked",
       "pass": (
           design["native_module"]["kernel_count"] == 2
           and design["runtime_ownership"][
               "opencl_level_zero_host_bridge_allowed_for_integration"]
           is False)},
  ]
  required = all(bool(row["pass"]) for row in checks)
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "sequence": args.sequence,
      "inputs": {
          "routes": _rel(args.routes),
          "predecessor": _rel(args.predecessor),
          "opencl_probe": _rel(args.opencl_probe),
          "vulkan_probe": _rel(args.vulkan_probe),
          "old_design": _rel(args.old_design),
          "cpu_source": _rel(args.cpu_source),
          "opencl_source": _rel(args.opencl_source),
      },
      "design": design,
      "checks": checks,
      "required_checks_passed": required,
      "component_design_passed": required,
      "component_source_allowed": required,
      "target_compile_allowed": False,
      "component_probe_allowed": False,
      "decode_integration_allowed": False,
      "token_row_allowed": False,
      "validation_or_test_allowed": False,
      "speedup_claims_allowed": False,
      "promotion_allowed": False,
      "disposition": (
          "accept_level_zero_ocloc_fused_postconv_recurrent_v1_design"
          if required else
          "reject_level_zero_ocloc_fused_postconv_recurrent_v1_design"),
      "selected_next_route": (
          SELECTED_NEXT_ROUTE if required else CURRENT_ROUTE),
      "next_route_reason": (
          "Add only the standalone Level Zero runtime shell and one source-"
          "controlled two-kernel OpenCL C module; compile/audit source without "
          "creating a kernel or accessing the model."
          if required else
          "Repair device identity, source attribution, component shape, or "
          "kill-number before any Level Zero source."),
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
          "component_design_passed": metrics["component_design_passed"],
          "candidate": metrics["design"]["candidate"],
          "selected_next_route": metrics["selected_next_route"],
          "target_compile_allowed": False,
          "token_row_allowed": False,
      }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  failed = [row["name"] for row in metrics["checks"] if not row["pass"]]
  lines = [
      f"# Seq{metrics['sequence']} Native Level Zero Component Design",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- candidate: `{metrics['design']['candidate']}`",
      f"- kernel count: `{metrics['design']['native_module']['kernel_count']}`",
      f"- whole-shell added ruler: `{metrics['design']['component_gate']['whole_shell_added_us_per_layer_max']} us/layer`",
      f"- failed_checks: `{failed}`",
      "",
      metrics["next_route_reason"],
      "",
      "This design gate used existing evidence only; no target command ran.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--sequence", type=int, default=614)
  parser.add_argument("--routes", type=Path,
                      default=ACTIVE / "routes-ledger.json")
  parser.add_argument(
      "--predecessor", type=Path,
      default=ROOT / (
          "output/seq613-gpu-level-zero-postconv-recurrent-component-"
          "preflight-gate-20260710Tseq613Z/metrics.json"))
  parser.add_argument(
      "--opencl-probe", type=Path,
      default=ROOT / (
          "output/seq604-all-linear-preprojection-parity-component-final-"
          "probe-gate-20260710Tseq604Z/metrics.json"))
  parser.add_argument(
      "--vulkan-probe", type=Path,
      default=ROOT / (
          "output/seq611-gpu-vulkan-postconv-recurrent-component-probe-gate-"
          "20260710Tseq611Z/metrics.json"))
  parser.add_argument(
      "--old-design", type=Path,
      default=ROOT / (
          "output/seq608-gpu-vulkan-postconv-recurrent-component-design-gate-"
          "20260710Tseq608Z/metrics.json"))
  parser.add_argument("--cpu-source", type=Path,
                      default=ROOT / "engine/src/gguf_loader.cpp")
  parser.add_argument("--opencl-source", type=Path,
                      default=ROOT / "engine/gpu/opencl/q4x8_matvec.cl")
  parser.add_argument(
      "--out-dir", type=Path,
      default=ROOT / (
          "output/seq614-gpu-level-zero-postconv-recurrent-component-design-"
          "gate-20260710Tseq614Z"))
  args = parser.parse_args()
  metrics = compute(args)
  write_outputs(metrics, args.out_dir)
  print(json.dumps({
      "required_checks_passed": metrics["required_checks_passed"],
      "component_design_passed": metrics["component_design_passed"],
      "disposition": metrics["disposition"],
      "selected_next_route": metrics["selected_next_route"],
      "out_dir": _rel(args.out_dir),
  }, sort_keys=True))
  return 0 if metrics["required_checks_passed"] else 2


if __name__ == "__main__":
  raise SystemExit(main())
