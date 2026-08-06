#!/usr/bin/env python3
"""Lock one native Vulkan postconv/recurrent component design."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import shlex
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
  sys.path.insert(0, str(TOOLS))

import iq36_local  # noqa: E402


SCHEMA_VERSION = (
    "intel-qwen36-gpu-vulkan-postconv-recurrent-component-design-v0")
CURRENT_ROUTE = "gpu_vulkan_postconv_recurrent_component_design_gate"
SELECTED_NEXT_ROUTE = (
    "gpu_vulkan_postconv_recurrent_component_source_gate")
DEFAULT_HOST = "local"
DEFAULT_ENV_SCRIPT = (
    "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh")
DEFAULT_REMOTE_ROOT = "/home/intel/intel-qwen36-gpu"


FEATURE_CPP = r'''
#include <vulkan/vulkan.h>

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

[[noreturn]] void Die(const std::string& message) {
  throw std::runtime_error(message);
}

void Check(VkResult result, const char* where) {
  if (result != VK_SUCCESS) {
    Die(std::string(where) + " failed with VkResult " +
        std::to_string(result));
  }
}

const char* Bool(VkBool32 value) { return value ? "true" : "false"; }

int main() {
  try {
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "iq36-vulkan-float-controls";
    app.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo create{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    create.pApplicationInfo = &app;
    VkInstance instance = VK_NULL_HANDLE;
    Check(vkCreateInstance(&create, nullptr, &instance), "vkCreateInstance");

    std::uint32_t count = 0;
    Check(vkEnumeratePhysicalDevices(instance, &count, nullptr),
          "vkEnumeratePhysicalDevices(count)");
    std::vector<VkPhysicalDevice> devices(count);
    Check(vkEnumeratePhysicalDevices(instance, &count, devices.data()),
          "vkEnumeratePhysicalDevices(list)");

    VkPhysicalDevice selected = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties properties{};
    for (VkPhysicalDevice device : devices) {
      VkPhysicalDeviceProperties candidate{};
      vkGetPhysicalDeviceProperties(device, &candidate);
      if (candidate.vendorID == 0x8086U) {
        selected = device;
        properties = candidate;
        break;
      }
    }
    if (selected == VK_NULL_HANDLE) Die("no Intel Vulkan device");

    VkPhysicalDeviceFeatures features{};
    vkGetPhysicalDeviceFeatures(selected, &features);
    VkPhysicalDeviceFloatControlsProperties controls{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT_CONTROLS_PROPERTIES};
    VkPhysicalDeviceProperties2 properties2{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    properties2.pNext = &controls;
    vkGetPhysicalDeviceProperties2(selected, &properties2);

    std::cout << "{";
    std::cout << "\"schema_version\":\"iq36-vulkan-float-controls-v0\",";
    std::cout << "\"device_name\":\"" << properties.deviceName << "\",";
    std::cout << "\"vendor_id\":" << properties.vendorID << ",";
    std::cout << "\"device_id\":" << properties.deviceID << ",";
    std::cout << "\"shader_float64\":" << Bool(features.shaderFloat64) << ",";
    std::cout << "\"denorm_behavior_independence\":"
              << static_cast<int>(controls.denormBehaviorIndependence) << ",";
    std::cout << "\"rounding_mode_independence\":"
              << static_cast<int>(controls.roundingModeIndependence) << ",";
    std::cout << "\"preserve_f32\":"
              << Bool(controls.shaderSignedZeroInfNanPreserveFloat32) << ",";
    std::cout << "\"denorm_preserve_f32\":"
              << Bool(controls.shaderDenormPreserveFloat32) << ",";
    std::cout << "\"rte_f32\":"
              << Bool(controls.shaderRoundingModeRTEFloat32) << ",";
    std::cout << "\"preserve_f64\":"
              << Bool(controls.shaderSignedZeroInfNanPreserveFloat64) << ",";
    std::cout << "\"denorm_preserve_f64\":"
              << Bool(controls.shaderDenormPreserveFloat64) << ",";
    std::cout << "\"rte_f64\":"
              << Bool(controls.shaderRoundingModeRTEFloat64);
    std::cout << "}" << std::endl;
    vkDestroyInstance(instance, nullptr);
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "Vulkan float-control query error: " << ex.what() << std::endl;
    return 1;
  }
}
'''


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


def _parse_json_line(stdout: str) -> dict[str, Any] | None:
  for line in reversed(stdout.splitlines()):
    line = line.strip()
    if not line.startswith("{"):
      continue
    try:
      value = json.loads(line)
    except json.JSONDecodeError:
      continue
    if isinstance(value, dict):
      return value
  return None


def _raw_probe(path: Path) -> dict[str, Any]:
  wrapper = _load(path)
  parsed = _parse_json_line(str(wrapper.get("stdout", "")))
  if parsed is None:
    raise ValueError(f"no probe JSON in {path}")
  return parsed


def compute(args: argparse.Namespace) -> dict[str, Any]:
  routes = _load(args.routes)
  predecessor = _load(args.predecessor)
  old_design = _load(args.old_design)
  repeat = _raw_probe(args.repeat_probe)
  confirm = _raw_probe(args.confirm_probe)
  cpu_source = args.cpu_source.read_text(encoding="utf-8")
  opencl_source = args.opencl_source.read_text(encoding="utf-8")
  args.out_dir.mkdir(parents=True, exist_ok=True)
  raw_dir = args.out_dir / "raw"
  raw_dir.mkdir(parents=True, exist_ok=True)
  local_cpp = args.out_dir / "vulkan_float_controls.cpp"
  local_cpp.write_text(FEATURE_CPP, encoding="utf-8")
  stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
  remote_dir = (
      f"{args.remote_root.rstrip('/')}/seq{args.sequence}-vulkan-design-{stamp}")

  predecessor_selects = (
      predecessor.get("required_checks_passed") is True
      and predecessor.get("component_design_allowed") is True
      and predecessor.get("component_source_allowed") is False
      and predecessor.get("selected_next_route") == CURRENT_ROUTE
      and _has_candidate(routes, 607, CURRENT_ROUTE)
      and _has_switch(
          routes, 607,
          "select_gpu_vulkan_postconv_recurrent_component_design_gate"))

  setup = iq36_local.run_target(
      args.host, "mkdir -p " + shlex.quote(remote_dir), args.timeout_s)
  transfer = (
      iq36_local.copy_to(
          args.host, local_cpp, f"{remote_dir}/vulkan_float_controls.cpp",
          args.timeout_s)
      if setup.get("returncode") == 0 else {})
  build_command = " && ".join([
      f"source {shlex.quote(args.env_script)} >/dev/null 2>&1",
      (
          "g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic "
          f"{shlex.quote(remote_dir + '/vulkan_float_controls.cpp')} -lvulkan "
          f"-o {shlex.quote(remote_dir + '/vulkan-float-controls')}")
  ])
  build = (
      iq36_local.run_target(
          args.host, f"bash -lc {shlex.quote(build_command)}", args.timeout_s)
      if transfer.get("returncode") == 0 else {})
  query_command = " && ".join([
      f"source {shlex.quote(args.env_script)} >/dev/null 2>&1",
      "export INTEL_FORCE_PROBE=b080",
      shlex.quote(remote_dir + "/vulkan-float-controls"),
  ])
  query = (
      iq36_local.run_target(
          args.host, f"bash -lc {shlex.quote(query_command)}", args.timeout_s)
      if build.get("returncode") == 0 else {})
  features = _parse_json_line(str(query.get("stdout", "")))
  cleanup = iq36_local.run_target(
      args.host, "rm -rf " + shlex.quote(remote_dir), args.timeout_s)
  iq36_local.write_json(raw_dir / "setup.json", setup)
  iq36_local.write_json(raw_dir / "transfer.json", transfer)
  iq36_local.write_json(raw_dir / "build.json", build)
  iq36_local.write_json(raw_dir / "query.json", query)
  iq36_local.write_json(raw_dir / "cleanup.json", cleanup)

  float_controls_ok = (
      isinstance(features, dict)
      and features.get("vendor_id") == 0x8086
      and features.get("shader_float64") is True
      and features.get("preserve_f32") is True
      and features.get("denorm_preserve_f32") is True
      and features.get("rte_f32") is True
      and features.get("preserve_f64") is True
      and features.get("denorm_preserve_f64") is True
      and features.get("rte_f64") is True)
  cpu_contract_ok = all(marker in cpu_source for marker in [
      "double sum = 0.0;",
      "sum += static_cast<double>(value) * static_cast<double>(value);",
      "state_head[i] *= decay;",
      "sum += state_row[col] * k_head[col];",
      "state_row[col] += k_head[col] * delta[row];",
      "sum += state_row[col] * q_head[col];",
      "sum_squares += output_head[i] * output_head[i];",
      "z_value * sigmoid_scalar(z_value)",
  ])
  old_exact_contract_ok = all(marker in opencl_source for marker in [
      "linear_attn_postconv_silu_split_cpuorder_f32",
      "linear_attn_postconv_qk_l2_cpuorder_f32",
      "linear_attn_delta_recurrent_final_cpuorder_nofma_f32",
      "#pragma OPENCL FP_CONTRACT OFF",
  ])
  prior_component_failed_only_postconv = all(
      row.get("comparisons", {}).get("exact_qkv_vs_cpu", {}).get(
          "mismatch_count") == 0
      and row.get("comparisons", {}).get("exact_conv_output_vs_cpu", {}).get(
          "mismatch_count") == 0
      and row.get("comparisons", {}).get("exact_conv_state_vs_cpu", {}).get(
          "mismatch_count") == 0
      and row.get("comparisons", {}).get(
          "exact_recurrent_state_vs_cpu", {}).get("mismatch_count", 0) > 0
      for row in (repeat, confirm))
  old_component_gate = old_design.get("design", {}).get("component_gate", {})
  added_us_max = old_component_gate.get("whole_shell_added_us_per_layer_max")
  kill_number_ok = (
      isinstance(added_us_max, (int, float)) and 0.0 < added_us_max < 10.0)

  design = {
      "schema_version": "intel-qwen36-vulkan-precise-postconv-recurrent-v0",
      "candidate": "vulkan_precise_postconv_recurrent_v1",
      "purpose": (
          "Replace only the captured postconv/recurrent arithmetic with a "
          "self-owned Vulkan component and test whether Vulkan float controls "
          "can reproduce the CPU boundary exactly within the floor budget."),
      "runtime_ownership": {
          "allowed_dependencies": ["libvulkan.so.1"],
          "forbidden_runtime_dependencies": ["llama.cpp", "OpenVINO"],
          "private_build_only_shader_toolchain": (
              "checksum-pinned glslc/libshaderc1 packages from seq607"),
          "source_controlled_glsl_required": True,
          "generated_spirv_identity_manifest_required": True,
          "generated_spirv_shipped_without_runtime_compiler": True,
          "component_probe_uploads_outside_timed_region": True,
          "resident_vulkan_buffers_required": True,
          "opencl_vulkan_host_bridge_allowed_for_integration": False,
          "integration_after_component_pass_requires_contiguous_vulkan_island_or_external_memory_contract": True,
      },
      "shape": {
          "head_dim": 128,
          "query_heads": 16,
          "value_heads": 32,
          "conv_output_values": 8192,
          "q_values": 2048,
          "v_values": 4096,
          "recurrent_state_values": 524288,
      },
      "dispatches": [
          {
              "name": "iq36_postconv_cpuorder",
              "count": 1,
              "workgroups": 64,
              "local_size": 128,
              "rule": (
                  "Compute CPU double sigmoid/cast-to-float SiLU for Q/K/V; "
                  "for Q/K, lane 0 performs the 128-value double L2 reduction "
                  "serially before float sqrt/max/scale."),
          },
          {
              "name": "iq36_delta_recurrent_cpuorder",
              "count": 1,
              "workgroups": 32,
              "local_size": 128,
              "rule": (
                  "One invocation owns one state row and performs decay/store, "
                  "serial K dot, delta, serial update/store, serial Q dot, then "
                  "lane 0 performs the 128-value RMS sum in CPU order."),
          },
      ],
      "arithmetic_contract": {
          "host_precomputed_existing_inputs": ["decay", "z_silu"],
          "glsl_precise_on_every_float_or_double_mul_add_intermediate": True,
          "spirv_no_contraction_decorations_required": True,
          "spirv_op_fma_forbidden": True,
          "spirv_float32_and_float64_denorm_preserve_required": True,
          "spirv_float32_and_float64_rte_required": True,
          "cpu_phase_stores_preserved": [
              "sigmoid_double_to_float",
              "l2_double_to_float_before_sqrt",
              "decayed_state",
              "updated_state",
              "attention",
              "rms_scale",
              "final_left_associative_multiplies",
          ],
      },
      "component_gate": {
          "payload": "captured layer0 token15 exact conv output and recurrent seed",
          "fresh_state_clone_per_sample": True,
          "samples_per_row": 11,
          "repeat_and_confirm": True,
          "bit_exact_boundaries": [
              "q_conv_predelta", "k_conv_predelta", "v_conv_predelta",
              "attention_output", "recurrent_state", "final_output"],
          "same_host_wall_timing_for_current_opencl_and_candidate_vulkan": True,
          "uploads_pipeline_creation_and_state_clone_outside_timed_region": True,
          "whole_shell_added_us_per_layer_max": added_us_max,
          "speed_claim_allowed": False,
      },
      "stop_condition": (
          "After one source implementation and one fresh target compile, run "
          "one paired repeat/confirm. If SPIR-V contract audit passes but any "
          "boundary remains non-exact or either paired row exceeds the kill-"
          "number, close this candidate without arithmetic/order variants."),
      "route_guards": {
          "source_only_next": True,
          "target_compile_before_source_gate": False,
          "component_execution_before_target_compile": False,
          "model_access_before_component_probe": False,
          "decode_or_token_before_component_pass": False,
          "integration_before_no_bridge_contract": False,
      },
  }
  checks = [
      {"name": "seq607_selected_vulkan_component_design_only",
       "pass": predecessor_selects},
      {"name": "target_float_controls_support_locked_cpuorder_design",
       "pass": query.get("returncode") == 0 and float_controls_ok,
       "detail": features},
      {"name": "cpu_postconv_recurrent_operation_order_is_explicit",
       "pass": cpu_contract_ok},
      {"name": "prior_opencl_candidate_failed_only_after_exact_conv_boundary",
       "pass": prior_component_failed_only_postconv},
      {"name": "old_exact_source_names_and_no_contraction_scope_exist",
       "pass": old_exact_contract_ok},
      {"name": "floor_derived_whole_shell_kill_number_reused",
       "pass": kill_number_ok,
       "detail": {"whole_shell_added_us_per_layer_max": added_us_max}},
      {"name": "one_candidate_has_two_dispatches_and_no_integration_bridge",
       "pass": (
           len(design["dispatches"]) == 2
           and design["runtime_ownership"][
               "opencl_vulkan_host_bridge_allowed_for_integration"] is False)},
      {"name": "feature_query_did_not_access_model_or_dispatch_shader",
       "pass": (
           "/home/intel/models" not in FEATURE_CPP
           and "vkCmdDispatch" not in FEATURE_CPP)},
      {"name": "remote_design_query_cleaned",
       "pass": cleanup.get("returncode") == 0},
  ]
  required = all(bool(row["pass"]) for row in checks)
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "sequence": args.sequence,
      "inputs": {
          "routes": _rel(args.routes),
          "predecessor": _rel(args.predecessor),
          "old_design": _rel(args.old_design),
          "repeat_probe": _rel(args.repeat_probe),
          "confirm_probe": _rel(args.confirm_probe),
          "cpu_source": _rel(args.cpu_source),
          "opencl_source": _rel(args.opencl_source),
          "host": args.host,
          "env_script": args.env_script,
          "generated_feature_query": _rel(local_cpp),
      },
      "float_controls": features,
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
          "accept_vulkan_precise_postconv_recurrent_v1_design"
          if required else "reject_vulkan_precise_postconv_recurrent_v1_design"),
      "selected_next_route": (
          SELECTED_NEXT_ROUTE if required else CURRENT_ROUTE),
      "next_route_reason": (
          "Add only the self-owned Vulkan ABI/runtime shell and the two named "
          "GLSL shaders, generate/audit SPIR-V, and compile locally without "
          "target execution or model access."
          if required else
          "Do not add Vulkan component source until target float controls, "
          "operation order, prior-boundary attribution, and budget all pass."),
  }


def write_outputs(metrics: dict[str, Any], out_dir: Path) -> None:
  (out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  (out_dir / "manifest.json").write_text(
      json.dumps({
          "schema_version": metrics["schema_version"],
          "workstream": metrics["workstream"],
          "tool": _rel(Path(__file__)),
          "inputs": metrics["inputs"],
          "component_design_passed": metrics["component_design_passed"],
          "selected_next_route": metrics["selected_next_route"],
          "target_compile_allowed": False,
          "token_row_allowed": False,
      }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  failed = [row["name"] for row in metrics["checks"] if not row["pass"]]
  controls = metrics.get("float_controls") or {}
  design = metrics["design"]
  lines = [
      f"# Seq{metrics['sequence']} Native Vulkan Component Design",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- candidate: `{design['candidate']}`",
      f"- device: `{controls.get('device_name')}`",
      f"- shader_float64: `{controls.get('shader_float64')}`",
      f"- whole-shell added ruler: `{design['component_gate']['whole_shell_added_us_per_layer_max']} us/layer`",
      f"- failed_checks: `{failed}`",
      "",
      metrics["next_route_reason"],
      "",
      "The target query created no shader pipeline, dispatched no component, and did not access the model.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--sequence", type=int, default=608)
  parser.add_argument("--routes", type=Path,
                      default=ACTIVE / "routes-ledger.json")
  parser.add_argument(
      "--predecessor", type=Path,
      default=ROOT / (
          "output/seq607-gpu-vulkan-postconv-recurrent-component-preflight-"
          "gate-20260710Tseq607Z/metrics.json"))
  parser.add_argument(
      "--old-design", type=Path,
      default=ROOT / (
          "output/seq597-all-linear-preprojection-parity-budget-design-gate-"
          "20260710Tseq597Z/metrics.json"))
  parser.add_argument(
      "--repeat-probe", type=Path,
      default=ROOT / (
          "output/seq604-all-linear-preprojection-parity-component-final-"
          "probe-gate-20260710Tseq604Z/raw/repeat-run.json"))
  parser.add_argument(
      "--confirm-probe", type=Path,
      default=ROOT / (
          "output/seq604-all-linear-preprojection-parity-component-final-"
          "probe-gate-20260710Tseq604Z/raw/confirm-run.json"))
  parser.add_argument("--cpu-source", type=Path,
                      default=ROOT / "engine/src/gguf_loader.cpp")
  parser.add_argument("--opencl-source", type=Path,
                      default=ROOT / "engine/gpu/opencl/q4x8_matvec.cl")
  parser.add_argument("--target", "--host", dest="host", metavar="TARGET", default=DEFAULT_HOST)
  parser.add_argument("--env-script", default=DEFAULT_ENV_SCRIPT)
  parser.add_argument("--staging-root", "--remote-root", dest="remote_root", metavar="PATH", default=DEFAULT_REMOTE_ROOT)
  parser.add_argument("--timeout-s", type=int, default=600)
  parser.add_argument(
      "--out-dir", type=Path,
      default=ROOT / (
          "output/seq608-gpu-vulkan-postconv-recurrent-component-design-gate-"
          "20260710Tseq608Z"))
  args = parser.parse_args()
  metrics = compute(args)
  write_outputs(metrics, args.out_dir)
  print(json.dumps({
      "required_checks_passed": metrics["required_checks_passed"],
      "disposition": metrics["disposition"],
      "component_design_passed": metrics["component_design_passed"],
      "selected_next_route": metrics["selected_next_route"],
      "out_dir": _rel(args.out_dir),
  }, sort_keys=True))
  return 0 if metrics["required_checks_passed"] else 2


if __name__ == "__main__":
  raise SystemExit(main())
