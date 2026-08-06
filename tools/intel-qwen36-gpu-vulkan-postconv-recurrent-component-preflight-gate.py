#!/usr/bin/env python3
"""Preflight the self-owned Vulkan component route without model access."""

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
    "intel-qwen36-gpu-vulkan-postconv-recurrent-component-preflight-v0"
)
CURRENT_ROUTE = "gpu_vulkan_postconv_recurrent_component_preflight_gate"
SELECTED_NEXT_ROUTE = "gpu_vulkan_postconv_recurrent_component_design_gate"
DEFAULT_HOST = "local"
DEFAULT_ENV_SCRIPT = "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh"
DEFAULT_REMOTE_ROOT = "/home/intel/intel-qwen36-gpu"
DEFAULT_LLAMA_SOURCE = (
    "/home/intel/intel-qwen36-r0/source/"
    "llama.cpp-7c158fbb4aec1bdc9c81d6ca0e785139f4826fae")
GLSLC_PACKAGE_URI = (
    "https://mirrors.aliyun.com/ubuntu/pool/universe/s/shaderc/"
    "glslc_2023.8-1build1_amd64.deb")
GLSLC_PACKAGE_SHA512 = (
    "c8280be46f7bbea414a30797fdace7c486831e00216caa96f88bf9335526f7542"
    "205963d9d912c5863e797b864246709a6f1f5143fb389f0475189d391e68c64")
SHADERC_PACKAGE_URI = (
    "https://mirrors.aliyun.com/ubuntu/pool/universe/s/shaderc/"
    "libshaderc1_2023.8-1build1_amd64.deb")
SHADERC_PACKAGE_SHA512 = (
    "17b67dda9e0e1a4248aee0dd8730a15b5b2f6a7842f82abc98fadb06db2d1710"
    "2a9086ea21991e5a83325cfc2f8ae31a43b2e7b3cfb8358337bc76953ef50269")


SMOKE_CPP = r'''
#include <vulkan/vulkan.h>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

[[noreturn]] void Die(const std::string& message) {
  throw std::runtime_error(message);
}

void Check(VkResult result, const char* where) {
  if (result != VK_SUCCESS) {
    Die(std::string(where) + " failed with VkResult " + std::to_string(result));
  }
}

std::string JsonEscape(const char* value) {
  std::string out;
  for (const char ch : std::string(value)) {
    if (ch == '\\' || ch == '"') out.push_back('\\');
    out.push_back(ch);
  }
  return out;
}

int main() {
  try {
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "iq36-vulkan-preflight";
    app.applicationVersion = 1;
    app.pEngineName = "intel-qwen36";
    app.engineVersion = 1;
    app.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo create{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    create.pApplicationInfo = &app;
    VkInstance instance = VK_NULL_HANDLE;
    Check(vkCreateInstance(&create, nullptr, &instance), "vkCreateInstance");

    std::uint32_t device_count = 0;
    Check(vkEnumeratePhysicalDevices(instance, &device_count, nullptr),
          "vkEnumeratePhysicalDevices(count)");
    if (device_count == 0) Die("no Vulkan physical devices");
    std::vector<VkPhysicalDevice> devices(device_count);
    Check(vkEnumeratePhysicalDevices(instance, &device_count, devices.data()),
          "vkEnumeratePhysicalDevices(list)");

    VkPhysicalDevice selected = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties selected_props{};
    std::uint32_t selected_queue = UINT32_MAX;
    for (VkPhysicalDevice device : devices) {
      VkPhysicalDeviceProperties props{};
      vkGetPhysicalDeviceProperties(device, &props);
      std::uint32_t queue_count = 0;
      vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_count, nullptr);
      std::vector<VkQueueFamilyProperties> queues(queue_count);
      vkGetPhysicalDeviceQueueFamilyProperties(
          device, &queue_count, queues.data());
      for (std::uint32_t index = 0; index < queue_count; ++index) {
        if ((queues[index].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0 &&
            props.vendorID == 0x8086U) {
          selected = device;
          selected_props = props;
          selected_queue = index;
          break;
        }
      }
      if (selected != VK_NULL_HANDLE) break;
    }
    if (selected == VK_NULL_HANDLE) Die("no Intel Vulkan compute queue");

    const float priority = 1.0f;
    VkDeviceQueueCreateInfo queue_create{
        VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queue_create.queueFamilyIndex = selected_queue;
    queue_create.queueCount = 1;
    queue_create.pQueuePriorities = &priority;
    VkDeviceCreateInfo device_create{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    device_create.queueCreateInfoCount = 1;
    device_create.pQueueCreateInfos = &queue_create;
    VkDevice logical = VK_NULL_HANDLE;
    Check(vkCreateDevice(selected, &device_create, nullptr, &logical),
          "vkCreateDevice");
    VkQueue queue = VK_NULL_HANDLE;
    vkGetDeviceQueue(logical, selected_queue, 0, &queue);
    if (queue == VK_NULL_HANDLE) Die("vkGetDeviceQueue returned null");

    std::cout << "{";
    std::cout << "\"schema_version\":\"iq36-vulkan-preflight-smoke-v0\",";
    std::cout << "\"device_count\":" << device_count << ",";
    std::cout << "\"device_name\":\"" << JsonEscape(selected_props.deviceName)
              << "\",";
    std::cout << "\"vendor_id\":" << selected_props.vendorID << ",";
    std::cout << "\"device_id\":" << selected_props.deviceID << ",";
    std::cout << "\"api_version\":" << selected_props.apiVersion << ",";
    std::cout << "\"compute_queue_family\":" << selected_queue << ",";
    std::cout << "\"instance_created\":true,";
    std::cout << "\"device_created\":true,";
    std::cout << "\"queue_acquired\":true";
    std::cout << "}" << std::endl;

    vkDeviceWaitIdle(logical);
    vkDestroyDevice(logical, nullptr);
    vkDestroyInstance(instance, nullptr);
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "Vulkan preflight error: " << ex.what() << std::endl;
    return 1;
  }
}
'''


SMOKE_GLSL = r'''#version 450
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
layout(set = 0, binding = 0) buffer Counter { uint value; } counter_buffer;
void main() {
  counter_buffer.value += 1u;
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


def _parse_key_values(stdout: str) -> dict[str, str]:
  values: dict[str, str] = {}
  for line in stdout.splitlines():
    if "=" not in line:
      continue
    key, value = line.split("=", maxsplit=1)
    values[key.strip()] = value.strip()
  return values


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


def compute(args: argparse.Namespace) -> dict[str, Any]:
  routes = _load(args.routes)
  predecessor = _load(args.predecessor)
  args.out_dir.mkdir(parents=True, exist_ok=True)
  raw_dir = args.out_dir / "raw"
  raw_dir.mkdir(parents=True, exist_ok=True)
  stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
  remote_dir = (
      f"{args.remote_root.rstrip('/')}/seq{args.sequence}-vulkan-preflight-{stamp}")
  local_cpp = args.out_dir / "vulkan_preflight.cpp"
  local_glsl = args.out_dir / "vulkan_preflight.comp"
  local_cpp.write_text(SMOKE_CPP, encoding="utf-8")
  local_glsl.write_text(SMOKE_GLSL, encoding="utf-8")

  predecessor_selects = (
      predecessor.get("required_checks_passed") is True
      and predecessor.get("vulkan_preflight_allowed") is True
      and predecessor.get("component_source_allowed") is False
      and predecessor.get("selected_next_route") == CURRENT_ROUTE
      and _has_candidate(routes, 606, CURRENT_ROUTE)
      and _has_switch(
          routes, 606,
          "select_gpu_vulkan_postconv_recurrent_component_preflight_gate"))
  setup = iq36_local.run_target(
      args.host, "mkdir -p " + shlex.quote(remote_dir), args.timeout_s)
  inventory_command = " && ".join([
      f"source {shlex.quote(args.env_script)} >/dev/null 2>&1",
      "printf 'vulkan_loader='; ldconfig -p | grep -q 'libvulkan.so.1' && echo true || echo false",
      "printf 'vulkan_header_system='; test -f /usr/include/vulkan/vulkan.h && echo true || echo false",
      "printf 'glslc='; command -v glslc || echo",
      "printf 'glslangValidator='; command -v glslangValidator || echo",
      "printf 'vulkaninfo='; command -v vulkaninfo || echo",
      "printf 'curl='; command -v curl || echo",
      "printf 'dpkg_deb='; command -v dpkg-deb || echo",
      (
          "printf 'llama_ssm_conv_source='; test -f "
          f"{shlex.quote(args.llama_source + '/ggml/src/ggml-vulkan/vulkan-shaders/ssm_conv.comp')} "
          "&& echo true || echo false"),
      (
          "printf 'llama_ssm_scan_source='; test -f "
          f"{shlex.quote(args.llama_source + '/ggml/src/ggml-vulkan/vulkan-shaders/ssm_scan.comp')} "
          "&& echo true || echo false"),
      (
          "printf 'llama_shader_generator_source='; test -f "
          f"{shlex.quote(args.llama_source + '/ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp')} "
          "&& echo true || echo false"),
  ])
  inventory_result = (
      iq36_local.run_target(
          args.host, f"bash -lc {shlex.quote(inventory_command)}", args.timeout_s)
      if setup.get("returncode") == 0 else {})
  inventory = _parse_key_values(str(inventory_result.get("stdout", "")))
  transfers = {
      "cpp": iq36_local.copy_to(
          args.host, local_cpp, f"{remote_dir}/vulkan_preflight.cpp",
          args.timeout_s),
      "glsl": iq36_local.copy_to(
          args.host, local_glsl, f"{remote_dir}/vulkan_preflight.comp",
          args.timeout_s),
  } if setup.get("returncode") == 0 else {}
  transfer_ok = (
      len(transfers) == 2
      and all(row.get("returncode") == 0 for row in transfers.values()))

  build_command = " && ".join([
      f"source {shlex.quote(args.env_script)} >/dev/null 2>&1",
      (
          "g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic "
          f"{shlex.quote(remote_dir + '/vulkan_preflight.cpp')} -lvulkan "
          f"-o {shlex.quote(remote_dir + '/iq36-vulkan-preflight')}")
  ])
  build = (
      iq36_local.run_target(
          args.host, f"bash -lc {shlex.quote(build_command)}", args.timeout_s)
      if transfer_ok else {})
  smoke_command = " && ".join([
      f"source {shlex.quote(args.env_script)} >/dev/null 2>&1",
      "export INTEL_FORCE_PROBE=b080",
      shlex.quote(remote_dir + "/iq36-vulkan-preflight"),
  ])
  smoke = (
      iq36_local.run_target(
          args.host, f"bash -lc {shlex.quote(smoke_command)}", args.timeout_s)
      if build.get("returncode") == 0 else {})
  smoke_json = _parse_json_line(str(smoke.get("stdout", "")))

  toolchain_root = remote_dir + "/shader-toolchain"
  glslc_deb = remote_dir + "/glslc.deb"
  shaderc_deb = remote_dir + "/libshaderc1.deb"
  pinned_toolchain_command = " && ".join([
      f"curl -fsSL {shlex.quote(GLSLC_PACKAGE_URI)} -o {shlex.quote(glslc_deb)}",
      f"printf '%s  %s\\n' {shlex.quote(GLSLC_PACKAGE_SHA512)} {shlex.quote(glslc_deb)} | sha512sum -c -",
      f"curl -fsSL {shlex.quote(SHADERC_PACKAGE_URI)} -o {shlex.quote(shaderc_deb)}",
      f"printf '%s  %s\\n' {shlex.quote(SHADERC_PACKAGE_SHA512)} {shlex.quote(shaderc_deb)} | sha512sum -c -",
      f"mkdir -p {shlex.quote(toolchain_root)}",
      f"dpkg-deb -x {shlex.quote(glslc_deb)} {shlex.quote(toolchain_root)}",
      f"dpkg-deb -x {shlex.quote(shaderc_deb)} {shlex.quote(toolchain_root)}",
      (
          f"LD_LIBRARY_PATH={shlex.quote(toolchain_root + '/usr/lib/x86_64-linux-gnu')} "
          f"{shlex.quote(toolchain_root + '/usr/bin/glslc')} --version"),
  ])
  pinned_toolchain = (
      iq36_local.run_target(
          args.host, pinned_toolchain_command, args.timeout_s)
      if transfer_ok and inventory.get("curl") and inventory.get("dpkg_deb")
      else {})
  if pinned_toolchain.get("returncode") == 0:
    shader_compiler = toolchain_root + "/usr/bin/glslc"
    shader_library_path = toolchain_root + "/usr/lib/x86_64-linux-gnu"
  elif inventory.get("glslc"):
    shader_compiler = inventory["glslc"]
    shader_library_path = ""
  else:
    shader_compiler = ""
    shader_library_path = ""
  if shader_compiler:
    shader_env = (
        f"LD_LIBRARY_PATH={shlex.quote(shader_library_path)} "
        if shader_library_path else "")
    shader_command = (
        f"{shader_env}{shlex.quote(shader_compiler)} -fshader-stage=compute "
        "--target-env=vulkan1.3 "
        f"{shlex.quote(remote_dir + '/vulkan_preflight.comp')} -o "
        f"{shlex.quote(remote_dir + '/vulkan_preflight.spv')}")
  else:
    shader_command = "false"
  shader_compile = (
      iq36_local.run_target(args.host, shader_command, args.timeout_s)
      if transfer_ok else {})
  shader_identity = (
      iq36_local.run_target(
          args.host,
          " && ".join([
              f"sha256sum {shlex.quote(remote_dir + '/vulkan_preflight.spv')}",
              f"od -An -tx4 -N4 {shlex.quote(remote_dir + '/vulkan_preflight.spv')}",
          ]),
          args.timeout_s)
      if shader_compile.get("returncode") == 0 else {})
  identity_lines = str(shader_identity.get("stdout", "")).splitlines()
  shader_sha256 = (
      identity_lines[0].split(maxsplit=1)[0]
      if identity_lines else None)
  shader_magic = identity_lines[1].strip() if len(identity_lines) > 1 else None
  process_check = iq36_local.run_target(
      args.host,
      "pgrep -af iq36-vulkan-preflight || true",
      args.timeout_s)
  lingering = [
      line for line in str(process_check.get("stdout", "")).splitlines()
      if "pgrep -af" not in line and "bash -lc" not in line
  ]
  cleanup = iq36_local.run_target(
      args.host, "rm -rf " + shlex.quote(remote_dir), args.timeout_s)

  iq36_local.write_json(raw_dir / "setup.json", setup)
  iq36_local.write_json(raw_dir / "inventory.json", inventory_result)
  iq36_local.write_json(raw_dir / "transfers.json", transfers)
  iq36_local.write_json(raw_dir / "build.json", build)
  iq36_local.write_json(raw_dir / "smoke.json", smoke)
  iq36_local.write_json(raw_dir / "shader-toolchain.json", pinned_toolchain)
  iq36_local.write_json(raw_dir / "shader-compile.json", shader_compile)
  iq36_local.write_json(raw_dir / "shader-identity.json", shader_identity)
  iq36_local.write_json(raw_dir / "process-check.json", process_check)
  iq36_local.write_json(raw_dir / "cleanup.json", cleanup)

  loader_header_ok = (
      inventory.get("vulkan_loader") == "true"
      and build.get("returncode") == 0)
  intel_compute_ok = (
      isinstance(smoke_json, dict)
      and smoke_json.get("vendor_id") == 0x8086
      and "Intel" in str(smoke_json.get("device_name", ""))
      and smoke_json.get("instance_created") is True
      and smoke_json.get("device_created") is True
      and smoke_json.get("queue_acquired") is True)
  shader_ok = (
      bool(shader_compiler)
      and (pinned_toolchain.get("returncode") == 0 or bool(inventory.get("glslc")))
      and shader_compile.get("returncode") == 0
      and isinstance(shader_sha256, str) and len(shader_sha256) == 64
      and shader_magic == "07230203")
  component_shader_reference_ok = (
      inventory.get("llama_ssm_conv_source") == "true"
      and inventory.get("llama_ssm_scan_source") == "true"
      and inventory.get("llama_shader_generator_source") == "true")
  source_is_preflight_only = (
      "/home/intel/models" not in SMOKE_CPP
      and "gguf" not in SMOKE_CPP.lower()
      and "vkCmdDispatch" not in SMOKE_CPP)
  checks = [
      {"name": "seq606_selected_vulkan_preflight_only",
       "pass": predecessor_selects},
      {"name": "vulkan_loader_and_compilation_header_route_present",
       "pass": loader_header_ok, "detail": inventory},
      {"name": "preflight_source_transferred_and_compiled",
       "pass": transfer_ok and build.get("returncode") == 0},
      {"name": "intel_vulkan_compute_queue_create_destroy_passed",
       "pass": smoke.get("returncode") == 0 and intel_compute_ok,
       "detail": smoke_json},
      {"name": "compute_shader_compiles_to_valid_spirv",
       "pass": shader_ok,
       "detail": {
           "compiler": shader_compiler,
           "pinned_private_toolchain": pinned_toolchain.get("returncode") == 0,
           "glslc_package_sha512": GLSLC_PACKAGE_SHA512,
           "shaderc_package_sha512": SHADERC_PACKAGE_SHA512,
           "sha256": shader_sha256,
           "magic": shader_magic,
       }},
      {"name": "captured_component_has_vulkan_shader_source_references",
       "pass": component_shader_reference_ok,
       "detail": {
           "llama_source": args.llama_source,
           "ssm_conv": inventory.get("llama_ssm_conv_source"),
           "ssm_scan": inventory.get("llama_ssm_scan_source"),
           "generator": inventory.get("llama_shader_generator_source"),
       }},
      {"name": "preflight_did_not_access_model_or_dispatch_component",
       "pass": source_is_preflight_only},
      {"name": "preflight_process_and_remote_directory_cleaned",
       "pass": not lingering and cleanup.get("returncode") == 0,
       "detail": {"lingering": lingering}},
  ]
  required = all(bool(row["pass"]) for row in checks)
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "sequence": args.sequence,
      "inputs": {
          "routes": _rel(args.routes),
          "predecessor": _rel(args.predecessor),
          "host": args.host,
          "env_script": args.env_script,
          "llama_source": args.llama_source,
          "generated_cpp": _rel(local_cpp),
          "generated_glsl": _rel(local_glsl),
          "pinned_glslc_package_uri": GLSLC_PACKAGE_URI,
          "pinned_shaderc_package_uri": SHADERC_PACKAGE_URI,
      },
      "inventory": inventory,
      "smoke": smoke_json,
      "shader": {
          "compiler": shader_compiler,
          "pinned_private_toolchain": pinned_toolchain.get("returncode") == 0,
          "sha256": shader_sha256,
          "magic": shader_magic,
      },
      "checks": checks,
      "required_checks_passed": required,
      "vulkan_preflight_passed": required,
      "component_design_allowed": required,
      "component_source_allowed": False,
      "component_probe_allowed": False,
      "decode_integration_allowed": False,
      "token_row_allowed": False,
      "validation_or_test_allowed": False,
      "speedup_claims_allowed": False,
      "promotion_allowed": False,
      "disposition": (
          "accept_native_vulkan_component_preflight"
          if required else "reject_or_repair_native_vulkan_component_preflight"),
      "selected_next_route": (
          SELECTED_NEXT_ROUTE if required else CURRENT_ROUTE),
      "next_route_reason": (
          "The target has an independent Intel Vulkan compute path, a "
          "checksum-pinned private GLSL-to-SPIR-V route, and denominator "
          "shader-source references. Design only the captured postconv/"
          "recurrent component next; engine source and model execution remain "
          "blocked."
          if required else
          "Repair or reject Vulkan loader/header/device/shader capability "
          "before component design; do not fall through to model execution."),
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
          "vulkan_preflight_passed": metrics["vulkan_preflight_passed"],
          "selected_next_route": metrics["selected_next_route"],
          "component_source_allowed": False,
          "token_row_allowed": False,
      }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  failed = [row["name"] for row in metrics["checks"] if not row["pass"]]
  lines = [
      f"# Seq{metrics['sequence']} Native Vulkan Component Preflight",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- device: `{(metrics.get('smoke') or {}).get('device_name')}`",
      f"- shader compiler: `{metrics['shader']['compiler']}`",
      f"- SPIR-V SHA256: `{metrics['shader']['sha256']}`",
      f"- failed_checks: `{failed}`",
      "",
      metrics["next_route_reason"],
      "",
      "The smoke created/destroyed Vulkan objects only. It did not access the model or dispatch a component.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--sequence", type=int, default=607)
  parser.add_argument("--routes", type=Path,
                      default=ACTIVE / "routes-ledger.json")
  parser.add_argument(
      "--predecessor", type=Path,
      default=ROOT / (
          "output/seq606-gpu-backend-runtime-route-reflection-gate-"
          "20260710Tseq606Z/metrics.json"))
  parser.add_argument("--target", "--host", dest="host", metavar="TARGET", default=DEFAULT_HOST)
  parser.add_argument("--env-script", default=DEFAULT_ENV_SCRIPT)
  parser.add_argument("--llama-source", default=DEFAULT_LLAMA_SOURCE)
  parser.add_argument("--staging-root", "--remote-root", dest="remote_root", metavar="PATH", default=DEFAULT_REMOTE_ROOT)
  parser.add_argument("--timeout-s", type=int, default=600)
  parser.add_argument(
      "--out-dir", type=Path,
      default=ROOT / (
          "output/seq607-gpu-vulkan-postconv-recurrent-component-preflight-"
          "gate-20260710Tseq607Z"))
  args = parser.parse_args()
  metrics = compute(args)
  write_outputs(metrics, args.out_dir)
  print(json.dumps({
      "required_checks_passed": metrics["required_checks_passed"],
      "disposition": metrics["disposition"],
      "vulkan_preflight_passed": metrics["vulkan_preflight_passed"],
      "selected_next_route": metrics["selected_next_route"],
      "out_dir": _rel(args.out_dir),
  }, sort_keys=True))
  return 0 if metrics["required_checks_passed"] else 2


if __name__ == "__main__":
  raise SystemExit(main())
