#!/usr/bin/env python3
"""Preflight a self-owned Level Zero component route without model access."""

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
    "intel-qwen36-gpu-level-zero-postconv-recurrent-component-preflight-v0")
CURRENT_ROUTE = (
    "gpu_level_zero_postconv_recurrent_component_preflight_gate")
SELECTED_NEXT_ROUTE = (
    "gpu_level_zero_postconv_recurrent_component_design_gate")
DEFAULT_HOST = "local"
DEFAULT_ENV_SCRIPT = (
    "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh")
DEFAULT_REMOTE_ROOT = "/home/intel/intel-qwen36-gpu"
TARGET_DEVICE_ID = 0xB080


SMOKE_CL = r'''
__kernel void iq36_level_zero_preflight(__global uint* value) {
  const uint index = (uint)get_global_id(0);
  value[index] += 1U;
}
'''


SMOKE_CPP = r'''
#include <level_zero/ze_api.h>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

[[noreturn]] void Die(const std::string& message) {
  throw std::runtime_error(message);
}

void Check(ze_result_t result, const char* where) {
  if (result != ZE_RESULT_SUCCESS) {
    Die(std::string(where) + " failed with ze_result_t " +
        std::to_string(static_cast<unsigned int>(result)));
  }
}

std::vector<std::uint8_t> ReadBinary(const std::string& path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) Die("failed to open module: " + path);
  const std::streamoff bytes = input.tellg();
  if (bytes <= 0) Die("empty module: " + path);
  std::vector<std::uint8_t> data(static_cast<std::size_t>(bytes));
  input.seekg(0, std::ios::beg);
  input.read(reinterpret_cast<char*>(data.data()), bytes);
  if (!input) Die("failed to read module: " + path);
  return data;
}

std::string JsonEscape(const char* value) {
  std::string out;
  for (const char ch : std::string(value)) {
    if (ch == '\\' || ch == '"') out.push_back('\\');
    out.push_back(ch);
  }
  return out;
}

int main(int argc, char** argv) {
  try {
    if (argc != 2) Die("usage: level-zero-preflight MODULE");
    const auto module_bytes = ReadBinary(argv[1]);
    Check(zeInit(ZE_INIT_FLAG_GPU_ONLY), "zeInit");
    std::uint32_t driver_count = 0;
    Check(zeDriverGet(&driver_count, nullptr), "zeDriverGet(count)");
    if (driver_count == 0) Die("no Level Zero drivers");
    std::vector<ze_driver_handle_t> drivers(driver_count);
    Check(zeDriverGet(&driver_count, drivers.data()), "zeDriverGet(list)");

    ze_driver_handle_t selected_driver = nullptr;
    ze_device_handle_t selected_device = nullptr;
    ze_device_properties_t selected_properties{
        ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES};
    std::uint32_t selected_ordinal = UINT32_MAX;
    std::uint32_t enumerated_devices = 0;
    for (ze_driver_handle_t driver : drivers) {
      std::uint32_t device_count = 0;
      Check(zeDeviceGet(driver, &device_count, nullptr), "zeDeviceGet(count)");
      std::vector<ze_device_handle_t> devices(device_count);
      Check(zeDeviceGet(driver, &device_count, devices.data()),
            "zeDeviceGet(list)");
      enumerated_devices += device_count;
      for (ze_device_handle_t device : devices) {
        ze_device_properties_t properties{ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES};
        Check(zeDeviceGetProperties(device, &properties),
              "zeDeviceGetProperties");
        if (properties.vendorId != 0x8086U || properties.deviceId != 0xB080U) {
          continue;
        }
        std::uint32_t group_count = 0;
        Check(zeDeviceGetCommandQueueGroupProperties(
                  device, &group_count, nullptr),
              "zeDeviceGetCommandQueueGroupProperties(count)");
        std::vector<ze_command_queue_group_properties_t> groups(group_count);
        for (auto& group : groups) {
          group.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES;
        }
        Check(zeDeviceGetCommandQueueGroupProperties(
                  device, &group_count, groups.data()),
              "zeDeviceGetCommandQueueGroupProperties(list)");
        for (std::uint32_t ordinal = 0; ordinal < group_count; ++ordinal) {
          if ((groups[ordinal].flags &
               ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) != 0U) {
            selected_driver = driver;
            selected_device = device;
            selected_properties = properties;
            selected_ordinal = ordinal;
            break;
          }
        }
        if (selected_device != nullptr) break;
      }
      if (selected_device != nullptr) break;
    }
    if (selected_device == nullptr) Die("PTL 0xb080 Level Zero device missing");

    ze_context_desc_t context_desc{ZE_STRUCTURE_TYPE_CONTEXT_DESC};
    ze_context_handle_t context = nullptr;
    Check(zeContextCreate(
              selected_driver, &context_desc, &context),
          "zeContextCreate");
    ze_command_queue_desc_t queue_desc{
        ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC};
    queue_desc.ordinal = selected_ordinal;
    queue_desc.index = 0;
    queue_desc.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
    queue_desc.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
    ze_command_queue_handle_t queue = nullptr;
    Check(zeCommandQueueCreate(
              context, selected_device, &queue_desc, &queue),
          "zeCommandQueueCreate");

    ze_module_desc_t module_desc{ZE_STRUCTURE_TYPE_MODULE_DESC};
    module_desc.format = ZE_MODULE_FORMAT_NATIVE;
    module_desc.inputSize = module_bytes.size();
    module_desc.pInputModule = module_bytes.data();
    module_desc.pBuildFlags = "";
    ze_module_handle_t module = nullptr;
    ze_module_build_log_handle_t build_log = nullptr;
    const ze_result_t module_result = zeModuleCreate(
        context, selected_device, &module_desc, &module, &build_log);
    if (module_result != ZE_RESULT_SUCCESS) {
      std::string detail;
      if (build_log != nullptr) {
        std::size_t log_size = 0;
        zeModuleBuildLogGetString(build_log, &log_size, nullptr);
        std::vector<char> log(log_size);
        if (log_size > 0) {
          zeModuleBuildLogGetString(build_log, &log_size, log.data());
          detail.assign(log.data(), log.data() + log_size);
        }
      }
      if (build_log != nullptr) zeModuleBuildLogDestroy(build_log);
      Die("zeModuleCreate failed with " +
          std::to_string(static_cast<unsigned int>(module_result)) +
          ": " + detail);
    }
    if (build_log != nullptr) zeModuleBuildLogDestroy(build_log);

    std::cout << "{";
    std::cout << "\"schema_version\":\"iq36-level-zero-preflight-smoke-v0\",";
    std::cout << "\"driver_count\":" << driver_count << ",";
    std::cout << "\"enumerated_device_count\":" << enumerated_devices << ",";
    std::cout << "\"device_name\":\""
              << JsonEscape(selected_properties.name) << "\",";
    std::cout << "\"vendor_id\":" << selected_properties.vendorId << ",";
    std::cout << "\"device_id\":" << selected_properties.deviceId << ",";
    std::cout << "\"compute_queue_ordinal\":" << selected_ordinal << ",";
    std::cout << "\"module_size_bytes\":" << module_bytes.size() << ",";
    std::cout << "\"context_created\":true,";
    std::cout << "\"queue_created\":true,";
    std::cout << "\"native_module_created\":true";
    std::cout << "}" << std::endl;

    Check(zeModuleDestroy(module), "zeModuleDestroy");
    Check(zeCommandQueueDestroy(queue), "zeCommandQueueDestroy");
    Check(zeContextDestroy(context), "zeContextDestroy");
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "Level Zero preflight error: " << ex.what() << std::endl;
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
  local_cpp = args.out_dir / "level_zero_preflight.cpp"
  local_cl = args.out_dir / "level_zero_preflight.cl"
  local_cpp.write_text(SMOKE_CPP, encoding="utf-8")
  local_cl.write_text(SMOKE_CL, encoding="utf-8")
  stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
  remote_dir = (
      f"{args.remote_root.rstrip('/')}/seq{args.sequence}-level-zero-{stamp}")
  predecessor_selects = (
      predecessor.get("required_checks_passed") is True
      and predecessor.get("vulkan_component_closed") is True
      and predecessor.get("level_zero_preflight_allowed") is True
      and predecessor.get("selected_next_route") == CURRENT_ROUTE
      and _has_candidate(routes, 612, CURRENT_ROUTE)
      and _has_switch(
          routes, 612,
          "select_gpu_level_zero_postconv_recurrent_component_preflight_gate"))

  setup = iq36_local.run_target(
      args.host, "mkdir -p " + shlex.quote(remote_dir), args.timeout_s)
  inventory_command = " && ".join([
      f"source {shlex.quote(args.env_script)} >/dev/null 2>&1",
      "printf 'ze_loader='; ldconfig -p | grep -q 'libze_loader.so' && echo true || echo false",
      "printf 'ze_header_env='; test -f /home/intel/intel-box-env/conda/include/level_zero/ze_api.h && echo true || echo false",
      "printf 'ocloc='; command -v ocloc || echo",
      "printf 'ptl_ocloc_id='; ocloc ids ptl 2>/dev/null | tail -1",
  ])
  inventory_result = (
      iq36_local.run_target(
          args.host, f"bash -lc {shlex.quote(inventory_command)}", args.timeout_s)
      if setup.get("returncode") == 0 else {})
  inventory = _parse_key_values(str(inventory_result.get("stdout", "")))
  transfers = {
      "cpp": iq36_local.copy_to(
          args.host, local_cpp, remote_dir + "/level_zero_preflight.cpp",
          args.timeout_s),
      "cl": iq36_local.copy_to(
          args.host, local_cl, remote_dir + "/level_zero_preflight.cl",
          args.timeout_s),
  } if setup.get("returncode") == 0 else {}
  transfer_ok = (
      len(transfers) == 2
      and all(row.get("returncode") == 0 for row in transfers.values()))
  native_module = remote_dir + "/iq36_level_zero_preflight.bin"
  ocloc_command = " ".join([
      shlex.quote(inventory.get("ocloc", "ocloc")), "compile",
      "-file", shlex.quote(remote_dir + "/level_zero_preflight.cl"),
      "-device", hex(TARGET_DEVICE_ID),
      "-output", "iq36_level_zero_preflight",
      "-out_dir", shlex.quote(remote_dir),
      "-output_no_suffix", "--format", "zebin", "-q",
  ])
  module_compile = (
      iq36_local.run_target(args.host, ocloc_command, args.timeout_s)
      if transfer_ok and inventory.get("ocloc") else {})
  module_identity = (
      iq36_local.run_target(
          args.host,
          " && ".join([
              f"test -s {shlex.quote(native_module)}",
              f"sha256sum {shlex.quote(native_module)}",
              f"file {shlex.quote(native_module)}",
          ]),
          args.timeout_s)
      if module_compile.get("returncode") == 0 else {})
  identity_lines = str(module_identity.get("stdout", "")).splitlines()
  module_sha = (
      identity_lines[0].split(maxsplit=1)[0] if identity_lines else None)
  build_command = " && ".join([
      f"source {shlex.quote(args.env_script)} >/dev/null 2>&1",
      (
          "g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic "
          f"{shlex.quote(remote_dir + '/level_zero_preflight.cpp')} "
          "-lze_loader "
          f"-o {shlex.quote(remote_dir + '/iq36-level-zero-preflight')}")
  ])
  build = (
      iq36_local.run_target(
          args.host, f"bash -lc {shlex.quote(build_command)}", args.timeout_s)
      if transfer_ok else {})
  smoke_command = " && ".join([
      f"source {shlex.quote(args.env_script)} >/dev/null 2>&1",
      shlex.quote(remote_dir + "/iq36-level-zero-preflight") + " " +
      shlex.quote(native_module),
  ])
  smoke = (
      iq36_local.run_target(
          args.host, f"bash -lc {shlex.quote(smoke_command)}", args.timeout_s)
      if build.get("returncode") == 0
      and module_identity.get("returncode") == 0 else {})
  smoke_json = _parse_json_line(str(smoke.get("stdout", "")))
  process_check = iq36_local.run_target(
      args.host, "pgrep -af iq36-level-zero-preflight || true",
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
  iq36_local.write_json(raw_dir / "module-compile.json", module_compile)
  iq36_local.write_json(raw_dir / "module-identity.json", module_identity)
  iq36_local.write_json(raw_dir / "build.json", build)
  iq36_local.write_json(raw_dir / "smoke.json", smoke)
  iq36_local.write_json(raw_dir / "process-check.json", process_check)
  iq36_local.write_json(raw_dir / "cleanup.json", cleanup)

  inventory_ok = (
      inventory.get("ze_loader") == "true"
      and inventory.get("ze_header_env") == "true"
      and bool(inventory.get("ocloc"))
      and inventory.get("ptl_ocloc_id") == "30.0.4")
  module_ok = (
      module_compile.get("returncode") == 0
      and module_identity.get("returncode") == 0
      and isinstance(module_sha, str) and len(module_sha) == 64)
  level_zero_ok = (
      smoke.get("returncode") == 0
      and isinstance(smoke_json, dict)
      and smoke_json.get("vendor_id") == 0x8086
      and smoke_json.get("device_id") == TARGET_DEVICE_ID
      and smoke_json.get("context_created") is True
      and smoke_json.get("queue_created") is True
      and smoke_json.get("native_module_created") is True)
  source_is_preflight_only = (
      "/home/intel/models" not in SMOKE_CPP
      and "zeKernelCreate" not in SMOKE_CPP
      and "zeCommandListAppendLaunchKernel" not in SMOKE_CPP)
  checks = [
      {"name": "seq612_selected_level_zero_preflight_only",
       "pass": predecessor_selects},
      {"name": "level_zero_loader_header_and_exact_ptl_ocloc_id_present",
       "pass": inventory_ok, "detail": inventory},
      {"name": "minimal_opencl_source_compiles_to_nonempty_ptl_native_module",
       "pass": module_ok,
       "detail": {"sha256": module_sha, "device": hex(TARGET_DEVICE_ID)}},
      {"name": "preflight_source_compiles_against_level_zero_headers",
       "pass": build.get("returncode") == 0},
      {"name": "ptl_context_compute_queue_and_native_module_create_destroy_pass",
       "pass": level_zero_ok, "detail": smoke_json},
      {"name": "preflight_created_no_kernel_and_accessed_no_model",
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
          "target_device_id": hex(TARGET_DEVICE_ID),
          "generated_cpp": _rel(local_cpp),
          "generated_opencl": _rel(local_cl),
      },
      "inventory": inventory,
      "native_module": {"sha256": module_sha},
      "smoke": smoke_json,
      "checks": checks,
      "required_checks_passed": required,
      "level_zero_preflight_passed": required,
      "component_design_allowed": required,
      "component_source_allowed": False,
      "component_probe_allowed": False,
      "decode_integration_allowed": False,
      "token_row_allowed": False,
      "validation_or_test_allowed": False,
      "speedup_claims_allowed": False,
      "promotion_allowed": False,
      "disposition": (
          "accept_native_level_zero_component_preflight"
          if required else "reject_native_level_zero_component_preflight"),
      "selected_next_route": (
          SELECTED_NEXT_ROUTE if required else CURRENT_ROUTE),
      "next_route_reason": (
          "PTL has an independent Level Zero context/compute queue and native "
          "ocloc module path. Design one bounded component only; source and "
          "kernel creation remain blocked."
          if required else
          "Close or repair Level Zero loader/header/PTL/module capability "
          "before any component design or source."),
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
          "native_module": metrics["native_module"],
          "level_zero_preflight_passed": metrics[
              "level_zero_preflight_passed"],
          "selected_next_route": metrics["selected_next_route"],
          "component_source_allowed": False,
          "token_row_allowed": False,
      }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  failed = [row["name"] for row in metrics["checks"] if not row["pass"]]
  lines = [
      f"# Seq{metrics['sequence']} Native Level Zero Component Preflight",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- device: `{(metrics.get('smoke') or {}).get('device_name')}`",
      f"- native module SHA256: `{metrics['native_module']['sha256']}`",
      f"- failed_checks: `{failed}`",
      "",
      metrics["next_route_reason"],
      "",
      "The smoke created no kernel, launched no command list, and did not access the model.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--sequence", type=int, default=613)
  parser.add_argument("--routes", type=Path,
                      default=ACTIVE / "routes-ledger.json")
  parser.add_argument(
      "--predecessor", type=Path,
      default=ROOT / (
          "output/seq612-gpu-vulkan-postconv-recurrent-component-route-close-"
          "gate-20260710Tseq612Z/metrics.json"))
  parser.add_argument("--target", "--host", dest="host", metavar="TARGET", default=DEFAULT_HOST)
  parser.add_argument("--env-script", default=DEFAULT_ENV_SCRIPT)
  parser.add_argument("--staging-root", "--remote-root", dest="remote_root", metavar="PATH", default=DEFAULT_REMOTE_ROOT)
  parser.add_argument("--timeout-s", type=int, default=600)
  parser.add_argument(
      "--out-dir", type=Path,
      default=ROOT / (
          "output/seq613-gpu-level-zero-postconv-recurrent-component-"
          "preflight-gate-20260710Tseq613Z"))
  args = parser.parse_args()
  metrics = compute(args)
  write_outputs(metrics, args.out_dir)
  print(json.dumps({
      "required_checks_passed": metrics["required_checks_passed"],
      "level_zero_preflight_passed": metrics["level_zero_preflight_passed"],
      "disposition": metrics["disposition"],
      "selected_next_route": metrics["selected_next_route"],
      "out_dir": _rel(args.out_dir),
  }, sort_keys=True))
  return 0 if metrics["required_checks_passed"] else 2


if __name__ == "__main__":
  raise SystemExit(main())
