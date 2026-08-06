#!/usr/bin/env python3
"""Source-gate the locked native Vulkan postconv/recurrent component."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
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
    "intel-qwen36-gpu-vulkan-postconv-recurrent-component-source-v0")
CURRENT_ROUTE = "gpu_vulkan_postconv_recurrent_component_source_gate"
SELECTED_NEXT_ROUTE = (
    "gpu_vulkan_postconv_recurrent_component_target_compile_gate")
DEFAULT_HOST = "local"
DEFAULT_ENV_SCRIPT = (
    "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh")
DEFAULT_REMOTE_ROOT = "/home/intel/intel-qwen36-gpu"
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


API_HARNESS = r'''
#include "intel_qwen36/gpu_vulkan_postconv_recurrent.hpp"

#include <type_traits>

static_assert(!std::is_copy_constructible_v<
              iq36::GpuVulkanPostconvRecurrentRunner>);
static_assert(std::is_move_constructible_v<
              iq36::GpuVulkanPostconvRecurrentRunner>);

int main() {
  iq36::GpuVulkanPostconvRecurrentInput input;
  iq36::GpuVulkanPostconvRecurrentRun run;
  return static_cast<int>(input.conv_output_raw.size() +
                          run.sample_wall_us.size());
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


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _run(command: list[str], out_dir: Path, stem: str) -> dict[str, Any]:
  completed = subprocess.run(
      command, cwd=ROOT, capture_output=True, text=True, check=False)
  stdout_path = out_dir / f"{stem}.stdout.txt"
  stderr_path = out_dir / f"{stem}.stderr.txt"
  stdout_path.write_text(completed.stdout, encoding="utf-8")
  stderr_path.write_text(completed.stderr, encoding="utf-8")
  return {
      "passed": completed.returncode == 0,
      "command": command,
      "returncode": completed.returncode,
      "stdout": _rel(stdout_path),
      "stderr": _rel(stderr_path),
  }


def _spirv_identity(spv: Path, assembly: Path) -> dict[str, Any]:
  data = spv.read_bytes() if spv.exists() else b""
  text = assembly.read_text(encoding="utf-8") if assembly.exists() else ""
  return {
      "sha256": hashlib.sha256(data).hexdigest() if data else None,
      "size_bytes": len(data),
      "magic": data[:4].hex() if len(data) >= 4 else None,
      "no_contraction_count": text.count("NoContraction"),
      "op_fma_count": text.count("OpFma"),
      "assembly_sha256": (
          hashlib.sha256(text.encode("utf-8")).hexdigest() if text else None),
  }


def compute(args: argparse.Namespace) -> dict[str, Any]:
  routes = _load(args.routes)
  predecessor = _load(args.predecessor)
  design = predecessor.get("design", {})
  header = args.header.read_text(encoding="utf-8")
  source = args.source.read_text(encoding="utf-8")
  postconv = args.postconv_shader.read_text(encoding="utf-8")
  recurrent = args.recurrent_shader.read_text(encoding="utf-8")
  cmake = args.cmake.read_text(encoding="utf-8")
  args.out_dir.mkdir(parents=True, exist_ok=True)
  raw_dir = args.out_dir / "raw"
  compile_dir = args.out_dir / "compile"
  generated_dir = args.out_dir / "generated"
  raw_dir.mkdir(parents=True, exist_ok=True)
  compile_dir.mkdir(parents=True, exist_ok=True)
  generated_dir.mkdir(parents=True, exist_ok=True)
  api_harness = compile_dir / "api_harness.cpp"
  api_harness.write_text(API_HARNESS, encoding="utf-8")

  predecessor_selects = (
      predecessor.get("required_checks_passed") is True
      and predecessor.get("component_source_allowed") is True
      and predecessor.get("target_compile_allowed") is False
      and predecessor.get("selected_next_route") == CURRENT_ROUTE
      and design.get("candidate") == "vulkan_precise_postconv_recurrent_v1"
      and _has_candidate(routes, 608, CURRENT_ROUTE)
      and _has_switch(
          routes, 608,
          "select_gpu_vulkan_postconv_recurrent_component_source_gate"))

  source_shape_ok = all(marker in source for marker in [
      "class GpuVulkanPostconvRecurrentRunner::Impl",
      "available.shaderFloat64",
      "VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT",
      "VK_MEMORY_PROPERTY_HOST_COHERENT_BIT",
      "vkCmdDispatch(command, 64, 1, 1);",
      "vkCmdDispatch(command, 32, 1, 1);",
      "VK_ACCESS_SHADER_WRITE_BIT",
      "VK_ACCESS_SHADER_READ_BIT",
      "buffers[6].mapped, input.recurrent_state.data()",
  ]) and source.count("vkCmdDispatch(") == 2
  api_shape_ok = all(marker in header for marker in [
      "struct GpuVulkanPostconvRecurrentInput",
      "struct GpuVulkanPostconvRecurrentRun",
      "class GpuVulkanPostconvRecurrentRunner",
      "const GpuVulkanPostconvRecurrentRunner&) = delete;",
      "std::unique_ptr<Impl> impl_;",
  ])
  postconv_shape_ok = all(marker in postconv for marker in [
      "#extension GL_ARB_gpu_shader_fp64 : require",
      "layout(local_size_x = 128",
      "precise double",
      "sum = sum + square;",
      "head_scale = 1.0 / denominator;",
  ]) and postconv.count("layout(binding =") == 4
  recurrent_shape_ok = all(marker in recurrent for marker in [
      "layout(local_size_x = 128",
      "precise float decayed",
      "sum_k = sum_k + product;",
      "state_output[state_base + col] = updated;",
      "sum_q = sum_q + product;",
      "sum_squares = sum_squares + square;",
      "precise float final_value = weighted * z_silu_input",
  ]) and recurrent.count("layout(binding =") == 11
  runtime_is_independent = not any(
      marker.lower() in (header + source).lower()
      for marker in ["llama", "openvino", "opencl", "cl_mem", "gguf"])
  no_runtime_compiler = not any(
      marker in source for marker in ["glslc", "shaderc", "system(", "popen("])
  not_in_default_build = "gpu_vulkan_postconv_recurrent.cpp" not in cmake

  api_compile = _run([
      args.cxx, "-std=c++17", "-Wall", "-Wextra", "-Wpedantic",
      "-Iengine/include", "-fsyntax-only", _rel(api_harness),
  ], compile_dir, "api-harness-syntax")
  code_volume = _run([
      "python3", "tools/intel-qwen36-code-volume-check.py",
  ], compile_dir, "code-volume")

  remote_dir = (
      f"{args.remote_root.rstrip('/')}/seq{args.sequence}-vulkan-source")
  setup = iq36_local.run_target(
      args.host,
      " && ".join([
          "rm -rf " + shlex.quote(remote_dir),
          "mkdir -p " + shlex.quote(remote_dir + "/include/intel_qwen36"),
          "mkdir -p " + shlex.quote(remote_dir + "/src"),
          "mkdir -p " + shlex.quote(remote_dir + "/shaders"),
          "mkdir -p " + shlex.quote(remote_dir + "/generated"),
      ]),
      args.timeout_s)
  transfers = {}
  if setup.get("returncode") == 0:
    transfers = {
        "header": iq36_local.copy_to(
            args.host, args.header,
            remote_dir + "/include/intel_qwen36/"
            "gpu_vulkan_postconv_recurrent.hpp", args.timeout_s),
        "source": iq36_local.copy_to(
            args.host, args.source,
            remote_dir + "/src/gpu_vulkan_postconv_recurrent.cpp",
            args.timeout_s),
        "postconv": iq36_local.copy_to(
            args.host, args.postconv_shader,
            remote_dir + "/shaders/iq36_postconv_cpuorder.comp",
            args.timeout_s),
        "recurrent": iq36_local.copy_to(
            args.host, args.recurrent_shader,
            remote_dir + "/shaders/iq36_delta_recurrent_cpuorder.comp",
            args.timeout_s),
    }
  transfer_ok = (
      len(transfers) == 4
      and all(row.get("returncode") == 0 for row in transfers.values()))

  glslc_deb = remote_dir + "/glslc.deb"
  shaderc_deb = remote_dir + "/libshaderc1.deb"
  toolchain_root = remote_dir + "/shader-toolchain"
  glslc = toolchain_root + "/usr/bin/glslc"
  shader_lib = toolchain_root + "/usr/lib/x86_64-linux-gnu"
  toolchain_command = " && ".join([
      f"curl -fsSL {shlex.quote(GLSLC_PACKAGE_URI)} -o {shlex.quote(glslc_deb)}",
      f"printf '%s  %s\\n' {shlex.quote(GLSLC_PACKAGE_SHA512)} {shlex.quote(glslc_deb)} | sha512sum -c -",
      f"curl -fsSL {shlex.quote(SHADERC_PACKAGE_URI)} -o {shlex.quote(shaderc_deb)}",
      f"printf '%s  %s\\n' {shlex.quote(SHADERC_PACKAGE_SHA512)} {shlex.quote(shaderc_deb)} | sha512sum -c -",
      f"mkdir -p {shlex.quote(toolchain_root)}",
      f"dpkg-deb -x {shlex.quote(glslc_deb)} {shlex.quote(toolchain_root)}",
      f"dpkg-deb -x {shlex.quote(shaderc_deb)} {shlex.quote(toolchain_root)}",
      f"LD_LIBRARY_PATH={shlex.quote(shader_lib)} {shlex.quote(glslc)} --version",
  ])
  toolchain = (
      iq36_local.run_target(args.host, toolchain_command, args.timeout_s)
      if transfer_ok else {})
  shader_commands = []
  for name in ("iq36_postconv_cpuorder", "iq36_delta_recurrent_cpuorder"):
    source_path = remote_dir + f"/shaders/{name}.comp"
    spv_path = remote_dir + f"/generated/{name}.spv"
    asm_path = remote_dir + f"/generated/{name}.spvasm"
    shader_commands.extend([
        (
            f"LD_LIBRARY_PATH={shlex.quote(shader_lib)} {shlex.quote(glslc)} "
            "-O0 -fshader-stage=compute --target-env=vulkan1.2 "
            f"{shlex.quote(source_path)} -o {shlex.quote(spv_path)}"),
        (
            f"LD_LIBRARY_PATH={shlex.quote(shader_lib)} {shlex.quote(glslc)} "
            "-O0 -S -fshader-stage=compute --target-env=vulkan1.2 "
            f"{shlex.quote(source_path)} -o {shlex.quote(asm_path)}"),
    ])
  shader_compile = (
      iq36_local.run_target(
          args.host, " && ".join(shader_commands), args.timeout_s)
      if toolchain.get("returncode") == 0 else {})
  syntax_command = " && ".join([
      f"source {shlex.quote(args.env_script)} >/dev/null 2>&1",
      (
          "g++ -std=c++17 -O0 -Wall -Wextra -Wpedantic -fsyntax-only "
          f"-I{shlex.quote(remote_dir + '/include')} "
          f"{shlex.quote(remote_dir + '/src/gpu_vulkan_postconv_recurrent.cpp')}")
  ])
  source_syntax = (
      iq36_local.run_target(
          args.host, f"bash -lc {shlex.quote(syntax_command)}", args.timeout_s)
      if transfer_ok else {})

  fetches = {}
  if shader_compile.get("returncode") == 0:
    for name in ("iq36_postconv_cpuorder", "iq36_delta_recurrent_cpuorder"):
      for suffix in ("spv", "spvasm"):
        local = generated_dir / f"{name}.{suffix}"
        fetches[f"{name}.{suffix}"] = iq36_local.copy_from(
            args.host, remote_dir + f"/generated/{name}.{suffix}",
            local, args.timeout_s)
  fetch_ok = (
      len(fetches) == 4
      and all(row.get("returncode") == 0 for row in fetches.values()))
  identities = {
      name: _spirv_identity(
          generated_dir / f"{name}.spv",
          generated_dir / f"{name}.spvasm")
      for name in ("iq36_postconv_cpuorder", "iq36_delta_recurrent_cpuorder")
  }
  spirv_contract_ok = fetch_ok and all(
      row.get("magic") == "03022307"
      and isinstance(row.get("sha256"), str)
      and len(row["sha256"]) == 64
      and row.get("no_contraction_count", 0) > 0
      and row.get("op_fma_count") == 0
      for row in identities.values())
  cleanup = iq36_local.run_target(
      args.host, "rm -rf " + shlex.quote(remote_dir), args.timeout_s)
  iq36_local.write_json(raw_dir / "setup.json", setup)
  iq36_local.write_json(raw_dir / "transfers.json", transfers)
  iq36_local.write_json(raw_dir / "shader-toolchain.json", toolchain)
  iq36_local.write_json(raw_dir / "shader-compile.json", shader_compile)
  iq36_local.write_json(raw_dir / "source-syntax.json", source_syntax)
  iq36_local.write_json(raw_dir / "fetches.json", fetches)
  iq36_local.write_json(raw_dir / "cleanup.json", cleanup)

  checks = [
      {"name": "seq608_selected_one_vulkan_source_candidate",
       "pass": predecessor_selects},
      {"name": "public_api_is_move_only_and_component_scoped",
       "pass": api_shape_ok and api_compile["passed"],
       "detail": api_compile},
      {"name": "runtime_owns_exactly_two_dispatches_and_resident_buffers",
       "pass": source_shape_ok},
      {"name": "postconv_shader_matches_locked_fp64_serial_l2_shape",
       "pass": postconv_shape_ok},
      {"name": "recurrent_shader_matches_locked_cpu_phase_shape",
       "pass": recurrent_shape_ok},
      {"name": "runtime_has_no_llama_openvino_opencl_or_compiler_dependency",
       "pass": runtime_is_independent and no_runtime_compiler},
      {"name": "component_is_not_wired_into_default_engine_build",
       "pass": not_in_default_build},
      {"name": "pinned_private_toolchain_generates_audited_spirv",
       "pass": (
           toolchain.get("returncode") == 0
           and shader_compile.get("returncode") == 0
           and spirv_contract_ok),
       "detail": identities},
      {"name": "complete_runtime_source_passes_target_header_syntax",
       "pass": source_syntax.get("returncode") == 0,
       "detail": source_syntax},
      {"name": "code_volume_ceiling_is_preserved",
       "pass": code_volume["passed"], "detail": code_volume},
      {"name": "source_gate_executed_no_model_component_or_token",
       "pass": (
           "/home/intel/models" not in " ".join(shader_commands)
           and "vkCmdDispatch" not in syntax_command)},
      {"name": "remote_source_staging_cleaned",
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
          "header": _rel(args.header),
          "header_sha256": _sha256(args.header),
          "source": _rel(args.source),
          "source_sha256": _sha256(args.source),
          "postconv_shader": _rel(args.postconv_shader),
          "postconv_shader_sha256": _sha256(args.postconv_shader),
          "recurrent_shader": _rel(args.recurrent_shader),
          "recurrent_shader_sha256": _sha256(args.recurrent_shader),
          "cmake": _rel(args.cmake),
          "host": args.host,
          "env_script": args.env_script,
          "glslc_package_uri": GLSLC_PACKAGE_URI,
          "glslc_package_sha512": GLSLC_PACKAGE_SHA512,
          "shaderc_package_uri": SHADERC_PACKAGE_URI,
          "shaderc_package_sha512": SHADERC_PACKAGE_SHA512,
      },
      "spirv": identities,
      "checks": checks,
      "required_checks_passed": required,
      "component_source_passed": required,
      "target_compile_allowed": required,
      "component_probe_allowed": False,
      "decode_integration_allowed": False,
      "token_row_allowed": False,
      "validation_or_test_allowed": False,
      "speedup_claims_allowed": False,
      "promotion_allowed": False,
      "disposition": (
          "accept_vulkan_precise_postconv_recurrent_v1_source"
          if required else "repair_vulkan_precise_postconv_recurrent_v1_source"),
      "selected_next_route": (
          SELECTED_NEXT_ROUTE if required else CURRENT_ROUTE),
      "next_route_reason": (
          "Target-compile one fresh component harness from these exact source "
          "and SPIR-V identities without execution or model access."
          if required else
          "Repair API/runtime/shader structure, SPIR-V audit, syntax, or "
          "hygiene before any target binary or component execution."),
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
          "spirv": metrics["spirv"],
          "component_source_passed": metrics["component_source_passed"],
          "selected_next_route": metrics["selected_next_route"],
          "component_probe_allowed": False,
          "token_row_allowed": False,
      }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  failed = [row["name"] for row in metrics["checks"] if not row["pass"]]
  lines = [
      f"# Seq{metrics['sequence']} Native Vulkan Component Source",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- postconv SPIR-V: `{metrics['spirv']['iq36_postconv_cpuorder']['sha256']}`",
      f"- recurrent SPIR-V: `{metrics['spirv']['iq36_delta_recurrent_cpuorder']['sha256']}`",
      f"- failed_checks: `{failed}`",
      "",
      metrics["next_route_reason"],
      "",
      "Only compiler/syntax commands ran. No Vulkan component, model, decode, or token executed.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--sequence", type=int, default=609)
  parser.add_argument("--routes", type=Path,
                      default=ACTIVE / "routes-ledger.json")
  parser.add_argument(
      "--predecessor", type=Path,
      default=ROOT / (
          "output/seq608-gpu-vulkan-postconv-recurrent-component-design-gate-"
          "20260710Tseq608Z/metrics.json"))
  parser.add_argument(
      "--header", type=Path,
      default=ROOT / (
          "engine/include/intel_qwen36/gpu_vulkan_postconv_recurrent.hpp"))
  parser.add_argument(
      "--source", type=Path,
      default=ROOT / "engine/src/gpu_vulkan_postconv_recurrent.cpp")
  parser.add_argument(
      "--postconv-shader", type=Path,
      default=ROOT / "engine/gpu/vulkan/iq36_postconv_cpuorder.comp")
  parser.add_argument(
      "--recurrent-shader", type=Path,
      default=ROOT / "engine/gpu/vulkan/iq36_delta_recurrent_cpuorder.comp")
  parser.add_argument("--cmake", type=Path,
                      default=ROOT / "engine/CMakeLists.txt")
  parser.add_argument("--target", "--host", dest="host", metavar="TARGET", default=DEFAULT_HOST)
  parser.add_argument("--env-script", default=DEFAULT_ENV_SCRIPT)
  parser.add_argument("--staging-root", "--remote-root", dest="remote_root", metavar="PATH", default=DEFAULT_REMOTE_ROOT)
  parser.add_argument("--timeout-s", type=int, default=600)
  parser.add_argument("--cxx", default="clang++")
  parser.add_argument(
      "--out-dir", type=Path,
      default=ROOT / (
          "output/seq609-gpu-vulkan-postconv-recurrent-component-source-gate-"
          "20260710Tseq609Z"))
  args = parser.parse_args()
  metrics = compute(args)
  write_outputs(metrics, args.out_dir)
  print(json.dumps({
      "required_checks_passed": metrics["required_checks_passed"],
      "disposition": metrics["disposition"],
      "component_source_passed": metrics["component_source_passed"],
      "target_compile_allowed": metrics["target_compile_allowed"],
      "selected_next_route": metrics["selected_next_route"],
      "out_dir": _rel(args.out_dir),
  }, sort_keys=True))
  return 0 if metrics["required_checks_passed"] else 2


if __name__ == "__main__":
  raise SystemExit(main())
