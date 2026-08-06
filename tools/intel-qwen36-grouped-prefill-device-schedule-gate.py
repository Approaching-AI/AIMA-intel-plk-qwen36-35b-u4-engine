#!/usr/bin/env python3
"""Gate the native router/top-8 and device-resident grouped-prefill schedule."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "intel-qwen36-grouped-prefill-device-schedule-gate-v4"
KERNEL = ROOT / "engine/gpu/opencl/grouped_s8_u4_f16_contribution_moe.cl"
SOURCE = ROOT / "engine/tools/grouped_prefill_device_schedule_probe.cpp"
CODEGEN_SOURCE = ROOT / "engine/tools/onednn_router_prefill_probe.cpp"
PERSISTENT_CODEGEN_SOURCE = (
    ROOT / "engine/tools/onednn_grouped_q4k_moe_component.cpp")
GGUF_SOURCE = ROOT / "engine/src/gguf_loader.cpp"
BOUNDARIES = ROOT / "engine/boundaries.json"
ONEDNN_PATCH = ROOT / "engine/gpu/opencl/onednn-grouped-s8-u4-fused.patch"
DEFAULT_MODEL = Path(
    "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf")
DEFAULT_HIDDEN = (
    ROOT / "output/onednn-q4k-routed-moe-component-gate-"
    "20260711Tseq646cleanZ/raw/capture/payloads/"
    "attn_post_norm-27__tok1023__ord0.bin")
DEFAULT_TOPK = (
    ROOT / "output/grouped-s8-u4-prefill-gate-20260711Tseq673cleanZ/raw/"
    "schedule-probes/layer-27.topk.i32")
DEFAULT_ROUTER = (
    ROOT / "output/onednn-q4k-routed-moe-component-gate-"
    "20260711Tseq646cleanZ/raw/capture/payloads/"
    "ffn_moe_weights_norm-27__tok1023__ord2.bin")
DEFAULT_ENV = Path(
    "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh")
DEFAULT_CMAKE = Path("/home/intel/intel-box-env/conda/bin/cmake")
DEFAULT_CXX = Path("/home/intel/intel-box-env/conda/bin/g++")
DEFAULT_ONEDNN_SOURCE = Path(
    "/home/intel/intel-qwen36-r0/source/"
    "oneDNN-01b479323f794da1a7a41a6fc084c7e11ccc2c3b")
DEFAULT_ONEDNN_BUILD = Path(
    "/home/intel/intel-qwen36-r0/build/onednn-01b479-ocl-grouped")
ONEDNN_COMMIT = "01b479323f794da1a7a41a6fc084c7e11ccc2c3b"
PATCHED_ONEDNN_PATHS = [
    "src/gpu/intel/matmul/grouped_micro_gemm.cl",
    "src/gpu/intel/matmul/grouped_micro_gemm.cpp",
    "src/gpu/intel/ocl/engine.cpp",
    "src/gpu/intel/ocl/kernel.cpp",
]
TARGET = "iq36-grouped-prefill-device-schedule-probe"
ROUTER_TARGET = "iq36-native-router-prefill-probe"
RUNTIME_TARGET = "iq36-grouped-s8-u4-prefill-resident-smoke"
CAP_US = 60.0
ROUTER_CAP_US = 500.0
RUNTIME_CAP_US = 9_526.177
RUNTIME_SCHEDULE_CAP_US = 100.0
NATIVE_RUNTIME_CAP_US = RUNTIME_CAP_US + ROUTER_CAP_US
NATIVE_ROUTER_SCHEDULE_CAP_US = ROUTER_CAP_US + RUNTIME_SCHEDULE_CAP_US
PREFILL_ARTIFACT = (
    ROOT / "output/grouped-s8-u4-prefill-gate-20260711Tseq673cleanZ/raw")
CAPTURE_PAYLOADS = (
    ROOT / "output/onednn-q4k-routed-moe-component-gate-"
    "20260711Tseq646cleanZ/raw/capture/payloads")


def iso_now() -> str:
  return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for block in iter(lambda: stream.read(1 << 20), b""):
      digest.update(block)
  return digest.hexdigest()


def run(command: list[str], timeout_s: int, cwd: Path = ROOT,
        env: dict[str, str] | None = None) -> dict[str, Any]:
  try:
    completed = subprocess.run(
        command, cwd=cwd, env=env, text=True, capture_output=True,
        timeout=timeout_s, check=False)
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "timed_out": False,
    }
  except subprocess.TimeoutExpired as error:
    return {
        "command": command,
        "returncode": 124,
        "stdout": error.stdout or "",
        "stderr": error.stderr or "",
        "timed_out": True,
    }


def run_intel(command: list[str], args: argparse.Namespace,
              cwd: Path = ROOT,
              extra_env: dict[str, str] | None = None) -> dict[str, Any]:
  exports = ["export INTEL_FORCE_PROBE=b080"]
  for key, value in sorted((extra_env or {}).items()):
    exports.append(f"export {key}={shlex.quote(value)}")
  shell = (
      f"source {shlex.quote(str(args.env_script))} >/dev/null 2>&1 && "
      + " && ".join(exports) + " && " + shlex.join(command))
  return run(["bash", "-lc", shell], args.timeout_s, cwd=cwd)


def write_run(raw: Path, label: str, result: dict[str, Any]) -> None:
  (raw / f"{label}.command.json").write_text(
      json.dumps(result["command"], indent=2) + "\n", encoding="utf-8")
  (raw / f"{label}.stdout").write_text(
      str(result.get("stdout", "")), encoding="utf-8")
  (raw / f"{label}.stderr").write_text(
      str(result.get("stderr", "")), encoding="utf-8")


def parse_json_line(result: dict[str, Any]) -> dict[str, Any]:
  lines = [line for line in str(result.get("stdout", "")).splitlines()
           if line.strip()]
  if not lines:
    return {}
  try:
    value = json.loads(lines[-1])
  except json.JSONDecodeError:
    return {}
  return value if isinstance(value, dict) else {}


def git_state() -> dict[str, Any]:
  commit = run(["git", "rev-parse", "HEAD"], 30)
  status = run(["git", "status", "--short"], 30)
  dirty_paths = [line for line in str(status["stdout"]).splitlines()
                 if line.strip()]
  return {
      "commit": str(commit["stdout"]).strip(),
      "dirty": bool(dirty_paths),
      "dirty_paths": dirty_paths,
  }


def check(name: str, passed: bool, **details: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **details}


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
  parser.add_argument("--hidden", type=Path, default=DEFAULT_HIDDEN)
  parser.add_argument("--topk", type=Path, default=DEFAULT_TOPK)
  parser.add_argument("--router", type=Path, default=DEFAULT_ROUTER)
  parser.add_argument("--kernel", type=Path, default=KERNEL)
  parser.add_argument("--build-dir", type=Path, default=ROOT / "build/engine")
  parser.add_argument("--env-script", type=Path, default=DEFAULT_ENV)
  parser.add_argument("--cmake", type=Path, default=DEFAULT_CMAKE)
  parser.add_argument("--cxx", type=Path, default=DEFAULT_CXX)
  parser.add_argument("--onednn-source", type=Path,
                      default=DEFAULT_ONEDNN_SOURCE)
  parser.add_argument("--onednn-build", type=Path,
                      default=DEFAULT_ONEDNN_BUILD)
  parser.add_argument("--warmup", type=int, default=5)
  parser.add_argument("--repeat", type=int, default=21)
  parser.add_argument("--timeout-s", type=int, default=600)
  parser.add_argument("--out-dir", type=Path)
  args = parser.parse_args()
  if args.warmup < 0 or args.repeat < 3 or args.timeout_s <= 0:
    parser.error("warmup/repeat/timeout are invalid")
  if args.out_dir is None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.out_dir = ROOT / f"output/grouped-prefill-device-schedule-gate-{stamp}"
  return args


def main() -> int:
  args = parse_args()
  out_dir = args.out_dir.resolve()
  raw = out_dir / "raw"
  raw.mkdir(parents=True, exist_ok=False)
  required = [args.model, args.hidden, args.topk, args.router, args.kernel,
              args.env_script, args.cmake, args.cxx, args.onednn_source,
              args.onednn_build, CODEGEN_SOURCE, GGUF_SOURCE, SOURCE,
              BOUNDARIES, ONEDNN_PATCH, ROOT / "engine/CMakeLists.txt",
              PERSISTENT_CODEGEN_SOURCE,
              args.onednn_build / "src/libdnnl.so",
              args.onednn_build / "include/oneapi/dnnl/dnnl_config.h",
              args.onednn_source / "include/oneapi/dnnl/dnnl.hpp",
              PREFILL_ARTIFACT / "prepacked",
              PREFILL_ARTIFACT / "gateup.0.bin",
              PREFILL_ARTIFACT / "down.0.bin",
              CAPTURE_PAYLOADS / "attn_post_norm-27__tok1023__ord0.bin",
              CAPTURE_PAYLOADS / "ffn_moe_topk-27__tok1023__ord1.bin",
              CAPTURE_PAYLOADS / "ffn_moe_weights_norm-27__tok1023__ord2.bin",
              CAPTURE_PAYLOADS / "ffn_moe_swiglu-27__tok1023__ord3.bin",
              CAPTURE_PAYLOADS / "ffn_moe_down-27__tok1023__ord4.bin",
              CAPTURE_PAYLOADS / "ffn_moe_out-27__tok1023__ord5.bin"]
  missing = [str(path) for path in required if not path.exists()]
  if missing:
    raise SystemExit("missing inputs: " + ", ".join(missing))
  if args.topk.stat().st_size != 8192 * 4:
    raise SystemExit("top-k input must contain 8192 int32 values")
  if args.router.stat().st_size != 8192 * 4:
    raise SystemExit("router input must contain 8192 float32 values")
  if args.hidden.stat().st_size != 1024 * 2048 * 4:
    raise SystemExit("router hidden input must contain 1024x2048 float32")

  created_at = iso_now()
  git = git_state()
  onednn_commit_run = run(
      ["git", "rev-parse", "HEAD"], 30, cwd=args.onednn_source)
  onednn_commit = str(onednn_commit_run["stdout"]).strip()
  onednn_diff_run = subprocess.run(
      ["git", "diff", "--unified=0", "--", *PATCHED_ONEDNN_PATHS],
      cwd=args.onednn_source, check=False, capture_output=True)
  onednn_status_run = subprocess.run(
      ["git", "status", "--short"], cwd=args.onednn_source, check=False,
      capture_output=True, text=True)
  expected_onednn_status = sorted(
      f" M {path}" for path in PATCHED_ONEDNN_PATHS)
  onednn_patch_exact = (
      onednn_diff_run.returncode == 0 and
      onednn_diff_run.stdout == ONEDNN_PATCH.read_bytes() and
      onednn_status_run.returncode == 0 and
      sorted(onednn_status_run.stdout.splitlines()) == expected_onednn_status)
  configure = run([
      str(args.cmake), "-S", str(ROOT / "engine"), "-B",
      str(args.build_dir)], args.timeout_s)
  write_run(raw, "configure", configure)
  build = run([
      str(args.cmake), "--build", str(args.build_dir), "--target", TARGET,
      ROUTER_TARGET, RUNTIME_TARGET, "-j16"], args.timeout_s)
  write_run(raw, "build", build)
  binary = args.build_dir / TARGET
  router_binary = args.build_dir / ROUTER_TARGET
  runtime_binary = args.build_dir / RUNTIME_TARGET

  onednn_build = run_intel([
      str(args.cmake), "--build", str(args.onednn_build), "--target", "dnnl",
      "-j16"], args)
  write_run(raw, "onednn-build", onednn_build)
  persistent_codegen_binary = raw / "onednn-grouped-persistent-codegen"
  persistent_codegen_build_command = [
      str(args.cxx), "-std=gnu++17", "-O3", "-DNDEBUG",
      "-DCL_TARGET_OPENCL_VERSION=300",
      f"-I{args.onednn_build / 'include'}",
      f"-I{args.onednn_source / 'include'}",
      str(PERSISTENT_CODEGEN_SOURCE),
      f"-L{args.onednn_build / 'src'}",
      f"-Wl,-rpath,{args.onednn_build / 'src'}", "-ldnnl", "-lOpenCL",
      "-o", str(persistent_codegen_binary),
  ]
  persistent_codegen_build = (
      run_intel(persistent_codegen_build_command, args)
      if onednn_build["returncode"] == 0 else {
          "command": persistent_codegen_build_command, "returncode": 125,
          "stdout": "", "stderr": "oneDNN build failed",
          "timed_out": False})
  write_run(raw, "persistent-codegen-build", persistent_codegen_build)
  persistent_codegen_command = [
      str(persistent_codegen_binary), "--model", str(args.model),
      "--weight-offset", "14585674336", "--weight-bytes", "301989888",
      "--input", str(
          CAPTURE_PAYLOADS / "attn_post_norm-27__tok1023__ord0.bin"),
      "--topk", str(
          CAPTURE_PAYLOADS / "ffn_moe_topk-27__tok1023__ord1.bin"),
      "--topk-stride", "1024", "--oracle", str(
          CAPTURE_PAYLOADS / "ffn_moe_swiglu-27__tok1023__ord3.bin"),
      "--down-weight-offset", "14431394400", "--down-weight-bytes",
      "150994944", "--router-weights", str(
          CAPTURE_PAYLOADS / "ffn_moe_weights_norm-27__tok1023__ord2.bin"),
      "--down-oracle", str(
          CAPTURE_PAYLOADS / "ffn_moe_down-27__tok1023__ord4.bin"),
      "--moe-oracle", str(
          CAPTURE_PAYLOADS / "ffn_moe_out-27__tok1023__ord5.bin"),
      "--warmup", "1", "--repeat", "1", "--kernel-cap-us", "9771.436",
  ]
  persistent_binaries: dict[str, Path] = {}
  persistent_generation: dict[str, dict[str, Any]] = {}
  persistent_disassembly: dict[str, dict[str, Any]] = {}
  for kind in ("gateup", "down"):
    prefix = raw / f"persistent-{kind}"
    generation = (
        run_intel(persistent_codegen_command, args, extra_env={
            "DNNL_PRIMITIVE_CACHE_CAPACITY": "0",
            "DNNL_VERBOSE": "0",
            "IQ36_GENERATE_S8_GROUPED": "1",
            "IQ36_GROUPED_FUSED_KIND": kind,
            "IQ36_GROUPED_PERSISTENT_DISPATCH": "1",
            "IQ36_DUMP_FUSED_PROGRAM_PREFIX": str(prefix),
            "IQ36_EXIT_AFTER_FUSED_DUMP": "1",
        }) if persistent_codegen_build["returncode"] == 0 else {
            "command": persistent_codegen_command, "returncode": 125,
            "stdout": "", "stderr": "persistent codegen build failed",
            "timed_out": False})
    write_run(raw, f"persistent-{kind}-generate", generation)
    persistent_generation[kind] = generation
    persistent_binary = raw / f"persistent-{kind}.0.bin"
    persistent_binaries[kind] = persistent_binary
    disasm_dir = raw / f"persistent-{kind}-disasm"
    disasm_command = [
        "ocloc", "disasm", "-file", str(persistent_binary), "-dump",
        str(disasm_dir), "-device", "0xb080",
    ]
    disassembly = (
        run_intel(disasm_command, args)
        if persistent_binary.exists() else {
            "command": disasm_command, "returncode": 125, "stdout": "",
            "stderr": "persistent binary was not generated",
            "timed_out": False})
    write_run(raw, f"persistent-{kind}-disasm", disassembly)
    build_options_path = disasm_dir / ".misc.buildOptions"
    ze_info_path = disasm_dir / ".ze_info"
    persistent_disassembly[kind] = {
        "run": disassembly,
        "build_options": (
            build_options_path.read_text(encoding="utf-8", errors="replace")
            if build_options_path.exists() else ""),
        "ze_info": (
            ze_info_path.read_text(encoding="utf-8", errors="replace")
            if ze_info_path.exists() else ""),
    }
  codegen_binary = raw / "onednn-router-prefill-probe"
  codegen_build_command = [
      str(args.cxx), "-std=gnu++17", "-O3", "-DNDEBUG",
      "-DCL_TARGET_OPENCL_VERSION=300", f"-I{ROOT / 'engine/include'}",
      f"-I{args.onednn_build / 'include'}",
      f"-I{args.onednn_source / 'include'}", str(CODEGEN_SOURCE),
      str(GGUF_SOURCE), f"-L{args.onednn_build / 'src'}",
      f"-Wl,-rpath,{args.onednn_build / 'src'}", "-ldnnl", "-lOpenCL",
      "-o", str(codegen_binary),
  ]
  codegen_build = (
      run_intel(codegen_build_command, args)
      if onednn_build["returncode"] == 0 else {
          "command": codegen_build_command, "returncode": 125,
          "stdout": "", "stderr": "oneDNN build failed",
          "timed_out": False})
  write_run(raw, "router-codegen-build", codegen_build)
  codegen_dir = raw / "router-codegen"
  codegen_dir.mkdir()
  router_weights_file = raw / "router-layer27-f32.bin"
  codegen_command = [
      str(codegen_binary), str(args.model), "27", str(args.hidden),
      str(args.topk), str(args.router), str(ROUTER_CAP_US),
      str(args.warmup), str(args.repeat), str(router_weights_file),
  ]
  codegen_run = (
      run_intel(codegen_command, args, cwd=codegen_dir, extra_env={
          "DNNL_JIT_DUMP": "1", "DNNL_PRIMITIVE_CACHE_CAPACITY": "0"})
      if codegen_build["returncode"] == 0 else {
          "command": codegen_command, "returncode": 125, "stdout": "",
          "stderr": "router codegen build failed", "timed_out": False})
  write_run(raw, "router-codegen", codegen_run)
  codegen_result = parse_json_line(codegen_run)
  native_programs = sorted(
      codegen_dir.glob("dnnl_dump_gpu_gemm_kernel_program.*.bin"))
  native_program = native_programs[0] if len(native_programs) == 1 else None
  disasm_dir = raw / "router-native-disasm"
  disasm_command = [
      "ocloc", "disasm", "-file",
      str(native_program) if native_program else "-", "-dump",
      str(disasm_dir), "-device", "0xb080",
  ]
  disasm = (
      run_intel(disasm_command, args)
      if native_program is not None else {
          "command": disasm_command, "returncode": 125, "stdout": "",
          "stderr": "native router program was not generated",
          "timed_out": False})
  write_run(raw, "router-native-disasm", disasm)
  ze_info_path = disasm_dir / ".ze_info"
  assembly_paths = list(disasm_dir.glob("*.asm")) if disasm_dir.exists() else []
  ze_info = (ze_info_path.read_text(encoding="utf-8", errors="replace")
             if ze_info_path.exists() else "")
  assembly = "\n".join(
      path.read_text(encoding="utf-8", errors="replace")
      for path in assembly_paths)
  ldd = (run(["ldd", str(router_binary)], args.timeout_s)
         if router_binary.exists() else {
      "command": ["ldd", str(router_binary)], "returncode": 125,
      "stdout": "",
      "stderr": "native probe binary missing", "timed_out": False})
  write_run(raw, "native-probe-ldd", ldd)

  probes: dict[str, dict[str, Any]] = {}
  for label in ("repeat", "confirm"):
    if configure["returncode"] == 0 and build["returncode"] == 0 and binary.exists():
      probe_run = run_intel([
          str(binary), str(args.kernel), str(args.topk), str(args.router),
          str(args.warmup), str(args.repeat)], args)
    else:
      probe_run = {
          "command": [], "returncode": 125, "stdout": "",
          "stderr": "build prerequisite failed", "timed_out": False}
    write_run(raw, label, probe_run)
    probes[label] = {
        "run": {key: probe_run[key] for key in
                ("command", "returncode", "timed_out")},
        "result": parse_json_line(probe_run),
    }

  router_probes: dict[str, dict[str, Any]] = {}
  for label in ("router-repeat", "router-confirm"):
    if (configure["returncode"] == 0 and build["returncode"] == 0 and
        router_binary.exists() and native_program is not None):
      router_run = run_intel([
          str(router_binary), "--router", str(args.model), "27",
          str(native_program), str(args.kernel), str(args.hidden),
          str(args.topk), str(args.router), str(args.warmup),
          str(args.repeat)], args)
    else:
      router_run = {
          "command": [], "returncode": 125, "stdout": "",
          "stderr": "native router prerequisite failed", "timed_out": False}
    write_run(raw, label, router_run)
    router_probes[label] = {
        "run": {key: router_run[key] for key in
                ("command", "returncode", "timed_out")},
        "result": parse_json_line(router_run),
    }

  runtime_probes: dict[str, dict[str, Any]] = {}
  runtime_command = [
      str(runtime_binary), str(PREFILL_ARTIFACT / "prepacked"),
      str(PREFILL_ARTIFACT / "gateup.0.bin"),
      str(PREFILL_ARTIFACT / "down.0.bin"), str(args.kernel),
      str(CAPTURE_PAYLOADS / "attn_post_norm-27__tok1023__ord0.bin"),
      str(CAPTURE_PAYLOADS / "ffn_moe_topk-27__tok1023__ord1.bin"),
      "1024",
      str(CAPTURE_PAYLOADS / "ffn_moe_weights_norm-27__tok1023__ord2.bin"),
      str(CAPTURE_PAYLOADS / "ffn_moe_out-27__tok1023__ord5.bin"),
      "--device-schedule",
  ]
  for label in ("runtime-repeat", "runtime-confirm"):
    if (configure["returncode"] == 0 and build["returncode"] == 0 and
        runtime_binary.exists()):
      runtime_run = run_intel(runtime_command, args)
    else:
      runtime_run = {
          "command": [], "returncode": 125, "stdout": "",
          "stderr": "build prerequisite failed", "timed_out": False}
    write_run(raw, label, runtime_run)
    runtime_probes[label] = {
        "run": {key: runtime_run[key] for key in
                ("command", "returncode", "timed_out")},
        "result": parse_json_line(runtime_run),
    }

  native_runtime_probes: dict[str, dict[str, Any]] = {}
  native_runtime_command = list(runtime_command)
  native_runtime_command[2] = str(persistent_binaries["gateup"])
  native_runtime_command[3] = str(persistent_binaries["down"])
  native_runtime_command.extend([
      "--native-router", str(native_program) if native_program else "-",
      str(router_weights_file), "--persistent-dispatch",
  ])
  for label in ("native-runtime-repeat", "native-runtime-confirm"):
    if (configure["returncode"] == 0 and build["returncode"] == 0 and
        runtime_binary.exists() and native_program is not None and
        router_weights_file.exists() and
        all(path.exists() for path in persistent_binaries.values())):
      native_runtime_run = run_intel(native_runtime_command, args)
    else:
      native_runtime_run = {
          "command": [], "returncode": 125, "stdout": "",
          "stderr": "native runtime prerequisite failed",
          "timed_out": False}
    write_run(raw, label, native_runtime_run)
    native_runtime_probes[label] = {
        "run": {key: native_runtime_run[key] for key in
                ("command", "returncode", "timed_out")},
        "result": parse_json_line(native_runtime_run),
    }

  checks = [
      check("repository_clean_at_gate", not git["dirty"],
            dirty_paths=git["dirty_paths"]),
      check("pinned_onednn_source_commit", onednn_commit == ONEDNN_COMMIT,
            observed=onednn_commit, expected=ONEDNN_COMMIT),
      check("onednn_source_diff_exactly_matches_repo_patch",
            onednn_patch_exact,
            dirty_paths=onednn_status_run.stdout.splitlines()),
      check("onednn_codegen_library_builds",
            onednn_build["returncode"] == 0),
      check("persistent_grouped_codegen_probe_builds",
            persistent_codegen_build["returncode"] == 0),
      check("router_codegen_probe_builds",
            codegen_build["returncode"] == 0),
      check("router_codegen_probe_passes",
            codegen_run["returncode"] == 0 and
            codegen_result.get("required_checks_passed") is True and
            codegen_result.get("set_top8_match_rows") == 1024 and
            codegen_result.get("missing_expert_count") == 0 and
            isinstance(codegen_result.get("router_median_us"), (int, float))
            and float(codegen_result["router_median_us"]) <= ROUTER_CAP_US,
            observed_us=codegen_result.get("router_median_us"),
            cap_us=ROUTER_CAP_US),
      check("router_f32_weights_materialized",
            router_weights_file.exists() and
            router_weights_file.stat().st_size == 256 * 2048 * 4,
            observed_bytes=(router_weights_file.stat().st_size
                            if router_weights_file.exists() else None)),
      check("single_native_router_program_generated",
            native_program is not None,
            generated=[path.name for path in native_programs]),
      check("native_router_program_disassembles_for_ptl",
            disasm["returncode"] == 0 and 'name: "gemm_kernel"' in ze_info and
            "grf_count: 256" in ze_info and "simd_size: 16" in ze_info and
            "load_block2d" in assembly and "mad (32|" in assembly),
      check("native_probe_links_no_onednn_or_openvino",
            ldd["returncode"] == 0 and
            "dnnl" not in str(ldd["stdout"]).lower() and
            "openvino" not in str(ldd["stdout"]).lower()),
      check("probe_target_configures", configure["returncode"] == 0),
      check("probe_target_builds", build["returncode"] == 0),
  ]
  for kind in ("gateup", "down"):
    binary_path = persistent_binaries[kind]
    disassembly = persistent_disassembly[kind]
    checks.extend([
        check(f"persistent_{kind}_program_generated",
              persistent_generation[kind]["returncode"] == 0 and
              binary_path.exists() and binary_path.stat().st_size > 0,
              bytes=(binary_path.stat().st_size
                     if binary_path.exists() else None)),
        check(f"persistent_{kind}_program_disassembles_for_ptl",
              disassembly["run"]["returncode"] == 0 and
              "-DIQ36_PERSISTENT_DISPATCH=1" in
                  disassembly["build_options"] and
              "required_work_group_size: [ 32, 4, 1 ]" in
                  disassembly["ze_info"]),
    ])
  for label, probe in probes.items():
    row = probe["result"]
    checks.extend([
        check(f"{label}_returns_success", probe["run"]["returncode"] == 0),
        check(f"{label}_schedule_semantics_exact",
              row.get("exact_schedule_match") is True and
              row.get("semantic_mismatch_count") == 0 and
              row.get("metadata_error_bitmap") == 0),
        check(f"{label}_native_logical_tiles_exact",
              row.get("native_gateup_task_count") == 6448 and
              row.get("native_down_task_count") == 12896 and
              row.get("native_gateup_task_mismatch_count") == 0 and
              row.get("native_down_task_mismatch_count") == 0),
        check(f"{label}_device_schedule_at_most_60_us",
              isinstance(row.get("device_median_us"), (int, float)) and
              float(row["device_median_us"]) <= CAP_US,
              observed_us=row.get("device_median_us"), cap_us=CAP_US),
    ])
  for label, probe in router_probes.items():
    row = probe["result"]
    checks.extend([
        check(f"{label}_returns_success", probe["run"]["returncode"] == 0),
        check(f"{label}_top8_set_exact",
              row.get("set_top8_match_rows") == 1024 and
              row.get("missing_expert_count") == 0),
        check(f"{label}_router_weights_within_0p002",
              isinstance(row.get("maximum_router_weight_abs_diff"),
                         (int, float)) and
              float(row["maximum_router_weight_abs_diff"]) <= 0.002,
              observed=row.get("maximum_router_weight_abs_diff"),
              cap=0.002),
        check(f"{label}_native_router_top8_at_most_500_us",
              isinstance(row.get("router_median_us"), (int, float)) and
              float(row["router_median_us"]) <= ROUTER_CAP_US,
              observed_us=row.get("router_median_us"), cap_us=ROUTER_CAP_US),
    ])
  for label, probe in runtime_probes.items():
    row = probe["result"]
    checks.extend([
        check(f"{label}_returns_success", probe["run"]["returncode"] == 0),
        check(f"{label}_resident_output_exact",
              row.get("resident_reuse_pass") is True and
              row.get("oracle_mismatch_count") == 0 and
              row.get("deterministic_mismatch_count") == 0),
        check(f"{label}_device_schedule_is_active",
              row.get("device_schedule") is True and
              row.get("device_schedule_run_count") == 2),
        check(f"{label}_bulk_schedule_transfer_removed",
              row.get("device_schedule_host_upload_bytes") == 131_072 and
              row.get("device_schedule_host_read_bytes") == 40),
        check(f"{label}_resident_component_under_cap",
              isinstance(row.get("second_complete_minimum_us"), (int, float))
              and float(row["second_complete_minimum_us"]) <= RUNTIME_CAP_US,
              observed_us=row.get("second_complete_minimum_us"),
              cap_us=RUNTIME_CAP_US),
        check(f"{label}_integrated_schedule_under_100_us",
              isinstance(row.get("device_schedule_us"), (int, float)) and
              float(row["device_schedule_us"]) <= RUNTIME_SCHEDULE_CAP_US,
              observed_us=row.get("device_schedule_us"),
              cap_us=RUNTIME_SCHEDULE_CAP_US),
    ])
  for label, probe in native_runtime_probes.items():
    row = probe["result"]
    checks.extend([
        check(f"{label}_returns_success", probe["run"]["returncode"] == 0),
        check(f"{label}_resident_output_exact",
              row.get("resident_reuse_pass") is True and
              row.get("oracle_mismatch_count") == 0 and
              row.get("deterministic_mismatch_count") == 0),
        check(f"{label}_native_router_is_active",
              row.get("device_schedule") is True and
              row.get("native_router") is True and
              row.get("device_schedule_run_count") == 2 and
              row.get("native_router_run_count") == 2),
        check(f"{label}_persistent_dispatch_is_active",
              row.get("persistent_dispatch") is True and
              row.get("persistent_dispatch_run_count") == 2 and
              row.get("persistent_workgroup_count") == 96),
        check(f"{label}_router_schedule_transfer_removed",
              row.get("device_schedule_host_upload_bytes") == 0 and
              row.get("device_schedule_host_read_bytes") == 0),
        check(f"{label}_native_program_and_weights_resident",
              row.get("program_load_count") == 4 and
              row.get("resident_weight_bytes") == 543_162_368 and
              row.get("maps_native_only") is True),
        check(f"{label}_integrated_component_under_registered_cap",
              isinstance(row.get("second_complete_minimum_us"), (int, float))
              and float(row["second_complete_minimum_us"])
                  <= NATIVE_RUNTIME_CAP_US,
              observed_us=row.get("second_complete_minimum_us"),
              cap_us=NATIVE_RUNTIME_CAP_US),
        check(f"{label}_native_router_and_schedule_under_600_us",
              isinstance(row.get("device_schedule_us"), (int, float)) and
              float(row["device_schedule_us"])
                  <= NATIVE_ROUTER_SCHEDULE_CAP_US,
              observed_us=row.get("device_schedule_us"),
              cap_us=NATIVE_ROUTER_SCHEDULE_CAP_US),
    ])
  required_checks_passed = all(item["pass"] for item in checks)
  disposition = (
      "accept_zero_readback_persistent_native_grouped_dispatch"
      if required_checks_passed
      else "reject_or_repair_native_router_or_device_schedule")
  result = {
      "schema_version": SCHEMA,
      "created_at": created_at,
      "git": git,
      "inputs": {
          "model": str(args.model.resolve()),
          "model_sha256": sha256(args.model),
          "hidden": str(args.hidden.resolve()),
          "hidden_sha256": sha256(args.hidden),
          "topk": str(args.topk.resolve()),
          "topk_sha256": sha256(args.topk),
          "router": str(args.router.resolve()),
          "router_sha256": sha256(args.router),
          "kernel": str(args.kernel.resolve()),
          "kernel_sha256": sha256(args.kernel),
          "native_router_program": (
              str(native_program.resolve()) if native_program else None),
          "native_router_program_sha256": (
              sha256(native_program) if native_program else None),
          "native_router_weights": (
              str(router_weights_file.resolve())
              if router_weights_file.exists() else None),
          "native_router_weights_sha256": (
              sha256(router_weights_file)
              if router_weights_file.exists() else None),
          "persistent_gateup_program": str(
              persistent_binaries["gateup"].resolve()),
          "persistent_gateup_program_sha256": (
              sha256(persistent_binaries["gateup"])
              if persistent_binaries["gateup"].exists() else None),
          "persistent_down_program": str(
              persistent_binaries["down"].resolve()),
          "persistent_down_program_sha256": (
              sha256(persistent_binaries["down"])
              if persistent_binaries["down"].exists() else None),
          "onednn_codegen_commit": onednn_commit,
          "onednn_codegen_patch_sha256": sha256(ONEDNN_PATCH),
          "case_id": "prefill_shape_008k_tile1024_layer27_routed",
      },
      "budget": {
          "native_router_top8_cap_us": ROUTER_CAP_US,
          "native_router_top8_all40_cap_us": ROUTER_CAP_US * 40,
          "integrated_native_runtime_cap_us": NATIVE_RUNTIME_CAP_US,
          "integrated_native_router_schedule_cap_us":
              NATIVE_ROUTER_SCHEDULE_CAP_US,
          "device_schedule_cap_us": CAP_US,
          "all40_cap_us": CAP_US * 40,
          "persistent_physical_workgroups": 96,
          "basis": (
              "at most 2.4 ms across 40 layers, below half of the 6.165 ms "
              "confirm headroom in the accepted mixed grouped-MoE window; "
              "one fixed physical workgroup per measured OpenCL compute unit"),
      },
      "codegen_probe": codegen_result,
      "probes": probes,
      "router_probes": router_probes,
      "runtime_probes": runtime_probes,
      "native_runtime_probes": native_runtime_probes,
      "checks": checks,
      "required_checks_passed": required_checks_passed,
      "disposition": disposition,
      "speedup_claims_allowed": False,
      "next_gate": (
          "attach the attention/linear-state tile to the zero-readback "
          "persistent grouped dispatch"
          if required_checks_passed else
          "profile or replace the failed native prefill control route"),
  }
  (out_dir / "result.json").write_text(
      json.dumps(result, indent=2) + "\n", encoding="utf-8")
  repeat = probes["repeat"]["result"]
  confirm = probes["confirm"]["result"]
  router_repeat = router_probes["router-repeat"]["result"]
  router_confirm = router_probes["router-confirm"]["result"]
  runtime_repeat = runtime_probes["runtime-repeat"]["result"]
  runtime_confirm = runtime_probes["runtime-confirm"]["result"]
  native_runtime_repeat = native_runtime_probes[
      "native-runtime-repeat"]["result"]
  native_runtime_confirm = native_runtime_probes[
      "native-runtime-confirm"]["result"]
  summary = [
      "# Persistent native router/top-8 grouped-prefill dispatch gate", "",
      f"- disposition: `{disposition}`",
      f"- commit: `{git['commit']}` (dirty: `{str(git['dirty']).lower()}`)",
      f"- repeat/confirm device median: "
      f"`{repeat.get('device_median_us')} / {confirm.get('device_median_us')} us`",
      f"- repeat/confirm CPU schedule median: "
      f"`{repeat.get('cpu_median_us')} / {confirm.get('cpu_median_us')} us`",
      f"- exact semantic schedule: "
      f"`{repeat.get('exact_schedule_match')} / "
      f"{confirm.get('exact_schedule_match')}`",
      f"- native router/top-8 repeat/confirm median: "
      f"`{router_repeat.get('router_median_us')} / "
      f"{router_confirm.get('router_median_us')} us`",
      f"- native router top-8 set rows: "
      f"`{router_repeat.get('set_top8_match_rows')} / "
      f"{router_confirm.get('set_top8_match_rows')}`",
      f"- native router maximum weight error: "
      f"`{router_repeat.get('maximum_router_weight_abs_diff')} / "
      f"{router_confirm.get('maximum_router_weight_abs_diff')}`", "",
      f"- integrated repeat/confirm complete minimum: "
      f"`{runtime_repeat.get('second_complete_minimum_us')} / "
      f"{runtime_confirm.get('second_complete_minimum_us')} us`",
      f"- integrated repeat/confirm schedule wall: "
      f"`{runtime_repeat.get('device_schedule_us')} / "
      f"{runtime_confirm.get('device_schedule_us')} us`",
      f"- native integrated repeat/confirm complete minimum: "
      f"`{native_runtime_repeat.get('second_complete_minimum_us')} / "
      f"{native_runtime_confirm.get('second_complete_minimum_us')} us`",
      f"- native router+schedule repeat/confirm wall: "
      f"`{native_runtime_repeat.get('device_schedule_us')} / "
      f"{native_runtime_confirm.get('device_schedule_us')} us`",
      f"- native router schedule upload bytes: "
      f"`{native_runtime_repeat.get('device_schedule_host_upload_bytes')} / "
      f"{native_runtime_confirm.get('device_schedule_host_upload_bytes')}`",
      f"- native router schedule read bytes: "
      f"`{native_runtime_repeat.get('device_schedule_host_read_bytes')} / "
      f"{native_runtime_confirm.get('device_schedule_host_read_bytes')}`",
      f"- persistent physical workgroups: "
      f"`{native_runtime_repeat.get('persistent_workgroup_count')} / "
      f"{native_runtime_confirm.get('persistent_workgroup_count')}`",
      "",
      "This is a 1024-token native control-path component gate, not full "
      "prefill throughput or a product speed claim.", "",
  ]
  (out_dir / "summary.md").write_text("\n".join(summary), encoding="utf-8")
  artifact_label = (
      str(out_dir.relative_to(ROOT)) if out_dir.is_relative_to(ROOT)
      else str(out_dir))
  print(json.dumps({
      "artifact": artifact_label,
      "disposition": disposition,
      "required_checks_passed": required_checks_passed,
  }, indent=2))
  return 0 if required_checks_passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
