#!/usr/bin/env python3
"""Compare pinned and current official Gemmstone N=8 attention generation.

M=16 is the bit-exact 256-key-block reopen condition. M=8 is the official
XeHPG thin-Q alternate and only qualifies for a later arithmetic boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CODEGEN = ROOT / "engine/tools/openvino_moe_micro_codegen.cpp"
CXX = Path("/home/intel/intel-box-env/conda/bin/c++")
OPENVINO_SOURCE = Path(
    "/home/intel/intel-qwen36-r0/source/openvino-90214e5be05"
)
DEFAULT_PINNED_SOURCE = Path(
    "/home/intel/intel-qwen36-r0/source/openvino-90214e5be05/"
    "src/plugins/intel_gpu/thirdparty/onednn_gpu"
)
DEFAULT_PINNED_BUILD = Path(
    "/home/intel/intel-qwen36-r0/build/onednn-20db-micro-static"
)
DEFAULT_CURRENT_SOURCE = Path(
    "/home/intel/intel-qwen36-r0/source/oneDNN-978698-n8"
)
DEFAULT_CURRENT_BUILD = Path(
    "/home/intel/intel-qwen36-r0/build/onednn-978698-n8-static-make"
)
PINNED_COMMIT = "20db47e2d3c4df1b66e93bed2e97d30da175512d"


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--out-dir", type=Path, required=True)
  parser.add_argument("--context", type=int, default=131072)
  parser.add_argument("--kq-unroll-m", type=int, choices=(8, 16), default=16)
  parser.add_argument("--timeout-s", type=int, default=600)
  parser.add_argument("--pinned-source", type=Path,
                      default=DEFAULT_PINNED_SOURCE)
  parser.add_argument("--pinned-build", type=Path,
                      default=DEFAULT_PINNED_BUILD)
  parser.add_argument("--current-source", type=Path,
                      default=DEFAULT_CURRENT_SOURCE)
  parser.add_argument("--current-build", type=Path,
                      default=DEFAULT_CURRENT_BUILD)
  return parser.parse_args()


def run(command: list[str], timeout_s: int,
        env: dict[str, str] | None = None) -> dict[str, Any]:
  completed = subprocess.run(
      command, cwd=ROOT, env=env, text=True, capture_output=True,
      timeout=timeout_s, check=False)
  return {
      "command": command,
      "returncode": completed.returncode,
      "stdout": completed.stdout,
      "stderr": completed.stderr,
  }


def git_text(directory: Path, *arguments: str) -> str:
  completed = subprocess.run(
      ["git", "-C", str(directory), *arguments], text=True,
      capture_output=True, check=True)
  return completed.stdout.strip()


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def file_inventory(directory: Path) -> list[dict[str, Any]]:
  if not directory.exists():
    return []
  return [
      {
          "file": str(path.relative_to(ROOT)),
          "bytes": path.stat().st_size,
          "sha256": sha256(path),
      }
      for path in sorted(directory.iterdir())
      if path.is_file()
  ]


def meminfo() -> dict[str, int]:
  values: dict[str, int] = {}
  for line in Path("/proc/meminfo").read_text().splitlines():
    key, value = line.split(":", 1)
    if key in {"MemAvailable", "SwapFree"}:
      values[f"{key}_bytes"] = int(value.split()[0]) * 1024
  return values


def cmake_settings(build: Path) -> dict[str, str]:
  wanted = {
      "CMAKE_BUILD_TYPE", "CMAKE_CXX_COMPILER",
      "ONEDNN_CPU_RUNTIME", "ONEDNN_GPU_RUNTIME",
      "ONEDNN_LIBRARY_TYPE", "ONEDNN_ENABLE_WORKLOAD",
      "ONEDNN_ENABLE_PRIMITIVE", "ONEDNN_ENABLE_PRIMITIVE_GPU_ISA",
  }
  settings: dict[str, str] = {}
  for line in (build / "CMakeCache.txt").read_text().splitlines():
    if not line or line.startswith(("#", "//")) or ":" not in line:
      continue
    key_type, separator, value = line.partition("=")
    key = key_type.partition(":")[0]
    if separator and key in wanted:
      settings[key] = value
  return settings


def build_command(source: Path, build: Path, binary: Path) -> list[str]:
  includes = [
      source / "src/gpu/intel/gemm/jit",
      source / "src/gpu/intel/gemm/jit/dnnl_gpu_intel_gemm_jit",
      build / "include",
      source / "include",
      source / "third_party/opencl",
      source / "third_party",
      source / "src",
      source / "src/gpu/intel/jit/config",
      source / "third_party/ngen",
      source / "src/gpu/intel/gemm/jit/include",
  ]
  return [
      str(CXX), "-std=c++17", "-O3", "-DNDEBUG", "-fopenmp",
      "-fno-operator-names", "-DCL_TARGET_OPENCL_VERSION=120",
      "-DDNNL_X64=1", "-DGEMMSTONE_BUILD_12HP",
      "-DGEMMSTONE_BUILD_12LP", "-DGEMMSTONE_BUILD_12P7",
      "-DGEMMSTONE_BUILD_12P8", "-DGEMMSTONE_BUILD_XE2",
      "-DGEMMSTONE_BUILD_XE3", "-DGEMMSTONE_BUILD_XE3P",
      "-DGEMMSTONE_CONFIG", "-DNGEN_CONFIG",
      *[f"-I{path}" for path in includes],
      str(CODEGEN), str(build / "src/libdnnl.a"),
      "-lOpenCL", "-ldl", "-lpthread", "-o", str(binary),
  ]


def parse_codegen_json(stdout: str) -> dict[str, Any] | None:
  for line in reversed(stdout.splitlines()):
    if not line.startswith("{") or not line.endswith("}"):
      continue
    try:
      value = json.loads(line)
    except json.JSONDecodeError:
      continue
    if isinstance(value, dict):
      return value
  return None


def provider_probe(label: str, source: Path, build: Path, commit: str,
                   raw: Path, context: int,
                   kq_unroll_m: int, timeout_s: int) -> dict[str, Any]:
  binary = raw / f"{label}-codegen"
  build_result = run(build_command(source, build, binary), timeout_s)
  (raw / f"{label}-build.json").write_text(
      json.dumps(build_result, indent=2) + "\n")
  if build_result["returncode"] != 0:
    return {
        "label": label,
        "source": str(source),
        "build": str(build),
        "commit": commit,
        "build_returncode": build_result["returncode"],
        "probe_returncode": None,
        "generated": False,
    }

  environment = os.environ.copy()
  if label == "pinned":
    environment["ONEDNN_VERBOSE"] = "debuginfo=4"
  else:
    environment.pop("ONEDNN_VERBOSE", None)
  probe_result = run([
      str(binary), "--attention-nhalf", "--context", str(context),
      "--attention-kq-unroll-m", str(kq_unroll_m),
      "--provider-commit", commit, "--dump-dir",
      str(raw / f"{label}-packages"),
  ], timeout_s, environment)
  (raw / f"{label}-probe.json").write_text(
      json.dumps(probe_result, indent=2) + "\n")
  parsed = parse_codegen_json(probe_result["stdout"])
  diagnostic_result = probe_result
  if label != "pinned" and probe_result["returncode"] != 0:
    debug_environment = environment.copy()
    debug_environment["ONEDNN_VERBOSE"] = "debuginfo=4"
    diagnostic_result = run([
        str(binary), "--attention-nhalf", "--context", str(context),
        "--attention-kq-unroll-m", str(kq_unroll_m),
        "--provider-commit", commit,
    ], timeout_s, debug_environment)
    (raw / f"{label}-probe-debug.json").write_text(
        json.dumps(diagnostic_result, indent=2) + "\n")
  package_directory = raw / f"{label}-packages"
  return {
      "label": label,
      "source": str(source),
      "build": str(build),
      "commit": commit,
      "source_head": git_text(source, "rev-parse", "HEAD"),
      "source_status": git_text(source, "status", "--short"),
      "provider_scope_status": git_text(
          source, "status", "--short", "--",
          "src/gpu/intel/gemm/jit", "src/gpu/intel/jit/config",
          "third_party/ngen"),
      "build_config": cmake_settings(build),
      "libdnnl_sha256": sha256(build / "src/libdnnl.a"),
      "codegen_sha256": sha256(binary),
      "build_returncode": build_result["returncode"],
      "probe_returncode": probe_result["returncode"],
      "generated": probe_result["returncode"] == 0 and parsed is not None,
      "result": parsed,
      "package_artifacts": file_inventory(package_directory),
      "diagnostic": (
          diagnostic_result["stdout"] + diagnostic_result["stderr"]).strip(),
  }


def requested_nhalf_packages(
    probe: dict[str, Any], kq_unroll_m: int) -> bool:
  result = probe.get("result")
  if not isinstance(result, dict):
    return False
  packages = result.get("packages")
  if not isinstance(packages, list) or len(packages) != 2:
    return False
  expected = {
      "attention_kq_nhalf": (kq_unroll_m, 8, 16, 1),
      "attention_vs_nhalf": (16, 8, 16, 1),
  }
  for package in packages:
    if not isinstance(package, dict) or package.get("kind") not in expected:
      return False
    settings = package.get("settings", {})
    observed = (
        settings.get("wg_tile_m", 0) // max(settings.get("sg_per_wg_m", 0), 1),
        settings.get("wg_tile_n", 0) // max(settings.get("sg_per_wg_n", 0), 1),
        settings.get("sg_per_wg_m"),
        settings.get("sg_per_wg_n"),
    )
    if observed != expected[package["kind"]]:
      return False
  return True


def main() -> int:
  args = parse_args()
  if args.context <= 0:
    raise SystemExit("--context must be positive")
  out = args.out_dir.resolve()
  raw = out / "raw"
  raw.mkdir(parents=True, exist_ok=False)
  required = [
      CODEGEN, CXX, OPENVINO_SOURCE,
      args.pinned_source, args.pinned_build / "src/libdnnl.a",
      args.current_source, args.current_build / "src/libdnnl.a",
  ]
  missing = [str(path) for path in required if not path.exists()]
  if missing:
    raise SystemExit("missing required paths: " + ", ".join(missing))

  start_memory = meminfo()
  pinned = provider_probe(
      "pinned", args.pinned_source.resolve(), args.pinned_build.resolve(),
      PINNED_COMMIT, raw, args.context, args.kq_unroll_m, args.timeout_s)
  current_commit = git_text(
      args.current_source.resolve(), "rev-parse", "HEAD")
  current = provider_probe(
      "current", args.current_source.resolve(), args.current_build.resolve(),
      current_commit, raw, args.context, args.kq_unroll_m, args.timeout_s)
  pinned_reproduces_closed_gate = (
      args.kq_unroll_m == 16 and not pinned["generated"]
      and "Functionality is unimplemented" in pinned["diagnostic"]
      and "gemm HHS" in pinned["diagnostic"]
      and "wg 1x16" in pinned["diagnostic"])
  provider_scopes_clean = (
      pinned.get("provider_scope_status") == ""
      and current.get("provider_scope_status") == "")
  current_probe_completed = (
      current.get("build_returncode") == 0
      and current.get("probe_returncode") is not None)
  current_requested_nhalf = (
      provider_scopes_clean
      and current_probe_completed
      and current["generated"]
      and requested_nhalf_packages(current, args.kq_unroll_m))
  if not current_probe_completed:
    verdict = "inconclusive_current_provider_probe_failed"
  elif args.kq_unroll_m == 16:
    verdict = (
        "reopen_for_isolated_resource_and_bit_exact_gate"
        if current_requested_nhalf
        else "remain_closed_no_new_official_native_capability")
  else:
    verdict = (
        "generated_official_m8_candidate_require_resource_and_bit_exact_gate"
        if current_requested_nhalf
        else "remain_closed_no_official_m8_generated_capability")
  route = (
      "openvino_attention_exact_gqa4_nhalf_generated_package"
      if args.kq_unroll_m == 16
      else "openvino_attention_official_hpg_thinq_m8_n8_generated_package")
  result = {
      "schema_version":
          "intel-qwen36-openvino-attention-nhalf-capability-gate-v1",
      "route": route,
      "context": args.context,
      "kq_unroll_m": args.kq_unroll_m,
      "key_block": args.kq_unroll_m * 16,
      "repository_head": git_text(ROOT, "rev-parse", "HEAD"),
      "repository_status": git_text(ROOT, "status", "--short"),
      "official_sources": {
          "openvino_origin_master": git_text(
              OPENVINO_SOURCE, "rev-parse", "origin/master"),
          "openvino_origin_master_onednn": git_text(
              OPENVINO_SOURCE, "rev-parse",
              "origin/master:src/plugins/intel_gpu/thirdparty/onednn_gpu"),
          "onednn_origin_main": git_text(
              args.current_source.resolve(), "rev-parse", "origin/main"),
      },
      "memory": {"start": start_memory, "end": meminfo()},
      "pinned": pinned,
      "current_official": current,
      "checks": {
          "pinned_reproduces_closed_gate": pinned_reproduces_closed_gate,
          "provider_scopes_clean": provider_scopes_clean,
          "current_probe_completed": current_probe_completed,
          "current_generates_requested_kq_and_vs_nhalf":
              current_requested_nhalf,
      },
      "verdict": verdict,
  }
  (out / "capability-gate-result.json").write_text(
      json.dumps(result, indent=2) + "\n")
  print(json.dumps({
      "verdict": result["verdict"],
      "pinned_reproduces_closed_gate": pinned_reproduces_closed_gate,
      "current_generates_requested_kq_and_vs_nhalf":
          current_requested_nhalf,
      "artifact": str(out.relative_to(ROOT)),
  }))
  required_control = (
      pinned_reproduces_closed_gate if args.kq_unroll_m == 16 else True)
  return 0 if required_control and current_probe_completed else 1


if __name__ == "__main__":
  sys.exit(main())
