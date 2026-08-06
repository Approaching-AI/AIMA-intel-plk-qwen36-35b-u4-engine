#!/usr/bin/env python3
"""Re-audit the existing exact-shape GMLP build without rebuilding it.

The original build gate required libOpenCL.so to remain in the executable's
ELF NEEDED list.  The link uses --as-needed while oneDNN resolves OpenCL entry
points at runtime, so that condition rejected an otherwise valid build.  This
audit verifies the original evidence, the configured OpenCL link edge, and the
runtime loader.  It starts no compiler, GPU context, model, or InferRequest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-onednn-gmlp-exact-component-build-audit-v0"
ORIGINAL = ROOT / (
    "output/onednn-gmlp-exact-component-build-"
    "20260731Tseq2224-clean/metrics.json")
EXPECTED_ORIGINAL_SHA256 = (
    "1c53f77c36996da6c339cff888c048d8e37e511da979b99e49e0c6f2ca7d18d2")
EXPECTED_BUILD_COMMIT = "c9f0b33681076cb86c0ab1291503910b16973ff9"
ONEDNN_HEAD = "8621740ea5e600468c76a11a3c0c1616977f978d"
SOURCE_WORKTREE = Path(
    "/home/intel/intel-qwen36-r0/source/oneDNN-862174-gmlp-exact")
BUILD_DIR = Path(
    "/home/intel/intel-qwen36-r0/build/onednn-862174-gmlp-exact")
TEST_BINARY = BUILD_DIR / "tests/gtests/internals/test_internals_gmlp"
BUILD_NINJA = BUILD_DIR / "build.ninja"
CMAKE_CACHE = BUILD_DIR / "CMakeCache.txt"
OCL_UTILS_HEADER = SOURCE_WORKTREE / "src/xpu/ocl/utils.hpp"
XPU_UTILS_SOURCE = SOURCE_WORKTREE / "src/xpu/utils.cpp"
OPENCL_ICD = Path("/home/intel/intel-box-env/conda/lib/libOpenCL.so")
EXPECTED_BINARY_SHA256 = (
    "61a749e95f6ead521ce17e13d50ea9f377e12a5121036ec5764bc3d8b24c8747")
PREFLIGHT_BYTES = 8 * 1024**3
ABORT_BYTES = 4 * 1024**3


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", required=True, type=Path)
  return parser.parse_args()


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    while chunk := stream.read(1024 * 1024):
      digest.update(chunk)
  return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise TypeError(f"expected JSON object: {path}")
  return value


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True) + "\n",
      encoding="utf-8")


def run(command: list[str], cwd: Path = ROOT) -> str:
  result = subprocess.run(
      command, cwd=cwd, check=False, capture_output=True, text=True,
      encoding="utf-8", errors="replace")
  if result.returncode != 0:
    raise RuntimeError(
        f"command failed ({result.returncode}): {command}\n"
        f"{result.stderr}")
  return result.stdout


def git(cwd: Path, *args: str) -> str:
  return run(["git", *args], cwd=cwd).strip()


def relative(path: Path) -> str:
  try:
    return path.resolve().relative_to(ROOT).as_posix()
  except ValueError:
    return str(path.resolve())


def available_memory_bytes() -> int:
  for line in Path("/proc/meminfo").read_text(
      encoding="utf-8").splitlines():
    if line.startswith("MemAvailable:"):
      return int(line.split()[1]) * 1024
  raise RuntimeError("MemAvailable is missing")


def repository_state(output: Path) -> dict[str, Any]:
  head = git(ROOT, "rev-parse", "HEAD")
  upstream = git(ROOT, "rev-parse", "@{u}")
  output_rel = relative(output)
  dirty = []
  for row in git(
      ROOT, "status", "--porcelain", "--untracked-files=all").splitlines():
    path = row[3:]
    if path == output_rel or path.startswith(output_rel + "/"):
      continue
    dirty.append(row)
  return {
      "branch": git(ROOT, "branch", "--show-current"),
      "commit": head,
      "upstream_commit": upstream,
      "pushed": head == upstream,
      "dirty": bool(dirty),
      "dirty_paths": dirty,
  }


def check(name: str, passed: bool, **evidence: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **evidence}


def cache_has(cache: str, name: str, value: str) -> bool:
  return re.search(
      rf"^{re.escape(name)}:(?:BOOL|STRING|INTERNAL|FILEPATH|PATH)="
      rf"{re.escape(value)}$",
      cache, flags=re.MULTILINE) is not None


def stage_is_safe(stage: dict[str, Any]) -> bool:
  monitor = stage.get("monitor", {})
  events = monitor.get("memory_events_max", {})
  return bool(
      stage.get("returncode") == 0
      and stage.get("timed_out") is False
      and stage.get("memory_guard_tripped") is False
      and stage.get("oom_observed") is False
      and int(monitor.get("system_available_min_bytes", 0)) >= ABORT_BYTES
      and int(events.get("oom", 0)) == 0
      and int(events.get("oom_kill", 0)) == 0
      and int(events.get("oom_group_kill", 0)) == 0)


def main() -> int:
  args = parse_args()
  output = args.output.resolve()
  output.mkdir(parents=True, exist_ok=False)
  memory_start = available_memory_bytes()
  if memory_start < PREFLIGHT_BYTES:
    raise SystemExit(
        f"audit preflight below 8 GiB: {memory_start}")

  required = (
      ORIGINAL, SOURCE_WORKTREE, TEST_BINARY, BUILD_NINJA, CMAKE_CACHE,
      OCL_UTILS_HEADER, XPU_UTILS_SOURCE, OPENCL_ICD)
  missing = [str(path) for path in required if not path.exists()]
  if missing:
    raise SystemExit("missing GMLP build audit inputs: " + ", ".join(missing))

  repo = repository_state(output)
  original = load_json(ORIGINAL)
  original_checks = original["checks"]
  original_failed = [
      row["name"] for row in original_checks if row["pass"] is False]
  original_other_checks_pass = all(
      row["pass"] is True
      for row in original_checks
      if row["name"] != "test_binary_created_and_linked")
  binary = {
      "path": str(TEST_BINARY),
      "exists": TEST_BINARY.is_file(),
      "bytes": TEST_BINARY.stat().st_size,
      "sha256": sha256(TEST_BINARY),
  }
  ldd = run(["/usr/bin/ldd", str(TEST_BINARY)])
  build_ninja = BUILD_NINJA.read_text(
      encoding="utf-8", errors="replace")
  build_ninja_lines = build_ninja.splitlines()
  target_link_index = next((
      index for index, line in enumerate(build_ninja_lines)
      if line.startswith(
          "build tests/gtests/internals/test_internals_gmlp: ")),
      -1)
  target_link_block = (
      "\n".join(build_ninja_lines[
          target_link_index:target_link_index + 16])
      if target_link_index >= 0 else "")
  target_link_edge = (
      build_ninja_lines[target_link_index]
      if target_link_index >= 0 else "")
  target_link_libraries = next((
      line.strip() for line in target_link_block.splitlines()
      if line.strip().startswith("LINK_LIBRARIES = ")
      and str(OPENCL_ICD) in line),
      "")
  cache = CMAKE_CACHE.read_text(encoding="utf-8", errors="replace")
  required_cache_values = (
      ("CMAKE_BUILD_TYPE", "Release"),
      ("DNNL_BUILD_TESTS", "ON"),
      ("DNNL_BUILD_EXAMPLES", "OFF"),
      ("DNNL_CPU_RUNTIME", "NONE"),
      ("DNNL_GPU_RUNTIME", "OCL"),
      ("DNNL_GPU_VENDOR", "INTEL"),
      ("DNNL_LIBRARY_TYPE", "SHARED"),
      ("DNNL_ENABLE_WORKLOAD", "INFERENCE"),
      ("DNNL_ENABLE_PRIMITIVE", "GATED_MLP;MATMUL;REORDER;ELTWISE"),
      ("DNNL_ENABLE_PRIMITIVE_GPU_ISA", "XE2"),
      ("ONEDNN_BUILD_GRAPH", "OFF"),
      ("OpenCL_INCLUDE_DIR", "/home/intel/intel-box-env/conda/include"),
      ("OpenCL_LIBRARY", str(OPENCL_ICD)),
  )
  ocl_header = OCL_UTILS_HEADER.read_text(
      encoding="utf-8", errors="replace")
  xpu_utils = XPU_UTILS_SOURCE.read_text(
      encoding="utf-8", errors="replace")
  source_commit = git(SOURCE_WORKTREE, "rev-parse", "HEAD")
  source_dirty = git(
      SOURCE_WORKTREE, "status", "--short", "--untracked-files=all")
  original_git = original["git"]
  original_binary = original["binary"]
  original_build = original["build"]
  original_configure = original["configure"]
  original_workers = original["workers"]

  checks = [
      check(
          "repository_clean_and_pushed_at_audit",
          repo["branch"] == "main" and repo["pushed"] and not repo["dirty"],
          **repo),
      check(
          "original_artifact_identity_exact",
          sha256(ORIGINAL) == EXPECTED_ORIGINAL_SHA256
          and original_git["commit"] == EXPECTED_BUILD_COMMIT
          and original_git["upstream_commit"] == EXPECTED_BUILD_COMMIT
          and original_git["dirty"] is False,
          path=relative(ORIGINAL), sha256=sha256(ORIGINAL),
          expected_sha256=EXPECTED_ORIGINAL_SHA256),
      check(
          "original_only_failed_invalid_elf_needed_assumption",
          original_failed == ["test_binary_created_and_linked"]
          and original_other_checks_pass,
          original_failed_checks=original_failed),
      check(
          "exact_clean_onednn_source_retained",
          source_commit == ONEDNN_HEAD and source_dirty == "",
          source_commit=source_commit, source_dirty=source_dirty),
      check(
          "configure_and_j1_build_succeeded_without_oom",
          stage_is_safe(original_configure)
          and stage_is_safe(original_build)
          and original_build["parallel_jobs"] == 1
          and original_build["target"] == "test_internals_gmlp",
          configure=original_configure, build=original_build),
      check(
          "exact_build_cache_retained",
          all(cache_has(cache, name, value)
              for name, value in required_cache_values),
          cache_path=str(CMAKE_CACHE),
          required_values=dict(required_cache_values)),
      check(
          "exact_binary_identity_and_dnnl_link_retained",
          binary["sha256"] == EXPECTED_BINARY_SHA256
          and binary == original_binary
          and "libdnnl.so.3" in ldd
          and "not found" not in ldd,
          binary=binary, ldd=ldd),
      check(
          "opencl_icd_was_configured_on_target_link_edge",
          str(OPENCL_ICD) in target_link_edge
          and str(OPENCL_ICD) in target_link_libraries,
          build_ninja=str(BUILD_NINJA),
          target_link_edge=target_link_edge,
          target_link_libraries=target_link_libraries),
      check(
          "onednn_runtime_opencl_loader_retained",
          '#define OCL_LIB_NAME "libOpenCL.so.1"' in ocl_header
          and "xpu::find_symbol(OCL_LIB_NAME, symbol)" in ocl_header
          and "dlopen(library_name, RTLD_NOW | RTLD_LOCAL)" in xpu_utils,
          ocl_utils_header=str(OCL_UTILS_HEADER),
          xpu_utils_source=str(XPU_UTILS_SOURCE),
          explanation=(
              "The link uses --as-needed and oneDNN resolves OpenCL symbols "
              "at runtime, so libOpenCL need not remain in ELF NEEDED.")),
      check(
          "audit_started_no_compiler_gpu_model_or_infer_request",
          original_workers["maximum_concurrent_workers"] == 1
          and original_workers["gpu_contexts_created"] == 0
          and original_workers["model_workers_started"] == 0
          and original_workers["infer_requests_created"] == 0,
          audit_compiler_invocations=0, audit_gpu_contexts_created=0,
          audit_model_workers_started=0,
          audit_infer_requests_created=0),
  ]
  passed = all(row["pass"] for row in checks)
  verdict = {
      "required_checks_passed": passed,
      "component_runs_admitted": passed,
      "product_build_admitted": False,
      "verdict": (
          "admit_existing_exact_gmlp_binary_for_serial_component"
          if passed else
          "reject_existing_exact_gmlp_binary_audit_failure"),
      "next_if_pass": (
          "run exactly eight strictly serial paired test processes for each "
          "registered GMLP_TEST shape; require ocl:micro_horz:any, component "
          "correctness, prefill delta UCB <= -0.001209 ms, and decode delta "
          "UCB <= 0"),
  }
  metrics = {
      "schema": SCHEMA,
      "workstream": WS,
      "created_at": datetime.now(timezone.utc).isoformat(),
      "git": repo,
      "memory": {
          "available_at_start_bytes": memory_start,
          "preflight_bytes": PREFLIGHT_BYTES,
          "abort_below_bytes": ABORT_BYTES,
      },
      "original": {
          "path": relative(ORIGINAL),
          "sha256": sha256(ORIGINAL),
          "failed_checks": original_failed,
      },
      "source": {
          "worktree": str(SOURCE_WORKTREE),
          "commit": source_commit,
          "dirty": bool(source_dirty),
      },
      "binary": binary,
      "ldd": ldd,
      "opencl_wiring": {
          "icd": str(OPENCL_ICD),
          "build_ninja": str(BUILD_NINJA),
          "target_link_edge": target_link_edge,
          "target_link_libraries": target_link_libraries,
          "runtime_loader_header": str(OCL_UTILS_HEADER),
          "runtime_loader_source": str(XPU_UTILS_SOURCE),
      },
      "workers": {
          "maximum_concurrent_workers": 1,
          "compiler_invocations": 0,
          "gpu_contexts_created": 0,
          "model_workers_started": 0,
          "infer_requests_created": 0,
      },
      "checks": checks,
      "verdict": verdict,
  }
  write_json(output / "metrics.json", metrics)
  (output / "report.md").write_text(
      "# oneDNN PR5059 exact GMLP component build audit\n\n"
      f"- Required checks: `{passed}`\n"
      f"- Verdict: `{verdict['verdict']}`\n"
      f"- Original metrics SHA256: `{sha256(ORIGINAL)}`\n"
      f"- Binary SHA256: `{binary['sha256']}`\n"
      "- Original configure/build: successful, `-j1`, no OOM\n"
      "- Correction: the OpenCL ICD is present on the target link edge; "
      "`--as-needed` may omit it from ELF NEEDED because oneDNN uses its "
      "runtime OpenCL loader\n"
      "- Audit compiler/GPU/model/InferRequest: `0/0/0/0`\n",
      encoding="utf-8")
  print(json.dumps({
      "output": relative(output),
      "required_checks_passed": passed,
      "verdict": verdict["verdict"],
      "binary_sha256": binary["sha256"],
      "component_runs_admitted": verdict["component_runs_admitted"],
  }, sort_keys=True), flush=True)
  return 0 if passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
