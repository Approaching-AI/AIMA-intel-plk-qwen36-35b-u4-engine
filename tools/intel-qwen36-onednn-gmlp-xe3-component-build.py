#!/usr/bin/env python3
"""Build only the seq2226-admitted Xe3 PR5059 GMLP test target."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-onednn-gmlp-xe3-component-build-v0"
BASE_TOOL = ROOT / "tools/intel-qwen36-onednn-gmlp-exact-component-build.py"
ADMISSION = ROOT / (
    "output/onednn-gmlp-pr5681-strategy-bound-"
    "20260731Tseq2226-clean/plan.json")
EXPECTED_ADMISSION_SHA256 = (
    "ab8a81f342fbdf510e362a8d2b815f6aa67bfe0c326b554e34b6b5f3ccabbe45")
ONEDNN_HEAD = "8621740ea5e600468c76a11a3c0c1616977f978d"


def load_base() -> Any:
  spec = importlib.util.spec_from_file_location(
      "iq36_onednn_gmlp_build_base", BASE_TOOL)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import build helper: {BASE_TOOL}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


BASE = load_base()


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", required=True, type=Path)
  parser.add_argument("--configure-timeout-s", default=300.0, type=float)
  parser.add_argument("--build-timeout-s", default=1800.0, type=float)
  parser.add_argument("--poll-interval-s", default=0.1, type=float)
  args = parser.parse_args()
  if (args.configure_timeout_s <= 0 or args.build_timeout_s <= 0
      or args.poll_interval_s <= 0):
    parser.error("timeouts and poll interval must be positive")
  return args


def load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise TypeError(f"expected JSON object: {path}")
  return value


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True) + "\n",
      encoding="utf-8")


def main() -> int:
  args = parse_args()
  output = args.output.resolve()
  raw = output / "raw"
  raw.mkdir(parents=True, exist_ok=False)
  if not ADMISSION.is_file():
    raise SystemExit(f"missing seq2226 admission: {ADMISSION}")

  admission = load_json(ADMISSION)
  plan = admission["plan"]
  source = Path(plan["source"]["worktree"])
  build_dir = Path(plan["build"]["build_dir"])
  test_binary = Path(plan["build"]["test_binary"])
  configure_command = [
      str(value) for value in plan["build"]["configure_command"]]
  build_command = [
      str(value) for value in plan["build"]["build_command"]]
  missing = [
      str(path) for path in (
          BASE_TOOL, source, BASE.SYSTEMD_RUN, BASE.SYSTEMCTL, BASE.TIME)
      if not path.exists()]
  if missing:
    raise SystemExit("missing Xe3 build inputs: " + ", ".join(missing))

  repo = BASE.repository_state(output)
  initial_memory = BASE.proc_meminfo()
  source_head_before = BASE.git(source, "rev-parse", "HEAD")
  source_status_before = BASE.git(
      source, "status", "--short", "--untracked-files=all")
  build_dir_fresh = not build_dir.exists()
  environment = os.environ.copy()
  conda_bin = "/home/intel/intel-box-env/conda/bin"
  conda_lib = "/home/intel/intel-box-env/conda/lib"
  environment["PATH"] = conda_bin + ":" + environment.get("PATH", "")
  environment["LD_LIBRARY_PATH"] = (
      conda_lib + ":" + environment.get("LD_LIBRARY_PATH", ""))

  configure = BASE.run_scoped(
      output, raw, "configure-xe3", configure_command,
      args.configure_timeout_s, args.poll_interval_s, environment)
  build = {
      "stage": "build-xe3",
      "returncode": None,
      "skipped": True,
  }
  if configure["returncode"] == 0:
    build = BASE.run_scoped(
        output, raw, "build-xe3", build_command,
        args.build_timeout_s, args.poll_interval_s, environment)
    build["skipped"] = False

  source_head_after = BASE.git(source, "rev-parse", "HEAD")
  source_status_after = BASE.git(
      source, "status", "--short", "--untracked-files=all")
  cache_path = build_dir / "CMakeCache.txt"
  config_path = build_dir / "include/oneapi/dnnl/dnnl_config.h"
  ninja_path = build_dir / "build.ninja"
  cache = (
      cache_path.read_text(encoding="utf-8", errors="replace")
      if cache_path.is_file() else "")
  config = (
      config_path.read_text(encoding="utf-8", errors="replace")
      if config_path.is_file() else "")
  ninja = (
      ninja_path.read_text(encoding="utf-8", errors="replace")
      if ninja_path.is_file() else "")
  target_link_edge = next((
      line for line in ninja.splitlines()
      if line.startswith(
          "build tests/gtests/internals/test_internals_gmlp: ")),
      "")
  ocl_header = source / "src/xpu/ocl/utils.hpp"
  xpu_source = source / "src/xpu/utils.cpp"
  ocl_text = (
      ocl_header.read_text(encoding="utf-8", errors="replace")
      if ocl_header.is_file() else "")
  xpu_text = (
      xpu_source.read_text(encoding="utf-8", errors="replace")
      if xpu_source.is_file() else "")
  binary = {
      "path": str(test_binary),
      "exists": test_binary.is_file(),
      "bytes": test_binary.stat().st_size if test_binary.is_file() else 0,
      "sha256": BASE.sha256(test_binary) if test_binary.is_file() else None,
  }
  ldd = (
      BASE.run(["/usr/bin/ldd", str(test_binary)])
      if test_binary.is_file() else "")
  build_stdout_path = raw / "build-xe3.stdout"
  build_stdout = (
      build_stdout_path.read_text(encoding="utf-8", errors="replace")
      if build_stdout_path.is_file() else "")
  compile_step_count = len(re.findall(
      r"\bBuilding (?:C|CXX) object\b", build_stdout))
  configure_ok = (
      configure["returncode"] == 0
      and not configure["timed_out"]
      and not configure["memory_guard_tripped"]
      and not configure["oom_observed"])
  build_ok = (
      build.get("returncode") == 0
      and not build.get("timed_out", False)
      and not build.get("memory_guard_tripped", False)
      and not build.get("oom_observed", False))
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
      ("DNNL_ENABLE_PRIMITIVE_GPU_ISA", "XE3"),
      ("ONEDNN_BUILD_GRAPH", "OFF"),
  )
  configure_min = int(
      configure["monitor"]["system_available_min_bytes"])
  build_min = int(
      build.get("monitor", {}).get("system_available_min_bytes", 0))
  configure_events = configure["monitor"]["memory_events_max"]
  build_events = build.get("monitor", {}).get("memory_events_max", {})
  checks = [
      BASE.check(
          "repository_clean_and_pushed_at_gate",
          repo["branch"] == "main" and repo["pushed"] and not repo["dirty"],
          **repo),
      BASE.check(
          "seq2226_admission_identity_exact",
          BASE.sha256(ADMISSION) == EXPECTED_ADMISSION_SHA256
          and admission["source_verdict"]["required_checks_passed"] is True
          and admission["source_verdict"][
              "xe3_component_rebuild_admitted"] is True
          and admission["source_verdict"]["pr5681_exact_fix_admitted"] is False
          and admission["source_verdict"]["product_build_admitted"] is False,
          admission_sha256=BASE.sha256(ADMISSION)),
      BASE.check(
          "clean_exact_source_reused_without_mutation",
          source_head_before == ONEDNN_HEAD
          and source_head_after == ONEDNN_HEAD
          and source_status_before == "" and source_status_after == "",
          source_head_before=source_head_before,
          source_head_after=source_head_after,
          source_status_before=source_status_before,
          source_status_after=source_status_after),
      BASE.check(
          "fresh_isolated_xe3_build_path",
          build_dir_fresh
          and str(build_dir).endswith("onednn-862174-gmlp-xe3-exact")
          and configure_command[4] == str(build_dir),
          build_dir=str(build_dir),
          build_dir_was_fresh=build_dir_fresh),
      BASE.check(
          "configure_succeeded_in_unique_scope",
          configure_ok, configure=configure),
      BASE.check(
          "configure_cache_and_registration_are_xe3_only",
          all(BASE.cache_has(cache, name, value)
              for name, value in required_cache_values)
          and "#define BUILD_PRIMITIVE_GPU_ISA_ALL 0" in config
          and "#define BUILD_XE2 0" in config
          and "#define BUILD_XE3 1" in config
          and "#define BUILD_XE3P 0" in config
          and "-DGEMMSTONE_BUILD_XE3" in ninja
          and "-DGEMMSTONE_BUILD_XE2" not in ninja,
          cache_path=str(cache_path),
          config_path=str(config_path),
          required_values=dict(required_cache_values),
          gemmstone_xe3_definitions=ninja.count("-DGEMMSTONE_BUILD_XE3"),
          gemmstone_xe2_definitions=ninja.count("-DGEMMSTONE_BUILD_XE2")),
      BASE.check(
          "sole_j1_test_target_build_succeeded",
          build_ok
          and build_command[-4:] == [
              "--target", "test_internals_gmlp", "-j", "1"],
          build=build),
      BASE.check(
          "test_binary_created_and_opencl_runtime_wired",
          binary["exists"] and binary["bytes"] > 0
          and "libdnnl.so" in ldd and "not found" not in ldd
          and "/home/intel/intel-box-env/conda/lib/libOpenCL.so"
          in target_link_edge
          and '#define OCL_LIB_NAME "libOpenCL.so.1"' in ocl_text
          and "xpu::find_symbol(OCL_LIB_NAME, symbol)" in ocl_text
          and "dlopen(library_name, RTLD_NOW | RTLD_LOCAL)" in xpu_text,
          binary=binary, ldd=ldd, target_link_edge=target_link_edge),
      BASE.check(
          "memory_policy_held_without_oom",
          int(initial_memory.get("MemAvailable", 0))
          >= int(plan["build"]["preflight_bytes"])
          and configure_ok and build_ok
          and configure_min >= int(plan["build"]["abort_below_bytes"])
          and build_min >= int(plan["build"]["abort_below_bytes"])
          and int(configure_events.get("oom_kill", 0)) == 0
          and int(build_events.get("oom_kill", 0)) == 0,
          preflight_bytes=plan["build"]["preflight_bytes"],
          abort_below_bytes=plan["build"]["abort_below_bytes"],
          configure_minimum_available_bytes=configure_min,
          build_minimum_available_bytes=build_min,
          configure_memory_events=configure_events,
          build_memory_events=build_events),
      BASE.check(
          "no_gpu_context_model_or_infer_request_ran",
          True, gpu_contexts_created=0, gpu_kernels_executed=0,
          model_workers_started=0, infer_requests_created=0,
          openvino_product_builds=0),
  ]
  required_checks_passed = all(row["pass"] for row in checks)
  verdict = {
      "required_checks_passed": required_checks_passed,
      "xe3_component_provider_runs_admitted": required_checks_passed,
      "product_build_admitted": False,
      "verdict": (
          "retain_xe3_exact_gmlp_binary_for_provider_gate"
          if required_checks_passed else
          "reject_xe3_component_run_build_or_identity_failure"),
      "next_if_pass": (
          "run one exact decode process, then one exact prefill process only "
          "if decode successfully creates and executes ocl:micro_horz:any; "
          "paired timing remains blocked until both providers pass"),
  }
  metrics = {
      "schema": SCHEMA,
      "workstream": WS,
      "created_at": datetime.now(timezone.utc).isoformat(),
      "git": repo,
      "admission": {
          "path": BASE.relative(ADMISSION),
          "sha256": BASE.sha256(ADMISSION),
      },
      "source": {
          "worktree": str(source),
          "commit": source_head_after,
          "status_before": source_status_before,
          "status_after": source_status_after,
      },
      "configure": configure,
      "build": {
          **build,
          "compile_step_count": compile_step_count,
          "parallel_jobs": 1,
          "target": "test_internals_gmlp",
      },
      "binary": binary,
      "ldd": ldd,
      "memory": {
          "initial": initial_memory,
          "preflight_bytes": plan["build"]["preflight_bytes"],
          "abort_below_bytes": plan["build"]["abort_below_bytes"],
      },
      "workers": {
          "maximum_concurrent_workers": 1,
          "configure_invocations": 1,
          "target_builds": int(not build.get("skipped", True)),
          "gpu_contexts_created": 0,
          "gpu_kernels_executed": 0,
          "model_workers_started": 0,
          "infer_requests_created": 0,
      },
      "checks": checks,
      "verdict": verdict,
  }
  write_json(output / "metrics.json", metrics)
  (output / "report.md").write_text(
      "# oneDNN PR5059 exact Xe3 GMLP component build\n\n"
      f"- Required checks: `{required_checks_passed}`\n"
      f"- Verdict: `{verdict['verdict']}`\n"
      f"- Configure/build elapsed: "
      f"`{configure.get('elapsed_seconds', 0):.3f}/"
      f"{build.get('elapsed_seconds', 0):.3f} s`\n"
      f"- Build target/jobs/ISA: `test_internals_gmlp / 1 / XE3`\n"
      f"- Binary SHA256: `{binary['sha256']}`\n"
      f"- Compile/GPU/model/InferRequest: "
      f"`{compile_step_count}/0/0/0`\n",
      encoding="utf-8")
  print(json.dumps({
      "output": BASE.relative(output),
      "required_checks_passed": required_checks_passed,
      "verdict": verdict["verdict"],
      "configure_returncode": configure.get("returncode"),
      "build_returncode": build.get("returncode"),
      "build_elapsed_seconds": build.get("elapsed_seconds"),
      "binary_sha256": binary["sha256"],
      "xe3_component_provider_runs_admitted": (
          verdict["xe3_component_provider_runs_admitted"]),
  }, sort_keys=True), flush=True)
  return 0 if required_checks_passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
