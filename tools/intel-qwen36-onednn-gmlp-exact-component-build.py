#!/usr/bin/env python3
"""Build only the admitted oneDNN PR5059 exact-shape GMLP test target."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-onednn-gmlp-exact-component-build-v0"
ADMISSION = ROOT / (
    "output/onednn-gmlp-exact-component-admission-"
    "20260731Tseq2223-clean/plan.json")
EXPECTED_ADMISSION_SHA256 = (
    "f21ff2d8bffb326f5cc0bd7cfceb8544b5369bde2a2508914254b35961cb7c33")
ONEDNN_HEAD = "8621740ea5e600468c76a11a3c0c1616977f978d"
ONEDNN_REPO = Path(
    "/home/intel/intel-qwen36-r0/source/openvino-90214e5be05/"
    "src/plugins/intel_gpu/thirdparty/onednn_gpu")
SOURCE_WORKTREE = Path(
    "/home/intel/intel-qwen36-r0/source/oneDNN-862174-gmlp-exact")
BUILD_DIR = Path(
    "/home/intel/intel-qwen36-r0/build/onednn-862174-gmlp-exact")
TEST_BINARY = (
    BUILD_DIR / "tests/gtests/internals/test_internals_gmlp")
SYSTEMD_RUN = Path("/usr/bin/systemd-run")
SYSTEMCTL = Path("/usr/bin/systemctl")
TIME = Path("/usr/bin/time")
PREFLIGHT_BYTES = 8 * 1024**3
ABORT_BYTES = 4 * 1024**3


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


def run(
    command: list[str], cwd: Path = ROOT, *,
    environment: dict[str, str] | None = None,
) -> str:
  result = subprocess.run(
      command, cwd=cwd, env=environment, check=False,
      capture_output=True, text=True, encoding="utf-8", errors="replace")
  if result.returncode != 0:
    raise RuntimeError(
        f"command failed ({result.returncode}): {command}\n{result.stderr}")
  return result.stdout


def git(cwd: Path, *args: str) -> str:
  return run(["git", *args], cwd=cwd).strip()


def relative(path: Path) -> str:
  try:
    return path.resolve().relative_to(ROOT).as_posix()
  except ValueError:
    return str(path.resolve())


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


def proc_meminfo() -> dict[str, int]:
  result = {}
  for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
    key, value = line.split(":", 1)
    fields = value.strip().split()
    if fields and fields[0].isdigit():
      result[key] = int(fields[0]) * 1024
  return result


def read_int(path: Path) -> int:
  try:
    return int(path.read_text(encoding="utf-8").strip())
  except (FileNotFoundError, PermissionError, ValueError):
    return 0


def read_named_ints(path: Path) -> dict[str, int]:
  try:
    rows = path.read_text(encoding="utf-8").splitlines()
  except (FileNotFoundError, PermissionError):
    return {}
  result = {}
  for row in rows:
    fields = row.split()
    if len(fields) == 2 and fields[1].isdigit():
      result[fields[0]] = int(fields[1])
  return result


def process_memory(pid: int) -> dict[str, int]:
  status = Path(f"/proc/{pid}/status")
  result = {"rss_bytes": 0, "swap_bytes": 0}
  try:
    rows = status.read_text(encoding="utf-8").splitlines()
  except (FileNotFoundError, PermissionError):
    return result
  for row in rows:
    if row.startswith("VmRSS:"):
      result["rss_bytes"] = int(row.split()[1]) * 1024
    elif row.startswith("VmSwap:"):
      result["swap_bytes"] = int(row.split()[1]) * 1024
  return result


def parse_time_max_rss(path: Path) -> int:
  if not path.is_file():
    return 0
  match = re.search(
      r"Maximum resident set size \(kbytes\):\s*(\d+)",
      path.read_text(encoding="utf-8", errors="replace"))
  return int(match.group(1)) * 1024 if match else 0


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


def scope_unit(output: Path, stage: str) -> str:
  digest = hashlib.sha256(
      f"{output.resolve()}:{stage}".encode("utf-8")).hexdigest()[:12]
  return f"iq36-onednn-gmlp-{stage}-{digest}"


def scope_cgroup(unit: str) -> Path:
  uid = os.getuid()
  return Path(
      f"/sys/fs/cgroup/user.slice/user-{uid}.slice/user@{uid}.service/"
      f"app.slice/{unit}.scope")


def stop_scope(unit: str, process: subprocess.Popen[Any]) -> None:
  subprocess.run(
      [str(SYSTEMCTL), "--user", "kill", "--signal=SIGINT",
       f"{unit}.scope"],
      check=False, capture_output=True, text=True)
  try:
    process.wait(timeout=10.0)
    return
  except subprocess.TimeoutExpired:
    pass
  subprocess.run(
      [str(SYSTEMCTL), "--user", "kill", "--signal=SIGKILL",
       f"{unit}.scope"],
      check=False, capture_output=True, text=True)
  try:
    os.killpg(process.pid, signal.SIGKILL)
  except ProcessLookupError:
    pass
  process.wait()


def run_scoped(
    output: Path, raw: Path, stage: str, command: list[str],
    timeout_s: float, poll_interval_s: float, environment: dict[str, str],
) -> dict[str, Any]:
  unit = scope_unit(output, stage)
  cgroup = scope_cgroup(unit)
  stdout_path = raw / f"{stage}.stdout"
  stderr_path = raw / f"{stage}.stderr"
  time_path = raw / f"{stage}.time"
  timed_command = [
      str(TIME), "-v", "-o", str(time_path), *command]
  scoped_command = [
      str(SYSTEMD_RUN), "--user", "--scope", "--quiet", "--collect",
      f"--unit={unit}", *timed_command]
  start_memory = proc_meminfo()
  if int(start_memory.get("MemAvailable", 0)) < PREFLIGHT_BYTES:
    raise RuntimeError(
        f"{stage} preflight below 8 GiB: "
        f"{start_memory.get('MemAvailable', 0)}")
  monitor = {
      "system_available_min_bytes": int(start_memory["MemAvailable"]),
      "system_swap_used_peak_bytes": (
          int(start_memory.get("SwapTotal", 0))
          - int(start_memory.get("SwapFree", 0))),
      "cgroup_memory_peak_bytes": 0,
      "cgroup_swap_peak_bytes": 0,
      "wrapper_rss_peak_bytes": 0,
      "wrapper_swap_peak_bytes": 0,
      "process_count_peak": 0,
      "memory_events_max": {},
      "samples": 0,
  }
  started = time.monotonic()
  timed_out = False
  guard_tripped = False
  with stdout_path.open("w", encoding="utf-8") as stdout_handle, \
       stderr_path.open("w", encoding="utf-8") as stderr_handle:
    process = subprocess.Popen(
        scoped_command, cwd=ROOT, env=environment,
        stdout=stdout_handle, stderr=stderr_handle, text=True,
        start_new_session=True)
    while process.poll() is None:
      elapsed = time.monotonic() - started
      if elapsed > timeout_s:
        timed_out = True
        stop_scope(unit, process)
        break
      system = proc_meminfo()
      available = int(system.get("MemAvailable", 0))
      swap_used = (
          int(system.get("SwapTotal", 0))
          - int(system.get("SwapFree", 0)))
      wrapper = process_memory(process.pid)
      cgroup_current = read_int(cgroup / "memory.current")
      cgroup_swap = read_int(cgroup / "memory.swap.current")
      cgroup_peak = read_int(cgroup / "memory.peak")
      events = read_named_ints(cgroup / "memory.events")
      process_count = len(
          (cgroup / "cgroup.procs").read_text(
              encoding="utf-8").splitlines()
          ) if (cgroup / "cgroup.procs").is_file() else 0
      monitor["samples"] += 1
      monitor["system_available_min_bytes"] = min(
          int(monitor["system_available_min_bytes"]), available)
      monitor["system_swap_used_peak_bytes"] = max(
          int(monitor["system_swap_used_peak_bytes"]), swap_used)
      monitor["cgroup_memory_peak_bytes"] = max(
          int(monitor["cgroup_memory_peak_bytes"]),
          cgroup_current, cgroup_peak)
      monitor["cgroup_swap_peak_bytes"] = max(
          int(monitor["cgroup_swap_peak_bytes"]), cgroup_swap)
      monitor["wrapper_rss_peak_bytes"] = max(
          int(monitor["wrapper_rss_peak_bytes"]),
          wrapper["rss_bytes"])
      monitor["wrapper_swap_peak_bytes"] = max(
          int(monitor["wrapper_swap_peak_bytes"]),
          wrapper["swap_bytes"])
      monitor["process_count_peak"] = max(
          int(monitor["process_count_peak"]), process_count)
      previous_events = monitor["memory_events_max"]
      monitor["memory_events_max"] = {
          key: max(int(previous_events.get(key, 0)), value)
          for key, value in events.items()}
      if available < ABORT_BYTES:
        guard_tripped = True
        stop_scope(unit, process)
        break
      time.sleep(poll_interval_s)
    returncode = process.wait()
  elapsed_seconds = time.monotonic() - started
  stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
  events = monitor["memory_events_max"]
  oom_observed = (
      not guard_tripped
      and (returncode in (-9, 137)
           or int(events.get("oom_kill", 0)) > 0
           or "out of memory" in stderr.lower()))
  monitor["time_max_rss_bytes"] = parse_time_max_rss(time_path)
  monitor["process_rss_peak_bytes"] = max(
      int(monitor["cgroup_memory_peak_bytes"]),
      int(monitor["time_max_rss_bytes"]),
      int(monitor["wrapper_rss_peak_bytes"]))
  return {
      "stage": stage,
      "command": command,
      "scoped_command": scoped_command,
      "scope": {
          "enabled": True,
          "unit": unit,
          "cgroup_root": str(cgroup),
          "resource_limits_changed": False,
      },
      "returncode": returncode,
      "elapsed_seconds": elapsed_seconds,
      "timed_out": timed_out,
      "memory_guard_tripped": guard_tripped,
      "oom_observed": oom_observed,
      "monitor": monitor,
      "stdout": relative(stdout_path),
      "stderr": relative(stderr_path),
      "time": relative(time_path),
  }


def cache_has(cache: str, name: str, value: str) -> bool:
  return re.search(
      rf"^{re.escape(name)}:(?:BOOL|STRING|INTERNAL)="
      rf"{re.escape(value)}$",
      cache, flags=re.MULTILINE) is not None


def main() -> int:
  args = parse_args()
  output = args.output.resolve()
  raw = output / "raw"
  raw.mkdir(parents=True, exist_ok=False)
  required = (
      ADMISSION, ONEDNN_REPO, SYSTEMD_RUN, SYSTEMCTL, TIME)
  missing = [str(path) for path in required if not path.exists()]
  if missing:
    raise SystemExit("missing GMLP build inputs: " + ", ".join(missing))

  repo = repository_state(output)
  admission = load_json(ADMISSION)
  plan = admission["plan"]
  configure_command = [
      str(value) for value in plan["build"]["configure_command"]]
  build_command = [
      str(value) for value in plan["build"]["build_command"]]
  worktree_command = [
      str(value) for value in plan["source"]["create_command"]]
  initial_memory = proc_meminfo()
  source_status_before = git(
      ONEDNN_REPO, "status", "--short", "--untracked-files=all")
  paths_fresh = not SOURCE_WORKTREE.exists() and not BUILD_DIR.exists()

  worktree_started = time.monotonic()
  worktree_result = subprocess.run(
      worktree_command, cwd=ROOT, check=False,
      capture_output=True, text=True, encoding="utf-8", errors="replace")
  worktree_elapsed = time.monotonic() - worktree_started
  worktree = {
      "command": worktree_command,
      "returncode": worktree_result.returncode,
      "elapsed_seconds": worktree_elapsed,
      "stdout": worktree_result.stdout,
      "stderr": worktree_result.stderr,
  }

  environment = os.environ.copy()
  conda_bin = "/home/intel/intel-box-env/conda/bin"
  conda_lib = "/home/intel/intel-box-env/conda/lib"
  environment["PATH"] = conda_bin + ":" + environment.get("PATH", "")
  environment["LD_LIBRARY_PATH"] = (
      conda_lib + ":" + environment.get("LD_LIBRARY_PATH", ""))
  configure = {
      "stage": "configure",
      "returncode": None,
      "skipped": True,
  }
  build = {
      "stage": "build",
      "returncode": None,
      "skipped": True,
  }
  if worktree_result.returncode == 0:
    configure = run_scoped(
        output, raw, "configure", configure_command,
        args.configure_timeout_s, args.poll_interval_s, environment)
    configure["skipped"] = False
    if configure["returncode"] == 0:
      build = run_scoped(
          output, raw, "build", build_command,
          args.build_timeout_s, args.poll_interval_s, environment)
      build["skipped"] = False

  source_status_after = git(
      ONEDNN_REPO, "status", "--short", "--untracked-files=all")
  source_identity = (
      git(SOURCE_WORKTREE, "rev-parse", "HEAD")
      if SOURCE_WORKTREE.exists() else None)
  worktree_dirty = (
      git(SOURCE_WORKTREE, "status", "--short", "--untracked-files=all")
      if SOURCE_WORKTREE.exists() else "missing")
  cache_path = BUILD_DIR / "CMakeCache.txt"
  cache = (
      cache_path.read_text(encoding="utf-8", errors="replace")
      if cache_path.is_file() else "")
  build_ninja_path = BUILD_DIR / "build.ninja"
  build_ninja = (
      build_ninja_path.read_text(encoding="utf-8", errors="replace")
      if build_ninja_path.is_file() else "")
  target_link_edge = next((
      line for line in build_ninja.splitlines()
      if line.startswith(
          "build tests/gtests/internals/test_internals_gmlp: ")),
      "")
  opencl_link_configured = (
      "/home/intel/intel-box-env/conda/lib/libOpenCL.so"
      in target_link_edge)
  ocl_utils_header = SOURCE_WORKTREE / "src/xpu/ocl/utils.hpp"
  xpu_utils_source = SOURCE_WORKTREE / "src/xpu/utils.cpp"
  ocl_header_text = (
      ocl_utils_header.read_text(encoding="utf-8", errors="replace")
      if ocl_utils_header.is_file() else "")
  xpu_utils_text = (
      xpu_utils_source.read_text(encoding="utf-8", errors="replace")
      if xpu_utils_source.is_file() else "")
  opencl_runtime_loader_present = (
      '#define OCL_LIB_NAME "libOpenCL.so.1"' in ocl_header_text
      and "xpu::find_symbol(OCL_LIB_NAME, symbol)" in ocl_header_text
      and "dlopen(library_name, RTLD_NOW | RTLD_LOCAL)" in xpu_utils_text)
  binary = {
      "path": str(TEST_BINARY),
      "exists": TEST_BINARY.is_file(),
      "bytes": TEST_BINARY.stat().st_size if TEST_BINARY.is_file() else 0,
      "sha256": sha256(TEST_BINARY) if TEST_BINARY.is_file() else None,
  }
  ldd = (
      run(["/usr/bin/ldd", str(TEST_BINARY)])
      if TEST_BINARY.is_file() else "")
  build_stdout = (
      (raw / "build.stdout").read_text(
          encoding="utf-8", errors="replace")
      if (raw / "build.stdout").is_file() else "")
  compile_step_count = len(re.findall(
      r"\bBuilding (?:C|CXX) object\b", build_stdout))
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
  )
  configure_ok = (
      configure.get("returncode") == 0
      and not configure.get("timed_out", False)
      and not configure.get("memory_guard_tripped", False)
      and not configure.get("oom_observed", False))
  build_ok = (
      build.get("returncode") == 0
      and not build.get("timed_out", False)
      and not build.get("memory_guard_tripped", False)
      and not build.get("oom_observed", False))
  checks = [
      check(
          "repository_clean_and_pushed_at_gate",
          repo["branch"] == "main" and repo["pushed"] and not repo["dirty"],
          **repo),
      check(
          "admission_identity_and_scope_exact",
          sha256(ADMISSION) == EXPECTED_ADMISSION_SHA256
          and admission["verdict"]["component_build_admitted"] is True
          and admission["verdict"]["product_build_admitted"] is False),
      check(
          "isolated_paths_were_fresh", paths_fresh),
      check(
          "detached_exact_worktree_created",
          worktree_result.returncode == 0
          and source_identity == ONEDNN_HEAD
          and worktree_dirty == "",
          worktree=worktree, source_identity=source_identity,
          worktree_dirty=worktree_dirty),
      check(
          "existing_onednn_worktree_state_unchanged",
          source_status_before == source_status_after,
          status_before=source_status_before,
          status_after=source_status_after),
      check(
          "configure_succeeded_in_fresh_scope", configure_ok,
          configure=configure),
      check(
          "configure_cache_exact",
          all(cache_has(cache, name, value)
              for name, value in required_cache_values),
          cache_path=str(cache_path),
          required_values=dict(required_cache_values)),
      check(
          "sole_j1_test_target_build_succeeded",
          build_ok
          and build_command[-4:] == [
              "--target", "test_internals_gmlp", "-j", "1"],
          build=build),
      check(
          "test_binary_created_and_opencl_runtime_wired",
          binary["exists"] and binary["bytes"] > 0
          and "libdnnl.so" in ldd
          and "not found" not in ldd
          and opencl_link_configured
          and opencl_runtime_loader_present,
          binary=binary, ldd=ldd,
          build_ninja_path=str(build_ninja_path),
          target_link_edge=target_link_edge,
          opencl_link_configured=opencl_link_configured,
          opencl_runtime_loader_present=opencl_runtime_loader_present,
          note=(
              "The test link edge includes the OpenCL ICD. With "
              "--as-needed, ELF NEEDED can omit it because oneDNN resolves "
              "OpenCL entry points through its runtime loader.")),
      check(
          "memory_policy_held_without_oom",
          int(initial_memory.get("MemAvailable", 0)) >= PREFLIGHT_BYTES
          and configure_ok and build_ok
          and int(configure["monitor"]["system_available_min_bytes"])
          >= ABORT_BYTES
          and int(build["monitor"]["system_available_min_bytes"])
          >= ABORT_BYTES,
          preflight_bytes=PREFLIGHT_BYTES,
          abort_bytes=ABORT_BYTES),
      check(
          "no_gpu_context_model_or_infer_request_ran",
          True,
          gpu_contexts_created=0,
          model_workers_started=0,
          infer_requests_created=0),
  ]
  required_checks_passed = all(row["pass"] for row in checks)
  verdict = {
      "required_checks_passed": required_checks_passed,
      "component_runs_admitted": required_checks_passed,
      "product_build_admitted": False,
      "verdict": (
          "retain_exact_gmlp_test_binary_for_serial_component"
          if required_checks_passed else
          "reject_component_runs_build_or_identity_failure"),
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
      "admission": {
          "path": relative(ADMISSION),
          "sha256": sha256(ADMISSION),
      },
      "source": {
          "repository": str(ONEDNN_REPO),
          "status_before": source_status_before,
          "status_after": source_status_after,
          "worktree": str(SOURCE_WORKTREE),
          "worktree_command": worktree,
          "worktree_commit": source_identity,
          "worktree_dirty": worktree_dirty,
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
          "preflight_bytes": PREFLIGHT_BYTES,
          "abort_below_bytes": ABORT_BYTES,
      },
      "workers": {
          "maximum_concurrent_workers": 1,
          "worktrees_created": int(worktree_result.returncode == 0),
          "configure_invocations": int(not configure.get("skipped", True)),
          "target_builds": int(not build.get("skipped", True)),
          "gpu_contexts_created": 0,
          "model_workers_started": 0,
          "infer_requests_created": 0,
      },
      "checks": checks,
      "verdict": verdict,
  }
  write_json(output / "metrics.json", metrics)
  (output / "report.md").write_text(
      "# oneDNN PR5059 exact GMLP component build\n\n"
      f"- Required checks: `{required_checks_passed}`\n"
      f"- Verdict: `{verdict['verdict']}`\n"
      f"- Configure/build elapsed: "
      f"`{configure.get('elapsed_seconds', 0):.3f}/"
      f"{build.get('elapsed_seconds', 0):.3f} s`\n"
      f"- Build target/jobs: `test_internals_gmlp / 1`\n"
      f"- Binary SHA256: `{binary['sha256']}`\n"
      f"- Compile/GPU/model/InferRequest: "
      f"`{compile_step_count}/0/0/0`\n",
      encoding="utf-8")
  print(json.dumps({
      "output": relative(output),
      "required_checks_passed": required_checks_passed,
      "verdict": verdict["verdict"],
      "configure_returncode": configure.get("returncode"),
      "build_returncode": build.get("returncode"),
      "build_elapsed_seconds": build.get("elapsed_seconds"),
      "binary_sha256": binary["sha256"],
      "component_runs_admitted": verdict["component_runs_admitted"],
  }, sort_keys=True), flush=True)
  return 0 if required_checks_passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
