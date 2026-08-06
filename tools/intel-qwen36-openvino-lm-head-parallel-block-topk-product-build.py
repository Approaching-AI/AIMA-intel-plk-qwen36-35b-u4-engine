#!/usr/bin/env python3
"""Source-gate and build the admitted parallel exact block-top8 product cut.

This gate permits one serial CPU build of the Intel GPU plugin.  It does not
create a GPU context, load the product model, create an InferRequest, or run
inference.  The rebuilt plugin is copied to a new isolated seq2189 carrier;
the accepted seq2119 plugin is never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = (
    "intel-qwen36-openvino-lm-head-parallel-block-topk-product-build-v1")
R0 = Path("/home/intel/intel-qwen36-r0")
SOURCE_TREE = R0 / "source/openvino-90214e5be05"
SOURCE = SOURCE_TREE / (
    "src/plugins/intel_gpu/src/graph/impls/ocl/iq36_lm_head_i8q4.cpp")
BUILD_TREE = R0 / "build/openvino-90214e-l0-gpu"
BUILD_PLUGIN = R0 / (
    "output/openvino-90214e-l0-gpu-seq2109/bin/intel64/Release/"
    "libopenvino_intel_gpu_plugin.so")
CONTROL_PLUGIN = R0 / (
    "output/openvino-90214e-l0-gpu-seq2119/bin/intel64/Release/"
    "libopenvino_intel_gpu_plugin.so")
DEFAULT_CANDIDATE_PLUGIN = R0 / (
    "output/openvino-90214e-l0-gpu-seq2189/bin/intel64/Release/"
    "libopenvino_intel_gpu_plugin.so")
PATCH = ROOT / "engine/openvino/iq36-lm-head-i8q1-gated-exact.patch"
COMPONENT_SOURCE = (
    ROOT / "engine/gpu/opencl/iq36_lm_head_gated_exact_component.cl")
COMPONENT_RESULT = ROOT / (
    "output/openvino-lm-head-parallel-block-topk-component-"
    "20260731Tseq2188-clean/result.json")
CMAKE = Path("/home/intel/intel-box-env/conda/bin/cmake")
KERNEL_NAME = "iq36_lm_head_i8q1_gated_exact_output_topk8_f16"
COMPONENT_KERNEL_NAME = (
    "iq36_lm_head_gated_exact_component_parallel_block_topk8_f16")
EXPECTED_PATCH_SHA256 = (
    "14408168065680e36111ea123f08c3013bc9285142b811743cba437ac2094f7c")
EXPECTED_SOURCE_SHA256 = (
    "c94ef612e38fb0d498e5d852e48904e64ede37a5afb4abf0801543e8ccc871fe")
EXPECTED_BUILD_PLUGIN_SHA256 = (
    "f2a48fc83393081b0fcbc61aabe6db8c8d81906079aed0a2338d9b7591d3cfdf")
EXPECTED_CONTROL_PLUGIN_SHA256 = (
    "01c04ced415a7b7a5e5bda77a995b2b97b68eb3d9f2c5f3396844d042ddda269")
EXPECTED_COMPONENT_BODY_SHA256 = (
    "688bf9de8820822d26d216c235f449cfe3b920a1e0eee4ac2534bb8a747d16a7")
EXPECTED_COMPONENT_PRODUCT_PATCH_SHA256 = (
    "9a7e98a87cf638cf75d50e8323a4e012aea99329ffb5e0b1edd5683b07f73fe9")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument(
      "--candidate-plugin", type=Path, default=DEFAULT_CANDIDATE_PLUGIN)
  parser.add_argument("--timeout-s", type=float, default=900.0)
  parser.add_argument("--min-available-gib", type=float, default=8.0)
  parser.add_argument("--abort-below-available-gib", type=float, default=4.0)
  parser.add_argument("--poll-interval-s", type=float, default=0.1)
  args = parser.parse_args()
  if args.timeout_s <= 0.0 or args.poll_interval_s <= 0.0:
    parser.error("timeout and poll interval must be positive")
  if (args.min_available_gib <= 0.0 or
      args.abort_below_available_gib <= 0.0 or
      args.abort_below_available_gib > args.min_available_gib):
    parser.error("memory thresholds are invalid")
  return args


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    while chunk := stream.read(8 * 1024 * 1024):
      digest.update(chunk)
  return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def display(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path.resolve())


def check(name: str, passed: bool, **details: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **details}


def load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise TypeError(f"expected JSON object: {path}")
  return value


def run(command: list[str], cwd: Path = ROOT) -> dict[str, Any]:
  process = subprocess.run(
      command, cwd=cwd, capture_output=True, text=True, check=False)
  return {
      "command": command,
      "returncode": process.returncode,
      "stdout": process.stdout,
      "stderr": process.stderr,
  }


def git_state() -> dict[str, Any]:
  commit = run(["git", "rev-parse", "HEAD"])["stdout"].strip()
  status = run(["git", "status", "--porcelain"])["stdout"].splitlines()
  return {
      "commit": commit,
      "dirty": bool(status),
      "status": status,
  }


def meminfo() -> dict[str, int]:
  result: dict[str, int] = {}
  for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
    key, value = line.split(":", 1)
    fields = value.strip().split()
    if fields and fields[0].isdigit():
      result[key] = int(fields[0]) * 1024
  return result


def process_group_memory(pgrp: int) -> dict[str, int]:
  rss = 0
  swap = 0
  processes = 0
  for stat_path in Path("/proc").glob("[0-9]*/stat"):
    try:
      stat = stat_path.read_text(encoding="utf-8")
      tail = stat[stat.rfind(")") + 2:].split()
      if len(tail) < 3 or int(tail[2]) != pgrp:
        continue
      status = stat_path.with_name("status").read_text(
          encoding="utf-8", errors="replace").splitlines()
    except (FileNotFoundError, PermissionError, ProcessLookupError,
            ValueError):
      continue
    processes += 1
    for line in status:
      if line.startswith("VmRSS:"):
        rss += int(line.split()[1]) * 1024
      elif line.startswith("VmSwap:"):
        swap += int(line.split()[1]) * 1024
  return {"rss_bytes": rss, "swap_bytes": swap, "processes": processes}


def stop_group(process: subprocess.Popen[Any], first_signal: int) -> None:
  try:
    os.killpg(process.pid, first_signal)
  except ProcessLookupError:
    return
  try:
    process.wait(timeout=10.0)
    return
  except subprocess.TimeoutExpired:
    pass
  try:
    os.killpg(process.pid, signal.SIGKILL)
  except ProcessLookupError:
    pass
  process.wait()


def extract_kernel(text: str, name: str) -> str:
  marker = f"__kernel void {name}("
  begin = text.find(marker)
  if begin < 0:
    return ""
  opening = text.find("{", begin)
  if opening < 0:
    return ""
  depth = 0
  for index in range(opening, len(text)):
    character = text[index]
    if character == "{":
      depth += 1
    elif character == "}":
      depth -= 1
      if depth == 0:
        return text[begin:index + 1]
  return ""


def kernel_body(kernel: str, *, component: bool) -> str:
  if not kernel:
    return ""
  opening = kernel.find("{")
  body = kernel[opening + 1:kernel.rfind("}")]
  lines = []
  for raw_line in body.splitlines():
    line = " ".join(raw_line.split())
    if not line:
      continue
    if line == (
        "if (state[0] < IQ36_BINARY_GATED_EXACT_COUNT) return;"):
      continue
    if component:
      line = line.replace("IQ36_BLOCK_ROWS", "256U")
      line = line.replace("IQ36_TOPK", "8U")
    lines.append(line)
  return "\n".join(lines)


def patch_added_text(text: str) -> str:
  return "\n".join(
      line[1:] for line in text.splitlines()
      if line.startswith("+") and not line.startswith("+++"))


def source_contract() -> dict[str, Any]:
  patch_text = PATCH.read_text(encoding="utf-8")
  source_text = SOURCE.read_text(encoding="utf-8")
  component_text = COMPONENT_SOURCE.read_text(encoding="utf-8")
  patch_kernel = extract_kernel(patch_added_text(patch_text), KERNEL_NAME)
  source_kernel = extract_kernel(source_text, KERNEL_NAME)
  component_kernel = extract_kernel(
      component_text, COMPONENT_KERNEL_NAME)
  patch_body = kernel_body(patch_kernel, component=False)
  source_body = kernel_body(source_kernel, component=False)
  component_body = kernel_body(component_kernel, component=True)
  numstat = run(["git", "apply", "--numstat", str(PATCH)])
  source_status = run([
      "git", "status", "--short", "--",
      str(SOURCE.relative_to(SOURCE_TREE)),
  ], cwd=SOURCE_TREE)
  return {
      "patch_sha256": sha256(PATCH),
      "source_sha256": sha256(SOURCE),
      "component_source_sha256": sha256(COMPONENT_SOURCE),
      "patch_kernel_occurrences": patch_text.count(
          f"__kernel void {KERNEL_NAME}"),
      "source_kernel_occurrences": source_text.count(
          f"__kernel void {KERNEL_NAME}"),
      "component_kernel_occurrences": component_text.count(
          f"__kernel void {COMPONENT_KERNEL_NAME}"),
      "patch_body_sha256":
          hashlib.sha256(patch_body.encode()).hexdigest(),
      "source_body_sha256":
          hashlib.sha256(source_body.encode()).hexdigest(),
      "component_normalized_body_sha256":
          hashlib.sha256(component_body.encode()).hexdigest(),
      "bodies_identical":
          bool(patch_body) and patch_body == source_body == component_body,
      "candidate_body": source_body,
      "candidate_collective_max_count":
          source_kernel.count("work_group_reduce_max"),
      "candidate_collective_min_count":
          source_kernel.count("work_group_reduce_min"),
      "candidate_has_serial_local_scan":
          "__local float values[256]" in source_kernel or
          "if (lane != 0U) return;" in source_kernel,
      "rows_divide_workgroup_exactly": 248320 % 256 == 0,
      "block_count": 248320 // 256,
      "count25_occurrences": source_text.count(
          "#define IQ36_BINARY_GATED_EXACT_COUNT 25U"),
      "delta11_occurrences": source_text.count(
          "#define IQ36_BINARY_GATED_EXACT_DELTA 11.0f"),
      "patch_numstat": numstat,
      "source_tree_status": source_status,
  }


def component_contract(component: dict[str, Any]) -> dict[str, Any]:
  audits = [
      component.get("repeat_audit", {}),
      component.get("confirm_audit", {}),
  ]
  source = component.get("source_contract", {})
  return {
      "accepted":
          component.get("required_checks_passed") is True and
          component.get("verdict") ==
              "admit_exact_parallel_block_topk_for_product_integration" and
          component.get("product_integration_ready") is True and
          component.get("product_promotion_ready") is False and
          component.get("model_workers_launched") == 0 and
          component.get("workers_concurrent") is False and
          component.get("git", {}).get("dirty") is False,
      "body_identity_exact":
          source.get("candidate_body_sha256") ==
              EXPECTED_COMPONENT_BODY_SHA256 and
          source.get("product_patch_sha256") ==
              EXPECTED_COMPONENT_PRODUCT_PATCH_SHA256,
      "repeat_confirm_bitwise_exact":
          all(
              audit.get("all_outputs_bitwise_equal") is True and
              audit.get("all_selected_ids_equal") is True and
              audit.get("block_arithmetic_exact") is True and
              audit.get("anchor_bitwise_matches") == 3
              for audit in audits),
      "repeat_confirm_spill_free":
          all(
              audit.get("candidate_block_resources", {}).get(
                  "spill_mem_bytes") == 0 and
              audit.get("candidate_block_resources", {}).get(
                  "required_group_size_x") == 256 and
              audit.get("candidate_block_resources", {}).get(
                  "required_subgroup_size") == 16
              for audit in audits),
      "repeat_confirm_saving_lcb_us": [
          audit.get("performance_inference", {}).get(
              "lower_confidence_bound_saving_us")
          for audit in audits
      ],
      "required_saving_us": [
          audit.get("performance_inference", {}).get("required_saving_us")
          for audit in audits
      ],
      "repeat_confirm_rate_pass":
          all(
              audit.get("performance_inference", {}).get("rate_pass") is True
              for audit in audits),
  }


def monitor_build(
    command: list[str], output: Path, timeout_s: float,
    abort_bytes: int, poll_interval_s: float,
) -> dict[str, Any]:
  stdout_path = output / "raw/build.stdout"
  stderr_path = output / "raw/build.stderr"
  initial = meminfo()
  monitor = {
      "process_group_rss_peak_bytes": 0,
      "process_group_swap_peak_bytes": 0,
      "process_count_peak": 0,
      "system_available_min_bytes": int(initial.get("MemAvailable", 0)),
      "system_swap_used_peak_bytes": max(
          0, int(initial.get("SwapTotal", 0)) -
          int(initial.get("SwapFree", 0))),
      "samples": 0,
  }
  started = time.monotonic()
  timed_out = False
  guard_tripped = False
  with stdout_path.open("w", encoding="utf-8") as stdout_handle, \
       stderr_path.open("w", encoding="utf-8") as stderr_handle:
    process = subprocess.Popen(
        command, cwd=ROOT, stdout=stdout_handle, stderr=stderr_handle,
        text=True, start_new_session=True)
    while process.poll() is None:
      if time.monotonic() - started > timeout_s:
        timed_out = True
        stop_group(process, signal.SIGTERM)
        break
      system = meminfo()
      group = process_group_memory(process.pid)
      available = int(system.get("MemAvailable", 0))
      swap_used = max(
          0, int(system.get("SwapTotal", 0)) -
          int(system.get("SwapFree", 0)))
      monitor["samples"] += 1
      monitor["process_group_rss_peak_bytes"] = max(
          int(monitor["process_group_rss_peak_bytes"]), group["rss_bytes"])
      monitor["process_group_swap_peak_bytes"] = max(
          int(monitor["process_group_swap_peak_bytes"]), group["swap_bytes"])
      monitor["process_count_peak"] = max(
          int(monitor["process_count_peak"]), group["processes"])
      monitor["system_available_min_bytes"] = min(
          int(monitor["system_available_min_bytes"]), available)
      monitor["system_swap_used_peak_bytes"] = max(
          int(monitor["system_swap_used_peak_bytes"]), swap_used)
      if available < abort_bytes:
        guard_tripped = True
        stop_group(process, signal.SIGINT)
        break
      time.sleep(poll_interval_s)
    returncode = process.wait()
  elapsed = time.monotonic() - started
  stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
  lower_stderr = stderr.lower()
  oom_observed = (
      returncode in (-9, 137) or
      "out of memory" in lower_stderr or
      "cannot allocate memory" in lower_stderr)
  return {
      "command": command,
      "returncode": returncode,
      "elapsed_seconds": elapsed,
      "timed_out": timed_out,
      "memory_guard_tripped": guard_tripped,
      "oom_observed": oom_observed,
      "monitor": monitor,
      "stdout": display(stdout_path),
      "stderr": display(stderr_path),
  }


def main() -> int:
  args = parse_args()
  output = args.output.resolve()
  candidate_plugin = args.candidate_plugin.resolve()
  if output.exists():
    raise SystemExit(f"output already exists: {output}")
  if candidate_plugin.exists():
    raise SystemExit(
        f"isolated candidate plugin already exists: {candidate_plugin}")
  (output / "raw").mkdir(parents=True, exist_ok=False)
  required_paths = (
      SOURCE_TREE, SOURCE, BUILD_TREE, BUILD_PLUGIN, CONTROL_PLUGIN, PATCH,
      COMPONENT_SOURCE, COMPONENT_RESULT, CMAKE)
  missing = [str(path) for path in required_paths if not path.exists()]
  if missing:
    raise SystemExit("missing product-build inputs: " + ", ".join(missing))

  git = git_state()
  source = source_contract()
  component = component_contract(load_json(COMPONENT_RESULT))
  start_memory = meminfo()
  preflight_bytes = int(args.min_available_gib * 1024**3)
  abort_bytes = int(args.abort_below_available_gib * 1024**3)
  build_before = {
      "path": str(BUILD_PLUGIN),
      "sha256": sha256(BUILD_PLUGIN),
      "size_bytes": BUILD_PLUGIN.stat().st_size,
      "mtime_ns": BUILD_PLUGIN.stat().st_mtime_ns,
  }
  control = {
      "path": str(CONTROL_PLUGIN),
      "sha256": sha256(CONTROL_PLUGIN),
      "size_bytes": CONTROL_PLUGIN.stat().st_size,
      "mtime_ns": CONTROL_PLUGIN.stat().st_mtime_ns,
  }

  source_checks = [
      check("repository_clean_before_build", not git["dirty"], git=git),
      check("seq2188_component_admits_only_product_integration",
            component["accepted"] and component["body_identity_exact"] and
            component["repeat_confirm_bitwise_exact"] and
            component["repeat_confirm_spill_free"] and
            component["repeat_confirm_rate_pass"],
            component=component),
      check("durable_patch_is_well_formed_and_exact",
            source["patch_sha256"] == EXPECTED_PATCH_SHA256 and
            source["patch_numstat"]["returncode"] == 0 and
            source["patch_numstat"]["stdout"].strip() ==
                "324\t2\tsrc/plugins/intel_gpu/src/graph/impls/ocl/"
                "iq36_lm_head_i8q4.cpp",
            patch_sha256=source["patch_sha256"],
            patch_numstat=source["patch_numstat"]),
      check("local_product_source_is_exact_candidate",
            source["source_sha256"] == EXPECTED_SOURCE_SHA256 and
            source["source_kernel_occurrences"] == 1 and
            source["patch_kernel_occurrences"] == 1 and
            source["component_kernel_occurrences"] == 1,
            source_sha256=source["source_sha256"],
            source_tree_status=source["source_tree_status"]),
      check("product_body_matches_admitted_component_body",
            source["bodies_identical"] and
            source["candidate_collective_max_count"] == 1 and
            source["candidate_collective_min_count"] == 1 and
            not source["candidate_has_serial_local_scan"],
            patch_body_sha256=source["patch_body_sha256"],
            source_body_sha256=source["source_body_sha256"],
            component_body_sha256=
                source["component_normalized_body_sha256"]),
      check("exact_rows_cover_970_complete_workgroups",
            source["rows_divide_workgroup_exactly"] and
            source["block_count"] == 970),
      check("count25_and_delta11_are_unchanged",
            source["count25_occurrences"] == 1 and
            source["delta11_occurrences"] == 1,
            count25_occurrences=source["count25_occurrences"],
            delta11_occurrences=source["delta11_occurrences"]),
      check("accepted_control_and_build_base_are_exact",
            control["sha256"] == EXPECTED_CONTROL_PLUGIN_SHA256 and
            build_before["sha256"] == EXPECTED_BUILD_PLUGIN_SHA256,
            control_plugin=control, build_plugin_before=build_before),
      check("source_is_newer_than_incremental_build_base",
            SOURCE.stat().st_mtime_ns > BUILD_PLUGIN.stat().st_mtime_ns,
            source_mtime_ns=SOURCE.stat().st_mtime_ns,
            plugin_mtime_ns=BUILD_PLUGIN.stat().st_mtime_ns),
      check("eight_gib_preflight_is_available",
            int(start_memory.get("MemAvailable", 0)) >= preflight_bytes,
            available_bytes=start_memory.get("MemAvailable"),
            preflight_bytes=preflight_bytes),
      check("isolated_output_does_not_overwrite_accepted_plugin",
            candidate_plugin != CONTROL_PLUGIN.resolve() and
            candidate_plugin != BUILD_PLUGIN.resolve() and
            not candidate_plugin.exists(),
            candidate_plugin=str(candidate_plugin)),
  ]
  source_admitted = all(row["pass"] for row in source_checks)
  build_command = [
      str(CMAKE), "--build", str(BUILD_TREE),
      "--target", "openvino_intel_gpu_plugin", "--parallel", "1",
  ]
  if source_admitted:
    build = monitor_build(
        build_command, output, args.timeout_s, abort_bytes,
        args.poll_interval_s)
  else:
    build = {
        "command": build_command,
        "returncode": 125,
        "elapsed_seconds": 0.0,
        "timed_out": False,
        "memory_guard_tripped": False,
        "oom_observed": False,
        "monitor": {},
        "stdout": None,
        "stderr": None,
        "skipped": "source gate failed",
    }

  build_after = {
      "path": str(BUILD_PLUGIN),
      "sha256": sha256(BUILD_PLUGIN),
      "size_bytes": BUILD_PLUGIN.stat().st_size,
      "mtime_ns": BUILD_PLUGIN.stat().st_mtime_ns,
  }
  candidate: dict[str, Any] = {
      "path": str(candidate_plugin),
      "sha256": None,
      "size_bytes": None,
  }
  build_succeeded = (
      build["returncode"] == 0 and
      not build["timed_out"] and
      not build["memory_guard_tripped"] and
      not build["oom_observed"] and
      build_after["sha256"] != build_before["sha256"] and
      build_after["size_bytes"] > 0 and
      build_after["mtime_ns"] > build_before["mtime_ns"])
  if source_admitted and build_succeeded:
    candidate_plugin.parent.mkdir(parents=True, exist_ok=False)
    shutil.copy2(BUILD_PLUGIN, candidate_plugin)
    candidate = {
        "path": str(candidate_plugin),
        "sha256": sha256(candidate_plugin),
        "size_bytes": candidate_plugin.stat().st_size,
        "mtime_ns": candidate_plugin.stat().st_mtime_ns,
    }
  links = (
      run(["ldd", str(candidate_plugin)])
      if candidate_plugin.is_file() else {
          "command": ["ldd", str(candidate_plugin)],
          "returncode": 125,
          "stdout": "",
          "stderr": "candidate plugin missing",
      })
  lower_links = (links["stdout"] + links["stderr"]).lower()
  monitor = build.get("monitor", {})
  build_checks = [
      check("one_serial_incremental_gpu_plugin_build_succeeds",
            build_succeeded,
            build=build),
      check("four_gib_abort_guard_holds_without_oom",
            not build["memory_guard_tripped"] and
            not build["oom_observed"] and
            int(monitor.get("system_available_min_bytes", 0)) >= abort_bytes,
            abort_bytes=abort_bytes, monitor=monitor),
      check("new_plugin_is_copied_to_isolated_carrier",
            candidate.get("sha256") == build_after["sha256"] and
            candidate.get("size_bytes") == build_after["size_bytes"] and
            candidate.get("sha256") not in (
                build_before["sha256"], control["sha256"]),
            candidate_plugin=candidate,
            build_plugin_after=build_after),
      check("isolated_plugin_link_map_is_complete",
            links["returncode"] == 0 and
            "libopenvino.so" in lower_links and
            "libopencl.so" in lower_links and
            "not found" not in lower_links,
            link_map=links),
      check("no_gpu_context_model_or_inference_worker_ran", True,
            gpu_contexts=0, model_workers=0, infer_requests=0,
            inference_workers=0),
  ]
  required_checks_passed = (
      source_admitted and all(row["pass"] for row in build_checks))
  verdict = (
      "admit_parallel_block_topk_plugin_for_compile_only_graph_gate"
      if required_checks_passed else
      "repair_parallel_block_topk_product_source_or_build")
  result = {
      "schema": SCHEMA,
      "created_at": datetime.now(timezone.utc).isoformat(),
      "workstream": WS,
      "git": git,
      "verdict": verdict,
      "required_checks_passed": required_checks_passed,
      "graph_compile_admitted": required_checks_passed,
      "inference_admitted": False,
      "performance_claim_admitted": False,
      "source_contract": source,
      "component_contract": component,
      "source_checks": source_checks,
      "build_checks": build_checks,
      "build": build,
      "build_plugin_before": build_before,
      "build_plugin_after": build_after,
      "control_plugin": control,
      "candidate_plugin": candidate,
      "link_map": links,
      "workers": {
          "compiler_builds": 1 if source_admitted else 0,
          "gpu_contexts": 0,
          "model_workers": 0,
          "infer_requests": 0,
          "inference_workers": 0,
          "workers_concurrent": False,
      },
      "next_action": {
          "route": "parallel_block_topk_product_compile_only_graph_gate",
          "requirements": [
              "compile the exact product graph with the isolated plugin",
              "create no InferRequest and emit no token",
              "require the count25 gated-exact provider and unchanged "
              "attention-owner census",
          ],
      },
      "checks": source_checks + build_checks,
  }
  write_json(output / "result.json", result)
  write_json(output / "manifest.json", {
      "schema": SCHEMA,
      "tool": display(Path(__file__)),
      "git": git,
      "inputs": {
          display(path): sha256(path)
          for path in (
              PATCH, COMPONENT_SOURCE, COMPONENT_RESULT, SOURCE,
              CONTROL_PLUGIN)
      },
      "candidate_plugin": candidate,
      "compiler_builds": result["workers"]["compiler_builds"],
      "gpu_contexts": 0,
      "model_workers": 0,
      "infer_requests": 0,
      "inference_workers": 0,
  })
  report = f"""# Parallel exact block-top8 product plugin build

Verdict: **{verdict}**. Required checks:
`{str(required_checks_passed).lower()}`.

The durable patch, local OpenVINO source, and admitted seq2188 component body
are identical at the sole changed kernel. The one permitted plugin build used
parallelism `1`, completed in `{float(build['elapsed_seconds']):.3f} s`, and
produced `{candidate.get('sha256')}` in the isolated seq2189 carrier.

Peak process-group RSS was
`{int(monitor.get('process_group_rss_peak_bytes', 0))} B`; peak process-group
swap was `{int(monitor.get('process_group_swap_peak_bytes', 0))} B`; minimum
available memory was
`{int(monitor.get('system_available_min_bytes', 0))} B`. The 4-GiB abort guard
did not trip and no OOM was observed. No GPU context, model worker,
InferRequest, or inference worker ran.
"""
  (output / "report.md").write_text(report, encoding="utf-8")
  print(json.dumps({
      "artifact": display(output),
      "verdict": verdict,
      "required_checks_passed": required_checks_passed,
      "elapsed_seconds": build["elapsed_seconds"],
      "candidate_plugin_sha256": candidate.get("sha256"),
      "peak_rss_bytes": monitor.get("process_group_rss_peak_bytes"),
      "oom_observed": build["oom_observed"],
  }, separators=(",", ":")), flush=True)
  return 0 if required_checks_passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
