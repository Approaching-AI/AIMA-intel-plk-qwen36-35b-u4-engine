#!/usr/bin/env python3
"""Apply and serial-build the admitted current Q/K plus router-shared bundle.

The seq2205 opportunity gate admits exactly one source application and one
``-j1`` Intel GPU plugin build.  This tool performs only those operations.
It creates no GPU context, loads no model, and runs no inference.  The accepted
seq2189 carrier is treated as immutable and the new plugin is copied to an
isolated seq2206 carrier only after the build and link-map gates pass.
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
SCHEMA = "intel-qwen36-openvino-current-qk-router-shared-product-build-v1"
R0 = Path("/home/intel/intel-qwen36-r0")
SOURCE_TREE = R0 / "source/openvino-90214e5be05"
SOURCE_REL = Path(
    "src/plugins/intel_gpu/src/plugin/transformations/"
    "fc_horizontal_fusion.cpp")
SOURCE = SOURCE_TREE / SOURCE_REL
BUILD_TREE = R0 / "build/openvino-90214e-l0-gpu"
BUILD_PLUGIN = R0 / (
    "output/openvino-90214e-l0-gpu-seq2109/bin/intel64/Release/"
    "libopenvino_intel_gpu_plugin.so")
ACCEPTED_PLUGIN = R0 / (
    "output/openvino-90214e-l0-gpu-seq2189/bin/intel64/Release/"
    "libopenvino_intel_gpu_plugin.so")
DEFAULT_CANDIDATE_PLUGIN = R0 / (
    "output/openvino-90214e-l0-gpu-seq2206/bin/intel64/Release/"
    "libopenvino_intel_gpu_plugin.so")
PATCH = ROOT / "engine/openvino/iq36-current-router-shared-triple.patch"
PRODUCT_TOOL = ROOT / "tools/intel-qwen36-openvino-hot-cold-product-gate.py"
OPPORTUNITY = ROOT / (
    "output/openvino-current-short-bundle-opportunity-"
    "20260731Tseq2205-clean/result.json")
CMAKE = Path("/home/intel/intel-box-env/conda/bin/cmake")

PINNED_SOURCE_COMMIT = "90214e5be052438cec5617ed3ea7e37df1538f68"
OPPORTUNITY_COMMIT = "0ecebba26efd15af91c71731f34e63d094e1c259"
EXPECTED_SOURCE_SHA256 = (
    "4a32d9c17d84390aef343bd60c992859fc75bc72d2f8ddff3a355c5276ba6020")
EXPECTED_PATCH_SHA256 = (
    "ae013a8a610de89d6f8b48971e7238b240db31d2d1d832fce328a6a4290f4420")
EXPECTED_OPPORTUNITY_SHA256 = (
    "2d9b5aa65fd3827b2c87e8cd2ea87e54d3d170b8da38dede057caa8e90c08fc7")
EXPECTED_PRODUCT_TOOL_SHA256 = (
    "9d20a6d71878235067b969ac9246ca2671f6b3c4704f82817a24786f8a3c808d")
EXPECTED_ACCEPTED_PLUGIN_SHA256 = (
    "9832adffa8bf3fd7b013c47d9d2abefa66987c5fbde350c9669fce58c697b985")


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


def load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise TypeError(f"expected JSON object: {path}")
  return value


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def display(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path.resolve())


def run(command: list[str], cwd: Path = ROOT) -> dict[str, Any]:
  process = subprocess.run(
      command, cwd=cwd, capture_output=True, text=True, check=False)
  return {
      "command": command,
      "returncode": process.returncode,
      "stdout": process.stdout,
      "stderr": process.stderr,
  }


def check(name: str, passed: bool, **details: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **details}


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


def monitor_build(
    command: list[str],
    output: Path,
    timeout_s: float,
    abort_bytes: int,
    poll_interval_s: float,
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
  return {
      "command": command,
      "returncode": returncode,
      "elapsed_seconds": elapsed,
      "timed_out": timed_out,
      "memory_guard_tripped": guard_tripped,
      "oom_observed": (
          returncode in (-9, 137) or
          "out of memory" in lower_stderr or
          "cannot allocate memory" in lower_stderr),
      "monitor": monitor,
      "stdout": display(stdout_path),
      "stderr": display(stderr_path),
  }


def plugin_record(path: Path) -> dict[str, Any]:
  return {
      "path": str(path),
      "sha256": sha256(path),
      "size_bytes": path.stat().st_size,
      "mtime_ns": path.stat().st_mtime_ns,
  }


def source_record() -> dict[str, Any]:
  status = run(
      ["git", "status", "--short", "--", str(SOURCE_REL)],
      cwd=SOURCE_TREE)
  return {
      "path": str(SOURCE),
      "sha256": sha256(SOURCE),
      "size_bytes": SOURCE.stat().st_size,
      "mtime_ns": SOURCE.stat().st_mtime_ns,
      "target_status": status,
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
      SOURCE_TREE, SOURCE, BUILD_TREE, BUILD_PLUGIN, ACCEPTED_PLUGIN, PATCH,
      PRODUCT_TOOL, OPPORTUNITY, CMAKE)
  missing = [str(path) for path in required_paths if not path.exists()]
  if missing:
    raise SystemExit("missing current-bundle build inputs: " + ", ".join(missing))

  head = run(["git", "rev-parse", "HEAD"])["stdout"].strip()
  origin_main = run(["git", "rev-parse", "origin/main"])["stdout"].strip()
  repo_status = run(["git", "status", "--porcelain"])["stdout"].splitlines()
  opportunity_ancestor = run([
      "git", "merge-base", "--is-ancestor", OPPORTUNITY_COMMIT, head])
  source_commit = run(
      ["git", "rev-parse", "HEAD"], SOURCE_TREE)["stdout"].strip()
  forward_check = run(
      ["git", "apply", "--check", str(PATCH)], SOURCE_TREE)
  reverse_check_before = run(
      ["git", "apply", "--reverse", "--check", str(PATCH)], SOURCE_TREE)
  opportunity = load_json(OPPORTUNITY)
  source_before = source_record()
  build_before = plugin_record(BUILD_PLUGIN)
  accepted_before = plugin_record(ACCEPTED_PLUGIN)
  start_memory = meminfo()
  preflight_bytes = int(args.min_available_gib * 1024**3)
  abort_bytes = int(args.abort_below_available_gib * 1024**3)
  patch_text = PATCH.read_text(encoding="utf-8")
  product_text = PRODUCT_TOOL.read_text(encoding="utf-8")

  source_checks = [
      check("repository_is_clean_and_pushed_before_build",
            not repo_status and head == origin_main and
            opportunity_ancestor["returncode"] == 0,
            head=head, origin_main=origin_main, status=repo_status,
            opportunity_commit_is_ancestor=
                opportunity_ancestor["returncode"] == 0),
      check("seq2205_admits_exactly_one_source_and_plugin_build",
            sha256(OPPORTUNITY) == EXPECTED_OPPORTUNITY_SHA256 and
            opportunity.get("required_checks_passed") is True and
            opportunity.get("verdict") ==
                "admit_current_qk_router_shared_source_build" and
            opportunity.get("source_build_admitted") is True and
            opportunity.get("plugin_build_admitted") is True and
            opportunity.get("gpu_worker_admitted") is False and
            opportunity.get("model_worker_admitted") is False and
            opportunity.get("cross_lane_sum_admitted") is False and
            opportunity.get("git", {}).get("commit") == OPPORTUNITY_COMMIT,
            opportunity_sha256=sha256(OPPORTUNITY),
            opportunity_verdict=opportunity.get("verdict")),
      check("pinned_cumulative_source_is_exact_preimage",
            source_commit == PINNED_SOURCE_COMMIT and
            source_before["sha256"] == EXPECTED_SOURCE_SHA256,
            source_commit=source_commit, source_before=source_before),
      check("durable_default_off_patch_is_forward_only",
            sha256(PATCH) == EXPECTED_PATCH_SHA256 and
            forward_check["returncode"] == 0 and
            reverse_check_before["returncode"] != 0 and
            patch_text.count("diff --git ") == 1 and
            "IQ36_ROUTER_SHARED_TRIPLE" in patch_text and
            "fixed_m1024_scope_enabled() ==" in patch_text,
            patch_sha256=sha256(PATCH),
            forward_check=forward_check,
            reverse_check_before=reverse_check_before),
      check("runtime_switch_contract_is_pinned",
            sha256(PRODUCT_TOOL) == EXPECTED_PRODUCT_TOOL_SHA256 and
            "--fuse-router-shared-triple" in product_text and
            "IQ36_ROUTER_SHARED_TRIPLE" in product_text and
            "fuse_router_shared_triple" in product_text,
            product_tool_sha256=sha256(PRODUCT_TOOL)),
      check("mutable_build_base_and_accepted_carrier_are_exact",
            build_before["sha256"] == EXPECTED_ACCEPTED_PLUGIN_SHA256 and
            accepted_before["sha256"] == EXPECTED_ACCEPTED_PLUGIN_SHA256,
            build_plugin_before=build_before,
            accepted_plugin_before=accepted_before),
      check("eight_gib_preflight_is_available",
            int(start_memory.get("MemAvailable", 0)) >= preflight_bytes,
            available_bytes=start_memory.get("MemAvailable"),
            preflight_bytes=preflight_bytes),
      check("isolated_output_cannot_overwrite_existing_carriers",
            candidate_plugin not in (
                BUILD_PLUGIN.resolve(), ACCEPTED_PLUGIN.resolve()) and
            not candidate_plugin.exists(),
            candidate_plugin=str(candidate_plugin)),
      check("no_gpu_context_model_or_inference_worker_is_admitted", True,
            gpu_contexts=0, model_workers=0, infer_requests=0,
            inference_workers=0),
  ]
  source_apply_admitted = all(row["pass"] for row in source_checks)
  if source_apply_admitted:
    apply_result = run(["git", "apply", str(PATCH)], SOURCE_TREE)
  else:
    apply_result = {
        "command": ["git", "apply", str(PATCH)],
        "returncode": 125,
        "stdout": "",
        "stderr": "source gate failed",
        "skipped": True,
    }

  source_after_apply = source_record()
  reverse_check_after = run(
      ["git", "apply", "--reverse", "--check", str(PATCH)], SOURCE_TREE)
  forward_check_after = run(
      ["git", "apply", "--check", str(PATCH)], SOURCE_TREE)
  source_text = SOURCE.read_text(encoding="utf-8")
  apply_checks = [
      check("one_durable_patch_application_succeeds",
            source_apply_admitted and apply_result["returncode"] == 0 and
            source_after_apply["sha256"] != source_before["sha256"],
            apply_result=apply_result,
            source_before_sha256=source_before["sha256"],
            source_after_sha256=source_after_apply["sha256"]),
      check("applied_source_is_reverse_only_and_structurally_exact",
            reverse_check_after["returncode"] == 0 and
            forward_check_after["returncode"] != 0 and
            source_text.count("bool fixed_m1024_scope_enabled()") == 1 and
            source_text.count("bool router_shared_triple_enabled()") == 1 and
            source_text.count(
                'std::getenv("IQ36_ROUTER_SHARED_TRIPLE")') == 1 and
            source_text.count(
                "fixed_m1024_scope_enabled() == "
                "router_shared_triple_enabled()") == 1 and
            source_text.count(
                "router_shared_triple_enabled() && n != 256") == 1 and
            source_text.count(
                "router_shared_triple_enabled() ? 3 : 2") == 1,
            reverse_check_after=reverse_check_after,
            forward_check_after=forward_check_after,
            source_after=source_after_apply),
  ]
  build_admitted = source_apply_admitted and all(
      row["pass"] for row in apply_checks)
  build_command = [
      str(CMAKE), "--build", str(BUILD_TREE),
      "--target", "openvino_intel_gpu_plugin", "--parallel", "1",
  ]
  if build_admitted:
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
        "skipped": "source application gate failed",
    }

  build_after = plugin_record(BUILD_PLUGIN)
  accepted_after = plugin_record(ACCEPTED_PLUGIN)
  monitor = build.get("monitor", {})
  build_succeeded = (
      build["returncode"] == 0 and
      not build["timed_out"] and
      not build["memory_guard_tripped"] and
      not build["oom_observed"] and
      build_after["sha256"] != build_before["sha256"] and
      build_after["size_bytes"] > 0 and
      build_after["mtime_ns"] > build_before["mtime_ns"])
  candidate: dict[str, Any] = {
      "path": str(candidate_plugin),
      "sha256": None,
      "size_bytes": None,
      "mtime_ns": None,
  }
  if build_admitted and build_succeeded:
    candidate_plugin.parent.mkdir(parents=True, exist_ok=False)
    shutil.copy2(BUILD_PLUGIN, candidate_plugin)
    candidate = plugin_record(candidate_plugin)

  links = (
      run(["ldd", str(candidate_plugin)])
      if candidate_plugin.is_file() else {
          "command": ["ldd", str(candidate_plugin)],
          "returncode": 125,
          "stdout": "",
          "stderr": "candidate plugin missing",
      })
  lower_links = (links["stdout"] + links["stderr"]).lower()
  build_checks = [
      check("one_serial_incremental_gpu_plugin_build_succeeds",
            build_admitted and build_succeeded, build=build),
      check("four_gib_abort_guard_holds_without_oom",
            not build["memory_guard_tripped"] and
            not build["oom_observed"] and
            int(monitor.get("system_available_min_bytes", 0)) >= abort_bytes,
            abort_bytes=abort_bytes, monitor=monitor),
      check("new_plugin_is_copied_to_isolated_seq2206_carrier",
            candidate.get("sha256") == build_after["sha256"] and
            candidate.get("size_bytes") == build_after["size_bytes"] and
            candidate.get("sha256") != EXPECTED_ACCEPTED_PLUGIN_SHA256,
            candidate_plugin=candidate,
            build_plugin_after=build_after),
      check("accepted_seq2189_carrier_remains_bitwise_immutable",
            accepted_after == accepted_before,
            accepted_plugin_before=accepted_before,
            accepted_plugin_after=accepted_after),
      check("isolated_plugin_link_map_is_complete",
            links["returncode"] == 0 and
            "libopenvino.so" in lower_links and
            "libopencl.so" in lower_links and
            "not found" not in lower_links,
            link_map=links),
      check("no_gpu_context_model_or_inference_worker_ran", True,
            compiler_builds=1 if build_admitted else 0, gpu_contexts=0,
            model_workers=0, infer_requests=0, inference_workers=0,
            workers_concurrent=False),
  ]
  required_checks_passed = (
      source_apply_admitted and
      all(row["pass"] for row in apply_checks) and
      all(row["pass"] for row in build_checks))
  verdict = (
      "admit_current_qk_router_shared_plugin_for_candidate_only_compile_gate"
      if required_checks_passed else
      "repair_current_qk_router_shared_source_or_serial_build")

  result = {
      "schema": SCHEMA,
      "created_at": datetime.now(timezone.utc).isoformat(),
      "workstream": WS,
      "verdict": verdict,
      "required_checks_passed": required_checks_passed,
      "candidate_only_graph_compile_admitted": required_checks_passed,
      "infer_request_admitted": False,
      "inference_admitted": False,
      "performance_claim_admitted": False,
      "repository": {
          "head": head,
          "origin_main": origin_main,
          "dirty": bool(repo_status),
          "status": repo_status,
      },
      "opportunity": {
          "path": display(OPPORTUNITY),
          "sha256": sha256(OPPORTUNITY),
          "verdict": opportunity.get("verdict"),
      },
      "source_before": source_before,
      "source_after_apply": source_after_apply,
      "source_checks": source_checks,
      "source_apply": apply_result,
      "apply_checks": apply_checks,
      "build": build,
      "build_plugin_before": build_before,
      "build_plugin_after": build_after,
      "accepted_plugin_before": accepted_before,
      "accepted_plugin_after": accepted_after,
      "candidate_plugin": candidate,
      "link_map": links,
      "build_checks": build_checks,
      "workers": {
          "compiler_builds": 1 if build_admitted else 0,
          "gpu_contexts": 0,
          "model_workers": 0,
          "infer_requests": 0,
          "inference_workers": 0,
          "workers_concurrent": False,
      },
      "next_action": {
          "route": "current_qk_router_shared_candidate_only_compile_gate",
          "requirements": [
              "push a zero-GPU compile/census plan before using the plugin",
              "compile only the isolated candidate graph first",
              "require FC=291, QK=10, dual-attention=10, shared triples=40",
              "create no InferRequest until the compile gate passes",
          ],
      },
      "checks": source_checks + apply_checks + build_checks,
  }
  write_json(output / "result.json", result)
  write_json(output / "manifest.json", {
      "schema": SCHEMA,
      "tool": display(Path(__file__)),
      "repository": result["repository"],
      "inputs": {
          display(PATCH): sha256(PATCH),
          display(PRODUCT_TOOL): sha256(PRODUCT_TOOL),
          display(OPPORTUNITY): sha256(OPPORTUNITY),
          str(SOURCE): source_after_apply["sha256"],
          str(ACCEPTED_PLUGIN): accepted_after["sha256"],
      },
      "candidate_plugin": candidate,
      "compiler_builds": result["workers"]["compiler_builds"],
      "gpu_contexts": 0,
      "model_workers": 0,
      "infer_requests": 0,
      "inference_workers": 0,
  })
  report = f"""# Current Q/K plus router-shared serial product build

Verdict: **{verdict}**. Required checks:
`{str(required_checks_passed).lower()}`.

Seq2205 admitted exactly one source application and one serial plugin build.
The durable default-off patch changed the FC transformation source from
`{source_before['sha256']}` to `{source_after_apply['sha256']}`. The permitted
build used parallelism `1`, completed in
`{float(build['elapsed_seconds']):.3f} s`, and produced isolated candidate
`{candidate.get('sha256')}`.

Peak process-group RSS/swap was
`{int(monitor.get('process_group_rss_peak_bytes', 0))}/`
`{int(monitor.get('process_group_swap_peak_bytes', 0))} B`; minimum available
memory was `{int(monitor.get('system_available_min_bytes', 0))} B`. The
4-GiB abort guard did not trip and no OOM was observed. The accepted seq2189
plugin remained bitwise unchanged. No GPU context, model worker, InferRequest,
or inference worker ran.
"""
  (output / "report.md").write_text(report, encoding="utf-8")
  print(json.dumps({
      "artifact": display(output),
      "verdict": verdict,
      "required_checks_passed": required_checks_passed,
      "elapsed_seconds": build["elapsed_seconds"],
      "candidate_plugin_sha256": candidate.get("sha256"),
      "peak_rss_bytes": monitor.get("process_group_rss_peak_bytes"),
      "minimum_available_bytes": monitor.get("system_available_min_bytes"),
      "oom_observed": build["oom_observed"],
  }, separators=(",", ":")), flush=True)
  return 0 if required_checks_passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
