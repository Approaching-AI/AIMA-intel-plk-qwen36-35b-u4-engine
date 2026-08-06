#!/usr/bin/env python3
"""Gate the corrected Xe3 PR5059 GMLP binary on the two exact shapes."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-onednn-gmlp-xe3-provider-gate-v0"
BASE_TOOL = ROOT / "tools/intel-qwen36-onednn-gmlp-exact-component-run.py"
BUILD_METRICS = ROOT / (
    "output/onednn-gmlp-xe3-component-build-"
    "20260731Tseq2227-clean/metrics.json")
BUILD_METRICS_SHA256 = (
    "b38d44c1977d0d6459468163d1d2444f218fbe641afaaf77aad7178b88964340")
SOURCE_WORKTREE = Path(
    "/home/intel/intel-qwen36-r0/source/oneDNN-862174-gmlp-exact")
ONEDNN_HEAD = "8621740ea5e600468c76a11a3c0c1616977f978d"
BUILD_DIR = Path(
    "/home/intel/intel-qwen36-r0/build/onednn-862174-gmlp-xe3-exact")
TEST_BINARY = BUILD_DIR / "tests/gtests/internals/test_internals_gmlp"
TEST_BINARY_SHA256 = (
    "506888812be9edb57e02a62a9d0550b22d0690718f4cebfc22ecbd4ea0addabc")
PREFLIGHT_BYTES = 8 * 1024**3
ABORT_BYTES = 4 * 1024**3


def load_base() -> Any:
  spec = importlib.util.spec_from_file_location(
      "intel_qwen36_onednn_gmlp_exact_component_run", BASE_TOOL)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load base runner: {BASE_TOOL}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  module.BUILD_DIR = BUILD_DIR
  module.TEST_BINARY = TEST_BINARY
  module.TEST_BINARY_SHA256 = TEST_BINARY_SHA256
  return module


BASE = load_base()
SHAPES = BASE.SHAPES
REQUIRED_PROVIDER = BASE.REQUIRED_PROVIDER


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", required=True, type=Path)
  parser.add_argument("--worker-timeout-s", default=600.0, type=float)
  parser.add_argument("--poll-interval-s", default=0.1, type=float)
  args = parser.parse_args()
  if args.worker_timeout_s <= 0 or args.poll_interval_s <= 0:
    parser.error("timeout and poll interval must be positive")
  return args


def summarize_worker(row: dict[str, Any]) -> dict[str, Any]:
  parsed = row["parsed"]
  return {
      "shape": row["shape"],
      "returncode": row["returncode"],
      "elapsed_seconds": row["elapsed_seconds"],
      "provider_count": parsed["provider_count"],
      "provider_success_count": parsed["provider_success_count"],
      "fallback_provider_success_count": len(
          parsed["fallback_provider_success_lines"]),
      "mismatches": parsed["mismatches"],
      "allowed": parsed["allowed"],
      "internal_ms": parsed["internal_ms"],
      "primitive_ms": parsed["primitive_ms"],
      "delta_ms": parsed["delta_ms"],
      "worker_valid": BASE.worker_valid(row),
      "minimum_available_bytes": row[
          "monitor"]["system_available_min_bytes"],
      "maximum_process_rss_bytes": row[
          "monitor"]["process_rss_peak_bytes"],
  }


def main() -> int:
  args = parse_args()
  output = args.output.resolve()
  output.mkdir(parents=True, exist_ok=False)

  required = (
      BASE_TOOL, BUILD_METRICS, SOURCE_WORKTREE, TEST_BINARY,
      BASE.SYSTEMD_RUN, BASE.SYSTEMCTL, BASE.TIME, BASE.INTEL_ICD)
  missing = [str(path) for path in required if not path.exists()]
  if missing:
    raise SystemExit("missing Xe3 GMLP provider inputs: " + ", ".join(missing))

  repo = BASE.repository_state(output)
  build = BASE.load_json(BUILD_METRICS)
  source_commit = BASE.git(SOURCE_WORKTREE, "rev-parse", "HEAD")
  source_dirty = BASE.git(
      SOURCE_WORKTREE, "status", "--short", "--untracked-files=all")
  initial_memory = BASE.proc_meminfo()
  build_identity_ok = bool(
      BASE.sha256(BUILD_METRICS) == BUILD_METRICS_SHA256
      and build["verdict"]["required_checks_passed"] is True
      and build["verdict"]["xe3_component_provider_runs_admitted"] is True
      and build["binary"]["sha256"] == TEST_BINARY_SHA256
      and build["binary"]["path"] == str(TEST_BINARY))
  identity_ok = bool(
      repo["branch"] == "main" and repo["pushed"] and not repo["dirty"]
      and build_identity_ok
      and source_commit == ONEDNN_HEAD and source_dirty == ""
      and BASE.sha256(TEST_BINARY) == TEST_BINARY_SHA256
      and BASE.INTEL_ICD.read_text(encoding="utf-8").strip()
      == BASE.EXPECTED_INTEL_ICD
      and int(initial_memory.get("MemAvailable", 0)) >= PREFLIGHT_BYTES)
  if not identity_ok:
    raise SystemExit(
        "Xe3 provider identity/repository/memory preflight failed before GPU")

  workers = []
  decode = BASE.run_worker(
      output, 1, "decode", SHAPES["decode"],
      args.worker_timeout_s, args.poll_interval_s)
  workers.append(decode)
  print(json.dumps(summarize_worker(decode), sort_keys=True), flush=True)

  decode_pass = BASE.worker_valid(decode)
  if decode_pass:
    prefill = BASE.run_worker(
        output, 1, "prefill", SHAPES["prefill"],
        args.worker_timeout_s, args.poll_interval_s)
    workers.append(prefill)
    print(json.dumps(summarize_worker(prefill), sort_keys=True), flush=True)

  by_shape = {row["shape"]: row for row in workers}
  prefill_pass = bool(
      "prefill" in by_shape and BASE.worker_valid(by_shape["prefill"]))
  units = [row["scope"]["unit"] for row in workers]
  intervals = [
      (float(row["started_monotonic"]), float(row["finished_monotonic"]))
      for row in workers]
  strictly_serial = all(
      intervals[index][0] >= intervals[index - 1][1]
      for index in range(1, len(intervals)))
  min_available = min(
      [int(initial_memory["MemAvailable"])]
      + [int(row["monitor"]["system_available_min_bytes"])
         for row in workers])
  peak_rss = max(
      [0] + [int(row["monitor"]["process_rss_peak_bytes"])
             for row in workers])
  peak_cgroup_swap = max(
      [0] + [int(row["monitor"]["cgroup_swap_peak_bytes"])
             for row in workers])
  no_oom = all(
      not row["oom_observed"]
      and not row["memory_guard_tripped"]
      and int(row["monitor"]["memory_events_max"].get("oom", 0)) == 0
      and int(row["monitor"]["memory_events_max"].get("oom_kill", 0)) == 0
      and int(row["monitor"]["memory_events_max"].get(
          "oom_group_kill", 0)) == 0
      for row in workers)

  checks = [
      BASE.check(
          "repository_clean_and_pushed_at_gate",
          repo["branch"] == "main" and repo["pushed"] and not repo["dirty"],
          **repo),
      BASE.check(
          "seq2227_build_identity_exact",
          build_identity_ok,
          build_metrics=BASE.relative(BUILD_METRICS),
          build_metrics_sha256=BASE.sha256(BUILD_METRICS),
          binary_sha256=BASE.sha256(TEST_BINARY)),
      BASE.check(
          "source_and_binary_identity_exact",
          source_commit == ONEDNN_HEAD and source_dirty == ""
          and BASE.sha256(TEST_BINARY) == TEST_BINARY_SHA256,
          source_commit=source_commit,
          source_dirty=source_dirty,
          binary_sha256=BASE.sha256(TEST_BINARY)),
      BASE.check(
          "opencl_icd_discovery_path_exact",
          BASE.INTEL_ICD.read_text(encoding="utf-8").strip()
          == BASE.EXPECTED_INTEL_ICD,
          vendor_directory=str(BASE.OCL_ICD_VENDOR_DIR),
          icd_file=str(BASE.INTEL_ICD),
          icd_library=BASE.INTEL_ICD.read_text(
              encoding="utf-8").strip()),
      BASE.check(
          "decode_exact_provider_correctness_and_timing_pass",
          decode_pass,
          worker=summarize_worker(decode)),
      BASE.check(
          "prefill_ran_only_after_decode_and_passed",
          decode_pass and prefill_pass and len(workers) == 2,
          prefill_started=bool("prefill" in by_shape),
          worker=(
              summarize_worker(by_shape["prefill"])
              if "prefill" in by_shape else None)),
      BASE.check(
          "workers_strictly_serial_unique_scoped_and_memory_safe",
          strictly_serial and len(units) == len(set(units))
          and no_oom and min_available >= ABORT_BYTES,
          maximum_concurrent_workers=1,
          units=units,
          minimum_available_bytes=min_available,
          maximum_process_rss_bytes=peak_rss,
          maximum_cgroup_swap_bytes=peak_cgroup_swap,
          no_oom=no_oom),
      BASE.check(
          "no_product_model_or_infer_request_ran",
          True,
          model_workers_started=0,
          infer_requests_created=0,
          openvino_product_builds=0),
  ]
  passed = all(row["pass"] for row in checks)
  failure_text = "\n".join(
      Path(row[path_key]).read_text(encoding="utf-8", errors="replace")
      for row in workers for path_key in ("stdout", "stderr"))
  unsupported_architecture = "Unsupported architecture" in failure_text
  no_matching_kernel = "No matching kernel" in failure_text
  finite_deltas = {
      name: row["parsed"]["delta_ms"]
      for name, row in by_shape.items()
      if row["parsed"]["delta_ms"] is not None
      and math.isfinite(float(row["parsed"]["delta_ms"]))
  }
  verdict = {
      "required_checks_passed": passed,
      "paired_component_timing_admitted": passed,
      "product_build_admitted": False,
      "verdict": (
          "admit_exact_xe3_gmlp_paired_component_timing"
          if passed else
          "hold_exact_xe3_gmlp_for_failure_classification"),
      "next_if_pass": (
          "run eight strictly serial interleaved decode/prefill component "
          "pairs and apply the registered one-sided 95% delta bounds"),
      "next_if_fail": (
          "classify the first exact Xe3 provider or generator failure against "
          "current upstream source before selecting another route"),
  }
  metrics = {
      "schema": SCHEMA,
      "workstream": WS,
      "created_at": datetime.now(timezone.utc).isoformat(),
      "git": repo,
      "inputs": {
          "build_metrics": {
              "path": BASE.relative(BUILD_METRICS),
              "sha256": BASE.sha256(BUILD_METRICS),
          },
          "source_commit": source_commit,
          "binary": str(TEST_BINARY),
          "binary_sha256": BASE.sha256(TEST_BINARY),
      },
      "protocol": {
          "shape_order": ["decode", "prefill_if_decode_passes"],
          "maximum_concurrent_workers": 1,
          "worker_timeout_s": args.worker_timeout_s,
          "required_provider": REQUIRED_PROVIDER,
          "shapes": SHAPES,
          "preflight_bytes": PREFLIGHT_BYTES,
          "abort_below_bytes": ABORT_BYTES,
      },
      "workers": workers,
      "observations": {
          "unsupported_architecture": unsupported_architecture,
          "no_matching_kernel": no_matching_kernel,
          "finite_delta_ms": finite_deltas,
      },
      "memory": {
          "initial": initial_memory,
          "minimum_available_bytes": min_available,
          "maximum_process_rss_bytes": peak_rss,
          "maximum_cgroup_swap_bytes": peak_cgroup_swap,
      },
      "process_census": {
          "workers_started": len(workers),
          "maximum_concurrent_workers": 1,
          "gpu_contexts_expected": len(workers),
          "model_workers_started": 0,
          "infer_requests_created": 0,
          "openvino_product_builds": 0,
      },
      "checks": checks,
      "verdict": verdict,
  }
  BASE.write_json(output / "metrics.json", metrics)
  (output / "report.md").write_text(
      "# oneDNN PR5059 exact Xe3 GMLP provider gate\n\n"
      f"- Required checks: `{passed}`\n"
      f"- Verdict: `{verdict['verdict']}`\n"
      f"- Workers: `{len(workers)}` strictly serial\n"
      f"- Provider: `{REQUIRED_PROVIDER}`\n"
      f"- Decode pass: `{decode_pass}`\n"
      f"- Prefill pass: `{prefill_pass}`\n"
      f"- Unsupported architecture / no matching kernel: "
      f"`{unsupported_architecture}/{no_matching_kernel}`\n"
      f"- Finite delta ms: `{finite_deltas}`\n"
      f"- Minimum available / peak RSS / peak cgroup swap B: "
      f"`{min_available}/{peak_rss}/{peak_cgroup_swap}`\n"
      "- Product model/InferRequest/build: `0/0/0`\n",
      encoding="utf-8")
  print(json.dumps({
      "output": BASE.relative(output),
      "required_checks_passed": passed,
      "verdict": verdict["verdict"],
      "workers_started": len(workers),
      "decode_pass": decode_pass,
      "prefill_pass": prefill_pass,
      "unsupported_architecture": unsupported_architecture,
      "no_matching_kernel": no_matching_kernel,
      "finite_delta_ms": finite_deltas,
      "minimum_available_bytes": min_available,
      "maximum_process_rss_bytes": peak_rss,
  }, sort_keys=True), flush=True)
  return 0 if passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
