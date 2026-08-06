#!/usr/bin/env python3
"""Measure corrected Xe3 PR5059 GMLP at the two exact product shapes."""

from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-onednn-gmlp-xe3-component-run-v0"
BASE_TOOL = ROOT / "tools/intel-qwen36-onednn-gmlp-exact-component-run.py"
PROVIDER_METRICS = ROOT / (
    "output/onednn-gmlp-xe3-provider-gate-"
    "20260731Tseq2228-clean/metrics.json")
PROVIDER_METRICS_SHA256 = (
    "997437a77fdcd15bd15c59f83a5c40a9ef4b6af04f1982b14dab7033e5b310a5")
SOURCE_WORKTREE = Path(
    "/home/intel/intel-qwen36-r0/source/oneDNN-862174-gmlp-exact")
ONEDNN_HEAD = "8621740ea5e600468c76a11a3c0c1616977f978d"
BUILD_DIR = Path(
    "/home/intel/intel-qwen36-r0/build/onednn-862174-gmlp-xe3-exact")
TEST_BINARY = BUILD_DIR / "tests/gtests/internals/test_internals_gmlp"
TEST_BINARY_SHA256 = (
    "506888812be9edb57e02a62a9d0550b22d0690718f4cebfc22ecbd4ea0addabc")
PAIRS_PER_SHAPE = 8
BOOTSTRAP_SEED = 22_290
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


def main() -> int:
  args = parse_args()
  output = args.output.resolve()
  output.mkdir(parents=True, exist_ok=False)
  required = (
      BASE_TOOL, PROVIDER_METRICS, SOURCE_WORKTREE, TEST_BINARY,
      BASE.SYSTEMD_RUN, BASE.SYSTEMCTL, BASE.TIME, BASE.INTEL_ICD)
  missing = [str(path) for path in required if not path.exists()]
  if missing:
    raise SystemExit("missing Xe3 GMLP component inputs: " + ", ".join(missing))

  repo = BASE.repository_state(output)
  provider = BASE.load_json(PROVIDER_METRICS)
  source_commit = BASE.git(SOURCE_WORKTREE, "rev-parse", "HEAD")
  source_dirty = BASE.git(
      SOURCE_WORKTREE, "status", "--short", "--untracked-files=all")
  initial_memory = BASE.proc_meminfo()
  provider_identity_ok = bool(
      BASE.sha256(PROVIDER_METRICS) == PROVIDER_METRICS_SHA256
      and provider["verdict"]["required_checks_passed"] is True
      and provider["verdict"]["paired_component_timing_admitted"] is True
      and provider["inputs"]["binary_sha256"] == TEST_BINARY_SHA256)
  identity_ok = bool(
      repo["branch"] == "main" and repo["pushed"] and not repo["dirty"]
      and provider_identity_ok
      and source_commit == ONEDNN_HEAD and source_dirty == ""
      and BASE.sha256(TEST_BINARY) == TEST_BINARY_SHA256
      and BASE.INTEL_ICD.read_text(encoding="utf-8").strip()
      == BASE.EXPECTED_INTEL_ICD
      and int(initial_memory.get("MemAvailable", 0)) >= PREFLIGHT_BYTES)
  if not identity_ok:
    raise SystemExit(
        "Xe3 component identity/repository/memory preflight failed before GPU")

  workers = []
  stop_reason = None
  for block in range(1, PAIRS_PER_SHAPE + 1):
    for shape_name in ("decode", "prefill"):
      row = BASE.run_worker(
          output, block, shape_name, SHAPES[shape_name],
          args.worker_timeout_s, args.poll_interval_s)
      workers.append(row)
      print(json.dumps({
          "block": block,
          "shape": shape_name,
          "returncode": row["returncode"],
          "elapsed_seconds": row["elapsed_seconds"],
          "provider_success_count": row[
              "parsed"]["provider_success_count"],
          "fallback_provider_success_count": len(
              row["parsed"]["fallback_provider_success_lines"]),
          "mismatches": row["parsed"]["mismatches"],
          "allowed": row["parsed"]["allowed"],
          "internal_ms": row["parsed"]["internal_ms"],
          "primitive_ms": row["parsed"]["primitive_ms"],
          "delta_ms": row["parsed"]["delta_ms"],
          "memory_available_min_bytes": row[
              "monitor"]["system_available_min_bytes"],
          "worker_valid": BASE.worker_valid(row),
      }, sort_keys=True), flush=True)
      if not BASE.worker_valid(row):
        stop_reason = (
            f"block {block} {shape_name} failed provider, correctness, "
            "timing, process, or memory checks")
        break
    if stop_reason:
      break

  by_shape = {
      shape_name: [
          row for row in workers if row["shape"] == shape_name]
      for shape_name in SHAPES}
  complete = all(
      len(rows) == PAIRS_PER_SHAPE for rows in by_shape.values())
  all_workers_valid = bool(
      complete and all(BASE.worker_valid(row) for row in workers))
  inference: dict[str, Any] = {}
  if all_workers_valid:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for shape_name in ("decode", "prefill"):
      deltas = [
          float(row["parsed"]["delta_ms"])
          for row in by_shape[shape_name]]
      result = BASE.bootstrap_median_delta(deltas, rng)
      result["bootstrap_seed"] = BOOTSTRAP_SEED
      result["delta_ucb_cap_ms"] = SHAPES[
          shape_name]["delta_ucb_cap_ms"]
      result["pass"] = (
          result["one_sided_95pct_ucb_delta_ms"]
          <= result["delta_ucb_cap_ms"])
      result["primitive_ms"] = [
          row["parsed"]["primitive_ms"] for row in by_shape[shape_name]]
      result["internal_ms"] = [
          row["parsed"]["internal_ms"] for row in by_shape[shape_name]]
      result["delta_ms"] = deltas
      inference[shape_name] = result

  performance_pass = bool(
      all_workers_valid
      and all(inference[shape]["pass"] for shape in SHAPES))
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
  all_memory_safe = all(
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
          "seq2228_provider_identity_exact",
          provider_identity_ok,
          provider_metrics=BASE.relative(PROVIDER_METRICS),
          provider_metrics_sha256=BASE.sha256(PROVIDER_METRICS)),
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
          "exact_eight_pairs_per_shape_completed",
          complete and len(workers) == 2 * PAIRS_PER_SHAPE,
          counts={key: len(value) for key, value in by_shape.items()},
          stop_reason=stop_reason),
      BASE.check(
          "workers_strictly_serial_unique_scoped_and_memory_safe",
          strictly_serial and len(units) == len(set(units))
          and all_memory_safe and min_available >= ABORT_BYTES,
          maximum_concurrent_workers=1,
          units=units,
          minimum_available_bytes=min_available,
          maximum_process_rss_bytes=peak_rss,
          maximum_cgroup_swap_bytes=peak_cgroup_swap),
      BASE.check(
          "all_workers_select_exact_provider_and_are_correct",
          all_workers_valid,
          required_provider=REQUIRED_PROVIDER,
          mismatch_rows=[{
              "block": row["block"],
              "shape": row["shape"],
              "mismatches": row["parsed"]["mismatches"],
              "allowed": row["parsed"]["allowed"],
          } for row in workers]),
      BASE.check(
          "prefill_delta_ucb_clears_registered_funding",
          bool(inference.get("prefill", {}).get("pass", False)),
          inference=inference.get("prefill")),
      BASE.check(
          "decode_delta_ucb_is_nonregressive",
          bool(inference.get("decode", {}).get("pass", False)),
          inference=inference.get("decode")),
      BASE.check(
          "no_product_model_or_infer_request_ran",
          True,
          model_workers_started=0,
          infer_requests_created=0,
          openvino_product_builds=0),
  ]
  passed = all(row["pass"] for row in checks)
  if all_workers_valid and not performance_pass:
    verdict_name = "reject_pr5059_v7_xe3_component_performance"
    next_action = (
        "close this exact PR5059 body without product integration and select "
        "the next distinct profile-backed kernel route")
  elif passed:
    verdict_name = "admit_exact_xe3_gmlp_product_integration_design"
    next_action = (
        "perform a zero-GPU exact OpenVINO integration source and version "
        "binding audit; do not build a product plugin yet")
  else:
    verdict_name = "hold_exact_xe3_gmlp_for_operational_failure_classification"
    next_action = (
        "classify the first operational/provider/correctness failure before "
        "selecting another route")
  verdict = {
      "required_checks_passed": passed,
      "component_measurement_valid": all_workers_valid,
      "performance_passed": performance_pass,
      "component_promotable": passed,
      "product_integration_design_admitted": passed,
      "product_build_admitted": False,
      "verdict": verdict_name,
      "next_action": next_action,
  }
  metrics = {
      "schema": SCHEMA,
      "workstream": WS,
      "created_at": datetime.now(timezone.utc).isoformat(),
      "git": repo,
      "inputs": {
          "provider_metrics": {
              "path": BASE.relative(PROVIDER_METRICS),
              "sha256": BASE.sha256(PROVIDER_METRICS),
          },
          "source_commit": source_commit,
          "binary": str(TEST_BINARY),
          "binary_sha256": BASE.sha256(TEST_BINARY),
      },
      "protocol": {
          "shape_order_per_block": ["decode", "prefill"],
          "pairs_per_shape": PAIRS_PER_SHAPE,
          "maximum_concurrent_workers": 1,
          "worker_timeout_s": args.worker_timeout_s,
          "required_provider": REQUIRED_PROVIDER,
          "bootstrap_resamples": BASE.BOOTSTRAP_RESAMPLES,
          "bootstrap_seed": BOOTSTRAP_SEED,
          "shapes": SHAPES,
      },
      "workers": workers,
      "inference": inference,
      "memory": {
          "initial": initial_memory,
          "preflight_bytes": PREFLIGHT_BYTES,
          "abort_below_bytes": ABORT_BYTES,
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
  prefill = inference.get("prefill", {})
  decode = inference.get("decode", {})
  (output / "report.md").write_text(
      "# oneDNN PR5059 exact Xe3 GMLP component\n\n"
      f"- Required checks: `{passed}`\n"
      f"- Measurement valid: `{all_workers_valid}`\n"
      f"- Performance pass: `{performance_pass}`\n"
      f"- Verdict: `{verdict_name}`\n"
      f"- Workers: `{len(workers)}` strictly serial\n"
      f"- Provider: `{REQUIRED_PROVIDER}`\n"
      f"- Prefill delta median/UCB/cap ms: "
      f"`{prefill.get('point_median_delta_ms')}/"
      f"{prefill.get('one_sided_95pct_ucb_delta_ms')}/-0.001209`\n"
      f"- Decode delta median/UCB/cap ms: "
      f"`{decode.get('point_median_delta_ms')}/"
      f"{decode.get('one_sided_95pct_ucb_delta_ms')}/0`\n"
      f"- Minimum available / peak RSS / peak cgroup swap B: "
      f"`{min_available}/{peak_rss}/{peak_cgroup_swap}`\n"
      "- Product model/InferRequest/build: `0/0/0`\n",
      encoding="utf-8")
  print(json.dumps({
      "output": BASE.relative(output),
      "required_checks_passed": passed,
      "measurement_valid": all_workers_valid,
      "performance_passed": performance_pass,
      "verdict": verdict_name,
      "workers_started": len(workers),
      "prefill_delta_ucb_ms": prefill.get(
          "one_sided_95pct_ucb_delta_ms"),
      "decode_delta_ucb_ms": decode.get(
          "one_sided_95pct_ucb_delta_ms"),
      "minimum_available_bytes": min_available,
      "maximum_process_rss_bytes": peak_rss,
  }, sort_keys=True), flush=True)
  return 0 if passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
