#!/usr/bin/env python3
"""Gate one exact parallel block-top8 LM-head fallback component cut.

The runner holds the full-I8 matvec output fixed and interleaves the accepted
lane0-serial block top-8 with one eight-round work-group reduction in ABBA
order.  Every captured hidden receives at least 20 paired blocks.  Full
fallback baseline/candidate outputs must remain bitwise equal before a
one-sided 95% saving lower bound can admit product integration.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import signal
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
BASE_PATH = (
    REPO /
    "tools/intel-qwen36-openvino-lm-head-gated-exact-component.py")
SPEC = importlib.util.spec_from_file_location("iq36_exact_component_base", BASE_PATH)
if SPEC is None or SPEC.loader is None:
  raise RuntimeError(f"cannot load component base: {BASE_PATH}")
BASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE)

WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-openvino-lm-head-parallel-block-topk-component-v1"
BASELINE = REPO / (
    "output/openvino-lm-head-gated-exact-component-"
    "20260731Tseq2187-clean/result.json")
REQUIRED_SAVING_US = 11.20375000000351
REFERENCE_PHASES = (63, 96, 129)
BOOTSTRAP_RESAMPLES = 20_000
BOOTSTRAP_SEED = 0x2188


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--baseline", type=Path, default=BASELINE)
  parser.add_argument("--capture-dir", type=Path, default=BASE.CAPTURE_DIR)
  parser.add_argument("--reference-dir", type=Path, default=BASE.REFERENCE_DIR)
  parser.add_argument("--cmake", type=Path, default=BASE.CMAKE)
  parser.add_argument("--build-dir", type=Path, default=BASE.BUILD_DIR)
  parser.add_argument("--warmup", type=int, default=3)
  parser.add_argument("--blocks", type=int, default=20)
  parser.add_argument("--first-phase", type=int, default=0)
  parser.add_argument("--last-phase", type=int, default=129)
  parser.add_argument("--timeout-s", type=int, default=600)
  parser.add_argument("--min-available-gib", type=float, default=8.0)
  parser.add_argument("--abort-below-available-gib", type=float, default=4.0)
  parser.add_argument("--poll-interval-s", type=float, default=0.1)
  args = parser.parse_args()
  if args.warmup < 0 or args.blocks < 20:
    parser.error("warmup must be nonnegative and blocks at least 20")
  if (args.first_phase < 0 or args.last_phase < args.first_phase or
      args.last_phase > 4096):
    parser.error("phase range must satisfy 0 <= first <= last <= 4096")
  if args.timeout_s <= 0 or args.poll_interval_s <= 0.0:
    parser.error("timeout and poll interval must be positive")
  if (args.min_available_gib < 0.0 or
      args.abort_below_available_gib < 0.0 or
      args.abort_below_available_gib > args.min_available_gib):
    parser.error("memory thresholds are invalid")
  return args


def source_contract() -> dict[str, Any]:
  source = BASE.MODULE_SOURCE.read_text(encoding="utf-8")
  begin = source.find(
      "__kernel void "
      "iq36_lm_head_gated_exact_component_parallel_block_topk8_f16")
  end = source.find(
      "__kernel void iq36_lm_head_gated_exact_component_topk8_merge_f32")
  body = source[begin:end] if begin >= 0 and end > begin else ""
  return {
      "candidate_kernel_occurrences": source.count(
          "__kernel void "
          "iq36_lm_head_gated_exact_component_parallel_block_topk8_f16"),
      "candidate_body_sha256":
          hashlib.sha256(body.encode("utf-8")).hexdigest(),
      "eight_round_constant_loop":
          "selected < IQ36_TOPK" in body and
          "#define IQ36_TOPK 8U" in source,
      "collective_max_count": body.count("work_group_reduce_max"),
      "collective_tie_min_count": body.count("work_group_reduce_min"),
      "exact_tie_rule":
          "lane_value == maximum ? (int)row : 0x7fffffff" in body,
      "selected_lane_is_removed":
          "if ((int)row == winner) lane_value = -INFINITY;" in body,
      "baseline_kernel_still_present": source.count(
          "__kernel void "
          "iq36_lm_head_gated_exact_component_block_topk8_f16") == 1,
      "product_patch_sha256": BASE.sha256(BASE.PRODUCT_PATCH),
  }


def launch_paired(
    args: argparse.Namespace, directory: Path, module: Path,
) -> dict[str, Any]:
  directory.mkdir(parents=True)
  stdout_path = directory / "component.stdout"
  stderr_path = directory / "component.stderr"
  command = [
      str(BASE.BINARY), str(BASE.MODEL_BIN), str(module),
      str(args.capture_dir), str(directory),
      str(args.warmup), str(args.blocks),
      str(args.first_phase), str(args.last_phase), "paired-block-topk",
  ]
  preflight = BASE.wait_for_memory(
      int(args.min_available_gib * 1024**3))
  abort_bytes = int(args.abort_below_available_gib * 1024**3)
  started = time.monotonic()
  monitor: dict[str, Any] = {
      "process_rss_peak_bytes": 0,
      "process_swap_peak_bytes": 0,
      "system_available_min_bytes": None,
      "system_swap_used_peak_bytes": 0,
      "sample_count": 0,
  }
  timed_out = False
  guard_tripped = False
  with stdout_path.open("w", encoding="utf-8") as stdout_handle, \
       stderr_path.open("w", encoding="utf-8") as stderr_handle:
    process = subprocess.Popen(
        command, cwd=REPO, stdout=stdout_handle, stderr=stderr_handle,
        text=True, start_new_session=True)
    while process.poll() is None:
      current = BASE.meminfo()
      process_row = BASE.process_memory(process.pid)
      available = int(current.get("MemAvailable", 0))
      swap_used = max(
          0, int(current.get("SwapTotal", 0)) -
          int(current.get("SwapFree", 0)))
      monitor["process_rss_peak_bytes"] = max(
          int(monitor["process_rss_peak_bytes"]),
          int(process_row["VmRSS"]))
      monitor["process_swap_peak_bytes"] = max(
          int(monitor["process_swap_peak_bytes"]),
          int(process_row["VmSwap"]))
      minimum = monitor["system_available_min_bytes"]
      monitor["system_available_min_bytes"] = (
          available if minimum is None else min(int(minimum), available))
      monitor["system_swap_used_peak_bytes"] = max(
          int(monitor["system_swap_used_peak_bytes"]), swap_used)
      monitor["sample_count"] = int(monitor["sample_count"]) + 1
      if available < abort_bytes:
        guard_tripped = True
        BASE.stop_process_group(process, signal.SIGTERM)
        break
      if time.monotonic() - started > args.timeout_s:
        timed_out = True
        BASE.stop_process_group(process, signal.SIGTERM)
        break
      time.sleep(args.poll_interval_s)
    returncode = process.wait()
  stderr = (
      stderr_path.read_text(encoding="utf-8", errors="replace")
      if stderr_path.is_file() else "")
  lower_stderr = stderr.lower()
  oom = (
      returncode in (-9, 137) or
      "out of memory" in lower_stderr or
      "ze_result_error_out_of_device_memory" in lower_stderr or
      "ze_result_error_out_of_host_memory" in lower_stderr)
  return {
      "command": command,
      "returncode": returncode,
      "timed_out": timed_out,
      "memory_preflight": preflight,
      "memory_guard": {
          "abort_below_bytes": abort_bytes,
          "tripped": guard_tripped,
      },
      "monitor": monitor,
      "oom_observed": oom,
      "elapsed_seconds": time.monotonic() - started,
      "stdout": BASE.display_path(stdout_path),
      "stderr": BASE.display_path(stderr_path),
      "result": BASE.parse_worker_stdout(stdout_path),
  }


def lower_bootstrap_median(values: list[float]) -> float:
  if not values or any(not math.isfinite(value) for value in values):
    return -math.inf
  rng = random.Random(BOOTSTRAP_SEED)
  medians = sorted(
      statistics.median(rng.choices(values, k=len(values)))
      for _ in range(BOOTSTRAP_RESAMPLES))
  rank = max(1, math.ceil(0.05 * len(medians)))
  return float(medians[rank - 1])


def distribution(values: list[float]) -> dict[str, Any]:
  if not values or any(not math.isfinite(value) for value in values):
    return {
        "sample_count": len(values), "min_us": math.nan,
        "median_us": math.nan, "mean_us": math.nan,
        "p95_us": math.nan, "max_us": math.nan,
    }
  ordered = sorted(values)
  rank = min(len(ordered) - 1, max(0, math.ceil(0.95 * len(ordered)) - 1))
  return {
      "sample_count": len(values),
      "min_us": min(values),
      "median_us": statistics.median(values),
      "mean_us": statistics.mean(values),
      "p95_us": ordered[rank],
      "max_us": max(values),
  }


def audit_worker(
    launched: dict[str, Any], phases: tuple[int, ...],
    reference_dir: Path,
) -> dict[str, Any]:
  result = launched.get("result", {})
  rows = {
      int(row["phase"]): row
      for row in result.get("phases", [])
      if isinstance(row, dict) and "phase" in row
  }
  baseline_samples: list[float] = []
  candidate_samples: list[float] = []
  saving_samples: list[float] = []
  saving_wall_samples: list[float] = []
  hidden_median_savings: list[float] = []
  output_hashes = []
  anchor_hashes = []
  arithmetic_exact = True
  expected_blocks = int(result.get("paired_blocks_per_hidden", -1))
  for phase in phases:
    row = rows.get(phase, {})
    baseline = [float(value) for value in
                row.get("baseline_block_samples_us", [])]
    candidate = [float(value) for value in
                 row.get("candidate_block_samples_us", [])]
    saving = [float(value) for value in
              row.get("saving_block_samples_us", [])]
    saving_wall = [float(value) for value in
                   row.get("saving_wall_samples_us", [])]
    if not (len(baseline) == len(candidate) == len(saving) ==
            expected_blocks):
      arithmetic_exact = False
    for lhs, rhs, delta in zip(baseline, candidate, saving):
      if abs((lhs - rhs) - delta) > 1e-6:
        arithmetic_exact = False
    baseline_samples.extend(baseline)
    candidate_samples.extend(candidate)
    saving_samples.extend(saving)
    saving_wall_samples.extend(saving_wall)
    if saving:
      hidden_median_savings.append(statistics.median(saving))
    raw_output = str(row.get("output", ""))
    output = Path(raw_output) if raw_output else Path("/nonexistent")
    if raw_output and not output.is_absolute():
      output = REPO / output
    output_hash = BASE.sha256(output) if output.is_file() else None
    output_hashes.append({
        "phase": phase,
        "path": BASE.display_path(output) if raw_output else None,
        "sha256": output_hash,
    })
    if phase in REFERENCE_PHASES:
      reference = reference_dir / f"step{phase:04d}-logits.f32"
      reference_hash = BASE.sha256(reference)
      anchor_hashes.append({
          "phase": phase,
          "candidate_sha256": output_hash,
          "reference_sha256": reference_hash,
          "match": output_hash == reference_hash,
      })
  lcb = lower_bootstrap_median(hidden_median_savings)
  return {
      "mode": result.get("mode"),
      "device_name": result.get("device_name"),
      "phase_count": len(rows),
      "worker_required_checks_passed":
          result.get("required_checks_passed"),
      "all_outputs_bitwise_equal":
          result.get("all_outputs_bitwise_equal"),
      "all_selected_ids_equal": result.get("all_selected_ids_equal"),
      "all_finite": result.get("all_finite"),
      "all_timings_positive": result.get("all_timings_positive"),
      "paired_blocks_per_hidden": expected_blocks,
      "schedule": result.get("schedule"),
      "baseline_block_resources": result.get("baseline_block_resources", {}),
      "candidate_block_resources":
          result.get("candidate_block_resources", {}),
      "block_arithmetic_exact": arithmetic_exact,
      "baseline": distribution(baseline_samples),
      "candidate": distribution(candidate_samples),
      "saving": distribution(saving_samples),
      "saving_wall": distribution(saving_wall_samples),
      "hidden_median_saving": distribution(hidden_median_savings),
      "performance_inference": {
          "method": "paired_one_sided_percentile_bootstrap_median",
          "schedule": "ABBA",
          "statistical_unit": "per-hidden median of paired block savings",
          "confidence": 0.95,
          "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
          "bootstrap_seed": BOOTSTRAP_SEED,
          "hidden_count": len(hidden_median_savings),
          "paired_block_count": len(saving_samples),
          "point_estimate_saving_us":
              statistics.median(hidden_median_savings)
              if hidden_median_savings else -math.inf,
          "lower_confidence_bound_saving_us": lcb,
          "required_saving_us": REQUIRED_SAVING_US,
          "rate_pass": lcb > REQUIRED_SAVING_US,
      },
      "output_hashes": output_hashes,
      "anchor_hashes": anchor_hashes,
      "anchor_bitwise_matches": sum(row["match"] for row in anchor_hashes),
      "phase_rows": [rows.get(phase, {}) for phase in phases],
  }


def write_summary(output: Path, result: dict[str, Any]) -> None:
  repeat = result["repeat_audit"]
  confirm = result["confirm_audit"]
  lines = [
      "# OpenVINO LM-head parallel block-top8 component",
      "",
      f"- verdict: `{result['verdict']}`",
      f"- required checks passed: "
      f"`{str(result['required_checks_passed']).lower()}`",
      f"- repeat/confirm baseline median: "
      f"`{repeat['baseline']['median_us']:.3f}` / "
      f"`{confirm['baseline']['median_us']:.3f} us`",
      f"- repeat/confirm candidate median: "
      f"`{repeat['candidate']['median_us']:.3f}` / "
      f"`{confirm['candidate']['median_us']:.3f} us`",
      f"- repeat/confirm saving median: "
      f"`{repeat['saving']['median_us']:.3f}` / "
      f"`{confirm['saving']['median_us']:.3f} us`",
      f"- repeat/confirm one-sided 95% saving LCB: "
      f"`{repeat['performance_inference']['lower_confidence_bound_saving_us']:.3f}` / "
      f"`{confirm['performance_inference']['lower_confidence_bound_saving_us']:.3f} us`",
      f"- required product cut: `{REQUIRED_SAVING_US:.3f} us`",
      f"- candidate local/private/spill bytes: "
      f"`{repeat['candidate_block_resources'].get('local_mem_bytes')} / "
      f"{repeat['candidate_block_resources'].get('private_mem_bytes')} / "
      f"{repeat['candidate_block_resources'].get('spill_mem_bytes')}`",
      f"- full forced-fallback equality: "
      f"`{repeat['phase_count']}/{repeat['phase_count']}` / "
      f"`{confirm['phase_count']}/{confirm['phase_count']}`",
      f"- accepted fallback anchors bitwise exact: "
      f"`{repeat['anchor_bitwise_matches']}/{len(REFERENCE_PHASES)}` / "
      f"`{confirm['anchor_bitwise_matches']}/{len(REFERENCE_PHASES)}`",
      f"- OOM/guard repeat/confirm: "
      f"`{result['repeat_worker']['oom_observed']}` / "
      f"`{result['repeat_worker']['memory_guard']['tripped']}`, "
      f"`{result['confirm_worker']['oom_observed']}` / "
      f"`{result['confirm_worker']['memory_guard']['tripped']}`",
      "",
      "This admits component integration only. No product plugin or model "
      "worker was built or launched, and no product speedup is claimed.",
      "",
  ]
  (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  args = parse_args()
  output = args.output.resolve()
  generated = output / "generated"
  generated.mkdir(parents=True, exist_ok=False)
  args.baseline = args.baseline.resolve()
  args.capture_dir = args.capture_dir.resolve()
  args.reference_dir = args.reference_dir.resolve()
  args.build_dir = args.build_dir.resolve()
  phases = tuple(range(args.first_phase, args.last_phase + 1))
  required = [
      BASE.MODEL_BIN, args.cmake, BASE.MODULE_SOURCE, BASE.CPP_SOURCE,
      BASE.BOUNDARIES, BASE.PRODUCT_PATCH, args.baseline,
      *[
          args.capture_dir / f"step{phase:04d}-lm-head-input.f32"
          for phase in phases
      ],
      *[
          args.reference_dir / f"step{phase:04d}-logits.f32"
          for phase in REFERENCE_PHASES if phase in phases
      ],
  ]
  missing = [str(path) for path in required if not path.is_file()]
  if missing:
    raise SystemExit("missing component inputs: " + ", ".join(missing))
  state = BASE.git_state(output)
  concurrent = BASE.other_gpu_workers()
  if concurrent:
    raise RuntimeError(f"concurrent GPU worker detected: {concurrent}")
  baseline = BASE.load_json(args.baseline)
  contract = source_contract()

  module_compile = BASE.run([
      "ocloc", "compile", "-file", str(BASE.MODULE_SOURCE),
      "-device", "0xb080", "-output", "iq36_parallel_block_topk",
      "-out_dir", str(generated), "-output_no_suffix", "--format", "zebin",
      "-options", "-cl-std=CL3.0", "-q",
  ], 120)
  module = generated / "iq36_parallel_block_topk.bin"
  module_validate = (
      BASE.run(["ocloc", "validate", "-file", str(module)], 60)
      if module.is_file() else {
          "command": [], "returncode": 125, "stdout": "",
          "stderr": "module missing", "timed_out": False,
      })
  configure = BASE.run([
      str(args.cmake), "-S", str(REPO / "engine"), "-B",
      str(args.build_dir), "-DCMAKE_BUILD_TYPE=Release",
  ])
  build = (
      BASE.run([
          str(args.cmake), "--build", str(args.build_dir),
          "--target", BASE.TARGET, "-j", "1",
      ], 300)
      if configure["returncode"] == 0 else {
          "command": [], "returncode": 125, "stdout": "",
          "stderr": "configure failed", "timed_out": False,
      })
  link_map = (
      BASE.run(["ldd", str(BASE.BINARY)], 30)
      if BASE.BINARY.is_file() else {
          "command": [], "returncode": 125, "stdout": "",
          "stderr": "binary missing", "timed_out": False,
      })
  runnable = (
      module_compile["returncode"] == 0 and
      module_validate["returncode"] == 0 and
      build["returncode"] == 0 and
      module.is_file() and BASE.BINARY.is_file())
  repeat = (
      launch_paired(args, output / "repeat", module)
      if runnable else {
          "returncode": 125, "timed_out": False,
          "memory_guard": {"tripped": False},
          "oom_observed": False, "result": {},
      })
  repeat_audit = audit_worker(repeat, phases, args.reference_dir)
  confirm = (
      launch_paired(args, output / "confirm", module)
      if repeat["returncode"] == 0 else {
          "returncode": 125, "timed_out": False,
          "memory_guard": {"tripped": False},
          "oom_observed": False, "result": {},
      })
  confirm_audit = audit_worker(confirm, phases, args.reference_dir)
  output_hash_match = all(
      lhs["sha256"] is not None and lhs["sha256"] == rhs["sha256"]
      for lhs, rhs in zip(
          repeat_audit["output_hashes"], confirm_audit["output_hashes"]))
  expected_blocks = len(phases) * args.blocks
  baseline_patch_hash = (
      baseline.get("inputs", {}).get("sources", {}).get(
          BASE.display_path(BASE.PRODUCT_PATCH)))
  lower_links = (
      link_map.get("stdout", "") + link_map.get("stderr", "")).lower()
  checks = [
      BASE.check("repository_clean_at_gate", not state["dirty"], git=state),
      BASE.check("no_concurrent_gpu_worker_at_launch", not concurrent,
                 concurrent=concurrent),
      BASE.check("seq2187_stage_baseline_is_accepted",
                 baseline.get("required_checks_passed") is True and
                 baseline.get("verdict") ==
                    "admit_stage_separated_gated_exact_component_baseline"),
      BASE.check("exactly_one_source_bound_parallel_block_topk_cut",
                 contract["candidate_kernel_occurrences"] == 1 and
                 contract["eight_round_constant_loop"] and
                 contract["collective_max_count"] == 1 and
                 contract["collective_tie_min_count"] == 1 and
                 contract["exact_tie_rule"] and
                 contract["selected_lane_is_removed"] and
                 contract["baseline_kernel_still_present"],
                 source_contract=contract),
      BASE.check("accepted_product_patch_is_unchanged",
                 baseline_patch_hash == contract["product_patch_sha256"],
                 baseline_sha256=baseline_patch_hash,
                 current_sha256=contract["product_patch_sha256"]),
      BASE.check("module_compiles_and_validates_for_ptl",
                 module_compile["returncode"] == 0 and
                 module_validate["returncode"] == 0 and module.is_file()),
      BASE.check("component_builds_serially",
                 configure["returncode"] == 0 and build["returncode"] == 0 and
                 BASE.BINARY.is_file()),
      BASE.check("level_zero_only_runtime_boundary",
                 link_map["returncode"] == 0 and
                 "libze_loader" in lower_links and
                 "openvino" not in lower_links and
                 "libdnnl" not in lower_links),
      BASE.check("repeat_and_confirm_complete_without_oom",
                 all(
                     worker["returncode"] == 0 and
                     not worker["timed_out"] and
                     not worker["memory_guard"]["tripped"] and
                     not worker["oom_observed"]
                     for worker in (repeat, confirm)),
                 repeat={
                     key: repeat.get(key) for key in (
                         "returncode", "timed_out", "memory_guard",
                         "oom_observed", "monitor")
                 },
                 confirm={
                     key: confirm.get(key) for key in (
                         "returncode", "timed_out", "memory_guard",
                         "oom_observed", "monitor")
                 }),
      BASE.check("all_forced_fallback_outputs_are_bitwise_equal",
                 all(
                     audit["phase_count"] == len(phases) and
                     audit["worker_required_checks_passed"] is True and
                     audit["all_outputs_bitwise_equal"] is True and
                     audit["all_selected_ids_equal"] is True and
                     audit["all_finite"] is True and
                     audit["all_timings_positive"] is True and
                     audit["anchor_bitwise_matches"] ==
                         len(REFERENCE_PHASES)
                     for audit in (repeat_audit, confirm_audit)),
                 reference_phases=list(REFERENCE_PHASES)),
      BASE.check("parallel_block_topk_codegen_is_spill_free",
                 all(
                     audit["candidate_block_resources"].get(
                         "required_group_size_x") == 256 and
                     audit["candidate_block_resources"].get(
                         "required_subgroup_size") == 16 and
                     audit["candidate_block_resources"].get(
                         "spill_mem_bytes") == 0
                     for audit in (repeat_audit, confirm_audit)),
                 repeat=repeat_audit["candidate_block_resources"],
                 confirm=confirm_audit["candidate_block_resources"]),
      BASE.check("repeat_and_confirm_outputs_are_bitwise_deterministic",
                 output_hash_match),
      BASE.check("every_hidden_has_twenty_exact_abba_blocks",
                 all(
                     audit["schedule"] == "ABBA" and
                     audit["paired_blocks_per_hidden"] == args.blocks and
                     audit["block_arithmetic_exact"] and
                     audit["baseline"]["sample_count"] == expected_blocks and
                     audit["candidate"]["sample_count"] == expected_blocks and
                     audit["saving"]["sample_count"] == expected_blocks and
                     audit["performance_inference"]["hidden_count"] ==
                         len(phases)
                     for audit in (repeat_audit, confirm_audit)),
                 expected_blocks=expected_blocks),
      BASE.check("paired_one_sided_95pct_saving_lcb_clears_product_cut",
                 all(
                     audit["performance_inference"]["rate_pass"] is True
                     for audit in (repeat_audit, confirm_audit)),
                 required_saving_us=REQUIRED_SAVING_US,
                 repeat=repeat_audit["performance_inference"],
                 confirm=confirm_audit["performance_inference"]),
  ]
  passed = all(row["pass"] for row in checks)
  verdict = (
      "admit_exact_parallel_block_topk_for_product_integration"
      if passed else
      "reject_exact_parallel_block_topk_component")
  result = {
      "schema": SCHEMA,
      "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
      "workstream": WS,
      "git": state,
      "verdict": verdict,
      "required_checks_passed": passed,
      "checks": checks,
      "source_contract": contract,
      "repeat_audit": repeat_audit,
      "confirm_audit": confirm_audit,
      "repeat_worker": {
          key: value for key, value in repeat.items() if key != "result"
      },
      "confirm_worker": {
          key: value for key, value in confirm.items() if key != "result"
      },
      "build": {
          "module_compile": module_compile,
          "module_validate": module_validate,
          "configure": configure,
          "build": build,
          "link_map": link_map,
      },
      "inputs": {
          "baseline": BASE.display_path(args.baseline),
          "baseline_sha256": BASE.sha256(args.baseline),
          "capture_dir": BASE.display_path(args.capture_dir),
          "reference_dir": BASE.display_path(args.reference_dir),
          "phase_range": [args.first_phase, args.last_phase],
          "paired_blocks_per_hidden": args.blocks,
          "warmup": args.warmup,
          "sources": {
              BASE.display_path(path): BASE.sha256(path)
              for path in (
                  BASE.MODULE_SOURCE, BASE.CPP_SOURCE,
                  BASE.BOUNDARIES, BASE.PRODUCT_PATCH)
          },
          "module": BASE.display_path(module),
          "module_sha256": BASE.sha256(module) if module.is_file() else None,
      },
      "gpu_component_workers_launched": 2,
      "model_workers_launched": 0,
      "stock_workers_launched": 0,
      "workers_concurrent": False,
      "product_integration_ready": passed,
      "product_promotion_ready": False,
      "speedup_claims_allowed": False,
  }
  BASE.write_json(output / "result.json", result)
  BASE.write_json(output / "manifest.json", {
      "schema": SCHEMA,
      "git": state,
      "verdict": verdict,
      "required_checks_passed": passed,
      "tool": BASE.display_path(Path(__file__)),
      "phase_range": [args.first_phase, args.last_phase],
      "paired_blocks_per_hidden": args.blocks,
      "gpu_component_workers_launched": 2,
      "model_workers_launched": 0,
      "memory_preflight_gib": args.min_available_gib,
      "memory_abort_gib": args.abort_below_available_gib,
      "product_integration_ready": passed,
      "product_promotion_ready": False,
      "speedup_claims_allowed": False,
  })
  write_summary(output, result)
  print(json.dumps({
      "output": BASE.display_path(output),
      "verdict": verdict,
      "required_checks_passed": passed,
      "repeat_baseline_median_us": repeat_audit["baseline"]["median_us"],
      "repeat_candidate_median_us": repeat_audit["candidate"]["median_us"],
      "repeat_saving_lcb_us": repeat_audit[
          "performance_inference"]["lower_confidence_bound_saving_us"],
      "confirm_baseline_median_us": confirm_audit["baseline"]["median_us"],
      "confirm_candidate_median_us": confirm_audit["candidate"]["median_us"],
      "confirm_saving_lcb_us": confirm_audit[
          "performance_inference"]["lower_confidence_bound_saving_us"],
  }, sort_keys=True))
  return 0 if passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
