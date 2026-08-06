#!/usr/bin/env python3
"""Run the one admitted Q/gate split-consumer relocation component.

This launches one candidate-only 2k/17-step worker with the PR36362 plugin and
the exact graph relocation admitted by the source gate.  The prior seq1304 raw
worker is the control; no additional control, stock, long, ABBA, or product
worker is launched.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import statistics
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = (
    "intel-qwen36-openvino-dynamic-split-consumer-relocation-component-v0")
BASE_PATH = ROOT / (
    "tools/intel-qwen36-openvino-accepted-carrier-profile-refresh.py")
BOUND = ROOT / (
    "output/openvino-dynamic-split-consumer-relocation-bound-"
    "20260717Tseq1306-cleanZ/metrics.json")
BUILD = ROOT / (
    "output/openvino-dynamic-split-inplace-plugin-build-"
    "20260717Tseq1303c-cleanZ/manifest.json")
CONTROL = ROOT / (
    "output/openvino-dynamic-split-inplace-component-"
    "20260717Tseq1304-control-2k-warm17-cleanZ/raw/2k/candidate/"
    "worker-result.json")
PLUGIN = Path(
    "/home/intel/intel-qwen36-r0/output/"
    "openvino-90214e-l0-gpu-dynamic-split-candidate-seq1304/"
    "bin/intel64/Release/libopenvino_intel_gpu_plugin.so")
EXPECTED_CORE_COUNTS = {
    "Assign": 60,
    "FullyConnectedCompressed": 371,
    "GatedDeltaNet": 30,
    "IQ36HotAttentionGQA": 10,
    "IQ36LinearConvSwish": 30,
    "RMS": 131,
}


def load_module() -> Any:
  spec = importlib.util.spec_from_file_location("iq36_refresh", BASE_PATH)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load {BASE_PATH}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


BASE = load_module()


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--plugin", type=Path, default=PLUGIN)
  parser.add_argument("--timeout-s", type=int, default=1200)
  parser.add_argument("--poll-interval-s", type=float, default=1.0)
  parser.add_argument("--min-available-gib", type=float, default=8.0)
  parser.add_argument("--abort-below-available-gib", type=float, default=4.0)
  parser.add_argument("--igc-library-dir", type=Path, default=None,
                      help=argparse.SUPPRESS)
  parser.add_argument(
      "--existing-candidate", type=Path,
      help=("analyze an already completed component without launching another "
            "worker; used only to make a conclusive decision from retained "
            "raw evidence"))
  args = parser.parse_args()
  if args.timeout_s <= 0 or args.poll_interval_s <= 0.0:
    parser.error("timeout and poll interval must be positive")
  if args.abort_below_available_gib > args.min_available_gib:
    parser.error("abort threshold must not exceed preflight threshold")
  if args.igc_library_dir is not None:
    parser.error("this component does not admit an IGC delta")
  return args


def load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise TypeError(f"expected JSON object: {path}")
  return value


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    while chunk := stream.read(8 * 1024 * 1024):
      digest.update(chunk)
  return digest.hexdigest()


def display(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path.resolve())


def git_state(output: Path) -> dict[str, Any]:
  commit = subprocess.run(
      ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
      capture_output=True, text=True).stdout.strip()
  rows = subprocess.run(
      ["git", "status", "--porcelain"], cwd=ROOT, check=True,
      capture_output=True, text=True).stdout.splitlines()
  allowed = {
      "tools/intel-qwen36-openvino-dynamic-split-consumer-relocation-component.py"}
  output_relative = str(output.resolve().relative_to(ROOT))
  dirty = []
  for row in rows:
    path = row[3:]
    if path in allowed or path.startswith(output_relative):
      continue
    dirty.append(row)
  return {
      "commit": commit,
      "dirty": bool(dirty),
      "dirty_paths": dirty,
      "allowed_uncommitted_tool_paths": sorted(allowed),
  }


def check(name: str, passed: bool, **detail: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **detail}


def target_profile(result: dict[str, Any]) -> dict[str, Any]:
  rows = result.get("full_profile")
  if not isinstance(rows, list):
    phases = result.get("phases", [])
    rows = phases[-1].get("full_profile") if phases else None
  if not isinstance(rows, list):
    raise TypeError("candidate worker has no full profile")
  target = [
      row for row in rows
      if "self_attn/prim::ListUnpack/VariadicSplit.out" in
         str(row.get("node_name", ""))]
  status_counts = Counter(str(row.get("status")) for row in target)
  executed_type_counts = Counter(
      str(row.get("node_type")) for row in target
      if row.get("status") == "Status.EXECUTED")
  optimized_type_counts = Counter(
      str(row.get("node_type")) for row in target
      if row.get("status") == "Status.OPTIMIZED_OUT")
  return {
      "rows": len(target),
      "status_counts": dict(sorted(status_counts.items())),
      "executed_type_counts": dict(sorted(executed_type_counts.items())),
      "optimized_type_counts": dict(sorted(optimized_type_counts.items())),
      "executed_raw_real_time_us_nonadditive": sum(
          float(row.get("real_time_us", 0.0)) for row in target
          if row.get("status") == "Status.EXECUTED"),
      "rows_detail": target,
  }


def stable_walls(result: dict[str, Any]) -> list[float]:
  walls = [float(row["wall_ms_diagnostic"])
           for row in result.get("phases", [])[1:]]
  if len(walls) != 17 or not all(math.isfinite(value) and value > 0.0
                                 for value in walls):
    raise ValueError("worker does not have 17 finite decode walls")
  return walls[1:]


def main() -> int:
  args = parse_args()
  output = args.output.resolve()
  worker_dir = output / "raw/2k/candidate"
  if args.existing_candidate is None:
    worker_dir.mkdir(parents=True, exist_ok=False)
  else:
    output.mkdir(parents=True, exist_ok=False)
    args.existing_candidate = args.existing_candidate.resolve()
  required_base = (
      BASE_PATH, BOUND, BUILD, CONTROL, args.plugin, BASE.REFERENCE_WORKER,
      BASE.WORKER, BASE.GRAPH_MODULE if hasattr(BASE, "GRAPH_MODULE") else
          ROOT / "tools/intel_qwen36_openvino_hot_cold_attention.py")
  existing_inputs: tuple[Path, ...] = ()
  if args.existing_candidate is not None:
    existing_inputs = (
        args.existing_candidate / "metrics.json",
        args.existing_candidate / "raw/2k/candidate/worker-result.json")
  required = required_base + existing_inputs
  missing = [display(Path(path)) for path in required if not Path(path).is_file()]
  if missing:
    raise SystemExit("missing relocation-component inputs: " + ", ".join(missing))

  git = git_state(output)
  concurrent = BASE.other_worker_pids()
  if concurrent:
    raise RuntimeError(f"concurrent OpenVINO worker detected: {concurrent}")
  bound = load_json(BOUND)
  build = load_json(BUILD)
  control = load_json(CONTROL)
  reference = load_json(BASE.REFERENCE_WORKER)
  plugin = args.plugin.resolve()
  plugin_hash = sha256(plugin)
  config, decode_tokens, expected_top1 = BASE.build_worker_config(
      worker_dir, reference, plugin)
  config["relocate_dynamic_split_consumers"] = True
  if args.existing_candidate is None:
    worker = BASE.launch_worker(args, worker_dir, config)
    workers_launched = 1
  else:
    prior = load_json(args.existing_candidate / "metrics.json")
    prior_result = load_json(
        args.existing_candidate / "raw/2k/candidate/worker-result.json")
    worker = {**prior["worker"], "result": prior_result}
    workers_launched = 0
  result = worker["result"]
  phases = result.get("phases", [])
  actual_top1 = [int(row.get("top1", -1)) for row in phases]
  candidate_profile = BASE.profile_audit(result) if result else {}
  split_profile = target_profile(result) if result else {}
  control_split = target_profile(control)
  source = result.get("source_summary") or {}
  control_stable = stable_walls(control)
  candidate_stable = stable_walls(result) if result else []
  control_median = statistics.median(control_stable)
  candidate_median = (
      statistics.median(candidate_stable) if candidate_stable else math.nan)
  observed_saving = control_median - candidate_median
  required_saving = float(bound["budget"]["seq1302_shortfall_ms"])
  activation_passed = (
      split_profile.get("rows") == 20
      and split_profile.get("status_counts") == {"Status.OPTIMIZED_OUT": 20}
      and split_profile.get("executed_type_counts") == {}
      and split_profile.get("optimized_type_counts") ==
          {"Crop": 10, "VariadicSplit": 10})
  correctness_passed = (
      len(phases) == 18
      and actual_top1 == expected_top1
      and all(row.get("logits_finite") is True for row in phases))
  core_census_passed = (
      candidate_profile.get("selected_executed_counts") ==
          EXPECTED_CORE_COUNTS)
  performance_passed = (
      math.isfinite(observed_saving) and observed_saving >= required_saving)
  worker_safe = (
      worker["returncode"] == 0
      and worker["timed_out"] is False
      and worker["memory_guard"]["tripped"] is False
      and worker["oom_observed"] is False
      and int(worker["monitor"]["system_available_min_bytes"] or 0) >=
          int(args.abort_below_available_gib * 1024**3))

  checks = [
      check("repository_clean_at_gate", not git["dirty"], git=git),
      check("seq1306_admits_exactly_one_relocation_candidate",
            bound.get("required_checks_passed") is True
            and bound.get("pr36362_alone_closed") is True
            and bound.get("relocation_candidate_admitted") is True
            and bound.get("candidate_workers_admitted") == 1
            and bound.get("additional_control_worker_admitted") is False),
      check("no_concurrent_worker_at_launch", not concurrent,
            concurrent=concurrent),
      check("one_candidate_worker_completes_above_stop_without_oom",
            worker_safe, worker={key: worker[key] for key in (
                "returncode", "timed_out", "memory_guard", "monitor",
                "oom_observed")}),
      check("worker_uses_exact_pr36362_candidate_plugin",
            plugin_hash == build["candidate_plugin"]["sha256"]
            and result.get("candidate_gpu_plugin_sha256") == plugin_hash),
      check("graph_executes_exact_ten_consumer_relocations",
            source.get("relocate_dynamic_split_consumers") is True
            and source.get("split_consumer_relocation_count") == 10,
            source_summary=source),
      check("teacher_forced_top1_is_exact", correctness_passed,
            expected_top1=expected_top1, actual_top1=actual_top1),
      check("core_execution_census_is_unchanged", core_census_passed,
            core_counts=candidate_profile.get("selected_executed_counts")),
      check("target_split_activation_outcome_is_completely_measured",
            split_profile.get("rows") == 20
            and sum(split_profile.get("status_counts", {}).values()) == 20,
            activation_passed=activation_passed,
            control=control_split, candidate=split_profile),
      check("profile_times_are_not_added_as_savings",
            candidate_profile.get("raw_profile_time_is_savings_evidence")
                is False),
  ]
  evidence_checks_passed = all(row["pass"] for row in checks)
  route_accepted = (
      evidence_checks_passed and activation_passed and correctness_passed
      and performance_passed)
  verdict = (
      "accept_q_gate_split_consumer_relocation_component"
      if route_accepted else
      "reject_q_gate_split_consumer_relocation_after_component"
      if evidence_checks_passed else "inconclusive")
  performance = {
      "stable_sample_rule": "drop first decode JIT sample",
      "stable_samples_per_side": 16,
      "control_median_ms": control_median,
      "candidate_median_ms": candidate_median,
      "observed_median_saving_ms": observed_saving,
      "required_incremental_saving_ms": required_saving,
      "margin_to_required_ms": observed_saving - required_saving,
      "control_mean_ms": statistics.mean(control_stable),
      "candidate_mean_ms": (
          statistics.mean(candidate_stable) if candidate_stable else None),
      "component_performance_passed": performance_passed,
      "speed_claim": False,
  }
  metrics = {
      "schema": SCHEMA,
      "created_at": datetime.now(timezone.utc).isoformat(),
      "workstream": WS,
      "git": git,
      "verdict": verdict,
      "evidence_checks_passed": evidence_checks_passed,
      "route_accepted": route_accepted,
      "activation_passed": activation_passed,
      "correctness_passed": correctness_passed,
      "performance_passed": performance_passed,
      "gpu_workers_launched": workers_launched,
      "context_candidate_workers_launched": 1,
      "stock_workers_launched": 0,
      "control_workers_launched": 0,
      "long_workers_launched": 0,
      "product_workers_launched": 0,
      "decode_tokens": decode_tokens,
      "expected_top1": expected_top1,
      "actual_top1": actual_top1,
      "worker": {key: value for key, value in worker.items()
                 if key != "result"},
      "source_summary": source,
      "profile": {
          "core": candidate_profile,
          "target_split": split_profile,
          "control_target_split": control_split,
      },
      "performance": performance,
      "checks": checks,
      "decision": {
          "close_relocation_route": evidence_checks_passed and not route_accepted,
          "retain_as_bundle_ingredient": route_accepted,
          "next_route": (
              "openvino_rms_igc_split_relocation_bundle_bound"
              if route_accepted else "openvino_upstream_capability_watch"),
          "reopen_condition": (
              "none for the unchanged PR36362 plugin, consumer relocation, "
              "repeat, or sample extension if this component closes"),
      },
  }
  write_json(output / "metrics.json", metrics)
  write_json(output / "manifest.json", {
      "schema": SCHEMA,
      "tool": display(Path(__file__)),
      "git": git,
      "plugin": str(plugin),
      "plugin_sha256": plugin_hash,
      "inputs": {display(Path(path)): sha256(Path(path)) for path in required},
      "memory_preflight_gib": args.min_available_gib,
      "memory_abort_gib": args.abort_below_available_gib,
      "stock_workers": 0,
      "control_workers": 0,
      "candidate_workers": workers_launched,
      "context_candidate_workers": 1,
  })
  report = f"""# Q/gate split consumer-relocation component

Verdict: **{verdict}**. Evidence checks:
`{str(evidence_checks_passed).lower()}`; activation:
`{str(activation_passed).lower()}`; correctness:
`{str(correctness_passed).lower()}`; short performance:
`{str(performance_passed).lower()}`.

Seq1307's exactly one candidate-only 2k/17-step worker ran against the retained
seq1304 control. This decision pass launched `{workers_launched}` workers. The
candidate used the exact PR36362 plugin and ten graph consumer
relocations. Target profile status is `{split_profile.get('status_counts')}`
versus control `{control_split.get('status_counts')}`. All 18 teacher-forced
top-1 tokens and the 371 FC / 10 attention / 30 GDN / 30 linear / 60 Assign /
131 RMS core census must remain exact.

After dropping the first decode JIT sample, diagnostic medians are
`{control_median:.6f} -> {candidate_median:.6f} ms`, an observed
`{observed_saving:.7f}-ms` saving versus the incremental
`{required_saving:.7f}-ms` cut. This single short component is not a speedup or
product claim. No stock, control, concurrent, long, ABBA, output512, or product
worker ran; OOM observed: `{str(worker['oom_observed']).lower()}`.
"""
  (output / "report.md").write_text(report, encoding="utf-8")
  print(json.dumps({
      "artifact": display(output), "verdict": verdict,
      "activation_passed": activation_passed,
      "correctness_passed": correctness_passed,
      "performance_passed": performance_passed,
      "worker_returncode": worker["returncode"],
      "oom_observed": worker["oom_observed"],
  }, separators=(",", ":")), flush=True)
  return 0 if evidence_checks_passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
