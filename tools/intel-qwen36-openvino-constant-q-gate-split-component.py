#!/usr/bin/env python3
"""Run the one admitted constant Q/gate split-length component.

This launches one candidate-only 2k/17-step worker with the retained pinned
control plugin, ten exact [256, 256] split-length constants, and the ten exact
consumer relocations.  Seq1304 is the retained control; no additional control,
stock, long, ABBA, or product worker is launched.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-openvino-constant-q-gate-split-component-v0"
HELPER_PATH = ROOT / (
    "tools/intel-qwen36-openvino-dynamic-split-consumer-relocation-"
    "component.py")
BOUND = ROOT / (
    "output/openvino-constant-q-gate-split-bound-"
    "20260717Tseq1309b-cleanZ/metrics.json")
BUILD = ROOT / (
    "output/openvino-dynamic-split-inplace-plugin-build-"
    "20260717Tseq1303c-cleanZ/manifest.json")
CONTROL = ROOT / (
    "output/openvino-dynamic-split-inplace-component-"
    "20260717Tseq1304-control-2k-warm17-cleanZ/raw/2k/candidate/"
    "worker-result.json")
PLUGIN = Path(
    "/home/intel/intel-qwen36-r0/output/"
    "openvino-90214e-l0-gpu-dynamic-split-control-seq1304/"
    "bin/intel64/Release/libopenvino_intel_gpu_plugin.so")
EXPECTED_CORE_COUNTS = {
    "Assign": 60,
    "FullyConnectedCompressed": 371,
    "GatedDeltaNet": 30,
    "IQ36HotAttentionGQA": 10,
    "IQ36LinearConvSwish": 30,
    "RMS": 131,
}


def load_module(path: Path, name: str) -> Any:
  spec = importlib.util.spec_from_file_location(name, path)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load {path}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


HELPER = load_module(HELPER_PATH, "iq36_split_relocation_component")
BASE = HELPER.BASE


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


def check(name: str, passed: bool, **detail: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **detail}


def main() -> int:
  args = parse_args()
  output = args.output.resolve()
  worker_dir = output / "raw/2k/candidate"
  worker_dir.mkdir(parents=True, exist_ok=False)
  required = (
      HELPER_PATH, BOUND, BUILD, CONTROL, args.plugin,
      BASE.REFERENCE_WORKER, BASE.WORKER,
      BASE.GRAPH_MODULE if hasattr(BASE, "GRAPH_MODULE") else
          ROOT / "tools/intel_qwen36_openvino_hot_cold_attention.py")
  missing = [display(Path(path)) for path in required if not Path(path).is_file()]
  if missing:
    raise SystemExit("missing constant-split component inputs: " +
                     ", ".join(missing))

  git = HELPER.git_state(output)
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
  config["constant_q_gate_split_lengths"] = True
  worker = BASE.launch_worker(args, worker_dir, config)
  result = worker["result"]
  phases = result.get("phases", [])
  actual_top1 = [int(row.get("top1", -1)) for row in phases]
  candidate_profile = BASE.profile_audit(result) if result else {}
  split_profile = HELPER.target_profile(result) if result else {}
  control_split = HELPER.target_profile(control)
  source = result.get("source_summary") or {}
  control_stable = HELPER.stable_walls(control)
  candidate_stable = HELPER.stable_walls(result) if result else []
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
      check("seq1309b_admits_exactly_one_constant_split_candidate",
            bound.get("required_checks_passed") is True
            and bound.get("constant_split_candidate_admitted") is True
            and bound.get("candidate_workers_admitted") == 1
            and bound.get("additional_control_worker_admitted") is False
            and bound.get("compiler_build_admitted") is False),
      check("no_concurrent_worker_at_launch", not concurrent,
            concurrent=concurrent),
      check("one_candidate_worker_completes_above_stop_without_oom",
            worker_safe, worker={key: worker[key] for key in (
                "returncode", "timed_out", "memory_guard", "monitor",
                "oom_observed")}),
      check("worker_uses_exact_retained_control_plugin",
            plugin_hash == build["control_plugin"]["sha256"]
            and result.get("candidate_gpu_plugin_sha256") == plugin_hash),
      check("graph_executes_exact_ten_folds_and_relocations",
            source.get("relocate_dynamic_split_consumers") is True
            and source.get("split_consumer_relocation_count") == 10
            and source.get("constant_q_gate_split_lengths") is True
            and source.get("split_length_fold_count") == 10,
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
      "retain_constant_q_gate_split_as_bundle_cut"
      if route_accepted else
      "reject_constant_q_gate_split_after_component"
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
      "gpu_workers_launched": 1,
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
          "close_constant_split_route": (
              evidence_checks_passed and not route_accepted),
          "retain_as_bundle_ingredient": route_accepted,
          "next_route": (
              "openvino_rms_igc_constant_split_bundle_bound"
              if route_accepted else "openvino_upstream_capability_watch"),
          "reopen_condition": (
              "none for the unchanged constant split and relocation graph, "
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
      "candidate_workers": 1,
  })
  report = f"""# Constant Q/gate split-length component

Verdict: **{verdict}**. Evidence checks:
`{str(evidence_checks_passed).lower()}`; activation:
`{str(activation_passed).lower()}`; correctness:
`{str(correctness_passed).lower()}`; short performance:
`{str(performance_passed).lower()}`.

Exactly one candidate-only 2k/17-step worker ran against retained seq1304.
The candidate used the exact pinned control plugin, ten I32 `[256, 256]`
split-length constants, and ten consumer relocations. Target profile status is
`{split_profile.get('status_counts')}` versus control
`{control_split.get('status_counts')}`. All 18 teacher-forced top-1 tokens and
the 371 FC / 10 attention / 30 GDN / 30 linear / 60 Assign / 131 RMS core
census must remain exact.

After dropping the first decode JIT sample, diagnostic medians are
`{control_median:.6f} -> {candidate_median:.6f} ms`, an observed
`{observed_saving:.7f}-ms` saving versus the incremental
`{required_saving:.7f}-ms` cut. This single short component is not a speedup or
product claim. No compiler, stock, control, concurrent, long, ABBA, output512,
or product worker ran; OOM observed: `{str(worker['oom_observed']).lower()}`.
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
