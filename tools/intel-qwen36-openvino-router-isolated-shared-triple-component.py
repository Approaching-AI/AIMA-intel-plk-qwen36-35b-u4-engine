#!/usr/bin/env python3
"""Run one guarded router-isolated shared-triple plus Q/K worker."""

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
SCHEMA = "intel-qwen36-openvino-router-isolated-shared-triple-component-v0"
QK_COMPONENT_PATH = ROOT / (
    "tools/intel-qwen36-openvino-qk-rope-layout-component.py")
BOUND = ROOT / (
    "output/openvino-router-isolated-shared-triple-bound-"
    "20260718Tseq1334-cleanZ/metrics.json")
BUDGET_BOUND = ROOT / (
    "output/openvino-fc-rms-igc-qk-rope-bundle-bound-"
    "20260718Tseq1328-cleanZ/metrics.json")
SOURCE_GATE = ROOT / (
    "output/openvino-router-isolated-shared-triple-source-gate-"
    "20260718Tseq1335-cleanZ/metrics.json")
BUILD = ROOT / (
    "output/openvino-router-isolated-shared-triple-build-"
    "20260718Tseq1336-cleanZ/metrics.json")
CONTROL = ROOT / (
    "output/openvino-dynamic-split-inplace-component-"
    "20260717Tseq1304-control-2k-warm17-cleanZ/raw/2k/candidate/"
    "worker-result.json")
QK_CANDIDATE = ROOT / (
    "output/openvino-qk-rope-layout-component-"
    "20260717Tseq1327-corrected-candidate-2k-warm17-cleanZ/"
    "raw/2k/candidate/worker-result.json")
PATCH = ROOT / "engine/openvino/iq36-router-isolated-shared-triple-fusion.patch"
SOURCE_TREE = Path(
    "/home/intel/intel-qwen36-r0/source/openvino-90214e5be05")
PLUGIN = Path(
    "/home/intel/intel-qwen36-r0/output/openvino-90214e-l0-gpu/"
    "bin/intel64/Release/libopenvino_intel_gpu_plugin.so")
TARGET_SOURCE = (
    "src/plugins/intel_gpu/src/plugin/transformations/"
    "fc_horizontal_fusion.cpp")
TARGET_TEST = (
    "src/plugins/intel_gpu/tests/unit/transformations/"
    "horizontal_fc_fusion_test.cpp")
EXPECTED_PLUGIN_SHA256 = (
    "117b574db32fd55edf88cb765842cedcffbecab367f15824dd91ced5d40aa8ad")
EXPECTED_CORE_COUNTS = {
    "Assign": 60,
    "FullyConnectedCompressed": 291,
    "GatedDeltaNet": 30,
    "IQ36HotAttentionGQA": 10,
    "IQ36LinearConvSwish": 30,
    "IQ36QKRopeLayout": 10,
    "RMS": 131,
}
LINEAR_SUFFIXES = (
    "linear_attn.in_proj_qkv/ov_ext::linear/MatMul",
    "linear_attn.in_proj_a/ov_ext::linear/MatMul",
    "linear_attn.in_proj_b/ov_ext::linear/MatMul",
    "linear_attn.in_proj_z/ov_ext::linear/MatMul",
)
SHARED_SUFFIXES = (
    "mlp.shared_expert_gate/ov_ext::linear/MatMul",
    "mlp.shared_expert.gate_proj/ov_ext::linear/MatMul",
    "mlp.shared_expert.up_proj/ov_ext::linear/MatMul",
)
ROUTER_GATE_SUFFIX = "mlp.gate/aten::linear/MatMul"


def load_module(path: Path, name: str) -> Any:
  spec = importlib.util.spec_from_file_location(name, path)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load {path}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


QK = load_module(QK_COMPONENT_PATH, "iq36_qk_component_for_shared_triple")
BASE = QK.BASE


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


def available_memory_bytes() -> int:
  for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
    if line.startswith("MemAvailable:"):
      return int(line.split()[1]) * 1024
  raise RuntimeError("MemAvailable missing from /proc/meminfo")


def git_state(output: Path) -> dict[str, Any]:
  commit = subprocess.run(
      ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
      capture_output=True, text=True).stdout.strip()
  rows = subprocess.run(
      ["git", "status", "--porcelain"], cwd=ROOT, check=True,
      capture_output=True, text=True).stdout.splitlines()
  allowed = {
      "engine/openvino/iq36-router-isolated-shared-triple-fusion.patch",
      "tools/intel-qwen36-openvino-router-isolated-shared-triple-source-gate.py",
      "tools/intel-qwen36-openvino-router-isolated-shared-triple-build.py",
      "tools/intel-qwen36-openvino-router-isolated-shared-triple-component.py",
  }
  relative_output = str(output.resolve().relative_to(ROOT))
  dirty = []
  for row in rows:
    path = row[3:]
    if path in allowed or path.startswith(relative_output):
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


def runtime_audit(result: dict[str, Any]) -> dict[str, Any]:
  rows = QK.profile_rows(result)
  executed = [row for row in rows
              if row.get("status") == "Status.EXECUTED"]
  counts = Counter(str(row.get("node_type")) for row in executed)
  fused4 = sorted(
      str(row.get("node_name")) for row in executed
      if row.get("node_type") == "FullyConnectedCompressed"
      and "_fused_4FCs" in str(row.get("node_name")))
  fused3 = sorted(
      str(row.get("node_name")) for row in executed
      if row.get("node_type") == "FullyConnectedCompressed"
      and "_fused_3FCs" in str(row.get("node_name")))
  fused3_shared = [name for name in fused3
                   if ".mlp.shared_expert" in name]
  fused3_qkv = [name for name in fused3
                if ".mlp.shared_expert" not in name]
  linear_originals = []
  shared_originals = []
  router_originals = []
  for row in executed:
    if row.get("node_type") != "FullyConnectedCompressed":
      continue
    name = str(row.get("node_name"))
    if "_fused_" in name:
      continue
    if any(name.endswith(suffix) for suffix in LINEAR_SUFFIXES):
      linear_originals.append(name)
    elif any(name.endswith(suffix) for suffix in SHARED_SUFFIXES):
      shared_originals.append(name)
    elif name.endswith(ROUTER_GATE_SUFFIX):
      router_originals.append(name)
  core_counts = {key: int(counts.get(key, 0))
                 for key in EXPECTED_CORE_COUNTS}
  qk = QK.runtime_audit(result)
  return {
      "executed_counts": dict(sorted(counts.items())),
      "core_counts": core_counts,
      "core_counts_exact": core_counts == EXPECTED_CORE_COUNTS,
      "fused_four_fc_count": len(fused4),
      "fused_four_fc_names": fused4,
      "fused_three_fc_count": len(fused3),
      "fused_shared_triple_count": len(fused3_shared),
      "fused_shared_triple_names": fused3_shared,
      "existing_fused_qkv_count": len(fused3_qkv),
      "existing_fused_qkv_names": fused3_qkv,
      "unfused_linear_original_count": len(linear_originals),
      "unfused_linear_original_names": sorted(linear_originals),
      "unfused_shared_original_count": len(shared_originals),
      "unfused_shared_original_names": sorted(shared_originals),
      "unfused_router_gate_count": len(router_originals),
      "unfused_router_gate_names": sorted(router_originals),
      "variadic_split_executed": int(counts.get("VariadicSplit", 0)),
      "crop_executed": int(counts.get("Crop", 0)),
      "old_qk_boundary_executed": qk["old_boundary_executed"],
      "qk_rope_layout_executed": qk["qk_rope_layout_executed"],
      "raw_profile_time_is_savings_evidence": False,
  }


def main() -> int:
  args = parse_args()
  output = args.output.resolve()
  worker_dir = output / "raw/2k/candidate"
  worker_dir.mkdir(parents=True, exist_ok=False)
  required = (
      BOUND, BUDGET_BOUND, SOURCE_GATE, BUILD, CONTROL, QK_CANDIDATE, PATCH,
      args.plugin, BASE.REFERENCE_WORKER, BASE.WORKER,
      QK.GRAPH_SOURCE, QK.WORKER_SOURCE, QK.KERNEL_SOURCE,
      QK.CUSTOM_CONFIG)
  missing = [display(Path(path)) for path in required
             if not Path(path).is_file()]
  if missing:
    raise SystemExit("missing shared-triple component inputs: " + ", ".join(missing))
  git = git_state(output)
  concurrent = BASE.other_worker_pids()
  if concurrent:
    raise RuntimeError(f"concurrent OpenVINO worker detected: {concurrent}")
  if available_memory_bytes() < int(args.min_available_gib * 1024**3):
    raise RuntimeError("preflight memory is below the serial worker minimum")

  bound = load_json(BOUND)
  budget_bound = load_json(BUDGET_BOUND)
  source_gate = load_json(SOURCE_GATE)
  build = load_json(BUILD)
  control = load_json(CONTROL)
  qk_candidate = load_json(QK_CANDIDATE)
  reference = load_json(BASE.REFERENCE_WORKER)
  plugin = args.plugin.resolve()
  plugin_hash = sha256(plugin)
  target_diff = subprocess.run([
      "git", "diff", "--", TARGET_SOURCE, TARGET_TEST,
  ], cwd=SOURCE_TREE, check=True, capture_output=True, text=True).stdout
  patch_text = PATCH.read_text(encoding="utf-8")

  preflight_checks = [
      check("repository_clean_at_gate", not git["dirty"], git=git),
      check("seq1334_admits_only_router_isolated_source",
            bound.get("required_checks_passed") is True
            and bound.get("source_edit_admitted") is True
            and bound.get("plugin_build_admitted") is False
            and bound.get("gpu_worker_admitted") is False),
      check("seq1335_admits_exactly_one_serial_build",
            source_gate.get("required_checks_passed") is True
            and source_gate.get("plugin_build_admitted") is True
            and source_gate.get("model_worker_admitted") is False),
      check("seq1336_build_identity_is_exact_and_safe",
            build.get("required_checks_passed") is True
            and build.get("candidate_plugin_retained") is True
            and build.get("build", {}).get("oom_observed") is False
            and build.get("build", {}).get("memory_guard_tripped") is False
            and build.get("candidate_plugin_after", {}).get("sha256")
                == EXPECTED_PLUGIN_SHA256),
      check("candidate_plugin_and_applied_patch_are_exact",
            plugin_hash == EXPECTED_PLUGIN_SHA256
            and target_diff == patch_text,
            plugin_sha256=plugin_hash, patch_sha256=sha256(PATCH)),
      check("activation_expectation_is_exact_371_to_291",
            bound.get("source_contract") == {
                "candidate_predicate": (
                    "when a shared input has four compressed FC users with "
                    "K=2048, exclude the unique N=256 branch from the matcher "
                    "and callback; retain max three and fuse N=[1,512,512] only"),
                "current_all_group_patch_is_rejected_input": True,
                "expected_existing_qkv_fused_three_groups": 10,
                "expected_fully_connected_compressed": 291,
                "expected_new_shared_fused_three_groups": 40,
                "expected_router_gate_unfused": 40,
                "stock_max_three": True}),
      check("no_concurrent_worker_at_launch", not concurrent,
            concurrent=concurrent),
  ]
  if not all(row["pass"] for row in preflight_checks):
    raise RuntimeError("shared-triple activation preflight did not pass")

  config, decode_tokens, expected_top1 = BASE.build_worker_config(
      worker_dir, reference, plugin)
  config["fuse_qk_rope_layout"] = True
  worker = BASE.launch_worker(args, worker_dir, config)
  result = worker["result"]
  phases = result.get("phases", [])
  actual_top1 = [int(row.get("top1", -1)) for row in phases]
  candidate_profile = runtime_audit(result) if result else {}
  control_profile = runtime_audit(control)
  qk_profile = runtime_audit(qk_candidate)
  source_summary = result.get("source_summary") or {}

  control_stable = QK.stable_walls(control)
  qk_stable = QK.stable_walls(qk_candidate)
  candidate_stable = QK.stable_walls(result) if result else []
  control_median = statistics.median(control_stable)
  qk_median = statistics.median(qk_stable)
  candidate_median = (
      statistics.median(candidate_stable) if candidate_stable else math.nan)
  total_saving_ms = control_median - candidate_median
  incremental_fc_saving_ms = qk_median - candidate_median
  required_total_saving_ms = float(
      budget_bound["budget"]["current_kill_number_ms"])
  required_incremental_fc_saving_ms = (
      required_total_saving_ms
      - float(budget_bound["budget"][
          "seq1327_qk_observed_component_point_ms"]))

  activation_passed = (
      candidate_profile.get("core_counts_exact") is True
      and candidate_profile.get("fused_four_fc_count") == 0
      and candidate_profile.get("fused_three_fc_count") == 50
      and candidate_profile.get("fused_shared_triple_count") == 40
      and candidate_profile.get("existing_fused_qkv_count") == 10
      and candidate_profile.get("unfused_shared_original_count") == 0
      and candidate_profile.get("unfused_router_gate_count") == 40
      and candidate_profile.get("unfused_linear_original_count") == 120
      and candidate_profile.get("old_qk_boundary_executed") == 0
      and candidate_profile.get("qk_rope_layout_executed") == 10)
  correctness_passed = (
      len(phases) == 18
      and actual_top1 == expected_top1
      and all(row.get("logits_finite") is True for row in phases))
  performance_passed = (
      math.isfinite(total_saving_ms)
      and total_saving_ms >= required_total_saving_ms
      and incremental_fc_saving_ms >= required_incremental_fc_saving_ms)
  worker_safe = (
      worker["returncode"] == 0 and worker["timed_out"] is False
      and worker["memory_guard"]["tripped"] is False
      and worker["oom_observed"] is False
      and int(worker["monitor"]["system_available_min_bytes"] or 0) >=
          int(args.abort_below_available_gib * 1024**3))
  evidence_checks = [
      *preflight_checks,
      check("one_candidate_worker_completes_above_stop_without_oom",
            worker_safe, worker={key: worker[key] for key in (
                "returncode", "timed_out", "memory_guard", "monitor",
                "oom_observed")}),
      check("worker_uses_exact_router_isolated_candidate_plugin",
            result.get("candidate_gpu_plugin_sha256") == plugin_hash,
            plugin_sha256=plugin_hash),
      check("graph_retains_exact_ten_qk_producers",
            source_summary.get("fuse_qk_rope_layout") is True
            and source_summary.get("qk_rope_layout_rewrite_count") == 10
            and source_summary.get("custom_count_after") == 10,
            source_summary=source_summary),
      check("all_40_shared_triples_activate_and_router_branches_remain",
            activation_passed, candidate_profile=candidate_profile),
      check("retained_control_and_qk_censuses_are_exact",
            control_profile.get("core_counts", {}).get(
                "FullyConnectedCompressed") == 371
            and qk_profile.get("core_counts", {}).get(
                "FullyConnectedCompressed") == 371
            and qk_profile.get("qk_rope_layout_executed") == 10),
      check("profile_times_are_not_added_as_savings",
            candidate_profile.get("raw_profile_time_is_savings_evidence")
                is False),
  ]
  evidence_checks_passed = all(row["pass"] for row in evidence_checks)
  route_accepted = (
      evidence_checks_passed and correctness_passed and performance_passed)
  verdict = (
      "retain_router_isolated_shared_triple_qk_component"
      if route_accepted else
      "reject_router_isolated_shared_triple_after_component"
      if evidence_checks_passed else "inconclusive")
  performance = {
      "stable_sample_rule": "drop first decode JIT sample",
      "stable_samples_per_artifact": 16,
      "control_median_ms": control_median,
      "retained_qk_median_ms": qk_median,
      "candidate_median_ms": candidate_median,
      "total_observed_saving_ms": total_saving_ms,
      "required_total_saving_ms": required_total_saving_ms,
      "total_margin_ms": total_saving_ms - required_total_saving_ms,
      "incremental_fc_observed_saving_ms": incremental_fc_saving_ms,
      "required_incremental_fc_saving_ms": (
          required_incremental_fc_saving_ms),
      "incremental_fc_margin_ms": (
          incremental_fc_saving_ms - required_incremental_fc_saving_ms),
      "control_mean_ms": statistics.mean(control_stable),
      "retained_qk_mean_ms": statistics.mean(qk_stable),
      "candidate_mean_ms": (
          statistics.mean(candidate_stable) if candidate_stable else None),
      "component_performance_passed": performance_passed,
      "cross_artifact_screen_only": True,
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
      "control_workers_launched": 0,
      "stock_workers_launched": 0,
      "long_workers_launched": 0,
      "product_workers_launched": 0,
      "decode_tokens": decode_tokens,
      "expected_top1": expected_top1,
      "actual_top1": actual_top1,
      "worker": {key: value for key, value in worker.items()
                 if key != "result"},
      "source_summary": source_summary,
      "profile": {
          "candidate": candidate_profile,
          "control": control_profile,
          "retained_qk": qk_profile,
      },
      "performance": performance,
      "evidence_checks": evidence_checks,
      "decision": {
          "close_unchanged_shared_triple_route": (
              evidence_checks_passed and not route_accepted),
          "retain_as_bundle_candidate": route_accepted,
          "next_route": (
              "openvino_router_isolated_shared_triple_qk_32k_preflight"
              if route_accepted else
              "openvino_rms_qk_or_upstream_capability_bound"),
          "reopen_condition": (
              "none for an unchanged router-isolated subset, repeat, or sample "
              "extension if this component closes"),
      },
  }
  write_json(output / "metrics.json", metrics)
  write_json(output / "manifest.json", {
      "schema": SCHEMA,
      "tool": display(Path(__file__)),
      "git": git,
      "plugin": str(plugin),
      "plugin_sha256": plugin_hash,
      "inputs": {display(Path(path)): sha256(Path(path))
                 for path in required},
      "memory_preflight_gib": args.min_available_gib,
      "memory_abort_gib": args.abort_below_available_gib,
      "candidate_workers": 1,
      "control_workers": 0,
      "stock_workers": 0,
  })
  report = f"""# Router-isolated shared-expert triple plus Q/K component

Verdict: **{verdict}**. Evidence checks:
`{str(evidence_checks_passed).lower()}`; activation:
`{str(activation_passed).lower()}`; correctness:
`{str(correctness_passed).lower()}`; short screen:
`{str(performance_passed).lower()}`.

Exactly one candidate-only 2k/17-step worker ran. Activation requires 40 new
shared-expert `_fused_3FCs`, 40 unfused router gates, 120 untouched linear FCs,
ten existing QKV triples, a total FC census of 291, and all ten retained Q/K
producers. No `_fused_4FCs` may execute.

All 18 teacher-forced top-1 IDs must remain exact. After dropping the first
decode JIT sample, the retained control, Q/K-only, and combined medians are
`{control_median:.6f}`, `{qk_median:.6f}`, and `{candidate_median:.6f} ms`.
The combined cross-artifact screen saves `{total_saving_ms:.6f} ms` versus the
`{required_total_saving_ms:.6f}-ms` kill-number; the shared-triple increment saves
`{incremental_fc_saving_ms:.6f} ms` versus its
`{required_incremental_fc_saving_ms:.6f}-ms` residual.

This is component evidence, not paired product inference or a speed claim. No
new control, stock, long, ABBA, output512, or product worker ran. OOM observed:
`{str(worker['oom_observed']).lower()}`.
"""
  (output / "report.md").write_text(report, encoding="utf-8")
  print(json.dumps({
      "artifact": display(output),
      "verdict": verdict,
      "activation_passed": activation_passed,
      "correctness_passed": correctness_passed,
      "performance_passed": performance_passed,
      "candidate_fc_count": candidate_profile.get(
          "core_counts", {}).get("FullyConnectedCompressed"),
      "fused_shared_triple_count": candidate_profile.get(
          "fused_shared_triple_count"),
      "unfused_router_gate_count": candidate_profile.get(
          "unfused_router_gate_count"),
      "candidate_median_ms": candidate_median,
      "total_saving_ms": total_saving_ms,
      "worker_returncode": worker["returncode"],
      "oom_observed": worker["oom_observed"],
  }, separators=(",", ":")), flush=True)
  return 0 if evidence_checks_passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
