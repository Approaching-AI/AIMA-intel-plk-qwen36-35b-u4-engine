#!/usr/bin/env python3
"""Run one guarded linear-tail-triple + PR36747 RMS + IGC 2.38.2 worker."""

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
SCHEMA = "intel-qwen36-openvino-linear-tail-rms-igc2382-component-v0"
QK_COMPONENT_PATH = ROOT / (
    "tools/intel-qwen36-openvino-qk-rope-layout-component.py")
BOUND = ROOT / (
    "output/openvino-linear-tail-rms-igc-bundle-bound-"
    "20260718Tseq1342-cleanZ/metrics.json")
SOURCE_GATE = ROOT / (
    "output/openvino-linear-tail-rms-igc2382-source-gate-"
    "20260718Tseq1343d-cleanZ/metrics.json")
BUILD = ROOT / (
    "output/openvino-linear-tail-rms-igc2382-build-"
    "20260718Tseq1344b-cleanZ/metrics.json")
CONTROL = ROOT / (
    "output/openvino-dynamic-split-inplace-component-"
    "20260717Tseq1304-control-2k-warm17-cleanZ/raw/2k/candidate/"
    "worker-result.json")
QK_CANDIDATE = ROOT / (
    "output/openvino-qk-rope-layout-component-"
    "20260717Tseq1327-corrected-candidate-2k-warm17-cleanZ/"
    "raw/2k/candidate/worker-result.json")
SHARED_CANDIDATE = ROOT / (
    "output/openvino-router-isolated-shared-triple-component-"
    "20260718Tseq1337-candidate-2k-warm17-cleanZ/"
    "raw/2k/candidate/worker-result.json")
PATCH = ROOT / "engine/openvino/iq36-linear-tail-triple-pr36747-rms.patch"
SOURCE_TREE = Path(
    "/home/intel/intel-qwen36-r0/source/openvino-90214e5be05")
PLUGIN = Path(
    "/home/intel/intel-qwen36-r0/output/openvino-90214e-l0-gpu/"
    "bin/intel64/Release/libopenvino_intel_gpu_plugin.so")
TARGET_PATHS = (
    "src/plugins/intel_gpu/src/plugin/transformations/"
    "fc_horizontal_fusion.cpp",
    "src/plugins/intel_gpu/tests/unit/transformations/"
    "horizontal_fc_fusion_test.cpp",
    "src/plugins/intel_gpu/src/kernel_selector/cl_kernels/"
    "mvn_gpu_bfyx_opt.cl",
    "src/plugins/intel_gpu/src/kernel_selector/cl_kernels/"
    "rms_gpu_bfyx_opt.cl",
    "src/plugins/intel_gpu/src/kernel_selector/kernels/mvn/"
    "mvn_kernel_bfyx_opt.cpp",
    "src/plugins/intel_gpu/src/kernel_selector/kernels/rms/"
    "rms_kernel_bfyx_opt.cpp",
    "src/plugins/intel_gpu/tests/unit/test_cases/mvn_gpu_test.cpp",
)
EXPECTED_PLUGIN_SHA256 = (
    "ac6b908b0cc5a8444672d94ccdaa9ecd08e2f2d3825a55d9b44e009c00d8fce1")
IGC_LIBRARY_DIR = Path("/tmp/iq36-igc-2.38.2-root/usr/local/lib")
EXPECTED_IGC_LIBRARIES = {
    "libigc.so.2":
        "ff0cc269af1b2f843521b9207c54370fddab25caa404b1322cbdb4598452da33",
    "libigdfcl.so.2":
        "edd0cc3c73fee76ce156b8a8281d5a747f2634bc81a95da0ca1af9e72abd8de2",
    "libopencl-clang2.so.17":
        "5ad86d1aa4c4b92ca5ff96cbe2ca96d888b5afc5517e3c23b1772983c4dec63b",
}
EXPECTED_CORE_COUNTS = {
    "Assign": 60,
    "FullyConnectedCompressed": 231,
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


QK = load_module(QK_COMPONENT_PATH, "iq36_qk_for_linear_tail_rms_igc")
BASE = QK.BASE


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--plugin", type=Path, default=PLUGIN)
  parser.add_argument("--timeout-s", type=int, default=1200)
  parser.add_argument("--poll-interval-s", type=float, default=1.0)
  parser.add_argument("--min-available-gib", type=float, default=8.0)
  parser.add_argument("--abort-below-available-gib", type=float, default=4.0)
  parser.add_argument("--igc-library-dir", type=Path, default=IGC_LIBRARY_DIR,
                      help=argparse.SUPPRESS)
  args = parser.parse_args()
  if args.timeout_s <= 0 or args.poll_interval_s <= 0.0:
    parser.error("timeout and poll interval must be positive")
  if args.abort_below_available_gib > args.min_available_gib:
    parser.error("abort threshold must not exceed preflight threshold")
  if args.igc_library_dir.resolve() != IGC_LIBRARY_DIR.resolve():
    parser.error("only the exact isolated IGC 2.38.2 directory is admitted")
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
      "engine/openvino/iq36-linear-tail-triple-pr36747-rms.patch",
      "tools/intel-qwen36-openvino-linear-tail-rms-igc2382-source-gate.py",
      "tools/intel-qwen36-openvino-linear-tail-rms-igc2382-build.py",
      "tools/intel-qwen36-openvino-linear-tail-rms-igc2382-component.py",
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
  fused3_linear = [name for name in fused3 if ".linear_attn." in name]
  fused3_qkv = [name for name in fused3
                if ".mlp.shared_expert" not in name
                and ".linear_attn." not in name]
  linear_originals = []
  linear_original_suffixes: Counter[str] = Counter()
  shared_originals = []
  router_originals = []
  for row in executed:
    if row.get("node_type") != "FullyConnectedCompressed":
      continue
    name = str(row.get("node_name"))
    if "_fused_" in name:
      continue
    suffix = next((item for item in LINEAR_SUFFIXES
                   if name.endswith(item)), None)
    if suffix is not None:
      linear_originals.append(name)
      linear_original_suffixes[suffix] += 1
    elif any(name.endswith(item) for item in SHARED_SUFFIXES):
      shared_originals.append(name)
    elif name.endswith(ROUTER_GATE_SUFFIX):
      router_originals.append(name)
  rms_exec_types = Counter(
      str(row.get("exec_type")) for row in executed
      if row.get("node_type") == "RMS")
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
      "fused_linear_tail_triple_count": len(fused3_linear),
      "fused_linear_tail_triple_names": fused3_linear,
      "existing_fused_qkv_count": len(fused3_qkv),
      "existing_fused_qkv_names": fused3_qkv,
      "unfused_linear_original_count": len(linear_originals),
      "unfused_linear_original_suffix_counts": dict(
          sorted(linear_original_suffixes.items())),
      "unfused_linear_original_names": sorted(linear_originals),
      "unfused_shared_original_count": len(shared_originals),
      "unfused_shared_original_names": sorted(shared_originals),
      "unfused_router_gate_count": len(router_originals),
      "unfused_router_gate_names": sorted(router_originals),
      "rms_executed_count": int(counts.get("RMS", 0)),
      "rms_exec_types": dict(sorted(rms_exec_types.items())),
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
  igc_paths = tuple(args.igc_library_dir / name
                    for name in EXPECTED_IGC_LIBRARIES)
  required = (
      BOUND, SOURCE_GATE, BUILD, CONTROL, QK_CANDIDATE, SHARED_CANDIDATE,
      PATCH, *igc_paths, args.plugin, BASE.REFERENCE_WORKER, BASE.WORKER,
      QK.GRAPH_SOURCE, QK.WORKER_SOURCE, QK.KERNEL_SOURCE, QK.CUSTOM_CONFIG)
  missing = [display(Path(path)) for path in required
             if not Path(path).is_file()]
  if missing:
    raise SystemExit("missing linear-tail RMS component inputs: "
                     + ", ".join(missing))
  git = git_state(output)
  concurrent = BASE.other_worker_pids()
  if concurrent:
    raise RuntimeError(f"concurrent OpenVINO worker detected: {concurrent}")
  if available_memory_bytes() < int(args.min_available_gib * 1024**3):
    raise RuntimeError("preflight memory is below the serial worker minimum")

  bound = load_json(BOUND)
  source_gate = load_json(SOURCE_GATE)
  build = load_json(BUILD)
  control = load_json(CONTROL)
  qk_candidate = load_json(QK_CANDIDATE)
  shared_candidate = load_json(SHARED_CANDIDATE)
  reference = load_json(BASE.REFERENCE_WORKER)
  plugin = args.plugin.resolve()
  plugin_hash = sha256(plugin)
  target_diff = subprocess.run(
      ["git", "diff", "--", *TARGET_PATHS], cwd=SOURCE_TREE, check=True,
      capture_output=True, text=True).stdout
  patch_text = PATCH.read_text(encoding="utf-8")
  observed_igc_hashes = {path.name: sha256(path) for path in igc_paths}
  expected_contract = {
      "expected_fully_connected_compressed": 231,
      "expected_fused_three_groups": 80,
      "expected_unfused_linear_qkv": 30,
      "global_max_fcs_to_fuse": 3,
      "isolated_igc2382": True,
      "linear_branch_left_unfused": 8192,
      "linear_k": 2048,
      "linear_tail_widths": [32, 32, 4096],
      "new_linear_tail_triples": 30,
      "pr36747_patch_sha256": (
          "5e0e17b5908a6aa1bb696442193d36e7d8108e5bd1d1335b031643bdda3665bf"),
      "preserve_existing_qkv_triples": 10,
      "preserve_shared_triples": 40,
      "preserve_unfused_router_gates": 40,
  }
  preflight_checks = [
      check("repository_clean_at_gate", not git["dirty"], git=git),
      check("seq1342_admits_only_linear_tail_rms_source_audit",
            bound.get("required_checks_passed") is True
            and bound.get("source_audit_admitted") is True
            and bound.get("source_edit_admitted") is False
            and bound.get("plugin_build_admitted") is False
            and bound.get("gpu_worker_admitted") is False),
      check("seq1343d_admits_exactly_one_serial_build",
            source_gate.get("required_checks_passed") is True
            and source_gate.get("plugin_build_admitted") is True
            and source_gate.get("model_worker_admitted") is False),
      check("seq1344b_build_identity_is_exact_and_safe",
            build.get("required_checks_passed") is True
            and build.get("candidate_plugin_retained") is True
            and build.get("build", {}).get("oom_observed") is False
            and build.get("build", {}).get("memory_guard_tripped") is False
            and build.get("candidate_plugin_after", {}).get("sha256")
                == EXPECTED_PLUGIN_SHA256),
      check("candidate_plugin_and_seven_file_patch_are_exact",
            plugin_hash == EXPECTED_PLUGIN_SHA256
            and target_diff == patch_text,
            plugin_sha256=plugin_hash, patch_sha256=sha256(PATCH)),
      check("activation_expectation_is_exact_371_to_231",
            bound.get("source_contract") == expected_contract),
      check("isolated_igc2382_libraries_are_exact",
            observed_igc_hashes == EXPECTED_IGC_LIBRARIES,
            observed=observed_igc_hashes,
            expected=EXPECTED_IGC_LIBRARIES),
      check("no_concurrent_worker_at_launch", not concurrent,
            concurrent=concurrent),
  ]
  if not all(row["pass"] for row in preflight_checks):
    raise RuntimeError("linear-tail RMS IGC activation preflight did not pass")

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
  shared_profile = runtime_audit(shared_candidate)
  source_summary = result.get("source_summary") or {}

  control_stable = QK.stable_walls(control)
  qk_stable = QK.stable_walls(qk_candidate)
  shared_stable = QK.stable_walls(shared_candidate)
  candidate_stable = QK.stable_walls(result) if result else []
  control_median = statistics.median(control_stable)
  qk_median = statistics.median(qk_stable)
  shared_median = statistics.median(shared_stable)
  candidate_median = (
      statistics.median(candidate_stable) if candidate_stable else math.nan)
  total_saving_ms = control_median - candidate_median
  incremental_bundle_saving_ms = shared_median - candidate_median
  required_total_saving_ms = float(bound["budget"]["kill_number_ms"])
  required_incremental_bundle_saving_ms = (
      required_total_saving_ms
      - float(bound["budget"]["measured_qk_shared_ms"]))

  activation_passed = (
      candidate_profile.get("core_counts_exact") is True
      and candidate_profile.get("fused_four_fc_count") == 0
      and candidate_profile.get("fused_three_fc_count") == 80
      and candidate_profile.get("fused_shared_triple_count") == 40
      and candidate_profile.get("fused_linear_tail_triple_count") == 30
      and candidate_profile.get("existing_fused_qkv_count") == 10
      and candidate_profile.get("unfused_shared_original_count") == 0
      and candidate_profile.get("unfused_router_gate_count") == 40
      and candidate_profile.get("unfused_linear_original_count") == 30
      and candidate_profile.get("unfused_linear_original_suffix_counts") == {
          LINEAR_SUFFIXES[0]: 30}
      and candidate_profile.get("rms_executed_count") == 131
      and candidate_profile.get("rms_exec_types") == {
          "rms_gpu_bfyx_opt__f16": 131}
      and candidate_profile.get("old_qk_boundary_executed") == 0
      and candidate_profile.get("qk_rope_layout_executed") == 10)
  correctness_passed = (
      len(phases) == 18
      and actual_top1 == expected_top1
      and all(row.get("logits_finite") is True for row in phases))
  performance_passed = (
      math.isfinite(total_saving_ms)
      and total_saving_ms >= required_total_saving_ms
      and incremental_bundle_saving_ms >=
          required_incremental_bundle_saving_ms)
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
      check("worker_uses_exact_candidate_plugin_and_igc2382",
            result.get("candidate_gpu_plugin_sha256") == plugin_hash
            and worker.get("igc_library_dir") == str(IGC_LIBRARY_DIR)
            and worker.get("ld_library_path_first") == str(IGC_LIBRARY_DIR),
            plugin_sha256=plugin_hash,
            worker_igc_library_dir=worker.get("igc_library_dir")),
      check("graph_retains_exact_ten_qk_producers",
            source_summary.get("fuse_qk_rope_layout") is True
            and source_summary.get("qk_rope_layout_rewrite_count") == 10
            and source_summary.get("custom_count_after") == 10,
            source_summary=source_summary),
      check("linear_tail_shared_qkv_and_rms_activation_is_exact",
            activation_passed, candidate_profile=candidate_profile),
      check("retained_control_qk_and_shared_censuses_are_exact",
            control_profile.get("core_counts", {}).get(
                "FullyConnectedCompressed") == 371
            and qk_profile.get("core_counts", {}).get(
                "FullyConnectedCompressed") == 371
            and qk_profile.get("qk_rope_layout_executed") == 10
            and shared_profile.get("core_counts", {}).get(
                "FullyConnectedCompressed") == 291
            and shared_profile.get("fused_shared_triple_count") == 40),
      check("profile_times_are_not_added_as_savings",
            candidate_profile.get("raw_profile_time_is_savings_evidence")
                is False),
  ]
  evidence_checks_passed = all(row["pass"] for row in evidence_checks)
  route_accepted = (
      evidence_checks_passed and correctness_passed and performance_passed)
  verdict = (
      "retain_linear_tail_rms_igc2382_component"
      if route_accepted else
      "reject_linear_tail_rms_igc2382_after_component"
      if evidence_checks_passed else "inconclusive")
  performance = {
      "stable_sample_rule": "drop first decode JIT sample",
      "stable_samples_per_artifact": 16,
      "control_median_ms": control_median,
      "retained_qk_median_ms": qk_median,
      "retained_qk_shared_median_ms": shared_median,
      "candidate_median_ms": candidate_median,
      "total_observed_saving_ms": total_saving_ms,
      "required_total_saving_ms": required_total_saving_ms,
      "total_margin_ms": total_saving_ms - required_total_saving_ms,
      "incremental_tail_rms_igc_observed_saving_ms": (
          incremental_bundle_saving_ms),
      "required_incremental_tail_rms_igc_saving_ms": (
          required_incremental_bundle_saving_ms),
      "incremental_tail_rms_igc_margin_ms": (
          incremental_bundle_saving_ms
          - required_incremental_bundle_saving_ms),
      "control_mean_ms": statistics.mean(control_stable),
      "retained_qk_mean_ms": statistics.mean(qk_stable),
      "retained_qk_shared_mean_ms": statistics.mean(shared_stable),
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
          "retained_qk_shared": shared_profile,
      },
      "performance": performance,
      "evidence_checks": evidence_checks,
      "decision": {
          "close_unchanged_linear_tail_rms_igc_route": (
              evidence_checks_passed and not route_accepted),
          "retain_as_bundle_candidate": route_accepted,
          "next_route": (
              "openvino_linear_tail_rms_igc2382_32k_preflight"
              if route_accepted else "openvino_upstream_capability_bound"),
          "reopen_condition": (
              "none for an unchanged tail-RMS-IGC bundle, repeat, or sample "
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
      "igc_library_dir": str(IGC_LIBRARY_DIR),
      "igc_libraries": observed_igc_hashes,
      "candidate_workers": 1,
      "control_workers": 0,
      "stock_workers": 0,
  })
  report = f"""# Linear-tail triple plus PR36747 RMS and IGC 2.38.2 component

Verdict: **{verdict}**. Evidence checks:
`{str(evidence_checks_passed).lower()}`; activation:
`{str(activation_passed).lower()}`; correctness:
`{str(correctness_passed).lower()}`; short screen:
`{str(performance_passed).lower()}`.

Exactly one candidate-only 2k/17-step worker ran under isolated IGC 2.38.2.
Activation requires 40 shared, 30 linear-tail, and ten QKV `_fused_3FCs`, no
four-way fusion, 30 independent linear `N=8192` branches, 40 routers, 231 FCs,
131 optimized RMS executions, and all ten retained Q/K producers.

All 18 teacher-forced top-1 IDs must remain exact. After dropping the first
decode JIT sample, control, Q/K-only, Q/K+shared, and candidate medians are
`{control_median:.6f}`, `{qk_median:.6f}`, `{shared_median:.6f}`, and
`{candidate_median:.6f} ms`. The candidate saves `{total_saving_ms:.6f} ms`
versus the `{required_total_saving_ms:.6f}-ms` kill-number; its incremental
bundle saves `{incremental_bundle_saving_ms:.6f} ms` versus the
`{required_incremental_bundle_saving_ms:.6f}-ms` residual.

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
      "fused_linear_tail_triple_count": candidate_profile.get(
          "fused_linear_tail_triple_count"),
      "unfused_linear_count": candidate_profile.get(
          "unfused_linear_original_count"),
      "candidate_median_ms": candidate_median,
      "total_saving_ms": total_saving_ms,
      "worker_returncode": worker["returncode"],
      "oom_observed": worker["oom_observed"],
  }, separators=(",", ":")), flush=True)
  return 0 if evidence_checks_passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
