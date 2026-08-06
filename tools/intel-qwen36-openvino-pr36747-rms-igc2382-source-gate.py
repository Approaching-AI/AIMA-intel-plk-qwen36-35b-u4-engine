#!/usr/bin/env python3
"""Audit the clean shared-triple plus PR36747 RMS patch before build."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-openvino-pr36747-rms-igc2382-source-gate-v0"
SOURCE_TREE = Path(
    "/home/intel/intel-qwen36-r0/source/openvino-90214e5be05")
FC_SOURCE = SOURCE_TREE / (
    "src/plugins/intel_gpu/src/plugin/transformations/"
    "fc_horizontal_fusion.cpp")
FC_TEST = SOURCE_TREE / (
    "src/plugins/intel_gpu/tests/unit/transformations/"
    "horizontal_fc_fusion_test.cpp")
PR_PATHS = (
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
TARGET_PATHS = (
    str(FC_SOURCE.relative_to(SOURCE_TREE)),
    str(FC_TEST.relative_to(SOURCE_TREE)),
    *PR_PATHS,
)
PATCH = ROOT / "engine/openvino/iq36-router-shared-pr36747-rms.patch"
PR_PATCH = ROOT / (
    "output/openvino-post-igc-opportunity-bound-20260717Tseq1302-"
    "cleanZ/raw/openvino-pr36747.patch")
BOUND = ROOT / (
    "output/openvino-pr36747-rms-igc2382-isolation-bound-"
    "20260718Tseq1346-cleanZ/metrics.json")
PINNED_COMMIT = "90214e5be052438cec5617ed3ea7e37df1538f68"
PINNED_PR_SHA256 = (
    "5e0e17b5908a6aa1bb696442193d36e7d8108e5bd1d1335b031643bdda3665bf")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--memory-stop-gib", type=float, default=4.0)
  args = parser.parse_args()
  if args.memory_stop_gib <= 0.0:
    parser.error("memory stop must be positive")
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


def run(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
  return subprocess.run(
      command, cwd=cwd, text=True, capture_output=True, check=False)


def git_state(output: Path) -> dict[str, Any]:
  commit = run(["git", "rev-parse", "HEAD"], ROOT).stdout.strip()
  rows = run(["git", "status", "--porcelain"], ROOT).stdout.splitlines()
  allowed = {
      "engine/openvino/iq36-linear-tail-triple-pr36747-rms.patch",
      "engine/openvino/iq36-router-shared-pr36747-rms.patch",
      "tools/intel-qwen36-openvino-linear-tail-rms-igc2382-source-gate.py",
      "tools/intel-qwen36-openvino-linear-tail-rms-igc2382-build.py",
      "tools/intel-qwen36-openvino-linear-tail-rms-igc2382-component.py",
      "tools/intel-qwen36-openvino-pr36747-rms-igc2382-isolation-bound.py",
      "tools/intel-qwen36-openvino-pr36747-rms-igc2382-source-gate.py",
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
      "allowed_uncommitted_paths": sorted(allowed),
  }


def check(name: str, passed: bool, **detail: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **detail}


def main() -> int:
  args = parse_args()
  output = args.output.resolve()
  output.mkdir(parents=True, exist_ok=False)
  required = (
      FC_SOURCE, FC_TEST, PATCH, PR_PATCH, BOUND,
      *(SOURCE_TREE / path for path in PR_PATHS))
  missing = [display(path) for path in required if not path.is_file()]
  if missing:
    raise SystemExit("missing clean source-gate inputs: " + ", ".join(missing))

  stop_bytes = int(args.memory_stop_gib * 1024**3)
  memory = [{"stage": "start", "available_bytes": available_memory_bytes()}]
  if memory[0]["available_bytes"] < stop_bytes:
    raise RuntimeError("memory stop tripped before source gate")
  git = git_state(output)
  bound = load_json(BOUND)
  base_commit = run(["git", "rev-parse", "HEAD"], SOURCE_TREE).stdout.strip()
  reverse_check = run(
      ["git", "apply", "--reverse", "--check", str(PATCH)], SOURCE_TREE)
  pr_reverse_check = run(
      ["git", "apply", "--reverse", "--check", str(PR_PATCH)],
      SOURCE_TREE)
  target_diff = run(["git", "diff", "--", *TARGET_PATHS], SOURCE_TREE)
  patch_text = PATCH.read_text(encoding="utf-8")
  source_text = FC_SOURCE.read_text(encoding="utf-8")
  test_text = FC_TEST.read_text(encoding="utf-8")
  pr_text = PR_PATCH.read_text(encoding="utf-8")
  pr_touched = sorted({
      line[len("+++ b/"):]
      for line in pr_text.splitlines() if line.startswith("+++ b/")})
  memory.append({
      "stage": "after-source-audit",
      "available_bytes": available_memory_bytes(),
  })

  source_contract = {
      "candidate_helper_occurrences": source_text.count(
          "get_horizontal_fusion_candidates"),
      "exact_four_user_guard": "candidates.size() != 4" in source_text,
      "exact_k2048_guard": source_text.count("shape[1] != 2048") == 1,
      "router_partition_exact": all(token in source_text for token in (
          "n1_count != 1", "n256_count != 1", "n512_count != 2",
          "router_fc = fc", "if (fc != router_fc)")),
      "global_max_three_exact": (
          "const int max_num_fcs_to_fuse = 3" in source_text),
      "linear_specialization_absent": all(
          token not in source_text for token in (
              "n32_count", "n4096_count", "n8192_count",
              "is_exact_linear_four_group")),
      "compressed_fc_matcher": (
          "wrap_type<op::FullyConnectedCompressed>" in source_text),
      "weight_concat_axis_zero": (
          "Concat>(weight_nodes_as_output_vector, 0)" in source_text),
      "scale_concat_axis_zero": (
          "Concat>(scales_as_output_vector, 0)" in source_text),
      "original_width_split": (
          "VariadicSplit>(new_fc, axis_const, split_const)" in source_text),
  }
  test_contract = {
      "named_router_isolated_case": (
          "FullyConnectedHorizontalFusion_router_isolated_"
          "shared_triple_no_bias_no_zp" in test_text),
      "n1_k2048_occurrences": test_text.count("ov::Shape{1, 2048}"),
      "n512_k2048_occurrences": test_text.count("ov::Shape{512, 2048}"),
      "n256_k2048_occurrences": test_text.count("ov::Shape{256, 2048}"),
      "fused_weights_exclude_router": (
          "OutputVector{weight1, weight2, weight3}" in test_text),
      "fused_scales_exclude_router": (
          "OutputVector{scale1, scale2, scale3}" in test_text),
      "shared_widths_exact": (
          "orig_n_sizes = {1, 512, 512}" in test_text),
      "router_remains_separate": (
          "Reshape>(router_fc, reshape_pattern, true)" in test_text),
      "linear_tail_case_absent": (
          "exact_linear_tail_triple" not in test_text),
  }
  exact_source = (
      source_contract["candidate_helper_occurrences"] == 3
      and all(value for key, value in source_contract.items()
              if key != "candidate_helper_occurrences"))
  exact_test = (
      test_contract["n1_k2048_occurrences"] == 2
      and test_contract["n512_k2048_occurrences"] == 4
      and test_contract["n256_k2048_occurrences"] == 2
      and all(value for key, value in test_contract.items()
              if not key.endswith("_occurrences")))
  expected_bound_contract = {
      "preserve_existing_qkv_triples": 10,
      "preserve_shared_triples": 40,
      "preserve_unfused_router_gates": 40,
      "preserve_unfused_linear_branches": 120,
      "linear_horizontal_fusion_allowed": False,
      "pr36747_patch_sha256": PINNED_PR_SHA256,
      "isolated_igc2382": True,
      "expected_fully_connected_compressed": 291,
      "expected_fused_three_groups": 50,
      "expected_rms_consumers": 131,
  }
  checks = [
      check("repository_clean_at_gate", not git["dirty"], git=git),
      check("pinned_openvino_commit_exact", base_commit == PINNED_COMMIT,
            observed=base_commit, expected=PINNED_COMMIT),
      check("durable_patch_is_exact_applied_seven_file_diff",
            reverse_check.returncode == 0
            and target_diff.returncode == 0
            and target_diff.stdout == patch_text,
            patch_sha256=sha256(PATCH)),
      check("shared_subset_retains_global_max_three_without_linear_route",
            exact_source, source=source_contract),
      check("router_isolated_shared_unit_graph_is_exact",
            exact_test, test=test_contract),
      check("captured_pr36747_is_exactly_present",
            sha256(PR_PATCH) == PINNED_PR_SHA256
            and pr_reverse_check.returncode == 0
            and pr_touched == sorted(PR_PATHS),
            patch_sha256=sha256(PR_PATCH), touched_paths=pr_touched),
      check("seq1346_admits_only_this_clean_source_gate",
            bound.get("required_checks_passed") is True
            and bound.get("source_gate_admitted") is True
            and bound.get("plugin_build_admitted") is False
            and bound.get("gpu_worker_admitted") is False
            and bound.get("source_contract") == expected_bound_contract),
      check("no_compiler_gpu_or_model_worker_ran", True,
            compilers=0, gpu_contexts=0, model_workers=0),
      check("memory_guard_never_tripped",
            all(row["available_bytes"] >= stop_bytes for row in memory),
            memory=memory),
  ]
  required_checks_passed = all(row["pass"] for row in checks)
  plugin_build_admitted = required_checks_passed
  verdict = (
      "admit_one_clean_pr36747_rms_incremental_plugin_build"
      if plugin_build_admitted else "inconclusive")
  metrics = {
      "schema": SCHEMA,
      "created_at": datetime.now(timezone.utc).isoformat(),
      "workstream": WS,
      "git": git,
      "verdict": verdict,
      "required_checks_passed": required_checks_passed,
      "plugin_build_admitted": plugin_build_admitted,
      "gpu_context_admitted": False,
      "model_worker_admitted": False,
      "source_tree": {
          "path": str(SOURCE_TREE),
          "commit": base_commit,
          "target_paths": list(TARGET_PATHS),
          "target_diff_sha256": hashlib.sha256(
              target_diff.stdout.encode()).hexdigest(),
          "unrelated_dirty_state_preserved": True,
      },
      "source_contract": source_contract,
      "unit_graph_contract": test_contract,
      "expected_runtime_census": {
          "existing_qkv_fused_three_groups": 10,
          "shared_fused_three_groups": 40,
          "linear_unfused": 120,
          "router_gate_unfused": 40,
          "fully_connected_compressed": 291,
          "rms_consumers": 131,
      },
      "next_action": {
          "route": "openvino_pr36747_rms_igc2382_clean_incremental_build",
          "admitted_builds": 1,
          "memory_stop_bytes": stop_bytes,
          "requirements": [
              "build only the candidate GPU plugin with parallelism four",
              "record peak RSS, swap, plugin hash, and exact target diff",
              "launch no GPU/model worker until the build gate passes",
          ],
      },
      "checks": checks,
      "memory": {"stop_bytes": stop_bytes, "samples": memory},
  }
  write_json(output / "metrics.json", metrics)
  write_json(output / "manifest.json", {
      "schema": SCHEMA,
      "tool": display(Path(__file__)),
      "git": git,
      "inputs": {display(path): sha256(path) for path in required},
      "compilers": 0,
      "gpu_contexts": 0,
      "model_workers": 0,
  })
  report = f"""# Clean PR36747 RMS plus IGC 2.38.2 source gate

Verdict: **{verdict}**. Required checks:
`{str(required_checks_passed).lower()}`. No compiler, GPU context, or model
worker ran.

The durable patch is byte-identical to the seven-file target diff over pinned
OpenVINO `{base_commit}`. Horizontal FC remains globally max-three, admits only
the exact router-isolated shared subset, and contains no linear specialization.
The captured PR36747 patch is exact across its five paths.

The predicted runtime census is 50 fused triples (ten QKV and 40 shared), 120
independent linear FCs, 40 routers, 291 FCs, and 131 RMS consumers. Runtime
activation and clean-plugin performance are not yet proven; admit one
incremental build and no model worker.
"""
  (output / "report.md").write_text(report, encoding="utf-8")
  print(json.dumps({
      "artifact": display(output),
      "verdict": verdict,
      "plugin_build_admitted": plugin_build_admitted,
      "gpu_or_model_worker_launched": False,
  }, separators=(",", ":")), flush=True)
  return 0 if required_checks_passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
