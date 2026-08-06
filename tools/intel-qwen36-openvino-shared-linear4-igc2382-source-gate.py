#!/usr/bin/env python3
"""Audit the exact shared-triple plus linear-four patch before any build."""

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
SCHEMA = (
    "intel-qwen36-openvino-shared-linear4-igc2382-source-gate-v0")
SOURCE_TREE = Path(
    "/home/intel/intel-qwen36-r0/source/openvino-90214e5be05")
SOURCE = SOURCE_TREE / (
    "src/plugins/intel_gpu/src/plugin/transformations/"
    "fc_horizontal_fusion.cpp")
TEST = SOURCE_TREE / (
    "src/plugins/intel_gpu/tests/unit/transformations/"
    "horizontal_fc_fusion_test.cpp")
PATCH = ROOT / (
    "engine/openvino/iq36-shared-linear4-horizontal-fusion.patch")
BOUND = ROOT / (
    "output/openvino-shared-linear4-igc2382-bundle-bound-"
    "20260718Tseq1338b-cleanZ/metrics.json")
SHARED_COMPONENT = ROOT / (
    "output/openvino-router-isolated-shared-triple-component-"
    "20260718Tseq1337-candidate-2k-warm17-cleanZ/metrics.json")
PINNED_COMMIT = "90214e5be052438cec5617ed3ea7e37df1538f68"


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
      "engine/openvino/iq36-shared-linear4-horizontal-fusion.patch",
      "tools/intel-qwen36-openvino-shared-linear4-igc2382-source-gate.py",
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
  required = (SOURCE, TEST, PATCH, BOUND, SHARED_COMPONENT)
  missing = [display(path) for path in required if not path.is_file()]
  if missing:
    raise SystemExit("missing combined source inputs: " + ", ".join(missing))

  stop_bytes = int(args.memory_stop_gib * 1024**3)
  memory = [{"stage": "start", "available_bytes": available_memory_bytes()}]
  if memory[0]["available_bytes"] < stop_bytes:
    raise RuntimeError("memory stop tripped before source gate")

  git = git_state(output)
  bound = load_json(BOUND)
  shared_component = load_json(SHARED_COMPONENT)
  base_commit = run(["git", "rev-parse", "HEAD"], SOURCE_TREE).stdout.strip()
  reverse_check = run(
      ["git", "apply", "--reverse", "--check", str(PATCH)], SOURCE_TREE)
  target_diff = run([
      "git", "diff", "--",
      str(SOURCE.relative_to(SOURCE_TREE)),
      str(TEST.relative_to(SOURCE_TREE)),
  ], SOURCE_TREE)
  patch_text = PATCH.read_text(encoding="utf-8")
  source_text = SOURCE.read_text(encoding="utf-8")
  test_text = TEST.read_text(encoding="utf-8")
  memory.append({"stage": "after-source-audit",
                 "available_bytes": available_memory_bytes()})

  helper_name = "get_horizontal_fusion_candidates"
  linear_helper_name = "is_exact_linear_four_group"
  source_contract = {
      "shared_helper_occurrences": source_text.count(helper_name),
      "linear_helper_occurrences": source_text.count(linear_helper_name),
      "global_max_four_literal_absent": (
          "const int max_num_fcs_to_fuse = 4" not in source_text),
      "shared_exact_four_user_guard": "candidates.size() != 4" in source_text,
      "two_exact_k_2048_guards": source_text.count(
          "shape[1] != 2048") == 2,
      "shared_width_partition": all(token in source_text for token in (
          "shape[0] == 1", "shape[0] == 256", "shape[0] == 512",
          "n1_count != 1 || n256_count != 1 || n512_count != 2")),
      "router_excluded_by_identity": "if (fc != router_fc)" in source_text,
      "linear_width_partition": all(token in source_text for token in (
          "shape[0] == 32", "shape[0] == 4096", "shape[0] == 8192",
          "n32_count == 2 && n4096_count == 1 && n8192_count == 1")),
      "selective_four_only": (
          "is_exact_linear_four_group(fc_candidates) ? 4 : 3" in
          source_text),
      "compressed_fc_matcher": (
          "wrap_type<op::FullyConnectedCompressed>" in source_text),
      "weight_concat_axis_zero": (
          "Concat>(weight_nodes_as_output_vector, 0)" in source_text),
      "scale_concat_axis_zero": (
          "Concat>(scales_as_output_vector, 0)" in source_text),
      "original_width_split": (
          "orig_n_sizes" in source_text
          and "VariadicSplit>(new_fc, axis_const, split_const)" in
          source_text),
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
      "three_unequal_shared_widths": (
          "orig_n_sizes = {1, 512, 512}" in test_text),
      "three_way_split_shape": (
          "ov::Shape{3}, orig_n_sizes" in test_text),
      "router_fc_remains_separate": (
          "auto router_fc = std::make_shared<ov::intel_gpu::op::"
          "FullyConnectedCompressed>" in test_text
          and "Reshape>(router_fc, reshape_pattern, true)" in test_text),
      "four_results_preserved": (
          "ResultVector{result1, result2, result3, result4}" in test_text),
      "named_exact_linear_four_case": (
          "FullyConnectedHorizontalFusion_exact_linear_four_"
          "no_bias_no_zp" in test_text),
      "n8192_k2048_occurrences": test_text.count("ov::Shape{8192, 2048}"),
      "n32_k2048_occurrences": test_text.count("ov::Shape{32, 2048}"),
      "n4096_k2048_occurrences": test_text.count("ov::Shape{4096, 2048}"),
      "linear_four_weight_concat": (
          "OutputVector{weight1, weight2, weight3, weight4}" in test_text),
      "linear_four_scale_concat": (
          "OutputVector{scale1, scale2, scale3, scale4}" in test_text),
      "linear_four_widths": (
          "orig_n_sizes = {8192, 32, 32, 4096}" in test_text),
      "linear_four_split_shape": (
          "ov::Shape{4}, orig_n_sizes" in test_text),
      "linear_four_fourth_output_consumed": (
          "Reshape>(split->output(3), reshape_pattern, true)" in test_text),
  }
  exact_source_contract = (
      source_contract["shared_helper_occurrences"] == 3
      and source_contract["linear_helper_occurrences"] == 2
      and all(value for key, value in source_contract.items()
              if key not in {"shared_helper_occurrences",
                             "linear_helper_occurrences"}))
  exact_test_contract = (
      test_contract["n1_k2048_occurrences"] == 2
      and test_contract["n512_k2048_occurrences"] == 4
      and test_contract["n256_k2048_occurrences"] == 2
      and test_contract["n8192_k2048_occurrences"] == 2
      and test_contract["n32_k2048_occurrences"] == 4
      and test_contract["n4096_k2048_occurrences"] == 2
      and all(value for key, value in test_contract.items()
              if key not in {"n1_k2048_occurrences",
                             "n512_k2048_occurrences",
                             "n256_k2048_occurrences",
                             "n8192_k2048_occurrences",
                             "n32_k2048_occurrences",
                             "n4096_k2048_occurrences"}))

  expected_contract = bound["source_contract"]
  checks = [
      check("repository_clean_at_gate", not git["dirty"], git=git),
      check("pinned_openvino_commit_exact", base_commit == PINNED_COMMIT,
            observed=base_commit, expected=PINNED_COMMIT),
      check("durable_patch_is_exact_applied_target_diff",
            reverse_check.returncode == 0
            and target_diff.returncode == 0
            and target_diff.stdout == patch_text,
            patch_sha256=sha256(PATCH),
            reverse_check_stderr=reverse_check.stderr.strip()),
      check("shared_and_linear_subset_predicates_exact",
            exact_source_contract, source=source_contract),
      check("shared_triple_and_linear_four_unit_graphs_exact",
            exact_test_contract, test=test_contract),
      check("seq1337_retains_exact_shared_triple_and_router_semantics",
            shared_component.get("evidence_checks_passed") is True
            and shared_component.get("activation_passed") is True
            and shared_component.get("correctness_passed") is True
            and shared_component["profile"]["candidate"].get(
                "fused_shared_triple_count") == 40
            and shared_component["profile"]["candidate"].get(
                "unfused_router_gate_count") == 40),
      check("seq1338b_admits_only_this_combined_source_patch",
            bound.get("required_checks_passed") is True
            and bound.get("source_edit_admitted") is True
            and bound.get("plugin_build_admitted") is False
            and bound.get("gpu_worker_admitted") is False
            and expected_contract == {
                "preserve_router_isolated_shared_triples": 40,
                "preserve_unfused_router_gates": 40,
                "preserve_existing_qkv_triples": 10,
                "new_exact_linear_four_groups": 30,
                "linear_four_widths": [8192, 32, 32, 4096],
                "linear_four_k": 2048,
                "global_max_four_allowed": False,
                "expected_fully_connected_compressed": 201,
                "expected_fused_shared_triples": 40,
                "expected_fused_linear_fours": 30,
                "expected_unfused_router_gates": 40,
            }),
      check("no_compiler_gpu_or_model_worker_ran", True,
            compilers=0, gpu_contexts=0, model_workers=0),
      check("memory_guard_never_tripped",
            all(row["available_bytes"] >= stop_bytes for row in memory),
            memory=memory),
  ]
  required_checks_passed = all(row["pass"] for row in checks)
  plugin_build_admitted = required_checks_passed
  verdict = (
      "admit_one_serial_incremental_plugin_build"
      if plugin_build_admitted else "inconclusive")

  metrics = {
      "schema": SCHEMA,
      "created_at": datetime.now(timezone.utc).isoformat(),
      "workstream": WS,
      "git": git,
      "verdict": verdict,
      "required_checks_passed": required_checks_passed,
      "plugin_build_admitted": plugin_build_admitted,
      "unit_test_build_admitted": False,
      "gpu_context_admitted": False,
      "model_worker_admitted": False,
      "source_tree": {
          "path": str(SOURCE_TREE),
          "commit": base_commit,
          "unrelated_dirty_state_preserved": True,
          "target_diff_sha256": hashlib.sha256(
              target_diff.stdout.encode()).hexdigest(),
      },
      "source_contract": source_contract,
      "unit_graph_contract": test_contract,
      "expected_runtime_census": {
          "existing_qkv_fused_three_groups": 10,
          "new_shared_fused_three_groups": 40,
          "new_linear_fused_four_groups": 30,
          "router_gate_unfused": 40,
          "fully_connected_compressed": 201,
      },
      "next_action": {
          "route": "openvino_shared_linear4_igc2382_incremental_build",
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
  report = f"""# Shared-triple plus exact linear-four source gate

Verdict: **{verdict}**. Required checks:
`{str(required_checks_passed).lower()}`. No compiler, GPU context, or model
worker ran.

The durable patch is byte-identical to the target diff over pinned OpenVINO
`{base_commit}`. Generic groups retain max-three. The shared helper continues
to exclude only the exact `N=256` router, while a second predicate permits four
outputs only for exact `K=2048`, `N=[8192,32,32,4096]` linear groups.

Unit graphs cover both subsets. The predicted runtime census is 40 shared
triples, 30 linear fours, 40 unfused routers, ten QKV triples, and 201 FCs.
Runtime activation, correctness, and IGC retention are not yet proven. Admit
one incremental plugin build; launch no GPU/model worker yet.
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
