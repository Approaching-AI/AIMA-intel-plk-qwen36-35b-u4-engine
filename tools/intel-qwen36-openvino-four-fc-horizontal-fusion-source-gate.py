#!/usr/bin/env python3
"""Verify the four-way compressed-FC patch before any plugin build."""

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
SCHEMA = "intel-qwen36-openvino-four-fc-horizontal-fusion-source-gate-v0"
SOURCE_TREE = Path(
    "/home/intel/intel-qwen36-r0/source/openvino-90214e5be05")
SOURCE = SOURCE_TREE / (
    "src/plugins/intel_gpu/src/plugin/transformations/"
    "fc_horizontal_fusion.cpp")
TEST = SOURCE_TREE / (
    "src/plugins/intel_gpu/tests/unit/transformations/"
    "horizontal_fc_fusion_test.cpp")
PATCH = ROOT / "engine/openvino/iq36-four-fc-horizontal-fusion.patch"
BOUND = ROOT / (
    "output/openvino-fc-rms-igc-qk-rope-bundle-bound-"
    "20260718Tseq1328-cleanZ/metrics.json")
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
      "engine/openvino/iq36-four-fc-horizontal-fusion.patch",
      "tools/intel-qwen36-openvino-fc-rms-igc-qk-rope-bundle-bound.py",
      "tools/intel-qwen36-openvino-four-fc-horizontal-fusion-source-gate.py",
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
  required = (SOURCE, TEST, PATCH, BOUND)
  missing = [display(path) for path in required if not path.is_file()]
  if missing:
    raise SystemExit("missing four-FC source inputs: " + ", ".join(missing))

  stop_bytes = int(args.memory_stop_gib * 1024**3)
  memory = [{"stage": "start", "available_bytes": available_memory_bytes()}]
  if memory[0]["available_bytes"] < stop_bytes:
    raise RuntimeError("memory stop tripped before source gate")
  git = git_state(output)
  bound = load_json(BOUND)
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

  source_contract = {
      "max_four_count": source_text.count(
          "const int max_num_fcs_to_fuse = 4;"),
      "max_three_count": source_text.count(
          "const int max_num_fcs_to_fuse = 3;"),
      "compressed_fc_matcher": (
          "wrap_type<op::FullyConnectedCompressed>" in source_text),
      "weight_concat_axis_zero": (
          "Concat>(weight_nodes_as_output_vector, 0)" in source_text),
      "scale_concat_axis_zero": (
          "Concat>(scales_as_output_vector, 0)" in source_text),
      "zero_point_concat_axis_zero": (
          "Concat>(zp_nodes_as_output_vector, 0)" in source_text),
      "original_width_split": (
          "orig_n_sizes" in source_text
          and "VariadicSplit>(new_fc, axis_const, split_const)" in
          source_text),
  }
  test_contract = {
      "fourth_weight_input_occurrences": test_text.count(
          "ov::Shape{64, 4096}"),
      "fourth_scale_input_occurrences": test_text.count(
          "ov::Shape{64, 32}"),
      "four_input_weight_concat": (
          "OutputVector{weight1, weight2, weight3, weight4}" in test_text),
      "four_input_scale_concat": (
          "OutputVector{scale1, scale2, scale3, scale4}" in test_text),
      "four_unequal_output_widths": (
          "orig_n_sizes = {1024, 512, 128, 64}" in test_text),
      "four_way_split_shape": (
          "ov::Shape{4}, orig_n_sizes" in test_text),
      "fourth_split_output_consumed": (
          "Reshape>(split->output(3), reshape_pattern, true)" in test_text),
      "four_results_expected": (
          "ResultVector{result1, result2, result3, result4}" in test_text),
  }
  exact_source_contract = (
      source_contract["max_four_count"] == 1
      and source_contract["max_three_count"] == 0
      and all(value for key, value in source_contract.items()
              if key not in {"max_four_count", "max_three_count"}))
  exact_test_contract = (
      test_contract["fourth_weight_input_occurrences"] == 2
      and test_contract["fourth_scale_input_occurrences"] == 2
      and all(value for key, value in test_contract.items()
              if key not in {"fourth_weight_input_occurrences",
                             "fourth_scale_input_occurrences"}))
  bound_counts = bound["locked_ir"]["counts"]
  bound_bytes = bound["locked_ir"]["parameter_bytes"]
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
      check("four_way_compressed_fc_source_contract_exact",
            exact_source_contract, source=source_contract),
      check("four_unequal_width_outputs_have_unit_graph_coverage",
            exact_test_contract, test=test_contract),
      check("seq1328_bound_remains_exact_and_source_only",
            bound.get("required_checks_passed") is True
            and bound.get("source_edit_admitted") is True
            and bound.get("compiler_build_admitted") is False
            and bound.get("conservative_product_bound_passed") is False),
      check("rewrite_scope_is_exact_70_groups_280_matmuls",
            bound_counts["candidate_four_fc_groups"] == 70
            and bound_counts["candidate_four_fc_matmuls"] == 280
            and bound_counts["existing_full_qkv_groups"] == 10,
            counts=bound_counts),
      check("rewrite_parameter_bytes_are_complete",
            bound_bytes == {
                "linear_attention": 409098240,
                "router_shared": 56568960,
                "full_attention_qkv": 101744640},
            parameter_bytes=bound_bytes),
      check("expected_runtime_census_is_exact",
            bound["runtime_census"] == {
                "current_fully_connected_compressed": 371,
                "existing_three_way_qkv_groups": 10,
                "new_four_way_groups": 70,
                "removed_fully_connected_compressed": 210,
                "target_fully_connected_compressed": 161}),
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
      "rewrite_contract": {
          "four_fc_groups": 70,
          "input_matmuls": 280,
          "output_slices": 280,
          "new_variadic_splits": 70,
          "expected_fully_connected_compressed": 161,
          "runtime_activation_proven": False,
      },
      "next_action": {
          "route": "openvino_four_fc_horizontal_fusion_incremental_build",
          "admitted_builds": 1,
          "memory_stop_bytes": stop_bytes,
          "requirements": [
              "build only the candidate GPU plugin, serially",
              "record peak RSS, swap, plugin hash, and exact target diff",
              "restore no source files and launch no GPU/model worker yet",
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
  report = f"""# Four-way compressed-FC horizontal-fusion source gate

Verdict: **{verdict}**. Required checks:
`{str(required_checks_passed).lower()}`. No compiler, GPU context, or model
worker ran.

The durable patch is byte-identical to the target diff applied over pinned
OpenVINO `{base_commit}`. It raises only the compressed horizontal-FC maximum
from three to four and extends the existing no-bias/no-ZP unit graph with four
unequal widths `[1024, 512, 128, 64]`. The existing Concat and VariadicSplit
mapping remains unchanged.

Seq1328 maps that one source limit to exactly 70 locked groups, 280 input
MatMuls, 280 restored outputs, and a predicted `371 -> 161` FC census. Runtime
activation is not yet proven. Admit one serial incremental GPU-plugin build
behind the 4-GiB stop; do not launch a GPU/model worker. OOM observed: false.
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
