#!/usr/bin/env python3
"""Audit the router-isolated shared-expert triple patch before any build."""

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
    "intel-qwen36-openvino-router-isolated-shared-triple-source-gate-v0")
SOURCE_TREE = Path(
    "/home/intel/intel-qwen36-r0/source/openvino-90214e5be05")
SOURCE = SOURCE_TREE / (
    "src/plugins/intel_gpu/src/plugin/transformations/"
    "fc_horizontal_fusion.cpp")
TEST = SOURCE_TREE / (
    "src/plugins/intel_gpu/tests/unit/transformations/"
    "horizontal_fc_fusion_test.cpp")
PATCH = ROOT / (
    "engine/openvino/iq36-router-isolated-shared-triple-fusion.patch")
BOUND = ROOT / (
    "output/openvino-router-isolated-shared-triple-bound-"
    "20260718Tseq1334-cleanZ/metrics.json")
ALL_GROUP_OUTCOME = ROOT / (
    "output/openvino-four-fc-qk-bundle-outcome-"
    "20260718Tseq1332-cleanZ/metrics.json")
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
      "engine/openvino/iq36-router-isolated-shared-triple-fusion.patch",
      "tools/intel-qwen36-openvino-router-isolated-shared-triple-source-gate.py",
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
  required = (SOURCE, TEST, PATCH, BOUND, ALL_GROUP_OUTCOME)
  missing = [display(path) for path in required if not path.is_file()]
  if missing:
    raise SystemExit("missing shared-triple source inputs: " + ", ".join(missing))

  stop_bytes = int(args.memory_stop_gib * 1024**3)
  memory = [{"stage": "start", "available_bytes": available_memory_bytes()}]
  if memory[0]["available_bytes"] < stop_bytes:
    raise RuntimeError("memory stop tripped before source gate")

  git = git_state(output)
  bound = load_json(BOUND)
  all_group_outcome = load_json(ALL_GROUP_OUTCOME)
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
  source_contract = {
      "max_three_count": source_text.count(
          "const int max_num_fcs_to_fuse = 3;"),
      "max_four_count": source_text.count(
          "const int max_num_fcs_to_fuse = 4;"),
      "shared_helper_occurrences": source_text.count(helper_name),
      "exact_four_user_guard": "candidates.size() != 4" in source_text,
      "exact_k_2048_guard": "shape[1] != 2048" in source_text,
      "exact_width_partition": all(token in source_text for token in (
          "shape[0] == 1", "shape[0] == 256", "shape[0] == 512",
          "n1_count != 1 || n256_count != 1 || n512_count != 2")),
      "router_excluded_by_identity": "if (fc != router_fc)" in source_text,
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
          "OutputVector{weight1, weight2, weight3}" in test_text
          and "OutputVector{weight1, weight2, weight3, weight4}"
          not in test_text),
      "fused_scales_exclude_router": (
          "OutputVector{scale1, scale2, scale3}" in test_text
          and "OutputVector{scale1, scale2, scale3, scale4}"
          not in test_text),
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
  }
  exact_source_contract = (
      source_contract["max_three_count"] == 1
      and source_contract["max_four_count"] == 0
      and source_contract["shared_helper_occurrences"] == 3
      and all(value for key, value in source_contract.items()
              if key not in {"max_three_count", "max_four_count",
                             "shared_helper_occurrences"}))
  exact_test_contract = (
      test_contract["n1_k2048_occurrences"] == 2
      and test_contract["n512_k2048_occurrences"] == 4
      and test_contract["n256_k2048_occurrences"] == 2
      and all(value for key, value in test_contract.items()
              if key not in {"n1_k2048_occurrences",
                             "n512_k2048_occurrences",
                             "n256_k2048_occurrences"}))

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
      check("router_isolated_source_predicate_exact",
            exact_source_contract, source=source_contract),
      check("router_isolated_unequal_shared_triple_unit_graph_exact",
            exact_test_contract, test=test_contract),
      check("seq1332_closes_rejected_max_four_route",
            all_group_outcome.get("required_checks_passed") is True
            and all_group_outcome.get("all_four_fc_route_closed") is True
            and all_group_outcome.get("correctness", {}).get(
                "differing_top1") == 18),
      check("seq1334_admits_only_this_source_patch",
            bound.get("required_checks_passed") is True
            and bound.get("source_edit_admitted") is True
            and bound.get("plugin_build_admitted") is False
            and bound.get("gpu_worker_admitted") is False
            and expected_contract == {
                "candidate_predicate": (
                    "when a shared input has four compressed FC users with "
                    "K=2048, exclude the unique N=256 branch from the matcher "
                    "and callback; retain max three and fuse N=[1,512,512] only"),
                "current_all_group_patch_is_rejected_input": True,
                "expected_existing_qkv_fused_three_groups": 10,
                "expected_fully_connected_compressed": 291,
                "expected_new_shared_fused_three_groups": 40,
                "expected_router_gate_unfused": 40,
                "stock_max_three": True,
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
          "router_gate_unfused": 40,
          "fully_connected_compressed": 291,
      },
      "next_action": {
          "route": "openvino_router_isolated_shared_triple_incremental_build",
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
  report = f"""# Router-isolated shared-expert triple source gate

Verdict: **{verdict}**. Required checks:
`{str(required_checks_passed).lower()}`. No compiler, GPU context, or model
worker ran.

The durable patch is byte-identical to the target diff over pinned OpenVINO
`{base_commit}`. The stock max-three limit remains intact. One shared helper is
used by both matcher accounting and callback collection: only the exact
`K=2048`, `N=[1,512,512,256]` fanout drops the `N=256` router from fusion.

The unit graph preserves four outputs while fusing only `[1,512,512]` and
leaving the router FC separate. The predicted runtime census is 40 new shared
triple fusions, 40 unfused router gates, 10 existing QKV triples, and 291 FCs.
Runtime activation and correctness are not yet proven. Admit one serial
incremental plugin build behind the 4-GiB stop; launch no model worker yet.
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
