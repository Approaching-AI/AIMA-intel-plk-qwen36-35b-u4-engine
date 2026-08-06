#!/usr/bin/env python3
"""Admit a clean PR36747 RMS + IGC route from the retained isolation worker."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-openvino-pr36747-rms-igc2382-isolation-bound-v0"
SOURCE_TREE = Path(
    "/home/intel/intel-qwen36-r0/source/openvino-90214e5be05")
FC_SOURCE = SOURCE_TREE / (
    "src/plugins/intel_gpu/src/plugin/transformations/"
    "fc_horizontal_fusion.cpp")
FC_TEST = SOURCE_TREE / (
    "src/plugins/intel_gpu/tests/unit/transformations/"
    "horizontal_fc_fusion_test.cpp")
TARGET_PATHS = (
    str(FC_SOURCE.relative_to(SOURCE_TREE)),
    str(FC_TEST.relative_to(SOURCE_TREE)),
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
PATCH = ROOT / "engine/openvino/iq36-router-shared-pr36747-rms.patch"
PR_PATCH = ROOT / (
    "output/openvino-post-igc-opportunity-bound-20260717Tseq1302-"
    "cleanZ/raw/openvino-pr36747.patch")
ISOLATION = ROOT / (
    "output/openvino-linear-tail-rms-igc2382-component-"
    "20260718Tseq1345-candidate-2k-warm17-cleanZ/metrics.json")
ISOLATION_MANIFEST = ISOLATION.with_name("manifest.json")
TAIL_CORRECTED = ROOT / (
    "output/openvino-linear-tail-rms-igc2382-component-"
    "20260718Tseq1345b-candidate-2k-warm17-cleanZ/metrics.json")
SHARED_COMPONENT = ROOT / (
    "output/openvino-router-isolated-shared-triple-component-"
    "20260718Tseq1337-candidate-2k-warm17-cleanZ/metrics.json")
TAIL_BOUND = ROOT / (
    "output/openvino-linear-tail-rms-igc-bundle-bound-"
    "20260718Tseq1342-cleanZ/metrics.json")
EXPECTED_ISOLATION_PLUGIN = (
    "bffa228a58214cbc7dfa78548b6d4b80960fadf5d712dc31713ed005dbaa2c40")
EXPECTED_PR_SHA256 = (
    "5e0e17b5908a6aa1bb696442193d36e7d8108e5bd1d1335b031643bdda3665bf")
EXPECTED_IGC_LIBRARIES = {
    "libigc.so.2":
        "ff0cc269af1b2f843521b9207c54370fddab25caa404b1322cbdb4598452da33",
    "libigdfcl.so.2":
        "edd0cc3c73fee76ce156b8a8281d5a747f2634bc81a95da0ca1af9e72abd8de2",
    "libopencl-clang2.so.17":
        "5ad86d1aa4c4b92ca5ff96cbe2ca96d888b5afc5517e3c23b1772983c4dec63b",
}


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
      PATCH, PR_PATCH, ISOLATION, ISOLATION_MANIFEST, TAIL_CORRECTED,
      SHARED_COMPONENT, TAIL_BOUND, FC_SOURCE, FC_TEST,
      *(SOURCE_TREE / path for path in TARGET_PATHS[2:]))
  missing = [display(path) for path in required if not path.is_file()]
  if missing:
    raise SystemExit("missing isolation-bound inputs: " + ", ".join(missing))

  stop_bytes = int(args.memory_stop_gib * 1024**3)
  start_available = available_memory_bytes()
  if start_available < stop_bytes:
    raise RuntimeError("memory stop tripped before isolation bound")
  git = git_state(output)
  isolation = load_json(ISOLATION)
  manifest = load_json(ISOLATION_MANIFEST)
  tail = load_json(TAIL_CORRECTED)
  shared = load_json(SHARED_COMPONENT)
  tail_bound = load_json(TAIL_BOUND)
  profile = isolation["profile"]["candidate"]
  tail_profile = tail["profile"]["candidate"]
  performance = isolation["performance"]
  worker = isolation["worker"]
  source_text = FC_SOURCE.read_text(encoding="utf-8")
  test_text = FC_TEST.read_text(encoding="utf-8")
  target_diff = run(["git", "diff", "--", *TARGET_PATHS], SOURCE_TREE)
  reverse_check = run(
      ["git", "apply", "--reverse", "--check", str(PATCH)], SOURCE_TREE)
  pr_reverse_check = run(
      ["git", "apply", "--reverse", "--check", str(PR_PATCH)],
      SOURCE_TREE)
  patch_text = PATCH.read_text(encoding="utf-8")
  end_available = available_memory_bytes()

  total_saving = float(performance["total_observed_saving_ms"])
  kill_number = float(performance["required_total_saving_ms"])
  incremental = float(
      performance["incremental_tail_rms_igc_observed_saving_ms"])
  incremental_required = float(
      performance["required_incremental_tail_rms_igc_saving_ms"])
  only_activation_check_failed = (
      sum(not row["pass"] for row in isolation["evidence_checks"]) == 1
      and next(row for row in isolation["evidence_checks"]
               if not row["pass"])["name"] ==
          "linear_tail_shared_qkv_and_rms_activation_is_exact")
  checks = [
      check("repository_clean_at_gate", not git["dirty"], git=git),
      check("seq1345_is_an_exact_safe_inactive_tail_isolation_worker",
            isolation.get("activation_passed") is False
            and isolation.get("correctness_passed") is True
            and isolation.get("performance_passed") is True
            and isolation.get("route_accepted") is False
            and only_activation_check_failed
            and worker.get("returncode") == 0
            and worker.get("timed_out") is False
            and worker.get("oom_observed") is False
            and worker.get("memory_guard", {}).get("tripped") is False),
      check("seq1345_runtime_census_is_exact_qk_shared_plus_rms",
            profile.get("core_counts", {}).get(
                "FullyConnectedCompressed") == 291
            and profile.get("fused_four_fc_count") == 0
            and profile.get("fused_three_fc_count") == 50
            and profile.get("fused_shared_triple_count") == 40
            and profile.get("fused_linear_tail_triple_count") == 0
            and profile.get("existing_fused_qkv_count") == 10
            and profile.get("unfused_linear_original_count") == 120
            and profile.get("unfused_router_gate_count") == 40
            and profile.get("rms_executed_count") == 131
            and profile.get("rms_exec_types") == {
                "rms_gpu_bfyx_opt__f16": 131}
            and profile.get("qk_rope_layout_executed") == 10),
      check("seq1345_teacher_forced_tokens_are_exact",
            isolation.get("actual_top1") == isolation.get("expected_top1")
            and len(isolation.get("actual_top1", [])) == 18),
      check("seq1345_plugin_and_igc_identities_are_exact",
            manifest.get("plugin_sha256") == EXPECTED_ISOLATION_PLUGIN
            and manifest.get("igc_libraries") == EXPECTED_IGC_LIBRARIES
            and worker.get("igc_library_dir") ==
                "/tmp/iq36-igc-2.38.2-root/usr/local/lib"
            and worker.get("ld_library_path_first") ==
                "/tmp/iq36-igc-2.38.2-root/usr/local/lib"),
      check("seq1345_cross_artifact_screen_clears_kill_number",
            math.isclose(total_saving, 3.075892000000003,
                         abs_tol=1e-12)
            and total_saving >= kill_number
            and incremental >= incremental_required,
            total_saving_ms=total_saving,
            kill_number_ms=kill_number,
            total_margin_ms=total_saving - kill_number,
            incremental_saving_ms=incremental,
            incremental_required_ms=incremental_required),
      check("seq1345b_conclusively_closes_linear_horizontal_fusion",
            tail.get("evidence_checks_passed") is True
            and tail.get("activation_passed") is True
            and tail.get("correctness_passed") is False
            and tail.get("performance_passed") is False
            and tail_profile.get("core_counts", {}).get(
                "FullyConnectedCompressed") == 231
            and tail_profile.get("fused_linear_tail_triple_count") == 30
            and tail_profile.get("unfused_linear_original_count") == 30
            and tail.get("worker", {}).get("oom_observed") is False),
      check("retained_shared_baseline_and_kill_arithmetic_are_exact",
            shared.get("correctness_passed") is True
            and shared["profile"]["candidate"].get(
                "fused_shared_triple_count") == 40
            and math.isclose(
                float(tail_bound["budget"]["kill_number_ms"]), kill_number,
                abs_tol=1e-12)
            and math.isclose(
                float(tail_bound["budget"]["measured_qk_shared_ms"]),
                1.9928330000000045, abs_tol=1e-12)),
      check("clean_source_removes_linear_subset_and_retains_router_subset",
            "n8192_count" not in source_text
            and "linear_tail" not in test_text
            and "n256_count" in source_text
            and "FullyConnectedHorizontalFusion_router_isolated_"
                "shared_triple_no_bias_no_zp" in test_text),
      check("clean_shared_pr36747_patch_is_exactly_applied",
            target_diff.returncode == 0
            and target_diff.stdout == patch_text
            and reverse_check.returncode == 0
            and pr_reverse_check.returncode == 0
            and sha256(PR_PATCH) == EXPECTED_PR_SHA256,
            patch_sha256=sha256(PATCH),
            pr_patch_sha256=sha256(PR_PATCH)),
      check("no_compiler_gpu_or_model_worker_ran", True,
            compilers=0, gpu_contexts=0, model_workers=0),
      check("memory_guard_never_tripped",
            min(start_available, end_available) >= stop_bytes,
            start_available_bytes=start_available,
            end_available_bytes=end_available,
            stop_bytes=stop_bytes),
  ]
  required_checks_passed = all(row["pass"] for row in checks)
  source_gate_admitted = required_checks_passed
  verdict = (
      "admit_clean_pr36747_rms_igc2382_source_gate"
      if source_gate_admitted else "inconclusive")
  metrics = {
      "schema": SCHEMA,
      "created_at": datetime.now(timezone.utc).isoformat(),
      "workstream": WS,
      "git": git,
      "verdict": verdict,
      "required_checks_passed": required_checks_passed,
      "source_gate_admitted": source_gate_admitted,
      "plugin_build_admitted": False,
      "gpu_worker_admitted": False,
      "speed_claim": False,
      "isolation": {
          "interpretation": (
              "seq1345 contains tail code but the missing N8192 pointer made "
              "that path inert; its exact runtime delta is the PR36747 plus "
              "IGC2.38.2 union over retained QK plus shared"),
          "candidate_plugin_sha256": EXPECTED_ISOLATION_PLUGIN,
          "candidate_median_ms": performance["candidate_median_ms"],
          "total_saving_ms": total_saving,
          "kill_number_ms": kill_number,
          "margin_ms": total_saving - kill_number,
          "correctness_passed": True,
          "runtime_fc_count": 291,
          "runtime_rms_count": 131,
      },
      "source_contract": {
          "preserve_existing_qkv_triples": 10,
          "preserve_shared_triples": 40,
          "preserve_unfused_router_gates": 40,
          "preserve_unfused_linear_branches": 120,
          "linear_horizontal_fusion_allowed": False,
          "pr36747_patch_sha256": EXPECTED_PR_SHA256,
          "isolated_igc2382": True,
          "expected_fully_connected_compressed": 291,
          "expected_fused_three_groups": 50,
          "expected_rms_consumers": 131,
      },
      "next_action": {
          "route": "openvino_pr36747_rms_igc2382_clean_source_gate",
          "requirements": [
              "audit the exact clean shared plus PR36747 seven-file patch",
              "build one incremental candidate plugin only after that gate",
              "launch no worker until plugin identity and memory checks pass",
          ],
      },
      "checks": checks,
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
  report = f"""# PR36747 RMS plus IGC 2.38.2 isolation bound

Verdict: **{verdict}**. Required checks:
`{str(required_checks_passed).lower()}`. No compiler, GPU context, or model
worker ran.

The retained seq1345 worker is a useful accidental isolation: its tail-fusion
code was present but inactive, and the runtime census remained exactly the
correct Q/K+shared graph (291 FCs, 40 shared triples, ten QKV triples, 120
independent linear FCs, 40 routers, and 131 RMS nodes). It kept all 18
teacher-forced token IDs exact under the pinned IGC 2.38.2 libraries.

Its 36.105984-ms short-screen median saves {total_saving:.6f} ms against the
39.181876-ms control, clearing the {kill_number:.6f}-ms kill-number by
{total_saving - kill_number:.6f} ms. This is cross-artifact component evidence,
not paired product inference or a speed claim.

The corrected seq1345b worker activated all 30 linear-tail triples but failed
both token correctness and the performance screen, closing linear horizontal
fusion. The next exact route removes it, retains shared fusion plus PR36747,
and admits only a clean source gate before any build or worker.
"""
  (output / "report.md").write_text(report, encoding="utf-8")
  print(json.dumps({
      "artifact": display(output),
      "verdict": verdict,
      "source_gate_admitted": source_gate_admitted,
      "isolation_correctness_passed": isolation.get("correctness_passed"),
      "isolation_total_saving_ms": total_saving,
      "kill_number_ms": kill_number,
      "gpu_or_model_worker_launched": False,
  }, separators=(",", ":")), flush=True)
  return 0 if required_checks_passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
