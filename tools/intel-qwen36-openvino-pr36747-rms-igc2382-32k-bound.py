#!/usr/bin/env python3
"""Admit one 32k candidate-only diagnostic for the clean RMS bundle.

This gate consumes the retained 2k component and the exact seq1172 stock
teacher row.  It audits source, plugin, IGC, prompt, token, census, memory,
and performance contracts without creating a compiler, GPU context, or model
worker.  Passing admits exactly one serial 32k/17-step candidate worker.
"""

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
SCHEMA = "intel-qwen36-openvino-pr36747-rms-igc2382-32k-bound-v0"
SOURCE_TREE = Path(
    "/home/intel/intel-qwen36-r0/source/openvino-90214e5be05")
PLUGIN = Path(
    "/home/intel/intel-qwen36-r0/output/openvino-90214e-l0-gpu/"
    "bin/intel64/Release/libopenvino_intel_gpu_plugin.so")
PATCH = ROOT / "engine/openvino/iq36-router-shared-pr36747-rms.patch"
SOURCE_GATE = ROOT / (
    "output/openvino-pr36747-rms-igc2382-source-gate-"
    "20260718Tseq1347-cleanZ/metrics.json")
BUILD = ROOT / (
    "output/openvino-pr36747-rms-igc2382-build-"
    "20260718Tseq1348-cleanZ/metrics.json")
COMPONENT = ROOT / (
    "output/openvino-pr36747-rms-igc2382-component-"
    "20260718Tseq1349-candidate-2k-warm17-cleanZ/metrics.json")
COMPONENT_MANIFEST = COMPONENT.with_name("manifest.json")
REFERENCE = ROOT / (
    "output/openvino-attention-phase-profile-"
    "20260715Tseq1172-l0-dq-restored-32k-warm17-cleanZ/"
    "raw/32k/stock/worker-result.json")
REFERENCE_TOKENS = REFERENCE.with_name("prompt-token-ids.u32")
PROMPT = ROOT / (
    "output/r0-oracle-prompt-materialization-20260626T082201Z/"
    "prompts/sentinel_032k.txt")
FRONTIER = ROOT / f"doc/active/{WS}/frontier.json"
MATRIX = ROOT / f"benchmarks/{WS}/acceptance-matrix.json"
GRAPH_SOURCE = ROOT / "tools/intel_qwen36_openvino_hot_cold_attention.py"
WORKER_SOURCE = ROOT / (
    "tools/intel-qwen36-openvino-hot-cold-attention-gate.py")
CUSTOM_CONFIG = ROOT / "engine/openvino/custom/iq36_hot_attention_gqa.xml"
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
IGC_LIBRARY_DIR = Path("/tmp/iq36-igc-2.38.2-root/usr/local/lib")
EXPECTED_IGC_LIBRARIES = {
    "libigc.so.2":
        "ff0cc269af1b2f843521b9207c54370fddab25caa404b1322cbdb4598452da33",
    "libigdfcl.so.2":
        "edd0cc3c73fee76ce156b8a8281d5a747f2634bc81a95da0ca1af9e72abd8de2",
    "libopencl-clang2.so.17":
        "5ad86d1aa4c4b92ca5ff96cbe2ca96d888b5afc5517e3c23b1772983c4dec63b",
}
EXPECTED_PLUGIN_SHA256 = (
    "432648af80a3da501d2b8d3611fcce04484b820dd963f59b8616728f44cfda64")
EXPECTED_PATCH_SHA256 = (
    "392f8fdc5d9d5521e3e2aaea7d3b9a6287238e2a60904a55eef02ec517f04e8d")
EXPECTED_TOKEN_SHA256 = (
    "3b26c4cbf7aec17e2e4e9d8ea9ac7b39052a20df0d04d1277d2a292f91ed651c")
EXPECTED_TOP1 = [
    271, 248068, 198, 8160, 579, 264, 7047, 1817, 25,
    271, 16, 13, 220, 2972, 2014, 53983, 2570, 5396,
]
EXPECTED_CORE_COUNTS = {
    "Assign": 60,
    "FullyConnectedCompressed": 291,
    "GatedDeltaNet": 30,
    "IQ36HotAttentionGQA": 10,
    "IQ36LinearConvSwish": 30,
    "IQ36QKRopeLayout": 10,
    "RMS": 131,
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
      "tools/intel-qwen36-openvino-pr36747-rms-igc2382-32k-bound.py",
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


def main() -> int:
  args = parse_args()
  output = args.output.resolve()
  output.mkdir(parents=True, exist_ok=False)
  igc_paths = tuple(IGC_LIBRARY_DIR / name for name in EXPECTED_IGC_LIBRARIES)
  required = (
      PATCH, SOURCE_GATE, BUILD, COMPONENT, COMPONENT_MANIFEST, REFERENCE,
      REFERENCE_TOKENS, PROMPT, FRONTIER, MATRIX, GRAPH_SOURCE, WORKER_SOURCE,
      CUSTOM_CONFIG, PLUGIN, *igc_paths,
      *(SOURCE_TREE / path for path in TARGET_PATHS),
  )
  missing = [display(path) for path in required if not path.is_file()]
  if missing:
    raise SystemExit("missing 32k-bound inputs: " + ", ".join(missing))

  stop_bytes = int(args.memory_stop_gib * 1024**3)
  start_available = available_memory_bytes()
  if start_available < stop_bytes:
    raise RuntimeError("memory stop tripped before 32k bound")
  git = git_state(output)
  source_gate = load_json(SOURCE_GATE)
  build = load_json(BUILD)
  component = load_json(COMPONENT)
  component_manifest = load_json(COMPONENT_MANIFEST)
  reference = load_json(REFERENCE)
  frontier = load_json(FRONTIER)
  matrix = load_json(MATRIX)
  graph_text = GRAPH_SOURCE.read_text(encoding="utf-8")
  worker_text = WORKER_SOURCE.read_text(encoding="utf-8")
  config_text = CUSTOM_CONFIG.read_text(encoding="utf-8")
  target_diff = run(["git", "diff", "--", *TARGET_PATHS], SOURCE_TREE)
  reverse_check = run(
      ["git", "apply", "--reverse", "--check", str(PATCH)], SOURCE_TREE)
  observed_igc = {path.name: sha256(path) for path in igc_paths}
  plugin_hash = sha256(PLUGIN)
  patch_hash = sha256(PATCH)
  token_hash = sha256(REFERENCE_TOKENS)
  phases = reference.get("phases", [])
  reference_top1 = [int(row.get("top1", -1)) for row in phases]
  profile = component.get("profile", {}).get("candidate", {})
  goal_budget = frontier.get("goal_budget", {})
  cap = float(goal_budget.get("per_token_ms", {}).get(
      "effective_cap", math.nan))
  floor = float(matrix.get("bootstrap_targets", {}).get(
      "decode_tokens_s", {}).get("32768", math.nan))
  smoothness_cap = float(matrix.get("smoothness", {}).get(
      "decode_tpot_p95_over_p50_max", math.nan))
  worker_contract = {
      "mode": "candidate",
      "lane": "32k",
      "prompt": str(PROMPT.resolve()),
      "prompt_tokens": 32768,
      "prefill_chunk_tokens": 8192,
      "decode_steps": 17,
      "decode_tokens": EXPECTED_TOP1[:17],
      "same_infer_request": True,
      "fixed_cold_capacity": 32768,
      "prefill_history_capacity": 32768,
      "initialize_hot_states": True,
      "skip_hot_state_self_bind": True,
      "capture_full_profile": True,
      "fuse_linear_conv_state": True,
      "fuse_qk_rope_layout": True,
      "pack_gdn_state": False,
      "phase_branch_prefill": False,
      "stock_prefill_custom_decode": False,
      "stock_prefill_sliced_decode": False,
      "static_phase_separated": False,
      "target_layers": list(range(3, 40, 4)),
      "plugin_sha256": EXPECTED_PLUGIN_SHA256,
      "igc_library_dir": str(IGC_LIBRARY_DIR),
      "memory_preflight_gib": 8.0,
      "memory_abort_gib": 4.0,
  }
  end_available = available_memory_bytes()
  checks = [
      check("repository_clean_at_gate", not git["dirty"], git=git),
      check("seq1347_and_seq1348_pin_the_exact_clean_build",
            source_gate.get("required_checks_passed") is True
            and source_gate.get("plugin_build_admitted") is True
            and source_gate.get("model_worker_admitted") is False
            and build.get("required_checks_passed") is True
            and build.get("candidate_plugin_retained") is True
            and build.get("candidate_plugin_after", {}).get("sha256") ==
                EXPECTED_PLUGIN_SHA256
            and build.get("build", {}).get("oom_observed") is False
            and build.get("build", {}).get("memory_guard_tripped") is False),
      check("seq1349_is_exact_correct_and_above_the_short_kill_number",
            component.get("evidence_checks_passed") is True
            and component.get("route_accepted") is True
            and component.get("activation_passed") is True
            and component.get("correctness_passed") is True
            and component.get("performance_passed") is True
            and component.get("actual_top1") == EXPECTED_TOP1
            and component.get("worker", {}).get("oom_observed") is False),
      check("seq1349_runtime_census_is_the_clean_bundle",
            profile.get("core_counts") == EXPECTED_CORE_COUNTS
            and profile.get("core_counts_exact") is True
            and profile.get("fused_four_fc_count") == 0
            and profile.get("fused_three_fc_count") == 50
            and profile.get("fused_shared_triple_count") == 40
            and profile.get("fused_linear_tail_triple_count") == 0
            and profile.get("existing_fused_qkv_count") == 10
            and profile.get("unfused_router_gate_count") == 40
            and profile.get("unfused_linear_original_count") == 120
            and profile.get("rms_executed_count") == 131
            and profile.get("rms_exec_types") == {
                "rms_gpu_bfyx_opt__f16": 131}
            and profile.get("old_qk_boundary_executed") == 0
            and profile.get("qk_rope_layout_executed") == 10),
      check("current_plugin_patch_and_igc_identities_are_exact",
            plugin_hash == EXPECTED_PLUGIN_SHA256
            and component_manifest.get("plugin_sha256") == plugin_hash
            and patch_hash == EXPECTED_PATCH_SHA256
            and target_diff.returncode == 0
            and target_diff.stdout == PATCH.read_text(encoding="utf-8")
            and reverse_check.returncode == 0
            and observed_igc == EXPECTED_IGC_LIBRARIES,
            plugin_sha256=plugin_hash, patch_sha256=patch_hash,
            igc_libraries=observed_igc),
      check("seq1172_stock_is_an_exact_32k_teacher_row",
            reference.get("mode") == "stock"
            and reference.get("lane") == "32k"
            and reference.get("prompt", {}).get("path") == str(PROMPT.resolve())
            and reference.get("prompt", {}).get("token_count") == 32768
            and reference.get("prompt", {}).get("prefill_chunk_tokens") == 8192
            and reference.get("prompt", {}).get("token_sha256") ==
                EXPECTED_TOKEN_SHA256
            and token_hash == EXPECTED_TOKEN_SHA256
            and len(phases) == 18
            and reference_top1 == EXPECTED_TOP1
            and phases[0].get("input_tokens") == 32768
            and all(row.get("input_tokens") == 1 for row in phases[1:])
            and all(row.get("logits_finite") is True for row in phases)
            and reference.get("same_infer_request") is True
            and reference.get("hot_state_self_bind_skipped") is True,
            top1=reference_top1, prompt_token_sha256=token_hash),
      check("graph_worker_and_custom_config_admit_the_exact_union",
            all(token in graph_text for token in (
                "fuse_qk_rope_layout",
                "query_concat.output(0).replace(qk_rope.output(0))",
                "key_concat.output(0).replace(qk_rope.output(1))"))
            and 'cfg.get("fuse_qk_rope_layout", False)' in worker_text
            and 'cfg.get("skip_hot_state_self_bind", False)' in worker_text
            and config_text.count(
                '<CustomLayer name="IQ36QKRopeLayout"') == 1),
      check("registered_32k_caps_are_exact",
            math.isclose(floor, 37.16, abs_tol=1e-12)
            and math.isclose(cap, 1000.0 / floor, abs_tol=5e-7)
            and math.isclose(smoothness_cap, 1.25, abs_tol=1e-12)),
      check("one_candidate_only_worker_contract_is_bounded",
            worker_contract["decode_steps"] == 17
            and worker_contract["memory_preflight_gib"] == 8.0
            and worker_contract["memory_abort_gib"] == 4.0),
      check("no_compiler_gpu_stock_or_product_worker_ran", True,
            compilers=0, gpu_contexts=0, candidate_workers=0,
            stock_workers=0, product_workers=0),
      check("memory_stop_never_tripped",
            min(start_available, end_available) >= stop_bytes,
            start_available_bytes=start_available,
            end_available_bytes=end_available, stop_bytes=stop_bytes),
  ]
  required_checks_passed = all(row["pass"] for row in checks)
  verdict = (
      "admit_one_serial_pr36747_rms_igc2382_32k_candidate"
      if required_checks_passed else "inconclusive")
  metrics = {
      "schema": SCHEMA,
      "created_at": datetime.now(timezone.utc).isoformat(),
      "workstream": WS,
      "git": git,
      "verdict": verdict,
      "required_checks_passed": required_checks_passed,
      "candidate_32k_workers_admitted": 1 if required_checks_passed else 0,
      "stock_workers_admitted": 0,
      "product_workers_admitted": 0,
      "abba_blocks_admitted": 0,
      "output512_admitted": False,
      "speed_claim": False,
      "expected_top1": EXPECTED_TOP1,
      "expected_core_counts": EXPECTED_CORE_COUNTS,
      "worker_contract": worker_contract,
      "diagnostic_gates": {
          "stable_sample_rule": "drop first decode JIT sample",
          "stable_samples": 16,
          "decode_wall_median_cap_ms": cap,
          "decode_floor_tokens_per_second": floor,
          "decode_tpot_p95_over_p50_max": smoothness_cap,
          "exact_teacher_top1_required": True,
          "finite_logits_required": True,
          "exact_runtime_census_required": True,
          "same_infer_request_required": True,
          "promotion_inference": False,
      },
      "next_action": {
          "run_exactly_one_candidate_32k_worker": required_checks_passed,
          "on_pass": "source-gate product harness integration",
          "on_fail": "close unchanged clean RMS bundle at long context",
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
      "candidate_workers": 0,
      "stock_workers": 0,
      "product_workers": 0,
  })
  report = f"""# Clean RMS bundle 32k candidate bound

Verdict: **{verdict}**. Required checks:
`{str(required_checks_passed).lower()}`. No compiler, GPU context, stock,
candidate, or product worker ran.

The admitted diagnostic is exactly one serial candidate-only 32k/17-step
worker using the seq1172 stock teacher IDs, the clean seq1349 plugin and
seven-file patch, isolated IGC 2.38.2, one InferRequest, 8-GiB preflight, and
the 4-GiB abort line. It must preserve all 18 IDs and the exact
291-FC/50-triple/131-RMS census.

After dropping the first decode JIT sample, the 16 stable walls must have a
median no greater than `{cap:.6f} ms` and p95/p50 no greater than
`{smoothness_cap:.2f}`. This is a diagnostic route gate, not paired product
inference or a speed claim. ABBA, stock, output512, and product workers remain
inadmissible.
"""
  (output / "report.md").write_text(report, encoding="utf-8")
  print(json.dumps({
      "artifact": display(output),
      "verdict": verdict,
      "required_checks_passed": required_checks_passed,
      "candidate_32k_workers_admitted": (
          1 if required_checks_passed else 0),
      "decode_wall_median_cap_ms": cap,
      "smoothness_cap": smoothness_cap,
  }, separators=(",", ":")), flush=True)
  return 0 if required_checks_passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
