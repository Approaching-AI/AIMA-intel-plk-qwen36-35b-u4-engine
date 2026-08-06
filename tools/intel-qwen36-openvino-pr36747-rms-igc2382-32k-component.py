#!/usr/bin/env python3
"""Run the one admitted clean RMS-bundle 32k candidate diagnostic.

The worker is teacher-forced from the retained seq1172 stock row and uses the
exact clean seq1349 plugin, source patch, isolated IGC 2.38.2 libraries, and
accepted linear-state alias.  No stock, product, ABBA, or output512 worker is
launched.  The result is a long-context route decision, not a speed claim.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-openvino-pr36747-rms-igc2382-32k-component-v0"
BASE_TOOL = ROOT / (
    "tools/intel-qwen36-openvino-pr36747-rms-igc2382-component.py")
BOUND = ROOT / (
    "output/openvino-pr36747-rms-igc2382-32k-bound-"
    "20260718Tseq1350-cleanZ/metrics.json")
BOUND_MANIFEST = BOUND.with_name("manifest.json")
SHORT_COMPONENT = ROOT / (
    "output/openvino-pr36747-rms-igc2382-component-"
    "20260718Tseq1349-candidate-2k-warm17-cleanZ/metrics.json")
REFERENCE = ROOT / (
    "output/openvino-attention-phase-profile-"
    "20260715Tseq1172-l0-dq-restored-32k-warm17-cleanZ/"
    "raw/32k/stock/worker-result.json")
REFERENCE_TOKENS = REFERENCE.with_name("prompt-token-ids.u32")
PROMPT = ROOT / (
    "output/r0-oracle-prompt-materialization-20260626T082201Z/"
    "prompts/sentinel_032k.txt")
PATCH = ROOT / "engine/openvino/iq36-router-shared-pr36747-rms.patch"
SOURCE_TREE = Path(
    "/home/intel/intel-qwen36-r0/source/openvino-90214e5be05")
PLUGIN = Path(
    "/home/intel/intel-qwen36-r0/output/openvino-90214e-l0-gpu/"
    "bin/intel64/Release/libopenvino_intel_gpu_plugin.so")
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
LINEAR_SUFFIXES = (
    "linear_attn.in_proj_qkv/ov_ext::linear/MatMul",
    "linear_attn.in_proj_a/ov_ext::linear/MatMul",
    "linear_attn.in_proj_b/ov_ext::linear/MatMul",
    "linear_attn.in_proj_z/ov_ext::linear/MatMul",
)


def load_module() -> Any:
  spec = importlib.util.spec_from_file_location(
      "iq36_clean_rms_component_base", BASE_TOOL)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load {BASE_TOOL}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


BASEMOD = load_module()
BASEMOD.EXPECTED_CORE_COUNTS = EXPECTED_CORE_COUNTS
AUDITMOD = BASEMOD.BASEMOD
AUDITMOD.EXPECTED_CORE_COUNTS = EXPECTED_CORE_COUNTS
BASE = BASEMOD.BASE


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--plugin", type=Path, default=PLUGIN)
  parser.add_argument("--timeout-s", type=int, default=1800)
  parser.add_argument("--poll-interval-s", type=float, default=1.0)
  parser.add_argument("--min-available-gib", type=float, default=8.0)
  parser.add_argument("--abort-below-available-gib", type=float, default=4.0)
  parser.add_argument("--igc-library-dir", type=Path, default=IGC_LIBRARY_DIR,
                      help=argparse.SUPPRESS)
  parser.add_argument("--reuse-completed-worker", action="store_true",
                      help=argparse.SUPPRESS)
  args = parser.parse_args()
  if args.timeout_s <= 0 or args.poll_interval_s <= 0.0:
    parser.error("timeout and poll interval must be positive")
  if args.abort_below_available_gib > args.min_available_gib:
    parser.error("abort threshold must not exceed preflight threshold")
  if not math.isclose(args.min_available_gib, 8.0, abs_tol=1e-12):
    parser.error("this diagnostic requires the exact 8-GiB preflight")
  if not math.isclose(
      args.abort_below_available_gib, 4.0, abs_tol=1e-12):
    parser.error("this diagnostic requires the exact 4-GiB abort line")
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
      "tools/intel-qwen36-openvino-pr36747-rms-igc2382-32k-component.py",
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


def percentile(values: list[float], quantile: float) -> float:
  if not values:
    return math.nan
  ordered = sorted(values)
  rank = (len(ordered) - 1) * quantile
  lower = math.floor(rank)
  upper = math.ceil(rank)
  if lower == upper:
    return ordered[lower]
  return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def main() -> int:
  args = parse_args()
  output = args.output.resolve()
  worker_dir = output / "raw/32k/candidate"
  if args.reuse_completed_worker:
    if not worker_dir.is_dir():
      raise RuntimeError("completed worker directory does not exist")
    if (output / "metrics.json").exists():
      raise RuntimeError("completed artifact already has metrics")
  else:
    worker_dir.mkdir(parents=True, exist_ok=False)
  igc_paths = tuple(args.igc_library_dir / name
                    for name in EXPECTED_IGC_LIBRARIES)
  required = (
      BOUND, BOUND_MANIFEST, SHORT_COMPONENT, REFERENCE, REFERENCE_TOKENS,
      PROMPT, PATCH, BASE_TOOL, args.plugin, *igc_paths, BASE.WORKER,
      BASEMOD.QK.GRAPH_SOURCE, BASEMOD.QK.WORKER_SOURCE,
      BASEMOD.QK.KERNEL_SOURCE, BASEMOD.QK.CUSTOM_CONFIG,
      *(SOURCE_TREE / path for path in BASEMOD.TARGET_PATHS),
  )
  missing = [display(Path(path)) for path in required
             if not Path(path).is_file()]
  if missing:
    raise SystemExit("missing 32k component inputs: " + ", ".join(missing))

  git = git_state(output)
  concurrent = BASE.other_worker_pids()
  if concurrent:
    raise RuntimeError(f"concurrent OpenVINO worker detected: {concurrent}")
  if available_memory_bytes() < int(args.min_available_gib * 1024**3):
    raise RuntimeError("preflight memory is below the serial worker minimum")

  bound = load_json(BOUND)
  short = load_json(SHORT_COMPONENT)
  reference = load_json(REFERENCE)
  plugin = args.plugin.resolve()
  plugin_hash = sha256(plugin)
  observed_igc = {path.name: sha256(path) for path in igc_paths}
  target_diff = subprocess.run(
      ["git", "diff", "--", *BASEMOD.TARGET_PATHS], cwd=SOURCE_TREE,
      check=True, capture_output=True, text=True).stdout
  reverse_check = subprocess.run(
      ["git", "apply", "--reverse", "--check", str(PATCH)],
      cwd=SOURCE_TREE, check=False, capture_output=True, text=True)
  expected_top1 = [int(row.get("top1", -1))
                   for row in reference.get("phases", [])]
  cap = float(bound.get("diagnostic_gates", {}).get(
      "decode_wall_median_cap_ms", math.nan))
  smoothness_cap = float(bound.get("diagnostic_gates", {}).get(
      "decode_tpot_p95_over_p50_max", math.nan))
  preflight_checks = [
      check("repository_clean_at_gate", not git["dirty"], git=git),
      check("seq1350_admits_exactly_one_candidate_32k_worker",
            bound.get("required_checks_passed") is True
            and bound.get("candidate_32k_workers_admitted") == 1
            and bound.get("stock_workers_admitted") == 0
            and bound.get("product_workers_admitted") == 0
            and bound.get("abba_blocks_admitted") == 0
            and bound.get("output512_admitted") is False),
      check("seq1349_short_bundle_is_retained_and_exact",
            short.get("evidence_checks_passed") is True
            and short.get("route_accepted") is True
            and short.get("activation_passed") is True
            and short.get("correctness_passed") is True
            and short.get("performance_passed") is True),
      check("plugin_patch_and_igc_are_still_exact",
            plugin_hash == EXPECTED_PLUGIN_SHA256
            and sha256(PATCH) == EXPECTED_PATCH_SHA256
            and target_diff == PATCH.read_text(encoding="utf-8")
            and reverse_check.returncode == 0
            and observed_igc == EXPECTED_IGC_LIBRARIES,
            plugin_sha256=plugin_hash, igc_libraries=observed_igc),
      check("stock_teacher_and_prompt_are_still_exact",
            expected_top1 == EXPECTED_TOP1
            and len(reference.get("phases", [])) == 18
            and reference.get("mode") == "stock"
            and reference.get("lane") == "32k"
            and reference.get("prompt", {}).get("path") == str(PROMPT.resolve())
            and reference.get("prompt", {}).get("token_count") == 32768
            and reference.get("prompt", {}).get("token_sha256") ==
                EXPECTED_TOKEN_SHA256
            and sha256(REFERENCE_TOKENS) == EXPECTED_TOKEN_SHA256),
      check("no_concurrent_worker_at_launch", not concurrent,
            concurrent=concurrent),
  ]
  if not all(row["pass"] for row in preflight_checks):
    raise RuntimeError("clean RMS bundle 32k preflight did not pass")

  decode_tokens = expected_top1[:17]
  config = {
      "collect_states": False,
      "custom_config": str(BASEMOD.QK.CUSTOM_CONFIG.resolve()),
      "candidate_gpu_plugin": str(plugin),
      "decode_steps": 17,
      "decode_tokens": decode_tokens,
      "device": "GPU",
      "lane": "32k",
      "mode": "candidate",
      "model_dir": str(BASE.MODEL_DIR.resolve()),
      "prompt": str(PROMPT.resolve()),
      "prefill_chunk_tokens": 8192,
      "fixed_cold_capacity": 32768,
      "initialize_hot_states": True,
      "skip_hot_state_self_bind": True,
      "dump_runtime_graph": False,
      "capture_full_profile": True,
      "fuse_linear_conv_state": True,
      "fuse_qk_rope_layout": True,
      "pack_gdn_state": False,
      "prefill_history_capacity": 32768,
      "phase_branch_prefill": False,
      "stock_prefill_custom_decode": False,
      "stock_prefill_sliced_decode": False,
      "static_phase_separated": False,
      "raw": str(worker_dir),
      "result": str(worker_dir / "worker-result.json"),
      "target_layers": list(range(3, 40, 4)),
  }
  if args.reuse_completed_worker:
    saved_config = load_json(worker_dir / "worker-config.json")
    if saved_config != config:
      raise RuntimeError("completed worker config is not the exact contract")
    result_path = worker_dir / "worker-result.json"
    stdout_text = (worker_dir / "worker.stdout").read_text(
        encoding="utf-8", errors="replace")
    stderr_text = (worker_dir / "worker.stderr").read_text(
        encoding="utf-8", errors="replace")
    recovered_result = load_json(result_path)
    memory_samples = recovered_result.get("memory_samples", {})
    available_samples = [
        int(memory_samples[key]) for key in (
            "before_language_compile", "after_language_compile")
        if isinstance(memory_samples.get(key), int)]
    available_min = min(available_samples) if available_samples else None
    oom_markers = ("out of memory", "cl_out_of_resources", "killed")
    recovered_oom = any(marker in stderr_text.lower()
                        for marker in oom_markers)
    complete_event = '"event": "complete"' in stdout_text
    if not complete_event or recovered_oom:
      raise RuntimeError("completed worker cannot be safely recovered")
    worker = {
        "command": [str(BASE.OV_PYTHON), str(BASE.WORKER),
                    "--worker-config", str(worker_dir / "worker-config.json")],
        "environment": {
            "IQ36_GPU_ALIAS_LINEAR_STATE_ASSIGN": "1",
            "NEO_CACHE_DIR": str(worker_dir / "neo-cache"),
            "NEO_CACHE_MAX_SIZE": str(4 * 1024**3),
            "NEO_CACHE_PERSISTENT": "1",
        },
        "igc_library_dir": str(IGC_LIBRARY_DIR),
        "ld_library_path_first": str(IGC_LIBRARY_DIR),
        "elapsed_seconds": None,
        "memory_preflight": {
            "available_bytes": (
                available_samples[0] if available_samples else None),
            "required_bytes": int(args.min_available_gib * 1024**3),
            "source": "worker_result_before_language_compile",
        },
        "memory_guard": {
            "abort_below_bytes": int(
                args.abort_below_available_gib * 1024**3),
            "tripped": False,
        },
        "monitor": {
            "process_rss_peak_bytes": None,
            "process_swap_peak_bytes": None,
            "sample_count": len(available_samples),
            "system_available_min_bytes": available_min,
            "system_swap_used_peak_bytes": None,
            "source": (
                "worker_result_compile_samples_after_wrapper_metadata_was_"
                "lost_to_postprocess_error"),
        },
        "oom_observed": False,
        "returncode": 0,
        "timed_out": False,
        "reused_completed_worker": True,
        "wrapper_monitor_recovered": False,
        "result": recovered_result,
    }
  else:
    worker = BASE.launch_worker(args, worker_dir, config)
  result = worker["result"]
  phases = result.get("phases", [])
  actual_top1 = [int(row.get("top1", -1)) for row in phases]
  profile_error = None
  try:
    profile = AUDITMOD.runtime_audit(result) if result else {}
  except (KeyError, TypeError, ValueError) as error:
    profile = {}
    profile_error = f"{type(error).__name__}: {error}"

  stable_walls: list[float] = []
  wall_error = None
  try:
    walls = [float(row["wall_ms_diagnostic"]) for row in phases[1:]]
    if len(walls) != 17 or not all(
        math.isfinite(value) and value > 0.0 for value in walls):
      raise ValueError("worker does not have 17 finite decode walls")
    stable_walls = walls[1:]
  except (KeyError, TypeError, ValueError) as error:
    wall_error = f"{type(error).__name__}: {error}"
  median_ms = statistics.median(stable_walls) if stable_walls else math.nan
  p50_ms = percentile(stable_walls, 0.50)
  p95_ms = percentile(stable_walls, 0.95)
  p95_over_p50 = (
      p95_ms / p50_ms if math.isfinite(p50_ms) and p50_ms > 0.0
      else math.nan)
  decode_tokens_per_second = (
      1000.0 / median_ms if math.isfinite(median_ms) and median_ms > 0.0
      else math.nan)

  expected_linear_suffixes = {suffix: 30 for suffix in LINEAR_SUFFIXES}
  activation_passed = (
      profile.get("core_counts") == EXPECTED_CORE_COUNTS
      and profile.get("core_counts_exact") is True
      and profile.get("fused_four_fc_count") == 0
      and profile.get("fused_three_fc_count") == 50
      and profile.get("fused_shared_triple_count") == 40
      and profile.get("fused_linear_tail_triple_count") == 0
      and profile.get("existing_fused_qkv_count") == 10
      and profile.get("unfused_shared_original_count") == 0
      and profile.get("unfused_router_gate_count") == 40
      and profile.get("unfused_linear_original_count") == 120
      and profile.get("unfused_linear_original_suffix_counts") ==
          expected_linear_suffixes
      and profile.get("rms_executed_count") == 131
      and profile.get("rms_exec_types") == {
          "rms_gpu_bfyx_opt__f16": 131}
      and profile.get("old_qk_boundary_executed") == 0
      and profile.get("qk_rope_layout_executed") == 10)
  correctness_passed = (
      len(phases) == 18
      and actual_top1 == expected_top1
      and all(row.get("logits_finite") is True for row in phases))
  source_summary = result.get("source_summary") or {}
  state_contract_passed = (
      result.get("mode") == "candidate"
      and result.get("lane") == "32k"
      and result.get("prompt", {}).get("path") == str(PROMPT.resolve())
      and result.get("prompt", {}).get("token_count") == 32768
      and result.get("prompt", {}).get("token_sha256") ==
          EXPECTED_TOKEN_SHA256
      and result.get("same_infer_request") is True
      and result.get("hot_state_self_bind_skipped") is True
      and len(phases) == 18
      and phases[0].get("input_tokens") == 32768
      and phases[0].get("total_tokens") == 32768
      and all(row.get("input_tokens") == 1 for row in phases[1:])
      and phases[-1].get("total_tokens") == 32785
      and source_summary.get("fixed_cold_capacity") == 32768
      and source_summary.get("prefill_history_capacity") == 32768
      and source_summary.get("initialize_hot_states") is True
      and source_summary.get("fuse_qk_rope_layout") is True
      and source_summary.get("qk_rope_layout_rewrite_count") == 10
      and source_summary.get("custom_count_after") == 10
      and source_summary.get("linear_conv_custom_count_after") == 30
      and source_summary.get("state_count_after") == 120
      and source_summary.get("sink_count_after") == 60)
  performance_passed = (
      len(stable_walls) == 16
      and math.isfinite(median_ms) and median_ms <= cap)
  smoothness_passed = (
      math.isfinite(p95_over_p50)
      and p95_over_p50 <= smoothness_cap)
  worker_safe = (
      worker["returncode"] == 0
      and worker["timed_out"] is False
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
      check("worker_uses_exact_plugin_igc_and_alias",
            result.get("candidate_gpu_plugin_sha256") == plugin_hash
            and worker.get("igc_library_dir") == str(IGC_LIBRARY_DIR)
            and worker.get("ld_library_path_first") == str(IGC_LIBRARY_DIR)
            and worker.get("environment", {}).get(
                "IQ36_GPU_ALIAS_LINEAR_STATE_ASSIGN") == "1"),
      check("runtime_census_is_exact_at_32k", activation_passed,
            profile=profile, profile_error=profile_error),
      check("same_request_32k_state_contract_is_exact",
            state_contract_passed, source_summary=source_summary),
      check("profile_times_are_not_added_as_savings",
            profile.get("raw_profile_time_is_savings_evidence") is False),
  ]
  evidence_checks_passed = all(row["pass"] for row in evidence_checks)
  route_advanced = (
      evidence_checks_passed and correctness_passed
      and performance_passed and smoothness_passed)
  conclusive = evidence_checks_passed
  verdict = (
      "advance_clean_pr36747_rms_igc2382_to_product_source_gate"
      if route_advanced else
      "reject_clean_pr36747_rms_igc2382_at_32k"
      if conclusive else "inconclusive")
  performance = {
      "stable_sample_rule": "drop first decode JIT sample",
      "decode_walls_all_ms": (
          [float(row["wall_ms_diagnostic"]) for row in phases[1:]]
          if len(phases) == 18 else []),
      "stable_decode_walls_ms": stable_walls,
      "stable_samples": len(stable_walls),
      "median_ms": median_ms if math.isfinite(median_ms) else None,
      "mean_ms": (
          statistics.mean(stable_walls) if stable_walls else None),
      "p50_ms": p50_ms if math.isfinite(p50_ms) else None,
      "p95_ms": p95_ms if math.isfinite(p95_ms) else None,
      "p95_over_p50": (
          p95_over_p50 if math.isfinite(p95_over_p50) else None),
      "decode_tokens_per_second": (
          decode_tokens_per_second
          if math.isfinite(decode_tokens_per_second) else None),
      "registered_median_cap_ms": cap,
      "registered_decode_floor_tokens_per_second": 37.16,
      "registered_p95_over_p50_max": smoothness_cap,
      "median_cap_passed": performance_passed,
      "smoothness_passed": smoothness_passed,
      "wall_error": wall_error,
      "diagnostic_only": True,
      "speed_claim": False,
  }
  metrics = {
      "schema": SCHEMA,
      "created_at": datetime.now(timezone.utc).isoformat(),
      "workstream": WS,
      "git": git,
      "verdict": verdict,
      "evidence_checks_passed": evidence_checks_passed,
      "conclusive": conclusive,
      "route_advanced": route_advanced,
      "activation_passed": activation_passed,
      "state_contract_passed": state_contract_passed,
      "correctness_passed": correctness_passed,
      "performance_passed": performance_passed,
      "smoothness_passed": smoothness_passed,
      "gpu_workers_launched": 1,
      "candidate_workers_launched": 1,
      "workers_launched_this_invocation": (
          0 if args.reuse_completed_worker else 1),
      "reused_completed_worker": args.reuse_completed_worker,
      "stock_workers_launched": 0,
      "product_workers_launched": 0,
      "abba_blocks_launched": 0,
      "output512_workers_launched": 0,
      "decode_tokens": decode_tokens,
      "expected_top1": expected_top1,
      "actual_top1": actual_top1,
      "worker": {key: value for key, value in worker.items()
                 if key != "result"},
      "source_summary": source_summary,
      "profile": profile,
      "performance": performance,
      "evidence_checks": evidence_checks,
      "decision": {
          "advance_to_product_source_gate": route_advanced,
          "next_route": (
              "openvino_pr36747_rms_igc2382_product_integration_source_gate"
              if route_advanced else "openvino_next_source_bounded_route"),
          "reopen_condition": (
              "none for unchanged source, plugin, IGC, or repeat sampling"
              if conclusive and not route_advanced else None),
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
      "igc_libraries": observed_igc,
      "candidate_workers": 1,
      "workers_launched_this_invocation": (
          0 if args.reuse_completed_worker else 1),
      "reused_completed_worker": args.reuse_completed_worker,
      "stock_workers": 0,
      "product_workers": 0,
  })
  report = f"""# Clean RMS bundle 32k candidate diagnostic

Verdict: **{verdict}**. Evidence checks:
`{str(evidence_checks_passed).lower()}`; activation:
`{str(activation_passed).lower()}`; state contract:
`{str(state_contract_passed).lower()}`; correctness:
`{str(correctness_passed).lower()}`; median cap:
`{str(performance_passed).lower()}`; smoothness:
`{str(smoothness_passed).lower()}`.

Exactly one serial candidate-only 32k/17-step worker ran. It used the retained
seq1172 stock teacher IDs, one InferRequest, the clean seq1349 plugin and
seven-file patch, isolated IGC 2.38.2, the linear-state alias, 8-GiB
preflight, and the 4-GiB abort line. No stock, product, ABBA, or output512
worker ran. Reused completed raw worker during postprocessing:
`{str(args.reuse_completed_worker).lower()}`.

After dropping the first decode JIT sample, the 16 stable walls have median
`{median_ms:.6f} ms`, throughput `{decode_tokens_per_second:.6f} tok/s`, and
p95/p50 `{p95_over_p50:.6f}`. The registered diagnostic limits are
`{cap:.6f} ms` and `{smoothness_cap:.2f}`. This is a long-context route gate,
not paired product inference or a speed claim. OOM observed:
`{str(worker['oom_observed']).lower()}`.
"""
  (output / "report.md").write_text(report, encoding="utf-8")
  print(json.dumps({
      "artifact": display(output),
      "verdict": verdict,
      "activation_passed": activation_passed,
      "state_contract_passed": state_contract_passed,
      "correctness_passed": correctness_passed,
      "median_cap_passed": performance_passed,
      "smoothness_passed": smoothness_passed,
      "median_ms": median_ms if math.isfinite(median_ms) else None,
      "decode_tokens_per_second": (
          decode_tokens_per_second
          if math.isfinite(decode_tokens_per_second) else None),
      "p95_over_p50": (
          p95_over_p50 if math.isfinite(p95_over_p50) else None),
      "worker_returncode": worker["returncode"],
      "oom_observed": worker["oom_observed"],
  }, separators=(",", ":")), flush=True)
  return 0 if conclusive else 2


if __name__ == "__main__":
  raise SystemExit(main())
