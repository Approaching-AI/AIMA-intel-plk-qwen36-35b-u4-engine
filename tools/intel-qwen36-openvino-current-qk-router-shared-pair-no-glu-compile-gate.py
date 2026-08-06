#!/usr/bin/env python3
"""Compile the Q/K plus N=1024 pair/decomposed-GLU graph without a request.

This is the first consumer of the isolated seq2217 plugin. It performs one
serial candidate-only full-graph compile under the established 8/4-GiB memory
guards and exits before creating an InferRequest or executing inference.
Executed FC and decomposed-GLU counts are deferred to the first correctness
worker.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = (
    "intel-qwen36-openvino-current-qk-router-shared-pair-no-glu-"
    "compile-gate-v1")
COMMON_TOOL = ROOT / (
    "tools/intel-qwen36-openvino-current-qk-router-shared-compile-gate.py")
PRODUCT_TOOL = ROOT / "tools/intel-qwen36-openvino-hot-cold-product-gate.py"
BUILD_GATE = ROOT / (
    "output/openvino-current-qk-router-shared-pair-no-glu-product-build-"
    "20260731Tseq2217a-clean/result.json")
PAIR_PATCH = ROOT / "engine/openvino/iq36-current-router-shared-pair.patch"
NO_GLU_PATCH = ROOT / (
    "engine/openvino/iq36-current-router-shared-pair-no-glu.patch")
SOURCE_TREE = Path(
    "/home/intel/intel-qwen36-r0/source/openvino-90214e5be05")
TRANSFORM_SOURCE = SOURCE_TREE / (
    "src/plugins/intel_gpu/src/plugin/transformations_pipeline.cpp")
FC_SOURCE = SOURCE_TREE / (
    "src/plugins/intel_gpu/src/plugin/transformations/"
    "fc_horizontal_fusion.cpp")
PLUGIN = Path(
    "/home/intel/intel-qwen36-r0/output/"
    "openvino-90214e-l0-gpu-seq2217/bin/intel64/Release/"
    "libopenvino_intel_gpu_plugin.so")
ACCEPTED_PLUGIN = Path(
    "/home/intel/intel-qwen36-r0/output/"
    "openvino-90214e-l0-gpu-seq2189/bin/intel64/Release/"
    "libopenvino_intel_gpu_plugin.so")
EXPECTED_COMMON_TOOL_SHA256 = (
    "62424b49fb21011aca87f15c65f79115be6f1cea759364bec4c99743f6e624e9")
EXPECTED_PRODUCT_TOOL_SHA256 = (
    "baa6cb5591766eb91dcb1456d0195216f10a4fafb9477fc3a357f8eb98a8c3b1")
EXPECTED_BUILD_GATE_SHA256 = (
    "7fa8c373b89ae0a14e8453dff95ab34e87893576609a13be1506baf5330a276c")
EXPECTED_PAIR_PATCH_SHA256 = (
    "092e1b3d23277cd1ab34577fc26f594efcfb0a837d72904b28b64ae01af36d3a")
EXPECTED_NO_GLU_PATCH_SHA256 = (
    "af1ead7982f2149268637c758502c7f6db81d5cdf2b0cbba905d2c47bddf524e")
EXPECTED_TRANSFORM_SOURCE_SHA256 = (
    "dd9d4c2eec7b9ba5d9bf889ac916f2b4c90e6922401524657d09b3f81892ff38")
EXPECTED_FC_SOURCE_SHA256 = (
    "1944c1af859c2ccd416a481da8d0bd336bbe39ad9a4bca0aed9ea56182b7996f")
EXPECTED_PLUGIN_SHA256 = (
    "5d12c9e2ebb239e72558c183351ff6abda7c37d4b9e52b8930f86326d28236a3")
EXPECTED_ACCEPTED_PLUGIN_SHA256 = (
    "9832adffa8bf3fd7b013c47d9d2abefa66987c5fbde350c9669fce58c697b985")
BUILD_GATE_COMMIT = "903743ebc58647e84ffd40c5bcdcbb7bc038380c"
LAYERS = (3, 7, 11, 15, 19, 23, 27, 31, 35, 39)
BUCKET = 2048
OUTPUT_TOKENS = 130
PREFILL_HISTORY_CAPACITY = 16384
EXACT_HISTORY_CAPACITY = 17408
PREFLIGHT_GIB = 8.0
MEMORY_STOP_GIB = 4.0


def load_module(name: str, path: Path) -> Any:
  spec = importlib.util.spec_from_file_location(name, path)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import {path}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


COMMON = load_module("iq36_current_pair_no_glu_compile_common", COMMON_TOOL)
PRODUCT = COMMON.PRODUCT


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out-dir", type=Path, required=True)
  parser.add_argument("--timeout-s", type=int, default=1800)
  args = parser.parse_args()
  if args.timeout_s <= 0:
    parser.error("timeout must be positive")
  return args


def main() -> int:
  args = parse_args()
  out = args.out_dir.resolve()
  if out.exists():
    raise SystemExit(f"output already exists: {out}")
  raw = out / "raw"
  raw.mkdir(parents=True, exist_ok=False)
  required_paths = (
      COMMON_TOOL, PRODUCT_TOOL, BUILD_GATE, PAIR_PATCH, NO_GLU_PATCH,
      TRANSFORM_SOURCE, FC_SOURCE, PLUGIN, ACCEPTED_PLUGIN,
      PRODUCT.CUSTOM_CONFIG, PRODUCT.MODEL_DIR, PRODUCT.MODEL_CONTRACT)
  missing = [str(path) for path in required_paths if not path.exists()]
  if missing:
    raise SystemExit(
        "missing shared-pair decomposed-GLU compile inputs: " +
        ", ".join(missing))

  git = PRODUCT.BOOT.git_state(out)
  head = COMMON.run(["git", "rev-parse", "HEAD"]).stdout.strip()
  origin_main = COMMON.run(
      ["git", "rev-parse", "origin/main"]).stdout.strip()
  ancestor = COMMON.run([
      "git", "merge-base", "--is-ancestor", BUILD_GATE_COMMIT, head])
  build_gate = PRODUCT.load_json(BUILD_GATE)
  plugin_sha = COMMON.sha256(PLUGIN)
  accepted_plugin_sha = COMMON.sha256(ACCEPTED_PLUGIN)
  model_identity = PRODUCT.BOOT.capture_model_identity(
      PRODUCT.MODEL_DIR.resolve(), PRODUCT.MODEL_CONTRACT.resolve())
  PRODUCT.write_json(out / "model-identity.json", model_identity)
  transform_text = TRANSFORM_SOURCE.read_text(encoding="utf-8")

  config = {
      "alias_linear_state_assign": True,
      "bucket": BUCKET,
      "candidate_dq_realloc_fastpath": True,
      "candidate_fc_stable_prepare_fastpath": True,
      "candidate_path": "hot_cold_custom",
      "capture_attention_history_layers": [],
      "capture_attention_history_steps": [],
      "capture_attention_layers": [],
      "capture_attention_steps": [],
      "capture_execution_census": True,
      "case_id":
          "sentinel_002k_current_qk_router_shared_pair_no_glu_compile_only",
      "compile_only": True,
      "custom_composition": "exact_phase",
      "decode_chunk256_layers": [],
      "decode_dual256_layers": [],
      "decode_f32_numerator_layers": [],
      "decode_page_sparse_layers": [],
      "decode_stock256_layers": [],
      "decode_stock_micro_layers": list(LAYERS),
      "decode_stock_partition_layers": [],
      "decode_stock_score_layers": [],
      "direct_ssm_state_assign": False,
      "exact_history_capacity": EXACT_HISTORY_CAPACITY,
      "exact_history_layers": list(LAYERS),
      "exact_phase_context_partition4": False,
      "exact_phase_dual_cohort": True,
      "fixed_fc_cohorts": [],
      "fixed_fc_manager_direct": False,
      "fixed_fc_manager_scope": "all",
      "fuse_fixed_fc": False,
      "fuse_linear_conv_state": True,
      "fuse_qk_rope_layout": True,
      "fuse_router_shared_pair": True,
      "fuse_router_shared_triple": False,
      "host_time_profiling": 0,
      "instantiate_only": False,
      "linear_state_alias_scope": "all",
      "lm_head_device_greedy_feedback": False,
      "lm_head_i8q1": True,
      "lm_head_i8q1_gated_exact": True,
      "lm_head_i8q1_gated_q4": False,
      "lm_head_i8q1_greedy_local2": False,
      "lm_head_i8q4": False,
      "lm_head_token_only_feedback": False,
      "mode": "candidate",
      "output_tokens": OUTPUT_TOKENS,
      "prefill_history_capacity": PREFILL_HISTORY_CAPACITY,
      "purpose": "current_qk_router_shared_pair_no_glu_compile_gate",
      "self_bind_hot_states": False,
      "target_layers": list(LAYERS),
      "timing_token_output": False,
  }
  worker_args = SimpleNamespace(
      abort_below_available_gib=MEMORY_STOP_GIB,
      candidate_gpu_plugin=PLUGIN,
      candidate_impls_cache_capacity=None,
      custom_config=PRODUCT.CUSTOM_CONFIG,
      device="GPU",
      min_available_gib=PREFLIGHT_GIB,
      model_dir=PRODUCT.MODEL_DIR,
      openvino_python=PRODUCT.OV_PYTHON,
      pack_gdn_state=False,
      poll_interval_s=1.0,
      prime_candidate_exact_decode_shape=False,
      resume=False,
      timeout_s=args.timeout_s,
      worker_transient_scope=True,
  )
  worker = PRODUCT.run_worker(worker_args, raw / "worker", config)
  result = worker.get("result", {})
  source = result.get("source_summary") or {}
  census = result.get("runtime_census") or {}
  compiler_cache = result.get("compiler_cache") or {}
  environment = worker.get("environment") or {}
  monitor = worker.get("monitor") or {}
  memory_guard = worker.get("memory_guard") or {}
  expected_attention_names = sorted(
      f"iq36_hot_attention_layer{layer}" for layer in LAYERS)
  expected_qk_names = sorted(
      f"iq36_qk_rope_layout_layer{layer}" for layer in LAYERS)
  qk_layers = sorted(
      int(row.get("layer")) for row in
      source.get("qk_rope_layout_rewrites", [])
      if isinstance(row, dict) and isinstance(row.get("layer"), int))
  minimum_available = int(
      monitor.get("system_available_min_bytes") or 0)
  stop_bytes = int(MEMORY_STOP_GIB * 1024**3)

  checks = [
      COMMON.check("repository_is_clean_and_pushed_at_compile_gate",
                   not git["dirty"] and head == origin_main and
                   ancestor.returncode == 0,
                   git=git, head=head, origin_main=origin_main,
                   build_commit_is_ancestor=ancestor.returncode == 0),
      COMMON.check("seq2217a_admits_only_candidate_graph_compile",
                   COMMON.sha256(BUILD_GATE) ==
                       EXPECTED_BUILD_GATE_SHA256 and
                   build_gate.get("required_checks_passed") is True and
                   build_gate.get("verdict") ==
                       "admit_pair_decomposed_glu_plugin_for_compile_gate" and
                   build_gate.get(
                       "candidate_only_graph_compile_admitted") is True and
                   build_gate.get("infer_request_admitted") is False and
                   build_gate.get("inference_admitted") is False and
                   build_gate.get("candidate_plugin", {}).get("sha256") ==
                       EXPECTED_PLUGIN_SHA256 and
                   build_gate.get("repository", {}).get("head") ==
                       BUILD_GATE_COMMIT,
                   build_gate_sha256=COMMON.sha256(BUILD_GATE),
                   build_gate_verdict=build_gate.get("verdict")),
      COMMON.check("isolated_candidate_and_accepted_plugins_are_exact",
                   plugin_sha == EXPECTED_PLUGIN_SHA256 and
                   accepted_plugin_sha == EXPECTED_ACCEPTED_PLUGIN_SHA256 and
                   PLUGIN.resolve() != ACCEPTED_PLUGIN.resolve(),
                   candidate_plugin=str(PLUGIN),
                   candidate_plugin_sha256=plugin_sha,
                   accepted_plugin_sha256=accepted_plugin_sha),
      COMMON.check("cumulative_pair_and_no_glu_sources_are_exact",
                   COMMON.sha256(COMMON_TOOL) ==
                       EXPECTED_COMMON_TOOL_SHA256 and
                   COMMON.sha256(PRODUCT_TOOL) ==
                       EXPECTED_PRODUCT_TOOL_SHA256 and
                   COMMON.sha256(PAIR_PATCH) ==
                       EXPECTED_PAIR_PATCH_SHA256 and
                   COMMON.sha256(NO_GLU_PATCH) ==
                       EXPECTED_NO_GLU_PATCH_SHA256 and
                   COMMON.sha256(TRANSFORM_SOURCE) ==
                       EXPECTED_TRANSFORM_SOURCE_SHA256 and
                   COMMON.sha256(FC_SOURCE) ==
                       EXPECTED_FC_SOURCE_SHA256 and
                   transform_text.count(
                       'std::getenv("IQ36_ROUTER_SHARED_PAIR")') == 1 and
                   transform_text.count(
                       "manager.register_pass<ov::pass::GLUFusion>();") == 1,
                   transform_source_sha256=COMMON.sha256(TRANSFORM_SOURCE),
                   fc_source_sha256=COMMON.sha256(FC_SOURCE)),
      COMMON.check("locked_model_identity",
                   model_identity.get("required_checks_passed") is True,
                   model_dir=model_identity.get("model_dir")),
      COMMON.check("one_serial_transient_compile_worker_completes",
                   worker.get("returncode") == 0 and
                   worker.get("timed_out") is False and
                   worker.get("oom_observed") is False and
                   worker.get("reused") is not True and
                   (worker.get("worker_transient_scope") or {}).get(
                       "enabled") is True),
      COMMON.check("compile_only_worker_creates_no_request_or_inference",
                   result.get("compile_only") is True and
                   result.get("instantiate_only") is False and
                   result.get("worker_created_infer_request") is False and
                   result.get("worker_executed_inference") is False and
                   "generated_token_ids" not in result and
                   "state_schema_after" not in result),
      COMMON.check("candidate_pair_flags_are_exact_and_mutually_exclusive",
                   result.get("candidate_gpu_plugin_sha256") == plugin_sha and
                   result.get("candidate_path") == "hot_cold_custom" and
                   result.get("custom_composition") == "exact_phase" and
                   config.get("fuse_qk_rope_layout") is True and
                   config.get("fuse_router_shared_pair") is True and
                   config.get("fuse_router_shared_triple") is False and
                   config.get("fuse_fixed_fc") is False and
                   config.get("fixed_fc_manager_direct") is False and
                   source.get("fuse_qk_rope_layout") is True and
                   environment.get("IQ36_ROUTER_SHARED_PAIR") == "1" and
                   "IQ36_ROUTER_SHARED_TRIPLE" not in environment and
                   "IQ36_FIXED_FC_MANAGER_SCOPE" not in environment,
                   note=(
                       "the exact transformed source makes this environment "
                       "skip GLUFusion; compile-only returns before flag echo")),
      COMMON.check("source_graph_has_ten_qk_and_dual_attention_owners",
                   source.get("target_layers") == list(LAYERS) and
                   source.get("decode_stock_micro_layers") == list(LAYERS) and
                   source.get("exact_phase_decode") is True and
                   source.get("exact_phase_dual_cohort") is True and
                   source.get("exact_phase_context_partition4") is False and
                   source.get("exact_history_layers") == list(LAYERS) and
                   source.get("exact_history_capacity") ==
                       EXACT_HISTORY_CAPACITY and
                   source.get("prefill_history_capacity") ==
                       PREFILL_HISTORY_CAPACITY and
                   source.get("fixed_cold_capacity") == BUCKET and
                   source.get("custom_count_after") == len(LAYERS) and
                   source.get("stock_sdpa_count_after") == 0 and
                   source.get("qk_rope_layout_rewrite_count") ==
                       len(LAYERS) and
                   qk_layers == list(LAYERS),
                   qk_layers=qk_layers),
      COMMON.check("compiled_runtime_retains_exact_custom_owner_census",
                   census.get("qk_rope_layout_custom_count") == len(LAYERS) and
                   census.get("hot_attention_custom_count") == len(LAYERS) and
                   census.get("linear_conv_custom_count") == 30 and
                   census.get("fixed_fc_custom_count") == 0 and
                   census.get("stock_sdpa_like_count") == 0 and
                   COMMON.runtime_names(
                       census, "iq36_hot_attention_layer") ==
                       expected_attention_names and
                   COMMON.runtime_names(
                       census, "iq36_qk_rope_layout_layer") ==
                       expected_qk_names,
                   observed_attention_names=COMMON.runtime_names(
                       census, "iq36_hot_attention_layer"),
                   observed_qk_names=COMMON.runtime_names(
                       census, "iq36_qk_rope_layout_layer")),
      COMMON.check("fresh_cache_binds_count25_full_logit_provider",
                   compiler_cache.get("lm_head_i8q1_gated_exact_env") == "1" and
                   compiler_cache.get("lm_head_i8q1_gated_q4_env") is None and
                   compiler_cache.get(
                       "lm_head_i8q1_greedy_local2_env") is None and
                   compiler_cache.get("lm_head_i8q1_token_only_env") is None and
                   compiler_cache.get("neo_cache_persistent") == "1" and
                   result.get("lm_head_i8q1") is True and
                   result.get("lm_head_i8q1_gated_exact") is True and
                   result.get("lm_head_i8q1_gated_q4") is False and
                   result.get("lm_head_i8q1_greedy_local2") is False and
                   result.get("lm_head_token_only_feedback") is False),
      COMMON.check("full_graph_compile_duration_is_finite",
                   isinstance(
                       result.get("language_compile_ms"), (int, float)) and
                   math.isfinite(float(
                       result.get("language_compile_ms", math.nan))) and
                   float(result.get("language_compile_ms", 0.0)) > 0.0,
                   language_compile_ms=result.get("language_compile_ms")),
      COMMON.check("worker_memory_guard_holds_without_oom",
                   int(monitor.get("process_rss_peak_bytes", -1)) >= 0 and
                   int(monitor.get("process_swap_peak_bytes", -1)) >= 0 and
                   memory_guard.get("tripped") is False and
                   minimum_available >= stop_bytes and
                   worker.get("oom_observed") is False,
                   monitor=monitor, stop_bytes=stop_bytes),
      COMMON.check("decomposed_glu_execution_census_is_deferred", True,
                   expected_next_execution_counts={
                       "FullyConnectedCompressed": 331,
                       "GLU": 0,
                       "Crop_delta_vs_qk_only": 0,
                       "Multiply_delta_vs_qk_only": 40,
                       "VariadicSplit_delta_vs_qk_only": 40,
                       "IQ36QKRopeLayout": 10,
                       "IQ36ExactPhaseDualCohortHotAttentionGQA": 10,
                       "shared_pairs": 40,
                       "independent_scalar_shared_gates": 40,
                       "independent_router_gates": 40,
                   },
                   note=(
                       "compile-only has no executed FC/type census; the next "
                       "single inference worker must prove exact topology")),
  ]
  required = all(row["pass"] for row in checks)
  verdict = (
      "admit_one_pair_decomposed_glu_output130_correctness_worker"
      if required else
      "repair_pair_decomposed_glu_candidate_compile")
  payload = {
      "schema": SCHEMA,
      "workstream": WS,
      "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
      "git": git,
      "verdict": verdict,
      "required_checks_passed": required,
      "candidate_output130_correctness_worker_admitted": required,
      "infer_requests_created": 0,
      "inference_workers_launched": 0,
      "performance_worker_admitted": False,
      "formal_performance_admitted": False,
      "checks": checks,
      "model_identity": PRODUCT.relative(out / "model-identity.json"),
      "plugin": {"path": str(PLUGIN), "sha256": plugin_sha},
      "worker": worker,
      "next_action": {
          "route": "pair_decomposed_glu_output130_correctness",
          "requirements": [
              "push an exact candidate-only output130 correctness gate",
              "run exactly one InferRequest worker under a fresh scope",
              "require 130 accepted logits, exact tokens, and product KLD",
              "require FC331, GLU0, and plus-40 decomposed split topology",
              "only a complete correctness pass may fund point timing",
          ],
      },
  }
  PRODUCT.write_json(out / "result.json", payload)
  PRODUCT.write_json(out / "manifest.json", {
      "schema": SCHEMA,
      "tool": PRODUCT.relative(Path(__file__)),
      "git": git,
      "inputs": {
          PRODUCT.relative(path): COMMON.sha256(path)
          for path in (
              COMMON_TOOL, PRODUCT_TOOL, BUILD_GATE, PAIR_PATCH, NO_GLU_PATCH,
              TRANSFORM_SOURCE, FC_SOURCE, PLUGIN, ACCEPTED_PLUGIN)
      },
      "plugin": payload["plugin"],
      "gpu_workers": 1,
      "graph_compiles": 1,
      "infer_requests": 0,
      "inference_workers": 0,
  })
  report = f"""# Q/K plus N=1024 pair/decomposed-GLU compile-only gate

Verdict: **{verdict}**. Required checks:
`{str(required).lower()}`.

One isolated candidate worker compiled the complete 2k/output130 graph in
`{float(result.get('language_compile_ms', 0.0)):.3f} ms`. The runtime model
retains exactly ten Q/K custom owners, ten dual-attention owners, and 30
linear custom owners. Exact pair/decomposed-GLU execution census is deferred
to the first inference.

Peak worker RSS/swap was
`{int(monitor.get('process_rss_peak_bytes', 0))}/`
`{int(monitor.get('process_swap_peak_bytes', 0))} B`; minimum available memory
was `{minimum_available} B`. No InferRequest, inference, token, OOM, or guard
event occurred.
"""
  (out / "report.md").write_text(report, encoding="utf-8")
  print(json.dumps({
      "artifact": PRODUCT.relative(out),
      "verdict": verdict,
      "required_checks_passed": required,
      "language_compile_ms": result.get("language_compile_ms"),
      "qk_owners": census.get("qk_rope_layout_custom_count"),
      "attention_owners": census.get("hot_attention_custom_count"),
      "peak_rss_bytes": monitor.get("process_rss_peak_bytes"),
      "minimum_available_bytes": minimum_available,
      "oom_observed": worker.get("oom_observed"),
  }, separators=(",", ":")), flush=True)
  return 0 if required else 2


if __name__ == "__main__":
  raise SystemExit(main())
