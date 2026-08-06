#!/usr/bin/env python3
"""Instantiate the 2k product graph with the seq2189 plugin, without inference.

This validates the real product graph/plugin ABI and request allocation.  The
GPU plugin defers dynamic LM-head implementation selection until first infer,
so provider trace and embedded-kernel selection remain explicit requirements
of the next correctness precheck.  This worker emits no token.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = (
    "intel-qwen36-openvino-lm-head-parallel-block-topk-product-"
    "compile-gate-v2")
PRODUCT_TOOL = ROOT / "tools/intel-qwen36-openvino-hot-cold-product-gate.py"
BUILD_GATE = ROOT / (
    "output/openvino-lm-head-parallel-block-topk-product-build-"
    "20260731Tseq2189-clean/result.json")
BASELINE_2K = ROOT / (
    "output/openvino-2k-gated-exact-timing-abba1-"
    "20260731Tseq2183-clean/gate.json")
BASELINE_2K_MANIFEST = ROOT / (
    "output/openvino-2k-gated-exact-timing-abba1-"
    "20260731Tseq2183-clean/manifest.json")
PATCH = ROOT / "engine/openvino/iq36-lm-head-i8q1-gated-exact.patch"
PLUGIN = Path(
    "/home/intel/intel-qwen36-r0/output/"
    "openvino-90214e-l0-gpu-seq2189/bin/intel64/Release/"
    "libopenvino_intel_gpu_plugin.so")
EXPECTED_PLUGIN_SHA256 = (
    "9832adffa8bf3fd7b013c47d9d2abefa66987c5fbde350c9669fce58c697b985")
EXPECTED_PATCH_SHA256 = (
    "14408168065680e36111ea123f08c3013bc9285142b811743cba437ac2094f7c")
LAYERS = (3, 7, 11, 15, 19, 23, 27, 31, 35, 39)
BUCKET = 2048
PREFILL_HISTORY_CAPACITY = 16384
EXACT_HISTORY_CAPACITY = 17408
PREFLIGHT_GIB = 8.0
MEMORY_STOP_GIB = 4.0
EXPECTED_PROVIDER = "+".join((
    "iq36_lm_head_q8_group256_f16_sums",
    "iq36_lm_head_i8q1_rowstripe8_matvec_local_top12_f16",
    "iq36_lm_head_i8_exact_local_top12_correction_f16",
    "iq36_lm_head_output_topk8_f16",
    "iq36_lm_head_topk8_merge_f32",
    "iq36_lm_head_i8_direct_topk8_correction_f16",
    "iq36_lm_head_i8q1_gated_exact_reset_f16",
    "iq36_lm_head_i8q1_gated_exact_collect_f16",
    "iq36_lm_head_i8_gated_exact_matvec_f16",
    "iq36_lm_head_i8q1_gated_exact_output_topk8_f16",
    "iq36_lm_head_i8q1_gated_exact_topk8_merge_f32",
    "iq36_lm_head_i8_gated_exact_topk8_correction_f16",
))


def load_module(name: str, path: Path) -> Any:
  spec = importlib.util.spec_from_file_location(name, path)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import {path}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


PRODUCT = load_module("iq36_parallel_topk_product", PRODUCT_TOOL)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out-dir", type=Path, required=True)
  parser.add_argument("--timeout-s", type=int, default=1800)
  args = parser.parse_args()
  if args.timeout_s <= 0:
    parser.error("timeout must be positive")
  return args


def check(name: str, passed: bool, **details: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **details}


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    while chunk := stream.read(8 * 1024 * 1024):
      digest.update(chunk)
  return digest.hexdigest()


def trace_rows(path: Path) -> list[dict[str, Any]]:
  if not path.is_file():
    return []
  rows = []
  for line in path.read_text(encoding="utf-8").splitlines():
    if not line.strip():
      continue
    value = json.loads(line)
    if isinstance(value, dict):
      rows.append(value)
  return rows


def runtime_names(census: dict[str, Any]) -> list[str]:
  return sorted(
      str(row.get("name")) for row in census.get("attention_rows", [])
      if row.get("layer_type") == "CustomGPUPrimitive" and
      str(row.get("name", "")).startswith("iq36_hot_attention_layer"))


def main() -> int:
  args = parse_args()
  out = args.out_dir.resolve()
  if out.exists():
    raise SystemExit(f"output already exists: {out}")
  raw = out / "raw"
  raw.mkdir(parents=True, exist_ok=False)
  required_paths = (
      PRODUCT_TOOL, BUILD_GATE, BASELINE_2K, BASELINE_2K_MANIFEST,
      PATCH, PLUGIN,
      PRODUCT.CUSTOM_CONFIG, PRODUCT.MODEL_DIR, PRODUCT.MODEL_CONTRACT)
  missing = [str(path) for path in required_paths if not path.exists()]
  if missing:
    raise SystemExit("missing product compile inputs: " + ", ".join(missing))

  git = PRODUCT.BOOT.git_state(out)
  build_gate = PRODUCT.load_json(BUILD_GATE)
  baseline = PRODUCT.load_json(BASELINE_2K)
  baseline_manifest = PRODUCT.load_json(BASELINE_2K_MANIFEST)
  plugin_sha = sha256(PLUGIN)
  patch_sha = sha256(PATCH)
  model_identity = PRODUCT.BOOT.capture_model_identity(
      PRODUCT.MODEL_DIR.resolve(), PRODUCT.MODEL_CONTRACT.resolve())
  PRODUCT.write_json(out / "model-identity.json", model_identity)

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
      "case_id": "sentinel_002k_parallel_block_topk_instantiate_only",
      "compile_only": False,
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
      "host_time_profiling": 0,
      "instantiate_only": True,
      "linear_state_alias_scope": "all",
      "lm_head_device_greedy_feedback": False,
      "lm_head_i8q1": True,
      "lm_head_i8q1_gated_exact": True,
      "lm_head_i8q1_gated_q4": False,
      "lm_head_i8q1_greedy_local2": False,
      "lm_head_i8q4": False,
      "lm_head_token_only_feedback": False,
      "mode": "candidate",
      "output_tokens": 512,
      "prefill_history_capacity": PREFILL_HISTORY_CAPACITY,
      "purpose": "product_instantiate_compile_gate",
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
  monitor = worker.get("monitor") or {}
  memory_guard = worker.get("memory_guard") or {}
  trace_path = raw / "worker/lm-head-i8q1-trace.jsonl"
  trace = trace_rows(trace_path)
  selections = [row for row in trace if row.get("stage") == "selection"]
  expected_names = sorted(
      f"iq36_hot_attention_layer{layer}" for layer in LAYERS)
  minimum_available = int(
      monitor.get("system_available_min_bytes") or 0)
  stop_bytes = int(MEMORY_STOP_GIB * 1024**3)

  checks = [
      check("repository_clean_at_gate", not git["dirty"], git=git),
      check("seq2189_build_gate_admits_only_graph_compile",
            build_gate.get("required_checks_passed") is True and
            build_gate.get("verdict") ==
                "admit_parallel_block_topk_plugin_for_compile_only_graph_gate"
            and build_gate.get("graph_compile_admitted") is True and
            build_gate.get("inference_admitted") is False and
            build_gate.get("performance_claim_admitted") is False and
            build_gate.get("candidate_plugin", {}).get("sha256") ==
                EXPECTED_PLUGIN_SHA256),
      check("isolated_seq2189_plugin_and_patch_are_exact",
            plugin_sha == EXPECTED_PLUGIN_SHA256 and
            patch_sha == EXPECTED_PATCH_SHA256,
            plugin=str(PLUGIN), plugin_sha256=plugin_sha,
            patch_sha256=patch_sha),
      check("accepted_2k_lane_binds_gated_exact_product_shape",
            baseline.get("run_checks_passed") is True and
            baseline.get("product_promotion_ready") is False and
            baseline.get("speedup_claims_allowed") is False and
            baseline_manifest.get("lm_head_i8q1") is True and
            baseline_manifest.get("lm_head_i8q1_gated_exact") is True and
            baseline_manifest.get("lm_head_i8q1_gated_q4") is False and
            baseline_manifest.get("lm_head_i8q1_greedy_local2") is False and
            baseline_manifest.get("lm_head_token_only_feedback") is False,
            baseline=str(BASELINE_2K)),
      check("locked_model_identity",
            model_identity.get("required_checks_passed") is True,
            model_dir=model_identity.get("model_dir")),
      check("single_serial_instantiate_worker_completes",
            worker.get("returncode") == 0 and
            worker.get("timed_out") is False and
            worker.get("oom_observed") is False and
            worker.get("reused") is not True and
            (worker.get("worker_transient_scope") or {}).get("enabled")
                is True),
      check("one_request_compiles_without_inference_or_tokens",
            result.get("compile_only") is False and
            result.get("instantiate_only") is True and
            result.get("worker_created_infer_request") is True and
            result.get("worker_executed_inference") is False and
            "generated_token_ids" not in result and
            "state_schema_after" not in result),
      check("fresh_cache_enables_only_count25_gated_exact_lm_head",
            compiler_cache.get("lm_head_i8q1_gated_exact_env") == "1" and
            compiler_cache.get("lm_head_i8q1_gated_q4_env") is None and
            compiler_cache.get("lm_head_i8q1_greedy_local2_env") is None and
            compiler_cache.get("lm_head_i8q1_token_only_env") is None and
            compiler_cache.get("neo_cache_persistent") == "1" and
            result.get("lm_head_i8q1") is True and
            result.get("lm_head_i8q1_gated_exact") is True and
            result.get("lm_head_i8q1_gated_q4") is False and
            result.get("lm_head_i8q1_greedy_local2") is False and
            result.get("lm_head_token_only_feedback") is False),
      check("provider_selection_is_deferred_until_first_inference",
            len(selections) == 0 and not trace_path.exists(),
            selection_count=len(selections),
            expected_provider_at_precheck=EXPECTED_PROVIDER,
            note=(
                "graph compilation and InferRequest creation do not select "
                "the dynamic LM-head implementation")),
      check("candidate_plugin_and_product_carrier_are_bound",
            result.get("candidate_gpu_plugin_sha256") == plugin_sha and
            result.get("candidate_path") == "hot_cold_custom" and
            result.get("custom_composition") == "exact_phase" and
            result.get("fuse_linear_conv_state") is True and
            result.get("alias_linear_state_assign") is True and
            result.get("linear_state_alias_scope") == "all" and
            result.get("candidate_dq_realloc_fastpath") is True and
            result.get("candidate_fc_stable_prepare_fastpath") is True),
      check("2k_source_retains_exactly_ten_dual_attention_owners",
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
            source.get("custom_count_after") == len(LAYERS) and
            source.get("stock_sdpa_count_after") == 0),
      check("runtime_retains_ten_custom_attention_owners",
            census.get("hot_attention_custom_count") == len(LAYERS) and
            census.get("stock_sdpa_like_count") == 0 and
            runtime_names(census) == expected_names,
            observed_names=runtime_names(census)),
      check("full_graph_compile_duration_is_finite",
            isinstance(result.get("language_compile_ms"), (int, float)) and
            math.isfinite(
                float(result.get("language_compile_ms", math.nan))) and
            float(result.get("language_compile_ms", 0.0)) > 0.0,
            language_compile_ms=result.get("language_compile_ms")),
      check("worker_memory_telemetry_and_guard_hold",
            int(monitor.get("process_rss_peak_bytes", -1)) >= 0 and
            int(monitor.get("process_swap_peak_bytes", -1)) >= 0 and
            memory_guard.get("tripped") is False and
            minimum_available >= stop_bytes,
            monitor=monitor, stop_bytes=stop_bytes),
  ]
  required = all(row["pass"] for row in checks)
  verdict = (
      "admit_parallel_block_topk_plugin_for_one_2k_product_precheck"
      if required else
      "repair_parallel_block_topk_product_compile")
  payload = {
      "schema": SCHEMA,
      "workstream": WS,
      "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
      "git": git,
      "verdict": verdict,
      "required_checks_passed": required,
      "product_precheck_admitted": required,
      "formal_performance_admitted": False,
      "inference_workers_launched": 0,
      "checks": checks,
      "model_identity": PRODUCT.relative(out / "model-identity.json"),
      "plugin": {"path": str(PLUGIN), "sha256": plugin_sha},
      "trace": {
          "path": PRODUCT.relative(trace_path),
          "row_count": len(trace),
          "selection_count": len(selections),
          "sha256": sha256(trace_path) if trace_path.is_file() else None,
          "expected_provider_at_precheck": EXPECTED_PROVIDER,
      },
      "worker": worker,
      "next_action": {
          "route": "parallel_block_topk_2k_product_precheck",
          "requirements": [
              "run one candidate 2k output130 correctness worker",
              "require exact accepted tokens, top1, KLD, provider, and memory",
              "only then spend one short ABBA precheck",
          ],
      },
  }
  PRODUCT.write_json(out / "result.json", payload)
  PRODUCT.write_json(out / "manifest.json", {
      "schema": SCHEMA,
      "tool": PRODUCT.relative(Path(__file__)),
      "git": git,
      "inputs": {
          PRODUCT.relative(path): sha256(path)
          for path in (
              PRODUCT_TOOL, BUILD_GATE, BASELINE_2K,
              BASELINE_2K_MANIFEST, PATCH, PLUGIN)
      },
      "plugin": payload["plugin"],
      "gpu_workers": 1,
      "infer_requests": 1,
      "inference_workers": 0,
  })
  report = f"""# Parallel exact block-top8 product compile gate

Verdict: **{verdict}**. Required checks: `{str(required).lower()}`.

One isolated 2k candidate worker compiled the full graph and created one
InferRequest in `{float(result.get('language_compile_ms', 0.0)):.3f} ms`.
The seq2189 plugin and count25 configuration are bound, while dynamic LM-head
provider selection remains deferred to the first infer as expected. No
inference ran and no token was emitted.

Peak worker RSS/swap telemetry was
`{int(monitor.get('process_rss_peak_bytes', 0))}/`
`{int(monitor.get('process_swap_peak_bytes', 0))} B`; minimum available memory
was `{minimum_available} B`. No OOM or memory guard event occurred.
"""
  (out / "report.md").write_text(report, encoding="utf-8")
  print(json.dumps({
      "artifact": PRODUCT.relative(out),
      "verdict": verdict,
      "required_checks_passed": required,
      "language_compile_ms": result.get("language_compile_ms"),
      "selection_count": len(selections),
      "peak_rss_bytes": monitor.get("process_rss_peak_bytes"),
      "oom_observed": worker.get("oom_observed"),
  }, separators=(",", ":")), flush=True)
  return 0 if required else 2


if __name__ == "__main__":
  raise SystemExit(main())
