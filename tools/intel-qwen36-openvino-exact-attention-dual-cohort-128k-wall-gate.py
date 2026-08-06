#!/usr/bin/env python3
"""Run one timing-only 128k/output512 ABBA wall probe for dual cohort."""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import math
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = (
    "intel-qwen36-openvino-exact-attention-dual-cohort-128k-"
    "wall-gate-v1")
PRODUCT_TOOL = ROOT / "tools/intel-qwen36-openvino-hot-cold-product-gate.py"
GRAPH_TOOL = ROOT / "tools/intel_qwen36_openvino_hot_cold_attention.py"
KERNEL = ROOT / "engine/openvino/custom/iq36_stock_micro_attention_oracle.cl"
CONFIG = ROOT / "engine/openvino/custom/iq36_hot_attention_gqa.xml"
SHORT_GATE = ROOT / (
    "output/openvino-exact-attention-dual-cohort-128k-short-correctness-"
    "20260723Tseq2138-clean/gate.json")
ACCEPTED_CORRECTNESS = ROOT / (
    "output/openvino-lm-head-gated-exact-count25-"
    "20260723Tseq2121-all10-128k-o512-abba1/correctness.json")
REFERENCE_STOCK = ROOT / (
    "output/openvino-lm-head-gated-exact-count25-"
    "20260723Tseq2121-all10-128k-o512-abba1/raw/sentinel_128k/"
    "correctness/stock/worker-result.json")
PROMPT = ROOT / (
    "output/r0-oracle-prompt-materialization-20260626T082201Z/"
    "prompts/sentinel_128k.txt")
PLUGIN = Path(
    "/home/intel/intel-qwen36-r0/output/"
    "openvino-90214e-l0-gpu-seq2119/bin/intel64/Release/"
    "libopenvino_intel_gpu_plugin.so")
EXPECTED_PLUGIN_SHA256 = (
    "01c04ced415a7b7a5e5bda77a995b2b97b68eb3d9f2c5f3396844d042ddda269")
EXPECTED_PROMPT_SHA256 = (
    "5233da1267b2d9af10f12fe920274a4c2c31f1b872351f6b0fc88491e50d84f4")
EXPECTED_TOKEN_SHA256 = (
    "6e9b963ebb86bdf0d4feeb7c252d4eed583eb7a395debf4533197e3552b20c17")
LAYERS = (3, 7, 11, 15, 19, 23, 27, 31, 35, 39)
BUCKET = 131072
EXACT_HISTORY_CAPACITY = 132096
OUTPUT_TOKENS = 512
PREFLIGHT_GIB = 8.0
MEMORY_STOP_GIB = 4.0
TARGET_RATIO = 1.10


def load_module(name: str, path: Path) -> ModuleType:
  spec = importlib.util.spec_from_file_location(name, path)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import {path}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


PRODUCT = load_module("iq36_dual_cohort_wall_product", PRODUCT_TOOL)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out-dir", type=Path, required=True)
  parser.add_argument("--timeout-s", type=int, default=3600)
  parser.add_argument("--resume", action="store_true")
  args = parser.parse_args()
  if args.timeout_s <= 0:
    parser.error("--timeout-s must be positive")
  return args


def check(name: str, passed: bool, **details: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **details}


def candidate_timing_isolation(result: dict[str, Any]) -> bool:
  trace = result.get("lm_head_i8q1_trace") or {}
  rows = trace.get("selection_rows") or []
  cache = result.get("compiler_cache") or {}
  return bool(
      result.get("lm_head_i8q1") is True
      and result.get("lm_head_i8q1_greedy_local2") is True
      and result.get("lm_head_i8q1_gated_exact") is False
      and result.get("lm_head_i8q1_gated_q4") is False
      and result.get("lm_head_token_only_feedback") is True
      and result.get("timing_token_output") is True
      and cache.get("lm_head_i8q1_greedy_local2_env") == "1"
      and cache.get("lm_head_i8q1_token_only_env") == "1"
      and bool(rows)
      and all(
          row.get("topk") == 2
          and row.get("correction_rows") == 1940
          and row.get("token_only") is True
          and row.get("compact_rows") == 2910
          and "local_top3_compact" in str(row.get("provider", ""))
          and "compact_top3_merge_top8" in str(row.get("provider", ""))
          and "direct_compact_top8_correction" in
              str(row.get("provider", ""))
          and "top8_encode_token" in str(row.get("provider", ""))
          for row in rows))


def exact_dual_runtime(result: dict[str, Any]) -> bool:
  source = result.get("source_summary") or {}
  runtime = result.get("runtime_census") or {}
  return bool(
      source.get("target_layers") == list(LAYERS)
      and source.get("decode_stock_micro_layers") == list(LAYERS)
      and source.get("exact_history_layers") == list(LAYERS)
      and source.get("exact_history_capacity") == EXACT_HISTORY_CAPACITY
      and source.get("exact_phase_decode") is True
      and source.get("exact_phase_dual_cohort") is True
      and source.get("exact_phase_context_partition4") is False
      and source.get("decode_page_sparse_layers") == []
      and source.get("custom_count_after") == len(LAYERS)
      and source.get("stock_sdpa_count_after") == 0
      and runtime.get("hot_attention_custom_count") == len(LAYERS)
      and runtime.get("stock_sdpa_like_count") == 0)


def stock_timing_isolation(result: dict[str, Any]) -> bool:
  return bool(
      result.get("mode") == "stock"
      and result.get("candidate_path") == "stock_sdpa"
      and result.get("candidate_gpu_plugin_sha256") is None
      and result.get("timing_token_output") is False
      and result.get("lm_head_i8q1") is False
      and result.get("lm_head_i8q1_greedy_local2") is False
      and result.get("lm_head_token_only_feedback") is False)


def memory_summary(run: dict[str, Any]) -> dict[str, Any]:
  monitor = run.get("monitor") or {}
  return {
      "elapsed_seconds": run.get("elapsed_seconds"),
      "memory_guard_tripped": (run.get("memory_guard") or {}).get("tripped"),
      "oom_observed": run.get("oom_observed"),
      "process_rss_peak_bytes": monitor.get("process_rss_peak_bytes"),
      "process_swap_peak_bytes": monitor.get("process_swap_peak_bytes"),
      "system_available_min_bytes": monitor.get(
          "system_available_min_bytes"),
  }


def main() -> int:
  args = parse_args()
  out = args.out_dir.resolve()
  if out.exists() and not args.resume:
    raise SystemExit(f"output already exists: {out}")
  raw = out / "raw"
  raw.mkdir(parents=True, exist_ok=args.resume)

  required_paths = (
      PRODUCT_TOOL, GRAPH_TOOL, KERNEL, CONFIG, SHORT_GATE,
      ACCEPTED_CORRECTNESS, REFERENCE_STOCK, PROMPT, PLUGIN,
      PRODUCT.ACCEPTANCE, PRODUCT.MODEL_CONTRACT, PRODUCT.MODEL_DIR)
  missing = [str(path) for path in required_paths if not path.exists()]
  if missing:
    raise SystemExit("missing dual-cohort wall inputs: " + ", ".join(missing))

  git = PRODUCT.BOOT.git_state(out)
  short_gate = PRODUCT.load_json(SHORT_GATE)
  accepted_correctness = PRODUCT.load_json(ACCEPTED_CORRECTNESS)
  reference_stock = PRODUCT.load_json(REFERENCE_STOCK)
  reference_ids = reference_stock.get("generated_token_ids") or []
  plugin_sha = PRODUCT.sha256_file(PLUGIN)
  prompt_sha = PRODUCT.sha256_file(PROMPT)
  model_identity = PRODUCT.BOOT.capture_model_identity(
      PRODUCT.MODEL_DIR.resolve(), PRODUCT.MODEL_CONTRACT.resolve())
  acceptance = PRODUCT.load_json(PRODUCT.ACCEPTANCE)
  PRODUCT.write_json(out / "model-identity.json", model_identity)
  PRODUCT.write_json(out / "host.json", PRODUCT.BOOT.capture_host())

  short_sources = short_gate.get("sources") or {}
  correctness_bound = bool(
      short_gate.get("required_checks_passed") is True
      and short_gate.get("short_correctness_admitted") is True
      and short_gate.get("timing_worker_admitted") is True
      and short_gate.get("verdict") ==
          "admit_exact_attention_dual_cohort_paired_wall_probe"
      and short_gate.get("git", {}).get("dirty") is False
      and short_sources.get(PRODUCT.relative(PRODUCT_TOOL)) ==
          PRODUCT.sha256_file(PRODUCT_TOOL)
      and short_sources.get(PRODUCT.relative(GRAPH_TOOL)) ==
          PRODUCT.sha256_file(GRAPH_TOOL)
      and short_sources.get(PRODUCT.relative(KERNEL)) ==
          PRODUCT.sha256_file(KERNEL)
      and short_sources.get(PRODUCT.relative(CONFIG)) ==
          PRODUCT.sha256_file(CONFIG)
      and accepted_correctness.get("required_checks_passed") is True
      and accepted_correctness.get("cases", [{}])[0].get(
          "required_checks_passed") is True
      and accepted_correctness.get("cases", [{}])[0].get("top1_rate") == 1.0
      and len(reference_ids) == OUTPUT_TOKENS
      and reference_stock.get("generated_token_ids_sha256") ==
          EXPECTED_TOKEN_SHA256
      and reference_stock.get("sentinel_pass") is True)

  worker_args = SimpleNamespace(
      abort_below_available_gib=MEMORY_STOP_GIB,
      candidate_gpu_plugin=PLUGIN,
      candidate_impls_cache_capacity=None,
      custom_config=CONFIG,
      device="GPU",
      min_available_gib=PREFLIGHT_GIB,
      model_dir=PRODUCT.MODEL_DIR,
      openvino_python=PRODUCT.OV_PYTHON,
      pack_gdn_state=False,
      poll_interval_s=1.0,
      prime_candidate_exact_decode_shape=False,
      resume=args.resume,
      timeout_s=args.timeout_s,
  )
  adaptive_defaults = {
      "adaptive_attention_exact_layers": [],
      "adaptive_attention_high_topk": 256,
      "adaptive_attention_high_topk_layers": [],
      "adaptive_attention_key_exact_layers": [],
      "adaptive_attention_key_residual1_layers": [],
      "adaptive_attention_packed_kv_layers": [],
      "adaptive_attention_packed_kv_variant": None,
      "adaptive_attention_topk": 512,
      "adaptive_attention_v16_layers": [],
      "adaptive_attention_value_residual1_layers": [],
  }
  common = {
      **adaptive_defaults,
      "alias_linear_state_assign": True,
      "bucket": BUCKET,
      "candidate_dq_realloc_fastpath": True,
      "candidate_fc_stable_prepare_fastpath": True,
      "candidate_impls_cache_capacity": None,
      "capture_attention_history_layers": [],
      "capture_attention_history_steps": [],
      "capture_attention_layers": [],
      "capture_attention_steps": [],
      "capture_execution_census": False,
      "capture_lm_head_hidden": False,
      "capture_prefill_profiles": False,
      "case_id": "sentinel_128k",
      "checkpoint_steps": [],
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
      "expected_answer": "indigo-trace-131072",
      "fixed_fc_cohorts": [],
      "fixed_fc_manager_direct": False,
      "fixed_fc_manager_scope": "all",
      "fuse_fixed_fc": False,
      "fuse_linear_conv_state": True,
      "host_time_profiling": 0,
      "linear_state_alias_scope": "all",
      "lm_head_device_greedy_feedback": False,
      "lm_head_i8q1": True,
      "lm_head_i8q1_gated_exact": True,
      "lm_head_i8q1_gated_q4": False,
      "lm_head_i8q1_greedy_local2": True,
      "lm_head_i8q4": False,
      "lm_head_token_only_feedback": True,
      "output_tokens": OUTPUT_TOKENS,
      "pack_gdn_state": False,
      "prefill_chunk_tokens": PRODUCT.FROZEN_CHUNK_TOKENS,
      "prefill_history_capacity": BUCKET,
      "prime_candidate_exact_decode_shape": False,
      "prompt": str(PROMPT.resolve()),
      "self_bind_hot_states": False,
      "target_layers": list(LAYERS),
      "warmup": True,
  }
  schedule = (
      ("stock-a1", "stock", "stock_sdpa"),
      ("candidate-b1", "candidate", "hot_cold_custom"),
      ("candidate-b2", "candidate", "hot_cold_custom"),
      ("stock-a2", "stock", "stock_sdpa"),
  )
  runs: dict[str, dict[str, Any]] = {}
  stopped_reason = None
  for label, mode, selected_path in schedule:
    run = PRODUCT.run_worker(worker_args, raw / label, {
        **common,
        "candidate_path": selected_path,
        "capture_logits": False,
        "mode": mode,
        "purpose": "paired_product_timing",
        "reference_result": str(REFERENCE_STOCK.resolve()),
        "timing_token_output": mode == "candidate",
    })
    run["worker"] = PRODUCT.relative(raw / label)
    runs[label] = run
    result = run.get("result") or {}
    if (run.get("returncode") != 0 or run.get("timed_out") or
        run.get("oom_observed") or
        (run.get("memory_guard") or {}).get("tripped")):
      stopped_reason = f"{label} failed"
      break
    if result.get("generated_token_ids") != reference_ids:
      stopped_reason = f"{label} token divergence"
      break
    if mode == "candidate" and (
        not exact_dual_runtime(result) or
        not candidate_timing_isolation(result)):
      stopped_reason = f"{label} candidate isolation mismatch"
      break
    if mode == "stock" and not stock_timing_isolation(result):
      stopped_reason = f"{label} stock isolation mismatch"
      break

  all_runs = [runs[label] for label, _, _ in schedule if label in runs]
  complete = len(runs) == len(schedule) and stopped_reason is None
  case = {
      "bucket": BUCKET,
      "candidate_path": "hot_cold_custom",
      "case_id": "sentinel_128k",
  }
  block = PRODUCT.block_summary(0, runs) if complete else None
  performance = (
      PRODUCT.performance_for_case(case, [block], acceptance)
      if block is not None else {})
  memory = (
      PRODUCT.memory_rollup(all_runs, acceptance)
      if all_runs else {"checks": [], "required_checks_passed": False})
  smoothness = (
      PRODUCT.smoothness_rollup([performance], all_runs, acceptance)
      if performance else {
          "checks": [], "required_checks_passed": False})
  phases = (block or {}).get("phases") or {}
  point_ratio_pass = bool(
      phases
      and all(
          math.isfinite(float(phases[phase]["ratio"]))
          and float(phases[phase]["ratio"]) >= TARGET_RATIO
          for phase in ("prefill_tokens_s", "decode_tokens_s", "total_rate")))
  absolute_pass = bool(
      performance.get("absolute_floors", {}).get("pass") is True)
  checks = [
      check("repository_clean_at_gate", not git["dirty"], git=git),
      check("optimized_short_correctness_and_512_token_oracle_are_bound",
            correctness_bound),
      check("accepted_plugin_prompt_and_model_are_exact",
            plugin_sha == EXPECTED_PLUGIN_SHA256
            and prompt_sha == EXPECTED_PROMPT_SHA256
            and model_identity.get("required_checks_passed") is True,
            plugin_sha256=plugin_sha, prompt_sha256=prompt_sha),
      check("single_abba_block_executes_strictly_serially", complete,
            schedule=[row[0] for row in schedule],
            stopped_reason=stopped_reason),
      check("all_timing_tokens_match_accepted_stock",
            complete and all(
                (run.get("result") or {}).get("generated_token_ids_sha256") ==
                    EXPECTED_TOKEN_SHA256
                for run in all_runs)),
      check("candidate_uses_exact_dual_owner_and_compact_timing_head",
            complete and all(
                exact_dual_runtime(runs[label]["result"])
                and candidate_timing_isolation(runs[label]["result"])
                for label in ("candidate-b1", "candidate-b2"))),
      check("stock_denominator_is_untouched",
            complete and all(
                stock_timing_isolation(runs[label]["result"])
                for label in ("stock-a1", "stock-a2"))),
      check("worker_memory_and_smoothness_checks_pass",
            memory.get("required_checks_passed") is True
            and smoothness.get("required_checks_passed") is True,
            memory_checks=memory.get("checks"),
            smoothness_checks=smoothness.get("checks")),
      check("diagnostic_point_ratios_clear_1p10", point_ratio_pass,
            phases=phases, target_ratio=TARGET_RATIO),
      check("candidate_absolute_prefill_and_decode_floors_pass",
            absolute_pass,
            absolute_floors=performance.get("absolute_floors")),
  ]
  required = all(row["pass"] for row in checks)
  admitted = required and not args.resume
  verdict = (
      "admit_exact_attention_dual_cohort_128k_multiblock_confirmation"
      if admitted else
      "development_exact_attention_dual_cohort_128k_wall"
      if required else
      "reject_exact_attention_dual_cohort_128k_product_wall")
  payload = {
      "schema_version": SCHEMA,
      "workstream": WS,
      "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
      "git": git,
      "verdict": verdict,
      "required_checks_passed": required,
      "multiblock_confirmation_admitted": admitted,
      "product_promotion_ready": False,
      "speedup_claims_allowed": False,
      "checks": checks,
      "correctness_binding": PRODUCT.relative(SHORT_GATE),
      "reference_stock": PRODUCT.relative(REFERENCE_STOCK),
      "plugin": {"path": str(PLUGIN), "sha256": plugin_sha},
      "prompt": {"path": PRODUCT.relative(PROMPT), "sha256": prompt_sha},
      "schedule": [row[0] for row in schedule],
      "workers": runs,
      "performance": performance,
      "memory": memory,
      "smoothness": smoothness,
      "next_route": (
          "openvino_exact_attention_dual_cohort_128k_multiblock_confirmation"
          if required else
          "profile_or_switch_exact_attention_dual_cohort_product_route"),
      "sources": {
          PRODUCT.relative(path): PRODUCT.sha256_file(path)
          for path in (
              PRODUCT_TOOL, GRAPH_TOOL, KERNEL, CONFIG, SHORT_GATE,
              ACCEPTED_CORRECTNESS, REFERENCE_STOCK)
      },
  }
  PRODUCT.write_json(out / "gate.json", payload)
  PRODUCT.write_json(out / "performance.json", performance)
  PRODUCT.write_json(out / "memory.json", memory)
  PRODUCT.write_json(out / "smoothness.json", smoothness)
  PRODUCT.write_json(out / "manifest.json", {
      "schema_version": SCHEMA,
      "workstream": WS,
      "git_commit": git["commit"],
      "verdict": verdict,
      "files": [
          "gate.json", "performance.json", "memory.json",
          "smoothness.json", "summary.md",
      ],
  })
  (out / "summary.md").write_text(
      "\n".join([
          "# Exact-attention dual-cohort 128k wall gate",
          "",
          f"- verdict: `{verdict}`",
          f"- required checks: `{'pass' if required else 'fail'}`",
          f"- candidate prefill/decode: "
          f"`{performance.get('absolute_floors', {}).get('prefill_median')} / "
          f"{performance.get('absolute_floors', {}).get('decode_median')} tok/s`",
          f"- prefill/decode/total point ratios: "
          f"`{phases.get('prefill_tokens_s', {}).get('ratio')} / "
          f"{phases.get('decode_tokens_s', {}).get('ratio')} / "
          f"{phases.get('total_rate', {}).get('ratio')}`",
          "- paired blocks: `1` (diagnostic; no speedup claim)",
          "",
      ]), encoding="utf-8")
  print(json.dumps({
      "output": PRODUCT.relative(out),
      "verdict": verdict,
      "required_checks_passed": required,
      "candidate_prefill_tokens_s":
          performance.get("absolute_floors", {}).get("prefill_median"),
      "candidate_decode_tokens_s":
          performance.get("absolute_floors", {}).get("decode_median"),
      "decode_point_ratio":
          phases.get("decode_tokens_s", {}).get("ratio"),
      "multiblock_confirmation_admitted": admitted,
      "speedup_claims_allowed": False,
  }, sort_keys=True), flush=True)
  return 0 if required else 2


if __name__ == "__main__":
  raise SystemExit(main())
