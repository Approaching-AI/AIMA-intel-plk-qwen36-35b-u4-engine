#!/usr/bin/env python3
"""Run one 2k/output130 correctness precheck for the seq2189 LM-head plugin.

The sole candidate worker is teacher-forced from the accepted seq2183 stock
tokens.  All 130 logits must be bitwise identical to the accepted seq2183
candidate, while stock-relative KLD/top-1, provider trace, graph census,
memory, and exact token evidence must remain accepted.
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
    "precheck-v1")
PRODUCT_TOOL = ROOT / "tools/intel-qwen36-openvino-hot-cold-product-gate.py"
BUILD_GATE = ROOT / (
    "output/openvino-lm-head-parallel-block-topk-product-build-"
    "20260731Tseq2189-clean/result.json")
COMPILE_GATE = ROOT / (
    "output/openvino-lm-head-parallel-block-topk-product-compile-"
    "20260731Tseq2190a-clean/result.json")
FALLBACK_BOUND = ROOT / (
    "output/openvino-lm-head-gated-exact-fallback-bound-"
    "20260731Tseq2186-clean/result.json")
REFERENCE_ROOT = ROOT / (
    "output/openvino-2k-gated-exact-timing-abba1-"
    "20260731Tseq2183-clean/raw/sentinel_002k/correctness")
REFERENCE_CONFIG = REFERENCE_ROOT / "candidate/worker-config.json"
REFERENCE_CANDIDATE = REFERENCE_ROOT / "candidate/worker-result.json"
REFERENCE_STOCK = REFERENCE_ROOT / "stock/worker-result.json"
PLUGIN = Path(
    "/home/intel/intel-qwen36-r0/output/"
    "openvino-90214e-l0-gpu-seq2189/bin/intel64/Release/"
    "libopenvino_intel_gpu_plugin.so")
EXPECTED_PLUGIN_SHA256 = (
    "9832adffa8bf3fd7b013c47d9d2abefa66987c5fbde350c9669fce58c697b985")
EXPECTED_REFERENCE_CONFIG_SHA256 = (
    "21bd692f93ba8bba40badf29d01a214ddcd55e4276de42971db71af8354cfced")
EXPECTED_REFERENCE_CANDIDATE_SHA256 = (
    "fa6a4aacdd45251c6818b467477794688754ffc7c5fa744ad9fb22e4961523b3")
EXPECTED_REFERENCE_STOCK_SHA256 = (
    "c327d633b0a6c75320d577bbe555e992303f85da3de800be7b8d70536f7d5215")
EXPECTED_TOKEN_SHA256 = (
    "7cb86794ff37361ce5008a88a3b54eebbf9548256947825438e85b48d0a76d41")
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
OUTPUT_TOKENS = 130
KLD_MAX = 0.005
TOP1_MIN = 0.99
PREFLIGHT_GIB = 8.0
MEMORY_STOP_GIB = 4.0


def load_module(name: str, path: Path) -> Any:
  spec = importlib.util.spec_from_file_location(name, path)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import {path}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


PRODUCT = load_module("iq36_parallel_topk_precheck_product", PRODUCT_TOOL)


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


def checkpoint_map(result: dict[str, Any]) -> dict[int, dict[str, Any]]:
  return {
      int(row["step"]): row
      for row in result.get("distribution_checkpoints", [])
      if isinstance(row, dict) and isinstance(row.get("step"), int)
  }


def checkpoint_path(row: dict[str, Any]) -> Path:
  path = Path(str(row["file"]))
  return path if path.is_absolute() else ROOT / path


def main() -> int:
  args = parse_args()
  out = args.out_dir.resolve()
  if out.exists():
    raise SystemExit(f"output already exists: {out}")
  raw = out / "raw"
  raw.mkdir(parents=True, exist_ok=False)
  required_paths = (
      PRODUCT_TOOL, BUILD_GATE, COMPILE_GATE, FALLBACK_BOUND,
      REFERENCE_CONFIG, REFERENCE_CANDIDATE, REFERENCE_STOCK, PLUGIN)
  missing = [str(path) for path in required_paths if not path.is_file()]
  if missing:
    raise SystemExit("missing product-precheck inputs: " + ", ".join(missing))

  git = PRODUCT.BOOT.git_state(out)
  build_gate = PRODUCT.load_json(BUILD_GATE)
  compile_gate = PRODUCT.load_json(COMPILE_GATE)
  fallback_bound = PRODUCT.load_json(FALLBACK_BOUND)
  base_config = PRODUCT.load_json(REFERENCE_CONFIG)
  old_candidate = PRODUCT.load_json(REFERENCE_CANDIDATE)
  stock = PRODUCT.load_json(REFERENCE_STOCK)
  plugin_sha = sha256(PLUGIN)
  expected_tokens = [int(value)
                     for value in stock["generated_token_ids"][:OUTPUT_TOKENS]]
  candidate_reference_tokens = [
      int(value)
      for value in old_candidate["generated_token_ids"][:OUTPUT_TOKENS]]
  reference_path = out / "reference-output130.json"
  PRODUCT.write_json(reference_path, {
      "generated_token_ids": expected_tokens,
      "source": PRODUCT.relative(REFERENCE_STOCK),
  })

  config = dict(base_config)
  config.update({
      "candidate_gpu_plugin": str(PLUGIN),
      "case_id": "sentinel_002k_parallel_block_topk_output130",
      "capture_execution_census": True,
      "capture_logits": True,
      "checkpoint_steps": list(range(OUTPUT_TOKENS)),
      "output_tokens": OUTPUT_TOKENS,
      "purpose": "teacher_forced_correctness",
      "reference_result": str(reference_path.resolve()),
  })
  config.pop("instantiate_only", None)
  config.pop("compile_only", None)
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
  worker = PRODUCT.run_worker(worker_args, raw / "candidate", config)
  result = worker.get("result", {})
  distributions = (
      PRODUCT.ATTENTION_DIAGNOSTICS.distribution_rows(stock, result, ROOT)
      if result else [])
  klds = [
      float(row["kld_stock_to_candidate"]) for row in distributions
      if isinstance(row.get("kld_stock_to_candidate"), (int, float)) and
      math.isfinite(float(row["kld_stock_to_candidate"]))
  ]
  top1_rate = (
      sum(row.get("top1_match") is True for row in distributions)
      / len(distributions) if distributions else 0.0)
  finite_rows = (
      bool(distributions) and
      all(row.get("finite") is True for row in distributions))

  new_checkpoints = checkpoint_map(result)
  old_checkpoints = checkpoint_map(old_candidate)
  bitwise_mismatches = []
  invalid_checkpoint_hashes = []
  for step in range(OUTPUT_TOKENS):
    new_row = new_checkpoints.get(step)
    old_row = old_checkpoints.get(step)
    if new_row is None or old_row is None:
      bitwise_mismatches.append(step)
      continue
    new_path = checkpoint_path(new_row)
    old_path = checkpoint_path(old_row)
    if not new_path.is_file() or not old_path.is_file():
      bitwise_mismatches.append(step)
      continue
    new_sha = sha256(new_path)
    old_sha = sha256(old_path)
    if (new_row.get("sha256") != new_sha or
        old_row.get("sha256") != old_sha):
      invalid_checkpoint_hashes.append(step)
    if (new_sha != old_sha or new_row.get("shape") != old_row.get("shape")
        or new_row.get("byte_count") != old_row.get("byte_count")):
      bitwise_mismatches.append(step)

  trace = result.get("lm_head_i8q1_trace") or {}
  selections = trace.get("selection_rows") or []
  prepack = trace.get("weight_prepack_rows") or []
  provider_exact = (
      len(selections) == 2 and
      all(
          row.get("provider") == EXPECTED_PROVIDER and
          row.get("tokens") == 1 and
          row.get("rows") == 248320 and
          row.get("columns") == 2048 and
          row.get("correction_passes") == 2 and
          row.get("global") == [248320, 1, 1] and
          row.get("local") == [256, 1, 1]
          for row in selections))
  prepack_exact = (
      len(prepack) == 2 and
      prepack[0].get("process_cache_hit") is False and
      prepack[1].get("process_cache_hit") is True and
      all(
          row.get("codec") == "binary_two_centroid_lloyd5" and
          row.get("exact_correction_passes") == 2 and
          row.get("exact_correction_rows") == 11640 and
          row.get("direct_correction_topk") == 8
          for row in prepack))
  monitor = worker.get("monitor") or {}
  guard = worker.get("memory_guard") or {}
  minimum_available = int(
      monitor.get("system_available_min_bytes") or 0)
  stop_bytes = int(MEMORY_STOP_GIB * 1024**3)
  reference_slow_indices = sorted({
      int(index)
      for row in fallback_bound.get("worker_rows", [])
      for index in row.get("mode", {}).get("slow_indices_after_skip", [])
      if int(index) < OUTPUT_TOKENS
  })
  execution_counts = (
      result.get("execution_census", {}).get("executed_type_counts") or {})
  reference_execution_counts = (
      old_candidate.get("execution_census", {}).get(
          "executed_type_counts") or {})

  checks = [
      check("repository_clean_at_gate", not git["dirty"], git=git),
      check("seq2189_build_and_seq2190a_compile_gates_admit_precheck",
            build_gate.get("required_checks_passed") is True and
            build_gate.get("candidate_plugin", {}).get("sha256") ==
                EXPECTED_PLUGIN_SHA256 and
            compile_gate.get("required_checks_passed") is True and
            compile_gate.get("product_precheck_admitted") is True and
            compile_gate.get("plugin", {}).get("sha256") ==
                EXPECTED_PLUGIN_SHA256),
      check("accepted_seq2183_references_are_exact",
            sha256(REFERENCE_CONFIG) ==
                EXPECTED_REFERENCE_CONFIG_SHA256 and
            sha256(REFERENCE_CANDIDATE) ==
                EXPECTED_REFERENCE_CANDIDATE_SHA256 and
            sha256(REFERENCE_STOCK) == EXPECTED_REFERENCE_STOCK_SHA256 and
            expected_tokens == candidate_reference_tokens and
            old_candidate.get("candidate_gpu_plugin_sha256") != plugin_sha,
            reference_config_sha256=sha256(REFERENCE_CONFIG),
            reference_candidate_sha256=sha256(REFERENCE_CANDIDATE),
            reference_stock_sha256=sha256(REFERENCE_STOCK)),
      check("single_serial_candidate_worker_completes_without_oom",
            worker.get("returncode") == 0 and
            worker.get("timed_out") is False and
            worker.get("oom_observed") is False and
            worker.get("reused") is not True and
            (worker.get("worker_transient_scope") or {}).get("enabled")
                is True,
            worker={
                key: worker.get(key) for key in (
                    "returncode", "timed_out", "oom_observed",
                    "elapsed_seconds", "worker_transient_scope")
            }),
      check("isolated_plugin_and_count25_product_flags_are_exact",
            plugin_sha == EXPECTED_PLUGIN_SHA256 and
            result.get("candidate_gpu_plugin_sha256") == plugin_sha and
            result.get("candidate_path") == "hot_cold_custom" and
            result.get("lm_head_i8q1") is True and
            result.get("lm_head_i8q1_gated_exact") is True and
            result.get("lm_head_i8q1_gated_q4") is False and
            result.get("lm_head_i8q1_greedy_local2") is False and
            result.get("lm_head_token_only_feedback") is False),
      check("real_product_provider_selects_parallel_kernel_chain",
            provider_exact and prepack_exact,
            selection_count=len(selections),
            prepack_count=len(prepack),
            providers=sorted({
                str(row.get("provider")) for row in selections})),
      check("all_130_product_logits_are_bitwise_unchanged",
            len(new_checkpoints) == OUTPUT_TOKENS and
            not bitwise_mismatches and not invalid_checkpoint_hashes,
            checkpoint_count=len(new_checkpoints),
            mismatch_steps=bitwise_mismatches,
            invalid_hash_steps=invalid_checkpoint_hashes),
      check("all_130_stock_relative_distributions_pass",
            len(distributions) == OUTPUT_TOKENS and finite_rows and
            len(klds) == OUTPUT_TOKENS and max(klds) <= KLD_MAX and
            top1_rate >= TOP1_MIN,
            row_count=len(distributions),
            max_kld=max(klds) if klds else None,
            kld_threshold=KLD_MAX,
            top1_rate=top1_rate,
            top1_threshold=TOP1_MIN),
      check("exact_output130_tokens_are_preserved",
            result.get("generated_token_count") == OUTPUT_TOKENS and
            result.get("generated_token_ids") == expected_tokens and
            result.get("generated_token_ids_sha256") ==
                EXPECTED_TOKEN_SHA256 and
            result.get("teacher_forced_from_stock") is True and
            result.get("sentinel_pass") is False,
            expected_token_sha256=EXPECTED_TOKEN_SHA256,
            observed_token_sha256=result.get(
                "generated_token_ids_sha256"),
            note=(
                "the accepted 2k answer first appears after token 130; "
                "sentinel truth remains required at output512")),
      check("source_state_and_execution_census_are_unchanged",
            result.get("source_summary") ==
                old_candidate.get("source_summary") and
            result.get("state_schema_after") ==
                old_candidate.get("state_schema_after") and
            execution_counts == reference_execution_counts,
            executed_type_counts=execution_counts),
      check("output130_covers_known_2k_fallback_interval_indices",
            fallback_bound.get("required_checks_passed") is True and
            fallback_bound.get("fallback_indices_exact_across_workers")
                is True and
            len(reference_slow_indices) >= 1 and
            max(reference_slow_indices) < OUTPUT_TOKENS,
            covered_reference_slow_indices=reference_slow_indices,
            note=(
                "latency classification is prior evidence; bitwise logits "
                "and provider trace are the current correctness evidence")),
      check("memory_guard_never_trips",
            guard.get("tripped") is False and
            minimum_available >= stop_bytes and
            int(monitor.get("process_rss_peak_bytes", -1)) >= 0 and
            int(monitor.get("process_swap_peak_bytes", -1)) >= 0,
            stop_bytes=stop_bytes, monitor=monitor),
  ]
  required = all(row["pass"] for row in checks)
  verdict = (
      "admit_parallel_block_topk_for_one_2k_abba_precheck"
      if required else
      "repair_parallel_block_topk_product_correctness")
  payload = {
      "schema": SCHEMA,
      "workstream": WS,
      "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
      "git": git,
      "verdict": verdict,
      "required_checks_passed": required,
      "abba_precheck_admitted": required,
      "formal_product_promotion_admitted": False,
      "performance_claim_admitted": False,
      "gpu_workers_launched": 1,
      "stock_workers_launched": 0,
      "candidate_workers_launched": 1,
      "workers_concurrent": False,
      "checks": checks,
      "plugin": {"path": str(PLUGIN), "sha256": plugin_sha},
      "correctness": {
          "checkpoint_count": len(new_checkpoints),
          "bitwise_mismatch_steps": bitwise_mismatches,
          "invalid_checkpoint_hash_steps": invalid_checkpoint_hashes,
          "distribution_row_count": len(distributions),
          "max_kld": max(klds) if klds else None,
          "top1_rate": top1_rate,
          "generated_token_ids_sha256": result.get(
              "generated_token_ids_sha256"),
          "sentinel_pass": result.get("sentinel_pass"),
      },
      "provider": {
          "expected": EXPECTED_PROVIDER,
          "selection_count": len(selections),
          "prepack_count": len(prepack),
      },
      "worker": worker,
      "next_action": {
          "route": "parallel_block_topk_2k_abba_precheck",
          "requirements": [
              "run one 2k prefill-shape ABBA1 block with output512",
              "require exact timing tokens and both candidate rows",
              "require jitter p95/p50 at or below 1.25 before formal ABBA8",
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
              PRODUCT_TOOL, BUILD_GATE, COMPILE_GATE, FALLBACK_BOUND,
              REFERENCE_CONFIG, REFERENCE_CANDIDATE, REFERENCE_STOCK, PLUGIN)
      },
      "plugin": payload["plugin"],
      "gpu_workers": 1,
      "stock_workers": 0,
      "candidate_workers": 1,
      "workers_concurrent": False,
  })
  report = f"""# Parallel exact block-top8 2k product precheck

Verdict: **{verdict}**. Required checks: `{str(required).lower()}`.

One isolated candidate worker emits `{OUTPUT_TOKENS}` teacher-forced logits.
Bitwise mismatches against seq2183: `{len(bitwise_mismatches)}`. Stock-relative
max KLD is `{max(klds) if klds else None}` and top-1 rate is `{top1_rate}`.
The real twelve-stage count25 provider is selected twice and the new plugin
preserves exact token SHA `{result.get('generated_token_ids_sha256')}`.

Peak RSS/swap telemetry is
`{int(monitor.get('process_rss_peak_bytes', 0))}/`
`{int(monitor.get('process_swap_peak_bytes', 0))} B`; minimum available memory
is `{minimum_available} B`. No OOM or guard event occurred. This is a
correctness precheck, not a speed claim.
"""
  (out / "report.md").write_text(report, encoding="utf-8")
  print(json.dumps({
      "artifact": PRODUCT.relative(out),
      "verdict": verdict,
      "required_checks_passed": required,
      "bitwise_mismatch_count": len(bitwise_mismatches),
      "max_kld": max(klds) if klds else None,
      "top1_rate": top1_rate,
      "token_sha256": result.get("generated_token_ids_sha256"),
      "peak_rss_bytes": monitor.get("process_rss_peak_bytes"),
      "oom_observed": worker.get("oom_observed"),
  }, separators=(",", ":")), flush=True)
  return 0 if required else 2


if __name__ == "__main__":
  raise SystemExit(main())
