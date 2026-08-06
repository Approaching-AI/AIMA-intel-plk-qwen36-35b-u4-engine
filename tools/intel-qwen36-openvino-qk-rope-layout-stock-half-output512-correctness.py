#!/usr/bin/env python3
"""Run one full-logit output512 stock-half Q/K correctness gate.

The sole candidate retains the exact seq2193 carrier and enables only the
seq2196 Q/K producer.  It is teacher-forced from the accepted stock row and
captures every output distribution plus the final execution census.  A pass
can fund formal incremental ABBA8 design; it is not a speed claim.
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
    "intel-qwen36-openvino-qk-rope-layout-stock-half-"
    "output512-correctness-v1")
BASE_TOOL = ROOT / (
    "tools/intel-qwen36-openvino-qk-rope-layout-"
    "exact-phase-product-precheck.py")
SEQ2197_GATE = ROOT / (
    "output/openvino-qk-rope-layout-stock-half-product-precheck-"
    "20260731Tseq2197-clean/result.json")
SEQ2198_GATE = ROOT / (
    "output/openvino-qk-rope-layout-stock-half-abba-precheck-"
    "20260731Tseq2198-clean/result.json")
FORMAL_ROOT = ROOT / (
    "output/openvino-lm-head-parallel-block-topk-2k-abba8-"
    "20260731Tseq2193-clean")
BASE_CANDIDATE = FORMAL_ROOT / (
    "raw/prefill_shape_002k/correctness/candidate/worker-result.json")
STOCK = FORMAL_ROOT / (
    "raw/prefill_shape_002k/correctness/stock/worker-result.json")
BASE_CONFIG = ROOT / (
    "output/openvino-lm-head-parallel-block-topk-2k-abba8-"
    "20260731Tseq2193-clean/raw/prefill_shape_002k/correctness/"
    "candidate/worker-config.json")
KERNEL_SOURCE = ROOT / "engine/openvino/custom/iq36_qk_rope_layout.cl"
PLUGIN = Path(
    "/home/intel/intel-qwen36-r0/output/"
    "openvino-90214e-l0-gpu-seq2189/bin/intel64/Release/"
    "libopenvino_intel_gpu_plugin.so")
EXPECTED_SHA256 = {
    BASE_TOOL: (
        "33f02a788133c7b9e7a89b31be9380c04fb24e6e439252e03346110889cd5e78"),
    SEQ2197_GATE: (
        "4a26a565d518a5f082b48979b9c807499aacdde04daa1c20a7de555e49cfe198"),
    SEQ2198_GATE: (
        "c989b34615b971c21784240698e1ef32a57633ed7549da5fae029e584ad1fbaa"),
    BASE_CANDIDATE: (
        "707fe8f4f47270afcaaa2499c6d1906841df4cd621071b656425279c1c17589d"),
    STOCK: (
        "5f7d0d0fbbde73e8e546a513fe294282d1f961ae72362ed2dd6900b2125d0da1"),
    BASE_CONFIG: (
        "86ab47064ead3de6e87609a3c432c68274362104e00f5d517fd6acd99795c267"),
    KERNEL_SOURCE: (
        "be2b1105df7503a24636615a94255e0683d0b8a73bbecd1c7b70d0b9f5306863"),
    PLUGIN: (
        "9832adffa8bf3fd7b013c47d9d2abefa66987c5fbde350c9669fce58c697b985"),
}
EXPECTED_TOKEN_SHA256 = (
    "0a7b56baf11a00512a786c0c825bba4733fda84eb5b87eb703c79344f508ea63")
LAYERS = tuple(range(3, 40, 4))
OUTPUT_TOKENS = 512
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


BASE = load_module("iq36_qk_output512_base", BASE_TOOL)
PRODUCT = BASE.PRODUCT


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out-dir", type=Path, required=True)
  parser.add_argument("--plan-only", action="store_true")
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


def provider_exact(result: dict[str, Any]) -> bool:
  trace = result.get("lm_head_i8q1_trace") or {}
  selections = trace.get("selection_rows") or []
  prepacks = trace.get("weight_prepack_rows") or []
  return (
      len(selections) == 2 and len(prepacks) == 2 and
      prepacks[0].get("process_cache_hit") is False and
      prepacks[1].get("process_cache_hit") is True and
      all(
          row.get("provider") == BASE.EXPECTED_PROVIDER and
          row.get("global") == [248320, 1, 1] and
          row.get("local") == [256, 1, 1] and
          row.get("correction_passes") == 2
          for row in selections))


def main() -> int:
  args = parse_args()
  out = args.out_dir.resolve()
  if out.exists():
    raise SystemExit(f"output already exists: {out}")
  raw = out / "raw"
  raw.mkdir(parents=True)
  required_paths = (*EXPECTED_SHA256, Path("/usr/bin/ocloc"))
  missing = [str(path) for path in required_paths if not path.is_file()]
  if missing:
    raise SystemExit("missing output512 inputs: " + ", ".join(missing))

  git = PRODUCT.BOOT.git_state(out)
  input_hashes = {path: sha256(path) for path in EXPECTED_SHA256}
  seq2197 = PRODUCT.load_json(SEQ2197_GATE)
  seq2198 = PRODUCT.load_json(SEQ2198_GATE)
  base_candidate = PRODUCT.load_json(BASE_CANDIDATE)
  stock = PRODUCT.load_json(STOCK)
  base_config = PRODUCT.load_json(BASE_CONFIG)
  config = dict(base_config)
  config.update({
      "candidate_gpu_plugin": str(PLUGIN),
      "capture_execution_census": True,
      "capture_logits": True,
      "case_id": "prefill_shape_002k_qk_rope_stock_half_output512",
      "checkpoint_steps": list(range(OUTPUT_TOKENS)),
      "fuse_qk_rope_layout": True,
      "output_tokens": OUTPUT_TOKENS,
      "purpose": "teacher_forced_correctness",
      "reference_result": str(STOCK.resolve()),
  })
  config.pop("compile_only", None)
  config.pop("instantiate_only", None)
  config_delta = {
      key: {"control": base_config.get(key), "candidate": config.get(key)}
      for key in sorted(set(base_config) | set(config))
      if base_config.get(key) != config.get(key)
  }
  static_checks = [
      check("repository_clean_at_gate", not git["dirty"], git=git),
      check("all_frozen_inputs_have_exact_hashes",
            all(input_hashes[path] == expected
                for path, expected in EXPECTED_SHA256.items()),
            observed={
                PRODUCT.relative(path): digest
                for path, digest in input_hashes.items()}),
      check("seq2197_and_seq2198_admit_only_output512_correctness",
            seq2197.get("required_checks_passed") is True and
            seq2197.get("abba_precheck_admitted") is True and
            seq2197.get("git", {}).get("commit") ==
                "d978d75d59f5990f9a41723d14390503fc540340" and
            seq2198.get("required_checks_passed") is True and
            seq2198.get("output512_correctness_gate_admitted") is True and
            seq2198.get("formal_product_promotion_admitted") is False and
            seq2198.get("performance_claim_admitted") is False and
            seq2198.get("git", {}).get("commit") ==
                "25679e7ca9ba55c999dd20939eb4080e41a6e896"),
      check("prefill_shape_output512_config_diff_is_exactly_qk_enablement",
            set(config_delta) == {"case_id", "fuse_qk_rope_layout"} and
            base_config.get("case_id") == "prefill_shape_002k" and
            str(base_config.get("prompt", "")).endswith(
                "/prompts/prefill_shape_002k.txt") and
            base_config.get("output_tokens") == OUTPUT_TOKENS and
            base_config.get("purpose") == "teacher_forced_correctness" and
            base_config.get("capture_execution_census") is True and
            base_config.get("capture_logits") is True and
            base_config.get("checkpoint_steps") ==
                list(range(OUTPUT_TOKENS)) and
            base_config.get("reference_result") == str(STOCK.resolve()) and
            base_config.get("fuse_qk_rope_layout") is None and
            config.get("fuse_qk_rope_layout") is True,
            config_delta=config_delta,
            prompt=base_config.get("prompt")),
  ]
  static_passed = all(row["pass"] for row in static_checks)
  if args.plan_only or not static_passed:
    verdict = (
        "admit_one_bound_prefill_shape_output512_qk_correctness_worker"
        if static_passed else
        "reject_output512_worker_before_gpu")
    payload = {
        "schema": SCHEMA,
        "workstream": WS,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git": git,
        "verdict": verdict,
        "required_checks_passed": static_passed,
        "plan_only": True,
        "gpu_workers_launched": 0,
        "checks": static_checks,
        "config_delta": config_delta,
    }
    PRODUCT.write_json(out / "result.json", payload)
    PRODUCT.write_json(out / "manifest.json", {
        "schema": SCHEMA,
        "tool": PRODUCT.relative(Path(__file__)),
        "git": git,
        "inputs": {
            PRODUCT.relative(path): digest
            for path, digest in input_hashes.items()},
        "plan_only": True,
        "gpu_workers": 0,
    })
    print(json.dumps({
        "artifact": PRODUCT.relative(out),
        "verdict": verdict,
        "required_checks_passed": static_passed,
        "config_delta": config_delta,
        "gpu_workers_launched": 0,
    }, separators=(",", ":")), flush=True)
    return 0 if static_passed else 2

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
  result = worker.get("result") or {}
  codegen = BASE.qk_codegen_audit(
      raw / "candidate/neo-cache", raw / "qk-codegen", args.timeout_s)
  PRODUCT.write_json(raw / "qk-codegen.json", codegen)

  stock_distribution = (
      BASE.distribution_summary(stock, result) if result else {})
  carrier_distribution = (
      BASE.distribution_summary(base_candidate, result) if result else {})
  base_checkpoints = base_candidate.get("distribution_checkpoints") or []
  checkpoints = result.get("distribution_checkpoints") or []
  base_hashes = [row.get("sha256") for row in base_checkpoints]
  checkpoint_hashes = [row.get("sha256") for row in checkpoints]
  source = result.get("source_summary") or {}
  baseline_source = base_candidate.get("source_summary") or {}
  counts = (
      (result.get("execution_census") or {}).get(
          "executed_type_counts") or {})
  baseline_counts = (
      (base_candidate.get("execution_census") or {}).get(
          "executed_type_counts") or {})
  boundaries = BASE.boundary_audit(result)
  monitor = worker.get("monitor") or {}
  guard = worker.get("memory_guard") or {}
  minimum_available = int(
      monitor.get("system_available_min_bytes") or 0)
  stop_bytes = int(MEMORY_STOP_GIB * 1024**3)
  codegen_rows = [
      row["assemblies"][0] for row in codegen["programs"]
      if row["returncode"] == 0 and len(row["assemblies"]) == 1]
  half_codegen_exact = (
      codegen["matching_program_count"] == 20 and
      codegen["unique_program_count"] == 2 and
      len(codegen_rows) == 2 and
      all(
          row["simd32_half_mul"] == 4 and
          row["simd32_half_mad"] == 4 and
          row["simd32_float_mul"] == 0 and
          row["simd32_float_mad"] == 0 and
          row["simd32_float_to_half_moves"] == 0
          for row in codegen_rows))

  checks = static_checks + [
      check("single_serial_candidate_completes_without_oom",
            worker.get("returncode") == 0 and
            worker.get("timed_out") is False and
            worker.get("oom_observed") is False and
            worker.get("reused") is not True and
            (worker.get("worker_transient_scope") or {}).get("enabled")
                is True),
      check("exact_parallel_carrier_and_provider_are_unchanged",
            result.get("mode") == "candidate" and
            result.get("candidate_path") == "hot_cold_custom" and
            result.get("custom_composition") == "exact_phase" and
            result.get("exact_phase_dual_cohort") is True and
            result.get("target_layers") == list(LAYERS) and
            result.get("decode_stock_micro_layers") == list(LAYERS) and
            result.get("candidate_gpu_plugin_sha256") ==
                EXPECTED_SHA256[PLUGIN] and
            result.get("lm_head_i8q1") is True and
            result.get("lm_head_i8q1_gated_exact") is True and
            result.get("lm_head_i8q1_greedy_local2") is False and
            result.get("lm_head_token_only_feedback") is False and
            provider_exact(result)),
      check("qk_source_state_and_nonboundary_execution_are_exact",
            result.get("fuse_qk_rope_layout") is True and
            source.get("fuse_qk_rope_layout") is True and
            source.get("qk_rope_layout_rewrite_count") == len(LAYERS) and
            BASE.without_qk_summary(source) ==
                BASE.without_qk_summary(baseline_source) and
            result.get("state_schema_after") ==
                base_candidate.get("state_schema_after") and
            baseline_counts.get("Gather") == 12 and
            counts.get("Gather") == 11 and
            BASE.without_qk_execution(counts) ==
                BASE.without_qk_execution(baseline_counts)),
      check("exact_qk_and_output_boundaries_execute",
            counts.get("IQ36QKRopeLayout") == len(LAYERS) and
            counts.get("IQ36ExactPhaseDualCohortHotAttentionGQA") ==
                len(LAYERS) and
            not boundaries["old_qk_rows"] and
            len(boundaries["output_transpose_rows"]) == len(LAYERS) and
            len(boundaries["output_gate_rows"]) == len(LAYERS),
            counts=counts, boundaries=boundaries),
      check("both_real_qk_shapes_keep_half_only_codegen",
            half_codegen_exact, codegen=codegen),
      check("all_output512_logits_are_bitwise_equal_to_current_carrier",
            len(base_hashes) == OUTPUT_TOKENS and
            len(checkpoint_hashes) == OUTPUT_TOKENS and
            checkpoint_hashes == base_hashes and
            carrier_distribution.get("row_count") == OUTPUT_TOKENS and
            carrier_distribution.get("finite") is True and
            carrier_distribution.get("max_kld") == 0.0 and
            carrier_distribution.get("top1_rate") == 1.0,
            current_carrier_distribution=carrier_distribution),
      check("all_output512_stock_distributions_pass",
            stock_distribution.get("row_count") == OUTPUT_TOKENS and
            stock_distribution.get("finite") is True and
            float(stock_distribution.get("max_kld", math.inf)) <= KLD_MAX and
            float(stock_distribution.get("top1_rate", 0.0)) >= TOP1_MIN,
            stock_distribution=stock_distribution,
            kld_threshold=KLD_MAX, top1_threshold=TOP1_MIN),
      check("exact_output512_tokens_are_preserved",
            result.get("generated_token_count") == OUTPUT_TOKENS and
            result.get("generated_token_ids_sha256") ==
                EXPECTED_TOKEN_SHA256 and
            result.get("generated_token_ids") ==
                stock.get("generated_token_ids") and
            result.get("teacher_forced_from_stock") is True),
      check("memory_guard_never_trips",
            guard.get("tripped") is False and
            minimum_available >= stop_bytes and
            int(monitor.get("process_rss_peak_bytes", -1)) >= 0 and
            int(monitor.get("process_swap_peak_bytes", -1)) >= 0,
            stop_bytes=stop_bytes, monitor=monitor),
  ]
  passed = all(row["pass"] for row in checks)
  verdict = (
      "admit_qk_rope_stock_half_formal_incremental_abba8_design"
      if passed else
      "reject_qk_rope_stock_half_before_formal_incremental_inference")
  payload = {
      "schema": SCHEMA,
      "workstream": WS,
      "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
      "git": git,
      "verdict": verdict,
      "required_checks_passed": passed,
      "formal_incremental_abba8_design_admitted": passed,
      "formal_product_promotion_admitted": False,
      "performance_claim_admitted": False,
      "correctness": {
          "stock_relative": stock_distribution,
          "current_carrier_relative": carrier_distribution,
          "bitwise_checkpoint_count": (
              OUTPUT_TOKENS if checkpoint_hashes == base_hashes else 0),
          "generated_token_ids_sha256": result.get(
              "generated_token_ids_sha256"),
      },
      "execution": {
          "executed_type_counts": counts,
          "boundary_audit": boundaries,
          "qk_codegen": codegen,
      },
      "checks": checks,
      "worker": worker,
      "next_action": {
          "route": "openvino_qk_rope_stock_half_formal_incremental_abba8",
          "requirements": [
              "reuse clean seq2198 as block zero",
              "add seven control-QK-QK-control blocks serially",
              "require one-sided 95 percent LCB at least 1.005",
              "retain output512 correctness and every memory guard",
          ],
      },
  }
  PRODUCT.write_json(out / "result.json", payload)
  PRODUCT.write_json(out / "manifest.json", {
      "schema": SCHEMA,
      "tool": PRODUCT.relative(Path(__file__)),
      "git": git,
      "inputs": {
          PRODUCT.relative(path): digest
          for path, digest in input_hashes.items()},
      "gpu_workers": 1,
      "candidate_workers": 1,
      "stock_workers": 0,
      "workers_concurrent": False,
  })
  report = f"""# Stock-half Q/K RoPE output512 correctness

Verdict: **{verdict}**. Required checks: `{str(passed).lower()}`.

Bitwise-equal current-carrier checkpoints:
`{OUTPUT_TOKENS if checkpoint_hashes == base_hashes else 0}/512`.
Stock-relative max KLD/top-1:
`{stock_distribution.get('max_kld')}/{stock_distribution.get('top1_rate')}`.
Both real Q/K shapes retain four SIMD32 half mul/mad pairs and no F32 pair.

Peak RSS/swap is `{int(monitor.get('process_rss_peak_bytes', 0))}/`
`{int(monitor.get('process_swap_peak_bytes', 0))} B`; minimum available
memory is `{minimum_available} B`. This is correctness evidence, not a speed
claim.
"""
  (out / "report.md").write_text(report, encoding="utf-8")
  print(json.dumps({
      "artifact": PRODUCT.relative(out),
      "verdict": verdict,
      "required_checks_passed": passed,
      "bitwise_checkpoint_count": (
          OUTPUT_TOKENS if checkpoint_hashes == base_hashes else 0),
      "max_kld": stock_distribution.get("max_kld"),
      "top1_rate": stock_distribution.get("top1_rate"),
      "peak_rss_bytes": monitor.get("process_rss_peak_bytes"),
      "oom_observed": worker.get("oom_observed"),
  }, separators=(",", ":")), flush=True)
  return 0 if passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
