#!/usr/bin/env python3
"""Run one control-QK-QK-control 2k incremental timing block.

All four workers use the accepted seq2189 plugin and exact-phase parallel
block-top8 carrier.  The A rows leave IQ36QKRopeLayout disabled; the B rows
enable only the seq2196 stock-half correction.  This ABBA1 gate checks exact
output512 tokens, carrier isolation, point movement, jitter, and memory.  One
block is exploratory admission evidence, never a formal speed claim.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import math
import statistics
from pathlib import Path
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = (
    "intel-qwen36-openvino-qk-rope-layout-stock-half-"
    "abba-precheck-v1")
PRODUCT_TOOL = ROOT / "tools/intel-qwen36-openvino-hot-cold-product-gate.py"
CORRECTNESS_GATE = ROOT / (
    "output/openvino-qk-rope-layout-stock-half-product-precheck-"
    "20260731Tseq2197-clean/result.json")
FORMAL_ROOT = ROOT / (
    "output/openvino-lm-head-parallel-block-topk-2k-abba8-"
    "20260731Tseq2193-clean")
FORMAL_GATE = FORMAL_ROOT / "gate.json"
REFERENCE_RESULT = FORMAL_ROOT / (
    "raw/prefill_shape_002k/correctness/stock/worker-result.json")
BASE_CONFIG = FORMAL_ROOT / (
    "raw/prefill_shape_002k/block00/candidate-b1/worker-config.json")
BASE_CONTROL_RESULT = FORMAL_ROOT / (
    "raw/prefill_shape_002k/block00/candidate-b1/worker-result.json")
KERNEL_SOURCE = ROOT / "engine/openvino/custom/iq36_qk_rope_layout.cl"
PLUGIN = Path(
    "/home/intel/intel-qwen36-r0/output/"
    "openvino-90214e-l0-gpu-seq2189/bin/intel64/Release/"
    "libopenvino_intel_gpu_plugin.so")
EXPECTED_SHA256 = {
    CORRECTNESS_GATE: (
        "4a26a565d518a5f082b48979b9c807499aacdde04daa1c20a7de555e49cfe198"),
    FORMAL_GATE: (
        "c125f51dde39d6080ed1b4a8698cb3864874fcf31e3acb5a38fffbae9c86ceee"),
    REFERENCE_RESULT: (
        "5f7d0d0fbbde73e8e546a513fe294282d1f961ae72362ed2dd6900b2125d0da1"),
    BASE_CONFIG: (
        "7cd2281731dafd08a6a58b7cf36da18c06d0e034f04dcc11bd394103661c2911"),
    BASE_CONTROL_RESULT: (
        "97fb14858ddbff603c8c4874946b40497bae3029410ad3bf99c5a4c3b1b3c1df"),
    KERNEL_SOURCE: (
        "be2b1105df7503a24636615a94255e0683d0b8a73bbecd1c7b70d0b9f5306863"),
    PLUGIN: (
        "9832adffa8bf3fd7b013c47d9d2abefa66987c5fbde350c9669fce58c697b985"),
}
EXPECTED_TOKEN_SHA256 = (
    "0a7b56baf11a00512a786c0c825bba4733fda84eb5b87eb703c79344f508ea63")
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
LAYERS = tuple(range(3, 40, 4))
SCHEDULE = ("control-a1", "qk-b1", "qk-b2", "control-a2")
JITTER_SKIP = 16
JITTER_MAX = 1.25
DECODE_RATIO_MIN = 1.005
TOTAL_RATIO_MIN = 1.005
PREFILL_RATIO_MIN = 0.995
NOISE_CUT_MS = 0.098685
PREFLIGHT_GIB = 8.0
MEMORY_STOP_GIB = 4.0


def load_module(name: str, path: Path) -> Any:
  spec = importlib.util.spec_from_file_location(name, path)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import {path}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


PRODUCT = load_module("iq36_qk_stock_half_abba_product", PRODUCT_TOOL)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out-dir", type=Path, required=True)
  parser.add_argument("--resume", action="store_true")
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


def finite(value: Any) -> bool:
  return (
      isinstance(value, (int, float)) and not isinstance(value, bool) and
      math.isfinite(float(value)))


def ratio_text(value: Any) -> str:
  return f"{float(value):.6f}" if finite(value) else "unavailable"


def tail_metrics(result: dict[str, Any]) -> dict[str, Any]:
  samples = [
      float(value) for value in
      result.get("decode_wall_ms", [])[JITTER_SKIP:]]
  p50 = PRODUCT.percentile(samples, 0.50)
  p95 = PRODUCT.percentile(samples, 0.95)
  ratio = p95 / p50 if p50 and p95 else None
  return {
      "sample_count": len(samples),
      "p50_ms": p50,
      "p95_ms": p95,
      "p95_over_p50": ratio,
      "jitter_pass": (
          finite(ratio) and float(ratio) <= JITTER_MAX),
  }


def provider_exact(result: dict[str, Any]) -> bool:
  trace = result.get("lm_head_i8q1_trace") or {}
  selections = trace.get("selection_rows") or []
  prepacks = trace.get("weight_prepack_rows") or []
  return (
      len(selections) == 2 and len(prepacks) == 2 and
      prepacks[0].get("process_cache_hit") is False and
      prepacks[1].get("process_cache_hit") is True and
      all(
          row.get("provider") == EXPECTED_PROVIDER and
          row.get("global") == [248320, 1, 1] and
          row.get("local") == [256, 1, 1] and
          row.get("correction_passes") == 2
          for row in selections))


def phase_summary(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
  control = [results["control-a1"], results["control-a2"]]
  candidate = [results["qk-b1"], results["qk-b2"]]
  phases = {}
  for key in ("prefill_tokens_s", "decode_tokens_s"):
    control_value = statistics.median(float(row[key]) for row in control)
    candidate_value = statistics.median(float(row[key]) for row in candidate)
    phases[key] = {
        "control": control_value,
        "candidate": candidate_value,
        "ratio": candidate_value / control_value,
    }
  control_total = statistics.median(
      1000.0 / float(row["total_wall_ms"]) for row in control)
  candidate_total = statistics.median(
      1000.0 / float(row["total_wall_ms"]) for row in candidate)
  phases["total_rate"] = {
      "control": control_total,
      "candidate": candidate_total,
      "ratio": candidate_total / control_total,
  }
  return phases


def main() -> int:
  args = parse_args()
  out = args.out_dir.resolve()
  if out.exists() and not args.resume:
    raise SystemExit(f"output already exists: {out}")
  raw = out / "raw"
  raw.mkdir(parents=True, exist_ok=args.resume)
  required_paths = (PRODUCT_TOOL, *EXPECTED_SHA256)
  missing = [str(path) for path in required_paths if not path.is_file()]
  if missing:
    raise SystemExit("missing incremental ABBA inputs: " + ", ".join(missing))

  git = PRODUCT.BOOT.git_state(out)
  input_hashes = {path: sha256(path) for path in EXPECTED_SHA256}
  correctness = PRODUCT.load_json(CORRECTNESS_GATE)
  formal = PRODUCT.load_json(FORMAL_GATE)
  reference = PRODUCT.load_json(REFERENCE_RESULT)
  base_config = PRODUCT.load_json(BASE_CONFIG)
  base_control = PRODUCT.load_json(BASE_CONTROL_RESULT)
  expected_tokens = reference.get("generated_token_ids")

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
      resume=args.resume,
      timeout_s=args.timeout_s,
      worker_transient_scope=True,
  )
  runs: dict[str, dict[str, Any]] = {}
  configs: dict[str, dict[str, Any]] = {}
  for label in SCHEDULE:
    qk_enabled = label.startswith("qk-")
    config = dict(base_config)
    config.update({
        "candidate_gpu_plugin": str(PLUGIN),
        "capture_execution_census": False,
        "capture_logits": False,
        "case_id": (
            "prefill_shape_002k_stock_half_qk"
            if qk_enabled else
            "prefill_shape_002k_qk_control"),
        "fuse_qk_rope_layout": qk_enabled,
        "output_tokens": 512,
        "purpose": "paired_product_timing",
        "reference_result": str(REFERENCE_RESULT.resolve()),
    })
    configs[label] = config
    runs[label] = PRODUCT.run_worker(
        worker_args, raw / "block00" / label, config)

  results = {label: run.get("result") or {} for label, run in runs.items()}
  complete = all(
      all(finite(results[label].get(key)) for key in (
          "prefill_tokens_s", "decode_tokens_s", "total_wall_ms"))
      for label in SCHEDULE)
  phases = phase_summary(results) if complete else {}
  tails = {label: tail_metrics(result) for label, result in results.items()}
  control_tail_ms = (
      statistics.median(
          float(tails[label]["p50_ms"])
          for label in ("control-a1", "control-a2"))
      if all(finite(tails[label]["p50_ms"])
             for label in ("control-a1", "control-a2")) else None)
  qk_tail_ms = (
      statistics.median(
          float(tails[label]["p50_ms"])
          for label in ("qk-b1", "qk-b2"))
      if all(finite(tails[label]["p50_ms"])
             for label in ("qk-b1", "qk-b2")) else None)
  tail_saving_ms = (
      control_tail_ms - qk_tail_ms
      if finite(control_tail_ms) and finite(qk_tail_ms) else None)
  scopes = [
      (run.get("worker_transient_scope") or {}).get("unit")
      for run in runs.values()]
  monitors = [run.get("monitor") or {} for run in runs.values()]
  peak_rss = max(
      int(row.get("process_rss_peak_bytes") or 0) for row in monitors)
  peak_swap = max(
      int(row.get("process_swap_peak_bytes") or 0) for row in monitors)
  minimum_available = min(
      int(row.get("system_available_min_bytes") or 0) for row in monitors)
  stop_bytes = int(MEMORY_STOP_GIB * 1024**3)
  state_schemas = [
      results[label].get("state_schema_after") for label in SCHEDULE]

  checks = [
      check("repository_clean_at_gate", not git["dirty"], git=git),
      check("all_frozen_inputs_have_exact_hashes",
            all(input_hashes[path] == expected
                for path, expected in EXPECTED_SHA256.items()),
            observed={
                PRODUCT.relative(path): digest
                for path, digest in input_hashes.items()}),
      check("seq2197_correctness_and_seq2193_control_are_bound",
            correctness.get("required_checks_passed") is True and
            correctness.get("abba_precheck_admitted") is True and
            correctness.get("correctness", {}).get(
                "stock_relative", {}).get("max_kld") ==
                0.0009099625244360312 and
            correctness.get("correctness", {}).get(
                "stock_relative", {}).get("top1_rate") == 1.0 and
            formal.get("run_checks_passed") is True and
            formal.get("product_promotion_ready") is False and
            formal.get("speedup_claims_allowed") is False and
            (base_control.get("source_summary") or {}).get(
                "fuse_qk_rope_layout") is False),
      check("strict_control_qk_qk_control_workers_complete_serially",
            list(runs) == list(SCHEDULE) and
            all(
                run.get("returncode") == 0 and
                run.get("timed_out") is False and
                run.get("oom_observed") is False and
                (run.get("reused") is not True or args.resume)
                for run in runs.values()) and
            len(scopes) == len(set(scopes)) and
            all(scope is not None for scope in scopes),
            schedule=list(runs), scopes=scopes,
            aggregation_reused_worker_evidence=args.resume),
      check("all_four_output512_token_streams_are_exact",
            reference.get("generated_token_ids_sha256") ==
                EXPECTED_TOKEN_SHA256 and
            len(expected_tokens or []) == 512 and
            all(
                result.get("generated_token_count") == 512 and
                result.get("generated_token_ids_sha256") ==
                    EXPECTED_TOKEN_SHA256 and
                result.get("generated_token_ids") == expected_tokens and
                result.get("teacher_forced_from_stock") is True
                for result in results.values()),
            expected_sha256=EXPECTED_TOKEN_SHA256,
            observed={
                label: result.get("generated_token_ids_sha256")
                for label, result in results.items()}),
      check("all_rows_retain_exact_parallel_carrier_and_provider",
            all(
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
                provider_exact(result)
                for result in results.values())),
      check("qk_flag_is_the_only_carrier_switch",
            all(
                results[label].get("fuse_qk_rope_layout") is False and
                (results[label].get("source_summary") or {}).get(
                    "qk_rope_layout_rewrite_count") == 0
                for label in ("control-a1", "control-a2")) and
            all(
                results[label].get("fuse_qk_rope_layout") is True and
                (results[label].get("source_summary") or {}).get(
                    "qk_rope_layout_rewrite_count") == len(LAYERS)
                for label in ("qk-b1", "qk-b2")) and
            len(state_schemas) == 4 and
            all(value == state_schemas[0] for value in state_schemas),
            effective_flags={
                label: results[label].get("fuse_qk_rope_layout")
                for label in SCHEDULE}),
      check("all_four_stable_tail_jitter_rows_pass",
            all(row["jitter_pass"] for row in tails.values()),
            threshold=JITTER_MAX, tails=tails),
      check("incremental_point_movement_clears_noise_cut",
            set(phases) == {
                "prefill_tokens_s", "decode_tokens_s", "total_rate"} and
            phases["prefill_tokens_s"]["ratio"] >= PREFILL_RATIO_MIN and
            phases["decode_tokens_s"]["ratio"] >= DECODE_RATIO_MIN and
            phases["total_rate"]["ratio"] >= TOTAL_RATIO_MIN and
            finite(tail_saving_ms) and
            float(tail_saving_ms) >= NOISE_CUT_MS,
            thresholds={
                "prefill_ratio": PREFILL_RATIO_MIN,
                "decode_ratio": DECODE_RATIO_MIN,
                "total_ratio": TOTAL_RATIO_MIN,
                "tail_saving_ms": NOISE_CUT_MS},
            phases=phases, control_tail_ms=control_tail_ms,
            qk_tail_ms=qk_tail_ms, tail_saving_ms=tail_saving_ms),
      check("timing_workers_capture_no_profile_or_logits",
            all(
                (result.get("execution_census") or {}).get(
                    "executed_type_counts") is None and
                not result.get("logit_checkpoints")
                for result in results.values())),
      check("memory_guard_holds_for_every_serial_worker",
            all(
                (run.get("memory_guard") or {}).get("tripped") is False
                for run in runs.values()) and
            minimum_available >= stop_bytes,
            peak_rss_bytes=peak_rss, peak_swap_bytes=peak_swap,
            minimum_available_bytes=minimum_available,
            stop_bytes=stop_bytes),
  ]
  passed = all(row["pass"] for row in checks)
  verdict = (
      "admit_qk_rope_stock_half_output512_correctness_and_formal_design"
      if passed else
      "reject_qk_rope_stock_half_before_formal_performance")
  payload = {
      "schema": SCHEMA,
      "workstream": WS,
      "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
      "git": git,
      "verdict": verdict,
      "required_checks_passed": passed,
      "output512_correctness_gate_admitted": passed,
      "formal_incremental_gate_design_admitted": passed,
      "formal_product_promotion_admitted": False,
      "performance_claim_admitted": False,
      "schedule": list(SCHEDULE),
      "workers_concurrent": False,
      "checks": checks,
      "phases": phases,
      "tails": tails,
      "tail_saving_ms": tail_saving_ms,
      "runs": runs,
      "next_action": {
          "route": "openvino_qk_rope_stock_half_output512_correctness",
          "requirements": [
              "run one isolated output512 full-logit candidate",
              "require exact tokens, KLD at most 0.005, and top1 at least 0.99",
              "reuse this ABBA1 only as block zero of a formal gate",
              "make no speed claim before eight paired incremental blocks",
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
      "schedule": list(SCHEDULE),
      "gpu_workers": 4,
      "control_workers": 2,
      "qk_workers": 2,
      "stock_workers": 0,
      "workers_concurrent": False,
  })
  report = f"""# Stock-half Q/K RoPE incremental ABBA precheck

Verdict: **{verdict}**. Required checks: `{str(passed).lower()}`.

All four output512 token streams are exact. Q/K-over-control
prefill/decode/total point ratios are
`{ratio_text(phases.get('prefill_tokens_s', {}).get('ratio'))}/`
`{ratio_text(phases.get('decode_tokens_s', {}).get('ratio'))}/`
`{ratio_text(phases.get('total_rate', {}).get('ratio'))}x`.
Stable-tail control/QK medians are
`{ratio_text(control_tail_ms)}/{ratio_text(qk_tail_ms)} ms`, a
`{ratio_text(tail_saving_ms)}-ms/token` saving.

All four workers are isolated and serial. Peak RSS/swap is
`{peak_rss}/{peak_swap} B`; minimum available memory is
`{minimum_available} B`. ABBA1 admits no formal performance claim.
"""
  (out / "report.md").write_text(report, encoding="utf-8")
  print(json.dumps({
      "artifact": PRODUCT.relative(out),
      "verdict": verdict,
      "required_checks_passed": passed,
      "phase_ratios": {
          key: value.get("ratio") for key, value in phases.items()},
      "tail_saving_ms": tail_saving_ms,
      "peak_rss_bytes": peak_rss,
      "peak_swap_bytes": peak_swap,
  }, separators=(",", ":")), flush=True)
  return 0 if passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
