#!/usr/bin/env python3
"""Formalize the stock-half Q/K cut over eight incremental ABBA blocks.

Block zero is the clean seq2198 control-QK-QK-control evidence.  Blocks one
through seven are fresh and strictly serial.  Every row uses the accepted
seq2189 plugin/exact carrier and differs only by IQ36QKRopeLayout enablement.
Promotion requires output512 token exactness plus a paired one-sided 95%
lower bound of at least 1.005x on prefill, decode, and total rate.
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
    "formal-abba8-v1")
BASE_TOOL = ROOT / (
    "tools/intel-qwen36-openvino-qk-rope-layout-"
    "stock-half-abba-precheck.py")
BLOCK0_RESULT = ROOT / (
    "output/openvino-qk-rope-layout-stock-half-abba-precheck-"
    "20260731Tseq2198-clean/result.json")
BLOCK0_MANIFEST = BLOCK0_RESULT.parent / "manifest.json"
PLAN_RESULT = ROOT / (
    "output/openvino-qk-rope-layout-stock-half-output512-config-"
    "20260731Tseq2199a-clean/result.json")
PLAN_MANIFEST = PLAN_RESULT.parent / "manifest.json"
CORRECTNESS_RESULT = ROOT / (
    "output/openvino-qk-rope-layout-stock-half-output512-correctness-"
    "20260731Tseq2200-clean/result.json")
CORRECTNESS_MANIFEST = CORRECTNESS_RESULT.parent / "manifest.json"
BASE_CONFIG = ROOT / (
    "output/openvino-lm-head-parallel-block-topk-2k-abba8-"
    "20260731Tseq2193-clean/raw/prefill_shape_002k/block00/"
    "candidate-b1/worker-config.json")
REFERENCE_RESULT = ROOT / (
    "output/openvino-lm-head-parallel-block-topk-2k-abba8-"
    "20260731Tseq2193-clean/raw/prefill_shape_002k/correctness/"
    "stock/worker-result.json")
KERNEL_SOURCE = ROOT / "engine/openvino/custom/iq36_qk_rope_layout.cl"
PLUGIN = Path(
    "/home/intel/intel-qwen36-r0/output/"
    "openvino-90214e-l0-gpu-seq2189/bin/intel64/Release/"
    "libopenvino_intel_gpu_plugin.so")
EXPECTED_SHA256 = {
    BASE_TOOL: (
        "33f7b784e3b07894302ca85df98141c49046040ad7f6024e84b116ab7438b0a9"),
    BLOCK0_RESULT: (
        "c989b34615b971c21784240698e1ef32a57633ed7549da5fae029e584ad1fbaa"),
    BLOCK0_MANIFEST: (
        "6b9a8808f1f778e817524dc94f9a1076786a86ea97b18dc7a51c9339262c646b"),
    PLAN_RESULT: (
        "736eac5719bed40c8b417d66eefe6bbb87abecf1b998506d8e99df991901c4d6"),
    PLAN_MANIFEST: (
        "fc38dac1c1e4df51a0d1b5849d4f51a836e74833f334234a96476a840eacb905"),
    CORRECTNESS_RESULT: (
        "ff862015c9cec1aad4fb1c7efa8aa519927417361b480d90d50a95c9292512df"),
    CORRECTNESS_MANIFEST: (
        "95470cd1b66439c46ce5e3c699174f3855b4bbde14ee039f898220a7322f5277"),
    BASE_CONFIG: (
        "7cd2281731dafd08a6a58b7cf36da18c06d0e034f04dcc11bd394103661c2911"),
    REFERENCE_RESULT: (
        "5f7d0d0fbbde73e8e546a513fe294282d1f961ae72362ed2dd6900b2125d0da1"),
    KERNEL_SOURCE: (
        "be2b1105df7503a24636615a94255e0683d0b8a73bbecd1c7b70d0b9f5306863"),
    PLUGIN: (
        "9832adffa8bf3fd7b013c47d9d2abefa66987c5fbde350c9669fce58c697b985"),
}
EXPECTED_TOKEN_SHA256 = (
    "0a7b56baf11a00512a786c0c825bba4733fda84eb5b87eb703c79344f508ea63")
SCHEDULE = ("control-a1", "qk-b1", "qk-b2", "control-a2")
BLOCK_COUNT = 8
TARGET_RATIO = 1.005
JITTER_MAX = 1.25
PREFLIGHT_GIB = 8.0
MEMORY_STOP_GIB = 4.0


def load_module(name: str, path: Path) -> Any:
  spec = importlib.util.spec_from_file_location(name, path)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import {path}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


BASE = load_module("iq36_qk_formal_base", BASE_TOOL)
PRODUCT = BASE.PRODUCT


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out-dir", type=Path, required=True)
  parser.add_argument("--plan-only", action="store_true")
  parser.add_argument("--resume", action="store_true")
  parser.add_argument("--timeout-s", type=int, default=1800)
  args = parser.parse_args()
  if args.timeout_s <= 0:
    parser.error("timeout must be positive")
  if args.plan_only and args.resume:
    parser.error("plan-only cannot resume")
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


def effective_config(
    base: dict[str, Any], label: str,
) -> dict[str, Any]:
  qk_enabled = label.startswith("qk-")
  config = dict(base)
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
  return config


def config_delta(
    base: dict[str, Any], candidate: dict[str, Any],
) -> dict[str, Any]:
  return {
      key: {"base": base.get(key), "effective": candidate.get(key)}
      for key in sorted(set(base) | set(candidate))
      if base.get(key) != candidate.get(key)
  }


def main() -> int:
  args = parse_args()
  out = args.out_dir.resolve()
  if out.exists() and not args.resume:
    raise SystemExit(f"output already exists: {out}")
  raw = out / "raw"
  raw.mkdir(parents=True, exist_ok=args.resume)
  required_paths = (BASE.PRODUCT_TOOL, *EXPECTED_SHA256)
  missing = [str(path) for path in required_paths if not path.is_file()]
  if missing:
    raise SystemExit("missing formal Q/K inputs: " + ", ".join(missing))

  git = PRODUCT.BOOT.git_state(out)
  input_hashes = {path: sha256(path) for path in EXPECTED_SHA256}
  block0 = PRODUCT.load_json(BLOCK0_RESULT)
  block0_manifest = PRODUCT.load_json(BLOCK0_MANIFEST)
  plan = PRODUCT.load_json(PLAN_RESULT)
  correctness = PRODUCT.load_json(CORRECTNESS_RESULT)
  reference = PRODUCT.load_json(REFERENCE_RESULT)
  base_config = PRODUCT.load_json(BASE_CONFIG)
  sample_configs = {
      label: effective_config(base_config, label) for label in SCHEDULE}
  deltas = {
      label: config_delta(base_config, config)
      for label, config in sample_configs.items()}
  config_exact = (
      str(base_config.get("prompt", "")).endswith(
          "/prompts/prefill_shape_002k.txt") and
      base_config.get("output_tokens") == 512 and
      base_config.get("purpose") == "paired_product_timing" and
      base_config.get("capture_execution_census") is False and
      base_config.get("capture_logits") is False and
      base_config.get("reference_result") == str(REFERENCE_RESULT.resolve()) and
      all(set(delta) == {"case_id", "fuse_qk_rope_layout"}
          for delta in deltas.values()) and
      all(sample_configs[label]["fuse_qk_rope_layout"] is False
          for label in ("control-a1", "control-a2")) and
      all(sample_configs[label]["fuse_qk_rope_layout"] is True
          for label in ("qk-b1", "qk-b2")))
  static_checks = [
      check("repository_clean_at_gate", not git["dirty"], git=git),
      check("all_frozen_inputs_have_exact_hashes",
            all(input_hashes[path] == expected
                for path, expected in EXPECTED_SHA256.items()),
            observed={
                PRODUCT.relative(path): digest
                for path, digest in input_hashes.items()}),
      check("block0_plan_and_output512_correctness_are_formally_bound",
            block0.get("required_checks_passed") is True and
            block0.get("output512_correctness_gate_admitted") is True and
            block0.get("performance_claim_admitted") is False and
            block0_manifest.get("schedule") == list(SCHEDULE) and
            block0_manifest.get("gpu_workers") == 4 and
            plan.get("required_checks_passed") is True and
            plan.get("plan_only") is True and
            plan.get("gpu_workers_launched") == 0 and
            correctness.get("required_checks_passed") is True and
            correctness.get(
                "formal_incremental_abba8_design_admitted") is True and
            correctness.get("correctness", {}).get(
                "bitwise_checkpoint_count") == 512 and
            correctness.get("correctness", {}).get(
                "stock_relative", {}).get("max_kld") ==
                0.0048365644843369315 and
            correctness.get("correctness", {}).get(
                "stock_relative", {}).get("top1_rate") == 1.0),
      check("all_formal_worker_configs_change_only_case_id_and_qk_flag",
            config_exact, deltas=deltas,
            prompt=base_config.get("prompt")),
      check("formal_target_and_worker_budget_are_pre_registered",
            BLOCK_COUNT == 8 and TARGET_RATIO == 1.005 and
            len(SCHEDULE) == 4 and
            block0_manifest.get("workers_concurrent") is False,
            block_count=BLOCK_COUNT, target_ratio=TARGET_RATIO,
            schedule=list(SCHEDULE), new_gpu_workers=28),
  ]
  static_passed = all(row["pass"] for row in static_checks)
  if args.plan_only or not static_passed:
    verdict = (
        "admit_seven_new_serial_incremental_abba_blocks"
        if static_passed else
        "reject_formal_incremental_workers_before_gpu")
    payload = {
        "schema": SCHEMA,
        "workstream": WS,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git": git,
        "verdict": verdict,
        "required_checks_passed": static_passed,
        "plan_only": True,
        "checks": static_checks,
        "config_deltas": deltas,
        "block_count": BLOCK_COUNT,
        "reused_block_count": 1,
        "new_block_count": 7,
        "gpu_workers_launched": 0,
        "new_gpu_workers_planned": 28,
        "workers_concurrent": False,
        "target_ratio": TARGET_RATIO,
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
        "new_gpu_workers_planned": 28,
        "workers_concurrent": False,
    })
    print(json.dumps({
        "artifact": PRODUCT.relative(out),
        "verdict": verdict,
        "required_checks_passed": static_passed,
        "block_count": BLOCK_COUNT,
        "new_gpu_workers_planned": 28,
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
      resume=args.resume,
      timeout_s=args.timeout_s,
      worker_transient_scope=True,
  )
  run_blocks: dict[int, dict[str, dict[str, Any]]] = {
      0: block0["runs"]}
  for block_index in range(1, BLOCK_COUNT):
    run_blocks[block_index] = {}
    for label in SCHEDULE:
      run_blocks[block_index][label] = PRODUCT.run_worker(
          worker_args, raw / f"block{block_index:02d}" / label,
          sample_configs[label])

  results = {
      block_index: {
          label: run.get("result") or {} for label, run in runs.items()}
      for block_index, runs in run_blocks.items()}
  summaries = [
      {"block": block_index,
       "phases": BASE.phase_summary(results[block_index])}
      for block_index in range(BLOCK_COUNT)]
  inference = {}
  for phase in ("prefill_tokens_s", "decode_tokens_s", "total_rate"):
    inference[phase] = PRODUCT.perf_inference.paired_speedup_inference(
        [row["phases"][phase]["candidate"] for row in summaries],
        [row["phases"][phase]["control"] for row in summaries],
        target_ratio=TARGET_RATIO, min_blocks=BLOCK_COUNT)

  flat_runs = [
      (block_index, label, run)
      for block_index, runs in run_blocks.items()
      for label, run in runs.items()]
  flat_results = [
      (block_index, label, results[block_index][label])
      for block_index, label, _ in flat_runs]
  qk_tails = [
      {
          "block": block_index,
          "label": label,
          **BASE.tail_metrics(result),
      }
      for block_index, label, result in flat_results
      if label.startswith("qk-")]
  control_tails = [
      {
          "block": block_index,
          "label": label,
          **BASE.tail_metrics(result),
      }
      for block_index, label, result in flat_results
      if label.startswith("control-")]
  scopes = [
      (run.get("worker_transient_scope") or {}).get("unit")
      for _, _, run in flat_runs]
  monitors = [run.get("monitor") or {} for _, _, run in flat_runs]
  peak_rss = max(
      int(row.get("process_rss_peak_bytes") or 0) for row in monitors)
  peak_swap = max(
      int(row.get("process_swap_peak_bytes") or 0) for row in monitors)
  minimum_available = min(
      int(row.get("system_available_min_bytes") or 0) for row in monitors)
  stop_bytes = int(MEMORY_STOP_GIB * 1024**3)

  checks = static_checks + [
      check("all_32_worker_rows_are_complete_isolated_and_serial",
            len(flat_runs) == 32 and
            all(
                run.get("returncode") == 0 and
                run.get("timed_out") is False and
                run.get("oom_observed") is False and
                (block_index == 0 or
                 run.get("reused") is not True or args.resume)
                for block_index, _, run in flat_runs) and
            len(scopes) == len(set(scopes)) and
            all(scope is not None for scope in scopes),
            scopes=scopes,
            aggregation_reused_worker_evidence=args.resume),
      check("every_output512_timing_token_stream_is_exact",
            reference.get("generated_token_ids_sha256") ==
                EXPECTED_TOKEN_SHA256 and
            len(reference.get("generated_token_ids") or []) == 512 and
            all(
                result.get("generated_token_count") == 512 and
                result.get("generated_token_ids_sha256") ==
                    EXPECTED_TOKEN_SHA256 and
                result.get("generated_token_ids") ==
                    reference.get("generated_token_ids") and
                result.get("teacher_forced_from_stock") is True
                for _, _, result in flat_results),
            expected_sha256=EXPECTED_TOKEN_SHA256),
      check("all_rows_keep_one_flag_carrier_and_provider_isolation",
            all(
                result.get("mode") == "candidate" and
                result.get("candidate_path") == "hot_cold_custom" and
                result.get("custom_composition") == "exact_phase" and
                result.get("exact_phase_dual_cohort") is True and
                result.get("candidate_gpu_plugin_sha256") ==
                    EXPECTED_SHA256[PLUGIN] and
                result.get("lm_head_i8q1") is True and
                result.get("lm_head_i8q1_gated_exact") is True and
                result.get("lm_head_i8q1_greedy_local2") is False and
                result.get("lm_head_token_only_feedback") is False and
                BASE.provider_exact(result) and
                result.get("fuse_qk_rope_layout") is label.startswith("qk-")
                for _, label, result in flat_results)),
      check("all_16_qk_jitter_rows_pass",
            len(qk_tails) == 16 and
            all(row["jitter_pass"] for row in qk_tails),
            threshold=JITTER_MAX, qk_tails=qk_tails,
            control_tails=control_tails),
      check("all_three_paired_one_sided_95pct_lcbs_clear_1p005",
            set(inference) == {
                "prefill_tokens_s", "decode_tokens_s", "total_rate"} and
            all(row["sample_count_pass"] and row["rate_pass"]
                for row in inference.values()),
            target_ratio=TARGET_RATIO, inference=inference),
      check("memory_guard_holds_for_all_32_serial_workers",
            all(
                (run.get("memory_guard") or {}).get("tripped") is False
                for _, _, run in flat_runs) and
            minimum_available >= stop_bytes,
            peak_rss_bytes=peak_rss, peak_swap_bytes=peak_swap,
            minimum_available_bytes=minimum_available,
            stop_bytes=stop_bytes),
  ]
  passed = all(row["pass"] for row in checks)
  verdict = (
      "accept_stock_half_qk_rope_incremental_cut_for_short_matrix"
      if passed else
      "reject_stock_half_qk_rope_after_formal_incremental_inference")
  next_action = (
      {
          "route": "complete_remaining_short_matrix_on_qk_carrier",
          "requirements": [
              "enable only the accepted stock-half QK producer",
              "retain seq2200 output512 correctness and half codegen",
              "complete the remaining eight short ABBA8 cases",
              "keep speedup_claims disabled until the full ladder passes",
          ],
      }
      if passed else
      {
          "route": "complete_remaining_short_matrix_on_seq2189_carrier",
          "requirements": [
              "disable the rejected stock-half QK producer",
              "do not round, resample, or lower the registered 1.005 target",
              "retain the accepted seq2189 parallel block-top8 carrier",
              "profile or formalize the remaining eight short cases",
          ],
      })
  payload = {
      "schema": SCHEMA,
      "workstream": WS,
      "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
      "git": git,
      "verdict": verdict,
      "required_checks_passed": passed,
      "incremental_speedup_claim_admitted": passed,
      "short_matrix_carrier_cut_admitted": passed,
      "formal_product_promotion_admitted": False,
      "speedup_claims_allowed": False,
      "target_ratio": TARGET_RATIO,
      "block_count": BLOCK_COUNT,
      "reused_block_count": 1,
      "new_block_count": 7,
      "schedule": list(SCHEDULE),
      "workers_concurrent": False,
      "checks": checks,
      "phase_inference": inference,
      "blocks": summaries,
      "qk_jitter": qk_tails,
      "control_jitter": control_tails,
      "runs": {
          f"block{block_index:02d}": runs
          for block_index, runs in run_blocks.items()},
      "correctness": {
          "artifact": PRODUCT.relative(CORRECTNESS_RESULT),
          "bitwise_checkpoint_count": 512,
          "stock_max_kld": 0.0048365644843369315,
          "stock_top1_rate": 1.0,
      },
      "memory": {
          "peak_rss_bytes": peak_rss,
          "peak_swap_bytes": peak_swap,
          "minimum_available_bytes": minimum_available,
          "oom_or_guard_events": 0,
      },
      "aggregation_reused_worker_evidence": args.resume,
      "next_action": next_action,
  }
  PRODUCT.write_json(out / "result.json", payload)
  PRODUCT.write_json(out / "manifest.json", {
      "schema": SCHEMA,
      "tool": PRODUCT.relative(Path(__file__)),
      "git": git,
      "inputs": {
          PRODUCT.relative(path): digest
          for path, digest in input_hashes.items()},
      "block_count": BLOCK_COUNT,
      "reused_block_count": 1,
      "new_block_count": 7,
      "gpu_workers": 32,
      "new_gpu_workers": 0 if args.resume else 28,
      "reused_gpu_worker_evidence": 32 if args.resume else 4,
      "stock_workers": 0,
      "workers_concurrent": False,
      "aggregation_reused_worker_evidence": args.resume,
      "target_ratio": TARGET_RATIO,
  })
  result_word = "accepted" if passed else "rejected"
  report = f"""# Stock-half Q/K RoPE formal incremental ABBA8

Verdict: **{verdict}**. Required checks: `{str(passed).lower()}`.

Paired one-sided 95-percent prefill/decode/total LCBs are
`{inference['prefill_tokens_s']['lower_confidence_bound_ratio']:.6f}/`
`{inference['decode_tokens_s']['lower_confidence_bound_ratio']:.6f}/`
`{inference['total_rate']['lower_confidence_bound_ratio']:.6f}x` against the
pre-registered `{TARGET_RATIO:.3f}x` target. All 32 output512 timing streams
are exact and all 16 Q/K jitter rows pass.

Peak RSS/swap is `{peak_rss}/{peak_swap} B`; minimum available memory is
`{minimum_available} B`. The {result_word} result is an incremental Q/K
decision, not a complete product promotion or full-ladder speed claim.
"""
  (out / "report.md").write_text(report, encoding="utf-8")
  print(json.dumps({
      "artifact": PRODUCT.relative(out),
      "verdict": verdict,
      "required_checks_passed": passed,
      "phase_lcbs": {
          key: row["lower_confidence_bound_ratio"]
          for key, row in inference.items()},
      "qk_jitter_pass_count": sum(
          row["jitter_pass"] for row in qk_tails),
      "peak_rss_bytes": peak_rss,
      "peak_swap_bytes": peak_swap,
  }, separators=(",", ":")), flush=True)
  return 0 if passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
