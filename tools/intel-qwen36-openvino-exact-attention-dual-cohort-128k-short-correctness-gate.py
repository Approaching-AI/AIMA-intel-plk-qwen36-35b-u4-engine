#!/usr/bin/env python3
"""Prove the all-ten dual-cohort owner on a 128k recurrent boundary."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = (
    "intel-qwen36-openvino-exact-attention-dual-cohort-128k-"
    "short-correctness-gate-v1")
PRODUCT_TOOL = ROOT / "tools/intel-qwen36-openvino-hot-cold-product-gate.py"
GRAPH_TOOL = ROOT / "tools/intel_qwen36_openvino_hot_cold_attention.py"
KERNEL = ROOT / "engine/openvino/custom/iq36_stock_micro_attention_oracle.cl"
CONFIG = ROOT / "engine/openvino/custom/iq36_hot_attention_gqa.xml"
COMPILE_GATE = ROOT / (
    "output/openvino-exact-attention-dual-cohort-all10-compile-"
    "20260723Tseq2137-clean/gate.json")
ACCEPTED_CORRECTNESS = ROOT / (
    "output/openvino-lm-head-gated-exact-count25-"
    "20260723Tseq2121-all10-128k-o512-abba1/correctness.json")
ACCEPTED_STOCK = ROOT / (
    "output/openvino-lm-head-gated-exact-count25-"
    "20260723Tseq2121-all10-128k-o512-abba1/raw/sentinel_128k/"
    "correctness/stock/worker-result.json")
ACCEPTED_CANDIDATE = ROOT / (
    "output/openvino-lm-head-gated-exact-count25-"
    "20260723Tseq2121-all10-128k-o512-abba1/raw/sentinel_128k/"
    "correctness/candidate/worker-result.json")
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
EXPECTED_TOKEN_IDS = [271, 248068, 271, 248069]
EXPECTED_STOCK_LOGIT_SHA256 = {
    0: "011d8999814fa4a047c5bc2e1b7e6f0d1f6560e1d019735a520a5b0ef3d8046e",
    1: "164803dcc5fc05f6bea129d1976b8d810147de744767921433a53d82dd21f82e",
    2: "8cef5902fd96b3d8643f8f0cfabd2cd6a66e9e01d66e9b91ca1943caac104309",
    3: "871f0bc0ef13cb7ed711b382b9bb27688e4e98f67dabe0476457d237d5a2e539",
}
EXPECTED_SINGLE_COHORT_LOGIT_SHA256 = {
    0: "011d8999814fa4a047c5bc2e1b7e6f0d1f6560e1d019735a520a5b0ef3d8046e",
    1: "c4b083d2f1e7e2bf5a09adaace1b5582a3a5d11a8aa9dbbb64f2180001f3228f",
    2: "758c39ad633305adea8637f425ab4a66bba266c914a234cd4c03494fcf8a53b9",
    3: "36e7ffe137edd67366a9c35ff3a95f88123f07c8461fab227c6f5a98bac0c746",
}
LAYERS = (3, 7, 11, 15, 19, 23, 27, 31, 35, 39)
BUCKET = 131072
EXACT_HISTORY_CAPACITY = 132096
OUTPUT_TOKENS = 4
PREFLIGHT_GIB = 8.0
MEMORY_STOP_GIB = 4.0


def load_module(name: str, path: Path) -> Any:
  spec = importlib.util.spec_from_file_location(name, path)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import {path}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


PRODUCT = load_module("iq36_dual_cohort_correctness_product", PRODUCT_TOOL)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out-dir", type=Path, required=True)
  parser.add_argument("--timeout-s", type=int, default=3600)
  parser.add_argument("--allow-dirty", action="store_true")
  args = parser.parse_args()
  if args.timeout_s <= 0:
    parser.error("--timeout-s must be positive")
  return args


def check(name: str, passed: bool, **details: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **details}


def file_sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def checkpoint_shas(result: dict[str, Any]) -> dict[int, str]:
  return {
      int(row["step"]): str(row["sha256"])
      for row in result.get("distribution_checkpoints", [])
  }


def memory_summary(run: dict[str, Any]) -> dict[str, Any]:
  monitor = run.get("monitor") or {}
  return {
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
  if out.exists():
    raise SystemExit(f"output already exists: {out}")
  raw = out / "raw"
  raw.mkdir(parents=True, exist_ok=False)

  required_paths = (
      PRODUCT_TOOL, GRAPH_TOOL, KERNEL, CONFIG, COMPILE_GATE,
      ACCEPTED_CORRECTNESS, ACCEPTED_STOCK, ACCEPTED_CANDIDATE, PROMPT,
      PLUGIN)
  missing = [str(path) for path in required_paths if not path.is_file()]
  if missing:
    raise SystemExit(
        "missing dual-cohort correctness inputs: " + ", ".join(missing))

  git = PRODUCT.BOOT.git_state(out)
  compile_gate = PRODUCT.load_json(COMPILE_GATE)
  accepted_correctness = PRODUCT.load_json(ACCEPTED_CORRECTNESS)
  accepted_stock = PRODUCT.load_json(ACCEPTED_STOCK)
  accepted_candidate = PRODUCT.load_json(ACCEPTED_CANDIDATE)
  plugin_sha = PRODUCT.sha256_file(PLUGIN)
  prompt_sha = file_sha256(PROMPT)
  model_identity = PRODUCT.BOOT.capture_model_identity(
      PRODUCT.MODEL_DIR.resolve(), PRODUCT.MODEL_CONTRACT.resolve())
  PRODUCT.write_json(out / "model-identity.json", model_identity)
  PRODUCT.write_json(out / "host.json", PRODUCT.BOOT.capture_host())

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
  case = {
      **adaptive_defaults,
      "alias_linear_state_assign": True,
      "bucket": BUCKET,
      "candidate_dq_realloc_fastpath": True,
      "candidate_fc_stable_prepare_fastpath": True,
      "candidate_impls_cache_capacity": None,
      "candidate_path": "hot_cold_custom",
      "capture_all_correctness_logits": False,
      "capture_attention_history_layers": [],
      "capture_attention_history_steps": [],
      "capture_attention_layers": [],
      "capture_attention_steps": [],
      "capture_lm_head_hidden": False,
      "case_id": "sentinel_128k_dual_cohort_short",
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
      "expected_tokens": BUCKET,
      "fixed_fc_cohorts": [],
      "fixed_fc_manager_direct": False,
      "fixed_fc_manager_scope": "all",
      "fuse_fixed_fc": False,
      "fuse_linear_conv_state": True,
      "linear_state_alias_scope": "all",
      "lm_head_device_greedy_feedback": False,
      "lm_head_i8q1": True,
      "lm_head_i8q1_gated_exact": True,
      "lm_head_i8q1_gated_q4": False,
      "lm_head_i8q1_greedy_local2": True,
      "lm_head_i8q4": False,
      "lm_head_token_only_feedback": True,
      "pack_gdn_state": False,
      "path": str(PROMPT.resolve()),
      "prefill_history_capacity": BUCKET,
      "prime_candidate_exact_decode_shape": False,
      "prompt_set": "sentinel",
      "sha256": EXPECTED_PROMPT_SHA256,
      "target_layers": list(LAYERS),
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
      # Exposing ten stock Q/K/V/attention boundaries at 128k inhibits the
      # denominator's fused memory plan and is not required here.  The four
      # full-logit fingerprints below are a stricter end-to-end equivalence
      # check for totals 131072 through 131075.
      "capture_attention_layers": [],
      "capture_attention_steps": [],
      "capture_execution_census": True,
      "capture_lm_head_hidden": False,
      "capture_prefill_profiles": False,
      "case_id": case["case_id"],
      "checkpoint_steps": list(range(OUTPUT_TOKENS)),
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
      "expected_answer": case["expected_answer"],
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
      "prefill_chunk_tokens": PRODUCT.FROZEN_CHUNK_TOKENS,
      "prefill_history_capacity": BUCKET,
      "prime_candidate_exact_decode_shape": False,
      "prompt": str(PROMPT.resolve()),
      "self_bind_hot_states": False,
      "target_layers": list(LAYERS),
      "warmup": True,
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
  )

  stock_dir = raw / "stock"
  stock_run = PRODUCT.run_worker(worker_args, stock_dir, {
      **common,
      "candidate_path": "stock_sdpa",
      "capture_logits": True,
      "mode": "stock",
      "purpose": "correctness_reference",
      "reference_result": None,
      "timing_token_output": False,
  })
  stock_run["worker"] = PRODUCT.relative(stock_dir)
  candidate_run: dict[str, Any] = {}
  if (stock_run.get("returncode") == 0 and
      stock_run.get("timed_out") is False and
      stock_run.get("oom_observed") is False):
    candidate_dir = raw / "candidate"
    candidate_run = PRODUCT.run_worker(worker_args, candidate_dir, {
        **common,
        "candidate_path": "hot_cold_custom",
        "capture_logits": True,
        "mode": "candidate",
        "purpose": "teacher_forced_correctness",
        "reference_result": str(
            (stock_dir / "worker-result.json").resolve()),
        "timing_token_output": False,
    })
    candidate_run["worker"] = PRODUCT.relative(candidate_dir)

  canonical = PRODUCT.correctness_for_case(
      case, stock_run, candidate_run)
  stock = stock_run.get("result") or {}
  candidate = candidate_run.get("result") or {}
  stock_shas = checkpoint_shas(stock)
  candidate_shas = checkpoint_shas(candidate)
  source = candidate.get("source_summary") or {}
  runtime = candidate.get("runtime_census") or {}
  execution = candidate.get("execution_census") or {}
  execution_counts = execution.get("executed_type_counts") or {}
  all_monitor_rows = [
      run.get("monitor") or {}
      for run in (stock_run, candidate_run) if run]
  stop_bytes = int(MEMORY_STOP_GIB * 1024**3)
  minimum_available = min(
      (int(row.get("system_available_min_bytes") or 0)
       for row in all_monitor_rows),
      default=0)

  checks = [
      check("repository_clean_at_gate",
            not git["dirty"] or args.allow_dirty,
            git=git, allow_dirty=args.allow_dirty),
      check("all_ten_compile_gate_admits_short_correctness",
            compile_gate.get("required_checks_passed") is True
            and compile_gate.get("correctness_worker_admitted") is True
            and compile_gate.get("verdict") ==
                "admit_exact_attention_dual_cohort_short_recurrent_correctness"
            and compile_gate.get("git", {}).get("dirty") is False
            and compile_gate.get("sources", {}).get(
                PRODUCT.relative(KERNEL)) == PRODUCT.sha256_file(KERNEL)
            and compile_gate.get("sources", {}).get(
                PRODUCT.relative(CONFIG)) == PRODUCT.sha256_file(CONFIG)
            and compile_gate.get("sources", {}).get(
                PRODUCT.relative(PRODUCT_TOOL)) ==
                PRODUCT.sha256_file(PRODUCT_TOOL)
            and compile_gate.get("sources", {}).get(
                PRODUCT.relative(GRAPH_TOOL)) ==
                PRODUCT.sha256_file(GRAPH_TOOL)),
      check("accepted_128k_single_cohort_correctness_is_bound",
            accepted_correctness.get("required_checks_passed") is True
            and accepted_correctness.get("cases", [{}])[0].get(
                "required_checks_passed") is True
            and checkpoint_shas(accepted_stock) ==
                {int(row["step"]): str(row["sha256"])
                 for row in accepted_stock.get(
                     "distribution_checkpoints", [])}
            and all(
                checkpoint_shas(accepted_stock).get(step) == sha
                for step, sha in EXPECTED_STOCK_LOGIT_SHA256.items())
            and all(
                checkpoint_shas(accepted_candidate).get(step) == sha
                for step, sha in
                EXPECTED_SINGLE_COHORT_LOGIT_SHA256.items())),
      check("accepted_seq2119_plugin_and_locked_prompt_are_exact",
            plugin_sha == EXPECTED_PLUGIN_SHA256
            and prompt_sha == EXPECTED_PROMPT_SHA256,
            plugin_sha256=plugin_sha, prompt_sha256=prompt_sha),
      check("locked_model_identity",
            model_identity.get("required_checks_passed") is True,
            model_dir=model_identity.get("model_dir")),
      check("stock_then_candidate_workers_execute_strictly_serially",
            stock_run.get("returncode") == 0
            and bool(candidate_run)
            and candidate_run.get("returncode") == 0
            and stock_run.get("timed_out") is False
            and candidate_run.get("timed_out") is False,
            sequence=["stock", "candidate"],
            stock_elapsed_seconds=stock_run.get("elapsed_seconds"),
            candidate_elapsed_seconds=candidate_run.get("elapsed_seconds")),
      check("canonical_product_correctness_gate_passes",
            canonical.get("required_checks_passed") is True,
            canonical_failed_checks=[
                row for row in canonical.get("checks", [])
                if row.get("required", True) and row.get("pass") is not True]),
      check("stock_short_fingerprint_reproduces_accepted_denominator",
            stock_shas == EXPECTED_STOCK_LOGIT_SHA256
            and stock.get("generated_token_ids") == EXPECTED_TOKEN_IDS,
            observed=stock_shas),
      check("dual_cohort_matches_accepted_single_cohort_bitwise",
            candidate_shas == EXPECTED_SINGLE_COHORT_LOGIT_SHA256
            and candidate.get("generated_token_ids") == EXPECTED_TOKEN_IDS,
            observed=candidate_shas),
      check("dynamic_partial_tails_have_complete_end_to_end_fingerprints",
            sorted(candidate_shas) == list(range(OUTPUT_TOKENS))
            and candidate_shas == EXPECTED_SINGLE_COHORT_LOGIT_SHA256
            and len(canonical.get("distribution_rows", [])) ==
                OUTPUT_TOKENS
            and all(
                row.get("finite") is True
                and row.get("same_shape") is True
                for row in canonical.get("distribution_rows", [])),
            exercised_totals=list(
                range(BUCKET, BUCKET + OUTPUT_TOKENS)),
            observed=candidate_shas),
      check("candidate_source_and_runtime_own_exactly_ten_dual_cohort_layers",
            source.get("target_layers") == list(LAYERS)
            and source.get("decode_stock_micro_layers") == list(LAYERS)
            and source.get("exact_phase_decode") is True
            and source.get("exact_phase_dual_cohort") is True
            and source.get("exact_phase_context_partition4") is False
            and source.get("decode_page_sparse_layers") == []
            and source.get("exact_history_layers") == list(LAYERS)
            and source.get("exact_history_capacity") ==
                EXACT_HISTORY_CAPACITY
            and source.get("prefill_history_capacity") == BUCKET
            and source.get("custom_count_after") == len(LAYERS)
            and source.get("stock_sdpa_count_after") == 0
            and runtime.get("hot_attention_custom_count") == len(LAYERS)
            and runtime.get("stock_sdpa_like_count") == 0),
      check("executed_attention_type_is_exclusively_dual_cohort",
            execution_counts.get(
                "IQ36ExactPhaseDualCohortHotAttentionGQA") == len(LAYERS)
            and execution_counts.get("IQ36ExactPhaseHotAttentionGQA", 0) == 0
            and execution_counts.get(
                "IQ36ExactPhaseContextPartition4HotAttentionGQA", 0) == 0
            and execution_counts.get(
                "ScaledDotProductAttention", 0) == 0
            and execution_counts.get("IndirectSDPA", 0) == 0,
            executed_type_counts=execution_counts),
      check("candidate_uses_accepted_product_carrier",
            candidate.get("candidate_gpu_plugin_sha256") == plugin_sha
            and candidate.get("custom_composition") == "exact_phase"
            and candidate.get("fuse_linear_conv_state") is True
            and candidate.get("alias_linear_state_assign") is True
            and candidate.get("linear_state_alias_scope") == "all"
            and candidate.get("candidate_dq_realloc_fastpath") is True
            and candidate.get("candidate_fc_stable_prepare_fastpath") is True
            and candidate.get("lm_head_i8q1") is True
            and candidate.get("lm_head_i8q1_gated_exact") is True),
      check("both_workers_record_memory_telemetry",
            len(all_monitor_rows) == 2
            and all(
                int(row.get("process_rss_peak_bytes", -1)) >= 0
                and int(row.get("process_swap_peak_bytes", -1)) >= 0
                for row in all_monitor_rows),
            stock=memory_summary(stock_run),
            candidate=memory_summary(candidate_run)),
      check("no_oom_and_established_memory_guard_never_trips",
            stock_run.get("oom_observed") is False
            and candidate_run.get("oom_observed") is False
            and (stock_run.get("memory_guard") or {}).get("tripped") is False
            and (candidate_run.get("memory_guard") or {}).get(
                "tripped") is False
            and minimum_available >= stop_bytes,
            stop_bytes=stop_bytes,
            minimum_available_bytes=minimum_available),
  ]
  required = all(row["pass"] for row in checks)
  admitted = required and not args.allow_dirty
  verdict = (
      "admit_exact_attention_dual_cohort_paired_wall_probe"
      if admitted else
      "development_dual_cohort_short_correctness" if required else
      "repair_exact_attention_dual_cohort_recurrent_correctness")
  payload = {
      "schema_version": SCHEMA,
      "workstream": WS,
      "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
      "git": git,
      "verdict": verdict,
      "required_checks_passed": required,
      "short_correctness_admitted": admitted,
      "timing_worker_admitted": admitted,
      "speedup_claims_allowed": False,
      "checks": checks,
      "canonical_correctness": canonical,
      "model_identity": PRODUCT.relative(out / "model-identity.json"),
      "plugin": {"path": str(PLUGIN), "sha256": plugin_sha},
      "prompt": {"path": PRODUCT.relative(PROMPT), "sha256": prompt_sha},
      "case": case,
      "worker_sequence": ["stock", "candidate"],
      "workers": {
          "stock": stock_run,
          "candidate": candidate_run,
      },
      "next_route": (
          "openvino_exact_attention_dual_cohort_128k_o512_abba1_wall"
          if admitted else
          "openvino_exact_attention_dual_cohort_correctness_repair"),
      "sources": {
          PRODUCT.relative(path): PRODUCT.sha256_file(path)
          for path in (
              PRODUCT_TOOL, GRAPH_TOOL, KERNEL, CONFIG, COMPILE_GATE,
              ACCEPTED_CORRECTNESS)
      },
  }
  PRODUCT.write_json(out / "gate.json", payload)
  PRODUCT.write_json(out / "correctness.json", {
      "schema_version": SCHEMA,
      "workstream": WS,
      "required_checks_passed": required,
      "checks": checks,
      "canonical_correctness": canonical,
  })
  (out / "summary.md").write_text(
      "\n".join([
          "# Exact-attention dual-cohort 128k short correctness",
          "",
          f"- verdict: `{verdict}`",
          f"- required checks: `{'pass' if required else 'fail'}`",
          f"- generated token IDs: `{candidate.get('generated_token_ids')}`",
          f"- max teacher-forced KLD: `{canonical.get('kld_max')}`",
          f"- teacher-forced top-1 rate: `{canonical.get('top1_rate')}`",
          f"- recurrent logit rows: "
          f"`{len(canonical.get('distribution_rows', []))}`",
          f"- dual-cohort execution count: "
          f"`{execution_counts.get('IQ36ExactPhaseDualCohortHotAttentionGQA')}`",
          f"- stock memory: `{json.dumps(memory_summary(stock_run), sort_keys=True)}`",
          f"- candidate memory: "
          f"`{json.dumps(memory_summary(candidate_run), sort_keys=True)}`",
          "- timing executed: `false`",
          "",
      ]), encoding="utf-8")
  print(json.dumps({
      "output": PRODUCT.relative(out),
      "verdict": verdict,
      "required_checks_passed": required,
      "timing_worker_admitted": admitted,
      "generated_token_ids": candidate.get("generated_token_ids"),
      "kld_max": canonical.get("kld_max"),
      "top1_rate": canonical.get("top1_rate"),
      "dual_cohort_execution_count": execution_counts.get(
          "IQ36ExactPhaseDualCohortHotAttentionGQA"),
  }, sort_keys=True), flush=True)
  return 0 if required else 2


if __name__ == "__main__":
  raise SystemExit(main())
