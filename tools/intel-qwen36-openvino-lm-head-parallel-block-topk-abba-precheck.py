#!/usr/bin/env python3
"""Run one exact 2k prefill-shape ABBA block for the seq2189 plugin.

This bounded timing precheck reuses the accepted seq2185 stock token reference
and worker shape, launches exactly four fresh workers in strict ABBA order,
and tests whether both candidate p95/p50 rows now clear 1.25.  It cannot make
a formal performance claim because one paired block is below the ABBA8 gate.
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
    "intel-qwen36-openvino-lm-head-parallel-block-topk-"
    "abba-precheck-v1")
PRODUCT_TOOL = ROOT / "tools/intel-qwen36-openvino-hot-cold-product-gate.py"
CORRECTNESS_GATE = ROOT / (
    "output/openvino-lm-head-parallel-block-topk-product-precheck-"
    "20260731Tseq2191-clean/result.json")
REFERENCE_ROOT = ROOT / (
    "output/openvino-short-nonsentinel-auto-abba1-"
    "20260731Tseq2185-clean/raw/prefill_shape_002k")
REFERENCE_RESULT = REFERENCE_ROOT / "correctness/stock/worker-result.json"
REFERENCE_BLOCK = REFERENCE_ROOT / "block00"
PLUGIN = Path(
    "/home/intel/intel-qwen36-r0/output/"
    "openvino-90214e-l0-gpu-seq2189/bin/intel64/Release/"
    "libopenvino_intel_gpu_plugin.so")
EXPECTED_PLUGIN_SHA256 = (
    "9832adffa8bf3fd7b013c47d9d2abefa66987c5fbde350c9669fce58c697b985")
EXPECTED_REFERENCE_RESULT_SHA256 = (
    "b4378eacb59a03932633affc6d96c6f663bd2eaee2fb381d1ba5eb9efbb13e52")
EXPECTED_CONFIG_SHA256 = {
    "stock-a1":
        "733d24838f3f9095c124aa57a6fc5cad367c7c4d93f41bd5ea0791a12a44e882",
    "candidate-b1":
        "096d364892c098a7c89b22623143b5eb6826586d8dd57d3c3d2790d3e20a6390",
    "candidate-b2":
        "050395ab05223e6f768424638f22b472c1823febfd063281955dd07d6e7be61d",
    "stock-a2":
        "025d28fe2cff2fc7a080877dde2b9448c2a10ad4b6d5ecb0cc9c1e57091f86da",
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
SCHEDULE = ("stock-a1", "candidate-b1", "candidate-b2", "stock-a2")
JITTER_SKIP = 16
JITTER_MAX = 1.25
POINT_RATIO_MIN = 0.98
PREFLIGHT_GIB = 8.0
MEMORY_STOP_GIB = 4.0


def load_module(name: str, path: Path) -> Any:
  spec = importlib.util.spec_from_file_location(name, path)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import {path}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


PRODUCT = load_module("iq36_parallel_topk_abba_product", PRODUCT_TOOL)


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


def jitter(result: dict[str, Any]) -> dict[str, Any]:
  samples = [
      float(value) for value in
      result.get("decode_wall_ms", [])[JITTER_SKIP:]
  ]
  p50 = PRODUCT.percentile(samples, 0.50)
  p95 = PRODUCT.percentile(samples, 0.95)
  ratio = p95 / p50 if p50 and p95 else None
  return {
      "sample_count": len(samples),
      "p50_ms": p50,
      "p95_ms": p95,
      "p95_over_p50": ratio,
      "pass": (
          isinstance(ratio, (int, float)) and
          math.isfinite(float(ratio)) and float(ratio) <= JITTER_MAX),
  }


def finite_number(value: Any) -> bool:
  return (
      isinstance(value, (int, float)) and
      not isinstance(value, bool) and
      math.isfinite(float(value)))


def ratio_text(value: Any) -> str:
  return f"{float(value):.6f}" if finite_number(value) else "unavailable"


def provider_exact(result: dict[str, Any]) -> bool:
  trace = result.get("lm_head_i8q1_trace") or {}
  selections = trace.get("selection_rows") or []
  prepack = trace.get("weight_prepack_rows") or []
  return (
      len(selections) == 2 and len(prepack) == 2 and
      prepack[0].get("process_cache_hit") is False and
      prepack[1].get("process_cache_hit") is True and
      all(
          row.get("provider") == EXPECTED_PROVIDER and
          row.get("global") == [248320, 1, 1] and
          row.get("local") == [256, 1, 1] and
          row.get("correction_passes") == 2
          for row in selections))


def main() -> int:
  args = parse_args()
  out = args.out_dir.resolve()
  if out.exists() and not args.resume:
    raise SystemExit(f"output already exists: {out}")
  raw = out / "raw"
  raw.mkdir(parents=True, exist_ok=args.resume)
  config_paths = {
      label: REFERENCE_BLOCK / label / "worker-config.json"
      for label in SCHEDULE
  }
  result_paths = {
      label: REFERENCE_BLOCK / label / "worker-result.json"
      for label in SCHEDULE
  }
  required_paths = (
      PRODUCT_TOOL, CORRECTNESS_GATE, REFERENCE_RESULT, PLUGIN,
      *config_paths.values(), *result_paths.values())
  missing = [str(path) for path in required_paths if not path.is_file()]
  if missing:
    raise SystemExit("missing ABBA-precheck inputs: " + ", ".join(missing))

  git = PRODUCT.BOOT.git_state(out)
  correctness_gate = PRODUCT.load_json(CORRECTNESS_GATE)
  reference = PRODUCT.load_json(REFERENCE_RESULT)
  plugin_sha = sha256(PLUGIN)
  reference_config_hashes = {
      label: sha256(path) for label, path in config_paths.items()
  }
  reference_runs = {
      label: PRODUCT.load_json(path) for label, path in result_paths.items()
  }
  old_jitter = {
      label: jitter(reference_runs[label])
      for label in ("candidate-b1", "candidate-b2")
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
      resume=args.resume,
      timeout_s=args.timeout_s,
      worker_transient_scope=True,
  )
  runs: dict[str, dict[str, Any]] = {}
  for label in SCHEDULE:
    config = PRODUCT.load_json(config_paths[label])
    config["case_id"] = "prefill_shape_002k_parallel_block_topk"
    config["candidate_gpu_plugin"] = (
        str(PLUGIN) if label.startswith("candidate") else None)
    config["reference_result"] = str(REFERENCE_RESULT.resolve())
    config["capture_execution_census"] = False
    config["capture_logits"] = False
    runs[label] = PRODUCT.run_worker(
        worker_args, raw / "block00" / label, config)

  results = {label: run.get("result", {}) for label, run in runs.items()}
  block_inputs_complete = all(
      all(
          finite_number(results[label].get(key))
          for key in ("prefill_tokens_s", "decode_tokens_s", "total_wall_ms"))
      for label in SCHEDULE)
  block = (
      PRODUCT.block_summary(0, runs)
      if block_inputs_complete else
      {"block": 0, "phases": {}})
  new_jitter = {
      label: jitter(results[label])
      for label in ("candidate-b1", "candidate-b2")
  }
  phase_ratios = {
      phase: float(row["ratio"])
      for phase, row in block["phases"].items()
      if finite_number(row.get("ratio"))
  }
  scopes = [
      (run.get("worker_transient_scope") or {}).get("unit")
      for run in runs.values()
  ]
  candidate_counts = [
      (results[label].get("execution_census") or {}).get(
          "executed_type_counts")
      for label in ("candidate-b1", "candidate-b2")
  ]
  old_candidate_counts = [
      (reference_runs[label].get("execution_census") or {}).get(
          "executed_type_counts")
      for label in ("candidate-b1", "candidate-b2")
  ]
  monitors = [run.get("monitor") or {} for run in runs.values()]
  peak_rss = max(
      int(row.get("process_rss_peak_bytes") or 0) for row in monitors)
  peak_swap = max(
      int(row.get("process_swap_peak_bytes") or 0) for row in monitors)
  minimum_available = min(
      int(row.get("system_available_min_bytes") or 0) for row in monitors)
  stop_bytes = int(MEMORY_STOP_GIB * 1024**3)

  checks = [
      check("repository_clean_at_gate", not git["dirty"], git=git),
      check("seq2191_correctness_admits_only_one_abba_precheck",
            correctness_gate.get("required_checks_passed") is True and
            correctness_gate.get("abba_precheck_admitted") is True and
            correctness_gate.get("formal_product_promotion_admitted")
                is False and
            correctness_gate.get("plugin", {}).get("sha256") ==
                EXPECTED_PLUGIN_SHA256),
      check("accepted_seq2185_abba_shape_and_reference_are_exact",
            sha256(REFERENCE_RESULT) ==
                EXPECTED_REFERENCE_RESULT_SHA256 and
            reference_config_hashes == EXPECTED_CONFIG_SHA256 and
            reference.get("generated_token_ids_sha256") ==
                EXPECTED_TOKEN_SHA256 and
            len(reference.get("generated_token_ids", [])) == 512,
            reference_result_sha256=sha256(REFERENCE_RESULT),
            config_sha256=reference_config_hashes),
      check("strict_abba_worker_evidence_complete_serially_without_oom",
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
      check("all_four_timing_token_streams_are_exact",
            all(
                result.get("generated_token_count") == 512 and
                result.get("generated_token_ids_sha256") ==
                    EXPECTED_TOKEN_SHA256 and
                result.get("generated_token_ids") ==
                    reference.get("generated_token_ids") and
                result.get("teacher_forced_from_stock") is True
                for result in results.values()),
            expected_sha256=EXPECTED_TOKEN_SHA256,
            observed={
                label: result.get("generated_token_ids_sha256")
                for label, result in results.items()
            }),
      check("stock_denominator_remains_untouched",
            all(
                results[label].get("mode") == "stock" and
                results[label].get("candidate_path") == "stock_sdpa" and
                results[label].get("candidate_gpu_plugin_sha256") is None and
                results[label].get("lm_head_i8q1") is False
                for label in ("stock-a1", "stock-a2"))),
      check("both_candidates_select_exact_seq2189_provider",
            plugin_sha == EXPECTED_PLUGIN_SHA256 and
            all(
                results[label].get("candidate_gpu_plugin_sha256") ==
                    plugin_sha and
                results[label].get("lm_head_i8q1") is True and
                results[label].get("lm_head_i8q1_gated_exact") is True and
                results[label].get("lm_head_i8q1_greedy_local2") is False and
                results[label].get("lm_head_token_only_feedback") is False and
                provider_exact(results[label])
                for label in ("candidate-b1", "candidate-b2")),
            plugin_sha256=plugin_sha),
      check("both_candidate_jitter_rows_clear_1p25",
            all(row["pass"] for row in new_jitter.values()),
            threshold=JITTER_MAX, old=old_jitter, new=new_jitter),
      check("candidate_jitter_improves_over_seq2185",
            all(
                finite_number(new_jitter[label]["p95_over_p50"]) and
                finite_number(old_jitter[label]["p95_over_p50"]) and
                float(new_jitter[label]["p95_over_p50"]) <
                    float(old_jitter[label]["p95_over_p50"])
                for label in new_jitter),
            old=old_jitter, new=new_jitter),
      check("all_abba1_phase_points_clear_short_guard",
            set(phase_ratios) == {
                "prefill_tokens_s", "decode_tokens_s", "total_rate"} and
            all(
                math.isfinite(value) and value >= POINT_RATIO_MIN
                for value in phase_ratios.values()),
            threshold=POINT_RATIO_MIN, phase_ratios=phase_ratios),
      check("timing_execution_census_omission_matches_seq2185",
            candidate_counts == old_candidate_counts == [None, None],
            executed_type_counts=candidate_counts),
      check("memory_guard_holds_for_every_serial_worker",
            all(
                (run.get("memory_guard") or {}).get("tripped") is False
                for run in runs.values()) and
            minimum_available >= stop_bytes,
            peak_rss_bytes=peak_rss, peak_swap_bytes=peak_swap,
            minimum_available_bytes=minimum_available,
            stop_bytes=stop_bytes),
  ]
  required = all(row["pass"] for row in checks)
  verdict = (
      "admit_parallel_block_topk_for_formal_2k_abba8"
      if required else
      "repair_or_reject_parallel_block_topk_product_timing")
  payload = {
      "schema": SCHEMA,
      "workstream": WS,
      "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
      "git": git,
      "verdict": verdict,
      "required_checks_passed": required,
      "formal_2k_abba8_admitted": required,
      "formal_product_promotion_admitted": False,
      "performance_claim_admitted": False,
      "schedule": list(SCHEDULE),
      "workers_concurrent": False,
      "checks": checks,
      "plugin": {"path": str(PLUGIN), "sha256": plugin_sha},
      "block": block,
      "old_jitter": old_jitter,
      "new_jitter": new_jitter,
      "runs": runs,
      "next_action": {
          "route": "parallel_block_topk_formal_2k_abba8",
          "requirements": [
              "run eight interleaved 2k prefill-shape ABBA blocks",
              "require paired one-sided 95 percent LCB at least 0.98",
              "require all sixteen candidate jitter rows at or below 1.25",
              "retain seq2191 correctness and exact timing tokens",
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
              PRODUCT_TOOL, CORRECTNESS_GATE, REFERENCE_RESULT, PLUGIN,
              *config_paths.values(), *result_paths.values())
      },
      "plugin": payload["plugin"],
      "schedule": list(SCHEDULE),
      "gpu_workers": len(SCHEDULE),
      "stock_workers": 2,
      "candidate_workers": 2,
      "workers_concurrent": False,
  })
  report = f"""# Parallel exact block-top8 2k ABBA precheck

Verdict: **{verdict}**. Required checks: `{str(required).lower()}`.

The single ABBA block preserves all four exact 512-token streams. Candidate
jitter moves from
`{ratio_text(old_jitter['candidate-b1']['p95_over_p50'])}/`
`{ratio_text(old_jitter['candidate-b2']['p95_over_p50'])}` to
`{ratio_text(new_jitter['candidate-b1']['p95_over_p50'])}/`
`{ratio_text(new_jitter['candidate-b2']['p95_over_p50'])}`.
Prefill/decode/total point ratios are
`{ratio_text(phase_ratios.get('prefill_tokens_s'))}/`
`{ratio_text(phase_ratios.get('decode_tokens_s'))}/`
`{ratio_text(phase_ratios.get('total_rate'))}x`.

All four workers are isolated and serial. Peak RSS/swap telemetry is
`{peak_rss}/{peak_swap} B`, minimum available memory is
`{minimum_available} B`, and no OOM or guard event occurs. ABBA1 is a bounded
precheck, not a formal speed claim.
"""
  (out / "report.md").write_text(report, encoding="utf-8")
  print(json.dumps({
      "artifact": PRODUCT.relative(out),
      "verdict": verdict,
      "required_checks_passed": required,
      "candidate_jitter": {
          label: row["p95_over_p50"] for label, row in new_jitter.items()
      },
      "phase_ratios": phase_ratios,
      "peak_rss_bytes": peak_rss,
      "peak_swap_bytes": peak_swap,
  }, separators=(",", ":")), flush=True)
  return 0 if required else 2


if __name__ == "__main__":
  raise SystemExit(main())
