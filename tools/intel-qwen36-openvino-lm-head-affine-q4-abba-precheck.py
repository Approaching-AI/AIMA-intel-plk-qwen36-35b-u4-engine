#!/usr/bin/env python3
"""Run one output512 ABBA block for affine-Q4 versus accepted seq2189.

All four workers use the same accepted 2k timing shape and stock token
reference.  The A workers bind the immutable seq2189 gated-exact carrier; the
B workers bind seq2291 and enable only its affine-Q4 count25 fallback.  This
incremental precheck requires exact tokens, provider isolation, stable jitter,
and a decode-wall point saving above the registered product kill-number.  One
block funds a formal stock-denominator ABBA8 but cannot make a speed claim.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import math
import statistics
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-openvino-lm-head-affine-q4-abba-precheck-v0"
PRODUCT_TOOL = ROOT / "tools/intel-qwen36-openvino-hot-cold-product-gate.py"
CORRECTNESS_GATE = ROOT / (
    "output/openvino-lm-head-affine-q4-output130-correctness-"
    "20260801Tseq2292-clean/result.json")
CORRECTNESS_MANIFEST = ROOT / (
    "output/openvino-lm-head-affine-q4-output130-correctness-"
    "20260801Tseq2292-clean/manifest.json")
REFERENCE_ROOT = ROOT / (
    "output/openvino-2k-gated-exact-timing-abba1-"
    "20260731Tseq2183-clean/raw/sentinel_002k")
REFERENCE_RESULT = REFERENCE_ROOT / "correctness/stock/worker-result.json"
REFERENCE_BLOCK = REFERENCE_ROOT / "block00"
CONTROL_PLUGIN = Path(
    "/home/intel/intel-qwen36-r0/output/"
    "openvino-90214e-l0-gpu-seq2189/bin/intel64/Release/"
    "libopenvino_intel_gpu_plugin.so")
CANDIDATE_PLUGIN = Path(
    "/home/intel/intel-qwen36-r0/output/"
    "openvino-90214e-l0-gpu-seq2291/bin/intel64/Release/"
    "libopenvino_intel_gpu_plugin.so")

EXPECTED_CORRECTNESS_RESULT_SHA256 = (
    "9799a2b755ac4eed47aff1e046d39775d1867f134593ca80b3fb861017219523")
EXPECTED_CORRECTNESS_MANIFEST_SHA256 = (
    "dc7b4bf6c7504f17bb0998d312f71479c01c6fd4e09415576c66e1334bb7ebfc")
EXPECTED_CONTROL_SHA256 = (
    "9832adffa8bf3fd7b013c47d9d2abefa66987c5fbde350c9669fce58c697b985")
EXPECTED_CANDIDATE_SHA256 = (
    "b63eede5177f4f9e05d02e97d9f24f52b4289504c2a7c7b4e06c580d1d880e12")
EXPECTED_REFERENCE_RESULT_SHA256 = (
    "c327d633b0a6c75320d577bbe555e992303f85da3de800be7b8d70536f7d5215")
EXPECTED_CONFIG_SHA256 = {
    "control-a1":
        "031093b19157578485819b48adfabe953c8c7781a3b691f160d5ad40f34c3559",
    "candidate-b1":
        "41ffd7bdb13871c46da9292516884c1f6cb50a0e1dea303794ea794b6cd5ebfb",
    "candidate-b2":
        "ae3922bf10e46d9bc23923176d801bbd7ac22f325be5c36bd7cf0358d0a9b441",
    "control-a2":
        "78be261089570039cf470e0c96fca9f3cb412b66e4ddb57a0c23c36119da4f5d",
}
EXPECTED_TOKEN_SHA256 = (
    "fb7820272ad3bdac1acac0506c2abb594795eab9cea34dbdcbc2326970319db7")
BASE_PROVIDER = "+".join((
    "iq36_lm_head_q8_group256_f16_sums",
    "iq36_lm_head_i8q1_rowstripe8_matvec_local_top12_f16",
    "iq36_lm_head_i8_exact_local_top12_correction_f16",
    "iq36_lm_head_output_topk8_f16",
    "iq36_lm_head_topk8_merge_f32",
    "iq36_lm_head_i8_direct_topk8_correction_f16",
))
CONTROL_PROVIDER = BASE_PROVIDER + "+" + "+".join((
    "iq36_lm_head_i8q1_gated_exact_reset_f16",
    "iq36_lm_head_i8q1_gated_exact_collect_f16",
    "iq36_lm_head_i8_gated_exact_matvec_f16",
    "iq36_lm_head_i8q1_gated_exact_output_topk8_f16",
    "iq36_lm_head_i8q1_gated_exact_topk8_merge_f32",
    "iq36_lm_head_i8_gated_exact_topk8_correction_f16",
))
CANDIDATE_PROVIDER = BASE_PROVIDER + "+" + "+".join((
    "iq36_lm_head_i8q1_gated_exact_reset_f16",
    "iq36_lm_head_i8q1_gated_exact_collect_f16",
    "iq36_lm_head_i8q1_affine_q4_hidden_group_norms_f16",
    "iq36_lm_head_i8q1_affine_q4_bound_select_f16",
    "iq36_lm_head_i8_affine_q4_exact_candidates_f16",
    "iq36_lm_head_i8_gated_exact_matvec_f16",
    "iq36_lm_head_i8q1_gated_exact_output_topk8_f16",
    "iq36_lm_head_i8q1_gated_exact_topk8_merge_f32",
    "iq36_lm_head_i8_gated_exact_topk8_correction_f16",
))
SCHEDULE = ("control-a1", "candidate-b1", "candidate-b2", "control-a2")
SOURCE_LABEL = {
    "control-a1": "stock-a1",
    "candidate-b1": "candidate-b1",
    "candidate-b2": "candidate-b2",
    "control-a2": "stock-a2",
}
JITTER_SKIP = 16
JITTER_MAX = 1.25
NONINFERIORITY_MIN = 0.98
DECODE_SAVING_MIN_MS = 0.011203750
PREFLIGHT_GIB = 8.0
MEMORY_STOP_GIB = 4.0


def load_module(name: str, path: Path) -> Any:
  spec = importlib.util.spec_from_file_location(name, path)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import {path}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


PRODUCT = load_module("iq36_affine_q4_abba_product", PRODUCT_TOOL)


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


def git_text(*args: str) -> str:
  return subprocess.check_output(
      ["git", *args], cwd=ROOT, text=True).strip()


def finite_number(value: Any) -> bool:
  return (
      isinstance(value, (int, float)) and
      not isinstance(value, bool) and
      math.isfinite(float(value)))


def jitter(result: dict[str, Any]) -> dict[str, Any]:
  samples = [
      float(value)
      for value in result.get("decode_wall_ms", [])[JITTER_SKIP:]
      if finite_number(value)
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
          finite_number(ratio) and float(ratio) <= JITTER_MAX),
  }


def provider_exact(
    result: dict[str, Any], affine_q4: bool,
) -> bool:
  trace = result.get("lm_head_i8q1_trace") or {}
  selections = trace.get("selection_rows") or []
  prepacks = trace.get("weight_prepack_rows") or []
  expected_provider = CANDIDATE_PROVIDER if affine_q4 else CONTROL_PROVIDER
  expected_packed_bytes = 336_225_280 if affine_q4 else 66_053_120
  expected_capacity = 16_812 if affine_q4 else 0
  expected_passes = 3 if affine_q4 else 2
  expected_codec = (
      "binary_two_centroid_lloyd5+gated_affine_q4_group128"
      if affine_q4 else "binary_two_centroid_lloyd5")
  return (
      len(selections) == 2 and len(prepacks) == 2 and
      prepacks[0].get("process_cache_hit") is False and
      prepacks[1].get("process_cache_hit") is True and
      all(
          row.get("provider") == expected_provider and
          row.get("packed_bytes") == expected_packed_bytes and
          row.get("adaptive_correction_capacity") == expected_capacity and
          row.get("correction_passes") == expected_passes and
          row.get("global") == [248320, 1, 1] and
          row.get("local") == [256, 1, 1]
          for row in selections) and
      all(
          row.get("codec") == expected_codec and
          row.get("packed_bytes") == expected_packed_bytes and
          row.get("adaptive_correction_capacity") == expected_capacity and
          row.get("exact_correction_passes") == expected_passes
          for row in prepacks))


def median_metric(
    results: dict[str, dict[str, Any]], labels: tuple[str, str], key: str,
) -> float | None:
  values = [results[label].get(key) for label in labels]
  if not all(finite_number(value) for value in values):
    return None
  return float(statistics.median(float(value) for value in values))


def main() -> int:
  args = parse_args()
  out = args.out_dir.resolve()
  if out.exists() and not args.resume:
    raise SystemExit(f"output already exists: {out}")
  raw = out / "raw"
  raw.mkdir(parents=True, exist_ok=args.resume)
  config_paths = {
      label: REFERENCE_BLOCK / SOURCE_LABEL[label] / "worker-config.json"
      for label in SCHEDULE
  }
  required_paths = (
      PRODUCT_TOOL, CORRECTNESS_GATE, CORRECTNESS_MANIFEST,
      REFERENCE_RESULT, CONTROL_PLUGIN, CANDIDATE_PLUGIN,
      *config_paths.values())
  missing = [str(path) for path in required_paths if not path.is_file()]
  if missing:
    raise SystemExit(
        "missing affine-Q4 ABBA-precheck inputs: " + ", ".join(missing))

  git = PRODUCT.BOOT.git_state(out)
  branch = git_text("branch", "--show-current")
  commit = git_text("rev-parse", "HEAD")
  upstream = git_text("rev-parse", "@{upstream}")
  correctness = PRODUCT.load_json(CORRECTNESS_GATE)
  reference = PRODUCT.load_json(REFERENCE_RESULT)
  control_sha = sha256(CONTROL_PLUGIN)
  candidate_sha = sha256(CANDIDATE_PLUGIN)
  reference_config_hashes = {
      label: sha256(path) for label, path in config_paths.items()
  }

  runs: dict[str, dict[str, Any]] = {}
  for label in SCHEDULE:
    affine_q4 = label.startswith("candidate")
    plugin = CANDIDATE_PLUGIN if affine_q4 else CONTROL_PLUGIN
    config = PRODUCT.load_json(config_paths[label])
    config.update({
        "candidate_gpu_plugin": str(plugin),
        "candidate_path": "hot_cold_custom",
        "capture_execution_census": False,
        "capture_logits": False,
        "case_id": "sentinel_002k_affine_q4_incremental",
        "lm_head_i8q1": True,
        "lm_head_i8q1_gated_exact": True,
        "lm_head_i8q1_gated_exact_affine_q4": affine_q4,
        "lm_head_i8q1_gated_q4": False,
        "lm_head_i8q1_greedy_local2": False,
        "lm_head_token_only_feedback": False,
        "mode": "candidate",
        "purpose": "paired_product_timing",
        "reference_result": str(REFERENCE_RESULT.resolve()),
    })
    worker_args = SimpleNamespace(
        abort_below_available_gib=MEMORY_STOP_GIB,
        candidate_gpu_plugin=plugin,
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
    runs[label] = PRODUCT.run_worker(
        worker_args, raw / "block00" / label, config)

  results = {label: run.get("result", {}) for label, run in runs.items()}
  control_labels = ("control-a1", "control-a2")
  candidate_labels = ("candidate-b1", "candidate-b2")
  control_prefill = median_metric(
      results, control_labels, "prefill_tokens_s")
  candidate_prefill = median_metric(
      results, candidate_labels, "prefill_tokens_s")
  control_decode = median_metric(
      results, control_labels, "decode_tokens_s")
  candidate_decode = median_metric(
      results, candidate_labels, "decode_tokens_s")
  control_total = median_metric(results, control_labels, "total_wall_ms")
  candidate_total = median_metric(results, candidate_labels, "total_wall_ms")
  phase_ratios = {
      "prefill_tokens_s": (
          candidate_prefill / control_prefill
          if candidate_prefill and control_prefill else None),
      "decode_tokens_s": (
          candidate_decode / control_decode
          if candidate_decode and control_decode else None),
      "total_rate": (
          control_total / candidate_total
          if control_total and candidate_total else None),
  }
  control_decode_samples = [
      float(value)
      for label in control_labels
      for value in results[label].get("decode_wall_ms", [])[JITTER_SKIP:]
      if finite_number(value)
  ]
  candidate_decode_samples = [
      float(value)
      for label in candidate_labels
      for value in results[label].get("decode_wall_ms", [])[JITTER_SKIP:]
      if finite_number(value)
  ]
  control_decode_mean_ms = (
      float(statistics.fmean(control_decode_samples))
      if control_decode_samples else None)
  candidate_decode_mean_ms = (
      float(statistics.fmean(candidate_decode_samples))
      if candidate_decode_samples else None)
  decode_saving_ms = (
      control_decode_mean_ms - candidate_decode_mean_ms
      if control_decode_mean_ms is not None and
      candidate_decode_mean_ms is not None else None)
  jitters = {label: jitter(results[label]) for label in SCHEDULE}

  scopes = [
      (run.get("worker_transient_scope") or {}).get("unit")
      for run in runs.values()
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
      check("repository_clean_pushed_main_at_gate",
            not git["dirty"] and branch == "main" and commit == upstream,
            git=git, branch=branch, commit=commit, upstream=upstream),
      check("seq2292_correctness_admits_only_one_abba_precheck",
            sha256(CORRECTNESS_GATE) ==
                EXPECTED_CORRECTNESS_RESULT_SHA256 and
            sha256(CORRECTNESS_MANIFEST) ==
                EXPECTED_CORRECTNESS_MANIFEST_SHA256 and
            correctness.get("required_checks_passed") is True and
            correctness.get("abba_precheck_admitted") is True and
            correctness.get("performance_claim_admitted") is False and
            correctness.get("plugin", {}).get("sha256") ==
                EXPECTED_CANDIDATE_SHA256,
            correctness_result_sha256=sha256(CORRECTNESS_GATE),
            correctness_manifest_sha256=sha256(CORRECTNESS_MANIFEST)),
      check("accepted_output512_shape_and_reference_are_exact",
            sha256(REFERENCE_RESULT) ==
                EXPECTED_REFERENCE_RESULT_SHA256 and
            reference_config_hashes == EXPECTED_CONFIG_SHA256 and
            reference.get("generated_token_ids_sha256") ==
                EXPECTED_TOKEN_SHA256 and
            len(reference.get("generated_token_ids", [])) == 512,
            reference_result_sha256=sha256(REFERENCE_RESULT),
            config_sha256=reference_config_hashes),
      check("accepted_control_and_candidate_plugins_are_exact",
            control_sha == EXPECTED_CONTROL_SHA256 and
            candidate_sha == EXPECTED_CANDIDATE_SHA256 and
            control_sha != candidate_sha,
            control_sha256=control_sha, candidate_sha256=candidate_sha),
      check("strict_abba_workers_complete_serially_without_oom",
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
            all(
                result.get("generated_token_count") == 512 and
                result.get("generated_token_ids_sha256") ==
                    EXPECTED_TOKEN_SHA256 and
                result.get("generated_token_ids") ==
                    reference.get("generated_token_ids") and
                result.get("teacher_forced_from_stock") is True and
                result.get("sentinel_pass") is True
                for result in results.values()),
            expected_sha256=EXPECTED_TOKEN_SHA256,
            observed={
                label: result.get("generated_token_ids_sha256")
                for label, result in results.items()
            }),
      check("both_control_workers_select_exact_seq2189_provider",
            all(
                results[label].get("candidate_gpu_plugin_sha256") ==
                    control_sha and
                results[label].get("lm_head_i8q1") is True and
                results[label].get("lm_head_i8q1_gated_exact") is True and
                results[label].get(
                    "lm_head_i8q1_gated_exact_affine_q4") is False and
                results[label].get("lm_head_i8q1_greedy_local2") is False and
                provider_exact(results[label], False)
                for label in control_labels)),
      check("both_candidate_workers_select_exact_affine_q4_provider",
            all(
                results[label].get("candidate_gpu_plugin_sha256") ==
                    candidate_sha and
                results[label].get("lm_head_i8q1") is True and
                results[label].get("lm_head_i8q1_gated_exact") is True and
                results[label].get(
                    "lm_head_i8q1_gated_exact_affine_q4") is True and
                results[label].get("lm_head_i8q1_greedy_local2") is False and
                provider_exact(results[label], True)
                for label in candidate_labels)),
      check("all_candidate_jitter_rows_clear_1p25",
            all(jitters[label]["pass"] for label in candidate_labels),
            threshold=JITTER_MAX, jitter=jitters),
      check("incremental_decode_saving_clears_kill_number",
            finite_number(decode_saving_ms) and
            float(decode_saving_ms) >= DECODE_SAVING_MIN_MS and
            finite_number(phase_ratios["decode_tokens_s"]) and
            float(phase_ratios["decode_tokens_s"]) > 1.0,
            control_decode_mean_ms=control_decode_mean_ms,
            candidate_decode_mean_ms=candidate_decode_mean_ms,
            decode_saving_ms=decode_saving_ms,
            minimum_saving_ms=DECODE_SAVING_MIN_MS,
            decode_ratio=phase_ratios["decode_tokens_s"]),
      check("prefill_and_total_points_remain_noninferior",
            all(
                finite_number(phase_ratios[key]) and
                float(phase_ratios[key]) >= NONINFERIORITY_MIN
                for key in ("prefill_tokens_s", "total_rate")),
            threshold=NONINFERIORITY_MIN, phase_ratios=phase_ratios),
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
      "admit_affine_q4_for_formal_2k_stock_abba8"
      if required else
      "reject_or_repair_affine_q4_product_timing")
  payload = {
      "schema": SCHEMA,
      "workstream": WS,
      "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
      "git": {
          **git, "branch": branch, "commit": commit, "upstream": upstream},
      "verdict": verdict,
      "required_checks_passed": required,
      "formal_2k_stock_abba8_admitted": required,
      "formal_product_promotion_admitted": False,
      "performance_claim_admitted": False,
      "schedule": list(SCHEDULE),
      "workers_concurrent": False,
      "checks": checks,
      "plugins": {
          "control": {"path": str(CONTROL_PLUGIN), "sha256": control_sha},
          "candidate": {
              "path": str(CANDIDATE_PLUGIN), "sha256": candidate_sha},
      },
      "phase_ratios": phase_ratios,
      "incremental_decode": {
          "control_mean_ms": control_decode_mean_ms,
          "candidate_mean_ms": candidate_decode_mean_ms,
          "saving_ms": decode_saving_ms,
          "minimum_saving_ms": DECODE_SAVING_MIN_MS,
      },
      "jitter": jitters,
      "runs": runs,
      "next_action": {
          "route": "affine_q4_formal_2k_stock_abba8",
          "requirements": [
              "run eight stock/candidate interleaved ABBA blocks",
              "require paired one-sided 95 percent LCB at least 0.98",
              "retain output512 sentinel and exact timing tokens",
              "record incremental result versus accepted seq2189 separately",
          ],
      },
  }
  PRODUCT.write_json(out / "result.json", payload)
  PRODUCT.write_json(out / "manifest.json", {
      "schema": SCHEMA,
      "tool": PRODUCT.relative(Path(__file__)),
      "git": payload["git"],
      "inputs": {
          PRODUCT.relative(path): sha256(path)
          for path in required_paths
      },
      "plugins": payload["plugins"],
      "schedule": list(SCHEDULE),
      "gpu_workers": len(SCHEDULE),
      "control_workers": 2,
      "candidate_workers": 2,
      "workers_concurrent": False,
  })
  report = f"""# Affine-Q4 LM-head incremental ABBA precheck

Verdict: **{verdict}**. Required checks: `{str(required).lower()}`.

One strict output512 ABBA block compares accepted seq2189 directly with
seq2291 affine-Q4. All four streams preserve exact token SHA
`{EXPECTED_TOKEN_SHA256}` and the sentinel. Pooled post-skip decode means
are `{control_decode_mean_ms}` and `{candidate_decode_mean_ms} ms/token`,
for a `{decode_saving_ms}-ms` candidate saving versus the registered
`{DECODE_SAVING_MIN_MS}-ms` kill-number. Prefill/decode/total point ratios are
`{phase_ratios['prefill_tokens_s']}/`
`{phase_ratios['decode_tokens_s']}/`
`{phase_ratios['total_rate']}x`.

Peak RSS/swap telemetry is `{peak_rss}/{peak_swap} B`, minimum available
memory is `{minimum_available} B`, and no OOM or guard event occurs. This
single incremental block only decides whether formal stock ABBA8 is funded.
"""
  (out / "report.md").write_text(report, encoding="utf-8")
  print(json.dumps({
      "artifact": PRODUCT.relative(out),
      "verdict": verdict,
      "required_checks_passed": required,
      "phase_ratios": phase_ratios,
      "decode_saving_ms": decode_saving_ms,
      "minimum_saving_ms": DECODE_SAVING_MIN_MS,
      "candidate_jitter": {
          label: jitters[label]["p95_over_p50"]
          for label in candidate_labels
      },
      "peak_rss_bytes": peak_rss,
      "peak_swap_bytes": peak_swap,
  }, separators=(",", ":")), flush=True)
  return 0 if required else 2


if __name__ == "__main__":
  raise SystemExit(main())
