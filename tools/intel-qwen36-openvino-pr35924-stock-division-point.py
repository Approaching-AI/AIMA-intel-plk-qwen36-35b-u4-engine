#!/usr/bin/env python3
"""Run one fresh-cache control/candidate PR35924 prefill point.

Both workers use the same accepted exact-phase product carrier and output512
stock token reference. The first worker binds the accepted seq2189 plugin; the
second binds the component- and product-correct seq2275 plugin. This two-row
gate may fund a formal interleaved ABBA design, but it cannot make a speed
claim.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import math
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-openvino-pr35924-stock-division-point-v0"
PRODUCT_TOOL = ROOT / "tools/intel-qwen36-openvino-hot-cold-product-gate.py"
CORRECTNESS_GATE = ROOT / (
    "output/openvino-pr35924-stock-division-exact-correctness-"
    "20260801Tseq2276-clean/result.json")
BOUND_GATE = ROOT / (
    "output/openvino-pr35924-grouped-postops-bound-"
    "20260731Tseq2231-clean/metrics.json")
REFERENCE_ROOT = ROOT / (
    "output/openvino-lm-head-parallel-block-topk-2k-abba8-"
    "20260731Tseq2193-clean/raw/prefill_shape_002k")
BASE_CONFIG = REFERENCE_ROOT / "block00/candidate-b1/worker-config.json"
REFERENCE_RESULT = REFERENCE_ROOT / "correctness/stock/worker-result.json"
CONTROL_PLUGIN = Path(
    "/home/intel/intel-qwen36-r0/output/"
    "openvino-90214e-l0-gpu-seq2189/bin/intel64/Release/"
    "libopenvino_intel_gpu_plugin.so")
CANDIDATE_PLUGIN = Path(
    "/home/intel/intel-qwen36-r0/output/"
    "openvino-90214e-l0-gpu-seq2275/bin/intel64/Release/"
    "libopenvino_intel_gpu_plugin.so")
EXPECTED_SHA256 = {
    PRODUCT_TOOL: (
        "d1b97f110ce79cd244f6bb3f1734a4aca5723c333e9126f827de38098e3e8759"),
    CORRECTNESS_GATE: (
        "d1c9de147358c995b16476b10d4cbe14f4ac2a8c7a752515fc104e095aa8ea8d"),
    BOUND_GATE: (
        "b75181257a81b124e2309cbb7baebd309242fcd65a7338d4e3084aae88583258"),
    BASE_CONFIG: (
        "7cd2281731dafd08a6a58b7cf36da18c06d0e034f04dcc11bd394103661c2911"),
    REFERENCE_RESULT: (
        "5f7d0d0fbbde73e8e546a513fe294282d1f961ae72362ed2dd6900b2125d0da1"),
    CONTROL_PLUGIN: (
        "9832adffa8bf3fd7b013c47d9d2abefa66987c5fbde350c9669fce58c697b985"),
    CANDIDATE_PLUGIN: (
        "b808e9b1dffe71439b8db94647566ffc88d928fab65d1abcd1be07848f6542ef"),
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
VARIANT = "materialized_f16_native_exp_stock_division_exact"
LAYERS = tuple(range(3, 40, 4))
SCHEDULE = ("control", "candidate")
OUTPUT_TOKENS = 512
JITTER_SKIP = 16
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


PRODUCT = load_module("iq36_pr35924_stock_division_point_product", PRODUCT_TOOL)


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


def finite(value: Any) -> bool:
  return (
      isinstance(value, (int, float)) and not isinstance(value, bool) and
      math.isfinite(float(value)))


def tail_metrics(result: dict[str, Any]) -> dict[str, Any]:
  samples = [
      float(value)
      for value in result.get("decode_wall_ms", [])[JITTER_SKIP:]
      if finite(value)
  ]
  p50 = PRODUCT.percentile(samples, 0.50)
  p95 = PRODUCT.percentile(samples, 0.95)
  ratio = p95 / p50 if p50 and p95 else None
  return {
      "sample_count": len(samples),
      "p50_ms": p50,
      "p95_ms": p95,
      "p95_over_p50": ratio,
      "jitter_pass": finite(ratio) and float(ratio) <= JITTER_MAX,
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


def git_output(*args: str) -> str:
  completed = subprocess.run(
      ["git", *args], cwd=ROOT, check=False, capture_output=True, text=True)
  if completed.returncode != 0:
    raise RuntimeError(completed.stderr.strip())
  return completed.stdout.strip()


def worker_args(
    args: argparse.Namespace, plugin: Path,
) -> SimpleNamespace:
  return SimpleNamespace(
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
      resume=False,
      timeout_s=args.timeout_s,
      worker_transient_scope=True,
  )


def main() -> int:
  args = parse_args()
  out = args.out_dir.resolve()
  if out.exists():
    raise SystemExit(f"output already exists: {out}")
  raw = out / "raw"
  raw.mkdir(parents=True, exist_ok=False)
  missing = [
      str(path) for path in EXPECTED_SHA256 if not path.is_file()]
  if missing:
    raise SystemExit("missing PR35924 point inputs: " + ", ".join(missing))

  git = PRODUCT.BOOT.git_state(out)
  head = git_output("rev-parse", "HEAD")
  origin_main = git_output("rev-parse", "origin/main")
  input_hashes = {path: sha256(path) for path in EXPECTED_SHA256}
  correctness = PRODUCT.load_json(CORRECTNESS_GATE)
  bound = PRODUCT.load_json(BOUND_GATE)
  base_config = PRODUCT.load_json(BASE_CONFIG)
  reference = PRODUCT.load_json(REFERENCE_RESULT)
  expected_tokens = reference.get("generated_token_ids") or []
  registered = bound.get("registered_prefill_cut") or {}
  target_ratio = registered.get("target_ratio")
  required_saving_ms = registered.get("required_total_cut_ms")

  runs: dict[str, dict[str, Any]] = {}
  for label, plugin in (
      ("control", CONTROL_PLUGIN),
      ("candidate", CANDIDATE_PLUGIN),
  ):
    config = dict(base_config)
    config.update({
        "candidate_gpu_plugin": str(plugin),
        "capture_execution_census": False,
        "capture_logits": False,
        "case_id": "prefill_shape_002k_pr35924_stock_division_point",
        "output_tokens": OUTPUT_TOKENS,
        "purpose": "paired_product_timing",
        "reference_result": str(REFERENCE_RESULT.resolve()),
    })
    runs[label] = PRODUCT.run_worker(
        worker_args(args, plugin), raw / label, config)

  results = {
      label: runs[label].get("result") or {} for label in SCHEDULE}
  timing_complete = all(
      all(finite(results[label].get(key)) for key in (
          "prefill_wall_ms", "prefill_tokens_s", "decode_tokens_s",
          "total_wall_ms"))
      for label in SCHEDULE)
  control = results["control"]
  candidate = results["candidate"]
  point: dict[str, Any] = {}
  if timing_complete:
    control_total_rate = 1000.0 / float(control["total_wall_ms"])
    candidate_total_rate = 1000.0 / float(candidate["total_wall_ms"])
    point = {
        "control": {
            "prefill_wall_ms": float(control["prefill_wall_ms"]),
            "prefill_tokens_s": float(control["prefill_tokens_s"]),
            "decode_tokens_s": float(control["decode_tokens_s"]),
            "total_wall_ms": float(control["total_wall_ms"]),
        },
        "candidate": {
            "prefill_wall_ms": float(candidate["prefill_wall_ms"]),
            "prefill_tokens_s": float(candidate["prefill_tokens_s"]),
            "decode_tokens_s": float(candidate["decode_tokens_s"]),
            "total_wall_ms": float(candidate["total_wall_ms"]),
        },
        "prefill_saving_ms": (
            float(control["prefill_wall_ms"]) -
            float(candidate["prefill_wall_ms"])),
        "prefill_ratio": (
            float(candidate["prefill_tokens_s"]) /
            float(control["prefill_tokens_s"])),
        "decode_ratio": (
            float(candidate["decode_tokens_s"]) /
            float(control["decode_tokens_s"])),
        "total_rate_ratio": candidate_total_rate / control_total_rate,
        "target_ratio": target_ratio,
        "required_saving_ms": required_saving_ms,
        "candidate_point_cap_ms": (
            float(control["prefill_wall_ms"]) / float(target_ratio)
            if finite(target_ratio) and float(target_ratio) > 0 else None),
    }

  tails = {
      label: tail_metrics(results[label]) for label in SCHEDULE}
  scopes = [
      (runs[label].get("worker_transient_scope") or {}).get("unit")
      for label in SCHEDULE]
  monitors = [runs[label].get("monitor") or {} for label in SCHEDULE]
  peak_rss = max(
      int(row.get("process_rss_peak_bytes") or 0) for row in monitors)
  peak_swap = max(
      int(row.get("process_swap_peak_bytes") or 0) for row in monitors)
  minimum_available = min(
      int(row.get("system_available_min_bytes") or 0) for row in monitors)
  stop_bytes = int(MEMORY_STOP_GIB * 1024**3)

  checks = [
      check(
          "repository_is_clean_and_pushed_at_point_gate",
          not git["dirty"] and head == origin_main,
          git=git, head=head, origin_main=origin_main),
      check(
          "all_frozen_inputs_have_exact_hashes",
          all(
              input_hashes[path] == expected
              for path, expected in EXPECTED_SHA256.items()),
          observed={
              PRODUCT.relative(path): digest
              for path, digest in input_hashes.items()}),
      check(
          "product_correctness_and_source_bound_admit_only_this_point",
          correctness.get("required_checks_passed") is True and
          correctness.get("point_performance_block_admitted") is True and
          correctness.get("formal_product_promotion_admitted") is False and
          correctness.get("performance_claim_admitted") is False and
          correctness.get("variant") == VARIANT and
          correctness.get("plugin", {}).get("sha256") ==
              EXPECTED_SHA256[CANDIDATE_PLUGIN] and
          correctness.get("correctness", {}).get(
              "bitwise_mismatch_count") == 0 and
          correctness.get("correctness", {}).get("top1_rate") == 1.0 and
          bound.get("verdict", {}).get("required_checks_passed") is True and
          bound.get("verdict", {}).get(
              "isolated_serial_candidate_plugin_build_admitted") is True and
          registered.get("target_ratio") == 1.005 and
          registered.get("required_total_cut_ms") ==
              4.837318171429018,
          variant=correctness.get("variant"),
          registered_prefill_cut=registered),
      check(
          "strict_fresh_control_then_candidate_workers_complete_serially",
          list(runs) == list(SCHEDULE) and
          all(
              runs[label].get("returncode") == 0 and
              runs[label].get("timed_out") is False and
              runs[label].get("oom_observed") is False and
              runs[label].get("reused") is not True and
              (runs[label].get(
                  "worker_transient_scope") or {}).get("enabled") is True
              for label in SCHEDULE) and
          len(scopes) == len(set(scopes)) and
          all(scope is not None for scope in scopes),
          schedule=list(runs), scopes=scopes),
      check(
          "both_output512_token_streams_are_exact",
          reference.get("generated_token_count") == OUTPUT_TOKENS and
          reference.get("generated_token_ids_sha256") ==
              EXPECTED_TOKEN_SHA256 and
          len(expected_tokens) == OUTPUT_TOKENS and
          all(
              results[label].get("generated_token_count") == OUTPUT_TOKENS and
              results[label].get("generated_token_ids_sha256") ==
                  EXPECTED_TOKEN_SHA256 and
              results[label].get("generated_token_ids") == expected_tokens and
              results[label].get("teacher_forced_from_stock") is True
              for label in SCHEDULE),
          expected_token_sha256=EXPECTED_TOKEN_SHA256,
          observed={
              label: results[label].get("generated_token_ids_sha256")
              for label in SCHEDULE}),
      check(
          "only_the_isolated_gpu_plugin_changes",
          all(
              results[label].get("mode") == "candidate" and
              results[label].get("candidate_path") == "hot_cold_custom" and
              results[label].get("custom_composition") == "exact_phase" and
              results[label].get("exact_phase_dual_cohort") is True and
              results[label].get("target_layers") == list(LAYERS) and
              results[label].get("decode_stock_micro_layers") ==
                  list(LAYERS) and
              results[label].get("candidate_gpu_plugin_sha256") ==
                  EXPECTED_SHA256[
                      CONTROL_PLUGIN if label == "control"
                      else CANDIDATE_PLUGIN] and
              results[label].get("lm_head_i8q1") is True and
              results[label].get("lm_head_i8q1_gated_exact") is True and
              results[label].get("lm_head_i8q1_gated_q4") is False and
              results[label].get("lm_head_i8q1_greedy_local2") is False and
              results[label].get("lm_head_token_only_feedback") is False and
              provider_exact(results[label])
              for label in SCHEDULE) and
          control.get("source_summary") == candidate.get("source_summary") and
          control.get("state_schema_after") ==
              candidate.get("state_schema_after"),
          plugins={
              label: results[label].get("candidate_gpu_plugin_sha256")
              for label in SCHEDULE}),
      check(
          "both_timing_rows_are_finite_and_stable",
          timing_complete and
          all(row["jitter_pass"] for row in tails.values()),
          jitter_threshold=JITTER_MAX, tails=tails),
      check(
          "candidate_prefill_point_clears_registered_cut",
          timing_complete and
          finite(point.get("prefill_saving_ms")) and
          float(point["prefill_saving_ms"]) >=
              float(required_saving_ms) and
          finite(point.get("prefill_ratio")) and
          float(point["prefill_ratio"]) >= float(target_ratio) and
          float(candidate["prefill_wall_ms"]) <=
              float(point["candidate_point_cap_ms"]),
          point=point),
      check(
          "timing_workers_capture_no_profiles_census_or_logits",
          all(
              not results[label].get("logit_checkpoints") and
              not results[label].get("distribution_checkpoints") and
              (results[label].get("execution_census") or {}).get(
                  "executed_type_counts") is None
              for label in SCHEDULE)),
      check(
          "memory_guard_holds_for_both_serial_workers",
          all(
              (runs[label].get("memory_guard") or {}).get("tripped") is False
              for label in SCHEDULE) and
          minimum_available >= stop_bytes,
          peak_rss_bytes=peak_rss, peak_swap_bytes=peak_swap,
          minimum_available_bytes=minimum_available,
          stop_bytes=stop_bytes),
  ]
  infrastructure_passed = all(
      row["pass"]
      for row in checks
      if row["name"] != "candidate_prefill_point_clears_registered_cut")
  passed = all(row["pass"] for row in checks)
  verdict = (
      "admit_pr35924_stock_division_exact_formal_abba_design"
      if passed else
      "do_not_fund_pr35924_formal_abba_from_point"
      if infrastructure_passed else
      "repair_pr35924_control_candidate_point_evidence")
  payload = {
      "schema": SCHEMA,
      "workstream": WS,
      "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
      "git": git,
      "variant": VARIANT,
      "verdict": verdict,
      "required_checks_passed": passed,
      "formal_abba_design_admitted": passed,
      "formal_product_promotion_admitted": False,
      "performance_claim_admitted": False,
      "schedule": list(SCHEDULE),
      "workers_concurrent": False,
      "checks": checks,
      "point": point,
      "tails": tails,
      "runs": runs,
      "memory": {
          "peak_rss_bytes": peak_rss,
          "peak_swap_bytes": peak_swap,
          "minimum_available_bytes": minimum_available,
      },
      "next_action": {
          "route": (
              "pr35924_stock_division_exact_formal_abba_design"
              if passed else "pr35924_route_conclusion"),
          "requirements": (
              [
                  "design an interleaved ABBA8 control/candidate gate",
                  "bind exact output512 tokens and both isolated plugins",
                  "use the paired one-sided 95% lower confidence bound",
                  "make no speed claim until the formal inference clears",
              ]
              if passed else [
                  "record the measured point and close or park the route",
                  "return to a kernel-side opportunity with a fresh bound",
              ]),
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
      "gpu_workers": 2,
      "control_workers": 1,
      "candidate_workers": 1,
      "workers_concurrent": False,
  })
  report = f"""# PR35924 stock-division control/candidate point

Verdict: **{verdict}**. Required checks: `{str(passed).lower()}`.

Both fresh output512 workers preserve the exact stock token stream. Candidate
prefill saving/ratio is
`{point.get('prefill_saving_ms')}/{point.get('prefill_ratio')}` against the
registered `{required_saving_ms}-ms / {target_ratio}x` cut. Decode and total
rate ratios are `{point.get('decode_ratio')}/{point.get('total_rate_ratio')}`.

Peak worker RSS/swap is `{peak_rss}/{peak_swap} B`; minimum available memory
is `{minimum_available} B`. This two-row point makes no speed claim.
"""
  (out / "report.md").write_text(report, encoding="utf-8")
  print(json.dumps({
      "artifact": PRODUCT.relative(out),
      "verdict": verdict,
      "required_checks_passed": passed,
      "prefill_saving_ms": point.get("prefill_saving_ms"),
      "prefill_ratio": point.get("prefill_ratio"),
      "decode_ratio": point.get("decode_ratio"),
      "total_rate_ratio": point.get("total_rate_ratio"),
      "peak_rss_bytes": peak_rss,
      "oom_observed": any(
          runs[label].get("oom_observed") is True for label in SCHEDULE),
  }, separators=(",", ":")), flush=True)
  return 0 if passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
