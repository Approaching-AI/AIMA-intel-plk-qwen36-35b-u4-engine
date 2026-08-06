#!/usr/bin/env python3
"""Run one 2k/output130 correctness gate for a PR35924 candidate plugin."""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import math
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-openvino-pr35924-swish-parity-correctness-v0"
PREVIOUS_TOOL = ROOT / (
    "tools/intel-qwen36-openvino-pr35924-product-correctness-trace.py")
BUILD_AUDIT = ROOT / (
    "output/openvino-pr35924-swish-parity-build-"
    "20260731Tseq2238a-clean/metrics.json")
PLUGIN = Path(
    "/home/intel/intel-qwen36-r0/output/"
    "openvino-90214e-l0-gpu-seq2238/bin/intel64/Release/"
    "libopenvino_intel_gpu_plugin.so")
EXPECTED_PREVIOUS_TOOL_SHA256 = (
    "e9fb88b90b77767b85b2303814808dae7f382720f044db6bb8c1f1fe460d1c05")
EXPECTED_BUILD_AUDIT_SHA256 = (
    "944e7c849de11843316739616bee6734f6c650214595a51fd3edda9d8e159562")
EXPECTED_PLUGIN_SHA256 = (
    "bbaaa6880695eab4381d2aa6bf32162ea318565d4c66b99f19dcef31689fbbd7")
BUILD_COMMIT = "5ad62c6a23df9bcf369c36baf49a4fbcf8936d2f"
MATERIALIZED_F16_BUILD_AUDIT = ROOT / (
    "output/openvino-pr35924-materialized-f16-build-"
    "20260801Tseq2252-clean/metrics.json")
MATERIALIZED_F16_PLUGIN = Path(
    "/home/intel/intel-qwen36-r0/output/"
    "openvino-90214e-l0-gpu-seq2252/bin/intel64/Release/"
    "libopenvino_intel_gpu_plugin.so")
EXPECTED_MATERIALIZED_F16_BUILD_AUDIT_SHA256 = (
    "4966c5b2f51938d3878bf5f97cfc892de7f92be8c0b4795ab352c32a8efffabe")
EXPECTED_MATERIALIZED_F16_PLUGIN_SHA256 = (
    "c04fc5c43f90b84bb606dfd5d251f9623d118b5f31a6d356713ba0cd74fb12ec")
MATERIALIZED_F16_BUILD_COMMIT = (
    "c494273abc5026baec41ccbb92b6b1f5238bf4c2")
MATERIALIZED_F16_MIDPOINT_BUILD_AUDIT = ROOT / (
    "output/openvino-pr35924-materialized-f16-midpoint-build-"
    "20260801Tseq2263-clean/metrics.json")
MATERIALIZED_F16_MIDPOINT_PLUGIN = Path(
    "/home/intel/intel-qwen36-r0/output/"
    "openvino-90214e-l0-gpu-seq2263/bin/intel64/Release/"
    "libopenvino_intel_gpu_plugin.so")
EXPECTED_MATERIALIZED_F16_MIDPOINT_BUILD_AUDIT_SHA256 = (
    "6972028f9dbb91734dfda62aba4ff9571baf29227754192f6cafb6d4077943a9")
EXPECTED_MATERIALIZED_F16_MIDPOINT_PLUGIN_SHA256 = (
    "f827e3441ba910bd865bfae0375852fe89c52b14694b4ca2109f98dfb150725c")
MATERIALIZED_F16_MIDPOINT_BUILD_COMMIT = (
    "228d2b22fdf911d31e2dc7ad7f69930ba4e5801c")
STOCK_DIVISION_BUILD_AUDIT = ROOT / (
    "output/openvino-pr35924-stock-division-exact-build-"
    "20260801Tseq2275-clean/metrics.json")
STOCK_DIVISION_PLUGIN = Path(
    "/home/intel/intel-qwen36-r0/output/"
    "openvino-90214e-l0-gpu-seq2275/bin/intel64/Release/"
    "libopenvino_intel_gpu_plugin.so")
EXPECTED_STOCK_DIVISION_BUILD_AUDIT_SHA256 = (
    "757ca9d761409d7be7a3e8359e731f07325121a92b35ca58e6bf68fc4500ae8e")
EXPECTED_STOCK_DIVISION_PLUGIN_SHA256 = (
    "b808e9b1dffe71439b8db94647566ffc88d928fab65d1abcd1be07848f6542ef")
STOCK_DIVISION_BUILD_COMMIT = (
    "5836f5f0bab966018a85ab7132e5fbd84698a235")
OUTPUT_TOKENS = 130
PREFLIGHT_GIB = 8.0
MEMORY_STOP_GIB = 4.0


def load_module(name: str, path: Path) -> Any:
  spec = importlib.util.spec_from_file_location(name, path)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import {path}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


PREVIOUS = load_module("iq36_pr35924_previous_correctness", PREVIOUS_TOOL)
PRODUCT = PREVIOUS.PRODUCT


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out-dir", type=Path, required=True)
  parser.add_argument("--timeout-s", type=int, default=1800)
  parser.add_argument(
      "--materialized-f16", action="store_true",
      help="run the component-admitted seq2252 candidate")
  parser.add_argument(
      "--materialized-f16-midpoint", action="store_true",
      help="run the exhaustive-F16-census-exact seq2263 candidate")
  parser.add_argument(
      "--stock-division-exact", action="store_true",
      help="run the component-exact seq2275 candidate")
  args = parser.parse_args()
  if args.timeout_s <= 0:
    parser.error("timeout must be positive")
  if sum((
      bool(args.materialized_f16),
      bool(args.materialized_f16_midpoint),
      bool(args.stock_division_exact),
  )) > 1:
    parser.error(
        "Swish parity variants are mutually exclusive")
  return args


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
  return subprocess.run(
      command, cwd=ROOT, text=True, capture_output=True, check=False)


def main() -> int:
  args = parse_args()
  out = args.out_dir.resolve()
  if out.exists():
    raise SystemExit(f"output already exists: {out}")
  raw = out / "raw"
  raw.mkdir(parents=True, exist_ok=False)
  stock_division_exact = bool(args.stock_division_exact)
  materialized_f16_midpoint = bool(args.materialized_f16_midpoint)
  materialized_f16 = bool(
      args.materialized_f16 or materialized_f16_midpoint
      or stock_division_exact)
  build_audit_path = (
      STOCK_DIVISION_BUILD_AUDIT
      if stock_division_exact
      else MATERIALIZED_F16_MIDPOINT_BUILD_AUDIT
      if materialized_f16_midpoint
      else MATERIALIZED_F16_BUILD_AUDIT
      if materialized_f16
      else BUILD_AUDIT)
  plugin = (
      STOCK_DIVISION_PLUGIN
      if stock_division_exact
      else MATERIALIZED_F16_MIDPOINT_PLUGIN
      if materialized_f16_midpoint
      else MATERIALIZED_F16_PLUGIN
      if materialized_f16
      else PLUGIN)
  expected_build_audit_sha256 = (
      EXPECTED_STOCK_DIVISION_BUILD_AUDIT_SHA256
      if stock_division_exact
      else EXPECTED_MATERIALIZED_F16_MIDPOINT_BUILD_AUDIT_SHA256
      if materialized_f16_midpoint
      else
      EXPECTED_MATERIALIZED_F16_BUILD_AUDIT_SHA256
      if materialized_f16 else EXPECTED_BUILD_AUDIT_SHA256)
  expected_plugin_sha256 = (
      EXPECTED_STOCK_DIVISION_PLUGIN_SHA256
      if stock_division_exact
      else EXPECTED_MATERIALIZED_F16_MIDPOINT_PLUGIN_SHA256
      if materialized_f16_midpoint
      else
      EXPECTED_MATERIALIZED_F16_PLUGIN_SHA256
      if materialized_f16 else EXPECTED_PLUGIN_SHA256)
  build_commit = (
      STOCK_DIVISION_BUILD_COMMIT
      if stock_division_exact
      else MATERIALIZED_F16_MIDPOINT_BUILD_COMMIT
      if materialized_f16_midpoint
      else
      MATERIALIZED_F16_BUILD_COMMIT if materialized_f16 else BUILD_COMMIT)
  variant = (
      "materialized_f16_native_exp_stock_division_exact"
      if stock_division_exact
      else "materialized_f16_native_exp_midpoint_exact"
      if materialized_f16_midpoint
      else "materialized_f16_native_exp"
      if materialized_f16
      else "openvino_swish_parity")
  required_paths = (
      PREVIOUS_TOOL, build_audit_path, PREVIOUS.PRODUCT_TOOL,
      PREVIOUS.REFERENCE_CONFIG, PREVIOUS.REFERENCE_CANDIDATE,
      PREVIOUS.REFERENCE_STOCK, plugin)
  missing = [str(path) for path in required_paths if not path.is_file()]
  if missing:
    raise SystemExit(
        "missing candidate-correctness inputs: " + ", ".join(missing))

  git = PRODUCT.BOOT.git_state(out)
  head = run(["git", "rev-parse", "HEAD"]).stdout.strip()
  origin_main = run(["git", "rev-parse", "origin/main"]).stdout.strip()
  ancestor = run([
      "git", "merge-base", "--is-ancestor", build_commit, head])
  build_audit = PRODUCT.load_json(build_audit_path)
  base_config = PRODUCT.load_json(PREVIOUS.REFERENCE_CONFIG)
  old_candidate = PRODUCT.load_json(PREVIOUS.REFERENCE_CANDIDATE)
  stock = PRODUCT.load_json(PREVIOUS.REFERENCE_STOCK)
  plugin_sha = PREVIOUS.sha256(plugin)
  expected_tokens = [
      int(value) for value in stock["generated_token_ids"][:OUTPUT_TOKENS]]
  candidate_reference_tokens = [
      int(value)
      for value in old_candidate["generated_token_ids"][:OUTPUT_TOKENS]]
  reference_path = out / "reference-output130.json"
  PRODUCT.write_json(reference_path, {
      "generated_token_ids": expected_tokens,
      "source": PRODUCT.relative(PREVIOUS.REFERENCE_STOCK),
  })

  config = dict(base_config)
  config.update({
      "candidate_gpu_plugin": str(plugin),
      "case_id": (
          "sentinel_002k_pr35924_stock_division_exact_output130"
          if stock_division_exact
          else "sentinel_002k_pr35924_materialized_f16_midpoint_output130"
          if materialized_f16_midpoint
          else "sentinel_002k_pr35924_materialized_f16_output130"
          if materialized_f16
          else "sentinel_002k_pr35924_candidate_output130"),
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

  new_checkpoints = PREVIOUS.checkpoint_map(result)
  old_checkpoints = PREVIOUS.checkpoint_map(old_candidate)
  bitwise_mismatches = []
  invalid_checkpoint_hashes = []
  for step in range(OUTPUT_TOKENS):
    new_row = new_checkpoints.get(step)
    old_row = old_checkpoints.get(step)
    if new_row is None or old_row is None:
      bitwise_mismatches.append(step)
      continue
    new_path = PREVIOUS.checkpoint_path(new_row)
    old_path = PREVIOUS.checkpoint_path(old_row)
    if not new_path.is_file() or not old_path.is_file():
      bitwise_mismatches.append(step)
      continue
    new_sha = PREVIOUS.sha256(new_path)
    old_sha = PREVIOUS.sha256(old_path)
    if (new_row.get("sha256") != new_sha or
        old_row.get("sha256") != old_sha):
      invalid_checkpoint_hashes.append(step)
    if (new_sha != old_sha or new_row.get("shape") != old_row.get("shape") or
        new_row.get("byte_count") != old_row.get("byte_count")):
      bitwise_mismatches.append(step)

  lm_trace = result.get("lm_head_i8q1_trace") or {}
  selections = lm_trace.get("selection_rows") or []
  prepack = lm_trace.get("weight_prepack_rows") or []
  provider_exact = (
      len(selections) == 2 and
      all(
          row.get("provider") == PREVIOUS.EXPECTED_LM_HEAD_PROVIDER and
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
  execution_counts = (
      result.get("execution_census", {}).get("executed_type_counts") or {})
  reference_execution_counts = (
      old_candidate.get("execution_census", {}).get(
          "executed_type_counts") or {})
  runtime_census = result.get("runtime_census") or {}
  moe_rows = runtime_census.get(
      "moe_3gemm_fused_compressed_rows") or []
  monitor = worker.get("monitor") or {}
  guard = worker.get("memory_guard") or {}
  minimum_available = int(
      monitor.get("system_available_min_bytes") or 0)
  stop_bytes = int(MEMORY_STOP_GIB * 1024**3)

  checks = [
      PREVIOUS.check(
          "repository_is_clean_and_pushed_at_correctness_gate",
          not git["dirty"] and head == origin_main and
          ancestor.returncode == 0,
          git=git, head=head, origin_main=origin_main,
          build_commit_is_ancestor=ancestor.returncode == 0),
      PREVIOUS.check(
          "candidate_build_admits_one_correctness_worker",
          PREVIOUS.sha256(build_audit_path) ==
              expected_build_audit_sha256 and
          build_audit.get("required_checks_passed") is True and
          build_audit.get(
              "candidate_output130_correctness_worker_admitted") is True and
          build_audit.get("variant") == variant and
          build_audit.get("candidate_plugin", {}).get("sha256") ==
              expected_plugin_sha256,
          variant=variant,
          build_audit_sha256=PREVIOUS.sha256(build_audit_path),
          build_audit_verdict=build_audit.get("verdict")),
      PREVIOUS.check(
          "tool_references_and_plugin_are_exact",
          PREVIOUS.sha256(PREVIOUS_TOOL) ==
              EXPECTED_PREVIOUS_TOOL_SHA256 and
          PREVIOUS.sha256(PREVIOUS.PRODUCT_TOOL) ==
              PREVIOUS.EXPECTED_PRODUCT_TOOL_SHA256 and
          PREVIOUS.sha256(PREVIOUS.REFERENCE_CONFIG) ==
              PREVIOUS.EXPECTED_REFERENCE_CONFIG_SHA256 and
          PREVIOUS.sha256(PREVIOUS.REFERENCE_CANDIDATE) ==
              PREVIOUS.EXPECTED_REFERENCE_CANDIDATE_SHA256 and
          PREVIOUS.sha256(PREVIOUS.REFERENCE_STOCK) ==
              PREVIOUS.EXPECTED_REFERENCE_STOCK_SHA256 and
          plugin_sha == expected_plugin_sha256 and
          expected_tokens == candidate_reference_tokens,
          previous_tool_sha256=PREVIOUS.sha256(PREVIOUS_TOOL),
          plugin_sha256=plugin_sha),
      PREVIOUS.check(
          "single_serial_candidate_worker_completes_without_oom",
          worker.get("returncode") == 0 and
          worker.get("timed_out") is False and
          worker.get("oom_observed") is False and
          worker.get("reused") is not True and
          (worker.get("worker_transient_scope") or {}).get("enabled") is True),
      PREVIOUS.check(
          "isolated_plugin_and_count25_product_flags_are_exact",
          result.get("candidate_gpu_plugin_sha256") == plugin_sha and
          result.get("candidate_path") == "hot_cold_custom" and
          result.get("lm_head_i8q1") is True and
          result.get("lm_head_i8q1_gated_exact") is True and
          result.get("lm_head_i8q1_gated_q4") is False and
          result.get("lm_head_i8q1_greedy_local2") is False and
          result.get("lm_head_token_only_feedback") is False),
      PREVIOUS.check(
          "real_lm_head_provider_remains_exact",
          provider_exact and prepack_exact,
          selection_count=len(selections), prepack_count=len(prepack)),
      PREVIOUS.check(
          "all_130_stock_relative_distributions_pass",
          len(distributions) == OUTPUT_TOKENS and finite_rows and
          len(klds) == OUTPUT_TOKENS and max(klds) <= PREVIOUS.KLD_MAX and
          top1_rate >= PREVIOUS.TOP1_MIN,
          row_count=len(distributions),
          max_kld=max(klds) if klds else None,
          kld_threshold=PREVIOUS.KLD_MAX,
          top1_rate=top1_rate, top1_threshold=PREVIOUS.TOP1_MIN),
      PREVIOUS.check(
          "exact_output130_tokens_are_preserved",
          result.get("generated_token_count") == OUTPUT_TOKENS and
          result.get("generated_token_ids") == expected_tokens and
          result.get("generated_token_ids_sha256") ==
              PREVIOUS.EXPECTED_TOKEN_SHA256 and
          result.get("teacher_forced_from_stock") is True,
          expected_token_sha256=PREVIOUS.EXPECTED_TOKEN_SHA256,
          observed_token_sha256=result.get("generated_token_ids_sha256")),
      PREVIOUS.check(
          "checkpoint_files_are_complete_and_self_consistent",
          len(new_checkpoints) == OUTPUT_TOKENS and
          not invalid_checkpoint_hashes,
          checkpoint_count=len(new_checkpoints),
          invalid_hash_steps=invalid_checkpoint_hashes,
          bitwise_mismatch_count=len(bitwise_mismatches)),
      PREVIOUS.check(
          "source_state_and_execution_census_are_preserved",
          result.get("source_summary") ==
              old_candidate.get("source_summary") and
          result.get("state_schema_after") ==
              old_candidate.get("state_schema_after") and
          execution_counts == reference_execution_counts and
          execution_counts.get("MOE3GemmFusedCompressed") == 40 and
          runtime_census.get(
              "moe_3gemm_fused_compressed_count") == 40 and
          len(moe_rows) == 40 and
          all(row.get("primitive_type") ==
              PREVIOUS.EXPECTED_MOE_PRIMITIVE for row in moe_rows),
          executed_type_counts=execution_counts),
      PREVIOUS.check(
          "memory_guard_never_trips",
          guard.get("tripped") is False and
          minimum_available >= stop_bytes and
          int(monitor.get("process_rss_peak_bytes", -1)) >= 0 and
          int(monitor.get("process_swap_peak_bytes", -1)) >= 0,
          stop_bytes=stop_bytes, monitor=monitor),
  ]
  required = all(row["pass"] for row in checks)
  verdict = (
      "admit_pr35924_candidate_for_one_2k_control_candidate_point_block"
      if required else
      "reject_pr35924_candidate_on_product_correctness")
  payload = {
      "schema": SCHEMA,
      "workstream": WS,
      "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
      "git": git,
      "variant": variant,
      "verdict": verdict,
      "required_checks_passed": required,
      "point_performance_block_admitted": required,
      "formal_product_promotion_admitted": False,
      "performance_claim_admitted": False,
      "gpu_workers_launched": 1,
      "stock_workers_launched": 0,
      "candidate_workers_launched": 1,
      "workers_concurrent": False,
      "checks": checks,
      "plugin": {"path": str(plugin), "sha256": plugin_sha},
      "correctness": {
          "checkpoint_count": len(new_checkpoints),
          "bitwise_mismatch_count": len(bitwise_mismatches),
          "bitwise_mismatch_steps": bitwise_mismatches,
          "distribution_row_count": len(distributions),
          "max_kld": max(klds) if klds else None,
          "top1_rate": top1_rate,
          "generated_token_ids_sha256": result.get(
              "generated_token_ids_sha256"),
      },
      "worker": worker,
      "next_action": {
          "route": "pr35924_candidate_2k_control_candidate_point_block",
          "requirements": [
              "run one fresh-cache control then candidate timing worker",
              "require exact timing tokens and negative prefill delta",
              "only a positive point block may fund formal paired inference",
          ],
      },
  }
  PRODUCT.write_json(out / "result.json", payload)
  PRODUCT.write_json(out / "manifest.json", {
      "schema": SCHEMA,
      "tool": PRODUCT.relative(Path(__file__)),
      "git": git,
      "inputs": {
          PRODUCT.relative(path): PREVIOUS.sha256(path)
          for path in required_paths
      },
      "plugin": payload["plugin"],
      "gpu_workers": 1,
      "stock_workers": 0,
      "candidate_workers": 1,
      "workers_concurrent": False,
  })
  report = f"""# PR35924 candidate 2k correctness gate

Verdict: **{verdict}**. Required checks: `{str(required).lower()}`.

One isolated candidate worker emits `{OUTPUT_TOKENS}` teacher-forced logits.
Stock-relative max KLD is `{max(klds) if klds else None}`, top-1 rate is
`{top1_rate}`, exact token SHA is
`{result.get('generated_token_ids_sha256')}`, and bitwise differences from the
old materialized boundary are `{len(bitwise_mismatches)}` rows.

Peak worker RSS/swap is
`{int(monitor.get('process_rss_peak_bytes', 0))}/`
`{int(monitor.get('process_swap_peak_bytes', 0))} B`; minimum available memory
is `{minimum_available} B`. This gate makes no speed claim.
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
