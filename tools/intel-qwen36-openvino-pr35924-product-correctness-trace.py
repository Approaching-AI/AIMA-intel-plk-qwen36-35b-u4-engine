#!/usr/bin/env python3
"""Run one traced 2k/output130 correctness inference for PR35924.

The sole candidate worker is teacher-forced from the accepted seq2183 stock
tokens.  The OpenCL interposer records only grouped micro-GEMM and prefill
SwiGLU launches, proving that the dynamic grouped provider executes while the
removed standalone SwiGLU kernel does not.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-openvino-pr35924-product-correctness-trace-v0"
PRODUCT_TOOL = ROOT / "tools/intel-qwen36-openvino-hot-cold-product-gate.py"
COMPILE_GATE = ROOT / (
    "output/openvino-pr35924-product-compile-"
    "20260731Tseq2234a-clean/result.json")
REFERENCE_ROOT = ROOT / (
    "output/openvino-2k-gated-exact-timing-abba1-"
    "20260731Tseq2183-clean/raw/sentinel_002k/correctness")
REFERENCE_CONFIG = REFERENCE_ROOT / "candidate/worker-config.json"
REFERENCE_CANDIDATE = REFERENCE_ROOT / "candidate/worker-result.json"
REFERENCE_STOCK = REFERENCE_ROOT / "stock/worker-result.json"
TRACE_SOURCE = ROOT / "engine/tools/opencl_dispatch_trace.cpp"
TRACE_LIBRARY = ROOT / "build/engine/iq36-opencl-dispatch-trace.so"
PLUGIN = Path(
    "/home/intel/intel-qwen36-r0/output/"
    "openvino-90214e-l0-gpu-seq2233/bin/intel64/Release/"
    "libopenvino_intel_gpu_plugin.so")
EXPECTED_PRODUCT_TOOL_SHA256 = (
    "d1b97f110ce79cd244f6bb3f1734a4aca5723c333e9126f827de38098e3e8759")
EXPECTED_COMPILE_GATE_SHA256 = (
    "fc878e0728e708f7648f30bbd8a1f422f206a77407dd5a62ea73f222b1f92135")
EXPECTED_REFERENCE_CONFIG_SHA256 = (
    "21bd692f93ba8bba40badf29d01a214ddcd55e4276de42971db71af8354cfced")
EXPECTED_REFERENCE_CANDIDATE_SHA256 = (
    "fa6a4aacdd45251c6818b467477794688754ffc7c5fa744ad9fb22e4961523b3")
EXPECTED_REFERENCE_STOCK_SHA256 = (
    "c327d633b0a6c75320d577bbe555e992303f85da3de800be7b8d70536f7d5215")
EXPECTED_TRACE_SOURCE_SHA256 = (
    "a3a823f17bc25d69e5ae0f8bb28b5092381d252c202eb7cd83735ea5435ec48d")
EXPECTED_TRACE_LIBRARY_SHA256 = (
    "713e007476acfca4d47036144ce68377189bfeb0c03b8b24be8727958f13439c")
EXPECTED_PLUGIN_SHA256 = (
    "c66c9be61ee31110a55c8a064ed1390bd3d21a3f1766a03fdea84a078a519849")
EXPECTED_TOKEN_SHA256 = (
    "7cb86794ff37361ce5008a88a3b54eebbf9548256947825438e85b48d0a76d41")
EXPECTED_MOE_PRIMITIVE = "ocl::moe::moe_3gemm_swiglu_opt___f16"
EXPECTED_LM_HEAD_PROVIDER = "+".join((
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
BUILD_AUDIT_COMMIT = "473d84c591c209d376e1f522e0f8125f355187d7"


def load_module(name: str, path: Path) -> Any:
  spec = importlib.util.spec_from_file_location(name, path)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import {path}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


PRODUCT = load_module("iq36_pr35924_correctness_product", PRODUCT_TOOL)


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


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
  return subprocess.run(
      command, cwd=ROOT, text=True, capture_output=True, check=False)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
  rows = []
  if not path.is_file():
    return rows
  for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
    if not line.strip():
      continue
    try:
      value = json.loads(line)
    except json.JSONDecodeError:
      continue
    if isinstance(value, dict):
      rows.append(value)
  return rows


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
      PRODUCT_TOOL, COMPILE_GATE, REFERENCE_CONFIG, REFERENCE_CANDIDATE,
      REFERENCE_STOCK, TRACE_SOURCE, TRACE_LIBRARY, PLUGIN)
  missing = [str(path) for path in required_paths if not path.is_file()]
  if missing:
    raise SystemExit(
        "missing PR35924 correctness inputs: " + ", ".join(missing))

  git = PRODUCT.BOOT.git_state(out)
  head = run(["git", "rev-parse", "HEAD"]).stdout.strip()
  origin_main = run(["git", "rev-parse", "origin/main"]).stdout.strip()
  ancestor = run([
      "git", "merge-base", "--is-ancestor", BUILD_AUDIT_COMMIT, head])
  compile_gate = PRODUCT.load_json(COMPILE_GATE)
  base_config = PRODUCT.load_json(REFERENCE_CONFIG)
  old_candidate = PRODUCT.load_json(REFERENCE_CANDIDATE)
  stock = PRODUCT.load_json(REFERENCE_STOCK)
  plugin_sha = sha256(PLUGIN)
  expected_tokens = [
      int(value) for value in stock["generated_token_ids"][:OUTPUT_TOKENS]]
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
      "case_id": "sentinel_002k_pr35924_grouped_postops_output130",
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
  trace_path = raw / "dispatch-trace.jsonl"
  marker_path = raw / "trace-active"
  marker_path.write_text(
      "sentinel_002k_pr35924_output130\n", encoding="utf-8")
  trace_environment = {
      "LD_PRELOAD": str(TRACE_LIBRARY.resolve()),
      "IQ36_OPENCL_TRACE_MARKER": str(marker_path.resolve()),
      "IQ36_OPENCL_TRACE_PATH": str(trace_path.resolve()),
      "IQ36_OPENCL_TRACE_FILTER": "grouped_micro_gemm,prefill_swiglu",
      "IQ36_OPENCL_TRACE_TIMING": "0",
  }
  previous_environment = {
      key: os.environ.get(key) for key in trace_environment}
  try:
    os.environ.update(trace_environment)
    worker = PRODUCT.run_worker(worker_args, raw / "candidate", config)
  finally:
    for key, previous in previous_environment.items():
      if previous is None:
        os.environ.pop(key, None)
      else:
        os.environ[key] = previous

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
    if (new_sha != old_sha or new_row.get("shape") != old_row.get("shape") or
        new_row.get("byte_count") != old_row.get("byte_count")):
      bitwise_mismatches.append(step)

  trace = result.get("lm_head_i8q1_trace") or {}
  selections = trace.get("selection_rows") or []
  prepack = trace.get("weight_prepack_rows") or []
  provider_exact = (
      len(selections) == 2 and
      all(
          row.get("provider") == EXPECTED_LM_HEAD_PROVIDER and
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

  trace_rows = load_jsonl(trace_path)
  dispatches = [
      row for row in trace_rows if row.get("event") == "ndrange"]
  grouped_dispatches = [
      row for row in dispatches
      if "grouped_micro_gemm" in str(row.get("kernel", ""))]
  old_swiglu_dispatches = [
      row for row in dispatches
      if "prefill_swiglu" in str(row.get("kernel", ""))]
  grouped_names = sorted({
      str(row.get("kernel")) for row in grouped_dispatches})
  old_swiglu_names = sorted({
      str(row.get("kernel")) for row in old_swiglu_dispatches})

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
      check("repository_is_clean_and_pushed_at_correctness_gate",
            not git["dirty"] and head == origin_main and
            ancestor.returncode == 0,
            git=git, head=head, origin_main=origin_main,
            build_audit_commit_is_ancestor=ancestor.returncode == 0),
      check("seq2234a_compile_gate_admits_one_traced_inference",
            sha256(COMPILE_GATE) == EXPECTED_COMPILE_GATE_SHA256 and
            compile_gate.get("required_checks_passed") is True and
            compile_gate.get(
                "candidate_output130_dispatch_trace_worker_admitted")
                is True and
            compile_gate.get("plugin", {}).get("sha256") ==
                EXPECTED_PLUGIN_SHA256,
            compile_gate_sha256=sha256(COMPILE_GATE),
            compile_gate_verdict=compile_gate.get("verdict")),
      check("runtime_references_trace_binary_and_plugin_are_exact",
            sha256(PRODUCT_TOOL) == EXPECTED_PRODUCT_TOOL_SHA256 and
            sha256(REFERENCE_CONFIG) ==
                EXPECTED_REFERENCE_CONFIG_SHA256 and
            sha256(REFERENCE_CANDIDATE) ==
                EXPECTED_REFERENCE_CANDIDATE_SHA256 and
            sha256(REFERENCE_STOCK) == EXPECTED_REFERENCE_STOCK_SHA256 and
            sha256(TRACE_SOURCE) == EXPECTED_TRACE_SOURCE_SHA256 and
            sha256(TRACE_LIBRARY) == EXPECTED_TRACE_LIBRARY_SHA256 and
            plugin_sha == EXPECTED_PLUGIN_SHA256 and
            expected_tokens == candidate_reference_tokens,
            product_tool_sha256=sha256(PRODUCT_TOOL),
            trace_source_sha256=sha256(TRACE_SOURCE),
            trace_library_sha256=sha256(TRACE_LIBRARY),
            plugin_sha256=plugin_sha),
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
            result.get("candidate_gpu_plugin_sha256") == plugin_sha and
            result.get("candidate_path") == "hot_cold_custom" and
            result.get("lm_head_i8q1") is True and
            result.get("lm_head_i8q1_gated_exact") is True and
            result.get("lm_head_i8q1_gated_q4") is False and
            result.get("lm_head_i8q1_greedy_local2") is False and
            result.get("lm_head_token_only_feedback") is False),
      check("real_lm_head_provider_remains_exact",
            provider_exact and prepack_exact,
            selection_count=len(selections),
            prepack_count=len(prepack),
            providers=sorted({
                str(row.get("provider")) for row in selections})),
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
            result.get("teacher_forced_from_stock") is True,
            expected_token_sha256=EXPECTED_TOKEN_SHA256,
            observed_token_sha256=result.get(
                "generated_token_ids_sha256")),
      check("checkpoint_files_are_complete_and_self_consistent",
            len(new_checkpoints) == OUTPUT_TOKENS and
            not invalid_checkpoint_hashes,
            checkpoint_count=len(new_checkpoints),
            invalid_hash_steps=invalid_checkpoint_hashes,
            bitwise_mismatch_count=len(bitwise_mismatches),
            note=(
                "bitwise equality is diagnostic because fused post-ops "
                "legitimately move the F16 materialization boundary")),
      check("source_state_and_outer_execution_census_are_preserved",
            result.get("source_summary") ==
                old_candidate.get("source_summary") and
            result.get("state_schema_after") ==
                old_candidate.get("state_schema_after") and
            execution_counts == reference_execution_counts and
            execution_counts.get("MOE3GemmFusedCompressed") == 40 and
            runtime_census.get(
                "moe_3gemm_fused_compressed_count") == 40 and
            len(moe_rows) == 40 and
            all(row.get("primitive_type") == EXPECTED_MOE_PRIMITIVE
                for row in moe_rows),
            executed_type_counts=execution_counts),
      check("trace_executes_grouped_microgemm_without_old_swiglu",
            trace_path.is_file() and
            len(grouped_dispatches) >= 120 and
            len(grouped_dispatches) % 40 == 0 and
            not old_swiglu_dispatches and
            all(row.get("status") == 0 for row in grouped_dispatches),
            dispatch_count=len(dispatches),
            grouped_dispatch_count=len(grouped_dispatches),
            grouped_kernel_names=grouped_names,
            old_swiglu_dispatch_count=len(old_swiglu_dispatches),
            old_swiglu_kernel_names=old_swiglu_names,
            trace_sha256=sha256(trace_path) if trace_path.is_file() else None),
      check("memory_guard_never_trips",
            guard.get("tripped") is False and
            minimum_available >= stop_bytes and
            int(monitor.get("process_rss_peak_bytes", -1)) >= 0 and
            int(monitor.get("process_swap_peak_bytes", -1)) >= 0,
            stop_bytes=stop_bytes, monitor=monitor),
  ]
  required = all(row["pass"] for row in checks)
  verdict = (
      "admit_pr35924_for_one_2k_control_candidate_point_block"
      if required else
      "reject_or_repair_pr35924_runtime_correctness_or_provider")
  payload = {
      "schema": SCHEMA,
      "workstream": WS,
      "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
      "git": git,
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
      "plugin": {"path": str(PLUGIN), "sha256": plugin_sha},
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
      "dispatch_trace": {
          "environment": trace_environment,
          "path": PRODUCT.relative(trace_path),
          "sha256": sha256(trace_path) if trace_path.is_file() else None,
          "row_count": len(trace_rows),
          "dispatch_count": len(dispatches),
          "grouped_dispatch_count": len(grouped_dispatches),
          "grouped_kernel_names": grouped_names,
          "old_swiglu_dispatch_count": len(old_swiglu_dispatches),
          "old_swiglu_kernel_names": old_swiglu_names,
      },
      "worker": worker,
      "next_action": {
          "route": "pr35924_2k_control_candidate_point_block",
          "requirements": [
              "derive the paired point cap from the registered 1.005x floor",
              "run one fresh-cache control then candidate timing worker",
              "require exact timing tokens and a negative prefill delta",
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
          PRODUCT.relative(path): sha256(path)
          for path in (
              PRODUCT_TOOL, COMPILE_GATE, REFERENCE_CONFIG,
              REFERENCE_CANDIDATE, REFERENCE_STOCK, TRACE_SOURCE,
              TRACE_LIBRARY, PLUGIN)
      },
      "plugin": payload["plugin"],
      "gpu_workers": 1,
      "stock_workers": 0,
      "candidate_workers": 1,
      "workers_concurrent": False,
  })
  report = f"""# PR35924 grouped-postops 2k correctness/dispatch gate

Verdict: **{verdict}**. Required checks: `{str(required).lower()}`.

One isolated candidate worker emits `{OUTPUT_TOKENS}` teacher-forced logits.
Stock-relative max KLD is `{max(klds) if klds else None}`, top-1 rate is
`{top1_rate}`, and exact token SHA is
`{result.get('generated_token_ids_sha256')}`. Bitwise differences from the
old materialization boundary are `{len(bitwise_mismatches)}` rows.

The OpenCL trace records `{len(grouped_dispatches)}` grouped micro-GEMM
launches and `{len(old_swiglu_dispatches)}` standalone prefill-SwiGLU
launches. Peak worker RSS/swap is
`{int(monitor.get('process_rss_peak_bytes', 0))}/`
`{int(monitor.get('process_swap_peak_bytes', 0))} B`; minimum available memory
is `{minimum_available} B`. This gate makes no speed claim.
"""
  (out / "report.md").write_text(report, encoding="utf-8")
  print(json.dumps({
      "artifact": PRODUCT.relative(out),
      "verdict": verdict,
      "required_checks_passed": required,
      "grouped_dispatch_count": len(grouped_dispatches),
      "old_swiglu_dispatch_count": len(old_swiglu_dispatches),
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
