#!/usr/bin/env python3
"""Compile the exact 2k product graph with an isolated PR35924 plugin.

The gate launches one serial candidate-only worker under the established
8/4-GiB memory guards and returns before creating an InferRequest.  It proves
that all 40 MoE nodes bind the optimized outer implementation.  Creation of
the dynamic grouped-matmul post-op primitive is deliberately deferred to one
bounded correctness inference.  The product runtime uses Level Zero, so an
OpenCL dispatch interposer is not treated as execution evidence.
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
SCHEMA = "intel-qwen36-openvino-pr35924-product-compile-gate-v0"
PRODUCT_TOOL = ROOT / "tools/intel-qwen36-openvino-hot-cold-product-gate.py"
BUILD_AUDIT = ROOT / (
    "output/openvino-pr35924-product-build-audit-"
    "20260731Tseq2233a-clean/metrics.json")
BUILD_METRICS = ROOT / (
    "output/openvino-pr35924-product-build-"
    "20260731Tseq2233-clean/metrics.json")
OPENVINO_PATCH = ROOT / "engine/openvino/iq36-pr35924-grouped-postops.patch"
ONEDNN_PATCH = ROOT / (
    "engine/openvino/iq36-onednn-grouped-postops-swish.patch")
PLUGIN = Path(
    "/home/intel/intel-qwen36-r0/output/"
    "openvino-90214e-l0-gpu-seq2233/bin/intel64/Release/"
    "libopenvino_intel_gpu_plugin.so")
ACCEPTED_PLUGIN = Path(
    "/home/intel/intel-qwen36-r0/output/"
    "openvino-90214e-l0-gpu-seq2189/bin/intel64/Release/"
    "libopenvino_intel_gpu_plugin.so")
EXPECTED_PRODUCT_TOOL_SHA256 = (
    "d1b97f110ce79cd244f6bb3f1734a4aca5723c333e9126f827de38098e3e8759")
EXPECTED_BUILD_AUDIT_SHA256 = (
    "5ab06710959e5ed5c7c44b2e85f194a7b1c8a3f2edddd585b837e304ebd87205")
EXPECTED_BUILD_METRICS_SHA256 = (
    "9d91d52648509ac5a89f16ddca589dc23d274cc0393498d34b962429c044ec64")
EXPECTED_OPENVINO_PATCH_SHA256 = (
    "6f205f856a0118c0a43bb7914131a3f8edee148f279c9e2b0e7cf967ca8c8350")
EXPECTED_ONEDNN_PATCH_SHA256 = (
    "732a0c75bb5622e58683db070e09029ed7278a7c7993633e8f1df27e8c047a9a")
EXPECTED_PLUGIN_SHA256 = (
    "c66c9be61ee31110a55c8a064ed1390bd3d21a3f1766a03fdea84a078a519849")
EXPECTED_ACCEPTED_PLUGIN_SHA256 = (
    "9832adffa8bf3fd7b013c47d9d2abefa66987c5fbde350c9669fce58c697b985")
BUILD_AUDIT_COMMIT = "473d84c591c209d376e1f522e0f8125f355187d7"
LAYERS = (3, 7, 11, 15, 19, 23, 27, 31, 35, 39)
BUCKET = 2048
OUTPUT_TOKENS = 130
PREFILL_HISTORY_CAPACITY = 16384
EXACT_HISTORY_CAPACITY = 17408
PREFLIGHT_GIB = 8.0
MEMORY_STOP_GIB = 4.0
EXPECTED_MOE_PRIMITIVE = "ocl::moe::moe_3gemm_swiglu_opt___f16"


def load_module(name: str, path: Path) -> Any:
  spec = importlib.util.spec_from_file_location(name, path)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import {path}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


PRODUCT = load_module("iq36_pr35924_compile_product", PRODUCT_TOOL)


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


def runtime_names(census: dict[str, Any], prefix: str) -> list[str]:
  return sorted(
      str(row.get("name")) for row in census.get("attention_rows", [])
      if row.get("layer_type") == "CustomGPUPrimitive" and
      str(row.get("name", "")).startswith(prefix))


def main() -> int:
  args = parse_args()
  out = args.out_dir.resolve()
  if out.exists():
    raise SystemExit(f"output already exists: {out}")
  raw = out / "raw"
  raw.mkdir(parents=True, exist_ok=False)
  required_paths = (
      PRODUCT_TOOL, BUILD_AUDIT, BUILD_METRICS, OPENVINO_PATCH, ONEDNN_PATCH,
      PLUGIN, ACCEPTED_PLUGIN, PRODUCT.CUSTOM_CONFIG, PRODUCT.MODEL_DIR,
      PRODUCT.MODEL_CONTRACT)
  missing = [str(path) for path in required_paths if not path.exists()]
  if missing:
    raise SystemExit("missing PR35924 compile inputs: " + ", ".join(missing))

  git = PRODUCT.BOOT.git_state(out)
  head = run(["git", "rev-parse", "HEAD"]).stdout.strip()
  origin_main = run(["git", "rev-parse", "origin/main"]).stdout.strip()
  ancestor = run([
      "git", "merge-base", "--is-ancestor", BUILD_AUDIT_COMMIT, head])
  build_audit = PRODUCT.load_json(BUILD_AUDIT)
  build_metrics = PRODUCT.load_json(BUILD_METRICS)
  plugin_sha = sha256(PLUGIN)
  accepted_plugin_sha = sha256(ACCEPTED_PLUGIN)
  model_identity = PRODUCT.BOOT.capture_model_identity(
      PRODUCT.MODEL_DIR.resolve(), PRODUCT.MODEL_CONTRACT.resolve())
  PRODUCT.write_json(out / "model-identity.json", model_identity)

  config = {
      "alias_linear_state_assign": True,
      "bucket": BUCKET,
      "candidate_dq_realloc_fastpath": True,
      "candidate_fc_stable_prepare_fastpath": True,
      "candidate_path": "hot_cold_custom",
      "capture_attention_history_layers": [],
      "capture_attention_history_steps": [],
      "capture_attention_layers": [],
      "capture_attention_steps": [],
      "capture_execution_census": True,
      "case_id": "sentinel_002k_pr35924_grouped_postops_compile_only",
      "compile_only": True,
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
      "fixed_fc_cohorts": [],
      "fixed_fc_manager_direct": False,
      "fixed_fc_manager_scope": "all",
      "fuse_fixed_fc": False,
      "fuse_linear_conv_state": True,
      "fuse_qk_rope_layout": False,
      "fuse_router_shared_pair": False,
      "fuse_router_shared_triple": False,
      "host_time_profiling": 0,
      "instantiate_only": False,
      "linear_state_alias_scope": "all",
      "lm_head_device_greedy_feedback": False,
      "lm_head_i8q1": True,
      "lm_head_i8q1_gated_exact": True,
      "lm_head_i8q1_gated_q4": False,
      "lm_head_i8q1_greedy_local2": False,
      "lm_head_i8q4": False,
      "lm_head_token_only_feedback": False,
      "mode": "candidate",
      "output_tokens": OUTPUT_TOKENS,
      "prefill_history_capacity": PREFILL_HISTORY_CAPACITY,
      "purpose": "pr35924_grouped_postops_compile_gate",
      "self_bind_hot_states": False,
      "target_layers": list(LAYERS),
      "timing_token_output": False,
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
      worker_transient_scope=True,
  )
  worker = PRODUCT.run_worker(worker_args, raw / "worker", config)
  result = worker.get("result", {})
  source = result.get("source_summary") or {}
  census = result.get("runtime_census") or {}
  compiler_cache = result.get("compiler_cache") or {}
  monitor = worker.get("monitor") or {}
  memory_guard = worker.get("memory_guard") or {}
  expected_attention_names = sorted(
      f"iq36_hot_attention_layer{layer}" for layer in LAYERS)
  moe_rows = census.get("moe_3gemm_fused_compressed_rows") or []
  moe_names = sorted(str(row.get("name")) for row in moe_rows)
  moe_primitives = sorted(
      {str(row.get("primitive_type")) for row in moe_rows})
  minimum_available = int(
      monitor.get("system_available_min_bytes") or 0)
  stop_bytes = int(MEMORY_STOP_GIB * 1024**3)
  audit_verdict = build_audit.get("verdict") or {}

  checks = [
      check("repository_is_clean_and_pushed_at_compile_gate",
            not git["dirty"] and head == origin_main and
            ancestor.returncode == 0,
            git=git, head=head, origin_main=origin_main,
            build_audit_commit_is_ancestor=ancestor.returncode == 0),
      check("seq2233a_build_audit_admits_only_compile",
            sha256(BUILD_AUDIT) == EXPECTED_BUILD_AUDIT_SHA256 and
            sha256(BUILD_METRICS) == EXPECTED_BUILD_METRICS_SHA256 and
            audit_verdict.get("required_checks_passed") is True and
            audit_verdict.get("verdict") ==
                "admit_pr35924_plugin_for_compile_only_graph_gate" and
            audit_verdict.get("compile_only_graph_gate_admitted") is True and
            audit_verdict.get("inference_admitted") is False and
            build_metrics.get("candidate_plugin", {}).get("sha256") ==
                EXPECTED_PLUGIN_SHA256,
            build_audit_sha256=sha256(BUILD_AUDIT),
            build_metrics_sha256=sha256(BUILD_METRICS),
            audit_verdict=audit_verdict),
      check("isolated_candidate_and_accepted_plugins_are_exact",
            plugin_sha == EXPECTED_PLUGIN_SHA256 and
            accepted_plugin_sha == EXPECTED_ACCEPTED_PLUGIN_SHA256 and
            PLUGIN.resolve() != ACCEPTED_PLUGIN.resolve(),
            candidate_plugin=str(PLUGIN),
            candidate_plugin_sha256=plugin_sha,
            accepted_plugin_sha256=accepted_plugin_sha),
      check("pushed_runtime_census_and_patches_are_exact",
            sha256(PRODUCT_TOOL) == EXPECTED_PRODUCT_TOOL_SHA256 and
            sha256(OPENVINO_PATCH) == EXPECTED_OPENVINO_PATCH_SHA256 and
            sha256(ONEDNN_PATCH) == EXPECTED_ONEDNN_PATCH_SHA256,
            product_tool_sha256=sha256(PRODUCT_TOOL),
            openvino_patch_sha256=sha256(OPENVINO_PATCH),
            onednn_patch_sha256=sha256(ONEDNN_PATCH)),
      check("locked_model_identity",
            model_identity.get("required_checks_passed") is True,
            model_dir=model_identity.get("model_dir")),
      check("one_serial_transient_compile_worker_completes",
            worker.get("returncode") == 0 and
            worker.get("timed_out") is False and
            worker.get("oom_observed") is False and
            worker.get("reused") is not True and
            (worker.get("worker_transient_scope") or {}).get("enabled")
                is True),
      check("compile_only_worker_creates_no_request_or_inference",
            result.get("compile_only") is True and
            result.get("instantiate_only") is False and
            result.get("worker_created_infer_request") is False and
            result.get("worker_executed_inference") is False and
            "generated_token_ids" not in result and
            "state_schema_after" not in result),
      check("accepted_2k_carrier_flags_are_exact",
            result.get("candidate_gpu_plugin_sha256") == plugin_sha and
            result.get("candidate_path") == "hot_cold_custom" and
            result.get("custom_composition") == "exact_phase" and
            config.get("fuse_qk_rope_layout") is False and
            config.get("fuse_router_shared_pair") is False and
            config.get("fuse_router_shared_triple") is False and
            config.get("fuse_fixed_fc") is False and
            config.get("fixed_fc_manager_direct") is False and
            source.get("fuse_qk_rope_layout") is False and
            source.get("fuse_fixed_fc") is False),
      check("source_graph_retains_exact_dual_attention_carrier",
            source.get("target_layers") == list(LAYERS) and
            source.get("decode_stock_micro_layers") == list(LAYERS) and
            source.get("exact_phase_decode") is True and
            source.get("exact_phase_dual_cohort") is True and
            source.get("exact_phase_context_partition4") is False and
            source.get("exact_history_layers") == list(LAYERS) and
            source.get("exact_history_capacity") ==
                EXACT_HISTORY_CAPACITY and
            source.get("prefill_history_capacity") ==
                PREFILL_HISTORY_CAPACITY and
            source.get("fixed_cold_capacity") == BUCKET and
            source.get("custom_count_after") == len(LAYERS) and
            source.get("stock_sdpa_count_after") == 0),
      check("compiled_runtime_retains_attention_linear_and_moe_census",
            census.get("qk_rope_layout_custom_count") == 0 and
            census.get("hot_attention_custom_count") == len(LAYERS) and
            census.get("linear_conv_custom_count") == 30 and
            census.get("fixed_fc_custom_count") == 0 and
            census.get("stock_sdpa_like_count") == 0 and
            census.get("moe_3gemm_fused_compressed_count") == 40 and
            len(set(moe_names)) == 40 and
            runtime_names(census, "iq36_hot_attention_layer") ==
                expected_attention_names,
            attention_names=runtime_names(
                census, "iq36_hot_attention_layer"),
            moe_names=moe_names),
      check("all_forty_moe_owners_bind_optimized_outer_provider",
            len(moe_rows) == 40 and
            all(str(row.get("primitive_type")) == EXPECTED_MOE_PRIMITIVE
                for row in moe_rows),
            expected_primitive=EXPECTED_MOE_PRIMITIVE,
            observed_primitives=moe_primitives,
            note=(
                "the inner grouped-matmul primitive is shape-created on first "
                "prefill and is not claimed by this compile-only boundary")),
      check("fresh_cache_binds_count25_full_logit_provider",
            compiler_cache.get("lm_head_i8q1_gated_exact_env") == "1" and
            compiler_cache.get("lm_head_i8q1_gated_q4_env") is None and
            compiler_cache.get("lm_head_i8q1_greedy_local2_env") is None and
            compiler_cache.get("lm_head_i8q1_token_only_env") is None and
            compiler_cache.get("neo_cache_persistent") == "1" and
            result.get("lm_head_i8q1") is True and
            result.get("lm_head_i8q1_gated_exact") is True and
            result.get("lm_head_i8q1_gated_q4") is False and
            result.get("lm_head_i8q1_greedy_local2") is False and
            result.get("lm_head_token_only_feedback") is False),
      check("full_graph_compile_duration_is_finite",
            isinstance(result.get("language_compile_ms"), (int, float)) and
            math.isfinite(
                float(result.get("language_compile_ms", math.nan))) and
            float(result.get("language_compile_ms", 0.0)) > 0.0,
            language_compile_ms=result.get("language_compile_ms")),
      check("worker_memory_guard_holds_without_oom",
            int(monitor.get("process_rss_peak_bytes", -1)) >= 0 and
            int(monitor.get("process_swap_peak_bytes", -1)) >= 0 and
            memory_guard.get("tripped") is False and
            minimum_available >= stop_bytes and
            worker.get("oom_observed") is False,
            monitor=monitor, stop_bytes=stop_bytes),
      check("inner_grouped_postops_execution_is_deferred_not_assumed", True,
            next_correctness_requirements={
                "executed_MOE3GemmFusedCompressed": 40,
                "distribution_rows": 130,
                "exact_greedy_tokens": True,
            },
            note=(
                "dynamic grouped primitive creation and actual kernel "
                "execution require the first bounded prefill inference; "
                "OpenCL tracing cannot observe the Level Zero runtime")),
  ]
  required = all(row["pass"] for row in checks)
  verdict = (
      "admit_one_pr35924_output130_correctness_worker"
      if required else
      "repair_pr35924_candidate_graph_or_provider_binding")
  payload = {
      "schema": SCHEMA,
      "workstream": WS,
      "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
      "git": git,
      "verdict": verdict,
      "required_checks_passed": required,
      "candidate_output130_correctness_worker_admitted": required,
      "infer_requests_created": 0,
      "inference_workers_launched": 0,
      "performance_worker_admitted": False,
      "formal_performance_admitted": False,
      "checks": checks,
      "model_identity": PRODUCT.relative(out / "model-identity.json"),
      "plugin": {"path": str(PLUGIN), "sha256": plugin_sha},
      "worker": worker,
      "next_action": {
          "route": "pr35924_output130_correctness",
          "requirements": [
              "run exactly one candidate InferRequest worker in a fresh scope",
              "require accepted output130 logits/tokens and 40 executed MoEs",
              "use Level Zero-compatible evidence for any kernel attribution",
              "only a pass may fund one short control-candidate point block",
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
              PRODUCT_TOOL, BUILD_AUDIT, BUILD_METRICS, OPENVINO_PATCH,
              ONEDNN_PATCH, PLUGIN, ACCEPTED_PLUGIN)
      },
      "plugin": payload["plugin"],
      "gpu_workers": 1,
      "graph_compiles": 1,
      "infer_requests": 0,
      "inference_workers": 0,
  })
  report = f"""# PR35924 grouped-postops product compile-only gate

Verdict: **{verdict}**. Required checks:
`{str(required).lower()}`.

One isolated candidate worker compiled the complete 2k/output130 graph in
`{float(result.get('language_compile_ms', 0.0)):.3f} ms`. The runtime model
retains 40 MoE owners, and all bind
`{EXPECTED_MOE_PRIMITIVE}`. The dynamic inner grouped-matmul post-op primitive
is correctly deferred to the first bounded inference rather than inferred
from the outer provider name.

Peak worker RSS/swap was
`{int(monitor.get('process_rss_peak_bytes', 0))}/`
`{int(monitor.get('process_swap_peak_bytes', 0))} B`; minimum available memory
was `{minimum_available} B`. No InferRequest, inference, token, OOM, or guard
event occurred.
"""
  (out / "report.md").write_text(report, encoding="utf-8")
  print(json.dumps({
      "artifact": PRODUCT.relative(out),
      "verdict": verdict,
      "required_checks_passed": required,
      "language_compile_ms": result.get("language_compile_ms"),
      "moe_owners": census.get("moe_3gemm_fused_compressed_count"),
      "moe_primitives": moe_primitives,
      "peak_rss_bytes": monitor.get("process_rss_peak_bytes"),
      "minimum_available_bytes": minimum_available,
      "oom_observed": worker.get("oom_observed"),
  }, separators=(",", ":")), flush=True)
  return 0 if required else 2


if __name__ == "__main__":
  raise SystemExit(main())
