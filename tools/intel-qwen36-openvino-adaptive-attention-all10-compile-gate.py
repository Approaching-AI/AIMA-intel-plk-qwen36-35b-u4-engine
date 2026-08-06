#!/usr/bin/env python3
"""Compile the full all-ten adaptive product graph without creating a request.

This is the first full-model boundary after the isolated adaptive component
gate.  One candidate worker builds the accepted product carrier with the
selected exact-correction top-k on all ten full-attention layers and compiles
it against the exact isolated plugin.  The worker exits before InferRequest
construction, state materialization, tokenization, or inference.  The existing
8-GiB preflight and 4-GiB stop remain the only memory controls.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-openvino-adaptive-attention-all10-compile-gate-v1"
ROUTE = "openvino_attention_adaptive_all10_graph_compile_boundary"
PASS_ROUTE = "openvino_attention_adaptive_all10_64k_correctness_boundary"
FAIL_ROUTE = "openvino_attention_adaptive_all10_graph_compile_repair"

PRODUCT_TOOL = ROOT / "tools/intel-qwen36-openvino-hot-cold-product-gate.py"
SOURCE_GATE = ROOT / (
    "output/openvino-adaptive-attention-source-abi-gate-"
    "20260721Tseq1738-topk1024-2048-clean/gate.json")
COMPILE_GATE = ROOT / (
    "output/openvino-adaptive-attention-compile-gate-"
    "20260721Tseq1727-block2d-all512-clean/gate.json")
GRAPH_DELTA_GATE = ROOT / (
    "output/openvino-adaptive-attention-graph-delta-gate-"
    "20260721Tseq1729-block2d-all512-clean/gate.json")
DEFAULT_PLUGIN = ROOT / (
    "output/openvino-adaptive-attention-compile-gate-"
    "20260721Tseq1727-block2d-all512-clean/raw/build/"
    "libopenvino_intel_gpu_plugin-adaptive.so")

LAYERS = (3, 7, 11, 15, 19, 23, 27, 31, 35, 39)
BUCKET = 65536
PREFILL_HISTORY_CAPACITY = 65536
EXACT_HISTORY_CAPACITY = 66560
PREFLIGHT_GIB = 8.0
MEMORY_STOP_GIB = 4.0


def load_module(name: str, path: Path) -> Any:
  spec = importlib.util.spec_from_file_location(name, path)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import {path}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


PRODUCT = load_module("iq36_adaptive_all10_product", PRODUCT_TOOL)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out-dir", type=Path, required=True)
  parser.add_argument("--candidate-plugin", type=Path, default=DEFAULT_PLUGIN)
  parser.add_argument(
      "--adaptive-attention-topk", type=int,
      choices=(128, 252, 256, 512, 1024, 2048), default=512)
  parser.add_argument(
      "--adaptive-attention-high-topk-layers",
      type=PRODUCT.parse_target_layers, default=())
  parser.add_argument(
      "--adaptive-attention-high-topk", type=int,
      choices=(128, 252, 256, 512, 1024, 2048), default=256)
  parser.add_argument(
      "--adaptive-attention-v16-layers", type=PRODUCT.parse_target_layers,
      default=())
  parser.add_argument(
      "--adaptive-attention-key-exact-layers",
      type=PRODUCT.parse_target_layers, default=())
  parser.add_argument(
      "--adaptive-attention-key-residual1-layers",
      type=PRODUCT.parse_target_layers, default=())
  parser.add_argument(
      "--adaptive-attention-value-residual1-layers",
      type=PRODUCT.parse_target_layers, default=())
  parser.add_argument(
      "--adaptive-attention-packed-kv-layers",
      type=PRODUCT.parse_target_layers, default=())
  parser.add_argument(
      "--adaptive-attention-packed-kv-variant",
      choices=("k6v7", "k7v7", "k7v8", "k8v7"))
  parser.add_argument(
      "--adaptive-attention-exact-layers", type=PRODUCT.parse_target_layers,
      default=())
  parser.add_argument("--timeout-s", type=int, default=3600)
  parser.add_argument("--allow-dirty", action="store_true")
  args = parser.parse_args()
  if args.timeout_s <= 0:
    parser.error("timeout must be positive")
  if not set(args.adaptive_attention_v16_layers).issubset(LAYERS):
    parser.error("adaptive V16 layers must be a subset of all ten layers")
  if not set(args.adaptive_attention_high_topk_layers).issubset(LAYERS):
    parser.error("adaptive high-top-k layers must be a subset of all ten")
  if (args.adaptive_attention_high_topk_layers and
      args.adaptive_attention_high_topk == args.adaptive_attention_topk):
    parser.error("adaptive high top-k must differ from the base top-k")
  topk_by_layer = {
      layer: (args.adaptive_attention_high_topk
              if layer in args.adaptive_attention_high_topk_layers else
              args.adaptive_attention_topk)
      for layer in LAYERS
  }
  if any(topk_by_layer[layer] != 512
         for layer in args.adaptive_attention_v16_layers):
    parser.error("adaptive V16 layers require top-k 512")
  if not set(args.adaptive_attention_key_exact_layers).issubset(LAYERS):
    parser.error("adaptive key-exact layers must be a subset of all ten")
  if any(topk_by_layer[layer] != 256
         for layer in args.adaptive_attention_key_exact_layers):
    parser.error("adaptive key-exact layers require top-k 256")
  residual1_layers = (
      set(args.adaptive_attention_key_residual1_layers) |
      set(args.adaptive_attention_value_residual1_layers))
  packed_kv_layers = set(args.adaptive_attention_packed_kv_layers)
  if not residual1_layers.issubset(LAYERS):
    parser.error("adaptive residual1 layers must be a subset of all ten")
  if any(topk_by_layer[layer] not in (256, 512)
         for layer in residual1_layers):
    parser.error("adaptive residual1 layers require top-k 256 or 512")
  if set(args.adaptive_attention_v16_layers) & residual1_layers:
    parser.error("adaptive V16 and residual1 layers must be disjoint")
  if (set(args.adaptive_attention_key_exact_layers) &
      (set(args.adaptive_attention_v16_layers) | residual1_layers)):
    parser.error(
        "adaptive key-exact, V16, and residual1 layers must be disjoint")
  if not packed_kv_layers.issubset(LAYERS):
    parser.error("adaptive packed K/V layers must be a subset of all ten")
  if bool(packed_kv_layers) != bool(args.adaptive_attention_packed_kv_variant):
    parser.error("adaptive packed K/V layers and variant are required together")
  if any(topk_by_layer[layer] not in (256, 512)
         for layer in packed_kv_layers):
    parser.error("adaptive packed K/V layers require top-k 256 or 512")
  if (args.adaptive_attention_packed_kv_variant != "k7v8" and
      any(topk_by_layer[layer] == 512 for layer in packed_kv_layers)):
    parser.error("only packed K7/V8 currently admits top-k 512")
  if not set(args.adaptive_attention_exact_layers).issubset(LAYERS):
    parser.error("adaptive exact layers must be a subset of all ten layers")
  if (set(args.adaptive_attention_exact_layers) &
      set(args.adaptive_attention_v16_layers)):
    parser.error("adaptive exact layers and V16 layers must be disjoint")
  if (set(args.adaptive_attention_exact_layers) &
      set(args.adaptive_attention_high_topk_layers)):
    parser.error("adaptive exact and high-top-k layers must be disjoint")
  if (set(args.adaptive_attention_exact_layers) &
      set(args.adaptive_attention_key_exact_layers)):
    parser.error("adaptive exact and key-exact layers must be disjoint")
  if set(args.adaptive_attention_exact_layers) & residual1_layers:
    parser.error("adaptive exact layers and residual1 layers must be disjoint")
  if (packed_kv_layers &
      (set(args.adaptive_attention_exact_layers) |
       set(args.adaptive_attention_v16_layers) |
       set(args.adaptive_attention_key_exact_layers) | residual1_layers)):
    parser.error(
        "adaptive packed K/V layers must be disjoint from exact, V16, "
        "key-exact, and residual1 layers")
  if set(args.adaptive_attention_exact_layers) == set(LAYERS):
    parser.error("adaptive exact layers must leave an adaptive layer")
  return args


def check(name: str, passed: bool, **details: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **details}


def source_sha(gate: dict[str, Any], relative_path: str) -> str:
  rows = gate.get("source_files", [])
  return next(
      (str(row.get("sha256", "")) for row in rows
       if row.get("path") == relative_path), "")


def adaptive_runtime_names(census: dict[str, Any]) -> list[str]:
  return sorted(
      str(row.get("name")) for row in census.get("attention_rows", [])
      if row.get("layer_type") == "CustomGPUPrimitive" and
      str(row.get("name", "")).startswith("iq36_hot_attention_layer"))


def summary_markdown(payload: dict[str, Any]) -> str:
  result = payload.get("worker", {}).get("result", {})
  monitor = payload.get("worker", {}).get("monitor", {})
  return "\n".join([
      "# Adaptive attention all-ten compile gate",
      "",
      f"- verdict: `{payload['verdict']}`",
      f"- required checks: `{'pass' if payload['required_checks_passed'] else 'fail'}`",
      f"- compile time: `{result.get('language_compile_ms')} ms`",
      f"- runtime adaptive owners: `{result.get('runtime_census', {}).get('hot_attention_custom_count')}`",
      f"- worker peak RSS: `{monitor.get('process_rss_peak_bytes')} bytes`",
      f"- worker peak swap: `{monitor.get('process_swap_peak_bytes')} bytes`",
      "- InferRequest created: `false`",
      "- inference executed: `false`",
      "",
      "This boundary proves full-model graph integration and compilation only;",
      "it is not correctness or end-to-end performance evidence.",
      "",
  ])


def main() -> int:
  args = parse_args()
  out = args.out_dir.resolve()
  if out.exists():
    raise SystemExit(f"output already exists: {out}")
  raw = out / "raw"
  raw.mkdir(parents=True, exist_ok=False)

  plugin = args.candidate_plugin.resolve()
  source_gate = PRODUCT.load_json(SOURCE_GATE)
  compile_gate = PRODUCT.load_json(COMPILE_GATE)
  graph_delta_gate = PRODUCT.load_json(GRAPH_DELTA_GATE)
  git = PRODUCT.BOOT.git_state(out)
  plugin_sha = PRODUCT.sha256_file(plugin) if plugin.is_file() else ""
  product_sha = PRODUCT.sha256_file(PRODUCT_TOOL)
  expected_plugin_sha = str(compile_gate.get("plugin_sha256", ""))
  model_identity = PRODUCT.BOOT.capture_model_identity(
      PRODUCT.MODEL_DIR.resolve(), PRODUCT.MODEL_CONTRACT.resolve())
  PRODUCT.write_json(out / "model-identity.json", model_identity)

  config = {
      "alias_linear_state_assign": True,
      "bucket": BUCKET,
      "candidate_dq_realloc_fastpath": True,
      "candidate_path": "hot_cold_custom",
      "capture_attention_history_layers": [],
      "capture_attention_history_steps": [],
      "capture_attention_layers": [],
      "capture_attention_steps": [],
      "capture_execution_census": False,
      "case_id": "sentinel_064k_compile_only",
      "compile_only": True,
      "custom_composition": "adaptive_i8_fixed",
      "adaptive_attention_topk": args.adaptive_attention_topk,
      "adaptive_attention_high_topk_layers": list(
          args.adaptive_attention_high_topk_layers),
      "adaptive_attention_high_topk": args.adaptive_attention_high_topk,
      "adaptive_attention_v16_layers": list(
          args.adaptive_attention_v16_layers),
      "adaptive_attention_key_exact_layers": list(
          args.adaptive_attention_key_exact_layers),
      "adaptive_attention_key_residual1_layers": list(
          args.adaptive_attention_key_residual1_layers),
      "adaptive_attention_value_residual1_layers": list(
          args.adaptive_attention_value_residual1_layers),
      "adaptive_attention_packed_kv_layers": list(
          args.adaptive_attention_packed_kv_layers),
      "adaptive_attention_packed_kv_variant": (
          args.adaptive_attention_packed_kv_variant),
      "adaptive_attention_exact_layers": list(
          args.adaptive_attention_exact_layers),
      "decode_chunk256_layers": [],
      "decode_dual256_layers": [],
      "decode_f32_numerator_layers": [],
      "decode_stock256_layers": [],
      "decode_stock_micro_layers": [],
      "decode_stock_partition_layers": [],
      "decode_stock_score_layers": [],
      "direct_ssm_state_assign": False,
      "exact_history_capacity": EXACT_HISTORY_CAPACITY,
      "exact_history_layers": list(LAYERS),
      "exact_phase_context_partition4": False,
      "fixed_fc_cohorts": [],
      "fixed_fc_manager_direct": False,
      "fixed_fc_manager_scope": "all",
      "fuse_fixed_fc": False,
      "fuse_linear_conv_state": True,
      "host_time_profiling": 0,
      "linear_state_alias_scope": "all",
      "lm_head_i8q1": False,
      "lm_head_i8q4": True,
      "mode": "candidate",
      "output_tokens": 512,
      "prefill_history_capacity": PREFILL_HISTORY_CAPACITY,
      "self_bind_hot_states": False,
      "target_layers": list(LAYERS),
      "timing_token_output": False,
  }
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
  )
  worker = PRODUCT.run_worker(worker_args, raw / "worker", config)
  result = worker.get("result", {})
  source = result.get("source_summary") or {}
  census = result.get("runtime_census") or {}
  monitor = worker.get("monitor") or {}
  expected_runtime_names = sorted(
      f"iq36_hot_attention_layer{layer}" for layer in LAYERS)
  expected_adaptive_layers = [
      layer for layer in LAYERS
      if layer not in args.adaptive_attention_exact_layers]
  expected_topk = {
      str(layer): (args.adaptive_attention_high_topk
                   if layer in args.adaptive_attention_high_topk_layers else
                   args.adaptive_attention_topk)
      for layer in expected_adaptive_layers}
  memory_guard = worker.get("memory_guard") or {}
  worker_min_available = int(monitor.get("system_available_min_bytes") or 0)
  stop_bytes = int(MEMORY_STOP_GIB * 1024**3)

  checks = [
      check("repository_clean_at_gate",
            not git["dirty"] or args.allow_dirty,
            git=git, allow_dirty=args.allow_dirty),
      check("seq1730_source_gate_is_exact_and_admitted",
            source_gate.get("compile_admitted") is True and
            source_gate.get("required_checks_passed") is True and
            source_gate.get("git", {}).get("commit") == git["commit"] and
            source_sha(source_gate, str(PRODUCT_TOOL.relative_to(ROOT))) ==
                product_sha,
            source_gate=PRODUCT.relative(SOURCE_GATE),
            product_sha256=product_sha),
      check("seq1727_plugin_is_exact",
            plugin.is_file() and plugin_sha == expected_plugin_sha,
            plugin=PRODUCT.relative(plugin), sha256=plugin_sha,
            expected_sha256=expected_plugin_sha),
      check("seq1729_graph_delta_admits_all10_compile",
            graph_delta_gate.get("required_checks_passed") is True and
            graph_delta_gate.get("all10_compile_worker_admitted") is True and
            graph_delta_gate.get("verdict") ==
                "admit_adaptive_attention_all10_graph_compile_boundary"),
      check("locked_model_identity", model_identity["required_checks_passed"],
            model_dir=model_identity["model_dir"]),
      check("single_serial_compile_worker_completes",
            worker.get("returncode") == 0 and
            worker.get("timed_out") is False and
            worker.get("oom_observed") is False),
      check("compile_only_worker_executes_no_request_or_inference",
            result.get("compile_only") is True and
            result.get("worker_created_infer_request") is False and
            result.get("worker_executed_inference") is False and
            "generated_token_ids" not in result and
            "state_schema_after" not in result),
      check(
          f"all_ten_adaptive_top{args.adaptive_attention_topk}_"
          "source_owners_are_exact",
          source.get("target_layers") == list(LAYERS) and
          source.get("adaptive_attention_layers") ==
              expected_adaptive_layers and
          source.get("adaptive_topk_by_layer") == expected_topk and
          source.get("adaptive_attention_high_topk_layers") ==
              list(args.adaptive_attention_high_topk_layers) and
          source.get("adaptive_attention_high_topk") ==
              args.adaptive_attention_high_topk and
          source.get("adaptive_attention_v16_layers") ==
              list(args.adaptive_attention_v16_layers) and
          source.get("adaptive_attention_key_exact_layers") ==
              list(args.adaptive_attention_key_exact_layers) and
          source.get("adaptive_attention_key_residual1_layers") ==
              list(args.adaptive_attention_key_residual1_layers) and
          source.get("adaptive_attention_value_residual1_layers") ==
              list(args.adaptive_attention_value_residual1_layers) and
          source.get("adaptive_attention_packed_kv_layers") ==
              list(args.adaptive_attention_packed_kv_layers) and
          source.get("adaptive_attention_packed_kv_variant") ==
              args.adaptive_attention_packed_kv_variant and
          source.get("adaptive_attention_exact_layers", []) ==
              list(args.adaptive_attention_exact_layers) and
          source.get("custom_count_after") == len(LAYERS) and
          source.get("stock_sdpa_count_after") == 0 and
          source.get("direct_i8_fixed_layout") is True and
          source.get("fixed_cold_capacity") == BUCKET and
          source.get("prefill_history_capacity") ==
              PREFILL_HISTORY_CAPACITY and
          source.get("exact_history_layers") == list(LAYERS) and
          source.get("exact_history_capacity") == EXACT_HISTORY_CAPACITY,
          observed_topk=source.get("adaptive_topk_by_layer")),
      check("compiled_runtime_retains_exactly_ten_custom_attention_owners",
            census.get("hot_attention_custom_count") == len(LAYERS) and
            census.get("stock_sdpa_like_count") == 0 and
            adaptive_runtime_names(census) == expected_runtime_names,
            observed_names=adaptive_runtime_names(census)),
      check("accepted_product_carrier_is_compiled",
            result.get("candidate_path") == "hot_cold_custom" and
            result.get("custom_composition") == "adaptive_i8_fixed" and
            result.get("adaptive_attention_topk") ==
                args.adaptive_attention_topk and
            result.get("adaptive_attention_high_topk_layers") ==
                list(args.adaptive_attention_high_topk_layers) and
            result.get("adaptive_attention_high_topk") ==
                args.adaptive_attention_high_topk and
            result.get("adaptive_attention_v16_layers") ==
                list(args.adaptive_attention_v16_layers) and
            result.get("adaptive_attention_key_exact_layers") ==
                list(args.adaptive_attention_key_exact_layers) and
            result.get("adaptive_attention_key_residual1_layers") ==
                list(args.adaptive_attention_key_residual1_layers) and
            result.get("adaptive_attention_value_residual1_layers") ==
                list(args.adaptive_attention_value_residual1_layers) and
            result.get("adaptive_attention_packed_kv_layers") ==
                list(args.adaptive_attention_packed_kv_layers) and
            result.get("adaptive_attention_packed_kv_variant") ==
                args.adaptive_attention_packed_kv_variant and
            result.get("adaptive_attention_exact_layers") ==
                list(args.adaptive_attention_exact_layers) and
            result.get("candidate_gpu_plugin_sha256") == plugin_sha and
            result.get("fuse_linear_conv_state") is True and
            result.get("alias_linear_state_assign") is True and
            result.get("linear_state_alias_scope") == "all" and
            result.get("candidate_dq_realloc_fastpath") is True and
            result.get("lm_head_i8q4") is True),
      check("full_graph_compile_duration_is_finite",
            isinstance(result.get("language_compile_ms"), (int, float)) and
            math.isfinite(float(result.get("language_compile_ms", math.nan))) and
            float(result.get("language_compile_ms", 0.0)) > 0.0,
            language_compile_ms=result.get("language_compile_ms")),
      check("worker_does_not_swap",
            int(monitor.get("process_swap_peak_bytes", -1)) == 0,
            process_swap_peak_bytes=monitor.get("process_swap_peak_bytes")),
      check("memory_guard_never_tripped",
            memory_guard.get("tripped") is False and
            worker_min_available >= stop_bytes,
            stop_bytes=stop_bytes,
            minimum_available_bytes=worker_min_available),
  ]
  required = all(row["pass"] for row in checks)
  admitted = required and not args.allow_dirty
  verdict = (
      "admit_adaptive_attention_all10_64k_correctness_boundary"
      if admitted else
      "development_all10_compile_only" if required else
      "repair_adaptive_attention_all10_graph_compile")
  payload = {
      "checks": checks,
      "correctness_worker_admitted": admitted,
      "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
      "git": git,
      "long_worker_admitted": False,
      "model_identity": PRODUCT.relative(out / "model-identity.json"),
      "next_route": PASS_ROUTE if admitted else FAIL_ROUTE,
      "product_worker_admitted": False,
      "adaptive_attention_v16_layers": list(
          args.adaptive_attention_v16_layers),
      "adaptive_attention_high_topk_layers": list(
          args.adaptive_attention_high_topk_layers),
      "adaptive_attention_high_topk": args.adaptive_attention_high_topk,
      "adaptive_attention_key_exact_layers": list(
          args.adaptive_attention_key_exact_layers),
      "adaptive_attention_key_residual1_layers": list(
          args.adaptive_attention_key_residual1_layers),
      "adaptive_attention_value_residual1_layers": list(
          args.adaptive_attention_value_residual1_layers),
      "adaptive_attention_packed_kv_layers": list(
          args.adaptive_attention_packed_kv_layers),
      "adaptive_attention_packed_kv_variant": (
          args.adaptive_attention_packed_kv_variant),
      "adaptive_attention_exact_layers": list(
          args.adaptive_attention_exact_layers),
      "required_checks_passed": required,
      "route": ROUTE,
      "schema": SCHEMA,
      "sources": {
          "compile_gate": PRODUCT.relative(COMPILE_GATE),
          "graph_delta_gate": PRODUCT.relative(GRAPH_DELTA_GATE),
          "source_gate": PRODUCT.relative(SOURCE_GATE),
      },
      "verdict": verdict,
      "worker": worker,
      "workstream": WS,
  }
  PRODUCT.write_json(out / "gate.json", payload)
  (out / "summary.md").write_text(
      summary_markdown(payload), encoding="utf-8")
  print(json.dumps({
      "correctness_worker_admitted": admitted,
      "language_compile_ms": result.get("language_compile_ms"),
      "output": PRODUCT.relative(out),
      "required_checks_passed": required,
      "runtime_adaptive_owners": census.get("hot_attention_custom_count"),
      "verdict": verdict,
  }, sort_keys=True), flush=True)
  return 0 if required else 2


if __name__ == "__main__":
  raise SystemExit(main())
