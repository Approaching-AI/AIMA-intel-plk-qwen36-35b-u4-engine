#!/usr/bin/env python3
"""Compile the all-ten dual-cohort product graph without creating a request."""

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
SCHEMA = (
    "intel-qwen36-openvino-exact-attention-dual-cohort-all10-"
    "compile-gate-v1")
PRODUCT_TOOL = ROOT / "tools/intel-qwen36-openvino-hot-cold-product-gate.py"
GRAPH_TOOL = ROOT / "tools/intel_qwen36_openvino_hot_cold_attention.py"
KERNEL = ROOT / (
    "engine/openvino/custom/iq36_stock_micro_attention_oracle.cl")
CONFIG = ROOT / "engine/openvino/custom/iq36_hot_attention_gqa.xml"
COMPONENT_GATE = ROOT / (
    "output/openvino-exact-attention-dual-cohort-component-"
    "20260723Tseq2135-clean/result.json")
PLUGIN = Path(
    "/home/intel/intel-qwen36-r0/output/"
    "openvino-90214e-l0-gpu-seq2119/bin/intel64/Release/"
    "libopenvino_intel_gpu_plugin.so")
EXPECTED_PLUGIN_SHA256 = (
    "01c04ced415a7b7a5e5bda77a995b2b97b68eb3d9f2c5f3396844d042ddda269")
LAYERS = (3, 7, 11, 15, 19, 23, 27, 31, 35, 39)
BUCKET = 131072
EXACT_HISTORY_CAPACITY = 132096
PREFLIGHT_GIB = 8.0
MEMORY_STOP_GIB = 4.0


def load_module(name: str, path: Path) -> Any:
  spec = importlib.util.spec_from_file_location(name, path)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import {path}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


PRODUCT = load_module("iq36_dual_cohort_product", PRODUCT_TOOL)


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


def runtime_names(census: dict[str, Any]) -> list[str]:
  return sorted(
      str(row.get("name")) for row in census.get("attention_rows", [])
      if row.get("layer_type") == "CustomGPUPrimitive" and
      str(row.get("name", "")).startswith("iq36_hot_attention_layer"))


def main() -> int:
  args = parse_args()
  out = args.out_dir.resolve()
  if out.exists():
    raise SystemExit(f"output already exists: {out}")
  raw = out / "raw"
  raw.mkdir(parents=True, exist_ok=False)

  required_paths = (
      PRODUCT_TOOL, GRAPH_TOOL, KERNEL, CONFIG, COMPONENT_GATE, PLUGIN)
  missing = [str(path) for path in required_paths if not path.is_file()]
  if missing:
    raise SystemExit(
        "missing dual-cohort graph inputs: " + ", ".join(missing))

  git = PRODUCT.BOOT.git_state(out)
  component = PRODUCT.load_json(COMPONENT_GATE)
  plugin_sha = PRODUCT.sha256_file(PLUGIN)
  kernel_text = KERNEL.read_text(encoding="utf-8")
  dual_anchor = kernel_text.index("  const uint dual_block_count")
  dual_begin = kernel_text.rfind(
      "#if defined(IQ36_STOCK_MICRO_DUAL_COHORT)", 0, dual_anchor)
  dual_end = kernel_text.index(
      "\n#else\n#if defined(IQ36_STOCK_MICRO_PAGE_SPARSE)", dual_anchor)
  dual_decode_source = kernel_text[dual_begin:dual_end]
  config_text = CONFIG.read_text(encoding="utf-8")
  graph_text = GRAPH_TOOL.read_text(encoding="utf-8")
  product_text = PRODUCT_TOOL.read_text(encoding="utf-8")
  source_checks = {
      "component_promotes_graph_integration":
          component.get("required_checks_passed") is True
          and component.get("verdict") ==
              "promote_exact_attention_dual_cohort_component"
          and component.get("component_promoted") is True
          and component.get("graph_integration_admitted") is True
          and component.get("model_worker_launched") is False
          and component.get("git", {}).get("dirty") is False
          and math.isclose(
              float(component.get(
                  "performance_inference", {}).get(
                      "upper_confidence_bound_ms", math.nan)),
              -0.46187500000000004, rel_tol=0.0, abs_tol=1.0e-12),
      "product_kernel_preserves_dynamic_exact_owner_contract":
          "IQ36_STOCK_MICRO_DUAL_COHORT" in kernel_text
          and "dual_producer &&" in kernel_text
          and "cohort_linear_local_id < 128U" in kernel_text
          and "const uint dual_block_count = IQ36_DIV_UP(" in kernel_text
          and "key_begin + (int)(sg_i_kq * ugemm_kq_sg_tile_m)"
              in kernel_text
          and "__attribute__((reqd_work_group_size(16, 32, 1)))"
              in kernel_text
          and kernel_text.count(
              "__local NamedBarrier_t* pipeline_barrier") == 1
          and kernel_text.count("named_barrier_init(32)") == 1
          and "float reduced_running_max = -INFINITY;"
              in dual_decode_source
          and "subgroup_row < IQ36_SG_PER_WG" in dual_decode_source
          and "tile_atomic_max_full(" not in dual_decode_source
          and "cooperative_prefetch_2d_rem(\n"
              "          value_chunk" not in dual_decode_source,
      "dual_custom_layer_keeps_prefill_and_two_decode_groups":
          '<CustomLayer name="IQ36ExactPhaseDualCohortHotAttentionGQA"'
              in config_text
          and "-DIQ36_STOCK_MICRO_DUAL_COHORT=1" in config_text
          and 'global="256-240*(X-1)/2065,'
              'Y+(64-Y)*(X-1)/2065,' in config_text
          and 'local="256-240*(X-1)/2065,'
              '1+31*(X-1)/2065,1"' in config_text,
      "graph_and_worker_expose_default_off_dual_config":
          "def exact_phase_dual_cohort_custom_class" in graph_text
          and "exact_phase_dual_cohort: bool = False" in graph_text
          and '"exact_phase_dual_cohort": exact_phase_dual_cohort'
              in graph_text
          and 'cfg.get("exact_phase_dual_cohort", False)' in product_text,
  }
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
      "capture_execution_census": False,
      "case_id": "sentinel_128k_dual_cohort_compile_only",
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
      "host_time_profiling": 0,
      "linear_state_alias_scope": "all",
      "lm_head_i8q1": True,
      "lm_head_i8q1_gated_exact": False,
      "lm_head_i8q1_gated_q4": False,
      "lm_head_i8q1_greedy_local2": False,
      "lm_head_i8q4": False,
      "mode": "candidate",
      "output_tokens": 512,
      "prefill_history_capacity": BUCKET,
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
  )
  worker = PRODUCT.run_worker(worker_args, raw / "worker", config)
  result = worker.get("result", {})
  source = result.get("source_summary") or {}
  census = result.get("runtime_census") or {}
  monitor = worker.get("monitor") or {}
  memory_guard = worker.get("memory_guard") or {}
  expected_names = sorted(
      f"iq36_hot_attention_layer{layer}" for layer in LAYERS)
  minimum_available = int(
      monitor.get("system_available_min_bytes") or 0)
  stop_bytes = int(MEMORY_STOP_GIB * 1024**3)

  checks = [
      check("repository_clean_at_gate",
            not git["dirty"] or args.allow_dirty,
            git=git, allow_dirty=args.allow_dirty),
      check("promoted_component_and_source_contract_are_bound",
            all(source_checks.values()), source_checks=source_checks),
      check("accepted_seq2119_plugin_is_exact",
            plugin_sha == EXPECTED_PLUGIN_SHA256,
            plugin=str(PLUGIN), sha256=plugin_sha),
      check("locked_model_identity",
            model_identity.get("required_checks_passed") is True,
            model_dir=model_identity.get("model_dir")),
      check("single_serial_compile_worker_completes",
            worker.get("returncode") == 0
            and worker.get("timed_out") is False
            and worker.get("oom_observed") is False),
      check("compile_only_worker_executes_no_request_or_inference",
            result.get("compile_only") is True
            and result.get("worker_created_infer_request") is False
            and result.get("worker_executed_inference") is False
            and "generated_token_ids" not in result
            and "state_schema_after" not in result),
      check("all_ten_dual_cohort_source_owners_are_exact",
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
            and source.get("stock_sdpa_count_after") == 0),
      check("compiled_runtime_retains_exactly_ten_custom_attention_owners",
            census.get("hot_attention_custom_count") == len(LAYERS)
            and census.get("stock_sdpa_like_count") == 0
            and runtime_names(census) == expected_names,
            observed_names=runtime_names(census)),
      check("accepted_product_carrier_is_compiled",
            result.get("candidate_path") == "hot_cold_custom"
            and result.get("custom_composition") == "exact_phase"
            and result.get("candidate_gpu_plugin_sha256") == plugin_sha
            and result.get("fuse_linear_conv_state") is True
            and result.get("alias_linear_state_assign") is True
            and result.get("linear_state_alias_scope") == "all"
            and result.get("candidate_dq_realloc_fastpath") is True
            and result.get("candidate_fc_stable_prepare_fastpath") is True
            and result.get("lm_head_i8q1") is True),
      check("full_graph_compile_duration_is_finite",
            isinstance(result.get("language_compile_ms"), (int, float))
            and math.isfinite(
                float(result.get("language_compile_ms", math.nan)))
            and float(result.get("language_compile_ms", 0.0)) > 0.0,
            language_compile_ms=result.get("language_compile_ms")),
      check("worker_memory_telemetry_is_recorded",
            int(monitor.get("process_rss_peak_bytes", -1)) >= 0
            and int(monitor.get("process_swap_peak_bytes", -1)) >= 0,
            process_rss_peak_bytes=monitor.get("process_rss_peak_bytes"),
            process_swap_peak_bytes=monitor.get(
                "process_swap_peak_bytes")),
      check("memory_guard_never_tripped",
            memory_guard.get("tripped") is False
            and minimum_available >= stop_bytes,
            stop_bytes=stop_bytes,
            minimum_available_bytes=minimum_available),
  ]
  required = all(row["pass"] for row in checks)
  admitted = required and not args.allow_dirty
  verdict = (
      "admit_exact_attention_dual_cohort_short_recurrent_correctness"
      if admitted else
      "development_dual_cohort_compile_only" if required else
      "repair_exact_attention_dual_cohort_product_compile")
  payload = {
      "schema_version": SCHEMA,
      "workstream": WS,
      "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
      "git": git,
      "verdict": verdict,
      "required_checks_passed": required,
      "correctness_worker_admitted": admitted,
      "timing_worker_admitted": False,
      "product_worker_admitted": False,
      "checks": checks,
      "model_identity": PRODUCT.relative(out / "model-identity.json"),
      "plugin": {
          "path": str(PLUGIN),
          "sha256": plugin_sha,
      },
      "source_checks": source_checks,
      "worker": worker,
      "next_route": (
          "openvino_exact_attention_dual_cohort_128k_short_recurrent_"
          "correctness" if admitted else
          "openvino_exact_attention_dual_cohort_product_compile_repair"),
      "sources": {
          PRODUCT.relative(path): PRODUCT.sha256_file(path)
          for path in (PRODUCT_TOOL, GRAPH_TOOL, KERNEL, CONFIG, COMPONENT_GATE)
      },
  }
  PRODUCT.write_json(out / "gate.json", payload)
  (out / "summary.md").write_text(
      "\n".join([
          "# Exact-attention dual-cohort all-ten compile gate",
          "",
          f"- verdict: `{verdict}`",
          f"- required checks: `{'pass' if required else 'fail'}`",
          f"- compile time: `{result.get('language_compile_ms')} ms`",
          f"- runtime attention owners: "
          f"`{census.get('hot_attention_custom_count')}`",
          f"- worker peak RSS: "
          f"`{monitor.get('process_rss_peak_bytes')} bytes`",
          f"- worker peak swap: "
          f"`{monitor.get('process_swap_peak_bytes')} bytes`",
          "- InferRequest created: `false`",
          "- inference executed: `false`",
          "",
      ]), encoding="utf-8")
  print(json.dumps({
      "output": PRODUCT.relative(out),
      "verdict": verdict,
      "required_checks_passed": required,
      "correctness_worker_admitted": admitted,
      "language_compile_ms": result.get("language_compile_ms"),
      "runtime_attention_owners": census.get("hot_attention_custom_count"),
  }, sort_keys=True), flush=True)
  return 0 if required else 2


if __name__ == "__main__":
  raise SystemExit(main())
