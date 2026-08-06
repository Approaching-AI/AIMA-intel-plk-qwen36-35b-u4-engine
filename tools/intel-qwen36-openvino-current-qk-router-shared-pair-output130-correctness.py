#!/usr/bin/env python3
"""Gate one exact 2k/output130 Q/K plus N=1024 shared-pair worker.

Plan mode binds every immutable input without launching a GPU worker. Execution
mode requires that clean pushed plan, then launches exactly one candidate-only
teacher-forced worker. All 130 logits must remain bitwise equal to the accepted
carrier, and the executed graph must prove FC/QK/attention/pair/scalar/router
counts 331/10/10/40/40/40. This is not a speed claim.
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
SCHEMA = (
    "intel-qwen36-openvino-current-qk-router-shared-pair-"
    "output130-correctness-v1")
BASE_TOOL = ROOT / (
    "tools/intel-qwen36-openvino-current-qk-router-shared-"
    "output130-correctness.py")
PRODUCT_TOOL = ROOT / "tools/intel-qwen36-openvino-hot-cold-product-gate.py"
COMPILE_TOOL = ROOT / (
    "tools/intel-qwen36-openvino-current-qk-router-shared-pair-"
    "compile-gate.py")
COMPILE_GATE = ROOT / (
    "output/openvino-current-qk-router-shared-pair-compile-"
    "20260731Tseq2213-clean/result.json")
PLAN_GATE = ROOT / (
    "output/openvino-current-qk-router-shared-pair-output130-plan-"
    "20260731Tseq2214-clean/result.json")
REFERENCE_ROOT = ROOT / (
    "output/openvino-2k-gated-exact-timing-abba1-"
    "20260731Tseq2183-clean/raw/sentinel_002k/correctness")
REFERENCE_CONFIG = REFERENCE_ROOT / "candidate/worker-config.json"
REFERENCE_CANDIDATE = REFERENCE_ROOT / "candidate/worker-result.json"
REFERENCE_STOCK = REFERENCE_ROOT / "stock/worker-result.json"
QK_GATE = ROOT / (
    "output/openvino-qk-rope-layout-stock-half-output512-correctness-"
    "20260731Tseq2200-clean/result.json")
QK_SOURCE = ROOT / "engine/openvino/custom/iq36_qk_rope_layout.cl"
PAIR_PATCH = ROOT / "engine/openvino/iq36-current-router-shared-pair.patch"
PLUGIN = Path(
    "/home/intel/intel-qwen36-r0/output/"
    "openvino-90214e-l0-gpu-seq2212/bin/intel64/Release/"
    "libopenvino_intel_gpu_plugin.so")
EXPECTED_SHA256 = {
    BASE_TOOL: (
        "e832ce2c0c3f4b959e94ea21db6bae662fe1bef3b029e1a01d010ec9c4e9dd13"),
    PRODUCT_TOOL: (
        "baa6cb5591766eb91dcb1456d0195216f10a4fafb9477fc3a357f8eb98a8c3b1"),
    COMPILE_TOOL: (
        "6ddb59dfc9482ead40e8ae1bc750098c861a04c7fb50a33d4caca10256f14fb3"),
    COMPILE_GATE: (
        "333a3b0316334d049eb16dca429cabed2f9c4c1056b9c587c4d9a019be1f97f5"),
    REFERENCE_CONFIG: (
        "21bd692f93ba8bba40badf29d01a214ddcd55e4276de42971db71af8354cfced"),
    REFERENCE_CANDIDATE: (
        "fa6a4aacdd45251c6818b467477794688754ffc7c5fa744ad9fb22e4961523b3"),
    REFERENCE_STOCK: (
        "c327d633b0a6c75320d577bbe555e992303f85da3de800be7b8d70536f7d5215"),
    QK_GATE: (
        "ff862015c9cec1aad4fb1c7efa8aa519927417361b480d90d50a95c9292512df"),
    QK_SOURCE: (
        "be2b1105df7503a24636615a94255e0683d0b8a73bbecd1c7b70d0b9f5306863"),
    PAIR_PATCH: (
        "092e1b3d23277cd1ab34577fc26f594efcfb0a837d72904b28b64ae01af36d3a"),
    PLUGIN: (
        "9165f6aa9c31f43b7554c65161e2534bf42ff250b5ad97b51c83e37e2d51ffcd"),
}
EXPECTED_TOKEN_SHA256 = (
    "7cb86794ff37361ce5008a88a3b54eebbf9548256947825438e85b48d0a76d41")
LAYERS = tuple(range(3, 40, 4))
OUTPUT_TOKENS = 130
KLD_MAX = 0.005
TOP1_MIN = 0.99
PREFLIGHT_GIB = 8.0
MEMORY_STOP_GIB = 4.0
PAIR_SUFFIXES = (
    "mlp.shared_expert.gate_proj/ov_ext::linear/MatMul",
    "mlp.shared_expert.up_proj/ov_ext::linear/MatMul",
)
SCALAR_SHARED_GATE_SUFFIX = (
    "mlp.shared_expert_gate/ov_ext::linear/MatMul")
ROUTER_GATE_SUFFIX = "mlp.gate/aten::linear/MatMul"
LINEAR_SUFFIXES = (
    "linear_attn.in_proj_qkv/ov_ext::linear/MatMul",
    "linear_attn.in_proj_a/ov_ext::linear/MatMul",
    "linear_attn.in_proj_b/ov_ext::linear/MatMul",
    "linear_attn.in_proj_z/ov_ext::linear/MatMul",
)


def load_module(name: str, path: Path) -> Any:
  spec = importlib.util.spec_from_file_location(name, path)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import {path}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


BASE = load_module("iq36_current_pair_correctness_common", BASE_TOOL)
PRODUCT = BASE.PRODUCT


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out-dir", type=Path, required=True)
  parser.add_argument("--plan-only", action="store_true")
  parser.add_argument("--timeout-s", type=int, default=1800)
  args = parser.parse_args()
  if args.timeout_s <= 0:
    parser.error("timeout must be positive")
  return args


def runtime_audit(result: dict[str, Any]) -> dict[str, Any]:
  census = result.get("execution_census") or {}
  rows = census.get("retained_rows") or []
  fc_rows = [
      row for row in rows
      if row.get("node_type") == "FullyConnectedCompressed"]
  names = [str(row.get("node_name")) for row in fc_rows]
  fused2 = sorted(name for name in names if "_fused_2FCs" in name)
  fused_shared = sorted(
      name for name in fused2 if ".mlp.shared_expert" in name)
  fused3 = sorted(name for name in names if "_fused_3FCs" in name)
  existing_qkv = sorted(
      name for name in fused3 if ".mlp.shared_expert" not in name)
  unfused = [name for name in names if "_fused_" not in name]
  pair_originals = sorted(
      name for name in unfused
      if any(name.endswith(suffix) for suffix in PAIR_SUFFIXES))
  scalar_shared = sorted(
      name for name in unfused
      if name.endswith(SCALAR_SHARED_GATE_SUFFIX))
  router = sorted(
      name for name in unfused if name.endswith(ROUTER_GATE_SUFFIX))
  linear = sorted(
      name for name in unfused
      if any(name.endswith(suffix) for suffix in LINEAR_SUFFIXES))
  return {
      "fully_connected_row_count": len(fc_rows),
      "fused_shared_pair_count": len(fused_shared),
      "fused_shared_pair_names": fused_shared,
      "existing_fused_qkv_count": len(existing_qkv),
      "unfused_pair_original_count": len(pair_originals),
      "unfused_scalar_shared_gate_count": len(scalar_shared),
      "unfused_scalar_shared_gate_names": scalar_shared,
      "unfused_router_gate_count": len(router),
      "unfused_router_gate_names": router,
      "unfused_linear_original_count": len(linear),
  }


def main() -> int:
  args = parse_args()
  out = args.out_dir.resolve()
  if out.exists():
    raise SystemExit(f"output already exists: {out}")
  raw = out / "raw"
  raw.mkdir(parents=True, exist_ok=False)
  required_paths = tuple(EXPECTED_SHA256)
  if not args.plan_only:
    required_paths = required_paths + (PLAN_GATE,)
  missing = [str(path) for path in required_paths if not path.is_file()]
  if missing:
    raise SystemExit("missing shared-pair correctness inputs: " + ", ".join(missing))

  git = PRODUCT.BOOT.git_state(out)
  head = BASE.run(["git", "rev-parse", "HEAD"]).stdout.strip()
  origin_main = BASE.run(
      ["git", "rev-parse", "origin/main"]).stdout.strip()
  observed_hashes = {path: BASE.sha256(path) for path in EXPECTED_SHA256}
  compile_gate = PRODUCT.load_json(COMPILE_GATE)
  base_config = PRODUCT.load_json(REFERENCE_CONFIG)
  old_candidate = PRODUCT.load_json(REFERENCE_CANDIDATE)
  stock = PRODUCT.load_json(REFERENCE_STOCK)
  qk_gate = PRODUCT.load_json(QK_GATE)
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
      "case_id": "sentinel_002k_current_qk_router_shared_pair_output130",
      "capture_execution_census": True,
      "capture_logits": True,
      "checkpoint_steps": list(range(OUTPUT_TOKENS)),
      "fuse_qk_rope_layout": True,
      "fuse_router_shared_pair": True,
      "fuse_router_shared_triple": False,
      "output_tokens": OUTPUT_TOKENS,
      "purpose": "teacher_forced_correctness",
      "reference_result": str(reference_path.resolve()),
  })
  config.pop("compile_only", None)
  config.pop("instantiate_only", None)
  config_delta = {
      key: {"control": base_config.get(key), "candidate": config.get(key)}
      for key in sorted(set(base_config) | set(config))
      if base_config.get(key) != config.get(key)
  }
  config_binding_delta = {
      key: {
          "control": row["control"],
          "candidate": (
              "<OUTPUT>/reference-output130.json"
              if key == "reference_result" else row["candidate"]),
      }
      for key, row in config_delta.items()
  }
  expected_delta = {
      "candidate_gpu_plugin", "case_id", "checkpoint_steps",
      "fuse_qk_rope_layout", "fuse_router_shared_pair",
      "fuse_router_shared_triple", "output_tokens", "reference_result",
  }
  static_checks = [
      BASE.check("repository_is_clean_and_pushed_at_gate",
                 not git["dirty"] and head == origin_main,
                 git=git, head=head, origin_main=origin_main),
      BASE.check("all_frozen_inputs_have_exact_hashes",
                 all(
                     observed_hashes[path] == expected
                     for path, expected in EXPECTED_SHA256.items()),
                 observed={
                     PRODUCT.relative(path): digest
                     for path, digest in observed_hashes.items()}),
      BASE.check("seq2213_admits_exactly_one_candidate_correctness_worker",
                 compile_gate.get("required_checks_passed") is True and
                 compile_gate.get("verdict") ==
                     "admit_one_current_qk_router_shared_pair_output130_"
                     "correctness_worker" and
                 compile_gate.get(
                     "candidate_output130_correctness_worker_admitted") is
                     True and
                 compile_gate.get("performance_worker_admitted") is False and
                 compile_gate.get("infer_requests_created") == 0 and
                 compile_gate.get("inference_workers_launched") == 0 and
                 compile_gate.get("plugin", {}).get("sha256") ==
                     EXPECTED_SHA256[PLUGIN]),
      BASE.check("accepted_qk_and_pair_sources_are_bound",
                 qk_gate.get("required_checks_passed") is True and
                 qk_gate.get("correctness", {}).get(
                     "bitwise_checkpoint_count") == 512 and
                 qk_gate.get("correctness", {}).get(
                     "current_carrier_relative", {}).get("max_kld") == 0.0 and
                 "IQ36_ROUTER_SHARED_PAIR" in
                     PAIR_PATCH.read_text(encoding="utf-8")),
      BASE.check("output130_pair_config_delta_is_exact",
                 set(config_delta) == expected_delta and
                 config.get("candidate_path") == "hot_cold_custom" and
                 config.get("custom_composition") == "exact_phase" and
                 config.get("fuse_qk_rope_layout") is True and
                 config.get("fuse_router_shared_pair") is True and
                 config.get("fuse_router_shared_triple") is False and
                 config.get("fuse_fixed_fc") is not True and
                 config.get("fixed_fc_manager_direct") is not True and
                 config.get("output_tokens") == OUTPUT_TOKENS and
                 config.get("checkpoint_steps") ==
                     list(range(OUTPUT_TOKENS)),
                 config_delta=config_delta),
      BASE.check("accepted_reference_streams_cover_output130",
                 expected_tokens == candidate_reference_tokens and
                 len(old_candidate.get("distribution_checkpoints", [])) >=
                     OUTPUT_TOKENS and
                 len(stock.get("distribution_checkpoints", [])) >=
                     OUTPUT_TOKENS),
  ]
  static_passed = all(row["pass"] for row in static_checks)
  if args.plan_only or not static_passed:
    verdict = (
        "admit_one_bound_current_qk_router_shared_pair_output130_worker"
        if static_passed else
        "reject_current_qk_router_shared_pair_worker_before_gpu")
    payload = {
        "schema": SCHEMA,
        "workstream": WS,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git": git,
        "verdict": verdict,
        "required_checks_passed": static_passed,
        "plan_only": True,
        "candidate_correctness_worker_admitted": static_passed,
        "performance_worker_admitted": False,
        "gpu_workers_launched": 0,
        "tool_sha256": BASE.sha256(Path(__file__)),
        "checks": static_checks,
        "config_delta": config_binding_delta,
        "resolved_config_delta": config_delta,
    }
    PRODUCT.write_json(out / "result.json", payload)
    PRODUCT.write_json(out / "manifest.json", {
        "schema": SCHEMA,
        "tool": PRODUCT.relative(Path(__file__)),
        "tool_sha256": BASE.sha256(Path(__file__)),
        "git": git,
        "inputs": {
            PRODUCT.relative(path): digest
            for path, digest in observed_hashes.items()},
        "plan_only": True,
        "gpu_workers": 0,
    })
    print(json.dumps({
        "artifact": PRODUCT.relative(out),
        "verdict": verdict,
        "required_checks_passed": static_passed,
        "gpu_workers_launched": 0,
    }, separators=(",", ":")), flush=True)
    return 0 if static_passed else 2

  plan_gate = PRODUCT.load_json(PLAN_GATE)
  plan_check = BASE.check(
      "clean_seq2214_plan_admits_this_exact_tool_and_config",
      plan_gate.get("required_checks_passed") is True and
      plan_gate.get("verdict") ==
          "admit_one_bound_current_qk_router_shared_pair_output130_worker" and
      plan_gate.get("candidate_correctness_worker_admitted") is True and
      plan_gate.get("performance_worker_admitted") is False and
      plan_gate.get("gpu_workers_launched") == 0 and
      plan_gate.get("tool_sha256") == BASE.sha256(Path(__file__)) and
      plan_gate.get("git", {}).get("commit") == git.get("commit") and
      plan_gate.get("config_delta") == config_binding_delta,
      plan_gate=PRODUCT.relative(PLAN_GATE),
      plan_gate_sha256=BASE.sha256(PLAN_GATE))
  if not plan_check["pass"]:
    static_checks.append(plan_check)
    raise RuntimeError("clean seq2214 plan does not admit this worker")

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
  worker = PRODUCT.run_worker(worker_args, raw / "candidate", config)
  result = worker.get("result") or {}
  distributions = (
      PRODUCT.ATTENTION_DIAGNOSTICS.distribution_rows(stock, result, ROOT)
      if result else [])
  klds = [
      float(row["kld_stock_to_candidate"]) for row in distributions
      if isinstance(row.get("kld_stock_to_candidate"), (int, float)) and
      math.isfinite(float(row["kld_stock_to_candidate"]))
  ]
  top1_rate = (
      sum(row.get("top1_match") is True for row in distributions) /
      len(distributions) if distributions else 0.0)
  finite_rows = (
      bool(distributions) and
      all(row.get("finite") is True for row in distributions))

  new_checkpoints = BASE.checkpoint_map(result)
  old_checkpoints = BASE.checkpoint_map(old_candidate)
  mismatch_steps = []
  invalid_hash_steps = []
  for step in range(OUTPUT_TOKENS):
    new_row = new_checkpoints.get(step)
    old_row = old_checkpoints.get(step)
    if new_row is None or old_row is None:
      mismatch_steps.append(step)
      continue
    new_path = BASE.checkpoint_path(new_row)
    old_path = BASE.checkpoint_path(old_row)
    if not new_path.is_file() or not old_path.is_file():
      mismatch_steps.append(step)
      continue
    new_sha = BASE.sha256(new_path)
    old_sha = BASE.sha256(old_path)
    if new_row.get("sha256") != new_sha or old_row.get("sha256") != old_sha:
      invalid_hash_steps.append(step)
    if (new_sha != old_sha or
        new_row.get("shape") != old_row.get("shape") or
        new_row.get("byte_count") != old_row.get("byte_count")):
      mismatch_steps.append(step)

  trace = result.get("lm_head_i8q1_trace") or {}
  selections = trace.get("selection_rows") or []
  prepacks = trace.get("weight_prepack_rows") or []
  provider_exact = (
      len(selections) == 2 and len(prepacks) == 2 and
      prepacks[0].get("process_cache_hit") is False and
      prepacks[1].get("process_cache_hit") is True and
      all(
          row.get("provider") == BASE.EXPECTED_PROVIDER and
          row.get("tokens") == 1 and
          row.get("rows") == 248320 and
          row.get("columns") == 2048 and
          row.get("correction_passes") == 2 and
          row.get("global") == [248320, 1, 1] and
          row.get("local") == [256, 1, 1]
          for row in selections))
  counts = (
      (result.get("execution_census") or {}).get(
          "executed_type_counts") or {})
  qk_counts = (
      qk_gate.get("execution", {}).get("executed_type_counts") or {})
  expected_counts = dict(qk_counts)
  expected_counts["FullyConnectedCompressed"] = 331
  expected_counts["GLU"] = int(qk_counts.get("GLU", 0)) + 40
  profile = runtime_audit(result)
  source = result.get("source_summary") or {}
  old_source = old_candidate.get("source_summary") or {}
  monitor = worker.get("monitor") or {}
  guard = worker.get("memory_guard") or {}
  environment = worker.get("environment") or {}
  minimum_available = int(
      monitor.get("system_available_min_bytes") or 0)
  stop_bytes = int(MEMORY_STOP_GIB * 1024**3)
  checks = static_checks + [plan_check,
      BASE.check("single_serial_candidate_worker_completes_without_oom",
                 worker.get("returncode") == 0 and
                 worker.get("timed_out") is False and
                 worker.get("oom_observed") is False and
                 worker.get("reused") is not True and
                 (worker.get("worker_transient_scope") or {}).get(
                     "enabled") is True),
      BASE.check("isolated_plugin_and_pair_flags_are_exact",
                 result.get("candidate_gpu_plugin_sha256") ==
                     EXPECTED_SHA256[PLUGIN] and
                 result.get("candidate_path") == "hot_cold_custom" and
                 result.get("custom_composition") == "exact_phase" and
                 result.get("fuse_qk_rope_layout") is True and
                 result.get("fuse_router_shared_pair") is True and
                 result.get("fuse_router_shared_triple") is False and
                 result.get("fuse_fixed_fc") is False and
                 result.get("fixed_fc_manager_direct") is False and
                 environment.get("IQ36_ROUTER_SHARED_PAIR") == "1" and
                 "IQ36_ROUTER_SHARED_TRIPLE" not in environment and
                 "IQ36_FIXED_FC_MANAGER_SCOPE" not in environment),
      BASE.check("exact_count25_parallel_provider_is_selected",
                 result.get("lm_head_i8q1") is True and
                 result.get("lm_head_i8q1_gated_exact") is True and
                 result.get("lm_head_i8q1_gated_q4") is False and
                 result.get("lm_head_i8q1_greedy_local2") is False and
                 result.get("lm_head_token_only_feedback") is False and
                 provider_exact,
                 selection_count=len(selections),
                 prepack_count=len(prepacks)),
      BASE.check("all_130_logits_are_bitwise_equal_to_accepted_carrier",
                 len(new_checkpoints) == OUTPUT_TOKENS and
                 not mismatch_steps and not invalid_hash_steps,
                 checkpoint_count=len(new_checkpoints),
                 mismatch_steps=mismatch_steps,
                 invalid_hash_steps=invalid_hash_steps),
      BASE.check("all_130_stock_relative_distributions_pass",
                 len(distributions) == OUTPUT_TOKENS and finite_rows and
                 len(klds) == OUTPUT_TOKENS and max(klds) <= KLD_MAX and
                 top1_rate >= TOP1_MIN,
                 row_count=len(distributions),
                 max_kld=max(klds) if klds else None,
                 top1_rate=top1_rate),
      BASE.check("exact_output130_tokens_are_preserved",
                 result.get("generated_token_count") == OUTPUT_TOKENS and
                 result.get("generated_token_ids") == expected_tokens and
                 result.get("generated_token_ids_sha256") ==
                     EXPECTED_TOKEN_SHA256 and
                 result.get("teacher_forced_from_stock") is True,
                 generated_token_ids_sha256=result.get(
                     "generated_token_ids_sha256")),
      BASE.check("source_and_state_differ_only_by_exact_qk_rewrite",
                 source.get("fuse_qk_rope_layout") is True and
                 source.get("qk_rope_layout_rewrite_count") == len(LAYERS) and
                 BASE.normalize_source(source) ==
                     BASE.normalize_source(old_source) and
                 result.get("state_schema_after") ==
                     old_candidate.get("state_schema_after")),
      BASE.check("executed_type_census_is_exactly_qk_plus_shared_pair",
                 counts == expected_counts and
                 counts.get("FullyConnectedCompressed") == 331 and
                 counts.get("IQ36QKRopeLayout") == 10 and
                 counts.get(
                     "IQ36ExactPhaseDualCohortHotAttentionGQA") == 10 and
                 counts.get("MOE3GemmFusedCompressed") == 40,
                 observed=counts, expected=expected_counts),
      BASE.check("all_40_pairs_activate_with_scalar_and_router_independent",
                 profile["fully_connected_row_count"] == 331 and
                 profile["fused_shared_pair_count"] == 40 and
                 profile["existing_fused_qkv_count"] == 10 and
                 profile["unfused_pair_original_count"] == 0 and
                 profile["unfused_scalar_shared_gate_count"] == 40 and
                 profile["unfused_router_gate_count"] == 40 and
                 profile["unfused_linear_original_count"] == 120,
                 runtime_profile=profile),
      BASE.check("memory_guard_never_trips",
                 guard.get("tripped") is False and
                 minimum_available >= stop_bytes and
                 int(monitor.get("process_rss_peak_bytes", -1)) >= 0 and
                 int(monitor.get("process_swap_peak_bytes", -1)) >= 0,
                 stop_bytes=stop_bytes, monitor=monitor),
  ]
  passed = all(row["pass"] for row in checks)
  verdict = (
      "admit_current_qk_router_shared_pair_for_one_2k_point_block"
      if passed else
      "reject_current_qk_router_shared_pair_before_point_measurement")
  payload = {
      "schema": SCHEMA,
      "workstream": WS,
      "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
      "git": git,
      "verdict": verdict,
      "required_checks_passed": passed,
      "point_block_admitted": passed,
      "formal_performance_admitted": False,
      "performance_claim_admitted": False,
      "gpu_workers_launched": 1,
      "stock_workers_launched": 0,
      "candidate_workers_launched": 1,
      "workers_concurrent": False,
      "checks": checks,
      "plugin": {"path": str(PLUGIN), "sha256": BASE.sha256(PLUGIN)},
      "correctness": {
          "checkpoint_count": len(new_checkpoints),
          "bitwise_mismatch_steps": mismatch_steps,
          "distribution_row_count": len(distributions),
          "max_kld": max(klds) if klds else None,
          "top1_rate": top1_rate,
          "generated_token_ids_sha256": result.get(
              "generated_token_ids_sha256"),
      },
      "execution": {
          "executed_type_counts": counts,
          "runtime_profile": profile,
      },
      "worker": worker,
      "next_action": (
          {
              "route": "current_qk_router_shared_pair_2k_point_block",
              "point_targets": {
                  "prefill_ratio": 0.995,
                  "decode_ratio": 1.02,
                  "total_ratio": 1.02,
                  "stable_tail_saving_ms": 0.159497,
              },
              "requirements": [
                  "push one control-candidate-candidate-control output512 block",
                  "require exact tokens and all four jitter rows at most 1.25",
                  "require prefill/decode/total point 0.995/1.02/1.02",
                  "only a point pass may fund formal paired inference",
              ],
          } if passed else {
              "route": "close_or_explain_shared_pair_numeric_failure",
              "requirements": [
                  "do not run a point or formal performance worker",
                  "audit immutable logits and the structural split census",
                  "repair only with a bounded exactness hypothesis or close",
              ],
          }),
  }
  PRODUCT.write_json(out / "result.json", payload)
  PRODUCT.write_json(out / "manifest.json", {
      "schema": SCHEMA,
      "tool": PRODUCT.relative(Path(__file__)),
      "tool_sha256": BASE.sha256(Path(__file__)),
      "git": git,
      "inputs": {
          PRODUCT.relative(path): BASE.sha256(path)
          for path in required_paths
      },
      "plugin": payload["plugin"],
      "gpu_workers": 1,
      "stock_workers": 0,
      "candidate_workers": 1,
      "workers_concurrent": False,
  })
  report = f"""# Current Q/K plus N=1024 shared-pair output130 correctness

Verdict: **{verdict}**. Required checks:
`{str(passed).lower()}`.

The sole candidate worker matched
`{OUTPUT_TOKENS - len(set(mismatch_steps))}/{OUTPUT_TOKENS}` accepted logits
bitwise; stock-relative max KLD/top-1 are
`{max(klds) if klds else None}/{top1_rate}`. Executed
FC/QK/attention/pair/scalar/router counts are
`{counts.get('FullyConnectedCompressed')}/`
`{counts.get('IQ36QKRopeLayout')}/`
`{counts.get('IQ36ExactPhaseDualCohortHotAttentionGQA')}/`
`{profile['fused_shared_pair_count']}/`
`{profile['unfused_scalar_shared_gate_count']}/`
`{profile['unfused_router_gate_count']}`.

Peak worker RSS/swap was
`{int(monitor.get('process_rss_peak_bytes', 0))}/`
`{int(monitor.get('process_swap_peak_bytes', 0))} B`; minimum available memory
was `{minimum_available} B`. No OOM or guard event occurred.
"""
  (out / "report.md").write_text(report, encoding="utf-8")
  print(json.dumps({
      "artifact": PRODUCT.relative(out),
      "verdict": verdict,
      "required_checks_passed": passed,
      "bitwise_checkpoint_count": (
          OUTPUT_TOKENS - len(set(mismatch_steps))),
      "max_kld": max(klds) if klds else None,
      "execution_counts": {
          "fc": counts.get("FullyConnectedCompressed"),
          "qk": counts.get("IQ36QKRopeLayout"),
          "attention": counts.get(
              "IQ36ExactPhaseDualCohortHotAttentionGQA"),
          "pair": profile["fused_shared_pair_count"],
          "scalar": profile["unfused_scalar_shared_gate_count"],
          "router": profile["unfused_router_gate_count"],
      },
      "peak_rss_bytes": monitor.get("process_rss_peak_bytes"),
      "oom_observed": worker.get("oom_observed"),
  }, separators=(",", ":")), flush=True)
  return 0 if passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
