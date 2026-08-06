#!/usr/bin/env python3
"""Bound one normalized-F16 two-cohort exact-attention successor.

This source-only gate launches no compiler, OpenCL kernel, plugin, or model
worker.  It binds the exact but sub-threshold triple-cohort component and
admits at most one materially different KQ+softmax producer / VS consumer
compiler gate.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import statistics
import subprocess
from pathlib import Path
from typing import Any

from iq36_perf_inference import (
    bootstrap_median_bound,
    dispersion_diagnostic,
    latency_cap_inference,
)


ROOT = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = (
    "intel-qwen36-openvino-exact-attention-"
    "normalized-dual-cohort-bound-v1")
MODEL_CONFIG = Path("/home/intel/Qwen3.6-35B-A3B-ov/config.json")
TARGET_CONTRACT = ROOT / "contracts/intel-qwen36-target-contract.json"
SOURCE = ROOT / "engine/gpu/opencl/exact_score_staging_component.cl"
SHIMS = ROOT / "engine/openvino/custom/iq36_decode_microkernel_shims.cl"
STATUS = ROOT / "doc/active" / WS / "STATUS.md"
ROUTES = ROOT / "doc/active" / WS / "routes-ledger.json"
DECOMPOSITION = ROOT / (
    "output/openvino-exact-attention-three-stage-component-"
    "20260724Tseq2144-clean/result.json")
CODEGEN_AUDIT = ROOT / (
    "output/openvino-exact-attention-triple-cohort-codegen-audit-"
    "20260724Tseq2145a-clean/result.json")
TRIPLE_COMPONENT = ROOT / (
    "output/openvino-exact-attention-triple-cohort-component-"
    "20260724Tseq2146-clean/result.json")

CONTEXT_TOKENS = 131_072
KEY_BLOCK = 256
KV_HEADS = 2
SCORE_COLUMNS = 16
SUBGROUP_SIZE = 16
PRODUCER_SUBGROUPS = 16
CONSUMER_SUBGROUPS = 16
TOTAL_SUBGROUPS = PRODUCER_SUBGROUPS + CONSUMER_SUBGROUPS
WORKGROUP_ITEMS = TOTAL_SUBGROUPS * SUBGROUP_SIZE
DELTA_CAP_MS = -0.1175998
MAX_DEFICIT_MS = 0.03
MAX_DEFICIT_FRACTION = 0.01
STAGE_BALANCE_CAP = 1.25
MIN_SAMPLES = 20

SLM_BUDGET = {
    "query_f16": 256 * SCORE_COLUMNS * 2,
    "normalized_score_f16_double_buffer": 2 * KEY_BLOCK * SCORE_COLUMNS * 2,
    "softmax_max_and_final_sum": 2 * KEY_BLOCK * 4,
    "accumulator_rescale_double_buffer": 2 * KEY_BLOCK * 4,
    "ugemm_scratch": 1,
    "output_incremental_after_normalized_reuse": 0,
}
SLM_UNPADDED_BYTES = sum(SLM_BUDGET.values())
SLM_PADDED_CEILING_BYTES = 28_704
DEVICE_LOCAL_MEMORY_BYTES = 128 * 1024

BLOCK_COUNT = CONTEXT_TOKENS // KEY_BLOCK
RAW_F32_SLM_ROUNDTRIP_BYTES = (
    KV_HEADS * BLOCK_COUNT * 2 * KEY_BLOCK * SCORE_COLUMNS * 4)
NORMALIZED_F16_SLM_ROUNDTRIP_BYTES = (
    KV_HEADS * BLOCK_COUNT * 2 * KEY_BLOCK * SCORE_COLUMNS * 2)
REMOVED_PIPELINE_RENDEZVOUS = KV_HEADS * BLOCK_COUNT


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out-dir", type=Path, required=True)
  parser.add_argument("--memory-stop-gib", type=float, default=4.0)
  args = parser.parse_args()
  if args.memory_stop_gib < 4.0:
    parser.error("--memory-stop-gib must be at least 4")
  return args


def load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise TypeError(f"expected JSON object: {path}")
  return value


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True) + "\n",
      encoding="utf-8")


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    while chunk := stream.read(1024 * 1024):
      digest.update(chunk)
  return digest.hexdigest()


def display(path: Path) -> str:
  try:
    return str(path.relative_to(ROOT))
  except ValueError:
    return str(path)


def available_memory_bytes() -> int:
  for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
    if line.startswith("MemAvailable:"):
      return int(line.split()[1]) * 1024
  raise RuntimeError("MemAvailable is missing")


def git_state(out_dir: Path) -> dict[str, Any]:
  commit = subprocess.run(
      ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
      capture_output=True, text=True).stdout.strip()
  dirty = subprocess.run(
      ["git", "status", "--porcelain"], cwd=ROOT, check=True,
      capture_output=True, text=True).stdout.splitlines()
  try:
    out_rel = str(out_dir.relative_to(ROOT))
  except ValueError:
    out_rel = ""
  dirty = [row for row in dirty if not out_rel or out_rel not in row]
  return {"commit": commit, "dirty": bool(dirty), "dirty_paths": dirty}


def check(name: str, passed: bool, **details: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **details}


def lower_floor_inference(
    values: list[float], floor: float, seed: int,
) -> dict[str, Any]:
  if len(values) != MIN_SAMPLES or any(
      not math.isfinite(value) or value <= 0.0 for value in values):
    return {
        "error": "samples must be 20 positive finite values",
        "rate_pass": False,
        "floor_ms": floor,
    }
  lower = bootstrap_median_bound(values, side="lower", seed=seed)
  return {
      "method": "one_sided_percentile_bootstrap_median",
      "confidence": 0.95,
      "bootstrap_resamples": 20_000,
      "bootstrap_seed": seed,
      "sample_count": len(values),
      "minimum_sample_count": MIN_SAMPLES,
      "point_estimate_ms": statistics.median(values),
      "lower_confidence_bound_ms": lower,
      "floor_ms": floor,
      "sample_count_pass": True,
      "rate_pass": lower >= floor,
      "dispersion": dispersion_diagnostic(values),
  }


def summary(payload: dict[str, Any]) -> str:
  inference = payload["stage_inference"]
  recovery = payload["recovery_contract"]
  resources = payload["projected_resource_contract"]
  return "\n".join([
      "# Normalized-F16 two-cohort attention bound",
      "",
      f"Verdict: **{payload['verdict']}**. Required checks: "
      f"`{str(payload['required_checks_passed']).lower()}`.",
      "",
      f"- triple UCB deficit / fraction: "
      f"`{recovery['triple_ucb_deficit_ms']} ms/layer / "
      f"{recovery['deficit_fraction_of_triple_median']}`",
      f"- producer/consumer balance median / 95% UCB / cap: "
      f"`{inference['stage_balance'].get('point_estimate_ms')} / "
      f"{inference['stage_balance'].get('upper_confidence_bound_ms')} / "
      f"{STAGE_BALANCE_CAP}`",
      f"- conservative ideal saving proxy median / 95% LCB: "
      f"`{inference['ideal_saving'].get('point_estimate_ms')} / "
      f"{inference['ideal_saving'].get('lower_confidence_bound_ms')} "
      "ms/layer`",
      f"- removed raw-F32 SLM traffic / rendezvous: "
      f"`{recovery['removed_raw_f32_slm_roundtrip_bytes']} B / "
      f"{recovery['removed_pipeline_rendezvous_per_layer']}`",
      f"- implied recovery bandwidth: "
      f"`{recovery['implied_removed_traffic_gb_s']} GB/s`",
      f"- projected padded SLM / two-WG use: "
      f"`{resources['slm_padded_ceiling_bytes']} / "
      f"{resources['two_workgroup_slm_bytes']} B`",
      "",
      "A pass admits one compiler/resource gate only.  No compiler, kernel,",
      "plugin, or model worker was launched by this bound.",
      "",
  ])


def main() -> int:
  args = parse_args()
  out_dir = args.out_dir.resolve()
  out_dir.mkdir(parents=True, exist_ok=False)
  stop_bytes = int(args.memory_stop_gib * 1024**3)
  memory_start = available_memory_bytes()
  if memory_start < stop_bytes:
    raise RuntimeError(
        f"memory stop at start: {memory_start} < {stop_bytes}")

  required_paths = (
      MODEL_CONFIG, TARGET_CONTRACT, SOURCE, SHIMS, STATUS, ROUTES,
      DECOMPOSITION, CODEGEN_AUDIT, TRIPLE_COMPONENT)
  missing = [str(path) for path in required_paths if not path.is_file()]
  if missing:
    raise SystemExit(
        "missing normalized-cohort bound inputs: " + ", ".join(missing))

  git = git_state(out_dir)
  model = load_json(MODEL_CONFIG).get("text_config", {})
  target = load_json(TARGET_CONTRACT)
  decomposition = load_json(DECOMPOSITION)
  audit = load_json(CODEGEN_AUDIT)
  triple = load_json(TRIPLE_COMPONENT)
  source = SOURCE.read_text(encoding="utf-8")
  shims = SHIMS.read_text(encoding="utf-8")
  status = STATUS.read_text(encoding="utf-8")
  routes = ROUTES.read_text(encoding="utf-8")

  layers = int(model.get("num_hidden_layers", -1))
  interval = int(model.get("full_attention_interval", -1))
  query_heads = int(model.get("num_attention_heads", -1))
  kv_heads = int(model.get("num_key_value_heads", -1))
  head_dim = int(model.get("head_dim", -1))
  model_pass = bool(
      layers == 40 and interval == 4
      and layers // interval == 10
      and query_heads == 16 and kv_heads == KV_HEADS
      and head_dim == 256 and query_heads // kv_heads == 8)

  decomposition_rows = decomposition.get(
      "result", {}).get("paired_samples", [])
  producer_proxy: list[float] = []
  consumer_proxy: list[float] = []
  balance_ratios: list[float] = []
  ideal_savings: list[float] = []
  for row in decomposition_rows:
    staged = row.get("three_stage", {}) if isinstance(row, dict) else {}
    kq = float(staged.get("kq_ms", math.nan))
    softmax_arithmetic = float(
        staged.get("softmax_arithmetic_ms", math.nan))
    consumer = float(
        staged.get("owner_residual_proxy_ms", math.nan))
    dual = float(row.get("dual_ms", math.nan))
    producer = kq + softmax_arithmetic
    producer_proxy.append(producer)
    consumer_proxy.append(consumer)
    balance_ratios.append(
        max(producer, consumer) / min(producer, consumer)
        if min(producer, consumer) > 0.0 else math.nan)
    ideal_savings.append(dual - max(producer, consumer))

  stage_inference = {
      "producer_latency":
          lower_floor_inference(producer_proxy, 0.0, 214701),
      "consumer_latency":
          lower_floor_inference(consumer_proxy, 0.0, 214702),
      "stage_balance":
          latency_cap_inference(
              balance_ratios, cap=STAGE_BALANCE_CAP,
              min_samples=MIN_SAMPLES, seed=214703)
          if len(balance_ratios) == MIN_SAMPLES
          and all(math.isfinite(value) and value > 0.0
                  for value in balance_ratios) else
          {"error": "invalid stage balance samples", "rate_pass": False},
      "ideal_saving":
          lower_floor_inference(
              ideal_savings, abs(DELTA_CAP_MS), 214704),
  }

  triple_inference = triple.get("performance_inference", {})
  triple_delta_point = float(
      triple_inference.get("point_estimate_ms", math.nan))
  triple_delta_ucb = float(
      triple_inference.get("upper_confidence_bound_ms", math.nan))
  triple_ucb_deficit = triple_delta_ucb - DELTA_CAP_MS
  triple_rows = triple.get("result", {}).get("paired_samples", [])
  triple_median = statistics.median(
      float(row["triple_ms"]) for row in triple_rows)
  deficit_fraction = triple_ucb_deficit / triple_median
  implied_bandwidth_gb_s = (
      RAW_F32_SLM_ROUNDTRIP_BYTES /
      (triple_ucb_deficit * 1.0e-3) / 1.0e9
      if triple_ucb_deficit > 0.0 else math.inf)
  recovery_contract = {
      "triple_delta_point_ms": triple_delta_point,
      "triple_delta_ucb_ms": triple_delta_ucb,
      "required_delta_cap_ms": DELTA_CAP_MS,
      "triple_ucb_deficit_ms": triple_ucb_deficit,
      "triple_median_ms": triple_median,
      "deficit_fraction_of_triple_median": deficit_fraction,
      "maximum_deficit_ms": MAX_DEFICIT_MS,
      "maximum_deficit_fraction": MAX_DEFICIT_FRACTION,
      "removed_raw_f32_slm_roundtrip_bytes":
          RAW_F32_SLM_ROUNDTRIP_BYTES,
      "retained_normalized_f16_slm_roundtrip_bytes":
          NORMALIZED_F16_SLM_ROUNDTRIP_BYTES,
      "removed_pipeline_rendezvous_per_layer":
          REMOVED_PIPELINE_RENDEZVOUS,
      "removed_middle_subgroups": 16,
      "implied_removed_traffic_gb_s": implied_bandwidth_gb_s,
  }
  projected_resources = {
      "producer_subgroups": PRODUCER_SUBGROUPS,
      "consumer_subgroups": CONSUMER_SUBGROUPS,
      "total_subgroups": TOTAL_SUBGROUPS,
      "workgroup_items": WORKGROUP_ITEMS,
      "slm_budget": SLM_BUDGET,
      "slm_unpadded_bytes": SLM_UNPADDED_BYTES,
      "slm_padded_ceiling_bytes": SLM_PADDED_CEILING_BYTES,
      "two_workgroup_slm_bytes": 2 * SLM_PADDED_CEILING_BYTES,
      "two_workgroup_margin_bytes":
          DEVICE_LOCAL_MEMORY_BYTES - 2 * SLM_PADDED_CEILING_BYTES,
      "device_local_memory_bytes": DEVICE_LOCAL_MEMORY_BYTES,
      "output_storage_contract":
          "reuse normalized-score slab only after final pipeline barrier",
  }

  decomposition_pass = bool(
      decomposition.get("required_checks_passed") is True
      and decomposition.get("result", {}).get("numeric_pass") is True
      and len(decomposition_rows) == MIN_SAMPLES)
  audit_pass = bool(
      audit.get("required_checks_passed") is True
      and audit.get("component_admitted") is True
      and audit.get("candidate_result", {}).get(
          "kernel_spill_memory_bytes") == 0
      and audit.get("candidate_result", {}).get(
          "kernel_local_memory_bytes") == 61_472)
  triple_failed_checks = [
      row.get("name") for row in triple.get("checks", [])
      if isinstance(row, dict) and row.get("pass") is False]
  triple_pass = bool(
      triple.get("verdict") ==
          "reject_exact_attention_triple_cohort_component"
      and triple.get("required_checks_passed") is False
      and triple.get("result", {}).get("numeric_pass") is True
      and triple.get("result", {}).get("output_mismatch_count") == 0
      and triple_failed_checks == [
          "one_sided_95pct_delta_ucb_clears_layer_kill_number"])
  source_checks = {
      "exact_generated_carriers_exist":
          source.count("iq36_component_score_tile first_score = ugemm_kq(")
              >= 2
          and source.count(
              "iq36_component_accumulator_tile chunk_accumulator = "
              "ugemm_vs(") >= 4,
      "chronological_softmax_recurrence_exists":
          "tile_vreduce_max(score, &running_max);" in source
          and "tile_vbroadcast_sub(&score, running_max);" in source
          and "tile_elementwise(score, iq36_component_scaled_exp);"
              in source
          and "block_rescale, running_max, iq36_component_rescale"
              in source
          and "tile_copy(running_max, old_running_max);" in source,
      "generated_packages_are_m256_n16_barrier_free":
          "#define ugemm_kq_wg_tile_m 256" in shims
          and "#define ugemm_kq_wg_tile_n 16" in shims
          and "#define ugemm_vs_wg_tile_m 256" in shims
          and "#define ugemm_vs_wg_tile_n 16" in shims
          and "#define ugemm_kq_barrier_count  0" in shims
          and "#define ugemm_vs_barrier_count  0" in shims,
      "route_and_exact_deficit_are_registered":
          "0.0250998-ms/layer" in status
          and "close_triple_cohort_and_bound_one_normalized_f16_dual_cohort"
              in routes,
  }
  resource_fit = bool(
      SLM_UNPADDED_BYTES == 28_673
      and SLM_UNPADDED_BYTES <= SLM_PADDED_CEILING_BYTES
      and 2 * SLM_PADDED_CEILING_BYTES <=
          DEVICE_LOCAL_MEMORY_BYTES
      and WORKGROUP_ITEMS == 512
      and TOTAL_SUBGROUPS == 32)
  recovery_is_bounded = bool(
      math.isfinite(triple_ucb_deficit)
      and 0.0 < triple_ucb_deficit <= MAX_DEFICIT_MS
      and 0.0 < deficit_fraction <= MAX_DEFICIT_FRACTION
      and RAW_F32_SLM_ROUNDTRIP_BYTES == 33_554_432
      and REMOVED_PIPELINE_RENDEZVOUS == 1024)

  checks = [
      check("repository_clean_at_bound", not git["dirty"], git=git),
      check("locked_model_geometry_is_exact", model_pass),
      check("seq2144_exact_decomposition_is_bound", decomposition_pass),
      check("seq2145_actual_resources_are_bound", audit_pass),
      check("seq2146_is_exact_and_only_performance_failed",
            triple_pass),
      check("triple_component_confidently_improves_but_misses_cap",
            math.isfinite(triple_delta_ucb)
            and triple_delta_ucb < 0.0
            and triple_delta_ucb > DELTA_CAP_MS,
            triple_inference=triple_inference),
      check("required_recovery_is_below_one_percent_and_30us",
            recovery_is_bounded,
            recovery_contract=recovery_contract),
      check("producer_and_consumer_proxy_are_balanced",
            stage_inference["stage_balance"].get("rate_pass") is True,
            inference=stage_inference["stage_balance"]),
      check("conservative_ideal_saving_lcb_clears_kill_number",
            stage_inference["ideal_saving"].get("rate_pass") is True,
            inference=stage_inference["ideal_saving"]),
      check("fixed_32_subgroup_normalized_slm_design_fits",
            resource_fit, projected_resources=projected_resources),
      check("source_retains_the_exact_generated_chronological_carrier",
            all(source_checks.values()), source_checks=source_checks),
      check("target_contract_remains_local_openvino_gpu",
            target.get("execution", {}).get("mode") == "local"
            and target.get("execution", {}).get(
                "network_transport_required") is False
            and target.get("runtime", {}).get("opencl_device") ==
                "Intel(R) Arc(TM) B390 GPU"),
      check("no_compiler_kernel_plugin_or_model_worker_launched", True),
  ]
  memory_end = available_memory_bytes()
  checks.append(check(
      "memory_guard_never_tripped",
      min(memory_start, memory_end) >= stop_bytes,
      minimum_available_bytes=min(memory_start, memory_end),
      memory_stop_bytes=stop_bytes))
  required = all(row["pass"] for row in checks)
  verdict = (
      "admit_one_exact_attention_normalized_dual_cohort_codegen_gate"
      if required else
      "close_exact_attention_normalized_dual_cohort_at_bound")
  sources = [
      {"path": display(path), "sha256": sha256(path)}
      for path in required_paths
  ]
  payload = {
      "schema_version": SCHEMA,
      "workstream": WS,
      "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
      "git": git,
      "verdict": verdict,
      "required_checks_passed": required,
      "compiler_resource_gate_admitted": required,
      "component_admitted": False,
      "kernel_enqueue_admitted": False,
      "graph_integration_admitted": False,
      "plugin_build_admitted": False,
      "model_worker_admitted": False,
      "product_claim_allowed": False,
      "checks": checks,
      "stage_samples": {
          "producer_proxy_ms": producer_proxy,
          "consumer_proxy_ms": consumer_proxy,
          "balance_ratios": balance_ratios,
          "ideal_savings_ms": ideal_savings,
      },
      "stage_inference": stage_inference,
      "recovery_contract": recovery_contract,
      "projected_resource_contract": projected_resources,
      "memory_stop_bytes": stop_bytes,
      "memory_samples": [
          {"label": "start", "available_bytes": memory_start},
          {"label": "end", "available_bytes": memory_end},
      ],
      "sources": sources,
      "compiler_workers_launched": False,
      "kernel_worker_launched": False,
      "model_worker_launched": False,
  }
  write_json(out_dir / "result.json", payload)
  write_json(out_dir / "manifest.json", {
      "schema_version": "intel-qwen36-artifact-manifest-v1",
      "workstream": WS,
      "git_commit": git["commit"],
      "verdict": verdict,
      "sources": sources,
      "files": ["result.json", "summary.md"],
  })
  (out_dir / "summary.md").write_text(
      summary(payload), encoding="utf-8")
  print(json.dumps({
      "artifact": display(out_dir),
      "verdict": verdict,
      "triple_ucb_deficit_ms": triple_ucb_deficit,
      "deficit_fraction": deficit_fraction,
      "stage_balance_ucb":
          stage_inference["stage_balance"].get(
              "upper_confidence_bound_ms"),
      "ideal_saving_lcb_ms":
          stage_inference["ideal_saving"].get(
              "lower_confidence_bound_ms"),
      "slm_padded_ceiling_bytes": SLM_PADDED_CEILING_BYTES,
      "compiler_resource_gate_admitted": required,
      "kernel_worker_launched": False,
      "model_worker_launched": False,
  }, separators=(",", ":")))
  return 0 if required else 2


if __name__ == "__main__":
  raise SystemExit(main())
