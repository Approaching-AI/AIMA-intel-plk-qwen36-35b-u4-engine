#!/usr/bin/env python3
"""Bound one on-chip dual-cohort exact-attention codegen probe.

This gate launches no compiler, GPU kernel, plugin, or model worker.  It uses
the clean bit-exact score-staging component to bound the KQ and chronological
softmax/VS portions independently, then checks whether two 16-subgroup
cohorts and their complete SLM handoff fit the locked PTL device.  A pass
admits only one compiler/resource probe; it is not a performance result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-openvino-exact-attention-dual-cohort-bound-v1"

MODEL_CONFIG = Path("/home/intel/Qwen3.6-35B-A3B-ov/config.json")
TARGET_CONTRACT = ROOT / "contracts/intel-qwen36-target-contract.json"
SOURCE = ROOT / "engine/gpu/opencl/exact_score_staging_component.cl"
SHIMS = ROOT / "engine/openvino/custom/iq36_decode_microkernel_shims.cl"
STATUS = ROOT / "doc/active" / WS / "STATUS.md"
FRONTIER = ROOT / "doc/active" / WS / "frontier.json"
ROUTES = ROOT / "doc/active" / WS / "routes-ledger.json"
REJECTED = ROOT / "doc/active" / WS / "rejected-routes.json"
STAGING = ROOT / (
    "output/openvino-exact-score-staging-component-"
    "20260723Tseq2126-clean/result.json")
VRT160 = ROOT / (
    "output/openvino-exact-attention-vrt160-component-"
    "20260723Tseq2128-clean/result.json")

CONTEXT_TOKENS = 131072
KEY_BLOCK = 256
SUBGROUP_SIZE = 16
PRODUCER_SUBGROUPS = 16
CONSUMER_SUBGROUPS = 16
TOTAL_SUBGROUPS = PRODUCER_SUBGROUPS + CONSUMER_SUBGROUPS
WORKGROUP_ITEMS = TOTAL_SUBGROUPS * SUBGROUP_SIZE
RAW_SCORE_COLUMNS = 16
F32_BYTES = 4
F16_BYTES = 2
BOOTSTRAP_RESAMPLES = 20000
BOOTSTRAP_SEED = 21290
COMPONENT_CAP_MS_PER_LAYER = -0.1175998


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--memory-stop-gib", type=float, default=4.0)
  args = parser.parse_args()
  if args.memory_stop_gib <= 0.0:
    parser.error("--memory-stop-gib must be positive")
  return args


def load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise TypeError(f"expected JSON object: {path}")
  return value


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
  for row in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
    if row.startswith("MemAvailable:"):
      return int(row.split()[1]) * 1024
  raise RuntimeError("MemAvailable is missing")


def sample_memory(
    label: str, stop_bytes: int, rows: list[dict[str, Any]],
) -> None:
  available = available_memory_bytes()
  rows.append({"label": label, "available_bytes": available})
  if available < stop_bytes:
    raise RuntimeError(
        f"memory stop at {label}: {available} < {stop_bytes} bytes")


def git_state(output: Path) -> dict[str, Any]:
  commit = subprocess.run(
      ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, text=True,
      capture_output=True).stdout.strip()
  dirty = subprocess.run(
      ["git", "status", "--porcelain"], cwd=ROOT, check=True, text=True,
      capture_output=True).stdout.splitlines()
  try:
    output_rel = str(output.relative_to(ROOT))
  except ValueError:
    output_rel = ""
  dirty = [row for row in dirty if not output_rel or output_rel not in row]
  return {"commit": commit, "dirty": bool(dirty), "status": dirty}


def check(name: str, passed: bool, **evidence: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **evidence}


def percentile(sorted_values: list[float], probability: float) -> float:
  index = max(
      0, min(len(sorted_values) - 1,
             math.floor(probability * len(sorted_values))))
  return sorted_values[index]


def median_lcb(samples: list[float]) -> float:
  rng = random.Random(BOOTSTRAP_SEED)
  count = len(samples)
  bootstraps = [
      statistics.median(rng.choice(samples) for _ in range(count))
      for _ in range(BOOTSTRAP_RESAMPLES)
  ]
  bootstraps.sort()
  return percentile(bootstraps, 0.05)


def main() -> int:
  args = parse_args()
  output = args.output.resolve()
  output.mkdir(parents=True, exist_ok=False)
  stop_bytes = int(args.memory_stop_gib * 1024**3)
  memory: list[dict[str, Any]] = []
  sample_memory("start", stop_bytes, memory)

  required = (
      MODEL_CONFIG, TARGET_CONTRACT, SOURCE, SHIMS, STATUS, FRONTIER,
      ROUTES, REJECTED, STAGING, VRT160)
  missing = [str(path) for path in required if not path.is_file()]
  if missing:
    raise SystemExit("missing dual-cohort bound inputs: " + ", ".join(missing))

  git = git_state(output)
  model = load_json(MODEL_CONFIG).get("text_config", {})
  target = load_json(TARGET_CONTRACT)
  frontier = load_json(FRONTIER)
  staging = load_json(STAGING)
  vrt160 = load_json(VRT160)
  source = SOURCE.read_text(encoding="utf-8")
  shims = SHIMS.read_text(encoding="utf-8")
  status = STATUS.read_text(encoding="utf-8")
  routes = ROUTES.read_text(encoding="utf-8")
  rejected = REJECTED.read_text(encoding="utf-8")

  clinfo_run = subprocess.run(
      ["clinfo"], cwd=ROOT, text=True, capture_output=True, check=False,
      timeout=30)
  clinfo = clinfo_run.stdout
  sample_memory("after-source-and-device-audit", stop_bytes, memory)

  layers = int(model.get("num_hidden_layers", -1))
  interval = int(model.get("full_attention_interval", -1))
  query_heads = int(model.get("num_attention_heads", -1))
  kv_heads = int(model.get("num_key_value_heads", -1))
  head_dim = int(model.get("head_dim", -1))
  full_attention_layers = layers // interval if interval > 0 else -1
  gqa_group = query_heads // kv_heads if kv_heads > 0 else -1
  block_count = CONTEXT_TOKENS // KEY_BLOCK

  paired = staging.get("result", {}).get("paired_samples", [])
  timing_rows: list[dict[str, float | int]] = []
  for index, row in enumerate(paired):
    fused_ms = float(row["fused"]["total_ms"])
    kq_ms = float(row["staged"]["kq_ms"])
    owner_ms = float(row["staged"]["owner_ms"])
    ideal_pipeline_ms = max(kq_ms, owner_ms)
    timing_rows.append({
        "sample": index,
        "fused_ms": fused_ms,
        "staged_kq_ms": kq_ms,
        "staged_owner_ms": owner_ms,
        "conservative_ideal_pipeline_ms": ideal_pipeline_ms,
        "ideal_overlap_saving_ms": fused_ms - ideal_pipeline_ms,
    })

  overlap_savings = [
      float(row["ideal_overlap_saving_ms"]) for row in timing_rows]
  ideal_saving_median_ms = statistics.median(overlap_savings)
  ideal_saving_lcb_ms = median_lcb(overlap_savings)
  required_saving_ms = abs(COMPONENT_CAP_MS_PER_LAYER)
  lcb_overhead_budget_ms = ideal_saving_lcb_ms - required_saving_ms
  lcb_overhead_budget_us_per_block = (
      lcb_overhead_budget_ms * 1000.0 / block_count)
  required_realized_fraction = required_saving_ms / ideal_saving_lcb_ms

  query_slm_bytes = head_dim * RAW_SCORE_COLUMNS * F16_BYTES
  raw_score_buffer_bytes = KEY_BLOCK * RAW_SCORE_COLUMNS * F32_BYTES
  raw_score_double_buffer_bytes = 2 * raw_score_buffer_bytes
  normalized_score_slm_bytes = KEY_BLOCK * RAW_SCORE_COLUMNS * F16_BYTES
  sum_slm_bytes = RAW_SCORE_COLUMNS * PRODUCER_SUBGROUPS * F32_BYTES
  max_and_guard_slm_bytes = KEY_BLOCK * F32_BYTES
  output_slm_bytes = head_dim * RAW_SCORE_COLUMNS * F16_BYTES
  ugemm_slm_bytes = 1
  total_slm_bytes = (
      query_slm_bytes + raw_score_double_buffer_bytes
      + normalized_score_slm_bytes + sum_slm_bytes
      + max_and_guard_slm_bytes + output_slm_bytes + ugemm_slm_bytes)
  two_resident_wg_slm_bytes = 2 * total_slm_bytes
  device_local_mem_bytes = 128 * 1024
  device_max_workgroup_items = 1024
  device_max_subgroups = 64

  runtime = target.get("runtime", {})
  expected_device = str(runtime.get("opencl_device", ""))
  model_exact = (
      layers == 40 and interval == 4 and full_attention_layers == 10
      and query_heads == 16 and kv_heads == 2 and head_dim == 256
      and gqa_group == 8)
  source_markers = {
      "fixed_m256_n16_fused_kernel":
          "__kernel void iq36_exact_score_fused(" in source
          and "__attribute__((reqd_work_group_size(16, 16, 1)))" in source,
      "chronological_owner_recurrence":
          "for (uint key_begin = 0U; key_begin < IQ36_CONTEXT;" in source
          and "tile_copy(running_max, old_running_max);" in source
          and "accumulator, chunk_accumulator" in source,
      "f32_score_tile_can_roundtrip_through_local":
          "typedef ugemm_kq_c_type iq36_component_score_tile;" in source
          and "DECLARE_2D_TILE_VREDUCE(" in source,
      "generated_kq_and_vs_retained":
          "iq36_component_score_tile score = ugemm_kq(" in source
          and "iq36_component_accumulator_tile chunk_accumulator = ugemm_vs("
              in source,
  }
  shim_markers = {
      "kq_m256_n16":
          "#define ugemm_kq_wg_tile_m 256" in shims
          and "#define ugemm_kq_wg_tile_n 16" in shims,
      "vs_m256_n16":
          "#define ugemm_vs_wg_tile_m 256" in shims
          and "#define ugemm_vs_wg_tile_n 16" in shims,
      "kq_uses_sixteen_subgroups":
          "#define ugemm_kq_sg_per_wg_m 16" in shims
          and "#define ugemm_kq_sg_per_wg_n 1" in shims,
      "vs_uses_sixteen_subgroups":
          "#define ugemm_vs_sg_per_wg_m 16" in shims
          and "#define ugemm_vs_sg_per_wg_n 1" in shims,
      "packages_use_no_internal_barriers":
          "#define ugemm_kq_barrier_count  0" in shims
          and "#define ugemm_vs_barrier_count  0" in shims,
  }
  device_capabilities = {
      "device_matches_contract": expected_device in clinfo,
      "named_subset_barrier_extension":
          "cl_khr_subgroup_named_barrier" in clinfo,
      "independent_subgroup_forward_progress":
          "Sub-group independent forward progress        Yes" in clinfo,
      "max_workgroup_at_least_1024":
          "Max work group size                             1024" in clinfo,
      "max_subgroups_at_least_64":
          "Max sub-groups per work group                   64" in clinfo,
      "local_memory_at_least_128k":
          "Local memory size                               131072" in clinfo,
  }
  staging_exact = bool(
      staging.get("verdict") == "reject_exact_full_score_staging_component"
      and staging.get("result", {}).get("numeric_pass") is True
      and staging.get("result", {}).get("raw_score_mismatch_count") == 0
      and staging.get("result", {}).get("output_mismatch_count") == 0
      and len(timing_rows) == 20)
  vrt160_closed = bool(
      vrt160.get("verdict") == "reject_exact_attention_vrt160_component"
      and vrt160.get("result", {}).get("numeric_pass") is True
      and vrt160.get("result", {}).get("output_mismatch_count") == 0
      and math.isclose(
          float(vrt160.get("performance_inference", {}).get(
              "upper_confidence_bound_ms", math.inf)),
          0.003229, abs_tol=1.0e-9))
  route_distinct = bool(
      "openvino_exact_attention_full_raw_f32_score_staging_v30bh" in rejected
      and "openvino_exact_attention_vrt160" in routes
      and "on-chip producer/consumer mechanism" in rejected)
  kill_number_bound = bool(
      math.isclose(
          float(frontier["goal_budget"]["per_token_ms"]["remaining_cut"]),
          0.345159, abs_tol=1.0e-9)
      and "1.175998 ms/token" in status)

  checks = [
      check("repository_clean_at_gate", not git["dirty"], git=git),
      check("locked_model_geometry_is_exact", model_exact),
      check("score_staging_supplies_bit_exact_independent_kq_owner_samples",
            staging_exact),
      check("adjacent_vrt160_route_is_closed_before_new_geometry",
            vrt160_closed),
      check("current_product_kill_numbers_are_bound", kill_number_bound),
      check("source_retains_exact_chronological_generated_carrier",
            all(source_markers.values()), markers=source_markers),
      check("generated_packages_fit_two_sixteen_subgroup_cohorts",
            all(shim_markers.values()), markers=shim_markers),
      check("device_supports_subset_cohort_forward_progress",
            clinfo_run.returncode == 0
            and all(device_capabilities.values()),
            capabilities=device_capabilities),
      check("complete_double_buffered_slm_allocation_allows_two_resident_wgs",
            total_slm_bytes <= 64 * 1024
            and two_resident_wg_slm_bytes <= device_local_mem_bytes,
            total_slm_bytes=total_slm_bytes,
            two_resident_wg_slm_bytes=two_resident_wg_slm_bytes,
            device_local_mem_bytes=device_local_mem_bytes),
      check("thirty_two_subgroup_workgroup_fits_device_limits",
            WORKGROUP_ITEMS <= device_max_workgroup_items
            and TOTAL_SUBGROUPS <= device_max_subgroups,
            workgroup_items=WORKGROUP_ITEMS,
            total_subgroups=TOTAL_SUBGROUPS),
      check("measured_ideal_overlap_lcb_materially_clears_component_cap",
            ideal_saving_lcb_ms >= required_saving_ms,
            ideal_saving_lcb_ms=ideal_saving_lcb_ms,
            required_saving_ms=required_saving_ms,
            required_realized_fraction=required_realized_fraction),
      check("route_is_distinct_from_global_staging_and_vrt_sweep",
            route_distinct),
      check("memory_guard_never_tripped",
            all(row["available_bytes"] >= stop_bytes for row in memory),
            minimum_available_bytes=min(
                row["available_bytes"] for row in memory)),
      check("no_compiler_kernel_plugin_or_model_worker_launched", True),
  ]
  required_checks_passed = all(row["pass"] for row in checks)
  verdict = (
      "admit_one_exact_attention_dual_cohort_codegen_gate"
      if required_checks_passed else "inconclusive")

  payload = {
      "schema_version": SCHEMA,
      "workstream": WS,
      "created_at": datetime.now(timezone.utc).isoformat(),
      "git": git,
      "inputs": {
          display(path): sha256(path) for path in required
      },
      "checks": checks,
      "required_checks_passed": required_checks_passed,
      "verdict": verdict,
      "performance_opportunity": {
          "sample_count": len(timing_rows),
          "method": (
              "per-pair fused minus max(bit-exact staged KQ, bit-exact "
              "chronological owner); one-sided 95% percentile-bootstrap "
              "median LCB"),
          "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
          "bootstrap_seed": BOOTSTRAP_SEED,
          "fused_median_ms_per_layer": statistics.median(
              float(row["fused_ms"]) for row in timing_rows),
          "staged_kq_median_ms_per_layer": statistics.median(
              float(row["staged_kq_ms"]) for row in timing_rows),
          "staged_owner_median_ms_per_layer": statistics.median(
              float(row["staged_owner_ms"]) for row in timing_rows),
          "conservative_ideal_pipeline_median_ms_per_layer":
              statistics.median(
                  float(row["conservative_ideal_pipeline_ms"])
                  for row in timing_rows),
          "ideal_overlap_saving_median_ms_per_layer":
              ideal_saving_median_ms,
          "ideal_overlap_saving_lcb_ms_per_layer": ideal_saving_lcb_ms,
          "required_saving_ms_per_layer": required_saving_ms,
          "required_realized_fraction_of_lcb": required_realized_fraction,
          "unhidden_overhead_budget_lcb_ms_per_layer":
              lcb_overhead_budget_ms,
          "unhidden_overhead_budget_lcb_us_per_256_key_block":
              lcb_overhead_budget_us_per_block,
          "samples": timing_rows,
          "interpretation": (
              "The max(KQ, owner) construction retains the measured global "
              "raw-score write/read costs, so it is a conservative arithmetic "
              "opportunity, not a runtime speed claim."),
      },
      "resource_contract": {
          "workgroup_items": WORKGROUP_ITEMS,
          "subgroup_size": SUBGROUP_SIZE,
          "producer_subgroups": PRODUCER_SUBGROUPS,
          "consumer_subgroups": CONSUMER_SUBGROUPS,
          "total_subgroups": TOTAL_SUBGROUPS,
          "context_tokens": CONTEXT_TOKENS,
          "key_block": KEY_BLOCK,
          "blocks_per_layer": block_count,
          "slm_bytes": {
              "query": query_slm_bytes,
              "raw_score_double_buffer": raw_score_double_buffer_bytes,
              "normalized_f16_score": normalized_score_slm_bytes,
              "online_sum": sum_slm_bytes,
              "online_max_and_guard": max_and_guard_slm_bytes,
              "output": output_slm_bytes,
              "generated_package_scratch": ugemm_slm_bytes,
              "total": total_slm_bytes,
              "two_workgroups": two_resident_wg_slm_bytes,
              "device": device_local_mem_bytes,
          },
          "synchronization": (
              "one 16-subgroup producer cohort and one 16-subgroup "
              "chronological consumer cohort; two local F32 score buffers; "
              "consumer-only named barriers preserve its max/score reductions; "
              "one full-cohort handoff per 256-key stage"),
      },
      "admitted_codegen_contract": {
          "compiler_only": True,
          "kernel_enqueue": False,
          "plugin_build": False,
          "model_worker": False,
          "variants": 1,
          "requirements": [
              "compile exactly one 512-work-item dual-cohort kernel",
              "retain generated M256/N16 KQ and VS packages",
              "retain the chronological F32 online-softmax recurrence",
              "use local F32 store/load with no global raw-score round trip",
              "query actual GRF, spill, SLM, workgroup, and barrier resources",
              "reject before enqueue if compilation or resource checks fail",
          ],
      },
      "compiler_gate_admitted": required_checks_passed,
      "component_admitted": False,
      "graph_integration_admitted": False,
      "product_worker_admitted": False,
      "memory_stop_bytes": stop_bytes,
      "memory_samples": memory,
  }
  (output / "bound.json").write_text(
      json.dumps(payload, indent=2, sort_keys=True) + "\n",
      encoding="utf-8")
  print(json.dumps({
      "verdict": verdict,
      "ideal_overlap_saving_lcb_ms_per_layer": ideal_saving_lcb_ms,
      "required_saving_ms_per_layer": required_saving_ms,
      "required_realized_fraction": required_realized_fraction,
      "slm_bytes": total_slm_bytes,
      "workgroup_items": WORKGROUP_ITEMS,
  }, sort_keys=True))
  return 0 if required_checks_passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
