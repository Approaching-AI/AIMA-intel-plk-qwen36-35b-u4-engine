#!/usr/bin/env python3
"""Source-bound one 64k adaptive block32-I8 attention component."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MIB = 1024 * 1024
HEAD_DIM = 256
QUERY_HEADS = 16
KV_HEADS = 2
GQA_GROUP = QUERY_HEADS // KV_HEADS
LAYERS = (3, 7, 11, 15, 19, 23, 27, 31, 35, 39)
HIGH_TOPK_LAYERS = (3, 7)
BASE_CONTEXT = 32768
TARGET_CONTEXT = 65536
HOT_TOKENS = 16384
COLD_TOKENS = TARGET_CONTEXT - HOT_TOKENS
CHUNK_TOKENS = 512
LOCAL_TOPK = 64
HIGH_TOPK = 512
LOW_TOPK = 256
QUANT_GROUP = 32
MEASURED_DENSE_INCREMENT_MS = 10.505773
MIN_MEMORY_BYTES = 4 * 1024**3

SOURCE_PATHS = (
    Path("engine/gpu/opencl/direct_i8_hotcold_gqa_decode.cl"),
    Path("engine/tools/direct_i8_hotcold_gqa_decode.cpp"),
    Path("tools/intel_qwen36_openvino_hot_cold_attention.py"),
    Path("engine/openvino/custom/iq36_hot_attention_single_owner.cl"),
    Path("engine/openvino/custom/iq36_stock_micro_attention_oracle.cl"),
    Path("engine/openvino/custom/iq36_hot_attention_gqa.xml"),
)


def parser() -> argparse.ArgumentParser:
  result = argparse.ArgumentParser(description=__doc__)
  result.add_argument("--adaptive-bound", type=Path, required=True)
  result.add_argument("--carrier-result", type=Path, required=True)
  result.add_argument("--product-worker", type=Path, required=True)
  result.add_argument("--out-dir", type=Path, required=True)
  return result


def load_json(path: Path) -> dict[str, Any]:
  return json.loads(path.read_text(encoding="utf-8"))


def git_state() -> dict[str, Any]:
  commit = subprocess.check_output(
      ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
  dirty_paths = subprocess.check_output(
      ["git", "status", "--short"], cwd=ROOT, text=True).splitlines()
  return {
      "commit": commit,
      "dirty": bool(dirty_paths),
      "dirty_paths": dirty_paths,
  }


def mem_available_bytes() -> int:
  for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
    if line.startswith("MemAvailable:"):
      return int(line.split()[1]) * 1024
  raise RuntimeError("MemAvailable missing from /proc/meminfo")


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as source:
    while chunk := source.read(1024 * 1024):
      digest.update(chunk)
  return digest.hexdigest()


def close(left: float, right: float, *, relative: float = 1.0e-12) -> bool:
  return math.isclose(left, right, rel_tol=relative, abs_tol=1.0e-6)


def check(name: str, passed: bool, **evidence: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **evidence}


def correction_identity_error() -> float:
  approximate_scores = (-1.75, 0.25, -0.5, 1.125, -2.0)
  exact_scores = (-1.5, -0.125, -0.375, 0.75, -1.875)
  approximate_values = (0.5, -1.25, 2.0, 0.75, -0.25)
  exact_values = (0.625, -1.0, 1.5, 1.25, -0.5)
  selected = (1, 3)
  approximate_max = max(approximate_scores)
  approximate_sum = sum(
      math.exp(score - approximate_max) for score in approximate_scores)
  approximate_numerator = sum(
      math.exp(score - approximate_max) * value
      for score, value in zip(approximate_scores, approximate_values))
  correction_max = max(
      approximate_max, *(exact_scores[index] for index in selected))
  corrected_sum = approximate_sum * math.exp(
      approximate_max - correction_max)
  corrected_numerator = approximate_numerator * math.exp(
      approximate_max - correction_max)
  for index in selected:
    approximate_weight = math.exp(
        approximate_scores[index] - correction_max)
    exact_weight = math.exp(exact_scores[index] - correction_max)
    corrected_sum += exact_weight - approximate_weight
    corrected_numerator += (
        exact_weight * exact_values[index] -
        approximate_weight * approximate_values[index])

  direct_sum = 0.0
  direct_numerator = 0.0
  for index, (score, value) in enumerate(zip(
      approximate_scores, approximate_values)):
    if index in selected:
      score = exact_scores[index]
      value = exact_values[index]
    weight = math.exp(score - correction_max)
    direct_sum += weight
    direct_numerator += weight * value
  return max(
      abs(corrected_sum - direct_sum),
      abs(corrected_numerator - direct_numerator),
      abs(corrected_numerator / corrected_sum -
          direct_numerator / direct_sum),
  )


def source_audit() -> tuple[dict[str, bool], list[dict[str, Any]]]:
  texts = {
      str(path): (ROOT / path).read_text(encoding="utf-8")
      for path in SOURCE_PATHS
  }
  carrier = texts[str(SOURCE_PATHS[0])]
  runner = texts[str(SOURCE_PATHS[1])]
  graph = texts[str(SOURCE_PATHS[2])]
  owner = texts[str(SOURCE_PATHS[3])]
  oracle = texts[str(SOURCE_PATHS[4])]
  registration = texts[str(SOURCE_PATHS[5])]
  checks = {
      "group32_dpas_scan_carrier": (
          "#define IQ36_QUANT_GROUP 32" in carrier and
          carrier.count("intel_sub_group_f16_f16_matrix_mad_k16") >= 4 and
          "iq36_direct_i8_hotcold_partial" in carrier),
      "global_max_sum_numerator_carrier": all(
          token in carrier for token in (
              "partial_max", "partial_sum", "partial_output",
              "iq36_direct_i8_hotcold_reduce")),
      "ordered_update_carrier": (
          "iq36_direct_i8_update_state" in carrier and
          "IQ36_COMPONENT_UPDATE_AFTER_ATTENTION" in runner and
          "partial_then_reduce_then_update" in runner),
      "no_scalar_local_f32_kv_tile": not any(
          token in carrier for token in (
              "__local float key", "__local float value",
              "float local_key", "float local_value")),
      "exact_history_capacity_is_parameterized": all(
          token in graph for token in (
              "exact_history_capacity", "physical_ring_capacities",
              "hot_key_storage_planes")),
      "fixed_cold_state_is_in_place": all(
          token in owner for token in (
              "fixed_cold_state", "iq36_direct_store_cold_key",
              "iq36_direct_store_cold_value")),
      "prefill_constructs_compressed_cold_state": all(
          token in owner for token in (
              "cold_append_tokens", "cold_key_append",
              "cold_value_append")),
      "exact_dense_sidecar_is_written": (
          "iq36_hot_key_dense_i32_base" in owner and
          "dense_hot_key[dense_key_index]" in owner and
          "hot_value[hot_value_base" in owner),
      "exact_dense_sidecar_is_read_by_oracle": (
          "iq36_hot_key_dense_i32_base" in oracle and
          "iq36_stock_micro_attention_oracle" in oracle),
      "single_owner_registration_has_six_state_outputs": (
          registration.count('type="output" port-index=') >= 6 and
          "IQ36ExactPhaseHotAttentionGQA" in registration and
          "IQ36_STOCK_MICRO_OWNER_WRITE_CURRENT=1" in registration),
      "current_runner_remains_32k_only": (
          "constexpr cl_uint kContextTokens = 32768" in runner and
          '"the admitted component context is exactly 32768 tokens"'
          in carrier),
  }
  files = [{
      "path": str(path),
      "sha256": sha256(ROOT / path),
  } for path in SOURCE_PATHS]
  return checks, files


def main() -> int:
  args = parser().parse_args()
  if args.out_dir.exists():
    raise SystemExit(f"output already exists: {args.out_dir}")
  available_before = mem_available_bytes()
  if available_before < MIN_MEMORY_BYTES:
    raise SystemExit(
        f"memory stop: {available_before} < {MIN_MEMORY_BYTES} bytes")

  adaptive = load_json(args.adaptive_bound)
  carrier = load_json(args.carrier_result)
  product = load_json(args.product_worker)
  source_checks, source_files = source_audit()
  repository = git_state()

  rule = adaptive.get("rule", {})
  traffic = adaptive.get("traffic", {})
  numeric = adaptive.get("numeric", {})
  carrier_result = carrier.get("result", {})
  carrier_inference = carrier.get("performance_inference", {})
  product_source = product.get("source_summary", {})

  adaptive_rule_checks = {
      "layers": tuple(rule.get("layers", ())) == LAYERS,
      "high_topk_layers": (
          tuple(rule.get("high_topk_layers", ())) == HIGH_TOPK_LAYERS),
      "hot_tokens": rule.get("hot_tokens") == HOT_TOKENS,
      "chunk_tokens": rule.get("candidate_chunk_tokens") == CHUNK_TOKENS,
      "local_topk": rule.get("local_topk_per_query") == LOCAL_TOPK,
      "high_topk": rule.get("high_topk_per_query") == HIGH_TOPK,
      "low_topk": rule.get("topk_per_query") == LOW_TOPK,
      "selection_record": (
          rule.get("selection_record") ==
          "F16 score plus U16 absolute cold-token index"),
      "selection_tie": (
          rule.get("selection_tie_break") ==
          "score descending, then absolute cold-token index ascending"),
      "u16_addressable": COLD_TOKENS <= 65536,
  }

  dense_increment_bytes = int(traffic.get("dense_f16_bytes", 0))
  bandwidth_bytes_per_second = (
      dense_increment_bytes / (MEASURED_DENSE_INCREMENT_MS / 1000.0))
  bandwidth_bytes_per_ms = bandwidth_bytes_per_second / 1000.0
  allowed_bytes = float(traffic.get("allowed_bytes", 0.0))
  admitted_before_aux_bytes = float(
      traffic.get("total_bytes_before_compute", 0.0))

  bitset_bytes_per_layer = KV_HEADS * math.ceil(COLD_TOKENS / 32) * 4
  bitset_clear_and_read_bytes_per_layer = 2 * bitset_bytes_per_layer
  aggregate_bytes_per_layer = QUERY_HEADS * (2 + HEAD_DIM) * 4
  aggregate_write_read_bytes_per_layer = 2 * aggregate_bytes_per_layer
  compressed_update_bytes_per_layer = int(
      KV_HEADS * HEAD_DIM * 2 * (1.0 + 2.0 / QUANT_GROUP))

  def auxiliary_bytes(topk: int) -> int:
    union_atomic_bytes = QUERY_HEADS * topk * 8
    return (
        bitset_clear_and_read_bytes_per_layer + union_atomic_bytes +
        aggregate_write_read_bytes_per_layer +
        compressed_update_bytes_per_layer)

  high_auxiliary_bytes = auxiliary_bytes(HIGH_TOPK)
  low_auxiliary_bytes = auxiliary_bytes(LOW_TOPK)
  total_auxiliary_bytes = (
      len(HIGH_TOPK_LAYERS) * high_auxiliary_bytes +
      (len(LAYERS) - len(HIGH_TOPK_LAYERS)) * low_auxiliary_bytes)
  headroom_after_aux_bytes = (
      allowed_bytes - admitted_before_aux_bytes - total_auxiliary_bytes)
  compute_dispatch_budget_ms = (
      headroom_after_aux_bytes / bandwidth_bytes_per_ms)
  compute_dispatch_budget_per_layer_ms = (
      compute_dispatch_budget_ms / len(LAYERS))

  compressed_scan_per_layer = (
      float(traffic["block32_scan_bytes"]) / len(LAYERS))
  candidate_workspace_per_layer = (
      float(traffic["candidate_workspace_read_write_bytes"]) /
      len(LAYERS))
  dense_increment_per_layer = dense_increment_bytes / len(LAYERS)
  correction_source_per_layer = (
      dense_increment_per_layer + compressed_scan_per_layer)

  def layer_cap(topk: int, aux_bytes: int) -> dict[str, float]:
    fraction = GQA_GROUP * topk / COLD_TOKENS
    bounded_bytes = (
        compressed_scan_per_layer + candidate_workspace_per_layer +
        correction_source_per_layer * fraction)
    cap_bytes = (
        bounded_bytes + aux_bytes + headroom_after_aux_bytes / len(LAYERS))
    return {
        "auxiliary_bytes": aux_bytes,
        "source_bound_bytes_before_aux": bounded_bytes,
        "matched_64k_minus_32k_ucb_cap_ms": (
            cap_bytes / bandwidth_bytes_per_ms),
        "topk_per_query": topk,
        "worst_union_fraction": fraction,
        "worst_union_rows_per_kv_head": GQA_GROUP * topk,
    }

  high_layer_cap = layer_cap(HIGH_TOPK, high_auxiliary_bytes)
  low_layer_cap = layer_cap(LOW_TOPK, low_auxiliary_bytes)
  weighted_incremental_cap_ms = (
      len(HIGH_TOPK_LAYERS) *
      high_layer_cap["matched_64k_minus_32k_ucb_cap_ms"] +
      (len(LAYERS) - len(HIGH_TOPK_LAYERS)) *
      low_layer_cap["matched_64k_minus_32k_ucb_cap_ms"])
  allowed_incremental_ms = allowed_bytes / bandwidth_bytes_per_ms

  exact_bytes_per_token_per_kv_head = HEAD_DIM * 2 * 2
  compressed_bytes_per_token_per_kv_head = int(
      HEAD_DIM * 2 * (1.0 + 2.0 / QUANT_GROUP))
  full_scan_bytes_per_layer = (
      COLD_TOKENS * KV_HEADS * compressed_bytes_per_token_per_kv_head +
      HOT_TOKENS * KV_HEADS * exact_bytes_per_token_per_kv_head)

  def full_layer_bytes(topk: int) -> int:
    union_rows = GQA_GROUP * topk
    correction_bytes = (
        KV_HEADS * union_rows *
        (compressed_bytes_per_token_per_kv_head +
         exact_bytes_per_token_per_kv_head))
    return int(
        full_scan_bytes_per_layer + correction_bytes +
        candidate_workspace_per_layer)

  full_high_bytes = full_layer_bytes(HIGH_TOPK)
  full_low_bytes = full_layer_bytes(LOW_TOPK)
  full_candidate_bytes = (
      len(HIGH_TOPK_LAYERS) * full_high_bytes +
      (len(LAYERS) - len(HIGH_TOPK_LAYERS)) * full_low_bytes)
  base_32k_dense_bytes = (
      BASE_CONTEXT * KV_HEADS * exact_bytes_per_token_per_kv_head *
      len(LAYERS))
  physical_increment_before_aux_bytes = (
      full_candidate_bytes - base_32k_dense_bytes)

  identity_error = correction_identity_error()
  product_expected_layers = list(LAYERS)
  product_state_checks = {
      "ten_custom_attention_owners": (
          product_source.get("custom_count_after") == len(LAYERS)),
      "all_exact_history_layers": (
          product_source.get("exact_history_layers") ==
          product_expected_layers),
      "exact_history_capacity_covers_64k": (
          int(product_source.get("exact_history_capacity", 0)) >=
          TARGET_CONTEXT),
      "fixed_cold_capacity_covers_target": (
          int(product_source.get("fixed_cold_capacity", 0)) >=
          COLD_TOKENS),
      "dense_key_sidecar_plane_present": (
          int(product_source.get("hot_key_storage_planes", 0)) >= 2),
      "dense_value_sidecar_capacity_covers_64k": (
          product_source.get("hot_value_shape", [0, 0, 0])[2] >=
          TARGET_CONTEXT),
      "compressed_cold_storage_present": (
          "block32 I8" in product_source.get("cold_storage", "")),
      "exact_dense_hot_storage_present": (
          "contiguous F16" in product_source.get("hot_storage", "") and
          "direct F16 V" in product_source.get("hot_storage", "")),
  }

  carrier_checks = {
      "promoted": carrier.get("component_promoted") is True,
      "all_required": carrier.get("required_checks_passed") is True,
      "clean_evidence": carrier.get("git", {}).get("dirty") is False,
      "group32_dpas": (
          carrier_result.get("algorithm") ==
          "direct_i8_block32_hot8192_f16_dpas" and
          carrier_result.get("quant_group") == QUANT_GROUP),
      "32k_shape": carrier_result.get("context_tokens") == BASE_CONTEXT,
      "numeric": (
          float(carrier_result.get("output_relative_l2", 1.0)) < 2.0e-4 and
          float(carrier_result.get("output_cosine", 0.0)) > 0.9999999),
      "timing_ucb": (
          float(carrier_inference.get("upper_confidence_bound_ms", math.inf))
          <= float(carrier_inference.get("cap_ms", -math.inf))),
  }

  budget = {
      "allowed_bytes": allowed_bytes,
      "allowed_incremental_ms": allowed_incremental_ms,
      "bandwidth_bytes_per_second": bandwidth_bytes_per_second,
      "bandwidth_gb_per_second": bandwidth_bytes_per_second / 1.0e9,
      "compute_dispatch_headroom_after_aux_bytes": (
          headroom_after_aux_bytes),
      "compute_dispatch_headroom_after_aux_mib": (
          headroom_after_aux_bytes / MIB),
      "compute_dispatch_budget_ms": compute_dispatch_budget_ms,
      "compute_dispatch_budget_per_layer_ms": (
          compute_dispatch_budget_per_layer_ms),
      "high_layer": high_layer_cap,
      "low_layer": low_layer_cap,
      "source_bound_bytes_before_aux": admitted_before_aux_bytes,
      "source_bound_mib_before_aux": admitted_before_aux_bytes / MIB,
      "topology_auxiliary_bytes": total_auxiliary_bytes,
      "topology_auxiliary_mib": total_auxiliary_bytes / MIB,
      "weighted_matched_64k_minus_32k_ucb_cap_ms": (
          weighted_incremental_cap_ms),
  }
  physical_cross_check = {
      "base_32k_dense_bytes": base_32k_dense_bytes,
      "candidate_full_64k_bytes_before_aux": full_candidate_bytes,
      "candidate_full_64k_mib_before_aux": full_candidate_bytes / MIB,
      "high_layer_full_bytes_before_aux": full_high_bytes,
      "low_layer_full_bytes_before_aux": full_low_bytes,
      "physical_increment_before_aux_bytes": (
          physical_increment_before_aux_bytes),
      "physical_increment_before_aux_mib": (
          physical_increment_before_aux_bytes / MIB),
      "physical_increment_with_aux_bytes": (
          physical_increment_before_aux_bytes + total_auxiliary_bytes),
      "physical_increment_with_aux_mib": (
          (physical_increment_before_aux_bytes + total_auxiliary_bytes) /
          MIB),
      "conservative_source_bound_remains_larger": (
          admitted_before_aux_bytes + total_auxiliary_bytes >=
          physical_increment_before_aux_bytes + total_auxiliary_bytes),
  }

  topology = {
      "dispatch_count_per_layer": 4,
      "dispatches": [
          {
              "name": "partial_scan_local_select",
              "work": (
                  "group32 DPAS mixed cold/hot scan; emit unnormalized F32 "
                  "partial max/sum/numerator and local top64 F16/U16 records; "
                  "clear the KV-head union bitset before the next dispatch"),
          },
          {
              "name": "select_reduce_union",
              "work": (
                  "one work-group per query head; load its 6144 records once "
                  "into local memory, use two F16-byte histograms to select "
                  "the exact deterministic top512/top256, atomically OR the "
                  "KV-head bitset, and reduce approximate global m/Z/N"),
          },
          {
              "name": "correct_normalize",
              "work": (
                  "one work-group per KV head; scan the union bitset, reread "
                  "compressed and exact-sidecar K/V, remove approximate "
                  "terms, insert exact terms, then normalize"),
          },
          {
              "name": "ordered_update",
              "work": (
                  "one post-attention owner appends compressed cold state and "
                  "the exact F16 sidecar; never bulk-build state in decode"),
          },
      ],
      "global_candidate_record_rereads": 1,
      "selection": (
          "F16 ordered score descending, then absolute U16 cold-token index "
          "ascending; histogram thresholding does not require a global sort"),
      "union_representation": (
          "two KV-head bitsets; correction scans bits directly, with no "
          "compaction dispatch"),
  }
  correction = {
      "definitions": "approximate aggregate is (m_a, Z_a, N_a); U is union",
      "rescale": "m_c = max(m_a, max_U(s_e))",
      "sum": (
          "Z_c = Z_a*exp(m_a-m_c) + "
          "sum_U(exp(s_e-m_c)-exp(s_a-m_c))"),
      "numerator": (
          "N_c = N_a*exp(m_a-m_c) + "
          "sum_U(exp(s_e-m_c)*v_e-exp(s_a-m_c)*v_a)"),
      "output": "N_c / Z_c",
      "identity_max_abs": identity_error,
  }

  checks = [
      check("repository_clean_at_bound", not repository["dirty"],
            git=repository),
      check("memory_preflight", available_before >= MIN_MEMORY_BYTES,
            available_bytes=available_before,
            stop_bytes=MIN_MEMORY_BYTES),
      check(
          "adaptive_numeric_and_traffic_bound_is_clean",
          adaptive.get("schema") ==
          "intel-qwen36-openvino-adaptive-attention-bound-v3" and
          adaptive.get("all_required_checks_pass") is True and
          adaptive.get("git", {}).get("dirty") is False and
          numeric.get("pass") is True and traffic.get("pass") is True,
          evidence_git=adaptive.get("git"),
          schema=adaptive.get("schema")),
      check("adaptive_geometry_is_locked", all(adaptive_rule_checks.values()),
            geometry_checks=adaptive_rule_checks),
      check("admitted_group32_dpas_carrier", all(carrier_checks.values()),
            carrier_checks=carrier_checks),
      check("existing_source_ownership_and_carriers",
            all(source_checks.values()), source_checks=source_checks),
      check("64k_product_state_already_has_both_representations",
            all(product_state_checks.values()),
            product_state_checks=product_state_checks),
      check("exact_replacement_algebra", identity_error <= 1.0e-12,
            identity_max_abs=identity_error),
      check(
          "complete_topology_fits_conservative_source_bound",
          headroom_after_aux_bytes > 0.0 and
          close(weighted_incremental_cap_ms, allowed_incremental_ms),
          budget=budget),
      check(
          "physical_full_state_cross_check_is_below_conservative_bound",
          physical_cross_check["conservative_source_bound_remains_larger"],
          physical_cross_check=physical_cross_check),
      check(
          "implementation_scope_is_one_standalone_component",
          topology["dispatch_count_per_layer"] == 4 and
          topology["global_candidate_record_rereads"] == 1,
          graph_integration_admitted=False,
          product_worker_admitted=False,
          long_worker_admitted=False),
  ]
  all_required_checks_pass = all(row["pass"] for row in checks)
  payload = {
      "adaptive_component_implementation_admitted": (
          all_required_checks_pass),
      "all_required_checks_pass": all_required_checks_pass,
      "budget": budget,
      "checks": checks,
      "correction": correction,
      "evidence": {
          "adaptive_bound": str(args.adaptive_bound),
          "carrier_result": str(args.carrier_result),
          "product_worker": str(args.product_worker),
          "source_files": source_files,
      },
      "git": repository,
      "graph_integration_admitted": False,
      "long_worker_admitted": False,
      "memory": {
          "available_after_bytes": mem_available_bytes(),
          "available_before_bytes": available_before,
          "process_max_rss_bytes": (
              resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
          "stop_bytes": MIN_MEMORY_BYTES,
      },
      "physical_traffic_cross_check": physical_cross_check,
      "product_claim_allowed": False,
      "product_worker_admitted": False,
      "schema": (
          "intel-qwen36-openvino-adaptive-attention-component-bound-v1"),
      "topology": topology,
  }
  args.out_dir.mkdir(parents=True)
  (args.out_dir / "bound.json").write_text(
      json.dumps(payload, indent=2, sort_keys=True) + "\n",
      encoding="utf-8")
  summary = [
      "# Adaptive attention standalone-component source bound",
      "",
      f"- all required checks pass: `{all_required_checks_pass}`",
      "- implementation admitted: "
      f"`{payload['adaptive_component_implementation_admitted']}`",
      "- dispatch topology: `4/layer` "
      "(scan+local-select, select+reduce+union, correct+normalize, update)",
      f"- conservative traffic before topology aux: "
      f"`{admitted_before_aux_bytes / MIB:.3f} MiB/token`",
      f"- topology aux: `{total_auxiliary_bytes / MIB:.6f} MiB/token`",
      f"- compute/dispatch headroom after aux: "
      f"`{headroom_after_aux_bytes / MIB:.3f} MiB`, "
      f"`{compute_dispatch_budget_ms:.6f} ms/token`",
      f"- matched high-layer incremental UCB cap: "
      f"`{high_layer_cap['matched_64k_minus_32k_ucb_cap_ms']:.6f} ms`",
      f"- matched low-layer incremental UCB cap: "
      f"`{low_layer_cap['matched_64k_minus_32k_ucb_cap_ms']:.6f} ms`",
      f"- weighted 2/8 incremental UCB cap: "
      f"`{weighted_incremental_cap_ms:.6f} ms/token`",
      f"- physical full-state increment with aux: "
      f"`{physical_cross_check['physical_increment_with_aux_mib']:.3f} MiB`",
      "",
      "This admits exactly one standalone component implementation. It does "
      "not admit graph integration, a long worker, a product run, or a speed "
      "claim. The component gate must measure matched 32k and 64k contexts "
      "with at least 20 interleaved samples and apply one-sided 95% UCBs.",
  ]
  (args.out_dir / "summary.md").write_text(
      "\n".join(summary) + "\n", encoding="utf-8")
  print(json.dumps({
      "all_required_checks_pass": all_required_checks_pass,
      "event": "adaptive_attention_component_bound_complete",
      "implementation_admitted": (
          payload["adaptive_component_implementation_admitted"]),
      "out_dir": str(args.out_dir),
  }, sort_keys=True))
  return 0 if all_required_checks_pass else 1


if __name__ == "__main__":
  raise SystemExit(main())
