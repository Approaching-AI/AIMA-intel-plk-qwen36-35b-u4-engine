#!/usr/bin/env python3
"""Source-bound one exact full-score raw-F32 KQ staging component.

This gate does not compile or launch a GPU/model worker.  It closes the
official N=8 package branch, distinguishes raw-score staging from the rejected
partition/duplicate/adaptive families, and derives the one-layer paired cap
that a standalone 128k component must clear before graph integration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-openvino-exact-score-staging-bound-v2"

MODEL_CONFIG = Path("/home/intel/Qwen3.6-35B-A3B-ov/config.json")
ORACLE = ROOT / "engine/openvino/custom/iq36_stock_micro_attention_oracle.cl"
SHIMS = ROOT / "engine/openvino/custom/iq36_decode_microkernel_shims.cl"
MULTIKERNEL_PATCH = (
    ROOT / "engine/openvino/iq36-custom-adaptive-attention-multikernel.patch")
ADAPTIVE_SOURCE = (
    ROOT / "engine/openvino/custom/iq36_adaptive_attention_decode.cl")
FRONTIER = ROOT / "doc/active" / WS / "frontier.json"
STATUS = ROOT / "doc/active" / WS / "STATUS.md"
EXACT_N8 = ROOT / (
    "output/openvino-exact-attention-nhalf-capability-"
    "20260723Tseq2123c-clean/capability-gate-result.json")
THINQ_N8 = ROOT / (
    "output/openvino-attention-hpg-thinq-m8-n8-capability-"
    "20260723Tseq2124b-clean/capability-gate-result.json")
PROFILE = ROOT / (
    "output/openvino-current-bundle-profile-refresh-"
    "20260723Tseq2122b-all10-128k-o3/metrics.json")

CONTEXT_TOKENS = 131072
KEY_BLOCK = 256
F32_BYTES = 4
F16_BYTES = 2
PLANNING_GB_S = 115.0
RAW_LPDDR_GB_S = 136.5
MIN_COMPONENT_SAMPLES = 20
KQ_SCORE_COLUMNS_PER_KV = 16


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


def official_gate_closed(
    payload: dict[str, Any], *, key_block: int, unroll_m: int,
) -> bool:
  current = payload.get("current_official", {})
  diagnostic = str(current.get("diagnostic", ""))
  return bool(
      payload.get("repository_status") == ""
      and payload.get("key_block") == key_block
      and payload.get("kq_unroll_m") == unroll_m
      and payload.get("checks", {}).get("provider_scopes_clean") is True
      and payload.get("checks", {}).get("current_probe_completed") is True
      and payload.get("checks", {}).get(
          "current_generates_requested_kq_and_vs_nhalf") is False
      and current.get("generated") is False
      and "Functionality is unimplemented" in diagnostic
      and "No matching kernel" in diagnostic)


def main() -> int:
  args = parse_args()
  output = args.output.resolve()
  output.mkdir(parents=True, exist_ok=False)
  stop_bytes = int(args.memory_stop_gib * 1024**3)
  memory: list[dict[str, Any]] = []
  sample_memory("start", stop_bytes, memory)

  required = (
      MODEL_CONFIG, ORACLE, SHIMS, MULTIKERNEL_PATCH, ADAPTIVE_SOURCE,
      FRONTIER, STATUS, EXACT_N8, THINQ_N8, PROFILE)
  missing = [str(path) for path in required if not path.is_file()]
  if missing:
    raise SystemExit("missing source-bound inputs: " + ", ".join(missing))

  git = git_state(output)
  model = load_json(MODEL_CONFIG).get("text_config", {})
  frontier = load_json(FRONTIER)
  exact_n8 = load_json(EXACT_N8)
  thinq_n8 = load_json(THINQ_N8)
  profile = load_json(PROFILE)
  oracle = ORACLE.read_text(encoding="utf-8")
  shims = SHIMS.read_text(encoding="utf-8")
  multikernel = MULTIKERNEL_PATCH.read_text(encoding="utf-8")
  adaptive = ADAPTIVE_SOURCE.read_text(encoding="utf-8")
  status = STATUS.read_text(encoding="utf-8")
  sample_memory("after-source-audit", stop_bytes, memory)

  layers = int(model.get("num_hidden_layers", -1))
  interval = int(model.get("full_attention_interval", -1))
  query_heads = int(model.get("num_attention_heads", -1))
  kv_heads = int(model.get("num_key_value_heads", -1))
  head_dim = int(model.get("head_dim", -1))
  full_attention_layers = layers // interval if interval > 0 else -1
  gqa_group = query_heads // kv_heads if kv_heads > 0 else -1
  blocks_per_kv_head = math.ceil(CONTEXT_TOKENS / KEY_BLOCK)

  median_cut_ms = 1.175998
  best_cut_ms = float(
      frontier["goal_budget"]["per_token_ms"]["remaining_cut"])
  per_layer_median_cut_ms = median_cut_ms / full_attention_layers
  per_layer_best_cut_ms = best_cut_ms / full_attention_layers

  # The generated M256/N16 KQ carrier publishes sixteen independent F32
  # columns per KV head before its F16 score-package handoff.  The eight
  # logical GQA heads are not a removable duplicate half at this boundary.
  score_elements_per_layer = (
      CONTEXT_TOKENS * kv_heads * KQ_SCORE_COLUMNS_PER_KV)
  score_one_way_bytes_per_layer = score_elements_per_layer * F32_BYTES
  score_roundtrip_bytes_per_layer = 2 * score_one_way_bytes_per_layer
  query_tile_bytes = gqa_group * head_dim * F16_BYTES
  staged_query_read_bytes_per_layer = (
      blocks_per_kv_head * kv_heads * query_tile_bytes)
  fused_query_read_bytes_per_layer = kv_heads * query_tile_bytes
  worst_extra_query_bytes_per_layer = (
      staged_query_read_bytes_per_layer - fused_query_read_bytes_per_layer)
  auxiliary_bytes_per_layer = (
      score_roundtrip_bytes_per_layer + worst_extra_query_bytes_per_layer)
  auxiliary_bytes_all_layers = auxiliary_bytes_per_layer * full_attention_layers

  def traffic_ms(byte_count: int, gb_s: float) -> float:
    return byte_count / (gb_s * 1_000_000_000.0) * 1000.0

  score_ms_planning = traffic_ms(
      score_roundtrip_bytes_per_layer * full_attention_layers,
      PLANNING_GB_S)
  worst_aux_ms_planning = traffic_ms(
      auxiliary_bytes_all_layers, PLANNING_GB_S)
  worst_aux_ms_raw = traffic_ms(
      auxiliary_bytes_all_layers, RAW_LPDDR_GB_S)
  required_fused_kq_saving_ms = median_cut_ms + worst_aux_ms_planning
  required_fused_kq_saving_per_layer_ms = (
      required_fused_kq_saving_ms / full_attention_layers)
  current_useful_groups_per_layer = kv_heads
  staged_kq_groups_per_layer = blocks_per_kv_head * kv_heads
  owner_groups_per_layer = kv_heads

  model_exact = (
      layers == 40 and interval == 4 and query_heads == 16
      and kv_heads == 2 and head_dim == 256 and full_attention_layers == 10
      and gqa_group == 8)
  oracle_markers = {
      "chronological_kq_loop":
          "for (int key_begin = 0; key_begin < micro_key_tokens;" in oracle,
      "generated_kq":
          "iq36_score_tile score = ugemm_kq(" in oracle,
      "f32_online_max_sum":
          "iq36_score_sum_tile running_sum;" in oracle
          and "iq36_score_sum_tile running_max;" in oracle,
      "f16_score_publication":
          "tile_copy_to_half2(score, score_half2);" in oracle
          and "tile_store_t_sys_src2(" in oracle,
      "generated_vs":
          "iq36_accumulator_tile chunk_accumulator = ugemm_vs(" in oracle,
      "chronological_rescale":
          "tile_binary(\n          old_running_max, running_max, iq36_rescale);"
          in oracle,
  }
  shim_markers = {
      "kq_m256_n16":
          "#define ugemm_kq_wg_tile_m 256" in shims
          and "#define ugemm_kq_wg_tile_n 16" in shims,
      "vs_m256_n16":
          "#define ugemm_vs_wg_tile_m 256" in shims
          and "#define ugemm_vs_wg_tile_n 16" in shims,
      "sixteen_subgroups":
          "#define ugemm_kq_sg_per_wg_m 16" in shims
          and "#define ugemm_kq_sg_per_wg_n 1" in shims,
      "systolic_packages":
          "#define ugemm_kq_systolic  1" in shims
          and "#define ugemm_vs_systolic  1" in shims,
  }
  multikernel_markers = {
      "ordered_device_chain":
          "std::vector<event::ptr> dependencies(events);" in multikernel
          and "dependencies = {last_event};" in multikernel,
      "all_stage_arguments":
          "stream.set_arguments(*_kernels[index]" in multikernel,
      "aggregate_stage_events":
          "stream.aggregate_events(stage_events" in multikernel,
  }
  distinct_from_adaptive = bool(
      "__global float* approximate_score" in adaptive
      and "iq36_adaptive_attention_select_reduce_union" in adaptive
      and "iq36_adaptive_attention_correct_normalize" in adaptive
      and "raw_f32_exact_kq_score" not in adaptive)
  profile_selects_attention = bool(
      profile.get("required_checks_passed") is True
      and profile.get("profile_rollup", {}).get(
          "dominant_retained_node_type") ==
          "IQ36ExactPhaseHotAttentionGQA")

  checks = [
      check("repository_clean_at_gate", not git["dirty"], git=git),
      check("locked_model_geometry_is_exact", model_exact),
      check("latest_official_m16_n8_remains_unsupported",
            official_gate_closed(exact_n8, key_block=256, unroll_m=16)),
      check("latest_official_m8_n8_alternate_remains_unsupported",
            official_gate_closed(thinq_n8, key_block=128, unroll_m=8)),
      check("accepted_wrapper_preserves_generated_chronological_contract",
            all(oracle_markers.values()), markers=oracle_markers),
      check("generated_shims_have_exact_m256_n16_geometry",
            all(shim_markers.values()), markers=shim_markers),
      check("device_ordered_multikernel_carrier_exists",
            all(multikernel_markers.values()), markers=multikernel_markers),
      check("route_is_distinct_from_adaptive_compressed_score_staging",
            distinct_from_adaptive),
      check("current_profile_selects_exact_attention_owner",
            profile_selects_attention),
      check("current_128k_kill_numbers_are_registered",
            math.isclose(best_cut_ms, 0.345159, abs_tol=1.0e-9)
            and "1.175998 ms/token" in status),
      check("full_n16_f32_score_scratch_is_bounded",
            score_one_way_bytes_per_layer == 16 * 1024**2
            and score_roundtrip_bytes_per_layer == 32 * 1024**2
            and score_roundtrip_bytes_per_layer * full_attention_layers
                == 320 * 1024**2),
      check("staging_materially_increases_useful_kq_parallelism",
            current_useful_groups_per_layer == 2
            and staged_kq_groups_per_layer == 1024
            and owner_groups_per_layer == 2),
      check("memory_guard_never_tripped",
            all(row["available_bytes"] >= stop_bytes for row in memory),
            minimum_available_bytes=min(
                row["available_bytes"] for row in memory)),
  ]
  required_checks_passed = all(row["pass"] for row in checks)
  verdict = (
      "admit_one_exact_raw_f32_score_staging_component"
      if required_checks_passed else "inconclusive")

  component_contract = {
      "context_tokens": CONTEXT_TOKENS,
      "sample_count": MIN_COMPONENT_SAMPLES,
      "schedule": "interleaved_fused_staged_staged_fused",
      "baseline": (
          "current generated M256/N16 KQ plus chronological F32 online "
          "softmax plus generated M256/N16 VS in one owner per KV head"),
      "candidate_stage_kq": (
          "one work-group per 256-key block per KV head; generated KQ; write "
          "all sixteen independent N16 carrier columns as raw F32 scores"),
      "candidate_stage_owner": (
          "one work-group per KV head; read raw scores in original key-block "
          "order; run the unchanged max/exp/sum recurrence and generated VS"),
      "correctness": (
          "bitwise F32 raw-score equality and bitwise F16 output equality; "
          "no independent partition normalization or merge"),
      "performance": {
          "paired_delta": "staged_total_ms - fused_total_ms",
          "one_sided_95pct_ucb_cap_ms_per_layer": (
              -per_layer_median_cut_ms),
          "best_row_diagnostic_cap_ms_per_layer": -per_layer_best_cut_ms,
          "required_fused_kq_saving_after_worst_aux_ms_per_layer":
              required_fused_kq_saving_per_layer_ms,
          "stage_event_times_required": ["kq_ms", "owner_ms", "total_ms"],
      },
      "stop_rule": (
          "reject before graph/plugin/model work unless bitwise equality and "
          "the paired one-sided 95% UCB both pass"),
  }
  payload = {
      "schema_version": SCHEMA,
      "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
      "workstream": WS,
      "git": git,
      "verdict": verdict,
      "required_checks_passed": required_checks_passed,
      "component_admitted": required_checks_passed,
      "graph_integration_admitted": False,
      "plugin_build_admitted": False,
      "model_worker_admitted": False,
      "gpu_worker_launched": False,
      "model_worker_launched": False,
      "geometry": {
          "full_attention_layers": full_attention_layers,
          "query_heads": query_heads,
          "kv_heads": kv_heads,
          "gqa_group": gqa_group,
          "kq_score_columns_per_kv": KQ_SCORE_COLUMNS_PER_KV,
          "head_dim": head_dim,
          "key_block": KEY_BLOCK,
          "blocks_per_kv_head": blocks_per_kv_head,
          "current_useful_groups_per_layer": current_useful_groups_per_layer,
          "staged_kq_groups_per_layer": staged_kq_groups_per_layer,
          "staged_owner_groups_per_layer": owner_groups_per_layer,
      },
      "traffic_bound": {
          "raw_score_one_way_bytes_per_layer": score_one_way_bytes_per_layer,
          "raw_score_roundtrip_bytes_per_layer":
              score_roundtrip_bytes_per_layer,
          "raw_score_roundtrip_bytes_all_layers":
              score_roundtrip_bytes_per_layer * full_attention_layers,
          "worst_extra_query_read_bytes_per_layer":
              worst_extra_query_bytes_per_layer,
          "worst_auxiliary_bytes_all_layers": auxiliary_bytes_all_layers,
          "score_roundtrip_ms_at_planning": score_ms_planning,
          "worst_auxiliary_ms_at_planning": worst_aux_ms_planning,
          "worst_auxiliary_ms_at_raw_lpddr": worst_aux_ms_raw,
          "planning_gb_s": PLANNING_GB_S,
          "raw_lpddr_gb_s": RAW_LPDDR_GB_S,
          "interpretation": (
              "charge every packed raw-score write/read and pessimistically "
              "charge every repeated query tile as DRAM; K and V remain one "
              "read each and are common to fused and staged paths"),
      },
      "kill_number": {
          "promotion_median_ms_per_token": median_cut_ms,
          "best_complete_row_ms_per_token": best_cut_ms,
          "promotion_median_ms_per_layer": per_layer_median_cut_ms,
          "best_complete_row_ms_per_layer": per_layer_best_cut_ms,
          "required_fused_kq_saving_after_worst_aux_ms":
              required_fused_kq_saving_ms,
      },
      "component_contract": component_contract,
      "checks": checks,
      "memory_stop_bytes": stop_bytes,
      "memory_samples": memory,
      "inputs": {display(path): sha256(path) for path in required},
  }
  (output / "bound.json").write_text(
      json.dumps(payload, indent=2) + "\n", encoding="utf-8")
  summary = f"""# Exact raw-F32 score-staging source bound

Verdict: **{verdict}**. Required checks:
`{str(required_checks_passed).lower()}`. No compiler or GPU/model worker ran.

Latest official M16/N8 and M8/N8 generation both remain unsupported. The
bounded alternative keeps the accepted generated M256/N16 KQ and VS packages,
but launches KQ as `{staged_kq_groups_per_layer}` useful work-groups per layer
instead of `{current_useful_groups_per_layer}`. One owner per KV head then
reads raw F32 scores in the original chronological order and performs the
unchanged online-softmax/VS recurrence. It does not normalize or merge
independent context partitions.

Preserving all sixteen independent generated-KQ columns per KV head costs
`{score_roundtrip_bytes_per_layer:,} B/layer` of raw-score write plus read, or
`{score_roundtrip_bytes_per_layer * full_attention_layers:,} B/token` over ten
layers. Pessimistically charging every repeated query tile raises auxiliary
traffic to `{auxiliary_bytes_all_layers:,} B/token`, or
`{worst_aux_ms_planning:.6f} ms` at `{PLANNING_GB_S:.0f} GB/s`. The component
therefore must save at least `{required_fused_kq_saving_ms:.6f} ms/token` of
fused KQ time before it can fund the current `{median_cut_ms:.6f}-ms` promotion
gap.

Admit exactly one standalone 128k paired component: 20 interleaved pairs,
bitwise raw-score/output equality, stage events, and a one-sided 95% UCB for
`staged - fused <= {-per_layer_median_cut_ms:.6f} ms/layer`. Reject before
graph, plugin, or model work if either the numeric or timing boundary fails.
"""
  (output / "summary.md").write_text(summary, encoding="utf-8")
  print(json.dumps({
      "artifact": display(output),
      "verdict": verdict,
      "component_admitted": required_checks_passed,
      "raw_score_roundtrip_mib_all_layers":
          score_roundtrip_bytes_per_layer * full_attention_layers / 1024**2,
      "worst_auxiliary_ms_at_planning": worst_aux_ms_planning,
      "required_fused_kq_saving_ms": required_fused_kq_saving_ms,
      "component_ucb_cap_ms_per_layer": -per_layer_median_cut_ms,
      "gpu_worker_launched": False,
  }, separators=(",", ":")))
  return 0 if required_checks_passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
