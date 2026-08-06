#!/usr/bin/env python3
"""Bound direct-I8 integration against the exact fixed-state OpenVINO ABI."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
GRAPH = ROOT / "tools/intel_qwen36_openvino_hot_cold_attention.py"
PRODUCT = ROOT / "tools/intel-qwen36-openvino-hot-cold-product-gate.py"
ATTENTION = ROOT / "engine/openvino/custom/iq36_hot_attention_single_owner.cl"
HELPERS = ROOT / "engine/openvino/custom/iq36_hot_attention_tiled_helpers.cl"
PREFILL = ROOT / "engine/openvino/custom/iq36_prefill_attention_tiled.cl"
COMPONENT = ROOT / (
    "output/openvino-direct-i8-attention-component-"
    "20260715Tseq1245-cleanZ/result.json")
SOURCE_BOUND = ROOT / (
    "output/openvino-direct-i8-attention-bound-"
    "20260715Tseq1244-cleanZ/metrics.json")
FRONTIER = ROOT / "doc/active" / WS / "frontier.json"
STATUS = ROOT / "doc/active" / WS / "STATUS.md"
SCHEMA = "intel-qwen36-openvino-direct-i8-integration-abi-bound-v0"

CONTEXT_TOKENS = 32768
HOT_WINDOW = 8192
PREFILL_CHUNK = 8192
PREFILL_RING = HOT_WINDOW + PREFILL_CHUNK
SINK_TOKENS = 1
FULL_LAYERS = 10
KV_HEADS = 2
HEAD_DIM = 256
QUANT_GROUP = 32
F16_BYTES = 2


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


def display_path(path: Path) -> str:
  try:
    return str(path.relative_to(ROOT))
  except ValueError:
    return str(path)


def available_memory_bytes() -> int:
  for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
    if line.startswith("MemAvailable:"):
      return int(line.split()[1]) * 1024
  raise RuntimeError("MemAvailable is missing")


def sample_memory(
    label: str, stop_bytes: int, rows: list[dict[str, Any]],
) -> None:
  available = available_memory_bytes()
  rows.append({"label": label, "available_bytes": available})
  if available < stop_bytes:
    raise RuntimeError(
        f"memory stop at {label}: {available} < {stop_bytes} bytes")


def git_state() -> dict[str, Any]:
  commit = subprocess.run(
      ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
      capture_output=True, check=True).stdout.strip()
  status = subprocess.run(
      ["git", "status", "--porcelain"], cwd=ROOT, text=True,
      capture_output=True, check=True).stdout.strip()
  return {"commit": commit, "dirty": bool(status), "status": status}


def check(name: str, passed: bool, **details: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **details}


def main() -> int:
  args = parse_args()
  output = args.output.resolve()
  output.mkdir(parents=True, exist_ok=False)
  stop_bytes = int(args.memory_stop_gib * 1024**3)
  memory: list[dict[str, Any]] = []
  sample_memory("start", stop_bytes, memory)

  required = (
      GRAPH, PRODUCT, ATTENTION, HELPERS, PREFILL, COMPONENT, SOURCE_BOUND,
      FRONTIER, STATUS)
  missing = [str(path) for path in required if not path.is_file()]
  if missing:
    raise SystemExit("missing integration-bound inputs: " + ", ".join(missing))
  git = git_state()
  graph = GRAPH.read_text(encoding="utf-8")
  product = PRODUCT.read_text(encoding="utf-8")
  attention = ATTENTION.read_text(encoding="utf-8")
  helpers = HELPERS.read_text(encoding="utf-8")
  prefill = PREFILL.read_text(encoding="utf-8")
  component = load_json(COMPONENT)
  source_bound = load_json(SOURCE_BOUND)
  frontier = load_json(FRONTIER)
  status = STATUS.read_text(encoding="utf-8")
  sample_memory("after-source-audit", stop_bytes, memory)

  graph_contract = {
      "logical_hot8192": "HOT_WINDOW = 8192" in graph,
      "continuation_chunk8192": "PREFILL_CHUNK_TOKENS = 8192" in graph,
      "minimum_ring_is_hot_plus_chunk":
          "RING_CAPACITY = HOT_WINDOW + PREFILL_CHUNK_TOKENS" in graph,
      "smaller_ring_rejected_by_constructor":
          "prefill_history_capacity < RING_CAPACITY" in graph,
      "fixed_cold_capacity_is_byte_stable":
          graph.count("fixed_cold_capacity + 1") >= 4,
      "fixed_cold_reads_are_direct":
          "effective_cold_reads = reads[2:]" in graph,
      "fixed_logical_length_is_position_derived":
          "past_tokens," in graph
          and "ov.opset13.constant(np.array(HOT_WINDOW" in graph,
      "dynamic_concat_assign_is_excluded_for_fixed":
          "if fixed_cold_capacity is None:" in graph
          and "present = ov.opset13.concat(" in graph,
      "product_currently_overallocates_hot_ring":
          "prefill_history_capacity=max(2 * FROZEN_CHUNK_TOKENS, bucket)"
          in product,
  }
  source_contract = {
      "single_custom_op_owns_all_state_planes":
          "The last work-group is the sole decode state owner" in attention,
      "fixed_cold_state_is_written_in_place":
          (attention.count("if (fixed_cold_state)") +
           prefill.count("if (fixed_cold_state)")) >= 4,
      "hot_k_is_already_block16_xmx_packed":
          "intel_sub_group_block_read8" in attention
          and "IQ36_KEY_TILE_TOKENS 16U" in helpers,
      "hot_v_is_already_f16_dpas":
          "iq36_block2d_load_f16_16x8" in attention
          and attention.count("intel_sub_group_f16_f16_matrix_mad_k16(") >= 3,
      "current_cold_v_path_is_scalar_and_must_be_replaced":
          "inline half iq36_cold_value_element" in helpers,
      "prefill_and_decode_share_state_helpers":
          "iq36_cold_key_fragment" in prefill
          and "iq36_partial_load_value" in prefill
          and "iq36_partial_load_value" in helpers,
  }
  component_result = component.get("result", {})
  component_inference = component.get("performance_inference", {})
  component_exact = (
      component.get("required_checks_passed") is True
      and component.get("component_promoted") is True
      and component_result.get("state_bytes") == 43515904
      and component_result.get("timed_scope") ==
          "append_quantize_qk_softmax_pv_workspace_reduce"
      and component_inference.get("rate_pass") is True
      and abs(float(component_inference.get(
          "upper_confidence_bound_ms", -1.0)) - 0.355312) < 1e-12)

  past_tokens = CONTEXT_TOKENS - 1
  logical_cold_tokens = CONTEXT_TOKENS - HOT_WINDOW
  dense_history_begin = max(SINK_TOKENS, past_tokens - PREFILL_RING)
  attended_cold_tokens = min(logical_cold_tokens, dense_history_begin)
  attended_hot_tokens = CONTEXT_TOKENS - attended_cold_tokens
  hot_bytes = (
      FULL_LAYERS * 2 * KV_HEADS * attended_hot_tokens * HEAD_DIM *
      F16_BYTES)
  cold_value_bytes = (
      FULL_LAYERS * 2 * KV_HEADS * attended_cold_tokens * HEAD_DIM)
  cold_scale_bytes = (
      FULL_LAYERS * 2 * KV_HEADS * attended_cold_tokens *
      (HEAD_DIM // QUANT_GROUP) * F16_BYTES)
  integration_state_bytes = hot_bytes + cold_value_bytes + cold_scale_bytes
  component_all_layer_state_bytes = int(
      source_bound["budget"]["direct_state_bytes_per_token"])
  state_ratio = integration_state_bytes / component_all_layer_state_bytes
  component_ucb_ms = float(
      component_inference["upper_confidence_bound_ms"])
  scaled_one_layer_ucb_ms = component_ucb_ms * state_ratio
  scaled_all_layer_ucb_ms = scaled_one_layer_ucb_ms * FULL_LAYERS
  registered_attention_ms = float(
      source_bound["budget"]["registered_attention_ms_per_token"])
  kill_number_ms = float(
      frontier["goal_budget"]["per_token_ms"]["remaining_cut"])
  scaled_saving_ms = registered_attention_ms - scaled_all_layer_ucb_ms
  scaled_margin_over_kill_ms = scaled_saving_ms - kill_number_ms
  planning_gb_s = float(source_bound["budget"]["planning_gb_s"])
  target_attention_ms = registered_attention_ms - kill_number_ms
  state_ms_at_planning = integration_state_bytes / planning_gb_s / 1_000_000.0
  planning_nonstate_margin_ms = target_attention_ms - state_ms_at_planning
  required_state_gb_s = (
      integration_state_bytes / target_attention_ms / 1_000_000.0)
  source_edit_fundable = (
      scaled_margin_over_kill_ms > 0.0
      and planning_nonstate_margin_ms > 0.0
      and required_state_gb_s < 106.525)

  design_contract = {
      "scope": "fixed-capacity product state only; dynamic cold states unchanged",
      "prefill_ring_capacity": PREFILL_RING,
      "cold_k_layout": "sentinel256_then_token16_block32_packed_i8",
      "cold_v_layout": "sentinel256_then_dimension_major_i8",
      "cold_scale_layout": "sentinel16_then_group_major_fp16",
      "hot_k_layout": "retain accepted token16_dim2_packed_f16",
      "hot_v_layout": "retain accepted token-major F16 block2D carrier",
      "steady_conversion_dispatches": 0,
      "required_source_changes": [
          "reduce product prefill_history_capacity from bucket to RING_CAPACITY",
          "write and read fixed cold K/V/scales in direct DPAS-friendly tiles",
          "vectorize complete cold-V token16 fragments before F16 DPAS",
          "teach state evidence decoders the fixed physical layout",
      ],
      "preserve": [
          "dynamic-state ABI", "logical block32 codec", "all state owners",
          "prefill and decode arithmetic", "stock worker isolation",
      ],
      "short_correctness_worker_admitted": False,
      "full_model_worker_admitted": False,
      "long_worker_admitted": False,
  }

  checks = [
      check("repository_clean_at_gate", not git["dirty"], git=git),
      check("fixed_product_graph_abi_is_exact", all(graph_contract.values()),
            graph_contract=graph_contract),
      check("accepted_custom_state_ownership_is_exact",
            all(source_contract.values()), source_contract=source_contract),
      check("clean_direct_i8_component_is_exact", component_exact),
      check("required_prefill_ring_attention_split_is_exact",
            past_tokens == 32767
            and logical_cold_tokens == 24576
            and dense_history_begin == 16383
            and attended_cold_tokens == 16383
            and attended_hot_tokens == 16385,
            past_tokens=past_tokens,
            logical_cold_tokens=logical_cold_tokens,
            dense_history_begin=dense_history_begin,
            attended_cold_tokens=attended_cold_tokens,
            attended_hot_tokens=attended_hot_tokens),
      check("byte_compatible_fixed_state_bound_is_fundable",
            source_edit_fundable
            and integration_state_bytes == 513811840,
            integration_state_bytes=integration_state_bytes,
            component_state_bytes=component_all_layer_state_bytes,
            state_ratio=state_ratio,
            scaled_one_layer_ucb_ms=scaled_one_layer_ucb_ms,
            scaled_all_layer_ucb_ms=scaled_all_layer_ucb_ms,
            scaled_saving_ms=scaled_saving_ms,
            scaled_margin_over_kill_ms=scaled_margin_over_kill_ms,
            state_ms_at_planning=state_ms_at_planning,
            planning_nonstate_margin_ms=planning_nonstate_margin_ms,
            required_state_gb_s=required_state_gb_s),
      check("status_selects_exact_abi_bound",
            "state-ABI/integration bound" in status
            and "2.837 ms/token" in status),
      check("memory_guard_never_tripped",
            all(row["available_bytes"] >= stop_bytes for row in memory),
            minimum_available_bytes=min(
                row["available_bytes"] for row in memory)),
  ]
  required_checks_passed = all(row["pass"] for row in checks)
  source_edit_admitted = required_checks_passed and source_edit_fundable
  verdict = (
      "admit_direct_i8_openvino_integration_source"
      if source_edit_admitted else
      "reject_direct_i8_openvino_integration_before_source"
      if required_checks_passed else "inconclusive")
  metrics = {
      "schema": SCHEMA,
      "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
      "workstream": WS,
      "git": git,
      "verdict": verdict,
      "required_checks_passed": required_checks_passed,
      "source_edit_admitted": source_edit_admitted,
      "compile_admitted": source_edit_admitted,
      "gpu_worker_launched": False,
      "short_correctness_worker_admitted": False,
      "full_model_worker_admitted": False,
      "long_worker_admitted": False,
      "budget": {
          "registered_attention_ms_per_token": registered_attention_ms,
          "kill_number_ms_per_token": kill_number_ms,
          "target_attention_ms_per_token": target_attention_ms,
          "past_tokens": past_tokens,
          "logical_cold_tokens": logical_cold_tokens,
          "attended_cold_tokens": attended_cold_tokens,
          "attended_hot_tokens": attended_hot_tokens,
          "hot_f16_bytes_per_token": hot_bytes,
          "cold_i8_value_bytes_per_token": cold_value_bytes,
          "cold_f16_scale_bytes_per_token": cold_scale_bytes,
          "integration_state_bytes_per_token": integration_state_bytes,
          "component_state_bytes_per_token": component_all_layer_state_bytes,
          "state_ratio_vs_component": state_ratio,
          "component_one_layer_ucb_ms": component_ucb_ms,
          "scaled_one_layer_ucb_ms": scaled_one_layer_ucb_ms,
          "scaled_all_layer_ucb_ms": scaled_all_layer_ucb_ms,
          "scaled_saving_ms": scaled_saving_ms,
          "scaled_margin_over_kill_ms": scaled_margin_over_kill_ms,
          "planning_gb_s": planning_gb_s,
          "state_ms_at_planning": state_ms_at_planning,
          "planning_nonstate_margin_ms": planning_nonstate_margin_ms,
          "required_state_gb_s": required_state_gb_s,
          "bound_rule": (
              "retain the mandatory 16k continuation-prefill ring, charge "
              "every attended hot F16 and cold block32-I8 K/V byte, scale the "
              "clean complete component UCB linearly by exact state bytes, "
              "and separately require the 115-GB/s planning floor to fit"),
      },
      "design_contract": design_contract,
      "checks": checks,
      "memory_stop_bytes": stop_bytes,
      "memory_samples": memory,
      "inputs": {display_path(path): sha256(path) for path in required},
  }
  (output / "metrics.json").write_text(
      json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
  summary = f"""# Direct-I8 OpenVINO integration ABI bound

Verdict: **{verdict}**. Required checks:
`{str(required_checks_passed).lower()}`. No compiler or GPU worker ran.

The real continuation-prefill ABI requires a 16k ring. At the 32k component
boundary this makes `{attended_cold_tokens:,}` tokens cold-I8 and
`{attended_hot_tokens:,}` tokens hot-F16, or `{integration_state_bytes:,}`
bytes/token across ten layers. Scaling the clean complete component UCB by the
exact `{state_ratio:.6f}x` state ratio gives `{scaled_all_layer_ucb_ms:.6f}
ms/token`, an attainable saving of `{scaled_saving_ms:.6f}` and margin
`{scaled_margin_over_kill_ms:.6f}` above the kill-number. Independently, the
115-GB/s planning line costs `{state_ms_at_planning:.6f} ms/token` and leaves
`{planning_nonstate_margin_ms:.6f}` for non-state work.

Fixed product buffers are byte-compatible and custom-op-owned in place, so the
source design adds zero steady conversion dispatches. Source/compile is admitted
only for that fixed-state design. Dynamic state, graph/full-model workers, long
rows, ABBA, and product claims remain blocked.
"""
  (output / "summary.md").write_text(summary, encoding="utf-8")
  print(json.dumps({
      "artifact": display_path(output),
      "verdict": verdict,
      "integration_state_bytes": integration_state_bytes,
      "scaled_all_layer_ucb_ms": scaled_all_layer_ucb_ms,
      "scaled_margin_over_kill_ms": scaled_margin_over_kill_ms,
      "gpu_worker_launched": False,
  }, separators=(",", ":")))
  return 0 if required_checks_passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
