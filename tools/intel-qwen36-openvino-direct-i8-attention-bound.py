#!/usr/bin/env python3
"""Gate one direct block32-I8 attention component from stored evidence.

This source-only gate distinguishes the closed scalar cold-state path from a
new register-tiled I8+F16-scale carrier.  It derives the exact 32k one-layer
cap before permitting any source, compile, or GPU component worker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-openvino-direct-i8-attention-bound-v0"

MODEL_CONFIG = Path("/home/intel/Qwen3.6-35B-A3B-ov/config.json")
ATTENTION_SOURCE = (
    REPO / "engine/openvino/custom/iq36_hot_attention_single_owner.cl")
ATTENTION_HELPERS = (
    REPO / "engine/openvino/custom/iq36_hot_attention_tiled_helpers.cl")
NATIVE_I8_SOURCE = (
    REPO / "engine/gpu/opencl/compressed_gqa_i8_kv_decode.cl")
DPAS_SOURCE = REPO / "engine/gpu/opencl/q6_splitplane_dpas.cl"
DENSE_BOUND = REPO / (
    "output/openvino-attention-dense-state-traffic-bound-"
    "20260715Tseq1241-cleanZ/metrics.json")
EXACT_BUCKET = REPO / (
    "output/openvino-exact-bucket-preflight-"
    "20260715Tseq1234b-cleanZ/metrics.json")
NATIVE_I8 = REPO / (
    "output/compressed-gqa-i8-kv-decode-"
    "20260713Tseq783-ci-confirm-cleanZ/result.json")
ACCEPTED = REPO / "doc/active" / WS / "accepted-cuts.json"
FRONTIER = REPO / "doc/active" / WS / "frontier.json"
STATUS = REPO / "doc/active" / WS / "STATUS.md"

CONTEXT_TOKENS = 32768
HOT_TOKENS = 8192
QUANT_GROUP = 32
F16_BYTES = 2
I8_BYTES = 1


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
    return str(path.relative_to(REPO))
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
      ["git", "rev-parse", "HEAD"], cwd=REPO, text=True,
      capture_output=True, check=True).stdout.strip()
  status = subprocess.run(
      ["git", "status", "--porcelain"], cwd=REPO, text=True,
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
      MODEL_CONFIG, ATTENTION_SOURCE, ATTENTION_HELPERS, NATIVE_I8_SOURCE,
      DPAS_SOURCE, DENSE_BOUND, EXACT_BUCKET, NATIVE_I8, ACCEPTED, FRONTIER,
      STATUS)
  missing = [str(path) for path in required if not path.is_file()]
  if missing:
    raise SystemExit("missing source-bound inputs: " + ", ".join(missing))

  git = git_state()
  config = load_json(MODEL_CONFIG)
  dense = load_json(DENSE_BOUND)
  exact_bucket = load_json(EXACT_BUCKET)
  native = load_json(NATIVE_I8)
  accepted = load_json(ACCEPTED)
  frontier = load_json(FRONTIER)
  attention_source = ATTENTION_SOURCE.read_text(encoding="utf-8")
  helper_source = ATTENTION_HELPERS.read_text(encoding="utf-8")
  native_source = NATIVE_I8_SOURCE.read_text(encoding="utf-8")
  dpas_source = DPAS_SOURCE.read_text(encoding="utf-8")
  status = STATUS.read_text(encoding="utf-8")
  sample_memory("after-stored-evidence", stop_bytes, memory)

  text_config = config.get("text_config", {})
  layers = int(text_config.get("num_hidden_layers", -1))
  interval = int(text_config.get("full_attention_interval", -1))
  q_heads = int(text_config.get("num_attention_heads", -1))
  kv_heads = int(text_config.get("num_key_value_heads", -1))
  head_dim = int(text_config.get("head_dim", -1))
  full_layers = layers // interval if interval > 0 else -1
  model_exact = (
      layers == 40 and interval == 4 and q_heads == 16 and
      kv_heads == 2 and head_dim == 256 and full_layers == 10)

  registered_attention_ms = float(
      dense["budget"]["registered_attention_ms_per_token"])
  kill_number_ms = float(
      frontier["goal_budget"]["per_token_ms"]["remaining_cut"])
  target_attention_ms = registered_attention_ms - kill_number_ms
  planning_gb_s = float(dense["budget"]["planning_gb_s"])

  cold_tokens = CONTEXT_TOKENS - HOT_TOKENS
  scale_groups = head_dim // QUANT_GROUP
  dense_hot_bytes = (
      full_layers * 2 * kv_heads * HOT_TOKENS * head_dim * F16_BYTES)
  cold_value_bytes = (
      full_layers * 2 * kv_heads * cold_tokens * head_dim * I8_BYTES)
  cold_scale_bytes = (
      full_layers * 2 * kv_heads * cold_tokens * scale_groups * F16_BYTES)
  direct_state_bytes = dense_hot_bytes + cold_value_bytes + cold_scale_bytes
  dense_state_bytes = int(
      dense["budget"]["mandatory_dense_k_plus_v_bytes_per_token"])
  state_reduction_bytes = dense_state_bytes - direct_state_bytes
  state_ms_at_planning = direct_state_bytes / planning_gb_s / 1_000_000.0
  traffic_only_nonstate_margin_ms = target_attention_ms - state_ms_at_planning
  state_only_route_fundable = traffic_only_nonstate_margin_ms > 0.0
  required_state_gb_s = direct_state_bytes / target_attention_ms / 1_000_000.0
  per_layer_state_bytes = direct_state_bytes / full_layers
  per_layer_complete_cap_ms = target_attention_ms / full_layers

  native_result = native.get("result", {})
  native_inference = native.get("performance_inference", {})
  native_bytes = int(native_result.get("compressed_kv_bytes", -1))
  native_partial_ms = float(
      native_result.get("confirm", {}).get("partial_ms", -1.0))
  native_total_ucb_ms = float(
      native_inference.get("upper_confidence_bound_ms", -1.0))
  native_partial_effective_gb_s = (
      native_bytes / native_partial_ms / 1_000_000.0)
  native_complete_effective_gb_s = (
      native_bytes / native_total_ucb_ms / 1_000_000.0)

  carrier_note = next(
      (row.get("note", "") for row in accepted.get("accepted", [])
       if row.get("id") == "level_zero_real_q4_q6_paired_carriers"), "")
  physical_carrier_gb_s = 106.525
  direct_component_has_headroom = (
      required_state_gb_s < physical_carrier_gb_s < planning_gb_s)

  matched = exact_bucket["capacity_projection"]["matched_phase_context"]
  scalar_cold_ms = float(matched["cold_i8_decode_ms"])
  dense_f16_ms = float(matched["dense_f16_decode_ms"])
  scalar_cold_regression_ms = scalar_cold_ms - dense_f16_ms

  source_contract = {
      "logical_hot_window_8192": "#define IQ36_HOT_WINDOW 8192U" in helper_source,
      "block32_scale_selection": "const uint scale_byte = (dim >> 5) << 1;"
          in helper_source,
      "key_vector_dequant_to_f16":
          "const half16 values = convert_half16(quantized) * as_half(scale_bits);"
          in helper_source,
      "value_scalar_dequant":
          "return convert_half(cold_value[value_index]) * as_half(scale_bits);"
          in helper_source,
      "dense_overlap_suppresses_logical_cold":
          "min(cold_tokens, dense_history_begin)" in attention_source,
      "f16_xmx_score_and_value":
          attention_source.count(
              "intel_sub_group_f16_f16_matrix_mad_k16(") >= 3,
      "native_i8_uses_local_f32_reconstruction":
          "__local float local_k[IQ36_HEAD_DIM];" in native_source and
          "__local float local_v[IQ36_HEAD_DIM];" in native_source,
      "target_i8_u8_dpas_intrinsic_available":
          "intel_sub_group_i8_u8_matrix_mad_k32" in dpas_source,
  }

  component_contract = {
      "context_tokens": CONTEXT_TOKENS,
      "full_attention_layers_extrapolated": full_layers,
      "component_layers": 1,
      "logical_hot_tokens": HOT_TOKENS,
      "cold_tokens": cold_tokens,
      "kv_heads": kv_heads,
      "head_dim": head_dim,
      "quant_group": QUANT_GROUP,
      "one_sided_95pct_ucb_cap_ms": per_layer_complete_cap_ms,
      "minimum_samples": 20,
      "numeric_cosine_min": 0.999,
      "numeric_relative_l2_max": 0.002,
      "required_algorithm": (
          "consume block32 signed-I8 plus logical-F16 scales from a "
          "DPAS-friendly K/V tile; vector/register dequant feeding F16 DPAS "
          "or signed-integer DPAS is allowed, scalar local-F32 reconstruction "
          "is not"),
      "timed_scope": (
          "current-token append/quantization, QK, softmax, PV, partial "
          "workspace, and final reduction"),
      "graph_integration_admitted": False,
      "long_worker_admitted": False,
  }

  checks = [
      check("repository_clean_at_gate", not git["dirty"], git=git),
      check("locked_model_architecture_is_exact", model_exact,
            layers=layers, interval=interval, q_heads=q_heads,
            kv_heads=kv_heads, head_dim=head_dim),
      check("accepted_direct_state_representation_source_is_exact",
            all(source_contract.values()), source_contract=source_contract),
      check("dense_attention_bound_and_current_budget_are_exact",
            dense.get("required_checks_passed") is True and
            dense.get("component_admitted") is False and
            abs(registered_attention_ms - 8.456) < 1e-9 and
            abs(kill_number_ms - 2.837085) < 1e-9),
      check("closed_scalar_cold_path_regresses_at_matched_32k_scope",
            exact_bucket.get("required_evidence_checks_passed") is True and
            abs(scalar_cold_ms - 21.908434) < 1e-9 and
            abs(dense_f16_ms - 13.718745) < 1e-9 and
            scalar_cold_regression_ms > 8.0,
            scalar_cold_ms=scalar_cold_ms, dense_f16_ms=dense_f16_ms,
            regression_ms=scalar_cold_regression_ms),
      check("clean_native_i8_component_proves_codec_numeric_semantics",
            native.get("required_checks_passed") is True and
            native_result.get("kv_dtype") == "int8_block32_fp16_scale" and
            native_result.get("numeric_pass") is True and
            float(native_result.get("output_cosine", 0.0)) >= 0.999 and
            float(native_result.get("output_relative_l2", 1.0)) <= 0.002 and
            native_inference.get("sample_count_pass") is True),
      check("scalar_native_carrier_is_not_reused_as_direct_dpas_evidence",
            native_partial_effective_gb_s < required_state_gb_s and
            native_complete_effective_gb_s < required_state_gb_s,
            native_partial_effective_gb_s=native_partial_effective_gb_s,
            native_complete_effective_gb_s=native_complete_effective_gb_s,
            required_state_only_gb_s=required_state_gb_s),
      check("direct_i8_state_byte_ceiling_can_fund_one_component",
            state_only_route_fundable and direct_component_has_headroom and
            direct_state_bytes == 435159040 and
            state_reduction_bytes == 235929600,
            dense_state_bytes=dense_state_bytes,
            direct_state_bytes=direct_state_bytes,
            state_reduction_bytes=state_reduction_bytes,
            state_ms_at_planning=state_ms_at_planning,
            target_attention_ms=target_attention_ms,
            nonstate_margin_ms=traffic_only_nonstate_margin_ms,
            required_state_only_gb_s=required_state_gb_s,
            physical_carrier_gb_s=physical_carrier_gb_s,
            planning_gb_s=planning_gb_s),
      check("physical_carrier_floor_is_registered",
            "106.525 GB/s" in carrier_note,
            accepted_cut="level_zero_real_q4_q6_paired_carriers"),
      check("status_selects_direct_i8_source_bound",
            "direct-I8 K/V attention" in status and
            ">2.837 ms/token" in status),
      check("memory_guard_never_tripped",
            all(row["available_bytes"] >= stop_bytes for row in memory),
            minimum_available_bytes=min(
                row["available_bytes"] for row in memory)),
  ]
  required_checks_passed = all(row["pass"] for row in checks)
  component_admitted = required_checks_passed and state_only_route_fundable
  verdict = (
      "admit_one_direct_i8_attention_component"
      if component_admitted else "reject_direct_i8_attention_before_source"
      if required_checks_passed else "inconclusive")

  metrics = {
      "schema": SCHEMA,
      "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
      "workstream": WS,
      "git": git,
      "verdict": verdict,
      "required_checks_passed": required_checks_passed,
      "component_admitted": component_admitted,
      "component_source_admitted": component_admitted,
      "component_compile_admitted": component_admitted,
      "graph_source_edit_admitted": False,
      "gpu_worker_launched": False,
      "long_worker_admitted": False,
      "budget": {
          "registered_attention_ms_per_token": registered_attention_ms,
          "kill_number_ms_per_token": kill_number_ms,
          "target_attention_ms_per_token": target_attention_ms,
          "dense_state_bytes_per_token": dense_state_bytes,
          "logical_hot_f16_bytes_per_token": dense_hot_bytes,
          "cold_i8_value_bytes_per_token": cold_value_bytes,
          "cold_f16_scale_bytes_per_token": cold_scale_bytes,
          "direct_state_bytes_per_token": direct_state_bytes,
          "state_reduction_bytes_per_token": state_reduction_bytes,
          "state_reduction_ratio": state_reduction_bytes / dense_state_bytes,
          "planning_gb_s": planning_gb_s,
          "state_ms_at_planning": state_ms_at_planning,
          "traffic_only_nonstate_margin_ms_per_token":
              traffic_only_nonstate_margin_ms,
          "required_state_only_gb_s": required_state_gb_s,
          "per_layer_state_bytes": per_layer_state_bytes,
          "per_layer_complete_ucb_cap_ms": per_layer_complete_cap_ms,
          "bound_rule": (
              "count one read of logical-hot8192 F16 K/V plus older block32 "
              "I8 K/V and one F16 scale per 32 values across all ten layers; "
              "the traffic-only line makes every arithmetic, workspace, "
              "synchronization, append, launch, and output cost free"),
      },
      "closed_scalar_baseline": {
          "openvino_cold_i8_ms_matched_nine_layers": scalar_cold_ms,
          "openvino_dense_f16_ms_matched_nine_layers": dense_f16_ms,
          "regression_ms": scalar_cold_regression_ms,
          "native_i8_partial_effective_gb_s": native_partial_effective_gb_s,
          "native_i8_complete_effective_gb_s": native_complete_effective_gb_s,
          "disposition": (
              "codec/numeric evidence only; local-F32 scalar reconstruction "
              "does not satisfy the admitted direct-DPAS component contract"),
      },
      "component_contract": component_contract,
      "source_contract": source_contract,
      "checks": checks,
      "memory_stop_bytes": stop_bytes,
      "memory_samples": memory,
      "inputs": {display_path(path): sha256(path) for path in required},
  }
  (output / "metrics.json").write_text(
      json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
  summary = f"""# Direct block32-I8 attention source bound

Verdict: **{verdict}**. Required evidence checks:
`{str(required_checks_passed).lower()}`. No compiler or GPU worker ran.

At 32k, logical-hot8192 F16 plus 24,576 cold block32-I8 tokens requires
`{direct_state_bytes:,} B/token` across ten full-attention layers, down
`{state_reduction_bytes:,} B` from dense F16. At the registered
`{planning_gb_s:.0f} GB/s` planning line this is
`{state_ms_at_planning:.6f} ms/token`, leaving
`{traffic_only_nonstate_margin_ms:.6f} ms/token` for all other attention work
inside the `{target_attention_ms:.6f}` target. The state-only requirement is
`{required_state_gb_s:.3f} GB/s`, below the clean `106.525 GB/s` packed carrier
floor and the planning line. This admits one component, not integration.

The existing cold path remains closed: its matched nine-layer row is
`{scalar_cold_ms:.6f}` versus `{dense_f16_ms:.6f} ms` for dense F16. The clean
native I8 component proves codec numerics but reconstructs K/V in local F32 and
reaches only `{native_partial_effective_gb_s:.3f} GB/s` on its partial event;
it is not direct-DPAS evidence.

The only admitted work is one 32k, one-layer, 20-sample component with a
one-sided 95% UCB at or below `{per_layer_complete_cap_ms:.6f} ms`. It must time
append/quantization, QK, softmax, PV, workspace, and final reduction, pass
cosine `>=0.999` and relative L2 `<=0.002`, and consume block32 I8 plus F16
scales from a DPAS-friendly tile without scalar local-F32 reconstruction.
Graph integration, a full-model worker, 32k product row, and long ladder remain
blocked.
"""
  (output / "summary.md").write_text(summary, encoding="utf-8")
  print(json.dumps({
      "artifact": display_path(output),
      "verdict": verdict,
      "direct_state_bytes": direct_state_bytes,
      "state_ms_at_planning": state_ms_at_planning,
      "nonstate_margin_ms": traffic_only_nonstate_margin_ms,
      "per_layer_component_cap_ms": per_layer_complete_cap_ms,
      "gpu_worker_launched": False,
  }, separators=(",", ":")))
  return 0 if required_checks_passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
