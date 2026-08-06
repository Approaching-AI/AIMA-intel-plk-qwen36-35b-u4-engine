#!/usr/bin/env python3
"""Bound a decode-wide whole-layer materialization/superkernel route.

The gate is source-only.  It intentionally overcounts graph-visible producer
writes, recurrent-state writes, attention workspace write/read traffic, and
the complete non-major event residual.  It never compiles or launches a GPU
worker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-openvino-whole-layer-traffic-bound-v0"

MODEL_ROOT = Path("/home/intel/Qwen3.6-35B-A3B-ov")
MODEL_XML = MODEL_ROOT / "openvino_language_model.xml"
MODEL_CONFIG = MODEL_ROOT / "config.json"
ATTENTION_SOURCE = (
    REPO / "engine/openvino/custom/iq36_hot_attention_single_owner.cl")
PROFILE = REPO / (
    "output/openvino-attention-phase-profile-20260715Tseq1136-"
    "dq-subgroup-32k-warm17-cleanZ/raw/32k/candidate/worker-result.json")
FC_BOUND = REPO / (
    "output/openvino-fc-micro-component-20260715Tseq1233-"
    "max-native-fused-nonzero-warm512-cleanZ/metrics.json")
LINEAR_COMPONENT = REPO / (
    "output/openvino-provider-aware-linear-component-"
    "20260715Tseq1237-fused-warm512-cleanZ/metrics.json")
FULL_ATTENTION_BOUND = REPO / (
    "output/openvino-full-attention-projection-consumer-bound-"
    "20260715Tseq1238-cleanZ/metrics.json")
ELEMENTWISE_BOUND = REPO / (
    "output/openvino-decode-elementwise-residual-bound-"
    "20260715Tseq1239-cleanZ/metrics.json")
ATTENTION_BOUND = REPO / (
    "output/openvino-attention-dense-state-traffic-bound-"
    "20260715Tseq1241-cleanZ/metrics.json")
FRONTIER = REPO / "doc/active" / WS / "frontier.json"
STATUS = REPO / "doc/active" / WS / "STATUS.md"

CONTEXT_TOKENS = 32768
PRODUCT_OUTPUT_TOKENS = 512
FP32_STRESS_BYTES = 4
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


def output_dims(node: ET.Element) -> list[list[int]]:
  output = node.find("output")
  if output is None:
    return []
  return [[int(dim.text or "-1") for dim in port.findall("dim")]
          for port in output]


def static_tail_elements(dims: list[int]) -> int:
  elements = 1
  for value in dims:
    if value > 0:
      elements *= value
  return elements


def runtime_and_fc_output_audit(
    profile: dict[str, Any],
) -> dict[str, Any]:
  root = ET.parse(MODEL_XML).getroot()
  layers = root.find("layers")
  if layers is None:
    raise ValueError("locked IR is missing layers")
  by_name = {node.attrib.get("name", ""): node for node in layers}
  executed = [
      row for row in profile.get("full_profile", [])
      if row.get("status") == "Status.EXECUTED"]
  counts = Counter(str(row.get("node_type", "")) for row in executed)
  fc_rows = [
      row for row in executed
      if row.get("node_type") == "FullyConnectedCompressed"]
  mapped = []
  missing = []
  for row in fc_rows:
    runtime_name = str(row.get("node_name", ""))
    ir_name = runtime_name.removesuffix("_fused_3FCs")
    node = by_name.get(ir_name)
    if node is None:
      missing.append(runtime_name)
      continue
    dims = output_dims(node)
    elements = static_tail_elements(dims[-1]) if dims else 0
    mapped.append({
        "runtime_name": runtime_name,
        "ir_name": ir_name,
        "output_dims": dims,
        "primary_output_elements": elements,
    })
  primary_elements = sum(row["primary_output_elements"] for row in mapped)
  fused_full_qkv = [
      row for row in mapped if row["runtime_name"].endswith("_fused_3FCs")]
  # The runtime fused node's primary port has the Q+gate 8192 elements from
  # the original q_proj. Add K and V (512 elements each) explicitly.
  fused_extra_elements = len(fused_full_qkv) * 2 * 512
  return {
      "executed_rows": len(executed),
      "executed_counts": dict(counts),
      "selected_counts": {
          key: counts[key] for key in (
              "FullyConnectedCompressed", "MOE3GemmFusedCompressed",
              "IQ36HotAttentionGQA", "GatedDeltaNet",
              "IQ36LinearConvSwish", "RMS")},
      "expected_selected_counts": {
          "FullyConnectedCompressed": 371,
          "MOE3GemmFusedCompressed": 40,
          "IQ36HotAttentionGQA": 10,
          "GatedDeltaNet": 30,
          "IQ36LinearConvSwish": 30,
          "RMS": 131,
      },
      "selected_counts_exact": {
          key: counts[key] for key in (
              "FullyConnectedCompressed", "MOE3GemmFusedCompressed",
              "IQ36HotAttentionGQA", "GatedDeltaNet",
              "IQ36LinearConvSwish", "RMS")} == {
          "FullyConnectedCompressed": 371,
          "MOE3GemmFusedCompressed": 40,
          "IQ36HotAttentionGQA": 10,
          "GatedDeltaNet": 30,
          "IQ36LinearConvSwish": 30,
          "RMS": 131,
      },
      "fc_runtime_rows": len(fc_rows),
      "fc_mapped_rows": len(mapped),
      "fc_missing_rows": missing,
      "fc_primary_output_elements": primary_elements,
      "fused_full_qkv_rows": len(fused_full_qkv),
      "fused_k_v_extra_elements": fused_extra_elements,
      "fc_all_output_elements_with_fused_k_v": (
          primary_elements + fused_extra_elements),
  }


def state_sizes(component: dict[str, Any]) -> dict[str, int]:
  row = next(
      (item for item in component.get("checks", [])
       if item.get("name") == "all_capture_input_sizes_are_exact"), {})
  observed = row.get("observed", {})
  return {
      "previous_conv_state_bytes_per_layer": int(
          observed.get("previous_conv_state", -1)),
      "gdn_state_bytes_per_layer": int(
          observed.get("initial_gdn_state", -1)),
  }


def main() -> int:
  args = parse_args()
  output = args.output.resolve()
  output.mkdir(parents=True, exist_ok=False)
  stop_bytes = int(args.memory_stop_gib * 1024**3)
  memory: list[dict[str, Any]] = []
  sample_memory("start", stop_bytes, memory)

  required = (
      MODEL_XML, MODEL_CONFIG, ATTENTION_SOURCE, PROFILE, FC_BOUND,
      LINEAR_COMPONENT, FULL_ATTENTION_BOUND, ELEMENTWISE_BOUND,
      ATTENTION_BOUND, FRONTIER, STATUS)
  missing = [str(path) for path in required if not path.is_file()]
  if missing:
    raise SystemExit("missing source-bound inputs: " + ", ".join(missing))

  git = git_state()
  config = load_json(MODEL_CONFIG)
  profile = load_json(PROFILE)
  fc = load_json(FC_BOUND)
  linear = load_json(LINEAR_COMPONENT)
  full_boundary = load_json(FULL_ATTENTION_BOUND)
  elementwise = load_json(ELEMENTWISE_BOUND)
  attention = load_json(ATTENTION_BOUND)
  frontier = load_json(FRONTIER)
  runtime = runtime_and_fc_output_audit(profile)
  states = state_sizes(linear)
  sample_memory("after-ir-and-stored-evidence", stop_bytes, memory)

  text_config = config.get("text_config", {})
  model = {
      "layers": int(text_config.get("num_hidden_layers", -1)),
      "full_attention_interval": int(
          text_config.get("full_attention_interval", -1)),
      "attention_heads": int(text_config.get("num_attention_heads", -1)),
      "kv_heads": int(text_config.get("num_key_value_heads", -1)),
      "head_dim": int(text_config.get("head_dim", -1)),
      "linear_value_heads": int(
          text_config.get("linear_num_value_heads", -1)),
      "linear_value_head_dim": int(
          text_config.get("linear_value_head_dim", -1)),
      "hidden_size": int(text_config.get("hidden_size", -1)),
      "linear_inner_size": int(
          text_config.get("linear_key_head_dim", 128)) * int(
              text_config.get("linear_num_key_heads", 32)),
  }
  full_layers = model["layers"] // model["full_attention_interval"]
  linear_layers = model["layers"] - full_layers
  model_exact = (
      model["layers"] == 40 and model["full_attention_interval"] == 4 and
      model["attention_heads"] == 16 and model["kv_heads"] == 2 and
      model["head_dim"] == 256 and model["linear_value_heads"] == 32 and
      model["linear_value_head_dim"] == 128 and
      model["hidden_size"] == 2048 and full_layers == 10 and
      linear_layers == 30)

  kill_number_ms = float(
      frontier["goal_budget"]["per_token_ms"]["remaining_cut"])
  small_tensor_gb_s = float(
      linear["component"]["state_read_write_gbps"])
  nonmajor_event_ms = float(
      elementwise["overlap_free_ceiling"]
      ["complete_nonmajor_event_residual_ms_per_token"])
  all_rms_elements = int(
      elementwise["overlap_free_ceiling"]["all_rms_decode_elements"])

  max_key_tokens = CONTEXT_TOKENS + PRODUCT_OUTPUT_TOKENS
  attention_chunks = math.ceil(max_key_tokens / 512)
  workspace_width = 2 + 8 * (2 + 256)
  bytes_by_scope = {
      # Use FP32 for every activation write even though the accepted custom
      # carrier is F16. This deliberately overstates removable traffic.
      "all_371_fc_outputs_including_lm_head_and_fused_k_v": (
          int(runtime["fc_all_output_elements_with_fused_k_v"]) *
          FP32_STRESS_BYTES),
      "all_40_fused_moe_outputs": (
          40 * model["hidden_size"] * FP32_STRESS_BYTES),
      "all_10_custom_attention_outputs": (
          full_layers * model["attention_heads"] * model["head_dim"] *
          FP32_STRESS_BYTES),
      "all_30_gdn_attention_outputs": (
          linear_layers * model["linear_value_heads"] *
          model["linear_value_head_dim"] * FP32_STRESS_BYTES),
      "all_30_linear_conv_outputs": (
          linear_layers * 8192 * FP32_STRESS_BYTES),
      "all_131_rms_outputs": all_rms_elements * FP32_STRESS_BYTES,
      # Also overcount one complete persistent-state write, although a
      # superkernel cannot delete it across decode tokens.
      "all_30_recurrent_state_writes": linear_layers * (
          states["previous_conv_state_bytes_per_layer"] +
          states["gdn_state_bytes_per_layer"]),
      # Count both workspace write and read for every product-output chunk.
      "all_10_attention_workspace_write_and_read": (
          full_layers * model["kv_heads"] * attention_chunks *
          workspace_width * 4 * 2),
      # Count packed K, duplicate dense K, V, and three more F16-equivalent
      # cold append/scale planes for each full-attention layer.
      "all_attention_state_and_append_writes": (
          full_layers * model["kv_heads"] * model["head_dim"] *
          F16_BYTES * 6),
  }
  gross_stress_bytes = sum(bytes_by_scope.values())
  gross_stress_traffic_ms = (
      gross_stress_bytes / small_tensor_gb_s / 1_000_000.0)
  duplicate_inclusive_stress_ceiling_ms = (
      gross_stress_traffic_ms + nonmajor_event_ms)
  residual_shortfall_ms = (
      kill_number_ms - duplicate_inclusive_stress_ceiling_ms)
  route_fundable = duplicate_inclusive_stress_ceiling_ms >= kill_number_ms

  source = ATTENTION_SOURCE.read_text(encoding="utf-8")
  status = STATUS.read_text(encoding="utf-8")
  closed_inputs_exact = (
      fc.get("required_checks_passed") is True and
      fc.get("route_stop_proven") is True and
      linear.get("required_checks_passed") is True and
      linear.get("verdict") ==
          "reject_provider_aware_route_before_graph_integration" and
      full_boundary.get("required_checks_passed") is True and
      full_boundary.get("component_admitted") is False and
      elementwise.get("required_checks_passed") is True and
      elementwise.get("component_admitted") is False and
      attention.get("required_checks_passed") is True and
      attention.get("component_admitted") is False)
  checks = [
      check("repository_clean_at_gate", not git["dirty"], git=git),
      check("locked_model_architecture_is_exact", model_exact, model=model),
      check("stored_runtime_major_execution_census_is_exact",
            runtime["selected_counts_exact"],
            counts=runtime["selected_counts"]),
      check("all_371_runtime_fc_outputs_map_to_locked_ir",
            runtime["fc_runtime_rows"] == 371 and
            runtime["fc_mapped_rows"] == 371 and
            not runtime["fc_missing_rows"] and
            runtime["fc_primary_output_elements"] == 915880 and
            runtime["fused_full_qkv_rows"] == 10 and
            runtime["fused_k_v_extra_elements"] == 10240,
            primary_elements=runtime["fc_primary_output_elements"],
            fused_extra_elements=runtime["fused_k_v_extra_elements"]),
      check("captured_linear_recurrent_state_sizes_are_exact",
            states == {
                "previous_conv_state_bytes_per_layer": 65536,
                "gdn_state_bytes_per_layer": 1048576}, states=states),
      check("attention_workspace_contract_is_exact",
            "workspace OUTPUT0 [B,2,G,2066]" in source and
            "decode  G = ceil(K/512)" in source and
            workspace_width == 2066 and attention_chunks == 65,
            workspace_width=workspace_width,
            product_attention_chunks=attention_chunks),
      check("all_closed_overlap_evidence_passes", closed_inputs_exact),
      check("status_registers_current_kill_and_cross_major_gate",
            "2.837" in status and "cross-major whole-layer" in status and
            abs(kill_number_ms - 2.837085) < 1e-9),
      check("duplicate_inclusive_stress_ceiling_is_below_kill_number",
            not route_fundable,
            stress_ceiling_ms_per_token=
                duplicate_inclusive_stress_ceiling_ms,
            kill_number_ms_per_token=kill_number_ms,
            residual_shortfall_ms_per_token=residual_shortfall_ms),
      check("memory_guard_never_tripped",
            all(row["available_bytes"] >= stop_bytes for row in memory),
            minimum_available_bytes=min(
                row["available_bytes"] for row in memory)),
  ]
  required_checks_passed = all(row["pass"] for row in checks)
  verdict = (
      "reject_whole_layer_materialization_superkernel_before_source"
      if required_checks_passed and not route_fundable else
      "admit_one_parameterized_whole_layer_component"
      if required_checks_passed else "inconclusive")

  metrics = {
      "schema": SCHEMA,
      "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
      "workstream": WS,
      "git": git,
      "verdict": verdict,
      "required_checks_passed": required_checks_passed,
      "component_admitted": required_checks_passed and route_fundable,
      "source_edit_admitted": False,
      "compile_admitted": False,
      "gpu_worker_launched": False,
      "long_worker_admitted": False,
      "budget": {
          "kill_number_ms_per_token": kill_number_ms,
          "small_tensor_gb_s": small_tensor_gb_s,
          "complete_nonmajor_event_residual_ms_per_token":
              nonmajor_event_ms,
          "bytes_by_scope": bytes_by_scope,
          "gross_duplicate_inclusive_bytes": gross_stress_bytes,
          "gross_traffic_ms_per_token": gross_stress_traffic_ms,
          "duplicate_inclusive_stress_ceiling_ms_per_token":
              duplicate_inclusive_stress_ceiling_ms,
          "residual_shortfall_ms_per_token": residual_shortfall_ms,
          "provider_or_level_zero_timeline_added_ms_per_token": 0.0,
          "major_kernel_algorithm_saving_added_ms_per_token": 0.0,
          "bound_rule": (
              "count every major producer output at FP32, one complete "
              "recurrent-state write, both attention-workspace directions, "
              "six F16-equivalent attention state/append planes, and the "
              "entire non-major event residual; add no non-additive provider "
              "timeline or already-closed major-kernel algorithm saving"),
          "overlap_note": (
              "the stress ceiling deliberately duplicates seq1233 FC outputs, "
              "seq1237 recurrent state, seq1238 full-attention edges, seq1239 "
              "RMS/non-major events, and seq1241 workspace; an independent "
              "overlap-free whole-layer ceiling can only be smaller"),
      },
      "model": model,
      "runtime_and_fc_output_audit": runtime,
      "state_sizes": states,
      "checks": checks,
      "memory_stop_bytes": stop_bytes,
      "memory_samples": memory,
      "inputs": {display_path(path): sha256(path) for path in required},
  }
  (output / "metrics.json").write_text(
      json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
  summary = f"""# Whole-layer materialization traffic bound

Verdict: **{verdict}**. Required evidence checks:
`{str(required_checks_passed).lower()}`. No compiler or GPU worker ran.

The stored accepted runtime census is exact: 371 compressed FC, 40 fused MoE,
10 custom attention, 30 GDN, 30 custom linear-conv, and 131 RMS executions.
All 371 FC outputs map back to the locked IR, including the ten fused Q/K/V
rows. The bound uses the exact captured conv/GDN state sizes and the product
32k/output512 maximum of `{attention_chunks}` attention workspace chunks.

This is intentionally much larger than a legal independent superkernel cut. It
charges every major output write at FP32, one complete write of all recurrent
state, both write and read of every attention workspace row, six
F16-equivalent attention state/append planes, and the entire
`{nonmajor_event_ms:.3f} ms/token` non-major event residual. It prices all
`{gross_stress_bytes:,}` bytes at the slow captured
`{small_tensor_gb_s:.6f} GB/s` state rate. It therefore duplicates every closed
seq1233/1237-1239/1241 edge rather than subtracting it.

Even this duplicate-inclusive stress ceiling is only
`{duplicate_inclusive_stress_ceiling_ms:.6f} ms/token` versus the
`{kill_number_ms:.6f}` kill-number, short by `{residual_shortfall_ms:.6f}`.
Provider/Level-Zero timelines and major-kernel algorithm savings are not added:
they are non-additive or already closed. The legal overlap-free ceiling is
strictly smaller. Source, compile, graph integration, 32k, ABBA, and output512
are not admitted for this route.
"""
  (output / "summary.md").write_text(summary, encoding="utf-8")
  print(json.dumps({
      "artifact": display_path(output),
      "verdict": verdict,
      "gross_stress_bytes": gross_stress_bytes,
      "gross_traffic_ms": gross_stress_traffic_ms,
      "stress_ceiling_ms": duplicate_inclusive_stress_ceiling_ms,
      "shortfall_ms": residual_shortfall_ms,
      "gpu_worker_launched": False,
  }, separators=(",", ":")))
  return 0 if required_checks_passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
