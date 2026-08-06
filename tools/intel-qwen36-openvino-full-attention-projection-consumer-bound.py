#!/usr/bin/env python3
"""Bound the decode full-attention projection-to-consumer boundary.

This is deliberately source-only.  It audits the ten locked Q/gate/K/V
projection chains, the accepted custom-attention splice, and stored provider
evidence.  It never compiles a model or launches a GPU worker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-openvino-full-attention-projection-consumer-bound-v0"

MODEL_XML = Path("/home/intel/Qwen3.6-35B-A3B-ov/openvino_language_model.xml")
GRAPH_SOURCE = REPO / "tools/intel_qwen36_openvino_hot_cold_attention.py"
STATUS = REPO / "doc/active" / WS / "STATUS.md"
FRONTIER = REPO / "doc/active" / WS / "frontier.json"
REJECTED = REPO / "doc/active" / WS / "rejected-routes.json"
FC_COMPONENT = REPO / (
    "output/openvino-fc-micro-component-20260715Tseq1233-"
    "max-native-fused-nonzero-warm512-cleanZ/metrics.json")
PROFILE_RESULT = REPO / (
    "output/openvino-attention-phase-profile-20260715Tseq1136-"
    "dq-subgroup-32k-warm17-cleanZ/raw/32k/candidate/worker-result.json")
EXACT_BUCKET = REPO / (
    "output/openvino-exact-bucket-preflight-20260715Tseq1234b-cleanZ/"
    "metrics.json")
HOST_PROFILES = (
    REPO / (
        "output/openvino-hot-cold-product-20260715Tseq1179-"
        "l0-ze-stream-profile-8k-o8-dirtyZ/raw/sentinel_008k/"
        "correctness/candidate/worker.stderr"),
    REPO / (
        "output/openvino-hot-cold-product-20260715Tseq1180-"
        "l0-sync-top-profile-8k-o8-dirtyZ/raw/sentinel_008k/"
        "correctness/candidate/worker.stderr"),
)

FULL_ATTENTION_LAYERS = tuple(range(3, 40, 4))
Q_PROJECTION = 8192
Q_CURRENT = 16 * 256
Q_GATE = 16 * 256
KV_CURRENT = 2 * 256
ROTARY_Q = 16 * 64
ROTARY_K = 2 * 64
F16_BYTES = 2

# Corrected, complete executed-event census registered by STATUS.  These
# buckets are used only as an event-side ceiling.  Level Zero timings are not
# added to them because the provider route proved those timelines non-additive.
EVENT_TOTAL_MS = 24.129
EVENT_FC_MS = 13.375
EVENT_ATTENTION_MS = 8.456
EVENT_GDN_MS = 1.319
EVENT_RMS_MS = 0.358
EVENT_LINEAR_CONV_MS = 0.193


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


def layer_name(layer: int, suffix: str) -> str:
  return (
      f"__module.model.model.language_model.layers.{layer}.self_attn/"
      f"{suffix}")


def member_name(layer: int, suffix: str) -> str:
  return (
      f"__module.model.model.language_model.layers.{layer}.self_attn."
      f"{suffix}")


def output_dims(node: ET.Element) -> list[list[int]]:
  output = node.find("output")
  if output is None:
    return []
  return [[int(dim.text or "-1") for dim in port.findall("dim")]
          for port in output]


def static_tail_elements(dims: list[int]) -> int:
  product = 1
  for value in dims:
    if value > 0:
      product *= value
  return product


def locked_ir_audit() -> dict[str, Any]:
  root = ET.parse(MODEL_XML).getroot()
  layers = root.find("layers")
  edges = root.find("edges")
  if layers is None or edges is None:
    raise ValueError("locked IR is missing layers or edges")
  by_name = {node.attrib.get("name", ""): node for node in layers}
  pairs = {(edge.attrib["from-layer"], edge.attrib["to-layer"])
           for edge in edges}

  rows: list[dict[str, Any]] = []
  for layer in FULL_ATTENTION_LAYERS:
    # Layer 39 was serialized V/K/Q rather than Q/K/V, so exporter-generated
    # suffixes differ while the graph and tensor contracts stay identical.
    suffix = ({
        "q_reshape": "aten::view/Reshape_2",
        "q_current": "aten::view/Reshape_3",
        "q_transpose": "aten::transpose/Transpose_2",
        "q_rotary_slice": "aten::slice/Slice_4",
        "q_rope": "aten::add/Add_1",
        "q_tail_slice": "aten::slice/Slice_7",
        "q_concat": "aten::cat/Concat_5",
        "k_reshape": "aten::view/Reshape_1",
        "k_transpose": "aten::transpose/Transpose_1",
        "k_rotary_slice": "aten::slice/Slice",
        "k_rope": "aten::add/Add",
        "k_tail_slice": "aten::slice/Slice_3",
        "k_concat": "aten::cat/Concat_2",
        "k_state_concat": "aten::cat/Concat_3",
        "v_reshape": "aten::view/Reshape",
        "v_transpose": "aten::transpose/Transpose",
        "v_state_concat": "aten::cat/Concat",
    } if layer == 39 else {
        "q_reshape": "aten::view/Reshape",
        "q_current": "aten::view/Reshape_1",
        "q_transpose": "aten::transpose/Transpose",
        "q_rotary_slice": "aten::slice/Slice",
        "q_rope": "aten::add/Add",
        "q_tail_slice": "aten::slice/Slice_3",
        "q_concat": "aten::cat/Concat_1",
        "k_reshape": "aten::view/Reshape_2",
        "k_transpose": "aten::transpose/Transpose_1",
        "k_rotary_slice": "aten::slice/Slice_4",
        "k_rope": "aten::add/Add_1",
        "k_tail_slice": "aten::slice/Slice_7",
        "k_concat": "aten::cat/Concat_3",
        "k_state_concat": "aten::cat/Concat_4",
        "v_reshape": "aten::view/Reshape_3",
        "v_transpose": "aten::transpose/Transpose_2",
        "v_state_concat": "aten::cat/Concat_5",
    })
    names = {
        "q_proj": member_name(
            layer, "q_proj/ov_ext::linear/MatMul"),
        "q_reshape": layer_name(layer, suffix["q_reshape"]),
        "q_split": layer_name(
            layer, "prim::ListUnpack/VariadicSplit"),
        "q_current": layer_name(layer, suffix["q_current"]),
        "q_norm": member_name(
            layer, "q_norm/aten::mul/Multiply_1"),
        "q_transpose": layer_name(layer, suffix["q_transpose"]),
        "q_rotary_slice": layer_name(layer, suffix["q_rotary_slice"]),
        "q_rope": layer_name(layer, suffix["q_rope"]),
        "q_tail_slice": layer_name(layer, suffix["q_tail_slice"]),
        "q_concat": layer_name(layer, suffix["q_concat"]),
        "q_gate": layer_name(layer, "aten::reshape/Reshape_3"),
        "k_proj": member_name(
            layer, "k_proj/ov_ext::linear/MatMul"),
        "k_reshape": layer_name(layer, suffix["k_reshape"]),
        "k_norm": member_name(
            layer, "k_norm/aten::mul/Multiply_1"),
        "k_transpose": layer_name(layer, suffix["k_transpose"]),
        "k_rotary_slice": layer_name(layer, suffix["k_rotary_slice"]),
        "k_rope": layer_name(layer, suffix["k_rope"]),
        "k_tail_slice": layer_name(layer, suffix["k_tail_slice"]),
        "k_concat": layer_name(layer, suffix["k_concat"]),
        "k_state_concat": layer_name(layer, suffix["k_state_concat"]),
        "v_proj": member_name(
            layer, "v_proj/ov_ext::linear/MatMul"),
        "v_reshape": layer_name(layer, suffix["v_reshape"]),
        "v_transpose": layer_name(layer, suffix["v_transpose"]),
        "v_state_concat": layer_name(layer, suffix["v_state_concat"]),
        "sdpa": layer_name(
            layer,
            "aten::scaled_dot_product_attention/ScaledDotProductAttention"),
    }
    nodes = {key: by_name.get(name) for key, name in names.items()}
    ids = {key: node.attrib["id"] if node is not None else None
           for key, node in nodes.items()}
    expected_edges = (
        ("q_proj", "q_reshape"), ("q_reshape", "q_split"),
        ("q_split", "q_current"), ("q_split", "q_gate"),
        ("q_norm", "q_transpose"),
        ("q_transpose", "q_rotary_slice"),
        ("q_transpose", "q_tail_slice"), ("q_rope", "q_concat"),
        ("q_tail_slice", "q_concat"), ("q_concat", "sdpa"),
        ("k_proj", "k_reshape"), ("k_norm", "k_transpose"),
        ("k_transpose", "k_rotary_slice"),
        ("k_transpose", "k_tail_slice"), ("k_rope", "k_concat"),
        ("k_tail_slice", "k_concat"),
        ("k_concat", "k_state_concat"),
        ("v_proj", "v_reshape"), ("v_reshape", "v_transpose"),
        ("v_transpose", "v_state_concat"),
    )
    direct = {
        f"{source}->{target}": (
            ids[source] is not None and ids[target] is not None and
            (str(ids[source]), str(ids[target])) in pairs)
        for source, target in expected_edges
    }
    shapes = {
        key: output_dims(node) if node is not None else []
        for key, node in nodes.items()
        if key in ("q_proj", "q_current", "q_concat", "q_gate",
                   "k_proj", "k_concat", "v_proj", "v_transpose")
    }
    shape_ok = (
        bool(shapes["q_proj"]) and
        static_tail_elements(shapes["q_proj"][-1]) == Q_PROJECTION and
        static_tail_elements(shapes["q_current"][-1]) == Q_CURRENT and
        static_tail_elements(shapes["q_concat"][-1]) == Q_CURRENT and
        static_tail_elements(shapes["q_gate"][-1]) == Q_GATE and
        static_tail_elements(shapes["k_proj"][-1]) == KV_CURRENT and
        static_tail_elements(shapes["k_concat"][-1]) == KV_CURRENT and
        static_tail_elements(shapes["v_proj"][-1]) == KV_CURRENT and
        static_tail_elements(shapes["v_transpose"][-1]) == KV_CURRENT)
    rows.append({
        "layer": layer,
        "names_present": all(node is not None for node in nodes.values()),
        "direct_edges": direct,
        "all_direct_edges": all(direct.values()),
        "shape_contract": shapes,
        "shape_contract_pass": shape_ok,
        "node_ids": ids,
    })
  return {
      "layers": list(FULL_ATTENTION_LAYERS),
      "rows": rows,
      "all_names_present": all(row["names_present"] for row in rows),
      "all_direct_edges": all(row["all_direct_edges"] for row in rows),
      "all_shapes_locked": all(row["shape_contract_pass"] for row in rows),
  }


def classify_boundary_profile(profile: dict[str, Any]) -> dict[str, Any]:
  rows = profile.get("full_profile", [])
  layer_pattern = re.compile(
      r"layers\.(3|7|11|15|19|23|27|31|35|39)\.self_attn")
  selected: list[dict[str, Any]] = []
  for row in rows:
    if row.get("status") != "Status.EXECUTED":
      continue
    name = str(row.get("node_name", ""))
    if not layer_pattern.search(name):
      continue
    node_type = str(row.get("node_type", ""))
    kind = None
    if (node_type == "FullyConnectedCompressed" and
        ".q_proj/ov_ext::linear/MatMul_fused_3FCs" in name):
      kind = "fused_q_gate_k_v_projection"
    elif node_type == "VariadicSplit" and "ListUnpack" in name:
      kind = "q_gate_split"
    elif node_type == "Crop" and "ListUnpack" in name:
      kind = "q_gate_crop"
    elif node_type == "RMS" and (".q_norm/" in name or ".k_norm/" in name):
      kind = "q_k_rms"
    elif node_type == "RoPE":
      kind = "q_k_rope"
    elif node_type == "Concat":
      kind = "q_k_rotary_concat"
    elif node_type == "StridedSlice" and any(
        name.endswith(suffix) for suffix in (
            "/aten::slice/Slice", "/aten::slice/Slice_3",
            "/aten::slice/Slice_4", "/aten::slice/Slice_7")):
      kind = "q_k_rotary_slice"
    elif node_type == "Transpose" and any(
        name.endswith(suffix) for suffix in (
            "/aten::transpose/Transpose",
            "/aten::transpose/Transpose_1",
            "/aten::transpose/Transpose_2")):
      kind = "q_k_v_transpose"
    if kind is not None:
      selected.append({**row, "boundary_kind": kind})

  counts = Counter(str(row["boundary_kind"]) for row in selected)
  expected = {
      "fused_q_gate_k_v_projection": 10,
      "q_gate_split": 10,
      "q_gate_crop": 10,
      "q_k_rms": 20,
      "q_k_rope": 20,
      "q_k_rotary_concat": 20,
      "q_k_rotary_slice": 40,
      "q_k_v_transpose": 30,
  }
  optimized_reorders = [
      row for row in rows
      if row.get("status") == "Status.OPTIMIZED_OUT" and
      str(row.get("node_type")) == "Reorder" and
      "_iq36_hot_attention_layer" in str(row.get("node_name", "")) and
      "_cldnn_custom_preprocess" in str(row.get("node_name", ""))]
  custom_attention = [
      row for row in rows
      if row.get("status") == "Status.EXECUTED" and
      row.get("node_type") == "IQ36HotAttentionGQA"]
  all_rms = [
      row for row in rows
      if row.get("status") == "Status.EXECUTED" and
      row.get("node_type") == "RMS"]
  qk_rms = [row for row in selected if row["boundary_kind"] == "q_k_rms"]
  rms_exec_types = sorted({str(row.get("exec_type")) for row in all_rms})
  all_rms_raw_us = sum(float(row.get("real_time_us", 0.0)) for row in all_rms)
  qk_rms_raw_us = sum(float(row.get("real_time_us", 0.0)) for row in qk_rms)
  return {
      "counts": dict(counts),
      "expected_counts": expected,
      "counts_exact": dict(counts) == expected,
      "selected_executed_rows": len(selected),
      "optimized_custom_preprocess_reorders": len(optimized_reorders),
      "executed_custom_attention": len(custom_attention),
      "all_rms_count": len(all_rms),
      "q_k_rms_count": len(qk_rms),
      "rms_exec_types": rms_exec_types,
      "all_rms_raw_us": all_rms_raw_us,
      "q_k_rms_raw_us": qk_rms_raw_us,
      "q_k_rms_same_type_share": (
          qk_rms_raw_us / all_rms_raw_us if all_rms_raw_us > 0.0 else None),
  }


def parse_host_profile(path: Path) -> dict[str, Any]:
  field = re.compile(r"(\w+)=([^ ]+)")
  segments: list[dict[str, Any]] = []
  stream_rows: list[dict[str, str]] = []
  for line in path.read_text(encoding="utf-8").splitlines():
    values = dict(field.findall(line))
    if "stage=ze_stream" in line:
      stream_rows.append(values)
    elif "stage=network " in line:
      def total(key: str) -> int:
        return sum(int(row.get(key, 0)) for row in stream_rows)
      segments.append({
          "network": values,
          "stream_count": len(stream_rows),
          "set_arguments_us": total("set_arguments_us"),
          "set_arguments_calls": total("set_arguments_calls"),
          "enqueue_kernel_us": total("enqueue_kernel_us"),
          "enqueue_kernel_calls": total("enqueue_kernel_calls"),
          "append_kernel_us": total("append_kernel_us"),
      })
      stream_rows = []
  steady = segments[-5:]
  return {
      "path": display_path(path),
      "segments": len(segments),
      "steady_segments": steady,
      "steady_enqueue_calls": sorted({
          int(row["enqueue_kernel_calls"]) for row in steady}),
      "steady_set_arguments_us_max": max(
          int(row["set_arguments_us"]) for row in steady),
      "steady_enqueue_us_per_call_max": max(
          float(row["enqueue_kernel_us"]) /
          int(row["enqueue_kernel_calls"])
          for row in steady if int(row["enqueue_kernel_calls"]) > 0),
  }


def main() -> int:
  args = parse_args()
  output = args.output.resolve()
  output.mkdir(parents=True, exist_ok=False)
  stop_bytes = int(args.memory_stop_gib * 1024**3)
  memory: list[dict[str, Any]] = []
  sample_memory("start", stop_bytes, memory)

  required = (
      MODEL_XML, GRAPH_SOURCE, STATUS, FRONTIER, REJECTED, FC_COMPONENT,
      PROFILE_RESULT, EXACT_BUCKET, *HOST_PROFILES)
  missing = [str(path) for path in required if not path.is_file()]
  if missing:
    raise SystemExit("missing source-bound inputs: " + ", ".join(missing))

  git = git_state()
  ir = locked_ir_audit()
  sample_memory("after-locked-ir-audit", stop_bytes, memory)
  graph_source = GRAPH_SOURCE.read_text(encoding="utf-8")
  status_text = STATUS.read_text(encoding="utf-8")
  frontier = load_json(FRONTIER)
  rejected = load_json(REJECTED)
  fc = load_json(FC_COMPONENT)
  profile = load_json(PROFILE_RESULT)
  exact_bucket = load_json(EXACT_BUCKET)
  runtime = classify_boundary_profile(profile)
  host = [parse_host_profile(path) for path in HOST_PROFILES]
  sample_memory("after-stored-evidence-audit", stop_bytes, memory)

  fc_saving_ms = float(fc["aggregate"]["optimistic_saving_ms"])
  kill_number_ms = float(
      frontier["goal_budget"]["per_token_ms"]["remaining_cut"])
  remaining_required_ms = kill_number_ms - fc_saving_ms

  other_event_ms = EVENT_TOTAL_MS - sum((
      EVENT_FC_MS, EVENT_ATTENTION_MS, EVENT_GDN_MS, EVENT_RMS_MS,
      EVENT_LINEAR_CONV_MS))
  rms_share = float(runtime["q_k_rms_same_type_share"] or 0.0)
  qk_rms_event_ceiling_ms = EVENT_RMS_MS * rms_share
  # Assign every other executed event in the complete model to this boundary,
  # then add the whole Q/K RMS share even though its arithmetic must remain.
  # This is intentionally much more favorable than the actual source scope.
  device_event_ceiling_ms = other_event_ms + qk_rms_event_ceiling_ms

  boundary_launches = int(runtime["selected_executed_rows"]) + 2
  all_set_arguments_us = max(
      int(row["steady_set_arguments_us_max"]) for row in host)
  enqueue_us_per_call = max(
      float(row["steady_enqueue_us_per_call_max"]) for row in host)
  provider_launch_ceiling_ms = (
      all_set_arguments_us + boundary_launches * enqueue_us_per_call) / 1000.0

  # The FC schedule remains intact.  Only the Q/K/V portion of its F16 output
  # can stop being materialized; the equally sized Q gate must still be
  # published for the post-attention gate.  Intermediate consumer traffic is
  # already overcharged by the complete event residual above.
  projection_output_bytes_per_layer = (
      Q_CURRENT + 2 * KV_CURRENT) * F16_BYTES
  projection_output_bytes = (
      len(FULL_ATTENTION_LAYERS) * projection_output_bytes_per_layer)
  small_tensor_gbps = 26.8865641025641
  projection_output_ceiling_ms = (
      projection_output_bytes / small_tensor_gbps / 1_000_000.0)

  # Provider host time and device event time are non-additive on the accepted
  # in-order carrier.  Taking the larger is the overlap-free launch envelope;
  # adding both would repeat the already-closed v28f provider route.
  adjacent_ceiling_ms = (
      max(device_event_ceiling_ms, provider_launch_ceiling_ms) +
      projection_output_ceiling_ms)
  residual_shortfall_ms = remaining_required_ms - adjacent_ceiling_ms
  route_fundable = adjacent_ceiling_ms >= remaining_required_ms

  v28f = next(
      (row for row in rejected.get("rejected", [])
       if row.get("route") ==
       "openvino_level_zero_prepare_and_assign_microcuts_v28f"), {})
  provider_envelope_ms = float(
      exact_bucket["optimistic_union_bound"]
      ["entire_provider_setup_append_envelope_ms_per_token"])
  status_has_census = all(value in status_text for value in (
      "24.129 ms/token", "13.375", "8.456", "1.319", "0.358",
      "0.193"))
  source_splice_ok = all(value in graph_source for value in (
      "query = normalize_attention_layout(target.input_value(0))",
      "key_assign.input_value(0).get_node().input_value(1)",
      "value_assign.input_value(0).get_node().input_value(1)",
      "target.output(0).replace(attention_output)",
  ))
  checks = [
      check("repository_clean_at_gate", not git["dirty"], git=git),
      check("locked_ir_has_exact_ten_projection_consumer_chains",
            ir["all_names_present"] and ir["all_direct_edges"] and
            ir["all_shapes_locked"] and len(ir["rows"]) == 10,
            layers=ir["layers"]),
      check("accepted_graph_consumes_postprocessed_q_and_current_k_v",
            source_splice_ok),
      check("seq1233_is_complete_fused_qkv_fc_ceiling",
            fc.get("required_checks_passed") is True and
            fc.get("route_stop_proven") is True and
            int(fc["aggregate"]["remaining_bytes_not_charged"]) == 0 and
            "full_attention_q_k_v" in
            fc["aggregate"]["native_fusion_groups"]),
      check("stored_profile_has_exact_boundary_event_counts",
            runtime["counts_exact"] and
            runtime["optimized_custom_preprocess_reorders"] == 130 and
            runtime["executed_custom_attention"] == 10,
            counts=runtime["counts"],
            optimized_reorders=
                runtime["optimized_custom_preprocess_reorders"]),
      check("rms_share_uses_only_one_executed_kernel_type",
            runtime["rms_exec_types"] == ["rms_gpu_bfyx_opt__f16"] and
            runtime["all_rms_count"] == 131 and
            runtime["q_k_rms_count"] == 20,
            exec_types=runtime["rms_exec_types"]),
      check("status_registers_corrected_complete_event_census",
            status_has_census),
      check("steady_provider_profiles_have_fixed_696_dispatches",
            all(row["steady_enqueue_calls"] == [696] for row in host),
            profiles=host),
      check("clean_exact_bucket_registers_complete_provider_envelope",
            exact_bucket.get("required_evidence_checks_passed") is True and
            provider_envelope_ms == 1.0,
            provider_envelope_ms=provider_envelope_ms),
      check("provider_prepare_route_is_closed_on_complete_wall",
            "does not improve the matched complete wall" in
            str(v28f.get("reason", "")), route=v28f),
      check("projection_consumer_ceiling_is_below_remaining_cut",
            not route_fundable,
            adjacent_ceiling_ms=adjacent_ceiling_ms,
            remaining_required_ms=remaining_required_ms,
            residual_shortfall_ms=residual_shortfall_ms),
      check("memory_guard_never_tripped",
            all(row["available_bytes"] >= stop_bytes for row in memory),
            minimum_available_bytes=min(
                row["available_bytes"] for row in memory)),
  ]
  required_checks_passed = all(row["pass"] for row in checks)
  verdict = (
      "reject_full_attention_projection_consumer_before_source"
      if required_checks_passed and not route_fundable else
      "admit_one_full_attention_projection_consumer_component"
      if required_checks_passed else "inconclusive")

  metrics = {
      "schema": SCHEMA,
      "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
      "workstream": WS,
      "git": git,
      "verdict": verdict,
      "required_checks_passed": required_checks_passed,
      "component_admitted": required_checks_passed and route_fundable,
      "source_edit_admitted": required_checks_passed and route_fundable,
      "compile_admitted": False,
      "gpu_worker_launched": False,
      "long_worker_admitted": False,
      "budget": {
          "kill_number_ms_per_token": kill_number_ms,
          "seq1233_optimistic_fc_saving_ms_per_token": fc_saving_ms,
          "remaining_required_ms_per_token": remaining_required_ms,
          "projection_consumer_ceiling_ms_per_token": adjacent_ceiling_ms,
          "residual_shortfall_ms_per_token": residual_shortfall_ms,
      },
      "overlap_free_ceiling": {
          "complete_other_event_residual_ms_per_token": other_event_ms,
          "q_k_rms_same_type_share": rms_share,
          "q_k_rms_event_ceiling_ms_per_token": qk_rms_event_ceiling_ms,
          "device_event_ceiling_ms_per_token": device_event_ceiling_ms,
          "boundary_executed_launches": boundary_launches,
          "all_graph_set_arguments_us_charged_to_boundary":
              all_set_arguments_us,
          "max_steady_enqueue_us_per_dispatch": enqueue_us_per_call,
          "provider_launch_ceiling_ms_per_token": provider_launch_ceiling_ms,
          "projection_output_bytes_per_layer":
              projection_output_bytes_per_layer,
          "projection_output_bytes_all_layers": projection_output_bytes,
          "small_tensor_gbps": small_tensor_gbps,
          "projection_output_ceiling_ms_per_token":
              projection_output_ceiling_ms,
          "union_rule": (
              "max(device event envelope, provider launch envelope) plus "
              "FC projection-output write; provider and device timelines "
              "are not added"),
          "optimism": [
              "assigns every non-major event in the whole model to this boundary",
              "charges the complete Q/K RMS share although RMS arithmetic remains",
              "charges every steady graph set-argument call to this boundary",
              "charges the worst stored steady enqueue cost to every boundary event",
              "charges projection output at the slow seq1237 small-tensor rate",
              "charges zero implementation, synchronization, or occupancy penalty",
          ],
      },
      "locked_ir_audit": ir,
      "runtime_profile_audit": runtime,
      "host_profile_audit": host,
      "checks": checks,
      "memory_stop_bytes": stop_bytes,
      "memory_samples": memory,
      "inputs": {
          display_path(path): sha256(path) for path in required
      },
  }
  (output / "metrics.json").write_text(
      json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
  summary = f"""# Full-attention projection-to-consumer source bound

Verdict: **{verdict}**. Required evidence checks:
`{str(required_checks_passed).lower()}`. No compiler or GPU worker ran.

The locked IR has ten identical Q/gate/K/V chains. The stored accepted runtime
executes exactly 10 fused Q/gate/K/V projections, 20 Q/gate split/Crop events,
20 Q/K RMS kernels, 20 RoPE kernels, 20 rotary concats, 40 rotary slices, and
30 Q/K/V transposes. All 130 custom-op preprocess reorders are already optimized
out. The accepted custom attention consumes postprocessed Q and current K/V, so
its arithmetic remains unchanged.

Seq1233 already supplies `{fc_saving_ms:.6f} ms/token`; the boundary must add
`{remaining_required_ms:.6f} ms/token`. The device-side ceiling gives the
boundary every non-major event in the entire model plus the whole same-type
Q/K RMS share: `{device_event_ceiling_ms:.6f} ms/token`. The independent host
envelope charges all graph set-argument work plus the worst stored steady
enqueue cost to all `{boundary_launches}` boundary events:
`{provider_launch_ceiling_ms:.6f} ms/token`. Because those timelines are
non-additive on the accepted in-order carrier, the bound takes the larger, not
their sum. Deleting every Q/K/V projection-output write adds only
`{projection_output_ceiling_ms:.6f} ms/token` even at seq1237's slow
small-tensor rate.

The complete overlap-free ceiling is therefore `{adjacent_ceiling_ms:.6f}
ms/token`, short by `{residual_shortfall_ms:.6f} ms/token`. Source, compile,
graph integration, 32k, ABBA, and output512 are not admitted for this route.
"""
  (output / "summary.md").write_text(summary, encoding="utf-8")
  print(json.dumps({
      "artifact": display_path(output),
      "verdict": verdict,
      "remaining_required_ms": remaining_required_ms,
      "projection_consumer_ceiling_ms": adjacent_ceiling_ms,
      "residual_shortfall_ms": residual_shortfall_ms,
      "gpu_worker_launched": False,
  }, separators=(",", ":")))
  return 0 if required_checks_passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
