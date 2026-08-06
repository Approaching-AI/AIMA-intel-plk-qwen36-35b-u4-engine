#!/usr/bin/env python3
"""Bound a decode-wide RMS/residual/rotary/gate fusion bundle.

This gate is deliberately source-only.  It audits the locked IR, the stored
accepted runtime profile, and the reduction/workgroup ownership of the current
RMS, GDN, and optimistic FC schedules.  It never compiles or launches a GPU
worker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-openvino-decode-elementwise-residual-bound-v0"

MODEL_ROOT = Path("/home/intel/Qwen3.6-35B-A3B-ov")
MODEL_XML = MODEL_ROOT / "openvino_language_model.xml"
MODEL_CONFIG = MODEL_ROOT / "config.json"
OPENVINO_GPU = Path(
    "/home/intel/intel-qwen36-r0/source/openvino-90214e5be05/"
    "src/plugins/intel_gpu")
RMS_KERNEL = (
    OPENVINO_GPU / "src/kernel_selector/cl_kernels/rms_gpu_bfyx_opt.cl")
RMS_DISPATCH = (
    OPENVINO_GPU /
    "src/kernel_selector/kernels/rms/rms_kernel_bfyx_opt.cpp")
GDN_KERNEL = (
    OPENVINO_GPU / "src/graph/impls/ocl_v2/gated_delta_net_ref.cl")
GDN_DISPATCH = (
    OPENVINO_GPU / "src/graph/impls/ocl_v2/gated_delta_net_ref.cpp")

STATUS = REPO / "doc/active" / WS / "STATUS.md"
FRONTIER = REPO / "doc/active" / WS / "frontier.json"
FC_COMPONENT = REPO / (
    "output/openvino-fc-micro-component-20260715Tseq1233-"
    "max-native-fused-nonzero-warm512-cleanZ/metrics.json")
PROFILE_RESULT = REPO / (
    "output/openvino-attention-phase-profile-20260715Tseq1136-"
    "dq-subgroup-32k-warm17-cleanZ/raw/32k/candidate/worker-result.json")
FULL_ATTENTION_BOUND = REPO / (
    "output/openvino-full-attention-projection-consumer-bound-"
    "20260715Tseq1238-cleanZ/metrics.json")

F16_BYTES = 2

# Corrected complete executed-event census registered in STATUS.  The entire
# non-major residual is charged as a deliberately favorable ceiling even
# though it includes required arithmetic and unrelated CPU/dynamic-shape work.
EVENT_TOTAL_MS = 24.129
EVENT_FC_MS = 13.375
EVENT_ATTENTION_MS = 8.456
EVENT_GDN_MS = 1.319
EVENT_RMS_MS = 0.358
EVENT_LINEAR_CONV_MS = 0.193

RMS_EXPECTED_COUNTS = {
    "input_layernorm": 40,
    "post_attention_layernorm": 40,
    "linear_attn_norm": 30,
    "q_norm": 10,
    "k_norm": 10,
    "final_norm": 1,
}
RMS_EXPECTED_SHAPES = {
    "input_layernorm": [-1, -1, 2048],
    "post_attention_layernorm": [-1, -1, 2048],
    "linear_attn_norm": [-1, 128],
    "q_norm": [-1, -1, 16, 256],
    "k_norm": [-1, -1, 2, 256],
    "final_norm": [-1, -1, 2048],
}
ELEMENTWISE_EXPECTED_COUNTS = {
    "residual_add": 80,
    "linear_gate_multiply": 30,
    "full_gate_multiply": 10,
    "linear_norm_swish": 30,
    "linear_softplus": 30,
    "q_k_rope": 20,
    "q_k_rotary_concat": 20,
    "q_k_rotary_slice": 40,
    "q_k_v_transpose": 30,
    "q_gate_crop": 10,
    "q_gate_split": 10,
}


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


def rms_category(name: str) -> str | None:
  if ".input_layernorm/" in name:
    return "input_layernorm"
  if ".post_attention_layernorm/" in name:
    return "post_attention_layernorm"
  if ".linear_attn.norm/" in name:
    return "linear_attn_norm"
  if ".self_attn.q_norm/" in name:
    return "q_norm"
  if ".self_attn.k_norm/" in name:
    return "k_norm"
  if name.startswith("__module.model.model.language_model.norm/"):
    return "final_norm"
  return None


def locked_ir_rms_audit(config: dict[str, Any]) -> dict[str, Any]:
  root = ET.parse(MODEL_XML).getroot()
  layers = root.find("layers")
  edges = root.find("edges")
  if layers is None or edges is None:
    raise ValueError("locked IR is missing layers or edges")
  by_name = {node.attrib.get("name", ""): node for node in layers}
  pairs = {(edge.attrib["from-layer"], edge.attrib["to-layer"])
           for edge in edges}

  targets = [
      node for node in layers
      if node.attrib.get("name", "").endswith("/aten::mul/Multiply_1")
      and rms_category(node.attrib.get("name", "")) is not None]
  counts: Counter[str] = Counter()
  rows: list[dict[str, Any]] = []
  for target in targets:
    name = target.attrib["name"]
    category = rms_category(name)
    assert category is not None
    counts[category] += 1
    prefix = name.removesuffix("/aten::mul/Multiply_1")
    expected_names = {
        "power": prefix + "/aten::pow/Power",
        "mean": prefix + "/aten::mean/ReduceMean",
        "add": prefix + "/aten::add/Add",
        "sqrt": prefix + "/aten::rsqrt/Sqrt",
        "divide": prefix + "/aten::rsqrt/Divide",
        "apply": prefix + "/aten::mul/Multiply",
        "affine": prefix + "/aten::mul/Multiply_1",
    }
    nodes = {key: by_name.get(value) for key, value in expected_names.items()}
    ids = {key: node.attrib["id"] if node is not None else None
           for key, node in nodes.items()}
    expected_edges = (
        ("power", "mean"), ("mean", "add"), ("add", "sqrt"),
        ("sqrt", "divide"), ("divide", "apply"),
        ("apply", "affine"),
    )
    direct = {
        f"{source}->{sink}": (
            ids[source] is not None and ids[sink] is not None and
            (str(ids[source]), str(ids[sink])) in pairs)
        for source, sink in expected_edges
    }
    shape = output_dims(target)
    rows.append({
        "name": name,
        "category": category,
        "names_present": all(node is not None for node in nodes.values()),
        "direct_reduction_chain": direct,
        "all_direct_reduction_edges": all(direct.values()),
        "output_shape": shape,
        "shape_pass": shape == [RMS_EXPECTED_SHAPES[category]],
        "node_ids": ids,
    })

  text_config = config.get("text_config", {})
  architecture = {
      "hidden_size": int(text_config.get("hidden_size", -1)),
      "attention_heads": int(text_config.get("num_attention_heads", -1)),
      "kv_heads": int(text_config.get("num_key_value_heads", -1)),
      "head_dim": int(text_config.get("head_dim", -1)),
      "linear_value_heads": int(
          text_config.get("linear_num_value_heads", -1)),
      "linear_value_head_dim": int(
          text_config.get("linear_value_head_dim", -1)),
  }
  architecture_expected = {
      "hidden_size": 2048,
      "attention_heads": 16,
      "kv_heads": 2,
      "head_dim": 256,
      "linear_value_heads": 32,
      "linear_value_head_dim": 128,
  }
  return {
      "counts": dict(counts),
      "expected_counts": RMS_EXPECTED_COUNTS,
      "counts_exact": dict(counts) == RMS_EXPECTED_COUNTS,
      "rows": rows,
      "all_names_present": all(row["names_present"] for row in rows),
      "all_reduction_edges_direct": all(
          row["all_direct_reduction_edges"] for row in rows),
      "all_shapes_locked": all(row["shape_pass"] for row in rows),
      "architecture": architecture,
      "architecture_expected": architecture_expected,
      "architecture_exact": architecture == architecture_expected,
  }


def classify_elementwise_row(row: dict[str, Any]) -> str | None:
  name = str(row.get("node_name", ""))
  node_type = str(row.get("node_type", ""))
  layer = r"layers\.\d+"
  if node_type == "Add" and (
      re.search(layer + r"\.mlp/aten::add/Add$", name) or
      re.search(layer + r"/aten::add/Add_1$", name)):
    return "residual_add"
  if (node_type == "Multiply" and
      re.search(layer + r"\.linear_attn/aten::mul/Multiply_6$", name)):
    return "linear_gate_multiply"
  if (node_type == "Multiply" and
      re.search(layer + r"\.self_attn/aten::mul/Multiply_6$", name)):
    return "full_gate_multiply"
  if (node_type == "Swish" and
      re.search(layer + r"\.linear_attn\.norm/aten::silu/Swish$", name)):
    return "linear_norm_swish"
  if (node_type == "SoftPlus" and
      re.search(layer + r"\.linear_attn/aten::softplus/SoftPlus$", name)):
    return "linear_softplus"
  if node_type == "RoPE" and ".self_attn/aten::add/Add" in name:
    return "q_k_rope"
  if (node_type == "Concat" and
      ".self_attn/aten::cat/Concat" in name):
    return "q_k_rotary_concat"
  if (node_type == "StridedSlice" and
      ".self_attn/aten::slice/Slice" in name):
    return "q_k_rotary_slice"
  if (node_type == "Transpose" and re.search(
      layer + r"\.self_attn/aten::transpose/Transpose(?:_[12])?$", name)):
    return "q_k_v_transpose"
  if (node_type == "Crop" and
      ".self_attn/prim::ListUnpack/VariadicSplit.out1" in name):
    return "q_gate_crop"
  if (node_type == "VariadicSplit" and
      ".self_attn/prim::ListUnpack/VariadicSplit.out0" in name):
    return "q_gate_split"
  return None


def runtime_profile_audit(
    profile: dict[str, Any], ir: dict[str, Any],
) -> dict[str, Any]:
  rows = profile.get("full_profile", [])
  executed = [row for row in rows if row.get("status") == "Status.EXECUTED"]
  rms_rows = [row for row in executed if row.get("node_type") == "RMS"]
  ir_names = {str(row["name"]) for row in ir["rows"]}
  runtime_names = {str(row.get("node_name", "")) for row in rms_rows}
  rms_counts: Counter[str] = Counter()
  raw_us: defaultdict[str, float] = defaultdict(float)
  for row in rms_rows:
    category = rms_category(str(row.get("node_name", "")))
    if category is not None:
      rms_counts[category] += 1
      raw_us[category] += float(row.get("real_time_us", 0.0))

  selected = []
  for row in executed:
    category = classify_elementwise_row(row)
    if category is not None:
      selected.append({**row, "boundary_kind": category})
  elementwise_counts = Counter(
      str(row["boundary_kind"]) for row in selected)
  return {
      "executed_rows": len(executed),
      "rms_count": len(rms_rows),
      "rms_counts": dict(rms_counts),
      "rms_expected_counts": RMS_EXPECTED_COUNTS,
      "rms_counts_exact": dict(rms_counts) == RMS_EXPECTED_COUNTS,
      "rms_names_match_locked_ir": runtime_names == ir_names,
      "rms_exec_types": sorted({
          str(row.get("exec_type", "")) for row in rms_rows}),
      "rms_raw_us_by_category": dict(raw_us),
      "rms_raw_us_total": sum(raw_us.values()),
      "elementwise_boundary_counts": dict(elementwise_counts),
      "elementwise_expected_counts": ELEMENTWISE_EXPECTED_COUNTS,
      "elementwise_counts_exact": (
          dict(elementwise_counts) == ELEMENTWISE_EXPECTED_COUNTS),
      "elementwise_boundary_executed_rows": len(selected),
  }


def source_schedule_audit(fc: dict[str, Any]) -> dict[str, Any]:
  rms_kernel = RMS_KERNEL.read_text(encoding="utf-8")
  rms_dispatch = RMS_DISPATCH.read_text(encoding="utf-8")
  gdn_kernel = GDN_KERNEL.read_text(encoding="utf-8")
  gdn_dispatch = GDN_DISPATCH.read_text(encoding="utf-8")
  rms_markers = (
      "rms = sub_group_reduce_add(rms);",
      "barrier(CLK_LOCAL_MEM_FENCE);",
      "slm_buf[0] = native_powr(sqrt(",
      "HAS_FUSED_OPS",
      "BLOCK_WRITE(output",
  )
  rms_dispatch_markers = (
      "dispatchData.gws[0] = dispatchData.lws[0];",
      "dispatchData.gws[1] = dispatchData.dataCount;",
      "dispatchData.lws[1] = 1;",
  )
  gdn_markers = (
      "wgs.global = {batch, head_nums, v_blocks * subgroup_size};",
      "wgs.local = {1, 1, subgroup_size};",
      "const size_t v_blocks = (v_head_dims + v_block_size - 1)",
  )

  fc_rows: list[dict[str, Any]] = []
  for cohort in fc.get("cohorts", []):
    if int(cohort.get("count", 0)) <= 0:
      continue
    runtime = cohort.get("runtime", {})
    package = cohort.get("package", {})
    global_size = [int(value) for value in runtime.get("global", [])]
    local_size = [int(value) for value in runtime.get("local", [])]
    global_items = math.prod(global_size) if global_size else 0
    local_items = math.prod(local_size) if local_size else 0
    workgroups = (
        global_items // local_items if local_items > 0 else 0)
    fc_rows.append({
        "name": cohort.get("name"),
        "count": int(cohort.get("count", 0)),
        "global": global_size,
        "local": local_size,
        "workgroups": workgroups,
        "barrier_count": int(package.get("barrier_count", -1)),
        "multiple_workgroups": workgroups > 1,
    })
  return {
      "rms_kernel_markers": {
          marker: marker in rms_kernel for marker in rms_markers},
      "rms_dispatch_markers": {
          marker: marker in rms_dispatch for marker in rms_dispatch_markers},
      "rms_reduction_and_dispatch_contract_present": (
          all(marker in rms_kernel for marker in rms_markers) and
          all(marker in rms_dispatch for marker in rms_dispatch_markers)),
      "rms_kernel_local_barrier_occurrences": rms_kernel.count(
          "barrier(CLK_LOCAL_MEM_FENCE);"),
      "rms_kernel_supports_fused_epilogue": "HAS_FUSED_OPS" in rms_kernel,
      "gdn_schedule_markers": {
          marker: marker in gdn_dispatch for marker in gdn_markers},
      "gdn_kernel_uses_subgroup_reductions": (
          gdn_kernel.count("sub_group_reduce_add") >= 4),
      "gdn_output_workgroups_per_head_at_decode": 128 // 4,
      "gdn_output_schedule_is_partitioned": (
          all(marker in gdn_dispatch for marker in gdn_markers) and
          gdn_kernel.count("sub_group_reduce_add") >= 4),
      "fc_cohorts": fc_rows,
      "all_seq1233_fc_cohorts_use_multiple_workgroups": (
          bool(fc_rows) and all(row["multiple_workgroups"] for row in fc_rows)),
      "all_seq1233_fc_cohorts_have_local_barrier": (
          bool(fc_rows) and all(row["barrier_count"] == 1 for row in fc_rows)),
  }


def rms_decode_elements(ir: dict[str, Any]) -> dict[str, int]:
  architecture = ir["architecture"]
  return {
      "input_layernorm": (
          RMS_EXPECTED_COUNTS["input_layernorm"] *
          int(architecture["hidden_size"])),
      "post_attention_layernorm": (
          RMS_EXPECTED_COUNTS["post_attention_layernorm"] *
          int(architecture["hidden_size"])),
      "linear_attn_norm": (
          RMS_EXPECTED_COUNTS["linear_attn_norm"] *
          int(architecture["linear_value_heads"]) *
          int(architecture["linear_value_head_dim"])),
      "q_norm": (
          RMS_EXPECTED_COUNTS["q_norm"] *
          int(architecture["attention_heads"]) *
          int(architecture["head_dim"])),
      "k_norm": (
          RMS_EXPECTED_COUNTS["k_norm"] *
          int(architecture["kv_heads"]) *
          int(architecture["head_dim"])),
      "final_norm": int(architecture["hidden_size"]),
  }


def main() -> int:
  args = parse_args()
  output = args.output.resolve()
  output.mkdir(parents=True, exist_ok=False)
  stop_bytes = int(args.memory_stop_gib * 1024**3)
  memory: list[dict[str, Any]] = []
  sample_memory("start", stop_bytes, memory)

  required = (
      MODEL_XML, MODEL_CONFIG, RMS_KERNEL, RMS_DISPATCH, GDN_KERNEL,
      GDN_DISPATCH, STATUS, FRONTIER, FC_COMPONENT, PROFILE_RESULT,
      FULL_ATTENTION_BOUND)
  missing = [str(path) for path in required if not path.is_file()]
  if missing:
    raise SystemExit("missing source-bound inputs: " + ", ".join(missing))

  git = git_state()
  config = load_json(MODEL_CONFIG)
  ir = locked_ir_rms_audit(config)
  sample_memory("after-locked-ir-audit", stop_bytes, memory)
  status_text = STATUS.read_text(encoding="utf-8")
  frontier = load_json(FRONTIER)
  fc = load_json(FC_COMPONENT)
  profile = load_json(PROFILE_RESULT)
  seq1238 = load_json(FULL_ATTENTION_BOUND)
  runtime = runtime_profile_audit(profile, ir)
  source = source_schedule_audit(fc)
  sample_memory("after-stored-source-audit", stop_bytes, memory)

  fc_saving_ms = float(fc["aggregate"]["optimistic_saving_ms"])
  kill_number_ms = float(
      frontier["goal_budget"]["per_token_ms"]["remaining_cut"])
  remaining_required_ms = kill_number_ms - fc_saving_ms
  other_event_ms = EVENT_TOTAL_MS - sum((
      EVENT_FC_MS, EVENT_ATTENTION_MS, EVENT_GDN_MS, EVENT_RMS_MS,
      EVENT_LINEAR_CONV_MS))

  elements = rms_decode_elements(ir)
  all_rms_elements = sum(elements.values())
  qk_elements = elements["q_norm"] + elements["k_norm"]
  independent_rms_elements = all_rms_elements - qk_elements
  edge_bytes_per_element = 2 * F16_BYTES
  small_tensor_gbps = float(
      seq1238["overlap_free_ceiling"]["small_tensor_gbps"])
  independent_rms_edge_bytes = (
      independent_rms_elements * edge_bytes_per_element)
  all_rms_edge_bytes = all_rms_elements * edge_bytes_per_element
  independent_rms_edge_ms = (
      independent_rms_edge_bytes / small_tensor_gbps / 1_000_000.0)
  duplicate_inclusive_rms_edge_ms = (
      all_rms_edge_bytes / small_tensor_gbps / 1_000_000.0)

  # One reduction-bearing dispatch remains for every RMS vector family.  A
  # native bundle may fold residual/activation/rotary/gate work into that
  # dispatch, or make a consumer apply the scale, but it cannot delete both
  # the adjacent event and the reduction dispatch without changing the
  # registered reduction/FC/GDN schedules.  Charge the entire non-major event
  # residual anyway.  Then stress the rejection by also charging every RMS
  # normalized-output write+read, including Q/K edges already covered by the
  # closed seq1238 route.  Neither seq1238's 0.560-ms envelope nor provider
  # time is added.
  independent_ceiling_ms = other_event_ms + independent_rms_edge_ms
  duplicate_inclusive_stress_ceiling_ms = (
      other_event_ms + duplicate_inclusive_rms_edge_ms)
  residual_shortfall_ms = (
      remaining_required_ms - duplicate_inclusive_stress_ceiling_ms)
  route_fundable = (
      duplicate_inclusive_stress_ceiling_ms >= remaining_required_ms)

  status_has_census = all(value in status_text for value in (
      "24.129 ms/token", "13.375", "8.456", "1.319", "0.358",
      "0.193"))
  seq1238_closed = (
      seq1238.get("required_checks_passed") is True and
      seq1238.get("component_admitted") is False and
      seq1238.get("verdict") ==
      "reject_full_attention_projection_consumer_before_source")
  checks = [
      check("repository_clean_at_gate", not git["dirty"], git=git),
      check("locked_ir_has_exact_131_rms_patterns",
            ir["counts_exact"] and ir["all_names_present"] and
            ir["all_reduction_edges_direct"] and ir["all_shapes_locked"],
            counts=ir["counts"]),
      check("locked_decode_architecture_is_exact",
            ir["architecture_exact"], architecture=ir["architecture"]),
      check("stored_profile_rms_matches_locked_ir_exactly",
            runtime["rms_count"] == 131 and
            runtime["rms_counts_exact"] and
            runtime["rms_names_match_locked_ir"] and
            runtime["rms_exec_types"] == ["rms_gpu_bfyx_opt__f16"],
            counts=runtime["rms_counts"],
            exec_types=runtime["rms_exec_types"]),
      check("stored_profile_has_exact_elementwise_boundary_counts",
            runtime["elementwise_counts_exact"],
            counts=runtime["elementwise_boundary_counts"]),
      check("seq1233_fc_schedule_is_complete_and_multi_workgroup",
            fc.get("required_checks_passed") is True and
            fc.get("route_stop_proven") is True and
            int(fc["aggregate"]["remaining_bytes_not_charged"]) == 0 and
            source["all_seq1233_fc_cohorts_use_multiple_workgroups"] and
            source["all_seq1233_fc_cohorts_have_local_barrier"],
            cohorts=source["fc_cohorts"]),
      check("rms_source_requires_reduction_dispatch_and_allows_epilogue",
            source["rms_reduction_and_dispatch_contract_present"] and
            source["rms_kernel_local_barrier_occurrences"] >= 3 and
            source["rms_kernel_supports_fused_epilogue"],
            local_barriers=source["rms_kernel_local_barrier_occurrences"]),
      check("gdn_output_schedule_is_partitioned_across_workgroups",
            source["gdn_output_schedule_is_partitioned"] and
            source["gdn_output_workgroups_per_head_at_decode"] == 32,
            workgroups_per_head=
                source["gdn_output_workgroups_per_head_at_decode"]),
      check("status_registers_corrected_complete_event_census",
            status_has_census),
      check("seq1238_is_closed_and_not_added_to_bundle",
            seq1238_closed,
            seq1238_ceiling_added_ms_per_token=0.0),
      check("duplicate_inclusive_stress_ceiling_is_below_remaining_cut",
            not route_fundable,
            stress_ceiling_ms_per_token=
                duplicate_inclusive_stress_ceiling_ms,
            remaining_required_ms_per_token=remaining_required_ms,
            residual_shortfall_ms_per_token=residual_shortfall_ms),
      check("memory_guard_never_tripped",
            all(row["available_bytes"] >= stop_bytes for row in memory),
            minimum_available_bytes=min(
                row["available_bytes"] for row in memory)),
  ]
  required_checks_passed = all(row["pass"] for row in checks)
  verdict = (
      "reject_decode_elementwise_residual_bundle_before_source"
      if required_checks_passed and not route_fundable else
      "admit_one_decode_elementwise_residual_component"
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
          "independent_elementwise_residual_ceiling_ms_per_token":
              independent_ceiling_ms,
          "duplicate_inclusive_stress_ceiling_ms_per_token":
              duplicate_inclusive_stress_ceiling_ms,
          "residual_shortfall_ms_per_token": residual_shortfall_ms,
      },
      "overlap_free_ceiling": {
          "complete_nonmajor_event_residual_ms_per_token": other_event_ms,
          "rms_event_bucket_added_ms_per_token": 0.0,
          "seq1238_ceiling_added_ms_per_token": 0.0,
          "provider_envelope_added_ms_per_token": 0.0,
          "rms_decode_elements": elements,
          "independent_rms_decode_elements_excluding_q_k":
              independent_rms_elements,
          "all_rms_decode_elements": all_rms_elements,
          "f16_write_plus_read_bytes_per_element": edge_bytes_per_element,
          "independent_rms_edge_bytes": independent_rms_edge_bytes,
          "duplicate_inclusive_all_rms_edge_bytes": all_rms_edge_bytes,
          "small_tensor_gbps": small_tensor_gbps,
          "independent_rms_edge_ms_per_token": independent_rms_edge_ms,
          "duplicate_inclusive_all_rms_edge_ms_per_token":
              duplicate_inclusive_rms_edge_ms,
          "independent_ceiling_ms_per_token": independent_ceiling_ms,
          "duplicate_inclusive_stress_ceiling_ms_per_token":
              duplicate_inclusive_stress_ceiling_ms,
          "union_rule": (
              "complete nonmajor event residual plus RMS normalized-output "
              "write/read only; retain one reduction-bearing dispatch and "
              "add neither seq1238 nor provider/device timelines"),
          "optimism": [
              "assigns the entire complete-model nonmajor event residual to the bundle",
              "treats required elementwise arithmetic and unrelated CPU shape work as free",
              "charges every removable RMS edge at the slow seq1237 small-tensor rate",
              "stress ceiling duplicates Q/K RMS edges already covered by seq1238",
              "charges zero implementation, synchronization, occupancy, or FC epilogue penalty",
          ],
      },
      "locked_ir_rms_audit": ir,
      "runtime_profile_audit": runtime,
      "source_schedule_audit": source,
      "checks": checks,
      "memory_stop_bytes": stop_bytes,
      "memory_samples": memory,
      "inputs": {display_path(path): sha256(path) for path in required},
  }
  (output / "metrics.json").write_text(
      json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
  summary = f"""# Decode elementwise/residual source bound

Verdict: **{verdict}**. Required evidence checks:
`{str(required_checks_passed).lower()}`. No compiler or GPU worker ran.

The locked IR and stored accepted runtime match exactly: `131` RMS nodes split
as 40 input, 40 post-attention, 30 linear-attention, 10 Q, 10 K, and one final
norm, all using `rms_gpu_bfyx_opt__f16`. The executed adjacent census is also
exact: 80 residual adds, 40 attention/linear gates, 30 Swish, 30 SoftPlus, and
the complete rotary/split/transpose consumer set.

The RMS source performs subgroup plus SLM reduction, three local barriers,
sqrt/reciprocal, affine apply, and output in one workgroup per normalized
vector. Its fused-op epilogue can absorb adjacent elementwise work, but one
reduction-bearing dispatch remains. Seq1233's FC cohorts all use many
workgroups, and GDN partitions each 128-value output head across 32 workgroups;
neither schedule provides a legal grid synchronization point for deleting that
reduction dependency.

Seq1233 already supplies `{fc_saving_ms:.6f} ms/token`; the bundle must add
`{remaining_required_ms:.6f} ms/token`. The bound gives it the *entire*
complete-model non-major event residual, `{other_event_ms:.6f} ms/token`, even
though that overcharges required arithmetic and unrelated CPU shape work. It
then deletes every independent RMS normalized-output write/read at seq1237's
slow `{small_tensor_gbps:.6f} GB/s` rate, adding
`{independent_rms_edge_ms:.6f} ms/token`. The independent ceiling is only
`{independent_ceiling_ms:.6f} ms/token`.

For a stronger rejection, the stress ceiling also duplicates all Q/K RMS edges
already covered by closed seq1238. It still reaches only
`{duplicate_inclusive_stress_ceiling_ms:.6f} ms/token`, short by
`{residual_shortfall_ms:.6f} ms/token`. Seq1238's `0.560-ms` envelope, the RMS
event bucket, and provider/device timelines are not added. Source, compile,
graph integration, 32k, ABBA, and output512 are not admitted for this route.
"""
  (output / "summary.md").write_text(summary, encoding="utf-8")
  print(json.dumps({
      "artifact": display_path(output),
      "verdict": verdict,
      "remaining_required_ms": remaining_required_ms,
      "independent_ceiling_ms": independent_ceiling_ms,
      "duplicate_inclusive_stress_ceiling_ms":
          duplicate_inclusive_stress_ceiling_ms,
      "residual_shortfall_ms": residual_shortfall_ms,
      "gpu_worker_launched": False,
  }, separators=(",", ":")))
  return 0 if required_checks_passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
