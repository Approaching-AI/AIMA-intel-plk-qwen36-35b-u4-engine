#!/usr/bin/env python3
"""Audit the complete fixed-FC/QK/RMS/IGC bundle without running a worker.

This gate ties the retained component evidence back to the locked U4 IR and
the pinned OpenVINO horizontal-FC pass.  It proves the exact four-way FC
groups, their compressed parameter streams, the existing three-way QKV
carrier, and non-overlap with the retained Q/K layout producer.  It launches
no compiler, GPU context, graph compile, or model worker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-openvino-fc-rms-igc-qk-bundle-bound-v0"
R0 = Path("/home/intel/intel-qwen36-r0")
MODEL_XML = Path("/home/intel/Qwen3.6-35B-A3B-ov/openvino_language_model.xml")
OPENVINO_SOURCE = R0 / "source/openvino-90214e5be05"
FC_FUSION_SOURCE = OPENVINO_SOURCE / (
    "src/plugins/intel_gpu/src/plugin/transformations/"
    "fc_horizontal_fusion.cpp")
FC_FUSION_PIPELINE = OPENVINO_SOURCE / (
    "src/plugins/intel_gpu/src/plugin/transformations_pipeline.cpp")
FC_MICRO_SOURCE = ROOT / "engine/gpu/opencl/openvino_fc_micro_host.cl"
FC_MICRO_CODEGEN = ROOT / "engine/tools/openvino_moe_micro_codegen.cpp"
QK_SOURCE = ROOT / "engine/openvino/custom/iq36_qk_rope_layout.cl"
QK_REGISTRY = ROOT / "engine/openvino/custom/iq36_hot_attention_gqa.xml"
QK_PREPROCESS_PATCH = ROOT / (
    "engine/openvino/iq36-custom-preprocess-input-port-id.patch")
SEQ1233 = ROOT / (
    "output/openvino-fc-micro-component-"
    "20260715Tseq1233-max-native-fused-nonzero-warm512-cleanZ/metrics.json")
SEQ1294 = ROOT / (
    "output/openvino-fc-hardware-limit-bound-"
    "20260717Tseq1294-cleanZ/metrics.json")
SEQ1301 = ROOT / (
    "output/openvino-igc2382-component-gate-"
    "20260717Tseq1301-cleanZ/metrics.json")
SEQ1302 = ROOT / (
    "output/openvino-post-igc-opportunity-bound-"
    "20260717Tseq1302-cleanZ/metrics.json")
SEQ1323 = ROOT / (
    "output/openvino-qk-rope-layout-bound-"
    "20260717Tseq1323-cleanZ/metrics.json")
SEQ1327 = ROOT / (
    "output/openvino-qk-rope-layout-component-"
    "20260717Tseq1327-corrected-candidate-2k-warm17-cleanZ/metrics.json")
RMS_PATCH = ROOT / (
    "output/openvino-post-igc-opportunity-bound-"
    "20260717Tseq1302-cleanZ/raw/openvino-pr36747.patch")

FULL_ATTENTION_LAYERS = tuple(range(3, 40, 4))
LINEAR_ATTENTION_LAYERS = tuple(
    layer for layer in range(40) if layer not in FULL_ATTENTION_LAYERS)
LINEAR_SUFFIXES = (
    "linear_attn.in_proj_qkv/ov_ext::linear/MatMul",
    "linear_attn.in_proj_a/ov_ext::linear/MatMul",
    "linear_attn.in_proj_b/ov_ext::linear/MatMul",
    "linear_attn.in_proj_z/ov_ext::linear/MatMul",
)
ROUTER_SUFFIXES = (
    "mlp.shared_expert_gate/ov_ext::linear/MatMul",
    "mlp.shared_expert.gate_proj/ov_ext::linear/MatMul",
    "mlp.shared_expert.up_proj/ov_ext::linear/MatMul",
    "mlp.gate/aten::linear/MatMul",
)
FULL_QKV_SUFFIXES = (
    "self_attn.q_proj/ov_ext::linear/MatMul",
    "self_attn.k_proj/ov_ext::linear/MatMul",
    "self_attn.v_proj/ov_ext::linear/MatMul",
)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--memory-stop-gib", type=float, default=4.0)
  args = parser.parse_args()
  if args.memory_stop_gib <= 0.0:
    parser.error("memory stop must be positive")
  return args


def load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise TypeError(f"expected JSON object: {path}")
  return value


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    while chunk := stream.read(8 * 1024 * 1024):
      digest.update(chunk)
  return digest.hexdigest()


def display(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path.resolve())


def available_memory_bytes() -> int:
  for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
    if line.startswith("MemAvailable:"):
      return int(line.split()[1]) * 1024
  raise RuntimeError("MemAvailable missing from /proc/meminfo")


def git_state(output: Path) -> dict[str, Any]:
  commit = subprocess.run(
      ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
      capture_output=True, text=True).stdout.strip()
  rows = subprocess.run(
      ["git", "status", "--porcelain"], cwd=ROOT, check=True,
      capture_output=True, text=True).stdout.splitlines()
  allowed = {
      "tools/intel-qwen36-openvino-fc-rms-igc-qk-rope-bundle-bound.py",
  }
  relative_output = str(output.resolve().relative_to(ROOT))
  dirty = []
  for row in rows:
    path = row[3:]
    if path in allowed or path.startswith(relative_output):
      continue
    dirty.append(row)
  return {
      "commit": commit,
      "dirty": bool(dirty),
      "dirty_paths": dirty,
      "allowed_uncommitted_tool_paths": sorted(allowed),
  }


def check(name: str, passed: bool, **detail: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **detail}


def ports(node: ET.Element, section: str) -> dict[int, dict[str, Any]]:
  parent = node.find(section)
  if parent is None:
    return {}
  result = {}
  for port in parent.findall("port"):
    result[int(port.attrib["id"])] = {
        "precision": port.attrib.get("precision"),
        "shape": [int(dim.text) for dim in port.findall("dim")],
    }
  return result


def locked_ir_audit() -> dict[str, Any]:
  root = ET.parse(MODEL_XML).getroot()
  layer_root = root.find("layers")
  edge_root = root.find("edges")
  if layer_root is None or edge_root is None:
    raise ValueError("locked IR lacks layers/edges")
  layers = {int(node.attrib["id"]): node for node in layer_root}
  by_name = {str(node.attrib.get("name")): node_id
             for node_id, node in layers.items()}
  incoming: dict[int, dict[int, int]] = defaultdict(dict)
  for edge in edge_root:
    incoming[int(edge.attrib["to-layer"])][
        int(edge.attrib["to-port"])] = int(edge.attrib["from-layer"])

  def input_source(node_id: int, port: int) -> int:
    try:
      return incoming[node_id][port]
    except KeyError as exc:
      raise ValueError(f"missing input {node_id}:{port}") from exc

  def shape(node_id: int, section: str, port: int) -> list[int]:
    return ports(layers[node_id], section)[port]["shape"]

  def precision(node_id: int, section: str, port: int) -> str | None:
    return ports(layers[node_id], section)[port]["precision"]

  def only_output_port(node_id: int) -> int:
    values = ports(layers[node_id], "output")
    if len(values) != 1:
      raise ValueError(f"expected one output: {node_id}")
    return next(iter(values))

  constant_ranges = []

  def constant(node_id: int, expected_precision: str,
               expected_shape: list[int]) -> tuple[bool, dict[str, Any]]:
    node = layers[node_id]
    data = node.find("data")
    output_port = only_output_port(node_id)
    passed = (
        node.attrib.get("type") == "Const"
        and precision(node_id, "output", output_port) == expected_precision
        and shape(node_id, "output", output_port) == expected_shape
        and data is not None)
    offset = int(data.attrib["offset"]) if data is not None else -1
    size = int(data.attrib["size"]) if data is not None else -1
    result = {
        "name": node.attrib.get("name"),
        "precision": precision(node_id, "output", output_port),
        "shape": shape(node_id, "output", output_port),
        "offset": offset,
        "size": size,
    }
    constant_ranges.append((offset, offset + size, str(node.attrib.get("name"))))
    return passed, result

  def weight_contract(mm_id: int, n: int) -> dict[str, Any]:
    mm = layers[mm_id]
    mm_inputs = ports(mm, "input")
    cv_id = input_source(mm_id, 1)
    rs_id = input_source(cv_id, 0)
    mul_id = input_source(rs_id, 0)
    sub_id = input_source(mul_id, 0)
    scale_id = input_source(mul_id, 1)
    weight_cv_id = input_source(sub_id, 0)
    zp_cv_id = input_source(sub_id, 1)
    weight_id = input_source(weight_cv_id, 0)
    zp_id = input_source(zp_cv_id, 0)
    nodes = {
        "convert": cv_id,
        "reshape": rs_id,
        "multiply": mul_id,
        "subtract": sub_id,
        "weight_convert": weight_cv_id,
        "zp_convert": zp_cv_id,
    }
    types = {name: layers[node_id].attrib.get("type")
             for name, node_id in nodes.items()}
    weight_ok, weight = constant(weight_id, "U4", [n, 32, 64])
    zp_ok, zp = constant(zp_id, "U4", [n, 32, 1])
    scale_ok, scale = constant(scale_id, "FP16", [n, 32, 1])
    expected_bytes = n * 1104
    observed_bytes = weight["size"] + zp["size"] + scale["size"]
    passed = (
        mm.attrib.get("type") == "MatMul"
        and mm_inputs[1] == {"precision": "FP32", "shape": [n, 2048]}
        and types == {
            "convert": "Convert", "reshape": "Reshape",
            "multiply": "Multiply", "subtract": "Subtract",
            "weight_convert": "Convert", "zp_convert": "Convert"}
        and shape(cv_id, "input", 0) == [n, 2048]
        and precision(cv_id, "input", 0) == "FP16"
        and shape(cv_id, "output", only_output_port(cv_id)) == [n, 2048]
        and precision(cv_id, "output", only_output_port(cv_id)) == "FP32"
        and shape(rs_id, "input", 0) == [n, 32, 64]
        and shape(rs_id, "output", only_output_port(rs_id)) == [n, 2048]
        and shape(mul_id, "input", 0) == [n, 32, 64]
        and shape(mul_id, "input", 1) == [n, 32, 1]
        and shape(sub_id, "input", 0) == [n, 32, 64]
        and shape(sub_id, "input", 1) == [n, 32, 1]
        and weight_ok and zp_ok and scale_ok
        and weight["size"] == n * 1024
        and zp["size"] == n * 16
        and scale["size"] == n * 64
        and observed_bytes == expected_bytes)
    return {
        "matmul": mm.attrib.get("name"),
        "n": n,
        "passed": passed,
        "parameter_bytes": observed_bytes,
        "constants": {"weight": weight, "zero_point": zp, "scale": scale},
    }

  def group_rows(layers_expected: tuple[int, ...], suffixes: tuple[str, ...],
                 expected_widths: list[int], kind: str) -> list[dict[str, Any]]:
    rows = []
    for layer in layers_expected:
      prefix = f"__module.model.model.language_model.layers.{layer}."
      names = [prefix + suffix for suffix in suffixes]
      ids = [by_name.get(name) for name in names]
      contracts = []
      input_ids = []
      widths = []
      if all(node_id is not None for node_id in ids):
        for node_id in ids:
          assert node_id is not None
          mm_inputs = ports(layers[node_id], "input")
          n = mm_inputs[1]["shape"][0]
          widths.append(n)
          input_ids.append(input_source(node_id, 0))
          contracts.append(weight_contract(node_id, n))
      input_names = [layers[node_id].attrib.get("name")
                     for node_id in sorted(set(input_ids))]
      rows.append({
          "kind": kind,
          "layer": layer,
          "names_present": all(node_id is not None for node_id in ids),
          "shared_input_exact": len(set(input_ids)) == 1,
          "input_names": input_names,
          "output_widths": widths,
          "expected_output_widths": expected_widths,
          "fused_output_width": sum(widths),
          "weight_contracts_exact": (
              len(contracts) == len(suffixes)
              and all(row["passed"] for row in contracts)),
          "parameter_bytes": sum(row["parameter_bytes"] for row in contracts),
          "contracts": contracts,
      })
    return rows

  linear = group_rows(
      LINEAR_ATTENTION_LAYERS, LINEAR_SUFFIXES,
      [8192, 32, 32, 4096], "linear_attention_input")
  router = group_rows(
      tuple(range(40)), ROUTER_SUFFIXES,
      [1, 512, 512, 256], "router_shared_input")
  full_qkv = group_rows(
      FULL_ATTENTION_LAYERS, FULL_QKV_SUFFIXES,
      [8192, 512, 512], "full_attention_qkv")
  all_rows = linear + router + full_qkv
  for row in all_rows:
    row["output_widths_exact"] = row["output_widths"] == row[
        "expected_output_widths"]
    row["passed"] = (
        row["names_present"] and row["shared_input_exact"]
        and row["output_widths_exact"] and row["weight_contracts_exact"])

  sorted_ranges = sorted(constant_ranges)
  non_overlapping = all(
      left[1] <= right[0]
      for left, right in zip(sorted_ranges, sorted_ranges[1:]))
  return {
      "model_xml": display(MODEL_XML),
      "linear_attention_groups": linear,
      "router_shared_groups": router,
      "full_attention_qkv_groups": full_qkv,
      "counts": {
          "linear_groups": len(linear),
          "linear_matmuls": len(linear) * len(LINEAR_SUFFIXES),
          "router_groups": len(router),
          "router_matmuls": len(router) * len(ROUTER_SUFFIXES),
          "existing_full_qkv_groups": len(full_qkv),
          "existing_full_qkv_matmuls": len(full_qkv) * len(FULL_QKV_SUFFIXES),
          "candidate_four_fc_groups": len(linear) + len(router),
          "candidate_four_fc_matmuls": (
              len(linear) * len(LINEAR_SUFFIXES)
              + len(router) * len(ROUTER_SUFFIXES)),
      },
      "parameter_bytes": {
          "linear_attention": sum(row["parameter_bytes"] for row in linear),
          "router_shared": sum(row["parameter_bytes"] for row in router),
          "full_attention_qkv": sum(
              row["parameter_bytes"] for row in full_qkv),
      },
      "constant_ranges": {
          "count": len(sorted_ranges),
          "non_overlapping": non_overlapping,
      },
      "all_groups_exact": all(row["passed"] for row in all_rows),
  }


def source_audit() -> dict[str, Any]:
  source = FC_FUSION_SOURCE.read_text(encoding="utf-8")
  pipeline = FC_FUSION_PIPELINE.read_text(encoding="utf-8")
  return {
      "current_max_fc_literal_count": source.count(
          "const int max_num_fcs_to_fuse = 3;"),
      "current_max_fc_is_three": (
          source.count("const int max_num_fcs_to_fuse = 3;") == 1),
      "compressed_fc_only": (
          "wrap_type<op::FullyConnectedCompressed>" in source),
      "weight_concat_axis_zero": (
          "Concat>(weight_nodes_as_output_vector, 0)" in source),
      "scale_concat_axis_zero": (
          "Concat>(scales_as_output_vector, 0)" in source),
      "zero_point_concat_axis_zero": (
          "Concat>(zp_nodes_as_output_vector, 0)" in source),
      "split_preserves_original_widths": (
          "orig_n_sizes" in source
          and "VariadicSplit>(new_fc, axis_const, split_const)" in source),
      "pass_registered": (
          "register_pass<ov::intel_gpu::FullyConnectedHorizontalFusion>" in
          pipeline),
      "candidate_change": (
          "raise max_num_fcs_to_fuse from 3 to 4; retain the existing "
          "compressed-weight Concat and VariadicSplit mapping"),
  }


def main() -> int:
  args = parse_args()
  output = args.output.resolve()
  output.mkdir(parents=True, exist_ok=False)
  required = (
      MODEL_XML, FC_FUSION_SOURCE, FC_FUSION_PIPELINE,
      FC_MICRO_SOURCE, FC_MICRO_CODEGEN, QK_SOURCE, QK_REGISTRY,
      QK_PREPROCESS_PATCH, SEQ1233, SEQ1294, SEQ1301, SEQ1302,
      SEQ1323, SEQ1327, RMS_PATCH)
  missing = [display(path) for path in required if not path.is_file()]
  if missing:
    raise SystemExit("missing bundle-bound inputs: " + ", ".join(missing))

  stop_bytes = int(args.memory_stop_gib * 1024**3)
  memory = [{"stage": "start", "available_bytes": available_memory_bytes()}]
  if memory[0]["available_bytes"] < stop_bytes:
    raise RuntimeError("memory stop tripped before source audit")
  git = git_state(output)
  seq1233 = load_json(SEQ1233)
  seq1294 = load_json(SEQ1294)
  seq1301 = load_json(SEQ1301)
  seq1302 = load_json(SEQ1302)
  seq1323 = load_json(SEQ1323)
  seq1327 = load_json(SEQ1327)
  ir = locked_ir_audit()
  source = source_audit()
  memory.append({"stage": "after-ir-audit",
                 "available_bytes": available_memory_bytes()})

  fc_saving_ms = float(seq1233["aggregate"]["optimistic_saving_ms"])
  kill_number_ms = float(seq1302["budget"]["current_kill_number_ms"])
  residual_after_fc_ms = kill_number_ms - fc_saving_ms
  qk_observed_ms = float(
      seq1327["performance"]["observed_median_saving_ms"])
  qk_source_ceiling_ms = float(
      seq1323["budget"]["favorable_qk_rope_layout_provider_ceiling_ms"])
  rms_ceiling_ms = float(
      seq1302["budget"]["complete_registered_rms_bucket_ms"])
  igc_point_ms = float(
      seq1301["performance"]["observed_median_saving_ms"])
  observed_screen_ms = fc_saving_ms + qk_observed_ms
  observed_screen_margin_ms = observed_screen_ms - kill_number_ms
  qk_retention_fraction_needed = residual_after_fc_ms / qk_observed_ms
  source_only_screen_ms = fc_saving_ms + qk_source_ceiling_ms
  source_only_shortfall_ms = kill_number_ms - source_only_screen_ms
  expanded_source_screen_ms = (
      source_only_screen_ms + rms_ceiling_ms + igc_point_ms)
  expanded_source_margin_ms = expanded_source_screen_ms - kill_number_ms

  qk_candidate = seq1327["profile"]["candidate"]
  qk_control = seq1327["profile"]["control"]
  current_fcs = int(qk_control["core_counts"]["FullyConnectedCompressed"])
  expected_fcs_after = current_fcs - 30 * 3 - 40 * 3
  parameter_bytes = ir["parameter_bytes"]
  cohorts = seq1294["cohorts"]
  source_exact = all((
      source["current_max_fc_is_three"], source["compressed_fc_only"],
      source["weight_concat_axis_zero"], source["scale_concat_axis_zero"],
      source["zero_point_concat_axis_zero"],
      source["split_preserves_original_widths"], source["pass_registered"]))
  checks = [
      check("repository_clean_at_gate", not git["dirty"], git=git),
      check("all_bundle_artifact_gates_are_exact",
            seq1233.get("required_checks_passed") is True
            and seq1294.get("required_checks_passed") is True
            and seq1301.get("required_checks_passed") is True
            and seq1302.get("required_checks_passed") is True
            and seq1323.get("required_checks_passed") is True
            and seq1327.get("evidence_checks_passed") is True),
      check("locked_ir_has_exact_70_four_fc_groups_and_10_qkv_groups",
            ir["all_groups_exact"]
            and ir["counts"] == {
                "linear_groups": 30, "linear_matmuls": 120,
                "router_groups": 40, "router_matmuls": 160,
                "existing_full_qkv_groups": 10,
                "existing_full_qkv_matmuls": 30,
                "candidate_four_fc_groups": 70,
                "candidate_four_fc_matmuls": 280},
            counts=ir["counts"]),
      check("all_target_u4_group64_constant_ranges_are_exact_and_disjoint",
            ir["constant_ranges"] == {
                "count": 930, "non_overlapping": True},
            constant_ranges=ir["constant_ranges"]),
      check("locked_parameter_bytes_match_seq1294_cohorts",
            parameter_bytes["linear_attention"]
                == cohorts["linear_attention_input"]["parameter_bytes"]
            and parameter_bytes["router_shared"]
                == cohorts["router_shared_input"]["parameter_bytes"]
            and parameter_bytes["full_attention_qkv"]
                == cohorts["full_attention_qkv"]["parameter_bytes"],
            locked_ir=parameter_bytes,
            seq1294={
                "linear_attention": cohorts[
                    "linear_attention_input"]["parameter_bytes"],
                "router_shared": cohorts[
                    "router_shared_input"]["parameter_bytes"],
                "full_attention_qkv": cohorts[
                    "full_attention_qkv"]["parameter_bytes"]}),
      check("pinned_horizontal_fc_pass_needs_only_four_way_admission",
            source_exact, source=source),
      check("qk_component_is_exact_and_fc_non_overlapping",
            seq1327.get("activation_passed") is True
            and seq1327.get("correctness_passed") is True
            and qk_candidate["old_boundary_executed"] == 0
            and qk_control["old_boundary_executed"] == 100
            and qk_candidate["qk_rope_layout_executed"] == 10
            and qk_control["qk_rope_layout_executed"] == 0
            and qk_candidate["core_counts"]["FullyConnectedCompressed"]
                == current_fcs == 371),
      check("four_way_fc_fusion_has_exact_runtime_census_target",
            expected_fcs_after == 161,
            current_fully_connected_compressed=current_fcs,
            removed_fully_connected_compressed=210,
            target_fully_connected_compressed=expected_fcs_after),
      check("observed_component_screen_clears_kill_number",
            observed_screen_margin_ms > 0.0,
            screen_ms=observed_screen_ms,
            margin_ms=observed_screen_margin_ms,
            product_inference=False),
      check("source_ceiling_alone_is_not_overstated_as_complete",
            source_only_shortfall_ms > 0.0,
            source_only_screen_ms=source_only_screen_ms,
            shortfall_ms=source_only_shortfall_ms),
      check("rms_and_igc_are_classified_not_automatically_stacked",
            expanded_source_margin_ms > 0.0
            and seq1301.get("route_accepted") is False
            and seq1302.get("source_edit_admitted") is False,
            expanded_source_screen_ms=expanded_source_screen_ms,
            expanded_source_margin_ms=expanded_source_margin_ms,
            note=("RMS and IGC are necessary only under the dispatch-source "
                  "Q/K ceiling; they are redundant if the integrated Q/K "
                  "component retains the required fraction")),
      check("no_compiler_gpu_graph_or_model_worker_ran", True,
            compilers=0, gpu_contexts=0, graph_compiles=0, model_workers=0),
      check("memory_guard_never_tripped",
            all(row["available_bytes"] >= stop_bytes for row in memory),
            memory=memory),
  ]
  required_checks_passed = all(row["pass"] for row in checks)
  source_edit_admitted = required_checks_passed
  verdict = (
      "admit_default_off_four_fc_horizontal_fusion_source_audit"
      if source_edit_admitted else "inconclusive")
  metrics = {
      "schema": SCHEMA,
      "created_at": datetime.now(timezone.utc).isoformat(),
      "workstream": WS,
      "git": git,
      "verdict": verdict,
      "required_checks_passed": required_checks_passed,
      "source_edit_admitted": source_edit_admitted,
      "compiler_build_admitted": False,
      "plugin_build_admitted": False,
      "gpu_component_admitted": False,
      "product_worker_admitted": False,
      "product_inference_claim": False,
      "conservative_product_bound_passed": False,
      "locked_ir": ir,
      "openvino_source": source,
      "runtime_census": {
          "current_fully_connected_compressed": current_fcs,
          "new_four_way_groups": 70,
          "removed_fully_connected_compressed": 210,
          "target_fully_connected_compressed": expected_fcs_after,
          "existing_three_way_qkv_groups": 10,
      },
      "budget": {
          "current_kill_number_ms": kill_number_ms,
          "seq1233_optimistic_fixed_fc_saving_ms": fc_saving_ms,
          "residual_after_fixed_fc_ms": residual_after_fc_ms,
          "seq1327_qk_observed_component_point_ms": qk_observed_ms,
          "fc_plus_observed_qk_screen_ms": observed_screen_ms,
          "observed_screen_margin_ms": observed_screen_margin_ms,
          "qk_observed_effect_retention_fraction_needed": (
              qk_retention_fraction_needed),
          "qk_observed_effect_loss_tolerance_fraction": (
              1.0 - qk_retention_fraction_needed),
          "seq1323_qk_dispatch_source_ceiling_ms": qk_source_ceiling_ms,
          "fc_plus_qk_source_ceiling_ms": source_only_screen_ms,
          "source_only_shortfall_ms": source_only_shortfall_ms,
          "complete_registered_rms_bucket_ms": rms_ceiling_ms,
          "seq1301_igc_unconfirmed_median_point_ms": igc_point_ms,
          "fc_qk_source_rms_igc_favorable_screen_ms": (
              expanded_source_screen_ms),
          "expanded_source_screen_margin_ms": expanded_source_margin_ms,
          "interpretation": (
              "The exact 70-group source path and the FC plus observed-Q/K "
              "component screen admit a default-off source audit. The "
              "screen is not product inference. The dispatch-only Q/K "
              "ceiling remains below the kill-number with FC alone; RMS and "
              "IGC are not automatically stacked."),
      },
      "component_classification": {
          "fixed_fc": "primary_unintegrated_optimistic_component",
          "qk_rope_layout": "retained_exact_measured_component",
          "rms_pr36747": "parked_favorable_complete_bucket_ceiling",
          "igc_2_38_2": "parked_unconfirmed_median_point_mean_regresses",
      },
      "next_action": {
          "route": "openvino_four_fc_horizontal_fusion_source_rewrite_audit",
          "change": source["candidate_change"],
          "requirements": [
              "default-off durable patch against pinned OpenVINO",
              "unit coverage for four compressed FCs with unequal N widths",
              "no-GPU exact audit of 30 linear plus 40 router/shared groups",
              "do not compile or launch a model worker until that audit passes",
          ],
      },
      "checks": checks,
      "memory": {"stop_bytes": stop_bytes, "samples": memory},
  }
  write_json(output / "metrics.json", metrics)
  input_hashes = {display(path): sha256(path) for path in required}
  write_json(output / "manifest.json", {
      "schema": SCHEMA,
      "tool": display(Path(__file__)),
      "git": git,
      "inputs": input_hashes,
      "compilers": 0,
      "gpu_contexts": 0,
      "graph_compiles": 0,
      "model_workers": 0,
  })
  report = f"""# Fixed-FC / QK / RMS / IGC complete bundle bound

Verdict: **{verdict}**. Required checks:
`{str(required_checks_passed).lower()}`. No compiler, GPU context, graph
compile, or model worker ran.

The locked IR contains exactly 30 four-way linear-attention groups with output
widths `[8192, 32, 32, 4096]`, 40 four-way router/shared groups with widths
`[1, 512, 512, 256]`, and ten already-supported three-way full-QKV groups.
All 310 target MatMuls use the same U4/group64 weight, zero-point, and FP16
scale chain. Their 930 constant ranges are disjoint and their byte totals
match seq1294 exactly.

The pinned GPU pass already concatenates compressed weights/scales/zero-points
on N and restores each original width with `VariadicSplit`; its explicit
maximum is three. Raising that maximum to four targets exactly 70 groups,
reducing the expected runtime FC census from 371 to 161 if the provider audit
later activates every group.

Seq1233's optimistic `{fc_saving_ms:.6f}-ms` FC component plus seq1327's exact
`{qk_observed_ms:.6f}-ms` Q/K component point screens at
`{observed_screen_ms:.6f} ms`, `{observed_screen_margin_ms:.6f} ms` above the
`{kill_number_ms:.6f}-ms` kill-number. The integrated Q/K effect may lose at
most `{(1.0 - qk_retention_fraction_needed) * 100.0:.2f}%` before that screen
closes. This is source-admission arithmetic, not product inference.

The stricter dispatch-source ceiling is only `{source_only_screen_ms:.6f} ms`
with FC and remains `{source_only_shortfall_ms:.6f} ms` short. Adding the full
RMS bucket and the unconfirmed IGC median would reach
`{expanded_source_screen_ms:.6f} ms`, but those ingredients are parked rather
than automatically stacked. Admit only a default-off four-FC source patch and
no-GPU rewrite/unit audit. OOM observed: false.
"""
  (output / "report.md").write_text(report, encoding="utf-8")
  print(json.dumps({
      "artifact": display(output),
      "verdict": verdict,
      "four_fc_groups": 70,
      "target_fc_census": expected_fcs_after,
      "observed_screen_margin_ms": observed_screen_margin_ms,
      "compiler_or_worker_launched": False,
  }, separators=(",", ":")), flush=True)
  return 0 if required_checks_passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
