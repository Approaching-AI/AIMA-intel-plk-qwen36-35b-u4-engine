#!/usr/bin/env python3
"""Source-bound the single-owner OpenVINO adaptive-attention integration."""

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
SCHEMA = "intel-qwen36-openvino-adaptive-attention-integration-bound-v0"

LAYERS = (3, 7, 11, 15, 19, 23, 27, 31, 35, 39)
HIGH_TOPK_LAYERS = (3, 7)
BASE_CONTEXT = 32768
TARGET_CONTEXT = 65536
OUTPUT_TOKENS = 512
HOT_TOKENS = 16384
CHUNK_TOKENS = 512
LOCAL_TOPK = 64
HIGH_TOPK = 512
LOW_TOPK = 256
QUERY_HEADS = 16
KV_HEADS = 2
GQA_GROUP = QUERY_HEADS // KV_HEADS
HEAD_DIM = 256
QUANT_GROUP = 32
EXACT_HISTORY_CAPACITY = 66560
F16_BYTES = 2
F32_BYTES = 4
MIB = 1024 * 1024

COMPONENT = ROOT / (
    "output/openvino-adaptive-attention-component-"
    "20260720Tseq1673-clean/result.json")
COMPONENT_BOUND = ROOT / (
    "output/openvino-adaptive-attention-component-bound-"
    "20260720Tseq1672-clean/bound.json")
DIRECT_INTEGRATION = ROOT / (
    "output/openvino-direct-i8-integration-"
    "20260715Tseq1251-layer3-32k-cleanZ/metrics.json")
PRODUCT_WORKER = ROOT / (
    "output/openvino-64k-q4-adaptive-correction-"
    "20260720Tseq1652-all10-o512-abba1/raw/sentinel_064k/block00/"
    "candidate-b1/worker-result.json")
REJECTED = ROOT / "doc/active" / WS / "rejected-routes.json"
STATUS = ROOT / "doc/active" / WS / "STATUS.md"

GRAPH = ROOT / "tools/intel_qwen36_openvino_hot_cold_attention.py"
HELPERS = ROOT / "engine/openvino/custom/iq36_hot_attention_tiled_helpers.cl"
PREFILL = ROOT / "engine/openvino/custom/iq36_prefill_attention_tiled.cl"
OWNER = ROOT / "engine/openvino/custom/iq36_hot_attention_single_owner.cl"
XML = ROOT / "engine/openvino/custom/iq36_hot_attention_gqa.xml"
COMPONENT_SOURCE = ROOT / "engine/gpu/opencl/direct_i8_hotcold_gqa_decode.cl"

OV_ROOT = Path(
    "/home/intel/intel-qwen36-r0/source/openvino-90214e5be05")
OV_COMMIT = "90214e5be05"
OV_CUSTOM_LAYER = OV_ROOT / "src/plugins/intel_gpu/src/plugin/custom_layer.cpp"
OV_CUSTOM_OP = OV_ROOT / "src/plugins/intel_gpu/src/plugin/ops/custom.cpp"
OV_CUSTOM_IMPL = OV_ROOT / (
    "src/plugins/intel_gpu/src/graph/impls/ocl/custom_primitive.cpp")
OV_MULTI_STAGE = OV_ROOT / (
    "src/plugins/intel_gpu/src/graph/impls/ocl/multi_stage_primitive.hpp")
OV_PRIMITIVE_BASE = OV_ROOT / (
    "src/plugins/intel_gpu/src/graph/impls/ocl/primitive_base.hpp")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out-dir", type=Path, required=True)
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


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    while chunk := stream.read(1024 * 1024):
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
  raise RuntimeError("MemAvailable is missing")


def git_state(out_dir: Path) -> dict[str, Any]:
  commit = subprocess.run(
      ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
      capture_output=True, text=True).stdout.strip()
  rows = subprocess.run(
      ["git", "status", "--porcelain"], cwd=ROOT, check=True,
      capture_output=True, text=True).stdout.splitlines()
  try:
    out_rel = str(out_dir.resolve().relative_to(ROOT))
  except ValueError:
    out_rel = ""
  rows = [row for row in rows if not out_rel or out_rel not in row]
  return {"commit": commit, "dirty": bool(rows), "dirty_paths": rows}


def check(name: str, passed: bool, **details: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **details}


def named_check(payload: dict[str, Any], name: str) -> dict[str, Any]:
  return next(
      (row for row in payload.get("checks", []) if row.get("name") == name),
      {})


def workspace_bytes(context_tokens: int) -> dict[str, Any]:
  cold_tokens = max(0, context_tokens - HOT_TOKENS)
  chunks = math.ceil(context_tokens / CHUNK_TOKENS)
  cold_chunks = math.ceil(cold_tokens / CHUNK_TOKENS)
  meta_values = KV_HEADS * chunks * GQA_GROUP
  correction_meta = KV_HEADS * cold_chunks * GQA_GROUP
  rows = {
      "partial_max": meta_values * F32_BYTES,
      "partial_sum": meta_values * F32_BYTES,
      "partial_numerator": meta_values * HEAD_DIM * F32_BYTES,
      "approximate_cold_scores": QUERY_HEADS * cold_tokens * F16_BYTES,
      "local_candidates": (
          QUERY_HEADS * cold_chunks * LOCAL_TOPK * F32_BYTES),
      "union_bitsets": KV_HEADS * math.ceil(cold_tokens / 32) * F32_BYTES,
      "aggregate_max": QUERY_HEADS * F32_BYTES,
      "aggregate_sum": QUERY_HEADS * F32_BYTES,
      "aggregate_numerator": QUERY_HEADS * HEAD_DIM * F32_BYTES,
      "correction_partial_max": correction_meta * F32_BYTES,
      "correction_partial_sum": correction_meta * F32_BYTES,
      "correction_partial_numerator": (
          correction_meta * HEAD_DIM * F32_BYTES),
      "correction_completion": KV_HEADS * F32_BYTES,
      "attention_output": QUERY_HEADS * HEAD_DIM * F32_BYTES,
  }
  raw_bytes = sum(rows.values())
  aligned_bytes = math.ceil(raw_bytes / 64) * 64
  return {
      "context_tokens": context_tokens,
      "cold_tokens": cold_tokens,
      "chunk_count": chunks,
      "cold_chunk_count": cold_chunks,
      "buffers": rows,
      "raw_bytes": raw_bytes,
      "aligned_bytes": aligned_bytes,
      "aligned_mib": aligned_bytes / MIB,
      "packed_f32_elements": math.ceil(aligned_bytes / F32_BYTES),
  }


def source_contract() -> tuple[dict[str, bool], list[dict[str, str]]]:
  paths = (
      GRAPH, HELPERS, PREFILL, OWNER, XML, COMPONENT_SOURCE,
      OV_CUSTOM_LAYER, OV_CUSTOM_OP, OV_CUSTOM_IMPL, OV_MULTI_STAGE,
      OV_PRIMITIVE_BASE)
  texts = {path: path.read_text(encoding="utf-8") for path in paths}
  graph = texts[GRAPH]
  helpers = texts[HELPERS]
  prefill = texts[PREFILL]
  owner = texts[OWNER]
  xml = texts[XML]
  component = texts[COMPONENT_SOURCE]
  custom_layer = texts[OV_CUSTOM_LAYER]
  custom_op = texts[OV_CUSTOM_OP]
  custom_impl = texts[OV_CUSTOM_IMPL]
  multi_stage = texts[OV_MULTI_STAGE]
  primitive_base = texts[OV_PRIMITIVE_BASE]
  checks = {
      "direct_i8_and_exact_history_are_composable": (
          "direct_i8_fixed_layout" in graph
          and "exact_history_layers" in graph
          and "physical_ring_capacities" in graph
          and not any(token in graph for token in (
              "direct_i8_fixed_layout and exact_history_layers",
              "exact_history_layers and direct_i8_fixed_layout"))),
      "packed_cold_and_dense_k_planes_have_existing_helpers": all(
          token in helpers for token in (
              "iq36_direct_cold_key_payload",
              "iq36_direct_cold_value_payload",
              "iq36_direct_cold_key_scale_payload",
              "iq36_direct_cold_value_scale_payload",
              "iq36_hot_key_dense_i32_base")),
      "dimension_major_v_plane_has_existing_prefill_and_update_writers": (
          "iq36_hot_value_dimension_i32_base" in helpers
          and "iq36_direct_store_hot_value_dimension" in prefill
          and "iq36_direct_store_hot_value_dimension" in owner),
      "adaptive_component_has_exactly_four_product_entrypoints": all(
          token in component for token in (
              "iq36_direct_i8_hotcold_partial",
              "iq36_adaptive_select_reduce_union",
              "iq36_adaptive_correct_normalize",
              "iq36_direct_i8_update_state")),
      "adaptive_component_uses_required_layouts": all(
          token in component for token in (
              "IQ36_ADAPTIVE_LOCAL_TOPK 64U",
              "IQ36_ADAPTIVE_CORRECTION_PARTITIONS",
              "iq36_adaptive_ordered_half",
              "approximate_cold_score")),
      "current_xml_direct_i8_abi_has_six_outputs": (
          "IQ36DirectI8HotAttentionGQA" in xml
          and xml.count('type="output" port-index=') >= 6),
      "current_simplegpu_xml_is_one_kernel_only": (
          "Multiple definition of Kernel" in custom_layer
          and "ProcessKernelNode(node.child(\"Kernel\"))" in custom_layer),
      "current_simplegpu_runtime_enqueues_only_front_kernel": (
          "enqueue_kernel(*_kernels.front()" in custom_impl
          and "get_kernels_source() override" in custom_impl),
      "dynamic_multi_output_binding_is_live": (
          "op->get_output_size()" in custom_op
          and "outputFormats.size() == op->get_output_size()" in custom_op),
      "plugin_has_proven_multistage_scheduling_primitives": all(
          token in multi_stage for token in (
              "multiple kernel scheduled",
              "get_internal_buffer_descs")) and all(
          token in primitive_base for token in (
              "needs_sub_kernels_sync",
              "stream.enqueue_kernel")),
  }
  files = [{"path": display(path), "sha256": sha256(path)} for path in paths]
  return checks, files


def main() -> int:
  args = parse_args()
  out_dir = args.out_dir.resolve()
  if out_dir.exists():
    raise SystemExit(f"output already exists: {out_dir}")
  required = (
      COMPONENT, COMPONENT_BOUND, DIRECT_INTEGRATION, PRODUCT_WORKER,
      REJECTED, STATUS, GRAPH, HELPERS, PREFILL, OWNER, XML,
      COMPONENT_SOURCE, OV_CUSTOM_LAYER, OV_CUSTOM_OP, OV_CUSTOM_IMPL,
      OV_MULTI_STAGE, OV_PRIMITIVE_BASE)
  missing = [str(path) for path in required if not path.is_file()]
  if missing:
    raise SystemExit("missing integration-bound inputs: " + ", ".join(missing))

  stop_bytes = int(args.memory_stop_gib * 1024**3)
  memory_before = available_memory_bytes()
  if memory_before < stop_bytes:
    raise SystemExit(
        f"memory stop: {memory_before} < {stop_bytes} bytes")
  out_dir.mkdir(parents=True, exist_ok=False)

  repository = git_state(out_dir)
  component = load_json(COMPONENT)
  component_bound = load_json(COMPONENT_BOUND)
  direct = load_json(DIRECT_INTEGRATION)
  product = load_json(PRODUCT_WORKER)
  rejected = load_json(REJECTED)
  status = STATUS.read_text(encoding="utf-8")
  sources, source_files = source_contract()
  ov_head = subprocess.run(
      ["git", "rev-parse", "HEAD"], cwd=OV_ROOT, check=True,
      capture_output=True, text=True).stdout.strip()

  direct_source_check = named_check(
      direct, "source_replaces_exact_selected_sdpa_and_kv_states")
  direct_source_rows = direct_source_check.get("source", [])
  direct_source = direct_source_rows[0] if len(direct_source_rows) == 1 else {}
  runtime_check = named_check(
      direct, "runtime_executes_exact_custom_and_stock_attention_counts")
  runtime_rows = [
      row for lane in runtime_check.get("runtime", []) for row in lane]
  direct_reorders = [
      row.get("node_name", "") for row in runtime_rows
      if row.get("layer_type") == "Reorder"]
  state_reorders = [
      name for name in direct_reorders
      if "iq36.hot." in name or "iq36.cold." in name]

  split_owner = next((
      row for row in rejected.get("rejected", [])
      if row.get("route") ==
          "openvino_attention_two_program_split_state_owner_v30ag"), {})
  product_source = product.get("source_summary", {})
  expected_layers = list(LAYERS)

  component_exact = bool(
      component.get("required_checks_passed") is True
      and component.get("component_promoted") is True
      and component.get("verdict") ==
          "promote_adaptive_attention_standalone_component"
      and component.get("git", {}).get("dirty") is False
      and math.isclose(
          float(component.get("weighted_ucb_ms", math.inf)),
          7.210838, rel_tol=0.0, abs_tol=1.0e-9)
      and math.isclose(
          float(component.get("weighted_cap_ms", -math.inf)),
          7.38283415319747, rel_tol=0.0, abs_tol=1.0e-9))
  direct_exact = bool(
      direct.get("required_checks_passed") is True
      and direct.get("direct_i8_fixed_layout") is True
      and direct_source_check.get("pass") is True
      and direct_source.get("direct_i8_fixed_layout") is True
      and direct_source.get("cold_storage") ==
          "fixed-capacity in-place block16-token/block32-dimension packed "
          "I8 K plus dimension-major I8 V and group-major exact F16 scales"
      and len(direct_source.get("custom_states", [])) == 6
      and runtime_check.get("pass") is True
      and len(direct_reorders) == 4
      and not state_reorders)
  product_sidecars = bool(
      product_source.get("custom_count_after") == len(LAYERS)
      and product_source.get("exact_history_layers") == expected_layers
      and int(product_source.get("exact_history_capacity", 0)) >=
          EXACT_HISTORY_CAPACITY
      and int(product_source.get("fixed_cold_capacity", 0)) >= TARGET_CONTEXT
      and int(product_source.get("hot_key_storage_planes", 0)) == 2
      and product_source.get("hot_value_shape", [0, 0, 0])[2] >=
          EXACT_HISTORY_CAPACITY
      and len(product_source.get("custom_states", [])) == 6 * len(LAYERS))
  product_layout_requires_replacement = bool(
      product_source.get("direct_i8_fixed_layout") is False
      and product_source.get("cold_storage") ==
          "fixed-capacity in-place signed block32 I8 plus exact F16 scale bytes")
  split_owner_exact = bool(
      split_owner.get("class") ==
          "second_custom_consumer_breaks_request_state_aliasing"
      and "entirely zero" in split_owner.get("reason", "")
      and split_owner.get("reopen_condition", "").startswith(
          "only with one state consumer"))

  base_workspace = workspace_bytes(BASE_CONTEXT)
  base_max_workspace = workspace_bytes(BASE_CONTEXT + OUTPUT_TOKENS)
  target_workspace = workspace_bytes(TARGET_CONTEXT)
  target_max_workspace = workspace_bytes(TARGET_CONTEXT + OUTPUT_TOKENS)
  constant_incremental_chunks = (
      target_workspace["chunk_count"] - base_workspace["chunk_count"] == 64
      and target_workspace["cold_chunk_count"] -
          base_workspace["cold_chunk_count"] == 64
      and target_max_workspace["chunk_count"] -
          base_max_workspace["chunk_count"] == 64
      and target_max_workspace["cold_chunk_count"] -
          base_max_workspace["cold_chunk_count"] == 64)

  packed_blocks = math.ceil((EXACT_HISTORY_CAPACITY + 1) / 16)
  old_hot_key_blocks = 2 * packed_blocks + 1
  new_hot_key_blocks = 3 * packed_blocks + 1
  dimension_major_v_plane_bytes_per_layer = (
      (new_hot_key_blocks - old_hot_key_blocks) * KV_HEADS * 2048 * 4)
  dimension_major_v_plane_bytes_all_layers = (
      dimension_major_v_plane_bytes_per_layer * len(LAYERS))
  workspace_bytes_all_layers = (
      target_max_workspace["aligned_bytes"] * len(LAYERS))
  persistent_memory_delta = (
      dimension_major_v_plane_bytes_all_layers + workspace_bytes_all_layers)

  weighted_ucb_ms = float(component["weighted_ucb_ms"])
  weighted_cap_ms = float(component["weighted_cap_ms"])
  component_headroom_ms = weighted_cap_ms - weighted_ucb_ms
  bandwidth_bytes_per_second = float(
      component_bound["budget"]["bandwidth_bytes_per_second"])
  # Seq1251 proves that only Q/current-K/current-V plus a tiny shape carrier
  # are reordered. Charge a complete read+write of those tensors, the F16
  # attention publication, and all four scalar length reads for every layer.
  query_bytes = QUERY_HEADS * HEAD_DIM * F16_BYTES
  current_kv_bytes = 2 * KV_HEADS * HEAD_DIM * F16_BYTES
  attention_bytes = QUERY_HEADS * HEAD_DIM * F16_BYTES
  wrapper_bytes_per_layer = (
      2 * (query_bytes + current_kv_bytes) + attention_bytes + 4 * 4)
  wrapper_bytes_all_layers = wrapper_bytes_per_layer * len(LAYERS)
  wrapper_ms = wrapper_bytes_all_layers / bandwidth_bytes_per_second * 1000.0
  remaining_weighted_margin_ms = component_headroom_ms - wrapper_ms

  design = {
      "route": "single_dynamic_custom_node_four_internal_kernels",
      "graph_nodes_per_layer": 1,
      "device_dispatches_per_decode_layer": 4,
      "state_consumers_per_layer": 1,
      "state_writer": "ordered_update_only",
      "topk_by_layer": {
          str(layer): HIGH_TOPK if layer in HIGH_TOPK_LAYERS else LOW_TOPK
          for layer in LAYERS},
      "prefill": (
          "retain the unified seq1251 cold-aware owner; construct packed "
          "block32 K, dimension-major V, exact F16 scales, packed/dense K, "
          "token-major V, and one dimension-major exact-V plane in place"),
      "decode": [
          "partial_scan_local_select", "select_reduce_union",
          "correct_normalize", "ordered_update"],
      "workspace": (
          "reuse output0 as one private packed F32 scratch allocation; the "
          "plugin schedules four kernels over the same outputs and inputs"),
      "plugin_cut": (
          "specialize the IQ36 adaptive custom primitive onto the existing "
          "multi-kernel scheduling machinery; compile one prefill entry for "
          "Q>1 and the four adaptive decode entries for Q=1"),
      "state_layout": {
          "cold_k": "seq1251 token16/block32 packed I8",
          "cold_v": "seq1251 dimension-major I8",
          "cold_scales": "seq1251 group-major F16 bytes",
          "hot_k": "packed F16 K plane",
          "exact_k": "existing contiguous F16 K plane",
          "prefill_v": "existing token-major F16 V state",
          "decode_exact_v": "third dimension-major F16 plane",
      },
      "forbidden": [
          "four graph custom nodes sharing request-owned state",
          "host synchronization or readback",
          "decode-time bulk state conversion or construction",
          "an extra Variable/Assign state owner",
          "a fifth device dispatch",
          "reusing seq1652's non-packed cold physical layout",
      ],
      "compile_worker_admitted": False,
      "one_layer_worker_admitted": False,
      "product_worker_admitted": False,
  }

  checks = [
      check("repository_clean_at_gate", not repository["dirty"],
            git=repository),
      check("clean_promoted_standalone_component_is_exact", component_exact),
      check("clean_seq1251_packed_state_abi_is_reusable", direct_exact,
            source=direct_source, runtime_reorders=direct_reorders,
            state_reorders=state_reorders),
      check(
          "seq1652_exact_sidecars_are_reusable_but_cold_layout_is_not",
          product_sidecars and product_layout_requires_replacement,
          product_sidecars=product_sidecars,
          product_layout_requires_replacement=
              product_layout_requires_replacement,
          product_direct_i8_fixed_layout=
              product_source.get("direct_i8_fixed_layout"),
          product_cold_storage=product_source.get("cold_storage")),
      check("multiple_graph_state_consumers_are_closed", split_owner_exact,
            evidence=split_owner),
      check("source_splice_points_and_plugin_capabilities_are_exact",
            all(sources.values()), source_contract=sources),
      check("pinned_openvino_source_commit_is_exact",
            ov_head.startswith(OV_COMMIT), observed=ov_head,
            expected_prefix=OV_COMMIT),
      check("single_owner_four_dispatch_design_is_exact",
            design["graph_nodes_per_layer"] == 1
            and design["device_dispatches_per_decode_layer"] == 4
            and design["state_consumers_per_layer"] == 1
            and design["state_writer"] == "ordered_update_only"),
      check("workspace_and_output512_chunk_delta_are_bounded",
            constant_incremental_chunks
            and target_max_workspace["aligned_bytes"] < 6 * MIB
            and workspace_bytes_all_layers < 56 * MIB,
            base=base_workspace, base_output512=base_max_workspace,
            target=target_workspace, target_output512=target_max_workspace,
            constant_incremental_chunks=constant_incremental_chunks,
            all_layer_workspace_bytes=workspace_bytes_all_layers),
      check("wrapper_traffic_fits_seq1673_weighted_margin",
            remaining_weighted_margin_ms > 0.15,
            weighted_ucb_ms=weighted_ucb_ms,
            weighted_cap_ms=weighted_cap_ms,
            component_headroom_ms=component_headroom_ms,
            wrapper_bytes_per_layer=wrapper_bytes_per_layer,
            wrapper_bytes_all_layers=wrapper_bytes_all_layers,
            wrapper_ms=wrapper_ms,
            remaining_weighted_margin_ms=remaining_weighted_margin_ms),
      check("persistent_memory_delta_is_source_bounded",
            persistent_memory_delta < 768 * MIB,
            packed_blocks_per_plane=packed_blocks,
            old_hot_key_blocks=old_hot_key_blocks,
            new_hot_key_blocks=new_hot_key_blocks,
            dimension_major_v_plane_bytes_per_layer=
                dimension_major_v_plane_bytes_per_layer,
            dimension_major_v_plane_bytes_all_layers=
                dimension_major_v_plane_bytes_all_layers,
            workspace_bytes_all_layers=workspace_bytes_all_layers,
            persistent_memory_delta_bytes=persistent_memory_delta,
            persistent_memory_delta_mib=persistent_memory_delta / MIB),
      check("status_selects_source_ownership_gate",
            "source/ownership gate" in status
            and "four device dispatches/layer" in status),
      check("memory_guard_never_tripped",
            available_memory_bytes() >= stop_bytes,
            available_before_bytes=memory_before,
            available_after_bytes=available_memory_bytes(),
            stop_bytes=stop_bytes),
  ]
  required_checks_passed = all(row["pass"] for row in checks)
  source_edit_admitted = required_checks_passed
  verdict = (
      "admit_single_owner_multikernel_graph_integration_source"
      if source_edit_admitted else "inconclusive")

  payload = {
      "schema": SCHEMA,
      "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
      "workstream": WS,
      "verdict": verdict,
      "required_checks_passed": required_checks_passed,
      "source_edit_admitted": source_edit_admitted,
      "compile_admitted": False,
      "one_layer_worker_admitted": False,
      "product_worker_admitted": False,
      "long_worker_admitted": False,
      "git": repository,
      "openvino_source": {
          "path": str(OV_ROOT),
          "head": ov_head,
          "expected_head_prefix": OV_COMMIT,
          "expected_head_matches": ov_head.startswith(OV_COMMIT),
      },
      "design": design,
      "workspace": {
          "base": base_workspace,
          "base_output512": base_max_workspace,
          "target": target_workspace,
          "target_output512": target_max_workspace,
          "all_layer_max_bytes": workspace_bytes_all_layers,
          "constant_matched_incremental_chunk_count":
              constant_incremental_chunks,
      },
      "traffic": {
          "bandwidth_bytes_per_second": bandwidth_bytes_per_second,
          "standalone_weighted_ucb_ms": weighted_ucb_ms,
          "weighted_cap_ms": weighted_cap_ms,
          "component_headroom_ms": component_headroom_ms,
          "wrapper_bytes_per_layer": wrapper_bytes_per_layer,
          "wrapper_bytes_all_layers": wrapper_bytes_all_layers,
          "wrapper_ms": wrapper_ms,
          "remaining_weighted_margin_ms": remaining_weighted_margin_ms,
          "rule": (
              "charge read+write of every seq1251-proven small preprocess "
              "tensor, one F16 attention publication, and four scalar "
              "length reads; prohibit any full-state reorder"),
      },
      "memory": {
          "dimension_major_v_plane_bytes_per_layer":
              dimension_major_v_plane_bytes_per_layer,
          "dimension_major_v_plane_bytes_all_layers":
              dimension_major_v_plane_bytes_all_layers,
          "workspace_bytes_all_layers": workspace_bytes_all_layers,
          "persistent_delta_bytes": persistent_memory_delta,
          "persistent_delta_mib": persistent_memory_delta / MIB,
          "runtime_memory_admitted": False,
      },
      "checks": checks,
      "inputs": {
          display(path): sha256(path) for path in required},
      "source_files": source_files,
  }
  write_json(out_dir / "bound.json", payload)
  summary = f"""# Adaptive-attention OpenVINO integration source bound

Verdict: **{verdict}**. Required checks:
`{str(required_checks_passed).lower()}`. No compiler or GPU worker ran.

The four-graph-node spelling is closed by seq1600-1602: a second custom
consumer receives temporary request-state buffers and loses the in-place state
publication. The admitted source shape is therefore one dynamic custom node per
layer with one state consumer and four plugin-internal decode kernels. This is
the existing OpenVINO multi-stage execution model applied to the seq1673
component, not a fifth dispatch or a host bridge.

The integration must reuse seq1251's packed block32 K, dimension-major V, and
group-major F16-scale ABI. Seq1652 contributes the exact K/V sidecar capacity,
but its non-packed cold physical layout is not reusable. A third K-state plane
holds dimension-major exact V for the measured DPAS carrier while the existing
token-major V plane protects prefill.

The largest output512 workspace is
`{target_max_workspace['aligned_bytes']:,}` bytes/layer
(`{target_max_workspace['aligned_mib']:.6f} MiB`). Ten workspaces plus the
third V planes add `{persistent_memory_delta / MIB:.3f} MiB` persistently.
The matched 64k-minus-32k chunk delta remains exactly 64 throughout output512.
Conservatively charged wrapper traffic costs `{wrapper_ms:.6f} ms/token`,
leaving `{remaining_weighted_margin_ms:.6f} ms/token` below seq1673's weighted
cap.

Only the repository source/patch cut is admitted next. Compile, one-layer
runtime, all-ten composition, long-context, and product claims remain gated.
"""
  (out_dir / "summary.md").write_text(summary, encoding="utf-8")
  print(json.dumps({
      "artifact": display(out_dir),
      "verdict": verdict,
      "required_checks_passed": required_checks_passed,
      "failed_checks": [row["name"] for row in checks if not row["pass"]],
      "persistent_memory_delta_mib": persistent_memory_delta / MIB,
      "remaining_weighted_margin_ms": remaining_weighted_margin_ms,
  }, indent=2))
  return 0 if required_checks_passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
