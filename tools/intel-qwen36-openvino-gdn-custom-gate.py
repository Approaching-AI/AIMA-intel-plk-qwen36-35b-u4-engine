#!/usr/bin/env python3
"""Gate real seq1024 GatedDeltaNet custom-GPU substitutions.

The component lane consumes the non-invasive stock boundary captured by the
OV1 gate-1a artifact.  The real-model lane replaces a requested prefix of the
30 layers from their fused token-major qkv/gate/beta boundaries, while stock
and candidate execute in isolated processes.  Wall times are diagnostic: this
gate proves numerical substitution only and makes no product-speed claim.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-openvino-gdn-custom-gate-v1"
OV_PYTHON = Path("/home/intel/ov/openvino_env/bin/python")
MODEL_DIR = Path("/home/intel/Qwen3.6-35B-A3B-ov")
MODEL_CONTRACT = ROOT / "contracts/qwen36-35b-a3b-openvino-u4-model-contract.json"
PROMPT = (
    ROOT / "output/r0-oracle-prompt-materialization-20260626T082201Z/"
    "prompts/sentinel_008k.txt")
BOUNDARY_ORACLE = (
    ROOT / "output/openvino-gdn-boundary-20260713Tseq802cleanZ")
CUSTOM_CONFIG = ROOT / "engine/openvino/custom/iq36_gated_delta_net.xml"
CUSTOM_SOURCE = ROOT / "engine/openvino/custom/iq36_gated_delta_net.cl"
LOOP_NAME = "Loop_1520"
SEQ_LEN = 1024
EXPECTED_PROMPT_TOKENS = 8192
EXPECTED_STATE_COUNT = 80
EXPECTED_STOCK_GDN_COUNT = 30
EXPECTED_PRIMITIVE = "ocl::gated_delta_net::ref___f16"
ATTENTION_ELEMENTS = 32 * SEQ_LEN * 128
STATE_ELEMENTS = 32 * 128 * 128
PACKED_ELEMENTS = ATTENTION_ELEMENTS + STATE_ELEMENTS
COMPILE_CONFIG = {
    "DYNAMIC_QUANTIZATION_GROUP_SIZE": 256,
    "PERFORMANCE_HINT": "LATENCY",
    "PERF_COUNT": True,
}


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
  parser.add_argument("--model-contract", type=Path, default=MODEL_CONTRACT)
  parser.add_argument("--prompt", type=Path, default=PROMPT)
  parser.add_argument("--boundary-oracle", type=Path, default=BOUNDARY_ORACLE)
  parser.add_argument("--custom-config", type=Path, default=CUSTOM_CONFIG)
  parser.add_argument("--custom-source", type=Path, default=CUSTOM_SOURCE)
  parser.add_argument("--device", default="GPU")
  parser.add_argument(
      "--replace-layers", type=int, default=1,
      help="replace this many leading GatedDeltaNet layers (1..30)")
  parser.add_argument(
      "--fuse-qkv-transpose", action="store_true",
      help=("feed the single-consumer pre-Transpose [B,8192,T] producer "
            "directly to a feature-major custom kernel"))
  parser.add_argument("--out-dir", type=Path)
  parser.add_argument("--timeout-s", type=int, default=1800)
  parser.add_argument("--worker-config", type=Path, help=argparse.SUPPRESS)
  args = parser.parse_args()
  if args.timeout_s <= 0:
    parser.error("timeout-s must be positive")
  if not 1 <= args.replace_layers <= EXPECTED_STOCK_GDN_COUNT:
    parser.error(
        f"replace-layers must be in 1..{EXPECTED_STOCK_GDN_COUNT}")
  if args.out_dir is None and args.worker_config is None:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.out_dir = ROOT / f"output/openvino-gdn-custom-{stamp}"
  return args


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
  with path.open("w", encoding="utf-8") as handle:
    for row in rows:
      handle.write(json.dumps(row, sort_keys=True) + "\n")


def load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise ValueError(f"{path}: expected JSON object")
  return value


def sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def sha256_array(value: Any, np: Any) -> str:
  return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def relative(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path.resolve())


def check(name: str, passed: bool, **evidence: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **evidence}


def git_state(out_dir: Path) -> dict[str, Any]:
  def git(*args: str) -> str:
    run = subprocess.run(
        ["git", *args], cwd=ROOT, check=False, capture_output=True,
        text=True, encoding="utf-8", errors="replace")
    return run.stdout.strip() if run.returncode == 0 else ""

  dirty = git("status", "--porcelain").splitlines()
  try:
    relative_out = str(out_dir.relative_to(ROOT))
  except ValueError:
    relative_out = ""
  dirty = [row for row in dirty if not relative_out or relative_out not in row]
  return {
      "commit": git("rev-parse", "HEAD"),
      "dirty": bool(dirty),
      "dirty_paths": dirty,
  }


def locked_file_rows(
    model_dir: Path, contract: dict[str, Any],
) -> list[dict[str, Any]]:
  rows = []
  for name, expected in sorted(
      contract.get("product_model", {}).get("locked_files", {}).items()):
    path = model_dir / name
    exists = path.is_file()
    size = path.stat().st_size if exists else None
    digest = sha256_file(path) if exists else None
    rows.append({
        "name": name,
        "path": str(path),
        "exists": exists,
        "bytes": size,
        "sha256": digest,
        "expected_bytes": expected.get("bytes"),
        "expected_sha256": expected.get("sha256"),
        "pass": (
            exists and size == expected.get("bytes") and
            digest == expected.get("sha256")),
    })
  return rows


def any_value(value: Any) -> Any:
  try:
    return value.value
  except Exception:
    return str(value)


def runtime_focus_rows(compiled: Any) -> list[dict[str, Any]]:
  rows = []
  for node in compiled.get_runtime_model().get_ordered_ops():
    info = {str(key): any_value(value)
            for key, value in node.get_rt_info().items()}
    layer_type = str(info.get("layerType"))
    if (layer_type not in ("GatedDeltaNet", "CustomGPUPrimitive") and
        "iq36_gdn" not in node.get_friendly_name().lower()):
      continue
    rows.append({
        "node_name": node.get_friendly_name(),
        "layer_type": layer_type,
        "primitive_type": str(info.get("primitiveType")),
        "runtime_precision": str(info.get("runtimePrecision")),
        "output_layouts": str(info.get("outputLayouts")),
        "output_precisions": str(info.get("outputPrecisions")),
    })
  return rows


def profile_focus_rows(request: Any) -> list[dict[str, Any]]:
  rows = []
  for row in request.get_profiling_info():
    if (row.node_type not in ("GatedDeltaNet", "IQ36GatedDeltaNet") and
        "iq36_gdn" not in row.node_name.lower()):
      continue
    rows.append({
        "node_name": row.node_name,
        "node_type": row.node_type,
        "exec_type": row.exec_type,
        "status": str(row.status),
        "real_time_us": row.real_time.total_seconds() * 1_000_000.0,
    })
  return rows


def profile_all_rows(request: Any) -> list[dict[str, Any]]:
  rows = []
  for row in request.get_profiling_info():
    rows.append({
        "node_name": row.node_name,
        "node_type": row.node_type,
        "exec_type": row.exec_type,
        "status": str(row.status),
        "real_time_us": row.real_time.total_seconds() * 1_000_000.0,
        "cpu_time_us": row.cpu_time.total_seconds() * 1_000_000.0,
    })
  return rows


def vector_metrics(reference: Any, candidate: Any, np: Any) -> dict[str, Any]:
  ref = np.asarray(reference, dtype=np.float64).reshape(-1)
  cand = np.asarray(candidate, dtype=np.float64).reshape(-1)
  diff = cand - ref
  ref_norm = float(np.linalg.norm(ref))
  cand_norm = float(np.linalg.norm(cand))
  denominator = ref_norm * cand_norm
  return {
      "count": int(ref.size),
      "finite": bool(np.isfinite(ref).all() and np.isfinite(cand).all()),
      "exact_bits": bool(np.array_equal(reference, candidate)),
      "max_abs": float(np.max(np.abs(diff))) if diff.size else 0.0,
      "relative_l2": (
          float(np.linalg.norm(diff) / ref_norm)
          if ref_norm else float(np.linalg.norm(diff))),
      "cosine": (
          float(np.dot(ref, cand) / denominator) if denominator else 1.0),
  }


def distribution_metrics(
    reference: Any, candidate: Any, np: Any,
) -> dict[str, Any]:
  ref = np.asarray(reference, dtype=np.float64).reshape(-1)
  cand = np.asarray(candidate, dtype=np.float64).reshape(-1)
  ref_probability = np.exp(ref - float(np.max(ref)))
  candidate_probability = np.exp(cand - float(np.max(cand)))
  ref_probability /= float(ref_probability.sum())
  candidate_probability /= float(candidate_probability.sum())
  epsilon = np.finfo(np.float64).tiny
  return {
      **vector_metrics(reference, candidate, np),
      "kld_reference_to_candidate": float(np.sum(
          ref_probability * (
              np.log(np.maximum(ref_probability, epsilon)) -
              np.log(np.maximum(candidate_probability, epsilon))))),
      "reference_top1": int(np.argmax(ref)),
      "candidate_top1": int(np.argmax(cand)),
      "top1_match": bool(int(np.argmax(ref)) == int(np.argmax(cand))),
  }


def custom_class(ov: Any) -> type:
  class IQ36GatedDeltaNet(ov.Op):
    def __init__(self, inputs: Any = None):
      super().__init__(self, inputs)
      self.constructor_validate_and_infer_types()

    def validate_and_infer_types(self) -> None:
      self.set_output_type(
          0, self.get_input_element_type(0),
          ov.PartialShape([1, 1, 1, PACKED_ELEMENTS]))

    def clone_with_new_inputs(self, new_inputs: Any) -> Any:
      return IQ36GatedDeltaNet(new_inputs)

    def visit_attributes(self, visitor: Any) -> bool:
      return True

  return IQ36GatedDeltaNet


def constant_i64(values: list[int], ov: Any, np: Any) -> Any:
  return ov.opset13.constant(np.asarray(values, dtype=np.int64))


def reshape_fixed(value: Any, shape: list[int], ov: Any, np: Any) -> Any:
  return ov.opset13.reshape(value, constant_i64(shape, ov, np), False)


def make_component_model(ov: Any, fuse_qkv_transpose: bool = False) -> Any:
  shapes = (
      ([1, 8192, 1, SEQ_LEN] if fuse_qkv_transpose else
       [1, SEQ_LEN, 64, 128]),
      [1, 32, 128, 128],
      [1, SEQ_LEN, 32, 1],
      [1, SEQ_LEN, 32, 1],
  )
  parameters = [
      ov.opset13.parameter(shape, ov.Type.f16, name=f"boundary_{index}")
      for index, shape in enumerate(shapes)
  ]
  operation = custom_class(ov)([parameter.output(0)
                                for parameter in parameters])
  operation.set_friendly_name("iq36_gdn_component")
  return ov.Model(
      [operation.output(0)], parameters, "iq36_gdn_component_oracle")


def make_candidate_model(
    core: Any, model_dir: Path, replace_layers: int, ov: Any, np: Any,
    fuse_qkv_transpose: bool = False,
) -> tuple[Any, dict[str, Any]]:
  model = core.read_model(str(model_dir / "openvino_language_model.xml"))
  loops = [
      node for node in model.get_ordered_ops()
      if node.get_type_name() == "Loop"
  ]
  if (len(loops) != EXPECTED_STOCK_GDN_COUNT or
      loops[0].get_friendly_name() != LOOP_NAME):
    raise RuntimeError(
        "model no longer exposes the locked 30-layer GatedDeltaNet loop set")

  selected_rows = []
  replaced_loop_names = []
  for layer_index, loop in enumerate(loops[:replace_layers]):
    value_transpose = loop.input_value(4).get_node()
    value_reshape = value_transpose.input_value(0).get_node()
    qkv_split = value_reshape.input_value(0).get_node()
    if (value_transpose.get_type_name() != "Transpose" or
        value_reshape.get_type_name() != "Reshape" or
        qkv_split.get_type_name() != "VariadicSplit"):
      raise RuntimeError(
          f"layer {layer_index} no longer exposes the fused qkv boundary")

    qkv = qkv_split.input_value(0)
    qkv_transpose = qkv.get_node()
    if fuse_qkv_transpose:
      transpose_order = qkv_transpose.input_value(1).get_node()
      try:
        transpose_order_values = [int(value) for value in
                                  transpose_order.data.tolist()]
      except Exception as exc:
        raise RuntimeError(
            f"layer {layer_index} qkv Transpose order is not constant") from exc
      qkv_consumers = list(qkv_transpose.output(0).get_target_inputs())
      if (qkv_transpose.get_type_name() != "Transpose" or
          transpose_order_values != [0, 2, 1] or
          len(qkv_consumers) != 1 or
          qkv_consumers[0].get_node().get_friendly_name() !=
          qkv_split.get_friendly_name()):
        raise RuntimeError(
            f"layer {layer_index} no longer exposes a single-consumer "
            "[0,2,1] qkv Transpose")
      qkv = qkv_transpose.input_value(0)
    gate = loop.input_value(5).get_node().input_value(0)
    beta = loop.input_value(6).get_node().input_value(0)
    state = loop.input_value(7)
    selected = (qkv, state, gate, beta)
    selected_rows.extend({
        "layer": layer_index,
        "loop_name": loop.get_friendly_name(),
        "input_index": input_index,
        "producer": value.get_node().get_friendly_name(),
        "producer_type": value.get_node().get_type_name(),
        "partial_shape": str(value.get_partial_shape()),
        "element_type": str(value.get_element_type()),
    } for input_index, value in enumerate(selected))

    qkv_f16 = ov.opset13.convert(reshape_fixed(
        qkv,
        ([1, 8192, 1, SEQ_LEN] if fuse_qkv_transpose else
         [1, SEQ_LEN, 64, 128]),
        ov, np), ov.Type.f16)
    state_f16 = ov.opset13.convert(state, ov.Type.f16)
    gate_f16 = ov.opset13.convert(reshape_fixed(
        gate, [1, SEQ_LEN, 32, 1], ov, np), ov.Type.f16)
    beta_f16 = ov.opset13.convert(reshape_fixed(
        beta, [1, SEQ_LEN, 32, 1], ov, np), ov.Type.f16)
    operation = custom_class(ov)([
        qkv_f16.output(0), state_f16.output(0),
        gate_f16.output(0), beta_f16.output(0)])
    operation.set_friendly_name(f"iq36_gdn_layer{layer_index}")

    attention_targets = list(loop.output(0).get_target_inputs())
    state_targets = list(loop.output(1).get_target_inputs())
    if len(attention_targets) != 1 or len(state_targets) != 1:
      raise RuntimeError(
          f"layer {layer_index} Loop outputs no longer have one consumer")
    attention_flat = attention_targets[0].get_node()
    state_flat = state_targets[0].get_node()
    attention_concat_targets = list(
        attention_flat.output(0).get_target_inputs())
    state_concat_targets = list(state_flat.output(0).get_target_inputs())
    attention_concat = (
        attention_concat_targets[0].get_node()
        if len(attention_concat_targets) == 1 else None)
    state_concat = (
        state_concat_targets[0].get_node()
        if len(state_concat_targets) == 1 else None)
    if (attention_flat.get_type_name() != "Reshape" or
        state_flat.get_type_name() != "Reshape" or
        attention_concat is None or state_concat is None or
        attention_concat.get_friendly_name() !=
        state_concat.get_friendly_name() or
        attention_concat.get_type_name() != "Concat"):
      raise RuntimeError(
          f"layer {layer_index} no longer exposes post-Loop flat Concat")
    concat_slices = [
        target.get_node()
        for target in attention_concat.output(0).get_target_inputs()
    ]
    attention_slice = next(
        (node for node in concat_slices
         if node.get_type_name() == "Slice" and
         node.input_value(1).get_node().get_type_name() == "Constant"),
        None)
    state_slice = next(
        (node for node in concat_slices
         if node.get_type_name() == "Slice" and
         node.input_value(1).get_node().get_type_name() != "Constant"),
        None)
    attention_semantic_targets = (
        list(attention_slice.output(0).get_target_inputs())
        if attention_slice is not None else [])
    state_semantic_targets = (
        list(state_slice.output(0).get_target_inputs())
        if state_slice is not None else [])
    if (len(attention_semantic_targets) != 1 or
        len(state_semantic_targets) != 1 or
        attention_semantic_targets[0].get_node().get_type_name() !=
        "Reshape" or
        state_semantic_targets[0].get_node().get_type_name() != "Reshape"):
      raise RuntimeError(
          f"layer {layer_index} no longer exposes two semantic Reshapes")
    attention_semantic = attention_semantic_targets[0].get_node()
    state_semantic = state_semantic_targets[0].get_node()
    attention_transpose_targets = list(
        attention_semantic.output(0).get_target_inputs())
    if (len(attention_transpose_targets) != 1 or
        attention_transpose_targets[0].get_node().get_type_name() !=
        "Transpose"):
      raise RuntimeError(
          f"layer {layer_index} no longer exposes output-layout Transpose")
    attention_transpose = attention_transpose_targets[0].get_node()

    packed = operation.output(0)
    flat_axis = constant_i64([3], ov, np)
    unit_step = constant_i64([1], ov, np)
    attention = ov.opset13.slice(
        packed, constant_i64([0], ov, np),
        constant_i64([ATTENTION_ELEMENTS], ov, np),
        unit_step, flat_axis)
    final_state = ov.opset13.slice(
        packed, constant_i64([ATTENTION_ELEMENTS], ov, np),
        constant_i64([PACKED_ELEMENTS], ov, np),
        unit_step, flat_axis)
    attention = reshape_fixed(
        attention, [1, SEQ_LEN, 32, 128], ov, np)
    final_state = reshape_fixed(
        final_state, [1, 32, 128, 128], ov, np)
    attention_transpose.output(0).replace(
        ov.opset13.convert(attention, ov.Type.f32).output(0))
    state_semantic.output(0).replace(
        ov.opset13.convert(final_state, ov.Type.f32).output(0))
    replaced_loop_names.append(loop.get_friendly_name())

  model.validate_nodes_and_infer_types()
  ordered = model.get_ordered_ops()
  remaining_names = {node.get_friendly_name() for node in ordered}
  summary = {
      "requested_replace_layers": replace_layers,
      "replaced_loop_names": replaced_loop_names,
      "selected_inputs": selected_rows,
      "custom_node_count": sum(
          node.get_type_name() == "IQ36GatedDeltaNet" for node in ordered),
      "replaced_loops_still_present": sorted(
          set(replaced_loop_names) & remaining_names),
      "remaining_loop_count": sum(
          node.get_type_name() == "Loop" for node in ordered),
      "operation_count": len(ordered),
      "packed_output_shape": [1, 1, 1, PACKED_ELEMENTS],
      "packed_layout": "flat_token_major_attention_then_state",
      "fuse_qkv_transpose": fuse_qkv_transpose,
      "qkv_input_layout": (
          "feature_major_bf1t" if fuse_qkv_transpose else
          "token_major_bt64x128"),
      "removed_qkv_transpose_count": (
          replace_layers if fuse_qkv_transpose else 0),
  }
  return model, summary


def make_inputs(
    embedding: Any, token_ids: Any, np: Any,
) -> dict[str, Any]:
  ids = np.asarray(token_ids, dtype=np.int64).reshape(1, -1)
  embedded = np.asarray(
      embedding({embedding.input(0): ids})[embedding.output(0)])
  positions = np.arange(ids.shape[1], dtype=np.int64)
  return {
      "attention_mask": np.ones((1, ids.shape[1]), dtype=np.int64),
      "beam_idx": np.zeros((1,), dtype=np.int32),
      "inputs_embeds": embedded.astype(np.float32, copy=False),
      "position_ids": np.tile(positions, (4, 1)).reshape(4, 1, -1),
  }


def state_signatures(request: Any, np: Any) -> list[dict[str, Any]]:
  rows = []
  for state in request.query_state():
    value = np.array(state.state.data, copy=True)
    rows.append({
        "name": str(state.name),
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "finite": bool(np.isfinite(value).all()),
        "sha256": sha256_array(value, np),
    })
  return sorted(rows, key=lambda row: row["name"])


def run_language_model(
    compiled: Any, inputs: dict[str, Any], logits_path: Path, np: Any,
) -> dict[str, Any]:
  request = compiled.create_infer_request()
  request.reset_state()
  started = time.perf_counter_ns()
  outputs = request.infer(inputs, share_outputs=False)
  wall_ms = (time.perf_counter_ns() - started) / 1_000_000.0
  logits_value = np.asarray(outputs[compiled.output(0)], dtype=np.float32)
  logits = np.array(logits_value[0, -1], dtype="<f4", copy=True)
  logits.tofile(logits_path)
  return {
      "logits_count": int(logits.size),
      "logits_finite": bool(np.isfinite(logits).all()),
      "logits_sha256": sha256_file(logits_path),
      "logits_top1": int(np.argmax(logits)),
      "logits_payload": relative(logits_path),
      "states": state_signatures(request, np),
      "profile": profile_focus_rows(request),
      "profile_all": profile_all_rows(request),
      "wall_ms_diagnostic": wall_ms,
  }


def run_component(
    core: Any, boundary_dir: Path, device: str, np: Any, ov: Any,
    fuse_qkv_transpose: bool = False,
) -> dict[str, Any]:
  model = make_component_model(ov, fuse_qkv_transpose)
  compiled = core.compile_model(model, device, COMPILE_CONFIG)
  shapes = (
      (1, SEQ_LEN, 64, 128),
      (1, 32, 128, 128),
      (1, SEQ_LEN, 32, 1),
      (1, SEQ_LEN, 32, 1),
  )
  indices = (0, 3, 4, 5)
  values = [
      np.fromfile(
          boundary_dir / f"dispatch000-arg{index}-before.bin",
          dtype="<f2").reshape(shape)
      for index, shape in zip(indices, shapes)
  ]
  if fuse_qkv_transpose:
    values[0] = np.ascontiguousarray(
        values[0].reshape(1, SEQ_LEN, 8192).transpose(0, 2, 1)
        .reshape(1, 8192, 1, SEQ_LEN))
  request = compiled.create_infer_request()
  request.infer(dict(zip(compiled.inputs, values)), share_outputs=False)
  started = time.perf_counter_ns()
  outputs = request.infer(
      dict(zip(compiled.inputs, values)), share_outputs=False)
  wall_ms = (time.perf_counter_ns() - started) / 1_000_000.0
  packed = np.asarray(outputs[compiled.output(0)]).reshape(-1)
  attention = packed[:ATTENTION_ELEMENTS].reshape(
      1, SEQ_LEN, 32, 128).transpose(0, 2, 1, 3)
  final_state = packed[ATTENTION_ELEMENTS:].reshape(1, 32, 128, 128)
  reference_attention = np.fromfile(
      boundary_dir / "dispatch000-arg6-after.bin", dtype="<f2").reshape(
          1, SEQ_LEN, 32, 128).transpose(0, 2, 1, 3)
  reference_state = np.fromfile(
      boundary_dir / "dispatch000-arg7-after.bin", dtype="<f2").reshape(
          1, 32, 128, 128)
  return {
      "attention": vector_metrics(reference_attention, attention, np),
      "final_state": vector_metrics(reference_state, final_state, np),
      "packed_shape": [1, 1, 1, int(packed.size)],
      "packed_dtype": str(packed.dtype),
      "packed_finite": bool(np.isfinite(packed).all()),
      "runtime": runtime_focus_rows(compiled),
      "profile": profile_focus_rows(request),
      "wall_ms_diagnostic": wall_ms,
  }


def compare_state_signatures(
    stock: list[dict[str, Any]], candidate: list[dict[str, Any]],
) -> dict[str, Any]:
  stock_by_name = {row["name"]: row for row in stock}
  candidate_by_name = {row["name"]: row for row in candidate}
  names_match = set(stock_by_name) == set(candidate_by_name)
  rows = []
  for name in sorted(set(stock_by_name) & set(candidate_by_name)):
    reference = stock_by_name[name]
    observed = candidate_by_name[name]
    rows.append({
        "name": name,
        "shape_match": reference["shape"] == observed["shape"],
        "dtype_match": reference["dtype"] == observed["dtype"],
        "finite": reference["finite"] and observed["finite"],
        "exact_bits": reference["sha256"] == observed["sha256"],
        "stock_sha256": reference["sha256"],
        "candidate_sha256": observed["sha256"],
    })
  return {
      "names_match": names_match,
      "stock_count": len(stock),
      "candidate_count": len(candidate),
      "all_exact_bits": (
          names_match and all(
              row["shape_match"] and row["dtype_match"] and
              row["finite"] and row["exact_bits"] for row in rows)),
      "mismatch_names": [row["name"] for row in rows
                         if not row["exact_bits"]],
      "rows": rows,
  }


def worker_main(config_path: Path) -> int:
  if Path(sys.prefix).resolve() != OV_PYTHON.parent.parent.resolve():
    raise SystemExit(f"worker requires {OV_PYTHON}, observed {sys.executable}")

  import numpy as np
  import openvino as ov
  import openvino_genai as ov_genai

  cfg = load_json(config_path)
  mode = str(cfg["mode"])
  model_dir = Path(cfg["model_dir"])
  prompt = Path(cfg["prompt"])
  result_path = Path(cfg["result_path"])
  logits_path = Path(cfg["logits_path"])
  device = str(cfg["device"])
  custom_config = Path(cfg["custom_config"])
  boundary_dir = Path(cfg["boundary_dir"])
  replace_layers = int(cfg["replace_layers"])
  fuse_qkv_transpose = bool(cfg.get("fuse_qkv_transpose", False))

  core = ov.Core()
  config_before = str(core.get_property(device, "CONFIG_FILE"))
  if mode == "candidate":
    core.set_property(device, {"CONFIG_FILE": str(custom_config.resolve())})
  config_after = str(core.get_property(device, "CONFIG_FILE"))
  embedding = core.compile_model(
      core.read_model(str(model_dir / "openvino_text_embeddings_model.xml")),
      "CPU", {"PERFORMANCE_HINT": "LATENCY"})
  tokenizer = ov_genai.Tokenizer(str(model_dir))
  prompt_ids = np.asarray(
      tokenizer.encode(prompt.read_text(encoding="utf-8")).input_ids.data
  ).reshape(-1).astype(np.int64)
  tile_ids = prompt_ids[:SEQ_LEN]
  token_payload = Path(cfg["token_payload"])
  np.ascontiguousarray(tile_ids, dtype="<u4").tofile(token_payload)
  inputs = make_inputs(embedding, tile_ids, np)

  source_summary = None
  if mode == "candidate":
    source, source_summary = make_candidate_model(
        core, model_dir, replace_layers, ov, np, fuse_qkv_transpose)
  elif mode == "stock":
    source = core.read_model(str(model_dir / "openvino_language_model.xml"))
  else:
    raise ValueError(f"unknown worker mode {mode}")
  started = time.perf_counter_ns()
  compiled = core.compile_model(source, device, COMPILE_CONFIG)
  compile_ms = (time.perf_counter_ns() - started) / 1_000_000.0
  runtime = runtime_focus_rows(compiled)
  inference = run_language_model(compiled, inputs, logits_path, np)

  checks = [
      check("worker_starts_without_custom_config", config_before == "",
            observed=config_before),
      check("exact_sentinel_8k_first_tile",
            len(prompt_ids) == EXPECTED_PROMPT_TOKENS and
            len(tile_ids) == SEQ_LEN,
            prompt_tokens=int(len(prompt_ids)), tile_tokens=int(len(tile_ids)),
            tile_sha256=sha256_file(token_payload)),
      check("language_logits_and_all_states_are_finite",
            inference["logits_finite"] and
            len(inference["states"]) == EXPECTED_STATE_COUNT and
            all(row["finite"] for row in inference["states"]),
            logits_count=inference["logits_count"],
            state_count=len(inference["states"])),
  ]
  component = None
  comparison = None
  no_config_error = ""
  if mode == "stock":
    stock_gdn = [row for row in runtime
                 if row["layer_type"] == "GatedDeltaNet"]
    checks.extend([
        check("stock_worker_never_loads_candidate_config",
              config_after == "", observed=config_after),
        check("stock_runtime_executes_all_30_gdn_primitives",
              len(stock_gdn) == EXPECTED_STOCK_GDN_COUNT and
              all(row["primitive_type"] == EXPECTED_PRIMITIVE
                  for row in stock_gdn), rows=stock_gdn),
    ])
  else:
    component = run_component(
        core, boundary_dir, device, np, ov, fuse_qkv_transpose)
    try:
      ov.Core().compile_model(
          make_component_model(ov, fuse_qkv_transpose), device)
    except Exception as exc:
      no_config_error = repr(exc)
    stock_result = load_json(Path(cfg["stock_result"]))
    stock_logits = np.fromfile(
        Path(stock_result["inference"]["logits_payload"]), dtype="<f4")
    candidate_logits = np.fromfile(logits_path, dtype="<f4")
    comparison = {
        "logits": distribution_metrics(
            stock_logits, candidate_logits, np),
        "states": compare_state_signatures(
            stock_result["inference"]["states"], inference["states"]),
    }
    candidate_gdn = [row for row in runtime
                     if row["layer_type"] == "GatedDeltaNet"]
    candidate_custom = [row for row in runtime
                        if row["layer_type"] == "CustomGPUPrimitive"]
    custom_profile = [row for row in inference["profile"]
                      if row["node_type"] == "IQ36GatedDeltaNet"]
    component_custom = [row for row in component["profile"]
                        if row["node_type"] == "IQ36GatedDeltaNet"]
    checks.extend([
        check("candidate_worker_loads_only_requested_custom_config",
              config_after == str(custom_config.resolve()),
              observed=config_after),
        check("simplegpu_without_config_rejects_custom_operation",
              bool(no_config_error), error=no_config_error),
        check("candidate_source_replaces_requested_loop_prefix",
              source_summary["requested_replace_layers"] == replace_layers and
              source_summary["custom_node_count"] == replace_layers and
              not source_summary["replaced_loops_still_present"] and
              source_summary["remaining_loop_count"] ==
              EXPECTED_STOCK_GDN_COUNT - replace_layers,
              summary=source_summary),
        check("component_packed_single_output_is_exact_to_seq802",
              component["packed_finite"] and
              component["attention"]["exact_bits"] and
              component["final_state"]["exact_bits"],
              attention=component["attention"],
              final_state=component["final_state"],
              packed_shape=component["packed_shape"]),
        check("candidate_runtime_has_requested_custom_stock_split",
              len(candidate_custom) == replace_layers and
              len(candidate_gdn) ==
              EXPECTED_STOCK_GDN_COUNT - replace_layers and
              all(row["primitive_type"] == EXPECTED_PRIMITIVE
                  for row in candidate_gdn),
              custom=candidate_custom, stock_gdn=candidate_gdn),
        check("custom_kernel_executes_in_component_and_real_model",
              len(component_custom) == 1 and
              len(custom_profile) == replace_layers and
              all(row["status"] == "Status.EXECUTED"
                  for row in component_custom + custom_profile),
              component=component_custom, real_model=custom_profile),
        check("real_model_logits_match_isolated_stock_exactly",
              comparison["logits"]["finite"] and
              comparison["logits"]["exact_bits"] and
              comparison["logits"]["kld_reference_to_candidate"] == 0.0 and
              comparison["logits"]["top1_match"],
              metrics=comparison["logits"]),
        check("real_model_all_80_states_match_isolated_stock_exactly",
              comparison["states"]["names_match"] and
              comparison["states"]["stock_count"] == EXPECTED_STATE_COUNT and
              comparison["states"]["candidate_count"] ==
              EXPECTED_STATE_COUNT and
              comparison["states"]["all_exact_bits"],
              summary={key: value for key, value in
                       comparison["states"].items() if key != "rows"}),
    ])
  passed = all(row["pass"] for row in checks)
  result = {
      "mode": mode,
      "replace_layers": replace_layers,
      "fuse_qkv_transpose": fuse_qkv_transpose,
      "checks": checks,
      "required_checks_passed": passed,
      "compile_ms": compile_ms,
      "config_before": config_before,
      "config_after": config_after,
      "custom_config": str(custom_config.resolve()) if mode == "candidate" else None,
      "source_summary": source_summary,
      "runtime": runtime,
      "inference": inference,
      "component": component,
      "comparison": comparison,
      "no_config_compile_error": no_config_error,
      "openvino_runtime_version": ov.get_version(),
      "openvino_genai_version": ov_genai.__version__,
      "prompt_tokens": int(len(prompt_ids)),
      "tile_tokens": int(len(tile_ids)),
      "token_payload": relative(token_payload),
  }
  write_json(result_path, result)
  print(json.dumps({
      "event": "worker_complete", "mode": mode,
      "required_checks_passed": passed,
  }, sort_keys=True), flush=True)
  return 0 if passed else 2


def run_worker(
    mode: str, raw: Path, base_config: dict[str, Any], timeout_s: int,
    stock_result: Path | None = None,
) -> tuple[subprocess.CompletedProcess[str], Path]:
  worker_dir = raw / mode
  worker_dir.mkdir()
  result_path = worker_dir / "result.json"
  config = {
      **base_config,
      "mode": mode,
      "result_path": str(result_path),
      "logits_path": str((worker_dir / "logits.f32").resolve()),
      "token_payload": str((worker_dir / "sentinel-first-1024.u32").resolve()),
  }
  if stock_result is not None:
    config["stock_result"] = str(stock_result.resolve())
  config_path = worker_dir / "worker-config.json"
  write_json(config_path, config)
  command = [
      str(OV_PYTHON), str(Path(__file__).resolve()),
      "--worker-config", str(config_path.resolve()),
  ]
  environment = os.environ.copy()
  environment.update({
      "NEO_CACHE_DIR": str((worker_dir / "neo-cache").resolve()),
      "NEO_CACHE_MAX_SIZE": str(4 * 1024 * 1024 * 1024),
      "NEO_CACHE_PERSISTENT": "1",
  })
  (worker_dir / "neo-cache").mkdir()
  try:
    run = subprocess.run(
        command, cwd=ROOT, env=environment, check=False,
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=timeout_s)
  except subprocess.TimeoutExpired as exc:
    run = subprocess.CompletedProcess(
        command, 124, str(exc.stdout or ""), str(exc.stderr or ""))
  (worker_dir / "stdout").write_text(run.stdout, encoding="utf-8")
  (worker_dir / "stderr").write_text(run.stderr, encoding="utf-8")
  write_json(worker_dir / "command.json", {
      "command": command,
      "environment": {key: environment[key] for key in (
          "NEO_CACHE_DIR", "NEO_CACHE_MAX_SIZE", "NEO_CACHE_PERSISTENT")},
      "returncode": run.returncode,
  })
  return run, result_path


def main() -> int:
  args = parse_args()
  if args.worker_config is not None:
    return worker_main(args.worker_config.resolve())

  out = args.out_dir.resolve()
  raw = out / "raw"
  raw.mkdir(parents=True, exist_ok=False)
  required = [
      args.model_contract, args.prompt, args.custom_config,
      args.custom_source, OV_PYTHON,
      args.model_dir / "openvino_language_model.xml",
      args.model_dir / "openvino_text_embeddings_model.xml",
      args.boundary_oracle / "correctness.json",
      args.boundary_oracle / "manifest.json",
  ]
  missing = [str(path) for path in required if not path.exists()]
  if missing:
    raise SystemExit("missing required paths: " + ", ".join(missing))

  created_at = dt.datetime.now(dt.timezone.utc).isoformat()
  git = git_state(out)
  contract = load_json(args.model_contract)
  locked_files = locked_file_rows(args.model_dir, contract)
  boundary_correctness = load_json(args.boundary_oracle / "correctness.json")
  boundary_manifest = load_json(args.boundary_oracle / "manifest.json")
  base_config = {
      "model_dir": str(args.model_dir.resolve()),
      "prompt": str(args.prompt.resolve()),
      "device": args.device,
      "replace_layers": args.replace_layers,
      "fuse_qkv_transpose": args.fuse_qkv_transpose,
      "custom_config": str(args.custom_config.resolve()),
      "boundary_dir": str((args.boundary_oracle / "raw/boundary").resolve()),
  }
  stock_run, stock_result_path = run_worker(
      "stock", raw, base_config, args.timeout_s)
  candidate_run, candidate_result_path = run_worker(
      "candidate", raw, base_config, args.timeout_s,
      stock_result=stock_result_path)
  stock = load_json(stock_result_path) if stock_result_path.is_file() else {}
  candidate = (
      load_json(candidate_result_path) if candidate_result_path.is_file()
      else {})
  runtime_contract = contract["runtime_contract"]["baseline"]
  checks = [
      check("repository_clean_at_gate", not git["dirty"], git=git),
      check("all_locked_model_files_match_contract",
            bool(locked_files) and all(row["pass"] for row in locked_files),
            rows=locked_files),
      check("seq802_boundary_oracle_is_clean_and_passed",
            boundary_correctness.get("required_checks_passed") is True and
            boundary_manifest.get("git", {}).get("dirty") is False,
            boundary_commit=boundary_manifest.get("git", {}).get("commit"),
            boundary_required_checks_passed=boundary_correctness.get(
                "required_checks_passed")),
      check("isolated_stock_worker_passes", stock_run.returncode == 0 and
            stock.get("required_checks_passed") is True,
            returncode=stock_run.returncode,
            failed=[row.get("name") for row in stock.get("checks", [])
                    if not row.get("pass")], stderr=stock_run.stderr[-2000:]),
      check("isolated_candidate_worker_passes",
            candidate_run.returncode == 0 and
            candidate.get("required_checks_passed") is True,
            returncode=candidate_run.returncode,
            failed=[row.get("name") for row in candidate.get("checks", [])
                    if not row.get("pass")],
            stderr=candidate_run.stderr[-2000:]),
      check("worker_runtime_versions_match_locked_contract",
            all(result.get("openvino_runtime_version") ==
                runtime_contract["openvino_runtime_version"] and
                result.get("openvino_genai_version") ==
                runtime_contract["openvino_genai_version"]
                for result in (stock, candidate)),
            stock_runtime=stock.get("openvino_runtime_version"),
            candidate_runtime=candidate.get("openvino_runtime_version"),
            stock_genai=stock.get("openvino_genai_version"),
            candidate_genai=candidate.get("openvino_genai_version")),
  ]
  required_passed = all(row["pass"] for row in checks)
  correctness = {
      "schema_version": SCHEMA,
      "workstream": WORKSTREAM,
      "required_checks_passed": required_passed,
      "checks": checks,
      "stock_checks": stock.get("checks", []),
      "candidate_checks": candidate.get("checks", []),
      "component_comparison": candidate.get("component"),
      "real_model_comparison": candidate.get("comparison"),
      "claim_boundary": (
          f"{args.replace_layers}-real-layer numeric substitution only"),
      "product_speedup_claim": False,
  }
  write_json(out / "correctness.json", correctness)
  write_jsonl(out / "metrics.jsonl", [
      {"metric_scope": "component", "boundary": boundary, **metrics}
      for boundary, metrics in (
          ("attention", candidate.get("component", {}).get("attention", {})),
          ("final_state", candidate.get("component", {}).get(
              "final_state", {})))
  ] + [
      {"metric_scope": "profile", "worker": worker, **row}
      for worker, result in (("stock", stock), ("candidate", candidate))
      for row in result.get("inference", {}).get("profile", [])
  ])
  write_json(out / "manifest.json", {
      "schema_version": SCHEMA,
      "workstream": WORKSTREAM,
      "created_at": created_at,
      "tool": relative(Path(__file__)),
      "command": sys.argv,
      "git": git,
      "model_contract": relative(args.model_contract),
      "model_contract_sha256": sha256_file(args.model_contract),
      "model_dir": str(args.model_dir),
      "locked_files": locked_files,
      "prompt": relative(args.prompt),
      "prompt_sha256": sha256_file(args.prompt),
      "boundary_oracle": relative(args.boundary_oracle),
      "boundary_oracle_commit": boundary_manifest.get("git", {}).get(
          "commit"),
      "custom_config": relative(args.custom_config),
      "custom_config_sha256": sha256_file(args.custom_config),
      "custom_source": relative(args.custom_source),
      "custom_source_sha256": sha256_file(args.custom_source),
      "compile_config": COMPILE_CONFIG,
      "replace_layers": args.replace_layers,
      "fuse_qkv_transpose": args.fuse_qkv_transpose,
      "stock_worker_returncode": stock_run.returncode,
      "candidate_worker_returncode": candidate_run.returncode,
      "stock_compile_ms": stock.get("compile_ms"),
      "candidate_compile_ms": candidate.get("compile_ms"),
      "stock_wall_ms_diagnostic": stock.get("inference", {}).get(
          "wall_ms_diagnostic"),
      "candidate_wall_ms_diagnostic": candidate.get("inference", {}).get(
          "wall_ms_diagnostic"),
      "required_checks_passed": required_passed,
      "product_speedup_claim": False,
  })
  failed = [row["name"] for row in checks if not row["pass"]]
  component = candidate.get("component", {})
  comparison = candidate.get("comparison", {})
  logits = comparison.get("logits", {})
  states = comparison.get("states", {})
  (out / "summary.md").write_text("\n".join([
      "# OpenVINO GatedDeltaNet custom substitution", "",
      f"- replaced real layers: `{args.replace_layers}` / "
      f"`{EXPECTED_STOCK_GDN_COUNT}`",
      f"- required checks: **{'PASS' if required_passed else 'FAIL'}**",
      f"- component attention exact: "
      f"`{component.get('attention', {}).get('exact_bits')}`",
      f"- component final state exact: "
      f"`{component.get('final_state', {}).get('exact_bits')}`",
      f"- real-model logits KLD / exact / top-1: "
      f"`{logits.get('kld_reference_to_candidate')}` / "
      f"`{logits.get('exact_bits')}` / `{logits.get('top1_match')}`",
      f"- real-model state tensors / exact: "
      f"`{states.get('candidate_count')}` / "
      f"`{states.get('all_exact_bits')}`",
      f"- failed checks: `{failed}`", "",
      f"This proves seq1024 numerical substitution for the requested "
      f"{args.replace_layers} real layer(s). ",
      "The sequential wall rows are diagnostics, not paired performance ",
      "evidence or a product speedup claim.", "",
  ]), encoding="utf-8")
  print(json.dumps({
      "required_checks_passed": required_passed,
      "component_attention_exact": component.get(
          "attention", {}).get("exact_bits"),
      "component_state_exact": component.get(
          "final_state", {}).get("exact_bits"),
      "real_logits_kld": logits.get("kld_reference_to_candidate"),
      "real_states_exact": states.get("all_exact_bits"),
      "replace_layers": args.replace_layers,
      "failed_checks": failed,
      "out_dir": relative(out),
  }, sort_keys=True))
  return 0 if required_passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
