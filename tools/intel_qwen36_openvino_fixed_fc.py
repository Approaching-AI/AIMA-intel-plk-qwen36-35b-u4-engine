"""Parameterized graph rewrite for the locked Qwen3.6 fixed FC families.

The product graph has 390 non-expert, non-LM-head compressed FC tensors. They
form 160 independent-input groups and only three custom-operation arities:
one, three, and four outputs. This module keeps that layer-count-independent
shape while exposing the existing packed U4 constants directly to SimpleGPU.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterable


MODEL_XML = Path(
    "/home/intel/Qwen3.6-35B-A3B-ov/openvino_language_model.xml")
FULL_ATTENTION_LAYERS = tuple(range(3, 40, 4))
EXPECTED_GROUP_COUNTS = {
    "linear_attention_input": 30,
    "full_attention_qkv": 10,
    "router_shared_input": 40,
    "attention_output": 40,
    "shared_expert_down": 40,
}
GROUP_SIZE = 64
WG_TILE_M = 64
LOCAL_X = 32


def fixed_fc_custom_classes(ov: Any) -> dict[int, type]:
  """Return the O(1) custom-op class family keyed by projection arity."""

  class IQ36FixedFCBase(ov.Op):
    ARITY = 0

    def __init__(
        self, inputs: Any = None, widths: tuple[int, ...] = (), k: int = 0,
    ):
      self._widths = tuple(int(width) for width in widths)
      self._k = int(k)
      self._attrs = {"k": self._k}
      self._attrs.update({
          f"m{index}": (self._widths[index]
                         if index < len(self._widths) else 0)
          for index in range(4)
      })
      super().__init__(self, inputs)
      if inputs is not None:
        self.constructor_validate_and_infer_types()

    def validate_and_infer_types(self) -> None:
      if len(self._widths) != self.ARITY:
        raise ValueError(
            f"{type(self).__name__} requires {self.ARITY} widths")
      expected_inputs = 2 + 3 * self.ARITY
      if self.get_input_size() != expected_inputs:
        raise ValueError(
            f"{type(self).__name__} requires {expected_inputs} inputs")
      activation = self.get_input_partial_shape(0)
      if not activation.rank.is_static:
        raise ValueError("fixed FC activation rank must be static")
      rank = activation.rank.get_length()
      if rank < 2:
        raise ValueError("fixed FC activation must have rank at least two")
      if (activation[-1].is_static and
          activation[-1].get_length() != self._k):
        raise ValueError("fixed FC activation K does not match its attribute")
      prefix = [activation[index] for index in range(rank - 1)]
      self.set_output_size(self.ARITY)
      for index, width in enumerate(self._widths):
        self.set_output_type(
            index, ov.Type.f16,
            ov.PartialShape(prefix + [ov.Dimension(width)]))

    def clone_with_new_inputs(self, new_inputs: Any) -> Any:
      return type(self)(new_inputs, self._widths, self._k)

    def visit_attributes(self, visitor: Any) -> bool:
      visitor.on_attributes(self._attrs)
      return True

  class IQ36FixedFC1(IQ36FixedFCBase):
    ARITY = 1

  class IQ36FixedFC3(IQ36FixedFCBase):
    ARITY = 3

  class IQ36FixedFC4(IQ36FixedFCBase):
    ARITY = 4

  return {1: IQ36FixedFC1, 3: IQ36FixedFC3, 4: IQ36FixedFC4}


def _input_node(node: Any, index: int = 0) -> Any:
  return node.input_value(index).get_node()


def _expect(node: Any, kind: str, context: str) -> Any:
  if node.get_type_name() != kind:
    raise ValueError(
        f"{context}: expected {kind}, observed {node.get_type_name()} "
        f"at {node.get_friendly_name()}")
  return node


def trace_compressed_projection(matmul: Any, ov: Any) -> dict[str, Any]:
  """Recover the packed U4/scale/zero-point constants from one exact FC."""
  name = matmul.get_friendly_name()
  _expect(matmul, "MatMul", name)
  weight_f32 = _expect(_input_node(matmul, 1), "Convert", name)
  reshape = _expect(_input_node(weight_f32), "Reshape", name)
  multiply = _expect(_input_node(reshape), "Multiply", name)
  subtract = _expect(_input_node(multiply), "Subtract", name)
  weight_convert = _expect(_input_node(subtract, 0), "Convert", name)
  zp_convert = _expect(_input_node(subtract, 1), "Convert", name)
  weight = _expect(_input_node(weight_convert), "Constant", name)
  zero_point = _expect(_input_node(zp_convert), "Constant", name)
  scale = _expect(_input_node(multiply, 1), "Constant", name)
  if weight.get_output_element_type(0) != ov.Type.u4:
    raise ValueError(f"{name}: compressed weight is not U4")
  if zero_point.get_output_element_type(0) != ov.Type.u4:
    raise ValueError(f"{name}: zero point is not U4")
  if scale.get_output_element_type(0) != ov.Type.f16:
    raise ValueError(f"{name}: scale is not F16")
  shape = list(weight.get_output_shape(0))
  if len(shape) != 3 or shape[2] != GROUP_SIZE:
    raise ValueError(f"{name}: unexpected packed weight shape {shape}")
  m, groups, group_size = (int(value) for value in shape)
  k = groups * group_size
  if list(zero_point.get_output_shape(0)) != [m, groups, 1]:
    raise ValueError(f"{name}: zero-point shape is not [M,G,1]")
  if list(scale.get_output_shape(0)) != [m, groups, 1]:
    raise ValueError(f"{name}: scale shape is not [M,G,1]")
  return {
      "matmul": matmul,
      "activation": matmul.input_value(0),
      "weight": weight,
      "scale": scale,
      "zero_point": zero_point,
      "m": m,
      "k": k,
      "groups": groups,
  }


def _projection_names(layer: int) -> list[tuple[str, tuple[str, ...]]]:
  prefix = f"__module.model.model.language_model.layers.{layer}."
  if layer in FULL_ATTENTION_LAYERS:
    input_group = (
        "full_attention_qkv",
        tuple(prefix + suffix for suffix in (
            "self_attn.q_proj/ov_ext::linear/MatMul",
            "self_attn.k_proj/ov_ext::linear/MatMul",
            "self_attn.v_proj/ov_ext::linear/MatMul")))
    output_name = prefix + "self_attn.o_proj/ov_ext::linear/MatMul"
  else:
    input_group = (
        "linear_attention_input",
        tuple(prefix + suffix for suffix in (
            "linear_attn.in_proj_qkv/ov_ext::linear/MatMul",
            "linear_attn.in_proj_a/ov_ext::linear/MatMul",
            "linear_attn.in_proj_b/ov_ext::linear/MatMul",
            "linear_attn.in_proj_z/ov_ext::linear/MatMul")))
    output_name = prefix + "linear_attn.out_proj/ov_ext::linear/MatMul"
  router = tuple(prefix + suffix for suffix in (
      "mlp.shared_expert_gate/ov_ext::linear/MatMul",
      "mlp.shared_expert.gate_proj/ov_ext::linear/MatMul",
      "mlp.shared_expert.up_proj/ov_ext::linear/MatMul",
      "mlp.gate/aten::linear/MatMul"))
  return [
      input_group,
      ("router_shared_input", router),
      ("attention_output", (output_name,)),
      ("shared_expert_down", (
          prefix + "mlp.shared_expert.down_proj/ov_ext::linear/MatMul",)),
  ]


def discover_fixed_fc_groups(model: Any, ov: Any) -> list[dict[str, Any]]:
  by_name = {node.get_friendly_name(): node
             for node in model.get_ordered_ops()}
  groups = []
  for layer in range(40):
    for cohort, names in _projection_names(layer):
      missing = [name for name in names if name not in by_name]
      if missing:
        raise ValueError(f"layer {layer} {cohort}: missing {missing}")
      projections = [trace_compressed_projection(by_name[name], ov)
                     for name in names]
      activation_ids = {
          (projection["activation"].get_node().get_instance_id(),
           projection["activation"].get_index())
          for projection in projections
      }
      if len(activation_ids) != 1:
        raise ValueError(
            f"layer {layer} {cohort}: projections do not share activation")
      ks = {projection["k"] for projection in projections}
      if len(ks) != 1:
        raise ValueError(f"layer {layer} {cohort}: K values differ")
      groups.append({
          "layer": layer,
          "cohort": cohort,
          "names": list(names),
          "projections": projections,
          "widths": tuple(projection["m"] for projection in projections),
          "k": projections[0]["k"],
      })
  counts = Counter(group["cohort"] for group in groups)
  if dict(counts) != EXPECTED_GROUP_COUNTS:
    raise ValueError(f"fixed FC group census differs: {dict(counts)}")
  if sum(len(group["projections"]) for group in groups) != 390:
    raise ValueError("fixed FC projection census is not 390")
  return groups


def _group_major_scale(ov: Any, np: Any, projection: dict[str, Any]) -> Any:
  m = projection["m"]
  groups = projection["groups"]
  values = np.asarray(
      projection["scale"].get_data(), dtype=np.float16
  ).reshape(m, groups, 1).transpose(1, 0, 2).copy().reshape(
      1, 1, groups, m)
  result = ov.opset13.constant(values, dtype=ov.Type.f16)
  result.set_friendly_name(
      projection["scale"].get_friendly_name() + "/iq36_group_major")
  return result


def _packed_weight(ov: Any, np: Any, projection: dict[str, Any]) -> Any:
  values = np.asarray(
      projection["weight"].get_data(), dtype=np.uint8
  ).reshape(1, 1, 1, -1).copy()
  result = ov.opset13.constant(values, dtype=ov.Type.u8)
  result.set_friendly_name(
      projection["weight"].get_friendly_name() + "/iq36_packed_u8")
  return result


def _group_major_zero_point(
    ov: Any, np: Any, projection: dict[str, Any],
) -> Any:
  m = projection["m"]
  groups = projection["groups"]
  logical_count = m * groups
  packed = np.asarray(
      projection["zero_point"].get_data(), dtype=np.uint8).reshape(-1)
  logical = np.empty(packed.size * 2, dtype=np.uint8)
  logical[0::2] = packed & np.uint8(0x0F)
  logical[1::2] = packed >> np.uint8(4)
  values = np.ascontiguousarray(
      logical[:logical_count].reshape(m, groups, 1).transpose(1, 0, 2)
  ).reshape(-1)
  if values.size % 2:
    raise ValueError("group-major U4 zero-point stream is not byte aligned")
  packed_values = (
      (values[0::2] & np.uint8(0x0F)) |
      ((values[1::2] & np.uint8(0x0F)) << np.uint8(4))
  ).reshape(1, 1, 1, -1)
  result = ov.opset13.constant(packed_values, dtype=ov.Type.u8)
  result.set_friendly_name(
      projection["zero_point"].get_friendly_name() +
      "/iq36_group_major")
  return result


def rewrite_fixed_fc(
    model: Any, ov: Any, np: Any,
    cohorts: Iterable[str] | None = None,
) -> dict[str, Any]:
  """Replace the selected fixed-FC cohorts with parameterized custom ops."""
  selected = tuple(EXPECTED_GROUP_COUNTS) if cohorts is None else tuple(cohorts)
  if not selected or len(set(selected)) != len(selected):
    raise ValueError("fixed FC cohorts must be non-empty and unique")
  unknown = sorted(set(selected) - set(EXPECTED_GROUP_COUNTS))
  if unknown:
    raise ValueError(f"unknown fixed FC cohorts: {unknown}")
  selected_set = set(selected)
  groups = [
      group for group in discover_fixed_fc_groups(model, ov)
      if group["cohort"] in selected_set
  ]
  classes = fixed_fc_custom_classes(ov)
  rows = []
  for group in groups:
    projections = group["projections"]
    widths = group["widths"]
    k = group["k"]
    arity = len(projections)
    activation = ov.opset13.convert(
        projections[0]["activation"], ov.Type.f16)
    activation.set_friendly_name(
        f"iq36_fixed_fc_input_f16_layer{group['layer']}_"
        f"{group['cohort']}")
    inputs = [activation.output(0)]
    for projection in projections:
      inputs.extend([
          _packed_weight(ov, np, projection).output(0),
          _group_major_scale(ov, np, projection).output(0),
          _group_major_zero_point(ov, np, projection).output(0),
      ])
    work_groups = sum((width + WG_TILE_M - 1) // WG_TILE_M
                      for width in widths)
    global_x = work_groups * LOCAL_X
    carrier = ov.opset13.constant(
        np.zeros((1, 1, 1, global_x), dtype=np.uint8))
    carrier.set_friendly_name(
        f"iq36_fixed_fc_workshape_layer{group['layer']}_"
        f"{group['cohort']}")
    inputs.append(carrier.output(0))
    operation = classes[arity](inputs, widths, k)
    operation.set_friendly_name(
        f"iq36_fixed_fc{arity}_layer{group['layer']}_"
        f"{group['cohort']}")
    consumers = []
    for index, projection in enumerate(projections):
      old_output = projection["matmul"].output(0)
      consumers.append(len(old_output.get_target_inputs()))
      converted = ov.opset13.convert(operation.output(index), ov.Type.f32)
      converted.set_friendly_name(
          projection["matmul"].get_friendly_name() + "/iq36_f16_to_f32")
      tensor_names = old_output.get_names()
      if tensor_names:
        converted.output(0).get_tensor().set_names(tensor_names)
      old_output.replace(converted.output(0))
    rows.append({
        "layer": group["layer"],
        "cohort": group["cohort"],
        "arity": arity,
        "widths": list(widths),
        "k": k,
        "global": [global_x, 1, 8],
        "local": [LOCAL_X, 1, 8],
        "old_consumer_counts": consumers,
        "operation": operation.get_friendly_name(),
        "output_type": "f16",
        "restored_edge_type": "f32",
    })
  model.validate_nodes_and_infer_types()
  after = model.get_ordered_ops()
  old_names = {name for group in groups for name in group["names"]}
  remaining = sorted(node.get_friendly_name() for node in after
                     if node.get_friendly_name() in old_names)
  custom_counts = Counter(node.get_type_name() for node in after
                          if node.get_type_name().startswith("IQ36FixedFC"))
  restore_nodes = [
      node for node in after
      if (node.get_type_name() == "Convert" and
          node.get_friendly_name().endswith("/iq36_f16_to_f32"))
  ]
  expected_custom = dict(Counter(
      f"IQ36FixedFC{len(group['projections'])}" for group in groups))
  if dict(custom_counts) != expected_custom:
    raise ValueError(f"fixed FC custom census differs: {dict(custom_counts)}")
  if remaining:
    raise ValueError(f"fixed FC MatMuls remain live: {remaining[:4]}")
  expected_projection_count = sum(
      len(group["projections"]) for group in groups)
  if len(restore_nodes) != expected_projection_count or any(
      node.get_input_element_type(0) != ov.Type.f16 or
      node.get_output_element_type(0) != ov.Type.f32
      for node in restore_nodes
  ):
    raise ValueError(
        "fixed FC F16-to-F32 restore edge census or types differ: "
        f"observed {len(restore_nodes)}, expected "
        f"{expected_projection_count}")
  return {
      "fixed_fc_selected_cohorts": list(selected),
      "fixed_fc_rewrite_count": len(rows),
      "fixed_fc_projection_count": sum(row["arity"] for row in rows),
      "fixed_fc_group_counts": dict(Counter(row["cohort"] for row in rows)),
      "fixed_fc_custom_counts": dict(custom_counts),
      "fixed_fc_old_matmuls_remaining": remaining,
      "fixed_fc_f16_to_f32_restore_count": len(restore_nodes),
      "fixed_fc_rows": rows,
  }


def read_and_rewrite_locked_model(
    core: Any, ov: Any, np: Any, model_xml: Path = MODEL_XML,
    cohorts: Iterable[str] | None = None,
) -> tuple[Any, dict[str, Any]]:
  model = core.read_model(str(model_xml))
  return model, rewrite_fixed_fc(model, ov, np, cohorts=cohorts)
