"""Graph construction for parameterized OpenVINO hot/cold attention."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any


TARGET_LAYER = 3
FULL_ATTENTION_LAYERS = tuple(range(3, 40, 4))
LINEAR_ATTENTION_LAYERS = tuple(
    layer for layer in range(40) if layer not in FULL_ATTENTION_LAYERS)
LINEAR_CONV_FEATURES = 8192
LINEAR_CONV_STATE = 4
HOT_WINDOW = 8192
ADAPTIVE_HOT_WINDOW = 16384
SINK_TOKENS = 1
PREFILL_CHUNK_TOKENS = 8192
# Keep one full continuation chunk outside the logical hot window.  During a
# chunked prefill every attention work-group reads the old hot window while
# the owner work-groups publish the new chunk.  The guard makes those writes
# non-overlapping without a device-wide work-group barrier.
RING_CAPACITY = HOT_WINDOW + PREFILL_CHUNK_TOKENS
HOT_CAPACITY = SINK_TOKENS + RING_CAPACITY
KV_HEADS = 2
Q_HEADS = 16
HEAD_DIM = 256
SCALE_BYTES = 16
GROUP16_SCALE_BYTES = 32
RESIDUAL1_BYTES = HEAD_DIM // 8
GROUP4_SCALE_BYTES = 128
GROUP2_SCALE_BYTES = 256
KEY_TILE_TOKENS = 16
HOT_KEY_BLOCKS = (HOT_CAPACITY + KEY_TILE_TOKENS - 1) // KEY_TILE_TOKENS
HOT_KEY_WORDS_PER_BLOCK = (HEAD_DIM // 2) * KEY_TILE_TOKENS
# One Variable owns the decode-packed plane, a prefill-contiguous F16 plane,
# and one isolated block for the decode arrival counter.
HOT_KEY_STORAGE_BLOCKS = 2 * HOT_KEY_BLOCKS + 1
PREFILL_QUERY_TILE = 32
GQA_GROUP = 8
DECODE_CHUNK_TOKENS = 512
PARTIAL_HEAD_WIDTH = 2 + HEAD_DIM
WORKSPACE_WIDTH = 2 + GQA_GROUP * PARTIAL_HEAD_WIDTH
STOCK256_WORKSPACE_WIDTH = 2 + 2 * GQA_GROUP * PARTIAL_HEAD_WIDTH
ADAPTIVE_LOCAL_TOPK = 64
ADAPTIVE_HIGH_TOPK_LAYERS = FULL_ATTENTION_LAYERS
ADAPTIVE_PACKED_KV_VARIANTS = ("k6v7", "k7v7", "k7v8", "k8v7")


def adaptive_workspace_f32_elements(max_chunks: int) -> int:
  """Packed output0 scratch for the four-stage adaptive decode owner."""
  if max_chunks < ADAPTIVE_HOT_WINDOW // DECODE_CHUNK_TOKENS:
    raise ValueError("adaptive decode carrier is shorter than the hot window")
  max_cold_chunks = max_chunks - ADAPTIVE_HOT_WINDOW // DECODE_CHUNK_TOKENS
  max_cold_tokens = max_cold_chunks * DECODE_CHUNK_TOKENS
  max_partitions = max_chunks * 2
  max_tokens = max_chunks * DECODE_CHUNK_TOKENS
  partial_meta = KV_HEADS * max_partitions * GQA_GROUP
  byte_count = sum((
      partial_meta * 4,                         # partial max
      partial_meta * 4,                         # partial sum
      partial_meta * HEAD_DIM * 4,              # partial numerator
      Q_HEADS * max_tokens * 4,                 # unscaled F32 KQ scores
      Q_HEADS * max_cold_chunks *
          ADAPTIVE_LOCAL_TOPK * 4,               # local candidate records
      KV_HEADS * ((max_cold_tokens + 31) // 32) * 4,
      KV_HEADS * 4,                             # completion counters
      Q_HEADS * HEAD_DIM * 4,                   # F32 attention publication
  ))
  return ((byte_count + 63) // 64) * 64 // 4

def stock_state_names(layer: int) -> tuple[str, str]:
  index = FULL_ATTENTION_LAYERS.index(layer)
  return (
      f"cache_params.past.key.{index}cache_params.present.key.{index}",
      f"cache_params.past.value.{index}cache_params.present.value.{index}",
  )


def layer_state_names(layer: int) -> tuple[str, ...]:
  return (
      f"iq36.hot.key.bits.layer{layer}",
      f"iq36.hot.value.bits.layer{layer}",
      f"iq36.cold.key.i8.layer{layer}",
      f"iq36.cold.value.i8.layer{layer}",
      f"iq36.cold.key.scale.bits.layer{layer}",
      f"iq36.cold.value.scale.bits.layer{layer}",
  )


def hot_state_names(layers: tuple[int, ...]) -> tuple[str, ...]:
  return tuple(name for layer in layers for name in layer_state_names(layer)[:2])


def cold_state_names(layers: tuple[int, ...]) -> tuple[str, ...]:
  return tuple(name for layer in layers for name in layer_state_names(layer)[2:4])


def scale_state_names(layers: tuple[int, ...]) -> tuple[str, ...]:
  return tuple(name for layer in layers for name in layer_state_names(layer)[4:])


def custom_state_names(layers: tuple[int, ...]) -> tuple[str, ...]:
  return tuple(name for layer in layers for name in layer_state_names(layer))


def hot_slots(tokens: Any, np: Any) -> Any:
  values = np.asarray(tokens, dtype=np.int64)
  return np.where(
      values < SINK_TOKENS,
      values,
      SINK_TOKENS + (values - SINK_TOKENS) % RING_CAPACITY)


STOCK_KEY, STOCK_VALUE = stock_state_names(TARGET_LAYER)
(HOT_KEY, HOT_VALUE, COLD_KEY, COLD_VALUE,
 COLD_KEY_SCALE, COLD_VALUE_SCALE) = layer_state_names(TARGET_LAYER)
HOT_STATES = (HOT_KEY, HOT_VALUE)
COLD_STATES = (COLD_KEY, COLD_VALUE)
SCALE_STATES = (COLD_KEY_SCALE, COLD_VALUE_SCALE)
CUSTOM_STATES = HOT_STATES + COLD_STATES + SCALE_STATES

PAST_SHAPE_NODE = "ShapeOf_318031"
PRESENT_SHAPE_NODE = (
    "__module.model.model.language_model.layers.3.self_attn/aten::size/"
    "ShapeOf_2")


def _fixed_fc_module() -> Any:
  path = Path(__file__).with_name("intel_qwen36_openvino_fixed_fc.py")
  spec = importlib.util.spec_from_file_location("iq36_fixed_fc_graph", path)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load fixed FC graph module from {path}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def custom_class(
    ov: Any, key_scale_bytes: int = SCALE_BYTES,
    value_scale_bytes: int | None = None,
) -> type:
  if value_scale_bytes is None:
    value_scale_bytes = key_scale_bytes

  class IQ36HotAttentionGQA(ov.Op):
    def __init__(self, inputs: Any = None):
      super().__init__(self, inputs)
      if inputs is not None:
        self.constructor_validate_and_infer_types()

    def validate_and_infer_types(self) -> None:
      self.set_output_size(6)
      query = self.get_input_partial_shape(0)
      query_tokens = query[2]
      workspace_heads = ov.Dimension.dynamic()
      decode_chunks = self.get_input_partial_shape(12)[3]
      work_groups = ov.Dimension.dynamic()
      workspace_width = ov.Dimension.dynamic()
      if query_tokens.is_static:
        query_length = query_tokens.get_length()
        if query_length != 1:
          workspace_heads = ov.Dimension(Q_HEADS)
          work_groups = ov.Dimension(
              (query_length + PREFILL_QUERY_TILE - 1) //
              PREFILL_QUERY_TILE)
          workspace_width = ov.Dimension(1)
        else:
          workspace_heads = ov.Dimension(KV_HEADS)
          workspace_width = ov.Dimension(WORKSPACE_WIDTH)
          if decode_chunks.is_static:
            work_groups = decode_chunks
      self.set_output_type(
          0, ov.Type.f32,
          ov.PartialShape([
              query[0], workspace_heads, work_groups, workspace_width]))
      self.set_output_type(
          1, self.get_input_element_type(0), query)
      scratch = self.get_input_partial_shape(10)
      self.set_output_type(2, ov.Type.i8, scratch)
      self.set_output_type(3, ov.Type.i8, scratch)
      key_scale_shape = ov.PartialShape([
          scratch[0], scratch[1], scratch[2], key_scale_bytes])
      value_scale_shape = ov.PartialShape([
          scratch[0], scratch[1], scratch[2], value_scale_bytes])
      self.set_output_type(4, ov.Type.i8, key_scale_shape)
      self.set_output_type(5, ov.Type.i8, value_scale_shape)

    def clone_with_new_inputs(self, new_inputs: Any) -> Any:
      return IQ36HotAttentionGQA(new_inputs)

    def visit_attributes(self, visitor: Any) -> bool:
      return True

  return IQ36HotAttentionGQA


def gated_output_custom_class(ov: Any) -> type:
  """Custom attention variant that returns gated token-major output."""

  class IQ36GatedHotAttentionGQA(ov.Op):
    def __init__(self, inputs: Any = None):
      super().__init__(self, inputs)
      if inputs is not None:
        self.constructor_validate_and_infer_types()

    def validate_and_infer_types(self) -> None:
      self.set_output_size(6)
      query = self.get_input_partial_shape(0)
      query_tokens = query[2]
      workspace_heads = ov.Dimension.dynamic()
      decode_chunks = self.get_input_partial_shape(12)[3]
      work_groups = ov.Dimension.dynamic()
      workspace_width = ov.Dimension.dynamic()
      if query_tokens.is_static:
        query_length = query_tokens.get_length()
        if query_length != 1:
          workspace_heads = ov.Dimension(Q_HEADS)
          work_groups = ov.Dimension(
              (query_length + PREFILL_QUERY_TILE - 1) //
              PREFILL_QUERY_TILE)
          workspace_width = ov.Dimension(1)
        else:
          workspace_heads = ov.Dimension(KV_HEADS)
          workspace_width = ov.Dimension(WORKSPACE_WIDTH)
          if decode_chunks.is_static:
            work_groups = decode_chunks
      self.set_output_type(
          0, ov.Type.f32,
          ov.PartialShape([
              query[0], workspace_heads, work_groups, workspace_width]))
      gate = self.get_input_partial_shape(13)
      if (gate.rank.is_static and gate.rank.get_length() == 4 and
          gate[2].is_static and gate[2].get_length() != Q_HEADS):
        raise ValueError("gated attention input must have 16 Q heads")
      self.set_output_type(
          1, self.get_input_element_type(0),
          ov.PartialShape([query[0], query[2], query[1], query[3]]))
      scratch = self.get_input_partial_shape(10)
      self.set_output_type(2, ov.Type.i8, scratch)
      self.set_output_type(3, ov.Type.i8, scratch)
      key_scale_shape = ov.PartialShape([
          scratch[0], scratch[1], scratch[2], SCALE_BYTES])
      self.set_output_type(4, ov.Type.i8, key_scale_shape)
      self.set_output_type(5, ov.Type.i8, key_scale_shape)

    def clone_with_new_inputs(self, new_inputs: Any) -> Any:
      return IQ36GatedHotAttentionGQA(new_inputs)

    def visit_attributes(self, visitor: Any) -> bool:
      return True

  return IQ36GatedHotAttentionGQA


def token_major_value_output_custom_class(ov: Any) -> type:
  """Custom attention with token-major current V and output layouts."""

  class IQ36TokenMajorValueAttentionGQA(ov.Op):
    def __init__(self, inputs: Any = None):
      super().__init__(self, inputs)
      if inputs is not None:
        self.constructor_validate_and_infer_types()

    def validate_and_infer_types(self) -> None:
      self.set_output_size(6)
      query = self.get_input_partial_shape(0)
      query_tokens = query[2]
      current_value = self.get_input_partial_shape(4)
      if (current_value.rank.is_static and
          current_value.rank.get_length() == 4 and
          current_value[2].is_static and
          current_value[2].get_length() != KV_HEADS):
        raise ValueError("token-major current value must have two KV heads")
      workspace_heads = ov.Dimension.dynamic()
      decode_chunks = self.get_input_partial_shape(12)[3]
      work_groups = ov.Dimension.dynamic()
      workspace_width = ov.Dimension.dynamic()
      if query_tokens.is_static:
        query_length = query_tokens.get_length()
        if query_length != 1:
          workspace_heads = ov.Dimension(Q_HEADS)
          work_groups = ov.Dimension(
              (query_length + PREFILL_QUERY_TILE - 1) //
              PREFILL_QUERY_TILE)
          workspace_width = ov.Dimension(1)
        else:
          workspace_heads = ov.Dimension(KV_HEADS)
          workspace_width = ov.Dimension(WORKSPACE_WIDTH)
          if decode_chunks.is_static:
            work_groups = decode_chunks
      self.set_output_type(
          0, ov.Type.f32,
          ov.PartialShape([
              query[0], workspace_heads, work_groups, workspace_width]))
      self.set_output_type(
          1, self.get_input_element_type(0),
          ov.PartialShape([query[0], query[2], query[1], query[3]]))
      scratch = self.get_input_partial_shape(10)
      self.set_output_type(2, ov.Type.i8, scratch)
      self.set_output_type(3, ov.Type.i8, scratch)
      scale_shape = ov.PartialShape([
          scratch[0], scratch[1], scratch[2], SCALE_BYTES])
      self.set_output_type(4, ov.Type.i8, scale_shape)
      self.set_output_type(5, ov.Type.i8, scale_shape)

    def clone_with_new_inputs(self, new_inputs: Any) -> Any:
      return IQ36TokenMajorValueAttentionGQA(new_inputs)

    def visit_attributes(self, visitor: Any) -> bool:
      return True

  return IQ36TokenMajorValueAttentionGQA


def attention_gated_dynamic_quantize_custom_class(ov: Any) -> type:
  """Transpose, gate, and group-64 quantize an attention output."""

  class IQ36GatedTransposeDynamicQuantize(ov.Op):
    def __init__(self, inputs: Any = None):
      super().__init__(self, inputs)
      if inputs is not None:
        self.constructor_validate_and_infer_types()

    def validate_and_infer_types(self) -> None:
      self.set_output_size(4)
      attention = self.get_input_partial_shape(0)
      gate = self.get_input_partial_shape(1)
      if attention.rank.is_static and attention.rank.get_length() != 4:
        raise ValueError("gated-DQ attention input must be rank four")
      if gate.rank.is_static and gate.rank.get_length() != 3:
        raise ValueError("gated-DQ gate input must be rank three")
      if (attention.rank.is_static and gate.rank.is_static and
          attention[1].is_static and attention[1].get_length() != Q_HEADS):
        raise ValueError("gated-DQ attention input must have 16 Q heads")
      if (attention.rank.is_static and gate.rank.is_static and
          attention[3].is_static and attention[3].get_length() != HEAD_DIM):
        raise ValueError("gated-DQ attention head dimension must be 256")
      hidden = ov.Dimension(Q_HEADS * HEAD_DIM)
      group_count = ov.Dimension(Q_HEADS * HEAD_DIM // 64)
      carrier_shape = ov.PartialShape([attention[0], attention[2], hidden])
      group_shape = ov.PartialShape([
          attention[0], attention[2], group_count])
      if (gate.rank.is_static and not gate.compatible(carrier_shape)):
        raise ValueError("gated-DQ gate shape must be [B,Q,4096]")
      # Output zero is the graph-level F32 shape carrier matched by the GPU
      # FullyConnectedCompressed pass.  The pass rewires the FC to outputs
      # one through three before custom-layer lowering, so the kernel never
      # materializes this otherwise-dead carrier.
      self.set_output_type(
          0, self.get_input_element_type(1), carrier_shape)
      self.set_output_type(1, ov.Type.i8, carrier_shape)
      self.set_output_type(2, ov.Type.f16, group_shape)
      self.set_output_type(3, ov.Type.i32, group_shape)

    def clone_with_new_inputs(self, new_inputs: Any) -> Any:
      return IQ36GatedTransposeDynamicQuantize(new_inputs)

    def visit_attributes(self, visitor: Any) -> bool:
      return True

  return IQ36GatedTransposeDynamicQuantize


def qk_rope_layout_custom_class(ov: Any) -> type:
  """Fuse token-major Q/K layout conversion with partial rotate-half RoPE."""

  class IQ36QKRopeLayout(ov.Op):
    def __init__(self, inputs: Any = None):
      super().__init__(self, inputs)
      if inputs is not None:
        self.constructor_validate_and_infer_types()

    def validate_and_infer_types(self) -> None:
      self.set_output_size(2)
      query = self.get_input_partial_shape(0)
      key = self.get_input_partial_shape(1)
      cosine = self.get_input_partial_shape(2)
      sine = self.get_input_partial_shape(3)
      if any(shape.rank.is_static and shape.rank.get_length() != 4
             for shape in (query, key, cosine, sine)):
        raise ValueError("Q/K RoPE-layout inputs must be rank four")
      if (query.rank.is_static and query[2].is_static and
          query[2].get_length() != Q_HEADS):
        raise ValueError("Q/K RoPE-layout query must have 16 heads")
      if (key.rank.is_static and key[2].is_static and
          key[2].get_length() != KV_HEADS):
        raise ValueError("Q/K RoPE-layout key must have two heads")
      if (query.rank.is_static and query[3].is_static and
          query[3].get_length() != HEAD_DIM):
        raise ValueError("Q/K RoPE-layout query head width must be 256")
      if (key.rank.is_static and key[3].is_static and
          key[3].get_length() != HEAD_DIM):
        raise ValueError("Q/K RoPE-layout key head width must be 256")
      for name, shape in (("cosine", cosine), ("sine", sine)):
        if (shape.rank.is_static and shape[3].is_static and
            shape[3].get_length() != 64):
          raise ValueError(f"Q/K RoPE-layout {name} width must be 64")
      self.set_output_type(
          0, self.get_input_element_type(0),
          ov.PartialShape([query[0], query[2], query[1], query[3]]))
      self.set_output_type(
          1, self.get_input_element_type(1),
          ov.PartialShape([key[0], key[2], key[1], key[3]]))

    def clone_with_new_inputs(self, new_inputs: Any) -> Any:
      return IQ36QKRopeLayout(new_inputs)

    def visit_attributes(self, visitor: Any) -> bool:
      return True

  return IQ36QKRopeLayout


def direct_i8_custom_class(ov: Any) -> type:
  base = custom_class(ov)

  class IQ36DirectI8HotAttentionGQA(base):
    def clone_with_new_inputs(self, new_inputs: Any) -> Any:
      return IQ36DirectI8HotAttentionGQA(new_inputs)

  return IQ36DirectI8HotAttentionGQA


def adaptive_attention_custom_class(
    ov: Any, topk: int, value_quant_group: int = 32,
    key_residual1: bool = False, value_residual1: bool = False,
    key_exact: bool = False, packed_kv_variant: str | None = None,
) -> type:
  """Single graph owner whose decode is four plugin-internal kernels."""
  if topk not in (128, 252, 256, 512, 1024, 2048):
    raise ValueError(
        "adaptive attention top-k must be 128, 252, 256, 512, 1024, or 2048")
  if value_quant_group not in (16, 32):
    raise ValueError("adaptive attention V quant group must be 16 or 32")
  if value_quant_group == 16 and topk != 512:
    raise ValueError("adaptive V16 is admitted only with top-512 correction")
  if (key_residual1 or value_residual1) and (
      topk not in (256, 512) or value_quant_group != 32):
    raise ValueError(
        "adaptive residual1 is admitted only with top-256/512 K32/V32")
  if key_exact and (
      topk != 256 or value_quant_group != 32 or
      key_residual1 or value_residual1):
    raise ValueError(
        "adaptive key-exact is admitted only with top-256 K32/V32")
  if packed_kv_variant not in (None, *ADAPTIVE_PACKED_KV_VARIANTS):
    raise ValueError(
        f"unknown adaptive packed K/V variant: {packed_kv_variant}")
  if packed_kv_variant is not None and (
      topk not in (256, 512) or value_quant_group != 32 or key_exact or
      key_residual1 or value_residual1):
    raise ValueError(
        "adaptive packed K/V is admitted only with top-256/512 K32/V32")
  if (packed_kv_variant is not None and topk == 512 and
      packed_kv_variant != "k7v8"):
    raise ValueError("only packed K7/V8 currently admits top-512 correction")
  base = custom_class(
      ov, SCALE_BYTES + (RESIDUAL1_BYTES if key_residual1 else 0),
      (GROUP16_SCALE_BYTES if value_quant_group == 16 else SCALE_BYTES) +
      (RESIDUAL1_BYTES if value_residual1 else 0))

  class IQ36AdaptiveHotAttentionGQA(base):
    def validate_and_infer_types(self) -> None:
      super().validate_and_infer_types()
      query = self.get_input_partial_shape(0)
      query_tokens = query[2]
      if query_tokens.is_static and query_tokens.get_length() == 1:
        decode_chunks = self.get_input_partial_shape(12)[3]
        packed = ov.Dimension.dynamic()
        if decode_chunks.is_static:
          packed = ov.Dimension(adaptive_workspace_f32_elements(
              decode_chunks.get_length()))
        self.set_output_type(
            0, ov.Type.f32,
            ov.PartialShape([
                query[0], ov.Dimension(1), ov.Dimension(1), packed]))

    def clone_with_new_inputs(self, new_inputs: Any) -> Any:
      return IQ36AdaptiveHotAttentionGQA(new_inputs)

  variant_prefix = (
      packed_kv_variant.upper() if packed_kv_variant is not None else
      "KeyExact" if key_exact else
      "KVResidual1" if key_residual1 and value_residual1 else
      "KResidual1" if key_residual1 else
      "VResidual1" if value_residual1 else "")
  IQ36AdaptiveHotAttentionGQA.__name__ = (
      f"IQ36AdaptiveV16Top{topk}HotAttentionGQA"
      if value_quant_group == 16 else
      f"IQ36Adaptive{variant_prefix}Top{topk}HotAttentionGQA")
  return IQ36AdaptiveHotAttentionGQA


def decode_chunk256_custom_class(ov: Any) -> type:
  """Unified attention whose decode workspace schedules 256-token chunks."""
  base = custom_class(ov)

  class IQ36DecodeChunk256HotAttentionGQA(base):
    def validate_and_infer_types(self) -> None:
      super().validate_and_infer_types()
      query = self.get_input_partial_shape(0)
      query_tokens = query[2]
      if (query_tokens.is_static and query_tokens.get_length() == 1):
        decode_chunks = self.get_input_partial_shape(12)[3]
        work_groups = ov.Dimension.dynamic()
        if decode_chunks.is_static:
          work_groups = ov.Dimension(2 * decode_chunks.get_length())
        self.set_output_type(
            0, ov.Type.f32,
            ov.PartialShape([
                query[0], ov.Dimension(KV_HEADS), work_groups,
                ov.Dimension(WORKSPACE_WIDTH)]))

    def clone_with_new_inputs(self, new_inputs: Any) -> Any:
      return IQ36DecodeChunk256HotAttentionGQA(new_inputs)

  return IQ36DecodeChunk256HotAttentionGQA


def stock_micro_owner_custom_class(ov: Any) -> type:
  """Single-owner attention whose decode is the actual stock microkernel."""
  base = custom_class(ov)

  class IQ36StockMicroOwnerHotAttentionGQA(base):
    def validate_and_infer_types(self) -> None:
      super().validate_and_infer_types()
      query = self.get_input_partial_shape(0)
      query_tokens = query[2]
      if query_tokens.is_static and query_tokens.get_length() == 1:
        self.set_output_type(
            0, ov.Type.f32,
            ov.PartialShape([
                query[0], ov.Dimension(KV_HEADS), ov.Dimension(1),
                ov.Dimension(WORKSPACE_WIDTH)]))

    def clone_with_new_inputs(self, new_inputs: Any) -> Any:
      return IQ36StockMicroOwnerHotAttentionGQA(new_inputs)

  return IQ36StockMicroOwnerHotAttentionGQA


def exact_phase_custom_class(ov: Any) -> type:
  """Fast custom prefill and exact stock-micro decode in one state owner."""
  base = stock_micro_owner_custom_class(ov)

  class IQ36ExactPhaseHotAttentionGQA(base):
    def clone_with_new_inputs(self, new_inputs: Any) -> Any:
      return IQ36ExactPhaseHotAttentionGQA(new_inputs)

  return IQ36ExactPhaseHotAttentionGQA


def exact_phase_dual_cohort_custom_class(ov: Any) -> type:
  """Fast prefill plus exact dual-cohort stock-micro decode."""
  base = stock_micro_owner_custom_class(ov)

  class IQ36ExactPhaseDualCohortHotAttentionGQA(base):
    def clone_with_new_inputs(self, new_inputs: Any) -> Any:
      return IQ36ExactPhaseDualCohortHotAttentionGQA(new_inputs)

  return IQ36ExactPhaseDualCohortHotAttentionGQA


def exact_phase_page_sparse_custom_class(ov: Any) -> type:
  """Exact phase owner with sampled-page decode on admitted long rows."""
  base = stock_micro_owner_custom_class(ov)

  class IQ36ExactPhasePageSparseHotAttentionGQA(base):
    def clone_with_new_inputs(self, new_inputs: Any) -> Any:
      return IQ36ExactPhasePageSparseHotAttentionGQA(new_inputs)

  return IQ36ExactPhasePageSparseHotAttentionGQA


def exact_phase_context_partition4_custom_class(ov: Any) -> type:
  """Exact phase owner with four context partitions per KV head."""
  base = stock_micro_owner_custom_class(ov)

  class IQ36ExactPhaseContextPartition4HotAttentionGQA(base):
    def validate_and_infer_types(self) -> None:
      super().validate_and_infer_types()
      query = self.get_input_partial_shape(0)
      query_tokens = query[2]
      if query_tokens.is_static and query_tokens.get_length() == 1:
        self.set_output_type(
            0, ov.Type.f32,
            ov.PartialShape([
                query[0], ov.Dimension(KV_HEADS), ov.Dimension(4),
                ov.Dimension(WORKSPACE_WIDTH)]))

    def clone_with_new_inputs(self, new_inputs: Any) -> Any:
      return IQ36ExactPhaseContextPartition4HotAttentionGQA(new_inputs)

  return IQ36ExactPhaseContextPartition4HotAttentionGQA


def f32_numerator_chunk256_custom_class(ov: Any) -> type:
  """256-token decode chunks with an F32 softmax-value numerator oracle."""
  base = decode_chunk256_custom_class(ov)

  class IQ36F32NumeratorChunk256HotAttentionGQA(base):
    def clone_with_new_inputs(self, new_inputs: Any) -> Any:
      return IQ36F32NumeratorChunk256HotAttentionGQA(new_inputs)

  return IQ36F32NumeratorChunk256HotAttentionGQA


def dual256_custom_class(ov: Any) -> type:
  """512-token carrier with two in-work-group 256-token reductions."""
  base = custom_class(ov)

  class IQ36Dual256HotAttentionGQA(base):
    def clone_with_new_inputs(self, new_inputs: Any) -> Any:
      return IQ36Dual256HotAttentionGQA(new_inputs)

  return IQ36Dual256HotAttentionGQA


def stock256_partials_custom_class(ov: Any) -> type:
  """512-token work-group that exports two stock-shaped F16 partials."""
  base = custom_class(ov)

  class IQ36Stock256PartialsHotAttentionGQA(base):
    def validate_and_infer_types(self) -> None:
      super().validate_and_infer_types()
      query = self.get_input_partial_shape(0)
      query_tokens = query[2]
      if query_tokens.is_static and query_tokens.get_length() == 1:
        decode_chunks = self.get_input_partial_shape(12)[3]
        work_groups = ov.Dimension.dynamic()
        if decode_chunks.is_static:
          work_groups = decode_chunks
        self.set_output_type(
            0, ov.Type.f32,
            ov.PartialShape([
                query[0], ov.Dimension(KV_HEADS), work_groups,
                ov.Dimension(STOCK256_WORKSPACE_WIDTH)]))

    def clone_with_new_inputs(self, new_inputs: Any) -> Any:
      return IQ36Stock256PartialsHotAttentionGQA(new_inputs)

  return IQ36Stock256PartialsHotAttentionGQA


def stock_score_chunk256_custom_class(ov: Any) -> type:
  """256-token carrier with stock scalar-FMA QK accumulation order."""
  base = decode_chunk256_custom_class(ov)

  class IQ36StockScoreChunk256HotAttentionGQA(base):
    def clone_with_new_inputs(self, new_inputs: Any) -> Any:
      return IQ36StockScoreChunk256HotAttentionGQA(new_inputs)

  return IQ36StockScoreChunk256HotAttentionGQA


def stock_partition_chunk256_custom_class(ov: Any) -> type:
  """256-token carrier with the complete stock partition arithmetic."""
  base = decode_chunk256_custom_class(ov)

  class IQ36StockPartitionChunk256HotAttentionGQA(base):
    def clone_with_new_inputs(self, new_inputs: Any) -> Any:
      return IQ36StockPartitionChunk256HotAttentionGQA(new_inputs)

  return IQ36StockPartitionChunk256HotAttentionGQA


def stock_micro_attention_oracle_custom_class(ov: Any) -> type:
  """Decode-only exact stock sdpa_micro arithmetic over custom F16 state."""
  class IQ36StockMicroAttentionOracle(ov.Op):
    def __init__(self, inputs: Any = None):
      super().__init__(self, inputs)
      if inputs is not None:
        self.constructor_validate_and_infer_types()

    def validate_and_infer_types(self) -> None:
      query = self.get_input_partial_shape(0)
      dependency = self.get_input_partial_shape(4)
      if (query.rank.is_static and query.rank.get_length() != 4):
        raise ValueError("stock-micro oracle query must be rank four")
      if (dependency.rank.is_static and dependency.rank.get_length() != 4):
        raise ValueError("stock-micro oracle dependency must be rank four")
      if not query.compatible(dependency):
        raise ValueError(
            "stock-micro oracle query and dependency shapes must be compatible")
      self.set_output_size(1)
      # The dependency is the existing attention result and therefore the
      # exact public contract that this replacement must preserve.  Deriving
      # the output from it also keeps the GPU plugin's dynamic-shape
      # specialization from inheriting the query's internal F16 layout.
      self.set_output_type(
          0, self.get_input_element_type(4), dependency)

    def clone_with_new_inputs(self, new_inputs: Any) -> Any:
      return IQ36StockMicroAttentionOracle(new_inputs)

    def visit_attributes(self, visitor: Any) -> bool:
      return True

  return IQ36StockMicroAttentionOracle


def direct_i8_group4_custom_class(ov: Any) -> type:
  base = custom_class(ov, GROUP4_SCALE_BYTES)

  class IQ36DirectI8Group4HotAttentionGQA(base):
    def clone_with_new_inputs(self, new_inputs: Any) -> Any:
      return IQ36DirectI8Group4HotAttentionGQA(new_inputs)

  return IQ36DirectI8Group4HotAttentionGQA


def direct_i8_hybrid_k2_v4_custom_class(ov: Any) -> type:
  base = custom_class(ov, GROUP2_SCALE_BYTES, GROUP4_SCALE_BYTES)

  class IQ36DirectI8HybridK2V4HotAttentionGQA(base):
    def clone_with_new_inputs(self, new_inputs: Any) -> Any:
      return IQ36DirectI8HybridK2V4HotAttentionGQA(new_inputs)

  return IQ36DirectI8HybridK2V4HotAttentionGQA


def state_decode_custom_class(ov: Any) -> type:
  base = custom_class(ov)

  class IQ36StateDecodeAttentionGQA(base):
    def clone_with_new_inputs(self, new_inputs: Any) -> Any:
      return IQ36StateDecodeAttentionGQA(new_inputs)

  return IQ36StateDecodeAttentionGQA


def if_bridge_custom_class(ov: Any) -> type:
  class IQ36IfBridge(ov.Op):
    def __init__(self, inputs: Any = None):
      super().__init__(self, inputs)
      if inputs is not None:
        self.constructor_validate_and_infer_types()

    def validate_and_infer_types(self) -> None:
      self.set_output_size(1)
      self.set_output_type(
          0, self.get_input_element_type(0),
          self.get_input_partial_shape(0))

    def clone_with_new_inputs(self, new_inputs: Any) -> Any:
      return IQ36IfBridge(new_inputs)

    def visit_attributes(self, visitor: Any) -> bool:
      return True

  return IQ36IfBridge


def linear_conv_custom_class(ov: Any) -> type:
  class IQ36LinearConvSwish(ov.Op):
    def __init__(self, inputs: Any = None):
      super().__init__(self, inputs)
      if inputs is not None:
        self.constructor_validate_and_infer_types()

    def validate_and_infer_types(self) -> None:
      self.set_output_size(2)
      self.set_output_type(
          0, self.get_input_element_type(0),
          self.get_input_partial_shape(0))
      self.set_output_type(
          1, self.get_input_element_type(1),
          self.get_input_partial_shape(1))

    def clone_with_new_inputs(self, new_inputs: Any) -> Any:
      return IQ36LinearConvSwish(new_inputs)

    def visit_attributes(self, visitor: Any) -> bool:
      return True

  return IQ36LinearConvSwish


def prefill_custom_classes(ov: Any) -> tuple[type, type]:
  class IQ36PrefillAttentionBase(ov.Op):
    def __init__(self, inputs: Any = None):
      super().__init__(self, inputs)
      if inputs is not None:
        self.constructor_validate_and_infer_types()

    def validate_and_infer_types(self) -> None:
      self.set_output_size(5)
      query = self.get_input_partial_shape(0)
      query_tokens = query[2]
      query_tiles = ov.Dimension.dynamic()
      if query_tokens.is_static:
        query_tiles = ov.Dimension(
            (query_tokens.get_length() + PREFILL_QUERY_TILE - 1) //
            PREFILL_QUERY_TILE)
      self.set_output_type(
          0, self.get_input_element_type(0),
          ov.PartialShape([
              query[0], query[1], query_tiles,
              PREFILL_QUERY_TILE * HEAD_DIM]))
      scratch = self.get_input_partial_shape(10)
      self.set_output_type(1, ov.Type.i8, scratch)
      self.set_output_type(2, ov.Type.i8, scratch)
      scale_shape = ov.PartialShape([
          scratch[0], scratch[1], scratch[2], SCALE_BYTES])
      self.set_output_type(3, ov.Type.i8, scale_shape)
      self.set_output_type(4, ov.Type.i8, scale_shape)

    def visit_attributes(self, visitor: Any) -> bool:
      return True

  class IQ36InitialPrefillAttentionGQA(IQ36PrefillAttentionBase):
    def clone_with_new_inputs(self, new_inputs: Any) -> Any:
      return IQ36InitialPrefillAttentionGQA(new_inputs)

  class IQ36ContinuationPrefillAttentionGQA(IQ36PrefillAttentionBase):
    def clone_with_new_inputs(self, new_inputs: Any) -> Any:
      return IQ36ContinuationPrefillAttentionGQA(new_inputs)

  return (IQ36InitialPrefillAttentionGQA,
          IQ36ContinuationPrefillAttentionGQA)


def _variable(ov: Any, variable_id: str, shape: list[int], dtype: Any) -> Any:
  info = ov.op.util.VariableInfo()
  info.variable_id = variable_id
  info.data_type = dtype
  info.data_shape = ov.PartialShape(shape)
  return ov.op.util.Variable(info)


def _scalar(ov: Any, np: Any, data: Any, index: int, axis: Any) -> Any:
  return ov.opset13.gather(
      data, ov.opset13.constant(np.array(index, dtype=np.int64)), axis)


def _vector(ov: Any, np: Any, value: Any) -> Any:
  return ov.opset13.unsqueeze(
      value, ov.opset13.constant(np.array([0], dtype=np.int64)))


def fuse_linear_conv_state_boundaries(
    model: Any, ov: Any, np: Any,
) -> list[dict[str, Any]]:
  """Replace every linear-attention conv/state boundary with one custom op."""
  operations = {node.get_friendly_name(): node
                for node in model.get_ordered_ops()}
  operation_type = linear_conv_custom_class(ov)
  qkv_shape = ov.opset13.constant(
      np.array([1, 1, -1, LINEAR_CONV_FEATURES], dtype=np.int64))
  state_shape = ov.opset13.constant(
      np.array([1, 1, LINEAR_CONV_FEATURES, LINEAR_CONV_STATE],
               dtype=np.int64))
  output_shape = ov.opset13.constant(
      np.array([1, -1, LINEAR_CONV_FEATURES], dtype=np.int64))
  assign_shape = ov.opset13.constant(
      np.array([1, LINEAR_CONV_FEATURES, LINEAR_CONV_STATE],
               dtype=np.int64))
  rows = []
  for layer in LINEAR_ATTENTION_LAYERS:
    prefix = (
        f"__module.model.model.language_model.layers.{layer}.linear_attn/")

    def named(suffix: str) -> Any:
      name = prefix + suffix
      if name not in operations:
        raise ValueError(f"linear conv boundary node missing: {name}")
      return operations[name]

    input_transpose = named("aten::transpose/Transpose")
    history_concat = named("aten::cat/Concat")
    convolution = named("aten::_convolution/GroupConvolution")
    output_transpose = named("aten::transpose/Transpose_1")
    state_slice = named("aten::slice/Slice_7")

    qkv_f16 = ov.opset13.convert(
        input_transpose.input_value(0), ov.Type.f16)
    qkv = ov.opset13.reshape(qkv_f16, qkv_shape, False)
    state_f16 = ov.opset13.convert(
        history_concat.input_value(0), ov.Type.f16)
    state = ov.opset13.reshape(state_f16, state_shape, False)
    weights_f16 = ov.opset13.convert(
        convolution.input_value(1), ov.Type.f16)
    weights = ov.opset13.reshape(weights_f16, state_shape, False)
    operation = operation_type([qkv, state, weights])
    operation.set_friendly_name(f"iq36_linear_conv_swish_layer{layer}")

    activated = ov.opset13.convert(
        ov.opset13.reshape(operation.output(0), output_shape, False),
        ov.Type.f32)
    activated.set_friendly_name(
        f"iq36_linear_conv_activated_layer{layer}")
    next_state = ov.opset13.convert(
        ov.opset13.reshape(operation.output(1), assign_shape, False),
        ov.Type.f32)
    next_state.set_friendly_name(
        f"iq36_linear_conv_next_state_layer{layer}")
    output_transpose.output(0).replace(activated.output(0))
    state_slice.output(0).replace(next_state.output(0))
    rows.append({
        "layer": layer,
        "operation": operation.get_friendly_name(),
        "qkv_source": (
            input_transpose.input_value(0).get_node().get_friendly_name()),
        "state_source": (
            history_concat.input_value(0).get_node().get_friendly_name()),
        "weights_source": (
            convolution.input_value(1).get_node().get_friendly_name()),
    })
  model.validate_nodes_and_infer_types()
  return rows


def bypass_ssm_state_assign_repacks(model: Any) -> list[dict[str, Any]]:
  """Feed each SSM Assign from the Loop's final-state output directly.

  The locked IR flattens both Loop outputs, concatenates them, slices the
  final-state segment back out, and reshapes it to the original state shape.
  Loop output 1 already has that exact shape.  Bypassing the repack exposes
  the real state producer to the runtime without changing arithmetic.
  """
  rows = []
  for assign in model.get_sinks():
    if assign.get_type_name() != "Assign":
      continue
    variable_id = assign.get_variable_id()
    if "cache_params.past.ssm." not in variable_id:
      continue
    final_reshape = assign.input_value(0).get_node()
    if final_reshape.get_type_name() != "Reshape":
      raise ValueError(f"unexpected SSM Assign tail for {variable_id}")
    state_slice = final_reshape.input_value(0).get_node()
    if state_slice.get_type_name() != "Slice":
      raise ValueError(f"unexpected SSM Assign slice for {variable_id}")
    repack = state_slice.input_value(0).get_node()
    if repack.get_type_name() != "Concat":
      raise ValueError(f"unexpected SSM Assign repack for {variable_id}")

    final_state_outputs = []
    for index in range(repack.get_input_size()):
      flattened = repack.input_value(index).get_node()
      if (flattened.get_type_name() != "Reshape" or
          flattened.get_input_size() < 1):
        continue
      source = flattened.input_value(0)
      if (source.get_node().get_type_name() == "Loop" and
          source.get_index() == 1):
        final_state_outputs.append(source)
    if len(final_state_outputs) != 1:
      raise ValueError(
          f"expected one Loop final-state output for {variable_id}, got "
          f"{len(final_state_outputs)}")
    final_state = final_state_outputs[0]
    if (final_state.get_partial_shape() !=
        assign.input_value(0).get_partial_shape()):
      raise ValueError(
          f"SSM final-state shape mismatch for {variable_id}: "
          f"{final_state.get_partial_shape()} versus "
          f"{assign.input_value(0).get_partial_shape()}")
    assign.input(0).replace_source_output(final_state)
    rows.append({
        "variable_id": variable_id,
        "loop": final_state.get_node().get_friendly_name(),
        "loop_output_index": final_state.get_index(),
        "old_repack": repack.get_friendly_name(),
        "old_slice": state_slice.get_friendly_name(),
        "old_reshape": final_reshape.get_friendly_name(),
        "shape": str(final_state.get_partial_shape()),
    })
  if len(rows) != len(LINEAR_ATTENTION_LAYERS):
    raise ValueError(
        f"expected {len(LINEAR_ATTENTION_LAYERS)} SSM state rewrites, "
        f"got {len(rows)}")
  model.validate_nodes_and_infer_types()
  return rows


def relocate_q_gate_split_consumers(
    model: Any, ov: Any,
) -> list[dict[str, Any]]:
  """Move reshape-only users away from the ten Q/gate split outputs.

  The Q reshape is shape preserving and can be bypassed.  The gate reshape is
  a pure flatten immediately before Sigmoid, so move it after that elementwise
  operation.  This leaves padding-aware elementwise users directly attached to
  each VariadicSplit output, which is the graph shape required by the dynamic
  split-length in-place Crop optimization.
  """
  operations = {
      node.get_friendly_name(): node for node in model.get_ordered_ops()}
  rows = []
  for layer in FULL_ATTENTION_LAYERS:
    prefix = (
        f"__module.model.model.language_model.layers.{layer}.self_attn/")

    def named(suffix: str) -> Any:
      name = prefix + suffix
      if name not in operations:
        raise ValueError(f"Q/gate split relocation node missing: {name}")
      return operations[name]

    split = named("prim::ListUnpack/VariadicSplit")
    q_targets = list(split.output(0).get_target_inputs())
    gate_targets = list(split.output(1).get_target_inputs())
    if len(q_targets) != 1 or len(gate_targets) != 1:
      raise ValueError(f"unexpected Q/gate split fanout at layer {layer}")
    q_reshape = q_targets[0].get_node()
    gate_reshape = gate_targets[0].get_node()
    gate_sigmoid_targets = list(gate_reshape.output(0).get_target_inputs())
    if (q_reshape.get_type_name() != "Reshape" or
        gate_reshape.get_type_name() != "Reshape" or
        len(gate_sigmoid_targets) != 1 or
        gate_sigmoid_targets[0].get_node().get_type_name() != "Sigmoid"):
      raise ValueError(f"unexpected Q/gate split topology at layer {layer}")
    gate_sigmoid = gate_sigmoid_targets[0].get_node()
    if (q_reshape.get_output_partial_shape(0) !=
        split.get_output_partial_shape(0)):
      raise ValueError(f"Q reshape is not shape preserving at layer {layer}")

    q_users = sorted(
        value.get_node().get_type_name()
        for value in q_reshape.output(0).get_target_inputs())
    gate_users = sorted(
        value.get_node().get_type_name()
        for value in gate_sigmoid.output(0).get_target_inputs())
    q_reshape.output(0).replace(split.output(0))
    relocated_sigmoid = ov.opset13.sigmoid(split.output(1))
    relocated_sigmoid.set_friendly_name(
        f"iq36_split_gate_sigmoid_layer{layer}")
    relocated_reshape = ov.opset13.reshape(
        relocated_sigmoid.output(0), gate_reshape.input_value(1), True)
    relocated_reshape.set_friendly_name(
        f"iq36_split_gate_reshape_layer{layer}")
    gate_sigmoid.output(0).replace(relocated_reshape.output(0))
    rows.append({
        "layer": layer,
        "split": split.get_friendly_name(),
        "q_bypassed_reshape": q_reshape.get_friendly_name(),
        "q_direct_user_types": q_users,
        "gate_moved_reshape": gate_reshape.get_friendly_name(),
        "gate_replaced_sigmoid": gate_sigmoid.get_friendly_name(),
        "gate_consumer_types": gate_users,
        "relocated_sigmoid": relocated_sigmoid.get_friendly_name(),
        "relocated_reshape": relocated_reshape.get_friendly_name(),
    })
  model.validate_nodes_and_infer_types()
  return rows


def fold_q_gate_split_lengths_to_constants(
    model: Any, ov: Any, np: Any,
) -> list[dict[str, Any]]:
  """Replace the ten locked Q/gate split-length subgraphs with constants.

  The product IR fixes the split axis at -1, the source width at 512, and both
  output widths at 256.  Preserve those facts as checked graph invariants and
  remove only the runtime ShapeOf/Concat path feeding VariadicSplit input 2.
  """
  operations = {
      node.get_friendly_name(): node for node in model.get_ordered_ops()}
  rows = []
  for layer in FULL_ATTENTION_LAYERS:
    name = (
        "__module.model.model.language_model.layers."
        f"{layer}.self_attn/prim::ListUnpack/VariadicSplit")
    if name not in operations:
      raise ValueError(f"Q/gate split node missing: {name}")
    split = operations[name]
    if split.get_type_name() != "VariadicSplit":
      raise ValueError(f"unexpected Q/gate split type at layer {layer}")
    axis_node = split.input_value(1).get_node()
    if (axis_node.get_type_name() != "Constant" or
        list(axis_node.get_vector()) != [-1]):
      raise ValueError(f"unexpected Q/gate split axis at layer {layer}")
    input_shape = split.get_input_partial_shape(0)
    output_shapes = [
        split.get_output_partial_shape(index)
        for index in range(split.get_output_size())]
    if (len(output_shapes) != 2 or input_shape.rank.is_dynamic or
        any(shape.rank.is_dynamic for shape in output_shapes)):
      raise ValueError(f"unexpected Q/gate split rank at layer {layer}")
    input_width = input_shape[-1]
    output_widths = [shape[-1] for shape in output_shapes]
    if (input_width.is_dynamic or any(dim.is_dynamic for dim in output_widths)):
      raise ValueError(f"dynamic Q/gate split width at layer {layer}")
    lengths = [int(dim.get_length()) for dim in output_widths]
    if int(input_width.get_length()) != 512 or lengths != [256, 256]:
      raise ValueError(f"unexpected Q/gate split widths at layer {layer}")
    old_lengths = split.input_value(2).get_node()
    if old_lengths.get_type_name() == "Constant":
      raise ValueError(f"Q/gate split lengths already constant at layer {layer}")
    lengths_constant = ov.opset13.constant(
        np.array(lengths, dtype=np.int32))
    lengths_constant.set_friendly_name(
        f"iq36_q_gate_split_lengths_layer{layer}")
    split.input(2).replace_source_output(lengths_constant.output(0))
    rows.append({
        "layer": layer,
        "split": split.get_friendly_name(),
        "axis": -1,
        "input_width": int(input_width.get_length()),
        "output_widths": lengths,
        "old_lengths_source": old_lengths.get_friendly_name(),
        "old_lengths_source_type": old_lengths.get_type_name(),
        "constant": lengths_constant.get_friendly_name(),
        "constant_type": str(lengths_constant.get_output_element_type(0)),
    })
  model.validate_nodes_and_infer_types()
  return rows


def make_candidate_model(
    core: Any, model_dir: Path, ov: Any, np: Any,
    target_layers: tuple[int, ...] = (TARGET_LAYER,),
    phase_branch_prefill: bool = False,
    stock_prefill_custom_decode: bool = False,
    stock_prefill_sliced_decode: bool = False,
    exact_phase_decode: bool = False,
    exact_phase_context_partition4: bool = False,
    exact_phase_dual_cohort: bool = False,
    static_phase: str | None = None,
    initialize_hot_states: bool = False,
    position_derived_mask: bool = False,
    paged_attention_layout: bool = False,
    paged_attention_anchor_layer: int | None = None,
    fixed_cold_capacity: int | None = None,
    prefill_history_capacity: int | None = None,
    exact_history_layers: tuple[int, ...] = (),
    exact_history_capacity: int | None = None,
    fuse_linear_conv_state: bool = False,
    direct_i8_fixed_layout: bool = False,
    direct_i8_group4_full_cold: bool = False,
    direct_i8_hybrid_k2_v4: bool = False,
    adaptive_attention_layers: tuple[int, ...] = (),
    adaptive_attention_topk: int = 512,
    adaptive_attention_high_topk_layers: tuple[int, ...] = (),
    adaptive_attention_high_topk: int = 256,
    adaptive_attention_v16_layers: tuple[int, ...] = (),
    adaptive_attention_key_exact_layers: tuple[int, ...] = (),
    adaptive_attention_key_residual1_layers: tuple[int, ...] = (),
    adaptive_attention_value_residual1_layers: tuple[int, ...] = (),
    adaptive_attention_packed_kv_layers: tuple[int, ...] = (),
    adaptive_attention_packed_kv_variant: str | None = None,
    decode_chunk256_layers: tuple[int, ...] = (),
    decode_f32_numerator_layers: tuple[int, ...] = (),
    decode_dual256_layers: tuple[int, ...] = (),
    decode_stock256_layers: tuple[int, ...] = (),
    decode_stock_score_layers: tuple[int, ...] = (),
    decode_stock_partition_layers: tuple[int, ...] = (),
    decode_stock_micro_layers: tuple[int, ...] = (),
    decode_page_sparse_layers: tuple[int, ...] = (),
    relocate_dynamic_split_consumers: bool = False,
    constant_q_gate_split_lengths: bool = False,
    fuse_attention_output_gate: bool = False,
    token_major_value_output: bool = False,
    attention_gated_dynamic_quantize: bool = False,
    fuse_qk_rope_layout: bool = False,
    fuse_fixed_fc: bool = False,
    fixed_fc_cohorts: tuple[str, ...] | None = None,
    direct_ssm_state_assign: bool = False,
    source_model: Any | None = None,
) -> tuple[Any, dict[str, Any]]:
  """Replace selected stock SDPAs and K/V states with bounded carriers."""
  target_layers = tuple(target_layers)
  exact_history_layers = tuple(exact_history_layers)
  adaptive_attention_layers = tuple(adaptive_attention_layers)
  adaptive_attention_high_topk_layers = tuple(
      adaptive_attention_high_topk_layers)
  adaptive_attention_v16_layers = tuple(adaptive_attention_v16_layers)
  adaptive_attention_key_exact_layers = tuple(
      adaptive_attention_key_exact_layers)
  adaptive_attention_key_residual1_layers = tuple(
      adaptive_attention_key_residual1_layers)
  adaptive_attention_value_residual1_layers = tuple(
      adaptive_attention_value_residual1_layers)
  adaptive_attention_packed_kv_layers = tuple(
      adaptive_attention_packed_kv_layers)
  if adaptive_attention_packed_kv_variant is not None:
    adaptive_attention_packed_kv_variant = (
        adaptive_attention_packed_kv_variant.lower())
  if adaptive_attention_topk not in (128, 252, 256, 512, 1024, 2048):
    raise ValueError(
        "adaptive_attention_topk must be 128, 252, 256, 512, 1024, or 2048")
  if adaptive_attention_high_topk not in (
      128, 252, 256, 512, 1024, 2048):
    raise ValueError(
        "adaptive_attention_high_topk must be 128, 252, 256, 512, 1024, or "
        "2048")
  if not adaptive_attention_layers and (
      adaptive_attention_topk != 512 or
      adaptive_attention_high_topk_layers):
    raise ValueError(
        "adaptive top-k variants require adaptive attention")
  decode_chunk256_layers = tuple(decode_chunk256_layers)
  decode_f32_numerator_layers = tuple(decode_f32_numerator_layers)
  decode_dual256_layers = tuple(decode_dual256_layers)
  decode_stock256_layers = tuple(decode_stock256_layers)
  decode_stock_score_layers = tuple(decode_stock_score_layers)
  decode_stock_partition_layers = tuple(decode_stock_partition_layers)
  decode_stock_micro_layers = tuple(decode_stock_micro_layers)
  decode_page_sparse_layers = tuple(decode_page_sparse_layers)
  if exact_phase_context_partition4 and (
      not exact_phase_decode or not decode_stock_micro_layers):
    raise ValueError(
        "exact_phase_context_partition4 requires exact stock-micro decode")
  if exact_phase_dual_cohort and (
      not exact_phase_decode or not decode_stock_micro_layers):
    raise ValueError(
        "exact_phase_dual_cohort requires exact stock-micro decode")
  if exact_phase_dual_cohort and (
      exact_phase_context_partition4 or decode_page_sparse_layers):
    raise ValueError(
        "dual-cohort exact decode is incompatible with partition/page routes")
  if (len(set(decode_page_sparse_layers)) !=
      len(decode_page_sparse_layers) or
      not set(decode_page_sparse_layers).issubset(
          decode_stock_micro_layers)):
    raise ValueError(
        "decode_page_sparse_layers must be a unique stock-micro subset")
  if decode_page_sparse_layers and not exact_phase_decode:
    raise ValueError("page-sparse decode requires exact-phase composition")
  if not target_layers or len(set(target_layers)) != len(target_layers):
    raise ValueError("target_layers must be non-empty and unique")
  if any(layer not in FULL_ATTENTION_LAYERS for layer in target_layers):
    raise ValueError(
        f"target_layers must be a subset of {FULL_ATTENTION_LAYERS}")
  if (len(set(exact_history_layers)) != len(exact_history_layers) or
      any(layer not in target_layers for layer in exact_history_layers)):
    raise ValueError(
        "exact_history_layers must be a unique subset of target_layers")
  if (len(set(adaptive_attention_layers)) !=
      len(adaptive_attention_layers) or
      any(layer not in target_layers for layer in adaptive_attention_layers)):
    raise ValueError(
        "adaptive_attention_layers must be a unique subset of target_layers")
  if (adaptive_attention_layers and
      set(adaptive_attention_layers) != set(target_layers)):
    raise ValueError(
        "adaptive attention must own every selected full-attention layer")
  if (len(set(adaptive_attention_high_topk_layers)) !=
      len(adaptive_attention_high_topk_layers) or
      not set(adaptive_attention_high_topk_layers).issubset(
          adaptive_attention_layers)):
    raise ValueError(
        "adaptive_attention_high_topk_layers must be a unique subset of "
        "adaptive_attention_layers")
  if (adaptive_attention_high_topk_layers and
      adaptive_attention_high_topk == adaptive_attention_topk):
    raise ValueError(
        "adaptive_attention_high_topk must differ from the base top-k")
  adaptive_topk_by_layer = {
      layer: (adaptive_attention_high_topk
              if layer in adaptive_attention_high_topk_layers else
              adaptive_attention_topk)
      for layer in adaptive_attention_layers
  }
  if (len(set(adaptive_attention_v16_layers)) !=
      len(adaptive_attention_v16_layers) or
      not set(adaptive_attention_v16_layers).issubset(
          adaptive_attention_layers)):
    raise ValueError(
        "adaptive_attention_v16_layers must be a unique subset of "
        "adaptive_attention_layers")
  if any(adaptive_topk_by_layer[layer] != 512
         for layer in adaptive_attention_v16_layers):
    raise ValueError("adaptive V16 is admitted only with top-512 correction")
  if (len(set(adaptive_attention_key_exact_layers)) !=
      len(adaptive_attention_key_exact_layers) or
      not set(adaptive_attention_key_exact_layers).issubset(
          adaptive_attention_layers)):
    raise ValueError(
        "adaptive_attention_key_exact_layers must be a unique subset of "
        "adaptive_attention_layers")
  if any(adaptive_topk_by_layer[layer] != 256
         for layer in adaptive_attention_key_exact_layers):
    raise ValueError("adaptive key-exact requires top-256 correction")
  for name, layers in (
      ("key", adaptive_attention_key_residual1_layers),
      ("value", adaptive_attention_value_residual1_layers)):
    if (len(set(layers)) != len(layers) or
        not set(layers).issubset(adaptive_attention_layers)):
      raise ValueError(
          f"adaptive_attention_{name}_residual1_layers must be a unique "
          "subset of adaptive_attention_layers")
  residual1_layers = (
      set(adaptive_attention_key_residual1_layers) |
      set(adaptive_attention_value_residual1_layers))
  if any(adaptive_topk_by_layer[layer] not in (256, 512)
         for layer in residual1_layers):
    raise ValueError("adaptive residual1 requires top-256/512 correction")
  if set(adaptive_attention_v16_layers) & residual1_layers:
    raise ValueError("adaptive V16 and residual1 layers must be disjoint")
  if (set(adaptive_attention_key_exact_layers) &
      (set(adaptive_attention_v16_layers) | residual1_layers)):
    raise ValueError(
        "adaptive key-exact, V16, and residual1 layers must be disjoint")
  if (len(set(adaptive_attention_packed_kv_layers)) !=
      len(adaptive_attention_packed_kv_layers) or
      not set(adaptive_attention_packed_kv_layers).issubset(
          adaptive_attention_layers)):
    raise ValueError(
        "adaptive_attention_packed_kv_layers must be a unique subset of "
        "adaptive_attention_layers")
  if adaptive_attention_packed_kv_variant not in (
      None, *ADAPTIVE_PACKED_KV_VARIANTS):
    raise ValueError(
        "adaptive_attention_packed_kv_variant must be one of "
        f"{ADAPTIVE_PACKED_KV_VARIANTS}")
  if bool(adaptive_attention_packed_kv_layers) != bool(
      adaptive_attention_packed_kv_variant):
    raise ValueError(
        "adaptive packed K/V layers and variant must be specified together")
  if any(adaptive_topk_by_layer[layer] not in (256, 512)
         for layer in adaptive_attention_packed_kv_layers):
    raise ValueError("adaptive packed K/V requires top-256/512 correction")
  if (adaptive_attention_packed_kv_variant != "k7v8" and
      any(adaptive_topk_by_layer[layer] == 512
          for layer in adaptive_attention_packed_kv_layers)):
    raise ValueError("only packed K7/V8 currently admits top-512 correction")
  if (set(adaptive_attention_packed_kv_layers) &
      (set(adaptive_attention_v16_layers) |
       set(adaptive_attention_key_exact_layers) | residual1_layers)):
    raise ValueError(
        "adaptive packed K/V is disjoint from V16, key-exact, and residual1")
  if (len(set(decode_chunk256_layers)) != len(decode_chunk256_layers) or
      any(layer not in target_layers for layer in decode_chunk256_layers)):
    raise ValueError(
        "decode_chunk256_layers must be a unique subset of target_layers")
  if (len(set(decode_dual256_layers)) != len(decode_dual256_layers) or
      any(layer not in target_layers for layer in decode_dual256_layers)):
    raise ValueError(
        "decode_dual256_layers must be a unique subset of target_layers")
  if (len(set(decode_f32_numerator_layers)) !=
      len(decode_f32_numerator_layers) or
      any(layer not in target_layers
          for layer in decode_f32_numerator_layers)):
    raise ValueError(
        "decode_f32_numerator_layers must be a unique subset of "
        "target_layers")
  if (len(set(decode_stock256_layers)) != len(decode_stock256_layers) or
      any(layer not in target_layers for layer in decode_stock256_layers)):
    raise ValueError(
        "decode_stock256_layers must be a unique subset of target_layers")
  if (len(set(decode_stock_score_layers)) !=
      len(decode_stock_score_layers) or
      any(layer not in target_layers for layer in decode_stock_score_layers)):
    raise ValueError(
        "decode_stock_score_layers must be a unique subset of target_layers")
  if (len(set(decode_stock_partition_layers)) !=
      len(decode_stock_partition_layers) or
      any(layer not in target_layers
          for layer in decode_stock_partition_layers)):
    raise ValueError(
        "decode_stock_partition_layers must be a unique subset of "
        "target_layers")
  if (len(set(decode_stock_micro_layers)) != len(decode_stock_micro_layers) or
      any(layer not in target_layers for layer in decode_stock_micro_layers)):
    raise ValueError(
        "decode_stock_micro_layers must be a unique subset of target_layers")
  decode_arithmetic_sets = (
      set(decode_chunk256_layers), set(decode_f32_numerator_layers),
      set(decode_dual256_layers), set(decode_stock256_layers),
      set(decode_stock_score_layers), set(decode_stock_partition_layers),
      set(decode_stock_micro_layers))
  if any(decode_arithmetic_sets[left] & decode_arithmetic_sets[right]
         for left in range(len(decode_arithmetic_sets))
         for right in range(left + 1, len(decode_arithmetic_sets))):
    raise ValueError(
        "decode arithmetic layer subsets must be pairwise disjoint")
  if adaptive_attention_layers and any(decode_arithmetic_sets):
    raise ValueError(
        "adaptive attention is exclusive with legacy decode arithmetic sets")
  if sum((phase_branch_prefill, stock_prefill_custom_decode,
          stock_prefill_sliced_decode, exact_phase_decode)) > 1:
    raise ValueError("dynamic phase composition modes are exclusive")
  if static_phase not in (None, "prefill", "decode"):
    raise ValueError("static_phase must be None, 'prefill', or 'decode'")
  if static_phase is not None and (
      phase_branch_prefill or stock_prefill_custom_decode or
      stock_prefill_sliced_decode or exact_phase_decode):
    raise ValueError(
        "static_phase is exclusive with dynamic phase composition modes")
  if ((stock_prefill_custom_decode or stock_prefill_sliced_decode) and
      paged_attention_layout):
    raise ValueError(
        "stock_prefill_custom_decode does not support paged layout")
  if fixed_cold_capacity is not None and fixed_cold_capacity < 1:
    raise ValueError("fixed_cold_capacity must be positive")
  if direct_i8_fixed_layout and fixed_cold_capacity is None:
    raise ValueError("direct_i8_fixed_layout requires fixed_cold_capacity")
  if direct_i8_group4_full_cold and not direct_i8_fixed_layout:
    raise ValueError(
        "direct_i8_group4_full_cold requires direct_i8_fixed_layout")
  if direct_i8_hybrid_k2_v4 and not direct_i8_fixed_layout:
    raise ValueError(
        "direct_i8_hybrid_k2_v4 requires direct_i8_fixed_layout")
  if direct_i8_group4_full_cold and direct_i8_hybrid_k2_v4:
    raise ValueError("direct-I8 fine-codec modes are mutually exclusive")
  if adaptive_attention_layers and not direct_i8_fixed_layout:
    raise ValueError(
        "adaptive attention requires the direct-I8 fixed layout")
  if adaptive_attention_layers and (
      direct_i8_group4_full_cold or direct_i8_hybrid_k2_v4):
    raise ValueError("adaptive attention admits exactly block32 I8 K/V")
  if decode_chunk256_layers and (
      phase_branch_prefill or stock_prefill_custom_decode or
      stock_prefill_sliced_decode or static_phase is not None or
      direct_i8_fixed_layout or paged_attention_layout or
      fuse_attention_output_gate or token_major_value_output or
      attention_gated_dynamic_quantize or fuse_qk_rope_layout):
    raise ValueError(
        "decode_chunk256_layers requires the unified F16 attention path")
  if decode_dual256_layers and (
      phase_branch_prefill or stock_prefill_custom_decode or
      stock_prefill_sliced_decode or static_phase is not None or
      direct_i8_fixed_layout or paged_attention_layout or
      fuse_attention_output_gate or token_major_value_output or
      attention_gated_dynamic_quantize or fuse_qk_rope_layout):
    raise ValueError(
        "decode_dual256_layers requires the unified F16 attention path")
  if decode_f32_numerator_layers and (
      phase_branch_prefill or stock_prefill_custom_decode or
      stock_prefill_sliced_decode or static_phase is not None or
      direct_i8_fixed_layout or paged_attention_layout or
      fuse_attention_output_gate or token_major_value_output or
      attention_gated_dynamic_quantize or fuse_qk_rope_layout):
    raise ValueError(
        "decode_f32_numerator_layers requires the unified F16 attention "
        "path")
  if decode_stock256_layers and (
      phase_branch_prefill or stock_prefill_custom_decode or
      stock_prefill_sliced_decode or static_phase is not None or
      direct_i8_fixed_layout or paged_attention_layout or
      fuse_attention_output_gate or token_major_value_output or
      attention_gated_dynamic_quantize or fuse_qk_rope_layout):
    raise ValueError(
        "decode_stock256_layers requires the unified F16 attention path")
  if decode_stock_score_layers and (
      phase_branch_prefill or stock_prefill_custom_decode or
      stock_prefill_sliced_decode or static_phase is not None or
      direct_i8_fixed_layout or paged_attention_layout or
      fuse_attention_output_gate or token_major_value_output or
      attention_gated_dynamic_quantize or fuse_qk_rope_layout):
    raise ValueError(
        "decode_stock_score_layers requires the unified F16 attention path")
  if decode_stock_partition_layers and (
      phase_branch_prefill or stock_prefill_custom_decode or
      stock_prefill_sliced_decode or static_phase is not None or
      direct_i8_fixed_layout or paged_attention_layout or
      fuse_attention_output_gate or token_major_value_output or
      attention_gated_dynamic_quantize or fuse_qk_rope_layout):
    raise ValueError(
        "decode_stock_partition_layers requires the unified F16 attention "
        "path")
  if decode_stock_micro_layers and (
      not (stock_prefill_custom_decode or exact_phase_decode) or
      phase_branch_prefill or stock_prefill_sliced_decode or
      static_phase is not None or
      direct_i8_fixed_layout or paged_attention_layout or
      fuse_attention_output_gate or token_major_value_output or
      attention_gated_dynamic_quantize or
      (fuse_qk_rope_layout and not exact_phase_decode)):
    raise ValueError(
        "decode_stock_micro_layers requires exact-phase or stock-prefill "
        "custom-decode F16 attention path")
  if (constant_q_gate_split_lengths and
      not relocate_dynamic_split_consumers):
    raise ValueError(
        "constant_q_gate_split_lengths requires consumer relocation")
  if fuse_attention_output_gate and (
      phase_branch_prefill or stock_prefill_custom_decode or
      stock_prefill_sliced_decode or static_phase is not None or
      paged_attention_layout or direct_i8_fixed_layout):
    raise ValueError(
        "fuse_attention_output_gate requires the unified F16 attention path")
  if token_major_value_output and (
      fuse_attention_output_gate or phase_branch_prefill or
      stock_prefill_custom_decode or stock_prefill_sliced_decode or
      static_phase is not None or paged_attention_layout or
      direct_i8_fixed_layout):
    raise ValueError(
        "token_major_value_output requires the unified F16 attention path")
  if attention_gated_dynamic_quantize and (
      fuse_attention_output_gate or token_major_value_output or
      phase_branch_prefill or stock_prefill_custom_decode or
      stock_prefill_sliced_decode or static_phase is not None or
      paged_attention_layout or direct_i8_fixed_layout):
    raise ValueError(
        "attention_gated_dynamic_quantize requires the unified F16 path")
  if fuse_qk_rope_layout and (
      fuse_attention_output_gate or token_major_value_output or
      attention_gated_dynamic_quantize or phase_branch_prefill or
      stock_prefill_custom_decode or stock_prefill_sliced_decode or
      static_phase is not None or paged_attention_layout or
      direct_i8_fixed_layout):
    raise ValueError(
        "fuse_qk_rope_layout requires the unmodified unified F16 path")
  if (direct_i8_fixed_layout and
      fixed_cold_capacity % KEY_TILE_TOKENS != 0):
    raise ValueError(
        "direct_i8_fixed_layout requires a block16-aligned cold capacity")
  if direct_i8_fixed_layout and (
      phase_branch_prefill or static_phase is not None or
      stock_prefill_custom_decode or stock_prefill_sliced_decode):
    raise ValueError("direct_i8_fixed_layout requires unified composition")
  if (prefill_history_capacity is not None and
      prefill_history_capacity < RING_CAPACITY):
    raise ValueError(
        f"prefill_history_capacity must be at least {RING_CAPACITY}")
  if (prefill_history_capacity is not None and
      prefill_history_capacity & (prefill_history_capacity - 1)):
    raise ValueError("prefill_history_capacity must be a power of two")
  if bool(exact_history_layers) != (exact_history_capacity is not None):
    raise ValueError(
        "exact_history_layers and exact_history_capacity are required "
        "together")
  if adaptive_attention_layers and (
      set(exact_history_layers) != set(adaptive_attention_layers) or
      exact_history_capacity is None or fixed_cold_capacity is None or
      exact_history_capacity < fixed_cold_capacity + DECODE_CHUNK_TOKENS):
    raise ValueError(
        "adaptive attention requires output512-safe exact history on every "
        "adaptive layer")
  base_history_capacity = (
      prefill_history_capacity
      if prefill_history_capacity is not None else RING_CAPACITY)
  if (exact_history_capacity is not None and
      exact_history_capacity <= base_history_capacity):
    raise ValueError(
        "exact_history_capacity must exceed prefill_history_capacity")
  if not set(decode_stock_micro_layers).issubset(exact_history_layers):
    raise ValueError(
        "decode_stock_micro_layers requires exact history for every layer")
  if decode_stock_micro_layers and fixed_cold_capacity is None:
    raise ValueError(
        "decode_stock_micro_layers requires fixed cold state capacity")
  if paged_attention_layout and not position_derived_mask:
    raise ValueError(
        "paged_attention_layout requires position_derived_mask")
  if paged_attention_anchor_layer is not None:
    if not paged_attention_layout:
      raise ValueError(
          "paged_attention_anchor_layer requires paged_attention_layout")
    if paged_attention_anchor_layer not in target_layers:
      raise ValueError(
          "paged_attention_anchor_layer must be one of target_layers")
  model = (
      source_model if source_model is not None else
      core.read_model(str(model_dir / "openvino_language_model.xml")))
  split_length_folds = (
      fold_q_gate_split_lengths_to_constants(model, ov, np)
      if constant_q_gate_split_lengths else [])
  split_consumer_relocations = (
      relocate_q_gate_split_consumers(model, ov)
      if relocate_dynamic_split_consumers else [])
  linear_conv_replacements = (
      fuse_linear_conv_state_boundaries(model, ov, np)
      if fuse_linear_conv_state else [])
  ssm_state_assign_rewrites = (
      bypass_ssm_state_assign_repacks(model)
      if direct_ssm_state_assign else [])
  before = model.get_ordered_ops()
  targets = {
      layer: next(
          node for node in before
          if node.get_type_name() == "ScaledDotProductAttention" and
          f"layers.{layer}.self_attn" in node.get_friendly_name())
      for layer in target_layers
  }
  shape_target = targets[target_layers[0]]
  assigns = {
      node.get_variable_id(): node for node in model.get_sinks()
      if node.get_type_name() == "Assign"
  }

  axis = ov.opset13.constant(np.array(0, dtype=np.int64))
  pa_layout_order = ov.opset13.constant(
      np.array([2, 1, 0, 3], dtype=np.int64))

  def normalize_attention_layout(value: Any) -> Any:
    if not paged_attention_layout:
      return value
    return ov.opset13.transpose(value, pa_layout_order).output(0)

  shape_query = normalize_attention_layout(shape_target.input_value(0))
  query_shape = ov.opset13.shape_of(shape_query, "i64")
  batch = _scalar(ov, np, query_shape, 0, axis)
  query_tokens = _scalar(ov, np, query_shape, 2, axis)
  if position_derived_mask:
    position_ids = next(
        value for value in model.inputs
        if value.get_any_name() == "position_ids")
    total = ov.opset13.add(
        ov.opset13.reduce_max(
            position_ids,
            ov.opset13.constant(np.array([0, 1, 2], dtype=np.int64)),
            False),
        ov.opset13.constant(np.array(1, dtype=np.int64)))
    generated_mask_shape = ov.opset13.concat([
        _vector(ov, np, batch),
        ov.opset13.constant(np.array([1], dtype=np.int64)),
        _vector(ov, np, query_tokens),
        _vector(ov, np, total),
    ], 0)
    generated_attention_mask = ov.opset13.broadcast(
        ov.opset13.constant(np.array(0, dtype=np.float32)),
        generated_mask_shape)
    generated_attention_mask.set_friendly_name(
        "iq36_position_derived_causal_mask")
    attention_mask = generated_attention_mask.output(0)
  else:
    mask_parameter = next(
        value for value in model.inputs
        if value.get_any_name() == "attention_mask")
    mask_shape = ov.opset13.shape_of(mask_parameter, "i64")
    total = _scalar(ov, np, mask_shape, 1, axis)
    attention_mask = shape_target.input_value(3)
  past_tokens = ov.opset13.subtract(total, query_tokens)
  logical_hot_window = (
      ADAPTIVE_HOT_WINDOW if adaptive_attention_layers else HOT_WINDOW)
  fresh_request = ov.opset13.equal(
      past_tokens, ov.opset13.constant(np.array(0, dtype=np.int64)))
  decode_chunks = ov.opset13.divide(
      ov.opset13.add(
          total,
          ov.opset13.constant(
              np.array(DECODE_CHUNK_TOKENS - 1, dtype=np.int64))),
      ov.opset13.constant(
          np.array(DECODE_CHUNK_TOKENS, dtype=np.int64)))
  decode_bucket_tokens = ov.opset13.multiply(
      decode_chunks,
      ov.opset13.constant(
          np.array(DECODE_CHUNK_TOKENS, dtype=np.int64)))
  if fixed_cold_capacity is None:
    custom_mask_tokens = ov.opset13.select(
        ov.opset13.equal(
            query_tokens,
            ov.opset13.constant(np.array(1, dtype=np.int64))),
        decode_bucket_tokens, total)
    custom_mask_shape = ov.opset13.concat([
        _vector(ov, np, batch),
        ov.opset13.constant(np.array([1], dtype=np.int64)),
        _vector(ov, np, query_tokens),
        _vector(ov, np, custom_mask_tokens),
    ], 0)
    custom_attention_mask = ov.opset13.broadcast(
        ov.opset13.constant(np.array(0, dtype=np.float32)),
        custom_mask_shape)
    decode_length_shape = ov.opset13.concat([
        ov.opset13.constant(np.array([1, 1, 1], dtype=np.int64)),
        _vector(ov, np, decode_chunks),
    ], 0)
  else:
    # Fixed product buckets carry the exact total in every element, while the
    # tensor shape reserves one additional 512-token decode chunk.  Both
    # prefill and decode therefore have one stable SimpleGPU signature per
    # phase instead of one signature per cumulative length.
    fixed_decode_chunks = (
        fixed_cold_capacity + 2 * DECODE_CHUNK_TOKENS - 1
    ) // DECODE_CHUNK_TOKENS
    custom_attention_mask = ov.opset13.constant(
        np.zeros((1, 1, 1, 1), dtype=np.float32))
    decode_length_shape = ov.opset13.constant(
        np.array([1, 1, 1, fixed_decode_chunks], dtype=np.int64))
  # The locked batch-one product has no padding, and both custom phase
  # implementations enforce causality explicitly.  The mask is a shape
  # carrier only; exact logical length comes from decode_length_carrier.
  custom_attention_mask.set_friendly_name(
      "iq36_bucketed_custom_attention_mask")
  decode_length_carrier = ov.opset13.broadcast(
      ov.opset13.convert(total, ov.Type.i32), decode_length_shape)
  decode_length_carrier.set_friendly_name(
      "iq36_bucketed_decode_length_carrier")

  constant_two = ov.opset13.constant(np.array([KV_HEADS], dtype=np.int64))
  constant_width = ov.opset13.constant(
      np.array([HEAD_DIM], dtype=np.int64))
  past_shape = ov.opset13.concat([
      _vector(ov, np, batch), constant_two, _vector(ov, np, past_tokens),
      constant_width], 0)
  present_shape = ov.opset13.concat([
      _vector(ov, np, batch), constant_two, _vector(ov, np, total),
      constant_width], 0)
  current_shape = ov.opset13.concat([
      _vector(ov, np, batch), constant_two, _vector(ov, np, query_tokens),
      constant_width], 0)
  shape_template = ov.opset13.broadcast(
      ov.opset13.constant(np.array(0, dtype=np.int8)), current_shape)
  shape_template.set_friendly_name("iq36_eviction_shape_template")

  desired_cold = ov.opset13.maximum(
      ov.opset13.subtract(
          total,
          ov.opset13.constant(
              np.array(logical_hot_window, dtype=np.int64))),
      ov.opset13.constant(np.array(0, dtype=np.int64)))
  if (TARGET_LAYER in target_layers and
      not stock_prefill_custom_decode and
      not stock_prefill_sliced_decode and static_phase != "prefill"):
    old_past_shape = next(
        node for node in before if node.get_friendly_name() == PAST_SHAPE_NODE)
    old_present_shape = next(
        node for node in before
        if node.get_friendly_name() == PRESENT_SHAPE_NODE)
    old_past_shape.output(0).replace(past_shape.output(0))
    old_present_shape.output(0).replace(present_shape.output(0))

  fine_full_cold = direct_i8_group4_full_cold or direct_i8_hybrid_k2_v4
  dimension_major_value_plane = fine_full_cold or bool(
      adaptive_attention_layers)
  operation_class = (
      state_decode_custom_class(ov)
      if (stock_prefill_custom_decode or stock_prefill_sliced_decode or
          static_phase is not None) else
      direct_i8_hybrid_k2_v4_custom_class(ov)
      if direct_i8_hybrid_k2_v4 else
      direct_i8_group4_custom_class(ov)
      if direct_i8_group4_full_cold else
      direct_i8_custom_class(ov) if direct_i8_fixed_layout else
      token_major_value_output_custom_class(ov)
      if token_major_value_output else
      gated_output_custom_class(ov) if fuse_attention_output_gate else
      custom_class(ov))
  adaptive_operation_classes = {}
  for adaptive_layer in adaptive_attention_layers:
    variant = (
        adaptive_topk_by_layer[adaptive_layer],
        16 if adaptive_layer in adaptive_attention_v16_layers else 32,
        adaptive_layer in adaptive_attention_key_residual1_layers,
        adaptive_layer in adaptive_attention_value_residual1_layers,
        adaptive_layer in adaptive_attention_key_exact_layers,
        (adaptive_attention_packed_kv_variant
         if adaptive_layer in adaptive_attention_packed_kv_layers else None),
    )
    if variant not in adaptive_operation_classes:
        adaptive_operation_classes[variant] = adaptive_attention_custom_class(
          ov, *variant)
  decode_chunk256_operation_class = (
      decode_chunk256_custom_class(ov)
      if decode_chunk256_layers else None)
  decode_f32_numerator_operation_class = (
      f32_numerator_chunk256_custom_class(ov)
      if decode_f32_numerator_layers else None)
  decode_dual256_operation_class = (
      dual256_custom_class(ov) if decode_dual256_layers else None)
  decode_stock256_operation_class = (
      stock256_partials_custom_class(ov)
      if decode_stock256_layers else None)
  decode_stock_score_operation_class = (
      stock_score_chunk256_custom_class(ov)
      if decode_stock_score_layers else None)
  decode_stock_partition_operation_class = (
      stock_partition_chunk256_custom_class(ov)
      if decode_stock_partition_layers else None)
  stock_micro_owner_operation_class = (
      stock_micro_owner_custom_class(ov)
      if decode_stock_micro_layers and not exact_phase_decode else None)
  exact_phase_operation_class = (
      (exact_phase_dual_cohort_custom_class(ov)
       if exact_phase_dual_cohort else
       exact_phase_context_partition4_custom_class(ov)
       if exact_phase_context_partition4 else exact_phase_custom_class(ov))
      if decode_stock_micro_layers and exact_phase_decode else None)
  exact_phase_page_sparse_operation_class = (
      exact_phase_page_sparse_custom_class(ov)
      if decode_page_sparse_layers else None)
  bridge_operation_class = (
      if_bridge_custom_class(ov) if stock_prefill_custom_decode else None)
  gated_dq_operation_class = (
      attention_gated_dynamic_quantize_custom_class(ov)
      if attention_gated_dynamic_quantize else None)
  qk_rope_operation_class = (
      qk_rope_layout_custom_class(ov) if fuse_qk_rope_layout else None)
  prefill_operation_classes = (
      prefill_custom_classes(ov) if phase_branch_prefill else None)
  new_sinks = []
  removed_stock_states = []
  operations = []
  attention_output_gate_fusions = []
  token_major_value_output_rewrites = []
  attention_gated_dynamic_quantize_rewrites = []
  qk_rope_layout_rewrites = []
  static_prefill_markers = []
  paged_attention_anchor_result = None
  physical_ring_capacities = {
      layer: (
          exact_history_capacity
          if layer in exact_history_layers else base_history_capacity)
      for layer in target_layers
  }
  physical_ring_capacity = max(physical_ring_capacities.values())
  physical_hot_capacity = SINK_TOKENS + physical_ring_capacity
  physical_hot_key_blocks = (
      (physical_hot_capacity + KEY_TILE_TOKENS - 1) // KEY_TILE_TOKENS)
  hot_key_storage_planes = 3 if dimension_major_value_plane else 2
  physical_hot_key_storage_blocks = (
      hot_key_storage_planes * physical_hot_key_blocks + 1)
  base_key_scale_bytes = (
      GROUP2_SCALE_BYTES if direct_i8_hybrid_k2_v4 else
      GROUP4_SCALE_BYTES if direct_i8_group4_full_cold else SCALE_BYTES)
  key_scale_bytes_by_layer = {
      layer: base_key_scale_bytes + (
          RESIDUAL1_BYTES
          if layer in adaptive_attention_key_residual1_layers else 0)
      for layer in target_layers
  }
  value_scale_bytes_by_layer = {
      layer: (
          GROUP16_SCALE_BYTES
          if layer in adaptive_attention_v16_layers else
          GROUP4_SCALE_BYTES if fine_full_cold else SCALE_BYTES) + (
              RESIDUAL1_BYTES
              if layer in adaptive_attention_value_residual1_layers else 0)
      for layer in target_layers
  }
  for layer in target_layers:
    layer_key_scale_bytes = key_scale_bytes_by_layer[layer]
    layer_value_scale_bytes = value_scale_bytes_by_layer[layer]
    layer_physical_ring_capacity = physical_ring_capacities[layer]
    layer_physical_hot_capacity = (
        SINK_TOKENS + layer_physical_ring_capacity)
    layer_physical_hot_key_blocks = (
        (layer_physical_hot_capacity + KEY_TILE_TOKENS - 1) //
        KEY_TILE_TOKENS)
    layer_physical_hot_key_storage_blocks = (
        hot_key_storage_planes * layer_physical_hot_key_blocks + 1)
    target = targets[layer]
    layer_operation_class = (
        adaptive_operation_classes[(
            adaptive_topk_by_layer[layer],
            16 if layer in adaptive_attention_v16_layers else 32,
            layer in adaptive_attention_key_residual1_layers,
            layer in adaptive_attention_value_residual1_layers,
            layer in adaptive_attention_key_exact_layers,
            (adaptive_attention_packed_kv_variant
             if layer in adaptive_attention_packed_kv_layers else None))]
        if layer in adaptive_attention_layers else
        exact_phase_page_sparse_operation_class
        if layer in decode_page_sparse_layers else
        exact_phase_operation_class
        if exact_phase_decode and layer in decode_stock_micro_layers else
        stock_micro_owner_operation_class
        if layer in decode_stock_micro_layers else
        decode_chunk256_operation_class
        if layer in decode_chunk256_layers else
        decode_f32_numerator_operation_class
        if layer in decode_f32_numerator_layers else
        decode_dual256_operation_class
        if layer in decode_dual256_layers else
        decode_stock256_operation_class
        if layer in decode_stock256_layers else
        decode_stock_score_operation_class
        if layer in decode_stock_score_layers else
        decode_stock_partition_operation_class
        if layer in decode_stock_partition_layers else operation_class)
    gate_multiply = None
    output_reshape = None
    value_transpose = None
    raw_gate_4d = None
    if fuse_attention_output_gate:
      prefix = (
          "__module.model.model.language_model.layers."
          f"{layer}.self_attn/")
      gate_multiply = next(
          node for node in before
          if node.get_friendly_name() == prefix + "aten::mul/Multiply_6")
      gate_sigmoid = gate_multiply.input_value(1).get_node()
      if gate_sigmoid.get_type_name() != "Sigmoid":
        raise ValueError(f"unexpected output gate at layer {layer}")
      gate_shape = ov.opset13.concat([
          _vector(ov, np, batch), _vector(ov, np, query_tokens),
          ov.opset13.constant(
              np.array([Q_HEADS, HEAD_DIM], dtype=np.int64)),
      ], 0)
      raw_gate_4d = ov.opset13.reshape(
          gate_sigmoid.input_value(0), gate_shape, False)
      raw_gate_4d.set_friendly_name(
          f"iq36_attention_output_raw_gate_layer{layer}")
    if token_major_value_output or attention_gated_dynamic_quantize:
      prefix = (
          "__module.model.model.language_model.layers."
          f"{layer}.self_attn/")
      output_reshape = next(
          node for node in before
          if node.get_friendly_name() == prefix + "aten::reshape/Reshape_2")
      gate_multiply = next(
          node for node in before
          if node.get_friendly_name() == prefix + "aten::mul/Multiply_6")
    stock_key, stock_value = stock_state_names(layer)
    key_assign = assigns[stock_key]
    value_assign = assigns[stock_value]
    if fuse_qk_rope_layout:
      if qk_rope_operation_class is None:
        raise ValueError("missing Q/K RoPE-layout custom operation")
      query_concat = target.input_value(0).get_node()
      key_state_concat = key_assign.input_value(0).get_node()
      key_concat = key_state_concat.input_value(1).get_node()
      if (query_concat.get_type_name() != "Concat" or
          key_concat.get_type_name() != "Concat"):
        raise ValueError(f"unexpected Q/K rotary concat at layer {layer}")
      query_rope = query_concat.input_value(0).get_node()
      query_tail = query_concat.input_value(1).get_node()
      key_rope = key_concat.input_value(0).get_node()
      key_tail = key_concat.input_value(1).get_node()
      if (query_rope.get_type_name() != "Add" or
          key_rope.get_type_name() != "Add" or
          query_tail.get_type_name() != "Slice" or
          key_tail.get_type_name() != "Slice"):
        raise ValueError(f"unexpected Q/K RoPE boundary at layer {layer}")
      query_transpose = query_tail.input_value(0).get_node()
      key_transpose = key_tail.input_value(0).get_node()
      if (query_transpose.get_type_name() != "Transpose" or
          key_transpose.get_type_name() != "Transpose"):
        raise ValueError(f"unexpected Q/K transpose at layer {layer}")
      query_cos_multiply = query_rope.input_value(0).get_node()
      query_sin_multiply = query_rope.input_value(1).get_node()
      key_cos_multiply = key_rope.input_value(0).get_node()
      key_sin_multiply = key_rope.input_value(1).get_node()
      if any(node.get_type_name() != "Multiply" for node in (
          query_cos_multiply, query_sin_multiply,
          key_cos_multiply, key_sin_multiply)):
        raise ValueError(f"unexpected Q/K RoPE multiply at layer {layer}")
      cosine = query_cos_multiply.input_value(1)
      sine = query_sin_multiply.input_value(1)
      if (cosine.get_node().get_friendly_name() !=
              key_cos_multiply.input_value(1).get_node().get_friendly_name()
          or sine.get_node().get_friendly_name() !=
              key_sin_multiply.input_value(1).get_node().get_friendly_name()):
        raise ValueError(f"Q/K RoPE tables are not shared at layer {layer}")
      qk_rope = qk_rope_operation_class([
          query_transpose.input_value(0), key_transpose.input_value(0),
          cosine, sine])
      qk_rope.set_friendly_name(f"iq36_qk_rope_layout_layer{layer}")
      query_concat.output(0).replace(qk_rope.output(0))
      key_concat.output(0).replace(qk_rope.output(1))
      qk_rope_layout_rewrites.append({
          "layer": layer,
          "operation": qk_rope.get_friendly_name(),
          "query_input": query_transpose.input_value(0).get_node(
              ).get_friendly_name(),
          "key_input": key_transpose.input_value(0).get_node(
              ).get_friendly_name(),
          "cosine": cosine.get_node().get_friendly_name(),
          "sine": sine.get_node().get_friendly_name(),
          "old_query_transpose": query_transpose.get_friendly_name(),
          "old_key_transpose": key_transpose.get_friendly_name(),
          "old_query_concat": query_concat.get_friendly_name(),
          "old_key_concat": key_concat.get_friendly_name(),
      })
    query = normalize_attention_layout(target.input_value(0))
    current_key = normalize_attention_layout(
        key_assign.input_value(0).get_node().input_value(1))
    current_value = normalize_attention_layout(
        value_assign.input_value(0).get_node().input_value(1))
    if token_major_value_output:
      value_transpose = current_value.get_node()
      if value_transpose.get_type_name() != "Transpose":
        raise ValueError(f"unexpected current-value layout at layer {layer}")
      transpose_order = value_transpose.input_value(1).get_node().get_vector()
      if list(transpose_order) != [0, 2, 1, 3]:
        raise ValueError(f"unexpected current-value order at layer {layer}")
      current_value = value_transpose.input_value(0)
    names = layer_state_names(layer)
    variables = [
        _variable(
            ov, names[0],
            [1, KV_HEADS, layer_physical_hot_key_storage_blocks,
             HOT_KEY_WORDS_PER_BLOCK],
            ov.Type.i32),
        _variable(
            ov, names[1],
            [1, KV_HEADS, layer_physical_hot_capacity, HEAD_DIM],
            ov.Type.f16),
        _variable(
            ov, names[2],
            [1, KV_HEADS,
             fixed_cold_capacity + 1
             if fixed_cold_capacity is not None else -1,
             HEAD_DIM], ov.Type.i8),
        _variable(
            ov, names[3],
            [1, KV_HEADS,
             fixed_cold_capacity + 1
             if fixed_cold_capacity is not None else -1,
             HEAD_DIM], ov.Type.i8),
        _variable(
            ov, names[4],
            [1, KV_HEADS,
             fixed_cold_capacity + 1
             if fixed_cold_capacity is not None else -1,
             layer_key_scale_bytes], ov.Type.i8),
        _variable(
            ov, names[5],
            [1, KV_HEADS,
             fixed_cold_capacity + 1
             if fixed_cold_capacity is not None else -1,
             layer_value_scale_bytes], ov.Type.i8),
    ]
    reads = []
    for index, variable in enumerate(variables):
      name = variable.get_info().variable_id
      if index < 2:
        if initialize_hot_states:
          initial_shape = (
              [1, KV_HEADS, layer_physical_hot_key_storage_blocks,
               HOT_KEY_WORDS_PER_BLOCK]
              if index == 0 else
              [1, KV_HEADS, layer_physical_hot_capacity, HEAD_DIM])
          initial_value = (
              np.array(0, dtype=np.int32) if index == 0 else
              np.array(0, dtype=np.float16))
          initial = ov.opset13.broadcast(
              ov.opset13.constant(initial_value),
              ov.opset13.constant(np.array(initial_shape, dtype=np.int64)))
          read = ov.opset13.read_value(
              initial.output(0), variable, name=f"{name}_read")
        else:
          read = ov.opset13.read_value(variable, name=f"{name}_read")
      else:
        width = (
            HEAD_DIM if index < 4 else
            layer_key_scale_bytes
            if index == 4 else layer_value_scale_bytes)
        # Row zero carries the logical length.  A fixed exact-bucket carrier
        # lets the sole custom state owner append in place without copying the
        # full cold history or recompiling a new state shape each token.
        if fixed_cold_capacity is None:
          initial = ov.opset13.constant(
              np.zeros((1, KV_HEADS, 1, width), dtype=np.int8))
        else:
          initial = ov.opset13.broadcast(
              ov.opset13.constant(np.array(0, dtype=np.int8)),
              ov.opset13.constant(np.array(
                  [1, KV_HEADS, fixed_cold_capacity + 1, width],
                  dtype=np.int64)))
        read = ov.opset13.read_value(initial, variable, name=f"{name}_read")
      reads.append(read)
    model.add_variables(variables)

    if fixed_cold_capacity is None:
      effective_cold_reads = []
      for index, read in enumerate(reads[2:]):
        physical_length = _scalar(
            ov, np, ov.opset13.shape_of(read.output(0), "i64"), 2, axis)
        effective_length = ov.opset13.select(
            fresh_request,
            ov.opset13.constant(np.array(1, dtype=np.int64)),
            physical_length)
        clipped = ov.opset13.slice(
            read.output(0),
            ov.opset13.constant(np.array([0], dtype=np.int64)),
            _vector(ov, np, effective_length),
            ov.opset13.constant(np.array([1], dtype=np.int64)),
            ov.opset13.constant(np.array([2], dtype=np.int64)))
        keep_previous = ov.opset13.convert(
            ov.opset13.logical_not(fresh_request), ov.Type.i8)
        effective = ov.opset13.multiply(clipped, keep_previous)
        effective.set_friendly_name(
            f"iq36_cold_request_reset_layer{layer}_state{index}")
        effective_cold_reads.append(effective)

      encoded_length = ov.opset13.slice(
          effective_cold_reads[0].output(0),
          ov.opset13.constant(np.array([0, 0, 0, 0], dtype=np.int64)),
          ov.opset13.constant(np.array([1, 1, 1, 3], dtype=np.int64)),
          ov.opset13.constant(np.array([1, 1, 1, 1], dtype=np.int64)),
          ov.opset13.constant(np.array([0, 1, 2, 3], dtype=np.int64)))
      encoded_length = ov.opset13.convert(
          ov.opset13.reshape(
              encoded_length,
              ov.opset13.constant(np.array([3], dtype=np.int64)), False),
          ov.Type.i64)
      logical_cold_scalar = ov.opset13.reduce_sum(
          ov.opset13.multiply(
              encoded_length,
              ov.opset13.constant(
                  np.array([1, 128, 16384], dtype=np.int64))),
          ov.opset13.constant(np.array([0], dtype=np.int64)), False)
    else:
      effective_cold_reads = reads[2:]
      # Sequential batch-one generation makes the existing logical cold
      # length a pure function of the current position.  This also resets a
      # reused InferRequest without zeroing the fixed-capacity buffers.
      logical_cold_scalar = ov.opset13.maximum(
          ov.opset13.subtract(
              past_tokens,
              ov.opset13.constant(
                  np.array(logical_hot_window, dtype=np.int64))),
          ov.opset13.constant(np.array(0, dtype=np.int64)))
    eviction_count = ov.opset13.subtract(desired_cold, logical_cold_scalar)
    eviction_count_4d = ov.opset13.convert(
        ov.opset13.reshape(
            eviction_count,
            ov.opset13.constant(np.array([1, 1, 1, 1], dtype=np.int64)),
            False),
        ov.Type.i32)

    operation_inputs = [
        query, reads[0].output(0), reads[1].output(0),
        current_key, current_value,
        *(value.output(0) for value in effective_cold_reads),
        custom_attention_mask.output(0),
        shape_template.output(0), eviction_count_4d.output(0),
        decode_length_carrier.output(0),
    ]
    if raw_gate_4d is not None:
      operation_inputs.append(raw_gate_4d.output(0))
    cold_output_offset = 2
    if not phase_branch_prefill:
      operation = layer_operation_class(operation_inputs)
      operation.set_friendly_name(f"iq36_hot_attention_layer{layer}")
      if static_phase == "prefill":
        marker_flat = ov.opset13.reshape(
            operation.output(0),
            ov.opset13.constant(np.array([-1], dtype=np.int64)), False)
        marker = ov.opset13.gather(
            marker_flat,
            ov.opset13.constant(np.array(0, dtype=np.int64)), axis)
        marker_result = ov.opset13.result(marker.output(0))
        marker_result.set_friendly_name(
            f"iq36_prefill_state_marker_layer{layer}")
        static_prefill_markers.append(marker_result)
        attention_output = None
      elif stock_prefill_sliced_decode:
        decode_condition = ov.opset13.equal(
            query_tokens,
            ov.opset13.constant(np.array(1, dtype=np.int64)))
        stock_begin = ov.opset13.select(
            decode_condition,
            ov.opset13.subtract(
                total, ov.opset13.constant(np.array(1, dtype=np.int64))),
            ov.opset13.constant(np.array(0, dtype=np.int64)))

        def slice_history(value: Any, token_axis: int) -> Any:
          sliced = ov.opset13.slice(
              value, _vector(ov, np, stock_begin), _vector(ov, np, total),
              ov.opset13.constant(np.array([1], dtype=np.int64)),
              ov.opset13.constant(
                  np.array([token_axis], dtype=np.int64)))
          return sliced.output(0)

        stock_attention = ov.opset13.scaled_dot_product_attention(
            query,
            slice_history(target.input_value(1), 2),
            slice_history(target.input_value(2), 2),
            slice_history(target.input_value(3), 3),
            causal=bool(target.get_attributes().get("causal", False)))
        stock_attention.set_friendly_name(
            f"iq36_stock_prefill_sliced_decode_layer{layer}")
        attention_merge = ov.opset13.select(
            decode_condition, operation.output(1),
            stock_attention.output(0))
        attention_merge.set_friendly_name(
            f"iq36_sliced_hybrid_attention_layer{layer}")
        attention_output = attention_merge.output(0)
      elif exact_phase_decode and layer in decode_stock_micro_layers:
        # The candidate plugin removes the inactive generated-microkernel
        # region before compiling each static phase.  One SimpleGPU state
        # owner therefore runs the fast prefill kernel for Q>1 and the exact
        # stock GQA8 microkernel for Q=1, with no host If, state copy, or
        # second custom consumer of the request-owned K/V buffers.
        attention_output = operation.output(1)
      elif not stock_prefill_custom_decode:
        attention_output = operation.output(1)
      else:
        bridge = bridge_operation_class([query])
        bridge.set_friendly_name(f"iq36_if_query_bridge_layer{layer}")
        stock_values = [bridge.output(0)] + [
            target.input_value(index)
            for index in range(1, target.get_input_size())]
        stock_parameters = [
            ov.opset13.parameter(
                value.get_partial_shape(), value.get_element_type(),
                name=f"iq36_stock_prefill_layer{layer}_input{index}")
            for index, value in enumerate(stock_values)
        ]
        stock_attention = ov.opset13.scaled_dot_product_attention(
            *(value.output(0) for value in stock_parameters),
            causal=bool(target.get_attributes().get("causal", False)))
        stock_result = ov.opset13.result(stock_attention.output(0))
        stock_body = ov.Model(
            [stock_result], stock_parameters,
            f"iq36_stock_prefill_body_layer{layer}")

        zero_parameters = [
            ov.opset13.parameter(
                value.get_partial_shape(), value.get_element_type(),
                name=f"iq36_stock_decode_zero_layer{layer}_input{index}")
            for index, value in enumerate(stock_values)
        ]
        zero_shape = ov.opset13.shape_of(zero_parameters[0], "i64")
        zero_attention = ov.opset13.broadcast(
            ov.opset13.constant(np.array(0, dtype=np.float32)),
            zero_shape)
        zero_result = ov.opset13.result(zero_attention.output(0))
        zero_body = ov.Model(
            [zero_result], zero_parameters,
            f"iq36_stock_decode_zero_body_layer{layer}")

        decode_condition = ov.opset13.equal(
            query_tokens,
            ov.opset13.constant(np.array(1, dtype=np.int64)))
        stock_prefill_condition = ov.opset13.logical_not(decode_condition)
        stock_selector = ov.opset13.if_op(
            stock_prefill_condition.output(0))
        stock_selector.set_then_body(stock_body)
        stock_selector.set_else_body(zero_body)
        for value, stock_parameter, zero_parameter in zip(
            stock_values, stock_parameters, zero_parameters):
          stock_selector.set_input(
              value, stock_parameter, zero_parameter)
        stock_selector.set_output(stock_result, zero_result)
        stock_selector.set_friendly_name(
            f"iq36_stock_prefill_select_layer{layer}")
        attention_merge = ov.opset13.select(
            decode_condition.output(0), operation.output(1),
            stock_selector.output(0))
        attention_merge.set_friendly_name(
            f"iq36_hybrid_attention_layer{layer}")
        attention_output = attention_merge.output(0)
    else:
      query_length_4d = ov.opset13.convert(
          ov.opset13.reshape(
              query_tokens,
              ov.opset13.constant(
                  np.array([1, 1, 1, 1], dtype=np.int64)), False),
          ov.Type.i32)
      branch_values = operation_inputs + [query_length_4d.output(0)]

      def make_parameters(prefix: str) -> list[Any]:
        return [
            ov.opset13.parameter(
                value.get_partial_shape(), value.get_element_type(),
                name=f"{prefix}_input{index}")
            for index, value in enumerate(branch_values)
      ]

      decode_parameters = make_parameters(f"iq36_decode_layer{layer}")
      decode_mask_carrier = ov.opset13.broadcast(
          ov.opset13.constant(np.array(0, dtype=np.int32)),
          ov.opset13.shape_of(decode_parameters[12], "i64"))
      decode_mask_carrier.set_friendly_name(
          f"iq36_decode_mask_carrier_layer{layer}")
      decode_inputs = (
          decode_parameters[:9] + [decode_mask_carrier] +
          decode_parameters[10:13])
      decode_operation = layer_operation_class(decode_inputs)
      decode_operation.set_friendly_name(
          f"iq36_hot_attention_decode_layer{layer}")
      decode_results = [
          ov.opset13.result(decode_operation.output(index))
          for index in range(1, 6)
      ]
      decode_body = ov.Model(
          decode_results, decode_parameters,
          f"iq36_decode_body_layer{layer}")

      prefill_parameters = make_parameters(f"iq36_prefill_layer{layer}")
      prefill_query_shape = ov.opset13.shape_of(
          prefill_parameters[0], "i64")
      prefill_batch = _scalar(
          ov, np, prefill_query_shape, 0, axis)
      prefill_tokens = _scalar(
          ov, np, prefill_query_shape, 2, axis)
      prefill_tiles = ov.opset13.divide(
          ov.opset13.add(
              prefill_tokens,
              ov.opset13.constant(
                  np.array(PREFILL_QUERY_TILE - 1, dtype=np.int64))),
          ov.opset13.constant(
              np.array(PREFILL_QUERY_TILE, dtype=np.int64)))
      prefill_tiled_shape = ov.opset13.concat([
          _vector(ov, np, prefill_batch),
          ov.opset13.constant(np.array([Q_HEADS], dtype=np.int64)),
          _vector(ov, np, prefill_tiles),
          ov.opset13.constant(
              np.array(
                  [PREFILL_QUERY_TILE * HEAD_DIM], dtype=np.int64)),
      ], 0)
      prefill_template = ov.opset13.broadcast(
          ov.opset13.constant(np.array(0, dtype=np.float16)),
          prefill_tiled_shape)
      prefill_inputs = prefill_parameters[:12] + [prefill_parameters[13]]

      def make_prefill_variant(
          prefix: str, operation_type: type,
      ) -> tuple[Any, list[Any], list[Any]]:
        parameters = [
            ov.opset13.parameter(
                value.get_partial_shape(), value.get_element_type(),
                name=f"{prefix}_input{index}")
            for index, value in enumerate(prefill_inputs)
        ]
        variant = operation_type(parameters)
        variant.set_friendly_name(prefix)
        results = [
            ov.opset13.result(variant.output(index))
            for index in range(5)
        ]
        return ov.Model(results, parameters, f"{prefix}_body"), parameters, results

      initial_type, continuation_type = prefill_operation_classes
      initial_body, initial_parameters, initial_results = make_prefill_variant(
          f"iq36_prefill_initial_layer{layer}", initial_type)
      continuation_body, continuation_parameters, continuation_results = (
          make_prefill_variant(
              f"iq36_prefill_continuation_layer{layer}",
              continuation_type))
      prefill_total = _scalar(
          ov, np, ov.opset13.shape_of(prefill_parameters[9], "i64"),
          3, axis)
      initial_condition = ov.opset13.equal(prefill_total, prefill_tokens)
      prefill_operation = ov.opset13.if_op(initial_condition.output(0))
      prefill_operation.set_then_body(initial_body)
      prefill_operation.set_else_body(continuation_body)
      for value, initial_parameter, continuation_parameter in zip(
          prefill_inputs, initial_parameters, continuation_parameters):
        prefill_operation.set_input(
            value.output(0), initial_parameter, continuation_parameter)
      for initial_result, continuation_result in zip(
          initial_results, continuation_results):
        prefill_operation.set_output(initial_result, continuation_result)
      prefill_operation.set_friendly_name(
          f"iq36_prefill_select_layer{layer}")
      expanded_shape = ov.opset13.concat([
          _vector(ov, np, prefill_batch),
          ov.opset13.constant(np.array([Q_HEADS], dtype=np.int64)),
          _vector(
              ov, np,
              ov.opset13.multiply(
                  prefill_tiles,
                  ov.opset13.constant(
                      np.array(PREFILL_QUERY_TILE, dtype=np.int64)))),
          ov.opset13.constant(np.array([HEAD_DIM], dtype=np.int64)),
      ], 0)
      expanded = ov.opset13.reshape(
          prefill_operation.output(0), expanded_shape, False)
      prefill_attention = ov.opset13.slice(
          expanded,
          ov.opset13.constant(np.array([0], dtype=np.int64)),
          _vector(ov, np, prefill_tokens),
          ov.opset13.constant(np.array([1], dtype=np.int64)),
          ov.opset13.constant(np.array([2], dtype=np.int64)))
      prefill_results = [
          ov.opset13.result(prefill_attention.output(0)),
          *(ov.opset13.result(prefill_operation.output(index))
            for index in range(1, 5)),
      ]
      prefill_body = ov.Model(
          prefill_results, prefill_parameters,
          f"iq36_prefill_body_layer{layer}")

      phase_condition = ov.opset13.equal(
          query_tokens,
          ov.opset13.constant(np.array(1, dtype=np.int64)))
      operation = ov.opset13.if_op(phase_condition.output(0))
      operation.set_then_body(decode_body)
      operation.set_else_body(prefill_body)
      for value, decode_parameter, prefill_parameter in zip(
          branch_values, decode_parameters, prefill_parameters):
        operation.set_input(
            value, decode_parameter, prefill_parameter)
      for decode_result, prefill_result in zip(
          decode_results, prefill_results):
        operation.set_output(decode_result, prefill_result)
      operation.set_friendly_name(f"iq36_hot_attention_layer{layer}")
      attention_output = operation.output(0)
      cold_output_offset = 1

    if paged_attention_layout and attention_output is not None:
      attention_output = ov.opset13.transpose(
          attention_output, pa_layout_order).output(0)
    if attention_output is not None:
      if fuse_attention_output_gate:
        gated_shape = ov.opset13.concat([
            _vector(ov, np, batch), _vector(ov, np, query_tokens),
            ov.opset13.constant(
                np.array([Q_HEADS * HEAD_DIM], dtype=np.int64)),
        ], 0)
        gated_flat = ov.opset13.reshape(
            attention_output, gated_shape, False)
        gated_flat.set_friendly_name(
            f"iq36_attention_output_gated_flat_layer{layer}")
        if gate_multiply is None:
          raise ValueError(f"output gate missing at layer {layer}")
        gate_multiply.output(0).replace(gated_flat.output(0))
        attention_output_gate_fusions.append({
            "layer": layer,
            "stock_attention": target.get_friendly_name(),
            "old_gate_multiply": gate_multiply.get_friendly_name(),
            "raw_gate": raw_gate_4d.get_friendly_name(),
            "operation": operation.get_friendly_name(),
            "gated_flat": gated_flat.get_friendly_name(),
        })
      elif token_major_value_output:
        flat_shape = ov.opset13.concat([
            _vector(ov, np, batch), _vector(ov, np, query_tokens),
            ov.opset13.constant(
                np.array([Q_HEADS * HEAD_DIM], dtype=np.int64)),
        ], 0)
        token_major_flat = ov.opset13.reshape(
            attention_output, flat_shape, False)
        token_major_flat.set_friendly_name(
            f"iq36_attention_token_major_flat_layer{layer}")
        if output_reshape is None or gate_multiply is None:
          raise ValueError(f"output layout chain missing at layer {layer}")
        output_reshape.output(0).replace(token_major_flat.output(0))
        token_major_value_output_rewrites.append({
            "layer": layer,
            "value_transpose": value_transpose.get_friendly_name(),
            "value_source": current_value.get_node().get_friendly_name(),
            "operation": operation.get_friendly_name(),
            "old_output_reshape": output_reshape.get_friendly_name(),
            "preserved_gate_multiply": gate_multiply.get_friendly_name(),
            "token_major_flat": token_major_flat.get_friendly_name(),
        })
      elif attention_gated_dynamic_quantize:
        if gate_multiply is None or gated_dq_operation_class is None:
          raise ValueError(f"gated-DQ output chain missing at layer {layer}")
        gate_sigmoid = gate_multiply.input_value(1)
        if gate_sigmoid.get_node().get_type_name() != "Sigmoid":
          raise ValueError(f"unexpected gated-DQ sigmoid at layer {layer}")
        gated_dq = gated_dq_operation_class([
            attention_output, gate_sigmoid])
        gated_dq.set_friendly_name(
            f"iq36_attention_gated_dynamic_quantize_layer{layer}")
        gate_multiply.output(0).replace(gated_dq.output(0))
        attention_gated_dynamic_quantize_rewrites.append({
            "layer": layer,
            "operation": operation.get_friendly_name(),
            "old_gate_multiply": gate_multiply.get_friendly_name(),
            "gate_sigmoid": gate_sigmoid.get_node().get_friendly_name(),
            "gated_dynamic_quantize": gated_dq.get_friendly_name(),
            "quantized_output_index": 1,
            "scale_output_index": 2,
            "precomputed_reduction_output_index": 3,
            "effective_group_size": 64,
        })
      else:
        target.output(0).replace(attention_output)
    if layer == paged_attention_anchor_layer:
      anchor_sum = ov.opset13.reduce_sum(
          target.output(0),
          ov.opset13.constant(np.array([0, 1, 2, 3], dtype=np.int64)),
          False)
      paged_attention_anchor_result = ov.opset13.result(anchor_sum)
      paged_attention_anchor_result.set_friendly_name(
          f"iq36_paged_attention_anchor_layer{layer}")
    operations.append(operation)

    if fixed_cold_capacity is None:
      digit0 = ov.opset13.floor_mod(
          desired_cold, ov.opset13.constant(np.array(128, dtype=np.int64)))
      digit1 = ov.opset13.floor_mod(
          ov.opset13.divide(
              desired_cold,
              ov.opset13.constant(np.array(128, dtype=np.int64))),
          ov.opset13.constant(np.array(128, dtype=np.int64)))
      digit2 = ov.opset13.floor_mod(
          ov.opset13.divide(
              desired_cold,
              ov.opset13.constant(np.array(16384, dtype=np.int64))),
          ov.opset13.constant(np.array(128, dtype=np.int64)))
      digits = ov.opset13.convert(ov.opset13.concat([
          _vector(ov, np, digit0), _vector(ov, np, digit1),
          _vector(ov, np, digit2)], 0), ov.Type.i8)
      key_sentinel_row = ov.opset13.concat([
          digits,
          ov.opset13.constant(np.zeros(HEAD_DIM - 3, dtype=np.int8))], 0)
      key_sentinel = ov.opset13.broadcast(
          ov.opset13.reshape(
              key_sentinel_row,
              ov.opset13.constant(
                  np.array([1, 1, 1, HEAD_DIM], dtype=np.int64)), False),
          ov.opset13.constant(
              np.array([1, KV_HEADS, 1, HEAD_DIM], dtype=np.int64)))

      for state_index, (variable, read, output_index) in enumerate(zip(
          variables[2:], effective_cold_reads,
          range(cold_output_offset, cold_output_offset + 4))):
        append = ov.opset13.slice(
            operation.output(output_index),
            ov.opset13.constant(np.array([0], dtype=np.int64)),
            _vector(ov, np, eviction_count),
            ov.opset13.constant(np.array([1], dtype=np.int64)),
            ov.opset13.constant(np.array([2], dtype=np.int64)))
        width = (
            HEAD_DIM if state_index < 2 else
            layer_key_scale_bytes if state_index == 2 else
            layer_value_scale_bytes)
        physical_length = _scalar(
            ov, np, ov.opset13.shape_of(read.output(0), "i64"), 2, axis)
        previous = ov.opset13.slice(
            read.output(0),
            ov.opset13.constant(np.array([1], dtype=np.int64)),
            _vector(ov, np, physical_length),
            ov.opset13.constant(np.array([1], dtype=np.int64)),
            ov.opset13.constant(np.array([2], dtype=np.int64)))
        sentinel = (key_sentinel if state_index == 0 else
                    ov.opset13.constant(
                        np.zeros((1, KV_HEADS, 1, width), dtype=np.int8)))
        present = ov.opset13.concat(
            [sentinel.output(0), previous.output(0), append.output(0)], 2)
        new_sinks.append(ov.opset13.assign(
            present, variable,
            name=f"{variable.get_info().variable_id}_assign"))
    if (not stock_prefill_custom_decode and
        not stock_prefill_sliced_decode and static_phase != "prefill" and
        layer != paged_attention_anchor_layer):
      removed_stock_states.extend((stock_key, stock_value))

  model.add_sinks(new_sinks)
  if static_prefill_markers:
    model.add_results(static_prefill_markers)
  if paged_attention_anchor_result is not None:
    model.add_results([paged_attention_anchor_result])
  for variable_id in removed_stock_states:
    model.remove_sink(assigns[variable_id])
    model.remove_variable(model.get_variable_by_id(variable_id))
  if fixed_fc_cohorts is not None and not fuse_fixed_fc:
    raise ValueError("fixed_fc_cohorts requires fuse_fixed_fc")
  fixed_fc_summary = (
      _fixed_fc_module().rewrite_fixed_fc(
          model, ov, np, cohorts=fixed_fc_cohorts)
      if fuse_fixed_fc else {})
  model.validate_nodes_and_infer_types()
  after = model.get_ordered_ops()
  adaptive_value_quant_group_by_layer = {
      str(layer): (16 if layer in adaptive_attention_v16_layers else 32)
      for layer in adaptive_attention_layers
  }
  uniform_value_scale_bytes = set(value_scale_bytes_by_layer.values())
  value_scale_bytes = (
      next(iter(uniform_value_scale_bytes))
      if len(uniform_value_scale_bytes) == 1 else None)
  uniform_key_scale_bytes = set(key_scale_bytes_by_layer.values())
  key_scale_bytes = (
      next(iter(uniform_key_scale_bytes))
      if len(uniform_key_scale_bytes) == 1 else None)
  summary = {
      "target_layer": target_layers[0] if len(target_layers) == 1 else None,
      "target_layers": list(target_layers),
      "target_names": [targets[layer].get_friendly_name()
                       for layer in target_layers],
      "stock_sdpa_count_before": sum(
          node.get_type_name() == "ScaledDotProductAttention"
          for node in before),
      "stock_sdpa_count_after": sum(
          node.get_type_name() == "ScaledDotProductAttention"
          for node in after),
      "custom_count_after": (
          len(target_layers) if phase_branch_prefill else
          sum(node.get_type_name() in (
                  "IQ36HotAttentionGQA", "IQ36GatedHotAttentionGQA",
                  "IQ36TokenMajorValueAttentionGQA",
                  "IQ36DirectI8HotAttentionGQA",
                  "IQ36DirectI8Group4HotAttentionGQA",
                  "IQ36DirectI8HybridK2V4HotAttentionGQA",
                  "IQ36AdaptiveV16Top512HotAttentionGQA",
                  "IQ36AdaptiveKResidual1Top512HotAttentionGQA",
                  "IQ36AdaptiveVResidual1Top512HotAttentionGQA",
                  "IQ36AdaptiveKVResidual1Top512HotAttentionGQA",
                  "IQ36AdaptiveKResidual1Top256HotAttentionGQA",
                  "IQ36AdaptiveVResidual1Top256HotAttentionGQA",
                  "IQ36AdaptiveKVResidual1Top256HotAttentionGQA",
                  "IQ36AdaptiveKeyExactTop256HotAttentionGQA",
                  "IQ36AdaptiveK6V7Top256HotAttentionGQA",
                  "IQ36AdaptiveK7V7Top256HotAttentionGQA",
                  "IQ36AdaptiveK7V8Top256HotAttentionGQA",
                  "IQ36AdaptiveK7V8Top512HotAttentionGQA",
                  "IQ36AdaptiveK8V7Top256HotAttentionGQA",
                  "IQ36AdaptiveTop2048HotAttentionGQA",
                  "IQ36AdaptiveTop1024HotAttentionGQA",
                  "IQ36AdaptiveTop512HotAttentionGQA",
                  "IQ36AdaptiveTop256HotAttentionGQA",
                  "IQ36AdaptiveTop252HotAttentionGQA",
                  "IQ36AdaptiveTop128HotAttentionGQA",
                  "IQ36DecodeChunk256HotAttentionGQA",
                  "IQ36StockMicroOwnerHotAttentionGQA",
                  "IQ36ExactPhaseHotAttentionGQA",
                  "IQ36ExactPhaseDualCohortHotAttentionGQA",
                  "IQ36ExactPhasePageSparseHotAttentionGQA",
                  "IQ36ExactPhaseContextPartition4HotAttentionGQA",
                  "IQ36F32NumeratorChunk256HotAttentionGQA",
                  "IQ36Dual256HotAttentionGQA",
                  "IQ36Stock256PartialsHotAttentionGQA",
                  "IQ36StockScoreChunk256HotAttentionGQA",
                  "IQ36StockPartitionChunk256HotAttentionGQA",
                  "IQ36StateDecodeAttentionGQA")
              for node in after)),
      "phase_branch_prefill": phase_branch_prefill,
      "stock_prefill_custom_decode": stock_prefill_custom_decode,
      "stock_prefill_sliced_decode": stock_prefill_sliced_decode,
      "exact_phase_decode": exact_phase_decode,
      "exact_phase_context_partition4": exact_phase_context_partition4,
      "exact_phase_dual_cohort": exact_phase_dual_cohort,
      "decode_page_sparse_layers": list(decode_page_sparse_layers),
      "static_phase": static_phase,
      "initialize_hot_states": initialize_hot_states,
      "position_derived_mask": position_derived_mask,
      "paged_attention_layout": paged_attention_layout,
      "paged_attention_anchor_layer": paged_attention_anchor_layer,
      "fixed_cold_capacity": fixed_cold_capacity,
      "direct_i8_fixed_layout": direct_i8_fixed_layout,
      "direct_i8_group4_full_cold": direct_i8_group4_full_cold,
      "direct_i8_hybrid_k2_v4": direct_i8_hybrid_k2_v4,
      "adaptive_attention_layers": list(adaptive_attention_layers),
      "adaptive_topk_by_layer": {
          str(layer): adaptive_topk_by_layer[layer]
          for layer in adaptive_attention_layers
      },
      "adaptive_attention_high_topk_layers": list(
          adaptive_attention_high_topk_layers),
      "adaptive_attention_high_topk": adaptive_attention_high_topk,
      "adaptive_attention_v16_layers": list(
          adaptive_attention_v16_layers),
      "adaptive_attention_key_exact_layers": list(
          adaptive_attention_key_exact_layers),
      "adaptive_attention_key_residual1_layers": list(
          adaptive_attention_key_residual1_layers),
      "adaptive_attention_value_residual1_layers": list(
          adaptive_attention_value_residual1_layers),
      "adaptive_attention_packed_kv_layers": list(
          adaptive_attention_packed_kv_layers),
      "adaptive_attention_packed_kv_variant": (
          adaptive_attention_packed_kv_variant),
      "adaptive_value_quant_group_by_layer": (
          adaptive_value_quant_group_by_layer),
      "adaptive_attention_value_quant_group": (
          16 if (adaptive_attention_layers and
                 len(adaptive_attention_v16_layers) ==
                     len(adaptive_attention_layers)) else
          32 if not adaptive_attention_v16_layers else None),
      "decode_chunk256_layers": list(decode_chunk256_layers),
      "decode_f32_numerator_layers": list(decode_f32_numerator_layers),
      "decode_dual256_layers": list(decode_dual256_layers),
      "decode_stock256_layers": list(decode_stock256_layers),
      "decode_stock_score_layers": list(decode_stock_score_layers),
      "decode_stock_partition_layers": list(
          decode_stock_partition_layers),
      "decode_stock_micro_layers": list(decode_stock_micro_layers),
      "direct_i8_quant_group": (
          2 if direct_i8_hybrid_k2_v4 else
          4 if direct_i8_group4_full_cold else 32),
      "direct_i8_key_quant_group": (
          2 if direct_i8_hybrid_k2_v4 else
          4 if direct_i8_group4_full_cold else 32),
      "direct_i8_value_quant_group": (
          16 if (adaptive_attention_layers and
                 len(adaptive_attention_v16_layers) ==
                     len(adaptive_attention_layers)) else
          32 if adaptive_attention_layers else
          4 if fine_full_cold else 32),
      "direct_i8_value_quant_group_by_layer": (
          adaptive_value_quant_group_by_layer
          if adaptive_attention_layers else {}),
      "prefill_history_capacity": prefill_history_capacity,
      "exact_history_layers": list(exact_history_layers),
      "exact_history_capacity": exact_history_capacity,
      "fuse_linear_conv_state": fuse_linear_conv_state,
      "direct_ssm_state_assign": direct_ssm_state_assign,
      "ssm_state_assign_rewrite_count": len(ssm_state_assign_rewrites),
      "ssm_state_assign_rewrites": ssm_state_assign_rewrites,
      "relocate_dynamic_split_consumers": (
          relocate_dynamic_split_consumers),
      "constant_q_gate_split_lengths": constant_q_gate_split_lengths,
      "split_length_fold_count": len(split_length_folds),
      "split_length_folds": split_length_folds,
      "split_consumer_relocation_count": len(split_consumer_relocations),
      "split_consumer_relocations": split_consumer_relocations,
      "fuse_attention_output_gate": fuse_attention_output_gate,
      "attention_output_gate_fusion_count": len(
          attention_output_gate_fusions),
      "attention_output_gate_fusions": attention_output_gate_fusions,
      "token_major_value_output": token_major_value_output,
      "token_major_value_output_rewrite_count": len(
          token_major_value_output_rewrites),
      "token_major_value_output_rewrites": token_major_value_output_rewrites,
      "attention_gated_dynamic_quantize": (
          attention_gated_dynamic_quantize),
      "attention_gated_dynamic_quantize_rewrite_count": len(
          attention_gated_dynamic_quantize_rewrites),
      "attention_gated_dynamic_quantize_rewrites": (
          attention_gated_dynamic_quantize_rewrites),
      "fuse_qk_rope_layout": fuse_qk_rope_layout,
      "qk_rope_layout_rewrite_count": len(qk_rope_layout_rewrites),
      "qk_rope_layout_rewrites": qk_rope_layout_rewrites,
      "fuse_fixed_fc": fuse_fixed_fc,
      "fixed_fc_summary": fixed_fc_summary,
      "linear_conv_replacement_count": len(linear_conv_replacements),
      "linear_conv_custom_count_after": sum(
          node.get_type_name() == "IQ36LinearConvSwish" for node in after),
      "linear_conv_replacements": linear_conv_replacements,
      "state_count_after": len(model.get_variables()),
      "sink_count_after": len(model.get_sinks()),
      "logical_hot_window": logical_hot_window,
      "exact_sink_tokens": SINK_TOKENS,
      "physical_ring_capacity": physical_ring_capacity,
      "physical_hot_capacity": physical_hot_capacity,
      "physical_ring_capacity_by_layer": {
          str(layer): physical_ring_capacities[layer]
          for layer in target_layers
      },
      "physical_hot_capacity_by_layer": {
          str(layer): SINK_TOKENS + physical_ring_capacities[layer]
          for layer in target_layers
      },
      "hot_storage": (
          "single-Variable dual K plane plus dimension-major F16 decode V "
          "plane; direct token-major F16 prefill V; logical F16 boundary"
          if dimension_major_value_plane else
          "single-Variable dual K plane: F16x2-packed I32 block16 for "
          "decode plus contiguous F16 for prefill; direct F16 V; logical "
          "F16 boundary"),
      "hot_key_shape": [
          1, KV_HEADS, physical_hot_key_storage_blocks,
          HOT_KEY_WORDS_PER_BLOCK],
      "hot_key_shape_by_layer": {
          str(layer): [
              1, KV_HEADS,
              (hot_key_storage_planes *
               ((SINK_TOKENS + physical_ring_capacities[layer] +
                 KEY_TILE_TOKENS - 1) // KEY_TILE_TOKENS) + 1),
              HOT_KEY_WORDS_PER_BLOCK]
          for layer in target_layers
      },
      "hot_key_packed_blocks": physical_hot_key_blocks,
      "hot_key_storage_planes": hot_key_storage_planes,
      "hot_value_shape": [1, KV_HEADS, physical_hot_capacity, HEAD_DIM],
      "hot_value_shape_by_layer": {
          str(layer): [
              1, KV_HEADS,
              SINK_TOKENS + physical_ring_capacities[layer], HEAD_DIM]
          for layer in target_layers
      },
      "scale_bytes": key_scale_bytes,
      "key_scale_bytes": key_scale_bytes,
      "key_scale_bytes_by_layer": {
          str(layer): key_scale_bytes_by_layer[layer]
          for layer in target_layers
      },
      "value_scale_bytes": value_scale_bytes,
      "value_scale_bytes_by_layer": {
          str(layer): value_scale_bytes_by_layer[layer]
          for layer in target_layers
      },
      "workspace_width": WORKSPACE_WIDTH,
      "adaptive_workspace_f32_elements": (
          adaptive_workspace_f32_elements(
              (fixed_cold_capacity + 2 * DECODE_CHUNK_TOKENS - 1) //
              DECODE_CHUNK_TOKENS)
          if adaptive_attention_layers else None),
      "prefill_query_tile": PREFILL_QUERY_TILE,
      "decode_chunk_tokens": DECODE_CHUNK_TOKENS,
      "cold_storage": (
          "fixed-capacity packed block32 I8 plus selected per-layer shared-"
          "scale 1-bit K/V residual planes"
          if residual1_layers else
          "fixed-capacity in-place block16-token/dim4 packed I8 K with "
          "group2 scales plus dimension-major I8 V with group4 scales"
          if direct_i8_hybrid_k2_v4 else
          "fixed-capacity in-place block16-token/group4-dimension packed I8 "
          "K plus dimension-major I8 V and group-major exact F16 scales"
          if direct_i8_group4_full_cold else
          "fixed-capacity in-place block16-token/block32-dimension packed "
          "I8 K plus dimension-major group16 I8 V and exact F16 scales"
          if (adaptive_attention_layers and
              len(adaptive_attention_v16_layers) ==
                  len(adaptive_attention_layers)) else
          "fixed-capacity in-place block16-token/block32-dimension packed "
          "I8 K plus per-layer group16/group32 dimension-major I8 V and "
          "exact F16 scales"
          if adaptive_attention_v16_layers else
          "fixed-capacity in-place block16-token/block32-dimension packed "
          "I8 K plus dimension-major I8 V and group-major exact F16 scales"
          if direct_i8_fixed_layout else
          "fixed-capacity in-place signed block32 I8 plus exact F16 scale "
          "bytes" if fixed_cold_capacity is not None else
          "append-only signed block32 I8 plus exact F16 scale bytes"),
      "length_carrier": (
          "stock prefill state plus custom position/query carriers"
          if static_phase == "prefill" else
          "position/query total plus ceil(total/512) decode bucket; stock "
          "past/present ShapeOf removed"),
      "custom_attention_mask": (
          "zero causal carrier; scalar shape for fixed product buckets and "
          "512-token decode shape buckets otherwise"),
      "removed_stock_states": removed_stock_states,
      "custom_states": list(custom_state_names(target_layers)),
      "query_shape": str(
          targets[target_layers[0]].input_value(0).get_partial_shape()),
      "current_key_shape": str(operation_inputs[3].get_partial_shape()),
      "current_value_shape": str(operation_inputs[4].get_partial_shape()),
      "mask_shape": str(
          targets[target_layers[0]].input_value(3).get_partial_shape()),
      "output_shapes": [
          str(operations[0].output(index).get_partial_shape())
          for index in range(operations[0].get_output_size())],
  }
  return model, summary


def bind_request_owned_hot_states(
    request: Any, layers: tuple[int, ...] = (TARGET_LAYER,),
) -> list[dict[str, Any]]:
  selected = set(hot_state_names(tuple(layers)))
  rows = []
  for state in request.query_state():
    if state.name not in selected:
      continue
    tensor = state.state
    state.state = tensor
    rows.append({
        "name": state.name,
        "shape": list(state.state.shape),
        "element_type": str(state.state.element_type),
        "bytes": int(state.state.byte_size),
    })
  return rows


def logical_cold_rows(value: Any) -> Any:
  """Strip the graph-owned physical sentinel from a cold state array."""
  return value[:, :, 1:, :]


def f16_scales_from_i8_bytes(value: Any, np: Any) -> Any:
  contiguous = np.ascontiguousarray(logical_cold_rows(value), dtype=np.int8)
  return contiguous.view(np.float16).reshape(
      contiguous.shape[0], contiguous.shape[1], contiguous.shape[2],
      contiguous.shape[3] // 2)


def hot_float_bits(value: Any, np: Any) -> Any:
  return np.ascontiguousarray(value, dtype=np.int32).view(np.float32)


def unpack_hot_key(value: Any, np: Any) -> Any:
  """Decode block16 F16x2-packed I32 key state into logical F32 rows."""
  packed = np.ascontiguousarray(value, dtype=np.int32)
  expected_blocks = (HOT_KEY_STORAGE_BLOCKS, 3 * HOT_KEY_BLOCKS + 1)
  if (packed.shape[:2] != (1, KV_HEADS) or
      packed.shape[2] not in expected_blocks or
      packed.shape[3] != HOT_KEY_WORDS_PER_BLOCK):
    raise ValueError(
        f"hot key shape {packed.shape} has no admitted storage layout")
  packed_plane = packed[:, :, :HOT_KEY_BLOCKS, :]
  pairs = packed_plane.view(np.float16).reshape(
      1, KV_HEADS, HOT_KEY_BLOCKS, HEAD_DIM // 2,
      KEY_TILE_TOKENS, 2)
  rows = pairs.transpose(0, 1, 2, 4, 3, 5).reshape(
      1, KV_HEADS, HOT_KEY_BLOCKS * KEY_TILE_TOKENS, HEAD_DIM)
  return rows[:, :, :HOT_CAPACITY, :].astype(np.float32)


def unpack_dimension_major_hot_value(value: Any, np: Any) -> Any:
  """Decode the group-4 carrier's dimension-major F16 hot-V plane."""
  packed = np.ascontiguousarray(value, dtype=np.int32)
  expected = (1, KV_HEADS, 3 * HOT_KEY_BLOCKS + 1,
              HOT_KEY_WORDS_PER_BLOCK)
  if packed.shape != expected:
    raise ValueError(f"group-4 hot key shape {packed.shape} != {expected}")
  plane = packed[:, :, 2 * HOT_KEY_BLOCKS:3 * HOT_KEY_BLOCKS, :]
  flat = plane.view(np.float16).reshape(1, KV_HEADS, -1)
  values = flat[:, :, :HEAD_DIM * HOT_CAPACITY].reshape(
      1, KV_HEADS, HEAD_DIM, HOT_CAPACITY)
  return values.transpose(0, 1, 3, 2).astype(np.float32)


def hot_value_rows(value: Any, np: Any) -> Any:
  """Expose direct-F16 value state at the same logical F32 boundary."""
  rows = np.ascontiguousarray(value, dtype=np.float16)
  expected = (1, KV_HEADS, HOT_CAPACITY, HEAD_DIM)
  if rows.shape != expected:
    raise ValueError(f"hot value shape {rows.shape} != {expected}")
  return rows.astype(np.float32)


def hot_state_rows(value: Any, kind: str, np: Any) -> Any:
  if kind == "key":
    return unpack_hot_key(value, np)
  if kind == "value":
    return hot_value_rows(value, np)
  raise ValueError(f"unknown hot state kind: {kind}")
