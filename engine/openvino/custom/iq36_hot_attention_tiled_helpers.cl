#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable

// Shared fixed-shape helpers for the single-owner tiled attention carrier.
// Hot K is F16x2-packed in I32 block16 planes for XMX; hot V is direct F16.
// One custom node owns both state planes and every prefill/decode transition.

#define IQ36_HEAD_DIM 256U
#define IQ36_Q_HEADS 16U
#define IQ36_KV_HEADS 2U
#define IQ36_GQA_GROUP 8U
#if (defined(IQ36_ADAPTIVE_PACKED_K6V7) + \
     defined(IQ36_ADAPTIVE_PACKED_K7V7) + \
     defined(IQ36_ADAPTIVE_PACKED_K7V8) + \
     defined(IQ36_ADAPTIVE_PACKED_K8V7)) > 1
#error "select exactly one adaptive packed K/V format"
#elif defined(IQ36_ADAPTIVE_PACKED_K6V7)
#define IQ36_ADAPTIVE_PACKED_KV 1
#define IQ36_KEY_QUANT_BITS 6U
#define IQ36_VALUE_QUANT_BITS 7U
#elif defined(IQ36_ADAPTIVE_PACKED_K7V7)
#define IQ36_ADAPTIVE_PACKED_KV 1
#define IQ36_KEY_QUANT_BITS 7U
#define IQ36_VALUE_QUANT_BITS 7U
#elif defined(IQ36_ADAPTIVE_PACKED_K7V8)
#define IQ36_ADAPTIVE_PACKED_KV 1
#define IQ36_ADAPTIVE_DIMENSION_MAJOR_V8 1
#define IQ36_KEY_QUANT_BITS 7U
#define IQ36_VALUE_QUANT_BITS 8U
#elif defined(IQ36_ADAPTIVE_PACKED_K8V7)
#define IQ36_ADAPTIVE_PACKED_KV 1
#define IQ36_KEY_QUANT_BITS 8U
#define IQ36_VALUE_QUANT_BITS 7U
#else
#define IQ36_KEY_QUANT_BITS 8U
#define IQ36_VALUE_QUANT_BITS 8U
#endif
#if defined(IQ36_ADAPTIVE_PACKED_KV)
#if !defined(IQ36_DIRECT_I8_FIXED_LAYOUT)
#error "packed K/V requires the direct fixed cold-state layout"
#endif
#endif
#define IQ36_KEY_QUANT_MAX ((1U << (IQ36_KEY_QUANT_BITS - 1U)) - 1U)
#define IQ36_VALUE_QUANT_MAX ((1U << (IQ36_VALUE_QUANT_BITS - 1U)) - 1U)
#if defined(IQ36_DIRECT_I8_HYBRID_K2_V4)
#if !defined(IQ36_DIRECT_I8_FIXED_LAYOUT)
#error "K2/V4 full-cold requires the direct-I8 fixed layout"
#endif
#define IQ36_KEY_QUANT_GROUP 2U
#define IQ36_VALUE_QUANT_GROUP 4U
#define IQ36_KEY_SCALE_GROUPS 128U
#define IQ36_VALUE_SCALE_GROUPS 64U
#elif defined(IQ36_DIRECT_I8_GROUP4_FULL_COLD)
#if !defined(IQ36_DIRECT_I8_FIXED_LAYOUT)
#error "group-4 full-cold requires the direct-I8 fixed layout"
#endif
#define IQ36_KEY_QUANT_GROUP 4U
#define IQ36_VALUE_QUANT_GROUP 4U
#define IQ36_KEY_SCALE_GROUPS 64U
#define IQ36_VALUE_SCALE_GROUPS 64U
#elif defined(IQ36_DIRECT_I8_VALUE16)
#if !defined(IQ36_DIRECT_I8_FIXED_LAYOUT)
#error "group-16 V full-cold requires the direct-I8 fixed layout"
#endif
#define IQ36_KEY_QUANT_GROUP 32U
#define IQ36_VALUE_QUANT_GROUP 16U
#define IQ36_KEY_SCALE_GROUPS 8U
#define IQ36_VALUE_SCALE_GROUPS 16U
#else
#define IQ36_KEY_QUANT_GROUP 32U
#define IQ36_VALUE_QUANT_GROUP 32U
#define IQ36_KEY_SCALE_GROUPS 8U
#define IQ36_VALUE_SCALE_GROUPS 8U
#endif
#define IQ36_KEY_PACK_WORDS \
    (IQ36_HEAD_DIM * IQ36_KEY_QUANT_BITS / 32U)
#define IQ36_VALUE_PACK_WORDS \
    (IQ36_HEAD_DIM * IQ36_VALUE_QUANT_BITS / 32U)
#define IQ36_KEY_SCALE_BYTES (IQ36_KEY_SCALE_GROUPS * 2U)
#define IQ36_VALUE_SCALE_BYTES (IQ36_VALUE_SCALE_GROUPS * 2U)
#define IQ36_RESIDUAL1_BYTES (IQ36_HEAD_DIM / 8U)
#if defined(IQ36_KEY_RESIDUAL1) || defined(IQ36_VALUE_RESIDUAL1)
#if !defined(IQ36_DIRECT_I8_FIXED_LAYOUT)
#error "residual1 requires the direct-I8 fixed layout"
#endif
#if IQ36_KEY_QUANT_GROUP != 32U || IQ36_VALUE_QUANT_GROUP != 32U
#error "residual1 is admitted only for grouped-32 K/V"
#endif
#endif
#ifndef IQ36_HOT_WINDOW
#define IQ36_HOT_WINDOW 8192U
#endif
#define IQ36_SINK_TOKENS 1U
#define IQ36_MAX_PREFILL_CHUNK_TOKENS 8192U
#define IQ36_KEY_TILE_TOKENS 16U
#define IQ36_KEY_WORDS_PER_BLOCK 2048U
#define IQ36_TOKEN_TILE 16U
#define IQ36_STAGE_TOKENS 128U
#define IQ36_BLOCKS_PER_STAGE (IQ36_STAGE_TOKENS / IQ36_TOKEN_TILE)
#define IQ36_BLOCKS_PER_CHUNK (IQ36_CHUNK_TOKENS / IQ36_TOKEN_TILE)
#ifndef IQ36_CHUNK_TOKENS
#define IQ36_CHUNK_TOKENS 512U
#endif
#if defined(IQ36_DUAL256_REDUCTION) || \
    defined(IQ36_STOCK256_PARTIALS)
#if IQ36_CHUNK_TOKENS != 512U
#error "Split-256 reduction requires the 512-token decode carrier"
#endif
#define IQ36_SPLIT256_REDUCTION 1
#define IQ36_DECODE_REDUCTION_PARTS 2U
#define IQ36_DECODE_BLOCKS_PER_PART 16U
#else
#define IQ36_DECODE_REDUCTION_PARTS 1U
#define IQ36_DECODE_BLOCKS_PER_PART IQ36_BLOCKS_PER_CHUNK
#endif
#define IQ36_PARTIAL_OUTPUT_OFFSET 2U
#define IQ36_PARTIAL_HEAD_WIDTH \
    (IQ36_PARTIAL_OUTPUT_OFFSET + IQ36_HEAD_DIM)
#define IQ36_PARTIAL_KV_WIDTH \
    (IQ36_GQA_GROUP * IQ36_PARTIAL_HEAD_WIDTH)
#ifndef IQ36_PREFILL_CHUNK_TOKENS
#define IQ36_PREFILL_CHUNK_TOKENS 128U
#endif
#define IQ36_PREFILL_BLOCKS_PER_CHUNK \
    (IQ36_PREFILL_CHUNK_TOKENS / IQ36_TOKEN_TILE)
#ifndef IQ36_PREFILL_QUERY_TILE
#define IQ36_PREFILL_QUERY_TILE 32U
#endif
#define IQ36_PREFILL_QUERY_GROUPS \
    (IQ36_PREFILL_QUERY_TILE / IQ36_GQA_GROUP)

// A direct SimpleGPU node is specialized once the dynamic input shapes are
// concrete.  Pull the relevant dimensions out of the generated initializer
// macros so the unified ABI can compile exactly one phase implementation.
#if defined(IQ36_UNIFIED_SHAPE_SPECIALIZATION)
#define IQ36_DIM2_IMPL(_0, _1, _2, ...) _2
#define IQ36_DIM2_EXPAND(...) IQ36_DIM2_IMPL(__VA_ARGS__)
#define IQ36_DIM3_IMPL(_0, _1, _2, _3, ...) _3
#define IQ36_DIM3_EXPAND(...) IQ36_DIM3_IMPL(__VA_ARGS__)
#define IQ36_STATIC_QUERY_TOKENS IQ36_DIM2_EXPAND(INPUT0_DIMS_INIT)
#define IQ36_STATIC_MASK_TOKENS IQ36_DIM3_EXPAND(INPUT9_DIMS_INIT)
#if IQ36_STATIC_QUERY_TOKENS == 1
#define IQ36_BUILD_DECODE_ONLY 1
#else
#define IQ36_BUILD_PREFILL_ONLY 1
#if !defined(IQ36_DIRECT_I8_FIXED_LAYOUT)
#define IQ36_PREFILL_FULL_HISTORY 1
#endif
#if !defined(IQ36_PREFILL_RUNTIME_LENGTH)
#if IQ36_STATIC_QUERY_TOKENS == IQ36_STATIC_MASK_TOKENS
#define IQ36_PREFILL_INITIAL 1
#else
#define IQ36_PREFILL_CONTINUATION 1
#endif
#endif
#endif
#if defined(IQ36_DIRECT_I8_FIXED_LAYOUT) && \
    defined(IQ36_PREFILL_USE_MICROKERNEL) && \
    (!defined(IQ36_PREFILL_FULL_HISTORY) || !IQ36_PREFILL_FULL_HISTORY)
#error "Direct-I8 fixed layout requires full history for microkernel prefill"
#endif
#endif
#define IQ36_ATTENTION_SCALE 0.0625f
#define IQ36_LOG2_E 1.442695f
#define IQ36_EXP2_SCALE (IQ36_ATTENTION_SCALE * IQ36_LOG2_E)
#if defined(IQ36_STOCK_SCORE_ORDER)
// The stock single-token kernel rounds Q * scale to F16 before its scalar
// dot.  Scores from that diagnostic path are already in scaled units.
#define IQ36_DECODE_EXP2_SCALE IQ36_LOG2_E
#else
#define IQ36_DECODE_EXP2_SCALE IQ36_EXP2_SCALE
#endif

#if defined(IQ36_FUSED_GATE_OUTPUT) || \
    defined(IQ36_TOKEN_MAJOR_VALUE_OUTPUT)
#define IQ36_TOKEN_MAJOR_OUTPUT 1
#endif

#if defined(IQ36_TOKEN_MAJOR_VALUE_OUTPUT)
#define IQ36_CURRENT_VALUE_HEAD_PITCH INPUT4_PITCHES[2]
#define IQ36_CURRENT_VALUE_TOKEN_PITCH INPUT4_PITCHES[1]
#define IQ36_CURRENT_VALUE_TOKENS INPUT4_DIMS[1]
#else
#define IQ36_CURRENT_VALUE_HEAD_PITCH INPUT4_PITCHES[1]
#define IQ36_CURRENT_VALUE_TOKEN_PITCH INPUT4_PITCHES[2]
#define IQ36_CURRENT_VALUE_TOKENS INPUT4_DIMS[2]
#endif

inline ulong iq36_current_value_index(
    const uint batch, const uint kv_head, const uint token, const uint dim) {
  return INPUT4_OFFSET + (ulong)batch * INPUT4_PITCHES[0] +
      (ulong)kv_head * IQ36_CURRENT_VALUE_HEAD_PITCH +
      (ulong)token * IQ36_CURRENT_VALUE_TOKEN_PITCH +
      (ulong)dim * INPUT4_PITCHES[3];
}

#if defined(IQ36_FUSED_GATE_OUTPUT)
inline OUTPUT1_TYPE iq36_gated_attention_value(
    const float attention_value,
    const __global INPUT13_TYPE* raw_gate,
    const ulong gate_index) {
  // The existing graph crosses the F16 attention/gate execution boundary
  // before generic_eltwise_ref__f16 evaluates Sigmoid and Multiply.  Keep
  // both explicit round points when subsuming that epilogue into this store.
  const half rounded_attention = convert_half_rte(attention_value);
  const half rounded_gate = convert_half_rte(raw_gate[gate_index]);
  const float gate_value = native_recip(
      1.0f + native_exp(-convert_float(rounded_gate)));
  const half gated = convert_half_rte(
      convert_float(rounded_attention) * gate_value);
  return (OUTPUT1_TYPE)gated;
}
#endif

ushort8 __builtin_IB_subgroup_block_read_flat_u16_m8k16v1(
    long, int, int, int, int2);
uint8 __builtin_IB_subgroup_block_read_cacheopts_transpose_u32_m32k4(
    long, int, int, int, int2, int);
float __builtin_IB_atomic_max_local_f32(__local float*, float);

inline half8 iq36_block2d_load_f16_16x8(
    const __global half* pointer,
    const int width_bytes,
    const int height,
    const int pitch_bytes,
    int x,
    const int y) {
  ulong address = as_long(pointer);
  const ulong prefix = address & 0x3fUL;
  address &= ~0x3fUL;
  x += (int)(prefix / sizeof(half));
  const int2 coordinate = (int2)(x, y);
  return as_half8(__builtin_IB_subgroup_block_read_flat_u16_m8k16v1(
      (long)address, width_bytes + (int)prefix - 1, height - 1,
      pitch_bytes - 1, coordinate));
}

inline uint iq36_hot_slot(const uint token) {
  const uint ring_capacity = (uint)INPUT2_DIMS[2] - IQ36_SINK_TOKENS;
  if (token < IQ36_SINK_TOKENS)
    return token;
  const uint ring_token = token - IQ36_SINK_TOKENS;
  // Exact-history lanes size the physical state to cover prompt plus decode,
  // so their hot path never wraps and need not round the capacity up to the
  // next power of two.  Approximate ring lanes retain a power-of-two capacity;
  // the compiler therefore folds the cold modulo below back to a bit mask.
  return ring_token < ring_capacity
      ? token
      : IQ36_SINK_TOKENS + ring_token % ring_capacity;
}

inline uint iq36_hot_key_packed_blocks() {
  return ((uint)INPUT2_DIMS[2] + IQ36_KEY_TILE_TOKENS - 1U) /
      IQ36_KEY_TILE_TOKENS;
}

inline ulong iq36_hot_key_dense_i32_base(
    const uint batch, const uint kv_head) {
  return INPUT1_OFFSET + (ulong)batch * INPUT1_PITCHES[0] +
      (ulong)kv_head * INPUT1_PITCHES[1] +
      (ulong)iq36_hot_key_packed_blocks() * INPUT1_PITCHES[2];
}

#if defined(IQ36_DIRECT_I8_GROUP4_FULL_COLD) && \
    !defined(IQ36_DIMENSION_MAJOR_VALUE_PLANE)
#define IQ36_DIMENSION_MAJOR_VALUE_PLANE 1
#endif

#if defined(IQ36_DIMENSION_MAJOR_VALUE_PLANE)
inline ulong iq36_hot_value_dimension_i32_base(
    const uint batch, const uint kv_head) {
  return INPUT1_OFFSET + (ulong)batch * INPUT1_PITCHES[0] +
      (ulong)kv_head * INPUT1_PITCHES[1] +
      (ulong)(2U * iq36_hot_key_packed_blocks()) * INPUT1_PITCHES[2];
}

inline void iq36_direct_store_hot_value_dimension(
    const uint batch,
    const uint kv_head,
    const uint slot,
    const uint dim,
    const half value,
    __global INPUT1_TYPE* hot_key_bits) {
  __global half* plane = (__global half*)&hot_key_bits[
      iq36_hot_value_dimension_i32_base(batch, kv_head)];
  plane[(ulong)dim * (uint)INPUT2_DIMS[2] + slot] = value;
}

inline int8 iq36_direct_hot_value_fragment(
    const uint batch,
    const uint kv_head,
    const uint block_slot,
    const uint dim,
    const __global INPUT1_TYPE* hot_key_bits) {
  const __global half* plane = (const __global half*)&hot_key_bits[
      iq36_hot_value_dimension_i32_base(batch, kv_head)];
  return as_int8(vload16(
      0, plane + (ulong)dim * (uint)INPUT2_DIMS[2] + block_slot));
}
#endif

inline uint iq36_cold_tokens(const __global INPUT5_TYPE* cold_key) {
  const uint d0 = (uint)(uchar)cold_key[INPUT5_OFFSET];
  const uint d1 = (uint)(uchar)cold_key[INPUT5_OFFSET + 1U];
  const uint d2 = (uint)(uchar)cold_key[INPUT5_OFFSET + 2U];
  return d0 + 128U * d1 + 16384U * d2;
}

#if defined(IQ36_DIRECT_I8_FIXED_LAYOUT)
inline uint iq36_direct_cold_capacity() {
  return (uint)INPUT5_DIMS[2] - 1U;
}

inline ulong iq36_direct_cold_key_payload(
    const uint batch, const uint kv_head) {
  return INPUT5_OFFSET + (ulong)batch * INPUT5_PITCHES[0] +
      (ulong)kv_head * INPUT5_PITCHES[1] + INPUT5_PITCHES[2];
}

inline ulong iq36_direct_cold_value_payload(
    const uint batch, const uint kv_head) {
  return INPUT6_OFFSET + (ulong)batch * INPUT6_PITCHES[0] +
      (ulong)kv_head * INPUT6_PITCHES[1] + INPUT6_PITCHES[2];
}

inline ulong iq36_direct_cold_key_scale_payload(
    const uint batch, const uint kv_head) {
  return INPUT7_OFFSET + (ulong)batch * INPUT7_PITCHES[0] +
      (ulong)kv_head * INPUT7_PITCHES[1] + INPUT7_PITCHES[2];
}

inline ulong iq36_direct_cold_value_scale_payload(
    const uint batch, const uint kv_head) {
  return INPUT8_OFFSET + (ulong)batch * INPUT8_PITCHES[0] +
      (ulong)kv_head * INPUT8_PITCHES[1] + INPUT8_PITCHES[2];
}

#if defined(IQ36_ADAPTIVE_PACKED_KV)
inline uint iq36_packed_code_piece(
    const uint code, const uint bit_position, const uint word) {
  const uint code_word = bit_position >> 5U;
  const uint shift = bit_position & 31U;
  uint piece = word == code_word ? code << shift : 0U;
  if (shift != 0U && word == code_word + 1U)
    piece |= code >> (32U - shift);
  return piece;
}

inline uint iq36_subgroup_pack_two_codes(
    const int value0,
    const int value1,
    const uint quant_bits,
    const uint word) {
  const uint lane = (uint)get_sub_group_local_id();
  const uint mask = (1U << quant_bits) - 1U;
  const uint code0 = (uint)value0 & mask;
  const uint code1 = (uint)value1 & mask;
  const uint bit0 = (lane * 2U) * quant_bits;
  const uint bit1 = bit0 + quant_bits;
  const uint piece =
      iq36_packed_code_piece(code0, bit0, word) |
      iq36_packed_code_piece(code1, bit1, word);
  // The bit fields are disjoint, so integer addition is an OR reduction with
  // an OpenCL subgroup primitive that is available on the pinned IGC stack.
  return sub_group_reduce_add(piece);
}

inline uint iq36_uint8_at(const uint8 words, const uint index) {
  switch (index) {
    case 0U: return words.s0;
    case 1U: return words.s1;
    case 2U: return words.s2;
    case 3U: return words.s3;
    case 4U: return words.s4;
    case 5U: return words.s5;
    case 6U: return words.s6;
    default: return words.s7;
  }
}

inline int iq36_unpack_signed_code(
    const uint8 words,
    const uint index,
    const uint quant_bits) {
  const uint bit = index * quant_bits;
  const uint word = bit >> 5U;
  const uint shift = bit & 31U;
  uint code = iq36_uint8_at(words, word) >> shift;
  if (shift + quant_bits > 32U)
    code |= iq36_uint8_at(words, word + 1U) << (32U - shift);
  const uint mask = (1U << quant_bits) - 1U;
  const uint sign = 1U << (quant_bits - 1U);
  code &= mask;
  return (int)((code ^ sign) - sign);
}

inline uint iq36_direct_cold_packed_word(
    const __global char* payload,
    const uint token,
    const uint words_per_token,
    const uint word) {
  const uint token_block = token / IQ36_TOKEN_TILE;
  const uint token_lane = token & (IQ36_TOKEN_TILE - 1U);
  const __global uint* packed = (const __global uint*)payload;
  return packed[
      ((ulong)token_block * words_per_token + word) * IQ36_TOKEN_TILE +
      token_lane];
}

inline uint iq36_direct_cold_packed_subgroup_word(
    const __global char* payload,
    const uint token,
    const uint words_per_token,
    const uint word) {
  const uint token_block = token / IQ36_TOKEN_TILE;
  const __global uint* packed = (const __global uint*)payload;
  return intel_sub_group_block_read(
      packed + ((ulong)token_block * words_per_token + word) *
          IQ36_TOKEN_TILE);
}

inline uint8 iq36_direct_cold_packed_subgroup_words6(
    const __global char* payload,
    const uint token,
    const uint words_per_token,
    const uint first_word) {
  return (uint8)(
      iq36_direct_cold_packed_subgroup_word(
          payload, token, words_per_token, first_word + 0U),
      iq36_direct_cold_packed_subgroup_word(
          payload, token, words_per_token, first_word + 1U),
      iq36_direct_cold_packed_subgroup_word(
          payload, token, words_per_token, first_word + 2U),
      iq36_direct_cold_packed_subgroup_word(
          payload, token, words_per_token, first_word + 3U),
      iq36_direct_cold_packed_subgroup_word(
          payload, token, words_per_token, first_word + 4U),
      iq36_direct_cold_packed_subgroup_word(
          payload, token, words_per_token, first_word + 5U),
      0U, 0U);
}

inline uint8 iq36_direct_cold_packed_subgroup_words7(
    const __global char* payload,
    const uint token,
    const uint words_per_token,
    const uint first_word) {
  return (uint8)(
      iq36_direct_cold_packed_subgroup_word(
          payload, token, words_per_token, first_word + 0U),
      iq36_direct_cold_packed_subgroup_word(
          payload, token, words_per_token, first_word + 1U),
      iq36_direct_cold_packed_subgroup_word(
          payload, token, words_per_token, first_word + 2U),
      iq36_direct_cold_packed_subgroup_word(
          payload, token, words_per_token, first_word + 3U),
      iq36_direct_cold_packed_subgroup_word(
          payload, token, words_per_token, first_word + 4U),
      iq36_direct_cold_packed_subgroup_word(
          payload, token, words_per_token, first_word + 5U),
      iq36_direct_cold_packed_subgroup_word(
          payload, token, words_per_token, first_word + 6U),
      0U);
}

inline uint8 iq36_direct_cold_packed_subgroup_words8(
    const __global char* payload,
    const uint token,
    const uint words_per_token,
    const uint first_word) {
  return (uint8)(
      iq36_direct_cold_packed_subgroup_word(
          payload, token, words_per_token, first_word + 0U),
      iq36_direct_cold_packed_subgroup_word(
          payload, token, words_per_token, first_word + 1U),
      iq36_direct_cold_packed_subgroup_word(
          payload, token, words_per_token, first_word + 2U),
      iq36_direct_cold_packed_subgroup_word(
          payload, token, words_per_token, first_word + 3U),
      iq36_direct_cold_packed_subgroup_word(
          payload, token, words_per_token, first_word + 4U),
      iq36_direct_cold_packed_subgroup_word(
          payload, token, words_per_token, first_word + 5U),
      iq36_direct_cold_packed_subgroup_word(
          payload, token, words_per_token, first_word + 6U),
      iq36_direct_cold_packed_subgroup_word(
          payload, token, words_per_token, first_word + 7U));
}

inline int iq36_direct_cold_packed_code(
    const __global char* payload,
    const uint token,
    const uint words_per_token,
    const uint first_word,
    const uint index,
    const uint quant_bits) {
  const uint bit = index * quant_bits;
  const uint word = bit >> 5U;
  const uint shift = bit & 31U;
  uint code = iq36_direct_cold_packed_word(
      payload, token, words_per_token, first_word + word) >> shift;
  if (shift + quant_bits > 32U) {
    code |= iq36_direct_cold_packed_word(
        payload, token, words_per_token, first_word + word + 1U) <<
        (32U - shift);
  }
  const uint mask = (1U << quant_bits) - 1U;
  const uint sign = 1U << (quant_bits - 1U);
  code &= mask;
  return (int)((code ^ sign) - sign);
}

inline void iq36_direct_store_cold_packed_word(
    __global char* payload,
    const uint token,
    const uint words_per_token,
    const uint word,
    const uint value) {
  const uint token_block = token / IQ36_TOKEN_TILE;
  const uint token_lane = token & (IQ36_TOKEN_TILE - 1U);
  __global uint* packed = (__global uint*)payload;
  packed[((ulong)token_block * words_per_token + word) *
      IQ36_TOKEN_TILE + token_lane] = value;
}
#endif

#if defined(IQ36_KEY_RESIDUAL1) || defined(IQ36_VALUE_RESIDUAL1)
inline int iq36_residual1_fine_quantize(
    const float value, const half stored_scale) {
  return clamp(
      (int)rint(value * 2.0f / convert_float(stored_scale)), -254, 254);
}

inline int iq36_residual1_base(const int fine) {
  return convert_int_rtn(convert_float(fine + 1) * 0.5f);
}

inline uint iq36_residual1_bit(const int fine, const int base) {
  return (uint)(fine - 2 * base + 1);
}

inline half16 iq36_residual1_low16(const uint bits) {
  const uint16 shifts = (uint16)(
      0U, 1U, 2U, 3U, 4U, 5U, 6U, 7U,
      8U, 9U, 10U, 11U, 12U, 13U, 14U, 15U);
  return convert_half16(((uint16)(bits) >> shifts) & (uint16)(1U)) *
      (half)0.5h - (half)0.5h;
}

inline half16 iq36_residual1_high16(const uint bits) {
  const uint16 shifts = (uint16)(
      16U, 17U, 18U, 19U, 20U, 21U, 22U, 23U,
      24U, 25U, 26U, 27U, 28U, 29U, 30U, 31U);
  return convert_half16(((uint16)(bits) >> shifts) & (uint16)(1U)) *
      (half)0.5h - (half)0.5h;
}
#endif

#if defined(IQ36_KEY_RESIDUAL1)
inline ulong iq36_direct_cold_key_residual1_payload(
    const uint batch, const uint kv_head) {
  return iq36_direct_cold_key_scale_payload(batch, kv_head) +
      (ulong)IQ36_KEY_SCALE_BYTES * iq36_direct_cold_capacity();
}

inline uint iq36_direct_cold_key_residual1_word(
    const uint batch,
    const uint kv_head,
    const uint token,
    const uint scale_group,
    const __global INPUT7_TYPE* cold_key_scale_bytes) {
  const uint token_block = token / IQ36_TOKEN_TILE;
  const uint token_lane = token & (IQ36_TOKEN_TILE - 1U);
  const __global uint* residual =
      (const __global uint*)&cold_key_scale_bytes[
          iq36_direct_cold_key_residual1_payload(batch, kv_head)];
  return residual[
      ((ulong)token_block * IQ36_KEY_SCALE_GROUPS + scale_group) *
          IQ36_TOKEN_TILE + token_lane];
}

inline half iq36_direct_cold_key_residual1_element(
    const uint batch,
    const uint kv_head,
    const uint token,
    const uint dim,
    const __global INPUT7_TYPE* cold_key_scale_bytes) {
  const uint bits = iq36_direct_cold_key_residual1_word(
      batch, kv_head, token, dim / IQ36_KEY_QUANT_GROUP,
      cold_key_scale_bytes);
  return (bits & (1U << (dim & 31U))) != 0U
      ? (half)0.0h : (half)-0.5h;
}
#endif

#if defined(IQ36_VALUE_RESIDUAL1)
inline ulong iq36_direct_cold_value_residual1_payload(
    const uint batch, const uint kv_head) {
  return iq36_direct_cold_value_scale_payload(batch, kv_head) +
      (ulong)IQ36_VALUE_SCALE_BYTES * iq36_direct_cold_capacity();
}

inline uint iq36_direct_cold_value_residual1_word_count() {
  return (iq36_direct_cold_capacity() + IQ36_TOKEN_TILE - 1U) /
      IQ36_TOKEN_TILE;
}

inline ushort iq36_direct_cold_value_residual1_word(
    const uint batch,
    const uint kv_head,
    const uint token_block,
    const uint dim,
    const __global INPUT8_TYPE* cold_value_scale_bytes) {
  const __global ushort* residual =
      (const __global ushort*)&cold_value_scale_bytes[
          iq36_direct_cold_value_residual1_payload(batch, kv_head)];
  return residual[
      (ulong)dim * iq36_direct_cold_value_residual1_word_count() +
      token_block];
}

inline half iq36_direct_cold_value_residual1_element(
    const uint batch,
    const uint kv_head,
    const uint token,
    const uint dim,
    const __global INPUT8_TYPE* cold_value_scale_bytes) {
  const ushort bits = iq36_direct_cold_value_residual1_word(
      batch, kv_head, token / IQ36_TOKEN_TILE, dim,
      cold_value_scale_bytes);
  return (bits & (1U << (token & (IQ36_TOKEN_TILE - 1U)))) != 0U
      ? (half)0.0h : (half)-0.5h;
}
#endif

inline half iq36_direct_cold_key_scale(
    const uint batch,
    const uint kv_head,
    const uint token,
    const uint scale_group,
    const __global INPUT7_TYPE* cold_key_scale_bytes) {
  const uint capacity = iq36_direct_cold_capacity();
  const __global half* scales = (const __global half*)
      &cold_key_scale_bytes[
          iq36_direct_cold_key_scale_payload(batch, kv_head)];
  return scales[(ulong)scale_group * capacity + token];
}

inline half iq36_direct_cold_value_scale(
    const uint batch,
    const uint kv_head,
    const uint token,
    const uint scale_group,
    const __global INPUT8_TYPE* cold_value_scale_bytes) {
  const uint capacity = iq36_direct_cold_capacity();
  const __global half* scales = (const __global half*)
      &cold_value_scale_bytes[
          iq36_direct_cold_value_scale_payload(batch, kv_head)];
  return scales[(ulong)scale_group * capacity + token];
}

#if IQ36_KEY_QUANT_GROUP == 32U
inline int16 iq36_direct_cold_key_group32_fragments(
    const uint batch,
    const uint kv_head,
    const uint token,
    const uint scale_group,
    const __global INPUT5_TYPE* cold_key,
    const __global INPUT7_TYPE* cold_key_scale_bytes) {
#if defined(IQ36_ADAPTIVE_PACKED_KV)
  const __global char* payload = (const __global char*)&cold_key[
      iq36_direct_cold_key_payload(batch, kv_head)];
  const uint8 words =
#if IQ36_KEY_QUANT_BITS == 6U
      iq36_direct_cold_packed_subgroup_words6(
#elif IQ36_KEY_QUANT_BITS == 7U
      iq36_direct_cold_packed_subgroup_words7(
#else
      iq36_direct_cold_packed_subgroup_words8(
#endif
      payload, token, IQ36_KEY_PACK_WORDS,
      scale_group * IQ36_KEY_QUANT_BITS);
  const half scale = iq36_direct_cold_key_scale(
      batch, kv_head, token, scale_group, cold_key_scale_bytes);
  const half16 low = (half16)(
      (half)iq36_unpack_signed_code(words, 0U, IQ36_KEY_QUANT_BITS),
      (half)iq36_unpack_signed_code(words, 1U, IQ36_KEY_QUANT_BITS),
      (half)iq36_unpack_signed_code(words, 2U, IQ36_KEY_QUANT_BITS),
      (half)iq36_unpack_signed_code(words, 3U, IQ36_KEY_QUANT_BITS),
      (half)iq36_unpack_signed_code(words, 4U, IQ36_KEY_QUANT_BITS),
      (half)iq36_unpack_signed_code(words, 5U, IQ36_KEY_QUANT_BITS),
      (half)iq36_unpack_signed_code(words, 6U, IQ36_KEY_QUANT_BITS),
      (half)iq36_unpack_signed_code(words, 7U, IQ36_KEY_QUANT_BITS),
      (half)iq36_unpack_signed_code(words, 8U, IQ36_KEY_QUANT_BITS),
      (half)iq36_unpack_signed_code(words, 9U, IQ36_KEY_QUANT_BITS),
      (half)iq36_unpack_signed_code(words, 10U, IQ36_KEY_QUANT_BITS),
      (half)iq36_unpack_signed_code(words, 11U, IQ36_KEY_QUANT_BITS),
      (half)iq36_unpack_signed_code(words, 12U, IQ36_KEY_QUANT_BITS),
      (half)iq36_unpack_signed_code(words, 13U, IQ36_KEY_QUANT_BITS),
      (half)iq36_unpack_signed_code(words, 14U, IQ36_KEY_QUANT_BITS),
      (half)iq36_unpack_signed_code(words, 15U, IQ36_KEY_QUANT_BITS));
  const half16 high = (half16)(
      (half)iq36_unpack_signed_code(words, 16U, IQ36_KEY_QUANT_BITS),
      (half)iq36_unpack_signed_code(words, 17U, IQ36_KEY_QUANT_BITS),
      (half)iq36_unpack_signed_code(words, 18U, IQ36_KEY_QUANT_BITS),
      (half)iq36_unpack_signed_code(words, 19U, IQ36_KEY_QUANT_BITS),
      (half)iq36_unpack_signed_code(words, 20U, IQ36_KEY_QUANT_BITS),
      (half)iq36_unpack_signed_code(words, 21U, IQ36_KEY_QUANT_BITS),
      (half)iq36_unpack_signed_code(words, 22U, IQ36_KEY_QUANT_BITS),
      (half)iq36_unpack_signed_code(words, 23U, IQ36_KEY_QUANT_BITS),
      (half)iq36_unpack_signed_code(words, 24U, IQ36_KEY_QUANT_BITS),
      (half)iq36_unpack_signed_code(words, 25U, IQ36_KEY_QUANT_BITS),
      (half)iq36_unpack_signed_code(words, 26U, IQ36_KEY_QUANT_BITS),
      (half)iq36_unpack_signed_code(words, 27U, IQ36_KEY_QUANT_BITS),
      (half)iq36_unpack_signed_code(words, 28U, IQ36_KEY_QUANT_BITS),
      (half)iq36_unpack_signed_code(words, 29U, IQ36_KEY_QUANT_BITS),
      (half)iq36_unpack_signed_code(words, 30U, IQ36_KEY_QUANT_BITS),
      (half)iq36_unpack_signed_code(words, 31U, IQ36_KEY_QUANT_BITS));
  return (int16)(as_int8(low * scale), as_int8(high * scale));
#else
  const uint token_block = token / IQ36_TOKEN_TILE;
  const ulong word =
      ((ulong)token_block * IQ36_KEY_PACK_WORDS + scale_group * 8U) *
          IQ36_TOKEN_TILE;
  const __global uint* packed = (const __global uint*)&cold_key[
      iq36_direct_cold_key_payload(batch, kv_head)];
  const uint8 raw = intel_sub_group_block_read8(packed + word);
  const half scale = iq36_direct_cold_key_scale(
      batch, kv_head, token, scale_group, cold_key_scale_bytes);
#if defined(IQ36_KEY_RESIDUAL1)
  const uint residual_bits = iq36_direct_cold_key_residual1_word(
      batch, kv_head, token, scale_group, cold_key_scale_bytes);
  const half16 low = convert_half16(as_char16(raw.s0123)) +
      iq36_residual1_low16(residual_bits);
  const half16 high = convert_half16(as_char16(raw.s4567)) +
      iq36_residual1_high16(residual_bits);
  return (int16)(
      as_int8(low * scale), as_int8(high * scale));
#else
  return (int16)(
      as_int8(convert_half16(as_char16(raw.s0123)) * scale),
      as_int8(convert_half16(as_char16(raw.s4567)) * scale));
#endif
#endif
}
#endif

inline char iq36_direct_cold_key_element(
    const uint batch,
    const uint kv_head,
    const uint token,
    const uint dim,
    const __global INPUT5_TYPE* cold_key) {
#if defined(IQ36_ADAPTIVE_PACKED_KV)
  const uint scale_group = dim / IQ36_KEY_QUANT_GROUP;
  const uint code_index = dim - scale_group * IQ36_KEY_QUANT_GROUP;
  const __global char* payload = (const __global char*)&cold_key[
      iq36_direct_cold_key_payload(batch, kv_head)];
  return (char)iq36_direct_cold_packed_code(
      payload, token, IQ36_KEY_PACK_WORDS,
      scale_group * IQ36_KEY_QUANT_BITS,
      code_index, IQ36_KEY_QUANT_BITS);
#else
  const uint token_block = token / IQ36_TOKEN_TILE;
  const uint token_lane = token & (IQ36_TOKEN_TILE - 1U);
  const uint byte_in_word = dim & 3U;
  const ulong word =
      ((ulong)token_block * IQ36_KEY_PACK_WORDS + (dim >> 2U)) *
          IQ36_TOKEN_TILE + token_lane;
  return (char)cold_key[
      iq36_direct_cold_key_payload(batch, kv_head) + word * 4U +
      byte_in_word];
#endif
}

inline half iq36_direct_cold_value_element(
    const uint batch,
    const uint kv_head,
    const uint token,
    const uint dim,
    const __global INPUT6_TYPE* cold_value,
    const __global INPUT8_TYPE* cold_value_scale_bytes) {
#if defined(IQ36_ADAPTIVE_PACKED_KV) && \
    !defined(IQ36_ADAPTIVE_DIMENSION_MAJOR_V8)
  const uint scale_group = dim / IQ36_VALUE_QUANT_GROUP;
  const uint code_index = dim - scale_group * IQ36_VALUE_QUANT_GROUP;
  const __global char* payload = (const __global char*)&cold_value[
      iq36_direct_cold_value_payload(batch, kv_head)];
  return (half)iq36_direct_cold_packed_code(
      payload, token, IQ36_VALUE_PACK_WORDS,
      scale_group * IQ36_VALUE_QUANT_BITS,
      code_index, IQ36_VALUE_QUANT_BITS) *
      iq36_direct_cold_value_scale(
          batch, kv_head, token, scale_group, cold_value_scale_bytes);
#else
  const uint capacity = iq36_direct_cold_capacity();
  const char quantized = (char)cold_value[
      iq36_direct_cold_value_payload(batch, kv_head) +
      (ulong)dim * capacity + token];
  half reconstructed = convert_half(quantized);
#if defined(IQ36_VALUE_RESIDUAL1)
  reconstructed += iq36_direct_cold_value_residual1_element(
      batch, kv_head, token, dim, cold_value_scale_bytes);
#endif
  return reconstructed * iq36_direct_cold_value_scale(
      batch, kv_head, token, dim / IQ36_VALUE_QUANT_GROUP,
      cold_value_scale_bytes);
#endif
}

inline int8 iq36_direct_cold_value_fragment(
    const uint batch,
    const uint kv_head,
    const uint block_token,
    const uint dim,
    const __global INPUT6_TYPE* cold_value,
    const __global INPUT8_TYPE* cold_value_scale_bytes) {
#if defined(IQ36_ADAPTIVE_PACKED_KV) && \
    !defined(IQ36_ADAPTIVE_DIMENSION_MAJOR_V8)
  half16 values = (half16)(0.0h);
  #pragma unroll
  for (uint row = 0U; row < IQ36_TOKEN_TILE; ++row) {
    values[row] = iq36_direct_cold_value_element(
        batch, kv_head, block_token + row, dim,
        cold_value, cold_value_scale_bytes);
  }
  return as_int8(values);
#else
  const uint capacity = iq36_direct_cold_capacity();
  const char16 quantized = vload16(
      0, (const __global char*)&cold_value[
          iq36_direct_cold_value_payload(batch, kv_head) +
          (ulong)dim * capacity + block_token]);
  const half16 scales = vload16(
      0, (const __global half*)&cold_value_scale_bytes[
          iq36_direct_cold_value_scale_payload(batch, kv_head)] +
          (ulong)(dim / IQ36_VALUE_QUANT_GROUP) * capacity + block_token);
  half16 reconstructed = convert_half16(quantized);
#if defined(IQ36_VALUE_RESIDUAL1)
  const uint residual_bits = (uint)iq36_direct_cold_value_residual1_word(
      batch, kv_head, block_token / IQ36_TOKEN_TILE, dim,
      cold_value_scale_bytes);
  reconstructed += iq36_residual1_low16(residual_bits);
#endif
  return as_int8(reconstructed * scales);
#endif
}

inline int16 iq36_direct_cold_value_group32_fragments_unscaled(
    const uint batch,
    const uint kv_head,
    const uint block_token,
    const uint scale_group,
    const __global INPUT6_TYPE* cold_value,
    const __global INPUT8_TYPE* cold_value_scale_bytes) {
#if defined(IQ36_ADAPTIVE_PACKED_KV) && \
    !defined(IQ36_ADAPTIVE_DIMENSION_MAJOR_V8)
  const uint token = block_token + (uint)get_sub_group_local_id();
  const __global char* payload = (const __global char*)&cold_value[
      iq36_direct_cold_value_payload(batch, kv_head)];
  const uint8 words =
#if IQ36_VALUE_QUANT_BITS == 6U
      iq36_direct_cold_packed_subgroup_words6(
#elif IQ36_VALUE_QUANT_BITS == 7U
      iq36_direct_cold_packed_subgroup_words7(
#else
      iq36_direct_cold_packed_subgroup_words8(
#endif
      payload, token, IQ36_VALUE_PACK_WORDS,
      scale_group * IQ36_VALUE_QUANT_BITS);
  const half16 low = (half16)(
      (half)iq36_unpack_signed_code(words, 0U, IQ36_VALUE_QUANT_BITS),
      (half)iq36_unpack_signed_code(words, 1U, IQ36_VALUE_QUANT_BITS),
      (half)iq36_unpack_signed_code(words, 2U, IQ36_VALUE_QUANT_BITS),
      (half)iq36_unpack_signed_code(words, 3U, IQ36_VALUE_QUANT_BITS),
      (half)iq36_unpack_signed_code(words, 4U, IQ36_VALUE_QUANT_BITS),
      (half)iq36_unpack_signed_code(words, 5U, IQ36_VALUE_QUANT_BITS),
      (half)iq36_unpack_signed_code(words, 6U, IQ36_VALUE_QUANT_BITS),
      (half)iq36_unpack_signed_code(words, 7U, IQ36_VALUE_QUANT_BITS),
      (half)iq36_unpack_signed_code(words, 8U, IQ36_VALUE_QUANT_BITS),
      (half)iq36_unpack_signed_code(words, 9U, IQ36_VALUE_QUANT_BITS),
      (half)iq36_unpack_signed_code(words, 10U, IQ36_VALUE_QUANT_BITS),
      (half)iq36_unpack_signed_code(words, 11U, IQ36_VALUE_QUANT_BITS),
      (half)iq36_unpack_signed_code(words, 12U, IQ36_VALUE_QUANT_BITS),
      (half)iq36_unpack_signed_code(words, 13U, IQ36_VALUE_QUANT_BITS),
      (half)iq36_unpack_signed_code(words, 14U, IQ36_VALUE_QUANT_BITS),
      (half)iq36_unpack_signed_code(words, 15U, IQ36_VALUE_QUANT_BITS));
  const half16 high = (half16)(
      (half)iq36_unpack_signed_code(words, 16U, IQ36_VALUE_QUANT_BITS),
      (half)iq36_unpack_signed_code(words, 17U, IQ36_VALUE_QUANT_BITS),
      (half)iq36_unpack_signed_code(words, 18U, IQ36_VALUE_QUANT_BITS),
      (half)iq36_unpack_signed_code(words, 19U, IQ36_VALUE_QUANT_BITS),
      (half)iq36_unpack_signed_code(words, 20U, IQ36_VALUE_QUANT_BITS),
      (half)iq36_unpack_signed_code(words, 21U, IQ36_VALUE_QUANT_BITS),
      (half)iq36_unpack_signed_code(words, 22U, IQ36_VALUE_QUANT_BITS),
      (half)iq36_unpack_signed_code(words, 23U, IQ36_VALUE_QUANT_BITS),
      (half)iq36_unpack_signed_code(words, 24U, IQ36_VALUE_QUANT_BITS),
      (half)iq36_unpack_signed_code(words, 25U, IQ36_VALUE_QUANT_BITS),
      (half)iq36_unpack_signed_code(words, 26U, IQ36_VALUE_QUANT_BITS),
      (half)iq36_unpack_signed_code(words, 27U, IQ36_VALUE_QUANT_BITS),
      (half)iq36_unpack_signed_code(words, 28U, IQ36_VALUE_QUANT_BITS),
      (half)iq36_unpack_signed_code(words, 29U, IQ36_VALUE_QUANT_BITS),
      (half)iq36_unpack_signed_code(words, 30U, IQ36_VALUE_QUANT_BITS),
      (half)iq36_unpack_signed_code(words, 31U, IQ36_VALUE_QUANT_BITS));
  return (int16)(as_int8(low), as_int8(high));
#else
  const uint capacity = iq36_direct_cold_capacity();
  const __global char* plane = (const __global char*)&cold_value[
      iq36_direct_cold_value_payload(batch, kv_head)];
  ulong address = as_long(plane);
  const ulong prefix = address & 0x3fUL;
  address &= ~0x3fUL;
  const uint8 packed =
      __builtin_IB_subgroup_block_read_cacheopts_transpose_u32_m32k4(
      (long)address, (int)capacity + (int)prefix - 1,
      (int)IQ36_HEAD_DIM - 1, (int)capacity - 1,
      (int2)((int)((block_token + (uint)prefix) / sizeof(uint)),
             (int)(scale_group * IQ36_VALUE_QUANT_GROUP)), 0);
  half16 low = convert_half16(as_char16(packed.s0246));
  half16 high = convert_half16(as_char16(packed.s1357));
#if defined(IQ36_VALUE_RESIDUAL1)
  const uint lane = (uint)get_sub_group_local_id();
  const uint token_block = block_token / IQ36_TOKEN_TILE;
  const uint low_bits = (uint)iq36_direct_cold_value_residual1_word(
      batch, kv_head, token_block,
      scale_group * IQ36_VALUE_QUANT_GROUP + lane,
      cold_value_scale_bytes);
  const uint high_bits = (uint)iq36_direct_cold_value_residual1_word(
      batch, kv_head, token_block,
      scale_group * IQ36_VALUE_QUANT_GROUP + lane + 16U,
      cold_value_scale_bytes);
  low += iq36_residual1_low16(low_bits);
  high += iq36_residual1_low16(high_bits);
#endif
  return (int16)(as_int8(low), as_int8(high));
#endif
}

#if defined(IQ36_ADAPTIVE_PACKED_KV)
inline void iq36_direct_store_cold_key_packed_word(
    const uint batch,
    const uint kv_head,
    const uint token,
    const uint scale_group,
    const uint word,
    const uint value,
    __global INPUT5_TYPE* cold_key) {
  __global char* payload = (__global char*)&cold_key[
      iq36_direct_cold_key_payload(batch, kv_head)];
  iq36_direct_store_cold_packed_word(
      payload, token, IQ36_KEY_PACK_WORDS,
      scale_group * IQ36_KEY_QUANT_BITS + word, value);
}

inline void iq36_direct_store_cold_value_packed_word(
    const uint batch,
    const uint kv_head,
    const uint token,
    const uint scale_group,
    const uint word,
    const uint value,
    __global INPUT6_TYPE* cold_value) {
  __global char* payload = (__global char*)&cold_value[
      iq36_direct_cold_value_payload(batch, kv_head)];
  iq36_direct_store_cold_packed_word(
      payload, token, IQ36_VALUE_PACK_WORDS,
      scale_group * IQ36_VALUE_QUANT_BITS + word, value);
}
#endif

inline void iq36_direct_store_cold_key(
    const uint batch,
    const uint kv_head,
    const uint token,
    const uint dim,
    const char value,
    __global INPUT5_TYPE* cold_key) {
  const uint token_block = token / IQ36_TOKEN_TILE;
  const uint token_lane = token & (IQ36_TOKEN_TILE - 1U);
  const uint byte_in_word = dim & 3U;
  const ulong word =
      ((ulong)token_block * IQ36_KEY_PACK_WORDS + (dim >> 2U)) *
          IQ36_TOKEN_TILE + token_lane;
  cold_key[iq36_direct_cold_key_payload(batch, kv_head) + word * 4U +
      byte_in_word] = (INPUT5_TYPE)value;
}

inline void iq36_direct_store_cold_value(
    const uint batch,
    const uint kv_head,
    const uint token,
    const uint dim,
    const char value,
    __global INPUT6_TYPE* cold_value) {
  const uint capacity = iq36_direct_cold_capacity();
  cold_value[iq36_direct_cold_value_payload(batch, kv_head) +
      (ulong)dim * capacity + token] = (INPUT6_TYPE)value;
}

inline void iq36_direct_store_cold_key_scale(
    const uint batch,
    const uint kv_head,
    const uint token,
    const uint scale_group,
    const half value,
    __global INPUT7_TYPE* cold_key_scale_bytes) {
  const uint capacity = iq36_direct_cold_capacity();
  __global half* scales = (__global half*)&cold_key_scale_bytes[
      iq36_direct_cold_key_scale_payload(batch, kv_head)];
  scales[(ulong)scale_group * capacity + token] = value;
}

inline void iq36_direct_store_cold_value_scale(
    const uint batch,
    const uint kv_head,
    const uint token,
    const uint scale_group,
    const half value,
    __global INPUT8_TYPE* cold_value_scale_bytes) {
  const uint capacity = iq36_direct_cold_capacity();
  __global half* scales = (__global half*)&cold_value_scale_bytes[
      iq36_direct_cold_value_scale_payload(batch, kv_head)];
  scales[(ulong)scale_group * capacity + token] = value;
}

#if defined(IQ36_KEY_RESIDUAL1)
inline void iq36_direct_store_cold_key_residual1_word(
    const uint batch,
    const uint kv_head,
    const uint token,
    const uint scale_group,
    const uint bits,
    __global INPUT7_TYPE* cold_key_scale_bytes) {
  const uint token_block = token / IQ36_TOKEN_TILE;
  const uint token_lane = token & (IQ36_TOKEN_TILE - 1U);
  __global uint* residual = (__global uint*)&cold_key_scale_bytes[
      iq36_direct_cold_key_residual1_payload(batch, kv_head)];
  residual[
      ((ulong)token_block * IQ36_KEY_SCALE_GROUPS + scale_group) *
          IQ36_TOKEN_TILE + token_lane] = bits;
}
#endif

#if defined(IQ36_VALUE_RESIDUAL1)
inline void iq36_direct_store_cold_value_residual1_bit(
    const uint batch,
    const uint kv_head,
    const uint token,
    const uint dim,
    const uint bit,
    __global INPUT8_TYPE* cold_value_scale_bytes) {
  const uint token_block = token / IQ36_TOKEN_TILE;
  const uint token_lane = token & (IQ36_TOKEN_TILE - 1U);
  __global ushort* residual = (__global ushort*)&cold_value_scale_bytes[
      iq36_direct_cold_value_residual1_payload(batch, kv_head)];
  const ulong index =
      (ulong)dim * iq36_direct_cold_value_residual1_word_count() +
      token_block;
  const ushort mask = (ushort)(bit << token_lane);
  residual[index] = token_lane == 0U
      ? mask : (ushort)(residual[index] | mask);
}
#endif
#endif

inline int8 iq36_cold_key_fragment(
    const uint batch,
    const uint kv_head,
    const uint token,
    const uint k_base,
    const __global INPUT5_TYPE* cold_key,
    const __global INPUT7_TYPE* cold_key_scale_bytes) {
#if defined(IQ36_DIRECT_I8_FIXED_LAYOUT)
  const uint token_block = token / IQ36_TOKEN_TILE;
#if defined(IQ36_DIRECT_I8_GROUP4_FULL_COLD)
  const uint scale_group = k_base / IQ36_KEY_QUANT_GROUP;
  const ulong word =
      ((ulong)token_block * IQ36_KEY_PACK_WORDS + (k_base >> 2U)) *
          IQ36_TOKEN_TILE;
  const __global uint* packed = (const __global uint*)&cold_key[
      iq36_direct_cold_key_payload(batch, kv_head)];
  const uint packed0 = intel_sub_group_block_read(
      packed + word + 0U * IQ36_TOKEN_TILE);
  const uint packed1 = intel_sub_group_block_read(
      packed + word + 1U * IQ36_TOKEN_TILE);
  const uint packed2 = intel_sub_group_block_read(
      packed + word + 2U * IQ36_TOKEN_TILE);
  const uint packed3 = intel_sub_group_block_read(
      packed + word + 3U * IQ36_TOKEN_TILE);
  const char16 quantized = (char16)(
      as_char4(packed0), as_char4(packed1),
      as_char4(packed2), as_char4(packed3));
  const half scale0 = iq36_direct_cold_key_scale(
      batch, kv_head, token, scale_group + 0U, cold_key_scale_bytes);
  const half scale1 = iq36_direct_cold_key_scale(
      batch, kv_head, token, scale_group + 1U, cold_key_scale_bytes);
  const half scale2 = iq36_direct_cold_key_scale(
      batch, kv_head, token, scale_group + 2U, cold_key_scale_bytes);
  const half scale3 = iq36_direct_cold_key_scale(
      batch, kv_head, token, scale_group + 3U, cold_key_scale_bytes);
#if defined(IQ36_DIRECT_I8_HYBRID_K2_V4)
  const half scale4 = iq36_direct_cold_key_scale(
      batch, kv_head, token, scale_group + 4U, cold_key_scale_bytes);
  const half scale5 = iq36_direct_cold_key_scale(
      batch, kv_head, token, scale_group + 5U, cold_key_scale_bytes);
  const half scale6 = iq36_direct_cold_key_scale(
      batch, kv_head, token, scale_group + 6U, cold_key_scale_bytes);
  const half scale7 = iq36_direct_cold_key_scale(
      batch, kv_head, token, scale_group + 7U, cold_key_scale_bytes);
  const half16 scales = (half16)(
      scale0, scale0, scale1, scale1,
      scale2, scale2, scale3, scale3,
      scale4, scale4, scale5, scale5,
      scale6, scale6, scale7, scale7);
#else
  const half16 scales = (half16)(
      scale0, scale0, scale0, scale0,
      scale1, scale1, scale1, scale1,
      scale2, scale2, scale2, scale2,
      scale3, scale3, scale3, scale3);
#endif
  return as_int8(convert_half16(quantized) * scales);
#else
  const uint scale_group = k_base >> 5U;
  const ulong word =
      ((ulong)token_block * IQ36_KEY_PACK_WORDS + scale_group * 8U) *
          IQ36_TOKEN_TILE;
  const __global uint* packed = (const __global uint*)&cold_key[
      iq36_direct_cold_key_payload(batch, kv_head)];
  const uint8 raw = intel_sub_group_block_read8(packed + word);
  const char16 quantized = (k_base & 16U) == 0U
      ? as_char16(raw.s0123) : as_char16(raw.s4567);
  const half scale = iq36_direct_cold_key_scale(
      batch, kv_head, token, scale_group, cold_key_scale_bytes);
  half16 reconstructed = convert_half16(quantized);
#if defined(IQ36_KEY_RESIDUAL1)
  const uint residual_bits = iq36_direct_cold_key_residual1_word(
      batch, kv_head, token, scale_group, cold_key_scale_bytes);
  reconstructed += (k_base & 16U) == 0U
      ? iq36_residual1_low16(residual_bits)
      : iq36_residual1_high16(residual_bits);
#endif
  return as_int8(reconstructed * scale);
#endif
#else
  const uint row = token + 1U;
  const uint scale_byte = (k_base >> 5) << 1;
  const uint value_index = INPUT5_OFFSET + batch * INPUT5_PITCHES[0] +
      kv_head * INPUT5_PITCHES[1] + row * INPUT5_PITCHES[2] +
      k_base * INPUT5_PITCHES[3];
  const uint scale_index = INPUT7_OFFSET + batch * INPUT7_PITCHES[0] +
      kv_head * INPUT7_PITCHES[1] + row * INPUT7_PITCHES[2] +
      scale_byte * INPUT7_PITCHES[3];
  const ushort scale_bits =
      (ushort)(uchar)cold_key_scale_bytes[scale_index] |
      ((ushort)(uchar)cold_key_scale_bytes[
          scale_index + INPUT7_PITCHES[3]] << 8);
  const char16 quantized = vload16(0, &cold_key[value_index]);
  const half16 values = convert_half16(quantized) * as_half(scale_bits);
  return as_int8(values);
#endif
}

inline half iq36_cold_value_element(
    const uint batch,
    const uint kv_head,
    const uint token,
    const uint dim,
    const __global INPUT6_TYPE* cold_value,
    const __global INPUT8_TYPE* cold_value_scale_bytes) {
#if defined(IQ36_DIRECT_I8_FIXED_LAYOUT)
  return iq36_direct_cold_value_element(
      batch, kv_head, token, dim, cold_value, cold_value_scale_bytes);
#else
  const uint row = token + 1U;
  const uint scale_byte = (dim >> 5) << 1;
  const uint value_index = INPUT6_OFFSET + batch * INPUT6_PITCHES[0] +
      kv_head * INPUT6_PITCHES[1] + row * INPUT6_PITCHES[2] +
      dim * INPUT6_PITCHES[3];
  const uint scale_index = INPUT8_OFFSET + batch * INPUT8_PITCHES[0] +
      kv_head * INPUT8_PITCHES[1] + row * INPUT8_PITCHES[2] +
      scale_byte * INPUT8_PITCHES[3];
  const ushort scale_bits =
      (ushort)(uchar)cold_value_scale_bytes[scale_index] |
      ((ushort)(uchar)cold_value_scale_bytes[
          scale_index + INPUT8_PITCHES[3]] << 8);
  return convert_half(cold_value[value_index]) * as_half(scale_bits);
#endif
}

inline float iq36_partial_load_key(
    const uint batch,
    const uint kv_head,
    const uint token,
    const uint past_tokens,
    const uint cold_tokens,
    const uint dim,
    const __global INPUT1_TYPE* hot_key_bits,
    const __global INPUT3_TYPE* current_key,
    const __global INPUT5_TYPE* cold_key,
    const __global INPUT7_TYPE* cold_key_scale_bytes) {
  if (token < IQ36_SINK_TOKENS && token < past_tokens) {
    const uint slot = iq36_hot_slot(token);
    const uint word = (dim >> 1) * IQ36_KEY_TILE_TOKENS +
        (slot & (IQ36_KEY_TILE_TOKENS - 1U));
    const uint index = INPUT1_OFFSET + batch * INPUT1_PITCHES[0] +
        kv_head * INPUT1_PITCHES[1] +
        (slot / IQ36_KEY_TILE_TOKENS) * INPUT1_PITCHES[2] +
        word * INPUT1_PITCHES[3];
    const half2 values = as_half2((int)hot_key_bits[index]);
    return convert_float(values[dim & 1U]);
  }
  if (token < cold_tokens) {
#if defined(IQ36_ADAPTIVE_KEY_EXACT)
    const uint slot = iq36_hot_slot(token);
    const uint word = (dim >> 1) * IQ36_KEY_TILE_TOKENS +
        (slot & (IQ36_KEY_TILE_TOKENS - 1U));
    const uint index = INPUT1_OFFSET + batch * INPUT1_PITCHES[0] +
        kv_head * INPUT1_PITCHES[1] +
        (slot / IQ36_KEY_TILE_TOKENS) * INPUT1_PITCHES[2] +
        word * INPUT1_PITCHES[3];
    const half2 values = as_half2((int)hot_key_bits[index]);
    return convert_float(values[dim & 1U]);
#else
#if defined(IQ36_DIRECT_I8_FIXED_LAYOUT)
    const char quantized = iq36_direct_cold_key_element(
        batch, kv_head, token, dim, cold_key);
    const half scale = iq36_direct_cold_key_scale(
        batch, kv_head, token, dim / IQ36_KEY_QUANT_GROUP,
        cold_key_scale_bytes);
    float reconstructed = convert_float(quantized);
#if defined(IQ36_KEY_RESIDUAL1)
    reconstructed += convert_float(iq36_direct_cold_key_residual1_element(
        batch, kv_head, token, dim, cold_key_scale_bytes));
#endif
    return reconstructed * convert_float(scale);
#else
    const uint scale_byte = (dim >> 5) << 1;
    const uint row = token + 1U;
    const uint value_index = INPUT5_OFFSET + batch * INPUT5_PITCHES[0] +
        kv_head * INPUT5_PITCHES[1] + row * INPUT5_PITCHES[2] +
        dim * INPUT5_PITCHES[3];
    const uint scale_index = INPUT7_OFFSET + batch * INPUT7_PITCHES[0] +
        kv_head * INPUT7_PITCHES[1] + row * INPUT7_PITCHES[2] +
        scale_byte * INPUT7_PITCHES[3];
    const ushort bits =
        (ushort)(uchar)cold_key_scale_bytes[scale_index] |
        ((ushort)(uchar)cold_key_scale_bytes[
            scale_index + INPUT7_PITCHES[3]] << 8);
    return convert_float(cold_key[value_index]) * convert_float(as_half(bits));
#endif
#endif
  }
  if (token < past_tokens) {
    const uint slot = iq36_hot_slot(token);
    const uint word = (dim >> 1) * IQ36_KEY_TILE_TOKENS +
        (slot & (IQ36_KEY_TILE_TOKENS - 1U));
    const uint index = INPUT1_OFFSET + batch * INPUT1_PITCHES[0] +
        kv_head * INPUT1_PITCHES[1] +
        (slot / IQ36_KEY_TILE_TOKENS) * INPUT1_PITCHES[2] +
        word * INPUT1_PITCHES[3];
    const half2 values = as_half2((int)hot_key_bits[index]);
    return convert_float(values[dim & 1U]);
  }
  const uint current_position = token - past_tokens;
  const uint index = INPUT3_OFFSET + batch * INPUT3_PITCHES[0] +
      kv_head * INPUT3_PITCHES[1] + current_position * INPUT3_PITCHES[2] +
      dim * INPUT3_PITCHES[3];
  return convert_float(current_key[index]);
}

inline float iq36_partial_load_value(
    const uint batch,
    const uint kv_head,
    const uint token,
    const uint past_tokens,
    const uint cold_tokens,
    const uint dim,
    const __global INPUT2_TYPE* hot_value_bits,
    const __global INPUT4_TYPE* current_value,
    const __global INPUT6_TYPE* cold_value,
    const __global INPUT8_TYPE* cold_value_scale_bytes) {
  if (token < IQ36_SINK_TOKENS && token < past_tokens) {
    const uint slot = iq36_hot_slot(token);
    const uint index = INPUT2_OFFSET + batch * INPUT2_PITCHES[0] +
        kv_head * INPUT2_PITCHES[1] + slot * INPUT2_PITCHES[2] +
        dim * INPUT2_PITCHES[3];
    return convert_float(hot_value_bits[index]);
  }
  if (token < cold_tokens) {
#if defined(IQ36_DIRECT_I8_FIXED_LAYOUT)
    return convert_float(iq36_direct_cold_value_element(
        batch, kv_head, token, dim, cold_value,
        cold_value_scale_bytes));
#else
    const uint scale_byte = (dim >> 5) << 1;
    const uint row = token + 1U;
    const uint value_index = INPUT6_OFFSET + batch * INPUT6_PITCHES[0] +
        kv_head * INPUT6_PITCHES[1] + row * INPUT6_PITCHES[2] +
        dim * INPUT6_PITCHES[3];
    const uint scale_index = INPUT8_OFFSET + batch * INPUT8_PITCHES[0] +
        kv_head * INPUT8_PITCHES[1] + row * INPUT8_PITCHES[2] +
        scale_byte * INPUT8_PITCHES[3];
    const ushort bits =
        (ushort)(uchar)cold_value_scale_bytes[scale_index] |
        ((ushort)(uchar)cold_value_scale_bytes[
            scale_index + INPUT8_PITCHES[3]] << 8);
    return convert_float(cold_value[value_index]) *
        convert_float(as_half(bits));
#endif
  }
  if (token < past_tokens) {
    const uint slot = iq36_hot_slot(token);
    const uint index = INPUT2_OFFSET + batch * INPUT2_PITCHES[0] +
        kv_head * INPUT2_PITCHES[1] + slot * INPUT2_PITCHES[2] +
        dim * INPUT2_PITCHES[3];
    return convert_float(hot_value_bits[index]);
  }
  const uint current_position = token - past_tokens;
  const ulong index = iq36_current_value_index(
      batch, kv_head, current_position, dim);
  return convert_float(current_value[index]);
}
