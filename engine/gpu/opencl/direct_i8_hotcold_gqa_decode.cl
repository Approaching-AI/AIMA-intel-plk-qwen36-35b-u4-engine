#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable

// One locked Qwen3.6 full-attention layer at 32k.  Cold K is block16-token,
// quant-group-dimension packed I8; cold V is dimension-major I8.  Their F16
// scales are group-major over tokens.  Hot K keeps the accepted block16 XMX
// packing and hot V is dimension-major F16.  The accepted component uses
// group 32; admitted refinements use K4/V4 or K2/V4.  Physical K packing is
// always four dimensions per word and is deliberately independent from the
// scale group.  No path reconstructs a scalar local-F32 K/V tile.

#define IQ36_HEAD_DIM 256U
#define IQ36_Q_HEADS 16U
#define IQ36_KV_HEADS 2U
#define IQ36_GQA_GROUP 8U
#ifndef IQ36_CONTEXT_TOKENS
#define IQ36_CONTEXT_TOKENS 32768U
#endif
#ifndef IQ36_HOT_TOKENS
#define IQ36_HOT_TOKENS 8192U
#endif
#define IQ36_COLD_TOKENS (IQ36_CONTEXT_TOKENS - IQ36_HOT_TOKENS)
#ifndef IQ36_QUANT_GROUP
#define IQ36_QUANT_GROUP 32
#endif
#ifndef IQ36_KEY_QUANT_GROUP
#define IQ36_KEY_QUANT_GROUP IQ36_QUANT_GROUP
#endif
#ifndef IQ36_VALUE_QUANT_GROUP
#define IQ36_VALUE_QUANT_GROUP IQ36_QUANT_GROUP
#endif
#if IQ36_KEY_QUANT_GROUP != 32 && IQ36_KEY_QUANT_GROUP != 4 && \
    IQ36_KEY_QUANT_GROUP != 2
#error "IQ36_KEY_QUANT_GROUP must be 32, 4, or 2"
#endif
#if IQ36_VALUE_QUANT_GROUP != 32 && IQ36_VALUE_QUANT_GROUP != 4
#error "IQ36_VALUE_QUANT_GROUP must be 32 or 4"
#endif
#define IQ36_KEY_SCALE_GROUPS (IQ36_HEAD_DIM / IQ36_KEY_QUANT_GROUP)
#define IQ36_VALUE_SCALE_GROUPS (IQ36_HEAD_DIM / IQ36_VALUE_QUANT_GROUP)
#define IQ36_KEY_PACK_WORDS (IQ36_HEAD_DIM / 4U)
#define IQ36_TOKEN_TILE 16U
#define IQ36_CHUNK_TOKENS 512U
#define IQ36_CHUNK_COUNT (IQ36_CONTEXT_TOKENS / IQ36_CHUNK_TOKENS)
#define IQ36_COLD_CHUNK_COUNT (IQ36_COLD_TOKENS / IQ36_CHUNK_TOKENS)
#define IQ36_HOT_CHUNK_COUNT (IQ36_HOT_TOKENS / IQ36_CHUNK_TOKENS)
#define IQ36_BLOCKS_PER_CHUNK 32U
#define IQ36_HOT_K_WORDS_PER_HEAD \
    (IQ36_HOT_TOKENS * IQ36_HEAD_DIM / 2U)
#define IQ36_COLD_K_WORDS_PER_HEAD \
    (IQ36_COLD_TOKENS * IQ36_HEAD_DIM / 4U)
#define IQ36_EXP2_SCALE (0.0625f * 1.4426950408889634f)

#if defined(IQ36_ADAPTIVE_ATTENTION)
#ifndef IQ36_ADAPTIVE_TOPK
#error "IQ36_ADAPTIVE_TOPK is required for adaptive attention"
#endif
#if IQ36_CONTEXT_TOKENS != 32768U && IQ36_CONTEXT_TOKENS != 65536U
#error "adaptive attention admits only matched 32768/65536 contexts"
#endif
#if IQ36_HOT_TOKENS != 16384U
#error "adaptive attention hot history is exactly 16384 tokens"
#endif
#if IQ36_QUANT_GROUP != 32 || IQ36_KEY_QUANT_GROUP != 32 || \
    IQ36_VALUE_QUANT_GROUP != 32
#error "adaptive attention admits exactly block32 I8 K/V"
#endif
#if IQ36_ADAPTIVE_TOPK != 256U && IQ36_ADAPTIVE_TOPK != 512U
#error "adaptive attention top-k must be 256 or 512"
#endif
#define IQ36_ADAPTIVE_LOCAL_TOPK 64U
#define IQ36_ADAPTIVE_CANDIDATE_COUNT \
    (IQ36_COLD_CHUNK_COUNT * IQ36_ADAPTIVE_LOCAL_TOPK)
#define IQ36_ADAPTIVE_MAX_UNION (IQ36_GQA_GROUP * IQ36_ADAPTIVE_TOPK)
#define IQ36_ADAPTIVE_UNION_WORDS ((IQ36_COLD_TOKENS + 31U) / 32U)
#define IQ36_ADAPTIVE_CORRECTION_PARTITIONS IQ36_COLD_CHUNK_COUNT
#define IQ36_ADAPTIVE_PARTITION_TOKENS IQ36_CHUNK_TOKENS
#else
#if IQ36_CONTEXT_TOKENS != 32768U
#error "the admitted component context is exactly 32768 tokens"
#endif
#endif
#if IQ36_HOT_TOKENS != 8192U && IQ36_HOT_TOKENS != 16384U
#error "IQ36_HOT_TOKENS must be 8192 or the admitted 16384 refinement"
#endif
#if IQ36_CONTEXT_TOKENS % IQ36_CHUNK_TOKENS != 0U || \
    IQ36_HOT_TOKENS % IQ36_CHUNK_TOKENS != 0U
#error "context and hot windows must be whole 512-token chunks"
#endif

// The default mixed entrypoint preserves the promoted component.  The two
// storage-class modes are compiled as separate programs: each owns only its
// live state arguments and maps its compact launch groups back to the original
// partial-workspace group index.  This changes neither chunk nor workspace
// shape and lets the compiler delete the inactive codec before any GPU trial.
#define IQ36_PARTIAL_STORAGE_MIXED 0
#define IQ36_PARTIAL_STORAGE_COLD 1
#define IQ36_PARTIAL_STORAGE_HOT 2
#ifndef IQ36_PARTIAL_STORAGE_CLASS
#define IQ36_PARTIAL_STORAGE_CLASS IQ36_PARTIAL_STORAGE_MIXED
#endif
#if IQ36_PARTIAL_STORAGE_CLASS == IQ36_PARTIAL_STORAGE_MIXED
#define IQ36_PARTIAL_KERNEL_NAME iq36_direct_i8_hotcold_partial
#elif IQ36_PARTIAL_STORAGE_CLASS == IQ36_PARTIAL_STORAGE_COLD
#define IQ36_PARTIAL_KERNEL_NAME iq36_direct_i8_cold_partial
#elif IQ36_PARTIAL_STORAGE_CLASS == IQ36_PARTIAL_STORAGE_HOT
#define IQ36_PARTIAL_KERNEL_NAME iq36_direct_f16_hot_partial
#else
#error "IQ36_PARTIAL_STORAGE_CLASS must be mixed, cold, or hot"
#endif
#if defined(IQ36_ADAPTIVE_ATTENTION) && \
    IQ36_PARTIAL_STORAGE_CLASS != IQ36_PARTIAL_STORAGE_MIXED
#error "adaptive attention uses the one mixed scan carrier"
#endif

ushort8 __builtin_IB_subgroup_block_read_flat_u16_m8k16v1(
    long, int, int, int, int2);

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
  return as_half8(__builtin_IB_subgroup_block_read_flat_u16_m8k16v1(
      (long)address, width_bytes + (int)prefix - 1, height - 1,
      pitch_bytes - 1, (int2)(x, y)));
}

#if defined(IQ36_ADAPTIVE_ATTENTION)
inline ushort iq36_adaptive_ordered_half(const ushort bits) {
  return (bits & 0x8000U) != 0U
      ? (ushort)~bits : (ushort)(bits ^ 0x8000U);
}

inline bool iq36_adaptive_record_better(const uint left, const uint right) {
  const ushort left_score = iq36_adaptive_ordered_half(
      (ushort)(left >> 16U));
  const ushort right_score = iq36_adaptive_ordered_half(
      (ushort)(right >> 16U));
  return left_score != right_score
      ? left_score > right_score
      : (ushort)left < (ushort)right;
}

inline half iq36_adaptive_cold_key(
    __global const uint* cold_k,
    __global const half* cold_k_scales,
    const uint kv_head,
    const uint token,
    const uint dim) {
  const uint token_block = token / IQ36_TOKEN_TILE;
  const uint token_lane = token & (IQ36_TOKEN_TILE - 1U);
  const ulong word =
      ((ulong)token_block * IQ36_KEY_PACK_WORDS + (dim >> 2U)) *
          IQ36_TOKEN_TILE + token_lane;
  const __global char* bytes = (__global const char*)(
      cold_k + (ulong)kv_head * IQ36_COLD_K_WORDS_PER_HEAD);
  const char quantized = bytes[word * 4U + (dim & 3U)];
  const half scale = cold_k_scales[
      ((ulong)kv_head * IQ36_KEY_SCALE_GROUPS +
       dim / IQ36_KEY_QUANT_GROUP) * IQ36_COLD_TOKENS + token];
  return convert_half(quantized) * scale;
}

inline half iq36_adaptive_cold_value(
    __global const char* cold_v,
    __global const half* cold_v_scales,
    const uint kv_head,
    const uint token,
    const uint dim) {
  const char quantized = cold_v[
      ((ulong)kv_head * IQ36_HEAD_DIM + dim) * IQ36_COLD_TOKENS + token];
  const half scale = cold_v_scales[
      ((ulong)kv_head * IQ36_VALUE_SCALE_GROUPS +
       dim / IQ36_VALUE_QUANT_GROUP) * IQ36_COLD_TOKENS + token];
  return convert_half(quantized) * scale;
}

inline half iq36_adaptive_hot_key(
    __global const uint* hot_k,
    const uint kv_head,
    const uint token,
    const uint dim) {
  const uint token_block = token / IQ36_TOKEN_TILE;
  const uint token_lane = token & (IQ36_TOKEN_TILE - 1U);
  const ulong word =
      ((ulong)token_block * (IQ36_HEAD_DIM / 2U) + (dim >> 1U)) *
          IQ36_TOKEN_TILE + token_lane;
  const uint packed = hot_k[
      (ulong)kv_head * IQ36_HOT_K_WORDS_PER_HEAD + word];
  return as_half((ushort)(dim & 1U ? packed >> 16U : packed & 0xffffU));
}

inline half iq36_adaptive_hot_value(
    __global const half* hot_v,
    const uint kv_head,
    const uint token,
    const uint dim) {
  return hot_v[
      ((ulong)kv_head * IQ36_HEAD_DIM + dim) * IQ36_HOT_TOKENS + token];
}
#endif

__attribute__((reqd_work_group_size(32, 1, 1)))
__attribute__((intel_reqd_sub_group_size(32)))
__kernel void iq36_direct_i8_update_state(
    __global const half* evicted_k,
    __global const half* evicted_v,
    __global const half* current_k,
    __global const half* current_v,
    __global char* cold_k,
    __global char* cold_v,
    __global half* cold_k_scales,
    __global half* cold_v_scales,
    __global half* hot_k,
    __global half* hot_v
#if defined(IQ36_ADAPTIVE_ATTENTION)
    , __global half* exact_cold_k,
    __global half* exact_cold_v
#endif
    ) {
  const uint lane = (uint)get_sub_group_local_id();
  const uint group = (uint)get_group_id(0);
  const uint groups_per_head =
      IQ36_KEY_SCALE_GROUPS + IQ36_VALUE_SCALE_GROUPS;
  const uint kv_head = group / groups_per_head;
  const uint within_head = group - kv_head * groups_per_head;
  const uint tensor = within_head >= IQ36_KEY_SCALE_GROUPS;
  const uint scale_group = tensor == 0U
      ? within_head : within_head - IQ36_KEY_SCALE_GROUPS;
  const uint quant_group = tensor == 0U
      ? IQ36_KEY_QUANT_GROUP : IQ36_VALUE_QUANT_GROUP;
  const uint dim = scale_group * quant_group + lane;
  const uint input_index = kv_head * IQ36_HEAD_DIM + dim;
  const bool active = lane < quant_group;
  const half evicted = active
      ? (tensor == 0U ? evicted_k[input_index] : evicted_v[input_index])
      : (half)0.0f;
  const float maximum = sub_group_reduce_max(fabs(convert_float(evicted)));
  const float scale = maximum == 0.0f ? 1.0f : maximum / 127.0f;
  const char quantized = convert_char_sat_rte(convert_float(evicted) / scale);
  const uint cold_token = IQ36_COLD_TOKENS - 1U;

  if (active && tensor == 0U) {
    const uint token_block = cold_token / IQ36_TOKEN_TILE;
    const uint token_lane = cold_token & (IQ36_TOKEN_TILE - 1U);
    const ulong cold_word =
        ((ulong)token_block * IQ36_KEY_PACK_WORDS + (dim >> 2U)) *
            IQ36_TOKEN_TILE + token_lane;
    cold_k[((ulong)kv_head * IQ36_COLD_K_WORDS_PER_HEAD + cold_word) * 4U +
        (dim & 3U)] = quantized;
    const uint hot_token = IQ36_HOT_TOKENS - 1U;
    const uint hot_block = hot_token / IQ36_TOKEN_TILE;
    const uint hot_lane = hot_token & (IQ36_TOKEN_TILE - 1U);
    const ulong hot_word =
        ((ulong)hot_block * (IQ36_HEAD_DIM / 2U) + (dim >> 1U)) *
            IQ36_TOKEN_TILE + hot_lane;
    hot_k[((ulong)kv_head * IQ36_HOT_K_WORDS_PER_HEAD + hot_word) * 2U +
        (dim & 1U)] = current_k[input_index];
  } else if (active) {
    cold_v[((ulong)kv_head * IQ36_HEAD_DIM + dim) * IQ36_COLD_TOKENS +
        cold_token] = quantized;
    hot_v[((ulong)kv_head * IQ36_HEAD_DIM + dim) * IQ36_HOT_TOKENS +
        (IQ36_HOT_TOKENS - 1U)] = current_v[input_index];
  }
  if (lane == 0U) {
    __global half* scales = tensor == 0U ? cold_k_scales : cold_v_scales;
    const uint scale_groups = tensor == 0U
        ? IQ36_KEY_SCALE_GROUPS : IQ36_VALUE_SCALE_GROUPS;
    scales[((ulong)kv_head * scale_groups + scale_group) *
        IQ36_COLD_TOKENS + cold_token] = convert_half_rte(scale);
  }
#if defined(IQ36_ADAPTIVE_ATTENTION)
  if (active) {
    __global half* exact = tensor == 0U ? exact_cold_k : exact_cold_v;
    exact[((ulong)kv_head * IQ36_COLD_TOKENS + cold_token) *
        IQ36_HEAD_DIM + dim] = evicted;
  }
#endif
}

__attribute__((reqd_work_group_size(128, 1, 1)))
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void IQ36_PARTIAL_KERNEL_NAME(
    __global const half* query,
#if IQ36_PARTIAL_STORAGE_CLASS != IQ36_PARTIAL_STORAGE_HOT
    __global const uint* cold_k,
    __global const char* cold_v,
    __global const half* cold_k_scales,
    __global const half* cold_v_scales,
#endif
#if IQ36_PARTIAL_STORAGE_CLASS != IQ36_PARTIAL_STORAGE_COLD
    __global const uint* hot_k,
    __global const half* hot_v,
#endif
    __global float* partial_max,
    __global float* partial_sum,
    __global float* partial_output
#if defined(IQ36_ADAPTIVE_ATTENTION)
    , __global half* approximate_cold_score,
    __global uint* local_candidates,
    __global uint* union_bits,
    __global uint* correction_completion
#endif
    ) {
  const uint local_id = (uint)get_local_id(0);
  const uint subgroup = (uint)get_sub_group_id();
  const uint lane = (uint)get_sub_group_local_id();
#if IQ36_PARTIAL_STORAGE_CLASS == IQ36_PARTIAL_STORAGE_MIXED
  const uint group = (uint)get_group_id(0);
  const uint kv_head = group / IQ36_CHUNK_COUNT;
  const uint chunk = group - kv_head * IQ36_CHUNK_COUNT;
#elif IQ36_PARTIAL_STORAGE_CLASS == IQ36_PARTIAL_STORAGE_COLD
  const uint launch_group = (uint)get_group_id(0);
  const uint kv_head = launch_group / IQ36_COLD_CHUNK_COUNT;
  const uint chunk = launch_group - kv_head * IQ36_COLD_CHUNK_COUNT;
  const uint group = kv_head * IQ36_CHUNK_COUNT + chunk;
#else
  const uint launch_group = (uint)get_group_id(0);
  const uint kv_head = launch_group / IQ36_HOT_CHUNK_COUNT;
  const uint hot_chunk = launch_group - kv_head * IQ36_HOT_CHUNK_COUNT;
  const uint chunk = IQ36_COLD_CHUNK_COUNT + hot_chunk;
  const uint group = kv_head * IQ36_CHUNK_COUNT + chunk;
#endif
  const uint chunk_begin = chunk * IQ36_CHUNK_TOKENS;
#if IQ36_PARTIAL_STORAGE_CLASS == IQ36_PARTIAL_STORAGE_MIXED
  const bool cold_chunk = chunk_begin < IQ36_COLD_TOKENS;
#endif

  __local float score_weight[IQ36_GQA_GROUP * IQ36_CHUNK_TOKENS];
  __local float block_max[IQ36_BLOCKS_PER_CHUNK * IQ36_GQA_GROUP];
  __local float block_sum[IQ36_BLOCKS_PER_CHUNK * IQ36_GQA_GROUP];
  __local float global_max[IQ36_GQA_GROUP];
  __local float global_sum[IQ36_GQA_GROUP];
#if defined(IQ36_ADAPTIVE_ATTENTION)
  __local uint selection_records[
      IQ36_GQA_GROUP * IQ36_ADAPTIVE_LOCAL_TOPK];
  if (group == 0U) {
    for (uint index = local_id;
         index < IQ36_KV_HEADS * IQ36_ADAPTIVE_UNION_WORDS;
         index += 128U) {
      union_bits[index] = 0U;
    }
    if (local_id < IQ36_KV_HEADS) {
      correction_completion[local_id] = 0U;
    }
  }
#endif

  float8 output0 = (float8)(0.0f);
  float8 output1 = (float8)(0.0f);
  const __global half* query_base =
      query + (ulong)kv_head * IQ36_GQA_GROUP * IQ36_HEAD_DIM;

  for (uint block = subgroup; block < IQ36_BLOCKS_PER_CHUNK; block += 8U) {
    const uint block_token = chunk_begin + block * IQ36_TOKEN_TILE;
    float8 score = (float8)(0.0f);
#if IQ36_PARTIAL_STORAGE_CLASS == IQ36_PARTIAL_STORAGE_MIXED
    if (cold_chunk) {
#endif
#if IQ36_PARTIAL_STORAGE_CLASS != IQ36_PARTIAL_STORAGE_HOT
      const uint token_block = block_token / IQ36_TOKEN_TILE;
      const uint token = block_token + lane;
      #if IQ36_KEY_QUANT_GROUP == 32
      #pragma unroll 1
      for (uint scale_group = 0U; scale_group < IQ36_KEY_SCALE_GROUPS;
           ++scale_group) {
        const ulong word =
            ((ulong)token_block * IQ36_KEY_PACK_WORDS + scale_group * 8U) *
                IQ36_TOKEN_TILE;
        const uint8 packed = intel_sub_group_block_read8(
            cold_k + (ulong)kv_head * IQ36_COLD_K_WORDS_PER_HEAD + word);
        const half scale = cold_k_scales[
            ((ulong)kv_head * IQ36_KEY_SCALE_GROUPS + scale_group) *
                IQ36_COLD_TOKENS + token];
        const half16 key0 = convert_half16(as_char16(packed.s0123)) * scale;
        const half16 key1 = convert_half16(as_char16(packed.s4567)) * scale;
        const uint k_base = scale_group * IQ36_KEY_QUANT_GROUP;
        const short8 query0 = as_short8(iq36_block2d_load_f16_16x8(
            query_base, IQ36_HEAD_DIM * (int)sizeof(half),
            IQ36_GQA_GROUP, IQ36_HEAD_DIM * (int)sizeof(half),
            k_base, 0));
        const short8 query1 = as_short8(iq36_block2d_load_f16_16x8(
            query_base, IQ36_HEAD_DIM * (int)sizeof(half),
            IQ36_GQA_GROUP, IQ36_HEAD_DIM * (int)sizeof(half),
            k_base + 16U, 0));
        score = intel_sub_group_f16_f16_matrix_mad_k16(
            query0, as_int8(key0), score);
        score = intel_sub_group_f16_f16_matrix_mad_k16(
            query1, as_int8(key1), score);
      }
      #else
      #pragma unroll 1
      for (uint k_block = 0U; k_block < IQ36_HEAD_DIM / 16U; ++k_block) {
        const uint scale_group =
            k_block * (16U / IQ36_KEY_QUANT_GROUP);
        const ulong word =
            ((ulong)token_block * IQ36_KEY_PACK_WORDS + k_block * 4U) *
                IQ36_TOKEN_TILE;
        const __global uint* cold_head =
            cold_k + (ulong)kv_head * IQ36_COLD_K_WORDS_PER_HEAD;
        const uint packed0 = intel_sub_group_block_read(
            cold_head + word + 0U * IQ36_TOKEN_TILE);
        const uint packed1 = intel_sub_group_block_read(
            cold_head + word + 1U * IQ36_TOKEN_TILE);
        const uint packed2 = intel_sub_group_block_read(
            cold_head + word + 2U * IQ36_TOKEN_TILE);
        const uint packed3 = intel_sub_group_block_read(
            cold_head + word + 3U * IQ36_TOKEN_TILE);
        const char16 quantized = (char16)(
            as_char4(packed0), as_char4(packed1),
            as_char4(packed2), as_char4(packed3));
        #define IQ36_LOAD_KEY_SCALE(offset) cold_k_scales[ \
            ((ulong)kv_head * IQ36_KEY_SCALE_GROUPS + scale_group + \
             (offset)) * IQ36_COLD_TOKENS + token]
        #if IQ36_KEY_QUANT_GROUP == 4
        const half scale0 = IQ36_LOAD_KEY_SCALE(0U);
        const half scale1 = IQ36_LOAD_KEY_SCALE(1U);
        const half scale2 = IQ36_LOAD_KEY_SCALE(2U);
        const half scale3 = IQ36_LOAD_KEY_SCALE(3U);
        const half16 scales = (half16)(
            scale0, scale0, scale0, scale0,
            scale1, scale1, scale1, scale1,
            scale2, scale2, scale2, scale2,
            scale3, scale3, scale3, scale3);
        #else
        const half scale0 = IQ36_LOAD_KEY_SCALE(0U);
        const half scale1 = IQ36_LOAD_KEY_SCALE(1U);
        const half scale2 = IQ36_LOAD_KEY_SCALE(2U);
        const half scale3 = IQ36_LOAD_KEY_SCALE(3U);
        const half scale4 = IQ36_LOAD_KEY_SCALE(4U);
        const half scale5 = IQ36_LOAD_KEY_SCALE(5U);
        const half scale6 = IQ36_LOAD_KEY_SCALE(6U);
        const half scale7 = IQ36_LOAD_KEY_SCALE(7U);
        const half16 scales = (half16)(
            scale0, scale0, scale1, scale1,
            scale2, scale2, scale3, scale3,
            scale4, scale4, scale5, scale5,
            scale6, scale6, scale7, scale7);
        #endif
        #undef IQ36_LOAD_KEY_SCALE
        const half16 key = convert_half16(quantized) * scales;
        const short8 query_fragment = as_short8(
            iq36_block2d_load_f16_16x8(
                query_base, IQ36_HEAD_DIM * (int)sizeof(half),
                IQ36_GQA_GROUP, IQ36_HEAD_DIM * (int)sizeof(half),
                k_block * 16U, 0));
        score = intel_sub_group_f16_f16_matrix_mad_k16(
            query_fragment, as_int8(key), score);
      }
      #endif
#endif
#if IQ36_PARTIAL_STORAGE_CLASS == IQ36_PARTIAL_STORAGE_MIXED
    } else {
#endif
#if IQ36_PARTIAL_STORAGE_CLASS != IQ36_PARTIAL_STORAGE_COLD
      const uint hot_token = block_token - IQ36_COLD_TOKENS;
      const uint token_block = hot_token / IQ36_TOKEN_TILE;
      #pragma unroll 1
      for (uint k_block = 0U; k_block < IQ36_HEAD_DIM / 16U; ++k_block) {
        const ulong word =
            ((ulong)token_block * (IQ36_HEAD_DIM / 2U) + k_block * 8U) *
            IQ36_TOKEN_TILE;
        const int8 key = as_int8(intel_sub_group_block_read8(
            hot_k + (ulong)kv_head * IQ36_HOT_K_WORDS_PER_HEAD + word));
        const short8 query_fragment = as_short8(iq36_block2d_load_f16_16x8(
            query_base, IQ36_HEAD_DIM * (int)sizeof(half),
            IQ36_GQA_GROUP, IQ36_HEAD_DIM * (int)sizeof(half),
            k_block * 16U, 0));
        score = intel_sub_group_f16_f16_matrix_mad_k16(
            query_fragment, key, score);
      }
#endif
#if IQ36_PARTIAL_STORAGE_CLASS == IQ36_PARTIAL_STORAGE_MIXED
    }
#endif
    #pragma unroll
    for (uint head = 0U; head < IQ36_GQA_GROUP; ++head) {
      score_weight[head * IQ36_CHUNK_TOKENS +
          block * IQ36_TOKEN_TILE + lane] = score[head];
#if defined(IQ36_ADAPTIVE_ATTENTION)
      if (cold_chunk) {
        const uint q_head = kv_head * IQ36_GQA_GROUP + head;
        approximate_cold_score[
            (ulong)q_head * IQ36_COLD_TOKENS + block_token + lane] =
            convert_half_rte(score[head] * 0.0625f);
      }
#endif
      const float maximum = sub_group_reduce_max(score[head]);
      if (lane == 0U) {
        block_max[block * IQ36_GQA_GROUP + head] = maximum;
      }
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);

#if defined(IQ36_ADAPTIVE_ATTENTION)
  if (cold_chunk) {
    ushort prefix = 0U;
    ushort processed_mask = 0U;
    uint rank = IQ36_ADAPTIVE_LOCAL_TOPK;
    for (int shift = 12; shift >= 0; shift -= 4) {
      uint lane_counts[16];
      #pragma unroll
      for (uint digit = 0U; digit < 16U; ++digit) {
        lane_counts[digit] = 0U;
      }
      for (uint token = lane; token < IQ36_CHUNK_TOKENS; token += 16U) {
        const ushort score_bits = as_ushort(convert_half_rte(
            score_weight[subgroup * IQ36_CHUNK_TOKENS + token] * 0.0625f));
        const ushort ordered = iq36_adaptive_ordered_half(score_bits);
        if ((ordered & processed_mask) == prefix) {
          ++lane_counts[(ordered >> shift) & 15U];
        }
      }
      ushort selected_digit = 0U;
      bool digit_found = false;
      #pragma unroll
      for (int digit = 15; digit >= 0; --digit) {
        const uint count = sub_group_reduce_add(lane_counts[digit]);
        if (!digit_found) {
          if (count >= rank) {
            selected_digit = (ushort)digit;
            digit_found = true;
          } else {
            rank -= count;
          }
        }
      }
      prefix = (ushort)(prefix | (selected_digit << shift));
      processed_mask = (ushort)(processed_mask | (15U << shift));
    }
    if (lane == 0U) {
      const uint head = subgroup;
      const uint q_head = kv_head * IQ36_GQA_GROUP + head;
      const ulong candidate_base =
          ((ulong)q_head * IQ36_COLD_CHUNK_COUNT + chunk) *
          IQ36_ADAPTIVE_LOCAL_TOPK;
      uint selected = 0U;
      for (uint token = 0U; token < IQ36_CHUNK_TOKENS; ++token) {
        const ushort score_bits = as_ushort(convert_half_rte(
            score_weight[head * IQ36_CHUNK_TOKENS + token] * 0.0625f));
        if (iq36_adaptive_ordered_half(score_bits) > prefix) {
          selection_records[head * IQ36_ADAPTIVE_LOCAL_TOPK + selected++] =
              ((uint)score_bits << 16U) | (chunk_begin + token);
        }
      }
      for (uint token = 0U;
           token < IQ36_CHUNK_TOKENS &&
           selected < IQ36_ADAPTIVE_LOCAL_TOPK; ++token) {
        const ushort score_bits = as_ushort(convert_half_rte(
            score_weight[head * IQ36_CHUNK_TOKENS + token] * 0.0625f));
        if (iq36_adaptive_ordered_half(score_bits) == prefix) {
          selection_records[head * IQ36_ADAPTIVE_LOCAL_TOPK + selected++] =
              ((uint)score_bits << 16U) | (chunk_begin + token);
        }
      }
      __local uint* records =
          selection_records + head * IQ36_ADAPTIVE_LOCAL_TOPK;
      for (int parent = (int)(IQ36_ADAPTIVE_LOCAL_TOPK / 2U) - 1;
           parent >= 0; --parent) {
        uint position = (uint)parent;
        const uint value = records[position];
        while (position * 2U + 1U < IQ36_ADAPTIVE_LOCAL_TOPK) {
          uint child = position * 2U + 1U;
          if (child + 1U < IQ36_ADAPTIVE_LOCAL_TOPK &&
              iq36_adaptive_record_better(
                  records[child + 1U], records[child])) {
            ++child;
          }
          if (!iq36_adaptive_record_better(records[child], value)) {
            break;
          }
          records[position] = records[child];
          position = child;
        }
        records[position] = value;
      }
      for (uint index = 0U; index < IQ36_ADAPTIVE_LOCAL_TOPK; ++index) {
        local_candidates[candidate_base + index] = records[index];
      }
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
#endif

  if (subgroup == 0U && lane < IQ36_GQA_GROUP) {
    float maximum = -INFINITY;
    #pragma unroll
    for (uint block = 0U; block < IQ36_BLOCKS_PER_CHUNK; ++block) {
      maximum = fmax(maximum,
          block_max[block * IQ36_GQA_GROUP + lane]);
    }
    global_max[lane] = maximum;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  for (uint block = subgroup; block < IQ36_BLOCKS_PER_CHUNK; block += 8U) {
    const uint token_in_chunk = block * IQ36_TOKEN_TILE + lane;
    #pragma unroll
    for (uint head = 0U; head < IQ36_GQA_GROUP; ++head) {
      const float weight = native_exp2(
          (score_weight[head * IQ36_CHUNK_TOKENS + token_in_chunk] -
           global_max[head]) * IQ36_EXP2_SCALE);
      score_weight[head * IQ36_CHUNK_TOKENS + token_in_chunk] = weight;
      const float sum = sub_group_reduce_add(weight);
      if (lane == 0U) {
        block_sum[block * IQ36_GQA_GROUP + head] = sum;
      }
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  if (subgroup == 0U && lane < IQ36_GQA_GROUP) {
    float sum = 0.0f;
    #pragma unroll
    for (uint block = 0U; block < IQ36_BLOCKS_PER_CHUNK; ++block) {
      sum += block_sum[block * IQ36_GQA_GROUP + lane];
    }
    global_sum[lane] = sum;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  const uint dim0 = subgroup * 16U + lane;
  const uint dim1 = 128U + dim0;
  #pragma unroll 1
  for (uint block = 0U; block < IQ36_BLOCKS_PER_CHUNK; ++block) {
    const uint token_in_chunk = block * IQ36_TOKEN_TILE + lane;
    const uint block_token = chunk_begin + block * IQ36_TOKEN_TILE;
    const short8 weights = (short8)(
        as_short(convert_half_rte(score_weight[
            0U * IQ36_CHUNK_TOKENS + token_in_chunk])),
        as_short(convert_half_rte(score_weight[
            1U * IQ36_CHUNK_TOKENS + token_in_chunk])),
        as_short(convert_half_rte(score_weight[
            2U * IQ36_CHUNK_TOKENS + token_in_chunk])),
        as_short(convert_half_rte(score_weight[
            3U * IQ36_CHUNK_TOKENS + token_in_chunk])),
        as_short(convert_half_rte(score_weight[
            4U * IQ36_CHUNK_TOKENS + token_in_chunk])),
        as_short(convert_half_rte(score_weight[
            5U * IQ36_CHUNK_TOKENS + token_in_chunk])),
        as_short(convert_half_rte(score_weight[
            6U * IQ36_CHUNK_TOKENS + token_in_chunk])),
        as_short(convert_half_rte(score_weight[
            7U * IQ36_CHUNK_TOKENS + token_in_chunk])));
    int8 value0;
    int8 value1;
#if IQ36_PARTIAL_STORAGE_CLASS == IQ36_PARTIAL_STORAGE_MIXED
    if (cold_chunk) {
#endif
#if IQ36_PARTIAL_STORAGE_CLASS != IQ36_PARTIAL_STORAGE_HOT
      const half16 scales0 = vload16(0, cold_v_scales +
          ((ulong)kv_head * IQ36_VALUE_SCALE_GROUPS +
           (dim0 / IQ36_VALUE_QUANT_GROUP)) *
              IQ36_COLD_TOKENS + block_token);
      const half16 scales1 = vload16(0, cold_v_scales +
          ((ulong)kv_head * IQ36_VALUE_SCALE_GROUPS +
           (dim1 / IQ36_VALUE_QUANT_GROUP)) *
              IQ36_COLD_TOKENS + block_token);
      const char16 quantized0 = vload16(0, cold_v +
          ((ulong)kv_head * IQ36_HEAD_DIM + dim0) * IQ36_COLD_TOKENS +
              block_token);
      const char16 quantized1 = vload16(0, cold_v +
          ((ulong)kv_head * IQ36_HEAD_DIM + dim1) * IQ36_COLD_TOKENS +
              block_token);
      value0 = as_int8(convert_half16(quantized0) * scales0);
      value1 = as_int8(convert_half16(quantized1) * scales1);
#endif
#if IQ36_PARTIAL_STORAGE_CLASS == IQ36_PARTIAL_STORAGE_MIXED
    } else {
#endif
#if IQ36_PARTIAL_STORAGE_CLASS != IQ36_PARTIAL_STORAGE_COLD
      const uint hot_token = block_token - IQ36_COLD_TOKENS;
      value0 = as_int8(vload16(0, hot_v +
          ((ulong)kv_head * IQ36_HEAD_DIM + dim0) * IQ36_HOT_TOKENS +
              hot_token));
      value1 = as_int8(vload16(0, hot_v +
          ((ulong)kv_head * IQ36_HEAD_DIM + dim1) * IQ36_HOT_TOKENS +
              hot_token));
#endif
#if IQ36_PARTIAL_STORAGE_CLASS == IQ36_PARTIAL_STORAGE_MIXED
    }
#endif
    output0 = intel_sub_group_f16_f16_matrix_mad_k16(
        weights, value0, output0);
    output1 = intel_sub_group_f16_f16_matrix_mad_k16(
        weights, value1, output1);
  }

  const ulong meta_base =
      ((ulong)group * IQ36_GQA_GROUP) * IQ36_HEAD_DIM;
  if (subgroup == 0U && lane < IQ36_GQA_GROUP) {
    partial_max[(ulong)group * IQ36_GQA_GROUP + lane] = global_max[lane];
    partial_sum[(ulong)group * IQ36_GQA_GROUP + lane] = global_sum[lane];
  }
  #pragma unroll
  for (uint head = 0U; head < IQ36_GQA_GROUP; ++head) {
    const ulong head_base = meta_base + (ulong)head * IQ36_HEAD_DIM;
    partial_output[head_base + dim0] = output0[head];
    partial_output[head_base + dim1] = output1[head];
  }
}

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void iq36_direct_i8_hotcold_reduce(
    __global const float* partial_max,
    __global const float* partial_sum,
    __global const float* partial_output,
    __global float* output) {
  const uint q_head = (uint)get_group_id(0);
  const uint dim = (uint)get_local_id(0);
  const uint kv_head = q_head / IQ36_GQA_GROUP;
  const uint head_in_kv = q_head - kv_head * IQ36_GQA_GROUP;
  float running_max = -INFINITY;
  float running_sum = 0.0f;
  float accumulated = 0.0f;
  for (uint chunk = 0U; chunk < IQ36_CHUNK_COUNT; ++chunk) {
    const uint group = kv_head * IQ36_CHUNK_COUNT + chunk;
    const ulong meta = (ulong)group * IQ36_GQA_GROUP + head_in_kv;
    const float next_max = fmax(running_max, partial_max[meta]);
    const float previous_scale = native_exp2(
        (running_max - next_max) * IQ36_EXP2_SCALE);
    const float partial_scale = native_exp2(
        (partial_max[meta] - next_max) * IQ36_EXP2_SCALE);
    accumulated = accumulated * previous_scale +
        partial_output[meta * IQ36_HEAD_DIM + dim] * partial_scale;
    running_sum = running_sum * previous_scale +
        partial_sum[meta] * partial_scale;
    running_max = next_max;
  }
  output[q_head * IQ36_HEAD_DIM + dim] =
      running_sum == 0.0f ? 0.0f : accumulated * native_recip(running_sum);
}

#if defined(IQ36_ADAPTIVE_ATTENTION)
__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void iq36_adaptive_select_reduce_union(
    __global const float* partial_max,
    __global const float* partial_sum,
    __global const float* partial_output,
    __global const uint* local_candidates,
    __global uint* union_bits,
    __global float* aggregate_max,
    __global float* aggregate_sum,
    __global float* aggregate_numerator) {
  const uint q_head = (uint)get_group_id(0);
  const uint local_id = (uint)get_local_id(0);
  const uint kv_head = q_head / IQ36_GQA_GROUP;
  const uint head_in_kv = q_head - kv_head * IQ36_GQA_GROUP;
  __local uint candidate_heaps[IQ36_ADAPTIVE_CANDIDATE_COUNT];
  __local ushort heap_chunks[IQ36_COLD_CHUNK_COUNT];
  __local uchar chunk_sizes[IQ36_COLD_CHUNK_COUNT];
  const ulong candidate_base =
      (ulong)q_head * IQ36_ADAPTIVE_CANDIDATE_COUNT;
  for (uint index = local_id; index < IQ36_ADAPTIVE_CANDIDATE_COUNT;
       index += 256U) {
    candidate_heaps[index] = local_candidates[candidate_base + index];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (local_id == 0U) {
    uint heap_size = IQ36_COLD_CHUNK_COUNT;
    for (uint chunk = 0U; chunk < heap_size; ++chunk) {
      heap_chunks[chunk] = (ushort)chunk;
      chunk_sizes[chunk] = (uchar)IQ36_ADAPTIVE_LOCAL_TOPK;
    }
    for (int parent = (int)(heap_size / 2U) - 1; parent >= 0; --parent) {
      uint position = (uint)parent;
      const ushort chunk = heap_chunks[position];
      const uint record = candidate_heaps[
          (uint)chunk * IQ36_ADAPTIVE_LOCAL_TOPK];
      while (position * 2U + 1U < heap_size) {
        uint child = position * 2U + 1U;
        ushort child_chunk = heap_chunks[child];
        uint child_record = candidate_heaps[
            (uint)child_chunk * IQ36_ADAPTIVE_LOCAL_TOPK];
        if (child + 1U < heap_size) {
          const ushort right_chunk = heap_chunks[child + 1U];
          const uint right_record = candidate_heaps[
              (uint)right_chunk * IQ36_ADAPTIVE_LOCAL_TOPK];
          if (iq36_adaptive_record_better(right_record, child_record)) {
            ++child;
            child_chunk = right_chunk;
            child_record = right_record;
          }
        }
        if (!iq36_adaptive_record_better(child_record, record)) {
          break;
        }
        heap_chunks[position] = child_chunk;
        position = child;
      }
      heap_chunks[position] = chunk;
    }
    for (uint selected = 0U;
         selected < IQ36_ADAPTIVE_TOPK && heap_size != 0U; ++selected) {
      const ushort best_chunk = heap_chunks[0];
      const uint chunk_base =
          (uint)best_chunk * IQ36_ADAPTIVE_LOCAL_TOPK;
      const uint record = candidate_heaps[chunk_base];
      const uint token = record & 0xffffU;
      atomic_or((volatile __global unsigned int*)&union_bits[
          (ulong)kv_head * IQ36_ADAPTIVE_UNION_WORDS + (token >> 5U)],
          1U << (token & 31U));

      const uint next_chunk_size = (uint)chunk_sizes[best_chunk] - 1U;
      chunk_sizes[best_chunk] = (uchar)next_chunk_size;
      if (next_chunk_size == 0U) {
        --heap_size;
        heap_chunks[0] = heap_chunks[heap_size];
      } else {
        uint position = 0U;
        const uint replacement =
            candidate_heaps[chunk_base + next_chunk_size];
        while (position * 2U + 1U < next_chunk_size) {
          uint child = position * 2U + 1U;
          if (child + 1U < next_chunk_size &&
              iq36_adaptive_record_better(
                  candidate_heaps[chunk_base + child + 1U],
                  candidate_heaps[chunk_base + child])) {
            ++child;
          }
          if (!iq36_adaptive_record_better(
                  candidate_heaps[chunk_base + child], replacement)) {
            break;
          }
          candidate_heaps[chunk_base + position] =
              candidate_heaps[chunk_base + child];
          position = child;
        }
        candidate_heaps[chunk_base + position] = replacement;
      }
      if (heap_size != 0U) {
        uint position = 0U;
        const ushort chunk = heap_chunks[0];
        const uint replacement = candidate_heaps[
            (uint)chunk * IQ36_ADAPTIVE_LOCAL_TOPK];
        while (position * 2U + 1U < heap_size) {
          uint child = position * 2U + 1U;
          ushort child_chunk = heap_chunks[child];
          uint child_record = candidate_heaps[
              (uint)child_chunk * IQ36_ADAPTIVE_LOCAL_TOPK];
          if (child + 1U < heap_size) {
            const ushort right_chunk = heap_chunks[child + 1U];
            const uint right_record = candidate_heaps[
                (uint)right_chunk * IQ36_ADAPTIVE_LOCAL_TOPK];
            if (iq36_adaptive_record_better(right_record, child_record)) {
              ++child;
              child_chunk = right_chunk;
              child_record = right_record;
            }
          }
          if (!iq36_adaptive_record_better(child_record, replacement)) {
            break;
          }
          heap_chunks[position] = child_chunk;
          position = child;
        }
        heap_chunks[position] = chunk;
      }
    }
  }

  float running_max = -INFINITY;
  float running_sum = 0.0f;
  float accumulated = 0.0f;
  for (uint chunk = 0U; chunk < IQ36_CHUNK_COUNT; ++chunk) {
    const uint group = kv_head * IQ36_CHUNK_COUNT + chunk;
    const ulong meta = (ulong)group * IQ36_GQA_GROUP + head_in_kv;
    const float next_max = fmax(running_max, partial_max[meta]);
    const float previous_scale = native_exp2(
        (running_max - next_max) * IQ36_EXP2_SCALE);
    const float next_scale = native_exp2(
        (partial_max[meta] - next_max) * IQ36_EXP2_SCALE);
    if (local_id < IQ36_HEAD_DIM) {
      accumulated = accumulated * previous_scale +
          partial_output[meta * IQ36_HEAD_DIM + local_id] * next_scale;
    }
    running_sum = running_sum * previous_scale + partial_sum[meta] * next_scale;
    running_max = next_max;
  }
  if (local_id == 0U) {
    aggregate_max[q_head] = running_max;
    aggregate_sum[q_head] = running_sum;
  }
  aggregate_numerator[(ulong)q_head * IQ36_HEAD_DIM + local_id] = accumulated;
}

__attribute__((reqd_work_group_size(128, 1, 1)))
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void iq36_adaptive_correct_normalize(
    __global const half* query,
    __global const uint* cold_k,
    __global const char* cold_v,
    __global const half* cold_k_scales,
    __global const half* cold_v_scales,
    __global const half* exact_cold_k,
    __global const half* exact_cold_v,
    __global const half* approximate_cold_score,
    __global const uint* union_bits,
    __global const float* aggregate_max,
    __global const float* aggregate_sum,
    __global const float* aggregate_numerator,
    __global float* correction_partial_max,
    __global float* correction_partial_sum,
    __global float* correction_partial_numerator,
    __global uint* correction_completion,
    __global float* output) {
  const uint launch_group = (uint)get_group_id(0);
  const uint kv_head = launch_group / IQ36_ADAPTIVE_CORRECTION_PARTITIONS;
  const uint partition = launch_group -
      kv_head * IQ36_ADAPTIVE_CORRECTION_PARTITIONS;
  const uint local_id = (uint)get_local_id(0);
  const uint head = (uint)get_sub_group_id();
  const uint lane = (uint)get_sub_group_local_id();
  const uint q_head = kv_head * IQ36_GQA_GROUP + head;
  const uint partition_begin = partition * IQ36_ADAPTIVE_PARTITION_TOKENS;
  const uint partition_limit =
      partition_begin + IQ36_ADAPTIVE_PARTITION_TOKENS;
  const uint partition_end = partition_limit < (uint)IQ36_COLD_TOKENS
      ? partition_limit : (uint)IQ36_COLD_TOKENS;
  __local ushort selected_indices[IQ36_ADAPTIVE_PARTITION_TOKENS];
  __local uint selected_count;
  __local half data_a[16U * IQ36_HEAD_DIM];
  __local half data_b[16U * IQ36_HEAD_DIM];
  __local float approximate_scores[IQ36_GQA_GROUP * 16U];
  __local float exact_scores[IQ36_GQA_GROUP * 16U];
  __local uint is_last_partition;

  if (local_id == 0U) {
    uint count = 0U;
    for (uint word_index = partition_begin / 32U;
         word_index < (partition_end + 31U) / 32U; ++word_index) {
      const uint word = union_bits[
          (ulong)kv_head * IQ36_ADAPTIVE_UNION_WORDS + word_index];
      for (uint bit = 0U; bit < 32U; ++bit) {
        const uint token = word_index * 32U + bit;
        if (token >= partition_begin && token < partition_end &&
            (word & (1U << bit)) != 0U) {
          selected_indices[count++] = (ushort)token;
        }
      }
    }
    selected_count = count;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  half query_fragment[16];
  float delta_numerator[16];
  #pragma unroll
  for (uint slot = 0U; slot < 16U; ++slot) {
    const uint dim = lane + slot * 16U;
    query_fragment[slot] = query[
        (ulong)q_head * IQ36_HEAD_DIM + dim];
    delta_numerator[slot] = 0.0f;
  }
  float correction_max = -INFINITY;
  float delta_sum = 0.0f;

  for (uint begin = 0U; begin < selected_count; begin += 16U) {
    const uint rows = min(16U, selected_count - begin);
    for (uint index = local_id; index < rows * IQ36_HEAD_DIM;
         index += 128U) {
      const uint row = index / IQ36_HEAD_DIM;
      const uint dim = index - row * IQ36_HEAD_DIM;
      const uint token = (uint)selected_indices[begin + row];
      const ulong exact_index =
          ((ulong)kv_head * IQ36_COLD_TOKENS + token) *
              IQ36_HEAD_DIM + dim;
      data_b[index] = exact_cold_k[exact_index];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint row = 0U; row < rows; ++row) {
      float exact_score = 0.0f;
      #pragma unroll
      for (uint slot = 0U; slot < 16U; ++slot) {
        const uint dim = lane + slot * 16U;
        exact_score = fma(
            convert_float(query_fragment[slot]),
            convert_float(data_b[row * IQ36_HEAD_DIM + dim]), exact_score);
      }
      exact_score = sub_group_reduce_add(exact_score);
      if (lane == 0U) {
        const uint token = (uint)selected_indices[begin + row];
        approximate_scores[head * 16U + row] = convert_float(
            approximate_cold_score[
                (ulong)q_head * IQ36_COLD_TOKENS + token]) * 16.0f;
        exact_scores[head * 16U + row] = exact_score;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint index = local_id; index < rows * IQ36_HEAD_DIM;
         index += 128U) {
      const uint row = index / IQ36_HEAD_DIM;
      const uint dim = index - row * IQ36_HEAD_DIM;
      const uint token = (uint)selected_indices[begin + row];
      data_a[index] = iq36_adaptive_cold_value(
          cold_v, cold_v_scales, kv_head, token, dim);
      const ulong exact_index =
          ((ulong)kv_head * IQ36_COLD_TOKENS + token) *
              IQ36_HEAD_DIM + dim;
      data_b[index] = exact_cold_v[exact_index];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint row = 0U; row < rows; ++row) {
      const float approximate_score = approximate_scores[head * 16U + row];
      const float exact_score = exact_scores[head * 16U + row];
      const float next_max = fmax(
          correction_max, fmax(approximate_score, exact_score));
      const float previous_scale = native_exp2(
          (correction_max - next_max) * IQ36_EXP2_SCALE);
      const float approximate_weight = native_exp2(
          (approximate_score - next_max) * IQ36_EXP2_SCALE);
      const float exact_weight = native_exp2(
          (exact_score - next_max) * IQ36_EXP2_SCALE);
      #pragma unroll
      for (uint slot = 0U; slot < 16U; ++slot) {
        const uint dim = lane + slot * 16U;
        delta_numerator[slot] = delta_numerator[slot] * previous_scale +
            exact_weight * convert_float(
                data_b[row * IQ36_HEAD_DIM + dim]) -
            approximate_weight * convert_float(
                data_a[row * IQ36_HEAD_DIM + dim]);
      }
      delta_sum = delta_sum * previous_scale +
          exact_weight - approximate_weight;
      correction_max = next_max;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  const ulong partial_meta =
      ((ulong)launch_group * IQ36_GQA_GROUP + head);
  if (lane == 0U) {
    correction_partial_max[partial_meta] = correction_max;
    correction_partial_sum[partial_meta] = delta_sum;
  }
  #pragma unroll
  for (uint slot = 0U; slot < 16U; ++slot) {
    const uint dim = lane + slot * 16U;
    correction_partial_numerator[
        partial_meta * IQ36_HEAD_DIM + dim] = delta_numerator[slot];
  }
  barrier(CLK_GLOBAL_MEM_FENCE);
  if (local_id == 0U) {
    atomic_work_item_fence(
        CLK_GLOBAL_MEM_FENCE, memory_order_seq_cst, memory_scope_device);
    const uint ticket = atomic_inc(
        (volatile __global unsigned int*)&correction_completion[kv_head]);
    atomic_work_item_fence(
        CLK_GLOBAL_MEM_FENCE, memory_order_seq_cst, memory_scope_device);
    is_last_partition =
        ticket == IQ36_ADAPTIVE_CORRECTION_PARTITIONS - 1U;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (is_last_partition != 0U) {
    atomic_work_item_fence(
        CLK_GLOBAL_MEM_FENCE, memory_order_seq_cst, memory_scope_device);
    float final_max = aggregate_max[q_head];
    for (uint part = 0U;
         part < IQ36_ADAPTIVE_CORRECTION_PARTITIONS; ++part) {
      const ulong meta =
          ((ulong)kv_head * IQ36_ADAPTIVE_CORRECTION_PARTITIONS + part) *
              IQ36_GQA_GROUP + head;
      final_max = fmax(final_max, correction_partial_max[meta]);
    }
    float final_sum = aggregate_sum[q_head] * native_exp2(
        (aggregate_max[q_head] - final_max) * IQ36_EXP2_SCALE);
    float final_numerator[16];
    #pragma unroll
    for (uint slot = 0U; slot < 16U; ++slot) {
      const uint dim = lane + slot * 16U;
      final_numerator[slot] = aggregate_numerator[
          (ulong)q_head * IQ36_HEAD_DIM + dim] * native_exp2(
              (aggregate_max[q_head] - final_max) * IQ36_EXP2_SCALE);
    }
    for (uint part = 0U;
         part < IQ36_ADAPTIVE_CORRECTION_PARTITIONS; ++part) {
      const ulong meta =
          ((ulong)kv_head * IQ36_ADAPTIVE_CORRECTION_PARTITIONS + part) *
              IQ36_GQA_GROUP + head;
      const float scale = native_exp2(
          (correction_partial_max[meta] - final_max) * IQ36_EXP2_SCALE);
      final_sum += correction_partial_sum[meta] * scale;
      #pragma unroll
      for (uint slot = 0U; slot < 16U; ++slot) {
        const uint dim = lane + slot * 16U;
        final_numerator[slot] += correction_partial_numerator[
            meta * IQ36_HEAD_DIM + dim] * scale;
      }
    }
    #pragma unroll
    for (uint slot = 0U; slot < 16U; ++slot) {
      const uint dim = lane + slot * 16U;
      output[(ulong)q_head * IQ36_HEAD_DIM + dim] =
          final_numerator[slot] * native_recip(final_sum);
    }
  }
}

__kernel void iq36_adaptive_reference_score(
    __global const half* query,
    __global const uint* cold_k,
    __global const half* cold_k_scales,
    __global const uint* hot_k,
    __global const half* exact_key,
    __global float* approximate_score,
    __global float* exact_score) {
  const uint index = (uint)get_global_id(0);
  if (index >= IQ36_Q_HEADS * IQ36_CONTEXT_TOKENS) return;
  const uint q_head = index / IQ36_CONTEXT_TOKENS;
  const uint token = index - q_head * IQ36_CONTEXT_TOKENS;
  const uint kv_head = q_head / IQ36_GQA_GROUP;
  float approximate_dot = 0.0f;
  float exact_dot = 0.0f;
  for (uint dim = 0U; dim < IQ36_HEAD_DIM; ++dim) {
    const float q = convert_float(query[q_head * IQ36_HEAD_DIM + dim]);
    const half approximate = token < IQ36_COLD_TOKENS
        ? iq36_adaptive_cold_key(
            cold_k, cold_k_scales, kv_head, token, dim)
        : iq36_adaptive_hot_key(
            hot_k, kv_head, token - IQ36_COLD_TOKENS, dim);
    approximate_dot = fma(q, convert_float(approximate), approximate_dot);
    exact_dot = fma(q, convert_float(exact_key[
        ((ulong)token * IQ36_KV_HEADS + kv_head) * IQ36_HEAD_DIM + dim]),
        exact_dot);
  }
  approximate_score[index] = approximate_dot;
  exact_score[index] = exact_dot;
}

__kernel void iq36_adaptive_reference_apply(
    __global const float* approximate_score,
    __global const float* exact_score,
    __global const char* cold_v,
    __global const half* cold_v_scales,
    __global const half* hot_v,
    __global const half* exact_value,
    __global const uint* union_bits,
    __global float* adaptive_output,
    __global float* exact_output) {
  const uint index = (uint)get_global_id(0);
  if (index >= IQ36_Q_HEADS * IQ36_HEAD_DIM) return;
  const uint q_head = index / IQ36_HEAD_DIM;
  const uint dim = index - q_head * IQ36_HEAD_DIM;
  const uint kv_head = q_head / IQ36_GQA_GROUP;
  float adaptive_max = -INFINITY;
  float adaptive_sum = 0.0f;
  float adaptive_numerator = 0.0f;
  float exact_maximum = -INFINITY;
  float exact_sum = 0.0f;
  float exact_numerator = 0.0f;
  for (uint token = 0U; token < IQ36_CONTEXT_TOKENS; ++token) {
    const bool cold = token < IQ36_COLD_TOKENS;
    const bool selected = cold && (union_bits[
        (ulong)kv_head * IQ36_ADAPTIVE_UNION_WORDS + (token >> 5U)] &
        (1U << (token & 31U))) != 0U;
    const float adaptive_score = selected
        ? exact_score[(ulong)q_head * IQ36_CONTEXT_TOKENS + token]
        : approximate_score[(ulong)q_head * IQ36_CONTEXT_TOKENS + token];
    const half adaptive_value = selected
        ? exact_value[
            ((ulong)token * IQ36_KV_HEADS + kv_head) * IQ36_HEAD_DIM + dim]
        : cold
        ? iq36_adaptive_cold_value(
            cold_v, cold_v_scales, kv_head, token, dim)
        : iq36_adaptive_hot_value(
            hot_v, kv_head, token - IQ36_COLD_TOKENS, dim);
    const float adaptive_next = fmax(adaptive_max, adaptive_score);
    const float adaptive_previous_scale = native_exp2(
        (adaptive_max - adaptive_next) * IQ36_EXP2_SCALE);
    const float adaptive_weight = native_exp2(
        (adaptive_score - adaptive_next) * IQ36_EXP2_SCALE);
    adaptive_numerator = adaptive_numerator * adaptive_previous_scale +
        convert_float(adaptive_value) * adaptive_weight;
    adaptive_sum = adaptive_sum * adaptive_previous_scale + adaptive_weight;
    adaptive_max = adaptive_next;

    const float score = exact_score[
        (ulong)q_head * IQ36_CONTEXT_TOKENS + token];
    const float exact_next = fmax(exact_maximum, score);
    const float exact_previous_scale = native_exp2(
        (exact_maximum - exact_next) * IQ36_EXP2_SCALE);
    const float exact_weight = native_exp2(
        (score - exact_next) * IQ36_EXP2_SCALE);
    exact_numerator = exact_numerator * exact_previous_scale +
        convert_float(exact_value[
            ((ulong)token * IQ36_KV_HEADS + kv_head) *
                IQ36_HEAD_DIM + dim]) * exact_weight;
    exact_sum = exact_sum * exact_previous_scale + exact_weight;
    exact_maximum = exact_next;
  }
  adaptive_output[index] = adaptive_numerator * native_recip(adaptive_sum);
  exact_output[index] = exact_numerator * native_recip(exact_sum);
}
#endif

__kernel void iq36_direct_i8_reference_score(
    __global const half* query,
    __global const float* key,
    __global float* score) {
  const uint index = (uint)get_global_id(0);
  if (index >= IQ36_Q_HEADS * IQ36_CONTEXT_TOKENS) return;
  const uint q_head = index / IQ36_CONTEXT_TOKENS;
  const uint token = index - q_head * IQ36_CONTEXT_TOKENS;
  const uint kv_head = q_head / IQ36_GQA_GROUP;
  float dot = 0.0f;
  #pragma unroll 1
  for (uint dim = 0U; dim < IQ36_HEAD_DIM; ++dim) {
    dot = fma(convert_float(query[q_head * IQ36_HEAD_DIM + dim]),
        key[((ulong)token * IQ36_KV_HEADS + kv_head) * IQ36_HEAD_DIM + dim],
        dot);
  }
  score[index] = dot;
}

__kernel void iq36_direct_i8_reference_apply(
    __global const float* score,
    __global const float* value,
    __global float* output) {
  const uint index = (uint)get_global_id(0);
  if (index >= IQ36_Q_HEADS * IQ36_HEAD_DIM) return;
  const uint q_head = index / IQ36_HEAD_DIM;
  const uint dim = index - q_head * IQ36_HEAD_DIM;
  const uint kv_head = q_head / IQ36_GQA_GROUP;
  float running_max = -INFINITY;
  float running_sum = 0.0f;
  float accumulated = 0.0f;
  for (uint token = 0U; token < IQ36_CONTEXT_TOKENS; ++token) {
    const float raw = score[(ulong)q_head * IQ36_CONTEXT_TOKENS + token];
    const float next_max = fmax(running_max, raw);
    const float previous_scale = native_exp2(
        (running_max - next_max) * IQ36_EXP2_SCALE);
    const float value_scale = native_exp2(
        (raw - next_max) * IQ36_EXP2_SCALE);
    accumulated = fma(
        value[((ulong)token * IQ36_KV_HEADS + kv_head) * IQ36_HEAD_DIM + dim],
        value_scale, accumulated * previous_scale);
    running_sum = running_sum * previous_scale + value_scale;
    running_max = next_max;
  }
  output[index] = accumulated / running_sum;
}
