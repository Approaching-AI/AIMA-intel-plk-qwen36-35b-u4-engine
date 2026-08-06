// Four-stage decode specialization for the single graph-owned IQ36 adaptive
// attention node.  The plugin compiles exactly one stage per program and
// chains partial -> select/reduce -> correction -> ordered update on device.
// Every stage deliberately keeps the same 13-input/six-output SimpleGPU ABI;
// output0 is the only private packed scratch allocation and only the ordered
// update stage mutates request-owned state.

#if defined(IQ36_BUILD_DECODE_ONLY) && \
    defined(IQ36_ADAPTIVE_ATTENTION_GRAPH) && \
    defined(IQ36_ADAPTIVE_STAGE)

#if !defined(IQ36_DIRECT_I8_FIXED_LAYOUT) || \
    !defined(IQ36_DIMENSION_MAJOR_VALUE_PLANE)
#error "adaptive graph decode requires packed block32 cold state and exact V"
#endif
#if IQ36_HOT_WINDOW != 16384U
#error "adaptive graph decode requires a 16384-token exact hot window"
#endif
#if IQ36_KEY_QUANT_GROUP != 32U || \
    (IQ36_VALUE_QUANT_GROUP != 32U && IQ36_VALUE_QUANT_GROUP != 16U)
#error "adaptive graph decode admits K32 with V32 or V16"
#endif
#if IQ36_ADAPTIVE_TOPK != 128U && IQ36_ADAPTIVE_TOPK != 252U && \
    IQ36_ADAPTIVE_TOPK != 256U && IQ36_ADAPTIVE_TOPK != 512U && \
    IQ36_ADAPTIVE_TOPK != 1024U && IQ36_ADAPTIVE_TOPK != 2048U
#error "adaptive graph top-k must be 128, 252, 256, 512, 1024, or 2048"
#endif

#define IQ36_ADAPTIVE_STAGE_PARTIAL 1
#define IQ36_ADAPTIVE_STAGE_SELECT 2
#define IQ36_ADAPTIVE_STAGE_CORRECT 3
#define IQ36_ADAPTIVE_STAGE_UPDATE 4
#define IQ36_ADAPTIVE_LOCAL_TOPK 64U
#define IQ36_ADAPTIVE_MAX_CHUNKS IQ36_DIM3_EXPAND(INPUT12_DIMS_INIT)
#define IQ36_ADAPTIVE_HOT_CHUNKS \
    (IQ36_HOT_WINDOW / IQ36_CHUNK_TOKENS)
#define IQ36_ADAPTIVE_MAX_COLD_CHUNKS \
    (IQ36_ADAPTIVE_MAX_CHUNKS - IQ36_ADAPTIVE_HOT_CHUNKS)
#define IQ36_ADAPTIVE_MAX_COLD_TOKENS \
    (IQ36_ADAPTIVE_MAX_COLD_CHUNKS * IQ36_CHUNK_TOKENS)
#define IQ36_ADAPTIVE_PARTITION_TOKENS 256U
#define IQ36_ADAPTIVE_MAX_PARTITIONS (IQ36_ADAPTIVE_MAX_CHUNKS * 2U)
#define IQ36_ADAPTIVE_PARTIAL_META \
    (IQ36_KV_HEADS * IQ36_ADAPTIVE_MAX_PARTITIONS * IQ36_GQA_GROUP)
#define IQ36_ADAPTIVE_MAX_TOKENS \
    (IQ36_ADAPTIVE_MAX_CHUNKS * IQ36_CHUNK_TOKENS)
#define IQ36_ADAPTIVE_UNION_WORDS \
    ((IQ36_ADAPTIVE_MAX_COLD_TOKENS + 31U) / 32U)

#if IQ36_ADAPTIVE_MAX_CHUNKS <= IQ36_ADAPTIVE_HOT_CHUNKS
#error "adaptive decode carrier must extend beyond its exact hot window"
#endif
#if IQ36_ADAPTIVE_MAX_COLD_TOKENS >= 65535U
#error "adaptive candidate records require a 16-bit cold-token index"
#endif

// Packed output0 offsets, expressed in F32 elements.  This ordering is the
// seq1674 source bound; every sub-buffer starts on a four-byte boundary and
// the graph rounds the final allocation to 64 bytes.
#define IQ36_AOFF_PARTIAL_MAX 0U
#define IQ36_AOFF_PARTIAL_SUM \
    (IQ36_AOFF_PARTIAL_MAX + IQ36_ADAPTIVE_PARTIAL_META)
#define IQ36_AOFF_PARTIAL_NUMERATOR \
    (IQ36_AOFF_PARTIAL_SUM + IQ36_ADAPTIVE_PARTIAL_META)
#define IQ36_AOFF_APPROXIMATE_SCORE \
    (IQ36_AOFF_PARTIAL_NUMERATOR + \
     IQ36_ADAPTIVE_PARTIAL_META * IQ36_HEAD_DIM)
#define IQ36_AOFF_LOCAL_CANDIDATES \
    (IQ36_AOFF_APPROXIMATE_SCORE + \
     IQ36_Q_HEADS * IQ36_ADAPTIVE_MAX_TOKENS)
#define IQ36_AOFF_UNION_BITS \
    (IQ36_AOFF_LOCAL_CANDIDATES + \
     IQ36_Q_HEADS * IQ36_ADAPTIVE_MAX_COLD_CHUNKS * \
         IQ36_ADAPTIVE_LOCAL_TOPK)
#define IQ36_AOFF_COMPLETION \
    (IQ36_AOFF_UNION_BITS + IQ36_KV_HEADS * IQ36_ADAPTIVE_UNION_WORDS)
#define IQ36_AOFF_ATTENTION \
    (IQ36_AOFF_COMPLETION + IQ36_KV_HEADS)
#define IQ36_ADAPTIVE_REQUIRED_F32 \
    (IQ36_AOFF_ATTENTION + IQ36_Q_HEADS * IQ36_HEAD_DIM)

#define IQ36_ADAPTIVE_ABI \
    const __global INPUT0_TYPE* query, \
    __global INPUT1_TYPE* hot_key_bits, \
    __global INPUT2_TYPE* hot_value, \
    const __global INPUT3_TYPE* current_key, \
    const __global INPUT4_TYPE* current_value, \
    __global INPUT5_TYPE* cold_key, \
    __global INPUT6_TYPE* cold_value, \
    __global INPUT7_TYPE* cold_key_scale_bytes, \
    __global INPUT8_TYPE* cold_value_scale_bytes, \
    const __global INPUT9_TYPE* mask, \
    const __global INPUT10_TYPE* eviction_shape_template, \
    const __global INPUT11_TYPE* eviction_count, \
    const __global INPUT12_TYPE* decode_length_carrier, \
    __global OUTPUT0_TYPE* workspace, \
    __global OUTPUT1_TYPE* output, \
    __global OUTPUT2_TYPE* cold_key_append, \
    __global OUTPUT3_TYPE* cold_value_append, \
    __global OUTPUT4_TYPE* cold_key_scale_append, \
    __global OUTPUT5_TYPE* cold_value_scale_append

inline __global float* iq36_adaptive_workspace_base(
    __global OUTPUT0_TYPE* workspace, const uint batch) {
  return (__global float*)&workspace[
      OUTPUT0_OFFSET + (ulong)batch * OUTPUT0_PITCHES[0]];
}

inline uint iq36_adaptive_key_tokens(
    const __global INPUT12_TYPE* decode_length_carrier,
    const uint batch) {
  return (uint)decode_length_carrier[
      INPUT12_OFFSET + (ulong)batch * INPUT12_PITCHES[0]];
}

inline uint iq36_adaptive_cold_tokens(const uint key_tokens) {
  const uint past_tokens = key_tokens - 1U;
  return past_tokens > IQ36_HOT_WINDOW
      ? past_tokens - IQ36_HOT_WINDOW : 0U;
}

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

inline uint iq36_adaptive_record_radix_key(const uint record) {
  const ushort ordered_score = iq36_adaptive_ordered_half(
      (ushort)(record >> 16U));
  const ushort token = (ushort)record;
  return ((uint)ordered_score << 16U) | (uint)(ushort)~token;
}

#if IQ36_ADAPTIVE_STAGE == IQ36_ADAPTIVE_STAGE_PARTIAL

__attribute__((reqd_work_group_size(128, 1, 1)))
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void iq36_adaptive_attention_partial(IQ36_ADAPTIVE_ABI) {
  const uint local_id = (uint)get_local_id(0);
  const uint subgroup = (uint)get_sub_group_id();
  const uint lane = (uint)get_sub_group_local_id();
  const uint chunk = (uint)get_group_id(1);
  const uint batch_kv = (uint)get_group_id(2);
  const uint batch = batch_kv / IQ36_KV_HEADS;
  const uint kv_head = batch_kv - batch * IQ36_KV_HEADS;
  const uint key_tokens = iq36_adaptive_key_tokens(
      decode_length_carrier, batch);
  const uint past_tokens = key_tokens - 1U;
  const uint cold_tokens = iq36_adaptive_cold_tokens(key_tokens);
  const uint chunk_count =
      (key_tokens + IQ36_CHUNK_TOKENS - 1U) / IQ36_CHUNK_TOKENS;
  const uint cold_chunk_count =
      (cold_tokens + IQ36_CHUNK_TOKENS - 1U) / IQ36_CHUNK_TOKENS;
  if (chunk >= chunk_count || chunk_count > IQ36_ADAPTIVE_MAX_CHUNKS ||
      cold_chunk_count > IQ36_ADAPTIVE_MAX_COLD_CHUNKS) return;

  __global float* base = iq36_adaptive_workspace_base(workspace, batch);
  __global float* approximate_score =
      base + IQ36_AOFF_APPROXIMATE_SCORE;
  __global uint* local_candidates =
      (__global uint*)(base + IQ36_AOFF_LOCAL_CANDIDATES);
  __global uint* union_bits =
      (__global uint*)(base + IQ36_AOFF_UNION_BITS);
  __global uint* completion =
      (__global uint*)(base + IQ36_AOFF_COMPLETION);

  // No partial work-group writes these buffers.  Clearing them in chunk zero
  // is therefore race-free; the plugin's event chain is the device barrier
  // before selection/correction consume them.
  if (chunk == 0U && kv_head == 0U) {
    for (uint index = local_id;
         index < IQ36_KV_HEADS * IQ36_ADAPTIVE_UNION_WORDS;
         index += 128U) {
      union_bits[index] = 0U;
    }
    if (local_id < IQ36_KV_HEADS) completion[local_id] = 0U;
  }

  __local float score_weight[IQ36_GQA_GROUP * IQ36_CHUNK_TOKENS];

  const uint chunk_begin = chunk * IQ36_CHUNK_TOKENS;
  const __global half* query_base = (const __global half*)&query[
      INPUT0_OFFSET + (ulong)batch * INPUT0_PITCHES[0]];
  for (uint block = subgroup; block < IQ36_BLOCKS_PER_CHUNK;
       block += 8U) {
    const uint block_token = chunk_begin + block * IQ36_TOKEN_TILE;
    const uint token = block_token + lane;
    const uint block_slot = iq36_hot_slot(block_token);
    const bool direct_cold_block =
#if defined(IQ36_ADAPTIVE_KEY_EXACT)
        false;
#else
        block_token >= IQ36_SINK_TOKENS &&
        block_token + IQ36_TOKEN_TILE <= cold_tokens;
#endif
    const bool exact_history_block =
#if defined(IQ36_ADAPTIVE_KEY_EXACT)
        block_token + IQ36_TOKEN_TILE <= past_tokens &&
#else
        block_token >= cold_tokens &&
        block_token + IQ36_TOKEN_TILE <= past_tokens &&
#endif
        (block_slot & (IQ36_KEY_TILE_TOKENS - 1U)) == 0U &&
        block_slot + IQ36_TOKEN_TILE <= (uint)INPUT2_DIMS[2];
    float8 score = (float8)(0.0f);
    if (direct_cold_block) {
      // One 32-byte subgroup read feeds both K16 MMAs.  Loading through the
      // generic K16 helper would fetch this packed block twice.
      #pragma unroll 1
      for (uint scale_group = 0U;
           scale_group < IQ36_KEY_SCALE_GROUPS; ++scale_group) {
        const uint k_base = scale_group * IQ36_KEY_QUANT_GROUP;
        const short8 query0 = as_short8(iq36_block2d_load_f16_16x8(
            query_base,
            (int)INPUT0_DIMS[3] * (int)sizeof(half),
            (int)INPUT0_DIMS[1],
            (int)INPUT0_PITCHES[1] * (int)sizeof(half),
            k_base, kv_head * IQ36_GQA_GROUP));
        const short8 query1 = as_short8(iq36_block2d_load_f16_16x8(
            query_base,
            (int)INPUT0_DIMS[3] * (int)sizeof(half),
            (int)INPUT0_DIMS[1],
            (int)INPUT0_PITCHES[1] * (int)sizeof(half),
            k_base + 16U, kv_head * IQ36_GQA_GROUP));
        const int16 key = iq36_direct_cold_key_group32_fragments(
            batch, kv_head, token, scale_group,
            cold_key, cold_key_scale_bytes);
        score = intel_sub_group_f16_f16_matrix_mad_k16(
            query0, key.lo, score);
        score = intel_sub_group_f16_f16_matrix_mad_k16(
            query1, key.hi, score);
      }
    } else {
      #pragma unroll 1
      for (uint k_block = 0U;
           k_block < IQ36_HEAD_DIM / IQ36_TOKEN_TILE; ++k_block) {
        const uint k_base = k_block * IQ36_TOKEN_TILE;
        const short8 query_fragment = as_short8(
            iq36_block2d_load_f16_16x8(
                query_base,
                (int)INPUT0_DIMS[3] * (int)sizeof(half),
                (int)INPUT0_DIMS[1],
                (int)INPUT0_PITCHES[1] * (int)sizeof(half),
                k_base, kv_head * IQ36_GQA_GROUP));
        int8 key_fragment;
        if (exact_history_block) {
          const ulong key_index = INPUT1_OFFSET +
              (ulong)batch * INPUT1_PITCHES[0] +
              (ulong)kv_head * INPUT1_PITCHES[1] +
              (ulong)(block_slot / IQ36_KEY_TILE_TOKENS) *
                  INPUT1_PITCHES[2] +
              (ulong)(k_block * 8U * IQ36_KEY_TILE_TOKENS) *
                  INPUT1_PITCHES[3];
          key_fragment = as_int8(intel_sub_group_block_read8(
              (const __global uint*)&hot_key_bits[key_index]));
        } else {
          #pragma unroll
          for (uint pair = 0U; pair < 8U; ++pair) {
            const uint dim0 = k_base + pair * 2U;
            half2 values = (half2)(0.0h);
            if (token < key_tokens) {
              values[0] = convert_half_rte(iq36_partial_load_key(
                  batch, kv_head, token, past_tokens, cold_tokens,
                  dim0, hot_key_bits, current_key, cold_key,
                  cold_key_scale_bytes));
              values[1] = convert_half_rte(iq36_partial_load_key(
                  batch, kv_head, token, past_tokens, cold_tokens,
                  dim0 + 1U, hot_key_bits, current_key, cold_key,
                  cold_key_scale_bytes));
            }
            key_fragment[pair] = as_int(values);
          }
        }
        score = intel_sub_group_f16_f16_matrix_mad_k16(
            query_fragment, key_fragment, score);
      }
    }
    #pragma unroll
    for (uint head = 0U; head < IQ36_GQA_GROUP; ++head) {
      const float value = token < key_tokens ? score[head] : -INFINITY;
      score_weight[head * IQ36_CHUNK_TOKENS +
          block * IQ36_TOKEN_TILE + lane] = value;
      if (token < key_tokens) {
        const uint q_head = kv_head * IQ36_GQA_GROUP + head;
        approximate_score[
            (ulong)q_head * IQ36_ADAPTIVE_MAX_TOKENS + token] = value;
      }
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  if (chunk < cold_chunk_count) {
    const uint cold_valid = min(
        (uint)IQ36_CHUNK_TOKENS, cold_tokens - chunk_begin);
    const uint rank_target = min(
        (uint)IQ36_ADAPTIVE_LOCAL_TOPK, cold_valid);
    if (rank_target != 0U) {
      ushort prefix = 0U;
      ushort processed_mask = 0U;
      uint rank = rank_target;
      for (int shift = 12; shift >= 0; shift -= 4) {
        uint16 lane_counts = (uint16)(0U);
        for (uint position = lane; position < cold_valid; position += 16U) {
          const ushort score_bits = as_ushort(convert_half_rte(
              score_weight[subgroup * IQ36_CHUNK_TOKENS + position] *
                  IQ36_ATTENTION_SCALE));
          const ushort ordered = iq36_adaptive_ordered_half(score_bits);
          if ((ordered & processed_mask) == prefix) {
            const uint digit = (ordered >> shift) & 15U;
            lane_counts += (uint16)(
                digit == 0U, digit == 1U, digit == 2U, digit == 3U,
                digit == 4U, digit == 5U, digit == 6U, digit == 7U,
                digit == 8U, digit == 9U, digit == 10U, digit == 11U,
                digit == 12U, digit == 13U, digit == 14U, digit == 15U);
          }
        }
        ushort selected_digit = 0U;
        bool found = false;
        #define IQ36_ADAPTIVE_RADIX_DIGIT(field, digit) \
          if (!found) { \
            const uint count = sub_group_reduce_add(lane_counts.field); \
            if (count >= rank) { \
              selected_digit = (ushort)(digit); \
              found = true; \
            } else { \
              rank -= count; \
            } \
          }
        IQ36_ADAPTIVE_RADIX_DIGIT(sf, 15U)
        IQ36_ADAPTIVE_RADIX_DIGIT(se, 14U)
        IQ36_ADAPTIVE_RADIX_DIGIT(sd, 13U)
        IQ36_ADAPTIVE_RADIX_DIGIT(sc, 12U)
        IQ36_ADAPTIVE_RADIX_DIGIT(sb, 11U)
        IQ36_ADAPTIVE_RADIX_DIGIT(sa, 10U)
        IQ36_ADAPTIVE_RADIX_DIGIT(s9, 9U)
        IQ36_ADAPTIVE_RADIX_DIGIT(s8, 8U)
        IQ36_ADAPTIVE_RADIX_DIGIT(s7, 7U)
        IQ36_ADAPTIVE_RADIX_DIGIT(s6, 6U)
        IQ36_ADAPTIVE_RADIX_DIGIT(s5, 5U)
        IQ36_ADAPTIVE_RADIX_DIGIT(s4, 4U)
        IQ36_ADAPTIVE_RADIX_DIGIT(s3, 3U)
        IQ36_ADAPTIVE_RADIX_DIGIT(s2, 2U)
        IQ36_ADAPTIVE_RADIX_DIGIT(s1, 1U)
        IQ36_ADAPTIVE_RADIX_DIGIT(s0, 0U)
        #undef IQ36_ADAPTIVE_RADIX_DIGIT
        prefix = (ushort)(prefix | (selected_digit << shift));
        processed_mask = (ushort)(processed_mask | (15U << shift));
      }
      const uint position_begin = lane * 32U;
      uint above_count = 0U;
      uint tie_count = 0U;
      #pragma unroll
      for (uint offset = 0U; offset < 32U; ++offset) {
        const uint position = position_begin + offset;
        if (position < cold_valid) {
          const ushort score_bits = as_ushort(convert_half_rte(
              score_weight[subgroup * IQ36_CHUNK_TOKENS + position] *
                  IQ36_ATTENTION_SCALE));
          const ushort ordered = iq36_adaptive_ordered_half(score_bits);
          above_count += ordered > prefix;
          tie_count += ordered == prefix;
        }
      }
      const uint above_offset =
          sub_group_scan_exclusive_add(above_count);
      const uint tie_offset = sub_group_scan_exclusive_add(tie_count);
      const uint above_total = sub_group_reduce_add(above_count);
      const uint tie_target = rank_target - above_total;
      const uint q_head = kv_head * IQ36_GQA_GROUP + subgroup;
      const ulong destination =
          ((ulong)q_head * IQ36_ADAPTIVE_MAX_COLD_CHUNKS + chunk) *
              IQ36_ADAPTIVE_LOCAL_TOPK;
      const uint invalid = ((uint)as_ushort((half)-INFINITY) << 16U) |
          0xffffU;
      for (uint index = rank_target + lane;
           index < IQ36_ADAPTIVE_LOCAL_TOPK; index += 16U) {
        local_candidates[destination + index] = invalid;
      }
      uint above_written = 0U;
      uint tie_written = 0U;
      #pragma unroll
      for (uint offset = 0U; offset < 32U; ++offset) {
        const uint position = position_begin + offset;
        if (position < cold_valid) {
          const ushort score_bits = as_ushort(convert_half_rte(
              score_weight[subgroup * IQ36_CHUNK_TOKENS + position] *
                  IQ36_ATTENTION_SCALE));
          const ushort ordered = iq36_adaptive_ordered_half(score_bits);
          const uint record =
              ((uint)score_bits << 16U) | (chunk_begin + position);
          if (ordered > prefix) {
            local_candidates[
                destination + above_offset + above_written++] = record;
          } else if (ordered == prefix) {
            const uint tie_rank = tie_offset + tie_written++;
            if (tie_rank < tie_target) {
              local_candidates[
                  destination + above_total + tie_rank] = record;
            }
          }
        }
      }
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
}

#elif IQ36_ADAPTIVE_STAGE == IQ36_ADAPTIVE_STAGE_SELECT

__attribute__((reqd_work_group_size(256, 1, 1)))
__attribute__((intel_reqd_sub_group_size(32)))
__kernel void iq36_adaptive_attention_select_reduce_union(
    IQ36_ADAPTIVE_ABI) {
  const uint local_id = (uint)get_local_id(0);
  const uint q_head = (uint)get_group_id(1);
  const uint batch = (uint)get_group_id(2);
  const uint kv_head = q_head / IQ36_GQA_GROUP;
  const uint key_tokens = iq36_adaptive_key_tokens(
      decode_length_carrier, batch);
  const uint cold_tokens = iq36_adaptive_cold_tokens(key_tokens);
  const uint cold_chunk_count =
      (cold_tokens + IQ36_CHUNK_TOKENS - 1U) / IQ36_CHUNK_TOKENS;
  __global float* base = iq36_adaptive_workspace_base(workspace, batch);
  __global const uint* local_candidates =
      (__global const uint*)(base + IQ36_AOFF_LOCAL_CANDIDATES);
  __global uint* union_bits =
      (__global uint*)(base + IQ36_AOFF_UNION_BITS);

  __local uint radix_histogram[8U * 16U];
  __local uint radix_prefix;
  __local uint radix_mask;
  __local uint radix_rank;
  const uint subgroup = (uint)get_sub_group_id();
  const uint lane = (uint)get_sub_group_local_id();
  const ulong candidate_base =
      (ulong)q_head * IQ36_ADAPTIVE_MAX_COLD_CHUNKS *
          IQ36_ADAPTIVE_LOCAL_TOPK;
  const uint candidate_count =
      cold_chunk_count * IQ36_ADAPTIVE_LOCAL_TOPK;
  if (candidate_count == 0U) return;
  if (local_id == 0U) {
    radix_prefix = 0U;
    radix_mask = 0U;
    radix_rank = min((uint)IQ36_ADAPTIVE_TOPK, candidate_count);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int shift = 28; shift >= 0; shift -= 4) {
    uint16 lane_counts = (uint16)(0U);
    for (uint index = local_id; index < candidate_count; index += 256U) {
      const uint key = iq36_adaptive_record_radix_key(
          local_candidates[candidate_base + index]);
      if ((key & radix_mask) == radix_prefix) {
        const uint digit = (key >> shift) & 15U;
        lane_counts += (uint16)(
            digit == 0U, digit == 1U, digit == 2U, digit == 3U,
            digit == 4U, digit == 5U, digit == 6U, digit == 7U,
            digit == 8U, digit == 9U, digit == 10U, digit == 11U,
            digit == 12U, digit == 13U, digit == 14U, digit == 15U);
      }
    }
    const uint16 subgroup_counts = (uint16)(
        sub_group_reduce_add(lane_counts.s0),
        sub_group_reduce_add(lane_counts.s1),
        sub_group_reduce_add(lane_counts.s2),
        sub_group_reduce_add(lane_counts.s3),
        sub_group_reduce_add(lane_counts.s4),
        sub_group_reduce_add(lane_counts.s5),
        sub_group_reduce_add(lane_counts.s6),
        sub_group_reduce_add(lane_counts.s7),
        sub_group_reduce_add(lane_counts.s8),
        sub_group_reduce_add(lane_counts.s9),
        sub_group_reduce_add(lane_counts.sa),
        sub_group_reduce_add(lane_counts.sb),
        sub_group_reduce_add(lane_counts.sc),
        sub_group_reduce_add(lane_counts.sd),
        sub_group_reduce_add(lane_counts.se),
        sub_group_reduce_add(lane_counts.sf));
    if (lane == 0U) {
      vstore16(subgroup_counts, subgroup, radix_histogram);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_id == 0U) {
      uint rank = radix_rank;
      uint selected_digit = 0U;
      bool found = false;
      #define IQ36_ADAPTIVE_GLOBAL_RADIX_DIGIT(digit) \
        if (!found) { \
          uint count = 0U; \
          for (uint sg = 0U; sg < 8U; ++sg) { \
            count += radix_histogram[sg * 16U + (digit)]; \
          } \
          if (count >= rank) { \
            selected_digit = (digit); \
            found = true; \
          } else { \
            rank -= count; \
          } \
        }
      IQ36_ADAPTIVE_GLOBAL_RADIX_DIGIT(15U)
      IQ36_ADAPTIVE_GLOBAL_RADIX_DIGIT(14U)
      IQ36_ADAPTIVE_GLOBAL_RADIX_DIGIT(13U)
      IQ36_ADAPTIVE_GLOBAL_RADIX_DIGIT(12U)
      IQ36_ADAPTIVE_GLOBAL_RADIX_DIGIT(11U)
      IQ36_ADAPTIVE_GLOBAL_RADIX_DIGIT(10U)
      IQ36_ADAPTIVE_GLOBAL_RADIX_DIGIT(9U)
      IQ36_ADAPTIVE_GLOBAL_RADIX_DIGIT(8U)
      IQ36_ADAPTIVE_GLOBAL_RADIX_DIGIT(7U)
      IQ36_ADAPTIVE_GLOBAL_RADIX_DIGIT(6U)
      IQ36_ADAPTIVE_GLOBAL_RADIX_DIGIT(5U)
      IQ36_ADAPTIVE_GLOBAL_RADIX_DIGIT(4U)
      IQ36_ADAPTIVE_GLOBAL_RADIX_DIGIT(3U)
      IQ36_ADAPTIVE_GLOBAL_RADIX_DIGIT(2U)
      IQ36_ADAPTIVE_GLOBAL_RADIX_DIGIT(1U)
      IQ36_ADAPTIVE_GLOBAL_RADIX_DIGIT(0U)
      #undef IQ36_ADAPTIVE_GLOBAL_RADIX_DIGIT
      radix_prefix |= selected_digit << shift;
      radix_mask |= 15U << shift;
      radix_rank = rank;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const uint selection_threshold = radix_prefix;
  for (uint index = local_id; index < candidate_count; index += 256U) {
    const uint record = local_candidates[candidate_base + index];
    const uint token = record & 0xffffU;
    if (token < cold_tokens &&
        iq36_adaptive_record_radix_key(record) >= selection_threshold) {
      atomic_or((volatile __global unsigned int*)&union_bits[
          (ulong)kv_head * IQ36_ADAPTIVE_UNION_WORDS + (token >> 5U)],
          1U << (token & 31U));
    }
  }
}

#elif IQ36_ADAPTIVE_STAGE == IQ36_ADAPTIVE_STAGE_CORRECT

#if 1
// Rebuild each 256-token online-softmax partition after selection, so exact
// K/V replacements enter at the same arithmetic point as stock sdpa_micro.
// Stage 1 writes KQ scores but deliberately does not read V; this stage reads
// V exactly once and therefore preserves the compressed-state traffic cut.
__attribute__((reqd_work_group_size(128, 1, 1)))
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void iq36_adaptive_attention_correct_normalize(
    IQ36_ADAPTIVE_ABI) {
  const uint local_id = (uint)get_local_id(0);
  const uint head = (uint)get_sub_group_id();
  const uint lane = (uint)get_sub_group_local_id();
  const uint partition = (uint)get_group_id(1);
  const uint batch_kv = (uint)get_group_id(2);
  const uint batch = batch_kv / IQ36_KV_HEADS;
  const uint kv_head = batch_kv - batch * IQ36_KV_HEADS;
  const uint q_head = kv_head * IQ36_GQA_GROUP + head;
  const uint key_tokens = iq36_adaptive_key_tokens(
      decode_length_carrier, batch);
  const uint past_tokens = key_tokens - 1U;
  const uint cold_tokens = iq36_adaptive_cold_tokens(key_tokens);
  const uint partition_count =
      (key_tokens + IQ36_ADAPTIVE_PARTITION_TOKENS - 1U) /
          IQ36_ADAPTIVE_PARTITION_TOKENS;
  if (partition >= partition_count ||
      partition_count > IQ36_ADAPTIVE_MAX_PARTITIONS) return;
  const uint begin_token =
      partition * IQ36_ADAPTIVE_PARTITION_TOKENS;
  const uint end_token = min(
      begin_token + IQ36_ADAPTIVE_PARTITION_TOKENS, key_tokens);

  __global float* base = iq36_adaptive_workspace_base(workspace, batch);
  __global float* partial_max = base + IQ36_AOFF_PARTIAL_MAX;
  __global float* partial_sum = base + IQ36_AOFF_PARTIAL_SUM;
  __global float* partial_numerator =
      base + IQ36_AOFF_PARTIAL_NUMERATOR;
  __global const float* approximate_score =
      base + IQ36_AOFF_APPROXIMATE_SCORE;
  __global const uint* union_bits =
      (__global const uint*)(base + IQ36_AOFF_UNION_BITS);
  __global uint* completion =
      (__global uint*)(base + IQ36_AOFF_COMPLETION);
  __global float* attention = base + IQ36_AOFF_ATTENTION;

  __local float score_weight[
      IQ36_GQA_GROUP * IQ36_ADAPTIVE_PARTITION_TOKENS];
  __local float block_max[
      (IQ36_ADAPTIVE_PARTITION_TOKENS / IQ36_TOKEN_TILE) *
          IQ36_GQA_GROUP];
  __local float block_sum[
      (IQ36_ADAPTIVE_PARTITION_TOKENS / IQ36_TOKEN_TILE) *
          IQ36_GQA_GROUP];
  __local float global_max[IQ36_GQA_GROUP];
  __local float global_sum[IQ36_GQA_GROUP];
#if !defined(IQ36_ADAPTIVE_KEY_EXACT)
  __local ushort selected_indices[IQ36_ADAPTIVE_PARTITION_TOKENS];
  __local uint selected_count;
  __local uint selected_subgroup_offsets[IQ36_GQA_GROUP];
  __local half exact_key[16U * IQ36_HEAD_DIM];
#endif
  __local uint is_last_partition;

  for (uint index = local_id;
       index < IQ36_GQA_GROUP * IQ36_ADAPTIVE_PARTITION_TOKENS;
       index += 128U) {
    const uint score_head = index / IQ36_ADAPTIVE_PARTITION_TOKENS;
    const uint position =
        index - score_head * IQ36_ADAPTIVE_PARTITION_TOKENS;
    const uint token = begin_token + position;
    const uint score_q_head =
        kv_head * IQ36_GQA_GROUP + score_head;
    score_weight[index] = token < end_token
        ? approximate_score[
            (ulong)score_q_head * IQ36_ADAPTIVE_MAX_TOKENS + token]
        : -INFINITY;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

#if !defined(IQ36_ADAPTIVE_KEY_EXACT)
  const uint token_base = begin_token + local_id * 2U;
  uint selected_mask = 0U;
  if (token_base < end_token && token_base < cold_tokens) {
    const uint word = union_bits[
        (ulong)kv_head * IQ36_ADAPTIVE_UNION_WORDS + (token_base >> 5U)];
    #pragma unroll
    for (uint offset = 0U; offset < 2U; ++offset) {
      const uint token = token_base + offset;
      if (token >= IQ36_SINK_TOKENS && token < end_token &&
          token < cold_tokens &&
          (word & (1U << (token & 31U))) != 0U) {
        selected_mask |= 1U << offset;
      }
    }
  }
  const uint lane_count = popcount(selected_mask);
  uint selected_offset = sub_group_scan_exclusive_add(lane_count);
  const uint subgroup_count = sub_group_reduce_add(lane_count);
  if (lane == 0U) selected_subgroup_offsets[head] = subgroup_count;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (local_id == 0U) {
    uint count = 0U;
    #pragma unroll
    for (uint subgroup = 0U; subgroup < IQ36_GQA_GROUP; ++subgroup) {
      const uint next = selected_subgroup_offsets[subgroup];
      selected_subgroup_offsets[subgroup] = count;
      count += next;
    }
    selected_count = count;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  selected_offset += selected_subgroup_offsets[head];
  #pragma unroll
  for (uint offset = 0U; offset < 2U; ++offset) {
    if ((selected_mask & (1U << offset)) != 0U) {
      selected_indices[selected_offset++] = (ushort)(token_base + offset);
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  half query_fragment[16];
  #pragma unroll
  for (uint slot = 0U; slot < 16U; ++slot) {
    const uint dim = lane + slot * 16U;
    query_fragment[slot] = (half)query[
        INPUT0_OFFSET + (ulong)batch * INPUT0_PITCHES[0] +
        (ulong)q_head * INPUT0_PITCHES[1] +
        (ulong)dim * INPUT0_PITCHES[3]];
  }
  const __global half* dense_key = (const __global half*)&hot_key_bits[
      iq36_hot_key_dense_i32_base(batch, kv_head)];
  for (uint selected_begin = 0U; selected_begin < selected_count;
       selected_begin += 16U) {
    const uint rows = min(16U, selected_count - selected_begin);
    for (uint index = local_id; index < rows * IQ36_HEAD_DIM;
         index += 128U) {
      const uint row = index / IQ36_HEAD_DIM;
      const uint dim = index - row * IQ36_HEAD_DIM;
      const uint token = (uint)selected_indices[selected_begin + row];
      const uint slot = iq36_hot_slot(token);
      exact_key[index] = dense_key[(ulong)slot * IQ36_HEAD_DIM + dim];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint row = 0U; row < rows; ++row) {
      float exact_score = 0.0f;
      #pragma unroll
      for (uint slot = 0U; slot < 16U; ++slot) {
        const uint dim = lane + slot * 16U;
        exact_score = fma(convert_float(query_fragment[slot]),
            convert_float(exact_key[row * IQ36_HEAD_DIM + dim]),
            exact_score);
      }
      exact_score = sub_group_reduce_add(exact_score);
      if (lane == 0U) {
        const uint token =
            (uint)selected_indices[selected_begin + row];
        score_weight[
            head * IQ36_ADAPTIVE_PARTITION_TOKENS +
            token - begin_token] = exact_score;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
#endif

  const uint partition_blocks =
      IQ36_ADAPTIVE_PARTITION_TOKENS / IQ36_TOKEN_TILE;
  for (uint block = head; block < partition_blocks;
       block += IQ36_GQA_GROUP) {
    const uint token_in_partition = block * IQ36_TOKEN_TILE + lane;
    #pragma unroll
    for (uint score_head = 0U; score_head < IQ36_GQA_GROUP;
         ++score_head) {
      const float value = score_weight[
          score_head * IQ36_ADAPTIVE_PARTITION_TOKENS +
          token_in_partition];
      const float maximum = sub_group_reduce_max(value);
      if (lane == 0U) {
        block_max[block * IQ36_GQA_GROUP + score_head] = maximum;
      }
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (head == 0U && lane < IQ36_GQA_GROUP) {
    float maximum = -INFINITY;
    #pragma unroll
    for (uint block = 0U; block < partition_blocks; ++block) {
      maximum = fmax(
          maximum, block_max[block * IQ36_GQA_GROUP + lane]);
    }
    global_max[lane] = maximum;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  for (uint block = head; block < partition_blocks;
       block += IQ36_GQA_GROUP) {
    const uint token_in_partition = block * IQ36_TOKEN_TILE + lane;
    #pragma unroll
    for (uint score_head = 0U; score_head < IQ36_GQA_GROUP;
         ++score_head) {
      const uint score_index =
          score_head * IQ36_ADAPTIVE_PARTITION_TOKENS +
          token_in_partition;
      const float weight = native_exp2(
          (score_weight[score_index] - global_max[score_head]) *
          IQ36_EXP2_SCALE);
      score_weight[score_index] = weight;
      const float sum = sub_group_reduce_add(weight);
      if (lane == 0U) {
        block_sum[block * IQ36_GQA_GROUP + score_head] = sum;
      }
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (head == 0U && lane < IQ36_GQA_GROUP) {
    float sum = 0.0f;
    #pragma unroll
    for (uint block = 0U; block < partition_blocks; ++block) {
      sum += block_sum[block * IQ36_GQA_GROUP + lane];
    }
    global_sum[lane] = sum;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  const uint dim0 = head * 32U + lane;
  const uint dim1 = dim0 + 16U;
  const __global half* dense_value = (const __global half*)&hot_key_bits[
      iq36_hot_value_dimension_i32_base(batch, kv_head)];
  float8 numerator0 = (float8)(0.0f);
  float8 numerator1 = (float8)(0.0f);
  for (uint block = 0U; block < partition_blocks; ++block) {
    const uint block_token = begin_token + block * IQ36_TOKEN_TILE;
    const uint token_in_partition = block * IQ36_TOKEN_TILE + lane;
    const uint block_slot = iq36_hot_slot(block_token);
    uint block_selected = 0U;
    if (block_token < cold_tokens &&
        block_token + IQ36_TOKEN_TILE > IQ36_SINK_TOKENS) {
      const uint word = union_bits[
          (ulong)kv_head * IQ36_ADAPTIVE_UNION_WORDS +
          (block_token >> 5U)];
      block_selected =
          (word >> (block_token & 31U)) & 0xffffU;
    }
    const bool direct_cold_block =
        block_selected == 0U &&
        block_token >= IQ36_SINK_TOKENS &&
        block_token + IQ36_TOKEN_TILE <= cold_tokens;
    const bool exact_history_block =
        block_token >= cold_tokens &&
        block_token + IQ36_TOKEN_TILE <= past_tokens &&
        (block_slot & (IQ36_KEY_TILE_TOKENS - 1U)) == 0U &&
        block_slot + IQ36_TOKEN_TILE <= (uint)INPUT2_DIMS[2];
    const short8 weights = (short8)(
        as_short(convert_half_rte(score_weight[
            0U * IQ36_ADAPTIVE_PARTITION_TOKENS + token_in_partition])),
        as_short(convert_half_rte(score_weight[
            1U * IQ36_ADAPTIVE_PARTITION_TOKENS + token_in_partition])),
        as_short(convert_half_rte(score_weight[
            2U * IQ36_ADAPTIVE_PARTITION_TOKENS + token_in_partition])),
        as_short(convert_half_rte(score_weight[
            3U * IQ36_ADAPTIVE_PARTITION_TOKENS + token_in_partition])),
        as_short(convert_half_rte(score_weight[
            4U * IQ36_ADAPTIVE_PARTITION_TOKENS + token_in_partition])),
        as_short(convert_half_rte(score_weight[
            5U * IQ36_ADAPTIVE_PARTITION_TOKENS + token_in_partition])),
        as_short(convert_half_rte(score_weight[
            6U * IQ36_ADAPTIVE_PARTITION_TOKENS + token_in_partition])),
        as_short(convert_half_rte(score_weight[
            7U * IQ36_ADAPTIVE_PARTITION_TOKENS + token_in_partition])));
    int8 value0;
    int8 value1;
    short8 weights0 = weights;
    short8 weights1 = weights;
    if (direct_cold_block) {
#if defined(IQ36_ADAPTIVE_PACKED_KV) && \
    !defined(IQ36_ADAPTIVE_DIMENSION_MAJOR_V8)
      // Diagnostic scalar reconstruction keeps the token-major packed ABI
      // while proving the value-tile orientation independently of the fast
      // transposed subgroup carrier.
      half16 values0 = (half16)(0.0h);
      half16 values1 = (half16)(0.0h);
      #pragma unroll
      for (uint row = 0U; row < IQ36_TOKEN_TILE; ++row) {
        const uint token = block_token + row;
        values0[row] = iq36_direct_cold_value_element(
            batch, kv_head, token, dim0, cold_value,
            cold_value_scale_bytes);
        values1[row] = iq36_direct_cold_value_element(
            batch, kv_head, token, dim1, cold_value,
            cold_value_scale_bytes);
      }
      value0 = as_int8(values0);
      value1 = as_int8(values1);
#else
      const uint value_token = block_token + lane;
      const half value_scale0 = iq36_direct_cold_value_scale(
          batch, kv_head, value_token, dim0 / IQ36_VALUE_QUANT_GROUP,
          cold_value_scale_bytes);
      const half value_scale1 = iq36_direct_cold_value_scale(
          batch, kv_head, value_token, dim1 / IQ36_VALUE_QUANT_GROUP,
          cold_value_scale_bytes);
#if defined(IQ36_ADAPTIVE_DIMENSION_MAJOR_V8)
      // The packed scalar carrier rounds (quantized V * scale) to F16 before
      // DPAS.  Preserve that arithmetic boundary while using one cooperative
      // 32x16 byte transpose for V: every lane contributes the scale for its
      // token, then all lanes reconstruct the common token-scale vector.
      const half16 value_scales0 = (half16)(
          sub_group_broadcast(value_scale0, 0U),
          sub_group_broadcast(value_scale0, 1U),
          sub_group_broadcast(value_scale0, 2U),
          sub_group_broadcast(value_scale0, 3U),
          sub_group_broadcast(value_scale0, 4U),
          sub_group_broadcast(value_scale0, 5U),
          sub_group_broadcast(value_scale0, 6U),
          sub_group_broadcast(value_scale0, 7U),
          sub_group_broadcast(value_scale0, 8U),
          sub_group_broadcast(value_scale0, 9U),
          sub_group_broadcast(value_scale0, 10U),
          sub_group_broadcast(value_scale0, 11U),
          sub_group_broadcast(value_scale0, 12U),
          sub_group_broadcast(value_scale0, 13U),
          sub_group_broadcast(value_scale0, 14U),
          sub_group_broadcast(value_scale0, 15U));
      const half16 value_scales1 = (half16)(
          sub_group_broadcast(value_scale1, 0U),
          sub_group_broadcast(value_scale1, 1U),
          sub_group_broadcast(value_scale1, 2U),
          sub_group_broadcast(value_scale1, 3U),
          sub_group_broadcast(value_scale1, 4U),
          sub_group_broadcast(value_scale1, 5U),
          sub_group_broadcast(value_scale1, 6U),
          sub_group_broadcast(value_scale1, 7U),
          sub_group_broadcast(value_scale1, 8U),
          sub_group_broadcast(value_scale1, 9U),
          sub_group_broadcast(value_scale1, 10U),
          sub_group_broadcast(value_scale1, 11U),
          sub_group_broadcast(value_scale1, 12U),
          sub_group_broadcast(value_scale1, 13U),
          sub_group_broadcast(value_scale1, 14U),
          sub_group_broadcast(value_scale1, 15U));
#else
      weights0 = as_short8(as_half8(weights) * value_scale0);
      weights1 = as_short8(as_half8(weights) * value_scale1);
#endif
      const int16 values =
          iq36_direct_cold_value_group32_fragments_unscaled(
              batch, kv_head, block_token,
              dim0 / IQ36_VALUE_QUANT_GROUP, cold_value,
              cold_value_scale_bytes);
#if defined(IQ36_ADAPTIVE_DIMENSION_MAJOR_V8)
      value0 = as_int8(as_half16(values.lo) * value_scales0);
      value1 = as_int8(as_half16(values.hi) * value_scales1);
#else
      value0 = values.lo;
      value1 = values.hi;
#endif
#endif
    } else if (exact_history_block) {
      value0 = iq36_direct_hot_value_fragment(
          batch, kv_head, block_slot, dim0, hot_key_bits);
      value1 = iq36_direct_hot_value_fragment(
          batch, kv_head, block_slot, dim1, hot_key_bits);
    } else {
      half16 values0 = (half16)(0.0h);
      half16 values1 = (half16)(0.0h);
      #pragma unroll
      for (uint row = 0U; row < IQ36_TOKEN_TILE; ++row) {
        const uint token = block_token + row;
        if (token < end_token) {
          const bool selected = token < cold_tokens &&
              (block_selected & (1U << row)) != 0U;
          if (selected) {
            const uint slot = iq36_hot_slot(token);
            values0[row] = dense_value[
                (ulong)dim0 * (uint)INPUT2_DIMS[2] + slot];
            values1[row] = dense_value[
                (ulong)dim1 * (uint)INPUT2_DIMS[2] + slot];
          } else {
            values0[row] = convert_half_rte(iq36_partial_load_value(
                batch, kv_head, token, past_tokens, cold_tokens, dim0,
                hot_value, current_value, cold_value,
                cold_value_scale_bytes));
            values1[row] = convert_half_rte(iq36_partial_load_value(
                batch, kv_head, token, past_tokens, cold_tokens, dim1,
                hot_value, current_value, cold_value,
                cold_value_scale_bytes));
          }
        }
      }
      value0 = as_int8(values0);
      value1 = as_int8(values1);
    }
    numerator0 = intel_sub_group_f16_f16_matrix_mad_k16(
        weights0, value0, numerator0);
    numerator1 = intel_sub_group_f16_f16_matrix_mad_k16(
        weights1, value1, numerator1);
  }

  const uint group =
      kv_head * IQ36_ADAPTIVE_MAX_PARTITIONS + partition;
  if (head == 0U && lane < IQ36_GQA_GROUP) {
    const ulong meta = (ulong)group * IQ36_GQA_GROUP + lane;
    partial_max[meta] = global_max[lane];
    partial_sum[meta] = global_sum[lane];
  }
  #pragma unroll
  for (uint output_head = 0U; output_head < IQ36_GQA_GROUP;
       ++output_head) {
    const ulong meta =
        ((ulong)group * IQ36_GQA_GROUP + output_head) * IQ36_HEAD_DIM;
    partial_numerator[meta + dim0] = numerator0[output_head];
    partial_numerator[meta + dim1] = numerator1[output_head];
  }
  barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);

  if (local_id == 0U) {
    atomic_work_item_fence(
        CLK_GLOBAL_MEM_FENCE, memory_order_release, memory_scope_device);
    const uint ticket = atomic_inc(
        (volatile __global unsigned int*)&completion[kv_head]);
    is_last_partition = ticket == partition_count - 1U;
    if (is_last_partition != 0U) {
      atomic_work_item_fence(
          CLK_GLOBAL_MEM_FENCE, memory_order_acquire, memory_scope_device);
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (is_last_partition == 0U) return;

  float running_max = -INFINITY;
  float running_sum = 0.0f;
  float final_numerator[16];
  #pragma unroll
  for (uint stripe = 0U; stripe < 16U; ++stripe) {
    final_numerator[stripe] = 0.0f;
  }
  for (uint part = 0U; part < partition_count; ++part) {
    const ulong meta =
        ((ulong)kv_head * IQ36_ADAPTIVE_MAX_PARTITIONS + part) *
            IQ36_GQA_GROUP + head;
    const float part_max = partial_max[meta];
    const float next_max = fmax(running_max, part_max);
    const float previous_scale = native_exp2(
        (running_max - next_max) * IQ36_EXP2_SCALE);
    const float part_scale = native_exp2(
        (part_max - next_max) * IQ36_EXP2_SCALE);
    running_sum = running_sum * previous_scale +
        partial_sum[meta] * part_scale;
    #pragma unroll
    for (uint stripe = 0U; stripe < 16U; ++stripe) {
      const uint dim = lane + stripe * 16U;
      final_numerator[stripe] =
          final_numerator[stripe] * previous_scale +
          partial_numerator[meta * IQ36_HEAD_DIM + dim] * part_scale;
    }
    running_max = next_max;
  }
  const float inverse_sum = native_recip(running_sum);
  #pragma unroll
  for (uint stripe = 0U; stripe < 16U; ++stripe) {
    const uint dim = lane + stripe * 16U;
    attention[(ulong)q_head * IQ36_HEAD_DIM + dim] =
        final_numerator[stripe] * inverse_sum;
  }
}

#else
__attribute__((reqd_work_group_size(128, 1, 1)))
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void iq36_adaptive_attention_correct_normalize(
    IQ36_ADAPTIVE_ABI) {
  const uint local_id = (uint)get_local_id(0);
  const uint head = (uint)get_sub_group_id();
  const uint lane = (uint)get_sub_group_local_id();
  const uint partition = (uint)get_group_id(1);
  const uint batch_kv = (uint)get_group_id(2);
  const uint batch = batch_kv / IQ36_KV_HEADS;
  const uint kv_head = batch_kv - batch * IQ36_KV_HEADS;
  const uint q_head = kv_head * IQ36_GQA_GROUP + head;
  const uint key_tokens = iq36_adaptive_key_tokens(
      decode_length_carrier, batch);
  const uint cold_tokens = iq36_adaptive_cold_tokens(key_tokens);
  const uint cold_chunk_count =
      (cold_tokens + IQ36_CHUNK_TOKENS - 1U) / IQ36_CHUNK_TOKENS;
  if (partition >= cold_chunk_count || cold_chunk_count == 0U) return;
  const uint begin_token = partition * IQ36_CHUNK_TOKENS;
  const uint end_token = min(
      begin_token + (uint)IQ36_CHUNK_TOKENS, cold_tokens);

  __global float* base = iq36_adaptive_workspace_base(workspace, batch);
  __global const half* approximate_score =
      (__global const half*)(base + IQ36_AOFF_APPROXIMATE_SCORE);
  __global const uint* union_bits =
      (__global const uint*)(base + IQ36_AOFF_UNION_BITS);
  __global const float* aggregate_max = base + IQ36_AOFF_AGGREGATE_MAX;
  __global const float* aggregate_sum = base + IQ36_AOFF_AGGREGATE_SUM;
  __global const float* aggregate_numerator =
      base + IQ36_AOFF_AGGREGATE_NUMERATOR;
  __global float* correction_max = base + IQ36_AOFF_CORRECTION_MAX;
  __global float* correction_sum = base + IQ36_AOFF_CORRECTION_SUM;
  __global float* correction_numerator =
      base + IQ36_AOFF_CORRECTION_NUMERATOR;
  __global uint* completion =
      (__global uint*)(base + IQ36_AOFF_COMPLETION);
  __global float* attention = base + IQ36_AOFF_ATTENTION;

  __local ushort selected_indices[IQ36_CHUNK_TOKENS];
  __local uint selected_count;
  __local uint selected_subgroup_offsets[IQ36_GQA_GROUP];
  __local half approximate_value[16U * IQ36_HEAD_DIM];
  __local half exact_value[16U * IQ36_HEAD_DIM];
  __local float approximate_scores[IQ36_GQA_GROUP * 16U];
  __local float exact_scores[IQ36_GQA_GROUP * 16U];
  __local uint is_last_partition;
  const uint token_base = begin_token + local_id * 4U;
  uint selected_mask = 0U;
  if (token_base < end_token) {
    const uint word = union_bits[
        (ulong)kv_head * IQ36_ADAPTIVE_UNION_WORDS + (token_base >> 5U)];
    #pragma unroll
    for (uint offset = 0U; offset < 4U; ++offset) {
      const uint token = token_base + offset;
      // The main scan reads sink rows from exact hot history even after they
      // fall inside the logical cold prefix.  They therefore have no
      // approximate contribution to replace; applying the cold-state
      // correction to them would subtract a value that was never accumulated.
      if (token >= IQ36_SINK_TOKENS && token < end_token &&
          (word & (1U << (token & 31U))) != 0U) {
        selected_mask |= 1U << offset;
      }
    }
  }
  const uint lane_count = popcount(selected_mask);
  uint selected_offset = sub_group_scan_exclusive_add(lane_count);
  const uint subgroup_count = sub_group_reduce_add(lane_count);
  if (lane == 0U) selected_subgroup_offsets[head] = subgroup_count;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (local_id == 0U) {
    uint count = 0U;
    #pragma unroll
    for (uint subgroup = 0U; subgroup < IQ36_GQA_GROUP; ++subgroup) {
      const uint next = selected_subgroup_offsets[subgroup];
      selected_subgroup_offsets[subgroup] = count;
      count += next;
    }
    selected_count = count;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  selected_offset += selected_subgroup_offsets[head];
  #pragma unroll
  for (uint offset = 0U; offset < 4U; ++offset) {
    if ((selected_mask & (1U << offset)) != 0U) {
      selected_indices[selected_offset++] = (ushort)(token_base + offset);
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  half query_fragment[16];
  float delta_numerator[16];
  #pragma unroll
  for (uint slot = 0U; slot < 16U; ++slot) {
    const uint dim = lane + slot * 16U;
    query_fragment[slot] = (half)query[
        INPUT0_OFFSET + (ulong)batch * INPUT0_PITCHES[0] +
        (ulong)q_head * INPUT0_PITCHES[1] +
        (ulong)dim * INPUT0_PITCHES[3]];
    delta_numerator[slot] = 0.0f;
  }
  float local_max = -INFINITY;
  float delta_sum = 0.0f;
  const __global half* query_base = (const __global half*)&query[
      INPUT0_OFFSET + (ulong)batch * INPUT0_PITCHES[0]];
  const __global half* dense_key = (const __global half*)&hot_key_bits[
      iq36_hot_key_dense_i32_base(batch, kv_head)];
  const __global half* dense_value = (const __global half*)&hot_key_bits[
      iq36_hot_value_dimension_i32_base(batch, kv_head)];
  for (uint selected_begin = 0U; selected_begin < selected_count;
       selected_begin += 16U) {
    const uint rows = min(16U, selected_count - selected_begin);
    for (uint index = local_id; index < rows * IQ36_HEAD_DIM;
         index += 128U) {
      const uint row = index / IQ36_HEAD_DIM;
      const uint dim = index - row * IQ36_HEAD_DIM;
      const uint token = (uint)selected_indices[selected_begin + row];
      const uint slot = iq36_hot_slot(token);
      exact_value[index] = dense_key[(ulong)slot * IQ36_HEAD_DIM + dim];
      // Reuse the value-replacement SLM plane as an approximate-K tile until
      // the score correction is complete.  The partial stage accumulates the
      // unrounded grouped-I8 score, while its F16 score sidecar exists only to
      // rank candidates.  Subtracting a weight reconstructed from that F16
      // sidecar leaves a first-order residual on every selected row.  Rebuild
      // the same F16 dequantized K element consumed by the partial DPAS so the
      // replacement subtracts the contribution that was actually accumulated.
      const char quantized_key = iq36_direct_cold_key_element(
          batch, kv_head, token, dim, cold_key);
      const half key_scale = iq36_direct_cold_key_scale(
          batch, kv_head, token, dim / IQ36_KEY_QUANT_GROUP,
          cold_key_scale_bytes);
      half reconstructed_key = convert_half(quantized_key);
#if defined(IQ36_KEY_RESIDUAL1)
      reconstructed_key += iq36_direct_cold_key_residual1_element(
          batch, kv_head, token, dim, cold_key_scale_bytes);
#endif
      approximate_value[index] = reconstructed_key * key_scale;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // One subgroup evaluates the approximate 16-token correction tile for all
    // eight query heads.  This is the same 8x16-by-16x16 systolic carrier as
    // the main partial scan, so the replacement subtracts the score arithmetic
    // that the aggregate actually consumed.  Keep the exact score on the
    // already-validated scalar path while this carrier is admitted.
    if (head == 0U) {
      float8 approximate_score_tile = (float8)(0.0f);
      #pragma unroll 1
      for (uint k_block = 0U;
           k_block < IQ36_HEAD_DIM / IQ36_TOKEN_TILE; ++k_block) {
        const uint k_base = k_block * IQ36_TOKEN_TILE;
        const short8 query_tile = as_short8(
            iq36_block2d_load_f16_16x8(
                query_base,
                (int)INPUT0_DIMS[3] * (int)sizeof(half),
                (int)INPUT0_DIMS[1],
                (int)INPUT0_PITCHES[1] * (int)sizeof(half),
                k_base, kv_head * IQ36_GQA_GROUP));
        int8 approximate_key_tile;
        #pragma unroll
        for (uint pair = 0U; pair < 8U; ++pair) {
          const uint dim = k_base + pair * 2U;
          half2 approximate_key = (half2)(0.0h);
          if (lane < rows) {
            const uint row_base = lane * IQ36_HEAD_DIM + dim;
            approximate_key = (half2)(
                approximate_value[row_base],
                approximate_value[row_base + 1U]);
          }
          approximate_key_tile[pair] = as_int(approximate_key);
        }
        approximate_score_tile = intel_sub_group_f16_f16_matrix_mad_k16(
            query_tile, approximate_key_tile, approximate_score_tile);
      }
      if (lane < rows) {
        #pragma unroll
        for (uint query_head = 0U;
             query_head < IQ36_GQA_GROUP; ++query_head) {
          approximate_scores[query_head * 16U + lane] =
              approximate_score_tile[query_head];
        }
      }
    }
    for (uint row = 0U; row < rows; ++row) {
      float exact_score = 0.0f;
      #pragma unroll
      for (uint slot = 0U; slot < 16U; ++slot) {
        const uint dim = lane + slot * 16U;
        exact_score = fma(convert_float(query_fragment[slot]),
            convert_float(exact_value[row * IQ36_HEAD_DIM + dim]),
            exact_score);
      }
      exact_score = sub_group_reduce_add(exact_score);
      if (lane == 0U) {
        exact_scores[head * 16U + row] = exact_score;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint index = local_id; index < rows * IQ36_HEAD_DIM;
         index += 128U) {
      const uint row = index / IQ36_HEAD_DIM;
      const uint dim = index - row * IQ36_HEAD_DIM;
      const uint token = (uint)selected_indices[selected_begin + row];
      const uint slot = iq36_hot_slot(token);
      approximate_value[index] = iq36_direct_cold_value_element(
          batch, kv_head, token, dim, cold_value,
          cold_value_scale_bytes);
      exact_value[index] = dense_value[
          (ulong)dim * (uint)INPUT2_DIMS[2] + slot];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint row = 0U; row < rows; ++row) {
      const float approximate = approximate_scores[head * 16U + row];
      const float exact = exact_scores[head * 16U + row];
      const float next_max = fmax(local_max, fmax(approximate, exact));
      const float previous_scale = native_exp2(
          (local_max - next_max) * IQ36_EXP2_SCALE);
      const float approximate_weight = native_exp2(
          (approximate - next_max) * IQ36_EXP2_SCALE);
      const float exact_weight = native_exp2(
          (exact - next_max) * IQ36_EXP2_SCALE);
      #pragma unroll
      for (uint slot = 0U; slot < 16U; ++slot) {
        const uint dim = lane + slot * 16U;
        delta_numerator[slot] = delta_numerator[slot] * previous_scale +
            exact_weight * convert_float(
                exact_value[row * IQ36_HEAD_DIM + dim]) -
            approximate_weight * convert_float(
                approximate_value[row * IQ36_HEAD_DIM + dim]);
      }
      delta_sum = delta_sum * previous_scale +
          exact_weight - approximate_weight;
      local_max = next_max;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  const ulong partial_meta =
      ((ulong)kv_head * IQ36_ADAPTIVE_MAX_COLD_CHUNKS + partition) *
          IQ36_GQA_GROUP + head;
  if (lane == 0U) {
    correction_max[partial_meta] = local_max;
    correction_sum[partial_meta] = delta_sum;
  }
  #pragma unroll
  for (uint slot = 0U; slot < 16U; ++slot) {
    const uint dim = lane + slot * 16U;
    correction_numerator[partial_meta * IQ36_HEAD_DIM + dim] =
        delta_numerator[slot];
  }
  barrier(CLK_GLOBAL_MEM_FENCE);
  if (local_id == 0U) {
    atomic_work_item_fence(
        CLK_GLOBAL_MEM_FENCE, memory_order_seq_cst, memory_scope_device);
    const uint ticket = atomic_inc(
        (volatile __global unsigned int*)&completion[kv_head]);
    atomic_work_item_fence(
        CLK_GLOBAL_MEM_FENCE, memory_order_seq_cst, memory_scope_device);
    is_last_partition = ticket == cold_chunk_count - 1U;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (is_last_partition != 0U) {
    atomic_work_item_fence(
        CLK_GLOBAL_MEM_FENCE, memory_order_seq_cst, memory_scope_device);
    float final_max = aggregate_max[q_head];
    for (uint part = 0U; part < cold_chunk_count; ++part) {
      const ulong meta =
          ((ulong)kv_head * IQ36_ADAPTIVE_MAX_COLD_CHUNKS + part) *
              IQ36_GQA_GROUP + head;
      final_max = fmax(final_max, correction_max[meta]);
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
    for (uint part = 0U; part < cold_chunk_count; ++part) {
      const ulong meta =
          ((ulong)kv_head * IQ36_ADAPTIVE_MAX_COLD_CHUNKS + part) *
              IQ36_GQA_GROUP + head;
      const float scale = native_exp2(
          (correction_max[meta] - final_max) * IQ36_EXP2_SCALE);
      final_sum += correction_sum[meta] * scale;
      #pragma unroll
      for (uint slot = 0U; slot < 16U; ++slot) {
        const uint dim = lane + slot * 16U;
        final_numerator[slot] += correction_numerator[
            meta * IQ36_HEAD_DIM + dim] * scale;
      }
    }
    #pragma unroll
    for (uint slot = 0U; slot < 16U; ++slot) {
      const uint dim = lane + slot * 16U;
      attention[(ulong)q_head * IQ36_HEAD_DIM + dim] =
          final_numerator[slot] * native_recip(final_sum);
    }
  }
}

#endif

#elif IQ36_ADAPTIVE_STAGE == IQ36_ADAPTIVE_STAGE_UPDATE

__attribute__((reqd_work_group_size(128, 1, 1)))
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void iq36_adaptive_attention_ordered_update(IQ36_ADAPTIVE_ABI) {
  const uint local_id = (uint)get_local_id(0);
  const uint subgroup = (uint)get_sub_group_id();
  const uint lane = (uint)get_sub_group_local_id();
  const uint kv_head = (uint)get_group_id(1);
  const uint batch = (uint)get_group_id(2);
  const uint key_tokens = iq36_adaptive_key_tokens(
      decode_length_carrier, batch);
  const uint past_tokens = key_tokens - 1U;
  const uint desired_cold_tokens = key_tokens > IQ36_HOT_WINDOW
      ? key_tokens - IQ36_HOT_WINDOW : 0U;
  const uint cold_append_tokens =
      (uint)eviction_count[INPUT11_OFFSET];
  const uint cold_tokens = desired_cold_tokens - cold_append_tokens;
  __global float* base = iq36_adaptive_workspace_base(workspace, batch);
  __global const float* attention = base + IQ36_AOFF_ATTENTION;

  for (uint index = local_id;
       index < IQ36_GQA_GROUP * IQ36_HEAD_DIM; index += 128U) {
    const uint head = index / IQ36_HEAD_DIM;
    const uint dim = index - head * IQ36_HEAD_DIM;
    const uint q_head = kv_head * IQ36_GQA_GROUP + head;
    output[OUTPUT1_OFFSET + (ulong)batch * OUTPUT1_PITCHES[0] +
        (ulong)q_head * OUTPUT1_PITCHES[1] +
        (ulong)dim * OUTPUT1_PITCHES[3]] = (OUTPUT1_TYPE)convert_half_rte(
            attention[(ulong)q_head * IQ36_HEAD_DIM + dim]);
  }

  const uint state_dim0 = local_id * 2U;
  const uint state_dim1 = state_dim0 + 1U;
  if (cold_append_tokens != 0U) {
    const uint block = subgroup;
    const float key0 = iq36_partial_load_key(
        batch, kv_head, cold_tokens, past_tokens, cold_tokens,
        state_dim0, hot_key_bits, current_key, cold_key,
        cold_key_scale_bytes);
    const float key1 = iq36_partial_load_key(
        batch, kv_head, cold_tokens, past_tokens, cold_tokens,
        state_dim1, hot_key_bits, current_key, cold_key,
        cold_key_scale_bytes);
    const float value0 = iq36_partial_load_value(
        batch, kv_head, cold_tokens, past_tokens, cold_tokens,
        state_dim0, hot_value, current_value, cold_value,
        cold_value_scale_bytes);
    const float value1 = iq36_partial_load_value(
        batch, kv_head, cold_tokens, past_tokens, cold_tokens,
        state_dim1, hot_value, current_value, cold_value,
        cold_value_scale_bytes);
    const float key_max = sub_group_reduce_max(
        fmax(fabs(key0), fabs(key1)));
    float value_max = fmax(fabs(value0), fabs(value1));
#if IQ36_VALUE_QUANT_GROUP == 16U
    // Each half subgroup owns one 16-dimension V scale while the full
    // subgroup continues to own one 32-dimension K scale.
    value_max = fmax(
        value_max, sub_group_shuffle_xor(value_max, 1U));
    value_max = fmax(
        value_max, sub_group_shuffle_xor(value_max, 2U));
    value_max = fmax(
        value_max, sub_group_shuffle_xor(value_max, 4U));
#else
    value_max = sub_group_reduce_max(value_max);
#endif
    const float key_scale = key_max == 0.0f
        ? 1.0f : key_max / (float)IQ36_KEY_QUANT_MAX;
    const float value_scale =
        value_max == 0.0f
        ? 1.0f : value_max / (float)IQ36_VALUE_QUANT_MAX;
#if defined(IQ36_KEY_RESIDUAL1)
    const half key_scale_stored = convert_half_rte(key_scale);
    const int key_fine0 = iq36_residual1_fine_quantize(
        key0, key_scale_stored);
    const int key_fine1 = iq36_residual1_fine_quantize(
        key1, key_scale_stored);
    const int key_q0 = iq36_residual1_base(key_fine0);
    const int key_q1 = iq36_residual1_base(key_fine1);
    const uint key_residual_bit0 = iq36_residual1_bit(key_fine0, key_q0);
    const uint key_residual_bit1 = iq36_residual1_bit(key_fine1, key_q1);
    const uint key_residual_word = sub_group_reduce_add(
        (key_residual_bit0 << (lane * 2U)) |
        (key_residual_bit1 << (lane * 2U + 1U)));
#else
    const int key_q0 = clamp(
        (int)rint(key0 / key_scale),
        -(int)IQ36_KEY_QUANT_MAX, (int)IQ36_KEY_QUANT_MAX);
    const int key_q1 = clamp(
        (int)rint(key1 / key_scale),
        -(int)IQ36_KEY_QUANT_MAX, (int)IQ36_KEY_QUANT_MAX);
#endif
#if defined(IQ36_VALUE_RESIDUAL1)
    const half value_scale_stored = convert_half_rte(value_scale);
    const int value_fine0 = iq36_residual1_fine_quantize(
        value0, value_scale_stored);
    const int value_fine1 = iq36_residual1_fine_quantize(
        value1, value_scale_stored);
    const int value_q0 = iq36_residual1_base(value_fine0);
    const int value_q1 = iq36_residual1_base(value_fine1);
    const uint value_residual_bit0 = iq36_residual1_bit(
        value_fine0, value_q0);
    const uint value_residual_bit1 = iq36_residual1_bit(
        value_fine1, value_q1);
    const uint value_residual_word = sub_group_reduce_add(
        (value_residual_bit0 << (lane * 2U)) |
        (value_residual_bit1 << (lane * 2U + 1U)));
#else
    const int value_q0 = clamp(
        (int)rint(value0 / value_scale),
        -(int)IQ36_VALUE_QUANT_MAX, (int)IQ36_VALUE_QUANT_MAX);
    const int value_q1 = clamp(
        (int)rint(value1 / value_scale),
        -(int)IQ36_VALUE_QUANT_MAX, (int)IQ36_VALUE_QUANT_MAX);
#endif
    const ulong key_append_base = OUTPUT2_OFFSET +
        (ulong)batch * OUTPUT2_PITCHES[0] +
        (ulong)kv_head * OUTPUT2_PITCHES[1];
    const ulong value_append_base = OUTPUT3_OFFSET +
        (ulong)batch * OUTPUT3_PITCHES[0] +
        (ulong)kv_head * OUTPUT3_PITCHES[1];
#if defined(IQ36_ADAPTIVE_PACKED_KV)
    #pragma unroll
    for (uint word = 0U; word < IQ36_KEY_QUANT_BITS; ++word) {
      const uint packed = iq36_subgroup_pack_two_codes(
          key_q0, key_q1, IQ36_KEY_QUANT_BITS, word);
      if (lane < 4U) {
        cold_key_append[key_append_base +
            (ulong)((block * IQ36_KEY_QUANT_BITS + word) * 4U + lane) *
                OUTPUT2_PITCHES[3]] =
            (OUTPUT2_TYPE)(packed >> (lane * 8U));
      }
      if (lane == 0U) {
        iq36_direct_store_cold_key_packed_word(
            batch, kv_head, cold_tokens, block, word, packed, cold_key);
      }
    }
    #pragma unroll
    for (uint word = 0U; word < IQ36_VALUE_QUANT_BITS; ++word) {
      const uint packed = iq36_subgroup_pack_two_codes(
          value_q0, value_q1, IQ36_VALUE_QUANT_BITS, word);
      if (lane < 4U) {
        cold_value_append[value_append_base +
            (ulong)((block * IQ36_VALUE_QUANT_BITS + word) * 4U + lane) *
                OUTPUT3_PITCHES[3]] =
            (OUTPUT3_TYPE)(packed >> (lane * 8U));
      }
#if !defined(IQ36_ADAPTIVE_DIMENSION_MAJOR_V8)
      if (lane == 0U) {
        iq36_direct_store_cold_value_packed_word(
            batch, kv_head, cold_tokens, block, word, packed, cold_value);
      }
#endif
    }
#if defined(IQ36_ADAPTIVE_DIMENSION_MAJOR_V8)
    iq36_direct_store_cold_value(
        batch, kv_head, cold_tokens, state_dim0,
        (char)value_q0, cold_value);
    iq36_direct_store_cold_value(
        batch, kv_head, cold_tokens, state_dim1,
        (char)value_q1, cold_value);
#endif
#else
    cold_key_append[key_append_base +
        (ulong)state_dim0 * OUTPUT2_PITCHES[3]] = (OUTPUT2_TYPE)key_q0;
    cold_key_append[key_append_base +
        (ulong)state_dim1 * OUTPUT2_PITCHES[3]] = (OUTPUT2_TYPE)key_q1;
    cold_value_append[value_append_base +
        (ulong)state_dim0 * OUTPUT3_PITCHES[3]] =
            (OUTPUT3_TYPE)value_q0;
    cold_value_append[value_append_base +
        (ulong)state_dim1 * OUTPUT3_PITCHES[3]] =
            (OUTPUT3_TYPE)value_q1;
    iq36_direct_store_cold_key(
        batch, kv_head, cold_tokens, state_dim0, (char)key_q0, cold_key);
    iq36_direct_store_cold_key(
        batch, kv_head, cold_tokens, state_dim1, (char)key_q1, cold_key);
    iq36_direct_store_cold_value(
        batch, kv_head, cold_tokens, state_dim0,
        (char)value_q0, cold_value);
    iq36_direct_store_cold_value(
        batch, kv_head, cold_tokens, state_dim1,
        (char)value_q1, cold_value);
#endif
#if defined(IQ36_KEY_RESIDUAL1)
    if (lane < 4U) {
      const ulong residual_index = OUTPUT4_OFFSET +
          (ulong)batch * OUTPUT4_PITCHES[0] +
          (ulong)kv_head * OUTPUT4_PITCHES[1] +
          (ulong)(IQ36_KEY_SCALE_BYTES + block * 4U + lane) *
              OUTPUT4_PITCHES[3];
      cold_key_scale_append[residual_index] =
          (OUTPUT4_TYPE)(key_residual_word >> (lane * 8U));
    }
    if (lane == 0U) {
      iq36_direct_store_cold_key_residual1_word(
          batch, kv_head, cold_tokens, block, key_residual_word,
          cold_key_scale_bytes);
    }
#endif
#if defined(IQ36_VALUE_RESIDUAL1)
    if (lane < 4U) {
      const ulong residual_index = OUTPUT5_OFFSET +
          (ulong)batch * OUTPUT5_PITCHES[0] +
          (ulong)kv_head * OUTPUT5_PITCHES[1] +
          (ulong)(IQ36_VALUE_SCALE_BYTES + block * 4U + lane) *
              OUTPUT5_PITCHES[3];
      cold_value_scale_append[residual_index] =
          (OUTPUT5_TYPE)(value_residual_word >> (lane * 8U));
    }
    iq36_direct_store_cold_value_residual1_bit(
        batch, kv_head, cold_tokens, state_dim0,
        value_residual_bit0, cold_value_scale_bytes);
    iq36_direct_store_cold_value_residual1_bit(
        batch, kv_head, cold_tokens, state_dim1,
        value_residual_bit1, cold_value_scale_bytes);
#endif
    if (lane < 2U) {
      const ushort key_bits = as_ushort(convert_half_rte(key_scale));
      const uint scale_x = block * 2U + lane;
      const ulong key_scale_index = OUTPUT4_OFFSET +
          (ulong)batch * OUTPUT4_PITCHES[0] +
          (ulong)kv_head * OUTPUT4_PITCHES[1] +
          (ulong)scale_x * OUTPUT4_PITCHES[3];
      cold_key_scale_append[key_scale_index] = (OUTPUT4_TYPE)(
          lane == 0U ? key_bits & 0xffU : key_bits >> 8);
      if (lane == 0U) {
        iq36_direct_store_cold_key_scale(
            batch, kv_head, cold_tokens, block,
            as_half(key_bits), cold_key_scale_bytes);
      }
    }
    const uint value_scale_group =
        block * (32U / IQ36_VALUE_QUANT_GROUP) +
        lane / (IQ36_VALUE_QUANT_GROUP / 2U);
    const uint value_scale_byte_lane =
        lane % (IQ36_VALUE_QUANT_GROUP / 2U);
    if (value_scale_byte_lane < 2U) {
      const ushort value_bits = as_ushort(convert_half_rte(value_scale));
      const uint scale_x = value_scale_group * 2U +
          value_scale_byte_lane;
      const ulong value_scale_index = OUTPUT5_OFFSET +
          (ulong)batch * OUTPUT5_PITCHES[0] +
          (ulong)kv_head * OUTPUT5_PITCHES[1] +
          (ulong)scale_x * OUTPUT5_PITCHES[3];
      cold_value_scale_append[value_scale_index] = (OUTPUT5_TYPE)(
          value_scale_byte_lane == 0U
              ? value_bits & 0xffU : value_bits >> 8);
      if (value_scale_byte_lane == 0U) {
        iq36_direct_store_cold_value_scale(
            batch, kv_head, cold_tokens, value_scale_group,
            as_half(value_bits), cold_value_scale_bytes);
      }
    }
  }

  if (local_id < 3U) {
    const uint divisor = local_id == 0U ? 1U :
        (local_id == 1U ? 128U : 16384U);
    cold_key[INPUT5_OFFSET + (ulong)batch * INPUT5_PITCHES[0] +
        (ulong)kv_head * INPUT5_PITCHES[1] +
        (ulong)local_id * INPUT5_PITCHES[3]] =
            (INPUT5_TYPE)((desired_cold_tokens / divisor) % 128U);
  }

  const uint slot = iq36_hot_slot(past_tokens);
  const ulong current_key_base = INPUT3_OFFSET +
      (ulong)batch * INPUT3_PITCHES[0] +
      (ulong)kv_head * INPUT3_PITCHES[1];
  const ulong current_value_base = iq36_current_value_index(
      batch, kv_head, 0U, 0U);
  const uint key_word = local_id * IQ36_KEY_TILE_TOKENS +
      (slot & (IQ36_KEY_TILE_TOKENS - 1U));
  const ulong hot_key_index = INPUT1_OFFSET +
      (ulong)batch * INPUT1_PITCHES[0] +
      (ulong)kv_head * INPUT1_PITCHES[1] +
      (ulong)(slot / IQ36_KEY_TILE_TOKENS) * INPUT1_PITCHES[2] +
      (ulong)key_word * INPUT1_PITCHES[3];
  hot_key_bits[hot_key_index] = (INPUT1_TYPE)as_int((half2)(
      (half)current_key[current_key_base +
          (ulong)state_dim0 * INPUT3_PITCHES[3]],
      (half)current_key[current_key_base +
          (ulong)state_dim1 * INPUT3_PITCHES[3]]));
  __global half* dense_key = (__global half*)&hot_key_bits[
      iq36_hot_key_dense_i32_base(batch, kv_head)];
  dense_key[(ulong)slot * IQ36_HEAD_DIM + state_dim0] =
      (half)current_key[current_key_base +
          (ulong)state_dim0 * INPUT3_PITCHES[3]];
  dense_key[(ulong)slot * IQ36_HEAD_DIM + state_dim1] =
      (half)current_key[current_key_base +
          (ulong)state_dim1 * INPUT3_PITCHES[3]];
  const half value0 = (half)current_value[current_value_base +
      (ulong)state_dim0 * INPUT4_PITCHES[3]];
  const half value1 = (half)current_value[current_value_base +
      (ulong)state_dim1 * INPUT4_PITCHES[3]];
  const ulong hot_value_base = INPUT2_OFFSET +
      (ulong)batch * INPUT2_PITCHES[0] +
      (ulong)kv_head * INPUT2_PITCHES[1] +
      (ulong)slot * INPUT2_PITCHES[2];
  hot_value[hot_value_base +
      (ulong)state_dim0 * INPUT2_PITCHES[3]] = (INPUT2_TYPE)value0;
  hot_value[hot_value_base +
      (ulong)state_dim1 * INPUT2_PITCHES[3]] = (INPUT2_TYPE)value1;
  iq36_direct_store_hot_value_dimension(
      batch, kv_head, slot, state_dim0, value0, hot_key_bits);
  iq36_direct_store_hot_value_dimension(
      batch, kv_head, slot, state_dim1, value1, hot_key_bits);
}

#else
#error "unknown IQ36 adaptive graph stage"
#endif

#undef IQ36_ADAPTIVE_ABI
#endif
