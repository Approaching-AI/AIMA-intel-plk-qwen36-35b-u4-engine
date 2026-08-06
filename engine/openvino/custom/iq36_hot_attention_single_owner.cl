// Single-dispatch, single-state-owner tiled attention for Qwen3.6 full-GQA.
//
// Work geometry comes from workspace OUTPUT0 [B,2,G,2066]:
//   prefill G = ceil(Q/32) * 8 (one query-head tile per work-group)
//   decode  G >= ceil(K/512)   (one active KV chunk per work-group; fixed
//                               carriers may launch inactive tail groups)
// OUTPUT0 is a private F32 workspace. Decode work-groups publish
// partial max/sum/numerators there; the last atomic arrival performs the final
// reduction and the sole request-state update. No custom node fans out the hot
// state inputs.

// The unified XML concatenates a prefill-only implementation before this
// source.  For concrete prefill shapes that implementation owns the common
// entry point; this file contributes only the decode specialization.
#if !defined(IQ36_UNIFIED_SHAPE_SPECIALIZATION) || \
    (defined(IQ36_BUILD_DECODE_ONLY) && \
     !defined(IQ36_ADAPTIVE_ATTENTION_GRAPH))

#define IQ36_WORKSPACE_PREFIX 2U
#define IQ36_WORKSPACE_WIDTH \
    (IQ36_WORKSPACE_PREFIX + IQ36_PARTIAL_KV_WIDTH)
// The 128k product bucket needs 258 work-groups at the default 512-token
// tile and 514 at the opt-in stock-order 256-token tile.
#define IQ36_MAX_DECODE_CHUNKS 514U
#define IQ36_DECODE_BLOCK_ROWS \
    (IQ36_BLOCKS_PER_CHUNK * IQ36_GQA_GROUP)
#define IQ36_DECODE_AUX_FLOATS \
    (2U * IQ36_DECODE_BLOCK_ROWS + \
     2U * IQ36_DECODE_REDUCTION_PARTS * IQ36_GQA_GROUP + 1U)
#define IQ36_PREFILL_AUX_FLOATS \
    (IQ36_PREFILL_BLOCKS_PER_CHUNK * IQ36_PREFILL_QUERY_TILE + \
     3U * IQ36_PREFILL_QUERY_TILE + \
     IQ36_PREFILL_BLOCKS_PER_CHUNK * IQ36_PREFILL_QUERY_TILE)
#if defined(IQ36_BUILD_DECODE_ONLY)
#define IQ36_LOCAL_AUX_FLOATS IQ36_DECODE_AUX_FLOATS
#else
#define IQ36_LOCAL_AUX_FLOATS IQ36_PREFILL_AUX_FLOATS
#endif

inline ulong iq36_workspace_row(
    const uint batch, const uint kv_head, const uint group) {
  return OUTPUT0_OFFSET +
      (ulong)batch * OUTPUT0_PITCHES[0] +
      (ulong)kv_head * OUTPUT0_PITCHES[1] +
      (ulong)group * OUTPUT0_PITCHES[2];
}

inline ulong iq36_workspace_head(
    const uint batch, const uint kv_head, const uint group,
    const uint gqa_head) {
  return iq36_workspace_row(batch, kv_head, group) +
      (IQ36_WORKSPACE_PREFIX +
       gqa_head * IQ36_PARTIAL_HEAD_WIDTH) * OUTPUT0_PITCHES[3];
}

#if defined(IQ36_STOCK256_PARTIALS)
inline ulong iq36_workspace_stock_partial_head(
    const uint batch, const uint kv_head, const uint group,
    const uint gqa_head, const uint partial) {
  return iq36_workspace_row(batch, kv_head, group) +
      (IQ36_WORKSPACE_PREFIX +
       (gqa_head * 2U + partial) * IQ36_PARTIAL_HEAD_WIDTH) *
          OUTPUT0_PITCHES[3];
}
#endif

inline ulong iq36_arrival_counter_index(
    const uint batch, const uint kv_head) {
  return INPUT1_OFFSET +
      (ulong)batch * INPUT1_PITCHES[0] +
      (ulong)kv_head * INPUT1_PITCHES[1] +
      (ulong)(INPUT1_DIMS[2] - 1U) * INPUT1_PITCHES[2] +
      (ulong)(INPUT1_DIMS[3] - 1U) * INPUT1_PITCHES[3];
}

__attribute__((reqd_work_group_size(128, 1, 1)))
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void iq36_hot_attention_single_owner(
    const __global INPUT0_TYPE* query,
    __global INPUT1_TYPE* hot_key_bits,
    __global INPUT2_TYPE* hot_value,
    const __global INPUT3_TYPE* current_key,
    const __global INPUT4_TYPE* current_value,
    __global INPUT5_TYPE* cold_key,
    __global INPUT6_TYPE* cold_value,
    __global INPUT7_TYPE* cold_key_scale_bytes,
    __global INPUT8_TYPE* cold_value_scale_bytes,
    const __global INPUT9_TYPE* mask,
    const __global INPUT10_TYPE* eviction_shape_template,
    const __global INPUT11_TYPE* eviction_count,
    const __global INPUT12_TYPE* decode_length_carrier,
#if defined(IQ36_FUSED_GATE_OUTPUT)
    const __global INPUT13_TYPE* raw_gate,
#endif
    __global OUTPUT0_TYPE* workspace,
    __global OUTPUT1_TYPE* output,
    __global OUTPUT2_TYPE* cold_key_append,
    __global OUTPUT3_TYPE* cold_value_append,
    __global OUTPUT4_TYPE* cold_key_scale_append,
    __global OUTPUT5_TYPE* cold_value_scale_append) {
  const uint local_id = (uint)get_local_id(0);
  const uint subgroup = (uint)get_sub_group_id();
  const uint lane = (uint)get_sub_group_local_id();
  const uint work_group = (uint)get_group_id(1);
  const uint batch_kv = (uint)get_group_id(2);
  const uint batch = batch_kv / IQ36_KV_HEADS;
  const uint kv_head = batch_kv - batch * IQ36_KV_HEADS;
#if defined(IQ36_BUILD_PREFILL_ONLY)
  const uint query_tokens = IQ36_STATIC_QUERY_TOKENS;
  const uint key_tokens = IQ36_STATIC_MASK_TOKENS;
#else
  const uint query_tokens = 1U;
  const uint key_tokens =
      (uint)decode_length_carrier[INPUT12_OFFSET];
#endif
  const uint past_tokens = key_tokens - query_tokens;
  const uint cold_append_tokens = (uint)eviction_count[INPUT11_OFFSET];
  const uint desired_cold_tokens = key_tokens > IQ36_HOT_WINDOW
      ? key_tokens - IQ36_HOT_WINDOW : 0U;
  const uint cold_tokens = desired_cold_tokens - cold_append_tokens;
  const bool fixed_cold_state =
      (uint)INPUT5_DIMS[2] >= desired_cold_tokens + 1U;

  // Prefill and decode are mutually exclusive, so overlay their local-memory
  // carriers.  This preserves the tiled prefill occupancy in the unified op.
  __local float local_score_weight[
      IQ36_GQA_GROUP * IQ36_CHUNK_TOKENS];
  __local float local_aux[IQ36_LOCAL_AUX_FLOATS];
  __local float* prefill_block_sum = &local_aux[0];
  __local float* prefill_running_max = &local_aux[256];
  __local float* prefill_running_sum = &local_aux[288];
  __local float* prefill_previous_scale = &local_aux[320];
  __local float* prefill_block_running_sum = &local_aux[352];
  __local float* local_block_max = &local_aux[0];
  __local float* local_block_sum = &local_aux[IQ36_DECODE_BLOCK_ROWS];
  __local float* local_global_max = &local_aux[
      2U * IQ36_DECODE_BLOCK_ROWS];
  __local float* local_global_sum = &local_aux[
      2U * IQ36_DECODE_BLOCK_ROWS +
      IQ36_DECODE_REDUCTION_PARTS * IQ36_GQA_GROUP];
  __local uint* local_is_last = (__local uint*)&local_aux[
      2U * IQ36_DECODE_BLOCK_ROWS +
      2U * IQ36_DECODE_REDUCTION_PARTS * IQ36_GQA_GROUP];

#if !defined(IQ36_BUILD_DECODE_ONLY)
  if (query_tokens != 1U) {
    // Map [B,2,ceil(Q/32)*8] back to the stock-shaped
    // [B,16,ceil(Q/32)] query-head tiles.
    const uint query_head_in_kv = work_group % IQ36_GQA_GROUP;
    const uint query_tile = work_group / IQ36_GQA_GROUP;
    const uint query_head =
        kv_head * IQ36_GQA_GROUP + query_head_in_kv;
    const uint query_begin = query_tile * IQ36_PREFILL_QUERY_TILE;
    const uint query_count = query_begin < query_tokens
        ? min((uint)IQ36_PREFILL_QUERY_TILE,
              query_tokens - query_begin)
        : 0U;
    if (query_count == 0U) return;

    if (local_id < query_count) {
      prefill_running_max[local_id] = -INFINITY;
      prefill_running_sum[local_id] = 0.0f;
    }
    for (uint index = local_id;
         index < IQ36_PREFILL_BLOCKS_PER_CHUNK *
                     IQ36_PREFILL_QUERY_TILE;
         index += 128U) {
      prefill_block_running_sum[index] = 0.0f;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    float8 output_accumulator0[IQ36_PREFILL_QUERY_GROUPS];
    float8 output_accumulator1[IQ36_PREFILL_QUERY_GROUPS];
    #pragma unroll
    for (uint group = 0U; group < IQ36_PREFILL_QUERY_GROUPS; ++group) {
      output_accumulator0[group] = (float8)(0.0f);
      output_accumulator1[group] = (float8)(0.0f);
    }
    const uint causal_tokens = past_tokens + query_begin + query_count;
    const __global half* query_base =
        (const __global half*)&query[
            INPUT0_OFFSET + (ulong)batch * INPUT0_PITCHES[0] +
            (ulong)query_head * INPUT0_PITCHES[1]];
    const __global half* value_base =
        (const __global half*)&current_value[
            iq36_current_value_index(batch, kv_head, 0U, 0U)];

    for (uint chunk_begin = 0U; chunk_begin < causal_tokens;
         chunk_begin += IQ36_PREFILL_CHUNK_TOKENS) {
      const uint chunk_tokens = min(
          (uint)IQ36_PREFILL_CHUNK_TOKENS,
          causal_tokens - chunk_begin);
      const uint valid_blocks =
          (chunk_tokens + IQ36_TOKEN_TILE - 1U) / IQ36_TOKEN_TILE;

      if (local_id < query_count) {
        prefill_previous_scale[local_id] = prefill_running_max[local_id];
      }
      if (subgroup < valid_blocks) {
        const uint token_in_chunk = subgroup * IQ36_TOKEN_TILE + lane;
        const uint token = chunk_begin + token_in_chunk;
        float8 score[IQ36_PREFILL_QUERY_GROUPS];
        #pragma unroll
        for (uint group = 0U; group < IQ36_PREFILL_QUERY_GROUPS; ++group) {
          score[group] = (float8)(0.0f);
        }
        #pragma unroll 1
        for (uint k_block = 0U;
             k_block < IQ36_HEAD_DIM / IQ36_TOKEN_TILE; ++k_block) {
          const uint k_base = k_block * IQ36_TOKEN_TILE;
          const uint block_token =
              chunk_begin + subgroup * IQ36_TOKEN_TILE;
          const uint block_slot = iq36_hot_slot(block_token);
          const bool prior_hot_contiguous =
              block_token + IQ36_TOKEN_TILE <= past_tokens &&
              (cold_tokens == 0U || block_token >= cold_tokens) &&
              (block_slot & (IQ36_KEY_TILE_TOKENS - 1U)) == 0U &&
              block_slot + IQ36_TOKEN_TILE <= (uint)INPUT2_DIMS[2];
          const bool prior_cold_contiguous =
              block_token >= IQ36_SINK_TOKENS &&
              block_token + IQ36_TOKEN_TILE <= cold_tokens;
          const bool current_contiguous = block_token >= past_tokens;
          int8 key_fragment;
          if (prior_hot_contiguous) {
            const ulong key_index = INPUT1_OFFSET +
                (ulong)batch * INPUT1_PITCHES[0] +
                (ulong)kv_head * INPUT1_PITCHES[1] +
                (ulong)(block_slot / IQ36_KEY_TILE_TOKENS) *
                    INPUT1_PITCHES[2] +
                (ulong)(k_block * 8U * IQ36_KEY_TILE_TOKENS) *
                    INPUT1_PITCHES[3];
            key_fragment = as_int8(intel_sub_group_block_read8(
                (const __global uint*)&hot_key_bits[key_index]));
          } else if (prior_cold_contiguous) {
            key_fragment = iq36_cold_key_fragment(
                batch, kv_head, token, k_base,
                cold_key, cold_key_scale_bytes);
          } else if (current_contiguous) {
            const uint current_token = token - past_tokens;
            const ulong key_index = INPUT3_OFFSET +
                (ulong)batch * INPUT3_PITCHES[0] +
                (ulong)kv_head * INPUT3_PITCHES[1] +
                (ulong)current_token * INPUT3_PITCHES[2] +
                (ulong)k_base * INPUT3_PITCHES[3];
            key_fragment = as_int8(vload16(
                0, (const __global half*)&current_key[key_index]));
          } else {
            half16 values;
            #pragma unroll
            for (uint offset = 0U; offset < IQ36_TOKEN_TILE; ++offset) {
              values[offset] = convert_half_rte(iq36_partial_load_key(
                  batch, kv_head, token, past_tokens, cold_tokens,
                  k_base + offset, hot_key_bits, current_key, cold_key,
                  cold_key_scale_bytes));
            }
            key_fragment = as_int8(values);
          }
          #pragma unroll
          for (uint group = 0U;
               group < IQ36_PREFILL_QUERY_GROUPS; ++group) {
            const short8 query_fragment = as_short8(
                iq36_block2d_load_f16_16x8(
                    query_base,
                    (int)INPUT0_DIMS[3] * (int)sizeof(half),
                    (int)query_tokens,
                    (int)INPUT0_PITCHES[2] * (int)sizeof(half),
                    k_base,
                    query_begin + group * IQ36_GQA_GROUP));
            score[group] =
                intel_sub_group_f16_f16_matrix_mad_k16(
                    query_fragment, key_fragment, score[group]);
          }
        }

        #pragma unroll
        for (uint group = 0U;
             group < IQ36_PREFILL_QUERY_GROUPS; ++group) {
          #pragma unroll
          for (uint row = 0U; row < IQ36_GQA_GROUP; ++row) {
            const uint query_in_tile = group * IQ36_GQA_GROUP + row;
            const uint query_position = query_begin + query_in_tile;
            float value = -INFINITY;
            if (query_in_tile < query_count &&
                token <= past_tokens + query_position) {
              const ulong mask_index = INPUT9_OFFSET +
                  (ulong)batch * INPUT9_PITCHES[0] +
                  (ulong)query_position * INPUT9_PITCHES[2] +
                  (ulong)token * INPUT9_PITCHES[3];
              value = score[group][row] +
                  convert_float(mask[mask_index]) /
                      IQ36_ATTENTION_SCALE;
            }
            local_score_weight[
                query_in_tile * IQ36_PREFILL_CHUNK_TOKENS +
                token_in_chunk] = value;
            const float maximum = sub_group_reduce_max(value);
            if (lane == 0U) {
              (void)__builtin_IB_atomic_max_local_f32(
                  &prefill_running_max[query_in_tile], maximum);
            }
          }
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);

      if (subgroup < valid_blocks) {
        const uint token_in_chunk = subgroup * IQ36_TOKEN_TILE + lane;
        for (uint query_in_tile = 0U; query_in_tile < query_count;
             ++query_in_tile) {
          const float weight = native_exp2(
              (local_score_weight[
                   query_in_tile * IQ36_PREFILL_CHUNK_TOKENS +
                   token_in_chunk] -
               prefill_running_max[query_in_tile]) * IQ36_EXP2_SCALE);
          local_score_weight[
              query_in_tile * IQ36_PREFILL_CHUNK_TOKENS +
              token_in_chunk] = weight;
          // Match stock sdpa_micro's tile_vreduce_add order.  The stock
          // kernel accumulates the 16 key lanes serially inside each query
          // row; a subgroup tree reduction is mathematically equivalent but
          // changes the last-bit rounding that compounds across ten layers.
          float sum = 0.0f;
          #pragma unroll
          for (uint token_lane = 0U;
               token_lane < IQ36_TOKEN_TILE; ++token_lane) {
            sum += sub_group_broadcast(weight, token_lane);
          }
          if (lane == 0U) {
            prefill_block_sum[
                subgroup * IQ36_PREFILL_QUERY_TILE +
                query_in_tile] = sum;
          }
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);

      if (local_id < query_count) {
        const float next_max = prefill_running_max[local_id];
        const float previous_scale = native_exp2(
            (prefill_previous_scale[local_id] - next_max) *
                IQ36_EXP2_SCALE);
        prefill_previous_scale[local_id] = previous_scale;
      }
      barrier(CLK_LOCAL_MEM_FENCE);

      // Stock keeps one running denominator per 16-key subgroup, rescales
      // each across 128-key chunks, and combines the eight subgroup totals
      // only after the final chunk.  Preserve that ordering exactly.
      if (lane == 0U) {
        for (uint query_in_tile = 0U; query_in_tile < query_count;
             ++query_in_tile) {
          const uint index =
              subgroup * IQ36_PREFILL_QUERY_TILE + query_in_tile;
          const float chunk_sum = subgroup < valid_blocks
              ? prefill_block_sum[index] : 0.0f;
          prefill_block_running_sum[index] =
              prefill_block_running_sum[index] *
                  prefill_previous_scale[query_in_tile] + chunk_sum;
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);

      float8 chunk_output0[IQ36_PREFILL_QUERY_GROUPS];
      float8 chunk_output1[IQ36_PREFILL_QUERY_GROUPS];
      #pragma unroll
      for (uint group = 0U;
           group < IQ36_PREFILL_QUERY_GROUPS; ++group) {
        chunk_output0[group] = (float8)(0.0f);
        chunk_output1[group] = (float8)(0.0f);
      }
      for (uint block = 0U; block < valid_blocks; ++block) {
        const uint token_in_chunk = block * IQ36_TOKEN_TILE + lane;
        const uint block_token = chunk_begin + block * IQ36_TOKEN_TILE;
        const uint block_slot = iq36_hot_slot(block_token);
        const bool prior_hot_contiguous =
            block_token + IQ36_TOKEN_TILE <= past_tokens &&
            (cold_tokens == 0U || block_token >= cold_tokens) &&
            block_slot + IQ36_TOKEN_TILE <= (uint)INPUT2_DIMS[2];
        const bool current_contiguous = block_token >= past_tokens;
        const __global half* state_value_base =
            (const __global half*)&hot_value[
                INPUT2_OFFSET + (ulong)batch * INPUT2_PITCHES[0] +
                (ulong)kv_head * INPUT2_PITCHES[1]];
        half8 value0;
        half8 value1;
        half8 value2;
        half8 value3;
        if (prior_hot_contiguous) {
          value0 = iq36_block2d_load_f16_16x8(
              state_value_base,
              (int)INPUT2_DIMS[3] * (int)sizeof(half),
              (int)INPUT2_DIMS[2],
              (int)INPUT2_PITCHES[2] * (int)sizeof(half),
              subgroup * IQ36_TOKEN_TILE, block_slot);
          value1 = iq36_block2d_load_f16_16x8(
              state_value_base,
              (int)INPUT2_DIMS[3] * (int)sizeof(half),
              (int)INPUT2_DIMS[2],
              (int)INPUT2_PITCHES[2] * (int)sizeof(half),
              subgroup * IQ36_TOKEN_TILE, block_slot + 8U);
          value2 = iq36_block2d_load_f16_16x8(
              state_value_base,
              (int)INPUT2_DIMS[3] * (int)sizeof(half),
              (int)INPUT2_DIMS[2],
              (int)INPUT2_PITCHES[2] * (int)sizeof(half),
              128U + subgroup * IQ36_TOKEN_TILE, block_slot);
          value3 = iq36_block2d_load_f16_16x8(
              state_value_base,
              (int)INPUT2_DIMS[3] * (int)sizeof(half),
              (int)INPUT2_DIMS[2],
              (int)INPUT2_PITCHES[2] * (int)sizeof(half),
              128U + subgroup * IQ36_TOKEN_TILE, block_slot + 8U);
        } else if (current_contiguous) {
          const uint current_token = block_token - past_tokens;
          value0 = iq36_block2d_load_f16_16x8(
              value_base,
              (int)INPUT4_DIMS[3] * (int)sizeof(half),
              (int)query_tokens,
              (int)IQ36_CURRENT_VALUE_TOKEN_PITCH * (int)sizeof(half),
              subgroup * IQ36_TOKEN_TILE, current_token);
          value1 = iq36_block2d_load_f16_16x8(
              value_base,
              (int)INPUT4_DIMS[3] * (int)sizeof(half),
              (int)query_tokens,
              (int)IQ36_CURRENT_VALUE_TOKEN_PITCH * (int)sizeof(half),
              subgroup * IQ36_TOKEN_TILE, current_token + 8U);
          value2 = iq36_block2d_load_f16_16x8(
              value_base,
              (int)INPUT4_DIMS[3] * (int)sizeof(half),
              (int)query_tokens,
              (int)IQ36_CURRENT_VALUE_TOKEN_PITCH * (int)sizeof(half),
              128U + subgroup * IQ36_TOKEN_TILE, current_token);
          value3 = iq36_block2d_load_f16_16x8(
              value_base,
              (int)INPUT4_DIMS[3] * (int)sizeof(half),
              (int)query_tokens,
              (int)IQ36_CURRENT_VALUE_TOKEN_PITCH * (int)sizeof(half),
              128U + subgroup * IQ36_TOKEN_TILE, current_token + 8U);
        } else {
          #pragma unroll
          for (uint row = 0U; row < 8U; ++row) {
            value0[row] = convert_half_rte(iq36_partial_load_value(
                batch, kv_head, block_token + row, past_tokens, cold_tokens,
                subgroup * IQ36_TOKEN_TILE + lane,
                hot_value, current_value, cold_value,
                cold_value_scale_bytes));
            value1[row] = convert_half_rte(iq36_partial_load_value(
                batch, kv_head, block_token + 8U + row,
                past_tokens, cold_tokens,
                subgroup * IQ36_TOKEN_TILE + lane,
                hot_value, current_value, cold_value,
                cold_value_scale_bytes));
            value2[row] = convert_half_rte(iq36_partial_load_value(
                batch, kv_head, block_token + row, past_tokens, cold_tokens,
                128U + subgroup * IQ36_TOKEN_TILE + lane,
                hot_value, current_value, cold_value,
                cold_value_scale_bytes));
            value3[row] = convert_half_rte(iq36_partial_load_value(
                batch, kv_head, block_token + 8U + row,
                past_tokens, cold_tokens,
                128U + subgroup * IQ36_TOKEN_TILE + lane,
                hot_value, current_value, cold_value,
                cold_value_scale_bytes));
          }
        }
        const int8 value_fragment0 = (int8)(
            as_int((half2)(value0[0], value0[1])),
            as_int((half2)(value0[2], value0[3])),
            as_int((half2)(value0[4], value0[5])),
            as_int((half2)(value0[6], value0[7])),
            as_int((half2)(value1[0], value1[1])),
            as_int((half2)(value1[2], value1[3])),
            as_int((half2)(value1[4], value1[5])),
            as_int((half2)(value1[6], value1[7])));
        const int8 value_fragment1 = (int8)(
            as_int((half2)(value2[0], value2[1])),
            as_int((half2)(value2[2], value2[3])),
            as_int((half2)(value2[4], value2[5])),
            as_int((half2)(value2[6], value2[7])),
            as_int((half2)(value3[0], value3[1])),
            as_int((half2)(value3[2], value3[3])),
            as_int((half2)(value3[4], value3[5])),
            as_int((half2)(value3[6], value3[7])));
        #pragma unroll
        for (uint group = 0U;
             group < IQ36_PREFILL_QUERY_GROUPS; ++group) {
          const uint query_group = group * IQ36_GQA_GROUP;
          const short8 weight_fragment = (short8)(
              as_short(convert_half_rte(local_score_weight[
                  (query_group + 0U) * IQ36_PREFILL_CHUNK_TOKENS +
                  token_in_chunk])),
              as_short(convert_half_rte(local_score_weight[
                  (query_group + 1U) * IQ36_PREFILL_CHUNK_TOKENS +
                  token_in_chunk])),
              as_short(convert_half_rte(local_score_weight[
                  (query_group + 2U) * IQ36_PREFILL_CHUNK_TOKENS +
                  token_in_chunk])),
              as_short(convert_half_rte(local_score_weight[
                  (query_group + 3U) * IQ36_PREFILL_CHUNK_TOKENS +
                  token_in_chunk])),
              as_short(convert_half_rte(local_score_weight[
                  (query_group + 4U) * IQ36_PREFILL_CHUNK_TOKENS +
                  token_in_chunk])),
              as_short(convert_half_rte(local_score_weight[
                  (query_group + 5U) * IQ36_PREFILL_CHUNK_TOKENS +
                  token_in_chunk])),
              as_short(convert_half_rte(local_score_weight[
                  (query_group + 6U) * IQ36_PREFILL_CHUNK_TOKENS +
                  token_in_chunk])),
              as_short(convert_half_rte(local_score_weight[
                  (query_group + 7U) * IQ36_PREFILL_CHUNK_TOKENS +
                  token_in_chunk])));
          chunk_output0[group] =
              intel_sub_group_f16_f16_matrix_mad_k16(
                  weight_fragment, value_fragment0,
                  chunk_output0[group]);
          chunk_output1[group] =
              intel_sub_group_f16_f16_matrix_mad_k16(
                  weight_fragment, value_fragment1,
                  chunk_output1[group]);
        }
      }

      #pragma unroll
      for (uint group = 0U;
           group < IQ36_PREFILL_QUERY_GROUPS; ++group) {
        const uint query_group = group * IQ36_GQA_GROUP;
        const float8 previous_scale = (float8)(
            prefill_previous_scale[query_group + 0U],
            prefill_previous_scale[query_group + 1U],
            prefill_previous_scale[query_group + 2U],
            prefill_previous_scale[query_group + 3U],
            prefill_previous_scale[query_group + 4U],
            prefill_previous_scale[query_group + 5U],
            prefill_previous_scale[query_group + 6U],
            prefill_previous_scale[query_group + 7U]);
        output_accumulator0[group] =
            output_accumulator0[group] * previous_scale +
            chunk_output0[group];
        output_accumulator1[group] =
            output_accumulator1[group] * previous_scale +
            chunk_output1[group];
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_id < query_count) {
      float sum = 0.0f;
      #pragma unroll
      for (uint block = 0U;
           block < IQ36_PREFILL_BLOCKS_PER_CHUNK; ++block) {
        sum += prefill_block_running_sum[
            block * IQ36_PREFILL_QUERY_TILE + local_id];
      }
      prefill_running_sum[local_id] = sum;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const uint dim = subgroup * IQ36_TOKEN_TILE + lane;
    #pragma unroll
    for (uint group = 0U;
         group < IQ36_PREFILL_QUERY_GROUPS; ++group) {
      #pragma unroll
      for (uint row = 0U; row < IQ36_GQA_GROUP; ++row) {
        const uint query_in_tile = group * IQ36_GQA_GROUP + row;
        if (query_in_tile < query_count) {
          const uint query_position = query_begin + query_in_tile;
          const float reciprocal = native_recip(
              prefill_running_sum[query_in_tile]);
          const float value0 = prefill_running_sum[query_in_tile] == 0.0f
              ? 0.0f : output_accumulator0[group][row] * reciprocal;
          const float value1 = prefill_running_sum[query_in_tile] == 0.0f
              ? 0.0f : output_accumulator1[group][row] * reciprocal;
#if defined(IQ36_TOKEN_MAJOR_OUTPUT)
          const ulong output0_index = OUTPUT1_OFFSET +
              (ulong)batch * OUTPUT1_PITCHES[0] +
              (ulong)query_position * OUTPUT1_PITCHES[1] +
              (ulong)query_head * OUTPUT1_PITCHES[2] +
              (ulong)dim * OUTPUT1_PITCHES[3];
#if defined(IQ36_FUSED_GATE_OUTPUT)
          const ulong gate0_index = INPUT13_OFFSET +
              (ulong)batch * INPUT13_PITCHES[0] +
              (ulong)query_position * INPUT13_PITCHES[1] +
              (ulong)query_head * INPUT13_PITCHES[2] +
              (ulong)dim * INPUT13_PITCHES[3];
          output[output0_index] = iq36_gated_attention_value(
              value0, raw_gate, gate0_index);
          output[output0_index + 128U * OUTPUT1_PITCHES[3]] =
              iq36_gated_attention_value(
                  value1, raw_gate,
                  gate0_index + 128U * INPUT13_PITCHES[3]);
#else
          output[output0_index] = (OUTPUT1_TYPE)value0;
          output[output0_index + 128U * OUTPUT1_PITCHES[3]] =
              (OUTPUT1_TYPE)value1;
#endif
#else
          const ulong output0_index = OUTPUT1_OFFSET +
              (ulong)batch * OUTPUT1_PITCHES[0] +
              (ulong)query_head * OUTPUT1_PITCHES[1] +
              (ulong)query_position * OUTPUT1_PITCHES[2] +
              (ulong)dim * OUTPUT1_PITCHES[3];
          output[output0_index] = (OUTPUT1_TYPE)value0;
          output[output0_index +
              128U * OUTPUT1_PITCHES[3]] = (OUTPUT1_TYPE)value1;
#endif
        }
      }
    }

    // Only one GQA head per KV head owns state and cold-codec writes.
    if (query_head_in_kv == 0U) {
      if (query_tile == 0U && local_id == 0U) {
        // Padding in the final packed-K block is graph-owned metadata.  Reset
        // the arrival word on each prefill so InferRequest reuse is safe.
        hot_key_bits[iq36_arrival_counter_index(batch, kv_head)] =
            (INPUT1_TYPE)0;
      }
      for (uint query_in_tile = 0U; query_in_tile < query_count;
           ++query_in_tile) {
        const uint token = query_begin + query_in_tile;
        const uint global_token = past_tokens + token;
        const ulong current_row_key = INPUT3_OFFSET +
            (ulong)batch * INPUT3_PITCHES[0] +
            (ulong)kv_head * INPUT3_PITCHES[1] +
            (ulong)token * INPUT3_PITCHES[2];
        const ulong current_row_value = iq36_current_value_index(
            batch, kv_head, token, 0U);

        if (global_token < IQ36_SINK_TOKENS ||
            global_token + IQ36_HOT_WINDOW >= key_tokens) {
          const uint slot = iq36_hot_slot(global_token);
          for (uint state_dim = local_id; state_dim < IQ36_HEAD_DIM;
               state_dim += 128U) {
            const ulong hot_value_index = INPUT2_OFFSET +
                (ulong)batch * INPUT2_PITCHES[0] +
                (ulong)kv_head * INPUT2_PITCHES[1] +
                (ulong)slot * INPUT2_PITCHES[2] +
                (ulong)state_dim * INPUT2_PITCHES[3];
            hot_value[hot_value_index] = (INPUT2_TYPE)
                current_value[current_row_value +
                    (ulong)state_dim * INPUT4_PITCHES[3]];
          }
          if (local_id < IQ36_HEAD_DIM / 2U) {
            const uint key_dim = local_id * 2U;
            const uint key_word = local_id * IQ36_KEY_TILE_TOKENS +
                (slot & (IQ36_KEY_TILE_TOKENS - 1U));
            const ulong hot_key_index = INPUT1_OFFSET +
                (ulong)batch * INPUT1_PITCHES[0] +
                (ulong)kv_head * INPUT1_PITCHES[1] +
                (ulong)(slot / IQ36_KEY_TILE_TOKENS) *
                    INPUT1_PITCHES[2] +
                (ulong)key_word * INPUT1_PITCHES[3];
            hot_key_bits[hot_key_index] = (INPUT1_TYPE)as_int((half2)(
                (half)current_key[current_row_key +
                    (ulong)key_dim * INPUT3_PITCHES[3]],
                (half)current_key[current_row_key +
                    (ulong)(key_dim + 1U) * INPUT3_PITCHES[3]]));
          }
        }

        if (token < cold_append_tokens) {
          const uint block = subgroup;
          const uint dim0 = block * 32U + lane * 2U;
          const uint dim1 = dim0 + 1U;
          const uint evicted_token = cold_tokens + token;
          const float key0 = iq36_partial_load_key(
              batch, kv_head, evicted_token, past_tokens, cold_tokens,
              dim0, hot_key_bits, current_key, cold_key,
              cold_key_scale_bytes);
          const float key1 = iq36_partial_load_key(
              batch, kv_head, evicted_token, past_tokens, cold_tokens,
              dim1, hot_key_bits, current_key, cold_key,
              cold_key_scale_bytes);
          const float value0 = iq36_partial_load_value(
              batch, kv_head, evicted_token, past_tokens, cold_tokens,
              dim0, hot_value, current_value, cold_value,
              cold_value_scale_bytes);
          const float value1 = iq36_partial_load_value(
              batch, kv_head, evicted_token, past_tokens, cold_tokens,
              dim1, hot_value, current_value, cold_value,
              cold_value_scale_bytes);
          const float key_max = sub_group_reduce_max(
              fmax(fabs(key0), fabs(key1)));
          const float value_max = sub_group_reduce_max(
              fmax(fabs(value0), fabs(value1)));
          const float key_scale = key_max == 0.0f
              ? 1.0f : key_max / 127.0f;
          const float value_scale = value_max == 0.0f
              ? 1.0f : value_max / 127.0f;
          const int key_q0 = clamp(
              (int)rint(key0 / key_scale), -127, 127);
          const int key_q1 = clamp(
              (int)rint(key1 / key_scale), -127, 127);
          const int value_q0 = clamp(
              (int)rint(value0 / value_scale), -127, 127);
          const int value_q1 = clamp(
              (int)rint(value1 / value_scale), -127, 127);
          const ulong cold_key_base = OUTPUT2_OFFSET +
              (ulong)batch * OUTPUT2_PITCHES[0] +
              (ulong)kv_head * OUTPUT2_PITCHES[1] +
              (ulong)token * OUTPUT2_PITCHES[2];
          const ulong cold_value_base = OUTPUT3_OFFSET +
              (ulong)batch * OUTPUT3_PITCHES[0] +
              (ulong)kv_head * OUTPUT3_PITCHES[1] +
              (ulong)token * OUTPUT3_PITCHES[2];
          cold_key_append[cold_key_base +
              (ulong)dim0 * OUTPUT2_PITCHES[3]] =
                  (OUTPUT2_TYPE)key_q0;
          cold_key_append[cold_key_base +
              (ulong)dim1 * OUTPUT2_PITCHES[3]] =
                  (OUTPUT2_TYPE)key_q1;
          cold_value_append[cold_value_base +
              (ulong)dim0 * OUTPUT3_PITCHES[3]] =
                  (OUTPUT3_TYPE)value_q0;
          cold_value_append[cold_value_base +
              (ulong)dim1 * OUTPUT3_PITCHES[3]] =
                  (OUTPUT3_TYPE)value_q1;
          if (lane < 2U) {
            const ushort key_bits = as_ushort(
                convert_half_rte(key_scale));
            const ushort value_bits = as_ushort(
                convert_half_rte(value_scale));
            const uint scale_x = block * 2U + lane;
            const ulong key_scale_index = OUTPUT4_OFFSET +
                (ulong)batch * OUTPUT4_PITCHES[0] +
                (ulong)kv_head * OUTPUT4_PITCHES[1] +
                (ulong)token * OUTPUT4_PITCHES[2] +
                (ulong)scale_x * OUTPUT4_PITCHES[3];
            const ulong value_scale_index = OUTPUT5_OFFSET +
                (ulong)batch * OUTPUT5_PITCHES[0] +
                (ulong)kv_head * OUTPUT5_PITCHES[1] +
                (ulong)token * OUTPUT5_PITCHES[2] +
                (ulong)scale_x * OUTPUT5_PITCHES[3];
            cold_key_scale_append[key_scale_index] = (OUTPUT4_TYPE)(
                lane == 0U ? key_bits & 0xffU : key_bits >> 8);
            cold_value_scale_append[value_scale_index] = (OUTPUT5_TYPE)(
                lane == 0U ? value_bits & 0xffU : value_bits >> 8);
          }
        }
      }
    }
    return;
  }
#endif

#if !defined(IQ36_BUILD_PREFILL_ONLY)
  // Decode: one work-group owns one 512-token partial for one KV head.
  const uint chunk = work_group;
  const uint scheduled_chunk_count = (uint)OUTPUT0_DIMS[2];
  const uint chunk_count =
      (key_tokens + IQ36_CHUNK_TOKENS - 1U) / IQ36_CHUNK_TOKENS;
  const uint chunk_begin = chunk * IQ36_CHUNK_TOKENS;
  const uint valid_tokens = chunk_begin < key_tokens
      ? min((uint)IQ36_CHUNK_TOKENS, key_tokens - chunk_begin)
      : 0U;
  const uint valid_blocks =
      (valid_tokens + IQ36_TOKEN_TILE - 1U) / IQ36_TOKEN_TILE;
  if (valid_tokens == 0U ||
      chunk_count > scheduled_chunk_count ||
      scheduled_chunk_count > IQ36_MAX_DECODE_CHUNKS) return;
  const uint dense_ring_capacity =
      (uint)INPUT2_DIMS[2] - IQ36_SINK_TOKENS;
  const uint dense_history_begin = past_tokens > dense_ring_capacity
      ? max((uint)IQ36_SINK_TOKENS,
            past_tokens - dense_ring_capacity)
      : (uint)IQ36_SINK_TOKENS;
#if defined(IQ36_DIRECT_I8_GROUP4_FULL_COLD)
  // The group-4 route is admitted specifically against the standalone full
  // logical-cold component.  Keep the larger continuation-safe ring, but do
  // not read its stale extra 8k rows during decode.
  const uint attention_cold_tokens = cold_tokens;
#else
  // Product buckets retain the entire prompt in the dense F16 ring.  Use the
  // quantized cold plane only for rows that generation has actually wrapped
  // over; otherwise dequantizing the logical cold prefix discards a faster,
  // exact copy that is already resident.
  const uint attention_cold_tokens =
      min(cold_tokens, dense_history_begin);
#endif

  float8 output_accumulator0 = (float8)(0.0f);
  float8 output_accumulator1 = (float8)(0.0f);
#if defined(IQ36_SPLIT256_REDUCTION)
  float8 output_accumulator0_part1 = (float8)(0.0f);
  float8 output_accumulator1_part1 = (float8)(0.0f);
#endif
  const __global half* query_base =
      (const __global half*)&query[
          INPUT0_OFFSET + (ulong)batch * INPUT0_PITCHES[0]];
#if defined(IQ36_STOCK_SCORE_ORDER)
  // Numeric oracle for stock sdpa_opt's non-compressed QK path.  One subgroup
  // owns one 16-token block.  For each token, lane L accumulates dimensions
  // L, L+16, ... L+240 in that exact order, after Q*scale has rounded to F16;
  // the subgroup tree then forms the score.  This deliberately trades DPAS
  // throughput for a one-layer attribution and leaves softmax/value math on
  // the existing custom path.
  for (uint block = subgroup; block < IQ36_BLOCKS_PER_CHUNK;
       block += 8U) {
    const uint block_token = chunk_begin + block * IQ36_TOKEN_TILE;
    float8 block_maximum = (float8)(-INFINITY);
    #pragma unroll 1
    for (uint token_lane = 0U; token_lane < IQ36_TOKEN_TILE;
         ++token_lane) {
      const uint token = block_token + token_lane;
      float8 score = (float8)(0.0f);
      if (token < key_tokens) {
        #pragma unroll
        for (uint stripe = 0U; stripe < IQ36_HEAD_DIM / 16U; ++stripe) {
          const uint dim = lane + stripe * 16U;
          const float key_value = convert_float(convert_half_rte(
              iq36_partial_load_key(
                  batch, kv_head, token, past_tokens,
                  attention_cold_tokens, dim, hot_key_bits, current_key,
                  cold_key, cold_key_scale_bytes)));
          #pragma unroll
          for (uint head = 0U; head < IQ36_GQA_GROUP; ++head) {
            const uint query_head =
                kv_head * IQ36_GQA_GROUP + head;
            const ulong query_index =
                (ulong)query_head * INPUT0_PITCHES[1] +
                (ulong)dim * INPUT0_PITCHES[3];
            const half scaled_query = convert_half_rte(
                convert_float(query_base[query_index]) *
                    IQ36_ATTENTION_SCALE);
            score[head] = mad(
                convert_float(scaled_query), key_value, score[head]);
          }
        }
      }
      #pragma unroll
      for (uint head = 0U; head < IQ36_GQA_GROUP; ++head) {
        const float value = token < key_tokens
            ? sub_group_reduce_add(score[head]) : -INFINITY;
        block_maximum[head] = fmax(block_maximum[head], value);
        if (lane == 0U) {
          local_score_weight[
              head * IQ36_CHUNK_TOKENS +
              block * IQ36_TOKEN_TILE + token_lane] = value;
        }
      }
    }
    if (lane == 0U) {
      #pragma unroll
      for (uint head = 0U; head < IQ36_GQA_GROUP; ++head) {
        local_block_max[block * IQ36_GQA_GROUP + head] =
            block_maximum[head];
      }
    }
  }
#else
  for (uint block = subgroup; block < IQ36_BLOCKS_PER_CHUNK;
       block += 8U) {
    const uint block_token = chunk_begin + block * IQ36_TOKEN_TILE;
    const uint token = block_token + lane;
    const uint block_slot = iq36_hot_slot(block_token);
    const bool packed_block_contiguous_raw =
        (block_slot & (IQ36_KEY_TILE_TOKENS - 1U)) == 0U &&
        block_slot + IQ36_TOKEN_TILE <= (uint)INPUT2_DIMS[2];
#if defined(IQ36_DIRECT_I8_FIXED_LAYOUT)
    const bool direct_cold_block =
        block_token >= IQ36_SINK_TOKENS &&
        block_token + IQ36_TOKEN_TILE <= attention_cold_tokens;
    const bool mixed_cold_block =
        !direct_cold_block && block_token < attention_cold_tokens &&
        block_token + IQ36_TOKEN_TILE > IQ36_SINK_TOKENS;
    const bool packed_block_contiguous =
        packed_block_contiguous_raw && !mixed_cold_block;
#else
    const bool packed_block_contiguous = packed_block_contiguous_raw;
#endif
    float8 score = (float8)(0.0f);
    #pragma unroll 1
    for (uint k_block = 0U;
         k_block < IQ36_HEAD_DIM / IQ36_TOKEN_TILE; ++k_block) {
      const uint k_base = k_block * IQ36_TOKEN_TILE;
      const uint q_head_base = kv_head * IQ36_GQA_GROUP;
      const short8 q_fragment = as_short8(
          iq36_block2d_load_f16_16x8(
              query_base,
              (int)INPUT0_DIMS[3] * (int)sizeof(half),
              (int)INPUT0_DIMS[1],
              (int)INPUT0_PITCHES[1] * (int)sizeof(half),
              k_base, q_head_base));
      int8 key_fragment;
      if (packed_block_contiguous) {
#if defined(IQ36_DIRECT_I8_FIXED_LAYOUT)
        if (direct_cold_block) {
          key_fragment = iq36_cold_key_fragment(
              batch, kv_head, token, k_base,
              cold_key, cold_key_scale_bytes);
        } else {
#endif
        const ulong key_index = INPUT1_OFFSET +
            (ulong)batch * INPUT1_PITCHES[0] +
            (ulong)kv_head * INPUT1_PITCHES[1] +
            (ulong)(block_slot / IQ36_KEY_TILE_TOKENS) *
                INPUT1_PITCHES[2] +
            (ulong)(k_block * 8U * IQ36_KEY_TILE_TOKENS) *
                INPUT1_PITCHES[3];
        key_fragment = as_int8(intel_sub_group_block_read8(
            (const __global uint*)&hot_key_bits[key_index]));
        if (token >= IQ36_SINK_TOKENS &&
            token < attention_cold_tokens) {
          key_fragment = iq36_cold_key_fragment(
              batch, kv_head, token, k_base,
              cold_key, cold_key_scale_bytes);
        } else if (token == past_tokens) {
          const ulong current_index = INPUT3_OFFSET +
              (ulong)batch * INPUT3_PITCHES[0] +
              (ulong)kv_head * INPUT3_PITCHES[1] +
              (ulong)k_base * INPUT3_PITCHES[3];
          key_fragment = as_int8(vload16(
              0, (const __global half*)&current_key[current_index]));
        }
#if defined(IQ36_DIRECT_I8_FIXED_LAYOUT)
        }
#endif
      } else {
        #pragma unroll 1
        for (uint pair = 0U; pair < 8U; ++pair) {
          const uint dim0 = k_base + pair * 2U;
          half2 values = (half2)(0.0h);
          if (token < key_tokens) {
            values[0] = convert_half_rte(iq36_partial_load_key(
                batch, kv_head, token, past_tokens,
                attention_cold_tokens,
                dim0, hot_key_bits, current_key, cold_key,
                cold_key_scale_bytes));
            values[1] = convert_half_rte(iq36_partial_load_key(
                batch, kv_head, token, past_tokens,
                attention_cold_tokens,
                dim0 + 1U, hot_key_bits, current_key, cold_key,
                cold_key_scale_bytes));
          }
          key_fragment[pair] = as_int(values);
        }
      }
      score = intel_sub_group_f16_f16_matrix_mad_k16(
          q_fragment, key_fragment, score);
    }
    // Product batch is one and every scheduled token is valid.  Decode is
    // causal by construction, so the full growing additive mask is identically
    // zero over [0,key_tokens).  Keeping it out of this path lets the graph use
    // a 512-token bucket carrier and avoids one GPU specialization per token.
    const float mask_value = token < key_tokens ? 0.0f : -INFINITY;
    #pragma unroll
    for (uint head = 0U; head < IQ36_GQA_GROUP; ++head) {
      const float value = token < key_tokens
          ? score[head] + mask_value : -INFINITY;
      local_score_weight[
          head * IQ36_CHUNK_TOKENS +
          block * IQ36_TOKEN_TILE + lane] = value;
      const float maximum = sub_group_reduce_max(value);
      if (lane == 0U) {
        local_block_max[block * IQ36_GQA_GROUP + head] = maximum;
      }
    }
  }
#endif
  barrier(CLK_LOCAL_MEM_FENCE);

  if (subgroup == 0U && lane < IQ36_GQA_GROUP) {
#if defined(IQ36_SPLIT256_REDUCTION)
    #pragma unroll
    for (uint part = 0U; part < IQ36_DECODE_REDUCTION_PARTS; ++part) {
      float value = -INFINITY;
      #pragma unroll
      for (uint block = 0U; block < IQ36_DECODE_BLOCKS_PER_PART; ++block) {
        value = fmax(value, local_block_max[
            (part * IQ36_DECODE_BLOCKS_PER_PART + block) *
                IQ36_GQA_GROUP + lane]);
      }
      local_global_max[part * IQ36_GQA_GROUP + lane] = value;
    }
#else
    float value = -INFINITY;
    #pragma unroll
    for (uint block = 0U; block < IQ36_BLOCKS_PER_CHUNK; ++block) {
      value = fmax(value,
          local_block_max[block * IQ36_GQA_GROUP + lane]);
    }
    local_global_max[lane] = value;
#endif
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  for (uint block = subgroup; block < IQ36_BLOCKS_PER_CHUNK;
       block += 8U) {
    const uint token_in_chunk = block * IQ36_TOKEN_TILE + lane;
#if defined(IQ36_SPLIT256_REDUCTION)
    const uint reduction_part =
        block / IQ36_DECODE_BLOCKS_PER_PART;
#endif
    #pragma unroll
    for (uint head = 0U; head < IQ36_GQA_GROUP; ++head) {
#if defined(IQ36_SPLIT256_REDUCTION)
#if defined(IQ36_STOCK256_PARTIALS)
      const float weight = token_in_chunk < valid_tokens
          ? native_exp(
              (local_score_weight[
                   head * IQ36_CHUNK_TOKENS + token_in_chunk] -
               local_global_max[
                   reduction_part * IQ36_GQA_GROUP + head]) *
                  IQ36_ATTENTION_SCALE)
          : 0.0f;
#else
      const float weight = token_in_chunk < valid_tokens
          ? native_exp2(
              (local_score_weight[
                   head * IQ36_CHUNK_TOKENS + token_in_chunk] -
               local_global_max[
                   reduction_part * IQ36_GQA_GROUP + head]) *
                  IQ36_DECODE_EXP2_SCALE)
          : 0.0f;
#endif
#else
#if defined(IQ36_STOCK_PARTITION_ARITHMETIC)
      const float weight = token_in_chunk < valid_tokens
          ? native_exp(
              local_score_weight[
                  head * IQ36_CHUNK_TOKENS + token_in_chunk] -
              local_global_max[head])
          : 0.0f;
#else
      const float weight = native_exp2(
          (local_score_weight[
               head * IQ36_CHUNK_TOKENS + token_in_chunk] -
           local_global_max[head]) * IQ36_DECODE_EXP2_SCALE);
#endif
#endif
      local_score_weight[
          head * IQ36_CHUNK_TOKENS + token_in_chunk] =
              weight;
      const float sum = sub_group_reduce_add(weight);
      if (lane == 0U) {
        local_block_sum[block * IQ36_GQA_GROUP + head] = sum;
      }
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  if (subgroup == 0U) {
#if defined(IQ36_STOCK256_PARTIALS) || \
    defined(IQ36_STOCK_PARTITION_ARITHMETIC)
    #pragma unroll
    for (uint part = 0U; part < IQ36_DECODE_REDUCTION_PARTS; ++part) {
      #pragma unroll
      for (uint head = 0U; head < IQ36_GQA_GROUP; ++head) {
        const float value = local_block_sum[
            (part * IQ36_DECODE_BLOCKS_PER_PART + lane) *
                IQ36_GQA_GROUP + head];
        const float sum = sub_group_reduce_add(value);
        if (lane == 0U) {
          local_global_sum[part * IQ36_GQA_GROUP + head] = sum;
        }
      }
    }
#elif defined(IQ36_SPLIT256_REDUCTION)
    if (lane < IQ36_GQA_GROUP) {
    #pragma unroll
    for (uint part = 0U; part < IQ36_DECODE_REDUCTION_PARTS; ++part) {
      float value = 0.0f;
      #pragma unroll
      for (uint block = 0U; block < IQ36_DECODE_BLOCKS_PER_PART; ++block) {
        value += local_block_sum[
            (part * IQ36_DECODE_BLOCKS_PER_PART + block) *
                IQ36_GQA_GROUP + lane];
      }
      local_global_sum[part * IQ36_GQA_GROUP + lane] = value;
    }
    }
#else
    if (lane < IQ36_GQA_GROUP) {
      float value = 0.0f;
      #pragma unroll
      for (uint block = 0U; block < IQ36_BLOCKS_PER_CHUNK; ++block) {
        value += local_block_sum[block * IQ36_GQA_GROUP + lane];
      }
      local_global_sum[lane] = value;
    }
#endif
  }
  barrier(CLK_LOCAL_MEM_FENCE);

#if defined(IQ36_DUAL256_REDUCTION)
  // Preserve stock's 256-key numerator rounding without doubling the number
  // of work-groups or workspace rows.  Each half is normalized and consumed
  // by XMX independently; the two F32 partials are merged in this work-group.
  // local_block_{max,sum}[0:8] are dead after the reductions, so reuse them
  // for the two merge scales without increasing SLM.
  if (subgroup == 0U && lane < IQ36_GQA_GROUP) {
    const float maximum0 = local_global_max[lane];
    const float sum0 = local_global_sum[lane];
    if (valid_blocks > IQ36_DECODE_BLOCKS_PER_PART) {
      const float maximum1 =
          local_global_max[IQ36_GQA_GROUP + lane];
      const float sum1 = local_global_sum[IQ36_GQA_GROUP + lane];
      const float maximum = fmax(maximum0, maximum1);
      const float scale0 = native_exp2(
          (maximum0 - maximum) * IQ36_DECODE_EXP2_SCALE);
      const float scale1 = native_exp2(
          (maximum1 - maximum) * IQ36_DECODE_EXP2_SCALE);
      local_block_max[lane] = scale0;
      local_block_sum[lane] = scale1;
      local_global_max[lane] = maximum;
      local_global_sum[lane] = sum0 * scale0 + sum1 * scale1;
    } else {
      local_block_max[lane] = 1.0f;
      local_block_sum[lane] = 0.0f;
      local_global_max[lane] = maximum0;
      local_global_sum[lane] = sum0;
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
#endif

#if defined(IQ36_STOCK256_PARTIALS) || \
    defined(IQ36_STOCK_PARTITION_ARITHMETIC)
  // Stock sdpa_opt normalizes each 256-token partition before converting the
  // weights to F16 for the value product.  Keep that rounding point distinct
  // from the final cross-partition normalization.
  for (uint block = subgroup; block < IQ36_BLOCKS_PER_CHUNK;
       block += 8U) {
    const uint token_in_chunk = block * IQ36_TOKEN_TILE + lane;
    const uint reduction_part =
        block / IQ36_DECODE_BLOCKS_PER_PART;
    #pragma unroll
    for (uint head = 0U; head < IQ36_GQA_GROUP; ++head) {
      const float denominator = local_global_sum[
          reduction_part * IQ36_GQA_GROUP + head];
      local_score_weight[
          head * IQ36_CHUNK_TOKENS + token_in_chunk] =
              token_in_chunk < valid_tokens && denominator != 0.0f
#if defined(IQ36_STOCK_PARTITION_ARITHMETIC)
              ? convert_float(convert_half_rte(local_score_weight[
                    head * IQ36_CHUNK_TOKENS + token_in_chunk])) /
                    denominator
#else
              ? local_score_weight[
                    head * IQ36_CHUNK_TOKENS + token_in_chunk] /
                    denominator
#endif
              : 0.0f;
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
#endif

  const uint dim0 = subgroup * IQ36_TOKEN_TILE + lane;
  const uint dim1 = 128U + dim0;
  const __global half* state_base =
      (const __global half*)&hot_value[
          INPUT2_OFFSET + (ulong)batch * INPUT2_PITCHES[0] +
          (ulong)kv_head * INPUT2_PITCHES[1]];
#if defined(IQ36_STOCK_PARTITION_ARITHMETIC)
  // Match stock Gemm2's OUTPUT_TYPE accumulator and token order exactly.
  // Each work-item owns two output dimensions; normalized partition weights
  // and values round to F16 before every F16 mad.
  half8 stock_accumulator0 = (half8)(0.0h);
  half8 stock_accumulator1 = (half8)(0.0h);
  #pragma unroll 1
  for (uint token_offset = 0U; token_offset < valid_tokens;
       ++token_offset) {
    const uint token = chunk_begin + token_offset;
    const half value0 = convert_half_rte(iq36_partial_load_value(
        batch, kv_head, token, past_tokens, attention_cold_tokens, dim0,
        hot_value, current_value, cold_value, cold_value_scale_bytes));
    const half value1 = convert_half_rte(iq36_partial_load_value(
        batch, kv_head, token, past_tokens, attention_cold_tokens, dim1,
        hot_value, current_value, cold_value, cold_value_scale_bytes));
    #pragma unroll
    for (uint head = 0U; head < IQ36_GQA_GROUP; ++head) {
      const half weight = convert_half_rte(local_score_weight[
          head * IQ36_CHUNK_TOKENS + token_offset]);
      stock_accumulator0[head] = mad(
          weight, value0, stock_accumulator0[head]);
      stock_accumulator1[head] = mad(
          weight, value1, stock_accumulator1[head]);
    }
  }
  output_accumulator0 = convert_float8(stock_accumulator0);
  output_accumulator1 = convert_float8(stock_accumulator1);
#else
  #pragma unroll 1
  for (uint block = 0U; block < valid_blocks; ++block) {
    const uint token_in_chunk = block * IQ36_TOKEN_TILE + lane;
    const uint block_token = chunk_begin + block * IQ36_TOKEN_TILE;
    const uint block_slot = iq36_hot_slot(block_token);
#if defined(IQ36_F32_NUMERATOR)
    // Arithmetic oracle: preserve the score/softmax schedule and remove only
    // the F32 -> F16 weight conversion plus the F16 value DPAS numerator.
    // This deliberately favors diagnosis over throughput and is selected for
    // one layer at a time; the production path below remains unchanged.
    #pragma unroll 1
    for (uint token_offset = 0U;
         token_offset < IQ36_TOKEN_TILE; ++token_offset) {
      const uint token = block_token + token_offset;
      if (token >= key_tokens) break;
      const float value0 = iq36_partial_load_value(
          batch, kv_head, token, past_tokens,
          attention_cold_tokens, dim0, hot_value, current_value,
          cold_value, cold_value_scale_bytes);
      const float value1 = iq36_partial_load_value(
          batch, kv_head, token, past_tokens,
          attention_cold_tokens, dim1, hot_value, current_value,
          cold_value, cold_value_scale_bytes);
      #pragma unroll
      for (uint head = 0U; head < IQ36_GQA_GROUP; ++head) {
        const float weight = local_score_weight[
            head * IQ36_CHUNK_TOKENS +
            block * IQ36_TOKEN_TILE + token_offset];
        output_accumulator0[head] = fma(
            weight, value0, output_accumulator0[head]);
        output_accumulator1[head] = fma(
            weight, value1, output_accumulator1[head]);
      }
    }
#else
    const bool contiguous =
        (block_slot & (IQ36_KEY_TILE_TOKENS - 1U)) == 0U &&
        block_slot + IQ36_TOKEN_TILE <= (uint)INPUT2_DIMS[2];
    const short8 weight_fragment = (short8)(
        as_short(convert_half_rte(local_score_weight[
            0U * IQ36_CHUNK_TOKENS + token_in_chunk])),
        as_short(convert_half_rte(local_score_weight[
            1U * IQ36_CHUNK_TOKENS + token_in_chunk])),
        as_short(convert_half_rte(local_score_weight[
            2U * IQ36_CHUNK_TOKENS + token_in_chunk])),
        as_short(convert_half_rte(local_score_weight[
            3U * IQ36_CHUNK_TOKENS + token_in_chunk])),
        as_short(convert_half_rte(local_score_weight[
            4U * IQ36_CHUNK_TOKENS + token_in_chunk])),
        as_short(convert_half_rte(local_score_weight[
            5U * IQ36_CHUNK_TOKENS + token_in_chunk])),
        as_short(convert_half_rte(local_score_weight[
            6U * IQ36_CHUNK_TOKENS + token_in_chunk])),
        as_short(convert_half_rte(local_score_weight[
            7U * IQ36_CHUNK_TOKENS + token_in_chunk])));
    int8 value_fragment0;
    int8 value_fragment1;
#if defined(IQ36_DIRECT_I8_FIXED_LAYOUT)
    const bool direct_cold_value_block =
        block_token >= IQ36_SINK_TOKENS &&
        block_token + IQ36_TOKEN_TILE <= attention_cold_tokens;
    if (direct_cold_value_block) {
      value_fragment0 = iq36_direct_cold_value_fragment(
          batch, kv_head, block_token, dim0,
          cold_value, cold_value_scale_bytes);
      value_fragment1 = iq36_direct_cold_value_fragment(
          batch, kv_head, block_token, dim1,
          cold_value, cold_value_scale_bytes);
    } else
#endif
#if defined(IQ36_DIRECT_I8_GROUP4_FULL_COLD)
    if (contiguous && block_token >= attention_cold_tokens &&
        block_token + IQ36_TOKEN_TILE <= past_tokens) {
      value_fragment0 = iq36_direct_hot_value_fragment(
          batch, kv_head, block_slot, dim0, hot_key_bits);
      value_fragment1 = iq36_direct_hot_value_fragment(
          batch, kv_head, block_slot, dim1, hot_key_bits);
    } else
#endif
    if (contiguous) {
      half8 a0 = iq36_block2d_load_f16_16x8(
          state_base,
          IQ36_HEAD_DIM * (int)sizeof(half),
          (int)INPUT2_DIMS[2],
          (int)INPUT2_PITCHES[2] * (int)sizeof(half),
          subgroup * IQ36_TOKEN_TILE, block_slot);
      half8 a1 = iq36_block2d_load_f16_16x8(
          state_base,
          IQ36_HEAD_DIM * (int)sizeof(half),
          (int)INPUT2_DIMS[2],
          (int)INPUT2_PITCHES[2] * (int)sizeof(half),
          subgroup * IQ36_TOKEN_TILE, block_slot + 8U);
      half8 b0 = iq36_block2d_load_f16_16x8(
          state_base,
          IQ36_HEAD_DIM * (int)sizeof(half),
          (int)INPUT2_DIMS[2],
          (int)INPUT2_PITCHES[2] * (int)sizeof(half),
          128U + subgroup * IQ36_TOKEN_TILE, block_slot);
      half8 b1 = iq36_block2d_load_f16_16x8(
          state_base,
          IQ36_HEAD_DIM * (int)sizeof(half),
          (int)INPUT2_DIMS[2],
          (int)INPUT2_PITCHES[2] * (int)sizeof(half),
          128U + subgroup * IQ36_TOKEN_TILE, block_slot + 8U);
      #pragma unroll 1
      for (uint row = 0U; row < 8U; ++row) {
        const uint token = block_token + row;
        if (token >= IQ36_SINK_TOKENS &&
            token < attention_cold_tokens) {
          a0[row] = iq36_cold_value_element(
              batch, kv_head, token, dim0,
              cold_value, cold_value_scale_bytes);
          b0[row] = iq36_cold_value_element(
              batch, kv_head, token, dim1,
              cold_value, cold_value_scale_bytes);
        } else if (token == past_tokens) {
          const ulong current0 = iq36_current_value_index(
              batch, kv_head, 0U, dim0);
          const ulong current1 = iq36_current_value_index(
              batch, kv_head, 0U, dim1);
          a0[row] = convert_half_rte(current_value[current0]);
          b0[row] = convert_half_rte(current_value[current1]);
        }
      }
      #pragma unroll 1
      for (uint row = 0U; row < 8U; ++row) {
        const uint token = block_token + 8U + row;
        if (token >= IQ36_SINK_TOKENS &&
            token < attention_cold_tokens) {
          a1[row] = iq36_cold_value_element(
              batch, kv_head, token, dim0,
              cold_value, cold_value_scale_bytes);
          b1[row] = iq36_cold_value_element(
              batch, kv_head, token, dim1,
              cold_value, cold_value_scale_bytes);
        } else if (token == past_tokens) {
          const ulong current0 = iq36_current_value_index(
              batch, kv_head, 0U, dim0);
          const ulong current1 = iq36_current_value_index(
              batch, kv_head, 0U, dim1);
          a1[row] = convert_half_rte(current_value[current0]);
          b1[row] = convert_half_rte(current_value[current1]);
        }
      }
      value_fragment0 = (int8)(
          as_int((half2)(a0[0], a0[1])),
          as_int((half2)(a0[2], a0[3])),
          as_int((half2)(a0[4], a0[5])),
          as_int((half2)(a0[6], a0[7])),
          as_int((half2)(a1[0], a1[1])),
          as_int((half2)(a1[2], a1[3])),
          as_int((half2)(a1[4], a1[5])),
          as_int((half2)(a1[6], a1[7])));
      value_fragment1 = (int8)(
          as_int((half2)(b0[0], b0[1])),
          as_int((half2)(b0[2], b0[3])),
          as_int((half2)(b0[4], b0[5])),
          as_int((half2)(b0[6], b0[7])),
          as_int((half2)(b1[0], b1[1])),
          as_int((half2)(b1[2], b1[3])),
          as_int((half2)(b1[4], b1[5])),
          as_int((half2)(b1[6], b1[7])));
    } else {
      #pragma unroll 1
      for (uint pair = 0U; pair < 8U; ++pair) {
        const uint token0 = block_token + pair * 2U;
        half2 values0 = (half2)(0.0h);
        half2 values1 = (half2)(0.0h);
        if (token0 < key_tokens) {
          values0[0] = convert_half_rte(iq36_partial_load_value(
              batch, kv_head, token0, past_tokens,
              attention_cold_tokens,
              dim0, hot_value, current_value, cold_value,
              cold_value_scale_bytes));
          values1[0] = convert_half_rte(iq36_partial_load_value(
              batch, kv_head, token0, past_tokens,
              attention_cold_tokens,
              dim1, hot_value, current_value, cold_value,
              cold_value_scale_bytes));
        }
        if (token0 + 1U < key_tokens) {
          values0[1] = convert_half_rte(iq36_partial_load_value(
              batch, kv_head, token0 + 1U, past_tokens,
              attention_cold_tokens,
              dim0, hot_value, current_value, cold_value,
              cold_value_scale_bytes));
          values1[1] = convert_half_rte(iq36_partial_load_value(
              batch, kv_head, token0 + 1U, past_tokens,
              attention_cold_tokens,
              dim1, hot_value, current_value, cold_value,
              cold_value_scale_bytes));
        }
        value_fragment0[pair] = as_int(values0);
        value_fragment1[pair] = as_int(values1);
      }
    }
#if defined(IQ36_SPLIT256_REDUCTION)
    if (block < IQ36_DECODE_BLOCKS_PER_PART) {
      output_accumulator0 = intel_sub_group_f16_f16_matrix_mad_k16(
          weight_fragment, value_fragment0, output_accumulator0);
      output_accumulator1 = intel_sub_group_f16_f16_matrix_mad_k16(
          weight_fragment, value_fragment1, output_accumulator1);
    } else {
      output_accumulator0_part1 =
          intel_sub_group_f16_f16_matrix_mad_k16(
              weight_fragment, value_fragment0,
              output_accumulator0_part1);
      output_accumulator1_part1 =
          intel_sub_group_f16_f16_matrix_mad_k16(
              weight_fragment, value_fragment1,
              output_accumulator1_part1);
    }
#if defined(IQ36_STOCK256_PARTIALS)
    // Stock accumulates its F16 value product in OUTPUT_TYPE.  Preserve a
    // hardware-friendly block16 rounding approximation while retaining XMX.
    output_accumulator0 = convert_float8(
        convert_half8_rte(output_accumulator0));
    output_accumulator1 = convert_float8(
        convert_half8_rte(output_accumulator1));
    output_accumulator0_part1 = convert_float8(
        convert_half8_rte(output_accumulator0_part1));
    output_accumulator1_part1 = convert_float8(
        convert_half8_rte(output_accumulator1_part1));
#endif
#else
    output_accumulator0 = intel_sub_group_f16_f16_matrix_mad_k16(
        weight_fragment, value_fragment0, output_accumulator0);
    output_accumulator1 = intel_sub_group_f16_f16_matrix_mad_k16(
        weight_fragment, value_fragment1, output_accumulator1);
#endif
#endif
  }
#endif

#if defined(IQ36_DUAL256_REDUCTION)
  #pragma unroll
  for (uint head = 0U; head < IQ36_GQA_GROUP; ++head) {
    output_accumulator0[head] =
        output_accumulator0[head] * local_block_max[head] +
        output_accumulator0_part1[head] * local_block_sum[head];
    output_accumulator1[head] =
        output_accumulator1[head] * local_block_max[head] +
        output_accumulator1_part1[head] * local_block_sum[head];
  }
#endif

  if (subgroup == 0U && lane < IQ36_GQA_GROUP) {
#if defined(IQ36_STOCK256_PARTIALS)
    #pragma unroll
    for (uint partial = 0U; partial < 2U; ++partial) {
      const ulong head_base = iq36_workspace_stock_partial_head(
          batch, kv_head, chunk, lane, partial);
      workspace[head_base] =
          (OUTPUT0_TYPE)local_global_max[
              partial * IQ36_GQA_GROUP + lane];
      workspace[head_base + OUTPUT0_PITCHES[3]] =
          (OUTPUT0_TYPE)local_global_sum[
              partial * IQ36_GQA_GROUP + lane];
    }
#else
    const ulong head_base = iq36_workspace_head(
        batch, kv_head, chunk, lane);
    workspace[head_base] =
        (OUTPUT0_TYPE)local_global_max[lane];
    workspace[head_base + OUTPUT0_PITCHES[3]] =
        (OUTPUT0_TYPE)local_global_sum[lane];
#endif
  }
  #pragma unroll
  for (uint head = 0U; head < IQ36_GQA_GROUP; ++head) {
#if defined(IQ36_STOCK256_PARTIALS)
    const ulong head_base0 = iq36_workspace_stock_partial_head(
        batch, kv_head, chunk, head, 0U);
    const ulong head_base1 = iq36_workspace_stock_partial_head(
        batch, kv_head, chunk, head, 1U);
    workspace[head_base0 +
        (IQ36_PARTIAL_OUTPUT_OFFSET + dim0) * OUTPUT0_PITCHES[3]] =
            (OUTPUT0_TYPE)convert_float(convert_half_rte(
                output_accumulator0[head]));
    workspace[head_base0 +
        (IQ36_PARTIAL_OUTPUT_OFFSET + dim1) * OUTPUT0_PITCHES[3]] =
            (OUTPUT0_TYPE)convert_float(convert_half_rte(
                output_accumulator1[head]));
    workspace[head_base1 +
        (IQ36_PARTIAL_OUTPUT_OFFSET + dim0) * OUTPUT0_PITCHES[3]] =
            (OUTPUT0_TYPE)convert_float(convert_half_rte(
                output_accumulator0_part1[head]));
    workspace[head_base1 +
        (IQ36_PARTIAL_OUTPUT_OFFSET + dim1) * OUTPUT0_PITCHES[3]] =
            (OUTPUT0_TYPE)convert_float(convert_half_rte(
                output_accumulator1_part1[head]));
#else
    const ulong head_base = iq36_workspace_head(
        batch, kv_head, chunk, head);
    workspace[head_base +
        (IQ36_PARTIAL_OUTPUT_OFFSET + dim0) * OUTPUT0_PITCHES[3]] =
            (OUTPUT0_TYPE)output_accumulator0[head];
    workspace[head_base +
        (IQ36_PARTIAL_OUTPUT_OFFSET + dim1) * OUTPUT0_PITCHES[3]] =
            (OUTPUT0_TYPE)output_accumulator1[head];
#endif
  }
  barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);

  if (local_id == 0U) {
    // Publish this work-group's workspace rows before joining the device-wide
    // arrival sequence.  A work-group barrier alone only provides work-group
    // scope and allowed the apparent "last" group to consume stale partials.
    atomic_work_item_fence(
        CLK_GLOBAL_MEM_FENCE, memory_order_release, memory_scope_device);
    volatile __global unsigned int* counter =
        (volatile __global unsigned int*)&hot_key_bits[
            iq36_arrival_counter_index(batch, kv_head)];
    const uint count_mask = 0x1ffU;
    const uint generation = key_tokens << 9U;
    uint observed = atomic_or(counter, 0U);
    while ((observed & ~count_mask) != generation) {
      const uint replaced = atomic_cmpxchg(counter, observed, generation);
      if (replaced == observed) {
        observed = generation;
        break;
      }
      observed = replaced;
    }
    const uint previous = atomic_inc(counter) & count_mask;
    local_is_last[0] = previous + 1U == chunk_count;
    if (local_is_last[0] != 0U) {
      atomic_work_item_fence(
          CLK_GLOBAL_MEM_FENCE, memory_order_acquire, memory_scope_device);
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (local_is_last[0] == 0U) return;

#if defined(IQ36_STOCK_PARTITION_ARITHMETIC)
  // Match stock sdpa_opt_finalization_stage.  The stock work-group has 256
  // work-items (sixteen subgroups); this 128-item carrier emulates those
  // sixteen reduction groups two at a time, including each lane's +256
  // partition stride, then combines F16 partition outputs in index order.
  const uint partial_count = chunk_count;
  #pragma unroll
  for (uint head = 0U; head < IQ36_GQA_GROUP; ++head) {
    for (uint group = subgroup; group < 16U; group += 8U) {
      float lane_maximum = -INFINITY;
      for (uint partial_chunk = group * 16U + lane;
           partial_chunk < partial_count; partial_chunk += 256U) {
        const ulong head_base = iq36_workspace_head(
            batch, kv_head, partial_chunk, head);
        lane_maximum = fmax(
            lane_maximum, convert_float(workspace[head_base]));
      }
      const float group_maximum = sub_group_reduce_max(lane_maximum);
      if (lane == 0U) local_block_max[group] = group_maximum;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (subgroup == 0U) {
      const float value = lane < 16U ? local_block_max[lane] : -INFINITY;
      const float maximum = sub_group_reduce_max(value);
      if (lane == 0U) local_global_max[head] = maximum;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const float global_maximum = local_global_max[head];
    for (uint group = subgroup; group < 16U; group += 8U) {
      float lane_sum = 0.0f;
      for (uint partial_chunk = group * 16U + lane;
           partial_chunk < partial_count; partial_chunk += 256U) {
        const ulong head_base = iq36_workspace_head(
            batch, kv_head, partial_chunk, head);
        const float partial_maximum =
            convert_float(workspace[head_base]);
        const float partial_sum = convert_float(
            workspace[head_base + OUTPUT0_PITCHES[3]]);
        const float adjusted_sum = partial_sum * native_exp(
            partial_maximum - global_maximum);
        local_score_weight[partial_chunk] = adjusted_sum;
        lane_sum += adjusted_sum;
      }
      const float group_sum = sub_group_reduce_add(lane_sum);
      if (lane == 0U) local_block_sum[group] = group_sum;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (subgroup == 0U) {
      const float value = lane < 16U ? local_block_sum[lane] : 0.0f;
      const float sum = sub_group_reduce_add(value);
      if (lane == 0U) local_global_sum[head] = sum;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    float accumulated0 = 0.0f;
    float accumulated1 = 0.0f;
    for (uint partial_chunk = 0U;
         partial_chunk < partial_count; ++partial_chunk) {
      const ulong head_base = iq36_workspace_head(
          batch, kv_head, partial_chunk, head);
      const float adjusted_sum = local_score_weight[partial_chunk];
      accumulated0 += convert_float(workspace[head_base +
          (IQ36_PARTIAL_OUTPUT_OFFSET + dim0) * OUTPUT0_PITCHES[3]]) *
          adjusted_sum;
      accumulated1 += convert_float(workspace[head_base +
          (IQ36_PARTIAL_OUTPUT_OFFSET + dim1) * OUTPUT0_PITCHES[3]]) *
          adjusted_sum;
    }
    const half denominator = convert_half_rte(local_global_sum[head]);
    const half numerator0 = convert_half_rte(accumulated0);
    const half numerator1 = convert_half_rte(accumulated1);
    const half result0 = denominator == (half)0.0h
        ? (half)0.0h : numerator0 / denominator;
    const half result1 = denominator == (half)0.0h
        ? (half)0.0h : numerator1 / denominator;
    const uint query_head = kv_head * IQ36_GQA_GROUP + head;
    const ulong output_base = OUTPUT1_OFFSET +
        (ulong)batch * OUTPUT1_PITCHES[0] +
        (ulong)query_head * OUTPUT1_PITCHES[1];
    output[output_base + (ulong)dim0 * OUTPUT1_PITCHES[3]] =
        (OUTPUT1_TYPE)result0;
    output[output_base + (ulong)dim1 * OUTPUT1_PITCHES[3]] =
        (OUTPUT1_TYPE)result1;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
#else
  #pragma unroll
  for (uint head = 0U; head < IQ36_GQA_GROUP; ++head) {
    float running_max = -INFINITY;
    float running_sum = 0.0f;
    float accumulated0 = 0.0f;
    float accumulated1 = 0.0f;
    const uint partial_count =
#if defined(IQ36_STOCK256_PARTIALS)
        (key_tokens + 255U) / 256U;
#else
        chunk_count;
#endif
    for (uint partial_chunk = 0U;
         partial_chunk < partial_count; ++partial_chunk) {
#if defined(IQ36_STOCK256_PARTIALS)
      const ulong head_base = iq36_workspace_stock_partial_head(
          batch, kv_head, partial_chunk / 2U, head,
          partial_chunk & 1U);
#else
      const ulong head_base = iq36_workspace_head(
          batch, kv_head, partial_chunk, head);
#endif
      const float partial_max = convert_float(workspace[head_base]);
      const float partial_sum = convert_float(
          workspace[head_base + OUTPUT0_PITCHES[3]]);
      const float next_max = fmax(running_max, partial_max);
#if defined(IQ36_STOCK256_PARTIALS)
      const float previous_scale = native_exp(
          (running_max - next_max) * IQ36_ATTENTION_SCALE);
      const float partial_scale = native_exp(
          (partial_max - next_max) * IQ36_ATTENTION_SCALE);
      const float partial_weight = partial_sum * partial_scale;
#else
      const float previous_scale = native_exp2(
          (running_max - next_max) * IQ36_DECODE_EXP2_SCALE);
      const float partial_scale = native_exp2(
          (partial_max - next_max) * IQ36_DECODE_EXP2_SCALE);
      const float partial_weight = partial_scale;
#endif
      accumulated0 =
          accumulated0 * previous_scale +
          convert_float(workspace[head_base +
              (IQ36_PARTIAL_OUTPUT_OFFSET + dim0) *
                  OUTPUT0_PITCHES[3]]) *
              partial_weight;
      accumulated1 =
          accumulated1 * previous_scale +
          convert_float(workspace[head_base +
              (IQ36_PARTIAL_OUTPUT_OFFSET + dim1) *
                  OUTPUT0_PITCHES[3]]) *
              partial_weight;
      running_sum = running_sum * previous_scale +
          partial_sum * partial_scale;
      running_max = next_max;
    }
    const float reciprocal = native_recip(running_sum);
    const uint query_head = kv_head * IQ36_GQA_GROUP + head;
#if defined(IQ36_TOKEN_MAJOR_OUTPUT)
    const ulong output_base = OUTPUT1_OFFSET +
        (ulong)batch * OUTPUT1_PITCHES[0] +
        (ulong)query_head * OUTPUT1_PITCHES[2];
#if defined(IQ36_FUSED_GATE_OUTPUT)
    const ulong gate_base = INPUT13_OFFSET +
        (ulong)batch * INPUT13_PITCHES[0] +
        (ulong)query_head * INPUT13_PITCHES[2];
    output[output_base + (ulong)dim0 * OUTPUT1_PITCHES[3]] =
        iq36_gated_attention_value(
            running_sum == 0.0f ? 0.0f : accumulated0 * reciprocal,
            raw_gate, gate_base + (ulong)dim0 * INPUT13_PITCHES[3]);
    output[output_base + (ulong)dim1 * OUTPUT1_PITCHES[3]] =
        iq36_gated_attention_value(
            running_sum == 0.0f ? 0.0f : accumulated1 * reciprocal,
            raw_gate, gate_base + (ulong)dim1 * INPUT13_PITCHES[3]);
#else
    output[output_base + (ulong)dim0 * OUTPUT1_PITCHES[3]] =
        (OUTPUT1_TYPE)(running_sum == 0.0f
            ? 0.0f : accumulated0 * reciprocal);
    output[output_base + (ulong)dim1 * OUTPUT1_PITCHES[3]] =
        (OUTPUT1_TYPE)(running_sum == 0.0f
            ? 0.0f : accumulated1 * reciprocal);
#endif
#else
    const ulong output_base = OUTPUT1_OFFSET +
        (ulong)batch * OUTPUT1_PITCHES[0] +
        (ulong)query_head * OUTPUT1_PITCHES[1];
    output[output_base + (ulong)dim0 * OUTPUT1_PITCHES[3]] =
        (OUTPUT1_TYPE)(running_sum == 0.0f
            ? 0.0f : accumulated0 * reciprocal);
    output[output_base + (ulong)dim1 * OUTPUT1_PITCHES[3]] =
        (OUTPUT1_TYPE)(running_sum == 0.0f
            ? 0.0f : accumulated1 * reciprocal);
#endif
  }
#endif

  // The last work-group is the sole decode state owner for this KV head.
  const uint state_dim0 = local_id * 2U;
  const uint state_dim1 = state_dim0 + 1U;
  const uint global_current_token = past_tokens;
  const uint slot = iq36_hot_slot(global_current_token);
  const ulong current_key_base = INPUT3_OFFSET +
      (ulong)batch * INPUT3_PITCHES[0] +
      (ulong)kv_head * INPUT3_PITCHES[1];
  const ulong current_value_base = iq36_current_value_index(
      batch, kv_head, 0U, 0U);

  if (cold_append_tokens != 0U) {
#if defined(IQ36_DIRECT_I8_GROUP4_FULL_COLD)
    const uint global_token = cold_tokens;
    #pragma unroll
    for (uint pass = 0U; pass < 2U; ++pass) {
      const uint cold_dim = pass * 128U + subgroup * 16U + lane;
      const uint key_scale_group = cold_dim / IQ36_KEY_QUANT_GROUP;
      const uint value_scale_group = cold_dim / IQ36_VALUE_QUANT_GROUP;
      const float key_value = iq36_partial_load_key(
          batch, kv_head, global_token, past_tokens, cold_tokens,
          cold_dim, hot_key_bits, current_key, cold_key,
          cold_key_scale_bytes);
      const float value_value = iq36_partial_load_value(
          batch, kv_head, global_token, past_tokens, cold_tokens,
          cold_dim, hot_value, current_value, cold_value,
          cold_value_scale_bytes);
      float key_max = fabs(key_value);
      key_max = fmax(key_max, sub_group_shuffle_xor(key_max, 1U));
#if IQ36_KEY_QUANT_GROUP == 4
      key_max = fmax(key_max, sub_group_shuffle_xor(key_max, 2U));
#endif
      float value_max = fabs(value_value);
      value_max = fmax(
          value_max, sub_group_shuffle_xor(value_max, 1U));
      value_max = fmax(
          value_max, sub_group_shuffle_xor(value_max, 2U));
      const float key_scale = key_max == 0.0f
          ? 1.0f : key_max / 127.0f;
      const float value_scale = value_max == 0.0f
          ? 1.0f : value_max / 127.0f;
      const int key_quantized = clamp(
          (int)rint(key_value / key_scale), -127, 127);
      const int value_quantized = clamp(
          (int)rint(value_value / value_scale), -127, 127);
      const ulong cold_key_index = OUTPUT2_OFFSET +
          (ulong)batch * OUTPUT2_PITCHES[0] +
          (ulong)kv_head * OUTPUT2_PITCHES[1] +
          (ulong)cold_dim * OUTPUT2_PITCHES[3];
      const ulong cold_value_index = OUTPUT3_OFFSET +
          (ulong)batch * OUTPUT3_PITCHES[0] +
          (ulong)kv_head * OUTPUT3_PITCHES[1] +
          (ulong)cold_dim * OUTPUT3_PITCHES[3];
      cold_key_append[cold_key_index] = (OUTPUT2_TYPE)key_quantized;
      cold_value_append[cold_value_index] =
          (OUTPUT3_TYPE)value_quantized;
      if (fixed_cold_state) {
        iq36_direct_store_cold_key(
            batch, kv_head, cold_tokens, cold_dim,
            (char)key_quantized, cold_key);
        iq36_direct_store_cold_value(
            batch, kv_head, cold_tokens, cold_dim,
            (char)value_quantized, cold_value);
      }
      if ((lane % IQ36_KEY_QUANT_GROUP) == 0U) {
        const ushort key_bits = as_ushort(convert_half_rte(key_scale));
        const uint scale_x = key_scale_group * 2U;
        const ulong key_scale_index = OUTPUT4_OFFSET +
            (ulong)batch * OUTPUT4_PITCHES[0] +
            (ulong)kv_head * OUTPUT4_PITCHES[1] +
            (ulong)scale_x * OUTPUT4_PITCHES[3];
        cold_key_scale_append[key_scale_index] =
            (OUTPUT4_TYPE)(key_bits & 0xffU);
        cold_key_scale_append[
            key_scale_index + OUTPUT4_PITCHES[3]] =
                (OUTPUT4_TYPE)(key_bits >> 8);
        if (fixed_cold_state) {
          iq36_direct_store_cold_key_scale(
              batch, kv_head, cold_tokens, key_scale_group,
              as_half(key_bits), cold_key_scale_bytes);
        }
      }
      if ((lane % IQ36_VALUE_QUANT_GROUP) == 0U) {
        const ushort value_bits = as_ushort(convert_half_rte(value_scale));
        const uint scale_x = value_scale_group * 2U;
        const ulong value_scale_index = OUTPUT5_OFFSET +
            (ulong)batch * OUTPUT5_PITCHES[0] +
            (ulong)kv_head * OUTPUT5_PITCHES[1] +
            (ulong)scale_x * OUTPUT5_PITCHES[3];
        cold_value_scale_append[value_scale_index] =
            (OUTPUT5_TYPE)(value_bits & 0xffU);
        cold_value_scale_append[
            value_scale_index + OUTPUT5_PITCHES[3]] =
                (OUTPUT5_TYPE)(value_bits >> 8);
        if (fixed_cold_state) {
          iq36_direct_store_cold_value_scale(
              batch, kv_head, cold_tokens, value_scale_group,
              as_half(value_bits), cold_value_scale_bytes);
        }
      }
    }
#else
    const uint global_token = cold_tokens;
    const uint block = subgroup;
    const float key0 = iq36_partial_load_key(
        batch, kv_head, global_token, past_tokens, cold_tokens,
        state_dim0, hot_key_bits, current_key, cold_key,
        cold_key_scale_bytes);
    const float key1 = iq36_partial_load_key(
        batch, kv_head, global_token, past_tokens, cold_tokens,
        state_dim1, hot_key_bits, current_key, cold_key,
        cold_key_scale_bytes);
    const float value0 = iq36_partial_load_value(
        batch, kv_head, global_token, past_tokens, cold_tokens,
        state_dim0, hot_value, current_value, cold_value,
        cold_value_scale_bytes);
    const float value1 = iq36_partial_load_value(
        batch, kv_head, global_token, past_tokens, cold_tokens,
        state_dim1, hot_value, current_value, cold_value,
        cold_value_scale_bytes);
    const float key_max = sub_group_reduce_max(
        fmax(fabs(key0), fabs(key1)));
    const float value_max = sub_group_reduce_max(
        fmax(fabs(value0), fabs(value1)));
    const float key_scale = key_max == 0.0f
        ? 1.0f : key_max / 127.0f;
    const float value_scale = value_max == 0.0f
        ? 1.0f : value_max / 127.0f;
    const int key_q0 = clamp(
        (int)rint(key0 / key_scale), -127, 127);
    const int key_q1 = clamp(
        (int)rint(key1 / key_scale), -127, 127);
    const int value_q0 = clamp(
        (int)rint(value0 / value_scale), -127, 127);
    const int value_q1 = clamp(
        (int)rint(value1 / value_scale), -127, 127);
    const ulong cold_key_base = OUTPUT2_OFFSET +
        (ulong)batch * OUTPUT2_PITCHES[0] +
        (ulong)kv_head * OUTPUT2_PITCHES[1];
    const ulong cold_value_base = OUTPUT3_OFFSET +
        (ulong)batch * OUTPUT3_PITCHES[0] +
        (ulong)kv_head * OUTPUT3_PITCHES[1];
    cold_key_append[cold_key_base +
        (ulong)state_dim0 * OUTPUT2_PITCHES[3]] =
            (OUTPUT2_TYPE)key_q0;
    cold_key_append[cold_key_base +
        (ulong)state_dim1 * OUTPUT2_PITCHES[3]] =
            (OUTPUT2_TYPE)key_q1;
    cold_value_append[cold_value_base +
        (ulong)state_dim0 * OUTPUT3_PITCHES[3]] =
            (OUTPUT3_TYPE)value_q0;
    cold_value_append[cold_value_base +
        (ulong)state_dim1 * OUTPUT3_PITCHES[3]] =
            (OUTPUT3_TYPE)value_q1;
    if (fixed_cold_state) {
#if defined(IQ36_DIRECT_I8_FIXED_LAYOUT)
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
#else
      const uint state_row = cold_tokens + 1U;
      const ulong fixed_key_base = INPUT5_OFFSET +
          (ulong)batch * INPUT5_PITCHES[0] +
          (ulong)kv_head * INPUT5_PITCHES[1] +
          (ulong)state_row * INPUT5_PITCHES[2];
      const ulong fixed_value_base = INPUT6_OFFSET +
          (ulong)batch * INPUT6_PITCHES[0] +
          (ulong)kv_head * INPUT6_PITCHES[1] +
          (ulong)state_row * INPUT6_PITCHES[2];
      cold_key[fixed_key_base +
          (ulong)state_dim0 * INPUT5_PITCHES[3]] = (INPUT5_TYPE)key_q0;
      cold_key[fixed_key_base +
          (ulong)state_dim1 * INPUT5_PITCHES[3]] = (INPUT5_TYPE)key_q1;
      cold_value[fixed_value_base +
          (ulong)state_dim0 * INPUT6_PITCHES[3]] = (INPUT6_TYPE)value_q0;
      cold_value[fixed_value_base +
          (ulong)state_dim1 * INPUT6_PITCHES[3]] = (INPUT6_TYPE)value_q1;
#endif
    }
    if (lane < 2U) {
      const ushort key_bits = as_ushort(convert_half_rte(key_scale));
      const ushort value_bits = as_ushort(convert_half_rte(value_scale));
      const uint scale_x = block * 2U + lane;
      const ulong key_scale_index = OUTPUT4_OFFSET +
          (ulong)batch * OUTPUT4_PITCHES[0] +
          (ulong)kv_head * OUTPUT4_PITCHES[1] +
          (ulong)scale_x * OUTPUT4_PITCHES[3];
      const ulong value_scale_index = OUTPUT5_OFFSET +
          (ulong)batch * OUTPUT5_PITCHES[0] +
          (ulong)kv_head * OUTPUT5_PITCHES[1] +
          (ulong)scale_x * OUTPUT5_PITCHES[3];
      cold_key_scale_append[key_scale_index] = (OUTPUT4_TYPE)(
          lane == 0U ? key_bits & 0xffU : key_bits >> 8);
      cold_value_scale_append[value_scale_index] = (OUTPUT5_TYPE)(
          lane == 0U ? value_bits & 0xffU : value_bits >> 8);
      if (fixed_cold_state) {
#if defined(IQ36_DIRECT_I8_FIXED_LAYOUT)
        if (lane == 0U) {
          iq36_direct_store_cold_key_scale(
              batch, kv_head, cold_tokens, block,
              as_half(key_bits), cold_key_scale_bytes);
          iq36_direct_store_cold_value_scale(
              batch, kv_head, cold_tokens, block,
              as_half(value_bits), cold_value_scale_bytes);
        }
#else
        const uint state_row = cold_tokens + 1U;
        const ulong fixed_key_scale = INPUT7_OFFSET +
            (ulong)batch * INPUT7_PITCHES[0] +
            (ulong)kv_head * INPUT7_PITCHES[1] +
            (ulong)state_row * INPUT7_PITCHES[2] +
            (ulong)scale_x * INPUT7_PITCHES[3];
        const ulong fixed_value_scale = INPUT8_OFFSET +
            (ulong)batch * INPUT8_PITCHES[0] +
            (ulong)kv_head * INPUT8_PITCHES[1] +
            (ulong)state_row * INPUT8_PITCHES[2] +
            (ulong)scale_x * INPUT8_PITCHES[3];
        cold_key_scale_bytes[fixed_key_scale] = (INPUT7_TYPE)(
            lane == 0U ? key_bits & 0xffU : key_bits >> 8);
        cold_value_scale_bytes[fixed_value_scale] = (INPUT8_TYPE)(
            lane == 0U ? value_bits & 0xffU : value_bits >> 8);
#endif
      }
    }
#endif
  }

  if (fixed_cold_state && local_id < 3U) {
    const uint divisor = local_id == 0U ? 1U :
        (local_id == 1U ? 128U : 16384U);
    cold_key[INPUT5_OFFSET +
        (ulong)batch * INPUT5_PITCHES[0] +
        (ulong)kv_head * INPUT5_PITCHES[1] +
        (ulong)local_id * INPUT5_PITCHES[3]] =
            (INPUT5_TYPE)((desired_cold_tokens / divisor) % 128U);
  }

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
  __global half* dense_hot_key = (__global half*)&hot_key_bits[
      iq36_hot_key_dense_i32_base(batch, kv_head)];
  const ulong dense_key_index =
      (ulong)slot * IQ36_HEAD_DIM + state_dim0;
  dense_hot_key[dense_key_index] = (half)current_key[
      current_key_base + (ulong)state_dim0 * INPUT3_PITCHES[3]];
  dense_hot_key[dense_key_index + 1U] = (half)current_key[
      current_key_base + (ulong)state_dim1 * INPUT3_PITCHES[3]];
  const ulong hot_value_base = INPUT2_OFFSET +
      (ulong)batch * INPUT2_PITCHES[0] +
      (ulong)kv_head * INPUT2_PITCHES[1] +
      (ulong)slot * INPUT2_PITCHES[2];
  hot_value[hot_value_base +
      (ulong)state_dim0 * INPUT2_PITCHES[3]] =
          (INPUT2_TYPE)current_value[current_value_base +
              (ulong)state_dim0 * INPUT4_PITCHES[3]];
  hot_value[hot_value_base +
      (ulong)state_dim1 * INPUT2_PITCHES[3]] =
          (INPUT2_TYPE)current_value[current_value_base +
              (ulong)state_dim1 * INPUT4_PITCHES[3]];
#if defined(IQ36_DIMENSION_MAJOR_VALUE_PLANE)
  iq36_direct_store_hot_value_dimension(
      batch, kv_head, slot, state_dim0,
      (half)current_value[current_value_base +
          (ulong)state_dim0 * INPUT4_PITCHES[3]],
      hot_key_bits);
  iq36_direct_store_hot_value_dimension(
      batch, kv_head, slot, state_dim1,
      (half)current_value[current_value_base +
          (ulong)state_dim1 * INPUT4_PITCHES[3]],
      hot_key_bits);
#endif
#endif
}

#endif
