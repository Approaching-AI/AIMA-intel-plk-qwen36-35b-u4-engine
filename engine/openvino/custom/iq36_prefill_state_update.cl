// Prefill-side state owner for the stock-prefill/custom-decode composition.
//
// The stock SDPA branch owns prompt attention.  This kernel performs only the
// request-state transition needed by the custom decode path: pack current K,
// retain current V, and encode rows that leave the exact hot window.  It uses
// the unified six-output ABI; workspace and attention are intentionally dead
// during prefill and are selected away by the outer GPU merge.

#if defined(IQ36_UNIFIED_SHAPE_SPECIALIZATION) && \
    defined(IQ36_BUILD_PREFILL_ONLY)

inline ulong iq36_prefill_arrival_counter_index(
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
    __global OUTPUT0_TYPE* workspace,
    __global OUTPUT1_TYPE* output,
    __global OUTPUT2_TYPE* cold_key_append,
    __global OUTPUT3_TYPE* cold_value_append,
    __global OUTPUT4_TYPE* cold_key_scale_append,
    __global OUTPUT5_TYPE* cold_value_scale_append) {
  const uint local_id = (uint)get_local_id(0);
  const uint subgroup = (uint)get_sub_group_id();
  const uint lane = (uint)get_sub_group_local_id();
  const uint query_tile = (uint)get_group_id(1);
  const uint batch_head = (uint)get_group_id(2);
  const uint batch = batch_head / IQ36_Q_HEADS;
  const uint query_head = batch_head - batch * IQ36_Q_HEADS;
  const uint kv_head = query_head / IQ36_GQA_GROUP;
  const uint query_tokens = IQ36_STATIC_QUERY_TOKENS;
  // Fixed product buckets deliberately use a scalar mask shape.  The exact
  // cumulative length is carried in input 12 for both initial and
  // continuation prefill, so deriving it from INPUT9_DIMS would underflow
  // past_tokens on every multi-token chunk.
  const uint key_tokens =
      (uint)decode_length_carrier[INPUT12_OFFSET];
  const uint past_tokens = key_tokens - query_tokens;
  const uint desired_cold_tokens = key_tokens > IQ36_HOT_WINDOW
      ? key_tokens - IQ36_HOT_WINDOW : 0U;
  const bool fixed_cold_state = (uint)INPUT5_DIMS[2] > key_tokens;
  const uint previous_cold_tokens = past_tokens > IQ36_HOT_WINDOW
      ? past_tokens - IQ36_HOT_WINDOW : 0U;
  const uint cold_append_tokens = fixed_cold_state
      ? desired_cold_tokens - previous_cold_tokens
      : (uint)eviction_count[INPUT11_OFFSET];
  const uint cold_tokens = desired_cold_tokens - cold_append_tokens;
  const uint query_begin = query_tile * IQ36_PREFILL_QUERY_TILE;
  const uint query_count = query_begin < query_tokens
      ? min((uint)IQ36_PREFILL_QUERY_TILE, query_tokens - query_begin) : 0U;

  // A tiny graph result consumes this marker in the static prefill graph.  It
  // keeps the state-only primitive live without routing Q/K/V through control
  // flow or copying the unused attention tensor to the host.
  if (local_id == 0U) {
    workspace[OUTPUT0_OFFSET +
        (ulong)batch * OUTPUT0_PITCHES[0] +
        (ulong)query_head * OUTPUT0_PITCHES[1] +
        (ulong)query_tile * OUTPUT0_PITCHES[2]] = (OUTPUT0_TYPE)0;
  }

  // Only the first GQA head owns each KV state.  The other seven head groups
  // are present solely because the shared ABI drives work sizes from the
  // query-head-shaped workspace.
  if ((query_head % IQ36_GQA_GROUP) != 0U || query_count == 0U) return;

  if (query_tile == 0U && local_id == 0U) {
    hot_key_bits[iq36_prefill_arrival_counter_index(batch, kv_head)] =
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
    const ulong current_row_value = INPUT4_OFFSET +
        (ulong)batch * INPUT4_PITCHES[0] +
        (ulong)kv_head * INPUT4_PITCHES[1] +
        (ulong)token * INPUT4_PITCHES[2];

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
            (ulong)(slot / IQ36_KEY_TILE_TOKENS) * INPUT1_PITCHES[2] +
            (ulong)key_word * INPUT1_PITCHES[3];
        hot_key_bits[hot_key_index] = (INPUT1_TYPE)as_int((half2)(
            (half)current_key[current_row_key +
                (ulong)key_dim * INPUT3_PITCHES[3]],
            (half)current_key[current_row_key +
                (ulong)(key_dim + 1U) * INPUT3_PITCHES[3]]));
        // The stock decode microkernel consumes the exact token-major F16
        // plane that follows the packed block16 plane.  Keep both views in
        // lockstep so the same request can switch from stock prefill to the
        // two-work-group GQA decode owner without a conversion dispatch.
        __global half* dense_key = (__global half*)&hot_key_bits[
            iq36_hot_key_dense_i32_base(batch, kv_head)];
        const ulong dense_key_index = (ulong)slot * IQ36_HEAD_DIM + key_dim;
        dense_key[dense_key_index] = (half)current_key[
            current_row_key + (ulong)key_dim * INPUT3_PITCHES[3]];
        dense_key[dense_key_index + 1U] = (half)current_key[
            current_row_key +
                (ulong)(key_dim + 1U) * INPUT3_PITCHES[3]];
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
      const int key_q0 = clamp((int)rint(key0 / key_scale), -127, 127);
      const int key_q1 = clamp((int)rint(key1 / key_scale), -127, 127);
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
          (ulong)dim0 * OUTPUT2_PITCHES[3]] = (OUTPUT2_TYPE)key_q0;
      cold_key_append[cold_key_base +
          (ulong)dim1 * OUTPUT2_PITCHES[3]] = (OUTPUT2_TYPE)key_q1;
      cold_value_append[cold_value_base +
          (ulong)dim0 * OUTPUT3_PITCHES[3]] = (OUTPUT3_TYPE)value_q0;
      cold_value_append[cold_value_base +
          (ulong)dim1 * OUTPUT3_PITCHES[3]] = (OUTPUT3_TYPE)value_q1;

      if (fixed_cold_state) {
        const uint state_row = cold_tokens + token + 1U;
        const ulong fixed_key_base = INPUT5_OFFSET +
            (ulong)batch * INPUT5_PITCHES[0] +
            (ulong)kv_head * INPUT5_PITCHES[1] +
            (ulong)state_row * INPUT5_PITCHES[2];
        const ulong fixed_value_base = INPUT6_OFFSET +
            (ulong)batch * INPUT6_PITCHES[0] +
            (ulong)kv_head * INPUT6_PITCHES[1] +
            (ulong)state_row * INPUT6_PITCHES[2];
        cold_key[fixed_key_base +
            (ulong)dim0 * INPUT5_PITCHES[3]] = (INPUT5_TYPE)key_q0;
        cold_key[fixed_key_base +
            (ulong)dim1 * INPUT5_PITCHES[3]] = (INPUT5_TYPE)key_q1;
        cold_value[fixed_value_base +
            (ulong)dim0 * INPUT6_PITCHES[3]] = (INPUT6_TYPE)value_q0;
        cold_value[fixed_value_base +
            (ulong)dim1 * INPUT6_PITCHES[3]] = (INPUT6_TYPE)value_q1;
      }

      if (lane < 2U) {
        const ushort key_bits = as_ushort(convert_half_rte(key_scale));
        const ushort value_bits = as_ushort(convert_half_rte(value_scale));
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
        if (fixed_cold_state) {
          const uint state_row = cold_tokens + token + 1U;
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
        }
      }
    }
  }

  if (fixed_cold_state && query_tile == 0U && local_id < 3U) {
    const uint divisor = local_id == 0U ? 1U :
        (local_id == 1U ? 128U : 16384U);
    cold_key[INPUT5_OFFSET +
        (ulong)batch * INPUT5_PITCHES[0] +
        (ulong)kv_head * INPUT5_PITCHES[1] +
        (ulong)local_id * INPUT5_PITCHES[3]] =
            (INPUT5_TYPE)((desired_cold_tokens / divisor) % 128U);
  }
}

#endif
