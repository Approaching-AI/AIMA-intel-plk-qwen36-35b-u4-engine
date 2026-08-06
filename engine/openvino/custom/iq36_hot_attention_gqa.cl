#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Fixed Qwen3.6 full-attention hot-state boundary:
//   input0 Q             [B,16,Q,256]
//   input1 hot K bits    [1,2,8194,256] I32
//   input2 hot V bits    [1,2,8194,256] I32
//   input3 current K     [B,2,Q,256]
//   input4 current V     [B,2,Q,256]
//   input5 cold K        [B,2,C,256] I8
//   input6 cold V        [B,2,C,256] I8
//   input7 cold K scales [B,2,C,16] I8 (eight little-endian F16 values)
//   input8 cold V scales [B,2,C,16] I8 (eight little-endian F16 values)
//   input9 causal mask   [B,1,Q,past+Q]
//   input10 eviction template [B,2,Q,256]
//   input11 logical eviction count [1,1,1,1] I32
//   output0 attention    [B,16,Q,256]
//   output1/2 cold K/V scratch [B,2,Q,256] I8
//   output3/4 cold K/V scale scratch [B,2,Q,16] I8
//
// The integer state planes retain IEEE-F32 bit patterns through the default
// F16 GPU inference policy. Cold scale planes preserve the exact two bytes of
// each logical F16 block32 scale. Slot zero pins the first attention-sink token
// exactly. The remaining ring has one guard slot beyond the logical hot8192
// window. At every decode step the current write therefore lands in the unique
// free slot rather than overwriting a token that another workgroup can still
// read. This permits bounded in-place update without a second custom-operation
// config or a global barrier.

#define IQ36_HEAD_DIM 256U
#define IQ36_Q_HEADS 16U
#define IQ36_KV_HEADS 2U
#define IQ36_GQA_GROUP 8U
#define IQ36_HOT_WINDOW 8192U
#define IQ36_SINK_TOKENS 1U
#define IQ36_RING_CAPACITY 8193U
#define IQ36_HOT_CAPACITY (IQ36_SINK_TOKENS + IQ36_RING_CAPACITY)
#define IQ36_ATTENTION_SCALE 0.0625f
#define IQ36_LOG2_E 1.442695f
#define IQ36_EXP2_SCALE (IQ36_ATTENTION_SCALE * IQ36_LOG2_E)

inline uint iq36_hot_slot(const uint token) {
  return token < IQ36_SINK_TOKENS
      ? token
      : IQ36_SINK_TOKENS +
            (token - IQ36_SINK_TOKENS) % IQ36_RING_CAPACITY;
}

inline float iq36_load_key(
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
    const uint value_index = INPUT1_OFFSET +
        batch * INPUT1_PITCHES[0] + kv_head * INPUT1_PITCHES[1] +
        slot * INPUT1_PITCHES[2] + dim * INPUT1_PITCHES[3];
    return as_float((int)hot_key_bits[value_index]);
  }
  if (token < cold_tokens) {
    const uint scale_byte = (dim >> 5) << 1;
    const uint cold_row = token + 1U;
    const uint value_index = INPUT5_OFFSET +
        batch * INPUT5_PITCHES[0] + kv_head * INPUT5_PITCHES[1] +
        cold_row * INPUT5_PITCHES[2] + dim * INPUT5_PITCHES[3];
    const uint scale_index = INPUT7_OFFSET +
        batch * INPUT7_PITCHES[0] + kv_head * INPUT7_PITCHES[1] +
        cold_row * INPUT7_PITCHES[2] + scale_byte * INPUT7_PITCHES[3];
    const ushort scale_bits =
        (ushort)(uchar)cold_key_scale_bytes[scale_index] |
        ((ushort)(uchar)cold_key_scale_bytes[
            scale_index + INPUT7_PITCHES[3]] << 8);
    return convert_float(cold_key[value_index]) *
        convert_float(as_half(scale_bits));
  }
  if (token < past_tokens) {
    const uint slot = iq36_hot_slot(token);
    const uint value_index = INPUT1_OFFSET +
        batch * INPUT1_PITCHES[0] + kv_head * INPUT1_PITCHES[1] +
        slot * INPUT1_PITCHES[2] + dim * INPUT1_PITCHES[3];
    return as_float((int)hot_key_bits[value_index]);
  }
  const uint current_position = token - past_tokens;
  const uint value_index = INPUT3_OFFSET +
      batch * INPUT3_PITCHES[0] + kv_head * INPUT3_PITCHES[1] +
      current_position * INPUT3_PITCHES[2] + dim * INPUT3_PITCHES[3];
  return convert_float(current_key[value_index]);
}

inline float iq36_load_value(
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
    const uint value_index = INPUT2_OFFSET +
        batch * INPUT2_PITCHES[0] + kv_head * INPUT2_PITCHES[1] +
        slot * INPUT2_PITCHES[2] + dim * INPUT2_PITCHES[3];
    return as_float((int)hot_value_bits[value_index]);
  }
  if (token < cold_tokens) {
    const uint scale_byte = (dim >> 5) << 1;
    const uint cold_row = token + 1U;
    const uint value_index = INPUT6_OFFSET +
        batch * INPUT6_PITCHES[0] + kv_head * INPUT6_PITCHES[1] +
        cold_row * INPUT6_PITCHES[2] + dim * INPUT6_PITCHES[3];
    const uint scale_index = INPUT8_OFFSET +
        batch * INPUT8_PITCHES[0] + kv_head * INPUT8_PITCHES[1] +
        cold_row * INPUT8_PITCHES[2] + scale_byte * INPUT8_PITCHES[3];
    const ushort scale_bits =
        (ushort)(uchar)cold_value_scale_bytes[scale_index] |
        ((ushort)(uchar)cold_value_scale_bytes[
            scale_index + INPUT8_PITCHES[3]] << 8);
    return convert_float(cold_value[value_index]) *
        convert_float(as_half(scale_bits));
  }
  if (token < past_tokens) {
    const uint slot = iq36_hot_slot(token);
    const uint value_index = INPUT2_OFFSET +
        batch * INPUT2_PITCHES[0] + kv_head * INPUT2_PITCHES[1] +
        slot * INPUT2_PITCHES[2] + dim * INPUT2_PITCHES[3];
    return as_float((int)hot_value_bits[value_index]);
  }
  const uint current_position = token - past_tokens;
  const uint value_index = INPUT4_OFFSET +
      batch * INPUT4_PITCHES[0] + kv_head * INPUT4_PITCHES[1] +
      current_position * INPUT4_PITCHES[2] + dim * INPUT4_PITCHES[3];
  return convert_float(current_value[value_index]);
}

__attribute__((reqd_work_group_size(256, 1, 1)))
__attribute__((intel_reqd_sub_group_size(32)))
__kernel void iq36_hot_attention_gqa(
    const __global INPUT0_TYPE* query,
    __global INPUT1_TYPE* hot_key_bits,
    __global INPUT2_TYPE* hot_value_bits,
    const __global INPUT3_TYPE* current_key,
    const __global INPUT4_TYPE* current_value,
    const __global INPUT5_TYPE* cold_key,
    const __global INPUT6_TYPE* cold_value,
    const __global INPUT7_TYPE* cold_key_scale_bytes,
    const __global INPUT8_TYPE* cold_value_scale_bytes,
    const __global INPUT9_TYPE* mask,
    const __global INPUT10_TYPE* eviction_shape_template,
    const __global INPUT11_TYPE* eviction_count,
    __global OUTPUT0_TYPE* output,
    __global OUTPUT1_TYPE* cold_key_append,
    __global OUTPUT2_TYPE* cold_value_append,
    __global OUTPUT3_TYPE* cold_key_scale_append,
    __global OUTPUT4_TYPE* cold_value_scale_append) {
  const uint dim = (uint)get_local_id(0);
  const uint query_position = (uint)get_group_id(1);
  const uint batch_head = (uint)get_group_id(2);
  const uint batch = batch_head / IQ36_Q_HEADS;
  const uint query_head = batch_head - batch * IQ36_Q_HEADS;
  const uint kv_head = query_head / IQ36_GQA_GROUP;
  const uint subgroup = (uint)get_sub_group_id();
  const uint lane = (uint)get_sub_group_local_id();
  const uint query_tokens = (uint)INPUT0_DIMS[2];
  const uint key_tokens = (uint)INPUT9_DIMS[3];
  const uint softmax_chunk = query_tokens == 1U ? 256U : 128U;
  const uint past_tokens = key_tokens - query_tokens;
  const uint cold_append_tokens = (uint)eviction_count[INPUT11_OFFSET];
  const uint desired_cold_tokens =
      key_tokens > IQ36_HOT_WINDOW ? key_tokens - IQ36_HOT_WINDOW : 0U;
  // Row zero is a graph-owned physical sentinel. The old logical length is
  // bound in eviction_count before this invocation; the graph rewrites the
  // sentinel after the custom output is available.
  const uint cold_tokens = desired_cold_tokens - cold_append_tokens;

  const uint query_index = INPUT0_OFFSET +
      batch * INPUT0_PITCHES[0] + query_head * INPUT0_PITCHES[1] +
      query_position * INPUT0_PITCHES[2] + dim * INPUT0_PITCHES[3];
  const float query_value = convert_float(query[query_index]);

  __local float subgroup_sums[8];
  __local float shared_score;
  float softmax_sum = 0.0f;
  float output_accumulator = 0.0f;
  float reference_max = -INFINITY;
  for (uint chunk_begin = 0U; chunk_begin < key_tokens;
       chunk_begin += softmax_chunk) {
    const uint chunk_end = min(chunk_begin + softmax_chunk, key_tokens);
    float chunk_max = -INFINITY;
    for (uint token = chunk_begin; token < chunk_end; ++token) {
      const float key_value = iq36_load_key(
          batch, kv_head, token, past_tokens, cold_tokens, dim,
          hot_key_bits, current_key, cold_key, cold_key_scale_bytes);
      const float subgroup_sum = sub_group_reduce_add(
          query_value * key_value);
      if (lane == 0U) subgroup_sums[subgroup] = subgroup_sum;
      barrier(CLK_LOCAL_MEM_FENCE);

      if (subgroup == 0U) {
        const float value_to_sum = lane < 8U ? subgroup_sums[lane] : 0.0f;
        const float dot = sub_group_reduce_add(value_to_sum);
        if (lane == 0U) {
          const uint mask_index = INPUT9_OFFSET +
              batch * INPUT9_PITCHES[0] +
              query_position * INPUT9_PITCHES[2] +
              token * INPUT9_PITCHES[3];
          shared_score = dot +
              convert_float(mask[mask_index]) / IQ36_ATTENTION_SCALE;
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      chunk_max = fmax(chunk_max, shared_score);
    }

    const float running_max = fmax(reference_max, chunk_max);
    if (chunk_begin != 0U) {
      const float rescale = native_exp2(
          (reference_max - running_max) * IQ36_EXP2_SCALE);
      output_accumulator *= rescale;
      softmax_sum *= rescale;
    }

    for (uint token = chunk_begin; token < chunk_end; ++token) {
      const float key_value = iq36_load_key(
          batch, kv_head, token, past_tokens, cold_tokens, dim,
          hot_key_bits, current_key, cold_key, cold_key_scale_bytes);
      const float value_value = iq36_load_value(
          batch, kv_head, token, past_tokens, cold_tokens, dim,
          hot_value_bits, current_value, cold_value, cold_value_scale_bytes);
      const float subgroup_sum = sub_group_reduce_add(
          query_value * key_value);
      if (lane == 0U) subgroup_sums[subgroup] = subgroup_sum;
      barrier(CLK_LOCAL_MEM_FENCE);

      if (subgroup == 0U) {
        const float value_to_sum = lane < 8U ? subgroup_sums[lane] : 0.0f;
        const float dot = sub_group_reduce_add(value_to_sum);
        if (lane == 0U) {
          const uint mask_index = INPUT9_OFFSET +
              batch * INPUT9_PITCHES[0] +
              query_position * INPUT9_PITCHES[2] +
              token * INPUT9_PITCHES[3];
          shared_score = dot +
              convert_float(mask[mask_index]) / IQ36_ATTENTION_SCALE;
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);

      const float weight = native_exp2(
          (shared_score - running_max) * IQ36_EXP2_SCALE);
      volatile half value_weight_half = convert_half_rte(weight);
      const float value_weight = convert_float(value_weight_half);
      output_accumulator = fma(
          value_value, value_weight, output_accumulator);
      softmax_sum += weight;
    }
    reference_max = running_max;
  }

  const uint output_index = OUTPUT0_OFFSET +
      batch * OUTPUT0_PITCHES[0] + query_head * OUTPUT0_PITCHES[1] +
      query_position * OUTPUT0_PITCHES[2] + dim * OUTPUT0_PITCHES[3];
  output[output_index] = (OUTPUT0_TYPE)(softmax_sum == 0.0f
      ? 0.0f : output_accumulator * native_recip(softmax_sum));

  if (query_head % IQ36_GQA_GROUP != 0U) return;

  if (query_position < cold_append_tokens) {
    const uint global_token = cold_tokens + query_position;
    float key_value;
    float value_value;
    if (global_token < past_tokens) {
      const uint slot = iq36_hot_slot(global_token);
      const uint hot_key_index = INPUT1_OFFSET +
          batch * INPUT1_PITCHES[0] + kv_head * INPUT1_PITCHES[1] +
          slot * INPUT1_PITCHES[2] + dim * INPUT1_PITCHES[3];
      const uint hot_value_index = INPUT2_OFFSET +
          batch * INPUT2_PITCHES[0] + kv_head * INPUT2_PITCHES[1] +
          slot * INPUT2_PITCHES[2] + dim * INPUT2_PITCHES[3];
      key_value = as_float((int)hot_key_bits[hot_key_index]);
      value_value = as_float((int)hot_value_bits[hot_value_index]);
    } else {
      const uint current_position = global_token - past_tokens;
      const uint current_key_index = INPUT3_OFFSET +
          batch * INPUT3_PITCHES[0] + kv_head * INPUT3_PITCHES[1] +
          current_position * INPUT3_PITCHES[2] + dim * INPUT3_PITCHES[3];
      const uint current_value_index = INPUT4_OFFSET +
          batch * INPUT4_PITCHES[0] + kv_head * INPUT4_PITCHES[1] +
          current_position * INPUT4_PITCHES[2] + dim * INPUT4_PITCHES[3];
      key_value = convert_float(current_key[current_key_index]);
      value_value = convert_float(current_value[current_value_index]);
    }
    const float key_max = sub_group_reduce_max(fabs(key_value));
    const float value_max = sub_group_reduce_max(fabs(value_value));
    const float key_scale = key_max == 0.0f ? 1.0f : key_max / 127.0f;
    const float value_scale = value_max == 0.0f ? 1.0f : value_max / 127.0f;
    const int key_quantized = clamp(
        (int)rint(key_value / key_scale), -127, 127);
    const int value_quantized = clamp(
        (int)rint(value_value / value_scale), -127, 127);
    const uint cold_key_index = OUTPUT1_OFFSET +
        batch * OUTPUT1_PITCHES[0] + kv_head * OUTPUT1_PITCHES[1] +
        query_position * OUTPUT1_PITCHES[2] + dim * OUTPUT1_PITCHES[3];
    const uint cold_value_index = OUTPUT2_OFFSET +
        batch * OUTPUT2_PITCHES[0] + kv_head * OUTPUT2_PITCHES[1] +
        query_position * OUTPUT2_PITCHES[2] + dim * OUTPUT2_PITCHES[3];
    cold_key_append[cold_key_index] = (OUTPUT1_TYPE)key_quantized;
    cold_value_append[cold_value_index] = (OUTPUT2_TYPE)value_quantized;
    if (lane < 2U) {
      const ushort key_bits = as_ushort(convert_half_rte(key_scale));
      const ushort value_bits = as_ushort(convert_half_rte(value_scale));
      const uint scale_x = (subgroup << 1) + lane;
      const uint key_scale_index = OUTPUT3_OFFSET +
          batch * OUTPUT3_PITCHES[0] + kv_head * OUTPUT3_PITCHES[1] +
          query_position * OUTPUT3_PITCHES[2] +
          scale_x * OUTPUT3_PITCHES[3];
      const uint value_scale_index = OUTPUT4_OFFSET +
          batch * OUTPUT4_PITCHES[0] + kv_head * OUTPUT4_PITCHES[1] +
          query_position * OUTPUT4_PITCHES[2] +
          scale_x * OUTPUT4_PITCHES[3];
      cold_key_scale_append[key_scale_index] = (OUTPUT3_TYPE)(
          lane == 0U ? key_bits & 0xffU : key_bits >> 8);
      cold_value_scale_append[value_scale_index] = (OUTPUT4_TYPE)(
          lane == 0U ? value_bits & 0xffU : value_bits >> 8);
    }
  }

  const uint global_current_token = past_tokens + query_position;
  if (global_current_token < IQ36_SINK_TOKENS ||
      global_current_token + IQ36_HOT_WINDOW >= key_tokens) {
    const uint current_key_index = INPUT3_OFFSET +
        batch * INPUT3_PITCHES[0] + kv_head * INPUT3_PITCHES[1] +
        query_position * INPUT3_PITCHES[2] + dim * INPUT3_PITCHES[3];
    const uint current_value_index = INPUT4_OFFSET +
        batch * INPUT4_PITCHES[0] + kv_head * INPUT4_PITCHES[1] +
        query_position * INPUT4_PITCHES[2] + dim * INPUT4_PITCHES[3];
    const uint slot = iq36_hot_slot(global_current_token);
    const uint hot_key_index = INPUT1_OFFSET +
        batch * INPUT1_PITCHES[0] + kv_head * INPUT1_PITCHES[1] +
        slot * INPUT1_PITCHES[2] + dim * INPUT1_PITCHES[3];
    const uint hot_value_index = INPUT2_OFFSET +
        batch * INPUT2_PITCHES[0] + kv_head * INPUT2_PITCHES[1] +
        slot * INPUT2_PITCHES[2] + dim * INPUT2_PITCHES[3];
    hot_key_bits[hot_key_index] = (INPUT1_TYPE)as_int(
        convert_float(current_key[current_key_index]));
    hot_value_bits[hot_value_index] = (INPUT2_TYPE)as_int(
        convert_float(current_value[current_value_index]));
  }

}
