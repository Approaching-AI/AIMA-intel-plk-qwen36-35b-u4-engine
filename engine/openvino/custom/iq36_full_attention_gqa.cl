#pragma OPENCL EXTENSION cl_khr_subgroups : enable

// Fixed Qwen3.6 full-attention arithmetic boundary:
//   input0 Q    [B,16,Q,256]
//   input1 K    [B, 2,K,256]
//   input2 V    [B, 2,K,256]
//   input3 mask [B, 1,Q,K]
//   output0     [B,16,Q,256]
//
// One 256-work-item group owns one query head and query position.  The eight
// subgroups cooperatively reduce the 256-wide Q.K dot and every lane retains
// one output dimension through a two-pass softmax.  This is the correctness
// carrier; long-context cold-state dequantization is added only after the real
// F32 layer boundary passes.

#define IQ36_HEAD_DIM 256U
#define IQ36_Q_HEADS 16U
#define IQ36_KV_HEADS 2U
#define IQ36_GQA_GROUP 8U
#define IQ36_ATTENTION_SCALE 0.0625f
#define IQ36_LOG2_E 1.442695f
#define IQ36_EXP2_SCALE (IQ36_ATTENTION_SCALE * IQ36_LOG2_E)

__attribute__((reqd_work_group_size(256, 1, 1)))
__attribute__((intel_reqd_sub_group_size(32)))
__kernel void iq36_full_attention_gqa(
    const __global INPUT0_TYPE* query,
    const __global INPUT1_TYPE* key,
    const __global INPUT2_TYPE* value,
    const __global INPUT3_TYPE* mask,
    __global OUTPUT0_TYPE* output) {
  const uint dim = (uint)get_local_id(0);
  const uint query_position = (uint)get_group_id(1);
  const uint batch_head = (uint)get_group_id(2);
  const uint batch = batch_head / IQ36_Q_HEADS;
  const uint query_head = batch_head - batch * IQ36_Q_HEADS;
  const uint kv_head = query_head / IQ36_GQA_GROUP;
  const uint subgroup = (uint)get_sub_group_id();
  const uint lane = (uint)get_sub_group_local_id();
  const uint query_tokens = (uint)INPUT0_DIMS[2];
  const uint key_tokens = (uint)INPUT1_DIMS[2];
  const uint softmax_chunk = query_tokens == 1U ? 256U : 128U;

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
    const uint chunk_end = min(
        chunk_begin + softmax_chunk, key_tokens);
    float chunk_max = -INFINITY;
    for (uint token = chunk_begin; token < chunk_end; ++token) {
      const uint key_index = INPUT1_OFFSET +
          batch * INPUT1_PITCHES[0] + kv_head * INPUT1_PITCHES[1] +
          token * INPUT1_PITCHES[2] + dim * INPUT1_PITCHES[3];
      const float partial = query_value * convert_float(key[key_index]);
      const float subgroup_sum = sub_group_reduce_add(partial);
      if (lane == 0U) subgroup_sums[subgroup] = subgroup_sum;
      barrier(CLK_LOCAL_MEM_FENCE);

      if (subgroup == 0U) {
        const float value_to_sum = lane < 8U ? subgroup_sums[lane] : 0.0f;
        const float dot = sub_group_reduce_add(value_to_sum);
        if (lane == 0U) {
          const uint mask_index = INPUT3_OFFSET +
              batch * INPUT3_PITCHES[0] +
              query_position * INPUT3_PITCHES[2] +
              token * INPUT3_PITCHES[3];
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
      const uint key_index = INPUT1_OFFSET +
          batch * INPUT1_PITCHES[0] + kv_head * INPUT1_PITCHES[1] +
          token * INPUT1_PITCHES[2] + dim * INPUT1_PITCHES[3];
      const float partial = query_value * convert_float(key[key_index]);
      const float subgroup_sum = sub_group_reduce_add(partial);
      if (lane == 0U) subgroup_sums[subgroup] = subgroup_sum;
      barrier(CLK_LOCAL_MEM_FENCE);

      if (subgroup == 0U) {
        const float value_to_sum = lane < 8U ? subgroup_sums[lane] : 0.0f;
        const float dot = sub_group_reduce_add(value_to_sum);
        if (lane == 0U) {
          const uint mask_index = INPUT3_OFFSET +
              batch * INPUT3_PITCHES[0] +
              query_position * INPUT3_PITCHES[2] +
              token * INPUT3_PITCHES[3];
          shared_score = dot +
              convert_float(mask[mask_index]) / IQ36_ATTENTION_SCALE;
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);

      const float weight = native_exp2(
          (shared_score - running_max) * IQ36_EXP2_SCALE);
      volatile half value_weight_half = convert_half_rte(weight);
      const float value_weight = convert_float(value_weight_half);
      const uint value_index = INPUT2_OFFSET +
          batch * INPUT2_PITCHES[0] + kv_head * INPUT2_PITCHES[1] +
          token * INPUT2_PITCHES[2] + dim * INPUT2_PITCHES[3];
      output_accumulator = fma(
          convert_float(value[value_index]), value_weight,
          output_accumulator);
      softmax_sum += weight;
    }
    reference_max = running_max;
  }

  const uint output_index = OUTPUT0_OFFSET +
      batch * OUTPUT0_PITCHES[0] + query_head * OUTPUT0_PITCHES[1] +
      query_position * OUTPUT0_PITCHES[2] + dim * OUTPUT0_PITCHES[3];
  output[output_index] = (OUTPUT0_TYPE)(softmax_sum == 0.0f
      ? 0.0f : output_accumulator * native_recip(softmax_sum));
}
