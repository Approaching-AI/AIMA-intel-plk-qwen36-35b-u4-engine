#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#pragma OPENCL EXTENSION cl_khr_subgroup_non_uniform_arithmetic : enable
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable

#define IQ36_HEAD_DIM 256U
#define IQ36_Q_HEADS 16U
#define IQ36_KV_HEADS 2U
#define IQ36_GQA_GROUP 8U
#define IQ36_TOKEN_TILE 16U
#define IQ36_CHUNK_TOKENS 256U

__kernel void iq36_f32_to_f16(__global const float* input,
                              __global half* output,
                              uint count) {
  const uint index = (uint)get_global_id(0);
  if (index < count) output[index] = convert_half_rte(input[index]);
}

__kernel void iq36_pack_k_dpas16(
    __global const float* input,
    uint context_tokens,
    __global half* output) {
  const uint index = (uint)get_global_id(0);
  const uint total = context_tokens * IQ36_KV_HEADS * IQ36_HEAD_DIM;
  if (index >= total) return;
  uint value = index;
  const uint token_lane = value % IQ36_TOKEN_TILE;
  value /= IQ36_TOKEN_TILE;
  const uint dim = value % IQ36_HEAD_DIM;
  value /= IQ36_HEAD_DIM;
  const uint token_block = value % (context_tokens / IQ36_TOKEN_TILE);
  const uint kv_head = value / (context_tokens / IQ36_TOKEN_TILE);
  const uint token = token_block * IQ36_TOKEN_TILE + token_lane;
  const ulong input_index =
      (ulong)token * (IQ36_KV_HEADS * IQ36_HEAD_DIM) +
      (ulong)kv_head * IQ36_HEAD_DIM + dim;
  output[index] = convert_half_rte(input[input_index]);
}

__attribute__((reqd_work_group_size(256, 1, 1)))
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void iq36_xmx_gqa_fp16_partial(
    __global const half* q,
    __global const half* k_dpas16,
    __global const half* v_history,
    uint context_tokens,
    __global float* partial_max,
    __global float* partial_sum,
    __global float* partial_output) {
  const uint subgroup = (uint)get_sub_group_id();
  const uint lane = (uint)get_sub_group_local_id();
  const uint chunk_count =
      (context_tokens + IQ36_CHUNK_TOKENS - 1U) / IQ36_CHUNK_TOKENS;
  const uint token_block_count = context_tokens / IQ36_TOKEN_TILE;
  const uint group = (uint)get_group_id(0);
  const uint kv_head = group / chunk_count;
  const uint chunk = group - kv_head * chunk_count;
  const uint first_block = chunk * (IQ36_CHUNK_TOKENS / IQ36_TOKEN_TILE);
  __local half local_weights[IQ36_GQA_GROUP * IQ36_TOKEN_TILE];
  __local float local_alpha[IQ36_GQA_GROUP];

  float8 output_acc = (float8)(0.0f);
  float8 running_max = (float8)(-INFINITY);
  float8 running_sum = (float8)(0.0f);
  for (uint tile = 0; tile < IQ36_CHUNK_TOKENS / IQ36_TOKEN_TILE; ++tile) {
    const uint token_block = first_block + tile;
    if (subgroup == 0U) {
      float8 score = (float8)(0.0f);
      #pragma unroll
      for (uint k_block = 0; k_block < IQ36_HEAD_DIM / 16U; ++k_block) {
        const uint k_base = k_block * 16U;
        const uint q_base = kv_head * IQ36_GQA_GROUP * IQ36_HEAD_DIM;
        const short8 q_fragment = (short8)(
            as_short(q[q_base + 0U * IQ36_HEAD_DIM + k_base + lane]),
            as_short(q[q_base + 1U * IQ36_HEAD_DIM + k_base + lane]),
            as_short(q[q_base + 2U * IQ36_HEAD_DIM + k_base + lane]),
            as_short(q[q_base + 3U * IQ36_HEAD_DIM + k_base + lane]),
            as_short(q[q_base + 4U * IQ36_HEAD_DIM + k_base + lane]),
            as_short(q[q_base + 5U * IQ36_HEAD_DIM + k_base + lane]),
            as_short(q[q_base + 6U * IQ36_HEAD_DIM + k_base + lane]),
            as_short(q[q_base + 7U * IQ36_HEAD_DIM + k_base + lane]));
        int8 k_fragment;
        #pragma unroll
        for (uint pair = 0; pair < 8U; ++pair) {
          const uint k0 = k_base + pair * 2U;
          const ulong base =
              (((ulong)kv_head * token_block_count + token_block) *
                   IQ36_HEAD_DIM + k0) * IQ36_TOKEN_TILE + lane;
          const half2 packed = (half2)(
              k_dpas16[base], k_dpas16[base + IQ36_TOKEN_TILE]);
          k_fragment[pair] = as_int(packed);
        }
        score = intel_sub_group_f16_f16_matrix_mad_k16(
            q_fragment, k_fragment, score);
      }
      score *= (float8)(0.0625f);
      float8 block_max;
      #pragma unroll
      for (uint head = 0; head < IQ36_GQA_GROUP; ++head) {
        block_max[head] = sub_group_reduce_max(score[head]);
      }
      const float8 next_max = fmax(running_max, block_max);
      const float8 previous_scale = native_exp(running_max - next_max);
      float8 block_sum;
      #pragma unroll
      for (uint head = 0; head < IQ36_GQA_GROUP; ++head) {
        const float weight = native_exp(score[head] - next_max[head]);
        local_weights[head * IQ36_TOKEN_TILE + lane] =
            convert_half_rte(weight);
        block_sum[head] = sub_group_reduce_add(weight);
      }
      running_sum = running_sum * previous_scale + block_sum;
      running_max = next_max;
      if (lane < IQ36_GQA_GROUP) local_alpha[lane] = previous_scale[lane];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const short8 weight_fragment = (short8)(
        as_short(local_weights[0U * IQ36_TOKEN_TILE + lane]),
        as_short(local_weights[1U * IQ36_TOKEN_TILE + lane]),
        as_short(local_weights[2U * IQ36_TOKEN_TILE + lane]),
        as_short(local_weights[3U * IQ36_TOKEN_TILE + lane]),
        as_short(local_weights[4U * IQ36_TOKEN_TILE + lane]),
        as_short(local_weights[5U * IQ36_TOKEN_TILE + lane]),
        as_short(local_weights[6U * IQ36_TOKEN_TILE + lane]),
        as_short(local_weights[7U * IQ36_TOKEN_TILE + lane]));
    int8 v_fragment;
    #pragma unroll
    for (uint pair = 0; pair < 8U; ++pair) {
      const uint token0 = token_block * IQ36_TOKEN_TILE + pair * 2U;
      const uint dim = subgroup * 16U + lane;
      const ulong v0 =
          (ulong)token0 * (IQ36_KV_HEADS * IQ36_HEAD_DIM) +
          (ulong)kv_head * IQ36_HEAD_DIM + dim;
      const ulong v1 = v0 + IQ36_KV_HEADS * IQ36_HEAD_DIM;
      v_fragment[pair] = as_int((half2)(v_history[v0], v_history[v1]));
    }
    const float8 block_output = intel_sub_group_f16_f16_matrix_mad_k16(
        weight_fragment, v_fragment, (float8)(0.0f));
    const float8 alpha = (float8)(
        local_alpha[0], local_alpha[1], local_alpha[2], local_alpha[3],
        local_alpha[4], local_alpha[5], local_alpha[6], local_alpha[7]);
    output_acc = output_acc * alpha + block_output;
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (subgroup == 0U && lane < IQ36_GQA_GROUP) {
    const uint meta = group * IQ36_GQA_GROUP + lane;
    partial_max[meta] = running_max[lane];
    partial_sum[meta] = running_sum[lane];
  }
  const uint dim = subgroup * 16U + lane;
  #pragma unroll
  for (uint head = 0; head < IQ36_GQA_GROUP; ++head) {
    const uint meta = group * IQ36_GQA_GROUP + head;
    partial_output[(ulong)meta * IQ36_HEAD_DIM + dim] = output_acc[head];
  }
}

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void iq36_xmx_gqa_partial_reduce(
    __global const float* partial_max,
    __global const float* partial_sum,
    __global const float* partial_output,
    __global const float* gate,
    uint context_tokens,
    __global float* output) {
  const uint q_head = (uint)get_group_id(0);
  const uint dim = (uint)get_local_id(0);
  const uint kv_head = q_head / IQ36_GQA_GROUP;
  const uint gqa_head = q_head - kv_head * IQ36_GQA_GROUP;
  const uint chunk_count =
      (context_tokens + IQ36_CHUNK_TOKENS - 1U) / IQ36_CHUNK_TOKENS;
  float running_max = -INFINITY;
  float running_sum = 0.0f;
  float output_acc = 0.0f;
  for (uint chunk = 0; chunk < chunk_count; ++chunk) {
    const uint group = kv_head * chunk_count + chunk;
    const uint meta = group * IQ36_GQA_GROUP + gqa_head;
    const float next_max = fmax(running_max, partial_max[meta]);
    const float previous_scale = native_exp(running_max - next_max);
    const float partial_scale = native_exp(partial_max[meta] - next_max);
    running_sum = running_sum * previous_scale +
        partial_sum[meta] * partial_scale;
    output_acc = output_acc * previous_scale +
        partial_output[(ulong)meta * IQ36_HEAD_DIM + dim] * partial_scale;
    running_max = next_max;
  }
  const uint index = q_head * IQ36_HEAD_DIM + dim;
  const float pregate = running_sum == 0.0f ? 0.0f : output_acc / running_sum;
  output[index] = pregate / (1.0f + native_exp(-gate[index]));
}

__kernel void iq36_reference_score_f32(
    __global const float* q,
    __global const float* k_history,
    uint context_tokens,
    float attention_scale,
    __global float* scores) {
  const uint index = (uint)get_global_id(0);
  const uint total = IQ36_Q_HEADS * context_tokens;
  if (index >= total) return;
  const uint q_head = index / context_tokens;
  const uint token = index - q_head * context_tokens;
  const uint kv_head = q_head / IQ36_GQA_GROUP;
  const ulong history_base =
      (ulong)token * (IQ36_KV_HEADS * IQ36_HEAD_DIM) +
      (ulong)kv_head * IQ36_HEAD_DIM;
  float dot = 0.0f;
  for (uint dim = 0; dim < IQ36_HEAD_DIM; ++dim) {
    dot = fma(q[q_head * IQ36_HEAD_DIM + dim],
              k_history[history_base + dim], dot);
  }
  scores[index] = dot * attention_scale;
}

__kernel void iq36_reference_apply_f32(
    __global const float* scores,
    __global const float* v_history,
    __global const float* gate,
    uint context_tokens,
    __global float* output) {
  const uint index = (uint)get_global_id(0);
  if (index >= IQ36_Q_HEADS * IQ36_HEAD_DIM) return;
  const uint q_head = index / IQ36_HEAD_DIM;
  const uint dim = index - q_head * IQ36_HEAD_DIM;
  const uint kv_head = q_head / IQ36_GQA_GROUP;
  float running_max = -INFINITY;
  float running_sum = 0.0f;
  float output_acc = 0.0f;
  for (uint token = 0; token < context_tokens; ++token) {
    const float score = scores[(ulong)q_head * context_tokens + token];
    const float next_max = fmax(running_max, score);
    const float previous_scale = exp(running_max - next_max);
    const float value_scale = exp(score - next_max);
    const ulong v_index =
        (ulong)token * (IQ36_KV_HEADS * IQ36_HEAD_DIM) +
        (ulong)kv_head * IQ36_HEAD_DIM + dim;
    output_acc = fma(
        v_history[v_index], value_scale, output_acc * previous_scale);
    running_sum = running_sum * previous_scale + value_scale;
    running_max = next_max;
  }
  const float pregate = running_sum == 0.0f ? 0.0f : output_acc / running_sum;
  output[index] = pregate / (1.0f + exp(-gate[index]));
}
