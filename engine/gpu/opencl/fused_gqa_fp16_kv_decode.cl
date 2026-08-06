#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define IQ36_HEAD_DIM 256U
#define IQ36_Q_HEADS 16U
#define IQ36_KV_HEADS 2U
#define IQ36_GQA_GROUP 8U
#define IQ36_CHUNK_TOKENS 256U

__kernel void iq36_f32_to_f16(__global const float* input,
                              __global half* output,
                              uint count) {
  const uint index = (uint)get_global_id(0);
  if (index < count) output[index] = convert_half_rte(input[index]);
}

__attribute__((reqd_work_group_size(256, 1, 1)))
__attribute__((intel_reqd_sub_group_size(32)))
__kernel void iq36_fused_gqa_fp16_partial(
    __global const float* q,
    __global const half* k_history,
    __global const half* v_history,
    uint context_tokens,
    float attention_scale,
    __global float* partial_max,
    __global float* partial_sum,
    __global float* partial_output) {
  const uint lid = (uint)get_local_id(0);
  const uint subgroup = (uint)get_sub_group_id();
  const uint lane = (uint)get_sub_group_local_id();
  const uint chunk_count =
      (context_tokens + IQ36_CHUNK_TOKENS - 1U) / IQ36_CHUNK_TOKENS;
  const uint group = (uint)get_group_id(0);
  const uint kv_head = group / chunk_count;
  const uint chunk = group - kv_head * chunk_count;
  const uint q_head = kv_head * IQ36_GQA_GROUP + subgroup;
  const uint begin = chunk * IQ36_CHUNK_TOKENS;
  const uint end = min(begin + IQ36_CHUNK_TOKENS, context_tokens);
  __local float local_k[IQ36_HEAD_DIM];
  __local float local_v[IQ36_HEAD_DIM];

  float running_max = -INFINITY;
  float running_sum = 0.0f;
  float output_acc[8] = {
      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  for (uint token = begin; token < end; ++token) {
    const ulong history_base =
        (ulong)token * (IQ36_KV_HEADS * IQ36_HEAD_DIM) +
        (ulong)kv_head * IQ36_HEAD_DIM;
    local_k[lid] = convert_float(k_history[history_base + lid]);
    local_v[lid] = convert_float(v_history[history_base + lid]);
    barrier(CLK_LOCAL_MEM_FENCE);

    float dot = 0.0f;
    for (uint item = 0; item < 8U; ++item) {
      const uint dim = lane + item * 32U;
      dot = fma(q[q_head * IQ36_HEAD_DIM + dim], local_k[dim], dot);
    }
    const float score = sub_group_reduce_add(dot) * attention_scale;
    const float next_max = fmax(running_max, score);
    const float previous_scale = native_exp(running_max - next_max);
    const float value_scale = native_exp(score - next_max);
    running_sum = running_sum * previous_scale + value_scale;
    for (uint item = 0; item < 8U; ++item) {
      const uint dim = lane + item * 32U;
      output_acc[item] = fma(
          local_v[dim], value_scale, output_acc[item] * previous_scale);
    }
    running_max = next_max;
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  const uint meta = group * IQ36_GQA_GROUP + subgroup;
  if (lane == 0U) {
    partial_max[meta] = running_max;
    partial_sum[meta] = running_sum;
  }
  for (uint item = 0; item < 8U; ++item) {
    const uint dim = lane + item * 32U;
    partial_output[(ulong)meta * IQ36_HEAD_DIM + dim] = output_acc[item];
  }
}

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void iq36_fused_gqa_partial_reduce(
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
