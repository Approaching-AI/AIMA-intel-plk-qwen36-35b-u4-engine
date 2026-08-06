#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define IQ36_HEAD_DIM 256U
#define IQ36_Q_HEADS 16U
#define IQ36_KV_HEADS 2U
#define IQ36_GQA_GROUP 8U
#define IQ36_QUANT_GROUP 32U
#define IQ36_SCALE_GROUPS 8U
#define IQ36_CHUNK_TOKENS 256U

__constant ushort IQ36_E4M3_TO_HALF_BITS[256] = {
    0x0000U, 0x1800U, 0x1c00U, 0x1e00U, 0x2000U, 0x2100U, 0x2200U, 0x2300U,
    0x2400U, 0x2480U, 0x2500U, 0x2580U, 0x2600U, 0x2680U, 0x2700U, 0x2780U,
    0x2800U, 0x2880U, 0x2900U, 0x2980U, 0x2a00U, 0x2a80U, 0x2b00U, 0x2b80U,
    0x2c00U, 0x2c80U, 0x2d00U, 0x2d80U, 0x2e00U, 0x2e80U, 0x2f00U, 0x2f80U,
    0x3000U, 0x3080U, 0x3100U, 0x3180U, 0x3200U, 0x3280U, 0x3300U, 0x3380U,
    0x3400U, 0x3480U, 0x3500U, 0x3580U, 0x3600U, 0x3680U, 0x3700U, 0x3780U,
    0x3800U, 0x3880U, 0x3900U, 0x3980U, 0x3a00U, 0x3a80U, 0x3b00U, 0x3b80U,
    0x3c00U, 0x3c80U, 0x3d00U, 0x3d80U, 0x3e00U, 0x3e80U, 0x3f00U, 0x3f80U,
    0x4000U, 0x4080U, 0x4100U, 0x4180U, 0x4200U, 0x4280U, 0x4300U, 0x4380U,
    0x4400U, 0x4480U, 0x4500U, 0x4580U, 0x4600U, 0x4680U, 0x4700U, 0x4780U,
    0x4800U, 0x4880U, 0x4900U, 0x4980U, 0x4a00U, 0x4a80U, 0x4b00U, 0x4b80U,
    0x4c00U, 0x4c80U, 0x4d00U, 0x4d80U, 0x4e00U, 0x4e80U, 0x4f00U, 0x4f80U,
    0x5000U, 0x5080U, 0x5100U, 0x5180U, 0x5200U, 0x5280U, 0x5300U, 0x5380U,
    0x5400U, 0x5480U, 0x5500U, 0x5580U, 0x5600U, 0x5680U, 0x5700U, 0x5780U,
    0x5800U, 0x5880U, 0x5900U, 0x5980U, 0x5a00U, 0x5a80U, 0x5b00U, 0x5b80U,
    0x5c00U, 0x5c80U, 0x5d00U, 0x5d80U, 0x5e00U, 0x5e80U, 0x5f00U, 0x5f00U,
    0x8000U, 0x9800U, 0x9c00U, 0x9e00U, 0xa000U, 0xa100U, 0xa200U, 0xa300U,
    0xa400U, 0xa480U, 0xa500U, 0xa580U, 0xa600U, 0xa680U, 0xa700U, 0xa780U,
    0xa800U, 0xa880U, 0xa900U, 0xa980U, 0xaa00U, 0xaa80U, 0xab00U, 0xab80U,
    0xac00U, 0xac80U, 0xad00U, 0xad80U, 0xae00U, 0xae80U, 0xaf00U, 0xaf80U,
    0xb000U, 0xb080U, 0xb100U, 0xb180U, 0xb200U, 0xb280U, 0xb300U, 0xb380U,
    0xb400U, 0xb480U, 0xb500U, 0xb580U, 0xb600U, 0xb680U, 0xb700U, 0xb780U,
    0xb800U, 0xb880U, 0xb900U, 0xb980U, 0xba00U, 0xba80U, 0xbb00U, 0xbb80U,
    0xbc00U, 0xbc80U, 0xbd00U, 0xbd80U, 0xbe00U, 0xbe80U, 0xbf00U, 0xbf80U,
    0xc000U, 0xc080U, 0xc100U, 0xc180U, 0xc200U, 0xc280U, 0xc300U, 0xc380U,
    0xc400U, 0xc480U, 0xc500U, 0xc580U, 0xc600U, 0xc680U, 0xc700U, 0xc780U,
    0xc800U, 0xc880U, 0xc900U, 0xc980U, 0xca00U, 0xca80U, 0xcb00U, 0xcb80U,
    0xcc00U, 0xcc80U, 0xcd00U, 0xcd80U, 0xce00U, 0xce80U, 0xcf00U, 0xcf80U,
    0xd000U, 0xd080U, 0xd100U, 0xd180U, 0xd200U, 0xd280U, 0xd300U, 0xd380U,
    0xd400U, 0xd480U, 0xd500U, 0xd580U, 0xd600U, 0xd680U, 0xd700U, 0xd780U,
    0xd800U, 0xd880U, 0xd900U, 0xd980U, 0xda00U, 0xda80U, 0xdb00U, 0xdb80U,
    0xdc00U, 0xdc80U, 0xdd00U, 0xdd80U, 0xde00U, 0xde80U, 0xdf00U, 0xdf00U,
};

uchar iq36_encode_e4m3(float value) {
  const uint sign = value < 0.0f ? 128U : 0U;
  const float magnitude = fabs(value);
  uint code = 0U;
  if (magnitude < 0.015625f) {
    code = min(convert_uint_rte(magnitude * 512.0f), 7U);
  } else {
    const ushort half_bits = as_ushort(convert_half_rte(magnitude));
    uint exponent = ((uint)half_bits >> 10U) - 8U;
    const uint half_mantissa = (uint)half_bits & 1023U;
    uint mantissa = half_mantissa >> 7U;
    const uint discarded = half_mantissa & 127U;
    if (discarded > 64U || (discarded == 64U && (mantissa & 1U) != 0U)) {
      ++mantissa;
      if (mantissa == 8U) {
        mantissa = 0U;
        ++exponent;
      }
    }
    code = min((exponent << 3U) | mantissa, 119U);
  }
  return (uchar)(sign | code);
}

__attribute__((reqd_work_group_size(32, 1, 1)))
__attribute__((intel_reqd_sub_group_size(32)))
__kernel void iq36_encode_current_scaled_e4m3_block32(
    __global const float* current_k,
    __global const float* current_v,
    __global uchar* k_history,
    __global uchar* v_history,
    __global half* k_scales,
    __global half* v_scales,
    uint context_tokens) {
  const uint lane = (uint)get_sub_group_local_id();
  const uint group = (uint)get_group_id(0);
  const uint tensor = group / (IQ36_KV_HEADS * IQ36_SCALE_GROUPS);
  const uint within_tensor =
      group - tensor * IQ36_KV_HEADS * IQ36_SCALE_GROUPS;
  const uint kv_head = within_tensor / IQ36_SCALE_GROUPS;
  const uint scale_group = within_tensor - kv_head * IQ36_SCALE_GROUPS;
  const uint dim = scale_group * IQ36_QUANT_GROUP + lane;
  __global const float* input = tensor == 0U ? current_k : current_v;
  __global uchar* output = tensor == 0U ? k_history : v_history;
  __global half* scales = tensor == 0U ? k_scales : v_scales;
  const float value = input[kv_head * IQ36_HEAD_DIM + dim];
  const float maximum = sub_group_reduce_max(fabs(value));
  const float scale = maximum == 0.0f ? 1.0f : maximum;
  const ulong token_head =
      (ulong)(context_tokens - 1U) * IQ36_KV_HEADS + kv_head;
  output[token_head * IQ36_HEAD_DIM + dim] = iq36_encode_e4m3(value / scale);
  if (lane == 0U) {
    scales[token_head * IQ36_SCALE_GROUPS + scale_group] =
        convert_half_rte(scale);
  }
}

__attribute__((reqd_work_group_size(256, 1, 1)))
__attribute__((intel_reqd_sub_group_size(32)))
__kernel void iq36_scaled_e4m3_gqa_partial(
    __global const float* q,
    __global const uchar* k_history,
    __global const uchar* v_history,
    __global const half* k_scales,
    __global const half* v_scales,
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
    const ulong token_head = (ulong)token * IQ36_KV_HEADS + kv_head;
    const ulong history_base = token_head * IQ36_HEAD_DIM;
    const ulong scale_base = token_head * IQ36_SCALE_GROUPS;
    float k_scale = lane == 0U
        ? convert_float(k_scales[scale_base + subgroup]) : 0.0f;
    float v_scale = lane == 0U
        ? convert_float(v_scales[scale_base + subgroup]) : 0.0f;
    k_scale = sub_group_broadcast(k_scale, 0U);
    v_scale = sub_group_broadcast(v_scale, 0U);
    const ushort k_bits = IQ36_E4M3_TO_HALF_BITS[
        (uint)k_history[history_base + lid]];
    const ushort v_bits = IQ36_E4M3_TO_HALF_BITS[
        (uint)v_history[history_base + lid]];
    local_k[lid] = convert_float(as_half(k_bits)) * k_scale;
    local_v[lid] = convert_float(as_half(v_bits)) * v_scale;
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
__kernel void iq36_scaled_e4m3_gqa_partial_reduce(
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
