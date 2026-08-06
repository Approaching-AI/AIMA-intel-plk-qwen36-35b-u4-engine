#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL FP_CONTRACT OFF

inline float iq36_cr_recip_normal_f32_u64_v1(float input) {
  const uint input_bits = as_uint(input);
  const uint exponent = (input_bits >> 23U) & 0xffU;
  const uint fraction = input_bits & 0x7fffffU;
  if ((input_bits >> 31U) != 0U || exponent == 0U || exponent == 0xffU) {
    return as_float(0x7fc00000U);
  }

  const ulong significand = 0x800000UL | (ulong)fraction;
  const int input_unbiased = (int)exponent - 127;
  ulong quotient;
  int output_unbiased;
  if (significand == 0x800000UL) {
    quotient = 0x800000UL;
    output_unbiased = -input_unbiased;
  } else {
    const ulong numerator = 1UL << 47U;
    quotient = numerator / significand;
    const ulong remainder = numerator % significand;
    const ulong twice_remainder = remainder << 1U;
    if (twice_remainder > significand ||
        (twice_remainder == significand && (quotient & 1UL) != 0UL)) {
      ++quotient;
    }
    output_unbiased = -input_unbiased - 1;
    if (quotient == 0x1000000UL) {
      quotient >>= 1U;
      ++output_unbiased;
    }
  }

  const int output_exponent = output_unbiased + 127;
  if (output_exponent <= 0 || output_exponent >= 255) {
    return as_float(0x7fc00000U);
  }
  const uint output_bits = ((uint)output_exponent << 23U) |
                           ((uint)quotient & 0x7fffffU);
  return as_float(output_bits);
}

__attribute__((reqd_work_group_size(128, 1, 1)))
__kernel void iq36_l0_postconv_cpuorder(
    __global const float* conv_output_raw,
    __global float* q_output,
    __global float* k_output,
    __global float* v_output,
    float norm_epsilon) {
  const uint group = (uint)get_group_id(0);
  const uint lane = (uint)get_local_id(0);
  if (group >= 64U) {
    return;
  }

  uint source_base;
  if (group < 16U) {
    source_base = group * 128U;
  } else if (group < 32U) {
    source_base = 2048U + (group - 16U) * 128U;
  } else {
    source_base = 4096U + (group - 32U) * 128U;
  }
  const float value = conv_output_raw[source_base + lane];
  const double x = (double)value;
  double sigmoid_double;
  if (x >= 0.0) {
    const double exp_value = exp(-x);
    sigmoid_double = 1.0 / (1.0 + exp_value);
  } else {
    const double exp_value = exp(x);
    sigmoid_double = exp_value / (1.0 + exp_value);
  }
  const float sigmoid = (float)sigmoid_double;
  const float silu = value * sigmoid;
  if (group >= 32U) {
    v_output[(group - 32U) * 128U + lane] = silu;
    return;
  }

  __local float head_values[128];
  __local float head_scale[1];
  head_values[lane] = silu;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lane == 0U) {
    double sum = 0.0;
    for (uint i = 0U; i < 128U; ++i) {
      const float head_value = head_values[i];
      sum = sum + (double)head_value * (double)head_value;
    }
    const float sum_f32 = (float)sum;
    head_scale[0] = iq36_cr_recip_normal_f32_u64_v1(
        fmax(sqrt(sum_f32), norm_epsilon));
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  const float normalized = silu * head_scale[0];
  if (group < 16U) {
    q_output[group * 128U + lane] = normalized;
  } else {
    k_output[(group - 16U) * 128U + lane] = normalized;
  }
}

__attribute__((reqd_work_group_size(128, 1, 1)))
__kernel void iq36_l0_delta_recurrent_cpuorder(
    __global const float* q,
    __global const float* k,
    __global const float* v,
    __global const float* decay,
    __global const float* beta,
    __global const float* state_in,
    __global const float* z_silu,
    __global const float* norm_weight,
    __global float* attention_output,
    __global float* state_out,
    __global float* final_output,
    float norm_epsilon,
    float attention_scale) {
  const uint value_head = (uint)get_group_id(0);
  const uint row = (uint)get_local_id(0);
  if (value_head >= 32U) {
    return;
  }
  const uint query_head = value_head % 16U;
  const uint q_base = query_head * 128U;
  const uint value_base = value_head * 128U;
  const uint state_base = value_head * 16384U + row * 128U;

  float sum_k = 0.0f;
  for (uint col = 0U; col < 128U; ++col) {
    const float decayed = state_in[state_base + col] * decay[value_head];
    state_out[state_base + col] = decayed;
    const float product = decayed * k[q_base + col];
    sum_k = sum_k + product;
  }
  const float difference = v[value_base + row] - sum_k;
  const float delta = difference * beta[value_head];

  for (uint col = 0U; col < 128U; ++col) {
    const float update = k[q_base + col] * delta;
    const float updated = state_out[state_base + col] + update;
    state_out[state_base + col] = updated;
  }

  float sum_q = 0.0f;
  for (uint col = 0U; col < 128U; ++col) {
    const float product = state_out[state_base + col] * q[q_base + col];
    sum_q = sum_q + product;
  }
  const float attention = sum_q * attention_scale;
  attention_output[value_base + row] = attention;
  __local float head_attention[128];
  __local float head_norm_scale[1];
  head_attention[row] = attention;
  barrier(CLK_LOCAL_MEM_FENCE);

  if (row == 0U) {
    float sum_squares = 0.0f;
    for (uint i = 0U; i < 128U; ++i) {
      const float square = head_attention[i] * head_attention[i];
      sum_squares = sum_squares + square;
    }
    const float mean_square = sum_squares / 128.0f;
    head_norm_scale[0] = 1.0f / sqrt(mean_square + norm_epsilon);
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  const float normalized = attention * head_norm_scale[0];
  const float weighted = normalized * norm_weight[row];
  final_output[value_base + row] = weighted * z_silu[value_base + row];
}

#pragma OPENCL FP_CONTRACT ON
