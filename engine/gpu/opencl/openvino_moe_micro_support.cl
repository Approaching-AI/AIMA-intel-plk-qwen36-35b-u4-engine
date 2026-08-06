#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define IQ36_TOKENS 1024U
#define IQ36_ROUTED_ROWS 8192U
#define IQ36_COMPLETE_ROWS 9216U
#define IQ36_HIDDEN 2048U
#define IQ36_INTERMEDIATE 512U

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void iq36_micro_gather_f16_sums32(
    __global const float *input,
    __global const int *token_map,
    __global half *grouped_input,
    __global float *sums32) {
  const uint lane = get_local_id(0);
  const uint task = get_group_id(0);
  const uint row = task >> 3U;
  const uint block = task & 7U;
  if (row >= IQ36_COMPLETE_ROWS) return;
  const uint token = (uint)token_map[row];
  const uint inner = block * 256U + lane;
  const half stored = convert_half_rte(
      input[token * IQ36_HIDDEN + inner]);
  grouped_input[row * IQ36_HIDDEN + inner] = stored;
  __local float values[256];
  values[lane] = convert_float(stored);
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lane < 8U) {
    float sum = 0.0f;
    for (uint index = 0U; index < 32U; ++index) {
      sum += values[lane * 32U + index];
    }
    sums32[row * 64U + block * 8U + lane] = sum;
  }
}

float iq36_micro_expf(float value) {
  const float r = 0x1.8p23f;
  const float z = fma(value, 0x1.715476p+0f, r);
  const float n = z - r;
  const float b = fma(
      -n, 0x1.7f7d1cp-20f,
      fma(-n, 0x1.62e4p-1f, value));
  const uint e = as_uint(z) << 23;
  const float k = as_float(e + as_uint(1.0f));
  const float u = b * b;
  const float j = fma(
      fma(fma(0x1.0e4020p-7f, b, 0x1.573e2ep-5f), u,
          fma(0x1.555e66p-3f, b, 0x1.fffdb6p-2f)),
      u, 0x1.ffffecp-1f * b);
  if (fabs(n) <= 126.0f) return fma(k, j, k);
  const uint g = (n <= 0.0f ? 0xffffffffU : 0U) & 0x82000000U;
  const float s1 = as_float(g + 0x7f000000U);
  if (fabs(n) > 192.0f) return s1 * s1;
  const float s2 = as_float(e - g);
  return fma(s2, j, s2) * s1;
}

float iq36_micro_silu(float value) {
  const float sigmoid = value >= 0.0f
      ? 1.0f / (1.0f + exp(-value))
      : exp(value) / (1.0f + exp(value));
  return value * sigmoid;
}

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void iq36_micro_q4k_residual_swiglu_sums32(
    __global const half *gate_main,
    __global const half *up_main,
    __global const float *input_sums32,
    __global const float *gate_mins,
    __global const float *up_mins,
    __global const int *row_expert,
    __global half *swiglu_output,
    __global float *down_sums32) {
  const uint lane = get_local_id(0);
  const uint task = get_group_id(0);
  const uint row = task >> 1U;
  const uint block = task & 1U;
  if (row >= IQ36_COMPLETE_ROWS) return;
  const uint output = block * 256U + lane;
  const uint expert = (uint)row_expert[row];
  float gate = convert_float(gate_main[row * IQ36_INTERMEDIATE + output]);
  float up = convert_float(up_main[row * IQ36_INTERMEDIATE + output]);
  for (uint group = 0U; group < 64U; ++group) {
    const float sum = input_sums32[row * 64U + group];
    const ulong coefficient =
        ((ulong)expert * 64UL + group) * IQ36_INTERMEDIATE + output;
    gate -= sum * gate_mins[coefficient];
    up -= sum * up_mins[coefficient];
  }
  const half stored = convert_half_rte(iq36_micro_silu(gate) * up);
  swiglu_output[row * IQ36_INTERMEDIATE + output] = stored;
  __local float values[256];
  values[lane] = convert_float(stored);
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lane < 8U) {
    float sum = 0.0f;
    for (uint index = 0U; index < 32U; ++index) {
      sum += values[lane * 32U + index];
    }
    down_sums32[row * 16U + block * 8U + lane] = sum;
  }
}

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void iq36_micro_q4k_down_residual_weight(
    __global const half *down_main,
    __global const float *down_sums32,
    __global const float *down_mins,
    __global const int *row_expert,
    __global const float *router_weights,
    __global float *contributions) {
  const uint lane = get_local_id(0);
  const uint task = get_group_id(0);
  const uint row = task >> 3U;
  const uint block = task & 7U;
  if (row >= IQ36_COMPLETE_ROWS) return;
  const uint output = block * 256U + lane;
  const uint expert = (uint)row_expert[row];
  float value = convert_float(down_main[row * IQ36_HIDDEN + output]);
  for (uint group = 0U; group < 16U; ++group) {
    const ulong coefficient =
        ((ulong)expert * 16UL + group) * IQ36_HIDDEN + output;
    value -= down_sums32[row * 16U + group] * down_mins[coefficient];
  }
  if (row < IQ36_ROUTED_ROWS) value *= router_weights[row];
  contributions[row * IQ36_HIDDEN + output] = value;
}

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void iq36_micro_shared_scalar_gate(
    __global const float *input,
    __global const float *weight,
    __global float *sigmoid_output) {
  const uint token = get_group_id(0);
  const uint lane = get_local_id(0);
  float value = 0.0f;
  for (uint inner = lane; inner < IQ36_HIDDEN; inner += 256U) {
    value += input[token * IQ36_HIDDEN + inner] * weight[inner];
  }
  __local float partial[256];
  partial[lane] = value;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint step = 128U; step > 0U; step >>= 1U) {
    if (lane < step) partial[lane] += partial[lane + step];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lane == 0U) {
    sigmoid_output[token] = 1.0f / (1.0f + exp(-partial[0]));
  }
}

__kernel void iq36_micro_complete_scatter_add(
    __global const float *contributions,
    __global const int *token_rank_to_row,
    __global const float *shared_gate,
    __global float *output) {
  const uint index = get_global_id(0);
  if (index >= IQ36_TOKENS * IQ36_HIDDEN) return;
  const uint token = index >> 11U;
  const uint hidden = index & 2047U;
  float value = 0.0f;
  for (uint rank = 0U; rank < 8U; ++rank) {
    const int row = token_rank_to_row[token * 8U + rank];
    value += contributions[(uint)row * IQ36_HIDDEN + hidden];
  }
  value += contributions[
      (IQ36_ROUTED_ROWS + token) * IQ36_HIDDEN + hidden] *
      shared_gate[token];
  output[index] = value;
}
