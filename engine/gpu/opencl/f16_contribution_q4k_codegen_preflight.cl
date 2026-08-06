#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

float8 restore_q4k_affine_f16_contribution(
    int8 dot, float8 scale, float8 minimum, float input_sum) {
  return convert_float8(dot) * scale - minimum * input_sum;
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void f16_contribution_q4k_gateup_preflight(
    __global const short8 *input,
    __global const uint4 *gate_codes,
    __global const uint4 *up_codes,
    __global const float8 *gate_scales,
    __global const float8 *gate_minimums,
    __global const float8 *up_scales,
    __global const float8 *up_minimums,
    __global const float *input_sums,
    __global half8 *swiglu_output) {
  const uint lane = get_sub_group_local_id();
  const int8 gate_dot = intel_sub_group_i8_u4_matrix_mad_k32(
      input[lane], gate_codes[lane], (int8)(0));
  const int8 up_dot = intel_sub_group_i8_u4_matrix_mad_k32(
      input[lane], up_codes[lane], (int8)(0));
  const float input_sum = input_sums[lane];
  const float8 gate = restore_q4k_affine_f16_contribution(
      gate_dot, gate_scales[lane], gate_minimums[lane], input_sum);
  const float8 up = restore_q4k_affine_f16_contribution(
      up_dot, up_scales[lane], up_minimums[lane], input_sum);
  const float8 swiglu = gate * (1.0f / (1.0f + exp(-gate))) * up;
  swiglu_output[lane] = convert_half8_rte(swiglu);
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void f16_contribution_q4k_down_preflight(
    __global const short8 *input,
    __global const uint4 *down_codes,
    __global const float8 *down_scales,
    __global const float8 *down_minimums,
    __global const float *input_sums,
    __global const float *router_weights,
    __global half8 *weighted_f16_contribution) {
  const uint lane = get_sub_group_local_id();
  const int8 down_dot = intel_sub_group_i8_u4_matrix_mad_k32(
      input[lane], down_codes[lane], (int8)(0));
  const float8 restored = restore_q4k_affine_f16_contribution(
      down_dot, down_scales[lane], down_minimums[lane], input_sums[lane]);
  weighted_f16_contribution[lane] =
      convert_half8_rte(restored * router_weights[lane]);
}
