#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void m8_u4_preflight(
    __global const short8 * a,
    __global const uint4 * b,
    __global int8 * output) {
  const uint lane = get_sub_group_local_id();
  output[lane] = intel_sub_group_i8_u4_matrix_mad_k32(
      a[lane], b[lane], (int8)(0));
}
