#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#pragma OPENCL EXTENSION cl_khr_subgroup_non_uniform_arithmetic : enable
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable

// One fixed XMX feasibility shape for ADR 0043's route reflection:
// 65,536 independent 8x16x128 F16-F16-F32 tiles.  Every accumulator
// contributes to a per-tile checksum so the compiler cannot elide work.
#define IQ36_DPAS_TILE_COUNT 65536U
#define IQ36_DPAS_M 8U
#define IQ36_DPAS_N 16U
#define IQ36_DPAS_K 128U

__attribute__((reqd_work_group_size(16, 1, 1)))
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void iq36_f16_dpas_8x16x128_preflight(
    __global const half* a,
    __global const half* b,
    __global float* checksum,
    __global float* first_tile) {
  const uint tile = (uint)get_group_id(0);
  const uint lane = (uint)get_sub_group_local_id();
  float8 accumulator = (float8)(0.0f);

  #pragma unroll
  for (uint k_block = 0; k_block < IQ36_DPAS_K / 16U; ++k_block) {
    const uint k_base = k_block * 16U;
    const short8 a_fragment = (short8)(
        as_short(a[0U * IQ36_DPAS_K + k_base + lane]),
        as_short(a[1U * IQ36_DPAS_K + k_base + lane]),
        as_short(a[2U * IQ36_DPAS_K + k_base + lane]),
        as_short(a[3U * IQ36_DPAS_K + k_base + lane]),
        as_short(a[4U * IQ36_DPAS_K + k_base + lane]),
        as_short(a[5U * IQ36_DPAS_K + k_base + lane]),
        as_short(a[6U * IQ36_DPAS_K + k_base + lane]),
        as_short(a[7U * IQ36_DPAS_K + k_base + lane]));
    int8 b_fragment;
    #pragma unroll
    for (uint pair = 0; pair < 8U; ++pair) {
      const uint k0 = k_base + pair * 2U;
      const half2 packed = (half2)(
          b[k0 * IQ36_DPAS_N + lane],
          b[(k0 + 1U) * IQ36_DPAS_N + lane]);
      b_fragment[pair] = as_int(packed);
    }
    accumulator = intel_sub_group_f16_f16_matrix_mad_k16(
        a_fragment, b_fragment, accumulator);
  }

  float tile_sum = accumulator.s0 + accumulator.s1 + accumulator.s2 +
      accumulator.s3 + accumulator.s4 + accumulator.s5 + accumulator.s6 +
      accumulator.s7;
  tile_sum = sub_group_reduce_add(tile_sum);
  if (lane == 0U) checksum[tile] = tile_sum;
  if (tile == 0U) {
    first_tile[0U * IQ36_DPAS_N + lane] = accumulator.s0;
    first_tile[1U * IQ36_DPAS_N + lane] = accumulator.s1;
    first_tile[2U * IQ36_DPAS_N + lane] = accumulator.s2;
    first_tile[3U * IQ36_DPAS_N + lane] = accumulator.s3;
    first_tile[4U * IQ36_DPAS_N + lane] = accumulator.s4;
    first_tile[5U * IQ36_DPAS_N + lane] = accumulator.s5;
    first_tile[6U * IQ36_DPAS_N + lane] = accumulator.s6;
    first_tile[7U * IQ36_DPAS_N + lane] = accumulator.s7;
  }
}
