#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Batch-1 decode host for one generated gemmstone U4/F16 microkernel and up
// to four independent FC parameter/output streams.  Every work-group belongs
// to exactly one projection, so pointer selection is uniform across every
// subgroup participating in the local-K reduction.
#define DECLARE_2D_TILE_OPS(tile_type, element_type, sg, br, bc, nbr, nbc)

/* IQ36_MICRO_SHIM */

#define IQ36_GROUPS(m) (((m) + ugemm_moe_wg_tile_m - 1) / \
                        ugemm_moe_wg_tile_m)

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void IQ36_KERNEL_NAME(
    __global const half *input_ptr,
    __global const uchar *weight0,
    __global half *out0,
    __global const half *scale0,
    __global const uchar *zp0,
    __global const uchar *weight1,
    __global half *out1,
    __global const half *scale1,
    __global const uchar *zp1,
    __global const uchar *weight2,
    __global half *out2,
    __global const half *scale2,
    __global const uchar *zp2,
    __global const uchar *weight3,
    __global half *out3,
    __global const half *scale3,
    __global const uchar *zp3,
    int k,
    int n) {
  const uint group = get_group_id(0);
  const uint groups0 = IQ36_GROUPS(IQ36_M0);
  const uint groups1 = IQ36_GROUPS(IQ36_M1);
  const uint groups2 = IQ36_GROUPS(IQ36_M2);
  uint projection_group = group;
  uint m = IQ36_M0;
  __global const uchar *weight_ptr = weight0;
  __global half *out_ptr = out0;
  __global const half *weight_scales = scale0;
  __global const uchar *weight_zps = zp0;
  if (group >= groups0) {
    projection_group -= groups0;
    m = IQ36_M1;
    weight_ptr = weight1;
    out_ptr = out1;
    weight_scales = scale1;
    weight_zps = zp1;
  }
  if (group >= groups0 + groups1) {
    projection_group = group - groups0 - groups1;
    m = IQ36_M2;
    weight_ptr = weight2;
    out_ptr = out2;
    weight_scales = scale2;
    weight_zps = zp2;
  }
  if (group >= groups0 + groups1 + groups2) {
    projection_group = group - groups0 - groups1 - groups2;
    m = IQ36_M3;
    weight_ptr = weight3;
    out_ptr = out3;
    weight_scales = scale3;
    weight_zps = zp3;
  }

  const uint subgroup_m =
      sub_group_broadcast(get_local_id(0) / 16U, 0);
  const uint subgroup_n = sub_group_broadcast(get_local_id(1), 0);
#if IQ36_K_PARALLEL_LOCAL
  const uint subgroup_k = sub_group_broadcast(get_local_id(2), 0);
#define IQ36_K_LOCAL_ARGS , subgroup_k
#define IQ36_SLM_ARG slm
#else
#define IQ36_K_LOCAL_ARGS
#define IQ36_SLM_ARG 0
#endif
  const uint wg_m = projection_group * ugemm_moe_wg_tile_m;
  const uint wg_n = get_group_id(1) * ugemm_moe_wg_tile_n;
#if IQ36_K_PARALLEL_LOCAL
  __local char slm[ugemm_moe_slm_size];
#endif
#if IQ36_M0 == 1
  // The universal 64-row microkernel cannot represent the graph's scalar
  // router projection without reading a fictitious row stride.
#if IQ36_K_PARALLEL_LOCAL
  __local float scalar_partial[256];
#endif
  if (group == 0U) {
    const uint local_linear =
        get_local_id(0) + get_local_size(0) *
            (get_local_id(1) + get_local_size(1) * get_local_id(2));
    const uint local_count =
        get_local_size(0) * get_local_size(1) * get_local_size(2);
#if IQ36_K_PARALLEL_LOCAL
    // Decode has one token: preserve the admitted 256-way K reduction.
    float sum = 0.0f;
    for (uint index = local_linear; index < (uint)k; index += local_count) {
      const uchar packed_weight = weight0[index >> 1];
      const uint weight = (index & 1U) != 0U
          ? packed_weight >> 4 : packed_weight & 0x0FU;
      const uint group_index = index >> 6;
      const uchar packed_zp = zp0[group_index >> 1];
      const uint zp = (group_index & 1U) != 0U
          ? packed_zp >> 4 : packed_zp & 0x0FU;
      const float dequant =
          ((float)((int)weight - (int)zp)) *
          convert_float(scale0[group_index]);
      sum = fma(convert_float(input_ptr[index]), dequant, sum);
    }
    scalar_partial[local_linear] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint stride = local_count >> 1; stride > 0U; stride >>= 1) {
      if (local_linear < stride) {
        scalar_partial[local_linear] += scalar_partial[local_linear + stride];
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (local_linear == 0U) {
      out0[0] = convert_half_rte(scalar_partial[0]);
    }
#else
    // Prefill assigns token columns directly to work-items. Parallelizing K
    // here would add a reduction/barrier for every column in the tile.
    for (uint column = local_linear; column < ugemm_moe_wg_tile_n;
         column += local_count) {
      const uint token = wg_n + column;
      if (token >= (uint)n) continue;
      float sum = 0.0f;
      for (uint index = 0U; index < (uint)k; ++index) {
        const uchar packed_weight = weight0[index >> 1];
        const uint weight = (index & 1U) != 0U
            ? packed_weight >> 4 : packed_weight & 0x0FU;
        const uint group_index = index >> 6;
        const uchar packed_zp = zp0[group_index >> 1];
        const uint zp = (group_index & 1U) != 0U
            ? packed_zp >> 4 : packed_zp & 0x0FU;
        const float dequant =
            ((float)((int)weight - (int)zp)) *
            convert_float(scale0[group_index]);
        sum = fma(convert_float(input_ptr[(ulong)token * (ulong)k + index]),
                  dequant, sum);
      }
      out0[token] = convert_half_rte(sum);
    }
#endif
    return;
  }
#endif

  const ugemm_moe_c_type tile = ugemm_moe(
      weight_ptr, k, input_ptr, k, m, n, k,
      wg_m, wg_n, 0, subgroup_m, subgroup_n IQ36_K_LOCAL_ARGS, IQ36_SLM_ARG,
      weight_scales, weight_zps, m);
#if IQ36_K_PARALLEL_LOCAL
  if (subgroup_k > 0U) return;
#endif

  const uint tile_m = wg_m + subgroup_m * ugemm_moe_sg_tile_m;
  const uint tile_n = wg_n + subgroup_n * ugemm_moe_sg_tile_n;
  const uint lane = get_sub_group_local_id();
  __attribute__((opencl_unroll_hint))
  for (uint column = 0U; column < ugemm_moe_sg_tile_n; ++column) {
    if (tile_n + column >= (uint)n) continue;
    __attribute__((opencl_unroll_hint))
    for (uint row0 = 0U; row0 < ugemm_moe_sg_tile_m;
         row0 += ugemm_moe_c_type_block0) {
      const uint row = row0 + lane;
      if (tile_m + row >= m) continue;
      const uint register_i =
          row0 / ugemm_moe_c_type_block0 +
          ugemm_moe_c_type_nblock0 *
              (column / ugemm_moe_c_type_block1);
      const uint register_j = column % ugemm_moe_c_type_block1;
      out_ptr[(ulong)(tile_n + column) * (ulong)m + tile_m + row] =
          convert_half_rte(tile.x[register_i][register_j]);
    }
  }
}
