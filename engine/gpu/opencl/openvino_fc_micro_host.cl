#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Fixed batch-1 decode host for a generated gemmstone U4/F16 microkernel.
#define DECLARE_2D_TILE_OPS(tile_type, element_type, sg, br, bc, nbr, nbc)

#if IQ36_ROW_MAJOR_METADATA
#define IQ36_METADATA_LD(m, k) ((k) / 64)
#else
#define IQ36_METADATA_LD(m, k) (m)
#endif

/* IQ36_MICRO_SHIM */

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void IQ36_KERNEL_NAME(
    __global const half *input_ptr,
    __global const uchar *weight_ptr,
    __global half *out_ptr,
    __global const half *weight_scales,
    __global const uchar *weight_zps,
    int m,
    int k,
    int n) {
  const uint subgroup_m =
      sub_group_broadcast(get_local_id(0) / 16U, 0);
  const uint subgroup_n = sub_group_broadcast(get_local_id(1), 0);
  const uint wg_m = get_group_id(0) * ugemm_moe_wg_tile_m;
  const uint wg_n = get_group_id(1) * ugemm_moe_wg_tile_n;
#if IQ36_K_PARALLEL_LOCAL
  const uint subgroup_k = sub_group_broadcast(get_local_id(2), 0);
  __local char slm[ugemm_moe_slm_size];
#define IQ36_K_LOCAL_ARGS , subgroup_k
#define IQ36_SLM_ARG slm
#else
#define IQ36_K_LOCAL_ARGS
#define IQ36_SLM_ARG 0
#endif

  const ugemm_moe_c_type tile = ugemm_moe(
      weight_ptr, k, input_ptr, k, m, n, k,
      wg_m, wg_n, 0, subgroup_m, subgroup_n IQ36_K_LOCAL_ARGS, IQ36_SLM_ARG,
      weight_scales, weight_zps, IQ36_METADATA_LD(m, k));
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
      if (tile_m + row >= (uint)m) continue;
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
