#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// The pinned gemmstone shim emits this declaration hook before the callable
// microkernel.  The native host stores the returned register tile directly,
// so none of the generic tile helper overloads are needed here.
#define DECLARE_2D_TILE_OPS(tile_type, element_type, sg, br, bc, nbr, nbc)

/* IQ36_MICRO_SHIM */

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void IQ36_KERNEL_NAME(
    __global const half *input_ptr,
    __global const uchar *weight_ptr,
    __global half *out_ptr,
    __global const int *experts_ids,
    __global const int *input_offset_per_expert,
    __global const int *n_array,
    int m,
    int k,
    __global const half *weight_scales,
    __global const uchar *weight_zps) {
  const uint batch = get_group_id(2);
  const int input_offset =
      sub_group_broadcast(input_offset_per_expert[batch], 0);
  const int expert_id = sub_group_broadcast(experts_ids[batch], 0);
  const int cur_n_tokens = sub_group_broadcast(n_array[batch], 0);

  input_ptr += (ulong)input_offset * (ulong)k;
  out_ptr += (ulong)input_offset * (ulong)m;
  weight_ptr += (ulong)expert_id * (ulong)m * (ulong)k / 2UL;
  const int num_groups = k / 32;
  weight_scales +=
      (ulong)expert_id * (ulong)m * (ulong)num_groups;
  weight_zps +=
      (ulong)expert_id * (ulong)m * (ulong)num_groups / 2UL;

  const uint subgroup_m =
      sub_group_broadcast(get_local_id(0) / 16U, 0);
  const uint subgroup_n = sub_group_broadcast(get_local_id(1), 0);
  const uint wg_m = get_group_id(0) * ugemm_moe_wg_tile_m;
  const uint wg_n = get_group_id(1) * ugemm_moe_wg_tile_n;
  if (wg_n >= (uint)cur_n_tokens) return;

  const ugemm_moe_c_type tile = ugemm_moe(
      weight_ptr, k, input_ptr, k, m, cur_n_tokens, k,
      wg_m, wg_n, 0, subgroup_m, subgroup_n, 0,
      weight_scales, weight_zps, m);
  const uint tile_m = wg_m + subgroup_m * ugemm_moe_sg_tile_m;
  const uint tile_n = wg_n + subgroup_n * ugemm_moe_sg_tile_n;
  const uint lane = get_sub_group_local_id();

  __attribute__((opencl_unroll_hint))
  for (uint column = 0U; column < 48U; ++column) {
    if (tile_n + column >= (uint)cur_n_tokens) continue;
    __attribute__((opencl_unroll_hint))
    for (uint row0 = 0U; row0 < 32U; row0 += 16U) {
      const uint row = row0 + lane;
      if (tile_m + row >= (uint)m) continue;
      const uint register_i = row0 / 16U + 2U * (column / 8U);
      const uint register_j = column & 7U;
      out_ptr[(ulong)(tile_n + column) * (ulong)m + tile_m + row] =
          convert_half_rte(tile.x[register_i][register_j]);
    }
  }
}
