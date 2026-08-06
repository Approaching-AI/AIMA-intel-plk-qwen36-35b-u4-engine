#pragma OPENCL EXTENSION cl_khr_fp16 : enable
/* IQ36_MICRO_SHIM */

// Standalone 128k component for the accepted generated M256/N16 attention
// packages.  The staged path materializes all sixteen independent generated
// KQ columns per KV head as raw F32 scores; one owner per KV head then replays
// the original chronological online-softmax and VS recurrence.

#define IQ36_D 256
#define IQ36_CONTEXT 131072
#define IQ36_Q_HEADS 16
#define IQ36_KV_HEADS 2
#define IQ36_GQA_GROUP 8
#define IQ36_KQ_COLUMNS 16
#define IQ36_SUBGROUP 16
#define SUBGROUP_SIZE IQ36_SUBGROUP
#define IQ36_SCALE_LOG2E 0.09016844f
#define IQ36_COMPONENT_CAPTURE 1
#define IQ36_COMPONENT_FUSED 2
#define IQ36_COMPONENT_STAGED 3
#define IQ36_COMPONENT_DUAL_COHORT 4
#define IQ36_COMPONENT_SOFTMAX_STAGE 5
#define IQ36_COMPONENT_VS_STAGE 6
#define IQ36_COMPONENT_SOFTMAX_TRAFFIC 7
#define IQ36_COMPONENT_TRIPLE_COHORT 8
#define IQ36_COMPONENT_NORMALIZED_DUAL_COHORT 9
#define IQ36_COMPONENT_DENSE_TRAFFIC 10
#define IQ36_COMPONENT_TRIPLE_OFFICIAL_PREFETCH 11
#ifndef IQ36_COMPONENT_PROGRAM
#error "IQ36_COMPONENT_PROGRAM must select capture, fused, staged, dual cohort, softmax stage, VS stage, softmax traffic, triple cohort, normalized dual cohort, dense traffic, or triple official prefetch"
#endif

#define IQ36_DIV_UP(x, y) (((x) + (y) - 1) / (y))
#define IQ36_SG_PER_WG (ugemm_kq_sg_per_wg_m * ugemm_kq_sg_per_wg_n)
#define IQ36_Q_TILE_SG_N \
  IQ36_DIV_UP(ugemm_kq_wg_tile_n, IQ36_SG_PER_WG)

typedef ugemm_kq_c_type iq36_component_score_tile;
typedef ugemm_vs_c_type iq36_component_accumulator_tile;
DECLARE_2D_TILE(
    iq36_component_query_tile, uint, IQ36_SUBGROUP, IQ36_D / 2, 1, 1,
    IQ36_Q_TILE_SG_N)
DECLARE_2D_TILE_BLOCK_OPS(
    iq36_component_query_tile, uint, IQ36_SUBGROUP, IQ36_D / 2, 1, 1,
    IQ36_Q_TILE_SG_N)
DECLARE_2D_TILE(
    iq36_component_accumulator_half_tile, half, IQ36_SUBGROUP,
    ugemm_vs_sg_tile_m, 8, 1, ugemm_vs_sg_tile_n / 8)
DECLARE_2D_TILE(
    iq36_component_score_half2_tile, uint, IQ36_SUBGROUP,
    ugemm_kq_c_type_block0, ugemm_kq_c_type_block1 / 2,
    ugemm_kq_c_type_nblock0, ugemm_kq_c_type_nblock1)
DECLARE_2D_TILE(
    iq36_component_score_sum_tile, float, IQ36_SUBGROUP,
    ugemm_kq_sg_tile_n, 1, 1, 1)
DECLARE_2D_TILE(
    iq36_component_accumulator_scale_tile, float, IQ36_SUBGROUP,
    ugemm_vs_sg_tile_n, 1, 1, 1)
DECLARE_2D_TILE_COPY_REBLOCK(
    iq36_component_accumulator_tile, IQ36_SUBGROUP,
    ugemm_vs_c_type_block0, ugemm_vs_c_type_block1,
    ugemm_vs_c_type_nblock0, ugemm_vs_c_type_nblock1,
    iq36_component_accumulator_half_tile, IQ36_SUBGROUP,
    ugemm_vs_sg_tile_m, 8, 1, ugemm_vs_sg_tile_n / 8)
DECLARE_2D_TILE_VREDUCE(
    iq36_component_score_tile, IQ36_SUBGROUP,
    ugemm_kq_c_type_block0, ugemm_kq_c_type_block1,
    ugemm_kq_c_type_nblock0, ugemm_kq_c_type_nblock1,
    iq36_component_score_sum_tile, IQ36_SUBGROUP,
    ugemm_kq_sg_tile_n, 1, 1, 1)
DECLARE_2D_TILE_HREDUCE(
    iq36_component_accumulator_tile, IQ36_SUBGROUP,
    ugemm_vs_c_type_block0, ugemm_vs_c_type_block1,
    ugemm_vs_c_type_nblock0, ugemm_vs_c_type_nblock1,
    iq36_component_accumulator_scale_tile, IQ36_SUBGROUP,
    ugemm_vs_sg_tile_n, 1, 1, 1)

#define iq36_component_binary_add(x, y) ((x) + (y))
#define iq36_component_binary_mul(x, y) ((x) * (y))
#define iq36_component_scaled_exp(x) \
  native_vexp2((x) * IQ36_SCALE_LOG2E)
#define iq36_component_rescale(x, y) \
  native_vexp2(((x) - (y)) * IQ36_SCALE_LOG2E)

inline void iq36_component_load_query(
    const __global half* query_head_base,
    __local half* query_slm,
    uint subgroup);
inline void iq36_component_initialize_max(
    __local float* max_slm,
    uint subgroup);

#if IQ36_COMPONENT_PROGRAM == IQ36_COMPONENT_DENSE_TRAFFIC
#define IQ36_DENSE_TRAFFIC_SUBGROUPS 48
#define IQ36_DENSE_TRAFFIC_WORKITEMS \
  (IQ36_SUBGROUP * IQ36_DENSE_TRAFFIC_SUBGROUPS)
#define IQ36_DENSE_TRAFFIC_UINT8S_PER_HEAD \
  (((ulong)IQ36_CONTEXT * IQ36_D * sizeof(half)) / sizeof(uint8))

// Two useful workgroups, exactly matching the accepted triple carrier's
// outer geometry.  Every uint8 in the selected KV head's K and V payload is
// consumed exactly once.  Separate output-dependent accumulators prevent
// either input stream from being removed while adding only a small checksum
// tail to the full 256-MiB read.
__attribute__((intel_reqd_sub_group_size(IQ36_SUBGROUP)))
__attribute__((reqd_work_group_size(16, 48, 1)))
__kernel void iq36_exact_attention_dense_traffic_ceiling(
    const __global half* key,
    const __global half* value,
    __global uint8* checksums) {
  const uint subgroup = sub_group_broadcast(get_local_id(1), 0);
  const uint linear_local_id =
      subgroup * IQ36_SUBGROUP + (uint)get_local_id(0);
  const uint kv_head = (uint)get_group_id(1);
  const ulong head_half_offset =
      (ulong)kv_head * IQ36_CONTEXT * IQ36_D;
  const __global uint* key_words =
      (const __global uint*)(key + head_half_offset);
  const __global uint* value_words =
      (const __global uint*)(value + head_half_offset);
  uint8 key_accumulator = (uint8)(0x9e3779b9U);
  uint8 value_accumulator = (uint8)(0x7f4a7c15U);

  for (ulong vector_index = linear_local_id;
       vector_index < IQ36_DENSE_TRAFFIC_UINT8S_PER_HEAD;
       vector_index += IQ36_DENSE_TRAFFIC_WORKITEMS) {
    const ulong word_offset = vector_index * 8UL;
    key_accumulator ^= vload8(0, key_words + word_offset);
    value_accumulator += vload8(0, value_words + word_offset);
  }

  const ulong checksum_base =
      (ulong)kv_head * 2UL * IQ36_DENSE_TRAFFIC_WORKITEMS;
  checksums[checksum_base + linear_local_id] = key_accumulator;
  checksums[
      checksum_base + IQ36_DENSE_TRAFFIC_WORKITEMS + linear_local_id] =
      value_accumulator;
}
#endif

#if IQ36_COMPONENT_PROGRAM == IQ36_COMPONENT_NORMALIZED_DUAL_COHORT
#define IQ36_NORMALIZED_PRODUCER_SUBGROUPS 16
#define IQ36_NORMALIZED_CONSUMER_SUBGROUPS 16
#define IQ36_NORMALIZED_TOTAL_SUBGROUPS 32
#define IQ36_NORMALIZED_BUFFER_ELEMENTS \
  (ugemm_kq_wg_tile_m * ugemm_kq_wg_tile_n)
#define IQ36_NORMALIZED_DOUBLE_UINTS \
  (IQ36_NORMALIZED_BUFFER_ELEMENTS)
#define IQ36_NORMALIZED_MAX_UINT_OFFSET \
  (IQ36_NORMALIZED_DOUBLE_UINTS)
#define IQ36_NORMALIZED_SUM_UINT_OFFSET \
  (IQ36_NORMALIZED_MAX_UINT_OFFSET + 256)
#define IQ36_NORMALIZED_RESCALE_UINT_OFFSET \
  (IQ36_NORMALIZED_SUM_UINT_OFFSET + 256)
#define IQ36_NORMALIZED_PIPELINE_SLAB_UINTS \
  (IQ36_NORMALIZED_RESCALE_UINT_OFFSET + 2 * 256)

// Exact two-cohort successor to the sub-threshold triple carrier.  The first
// cohort keeps generated KQ and chronological softmax in registers and hands
// only the normalized F16 image plus exact accumulator rescale through
// double-buffered SLM.  The second cohort runs generated VS.  This removes
// the raw-F32 SLM roundtrip and one cross-cohort rendezvous per block.
__attribute__((intel_reqd_sub_group_size(IQ36_SUBGROUP)))
__attribute__((reqd_work_group_size(16, 32, 1)))
__kernel void iq36_exact_score_normalized_dual_cohort(
    const __global half* query,
    const __global half* key,
    const __global half* value,
    __global half* output) {
  const uint subgroup = sub_group_broadcast(get_local_id(1), 0);
  const bool producer =
      subgroup < (uint)IQ36_NORMALIZED_PRODUCER_SUBGROUPS;
  const uint cohort_subgroup = producer
      ? subgroup
      : subgroup - (uint)IQ36_NORMALIZED_PRODUCER_SUBGROUPS;
  const uint cohort_linear_local_id =
      (uint)get_local_id(0) + IQ36_SUBGROUP * cohort_subgroup;
  const uint kv_head = (uint)get_group_id(1);
  const __global half* query_head_base =
      query + (ulong)kv_head * IQ36_GQA_GROUP * IQ36_D;
  const __global half* key_base =
      key + (ulong)kv_head * IQ36_CONTEXT * IQ36_D;
  const __global half* value_base =
      value + (ulong)kv_head * IQ36_CONTEXT * IQ36_D;

  __local half query_slm[IQ36_D * ugemm_kq_wg_tile_n];
  __local uint pipeline_slab[IQ36_NORMALIZED_PIPELINE_SLAB_UINTS];
  __local char ugemm_slm[1];
  __local half* normalized_score_double_slm =
      (__local half*)pipeline_slab;
  __local float* max_slm =
      (__local float*)(
          pipeline_slab + IQ36_NORMALIZED_MAX_UINT_OFFSET);
  __local float* sum_slm =
      (__local float*)(
          pipeline_slab + IQ36_NORMALIZED_SUM_UINT_OFFSET);
  __local float* rescale_double_slm =
      (__local float*)(
          pipeline_slab + IQ36_NORMALIZED_RESCALE_UINT_OFFSET);
  __local NamedBarrier_t* producer_internal_barrier =
      named_barrier_init(IQ36_NORMALIZED_PRODUCER_SUBGROUPS);
  __local NamedBarrier_t* pipeline_barrier =
      named_barrier_init(IQ36_NORMALIZED_TOTAL_SUBGROUPS);
  __local NamedBarrier_t* consumer_internal_barrier =
      named_barrier_init(IQ36_NORMALIZED_CONSUMER_SUBGROUPS);

  iq36_component_accumulator_tile accumulator;
  iq36_component_score_sum_tile running_sum;
  iq36_component_score_sum_tile running_max;
  iq36_component_score_sum_tile old_running_max;
  if (producer) {
    iq36_component_load_query(
        query_head_base, query_slm, cohort_subgroup);
    iq36_component_initialize_max(max_slm, cohort_subgroup);
    tile_fill(running_sum, 0.0f);
    tile_fill(running_max, -INFINITY);
  } else {
    tile_fill(accumulator, 0.0f);
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  const uint block_count = IQ36_CONTEXT / ugemm_kq_wg_tile_m;
  for (uint block = 0U; block < block_count; ++block) {
    const uint key_begin = block * ugemm_kq_wg_tile_m;
    const bool first = block == 0U;
    const bool last = block + 1U == block_count;
    __local half* current_normalized_score =
        normalized_score_double_slm +
        (block & 1U) * IQ36_NORMALIZED_BUFFER_ELEMENTS;
    __local float* current_rescale =
        rescale_double_slm + (block & 1U) * 256;
    if (producer) {
      cooperative_prefetch_2d_rem(
          key_base, IQ36_D, IQ36_CONTEXT,
          ugemm_kq_wg_tile_m, 64, IQ36_D,
          cohort_subgroup, IQ36_NORMALIZED_PRODUCER_SUBGROUPS,
          IQ36_SUBGROUP, LSC_LDCC_L1C_L3C);
      iq36_component_score_tile score = ugemm_kq(
          key_base, IQ36_D, query_slm, IQ36_D, IQ36_CONTEXT,
          ugemm_kq_wg_tile_n, IQ36_D, (int)key_begin, 0, 0,
          cohort_subgroup, 0, ugemm_slm);
      tile_vreduce_max(score, &running_max);
      max_slm[
          cohort_subgroup * IQ36_SUBGROUP +
          (uint)get_sub_group_local_id()] =
          tile_access(
              running_max, 0, 0, IQ36_SUBGROUP,
              ugemm_kq_sg_tile_n, 1, 1);
      work_group_named_barrier(
          producer_internal_barrier, CLK_LOCAL_MEM_FENCE);
      float reduced_running_max = -INFINITY;
      #pragma unroll
      for (uint subgroup_row = 0U;
           subgroup_row < IQ36_NORMALIZED_PRODUCER_SUBGROUPS;
           ++subgroup_row) {
        reduced_running_max = max(
            reduced_running_max,
            max_slm[
                subgroup_row * IQ36_SUBGROUP +
                (uint)get_sub_group_local_id()]);
      }
      tile_access(
          running_max, 0, 0, IQ36_SUBGROUP,
          ugemm_kq_sg_tile_n, 1, 1) =
          reduced_running_max;
      tile_vbroadcast_sub(&score, running_max);
      tile_elementwise(score, iq36_component_scaled_exp);
      iq36_component_score_sum_tile chunk_sum;
      tile_fill(chunk_sum, 0.0f);
      tile_vreduce_add(score, &chunk_sum);
      iq36_component_score_half2_tile score_half2;
      tile_copy_to_half2(score, score_half2);
      tile_store_t_sys_src2(
          score_half2, (__local uint*)current_normalized_score,
          ugemm_vs_sg_tile_n, ugemm_kq_wg_tile_m / 2,
          (cohort_subgroup * ugemm_kq_sg_tile_m) / 2, 0);

      iq36_component_score_sum_tile block_rescale;
      if (first) {
        tile_fill(block_rescale, 1.0f);
      } else {
        tile_copy(old_running_max, block_rescale);
        tile_binary(
            block_rescale, running_max, iq36_component_rescale);
        tile_binary(
            running_sum, block_rescale,
            iq36_component_binary_mul);
      }
      tile_store_full(
          block_rescale, current_rescale,
          ugemm_kq_wg_tile_n, 0, cohort_subgroup);
      tile_binary(
          running_sum, chunk_sum, iq36_component_binary_add);
      tile_copy(running_max, old_running_max);
      if (last) {
        tile_store_full(
            running_sum, sum_slm, ugemm_kq_wg_tile_n,
            0, cohort_subgroup);
      }
      work_group_named_barrier(
          pipeline_barrier, CLK_LOCAL_MEM_FENCE);
    } else {
      work_group_named_barrier(
          pipeline_barrier, CLK_LOCAL_MEM_FENCE);
      if (!first) {
        iq36_component_score_sum_tile block_rescale;
        tile_load_full(
            &block_rescale, current_rescale,
            ugemm_kq_wg_tile_n, 0, cohort_subgroup);
        iq36_component_accumulator_scale_tile accumulator_scale;
        tile_copy(block_rescale, accumulator_scale);
        tile_hbroadcast_mul(&accumulator, accumulator_scale);
      }
      iq36_component_accumulator_tile chunk_accumulator = ugemm_vs(
          value_base + (ulong)key_begin * IQ36_D,
          IQ36_D, current_normalized_score,
          ugemm_kq_wg_tile_m,
          IQ36_D, ugemm_kq_wg_tile_n,
          ugemm_kq_wg_tile_m,
          0, 0, 0, cohort_subgroup, 0, ugemm_slm);
      tile_binary(
          accumulator, chunk_accumulator,
          iq36_component_binary_add);
    }
  }

  if (!producer) {
    iq36_component_accumulator_scale_tile total_sum;
    iq36_component_accumulator_scale_tile partial_sum;
    tile_fill(total_sum, 0.0f);
    #pragma unroll
    for (uint subgroup_row = 0U;
         subgroup_row < ugemm_kq_sg_per_wg_m;
         ++subgroup_row) {
      tile_load_full(
          &partial_sum, sum_slm, ugemm_kq_wg_tile_n,
          0, subgroup_row);
      tile_binary(
          total_sum, partial_sum, iq36_component_binary_add);
    }
    tile_elementwise(total_sum, native_vrecip);
    tile_hbroadcast_mul(&accumulator, total_sum);
    iq36_component_accumulator_half_tile output_tile;
    tile_copy_reblock(accumulator, &output_tile);
    __local half* output_slm = normalized_score_double_slm;
    tile_store_full(
        output_tile, output_slm, IQ36_D,
        cohort_subgroup * ugemm_vs_sg_tile_m, 0);
    work_group_named_barrier(
        consumer_internal_barrier, CLK_LOCAL_MEM_FENCE);
    #pragma unroll
    for (uint head = 0U; head < IQ36_GQA_GROUP; ++head) {
      output[
          ((ulong)kv_head * IQ36_GQA_GROUP + head) * IQ36_D +
          cohort_linear_local_id] =
          output_slm[
              cohort_linear_local_id + head * IQ36_D];
    }
  }
}
#endif

#if IQ36_COMPONENT_PROGRAM == IQ36_COMPONENT_TRIPLE_COHORT || \
    IQ36_COMPONENT_PROGRAM == IQ36_COMPONENT_TRIPLE_OFFICIAL_PREFETCH
#define IQ36_TRIPLE_KQ_SUBGROUPS 16
#define IQ36_TRIPLE_SOFTMAX_SUBGROUPS 16
#define IQ36_TRIPLE_VS_SUBGROUPS 16
#define IQ36_TRIPLE_TOTAL_SUBGROUPS 48
#define IQ36_TRIPLE_RAW_BUFFER_ELEMENTS \
  (ugemm_kq_wg_tile_m * ugemm_kq_wg_tile_n)
#define IQ36_TRIPLE_NORMALIZED_BUFFER_ELEMENTS \
  (ugemm_kq_wg_tile_m * ugemm_kq_wg_tile_n)
#define IQ36_TRIPLE_RAW_DOUBLE_UINTS \
  (2 * IQ36_TRIPLE_RAW_BUFFER_ELEMENTS)
#define IQ36_TRIPLE_NORMALIZED_DOUBLE_UINTS \
  (IQ36_TRIPLE_NORMALIZED_BUFFER_ELEMENTS)
#define IQ36_TRIPLE_MAX_UINT_OFFSET \
  (IQ36_TRIPLE_RAW_DOUBLE_UINTS + \
   IQ36_TRIPLE_NORMALIZED_DOUBLE_UINTS)
#define IQ36_TRIPLE_SUM_UINT_OFFSET \
  (IQ36_TRIPLE_MAX_UINT_OFFSET + 256)
#define IQ36_TRIPLE_RESCALE_UINT_OFFSET \
  (IQ36_TRIPLE_SUM_UINT_OFFSET + 256)
#define IQ36_TRIPLE_PIPELINE_SLAB_UINTS \
  (IQ36_TRIPLE_RESCALE_UINT_OFFSET + 2 * 256)

// Compiler-only successor to the accepted dual carrier.  Three independent
// sixteen-subgroup cohorts overlap generated KQ, the exact chronological
// softmax recurrence, and generated VS.  Raw F32 and normalized F16 score
// images cross only double-buffered SLM.  After the final pipeline barrier,
// the VS cohort reuses the dead raw-score slab for output reblocking.
__attribute__((intel_reqd_sub_group_size(IQ36_SUBGROUP)))
__attribute__((reqd_work_group_size(16, 48, 1)))
__kernel void iq36_exact_score_triple_cohort(
    const __global half* query,
    const __global half* key,
    const __global half* value,
    __global half* output) {
  const uint subgroup = sub_group_broadcast(get_local_id(1), 0);
  const bool kq_cohort =
      subgroup < (uint)IQ36_TRIPLE_KQ_SUBGROUPS;
  const bool softmax_cohort =
      subgroup >= (uint)IQ36_TRIPLE_KQ_SUBGROUPS &&
      subgroup <
          (uint)(IQ36_TRIPLE_KQ_SUBGROUPS +
                 IQ36_TRIPLE_SOFTMAX_SUBGROUPS);
  const uint cohort_subgroup = kq_cohort
      ? subgroup
      : (softmax_cohort
          ? subgroup - (uint)IQ36_TRIPLE_KQ_SUBGROUPS
          : subgroup -
              (uint)(IQ36_TRIPLE_KQ_SUBGROUPS +
                     IQ36_TRIPLE_SOFTMAX_SUBGROUPS));
  const uint cohort_linear_local_id =
      (uint)get_local_id(0) + IQ36_SUBGROUP * cohort_subgroup;
  const uint kv_head = (uint)get_group_id(1);
  const __global half* query_head_base =
      query + (ulong)kv_head * IQ36_GQA_GROUP * IQ36_D;
  const __global half* key_base =
      key + (ulong)kv_head * IQ36_CONTEXT * IQ36_D;
  const __global half* value_base =
      value + (ulong)kv_head * IQ36_CONTEXT * IQ36_D;

  __local half query_slm[IQ36_D * ugemm_kq_wg_tile_n];
  __local uint pipeline_slab[IQ36_TRIPLE_PIPELINE_SLAB_UINTS];
  __local char ugemm_slm[1];
  __local float* raw_score_double_slm =
      (__local float*)pipeline_slab;
  __local half* normalized_score_double_slm =
      (__local half*)(
          pipeline_slab + IQ36_TRIPLE_RAW_DOUBLE_UINTS);
  __local float* max_slm =
      (__local float*)(
          pipeline_slab + IQ36_TRIPLE_MAX_UINT_OFFSET);
  __local float* sum_slm =
      (__local float*)(
          pipeline_slab + IQ36_TRIPLE_SUM_UINT_OFFSET);
  __local float* rescale_double_slm =
      (__local float*)(
          pipeline_slab + IQ36_TRIPLE_RESCALE_UINT_OFFSET);
  __local NamedBarrier_t* kq_softmax_barrier =
      named_barrier_init(
          IQ36_TRIPLE_KQ_SUBGROUPS +
          IQ36_TRIPLE_SOFTMAX_SUBGROUPS);
  __local NamedBarrier_t* softmax_vs_barrier =
      named_barrier_init(
          IQ36_TRIPLE_SOFTMAX_SUBGROUPS +
          IQ36_TRIPLE_VS_SUBGROUPS);
  __local NamedBarrier_t* softmax_internal_barrier =
      named_barrier_init(IQ36_TRIPLE_SOFTMAX_SUBGROUPS);
  __local NamedBarrier_t* vs_internal_barrier =
      named_barrier_init(IQ36_TRIPLE_VS_SUBGROUPS);

  iq36_component_accumulator_tile accumulator;
  iq36_component_score_sum_tile running_sum;
  iq36_component_score_sum_tile running_max;
  iq36_component_score_sum_tile old_running_max;
  if (kq_cohort) {
    iq36_component_load_query(
        query_head_base, query_slm, cohort_subgroup);
  } else if (softmax_cohort) {
    iq36_component_initialize_max(max_slm, cohort_subgroup);
    tile_fill(running_sum, 0.0f);
    tile_fill(running_max, -INFINITY);
  } else {
    tile_fill(accumulator, 0.0f);
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  if (kq_cohort) {
#if IQ36_COMPONENT_PROGRAM == IQ36_COMPONENT_TRIPLE_OFFICIAL_PREFETCH
    cooperative_prefetch_2d_rem(
        key_base, IQ36_CONTEXT, IQ36_D,
        ugemm_kq_wg_tile_m, IQ36_D, IQ36_D,
        cohort_subgroup, IQ36_TRIPLE_KQ_SUBGROUPS,
        IQ36_SUBGROUP, LSC_LDCC_L1C_L3C);
#else
    cooperative_prefetch_2d_rem(
        key_base, IQ36_D, IQ36_CONTEXT,
        ugemm_kq_wg_tile_m, 64, IQ36_D,
        cohort_subgroup, IQ36_TRIPLE_KQ_SUBGROUPS,
        IQ36_SUBGROUP, LSC_LDCC_L1C_L3C);
#endif
    iq36_component_score_tile first_score = ugemm_kq(
        key_base, IQ36_D, query_slm, IQ36_D, IQ36_CONTEXT,
        ugemm_kq_wg_tile_n, IQ36_D, 0, 0, 0,
        cohort_subgroup, 0, ugemm_slm);
    tile_store_full(
        first_score, raw_score_double_slm,
        ugemm_kq_wg_tile_m,
        cohort_subgroup * ugemm_kq_sg_tile_m, 0);
    work_group_named_barrier(
        kq_softmax_barrier, CLK_LOCAL_MEM_FENCE);
  } else if (softmax_cohort) {
    work_group_named_barrier(
        kq_softmax_barrier, CLK_LOCAL_MEM_FENCE);
  }

  const uint block_count = IQ36_CONTEXT / ugemm_kq_wg_tile_m;
  for (uint block = 0U; block < block_count; ++block) {
    const uint key_begin = block * ugemm_kq_wg_tile_m;
    const bool first = block == 0U;
    const bool last = block + 1U == block_count;
    if (kq_cohort) {
      const uint next_block = block + 1U;
      if (next_block < block_count) {
        __local float* next_raw_score =
            raw_score_double_slm +
            (next_block & 1U) *
                IQ36_TRIPLE_RAW_BUFFER_ELEMENTS;
#if IQ36_COMPONENT_PROGRAM == IQ36_COMPONENT_TRIPLE_OFFICIAL_PREFETCH
        const __global half* next_key =
            key_base +
            (ulong)next_block * ugemm_kq_wg_tile_m * IQ36_D;
        cooperative_prefetch_2d_rem(
            next_key,
            IQ36_CONTEXT - next_block * ugemm_kq_wg_tile_m,
            IQ36_D,
            ugemm_kq_wg_tile_m, IQ36_D, IQ36_D,
            cohort_subgroup, IQ36_TRIPLE_KQ_SUBGROUPS,
            IQ36_SUBGROUP, LSC_LDCC_L1C_L3C);
#else
        cooperative_prefetch_2d_rem(
            key_base, IQ36_D, IQ36_CONTEXT,
            ugemm_kq_wg_tile_m, 64, IQ36_D,
            cohort_subgroup, IQ36_TRIPLE_KQ_SUBGROUPS,
            IQ36_SUBGROUP, LSC_LDCC_L1C_L3C);
#endif
        iq36_component_score_tile next_score = ugemm_kq(
            key_base, IQ36_D, query_slm, IQ36_D, IQ36_CONTEXT,
            ugemm_kq_wg_tile_n, IQ36_D,
            (int)(next_block * ugemm_kq_wg_tile_m), 0, 0,
            cohort_subgroup, 0, ugemm_slm);
        tile_store_full(
            next_score, next_raw_score,
            ugemm_kq_wg_tile_m,
            cohort_subgroup * ugemm_kq_sg_tile_m, 0);
      }
      work_group_named_barrier(
          kq_softmax_barrier, CLK_LOCAL_MEM_FENCE);
    } else if (softmax_cohort) {
      __local float* current_raw_score =
          raw_score_double_slm +
          (block & 1U) * IQ36_TRIPLE_RAW_BUFFER_ELEMENTS;
      __local half* current_normalized_score =
          normalized_score_double_slm +
          (block & 1U) *
              IQ36_TRIPLE_NORMALIZED_BUFFER_ELEMENTS;
      __local float* current_rescale =
          rescale_double_slm + (block & 1U) * 256;
      iq36_component_score_tile score;
      tile_load_full(
          &score, current_raw_score, ugemm_kq_wg_tile_m,
          cohort_subgroup * ugemm_kq_sg_tile_m, 0);
      tile_vreduce_max(score, &running_max);
      max_slm[
          cohort_subgroup * IQ36_SUBGROUP +
          (uint)get_sub_group_local_id()] =
          tile_access(
              running_max, 0, 0, IQ36_SUBGROUP,
              ugemm_kq_sg_tile_n, 1, 1);
      work_group_named_barrier(
          softmax_internal_barrier, CLK_LOCAL_MEM_FENCE);
      float reduced_running_max = -INFINITY;
      #pragma unroll
      for (uint subgroup_row = 0U;
           subgroup_row < IQ36_TRIPLE_SOFTMAX_SUBGROUPS;
           ++subgroup_row) {
        reduced_running_max = max(
            reduced_running_max,
            max_slm[
                subgroup_row * IQ36_SUBGROUP +
                (uint)get_sub_group_local_id()]);
      }
      tile_access(
          running_max, 0, 0, IQ36_SUBGROUP,
          ugemm_kq_sg_tile_n, 1, 1) =
          reduced_running_max;
      tile_vbroadcast_sub(&score, running_max);
      tile_elementwise(score, iq36_component_scaled_exp);
      iq36_component_score_sum_tile chunk_sum;
      tile_fill(chunk_sum, 0.0f);
      tile_vreduce_add(score, &chunk_sum);
      iq36_component_score_half2_tile score_half2;
      tile_copy_to_half2(score, score_half2);
      tile_store_t_sys_src2(
          score_half2, (__local uint*)current_normalized_score,
          ugemm_vs_sg_tile_n, ugemm_kq_wg_tile_m / 2,
          (cohort_subgroup * ugemm_kq_sg_tile_m) / 2, 0);

      iq36_component_score_sum_tile block_rescale;
      if (first) {
        tile_fill(block_rescale, 1.0f);
      } else {
        tile_copy(old_running_max, block_rescale);
        tile_binary(
            block_rescale, running_max, iq36_component_rescale);
        tile_binary(
            running_sum, block_rescale,
            iq36_component_binary_mul);
      }
      tile_store_full(
          block_rescale, current_rescale,
          ugemm_kq_wg_tile_n, 0, cohort_subgroup);
      tile_binary(
          running_sum, chunk_sum, iq36_component_binary_add);
      tile_copy(running_max, old_running_max);
      if (last) {
        tile_store_full(
            running_sum, sum_slm, ugemm_kq_wg_tile_n,
            0, cohort_subgroup);
      }
      work_group_named_barrier(
          softmax_vs_barrier, CLK_LOCAL_MEM_FENCE);
      work_group_named_barrier(
          kq_softmax_barrier, CLK_LOCAL_MEM_FENCE);
    } else {
      __local half* current_normalized_score =
          normalized_score_double_slm +
          (block & 1U) *
              IQ36_TRIPLE_NORMALIZED_BUFFER_ELEMENTS;
      __local float* current_rescale =
          rescale_double_slm + (block & 1U) * 256;
      work_group_named_barrier(
          softmax_vs_barrier, CLK_LOCAL_MEM_FENCE);
      if (!first) {
        iq36_component_score_sum_tile block_rescale;
        tile_load_full(
            &block_rescale, current_rescale,
            ugemm_kq_wg_tile_n, 0, cohort_subgroup);
        iq36_component_accumulator_scale_tile accumulator_scale;
        tile_copy(block_rescale, accumulator_scale);
        tile_hbroadcast_mul(&accumulator, accumulator_scale);
      }
      iq36_component_accumulator_tile chunk_accumulator = ugemm_vs(
          value_base + (ulong)key_begin * IQ36_D,
          IQ36_D, current_normalized_score,
          ugemm_kq_wg_tile_m,
          IQ36_D, ugemm_kq_wg_tile_n,
          ugemm_kq_wg_tile_m,
          0, 0, 0, cohort_subgroup, 0, ugemm_slm);
      tile_binary(
          accumulator, chunk_accumulator,
          iq36_component_binary_add);
    }
  }

  if (!kq_cohort && !softmax_cohort) {
    iq36_component_accumulator_scale_tile total_sum;
    iq36_component_accumulator_scale_tile partial_sum;
    tile_fill(total_sum, 0.0f);
    #pragma unroll
    for (uint subgroup_row = 0U;
         subgroup_row < ugemm_kq_sg_per_wg_m;
         ++subgroup_row) {
      tile_load_full(
          &partial_sum, sum_slm, ugemm_kq_wg_tile_n,
          0, subgroup_row);
      tile_binary(
          total_sum, partial_sum, iq36_component_binary_add);
    }
    tile_elementwise(total_sum, native_vrecip);
    tile_hbroadcast_mul(&accumulator, total_sum);
    iq36_component_accumulator_half_tile output_tile;
    tile_copy_reblock(accumulator, &output_tile);
    __local half* output_slm =
        (__local half*)raw_score_double_slm;
    tile_store_full(
        output_tile, output_slm, IQ36_D,
        cohort_subgroup * ugemm_vs_sg_tile_m, 0);
    work_group_named_barrier(
        vs_internal_barrier, CLK_LOCAL_MEM_FENCE);
    #pragma unroll
    for (uint head = 0U; head < IQ36_GQA_GROUP; ++head) {
      output[
          ((ulong)kv_head * IQ36_GQA_GROUP + head) * IQ36_D +
          cohort_linear_local_id] =
          output_slm[
              cohort_linear_local_id + head * IQ36_D];
    }
  }
}
#endif

#if IQ36_COMPONENT_PROGRAM == IQ36_COMPONENT_DUAL_COHORT
#define IQ36_DUAL_PRODUCER_SUBGROUPS 16
#define IQ36_DUAL_CONSUMER_SUBGROUPS 16
#define IQ36_DUAL_TOTAL_SUBGROUPS 32
#define IQ36_DUAL_RAW_BUFFER_ELEMENTS \
  (ugemm_kq_wg_tile_m * ugemm_kq_wg_tile_n)

__attribute__((intel_reqd_sub_group_size(IQ36_SUBGROUP)))
__attribute__((reqd_work_group_size(16, 32, 1)))
__kernel void iq36_exact_score_dual_cohort(
    const __global half* query,
    const __global half* key,
    const __global half* value,
    __global half* output) {
  const uint subgroup = sub_group_broadcast(get_local_id(1), 0);
  const bool producer =
      subgroup < (uint)IQ36_DUAL_PRODUCER_SUBGROUPS;
  const uint cohort_subgroup = producer
      ? subgroup : subgroup - (uint)IQ36_DUAL_PRODUCER_SUBGROUPS;
  const uint cohort_linear_local_id =
      (uint)get_local_id(0) + IQ36_SUBGROUP * cohort_subgroup;
  const uint kv_head = (uint)get_group_id(1);
  const __global half* query_head_base =
      query + (ulong)kv_head * IQ36_GQA_GROUP * IQ36_D;
  const __global half* key_base =
      key + (ulong)kv_head * IQ36_CONTEXT * IQ36_D;
  const __global half* value_base =
      value + (ulong)kv_head * IQ36_CONTEXT * IQ36_D;

  __local half query_slm[IQ36_D * ugemm_kq_wg_tile_n];
  __local float raw_score_double_slm[
      2 * IQ36_DUAL_RAW_BUFFER_ELEMENTS];
  __local half score_slm[
      ugemm_kq_wg_tile_m * ugemm_kq_wg_tile_n];
  __local float sum_slm[
      ugemm_kq_wg_tile_n * ugemm_kq_sg_per_wg_m];
  __local float max_and_guard_slm[256];
  __local half output_slm[IQ36_D * ugemm_vs_wg_tile_n];
  __local char ugemm_slm[1];
  __local float* max_slm = max_and_guard_slm;
  __local NamedBarrier_t* consumer_barrier =
      named_barrier_init(IQ36_DUAL_CONSUMER_SUBGROUPS);
  __local NamedBarrier_t* pipeline_barrier =
      named_barrier_init(IQ36_DUAL_TOTAL_SUBGROUPS);

  iq36_component_accumulator_tile accumulator;
  iq36_component_score_sum_tile running_sum;
  iq36_component_score_sum_tile running_max;
  iq36_component_score_sum_tile old_running_max;
  if (producer) {
    iq36_component_load_query(
        query_head_base, query_slm, cohort_subgroup);
  } else {
    iq36_component_initialize_max(max_slm, cohort_subgroup);
    tile_fill(accumulator, 0.0f);
    tile_fill(running_sum, 0.0f);
    tile_fill(running_max, -INFINITY);
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  if (producer) {
    cooperative_prefetch_2d_rem(
        key_base, IQ36_D, IQ36_CONTEXT,
        ugemm_kq_wg_tile_m, 64, IQ36_D,
        cohort_subgroup, IQ36_DUAL_PRODUCER_SUBGROUPS,
        IQ36_SUBGROUP, LSC_LDCC_L1C_L3C);
    iq36_component_score_tile first_score = ugemm_kq(
        key_base, IQ36_D, query_slm, IQ36_D, IQ36_CONTEXT,
        ugemm_kq_wg_tile_n, IQ36_D, 0, 0, 0,
        cohort_subgroup, 0, ugemm_slm);
    tile_store_full(
        first_score, raw_score_double_slm, ugemm_kq_wg_tile_m,
        cohort_subgroup * ugemm_kq_sg_tile_m, 0);
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  const uint block_count = IQ36_CONTEXT / ugemm_kq_wg_tile_m;
  for (uint block = 0U; block < block_count; ++block) {
    const uint key_begin = block * ugemm_kq_wg_tile_m;
    const bool first = block == 0U;
    const bool last = block + 1U == block_count;
    if (producer) {
      const uint next_block = block + 1U;
      if (next_block < block_count) {
        __local float* next_raw_score =
            raw_score_double_slm +
            (next_block & 1U) * IQ36_DUAL_RAW_BUFFER_ELEMENTS;
        cooperative_prefetch_2d_rem(
            key_base, IQ36_D, IQ36_CONTEXT,
            ugemm_kq_wg_tile_m, 64, IQ36_D,
            cohort_subgroup, IQ36_DUAL_PRODUCER_SUBGROUPS,
            IQ36_SUBGROUP, LSC_LDCC_L1C_L3C);
        iq36_component_score_tile next_score = ugemm_kq(
            key_base, IQ36_D, query_slm, IQ36_D, IQ36_CONTEXT,
            ugemm_kq_wg_tile_n, IQ36_D,
            (int)(next_block * ugemm_kq_wg_tile_m), 0, 0,
            cohort_subgroup, 0, ugemm_slm);
        tile_store_full(
            next_score, next_raw_score, ugemm_kq_wg_tile_m,
            cohort_subgroup * ugemm_kq_sg_tile_m, 0);
      }
    } else {
      __local float* current_raw_score =
          raw_score_double_slm +
          (block & 1U) * IQ36_DUAL_RAW_BUFFER_ELEMENTS;
      iq36_component_score_tile score;
      tile_load_full(
          &score, current_raw_score, ugemm_kq_wg_tile_m,
          cohort_subgroup * ugemm_kq_sg_tile_m, 0);
      tile_vreduce_max(score, &running_max);
      max_slm[
          cohort_subgroup * IQ36_SUBGROUP +
          (uint)get_sub_group_local_id()] =
          tile_access(
              running_max, 0, 0, IQ36_SUBGROUP,
              ugemm_kq_sg_tile_n, 1, 1);
      work_group_named_barrier(
          consumer_barrier, CLK_LOCAL_MEM_FENCE);
      float reduced_running_max = -INFINITY;
      #pragma unroll
      for (uint subgroup_row = 0U;
           subgroup_row < IQ36_DUAL_CONSUMER_SUBGROUPS;
           ++subgroup_row) {
        reduced_running_max = max(
            reduced_running_max,
            max_slm[
                subgroup_row * IQ36_SUBGROUP +
                (uint)get_sub_group_local_id()]);
      }
      tile_access(
          running_max, 0, 0, IQ36_SUBGROUP,
          ugemm_kq_sg_tile_n, 1, 1) =
          reduced_running_max;
      tile_vbroadcast_sub(&score, running_max);
      tile_elementwise(score, iq36_component_scaled_exp);
      iq36_component_score_sum_tile chunk_sum;
      tile_fill(chunk_sum, 0.0f);
      tile_vreduce_add(score, &chunk_sum);
      iq36_component_score_half2_tile score_half2;
      tile_copy_to_half2(score, score_half2);
      tile_store_t_sys_src2(
          score_half2, (__local uint*)score_slm,
          ugemm_vs_sg_tile_n, ugemm_kq_wg_tile_m / 2,
          (cohort_subgroup * ugemm_kq_sg_tile_m) / 2, 0);
      if (!first) {
        tile_binary(
            old_running_max, running_max, iq36_component_rescale);
        tile_binary(
            running_sum, old_running_max, iq36_component_binary_mul);
        iq36_component_accumulator_scale_tile accumulator_scale;
        tile_copy(old_running_max, accumulator_scale);
        tile_hbroadcast_mul(&accumulator, accumulator_scale);
      }
      tile_binary(running_sum, chunk_sum, iq36_component_binary_add);
      tile_copy(running_max, old_running_max);
      if (last) {
        tile_store_full(
            running_sum, sum_slm, ugemm_kq_wg_tile_n,
            0, cohort_subgroup);
      }
      work_group_named_barrier(
          consumer_barrier, CLK_LOCAL_MEM_FENCE);
      iq36_component_accumulator_tile chunk_accumulator = ugemm_vs(
          value_base + (ulong)key_begin * IQ36_D,
          IQ36_D, score_slm, ugemm_kq_wg_tile_m,
          IQ36_D, ugemm_kq_wg_tile_n, ugemm_kq_wg_tile_m,
          0, 0, 0, cohort_subgroup, 0, ugemm_slm);
      tile_binary(
          accumulator, chunk_accumulator, iq36_component_binary_add);
    }
    work_group_named_barrier(
        pipeline_barrier, CLK_LOCAL_MEM_FENCE);
  }

  if (!producer) {
    iq36_component_accumulator_scale_tile total_sum;
    iq36_component_accumulator_scale_tile partial_sum;
    tile_fill(total_sum, 0.0f);
    #pragma unroll
    for (uint subgroup_row = 0U;
         subgroup_row < ugemm_kq_sg_per_wg_m; ++subgroup_row) {
      tile_load_full(
          &partial_sum, sum_slm, ugemm_kq_wg_tile_n,
          0, subgroup_row);
      tile_binary(total_sum, partial_sum, iq36_component_binary_add);
    }
    tile_elementwise(total_sum, native_vrecip);
    tile_hbroadcast_mul(&accumulator, total_sum);
    iq36_component_accumulator_half_tile output_tile;
    tile_copy_reblock(accumulator, &output_tile);
    tile_store_full(
        output_tile, output_slm, IQ36_D,
        cohort_subgroup * ugemm_vs_sg_tile_m, 0);
    work_group_named_barrier(
        consumer_barrier, CLK_LOCAL_MEM_FENCE);
    #pragma unroll
    for (uint head = 0U; head < IQ36_GQA_GROUP; ++head) {
      output[
          ((ulong)kv_head * IQ36_GQA_GROUP + head) * IQ36_D +
          cohort_linear_local_id] =
          output_slm[cohort_linear_local_id + head * IQ36_D];
    }
  }
}
#endif

#if IQ36_COMPONENT_PROGRAM == IQ36_COMPONENT_CAPTURE
__kernel void iq36_exact_score_init_query(__global half* query) {
  const uint index = (uint)get_global_id(0);
  if (index >= IQ36_Q_HEADS * IQ36_D) return;
  const int centered = (int)((index * 29U + 17U) & 255U) - 128;
  query[index] = convert_half_rte((float)centered / 1024.0f);
}

__kernel void iq36_exact_score_init_history(
    __global half* key,
    __global half* value) {
  const ulong index = (ulong)get_global_id(0);
  const ulong count =
      (ulong)IQ36_KV_HEADS * IQ36_CONTEXT * IQ36_D;
  if (index >= count) return;
  const uint dimension = (uint)(index % IQ36_D);
  const uint token = (uint)((index / IQ36_D) % IQ36_CONTEXT);
  const uint kv_head = (uint)(index / ((ulong)IQ36_CONTEXT * IQ36_D));
  const int key_centered =
      (int)((token * 13U + dimension * 7U + kv_head * 31U + 5U) & 255U)
      - 128;
  const int value_centered =
      (int)((token * 3U + dimension * 19U + kv_head * 23U + 9U) & 255U)
      - 128;
  key[index] = convert_half_rte((float)key_centered / 2048.0f);
  value[index] = convert_half_rte(
      0.02f + (float)value_centered / 4096.0f);
}
#endif

inline void iq36_component_load_real_scores(
    const __global float* raw_score,
    uint kv_head,
    uint key_begin,
    uint subgroup,
    iq36_component_score_tile* score) {
  const uint lane = (uint)get_sub_group_local_id();
  const uint row =
      key_begin + subgroup * ugemm_kq_sg_tile_m + lane;
  #pragma unroll
  for (uint column = 0U; column < IQ36_KQ_COLUMNS; ++column) {
    const float value = raw_score[
        ((ulong)kv_head * IQ36_KQ_COLUMNS + column) * IQ36_CONTEXT +
        row];
    tile_access(
        *score, 0, column, IQ36_SUBGROUP,
        ugemm_kq_c_type_block0, ugemm_kq_c_type_block1,
        ugemm_kq_c_type_nblock0) = value;
  }
}

inline void iq36_component_load_query(
    const __global half* query_head_base,
    __local half* query_slm,
    uint subgroup) {
  iq36_component_query_tile query_tile;
  const uint query_copy = IQ36_Q_TILE_SG_N * subgroup;
  const uint query_column = subgroup & 7U;
  tile_load_block(
      &query_tile, (const __global uint*)query_head_base, 8, IQ36_D / 2,
      0, query_column);
  tile_store_t_sys_src1(
      query_tile, (__local uint*)query_slm, IQ36_D / 2, query_copy, 0);
}

#if IQ36_COMPONENT_PROGRAM == IQ36_COMPONENT_CAPTURE
__attribute__((intel_reqd_sub_group_size(IQ36_SUBGROUP)))
__attribute__((reqd_work_group_size(16, 16, 1)))
__kernel void iq36_exact_score_serial_capture(
    const __global half* query,
    const __global half* key,
    __global float* raw_score) {
  const uint subgroup = sub_group_broadcast(get_local_id(1), 0);
  const uint kv_head = (uint)get_group_id(1);
  const __global half* query_head_base =
      query + (ulong)kv_head * IQ36_GQA_GROUP * IQ36_D;
  const __global half* key_base =
      key + (ulong)kv_head * IQ36_CONTEXT * IQ36_D;
  __local half query_slm[IQ36_D * ugemm_kq_wg_tile_n];
  __local char ugemm_slm[1];
  iq36_component_load_query(query_head_base, query_slm, subgroup);
  barrier(CLK_LOCAL_MEM_FENCE);
  const uint sg_i = subgroup % ugemm_kq_sg_per_wg_m;
  const uint sg_j = subgroup / ugemm_kq_sg_per_wg_m;
  for (uint key_begin = 0U; key_begin < IQ36_CONTEXT;
       key_begin += ugemm_kq_wg_tile_m) {
    iq36_component_score_tile score = ugemm_kq(
        key_base, IQ36_D, query_slm, IQ36_D, IQ36_CONTEXT,
        ugemm_kq_wg_tile_n, IQ36_D, (int)key_begin, 0, 0,
        sg_i, sg_j, ugemm_slm);
    const uint row =
        key_begin + sg_i * ugemm_kq_sg_tile_m +
        (uint)get_sub_group_local_id();
    #pragma unroll
    for (uint column = 0U; column < IQ36_KQ_COLUMNS; ++column) {
      raw_score[
          ((ulong)kv_head * IQ36_KQ_COLUMNS + column) * IQ36_CONTEXT +
          row] =
          tile_access(
              score, 0, column, IQ36_SUBGROUP,
              ugemm_kq_c_type_block0, ugemm_kq_c_type_block1,
              ugemm_kq_c_type_nblock0);
    }
  }
}
#endif

#if IQ36_COMPONENT_PROGRAM == IQ36_COMPONENT_STAGED
__attribute__((intel_reqd_sub_group_size(IQ36_SUBGROUP)))
__attribute__((reqd_work_group_size(16, 16, 1)))
__kernel void iq36_exact_score_kq_stage(
    const __global half* query,
    const __global half* key,
    __global float* raw_score) {
  const uint subgroup = sub_group_broadcast(get_local_id(1), 0);
  const uint group = (uint)get_group_id(1);
  const uint block_count = IQ36_CONTEXT / ugemm_kq_wg_tile_m;
  const uint kv_head = group / block_count;
  const uint block = group - kv_head * block_count;
  const uint key_begin = block * ugemm_kq_wg_tile_m;
  const __global half* query_head_base =
      query + (ulong)kv_head * IQ36_GQA_GROUP * IQ36_D;
  const __global half* key_base =
      key + (ulong)kv_head * IQ36_CONTEXT * IQ36_D;
  __local half query_slm[IQ36_D * ugemm_kq_wg_tile_n];
  __local char ugemm_slm[1];
  iq36_component_load_query(query_head_base, query_slm, subgroup);
  barrier(CLK_LOCAL_MEM_FENCE);
  const uint sg_i = subgroup % ugemm_kq_sg_per_wg_m;
  const uint sg_j = subgroup / ugemm_kq_sg_per_wg_m;
  cooperative_prefetch_2d_rem(
      key_base + (ulong)key_begin * IQ36_D,
      IQ36_D, ugemm_kq_wg_tile_m,
      ugemm_kq_wg_tile_m, 64, IQ36_D,
      subgroup, IQ36_SG_PER_WG, IQ36_SUBGROUP, LSC_LDCC_L1C_L3C);
  iq36_component_score_tile score = ugemm_kq(
      key_base, IQ36_D, query_slm, IQ36_D, IQ36_CONTEXT,
      ugemm_kq_wg_tile_n, IQ36_D, (int)key_begin, 0, 0,
      sg_i, sg_j, ugemm_slm);
  const uint row =
      key_begin + sg_i * ugemm_kq_sg_tile_m +
      (uint)get_sub_group_local_id();
  #pragma unroll
  for (uint column = 0U; column < IQ36_KQ_COLUMNS; ++column) {
    raw_score[
        ((ulong)kv_head * IQ36_KQ_COLUMNS + column) * IQ36_CONTEXT +
        row] =
        tile_access(
            score, 0, column, IQ36_SUBGROUP,
            ugemm_kq_c_type_block0, ugemm_kq_c_type_block1,
            ugemm_kq_c_type_nblock0);
  }
}
#endif

inline void iq36_component_initialize_max(
    __local float* max_slm,
    uint subgroup) {
  const uint max_columns_per_subgroup =
      IQ36_DIV_UP(
          ugemm_kq_wg_tile_n,
          IQ36_SUBGROUP * IQ36_SG_PER_WG);
  #pragma unroll
  for (uint column = 0; column < max_columns_per_subgroup; ++column) {
    intel_sub_group_block_write(
        (__local uint*)&max_slm[
            (column + subgroup * max_columns_per_subgroup) *
            IQ36_SUBGROUP],
        as_uint(-INFINITY));
  }
}

inline void iq36_component_store_output(
    iq36_component_accumulator_tile accumulator,
    iq36_component_accumulator_scale_tile total_sum,
    __local half* output_slm,
    __global half* output,
    uint kv_head,
    uint linear_local_id,
    uint sg_i_vs) {
  tile_elementwise(total_sum, native_vrecip);
  tile_hbroadcast_mul(&accumulator, total_sum);
  iq36_component_accumulator_half_tile output_tile;
  tile_copy_reblock(accumulator, &output_tile);
  const uint output_row = sg_i_vs * ugemm_vs_sg_tile_m;
  tile_store_full(
      output_tile, output_slm, IQ36_D, output_row, 0);
  barrier(CLK_LOCAL_MEM_FENCE);
  #pragma unroll
  for (uint head = 0U; head < IQ36_GQA_GROUP; ++head) {
    output[
        ((ulong)kv_head * IQ36_GQA_GROUP + head) * IQ36_D +
        linear_local_id] =
        output_slm[linear_local_id + head * IQ36_D];
  }
}

#if IQ36_COMPONENT_PROGRAM == IQ36_COMPONENT_SOFTMAX_STAGE
// Standalone middle-stage ruler for a possible three-cohort pipeline.  It
// preserves the accepted chronological F32 recurrence, writes the exact F16
// score-SLM image consumed by generated VS, and records the per-block
// accumulator rescale plus the final per-row denominator.  The global
// intermediates deliberately make this a conservative stage bound; a product
// pipeline would hand the same bits across double-buffered SLM.
__attribute__((intel_reqd_sub_group_size(IQ36_SUBGROUP)))
__attribute__((reqd_work_group_size(16, 16, 1)))
__kernel void iq36_exact_score_softmax_stage(
    const __global float* raw_score,
    __global half* normalized_score,
    __global float* accumulator_rescale,
    __global float* final_sum) {
  const uint subgroup = sub_group_broadcast(get_local_id(1), 0);
  const uint linear_local_id =
      (uint)get_local_id(0) + IQ36_SUBGROUP * subgroup;
  const uint kv_head = (uint)get_group_id(1);
  const uint block_count = IQ36_CONTEXT / ugemm_kq_wg_tile_m;
  __local half score_slm[
      ugemm_kq_wg_tile_m * ugemm_kq_wg_tile_n];
  __local float rescale_slm[
      ugemm_kq_wg_tile_n * ugemm_kq_sg_per_wg_m];
  __local float sum_slm[
      ugemm_kq_wg_tile_n * ugemm_kq_sg_per_wg_m];
  __local float max_slm[256];
  __local NamedBarrier_t* softmax_barrier =
      named_barrier_init(IQ36_SG_PER_WG);

  iq36_component_initialize_max(max_slm, subgroup);
  iq36_component_score_sum_tile running_sum;
  iq36_component_score_sum_tile running_max;
  iq36_component_score_sum_tile old_running_max;
  tile_fill(running_sum, 0.0f);
  tile_fill(running_max, -INFINITY);
  barrier(CLK_LOCAL_MEM_FENCE);

  for (uint block = 0U; block < block_count; ++block) {
    const uint key_begin = block * ugemm_kq_wg_tile_m;
    const bool first = block == 0U;
    const bool last = block + 1U == block_count;
    iq36_component_score_tile score;
    iq36_component_load_real_scores(
        raw_score, kv_head, key_begin, subgroup, &score);
    tile_vreduce_max(score, &running_max);
    max_slm[
        subgroup * IQ36_SUBGROUP +
        (uint)get_sub_group_local_id()] =
        tile_access(
            running_max, 0, 0, IQ36_SUBGROUP,
            ugemm_kq_sg_tile_n, 1, 1);
    work_group_named_barrier(
        softmax_barrier, CLK_LOCAL_MEM_FENCE);
    float reduced_running_max = -INFINITY;
    #pragma unroll
    for (uint subgroup_row = 0U;
         subgroup_row < IQ36_SG_PER_WG; ++subgroup_row) {
      reduced_running_max = max(
          reduced_running_max,
          max_slm[
              subgroup_row * IQ36_SUBGROUP +
              (uint)get_sub_group_local_id()]);
    }
    tile_access(
        running_max, 0, 0, IQ36_SUBGROUP,
        ugemm_kq_sg_tile_n, 1, 1) =
        reduced_running_max;
    tile_vbroadcast_sub(&score, running_max);
    tile_elementwise(score, iq36_component_scaled_exp);
    iq36_component_score_sum_tile chunk_sum;
    tile_fill(chunk_sum, 0.0f);
    tile_vreduce_add(score, &chunk_sum);
    iq36_component_score_half2_tile score_half2;
    tile_copy_to_half2(score, score_half2);
    tile_store_t_sys_src2(
        score_half2, (__local uint*)score_slm,
        ugemm_vs_sg_tile_n, ugemm_kq_wg_tile_m / 2,
        (subgroup * ugemm_kq_sg_tile_m) / 2, 0);

    iq36_component_score_sum_tile block_rescale;
    if (first) {
      tile_fill(block_rescale, 1.0f);
    } else {
      tile_copy(old_running_max, block_rescale);
      tile_binary(
          block_rescale, running_max, iq36_component_rescale);
      tile_binary(
          running_sum, block_rescale, iq36_component_binary_mul);
    }
    tile_store_full(
        block_rescale, rescale_slm, ugemm_kq_wg_tile_n,
        0, subgroup);
    tile_binary(running_sum, chunk_sum, iq36_component_binary_add);
    tile_copy(running_max, old_running_max);
    if (last) {
      tile_store_full(
          running_sum, sum_slm, ugemm_kq_wg_tile_n,
          0, subgroup);
    }
    work_group_named_barrier(
        softmax_barrier, CLK_LOCAL_MEM_FENCE);

    const ulong score_block_base =
        ((ulong)kv_head * block_count + block) *
        ugemm_kq_wg_tile_m * ugemm_kq_wg_tile_n;
    for (uint index = linear_local_id;
         index < ugemm_kq_wg_tile_m * ugemm_kq_wg_tile_n;
         index += IQ36_SUBGROUP * IQ36_SG_PER_WG) {
      normalized_score[score_block_base + index] = score_slm[index];
    }
    const ulong state_block_base =
        ((ulong)kv_head * block_count + block) *
        ugemm_kq_wg_tile_n * ugemm_kq_sg_per_wg_m;
    accumulator_rescale[state_block_base + linear_local_id] =
        rescale_slm[linear_local_id];
    if (last) {
      final_sum[
          (ulong)kv_head * ugemm_kq_wg_tile_n *
              ugemm_kq_sg_per_wg_m +
          linear_local_id] = sum_slm[linear_local_id];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
}
#endif

#if IQ36_COMPONENT_PROGRAM == IQ36_COMPONENT_SOFTMAX_TRAFFIC
// Matched raw-score read, F32->F16 score image, metadata write, and barrier
// control for the exact middle stage.  Exact-minus-control timing isolates
// the max/exp/sum/rescale arithmetic from conservative global staging.
__attribute__((intel_reqd_sub_group_size(IQ36_SUBGROUP)))
__attribute__((reqd_work_group_size(16, 16, 1)))
__kernel void iq36_exact_score_softmax_traffic(
    const __global float* raw_score,
    __global half* normalized_score,
    __global float* accumulator_rescale,
    __global float* final_sum) {
  const uint subgroup = sub_group_broadcast(get_local_id(1), 0);
  const uint linear_local_id =
      (uint)get_local_id(0) + IQ36_SUBGROUP * subgroup;
  const uint kv_head = (uint)get_group_id(1);
  const uint block_count = IQ36_CONTEXT / ugemm_kq_wg_tile_m;
  __local half score_slm[
      ugemm_kq_wg_tile_m * ugemm_kq_wg_tile_n];
  __local float rescale_slm[
      ugemm_kq_wg_tile_n * ugemm_kq_sg_per_wg_m];
  __local float sum_slm[
      ugemm_kq_wg_tile_n * ugemm_kq_sg_per_wg_m];
  __local float max_slm[256];
  __local NamedBarrier_t* traffic_barrier =
      named_barrier_init(IQ36_SG_PER_WG);

  iq36_component_initialize_max(max_slm, subgroup);
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint block = 0U; block < block_count; ++block) {
    const uint key_begin = block * ugemm_kq_wg_tile_m;
    const bool last = block + 1U == block_count;
    iq36_component_score_tile score;
    iq36_component_load_real_scores(
        raw_score, kv_head, key_begin, subgroup, &score);
    work_group_named_barrier(
        traffic_barrier, CLK_LOCAL_MEM_FENCE);
    iq36_component_score_half2_tile score_half2;
    tile_copy_to_half2(score, score_half2);
    tile_store_t_sys_src2(
        score_half2, (__local uint*)score_slm,
        ugemm_vs_sg_tile_n, ugemm_kq_wg_tile_m / 2,
        (subgroup * ugemm_kq_sg_tile_m) / 2, 0);
    iq36_component_score_sum_tile unit_rescale;
    tile_fill(unit_rescale, 1.0f);
    tile_store_full(
        unit_rescale, rescale_slm, ugemm_kq_wg_tile_n,
        0, subgroup);
    if (last) {
      iq36_component_score_sum_tile zero_sum;
      tile_fill(zero_sum, 0.0f);
      tile_store_full(
          zero_sum, sum_slm, ugemm_kq_wg_tile_n,
          0, subgroup);
    }
    work_group_named_barrier(
        traffic_barrier, CLK_LOCAL_MEM_FENCE);

    const ulong score_block_base =
        ((ulong)kv_head * block_count + block) *
        ugemm_kq_wg_tile_m * ugemm_kq_wg_tile_n;
    for (uint index = linear_local_id;
         index < ugemm_kq_wg_tile_m * ugemm_kq_wg_tile_n;
         index += IQ36_SUBGROUP * IQ36_SG_PER_WG) {
      normalized_score[score_block_base + index] = score_slm[index];
    }
    const ulong state_block_base =
        ((ulong)kv_head * block_count + block) *
        ugemm_kq_wg_tile_n * ugemm_kq_sg_per_wg_m;
    accumulator_rescale[state_block_base + linear_local_id] =
        rescale_slm[linear_local_id];
    if (last) {
      final_sum[
          (ulong)kv_head * ugemm_kq_wg_tile_n *
              ugemm_kq_sg_per_wg_m +
          linear_local_id] = sum_slm[linear_local_id];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
}
#endif

#if IQ36_COMPONENT_PROGRAM == IQ36_COMPONENT_VS_STAGE
// Standalone generated-VS ruler.  It consumes the exact softmax-stage SLM
// images and replays the accepted per-block accumulator rescale/order.
__attribute__((intel_reqd_sub_group_size(IQ36_SUBGROUP)))
__attribute__((reqd_work_group_size(16, 16, 1)))
__kernel void iq36_exact_score_vs_stage(
    const __global half* normalized_score,
    const __global float* accumulator_rescale,
    const __global float* final_sum,
    const __global half* value,
    __global half* output) {
  const uint subgroup = sub_group_broadcast(get_local_id(1), 0);
  const uint linear_local_id =
      (uint)get_local_id(0) + IQ36_SUBGROUP * subgroup;
  const uint kv_head = (uint)get_group_id(1);
  const uint block_count = IQ36_CONTEXT / ugemm_kq_wg_tile_m;
  const __global half* value_base =
      value + (ulong)kv_head * IQ36_CONTEXT * IQ36_D;
  __local half score_slm[
      ugemm_kq_wg_tile_m * ugemm_kq_wg_tile_n];
  __local float rescale_slm[
      ugemm_kq_wg_tile_n * ugemm_kq_sg_per_wg_m];
  __local float sum_slm[
      ugemm_kq_wg_tile_n * ugemm_kq_sg_per_wg_m];
  __local half output_slm[IQ36_D * ugemm_vs_wg_tile_n];
  __local char ugemm_slm[1];

  iq36_component_accumulator_tile accumulator;
  tile_fill(accumulator, 0.0f);
  for (uint block = 0U; block < block_count; ++block) {
    const uint key_begin = block * ugemm_kq_wg_tile_m;
    const ulong score_block_base =
        ((ulong)kv_head * block_count + block) *
        ugemm_kq_wg_tile_m * ugemm_kq_wg_tile_n;
    for (uint index = linear_local_id;
         index < ugemm_kq_wg_tile_m * ugemm_kq_wg_tile_n;
         index += IQ36_SUBGROUP * IQ36_SG_PER_WG) {
      score_slm[index] = normalized_score[score_block_base + index];
    }
    const ulong state_block_base =
        ((ulong)kv_head * block_count + block) *
        ugemm_kq_wg_tile_n * ugemm_kq_sg_per_wg_m;
    rescale_slm[linear_local_id] =
        accumulator_rescale[state_block_base + linear_local_id];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (block != 0U) {
      iq36_component_score_sum_tile block_rescale;
      tile_load_full(
          &block_rescale, rescale_slm, ugemm_kq_wg_tile_n,
          0, subgroup);
      iq36_component_accumulator_scale_tile accumulator_scale;
      tile_copy(block_rescale, accumulator_scale);
      tile_hbroadcast_mul(&accumulator, accumulator_scale);
    }
    iq36_component_accumulator_tile chunk_accumulator = ugemm_vs(
        value_base + (ulong)key_begin * IQ36_D,
        IQ36_D, score_slm, ugemm_kq_wg_tile_m,
        IQ36_D, ugemm_kq_wg_tile_n, ugemm_kq_wg_tile_m,
        0, 0, 0, subgroup, 0, ugemm_slm);
    tile_binary(
        accumulator, chunk_accumulator, iq36_component_binary_add);
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  sum_slm[linear_local_id] =
      final_sum[
          (ulong)kv_head * ugemm_kq_wg_tile_n *
              ugemm_kq_sg_per_wg_m +
          linear_local_id];
  barrier(CLK_LOCAL_MEM_FENCE);
  iq36_component_accumulator_scale_tile total_sum;
  iq36_component_accumulator_scale_tile partial_sum;
  tile_fill(total_sum, 0.0f);
  #pragma unroll
  for (uint subgroup_row = 0U;
       subgroup_row < ugemm_kq_sg_per_wg_m; ++subgroup_row) {
    tile_load_full(
        &partial_sum, sum_slm, ugemm_kq_wg_tile_n,
        0, subgroup_row);
    tile_binary(total_sum, partial_sum, iq36_component_binary_add);
  }
  iq36_component_store_output(
      accumulator, total_sum, output_slm, output,
      kv_head, linear_local_id, subgroup);
}
#endif

#if IQ36_COMPONENT_PROGRAM == IQ36_COMPONENT_FUSED
__attribute__((intel_reqd_sub_group_size(IQ36_SUBGROUP)))
__attribute__((reqd_work_group_size(16, 16, 1)))
__kernel void iq36_exact_score_fused(
    const __global half* query,
    const __global half* key,
    const __global half* value,
    __global half* output) {
  const uint subgroup = sub_group_broadcast(get_local_id(1), 0);
  const uint linear_local_id =
      (uint)get_local_id(0) + IQ36_SUBGROUP * (uint)get_local_id(1);
  const uint kv_head = (uint)get_group_id(1);
  const __global half* query_head_base =
      query + (ulong)kv_head * IQ36_GQA_GROUP * IQ36_D;
  const __global half* key_base =
      key + (ulong)kv_head * IQ36_CONTEXT * IQ36_D;
  const __global half* value_base =
      value + (ulong)kv_head * IQ36_CONTEXT * IQ36_D;
  const uint sg_i_kq = subgroup % ugemm_kq_sg_per_wg_m;
  const uint sg_j_kq = subgroup / ugemm_kq_sg_per_wg_m;
  const uint sg_i_vs = subgroup % ugemm_vs_sg_per_wg_m;
  const uint sg_j_vs = subgroup / ugemm_vs_sg_per_wg_m;
  __local half query_slm[IQ36_D * ugemm_kq_wg_tile_n];
  __local half score_slm[
      ugemm_kq_wg_tile_m * ugemm_kq_wg_tile_n];
  __local float sum_slm[
      ugemm_kq_wg_tile_n * ugemm_kq_sg_per_wg_m];
  __local float max_and_guard_slm[256];
  __local half output_slm[IQ36_D * ugemm_vs_wg_tile_n];
  __local char ugemm_slm[1];
  __local float* max_slm = max_and_guard_slm;
  iq36_component_load_query(query_head_base, query_slm, subgroup);
  iq36_component_initialize_max(max_slm, subgroup);
  iq36_component_accumulator_tile accumulator;
  tile_fill(accumulator, 0.0f);
  iq36_component_score_sum_tile running_sum;
  iq36_component_score_sum_tile running_max;
  iq36_component_score_sum_tile old_running_max;
  tile_fill(running_sum, 0.0f);
  tile_fill(running_max, -INFINITY);
  barrier(CLK_LOCAL_MEM_FENCE);

  for (uint key_begin = 0U; key_begin < IQ36_CONTEXT;
       key_begin += ugemm_kq_wg_tile_m) {
    const bool first = key_begin == 0U;
    const bool last =
        key_begin + ugemm_kq_wg_tile_m >= IQ36_CONTEXT;
    cooperative_prefetch_2d_rem(
        key_base, IQ36_D, IQ36_CONTEXT,
        ugemm_kq_wg_tile_m, 64, IQ36_D,
        subgroup, IQ36_SG_PER_WG, IQ36_SUBGROUP, LSC_LDCC_L1C_L3C);
    iq36_component_score_tile score = ugemm_kq(
        key_base, IQ36_D, query_slm, IQ36_D, IQ36_CONTEXT,
        ugemm_kq_wg_tile_n, IQ36_D, (int)key_begin, 0, 0,
        sg_i_kq, sg_j_kq, ugemm_slm);
    tile_vreduce_max(score, &running_max);
    tile_atomic_max_full(
        running_max, max_slm, ugemm_kq_wg_tile_n, 0, 0);
    intel_work_group_barrier_arrive(CLK_LOCAL_MEM_FENCE);
    cooperative_prefetch_2d_rem(
        value_base + (ulong)key_begin * IQ36_D,
        IQ36_D, IQ36_CONTEXT - key_begin,
        64, ugemm_kq_wg_tile_m, IQ36_D,
        subgroup, IQ36_SG_PER_WG, IQ36_SUBGROUP, LSC_LDCC_L1C_L3C);
    intel_work_group_barrier_wait(CLK_LOCAL_MEM_FENCE);
    tile_load_full(
        &running_max, max_slm, ugemm_kq_wg_tile_n, 0, 0);
    tile_vbroadcast_sub(&score, running_max);
    tile_elementwise(score, iq36_component_scaled_exp);
    iq36_component_score_sum_tile chunk_sum;
    tile_fill(chunk_sum, 0.0f);
    tile_vreduce_add(score, &chunk_sum);
    iq36_component_score_half2_tile score_half2;
    tile_copy_to_half2(score, score_half2);
    tile_store_t_sys_src2(
        score_half2, (__local uint*)score_slm,
        ugemm_vs_sg_tile_n, ugemm_kq_wg_tile_m / 2,
        (sg_i_kq * ugemm_kq_sg_tile_m) / 2, 0);
    intel_work_group_barrier_arrive(CLK_LOCAL_MEM_FENCE);
    if (!first) {
      tile_binary(
          old_running_max, running_max, iq36_component_rescale);
      tile_binary(
          running_sum, old_running_max, iq36_component_binary_mul);
      iq36_component_accumulator_scale_tile accumulator_scale;
      tile_copy(old_running_max, accumulator_scale);
      tile_hbroadcast_mul(&accumulator, accumulator_scale);
    }
    tile_binary(running_sum, chunk_sum, iq36_component_binary_add);
    tile_copy(running_max, old_running_max);
    if (last) {
      tile_store_full(
          running_sum, sum_slm, ugemm_kq_wg_tile_n, 0, sg_i_kq);
    }
    intel_work_group_barrier_wait(CLK_LOCAL_MEM_FENCE);
    if (last && ugemm_vs_barrier_count == 0) {
      intel_work_group_barrier_arrive(CLK_LOCAL_MEM_FENCE);
    }
    iq36_component_accumulator_tile chunk_accumulator = ugemm_vs(
        value_base + (ulong)key_begin * IQ36_D,
        IQ36_D, score_slm, ugemm_kq_wg_tile_m,
        IQ36_D, ugemm_kq_wg_tile_n, ugemm_kq_wg_tile_m,
        0, 0, 0, sg_i_vs, sg_j_vs, ugemm_slm);
    tile_binary(
        accumulator, chunk_accumulator, iq36_component_binary_add);
  }
  if (ugemm_vs_barrier_count == 0) {
    intel_work_group_barrier_wait(CLK_LOCAL_MEM_FENCE);
  }
  iq36_component_accumulator_scale_tile total_sum;
  iq36_component_accumulator_scale_tile partial_sum;
  tile_fill(total_sum, 0.0f);
  #pragma unroll
  for (uint subgroup_row = 0U;
       subgroup_row < ugemm_kq_sg_per_wg_m; ++subgroup_row) {
    tile_load_full(
        &partial_sum, sum_slm, ugemm_kq_wg_tile_n,
        ugemm_vs_sg_tile_n * sg_j_vs, subgroup_row);
    tile_binary(total_sum, partial_sum, iq36_component_binary_add);
  }
  iq36_component_store_output(
      accumulator, total_sum, output_slm, output,
      kv_head, linear_local_id, sg_i_vs);
}
#endif

#if IQ36_COMPONENT_PROGRAM == IQ36_COMPONENT_STAGED
__attribute__((intel_reqd_sub_group_size(IQ36_SUBGROUP)))
__attribute__((reqd_work_group_size(16, 16, 1)))
__kernel void iq36_exact_score_owner_stage(
    const __global float* raw_score,
    const __global half* value,
    __global half* output) {
  const uint subgroup = sub_group_broadcast(get_local_id(1), 0);
  const uint linear_local_id =
      (uint)get_local_id(0) + IQ36_SUBGROUP * (uint)get_local_id(1);
  const uint kv_head = (uint)get_group_id(1);
  const __global half* value_base =
      value + (ulong)kv_head * IQ36_CONTEXT * IQ36_D;
  const uint sg_i_kq = subgroup % ugemm_kq_sg_per_wg_m;
  const uint sg_i_vs = subgroup % ugemm_vs_sg_per_wg_m;
  const uint sg_j_vs = subgroup / ugemm_vs_sg_per_wg_m;
  __local half score_slm[
      ugemm_kq_wg_tile_m * ugemm_kq_wg_tile_n];
  __local float sum_slm[
      ugemm_kq_wg_tile_n * ugemm_kq_sg_per_wg_m];
  __local float max_and_guard_slm[256];
  __local half output_slm[IQ36_D * ugemm_vs_wg_tile_n];
  __local char ugemm_slm[1];
  __local float* max_slm = max_and_guard_slm;
  iq36_component_initialize_max(max_slm, subgroup);
  iq36_component_accumulator_tile accumulator;
  tile_fill(accumulator, 0.0f);
  iq36_component_score_sum_tile running_sum;
  iq36_component_score_sum_tile running_max;
  iq36_component_score_sum_tile old_running_max;
  tile_fill(running_sum, 0.0f);
  tile_fill(running_max, -INFINITY);
  barrier(CLK_LOCAL_MEM_FENCE);

  for (uint key_begin = 0U; key_begin < IQ36_CONTEXT;
       key_begin += ugemm_kq_wg_tile_m) {
    const bool first = key_begin == 0U;
    const bool last =
        key_begin + ugemm_kq_wg_tile_m >= IQ36_CONTEXT;
    iq36_component_score_tile score;
    iq36_component_load_real_scores(
        raw_score, kv_head, key_begin, sg_i_kq, &score);
    tile_vreduce_max(score, &running_max);
    tile_atomic_max_full(
        running_max, max_slm, ugemm_kq_wg_tile_n, 0, 0);
    intel_work_group_barrier_arrive(CLK_LOCAL_MEM_FENCE);
    cooperative_prefetch_2d_rem(
        value_base + (ulong)key_begin * IQ36_D,
        IQ36_D, IQ36_CONTEXT - key_begin,
        64, ugemm_kq_wg_tile_m, IQ36_D,
        subgroup, IQ36_SG_PER_WG, IQ36_SUBGROUP, LSC_LDCC_L1C_L3C);
    intel_work_group_barrier_wait(CLK_LOCAL_MEM_FENCE);
    tile_load_full(
        &running_max, max_slm, ugemm_kq_wg_tile_n, 0, 0);
    tile_vbroadcast_sub(&score, running_max);
    tile_elementwise(score, iq36_component_scaled_exp);
    iq36_component_score_sum_tile chunk_sum;
    tile_fill(chunk_sum, 0.0f);
    tile_vreduce_add(score, &chunk_sum);
    iq36_component_score_half2_tile score_half2;
    tile_copy_to_half2(score, score_half2);
    tile_store_t_sys_src2(
        score_half2, (__local uint*)score_slm,
        ugemm_vs_sg_tile_n, ugemm_kq_wg_tile_m / 2,
        (sg_i_kq * ugemm_kq_sg_tile_m) / 2, 0);
    intel_work_group_barrier_arrive(CLK_LOCAL_MEM_FENCE);
    if (!first) {
      tile_binary(
          old_running_max, running_max, iq36_component_rescale);
      tile_binary(
          running_sum, old_running_max, iq36_component_binary_mul);
      iq36_component_accumulator_scale_tile accumulator_scale;
      tile_copy(old_running_max, accumulator_scale);
      tile_hbroadcast_mul(&accumulator, accumulator_scale);
    }
    tile_binary(running_sum, chunk_sum, iq36_component_binary_add);
    tile_copy(running_max, old_running_max);
    if (last) {
      tile_store_full(
          running_sum, sum_slm, ugemm_kq_wg_tile_n, 0, sg_i_kq);
    }
    intel_work_group_barrier_wait(CLK_LOCAL_MEM_FENCE);
    if (last && ugemm_vs_barrier_count == 0) {
      intel_work_group_barrier_arrive(CLK_LOCAL_MEM_FENCE);
    }
    iq36_component_accumulator_tile chunk_accumulator = ugemm_vs(
        value_base + (ulong)key_begin * IQ36_D,
        IQ36_D, score_slm, ugemm_kq_wg_tile_m,
        IQ36_D, ugemm_kq_wg_tile_n, ugemm_kq_wg_tile_m,
        0, 0, 0, sg_i_vs, sg_j_vs, ugemm_slm);
    tile_binary(
        accumulator, chunk_accumulator, iq36_component_binary_add);
  }
  if (ugemm_vs_barrier_count == 0) {
    intel_work_group_barrier_wait(CLK_LOCAL_MEM_FENCE);
  }
  iq36_component_accumulator_scale_tile total_sum;
  iq36_component_accumulator_scale_tile partial_sum;
  tile_fill(total_sum, 0.0f);
  #pragma unroll
  for (uint subgroup_row = 0U;
       subgroup_row < ugemm_kq_sg_per_wg_m; ++subgroup_row) {
    tile_load_full(
        &partial_sum, sum_slm, ugemm_kq_wg_tile_n,
        ugemm_vs_sg_tile_n * sg_j_vs, subgroup_row);
    tile_binary(total_sum, partial_sum, iq36_component_binary_add);
  }
  iq36_component_store_output(
      accumulator, total_sum, output_slm, output,
      kv_head, linear_local_id, sg_i_vs);
}
#endif
