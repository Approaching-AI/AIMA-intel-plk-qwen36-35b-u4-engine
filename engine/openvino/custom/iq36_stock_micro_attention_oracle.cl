// Decode-only oracle for the locked Qwen3.6 full-attention boundary.
//
// The surrounding shim is extracted from the actual stock sdpa_micro JIT.
// Keep this wrapper's work-group geometry and arithmetic order aligned with
// that source; this is an oracle for reuse, not another scalar approximation.

#if !defined(IQ36_STOCK_MICRO_OWNER) || \
    defined(IQ36_BUILD_DECODE_ONLY)

#if defined(IQ36_STOCK_MICRO_OWNER)
#define IQ36_ORACLE_LENGTH_OFFSET INPUT12_OFFSET
#define IQ36_ORACLE_OUTPUT_OFFSET OUTPUT1_OFFSET
#define IQ36_ORACLE_OUTPUT_PITCHES OUTPUT1_PITCHES
#define IQ36_ORACLE_OUTPUT_TYPE OUTPUT1_TYPE
#else
#define IQ36_ORACLE_LENGTH_OFFSET INPUT3_OFFSET
#define IQ36_ORACLE_OUTPUT_OFFSET OUTPUT0_OFFSET
#define IQ36_ORACLE_OUTPUT_PITCHES OUTPUT0_PITCHES
#define IQ36_ORACLE_OUTPUT_TYPE OUTPUT0_TYPE
#endif

#define IQ36_D 256
#define IQ36_SUBGROUP 16
#define SUBGROUP_SIZE IQ36_SUBGROUP
#define IQ36_SCALE_LOG2E 0.09016844f
#define IQ36_DIV_UP(x, y) (((x) + (y) - 1) / (y))
#if defined(IQ36_STOCK_MICRO_PAGE_SPARSE)
#define IQ36_PAGE_SINK_TOKENS 4U
#define IQ36_PAGE_TOKENS 512U
#define IQ36_PAGE_SAMPLES 16U
#define IQ36_PAGE_KEEP 64U
#define IQ36_PAGE_MAX_COUNT 130U
#define IQ36_PAGE_MAX_TILES 520U
#define IQ36_PAGE_TILE_TOKENS 256U
#endif
#ifndef IQ36_STOCK_MICRO_GROUPS_PER_KV
#define IQ36_STOCK_MICRO_GROUPS_PER_KV 1U
#endif
#if IQ36_STOCK_MICRO_GROUPS_PER_KV != 1 && \
    IQ36_STOCK_MICRO_GROUPS_PER_KV != 2 && \
    IQ36_STOCK_MICRO_GROUPS_PER_KV != 4
#error "stock-micro groups per KV head must be one, two, or four"
#endif
#if defined(IQ36_STOCK_MICRO_CONTEXT_PARTITION4) && \
    IQ36_STOCK_MICRO_GROUPS_PER_KV != 4
#error "stock-micro context partition4 requires four groups per KV head"
#endif
#if defined(IQ36_STOCK_MICRO_PAGE_SPARSE) && \
    defined(IQ36_STOCK_MICRO_CONTEXT_PARTITION4)
#error "page-sparse stock-micro owns one work-group per KV head"
#endif
#if defined(IQ36_STOCK_MICRO_DUAL_COHORT) && \
    (!defined(IQ36_STOCK_MICRO_OWNER) || \
     IQ36_STOCK_MICRO_GROUPS_PER_KV != 1 || \
     defined(IQ36_STOCK_MICRO_PAGE_SPARSE) || \
     defined(IQ36_STOCK_MICRO_CONTEXT_PARTITION4))
#error "dual-cohort stock-micro requires the unpartitioned exact state owner"
#endif
#define IQ36_SG_PER_WG (ugemm_kq_sg_per_wg_m * ugemm_kq_sg_per_wg_n)
#define IQ36_Q_TILE_SG_N \
  IQ36_DIV_UP(ugemm_kq_wg_tile_n, IQ36_SG_PER_WG)

typedef ugemm_kq_c_type iq36_score_tile;
typedef ugemm_vs_c_type iq36_accumulator_tile;
DECLARE_2D_TILE(
    iq36_query_tile, uint, IQ36_SUBGROUP, IQ36_D / 2, 1, 1,
    IQ36_Q_TILE_SG_N)
DECLARE_2D_TILE_BLOCK_OPS(
    iq36_query_tile, uint, IQ36_SUBGROUP, IQ36_D / 2, 1, 1,
    IQ36_Q_TILE_SG_N)
DECLARE_2D_TILE(
    iq36_accumulator_half_tile, half, IQ36_SUBGROUP,
    ugemm_vs_sg_tile_m, 8, 1, ugemm_vs_sg_tile_n / 8)
DECLARE_2D_TILE(
    iq36_score_half2_tile, uint, IQ36_SUBGROUP,
    ugemm_kq_c_type_block0, ugemm_kq_c_type_block1 / 2,
    ugemm_kq_c_type_nblock0, ugemm_kq_c_type_nblock1)
DECLARE_2D_TILE(
    iq36_score_sum_tile, float, IQ36_SUBGROUP,
    ugemm_kq_sg_tile_n, 1, 1, 1)
DECLARE_2D_TILE(
    iq36_accumulator_scale_tile, float, IQ36_SUBGROUP,
    ugemm_vs_sg_tile_n, 1, 1, 1)
DECLARE_2D_TILE(
    iq36_remainder_mask_tile, float, IQ36_SUBGROUP,
    ugemm_kq_sg_tile_m, 1, 1, 1)
DECLARE_2D_TILE_COPY_REBLOCK(
    iq36_accumulator_tile, IQ36_SUBGROUP,
    ugemm_vs_c_type_block0, ugemm_vs_c_type_block1,
    ugemm_vs_c_type_nblock0, ugemm_vs_c_type_nblock1,
    iq36_accumulator_half_tile, IQ36_SUBGROUP,
    ugemm_vs_sg_tile_m, 8, 1, ugemm_vs_sg_tile_n / 8)
DECLARE_2D_TILE_VREDUCE(
    iq36_score_tile, IQ36_SUBGROUP,
    ugemm_kq_c_type_block0, ugemm_kq_c_type_block1,
    ugemm_kq_c_type_nblock0, ugemm_kq_c_type_nblock1,
    iq36_score_sum_tile, IQ36_SUBGROUP,
    ugemm_kq_sg_tile_n, 1, 1, 1)
DECLARE_2D_TILE_HREDUCE(
    iq36_score_tile, IQ36_SUBGROUP,
    ugemm_kq_c_type_block0, ugemm_kq_c_type_block1,
    ugemm_kq_c_type_nblock0, ugemm_kq_c_type_nblock1,
    iq36_remainder_mask_tile, IQ36_SUBGROUP,
    ugemm_kq_sg_tile_m, 1, 1, 1)
DECLARE_2D_TILE_HREDUCE(
    iq36_accumulator_tile, IQ36_SUBGROUP,
    ugemm_vs_c_type_block0, ugemm_vs_c_type_block1,
    ugemm_vs_c_type_nblock0, ugemm_vs_c_type_nblock1,
    iq36_accumulator_scale_tile, IQ36_SUBGROUP,
    ugemm_vs_sg_tile_n, 1, 1, 1)

#define iq36_binary_add(x, y) ((x) + (y))
#define iq36_binary_mul(x, y) ((x) * (y))
#define iq36_scaled_exp(x) native_vexp2((x) * IQ36_SCALE_LOG2E)
#define iq36_rescale(x, y) \
  native_vexp2(((x) - (y)) * IQ36_SCALE_LOG2E)

#if defined(IQ36_STOCK_MICRO_CONTEXT_PARTITION4)
#define IQ36_STOCK_MICRO_PARTITIONS 4U
#define IQ36_STOCK_MICRO_PARTIAL_OFFSET 2U
#define IQ36_STOCK_MICRO_PARTIAL_HEAD_WIDTH (2U + IQ36_D)

inline ulong iq36_stock_micro_workspace_head(
    const uint batch,
    const uint kv_head,
    const uint partition,
    const uint head) {
  return OUTPUT0_OFFSET +
      (ulong)batch * OUTPUT0_PITCHES[0] +
      (ulong)kv_head * OUTPUT0_PITCHES[1] +
      (ulong)partition * OUTPUT0_PITCHES[2] +
      (IQ36_STOCK_MICRO_PARTIAL_OFFSET +
       head * IQ36_STOCK_MICRO_PARTIAL_HEAD_WIDTH) * OUTPUT0_PITCHES[3];
}

inline ulong iq36_stock_micro_arrival_counter_index(
    const uint batch, const uint kv_head) {
  return INPUT1_OFFSET +
      (ulong)batch * INPUT1_PITCHES[0] +
      (ulong)kv_head * INPUT1_PITCHES[1] +
      (ulong)(INPUT1_DIMS[2] - 1U) * INPUT1_PITCHES[2] +
      (ulong)(INPUT1_DIMS[3] - 1U) * INPUT1_PITCHES[3];
}
#endif

__attribute__((intel_reqd_sub_group_size(IQ36_SUBGROUP)))
#if defined(IQ36_STOCK_MICRO_OWNER)
#if defined(IQ36_STOCK_MICRO_DUAL_COHORT)
__attribute__((reqd_work_group_size(16, 32, 1)))
#else
__attribute__((reqd_work_group_size(16, 16, 1)))
#endif
__kernel void iq36_hot_attention_single_owner(
    const __global INPUT0_TYPE* query,
    __global INPUT1_TYPE* hot_key_bits,
    __global INPUT2_TYPE* hot_value,
    const __global INPUT3_TYPE* current_key,
    const __global INPUT4_TYPE* current_value,
    __global INPUT5_TYPE* cold_key,
    __global INPUT6_TYPE* cold_value,
    __global INPUT7_TYPE* cold_key_scale_bytes,
    __global INPUT8_TYPE* cold_value_scale_bytes,
    const __global INPUT9_TYPE* mask,
    const __global INPUT10_TYPE* eviction_shape_template,
    const __global INPUT11_TYPE* eviction_count,
    const __global INPUT12_TYPE* decode_length_carrier,
    __global OUTPUT0_TYPE* workspace,
    __global OUTPUT1_TYPE* output,
    __global OUTPUT2_TYPE* cold_key_append,
    __global OUTPUT3_TYPE* cold_value_append,
    __global OUTPUT4_TYPE* cold_key_scale_append,
    __global OUTPUT5_TYPE* cold_value_scale_append) {
#else
__kernel void iq36_stock_micro_attention_oracle(
    const __global INPUT0_TYPE* query,
    const __global INPUT1_TYPE* hot_key_bits,
    const __global INPUT2_TYPE* hot_value,
    const __global INPUT3_TYPE* decode_length_carrier,
    const __global INPUT4_TYPE* dependency,
    __global OUTPUT0_TYPE* output) {
#endif
#if defined(IQ36_STOCK_MICRO_DUAL_COHORT)
  const uint physical_subgroup =
      sub_group_broadcast(get_local_id(1), 0);
  const bool dual_producer = physical_subgroup < 16U;
  const uint subgroup =
      dual_producer ? physical_subgroup : physical_subgroup - 16U;
#else
  const uint subgroup = sub_group_broadcast(get_local_id(1), 0);
#endif
  const uint linear_local_id =
      (uint)get_local_id(0) + IQ36_SUBGROUP * (uint)get_local_id(1);
#if defined(IQ36_STOCK_MICRO_DUAL_COHORT)
  const uint cohort_linear_local_id =
      (uint)get_local_id(0) + IQ36_SUBGROUP * subgroup;
#else
  const uint cohort_linear_local_id = linear_local_id;
#endif
  const uint attention_group = (uint)get_group_id(1);
  const uint kv_head = attention_group / IQ36_STOCK_MICRO_GROUPS_PER_KV;
  const uint output_group =
      attention_group % IQ36_STOCK_MICRO_GROUPS_PER_KV;
  const uint batch = (uint)get_group_id(2);
  const uint query_tokens = (uint)INPUT0_DIMS[2];
  const int key_tokens =
      (int)decode_length_carrier[IQ36_ORACLE_LENGTH_OFFSET];
#if defined(IQ36_STOCK_MICRO_CONTEXT_PARTITION4)
  // Partition only on stock's 256-key package boundary.  The final partition
  // owns the current K/V row, so no other work-group observes an unpublished
  // state write and the total K/V traffic remains unchanged.
  const uint context_partition = output_group;
  const uint key_block_count =
      ((uint)key_tokens + ugemm_kq_wg_tile_m - 1U) /
      ugemm_kq_wg_tile_m;
  const uint partition_block_begin =
      key_block_count * context_partition / IQ36_STOCK_MICRO_PARTITIONS;
  const uint partition_block_end =
      key_block_count * (context_partition + 1U) /
      IQ36_STOCK_MICRO_PARTITIONS;
  const uint partition_begin =
      partition_block_begin * ugemm_kq_wg_tile_m;
  const uint partition_end = context_partition + 1U ==
          IQ36_STOCK_MICRO_PARTITIONS
      ? (uint)key_tokens
      : min((uint)key_tokens,
            partition_block_end * ugemm_kq_wg_tile_m);
  const int partition_tokens = (int)(partition_end - partition_begin);
#endif

#if !defined(IQ36_STOCK_MICRO_OWNER)
  // Execute this operation outside host-controlled If.  On prefill it is a
  // format-aware pass-through that retains the state-owner dependency.  On
  // decode, that dependency guarantees the in-place K/V writes are visible
  // before this kernel reads the same state buffers.
  if (query_tokens != 1U) {
    #pragma unroll
    for (uint head = 0U; head < 8U; ++head) {
      const uint query_head = kv_head * 8U + head;
      const ulong output_base = IQ36_ORACLE_OUTPUT_OFFSET +
          (ulong)batch * IQ36_ORACLE_OUTPUT_PITCHES[0] +
          (ulong)query_head * IQ36_ORACLE_OUTPUT_PITCHES[1];
      output[output_base +
             (ulong)linear_local_id * IQ36_ORACLE_OUTPUT_PITCHES[3]] =
          (IQ36_ORACLE_OUTPUT_TYPE)dependency[
              INPUT4_OFFSET +
              (ulong)batch * INPUT4_PITCHES[0] +
              (ulong)query_head * INPUT4_PITCHES[1] +
              (ulong)linear_local_id * INPUT4_PITCHES[3]];
    }
    return;
  }

  // The oracle is admitted only on an exact-history layer.  Retain a safe
  // graph-level fallback if a caller violates that contract.
  if (key_tokens <= 0 || key_tokens > (int)INPUT2_DIMS[2]) {
    #pragma unroll
    for (uint head = 0U; head < 8U; ++head) {
      const uint query_head = kv_head * 8U + head;
      const ulong output_base = IQ36_ORACLE_OUTPUT_OFFSET +
          (ulong)batch * IQ36_ORACLE_OUTPUT_PITCHES[0] +
          (ulong)query_head * IQ36_ORACLE_OUTPUT_PITCHES[1];
      output[output_base +
             (ulong)linear_local_id * IQ36_ORACLE_OUTPUT_PITCHES[3]] =
          (IQ36_ORACLE_OUTPUT_TYPE)dependency[
              INPUT4_OFFSET +
              (ulong)batch * INPUT4_PITCHES[0] +
              (ulong)query_head * INPUT4_PITCHES[1] +
              (ulong)linear_local_id * INPUT4_PITCHES[3]];
    }
    return;
  }
#endif

#if defined(IQ36_STOCK_MICRO_OWNER) && \
    defined(IQ36_STOCK_MICRO_OWNER_WRITE_CURRENT)
  // This operation remains the sole consumer and owner of the request state.
  // Publish the current row before the stock microkernel reads the complete
  // exact-history plane; all 256 work-items then synchronize on the write.
  if (
#if defined(IQ36_STOCK_MICRO_DUAL_COHORT)
      dual_producer &&
#endif
      cohort_linear_local_id < 128U
#if defined(IQ36_STOCK_MICRO_CONTEXT_PARTITION4)
      && context_partition + 1U == IQ36_STOCK_MICRO_PARTITIONS
#endif
  ) {
    const uint dim0 = cohort_linear_local_id * 2U;
    const uint dim1 = dim0 + 1U;
    const uint current_token = (uint)key_tokens - 1U;
    const uint slot = iq36_hot_slot(current_token);
    const ulong current_key_base = INPUT3_OFFSET +
        (ulong)batch * INPUT3_PITCHES[0] +
        (ulong)kv_head * INPUT3_PITCHES[1];
    const ulong current_value_base = iq36_current_value_index(
        batch, kv_head, 0U, 0U);
    const uint key_word = cohort_linear_local_id * IQ36_KEY_TILE_TOKENS +
        (slot & (IQ36_KEY_TILE_TOKENS - 1U));
    const ulong packed_key_index = INPUT1_OFFSET +
        (ulong)batch * INPUT1_PITCHES[0] +
        (ulong)kv_head * INPUT1_PITCHES[1] +
        (ulong)(slot / IQ36_KEY_TILE_TOKENS) * INPUT1_PITCHES[2] +
        (ulong)key_word * INPUT1_PITCHES[3];
    hot_key_bits[packed_key_index] = (INPUT1_TYPE)as_int((half2)(
        (half)current_key[current_key_base +
            (ulong)dim0 * INPUT3_PITCHES[3]],
        (half)current_key[current_key_base +
            (ulong)dim1 * INPUT3_PITCHES[3]]));
    __global half* dense_key = (__global half*)&hot_key_bits[
        iq36_hot_key_dense_i32_base(batch, kv_head)];
    const ulong dense_key_index =
        (ulong)slot * IQ36_D + dim0;
    dense_key[dense_key_index] = (half)current_key[
        current_key_base + (ulong)dim0 * INPUT3_PITCHES[3]];
    dense_key[dense_key_index + 1U] = (half)current_key[
        current_key_base + (ulong)dim1 * INPUT3_PITCHES[3]];
    const ulong value_base = INPUT2_OFFSET +
        (ulong)batch * INPUT2_PITCHES[0] +
        (ulong)kv_head * INPUT2_PITCHES[1] +
        (ulong)slot * INPUT2_PITCHES[2];
    hot_value[value_base + (ulong)dim0 * INPUT2_PITCHES[3]] =
        (INPUT2_TYPE)current_value[
            current_value_base + (ulong)dim0 * INPUT4_PITCHES[3]];
    hot_value[value_base + (ulong)dim1 * INPUT2_PITCHES[3]] =
        (INPUT2_TYPE)current_value[
            current_value_base + (ulong)dim1 * INPUT4_PITCHES[3]];
  }
  barrier(CLK_GLOBAL_MEM_FENCE);
#endif

  const uint packed_blocks = ((uint)INPUT1_DIMS[2] - 1U) / 2U;
  const __global half* key = (const __global half*)&hot_key_bits[
      INPUT1_OFFSET +
      (ulong)batch * INPUT1_PITCHES[0] +
      (ulong)kv_head * INPUT1_PITCHES[1] +
      (ulong)packed_blocks * INPUT1_PITCHES[2]];
  const __global half* value = (const __global half*)&hot_value[
      INPUT2_OFFSET +
      (ulong)batch * INPUT2_PITCHES[0] +
      (ulong)kv_head * INPUT2_PITCHES[1]];
  const __global half* query_head_base = (const __global half*)&query[
      INPUT0_OFFSET +
      (ulong)batch * INPUT0_PITCHES[0] +
      (ulong)(kv_head * 8U) * INPUT0_PITCHES[1]];

#if defined(IQ36_STOCK_MICRO_CONTEXT_PARTITION4)
  key += (ulong)partition_begin * IQ36_D;
  value += (ulong)partition_begin * IQ36_D;
  const int micro_key_tokens = partition_tokens;
#else
  const int micro_key_tokens = key_tokens;
#endif

  __builtin_assume_aligned(key, 128);
  __builtin_assume_aligned(value, 128);
  __builtin_assume_aligned(query_head_base, 128);

  const uint sg_i_kq = subgroup % ugemm_kq_sg_per_wg_m;
  const uint sg_j_kq = subgroup / ugemm_kq_sg_per_wg_m;
  const uint sg_i_vs = subgroup % ugemm_vs_sg_per_wg_m;
  const uint sg_j_vs = subgroup / ugemm_vs_sg_per_wg_m;

#define IQ36_Q_SLM_BYTES \
  (IQ36_D * ugemm_kq_wg_tile_n * sizeof(half))
#define IQ36_S_SLM_BYTES \
  (ugemm_kq_wg_tile_m * ugemm_kq_wg_tile_n * sizeof(half))
#define IQ36_SUM_SLM_BYTES \
  (ugemm_kq_wg_tile_n * ugemm_kq_sg_per_wg_m * sizeof(float))
#define IQ36_MAX_SLM_BYTES (ugemm_kq_wg_tile_n * sizeof(float))
#if defined(IQ36_STOCK_MICRO_CONTEXT_PARTITION4)
#define IQ36_OUTPUT_SLM_BYTES \
  (IQ36_D * ugemm_vs_wg_tile_n * sizeof(float))
#else
#define IQ36_OUTPUT_SLM_BYTES \
  (IQ36_D * ugemm_vs_wg_tile_n * sizeof(half))
#endif
#define IQ36_UGEMM_SLM_BYTES \
  ((ugemm_kq_slm_size > ugemm_vs_slm_size) \
       ? ugemm_kq_slm_size : ugemm_vs_slm_size)
#if defined(IQ36_STOCK_MICRO_DUAL_COHORT)
#define IQ36_DUAL_RAW_BUFFER_ELEMENTS \
  (ugemm_kq_wg_tile_m * ugemm_kq_wg_tile_n)
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
  __local NamedBarrier_t* consumer_barrier = named_barrier_init(16);
  __local NamedBarrier_t* pipeline_barrier = named_barrier_init(32);
#else
  __local char slm[
      IQ36_Q_SLM_BYTES + IQ36_S_SLM_BYTES + IQ36_SUM_SLM_BYTES +
      IQ36_MAX_SLM_BYTES + IQ36_OUTPUT_SLM_BYTES +
      IQ36_UGEMM_SLM_BYTES];
  __local half* query_slm = (__local half*)&slm[0];
  __local half* score_slm =
      (__local half*)&slm[IQ36_Q_SLM_BYTES];
  __local float* sum_slm = (__local float*)&slm[
      IQ36_Q_SLM_BYTES + IQ36_S_SLM_BYTES];
  __local float* max_slm = (__local float*)&slm[
      IQ36_Q_SLM_BYTES + IQ36_S_SLM_BYTES + IQ36_SUM_SLM_BYTES];
#if defined(IQ36_STOCK_MICRO_CONTEXT_PARTITION4)
  __local float* output_slm = (__local float*)&slm[
#else
  __local half* output_slm = (__local half*)&slm[
#endif
      IQ36_Q_SLM_BYTES + IQ36_S_SLM_BYTES + IQ36_SUM_SLM_BYTES +
      IQ36_MAX_SLM_BYTES];
  __local uint* ugemm_slm = (__local uint*)&slm[
      IQ36_Q_SLM_BYTES + IQ36_S_SLM_BYTES + IQ36_SUM_SLM_BYTES +
      IQ36_MAX_SLM_BYTES + IQ36_OUTPUT_SLM_BYTES];
#endif
#if defined(IQ36_STOCK_MICRO_PAGE_SPARSE)
  __local half iq36_page_query_proxy[IQ36_D];
  __local float iq36_page_priority[IQ36_PAGE_MAX_COUNT];
  __local uint iq36_selected_pages[IQ36_PAGE_KEEP];
  __local uint iq36_tile_begin[IQ36_PAGE_MAX_TILES];
  __local uint iq36_tile_length[IQ36_PAGE_MAX_TILES];
  __local uint iq36_tile_count;
#endif
  const bool need_sum_barrier = ugemm_vs_barrier_count == 0;
  iq36_accumulator_tile accumulator;
  iq36_score_sum_tile running_sum;
  iq36_score_sum_tile running_max;
  iq36_score_sum_tile old_running_max;
#if defined(IQ36_STOCK_MICRO_DUAL_COHORT)
  if (dual_producer) {
    iq36_query_tile query_tile;
    const uint query_copy = IQ36_Q_TILE_SG_N * subgroup;
    const uint query_column = subgroup & 7U;
    tile_load_block(
        &query_tile, (const __global uint*)query_head_base, 8, IQ36_D / 2,
        0, query_column);
    tile_store_t_sys_src1(
        query_tile, (__local uint*)query_slm, IQ36_D / 2, query_copy, 0);
  } else {
    const uint max_columns_per_subgroup =
        IQ36_DIV_UP(
            ugemm_kq_wg_tile_n,
            IQ36_SUBGROUP * IQ36_SG_PER_WG);
    const float negative_infinity = -INFINITY;
    #pragma unroll
    for (uint column = 0; column < max_columns_per_subgroup; ++column) {
      intel_sub_group_block_write(
          (__local uint*)&max_slm[
              (column + subgroup * max_columns_per_subgroup) *
              IQ36_SUBGROUP],
          as_uint(negative_infinity));
    }
    tile_fill(accumulator, 0.0f);
    tile_fill(running_sum, 0.0f);
    tile_fill(running_max, -INFINITY);
  }
#else
  iq36_query_tile query_tile;
  const uint query_copy = IQ36_Q_TILE_SG_N * subgroup;
  const uint query_column = subgroup & 7U;
  tile_load_block(
      &query_tile, (const __global uint*)query_head_base, 8, IQ36_D / 2,
      0, query_column);
  tile_store_t_sys_src1(
      query_tile, (__local uint*)query_slm, IQ36_D / 2, query_copy, 0);
  const uint max_columns_per_subgroup =
      IQ36_DIV_UP(
          ugemm_kq_wg_tile_n,
          IQ36_SUBGROUP * IQ36_SG_PER_WG);
  const float negative_infinity = -INFINITY;
  #pragma unroll
  for (uint column = 0; column < max_columns_per_subgroup; ++column) {
    intel_sub_group_block_write(
        (__local uint*)&max_slm[
            (column + subgroup * max_columns_per_subgroup) *
            IQ36_SUBGROUP],
        as_uint(negative_infinity));
  }
  tile_fill(accumulator, 0.0f);
  tile_fill(running_sum, 0.0f);
  tile_fill(running_max, -INFINITY);
#endif
  barrier(CLK_LOCAL_MEM_FENCE);

#if defined(IQ36_STOCK_MICRO_DUAL_COHORT)
  if (dual_producer) {
    cooperative_prefetch_2d_rem(
        key, IQ36_D, micro_key_tokens,
        ugemm_kq_wg_tile_m, 64, IQ36_D,
        subgroup, IQ36_SG_PER_WG, IQ36_SUBGROUP, LSC_LDCC_L1C_L3C);
    iq36_score_tile first_score = ugemm_kq(
        key, IQ36_D, query_slm, IQ36_D, micro_key_tokens,
        ugemm_kq_wg_tile_n, IQ36_D, 0, 0, 0,
        sg_i_kq, sg_j_kq, (__local char*)ugemm_slm);
    tile_store_full(
        first_score, raw_score_double_slm, ugemm_kq_wg_tile_m,
        sg_i_kq * ugemm_kq_sg_tile_m, 0);
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  const uint dual_block_count = IQ36_DIV_UP(
      (uint)micro_key_tokens, ugemm_kq_wg_tile_m);
  for (uint block = 0U; block < dual_block_count; ++block) {
    const int key_begin = (int)(block * ugemm_kq_wg_tile_m);
    const bool first = block == 0U;
    const bool last = block + 1U == dual_block_count;
    const int key_chunk =
        min(micro_key_tokens - key_begin, ugemm_kq_wg_tile_m);
    if (dual_producer) {
      const uint next_block = block + 1U;
      if (next_block < dual_block_count) {
        __local float* next_raw_score =
            raw_score_double_slm +
            (next_block & 1U) * IQ36_DUAL_RAW_BUFFER_ELEMENTS;
        cooperative_prefetch_2d_rem(
            key, IQ36_D, micro_key_tokens,
            ugemm_kq_wg_tile_m, 64, IQ36_D,
            subgroup, IQ36_SG_PER_WG, IQ36_SUBGROUP,
            LSC_LDCC_L1C_L3C);
        iq36_score_tile next_score = ugemm_kq(
            key, IQ36_D, query_slm, IQ36_D, micro_key_tokens,
            ugemm_kq_wg_tile_n, IQ36_D,
            (int)(next_block * ugemm_kq_wg_tile_m), 0, 0,
            sg_i_kq, sg_j_kq, (__local char*)ugemm_slm);
        tile_store_full(
            next_score, next_raw_score, ugemm_kq_wg_tile_m,
            sg_i_kq * ugemm_kq_sg_tile_m, 0);
      }
    } else {
      __local float* current_raw_score =
          raw_score_double_slm +
          (block & 1U) * IQ36_DUAL_RAW_BUFFER_ELEMENTS;
      iq36_score_tile score;
      tile_load_full(
          &score, current_raw_score, ugemm_kq_wg_tile_m,
          sg_i_kq * ugemm_kq_sg_tile_m, 0);

      iq36_remainder_mask_tile key_mask;
      #pragma unroll
      for (uint row = 0;
           row < ugemm_kq_sg_tile_m / IQ36_SUBGROUP; ++row) {
        key_mask.x[0][row] =
            key_begin + (int)(sg_i_kq * ugemm_kq_sg_tile_m) +
                    (int)(row * IQ36_SUBGROUP) +
                    (int)get_sub_group_local_id() < micro_key_tokens
                ? nan(0u)
                : -INFINITY;
      }
      tile_hbroadcast_min(&score, key_mask);
      tile_vreduce_max(score, &running_max);
      max_slm[
          subgroup * IQ36_SUBGROUP +
          (uint)get_sub_group_local_id()] =
          tile_access(
              running_max, 0, 0, IQ36_SUBGROUP,
              ugemm_kq_sg_tile_n, 1, 1);
      work_group_named_barrier(
          consumer_barrier, CLK_LOCAL_MEM_FENCE);
      float reduced_running_max = -INFINITY;
      #pragma unroll
      for (uint subgroup_row = 0U;
           subgroup_row < IQ36_SG_PER_WG;
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
      tile_elementwise(score, iq36_scaled_exp);

      iq36_score_sum_tile chunk_sum;
      tile_fill(chunk_sum, 0.0f);
      tile_vreduce_add(score, &chunk_sum);
      iq36_score_half2_tile score_half2;
      tile_copy_to_half2(score, score_half2);
      tile_store_t_sys_src2(
          score_half2, (__local uint*)score_slm,
          ugemm_vs_sg_tile_n, ugemm_kq_wg_tile_m / 2,
          (sg_i_kq * ugemm_kq_sg_tile_m) / 2, 0);
      if (!first) {
        tile_binary(
            old_running_max, running_max, iq36_rescale);
        tile_binary(running_sum, old_running_max, iq36_binary_mul);
        iq36_accumulator_scale_tile accumulator_scale;
        tile_copy(old_running_max, accumulator_scale);
        tile_hbroadcast_mul(&accumulator, accumulator_scale);
      }
      tile_binary(running_sum, chunk_sum, iq36_binary_add);
      tile_copy(running_max, old_running_max);
      if (last) {
        tile_store_full(
            running_sum, sum_slm, ugemm_kq_wg_tile_n, 0, sg_i_kq);
      }
      work_group_named_barrier(
          consumer_barrier, CLK_LOCAL_MEM_FENCE);
      const __global half* value_chunk =
          value + (ulong)key_begin * IQ36_D;
      iq36_accumulator_tile chunk_accumulator = ugemm_vs(
          value_chunk, IQ36_D, score_slm, ugemm_kq_wg_tile_m,
          IQ36_D, ugemm_kq_wg_tile_n, key_chunk, 0, 0, 0,
          sg_i_vs, sg_j_vs, (__local char*)ugemm_slm);
      tile_binary(accumulator, chunk_accumulator, iq36_binary_add);
    }
    work_group_named_barrier(
        pipeline_barrier, CLK_LOCAL_MEM_FENCE);
  }

  if (!dual_producer) {
    iq36_accumulator_scale_tile total_sum;
    iq36_accumulator_scale_tile partial_sum;
    tile_fill(total_sum, 0.0f);
    #pragma unroll
    for (uint subgroup_row = 0U;
         subgroup_row < ugemm_kq_sg_per_wg_m; ++subgroup_row) {
      tile_load_full(
          &partial_sum, sum_slm, ugemm_kq_wg_tile_n,
          ugemm_vs_sg_tile_n * sg_j_vs, subgroup_row);
      tile_binary(total_sum, partial_sum, iq36_binary_add);
    }
    tile_elementwise(total_sum, native_vrecip);
    tile_hbroadcast_mul(&accumulator, total_sum);

    iq36_accumulator_half_tile output_tile;
    tile_copy_reblock(accumulator, &output_tile);
    const uint output_row = sg_i_vs * ugemm_vs_sg_tile_m;
    tile_store_full(
        output_tile, output_slm, IQ36_D, output_row, 0);
    work_group_named_barrier(
        consumer_barrier, CLK_LOCAL_MEM_FENCE);
    #pragma unroll
    for (uint head = 0U; head < 8U; ++head) {
      const uint query_head = kv_head * 8U + head;
      const ulong output_base = IQ36_ORACLE_OUTPUT_OFFSET +
          (ulong)batch * IQ36_ORACLE_OUTPUT_PITCHES[0] +
          (ulong)query_head * IQ36_ORACLE_OUTPUT_PITCHES[1];
      output[output_base +
             (ulong)cohort_linear_local_id *
                 IQ36_ORACLE_OUTPUT_PITCHES[3]] =
          (IQ36_ORACLE_OUTPUT_TYPE)output_slm[
              cohort_linear_local_id + head * IQ36_D];
    }
  }
#else
#if defined(IQ36_STOCK_MICRO_PAGE_SPARSE)
  const bool iq36_page_sparse_active =
      key_tokens >= 65536 &&
      key_tokens <= (int)(
          IQ36_PAGE_SINK_TOKENS +
          IQ36_PAGE_MAX_COUNT * IQ36_PAGE_TOKENS);
  if (iq36_page_sparse_active) {
    if (linear_local_id < IQ36_D) {
      float query_proxy = 0.0f;
      #pragma unroll
      for (uint head = 0U; head < 8U; ++head) {
        query_proxy += convert_float(
            query_head_base[(ulong)head * IQ36_D + linear_local_id]);
      }
      iq36_page_query_proxy[linear_local_id] =
          convert_half(query_proxy * 0.125f);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    const uint page_token_count =
        (uint)key_tokens - IQ36_PAGE_SINK_TOKENS;
    const uint page_count = IQ36_DIV_UP(
        page_token_count, IQ36_PAGE_TOKENS);
    for (uint page = subgroup; page < page_count;
         page += IQ36_SG_PER_WG) {
      const uint page_begin =
          IQ36_PAGE_SINK_TOKENS + page * IQ36_PAGE_TOKENS;
      const uint page_length = min(
          IQ36_PAGE_TOKENS, (uint)key_tokens - page_begin);
      const uint sample_count = min(IQ36_PAGE_SAMPLES, page_length);
      const bool sample_active =
          (uint)get_sub_group_local_id() < sample_count;
      const uint sample_token = page_begin +
          ((2U * (uint)get_sub_group_local_id() + 1U) * page_length) /
              (2U * sample_count);
      float sample_score = 0.0f;
      if (sample_active) {
        #pragma unroll 1
        for (uint dim = 0U; dim < IQ36_D; ++dim) {
          sample_score += convert_float(
              key[(ulong)sample_token * IQ36_D + dim]) * convert_float(
                  iq36_page_query_proxy[dim]);
        }
      }
      const float lane_score = sample_active ? sample_score : -INFINITY;
      const float maximum = sub_group_reduce_max(lane_score);
      const float weight = sample_active
          ? native_exp2((lane_score - maximum) * IQ36_SCALE_LOG2E)
          : 0.0f;
      const float sum = sub_group_reduce_add(weight);
      if (get_sub_group_local_id() == 0U) {
        iq36_page_priority[page] = maximum * 0.0625f + native_log(
            sum * ((float)page_length / (float)sample_count));
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (linear_local_id == 0U) {
      for (uint selected = 0U; selected < IQ36_PAGE_KEEP; ++selected) {
        float best_priority = -INFINITY;
        uint best_page = 0xffffffffU;
        for (uint page = 0U; page < page_count; ++page) {
          const float candidate = iq36_page_priority[page];
          if (candidate > best_priority ||
              (candidate == best_priority && page < best_page)) {
            best_priority = candidate;
            best_page = page;
          }
        }
        iq36_selected_pages[selected] = best_page;
        iq36_page_priority[best_page] = -INFINITY;
      }
      for (uint index = 1U; index < IQ36_PAGE_KEEP; ++index) {
        const uint page = iq36_selected_pages[index];
        uint position = index;
        while (position > 0U &&
               iq36_selected_pages[position - 1U] > page) {
          iq36_selected_pages[position] =
              iq36_selected_pages[position - 1U];
          --position;
        }
        iq36_selected_pages[position] = page;
      }
      uint tiles = 0U;
      iq36_tile_begin[tiles] = 0U;
      iq36_tile_length[tiles++] = IQ36_PAGE_SINK_TOKENS;
      for (uint selected = 0U; selected < IQ36_PAGE_KEEP; ++selected) {
        const uint begin = IQ36_PAGE_SINK_TOKENS +
            iq36_selected_pages[selected] * IQ36_PAGE_TOKENS;
        uint remaining = min(
            IQ36_PAGE_TOKENS, (uint)key_tokens - begin);
        for (uint offset = 0U; remaining != 0U;
             offset += IQ36_PAGE_TILE_TOKENS) {
          const uint length = min(IQ36_PAGE_TILE_TOKENS, remaining);
          iq36_tile_begin[tiles] = begin + offset;
          iq36_tile_length[tiles++] = length;
          remaining -= length;
        }
      }
      iq36_tile_count = tiles;
    }
  } else if (linear_local_id == 0U) {
    const uint tiles = IQ36_DIV_UP(
        (uint)micro_key_tokens, IQ36_PAGE_TILE_TOKENS);
    for (uint tile = 0U; tile < tiles; ++tile) {
      const uint begin = tile * IQ36_PAGE_TILE_TOKENS;
      iq36_tile_begin[tile] = begin;
      iq36_tile_length[tile] = min(
          IQ36_PAGE_TILE_TOKENS, (uint)micro_key_tokens - begin);
    }
    iq36_tile_count = tiles;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
#endif

#if defined(IQ36_STOCK_MICRO_PAGE_SPARSE)
  for (uint iq36_tile = 0U; iq36_tile < iq36_tile_count; ++iq36_tile) {
    const int key_begin = 0;
    const bool first = iq36_tile == 0U;
    const bool last = iq36_tile + 1U == iq36_tile_count;
    const int key_chunk = (int)iq36_tile_length[iq36_tile];
    const __global half* iq36_tile_key =
        key + (ulong)iq36_tile_begin[iq36_tile] * IQ36_D;
    const __global half* iq36_tile_value =
        value + (ulong)iq36_tile_begin[iq36_tile] * IQ36_D;
#else
  for (int key_begin = 0; key_begin < micro_key_tokens;
       key_begin += ugemm_kq_wg_tile_m) {
    const bool first = key_begin == 0;
    const bool last =
        key_begin + ugemm_kq_wg_tile_m >= micro_key_tokens;
    const int key_chunk =
        min(micro_key_tokens - key_begin, ugemm_kq_wg_tile_m);
#endif
    const uint sg_key_begin = sg_i_kq * ugemm_kq_sg_tile_m;
#if defined(IQ36_STOCK_MICRO_PAGE_SPARSE)
    const __global half* iq36_active_key = iq36_tile_key;
    const __global half* iq36_active_value = iq36_tile_value;
    const int iq36_active_tokens = key_chunk;
    const int iq36_active_begin = 0;
#else
    const __global half* iq36_active_key = key;
    const __global half* iq36_active_value = value;
    const int iq36_active_tokens = micro_key_tokens;
    const int iq36_active_begin = key_begin;
#endif

    cooperative_prefetch_2d_rem(
        iq36_active_key, IQ36_D, iq36_active_tokens,
        ugemm_kq_wg_tile_m, 64, IQ36_D,
        subgroup, IQ36_SG_PER_WG, IQ36_SUBGROUP, LSC_LDCC_L1C_L3C);

    iq36_score_tile score = ugemm_kq(
        iq36_active_key, IQ36_D, query_slm, IQ36_D, iq36_active_tokens,
        ugemm_kq_wg_tile_n, IQ36_D, iq36_active_begin, 0, 0,
        sg_i_kq, sg_j_kq, (__local char*)ugemm_slm);

    iq36_remainder_mask_tile key_mask;
    #pragma unroll
    for (uint row = 0;
         row < ugemm_kq_sg_tile_m / IQ36_SUBGROUP; ++row) {
      key_mask.x[0][row] =
          iq36_active_begin + (int)sg_key_begin +
                  (int)(row * IQ36_SUBGROUP) +
                  (int)get_sub_group_local_id() < iq36_active_tokens
              ? nan(0u)
              : -INFINITY;
    }
    tile_hbroadcast_min(&score, key_mask);
    tile_vreduce_max(score, &running_max);
    tile_atomic_max_full(
        running_max, max_slm, ugemm_kq_wg_tile_n, 0, 0);
    intel_work_group_barrier_arrive(CLK_LOCAL_MEM_FENCE);

    cooperative_prefetch_2d_rem(
        iq36_active_value, IQ36_D,
        iq36_active_tokens - iq36_active_begin,
        64, ugemm_kq_wg_tile_m, IQ36_D,
        subgroup, IQ36_SG_PER_WG, IQ36_SUBGROUP, LSC_LDCC_L1C_L3C);

    intel_work_group_barrier_wait(CLK_LOCAL_MEM_FENCE);
    tile_load_full(
        &running_max, max_slm, ugemm_kq_wg_tile_n, 0, 0);
    tile_vbroadcast_sub(&score, running_max);
    tile_elementwise(score, iq36_scaled_exp);

    iq36_score_sum_tile chunk_sum;
    tile_fill(chunk_sum, 0.0f);
    tile_vreduce_add(score, &chunk_sum);
    iq36_score_half2_tile score_half2;
    tile_copy_to_half2(score, score_half2);
    tile_store_t_sys_src2(
        score_half2, (__local uint*)score_slm,
        ugemm_vs_sg_tile_n, ugemm_kq_wg_tile_m / 2,
        sg_key_begin / 2, 0);
    intel_work_group_barrier_arrive(CLK_LOCAL_MEM_FENCE);

    if (!first) {
      tile_binary(
          old_running_max, running_max, iq36_rescale);
      tile_binary(running_sum, old_running_max, iq36_binary_mul);
      iq36_accumulator_scale_tile accumulator_scale;
      tile_copy(old_running_max, accumulator_scale);
      tile_hbroadcast_mul(&accumulator, accumulator_scale);
    }
    tile_binary(running_sum, chunk_sum, iq36_binary_add);
    tile_copy(running_max, old_running_max);
    if (last) {
      tile_store_full(
          running_sum, sum_slm, ugemm_kq_wg_tile_n, 0, sg_i_kq);
    }

    intel_work_group_barrier_wait(CLK_LOCAL_MEM_FENCE);
    if (last && need_sum_barrier) {
      intel_work_group_barrier_arrive(CLK_LOCAL_MEM_FENCE);
    }
    iq36_accumulator_tile chunk_accumulator = ugemm_vs(
        iq36_active_value, IQ36_D, score_slm, ugemm_kq_wg_tile_m,
        IQ36_D, ugemm_kq_wg_tile_n, key_chunk, 0, 0, 0,
        sg_i_vs, sg_j_vs, (__local char*)ugemm_slm);
#if !defined(IQ36_STOCK_MICRO_PAGE_SPARSE)
    value += IQ36_D * ugemm_kq_wg_tile_m;
#endif
    tile_binary(accumulator, chunk_accumulator, iq36_binary_add);
  }

  if (need_sum_barrier) {
    intel_work_group_barrier_wait(CLK_LOCAL_MEM_FENCE);
  }
  iq36_accumulator_scale_tile total_sum;
  iq36_accumulator_scale_tile partial_sum;
  tile_fill(total_sum, 0.0f);
  #pragma unroll
  for (uint subgroup_row = 0;
       subgroup_row < ugemm_kq_sg_per_wg_m; ++subgroup_row) {
    tile_load_full(
        &partial_sum, sum_slm, ugemm_kq_wg_tile_n,
        ugemm_vs_sg_tile_n * sg_j_vs, subgroup_row);
    tile_binary(total_sum, partial_sum, iq36_binary_add);
  }

#if defined(IQ36_STOCK_MICRO_CONTEXT_PARTITION4)
  // Keep both 8-column package outputs live before exporting the eight real
  // GQA columns.  The package fuser requires both inline-assembly results;
  // a direct first-half-only access lets IGC replace the second with `null`.
  // F32 staging also preserves the exact-package numerator for the merge.
  const uint output_row = sg_i_vs * ugemm_vs_sg_tile_m;
  tile_store_full(
      accumulator, output_slm, IQ36_D, output_row, 0);
  barrier(CLK_LOCAL_MEM_FENCE);

  // Each subgroup owns one 16-dimension row block and publishes the eight
  // real GQA columns.
  const uint lane = (uint)get_sub_group_local_id();
  const uint output_dimension =
      sg_i_vs * ugemm_vs_sg_tile_m + lane;
  #pragma unroll
  for (uint head = 0U; head < 8U; ++head) {
    const ulong head_base = iq36_stock_micro_workspace_head(
        batch, kv_head, context_partition, head);
    workspace[head_base +
        (IQ36_STOCK_MICRO_PARTIAL_OFFSET + output_dimension) *
            OUTPUT0_PITCHES[3]] = (OUTPUT0_TYPE)output_slm[
                output_dimension + head * IQ36_D];
  }
  if (subgroup == 0U && lane < 8U) {
    const ulong head_base = iq36_stock_micro_workspace_head(
        batch, kv_head, context_partition, lane);
    workspace[head_base] = (OUTPUT0_TYPE)max_slm[lane];
    workspace[head_base + OUTPUT0_PITCHES[3]] =
        (OUTPUT0_TYPE)tile_access(
            total_sum, 0, 0, IQ36_SUBGROUP,
            ugemm_vs_sg_tile_n, 1, 1);
  }
  barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);

  __local uint* partition_is_last = (__local uint*)max_slm;
  if (linear_local_id == 0U) {
    atomic_work_item_fence(
        CLK_GLOBAL_MEM_FENCE, memory_order_release, memory_scope_device);
    volatile __global unsigned int* counter =
        (volatile __global unsigned int*)&hot_key_bits[
            iq36_stock_micro_arrival_counter_index(batch, kv_head)];
    const uint count_mask = 0x7U;
    const uint generation = (uint)key_tokens << 3U;
    uint observed = atomic_or(counter, 0U);
    while ((observed & ~count_mask) != generation) {
      const uint replaced = atomic_cmpxchg(counter, observed, generation);
      if (replaced == observed) {
        observed = generation;
        break;
      }
      observed = replaced;
    }
    const uint previous = atomic_inc(counter) & count_mask;
    partition_is_last[0] =
        previous + 1U == IQ36_STOCK_MICRO_PARTITIONS;
    if (partition_is_last[0] != 0U) {
      atomic_work_item_fence(
          CLK_GLOBAL_MEM_FENCE, memory_order_acquire, memory_scope_device);
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (partition_is_last[0] == 0U) return;

  // Reuse the now-dead output staging plane for the online-softmax scales and
  // one denominator per GQA head.  Keep the same chronological recurrence as
  // the stock package: rescale the accumulated prefix, then add this
  // partition.  Only eight work-items evaluate exponentials; all 256
  // dimensions replay the same four scale pairs for their F32 numerators.
  __local float* merge_old_scale = (__local float*)output_slm;
  __local float* merge_partition_scale =
      merge_old_scale + 8U * IQ36_STOCK_MICRO_PARTITIONS;
  __local float* merge_sum =
      merge_partition_scale + 8U * IQ36_STOCK_MICRO_PARTITIONS;
  if (linear_local_id < 8U) {
    const uint head = linear_local_id;
    float running_maximum = -INFINITY;
    float denominator = 0.0f;
    #pragma unroll
    for (uint partition = 0U;
         partition < IQ36_STOCK_MICRO_PARTITIONS; ++partition) {
      const ulong head_base = iq36_stock_micro_workspace_head(
          batch, kv_head, partition, head);
      const float partition_maximum = convert_float(workspace[head_base]);
      const float next_maximum = fmax(running_maximum, partition_maximum);
      const float old_scale = native_exp2(
          (running_maximum - next_maximum) * IQ36_SCALE_LOG2E);
      const float partition_scale = native_exp2(
          (partition_maximum - next_maximum) *
          IQ36_SCALE_LOG2E);
      const uint scale_index =
          head * IQ36_STOCK_MICRO_PARTITIONS + partition;
      merge_old_scale[scale_index] = old_scale;
      merge_partition_scale[scale_index] = partition_scale;
      denominator *= old_scale;
      denominator += convert_float(
          workspace[head_base + OUTPUT0_PITCHES[3]]) * partition_scale;
      running_maximum = next_maximum;
    }
    merge_sum[head] = denominator;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  #pragma unroll
  for (uint head = 0U; head < 8U; ++head) {
    float numerator = 0.0f;
    #pragma unroll
    for (uint partition = 0U;
         partition < IQ36_STOCK_MICRO_PARTITIONS; ++partition) {
      const ulong head_base = iq36_stock_micro_workspace_head(
          batch, kv_head, partition, head);
      const uint scale_index =
          head * IQ36_STOCK_MICRO_PARTITIONS + partition;
      numerator *= merge_old_scale[scale_index];
      numerator += convert_float(workspace[head_base +
          (IQ36_STOCK_MICRO_PARTIAL_OFFSET + linear_local_id) *
              OUTPUT0_PITCHES[3]]) *
          merge_partition_scale[scale_index];
    }
    const half result = convert_half_rte(
        numerator * native_recip(merge_sum[head]));
    const uint query_head = kv_head * 8U + head;
    const ulong output_base = IQ36_ORACLE_OUTPUT_OFFSET +
        (ulong)batch * IQ36_ORACLE_OUTPUT_PITCHES[0] +
        (ulong)query_head * IQ36_ORACLE_OUTPUT_PITCHES[1];
    output[output_base +
           (ulong)linear_local_id * IQ36_ORACLE_OUTPUT_PITCHES[3]] =
        (IQ36_ORACLE_OUTPUT_TYPE)result;
  }
#else
  tile_elementwise(total_sum, native_vrecip);
  tile_hbroadcast_mul(&accumulator, total_sum);

  iq36_accumulator_half_tile output_tile;
  tile_copy_reblock(accumulator, &output_tile);
  const uint output_row = sg_i_vs * ugemm_vs_sg_tile_m;
  // Keep every generated VS package result live.  Unlike stock's dynamic-q
  // program, this custom op has a compile-time q=1 output shape; a direct
  // one-column store lets IGC delete the second result operand before the
  // vISA package fuser resolves it.  Full local staging preserves the exact
  // package ABI, then the work-group copies only the real query column.
  tile_store_full(
      output_tile, output_slm, IQ36_D, output_row, 0);
  barrier(CLK_LOCAL_MEM_FENCE);
  const uint output_heads = 8U / IQ36_STOCK_MICRO_GROUPS_PER_KV;
  const uint output_head_begin = output_group * output_heads;
  #pragma unroll
  for (uint local_head = 0U; local_head < output_heads; ++local_head) {
    const uint head = output_head_begin + local_head;
    const uint query_head = kv_head * 8U + head;
    const ulong output_base = IQ36_ORACLE_OUTPUT_OFFSET +
        (ulong)batch * IQ36_ORACLE_OUTPUT_PITCHES[0] +
        (ulong)query_head * IQ36_ORACLE_OUTPUT_PITCHES[1];
    output[output_base +
           (ulong)linear_local_id * IQ36_ORACLE_OUTPUT_PITCHES[3]] =
        (IQ36_ORACLE_OUTPUT_TYPE)output_slm[
            linear_local_id + head * IQ36_D];
  }
#endif
#endif
}

#undef IQ36_D
#undef IQ36_DIV_UP
#undef IQ36_MAX_SLM_BYTES
#undef IQ36_OUTPUT_SLM_BYTES
#undef IQ36_Q_SLM_BYTES
#undef IQ36_Q_TILE_SG_N
#undef IQ36_SCALE_LOG2E
#undef IQ36_SG_PER_WG
#undef IQ36_STOCK_MICRO_GROUPS_PER_KV
#if defined(IQ36_STOCK_MICRO_CONTEXT_PARTITION4)
#undef IQ36_STOCK_MICRO_PARTIAL_HEAD_WIDTH
#undef IQ36_STOCK_MICRO_PARTIAL_OFFSET
#undef IQ36_STOCK_MICRO_PARTITIONS
#endif
#undef IQ36_SUBGROUP
#undef SUBGROUP_SIZE
#undef IQ36_SUM_SLM_BYTES
#undef IQ36_S_SLM_BYTES
#undef IQ36_UGEMM_SLM_BYTES
#if defined(IQ36_DUAL_RAW_BUFFER_ELEMENTS)
#undef IQ36_DUAL_RAW_BUFFER_ELEMENTS
#endif
#undef iq36_binary_add
#undef iq36_binary_mul
#undef iq36_rescale
#undef iq36_scaled_exp

#undef IQ36_ORACLE_LENGTH_OFFSET
#undef IQ36_ORACLE_OUTPUT_OFFSET
#undef IQ36_ORACLE_OUTPUT_PITCHES
#undef IQ36_ORACLE_OUTPUT_TYPE

#endif
