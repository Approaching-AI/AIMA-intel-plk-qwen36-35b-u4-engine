// Prefill-only tiled GQA carrier recovered from the bounded split prototype.
//
// One work-group owns 32 query rows for one query head.  It computes the
// stock-shaped 128-key softmax stages and, for the first GQA head of each KV
// head, publishes the current K/V rows into the request-owned hot carrier.
// Decode is deliberately absent: graph control flow selects the existing
// single-owner decode operation instead.

#if !defined(IQ36_UNIFIED_SHAPE_SPECIALIZATION) || \
    defined(IQ36_BUILD_PREFILL_ONLY)

#if defined(IQ36_UNIFIED_SHAPE_SPECIALIZATION)
#define IQ36_PREFILL_KERNEL_ENTRY iq36_hot_attention_single_owner
#define IQ36_PREFILL_ATTENTION_TYPE OUTPUT1_TYPE
#define IQ36_PREFILL_COLD_KEY_OFFSET OUTPUT2_OFFSET
#define IQ36_PREFILL_COLD_KEY_PITCHES OUTPUT2_PITCHES
#define IQ36_PREFILL_COLD_KEY_TYPE OUTPUT2_TYPE
#define IQ36_PREFILL_COLD_VALUE_OFFSET OUTPUT3_OFFSET
#define IQ36_PREFILL_COLD_VALUE_PITCHES OUTPUT3_PITCHES
#define IQ36_PREFILL_COLD_VALUE_TYPE OUTPUT3_TYPE
#define IQ36_PREFILL_KEY_SCALE_OFFSET OUTPUT4_OFFSET
#define IQ36_PREFILL_KEY_SCALE_PITCHES OUTPUT4_PITCHES
#define IQ36_PREFILL_KEY_SCALE_TYPE OUTPUT4_TYPE
#define IQ36_PREFILL_VALUE_SCALE_OFFSET OUTPUT5_OFFSET
#define IQ36_PREFILL_VALUE_SCALE_PITCHES OUTPUT5_PITCHES
#define IQ36_PREFILL_VALUE_SCALE_TYPE OUTPUT5_TYPE
#else
#define IQ36_PREFILL_KERNEL_ENTRY iq36_prefill_attention_tiled
#define IQ36_PREFILL_ATTENTION_TYPE OUTPUT0_TYPE
#define IQ36_PREFILL_COLD_KEY_OFFSET OUTPUT1_OFFSET
#define IQ36_PREFILL_COLD_KEY_PITCHES OUTPUT1_PITCHES
#define IQ36_PREFILL_COLD_KEY_TYPE OUTPUT1_TYPE
#define IQ36_PREFILL_COLD_VALUE_OFFSET OUTPUT2_OFFSET
#define IQ36_PREFILL_COLD_VALUE_PITCHES OUTPUT2_PITCHES
#define IQ36_PREFILL_COLD_VALUE_TYPE OUTPUT2_TYPE
#define IQ36_PREFILL_KEY_SCALE_OFFSET OUTPUT3_OFFSET
#define IQ36_PREFILL_KEY_SCALE_PITCHES OUTPUT3_PITCHES
#define IQ36_PREFILL_KEY_SCALE_TYPE OUTPUT3_TYPE
#define IQ36_PREFILL_VALUE_SCALE_OFFSET OUTPUT4_OFFSET
#define IQ36_PREFILL_VALUE_SCALE_PITCHES OUTPUT4_PITCHES
#define IQ36_PREFILL_VALUE_SCALE_TYPE OUTPUT4_TYPE
#endif

#if defined(IQ36_PREFILL_USE_MICROKERNEL)
// The generated shims expose the same KQ and VS tiles used by the pinned
// stock prefill provider.  Derive the cooperative query copy from their
// subgroup geometry so compile-only gates can test alternate query tiles.
// Keep the surrounding online-softmax carrier source-level so it can select
// between the request-owned dense hot plane and the current K/V input at each
// aligned 128-token boundary.
typedef ugemm_kq_c_type iq36_micro_score_tile;
typedef ugemm_vs_c_type iq36_micro_output_tile;
#define IQ36_MICRO_SUBGROUPS \
    (ugemm_kq_sg_per_wg_m * ugemm_kq_sg_per_wg_n)
#if IQ36_PREFILL_QUERY_TILE == 32 && \
    ugemm_kq_sg_per_wg_m == 8 && ugemm_kq_sg_per_wg_n == 1 && \
    ugemm_vs_sg_per_wg_m == 8 && ugemm_vs_sg_per_wg_n == 1
#define IQ36_MICRO_QUERY_COPY_ROWS 4
#define IQ36_MICRO_KQ_I(subgroup_id) (subgroup_id)
#define IQ36_MICRO_KQ_J(subgroup_id) 0
#define IQ36_MICRO_VS_I(subgroup_id) (subgroup_id)
#define IQ36_MICRO_VS_J(subgroup_id) 0
#else
#define IQ36_MICRO_QUERY_COPY_ROWS \
    ((IQ36_PREFILL_QUERY_TILE + IQ36_MICRO_SUBGROUPS - 1) / \
     IQ36_MICRO_SUBGROUPS)
#define IQ36_MICRO_KQ_I(subgroup_id) \
    ((subgroup_id) % ugemm_kq_sg_per_wg_m)
#define IQ36_MICRO_KQ_J(subgroup_id) \
    ((subgroup_id) / ugemm_kq_sg_per_wg_m)
#define IQ36_MICRO_VS_I(subgroup_id) \
    ((subgroup_id) % ugemm_vs_sg_per_wg_m)
#define IQ36_MICRO_VS_J(subgroup_id) \
    ((subgroup_id) / ugemm_vs_sg_per_wg_m)
#endif
DECLARE_2D_TILE(
    iq36_micro_query_tile, uint, 16, IQ36_HEAD_DIM / 2, 1, 1,
    IQ36_MICRO_QUERY_COPY_ROWS)
DECLARE_2D_TILE(
    iq36_micro_output_half_tile, half, 16,
    ugemm_vs_sg_tile_m, 8, 1, ugemm_vs_sg_tile_n / 8)
DECLARE_2D_TILE(
    iq36_micro_score_half2_tile, uint, 16,
    ugemm_kq_c_type_block0, ugemm_kq_c_type_block1 / 2,
    ugemm_kq_c_type_nblock0, ugemm_kq_c_type_nblock1)
DECLARE_2D_TILE(
    iq36_micro_sum_tile, float, 16, ugemm_kq_sg_tile_n, 1, 1, 1)
DECLARE_2D_TILE(
    iq36_micro_scale_tile, float, 16, ugemm_vs_sg_tile_n, 1, 1, 1)
DECLARE_2D_TILE_COPY_REBLOCK(
    iq36_micro_output_tile, 16,
    ugemm_vs_c_type_block0, ugemm_vs_c_type_block1,
    ugemm_vs_c_type_nblock0, ugemm_vs_c_type_nblock1,
    iq36_micro_output_half_tile, 16,
    ugemm_vs_sg_tile_m, 8, 1, ugemm_vs_sg_tile_n / 8)
DECLARE_2D_TILE_VREDUCE(
    iq36_micro_score_tile, 16,
    ugemm_kq_c_type_block0, ugemm_kq_c_type_block1,
    ugemm_kq_c_type_nblock0, ugemm_kq_c_type_nblock1,
    iq36_micro_sum_tile, 16, ugemm_kq_sg_tile_n, 1, 1, 1)
DECLARE_2D_TILE_HREDUCE(
    iq36_micro_output_tile, 16,
    ugemm_vs_c_type_block0, ugemm_vs_c_type_block1,
    ugemm_vs_c_type_nblock0, ugemm_vs_c_type_nblock1,
    iq36_micro_scale_tile, 16, ugemm_vs_sg_tile_n, 1, 1, 1)
#if ugemm_kq_wg_tile_n == ugemm_vs_wg_tile_n && \
    ugemm_kq_sg_tile_n != ugemm_vs_sg_tile_n && \
    (ugemm_kq_sg_tile_n % ugemm_vs_sg_tile_n) == 0
DECLARE_2D_TILE_RSELECT(
    iq36_micro_scale_tile, 16, ugemm_vs_sg_tile_n, 1, 1, 1,
    iq36_micro_sum_tile, 16, ugemm_kq_sg_tile_n, 1, 1, 1)
#endif
#define iq36_micro_add(x, y) ((x) + (y))
#endif

#if defined(IQ36_PREFILL_USE_MICROKERNEL) && \
    !(IQ36_PREFILL_QUERY_TILE == 32 && \
      ugemm_kq_sg_per_wg_m == 8 && ugemm_kq_sg_per_wg_n == 1)
#define IQ36_PREFILL_WORKGROUP_SIZE \
    (16 * ugemm_kq_sg_per_wg_m * ugemm_kq_sg_per_wg_n)
#else
#define IQ36_PREFILL_WORKGROUP_SIZE 128
#endif

#if defined(IQ36_PREFILL_LONG_CONTEXT_PREFETCH) && \
    !defined(IQ36_PREFILL_PREFETCH_MIN_KEY_TOKENS)
#define IQ36_PREFILL_PREFETCH_MIN_KEY_TOKENS 32768U
#endif

__attribute__((reqd_work_group_size(IQ36_PREFILL_WORKGROUP_SIZE, 1, 1)))
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void IQ36_PREFILL_KERNEL_ENTRY(
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
    const __global INPUT12_TYPE* query_length,
#if defined(IQ36_FUSED_GATE_OUTPUT)
    const __global INPUT13_TYPE* raw_gate,
#endif
#if defined(IQ36_UNIFIED_SHAPE_SPECIALIZATION)
    __global OUTPUT0_TYPE* workspace,
    __global OUTPUT1_TYPE* output,
    __global OUTPUT2_TYPE* cold_key_append,
    __global OUTPUT3_TYPE* cold_value_append,
    __global OUTPUT4_TYPE* cold_key_scale_append,
    __global OUTPUT5_TYPE* cold_value_scale_append) {
#else
    __global OUTPUT0_TYPE* output,
    __global OUTPUT1_TYPE* cold_key_append,
    __global OUTPUT2_TYPE* cold_value_append,
    __global OUTPUT3_TYPE* cold_key_scale_append,
    __global OUTPUT4_TYPE* cold_value_scale_append) {
#endif
  const uint local_id = (uint)get_local_id(0);
  const uint subgroup = (uint)get_sub_group_id();
  const uint lane = (uint)get_sub_group_local_id();
  const uint query_tile = (uint)get_group_id(1);
  const uint batch_head = (uint)get_group_id(2);
  const uint batch = batch_head / IQ36_Q_HEADS;
  const uint query_head = batch_head - batch * IQ36_Q_HEADS;
  const uint kv_head = query_head / IQ36_GQA_GROUP;
  const uint batch_kv = batch * IQ36_KV_HEADS + kv_head;
  // Prefill is shape-specialized by the GPU plugin for every resident chunk.
#if defined(IQ36_UNIFIED_SHAPE_SPECIALIZATION)
  const uint query_tokens = IQ36_STATIC_QUERY_TOKENS;
#else
  const uint query_tokens = (uint)INPUT0_DIMS[2];
#endif
#if defined(IQ36_PREFILL_INITIAL)
  // A distinct operation identity makes these facts literal compile-time
  // constants.  IGC does not fold INPUT*_DIMS compound literals early enough
  // to remove the generic history and codec paths on its own.
  const uint key_tokens = query_tokens;
  const uint past_tokens = 0U;
  const uint desired_cold_tokens = 0U;
  const bool fixed_cold_state = true;
  const uint cold_append_tokens = 0U;
  const uint cold_tokens = 0U;
#else
#if defined(IQ36_PREFILL_RUNTIME_LENGTH)
  const uint key_tokens = (uint)query_length[INPUT12_OFFSET];
#else
  const uint key_tokens = (uint)INPUT9_DIMS[3];
#endif
  const uint past_tokens = key_tokens - query_tokens;
  const uint desired_cold_tokens = key_tokens > IQ36_HOT_WINDOW
      ? key_tokens - IQ36_HOT_WINDOW : 0U;
  // A fixed bucket carrier is deliberately larger than the logical key
  // extent.  Dynamic append-only state is always <= key_tokens-HOT_WINDOW+1,
  // so this is a compile-time discriminator without another runtime input.
  const bool fixed_cold_state = (uint)INPUT5_DIMS[2] > key_tokens;
  const uint previous_cold_tokens = past_tokens > IQ36_HOT_WINDOW
      ? past_tokens - IQ36_HOT_WINDOW : 0U;
  const uint cold_append_tokens = fixed_cold_state
      ? desired_cold_tokens - previous_cold_tokens
      : (uint)eviction_count[INPUT11_OFFSET];
  const uint cold_tokens = desired_cold_tokens - cold_append_tokens;
#endif
#if IQ36_PREFILL_FULL_HISTORY
  const bool full_prefill_history = true;
  const uint attention_cold_tokens = 0U;
#else
  const bool full_prefill_history = (uint)INPUT2_DIMS[2] > key_tokens;
  const uint attention_cold_tokens =
      full_prefill_history ? 0U : cold_tokens;
#endif
  const uint query_begin = query_tile * IQ36_PREFILL_QUERY_TILE;
  const uint query_count = query_begin < query_tokens
      ? min((uint)IQ36_PREFILL_QUERY_TILE, query_tokens - query_begin) : 0U;
  const uint dim = subgroup * IQ36_TOKEN_TILE + lane;
  const uint output_tile_width =
      IQ36_PREFILL_QUERY_TILE * IQ36_HEAD_DIM;
  const ulong output_tile =
      ((ulong)batch_head * (ulong)get_global_size(1) + query_tile) *
          output_tile_width;

  if (query_count == 0U) return;

#if defined(IQ36_PREFILL_USE_MICROKERNEL)
#if !defined(IQ36_PREFILL_FULL_HISTORY) || !IQ36_PREFILL_FULL_HISTORY
#error "The generated prefill microkernels require the full F16 history plane"
#endif
  const uint causal_tokens = past_tokens + query_begin + query_count;
  const __global half* query_base = (const __global half*)&query[
      INPUT0_OFFSET + (ulong)batch * INPUT0_PITCHES[0] +
      (ulong)query_head * INPUT0_PITCHES[1]];
  const __global half* current_key_base = (const __global half*)&current_key[
      INPUT3_OFFSET + (ulong)batch * INPUT3_PITCHES[0] +
      (ulong)kv_head * INPUT3_PITCHES[1]];
  const __global half* current_value_base =
      (const __global half*)&current_value[
          iq36_current_value_index(batch, kv_head, 0U, 0U)];
  const __global half* state_key_base =
      (const __global half*)&hot_key_bits[
          iq36_hot_key_dense_i32_base(batch, kv_head)];
  const __global half* state_value_base = (const __global half*)&hot_value[
      INPUT2_OFFSET + (ulong)batch * INPUT2_PITCHES[0] +
      (ulong)kv_head * INPUT2_PITCHES[1]];

#define IQ36_MICRO_Q_SLM_BYTES \
    (IQ36_HEAD_DIM * IQ36_PREFILL_QUERY_TILE * sizeof(half))
#define IQ36_MICRO_S_SLM_BYTES \
    (IQ36_PREFILL_CHUNK_TOKENS * IQ36_PREFILL_QUERY_TILE * sizeof(half))
#define IQ36_MICRO_SUM_SLM_BYTES \
    (IQ36_PREFILL_QUERY_TILE * IQ36_PREFILL_BLOCKS_PER_CHUNK * sizeof(float))
#define IQ36_MICRO_MAX_SLM_BYTES \
    (IQ36_PREFILL_QUERY_TILE * sizeof(float))
  __local char micro_storage[
      IQ36_MICRO_Q_SLM_BYTES + IQ36_MICRO_S_SLM_BYTES +
      IQ36_MICRO_SUM_SLM_BYTES + IQ36_MICRO_MAX_SLM_BYTES + 1U];
  __local half* micro_query = (__local half*)&micro_storage[0];
  __local half* micro_score =
      (__local half*)&micro_storage[IQ36_MICRO_Q_SLM_BYTES];
  __local float* micro_sum = (__local float*)&micro_storage[
      IQ36_MICRO_Q_SLM_BYTES + IQ36_MICRO_S_SLM_BYTES];
  __local float* micro_max = (__local float*)&micro_storage[
      IQ36_MICRO_Q_SLM_BYTES + IQ36_MICRO_S_SLM_BYTES +
      IQ36_MICRO_SUM_SLM_BYTES];
  __local char* micro_scratch = (__local char*)&micro_storage[
      IQ36_MICRO_Q_SLM_BYTES + IQ36_MICRO_S_SLM_BYTES +
      IQ36_MICRO_SUM_SLM_BYTES + IQ36_MICRO_MAX_SLM_BYTES];

  iq36_micro_query_tile query_fragment;
  const uint query_copy = IQ36_MICRO_QUERY_COPY_ROWS * subgroup;
  tile_load(
      &query_fragment, (const __global uint*)query_base,
      (int)((IQ36_HEAD_DIM + 1U) / 2U), (int)query_tokens,
      (int)(INPUT0_PITCHES[2] / 2U), 0,
      (int)(query_begin + query_copy));
  tile_store_t_sys_src1(
      query_fragment, (__local uint*)micro_query,
      (int)(IQ36_HEAD_DIM / 2U), (int)query_copy, 0);
  if (local_id < IQ36_PREFILL_QUERY_TILE)
    micro_max[local_id] = -INFINITY;

  iq36_micro_output_tile output_accumulator;
  iq36_micro_sum_tile running_sum;
  iq36_micro_sum_tile running_max;
  iq36_micro_sum_tile previous_max;
  tile_fill(output_accumulator, 0.0f);
  tile_fill(running_sum, 0.0f);
  tile_fill(running_max, -INFINITY);
  barrier(CLK_LOCAL_MEM_FENCE);

#if defined(IQ36_PREFILL_LONG_CONTEXT_PREFETCH)
  // The stock generated SDPA overlaps its first K tile with setup and then
  // overlaps each following K tile plus the current V tile with softmax/VS.
  // Preserve the accepted short-context schedule below the measured 32k
  // crossover, where this carrier already beats the generated provider.
  if (key_tokens > IQ36_PREFILL_PREFETCH_MIN_KEY_TOKENS) {
    const bool first_prior = past_tokens != 0U;
    const __global half* first_key = first_prior
        ? state_key_base + (ulong)iq36_hot_slot(0U) * IQ36_HEAD_DIM
        : current_key_base;
    const int first_key_pitch = first_prior
        ? (int)IQ36_HEAD_DIM : (int)INPUT3_PITCHES[2];
    cooperative_prefetch_2d_rem(
        first_key, IQ36_HEAD_DIM,
        min((uint)IQ36_PREFILL_CHUNK_TOKENS, causal_tokens),
        64U, IQ36_PREFILL_CHUNK_TOKENS, first_key_pitch,
          subgroup, IQ36_MICRO_SUBGROUPS, IQ36_TOKEN_TILE,
        LSC_LDCC_L1C_L3C);
  }
#endif

#if defined(IQ36_PREFILL_ARITHMETIC_ROOFLINE)
  // Component-only arithmetic ceiling.  Keep the exact extracted KQ/VS
  // packages and their score handoff, but remove mask, softmax, state, and
  // graph work.  A runtime repeat count prevents package replication by loop
  // unrolling; product XML never defines this branch.
  const uint roofline_tokens = (uint)INPUT3_DIMS[2];
  const uint roofline_repeats =
      (uint)eviction_count[INPUT11_OFFSET];
#pragma unroll 1
  for (uint repeat = 0U; repeat < roofline_repeats; ++repeat) {
    iq36_micro_score_tile score = ugemm_kq(
        current_key_base, (int)INPUT3_PITCHES[2],
        micro_query, IQ36_HEAD_DIM,
        roofline_tokens, IQ36_PREFILL_QUERY_TILE, IQ36_HEAD_DIM,
        0, 0, 0, IQ36_MICRO_KQ_I(subgroup),
        IQ36_MICRO_KQ_J(subgroup), micro_scratch);
    iq36_micro_score_half2_tile score_half;
    tile_copy_to_half2(score, score_half);
    tile_store_t_sys_src2(
        score_half, (__local uint*)micro_score,
        ugemm_vs_sg_tile_n, roofline_tokens / 2U,
        IQ36_MICRO_KQ_I(subgroup) * ugemm_kq_sg_tile_m / 2U,
        IQ36_MICRO_KQ_J(subgroup) * ugemm_kq_sg_tile_n);
    barrier(CLK_LOCAL_MEM_FENCE);

    iq36_micro_output_tile product = ugemm_vs(
        current_value_base, (int)IQ36_CURRENT_VALUE_TOKEN_PITCH,
        micro_score, roofline_tokens,
        IQ36_HEAD_DIM, IQ36_PREFILL_QUERY_TILE, roofline_tokens,
        0, 0, 0, IQ36_MICRO_VS_I(subgroup),
        IQ36_MICRO_VS_J(subgroup), micro_scratch);
    tile_binary(output_accumulator, product, iq36_micro_add);
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  iq36_micro_output_half_tile roofline_output;
  tile_copy_reblock(output_accumulator, &roofline_output);
  tile_store(
      roofline_output, (__global half*)&output[output_tile],
      IQ36_HEAD_DIM, query_count, IQ36_HEAD_DIM,
      IQ36_MICRO_VS_I(subgroup) * ugemm_vs_sg_tile_m,
      IQ36_MICRO_VS_J(subgroup) * ugemm_vs_sg_tile_n);
  return;
#else
  for (uint chunk_begin = 0U; chunk_begin < causal_tokens;
       chunk_begin += IQ36_PREFILL_CHUNK_TOKENS) {
    const bool first = chunk_begin == 0U;
    const bool last =
        chunk_begin + IQ36_PREFILL_CHUNK_TOKENS >= causal_tokens;
    const uint chunk_tokens = min(
        (uint)IQ36_PREFILL_CHUNK_TOKENS, causal_tokens - chunk_begin);
    const bool prior_chunk = chunk_begin < past_tokens;
    const uint source_token = prior_chunk
        ? iq36_hot_slot(chunk_begin) : chunk_begin - past_tokens;
    const __global half* key_chunk = prior_chunk
        ? state_key_base + (ulong)source_token * IQ36_HEAD_DIM
        : current_key_base +
              (ulong)source_token * INPUT3_PITCHES[2];
    const __global half* value_chunk = prior_chunk
        ? state_value_base +
              (ulong)source_token * INPUT2_PITCHES[2]
        : current_value_base +
              (ulong)source_token * IQ36_CURRENT_VALUE_TOKEN_PITCH;
    const int key_pitch = prior_chunk
        ? (int)IQ36_HEAD_DIM : (int)INPUT3_PITCHES[2];
    const int value_pitch = prior_chunk
        ? (int)INPUT2_PITCHES[2] :
          (int)IQ36_CURRENT_VALUE_TOKEN_PITCH;

    iq36_micro_score_tile score = ugemm_kq(
        key_chunk, key_pitch, micro_query, IQ36_HEAD_DIM,
        chunk_tokens, IQ36_PREFILL_QUERY_TILE, IQ36_HEAD_DIM,
        0, 0, 0, IQ36_MICRO_KQ_I(subgroup),
        IQ36_MICRO_KQ_J(subgroup), micro_scratch);
#if defined(IQ36_PREFILL_LONG_CONTEXT_PREFETCH)
    if (key_tokens > IQ36_PREFILL_PREFETCH_MIN_KEY_TOKENS) {
      cooperative_prefetch_2d_rem(
          value_chunk, IQ36_HEAD_DIM, chunk_tokens,
          64U, IQ36_PREFILL_CHUNK_TOKENS, value_pitch,
          subgroup, IQ36_MICRO_SUBGROUPS, IQ36_TOKEN_TILE,
          LSC_LDCC_L1C_L3C);
    }
#endif
    const uint subgroup_key =
        IQ36_MICRO_KQ_I(subgroup) * ugemm_kq_sg_tile_m;
    const uint subgroup_query_kq =
        IQ36_MICRO_KQ_J(subgroup) * ugemm_kq_sg_tile_n;
#define iq36_micro_invalid(key_position, query_position) \
    ((key_position) >= causal_tokens || (key_position) > (query_position))
    tile_predicated_assignment_t(
        score, chunk_begin + subgroup_key,
        past_tokens + query_begin + subgroup_query_kq,
        iq36_micro_invalid, -FLT_MAX, 16,
        ugemm_kq_c_type_block0, ugemm_kq_c_type_block1,
        ugemm_kq_c_type_nblock0, ugemm_kq_c_type_nblock1);
#undef iq36_micro_invalid

    tile_vreduce_max(score, &running_max);
    tile_atomic_max_full(
        running_max, micro_max, IQ36_PREFILL_QUERY_TILE,
        subgroup_query_kq, 0);
    barrier(CLK_LOCAL_MEM_FENCE);
    tile_load_full(
        &running_max, micro_max, IQ36_PREFILL_QUERY_TILE,
        subgroup_query_kq, 0);
    tile_vbroadcast_sub(&score, running_max);
#define iq36_micro_scaled_exp(value) \
    native_vexp2((value) * IQ36_EXP2_SCALE)
    tile_elementwise(score, iq36_micro_scaled_exp);
#undef iq36_micro_scaled_exp

    iq36_micro_sum_tile chunk_sum;
    tile_fill(chunk_sum, 0.0f);
    tile_vreduce_add(score, &chunk_sum);
    iq36_micro_score_half2_tile score_half;
    tile_copy_to_half2(score, score_half);
    tile_store_t_sys_src2(
        score_half, (__local uint*)micro_score,
        ugemm_vs_sg_tile_n, IQ36_PREFILL_CHUNK_TOKENS / 2U,
        subgroup_key / 2U, subgroup_query_kq);
    barrier(CLK_LOCAL_MEM_FENCE);

    if (!first) {
#define iq36_micro_exp_delta(old_value, new_value) \
      native_vexp2(((old_value) - (new_value)) * IQ36_EXP2_SCALE)
#define iq36_micro_multiply(x, y) ((x) * (y))
      tile_binary(previous_max, running_max, iq36_micro_exp_delta);
      tile_binary(running_sum, previous_max, iq36_micro_multiply);
      iq36_micro_scale_tile output_scale;
#if ugemm_kq_wg_tile_n == ugemm_vs_wg_tile_n && \
    ugemm_kq_sg_tile_n == ugemm_vs_sg_tile_n
      tile_copy(previous_max, output_scale);
#elif ugemm_kq_wg_tile_n == ugemm_vs_wg_tile_n && \
      (ugemm_kq_sg_tile_n % ugemm_vs_sg_tile_n) == 0
      tile_rselect(
          &output_scale, previous_max,
          IQ36_MICRO_VS_J(subgroup) %
              (ugemm_kq_sg_tile_n / ugemm_vs_sg_tile_n));
#else
#error "unsupported KQ-to-VS query-tile mapping"
#endif
      tile_hbroadcast_mul(&output_accumulator, output_scale);
#undef iq36_micro_multiply
#undef iq36_micro_exp_delta
    }
    tile_binary(running_sum, chunk_sum, iq36_micro_add);
    tile_copy(running_max, previous_max);
    if (last) {
      tile_store_full(
          running_sum, micro_sum, IQ36_PREFILL_QUERY_TILE,
          subgroup_query_kq, IQ36_MICRO_KQ_I(subgroup));
    }

#if defined(IQ36_PREFILL_LONG_CONTEXT_PREFETCH)
    if (key_tokens > IQ36_PREFILL_PREFETCH_MIN_KEY_TOKENS && !last) {
      const uint next_chunk_begin =
          chunk_begin + IQ36_PREFILL_CHUNK_TOKENS;
      const bool next_prior = next_chunk_begin < past_tokens;
      const uint next_source_token = next_prior
          ? iq36_hot_slot(next_chunk_begin)
          : next_chunk_begin - past_tokens;
      const __global half* next_key = next_prior
          ? state_key_base + (ulong)next_source_token * IQ36_HEAD_DIM
          : current_key_base +
                (ulong)next_source_token * INPUT3_PITCHES[2];
      const int next_key_pitch = next_prior
          ? (int)IQ36_HEAD_DIM : (int)INPUT3_PITCHES[2];
      cooperative_prefetch_2d_rem(
          next_key, IQ36_HEAD_DIM,
          min((uint)IQ36_PREFILL_CHUNK_TOKENS,
              causal_tokens - next_chunk_begin),
          IQ36_HEAD_DIM, IQ36_PREFILL_CHUNK_TOKENS, next_key_pitch,
          subgroup, IQ36_MICRO_SUBGROUPS, IQ36_TOKEN_TILE,
          LSC_LDCC_L1C_L3C);
    }
#endif

    iq36_micro_output_tile chunk_output = ugemm_vs(
        value_chunk, value_pitch, micro_score,
        IQ36_PREFILL_CHUNK_TOKENS,
        IQ36_HEAD_DIM, IQ36_PREFILL_QUERY_TILE, chunk_tokens,
        0, 0, 0, IQ36_MICRO_VS_I(subgroup),
        IQ36_MICRO_VS_J(subgroup), micro_scratch);
    tile_binary(output_accumulator, chunk_output, iq36_micro_add);
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  iq36_micro_scale_tile reciprocal_sum;
  iq36_micro_scale_tile subgroup_sum;
  tile_fill(reciprocal_sum, 0.0f);
  for (uint key_subgroup = 0U;
       key_subgroup < IQ36_PREFILL_BLOCKS_PER_CHUNK; ++key_subgroup) {
    tile_load_full(
        &subgroup_sum, micro_sum, IQ36_PREFILL_QUERY_TILE,
        IQ36_MICRO_VS_J(subgroup) * ugemm_vs_sg_tile_n,
        key_subgroup);
    tile_binary(reciprocal_sum, subgroup_sum, iq36_micro_add);
  }
#if ugemm_vs_sg_tile_n == 16
  tile_elementwise(reciprocal_sum, native_vrecip);
#else
  tile_elementwise(reciprocal_sum, native_recip);
#endif
  tile_hbroadcast_mul(&output_accumulator, reciprocal_sum);
#if defined(IQ36_TOKEN_MAJOR_OUTPUT)
  // The XMX accumulator is blocked as 32 output dimensions by 32 queries per
  // subgroup. Store it directly as [B,Q,H,D] without materializing the
  // head-major tensor. The gated variant additionally owns its old epilogue.
  #pragma unroll
  for (uint query_in_tile = 0U;
       query_in_tile < IQ36_PREFILL_QUERY_TILE; ++query_in_tile) {
    if (query_in_tile < query_count) {
      #pragma unroll
      for (uint dimension_block = 0U;
           dimension_block < ugemm_vs_sg_tile_m;
           dimension_block += 16U) {
        const uint output_dimension =
            IQ36_MICRO_VS_I(subgroup) * ugemm_vs_sg_tile_m +
            dimension_block + lane;
        const uint query_position = query_begin + query_in_tile;
        const ulong output_index = OUTPUT1_OFFSET +
            (ulong)batch * OUTPUT1_PITCHES[0] +
            (ulong)query_position * OUTPUT1_PITCHES[1] +
            (ulong)query_head * OUTPUT1_PITCHES[2] +
            (ulong)output_dimension * OUTPUT1_PITCHES[3];
        #if defined(IQ36_FUSED_GATE_OUTPUT)
        const ulong gate_index = INPUT13_OFFSET +
            (ulong)batch * INPUT13_PITCHES[0] +
            (ulong)query_position * INPUT13_PITCHES[1] +
            (ulong)query_head * INPUT13_PITCHES[2] +
            (ulong)output_dimension * INPUT13_PITCHES[3];
        output[output_index] = iq36_gated_attention_value(
            tile_access(
                output_accumulator, dimension_block, query_in_tile,
                16, ugemm_vs_c_type_block0, ugemm_vs_c_type_block1,
                ugemm_vs_c_type_nblock0),
            raw_gate, gate_index);
        #else
        output[output_index] = (OUTPUT1_TYPE)tile_access(
            output_accumulator, dimension_block, query_in_tile,
            16, ugemm_vs_c_type_block0, ugemm_vs_c_type_block1,
            ugemm_vs_c_type_nblock0);
        #endif
      }
    }
  }
#else
  iq36_micro_output_half_tile output_half;
  tile_copy_reblock(output_accumulator, &output_half);
  tile_store(
      output_half, (__global half*)&output[output_tile],
      IQ36_HEAD_DIM, query_count, IQ36_HEAD_DIM,
      IQ36_MICRO_VS_I(subgroup) * ugemm_vs_sg_tile_m,
      IQ36_MICRO_VS_J(subgroup) * ugemm_vs_sg_tile_n);
#endif
#endif

#undef IQ36_MICRO_MAX_SLM_BYTES
#undef IQ36_MICRO_SUM_SLM_BYTES
#undef IQ36_MICRO_S_SLM_BYTES
#undef IQ36_MICRO_Q_SLM_BYTES
#undef IQ36_MICRO_VS_J
#undef IQ36_MICRO_VS_I
#undef IQ36_MICRO_KQ_J
#undef IQ36_MICRO_KQ_I
#undef IQ36_MICRO_QUERY_COPY_ROWS
#undef IQ36_MICRO_SUBGROUPS
#else
  // Softmax is invariant to a common shift.  The observed all-layer 8k score
  // maxima cluster around 128, so centering every finite score there retains
  // materially more useful precision in the same 8 KiB F16 carrier without
  // a per-score SLM correction load.
  __local half local_score_weight[
      IQ36_PREFILL_QUERY_TILE * IQ36_PREFILL_CHUNK_TOKENS];
  __local float local_block_sum[
      IQ36_PREFILL_BLOCKS_PER_CHUNK * IQ36_PREFILL_QUERY_TILE];
  __local float local_running_max[IQ36_PREFILL_QUERY_TILE];
  __local float local_running_sum[IQ36_PREFILL_QUERY_TILE];
  __local float local_previous_scale[IQ36_PREFILL_QUERY_TILE];
  if (local_id < query_count) {
    local_running_max[local_id] = -INFINITY;
    local_running_sum[local_id] = 0.0f;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  float8 output_accumulator0[IQ36_PREFILL_QUERY_GROUPS];
  float8 output_accumulator1[IQ36_PREFILL_QUERY_GROUPS];
  #pragma unroll
  for (uint group = 0U; group < IQ36_PREFILL_QUERY_GROUPS; ++group) {
    output_accumulator0[group] = (float8)(0.0f);
    output_accumulator1[group] = (float8)(0.0f);
  }
  const uint causal_tokens = past_tokens + query_begin + query_count;
#if defined(IQ36_PREFILL_INITIAL)
  // Exact BFYX layouts are contiguous in the initial branch.  Flatten the
  // hot loop's query/current addressing so the compiler does not materialize
  // dynamic pitch arrays in every QK and V iteration.
  const __global half* query_base =
      (const __global half*)&query[
          (ulong)batch_head * query_tokens * IQ36_HEAD_DIM];
  const __global half* value_base =
      (const __global half*)&current_value[
          iq36_current_value_index(batch, kv_head, 0U, 0U)];
#else
  const __global half* query_base =
      (const __global half*)&query[
          INPUT0_OFFSET + (ulong)batch * INPUT0_PITCHES[0] +
          (ulong)query_head * INPUT0_PITCHES[1]];
  const __global half* value_base =
      (const __global half*)&current_value[
          iq36_current_value_index(batch, kv_head, 0U, 0U)];
#endif

  for (uint chunk_begin = 0U; chunk_begin < causal_tokens;
       chunk_begin += IQ36_PREFILL_CHUNK_TOKENS) {
    const uint chunk_tokens = min(
        (uint)IQ36_PREFILL_CHUNK_TOKENS, causal_tokens - chunk_begin);
    const uint valid_blocks =
        (chunk_tokens + IQ36_TOKEN_TILE - 1U) / IQ36_TOKEN_TILE;

    if (local_id < query_count) {
      local_previous_scale[local_id] = local_running_max[local_id];
    }

    if (subgroup < valid_blocks) {
      const uint token_in_chunk = subgroup * IQ36_TOKEN_TILE + lane;
      const uint token = chunk_begin + token_in_chunk;
      const uint block_token =
          chunk_begin + subgroup * IQ36_TOKEN_TILE;
      const uint block_slot = iq36_hot_slot(block_token);
      const bool prior_hot_contiguous =
          block_token + IQ36_TOKEN_TILE <= past_tokens &&
          (attention_cold_tokens == 0U ||
           block_token >= attention_cold_tokens) &&
          (block_slot & (IQ36_KEY_TILE_TOKENS - 1U)) == 0U &&
          block_slot + IQ36_TOKEN_TILE <= (uint)INPUT2_DIMS[2];
      const bool prior_cold_contiguous =
          block_token >= IQ36_SINK_TOKENS &&
          block_token + IQ36_TOKEN_TILE <= attention_cold_tokens;
      const bool current_contiguous = block_token >= past_tokens;
      float8 score[IQ36_PREFILL_QUERY_GROUPS];
      #pragma unroll
      for (uint group = 0U; group < IQ36_PREFILL_QUERY_GROUPS; ++group) {
        score[group] = (float8)(0.0f);
      }
      #pragma unroll 1
      for (uint k_block = 0U;
           k_block < IQ36_HEAD_DIM / IQ36_TOKEN_TILE; ++k_block) {
        const uint k_base = k_block * IQ36_TOKEN_TILE;
        int8 key_fragment;
        if (prior_hot_contiguous) {
          const __global half* dense_hot_key =
              (const __global half*)&hot_key_bits[
                  iq36_hot_key_dense_i32_base(batch, kv_head)];
          const ulong key_index =
              (ulong)iq36_hot_slot(token) * IQ36_HEAD_DIM + k_base;
          key_fragment = as_int8(vload16(0, &dense_hot_key[key_index]));
        } else if (prior_cold_contiguous) {
          key_fragment = iq36_cold_key_fragment(
              batch, kv_head, token, k_base,
              cold_key, cold_key_scale_bytes);
        } else if (current_contiguous) {
          const uint current_token = token - past_tokens;
#if defined(IQ36_PREFILL_INITIAL)
          const ulong key_index =
              ((ulong)batch_kv * query_tokens + current_token) *
                  IQ36_HEAD_DIM + k_base;
#else
          const ulong key_index = INPUT3_OFFSET +
              (ulong)batch * INPUT3_PITCHES[0] +
              (ulong)kv_head * INPUT3_PITCHES[1] +
              (ulong)current_token * INPUT3_PITCHES[2] +
              (ulong)k_base * INPUT3_PITCHES[3];
#endif
          key_fragment = as_int8(vload16(
              0, (const __global half*)&current_key[key_index]));
        } else {
          half16 values;
          #pragma unroll
          for (uint offset = 0U; offset < IQ36_TOKEN_TILE; ++offset) {
            values[offset] = convert_half_rte(iq36_partial_load_key(
                batch, kv_head, token, past_tokens, attention_cold_tokens,
                k_base + offset, hot_key_bits, current_key, cold_key,
                cold_key_scale_bytes));
          }
          key_fragment = as_int8(values);
        }
        #pragma unroll
        for (uint group = 0U;
             group < IQ36_PREFILL_QUERY_GROUPS; ++group) {
          const short8 query_fragment = as_short8(
              iq36_block2d_load_f16_16x8(
                  query_base,
#if defined(IQ36_PREFILL_INITIAL)
                  IQ36_HEAD_DIM * (int)sizeof(half),
                  (int)query_tokens,
                  IQ36_HEAD_DIM * (int)sizeof(half),
#else
                  (int)INPUT0_DIMS[3] * (int)sizeof(half),
                  (int)query_tokens,
                  (int)INPUT0_PITCHES[2] * (int)sizeof(half),
#endif
                  k_base,
                  query_begin + group * IQ36_GQA_GROUP));
          score[group] = intel_sub_group_f16_f16_matrix_mad_k16(
              query_fragment, key_fragment, score[group]);
        }
      }

      #pragma unroll
      for (uint group = 0U;
           group < IQ36_PREFILL_QUERY_GROUPS; ++group) {
        #pragma unroll
        for (uint row = 0U; row < IQ36_GQA_GROUP; ++row) {
          const uint query_in_tile = group * IQ36_GQA_GROUP + row;
          const uint query_position = query_begin + query_in_tile;
          float centered_value = -INFINITY;
          if (query_in_tile < query_count &&
              token <= past_tokens + query_position) {
            // The locked batch-one product has no padding: the explicit
            // causal predicate is the complete mask, and every admitted mask
            // element is zero.  Elide the quadratic mask read entirely.
            centered_value = score[group][row] - 128.0f;
          }
          local_score_weight[
              query_in_tile * IQ36_PREFILL_CHUNK_TOKENS + token_in_chunk] =
                  convert_half_rte(centered_value);
          const float maximum = sub_group_reduce_max(centered_value);
          if (lane == 0U) {
            (void)__builtin_IB_atomic_max_local_f32(
                &local_running_max[query_in_tile], maximum);
          }
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (subgroup < valid_blocks) {
      const uint token_in_chunk = subgroup * IQ36_TOKEN_TILE + lane;
      for (uint query_in_tile = 0U; query_in_tile < query_count;
           ++query_in_tile) {
        const float weight = native_exp2(
            (convert_float(local_score_weight[
                 query_in_tile * IQ36_PREFILL_CHUNK_TOKENS + token_in_chunk]) -
             local_running_max[query_in_tile]) * IQ36_EXP2_SCALE);
        local_score_weight[
            query_in_tile * IQ36_PREFILL_CHUNK_TOKENS + token_in_chunk] =
                convert_half_rte(weight);
        const float sum = sub_group_reduce_add(weight);
        if (lane == 0U) {
          local_block_sum[
              subgroup * IQ36_PREFILL_QUERY_TILE + query_in_tile] = sum;
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_id < query_count) {
      float chunk_sum = 0.0f;
      for (uint block = 0U; block < valid_blocks; ++block) {
        chunk_sum += local_block_sum[
            block * IQ36_PREFILL_QUERY_TILE + local_id];
      }
      const float next_max = local_running_max[local_id];
      const float previous_scale = native_exp2(
          (local_previous_scale[local_id] - next_max) *
              IQ36_EXP2_SCALE);
      local_previous_scale[local_id] = previous_scale;
      local_running_sum[local_id] =
          local_running_sum[local_id] * previous_scale + chunk_sum;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    float8 chunk_output0[IQ36_PREFILL_QUERY_GROUPS];
    float8 chunk_output1[IQ36_PREFILL_QUERY_GROUPS];
    #pragma unroll
    for (uint group = 0U; group < IQ36_PREFILL_QUERY_GROUPS; ++group) {
      chunk_output0[group] = (float8)(0.0f);
      chunk_output1[group] = (float8)(0.0f);
    }
    for (uint block = 0U; block < valid_blocks; ++block) {
      const uint token_in_chunk = block * IQ36_TOKEN_TILE + lane;
      const uint block_token = chunk_begin + block * IQ36_TOKEN_TILE;
      const uint block_slot = iq36_hot_slot(block_token);
      const bool prior_hot_contiguous =
          block_token + IQ36_TOKEN_TILE <= past_tokens &&
          (attention_cold_tokens == 0U ||
           block_token >= attention_cold_tokens) &&
          block_slot + IQ36_TOKEN_TILE <= (uint)INPUT2_DIMS[2];
      const bool current_contiguous = block_token >= past_tokens;
      const __global half* state_value_base =
          (const __global half*)&hot_value[
              INPUT2_OFFSET + (ulong)batch * INPUT2_PITCHES[0] +
              (ulong)kv_head * INPUT2_PITCHES[1]];
      half8 value0;
      half8 value1;
      half8 value2;
      half8 value3;
#if defined(IQ36_DIRECT_I8_FIXED_LAYOUT)
      const bool direct_cold_value_block =
          block_token >= IQ36_SINK_TOKENS &&
          block_token + IQ36_TOKEN_TILE <= attention_cold_tokens;
      if (direct_cold_value_block) {
        const int8 direct0 = iq36_direct_cold_value_fragment(
            batch, kv_head, block_token,
            subgroup * IQ36_TOKEN_TILE + lane,
            cold_value, cold_value_scale_bytes);
        const int8 direct1 = iq36_direct_cold_value_fragment(
            batch, kv_head, block_token,
            128U + subgroup * IQ36_TOKEN_TILE + lane,
            cold_value, cold_value_scale_bytes);
        value0 = as_half8(direct0.s0123);
        value1 = as_half8(direct0.s4567);
        value2 = as_half8(direct1.s0123);
        value3 = as_half8(direct1.s4567);
      } else
#endif
      if (prior_hot_contiguous) {
        value0 = iq36_block2d_load_f16_16x8(
            state_value_base,
            (int)INPUT2_DIMS[3] * (int)sizeof(half),
            (int)INPUT2_DIMS[2],
            (int)INPUT2_PITCHES[2] * (int)sizeof(half),
            subgroup * IQ36_TOKEN_TILE, block_slot);
        value1 = iq36_block2d_load_f16_16x8(
            state_value_base,
            (int)INPUT2_DIMS[3] * (int)sizeof(half),
            (int)INPUT2_DIMS[2],
            (int)INPUT2_PITCHES[2] * (int)sizeof(half),
            subgroup * IQ36_TOKEN_TILE, block_slot + 8U);
        value2 = iq36_block2d_load_f16_16x8(
            state_value_base,
            (int)INPUT2_DIMS[3] * (int)sizeof(half),
            (int)INPUT2_DIMS[2],
            (int)INPUT2_PITCHES[2] * (int)sizeof(half),
            128U + subgroup * IQ36_TOKEN_TILE, block_slot);
        value3 = iq36_block2d_load_f16_16x8(
            state_value_base,
            (int)INPUT2_DIMS[3] * (int)sizeof(half),
            (int)INPUT2_DIMS[2],
            (int)INPUT2_PITCHES[2] * (int)sizeof(half),
            128U + subgroup * IQ36_TOKEN_TILE, block_slot + 8U);
      } else if (current_contiguous) {
        const uint current_token = block_token - past_tokens;
        value0 = iq36_block2d_load_f16_16x8(
            value_base,
#if defined(IQ36_PREFILL_INITIAL)
            IQ36_HEAD_DIM * (int)sizeof(half),
            (int)query_tokens,
            (int)IQ36_CURRENT_VALUE_TOKEN_PITCH * (int)sizeof(half),
#else
            (int)INPUT4_DIMS[3] * (int)sizeof(half),
            (int)query_tokens,
            (int)IQ36_CURRENT_VALUE_TOKEN_PITCH * (int)sizeof(half),
#endif
            subgroup * IQ36_TOKEN_TILE, current_token);
        value1 = iq36_block2d_load_f16_16x8(
            value_base,
#if defined(IQ36_PREFILL_INITIAL)
            IQ36_HEAD_DIM * (int)sizeof(half),
            (int)query_tokens,
            (int)IQ36_CURRENT_VALUE_TOKEN_PITCH * (int)sizeof(half),
#else
            (int)INPUT4_DIMS[3] * (int)sizeof(half),
            (int)query_tokens,
            (int)IQ36_CURRENT_VALUE_TOKEN_PITCH * (int)sizeof(half),
#endif
            subgroup * IQ36_TOKEN_TILE, current_token + 8U);
        value2 = iq36_block2d_load_f16_16x8(
            value_base,
#if defined(IQ36_PREFILL_INITIAL)
            IQ36_HEAD_DIM * (int)sizeof(half),
            (int)query_tokens,
            (int)IQ36_CURRENT_VALUE_TOKEN_PITCH * (int)sizeof(half),
#else
            (int)INPUT4_DIMS[3] * (int)sizeof(half),
            (int)query_tokens,
            (int)IQ36_CURRENT_VALUE_TOKEN_PITCH * (int)sizeof(half),
#endif
            128U + subgroup * IQ36_TOKEN_TILE, current_token);
        value3 = iq36_block2d_load_f16_16x8(
            value_base,
#if defined(IQ36_PREFILL_INITIAL)
            IQ36_HEAD_DIM * (int)sizeof(half),
            (int)query_tokens,
            (int)IQ36_CURRENT_VALUE_TOKEN_PITCH * (int)sizeof(half),
#else
            (int)INPUT4_DIMS[3] * (int)sizeof(half),
            (int)query_tokens,
            (int)IQ36_CURRENT_VALUE_TOKEN_PITCH * (int)sizeof(half),
#endif
            128U + subgroup * IQ36_TOKEN_TILE, current_token + 8U);
      } else {
        #pragma unroll
        for (uint row = 0U; row < 8U; ++row) {
          value0[row] = convert_half_rte(iq36_partial_load_value(
              batch, kv_head, block_token + row, past_tokens,
              attention_cold_tokens,
              subgroup * IQ36_TOKEN_TILE + lane,
              hot_value, current_value, cold_value,
              cold_value_scale_bytes));
          value1[row] = convert_half_rte(iq36_partial_load_value(
              batch, kv_head, block_token + 8U + row,
              past_tokens, attention_cold_tokens,
              subgroup * IQ36_TOKEN_TILE + lane,
              hot_value, current_value, cold_value,
              cold_value_scale_bytes));
          value2[row] = convert_half_rte(iq36_partial_load_value(
              batch, kv_head, block_token + row, past_tokens,
              attention_cold_tokens,
              128U + subgroup * IQ36_TOKEN_TILE + lane,
              hot_value, current_value, cold_value,
              cold_value_scale_bytes));
          value3[row] = convert_half_rte(iq36_partial_load_value(
              batch, kv_head, block_token + 8U + row,
              past_tokens, attention_cold_tokens,
              128U + subgroup * IQ36_TOKEN_TILE + lane,
              hot_value, current_value, cold_value,
              cold_value_scale_bytes));
        }
      }
      const int8 value_fragment0 = (int8)(
          as_int((half2)(value0[0], value0[1])),
          as_int((half2)(value0[2], value0[3])),
          as_int((half2)(value0[4], value0[5])),
          as_int((half2)(value0[6], value0[7])),
          as_int((half2)(value1[0], value1[1])),
          as_int((half2)(value1[2], value1[3])),
          as_int((half2)(value1[4], value1[5])),
          as_int((half2)(value1[6], value1[7])));
      const int8 value_fragment1 = (int8)(
          as_int((half2)(value2[0], value2[1])),
          as_int((half2)(value2[2], value2[3])),
          as_int((half2)(value2[4], value2[5])),
          as_int((half2)(value2[6], value2[7])),
          as_int((half2)(value3[0], value3[1])),
          as_int((half2)(value3[2], value3[3])),
          as_int((half2)(value3[4], value3[5])),
          as_int((half2)(value3[6], value3[7])));
      #pragma unroll
      for (uint group = 0U;
           group < IQ36_PREFILL_QUERY_GROUPS; ++group) {
        const uint query_group = group * IQ36_GQA_GROUP;
        const short8 weight_fragment = (short8)(
            as_short(local_score_weight[(query_group + 0U) *
                IQ36_PREFILL_CHUNK_TOKENS + token_in_chunk]),
            as_short(local_score_weight[(query_group + 1U) *
                IQ36_PREFILL_CHUNK_TOKENS + token_in_chunk]),
            as_short(local_score_weight[(query_group + 2U) *
                IQ36_PREFILL_CHUNK_TOKENS + token_in_chunk]),
            as_short(local_score_weight[(query_group + 3U) *
                IQ36_PREFILL_CHUNK_TOKENS + token_in_chunk]),
            as_short(local_score_weight[(query_group + 4U) *
                IQ36_PREFILL_CHUNK_TOKENS + token_in_chunk]),
            as_short(local_score_weight[(query_group + 5U) *
                IQ36_PREFILL_CHUNK_TOKENS + token_in_chunk]),
            as_short(local_score_weight[(query_group + 6U) *
                IQ36_PREFILL_CHUNK_TOKENS + token_in_chunk]),
            as_short(local_score_weight[(query_group + 7U) *
                IQ36_PREFILL_CHUNK_TOKENS + token_in_chunk]));
        chunk_output0[group] = intel_sub_group_f16_f16_matrix_mad_k16(
            weight_fragment, value_fragment0, chunk_output0[group]);
        chunk_output1[group] = intel_sub_group_f16_f16_matrix_mad_k16(
            weight_fragment, value_fragment1, chunk_output1[group]);
      }
    }

    #pragma unroll
    for (uint group = 0U; group < IQ36_PREFILL_QUERY_GROUPS; ++group) {
      const uint query_group = group * IQ36_GQA_GROUP;
      const float8 previous_scale = (float8)(
          local_previous_scale[query_group + 0U],
          local_previous_scale[query_group + 1U],
          local_previous_scale[query_group + 2U],
          local_previous_scale[query_group + 3U],
          local_previous_scale[query_group + 4U],
          local_previous_scale[query_group + 5U],
          local_previous_scale[query_group + 6U],
          local_previous_scale[query_group + 7U]);
      output_accumulator0[group] =
          output_accumulator0[group] * previous_scale +
          chunk_output0[group];
      output_accumulator1[group] =
          output_accumulator1[group] * previous_scale +
          chunk_output1[group];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  #pragma unroll
  for (uint group = 0U; group < IQ36_PREFILL_QUERY_GROUPS; ++group) {
    #pragma unroll
    for (uint row = 0U; row < IQ36_GQA_GROUP; ++row) {
      const uint query_in_tile = group * IQ36_GQA_GROUP + row;
      if (query_in_tile < query_count) {
        const float reciprocal = native_recip(
            local_running_sum[query_in_tile]);
        const float value0 = local_running_sum[query_in_tile] == 0.0f
            ? 0.0f : output_accumulator0[group][row] * reciprocal;
        const float value1 = local_running_sum[query_in_tile] == 0.0f
            ? 0.0f : output_accumulator1[group][row] * reciprocal;
#if defined(IQ36_TOKEN_MAJOR_OUTPUT)
        const uint query_position = query_begin + query_in_tile;
        const ulong output0_index = OUTPUT1_OFFSET +
            (ulong)batch * OUTPUT1_PITCHES[0] +
            (ulong)query_position * OUTPUT1_PITCHES[1] +
            (ulong)query_head * OUTPUT1_PITCHES[2] +
            (ulong)dim * OUTPUT1_PITCHES[3];
        #if defined(IQ36_FUSED_GATE_OUTPUT)
        const ulong gate0_index = INPUT13_OFFSET +
            (ulong)batch * INPUT13_PITCHES[0] +
            (ulong)query_position * INPUT13_PITCHES[1] +
            (ulong)query_head * INPUT13_PITCHES[2] +
            (ulong)dim * INPUT13_PITCHES[3];
        output[output0_index] = iq36_gated_attention_value(
            value0, raw_gate, gate0_index);
        output[output0_index + 128U * OUTPUT1_PITCHES[3]] =
            iq36_gated_attention_value(
                value1, raw_gate,
                gate0_index + 128U * INPUT13_PITCHES[3]);
        #else
        output[output0_index] = (OUTPUT1_TYPE)value0;
        output[output0_index + 128U * OUTPUT1_PITCHES[3]] =
            (OUTPUT1_TYPE)value1;
        #endif
#else
        output[output_tile +
            (ulong)query_in_tile * IQ36_HEAD_DIM + dim] =
                (IQ36_PREFILL_ATTENTION_TYPE)value0;
        output[output_tile +
            (ulong)query_in_tile * IQ36_HEAD_DIM + 128U + dim] =
                (IQ36_PREFILL_ATTENTION_TYPE)value1;
#endif
      }
    }
  }
#endif

  // Exactly one query-head work-group owns each KV state and codec row.
  if ((query_head % IQ36_GQA_GROUP) == 0U) {
    for (uint query_in_tile = 0U; query_in_tile < query_count;
         ++query_in_tile) {
      const uint token = query_begin + query_in_tile;
      const uint global_token = past_tokens + token;
      const ulong current_row_key = INPUT3_OFFSET +
          (ulong)batch * INPUT3_PITCHES[0] +
          (ulong)kv_head * INPUT3_PITCHES[1] +
          (ulong)token * INPUT3_PITCHES[2];
      const ulong current_row_value = iq36_current_value_index(
          batch, kv_head, token, 0U);

      if (full_prefill_history || global_token < IQ36_SINK_TOKENS ||
          global_token + IQ36_HOT_WINDOW >= key_tokens) {
        const uint slot = iq36_hot_slot(global_token);
        for (uint state_dim = local_id; state_dim < IQ36_HEAD_DIM;
             state_dim += 128U) {
          const ulong hot_value_index = INPUT2_OFFSET +
              (ulong)batch * INPUT2_PITCHES[0] +
              (ulong)kv_head * INPUT2_PITCHES[1] +
              (ulong)slot * INPUT2_PITCHES[2] +
              (ulong)state_dim * INPUT2_PITCHES[3];
          hot_value[hot_value_index] = (INPUT2_TYPE)
              current_value[current_row_value +
                  (ulong)state_dim * INPUT4_PITCHES[3]];
#if defined(IQ36_DIMENSION_MAJOR_VALUE_PLANE)
          iq36_direct_store_hot_value_dimension(
              batch, kv_head, slot, state_dim,
              (half)current_value[current_row_value +
                  (ulong)state_dim * INPUT4_PITCHES[3]],
              hot_key_bits);
#endif
        }
        if (local_id < IQ36_HEAD_DIM / 2U) {
          const uint key_dim = local_id * 2U;
          const uint key_word = local_id * IQ36_KEY_TILE_TOKENS +
              (slot & (IQ36_KEY_TILE_TOKENS - 1U));
          const ulong hot_key_index = INPUT1_OFFSET +
              (ulong)batch * INPUT1_PITCHES[0] +
              (ulong)kv_head * INPUT1_PITCHES[1] +
              (ulong)(slot / IQ36_KEY_TILE_TOKENS) * INPUT1_PITCHES[2] +
              (ulong)key_word * INPUT1_PITCHES[3];
          hot_key_bits[hot_key_index] = (INPUT1_TYPE)as_int((half2)(
              (half)current_key[current_row_key +
                  (ulong)key_dim * INPUT3_PITCHES[3]],
              (half)current_key[current_row_key +
                  (ulong)(key_dim + 1U) * INPUT3_PITCHES[3]]));
          __global half* dense_hot_key = (__global half*)&hot_key_bits[
              iq36_hot_key_dense_i32_base(batch, kv_head)];
          const ulong dense_key_index =
              (ulong)slot * IQ36_HEAD_DIM + key_dim;
          dense_hot_key[dense_key_index] = (half)current_key[
              current_row_key + (ulong)key_dim * INPUT3_PITCHES[3]];
          dense_hot_key[dense_key_index + 1U] = (half)current_key[
              current_row_key +
              (ulong)(key_dim + 1U) * INPUT3_PITCHES[3]];
        }
      }

      if (token < cold_append_tokens) {
#if defined(IQ36_DIRECT_I8_GROUP4_FULL_COLD)
        const uint evicted_token = cold_tokens + token;
        #pragma unroll
        for (uint pass = 0U; pass < 2U; ++pass) {
          const uint state_dim = pass * 128U + subgroup * 16U + lane;
          const uint key_scale_group =
              state_dim / IQ36_KEY_QUANT_GROUP;
          const uint value_scale_group =
              state_dim / IQ36_VALUE_QUANT_GROUP;
          const float key_value = iq36_partial_load_key(
              batch, kv_head, evicted_token, past_tokens,
              attention_cold_tokens,
              state_dim, hot_key_bits, current_key, cold_key,
              cold_key_scale_bytes);
          const float value_value = iq36_partial_load_value(
              batch, kv_head, evicted_token, past_tokens,
              attention_cold_tokens,
              state_dim, hot_value, current_value, cold_value,
              cold_value_scale_bytes);
          float key_max = fabs(key_value);
          key_max = fmax(key_max, sub_group_shuffle_xor(key_max, 1U));
#if IQ36_KEY_QUANT_GROUP == 4
          key_max = fmax(key_max, sub_group_shuffle_xor(key_max, 2U));
#endif
          float value_max = fabs(value_value);
          value_max = fmax(
              value_max, sub_group_shuffle_xor(value_max, 1U));
          value_max = fmax(
              value_max, sub_group_shuffle_xor(value_max, 2U));
          const float key_scale = key_max == 0.0f
              ? 1.0f : key_max / 127.0f;
          const float value_scale = value_max == 0.0f
              ? 1.0f : value_max / 127.0f;
          const int key_quantized = clamp(
              (int)rint(key_value / key_scale), -127, 127);
          const int value_quantized = clamp(
              (int)rint(value_value / value_scale), -127, 127);
          const ulong cold_key_index = IQ36_PREFILL_COLD_KEY_OFFSET +
              (ulong)batch * IQ36_PREFILL_COLD_KEY_PITCHES[0] +
              (ulong)kv_head * IQ36_PREFILL_COLD_KEY_PITCHES[1] +
              (ulong)token * IQ36_PREFILL_COLD_KEY_PITCHES[2] +
              (ulong)state_dim * IQ36_PREFILL_COLD_KEY_PITCHES[3];
          const ulong cold_value_index = IQ36_PREFILL_COLD_VALUE_OFFSET +
              (ulong)batch * IQ36_PREFILL_COLD_VALUE_PITCHES[0] +
              (ulong)kv_head * IQ36_PREFILL_COLD_VALUE_PITCHES[1] +
              (ulong)token * IQ36_PREFILL_COLD_VALUE_PITCHES[2] +
              (ulong)state_dim * IQ36_PREFILL_COLD_VALUE_PITCHES[3];
          cold_key_append[cold_key_index] =
              (IQ36_PREFILL_COLD_KEY_TYPE)key_quantized;
          cold_value_append[cold_value_index] =
              (IQ36_PREFILL_COLD_VALUE_TYPE)value_quantized;
          if (fixed_cold_state) {
            iq36_direct_store_cold_key(
                batch, kv_head, evicted_token, state_dim,
                (char)key_quantized, cold_key);
            iq36_direct_store_cold_value(
                batch, kv_head, evicted_token, state_dim,
                (char)value_quantized, cold_value);
          }
          if ((lane % IQ36_KEY_QUANT_GROUP) == 0U) {
            const ushort key_bits = as_ushort(
                convert_half_rte(key_scale));
            const uint scale_x = key_scale_group * 2U;
            const ulong key_scale_index = IQ36_PREFILL_KEY_SCALE_OFFSET +
                (ulong)batch * IQ36_PREFILL_KEY_SCALE_PITCHES[0] +
                (ulong)kv_head * IQ36_PREFILL_KEY_SCALE_PITCHES[1] +
                (ulong)token * IQ36_PREFILL_KEY_SCALE_PITCHES[2] +
                (ulong)scale_x * IQ36_PREFILL_KEY_SCALE_PITCHES[3];
            cold_key_scale_append[key_scale_index] =
                (IQ36_PREFILL_KEY_SCALE_TYPE)(key_bits & 0xffU);
            cold_key_scale_append[key_scale_index +
                IQ36_PREFILL_KEY_SCALE_PITCHES[3]] =
                    (IQ36_PREFILL_KEY_SCALE_TYPE)(key_bits >> 8);
            if (fixed_cold_state) {
              iq36_direct_store_cold_key_scale(
                  batch, kv_head, evicted_token, key_scale_group,
                  as_half(key_bits), cold_key_scale_bytes);
            }
          }
          if ((lane % IQ36_VALUE_QUANT_GROUP) == 0U) {
            const ushort value_bits = as_ushort(
                convert_half_rte(value_scale));
            const uint scale_x = value_scale_group * 2U;
            const ulong value_scale_index = IQ36_PREFILL_VALUE_SCALE_OFFSET +
                (ulong)batch * IQ36_PREFILL_VALUE_SCALE_PITCHES[0] +
                (ulong)kv_head * IQ36_PREFILL_VALUE_SCALE_PITCHES[1] +
                (ulong)token * IQ36_PREFILL_VALUE_SCALE_PITCHES[2] +
                (ulong)scale_x * IQ36_PREFILL_VALUE_SCALE_PITCHES[3];
            cold_value_scale_append[value_scale_index] =
                (IQ36_PREFILL_VALUE_SCALE_TYPE)(value_bits & 0xffU);
            cold_value_scale_append[value_scale_index +
                IQ36_PREFILL_VALUE_SCALE_PITCHES[3]] =
                    (IQ36_PREFILL_VALUE_SCALE_TYPE)(value_bits >> 8);
            if (fixed_cold_state) {
              iq36_direct_store_cold_value_scale(
                  batch, kv_head, evicted_token, value_scale_group,
                  as_half(value_bits), cold_value_scale_bytes);
            }
          }
        }
#else
        const uint block = subgroup;
        const uint dim0 = block * 32U + lane * 2U;
        const uint dim1 = dim0 + 1U;
        const uint evicted_token = cold_tokens + token;
        const float key0 = iq36_partial_load_key(
            batch, kv_head, evicted_token, past_tokens,
            attention_cold_tokens,
            dim0, hot_key_bits, current_key, cold_key,
            cold_key_scale_bytes);
        const float key1 = iq36_partial_load_key(
            batch, kv_head, evicted_token, past_tokens,
            attention_cold_tokens,
            dim1, hot_key_bits, current_key, cold_key,
            cold_key_scale_bytes);
        const float value0 = iq36_partial_load_value(
            batch, kv_head, evicted_token, past_tokens,
            attention_cold_tokens,
            dim0, hot_value, current_value, cold_value,
            cold_value_scale_bytes);
        const float value1 = iq36_partial_load_value(
            batch, kv_head, evicted_token, past_tokens,
            attention_cold_tokens,
            dim1, hot_value, current_value, cold_value,
            cold_value_scale_bytes);
        const float key_max = sub_group_reduce_max(
            fmax(fabs(key0), fabs(key1)));
        float value_max = fmax(fabs(value0), fabs(value1));
#if IQ36_VALUE_QUANT_GROUP == 16U
        value_max = fmax(
            value_max, sub_group_shuffle_xor(value_max, 1U));
        value_max = fmax(
            value_max, sub_group_shuffle_xor(value_max, 2U));
        value_max = fmax(
            value_max, sub_group_shuffle_xor(value_max, 4U));
#else
        value_max = sub_group_reduce_max(value_max);
#endif
        const float key_scale = key_max == 0.0f
            ? 1.0f : key_max / (float)IQ36_KEY_QUANT_MAX;
        const float value_scale = value_max == 0.0f
            ? 1.0f : value_max / (float)IQ36_VALUE_QUANT_MAX;
#if defined(IQ36_KEY_RESIDUAL1)
        const half key_scale_stored = convert_half_rte(key_scale);
        const int key_fine0 = iq36_residual1_fine_quantize(
            key0, key_scale_stored);
        const int key_fine1 = iq36_residual1_fine_quantize(
            key1, key_scale_stored);
        const int key_q0 = iq36_residual1_base(key_fine0);
        const int key_q1 = iq36_residual1_base(key_fine1);
        const uint key_residual_bit0 = iq36_residual1_bit(
            key_fine0, key_q0);
        const uint key_residual_bit1 = iq36_residual1_bit(
            key_fine1, key_q1);
        const uint key_residual_word = sub_group_reduce_add(
            (key_residual_bit0 << (lane * 2U)) |
            (key_residual_bit1 << (lane * 2U + 1U)));
#else
        const int key_q0 = clamp(
            (int)rint(key0 / key_scale),
            -(int)IQ36_KEY_QUANT_MAX, (int)IQ36_KEY_QUANT_MAX);
        const int key_q1 = clamp(
            (int)rint(key1 / key_scale),
            -(int)IQ36_KEY_QUANT_MAX, (int)IQ36_KEY_QUANT_MAX);
#endif
#if defined(IQ36_VALUE_RESIDUAL1)
        const half value_scale_stored = convert_half_rte(value_scale);
        const int value_fine0 = iq36_residual1_fine_quantize(
            value0, value_scale_stored);
        const int value_fine1 = iq36_residual1_fine_quantize(
            value1, value_scale_stored);
        const int value_q0 = iq36_residual1_base(value_fine0);
        const int value_q1 = iq36_residual1_base(value_fine1);
        const uint value_residual_bit0 = iq36_residual1_bit(
            value_fine0, value_q0);
        const uint value_residual_bit1 = iq36_residual1_bit(
            value_fine1, value_q1);
        const uint value_residual_word = sub_group_reduce_add(
            (value_residual_bit0 << (lane * 2U)) |
            (value_residual_bit1 << (lane * 2U + 1U)));
#else
        const int value_q0 = clamp(
            (int)rint(value0 / value_scale),
            -(int)IQ36_VALUE_QUANT_MAX, (int)IQ36_VALUE_QUANT_MAX);
        const int value_q1 = clamp(
            (int)rint(value1 / value_scale),
            -(int)IQ36_VALUE_QUANT_MAX, (int)IQ36_VALUE_QUANT_MAX);
#endif
        const ulong cold_key_base = IQ36_PREFILL_COLD_KEY_OFFSET +
            (ulong)batch * IQ36_PREFILL_COLD_KEY_PITCHES[0] +
            (ulong)kv_head * IQ36_PREFILL_COLD_KEY_PITCHES[1] +
            (ulong)token * IQ36_PREFILL_COLD_KEY_PITCHES[2];
        const ulong cold_value_base = IQ36_PREFILL_COLD_VALUE_OFFSET +
            (ulong)batch * IQ36_PREFILL_COLD_VALUE_PITCHES[0] +
            (ulong)kv_head * IQ36_PREFILL_COLD_VALUE_PITCHES[1] +
            (ulong)token * IQ36_PREFILL_COLD_VALUE_PITCHES[2];
#if defined(IQ36_ADAPTIVE_PACKED_KV)
        #pragma unroll
        for (uint word = 0U; word < IQ36_KEY_QUANT_BITS; ++word) {
          const uint packed = iq36_subgroup_pack_two_codes(
              key_q0, key_q1, IQ36_KEY_QUANT_BITS, word);
          if (lane < 4U) {
            cold_key_append[cold_key_base +
                (ulong)((block * IQ36_KEY_QUANT_BITS + word) * 4U + lane) *
                    IQ36_PREFILL_COLD_KEY_PITCHES[3]] =
                (IQ36_PREFILL_COLD_KEY_TYPE)(packed >> (lane * 8U));
          }
          if (fixed_cold_state && lane == 0U) {
            const uint direct_token = cold_tokens + token;
            iq36_direct_store_cold_key_packed_word(
                batch, kv_head, direct_token, block, word, packed, cold_key);
          }
        }
        #pragma unroll
        for (uint word = 0U; word < IQ36_VALUE_QUANT_BITS; ++word) {
          const uint packed = iq36_subgroup_pack_two_codes(
              value_q0, value_q1, IQ36_VALUE_QUANT_BITS, word);
          if (lane < 4U) {
            cold_value_append[cold_value_base +
                (ulong)((block * IQ36_VALUE_QUANT_BITS + word) * 4U + lane) *
                    IQ36_PREFILL_COLD_VALUE_PITCHES[3]] =
                (IQ36_PREFILL_COLD_VALUE_TYPE)(packed >> (lane * 8U));
          }
#if !defined(IQ36_ADAPTIVE_DIMENSION_MAJOR_V8)
          if (fixed_cold_state && lane == 0U) {
            const uint direct_token = cold_tokens + token;
            iq36_direct_store_cold_value_packed_word(
                batch, kv_head, direct_token, block, word, packed, cold_value);
          }
#endif
        }
#if defined(IQ36_ADAPTIVE_DIMENSION_MAJOR_V8)
        if (fixed_cold_state) {
          const uint direct_token = cold_tokens + token;
          iq36_direct_store_cold_value(
              batch, kv_head, direct_token, dim0,
              (char)value_q0, cold_value);
          iq36_direct_store_cold_value(
              batch, kv_head, direct_token, dim1,
              (char)value_q1, cold_value);
        }
#endif
#else
        cold_key_append[cold_key_base +
            (ulong)dim0 * IQ36_PREFILL_COLD_KEY_PITCHES[3]] =
                (IQ36_PREFILL_COLD_KEY_TYPE)key_q0;
        cold_key_append[cold_key_base +
            (ulong)dim1 * IQ36_PREFILL_COLD_KEY_PITCHES[3]] =
                (IQ36_PREFILL_COLD_KEY_TYPE)key_q1;
        cold_value_append[cold_value_base +
            (ulong)dim0 * IQ36_PREFILL_COLD_VALUE_PITCHES[3]] =
                (IQ36_PREFILL_COLD_VALUE_TYPE)value_q0;
        cold_value_append[cold_value_base +
            (ulong)dim1 * IQ36_PREFILL_COLD_VALUE_PITCHES[3]] =
                (IQ36_PREFILL_COLD_VALUE_TYPE)value_q1;
#endif
        if (fixed_cold_state) {
#if defined(IQ36_DIRECT_I8_FIXED_LAYOUT)
#if !defined(IQ36_ADAPTIVE_PACKED_KV)
          const uint direct_token = cold_tokens + token;
          iq36_direct_store_cold_key(
              batch, kv_head, direct_token, dim0, (char)key_q0, cold_key);
          iq36_direct_store_cold_key(
              batch, kv_head, direct_token, dim1, (char)key_q1, cold_key);
          iq36_direct_store_cold_value(
              batch, kv_head, direct_token, dim0,
              (char)value_q0, cold_value);
          iq36_direct_store_cold_value(
              batch, kv_head, direct_token, dim1,
              (char)value_q1, cold_value);
#endif
#if defined(IQ36_KEY_RESIDUAL1)
          if (lane == 0U) {
            iq36_direct_store_cold_key_residual1_word(
                batch, kv_head, direct_token, block,
                key_residual_word, cold_key_scale_bytes);
          }
#endif
#if defined(IQ36_VALUE_RESIDUAL1)
          iq36_direct_store_cold_value_residual1_bit(
              batch, kv_head, direct_token, dim0,
              value_residual_bit0, cold_value_scale_bytes);
          iq36_direct_store_cold_value_residual1_bit(
              batch, kv_head, direct_token, dim1,
              value_residual_bit1, cold_value_scale_bytes);
#endif
#else
          const uint state_row = cold_tokens + token + 1U;
          const ulong fixed_key_base = INPUT5_OFFSET +
              (ulong)batch * INPUT5_PITCHES[0] +
              (ulong)kv_head * INPUT5_PITCHES[1] +
              (ulong)state_row * INPUT5_PITCHES[2];
          const ulong fixed_value_base = INPUT6_OFFSET +
              (ulong)batch * INPUT6_PITCHES[0] +
              (ulong)kv_head * INPUT6_PITCHES[1] +
              (ulong)state_row * INPUT6_PITCHES[2];
          cold_key[fixed_key_base +
              (ulong)dim0 * INPUT5_PITCHES[3]] = (INPUT5_TYPE)key_q0;
          cold_key[fixed_key_base +
              (ulong)dim1 * INPUT5_PITCHES[3]] = (INPUT5_TYPE)key_q1;
          cold_value[fixed_value_base +
              (ulong)dim0 * INPUT6_PITCHES[3]] = (INPUT6_TYPE)value_q0;
          cold_value[fixed_value_base +
              (ulong)dim1 * INPUT6_PITCHES[3]] = (INPUT6_TYPE)value_q1;
#endif
        }
#if defined(IQ36_KEY_RESIDUAL1)
        if (lane < 4U) {
          const ulong residual_index = IQ36_PREFILL_KEY_SCALE_OFFSET +
              (ulong)batch * IQ36_PREFILL_KEY_SCALE_PITCHES[0] +
              (ulong)kv_head * IQ36_PREFILL_KEY_SCALE_PITCHES[1] +
              (ulong)token * IQ36_PREFILL_KEY_SCALE_PITCHES[2] +
              (ulong)(IQ36_KEY_SCALE_BYTES + block * 4U + lane) *
                  IQ36_PREFILL_KEY_SCALE_PITCHES[3];
          cold_key_scale_append[residual_index] =
              (IQ36_PREFILL_KEY_SCALE_TYPE)(
                  key_residual_word >> (lane * 8U));
        }
#endif
#if defined(IQ36_VALUE_RESIDUAL1)
        if (lane < 4U) {
          const ulong residual_index = IQ36_PREFILL_VALUE_SCALE_OFFSET +
              (ulong)batch * IQ36_PREFILL_VALUE_SCALE_PITCHES[0] +
              (ulong)kv_head * IQ36_PREFILL_VALUE_SCALE_PITCHES[1] +
              (ulong)token * IQ36_PREFILL_VALUE_SCALE_PITCHES[2] +
              (ulong)(IQ36_VALUE_SCALE_BYTES + block * 4U + lane) *
                  IQ36_PREFILL_VALUE_SCALE_PITCHES[3];
          cold_value_scale_append[residual_index] =
              (IQ36_PREFILL_VALUE_SCALE_TYPE)(
                  value_residual_word >> (lane * 8U));
        }
#endif
        if (lane < 2U) {
          const ushort key_bits = as_ushort(convert_half_rte(key_scale));
          const uint scale_x = block * 2U + lane;
          const ulong key_scale_index = IQ36_PREFILL_KEY_SCALE_OFFSET +
              (ulong)batch * IQ36_PREFILL_KEY_SCALE_PITCHES[0] +
              (ulong)kv_head * IQ36_PREFILL_KEY_SCALE_PITCHES[1] +
              (ulong)token * IQ36_PREFILL_KEY_SCALE_PITCHES[2] +
              (ulong)scale_x * IQ36_PREFILL_KEY_SCALE_PITCHES[3];
          cold_key_scale_append[key_scale_index] =
              (IQ36_PREFILL_KEY_SCALE_TYPE)(
              lane == 0U ? key_bits & 0xffU : key_bits >> 8);
          if (fixed_cold_state) {
#if defined(IQ36_DIRECT_I8_FIXED_LAYOUT)
            if (lane == 0U) {
              const uint direct_token = cold_tokens + token;
              iq36_direct_store_cold_key_scale(
                  batch, kv_head, direct_token, block,
                  as_half(key_bits), cold_key_scale_bytes);
            }
#else
            const uint state_row = cold_tokens + token + 1U;
            const ulong fixed_key_scale = INPUT7_OFFSET +
                (ulong)batch * INPUT7_PITCHES[0] +
                (ulong)kv_head * INPUT7_PITCHES[1] +
                (ulong)state_row * INPUT7_PITCHES[2] +
                (ulong)scale_x * INPUT7_PITCHES[3];
            cold_key_scale_bytes[fixed_key_scale] = (INPUT7_TYPE)(
                lane == 0U ? key_bits & 0xffU : key_bits >> 8);
#endif
          }
        }
        const uint value_scale_group =
            block * (32U / IQ36_VALUE_QUANT_GROUP) +
            lane / (IQ36_VALUE_QUANT_GROUP / 2U);
        const uint value_scale_byte_lane =
            lane % (IQ36_VALUE_QUANT_GROUP / 2U);
        if (value_scale_byte_lane < 2U) {
          const ushort value_bits = as_ushort(
              convert_half_rte(value_scale));
          const uint scale_x = value_scale_group * 2U +
              value_scale_byte_lane;
          const ulong value_scale_index = IQ36_PREFILL_VALUE_SCALE_OFFSET +
              (ulong)batch * IQ36_PREFILL_VALUE_SCALE_PITCHES[0] +
              (ulong)kv_head * IQ36_PREFILL_VALUE_SCALE_PITCHES[1] +
              (ulong)token * IQ36_PREFILL_VALUE_SCALE_PITCHES[2] +
              (ulong)scale_x * IQ36_PREFILL_VALUE_SCALE_PITCHES[3];
          cold_value_scale_append[value_scale_index] =
              (IQ36_PREFILL_VALUE_SCALE_TYPE)(
                  value_scale_byte_lane == 0U
                      ? value_bits & 0xffU : value_bits >> 8);
          if (fixed_cold_state) {
#if defined(IQ36_DIRECT_I8_FIXED_LAYOUT)
            if (value_scale_byte_lane == 0U) {
              const uint direct_token = cold_tokens + token;
              iq36_direct_store_cold_value_scale(
                  batch, kv_head, direct_token, value_scale_group,
                  as_half(value_bits), cold_value_scale_bytes);
            }
#else
            const uint state_row = cold_tokens + token + 1U;
            const ulong fixed_value_scale = INPUT8_OFFSET +
                (ulong)batch * INPUT8_PITCHES[0] +
                (ulong)kv_head * INPUT8_PITCHES[1] +
                (ulong)state_row * INPUT8_PITCHES[2] +
                (ulong)scale_x * INPUT8_PITCHES[3];
            cold_value_scale_bytes[fixed_value_scale] = (INPUT8_TYPE)(
                value_scale_byte_lane == 0U
                    ? value_bits & 0xffU : value_bits >> 8);
#endif
          }
        }
#endif
      }
    }
    if (fixed_cold_state && query_tile == 0U && local_id < 3U) {
      const uint divisor = local_id == 0U ? 1U :
          (local_id == 1U ? 128U : 16384U);
      cold_key[INPUT5_OFFSET +
          (ulong)batch * INPUT5_PITCHES[0] +
          (ulong)kv_head * INPUT5_PITCHES[1] +
          (ulong)local_id * INPUT5_PITCHES[3]] =
              (INPUT5_TYPE)((desired_cold_tokens / divisor) % 128U);
    }
  }
}

#endif
