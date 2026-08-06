#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable

short iq36_pack_i8_pair(char low, char high) {
  const ushort bits = (ushort)(uchar)low | ((ushort)(uchar)high << 8);
  return as_short(bits);
}

short8 iq36_load_rows8(
    __global const char *source,
    uint row_start,
    uint row_end,
    uint group,
    uint lane) {
  short8 packed = (short8)(0);
  if (lane >= 16U) return packed;
  const uint column = group * 32U + lane * 2U;
#define IQ36_LOAD_ROW(component, offset)                                      \
  do {                                                                         \
    const uint row = row_start + (offset);                                     \
    if (row < row_end) {                                                       \
      packed.component = iq36_pack_i8_pair(                                   \
          source[row * 512U + column], source[row * 512U + column + 1U]);      \
    }                                                                          \
  } while (0)
  IQ36_LOAD_ROW(s0, 0U);
  IQ36_LOAD_ROW(s1, 1U);
  IQ36_LOAD_ROW(s2, 2U);
  IQ36_LOAD_ROW(s3, 3U);
  IQ36_LOAD_ROW(s4, 4U);
  IQ36_LOAD_ROW(s5, 5U);
  IQ36_LOAD_ROW(s6, 6U);
  IQ36_LOAD_ROW(s7, 7U);
#undef IQ36_LOAD_ROW
  return packed;
}

short8 iq36_load_rows8_k16(
    __global const char *source,
    uint row_start,
    uint row_end,
    uint group,
    uint lane) {
  short8 packed = (short8)(0);
  if (lane >= 8U) return packed;
  const uint column = group * 16U + lane * 2U;
#define IQ36_LOAD_ROW_K16(component, offset)                                  \
  do {                                                                         \
    const uint row = row_start + (offset);                                     \
    if (row < row_end) {                                                       \
      packed.component = iq36_pack_i8_pair(                                   \
          source[row * 512U + column], source[row * 512U + column + 1U]);      \
    }                                                                          \
  } while (0)
  IQ36_LOAD_ROW_K16(s0, 0U);
  IQ36_LOAD_ROW_K16(s1, 1U);
  IQ36_LOAD_ROW_K16(s2, 2U);
  IQ36_LOAD_ROW_K16(s3, 3U);
  IQ36_LOAD_ROW_K16(s4, 4U);
  IQ36_LOAD_ROW_K16(s5, 5U);
  IQ36_LOAD_ROW_K16(s6, 6U);
  IQ36_LOAD_ROW_K16(s7, 7U);
#undef IQ36_LOAD_ROW_K16
  return packed;
}

int8 iq36_load_sums8(
    __global const short *source_sums,
    uint row_start,
    uint row_end,
    uint group) {
  int8 values = (int8)(0);
#define IQ36_LOAD_SUM(component, offset)                                      \
  do {                                                                         \
    const uint row = row_start + (offset);                                     \
    if (row < row_end) values.component = source_sums[row * 16U + group];      \
  } while (0)
  IQ36_LOAD_SUM(s0, 0U);
  IQ36_LOAD_SUM(s1, 1U);
  IQ36_LOAD_SUM(s2, 2U);
  IQ36_LOAD_SUM(s3, 3U);
  IQ36_LOAD_SUM(s4, 4U);
  IQ36_LOAD_SUM(s5, 5U);
  IQ36_LOAD_SUM(s6, 6U);
  IQ36_LOAD_SUM(s7, 7U);
#undef IQ36_LOAD_SUM
  return values;
}

int8 iq36_load_sums8_k16(
    __global const short *source_sums,
    uint row_start,
    uint row_end,
    uint group) {
  int8 values = (int8)(0);
#define IQ36_LOAD_SUM_K16(component, offset)                                  \
  do {                                                                         \
    const uint row = row_start + (offset);                                     \
    if (row < row_end) values.component = source_sums[row * 32U + group];      \
  } while (0)
  IQ36_LOAD_SUM_K16(s0, 0U);
  IQ36_LOAD_SUM_K16(s1, 1U);
  IQ36_LOAD_SUM_K16(s2, 2U);
  IQ36_LOAD_SUM_K16(s3, 3U);
  IQ36_LOAD_SUM_K16(s4, 4U);
  IQ36_LOAD_SUM_K16(s5, 5U);
  IQ36_LOAD_SUM_K16(s6, 6U);
  IQ36_LOAD_SUM_K16(s7, 7U);
#undef IQ36_LOAD_SUM_K16
  return values;
}

float8 iq36_load_source_scales8(
    __global const float *source_scales,
    uint row_start,
    uint row_end,
    uint block) {
  float8 values = (float8)(0.0f);
#define IQ36_LOAD_SCALE(component, offset)                                    \
  do {                                                                         \
    const uint row = row_start + (offset);                                     \
    if (row < row_end) values.component = source_scales[row * 2U + block];     \
  } while (0)
  IQ36_LOAD_SCALE(s0, 0U);
  IQ36_LOAD_SCALE(s1, 1U);
  IQ36_LOAD_SCALE(s2, 2U);
  IQ36_LOAD_SCALE(s3, 3U);
  IQ36_LOAD_SCALE(s4, 4U);
  IQ36_LOAD_SCALE(s5, 5U);
  IQ36_LOAD_SCALE(s6, 6U);
  IQ36_LOAD_SCALE(s7, 7U);
#undef IQ36_LOAD_SCALE
  return values;
}

void iq36_store_rows8(
    __global half *output,
    __global const float *router_weights,
    uint row_start,
    uint row_end,
    uint output_column,
    float8 values) {
#define IQ36_STORE_ROW(component, offset)                                     \
  do {                                                                         \
    const uint row = row_start + (offset);                                     \
    if (row < row_end) {                                                       \
      output[row * 2048U + output_column] = convert_half_rte(                  \
          values.component * router_weights[row]);                             \
    }                                                                          \
  } while (0)
  IQ36_STORE_ROW(s0, 0U);
  IQ36_STORE_ROW(s1, 1U);
  IQ36_STORE_ROW(s2, 2U);
  IQ36_STORE_ROW(s3, 3U);
  IQ36_STORE_ROW(s4, 4U);
  IQ36_STORE_ROW(s5, 5U);
  IQ36_STORE_ROW(s6, 6U);
  IQ36_STORE_ROW(s7, 7U);
#undef IQ36_STORE_ROW
}

void iq36_store_rows8_f32(
    __global float *output,
    __global const float *router_weights,
    uint row_start,
    uint row_end,
    uint output_column,
    float8 values) {
#define IQ36_STORE_ROW_F32(component, offset)                                 \
  do {                                                                         \
    const uint row = row_start + (offset);                                     \
    if (row < row_end) {                                                       \
      output[row * 2048U + output_column] =                                   \
          values.component * router_weights[row];                             \
    }                                                                          \
  } while (0)
  IQ36_STORE_ROW_F32(s0, 0U);
  IQ36_STORE_ROW_F32(s1, 1U);
  IQ36_STORE_ROW_F32(s2, 2U);
  IQ36_STORE_ROW_F32(s3, 3U);
  IQ36_STORE_ROW_F32(s4, 4U);
  IQ36_STORE_ROW_F32(s5, 5U);
  IQ36_STORE_ROW_F32(s6, 6U);
  IQ36_STORE_ROW_F32(s7, 7U);
#undef IQ36_STORE_ROW_F32
}

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(16, 1, 1)))
__kernel void iq36_grouped_s8_u8_q6_surrogate_down_m32(
    __global const uchar *weights,
    __global const float *weight_scales,
    __global const int4 *work_tiles,
    __global const char *source,
    __global const float *source_scales,
    __global const short *source_sums,
    __global const float *router_weights,
    __global half *output) {
  const uint output_tile = get_group_id(0);
  const uint work_tile = get_group_id(1);
  const uint lane = get_sub_group_local_id();
  const int4 tile = work_tiles[work_tile];
  const uint expert = (uint)tile.x;
  const uint row_start = (uint)tile.y;
  const uint row_end = (uint)tile.z;
  const uint output_column = output_tile * 16U + lane;
  float8 accum0 = (float8)(0.0f);
  float8 accum1 = (float8)(0.0f);
  float8 accum2 = (float8)(0.0f);
  float8 accum3 = (float8)(0.0f);

  for (uint group = 0U; group < 16U; ++group) {
    const ulong weight_index =
        (((((ulong)expert * 128UL + (ulong)output_tile) * 16UL +
           (ulong)group) * 16UL + (ulong)lane) * 32UL);
    const uint8 packed_weight = vload8(
        0, (__global const uint *)(weights + weight_index));
    const ulong scale_index =
        ((((ulong)expert * 128UL + (ulong)output_tile) * 16UL +
          (ulong)group) * 16UL + (ulong)lane);
    const float weight_scale = weight_scales[scale_index];
    const uint block = group >> 3;

#define IQ36_ACCUMULATE(accumulator, row_offset)                              \
    do {                                                                       \
      const uint rows = row_start + (row_offset);                              \
      if (rows < row_end) {                                                    \
        const short8 input = iq36_load_rows8(                                 \
            source, rows, row_end, group, lane);                               \
        int8 dot = intel_sub_group_i8_u8_matrix_mad_k32(                      \
            input, packed_weight, (int8)(0));                                  \
        dot -= iq36_load_sums8(source_sums, rows, row_end, group) * 128;       \
        const float8 scales = iq36_load_source_scales8(                       \
            source_scales, rows, row_end, block);                              \
        accumulator = fma(                                                     \
            convert_float8(dot), scales * weight_scale, accumulator);          \
      }                                                                        \
    } while (0)
    IQ36_ACCUMULATE(accum0, 0U);
    IQ36_ACCUMULATE(accum1, 8U);
    IQ36_ACCUMULATE(accum2, 16U);
    IQ36_ACCUMULATE(accum3, 24U);
#undef IQ36_ACCUMULATE
  }

  iq36_store_rows8(output, router_weights, row_start, row_end,
                    output_column, accum0);
  iq36_store_rows8(output, router_weights, row_start + 8U, row_end,
                    output_column, accum1);
  iq36_store_rows8(output, router_weights, row_start + 16U, row_end,
                    output_column, accum2);
  iq36_store_rows8(output, router_weights, row_start + 24U, row_end,
                    output_column, accum3);
}

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(16, 1, 1)))
__kernel void iq36_grouped_s8_u8_q6_surrogate_down_m32_f16scale(
    __global const uchar *weights,
    __global const half *weight_scales,
    __global const int4 *work_tiles,
    __global const char *source,
    __global const float *source_scales,
    __global const short *source_sums,
    __global const float *router_weights,
    __global half *output) {
  const uint output_tile = get_group_id(0);
  const uint work_tile = get_group_id(1);
  const uint lane = get_sub_group_local_id();
  const int4 tile = work_tiles[work_tile];
  const uint expert = (uint)tile.x;
  const uint row_start = (uint)tile.y;
  const uint row_end = (uint)tile.z;
  const uint output_column = output_tile * 16U + lane;
  float8 accum0 = (float8)(0.0f);
  float8 accum1 = (float8)(0.0f);
  float8 accum2 = (float8)(0.0f);
  float8 accum3 = (float8)(0.0f);

  for (uint group = 0U; group < 16U; ++group) {
    const ulong weight_index =
        (((((ulong)expert * 128UL + (ulong)output_tile) * 16UL +
           (ulong)group) * 16UL + (ulong)lane) * 32UL);
    const uint8 packed_weight = vload8(
        0, (__global const uint *)(weights + weight_index));
    const ulong scale_index =
        ((((ulong)expert * 128UL + (ulong)output_tile) * 16UL +
          (ulong)group) * 16UL + (ulong)lane);
    const float weight_scale = convert_float(weight_scales[scale_index]);
    const uint block = group >> 3;

#define IQ36_ACCUMULATE(accumulator, row_offset)                              \
    do {                                                                       \
      const uint rows = row_start + (row_offset);                              \
      if (rows < row_end) {                                                    \
        const short8 input = iq36_load_rows8(                                 \
            source, rows, row_end, group, lane);                               \
        int8 dot = intel_sub_group_i8_u8_matrix_mad_k32(                      \
            input, packed_weight, (int8)(0));                                  \
        dot -= iq36_load_sums8(source_sums, rows, row_end, group) * 128;       \
        const float8 scales = iq36_load_source_scales8(                       \
            source_scales, rows, row_end, block);                              \
        accumulator = fma(                                                     \
            convert_float8(dot), scales * weight_scale, accumulator);          \
      }                                                                        \
    } while (0)
    IQ36_ACCUMULATE(accum0, 0U);
    IQ36_ACCUMULATE(accum1, 8U);
    IQ36_ACCUMULATE(accum2, 16U);
    IQ36_ACCUMULATE(accum3, 24U);
#undef IQ36_ACCUMULATE
  }

  iq36_store_rows8(output, router_weights, row_start, row_end,
                    output_column, accum0);
  iq36_store_rows8(output, router_weights, row_start + 8U, row_end,
                    output_column, accum1);
  iq36_store_rows8(output, router_weights, row_start + 16U, row_end,
                    output_column, accum2);
  iq36_store_rows8(output, router_weights, row_start + 24U, row_end,
                    output_column, accum3);
}

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(16, 1, 1)))
__kernel void iq36_grouped_s8_u8_q6_surrogate_down_m16_f16scale_compact(
    __global const uchar *weights,
    __global const half *weight_scales,
    __global const int *expert_offsets,
    __global const uint *task_coordinates,
    __global const char *source,
    __global const float *source_scales,
    __global const short *source_sums,
    __global const float *router_weights,
    __global half *output) {
  const uint task = get_group_id(0);
  const uint coordinates = task_coordinates[task];
  const uint expert = coordinates & 255U;
  const uint output_tile = (coordinates >> 8) & 127U;
  const uint m_tile = coordinates >> 15;
  const uint expert_begin = expert == 0U
      ? 0U : (uint)expert_offsets[expert - 1U];
  const uint row_end = (uint)expert_offsets[expert];
  const uint row_start = expert_begin + m_tile * 16U;
  const uint lane = get_sub_group_local_id();
  const uint output_column = output_tile * 16U + lane;
  float8 accum0 = (float8)(0.0f);
  float8 accum1 = (float8)(0.0f);

  for (uint group = 0U; group < 16U; ++group) {
    const ulong weight_index =
        (((((ulong)expert * 128UL + (ulong)output_tile) * 16UL +
           (ulong)group) * 16UL + (ulong)lane) * 32UL);
    const uint8 packed_weight = vload8(
        0, (__global const uint *)(weights + weight_index));
    const ulong scale_index =
        ((((ulong)expert * 128UL + (ulong)output_tile) * 16UL +
          (ulong)group) * 16UL + (ulong)lane);
    const float weight_scale = convert_float(weight_scales[scale_index]);
    const uint block = group >> 3;

#define IQ36_ACCUMULATE_M16(accumulator, row_offset)                          \
    do {                                                                       \
      const uint rows = row_start + (row_offset);                              \
      if (rows < row_end) {                                                    \
        const short8 input = iq36_load_rows8(                                 \
            source, rows, row_end, group, lane);                               \
        int8 dot = intel_sub_group_i8_u8_matrix_mad_k32(                      \
            input, packed_weight, (int8)(0));                                  \
        dot -= iq36_load_sums8(source_sums, rows, row_end, group) * 128;       \
        const float8 scales = iq36_load_source_scales8(                       \
            source_scales, rows, row_end, block);                              \
        accumulator = fma(                                                     \
            convert_float8(dot), scales * weight_scale, accumulator);          \
      }                                                                        \
    } while (0)
    IQ36_ACCUMULATE_M16(accum0, 0U);
    IQ36_ACCUMULATE_M16(accum1, 8U);
#undef IQ36_ACCUMULATE_M16
  }

  iq36_store_rows8(output, router_weights, row_start, row_end,
                    output_column, accum0);
  iq36_store_rows8(output, router_weights, row_start + 8U, row_end,
                    output_column, accum1);
}

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(16, 1, 1)))
__kernel void iq36_grouped_s8_u8_q6_exact_down_m16_compact(
    __global const uchar *weights,
    __global const float *weight_scales,
    __global const int *expert_offsets,
    __global const uint *task_coordinates,
    __global const char *source,
    __global const float *source_scales,
    __global const short *source_sums,
    __global const float *router_weights,
    __global half *output) {
  const uint task = get_group_id(0);
  const uint coordinates = task_coordinates[task];
  const uint expert = coordinates & 255U;
  const uint output_tile = (coordinates >> 8) & 127U;
  const uint m_tile = coordinates >> 15;
  const uint expert_begin = expert == 0U
      ? 0U : (uint)expert_offsets[expert - 1U];
  const uint row_end = (uint)expert_offsets[expert];
  const uint row_start = expert_begin + m_tile * 16U;
  const uint lane = get_sub_group_local_id();
  const uint output_column = output_tile * 16U + lane;
  float8 accum0 = (float8)(0.0f);
  float8 accum1 = (float8)(0.0f);

  for (uint group = 0U; group < 32U; ++group) {
    const ulong weight_index =
        (((((ulong)expert * 128UL + (ulong)output_tile) * 32UL +
           (ulong)group) * 16UL + (ulong)lane) * 16UL);
    const uint4 exact_weight = vload4(
        0, (__global const uint *)(weights + weight_index));
    const uint8 packed_weight = (uint8)(
        exact_weight.s0, exact_weight.s1, exact_weight.s2, exact_weight.s3,
        0x80808080U, 0x80808080U, 0x80808080U, 0x80808080U);
    const ulong scale_index =
        ((((ulong)expert * 128UL + (ulong)output_tile) * 32UL +
          (ulong)group) * 16UL + (ulong)lane);
    const float weight_scale = weight_scales[scale_index];
    const uint block = group >> 4;

#define IQ36_ACCUMULATE_EXACT_K16(accumulator, row_offset)                    \
    do {                                                                       \
      const uint rows = row_start + (row_offset);                              \
      if (rows < row_end) {                                                    \
        const short8 input = iq36_load_rows8_k16(                             \
            source, rows, row_end, group, lane);                               \
        int8 dot = intel_sub_group_i8_u8_matrix_mad_k32(                      \
            input, packed_weight, (int8)(0));                                  \
        dot -= iq36_load_sums8_k16(                                           \
            source_sums, rows, row_end, group) * 128;                          \
        const float8 scales = iq36_load_source_scales8(                       \
            source_scales, rows, row_end, block);                              \
        accumulator = fma(                                                     \
            convert_float8(dot), scales * weight_scale, accumulator);          \
      }                                                                        \
    } while (0)
    IQ36_ACCUMULATE_EXACT_K16(accum0, 0U);
    IQ36_ACCUMULATE_EXACT_K16(accum1, 8U);
#undef IQ36_ACCUMULATE_EXACT_K16
  }

  iq36_store_rows8(output, router_weights, row_start, row_end,
                    output_column, accum0);
  iq36_store_rows8(output, router_weights, row_start + 8U, row_end,
                    output_column, accum1);
}

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(16, 1, 1)))
__kernel void iq36_grouped_s8_u8_q6_exact_down_m16_compact_f32(
    __global const uchar *weights,
    __global const float *weight_scales,
    __global const int *expert_offsets,
    __global const uint *task_coordinates,
    __global const char *source,
    __global const float *source_scales,
    __global const short *source_sums,
    __global const float *router_weights,
    __global float *output) {
  const uint task = get_group_id(0);
  const uint coordinates = task_coordinates[task];
  const uint expert = coordinates & 255U;
  const uint output_tile = (coordinates >> 8) & 127U;
  const uint m_tile = coordinates >> 15;
  const uint expert_begin = expert == 0U
      ? 0U : (uint)expert_offsets[expert - 1U];
  const uint row_end = (uint)expert_offsets[expert];
  const uint row_start = expert_begin + m_tile * 16U;
  const uint lane = get_sub_group_local_id();
  const uint output_column = output_tile * 16U + lane;
  float8 accum0 = (float8)(0.0f);
  float8 accum1 = (float8)(0.0f);

  for (uint group = 0U; group < 32U; ++group) {
    const ulong weight_index =
        (((((ulong)expert * 128UL + (ulong)output_tile) * 32UL +
           (ulong)group) * 16UL + (ulong)lane) * 16UL);
    const uint4 exact_weight = vload4(
        0, (__global const uint *)(weights + weight_index));
    const uint8 packed_weight = (uint8)(
        exact_weight.s0, exact_weight.s1, exact_weight.s2, exact_weight.s3,
        0x80808080U, 0x80808080U, 0x80808080U, 0x80808080U);
    const ulong scale_index =
        ((((ulong)expert * 128UL + (ulong)output_tile) * 32UL +
          (ulong)group) * 16UL + (ulong)lane);
    const float weight_scale = weight_scales[scale_index];
    const uint block = group >> 4;

#define IQ36_ACCUMULATE_EXACT_F32(accumulator, row_offset)                    \
    do {                                                                       \
      const uint rows = row_start + (row_offset);                              \
      if (rows < row_end) {                                                    \
        const short8 input = iq36_load_rows8_k16(                             \
            source, rows, row_end, group, lane);                               \
        int8 dot = intel_sub_group_i8_u8_matrix_mad_k32(                      \
            input, packed_weight, (int8)(0));                                  \
        dot -= iq36_load_sums8_k16(                                           \
            source_sums, rows, row_end, group) * 128;                          \
        const float8 scales = iq36_load_source_scales8(                       \
            source_scales, rows, row_end, block);                              \
        accumulator = fma(                                                     \
            convert_float8(dot), scales * weight_scale, accumulator);          \
      }                                                                        \
    } while (0)
    IQ36_ACCUMULATE_EXACT_F32(accum0, 0U);
    IQ36_ACCUMULATE_EXACT_F32(accum1, 8U);
#undef IQ36_ACCUMULATE_EXACT_F32
  }

  iq36_store_rows8_f32(output, router_weights, row_start, row_end,
                        output_column, accum0);
  iq36_store_rows8_f32(output, router_weights, row_start + 8U, row_end,
                        output_column, accum1);
}

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(16, 1, 1)))
__kernel void iq36_grouped_s8_u8_q6_exact_block_down_m16_compact_f32(
    __global const uchar *weights,
    __global const char *weight_integer_scales,
    __global const float *weight_block_scales,
    __global const int *expert_offsets,
    __global const uint *task_coordinates,
    __global const char *source,
    __global const float *source_scales,
    __global const short *source_sums,
    __global const float *router_weights,
    __global float *output) {
  const uint task = get_group_id(0);
  const uint coordinates = task_coordinates[task];
  const uint expert = coordinates & 255U;
  const uint output_tile = (coordinates >> 8) & 127U;
  const uint m_tile = coordinates >> 15;
  const uint expert_begin = expert == 0U
      ? 0U : (uint)expert_offsets[expert - 1U];
  const uint row_end = (uint)expert_offsets[expert];
  const uint row_start = expert_begin + m_tile * 16U;
  const uint lane = get_sub_group_local_id();
  const uint output_column = output_tile * 16U + lane;
  float8 accum0 = (float8)(0.0f);
  float8 accum1 = (float8)(0.0f);
  int8 block_dot0 = (int8)(0);
  int8 block_dot1 = (int8)(0);

  for (uint group = 0U; group < 32U; ++group) {
    const ulong weight_index =
        (((((ulong)expert * 128UL + (ulong)output_tile) * 32UL +
           (ulong)group) * 16UL + (ulong)lane) * 16UL);
    const uint4 exact_weight = vload4(
        0, (__global const uint *)(weights + weight_index));
    const uint8 packed_weight = (uint8)(
        exact_weight.s0, exact_weight.s1, exact_weight.s2, exact_weight.s3,
        0x80808080U, 0x80808080U, 0x80808080U, 0x80808080U);
    const ulong scale_index =
        ((((ulong)expert * 128UL + (ulong)output_tile) * 32UL +
          (ulong)group) * 16UL + (ulong)lane);
    const int weight_scale = (int)weight_integer_scales[scale_index];

#define IQ36_ACCUMULATE_EXACT_BLOCK(integer_accumulator, row_offset)         \
    do {                                                                       \
      const uint rows = row_start + (row_offset);                              \
      if (rows < row_end) {                                                    \
        const short8 input = iq36_load_rows8_k16(                             \
            source, rows, row_end, group, lane);                               \
        int8 dot = intel_sub_group_i8_u8_matrix_mad_k32(                      \
            input, packed_weight, (int8)(0));                                  \
        dot -= iq36_load_sums8_k16(                                           \
            source_sums, rows, row_end, group) * 128;                          \
        integer_accumulator += dot * weight_scale;                             \
      }                                                                        \
    } while (0)
    IQ36_ACCUMULATE_EXACT_BLOCK(block_dot0, 0U);
    IQ36_ACCUMULATE_EXACT_BLOCK(block_dot1, 8U);
#undef IQ36_ACCUMULATE_EXACT_BLOCK

    if ((group & 15U) == 15U) {
      const uint block = group >> 4;
      const ulong block_scale_index =
          ((((ulong)expert * 128UL + (ulong)output_tile) * 2UL +
            (ulong)block) * 16UL + (ulong)lane);
      const float weight_d = weight_block_scales[block_scale_index];
      const float8 source_d0 = iq36_load_source_scales8(
          source_scales, row_start, row_end, block);
      const float8 source_d1 = iq36_load_source_scales8(
          source_scales, row_start + 8U, row_end, block);
      accum0 = fma(convert_float8(block_dot0), source_d0 * weight_d, accum0);
      accum1 = fma(convert_float8(block_dot1), source_d1 * weight_d, accum1);
      block_dot0 = (int8)(0);
      block_dot1 = (int8)(0);
    }
  }

  iq36_store_rows8_f32(output, router_weights, row_start, row_end,
                        output_column, accum0);
  iq36_store_rows8_f32(output, router_weights, row_start + 8U, row_end,
                        output_column, accum1);
}

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(16, 1, 1)))
__kernel void iq36_grouped_s8_u8_q6_surrogate_down_m24_f16scale_table(
    __global const uchar *weights,
    __global const half *weight_scales,
    __global const int4 *tasks,
    __global const char *source,
    __global const float *source_scales,
    __global const short *source_sums,
    __global const float *router_weights,
    __global half *output) {
  const uint task = get_group_id(0);
  const int4 coordinates = tasks[task];
  const uint expert = (uint)coordinates.x;
  const uint row_start = (uint)coordinates.y;
  const uint row_end = (uint)coordinates.z;
  const uint output_tile = (uint)coordinates.w;
  const uint lane = get_sub_group_local_id();
  const uint output_column = output_tile * 16U + lane;
  float8 accum0 = (float8)(0.0f);
  float8 accum1 = (float8)(0.0f);
  float8 accum2 = (float8)(0.0f);

  for (uint group = 0U; group < 16U; ++group) {
    const ulong weight_index =
        (((((ulong)expert * 128UL + (ulong)output_tile) * 16UL +
           (ulong)group) * 16UL + (ulong)lane) * 32UL);
    const uint8 packed_weight = vload8(
        0, (__global const uint *)(weights + weight_index));
    const ulong scale_index =
        ((((ulong)expert * 128UL + (ulong)output_tile) * 16UL +
          (ulong)group) * 16UL + (ulong)lane);
    const float weight_scale = convert_float(weight_scales[scale_index]);
    const uint block = group >> 3;

#define IQ36_ACCUMULATE_M24(accumulator, row_offset)                          \
    do {                                                                       \
      const uint rows = row_start + (row_offset);                              \
      if (rows < row_end) {                                                    \
        const short8 input = iq36_load_rows8(                                 \
            source, rows, row_end, group, lane);                               \
        int8 dot = intel_sub_group_i8_u8_matrix_mad_k32(                      \
            input, packed_weight, (int8)(0));                                  \
        dot -= iq36_load_sums8(source_sums, rows, row_end, group) * 128;       \
        const float8 scales = iq36_load_source_scales8(                       \
            source_scales, rows, row_end, block);                              \
        accumulator = fma(                                                     \
            convert_float8(dot), scales * weight_scale, accumulator);          \
      }                                                                        \
    } while (0)
    IQ36_ACCUMULATE_M24(accum0, 0U);
    IQ36_ACCUMULATE_M24(accum1, 8U);
    IQ36_ACCUMULATE_M24(accum2, 16U);
#undef IQ36_ACCUMULATE_M24
  }

  iq36_store_rows8(output, router_weights, row_start, row_end,
                    output_column, accum0);
  iq36_store_rows8(output, router_weights, row_start + 8U, row_end,
                    output_column, accum1);
  iq36_store_rows8(output, router_weights, row_start + 16U, row_end,
                    output_column, accum2);
}

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(16, 1, 1)))
__kernel void iq36_grouped_s8_u8_q6_surrogate_down_m32_f16scale_table(
    __global const uchar *weights,
    __global const half *weight_scales,
    __global const int4 *tasks,
    __global const char *source,
    __global const float *source_scales,
    __global const short *source_sums,
    __global const float *router_weights,
    __global half *output) {
  const uint task = get_group_id(0);
  const int4 coordinates = tasks[task];
  const uint expert = (uint)coordinates.x;
  const uint row_start = (uint)coordinates.y;
  const uint row_end = (uint)coordinates.z;
  const uint output_tile = (uint)coordinates.w;
  const uint lane = get_sub_group_local_id();
  const uint output_column = output_tile * 16U + lane;
  float8 accum0 = (float8)(0.0f);
  float8 accum1 = (float8)(0.0f);
  float8 accum2 = (float8)(0.0f);
  float8 accum3 = (float8)(0.0f);

  for (uint group = 0U; group < 16U; ++group) {
    const ulong weight_index =
        (((((ulong)expert * 128UL + (ulong)output_tile) * 16UL +
           (ulong)group) * 16UL + (ulong)lane) * 32UL);
    const uint8 packed_weight = vload8(
        0, (__global const uint *)(weights + weight_index));
    const ulong scale_index =
        ((((ulong)expert * 128UL + (ulong)output_tile) * 16UL +
          (ulong)group) * 16UL + (ulong)lane);
    const float weight_scale = convert_float(weight_scales[scale_index]);
    const uint block = group >> 3;

#define IQ36_ACCUMULATE(accumulator, row_offset)                              \
    do {                                                                       \
      const uint rows = row_start + (row_offset);                              \
      if (rows < row_end) {                                                    \
        const short8 input = iq36_load_rows8(                                 \
            source, rows, row_end, group, lane);                               \
        int8 dot = intel_sub_group_i8_u8_matrix_mad_k32(                      \
            input, packed_weight, (int8)(0));                                  \
        dot -= iq36_load_sums8(source_sums, rows, row_end, group) * 128;       \
        const float8 scales = iq36_load_source_scales8(                       \
            source_scales, rows, row_end, block);                              \
        accumulator = fma(                                                     \
            convert_float8(dot), scales * weight_scale, accumulator);          \
      }                                                                        \
    } while (0)
    IQ36_ACCUMULATE(accum0, 0U);
    IQ36_ACCUMULATE(accum1, 8U);
    IQ36_ACCUMULATE(accum2, 16U);
    IQ36_ACCUMULATE(accum3, 24U);
#undef IQ36_ACCUMULATE
  }

  iq36_store_rows8(output, router_weights, row_start, row_end,
                    output_column, accum0);
  iq36_store_rows8(output, router_weights, row_start + 8U, row_end,
                    output_column, accum1);
  iq36_store_rows8(output, router_weights, row_start + 16U, row_end,
                    output_column, accum2);
  iq36_store_rows8(output, router_weights, row_start + 24U, row_end,
                    output_column, accum3);
}

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void iq36_grouped_s8_u8_q6_surrogate_down_m32_f16scale_wg64(
    __global const uchar *weights,
    __global const half *weight_scales,
    __global const int4 *work_tiles,
    __global const char *source,
    __global const float *source_scales,
    __global const short *source_sums,
    __global const float *router_weights,
    __global half *output) {
  const uint output_tile = get_group_id(0) * 4U + get_sub_group_id();
  const uint work_tile = get_group_id(1);
  const uint lane = get_sub_group_local_id();
  const int4 tile = work_tiles[work_tile];
  const uint expert = (uint)tile.x;
  const uint row_start = (uint)tile.y;
  const uint row_end = (uint)tile.z;
  const uint output_column = output_tile * 16U + lane;
  float8 accum0 = (float8)(0.0f);
  float8 accum1 = (float8)(0.0f);
  float8 accum2 = (float8)(0.0f);
  float8 accum3 = (float8)(0.0f);

  for (uint group = 0U; group < 16U; ++group) {
    const ulong weight_index =
        (((((ulong)expert * 128UL + (ulong)output_tile) * 16UL +
           (ulong)group) * 16UL + (ulong)lane) * 32UL);
    const uint8 packed_weight = vload8(
        0, (__global const uint *)(weights + weight_index));
    const ulong scale_index =
        ((((ulong)expert * 128UL + (ulong)output_tile) * 16UL +
          (ulong)group) * 16UL + (ulong)lane);
    const float weight_scale = convert_float(weight_scales[scale_index]);
    const uint block = group >> 3;

#define IQ36_ACCUMULATE(accumulator, row_offset)                              \
    do {                                                                       \
      const uint rows = row_start + (row_offset);                              \
      if (rows < row_end) {                                                    \
        const short8 input = iq36_load_rows8(                                 \
            source, rows, row_end, group, lane);                               \
        int8 dot = intel_sub_group_i8_u8_matrix_mad_k32(                      \
            input, packed_weight, (int8)(0));                                  \
        dot -= iq36_load_sums8(source_sums, rows, row_end, group) * 128;       \
        const float8 scales = iq36_load_source_scales8(                       \
            source_scales, rows, row_end, block);                              \
        accumulator = fma(                                                     \
            convert_float8(dot), scales * weight_scale, accumulator);          \
      }                                                                        \
    } while (0)
    IQ36_ACCUMULATE(accum0, 0U);
    IQ36_ACCUMULATE(accum1, 8U);
    IQ36_ACCUMULATE(accum2, 16U);
    IQ36_ACCUMULATE(accum3, 24U);
#undef IQ36_ACCUMULATE
  }

  iq36_store_rows8(output, router_weights, row_start, row_end,
                    output_column, accum0);
  iq36_store_rows8(output, router_weights, row_start + 8U, row_end,
                    output_column, accum1);
  iq36_store_rows8(output, router_weights, row_start + 16U, row_end,
                    output_column, accum2);
  iq36_store_rows8(output, router_weights, row_start + 24U, row_end,
                    output_column, accum3);
}

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(32, 1, 1)))
__kernel void iq36_grouped_s8_u8_q6_surrogate_down_m64_shared(
    __global const uchar *weights,
    __global const float *weight_scales,
    __global const int4 *work_tiles,
    __global const char *source,
    __global const float *source_scales,
    __global const short *source_sums,
    __global const float *router_weights,
    __global half *output) {
  const uint output_tile = get_group_id(0);
  const uint work_tile = get_group_id(1);
  const uint subgroup = get_sub_group_id();
  const uint lane = get_sub_group_local_id();
  const int4 tile = work_tiles[work_tile];
  const uint expert = (uint)tile.x;
  const uint row_start = (uint)tile.y + subgroup * 32U;
  const uint row_end = (uint)tile.z;
  const uint output_column = output_tile * 16U + lane;
  __local uchar weight_cache[16 * 32];
  float8 accum0 = (float8)(0.0f);
  float8 accum1 = (float8)(0.0f);
  float8 accum2 = (float8)(0.0f);
  float8 accum3 = (float8)(0.0f);

  for (uint group = 0U; group < 16U; ++group) {
    const ulong weight_index =
        (((((ulong)expert * 128UL + (ulong)output_tile) * 16UL +
           (ulong)group) * 16UL + (ulong)lane) * 32UL);
    if (subgroup == 0U) {
      const uint8 loaded = vload8(
          0, (__global const uint *)(weights + weight_index));
      vstore8(loaded, 0,
              (__local uint *)(weight_cache + lane * 32U));
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    const uint8 packed_weight = vload8(
        0, (__local const uint *)(weight_cache + lane * 32U));
    const ulong scale_index =
        ((((ulong)expert * 128UL + (ulong)output_tile) * 16UL +
          (ulong)group) * 16UL + (ulong)lane);
    const float weight_scale = weight_scales[scale_index];
    const uint block = group >> 3;

#define IQ36_ACCUMULATE(accumulator, row_offset)                              \
    do {                                                                       \
      const uint rows = row_start + (row_offset);                              \
      const short8 input = iq36_load_rows8(                                   \
          source, rows, row_end, group, lane);                                 \
      int8 dot = intel_sub_group_i8_u8_matrix_mad_k32(                        \
          input, packed_weight, (int8)(0));                                    \
      dot -= iq36_load_sums8(source_sums, rows, row_end, group) * 128;         \
      const float8 scales = iq36_load_source_scales8(                         \
          source_scales, rows, row_end, block);                                \
      accumulator = fma(                                                       \
          convert_float8(dot), scales * weight_scale, accumulator);            \
    } while (0)
    IQ36_ACCUMULATE(accum0, 0U);
    IQ36_ACCUMULATE(accum1, 8U);
    IQ36_ACCUMULATE(accum2, 16U);
    IQ36_ACCUMULATE(accum3, 24U);
#undef IQ36_ACCUMULATE
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  iq36_store_rows8(output, router_weights, row_start, row_end,
                    output_column, accum0);
  iq36_store_rows8(output, router_weights, row_start + 8U, row_end,
                    output_column, accum1);
  iq36_store_rows8(output, router_weights, row_start + 16U, row_end,
                    output_column, accum2);
  iq36_store_rows8(output, router_weights, row_start + 24U, row_end,
                    output_column, accum3);
}

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(16, 1, 1)))
__kernel void iq36_grouped_s8_u8_q6_surrogate_down_m48(
    __global const uchar *weights,
    __global const float *weight_scales,
    __global const int4 *work_tiles,
    __global const char *source,
    __global const float *source_scales,
    __global const short *source_sums,
    __global const float *router_weights,
    __global half *output) {
  const uint output_tile = get_group_id(0);
  const uint work_tile = get_group_id(1);
  const uint lane = get_sub_group_local_id();
  const int4 tile = work_tiles[work_tile];
  const uint expert = (uint)tile.x;
  const uint row_start = (uint)tile.y;
  const uint row_end = (uint)tile.z;
  const uint output_column = output_tile * 16U + lane;
  float8 accum0 = (float8)(0.0f);
  float8 accum1 = (float8)(0.0f);
  float8 accum2 = (float8)(0.0f);
  float8 accum3 = (float8)(0.0f);
  float8 accum4 = (float8)(0.0f);
  float8 accum5 = (float8)(0.0f);

  for (uint group = 0U; group < 16U; ++group) {
    const ulong weight_index =
        (((((ulong)expert * 128UL + (ulong)output_tile) * 16UL +
           (ulong)group) * 16UL + (ulong)lane) * 32UL);
    const uint8 packed_weight = vload8(
        0, (__global const uint *)(weights + weight_index));
    const ulong scale_index =
        ((((ulong)expert * 128UL + (ulong)output_tile) * 16UL +
          (ulong)group) * 16UL + (ulong)lane);
    const float weight_scale = weight_scales[scale_index];
    const uint block = group >> 3;

#define IQ36_ACCUMULATE(accumulator, row_offset)                              \
    do {                                                                       \
      const uint rows = row_start + (row_offset);                              \
      const short8 input = iq36_load_rows8(                                   \
          source, rows, row_end, group, lane);                                 \
      int8 dot = intel_sub_group_i8_u8_matrix_mad_k32(                        \
          input, packed_weight, (int8)(0));                                    \
      dot -= iq36_load_sums8(source_sums, rows, row_end, group) * 128;         \
      const float8 scales = iq36_load_source_scales8(                         \
          source_scales, rows, row_end, block);                                \
      accumulator = fma(                                                       \
          convert_float8(dot), scales * weight_scale, accumulator);            \
    } while (0)
    IQ36_ACCUMULATE(accum0, 0U);
    IQ36_ACCUMULATE(accum1, 8U);
    IQ36_ACCUMULATE(accum2, 16U);
    IQ36_ACCUMULATE(accum3, 24U);
    IQ36_ACCUMULATE(accum4, 32U);
    IQ36_ACCUMULATE(accum5, 40U);
#undef IQ36_ACCUMULATE
  }

  iq36_store_rows8(output, router_weights, row_start, row_end,
                    output_column, accum0);
  iq36_store_rows8(output, router_weights, row_start + 8U, row_end,
                    output_column, accum1);
  iq36_store_rows8(output, router_weights, row_start + 16U, row_end,
                    output_column, accum2);
  iq36_store_rows8(output, router_weights, row_start + 24U, row_end,
                    output_column, accum3);
  iq36_store_rows8(output, router_weights, row_start + 32U, row_end,
                    output_column, accum4);
  iq36_store_rows8(output, router_weights, row_start + 40U, row_end,
                    output_column, accum5);
}

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(16, 1, 1)))
__kernel void iq36_grouped_s8_u8_q6_surrogate_down_m64(
    __global const uchar *weights,
    __global const float *weight_scales,
    __global const int4 *work_tiles,
    __global const char *source,
    __global const float *source_scales,
    __global const short *source_sums,
    __global const float *router_weights,
    __global half *output) {
  const uint output_tile = get_group_id(0);
  const uint work_tile = get_group_id(1);
  const uint lane = get_sub_group_local_id();
  const int4 tile = work_tiles[work_tile];
  const uint expert = (uint)tile.x;
  const uint row_start = (uint)tile.y;
  const uint row_end = (uint)tile.z;
  const uint output_column = output_tile * 16U + lane;
  float8 accum0 = (float8)(0.0f);
  float8 accum1 = (float8)(0.0f);
  float8 accum2 = (float8)(0.0f);
  float8 accum3 = (float8)(0.0f);
  float8 accum4 = (float8)(0.0f);
  float8 accum5 = (float8)(0.0f);
  float8 accum6 = (float8)(0.0f);
  float8 accum7 = (float8)(0.0f);

  for (uint group = 0U; group < 16U; ++group) {
    const ulong weight_index =
        (((((ulong)expert * 128UL + (ulong)output_tile) * 16UL +
           (ulong)group) * 16UL + (ulong)lane) * 32UL);
    const uint8 packed_weight = vload8(
        0, (__global const uint *)(weights + weight_index));
    const ulong scale_index =
        ((((ulong)expert * 128UL + (ulong)output_tile) * 16UL +
          (ulong)group) * 16UL + (ulong)lane);
    const float weight_scale = weight_scales[scale_index];
    const uint block = group >> 3;

#define IQ36_ACCUMULATE(accumulator, row_offset)                              \
    do {                                                                       \
      const uint rows = row_start + (row_offset);                              \
      const short8 input = iq36_load_rows8(                                   \
          source, rows, row_end, group, lane);                                 \
      int8 dot = intel_sub_group_i8_u8_matrix_mad_k32(                        \
          input, packed_weight, (int8)(0));                                    \
      dot -= iq36_load_sums8(source_sums, rows, row_end, group) * 128;         \
      const float8 scales = iq36_load_source_scales8(                         \
          source_scales, rows, row_end, block);                                \
      accumulator = fma(                                                       \
          convert_float8(dot), scales * weight_scale, accumulator);            \
    } while (0)
    IQ36_ACCUMULATE(accum0, 0U);
    IQ36_ACCUMULATE(accum1, 8U);
    IQ36_ACCUMULATE(accum2, 16U);
    IQ36_ACCUMULATE(accum3, 24U);
    IQ36_ACCUMULATE(accum4, 32U);
    IQ36_ACCUMULATE(accum5, 40U);
    IQ36_ACCUMULATE(accum6, 48U);
    IQ36_ACCUMULATE(accum7, 56U);
#undef IQ36_ACCUMULATE
  }

  iq36_store_rows8(output, router_weights, row_start, row_end,
                    output_column, accum0);
  iq36_store_rows8(output, router_weights, row_start + 8U, row_end,
                    output_column, accum1);
  iq36_store_rows8(output, router_weights, row_start + 16U, row_end,
                    output_column, accum2);
  iq36_store_rows8(output, router_weights, row_start + 24U, row_end,
                    output_column, accum3);
  iq36_store_rows8(output, router_weights, row_start + 32U, row_end,
                    output_column, accum4);
  iq36_store_rows8(output, router_weights, row_start + 40U, row_end,
                    output_column, accum5);
  iq36_store_rows8(output, router_weights, row_start + 48U, row_end,
                    output_column, accum6);
  iq36_store_rows8(output, router_weights, row_start + 56U, row_end,
                    output_column, accum7);
}
