#pragma OPENCL EXTENSION cl_khr_fp16 : enable

int iq36_nearest_int(float value) {
  const float shifted = value + 12582912.0f;
  return (as_int(shifted) & 0x007fffff) - 0x00400000;
}

// Six kernels form one device-resident 1024-token/top-8 schedule build. The
// split gives the GPU enough workgroups for histogram, scatter, and coordinate
// generation; the production Level Zero list can record the fixed sequence
// once. Compact row order is intentionally unspecified because inverse_map
// preserves every source assignment consumed by the deterministic scatter.
__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void iq36_grouped_schedule_reset_1024(
    __global uint *counts,
    __global uint *cursors,
    __global uint *metadata) {
  const uint expert = get_local_id(0);
  counts[expert] = 0U;
  cursors[expert] = 0U;
  if (expert < 7U) metadata[expert] = 0U;
}

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void iq36_grouped_schedule_count_1024(
    __global const int *topk_ids,
    __global uint *partial_counts,
    __global uint *metadata) {
  const uint lane = get_local_id(0);
  const uint chunk = get_group_id(0);
  const uint source = get_global_id(0);
  __local uint local_counts[256];
  __local uint error;
  local_counts[lane] = 0U;
  if (lane == 0U) error = 0U;
  barrier(CLK_LOCAL_MEM_FENCE);
  const int selected = topk_ids[source];
  if (selected < 0 || selected >= 256) {
    atomic_or((volatile __local uint *)&error, 1U);
  } else {
    const uint rank = source & 7U;
    for (uint prior = 0U; prior < rank; ++prior) {
      if (selected == topk_ids[source - rank + prior]) {
        atomic_or((volatile __local uint *)&error, 2U);
      }
    }
    atomic_inc((volatile __local uint *)&local_counts[(uint)selected]);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  partial_counts[chunk * 256U + lane] = local_counts[lane];
  if (lane == 0U && error != 0U) {
    atomic_or((volatile __global uint *)&metadata[4], error);
  }
}

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void iq36_grouped_schedule_prefix_1024(
    __global const uint *partial_counts,
    __global uint *counts,
    __global uint *begins,
    __global uint *cursors,
    __global int *offsets,
    __global uint *gateup_bases,
    __global uint *down_bases,
    __global uint *native_gateup_bases,
    __global uint *native_down_bases,
    __global uint *chunk_bases,
    __global uint *metadata) {
  const uint lane = get_local_id(0);
  uint count = 0U;
  for (uint chunk = 0U; chunk < 32U; ++chunk) {
    count += partial_counts[chunk * 256U + lane];
  }
  counts[lane] = count;
  __local uint count_prefix[256];
  __local uint tile_prefix[256];
  __local uint native_tile_prefix[256];
  __local uint active_prefix[256];
  __local uint maxima[256];
  count_prefix[lane] = count;
  tile_prefix[lane] = (count + 15U) >> 4U;
  native_tile_prefix[lane] = (count + 31U) >> 5U;
  active_prefix[lane] = count != 0U;
  maxima[lane] = count;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint step = 1U; step < 256U; step <<= 1U) {
    const uint add_count = lane >= step ? count_prefix[lane - step] : 0U;
    const uint add_tiles = lane >= step ? tile_prefix[lane - step] : 0U;
    const uint add_native_tiles =
        lane >= step ? native_tile_prefix[lane - step] : 0U;
    const uint add_active = lane >= step ? active_prefix[lane - step] : 0U;
    barrier(CLK_LOCAL_MEM_FENCE);
    count_prefix[lane] += add_count;
    tile_prefix[lane] += add_tiles;
    native_tile_prefix[lane] += add_native_tiles;
    active_prefix[lane] += add_active;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  for (uint step = 128U; step > 0U; step >>= 1U) {
    if (lane < step) maxima[lane] = max(maxima[lane], maxima[lane + step]);
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const uint begin = lane == 0U ? 0U : count_prefix[lane - 1U];
  const uint tile_begin = lane == 0U ? 0U : tile_prefix[lane - 1U];
  const uint native_tile_begin =
      lane == 0U ? 0U : native_tile_prefix[lane - 1U];
  begins[lane] = begin;
  uint chunk_begin = begin;
  for (uint chunk = 0U; chunk < 32U; ++chunk) {
    chunk_bases[chunk * 256U + lane] = chunk_begin;
    chunk_begin += partial_counts[chunk * 256U + lane];
  }
  cursors[lane] = chunk_begin;
  offsets[lane] = (int)count_prefix[lane];
  gateup_bases[lane] = 64U * tile_begin;
  down_bases[lane] = 128U * tile_begin;
  native_gateup_bases[lane] = 16U * native_tile_begin;
  native_down_bases[lane] = 32U * native_tile_begin;
  if (lane == 0U && count_prefix[255] != 8192U) {
    atomic_or((volatile __global uint *)&metadata[4], 4U);
  }
  if (lane == 0U) {
    metadata[0] = 64U * tile_prefix[255];
    metadata[1] = 128U * tile_prefix[255];
    metadata[2] = active_prefix[255];
    metadata[3] = maxima[0];
    metadata[5] = 16U * native_tile_prefix[255];
    metadata[6] = 32U * native_tile_prefix[255];
  }
}

// The oneDNN-generated native kernels use a 64-output by 32-row workgroup
// tile. These compact coordinates let a fixed set of physical workgroups walk
// only live logical tiles, so the host never needs the maximum expert count.
__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void iq36_grouped_schedule_native_tasks_1024(
    __global const uint *counts,
    __global const uint *cursors,
    __global const int *offsets,
    __global const uint *native_gateup_bases,
    __global const uint *native_down_bases,
    __global uint *native_gateup_task_coordinates,
    __global uint *native_down_task_coordinates,
    __global uint *metadata) {
  const uint expert = get_group_id(0);
  const uint lane = get_local_id(0);
  const uint m_tiles = (counts[expert] + 31U) >> 5U;
  const uint gateup_count = 16U * m_tiles;
  for (uint item = lane; item < gateup_count; item += 256U) {
    const uint output_tile = item / m_tiles;
    const uint m_tile = item - output_tile * m_tiles;
    native_gateup_task_coordinates[native_gateup_bases[expert] + item] =
        expert | (output_tile << 8U) | (m_tile << 15U);
  }
  const uint down_count = 32U * m_tiles;
  for (uint item = lane; item < down_count; item += 256U) {
    const uint output_tile = item / m_tiles;
    const uint m_tile = item - output_tile * m_tiles;
    native_down_task_coordinates[native_down_bases[expert] + item] =
        expert | (output_tile << 8U) | (m_tile << 15U);
  }
  if (lane == 0U && cursors[expert] != (uint)offsets[expert]) {
    atomic_or((volatile __global uint *)&metadata[4], 8U);
  }
}

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void iq36_grouped_schedule_scatter_1024(
    __global const int *topk_ids,
    __global const float *router_weights,
    __global const uint *chunk_bases,
    __global int *token_map,
    __global int *inverse_map,
    __global float *compact_router_weights) {
  const uint lane = get_local_id(0);
  const uint chunk = get_group_id(0);
  const uint source = get_global_id(0);
  __local uint local_cursors[256];
  local_cursors[lane] = chunk_bases[chunk * 256U + lane];
  barrier(CLK_LOCAL_MEM_FENCE);
  const int selected = topk_ids[source];
  if (selected < 0 || selected >= 256) return;
  const uint row = atomic_inc(
      (volatile __local uint *)&local_cursors[(uint)selected]);
  token_map[row] = (int)(source >> 3U);
  inverse_map[source] = (int)row;
  compact_router_weights[row] = router_weights[source];
}

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void iq36_grouped_schedule_tasks_1024(
    __global const uint *counts,
    __global const uint *cursors,
    __global const int *offsets,
    __global const uint *gateup_bases,
    __global const uint *down_bases,
    __global uint *gateup_task_coordinates,
    __global uint *down_task_coordinates,
    __global uint *metadata) {
  const uint expert = get_group_id(0);
  const uint lane = get_local_id(0);
  const uint m_tiles = (counts[expert] + 15U) >> 4U;
  const uint gateup_count = 64U * m_tiles;
  for (uint item = lane; item < gateup_count; item += 256U) {
    const uint output_tile = item / m_tiles;
    const uint m_tile = item - output_tile * m_tiles;
    gateup_task_coordinates[gateup_bases[expert] + item] =
        expert | (output_tile << 8U) | (m_tile << 15U);
  }
  const uint down_count = 128U * m_tiles;
  for (uint item = lane; item < down_count; item += 256U) {
    const uint output_tile = item / m_tiles;
    const uint m_tile = item - output_tile * m_tiles;
    down_task_coordinates[down_bases[expert] + item] =
        expert | (output_tile << 8U) | (m_tile << 15U);
  }
  if (lane == 0U && cursors[expert] != (uint)offsets[expert]) {
    atomic_or((volatile __global uint *)&metadata[4], 8U);
  }
}

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void iq36_router_topk8_1024(
    __global const float *logits,
    __global int *selected_positions,
    __global float *normalized_weights) {
  const uint token = get_group_id(0);
  const uint lane = get_local_id(0);
  const uint subgroup = get_sub_group_id();
  const uint subgroup_lane = get_sub_group_local_id();
  __local float candidate_values[16 * 8];
  __local uint candidate_ids[16 * 8];
  float remaining = logits[token * 256U + lane];
  for (uint index = 0U; index < 8U; ++index) {
    const float best_value = sub_group_reduce_max(remaining);
    const uint best_id = sub_group_reduce_min(
        remaining == best_value ? lane : 0xffffffffU);
    if (lane == best_id) remaining = -INFINITY;
    if (subgroup_lane == 0U) {
      candidate_values[subgroup * 8U + index] = best_value;
      candidate_ids[subgroup * 8U + index] = best_id;
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (subgroup != 0U) return;

  float lane_values[8];
  uint lane_ids[8];
  for (uint index = 0U; index < 8U; ++index) {
    lane_values[index] = candidate_values[subgroup_lane * 8U + index];
    lane_ids[index] = candidate_ids[subgroup_lane * 8U + index];
  }
  float best_values[8];
  uint best_ids[8];
  for (uint rank = 0U; rank < 8U; ++rank) {
    float lane_best_value = -INFINITY;
    uint lane_best_id = 0xffffffffU;
    for (uint index = 0U; index < 8U; ++index) {
      if (lane_values[index] > lane_best_value ||
          (lane_values[index] == lane_best_value &&
           lane_ids[index] < lane_best_id)) {
        lane_best_value = lane_values[index];
        lane_best_id = lane_ids[index];
      }
    }
    best_values[rank] = sub_group_reduce_max(lane_best_value);
    best_ids[rank] = sub_group_reduce_min(
        lane_best_value == best_values[rank]
            ? lane_best_id : 0xffffffffU);
    for (uint index = 0U; index < 8U; ++index) {
      if (lane_ids[index] == best_ids[rank]) {
        lane_values[index] = -INFINITY;
      }
    }
  }
  if (subgroup_lane != 0U) return;
  const float maximum = best_values[0];
  float denominator = 0.0f;
  float exponentials[8];
  for (uint index = 0U; index < 8U; ++index) {
    exponentials[index] = exp(best_values[index] - maximum);
    denominator += exponentials[index];
  }
  denominator = fmax(denominator, 0.001f);
  for (uint index = 0U; index < 8U; ++index) {
    selected_positions[token * 8U + index] = (int)best_ids[index];
    normalized_weights[token * 8U + index] =
        exponentials[index] / denominator;
  }
}

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void iq36_quantize_tokens_q8(
    __global const float *input,
    __global char *q8,
    __global float *scales,
    __global char *sum_low,
    __global char *sum_high,
    uint row_count) {
  const uint lane = get_local_id(0);
  const uint task = get_group_id(0);
  const uint row = task >> 3;
  const uint block = task & 7U;
  if (row >= row_count) return;

  __local float values[256];
  __local float maxima[256];
  __local uint indices[256];
  __local int quantized[256];
  const float value = input[row * 2048U + block * 256U + lane];
  values[lane] = value;
  maxima[lane] = fabs(value);
  indices[lane] = lane;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint step = 128U; step > 0U; step >>= 1U) {
    if (lane < step) {
      const float rhs = maxima[lane + step];
      const uint rhs_index = indices[lane + step];
      if (rhs > maxima[lane] ||
          (rhs == maxima[lane] && rhs_index < indices[lane])) {
        maxima[lane] = rhs;
        indices[lane] = rhs_index;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float max_value = values[indices[0]];
  const float inverse_scale = maxima[0] == 0.0f ? 0.0f : -127.0f / max_value;
  const float scale = inverse_scale == 0.0f ? 0.0f : 1.0f / inverse_scale;
  const int q = inverse_scale == 0.0f
      ? 0 : min(127, iq36_nearest_int(inverse_scale * value));
  quantized[lane] = q;
  q8[row * 2048U + block * 256U + lane] = (char)q;
  if (lane == 0U) scales[row * 8U + block] = scale;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lane < 8U) {
    int sum = 0;
    for (uint item = 0U; item < 32U; ++item) {
      sum += quantized[lane * 32U + item];
    }
    const uint group = block * 8U + lane;
    const int low = ((sum + 128) & 255) - 128;
    sum_low[row * 64U + group] = (char)low;
    sum_high[row * 64U + group] = (char)((sum - low) / 256);
  }
}

__kernel void iq36_gather_quantized_q8(
    __global const int *token_map,
    __global const char *token_q8,
    __global const float *token_scales,
    __global const char *token_sum_low,
    __global const char *token_sum_high,
    __global char *grouped_q8,
    __global float *grouped_scales,
    __global char *grouped_sum_low,
    __global char *grouped_sum_high,
    uint row_count) {
  const uint index = get_global_id(0);
  if (index >= row_count * 2048U) return;
  const uint row = index >> 11;
  const uint inner = index & 2047U;
  const uint token = (uint)token_map[row];
  grouped_q8[index] = token_q8[token * 2048U + inner];
  if (inner < 8U) {
    grouped_scales[row * 8U + inner] = token_scales[token * 8U + inner];
  }
  if (inner < 64U) {
    grouped_sum_low[row * 64U + inner] =
        token_sum_low[token * 64U + inner];
    grouped_sum_high[row * 64U + inner] =
        token_sum_high[token * 64U + inner];
  }
}

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void iq36_quantize_swiglu_q8(
    __global const half *input,
    __global char *q8,
    __global float *scales,
    __global char *sum_low,
    __global char *sum_high,
    uint row_count) {
  const uint lane = get_local_id(0);
  const uint task = get_group_id(0);
  const uint row = task >> 1;
  const uint block = task & 1U;
  if (row >= row_count) return;

  __local float values[256];
  __local float maxima[256];
  __local uint indices[256];
  __local int quantized[256];
  const float value = convert_float(
      input[row * 512U + block * 256U + lane]);
  values[lane] = value;
  maxima[lane] = fabs(value);
  indices[lane] = lane;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint step = 128U; step > 0U; step >>= 1U) {
    if (lane < step) {
      const float rhs = maxima[lane + step];
      const uint rhs_index = indices[lane + step];
      if (rhs > maxima[lane] ||
          (rhs == maxima[lane] && rhs_index < indices[lane])) {
        maxima[lane] = rhs;
        indices[lane] = rhs_index;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float max_value = values[indices[0]];
  const float inverse_scale = maxima[0] == 0.0f ? 0.0f : -127.0f / max_value;
  const float scale = inverse_scale == 0.0f ? 0.0f : 1.0f / inverse_scale;
  const int q = inverse_scale == 0.0f
      ? 0 : min(127, iq36_nearest_int(inverse_scale * value));
  quantized[lane] = q;
  q8[row * 512U + block * 256U + lane] = (char)q;
  if (lane == 0U) scales[row * 2U + block] = scale;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lane < 8U) {
    int sum = 0;
    for (uint item = 0U; item < 32U; ++item) {
      sum += quantized[lane * 32U + item];
    }
    const uint group = block * 8U + lane;
    const int low = ((sum + 128) & 255) - 128;
    sum_low[row * 16U + group] = (char)low;
    sum_high[row * 16U + group] = (char)((sum - low) / 256);
  }
}

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void iq36_quantize_swiglu_f32_q8(
    __global const float *input,
    __global char *q8,
    __global float *scales,
    __global char *sum_low,
    __global char *sum_high,
    uint row_count) {
  const uint lane = get_local_id(0);
  const uint task = get_group_id(0);
  const uint row = task >> 1;
  const uint block = task & 1U;
  if (row >= row_count) return;

  __local float values[256];
  __local float maxima[256];
  __local uint indices[256];
  __local int quantized[256];
  const float value = input[row * 512U + block * 256U + lane];
  values[lane] = value;
  maxima[lane] = fabs(value);
  indices[lane] = lane;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint step = 128U; step > 0U; step >>= 1U) {
    if (lane < step) {
      const float rhs = maxima[lane + step];
      const uint rhs_index = indices[lane + step];
      if (rhs > maxima[lane] ||
          (rhs == maxima[lane] && rhs_index < indices[lane])) {
        maxima[lane] = rhs;
        indices[lane] = rhs_index;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float max_value = values[indices[0]];
  const float inverse_scale = maxima[0] == 0.0f ? 0.0f : -127.0f / max_value;
  const float scale = inverse_scale == 0.0f ? 0.0f : 1.0f / inverse_scale;
  const int q = inverse_scale == 0.0f
      ? 0 : min(127, iq36_nearest_int(inverse_scale * value));
  quantized[lane] = q;
  q8[row * 512U + block * 256U + lane] = (char)q;
  if (lane == 0U) scales[row * 2U + block] = scale;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lane < 8U) {
    int sum = 0;
    for (uint item = 0U; item < 32U; ++item) {
      sum += quantized[lane * 32U + item];
    }
    const uint group = block * 8U + lane;
    const int low = ((sum + 128) & 255) - 128;
    sum_low[row * 16U + group] = (char)low;
    sum_high[row * 16U + group] = (char)((sum - low) / 256);
  }
}

short iq36_q4_pack_i8_pair(char low, char high) {
  const ushort bits = (ushort)(uchar)low | ((ushort)(uchar)high << 8);
  return as_short(bits);
}

short8 iq36_q4_load_rows8(
    __global const char *source,
    uint row_start,
    uint row_end,
    uint group,
    uint lane,
    uint row_stride) {
  short8 packed = (short8)(0);
  if (lane >= 16U) return packed;
  const uint column = group * 32U + lane * 2U;
#define IQ36_Q4_LOAD_ROW(component, offset)                                  \
  do {                                                                         \
    const uint row = row_start + (offset);                                     \
    if (row < row_end) {                                                       \
      packed.component = iq36_q4_pack_i8_pair(                               \
          source[row * row_stride + column],                                  \
          source[row * row_stride + column + 1U]);                            \
    }                                                                          \
  } while (0)
  IQ36_Q4_LOAD_ROW(s0, 0U);
  IQ36_Q4_LOAD_ROW(s1, 1U);
  IQ36_Q4_LOAD_ROW(s2, 2U);
  IQ36_Q4_LOAD_ROW(s3, 3U);
  IQ36_Q4_LOAD_ROW(s4, 4U);
  IQ36_Q4_LOAD_ROW(s5, 5U);
  IQ36_Q4_LOAD_ROW(s6, 6U);
  IQ36_Q4_LOAD_ROW(s7, 7U);
#undef IQ36_Q4_LOAD_ROW
  return packed;
}

int8 iq36_q4_load_sums8(
    __global const char *sum_low,
    __global const char *sum_high,
    uint row_start,
    uint row_end,
    uint group,
    uint groups_per_row) {
  int8 values = (int8)(0);
#define IQ36_Q4_LOAD_SUM(component, offset)                                  \
  do {                                                                         \
    const uint row = row_start + (offset);                                     \
    if (row < row_end) {                                                       \
      const uint index = row * groups_per_row + group;                         \
      values.component = (int)sum_low[index] +                                \
          256 * (int)sum_high[index];                                          \
    }                                                                          \
  } while (0)
  IQ36_Q4_LOAD_SUM(s0, 0U);
  IQ36_Q4_LOAD_SUM(s1, 1U);
  IQ36_Q4_LOAD_SUM(s2, 2U);
  IQ36_Q4_LOAD_SUM(s3, 3U);
  IQ36_Q4_LOAD_SUM(s4, 4U);
  IQ36_Q4_LOAD_SUM(s5, 5U);
  IQ36_Q4_LOAD_SUM(s6, 6U);
  IQ36_Q4_LOAD_SUM(s7, 7U);
#undef IQ36_Q4_LOAD_SUM
  return values;
}

float8 iq36_q4_load_source_scales8(
    __global const float *source_scales,
    uint row_start,
    uint row_end,
    uint block,
    uint blocks_per_row) {
  float8 values = (float8)(0.0f);
#define IQ36_Q4_LOAD_SOURCE_D(component, offset)                             \
  do {                                                                         \
    const uint row = row_start + (offset);                                     \
    if (row < row_end) {                                                       \
      values.component = source_scales[row * blocks_per_row + block];         \
    }                                                                          \
  } while (0)
  IQ36_Q4_LOAD_SOURCE_D(s0, 0U);
  IQ36_Q4_LOAD_SOURCE_D(s1, 1U);
  IQ36_Q4_LOAD_SOURCE_D(s2, 2U);
  IQ36_Q4_LOAD_SOURCE_D(s3, 3U);
  IQ36_Q4_LOAD_SOURCE_D(s4, 4U);
  IQ36_Q4_LOAD_SOURCE_D(s5, 5U);
  IQ36_Q4_LOAD_SOURCE_D(s6, 6U);
  IQ36_Q4_LOAD_SOURCE_D(s7, 7U);
#undef IQ36_Q4_LOAD_SOURCE_D
  return values;
}

uint8 iq36_q4_unpack_weights32(
    __global const uchar *weights, ulong byte_index) {
  const uchar16 packed = vload16(0, weights + byte_index);
  const uchar16 first = (uchar16)(
      packed.s0 & 15U, packed.s0 >> 4, packed.s1 & 15U, packed.s1 >> 4,
      packed.s2 & 15U, packed.s2 >> 4, packed.s3 & 15U, packed.s3 >> 4,
      packed.s4 & 15U, packed.s4 >> 4, packed.s5 & 15U, packed.s5 >> 4,
      packed.s6 & 15U, packed.s6 >> 4, packed.s7 & 15U, packed.s7 >> 4);
  const uchar16 second = (uchar16)(
      packed.s8 & 15U, packed.s8 >> 4, packed.s9 & 15U, packed.s9 >> 4,
      packed.sa & 15U, packed.sa >> 4, packed.sb & 15U, packed.sb >> 4,
      packed.sc & 15U, packed.sc >> 4, packed.sd & 15U, packed.sd >> 4,
      packed.se & 15U, packed.se >> 4, packed.sf & 15U, packed.sf >> 4);
  const uint4 first_u32 = as_uint4(first);
  const uint4 second_u32 = as_uint4(second);
  return (uint8)(first_u32.s0, first_u32.s1, first_u32.s2, first_u32.s3,
                 second_u32.s0, second_u32.s1, second_u32.s2,
                 second_u32.s3);
}

float iq36_ggml_avx2_expf(float value) {
  const float r = 0x1.8p23f;
  const float z = fma(value, 0x1.715476p+0f, r);
  const float n = z - r;
  const float b = fma(
      -n, 0x1.7f7d1cp-20f,
      fma(-n, 0x1.62e4p-1f, value));
  const uint e = as_uint(z) << 23;
  const float k = as_float(e + as_uint(1.0f));
  const float u = b * b;
  const float j = fma(
      fma(fma(0x1.0e4020p-7f, b, 0x1.573e2ep-5f), u,
          fma(0x1.555e66p-3f, b, 0x1.fffdb6p-2f)),
      u, 0x1.ffffecp-1f * b);
  if (fabs(n) <= 126.0f) return fma(k, j, k);
  const uint g = (n <= 0.0f ? 0xffffffffU : 0U) & 0x82000000U;
  const float s1 = as_float(g + 0x7f000000U);
  if (fabs(n) > 192.0f) return s1 * s1;
  const float s2 = as_float(e - g);
  return fma(s2, j, s2) * s1;
}

float iq36_q4_silu(float value) {
  return value / (1.0f + iq36_ggml_avx2_expf(0.0f - value));
}

void iq36_q4_store_swiglu8(
    __global half *output_f16,
    __global float *output_f32,
    uint row_start,
    uint row_end,
    uint output_column,
    float8 values) {
  const uint lane = get_sub_group_local_id();
  if ((lane & 1U) != 0U) return;
#define IQ36_Q4_STORE_SWIGLU(component, offset)                              \
  do {                                                                         \
    const uint row = row_start + (offset);                                     \
    if (row < row_end) {                                                       \
      const float up = intel_sub_group_shuffle(                               \
          values.component, lane | 1U);                                       \
      const float result = iq36_q4_silu(values.component) * up;               \
      const uint index = row * 512U + output_column / 2U;                     \
      output_f16[index] = convert_half_rte(result);                            \
      output_f32[index] = result;                                              \
    }                                                                          \
  } while (0)
  IQ36_Q4_STORE_SWIGLU(s0, 0U);
  IQ36_Q4_STORE_SWIGLU(s1, 1U);
  IQ36_Q4_STORE_SWIGLU(s2, 2U);
  IQ36_Q4_STORE_SWIGLU(s3, 3U);
  IQ36_Q4_STORE_SWIGLU(s4, 4U);
  IQ36_Q4_STORE_SWIGLU(s5, 5U);
  IQ36_Q4_STORE_SWIGLU(s6, 6U);
  IQ36_Q4_STORE_SWIGLU(s7, 7U);
#undef IQ36_Q4_STORE_SWIGLU
}

void iq36_q4_store_down8(
    __global float *output,
    __global const float *router_weights,
    uint row_start,
    uint row_end,
    uint output_column,
    float8 values) {
#define IQ36_Q4_STORE_DOWN(component, offset)                                \
  do {                                                                         \
    const uint row = row_start + (offset);                                     \
    if (row < row_end) {                                                       \
      output[row * 2048U + output_column] =                                   \
          values.component * router_weights[row];                             \
    }                                                                          \
  } while (0)
  IQ36_Q4_STORE_DOWN(s0, 0U);
  IQ36_Q4_STORE_DOWN(s1, 1U);
  IQ36_Q4_STORE_DOWN(s2, 2U);
  IQ36_Q4_STORE_DOWN(s3, 3U);
  IQ36_Q4_STORE_DOWN(s4, 4U);
  IQ36_Q4_STORE_DOWN(s5, 5U);
  IQ36_Q4_STORE_DOWN(s6, 6U);
  IQ36_Q4_STORE_DOWN(s7, 7U);
#undef IQ36_Q4_STORE_DOWN
}

#define IQ36_Q4_EXACT_BLOCK_ACCUMULATE(                                      \
    row_offset, value_accumulator, scale_accumulator, min_accumulator,       \
    row_stride, groups_per_row)                                               \
  do {                                                                         \
    const uint rows = row_start + (row_offset);                                \
    if (rows < row_end) {                                                      \
      const short8 input = iq36_q4_load_rows8(                                \
          source, rows, row_end, group, lane, row_stride);                    \
      const int8 dot = intel_sub_group_i8_u8_matrix_mad_k32(                  \
          input, unpacked_weight, (int8)(0));                                  \
      const int8 sums = iq36_q4_load_sums8(                                   \
          sum_low, sum_high, rows, row_end, group, groups_per_row);           \
      scale_accumulator += dot * scale_code;                                   \
      min_accumulator += sums * min_code;                                      \
    }                                                                          \
    if (group_in_block == 7U) {                                                \
      const float8 source_d = iq36_q4_load_source_scales8(                    \
          source_scales, rows, row_end, block, blocks_per_row);               \
      value_accumulator = fma(                                                 \
          convert_float8(scale_accumulator), source_d * weight_d,             \
          value_accumulator);                                                  \
      value_accumulator = fma(                                                 \
          convert_float8(min_accumulator), -source_d * weight_dmin,           \
          value_accumulator);                                                  \
      scale_accumulator = (int8)(0);                                           \
      min_accumulator = (int8)(0);                                             \
    }                                                                          \
  } while (0)

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(16, 1, 1)))
__kernel void iq36_grouped_q4_exact_block_gateup_m16(
    __global const uchar *weights,
    __global const uchar *scale_codes,
    __global const uchar *min_codes,
    __global const float *block_ds,
    __global const float *block_dmins,
    __global const int *expert_offsets,
    __global const uint *task_coordinates,
    __global const char *source,
    __global const float *source_scales,
    __global const char *sum_low,
    __global const char *sum_high,
    __global half *output_f16,
    __global float *output_f32) {
  const uint coordinates = task_coordinates[get_group_id(0)];
  const uint expert = coordinates & 255U;
  const uint output_tile = (coordinates >> 8) & 127U;
  if (output_tile >= 64U) return;
  const uint m_tile = coordinates >> 15;
  const uint expert_begin = expert == 0U
      ? 0U : (uint)expert_offsets[expert - 1U];
  const uint row_end = (uint)expert_offsets[expert];
  const uint row_start = expert_begin + m_tile * 16U;
  const uint lane = get_sub_group_local_id();
  const uint output_column = output_tile * 16U + lane;
  const uint blocks_per_row = 8U;
  const uint groups_per_row = 64U;
  float8 values0 = (float8)(0.0f);
  float8 values1 = (float8)(0.0f);
  int8 scale_sum0 = (int8)(0), scale_sum1 = (int8)(0);
  int8 min_sum0 = (int8)(0), min_sum1 = (int8)(0);
  for (uint group = 0U; group < groups_per_row; ++group) {
    const uint block = group >> 3;
    const uint group_in_block = group & 7U;
    const ulong compact_block =
        ((ulong)expert * 1024UL + output_column) * blocks_per_row + block;
    const uchar scale_code = scale_codes[compact_block * 8UL + group_in_block];
    const uchar min_code = min_codes[compact_block * 8UL + group_in_block];
    const float weight_d = block_ds[compact_block];
    const float weight_dmin = block_dmins[compact_block];
    const ulong weight_index =
        (((ulong)expert * 1024UL + output_column) * 1024UL) +
        (ulong)group * 16UL;
    const uint8 unpacked_weight =
        iq36_q4_unpack_weights32(weights, weight_index);
    IQ36_Q4_EXACT_BLOCK_ACCUMULATE(
        0U, values0, scale_sum0, min_sum0, 2048U, groups_per_row);
    IQ36_Q4_EXACT_BLOCK_ACCUMULATE(
        8U, values1, scale_sum1, min_sum1, 2048U, groups_per_row);
  }
  iq36_q4_store_swiglu8(
      output_f16, output_f32, row_start, row_end, output_column, values0);
  iq36_q4_store_swiglu8(
      output_f16, output_f32, row_start + 8U, row_end, output_column, values1);
}

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(16, 1, 1)))
__kernel void iq36_grouped_q4_exact_block_down_m16_f32(
    __global const uchar *weights,
    __global const uchar *scale_codes,
    __global const uchar *min_codes,
    __global const float *block_ds,
    __global const float *block_dmins,
    __global const int *expert_offsets,
    __global const uint *task_coordinates,
    __global const char *source,
    __global const float *source_scales,
    __global const char *sum_low,
    __global const char *sum_high,
    __global const float *router_weights,
    __global float *output) {
  const uint coordinates = task_coordinates[get_group_id(0)];
  const uint expert = coordinates & 255U;
  const uint output_tile = (coordinates >> 8) & 127U;
  const uint m_tile = coordinates >> 15;
  const uint expert_begin = expert == 0U
      ? 0U : (uint)expert_offsets[expert - 1U];
  const uint row_end = (uint)expert_offsets[expert];
  const uint row_start = expert_begin + m_tile * 16U;
  const uint lane = get_sub_group_local_id();
  const uint output_column = output_tile * 16U + lane;
  const uint blocks_per_row = 2U;
  const uint groups_per_row = 16U;
  float8 values0 = (float8)(0.0f);
  float8 values1 = (float8)(0.0f);
  int8 scale_sum0 = (int8)(0), scale_sum1 = (int8)(0);
  int8 min_sum0 = (int8)(0), min_sum1 = (int8)(0);
  for (uint group = 0U; group < groups_per_row; ++group) {
    const uint block = group >> 3;
    const uint group_in_block = group & 7U;
    const ulong compact_block =
        ((ulong)expert * 2048UL + output_column) * blocks_per_row + block;
    const uchar scale_code = scale_codes[compact_block * 8UL + group_in_block];
    const uchar min_code = min_codes[compact_block * 8UL + group_in_block];
    const float weight_d = block_ds[compact_block];
    const float weight_dmin = block_dmins[compact_block];
    const ulong weight_index =
        (((ulong)expert * 2048UL + output_column) * 256UL) +
        (ulong)group * 16UL;
    const uint8 unpacked_weight =
        iq36_q4_unpack_weights32(weights, weight_index);
    IQ36_Q4_EXACT_BLOCK_ACCUMULATE(
        0U, values0, scale_sum0, min_sum0, 512U, groups_per_row);
    IQ36_Q4_EXACT_BLOCK_ACCUMULATE(
        8U, values1, scale_sum1, min_sum1, 512U, groups_per_row);
  }
  iq36_q4_store_down8(
      output, router_weights, row_start, row_end, output_column, values0);
  iq36_q4_store_down8(
      output, router_weights, row_start + 8U, row_end, output_column, values1);
}

#undef IQ36_Q4_EXACT_BLOCK_ACCUMULATE

__kernel void iq36_scatter_f16_contributions(
    __global const half *contributions,
    __global const int *token_rank_to_row,
    __global float *output) {
  const uint index = get_global_id(0);
  if (index >= 1024U * 2048U) return;
  const uint token = index >> 11;
  const uint hidden = index & 2047U;
  float sum = 0.0f;
  for (uint rank = 0U; rank < 8U; ++rank) {
    const int row = token_rank_to_row[token * 8U + rank];
    sum += convert_float(contributions[(uint)row * 2048U + hidden]);
  }
  output[index] = sum;
}

__kernel void iq36_scatter_f32_contributions(
    __global const float *contributions,
    __global const int *token_rank_to_row,
    __global float *output) {
  const uint index = get_global_id(0);
  if (index >= 1024U * 2048U) return;
  const uint token = index >> 11;
  const uint hidden = index & 2047U;
  float sum = 0.0f;
  for (uint rank = 0U; rank < 8U; ++rank) {
    const int row = token_rank_to_row[token * 8U + rank];
    sum += contributions[(uint)row * 2048U + hidden];
  }
  output[index] = sum;
}

__kernel void iq36_combine_q8_sums16(
    __global const char *sum_low,
    __global const char *sum_high,
    __global short *sums) {
  const uint index = get_global_id(0);
  if (index >= 8192U * 16U) return;
  sums[index] = (short)((int)sum_low[index] + 256 * (int)sum_high[index]);
}

__kernel void iq36_sum_q8_k16(
    __global const char *q8,
    __global short *sums) {
  const uint index = get_global_id(0);
  if (index >= 8192U * 32U) return;
  const uint row = index >> 5;
  const uint group = index & 31U;
  int sum = 0;
  for (uint within = 0U; within < 16U; ++within) {
    sum += (int)q8[row * 512U + group * 16U + within];
  }
  sums[index] = (short)sum;
}
