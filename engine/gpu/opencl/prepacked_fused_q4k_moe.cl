#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable

int nearest_int(float value) {
  const float shifted = value + 12582912.0f;
  return (as_int(shifted) & 0x007fffff) - 0x00400000;
}

float swiglu(float gate, float up) {
  const float sigmoid =
      gate >= 0.0f ? 1.0f / (1.0f + exp(-gate))
                   : exp(gate) / (1.0f + exp(gate));
  return gate * sigmoid * up;
}

short pack_q8_pair_local(__local const char * values, uint index) {
  const uchar lo = as_uchar(values[index]);
  const uchar hi = as_uchar(values[index + 1U]);
  return as_short((ushort)((ushort)lo | ((ushort)hi << 8)));
}

uint4 load_u4_group(__global const uchar * codes, uint weight,
                    uint output, uint group, uint outputs,
                    uint groups) {
  const ulong offset =
      (((ulong)weight * outputs + output) * groups + group) * 16UL;
  return vload4(0, (__global const uint *)(codes + offset));
}

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void quantize_input_q8k(
    __global const float * input,
    __global char * q8,
    __global float * scales,
    __global float * sums32_scaled) {
  const uint lane = get_local_id(0);
  const uint group = get_group_id(0);
  const uint row = group >> 3;
  const uint block = group & 7U;
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
  const int quantized_value = inverse_scale == 0.0f
      ? 0 : min(127, nearest_int(inverse_scale * value));
  quantized[lane] = quantized_value;
  q8[row * 2048U + block * 256U + lane] = (char)quantized_value;
  if (lane == 0U) scales[row * 8U + block] = scale;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lane < 8U) {
    int sum = 0;
    for (uint index = 0U; index < 32U; ++index) {
      sum += quantized[lane * 32U + index];
    }
    sums32_scaled[row * 64U + block * 8U + lane] =
        scale * (float)sum;
  }
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void prepacked_q4k_gateup_swiglu(
    __global const uchar * codes,
    __global const float * weight_scales,
    __global const float * min_weights,
    __global const char * q8,
    __global const float * source_scales,
    __global const float * source_sums,
    __global const uint * task_weight,
    __global const uint * task_base,
    __global const uint * task_count,
    __global const uint * bucket_token,
    __global float * output,
    uint task_total,
    uint bucket_m,
    __local char * q8_tile) {
  const uint workgroup = get_group_id(0);
  const uint task = workgroup >> 3;
  const uint n_block = workgroup & 7U;
  if (task >= task_total) return;
  const uint subgroup = get_sub_group_id();
  const uint lane = get_sub_group_local_id();
  const uint n_tile = subgroup & 3U;
  const uint m_tile = subgroup >> 2;
  const uint inner = n_block * 64U + n_tile * 16U + lane;
  const uint row_base = m_tile * 8U;
  const uint weight = task_weight[task];
  const uint bucket_base = task_base[task];
  const uint row_count = task_count[task];
  float8 gate_main = (float8)(0.0f);
  float8 gate_min = (float8)(0.0f);
  float8 up_main = (float8)(0.0f);
  float8 up_min = (float8)(0.0f);

  for (uint block = 0U; block < 8U; ++block) {
    const uint tile_values = bucket_m * 256U;
    for (uint offset = get_local_id(0); offset < tile_values;
         offset += get_local_size(0)) {
      const uint row = offset >> 8;
      const uint k = offset & 255U;
      char value = (char)0;
      if (row < row_count) {
        const uint token = bucket_token[bucket_base + row];
        value = q8[token * 2048U + block * 256U + k];
      }
      q8_tile[offset] = value;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    float8 q8_scale = (float8)(0.0f);
    for (uint m = 0U; m < 8U; ++m) {
      const uint row = row_base + m;
      if (row < row_count) {
        const uint token = bucket_token[bucket_base + row];
        q8_scale[m] = source_scales[token * 8U + block];
      }
    }
    for (uint group_in_block = 0U; group_in_block < 8U;
         ++group_in_block) {
      const uint group = block * 8U + group_in_block;
      short8 pairs = (short8)(0);
      float8 sums = (float8)(0.0f);
      for (uint m = 0U; m < 8U; ++m) {
        const uint row = row_base + m;
        if (row < row_count) {
          const uint local_index =
              row * 256U + group_in_block * 32U + lane * 2U;
          pairs[m] = pack_q8_pair_local(q8_tile, local_index);
          const uint token = bucket_token[bucket_base + row];
          sums[m] = source_sums[token * 64U + group];
        }
      }
      const uint4 gate_codes =
          load_u4_group(codes, weight, inner, group, 1024U, 64U);
      const uint4 up_codes =
          load_u4_group(codes, weight, 512U + inner, group, 1024U, 64U);
      const int8 gate_dot = intel_sub_group_i8_u4_matrix_mad_k32(
          pairs, gate_codes, (int8)(0));
      const int8 up_dot = intel_sub_group_i8_u4_matrix_mad_k32(
          pairs, up_codes, (int8)(0));
      const ulong gate_index =
          ((ulong)weight * 1024UL + inner) * 64UL + group;
      const ulong up_index =
          ((ulong)weight * 1024UL + 512UL + inner) * 64UL + group;
      gate_main += convert_float8(gate_dot) * q8_scale *
                   weight_scales[gate_index];
      up_main += convert_float8(up_dot) * q8_scale *
                 weight_scales[up_index];
      gate_min += sums * min_weights[gate_index];
      up_min += sums * min_weights[up_index];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  for (uint m = 0U; m < 8U; ++m) {
    const uint row = row_base + m;
    if (row < row_count) {
      const float gate = gate_main[m] - gate_min[m];
      const float up = up_main[m] - up_min[m];
      output[(bucket_base + row) * 512U + inner] = swiglu(gate, up);
    }
  }
}

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void quantize_swiglu_q8k(
    __global const float * input,
    __global char * q8,
    __global float * scales,
    __global float * sums32_scaled) {
  const uint lane = get_local_id(0);
  const uint group = get_group_id(0);
  const uint row = group >> 1;
  const uint block = group & 1U;
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
  const int quantized_value = inverse_scale == 0.0f
      ? 0 : min(127, nearest_int(inverse_scale * value));
  quantized[lane] = quantized_value;
  q8[row * 512U + block * 256U + lane] = (char)quantized_value;
  if (lane == 0U) scales[row * 2U + block] = scale;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lane < 8U) {
    int sum = 0;
    for (uint index = 0U; index < 32U; ++index) {
      sum += quantized[lane * 32U + index];
    }
    sums32_scaled[row * 16U + block * 8U + lane] =
        scale * (float)sum;
  }
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void prepacked_q4k_down_weighted(
    __global const uchar * codes,
    __global const float * weight_scales,
    __global const float * min_weights,
    __global const char * q8,
    __global const float * source_scales,
    __global const float * source_sums,
    __global const uint * task_weight,
    __global const uint * task_base,
    __global const uint * task_count,
    __global const float * bucket_weights,
    __global float * contributions,
    uint task_total,
    uint bucket_m,
    __local char * q8_tile) {
  const uint workgroup = get_group_id(0);
  const uint task = workgroup >> 5;
  const uint n_block = workgroup & 31U;
  if (task >= task_total) return;
  const uint subgroup = get_sub_group_id();
  const uint lane = get_sub_group_local_id();
  const uint n_tile = subgroup & 3U;
  const uint m_tile = subgroup >> 2;
  const uint hidden = n_block * 64U + n_tile * 16U + lane;
  const uint row_base = m_tile * 8U;
  const uint weight = task_weight[task];
  const uint bucket_base = task_base[task];
  const uint row_count = task_count[task];
  float8 main_term = (float8)(0.0f);
  float8 min_term = (float8)(0.0f);

  for (uint block = 0U; block < 2U; ++block) {
    const uint tile_values = bucket_m * 256U;
    for (uint offset = get_local_id(0); offset < tile_values;
         offset += get_local_size(0)) {
      const uint row = offset >> 8;
      const uint k = offset & 255U;
      const uint bucket = bucket_base + row;
      q8_tile[offset] = row < row_count
          ? q8[bucket * 512U + block * 256U + k] : (char)0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    float8 q8_scale = (float8)(0.0f);
    for (uint m = 0U; m < 8U; ++m) {
      const uint row = row_base + m;
      if (row < row_count) {
        q8_scale[m] = source_scales[(bucket_base + row) * 2U + block];
      }
    }
    for (uint group_in_block = 0U; group_in_block < 8U;
         ++group_in_block) {
      const uint group = block * 8U + group_in_block;
      short8 pairs = (short8)(0);
      float8 sums = (float8)(0.0f);
      for (uint m = 0U; m < 8U; ++m) {
        const uint row = row_base + m;
        if (row < row_count) {
          const uint local_index =
              row * 256U + group_in_block * 32U + lane * 2U;
          pairs[m] = pack_q8_pair_local(q8_tile, local_index);
          sums[m] = source_sums[(bucket_base + row) * 16U + group];
        }
      }
      const uint4 packed =
          load_u4_group(codes, weight, hidden, group, 2048U, 16U);
      const int8 dot = intel_sub_group_i8_u4_matrix_mad_k32(
          pairs, packed, (int8)(0));
      const ulong weight_index =
          ((ulong)weight * 2048UL + hidden) * 16UL + group;
      main_term += convert_float8(dot) * q8_scale *
                   weight_scales[weight_index];
      min_term += sums * min_weights[weight_index];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  for (uint m = 0U; m < 8U; ++m) {
    const uint row = row_base + m;
    if (row < row_count) {
      const uint bucket = bucket_base + row;
      contributions[bucket * 2048U + hidden] =
          (main_term[m] - min_term[m]) * bucket_weights[bucket];
    }
  }
}

__kernel void scatter_routed_output(
    __global const float * contributions,
    __global const int * token_rank_to_bucket,
    __global float * output) {
  const uint index = get_global_id(0);
  if (index >= 1024U * 2048U) return;
  const uint token = index / 2048U;
  const uint hidden = index - token * 2048U;
  float sum = 0.0f;
  for (uint rank = 0U; rank < 8U; ++rank) {
    const int bucket = token_rank_to_bucket[token * 8U + rank];
    sum += contributions[(uint)bucket * 2048U + hidden];
  }
  output[index] = sum;
}
