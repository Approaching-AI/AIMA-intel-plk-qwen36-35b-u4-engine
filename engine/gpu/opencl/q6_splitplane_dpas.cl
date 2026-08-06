#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable

uint iq36_pack_u8x4(uchar x0, uchar x1, uchar x2, uchar x3) {
  return (uint)x0 | ((uint)x1 << 8) | ((uint)x2 << 16) | ((uint)x3 << 24);
}

short iq36_pack_i8_pair(char x0, char x1) {
  const ushort bits = (ushort)(uchar)x0 | ((ushort)(uchar)x1 << 8);
  return as_short(bits);
}

float iq36_half_to_float(ushort h) {
  uint sign = ((uint)h & 0x8000U) << 16;
  uint exp = ((uint)h >> 10) & 0x1FU;
  uint mantissa = (uint)h & 0x03FFU;
  uint bits = 0U;
  if (exp == 0U) {
    if (mantissa == 0U) {
      bits = sign;
    } else {
      uint shift = 0U;
      while ((mantissa & 0x0400U) == 0U) {
        mantissa <<= 1;
        shift += 1U;
      }
      mantissa &= 0x03FFU;
      bits = sign | ((127U - 14U - shift) << 23) | (mantissa << 13);
    }
  } else if (exp == 0x1FU) {
    bits = sign | 0x7F800000U | (mantissa << 13);
  } else {
    bits = sign | ((exp + 112U) << 23) | (mantissa << 13);
  }
  return as_float(bits);
}

ushort iq36_load_u16(__global const uchar* p) {
  return (ushort)p[0] | ((ushort)p[1] << 8);
}

uchar iq36_q6_value(__global const uchar* block, uint group, uint index) {
  const uint half_index = group >> 3;
  const uint group_in_half = group & 7U;
  const uint quadrant = group_in_half >> 1;
  const uint subgroup = group_in_half & 1U;
  const uint lane = subgroup * 16U + index;
  const uint ql_base = half_index * 64U;
  const uint qh_base = 128U + half_index * 32U;
  const uchar high = block[qh_base + lane];
  if (quadrant == 0U) {
    return (block[ql_base + lane] & (uchar)15) |
           (uchar)(((high >> 0) & 3U) << 4);
  }
  if (quadrant == 1U) {
    return (block[ql_base + 32U + lane] & (uchar)15) |
           (uchar)(((high >> 2) & 3U) << 4);
  }
  if (quadrant == 2U) {
    return (block[ql_base + lane] >> 4) |
           (uchar)(((high >> 4) & 3U) << 4);
  }
  return (block[ql_base + 32U + lane] >> 4) |
         (uchar)(((high >> 6) & 3U) << 4);
}

uint8 iq36_pack_q6_group16(__global const uchar* block, uint group) {
  return (uint8)(
      iq36_pack_u8x4(
          iq36_q6_value(block, group, 0U),
          iq36_q6_value(block, group, 1U),
          iq36_q6_value(block, group, 2U),
          iq36_q6_value(block, group, 3U)),
      iq36_pack_u8x4(
          iq36_q6_value(block, group, 4U),
          iq36_q6_value(block, group, 5U),
          iq36_q6_value(block, group, 6U),
          iq36_q6_value(block, group, 7U)),
      iq36_pack_u8x4(
          iq36_q6_value(block, group, 8U),
          iq36_q6_value(block, group, 9U),
          iq36_q6_value(block, group, 10U),
          iq36_q6_value(block, group, 11U)),
      iq36_pack_u8x4(
          iq36_q6_value(block, group, 12U),
          iq36_q6_value(block, group, 13U),
          iq36_q6_value(block, group, 14U),
          iq36_q6_value(block, group, 15U)),
      0U, 0U, 0U, 0U);
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void q6k_splitplane_dpas_rowtile16(
    __global const uchar* q6_rows,
    __global const char* q8_qs,
    __global const short* q8_bsums,
    __global const float* q8_d,
    uint rows,
    uint blocks_per_row,
    __global float* output) {
  const uint row = get_global_id(0);
  const uint lane = get_sub_group_local_id();
  const uint row_bytes = blocks_per_row * 210U;
  __global const uchar* row_raw = q6_rows + (ulong)row * (ulong)row_bytes;
  float sum = 0.0f;

  for (uint block_index = 0U; block_index < blocks_per_row; ++block_index) {
    __global const uchar* block = row_raw + block_index * 210U;
    int block_sum = 0;
    for (uint group = 0U; group < 16U; ++group) {
      const uint input_base = block_index * 256U + group * 16U;
      short a = (short)0;
      if (lane < 8U) {
        a = iq36_pack_i8_pair(
            q8_qs[input_base + lane * 2U],
            q8_qs[input_base + lane * 2U + 1U]);
      }
      const uint8 b = iq36_pack_q6_group16(block, group);
      const int unsigned_dot =
          intel_sub_group_i8_u8_matrix_mad_k32(a, b, 0);
      const int centered_dot =
          unsigned_dot - 32 * (int)q8_bsums[block_index * 16U + group];
      const int scale = (int)((__global const char*)(block + 192U))[group];
      block_sum += scale * centered_dot;
    }
    const float combined_scale =
        iq36_half_to_float(iq36_load_u16(block + 208U)) * q8_d[block_index];
    sum = fma(combined_scale, (float)block_sum, sum);
  }
  if (row < rows) {
    output[row] = sum;
  }
}
