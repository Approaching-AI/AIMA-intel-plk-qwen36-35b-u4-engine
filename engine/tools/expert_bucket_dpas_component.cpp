#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kTokenCount = 1024;
constexpr std::size_t kHiddenSize = 2048;
constexpr std::size_t kExpertCount = 256;
constexpr std::size_t kSelectedExperts = 8;
constexpr std::size_t kIntermediateSize = 512;
constexpr std::size_t kAssignments = kTokenCount * kSelectedExperts;
constexpr std::size_t kQ4BlockBytes = 144;
constexpr std::size_t kQ8BlockBytes = 292;
constexpr std::size_t kBlocksPerRow = kHiddenSize / 256;
constexpr std::size_t kQ4RowBytes = kBlocksPerRow * kQ4BlockBytes;
constexpr std::size_t kQ8TokenBytes = kBlocksPerRow * kQ8BlockBytes;
constexpr std::size_t kRowsPerExpert = kIntermediateSize * 2;
constexpr std::size_t kExpertWeightBytes =
    kRowsPerExpert * kQ4RowBytes;
constexpr std::size_t kExpectedWeightBytes =
    kExpertCount * kExpertWeightBytes;
constexpr std::size_t kDownInputSize = 512;
constexpr std::size_t kDownOutputSize = 2048;
constexpr std::size_t kDownBlocksPerRow = 2;
constexpr std::size_t kDownRowsPerExpert = kDownOutputSize;
constexpr std::size_t kDownExpertWeightBytes =
    kDownRowsPerExpert * kDownBlocksPerRow * kQ4BlockBytes;
constexpr std::size_t kExpectedDownWeightBytes =
    kExpertCount * kDownExpertWeightBytes;
constexpr std::size_t kM1TaskTokens = 128;
constexpr std::size_t kM8TaskTokens = 64;
constexpr double kMismatchThreshold = 5e-3;

const char* kOpenClSource = R"CLC(
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable

uint load_u16_local(const __local uchar * p) {
  return (uint)p[0] | ((uint)p[1] << 8);
}

short load_i16_global(const __global uchar * p) {
  ushort bits = (ushort)p[0] | ((ushort)p[1] << 8);
  return as_short(bits);
}

float load_f32_global(const __global uchar * p) {
  uint bits = (uint)p[0] | ((uint)p[1] << 8) |
              ((uint)p[2] << 16) | ((uint)p[3] << 24);
  return as_float(bits);
}

float half_to_float(uint h) {
  uint sign = (h & 0x8000U) << 16;
  uint exp = (h >> 10) & 0x1FU;
  uint mantissa = h & 0x03FFU;
  uint out = 0U;
  if (exp == 0U) {
    if (mantissa == 0U) {
      out = sign;
    } else {
      uint shift = 0U;
      while ((mantissa & 0x0400U) == 0U) {
        mantissa <<= 1;
        shift += 1U;
      }
      mantissa &= 0x03FFU;
      out = sign | ((127U - 14U - shift) << 23) | (mantissa << 13);
    }
  } else if (exp == 0x1FU) {
    out = sign | 0x7F800000U | (mantissa << 13);
  } else {
    out = sign | ((exp + 112U) << 23) | (mantissa << 13);
  }
  return as_float(out);
}

uchar get_scale_k4_local(int j, const __local uchar * q) {
  if (j < 4) {
    return q[j] & 63U;
  }
  return (q[j + 4] & 0x0FU) | ((q[j - 4] >> 6) << 4);
}

uchar get_min_k4_local(int j, const __local uchar * q) {
  if (j < 4) {
    return q[j + 4] & 63U;
  }
  return (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
}

char q8_value(const __global uchar * q8_block, uint index) {
  return (char)q8_block[4U + index];
}

short q8_bsum(const __global uchar * q8_block, uint index) {
  return load_i16_global(q8_block + 260U + index * 2U);
}

uint pack_u8x4(uchar x0, uchar x1, uchar x2, uchar x3) {
  return (uint)x0 | ((uint)x1 << 8) | ((uint)x2 << 16) |
         ((uint)x3 << 24);
}

int q4_value_local(const __local uchar * qs, uint index) {
  uint segment = index / 64U;
  uint offset = index - segment * 64U;
  uchar packed = qs[segment * 32U + (offset & 31U)];
  return offset < 32U ? (int)(packed & 0x0FU) : (int)(packed >> 4);
}

uint8 pack_q4_row32_local(const __local uchar * qs, uint k_base) {
  return (uint8)(
      pack_u8x4((uchar)q4_value_local(qs, k_base + 0U),
                (uchar)q4_value_local(qs, k_base + 1U),
                (uchar)q4_value_local(qs, k_base + 2U),
                (uchar)q4_value_local(qs, k_base + 3U)),
      pack_u8x4((uchar)q4_value_local(qs, k_base + 4U),
                (uchar)q4_value_local(qs, k_base + 5U),
                (uchar)q4_value_local(qs, k_base + 6U),
                (uchar)q4_value_local(qs, k_base + 7U)),
      pack_u8x4((uchar)q4_value_local(qs, k_base + 8U),
                (uchar)q4_value_local(qs, k_base + 9U),
                (uchar)q4_value_local(qs, k_base + 10U),
                (uchar)q4_value_local(qs, k_base + 11U)),
      pack_u8x4((uchar)q4_value_local(qs, k_base + 12U),
                (uchar)q4_value_local(qs, k_base + 13U),
                (uchar)q4_value_local(qs, k_base + 14U),
                (uchar)q4_value_local(qs, k_base + 15U)),
      pack_u8x4((uchar)q4_value_local(qs, k_base + 16U),
                (uchar)q4_value_local(qs, k_base + 17U),
                (uchar)q4_value_local(qs, k_base + 18U),
                (uchar)q4_value_local(qs, k_base + 19U)),
      pack_u8x4((uchar)q4_value_local(qs, k_base + 20U),
                (uchar)q4_value_local(qs, k_base + 21U),
                (uchar)q4_value_local(qs, k_base + 22U),
                (uchar)q4_value_local(qs, k_base + 23U)),
      pack_u8x4((uchar)q4_value_local(qs, k_base + 24U),
                (uchar)q4_value_local(qs, k_base + 25U),
                (uchar)q4_value_local(qs, k_base + 26U),
                (uchar)q4_value_local(qs, k_base + 27U)),
      pack_u8x4((uchar)q4_value_local(qs, k_base + 28U),
                (uchar)q4_value_local(qs, k_base + 29U),
                (uchar)q4_value_local(qs, k_base + 30U),
                (uchar)q4_value_local(qs, k_base + 31U)));
}

uint pack_u4x8(uchar x0, uchar x1, uchar x2, uchar x3,
               uchar x4, uchar x5, uchar x6, uchar x7) {
  return (uint)x0 | ((uint)x1 << 4) | ((uint)x2 << 8) |
         ((uint)x3 << 12) | ((uint)x4 << 16) | ((uint)x5 << 20) |
         ((uint)x6 << 24) | ((uint)x7 << 28);
}

uint4 pack_q4_row32_u4_local(const __local uchar * qs, uint k_base) {
  return (uint4)(
      pack_u4x8(
          (uchar)q4_value_local(qs, k_base + 0U),
          (uchar)q4_value_local(qs, k_base + 1U),
          (uchar)q4_value_local(qs, k_base + 2U),
          (uchar)q4_value_local(qs, k_base + 3U),
          (uchar)q4_value_local(qs, k_base + 4U),
          (uchar)q4_value_local(qs, k_base + 5U),
          (uchar)q4_value_local(qs, k_base + 6U),
          (uchar)q4_value_local(qs, k_base + 7U)),
      pack_u4x8(
          (uchar)q4_value_local(qs, k_base + 8U),
          (uchar)q4_value_local(qs, k_base + 9U),
          (uchar)q4_value_local(qs, k_base + 10U),
          (uchar)q4_value_local(qs, k_base + 11U),
          (uchar)q4_value_local(qs, k_base + 12U),
          (uchar)q4_value_local(qs, k_base + 13U),
          (uchar)q4_value_local(qs, k_base + 14U),
          (uchar)q4_value_local(qs, k_base + 15U)),
      pack_u4x8(
          (uchar)q4_value_local(qs, k_base + 16U),
          (uchar)q4_value_local(qs, k_base + 17U),
          (uchar)q4_value_local(qs, k_base + 18U),
          (uchar)q4_value_local(qs, k_base + 19U),
          (uchar)q4_value_local(qs, k_base + 20U),
          (uchar)q4_value_local(qs, k_base + 21U),
          (uchar)q4_value_local(qs, k_base + 22U),
          (uchar)q4_value_local(qs, k_base + 23U)),
      pack_u4x8(
          (uchar)q4_value_local(qs, k_base + 24U),
          (uchar)q4_value_local(qs, k_base + 25U),
          (uchar)q4_value_local(qs, k_base + 26U),
          (uchar)q4_value_local(qs, k_base + 27U),
          (uchar)q4_value_local(qs, k_base + 28U),
          (uchar)q4_value_local(qs, k_base + 29U),
          (uchar)q4_value_local(qs, k_base + 30U),
          (uchar)q4_value_local(qs, k_base + 31U)));
}

short pack_q8_pair(const __global uchar * q8_block, uint k_base) {
  const uchar lo = as_uchar(q8_value(q8_block, k_base));
  const uchar hi = as_uchar(q8_value(q8_block, k_base + 1U));
  return as_short((ushort)((ushort)lo | ((ushort)hi << 8)));
}

int q4q8_dot32(uint8 q4_packed,
               const __global uchar * q8_block,
               uint k_base) {
  const uint lane = get_sub_group_local_id();
  const short q8_pair = pack_q8_pair(q8_block, k_base + lane * 2U);
  return intel_sub_group_i8_u8_matrix_mad_k32(q8_pair, q4_packed, 0);
}

int8 q4q8_dot32_m8_u4(short8 q8_pairs, uint4 q4_packed) {
  return intel_sub_group_i8_u4_matrix_mad_k32(
      q8_pairs, q4_packed, (int8)(0));
}

float swiglu(float gate, float up) {
  const float sigmoid =
      gate >= 0.0f ? 1.0f / (1.0f + exp(-gate))
                   : exp(gate) / (1.0f + exp(gate));
  return gate * sigmoid * up;
}

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(128, 1, 1)))
__kernel void expert_bucket_gateup_swiglu(
    __global const uchar * q4_rows,
    __global const uchar * q8_tokens,
    __global const uint * task_expert,
    __global const uint * task_bucket_base,
    __global const uint * task_token_count,
    __global const uint * bucket_token,
    __global float * output,
    uint task_total,
    __local uchar * q4_tile) {
  const uint row_tile_count = 32U;
  const uint workgroup = get_group_id(0);
  const uint task = workgroup / row_tile_count;
  const uint row_tile = workgroup - task * row_tile_count;
  if (task >= task_total) {
    return;
  }

  const uint local_id = get_local_id(0);
  const uint local_size = get_local_size(0);
  const uint subgroup = get_sub_group_id();
  const uint lane = get_sub_group_local_id();
  const uint expert = task_expert[task];
  const uint bucket_base = task_bucket_base[task];
  const uint token_count = task_token_count[task];
  const uint inner = row_tile * 16U + lane;

  float gate_sum[16];
  float gate_min[16];
  float up_sum[16];
  float up_min[16];
  for (uint t = 0U; t < 16U; ++t) {
    gate_sum[t] = 0.0f;
    gate_min[t] = 0.0f;
    up_sum[t] = 0.0f;
    up_min[t] = 0.0f;
  }

  const uint q4_row_bytes = 1152U;
  const uint q8_token_bytes = 2336U;
  const uint q4_block_bytes = 144U;
  const uint q8_block_bytes = 292U;
  const uint local_tile_bytes = 32U * q4_block_bytes;

  for (uint block = 0U; block < 8U; ++block) {
    for (uint offset = local_id; offset < local_tile_bytes;
         offset += local_size) {
      const uint plane_row = offset / q4_block_bytes;
      const uint byte = offset - plane_row * q4_block_bytes;
      const uint up_plane = plane_row >= 16U ? 1U : 0U;
      const uint row_lane = plane_row & 15U;
      const uint source_row = expert * 1024U + up_plane * 512U +
                              row_tile * 16U + row_lane;
      q4_tile[offset] = q4_rows[source_row * q4_row_bytes +
                                block * q4_block_bytes + byte];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const __local uchar * gate_block = q4_tile + lane * q4_block_bytes;
    const __local uchar * up_block =
        q4_tile + (16U + lane) * q4_block_bytes;
    const __local uchar * gate_scales = gate_block + 4U;
    const __local uchar * up_scales = up_block + 4U;
    const __local uchar * gate_qs = gate_block + 16U;
    const __local uchar * up_qs = up_block + 16U;
    const float gate_d = half_to_float(load_u16_local(gate_block));
    const float gate_dmin = half_to_float(load_u16_local(gate_block + 2U));
    const float up_d = half_to_float(load_u16_local(up_block));
    const float up_dmin = half_to_float(load_u16_local(up_block + 2U));

    for (uint scale_index = 0U; scale_index < 8U; ++scale_index) {
      const int gate_scale =
          (int)get_scale_k4_local((int)scale_index, gate_scales);
      const int up_scale =
          (int)get_scale_k4_local((int)scale_index, up_scales);
      const uint k_base = scale_index * 32U;
      const uint8 gate_packed = pack_q4_row32_local(gate_qs, k_base);
      const uint8 up_packed = pack_q4_row32_local(up_qs, k_base);
      for (uint t = 0U; t < 16U; ++t) {
        const uint local_token = subgroup * 16U + t;
        if (local_token >= token_count) {
          continue;
        }
        const uint bucket = bucket_base + local_token;
        const uint token = bucket_token[bucket];
        const __global uchar * q8_block =
            q8_tokens + token * q8_token_bytes + block * q8_block_bytes;
        const float q8_d = load_f32_global(q8_block);
        const int gate_dot = q4q8_dot32(gate_packed, q8_block, k_base);
        const int up_dot = q4q8_dot32(up_packed, q8_block, k_base);
        gate_sum[t] += gate_d * q8_d * (float)(gate_scale * gate_dot);
        up_sum[t] += up_d * q8_d * (float)(up_scale * up_dot);
      }
    }

    for (uint bsum_index = 0U; bsum_index < 16U; ++bsum_index) {
      const int gate_minimum =
          (int)get_min_k4_local((int)(bsum_index >> 1), gate_scales);
      const int up_minimum =
          (int)get_min_k4_local((int)(bsum_index >> 1), up_scales);
      for (uint t = 0U; t < 16U; ++t) {
        const uint local_token = subgroup * 16U + t;
        if (local_token >= token_count) {
          continue;
        }
        const uint bucket = bucket_base + local_token;
        const uint token = bucket_token[bucket];
        const __global uchar * q8_block =
            q8_tokens + token * q8_token_bytes + block * q8_block_bytes;
        const int bsum = (int)q8_bsum(q8_block, bsum_index);
        const float q8_d = load_f32_global(q8_block);
        gate_min[t] += gate_dmin * q8_d * (float)(bsum * gate_minimum);
        up_min[t] += up_dmin * q8_d * (float)(bsum * up_minimum);
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  for (uint t = 0U; t < 16U; ++t) {
    const uint local_token = subgroup * 16U + t;
    if (local_token < token_count) {
      const uint bucket = bucket_base + local_token;
      const float gate = gate_sum[t] - gate_min[t];
      const float up = up_sum[t] - up_min[t];
      output[bucket * 512U + inner] = swiglu(gate, up);
    }
  }
}

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(128, 1, 1)))
__kernel void expert_bucket_gateup_swiglu_m8_u4(
    __global const uchar * q4_rows,
    __global const uchar * q8_tokens,
    __global const uint * task_expert,
    __global const uint * task_bucket_base,
    __global const uint * task_token_count,
    __global const uint * bucket_token,
    __global float * output,
    uint task_total,
    __local uchar * q4_tile) {
  const uint row_tile_count = 32U;
  const uint workgroup = get_group_id(0);
  const uint task = workgroup / row_tile_count;
  const uint row_tile = workgroup - task * row_tile_count;
  if (task >= task_total) {
    return;
  }

  const uint local_id = get_local_id(0);
  const uint local_size = get_local_size(0);
  const uint subgroup = get_sub_group_id();
  const uint lane = get_sub_group_local_id();
  const uint expert = task_expert[task];
  const uint bucket_base = task_bucket_base[task];
  const uint token_count = task_token_count[task];
  const uint inner = row_tile * 16U + lane;

  float8 gate_sum = (float8)(0.0f);
  float8 gate_min = (float8)(0.0f);
  float8 up_sum = (float8)(0.0f);
  float8 up_min = (float8)(0.0f);

  const uint q4_row_bytes = 1152U;
  const uint q8_token_bytes = 2336U;
  const uint q4_block_bytes = 144U;
  const uint q8_block_bytes = 292U;
  const uint local_tile_bytes = 32U * q4_block_bytes;

  for (uint block = 0U; block < 8U; ++block) {
    for (uint offset = local_id; offset < local_tile_bytes;
         offset += local_size) {
      const uint plane_row = offset / q4_block_bytes;
      const uint byte = offset - plane_row * q4_block_bytes;
      const uint up_plane = plane_row >= 16U ? 1U : 0U;
      const uint row_lane = plane_row & 15U;
      const uint source_row = expert * 1024U + up_plane * 512U +
                              row_tile * 16U + row_lane;
      q4_tile[offset] = q4_rows[source_row * q4_row_bytes +
                                block * q4_block_bytes + byte];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const __local uchar * gate_block = q4_tile + lane * q4_block_bytes;
    const __local uchar * up_block =
        q4_tile + (16U + lane) * q4_block_bytes;
    const __local uchar * gate_scales = gate_block + 4U;
    const __local uchar * up_scales = up_block + 4U;
    const __local uchar * gate_qs = gate_block + 16U;
    const __local uchar * up_qs = up_block + 16U;
    const float gate_d = half_to_float(load_u16_local(gate_block));
    const float gate_dmin = half_to_float(load_u16_local(gate_block + 2U));
    const float up_d = half_to_float(load_u16_local(up_block));
    const float up_dmin = half_to_float(load_u16_local(up_block + 2U));

    uint token_ids[8];
    float8 q8_scales = (float8)(0.0f);
    for (uint t = 0U; t < 8U; ++t) {
      const uint local_token = subgroup * 8U + t;
      if (local_token < token_count) {
        const uint token = bucket_token[bucket_base + local_token];
        token_ids[t] = token;
        const __global uchar * q8_block =
            q8_tokens + token * q8_token_bytes + block * q8_block_bytes;
        q8_scales[t] = load_f32_global(q8_block);
      } else {
        token_ids[t] = 0U;
        q8_scales[t] = 0.0f;
      }
    }

    for (uint scale_index = 0U; scale_index < 8U; ++scale_index) {
      const int gate_scale =
          (int)get_scale_k4_local((int)scale_index, gate_scales);
      const int up_scale =
          (int)get_scale_k4_local((int)scale_index, up_scales);
      const uint k_base = scale_index * 32U;
      const uint4 gate_packed = pack_q4_row32_u4_local(gate_qs, k_base);
      const uint4 up_packed = pack_q4_row32_u4_local(up_qs, k_base);
      short8 q8_pairs = (short8)(0);
      for (uint t = 0U; t < 8U; ++t) {
        const __global uchar * q8_block =
            q8_tokens + token_ids[t] * q8_token_bytes +
            block * q8_block_bytes;
        q8_pairs[t] = q8_scales[t] == 0.0f
            ? (short)0 : pack_q8_pair(q8_block, k_base + lane * 2U);
      }
      const int8 gate_dot = q4q8_dot32_m8_u4(q8_pairs, gate_packed);
      const int8 up_dot = q4q8_dot32_m8_u4(q8_pairs, up_packed);
      gate_sum += q8_scales * (gate_d * (float)gate_scale) *
                  convert_float8(gate_dot);
      up_sum += q8_scales * (up_d * (float)up_scale) *
                convert_float8(up_dot);
    }

    for (uint bsum_index = 0U; bsum_index < 16U; ++bsum_index) {
      const int gate_minimum =
          (int)get_min_k4_local((int)(bsum_index >> 1), gate_scales);
      const int up_minimum =
          (int)get_min_k4_local((int)(bsum_index >> 1), up_scales);
      int8 sums = (int8)(0);
      for (uint t = 0U; t < 8U; ++t) {
        const __global uchar * q8_block =
            q8_tokens + token_ids[t] * q8_token_bytes +
            block * q8_block_bytes;
        sums[t] = q8_scales[t] == 0.0f
            ? 0 : (int)q8_bsum(q8_block, bsum_index);
      }
      const float8 sums_f32 = convert_float8(sums);
      gate_min += q8_scales * (gate_dmin * (float)gate_minimum) * sums_f32;
      up_min += q8_scales * (up_dmin * (float)up_minimum) * sums_f32;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  for (uint t = 0U; t < 8U; ++t) {
    const uint local_token = subgroup * 8U + t;
    if (local_token < token_count) {
      const uint bucket = bucket_base + local_token;
      const float gate = gate_sum[t] - gate_min[t];
      const float up = up_sum[t] - up_min[t];
      output[bucket * 512U + inner] = swiglu(gate, up);
    }
  }
}
)CLC";

using cl_int = std::int32_t;
using cl_uint = std::uint32_t;
using cl_ulong = std::uint64_t;
using cl_bool = cl_uint;
using cl_bitfield = cl_ulong;
using cl_device_type = cl_bitfield;
using cl_platform_info = cl_uint;
using cl_device_info = cl_uint;
using cl_context_properties = std::intptr_t;
using cl_command_queue_properties = cl_bitfield;
using cl_mem_flags = cl_bitfield;
using cl_program_build_info = cl_uint;
using cl_profiling_info = cl_uint;
using cl_platform_id = struct _cl_platform_id*;
using cl_device_id = struct _cl_device_id*;
using cl_context = struct _cl_context*;
using cl_command_queue = struct _cl_command_queue*;
using cl_mem = struct _cl_mem*;
using cl_program = struct _cl_program*;
using cl_kernel = struct _cl_kernel*;
using cl_event = struct _cl_event*;

constexpr cl_int kClSuccess = 0;
constexpr cl_bool kClTrue = 1;
constexpr cl_device_type kClDeviceTypeGpu = 1ULL << 2;
constexpr cl_mem_flags kClMemReadWrite = 1ULL << 0;
constexpr cl_mem_flags kClMemReadOnly = 1ULL << 2;
constexpr cl_platform_info kClPlatformName = 0x0902;
constexpr cl_device_info kClDeviceName = 0x102B;
constexpr cl_program_build_info kClProgramBuildLog = 0x1183;
constexpr cl_profiling_info kClProfilingCommandStart = 0x1282;
constexpr cl_profiling_info kClProfilingCommandEnd = 0x1283;
constexpr cl_command_queue_properties kClQueueProfilingEnable = 1ULL << 1;

[[noreturn]] void Die(const std::string& message) {
  throw std::runtime_error(message);
}

void Require(bool ok, const std::string& message) {
  if (!ok) {
    Die(message);
  }
}

void Check(cl_int code, const std::string& where) {
  if (code != kClSuccess) {
    Die(where + " failed with OpenCL error " + std::to_string(code));
  }
}

template <typename Function>
Function LoadSymbol(void* library, const char* name) {
  void* symbol = dlsym(library, name);
  if (symbol == nullptr) {
    Die(std::string("missing OpenCL symbol: ") + name);
  }
  return reinterpret_cast<Function>(symbol);
}

struct OpenClApi {
  void* library = nullptr;
  cl_int (*clGetPlatformIDs)(cl_uint, cl_platform_id*, cl_uint*) = nullptr;
  cl_int (*clGetPlatformInfo)(cl_platform_id, cl_platform_info, std::size_t,
                              void*, std::size_t*) = nullptr;
  cl_int (*clGetDeviceIDs)(cl_platform_id, cl_device_type, cl_uint,
                           cl_device_id*, cl_uint*) = nullptr;
  cl_int (*clGetDeviceInfo)(cl_device_id, cl_device_info, std::size_t, void*,
                            std::size_t*) = nullptr;
  cl_context (*clCreateContext)(const cl_context_properties*, cl_uint,
                                const cl_device_id*, void*, void*, cl_int*) = nullptr;
  cl_int (*clReleaseContext)(cl_context) = nullptr;
  cl_command_queue (*clCreateCommandQueue)(cl_context, cl_device_id,
                                            cl_command_queue_properties,
                                            cl_int*) = nullptr;
  cl_int (*clReleaseCommandQueue)(cl_command_queue) = nullptr;
  cl_mem (*clCreateBuffer)(cl_context, cl_mem_flags, std::size_t, void*,
                           cl_int*) = nullptr;
  cl_int (*clReleaseMemObject)(cl_mem) = nullptr;
  cl_program (*clCreateProgramWithSource)(cl_context, cl_uint, const char**,
                                          const std::size_t*, cl_int*) = nullptr;
  cl_int (*clBuildProgram)(cl_program, cl_uint, const cl_device_id*, const char*,
                           void*, void*) = nullptr;
  cl_int (*clGetProgramBuildInfo)(cl_program, cl_device_id,
                                  cl_program_build_info, std::size_t, void*,
                                  std::size_t*) = nullptr;
  cl_int (*clReleaseProgram)(cl_program) = nullptr;
  cl_kernel (*clCreateKernel)(cl_program, const char*, cl_int*) = nullptr;
  cl_int (*clSetKernelArg)(cl_kernel, cl_uint, std::size_t, const void*) = nullptr;
  cl_int (*clReleaseKernel)(cl_kernel) = nullptr;
  cl_int (*clEnqueueWriteBuffer)(cl_command_queue, cl_mem, cl_bool, std::size_t,
                                 std::size_t, const void*, cl_uint,
                                 const cl_event*, cl_event*) = nullptr;
  cl_int (*clEnqueueReadBuffer)(cl_command_queue, cl_mem, cl_bool, std::size_t,
                                std::size_t, void*, cl_uint, const cl_event*,
                                cl_event*) = nullptr;
  cl_int (*clEnqueueNDRangeKernel)(cl_command_queue, cl_kernel, cl_uint,
                                   const std::size_t*, const std::size_t*,
                                   const std::size_t*, cl_uint, const cl_event*,
                                   cl_event*) = nullptr;
  cl_int (*clFinish)(cl_command_queue) = nullptr;
  cl_int (*clGetEventProfilingInfo)(cl_event, cl_profiling_info, std::size_t,
                                    void*, std::size_t*) = nullptr;
  cl_int (*clReleaseEvent)(cl_event) = nullptr;

  OpenClApi() {
    library = dlopen("libOpenCL.so.1", RTLD_NOW | RTLD_LOCAL);
    Require(library != nullptr, "could not load libOpenCL.so.1");
    clGetPlatformIDs = LoadSymbol<decltype(clGetPlatformIDs)>(
        library, "clGetPlatformIDs");
    clGetPlatformInfo = LoadSymbol<decltype(clGetPlatformInfo)>(
        library, "clGetPlatformInfo");
    clGetDeviceIDs = LoadSymbol<decltype(clGetDeviceIDs)>(
        library, "clGetDeviceIDs");
    clGetDeviceInfo = LoadSymbol<decltype(clGetDeviceInfo)>(
        library, "clGetDeviceInfo");
    clCreateContext = LoadSymbol<decltype(clCreateContext)>(
        library, "clCreateContext");
    clReleaseContext = LoadSymbol<decltype(clReleaseContext)>(
        library, "clReleaseContext");
    clCreateCommandQueue = LoadSymbol<decltype(clCreateCommandQueue)>(
        library, "clCreateCommandQueue");
    clReleaseCommandQueue = LoadSymbol<decltype(clReleaseCommandQueue)>(
        library, "clReleaseCommandQueue");
    clCreateBuffer = LoadSymbol<decltype(clCreateBuffer)>(
        library, "clCreateBuffer");
    clReleaseMemObject = LoadSymbol<decltype(clReleaseMemObject)>(
        library, "clReleaseMemObject");
    clCreateProgramWithSource = LoadSymbol<decltype(clCreateProgramWithSource)>(
        library, "clCreateProgramWithSource");
    clBuildProgram = LoadSymbol<decltype(clBuildProgram)>(
        library, "clBuildProgram");
    clGetProgramBuildInfo = LoadSymbol<decltype(clGetProgramBuildInfo)>(
        library, "clGetProgramBuildInfo");
    clReleaseProgram = LoadSymbol<decltype(clReleaseProgram)>(
        library, "clReleaseProgram");
    clCreateKernel = LoadSymbol<decltype(clCreateKernel)>(
        library, "clCreateKernel");
    clSetKernelArg = LoadSymbol<decltype(clSetKernelArg)>(
        library, "clSetKernelArg");
    clReleaseKernel = LoadSymbol<decltype(clReleaseKernel)>(
        library, "clReleaseKernel");
    clEnqueueWriteBuffer = LoadSymbol<decltype(clEnqueueWriteBuffer)>(
        library, "clEnqueueWriteBuffer");
    clEnqueueReadBuffer = LoadSymbol<decltype(clEnqueueReadBuffer)>(
        library, "clEnqueueReadBuffer");
    clEnqueueNDRangeKernel = LoadSymbol<decltype(clEnqueueNDRangeKernel)>(
        library, "clEnqueueNDRangeKernel");
    clFinish = LoadSymbol<decltype(clFinish)>(library, "clFinish");
    clGetEventProfilingInfo = LoadSymbol<decltype(clGetEventProfilingInfo)>(
        library, "clGetEventProfilingInfo");
    clReleaseEvent = LoadSymbol<decltype(clReleaseEvent)>(
        library, "clReleaseEvent");
  }

  ~OpenClApi() {
    if (library != nullptr) {
      dlclose(library);
    }
  }
};

struct Args {
  std::string model;
  std::string input;
  std::string topk;
  std::string oracle;
  std::string router_weights;
  std::string down_oracle;
  std::string moe_oracle;
  std::string kernel_source;
  std::string kernel_mode = "m1_u8";
  std::string device_substring = "B390";
  std::uint64_t weight_offset = 0;
  std::uint64_t weight_bytes = 0;
  std::uint64_t down_weight_offset = 0;
  std::uint64_t down_weight_bytes = 0;
  std::size_t topk_stride = 0;
  int repeat = 11;
  double kernel_cap_us = 0.0;
};

Args ParseArgs(int argc, char** argv) {
  Args args;
  for (int index = 1; index < argc; ++index) {
    const std::string option = argv[index];
    const auto Value = [&](const char* name) -> std::string {
      Require(index + 1 < argc, std::string("missing value for ") + name);
      return argv[++index];
    };
    if (option == "--model") args.model = Value("--model");
    else if (option == "--input") args.input = Value("--input");
    else if (option == "--topk") args.topk = Value("--topk");
    else if (option == "--oracle") args.oracle = Value("--oracle");
    else if (option == "--router-weights") {
      args.router_weights = Value("--router-weights");
    } else if (option == "--down-oracle") {
      args.down_oracle = Value("--down-oracle");
    } else if (option == "--moe-oracle") {
      args.moe_oracle = Value("--moe-oracle");
    } else if (option == "--kernel-source") {
      args.kernel_source = Value("--kernel-source");
    }
    else if (option == "--kernel-mode") {
      args.kernel_mode = Value("--kernel-mode");
    }
    else if (option == "--weight-offset") {
      args.weight_offset = std::stoull(Value("--weight-offset"));
    } else if (option == "--weight-bytes") {
      args.weight_bytes = std::stoull(Value("--weight-bytes"));
    } else if (option == "--down-weight-offset") {
      args.down_weight_offset = std::stoull(Value("--down-weight-offset"));
    } else if (option == "--down-weight-bytes") {
      args.down_weight_bytes = std::stoull(Value("--down-weight-bytes"));
    } else if (option == "--topk-stride") {
      args.topk_stride = std::stoull(Value("--topk-stride"));
    } else if (option == "--repeat") {
      args.repeat = std::stoi(Value("--repeat"));
    } else if (option == "--kernel-cap-us") {
      args.kernel_cap_us = std::stod(Value("--kernel-cap-us"));
    } else if (option == "--device-substring") {
      args.device_substring = Value("--device-substring");
    } else {
      Die("unknown argument: " + option);
    }
  }
  Require(!args.model.empty() && !args.input.empty() && !args.topk.empty() &&
              !args.oracle.empty(),
          "model, input, topk, and oracle are required");
  Require(args.weight_bytes == kExpectedWeightBytes,
          "weight byte count does not match locked Q4_K gate/up tensor");
  Require(args.topk_stride >= kSelectedExperts * sizeof(std::int32_t),
          "top-k stride is too small");
  Require(args.repeat > 0 && args.kernel_cap_us > 0.0,
          "repeat and kernel cap must be positive");
  Require(args.kernel_mode == "m1_u8" || args.kernel_mode == "m8_u4" ||
              args.kernel_mode == "prepacked_routed",
          "kernel mode must be m1_u8, m8_u4, or prepacked_routed");
  if (args.kernel_mode == "prepacked_routed") {
    Require(!args.router_weights.empty() && !args.down_oracle.empty() &&
                !args.moe_oracle.empty() && !args.kernel_source.empty() &&
                args.down_weight_bytes == kExpectedDownWeightBytes,
            "prepacked routed mode requires kernel, down weights, and oracles");
  }
  return args;
}

template <typename Value>
std::vector<Value> ReadVector(const std::string& path,
                              std::size_t expected_count) {
  std::ifstream input(path, std::ios::binary);
  Require(static_cast<bool>(input), "could not open input: " + path);
  input.seekg(0, std::ios::end);
  const auto size = input.tellg();
  Require(size >= 0, "could not determine input size: " + path);
  Require(static_cast<std::uint64_t>(size) ==
              expected_count * sizeof(Value),
          "input size mismatch: " + path);
  input.seekg(0, std::ios::beg);
  std::vector<Value> values(expected_count);
  input.read(reinterpret_cast<char*>(values.data()),
             static_cast<std::streamsize>(values.size() * sizeof(Value)));
  Require(static_cast<bool>(input), "could not read input: " + path);
  return values;
}

std::vector<std::uint8_t> ReadBytes(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  Require(static_cast<bool>(input), "could not open input: " + path);
  input.seekg(0, std::ios::end);
  const auto size = input.tellg();
  Require(size >= 0, "could not determine input size: " + path);
  input.seekg(0, std::ios::beg);
  std::vector<std::uint8_t> values(static_cast<std::size_t>(size));
  input.read(reinterpret_cast<char*>(values.data()),
             static_cast<std::streamsize>(values.size()));
  Require(static_cast<bool>(input), "could not read input: " + path);
  return values;
}

std::vector<std::uint8_t> ReadModelSlice(
    const std::string& model_path, std::uint64_t offset, std::uint64_t bytes) {
  std::ifstream model(model_path, std::ios::binary);
  Require(static_cast<bool>(model), "could not open model");
  model.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
  Require(static_cast<bool>(model), "could not seek model weight");
  std::vector<std::uint8_t> weight(static_cast<std::size_t>(bytes));
  model.read(reinterpret_cast<char*>(weight.data()),
             static_cast<std::streamsize>(weight.size()));
  Require(model.gcount() == static_cast<std::streamsize>(weight.size()),
          "could not read complete model weight");
  return weight;
}

void StoreU16(std::vector<std::uint8_t>& data, std::size_t offset,
              std::uint16_t value) {
  data[offset] = static_cast<std::uint8_t>(value & 0xffU);
  data[offset + 1] = static_cast<std::uint8_t>((value >> 8) & 0xffU);
}

void StoreI16(std::vector<std::uint8_t>& data, std::size_t offset,
              std::int16_t value) {
  StoreU16(data, offset, static_cast<std::uint16_t>(value));
}

void StoreF32(std::vector<std::uint8_t>& data, std::size_t offset,
              float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  data[offset] = static_cast<std::uint8_t>(bits & 0xffU);
  data[offset + 1] = static_cast<std::uint8_t>((bits >> 8) & 0xffU);
  data[offset + 2] = static_cast<std::uint8_t>((bits >> 16) & 0xffU);
  data[offset + 3] = static_cast<std::uint8_t>((bits >> 24) & 0xffU);
}

int NearestInt(float value) {
  const float shifted = value + 12582912.0f;
  int bits = 0;
  std::memcpy(&bits, &shifted, sizeof(bits));
  return (bits & 0x007fffff) - 0x00400000;
}

std::vector<std::uint8_t> QuantizeQ8K(const std::vector<float>& input) {
  std::vector<std::uint8_t> q8(kTokenCount * kQ8TokenBytes, 0);
  for (std::size_t token = 0; token < kTokenCount; ++token) {
    for (std::size_t block = 0; block < kBlocksPerRow; ++block) {
      const float* source = input.data() + token * kHiddenSize + block * 256;
      float max_value = 0.0f;
      float absolute_max = 0.0f;
      for (std::size_t index = 0; index < 256; ++index) {
        const float absolute = std::fabs(source[index]);
        if (absolute > absolute_max) {
          absolute_max = absolute;
          max_value = source[index];
        }
      }
      const std::size_t base =
          token * kQ8TokenBytes + block * kQ8BlockBytes;
      if (absolute_max == 0.0f) {
        continue;
      }
      const float inverse_scale = -127.0f / max_value;
      StoreF32(q8, base, 1.0f / inverse_scale);
      int sums[16] = {};
      for (std::size_t index = 0; index < 256; ++index) {
        const int value = std::min(127, NearestInt(inverse_scale * source[index]));
        q8[base + 4 + index] = static_cast<std::uint8_t>(
            static_cast<std::int8_t>(value));
        sums[index / 16] += value;
      }
      for (std::size_t index = 0; index < 16; ++index) {
        StoreI16(q8, base + 260 + index * 2,
                 static_cast<std::int16_t>(sums[index]));
      }
    }
  }
  return q8;
}

std::uint16_t LoadU16(const std::uint8_t* data) {
  return static_cast<std::uint16_t>(data[0]) |
         static_cast<std::uint16_t>(data[1] << 8);
}

float HalfToFloat(std::uint16_t value) {
  const std::uint32_t sign = (value & 0x8000U) << 16;
  std::uint32_t exponent = (value >> 10) & 0x1fU;
  std::uint32_t mantissa = value & 0x3ffU;
  std::uint32_t bits = 0;
  if (exponent == 0) {
    if (mantissa == 0) {
      bits = sign;
    } else {
      std::uint32_t shift = 0;
      while ((mantissa & 0x400U) == 0) {
        mantissa <<= 1;
        ++shift;
      }
      mantissa &= 0x3ffU;
      bits = sign | ((127U - 14U - shift) << 23) | (mantissa << 13);
    }
  } else if (exponent == 0x1fU) {
    bits = sign | 0x7f800000U | (mantissa << 13);
  } else {
    bits = sign | ((exponent + 112U) << 23) | (mantissa << 13);
  }
  float result = 0.0f;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

std::uint8_t GetScale(std::size_t index, const std::uint8_t* scales) {
  if (index < 4) return scales[index] & 63U;
  return static_cast<std::uint8_t>(
      (scales[index + 4] & 15U) | ((scales[index - 4] >> 6) << 4));
}

std::uint8_t GetMinimum(std::size_t index, const std::uint8_t* scales) {
  if (index < 4) return scales[index + 4] & 63U;
  return static_cast<std::uint8_t>(
      (scales[index + 4] >> 4) | ((scales[index] >> 6) << 4));
}

std::uint8_t Q4Value(const std::uint8_t* block, std::size_t index) {
  const std::size_t segment = index / 64;
  const std::size_t offset = index - segment * 64;
  const std::uint8_t packed = block[16 + segment * 32 + (offset & 31)];
  return offset < 32 ? packed & 15U : packed >> 4;
}

struct PrepackedWeights {
  std::vector<std::uint8_t> codes;
  std::vector<float> scales;
  std::vector<float> mins;
  std::uint64_t code_count = 0;
  std::uint64_t mismatch_count = 0;
};

PrepackedWeights PrepackQ4K(
    const std::vector<std::uint8_t>& raw,
    const std::vector<cl_uint>& active_experts, std::size_t outputs,
    std::size_t blocks_per_row) {
  const std::size_t groups = blocks_per_row * 8;
  const std::size_t values_per_expert = outputs * blocks_per_row * 256;
  Require(raw.size() == kExpertCount * outputs * blocks_per_row * kQ4BlockBytes,
          "raw Q4_K tensor size mismatch");
  PrepackedWeights packed;
  packed.codes.resize(active_experts.size() * values_per_expert / 2, 0);
  packed.scales.resize(active_experts.size() * outputs * groups, 0.0f);
  packed.mins.resize(packed.scales.size(), 0.0f);
  for (std::size_t active = 0; active < active_experts.size(); ++active) {
    const std::size_t expert = active_experts[active];
    for (std::size_t output = 0; output < outputs; ++output) {
      for (std::size_t group = 0; group < groups; ++group) {
        const std::size_t block_index = group / 8;
        const std::size_t group_in_block = group & 7;
        const std::size_t row = expert * outputs + output;
        const std::uint8_t* block = raw.data() +
            (row * blocks_per_row + block_index) * kQ4BlockBytes;
        const float d = HalfToFloat(LoadU16(block));
        const float dmin = HalfToFloat(LoadU16(block + 2));
        const std::size_t coefficient =
            (active * outputs + output) * groups + group;
        packed.scales[coefficient] =
            d * static_cast<float>(GetScale(group_in_block, block + 4));
        packed.mins[coefficient] =
            dmin * static_cast<float>(GetMinimum(group_in_block, block + 4));
        const std::size_t code_base = coefficient * 16;
        for (std::size_t index = 0; index < 32; index += 2) {
          const std::uint8_t low =
              Q4Value(block, group_in_block * 32 + index);
          const std::uint8_t high =
              Q4Value(block, group_in_block * 32 + index + 1);
          const std::uint8_t value =
              static_cast<std::uint8_t>(low | (high << 4));
          packed.codes[code_base + index / 2] = value;
          packed.code_count += 2;
          packed.mismatch_count += static_cast<std::uint64_t>(
              (value & 15U) != low);
          packed.mismatch_count += static_cast<std::uint64_t>(
              (value >> 4) != high);
        }
      }
    }
  }
  return packed;
}

std::string ReadText(const std::string& path) {
  std::ifstream input(path);
  Require(static_cast<bool>(input), "could not open text input: " + path);
  std::ostringstream stream;
  stream << input.rdbuf();
  Require(static_cast<bool>(input) || input.eof(),
          "could not read text input: " + path);
  return stream.str();
}

struct BucketPlan {
  std::vector<cl_uint> bucket_token;
  std::vector<cl_uint> bucket_rank;
  std::vector<cl_uint> active_expert_ids;
  std::vector<cl_uint> task_expert;
  std::vector<cl_uint> task_bucket_base;
  std::vector<cl_uint> task_token_count;
  std::array<std::vector<cl_uint>, 4> fused_task_weight;
  std::array<std::vector<cl_uint>, 4> fused_task_base;
  std::array<std::vector<cl_uint>, 4> fused_task_count;
  std::size_t active_experts = 0;
  std::size_t max_group_m = 0;
};

std::size_t FusedBucketIndex(std::size_t count) {
  if (count <= 8) return 0;
  if (count <= 16) return 1;
  if (count <= 32) return 2;
  Require(count <= 64, "fused task exceeds M=64");
  return 3;
}

BucketPlan BuildPlan(const std::vector<std::uint8_t>& topk,
                     std::size_t stride,
                     std::size_t task_tokens) {
  Require(task_tokens > 0 && task_tokens <= 128,
          "task token count is invalid");
  std::vector<std::vector<std::pair<cl_uint, cl_uint>>> groups(kExpertCount);
  for (std::size_t token = 0; token < kTokenCount; ++token) {
    bool seen[kExpertCount] = {};
    for (std::size_t rank = 0; rank < kSelectedExperts; ++rank) {
      const std::size_t offset = token * stride + rank * sizeof(std::int32_t);
      Require(offset + sizeof(std::int32_t) <= topk.size(),
              "top-k payload is truncated");
      std::int32_t expert = -1;
      std::memcpy(&expert, topk.data() + offset, sizeof(expert));
      Require(expert >= 0 && expert < static_cast<std::int32_t>(kExpertCount),
              "top-k expert is out of range");
      Require(!seen[expert], "top-k expert is duplicated within a token");
      seen[expert] = true;
      groups[static_cast<std::size_t>(expert)].push_back(
          {static_cast<cl_uint>(token), static_cast<cl_uint>(rank)});
    }
  }

  BucketPlan plan;
  plan.bucket_token.reserve(kAssignments);
  plan.bucket_rank.reserve(kAssignments);
  for (std::size_t expert = 0; expert < kExpertCount; ++expert) {
    const auto& group = groups[expert];
    if (group.empty()) {
      continue;
    }
    const cl_uint active_index = static_cast<cl_uint>(plan.active_expert_ids.size());
    plan.active_expert_ids.push_back(static_cast<cl_uint>(expert));
    ++plan.active_experts;
    plan.max_group_m = std::max(plan.max_group_m, group.size());
    const std::size_t group_base = plan.bucket_token.size();
    for (const auto& assignment : group) {
      plan.bucket_token.push_back(assignment.first);
      plan.bucket_rank.push_back(assignment.second);
    }
    for (std::size_t offset = 0; offset < group.size(); offset += task_tokens) {
      plan.task_expert.push_back(static_cast<cl_uint>(expert));
      plan.task_bucket_base.push_back(
          static_cast<cl_uint>(group_base + offset));
      plan.task_token_count.push_back(static_cast<cl_uint>(
          std::min(task_tokens, group.size() - offset)));
    }
    for (std::size_t offset = 0; offset < group.size(); offset += 64) {
      const std::size_t count = std::min<std::size_t>(64, group.size() - offset);
      const std::size_t bucket = FusedBucketIndex(count);
      plan.fused_task_weight[bucket].push_back(active_index);
      plan.fused_task_base[bucket].push_back(
          static_cast<cl_uint>(group_base + offset));
      plan.fused_task_count[bucket].push_back(static_cast<cl_uint>(count));
    }
  }
  Require(plan.bucket_token.size() == kAssignments,
          "bucket assignment count mismatch");
  Require(plan.task_expert.size() == plan.task_bucket_base.size() &&
              plan.task_expert.size() == plan.task_token_count.size(),
          "task descriptor size mismatch");
  Require(plan.active_expert_ids.size() == plan.active_experts,
          "active expert map size mismatch");
  for (std::size_t bucket = 0; bucket < 4; ++bucket) {
    Require(plan.fused_task_weight[bucket].size() ==
                    plan.fused_task_base[bucket].size() &&
                plan.fused_task_weight[bucket].size() ==
                    plan.fused_task_count[bucket].size(),
            "fused task descriptor size mismatch");
  }
  return plan;
}

std::string PlatformInfo(OpenClApi& api, cl_platform_id platform,
                         cl_platform_info field) {
  std::size_t size = 0;
  Check(api.clGetPlatformInfo(platform, field, 0, nullptr, &size),
        "clGetPlatformInfo size");
  std::string value(size, '\0');
  Check(api.clGetPlatformInfo(platform, field, size, value.data(), nullptr),
        "clGetPlatformInfo value");
  while (!value.empty() && value.back() == '\0') value.pop_back();
  return value;
}

std::string DeviceInfo(OpenClApi& api, cl_device_id device,
                       cl_device_info field) {
  std::size_t size = 0;
  Check(api.clGetDeviceInfo(device, field, 0, nullptr, &size),
        "clGetDeviceInfo size");
  std::string value(size, '\0');
  Check(api.clGetDeviceInfo(device, field, size, value.data(), nullptr),
        "clGetDeviceInfo value");
  while (!value.empty() && value.back() == '\0') value.pop_back();
  return value;
}

struct DeviceSelection {
  cl_platform_id platform = nullptr;
  cl_device_id device = nullptr;
  std::string platform_name;
  std::string device_name;
};

DeviceSelection SelectDevice(OpenClApi& api, const std::string& substring) {
  cl_uint platform_count = 0;
  Check(api.clGetPlatformIDs(0, nullptr, &platform_count),
        "clGetPlatformIDs count");
  std::vector<cl_platform_id> platforms(platform_count);
  Check(api.clGetPlatformIDs(platform_count, platforms.data(), nullptr),
        "clGetPlatformIDs");
  for (cl_platform_id platform : platforms) {
    cl_uint device_count = 0;
    if (api.clGetDeviceIDs(platform, kClDeviceTypeGpu, 0, nullptr,
                           &device_count) != kClSuccess ||
        device_count == 0) {
      continue;
    }
    std::vector<cl_device_id> devices(device_count);
    Check(api.clGetDeviceIDs(platform, kClDeviceTypeGpu, device_count,
                             devices.data(), nullptr),
          "clGetDeviceIDs");
    for (cl_device_id device : devices) {
      const std::string name = DeviceInfo(api, device, kClDeviceName);
      if (substring.empty() || name.find(substring) != std::string::npos) {
        return {platform, device,
                PlatformInfo(api, platform, kClPlatformName), name};
      }
    }
  }
  Die("no matching OpenCL GPU");
}

std::string BuildLog(OpenClApi& api, cl_program program, cl_device_id device) {
  std::size_t size = 0;
  api.clGetProgramBuildInfo(program, device, kClProgramBuildLog, 0, nullptr,
                            &size);
  std::string value(size, '\0');
  if (size > 0) {
    api.clGetProgramBuildInfo(program, device, kClProgramBuildLog, size,
                              value.data(), nullptr);
  }
  while (!value.empty() && value.back() == '\0') value.pop_back();
  return value;
}

double EventMicroseconds(OpenClApi& api, cl_event event) {
  cl_ulong start = 0;
  cl_ulong end = 0;
  Check(api.clGetEventProfilingInfo(event, kClProfilingCommandStart,
                                    sizeof(start), &start, nullptr),
        "clGetEventProfilingInfo start");
  Check(api.clGetEventProfilingInfo(event, kClProfilingCommandEnd,
                                    sizeof(end), &end, nullptr),
        "clGetEventProfilingInfo end");
  return static_cast<double>(end - start) / 1000.0;
}

struct CompareStats {
  bool finite = true;
  std::size_t count = 0;
  std::size_t mismatch_count = 0;
  double max_abs = 0.0;
  double mean_abs = 0.0;
  double rmse = 0.0;
  double cosine = 0.0;
};

CompareStats Compare(const std::vector<float>& output,
                     const std::vector<float>& oracle,
                     const BucketPlan& plan) {
  Require(output.size() == kAssignments * kIntermediateSize,
          "output size mismatch");
  Require(oracle.size() == kAssignments * kIntermediateSize,
          "oracle size mismatch");
  long double absolute_sum = 0.0L;
  long double square_sum = 0.0L;
  long double dot = 0.0L;
  long double norm_output = 0.0L;
  long double norm_oracle = 0.0L;
  CompareStats stats;
  stats.count = output.size();
  for (std::size_t bucket = 0; bucket < kAssignments; ++bucket) {
    const std::size_t token = plan.bucket_token[bucket];
    const std::size_t rank = plan.bucket_rank[bucket];
    const std::size_t oracle_base =
        (token * kSelectedExperts + rank) * kIntermediateSize;
    const std::size_t output_base = bucket * kIntermediateSize;
    for (std::size_t inner = 0; inner < kIntermediateSize; ++inner) {
      const double actual = output[output_base + inner];
      const double expected = oracle[oracle_base + inner];
      if (!std::isfinite(actual) || !std::isfinite(expected)) {
        stats.finite = false;
      }
      const double difference = actual - expected;
      const double absolute = std::fabs(difference);
      stats.max_abs = std::max(stats.max_abs, absolute);
      absolute_sum += absolute;
      square_sum += difference * difference;
      dot += actual * expected;
      norm_output += actual * actual;
      norm_oracle += expected * expected;
      if (absolute > kMismatchThreshold) {
        ++stats.mismatch_count;
      }
    }
  }
  stats.mean_abs = static_cast<double>(absolute_sum / stats.count);
  stats.rmse = std::sqrt(static_cast<double>(square_sum / stats.count));
  const long double norm = std::sqrt(norm_output * norm_oracle);
  stats.cosine = norm == 0.0L ? 1.0 : static_cast<double>(dot / norm);
  return stats;
}

CompareStats CompareFlat(const std::vector<float>& output,
                         const std::vector<float>& oracle) {
  Require(output.size() == oracle.size(), "flat comparison size mismatch");
  long double absolute_sum = 0.0L;
  long double square_sum = 0.0L;
  long double dot = 0.0L;
  long double norm_output = 0.0L;
  long double norm_oracle = 0.0L;
  CompareStats stats;
  stats.count = output.size();
  for (std::size_t index = 0; index < output.size(); ++index) {
    const double actual = output[index];
    const double expected = oracle[index];
    if (!std::isfinite(actual) || !std::isfinite(expected)) stats.finite = false;
    const double difference = actual - expected;
    const double absolute = std::fabs(difference);
    stats.max_abs = std::max(stats.max_abs, absolute);
    absolute_sum += absolute;
    square_sum += difference * difference;
    dot += actual * expected;
    norm_output += actual * actual;
    norm_oracle += expected * expected;
    if (absolute > kMismatchThreshold) ++stats.mismatch_count;
  }
  stats.mean_abs = static_cast<double>(absolute_sum / stats.count);
  stats.rmse = std::sqrt(static_cast<double>(square_sum / stats.count));
  const long double norm = std::sqrt(norm_output * norm_oracle);
  stats.cosine = norm == 0.0L ? 1.0 : static_cast<double>(dot / norm);
  return stats;
}

bool ComparePass(const CompareStats& stats) {
  return stats.finite && stats.mismatch_count == 0 &&
         stats.max_abs <= kMismatchThreshold && stats.rmse <= 5e-4 &&
         stats.cosine >= 0.999;
}

void PrintCompare(const char* name, const CompareStats& stats) {
  std::cout << "\"" << name << "\":{\"compared_value_count\":"
            << stats.count << ",\"cosine\":" << stats.cosine
            << ",\"finite\":" << (stats.finite ? "true" : "false")
            << ",\"max_abs_diff\":" << stats.max_abs
            << ",\"mean_abs_diff\":" << stats.mean_abs
            << ",\"mismatch_count\":" << stats.mismatch_count
            << ",\"rmse\":" << stats.rmse << "},";
}

std::string JsonEscape(const std::string& value) {
  std::string result;
  for (const char character : value) {
    if (character == '\\' || character == '"') result.push_back('\\');
    if (character == '\n') result += "\\n";
    else if (character != '\r') result.push_back(character);
  }
  return result;
}

template <typename Value>
cl_mem CreateAndWrite(OpenClApi& api, cl_context context,
                      cl_command_queue queue, const std::vector<Value>& values,
                      cl_mem_flags flags, const std::string& label) {
  cl_int error = kClSuccess;
  const std::size_t bytes = values.size() * sizeof(Value);
  cl_mem buffer = api.clCreateBuffer(context, flags, bytes, nullptr, &error);
  Check(error, "clCreateBuffer " + label);
  Check(api.clEnqueueWriteBuffer(queue, buffer, kClTrue, 0, bytes,
                                 values.data(), 0, nullptr, nullptr),
        "clEnqueueWriteBuffer " + label);
  return buffer;
}

cl_mem CreateBuffer(OpenClApi& api, cl_context context, std::size_t bytes,
                    const std::string& label) {
  cl_int error = kClSuccess;
  cl_mem buffer =
      api.clCreateBuffer(context, kClMemReadWrite, bytes, nullptr, &error);
  Check(error, "clCreateBuffer " + label);
  return buffer;
}

template <typename Value>
std::vector<Value> ReadBuffer(OpenClApi& api, cl_command_queue queue,
                              cl_mem buffer, std::size_t count,
                              const std::string& label) {
  std::vector<Value> values(count);
  Check(api.clEnqueueReadBuffer(queue, buffer, kClTrue, 0,
                                values.size() * sizeof(Value), values.data(),
                                0, nullptr, nullptr),
        "clEnqueueReadBuffer " + label);
  return values;
}

int RunPrepackedRouted(const Args& args) {
  const auto input = ReadVector<float>(args.input, kTokenCount * kHiddenSize);
  const auto topk = ReadBytes(args.topk);
  const auto oracle =
      ReadVector<float>(args.oracle, kAssignments * kIntermediateSize);
  const auto router_weights =
      ReadVector<float>(args.router_weights, kAssignments);
  const auto down_oracle =
      ReadVector<float>(args.down_oracle, kAssignments * kDownOutputSize);
  const auto moe_oracle =
      ReadVector<float>(args.moe_oracle, kTokenCount * kHiddenSize);
  const BucketPlan plan = BuildPlan(topk, args.topk_stride, 64);
  auto gate_raw = ReadModelSlice(
      args.model, args.weight_offset, args.weight_bytes);
  PrepackedWeights gate_weights = PrepackQ4K(
      gate_raw, plan.active_expert_ids, kRowsPerExpert, kBlocksPerRow);
  std::vector<std::uint8_t>().swap(gate_raw);
  auto down_raw = ReadModelSlice(
      args.model, args.down_weight_offset, args.down_weight_bytes);
  PrepackedWeights down_weights = PrepackQ4K(
      down_raw, plan.active_expert_ids, kDownRowsPerExpert,
      kDownBlocksPerRow);
  std::vector<std::uint8_t>().swap(down_raw);

  std::vector<float> bucket_weights(kAssignments);
  std::vector<std::int32_t> inverse_map(kAssignments, -1);
  for (std::size_t bucket = 0; bucket < kAssignments; ++bucket) {
    const std::size_t token = plan.bucket_token[bucket];
    const std::size_t rank = plan.bucket_rank[bucket];
    bucket_weights[bucket] = router_weights[token * kSelectedExperts + rank];
    inverse_map[token * kSelectedExperts + rank] =
        static_cast<std::int32_t>(bucket);
  }
  Require(std::none_of(inverse_map.begin(), inverse_map.end(),
                       [](std::int32_t value) { return value < 0; }),
          "inverse map is incomplete");

  OpenClApi api;
  const DeviceSelection selection = SelectDevice(api, args.device_substring);
  cl_int error = kClSuccess;
  cl_context context = api.clCreateContext(
      nullptr, 1, &selection.device, nullptr, nullptr, &error);
  Check(error, "clCreateContext");
  cl_command_queue queue = api.clCreateCommandQueue(
      context, selection.device, kClQueueProfilingEnable, &error);
  Check(error, "clCreateCommandQueue");
  const std::string source = ReadText(args.kernel_source);
  const char* source_data = source.c_str();
  const std::size_t source_size = source.size();
  cl_program program = api.clCreateProgramWithSource(
      context, 1, &source_data, &source_size, &error);
  Check(error, "clCreateProgramWithSource fused");
  const cl_int build_code = api.clBuildProgram(
      program, 1, &selection.device, "-cl-std=CL2.0", nullptr, nullptr);
  const std::string build_log = BuildLog(api, program, selection.device);
  if (build_code != kClSuccess) Die("OpenCL fused build failed: " + build_log);

  const auto make_kernel = [&](const char* name) {
    cl_kernel kernel = api.clCreateKernel(program, name, &error);
    Check(error, std::string("clCreateKernel ") + name);
    return kernel;
  };
  cl_kernel input_quant = make_kernel("quantize_input_q8k");
  cl_kernel swiglu_quant = make_kernel("quantize_swiglu_q8k");
  cl_kernel scatter = make_kernel("scatter_routed_output");
  std::array<cl_kernel, 4> gate_kernels{};
  std::array<cl_kernel, 4> down_kernels{};
  for (std::size_t bucket = 0; bucket < 4; ++bucket) {
    gate_kernels[bucket] = make_kernel("prepacked_q4k_gateup_swiglu");
    down_kernels[bucket] = make_kernel("prepacked_q4k_down_weighted");
  }

  std::vector<cl_mem> buffers;
  const auto keep = [&](cl_mem buffer) {
    buffers.push_back(buffer);
    return buffer;
  };
  cl_mem input_buffer = keep(CreateAndWrite(
      api, context, queue, input, kClMemReadOnly, "fused input"));
  cl_mem gate_code_buffer = keep(CreateAndWrite(
      api, context, queue, gate_weights.codes, kClMemReadOnly, "gate codes"));
  cl_mem gate_scale_buffer = keep(CreateAndWrite(
      api, context, queue, gate_weights.scales, kClMemReadOnly, "gate scales"));
  cl_mem gate_min_buffer = keep(CreateAndWrite(
      api, context, queue, gate_weights.mins, kClMemReadOnly, "gate mins"));
  cl_mem down_code_buffer = keep(CreateAndWrite(
      api, context, queue, down_weights.codes, kClMemReadOnly, "down codes"));
  cl_mem down_scale_buffer = keep(CreateAndWrite(
      api, context, queue, down_weights.scales, kClMemReadOnly, "down scales"));
  cl_mem down_min_buffer = keep(CreateAndWrite(
      api, context, queue, down_weights.mins, kClMemReadOnly, "down mins"));
  cl_mem token_buffer = keep(CreateAndWrite(
      api, context, queue, plan.bucket_token, kClMemReadOnly, "bucket token"));
  cl_mem weight_buffer = keep(CreateAndWrite(
      api, context, queue, bucket_weights, kClMemReadOnly, "bucket weights"));
  cl_mem inverse_buffer = keep(CreateAndWrite(
      api, context, queue, inverse_map, kClMemReadOnly, "inverse map"));

  cl_mem input_q8 = keep(CreateBuffer(
      api, context, kTokenCount * kHiddenSize, "input q8"));
  cl_mem input_scales = keep(CreateBuffer(
      api, context, kTokenCount * kBlocksPerRow * sizeof(float),
      "input scales"));
  cl_mem input_sums = keep(CreateBuffer(
      api, context, kTokenCount * kBlocksPerRow * 8 * sizeof(float),
      "input sums"));
  cl_mem swiglu_output = keep(CreateBuffer(
      api, context, kAssignments * kIntermediateSize * sizeof(float),
      "swiglu output"));
  cl_mem down_q8 = keep(CreateBuffer(
      api, context, kAssignments * kDownInputSize, "down q8"));
  cl_mem down_scales = keep(CreateBuffer(
      api, context, kAssignments * kDownBlocksPerRow * sizeof(float),
      "down scales"));
  cl_mem down_sums = keep(CreateBuffer(
      api, context, kAssignments * kDownBlocksPerRow * 8 * sizeof(float),
      "down sums"));
  cl_mem contributions = keep(CreateBuffer(
      api, context, kAssignments * kDownOutputSize * sizeof(float),
      "contributions"));
  cl_mem routed_output = keep(CreateBuffer(
      api, context, kTokenCount * kHiddenSize * sizeof(float),
      "routed output"));

  std::array<cl_mem, 4> task_weight_buffers{};
  std::array<cl_mem, 4> task_base_buffers{};
  std::array<cl_mem, 4> task_count_buffers{};
  constexpr std::array<cl_uint, 4> bucket_m = {8, 16, 32, 64};
  for (std::size_t bucket = 0; bucket < 4; ++bucket) {
    Require(!plan.fused_task_weight[bucket].empty(),
            "fused task bucket is empty");
    task_weight_buffers[bucket] = keep(CreateAndWrite(
        api, context, queue, plan.fused_task_weight[bucket], kClMemReadOnly,
        "fused task weight"));
    task_base_buffers[bucket] = keep(CreateAndWrite(
        api, context, queue, plan.fused_task_base[bucket], kClMemReadOnly,
        "fused task base"));
    task_count_buffers[bucket] = keep(CreateAndWrite(
        api, context, queue, plan.fused_task_count[bucket], kClMemReadOnly,
        "fused task count"));
  }

  const auto set_arg = [&](cl_kernel kernel, cl_uint index, const auto& value,
                           const char* label) {
    Check(api.clSetKernelArg(kernel, index, sizeof(value), &value), label);
  };
  set_arg(input_quant, 0, input_buffer, "input quant input");
  set_arg(input_quant, 1, input_q8, "input quant q8");
  set_arg(input_quant, 2, input_scales, "input quant scales");
  set_arg(input_quant, 3, input_sums, "input quant sums");
  set_arg(swiglu_quant, 0, swiglu_output, "swiglu quant input");
  set_arg(swiglu_quant, 1, down_q8, "swiglu quant q8");
  set_arg(swiglu_quant, 2, down_scales, "swiglu quant scales");
  set_arg(swiglu_quant, 3, down_sums, "swiglu quant sums");
  set_arg(scatter, 0, contributions, "scatter contributions");
  set_arg(scatter, 1, inverse_buffer, "scatter inverse");
  set_arg(scatter, 2, routed_output, "scatter output");

  for (std::size_t bucket = 0; bucket < 4; ++bucket) {
    cl_kernel gate = gate_kernels[bucket];
    const cl_uint tasks =
        static_cast<cl_uint>(plan.fused_task_weight[bucket].size());
    set_arg(gate, 0, gate_code_buffer, "gate codes");
    set_arg(gate, 1, gate_scale_buffer, "gate scales");
    set_arg(gate, 2, gate_min_buffer, "gate mins");
    set_arg(gate, 3, input_q8, "gate q8");
    set_arg(gate, 4, input_scales, "gate source scales");
    set_arg(gate, 5, input_sums, "gate source sums");
    set_arg(gate, 6, task_weight_buffers[bucket], "gate task weight");
    set_arg(gate, 7, task_base_buffers[bucket], "gate task base");
    set_arg(gate, 8, task_count_buffers[bucket], "gate task count");
    set_arg(gate, 9, token_buffer, "gate bucket token");
    set_arg(gate, 10, swiglu_output, "gate output");
    set_arg(gate, 11, tasks, "gate task total");
    set_arg(gate, 12, bucket_m[bucket], "gate bucket m");
    Check(api.clSetKernelArg(
              gate, 13, bucket_m[bucket] * 256, nullptr),
          "gate local q8");

    cl_kernel down = down_kernels[bucket];
    set_arg(down, 0, down_code_buffer, "down codes");
    set_arg(down, 1, down_scale_buffer, "down scales");
    set_arg(down, 2, down_min_buffer, "down mins");
    set_arg(down, 3, down_q8, "down q8");
    set_arg(down, 4, down_scales, "down source scales");
    set_arg(down, 5, down_sums, "down source sums");
    set_arg(down, 6, task_weight_buffers[bucket], "down task weight");
    set_arg(down, 7, task_base_buffers[bucket], "down task base");
    set_arg(down, 8, task_count_buffers[bucket], "down task count");
    set_arg(down, 9, weight_buffer, "down bucket weights");
    set_arg(down, 10, contributions, "down contributions");
    set_arg(down, 11, tasks, "down task total");
    set_arg(down, 12, bucket_m[bucket], "down bucket m");
    Check(api.clSetKernelArg(
              down, 13, bucket_m[bucket] * 256, nullptr),
          "down local q8");
  }

  const auto execute = [&]() {
    constexpr std::size_t quant_local = 256;
    const std::size_t input_global =
        kTokenCount * kBlocksPerRow * quant_local;
    Check(api.clEnqueueNDRangeKernel(queue, input_quant, 1, nullptr,
                                     &input_global, &quant_local, 0, nullptr,
                                     nullptr), "enqueue input quant");
    for (std::size_t bucket = 0; bucket < 4; ++bucket) {
      const std::size_t local = bucket_m[bucket] * 8;
      const std::size_t global =
          plan.fused_task_weight[bucket].size() * 8 * local;
      Check(api.clEnqueueNDRangeKernel(queue, gate_kernels[bucket], 1,
                                       nullptr, &global, &local, 0, nullptr,
                                       nullptr), "enqueue fused gate");
    }
    const std::size_t swiglu_global =
        kAssignments * kDownBlocksPerRow * quant_local;
    Check(api.clEnqueueNDRangeKernel(queue, swiglu_quant, 1, nullptr,
                                     &swiglu_global, &quant_local, 0, nullptr,
                                     nullptr), "enqueue swiglu quant");
    for (std::size_t bucket = 0; bucket < 4; ++bucket) {
      const std::size_t local = bucket_m[bucket] * 8;
      const std::size_t global =
          plan.fused_task_weight[bucket].size() * 32 * local;
      Check(api.clEnqueueNDRangeKernel(queue, down_kernels[bucket], 1,
                                       nullptr, &global, &local, 0, nullptr,
                                       nullptr), "enqueue fused down");
    }
    constexpr std::size_t scatter_local = 256;
    const std::size_t scatter_global = kTokenCount * kHiddenSize;
    Check(api.clEnqueueNDRangeKernel(queue, scatter, 1, nullptr,
                                     &scatter_global, &scatter_local, 0,
                                     nullptr, nullptr), "enqueue scatter");
    Check(api.clFinish(queue), "clFinish fused routed");
  };

  for (int warmup = 0; warmup < 2; ++warmup) execute();
  std::vector<double> samples;
  samples.reserve(static_cast<std::size_t>(args.repeat));
  for (int repeat = 0; repeat < args.repeat; ++repeat) {
    const auto begin = std::chrono::steady_clock::now();
    execute();
    const auto end = std::chrono::steady_clock::now();
    samples.push_back(
        std::chrono::duration<double, std::micro>(end - begin).count());
  }

  std::array<double, 5> stage_profile_us{};
  std::vector<std::pair<std::size_t, cl_event>> profile_events;
  const auto profile_enqueue = [&](cl_kernel kernel, std::size_t global,
                                   std::size_t local, std::size_t stage,
                                   const char* label) {
    cl_event event = nullptr;
    Check(api.clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global,
                                     &local, 0, nullptr, &event), label);
    profile_events.push_back({stage, event});
  };
  constexpr std::size_t profile_quant_local = 256;
  profile_enqueue(input_quant, kTokenCount * kBlocksPerRow * 256,
                  profile_quant_local, 0, "profile input quant");
  for (std::size_t bucket = 0; bucket < 4; ++bucket) {
    const std::size_t local = bucket_m[bucket] * 8;
    profile_enqueue(gate_kernels[bucket],
                    plan.fused_task_weight[bucket].size() * 8 * local,
                    local, 1, "profile gate");
  }
  profile_enqueue(swiglu_quant, kAssignments * kDownBlocksPerRow * 256,
                  profile_quant_local, 2, "profile swiglu quant");
  for (std::size_t bucket = 0; bucket < 4; ++bucket) {
    const std::size_t local = bucket_m[bucket] * 8;
    profile_enqueue(down_kernels[bucket],
                    plan.fused_task_weight[bucket].size() * 32 * local,
                    local, 3, "profile down");
  }
  constexpr std::size_t profile_scatter_local = 256;
  profile_enqueue(scatter, kTokenCount * kHiddenSize,
                  profile_scatter_local, 4, "profile scatter");
  Check(api.clFinish(queue), "clFinish fused profile");
  for (const auto& [stage, event] : profile_events) {
    stage_profile_us[stage] += EventMicroseconds(api, event);
    api.clReleaseEvent(event);
  }

  const auto swiglu = ReadBuffer<float>(
      api, queue, swiglu_output, kAssignments * kIntermediateSize, "swiglu");
  const auto weighted_down = ReadBuffer<float>(
      api, queue, contributions, kAssignments * kDownOutputSize,
      "weighted down");
  const auto routed = ReadBuffer<float>(
      api, queue, routed_output, kTokenCount * kHiddenSize, "routed output");
  const CompareStats gate_compare = Compare(swiglu, oracle, plan);
  std::vector<float> weighted_oracle(kAssignments * kDownOutputSize);
  for (std::size_t bucket = 0; bucket < kAssignments; ++bucket) {
    const std::size_t token = plan.bucket_token[bucket];
    const std::size_t rank = plan.bucket_rank[bucket];
    const std::size_t source_index = token * kSelectedExperts + rank;
    for (std::size_t hidden = 0; hidden < kDownOutputSize; ++hidden) {
      weighted_oracle[bucket * kDownOutputSize + hidden] =
          down_oracle[source_index * kDownOutputSize + hidden] *
          router_weights[source_index];
    }
  }
  const CompareStats down_compare =
      CompareFlat(weighted_down, weighted_oracle);
  const CompareStats moe_compare = CompareFlat(routed, moe_oracle);
  std::vector<double> sorted = samples;
  std::sort(sorted.begin(), sorted.end());
  const double minimum_us = sorted.front();
  const double median_us = sorted[sorted.size() / 2];
  const double mean_us =
      std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
  const std::uint64_t code_count =
      gate_weights.code_count + down_weights.code_count;
  const std::uint64_t mismatch_count =
      gate_weights.mismatch_count + down_weights.mismatch_count;
  const std::uint64_t expected_codes = plan.active_experts *
      (kRowsPerExpert * kHiddenSize +
       kDownRowsPerExpert * kDownInputSize);
  const bool repack_pass =
      code_count == expected_codes && mismatch_count == 0;
  const bool correctness_pass = repack_pass && ComparePass(gate_compare) &&
      ComparePass(down_compare) && ComparePass(moe_compare);
  const bool performance_pass = minimum_us <= args.kernel_cap_us;
  const std::uint64_t resident_bytes =
      gate_weights.codes.size() + down_weights.codes.size() +
      (gate_weights.scales.size() + gate_weights.mins.size() +
       down_weights.scales.size() + down_weights.mins.size()) * sizeof(float);

  std::cout << std::setprecision(12) << "{";
  std::cout << "\"active_experts\":" << plan.active_experts << ",";
  std::cout << "\"assignment_count\":" << kAssignments << ",";
  std::cout << "\"build_log\":\"" << JsonEscape(build_log) << "\",";
  PrintCompare("compare", gate_compare);
  PrintCompare("weighted_down_compare", down_compare);
  PrintCompare("moe_compare", moe_compare);
  std::cout << "\"correctness_pass\":"
            << (correctness_pass ? "true" : "false") << ",";
  std::cout << "\"device_name\":\"" << JsonEscape(selection.device_name)
            << "\",";
  std::cout << "\"kernel_cap_us\":" << args.kernel_cap_us << ",";
  std::cout << "\"kernel_mean_us\":" << mean_us << ",";
  std::cout << "\"kernel_median_us\":" << median_us << ",";
  std::cout << "\"kernel_min_us\":" << minimum_us << ",";
  std::cout << "\"kernel_mode\":\"prepacked_routed\",";
  std::cout << "\"performance_pass\":"
            << (performance_pass ? "true" : "false") << ",";
  std::cout << "\"platform_name\":\""
            << JsonEscape(selection.platform_name) << "\",";
  std::cout << "\"repack_mismatch_count\":" << mismatch_count << ",";
  std::cout << "\"repack_pass\":" << (repack_pass ? "true" : "false")
            << ",";
  std::cout << "\"repacked_q4_code_count\":" << code_count << ",";
  std::cout << "\"resident_prepacked_bytes\":" << resident_bytes << ",";
  std::cout << "\"samples_us\":[";
  for (std::size_t index = 0; index < samples.size(); ++index) {
    if (index != 0) std::cout << ",";
    std::cout << samples[index];
  }
  std::cout << "],\"stage_profile_us\":{"
            << "\"input_quant\":" << stage_profile_us[0] << ","
            << "\"gate_up\":" << stage_profile_us[1] << ","
            << "\"swiglu_quant\":" << stage_profile_us[2] << ","
            << "\"down\":" << stage_profile_us[3] << ","
            << "\"scatter\":" << stage_profile_us[4] << "},";
  std::cout << "\"task_buckets\":[";
  for (std::size_t bucket = 0; bucket < 4; ++bucket) {
    if (bucket != 0) std::cout << ",";
    std::cout << "{\"m\":" << bucket_m[bucket]
              << ",\"tasks\":" << plan.fused_task_weight[bucket].size()
              << "}";
  }
  std::cout << "]}\n";

  for (cl_mem buffer : buffers) api.clReleaseMemObject(buffer);
  for (cl_kernel kernel : down_kernels) api.clReleaseKernel(kernel);
  for (cl_kernel kernel : gate_kernels) api.clReleaseKernel(kernel);
  api.clReleaseKernel(scatter);
  api.clReleaseKernel(swiglu_quant);
  api.clReleaseKernel(input_quant);
  api.clReleaseProgram(program);
  api.clReleaseCommandQueue(queue);
  api.clReleaseContext(context);
  return correctness_pass && performance_pass ? 0 : 2;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args args = ParseArgs(argc, argv);
    if (args.kernel_mode == "prepacked_routed") {
      return RunPrepackedRouted(args);
    }
    const auto input = ReadVector<float>(args.input, kTokenCount * kHiddenSize);
    const auto topk = ReadBytes(args.topk);
    const auto oracle =
        ReadVector<float>(args.oracle, kAssignments * kIntermediateSize);
    const auto q4 = ReadModelSlice(
        args.model, args.weight_offset, args.weight_bytes);
    const auto q8 = QuantizeQ8K(input);
    const std::size_t task_tokens = args.kernel_mode == "m8_u4"
        ? kM8TaskTokens : kM1TaskTokens;
    const BucketPlan plan = BuildPlan(topk, args.topk_stride, task_tokens);

    OpenClApi api;
    const DeviceSelection selection = SelectDevice(api, args.device_substring);
    cl_int error = kClSuccess;
    cl_context context =
        api.clCreateContext(nullptr, 1, &selection.device, nullptr, nullptr, &error);
    Check(error, "clCreateContext");
    cl_command_queue queue = api.clCreateCommandQueue(
        context, selection.device, kClQueueProfilingEnable, &error);
    Check(error, "clCreateCommandQueue");

    const std::size_t source_length = std::strlen(kOpenClSource);
    cl_program program = api.clCreateProgramWithSource(
        context, 1, &kOpenClSource, &source_length, &error);
    Check(error, "clCreateProgramWithSource");
    const cl_int build_code =
        api.clBuildProgram(program, 1, &selection.device, "-cl-std=CL2.0",
                           nullptr, nullptr);
    const std::string build_log = BuildLog(api, program, selection.device);
    if (build_code != kClSuccess) {
      Die("OpenCL build failed: " + build_log);
    }
    const char* kernel_name = args.kernel_mode == "m8_u4"
        ? "expert_bucket_gateup_swiglu_m8_u4"
        : "expert_bucket_gateup_swiglu";
    cl_kernel kernel = api.clCreateKernel(program, kernel_name, &error);
    Check(error, "clCreateKernel");

    cl_mem q4_buffer = CreateAndWrite(
        api, context, queue, q4, kClMemReadOnly, "q4");
    cl_mem q8_buffer = CreateAndWrite(
        api, context, queue, q8, kClMemReadOnly, "q8");
    cl_mem expert_buffer = CreateAndWrite(
        api, context, queue, plan.task_expert, kClMemReadOnly, "task expert");
    cl_mem base_buffer = CreateAndWrite(
        api, context, queue, plan.task_bucket_base, kClMemReadOnly, "task base");
    cl_mem count_buffer = CreateAndWrite(
        api, context, queue, plan.task_token_count, kClMemReadOnly, "task count");
    cl_mem token_buffer = CreateAndWrite(
        api, context, queue, plan.bucket_token, kClMemReadOnly, "bucket token");
    std::vector<float> output(kAssignments * kIntermediateSize, 0.0f);
    cl_mem output_buffer = api.clCreateBuffer(
        context, kClMemReadWrite, output.size() * sizeof(float), nullptr, &error);
    Check(error, "clCreateBuffer output");

    const cl_uint task_total = static_cast<cl_uint>(plan.task_expert.size());
    Check(api.clSetKernelArg(kernel, 0, sizeof(q4_buffer), &q4_buffer),
          "clSetKernelArg q4");
    Check(api.clSetKernelArg(kernel, 1, sizeof(q8_buffer), &q8_buffer),
          "clSetKernelArg q8");
    Check(api.clSetKernelArg(kernel, 2, sizeof(expert_buffer), &expert_buffer),
          "clSetKernelArg task expert");
    Check(api.clSetKernelArg(kernel, 3, sizeof(base_buffer), &base_buffer),
          "clSetKernelArg task base");
    Check(api.clSetKernelArg(kernel, 4, sizeof(count_buffer), &count_buffer),
          "clSetKernelArg task count");
    Check(api.clSetKernelArg(kernel, 5, sizeof(token_buffer), &token_buffer),
          "clSetKernelArg bucket token");
    Check(api.clSetKernelArg(kernel, 6, sizeof(output_buffer), &output_buffer),
          "clSetKernelArg output");
    Check(api.clSetKernelArg(kernel, 7, sizeof(task_total), &task_total),
          "clSetKernelArg task total");
    const std::size_t local_weight_bytes = 32 * kQ4BlockBytes;
    Check(api.clSetKernelArg(kernel, 8, local_weight_bytes, nullptr),
          "clSetKernelArg local weight");

    const std::size_t local = 128;
    const std::size_t global =
        plan.task_expert.size() * (kIntermediateSize / 16) * local;
    for (int warmup = 0; warmup < 2; ++warmup) {
      Check(api.clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global,
                                       &local, 0, nullptr, nullptr),
            "clEnqueueNDRangeKernel warmup");
      Check(api.clFinish(queue), "clFinish warmup");
    }

    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(args.repeat));
    for (int repeat = 0; repeat < args.repeat; ++repeat) {
      cl_event event = nullptr;
      Check(api.clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global,
                                       &local, 0, nullptr, &event),
            "clEnqueueNDRangeKernel timed");
      Check(api.clFinish(queue), "clFinish timed");
      samples.push_back(EventMicroseconds(api, event));
      api.clReleaseEvent(event);
    }
    Check(api.clEnqueueReadBuffer(queue, output_buffer, kClTrue, 0,
                                  output.size() * sizeof(float), output.data(),
                                  0, nullptr, nullptr),
          "clEnqueueReadBuffer output");

    const CompareStats compare = Compare(output, oracle, plan);
    std::vector<double> sorted = samples;
    std::sort(sorted.begin(), sorted.end());
    const double minimum_us = sorted.front();
    const double median_us = sorted[sorted.size() / 2];
    const double mean_us =
        std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
    const std::uint64_t task_weight_bytes =
        static_cast<std::uint64_t>(plan.task_expert.size()) * kExpertWeightBytes;
    const std::uint64_t timed_bytes =
        task_weight_bytes + q8.size() + output.size() * sizeof(float) +
        plan.bucket_token.size() * sizeof(cl_uint) +
        plan.task_expert.size() * sizeof(cl_uint) * 3;
    const double effective_gb_s = timed_bytes / (minimum_us * 1000.0);
    const bool correctness_pass =
        compare.finite && compare.mismatch_count == 0 &&
        compare.max_abs <= kMismatchThreshold && compare.rmse <= 5e-4 &&
        compare.cosine >= 0.999;
    const bool performance_pass = minimum_us <= args.kernel_cap_us;

    std::cout << std::setprecision(12) << "{";
    std::cout << "\"active_experts\":" << plan.active_experts << ",";
    std::cout << "\"assignment_count\":" << plan.bucket_token.size() << ",";
    std::cout << "\"build_log\":\"" << JsonEscape(build_log) << "\",";
    std::cout << "\"compare\":{";
    std::cout << "\"compared_value_count\":" << compare.count << ",";
    std::cout << "\"cosine\":" << compare.cosine << ",";
    std::cout << "\"finite\":" << (compare.finite ? "true" : "false") << ",";
    std::cout << "\"max_abs_diff\":" << compare.max_abs << ",";
    std::cout << "\"mean_abs_diff\":" << compare.mean_abs << ",";
    std::cout << "\"mismatch_count\":" << compare.mismatch_count << ",";
    std::cout << "\"rmse\":" << compare.rmse << "},";
    std::cout << "\"correctness_pass\":"
              << (correctness_pass ? "true" : "false") << ",";
    std::cout << "\"device_name\":\"" << JsonEscape(selection.device_name)
              << "\",";
    std::cout << "\"effective_gb_s\":" << effective_gb_s << ",";
    std::cout << "\"global_work_items\":" << global << ",";
    std::cout << "\"kernel_cap_us\":" << args.kernel_cap_us << ",";
    std::cout << "\"kernel_mode\":\"" << args.kernel_mode << "\",";
    std::cout << "\"kernel_mean_us\":" << mean_us << ",";
    std::cout << "\"kernel_median_us\":" << median_us << ",";
    std::cout << "\"kernel_min_us\":" << minimum_us << ",";
    std::cout << "\"kernel_normalized_per_64_us\":"
              << minimum_us / 16.0 << ",";
    std::cout << "\"max_group_m\":" << plan.max_group_m << ",";
    std::cout << "\"performance_pass\":"
              << (performance_pass ? "true" : "false") << ",";
    std::cout << "\"platform_name\":\""
              << JsonEscape(selection.platform_name) << "\",";
    std::cout << "\"samples_us\":[";
    for (std::size_t index = 0; index < samples.size(); ++index) {
      if (index != 0) std::cout << ",";
      std::cout << samples[index];
    }
    std::cout << "],";
    std::cout << "\"task_count\":" << plan.task_expert.size() << ",";
    std::cout << "\"task_tokens\":" << task_tokens << ",";
    std::cout << "\"task_weight_load_bytes\":" << task_weight_bytes << ",";
    std::cout << "\"timed_traffic_bytes\":" << timed_bytes;
    std::cout << "}\n";

    api.clReleaseMemObject(output_buffer);
    api.clReleaseMemObject(token_buffer);
    api.clReleaseMemObject(count_buffer);
    api.clReleaseMemObject(base_buffer);
    api.clReleaseMemObject(expert_buffer);
    api.clReleaseMemObject(q8_buffer);
    api.clReleaseMemObject(q4_buffer);
    api.clReleaseKernel(kernel);
    api.clReleaseProgram(program);
    api.clReleaseCommandQueue(queue);
    api.clReleaseContext(context);
    return correctness_pass && performance_pass ? 0 : 2;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << "\n";
    return 1;
  }
}
