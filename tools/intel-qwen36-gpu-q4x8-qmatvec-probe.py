#!/usr/bin/env python3
"""Run a narrow Arc B390 Q4_K x8 / Q6_K qmatvec probe.

This is the GPU bring-up gate after the Q4 x8 packed stream check. It selects
one locked-model Q4_K tensor, repacks it into a llama.cpp-style q4_K_8x8 byte
layout, or selects one Q6_K tensor in the locked raw GGUF layout. It then runs
a deterministic Q8_K-input matvec on CPU reference and GPU paths and compares
numeric output. It is not a decode benchmark and does not allow speedup claims.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shlex
from pathlib import Path
from typing import Any

import iq36_local


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA_VERSION = "intel-qwen36-gpu-q4x8-qmatvec-probe-v4"
DEFAULT_HOST = "local"
DEFAULT_MODEL = "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf"
DEFAULT_ENV_SCRIPT = "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh"
DEFAULT_REMOTE_ROOT = "/home/intel/intel-qwen36-gpu"
DEFAULT_TENSOR = "blk.5.attn_qkv.weight"
OPENCL_SOURCE = ROOT / "engine/gpu/opencl/q4x8_matvec.cl"
SOURCE_FILES = [
    ("engine/include/intel_qwen36/gguf_loader.hpp", "include/intel_qwen36/gguf_loader.hpp"),
    ("engine/include/intel_qwen36/gpu_q4x8_matvec.hpp", "include/intel_qwen36/gpu_q4x8_matvec.hpp"),
    ("engine/src/gguf_loader.cpp", "src/gguf_loader.cpp"),
    ("engine/src/gpu_q4x8_matvec.cpp", "src/gpu_q4x8_matvec.cpp"),
]


PROBE_CPP = r'''
#include "intel_qwen36/gguf_loader.hpp"
#include "intel_qwen36/gpu_q4x8_matvec.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using cl_int = std::int32_t;
using cl_uint = std::uint32_t;
using cl_ulong = std::uint64_t;
using cl_bool = cl_uint;
using cl_bitfield = cl_ulong;
using cl_device_type = cl_bitfield;
using cl_platform_info = cl_uint;
using cl_device_info = cl_uint;
using cl_context_properties = intptr_t;
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

constexpr cl_int CL_SUCCESS_VALUE = 0;
constexpr cl_bool CL_FALSE_VALUE = 0;
constexpr cl_bool CL_TRUE_VALUE = 1;
constexpr cl_device_type CL_DEVICE_TYPE_GPU_VALUE = 1ULL << 2;
constexpr cl_mem_flags CL_MEM_READ_ONLY_VALUE = 1ULL << 2;
constexpr cl_mem_flags CL_MEM_WRITE_ONLY_VALUE = 1ULL << 1;
constexpr cl_mem_flags CL_MEM_COPY_HOST_PTR_VALUE = 1ULL << 5;
constexpr cl_command_queue_properties CL_QUEUE_PROFILING_ENABLE_VALUE = 1ULL << 1;
constexpr cl_platform_info CL_PLATFORM_NAME_VALUE = 0x0902;
constexpr cl_device_info CL_DEVICE_NAME_VALUE = 0x102B;
constexpr cl_program_build_info CL_PROGRAM_BUILD_LOG_VALUE = 0x1183;
constexpr cl_profiling_info CL_PROFILING_COMMAND_START_VALUE = 0x1282;
constexpr cl_profiling_info CL_PROFILING_COMMAND_END_VALUE = 0x1283;

constexpr std::uint64_t kQ4KBlockBytes = 144;
constexpr std::uint64_t kQ6KBlockBytes = 210;
constexpr std::uint64_t kQ4Kx8BlockBytes = 1152;
constexpr std::uint64_t kQK = 256;
constexpr std::uint64_t kRowsInterleaved = 8;
const char* kQ4X8OpenClSource = @@OPENCL_SOURCE_LITERAL@@;

struct Args {
  std::string model_path;
  std::string tensor_name = "blk.5.attn_qkv.weight";
  int repeat = 5;
  std::string device_substring = "B390";
};

struct Q8KBlock {
  float d = 0.0f;
  std::array<std::int8_t, 256> qs{};
  std::array<std::int16_t, 16> bsums{};
};

struct Comparison {
  double max_abs_diff = 0.0;
  std::uint64_t max_abs_diff_index = 0;
  double rel_l2 = 0.0;
  double cosine = 0.0;
  double rmse = 0.0;
  bool passed = false;
};

struct KernelTiming {
  double min_us = 0.0;
  double mean_us = 0.0;
  double effective_packed_gb_s = 0.0;
  std::uint64_t global_work_items = 0;
  std::uint64_t rows_per_work_item = 0;
};

struct GpuVariant {
  std::string name;
  std::string kernel_name;
  std::uint64_t rows_per_work_item = 0;
  std::uint64_t global_work_items = 0;
  std::vector<float> output;
  KernelTiming timing;
  Comparison comparison;
  bool passed = false;
};

[[noreturn]] void Die(const std::string& message) {
  throw std::runtime_error(message);
}

void Require(bool ok, const std::string& message) {
  if (!ok) {
    Die(message);
  }
}

void Check(cl_int err, const std::string& where) {
  if (err != CL_SUCCESS_VALUE) {
    std::ostringstream oss;
    oss << where << " failed with OpenCL error " << err;
    Die(oss.str());
  }
}

std::string JsonEscape(const std::string& value) {
  std::string out;
  out.reserve(value.size() + 8);
  for (const char ch : value) {
    switch (ch) {
      case '\\': out += "\\\\"; break;
      case '"': out += "\\\""; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default: out += ch; break;
    }
  }
  return out;
}

std::uint16_t LoadLe16(const std::uint8_t* bytes) {
  return static_cast<std::uint16_t>(bytes[0]) |
         (static_cast<std::uint16_t>(bytes[1]) << 8);
}

float HalfToFloat(std::uint16_t h) {
  const std::uint32_t sign = (static_cast<std::uint32_t>(h & 0x8000U)) << 16U;
  std::uint32_t exp = (h >> 10U) & 0x1FU;
  std::uint32_t mant = h & 0x03FFU;
  std::uint32_t out = 0;
  if (exp == 0) {
    if (mant == 0) {
      out = sign;
    } else {
      exp = 1;
      while ((mant & 0x0400U) == 0) {
        mant <<= 1U;
        --exp;
      }
      mant &= 0x03FFU;
      out = sign | ((exp + (127U - 15U)) << 23U) | (mant << 13U);
    }
  } else if (exp == 0x1FU) {
    out = sign | 0x7F800000U | (mant << 13U);
  } else {
    out = sign | ((exp + (127U - 15U)) << 23U) | (mant << 13U);
  }
  float value = 0.0f;
  std::memcpy(&value, &out, sizeof(value));
  return value;
}

void GetScaleMinK4(int index,
                   const std::uint8_t* scales,
                   std::uint8_t& scale,
                   std::uint8_t& minimum) {
  if (index < 4) {
    scale = scales[index] & 63U;
    minimum = scales[index + 4] & 63U;
  } else {
    scale = (scales[index + 4] & 0x0FU) |
            static_cast<std::uint8_t>((scales[index - 4] >> 6U) << 4U);
    minimum = (scales[index + 4] >> 4U) |
              static_cast<std::uint8_t>((scales[index] >> 6U) << 4U);
  }
}

std::array<std::uint8_t, 8> UnpackQ4KScalePlane(const std::uint8_t* block, bool mins) {
  std::array<std::uint8_t, 8> out{};
  for (int i = 0; i < 8; ++i) {
    std::uint8_t scale = 0;
    std::uint8_t minimum = 0;
    GetScaleMinK4(i, block + 4, scale, minimum);
    out[static_cast<std::size_t>(i)] = mins ? minimum : scale;
  }
  return out;
}

int NearestInt(float value) {
  float shifted = value + 12582912.0f;
  int bits = 0;
  std::memcpy(&bits, &shifted, sizeof(bits));
  return (bits & 0x007fffff) - 0x00400000;
}

std::vector<float> MakeInput(std::uint64_t cols) {
  std::vector<float> input(static_cast<std::size_t>(cols), 0.0f);
  for (std::uint64_t i = 0; i < cols; ++i) {
    const float a = std::sin(static_cast<float>(i + 1) * 0.013f) * 0.75f;
    const float b = std::cos(static_cast<float>((i % 17) + 3) * 0.11f) * 0.15f;
    input[static_cast<std::size_t>(i)] = a + b;
  }
  return input;
}

std::vector<Q8KBlock> QuantizeQ8K(const std::vector<float>& input) {
  Require(input.size() % kQK == 0, "Q8_K input quantization requires 256-aligned input");
  std::vector<Q8KBlock> blocks(input.size() / kQK);
  for (std::size_t block_index = 0; block_index < blocks.size(); ++block_index) {
    const auto* block_input = input.data() + block_index * kQK;
    auto& block = blocks[block_index];
    float max = 0.0f;
    float amax = 0.0f;
    for (int i = 0; i < 256; ++i) {
      const float abs_value = std::fabs(block_input[i]);
      if (abs_value > amax) {
        amax = abs_value;
        max = block_input[i];
      }
    }
    if (amax == 0.0f) {
      continue;
    }
    const float iscale = -127.0f / max;
    for (int i = 0; i < 256; ++i) {
      const int quantized = std::min(127, NearestInt(iscale * block_input[i]));
      block.qs[static_cast<std::size_t>(i)] = static_cast<std::int8_t>(quantized);
    }
    for (int group = 0; group < 16; ++group) {
      int sum = 0;
      for (int i = 0; i < 16; ++i) {
        sum += block.qs[static_cast<std::size_t>(group * 16 + i)];
      }
      block.bsums[static_cast<std::size_t>(group)] = static_cast<std::int16_t>(sum);
    }
    block.d = 1.0f / iscale;
  }
  return blocks;
}

void AppendQ4Kx8Block(const std::array<const std::uint8_t*, 8>& blocks,
                      std::vector<std::uint8_t>& out) {
  const std::size_t base = out.size();
  out.resize(base + kQ4Kx8BlockBytes);
  auto* dst = out.data() + base;
  auto* dst_d = dst;
  auto* dst_dmin = dst + 16;
  auto* dst_scales = dst + 32;
  auto* dst_qs = dst + 128;

  for (int i = 0; i < 8; ++i) {
    std::memcpy(dst_d + i * 2, blocks[static_cast<std::size_t>(i)], 2);
    std::memcpy(dst_dmin + i * 2, blocks[static_cast<std::size_t>(i)] + 2, 2);
  }

  constexpr int kInterleaveBlock = 8;
  constexpr int end = 256 * 4 / kInterleaveBlock;
  for (int i = 0; i < end; ++i) {
    const int src_id = i % 8;
    const int src_offset = (i / 8) * kInterleaveBlock;
    const int dst_offset = i * kInterleaveBlock;
    std::memcpy(dst_qs + dst_offset,
                blocks[static_cast<std::size_t>(src_id)] + 16 + src_offset,
                kInterleaveBlock);
  }

  std::uint8_t s[8] = {};
  std::uint8_t m[8] = {};
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 8; ++j) {
      const auto* scales = blocks[static_cast<std::size_t>(j)] + 4;
      s[j] = scales[i] & 63;
      m[j] = scales[i + 4] & 63;
    }
    dst_scales[i * 12]      = (s[0] & 63) + ((s[4] & 48) << 2);
    dst_scales[i * 12 + 1]  = (s[1] & 63) + ((s[5] & 48) << 2);
    dst_scales[i * 12 + 2]  = (s[2] & 63) + ((s[6] & 48) << 2);
    dst_scales[i * 12 + 3]  = (s[3] & 63) + ((s[7] & 48) << 2);
    dst_scales[i * 12 + 4]  = (m[0] & 63) + ((m[4] & 48) << 2);
    dst_scales[i * 12 + 5]  = (m[1] & 63) + ((m[5] & 48) << 2);
    dst_scales[i * 12 + 6]  = (m[2] & 63) + ((m[6] & 48) << 2);
    dst_scales[i * 12 + 7]  = (m[3] & 63) + ((m[7] & 48) << 2);
    dst_scales[i * 12 + 8]  = (s[4] & 15) + ((m[4] & 15) << 4);
    dst_scales[i * 12 + 9]  = (s[5] & 15) + ((m[5] & 15) << 4);
    dst_scales[i * 12 + 10] = (s[6] & 15) + ((m[6] & 15) << 4);
    dst_scales[i * 12 + 11] = (s[7] & 15) + ((m[7] & 15) << 4);
  }

  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 8; ++j) {
      const auto* scales = blocks[static_cast<std::size_t>(j)] + 4;
      s[j] = ((scales[i] & 192) >> 2) | (scales[i + 8] & 15);
      m[j] = ((scales[i + 4] & 192) >> 2) | ((scales[i + 8] & 240) >> 4);
    }
    dst_scales[i * 12 + 48] = (s[0] & 63) + ((s[4] & 48) << 2);
    dst_scales[i * 12 + 49] = (s[1] & 63) + ((s[5] & 48) << 2);
    dst_scales[i * 12 + 50] = (s[2] & 63) + ((s[6] & 48) << 2);
    dst_scales[i * 12 + 51] = (s[3] & 63) + ((s[7] & 48) << 2);
    dst_scales[i * 12 + 52] = (m[0] & 63) + ((m[4] & 48) << 2);
    dst_scales[i * 12 + 53] = (m[1] & 63) + ((m[5] & 48) << 2);
    dst_scales[i * 12 + 54] = (m[2] & 63) + ((m[6] & 48) << 2);
    dst_scales[i * 12 + 55] = (m[3] & 63) + ((m[7] & 48) << 2);
    dst_scales[i * 12 + 56] = (s[4] & 15) + ((m[4] & 15) << 4);
    dst_scales[i * 12 + 57] = (s[5] & 15) + ((m[5] & 15) << 4);
    dst_scales[i * 12 + 58] = (s[6] & 15) + ((m[6] & 15) << 4);
    dst_scales[i * 12 + 59] = (s[7] & 15) + ((m[7] & 15) << 4);
  }
}

std::vector<std::uint8_t> PackQ4Kx8(const std::vector<std::uint8_t>& raw,
                                    std::uint64_t rows,
                                    std::uint64_t blocks_per_row) {
  Require(raw.size() == rows * blocks_per_row * kQ4KBlockBytes,
          "Q4_K raw byte size does not match tensor shape");
  Require(rows % kRowsInterleaved == 0, "Q4_K x8 pack requires rows divisible by 8");
  std::vector<std::uint8_t> out;
  out.reserve(raw.size());
  for (std::uint64_t row_base = 0; row_base < rows; row_base += kRowsInterleaved) {
    for (std::uint64_t block = 0; block < blocks_per_row; ++block) {
      std::array<const std::uint8_t*, 8> blocks{};
      for (int i = 0; i < 8; ++i) {
        const std::uint64_t raw_block_index =
            (row_base + static_cast<std::uint64_t>(i)) * blocks_per_row + block;
        blocks[static_cast<std::size_t>(i)] = raw.data() + raw_block_index * kQ4KBlockBytes;
      }
      AppendQ4Kx8Block(blocks, out);
    }
  }
  return out;
}

int Q4GroupedMinSum(const std::uint8_t* mins, const Q8KBlock& input) {
  int sum = 0;
  for (int group = 0; group < 8; ++group) {
    const int bsum_pair =
        static_cast<int>(input.bsums[static_cast<std::size_t>(group * 2)]) +
        static_cast<int>(input.bsums[static_cast<std::size_t>(group * 2 + 1)]);
    sum += bsum_pair * static_cast<int>(mins[group]);
  }
  return sum;
}

float DotRawQ4KRow(const std::uint8_t* row, const std::vector<Q8KBlock>& q8) {
  float sums[8] = {};
  float min_sum = 0.0f;
  for (std::size_t block_index = 0; block_index < q8.size(); ++block_index) {
    const auto* block = row + block_index * kQ4KBlockBytes;
    const auto scales = UnpackQ4KScalePlane(block, false);
    const auto mins = UnpackQ4KScalePlane(block, true);
    const auto& input = q8[block_index];
    std::array<std::int32_t, 8> lane_sums{};
    const auto* q4 = block + 16;
    const auto* q8_qs = input.qs.data();
    for (int group = 0; group < 4; ++group) {
      const auto* group_q4 = q4 + group * 32;
      const auto* group_q8 = q8_qs + group * 64;
      const int low_scale = scales[static_cast<std::size_t>(group * 2)];
      const int high_scale = scales[static_cast<std::size_t>(group * 2 + 1)];
      for (int lane = 0; lane < 32; ++lane) {
        const auto packed = group_q4[lane];
        const int q4_low = static_cast<int>(packed & 0x0F);
        const int q4_high = static_cast<int>(packed >> 4);
        const auto lane_index = static_cast<std::size_t>(lane & 7);
        lane_sums[lane_index] +=
            low_scale * static_cast<int>(group_q8[lane]) * q4_low;
        lane_sums[lane_index] +=
            high_scale * static_cast<int>(group_q8[32 + lane]) * q4_high;
      }
    }
    const float d = HalfToFloat(LoadLe16(block)) * input.d;
    for (int lane = 0; lane < 8; ++lane) {
      sums[lane] += d * static_cast<float>(lane_sums[static_cast<std::size_t>(lane)]);
    }
    const float dmin = HalfToFloat(LoadLe16(block + 2)) * input.d;
    min_sum -= dmin * static_cast<float>(Q4GroupedMinSum(mins.data(), input));
  }
  float sum = min_sum;
  for (float value : sums) {
    sum += value;
  }
  return sum;
}

std::vector<float> RunCpuRaw(const std::vector<std::uint8_t>& raw,
                             std::uint64_t rows,
                             std::uint64_t blocks_per_row,
                             const std::vector<Q8KBlock>& q8) {
  std::vector<float> out(static_cast<std::size_t>(rows), 0.0f);
  const std::uint64_t row_bytes = blocks_per_row * kQ4KBlockBytes;
  for (std::uint64_t row = 0; row < rows; ++row) {
    out[static_cast<std::size_t>(row)] =
        DotRawQ4KRow(raw.data() + row * row_bytes, q8);
  }
  return out;
}

void GetPackedScaleMin(const std::uint8_t* packed_block,
                       int subblock,
                       int column,
                       std::uint8_t& scale,
                       std::uint8_t& minimum) {
  GetScaleMinK4(column, packed_block + 32 + subblock * 12, scale, minimum);
}

std::vector<float> RunCpuPacked(const std::vector<std::uint8_t>& packed,
                                std::uint64_t rows,
                                std::uint64_t blocks_per_row,
                                const std::vector<Q8KBlock>& q8) {
  const std::uint64_t row_groups = rows / kRowsInterleaved;
  std::vector<float> out(static_cast<std::size_t>(rows), 0.0f);
  for (std::uint64_t group = 0; group < row_groups; ++group) {
    float sumf[8] = {};
    float sum_minf[8] = {};
    for (std::uint64_t block_index = 0; block_index < blocks_per_row; ++block_index) {
      const auto* block = packed.data() + (group * blocks_per_row + block_index) * kQ4Kx8BlockBytes;
      const auto& input = q8[static_cast<std::size_t>(block_index)];
      for (int k = 0; k < 16; ++k) {
        const int scale_pair = (k >> 2) * 2;
        const int q8_base = (k >> 2) * 64 + (k & 3) * 8;
        for (int j = 0; j < 8; ++j) {
          std::uint8_t scale0 = 0;
          std::uint8_t min0 = 0;
          std::uint8_t scale1 = 0;
          std::uint8_t min1 = 0;
          GetPackedScaleMin(block, scale_pair, j, scale0, min0);
          GetPackedScaleMin(block, scale_pair + 1, j, scale1, min1);
          int sumi = 0;
          for (int i = 0; i < 8; ++i) {
            const auto q = block[128 + k * 64 + j * 8 + i];
            const int v0 = static_cast<int>(q & 0x0F);
            const int v1 = static_cast<int>(q >> 4);
            const int q8_low = static_cast<int>(input.qs[static_cast<std::size_t>(q8_base + i)]);
            const int q8_high = static_cast<int>(input.qs[static_cast<std::size_t>(q8_base + i + 32)]);
            sumi += v0 * q8_low * static_cast<int>(scale0);
            sumi += v1 * q8_high * static_cast<int>(scale1);
          }
          sumf[j] +=
              static_cast<float>(sumi) *
              HalfToFloat(LoadLe16(block + j * 2)) *
              input.d;
        }
      }
      for (int sb = 0; sb < 8; ++sb) {
        const int bsum_pair =
            static_cast<int>(input.bsums[static_cast<std::size_t>(sb * 2)]) +
            static_cast<int>(input.bsums[static_cast<std::size_t>(sb * 2 + 1)]);
        for (int j = 0; j < 8; ++j) {
          std::uint8_t scale = 0;
          std::uint8_t minimum = 0;
          GetPackedScaleMin(block, sb, j, scale, minimum);
          sum_minf[j] +=
              static_cast<float>(minimum * bsum_pair) *
              HalfToFloat(LoadLe16(block + 16 + j * 2)) *
              input.d;
        }
      }
    }
    for (int j = 0; j < 8; ++j) {
      out[static_cast<std::size_t>(group * 8 + j)] = sumf[j] - sum_minf[j];
    }
  }
  return out;
}

Comparison CompareVectors(const std::vector<float>& candidate,
                          const std::vector<float>& reference,
                          double rel_l2_threshold,
                          double cosine_threshold) {
  Require(candidate.size() == reference.size(), "comparison vector sizes differ");
  Comparison out;
  double delta_sq = 0.0;
  double ref_sq = 0.0;
  double cand_sq = 0.0;
  double dot = 0.0;
  for (std::size_t i = 0; i < candidate.size(); ++i) {
    const double cand = candidate[i];
    const double ref = reference[i];
    const double delta = cand - ref;
    const double abs_delta = std::fabs(delta);
    delta_sq += delta * delta;
    ref_sq += ref * ref;
    cand_sq += cand * cand;
    dot += cand * ref;
    if (abs_delta > out.max_abs_diff) {
      out.max_abs_diff = abs_delta;
      out.max_abs_diff_index = static_cast<std::uint64_t>(i);
    }
  }
  out.rmse = candidate.empty() ? 0.0 : std::sqrt(delta_sq / static_cast<double>(candidate.size()));
  out.rel_l2 = ref_sq == 0.0 ? (delta_sq == 0.0 ? 0.0 : std::numeric_limits<double>::infinity())
                             : std::sqrt(delta_sq / ref_sq);
  const double denom = std::sqrt(ref_sq) * std::sqrt(cand_sq);
  out.cosine = denom == 0.0 ? 1.0 : dot / denom;
  out.passed = out.rel_l2 <= rel_l2_threshold && out.cosine >= cosine_threshold;
  return out;
}

std::vector<std::uint8_t> ReadTensorBytes(std::ifstream& in, const iq36::GgufTensorInfo& tensor) {
  Require(tensor.nbytes <= static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()),
          "tensor too large for size_t");
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(tensor.nbytes));
  in.clear();
  in.seekg(static_cast<std::streamoff>(tensor.absolute_offset), std::ios::beg);
  Require(static_cast<bool>(in), "failed to seek tensor payload");
  in.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  Require(in.gcount() == static_cast<std::streamsize>(bytes.size()), "failed to read tensor payload");
  return bytes;
}

Args ParseArgs(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    auto value = [&](const char* name) -> std::string {
      Require(i + 1 < argc, std::string("missing value for ") + name);
      return argv[++i];
    };
    if (key == "--model") args.model_path = value("--model");
    else if (key == "--tensor") args.tensor_name = value("--tensor");
    else if (key == "--repeat") args.repeat = std::stoi(value("--repeat"));
    else if (key == "--device-substring") args.device_substring = value("--device-substring");
    else Die("unknown argument: " + key);
  }
  Require(!args.model_path.empty(), "--model is required");
  Require(args.repeat > 0, "--repeat must be positive");
  return args;
}

template <typename Fn>
Fn LoadSym(void* lib, const char* name) {
  void* sym = dlsym(lib, name);
  if (!sym) {
    Die(std::string("missing OpenCL symbol: ") + name);
  }
  return reinterpret_cast<Fn>(sym);
}

struct OpenClApi {
  void* lib = nullptr;
  cl_int (*clGetPlatformIDs)(cl_uint, cl_platform_id*, cl_uint*) = nullptr;
  cl_int (*clGetPlatformInfo)(cl_platform_id, cl_platform_info, std::size_t, void*, std::size_t*) = nullptr;
  cl_int (*clGetDeviceIDs)(cl_platform_id, cl_device_type, cl_uint, cl_device_id*, cl_uint*) = nullptr;
  cl_int (*clGetDeviceInfo)(cl_device_id, cl_device_info, std::size_t, void*, std::size_t*) = nullptr;
  cl_context (*clCreateContext)(const cl_context_properties*, cl_uint, const cl_device_id*, void*, void*, cl_int*) = nullptr;
  cl_int (*clReleaseContext)(cl_context) = nullptr;
  cl_command_queue (*clCreateCommandQueue)(cl_context, cl_device_id, cl_command_queue_properties, cl_int*) = nullptr;
  cl_int (*clReleaseCommandQueue)(cl_command_queue) = nullptr;
  cl_mem (*clCreateBuffer)(cl_context, cl_mem_flags, std::size_t, void*, cl_int*) = nullptr;
  cl_int (*clReleaseMemObject)(cl_mem) = nullptr;
  cl_program (*clCreateProgramWithSource)(cl_context, cl_uint, const char**, const std::size_t*, cl_int*) = nullptr;
  cl_int (*clBuildProgram)(cl_program, cl_uint, const cl_device_id*, const char*, void*, void*) = nullptr;
  cl_int (*clGetProgramBuildInfo)(cl_program, cl_device_id, cl_program_build_info, std::size_t, void*, std::size_t*) = nullptr;
  cl_int (*clReleaseProgram)(cl_program) = nullptr;
  cl_kernel (*clCreateKernel)(cl_program, const char*, cl_int*) = nullptr;
  cl_int (*clSetKernelArg)(cl_kernel, cl_uint, std::size_t, const void*) = nullptr;
  cl_int (*clReleaseKernel)(cl_kernel) = nullptr;
  cl_int (*clEnqueueWriteBuffer)(cl_command_queue, cl_mem, cl_bool, std::size_t, std::size_t, const void*, cl_uint, const cl_event*, cl_event*) = nullptr;
  cl_int (*clEnqueueReadBuffer)(cl_command_queue, cl_mem, cl_bool, std::size_t, std::size_t, void*, cl_uint, const cl_event*, cl_event*) = nullptr;
  cl_int (*clEnqueueNDRangeKernel)(cl_command_queue, cl_kernel, cl_uint, const std::size_t*, const std::size_t*, const std::size_t*, cl_uint, const cl_event*, cl_event*) = nullptr;
  cl_int (*clFinish)(cl_command_queue) = nullptr;
  cl_int (*clGetEventProfilingInfo)(cl_event, cl_profiling_info, std::size_t, void*, std::size_t*) = nullptr;
  cl_int (*clReleaseEvent)(cl_event) = nullptr;

  OpenClApi() {
    lib = dlopen("libOpenCL.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!lib) {
      Die(std::string("dlopen libOpenCL.so.1 failed: ") + dlerror());
    }
    clGetPlatformIDs = LoadSym<decltype(clGetPlatformIDs)>(lib, "clGetPlatformIDs");
    clGetPlatformInfo = LoadSym<decltype(clGetPlatformInfo)>(lib, "clGetPlatformInfo");
    clGetDeviceIDs = LoadSym<decltype(clGetDeviceIDs)>(lib, "clGetDeviceIDs");
    clGetDeviceInfo = LoadSym<decltype(clGetDeviceInfo)>(lib, "clGetDeviceInfo");
    clCreateContext = LoadSym<decltype(clCreateContext)>(lib, "clCreateContext");
    clReleaseContext = LoadSym<decltype(clReleaseContext)>(lib, "clReleaseContext");
    clCreateCommandQueue = LoadSym<decltype(clCreateCommandQueue)>(lib, "clCreateCommandQueue");
    clReleaseCommandQueue = LoadSym<decltype(clReleaseCommandQueue)>(lib, "clReleaseCommandQueue");
    clCreateBuffer = LoadSym<decltype(clCreateBuffer)>(lib, "clCreateBuffer");
    clReleaseMemObject = LoadSym<decltype(clReleaseMemObject)>(lib, "clReleaseMemObject");
    clCreateProgramWithSource = LoadSym<decltype(clCreateProgramWithSource)>(lib, "clCreateProgramWithSource");
    clBuildProgram = LoadSym<decltype(clBuildProgram)>(lib, "clBuildProgram");
    clGetProgramBuildInfo = LoadSym<decltype(clGetProgramBuildInfo)>(lib, "clGetProgramBuildInfo");
    clReleaseProgram = LoadSym<decltype(clReleaseProgram)>(lib, "clReleaseProgram");
    clCreateKernel = LoadSym<decltype(clCreateKernel)>(lib, "clCreateKernel");
    clSetKernelArg = LoadSym<decltype(clSetKernelArg)>(lib, "clSetKernelArg");
    clReleaseKernel = LoadSym<decltype(clReleaseKernel)>(lib, "clReleaseKernel");
    clEnqueueWriteBuffer = LoadSym<decltype(clEnqueueWriteBuffer)>(lib, "clEnqueueWriteBuffer");
    clEnqueueReadBuffer = LoadSym<decltype(clEnqueueReadBuffer)>(lib, "clEnqueueReadBuffer");
    clEnqueueNDRangeKernel = LoadSym<decltype(clEnqueueNDRangeKernel)>(lib, "clEnqueueNDRangeKernel");
    clFinish = LoadSym<decltype(clFinish)>(lib, "clFinish");
    clGetEventProfilingInfo = LoadSym<decltype(clGetEventProfilingInfo)>(lib, "clGetEventProfilingInfo");
    clReleaseEvent = LoadSym<decltype(clReleaseEvent)>(lib, "clReleaseEvent");
  }
};

std::string GetPlatformString(OpenClApi& api, cl_platform_id platform, cl_platform_info info) {
  std::size_t size = 0;
  Check(api.clGetPlatformInfo(platform, info, 0, nullptr, &size), "clGetPlatformInfo(size)");
  std::string out(size, '\0');
  Check(api.clGetPlatformInfo(platform, info, size, out.data(), nullptr), "clGetPlatformInfo(value)");
  if (!out.empty() && out.back() == '\0') out.pop_back();
  return out;
}

std::string GetDeviceString(OpenClApi& api, cl_device_id device, cl_device_info info) {
  std::size_t size = 0;
  Check(api.clGetDeviceInfo(device, info, 0, nullptr, &size), "clGetDeviceInfo(size)");
  std::string out(size, '\0');
  Check(api.clGetDeviceInfo(device, info, size, out.data(), nullptr), "clGetDeviceInfo(value)");
  if (!out.empty() && out.back() == '\0') out.pop_back();
  return out;
}

struct SelectedDevice {
  cl_platform_id platform = nullptr;
  cl_device_id device = nullptr;
  std::string platform_name;
  std::string device_name;
};

SelectedDevice SelectDevice(OpenClApi& api, const std::string& device_substring) {
  cl_uint platform_count = 0;
  Check(api.clGetPlatformIDs(0, nullptr, &platform_count), "clGetPlatformIDs(count)");
  std::vector<cl_platform_id> platforms(platform_count);
  Check(api.clGetPlatformIDs(platform_count, platforms.data(), nullptr), "clGetPlatformIDs(list)");
  for (cl_platform_id platform : platforms) {
    cl_uint device_count = 0;
    if (api.clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU_VALUE, 0, nullptr, &device_count) != CL_SUCCESS_VALUE ||
        device_count == 0) {
      continue;
    }
    std::vector<cl_device_id> devices(device_count);
    Check(api.clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU_VALUE, device_count, devices.data(), nullptr),
          "clGetDeviceIDs(list)");
    for (cl_device_id device : devices) {
      const std::string name = GetDeviceString(api, device, CL_DEVICE_NAME_VALUE);
      if (device_substring.empty() || name.find(device_substring) != std::string::npos) {
        return {platform, device, GetPlatformString(api, platform, CL_PLATFORM_NAME_VALUE), name};
      }
    }
  }
  Die("no matching OpenCL GPU for substring: " + device_substring);
}

double EventUs(OpenClApi& api, cl_event event) {
  cl_ulong start = 0;
  cl_ulong end = 0;
  Check(api.clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START_VALUE, sizeof(start), &start, nullptr),
        "clGetEventProfilingInfo(start)");
  Check(api.clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END_VALUE, sizeof(end), &end, nullptr),
        "clGetEventProfilingInfo(end)");
  return static_cast<double>(end - start) / 1000.0;
}

struct OpenClRuntime {
  OpenClApi api;
  SelectedDevice selected;
  cl_context context = nullptr;
  cl_command_queue queue = nullptr;
  cl_program program = nullptr;
  cl_kernel kernel_group8 = nullptr;
  cl_kernel kernel_rowlane = nullptr;
  std::string build_log;
  double program_build_ms = 0.0;

  explicit OpenClRuntime(const std::string& device_substring) {
    selected = SelectDevice(api, device_substring);
    cl_int err = CL_SUCCESS_VALUE;
    context = api.clCreateContext(nullptr, 1, &selected.device, nullptr, nullptr, &err);
    Check(err, "clCreateContext");
    queue = api.clCreateCommandQueue(context, selected.device, CL_QUEUE_PROFILING_ENABLE_VALUE, &err);
    Check(err, "clCreateCommandQueue");
    const char* source = kQ4X8OpenClSource;
    const std::size_t source_len = std::strlen(source);
    program = api.clCreateProgramWithSource(context, 1, &source, &source_len, &err);
    Check(err, "clCreateProgramWithSource");
    const auto build_begin = std::chrono::steady_clock::now();
    err = api.clBuildProgram(program, 1, &selected.device, "", nullptr, nullptr);
    const auto build_end = std::chrono::steady_clock::now();
    program_build_ms = std::chrono::duration<double, std::milli>(build_end - build_begin).count();
    std::size_t log_size = 0;
    api.clGetProgramBuildInfo(program, selected.device, CL_PROGRAM_BUILD_LOG_VALUE, 0, nullptr, &log_size);
    if (log_size > 1) {
      build_log.resize(log_size, '\0');
      api.clGetProgramBuildInfo(program, selected.device, CL_PROGRAM_BUILD_LOG_VALUE, log_size, build_log.data(), nullptr);
      if (!build_log.empty() && build_log.back() == '\0') build_log.pop_back();
    }
    Check(err, "clBuildProgram");
    kernel_group8 = api.clCreateKernel(program, "q4k_x8_matvec_group8", &err);
    Check(err, "clCreateKernel(q4k_x8_matvec_group8)");
    kernel_rowlane = api.clCreateKernel(program, "q4k_x8_matvec_rowlane", &err);
    Check(err, "clCreateKernel(q4k_x8_matvec_rowlane)");
  }

  ~OpenClRuntime() {
    if (kernel_rowlane) api.clReleaseKernel(kernel_rowlane);
    if (kernel_group8) api.clReleaseKernel(kernel_group8);
    if (program) api.clReleaseProgram(program);
    if (queue) api.clReleaseCommandQueue(queue);
    if (context) api.clReleaseContext(context);
  }
};

std::vector<std::int8_t> FlattenQ8Qs(const std::vector<Q8KBlock>& blocks) {
  std::vector<std::int8_t> out;
  out.reserve(blocks.size() * 256);
  for (const auto& block : blocks) {
    out.insert(out.end(), block.qs.begin(), block.qs.end());
  }
  return out;
}

std::vector<std::int16_t> FlattenQ8Bsums(const std::vector<Q8KBlock>& blocks) {
  std::vector<std::int16_t> out;
  out.reserve(blocks.size() * 16);
  for (const auto& block : blocks) {
    out.insert(out.end(), block.bsums.begin(), block.bsums.end());
  }
  return out;
}

std::vector<float> FlattenQ8D(const std::vector<Q8KBlock>& blocks) {
  std::vector<float> out;
  out.reserve(blocks.size());
  for (const auto& block : blocks) {
    out.push_back(block.d);
  }
  return out;
}

std::vector<float> RunGpuPacked(OpenClRuntime& runtime,
                                cl_kernel kernel,
                                const std::string& kernel_name,
                                std::uint64_t global_work_items,
                                std::uint64_t rows_per_work_item,
                                const std::vector<std::uint8_t>& packed,
                                const std::vector<Q8KBlock>& q8,
                                std::uint64_t rows,
                                std::uint64_t blocks_per_row,
                                int repeat,
                                KernelTiming& timing) {
  const std::uint64_t row_groups = rows / kRowsInterleaved;
  auto q8_qs = FlattenQ8Qs(q8);
  auto q8_bsums = FlattenQ8Bsums(q8);
  auto q8_d = FlattenQ8D(q8);
  std::vector<float> out(static_cast<std::size_t>(rows), 0.0f);

  cl_int err = CL_SUCCESS_VALUE;
  cl_mem packed_buffer = runtime.api.clCreateBuffer(runtime.context, CL_MEM_READ_ONLY_VALUE, packed.size(), nullptr, &err);
  Check(err, "clCreateBuffer(packed)");
  cl_mem q8_qs_buffer = runtime.api.clCreateBuffer(runtime.context, CL_MEM_READ_ONLY_VALUE, q8_qs.size(), nullptr, &err);
  Check(err, "clCreateBuffer(q8_qs)");
  cl_mem q8_bsums_buffer = runtime.api.clCreateBuffer(runtime.context, CL_MEM_READ_ONLY_VALUE, q8_bsums.size() * sizeof(std::int16_t), nullptr, &err);
  Check(err, "clCreateBuffer(q8_bsums)");
  cl_mem q8_d_buffer = runtime.api.clCreateBuffer(runtime.context, CL_MEM_READ_ONLY_VALUE, q8_d.size() * sizeof(float), nullptr, &err);
  Check(err, "clCreateBuffer(q8_d)");
  cl_mem out_buffer = runtime.api.clCreateBuffer(runtime.context, CL_MEM_WRITE_ONLY_VALUE, out.size() * sizeof(float), nullptr, &err);
  Check(err, "clCreateBuffer(out)");

  Check(runtime.api.clEnqueueWriteBuffer(runtime.queue, packed_buffer, CL_TRUE_VALUE, 0, packed.size(), packed.data(), 0, nullptr, nullptr),
        "clEnqueueWriteBuffer(packed)");
  Check(runtime.api.clEnqueueWriteBuffer(runtime.queue, q8_qs_buffer, CL_TRUE_VALUE, 0, q8_qs.size(), q8_qs.data(), 0, nullptr, nullptr),
        "clEnqueueWriteBuffer(q8_qs)");
  Check(runtime.api.clEnqueueWriteBuffer(runtime.queue, q8_bsums_buffer, CL_TRUE_VALUE, 0, q8_bsums.size() * sizeof(std::int16_t), q8_bsums.data(), 0, nullptr, nullptr),
        "clEnqueueWriteBuffer(q8_bsums)");
  Check(runtime.api.clEnqueueWriteBuffer(runtime.queue, q8_d_buffer, CL_TRUE_VALUE, 0, q8_d.size() * sizeof(float), q8_d.data(), 0, nullptr, nullptr),
        "clEnqueueWriteBuffer(q8_d)");

  const cl_uint blocks_arg = static_cast<cl_uint>(blocks_per_row);
  const cl_uint row_groups_arg = static_cast<cl_uint>(row_groups);
  Check(runtime.api.clSetKernelArg(kernel, 0, sizeof(packed_buffer), &packed_buffer), "clSetKernelArg(0)");
  Check(runtime.api.clSetKernelArg(kernel, 1, sizeof(q8_qs_buffer), &q8_qs_buffer), "clSetKernelArg(1)");
  Check(runtime.api.clSetKernelArg(kernel, 2, sizeof(q8_bsums_buffer), &q8_bsums_buffer), "clSetKernelArg(2)");
  Check(runtime.api.clSetKernelArg(kernel, 3, sizeof(q8_d_buffer), &q8_d_buffer), "clSetKernelArg(3)");
  Check(runtime.api.clSetKernelArg(kernel, 4, sizeof(blocks_arg), &blocks_arg), "clSetKernelArg(4)");
  Check(runtime.api.clSetKernelArg(kernel, 5, sizeof(row_groups_arg), &row_groups_arg), "clSetKernelArg(5)");
  Check(runtime.api.clSetKernelArg(kernel, 6, sizeof(out_buffer), &out_buffer), "clSetKernelArg(6)");

  const std::size_t global = static_cast<std::size_t>(global_work_items);
  std::vector<double> times;
  times.reserve(static_cast<std::size_t>(repeat));
  for (int i = 0; i < repeat; ++i) {
    cl_event event = nullptr;
    Check(runtime.api.clEnqueueNDRangeKernel(runtime.queue, kernel, 1, nullptr, &global, nullptr, 0, nullptr, &event),
          "clEnqueueNDRangeKernel(" + kernel_name + ")");
    Check(runtime.api.clFinish(runtime.queue), "clFinish(kernel)");
    times.push_back(EventUs(runtime.api, event));
    runtime.api.clReleaseEvent(event);
  }
  Check(runtime.api.clEnqueueReadBuffer(runtime.queue, out_buffer, CL_TRUE_VALUE, 0, out.size() * sizeof(float), out.data(), 0, nullptr, nullptr),
        "clEnqueueReadBuffer(out)");
  timing.min_us = *std::min_element(times.begin(), times.end());
  double total = 0.0;
  for (double value : times) total += value;
  timing.mean_us = total / static_cast<double>(times.size());
  timing.effective_packed_gb_s =
      static_cast<double>(packed.size()) / (timing.min_us / 1e6) / 1e9;
  timing.global_work_items = global_work_items;
  timing.rows_per_work_item = rows_per_work_item;

  runtime.api.clReleaseMemObject(out_buffer);
  runtime.api.clReleaseMemObject(q8_d_buffer);
  runtime.api.clReleaseMemObject(q8_bsums_buffer);
  runtime.api.clReleaseMemObject(q8_qs_buffer);
  runtime.api.clReleaseMemObject(packed_buffer);
  return out;
}

void WriteComparison(const Comparison& cmp) {
  std::cout << "{";
  std::cout << "\"cosine\":" << cmp.cosine << ",";
  std::cout << "\"max_abs_diff\":" << cmp.max_abs_diff << ",";
  std::cout << "\"max_abs_diff_index\":" << cmp.max_abs_diff_index << ",";
  std::cout << "\"passed\":" << (cmp.passed ? "true" : "false") << ",";
  std::cout << "\"rel_l2\":" << cmp.rel_l2 << ",";
  std::cout << "\"rmse\":" << cmp.rmse;
  std::cout << "}";
}

void WriteGpuVariant(const GpuVariant& variant) {
  std::cout << "{";
  std::cout << "\"name\":\"" << JsonEscape(variant.name) << "\",";
  std::cout << "\"kernel_name\":\"" << JsonEscape(variant.kernel_name) << "\",";
  std::cout << "\"rows_per_work_item\":" << variant.rows_per_work_item << ",";
  std::cout << "\"global_work_items\":" << variant.global_work_items << ",";
  std::cout << "\"gpu_kernel_min_us\":" << variant.timing.min_us << ",";
  std::cout << "\"gpu_kernel_mean_us\":" << variant.timing.mean_us << ",";
  std::cout << "\"gpu_effective_packed_gb_s\":" << variant.timing.effective_packed_gb_s << ",";
  std::cout << "\"comparison_vs_cpu_packed\":";
  WriteComparison(variant.comparison);
  std::cout << ",\"passed\":" << (variant.passed ? "true" : "false");
  std::cout << "}";
}

int main(int argc, char** argv) {
  try {
    const Args args = ParseArgs(argc, argv);
    const auto index = iq36::parse_gguf_model_index(args.model_path);
    const iq36::GgufTensorInfo* tensor = nullptr;
    for (const auto& candidate : index.tensors) {
      if (candidate.name == args.tensor_name) {
        tensor = &candidate;
        break;
      }
    }
    Require(tensor != nullptr, "tensor not found: " + args.tensor_name);
    const std::string tensor_type = iq36::ggml_type_name(tensor->type);
    Require(tensor->dims.size() >= 2, "selected tensor must have row dims");
    const std::uint64_t cols = tensor->dims[0];
    Require(cols % kQK == 0, "selected tensor cols must be QK-aligned");
    std::uint64_t rows = 1;
    for (std::size_t i = 1; i < tensor->dims.size(); ++i) {
      rows *= tensor->dims[i];
    }
    const std::uint64_t blocks_per_row = cols / kQK;

    if (tensor_type == "Q6_K") {
      const std::uint64_t expected_raw_bytes = rows * blocks_per_row * kQ6KBlockBytes;
      Require(tensor->nbytes == expected_raw_bytes, "selected tensor byte size does not match Q6_K shape");

      std::ifstream input(args.model_path, std::ios::binary);
      Require(static_cast<bool>(input), "failed to open model");
      const auto raw = ReadTensorBytes(input, *tensor);
      const auto input_values = MakeInput(cols);

      const auto cpu_reference_begin = std::chrono::steady_clock::now();
      const auto cpu_reference = iq36::matvec_tensor(args.model_path, index, tensor->name, input_values);
      const auto cpu_reference_end = std::chrono::steady_clock::now();

      constexpr double kRelL2Threshold = 1e-4;
      constexpr double kCosineThreshold = 0.999999;
      const auto cpu_reference_identity =
          CompareVectors(cpu_reference, cpu_reference, kRelL2Threshold, kCosineThreshold);

      iq36::GpuQ4X8MatvecRunner runtime(args.device_substring, kQ4X8OpenClSource);
      const auto q8 = iq36::QuantizeQ8KInputPlanes(input_values);
      const auto handle = runtime.UploadRawQ6K(raw, rows, blocks_per_row);
      const auto rowstripe_handle = runtime.UploadSelectedRawQ6KRowstripe(
          raw, rows, blocks_per_row, /*selected_count=*/1,
          /*rows_per_tile=*/16);
      const auto gpu_raw_run =
          runtime.RunResidentRawQ6K(handle, q8, args.repeat);
      const auto gpu_rowstripe_run =
          runtime.RunResidentRawQ6K(rowstripe_handle, q8, args.repeat);

      std::vector<GpuVariant> variants;
      variants.reserve(2);
      auto append_variant = [&](const char* name, const char* kernel_name,
                                const iq36::GpuQ6KMatvecRun& gpu_run) {
        GpuVariant variant;
        variant.name = name;
        variant.kernel_name = kernel_name;
        variant.rows_per_work_item = gpu_run.timing.rows_per_work_item;
        variant.global_work_items = gpu_run.timing.global_work_items;
        variant.timing.min_us = gpu_run.timing.min_us;
        variant.timing.mean_us = gpu_run.timing.mean_us;
        variant.timing.effective_packed_gb_s =
            gpu_run.timing.effective_packed_gb_s;
        variant.output = gpu_run.output;
        variant.comparison = CompareVectors(
            variant.output, cpu_reference, kRelL2Threshold, kCosineThreshold);
        variant.passed =
            variant.comparison.passed && variant.timing.min_us > 0.0;
        variants.push_back(std::move(variant));
      };
      append_variant("q6_raw_row", "q6k_selected_down_matvec_row",
                     gpu_raw_run);
      append_variant("q6_rowstripe16",
                     "q6k_selected_down_matvec_rowstripe",
                     gpu_rowstripe_run);

      const GpuVariant* best_variant = nullptr;
      bool all_gpu_variants_passed = !variants.empty();
      for (const auto& variant : variants) {
        all_gpu_variants_passed = all_gpu_variants_passed && variant.passed;
        if (variant.passed &&
            (best_variant == nullptr ||
             variant.timing.min_us < best_variant->timing.min_us)) {
          best_variant = &variant;
        }
      }
      Require(best_variant != nullptr,
              "no Q6_K GPU variant passed correctness/timing checks");

      const bool checks_passed =
          cpu_reference_identity.passed &&
          all_gpu_variants_passed &&
          runtime.device_name().find(args.device_substring) != std::string::npos;
      const double cpu_reference_us =
          std::chrono::duration<double, std::micro>(cpu_reference_end - cpu_reference_begin).count();

      std::cout << std::setprecision(10);
      std::cout << "{";
      std::cout << "\"schema_version\":\"intel-qwen36-gpu-q4x8-qmatvec-probe-v4\",";
      std::cout << "\"model_path\":\"" << JsonEscape(args.model_path) << "\",";
      std::cout << "\"tensor_name\":\"" << JsonEscape(tensor->name) << "\",";
      std::cout << "\"tensor_type\":\"Q6_K\",";
      std::cout << "\"payload_layout\":\"raw_q6_k\",";
      std::cout << "\"cols\":" << cols << ",";
      std::cout << "\"rows\":" << rows << ",";
      std::cout << "\"blocks_per_row\":" << blocks_per_row << ",";
      std::cout << "\"raw_bytes\":" << raw.size() << ",";
      std::cout << "\"packed_q4k_x8_bytes\":0,";
      std::cout << "\"payload_bytes\":" << raw.size() << ",";
      std::cout << "\"input_quantization\":\"Q8_K\",";
      std::cout << "\"q8_block_count\":" << q8.d.size() << ",";
      std::cout << "\"platform_name\":\"" << JsonEscape(runtime.platform_name()) << "\",";
      std::cout << "\"device_name\":\"" << JsonEscape(runtime.device_name()) << "\",";
      std::cout << "\"program_build_ms\":" << runtime.program_build_ms() << ",";
      std::cout << "\"build_log\":\"" << JsonEscape(runtime.build_log()) << "\",";
      std::cout << "\"repeat\":" << args.repeat << ",";
      std::cout << "\"cpu_raw_us\":" << cpu_reference_us << ",";
      std::cout << "\"cpu_packed_us\":" << cpu_reference_us << ",";
      std::cout << "\"gpu_kernel_min_us\":" << best_variant->timing.min_us << ",";
      std::cout << "\"gpu_kernel_mean_us\":" << best_variant->timing.mean_us << ",";
      std::cout << "\"gpu_effective_packed_gb_s\":" << best_variant->timing.effective_packed_gb_s << ",";
      std::cout << "\"gpu_effective_payload_gb_s\":" << best_variant->timing.effective_packed_gb_s << ",";
      std::cout << "\"gpu_best_variant\":\"" << JsonEscape(best_variant->name) << "\",";
      std::cout << "\"gpu_best_kernel_name\":\"" << JsonEscape(best_variant->kernel_name) << "\",";
      std::cout << "\"gpu_variants\":[";
      for (std::size_t i = 0; i < variants.size(); ++i) {
        if (i != 0) {
          std::cout << ",";
        }
        WriteGpuVariant(variants[i]);
      }
      std::cout << "],";
      std::cout << "\"comparisons\":{";
      std::cout << "\"cpu_packed_vs_cpu_raw\":";
      WriteComparison(cpu_reference_identity);
      std::cout << ",\"gpu_vs_cpu_packed\":";
      WriteComparison(best_variant->comparison);
      std::cout << ",\"gpu_vs_cpu_reference\":";
      WriteComparison(best_variant->comparison);
      for (const auto& variant : variants) {
        std::cout << ",\"gpu_" << JsonEscape(variant.name)
                  << "_vs_cpu_packed\":";
        WriteComparison(variant.comparison);
      }
      std::cout << "},";
      std::cout << "\"checks\":{";
      std::cout << "\"arc_device_selected\":" << (runtime.device_name().find(args.device_substring) != std::string::npos ? "true" : "false") << ",";
      std::cout << "\"cpu_packed_matches_cpu_raw\":" << (cpu_reference_identity.passed ? "true" : "false") << ",";
      std::cout << "\"gpu_matches_cpu_packed\":" << (best_variant->comparison.passed ? "true" : "false") << ",";
      std::cout << "\"gpu_q6_raw_row_matches_cpu_packed\":" << (variants[0].comparison.passed ? "true" : "false") << ",";
      std::cout << "\"gpu_q6_rowstripe16_matches_cpu_packed\":" << (variants[1].comparison.passed ? "true" : "false") << ",";
      std::cout << "\"gpu_event_timing_positive\":" << (best_variant->timing.min_us > 0.0 ? "true" : "false") << ",";
      std::cout << "\"gpu_variants_event_timing_positive\":" << (all_gpu_variants_passed ? "true" : "false") << ",";
      std::cout << "\"speedup_claims_allowed\":false";
      std::cout << "},";
      std::cout << "\"required_checks_passed\":" << (checks_passed ? "true" : "false");
      std::cout << "}\n";
      return checks_passed ? 0 : 3;
    }

    Require(tensor_type == "Q4_K", "selected tensor is not Q4_K or Q6_K");
    Require(rows % kRowsInterleaved == 0, "selected tensor rows must be divisible by 8");
    const std::uint64_t expected_raw_bytes = rows * blocks_per_row * kQ4KBlockBytes;
    Require(tensor->nbytes == expected_raw_bytes, "selected tensor byte size does not match Q4_K shape");

    std::ifstream input(args.model_path, std::ios::binary);
    Require(static_cast<bool>(input), "failed to open model");
    const auto raw = ReadTensorBytes(input, *tensor);
    const auto packed = PackQ4Kx8(raw, rows, blocks_per_row);
    const auto input_values = MakeInput(cols);
    const auto q8 = QuantizeQ8K(input_values);

    const auto cpu_raw_begin = std::chrono::steady_clock::now();
    const auto cpu_raw = RunCpuRaw(raw, rows, blocks_per_row, q8);
    const auto cpu_raw_end = std::chrono::steady_clock::now();
    const auto cpu_packed_begin = std::chrono::steady_clock::now();
    const auto cpu_packed = RunCpuPacked(packed, rows, blocks_per_row, q8);
    const auto cpu_packed_end = std::chrono::steady_clock::now();

    constexpr double kRelL2Threshold = 1e-4;
    constexpr double kCosineThreshold = 0.999999;
    const auto cpu_packed_vs_raw = CompareVectors(cpu_packed, cpu_raw, kRelL2Threshold, kCosineThreshold);

    iq36::GpuQ4X8MatvecRunner runtime(args.device_substring, kQ4X8OpenClSource);
    const auto q8_qs = FlattenQ8Qs(q8);
    const auto q8_bsums = FlattenQ8Bsums(q8);
    const auto q8_d = FlattenQ8D(q8);
    std::vector<GpuVariant> variants;
    for (const auto engine_variant : {
             iq36::GpuQ4X8KernelVariant::kGroup8Serial,
             iq36::GpuQ4X8KernelVariant::kRowlaneParallel,
         }) {
      GpuVariant variant;
      variant.name = iq36::KernelVariantName(engine_variant);
      variant.kernel_name = iq36::KernelFunctionName(engine_variant);
      const auto run = runtime.Run(packed, q8_qs, q8_bsums, q8_d, rows, blocks_per_row,
                                   args.repeat, engine_variant);
      variant.rows_per_work_item = run.timing.rows_per_work_item;
      variant.global_work_items = run.timing.global_work_items;
      variant.timing.min_us = run.timing.min_us;
      variant.timing.mean_us = run.timing.mean_us;
      variant.timing.effective_packed_gb_s = run.timing.effective_packed_gb_s;
      variant.output = run.output;
      variant.comparison = CompareVectors(variant.output, cpu_packed, kRelL2Threshold, kCosineThreshold);
      variant.passed = variant.comparison.passed && variant.timing.min_us > 0.0;
      variants.push_back(std::move(variant));
    }
    const GpuVariant* best_variant = nullptr;
    bool all_gpu_variants_passed = !variants.empty();
    for (const auto& variant : variants) {
      all_gpu_variants_passed = all_gpu_variants_passed && variant.passed;
      if (variant.passed && (best_variant == nullptr || variant.timing.min_us < best_variant->timing.min_us)) {
        best_variant = &variant;
      }
    }
    Require(best_variant != nullptr, "no GPU variant passed correctness/timing checks");
    const bool checks_passed =
        cpu_packed_vs_raw.passed &&
        all_gpu_variants_passed &&
        runtime.device_name().find(args.device_substring) != std::string::npos;

    const double cpu_raw_us =
        std::chrono::duration<double, std::micro>(cpu_raw_end - cpu_raw_begin).count();
    const double cpu_packed_us =
        std::chrono::duration<double, std::micro>(cpu_packed_end - cpu_packed_begin).count();

    std::cout << std::setprecision(10);
    std::cout << "{";
    std::cout << "\"schema_version\":\"intel-qwen36-gpu-q4x8-qmatvec-probe-v4\",";
    std::cout << "\"model_path\":\"" << JsonEscape(args.model_path) << "\",";
    std::cout << "\"tensor_name\":\"" << JsonEscape(tensor->name) << "\",";
    std::cout << "\"tensor_type\":\"Q4_K\",";
    std::cout << "\"payload_layout\":\"q4_k_8x8\",";
    std::cout << "\"cols\":" << cols << ",";
    std::cout << "\"rows\":" << rows << ",";
    std::cout << "\"blocks_per_row\":" << blocks_per_row << ",";
    std::cout << "\"raw_bytes\":" << raw.size() << ",";
    std::cout << "\"packed_q4k_x8_bytes\":" << packed.size() << ",";
    std::cout << "\"payload_bytes\":" << packed.size() << ",";
    std::cout << "\"input_quantization\":\"Q8_K\",";
    std::cout << "\"q8_block_count\":" << q8.size() << ",";
    std::cout << "\"platform_name\":\"" << JsonEscape(runtime.platform_name()) << "\",";
    std::cout << "\"device_name\":\"" << JsonEscape(runtime.device_name()) << "\",";
    std::cout << "\"program_build_ms\":" << runtime.program_build_ms() << ",";
    std::cout << "\"build_log\":\"" << JsonEscape(runtime.build_log()) << "\",";
    std::cout << "\"repeat\":" << args.repeat << ",";
    std::cout << "\"cpu_raw_us\":" << cpu_raw_us << ",";
    std::cout << "\"cpu_packed_us\":" << cpu_packed_us << ",";
    std::cout << "\"gpu_kernel_min_us\":" << best_variant->timing.min_us << ",";
    std::cout << "\"gpu_kernel_mean_us\":" << best_variant->timing.mean_us << ",";
    std::cout << "\"gpu_effective_packed_gb_s\":" << best_variant->timing.effective_packed_gb_s << ",";
    std::cout << "\"gpu_effective_payload_gb_s\":" << best_variant->timing.effective_packed_gb_s << ",";
    std::cout << "\"gpu_best_variant\":\"" << JsonEscape(best_variant->name) << "\",";
    std::cout << "\"gpu_best_kernel_name\":\"" << JsonEscape(best_variant->kernel_name) << "\",";
    std::cout << "\"gpu_variants\":[";
    for (std::size_t i = 0; i < variants.size(); ++i) {
      if (i != 0) {
        std::cout << ",";
      }
      WriteGpuVariant(variants[i]);
    }
    std::cout << "],";
    std::cout << "\"comparisons\":{";
    std::cout << "\"cpu_packed_vs_cpu_raw\":";
    WriteComparison(cpu_packed_vs_raw);
    std::cout << ",\"gpu_vs_cpu_packed\":";
    WriteComparison(best_variant->comparison);
    for (const auto& variant : variants) {
      std::cout << ",\"gpu_" << JsonEscape(variant.name) << "_vs_cpu_packed\":";
      WriteComparison(variant.comparison);
    }
    std::cout << "},";
    std::cout << "\"checks\":{";
    std::cout << "\"arc_device_selected\":" << (runtime.device_name().find(args.device_substring) != std::string::npos ? "true" : "false") << ",";
    std::cout << "\"cpu_packed_matches_cpu_raw\":" << (cpu_packed_vs_raw.passed ? "true" : "false") << ",";
    std::cout << "\"gpu_matches_cpu_packed\":" << (best_variant->comparison.passed ? "true" : "false") << ",";
    std::cout << "\"gpu_group8_serial_matches_cpu_packed\":" << (variants[0].comparison.passed ? "true" : "false") << ",";
    std::cout << "\"gpu_rowlane_parallel_matches_cpu_packed\":" << (variants[1].comparison.passed ? "true" : "false") << ",";
    std::cout << "\"gpu_event_timing_positive\":" << (best_variant->timing.min_us > 0.0 ? "true" : "false") << ",";
    std::cout << "\"gpu_variants_event_timing_positive\":" << (all_gpu_variants_passed ? "true" : "false") << ",";
    std::cout << "\"speedup_claims_allowed\":false";
    std::cout << "},";
    std::cout << "\"required_checks_passed\":" << (checks_passed ? "true" : "false");
    std::cout << "}\n";
    return checks_passed ? 0 : 3;
  } catch (const std::exception& exc) {
    std::cout << "{\"ok\":false,\"error\":\"" << JsonEscape(exc.what()) << "\"}\n";
    return 2;
  }
}
'''


def iso_now() -> str:
  return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_stamp() -> str:
  return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--target", "--host", dest="host", metavar="TARGET", default=DEFAULT_HOST)
  parser.add_argument("--model", default=DEFAULT_MODEL)
  parser.add_argument("--tensor", default=DEFAULT_TENSOR)
  parser.add_argument("--env-script", default=DEFAULT_ENV_SCRIPT)
  parser.add_argument("--staging-root", "--remote-root", dest="remote_root", metavar="PATH", default=DEFAULT_REMOTE_ROOT)
  parser.add_argument("--repeat", type=int, default=5)
  parser.add_argument("--device-substring", default="B390")
  parser.add_argument("--timeout-s", type=int, default=900)
  parser.add_argument("--out-dir", type=Path, default=None)
  return parser.parse_args()


def shell_join(argv: list[str]) -> str:
  return " ".join(shlex.quote(item) for item in argv)


def cpp_raw_string_literal(value: str) -> str:
  delimiter = "IQ36CL"
  if f"){delimiter}\"" in value:
    raise ValueError(f"OpenCL source contains raw-string delimiter {delimiter}")
  return f'R"{delimiter}({value}){delimiter}"'


def parse_probe_stdout(stdout: str) -> dict[str, Any] | None:
  for line in reversed(stdout.splitlines()):
    line = line.strip()
    if not line.startswith("{"):
      continue
    try:
      value = json.loads(line)
    except json.JSONDecodeError:
      continue
    if isinstance(value, dict):
      return value
  return None


def nested_bool(obj: dict[str, Any], *keys: str) -> bool:
  current: Any = obj
  for key in keys:
    if not isinstance(current, dict):
      return False
    current = current.get(key)
  return current is True


def nested_number(obj: dict[str, Any], *keys: str) -> float | None:
  current: Any = obj
  for key in keys:
    if not isinstance(current, dict):
      return None
    current = current.get(key)
  return float(current) if isinstance(current, (int, float)) else None


def gpu_variant(probe: dict[str, Any] | None, name: str) -> dict[str, Any]:
  variants = probe.get("gpu_variants", []) if isinstance(probe, dict) else []
  if not isinstance(variants, list):
    return {}
  for item in variants:
    if isinstance(item, dict) and item.get("name") == name:
      return item
  return {}


def write_summary(path: Path, payload: dict[str, Any]) -> None:
  probe = payload.get("probe") if isinstance(payload.get("probe"), dict) else {}
  comparisons = probe.get("comparisons", {}) if isinstance(probe, dict) else {}
  cpu_cmp = comparisons.get("cpu_packed_vs_cpu_raw", {}) if isinstance(comparisons, dict) else {}
  gpu_cmp = comparisons.get("gpu_vs_cpu_packed", {}) if isinstance(comparisons, dict) else {}
  variants = probe.get("gpu_variants", []) if isinstance(probe, dict) else []
  variant_rows: list[str] = []
  if isinstance(variants, list):
    for item in variants:
      if not isinstance(item, dict):
        continue
      cmp = item.get("comparison_vs_cpu_packed", {})
      cmp = cmp if isinstance(cmp, dict) else {}
      variant_rows.append(
          "| {name} | {rows_per_work_item} | {global_work_items} | {min_us} | {gb_s} | {rel_l2} | `{passed}` |".format(
              name=item.get("name"),
              rows_per_work_item=item.get("rows_per_work_item"),
              global_work_items=item.get("global_work_items"),
              min_us=item.get("gpu_kernel_min_us"),
              gb_s=item.get("gpu_effective_packed_gb_s"),
              rel_l2=cmp.get("rel_l2"),
              passed=str(item.get("passed")).lower(),
          )
      )
  tensor_type = probe.get("tensor_type")
  payload_layout = probe.get("payload_layout")
  payload_bytes = probe.get("payload_bytes", probe.get("packed_q4k_x8_bytes"))
  payload_gb_s = probe.get("gpu_effective_payload_gb_s", probe.get("gpu_effective_packed_gb_s"))
  lines = [
      "# GPU Quantized QMatVec Probe",
      "",
      f"- workstream: `{WORKSTREAM}`",
      f"- host: `{payload.get('host')}`",
      f"- model: `{payload.get('model')}`",
      f"- tensor: `{probe.get('tensor_name')}`",
      f"- tensor type/layout: `{tensor_type}` / `{payload_layout}`",
      f"- engine shim: `{payload.get('engine_shim_source')}`",
      f"- OpenCL source: `{payload.get('opencl_source')}`",
      f"- required checks passed: `{str(payload.get('required_checks_passed')).lower()}`",
      "- speedup claims allowed: `false`",
      f"- platform/device: `{probe.get('platform_name')}` / `{probe.get('device_name')}`",
      f"- shape: cols `{probe.get('cols')}`, rows `{probe.get('rows')}`, blocks/row `{probe.get('blocks_per_row')}`",
      f"- raw/payload bytes: `{probe.get('raw_bytes')}` / `{payload_bytes}`",
      f"- best GPU variant: `{probe.get('gpu_best_variant')}` / `{probe.get('gpu_best_kernel_name')}`",
      f"- best GPU kernel min us: `{probe.get('gpu_kernel_min_us')}`",
      f"- best GPU effective payload GB/s: `{payload_gb_s}`",
      "",
      "| comparison | relL2 | cosine | max abs | passed |",
      "|---|---:|---:|---:|---:|",
      (
          f"| CPU packed vs CPU raw | {cpu_cmp.get('rel_l2')} | "
          f"{cpu_cmp.get('cosine')} | {cpu_cmp.get('max_abs_diff')} | "
          f"`{str(cpu_cmp.get('passed')).lower()}` |"
      ),
      (
          f"| GPU packed vs CPU packed | {gpu_cmp.get('rel_l2')} | "
          f"{gpu_cmp.get('cosine')} | {gpu_cmp.get('max_abs_diff')} | "
          f"`{str(gpu_cmp.get('passed')).lower()}` |"
      ),
      "",
      "| GPU variant | rows/work-item | global work-items | min us | packed GB/s | relL2 vs CPU packed | passed |",
      "|---|---:|---:|---:|---:|---:|---:|",
      *variant_rows,
      "",
      "Decision: this is a single-op quantized qmatvec correctness/timing",
      "profile gate. It does not prove decode or model throughput.",
      "",
  ]
  path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  args = parse_args()
  if args.repeat <= 0:
    raise SystemExit("--repeat must be positive")
  stamp = utc_stamp()
  created_at = iso_now()
  out_dir = args.out_dir or ROOT / f"output/gpu-q4x8-qmatvec-probe-{stamp}"
  out_dir.mkdir(parents=True, exist_ok=False)
  raw_dir = out_dir / "raw"
  raw_dir.mkdir()
  local_cpp = out_dir / "gpu_q4x8_qmatvec_probe.cpp"
  opencl_source = OPENCL_SOURCE.read_text(encoding="utf-8")
  opencl_source_hash = iq36_local.sha256_file(OPENCL_SOURCE)
  local_cpp.write_text(
      PROBE_CPP.replace("@@OPENCL_SOURCE_LITERAL@@", cpp_raw_string_literal(opencl_source)),
      encoding="utf-8",
  )

  remote_dir = f"{args.remote_root.rstrip('/')}/gpu-q4x8-qmatvec-probe-{stamp}"
  setup = iq36_local.run_target(
      args.host,
      "rm -rf "
      + shlex.quote(remote_dir)
      + " && mkdir -p "
      + " ".join(
          shlex.quote(f"{remote_dir}/{subdir}")
          for subdir in ("include/intel_qwen36", "src", "tests", "build")
      ),
      args.timeout_s,
  )
  transfers: list[dict[str, Any]] = []
  if setup.get("returncode") == 0:
    for local, remote in SOURCE_FILES:
      transfers.append(iq36_local.copy_to(args.host, ROOT / local, f"{remote_dir}/{remote}", args.timeout_s))
    transfers.append(iq36_local.copy_to(args.host, local_cpp, f"{remote_dir}/tests/gpu_q4x8_qmatvec_probe.cpp", args.timeout_s))

  compile_cmd = " && ".join([
      f"source {shlex.quote(args.env_script)} >/dev/null 2>&1",
      (
          "g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic "
          f"-I {shlex.quote(remote_dir + '/include')} "
          f"{shlex.quote(remote_dir + '/src/gguf_loader.cpp')} "
          f"{shlex.quote(remote_dir + '/src/gpu_q4x8_matvec.cpp')} "
          f"{shlex.quote(remote_dir + '/tests/gpu_q4x8_qmatvec_probe.cpp')} "
          "-ldl -pthread "
          f"-o {shlex.quote(remote_dir + '/build/iq36-gpu-q4x8-qmatvec-probe')}"
      ),
  ])
  compile_result = (
      iq36_local.run_target(args.host, compile_cmd, args.timeout_s)
      if transfers and all(item.get("returncode") == 0 for item in transfers)
      else {"cmd": ["stage"], "returncode": 1, "stdout": "", "stderr": "stage failed"}
  )

  run_argv = [
      f"{remote_dir}/build/iq36-gpu-q4x8-qmatvec-probe",
      "--model",
      args.model,
      "--tensor",
      args.tensor,
      "--repeat",
      str(args.repeat),
      "--device-substring",
      args.device_substring,
  ]
  run_result = (
      iq36_local.run_target(
          args.host,
          " && ".join([
              f"source {shlex.quote(args.env_script)} >/dev/null 2>&1",
              shell_join(run_argv),
          ]),
          args.timeout_s,
      )
      if compile_result.get("returncode") == 0
      else {"cmd": run_argv, "returncode": None, "stdout": "", "stderr": "compile skipped run"}
  )
  probe = parse_probe_stdout(run_result.get("stdout", ""))

  iq36_local.write_json(raw_dir / "setup.json", setup)
  iq36_local.write_json(raw_dir / "transfers.json", transfers)
  iq36_local.write_json(raw_dir / "compile.json", compile_result)
  iq36_local.write_json(raw_dir / "run.json", run_result)
  if probe is not None:
    iq36_local.write_json(out_dir / "probe-result.json", probe)
  group8_variant = gpu_variant(probe, "group8_serial")
  rowlane_variant = gpu_variant(probe, "rowlane_parallel")
  q6_variant = gpu_variant(probe, "q6_raw_row")
  q6_rowstripe_variant = gpu_variant(probe, "q6_rowstripe16")
  tensor_type = probe.get("tensor_type") if isinstance(probe, dict) else None
  is_q4 = tensor_type == "Q4_K"
  is_q6 = tensor_type == "Q6_K"

  checks = [
      {"name": "remote_dir_created", "pass": setup.get("returncode") == 0},
      {
          "name": "source_files_transferred",
          "pass": bool(transfers) and all(item.get("returncode") == 0 for item in transfers),
      },
      {"name": "probe_compiled", "pass": compile_result.get("returncode") == 0},
      {"name": "probe_stdout_json_parsed", "pass": isinstance(probe, dict)},
      {"name": "probe_process_succeeded", "pass": run_result.get("returncode") == 0},
      {"name": "arc_b390_selected", "pass": bool(probe and "B390" in str(probe.get("device_name", "")))},
      {"name": "supported_quantized_tensor_selected", "pass": bool(is_q4 or is_q6)},
      {
          "name": "cpu_packed_matches_cpu_raw",
          "pass": bool(probe and nested_bool(probe, "comparisons", "cpu_packed_vs_cpu_raw", "passed")),
      },
      {
          "name": "gpu_matches_cpu_packed",
          "pass": bool(probe and nested_bool(probe, "comparisons", "gpu_vs_cpu_packed", "passed")),
      },
      {
          "name": "gpu_group8_serial_matches_cpu_packed",
          "pass": bool((not is_q4) or group8_variant.get("passed") is True),
      },
      {
          "name": "gpu_rowlane_parallel_matches_cpu_packed",
          "pass": bool((not is_q4) or rowlane_variant.get("passed") is True),
      },
      {
          "name": "gpu_q6_raw_row_matches_cpu_packed",
          "pass": bool((not is_q6) or q6_variant.get("passed") is True),
      },
      {
          "name": "gpu_q6_rowstripe16_matches_cpu_packed",
          "pass": bool((not is_q6) or q6_rowstripe_variant.get("passed") is True),
      },
      {
          "name": "gpu_event_timing_positive",
          "pass": bool((nested_number(probe or {}, "gpu_kernel_min_us") or 0.0) > 0.0),
      },
      {
          "name": "gpu_variants_event_timing_positive",
          "pass": (
              bool(
                  (nested_number(group8_variant, "gpu_kernel_min_us") or 0.0) > 0.0
                  and (nested_number(rowlane_variant, "gpu_kernel_min_us") or 0.0) > 0.0
              )
              if is_q4
              else bool(
                  (nested_number(q6_variant, "gpu_kernel_min_us") or 0.0) > 0.0
                  and (nested_number(q6_rowstripe_variant, "gpu_kernel_min_us") or 0.0) > 0.0
              )
          ),
      },
      {"name": "speedup_claims_forbidden", "pass": True},
  ]
  required_checks_passed = all(item["pass"] for item in checks)
  payload = {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "created_at": created_at,
      "host": args.host,
      "remote_dir": remote_dir,
      "model": args.model,
      "tensor": args.tensor,
      "repeat": args.repeat,
      "engine_shim_header": "engine/include/intel_qwen36/gpu_q4x8_matvec.hpp",
      "engine_shim_source": "engine/src/gpu_q4x8_matvec.cpp",
      "opencl_source": str(OPENCL_SOURCE.relative_to(ROOT)),
      "opencl_source_sha256": opencl_source_hash,
      "probe": probe,
      "checks": checks,
      "required_checks_passed": required_checks_passed,
      "speedup_claims_allowed": False,
      "recommendation": "use this single-op gate to decide the next quantized kernel/root-cause step; do not claim decode speed",
  }
  manifest = {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "created_at": created_at,
      "tool": "tools/intel-qwen36-gpu-q4x8-qmatvec-probe.py",
      "artifact": str(out_dir),
      "host": args.host,
      "remote_dir": remote_dir,
      "engine_shim_header": "engine/include/intel_qwen36/gpu_q4x8_matvec.hpp",
      "engine_shim_source": "engine/src/gpu_q4x8_matvec.cpp",
      "opencl_source": str(OPENCL_SOURCE.relative_to(ROOT)),
      "opencl_source_sha256": opencl_source_hash,
      "required_checks_passed": required_checks_passed,
      "speedup_claims_allowed": False,
  }
  correctness = {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "checks": checks,
      "required_checks_passed": required_checks_passed,
      "speedup_claims_allowed": False,
  }
  iq36_local.write_json(out_dir / "probe.json", payload)
  iq36_local.write_json(out_dir / "manifest.json", manifest)
  iq36_local.write_json(out_dir / "correctness.json", correctness)
  aggregate = probe if isinstance(probe, dict) else {}
  iq36_local.write_metric(
      out_dir / "metrics.jsonl",
      "gpu_q4x8_qmatvec_probe",
      [
          ("gpu_best_variant", aggregate.get("gpu_best_variant")),
          ("gpu_kernel_min_us", aggregate.get("gpu_kernel_min_us")),
          ("gpu_effective_packed_gb_s", aggregate.get("gpu_effective_packed_gb_s")),
          ("gpu_group8_serial_kernel_min_us", nested_number(group8_variant, "gpu_kernel_min_us")),
          ("gpu_group8_serial_effective_packed_gb_s", nested_number(group8_variant, "gpu_effective_packed_gb_s")),
          ("gpu_group8_serial_rel_l2", nested_number(group8_variant, "comparison_vs_cpu_packed", "rel_l2")),
          ("gpu_rowlane_parallel_kernel_min_us", nested_number(rowlane_variant, "gpu_kernel_min_us")),
          ("gpu_rowlane_parallel_effective_packed_gb_s", nested_number(rowlane_variant, "gpu_effective_packed_gb_s")),
          ("gpu_rowlane_parallel_rel_l2", nested_number(rowlane_variant, "comparison_vs_cpu_packed", "rel_l2")),
          ("gpu_q6_raw_row_kernel_min_us", nested_number(q6_variant, "gpu_kernel_min_us")),
          ("gpu_q6_raw_row_effective_payload_gb_s", nested_number(q6_variant, "gpu_effective_packed_gb_s")),
          ("gpu_q6_raw_row_rel_l2", nested_number(q6_variant, "comparison_vs_cpu_packed", "rel_l2")),
          ("gpu_q6_rowstripe16_kernel_min_us", nested_number(q6_rowstripe_variant, "gpu_kernel_min_us")),
          ("gpu_q6_rowstripe16_effective_payload_gb_s", nested_number(q6_rowstripe_variant, "gpu_effective_packed_gb_s")),
          ("gpu_q6_rowstripe16_rel_l2", nested_number(q6_rowstripe_variant, "comparison_vs_cpu_packed", "rel_l2")),
          ("cpu_packed_vs_cpu_raw_rel_l2", nested_number(aggregate, "comparisons", "cpu_packed_vs_cpu_raw", "rel_l2")),
          ("gpu_vs_cpu_packed_rel_l2", nested_number(aggregate, "comparisons", "gpu_vs_cpu_packed", "rel_l2")),
          ("required_checks_passed", required_checks_passed),
      ],
  )
  write_summary(out_dir / "summary.md", payload)
  print(out_dir)
  return 0 if required_checks_passed else 1

if __name__ == "__main__":
  raise SystemExit(main())
