#!/usr/bin/env python3
"""Run a GPU raw/plane/q4-x8 tensor stream probe on Arc B390.

This is the next GPU bring-up gate after the minimal OpenCL runtime probe. It
uses the project GGUF loader to select representative Q4_K/Q6_K tensors from the
locked model, repacks them into the existing plane layouts, and streams raw vs
repacked byte layouts through an OpenCL checksum kernel. For Q4_K tensors it
also emits a llama.cpp-style q4_K_8x8 packed stream. It is not a model throughput
benchmark and does not allow speedup claims.
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
SCHEMA_VERSION = "intel-qwen36-gpu-repack-stream-probe-v1"
DEFAULT_HOST = "local"
DEFAULT_MODEL = "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf"
DEFAULT_ENV_SCRIPT = "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh"
DEFAULT_REMOTE_ROOT = "/home/intel/intel-qwen36-gpu"
DEFAULT_LANES = ["attn_qkv.weight:Q4_K", "ffn_down_exps.weight:Q6_K"]
SOURCE_FILES = [
    ("engine/include/intel_qwen36/gguf_loader.hpp", "include/intel_qwen36/gguf_loader.hpp"),
    ("engine/src/gguf_loader.cpp", "src/gguf_loader.cpp"),
]


PROBE_CPP = r'''
#include "intel_qwen36/gguf_loader.hpp"

#include <algorithm>
#include <array>
#include <chrono>
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
constexpr cl_device_type CL_DEVICE_TYPE_GPU_VALUE = 1ULL << 2;
constexpr cl_mem_flags CL_MEM_READ_ONLY_VALUE = 1ULL << 2;
constexpr cl_mem_flags CL_MEM_WRITE_ONLY_VALUE = 1ULL << 1;
constexpr cl_command_queue_properties CL_QUEUE_PROFILING_ENABLE_VALUE = 1ULL << 1;
constexpr cl_platform_info CL_PLATFORM_NAME_VALUE = 0x0902;
constexpr cl_device_info CL_DEVICE_NAME_VALUE = 0x102B;
constexpr cl_program_build_info CL_PROGRAM_BUILD_LOG_VALUE = 0x1183;
constexpr cl_profiling_info CL_PROFILING_COMMAND_START_VALUE = 0x1282;
constexpr cl_profiling_info CL_PROFILING_COMMAND_END_VALUE = 0x1283;

struct LaneSpec {
  std::string suffix;
  std::string type_name;
};

struct Args {
  std::string model_path;
  std::vector<LaneSpec> lanes;
  int max_tensors_per_lane = 1;
  int repeat = 5;
  std::uint64_t work_items = 4096;
  std::string device_substring = "B390";
};

struct RepackedTensor {
  std::string layout;
  std::uint64_t block_count = 0;
  std::vector<std::uint8_t> q4;
  std::vector<std::uint8_t> ql;
  std::vector<std::uint8_t> qh;
  std::vector<std::uint8_t> scales;
  std::vector<std::uint8_t> mins;
  std::vector<std::uint8_t> d_values;
  std::vector<std::uint8_t> dmin_values;
};

struct CpuChecksum {
  std::uint64_t sum = 0;
  std::uint64_t weighted = 0;
};

struct StreamResult {
  std::string label;
  std::uint64_t byte_count = 0;
  CpuChecksum cpu;
  CpuChecksum gpu;
  bool checksum_match = false;
  double write_us = 0.0;
  double read_us = 0.0;
  double kernel_min_us = 0.0;
  double kernel_mean_us = 0.0;
  double kernel_min_gb_s = 0.0;
  double kernel_mean_gb_s = 0.0;
  double host_to_device_gb_s = 0.0;
};

struct ProbeRow {
  std::uint64_t absolute_offset = 0;
  std::uint64_t block_count = 0;
  std::uint64_t blocks_per_row = 0;
  std::vector<std::uint64_t> dims;
  int layer_index = -1;
  std::string layout;
  std::string name;
  bool packed_q4k_x8_available = false;
  std::string packed_q4k_x8_layout;
  std::uint64_t packed_q4k_x8_bytes = 0;
  double packed_q4k_x8_overhead_ratio = 0.0;
  double packed_q4k_x8_repack_gb_s = 0.0;
  std::uint64_t packed_q4k_x8_repack_ns = 0;
  std::uint64_t raw_bytes = 0;
  std::uint64_t repack_ns = 0;
  double repack_gb_s = 0.0;
  std::uint64_t repacked_bytes = 0;
  double repacked_overhead_ratio = 0.0;
  std::string selected_by_lane;
  std::string suffix;
  std::string type_name;
  std::uint64_t q4_bytes = 0;
  std::uint64_t ql_bytes = 0;
  std::uint64_t qh_bytes = 0;
  std::uint64_t scale_bytes = 0;
  std::uint64_t min_bytes = 0;
  std::uint64_t d_bytes = 0;
  std::uint64_t dmin_bytes = 0;
  StreamResult raw_stream;
  StreamResult packed_q4k_x8_stream;
  StreamResult repacked_stream;
  StreamResult repacked_quant_only_stream;
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
    if (api.clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU_VALUE, 0, nullptr, &device_count) != CL_SUCCESS_VALUE || device_count == 0) {
      continue;
    }
    std::vector<cl_device_id> devices(device_count);
    Check(api.clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU_VALUE, device_count, devices.data(), nullptr), "clGetDeviceIDs(list)");
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
  Check(api.clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START_VALUE, sizeof(start), &start, nullptr), "clGetEventProfilingInfo(start)");
  Check(api.clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END_VALUE, sizeof(end), &end, nullptr), "clGetEventProfilingInfo(end)");
  return static_cast<double>(end - start) / 1000.0;
}

CpuChecksum ChecksumCpu(const std::vector<std::uint8_t>& bytes) {
  CpuChecksum out;
  for (std::uint64_t i = 0; i < static_cast<std::uint64_t>(bytes.size()); ++i) {
    const auto value = static_cast<std::uint64_t>(bytes[static_cast<std::size_t>(i)]);
    out.sum += value;
    out.weighted += (i + 1ULL) * (value + 1ULL);
  }
  return out;
}

std::array<std::uint8_t, 8> UnpackQ4KScalePlane(const std::uint8_t* block, bool mins) {
  constexpr std::uint32_t kMask1 = 0x3f3f3f3f;
  constexpr std::uint32_t kMask2 = 0x0f0f0f0f;
  constexpr std::uint32_t kMask3 = 0x03030303;
  std::array<std::uint32_t, 4> unpacked{};
  std::memcpy(unpacked.data(), block + 4, 12);
  unpacked[3] = ((unpacked[2] >> 4) & kMask2) | (((unpacked[1] >> 6) & kMask3) << 4);
  const std::uint32_t aux_scales = unpacked[1] & kMask1;
  unpacked[1] = (unpacked[2] & kMask2) | (((unpacked[0] >> 6) & kMask3) << 4);
  unpacked[2] = aux_scales;
  unpacked[0] &= kMask1;

  std::array<std::uint8_t, 8> out{};
  const auto* base = reinterpret_cast<const std::uint8_t*>(unpacked.data());
  std::memcpy(out.data(), base + (mins ? 8 : 0), out.size());
  return out;
}

RepackedTensor RepackQ4K(const std::vector<std::uint8_t>& raw) {
  constexpr std::uint64_t kBlockBytes = 144;
  Require(raw.size() % kBlockBytes == 0, "Q4_K tensor is not block-aligned");
  RepackedTensor out;
  out.layout = "q4k_plane_v0";
  out.block_count = static_cast<std::uint64_t>(raw.size()) / kBlockBytes;
  out.q4.resize(out.block_count * 128);
  out.scales.resize(out.block_count * 8);
  out.mins.resize(out.block_count * 8);
  out.d_values.resize(out.block_count * 2);
  out.dmin_values.resize(out.block_count * 2);
  for (std::uint64_t block_index = 0; block_index < out.block_count; ++block_index) {
    const auto* block = raw.data() + block_index * kBlockBytes;
    std::memcpy(out.d_values.data() + block_index * 2, block, 2);
    std::memcpy(out.dmin_values.data() + block_index * 2, block + 2, 2);
    const auto scales = UnpackQ4KScalePlane(block, false);
    const auto mins = UnpackQ4KScalePlane(block, true);
    std::memcpy(out.scales.data() + block_index * 8, scales.data(), scales.size());
    std::memcpy(out.mins.data() + block_index * 8, mins.data(), mins.size());
    std::memcpy(out.q4.data() + block_index * 128, block + 16, 128);
  }
  return out;
}

RepackedTensor RepackQ6K(const std::vector<std::uint8_t>& raw) {
  constexpr std::uint64_t kBlockBytes = 210;
  Require(raw.size() % kBlockBytes == 0, "Q6_K tensor is not block-aligned");
  RepackedTensor out;
  out.layout = "q6k_plane_v0";
  out.block_count = static_cast<std::uint64_t>(raw.size()) / kBlockBytes;
  out.ql.resize(out.block_count * 128);
  out.qh.resize(out.block_count * 64);
  out.scales.resize(out.block_count * 16);
  out.d_values.resize(out.block_count * 2);
  for (std::uint64_t block_index = 0; block_index < out.block_count; ++block_index) {
    const auto* block = raw.data() + block_index * kBlockBytes;
    std::memcpy(out.ql.data() + block_index * 128, block, 128);
    std::memcpy(out.qh.data() + block_index * 64, block + 128, 64);
    std::memcpy(out.scales.data() + block_index * 16, block + 192, 16);
    std::memcpy(out.d_values.data() + block_index * 2, block + 208, 2);
  }
  return out;
}

std::uint64_t ProductDimsFrom(const std::vector<std::uint64_t>& dims, std::size_t begin) {
  Require(begin < dims.size(), "invalid dim product start");
  std::uint64_t out = 1;
  for (std::size_t i = begin; i < dims.size(); ++i) {
    Require(dims[i] > 0, "tensor dim is zero");
    Require(out <= std::numeric_limits<std::uint64_t>::max() / dims[i],
            "tensor dim product overflows uint64");
    out *= dims[i];
  }
  return out;
}

std::uint64_t BlocksPerQ4KRow(const std::vector<std::uint64_t>& dims) {
  constexpr std::uint64_t kQk = 256;
  Require(dims.size() >= 2, "Q4_K x8 pack expects at least 2 tensor dims");
  Require(dims[0] % kQk == 0, "Q4_K row width is not a multiple of QK_K");
  return dims[0] / kQk;
}

void AppendQ4Kx8Block(const std::array<const std::uint8_t*, 8>& blocks,
                      std::vector<std::uint8_t>& out) {
  constexpr std::uint64_t kPackedBlockBytes = 1152;
  constexpr int kInterleaveBlock = 8;
  const std::size_t base = out.size();
  out.resize(base + kPackedBlockBytes);
  auto* dst = out.data() + base;
  auto* dst_d = dst;
  auto* dst_dmin = dst + 16;
  auto* dst_scales = dst + 32;
  auto* dst_qs = dst + 128;

  for (int i = 0; i < 8; ++i) {
    std::memcpy(dst_d + i * 2, blocks[static_cast<std::size_t>(i)], 2);
    std::memcpy(dst_dmin + i * 2, blocks[static_cast<std::size_t>(i)] + 2, 2);
  }

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

std::vector<std::uint8_t> PackQ4Kx8LlamaBytes(const std::vector<std::uint8_t>& raw,
                                              const std::vector<std::uint64_t>& dims) {
  constexpr std::uint64_t kRawBlockBytes = 144;
  const std::uint64_t blocks_per_row = BlocksPerQ4KRow(dims);
  const std::uint64_t row_count = ProductDimsFrom(dims, 1);
  Require(row_count % 8 == 0, "Q4_K x8 pack expects row count multiple of 8");
  Require(raw.size() == row_count * blocks_per_row * kRawBlockBytes,
          "Q4_K x8 pack raw size does not match dims");
  std::vector<std::uint8_t> out;
  out.reserve(raw.size());
  for (std::uint64_t row_base = 0; row_base < row_count; row_base += 8) {
    for (std::uint64_t block = 0; block < blocks_per_row; ++block) {
      std::array<const std::uint8_t*, 8> blocks{};
      for (int i = 0; i < 8; ++i) {
        const std::uint64_t raw_block_index =
            (row_base + static_cast<std::uint64_t>(i)) * blocks_per_row + block;
        blocks[static_cast<std::size_t>(i)] = raw.data() + raw_block_index * kRawBlockBytes;
      }
      AppendQ4Kx8Block(blocks, out);
    }
  }
  return out;
}

void AppendVector(std::vector<std::uint8_t>& out, const std::vector<std::uint8_t>& in) {
  out.insert(out.end(), in.begin(), in.end());
}

std::vector<std::uint8_t> FullRepackedBytes(const RepackedTensor& repacked) {
  std::vector<std::uint8_t> out;
  out.reserve(repacked.q4.size() + repacked.ql.size() + repacked.qh.size() +
              repacked.scales.size() + repacked.mins.size() +
              repacked.d_values.size() + repacked.dmin_values.size());
  AppendVector(out, repacked.q4);
  AppendVector(out, repacked.ql);
  AppendVector(out, repacked.qh);
  AppendVector(out, repacked.scales);
  AppendVector(out, repacked.mins);
  AppendVector(out, repacked.d_values);
  AppendVector(out, repacked.dmin_values);
  return out;
}

std::vector<std::uint8_t> QuantOnlyRepackedBytes(const RepackedTensor& repacked) {
  std::vector<std::uint8_t> out;
  out.reserve(repacked.q4.size() + repacked.ql.size() + repacked.qh.size());
  AppendVector(out, repacked.q4);
  AppendVector(out, repacked.ql);
  AppendVector(out, repacked.qh);
  return out;
}

std::vector<std::uint8_t> ReadTensorBytes(std::ifstream& in, const iq36::GgufTensorInfo& tensor) {
  Require(tensor.nbytes <= static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()),
          "tensor too large for size_t");
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(tensor.nbytes));
  in.clear();
  in.seekg(static_cast<std::streamoff>(tensor.absolute_offset), std::ios::beg);
  Require(static_cast<bool>(in), "failed to seek to tensor payload");
  in.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  Require(in.gcount() == static_cast<std::streamsize>(bytes.size()), "failed to read full tensor payload");
  return bytes;
}

LaneSpec ParseLane(const std::string& value) {
  const auto split = value.rfind(':');
  Require(split != std::string::npos && split > 0 && split + 1 < value.size(),
          "lane must use suffix:quant format");
  return LaneSpec{value.substr(0, split), value.substr(split + 1)};
}

std::string LaneKey(const LaneSpec& lane) {
  return lane.suffix + ":" + lane.type_name;
}

bool TensorMatchesLane(const iq36::GgufTensorInfo& tensor,
                       const std::string& tensor_type_name,
                       const LaneSpec& lane) {
  const auto suffix = tensor.suffix.empty() ? tensor.name : tensor.suffix;
  return suffix == lane.suffix && tensor_type_name == lane.type_name;
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
    else if (key == "--lane") args.lanes.push_back(ParseLane(value("--lane")));
    else if (key == "--max-tensors-per-lane") args.max_tensors_per_lane = std::stoi(value("--max-tensors-per-lane"));
    else if (key == "--repeat") args.repeat = std::stoi(value("--repeat"));
    else if (key == "--work-items") args.work_items = std::stoull(value("--work-items"));
    else if (key == "--device-substring") args.device_substring = value("--device-substring");
    else Die("unknown argument: " + key);
  }
  Require(!args.model_path.empty(), "--model is required");
  if (args.lanes.empty()) {
    args.lanes.push_back(ParseLane("attn_qkv.weight:Q4_K"));
    args.lanes.push_back(ParseLane("ffn_down_exps.weight:Q6_K"));
  }
  Require(args.max_tensors_per_lane > 0, "--max-tensors-per-lane must be positive");
  Require(args.repeat > 0, "--repeat must be positive");
  Require(args.work_items > 0, "--work-items must be positive");
  for (const auto& lane : args.lanes) {
    Require(lane.type_name == "Q4_K" || lane.type_name == "Q6_K",
            "only Q4_K and Q6_K lanes are supported");
  }
  return args;
}

struct OpenClRuntime {
  OpenClApi api;
  SelectedDevice selected;
  cl_context context = nullptr;
  cl_command_queue queue = nullptr;
  cl_program program = nullptr;
  cl_kernel kernel = nullptr;
  std::string build_log;
  double program_build_ms = 0.0;

  explicit OpenClRuntime(const std::string& device_substring) {
    selected = SelectDevice(api, device_substring);
    cl_int err = CL_SUCCESS_VALUE;
    context = api.clCreateContext(nullptr, 1, &selected.device, nullptr, nullptr, &err);
    Check(err, "clCreateContext");
    queue = api.clCreateCommandQueue(context, selected.device, CL_QUEUE_PROFILING_ENABLE_VALUE, &err);
    Check(err, "clCreateCommandQueue");
    const char* source = R"CLC(
__kernel void stream_checksum(__global const uchar* data, ulong n, __global ulong* partial) {
  const ulong gid = (ulong)get_global_id(0);
  const ulong gsize = (ulong)get_global_size(0);
  ulong sum = 0;
  ulong weighted = 0;
  for (ulong i = gid; i < n; i += gsize) {
    const ulong value = (ulong)data[i];
    sum += value;
    weighted += (i + 1UL) * (value + 1UL);
  }
  partial[gid] = sum;
  partial[gid + gsize] = weighted;
}
)CLC";
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
    kernel = api.clCreateKernel(program, "stream_checksum", &err);
    Check(err, "clCreateKernel(stream_checksum)");
  }

  ~OpenClRuntime() {
    if (kernel) api.clReleaseKernel(kernel);
    if (program) api.clReleaseProgram(program);
    if (queue) api.clReleaseCommandQueue(queue);
    if (context) api.clReleaseContext(context);
  }
};

StreamResult RunStream(OpenClRuntime& runtime,
                       const std::string& label,
                       const std::vector<std::uint8_t>& bytes,
                       std::uint64_t work_items,
                       int repeat) {
  Require(!bytes.empty(), label + " stream is empty");
  StreamResult result;
  result.label = label;
  result.byte_count = static_cast<std::uint64_t>(bytes.size());
  result.cpu = ChecksumCpu(bytes);
  cl_int err = CL_SUCCESS_VALUE;
  cl_mem data_buffer = runtime.api.clCreateBuffer(runtime.context, CL_MEM_READ_ONLY_VALUE, bytes.size(), nullptr, &err);
  Check(err, "clCreateBuffer(data)");
  std::vector<cl_ulong> partial(static_cast<std::size_t>(work_items * 2ULL), 0);
  cl_mem partial_buffer = runtime.api.clCreateBuffer(runtime.context, CL_MEM_WRITE_ONLY_VALUE, partial.size() * sizeof(cl_ulong), nullptr, &err);
  Check(err, "clCreateBuffer(partial)");

  cl_event write_event = nullptr;
  Check(runtime.api.clEnqueueWriteBuffer(runtime.queue, data_buffer, CL_FALSE_VALUE, 0, bytes.size(), bytes.data(), 0, nullptr, &write_event),
        "clEnqueueWriteBuffer");
  Check(runtime.api.clFinish(runtime.queue), "clFinish(write)");
  result.write_us = EventUs(runtime.api, write_event);
  runtime.api.clReleaseEvent(write_event);

  const cl_ulong n = static_cast<cl_ulong>(bytes.size());
  Check(runtime.api.clSetKernelArg(runtime.kernel, 0, sizeof(data_buffer), &data_buffer), "clSetKernelArg(0)");
  Check(runtime.api.clSetKernelArg(runtime.kernel, 1, sizeof(n), &n), "clSetKernelArg(1)");
  Check(runtime.api.clSetKernelArg(runtime.kernel, 2, sizeof(partial_buffer), &partial_buffer), "clSetKernelArg(2)");
  const std::size_t global = static_cast<std::size_t>(work_items);
  std::vector<double> kernel_us;
  kernel_us.reserve(static_cast<std::size_t>(repeat));
  for (int i = 0; i < repeat; ++i) {
    cl_event kernel_event = nullptr;
    Check(runtime.api.clEnqueueNDRangeKernel(runtime.queue, runtime.kernel, 1, nullptr, &global, nullptr, 0, nullptr, &kernel_event),
          "clEnqueueNDRangeKernel");
    Check(runtime.api.clFinish(runtime.queue), "clFinish(kernel)");
    kernel_us.push_back(EventUs(runtime.api, kernel_event));
    runtime.api.clReleaseEvent(kernel_event);
  }

  cl_event read_event = nullptr;
  Check(runtime.api.clEnqueueReadBuffer(runtime.queue, partial_buffer, CL_FALSE_VALUE, 0, partial.size() * sizeof(cl_ulong), partial.data(), 0, nullptr, &read_event),
        "clEnqueueReadBuffer");
  Check(runtime.api.clFinish(runtime.queue), "clFinish(read)");
  result.read_us = EventUs(runtime.api, read_event);
  runtime.api.clReleaseEvent(read_event);

  for (std::size_t i = 0; i < static_cast<std::size_t>(work_items); ++i) {
    result.gpu.sum += static_cast<std::uint64_t>(partial[i]);
    result.gpu.weighted += static_cast<std::uint64_t>(partial[i + static_cast<std::size_t>(work_items)]);
  }
  result.checksum_match =
      result.cpu.sum == result.gpu.sum && result.cpu.weighted == result.gpu.weighted;
  result.kernel_min_us = *std::min_element(kernel_us.begin(), kernel_us.end());
  double total_us = 0.0;
  for (double value : kernel_us) total_us += value;
  result.kernel_mean_us = total_us / static_cast<double>(kernel_us.size());
  const double byte_count = static_cast<double>(bytes.size());
  result.kernel_min_gb_s = byte_count / (result.kernel_min_us / 1e6) / 1e9;
  result.kernel_mean_gb_s = byte_count / (result.kernel_mean_us / 1e6) / 1e9;
  result.host_to_device_gb_s = byte_count / (result.write_us / 1e6) / 1e9;
  runtime.api.clReleaseMemObject(partial_buffer);
  runtime.api.clReleaseMemObject(data_buffer);
  return result;
}

ProbeRow ProbeTensor(OpenClRuntime& runtime,
                     std::ifstream& input,
                     const iq36::GgufTensorInfo& tensor,
                     const LaneSpec& lane,
                     std::uint64_t work_items,
                     int repeat) {
  const auto type_name = iq36::ggml_type_name(tensor.type);
  const auto raw = ReadTensorBytes(input, tensor);
  const auto repack_begin = std::chrono::steady_clock::now();
  const RepackedTensor repacked = type_name == "Q4_K" ? RepackQ4K(raw) : RepackQ6K(raw);
  const auto repack_end = std::chrono::steady_clock::now();
  auto repack_ns = static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(repack_end - repack_begin).count());
  if (repack_ns == 0) repack_ns = 1;
  const auto repacked_bytes = FullRepackedBytes(repacked);
  const auto quant_only_bytes = QuantOnlyRepackedBytes(repacked);
  std::vector<std::uint8_t> packed_q4k_x8_bytes;
  std::uint64_t packed_q4k_x8_repack_ns = 0;
  if (type_name == "Q4_K") {
    const auto packed_begin = std::chrono::steady_clock::now();
    packed_q4k_x8_bytes = PackQ4Kx8LlamaBytes(raw, tensor.dims);
    const auto packed_end = std::chrono::steady_clock::now();
    packed_q4k_x8_repack_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(packed_end - packed_begin).count());
    if (packed_q4k_x8_repack_ns == 0) packed_q4k_x8_repack_ns = 1;
  }

  ProbeRow row;
  row.absolute_offset = tensor.absolute_offset;
  row.block_count = repacked.block_count;
  if (type_name == "Q4_K") {
    row.blocks_per_row = BlocksPerQ4KRow(tensor.dims);
  }
  row.dims = tensor.dims;
  row.layer_index = tensor.layer_index;
  row.layout = repacked.layout;
  row.name = tensor.name;
  row.packed_q4k_x8_available = type_name == "Q4_K";
  row.packed_q4k_x8_layout = type_name == "Q4_K" ? "q4k_x8_llama_v0" : "";
  row.packed_q4k_x8_bytes = static_cast<std::uint64_t>(packed_q4k_x8_bytes.size());
  row.packed_q4k_x8_overhead_ratio =
      raw.empty() ? 0.0 : static_cast<double>(packed_q4k_x8_bytes.size()) / static_cast<double>(raw.size());
  row.packed_q4k_x8_repack_ns = packed_q4k_x8_repack_ns;
  row.packed_q4k_x8_repack_gb_s =
      packed_q4k_x8_repack_ns == 0
          ? 0.0
          : static_cast<double>(raw.size()) / static_cast<double>(packed_q4k_x8_repack_ns);
  row.raw_bytes = static_cast<std::uint64_t>(raw.size());
  row.repack_ns = repack_ns;
  row.repack_gb_s = static_cast<double>(raw.size()) / static_cast<double>(repack_ns);
  row.repacked_bytes = static_cast<std::uint64_t>(repacked_bytes.size());
  row.repacked_overhead_ratio =
      raw.empty() ? 0.0 : static_cast<double>(repacked_bytes.size()) / static_cast<double>(raw.size());
  row.selected_by_lane = LaneKey(lane);
  row.suffix = tensor.suffix.empty() ? tensor.name : tensor.suffix;
  row.type_name = type_name;
  row.q4_bytes = static_cast<std::uint64_t>(repacked.q4.size());
  row.ql_bytes = static_cast<std::uint64_t>(repacked.ql.size());
  row.qh_bytes = static_cast<std::uint64_t>(repacked.qh.size());
  row.scale_bytes = static_cast<std::uint64_t>(repacked.scales.size());
  row.min_bytes = static_cast<std::uint64_t>(repacked.mins.size());
  row.d_bytes = static_cast<std::uint64_t>(repacked.d_values.size());
  row.dmin_bytes = static_cast<std::uint64_t>(repacked.dmin_values.size());
  row.raw_stream = RunStream(runtime, "raw", raw, work_items, repeat);
  if (row.packed_q4k_x8_available) {
    row.packed_q4k_x8_stream =
        RunStream(runtime, "q4k_x8_llama_v0", packed_q4k_x8_bytes, work_items, repeat);
  }
  row.repacked_stream = RunStream(runtime, "repacked", repacked_bytes, work_items, repeat);
  row.repacked_quant_only_stream = RunStream(runtime, "repacked_quant_only", quant_only_bytes, work_items, repeat);
  return row;
}

void WriteDims(const std::vector<std::uint64_t>& dims) {
  std::cout << "[";
  for (std::size_t i = 0; i < dims.size(); ++i) {
    if (i != 0) std::cout << ",";
    std::cout << dims[i];
  }
  std::cout << "]";
}

void WriteChecksum(const CpuChecksum& checksum) {
  std::cout << "{\"sum\":" << checksum.sum << ",\"weighted\":" << checksum.weighted << "}";
}

void WriteStream(const StreamResult& stream) {
  std::cout << "{";
  std::cout << "\"byte_count\":" << stream.byte_count << ",";
  std::cout << "\"checksum_match\":" << (stream.checksum_match ? "true" : "false") << ",";
  std::cout << "\"cpu_checksum\":";
  WriteChecksum(stream.cpu);
  std::cout << ",\"gpu_checksum\":";
  WriteChecksum(stream.gpu);
  std::cout << ",\"host_to_device_gb_s\":" << stream.host_to_device_gb_s << ",";
  std::cout << "\"kernel_mean_gb_s\":" << stream.kernel_mean_gb_s << ",";
  std::cout << "\"kernel_mean_us\":" << stream.kernel_mean_us << ",";
  std::cout << "\"kernel_min_gb_s\":" << stream.kernel_min_gb_s << ",";
  std::cout << "\"kernel_min_us\":" << stream.kernel_min_us << ",";
  std::cout << "\"label\":\"" << JsonEscape(stream.label) << "\",";
  std::cout << "\"read_us\":" << stream.read_us << ",";
  std::cout << "\"write_us\":" << stream.write_us;
  std::cout << "}";
}

void WriteRow(const ProbeRow& row) {
  std::cout << "{";
  std::cout << "\"absolute_offset\":" << row.absolute_offset << ",";
  std::cout << "\"block_count\":" << row.block_count << ",";
  std::cout << "\"blocks_per_row\":" << row.blocks_per_row << ",";
  std::cout << "\"dims\":";
  WriteDims(row.dims);
  std::cout << ",\"layer_index\":" << row.layer_index << ",";
  std::cout << "\"layout\":\"" << JsonEscape(row.layout) << "\",";
  std::cout << "\"name\":\"" << JsonEscape(row.name) << "\",";
  std::cout << "\"packed_q4k_x8\":{";
  std::cout << "\"available\":" << (row.packed_q4k_x8_available ? "true" : "false") << ",";
  std::cout << "\"byte_count\":" << row.packed_q4k_x8_bytes << ",";
  std::cout << "\"layout\":\"" << JsonEscape(row.packed_q4k_x8_layout) << "\",";
  std::cout << "\"overhead_ratio\":" << row.packed_q4k_x8_overhead_ratio << ",";
  std::cout << "\"repack_gb_s\":" << row.packed_q4k_x8_repack_gb_s << ",";
  std::cout << "\"repack_ns\":" << row.packed_q4k_x8_repack_ns << ",";
  std::cout << "\"stream\":";
  WriteStream(row.packed_q4k_x8_stream);
  std::cout << "},";
  std::cout << "\"plane_bytes\":{";
  std::cout << "\"d\":" << row.d_bytes << ",";
  std::cout << "\"dmin\":" << row.dmin_bytes << ",";
  std::cout << "\"mins\":" << row.min_bytes << ",";
  std::cout << "\"q4\":" << row.q4_bytes << ",";
  std::cout << "\"qh\":" << row.qh_bytes << ",";
  std::cout << "\"ql\":" << row.ql_bytes << ",";
  std::cout << "\"scales\":" << row.scale_bytes;
  std::cout << "},";
  std::cout << "\"raw_bytes\":" << row.raw_bytes << ",";
  std::cout << "\"raw_stream\":";
  WriteStream(row.raw_stream);
  std::cout << ",\"repack_gb_s\":" << row.repack_gb_s << ",";
  std::cout << "\"repack_ns\":" << row.repack_ns << ",";
  std::cout << "\"repacked_bytes\":" << row.repacked_bytes << ",";
  std::cout << "\"repacked_overhead_ratio\":" << row.repacked_overhead_ratio << ",";
  std::cout << "\"repacked_quant_only_stream\":";
  WriteStream(row.repacked_quant_only_stream);
  std::cout << ",\"repacked_stream\":";
  WriteStream(row.repacked_stream);
  std::cout << ",\"selected_by_lane\":\"" << JsonEscape(row.selected_by_lane) << "\",";
  std::cout << "\"suffix\":\"" << JsonEscape(row.suffix) << "\",";
  std::cout << "\"type_name\":\"" << JsonEscape(row.type_name) << "\"";
  std::cout << "}";
}

int main(int argc, char** argv) {
  try {
    const Args args = ParseArgs(argc, argv);
    const auto index = iq36::parse_gguf_model_index(args.model_path);
    const auto load_map = iq36::validate_qwen36_load_map(index);
    std::ifstream input(args.model_path, std::ios::binary);
    Require(static_cast<bool>(input), "failed to open model");
    OpenClRuntime runtime(args.device_substring);

    std::map<std::string, int> selected_counts;
    std::vector<ProbeRow> rows;
    for (const auto& tensor : index.tensors) {
      const auto type_name = iq36::ggml_type_name(tensor.type);
      for (const auto& lane : args.lanes) {
        const auto key = LaneKey(lane);
        if (selected_counts[key] >= args.max_tensors_per_lane) {
          continue;
        }
        if (!TensorMatchesLane(tensor, type_name, lane)) {
          continue;
        }
        rows.push_back(ProbeTensor(runtime, input, tensor, lane, args.work_items, args.repeat));
        ++selected_counts[key];
        break;
      }
    }
    Require(!rows.empty(), "no tensors selected");

    std::uint64_t raw_bytes = 0;
    std::uint64_t repacked_bytes = 0;
    std::uint64_t packed_q4k_x8_bytes = 0;
    std::uint64_t packed_q4k_x8_rows = 0;
    bool checksum_ok = true;
    double raw_min_gb_s = 0.0;
    double repacked_min_gb_s = 0.0;
    double quant_min_gb_s = 0.0;
    double packed_q4k_x8_min_gb_s = 0.0;
    for (const auto& row : rows) {
      raw_bytes += row.raw_bytes;
      repacked_bytes += row.repacked_bytes;
      packed_q4k_x8_bytes += row.packed_q4k_x8_bytes;
      checksum_ok = checksum_ok && row.raw_stream.checksum_match &&
                    row.repacked_stream.checksum_match &&
                    row.repacked_quant_only_stream.checksum_match &&
                    (!row.packed_q4k_x8_available || row.packed_q4k_x8_stream.checksum_match);
      raw_min_gb_s += row.raw_stream.kernel_min_gb_s;
      repacked_min_gb_s += row.repacked_stream.kernel_min_gb_s;
      quant_min_gb_s += row.repacked_quant_only_stream.kernel_min_gb_s;
      if (row.packed_q4k_x8_available) {
        packed_q4k_x8_min_gb_s += row.packed_q4k_x8_stream.kernel_min_gb_s;
        ++packed_q4k_x8_rows;
      }
    }
    raw_min_gb_s /= static_cast<double>(rows.size());
    repacked_min_gb_s /= static_cast<double>(rows.size());
    quant_min_gb_s /= static_cast<double>(rows.size());
    const double packed_q4k_x8_min_gb_s_mean =
        packed_q4k_x8_rows == 0
            ? 0.0
            : packed_q4k_x8_min_gb_s / static_cast<double>(packed_q4k_x8_rows);

    std::cout << std::setprecision(10);
    std::cout << "{";
    std::cout << "\"schema_version\":\"intel-qwen36-gpu-repack-stream-probe-v1\",";
    std::cout << "\"model_path\":\"" << JsonEscape(args.model_path) << "\",";
    std::cout << "\"file_size_bytes\":" << index.file_size_bytes << ",";
    std::cout << "\"platform_name\":\"" << JsonEscape(runtime.selected.platform_name) << "\",";
    std::cout << "\"device_name\":\"" << JsonEscape(runtime.selected.device_name) << "\",";
    std::cout << "\"program_build_ms\":" << runtime.program_build_ms << ",";
    std::cout << "\"build_log\":\"" << JsonEscape(runtime.build_log) << "\",";
    std::cout << "\"repeat\":" << args.repeat << ",";
    std::cout << "\"work_items\":" << args.work_items << ",";
    std::cout << "\"max_tensors_per_lane\":" << args.max_tensors_per_lane << ",";
    std::cout << "\"native_gguf_load_map_ready\":" << (load_map.ready ? "true" : "false") << ",";
    std::cout << "\"lanes\":[";
    for (std::size_t i = 0; i < args.lanes.size(); ++i) {
      if (i != 0) std::cout << ",";
      std::cout << "{\"suffix\":\"" << JsonEscape(args.lanes[i].suffix)
                << "\",\"type_name\":\"" << JsonEscape(args.lanes[i].type_name) << "\"}";
    }
    std::cout << "],";
    std::cout << "\"rows\":[";
    for (std::size_t i = 0; i < rows.size(); ++i) {
      if (i != 0) std::cout << ",";
      WriteRow(rows[i]);
    }
    std::cout << "],";
    std::cout << "\"aggregate\":{";
    std::cout << "\"checksum_match_all\":" << (checksum_ok ? "true" : "false") << ",";
    std::cout << "\"packed_q4k_x8_bytes\":" << packed_q4k_x8_bytes << ",";
    std::cout << "\"packed_q4k_x8_kernel_min_gb_s_mean\":" << packed_q4k_x8_min_gb_s_mean << ",";
    std::cout << "\"packed_q4k_x8_tensor_count\":" << packed_q4k_x8_rows << ",";
    std::cout << "\"raw_bytes\":" << raw_bytes << ",";
    std::cout << "\"repacked_bytes\":" << repacked_bytes << ",";
    std::cout << "\"repacked_overhead_ratio\":"
              << (raw_bytes == 0 ? 0.0 : static_cast<double>(repacked_bytes) / static_cast<double>(raw_bytes)) << ",";
    std::cout << "\"raw_kernel_min_gb_s_mean\":" << raw_min_gb_s << ",";
    std::cout << "\"repacked_kernel_min_gb_s_mean\":" << repacked_min_gb_s << ",";
    std::cout << "\"repacked_quant_only_kernel_min_gb_s_mean\":" << quant_min_gb_s << ",";
    std::cout << "\"selected_tensor_count\":" << rows.size();
    std::cout << "}";
    std::cout << "}\n";
    return checksum_ok ? 0 : 3;
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
  parser.add_argument("--env-script", default=DEFAULT_ENV_SCRIPT)
  parser.add_argument("--staging-root", "--remote-root", dest="remote_root", metavar="PATH", default=DEFAULT_REMOTE_ROOT)
  parser.add_argument("--lane", action="append", default=None)
  parser.add_argument("--max-tensors-per-lane", type=int, default=1)
  parser.add_argument("--repeat", type=int, default=5)
  parser.add_argument("--work-items", type=int, default=4096)
  parser.add_argument("--device-substring", default="B390")
  parser.add_argument("--timeout-s", type=int, default=900)
  parser.add_argument("--out-dir", type=Path, default=None)
  return parser.parse_args()


def shell_join(argv: list[str]) -> str:
  return " ".join(shlex.quote(item) for item in argv)


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


def stream_gb_s(row: dict[str, Any], key: str) -> float:
  value = row.get(key, {})
  if not isinstance(value, dict):
    return 0.0
  metric = value.get("kernel_min_gb_s")
  return float(metric) if isinstance(metric, (int, float)) else 0.0


def q4_x8_packed_stream_gb_s(row: dict[str, Any]) -> float:
  packed = row.get("packed_q4k_x8", {})
  if not isinstance(packed, dict):
    return 0.0
  stream = packed.get("stream", {})
  if not isinstance(stream, dict):
    return 0.0
  metric = stream.get("kernel_min_gb_s")
  return float(metric) if isinstance(metric, (int, float)) else 0.0


def write_summary(path: Path, payload: dict[str, Any]) -> None:
  probe = payload.get("probe") if isinstance(payload.get("probe"), dict) else {}
  rows = probe.get("rows", []) if isinstance(probe, dict) else []
  aggregate = probe.get("aggregate", {}) if isinstance(probe, dict) else {}
  row_types = {
      row.get("type_name")
      for row in rows
      if isinstance(row, dict) and isinstance(row.get("type_name"), str)
  } if isinstance(rows, list) else set()
  lines = [
      "# GPU Raw/Plane/Q4-X8 Tensor Stream Probe",
      "",
      f"- workstream: `{WORKSTREAM}`",
      f"- host: `{payload.get('host')}`",
      f"- model: `{payload.get('model')}`",
      f"- required checks passed: `{str(payload.get('required_checks_passed')).lower()}`",
      "- speedup claims allowed: `false`",
      f"- platform/device: `{probe.get('platform_name')}` / `{probe.get('device_name')}`",
      f"- selected tensors: `{aggregate.get('selected_tensor_count', 0)}`",
      f"- checksum match all streams: `{str(aggregate.get('checksum_match_all')).lower()}`",
      f"- raw/repacked bytes: `{aggregate.get('raw_bytes')}` / `{aggregate.get('repacked_bytes')}`",
      f"- q4 x8 packed bytes: `{aggregate.get('packed_q4k_x8_bytes')}`",
      f"- repacked overhead ratio: `{aggregate.get('repacked_overhead_ratio')}`",
      "",
      "| lane | tensor | plane layout | q4 x8 layout | raw MB | plane MB | q4 x8 MB | raw GB/s | plane GB/s | q4 x8 GB/s | quant-only GB/s | checksums |",
      "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
  ]
  if isinstance(rows, list):
    for row in rows:
      if not isinstance(row, dict):
        continue
      packed = row.get("packed_q4k_x8", {})
      if not isinstance(packed, dict):
        packed = {}
      packed_stream = packed.get("stream", {})
      if not isinstance(packed_stream, dict):
        packed_stream = {}
      checksums = [
          row.get("raw_stream", {}).get("checksum_match"),
          row.get("repacked_stream", {}).get("checksum_match"),
          row.get("repacked_quant_only_stream", {}).get("checksum_match"),
      ]
      if packed.get("available") is True:
        checksums.append(packed_stream.get("checksum_match"))
      lines.append(
          "| "
          + " | ".join([
              f"`{row.get('selected_by_lane')}`",
              f"`{row.get('name')}`",
              f"`{row.get('layout')}`",
              f"`{packed.get('layout') or 'n/a'}`",
              f"{float(row.get('raw_bytes', 0)) / 1e6:.1f}",
              f"{float(row.get('repacked_bytes', 0)) / 1e6:.1f}",
              f"{float(packed.get('byte_count', 0) or 0) / 1e6:.1f}",
              f"{stream_gb_s(row, 'raw_stream'):.3f}",
              f"{stream_gb_s(row, 'repacked_stream'):.3f}",
              f"{q4_x8_packed_stream_gb_s(row):.3f}",
              f"{stream_gb_s(row, 'repacked_quant_only_stream'):.3f}",
              "`" + "/".join("true" if item is True else "false" for item in checksums) + "`",
          ])
          + " |"
      )
  lines += [
      "",
      "Decision: this is a GPU tensor stream layout gate, not model throughput",
      "evidence.",
  ]
  if "Q4_K" in row_types:
    lines.append(
        "Use any passing Q4 x8 stream as input to the next narrow qmatvec "
        "correctness/timing probe before decode wiring."
    )
  if "Q6_K" in row_types:
    lines.append(
        "For Q6_K, compare raw vs plane-stream bandwidth before reopening a "
        "selected-down layout route."
    )
  lines.append("")
  path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  args = parse_args()
  if args.max_tensors_per_lane <= 0:
    raise SystemExit("--max-tensors-per-lane must be positive")
  if args.repeat <= 0:
    raise SystemExit("--repeat must be positive")
  if args.work_items <= 0:
    raise SystemExit("--work-items must be positive")
  lanes = args.lane or list(DEFAULT_LANES)
  stamp = utc_stamp()
  created_at = iso_now()
  out_dir = args.out_dir or ROOT / f"output/gpu-repack-stream-probe-{stamp}"
  out_dir.mkdir(parents=True, exist_ok=False)
  raw_dir = out_dir / "raw"
  raw_dir.mkdir()
  local_cpp = out_dir / "gpu_repack_stream_probe.cpp"
  local_cpp.write_text(PROBE_CPP, encoding="utf-8")

  remote_dir = f"{args.remote_root.rstrip('/')}/gpu-repack-stream-probe-{stamp}"
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
    transfers.append(iq36_local.copy_to(args.host, local_cpp, f"{remote_dir}/tests/gpu_repack_stream_probe.cpp", args.timeout_s))

  compile_cmd = " && ".join([
      f"source {shlex.quote(args.env_script)} >/dev/null 2>&1",
      (
          "g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic "
          f"-I {shlex.quote(remote_dir + '/include')} "
          f"{shlex.quote(remote_dir + '/src/gguf_loader.cpp')} "
          f"{shlex.quote(remote_dir + '/tests/gpu_repack_stream_probe.cpp')} "
          "-ldl -pthread "
          f"-o {shlex.quote(remote_dir + '/build/iq36-gpu-repack-stream-probe')}"
      ),
  ])
  compile_result = (
      iq36_local.run_target(args.host, compile_cmd, args.timeout_s)
      if transfers and all(item.get("returncode") == 0 for item in transfers)
      else {"cmd": ["stage"], "returncode": 1, "stdout": "", "stderr": "stage failed"}
  )
  run_argv = [
      f"{remote_dir}/build/iq36-gpu-repack-stream-probe",
      "--model",
      args.model,
      "--max-tensors-per-lane",
      str(args.max_tensors_per_lane),
      "--repeat",
      str(args.repeat),
      "--work-items",
      str(args.work_items),
      "--device-substring",
      args.device_substring,
  ]
  for lane in lanes:
    run_argv += ["--lane", lane]
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

  rows = probe.get("rows", []) if isinstance(probe, dict) else []
  aggregate = probe.get("aggregate", {}) if isinstance(probe, dict) else {}
  selected_lanes = {
      row.get("selected_by_lane")
      for row in rows
      if isinstance(row, dict) and isinstance(row.get("selected_by_lane"), str)
  }

  def stream_ok(row: dict[str, Any], key: str) -> bool:
    stream = row.get(key)
    return (
        isinstance(stream, dict)
        and stream.get("checksum_match") is True
        and isinstance(stream.get("kernel_min_gb_s"), (int, float))
        and stream["kernel_min_gb_s"] > 0
    )

  def q4_x8_packed_ok(row: dict[str, Any]) -> bool:
    packed = row.get("packed_q4k_x8")
    if not isinstance(packed, dict):
      return False
    stream = packed.get("stream")
    return (
        packed.get("available") is True
        and packed.get("layout") == "q4k_x8_llama_v0"
        and packed.get("byte_count") == row.get("raw_bytes")
        and 0.999 <= float(packed.get("overhead_ratio", 0.0)) <= 1.001
        and isinstance(stream, dict)
        and stream.get("checksum_match") is True
        and isinstance(stream.get("kernel_min_gb_s"), (int, float))
        and stream["kernel_min_gb_s"] > 0
    )

  q4_rows = [
      row
      for row in rows
      if isinstance(row, dict) and row.get("type_name") == "Q4_K"
  ]
  q6_rows = [
      row
      for row in rows
      if isinstance(row, dict) and row.get("type_name") == "Q6_K"
  ]

  checks = [
      {"name": "remote_dir_created", "pass": setup.get("returncode") == 0},
      {
          "name": "source_files_transferred",
          "pass": bool(transfers) and all(item.get("returncode") == 0 for item in transfers),
      },
      {"name": "probe_compiled", "pass": compile_result.get("returncode") == 0},
      {"name": "probe_stdout_json_parsed", "pass": isinstance(probe, dict)},
      {"name": "probe_process_succeeded", "pass": run_result.get("returncode") == 0},
      {
          "name": "arc_b390_selected",
          "pass": bool(probe and "B390" in str(probe.get("device_name", ""))),
      },
      {
          "name": "requested_lanes_selected",
          "pass": all(lane in selected_lanes for lane in lanes),
      },
      {
          "name": "raw_repacked_quant_stream_checksums_match",
          "pass": bool(rows)
          and all(
              isinstance(row, dict)
              and stream_ok(row, "raw_stream")
              and stream_ok(row, "repacked_stream")
              and stream_ok(row, "repacked_quant_only_stream")
              for row in rows
          ),
      },
      {
          "name": "q4_x8_llama_packed_stream_checksums_match",
          "pass": not q4_rows or all(q4_x8_packed_ok(row) for row in q4_rows),
      },
      {
          "name": "q4_overhead_within_plane_v0_bound",
          "pass": not q4_rows or all(
              float(row.get("repacked_overhead_ratio", 99.0)) <= 1.03
              for row in q4_rows
          ),
      },
      {
          "name": "q6_overhead_near_raw",
          "pass": not q6_rows or all(
              0.999 <= float(row.get("repacked_overhead_ratio", 0.0)) <= 1.001
              for row in q6_rows
          ),
      },
      {
          "name": "aggregate_records_selected_tensors",
          "pass": isinstance(aggregate.get("selected_tensor_count"), int)
          and aggregate.get("selected_tensor_count", 0) >= len(lanes),
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
      "lanes": lanes,
      "max_tensors_per_lane": args.max_tensors_per_lane,
      "repeat": args.repeat,
      "work_items": args.work_items,
      "probe": probe,
      "checks": checks,
      "required_checks_passed": required_checks_passed,
      "speedup_claims_allowed": False,
      "recommendation": "use the passing q4k_x8_llama_v0 stream gate for the next narrow qmatvec probe before any full decode backend",
  }
  manifest = {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "created_at": created_at,
      "tool": "tools/intel-qwen36-gpu-repack-stream-probe.py",
      "artifact": str(out_dir),
      "host": args.host,
      "remote_dir": remote_dir,
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
  iq36_local.write_metric(
      out_dir / "metrics.jsonl",
      "gpu_repack_stream_probe",
      [
          ("selected_tensor_count", aggregate.get("selected_tensor_count") if isinstance(aggregate, dict) else None),
          ("raw_kernel_min_gb_s_mean", aggregate.get("raw_kernel_min_gb_s_mean") if isinstance(aggregate, dict) else None),
          ("repacked_kernel_min_gb_s_mean", aggregate.get("repacked_kernel_min_gb_s_mean") if isinstance(aggregate, dict) else None),
          ("repacked_quant_only_kernel_min_gb_s_mean", aggregate.get("repacked_quant_only_kernel_min_gb_s_mean") if isinstance(aggregate, dict) else None),
          ("packed_q4k_x8_kernel_min_gb_s_mean", aggregate.get("packed_q4k_x8_kernel_min_gb_s_mean") if isinstance(aggregate, dict) else None),
          ("packed_q4k_x8_tensor_count", aggregate.get("packed_q4k_x8_tensor_count") if isinstance(aggregate, dict) else None),
          ("checksum_match_all", aggregate.get("checksum_match_all") if isinstance(aggregate, dict) else None),
      ],
  )
  write_summary(out_dir / "summary.md", payload)
  print(out_dir)
  return 0 if required_checks_passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
