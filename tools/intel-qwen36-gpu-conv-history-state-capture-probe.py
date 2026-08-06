#!/usr/bin/env python3
"""Capture the layer-5 linear-attention conv history state and hand it to GPU conv."""

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
SCHEMA_VERSION = "intel-qwen36-gpu-conv-history-state-capture-probe-v0"
DEFAULT_HOST = "local"
DEFAULT_MODEL = "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf"
DEFAULT_ENV_SCRIPT = "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh"
DEFAULT_REMOTE_ROOT = "/home/intel/intel-qwen36-gpu"
DEFAULT_ORACLE_BUNDLE = ROOT / "oracle/r0-oracle-bundle-20260627T060028Z"
OPENCL_SOURCE = ROOT / "engine/gpu/opencl/q4x8_matvec.cl"
CAPTURE_TOOL = ROOT / "tools/intel-qwen36-r0-boundary-capture-run.py"
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
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

const char* kQ4X8OpenClSource = @@OPENCL_SOURCE_LITERAL@@;

constexpr int kHiddenSize = 2048;
constexpr int kQkvMixedSize = 8192;
constexpr int kConvKernelSize = 4;
constexpr int kConvStateSize = (kConvKernelSize - 1) * kQkvMixedSize;

struct Args {
  std::string model_path;
  std::string payload_dir;
  int layer = 5;
  int repeat = 7;
  std::string device_substring = "B390";
};

[[noreturn]] void Die(const std::string& message) {
  throw std::runtime_error(message);
}

void Require(bool ok, const std::string& message) {
  if (!ok) {
    Die(message);
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

std::string JoinPath(const std::string& dir, const std::string& name) {
  return (!dir.empty() && dir.back() == '/') ? dir + name : dir + "/" + name;
}

std::string LayerTensorName(int layer, const std::string& suffix) {
  return "blk." + std::to_string(layer) + "." + suffix;
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
    else if (key == "--payload-dir") args.payload_dir = value("--payload-dir");
    else if (key == "--layer") args.layer = std::stoi(value("--layer"));
    else if (key == "--repeat") args.repeat = std::stoi(value("--repeat"));
    else if (key == "--device-substring") args.device_substring = value("--device-substring");
    else Die("unknown argument: " + key);
  }
  Require(!args.model_path.empty(), "--model is required");
  Require(!args.payload_dir.empty(), "--payload-dir is required");
  Require(args.layer >= 0 && args.layer < 40, "--layer is out of range");
  Require(args.repeat > 0, "--repeat must be positive");
  return args;
}

std::vector<std::uint8_t> ReadTensorBytes(std::ifstream& in,
                                          const iq36::GgufTensorInfo& tensor) {
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(tensor.nbytes));
  in.clear();
  in.seekg(static_cast<std::streamoff>(tensor.absolute_offset), std::ios::beg);
  Require(static_cast<bool>(in), "failed to seek tensor payload");
  in.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  Require(in.gcount() == static_cast<std::streamsize>(bytes.size()), "failed to read tensor payload");
  return bytes;
}

std::vector<float> ReadF32Tensor(std::ifstream& in,
                                 const iq36::GgufTensorInfo& tensor,
                                 std::size_t expected_values) {
  Require(tensor.type == 0, "tensor is not F32: " + tensor.name);
  Require(tensor.nbytes == expected_values * sizeof(float), "F32 tensor byte size mismatch: " + tensor.name);
  const auto bytes = ReadTensorBytes(in, tensor);
  std::vector<float> values(expected_values, 0.0f);
  std::memcpy(values.data(), bytes.data(), bytes.size());
  return values;
}

bool ComparePassed(const iq36::VectorCompareStats& stats) {
  return stats.same_size &&
         stats.finite &&
         stats.mismatch_count == 0 &&
         stats.max_abs_diff <= 5e-3 &&
         stats.rmse <= 1e-3 &&
         stats.cosine >= 0.99999;
}

void WriteCompare(const iq36::VectorCompareStats& stats) {
  std::cout << "{";
  std::cout << "\"compared_value_count\":" << stats.compared_value_count << ",";
  std::cout << "\"cosine\":" << stats.cosine << ",";
  std::cout << "\"finite\":" << (stats.finite ? "true" : "false") << ",";
  std::cout << "\"max_abs_diff\":" << stats.max_abs_diff << ",";
  std::cout << "\"mean_abs_diff\":" << stats.mean_abs_diff << ",";
  std::cout << "\"mismatch_count\":" << stats.mismatch_count << ",";
  std::cout << "\"rmse\":" << stats.rmse << ",";
  std::cout << "\"same_size\":" << (stats.same_size ? "true" : "false");
  std::cout << "}";
}

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

constexpr cl_int kClSuccess = 0;
constexpr cl_bool kClTrue = 1;
constexpr cl_device_type kClDeviceTypeGpu = 1ULL << 2;
constexpr cl_mem_flags kClMemReadOnly = 1ULL << 2;
constexpr cl_mem_flags kClMemWriteOnly = 1ULL << 1;
constexpr cl_command_queue_properties kClQueueProfilingEnable = 1ULL << 1;
constexpr cl_platform_info kClPlatformName = 0x0902;
constexpr cl_device_info kClDeviceName = 0x102B;
constexpr cl_program_build_info kClProgramBuildLog = 0x1183;
constexpr cl_profiling_info kClProfilingCommandStart = 0x1282;
constexpr cl_profiling_info kClProfilingCommandEnd = 0x1283;

void Check(cl_int err, const std::string& where) {
  if (err != kClSuccess) {
    std::ostringstream oss;
    oss << where << " failed with OpenCL error " << err;
    Die(oss.str());
  }
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

  ~OpenClApi() {
    if (lib) {
      dlclose(lib);
    }
  }
};

std::string PlatformString(OpenClApi& api, cl_platform_id platform, cl_platform_info info) {
  std::size_t size = 0;
  Check(api.clGetPlatformInfo(platform, info, 0, nullptr, &size), "clGetPlatformInfo(size)");
  std::string out(size, '\0');
  Check(api.clGetPlatformInfo(platform, info, size, out.data(), nullptr), "clGetPlatformInfo(value)");
  if (!out.empty() && out.back() == '\0') {
    out.pop_back();
  }
  return out;
}

std::string DeviceString(OpenClApi& api, cl_device_id device, cl_device_info info) {
  std::size_t size = 0;
  Check(api.clGetDeviceInfo(device, info, 0, nullptr, &size), "clGetDeviceInfo(size)");
  std::string out(size, '\0');
  Check(api.clGetDeviceInfo(device, info, size, out.data(), nullptr), "clGetDeviceInfo(value)");
  if (!out.empty() && out.back() == '\0') {
    out.pop_back();
  }
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
    if (api.clGetDeviceIDs(platform, kClDeviceTypeGpu, 0, nullptr, &device_count) != kClSuccess ||
        device_count == 0) {
      continue;
    }
    std::vector<cl_device_id> devices(device_count);
    Check(api.clGetDeviceIDs(platform, kClDeviceTypeGpu, device_count, devices.data(), nullptr),
          "clGetDeviceIDs(list)");
    for (cl_device_id device : devices) {
      const std::string name = DeviceString(api, device, kClDeviceName);
      if (device_substring.empty() || name.find(device_substring) != std::string::npos) {
        return {platform, device, PlatformString(api, platform, kClPlatformName), name};
      }
    }
  }
  Die("no matching OpenCL GPU for substring: " + device_substring);
}

double EventUs(OpenClApi& api, cl_event event) {
  cl_ulong start = 0;
  cl_ulong end = 0;
  Check(api.clGetEventProfilingInfo(event, kClProfilingCommandStart, sizeof(start), &start, nullptr),
        "clGetEventProfilingInfo(start)");
  Check(api.clGetEventProfilingInfo(event, kClProfilingCommandEnd, sizeof(end), &end, nullptr),
        "clGetEventProfilingInfo(end)");
  return static_cast<double>(end - start) / 1000.0;
}

std::string BuildLog(OpenClApi& api, cl_program program, cl_device_id device) {
  std::size_t log_size = 0;
  api.clGetProgramBuildInfo(program, device, kClProgramBuildLog, 0, nullptr, &log_size);
  if (log_size == 0) {
    return "";
  }
  std::string log(log_size, '\0');
  api.clGetProgramBuildInfo(program, device, kClProgramBuildLog, log_size, log.data(), nullptr);
  if (!log.empty() && log.back() == '\0') {
    log.pop_back();
  }
  return log;
}

void ReleaseMem(OpenClApi& api, cl_mem* mem) {
  if (*mem) {
    api.clReleaseMemObject(*mem);
    *mem = nullptr;
  }
}

struct GpuConvProbeRun {
  std::vector<float> qkv_mixed;
  std::vector<float> conv_output_raw;
  std::vector<float> conv_state;
  double qkv_min_us = 0.0;
  double qkv_mean_us = 0.0;
  double qkv_effective_packed_gb_s = 0.0;
  double conv_min_us = 0.0;
  double conv_mean_us = 0.0;
  std::string platform_name;
  std::string device_name;
  std::string build_log;
  double program_build_ms = 0.0;
};

GpuConvProbeRun RunGpuQ6KThenConv(
    const std::vector<std::uint8_t>& q6_raw,
    const std::vector<std::int8_t>& q8_qs,
    const std::vector<float>& q8_d,
    const std::vector<float>& conv_weights,
    const std::vector<float>& conv_state,
    std::uint64_t rows,
    std::uint64_t blocks_per_row,
    std::uint64_t conv_kernel_size,
    int repeat,
    const std::string& device_substring) {
  Require(q6_raw.size() == rows * blocks_per_row * 210ULL, "Q6_K raw byte size mismatch");
  Require(q8_qs.size() == blocks_per_row * 256ULL, "Q6_K Q8 qs size mismatch");
  Require(q8_d.size() == blocks_per_row, "Q6_K Q8 d size mismatch");
  Require(conv_weights.size() == rows * conv_kernel_size, "conv weight size mismatch");
  Require(conv_state.size() == rows * (conv_kernel_size - 1), "conv state size mismatch");
  GpuConvProbeRun run;
  run.qkv_mixed.assign(static_cast<std::size_t>(rows), 0.0f);
  run.conv_output_raw.assign(static_cast<std::size_t>(rows), 0.0f);
  run.conv_state.assign(static_cast<std::size_t>(rows * (conv_kernel_size - 1)), 0.0f);

  OpenClApi api;
  const auto selected = SelectDevice(api, device_substring);
  run.platform_name = selected.platform_name;
  run.device_name = selected.device_name;
  cl_int err = kClSuccess;
  cl_context context =
      api.clCreateContext(nullptr, 1, &selected.device, nullptr, nullptr, &err);
  Check(err, "clCreateContext(q6 preconv)");
  cl_command_queue queue =
      api.clCreateCommandQueue(context, selected.device, kClQueueProfilingEnable, &err);
  Check(err, "clCreateCommandQueue(q6 preconv)");
  const char* source = kQ4X8OpenClSource;
  const std::size_t source_len = std::strlen(kQ4X8OpenClSource);
  cl_program program =
      api.clCreateProgramWithSource(context, 1, &source, &source_len, &err);
  Check(err, "clCreateProgramWithSource(q6 preconv)");
  const auto build_begin = std::chrono::steady_clock::now();
  err = api.clBuildProgram(program, 1, &selected.device, "", nullptr, nullptr);
  const auto build_end = std::chrono::steady_clock::now();
  run.program_build_ms =
      std::chrono::duration<double, std::milli>(build_end - build_begin).count();
  run.build_log = BuildLog(api, program, selected.device);
  Check(err, "clBuildProgram(q6 preconv)");
  cl_kernel q6_kernel = api.clCreateKernel(program, "q6k_selected_down_matvec_row", &err);
  Check(err, "clCreateKernel(q6k_selected_down_matvec_row)");
  cl_kernel conv_kernel = api.clCreateKernel(program, "linear_attn_conv_f32", &err);
  Check(err, "clCreateKernel(linear_attn_conv_f32)");

  cl_mem q6_buffer = nullptr;
  cl_mem q8_qs_buffer = nullptr;
  cl_mem q8_d_buffer = nullptr;
  cl_mem qkv_buffer = nullptr;
  cl_mem conv_weight_buffer = nullptr;
  cl_mem conv_state_buffer = nullptr;
  cl_mem conv_output_buffer = nullptr;
  cl_mem next_state_buffer = nullptr;
  try {
    q6_buffer = api.clCreateBuffer(context, kClMemReadOnly, q6_raw.size(), nullptr, &err);
    Check(err, "clCreateBuffer(q6 raw)");
    q8_qs_buffer = api.clCreateBuffer(
        context, kClMemReadOnly, q8_qs.size() * sizeof(std::int8_t), nullptr, &err);
    Check(err, "clCreateBuffer(q6 q8 qs)");
    q8_d_buffer = api.clCreateBuffer(
        context, kClMemReadOnly, q8_d.size() * sizeof(float), nullptr, &err);
    Check(err, "clCreateBuffer(q6 q8 d)");
    qkv_buffer = api.clCreateBuffer(
        context, kClMemWriteOnly, run.qkv_mixed.size() * sizeof(float), nullptr, &err);
    Check(err, "clCreateBuffer(q6 qkv output)");
    conv_weight_buffer = api.clCreateBuffer(
        context, kClMemReadOnly, conv_weights.size() * sizeof(float), nullptr, &err);
    Check(err, "clCreateBuffer(q6 conv weights)");
    conv_state_buffer = api.clCreateBuffer(
        context, kClMemReadOnly, conv_state.size() * sizeof(float), nullptr, &err);
    Check(err, "clCreateBuffer(q6 conv state)");
    conv_output_buffer = api.clCreateBuffer(
        context, kClMemWriteOnly, run.conv_output_raw.size() * sizeof(float), nullptr, &err);
    Check(err, "clCreateBuffer(q6 conv output)");
    next_state_buffer = api.clCreateBuffer(
        context, kClMemWriteOnly, run.conv_state.size() * sizeof(float), nullptr, &err);
    Check(err, "clCreateBuffer(q6 next conv state)");
    Check(api.clEnqueueWriteBuffer(queue, q6_buffer, kClTrue, 0,
                                   q6_raw.size(), q6_raw.data(), 0, nullptr, nullptr),
          "clEnqueueWriteBuffer(q6 raw)");
    Check(api.clEnqueueWriteBuffer(queue, q8_qs_buffer, kClTrue, 0,
                                   q8_qs.size() * sizeof(std::int8_t), q8_qs.data(),
                                   0, nullptr, nullptr),
          "clEnqueueWriteBuffer(q6 q8 qs)");
    Check(api.clEnqueueWriteBuffer(queue, q8_d_buffer, kClTrue, 0,
                                   q8_d.size() * sizeof(float), q8_d.data(),
                                   0, nullptr, nullptr),
          "clEnqueueWriteBuffer(q6 q8 d)");
    Check(api.clEnqueueWriteBuffer(queue, conv_weight_buffer, kClTrue, 0,
                                   conv_weights.size() * sizeof(float), conv_weights.data(),
                                   0, nullptr, nullptr),
          "clEnqueueWriteBuffer(q6 conv weights)");
    Check(api.clEnqueueWriteBuffer(queue, conv_state_buffer, kClTrue, 0,
                                   conv_state.size() * sizeof(float), conv_state.data(),
                                   0, nullptr, nullptr),
          "clEnqueueWriteBuffer(q6 conv state)");
    const cl_uint rows_arg = static_cast<cl_uint>(rows);
    const cl_uint blocks_arg = static_cast<cl_uint>(blocks_per_row);
    const cl_uint conv_kernel_arg = static_cast<cl_uint>(conv_kernel_size);
    Check(api.clSetKernelArg(q6_kernel, 0, sizeof(q6_buffer), &q6_buffer),
          "clSetKernelArg(q6 0)");
    Check(api.clSetKernelArg(q6_kernel, 1, sizeof(q8_qs_buffer), &q8_qs_buffer),
          "clSetKernelArg(q6 1)");
    Check(api.clSetKernelArg(q6_kernel, 2, sizeof(q8_d_buffer), &q8_d_buffer),
          "clSetKernelArg(q6 2)");
    Check(api.clSetKernelArg(q6_kernel, 3, sizeof(rows_arg), &rows_arg),
          "clSetKernelArg(q6 3)");
    Check(api.clSetKernelArg(q6_kernel, 4, sizeof(blocks_arg), &blocks_arg),
          "clSetKernelArg(q6 4)");
    Check(api.clSetKernelArg(q6_kernel, 5, sizeof(qkv_buffer), &qkv_buffer),
          "clSetKernelArg(q6 5)");
    Check(api.clSetKernelArg(conv_kernel, 0, sizeof(qkv_buffer), &qkv_buffer),
          "clSetKernelArg(conv 0)");
    Check(api.clSetKernelArg(conv_kernel, 1, sizeof(conv_state_buffer), &conv_state_buffer),
          "clSetKernelArg(conv 1)");
    Check(api.clSetKernelArg(conv_kernel, 2, sizeof(conv_weight_buffer), &conv_weight_buffer),
          "clSetKernelArg(conv 2)");
    Check(api.clSetKernelArg(conv_kernel, 3, sizeof(rows_arg), &rows_arg),
          "clSetKernelArg(conv 3)");
    Check(api.clSetKernelArg(conv_kernel, 4, sizeof(conv_kernel_arg), &conv_kernel_arg),
          "clSetKernelArg(conv 4)");
    Check(api.clSetKernelArg(conv_kernel, 5, sizeof(conv_output_buffer), &conv_output_buffer),
          "clSetKernelArg(conv 5)");
    Check(api.clSetKernelArg(conv_kernel, 6, sizeof(next_state_buffer), &next_state_buffer),
          "clSetKernelArg(conv 6)");
    const std::size_t global = static_cast<std::size_t>(rows);
    run.qkv_min_us = std::numeric_limits<double>::infinity();
    run.conv_min_us = std::numeric_limits<double>::infinity();
    for (int i = 0; i < repeat; ++i) {
      cl_event q6_event = nullptr;
      cl_event conv_event = nullptr;
      Check(api.clEnqueueNDRangeKernel(queue, q6_kernel, 1, nullptr, &global, nullptr,
                                       0, nullptr, &q6_event),
            "clEnqueueNDRangeKernel(q6)");
      Check(api.clEnqueueNDRangeKernel(queue, conv_kernel, 1, nullptr, &global, nullptr,
                                       0, nullptr, &conv_event),
            "clEnqueueNDRangeKernel(conv)");
      Check(api.clFinish(queue), "clFinish(q6 preconv)");
      const double q6_us = EventUs(api, q6_event);
      const double conv_us = EventUs(api, conv_event);
      run.qkv_min_us = std::min(run.qkv_min_us, q6_us);
      run.qkv_mean_us += q6_us;
      run.conv_min_us = std::min(run.conv_min_us, conv_us);
      run.conv_mean_us += conv_us;
      api.clReleaseEvent(q6_event);
      api.clReleaseEvent(conv_event);
    }
    run.qkv_mean_us /= static_cast<double>(repeat);
    run.conv_mean_us /= static_cast<double>(repeat);
    run.qkv_effective_packed_gb_s =
        static_cast<double>(q6_raw.size()) / run.qkv_min_us / 1000.0;
    Check(api.clEnqueueReadBuffer(queue, qkv_buffer, kClTrue, 0,
                                  run.qkv_mixed.size() * sizeof(float),
                                  run.qkv_mixed.data(), 0, nullptr, nullptr),
          "clEnqueueReadBuffer(q6 qkv)");
    Check(api.clEnqueueReadBuffer(queue, conv_output_buffer, kClTrue, 0,
                                  run.conv_output_raw.size() * sizeof(float),
                                  run.conv_output_raw.data(), 0, nullptr, nullptr),
          "clEnqueueReadBuffer(q6 conv output)");
    Check(api.clEnqueueReadBuffer(queue, next_state_buffer, kClTrue, 0,
                                  run.conv_state.size() * sizeof(float),
                                  run.conv_state.data(), 0, nullptr, nullptr),
          "clEnqueueReadBuffer(q6 conv state)");
  } catch (...) {
    ReleaseMem(api, &next_state_buffer);
    ReleaseMem(api, &conv_output_buffer);
    ReleaseMem(api, &conv_state_buffer);
    ReleaseMem(api, &conv_weight_buffer);
    ReleaseMem(api, &qkv_buffer);
    ReleaseMem(api, &q8_d_buffer);
    ReleaseMem(api, &q8_qs_buffer);
    ReleaseMem(api, &q6_buffer);
    api.clReleaseKernel(conv_kernel);
    api.clReleaseKernel(q6_kernel);
    api.clReleaseProgram(program);
    api.clReleaseCommandQueue(queue);
    api.clReleaseContext(context);
    throw;
  }
  ReleaseMem(api, &next_state_buffer);
  ReleaseMem(api, &conv_output_buffer);
  ReleaseMem(api, &conv_state_buffer);
  ReleaseMem(api, &conv_weight_buffer);
  ReleaseMem(api, &qkv_buffer);
  ReleaseMem(api, &q8_d_buffer);
  ReleaseMem(api, &q8_qs_buffer);
  ReleaseMem(api, &q6_buffer);
  api.clReleaseKernel(conv_kernel);
  api.clReleaseKernel(q6_kernel);
  api.clReleaseProgram(program);
  api.clReleaseCommandQueue(queue);
  api.clReleaseContext(context);
  return run;
}

int main(int argc, char** argv) {
  try {
    const Args args = ParseArgs(argc, argv);
    const auto index = iq36::parse_gguf_model_index(args.model_path);
    const auto load_map = iq36::validate_qwen36_load_map(index);

    const auto oracle_attn_norm =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "attn_norm.bin"));
    const auto oracle_qkv =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "linear_attn_qkv_mixed.bin"));
    const auto oracle_conv =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "conv_output_raw.bin"));
    const auto captured_conv_state =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "conv_state.bin"));
    Require(oracle_attn_norm.size() == kHiddenSize, "oracle attn_norm size mismatch");
    Require(oracle_qkv.size() == kQkvMixedSize, "oracle qkv size mismatch");
    Require(oracle_conv.size() == kQkvMixedSize, "oracle conv size mismatch");
    Require(captured_conv_state.size() == kConvStateSize, "captured conv state size mismatch");

    const auto cpu_preconv = iq36::run_qwen36_linear_attention_preconv_core(
        args.model_path, index, args.layer, oracle_attn_norm);
    const auto cpu_conv = iq36::run_qwen36_linear_attention_conv_core(
        args.model_path, index, args.layer, cpu_preconv.qkv_mixed, captured_conv_state);

    const auto* qkv_tensor = iq36::find_tensor(index, LayerTensorName(args.layer, "attn_qkv.weight"));
    const auto* conv_tensor = iq36::find_tensor(index, LayerTensorName(args.layer, "ssm_conv1d.weight"));
    Require(qkv_tensor != nullptr && conv_tensor != nullptr, "required tensors missing");
    Require(qkv_tensor->type == 12 || qkv_tensor->type == 14, "qkv tensor is not Q4_K or Q6_K");
    Require(qkv_tensor->dims == std::vector<std::uint64_t>{kHiddenSize, kQkvMixedSize},
            "qkv tensor shape mismatch");
    Require(conv_tensor->type == 0, "conv tensor is not F32");
    Require(conv_tensor->dims == std::vector<std::uint64_t>{kConvKernelSize, kQkvMixedSize},
            "conv tensor shape mismatch");

    std::ifstream model(args.model_path, std::ios::binary);
    Require(static_cast<bool>(model), "failed to open model");
    const auto qkv_raw = ReadTensorBytes(model, *qkv_tensor);
    const auto conv_weights = ReadF32Tensor(
        model, *conv_tensor, static_cast<std::size_t>(kQkvMixedSize * kConvKernelSize));
    const auto q8 = iq36::QuantizeQ8KInputPlanes(oracle_attn_norm);
    GpuConvProbeRun gpu;
    if (qkv_tensor->type == 12) {
      const auto qkv_packed = iq36::PackQ4Kx8(qkv_raw, kQkvMixedSize, kHiddenSize / 256);
      iq36::GpuQ4X8MatvecRunner runner(args.device_substring, kQ4X8OpenClSource);
      const auto q4_gpu = runner.RunThenConv(
          qkv_packed,
          q8.qs,
          q8.bsums,
          q8.d,
          conv_weights,
          captured_conv_state,
          kQkvMixedSize,
          kHiddenSize / 256,
          kConvKernelSize,
          args.repeat,
          iq36::GpuQ4X8KernelVariant::kRowlaneParallel);
      gpu.qkv_mixed = q4_gpu.qkv_mixed;
      gpu.conv_output_raw = q4_gpu.conv_output_raw;
      gpu.conv_state = q4_gpu.conv_state;
      gpu.qkv_min_us = q4_gpu.timing.matvec.min_us;
      gpu.qkv_mean_us = q4_gpu.timing.matvec.mean_us;
      gpu.qkv_effective_packed_gb_s = q4_gpu.timing.matvec.effective_packed_gb_s;
      gpu.conv_min_us = q4_gpu.timing.conv_min_us;
      gpu.conv_mean_us = q4_gpu.timing.conv_mean_us;
      gpu.platform_name = runner.platform_name();
      gpu.device_name = runner.device_name();
      gpu.build_log = runner.build_log();
      gpu.program_build_ms = runner.program_build_ms();
    } else {
      gpu = RunGpuQ6KThenConv(
          qkv_raw,
          q8.qs,
          q8.d,
          conv_weights,
          captured_conv_state,
          kQkvMixedSize,
          kHiddenSize / 256,
          kConvKernelSize,
          args.repeat,
          args.device_substring);
    }

    const auto qkv_cpu_vs_oracle =
        iq36::compare_vectors(cpu_preconv.qkv_mixed, oracle_qkv, 5e-3);
    const auto qkv_gpu_vs_cpu =
        iq36::compare_vectors(gpu.qkv_mixed, cpu_preconv.qkv_mixed, 5e-3);
    const auto qkv_gpu_vs_oracle =
        iq36::compare_vectors(gpu.qkv_mixed, oracle_qkv, 5e-3);
    const auto conv_cpu_vs_oracle =
        iq36::compare_vectors(cpu_conv.conv_output_raw, oracle_conv, 5e-3);
    const auto conv_gpu_vs_cpu =
        iq36::compare_vectors(gpu.conv_output_raw, cpu_conv.conv_output_raw, 5e-3);
    const auto conv_gpu_vs_oracle =
        iq36::compare_vectors(gpu.conv_output_raw, oracle_conv, 5e-3);
    const auto conv_state_gpu_vs_cpu =
        iq36::compare_vectors(gpu.conv_state, cpu_conv.conv_state, 5e-3);

    const bool qkv_matches =
        ComparePassed(qkv_cpu_vs_oracle) &&
        ComparePassed(qkv_gpu_vs_cpu) &&
        ComparePassed(qkv_gpu_vs_oracle);
    const bool captured_state_conv_matches_oracle =
        ComparePassed(conv_cpu_vs_oracle) &&
        ComparePassed(conv_gpu_vs_cpu) &&
        ComparePassed(conv_gpu_vs_oracle) &&
        ComparePassed(conv_state_gpu_vs_cpu);
    const bool timings_positive =
        gpu.qkv_min_us > 0.0 && gpu.conv_min_us > 0.0;
    const bool checks_passed =
        load_map.ready &&
        gpu.device_name.find(args.device_substring) != std::string::npos &&
        qkv_matches &&
        captured_state_conv_matches_oracle &&
        timings_positive;

    std::cout << std::setprecision(10);
    std::cout << "{";
    std::cout << "\"schema_version\":\"intel-qwen36-gpu-conv-history-state-capture-probe-v0\",";
    std::cout << "\"model_path\":\"" << JsonEscape(args.model_path) << "\",";
    std::cout << "\"layer\":" << args.layer << ",";
    std::cout << "\"qkv_tensor_type\":\"" << JsonEscape(iq36::ggml_type_name(qkv_tensor->type)) << "\",";
    std::cout << "\"platform_name\":\"" << JsonEscape(gpu.platform_name) << "\",";
    std::cout << "\"device_name\":\"" << JsonEscape(gpu.device_name) << "\",";
    std::cout << "\"program_build_ms\":" << gpu.program_build_ms << ",";
    std::cout << "\"build_log\":\"" << JsonEscape(gpu.build_log) << "\",";
    std::cout << "\"repeat\":" << args.repeat << ",";
    std::cout << "\"captured_conv_state_values\":" << captured_conv_state.size() << ",";
    std::cout << "\"timings\":{";
    std::cout << "\"qkv_gpu_kernel_min_us\":" << gpu.qkv_min_us << ",";
    std::cout << "\"qkv_gpu_kernel_mean_us\":" << gpu.qkv_mean_us << ",";
    std::cout << "\"qkv_gpu_effective_packed_gb_s\":" << gpu.qkv_effective_packed_gb_s << ",";
    std::cout << "\"conv_gpu_kernel_min_us\":" << gpu.conv_min_us << ",";
    std::cout << "\"conv_gpu_kernel_mean_us\":" << gpu.conv_mean_us;
    std::cout << "},\"comparisons\":{";
    std::cout << "\"linear_attn_qkv_mixed\":{";
    std::cout << "\"cpu_vs_oracle\":";
    WriteCompare(qkv_cpu_vs_oracle);
    std::cout << ",\"gpu_vs_cpu\":";
    WriteCompare(qkv_gpu_vs_cpu);
    std::cout << ",\"gpu_vs_oracle\":";
    WriteCompare(qkv_gpu_vs_oracle);
    std::cout << "},\"conv_output_raw\":{";
    std::cout << "\"cpu_captured_state_vs_oracle\":";
    WriteCompare(conv_cpu_vs_oracle);
    std::cout << ",\"gpu_vs_cpu_captured_state\":";
    WriteCompare(conv_gpu_vs_cpu);
    std::cout << ",\"gpu_captured_state_vs_oracle\":";
    WriteCompare(conv_gpu_vs_oracle);
    std::cout << "},\"conv_state_after\":{";
    std::cout << "\"gpu_vs_cpu_captured_state\":";
    WriteCompare(conv_state_gpu_vs_cpu);
    std::cout << "}";
    std::cout << "},\"checks\":{";
    std::cout << "\"load_map_ready\":" << (load_map.ready ? "true" : "false") << ",";
    std::cout << "\"arc_device_selected\":" << (gpu.device_name.find(args.device_substring) != std::string::npos ? "true" : "false") << ",";
    std::cout << "\"captured_conv_state_input_boundary\":true,";
    std::cout << "\"qkv_matches_oracle\":" << (qkv_matches ? "true" : "false") << ",";
    std::cout << "\"captured_state_conv_matches_oracle\":" << (captured_state_conv_matches_oracle ? "true" : "false") << ",";
    std::cout << "\"gpu_event_timing_positive\":" << (timings_positive ? "true" : "false") << ",";
    std::cout << "\"speedup_claims_allowed\":false";
    std::cout << "},\"required_checks_passed\":" << (checks_passed ? "true" : "false");
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
  parser.add_argument("--env-script", default=DEFAULT_ENV_SCRIPT)
  parser.add_argument("--staging-root", "--remote-root", dest="remote_root", metavar="PATH", default=DEFAULT_REMOTE_ROOT)
  parser.add_argument("--oracle-bundle", type=Path, default=DEFAULT_ORACLE_BUNDLE)
  parser.add_argument("--layer", type=int, default=5)
  parser.add_argument("--repeat", type=int, default=7)
  parser.add_argument("--device-substring", default="B390")
  parser.add_argument("--timeout-s", type=int, default=1200)
  parser.add_argument("--capture-timeout-s", type=int, default=3600)
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


def load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise SystemExit(f"{path}: expected JSON object")
  return value


def rel(path: Path) -> str:
  return str(path.resolve().relative_to(ROOT))


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


def capture_conv_state(args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
  capture_dir = out_dir / "capture"
  capture_cmd = [
      "python3",
      str(CAPTURE_TOOL),
      "--host", args.host,
      "--filter", f"^attn_norm-{args.layer}$",
      "--filter", f"^linear_attn_qkv_mixed-{args.layer}$",
      "--filter", f"^conv_output_raw-{args.layer}$",
      "--filter", f"^conv_states.*-{args.layer}$",
      "--filter", f"^conv_state_.*-{args.layer}$",
      "--out-dir", str(capture_dir),
      "--timeout-s", str(args.capture_timeout_s),
  ]
  result = iq36_local.run(capture_cmd, args.capture_timeout_s + 120)
  return {
      "cmd": capture_cmd,
      "result": result,
      "capture_dir": capture_dir,
  }


def find_tensor(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
  for row in rows:
    if row.get("tensor_name") == name:
      return row
  raise SystemExit(f"capture output missing tensor {name}")


def capture_payloads(capture_dir: Path, layer: int) -> dict[str, dict[str, Any]]:
  remote_output = capture_dir / "remote-output"
  rows = iq36_local.load_jsonl(remote_output / "tensor-dumps.jsonl")
  wanted = {
      "attn_norm": (f"attn_norm-{layer}", "attn_norm.bin", 8192, [2048, 1, 1, 1]),
      "linear_attn_qkv_mixed": (
          f"linear_attn_qkv_mixed-{layer}", "linear_attn_qkv_mixed.bin", 32768, [8192, 1, 1, 1]),
      "conv_output_raw": (
          f"conv_output_raw-{layer}", "conv_output_raw.bin", 32768, [8192, 1, 1, 1]),
      "conv_state": (
          f"conv_states-{layer}", "conv_state.bin", 98304, [24576, 1, 1, 1]),
      "conv_state_reshaped": (
          f"conv_states_reshaped-{layer}", "conv_state_reshaped.bin", 98304, [3, 8192, 1, 1]),
  }
  payloads: dict[str, dict[str, Any]] = {}
  for key, (tensor_name, stage_name, size_bytes, ne) in wanted.items():
    row = find_tensor(rows, tensor_name)
    payload_path = remote_output / str(row.get("payload_path"))
    if not payload_path.exists():
      raise SystemExit(f"capture payload missing: {payload_path}")
    if payload_path.stat().st_size != size_bytes:
      raise SystemExit(f"capture payload size mismatch for {tensor_name}: {payload_path.stat().st_size}")
    if row.get("tensor_type") != "f32" or row.get("ne") != ne:
      raise SystemExit(f"capture tensor metadata mismatch for {tensor_name}")
    payloads[key] = {
        "local_path": payload_path.resolve(),
        "path": rel(payload_path),
        "sha256": iq36_local.sha256_file(payload_path),
        "size_bytes": size_bytes,
        "stage_name": stage_name,
        "tensor_name": tensor_name,
        "tensor_type": row.get("tensor_type"),
        "ne": row.get("ne"),
        "nb": row.get("nb"),
        "nbytes": row.get("nbytes"),
        "payload_path": row.get("payload_path"),
    }
  return payloads


def slim_payloads(payloads: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
  return {
      name: {key: value for key, value in payload.items() if key != "local_path"}
      for name, payload in payloads.items()
  }


def write_summary(path: Path, payload: dict[str, Any]) -> None:
  probe = payload.get("probe") if isinstance(payload.get("probe"), dict) else {}
  timings = probe.get("timings", {}) if isinstance(probe, dict) else {}
  comparisons = probe.get("comparisons", {}) if isinstance(probe, dict) else {}
  qkv = comparisons.get("linear_attn_qkv_mixed", {}) if isinstance(comparisons, dict) else {}
  conv = comparisons.get("conv_output_raw", {}) if isinstance(comparisons, dict) else {}
  lines = [
      "# GPU Conv-History State Capture Probe",
      "",
      f"- workstream: `{WORKSTREAM}`",
      f"- host: `{payload.get('host')}`",
      f"- model: `{payload.get('model')}`",
      f"- layer: `{payload.get('layer')}`",
      f"- required checks passed: `{str(payload.get('required_checks_passed')).lower()}`",
      "- speedup claims allowed: `false`",
      f"- capture artifact: `{payload.get('capture_artifact')}`",
      f"- platform/device: `{probe.get('platform_name')}` / `{probe.get('device_name')}`",
      "",
      "| boundary | min us | max abs | RMSE |",
      "|---|---:|---:|---:|",
      "| qkv GPU vs oracle | "
      f"{timings.get('qkv_gpu_kernel_min_us')} | "
      f"{qkv.get('gpu_vs_oracle', {}).get('max_abs_diff')} | "
      f"{qkv.get('gpu_vs_oracle', {}).get('rmse')} |",
      "| conv GPU captured-state vs oracle | "
      f"{timings.get('conv_gpu_kernel_min_us')} | "
      f"{conv.get('gpu_captured_state_vs_oracle', {}).get('max_abs_diff')} | "
      f"{conv.get('gpu_captured_state_vs_oracle', {}).get('rmse')} |",
      "",
      "The capture uses llama.cpp `build_conv_state` tensor `conv_states` before",
      f"the layer-{payload.get('layer')} convolution input is formed. The probe then feeds that captured",
      "pre-token state to the GPU Q4/Q6 QKV + F32 conv path and compares",
      f"`conv_output_raw-{payload.get('layer')}` against the captured oracle tensor.",
      "This is captured single-layer evidence only; it does not prove decode or",
      "model throughput.",
      "",
  ]
  path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  args = parse_args()
  stamp = utc_stamp()
  created_at = iso_now()
  out_dir = args.out_dir or ROOT / f"output/gpu-conv-history-state-capture-probe-{stamp}"
  out_dir.mkdir(parents=True, exist_ok=False)
  raw_dir = out_dir / "raw"
  raw_dir.mkdir()

  capture = capture_conv_state(args, out_dir)
  iq36_local.write_json(raw_dir / "capture-command.json", {
      "cmd": capture["cmd"],
      "result": capture["result"],
      "capture_dir": rel(capture["capture_dir"]),
  })
  capture_dir = capture["capture_dir"]
  capture_correctness = (
      load_json(capture_dir / "correctness.json")
      if (capture_dir / "correctness.json").exists()
      else {}
  )
  capture_run = (
      load_json(capture_dir / "capture-run.json")
      if (capture_dir / "capture-run.json").exists()
      else {}
  )
  payloads = (
      capture_payloads(capture_dir, args.layer)
      if capture_correctness.get("required_checks_passed") is True
      else {}
  )

  opencl_source = OPENCL_SOURCE.read_text(encoding="utf-8")
  opencl_source_hash = iq36_local.sha256_file(OPENCL_SOURCE)
  local_cpp = out_dir / "gpu_conv_history_state_capture_probe.cpp"
  local_cpp.write_text(
      PROBE_CPP.replace("@@OPENCL_SOURCE_LITERAL@@", cpp_raw_string_literal(opencl_source)),
      encoding="utf-8",
  )

  remote_dir = f"{args.remote_root.rstrip('/')}/gpu-conv-history-state-capture-probe-{stamp}"
  setup = iq36_local.run_target(
      args.host,
      "rm -rf " + shlex.quote(remote_dir) + " && mkdir -p "
      + " ".join(
          shlex.quote(f"{remote_dir}/{subdir}")
          for subdir in ("include/intel_qwen36", "src", "tests", "build", "oracle")
      ),
      args.timeout_s,
  )
  transfers: list[dict[str, Any]] = []
  payload_transfers: dict[str, dict[str, Any]] = {
      name: {"returncode": 1, "stdout": "", "stderr": "stage failed"}
      for name in payloads
      if name != "conv_state_reshaped"
  }
  remote_payload_dir = f"{remote_dir}/oracle"
  if setup.get("returncode") == 0 and payloads:
    for local, remote in SOURCE_FILES:
      transfers.append(iq36_local.copy_to(args.host, ROOT / local, f"{remote_dir}/{remote}", args.timeout_s))
    transfers.append(iq36_local.copy_to(args.host, local_cpp, f"{remote_dir}/tests/gpu_conv_history_state_capture_probe.cpp", args.timeout_s))
    for name, payload in payloads.items():
      if name == "conv_state_reshaped":
        continue
      payload_transfers[name] = iq36_local.copy_to(
          args.host,
          payload["local_path"],
          f"{remote_payload_dir}/{payload['stage_name']}",
          args.timeout_s,
      )

  compile_cmd = " && ".join([
      f"source {shlex.quote(args.env_script)} >/dev/null 2>&1",
      (
          "g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic "
          f"-I {shlex.quote(remote_dir + '/include')} "
          f"{shlex.quote(remote_dir + '/src/gguf_loader.cpp')} "
          f"{shlex.quote(remote_dir + '/src/gpu_q4x8_matvec.cpp')} "
          f"{shlex.quote(remote_dir + '/tests/gpu_conv_history_state_capture_probe.cpp')} "
          "-ldl -pthread "
          f"-o {shlex.quote(remote_dir + '/build/iq36-gpu-conv-history-state-capture-probe')}"
      ),
  ])
  stage_ok = (
      capture_correctness.get("required_checks_passed") is True
      and bool(payloads)
      and setup.get("returncode") == 0
      and transfers
      and all(item.get("returncode") == 0 for item in transfers)
      and all(item.get("returncode") == 0 for item in payload_transfers.values())
  )
  compile_result = (
      iq36_local.run_target(args.host, compile_cmd, args.timeout_s)
      if stage_ok
      else {"cmd": ["stage"], "returncode": 1, "stdout": "", "stderr": "stage failed"}
  )
  run_argv = [
      f"{remote_dir}/build/iq36-gpu-conv-history-state-capture-probe",
      "--model", args.model,
      "--payload-dir", remote_payload_dir,
      "--layer", str(args.layer),
      "--repeat", str(args.repeat),
      "--device-substring", args.device_substring,
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
  iq36_local.write_json(raw_dir / "payload-transfers.json", payload_transfers)
  iq36_local.write_json(raw_dir / "compile.json", compile_result)
  iq36_local.write_json(raw_dir / "run.json", run_result)
  if probe is not None:
    iq36_local.write_json(out_dir / "probe-result.json", probe)

  conv_state_payload = payloads.get("conv_state", {})
  conv_state_reshaped_payload = payloads.get("conv_state_reshaped", {})
  checks = [
      {"name": "boundary_capture_succeeded", "pass": capture_correctness.get("required_checks_passed") is True},
      {"name": "captured_conv_state_present", "pass": bool(conv_state_payload)},
      {
          "name": "captured_conv_state_shape_ok",
          "pass": conv_state_payload.get("size_bytes") == 98304
          and conv_state_payload.get("ne") == [24576, 1, 1, 1],
      },
      {
          "name": "captured_conv_state_reshape_same_payload",
          "pass": conv_state_payload.get("sha256") == conv_state_reshaped_payload.get("sha256"),
      },
      {"name": "remote_dir_created", "pass": setup.get("returncode") == 0},
      {"name": "source_files_transferred", "pass": bool(transfers) and all(item.get("returncode") == 0 for item in transfers)},
      {"name": "capture_payloads_transferred", "pass": bool(payload_transfers) and all(item.get("returncode") == 0 for item in payload_transfers.values())},
      {"name": "probe_compiled", "pass": compile_result.get("returncode") == 0},
      {"name": "probe_stdout_json_parsed", "pass": isinstance(probe, dict)},
      {"name": "probe_process_succeeded", "pass": run_result.get("returncode") == 0},
      {"name": "arc_b390_selected", "pass": bool(probe and "B390" in str(probe.get("device_name", "")))},
      {"name": "qkv_matches_oracle", "pass": bool(probe and nested_bool(probe, "checks", "qkv_matches_oracle"))},
      {"name": "captured_state_conv_matches_oracle", "pass": bool(probe and nested_bool(probe, "checks", "captured_state_conv_matches_oracle"))},
      {"name": "gpu_event_timing_positive", "pass": bool(probe and nested_bool(probe, "checks", "gpu_event_timing_positive"))},
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
      "oracle_bundle": str(args.oracle_bundle.resolve().relative_to(ROOT)),
      "layer": args.layer,
      "repeat": args.repeat,
      "capture_artifact": rel(capture_dir),
      "capture_analysis": capture_run.get("capture_analysis", {}),
      "capture_source_boundary": {
          "llama_cpp_source": "src/models/delta-net-base.cpp",
          "function": "llm_build_delta_net_base::build_conv_state",
          "tensor": f"conv_states-{args.layer}",
          "meaning": "pre-token linear-attention convolution history state read before conv_input concat",
      },
      "payloads": slim_payloads(payloads),
      "engine_shim_header": "engine/include/intel_qwen36/gpu_q4x8_matvec.hpp",
      "engine_shim_source": "engine/src/gpu_q4x8_matvec.cpp",
      "opencl_source": str(OPENCL_SOURCE.relative_to(ROOT)),
      "opencl_source_sha256": opencl_source_hash,
      "probe": probe,
      "checks": checks,
      "required_checks_passed": required_checks_passed,
      "speedup_claims_allowed": False,
  }
  manifest = {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "created_at": created_at,
      "tool": "tools/intel-qwen36-gpu-conv-history-state-capture-probe.py",
      "artifact": str(out_dir),
      "host": args.host,
      "remote_dir": remote_dir,
      "layer": args.layer,
      "capture_artifact": rel(capture_dir),
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
  timings = aggregate.get("timings", {}) if isinstance(aggregate.get("timings"), dict) else {}
  comparisons = aggregate.get("comparisons", {}) if isinstance(aggregate.get("comparisons"), dict) else {}
  iq36_local.write_metric(
      out_dir / "metrics.jsonl",
      "gpu_conv_history_state_capture_probe",
      [
          ("required_checks_passed", required_checks_passed),
          ("captured_conv_state_value_count", 24576 if conv_state_payload else None),
          ("qkv_kernel_min_us", nested_number(timings, "qkv_gpu_kernel_min_us")),
          ("conv_kernel_min_us", nested_number(timings, "conv_gpu_kernel_min_us")),
          ("qkv_gpu_vs_oracle_max_abs_diff", nested_number(comparisons, "linear_attn_qkv_mixed", "gpu_vs_oracle", "max_abs_diff")),
          ("conv_cpu_captured_state_vs_oracle_max_abs_diff", nested_number(comparisons, "conv_output_raw", "cpu_captured_state_vs_oracle", "max_abs_diff")),
          ("conv_gpu_captured_state_vs_oracle_max_abs_diff", nested_number(comparisons, "conv_output_raw", "gpu_captured_state_vs_oracle", "max_abs_diff")),
          ("conv_gpu_vs_cpu_captured_state_max_abs_diff", nested_number(comparisons, "conv_output_raw", "gpu_vs_cpu_captured_state", "max_abs_diff")),
      ],
  )
  write_summary(out_dir / "summary.md", payload)
  print(out_dir)
  return 0 if required_checks_passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
