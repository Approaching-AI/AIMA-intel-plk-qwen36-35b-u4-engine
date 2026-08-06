#!/usr/bin/env python3
"""Run the GPU captured layer shell handoff probe."""

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
SCHEMA_VERSION = "intel-qwen36-gpu-captured-layer-shell-probe-v0"
DEFAULT_HOST = "local"
DEFAULT_MODEL = "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf"
DEFAULT_ENV_SCRIPT = "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh"
DEFAULT_REMOTE_ROOT = "/home/intel/intel-qwen36-gpu"
DEFAULT_ORACLE_BUNDLE = ROOT / "oracle/r0-oracle-bundle-20260627T060028Z"
PAYLOAD_ROOT = ROOT / "output/r0-boundary-capture-run-20260627T054024Z/remote-output/payloads"
OPENCL_SOURCE = ROOT / "engine/gpu/opencl/q4x8_matvec.cl"
SOURCE_FILES = [
    ("engine/include/intel_qwen36/gguf_loader.hpp", "include/intel_qwen36/gguf_loader.hpp"),
    ("engine/src/gguf_loader.cpp", "src/gguf_loader.cpp"),
]
PAYLOAD_SPECS = {
    "attn_residual": ("attn_residual.bin", "attn_residual-{layer}__tok15__ord208.bin", 8192),
    "attn_post_norm": ("attn_post_norm.bin", "attn_post_norm-{layer}__tok15__ord209.bin", 8192),
    "ffn_moe_weights_norm": (
        "ffn_moe_weights_norm.bin",
        "ffn_moe_weights_norm-{layer}__tok15__ord214.bin",
        32,
    ),
    "ffn_moe_down": ("ffn_moe_down.bin", "ffn_moe_down-{layer}__tok15__ord219.bin", 65536),
    "ffn_moe_weighted": (
        "ffn_moe_weighted.bin",
        "ffn_moe_weighted-{layer}__tok15__ord220.bin",
        65536,
    ),
    "ffn_moe_out": ("ffn_moe_out.bin", "ffn_moe_out-{layer}__tok15__ord221.bin", 8192),
    "ffn_shexp": ("ffn_shexp.bin", "ffn_shexp-{layer}__tok15__ord222.bin", 8192),
    "shared_gate": (
        "shared_expert_gate.bin",
        "shared_expert_gate-{layer}__tok15__ord223.bin",
        4,
    ),
    "shared_gate_sigmoid": (
        "shared_expert_gate_sigmoid.bin",
        "shared_expert_gate_sigmoid-{layer}__tok15__ord224.bin",
        4,
    ),
    "ffn_shexp_gated": (
        "ffn_shexp_gated.bin",
        "ffn_shexp_gated-{layer}__tok15__ord225.bin",
        8192,
    ),
    "ffn_out": ("ffn_out.bin", "ffn_out-{layer}__tok15__ord226.bin", 8192),
    "layer_output": ("layer_output.bin", "l_out-{layer}__tok15__ord227.bin", 8192),
}


PROBE_CPP = r'''
#include "intel_qwen36/gguf_loader.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

const char* kOpenClSource = @@OPENCL_SOURCE_LITERAL@@;

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

constexpr int kLayerCount = 40;
constexpr int kHiddenSize = 2048;
constexpr int kExpertUsedCount = 8;
constexpr int kWeightedValueCount = kHiddenSize * kExpertUsedCount;
constexpr int kSourceTokenPosition = 15;
constexpr double kMismatchThreshold = 5e-3;
constexpr double kMaxAbsDiffThreshold = 5e-3;
constexpr double kRmseThreshold = 5e-4;
constexpr double kMinCosine = 0.999;

struct Args {
  std::string model_path;
  std::string payload_dir;
  int layer = 5;
  int repeat = 11;
  std::string device_substring = "B390";
};

struct SelectedDevice {
  cl_platform_id platform = nullptr;
  cl_device_id device = nullptr;
  std::string platform_name;
  std::string device_name;
};

struct ShellTiming {
  double weighted_min_us = 0.0;
  double weighted_mean_us = 0.0;
  double shared_gate_matvec_min_us = 0.0;
  double shared_gate_matvec_mean_us = 0.0;
  double shared_gate_apply_min_us = 0.0;
  double shared_gate_apply_mean_us = 0.0;
  double ffn_output_add_min_us = 0.0;
  double ffn_output_add_mean_us = 0.0;
  double residual_add_min_us = 0.0;
  double residual_add_mean_us = 0.0;
  double shell_sum_min_us = 0.0;
  double shell_sum_mean_us = 0.0;
  std::uint64_t hidden_global_work_items = 0;
  std::uint64_t shared_gate_global_work_items = 0;
};

struct ShellRun {
  std::vector<float> weighted;
  std::vector<float> moe_out;
  std::vector<float> shared_gate;
  std::vector<float> shared_gate_sigmoid;
  std::vector<float> shared_gated;
  std::vector<float> ffn_out;
  std::vector<float> layer_output;
  ShellTiming timing;
  std::string platform_name;
  std::string device_name;
  std::string build_log;
  double program_build_ms = 0.0;
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

std::string PlatformString(OpenClApi& api, cl_platform_id platform, cl_platform_info info) {
  std::size_t size = 0;
  Check(api.clGetPlatformInfo(platform, info, 0, nullptr, &size), "clGetPlatformInfo(size)");
  std::string out(size, '\0');
  Check(api.clGetPlatformInfo(platform, info, size, out.data(), nullptr), "clGetPlatformInfo(value)");
  if (!out.empty() && out.back() == '\0') out.pop_back();
  return out;
}

std::string DeviceString(OpenClApi& api, cl_device_id device, cl_device_info info) {
  std::size_t size = 0;
  Check(api.clGetDeviceInfo(device, info, 0, nullptr, &size), "clGetDeviceInfo(size)");
  std::string out(size, '\0');
  Check(api.clGetDeviceInfo(device, info, size, out.data(), nullptr), "clGetDeviceInfo(value)");
  if (!out.empty() && out.back() == '\0') out.pop_back();
  return out;
}

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

std::string BuildLog(OpenClApi& api, cl_program program, cl_device_id device) {
  std::size_t size = 0;
  const cl_int status =
      api.clGetProgramBuildInfo(program, device, kClProgramBuildLog, 0, nullptr, &size);
  if (status != kClSuccess || size == 0) {
    return "";
  }
  std::string out(size, '\0');
  Check(api.clGetProgramBuildInfo(program, device, kClProgramBuildLog, size, out.data(), nullptr),
        "clGetProgramBuildInfo(log)");
  if (!out.empty() && out.back() == '\0') out.pop_back();
  return out;
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

void ReleaseMem(OpenClApi& api, cl_mem* mem) {
  if (*mem) {
    api.clReleaseMemObject(*mem);
    *mem = nullptr;
  }
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

std::vector<float> ReadF32TensorPayload(std::ifstream& in,
                                        const iq36::GgufTensorInfo& tensor,
                                        std::size_t expected_values) {
  Require(tensor.type == 0, "tensor is not F32: " + tensor.name);
  Require(tensor.nbytes == expected_values * sizeof(float), "F32 tensor byte size mismatch");
  const auto bytes = ReadTensorBytes(in, tensor);
  std::vector<float> values(expected_values, 0.0f);
  std::memcpy(values.data(), bytes.data(), bytes.size());
  return values;
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
  Require(args.layer >= 0 && args.layer < kLayerCount, "--layer is out of range");
  Require(args.repeat > 0, "--repeat must be positive");
  return args;
}

bool ComparePassed(const iq36::VectorCompareStats& stats) {
  return stats.same_size &&
         stats.finite &&
         stats.mismatch_count == 0 &&
         stats.max_abs_diff <= kMaxAbsDiffThreshold &&
         stats.rmse <= kRmseThreshold &&
         stats.cosine >= kMinCosine;
}

double Mean(const std::vector<double>& values) {
  return std::accumulate(values.begin(), values.end(), 0.0) /
         static_cast<double>(values.size());
}

double Min(const std::vector<double>& values) {
  return *std::min_element(values.begin(), values.end());
}

ShellRun RunGpuShell(const std::vector<float>& gate_weights,
                     const std::vector<float>& attn_post_norm,
                     const std::vector<float>& ffn_moe_down,
                     const std::vector<float>& weights_norm,
                     const std::vector<float>& ffn_shexp,
                     const std::vector<float>& attn_residual,
                     const std::string& device_substring,
                     int repeat) {
  Require(gate_weights.size() == kHiddenSize, "shared gate weight size mismatch");
  Require(attn_post_norm.size() == kHiddenSize, "attn_post_norm size mismatch");
  Require(ffn_moe_down.size() == kWeightedValueCount, "ffn_moe_down size mismatch");
  Require(weights_norm.size() == kExpertUsedCount, "weights_norm size mismatch");
  Require(ffn_shexp.size() == kHiddenSize, "ffn_shexp size mismatch");
  Require(attn_residual.size() == kHiddenSize, "attn_residual size mismatch");

  OpenClApi api;
  ShellRun run;
  run.weighted.assign(kWeightedValueCount, 0.0f);
  run.moe_out.assign(kHiddenSize, 0.0f);
  run.shared_gate.assign(1, 0.0f);
  run.shared_gate_sigmoid.assign(1, 0.0f);
  run.shared_gated.assign(kHiddenSize, 0.0f);
  run.ffn_out.assign(kHiddenSize, 0.0f);
  run.layer_output.assign(kHiddenSize, 0.0f);
  const auto selected = SelectDevice(api, device_substring);
  run.platform_name = selected.platform_name;
  run.device_name = selected.device_name;

  cl_int err = kClSuccess;
  cl_context context = api.clCreateContext(nullptr, 1, &selected.device, nullptr, nullptr, &err);
  Check(err, "clCreateContext");
  cl_command_queue queue = api.clCreateCommandQueue(context, selected.device, kClQueueProfilingEnable, &err);
  Check(err, "clCreateCommandQueue");
  const char* source = kOpenClSource;
  const std::size_t source_len = std::strlen(kOpenClSource);
  cl_program program = api.clCreateProgramWithSource(context, 1, &source, &source_len, &err);
  Check(err, "clCreateProgramWithSource");
  const auto build_begin = std::chrono::steady_clock::now();
  err = api.clBuildProgram(program, 1, &selected.device, "", nullptr, nullptr);
  const auto build_end = std::chrono::steady_clock::now();
  run.program_build_ms = std::chrono::duration<double, std::milli>(build_end - build_begin).count();
  run.build_log = BuildLog(api, program, selected.device);
  Check(err, "clBuildProgram");
  cl_kernel weighted_kernel = api.clCreateKernel(program, "ffn_moe_weighted_aggregate_f32", &err);
  Check(err, "clCreateKernel(ffn_moe_weighted_aggregate_f32)");
  cl_kernel gate_matvec_kernel = api.clCreateKernel(program, "f32_matvec_row_f32", &err);
  Check(err, "clCreateKernel(f32_matvec_row_f32)");
  cl_kernel gate_apply_kernel = api.clCreateKernel(program, "shared_expert_gate_apply_f32", &err);
  Check(err, "clCreateKernel(shared_expert_gate_apply_f32)");
  cl_kernel ffn_add_kernel = api.clCreateKernel(program, "ffn_output_add_f32", &err);
  Check(err, "clCreateKernel(ffn_output_add_f32)");
  cl_kernel residual_kernel = api.clCreateKernel(program, "post_ffn_residual_add_f32", &err);
  Check(err, "clCreateKernel(post_ffn_residual_add_f32)");

  cl_mem down_buffer = nullptr, weights_buffer = nullptr, weighted_buffer = nullptr;
  cl_mem moe_out_buffer = nullptr, gate_weights_buffer = nullptr, attn_post_norm_buffer = nullptr;
  cl_mem ffn_shexp_buffer = nullptr, shared_gate_buffer = nullptr, shared_sigmoid_buffer = nullptr;
  cl_mem shared_gated_buffer = nullptr, ffn_out_buffer = nullptr, attn_residual_buffer = nullptr;
  cl_mem layer_output_buffer = nullptr;
  try {
    auto make_read = [&](const std::vector<float>& values, const char* name) -> cl_mem {
      cl_int local_err = kClSuccess;
      cl_mem mem = api.clCreateBuffer(context, kClMemReadOnly,
                                      values.size() * sizeof(float), nullptr, &local_err);
      Check(local_err, std::string("clCreateBuffer(") + name + ")");
      Check(api.clEnqueueWriteBuffer(queue, mem, kClTrue, 0,
                                     values.size() * sizeof(float), values.data(),
                                     0, nullptr, nullptr),
            std::string("clEnqueueWriteBuffer(") + name + ")");
      return mem;
    };
    auto make_write = [&](std::size_t values, const char* name) -> cl_mem {
      cl_int local_err = kClSuccess;
      cl_mem mem = api.clCreateBuffer(context, kClMemWriteOnly,
                                      values * sizeof(float), nullptr, &local_err);
      Check(local_err, std::string("clCreateBuffer(") + name + ")");
      return mem;
    };

    down_buffer = make_read(ffn_moe_down, "down");
    weights_buffer = make_read(weights_norm, "weights");
    gate_weights_buffer = make_read(gate_weights, "gate_weights");
    attn_post_norm_buffer = make_read(attn_post_norm, "attn_post_norm");
    ffn_shexp_buffer = make_read(ffn_shexp, "ffn_shexp");
    attn_residual_buffer = make_read(attn_residual, "attn_residual");
    weighted_buffer = make_write(run.weighted.size(), "weighted");
    moe_out_buffer = make_write(run.moe_out.size(), "moe_out");
    shared_gate_buffer = make_write(run.shared_gate.size(), "shared_gate");
    shared_sigmoid_buffer = make_write(run.shared_gate_sigmoid.size(), "shared_sigmoid");
    shared_gated_buffer = make_write(run.shared_gated.size(), "shared_gated");
    ffn_out_buffer = make_write(run.ffn_out.size(), "ffn_out");
    layer_output_buffer = make_write(run.layer_output.size(), "layer_output");

    const cl_uint hidden_arg = kHiddenSize;
    const cl_uint expert_arg = kExpertUsedCount;
    const cl_uint rows_arg = 1;
    Check(api.clSetKernelArg(weighted_kernel, 0, sizeof(down_buffer), &down_buffer), "clSetKernelArg(weighted 0)");
    Check(api.clSetKernelArg(weighted_kernel, 1, sizeof(weights_buffer), &weights_buffer), "clSetKernelArg(weighted 1)");
    Check(api.clSetKernelArg(weighted_kernel, 2, sizeof(hidden_arg), &hidden_arg), "clSetKernelArg(weighted 2)");
    Check(api.clSetKernelArg(weighted_kernel, 3, sizeof(expert_arg), &expert_arg), "clSetKernelArg(weighted 3)");
    Check(api.clSetKernelArg(weighted_kernel, 4, sizeof(weighted_buffer), &weighted_buffer), "clSetKernelArg(weighted 4)");
    Check(api.clSetKernelArg(weighted_kernel, 5, sizeof(moe_out_buffer), &moe_out_buffer), "clSetKernelArg(weighted 5)");

    Check(api.clSetKernelArg(gate_matvec_kernel, 0, sizeof(gate_weights_buffer), &gate_weights_buffer), "clSetKernelArg(gate matvec 0)");
    Check(api.clSetKernelArg(gate_matvec_kernel, 1, sizeof(attn_post_norm_buffer), &attn_post_norm_buffer), "clSetKernelArg(gate matvec 1)");
    Check(api.clSetKernelArg(gate_matvec_kernel, 2, sizeof(hidden_arg), &hidden_arg), "clSetKernelArg(gate matvec 2)");
    Check(api.clSetKernelArg(gate_matvec_kernel, 3, sizeof(rows_arg), &rows_arg), "clSetKernelArg(gate matvec 3)");
    Check(api.clSetKernelArg(gate_matvec_kernel, 4, sizeof(shared_gate_buffer), &shared_gate_buffer), "clSetKernelArg(gate matvec 4)");

    Check(api.clSetKernelArg(gate_apply_kernel, 0, sizeof(ffn_shexp_buffer), &ffn_shexp_buffer), "clSetKernelArg(gate apply 0)");
    Check(api.clSetKernelArg(gate_apply_kernel, 1, sizeof(shared_gate_buffer), &shared_gate_buffer), "clSetKernelArg(gate apply 1)");
    Check(api.clSetKernelArg(gate_apply_kernel, 2, sizeof(hidden_arg), &hidden_arg), "clSetKernelArg(gate apply 2)");
    Check(api.clSetKernelArg(gate_apply_kernel, 3, sizeof(shared_sigmoid_buffer), &shared_sigmoid_buffer), "clSetKernelArg(gate apply 3)");
    Check(api.clSetKernelArg(gate_apply_kernel, 4, sizeof(shared_gated_buffer), &shared_gated_buffer), "clSetKernelArg(gate apply 4)");

    Check(api.clSetKernelArg(ffn_add_kernel, 0, sizeof(moe_out_buffer), &moe_out_buffer), "clSetKernelArg(ffn add 0)");
    Check(api.clSetKernelArg(ffn_add_kernel, 1, sizeof(shared_gated_buffer), &shared_gated_buffer), "clSetKernelArg(ffn add 1)");
    Check(api.clSetKernelArg(ffn_add_kernel, 2, sizeof(hidden_arg), &hidden_arg), "clSetKernelArg(ffn add 2)");
    Check(api.clSetKernelArg(ffn_add_kernel, 3, sizeof(ffn_out_buffer), &ffn_out_buffer), "clSetKernelArg(ffn add 3)");

    Check(api.clSetKernelArg(residual_kernel, 0, sizeof(attn_residual_buffer), &attn_residual_buffer), "clSetKernelArg(residual 0)");
    Check(api.clSetKernelArg(residual_kernel, 1, sizeof(ffn_out_buffer), &ffn_out_buffer), "clSetKernelArg(residual 1)");
    Check(api.clSetKernelArg(residual_kernel, 2, sizeof(hidden_arg), &hidden_arg), "clSetKernelArg(residual 2)");
    Check(api.clSetKernelArg(residual_kernel, 3, sizeof(layer_output_buffer), &layer_output_buffer), "clSetKernelArg(residual 3)");

    const std::size_t hidden_global = kHiddenSize;
    const std::size_t gate_global = 1;
    std::vector<double> weighted_times, gate_matvec_times, gate_apply_times;
    std::vector<double> ffn_add_times, residual_times, shell_sum_times;
    weighted_times.reserve(static_cast<std::size_t>(repeat));
    gate_matvec_times.reserve(static_cast<std::size_t>(repeat));
    gate_apply_times.reserve(static_cast<std::size_t>(repeat));
    ffn_add_times.reserve(static_cast<std::size_t>(repeat));
    residual_times.reserve(static_cast<std::size_t>(repeat));
    shell_sum_times.reserve(static_cast<std::size_t>(repeat));
    for (int i = 0; i < repeat; ++i) {
      cl_event weighted_event = nullptr;
      cl_event gate_matvec_event = nullptr;
      cl_event gate_apply_event = nullptr;
      cl_event ffn_add_event = nullptr;
      cl_event residual_event = nullptr;
      Check(api.clEnqueueNDRangeKernel(queue, weighted_kernel, 1, nullptr,
                                       &hidden_global, nullptr, 0, nullptr,
                                       &weighted_event),
            "clEnqueueNDRangeKernel(ffn_moe_weighted_aggregate_f32)");
      Check(api.clEnqueueNDRangeKernel(queue, gate_matvec_kernel, 1, nullptr,
                                       &gate_global, nullptr, 0, nullptr,
                                       &gate_matvec_event),
            "clEnqueueNDRangeKernel(f32_matvec_row_f32)");
      Check(api.clEnqueueNDRangeKernel(queue, gate_apply_kernel, 1, nullptr,
                                       &hidden_global, nullptr, 0, nullptr,
                                       &gate_apply_event),
            "clEnqueueNDRangeKernel(shared_expert_gate_apply_f32)");
      Check(api.clEnqueueNDRangeKernel(queue, ffn_add_kernel, 1, nullptr,
                                       &hidden_global, nullptr, 0, nullptr,
                                       &ffn_add_event),
            "clEnqueueNDRangeKernel(ffn_output_add_f32)");
      Check(api.clEnqueueNDRangeKernel(queue, residual_kernel, 1, nullptr,
                                       &hidden_global, nullptr, 0, nullptr,
                                       &residual_event),
            "clEnqueueNDRangeKernel(post_ffn_residual_add_f32)");
      Check(api.clFinish(queue), "clFinish(captured layer shell)");
      const double weighted_us = EventUs(api, weighted_event);
      const double gate_matvec_us = EventUs(api, gate_matvec_event);
      const double gate_apply_us = EventUs(api, gate_apply_event);
      const double ffn_add_us = EventUs(api, ffn_add_event);
      const double residual_us = EventUs(api, residual_event);
      weighted_times.push_back(weighted_us);
      gate_matvec_times.push_back(gate_matvec_us);
      gate_apply_times.push_back(gate_apply_us);
      ffn_add_times.push_back(ffn_add_us);
      residual_times.push_back(residual_us);
      shell_sum_times.push_back(weighted_us + gate_matvec_us + gate_apply_us +
                                ffn_add_us + residual_us);
      api.clReleaseEvent(weighted_event);
      api.clReleaseEvent(gate_matvec_event);
      api.clReleaseEvent(gate_apply_event);
      api.clReleaseEvent(ffn_add_event);
      api.clReleaseEvent(residual_event);
    }

    auto read = [&](cl_mem buffer, std::vector<float>& values, const char* name) {
      Check(api.clEnqueueReadBuffer(queue, buffer, kClTrue, 0,
                                    values.size() * sizeof(float), values.data(),
                                    0, nullptr, nullptr),
            std::string("clEnqueueReadBuffer(") + name + ")");
    };
    read(weighted_buffer, run.weighted, "weighted");
    read(moe_out_buffer, run.moe_out, "moe_out");
    read(shared_gate_buffer, run.shared_gate, "shared_gate");
    read(shared_sigmoid_buffer, run.shared_gate_sigmoid, "shared_sigmoid");
    read(shared_gated_buffer, run.shared_gated, "shared_gated");
    read(ffn_out_buffer, run.ffn_out, "ffn_out");
    read(layer_output_buffer, run.layer_output, "layer_output");

    run.timing.weighted_min_us = Min(weighted_times);
    run.timing.weighted_mean_us = Mean(weighted_times);
    run.timing.shared_gate_matvec_min_us = Min(gate_matvec_times);
    run.timing.shared_gate_matvec_mean_us = Mean(gate_matvec_times);
    run.timing.shared_gate_apply_min_us = Min(gate_apply_times);
    run.timing.shared_gate_apply_mean_us = Mean(gate_apply_times);
    run.timing.ffn_output_add_min_us = Min(ffn_add_times);
    run.timing.ffn_output_add_mean_us = Mean(ffn_add_times);
    run.timing.residual_add_min_us = Min(residual_times);
    run.timing.residual_add_mean_us = Mean(residual_times);
    run.timing.shell_sum_min_us = Min(shell_sum_times);
    run.timing.shell_sum_mean_us = Mean(shell_sum_times);
    run.timing.hidden_global_work_items = kHiddenSize;
    run.timing.shared_gate_global_work_items = 1;
  } catch (...) {
    ReleaseMem(api, &layer_output_buffer);
    ReleaseMem(api, &attn_residual_buffer);
    ReleaseMem(api, &ffn_out_buffer);
    ReleaseMem(api, &shared_gated_buffer);
    ReleaseMem(api, &shared_sigmoid_buffer);
    ReleaseMem(api, &shared_gate_buffer);
    ReleaseMem(api, &ffn_shexp_buffer);
    ReleaseMem(api, &attn_post_norm_buffer);
    ReleaseMem(api, &gate_weights_buffer);
    ReleaseMem(api, &moe_out_buffer);
    ReleaseMem(api, &weighted_buffer);
    ReleaseMem(api, &weights_buffer);
    ReleaseMem(api, &down_buffer);
    api.clReleaseKernel(residual_kernel);
    api.clReleaseKernel(ffn_add_kernel);
    api.clReleaseKernel(gate_apply_kernel);
    api.clReleaseKernel(gate_matvec_kernel);
    api.clReleaseKernel(weighted_kernel);
    api.clReleaseProgram(program);
    api.clReleaseCommandQueue(queue);
    api.clReleaseContext(context);
    throw;
  }
  ReleaseMem(api, &layer_output_buffer);
  ReleaseMem(api, &attn_residual_buffer);
  ReleaseMem(api, &ffn_out_buffer);
  ReleaseMem(api, &shared_gated_buffer);
  ReleaseMem(api, &shared_sigmoid_buffer);
  ReleaseMem(api, &shared_gate_buffer);
  ReleaseMem(api, &ffn_shexp_buffer);
  ReleaseMem(api, &attn_post_norm_buffer);
  ReleaseMem(api, &gate_weights_buffer);
  ReleaseMem(api, &moe_out_buffer);
  ReleaseMem(api, &weighted_buffer);
  ReleaseMem(api, &weights_buffer);
  ReleaseMem(api, &down_buffer);
  api.clReleaseKernel(residual_kernel);
  api.clReleaseKernel(ffn_add_kernel);
  api.clReleaseKernel(gate_apply_kernel);
  api.clReleaseKernel(gate_matvec_kernel);
  api.clReleaseKernel(weighted_kernel);
  api.clReleaseProgram(program);
  api.clReleaseCommandQueue(queue);
  api.clReleaseContext(context);
  return run;
}

void WriteCompare(const iq36::VectorCompareStats& stats) {
  std::cout << "{";
  std::cout << "\"compared_value_count\":" << stats.compared_value_count << ",";
  std::cout << "\"cosine\":" << std::setprecision(10) << stats.cosine << ",";
  std::cout << "\"finite\":" << (stats.finite ? "true" : "false") << ",";
  std::cout << "\"max_abs_diff\":" << std::setprecision(10) << stats.max_abs_diff << ",";
  std::cout << "\"mean_abs_diff\":" << std::setprecision(10) << stats.mean_abs_diff << ",";
  std::cout << "\"mismatch_count\":" << stats.mismatch_count << ",";
  std::cout << "\"rmse\":" << std::setprecision(10) << stats.rmse << ",";
  std::cout << "\"same_size\":" << (stats.same_size ? "true" : "false");
  std::cout << "}";
}

void WriteCompareGroup(const iq36::VectorCompareStats& cpu_vs_oracle,
                       const iq36::VectorCompareStats& gpu_vs_cpu,
                       const iq36::VectorCompareStats& gpu_vs_oracle) {
  std::cout << "{";
  std::cout << "\"cpu_vs_oracle\":";
  WriteCompare(cpu_vs_oracle);
  std::cout << ",\"gpu_vs_cpu\":";
  WriteCompare(gpu_vs_cpu);
  std::cout << ",\"gpu_vs_oracle\":";
  WriteCompare(gpu_vs_oracle);
  std::cout << "}";
}

int main(int argc, char** argv) {
  try {
    const Args args = ParseArgs(argc, argv);
    const auto index = iq36::parse_gguf_model_index(args.model_path);
    const auto load_map = iq36::validate_qwen36_load_map(index);
    const std::string gate_tensor_name =
        LayerTensorName(args.layer, "ffn_gate_inp_shexp.weight");
    const auto* gate_tensor = iq36::find_tensor(index, gate_tensor_name);
    Require(gate_tensor != nullptr, "shared expert gate tensor missing");
    const bool gate_tensor_shape_ok =
        gate_tensor->type == 0 &&
        gate_tensor->dims == std::vector<std::uint64_t>{kHiddenSize};

    const auto attn_residual =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "attn_residual.bin"));
    const auto attn_post_norm =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "attn_post_norm.bin"));
    const auto ffn_moe_down =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "ffn_moe_down.bin"));
    const auto weights_norm =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "ffn_moe_weights_norm.bin"));
    const auto oracle_weighted =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "ffn_moe_weighted.bin"));
    const auto oracle_moe_out =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "ffn_moe_out.bin"));
    const auto ffn_shexp =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "ffn_shexp.bin"));
    const auto oracle_shared_gate =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "shared_expert_gate.bin"));
    const auto oracle_shared_sigmoid =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "shared_expert_gate_sigmoid.bin"));
    const auto oracle_shared_gated =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "ffn_shexp_gated.bin"));
    const auto oracle_ffn_out =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "ffn_out.bin"));
    const auto oracle_layer_output =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "layer_output.bin"));

    const bool payload_counts_ok =
        attn_residual.size() == kHiddenSize &&
        attn_post_norm.size() == kHiddenSize &&
        ffn_moe_down.size() == kWeightedValueCount &&
        weights_norm.size() == kExpertUsedCount &&
        oracle_weighted.size() == kWeightedValueCount &&
        oracle_moe_out.size() == kHiddenSize &&
        ffn_shexp.size() == kHiddenSize &&
        oracle_shared_gate.size() == 1 &&
        oracle_shared_sigmoid.size() == 1 &&
        oracle_shared_gated.size() == kHiddenSize &&
        oracle_ffn_out.size() == kHiddenSize &&
        oracle_layer_output.size() == kHiddenSize;
    Require(payload_counts_ok, "payload size mismatch");

    std::ifstream model(args.model_path, std::ios::binary);
    Require(static_cast<bool>(model), "model file could not be opened");
    const auto gate_weights =
        ReadF32TensorPayload(model, *gate_tensor, static_cast<std::size_t>(kHiddenSize));

    const auto native_weighted =
        iq36::apply_expert_weights(ffn_moe_down, weights_norm, kHiddenSize);
    const auto native_moe_out =
        iq36::aggregate_experts(native_weighted, kExpertUsedCount, kHiddenSize);
    const auto native_shared_gate =
        iq36::matvec_tensor(args.model_path, index, gate_tensor_name, attn_post_norm);
    Require(native_shared_gate.size() == 1, "native shared gate size mismatch");
    const std::vector<float> native_shared_sigmoid{
        iq36::sigmoid_scalar(native_shared_gate[0])};
    const auto native_shared_gated =
        iq36::multiply_by_scalar(ffn_shexp, native_shared_sigmoid[0]);
    const auto native_ffn_out =
        iq36::add_vectors(native_moe_out, native_shared_gated);
    const auto native_layer_output =
        iq36::add_vectors(attn_residual, native_ffn_out);

    const auto gpu = RunGpuShell(gate_weights, attn_post_norm, ffn_moe_down,
                                 weights_norm, ffn_shexp, attn_residual,
                                 args.device_substring, args.repeat);

    const auto weighted_cpu_vs_oracle =
        iq36::compare_vectors(native_weighted, oracle_weighted, kMismatchThreshold);
    const auto weighted_gpu_vs_cpu =
        iq36::compare_vectors(gpu.weighted, native_weighted, kMismatchThreshold);
    const auto weighted_gpu_vs_oracle =
        iq36::compare_vectors(gpu.weighted, oracle_weighted, kMismatchThreshold);
    const auto moe_out_cpu_vs_oracle =
        iq36::compare_vectors(native_moe_out, oracle_moe_out, kMismatchThreshold);
    const auto moe_out_gpu_vs_cpu =
        iq36::compare_vectors(gpu.moe_out, native_moe_out, kMismatchThreshold);
    const auto moe_out_gpu_vs_oracle =
        iq36::compare_vectors(gpu.moe_out, oracle_moe_out, kMismatchThreshold);
    const auto shared_gate_cpu_vs_oracle =
        iq36::compare_vectors(native_shared_gate, oracle_shared_gate, kMismatchThreshold);
    const auto shared_gate_gpu_vs_cpu =
        iq36::compare_vectors(gpu.shared_gate, native_shared_gate, kMismatchThreshold);
    const auto shared_gate_gpu_vs_oracle =
        iq36::compare_vectors(gpu.shared_gate, oracle_shared_gate, kMismatchThreshold);
    const auto shared_sigmoid_cpu_vs_oracle =
        iq36::compare_vectors(native_shared_sigmoid, oracle_shared_sigmoid, kMismatchThreshold);
    const auto shared_sigmoid_gpu_vs_cpu =
        iq36::compare_vectors(gpu.shared_gate_sigmoid, native_shared_sigmoid, kMismatchThreshold);
    const auto shared_sigmoid_gpu_vs_oracle =
        iq36::compare_vectors(gpu.shared_gate_sigmoid, oracle_shared_sigmoid, kMismatchThreshold);
    const auto shared_gated_cpu_vs_oracle =
        iq36::compare_vectors(native_shared_gated, oracle_shared_gated, kMismatchThreshold);
    const auto shared_gated_gpu_vs_cpu =
        iq36::compare_vectors(gpu.shared_gated, native_shared_gated, kMismatchThreshold);
    const auto shared_gated_gpu_vs_oracle =
        iq36::compare_vectors(gpu.shared_gated, oracle_shared_gated, kMismatchThreshold);
    const auto ffn_out_cpu_vs_oracle =
        iq36::compare_vectors(native_ffn_out, oracle_ffn_out, kMismatchThreshold);
    const auto ffn_out_gpu_vs_cpu =
        iq36::compare_vectors(gpu.ffn_out, native_ffn_out, kMismatchThreshold);
    const auto ffn_out_gpu_vs_oracle =
        iq36::compare_vectors(gpu.ffn_out, oracle_ffn_out, kMismatchThreshold);
    const auto layer_cpu_vs_oracle =
        iq36::compare_vectors(native_layer_output, oracle_layer_output, kMismatchThreshold);
    const auto layer_gpu_vs_cpu =
        iq36::compare_vectors(gpu.layer_output, native_layer_output, kMismatchThreshold);
    const auto layer_gpu_vs_oracle =
        iq36::compare_vectors(gpu.layer_output, oracle_layer_output, kMismatchThreshold);

    const bool comparisons_passed =
        ComparePassed(weighted_cpu_vs_oracle) &&
        ComparePassed(weighted_gpu_vs_cpu) &&
        ComparePassed(weighted_gpu_vs_oracle) &&
        ComparePassed(moe_out_cpu_vs_oracle) &&
        ComparePassed(moe_out_gpu_vs_cpu) &&
        ComparePassed(moe_out_gpu_vs_oracle) &&
        ComparePassed(shared_gate_cpu_vs_oracle) &&
        ComparePassed(shared_gate_gpu_vs_cpu) &&
        ComparePassed(shared_gate_gpu_vs_oracle) &&
        ComparePassed(shared_sigmoid_cpu_vs_oracle) &&
        ComparePassed(shared_sigmoid_gpu_vs_cpu) &&
        ComparePassed(shared_sigmoid_gpu_vs_oracle) &&
        ComparePassed(shared_gated_cpu_vs_oracle) &&
        ComparePassed(shared_gated_gpu_vs_cpu) &&
        ComparePassed(shared_gated_gpu_vs_oracle) &&
        ComparePassed(ffn_out_cpu_vs_oracle) &&
        ComparePassed(ffn_out_gpu_vs_cpu) &&
        ComparePassed(ffn_out_gpu_vs_oracle) &&
        ComparePassed(layer_cpu_vs_oracle) &&
        ComparePassed(layer_gpu_vs_cpu) &&
        ComparePassed(layer_gpu_vs_oracle);
    const bool timing_positive =
        gpu.timing.weighted_min_us > 0.0 &&
        gpu.timing.shared_gate_matvec_min_us > 0.0 &&
        gpu.timing.shared_gate_apply_min_us > 0.0 &&
        gpu.timing.ffn_output_add_min_us > 0.0 &&
        gpu.timing.residual_add_min_us > 0.0 &&
        gpu.timing.shell_sum_min_us > 0.0;
    const bool arc_selected =
        gpu.device_name.find(args.device_substring) != std::string::npos;
    const bool required_checks_passed =
        load_map.ready &&
        gate_tensor_shape_ok &&
        payload_counts_ok &&
        arc_selected &&
        comparisons_passed &&
        timing_positive;

    std::cout << std::setprecision(10);
    std::cout << "{";
    std::cout << "\"schema_version\":\"intel-qwen36-gpu-captured-layer-shell-probe-v0\",";
    std::cout << "\"model_path\":\"" << JsonEscape(args.model_path) << "\",";
    std::cout << "\"layer\":" << args.layer << ",";
    std::cout << "\"source_token_position\":" << kSourceTokenPosition << ",";
    std::cout << "\"hidden_size\":" << kHiddenSize << ",";
    std::cout << "\"selected_expert_count\":" << kExpertUsedCount << ",";
    std::cout << "\"weighted_value_count\":" << kWeightedValueCount << ",";
    std::cout << "\"shared_gate_tensor_name\":\"" << JsonEscape(gate_tensor_name) << "\",";
    std::cout << "\"platform_name\":\"" << JsonEscape(gpu.platform_name) << "\",";
    std::cout << "\"device_name\":\"" << JsonEscape(gpu.device_name) << "\",";
    std::cout << "\"build_log\":\"" << JsonEscape(gpu.build_log) << "\",";
    std::cout << "\"program_build_ms\":" << gpu.program_build_ms << ",";
    std::cout << "\"repeat\":" << args.repeat << ",";
    std::cout << "\"timings\":{";
    std::cout << "\"moe_weighted_aggregate_min_us\":" << gpu.timing.weighted_min_us << ",";
    std::cout << "\"moe_weighted_aggregate_mean_us\":" << gpu.timing.weighted_mean_us << ",";
    std::cout << "\"shared_gate_matvec_min_us\":" << gpu.timing.shared_gate_matvec_min_us << ",";
    std::cout << "\"shared_gate_matvec_mean_us\":" << gpu.timing.shared_gate_matvec_mean_us << ",";
    std::cout << "\"shared_gate_apply_min_us\":" << gpu.timing.shared_gate_apply_min_us << ",";
    std::cout << "\"shared_gate_apply_mean_us\":" << gpu.timing.shared_gate_apply_mean_us << ",";
    std::cout << "\"ffn_output_add_min_us\":" << gpu.timing.ffn_output_add_min_us << ",";
    std::cout << "\"ffn_output_add_mean_us\":" << gpu.timing.ffn_output_add_mean_us << ",";
    std::cout << "\"post_ffn_residual_add_min_us\":" << gpu.timing.residual_add_min_us << ",";
    std::cout << "\"post_ffn_residual_add_mean_us\":" << gpu.timing.residual_add_mean_us << ",";
    std::cout << "\"captured_layer_shell_kernel_sum_min_us\":" << gpu.timing.shell_sum_min_us << ",";
    std::cout << "\"captured_layer_shell_kernel_sum_mean_us\":" << gpu.timing.shell_sum_mean_us << ",";
    std::cout << "\"hidden_global_work_items\":" << gpu.timing.hidden_global_work_items << ",";
    std::cout << "\"shared_gate_global_work_items\":" << gpu.timing.shared_gate_global_work_items;
    std::cout << "},";
    std::cout << "\"comparisons\":{";
    std::cout << "\"ffn_moe_weighted\":";
    WriteCompareGroup(weighted_cpu_vs_oracle, weighted_gpu_vs_cpu, weighted_gpu_vs_oracle);
    std::cout << ",\"ffn_moe_out\":";
    WriteCompareGroup(moe_out_cpu_vs_oracle, moe_out_gpu_vs_cpu, moe_out_gpu_vs_oracle);
    std::cout << ",\"shared_gate\":";
    WriteCompareGroup(shared_gate_cpu_vs_oracle, shared_gate_gpu_vs_cpu, shared_gate_gpu_vs_oracle);
    std::cout << ",\"shared_gate_sigmoid\":";
    WriteCompareGroup(shared_sigmoid_cpu_vs_oracle, shared_sigmoid_gpu_vs_cpu, shared_sigmoid_gpu_vs_oracle);
    std::cout << ",\"ffn_shexp_gated\":";
    WriteCompareGroup(shared_gated_cpu_vs_oracle, shared_gated_gpu_vs_cpu, shared_gated_gpu_vs_oracle);
    std::cout << ",\"ffn_out\":";
    WriteCompareGroup(ffn_out_cpu_vs_oracle, ffn_out_gpu_vs_cpu, ffn_out_gpu_vs_oracle);
    std::cout << ",\"layer_output\":";
    WriteCompareGroup(layer_cpu_vs_oracle, layer_gpu_vs_cpu, layer_gpu_vs_oracle);
    std::cout << "},";
    std::cout << "\"checks\":{";
    std::cout << "\"load_map_ready\":" << (load_map.ready ? "true" : "false") << ",";
    std::cout << "\"gate_tensor_shape_ok\":" << (gate_tensor_shape_ok ? "true" : "false") << ",";
    std::cout << "\"payload_counts_ok\":" << (payload_counts_ok ? "true" : "false") << ",";
    std::cout << "\"arc_device_selected\":" << (arc_selected ? "true" : "false") << ",";
    std::cout << "\"captured_layer_shell_matches_oracle\":"
              << (comparisons_passed ? "true" : "false") << ",";
    std::cout << "\"gpu_event_timing_positive\":" << (timing_positive ? "true" : "false") << ",";
    std::cout << "\"speedup_claims_forbidden\":true";
    std::cout << "},";
    std::cout << "\"required_checks_passed\":"
              << (required_checks_passed ? "true" : "false") << ",";
    std::cout << "\"speedup_claims_allowed\":false";
    std::cout << "}\n";
    return required_checks_passed ? 0 : 1;
  } catch (const std::exception& ex) {
    std::cerr << "error: " << ex.what() << "\n";
    return 2;
  }
}
'''


def utc_stamp() -> str:
  return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def iso_now() -> str:
  return dt.datetime.now(dt.timezone.utc).isoformat()


def cpp_raw_string_literal(value: str) -> str:
  delimiter = "IQ36_OPENCL"
  return f'R"{delimiter}({value}){delimiter}"'


def shell_join(argv: list[str]) -> str:
  return " ".join(shlex.quote(item) for item in argv)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--target", "--host", dest="host", metavar="TARGET", default=DEFAULT_HOST)
  parser.add_argument("--model", default=DEFAULT_MODEL)
  parser.add_argument("--env-script", default=DEFAULT_ENV_SCRIPT)
  parser.add_argument("--staging-root", "--remote-root", dest="remote_root", metavar="PATH", default=DEFAULT_REMOTE_ROOT)
  parser.add_argument("--oracle-bundle", type=Path, default=DEFAULT_ORACLE_BUNDLE)
  parser.add_argument("--layer", type=int, default=5)
  parser.add_argument("--repeat", type=int, default=11)
  parser.add_argument("--device-substring", default="B390")
  parser.add_argument("--out-dir", type=Path)
  parser.add_argument("--timeout-s", type=int, default=240)
  return parser.parse_args()


def resolve_payloads(layer: int) -> dict[str, dict[str, Any]]:
  payloads: dict[str, dict[str, Any]] = {}
  for name, (stage_name, pattern, expected_bytes) in PAYLOAD_SPECS.items():
    path = PAYLOAD_ROOT / pattern.format(layer=layer)
    if not path.exists():
      raise SystemExit(f"missing payload for {name}: {path}")
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
      raise SystemExit(
          f"payload byte mismatch for {name}: expected {expected_bytes}, got {actual_bytes}: {path}"
      )
    payloads[name] = {
        "stage_name": stage_name,
        "local_path": path,
        "source_name": path.name,
        "bytes": actual_bytes,
        "sha256": iq36_local.sha256_file(path),
    }
  return payloads


def parse_probe_stdout(stdout: str) -> dict[str, Any] | None:
  for line in reversed(stdout.splitlines()):
    line = line.strip()
    if line.startswith("{") and line.endswith("}"):
      return json.loads(line)
  return None


def nested_bool(obj: dict[str, Any], *keys: str) -> bool:
  current: Any = obj
  for key in keys:
    if not isinstance(current, dict):
      return False
    current = current.get(key)
  return bool(current)


def nested_number(obj: dict[str, Any], *keys: str) -> float | None:
  current: Any = obj
  for key in keys:
    if not isinstance(current, dict):
      return None
    current = current.get(key)
  return float(current) if isinstance(current, (int, float)) else None


def write_summary(path: Path, payload: dict[str, Any]) -> None:
  probe = payload.get("probe") if isinstance(payload.get("probe"), dict) else {}
  comparisons = probe.get("comparisons", {}) if isinstance(probe, dict) else {}
  timings = probe.get("timings", {}) if isinstance(probe, dict) else {}
  lines = [
      "# GPU Captured Layer Shell Probe",
      "",
      f"- workstream: `{WORKSTREAM}`",
      f"- host: `{payload.get('host')}`",
      f"- model: `{payload.get('model')}`",
      f"- layer: `{payload.get('layer')}`",
      f"- required checks passed: `{str(payload.get('required_checks_passed')).lower()}`",
      "- speedup claims allowed: `false`",
      f"- platform/device: `{probe.get('platform_name')}` / `{probe.get('device_name')}`",
      f"- hidden size: `{probe.get('hidden_size')}`",
      f"- shared gate tensor: `{probe.get('shared_gate_tensor_name')}`",
      "",
      "| output | comparison | max abs | RMSE |",
      "|---|---|---:|---:|",
  ]
  for name in (
      "ffn_moe_weighted",
      "ffn_moe_out",
      "shared_gate",
      "shared_gate_sigmoid",
      "ffn_shexp_gated",
      "ffn_out",
      "layer_output",
  ):
    group = comparisons.get(name, {}) if isinstance(comparisons, dict) else {}
    if not isinstance(group, dict):
      continue
    for lane in ("cpu_vs_oracle", "gpu_vs_cpu", "gpu_vs_oracle"):
      cmp = group.get(lane, {}) if isinstance(group.get(lane), dict) else {}
      lines.append(f"| {name} | {lane} | {cmp.get('max_abs_diff')} | {cmp.get('rmse')} |")
  lines += [
      "",
      "| kernel | min us | mean us |",
      "|---|---:|---:|",
      f"| moe_weighted_aggregate | {timings.get('moe_weighted_aggregate_min_us')} | {timings.get('moe_weighted_aggregate_mean_us')} |",
      f"| shared_gate_matvec | {timings.get('shared_gate_matvec_min_us')} | {timings.get('shared_gate_matvec_mean_us')} |",
      f"| shared_gate_apply | {timings.get('shared_gate_apply_min_us')} | {timings.get('shared_gate_apply_mean_us')} |",
      f"| ffn_output_add | {timings.get('ffn_output_add_min_us')} | {timings.get('ffn_output_add_mean_us')} |",
      f"| post_ffn_residual_add | {timings.get('post_ffn_residual_add_min_us')} | {timings.get('post_ffn_residual_add_mean_us')} |",
      f"| captured_layer_shell_sum | {timings.get('captured_layer_shell_kernel_sum_min_us')} | {timings.get('captured_layer_shell_kernel_sum_mean_us')} |",
      "",
      "The probe composes closed layer-5 GPU component kernels from captured",
      "intermediate boundaries and emits captured `l_out`. It is a single-layer",
      "component shell only; it does not prove prompt/token decode throughput.",
      "",
  ]
  path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  args = parse_args()
  stamp = utc_stamp()
  created_at = iso_now()
  out_dir = args.out_dir or ROOT / f"output/gpu-captured-layer-shell-probe-{stamp}"
  out_dir.mkdir(parents=True, exist_ok=False)
  raw_dir = out_dir / "raw"
  raw_dir.mkdir()
  payloads = resolve_payloads(args.layer)
  opencl_source = OPENCL_SOURCE.read_text(encoding="utf-8")
  opencl_source_hash = iq36_local.sha256_file(OPENCL_SOURCE)
  local_cpp = out_dir / "gpu_captured_layer_shell_probe.cpp"
  local_cpp.write_text(
      PROBE_CPP.replace("@@OPENCL_SOURCE_LITERAL@@", cpp_raw_string_literal(opencl_source)),
      encoding="utf-8",
  )

  remote_dir = f"{args.remote_root.rstrip('/')}/gpu-captured-layer-shell-probe-{stamp}"
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
  }
  remote_payload_dir = f"{remote_dir}/oracle"
  if setup.get("returncode") == 0:
    for local, remote in SOURCE_FILES:
      transfers.append(iq36_local.copy_to(args.host, ROOT / local, f"{remote_dir}/{remote}", args.timeout_s))
    transfers.append(iq36_local.copy_to(args.host, local_cpp, f"{remote_dir}/tests/gpu_captured_layer_shell_probe.cpp", args.timeout_s))
    for name, payload in payloads.items():
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
          f"{shlex.quote(remote_dir + '/tests/gpu_captured_layer_shell_probe.cpp')} "
          "-ldl -pthread "
          f"-o {shlex.quote(remote_dir + '/build/iq36-gpu-captured-layer-shell-probe')}"
      ),
  ])
  stage_ok = (
      setup.get("returncode") == 0
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
      f"{remote_dir}/build/iq36-gpu-captured-layer-shell-probe",
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

  checks = [
      {"name": "remote_dir_created", "pass": setup.get("returncode") == 0},
      {"name": "source_files_transferred", "pass": bool(transfers) and all(item.get("returncode") == 0 for item in transfers)},
      {"name": "oracle_payloads_transferred", "pass": all(item.get("returncode") == 0 for item in payload_transfers.values())},
      {"name": "probe_compiled", "pass": compile_result.get("returncode") == 0},
      {"name": "probe_stdout_json_parsed", "pass": isinstance(probe, dict)},
      {"name": "probe_process_succeeded", "pass": run_result.get("returncode") == 0},
      {"name": "arc_b390_selected", "pass": bool(probe and "B390" in str(probe.get("device_name", "")))},
      {"name": "captured_layer_shell_matches_oracle", "pass": bool(probe and nested_bool(probe, "checks", "captured_layer_shell_matches_oracle"))},
      {"name": "gpu_event_timing_positive", "pass": bool(probe and nested_bool(probe, "checks", "gpu_event_timing_positive"))},
      {"name": "speedup_claims_forbidden", "pass": True},
  ]
  required_checks_passed = all(item["pass"] for item in checks)
  slim_payloads = {
      name: {key: value for key, value in payload.items() if key != "local_path"}
      for name, payload in payloads.items()
  }
  payload = {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "created_at": created_at,
      "host": args.host,
      "remote_dir": remote_dir,
      "model": args.model,
      "oracle_bundle": str(args.oracle_bundle.resolve().relative_to(ROOT)),
      "payloads": slim_payloads,
      "layer": args.layer,
      "repeat": args.repeat,
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
      "tool": "tools/intel-qwen36-gpu-captured-layer-shell-probe.py",
      "artifact": str(out_dir),
      "host": args.host,
      "remote_dir": remote_dir,
      "layer": args.layer,
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
      "gpu_captured_layer_shell_probe",
      [
          ("required_checks_passed", required_checks_passed),
          ("captured_layer_shell_kernel_sum_min_us", nested_number(timings, "captured_layer_shell_kernel_sum_min_us")),
          ("layer_output_gpu_vs_oracle_max_abs_diff", nested_number(comparisons, "layer_output", "gpu_vs_oracle", "max_abs_diff")),
          ("ffn_out_gpu_vs_oracle_max_abs_diff", nested_number(comparisons, "ffn_out", "gpu_vs_oracle", "max_abs_diff")),
          ("ffn_moe_out_gpu_vs_oracle_max_abs_diff", nested_number(comparisons, "ffn_moe_out", "gpu_vs_oracle", "max_abs_diff")),
      ],
  )
  write_summary(out_dir / "summary.md", payload)
  print(out_dir)
  return 0 if required_checks_passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
