#!/usr/bin/env python3
"""Run the GPU shared expert gate handoff probe."""

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
SCHEMA_VERSION = "intel-qwen36-gpu-shared-expert-gate-probe-v0"
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
    "attn_post_norm": ("attn_post_norm.bin", "attn_post_norm-{layer}__tok15__ord209.bin", 8192),
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

struct SharedGateTiming {
  double gate_matvec_min_us = 0.0;
  double gate_matvec_mean_us = 0.0;
  double gate_matvec_effective_weight_gb_s = 0.0;
  double gate_apply_min_us = 0.0;
  double gate_apply_mean_us = 0.0;
  double gate_apply_effective_io_gb_s = 0.0;
  std::uint64_t gate_matvec_global_work_items = 0;
  std::uint64_t gate_apply_global_work_items = 0;
};

struct SharedGateRun {
  std::vector<float> shared_gate;
  std::vector<float> shared_gate_sigmoid;
  std::vector<float> shared_gated;
  SharedGateTiming timing;
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

SharedGateRun RunGpuSharedGate(const std::vector<float>& gate_weights,
                               const std::vector<float>& attn_post_norm,
                               const std::vector<float>& shared_down,
                               const std::string& device_substring,
                               int repeat) {
  Require(gate_weights.size() == kHiddenSize, "shared gate weight size mismatch");
  Require(attn_post_norm.size() == kHiddenSize, "attn post norm size mismatch");
  Require(shared_down.size() == kHiddenSize, "shared down size mismatch");
  OpenClApi api;
  SharedGateRun run;
  run.shared_gate.assign(1, 0.0f);
  run.shared_gate_sigmoid.assign(1, 0.0f);
  run.shared_gated.assign(kHiddenSize, 0.0f);
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
  cl_kernel matvec_kernel = api.clCreateKernel(program, "f32_matvec_row_f32", &err);
  Check(err, "clCreateKernel(f32_matvec_row_f32)");
  cl_kernel apply_kernel = api.clCreateKernel(program, "shared_expert_gate_apply_f32", &err);
  Check(err, "clCreateKernel(shared_expert_gate_apply_f32)");

  cl_mem weights_buffer = nullptr, input_buffer = nullptr, shared_down_buffer = nullptr;
  cl_mem gate_buffer = nullptr, sigmoid_buffer = nullptr, gated_buffer = nullptr;
  try {
    weights_buffer = api.clCreateBuffer(context, kClMemReadOnly,
                                        gate_weights.size() * sizeof(float), nullptr, &err);
    Check(err, "clCreateBuffer(weights)");
    input_buffer = api.clCreateBuffer(context, kClMemReadOnly,
                                      attn_post_norm.size() * sizeof(float), nullptr, &err);
    Check(err, "clCreateBuffer(input)");
    shared_down_buffer = api.clCreateBuffer(context, kClMemReadOnly,
                                            shared_down.size() * sizeof(float), nullptr, &err);
    Check(err, "clCreateBuffer(shared_down)");
    gate_buffer = api.clCreateBuffer(context, kClMemWriteOnly,
                                     run.shared_gate.size() * sizeof(float), nullptr, &err);
    Check(err, "clCreateBuffer(gate)");
    sigmoid_buffer = api.clCreateBuffer(context, kClMemWriteOnly,
                                        run.shared_gate_sigmoid.size() * sizeof(float), nullptr, &err);
    Check(err, "clCreateBuffer(sigmoid)");
    gated_buffer = api.clCreateBuffer(context, kClMemWriteOnly,
                                      run.shared_gated.size() * sizeof(float), nullptr, &err);
    Check(err, "clCreateBuffer(gated)");

    Check(api.clEnqueueWriteBuffer(queue, weights_buffer, kClTrue, 0,
                                   gate_weights.size() * sizeof(float),
                                   gate_weights.data(), 0, nullptr, nullptr),
          "clEnqueueWriteBuffer(weights)");
    Check(api.clEnqueueWriteBuffer(queue, input_buffer, kClTrue, 0,
                                   attn_post_norm.size() * sizeof(float),
                                   attn_post_norm.data(), 0, nullptr, nullptr),
          "clEnqueueWriteBuffer(input)");
    Check(api.clEnqueueWriteBuffer(queue, shared_down_buffer, kClTrue, 0,
                                   shared_down.size() * sizeof(float),
                                   shared_down.data(), 0, nullptr, nullptr),
          "clEnqueueWriteBuffer(shared_down)");

    const cl_uint cols_arg = kHiddenSize;
    const cl_uint rows_arg = 1;
    const cl_uint hidden_arg = kHiddenSize;
    Check(api.clSetKernelArg(matvec_kernel, 0, sizeof(weights_buffer), &weights_buffer), "clSetKernelArg(matvec 0)");
    Check(api.clSetKernelArg(matvec_kernel, 1, sizeof(input_buffer), &input_buffer), "clSetKernelArg(matvec 1)");
    Check(api.clSetKernelArg(matvec_kernel, 2, sizeof(cols_arg), &cols_arg), "clSetKernelArg(matvec 2)");
    Check(api.clSetKernelArg(matvec_kernel, 3, sizeof(rows_arg), &rows_arg), "clSetKernelArg(matvec 3)");
    Check(api.clSetKernelArg(matvec_kernel, 4, sizeof(gate_buffer), &gate_buffer), "clSetKernelArg(matvec 4)");
    Check(api.clSetKernelArg(apply_kernel, 0, sizeof(shared_down_buffer), &shared_down_buffer), "clSetKernelArg(apply 0)");
    Check(api.clSetKernelArg(apply_kernel, 1, sizeof(gate_buffer), &gate_buffer), "clSetKernelArg(apply 1)");
    Check(api.clSetKernelArg(apply_kernel, 2, sizeof(hidden_arg), &hidden_arg), "clSetKernelArg(apply 2)");
    Check(api.clSetKernelArg(apply_kernel, 3, sizeof(sigmoid_buffer), &sigmoid_buffer), "clSetKernelArg(apply 3)");
    Check(api.clSetKernelArg(apply_kernel, 4, sizeof(gated_buffer), &gated_buffer), "clSetKernelArg(apply 4)");

    const std::size_t matvec_global = 1;
    const std::size_t apply_global = kHiddenSize;
    std::vector<double> matvec_times;
    std::vector<double> apply_times;
    matvec_times.reserve(static_cast<std::size_t>(repeat));
    apply_times.reserve(static_cast<std::size_t>(repeat));
    for (int i = 0; i < repeat; ++i) {
      cl_event matvec_event = nullptr;
      cl_event apply_event = nullptr;
      Check(api.clEnqueueNDRangeKernel(queue, matvec_kernel, 1, nullptr,
                                       &matvec_global, nullptr, 0, nullptr,
                                       &matvec_event),
            "clEnqueueNDRangeKernel(f32_matvec_row_f32)");
      Check(api.clEnqueueNDRangeKernel(queue, apply_kernel, 1, nullptr,
                                       &apply_global, nullptr, 0, nullptr,
                                       &apply_event),
            "clEnqueueNDRangeKernel(shared_expert_gate_apply_f32)");
      Check(api.clFinish(queue), "clFinish(shared gate)");
      matvec_times.push_back(EventUs(api, matvec_event));
      apply_times.push_back(EventUs(api, apply_event));
      api.clReleaseEvent(matvec_event);
      api.clReleaseEvent(apply_event);
    }

    Check(api.clEnqueueReadBuffer(queue, gate_buffer, kClTrue, 0,
                                  run.shared_gate.size() * sizeof(float),
                                  run.shared_gate.data(), 0, nullptr, nullptr),
          "clEnqueueReadBuffer(gate)");
    Check(api.clEnqueueReadBuffer(queue, sigmoid_buffer, kClTrue, 0,
                                  run.shared_gate_sigmoid.size() * sizeof(float),
                                  run.shared_gate_sigmoid.data(), 0, nullptr, nullptr),
          "clEnqueueReadBuffer(sigmoid)");
    Check(api.clEnqueueReadBuffer(queue, gated_buffer, kClTrue, 0,
                                  run.shared_gated.size() * sizeof(float),
                                  run.shared_gated.data(), 0, nullptr, nullptr),
          "clEnqueueReadBuffer(gated)");

    run.timing.gate_matvec_min_us = *std::min_element(matvec_times.begin(), matvec_times.end());
    run.timing.gate_matvec_mean_us =
        std::accumulate(matvec_times.begin(), matvec_times.end(), 0.0) /
        static_cast<double>(matvec_times.size());
    run.timing.gate_apply_min_us = *std::min_element(apply_times.begin(), apply_times.end());
    run.timing.gate_apply_mean_us =
        std::accumulate(apply_times.begin(), apply_times.end(), 0.0) /
        static_cast<double>(apply_times.size());
    const double matvec_weight_bytes = static_cast<double>(gate_weights.size() * sizeof(float));
    const double apply_io_bytes =
        static_cast<double>((shared_down.size() + run.shared_gated.size() + 2) * sizeof(float));
    run.timing.gate_matvec_effective_weight_gb_s =
        matvec_weight_bytes / (run.timing.gate_matvec_min_us / 1e6) / 1e9;
    run.timing.gate_apply_effective_io_gb_s =
        apply_io_bytes / (run.timing.gate_apply_min_us / 1e6) / 1e9;
    run.timing.gate_matvec_global_work_items = matvec_global;
    run.timing.gate_apply_global_work_items = apply_global;
  } catch (...) {
    ReleaseMem(api, &gated_buffer);
    ReleaseMem(api, &sigmoid_buffer);
    ReleaseMem(api, &gate_buffer);
    ReleaseMem(api, &shared_down_buffer);
    ReleaseMem(api, &input_buffer);
    ReleaseMem(api, &weights_buffer);
    api.clReleaseKernel(apply_kernel);
    api.clReleaseKernel(matvec_kernel);
    api.clReleaseProgram(program);
    api.clReleaseCommandQueue(queue);
    api.clReleaseContext(context);
    throw;
  }
  ReleaseMem(api, &gated_buffer);
  ReleaseMem(api, &sigmoid_buffer);
  ReleaseMem(api, &gate_buffer);
  ReleaseMem(api, &shared_down_buffer);
  ReleaseMem(api, &input_buffer);
  ReleaseMem(api, &weights_buffer);
  api.clReleaseKernel(apply_kernel);
  api.clReleaseKernel(matvec_kernel);
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
    const std::string tensor_name =
        LayerTensorName(args.layer, "ffn_gate_inp_shexp.weight");
    const auto* tensor = iq36::find_tensor(index, tensor_name);
    Require(tensor != nullptr, "shared expert gate tensor missing");
    const bool tensor_shape_ok =
        tensor->type == 0 && tensor->dims == std::vector<std::uint64_t>{kHiddenSize};

    const auto attn_post_norm =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "attn_post_norm.bin"));
    const auto ffn_shexp =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "ffn_shexp.bin"));
    const auto oracle_shared_gate =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "shared_expert_gate.bin"));
    const auto oracle_shared_gate_sigmoid =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "shared_expert_gate_sigmoid.bin"));
    const auto oracle_shexp_gated =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "ffn_shexp_gated.bin"));
    Require(attn_post_norm.size() == kHiddenSize, "attn post norm payload size mismatch");
    Require(ffn_shexp.size() == kHiddenSize, "ffn_shexp payload size mismatch");
    Require(oracle_shared_gate.size() == 1, "shared gate payload size mismatch");
    Require(oracle_shared_gate_sigmoid.size() == 1, "shared gate sigmoid payload size mismatch");
    Require(oracle_shexp_gated.size() == kHiddenSize, "ffn_shexp_gated payload size mismatch");

    std::ifstream model(args.model_path, std::ios::binary);
    Require(static_cast<bool>(model), "model file could not be opened");
    const auto gate_weights =
        ReadF32TensorPayload(model, *tensor, static_cast<std::size_t>(kHiddenSize));

    const auto native_shared_gate =
        iq36::matvec_tensor(args.model_path, index, tensor_name, attn_post_norm);
    Require(native_shared_gate.size() == 1, "native shared gate size mismatch");
    const std::vector<float> native_shared_gate_sigmoid{
        iq36::sigmoid_scalar(native_shared_gate[0])};
    const auto native_shexp_gated =
        iq36::multiply_by_scalar(ffn_shexp, native_shared_gate_sigmoid[0]);
    const auto gpu =
        RunGpuSharedGate(gate_weights, attn_post_norm, ffn_shexp,
                         args.device_substring, args.repeat);

    const auto gate_cpu_vs_oracle =
        iq36::compare_vectors(native_shared_gate, oracle_shared_gate, kMismatchThreshold);
    const auto gate_gpu_vs_cpu =
        iq36::compare_vectors(gpu.shared_gate, native_shared_gate, kMismatchThreshold);
    const auto gate_gpu_vs_oracle =
        iq36::compare_vectors(gpu.shared_gate, oracle_shared_gate, kMismatchThreshold);
    const auto sigmoid_cpu_vs_oracle =
        iq36::compare_vectors(native_shared_gate_sigmoid, oracle_shared_gate_sigmoid, kMismatchThreshold);
    const auto sigmoid_gpu_vs_cpu =
        iq36::compare_vectors(gpu.shared_gate_sigmoid, native_shared_gate_sigmoid, kMismatchThreshold);
    const auto sigmoid_gpu_vs_oracle =
        iq36::compare_vectors(gpu.shared_gate_sigmoid, oracle_shared_gate_sigmoid, kMismatchThreshold);
    const auto gated_cpu_vs_oracle =
        iq36::compare_vectors(native_shexp_gated, oracle_shexp_gated, kMismatchThreshold);
    const auto gated_gpu_vs_cpu =
        iq36::compare_vectors(gpu.shared_gated, native_shexp_gated, kMismatchThreshold);
    const auto gated_gpu_vs_oracle =
        iq36::compare_vectors(gpu.shared_gated, oracle_shexp_gated, kMismatchThreshold);

    const bool load_map_ready = load_map.ready;
    const bool payload_counts_ok =
        attn_post_norm.size() == kHiddenSize &&
        ffn_shexp.size() == kHiddenSize &&
        oracle_shared_gate.size() == 1 &&
        oracle_shared_gate_sigmoid.size() == 1 &&
        oracle_shexp_gated.size() == kHiddenSize;
    const bool gate_matches =
        ComparePassed(gate_cpu_vs_oracle) &&
        ComparePassed(gate_gpu_vs_cpu) &&
        ComparePassed(gate_gpu_vs_oracle) &&
        ComparePassed(sigmoid_cpu_vs_oracle) &&
        ComparePassed(sigmoid_gpu_vs_cpu) &&
        ComparePassed(sigmoid_gpu_vs_oracle) &&
        ComparePassed(gated_cpu_vs_oracle) &&
        ComparePassed(gated_gpu_vs_cpu) &&
        ComparePassed(gated_gpu_vs_oracle);
    const bool timing_positive =
        gpu.timing.gate_matvec_min_us > 0.0 &&
        gpu.timing.gate_apply_min_us > 0.0;
    const bool required_checks_passed =
        load_map_ready &&
        tensor_shape_ok &&
        payload_counts_ok &&
        gpu.device_name.find(args.device_substring) != std::string::npos &&
        gate_matches &&
        timing_positive;

    std::cout << std::setprecision(10);
    std::cout << "{";
    std::cout << "\"schema_version\":\"intel-qwen36-gpu-shared-expert-gate-probe-v0\",";
    std::cout << "\"model_path\":\"" << JsonEscape(args.model_path) << "\",";
    std::cout << "\"layer\":" << args.layer << ",";
    std::cout << "\"source_token_position\":" << kSourceTokenPosition << ",";
    std::cout << "\"hidden_size\":" << kHiddenSize << ",";
    std::cout << "\"tensor_name\":\"" << JsonEscape(tensor_name) << "\",";
    std::cout << "\"tensor_type\":" << tensor->type << ",";
    std::cout << "\"tensor_nbytes\":" << tensor->nbytes << ",";
    std::cout << "\"platform_name\":\"" << JsonEscape(gpu.platform_name) << "\",";
    std::cout << "\"device_name\":\"" << JsonEscape(gpu.device_name) << "\",";
    std::cout << "\"build_log\":\"" << JsonEscape(gpu.build_log) << "\",";
    std::cout << "\"program_build_ms\":" << gpu.program_build_ms << ",";
    std::cout << "\"repeat\":" << args.repeat << ",";
    std::cout << "\"timings\":{";
    std::cout << "\"shared_gate_matvec_gpu_kernel_min_us\":"
              << gpu.timing.gate_matvec_min_us << ",";
    std::cout << "\"shared_gate_matvec_gpu_kernel_mean_us\":"
              << gpu.timing.gate_matvec_mean_us << ",";
    std::cout << "\"shared_gate_matvec_gpu_effective_weight_gb_s\":"
              << gpu.timing.gate_matvec_effective_weight_gb_s << ",";
    std::cout << "\"shared_gate_apply_gpu_kernel_min_us\":"
              << gpu.timing.gate_apply_min_us << ",";
    std::cout << "\"shared_gate_apply_gpu_kernel_mean_us\":"
              << gpu.timing.gate_apply_mean_us << ",";
    std::cout << "\"shared_gate_apply_gpu_effective_io_gb_s\":"
              << gpu.timing.gate_apply_effective_io_gb_s << ",";
    std::cout << "\"shared_gate_matvec_global_work_items\":"
              << gpu.timing.gate_matvec_global_work_items << ",";
    std::cout << "\"shared_gate_apply_global_work_items\":"
              << gpu.timing.gate_apply_global_work_items;
    std::cout << "},";
    std::cout << "\"comparisons\":{";
    std::cout << "\"shared_gate\":";
    WriteCompareGroup(gate_cpu_vs_oracle, gate_gpu_vs_cpu, gate_gpu_vs_oracle);
    std::cout << ",\"shared_gate_sigmoid\":";
    WriteCompareGroup(sigmoid_cpu_vs_oracle, sigmoid_gpu_vs_cpu, sigmoid_gpu_vs_oracle);
    std::cout << ",\"ffn_shexp_gated\":";
    WriteCompareGroup(gated_cpu_vs_oracle, gated_gpu_vs_cpu, gated_gpu_vs_oracle);
    std::cout << "},";
    std::cout << "\"checks\":{";
    std::cout << "\"load_map_ready\":" << (load_map_ready ? "true" : "false") << ",";
    std::cout << "\"tensor_shape_ok\":" << (tensor_shape_ok ? "true" : "false") << ",";
    std::cout << "\"payload_counts_ok\":" << (payload_counts_ok ? "true" : "false") << ",";
    std::cout << "\"arc_device_selected\":"
              << (gpu.device_name.find(args.device_substring) != std::string::npos ? "true" : "false") << ",";
    std::cout << "\"shared_expert_gate_matches_oracle\":"
              << (gate_matches ? "true" : "false") << ",";
    std::cout << "\"gpu_event_timing_positive\":"
              << (timing_positive ? "true" : "false") << ",";
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
      "# GPU Shared Expert Gate Probe",
      "",
      f"- workstream: `{WORKSTREAM}`",
      f"- host: `{payload.get('host')}`",
      f"- model: `{payload.get('model')}`",
      f"- layer: `{payload.get('layer')}`",
      f"- required checks passed: `{str(payload.get('required_checks_passed')).lower()}`",
      "- speedup claims allowed: `false`",
      f"- platform/device: `{probe.get('platform_name')}` / `{probe.get('device_name')}`",
      f"- tensor: `{probe.get('tensor_name')}`",
      f"- hidden size: `{probe.get('hidden_size')}`",
      "",
      "| output | comparison | max abs | RMSE |",
      "|---|---|---:|---:|",
  ]
  for name in ("shared_gate", "shared_gate_sigmoid", "ffn_shexp_gated"):
    group = comparisons.get(name, {}) if isinstance(comparisons, dict) else {}
    if not isinstance(group, dict):
      continue
    for lane in ("cpu_vs_oracle", "gpu_vs_cpu", "gpu_vs_oracle"):
      cmp = group.get(lane, {}) if isinstance(group.get(lane), dict) else {}
      lines.append(f"| {name} | {lane} | {cmp.get('max_abs_diff')} | {cmp.get('rmse')} |")
  lines += [
      "",
      "| kernel | min us | mean us | GB/s |",
      "|---|---:|---:|---:|",
      "| shared_gate_matvec | "
      f"{timings.get('shared_gate_matvec_gpu_kernel_min_us')} | "
      f"{timings.get('shared_gate_matvec_gpu_kernel_mean_us')} | "
      f"{timings.get('shared_gate_matvec_gpu_effective_weight_gb_s')} |",
      "| shared_gate_apply | "
      f"{timings.get('shared_gate_apply_gpu_kernel_min_us')} | "
      f"{timings.get('shared_gate_apply_gpu_kernel_mean_us')} | "
      f"{timings.get('shared_gate_apply_gpu_effective_io_gb_s')} |",
      "",
      "The probe computes the scalar shared expert input gate with an F32 matvec,",
      "then applies sigmoid and multiplies the captured shared expert output on",
      "GPU. This is component evidence only; it does not prove decode or model",
      "throughput.",
      "",
  ]
  path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  args = parse_args()
  stamp = utc_stamp()
  created_at = iso_now()
  out_dir = args.out_dir or ROOT / f"output/gpu-shared-expert-gate-probe-{stamp}"
  out_dir.mkdir(parents=True, exist_ok=False)
  raw_dir = out_dir / "raw"
  raw_dir.mkdir()
  payloads = resolve_payloads(args.layer)
  opencl_source = OPENCL_SOURCE.read_text(encoding="utf-8")
  opencl_source_hash = iq36_local.sha256_file(OPENCL_SOURCE)
  local_cpp = out_dir / "gpu_shared_expert_gate_probe.cpp"
  local_cpp.write_text(
      PROBE_CPP.replace("@@OPENCL_SOURCE_LITERAL@@", cpp_raw_string_literal(opencl_source)),
      encoding="utf-8",
  )

  remote_dir = f"{args.remote_root.rstrip('/')}/gpu-shared-expert-gate-probe-{stamp}"
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
    transfers.append(iq36_local.copy_to(args.host, local_cpp, f"{remote_dir}/tests/gpu_shared_expert_gate_probe.cpp", args.timeout_s))
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
          f"{shlex.quote(remote_dir + '/tests/gpu_shared_expert_gate_probe.cpp')} "
          "-ldl -pthread "
          f"-o {shlex.quote(remote_dir + '/build/iq36-gpu-shared-expert-gate-probe')}"
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
      f"{remote_dir}/build/iq36-gpu-shared-expert-gate-probe",
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
      {"name": "shared_expert_gate_matches_oracle", "pass": bool(probe and nested_bool(probe, "checks", "shared_expert_gate_matches_oracle"))},
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
      "tool": "tools/intel-qwen36-gpu-shared-expert-gate-probe.py",
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
      "gpu_shared_expert_gate_probe",
      [
          ("required_checks_passed", required_checks_passed),
          ("shared_gate_matvec_kernel_min_us", nested_number(timings, "shared_gate_matvec_gpu_kernel_min_us")),
          ("shared_gate_apply_kernel_min_us", nested_number(timings, "shared_gate_apply_gpu_kernel_min_us")),
          ("shared_gate_gpu_vs_oracle_max_abs_diff", nested_number(comparisons, "shared_gate", "gpu_vs_oracle", "max_abs_diff")),
          ("shared_gate_sigmoid_gpu_vs_oracle_max_abs_diff", nested_number(comparisons, "shared_gate_sigmoid", "gpu_vs_oracle", "max_abs_diff")),
          ("ffn_shexp_gated_gpu_vs_oracle_max_abs_diff", nested_number(comparisons, "ffn_shexp_gated", "gpu_vs_oracle", "max_abs_diff")),
      ],
  )
  write_summary(out_dir / "summary.md", payload)
  print(out_dir)
  return 0 if required_checks_passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
