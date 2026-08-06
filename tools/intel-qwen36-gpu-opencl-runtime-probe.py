#!/usr/bin/env python3
"""Run a minimal Arc B390 OpenCL runtime + source-stream probe.

This is a GPU bring-up gate, not a speedup benchmark. It proves the selected
OpenCL device can compile a tiny source kernel, create buffers, stream bytes
from a deterministic slice of the locked GGUF file, collect event timing, and
match a CPU checksum. The next step can replace the byte slice with a repacked
tensor layout without changing the runtime plumbing.
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
SCHEMA_VERSION = "intel-qwen36-gpu-opencl-runtime-source-stream-probe-v0"
DEFAULT_HOST = "local"
DEFAULT_REMOTE_ROOT = "/home/intel/intel-qwen36-gpu"
DEFAULT_MODEL = "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf"
DEFAULT_ENV_SCRIPT = "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh"


PROBE_CPP = r'''
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
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
constexpr cl_bool CL_TRUE_VALUE = 1;
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

struct Args {
  std::string source_file;
  std::uint64_t byte_count = 64ULL * 1024ULL * 1024ULL;
  std::uint64_t offset = 0;
  std::uint64_t work_items = 4096;
  std::uint64_t repeat = 5;
  std::string device_substring = "B390";
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

[[noreturn]] void Die(const std::string& message) {
  throw std::runtime_error(message);
}

void Check(cl_int err, const std::string& where) {
  if (err != CL_SUCCESS_VALUE) {
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

  explicit OpenClApi(const std::string& path) {
    lib = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!lib) {
      Die(std::string("dlopen failed: ") + dlerror());
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

std::string GetInfoString(OpenClApi& api, cl_platform_id platform, cl_platform_info info) {
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
  if (platform_count == 0) {
    Die("no OpenCL platforms");
  }
  std::vector<cl_platform_id> platforms(platform_count);
  Check(api.clGetPlatformIDs(platform_count, platforms.data(), nullptr), "clGetPlatformIDs(list)");
  for (cl_platform_id platform : platforms) {
    cl_uint device_count = 0;
    const cl_int count_err = api.clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU_VALUE, 0, nullptr, &device_count);
    if (count_err != CL_SUCCESS_VALUE || device_count == 0) {
      continue;
    }
    std::vector<cl_device_id> devices(device_count);
    Check(api.clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU_VALUE, device_count, devices.data(), nullptr), "clGetDeviceIDs(list)");
    for (cl_device_id device : devices) {
      const std::string name = GetDeviceString(api, device, CL_DEVICE_NAME_VALUE);
      if (device_substring.empty() || name.find(device_substring) != std::string::npos) {
        return {platform, device, GetInfoString(api, platform, CL_PLATFORM_NAME_VALUE), name};
      }
    }
  }
  Die("no matching OpenCL GPU device for substring: " + device_substring);
}

std::vector<std::uint8_t> ReadBytes(const std::string& path, std::uint64_t offset, std::uint64_t byte_count) {
  std::vector<std::uint8_t> data(static_cast<std::size_t>(byte_count));
  if (path == "__synthetic__") {
    for (std::uint64_t i = 0; i < byte_count; ++i) {
      data[static_cast<std::size_t>(i)] = static_cast<std::uint8_t>((i * 131ULL + 17ULL) & 0xFFU);
    }
    return data;
  }
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    Die("source file could not be opened: " + path);
  }
  input.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
  if (!input) {
    Die("source file seek failed");
  }
  input.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size()));
  if (input.gcount() != static_cast<std::streamsize>(data.size())) {
    Die("source file did not contain requested byte range");
  }
  return data;
}

std::uint64_t SumBytes(const std::vector<std::uint8_t>& data) {
  std::uint64_t sum = 0;
  for (std::uint8_t byte : data) {
    sum += byte;
  }
  return sum;
}

std::uint64_t SumUlongs(const std::vector<cl_ulong>& values) {
  std::uint64_t sum = 0;
  for (cl_ulong value : values) {
    sum += static_cast<std::uint64_t>(value);
  }
  return sum;
}

double EventUs(OpenClApi& api, cl_event event) {
  cl_ulong start = 0;
  cl_ulong end = 0;
  Check(api.clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START_VALUE, sizeof(start), &start, nullptr), "clGetEventProfilingInfo(start)");
  Check(api.clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END_VALUE, sizeof(end), &end, nullptr), "clGetEventProfilingInfo(end)");
  return static_cast<double>(end - start) / 1000.0;
}

Args ParseArgs(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    auto need_value = [&](const char* name) -> std::string {
      if (i + 1 >= argc) Die(std::string("missing value for ") + name);
      return argv[++i];
    };
    if (key == "--source-file") args.source_file = need_value("--source-file");
    else if (key == "--bytes") args.byte_count = std::stoull(need_value("--bytes"));
    else if (key == "--offset") args.offset = std::stoull(need_value("--offset"));
    else if (key == "--work-items") args.work_items = std::stoull(need_value("--work-items"));
    else if (key == "--repeat") args.repeat = std::stoull(need_value("--repeat"));
    else if (key == "--device-substring") args.device_substring = need_value("--device-substring");
    else Die("unknown argument: " + key);
  }
  if (args.source_file.empty()) Die("--source-file is required");
  if (args.byte_count == 0 || args.work_items == 0 || args.repeat == 0) {
    Die("--bytes, --work-items, and --repeat must be positive");
  }
  return args;
}

int main(int argc, char** argv) {
  try {
    const Args args = ParseArgs(argc, argv);
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<std::uint8_t> data = ReadBytes(args.source_file, args.offset, args.byte_count);
    const auto t1 = std::chrono::steady_clock::now();
    const std::uint64_t cpu_sum = SumBytes(data);
    OpenClApi api("libOpenCL.so.1");
    const SelectedDevice selected = SelectDevice(api, args.device_substring);

    cl_int err = CL_SUCCESS_VALUE;
    cl_context context = api.clCreateContext(nullptr, 1, &selected.device, nullptr, nullptr, &err);
    Check(err, "clCreateContext");
    cl_command_queue queue = api.clCreateCommandQueue(context, selected.device, CL_QUEUE_PROFILING_ENABLE_VALUE, &err);
    Check(err, "clCreateCommandQueue");

    const char* kernel_source = R"CLC(
__kernel void stream_checksum(__global const uchar* data, ulong n, __global ulong* partial) {
  const ulong gid = (ulong)get_global_id(0);
  const ulong gsize = (ulong)get_global_size(0);
  ulong sum = 0;
  for (ulong i = gid; i < n; i += gsize) {
    sum += (ulong)data[i];
  }
  partial[gid] = sum;
}
)CLC";
    const std::size_t source_len = std::strlen(kernel_source);
    cl_program program = api.clCreateProgramWithSource(context, 1, &kernel_source, &source_len, &err);
    Check(err, "clCreateProgramWithSource");
    const auto build_start = std::chrono::steady_clock::now();
    err = api.clBuildProgram(program, 1, &selected.device, "", nullptr, nullptr);
    const auto build_end = std::chrono::steady_clock::now();
    std::string build_log;
    {
      std::size_t log_size = 0;
      api.clGetProgramBuildInfo(program, selected.device, CL_PROGRAM_BUILD_LOG_VALUE, 0, nullptr, &log_size);
      if (log_size > 1) {
        build_log.resize(log_size, '\0');
        api.clGetProgramBuildInfo(program, selected.device, CL_PROGRAM_BUILD_LOG_VALUE, log_size, build_log.data(), nullptr);
        if (!build_log.empty() && build_log.back() == '\0') build_log.pop_back();
      }
    }
    Check(err, "clBuildProgram");
    cl_kernel kernel = api.clCreateKernel(program, "stream_checksum", &err);
    Check(err, "clCreateKernel(stream_checksum)");

    cl_mem data_buffer = api.clCreateBuffer(context, CL_MEM_READ_ONLY_VALUE, data.size(), nullptr, &err);
    Check(err, "clCreateBuffer(data)");
    std::vector<cl_ulong> partial(static_cast<std::size_t>(args.work_items), 0);
    cl_mem partial_buffer = api.clCreateBuffer(context, CL_MEM_WRITE_ONLY_VALUE, partial.size() * sizeof(cl_ulong), nullptr, &err);
    Check(err, "clCreateBuffer(partial)");

    cl_event write_event = nullptr;
    Check(api.clEnqueueWriteBuffer(queue, data_buffer, CL_FALSE_VALUE, 0, data.size(), data.data(), 0, nullptr, &write_event), "clEnqueueWriteBuffer(data)");
    Check(api.clFinish(queue), "clFinish(write)");
    const double write_us = EventUs(api, write_event);
    api.clReleaseEvent(write_event);

    const cl_ulong n = static_cast<cl_ulong>(data.size());
    Check(api.clSetKernelArg(kernel, 0, sizeof(data_buffer), &data_buffer), "clSetKernelArg(0)");
    Check(api.clSetKernelArg(kernel, 1, sizeof(n), &n), "clSetKernelArg(1)");
    Check(api.clSetKernelArg(kernel, 2, sizeof(partial_buffer), &partial_buffer), "clSetKernelArg(2)");

    const std::size_t global = static_cast<std::size_t>(args.work_items);
    std::vector<double> kernel_us;
    kernel_us.reserve(static_cast<std::size_t>(args.repeat));
    for (std::uint64_t r = 0; r < args.repeat; ++r) {
      cl_event kernel_event = nullptr;
      Check(api.clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global, nullptr, 0, nullptr, &kernel_event), "clEnqueueNDRangeKernel");
      Check(api.clFinish(queue), "clFinish(kernel)");
      kernel_us.push_back(EventUs(api, kernel_event));
      api.clReleaseEvent(kernel_event);
    }

    cl_event read_event = nullptr;
    Check(api.clEnqueueReadBuffer(queue, partial_buffer, CL_FALSE_VALUE, 0, partial.size() * sizeof(cl_ulong), partial.data(), 0, nullptr, &read_event), "clEnqueueReadBuffer(partial)");
    Check(api.clFinish(queue), "clFinish(read)");
    const double read_us = EventUs(api, read_event);
    api.clReleaseEvent(read_event);

    const std::uint64_t gpu_sum = SumUlongs(partial);
    const auto t2 = std::chrono::steady_clock::now();
    const double read_source_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double build_ms = std::chrono::duration<double, std::milli>(build_end - build_start).count();
    const double process_ms = std::chrono::duration<double, std::milli>(t2 - t0).count();
    const double kernel_min_us = *std::min_element(kernel_us.begin(), kernel_us.end());
    double kernel_sum_us = 0.0;
    for (double value : kernel_us) kernel_sum_us += value;
    const double kernel_mean_us = kernel_sum_us / static_cast<double>(kernel_us.size());
    const double bytes = static_cast<double>(data.size());
    const double kernel_min_gb_s = bytes / (kernel_min_us / 1e6) / 1e9;
    const double kernel_mean_gb_s = bytes / (kernel_mean_us / 1e6) / 1e9;
    const double h2d_gb_s = bytes / (write_us / 1e6) / 1e9;

    std::cout
      << "{"
      << "\"ok\":true,"
      << "\"source_file\":\"" << JsonEscape(args.source_file) << "\","
      << "\"byte_count\":" << data.size() << ","
      << "\"offset\":" << args.offset << ","
      << "\"work_items\":" << args.work_items << ","
      << "\"repeat\":" << args.repeat << ","
      << "\"platform_name\":\"" << JsonEscape(selected.platform_name) << "\","
      << "\"device_name\":\"" << JsonEscape(selected.device_name) << "\","
      << "\"kernel_name\":\"stream_checksum\","
      << "\"cpu_checksum_sum\":" << cpu_sum << ","
      << "\"gpu_checksum_sum\":" << gpu_sum << ","
      << "\"checksum_match\":" << (cpu_sum == gpu_sum ? "true" : "false") << ","
      << "\"read_source_ms\":" << read_source_ms << ","
      << "\"program_build_ms\":" << build_ms << ","
      << "\"write_buffer_us\":" << write_us << ","
      << "\"read_buffer_us\":" << read_us << ","
      << "\"kernel_min_us\":" << kernel_min_us << ","
      << "\"kernel_mean_us\":" << kernel_mean_us << ","
      << "\"kernel_min_gb_s\":" << kernel_min_gb_s << ","
      << "\"kernel_mean_gb_s\":" << kernel_mean_gb_s << ","
      << "\"host_to_device_gb_s\":" << h2d_gb_s << ","
      << "\"process_total_ms\":" << process_ms << ","
      << "\"build_log\":\"" << JsonEscape(build_log) << "\""
      << "}\n";

    api.clReleaseMemObject(partial_buffer);
    api.clReleaseMemObject(data_buffer);
    api.clReleaseKernel(kernel);
    api.clReleaseProgram(program);
    api.clReleaseCommandQueue(queue);
    api.clReleaseContext(context);
    return cpu_sum == gpu_sum ? 0 : 3;
  } catch (const std::exception& ex) {
    std::cout << "{\"ok\":false,\"error\":\"" << JsonEscape(ex.what()) << "\"}\n";
    return 2;
  }
}
'''


def utc_stamp() -> str:
  return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def iso_now() -> str:
  return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--target", "--host", dest="host", metavar="TARGET", default=DEFAULT_HOST)
  parser.add_argument("--staging-root", "--remote-root", dest="remote_root", metavar="PATH", default=DEFAULT_REMOTE_ROOT)
  parser.add_argument("--env-script", default=DEFAULT_ENV_SCRIPT)
  parser.add_argument("--source-file", default=DEFAULT_MODEL)
  parser.add_argument("--bytes", type=int, default=64 * 1024 * 1024)
  parser.add_argument("--offset", type=int, default=0)
  parser.add_argument("--work-items", type=int, default=4096)
  parser.add_argument("--repeat", type=int, default=5)
  parser.add_argument("--device-substring", default="B390")
  parser.add_argument("--timeout-s", type=int, default=300)
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


def write_summary(path: Path, payload: dict[str, Any]) -> None:
  probe = payload.get("probe_result") or {}
  checks = payload.get("checks", [])
  lines = [
      "# GPU OpenCL Runtime Source-Stream Probe",
      "",
      f"- workstream: `{WORKSTREAM}`",
      f"- required checks passed: `{str(payload.get('required_checks_passed')).lower()}`",
      "- speedup claims allowed: `false`",
      f"- host: `{payload.get('host')}`",
      f"- remote dir: `{payload.get('remote_dir')}`",
      f"- source file: `{probe.get('source_file')}`",
      f"- byte count: `{probe.get('byte_count')}`",
      f"- platform/device: `{probe.get('platform_name')}` / `{probe.get('device_name')}`",
      f"- checksum match: `{str(probe.get('checksum_match')).lower()}`",
      f"- kernel min/mean GB/s: `{probe.get('kernel_min_gb_s')}` / `{probe.get('kernel_mean_gb_s')}`",
      f"- host-to-device GB/s: `{probe.get('host_to_device_gb_s')}`",
      "",
      "| check | pass |",
      "|---|---:|",
  ]
  for check in checks:
    lines.append(f"| `{check['name']}` | `{str(check['pass']).lower()}` |")
  lines.append("")
  if payload.get("required_checks_passed") is True:
    lines += [
        "Decision: OpenCL runtime, kernel build, device buffer stream, event timing,",
        "and checksum plumbing are now proven on Arc B390. This is not a throughput",
        "or model speedup claim; it is the next gate before a repacked tensor stream",
        "probe.",
    ]
  else:
    lines += [
        "Decision: this probe did not pass. Inspect `raw/compile.json` and",
        "`raw/run.json`; do not use this artifact as GPU bring-up evidence.",
    ]
  lines.append("")
  path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  args = parse_args()
  stamp = utc_stamp()
  out_dir = args.out_dir or ROOT / f"output/gpu-opencl-runtime-source-stream-probe-{stamp}"
  out_dir.mkdir(parents=True, exist_ok=False)
  raw_dir = out_dir / "raw"
  raw_dir.mkdir()
  cpp_path = out_dir / "opencl_runtime_source_stream_probe.cpp"
  cpp_path.write_text(PROBE_CPP, encoding="utf-8")

  created_at = iso_now()
  remote_dir = f"{args.remote_root.rstrip('/')}/opencl-runtime-probe-{stamp}"
  remote_cpp = f"{remote_dir}/probe.cpp"
  remote_exe = f"{remote_dir}/probe"

  setup = iq36_local.run_target(
      args.host,
      f"rm -rf {shlex.quote(remote_dir)} && mkdir -p {shlex.quote(remote_dir)}",
      args.timeout_s,
  )
  source_copy = iq36_local.copy_to(args.host, cpp_path, remote_cpp, args.timeout_s)
  compile_cmd = " && ".join([
      f"source {shlex.quote(args.env_script)} >/dev/null 2>&1",
      (
          f"g++ -std=c++17 -O3 -Wall -Wextra -Wpedantic "
          f"{shlex.quote(remote_cpp)} -ldl -o {shlex.quote(remote_exe)}"
      ),
  ])
  compile_result = iq36_local.run_target(args.host, compile_cmd, args.timeout_s)
  run_argv = [
      remote_exe,
      "--source-file",
      args.source_file,
      "--bytes",
      str(args.bytes),
      "--offset",
      str(args.offset),
      "--work-items",
      str(args.work_items),
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
      if compile_result["returncode"] == 0
      else {"cmd": run_argv, "returncode": None, "stdout": "", "stderr": "compile skipped run"}
  )
  probe_result = parse_probe_stdout(run_result.get("stdout", ""))

  iq36_local.write_json(raw_dir / "setup.json", setup)
  iq36_local.write_json(raw_dir / "copy.json", source_copy)
  iq36_local.write_json(raw_dir / "compile.json", compile_result)
  iq36_local.write_json(raw_dir / "run.json", run_result)
  if probe_result is not None:
    iq36_local.write_json(out_dir / "probe-result.json", probe_result)

  checks = [
      {"name": "remote_dir_created", "pass": setup["returncode"] == 0},
      {"name": "probe_source_transferred", "pass": source_copy["returncode"] == 0},
      {"name": "probe_compiled", "pass": compile_result["returncode"] == 0},
      {"name": "probe_stdout_json_parsed", "pass": isinstance(probe_result, dict)},
      {"name": "probe_process_succeeded", "pass": run_result.get("returncode") == 0},
      {"name": "opencl_runtime_ok", "pass": bool(probe_result and probe_result.get("ok") is True)},
      {"name": "arc_b390_selected", "pass": bool(probe_result and "B390" in str(probe_result.get("device_name", "")))},
      {"name": "checksum_match", "pass": bool(probe_result and probe_result.get("checksum_match") is True)},
      {"name": "kernel_timing_positive", "pass": bool(probe_result and probe_result.get("kernel_min_us", 0) > 0)},
      {"name": "speedup_claims_forbidden", "pass": True},
  ]
  required_checks_passed = all(item["pass"] for item in checks)
  payload = {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "created_at": created_at,
      "host": args.host,
      "remote_dir": remote_dir,
      "source_file": args.source_file,
      "env_script": args.env_script,
      "byte_count_requested": args.bytes,
      "offset": args.offset,
      "work_items": args.work_items,
      "repeat": args.repeat,
      "probe_result": probe_result,
      "checks": checks,
      "required_checks_passed": required_checks_passed,
      "speedup_claims_allowed": False,
      "recommendation": (
          "use this runtime gate for the next repacked tensor source-stream "
          "probe; do not port the full OpenCL microbench monolith first"
      ),
  }
  manifest = {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "created_at": created_at,
      "tool": "tools/intel-qwen36-gpu-opencl-runtime-probe.py",
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
      "gpu_opencl_runtime_source_stream_probe",
      [
          ("byte_count", None if probe_result is None else probe_result.get("byte_count")),
          ("kernel_min_gb_s", None if probe_result is None else probe_result.get("kernel_min_gb_s")),
          ("kernel_mean_gb_s", None if probe_result is None else probe_result.get("kernel_mean_gb_s")),
          ("host_to_device_gb_s", None if probe_result is None else probe_result.get("host_to_device_gb_s")),
          ("checksum_match", None if probe_result is None else probe_result.get("checksum_match")),
      ],
  )
  write_summary(out_dir / "summary.md", payload)
  print(out_dir)
  return 0 if required_checks_passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
