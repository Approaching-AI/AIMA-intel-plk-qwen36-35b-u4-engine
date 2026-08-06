#!/usr/bin/env python3
"""Check whether the target OpenCL runtime supports CAS-based float accumulation.

This is a route gate for the selected/shared Q6 down-to-tail design. A viable
parallel down-to-tail kernel needs the existing rows_per_expert*9 Q6 work-items
to contribute into hidden-sized output without serializing the eight selected
experts plus shared row. This gate proves the small primitive only; it is not
decode speed evidence.
"""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Any

import iq36_local


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
DEFAULT_HOST = "local"
DEFAULT_ENV_SCRIPT = "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh"
DEFAULT_REMOTE_ROOT = "/home/intel/intel-qwen36-gpu"
DEFAULT_OUT_DIR = ROOT / "output/opencl-atomic-f32-gate-20260706Tseq67Z"
SCHEMA_VERSION = "intel-qwen36-opencl-atomic-f32-gate-v0"


CPP_SOURCE = r'''
#include <cmath>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using cl_bool = std::uint32_t;
using cl_bitfield = std::uint64_t;
using cl_command_queue_properties = cl_bitfield;
using cl_device_type = cl_bitfield;
using cl_mem_flags = cl_bitfield;
using cl_int = std::int32_t;
using cl_uint = std::uint32_t;
using cl_platform_id = struct _cl_platform_id*;
using cl_device_id = struct _cl_device_id*;
using cl_context = struct _cl_context*;
using cl_command_queue = struct _cl_command_queue*;
using cl_program = struct _cl_program*;
using cl_kernel = struct _cl_kernel*;
using cl_mem = struct _cl_mem*;

constexpr cl_int CL_SUCCESS_VALUE = 0;
constexpr cl_bool CL_TRUE_VALUE = 1;
constexpr cl_device_type CL_DEVICE_TYPE_GPU_VALUE = 1ULL << 2;
constexpr cl_mem_flags CL_MEM_READ_ONLY_VALUE = 1ULL << 2;
constexpr cl_mem_flags CL_MEM_READ_WRITE_VALUE = 1ULL << 0;
constexpr cl_uint CL_PLATFORM_NAME_VALUE = 0x0902;
constexpr cl_uint CL_DEVICE_NAME_VALUE = 0x102B;
constexpr cl_uint CL_PROGRAM_BUILD_LOG_VALUE = 0x1183;

struct Api {
  void* lib = nullptr;
  cl_int (*clGetPlatformIDs)(cl_uint, cl_platform_id*, cl_uint*) = nullptr;
  cl_int (*clGetPlatformInfo)(cl_platform_id, cl_uint, std::size_t, void*, std::size_t*) = nullptr;
  cl_int (*clGetDeviceIDs)(cl_platform_id, cl_device_type, cl_uint, cl_device_id*, cl_uint*) = nullptr;
  cl_int (*clGetDeviceInfo)(cl_device_id, cl_uint, std::size_t, void*, std::size_t*) = nullptr;
  cl_context (*clCreateContext)(const void*, cl_uint, const cl_device_id*, void*, void*, cl_int*) = nullptr;
  cl_int (*clReleaseContext)(cl_context) = nullptr;
  cl_command_queue (*clCreateCommandQueue)(cl_context, cl_device_id, cl_command_queue_properties, cl_int*) = nullptr;
  cl_int (*clReleaseCommandQueue)(cl_command_queue) = nullptr;
  cl_program (*clCreateProgramWithSource)(cl_context, cl_uint, const char**, const std::size_t*, cl_int*) = nullptr;
  cl_int (*clBuildProgram)(cl_program, cl_uint, const cl_device_id*, const char*, void*, void*) = nullptr;
  cl_int (*clGetProgramBuildInfo)(cl_program, cl_device_id, cl_uint, std::size_t, void*, std::size_t*) = nullptr;
  cl_int (*clReleaseProgram)(cl_program) = nullptr;
  cl_kernel (*clCreateKernel)(cl_program, const char*, cl_int*) = nullptr;
  cl_int (*clSetKernelArg)(cl_kernel, cl_uint, std::size_t, const void*) = nullptr;
  cl_int (*clReleaseKernel)(cl_kernel) = nullptr;
  cl_mem (*clCreateBuffer)(cl_context, cl_mem_flags, std::size_t, void*, cl_int*) = nullptr;
  cl_int (*clReleaseMemObject)(cl_mem) = nullptr;
  cl_int (*clEnqueueWriteBuffer)(cl_command_queue, cl_mem, cl_bool, std::size_t, std::size_t, const void*, cl_uint, const void*, void*) = nullptr;
  cl_int (*clEnqueueReadBuffer)(cl_command_queue, cl_mem, cl_bool, std::size_t, std::size_t, void*, cl_uint, const void*, void*) = nullptr;
  cl_int (*clEnqueueNDRangeKernel)(cl_command_queue, cl_kernel, cl_uint, const std::size_t*, const std::size_t*, const std::size_t*, cl_uint, const void*, void*) = nullptr;
  cl_int (*clFinish)(cl_command_queue) = nullptr;

  template <typename T>
  T load(const char* name) {
    void* sym = dlsym(lib, name);
    if (sym == nullptr) throw std::runtime_error(std::string("missing OpenCL symbol ") + name);
    return reinterpret_cast<T>(sym);
  }

  Api() {
    lib = dlopen("libOpenCL.so.1", RTLD_NOW | RTLD_LOCAL);
    if (lib == nullptr) lib = dlopen("libOpenCL.so", RTLD_NOW | RTLD_LOCAL);
    if (lib == nullptr) throw std::runtime_error("failed to load libOpenCL");
    clGetPlatformIDs = load<decltype(clGetPlatformIDs)>("clGetPlatformIDs");
    clGetPlatformInfo = load<decltype(clGetPlatformInfo)>("clGetPlatformInfo");
    clGetDeviceIDs = load<decltype(clGetDeviceIDs)>("clGetDeviceIDs");
    clGetDeviceInfo = load<decltype(clGetDeviceInfo)>("clGetDeviceInfo");
    clCreateContext = load<decltype(clCreateContext)>("clCreateContext");
    clReleaseContext = load<decltype(clReleaseContext)>("clReleaseContext");
    clCreateCommandQueue = load<decltype(clCreateCommandQueue)>("clCreateCommandQueue");
    clReleaseCommandQueue = load<decltype(clReleaseCommandQueue)>("clReleaseCommandQueue");
    clCreateProgramWithSource = load<decltype(clCreateProgramWithSource)>("clCreateProgramWithSource");
    clBuildProgram = load<decltype(clBuildProgram)>("clBuildProgram");
    clGetProgramBuildInfo = load<decltype(clGetProgramBuildInfo)>("clGetProgramBuildInfo");
    clReleaseProgram = load<decltype(clReleaseProgram)>("clReleaseProgram");
    clCreateKernel = load<decltype(clCreateKernel)>("clCreateKernel");
    clSetKernelArg = load<decltype(clSetKernelArg)>("clSetKernelArg");
    clReleaseKernel = load<decltype(clReleaseKernel)>("clReleaseKernel");
    clCreateBuffer = load<decltype(clCreateBuffer)>("clCreateBuffer");
    clReleaseMemObject = load<decltype(clReleaseMemObject)>("clReleaseMemObject");
    clEnqueueWriteBuffer = load<decltype(clEnqueueWriteBuffer)>("clEnqueueWriteBuffer");
    clEnqueueReadBuffer = load<decltype(clEnqueueReadBuffer)>("clEnqueueReadBuffer");
    clEnqueueNDRangeKernel = load<decltype(clEnqueueNDRangeKernel)>("clEnqueueNDRangeKernel");
    clFinish = load<decltype(clFinish)>("clFinish");
  }
  ~Api() { if (lib != nullptr) dlclose(lib); }
};

void check(cl_int code, const char* what) {
  if (code != CL_SUCCESS_VALUE) {
    throw std::runtime_error(std::string(what) + " failed: " + std::to_string(code));
  }
}

std::string info_string(Api& api, cl_platform_id platform, cl_device_id device, cl_uint kind) {
  std::size_t size = 0;
  if (device == nullptr) check(api.clGetPlatformInfo(platform, kind, 0, nullptr, &size), "clGetPlatformInfo(size)");
  else check(api.clGetDeviceInfo(device, kind, 0, nullptr, &size), "clGetDeviceInfo(size)");
  std::string out(size, '\0');
  if (device == nullptr) check(api.clGetPlatformInfo(platform, kind, out.size(), out.data(), nullptr), "clGetPlatformInfo(value)");
  else check(api.clGetDeviceInfo(device, kind, out.size(), out.data(), nullptr), "clGetDeviceInfo(value)");
  while (!out.empty() && out.back() == '\0') out.pop_back();
  return out;
}

struct Selection {
  cl_platform_id platform = nullptr;
  cl_device_id device = nullptr;
  std::string platform_name;
  std::string device_name;
};

Selection select_device(Api& api, const std::string& needle) {
  cl_uint platform_count = 0;
  check(api.clGetPlatformIDs(0, nullptr, &platform_count), "clGetPlatformIDs(count)");
  std::vector<cl_platform_id> platforms(platform_count);
  check(api.clGetPlatformIDs(platform_count, platforms.data(), nullptr), "clGetPlatformIDs(list)");
  for (cl_platform_id platform : platforms) {
    cl_uint device_count = 0;
    if (api.clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU_VALUE, 0, nullptr, &device_count) != CL_SUCCESS_VALUE) continue;
    std::vector<cl_device_id> devices(device_count);
    check(api.clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU_VALUE, device_count, devices.data(), nullptr), "clGetDeviceIDs(list)");
    for (cl_device_id device : devices) {
      const auto name = info_string(api, platform, device, CL_DEVICE_NAME_VALUE);
      if (needle.empty() || name.find(needle) != std::string::npos) {
        return {platform, device, info_string(api, platform, nullptr, CL_PLATFORM_NAME_VALUE), name};
      }
    }
  }
  throw std::runtime_error("no matching GPU device");
}

std::string build_log(Api& api, cl_program program, cl_device_id device) {
  std::size_t size = 0;
  api.clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG_VALUE, 0, nullptr, &size);
  std::string out(size, '\0');
  if (size > 0) api.clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG_VALUE, out.size(), out.data(), nullptr);
  while (!out.empty() && out.back() == '\0') out.pop_back();
  return out;
}

const char* kSource = R"CLC(
inline void atomic_add_f32(volatile __global uint* ptr, float value) {
  uint old_bits;
  uint new_bits;
  do {
    old_bits = *ptr;
    new_bits = as_uint(as_float(old_bits) + value);
  } while (atomic_cmpxchg(ptr, old_bits, new_bits) != old_bits);
}

__kernel void atomic_accum9(__global uint* output_bits,
                            __global const float* values,
                            uint hidden_size) {
  const uint gid = (uint)get_global_id(0);
  const uint row = gid % hidden_size;
  atomic_add_f32((volatile __global uint*)(output_bits + row), values[gid]);
}
)CLC";

int main(int argc, char** argv) {
  std::string device_filter = "B390";
  if (argc > 2 && std::string(argv[1]) == "--device") device_filter = argv[2];
  try {
    Api api;
    const auto selected = select_device(api, device_filter);
    cl_int err = 0;
    cl_context context = api.clCreateContext(nullptr, 1, &selected.device, nullptr, nullptr, &err);
    check(err, "clCreateContext");
    cl_command_queue queue = api.clCreateCommandQueue(context, selected.device, 0, &err);
    check(err, "clCreateCommandQueue");
    const char* source = kSource;
    const std::size_t source_len = std::strlen(kSource);
    cl_program program = api.clCreateProgramWithSource(context, 1, &source, &source_len, &err);
    check(err, "clCreateProgramWithSource");
    err = api.clBuildProgram(program, 1, &selected.device, "", nullptr, nullptr);
    const auto log = build_log(api, program, selected.device);
    check(err, "clBuildProgram");
    cl_kernel kernel = api.clCreateKernel(program, "atomic_accum9", &err);
    check(err, "clCreateKernel");

    constexpr std::size_t hidden = 2048;
    constexpr std::size_t contributors = 9;
    std::vector<float> values(hidden * contributors, 0.0f);
    std::vector<float> expected(hidden, 0.0f);
    for (std::size_t c = 0; c < contributors; ++c) {
      for (std::size_t row = 0; row < hidden; ++row) {
        const float value = 0.0001f * static_cast<float>((row % 17) + 1) +
                            0.125f * static_cast<float>(c + 1);
        values[c * hidden + row] = value;
        expected[row] += value;
      }
    }
    std::vector<std::uint32_t> zero_bits(hidden, 0);
    std::vector<std::uint32_t> out_bits(hidden, 0);
    cl_mem out = api.clCreateBuffer(context, CL_MEM_READ_WRITE_VALUE, hidden * sizeof(std::uint32_t), nullptr, &err);
    check(err, "clCreateBuffer(out)");
    cl_mem add = api.clCreateBuffer(context, CL_MEM_READ_ONLY_VALUE, values.size() * sizeof(float), nullptr, &err);
    check(err, "clCreateBuffer(add)");
    check(api.clEnqueueWriteBuffer(queue, out, CL_TRUE_VALUE, 0, hidden * sizeof(std::uint32_t), zero_bits.data(), 0, nullptr, nullptr), "write out");
    check(api.clEnqueueWriteBuffer(queue, add, CL_TRUE_VALUE, 0, values.size() * sizeof(float), values.data(), 0, nullptr, nullptr), "write values");
    const cl_uint hidden_arg = static_cast<cl_uint>(hidden);
    check(api.clSetKernelArg(kernel, 0, sizeof(out), &out), "arg0");
    check(api.clSetKernelArg(kernel, 1, sizeof(add), &add), "arg1");
    check(api.clSetKernelArg(kernel, 2, sizeof(hidden_arg), &hidden_arg), "arg2");
    const std::size_t global = values.size();
    check(api.clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr), "enqueue");
    check(api.clFinish(queue), "finish");
    check(api.clEnqueueReadBuffer(queue, out, CL_TRUE_VALUE, 0, hidden * sizeof(std::uint32_t), out_bits.data(), 0, nullptr, nullptr), "read");
    double max_abs = 0.0;
    for (std::size_t i = 0; i < hidden; ++i) {
      float got = 0.0f;
      std::memcpy(&got, &out_bits[i], sizeof(float));
      max_abs = std::max(max_abs, std::abs(static_cast<double>(got - expected[i])));
    }
    const bool passed = max_abs < 1e-5;
    std::cout << "{"
              << "\"platform\":\"" << selected.platform_name << "\","
              << "\"device\":\"" << selected.device_name << "\","
              << "\"hidden\":" << hidden << ","
              << "\"contributors\":" << contributors << ","
              << "\"max_abs\":" << max_abs << ","
              << "\"atomic_f32_required_checks_passed\":" << (passed ? "true" : "false") << ","
              << "\"build_log_size\":" << log.size()
              << "}" << std::endl;
    api.clReleaseMemObject(add);
    api.clReleaseMemObject(out);
    api.clReleaseKernel(kernel);
    api.clReleaseProgram(program);
    api.clReleaseCommandQueue(queue);
    api.clReleaseContext(context);
    return passed ? 0 : 2;
  } catch (const std::exception& exc) {
    std::cout << "{\"atomic_f32_required_checks_passed\":false,\"error\":\""
              << exc.what() << "\"}" << std::endl;
    return 1;
  }
}
'''


def _parse_last_json(stdout: str) -> dict[str, Any]:
  for line in reversed(stdout.splitlines()):
    line = line.strip()
    if line.startswith("{") and line.endswith("}"):
      value = json.loads(line)
      if isinstance(value, dict):
        return value
  return {"atomic_f32_required_checks_passed": False, "parse_error": "no JSON line"}


def run_gate(args: argparse.Namespace) -> dict[str, Any]:
  out_dir = args.out_dir
  raw_dir = out_dir / "raw"
  raw_dir.mkdir(parents=True, exist_ok=True)
  local_cpp = raw_dir / "opencl_atomic_f32_gate.cpp"
  local_cpp.write_text(CPP_SOURCE, encoding="utf-8")
  remote_dir = f"{args.remote_root}/opencl-atomic-f32-gate-seq67"
  setup = iq36_local.run_target(
      args.host,
      f"rm -rf {shlex.quote(remote_dir)} && mkdir -p {shlex.quote(remote_dir + '/build')} {shlex.quote(remote_dir + '/tests')}",
      args.timeout_s,
  )
  transfer = iq36_local.copy_to(
      args.host, local_cpp, f"{remote_dir}/tests/opencl_atomic_f32_gate.cpp", args.timeout_s
  )
  compile_cmd = " && ".join([
      f"source {shlex.quote(args.env_script)} >/dev/null 2>&1",
      "g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic "
      f"{shlex.quote(remote_dir + '/tests/opencl_atomic_f32_gate.cpp')} "
      "-ldl -o "
      f"{shlex.quote(remote_dir + '/build/opencl-atomic-f32-gate')}",
  ])
  compile_result = iq36_local.run_target(args.host, compile_cmd, args.timeout_s)
  run_cmd = " && ".join([
      f"source {shlex.quote(args.env_script)} >/dev/null 2>&1",
      f"{shlex.quote(remote_dir + '/build/opencl-atomic-f32-gate')} --device B390",
  ])
  run_result = (
      iq36_local.run_target(args.host, run_cmd, args.timeout_s)
      if compile_result["returncode"] == 0
      else {"cmd": ["skipped"], "returncode": 127, "stdout": "", "stderr": "compile failed"}
  )
  parsed = _parse_last_json(str(run_result.get("stdout", "")))
  result = {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "host": args.host,
      "remote_dir": remote_dir,
      "compile_returncode": compile_result["returncode"],
      "run_returncode": run_result["returncode"],
      "atomic_f32_required_checks_passed": (
          compile_result["returncode"] == 0
          and run_result["returncode"] == 0
          and bool(parsed.get("atomic_f32_required_checks_passed"))
      ),
      "target_result": parsed,
      "route_interpretation": (
          "CAS-based float accumulation is available for a parallel "
          "selected/shared Q6 down-to-tail reduction prototype."
      ),
      "speedup_claims_allowed": False,
  }
  iq36_local.write_json(raw_dir / "setup.json", setup)
  iq36_local.write_json(raw_dir / "transfer.json", transfer)
  iq36_local.write_json(raw_dir / "compile.json", compile_result)
  iq36_local.write_json(raw_dir / "run.json", run_result)
  iq36_local.write_json(out_dir / "metrics.json", result)
  (out_dir / "SUMMARY.md").write_text(
      "\n".join([
          "# OpenCL Atomic F32 Gate",
          "",
          f"- workstream: `{WORKSTREAM}`",
          f"- host: `{args.host}`",
          f"- compile returncode: `{compile_result['returncode']}`",
          f"- run returncode: `{run_result['returncode']}`",
          f"- required checks passed: `{str(result['atomic_f32_required_checks_passed']).lower()}`",
          f"- max abs: `{parsed.get('max_abs')}`",
          "",
          "This is a route primitive check only, not decode speed evidence.",
          "",
      ]),
      encoding="utf-8",
  )
  return result


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--target", "--host", dest="host", metavar="TARGET", default=DEFAULT_HOST)
  parser.add_argument("--env-script", default=DEFAULT_ENV_SCRIPT)
  parser.add_argument("--staging-root", "--remote-root", dest="remote_root", metavar="PATH", default=DEFAULT_REMOTE_ROOT)
  parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
  parser.add_argument("--timeout-s", type=int, default=300)
  return parser.parse_args()


def main() -> int:
  args = parse_args()
  result = run_gate(args)
  print(json.dumps(result, indent=2, sort_keys=True))
  return 0 if result["atomic_f32_required_checks_passed"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
