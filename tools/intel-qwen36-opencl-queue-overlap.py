#!/usr/bin/env python3
"""Measure whether the target OpenCL runtime overlaps independent queues.

The selected/shared FFN branch-overlap route needs two independent command
queues in one context before it is worth adding a same-context resident branch
API. This tool runs a synthetic same-kernel A+B workload serially and then on
two command queues, writing a small evidence bundle under `output/`.
"""

from __future__ import annotations

import argparse
import json
import shlex
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import iq36_local


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HOST = "local"
DEFAULT_ENV_SCRIPT = "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh"
DEFAULT_REMOTE_ROOT = "/home/intel/intel-qwen36-gpu"


CPP_SOURCE = r'''
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <iostream>
#include <numeric>
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
using cl_ulong = std::uint64_t;
using cl_platform_id = struct _cl_platform_id*;
using cl_device_id = struct _cl_device_id*;
using cl_context = struct _cl_context*;
using cl_command_queue = struct _cl_command_queue*;
using cl_program = struct _cl_program*;
using cl_kernel = struct _cl_kernel*;
using cl_mem = struct _cl_mem*;
using cl_event = struct _cl_event*;

constexpr cl_int CL_SUCCESS_VALUE = 0;
constexpr cl_bool CL_TRUE_VALUE = 1;
constexpr cl_device_type CL_DEVICE_TYPE_GPU_VALUE = 1ULL << 2;
constexpr cl_mem_flags CL_MEM_WRITE_ONLY_VALUE = 1ULL << 1;
constexpr cl_command_queue_properties CL_QUEUE_PROFILING_ENABLE_VALUE = 1ULL << 1;
constexpr cl_uint CL_PLATFORM_NAME_VALUE = 0x0902;
constexpr cl_uint CL_DEVICE_NAME_VALUE = 0x102B;
constexpr cl_uint CL_PROGRAM_BUILD_LOG_VALUE = 0x1183;
constexpr cl_uint CL_PROFILING_COMMAND_START_VALUE = 0x1282;
constexpr cl_uint CL_PROFILING_COMMAND_END_VALUE = 0x1284;

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
  cl_int (*clEnqueueNDRangeKernel)(cl_command_queue, cl_kernel, cl_uint, const std::size_t*, const std::size_t*, const std::size_t*, cl_uint, const cl_event*, cl_event*) = nullptr;
  cl_int (*clFinish)(cl_command_queue) = nullptr;
  cl_int (*clGetEventProfilingInfo)(cl_event, cl_uint, std::size_t, void*, std::size_t*) = nullptr;
  cl_int (*clReleaseEvent)(cl_event) = nullptr;

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
    clEnqueueNDRangeKernel = load<decltype(clEnqueueNDRangeKernel)>("clEnqueueNDRangeKernel");
    clFinish = load<decltype(clFinish)>("clFinish");
    clGetEventProfilingInfo = load<decltype(clGetEventProfilingInfo)>("clGetEventProfilingInfo");
    clReleaseEvent = load<decltype(clReleaseEvent)>("clReleaseEvent");
  }

  ~Api() {
    if (lib != nullptr) dlclose(lib);
  }
};

void check(cl_int code, const char* what) {
  if (code != CL_SUCCESS_VALUE) {
    throw std::runtime_error(std::string(what) + " failed: " + std::to_string(code));
  }
}

std::string platform_name(Api& api, cl_platform_id platform) {
  std::size_t size = 0;
  check(api.clGetPlatformInfo(platform, CL_PLATFORM_NAME_VALUE, 0, nullptr, &size),
        "clGetPlatformInfo(size)");
  std::string out(size, '\0');
  check(api.clGetPlatformInfo(platform, CL_PLATFORM_NAME_VALUE, out.size(), out.data(), nullptr),
        "clGetPlatformInfo(name)");
  while (!out.empty() && out.back() == '\0') out.pop_back();
  return out;
}

std::string device_name(Api& api, cl_device_id device) {
  std::size_t size = 0;
  check(api.clGetDeviceInfo(device, CL_DEVICE_NAME_VALUE, 0, nullptr, &size),
        "clGetDeviceInfo(size)");
  std::string out(size, '\0');
  check(api.clGetDeviceInfo(device, CL_DEVICE_NAME_VALUE, out.size(), out.data(), nullptr),
        "clGetDeviceInfo(name)");
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
    cl_int rc = api.clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU_VALUE, 0, nullptr, &device_count);
    if (rc != CL_SUCCESS_VALUE || device_count == 0) continue;
    std::vector<cl_device_id> devices(device_count);
    check(api.clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU_VALUE, device_count, devices.data(), nullptr),
          "clGetDeviceIDs(list)");
    for (cl_device_id device : devices) {
      const std::string dn = device_name(api, device);
      if (needle.empty() || dn.find(needle) != std::string::npos) {
        return {platform, device, platform_name(api, platform), dn};
      }
    }
  }
  throw std::runtime_error("no matching GPU device for substring: " + needle);
}

double event_us(Api& api, cl_event event) {
  if (event == nullptr) return 0.0;
  cl_ulong start = 0;
  cl_ulong end = 0;
  check(api.clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START_VALUE,
                                    sizeof(start), &start, nullptr),
        "clGetEventProfilingInfo(start)");
  check(api.clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END_VALUE,
                                    sizeof(end), &end, nullptr),
        "clGetEventProfilingInfo(end)");
  return static_cast<double>(end - start) / 1000.0;
}

struct Args {
  std::string device = "B390";
  std::size_t global = 262144;
  std::size_t local = 256;
  cl_uint iters = 65536;
  int repeats = 5;
};

Args parse_args(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    std::string key = argv[i];
    auto need_value = [&](const char* name) -> const char* {
      if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + name);
      return argv[++i];
    };
    if (key == "--device") args.device = need_value("--device");
    else if (key == "--global") args.global = static_cast<std::size_t>(std::stoull(need_value("--global")));
    else if (key == "--local") args.local = static_cast<std::size_t>(std::stoull(need_value("--local")));
    else if (key == "--iters") args.iters = static_cast<cl_uint>(std::stoul(need_value("--iters")));
    else if (key == "--repeats") args.repeats = std::stoi(need_value("--repeats"));
    else throw std::runtime_error("unknown arg: " + key);
  }
  if (args.repeats <= 0) throw std::runtime_error("repeats must be positive");
  if (args.global == 0 || args.local == 0 || args.global % args.local != 0) {
    throw std::runtime_error("global must be a positive multiple of local");
  }
  return args;
}

const char* kSource = R"CLC(
__kernel void busy_kernel(__global float* out, const uint iters, const float seed) {
  const size_t gid = get_global_id(0);
  float x = seed + (float)(gid & 1023) * 0.000977f;
  for (uint i = 0; i < iters; ++i) {
    x = fma(x, 1.000000119f, 0.000000131f);
    x = x - floor(x * 0.000244140625f) * 4096.0f;
  }
  out[gid] = x;
}
)CLC";

struct RunTimes {
  double wall_ms = 0.0;
  double event_a_us = 0.0;
  double event_b_us = 0.0;
};

RunTimes run_serial(Api& api, cl_command_queue queue, cl_kernel ka, cl_kernel kb,
                    std::size_t global, std::size_t local) {
  cl_event ea = nullptr;
  cl_event eb = nullptr;
  auto begin = std::chrono::steady_clock::now();
  check(api.clEnqueueNDRangeKernel(queue, ka, 1, nullptr, &global, &local,
                                   0, nullptr, &ea),
        "clEnqueueNDRangeKernel(serial a)");
  check(api.clEnqueueNDRangeKernel(queue, kb, 1, nullptr, &global, &local,
                                   0, nullptr, &eb),
        "clEnqueueNDRangeKernel(serial b)");
  check(api.clFinish(queue), "clFinish(serial)");
  auto end = std::chrono::steady_clock::now();
  RunTimes t;
  t.wall_ms = std::chrono::duration<double, std::milli>(end - begin).count();
  t.event_a_us = event_us(api, ea);
  t.event_b_us = event_us(api, eb);
  api.clReleaseEvent(ea);
  api.clReleaseEvent(eb);
  return t;
}

RunTimes run_parallel(Api& api, cl_command_queue qa, cl_command_queue qb,
                      cl_kernel ka, cl_kernel kb, std::size_t global,
                      std::size_t local) {
  cl_event ea = nullptr;
  cl_event eb = nullptr;
  auto begin = std::chrono::steady_clock::now();
  check(api.clEnqueueNDRangeKernel(qa, ka, 1, nullptr, &global, &local,
                                   0, nullptr, &ea),
        "clEnqueueNDRangeKernel(parallel a)");
  check(api.clEnqueueNDRangeKernel(qb, kb, 1, nullptr, &global, &local,
                                   0, nullptr, &eb),
        "clEnqueueNDRangeKernel(parallel b)");
  check(api.clFinish(qa), "clFinish(parallel a)");
  check(api.clFinish(qb), "clFinish(parallel b)");
  auto end = std::chrono::steady_clock::now();
  RunTimes t;
  t.wall_ms = std::chrono::duration<double, std::milli>(end - begin).count();
  t.event_a_us = event_us(api, ea);
  t.event_b_us = event_us(api, eb);
  api.clReleaseEvent(ea);
  api.clReleaseEvent(eb);
  return t;
}

double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

int main(int argc, char** argv) {
  try {
    const Args args = parse_args(argc, argv);
    Api api;
    Selection sel = select_device(api, args.device);
    cl_int err = CL_SUCCESS_VALUE;
    cl_context context = api.clCreateContext(nullptr, 1, &sel.device, nullptr, nullptr, &err);
    check(err, "clCreateContext");
    cl_command_queue q1 = api.clCreateCommandQueue(
        context, sel.device, CL_QUEUE_PROFILING_ENABLE_VALUE, &err);
    check(err, "clCreateCommandQueue(q1)");
    cl_command_queue q2 = api.clCreateCommandQueue(
        context, sel.device, CL_QUEUE_PROFILING_ENABLE_VALUE, &err);
    check(err, "clCreateCommandQueue(q2)");
    const std::size_t source_len = std::strlen(kSource);
    cl_program program =
        api.clCreateProgramWithSource(context, 1, &kSource, &source_len, &err);
    check(err, "clCreateProgramWithSource");
    err = api.clBuildProgram(program, 1, &sel.device, "", nullptr, nullptr);
    if (err != CL_SUCCESS_VALUE) {
      std::size_t log_size = 0;
      api.clGetProgramBuildInfo(program, sel.device, CL_PROGRAM_BUILD_LOG_VALUE,
                                0, nullptr, &log_size);
      std::string log(log_size, '\0');
      if (log_size > 0) {
        api.clGetProgramBuildInfo(program, sel.device, CL_PROGRAM_BUILD_LOG_VALUE,
                                  log.size(), log.data(), nullptr);
      }
      throw std::runtime_error("clBuildProgram failed: " + std::to_string(err) + "\n" + log);
    }
    cl_kernel ka = api.clCreateKernel(program, "busy_kernel", &err);
    check(err, "clCreateKernel(a)");
    cl_kernel kb = api.clCreateKernel(program, "busy_kernel", &err);
    check(err, "clCreateKernel(b)");
    cl_mem out_a = api.clCreateBuffer(
        context, CL_MEM_WRITE_ONLY_VALUE, args.global * sizeof(float), nullptr, &err);
    check(err, "clCreateBuffer(out_a)");
    cl_mem out_b = api.clCreateBuffer(
        context, CL_MEM_WRITE_ONLY_VALUE, args.global * sizeof(float), nullptr, &err);
    check(err, "clCreateBuffer(out_b)");
    const float seed_a = 1.0f;
    const float seed_b = 3.0f;
    check(api.clSetKernelArg(ka, 0, sizeof(out_a), &out_a), "clSetKernelArg(a0)");
    check(api.clSetKernelArg(ka, 1, sizeof(args.iters), &args.iters), "clSetKernelArg(a1)");
    check(api.clSetKernelArg(ka, 2, sizeof(seed_a), &seed_a), "clSetKernelArg(a2)");
    check(api.clSetKernelArg(kb, 0, sizeof(out_b), &out_b), "clSetKernelArg(b0)");
    check(api.clSetKernelArg(kb, 1, sizeof(args.iters), &args.iters), "clSetKernelArg(b1)");
    check(api.clSetKernelArg(kb, 2, sizeof(seed_b), &seed_b), "clSetKernelArg(b2)");

    (void)run_serial(api, q1, ka, kb, args.global, args.local);
    std::vector<double> serial_wall;
    std::vector<double> parallel_wall;
    std::vector<double> serial_device_sum;
    std::vector<double> parallel_device_max;
    for (int i = 0; i < args.repeats; ++i) {
      const auto s = run_serial(api, q1, ka, kb, args.global, args.local);
      const auto p = run_parallel(api, q1, q2, ka, kb, args.global, args.local);
      serial_wall.push_back(s.wall_ms);
      parallel_wall.push_back(p.wall_ms);
      serial_device_sum.push_back((s.event_a_us + s.event_b_us) / 1000.0);
      parallel_device_max.push_back(std::max(p.event_a_us, p.event_b_us) / 1000.0);
    }
    const double serial_median = median(serial_wall);
    const double parallel_median = median(parallel_wall);
    const double speedup = serial_median / parallel_median;
    const bool overlap_observed = speedup >= 1.20;

    std::cout << "{";
    std::cout << "\"schema\":\"intel-qwen36-opencl-queue-overlap-v0\",";
    std::cout << "\"platform\":\"" << sel.platform_name << "\",";
    std::cout << "\"device\":\"" << sel.device_name << "\",";
    std::cout << "\"global\":" << args.global << ",";
    std::cout << "\"local\":" << args.local << ",";
    std::cout << "\"iters\":" << args.iters << ",";
    std::cout << "\"repeats\":" << args.repeats << ",";
    std::cout << "\"serial_wall_median_ms\":" << serial_median << ",";
    std::cout << "\"parallel_wall_median_ms\":" << parallel_median << ",";
    std::cout << "\"serial_device_sum_median_ms\":" << median(serial_device_sum) << ",";
    std::cout << "\"parallel_device_max_median_ms\":" << median(parallel_device_max) << ",";
    std::cout << "\"parallel_speedup\":" << speedup << ",";
    std::cout << "\"overlap_observed\":" << (overlap_observed ? "true" : "false");
    std::cout << "}\n";

    api.clReleaseMemObject(out_b);
    api.clReleaseMemObject(out_a);
    api.clReleaseKernel(kb);
    api.clReleaseKernel(ka);
    api.clReleaseProgram(program);
    api.clReleaseCommandQueue(q2);
    api.clReleaseCommandQueue(q1);
    api.clReleaseContext(context);
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "error: " << ex.what() << "\n";
    return 1;
  }
}
'''


def write_summary(out_dir: Path, result: dict[str, Any], checks: list[dict[str, Any]]) -> None:
  lines = [
      "# OpenCL queue overlap",
      "",
      f"- device: `{result.get('device')}`",
      f"- serial wall median: `{result.get('serial_wall_median_ms')}` ms",
      f"- parallel wall median: `{result.get('parallel_wall_median_ms')}` ms",
      f"- parallel speedup: `{result.get('parallel_speedup')}`",
      f"- overlap observed: `{result.get('overlap_observed')}`",
      "",
      "## Checks",
      "",
  ]
  for check in checks:
    lines.append(f"- {check['name']}: `{check['pass']}`")
  (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--target", "--host", dest="host", metavar="TARGET", default=DEFAULT_HOST)
  parser.add_argument("--env-script", default=DEFAULT_ENV_SCRIPT)
  parser.add_argument("--staging-root", "--remote-root", dest="remote_root", metavar="PATH", default=DEFAULT_REMOTE_ROOT)
  parser.add_argument("--device", default="B390")
  parser.add_argument("--global-size", type=int, default=262144)
  parser.add_argument("--local-size", type=int, default=256)
  parser.add_argument("--iters", type=int, default=65536)
  parser.add_argument("--repeats", type=int, default=5)
  parser.add_argument("--timeout-s", type=int, default=180)
  args = parser.parse_args()

  stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
  out_dir = ROOT / "output" / f"opencl-queue-overlap-{stamp}"
  raw_dir = out_dir / "raw"
  raw_dir.mkdir(parents=True, exist_ok=True)
  local_cpp = raw_dir / "opencl_queue_overlap.cpp"
  local_cpp.write_text(CPP_SOURCE, encoding="utf-8")

  remote_dir = f"{args.remote_root.rstrip('/')}/opencl-queue-overlap-{stamp}"
  setup = iq36_local.run_target(
      args.host,
      f"rm -rf {shlex.quote(remote_dir)} && mkdir -p {shlex.quote(remote_dir + '/build')}",
      args.timeout_s,
  )
  transfer = (
      iq36_local.copy_to(
          args.host, local_cpp, f"{remote_dir}/opencl_queue_overlap.cpp",
          args.timeout_s,
      )
      if setup.get("returncode") == 0
      else {"returncode": None, "stderr": "setup failed"}
  )
  compile_cmd = " && ".join([
      f"source {shlex.quote(args.env_script)}",
      (
          "g++ -O2 -std=c++17 "
          f"{shlex.quote(remote_dir + '/opencl_queue_overlap.cpp')} "
          f"-ldl -o {shlex.quote(remote_dir + '/build/opencl_queue_overlap')}"
      ),
  ])
  compile_result = (
      iq36_local.run_target(args.host, f"bash -lc {shlex.quote(compile_cmd)}", args.timeout_s)
      if transfer.get("returncode") == 0
      else {"returncode": None, "stderr": "transfer failed"}
  )
  run_argv = [
      f"{remote_dir}/build/opencl_queue_overlap",
      "--device", args.device,
      "--global", str(args.global_size),
      "--local", str(args.local_size),
      "--iters", str(args.iters),
      "--repeats", str(args.repeats),
  ]
  run_cmd = " && ".join([
      f"source {shlex.quote(args.env_script)}",
      " ".join(shlex.quote(part) for part in run_argv),
  ])
  run_result = (
      iq36_local.run_target(args.host, f"bash -lc {shlex.quote(run_cmd)}", args.timeout_s)
      if compile_result.get("returncode") == 0
      else {"returncode": None, "stdout": "", "stderr": "compile failed"}
  )

  iq36_local.write_json(raw_dir / "setup.json", setup)
  iq36_local.write_json(raw_dir / "transfer.json", transfer)
  iq36_local.write_json(raw_dir / "compile.json", compile_result)
  iq36_local.write_json(raw_dir / "run.json", run_result)

  result: dict[str, Any] = {}
  parse_ok = False
  if run_result.get("returncode") == 0:
    try:
      result = json.loads(str(run_result.get("stdout", "")).strip().splitlines()[-1])
      parse_ok = isinstance(result, dict)
    except (IndexError, json.JSONDecodeError):
      parse_ok = False
  result.update({
      "local_cpp": str(local_cpp),
      "remote_dir": remote_dir,
      "host": args.host,
  })
  checks = [
      {"name": "remote_dir_created", "pass": setup.get("returncode") == 0},
      {"name": "source_transferred", "pass": transfer.get("returncode") == 0},
      {"name": "compiled", "pass": compile_result.get("returncode") == 0},
      {"name": "ran", "pass": run_result.get("returncode") == 0},
      {"name": "json_parsed", "pass": parse_ok},
      {"name": "overlap_observed", "pass": bool(result.get("overlap_observed"))},
  ]
  iq36_local.write_json(out_dir / "result.json", result)
  iq36_local.write_json(out_dir / "checks.json", checks)
  iq36_local.write_json(out_dir / "manifest.json", {
      "schema": "intel-qwen36-opencl-queue-overlap-manifest-v0",
      "created_at": datetime.now(timezone.utc).isoformat(),
      "host": args.host,
      "remote_dir": remote_dir,
      "device": args.device,
      "global_size": args.global_size,
      "local_size": args.local_size,
      "iters": args.iters,
      "repeats": args.repeats,
      "cpp_sha256": iq36_local.sha256_file(local_cpp),
      "checks": checks,
  })
  write_summary(out_dir, result, checks)
  print(out_dir)
  if not all(check["pass"] for check in checks[:5]):
    return 1
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
