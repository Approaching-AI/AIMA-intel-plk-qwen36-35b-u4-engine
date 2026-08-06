#include "intel_qwen36/gpu_q4x8_matvec.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace iq36 {
namespace {

using cl_bool = std::uint32_t;
using cl_command_queue = struct _cl_command_queue*;
using cl_command_queue_properties = std::uint64_t;
using cl_context = struct _cl_context*;
using cl_context_properties = std::intptr_t;
using cl_device_id = struct _cl_device_id*;
using cl_device_info = std::uint32_t;
using cl_device_type = std::uint64_t;
using cl_event = struct _cl_event*;
using cl_int = std::int32_t;
using cl_kernel = struct _cl_kernel*;
using cl_mem = struct _cl_mem*;
using cl_mem_flags = std::uint64_t;
using cl_platform_id = struct _cl_platform_id*;
using cl_platform_info = std::uint32_t;
using cl_profiling_info = std::uint32_t;
using cl_program = struct _cl_program*;
using cl_program_build_info = std::uint32_t;
using cl_uint = std::uint32_t;
using cl_ulong = std::uint64_t;

constexpr cl_int kClSuccess = 0;
constexpr cl_bool kClFalse = 0;
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

const char* kQ4KCpuOrderSource = R"ZCL(
#pragma OPENCL FP_CONTRACT OFF

float half_to_float(ushort h) {
  uint sign = ((uint)h & 0x8000U) << 16;
  uint exp = ((uint)h >> 10) & 0x1FU;
  uint mant = (uint)h & 0x03FFU;
  uint out;
  if (exp == 0U) {
    if (mant == 0U) {
      out = sign;
    } else {
      exp = 1U;
      while ((mant & 0x0400U) == 0U) {
        mant <<= 1U;
        exp -= 1U;
      }
      mant &= 0x03FFU;
      out = sign | ((exp + 112U) << 23) | (mant << 13);
    }
  } else if (exp == 31U) {
    out = sign | 0x7F800000U | (mant << 13);
  } else {
    out = sign | ((exp + 112U) << 23) | (mant << 13);
  }
  return as_float(out);
}

ushort load_le16(__global const uchar* p) {
  return (ushort)p[0] | ((ushort)p[1] << 8);
}

uint load_le32(__global const uchar* p) {
  return (uint)p[0] | ((uint)p[1] << 8) |
         ((uint)p[2] << 16) | ((uint)p[3] << 24);
}

uchar byte_from_word(uint value, uint index) {
  return (uchar)((value >> (index * 8U)) & 0xFFU);
}

__kernel void q4k_cpu_order_matvec(__global const uchar* raw,
                                   __global const char* q8_qs,
                                   __global const short* q8_bsums,
                                   __global const float* q8_d,
                                   uint blocks_per_row,
                                   uint rows,
                                   __global float* out) {
  const uint row = (uint)get_global_id(0);
  if (row >= rows) {
    return;
  }

  float sums[8];
  for (int lane = 0; lane < 8; ++lane) {
    sums[lane] = 0.0f;
  }
  float min_sum = 0.0f;

  for (uint block_index = 0; block_index < blocks_per_row; ++block_index) {
    __global const uchar* block =
        raw + ((ulong)row * (ulong)blocks_per_row + (ulong)block_index) * 144UL;

    uchar aux[256];
    uint aux_pos = 0U;
    __global const uchar* q4 = block + 16;
    for (int group = 0; group < 4; ++group) {
      for (int lane = 0; lane < 32; ++lane) {
        aux[aux_pos + (uint)lane] = q4[lane] & 0x0FU;
      }
      aux_pos += 32U;
      for (int lane = 0; lane < 32; ++lane) {
        aux[aux_pos + (uint)lane] = q4[lane] >> 4;
      }
      aux_pos += 32U;
      q4 += 32;
    }

    const uint kMask1 = 0x3f3f3f3fU;
    const uint kMask2 = 0x0f0f0f0fU;
    const uint kMask3 = 0x03030303U;
    uint u0 = load_le32(block + 4);
    uint u1 = load_le32(block + 8);
    uint u2 = load_le32(block + 12);
    uint u3 = ((u2 >> 4) & kMask2) | (((u1 >> 6) & kMask3) << 4);
    const uint aux_scales = u1 & kMask1;
    u1 = (u2 & kMask2) | (((u0 >> 6) & kMask3) << 4);
    u2 = aux_scales;
    u0 &= kMask1;

    uchar scales[8];
    scales[0] = byte_from_word(u0, 0U);
    scales[1] = byte_from_word(u0, 1U);
    scales[2] = byte_from_word(u0, 2U);
    scales[3] = byte_from_word(u0, 3U);
    scales[4] = byte_from_word(u1, 0U);
    scales[5] = byte_from_word(u1, 1U);
    scales[6] = byte_from_word(u1, 2U);
    scales[7] = byte_from_word(u1, 3U);
    uchar mins[8];
    mins[0] = byte_from_word(u2, 0U);
    mins[1] = byte_from_word(u2, 1U);
    mins[2] = byte_from_word(u2, 2U);
    mins[3] = byte_from_word(u2, 3U);
    mins[4] = byte_from_word(u3, 0U);
    mins[5] = byte_from_word(u3, 1U);
    mins[6] = byte_from_word(u3, 2U);
    mins[7] = byte_from_word(u3, 3U);

    int grouped_min_sum = 0;
    __global const short* block_bsums = q8_bsums + (ulong)block_index * 16UL;
    for (int group = 0; group < 16; ++group) {
      grouped_min_sum += (int)block_bsums[group] * (int)mins[group / 2];
    }

    int lane_sums[8];
    for (int lane = 0; lane < 8; ++lane) {
      lane_sums[lane] = 0;
    }
    __global const char* block_q8 = q8_qs + (ulong)block_index * 256UL;
    uint q8_pos = 0U;
    aux_pos = 0U;
    int scale_index = 0;
    for (int group = 0; group < 8; ++group) {
      const int scale = (int)scales[scale_index++];
      for (int repeat = 0; repeat < 4; ++repeat) {
        for (int lane = 0; lane < 8; ++lane) {
          lane_sums[lane] +=
              scale * ((int)block_q8[q8_pos + (uint)lane] *
                       (int)aux[aux_pos + (uint)lane]);
        }
        q8_pos += 8U;
        aux_pos += 8U;
      }
    }

    const float d = half_to_float(load_le16(block)) * q8_d[block_index];
    for (int lane = 0; lane < 8; ++lane) {
      sums[lane] += d * (float)lane_sums[lane];
    }
    const float dmin = half_to_float(load_le16(block + 2)) * q8_d[block_index];
    min_sum -= dmin * (float)grouped_min_sum;
  }

  float sum = min_sum;
  for (int lane = 0; lane < 8; ++lane) {
    sum += sums[lane];
  }
  out[row] = sum;
}
)ZCL";

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

struct SelectedDevice {
  cl_platform_id platform = nullptr;
  cl_device_id device = nullptr;
  std::string platform_name;
  std::string device_name;
};

std::string DeviceString(OpenClApi& api, cl_device_id device, cl_device_info info) {
  std::size_t size = 0;
  Check(api.clGetDeviceInfo(device, info, 0, nullptr, &size), "clGetDeviceInfo(size)");
  std::string out(size, '\0');
  Check(api.clGetDeviceInfo(device, info, size, out.data(), nullptr), "clGetDeviceInfo(value)");
  if (!out.empty() && out.back() == '\0') out.pop_back();
  return out;
}

std::string PlatformString(OpenClApi& api, cl_platform_id platform, cl_platform_info info) {
  std::size_t size = 0;
  Check(api.clGetPlatformInfo(platform, info, 0, nullptr, &size), "clGetPlatformInfo(size)");
  std::string out(size, '\0');
  Check(api.clGetPlatformInfo(platform, info, size, out.data(), nullptr),
        "clGetPlatformInfo(value)");
  if (!out.empty() && out.back() == '\0') out.pop_back();
  return out;
}

SelectedDevice SelectDevice(OpenClApi& api, const std::string& device_substring) {
  cl_uint platform_count = 0;
  Check(api.clGetPlatformIDs(0, nullptr, &platform_count), "clGetPlatformIDs(count)");
  std::vector<cl_platform_id> platforms(platform_count);
  Check(api.clGetPlatformIDs(platform_count, platforms.data(), nullptr),
        "clGetPlatformIDs(list)");
  for (cl_platform_id platform : platforms) {
    cl_uint device_count = 0;
    if (api.clGetDeviceIDs(platform, kClDeviceTypeGpu, 0, nullptr, &device_count) !=
            kClSuccess ||
        device_count == 0) {
      continue;
    }
    std::vector<cl_device_id> devices(device_count);
    Check(api.clGetDeviceIDs(platform, kClDeviceTypeGpu, device_count,
                             devices.data(), nullptr),
          "clGetDeviceIDs(list)");
    for (cl_device_id device : devices) {
      const std::string name = DeviceString(api, device, kClDeviceName);
      if (device_substring.empty() ||
          name.find(device_substring) != std::string::npos) {
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
  Check(api.clGetProgramBuildInfo(program, device, kClProgramBuildLog, size,
                                  out.data(), nullptr),
        "clGetProgramBuildInfo(log)");
  if (!out.empty() && out.back() == '\0') out.pop_back();
  return out;
}

bool OpenClEventCollectionEnabled() {
  static const bool disabled =
      std::getenv("IQ36_OPENCL_SKIP_EVENT_US") != nullptr ||
      std::getenv("IQ36_OPENCL_NO_QUEUE_PROFILING") != nullptr;
  return !disabled;
}

cl_event* EventOut(cl_event* event) {
  return OpenClEventCollectionEnabled() ? event : nullptr;
}

void ReleaseEvent(OpenClApi& api, cl_event* event) {
  if (*event != nullptr) {
    api.clReleaseEvent(*event);
    *event = nullptr;
  }
}

double EventUs(OpenClApi& api, cl_event event) {
  if (event == nullptr || !OpenClEventCollectionEnabled()) {
    return 0.0;
  }
  cl_ulong start = 0;
  cl_ulong end = 0;
  Check(api.clGetEventProfilingInfo(event, kClProfilingCommandStart, sizeof(start),
                                    &start, nullptr),
        "clGetEventProfilingInfo(start)");
  Check(api.clGetEventProfilingInfo(event, kClProfilingCommandEnd, sizeof(end),
                                    &end, nullptr),
        "clGetEventProfilingInfo(end)");
  return static_cast<double>(end - start) / 1000.0;
}

cl_command_queue_properties QueueProperties() {
  static const bool no_queue_profiling =
      std::getenv("IQ36_OPENCL_NO_QUEUE_PROFILING") != nullptr;
  return no_queue_profiling ? 0 : kClQueueProfilingEnable;
}

void ReleaseMem(OpenClApi& api, cl_mem* mem) {
  if (*mem) {
    api.clReleaseMemObject(*mem);
    *mem = nullptr;
  }
}

void ValidateQ4KCpuOrderShape(const std::vector<std::uint8_t>& raw,
                              std::uint64_t rows,
                              std::uint64_t blocks_per_row) {
  Require(rows > 0 && rows <= 65535, "Q4 CPU-order rows out of range");
  Require(blocks_per_row > 0 && blocks_per_row <= 255,
          "Q4 CPU-order blocks_per_row out of range");
  Require(raw.size() == static_cast<std::size_t>(rows * blocks_per_row * 144),
          "Q4 CPU-order raw byte size mismatch");
}

void ValidateQ4KCpuOrderQ8(const GpuQ8KInputPlanes& q8,
                           std::uint64_t blocks_per_row,
                           int repeat) {
  Require(q8.qs.size() == static_cast<std::size_t>(blocks_per_row * 256),
          "Q4 CPU-order Q8 qs size mismatch");
  Require(q8.bsums.size() == static_cast<std::size_t>(blocks_per_row * 16),
          "Q4 CPU-order Q8 bsums size mismatch");
  Require(q8.d.size() == static_cast<std::size_t>(blocks_per_row),
          "Q4 CPU-order Q8 scale size mismatch");
  Require(repeat > 0, "Q4 CPU-order repeat must be positive");
}

}  // namespace

class GpuQ4KCpuOrderMatvecRunner::Impl {
 public:
  struct ResidentRaw {
    cl_mem buffer = nullptr;
    std::uint64_t rows = 0;
    std::uint64_t blocks_per_row = 0;
    std::size_t raw_bytes = 0;
  };

  struct ScratchBuffer {
    cl_mem mem = nullptr;
    std::size_t bytes = 0;
    cl_mem_flags flags = 0;
  };

  explicit Impl(const std::string& device_substring) {
    selected_ = SelectDevice(api_, device_substring);
    cl_int err = kClSuccess;
    context_ =
        api_.clCreateContext(nullptr, 1, &selected_.device, nullptr, nullptr, &err);
    Check(err, "clCreateContext(Q4 CPU-order runner)");
    queue_ = api_.clCreateCommandQueue(
        context_, selected_.device, QueueProperties(), &err);
    Check(err, "clCreateCommandQueue(Q4 CPU-order runner)");
    const char* source = kQ4KCpuOrderSource;
    const std::size_t source_len = std::strlen(kQ4KCpuOrderSource);
    program_ = api_.clCreateProgramWithSource(context_, 1, &source, &source_len, &err);
    Check(err, "clCreateProgramWithSource(Q4 CPU-order runner)");
    const auto build_begin = std::chrono::steady_clock::now();
    err = api_.clBuildProgram(program_, 1, &selected_.device, "", nullptr, nullptr);
    const auto build_end = std::chrono::steady_clock::now();
    program_build_ms_ =
        std::chrono::duration<double, std::milli>(build_end - build_begin).count();
    build_log_ = BuildLog(api_, program_, selected_.device);
    Check(err, "clBuildProgram(Q4 CPU-order runner)");
    kernel_ = api_.clCreateKernel(program_, "q4k_cpu_order_matvec", &err);
    Check(err, "clCreateKernel(q4k_cpu_order_matvec runner)");
  }

  ~Impl() {
    ReleaseScratchBuffer(q8_qs_scratch_);
    ReleaseScratchBuffer(q8_bsums_scratch_);
    ReleaseScratchBuffer(q8_d_scratch_);
    ReleaseScratchBuffer(out_scratch_);
    ClearResidentRawQ4KCpuOrder();
    if (kernel_) api_.clReleaseKernel(kernel_);
    if (program_) api_.clReleaseProgram(program_);
    if (queue_) api_.clReleaseCommandQueue(queue_);
    if (context_) api_.clReleaseContext(context_);
  }

  const std::string& platform_name() const { return selected_.platform_name; }
  const std::string& device_name() const { return selected_.device_name; }
  const std::string& build_log() const { return build_log_; }
  double program_build_ms() const { return program_build_ms_; }

  void ReleaseScratchBuffer(ScratchBuffer& scratch) {
    ReleaseMem(api_, &scratch.mem);
    scratch.bytes = 0;
    scratch.flags = 0;
  }

  cl_mem EnsureScratchBuffer(ScratchBuffer& scratch,
                             std::size_t bytes,
                             cl_mem_flags flags,
                             const char* label) {
    Require(bytes > 0, std::string(label) + " scratch buffer must be nonempty");
    if (scratch.mem != nullptr && scratch.bytes == bytes &&
        scratch.flags == flags) {
      return scratch.mem;
    }
    ReleaseScratchBuffer(scratch);
    cl_int err = kClSuccess;
    scratch.mem = api_.clCreateBuffer(context_, flags, bytes, nullptr, &err);
    Check(err, std::string("clCreateBuffer(") + label + ")");
    scratch.bytes = bytes;
    scratch.flags = flags;
    return scratch.mem;
  }

  std::uint64_t UploadRawQ4KCpuOrder(const std::vector<std::uint8_t>& raw,
                                     std::uint64_t rows,
                                     std::uint64_t blocks_per_row) {
    ValidateQ4KCpuOrderShape(raw, rows, blocks_per_row);
    cl_int err = kClSuccess;
    cl_mem buffer =
        api_.clCreateBuffer(context_, kClMemReadOnly, raw.size(), nullptr, &err);
    Check(err, "clCreateBuffer(resident raw Q4 CPU-order)");
    try {
      Check(api_.clEnqueueWriteBuffer(queue_, buffer, kClTrue, 0, raw.size(),
                                      raw.data(), 0, nullptr, nullptr),
            "clEnqueueWriteBuffer(resident raw Q4 CPU-order)");
    } catch (...) {
      ReleaseMem(api_, &buffer);
      throw;
    }
    const std::uint64_t handle = next_handle_++;
    resident_.emplace(handle, ResidentRaw{buffer, rows, blocks_per_row, raw.size()});
    return handle;
  }

  GpuQ4KCpuOrderMatvecRun RunResidentRawQ4KCpuOrder(
      std::uint64_t handle,
      const GpuQ8KInputPlanes& q8,
      int repeat) {
    const auto& resident = ResidentForHandle(handle);
    ValidateQ4KCpuOrderQ8(q8, resident.blocks_per_row, repeat);
    GpuQ4KCpuOrderMatvecRun run;
    run.output.assign(static_cast<std::size_t>(resident.rows), 0.0f);
    run.platform_name = selected_.platform_name;
    run.device_name = selected_.device_name;
    run.build_log = build_log_;
    run.program_build_ms = 0.0;

    cl_mem q8_qs_buffer = EnsureScratchBuffer(
        q8_qs_scratch_, q8.qs.size() * sizeof(std::int8_t), kClMemReadOnly,
        "resident Q4 CPU-order q8 qs");
    cl_mem q8_bsums_buffer = EnsureScratchBuffer(
        q8_bsums_scratch_, q8.bsums.size() * sizeof(std::int16_t),
        kClMemReadOnly, "resident Q4 CPU-order q8 bsums");
    cl_mem q8_d_buffer = EnsureScratchBuffer(
        q8_d_scratch_, q8.d.size() * sizeof(float), kClMemReadOnly,
        "resident Q4 CPU-order q8 d");
    cl_mem out_buffer = EnsureScratchBuffer(
        out_scratch_, run.output.size() * sizeof(float), kClMemWriteOnly,
        "resident Q4 CPU-order out");
    Check(api_.clEnqueueWriteBuffer(queue_, q8_qs_buffer, kClFalse, 0,
                                    q8.qs.size() * sizeof(std::int8_t),
                                    q8.qs.data(), 0, nullptr, nullptr),
          "clEnqueueWriteBuffer(resident q8 qs Q4 CPU-order)");
    Check(api_.clEnqueueWriteBuffer(queue_, q8_bsums_buffer, kClFalse, 0,
                                    q8.bsums.size() * sizeof(std::int16_t),
                                    q8.bsums.data(), 0, nullptr, nullptr),
          "clEnqueueWriteBuffer(resident q8 bsums Q4 CPU-order)");
    Check(api_.clEnqueueWriteBuffer(queue_, q8_d_buffer, kClFalse, 0,
                                    q8.d.size() * sizeof(float), q8.d.data(), 0,
                                    nullptr, nullptr),
          "clEnqueueWriteBuffer(resident q8 d Q4 CPU-order)");
    const bool read_in_kernel = repeat == 1;
    run.timing = RunKernel(
        resident, q8_qs_buffer, q8_bsums_buffer, q8_d_buffer, out_buffer,
        repeat, read_in_kernel ? run.output.data() : nullptr,
        read_in_kernel ? run.output.size() * sizeof(float) : 0);
    if (!read_in_kernel) {
      Check(api_.clEnqueueReadBuffer(queue_, out_buffer, kClTrue, 0,
                                     run.output.size() * sizeof(float),
                                     run.output.data(), 0, nullptr, nullptr),
            "clEnqueueReadBuffer(resident Q4 CPU-order out)");
    }
    return run;
  }

  void ClearResidentRawQ4KCpuOrder() {
    for (auto& item : resident_) {
      ReleaseMem(api_, &item.second.buffer);
    }
    resident_.clear();
  }

 private:
  const ResidentRaw& ResidentForHandle(std::uint64_t handle) const {
    const auto it = resident_.find(handle);
    Require(it != resident_.end(), "resident Q4 CPU-order handle not found");
    return it->second;
  }

  GpuQ4KCpuOrderMatvecTiming RunKernel(const ResidentRaw& resident,
                                       cl_mem q8_qs_buffer,
                                       cl_mem q8_bsums_buffer,
                                       cl_mem q8_d_buffer,
                                       cl_mem out_buffer,
                                       int repeat,
                                       float* read_output = nullptr,
                                       std::size_t read_output_bytes = 0) {
    const cl_uint blocks_arg = static_cast<cl_uint>(resident.blocks_per_row);
    const cl_uint rows_arg = static_cast<cl_uint>(resident.rows);
    Check(api_.clSetKernelArg(kernel_, 0, sizeof(resident.buffer), &resident.buffer),
          "clSetKernelArg(resident Q4 CPU-order raw)");
    Check(api_.clSetKernelArg(kernel_, 1, sizeof(q8_qs_buffer), &q8_qs_buffer),
          "clSetKernelArg(resident Q4 CPU-order q8_qs)");
    Check(api_.clSetKernelArg(kernel_, 2, sizeof(q8_bsums_buffer), &q8_bsums_buffer),
          "clSetKernelArg(resident Q4 CPU-order q8_bsums)");
    Check(api_.clSetKernelArg(kernel_, 3, sizeof(q8_d_buffer), &q8_d_buffer),
          "clSetKernelArg(resident Q4 CPU-order q8_d)");
    Check(api_.clSetKernelArg(kernel_, 4, sizeof(blocks_arg), &blocks_arg),
          "clSetKernelArg(resident Q4 CPU-order blocks)");
    Check(api_.clSetKernelArg(kernel_, 5, sizeof(rows_arg), &rows_arg),
          "clSetKernelArg(resident Q4 CPU-order rows)");
    Check(api_.clSetKernelArg(kernel_, 6, sizeof(out_buffer), &out_buffer),
          "clSetKernelArg(resident Q4 CPU-order out)");

    const std::size_t global = static_cast<std::size_t>(resident.rows);
    double total_us = 0.0;
    double min_us = std::numeric_limits<double>::infinity();
    for (int i = 0; i < repeat; ++i) {
      cl_event event = nullptr;
      Check(api_.clEnqueueNDRangeKernel(queue_, kernel_, 1, nullptr, &global,
                                        nullptr, 0, nullptr,
                                        EventOut(&event)),
            "clEnqueueNDRangeKernel(resident Q4 CPU-order)");
      if (read_output != nullptr) {
        Require(read_output_bytes > 0,
                "resident Q4 CPU-order read output size is zero");
        Check(api_.clEnqueueReadBuffer(queue_, out_buffer, kClTrue, 0,
                                       read_output_bytes, read_output, 0,
                                       nullptr, nullptr),
              "clEnqueueReadBuffer(resident Q4 CPU-order out)");
      } else {
        Check(api_.clFinish(queue_), "clFinish(resident Q4 CPU-order)");
      }
      const double elapsed = EventUs(api_, event);
      ReleaseEvent(api_, &event);
      total_us += elapsed;
      min_us = std::min(min_us, elapsed);
    }
    GpuQ4KCpuOrderMatvecTiming timing;
    timing.min_us = min_us;
    timing.mean_us = total_us / static_cast<double>(repeat);
    timing.effective_raw_gb_s =
        static_cast<double>(resident.raw_bytes) / (min_us / 1e6) / 1e9;
    timing.global_work_items = resident.rows;
    return timing;
  }

  OpenClApi api_;
  SelectedDevice selected_;
  cl_context context_ = nullptr;
  cl_command_queue queue_ = nullptr;
  cl_program program_ = nullptr;
  cl_kernel kernel_ = nullptr;
  std::string build_log_;
  double program_build_ms_ = 0.0;
  ScratchBuffer q8_qs_scratch_;
  ScratchBuffer q8_bsums_scratch_;
  ScratchBuffer q8_d_scratch_;
  ScratchBuffer out_scratch_;
  std::unordered_map<std::uint64_t, ResidentRaw> resident_;
  std::uint64_t next_handle_ = 1;
};

GpuQ4KCpuOrderMatvecRunner::GpuQ4KCpuOrderMatvecRunner(
    std::string device_substring)
    : impl_(std::make_unique<Impl>(device_substring)) {}

GpuQ4KCpuOrderMatvecRunner::~GpuQ4KCpuOrderMatvecRunner() = default;
GpuQ4KCpuOrderMatvecRunner::GpuQ4KCpuOrderMatvecRunner(
    GpuQ4KCpuOrderMatvecRunner&&) noexcept = default;
GpuQ4KCpuOrderMatvecRunner& GpuQ4KCpuOrderMatvecRunner::operator=(
    GpuQ4KCpuOrderMatvecRunner&&) noexcept = default;

const std::string& GpuQ4KCpuOrderMatvecRunner::platform_name() const {
  return impl_->platform_name();
}
const std::string& GpuQ4KCpuOrderMatvecRunner::device_name() const {
  return impl_->device_name();
}
const std::string& GpuQ4KCpuOrderMatvecRunner::build_log() const {
  return impl_->build_log();
}
double GpuQ4KCpuOrderMatvecRunner::program_build_ms() const {
  return impl_->program_build_ms();
}
std::uint64_t GpuQ4KCpuOrderMatvecRunner::UploadRawQ4KCpuOrder(
    const std::vector<std::uint8_t>& raw,
    std::uint64_t rows,
    std::uint64_t blocks_per_row) {
  return impl_->UploadRawQ4KCpuOrder(raw, rows, blocks_per_row);
}
GpuQ4KCpuOrderMatvecRun GpuQ4KCpuOrderMatvecRunner::RunResidentRawQ4KCpuOrder(
    std::uint64_t handle,
    const GpuQ8KInputPlanes& q8,
    int repeat) {
  return impl_->RunResidentRawQ4KCpuOrder(handle, q8, repeat);
}
void GpuQ4KCpuOrderMatvecRunner::ClearResidentRawQ4KCpuOrder() {
  impl_->ClearResidentRawQ4KCpuOrder();
}

GpuQ4KCpuOrderMatvecRun RunQ4KCpuOrderMatvec(
    const std::vector<std::uint8_t>& raw,
    const GpuQ8KInputPlanes& q8,
    std::uint64_t rows,
    std::uint64_t blocks_per_row,
    const std::string& device_substring,
    int repeat) {
  Require(rows > 0 && rows <= 65535, "Q4 CPU-order rows out of range");
  Require(blocks_per_row > 0 && blocks_per_row <= 255,
          "Q4 CPU-order blocks_per_row out of range");
  Require(raw.size() == static_cast<std::size_t>(rows * blocks_per_row * 144),
          "Q4 CPU-order raw byte size mismatch");
  Require(q8.qs.size() == static_cast<std::size_t>(blocks_per_row * 256),
          "Q4 CPU-order Q8 qs size mismatch");
  Require(q8.bsums.size() == static_cast<std::size_t>(blocks_per_row * 16),
          "Q4 CPU-order Q8 bsums size mismatch");
  Require(q8.d.size() == static_cast<std::size_t>(blocks_per_row),
          "Q4 CPU-order Q8 scale size mismatch");
  Require(repeat > 0, "Q4 CPU-order repeat must be positive");

  OpenClApi api;
  GpuQ4KCpuOrderMatvecRun run;
  run.output.assign(static_cast<std::size_t>(rows), 0.0f);
  const auto selected = SelectDevice(api, device_substring);
  run.platform_name = selected.platform_name;
  run.device_name = selected.device_name;

  cl_int err = kClSuccess;
  cl_context context =
      api.clCreateContext(nullptr, 1, &selected.device, nullptr, nullptr, &err);
  Check(err, "clCreateContext(Q4 CPU-order)");
  cl_command_queue queue =
      api.clCreateCommandQueue(context, selected.device, QueueProperties(), &err);
  Check(err, "clCreateCommandQueue(Q4 CPU-order)");

  const char* source = kQ4KCpuOrderSource;
  const std::size_t source_len = std::strlen(kQ4KCpuOrderSource);
  cl_program program =
      api.clCreateProgramWithSource(context, 1, &source, &source_len, &err);
  Check(err, "clCreateProgramWithSource(Q4 CPU-order)");
  const auto build_begin = std::chrono::steady_clock::now();
  err = api.clBuildProgram(program, 1, &selected.device, "", nullptr, nullptr);
  const auto build_end = std::chrono::steady_clock::now();
  run.program_build_ms =
      std::chrono::duration<double, std::milli>(build_end - build_begin).count();
  run.build_log = BuildLog(api, program, selected.device);
  Check(err, "clBuildProgram(Q4 CPU-order)");
  cl_kernel kernel = api.clCreateKernel(program, "q4k_cpu_order_matvec", &err);
  Check(err, "clCreateKernel(q4k_cpu_order_matvec)");

  cl_mem raw_buffer =
      api.clCreateBuffer(context, kClMemReadOnly, raw.size(), nullptr, &err);
  Check(err, "clCreateBuffer(raw Q4 CPU-order)");
  cl_mem q8_qs_buffer = api.clCreateBuffer(
      context, kClMemReadOnly, q8.qs.size() * sizeof(std::int8_t), nullptr, &err);
  Check(err, "clCreateBuffer(q8 qs Q4 CPU-order)");
  cl_mem q8_bsums_buffer = api.clCreateBuffer(
      context, kClMemReadOnly, q8.bsums.size() * sizeof(std::int16_t), nullptr, &err);
  Check(err, "clCreateBuffer(q8 bsums Q4 CPU-order)");
  cl_mem q8_d_buffer = api.clCreateBuffer(
      context, kClMemReadOnly, q8.d.size() * sizeof(float), nullptr, &err);
  Check(err, "clCreateBuffer(q8 d Q4 CPU-order)");
  cl_mem out_buffer = api.clCreateBuffer(
      context, kClMemWriteOnly, run.output.size() * sizeof(float), nullptr, &err);
  Check(err, "clCreateBuffer(out Q4 CPU-order)");

  Check(api.clEnqueueWriteBuffer(queue, raw_buffer, kClTrue, 0, raw.size(),
                                 raw.data(), 0, nullptr, nullptr),
        "clEnqueueWriteBuffer(raw Q4 CPU-order)");
  Check(api.clEnqueueWriteBuffer(queue, q8_qs_buffer, kClTrue, 0,
                                 q8.qs.size() * sizeof(std::int8_t),
                                 q8.qs.data(), 0, nullptr, nullptr),
        "clEnqueueWriteBuffer(q8 qs Q4 CPU-order)");
  Check(api.clEnqueueWriteBuffer(queue, q8_bsums_buffer, kClTrue, 0,
                                 q8.bsums.size() * sizeof(std::int16_t),
                                 q8.bsums.data(), 0, nullptr, nullptr),
        "clEnqueueWriteBuffer(q8 bsums Q4 CPU-order)");
  Check(api.clEnqueueWriteBuffer(queue, q8_d_buffer, kClTrue, 0,
                                 q8.d.size() * sizeof(float), q8.d.data(), 0,
                                 nullptr, nullptr),
        "clEnqueueWriteBuffer(q8 d Q4 CPU-order)");

  const cl_uint blocks_arg = static_cast<cl_uint>(blocks_per_row);
  const cl_uint rows_arg = static_cast<cl_uint>(rows);
  Check(api.clSetKernelArg(kernel, 0, sizeof(raw_buffer), &raw_buffer),
        "clSetKernelArg(Q4 CPU-order raw)");
  Check(api.clSetKernelArg(kernel, 1, sizeof(q8_qs_buffer), &q8_qs_buffer),
        "clSetKernelArg(Q4 CPU-order q8_qs)");
  Check(api.clSetKernelArg(kernel, 2, sizeof(q8_bsums_buffer), &q8_bsums_buffer),
        "clSetKernelArg(Q4 CPU-order q8_bsums)");
  Check(api.clSetKernelArg(kernel, 3, sizeof(q8_d_buffer), &q8_d_buffer),
        "clSetKernelArg(Q4 CPU-order q8_d)");
  Check(api.clSetKernelArg(kernel, 4, sizeof(blocks_arg), &blocks_arg),
        "clSetKernelArg(Q4 CPU-order blocks)");
  Check(api.clSetKernelArg(kernel, 5, sizeof(rows_arg), &rows_arg),
        "clSetKernelArg(Q4 CPU-order rows)");
  Check(api.clSetKernelArg(kernel, 6, sizeof(out_buffer), &out_buffer),
        "clSetKernelArg(Q4 CPU-order out)");

  const std::size_t global = static_cast<std::size_t>(rows);
  double total_us = 0.0;
  double min_us = std::numeric_limits<double>::infinity();
  for (int i = 0; i < repeat; ++i) {
    cl_event event = nullptr;
    Check(api.clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global, nullptr,
                                     0, nullptr, EventOut(&event)),
          "clEnqueueNDRangeKernel(Q4 CPU-order)");
    Check(api.clFinish(queue), "clFinish(Q4 CPU-order)");
    const double elapsed = EventUs(api, event);
    ReleaseEvent(api, &event);
    total_us += elapsed;
    min_us = std::min(min_us, elapsed);
  }
  run.timing.min_us = min_us;
  run.timing.mean_us = total_us / static_cast<double>(repeat);
  run.timing.effective_raw_gb_s =
      static_cast<double>(raw.size()) / (min_us / 1e6) / 1e9;
  run.timing.global_work_items = rows;

  Check(api.clEnqueueReadBuffer(queue, out_buffer, kClTrue, 0,
                                run.output.size() * sizeof(float),
                                run.output.data(), 0, nullptr, nullptr),
        "clEnqueueReadBuffer(Q4 CPU-order out)");

  ReleaseMem(api, &out_buffer);
  ReleaseMem(api, &q8_d_buffer);
  ReleaseMem(api, &q8_bsums_buffer);
  ReleaseMem(api, &q8_qs_buffer);
  ReleaseMem(api, &raw_buffer);
  api.clReleaseKernel(kernel);
  api.clReleaseProgram(program);
  api.clReleaseCommandQueue(queue);
  api.clReleaseContext(context);
  return run;
}

}  // namespace iq36
