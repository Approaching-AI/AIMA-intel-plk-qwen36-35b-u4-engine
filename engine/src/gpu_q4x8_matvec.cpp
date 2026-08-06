#include "intel_qwen36/gpu_q4x8_matvec.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace iq36 {
namespace {

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
constexpr cl_bool kClFalse = 0;
constexpr cl_bool kClTrue = 1;
constexpr cl_device_type kClDeviceTypeGpu = 1ULL << 2;
constexpr cl_mem_flags kClMemReadOnly = 1ULL << 2;
constexpr cl_mem_flags kClMemWriteOnly = 1ULL << 1;
constexpr cl_mem_flags kClMemReadWrite = 1ULL;
constexpr cl_command_queue_properties kClQueueProfilingEnable = 1ULL << 1;
constexpr cl_platform_info kClPlatformName = 0x0902;
constexpr cl_device_info kClDeviceName = 0x102B;
constexpr cl_program_build_info kClProgramBuildLog = 0x1183;
constexpr cl_profiling_info kClProfilingCommandStart = 0x1282;
constexpr cl_profiling_info kClProfilingCommandEnd = 0x1283;

constexpr std::uint64_t kRowsInterleaved = 8;
constexpr std::uint64_t kQ4Kx8BlockBytes = 1152;
constexpr std::uint64_t kQ8QsPerBlock = 256;
constexpr std::uint64_t kQ8BsumsPerBlock = 16;
constexpr std::uint64_t kQ4KBlockBytes = 144;
constexpr std::uint64_t kQ6KBlockBytes = 210;
constexpr std::size_t kRmsNormScaleLocalSize = 2;

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
  cl_int (*clEnqueueCopyBuffer)(cl_command_queue, cl_mem, cl_mem, std::size_t, std::size_t, std::size_t, cl_uint, const cl_event*, cl_event*) = nullptr;
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
    clEnqueueCopyBuffer = LoadSym<decltype(clEnqueueCopyBuffer)>(lib, "clEnqueueCopyBuffer");
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

bool OpenClEventCollectionEnabled() {
  static const bool enabled =
      std::getenv("IQ36_OPENCL_SKIP_EVENT_US") != nullptr ||
      std::getenv("IQ36_OPENCL_NO_QUEUE_PROFILING") != nullptr;
  return !enabled;
}

cl_event* EventOut(cl_event* event) {
  return OpenClEventCollectionEnabled() ? event : nullptr;
}

bool DeferFfnDownFinishBundle() {
  static const bool enabled =
      std::getenv("IQ36_DEFER_FFN_DOWN_FINISH_BUNDLE") != nullptr;
  return enabled && !OpenClEventCollectionEnabled();
}

bool SelectedDownSubmitSplitProfile() {
  static const bool enabled =
      std::getenv("IQ36_SELECTED_DOWN_SUBMIT_SPLIT_PROFILE") != nullptr;
  return enabled;
}

bool AttentionFrontHandoffMatvecSubmitSplitProfile() {
  static const bool enabled =
      std::getenv("IQ36_ATTENTION_FRONT_HANDOFF_MATVEC_SUBMIT_SPLIT_PROFILE") != nullptr;
  return enabled;
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
  Check(api.clGetEventProfilingInfo(event, kClProfilingCommandStart, sizeof(start), &start, nullptr),
        "clGetEventProfilingInfo(start)");
  Check(api.clGetEventProfilingInfo(event, kClProfilingCommandEnd, sizeof(end), &end, nullptr),
        "clGetEventProfilingInfo(end)");
  return static_cast<double>(end - start) / 1000.0;
}

std::uint64_t WallNs(std::chrono::steady_clock::time_point begin,
                     std::chrono::steady_clock::time_point end) {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin)
          .count());
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

void ValidateRunInputs(const std::vector<std::uint8_t>& packed,
                       const std::vector<std::int8_t>& q8_qs,
                       const std::vector<std::int16_t>& q8_bsums,
                       const std::vector<float>& q8_d,
                       std::uint64_t rows,
                       std::uint64_t blocks_per_row,
                       int repeat) {
  Require(rows > 0 && rows % kRowsInterleaved == 0, "rows must be nonzero and divisible by 8");
  Require(blocks_per_row > 0, "blocks_per_row must be nonzero");
  Require(repeat > 0, "repeat must be positive");
  Require(packed.size() == rows / kRowsInterleaved * blocks_per_row * kQ4Kx8BlockBytes,
          "packed Q4_K x8 byte size does not match shape");
  Require(q8_qs.size() == blocks_per_row * kQ8QsPerBlock, "Q8_K qs size does not match blocks_per_row");
  Require(q8_bsums.size() == blocks_per_row * kQ8BsumsPerBlock,
          "Q8_K bsums size does not match blocks_per_row");
  Require(q8_d.size() == blocks_per_row, "Q8_K scale size does not match blocks_per_row");
}

void ValidatePackedQ4X8Inputs(const std::vector<std::uint8_t>& packed,
                              std::uint64_t rows,
                              std::uint64_t blocks_per_row) {
  Require(rows > 0 && rows % kRowsInterleaved == 0, "rows must be nonzero and divisible by 8");
  Require(blocks_per_row > 0, "blocks_per_row must be nonzero");
  Require(packed.size() == rows / kRowsInterleaved * blocks_per_row * kQ4Kx8BlockBytes,
          "packed Q4_K x8 byte size does not match shape");
}

void ValidateRawQ6KInputs(const std::vector<std::uint8_t>& raw,
                          std::uint64_t rows,
                          std::uint64_t blocks_per_row) {
  Require(rows > 0, "Q6_K rows must be nonzero");
  Require(blocks_per_row > 0, "Q6_K blocks_per_row must be nonzero");
  Require(raw.size() == rows * blocks_per_row * kQ6KBlockBytes,
          "raw Q6_K byte size does not match shape");
}

void ValidateRawQ4KCpuOrderInputs(const std::vector<std::uint8_t>& raw,
                                  std::uint64_t rows,
                                  std::uint64_t blocks_per_row) {
  Require(rows > 0, "Q4 CPU-order rows must be nonzero");
  Require(blocks_per_row > 0,
          "Q4 CPU-order blocks_per_row must be nonzero");
  Require(raw.size() == rows * blocks_per_row * kQ4KBlockBytes,
          "raw Q4 CPU-order byte size does not match shape");
}

struct Q6KRowstripeLayout {
  std::vector<std::uint8_t> bytes;
  std::uint64_t rows_per_tile = 0;
  std::uint64_t row_tile_count = 0;
};

Q6KRowstripeLayout BuildSelectedRawQ6KRowstripe(
    const std::vector<std::uint8_t>& raw,
    std::uint64_t rows_per_expert,
    std::uint64_t blocks_per_row,
    std::uint64_t selected_count,
    std::uint64_t rows_per_tile) {
  Require(rows_per_expert > 0, "Q6_K rowstripe rows_per_expert must be nonzero");
  Require(blocks_per_row > 0, "Q6_K rowstripe blocks_per_row must be nonzero");
  Require(selected_count > 0, "Q6_K rowstripe selected_count must be nonzero");
  Require(rows_per_tile > 0, "Q6_K rowstripe rows_per_tile must be nonzero");
  Require(raw.size() == selected_count * rows_per_expert * blocks_per_row *
                            kQ6KBlockBytes,
          "Q6_K rowstripe raw byte size does not match shape");
  struct Segment {
    std::uint64_t offset;
    std::uint64_t bytes;
  };
  constexpr Segment kSegments[] = {
      {0ULL, 64ULL},
      {128ULL, 32ULL},
      {64ULL, 64ULL},
      {160ULL, 32ULL},
      {192ULL, 16ULL},
      {208ULL, 2ULL},
  };
  Q6KRowstripeLayout layout;
  layout.rows_per_tile = rows_per_tile;
  layout.row_tile_count =
      (rows_per_expert + rows_per_tile - 1ULL) / rows_per_tile;
  layout.bytes.reserve(static_cast<std::size_t>(
      selected_count * layout.row_tile_count * blocks_per_row *
      rows_per_tile * kQ6KBlockBytes));
  for (std::uint64_t selected = 0; selected < selected_count; ++selected) {
    for (std::uint64_t row_tile = 0; row_tile < layout.row_tile_count;
         ++row_tile) {
      for (std::uint64_t block = 0; block < blocks_per_row; ++block) {
        for (const auto& segment : kSegments) {
          for (std::uint64_t lane = 0; lane < rows_per_tile; ++lane) {
            const std::uint64_t row = row_tile * rows_per_tile + lane;
            if (row >= rows_per_expert) {
              layout.bytes.insert(
                  layout.bytes.end(),
                  static_cast<std::size_t>(segment.bytes), 0U);
              continue;
            }
            const std::uint8_t* src =
                raw.data() +
                ((selected * rows_per_expert + row) * blocks_per_row + block) *
                    kQ6KBlockBytes +
                segment.offset;
            layout.bytes.insert(
                layout.bytes.end(), src,
                src + static_cast<std::ptrdiff_t>(segment.bytes));
          }
        }
      }
    }
  }
  Require(layout.bytes.size() ==
              static_cast<std::size_t>(
                  selected_count * layout.row_tile_count * blocks_per_row *
                  rows_per_tile * kQ6KBlockBytes),
          "Q6_K rowstripe byte size mismatch");
  return layout;
}

Q6KRowstripeLayout BuildSelectedRawQ6KRowstripeCoalesced(
    const std::vector<std::uint8_t>& raw,
    std::uint64_t rows_per_expert,
    std::uint64_t blocks_per_row,
    std::uint64_t selected_count,
    std::uint64_t rows_per_tile) {
  Require(rows_per_expert > 0,
          "coalesced Q6_K rowstripe rows_per_expert must be nonzero");
  Require(blocks_per_row > 0,
          "coalesced Q6_K rowstripe blocks_per_row must be nonzero");
  Require(selected_count > 0,
          "coalesced Q6_K rowstripe selected_count must be nonzero");
  Require(rows_per_tile > 0,
          "coalesced Q6_K rowstripe rows_per_tile must be nonzero");
  Require(raw.size() == selected_count * rows_per_expert * blocks_per_row *
                            kQ6KBlockBytes,
          "coalesced Q6_K rowstripe raw byte size does not match shape");
  struct Segment {
    std::uint64_t offset;
    std::uint64_t bytes;
  };
  constexpr Segment kSegments[] = {
      {0ULL, 64ULL},
      {128ULL, 32ULL},
      {64ULL, 64ULL},
      {160ULL, 32ULL},
      {192ULL, 16ULL},
      {208ULL, 2ULL},
  };
  Q6KRowstripeLayout layout;
  layout.rows_per_tile = rows_per_tile;
  layout.row_tile_count =
      (rows_per_expert + rows_per_tile - 1ULL) / rows_per_tile;
  layout.bytes.reserve(static_cast<std::size_t>(
      selected_count * layout.row_tile_count * blocks_per_row *
      rows_per_tile * kQ6KBlockBytes));
  for (std::uint64_t selected = 0; selected < selected_count; ++selected) {
    for (std::uint64_t row_tile = 0; row_tile < layout.row_tile_count;
         ++row_tile) {
      for (std::uint64_t block = 0; block < blocks_per_row; ++block) {
        for (const auto& segment : kSegments) {
          for (std::uint64_t byte = 0; byte < segment.bytes; ++byte) {
            for (std::uint64_t lane = 0; lane < rows_per_tile; ++lane) {
              const std::uint64_t row = row_tile * rows_per_tile + lane;
              if (row >= rows_per_expert) {
                layout.bytes.push_back(0U);
                continue;
              }
              const auto index =
                  ((selected * rows_per_expert + row) * blocks_per_row +
                   block) * kQ6KBlockBytes + segment.offset + byte;
              layout.bytes.push_back(raw[static_cast<std::size_t>(index)]);
            }
          }
        }
      }
    }
  }
  return layout;
}

void ValidateQ8InputPlanes(const std::vector<std::int8_t>& q8_qs,
                           const std::vector<std::int16_t>& q8_bsums,
                           const std::vector<float>& q8_d,
                           std::uint64_t blocks_per_row,
                           int repeat) {
  Require(repeat > 0, "repeat must be positive");
  Require(q8_qs.size() == blocks_per_row * kQ8QsPerBlock, "Q8_K qs size does not match blocks_per_row");
  Require(q8_bsums.size() == blocks_per_row * kQ8BsumsPerBlock,
          "Q8_K bsums size does not match blocks_per_row");
  Require(q8_d.size() == blocks_per_row, "Q8_K scale size does not match blocks_per_row");
}

void ValidateSelectedQ4X8Q8InputPlanes(const std::vector<std::int8_t>& q8_qs,
                                       const std::vector<std::int16_t>& q8_bsums,
                                       const std::vector<float>& q8_d,
                                       std::uint64_t blocks_per_row,
                                       std::uint64_t selected_count,
                                       int repeat) {
  Require(selected_count > 0, "selected Q4_K count must be nonzero");
  Require(repeat > 0, "repeat must be positive");
  Require(q8_qs.size() == selected_count * blocks_per_row * kQ8QsPerBlock,
          "Q8_K qs size does not match selected Q4_K blocks_per_row");
  Require(q8_bsums.size() ==
              selected_count * blocks_per_row * kQ8BsumsPerBlock,
          "Q8_K bsums size does not match selected Q4_K blocks_per_row");
  Require(q8_d.size() == selected_count * blocks_per_row,
          "Q8_K scale size does not match selected Q4_K blocks_per_row");
}

void ValidateQ6KQ8InputPlanes(const GpuQ8KInputPlanes& q8,
                              std::uint64_t blocks_per_row,
                              int repeat) {
  Require(repeat > 0, "repeat must be positive");
  Require(q8.qs.size() == blocks_per_row * kQ8QsPerBlock,
          "Q8_K qs size does not match Q6_K blocks_per_row");
  Require(q8.d.size() == blocks_per_row,
          "Q8_K scale size does not match Q6_K blocks_per_row");
}

void ValidateSelectedQ6KQ8InputPlanes(const GpuQ8KInputPlanes& q8,
                                      std::uint64_t blocks_per_row,
                                      std::uint64_t selected_count,
                                      int repeat) {
  Require(selected_count > 0, "selected Q6_K count must be nonzero");
  Require(repeat > 0, "repeat must be positive");
  Require(q8.qs.size() == selected_count * blocks_per_row * kQ8QsPerBlock,
          "Q8_K qs size does not match selected Q6_K blocks_per_row");
  Require(q8.d.size() == selected_count * blocks_per_row,
          "Q8_K scale size does not match selected Q6_K blocks_per_row");
}

void ValidateF32MatvecInputs(const std::vector<float>& weights,
                             const std::vector<float>& input,
                             std::uint64_t rows, std::uint64_t cols,
                             int repeat) {
  Require(rows > 0, "F32 matvec rows must be nonzero");
  Require(cols > 0, "F32 matvec cols must be nonzero");
  Require(repeat > 0, "repeat must be positive");
  Require(input.size() == static_cast<std::size_t>(cols), "F32 matvec input size does not match cols");
  Require(weights.size() == static_cast<std::size_t>(rows * cols), "F32 matvec weight size does not match rows * cols");
}

void ValidateF32MatvecWeights(const std::vector<float>& weights,
                              std::uint64_t rows,
                              std::uint64_t cols) {
  Require(rows > 0, "F32 matvec rows must be nonzero");
  Require(cols > 0, "F32 matvec cols must be nonzero");
  Require(weights.size() == static_cast<std::size_t>(rows * cols),
          "F32 matvec weight size does not match rows * cols");
}

void ValidateF32MatvecInput(const std::vector<float>& input,
                            std::uint64_t cols,
                            int repeat) {
  Require(cols > 0, "F32 matvec cols must be nonzero");
  Require(repeat > 0, "repeat must be positive");
  Require(input.size() == static_cast<std::size_t>(cols),
          "F32 matvec input size does not match cols");
}

void ValidateRmsNormInputs(const std::vector<float>& input,
                           const std::vector<float>& weight,
                           int repeat) {
  Require(!input.empty(), "RMSNorm input must be nonempty");
  Require(repeat > 0, "repeat must be positive");
  Require(input.size() == weight.size(), "RMSNorm input/weight size mismatch");
}

void ValidateResidualRmsNormInputs(const std::vector<float>& residual_input,
                                   const std::vector<float>& residual_delta,
                                   const std::vector<float>& norm_weight,
                                   int repeat) {
  Require(!residual_input.empty(), "residual RMSNorm input must be nonempty");
  Require(repeat > 0, "repeat must be positive");
  Require(residual_delta.size() == residual_input.size(),
          "residual RMSNorm delta size mismatch");
  Require(norm_weight.size() == residual_input.size(),
          "residual RMSNorm weight size mismatch");
}

void ValidateFullAttentionCoreGateInputs(
    const std::vector<float>& q_rope,
    const std::vector<float>& k_history_flat,
    const std::vector<float>& v_history_flat,
    const std::vector<float>& q_full,
    std::uint64_t token_count,
    std::uint64_t head_dim,
    std::uint64_t q_head_count,
    std::uint64_t kv_head_count,
    int repeat) {
  Require(token_count > 0, "full attention token_count must be nonzero");
  Require(head_dim > 0, "full attention head_dim must be nonzero");
  Require(q_head_count > 0, "full attention q_head_count must be nonzero");
  Require(kv_head_count > 0, "full attention kv_head_count must be nonzero");
  Require(q_head_count % kv_head_count == 0,
          "full attention q heads must be divisible by kv heads");
  Require(repeat > 0, "repeat must be positive");
  const std::uint64_t q_values = head_dim * q_head_count;
  const std::uint64_t kv_values = head_dim * kv_head_count;
  Require(q_rope.size() == static_cast<std::size_t>(q_values),
          "full attention q_rope size mismatch");
  Require(q_full.size() == static_cast<std::size_t>(q_values * 2),
          "full attention q_full size mismatch");
  Require(k_history_flat.size() ==
              static_cast<std::size_t>(token_count * kv_values),
          "full attention k history size mismatch");
  Require(v_history_flat.size() ==
              static_cast<std::size_t>(token_count * kv_values),
          "full attention v history size mismatch");
}

void ValidateConvWeights(const std::vector<float>& conv_weights,
                         std::uint64_t rows,
                         std::uint64_t conv_kernel_size) {
  Require(conv_kernel_size >= 2, "conv kernel size must be at least 2");
  Require(conv_weights.size() == rows * conv_kernel_size,
          "conv weight size does not match rows * kernel_size");
}

void ValidateConvState(const std::vector<float>& conv_state,
                       std::uint64_t rows,
                       std::uint64_t conv_kernel_size) {
  Require(conv_kernel_size >= 2, "conv kernel size must be at least 2");
  Require(conv_state.size() == rows * (conv_kernel_size - 1),
          "conv state size does not match rows * (kernel_size - 1)");
}

void ValidateConvInputs(const std::vector<float>& conv_weights,
                        const std::vector<float>& conv_state,
                        std::uint64_t rows,
                        std::uint64_t conv_kernel_size) {
  ValidateConvWeights(conv_weights, rows, conv_kernel_size);
  ValidateConvState(conv_state, rows, conv_kernel_size);
}

void ValidatePostConvPrepInputs(const std::vector<float>& conv_output_raw,
                                std::uint64_t head_dim,
                                std::uint64_t query_heads,
                                std::uint64_t value_heads,
                                int repeat) {
  Require(head_dim > 0, "postconv head_dim must be nonzero");
  Require(query_heads > 0, "postconv query_heads must be nonzero");
  Require(value_heads > 0, "postconv value_heads must be nonzero");
  Require(repeat > 0, "repeat must be positive");
  const std::uint64_t q_values = head_dim * query_heads;
  const std::uint64_t v_values = head_dim * value_heads;
  Require(conv_output_raw.size() == static_cast<std::size_t>(2 * q_values + v_values),
          "postconv raw conv output size mismatch");
}

void ValidateLinearAttentionDeltaInputSizes(
    const std::vector<float>& q,
    const std::vector<float>& k,
    const std::vector<float>& v,
    const std::vector<float>& gate,
    const std::vector<float>& beta,
    std::size_t recurrent_state_values,
    const std::vector<float>& z,
    const std::vector<float>& norm_weight,
    std::uint64_t head_dim,
    std::uint64_t query_heads,
    std::uint64_t value_heads,
    int repeat) {
  Require(head_dim > 0, "linear attention delta head_dim must be nonzero");
  Require(query_heads > 0, "linear attention delta query_heads must be nonzero");
  Require(value_heads > 0, "linear attention delta value_heads must be nonzero");
  Require(value_heads % query_heads == 0,
          "linear attention delta value_heads must broadcast query_heads");
  Require(repeat > 0, "repeat must be positive");
  const std::uint64_t q_values = head_dim * query_heads;
  const std::uint64_t v_values = head_dim * value_heads;
  Require(q.size() == static_cast<std::size_t>(q_values),
          "linear attention delta q size mismatch");
  Require(k.size() == static_cast<std::size_t>(q_values),
          "linear attention delta k size mismatch");
  Require(v.size() == static_cast<std::size_t>(v_values),
          "linear attention delta v size mismatch");
  Require(gate.size() == static_cast<std::size_t>(value_heads),
          "linear attention delta gate size mismatch");
  Require(beta.size() == static_cast<std::size_t>(value_heads),
          "linear attention delta beta size mismatch");
  Require(recurrent_state_values ==
              static_cast<std::size_t>(head_dim * head_dim * value_heads),
          "linear attention delta recurrent state size mismatch");
  Require(z.size() == static_cast<std::size_t>(v_values),
          "linear attention delta z size mismatch");
  Require(norm_weight.size() == static_cast<std::size_t>(head_dim),
          "linear attention delta norm weight size mismatch");
}

void ValidateLinearAttentionDeltaInputs(const std::vector<float>& q,
                                        const std::vector<float>& k,
                                        const std::vector<float>& v,
                                        const std::vector<float>& gate,
                                        const std::vector<float>& beta,
                                        const std::vector<float>& recurrent_state,
                                        const std::vector<float>& z,
                                        const std::vector<float>& norm_weight,
                                        std::uint64_t head_dim,
                                        std::uint64_t query_heads,
                                        std::uint64_t value_heads,
                                        int repeat) {
  ValidateLinearAttentionDeltaInputSizes(
      q, k, v, gate, beta, recurrent_state.size(), z, norm_weight, head_dim,
      query_heads, value_heads, repeat);
}

int NearestInt(float value) {
  float shifted = value + 12582912.0f;
  int bits = 0;
  std::memcpy(&bits, &shifted, sizeof(bits));
  return (bits & 0x007fffff) - 0x00400000;
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
  for (int i = 0; i < 128; ++i) {
    const int src_id = i % 8;
    const int src_offset = (i / 8) * 8;
    std::memcpy(dst_qs + i * 8, blocks[static_cast<std::size_t>(src_id)] + 16 + src_offset, 8);
  }

  std::uint8_t s[8] = {};
  std::uint8_t m[8] = {};
  for (int phase = 0; phase < 2; ++phase) {
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 8; ++j) {
        const auto* scales = blocks[static_cast<std::size_t>(j)] + 4;
        if (phase == 0) {
          s[j] = scales[i] & 63;
          m[j] = scales[i + 4] & 63;
        } else {
          s[j] = ((scales[i] & 192) >> 2) | (scales[i + 8] & 15);
          m[j] = ((scales[i + 4] & 192) >> 2) | ((scales[i + 8] & 240) >> 4);
        }
      }
      const int offset = phase == 0 ? i * 12 : i * 12 + 48;
      dst_scales[offset + 0] = (s[0] & 63) + ((s[4] & 48) << 2);
      dst_scales[offset + 1] = (s[1] & 63) + ((s[5] & 48) << 2);
      dst_scales[offset + 2] = (s[2] & 63) + ((s[6] & 48) << 2);
      dst_scales[offset + 3] = (s[3] & 63) + ((s[7] & 48) << 2);
      dst_scales[offset + 4] = (m[0] & 63) + ((m[4] & 48) << 2);
      dst_scales[offset + 5] = (m[1] & 63) + ((m[5] & 48) << 2);
      dst_scales[offset + 6] = (m[2] & 63) + ((m[6] & 48) << 2);
      dst_scales[offset + 7] = (m[3] & 63) + ((m[7] & 48) << 2);
      dst_scales[offset + 8] = (s[4] & 15) + ((m[4] & 15) << 4);
      dst_scales[offset + 9] = (s[5] & 15) + ((m[5] & 15) << 4);
      dst_scales[offset + 10] = (s[6] & 15) + ((m[6] & 15) << 4);
      dst_scales[offset + 11] = (s[7] & 15) + ((m[7] & 15) << 4);
    }
  }
}

}  // namespace

const char* KernelVariantName(GpuQ4X8KernelVariant variant) {
  switch (variant) {
    case GpuQ4X8KernelVariant::kGroup8Serial:
      return "group8_serial";
    case GpuQ4X8KernelVariant::kRowlaneParallel:
      return "rowlane_parallel";
  }
  return "unknown";
}

const char* KernelFunctionName(GpuQ4X8KernelVariant variant) {
  switch (variant) {
    case GpuQ4X8KernelVariant::kGroup8Serial:
      return "q4k_x8_matvec_group8";
    case GpuQ4X8KernelVariant::kRowlaneParallel:
      return "q4k_x8_matvec_rowlane";
  }
  return "unknown";
}

std::uint64_t RowsPerWorkItem(GpuQ4X8KernelVariant variant) {
  return variant == GpuQ4X8KernelVariant::kGroup8Serial ? 8 : 1;
}

GpuQ8KInputPlanes QuantizeQ8KInputPlanes(const std::vector<float>& input) {
  Require(input.size() % 256 == 0, "Q8_K input quantization requires 256-aligned input");
  GpuQ8KInputPlanes planes;
  const std::size_t block_count = input.size() / 256;
  planes.qs.resize(block_count * 256);
  planes.bsums.resize(block_count * 16);
  planes.d.resize(block_count);
  for (std::size_t block_index = 0; block_index < block_count; ++block_index) {
    const auto* block_input = input.data() + block_index * 256;
    float max = 0.0f;
    float amax = 0.0f;
    for (int i = 0; i < 256; ++i) {
      const float abs_value = std::abs(block_input[i]);
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
      planes.qs[block_index * 256 + static_cast<std::size_t>(i)] =
          static_cast<std::int8_t>(quantized);
    }
    for (int group = 0; group < 16; ++group) {
      int sum = 0;
      for (int i = 0; i < 16; ++i) {
        sum += planes.qs[block_index * 256 + static_cast<std::size_t>(group * 16 + i)];
      }
      planes.bsums[block_index * 16 + static_cast<std::size_t>(group)] =
          static_cast<std::int16_t>(sum);
    }
    planes.d[block_index] = 1.0f / iscale;
  }
  return planes;
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

PackedQ6KRowstripe PackQ6KRowstripe(
    const std::vector<std::uint8_t>& raw,
    std::uint64_t rows_per_expert,
    std::uint64_t blocks_per_row,
    std::uint64_t expert_count,
    std::uint64_t rows_per_tile) {
  auto internal = BuildSelectedRawQ6KRowstripe(
      raw, rows_per_expert, blocks_per_row, expert_count, rows_per_tile);
  PackedQ6KRowstripe result;
  result.bytes = std::move(internal.bytes);
  result.rows_per_tile = internal.rows_per_tile;
  result.row_tile_count = internal.row_tile_count;
  return result;
}

PackedQ6KRowstripe PackQ6KRowstripeCoalesced(
    const std::vector<std::uint8_t>& raw,
    std::uint64_t rows_per_expert,
    std::uint64_t blocks_per_row,
    std::uint64_t expert_count,
    std::uint64_t rows_per_tile) {
  auto internal = BuildSelectedRawQ6KRowstripeCoalesced(
      raw, rows_per_expert, blocks_per_row, expert_count, rows_per_tile);
  PackedQ6KRowstripe result;
  result.bytes = std::move(internal.bytes);
  result.rows_per_tile = internal.rows_per_tile;
  result.row_tile_count = internal.row_tile_count;
  return result;
}

class GpuQ4X8MatvecRunner::Impl {
 public:
  struct ResidentPackedQ4X8 {
    cl_mem buffer = nullptr;
    std::uint64_t rows = 0;
    std::uint64_t blocks_per_row = 0;
    std::size_t packed_bytes = 0;
  };

  struct ResidentRawQ6K {
    cl_mem buffer = nullptr;
    std::uint64_t rows = 0;
    std::uint64_t blocks_per_row = 0;
    std::size_t raw_bytes = 0;
    bool rowstripe = false;
    std::uint64_t rows_per_tile = 0;
    std::uint64_t row_tile_count = 0;
  };

  struct ResidentRawQ4KCpuOrder {
    cl_mem buffer = nullptr;
    std::uint64_t rows = 0;
    std::uint64_t blocks_per_row = 0;
    std::size_t raw_bytes = 0;
  };

  struct ResidentConvWeights {
    cl_mem buffer = nullptr;
    std::uint64_t rows = 0;
    std::uint64_t conv_kernel_size = 0;
    std::size_t weight_bytes = 0;
  };

  struct ResidentF32Matvec {
    cl_mem buffer = nullptr;
    std::uint64_t rows = 0;
    std::uint64_t cols = 0;
    std::size_t weight_bytes = 0;
  };

  struct ResidentF32Buffer {
    cl_mem buffer = nullptr;
    std::size_t values = 0;
    std::size_t bytes = 0;
    bool owned = true;
  };

  struct ScratchBuffer {
    cl_mem mem = nullptr;
    std::size_t bytes = 0;
    cl_mem_flags flags = 0;
  };

  Impl(std::string device_substring, const std::string& opencl_source) {
    selected_ = SelectDevice(api_, device_substring);
    cl_int err = kClSuccess;
    context_ = api_.clCreateContext(nullptr, 1, &selected_.device, nullptr, nullptr, &err);
    Check(err, "clCreateContext");
    queue_ =
        api_.clCreateCommandQueue(context_, selected_.device, QueueProperties(), &err);
    Check(err, "clCreateCommandQueue");
    const char* source = opencl_source.c_str();
    const std::size_t source_len = opencl_source.size();
    program_ = api_.clCreateProgramWithSource(context_, 1, &source, &source_len, &err);
    Check(err, "clCreateProgramWithSource");
    const auto build_begin = std::chrono::steady_clock::now();
    err = api_.clBuildProgram(program_, 1, &selected_.device, "", nullptr, nullptr);
    const auto build_end = std::chrono::steady_clock::now();
    program_build_ms_ = std::chrono::duration<double, std::milli>(build_end - build_begin).count();
    CaptureBuildLog();
    Check(err, "clBuildProgram");
    kernel_group8_ = CreateKernel(GpuQ4X8KernelVariant::kGroup8Serial);
    kernel_rowlane_ = CreateKernel(GpuQ4X8KernelVariant::kRowlaneParallel);
    kernel_rowblock16_ =
        CreateNamedKernel("q4k_x8_matvec_rowblock16_reduce");
    kernel_rowblock16_cpuorder_finalize_ =
        CreateNamedKernel("q4k_x8_matvec_rowblock16_cpuorder_finalize");
    kernel_rowlane_localq8_ =
        CreateNamedKernel("q4k_x8_matvec_rowlane_localq8");
    kernel_rowlane_expert8_ =
        CreateNamedKernel("q4k_x8_matvec_rowlane_expert8");
    kernel_rowlane_expert8_localq8_ =
        CreateNamedKernel("q4k_x8_matvec_rowlane_expert8_localq8");
    kernel_rowlane_expert8_plus_shared_localq8_ =
        CreateNamedKernel("q4k_x8_matvec_rowlane_expert8_plus_shared_localq8");
    kernel_rowlane_topk_indexed_expert8_plus_shared_localq8_ =
        CreateNamedKernel(
            "q4k_x8_matvec_topk_indexed_expert8_plus_shared_localq8");
    kernel_rowlane_expert8_multiq8_ =
        CreateNamedKernel("q4k_x8_matvec_rowlane_expert8_multiq8");
    kernel_rowlane_expert8_plus_shared_multiq8_ =
        CreateNamedKernel("q4k_x8_selected_down_expert8_plus_shared_q4");
    kernel_rowlane_expert8_f32input_ =
        CreateNamedKernel("q4k_x8_matvec_rowlane_expert8_f32input");
    kernel_q4_cpu_order_ = CreateNamedKernel("q4k_cpu_order_matvec");
    kernel_q6_matvec_row_ = CreateNamedKernel("q6k_selected_down_matvec_row");
    kernel_q6_linear_qkv_cpuorder_ =
        CreateNamedKernel("q6k_linear_qkv_cpuorder_nofma");
    kernel_q6_matvec_rowstripe_ =
        CreateNamedKernel("q6k_selected_down_matvec_rowstripe");
    kernel_q6_matvec_rowstripe_localq8_ =
        CreateNamedKernel("q6k_selected_down_matvec_rowstripe_localq8");
    kernel_q6_matvec_row_expert8_ =
        CreateNamedKernel("q6k_selected_down_matvec_row_expert8");
    kernel_q6_matvec_rowstripe_expert8_ =
        CreateNamedKernel("q6k_selected_down_matvec_rowstripe_expert8");
    kernel_q6_matvec_rowstripe_expert8_plus_shared_ =
        CreateNamedKernel(
            "q6k_selected_down_matvec_rowstripe_expert8_plus_shared_raw");
    kernel_q6_matvec_rowstripe_expert8_plus_shared_tail_atomic_ =
        CreateNamedKernel(
            "q6k_selected_down_matvec_rowstripe_expert8_plus_shared_tail_atomic_raw");
    kernel_q6_matvec_rowstripe_expert8_plus_shared_tail_contrib_ =
        CreateNamedKernel(
            "q6k_selected_down_matvec_rowstripe_expert8_plus_shared_tail_contrib_raw");
    kernel_ffn_tail_reduce9_contrib_ =
        CreateNamedKernel("ffn_tail_reduce9_contrib_f32");
    kernel_f32_topk8_blocks_ = CreateNamedKernel("f32_topk8_blocks");
    kernel_f32_topk16_blocks_ = CreateNamedKernel("f32_topk16_blocks");
    kernel_qkv_delta_sparse_overlay_ =
        CreateNamedKernel("qkv_delta_sparse_overlay_f32");
    kernel_qkv_delta_blockq16_overlay_ =
        CreateNamedKernel("qkv_delta_blockq16_overlay_f32");
    kernel_f32_matvec_ = CreateNamedKernel("f32_matvec_row_f32");
    kernel_conv_ = CreateNamedKernel("linear_attn_conv_f32");
    kernel_conv_cpuorder_ =
        CreateNamedKernel("linear_attn_conv_cpuorder_nofma_f32");
    kernel_postconv_silu_split_ = CreateNamedKernel("linear_attn_postconv_silu_split_f32");
    kernel_postconv_silu_split_cpuorder_ =
        CreateNamedKernel("linear_attn_postconv_silu_split_cpuorder_f32");
    kernel_postconv_l2_q_ = CreateNamedKernel("linear_attn_l2_norm_heads_f32");
    kernel_postconv_l2_k_ = CreateNamedKernel("linear_attn_l2_norm_heads_f32");
    kernel_postconv_l2_qk_ =
        CreateNamedKernel("linear_attn_l2_norm_qk_heads_f32");
    kernel_postconv_l2_qk_cpuorder_ =
        CreateNamedKernel("linear_attn_postconv_qk_l2_cpuorder_f32");
    kernel_postconv_fused_qk_l2_ =
        CreateNamedKernel("linear_attn_postconv_fused_qk_l2_f32");
    kernel_delta_recurrent_ = CreateNamedKernel("linear_attn_delta_recurrent_f32");
    kernel_delta_recurrent_qk_local_ =
        CreateNamedKernel("linear_attn_delta_recurrent_qk_local_f32");
    kernel_delta_recurrent_final_qk_local_ =
        CreateNamedKernel("linear_attn_delta_recurrent_final_qk_local_f32");
    kernel_delta_recurrent_final_cpu_shape_qk_local_ =
        CreateNamedKernel(
            "linear_attn_delta_recurrent_final_cpu_shape_qk_local_f32");
    kernel_delta_recurrent_final_cpuorder_ = CreateNamedKernel(
        "linear_attn_delta_recurrent_final_cpuorder_nofma_f32");
    kernel_delta_final_ = CreateNamedKernel("linear_attn_final_norm_f32");
    kernel_delta_final_cpu_shape_ =
        CreateNamedKernel("linear_attn_final_norm_cpu_shape_f32");
    kernel_ffn_swiglu_ = CreateNamedKernel("ffn_moe_swiglu_f32");
    kernel_ffn_swiglu_reorder_ =
        CreateNamedKernel("ffn_moe_swiglu_reorder_f32");
    kernel_q8_quantize_ = CreateNamedKernel("q8k_quantize_f32_blocks");
    kernel_q8_quantize_bsums_ =
        CreateNamedKernel("q8k_quantize_f32_blocks_with_bsums");
    kernel_ffn_weighted_ = CreateNamedKernel("ffn_moe_weighted_aggregate_f32");
    kernel_ffn_gate_apply_ = CreateNamedKernel("shared_expert_gate_apply_f32");
    kernel_ffn_output_add_ = CreateNamedKernel("ffn_output_add_f32");
    kernel_post_ffn_residual_ = CreateNamedKernel("post_ffn_residual_add_f32");
    kernel_ffn_tail_init_residual_bits_ =
        CreateNamedKernel("ffn_tail_init_residual_bits_f32");
    kernel_ffn_tail_reduce_down_atomic_ =
        CreateNamedKernel("ffn_tail_reduce_down_atomic_f32");
    kernel_ffn_tail_fused_output_ =
        CreateNamedKernel("ffn_tail_fused_output_f32");
    kernel_rmsnorm_hidden_ = CreateNamedKernel("rms_norm_hidden_f32");
    kernel_rmsnorm_hidden_scale_ =
        CreateNamedKernel("rms_norm_hidden_scale_f32");
    kernel_rmsnorm_hidden_apply_scale_ =
        CreateNamedKernel("rms_norm_hidden_apply_scale_f32");
    kernel_full_attn_core_ = CreateNamedKernel("full_attn_core_f32");
    kernel_full_attn_score_ = CreateNamedKernel("full_attn_score_f32");
    kernel_full_attn_apply_score_gate_ =
        CreateNamedKernel("full_attn_apply_score_gate_f32");
    kernel_full_attn_gate_ = CreateNamedKernel("full_attn_gate_f32");
    kernel_full_attn_qk_norm_rope_ =
        CreateNamedKernel("full_attn_qk_norm_rope_f32");
  }

  ~Impl() {
    if (!pending_host_uploads_.empty() && queue_ != nullptr) {
      (void)api_.clFinish(queue_);
      pending_host_uploads_.clear();
    }
    ReleaseScratchBuffer(expert8_swiglu_scratch_q8_qs_);
    ReleaseScratchBuffer(expert8_swiglu_scratch_q8_bsums_);
    ReleaseScratchBuffer(expert8_swiglu_scratch_q8_d_);
    ReleaseScratchBuffer(expert8_swiglu_scratch_gate_up_);
    ReleaseScratchBuffer(expert8_swiglu_scratch_swiglu_);
    ReleaseScratchBuffer(expert8_q4_down_scratch_q8_qs_);
    ReleaseScratchBuffer(expert8_q4_down_scratch_q8_bsums_);
    ReleaseScratchBuffer(expert8_q4_down_scratch_q8_d_);
    ReleaseScratchBuffer(expert8_q4_down_scratch_out_);
    ReleaseScratchBuffer(selected_shared_q4_down_scratch_selected_q8_qs_);
    ReleaseScratchBuffer(selected_shared_q4_down_scratch_selected_q8_bsums_);
    ReleaseScratchBuffer(selected_shared_q4_down_scratch_selected_q8_d_);
    ReleaseScratchBuffer(selected_shared_q4_down_scratch_shared_q8_qs_);
    ReleaseScratchBuffer(selected_shared_q4_down_scratch_shared_q8_bsums_);
    ReleaseScratchBuffer(selected_shared_q4_down_scratch_shared_q8_d_);
    ReleaseScratchBuffer(selected_shared_q4_down_scratch_selected_out_);
    ReleaseScratchBuffer(selected_shared_q4_down_scratch_shared_out_);
    ReleaseScratchBuffer(swiglu_handoff_scratch_q8_qs_);
    ReleaseScratchBuffer(swiglu_handoff_scratch_q8_bsums_);
    ReleaseScratchBuffer(swiglu_handoff_scratch_q8_d_);
    ReleaseScratchBuffer(swiglu_handoff_scratch_gate_up_);
    ReleaseScratchBuffer(swiglu_handoff_scratch_source_map_);
    ReleaseScratchBuffer(swiglu_handoff_scratch_swiglu_);
    ReleaseScratchBuffer(expert8_q6_down_scratch_q8_qs_);
    ReleaseScratchBuffer(expert8_q6_down_scratch_q8_d_);
    ReleaseScratchBuffer(expert8_q6_down_scratch_out_);
    ReleaseScratchBuffer(resident_q6_handoff_scratch_q8_qs_);
    ReleaseScratchBuffer(resident_q6_handoff_scratch_q8_d_);
    ReleaseScratchBuffer(resident_q6_handoff_scratch_out_);
    ReleaseScratchBuffer(q6_conv_state_scratch_q8_qs_);
    ReleaseScratchBuffer(q6_conv_state_scratch_q8_d_);
    ReleaseScratchBuffer(q6_conv_state_scratch_qkv_);
    ReleaseScratchBuffer(q6_conv_state_scratch_conv_output_);
    ReleaseScratchBuffer(q6_conv_state_scratch_next_state_);
    ReleaseScratchBuffer(packed_conv_state_scratch_q8_qs_);
    ReleaseScratchBuffer(packed_conv_state_scratch_q8_bsums_);
    ReleaseScratchBuffer(packed_conv_state_scratch_q8_d_);
    ReleaseScratchBuffer(packed_conv_state_scratch_qkv_);
    ReleaseScratchBuffer(packed_conv_state_scratch_conv_output_);
    ReleaseScratchBuffer(packed_conv_state_scratch_next_state_);
    ReleaseScratchBuffer(postconv_prep_scratch_raw_);
    ReleaseScratchBuffer(postconv_prep_scratch_silu_);
    ReleaseScratchBuffer(postconv_prep_scratch_q_);
    ReleaseScratchBuffer(postconv_prep_scratch_k_);
    ReleaseScratchBuffer(postconv_prep_scratch_v_);
    ReleaseScratchBuffer(postconv_prep_scratch_q_norm_);
    ReleaseScratchBuffer(postconv_prep_scratch_k_norm_);
    ReleaseScratchBuffer(linear_delta_scratch_q_);
    ReleaseScratchBuffer(linear_delta_scratch_k_);
    ReleaseScratchBuffer(linear_delta_scratch_v_);
    ReleaseScratchBuffer(linear_delta_scratch_gate_);
    ReleaseScratchBuffer(linear_delta_scratch_beta_);
    ReleaseScratchBuffer(linear_delta_scratch_z_);
    ReleaseScratchBuffer(linear_delta_scratch_norm_);
    ReleaseScratchBuffer(linear_delta_scratch_attention_);
    ReleaseScratchBuffer(linear_delta_scratch_final_);
    ReleaseScratchBuffer(f32_input_q4_scratch_q8_qs_);
    ReleaseScratchBuffer(f32_input_q4_scratch_q8_bsums_);
    ReleaseScratchBuffer(f32_input_q4_scratch_q8_d_);
    ReleaseScratchBuffer(f32_input_q4_scratch_output_);
    ReleaseScratchBuffer(q4_cpu_order_scratch_q8_qs_);
    ReleaseScratchBuffer(q4_cpu_order_scratch_q8_bsums_);
    ReleaseScratchBuffer(q4_cpu_order_scratch_q8_d_);
    ReleaseScratchBuffer(q4_cpu_order_scratch_out_);
    ReleaseScratchBuffer(f32_input_q6_scratch_q8_qs_);
    ReleaseScratchBuffer(f32_input_q6_scratch_q8_d_);
    ReleaseScratchBuffer(f32_input_q6_scratch_output_);
    ReleaseScratchBuffer(linear_preconv_shared_q8_qs_);
    ReleaseScratchBuffer(linear_preconv_shared_q8_bsums_);
    ReleaseScratchBuffer(linear_preconv_shared_q8_d_);
    ReleaseScratchBuffer(linear_preconv_shared_qkv_);
    ReleaseScratchBuffer(linear_preconv_shared_conv_output_);
    ReleaseScratchBuffer(linear_preconv_shared_next_state_);
    ReleaseScratchBuffer(linear_preconv_shared_alpha_beta_z_);
    ReleaseScratchBuffer(qkv_delta_sparse_overlay_scratch_indices_);
    ReleaseScratchBuffer(qkv_delta_blockq16_overlay_scratch_indices_);
    ReleaseScratchBuffer(qkv_delta_blockq16_overlay_scratch_q_delta_);
    ReleaseScratchBuffer(qkv_delta_blockq16_overlay_scratch_scales_);
    ReleaseScratchBuffer(resident_f32_matvec_scratch_output_);
    ReleaseScratchBuffer(ffn_tail_scratch_down_);
    ReleaseScratchBuffer(ffn_tail_scratch_weights_);
    ReleaseScratchBuffer(ffn_tail_scratch_weighted_);
    ReleaseScratchBuffer(ffn_tail_scratch_moe_out_);
    ReleaseScratchBuffer(ffn_tail_scratch_gate_weights_);
    ReleaseScratchBuffer(ffn_tail_scratch_attn_post_norm_);
    ReleaseScratchBuffer(ffn_tail_scratch_ffn_shexp_);
    ReleaseScratchBuffer(ffn_tail_scratch_shared_gate_);
    ReleaseScratchBuffer(ffn_tail_scratch_shared_sigmoid_);
    ReleaseScratchBuffer(ffn_tail_scratch_shared_gated_);
    ReleaseScratchBuffer(ffn_tail_scratch_ffn_out_);
    ReleaseScratchBuffer(ffn_tail_scratch_attn_residual_);
    ReleaseScratchBuffer(ffn_tail_scratch_layer_output_);
    ReleaseScratchBuffer(ffn_tail_scratch_contrib_);
    ReleaseScratchBuffer(attention_front_scratch_projection_);
    ReleaseScratchBuffer(attention_front_scratch_residual_input_);
    ReleaseScratchBuffer(attention_front_scratch_residual_);
    ReleaseScratchBuffer(attention_front_scratch_normalized_);
    ReleaseScratchBuffer(attention_front_scratch_q8_qs_);
    ReleaseScratchBuffer(attention_front_scratch_q8_bsums_);
    ReleaseScratchBuffer(attention_front_scratch_q8_d_);
    ReleaseScratchBuffer(attention_front_scratch_norm_weight_);
    ReleaseScratchBuffer(rmsnorm_hidden_scratch_input_);
    ReleaseScratchBuffer(rmsnorm_hidden_scratch_weight_);
    ReleaseScratchBuffer(rmsnorm_hidden_scratch_output_);
    ReleaseScratchBuffer(rmsnorm_hidden_scratch_scale_);
    ReleaseScratchBuffer(full_core_handoff_scratch_q_rope_);
    ReleaseScratchBuffer(full_core_handoff_scratch_k_history_);
    ReleaseScratchBuffer(full_core_handoff_scratch_v_history_);
    ReleaseScratchBuffer(full_core_handoff_scratch_q_full_);
    ReleaseScratchBuffer(full_core_qk_norm_rope_scratch_q_rope_);
    ReleaseScratchBuffer(full_core_qk_norm_rope_scratch_k_rope_);
    ReleaseScratchBuffer(full_core_qk_norm_rope_scratch_rope_cache_);
    ReleaseScratchBuffer(full_core_handoff_scratch_scores_);
    ReleaseScratchBuffer(full_core_handoff_scratch_pregate_);
    ReleaseScratchBuffer(full_core_handoff_scratch_gated_);
    ReleaseScratchBuffer(full_core_handoff_scratch_q8_qs_);
    ReleaseScratchBuffer(full_core_handoff_scratch_q8_bsums_);
    ReleaseScratchBuffer(full_core_handoff_scratch_q8_d_);
    ReleaseScratchBuffer(full_core_handoff_scratch_projection_);
    ReleaseScratchBuffer(full_core_handoff_scratch_residual_input_);
    ReleaseScratchBuffer(full_core_handoff_scratch_norm_weight_);
    ReleaseScratchBuffer(full_core_handoff_scratch_residual_);
    ReleaseScratchBuffer(full_core_handoff_scratch_normalized_);
    ClearResidentF32Buffers();
    ClearResidentF32Matvec();
    ClearResidentConvWeights();
    ClearResidentRawQ6K();
    ClearResidentRawQ4KCpuOrder();
    ClearResidentPackedQ4X8();
    if (kernel_full_attn_gate_) api_.clReleaseKernel(kernel_full_attn_gate_);
    if (kernel_full_attn_apply_score_gate_) {
      api_.clReleaseKernel(kernel_full_attn_apply_score_gate_);
    }
    if (kernel_full_attn_score_) api_.clReleaseKernel(kernel_full_attn_score_);
    if (kernel_full_attn_core_) api_.clReleaseKernel(kernel_full_attn_core_);
    if (kernel_full_attn_qk_norm_rope_) {
      api_.clReleaseKernel(kernel_full_attn_qk_norm_rope_);
    }
    if (kernel_rmsnorm_hidden_apply_scale_) {
      api_.clReleaseKernel(kernel_rmsnorm_hidden_apply_scale_);
    }
    if (kernel_rmsnorm_hidden_scale_) {
      api_.clReleaseKernel(kernel_rmsnorm_hidden_scale_);
    }
    if (kernel_rmsnorm_hidden_) api_.clReleaseKernel(kernel_rmsnorm_hidden_);
    if (kernel_ffn_tail_fused_output_) {
      api_.clReleaseKernel(kernel_ffn_tail_fused_output_);
    }
    if (kernel_ffn_tail_reduce_down_atomic_) {
      api_.clReleaseKernel(kernel_ffn_tail_reduce_down_atomic_);
    }
    if (kernel_ffn_tail_init_residual_bits_) {
      api_.clReleaseKernel(kernel_ffn_tail_init_residual_bits_);
    }
    if (kernel_post_ffn_residual_) api_.clReleaseKernel(kernel_post_ffn_residual_);
    if (kernel_ffn_output_add_) api_.clReleaseKernel(kernel_ffn_output_add_);
    if (kernel_ffn_gate_apply_) api_.clReleaseKernel(kernel_ffn_gate_apply_);
    if (kernel_ffn_weighted_) api_.clReleaseKernel(kernel_ffn_weighted_);
    if (kernel_q8_quantize_bsums_) api_.clReleaseKernel(kernel_q8_quantize_bsums_);
    if (kernel_q8_quantize_) api_.clReleaseKernel(kernel_q8_quantize_);
    if (kernel_ffn_swiglu_reorder_) api_.clReleaseKernel(kernel_ffn_swiglu_reorder_);
    if (kernel_ffn_swiglu_) api_.clReleaseKernel(kernel_ffn_swiglu_);
    if (kernel_delta_final_cpu_shape_) api_.clReleaseKernel(kernel_delta_final_cpu_shape_);
    if (kernel_delta_final_) api_.clReleaseKernel(kernel_delta_final_);
    if (kernel_delta_recurrent_final_cpu_shape_qk_local_) {
      api_.clReleaseKernel(kernel_delta_recurrent_final_cpu_shape_qk_local_);
    }
    if (kernel_delta_recurrent_final_cpuorder_) {
      api_.clReleaseKernel(kernel_delta_recurrent_final_cpuorder_);
    }
    if (kernel_delta_recurrent_final_qk_local_) api_.clReleaseKernel(kernel_delta_recurrent_final_qk_local_);
    if (kernel_delta_recurrent_qk_local_) api_.clReleaseKernel(kernel_delta_recurrent_qk_local_);
    if (kernel_delta_recurrent_) api_.clReleaseKernel(kernel_delta_recurrent_);
    if (kernel_postconv_fused_qk_l2_) {
      api_.clReleaseKernel(kernel_postconv_fused_qk_l2_);
    }
    if (kernel_postconv_l2_qk_) api_.clReleaseKernel(kernel_postconv_l2_qk_);
    if (kernel_postconv_l2_qk_cpuorder_) {
      api_.clReleaseKernel(kernel_postconv_l2_qk_cpuorder_);
    }
    if (kernel_postconv_l2_k_) api_.clReleaseKernel(kernel_postconv_l2_k_);
    if (kernel_postconv_l2_q_) api_.clReleaseKernel(kernel_postconv_l2_q_);
    if (kernel_postconv_silu_split_) api_.clReleaseKernel(kernel_postconv_silu_split_);
    if (kernel_postconv_silu_split_cpuorder_) {
      api_.clReleaseKernel(kernel_postconv_silu_split_cpuorder_);
    }
    if (kernel_conv_) api_.clReleaseKernel(kernel_conv_);
    if (kernel_conv_cpuorder_) api_.clReleaseKernel(kernel_conv_cpuorder_);
    if (kernel_f32_matvec_) api_.clReleaseKernel(kernel_f32_matvec_);
    if (kernel_qkv_delta_sparse_overlay_) {
      api_.clReleaseKernel(kernel_qkv_delta_sparse_overlay_);
    }
    if (kernel_qkv_delta_blockq16_overlay_) {
      api_.clReleaseKernel(kernel_qkv_delta_blockq16_overlay_);
    }
    if (kernel_f32_topk16_blocks_) api_.clReleaseKernel(kernel_f32_topk16_blocks_);
    if (kernel_f32_topk8_blocks_) api_.clReleaseKernel(kernel_f32_topk8_blocks_);
    if (kernel_q6_matvec_rowstripe_expert8_plus_shared_) {
      api_.clReleaseKernel(kernel_q6_matvec_rowstripe_expert8_plus_shared_);
    }
    if (kernel_q6_matvec_rowstripe_expert8_plus_shared_tail_atomic_) {
      api_.clReleaseKernel(
          kernel_q6_matvec_rowstripe_expert8_plus_shared_tail_atomic_);
    }
    if (kernel_q6_matvec_rowstripe_expert8_plus_shared_tail_contrib_) {
      api_.clReleaseKernel(
          kernel_q6_matvec_rowstripe_expert8_plus_shared_tail_contrib_);
    }
    if (kernel_ffn_tail_reduce9_contrib_) {
      api_.clReleaseKernel(kernel_ffn_tail_reduce9_contrib_);
    }
    if (kernel_q6_matvec_rowstripe_expert8_) api_.clReleaseKernel(kernel_q6_matvec_rowstripe_expert8_);
    if (kernel_q6_matvec_row_expert8_) api_.clReleaseKernel(kernel_q6_matvec_row_expert8_);
    if (kernel_q6_matvec_rowstripe_localq8_) api_.clReleaseKernel(kernel_q6_matvec_rowstripe_localq8_);
    if (kernel_q6_matvec_rowstripe_) api_.clReleaseKernel(kernel_q6_matvec_rowstripe_);
    if (kernel_q6_matvec_row_) api_.clReleaseKernel(kernel_q6_matvec_row_);
    if (kernel_q6_linear_qkv_cpuorder_) {
      api_.clReleaseKernel(kernel_q6_linear_qkv_cpuorder_);
    }
    if (kernel_q4_cpu_order_) api_.clReleaseKernel(kernel_q4_cpu_order_);
    if (kernel_rowlane_expert8_multiq8_) {
      api_.clReleaseKernel(kernel_rowlane_expert8_multiq8_);
    }
    if (kernel_rowlane_expert8_plus_shared_multiq8_) {
      api_.clReleaseKernel(kernel_rowlane_expert8_plus_shared_multiq8_);
    }
    if (kernel_rowlane_expert8_f32input_) {
      api_.clReleaseKernel(kernel_rowlane_expert8_f32input_);
    }
    if (kernel_rowlane_expert8_plus_shared_localq8_) {
      api_.clReleaseKernel(kernel_rowlane_expert8_plus_shared_localq8_);
    }
    if (kernel_rowlane_topk_indexed_expert8_plus_shared_localq8_) {
      api_.clReleaseKernel(
          kernel_rowlane_topk_indexed_expert8_plus_shared_localq8_);
    }
    if (kernel_rowlane_expert8_) api_.clReleaseKernel(kernel_rowlane_expert8_);
    if (kernel_rowlane_) api_.clReleaseKernel(kernel_rowlane_);
    if (kernel_rowblock16_cpuorder_finalize_) {
      api_.clReleaseKernel(kernel_rowblock16_cpuorder_finalize_);
    }
    if (kernel_rowblock16_) api_.clReleaseKernel(kernel_rowblock16_);
    if (kernel_group8_) api_.clReleaseKernel(kernel_group8_);
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

  std::uint64_t UploadPackedQ4X8(const std::vector<std::uint8_t>& packed,
                                 std::uint64_t rows,
                                 std::uint64_t blocks_per_row) {
    return UploadPackedQ4X8Internal(packed, rows, blocks_per_row, false);
  }

  std::uint64_t UploadPackedQ4X8Deferred(
      const std::vector<std::uint8_t>& packed,
      std::uint64_t rows,
      std::uint64_t blocks_per_row) {
    return UploadPackedQ4X8Internal(packed, rows, blocks_per_row, true);
  }

  std::uint64_t UploadPackedQ4X8Internal(
      const std::vector<std::uint8_t>& packed,
      std::uint64_t rows,
      std::uint64_t blocks_per_row,
      bool deferred) {
    ValidatePackedQ4X8Inputs(packed, rows, blocks_per_row);
    cl_int err = kClSuccess;
    cl_mem buffer = api_.clCreateBuffer(context_, kClMemReadOnly, packed.size(), nullptr, &err);
    Check(err, "clCreateBuffer(resident packed)");
    try {
      if (deferred) {
        pending_host_uploads_.push_back(packed);
        const auto& staging = pending_host_uploads_.back();
        Check(api_.clEnqueueWriteBuffer(queue_, buffer, kClFalse, 0,
                                        staging.size(), staging.data(), 0,
                                        nullptr, nullptr),
              "clEnqueueWriteBuffer(resident packed deferred)");
      } else {
        Check(api_.clEnqueueWriteBuffer(queue_, buffer, kClTrue, 0,
                                        packed.size(), packed.data(), 0,
                                        nullptr, nullptr),
              "clEnqueueWriteBuffer(resident packed)");
      }
    } catch (...) {
      ReleaseMem(api_, &buffer);
      throw;
    }
    const std::uint64_t handle = next_resident_handle_++;
    resident_packed_q4x8_.emplace(
        handle, ResidentPackedQ4X8{buffer, rows, blocks_per_row, packed.size()});
    return handle;
  }

  std::uint64_t ConcatResidentPackedQ4X8(
      const std::vector<std::uint64_t>& handles) {
    Require(!handles.empty(), "resident packed Q4 concat handles empty");
    std::uint64_t rows = 0;
    std::uint64_t blocks_per_row = 0;
    std::size_t packed_bytes = 0;
    std::vector<ResidentPackedQ4X8> parts;
    parts.reserve(handles.size());
    for (const auto handle : handles) {
      const auto& part = ResidentPackedQ4X8ForHandle(handle);
      if (parts.empty()) {
        blocks_per_row = part.blocks_per_row;
      } else {
        Require(part.blocks_per_row == blocks_per_row,
                "resident packed Q4 concat blocks_per_row mismatch");
      }
      rows += part.rows;
      packed_bytes += part.packed_bytes;
      parts.push_back(part);
    }
    cl_int err = kClSuccess;
    cl_mem buffer =
        api_.clCreateBuffer(context_, kClMemReadWrite, packed_bytes, nullptr, &err);
    Check(err, "clCreateBuffer(resident packed concat)");
    try {
      std::size_t dst_offset = 0;
      for (const auto& part : parts) {
        Check(api_.clEnqueueCopyBuffer(queue_, part.buffer, buffer, 0,
                                       dst_offset, part.packed_bytes, 0,
                                       nullptr, nullptr),
              "clEnqueueCopyBuffer(resident packed concat)");
        dst_offset += part.packed_bytes;
      }
      // The queue is in-order; the consuming matvec kernel will observe these
      // copies without forcing a host-side drain on every selected-set miss.
    } catch (...) {
      ReleaseMem(api_, &buffer);
      throw;
    }
    const std::uint64_t handle = next_resident_handle_++;
    resident_packed_q4x8_.emplace(
        handle, ResidentPackedQ4X8{buffer, rows, blocks_per_row, packed_bytes});
    return handle;
  }

  std::uint64_t UploadRawQ4KCpuOrder(const std::vector<std::uint8_t>& raw,
                                     std::uint64_t rows,
                                     std::uint64_t blocks_per_row) {
    ValidateRawQ4KCpuOrderInputs(raw, rows, blocks_per_row);
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
    const std::uint64_t handle = next_resident_handle_++;
    resident_raw_q4_cpu_order_.emplace(
        handle, ResidentRawQ4KCpuOrder{buffer, rows, blocks_per_row,
                                       raw.size()});
    return handle;
  }

  GpuQ4X8MatvecRun RunResidentPackedQ4X8(
      std::uint64_t handle,
      const std::vector<std::int8_t>& q8_qs,
      const std::vector<std::int16_t>& q8_bsums,
      const std::vector<float>& q8_d,
      int repeat,
      GpuQ4X8KernelVariant variant) {
    const auto& resident = ResidentPackedQ4X8ForHandle(handle);
    ValidateQ8InputPlanes(q8_qs, q8_bsums, q8_d, resident.blocks_per_row, repeat);
    GpuQ4X8MatvecRun run;
    run.output.assign(static_cast<std::size_t>(resident.rows), 0.0f);
    cl_mem q8_qs_buffer = nullptr, q8_bsums_buffer = nullptr;
    cl_mem q8_d_buffer = nullptr, out_buffer = nullptr;
    try {
      CreateRunBuffersWithoutPacked(q8_qs, q8_bsums, q8_d, run.output,
                                    &q8_qs_buffer, &q8_bsums_buffer,
                                    &q8_d_buffer, &out_buffer,
                                    kClMemWriteOnly, true);
      const std::uint64_t row_groups = resident.rows / kRowsInterleaved;
      const std::uint64_t global_work_items =
          variant == GpuQ4X8KernelVariant::kGroup8Serial ? row_groups : resident.rows;
      const bool read_in_kernel = repeat == 1;
      run.timing = RunKernel(
          variant, resident.buffer, q8_qs_buffer, q8_bsums_buffer, q8_d_buffer,
          out_buffer, resident.blocks_per_row, row_groups, global_work_items,
          repeat, read_in_kernel ? run.output.data() : nullptr,
          read_in_kernel ? run.output.size() * sizeof(float) : 0);
      if (!read_in_kernel) {
        Check(api_.clEnqueueReadBuffer(queue_, out_buffer, kClTrue, 0,
                                       run.output.size() * sizeof(float),
                                       run.output.data(), 0, nullptr, nullptr),
              "clEnqueueReadBuffer(resident out)");
      }
    } catch (...) {
      ReleaseMem(api_, &out_buffer);
      ReleaseMem(api_, &q8_d_buffer);
      ReleaseMem(api_, &q8_bsums_buffer);
      ReleaseMem(api_, &q8_qs_buffer);
      throw;
    }
    ReleaseMem(api_, &out_buffer);
    ReleaseMem(api_, &q8_d_buffer);
    ReleaseMem(api_, &q8_bsums_buffer);
    ReleaseMem(api_, &q8_qs_buffer);
    return run;
  }

  GpuDeviceQ8Q4X8MatvecRun
  RunF32InputHandleDeviceQ8ThenResidentPackedQ4X8(
      std::uint64_t handle,
      std::uint64_t input_handle,
      int repeat,
      GpuQ4X8KernelVariant variant,
      bool readback_output = true) {
    const auto& resident = ResidentPackedQ4X8ForHandle(handle);
    const auto& input = ResidentF32BufferForHandle(input_handle);
    Require(input.values == resident.blocks_per_row * kQ8QsPerBlock,
            "device-Q8 Q4 input handle size mismatch");
    Require(repeat > 0, "device-Q8 Q4 repeat must be positive");
    GpuDeviceQ8Q4X8MatvecRun run;
    run.output_host_valid = readback_output;
    if (readback_output) {
      run.output.assign(static_cast<std::size_t>(resident.rows), 0.0f);
    }

    const std::uint64_t block_count = resident.blocks_per_row;
    cl_mem q8_qs_buffer = EnsureScratchBuffer(
        f32_input_q4_scratch_q8_qs_,
        static_cast<std::size_t>(block_count) * kQ8QsPerBlock *
            sizeof(std::int8_t),
        kClMemReadWrite, "device-Q8 Q4 q8_qs");
    cl_mem q8_bsums_buffer = EnsureScratchBuffer(
        f32_input_q4_scratch_q8_bsums_,
        static_cast<std::size_t>(block_count) * kQ8BsumsPerBlock *
            sizeof(std::int16_t),
        kClMemReadWrite, "device-Q8 Q4 q8_bsums");
    cl_mem q8_d_buffer = EnsureScratchBuffer(
        f32_input_q4_scratch_q8_d_,
        static_cast<std::size_t>(block_count) * sizeof(float),
        kClMemReadWrite, "device-Q8 Q4 q8_d");
    cl_mem output_buffer = EnsureScratchBuffer(
        f32_input_q4_scratch_output_,
        static_cast<std::size_t>(resident.rows) * sizeof(float),
        kClMemReadWrite, "device-Q8 Q4 output");

    const auto q8_timing = RunQ8QuantizeWithBsumsKernel(
        input.buffer, block_count, q8_qs_buffer, q8_bsums_buffer,
        q8_d_buffer, repeat);
    const std::uint64_t row_groups = resident.rows / kRowsInterleaved;
    const std::uint64_t global_work_items =
        variant == GpuQ4X8KernelVariant::kGroup8Serial ? row_groups
                                                       : resident.rows;
    const bool read_in_kernel = readback_output && repeat == 1;
    run.timing.matvec = RunKernel(
        variant, resident.buffer, q8_qs_buffer, q8_bsums_buffer, q8_d_buffer,
        output_buffer, resident.blocks_per_row, row_groups, global_work_items,
        repeat, read_in_kernel ? run.output.data() : nullptr,
        read_in_kernel ? run.output.size() * sizeof(float) : 0);
    if (readback_output && !read_in_kernel) {
      Check(api_.clEnqueueReadBuffer(queue_, output_buffer, kClTrue, 0,
                                     run.output.size() * sizeof(float),
                                     run.output.data(), 0, nullptr, nullptr),
            "clEnqueueReadBuffer(device-Q8 Q4 output)");
    }
    run.output_handle = RegisterF32BufferAlias(
        &f32_input_q4_output_alias_handle_, output_buffer,
        static_cast<std::size_t>(resident.rows));
    run.timing.q8_quantize_min_us = q8_timing.min_us;
    run.timing.q8_quantize_mean_us = q8_timing.mean_us;
    run.timing.q8_quantize_global_work_items = q8_timing.global_work_items;
    run.timing.shell_sum_min_us =
        q8_timing.min_us + run.timing.matvec.min_us;
    run.timing.shell_sum_mean_us =
        q8_timing.mean_us + run.timing.matvec.mean_us;
    return run;
  }

  GpuQ4KCpuOrderMatvecRun
  RunF32InputHandleDeviceQ8ThenResidentRawQ4KCpuOrder(
      std::uint64_t handle,
      std::uint64_t input_handle,
      int repeat,
      bool readback_output = true) {
    const auto& resident = ResidentRawQ4KCpuOrderForHandle(handle);
    const auto& input = ResidentF32BufferForHandle(input_handle);
    Require(input.values == resident.blocks_per_row * kQ8QsPerBlock,
            "device-Q8 Q4 CPU-order input handle size mismatch");
    Require(repeat > 0, "device-Q8 Q4 CPU-order repeat must be positive");

    GpuQ4KCpuOrderMatvecRun run;
    run.platform_name = selected_.platform_name;
    run.device_name = selected_.device_name;
    run.build_log = build_log_;
    run.program_build_ms = program_build_ms_;
    if (readback_output) {
      run.output.assign(static_cast<std::size_t>(resident.rows), 0.0f);
    }

    const std::uint64_t block_count = resident.blocks_per_row;
    cl_mem q8_qs_buffer = EnsureScratchBuffer(
        q4_cpu_order_scratch_q8_qs_,
        static_cast<std::size_t>(block_count) * kQ8QsPerBlock *
            sizeof(std::int8_t),
        kClMemReadWrite, "device-Q8 Q4 CPU-order q8_qs");
    cl_mem q8_bsums_buffer = EnsureScratchBuffer(
        q4_cpu_order_scratch_q8_bsums_,
        static_cast<std::size_t>(block_count) * kQ8BsumsPerBlock *
            sizeof(std::int16_t),
        kClMemReadWrite, "device-Q8 Q4 CPU-order q8_bsums");
    cl_mem q8_d_buffer = EnsureScratchBuffer(
        q4_cpu_order_scratch_q8_d_,
        static_cast<std::size_t>(block_count) * sizeof(float),
        kClMemReadWrite, "device-Q8 Q4 CPU-order q8_d");
    cl_mem output_buffer = EnsureScratchBuffer(
        q4_cpu_order_scratch_out_,
        static_cast<std::size_t>(resident.rows) * sizeof(float),
        kClMemReadWrite, "device-Q8 Q4 CPU-order output");

    (void)RunQ8QuantizeWithBsumsKernel(
        input.buffer, block_count, q8_qs_buffer, q8_bsums_buffer,
        q8_d_buffer, repeat);
    const bool read_in_kernel = readback_output && repeat == 1;
    run.timing = RunQ4KCpuOrderKernel(
        resident, q8_qs_buffer, q8_bsums_buffer, q8_d_buffer, output_buffer,
        repeat, read_in_kernel ? run.output.data() : nullptr,
        read_in_kernel ? run.output.size() * sizeof(float) : 0);
    if (readback_output && !read_in_kernel) {
      Check(api_.clEnqueueReadBuffer(queue_, output_buffer, kClTrue, 0,
                                     run.output.size() * sizeof(float),
                                     run.output.data(), 0, nullptr, nullptr),
            "clEnqueueReadBuffer(device-Q8 Q4 CPU-order output)");
    }
    return run;
  }

  GpuQ4X8MatvecRun RunResidentPackedQ4X8Expert8PerExpertQ8(
      const std::vector<std::uint64_t>& handles,
      const std::vector<std::int8_t>& q8_qs,
      const std::vector<std::int16_t>& q8_bsums,
      const std::vector<float>& q8_d,
      int repeat,
      GpuQ4X8KernelVariant variant,
      bool readback_output = true) {
    Require(handles.size() == 8,
            "expert8 selected Q4_K requires exactly 8 handles");
    Require(variant == GpuQ4X8KernelVariant::kRowlaneParallel,
            "expert8 selected Q4_K requires rowlane variant");
    std::array<cl_mem, 8> packed_buffers{};
    std::uint64_t rows_per_expert = 0;
    std::uint64_t blocks_per_row = 0;
    for (std::size_t i = 0; i < handles.size(); ++i) {
      const auto& resident = ResidentPackedQ4X8ForHandle(handles[i]);
      if (i == 0) {
        rows_per_expert = resident.rows;
        blocks_per_row = resident.blocks_per_row;
      } else {
        Require(resident.rows == rows_per_expert,
                "expert8 selected Q4_K row count mismatch");
        Require(resident.blocks_per_row == blocks_per_row,
                "expert8 selected Q4_K block count mismatch");
      }
      packed_buffers[i] = resident.buffer;
    }
    Require(rows_per_expert > 0,
            "expert8 selected Q4_K rows_per_expert must be nonzero");
    Require(rows_per_expert % kRowsInterleaved == 0,
            "expert8 selected Q4_K rows must be divisible by 8");
    ValidateSelectedQ4X8Q8InputPlanes(
        q8_qs, q8_bsums, q8_d, blocks_per_row, handles.size(), repeat);
    GpuQ4X8MatvecRun run;
    const std::size_t output_values =
        static_cast<std::size_t>(rows_per_expert * handles.size());
    if (readback_output) {
      run.output.assign(output_values, 0.0f);
    }
    cl_mem q8_qs_buffer = EnsureScratchBuffer(
        expert8_q4_down_scratch_q8_qs_, q8_qs.size(), kClMemReadOnly,
        "expert8 Q4 down q8_qs");
    cl_mem q8_bsums_buffer = EnsureScratchBuffer(
        expert8_q4_down_scratch_q8_bsums_,
        q8_bsums.size() * sizeof(std::int16_t), kClMemReadOnly,
        "expert8 Q4 down q8_bsums");
    cl_mem q8_d_buffer = EnsureScratchBuffer(
        expert8_q4_down_scratch_q8_d_, q8_d.size() * sizeof(float),
        kClMemReadOnly, "expert8 Q4 down q8_d");
    cl_mem out_buffer = EnsureScratchBuffer(
        expert8_q4_down_scratch_out_, output_values * sizeof(float),
        kClMemWriteOnly, "expert8 Q4 down out");
    const bool defer_finish =
        !readback_output && DeferFfnDownFinishBundle();
    if (defer_finish) {
      pending_host_uploads_.reserve(
          pending_host_uploads_.size() + 3U);
    }
    const void* q8_qs_data =
        defer_finish ? StagePendingHostUpload(q8_qs.data(), q8_qs.size())
                     : static_cast<const void*>(q8_qs.data());
    const void* q8_bsums_data =
        defer_finish
            ? StagePendingHostUpload(
                  q8_bsums.data(), q8_bsums.size() * sizeof(std::int16_t))
            : static_cast<const void*>(q8_bsums.data());
    const void* q8_d_data =
        defer_finish
            ? StagePendingHostUpload(q8_d.data(), q8_d.size() * sizeof(float))
            : static_cast<const void*>(q8_d.data());
    Check(api_.clEnqueueWriteBuffer(queue_, q8_qs_buffer, kClFalse, 0,
                                    q8_qs.size(), q8_qs_data, 0, nullptr,
                                    nullptr),
          "clEnqueueWriteBuffer(expert8 Q4 down q8_qs)");
    Check(api_.clEnqueueWriteBuffer(
              queue_, q8_bsums_buffer, kClFalse, 0,
              q8_bsums.size() * sizeof(std::int16_t), q8_bsums_data, 0,
              nullptr, nullptr),
          "clEnqueueWriteBuffer(expert8 Q4 down q8_bsums)");
    Check(api_.clEnqueueWriteBuffer(queue_, q8_d_buffer, kClFalse, 0,
                                    q8_d.size() * sizeof(float), q8_d_data,
                                    0, nullptr, nullptr),
          "clEnqueueWriteBuffer(expert8 Q4 down q8_d)");
    const std::uint64_t row_groups = rows_per_expert / kRowsInterleaved;
    const bool read_in_kernel = readback_output && repeat == 1;
    run.timing = RunExpert8MultiQ8Kernel(
        packed_buffers, q8_qs_buffer, q8_bsums_buffer, q8_d_buffer,
        out_buffer, blocks_per_row, row_groups, rows_per_expert * handles.size(),
        repeat, read_in_kernel ? run.output.data() : nullptr,
        read_in_kernel ? run.output.size() * sizeof(float) : 0,
        defer_finish);
    if (readback_output && !read_in_kernel) {
      Check(api_.clEnqueueReadBuffer(queue_, out_buffer, kClTrue, 0,
                                     run.output.size() * sizeof(float),
                                     run.output.data(), 0, nullptr, nullptr),
            "clEnqueueReadBuffer(expert8 selected Q4_K out)");
    }
    run.output_handle = RegisterF32BufferAlias(
        &expert8_q4_down_output_alias_handle_, out_buffer, output_values);
    return run;
  }

  GpuQ4X8SelectedSharedMatvecRun
  RunResidentPackedQ4X8Expert8PlusSharedPerExpertQ8(
      const std::vector<std::uint64_t>& selected_handles,
      std::uint64_t shared_handle,
      const GpuQ8KInputPlanes& selected_q8,
      const GpuQ8KInputPlanes& shared_q8,
      int repeat,
      bool readback_selected_output = true,
      bool readback_shared_output = true) {
    const auto input_setup_begin = std::chrono::steady_clock::now();
    Require(selected_handles.size() == 8,
            "selected+shared Q4_K requires exactly 8 selected handles");
    std::array<cl_mem, 8> selected_buffers{};
    std::uint64_t rows_per_expert = 0;
    std::uint64_t blocks_per_row = 0;
    for (std::size_t i = 0; i < selected_handles.size(); ++i) {
      const auto& resident = ResidentPackedQ4X8ForHandle(selected_handles[i]);
      if (i == 0) {
        rows_per_expert = resident.rows;
        blocks_per_row = resident.blocks_per_row;
      } else {
        Require(resident.rows == rows_per_expert,
                "selected+shared selected Q4_K row count mismatch");
        Require(resident.blocks_per_row == blocks_per_row,
                "selected+shared selected Q4_K block count mismatch");
      }
      selected_buffers[i] = resident.buffer;
    }
    const auto& shared = ResidentPackedQ4X8ForHandle(shared_handle);
    Require(rows_per_expert > 0,
            "selected+shared Q4_K rows_per_expert must be nonzero");
    Require(rows_per_expert % kRowsInterleaved == 0,
            "selected+shared Q4_K rows must be divisible by 8");
    Require(shared.rows == rows_per_expert,
            "selected+shared shared Q4_K row count mismatch");
    Require(shared.blocks_per_row == blocks_per_row,
            "selected+shared shared Q4_K block count mismatch");
    ValidateSelectedQ4X8Q8InputPlanes(
        selected_q8.qs, selected_q8.bsums, selected_q8.d, blocks_per_row,
        selected_handles.size(), repeat);
    ValidateQ8InputPlanes(shared_q8.qs, shared_q8.bsums, shared_q8.d,
                          blocks_per_row, repeat);

    GpuQ4X8SelectedSharedMatvecRun run;
    const std::size_t selected_output_values =
        static_cast<std::size_t>(rows_per_expert * selected_handles.size());
    const std::size_t shared_output_values =
        static_cast<std::size_t>(rows_per_expert);
    if (readback_selected_output) {
      run.selected_output.assign(selected_output_values, 0.0f);
    }
    if (readback_shared_output) {
      run.shared_output.assign(shared_output_values, 0.0f);
    }

    cl_mem selected_q8_qs_buffer = EnsureScratchBuffer(
        selected_shared_q4_down_scratch_selected_q8_qs_,
        selected_q8.qs.size(), kClMemReadOnly,
        "selected+shared Q4 down selected q8_qs");
    cl_mem selected_q8_bsums_buffer = EnsureScratchBuffer(
        selected_shared_q4_down_scratch_selected_q8_bsums_,
        selected_q8.bsums.size() * sizeof(std::int16_t), kClMemReadOnly,
        "selected+shared Q4 down selected q8_bsums");
    cl_mem selected_q8_d_buffer = EnsureScratchBuffer(
        selected_shared_q4_down_scratch_selected_q8_d_,
        selected_q8.d.size() * sizeof(float), kClMemReadOnly,
        "selected+shared Q4 down selected q8_d");
    cl_mem shared_q8_qs_buffer = EnsureScratchBuffer(
        selected_shared_q4_down_scratch_shared_q8_qs_, shared_q8.qs.size(),
        kClMemReadOnly, "selected+shared Q4 down shared q8_qs");
    cl_mem shared_q8_bsums_buffer = EnsureScratchBuffer(
        selected_shared_q4_down_scratch_shared_q8_bsums_,
        shared_q8.bsums.size() * sizeof(std::int16_t), kClMemReadOnly,
        "selected+shared Q4 down shared q8_bsums");
    cl_mem shared_q8_d_buffer = EnsureScratchBuffer(
        selected_shared_q4_down_scratch_shared_q8_d_,
        shared_q8.d.size() * sizeof(float), kClMemReadOnly,
        "selected+shared Q4 down shared q8_d");
    cl_mem selected_out_buffer = EnsureScratchBuffer(
        selected_shared_q4_down_scratch_selected_out_,
        selected_output_values * sizeof(float), kClMemWriteOnly,
        "selected+shared Q4 down selected out");
    cl_mem shared_out_buffer = EnsureScratchBuffer(
        selected_shared_q4_down_scratch_shared_out_,
        shared_output_values * sizeof(float), kClMemWriteOnly,
        "selected+shared Q4 down shared out");
    run.selected_output_handle = RegisterF32BufferAlias(
        &selected_shared_q4_down_selected_output_alias_handle_,
        selected_out_buffer, selected_output_values);
    run.shared_output_handle = RegisterF32BufferAlias(
        &selected_shared_q4_down_shared_output_alias_handle_,
        shared_out_buffer, shared_output_values);

    const bool defer_finish =
        !readback_selected_output && !readback_shared_output &&
        DeferFfnDownFinishBundle();
    if (defer_finish) {
      pending_host_uploads_.reserve(pending_host_uploads_.size() + 6U);
    }
    const auto input_setup_wall_ns =
        WallNs(input_setup_begin, std::chrono::steady_clock::now());
    const auto input_write_begin = std::chrono::steady_clock::now();
    const void* selected_q8_qs_data =
        defer_finish
            ? StagePendingHostUpload(selected_q8.qs.data(),
                                     selected_q8.qs.size())
            : static_cast<const void*>(selected_q8.qs.data());
    const void* selected_q8_bsums_data =
        defer_finish
            ? StagePendingHostUpload(
                  selected_q8.bsums.data(),
                  selected_q8.bsums.size() * sizeof(std::int16_t))
            : static_cast<const void*>(selected_q8.bsums.data());
    const void* selected_q8_d_data =
        defer_finish
            ? StagePendingHostUpload(selected_q8.d.data(),
                                     selected_q8.d.size() * sizeof(float))
            : static_cast<const void*>(selected_q8.d.data());
    const void* shared_q8_qs_data =
        defer_finish
            ? StagePendingHostUpload(shared_q8.qs.data(),
                                     shared_q8.qs.size())
            : static_cast<const void*>(shared_q8.qs.data());
    const void* shared_q8_bsums_data =
        defer_finish
            ? StagePendingHostUpload(
                  shared_q8.bsums.data(),
                  shared_q8.bsums.size() * sizeof(std::int16_t))
            : static_cast<const void*>(shared_q8.bsums.data());
    const void* shared_q8_d_data =
        defer_finish
            ? StagePendingHostUpload(shared_q8.d.data(),
                                     shared_q8.d.size() * sizeof(float))
            : static_cast<const void*>(shared_q8.d.data());
    Check(api_.clEnqueueWriteBuffer(queue_, selected_q8_qs_buffer, kClFalse,
                                    0, selected_q8.qs.size(),
                                    selected_q8_qs_data, 0, nullptr,
                                    nullptr),
          "clEnqueueWriteBuffer(selected+shared Q4 selected q8_qs)");
    Check(api_.clEnqueueWriteBuffer(
              queue_, selected_q8_bsums_buffer, kClFalse, 0,
              selected_q8.bsums.size() * sizeof(std::int16_t),
              selected_q8_bsums_data, 0, nullptr, nullptr),
          "clEnqueueWriteBuffer(selected+shared Q4 selected q8_bsums)");
    Check(api_.clEnqueueWriteBuffer(queue_, selected_q8_d_buffer, kClFalse,
                                    0,
                                    selected_q8.d.size() * sizeof(float),
                                    selected_q8_d_data, 0, nullptr, nullptr),
          "clEnqueueWriteBuffer(selected+shared Q4 selected q8_d)");
    Check(api_.clEnqueueWriteBuffer(queue_, shared_q8_qs_buffer, kClFalse,
                                    0, shared_q8.qs.size(),
                                    shared_q8_qs_data, 0, nullptr, nullptr),
          "clEnqueueWriteBuffer(selected+shared Q4 shared q8_qs)");
    Check(api_.clEnqueueWriteBuffer(
              queue_, shared_q8_bsums_buffer, kClFalse, 0,
              shared_q8.bsums.size() * sizeof(std::int16_t),
              shared_q8_bsums_data, 0, nullptr, nullptr),
          "clEnqueueWriteBuffer(selected+shared Q4 shared q8_bsums)");
    Check(api_.clEnqueueWriteBuffer(queue_, shared_q8_d_buffer, kClFalse, 0,
                                    shared_q8.d.size() * sizeof(float),
                                    shared_q8_d_data, 0, nullptr, nullptr),
          "clEnqueueWriteBuffer(selected+shared Q4 shared q8_d)");
    const auto input_write_wall_ns =
        WallNs(input_write_begin, std::chrono::steady_clock::now());

    const std::uint64_t row_groups = rows_per_expert / kRowsInterleaved;
    run.timing = RunExpert8PlusSharedMultiQ8Kernel(
        selected_buffers, shared.buffer, selected_q8_qs_buffer,
        selected_q8_bsums_buffer, selected_q8_d_buffer, shared_q8_qs_buffer,
        shared_q8_bsums_buffer, shared_q8_d_buffer, selected_out_buffer,
        shared_out_buffer, blocks_per_row, row_groups,
        rows_per_expert * (selected_handles.size() + 1), repeat,
        defer_finish);
    run.timing.input_setup_wall_ns += input_setup_wall_ns;
    run.timing.input_write_wall_ns += input_write_wall_ns;

    if (readback_selected_output) {
      const auto read_begin = std::chrono::steady_clock::now();
      Check(api_.clEnqueueReadBuffer(
                queue_, selected_out_buffer, kClTrue, 0,
                run.selected_output.size() * sizeof(float),
                run.selected_output.data(), 0, nullptr, nullptr),
            "clEnqueueReadBuffer(selected+shared selected Q4_K out)");
      run.timing.output_read_wall_ns +=
          WallNs(read_begin, std::chrono::steady_clock::now());
    }
    if (readback_shared_output) {
      const auto read_begin = std::chrono::steady_clock::now();
      Check(api_.clEnqueueReadBuffer(
                queue_, shared_out_buffer, kClTrue, 0,
                run.shared_output.size() * sizeof(float),
                run.shared_output.data(), 0, nullptr, nullptr),
            "clEnqueueReadBuffer(selected+shared shared Q4_K out)");
      run.timing.output_read_wall_ns +=
          WallNs(read_begin, std::chrono::steady_clock::now());
    }
    return run;
  }

  std::uint64_t UploadRawQ6K(const std::vector<std::uint8_t>& raw,
                             std::uint64_t rows,
                             std::uint64_t blocks_per_row) {
    return UploadRawQ6KInternal(raw, rows, blocks_per_row, false);
  }

  std::uint64_t UploadRawQ6KDeferred(const std::vector<std::uint8_t>& raw,
                                     std::uint64_t rows,
                                     std::uint64_t blocks_per_row) {
    return UploadRawQ6KInternal(raw, rows, blocks_per_row, true);
  }

  std::uint64_t UploadSelectedRawQ6KRowstripe(
      const std::vector<std::uint8_t>& raw,
      std::uint64_t rows_per_expert,
      std::uint64_t blocks_per_row,
      std::uint64_t selected_count,
      std::uint64_t rows_per_tile) {
    return UploadSelectedRawQ6KRowstripeInternal(
        raw, rows_per_expert, blocks_per_row, selected_count, rows_per_tile,
        false);
  }

  std::uint64_t UploadSelectedRawQ6KRowstripeDeferred(
      const std::vector<std::uint8_t>& raw,
      std::uint64_t rows_per_expert,
      std::uint64_t blocks_per_row,
      std::uint64_t selected_count,
      std::uint64_t rows_per_tile) {
    return UploadSelectedRawQ6KRowstripeInternal(
        raw, rows_per_expert, blocks_per_row, selected_count, rows_per_tile,
        true);
  }

  std::uint64_t UploadRawQ6KInternal(const std::vector<std::uint8_t>& raw,
                                     std::uint64_t rows,
                                     std::uint64_t blocks_per_row,
                                     bool deferred) {
    ValidateRawQ6KInputs(raw, rows, blocks_per_row);
    cl_int err = kClSuccess;
    cl_mem buffer =
        api_.clCreateBuffer(context_, kClMemReadOnly, raw.size(), nullptr, &err);
    Check(err, "clCreateBuffer(resident raw Q6_K)");
    try {
      if (deferred) {
        pending_host_uploads_.push_back(raw);
        const auto& staging = pending_host_uploads_.back();
        Check(api_.clEnqueueWriteBuffer(queue_, buffer, kClFalse, 0,
                                        staging.size(), staging.data(), 0,
                                        nullptr, nullptr),
              "clEnqueueWriteBuffer(resident raw Q6_K deferred)");
      } else {
        Check(api_.clEnqueueWriteBuffer(queue_, buffer, kClTrue, 0,
                                        raw.size(), raw.data(), 0, nullptr,
                                        nullptr),
              "clEnqueueWriteBuffer(resident raw Q6_K)");
      }
    } catch (...) {
      ReleaseMem(api_, &buffer);
      throw;
    }
    const std::uint64_t handle = next_resident_handle_++;
    resident_raw_q6k_.emplace(handle, ResidentRawQ6K{
                                          buffer, rows, blocks_per_row, raw.size()});
    return handle;
  }

  std::uint64_t UploadSelectedRawQ6KRowstripeInternal(
      const std::vector<std::uint8_t>& raw,
      std::uint64_t rows_per_expert,
      std::uint64_t blocks_per_row,
      std::uint64_t selected_count,
      std::uint64_t rows_per_tile,
      bool deferred) {
    const auto layout = BuildSelectedRawQ6KRowstripe(
        raw, rows_per_expert, blocks_per_row, selected_count, rows_per_tile);
    cl_int err = kClSuccess;
    cl_mem buffer = api_.clCreateBuffer(
        context_, kClMemReadOnly, layout.bytes.size(), nullptr, &err);
    Check(err, "clCreateBuffer(resident rowstripe Q6_K)");
    try {
      if (deferred) {
        pending_host_uploads_.push_back(layout.bytes);
        const auto& staging = pending_host_uploads_.back();
        Check(api_.clEnqueueWriteBuffer(queue_, buffer, kClFalse, 0,
                                        staging.size(), staging.data(), 0,
                                        nullptr, nullptr),
              "clEnqueueWriteBuffer(resident rowstripe Q6_K deferred)");
      } else {
        Check(api_.clEnqueueWriteBuffer(queue_, buffer, kClTrue, 0,
                                        layout.bytes.size(),
                                        layout.bytes.data(), 0, nullptr,
                                        nullptr),
              "clEnqueueWriteBuffer(resident rowstripe Q6_K)");
      }
    } catch (...) {
      ReleaseMem(api_, &buffer);
      throw;
    }
    const std::uint64_t handle = next_resident_handle_++;
    resident_raw_q6k_.emplace(
        handle, ResidentRawQ6K{buffer,
                               rows_per_expert * selected_count,
                               blocks_per_row,
                               layout.bytes.size(),
                               true,
                               layout.rows_per_tile,
                               layout.row_tile_count});
    return handle;
  }

  void ClearPendingHostUploadsAfterQueueDrain() {
    pending_host_uploads_.clear();
  }

  const void* StagePendingHostUpload(const void* data, std::size_t bytes) {
    pending_host_uploads_.emplace_back(bytes);
    auto& staging = pending_host_uploads_.back();
    std::memcpy(staging.data(), data, bytes);
    return staging.data();
  }

  std::uint64_t ConcatResidentRawQ6K(
      const std::vector<std::uint64_t>& handles) {
    Require(!handles.empty(), "resident Q6_K concat handles empty");
    std::uint64_t rows = 0;
    std::uint64_t blocks_per_row = 0;
    bool rowstripe = false;
    std::uint64_t rows_per_tile = 0;
    std::uint64_t row_tile_count = 0;
    std::size_t raw_bytes = 0;
    std::vector<ResidentRawQ6K> parts;
    parts.reserve(handles.size());
    for (const auto handle : handles) {
      const auto& part = ResidentRawQ6KForHandle(handle);
      if (parts.empty()) {
        blocks_per_row = part.blocks_per_row;
        rowstripe = part.rowstripe;
        rows_per_tile = part.rows_per_tile;
        row_tile_count = part.row_tile_count;
      } else {
        Require(part.blocks_per_row == blocks_per_row,
                "resident Q6_K concat blocks_per_row mismatch");
        Require(part.rowstripe == rowstripe,
                "resident Q6_K concat layout mismatch");
        Require(part.rows_per_tile == rows_per_tile,
                "resident Q6_K concat rows_per_tile mismatch");
        Require(part.row_tile_count == row_tile_count,
                "resident Q6_K concat row_tile_count mismatch");
      }
      rows += part.rows;
      raw_bytes += part.raw_bytes;
      parts.push_back(part);
    }
    cl_int err = kClSuccess;
    cl_mem buffer =
        api_.clCreateBuffer(context_, kClMemReadWrite, raw_bytes, nullptr, &err);
    Check(err, "clCreateBuffer(resident Q6_K concat)");
    try {
      std::size_t dst_offset = 0;
      for (const auto& part : parts) {
        Check(api_.clEnqueueCopyBuffer(queue_, part.buffer, buffer, 0,
                                       dst_offset, part.raw_bytes, 0,
                                       nullptr, nullptr),
              "clEnqueueCopyBuffer(resident Q6_K concat)");
        dst_offset += part.raw_bytes;
      }
      // The queue is in-order; the consuming Q6 kernel will observe these
      // copies without forcing a host-side drain on every selected-set miss.
    } catch (...) {
      ReleaseMem(api_, &buffer);
      throw;
    }
    const std::uint64_t handle = next_resident_handle_++;
    resident_raw_q6k_.emplace(
        handle, ResidentRawQ6K{buffer, rows, blocks_per_row, raw_bytes,
                               rowstripe, rows_per_tile, row_tile_count});
    return handle;
  }

  static void InsertGpuTopK(std::vector<GpuTopKRow>& rows,
                            GpuTopKRow row,
                            int topk) {
    auto pos = rows.begin();
    while (pos != rows.end() && pos->value >= row.value) {
      ++pos;
    }
    rows.insert(pos, row);
    if (static_cast<int>(rows.size()) > topk) {
      rows.pop_back();
    }
  }

  GpuQ6KMatvecRun RunResidentRawQ6K(std::uint64_t handle,
                                    const GpuQ8KInputPlanes& q8,
                                    int repeat) {
    const auto& resident = ResidentRawQ6KForHandle(handle);
    ValidateQ6KQ8InputPlanes(q8, resident.blocks_per_row, repeat);
    GpuQ6KMatvecRun run;
    run.output.assign(static_cast<std::size_t>(resident.rows), 0.0f);
    cl_mem q8_qs_buffer = nullptr;
    cl_mem q8_d_buffer = nullptr;
    cl_mem out_buffer = nullptr;
    auto release_all = [&]() {
      ReleaseMem(api_, &out_buffer);
      ReleaseMem(api_, &q8_d_buffer);
      ReleaseMem(api_, &q8_qs_buffer);
    };
    try {
      CreateQ6RunBuffers(q8, run.output, &q8_qs_buffer, &q8_d_buffer,
                         &out_buffer);
      run.timing =
          resident.rowstripe
              ? RunQ6KSelectedRowstripeKernel(
                    resident.buffer, q8_qs_buffer, q8_d_buffer, out_buffer,
                    resident.rows, resident.blocks_per_row, resident.rows,
                    resident.rows_per_tile, repeat)
              : RunQ6KKernel(resident.buffer, q8_qs_buffer, q8_d_buffer,
                             out_buffer, resident.rows,
                             resident.blocks_per_row, repeat);
      Check(api_.clEnqueueReadBuffer(queue_, out_buffer, kClTrue, 0,
                                     run.output.size() * sizeof(float),
                                     run.output.data(), 0, nullptr, nullptr),
            "clEnqueueReadBuffer(resident Q6_K out)");
    } catch (...) {
      release_all();
      throw;
    }
    release_all();
    return run;
  }

  GpuQ6KMatvecRun RunResidentRawQ6KToF32Handle(std::uint64_t handle,
                                               const GpuQ8KInputPlanes& q8,
                                               int repeat) {
    const auto& resident = ResidentRawQ6KForHandle(handle);
    ValidateQ6KQ8InputPlanes(q8, resident.blocks_per_row, repeat);
    GpuQ6KMatvecRun run;
    cl_mem q8_qs_buffer = EnsureScratchBuffer(
        resident_q6_handoff_scratch_q8_qs_, q8.qs.size(), kClMemReadOnly,
        "resident Q6 handoff q8_qs");
    cl_mem q8_d_buffer = EnsureScratchBuffer(
        resident_q6_handoff_scratch_q8_d_, q8.d.size() * sizeof(float),
        kClMemReadOnly, "resident Q6 handoff q8_d");
    cl_mem out_buffer = EnsureScratchBuffer(
        resident_q6_handoff_scratch_out_,
        static_cast<std::size_t>(resident.rows) * sizeof(float),
        kClMemWriteOnly, "resident Q6 handoff out");
    run.output_handle = RegisterF32BufferAlias(
        &resident_q6_handoff_output_alias_handle_, out_buffer,
        static_cast<std::size_t>(resident.rows));
    const bool defer_finish = DeferFfnDownFinishBundle();
    const void* q8_qs_data =
        defer_finish ? StagePendingHostUpload(q8.qs.data(), q8.qs.size())
                     : static_cast<const void*>(q8.qs.data());
    const void* q8_d_data =
        defer_finish
            ? StagePendingHostUpload(q8.d.data(), q8.d.size() * sizeof(float))
            : static_cast<const void*>(q8.d.data());
    Check(api_.clEnqueueWriteBuffer(queue_, q8_qs_buffer, kClFalse, 0,
                                    q8.qs.size(), q8_qs_data, 0, nullptr,
                                    nullptr),
          "clEnqueueWriteBuffer(resident Q6 handoff q8_qs)");
    Check(api_.clEnqueueWriteBuffer(queue_, q8_d_buffer, kClFalse, 0,
                                    q8.d.size() * sizeof(float), q8_d_data,
                                    0, nullptr, nullptr),
          "clEnqueueWriteBuffer(resident Q6 handoff q8_d)");
    run.timing =
        resident.rowstripe
            ? RunQ6KSelectedRowstripeKernel(
                  resident.buffer, q8_qs_buffer, q8_d_buffer, out_buffer,
                  resident.rows, resident.blocks_per_row, resident.rows,
                  resident.rows_per_tile, repeat, defer_finish)
            : RunQ6KKernel(resident.buffer, q8_qs_buffer, q8_d_buffer,
                           out_buffer, resident.rows,
                           resident.blocks_per_row, repeat, defer_finish);
    return run;
  }

  GpuDeviceQ8Q6KMatvecRun RunF32InputHandleDeviceQ8ThenResidentRawQ6K(
      std::uint64_t handle,
      std::uint64_t input_handle,
      int repeat,
      bool readback_output = true) {
    const auto& resident = ResidentRawQ6KForHandle(handle);
    const auto& input = ResidentF32BufferForHandle(input_handle);
    Require(input.values == resident.blocks_per_row * kQ8QsPerBlock,
            "device-Q8 Q6 input handle size mismatch");
    Require(repeat > 0, "device-Q8 Q6 repeat must be positive");
    GpuDeviceQ8Q6KMatvecRun run;
    run.output_host_valid = readback_output;
    if (readback_output) {
      run.output.assign(static_cast<std::size_t>(resident.rows), 0.0f);
    }

    const std::uint64_t block_count = resident.blocks_per_row;
    cl_mem q8_qs_buffer = EnsureScratchBuffer(
        f32_input_q6_scratch_q8_qs_,
        static_cast<std::size_t>(block_count) * kQ8QsPerBlock *
            sizeof(std::int8_t),
        kClMemReadWrite, "device-Q8 Q6 q8_qs");
    cl_mem q8_d_buffer = EnsureScratchBuffer(
        f32_input_q6_scratch_q8_d_,
        static_cast<std::size_t>(block_count) * sizeof(float),
        kClMemReadWrite, "device-Q8 Q6 q8_d");
    cl_mem output_buffer = EnsureScratchBuffer(
        f32_input_q6_scratch_output_,
        static_cast<std::size_t>(resident.rows) * sizeof(float),
        kClMemReadWrite, "device-Q8 Q6 output");

    const auto q8_timing = RunQ8QuantizeKernel(
        input.buffer, block_count, q8_qs_buffer, q8_d_buffer, repeat);
    run.timing.matvec =
        resident.rowstripe
            ? RunQ6KSelectedRowstripeKernel(
                  resident.buffer, q8_qs_buffer, q8_d_buffer, output_buffer,
                  resident.rows, resident.blocks_per_row, resident.rows,
                  resident.rows_per_tile, repeat)
            : RunQ6KKernel(resident.buffer, q8_qs_buffer, q8_d_buffer,
                           output_buffer, resident.rows,
                           resident.blocks_per_row, repeat);
    if (readback_output) {
      Check(api_.clEnqueueReadBuffer(queue_, output_buffer, kClTrue, 0,
                                     run.output.size() * sizeof(float),
                                     run.output.data(), 0, nullptr, nullptr),
            "clEnqueueReadBuffer(device-Q8 Q6 output)");
    }
    run.output_handle = RegisterF32BufferAlias(
        &f32_input_q6_output_alias_handle_, output_buffer,
        static_cast<std::size_t>(resident.rows));
    run.timing.q8_quantize_min_us = q8_timing.min_us;
    run.timing.q8_quantize_mean_us = q8_timing.mean_us;
    run.timing.q8_quantize_global_work_items = q8_timing.global_work_items;
    run.timing.shell_sum_min_us =
        q8_timing.min_us + run.timing.matvec.min_us;
    run.timing.shell_sum_mean_us =
        q8_timing.mean_us + run.timing.matvec.mean_us;
    return run;
  }

  GpuQ6KTopKRun RunResidentRawQ6KTopK(std::uint64_t handle,
                                      const GpuQ8KInputPlanes& q8,
                                      int topk,
                                      int repeat) {
    const auto& resident = ResidentRawQ6KForHandle(handle);
    Require(topk > 0 && topk <= 8, "resident Q6_K top-k must be in 1..8");
    ValidateQ6KQ8InputPlanes(q8, resident.blocks_per_row, repeat);
    constexpr std::uint64_t kTopKBlockSize = 256;
    const std::uint64_t partial_count =
        (resident.rows + kTopKBlockSize - 1) / kTopKBlockSize;
    GpuQ6KTopKRun run;
    cl_mem q8_qs_buffer = nullptr;
    cl_mem q8_d_buffer = nullptr;
    cl_mem out_buffer = nullptr;
    cl_mem partial_ids_buffer = nullptr;
    cl_mem partial_values_buffer = nullptr;
    auto release_all = [&]() {
      ReleaseMem(api_, &partial_values_buffer);
      ReleaseMem(api_, &partial_ids_buffer);
      ReleaseMem(api_, &out_buffer);
      ReleaseMem(api_, &q8_d_buffer);
      ReleaseMem(api_, &q8_qs_buffer);
    };
    try {
      std::vector<float> output(
          static_cast<std::size_t>(resident.rows), 0.0f);
      CreateQ6RunBuffers(q8, output, &q8_qs_buffer, &q8_d_buffer,
                         &out_buffer);
      cl_int err = kClSuccess;
      partial_ids_buffer = api_.clCreateBuffer(
          context_, kClMemWriteOnly,
          static_cast<std::size_t>(partial_count * 8) * sizeof(std::int32_t),
          nullptr, &err);
      Check(err, "clCreateBuffer(Q6 top-k partial ids)");
      partial_values_buffer = api_.clCreateBuffer(
          context_, kClMemWriteOnly,
          static_cast<std::size_t>(partial_count * 8) * sizeof(float),
          nullptr, &err);
      Check(err, "clCreateBuffer(Q6 top-k partial values)");

      run.timing.matvec =
          resident.rowstripe
              ? RunQ6KSelectedRowstripeKernel(
                    resident.buffer, q8_qs_buffer, q8_d_buffer, out_buffer,
                    resident.rows, resident.blocks_per_row, resident.rows,
                    resident.rows_per_tile, repeat)
              : RunQ6KKernel(resident.buffer, q8_qs_buffer, q8_d_buffer,
                             out_buffer, resident.rows,
                             resident.blocks_per_row, repeat);
      const auto partial_timing = RunF32TopK8BlocksKernel(
          out_buffer, resident.rows, kTopKBlockSize, partial_ids_buffer,
          partial_values_buffer, partial_count, repeat);
      run.timing.partial_topk_min_us = partial_timing.min_us;
      run.timing.partial_topk_mean_us = partial_timing.mean_us;
      run.timing.partial_topk_global_work_items =
          partial_timing.global_work_items;
      run.timing.shell_sum_min_us =
          run.timing.matvec.min_us + run.timing.partial_topk_min_us;
      run.timing.shell_sum_mean_us =
          run.timing.matvec.mean_us + run.timing.partial_topk_mean_us;

      std::vector<std::int32_t> partial_ids(
          static_cast<std::size_t>(partial_count * 8), -1);
      std::vector<float> partial_values(
          static_cast<std::size_t>(partial_count * 8),
          -std::numeric_limits<float>::infinity());
      Check(api_.clEnqueueReadBuffer(
                queue_, partial_ids_buffer, kClTrue, 0,
                partial_ids.size() * sizeof(std::int32_t),
                partial_ids.data(), 0, nullptr, nullptr),
            "clEnqueueReadBuffer(Q6 top-k partial ids)");
      Check(api_.clEnqueueReadBuffer(
                queue_, partial_values_buffer, kClTrue, 0,
                partial_values.size() * sizeof(float), partial_values.data(),
                0, nullptr, nullptr),
            "clEnqueueReadBuffer(Q6 top-k partial values)");
      for (std::size_t i = 0; i < partial_values.size(); ++i) {
        if (partial_ids[i] < 0) {
          continue;
        }
        InsertGpuTopK(run.topk, GpuTopKRow{partial_ids[i], partial_values[i]},
                      topk);
      }
    } catch (...) {
      release_all();
      throw;
    }
    release_all();
    return run;
  }

  GpuQ6KMatvecRun RunResidentRawQ6KSelected(
      std::uint64_t handle,
      const GpuQ8KInputPlanes& q8,
      std::uint64_t rows_per_expert,
      std::uint64_t selected_count,
      int repeat,
      bool readback_output = true) {
    const auto& resident = ResidentRawQ6KForHandle(handle);
    Require(rows_per_expert > 0, "selected Q6_K rows_per_expert must be nonzero");
    Require(selected_count > 0, "selected Q6_K selected_count must be nonzero");
    Require(resident.rows == rows_per_expert * selected_count,
            "resident selected Q6_K row count mismatch");
    ValidateSelectedQ6KQ8InputPlanes(
        q8, resident.blocks_per_row, selected_count, repeat);
    GpuQ6KMatvecRun run;
    const std::size_t output_values = static_cast<std::size_t>(resident.rows);
    if (readback_output) {
      run.output.assign(output_values, 0.0f);
    }
    if (!readback_output) {
      cl_mem q8_qs_buffer = EnsureScratchBuffer(
          resident_selected_q6_handoff_scratch_q8_qs_, q8.qs.size(),
          kClMemReadOnly,
          "resident selected Q6 handoff q8_qs");
      cl_mem q8_d_buffer = EnsureScratchBuffer(
          resident_selected_q6_handoff_scratch_q8_d_,
          q8.d.size() * sizeof(float),
          kClMemReadOnly, "resident selected Q6 handoff q8_d");
      cl_mem out_buffer = EnsureScratchBuffer(
          resident_selected_q6_handoff_scratch_out_,
          output_values * sizeof(float), kClMemWriteOnly,
          "resident selected Q6 handoff out");
      run.output_handle = RegisterF32BufferAlias(
          &resident_selected_q6_handoff_output_alias_handle_, out_buffer,
          output_values);
      Check(api_.clEnqueueWriteBuffer(queue_, q8_qs_buffer, kClFalse, 0,
                                      q8.qs.size(), q8.qs.data(), 0,
                                      nullptr, nullptr),
            "clEnqueueWriteBuffer(resident selected Q6 handoff q8_qs)");
      Check(api_.clEnqueueWriteBuffer(queue_, q8_d_buffer, kClFalse, 0,
                                      q8.d.size() * sizeof(float),
                                      q8.d.data(), 0, nullptr, nullptr),
            "clEnqueueWriteBuffer(resident selected Q6 handoff q8_d)");
      run.timing =
          resident.rowstripe
              ? RunQ6KSelectedRowstripeKernel(
                    resident.buffer, q8_qs_buffer, q8_d_buffer, out_buffer,
                    rows_per_expert, resident.blocks_per_row, resident.rows,
                    resident.rows_per_tile, repeat)
              : RunQ6KSelectedKernel(
                    resident.buffer, q8_qs_buffer, q8_d_buffer, out_buffer,
                    rows_per_expert, resident.blocks_per_row, resident.rows,
                    repeat);
      return run;
    }
    cl_mem q8_qs_buffer = nullptr;
    cl_mem q8_d_buffer = nullptr;
    cl_mem out_buffer = nullptr;
    auto release_all = [&]() {
      ReleaseMem(api_, &out_buffer);
      ReleaseMem(api_, &q8_d_buffer);
      ReleaseMem(api_, &q8_qs_buffer);
    };
    try {
      CreateQ6RunBuffers(q8, run.output, &q8_qs_buffer, &q8_d_buffer,
                         &out_buffer);
      run.timing =
          resident.rowstripe
              ? RunQ6KSelectedRowstripeKernel(
                    resident.buffer, q8_qs_buffer, q8_d_buffer, out_buffer,
                    rows_per_expert, resident.blocks_per_row, resident.rows,
                    resident.rows_per_tile, repeat)
              : RunQ6KSelectedKernel(
                    resident.buffer, q8_qs_buffer, q8_d_buffer, out_buffer,
                    rows_per_expert, resident.blocks_per_row, resident.rows,
                    repeat);
      Check(api_.clEnqueueReadBuffer(queue_, out_buffer, kClTrue, 0,
                                     run.output.size() * sizeof(float),
                                     run.output.data(), 0, nullptr, nullptr),
            "clEnqueueReadBuffer(resident selected Q6_K out)");
    } catch (...) {
      release_all();
      throw;
    }
    release_all();
    return run;
  }

  GpuQ6KMatvecRun RunResidentRawQ6KExpert8(
      const std::vector<std::uint64_t>& handles,
      const GpuQ8KInputPlanes& q8,
      std::uint64_t rows_per_expert,
      int repeat,
      bool readback_output = true) {
    const auto input_setup_begin = std::chrono::steady_clock::now();
    const auto wall_ns =
        [](std::chrono::steady_clock::time_point begin,
           std::chrono::steady_clock::time_point end) -> std::uint64_t {
      return static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin)
              .count());
    };
    Require(handles.size() == 8,
            "expert8 selected Q6_K requires exactly 8 handles");
    Require(rows_per_expert > 0,
            "expert8 selected Q6_K rows_per_expert must be nonzero");
    std::array<cl_mem, 8> raw_buffers{};
    std::uint64_t blocks_per_row = 0;
    bool rowstripe = false;
    std::uint64_t rows_per_tile = 0;
    std::uint64_t row_tile_count = 0;
    for (std::size_t i = 0; i < handles.size(); ++i) {
      const auto& resident = ResidentRawQ6KForHandle(handles[i]);
      Require(resident.rows == rows_per_expert,
              "expert8 selected Q6_K row count mismatch");
      if (i == 0) {
        blocks_per_row = resident.blocks_per_row;
        rowstripe = resident.rowstripe;
        rows_per_tile = resident.rows_per_tile;
        row_tile_count = resident.row_tile_count;
      } else {
        Require(resident.blocks_per_row == blocks_per_row,
                "expert8 selected Q6_K block count mismatch");
        Require(resident.rowstripe == rowstripe,
                "expert8 selected Q6_K rowstripe layout mismatch");
        Require(resident.rows_per_tile == rows_per_tile,
                "expert8 selected Q6_K rows_per_tile mismatch");
        Require(resident.row_tile_count == row_tile_count,
                "expert8 selected Q6_K row tile count mismatch");
      }
      if (rowstripe) {
        Require(resident.rows_per_tile > 0,
                "expert8 selected Q6_K rowstripe tile size missing");
        Require(resident.row_tile_count > 0,
                "expert8 selected Q6_K rowstripe tile count missing");
      }
      raw_buffers[i] = resident.buffer;
    }
    const std::uint64_t selected_count = handles.size();
    ValidateSelectedQ6KQ8InputPlanes(q8, blocks_per_row, selected_count, repeat);
    GpuQ6KMatvecRun run;
    const std::size_t output_values =
        static_cast<std::size_t>(rows_per_expert * selected_count);
    if (readback_output) {
      run.output.assign(output_values, 0.0f);
    }
    cl_mem q8_qs_buffer = EnsureScratchBuffer(
        expert8_q6_down_scratch_q8_qs_, q8.qs.size(), kClMemReadOnly,
        "expert8 Q6 down q8_qs");
    cl_mem q8_d_buffer = EnsureScratchBuffer(
        expert8_q6_down_scratch_q8_d_, q8.d.size() * sizeof(float),
        kClMemReadOnly, "expert8 Q6 down q8_d");
    cl_mem out_buffer = EnsureScratchBuffer(
        expert8_q6_down_scratch_out_, output_values * sizeof(float),
        kClMemWriteOnly, "expert8 Q6 down out");
    run.output_handle = RegisterF32BufferAlias(
        &expert8_q6_down_output_alias_handle_, out_buffer, output_values);
    const auto input_setup_wall_ns =
        wall_ns(input_setup_begin, std::chrono::steady_clock::now());
    const auto input_write_begin = std::chrono::steady_clock::now();
    Check(api_.clEnqueueWriteBuffer(queue_, q8_qs_buffer, kClFalse, 0,
                                    q8.qs.size(), q8.qs.data(), 0, nullptr,
                                    nullptr),
          "clEnqueueWriteBuffer(expert8 Q6 down q8_qs)");
    Check(api_.clEnqueueWriteBuffer(queue_, q8_d_buffer, kClFalse, 0,
                                    q8.d.size() * sizeof(float), q8.d.data(),
                                    0, nullptr, nullptr),
          "clEnqueueWriteBuffer(expert8 Q6 down q8_d)");
    const auto input_write_wall_ns =
        wall_ns(input_write_begin, std::chrono::steady_clock::now());
    const bool read_in_kernel = readback_output && repeat == 1;
    run.timing =
        rowstripe
            ? RunQ6KExpert8RowstripeKernel(
                  raw_buffers, q8_qs_buffer, q8_d_buffer, out_buffer,
                  rows_per_expert, blocks_per_row, rows_per_tile,
                  rows_per_expert * selected_count, repeat,
                  read_in_kernel ? run.output.data() : nullptr,
                  read_in_kernel ? run.output.size() * sizeof(float) : 0)
            : RunQ6KExpert8Kernel(
                  raw_buffers, q8_qs_buffer, q8_d_buffer, out_buffer,
                  rows_per_expert, blocks_per_row,
                  rows_per_expert * selected_count, repeat,
                  read_in_kernel ? run.output.data() : nullptr,
                  read_in_kernel ? run.output.size() * sizeof(float) : 0);
    run.timing.input_setup_wall_ns += input_setup_wall_ns;
    run.timing.input_write_wall_ns += input_write_wall_ns;
    if (readback_output && !read_in_kernel) {
      const auto read_begin = std::chrono::steady_clock::now();
      Check(api_.clEnqueueReadBuffer(queue_, out_buffer, kClTrue, 0,
                                     run.output.size() * sizeof(float),
                                     run.output.data(), 0, nullptr, nullptr),
            "clEnqueueReadBuffer(expert8 selected Q6_K out)");
      run.timing.output_read_wall_ns +=
          wall_ns(read_begin, std::chrono::steady_clock::now());
    }
    return run;
  }

  GpuQ6KSelectedSharedMatvecRun RunResidentRawQ6KExpert8PlusShared(
      const std::vector<std::uint64_t>& selected_handles,
      std::uint64_t shared_handle,
      const GpuQ8KInputPlanes& selected_q8,
      const GpuQ8KInputPlanes& shared_q8,
      std::uint64_t rows_per_expert,
      int repeat,
      bool readback_selected_output = true,
      bool readback_shared_output = true) {
    const auto input_setup_begin = std::chrono::steady_clock::now();
    const auto wall_ns =
        [](std::chrono::steady_clock::time_point begin,
           std::chrono::steady_clock::time_point end) -> std::uint64_t {
      return static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin)
              .count());
    };
    Require(selected_handles.size() == 8,
            "selected+shared Q6_K requires exactly 8 selected handles");
    Require(rows_per_expert > 0,
            "selected+shared Q6_K rows_per_expert must be nonzero");
    std::array<cl_mem, 8> selected_buffers{};
    std::uint64_t blocks_per_row = 0;
    std::uint64_t rows_per_tile = 0;
    for (std::size_t i = 0; i < selected_handles.size(); ++i) {
      const auto& resident = ResidentRawQ6KForHandle(selected_handles[i]);
      Require(resident.rows == rows_per_expert,
              "selected+shared selected Q6_K row count mismatch");
      Require(resident.rowstripe,
              "selected+shared selected Q6_K requires rowstripe layout");
      if (i == 0) {
        blocks_per_row = resident.blocks_per_row;
        rows_per_tile = resident.rows_per_tile;
      } else {
        Require(resident.blocks_per_row == blocks_per_row,
                "selected+shared selected Q6_K block count mismatch");
        Require(resident.rows_per_tile == rows_per_tile,
                "selected+shared selected Q6_K rows_per_tile mismatch");
      }
      selected_buffers[i] = resident.buffer;
    }
    const auto& shared = ResidentRawQ6KForHandle(shared_handle);
    Require(shared.rows == rows_per_expert,
            "selected+shared shared Q6_K row count mismatch");
    Require(shared.blocks_per_row == blocks_per_row,
            "selected+shared shared Q6_K block count mismatch");
    Require(!shared.rowstripe,
            "selected+shared shared Q6_K requires raw row layout");
    ValidateSelectedQ6KQ8InputPlanes(
        selected_q8, blocks_per_row, selected_handles.size(), repeat);
    ValidateQ6KQ8InputPlanes(shared_q8, blocks_per_row, repeat);

    GpuQ6KSelectedSharedMatvecRun run;
    const std::size_t selected_output_values =
        static_cast<std::size_t>(rows_per_expert * selected_handles.size());
    const std::size_t shared_output_values =
        static_cast<std::size_t>(rows_per_expert);
    if (readback_selected_output) {
      run.selected_output.assign(selected_output_values, 0.0f);
    }
    if (readback_shared_output) {
      run.shared_output.assign(shared_output_values, 0.0f);
    }

    cl_mem selected_q8_qs_buffer = EnsureScratchBuffer(
        selected_shared_q6_down_scratch_selected_q8_qs_, selected_q8.qs.size(),
        kClMemReadOnly, "selected+shared Q6 down selected q8_qs");
    cl_mem selected_q8_d_buffer = EnsureScratchBuffer(
        selected_shared_q6_down_scratch_selected_q8_d_,
        selected_q8.d.size() * sizeof(float), kClMemReadOnly,
        "selected+shared Q6 down selected q8_d");
    cl_mem shared_q8_qs_buffer = EnsureScratchBuffer(
        selected_shared_q6_down_scratch_shared_q8_qs_, shared_q8.qs.size(),
        kClMemReadOnly, "selected+shared Q6 down shared q8_qs");
    cl_mem shared_q8_d_buffer = EnsureScratchBuffer(
        selected_shared_q6_down_scratch_shared_q8_d_,
        shared_q8.d.size() * sizeof(float), kClMemReadOnly,
        "selected+shared Q6 down shared q8_d");
    cl_mem selected_out_buffer = EnsureScratchBuffer(
        selected_shared_q6_down_scratch_selected_out_,
        selected_output_values * sizeof(float), kClMemWriteOnly,
        "selected+shared Q6 down selected out");
    cl_mem shared_out_buffer = EnsureScratchBuffer(
        selected_shared_q6_down_scratch_shared_out_,
        shared_output_values * sizeof(float), kClMemWriteOnly,
        "selected+shared Q6 down shared out");
    run.selected_output_handle = RegisterF32BufferAlias(
        &selected_shared_q6_down_selected_output_alias_handle_,
        selected_out_buffer, selected_output_values);
    run.shared_output_handle = RegisterF32BufferAlias(
        &selected_shared_q6_down_shared_output_alias_handle_,
        shared_out_buffer, shared_output_values);

    const bool defer_finish =
        !readback_selected_output && !readback_shared_output &&
        DeferFfnDownFinishBundle();
    if (defer_finish) {
      pending_host_uploads_.reserve(pending_host_uploads_.size() + 4U);
    }
    const auto input_setup_wall_ns =
        wall_ns(input_setup_begin, std::chrono::steady_clock::now());
    const auto input_write_begin = std::chrono::steady_clock::now();
    const void* selected_q8_qs_data =
        defer_finish
            ? StagePendingHostUpload(selected_q8.qs.data(),
                                     selected_q8.qs.size())
            : static_cast<const void*>(selected_q8.qs.data());
    const void* selected_q8_d_data =
        defer_finish
            ? StagePendingHostUpload(selected_q8.d.data(),
                                     selected_q8.d.size() * sizeof(float))
            : static_cast<const void*>(selected_q8.d.data());
    const void* shared_q8_qs_data =
        defer_finish
            ? StagePendingHostUpload(shared_q8.qs.data(),
                                     shared_q8.qs.size())
            : static_cast<const void*>(shared_q8.qs.data());
    const void* shared_q8_d_data =
        defer_finish
            ? StagePendingHostUpload(shared_q8.d.data(),
                                     shared_q8.d.size() * sizeof(float))
            : static_cast<const void*>(shared_q8.d.data());
    Check(api_.clEnqueueWriteBuffer(queue_, selected_q8_qs_buffer, kClFalse,
                                    0, selected_q8.qs.size(),
                                    selected_q8_qs_data, 0, nullptr,
                                    nullptr),
          "clEnqueueWriteBuffer(selected+shared Q6 selected q8_qs)");
    Check(api_.clEnqueueWriteBuffer(queue_, selected_q8_d_buffer, kClFalse,
                                    0, selected_q8.d.size() * sizeof(float),
                                    selected_q8_d_data, 0, nullptr,
                                    nullptr),
          "clEnqueueWriteBuffer(selected+shared Q6 selected q8_d)");
    Check(api_.clEnqueueWriteBuffer(queue_, shared_q8_qs_buffer, kClFalse,
                                    0, shared_q8.qs.size(),
                                    shared_q8_qs_data, 0, nullptr,
                                    nullptr),
          "clEnqueueWriteBuffer(selected+shared Q6 shared q8_qs)");
    Check(api_.clEnqueueWriteBuffer(queue_, shared_q8_d_buffer, kClFalse,
                                    0, shared_q8.d.size() * sizeof(float),
                                    shared_q8_d_data, 0, nullptr, nullptr),
          "clEnqueueWriteBuffer(selected+shared Q6 shared q8_d)");
    const auto input_write_wall_ns =
        wall_ns(input_write_begin, std::chrono::steady_clock::now());

    run.timing = RunQ6KExpert8RowstripePlusSharedKernel(
        selected_buffers, shared.buffer, selected_q8_qs_buffer,
        selected_q8_d_buffer, shared_q8_qs_buffer, shared_q8_d_buffer,
        selected_out_buffer, shared_out_buffer, rows_per_expert,
        blocks_per_row, rows_per_tile, repeat, defer_finish);
    run.timing.input_setup_wall_ns += input_setup_wall_ns;
    run.timing.input_write_wall_ns += input_write_wall_ns;

    if (readback_selected_output) {
      const auto read_begin = std::chrono::steady_clock::now();
      Check(api_.clEnqueueReadBuffer(
                queue_, selected_out_buffer, kClTrue, 0,
                run.selected_output.size() * sizeof(float),
                run.selected_output.data(), 0, nullptr, nullptr),
            "clEnqueueReadBuffer(selected+shared selected Q6_K out)");
      run.timing.output_read_wall_ns +=
          wall_ns(read_begin, std::chrono::steady_clock::now());
    }
    if (readback_shared_output) {
      const auto read_begin = std::chrono::steady_clock::now();
      Check(api_.clEnqueueReadBuffer(
                queue_, shared_out_buffer, kClTrue, 0,
                run.shared_output.size() * sizeof(float),
                run.shared_output.data(), 0, nullptr, nullptr),
            "clEnqueueReadBuffer(selected+shared shared Q6_K out)");
      run.timing.output_read_wall_ns +=
          wall_ns(read_begin, std::chrono::steady_clock::now());
    }
    return run;
  }

  GpuFfnTailRun RunResidentRawQ6KExpert8PlusSharedToFfnTailAtomic(
      const std::vector<std::uint64_t>& selected_handles,
      std::uint64_t shared_handle,
      const GpuQ8KInputPlanes& selected_q8,
      const GpuQ8KInputPlanes& shared_q8,
      std::uint64_t shared_gate_matvec_handle,
      std::uint64_t attn_post_norm_handle,
      const std::vector<float>& weights_norm,
      std::uint64_t attn_residual_handle,
      std::uint64_t rows_per_expert,
      int repeat,
      bool readback_layer_output = true) {
    Require(selected_handles.size() == 8,
            "direct Q6 down-tail requires exactly 8 selected handles");
    Require(rows_per_expert > 0,
            "direct Q6 down-tail rows_per_expert must be nonzero");
    Require(repeat > 0, "direct Q6 down-tail repeat must be positive");
    Require(weights_norm.size() == selected_handles.size(),
            "direct Q6 down-tail router weight size mismatch");
    std::array<cl_mem, 8> selected_buffers{};
    std::uint64_t blocks_per_row = 0;
    std::uint64_t rows_per_tile = 0;
    for (std::size_t i = 0; i < selected_handles.size(); ++i) {
      const auto& resident = ResidentRawQ6KForHandle(selected_handles[i]);
      Require(resident.rows == rows_per_expert,
              "direct Q6 down-tail selected row count mismatch");
      Require(resident.rowstripe,
              "direct Q6 down-tail selected Q6_K requires rowstripe layout");
      if (i == 0) {
        blocks_per_row = resident.blocks_per_row;
        rows_per_tile = resident.rows_per_tile;
      } else {
        Require(resident.blocks_per_row == blocks_per_row,
                "direct Q6 down-tail selected block count mismatch");
        Require(resident.rows_per_tile == rows_per_tile,
                "direct Q6 down-tail selected rows_per_tile mismatch");
      }
      selected_buffers[i] = resident.buffer;
    }
    const auto& shared = ResidentRawQ6KForHandle(shared_handle);
    Require(shared.rows == rows_per_expert,
            "direct Q6 down-tail shared Q6_K row count mismatch");
    Require(shared.blocks_per_row == blocks_per_row,
            "direct Q6 down-tail shared Q6_K block count mismatch");
    Require(!shared.rowstripe,
            "direct Q6 down-tail shared Q6_K requires raw row layout");
    ValidateSelectedQ6KQ8InputPlanes(
        selected_q8, blocks_per_row, selected_handles.size(), repeat);
    ValidateQ6KQ8InputPlanes(shared_q8, blocks_per_row, repeat);

    const auto shared_gate =
        RunResidentF32MatvecFromInputHandle(
            shared_gate_matvec_handle, attn_post_norm_handle, repeat, false);
    const auto& shared_gate_buffer =
        ResidentF32BufferForHandle(shared_gate.output_handle);
    Require(shared_gate_buffer.values == 1,
            "direct Q6 down-tail shared gate scalar size mismatch");
    const auto& attn_residual = ResidentF32BufferForHandle(attn_residual_handle);
    Require(attn_residual.values == static_cast<std::size_t>(rows_per_expert),
            "direct Q6 down-tail residual size mismatch");

    GpuFfnTailRun run;
    run.layer_output_host_valid = readback_layer_output;
    if (readback_layer_output) {
      run.layer_output.assign(static_cast<std::size_t>(rows_per_expert), 0.0f);
    }

    cl_mem selected_q8_qs_buffer = EnsureScratchBuffer(
        selected_shared_q6_down_scratch_selected_q8_qs_, selected_q8.qs.size(),
        kClMemReadOnly, "direct Q6 down-tail selected q8_qs");
    cl_mem selected_q8_d_buffer = EnsureScratchBuffer(
        selected_shared_q6_down_scratch_selected_q8_d_,
        selected_q8.d.size() * sizeof(float), kClMemReadOnly,
        "direct Q6 down-tail selected q8_d");
    cl_mem shared_q8_qs_buffer = EnsureScratchBuffer(
        selected_shared_q6_down_scratch_shared_q8_qs_, shared_q8.qs.size(),
        kClMemReadOnly, "direct Q6 down-tail shared q8_qs");
    cl_mem shared_q8_d_buffer = EnsureScratchBuffer(
        selected_shared_q6_down_scratch_shared_q8_d_,
        shared_q8.d.size() * sizeof(float), kClMemReadOnly,
        "direct Q6 down-tail shared q8_d");
    cl_mem weights_buffer = EnsureScratchBuffer(
        ffn_tail_scratch_weights_, weights_norm.size() * sizeof(float),
        kClMemReadOnly, "direct Q6 down-tail weights");
    cl_mem layer_output_buffer = EnsureScratchBuffer(
        ffn_tail_scratch_layer_output_,
        static_cast<std::size_t>(rows_per_expert) * sizeof(float),
        kClMemReadWrite, "direct Q6 down-tail layer output");

    Check(api_.clEnqueueWriteBuffer(queue_, selected_q8_qs_buffer, kClFalse,
                                    0, selected_q8.qs.size(),
                                    selected_q8.qs.data(), 0, nullptr,
                                    nullptr),
          "clEnqueueWriteBuffer(direct Q6 down-tail selected q8_qs)");
    Check(api_.clEnqueueWriteBuffer(queue_, selected_q8_d_buffer, kClFalse,
                                    0, selected_q8.d.size() * sizeof(float),
                                    selected_q8.d.data(), 0, nullptr,
                                    nullptr),
          "clEnqueueWriteBuffer(direct Q6 down-tail selected q8_d)");
    Check(api_.clEnqueueWriteBuffer(queue_, shared_q8_qs_buffer, kClFalse,
                                    0, shared_q8.qs.size(),
                                    shared_q8.qs.data(), 0, nullptr,
                                    nullptr),
          "clEnqueueWriteBuffer(direct Q6 down-tail shared q8_qs)");
    Check(api_.clEnqueueWriteBuffer(queue_, shared_q8_d_buffer, kClFalse,
                                    0, shared_q8.d.size() * sizeof(float),
                                    shared_q8.d.data(), 0, nullptr, nullptr),
          "clEnqueueWriteBuffer(direct Q6 down-tail shared q8_d)");
    Check(api_.clEnqueueWriteBuffer(queue_, weights_buffer, kClFalse, 0,
                                    weights_norm.size() * sizeof(float),
                                    weights_norm.data(), 0, nullptr, nullptr),
          "clEnqueueWriteBuffer(direct Q6 down-tail weights)");

    const cl_uint rows_arg = static_cast<cl_uint>(rows_per_expert);
    const cl_uint blocks_arg = static_cast<cl_uint>(blocks_per_row);
    const cl_uint rows_per_tile_arg = static_cast<cl_uint>(rows_per_tile);
    cl_kernel init_kernel = kernel_ffn_tail_init_residual_bits_;
    cl_kernel direct_kernel =
        kernel_q6_matvec_rowstripe_expert8_plus_shared_tail_atomic_;
    Require(init_kernel != nullptr,
            "direct Q6 down-tail residual init kernel missing");
    Require(direct_kernel != nullptr,
            "direct Q6 down-tail contribution kernel missing");

    Check(api_.clSetKernelArg(init_kernel, 0, sizeof(attn_residual.buffer),
                              &attn_residual.buffer),
          "clSetKernelArg(direct Q6 down-tail init 0)");
    Check(api_.clSetKernelArg(init_kernel, 1, sizeof(rows_arg),
                              &rows_arg),
          "clSetKernelArg(direct Q6 down-tail init 1)");
    Check(api_.clSetKernelArg(init_kernel, 2, sizeof(layer_output_buffer),
                              &layer_output_buffer),
          "clSetKernelArg(direct Q6 down-tail init 2)");

    for (std::size_t i = 0; i < selected_buffers.size(); ++i) {
      Check(api_.clSetKernelArg(direct_kernel, static_cast<cl_uint>(i),
                                sizeof(selected_buffers[i]),
                                &selected_buffers[i]),
            "clSetKernelArg(direct Q6 down-tail selected raw)");
    }
    Check(api_.clSetKernelArg(direct_kernel, 8, sizeof(shared.buffer),
                              &shared.buffer),
          "clSetKernelArg(direct Q6 down-tail shared raw)");
    Check(api_.clSetKernelArg(direct_kernel, 9,
                              sizeof(selected_q8_qs_buffer),
                              &selected_q8_qs_buffer),
          "clSetKernelArg(direct Q6 down-tail selected q8_qs)");
    Check(api_.clSetKernelArg(direct_kernel, 10,
                              sizeof(selected_q8_d_buffer),
                              &selected_q8_d_buffer),
          "clSetKernelArg(direct Q6 down-tail selected q8_d)");
    Check(api_.clSetKernelArg(direct_kernel, 11,
                              sizeof(shared_q8_qs_buffer),
                              &shared_q8_qs_buffer),
          "clSetKernelArg(direct Q6 down-tail shared q8_qs)");
    Check(api_.clSetKernelArg(direct_kernel, 12,
                              sizeof(shared_q8_d_buffer),
                              &shared_q8_d_buffer),
          "clSetKernelArg(direct Q6 down-tail shared q8_d)");
    Check(api_.clSetKernelArg(direct_kernel, 13, sizeof(weights_buffer),
                              &weights_buffer),
          "clSetKernelArg(direct Q6 down-tail weights)");
    Check(api_.clSetKernelArg(direct_kernel, 14,
                              sizeof(shared_gate_buffer.buffer),
                              &shared_gate_buffer.buffer),
          "clSetKernelArg(direct Q6 down-tail shared gate)");
    Check(api_.clSetKernelArg(direct_kernel, 15, sizeof(rows_arg),
                              &rows_arg),
          "clSetKernelArg(direct Q6 down-tail rows)");
    Check(api_.clSetKernelArg(direct_kernel, 16, sizeof(blocks_arg),
                              &blocks_arg),
          "clSetKernelArg(direct Q6 down-tail blocks)");
    Check(api_.clSetKernelArg(direct_kernel, 17,
                              sizeof(rows_per_tile_arg),
                              &rows_per_tile_arg),
          "clSetKernelArg(direct Q6 down-tail rows_per_tile)");
    Check(api_.clSetKernelArg(direct_kernel, 18,
                              sizeof(layer_output_buffer),
                              &layer_output_buffer),
          "clSetKernelArg(direct Q6 down-tail output)");

    const std::size_t init_global = static_cast<std::size_t>(rows_per_expert);
    const std::size_t direct_global =
        static_cast<std::size_t>(rows_per_expert * 9);
    constexpr std::size_t kExpert8Q6RowstripeLocalSize = 64;
    const std::size_t* direct_local =
        (direct_global % kExpert8Q6RowstripeLocalSize == 0)
            ? &kExpert8Q6RowstripeLocalSize
            : nullptr;
    std::vector<double> init_times;
    std::vector<double> direct_times;
    std::vector<double> shell_sum_times;
    init_times.reserve(static_cast<std::size_t>(repeat));
    direct_times.reserve(static_cast<std::size_t>(repeat));
    shell_sum_times.reserve(static_cast<std::size_t>(repeat));
    for (int i = 0; i < repeat; ++i) {
      cl_event init_event = nullptr;
      cl_event direct_event = nullptr;
      Check(api_.clEnqueueNDRangeKernel(
                queue_, init_kernel, 1, nullptr, &init_global, nullptr, 0,
                nullptr, EventOut(&init_event)),
            "clEnqueueNDRangeKernel(direct Q6 down-tail init)");
      Check(api_.clEnqueueNDRangeKernel(
                queue_, direct_kernel, 1, nullptr, &direct_global,
                direct_local, 0, nullptr, EventOut(&direct_event)),
            "clEnqueueNDRangeKernel(direct Q6 down-tail contribution)");
      Check(api_.clFinish(queue_), "clFinish(direct Q6 down-tail)");
      const double init_us = EventUs(api_, init_event);
      const double direct_us = EventUs(api_, direct_event);
      init_times.push_back(init_us);
      direct_times.push_back(direct_us);
      shell_sum_times.push_back(
          shared_gate.timing.min_us + init_us + direct_us);
      ReleaseEvent(api_, &direct_event);
      ReleaseEvent(api_, &init_event);
    }

    if (readback_layer_output) {
      Check(api_.clEnqueueReadBuffer(queue_, layer_output_buffer, kClTrue, 0,
                                     run.layer_output.size() * sizeof(float),
                                     run.layer_output.data(), 0, nullptr,
                                     nullptr),
            "clEnqueueReadBuffer(direct Q6 down-tail layer output)");
    }
    ClearPendingHostUploadsAfterQueueDrain();
    run.layer_output_handle = RegisterF32BufferAlias(
        &ffn_tail_layer_output_alias_handle_, layer_output_buffer,
        static_cast<std::size_t>(rows_per_expert));

    const double init_min_us =
        *std::min_element(init_times.begin(), init_times.end());
    const double init_mean_us =
        std::accumulate(init_times.begin(), init_times.end(), 0.0) /
        static_cast<double>(init_times.size());
    const double direct_min_us =
        *std::min_element(direct_times.begin(), direct_times.end());
    const double direct_mean_us =
        std::accumulate(direct_times.begin(), direct_times.end(), 0.0) /
        static_cast<double>(direct_times.size());
    run.timing.weighted_min_us = direct_min_us;
    run.timing.weighted_mean_us = direct_mean_us;
    run.timing.shared_gate_matvec_min_us = shared_gate.timing.min_us;
    run.timing.shared_gate_matvec_mean_us = shared_gate.timing.mean_us;
    run.timing.shared_gate_apply_min_us = 0.0;
    run.timing.shared_gate_apply_mean_us = 0.0;
    run.timing.ffn_output_add_min_us = 0.0;
    run.timing.ffn_output_add_mean_us = 0.0;
    run.timing.residual_add_min_us = init_min_us;
    run.timing.residual_add_mean_us = init_mean_us;
    run.timing.shell_sum_min_us =
        *std::min_element(shell_sum_times.begin(), shell_sum_times.end());
    run.timing.shell_sum_mean_us =
        std::accumulate(shell_sum_times.begin(), shell_sum_times.end(), 0.0) /
        static_cast<double>(shell_sum_times.size());
    run.timing.hidden_global_work_items = direct_global;
    run.timing.shared_gate_global_work_items =
        shared_gate.timing.global_work_items;
    return run;
  }

  void ClearResidentRawQ6K() {
    for (auto& item : resident_raw_q6k_) {
      ReleaseMem(api_, &item.second.buffer);
    }
    resident_raw_q6k_.clear();
  }

  GpuQ4X8ConvHandoffRun RunResidentPackedQ4X8ThenConv(
      std::uint64_t handle,
      const std::vector<std::int8_t>& q8_qs,
      const std::vector<std::int16_t>& q8_bsums,
      const std::vector<float>& q8_d,
      const std::vector<float>& conv_weights,
      const std::vector<float>& conv_state,
      std::uint64_t conv_kernel_size,
      int repeat,
      GpuQ4X8KernelVariant variant) {
    const auto& resident = ResidentPackedQ4X8ForHandle(handle);
    ValidateQ8InputPlanes(q8_qs, q8_bsums, q8_d, resident.blocks_per_row, repeat);
    ValidateConvInputs(conv_weights, conv_state, resident.rows, conv_kernel_size);
    GpuQ4X8ConvHandoffRun run;
    run.qkv_mixed.assign(static_cast<std::size_t>(resident.rows), 0.0f);
    run.conv_output_raw.assign(static_cast<std::size_t>(resident.rows), 0.0f);
    run.conv_state.assign(static_cast<std::size_t>(resident.rows * (conv_kernel_size - 1)), 0.0f);
    cl_mem q8_qs_buffer = nullptr, q8_bsums_buffer = nullptr, q8_d_buffer = nullptr;
    cl_mem qkv_buffer = nullptr, conv_weights_buffer = nullptr;
    cl_mem conv_state_buffer = nullptr, conv_output_buffer = nullptr, next_state_buffer = nullptr;
    try {
      CreateRunBuffersWithoutPacked(q8_qs, q8_bsums, q8_d, run.qkv_mixed,
                                    &q8_qs_buffer, &q8_bsums_buffer,
                                    &q8_d_buffer, &qkv_buffer);
      CreateConvBuffers(conv_weights, conv_state, run.conv_output_raw, run.conv_state,
                        &conv_weights_buffer, &conv_state_buffer, &conv_output_buffer,
                        &next_state_buffer);
      const std::uint64_t row_groups = resident.rows / kRowsInterleaved;
      const std::uint64_t matvec_global =
          variant == GpuQ4X8KernelVariant::kGroup8Serial ? row_groups : resident.rows;
      run.timing = RunHandoffKernels(variant, resident.buffer, q8_qs_buffer,
                                     q8_bsums_buffer, q8_d_buffer, qkv_buffer,
                                     conv_weights_buffer, conv_state_buffer,
                                     conv_output_buffer, next_state_buffer,
                                     resident.blocks_per_row, row_groups,
                                     matvec_global, resident.rows,
                                     conv_kernel_size, repeat);
      Check(api_.clEnqueueReadBuffer(queue_, qkv_buffer, kClTrue, 0,
                                     run.qkv_mixed.size() * sizeof(float),
                                     run.qkv_mixed.data(), 0, nullptr, nullptr),
            "clEnqueueReadBuffer(resident qkv)");
      Check(api_.clEnqueueReadBuffer(queue_, conv_output_buffer, kClTrue, 0,
                                     run.conv_output_raw.size() * sizeof(float),
                                     run.conv_output_raw.data(), 0, nullptr, nullptr),
            "clEnqueueReadBuffer(resident conv output)");
      Check(api_.clEnqueueReadBuffer(queue_, next_state_buffer, kClTrue, 0,
                                     run.conv_state.size() * sizeof(float),
                                     run.conv_state.data(), 0, nullptr, nullptr),
            "clEnqueueReadBuffer(resident next state)");
    } catch (...) {
      ReleaseMem(api_, &next_state_buffer);
      ReleaseMem(api_, &conv_output_buffer);
      ReleaseMem(api_, &conv_state_buffer);
      ReleaseMem(api_, &conv_weights_buffer);
      ReleaseMem(api_, &qkv_buffer);
      ReleaseMem(api_, &q8_d_buffer);
      ReleaseMem(api_, &q8_bsums_buffer);
      ReleaseMem(api_, &q8_qs_buffer);
      throw;
    }
    ReleaseMem(api_, &next_state_buffer);
    ReleaseMem(api_, &conv_output_buffer);
    ReleaseMem(api_, &conv_state_buffer);
    ReleaseMem(api_, &conv_weights_buffer);
    ReleaseMem(api_, &qkv_buffer);
    ReleaseMem(api_, &q8_d_buffer);
    ReleaseMem(api_, &q8_bsums_buffer);
    ReleaseMem(api_, &q8_qs_buffer);
    return run;
  }

  GpuQ4X8SwiGluHandoffRun RunResidentPackedQ4X8ThenSwiGlu(
      std::uint64_t handle,
      const std::vector<std::int8_t>& q8_qs,
      const std::vector<std::int16_t>& q8_bsums,
      const std::vector<float>& q8_d,
      std::uint64_t intermediate_size,
      const std::vector<std::uint32_t>& source_expert_by_output,
      int repeat,
      GpuQ4X8KernelVariant variant) {
    const auto& resident = ResidentPackedQ4X8ForHandle(handle);
    Require(intermediate_size > 0, "SwiGLU handoff intermediate size must be nonzero");
    Require(!source_expert_by_output.empty(),
            "SwiGLU handoff source expert map must be nonempty");
    const std::uint64_t expert_count = source_expert_by_output.size();
    Require(resident.rows == intermediate_size * expert_count * 2,
            "SwiGLU handoff resident row count mismatch");
    for (const auto source : source_expert_by_output) {
      Require(source < expert_count, "SwiGLU handoff source expert index out of range");
    }
    ValidateQ8InputPlanes(q8_qs, q8_bsums, q8_d, resident.blocks_per_row, repeat);

    GpuQ4X8SwiGluHandoffRun run;
    run.swiglu.assign(static_cast<std::size_t>(intermediate_size * expert_count), 0.0f);
    cl_mem q8_qs_buffer = EnsureScratchBuffer(
        swiglu_handoff_scratch_q8_qs_, q8_qs.size(), kClMemReadOnly,
        "SwiGLU handoff q8_qs");
    cl_mem q8_bsums_buffer = EnsureScratchBuffer(
        swiglu_handoff_scratch_q8_bsums_,
        q8_bsums.size() * sizeof(std::int16_t), kClMemReadOnly,
        "SwiGLU handoff q8_bsums");
    cl_mem q8_d_buffer = EnsureScratchBuffer(
        swiglu_handoff_scratch_q8_d_, q8_d.size() * sizeof(float),
        kClMemReadOnly, "SwiGLU handoff q8_d");
    cl_mem gate_up_buffer = EnsureScratchBuffer(
        swiglu_handoff_scratch_gate_up_,
        static_cast<std::size_t>(resident.rows) * sizeof(float),
        kClMemReadWrite, "SwiGLU handoff gate_up");
    cl_mem swiglu_buffer = EnsureScratchBuffer(
        swiglu_handoff_scratch_swiglu_, run.swiglu.size() * sizeof(float),
        kClMemWriteOnly, "SwiGLU handoff output");
    bool identity_source_order = true;
    for (std::size_t i = 0; i < source_expert_by_output.size(); ++i) {
      if (source_expert_by_output[i] != i) {
        identity_source_order = false;
        break;
      }
    }
    Check(api_.clEnqueueWriteBuffer(queue_, q8_qs_buffer, kClFalse, 0,
                                    q8_qs.size(), q8_qs.data(), 0, nullptr,
                                    nullptr),
          "clEnqueueWriteBuffer(SwiGLU handoff q8_qs)");
    Check(api_.clEnqueueWriteBuffer(
              queue_, q8_bsums_buffer, kClFalse, 0,
              q8_bsums.size() * sizeof(std::int16_t), q8_bsums.data(), 0,
              nullptr, nullptr),
          "clEnqueueWriteBuffer(SwiGLU handoff q8_bsums)");
    Check(api_.clEnqueueWriteBuffer(queue_, q8_d_buffer, kClFalse, 0,
                                    q8_d.size() * sizeof(float),
                                    q8_d.data(), 0, nullptr, nullptr),
          "clEnqueueWriteBuffer(SwiGLU handoff q8_d)");

    const std::uint64_t row_groups = resident.rows / kRowsInterleaved;
    const std::uint64_t matvec_global =
        variant == GpuQ4X8KernelVariant::kGroup8Serial ? row_groups : resident.rows;
    const bool read_in_kernel = identity_source_order && repeat == 1;
    if (identity_source_order) {
      run.timing = RunMatvecThenSwiGluKernel(
          variant, resident.buffer, q8_qs_buffer, q8_bsums_buffer, q8_d_buffer,
          gate_up_buffer, swiglu_buffer, resident.blocks_per_row, row_groups,
          matvec_global, intermediate_size, expert_count, repeat,
          read_in_kernel ? run.swiglu.data() : nullptr,
          read_in_kernel ? run.swiglu.size() * sizeof(float) : 0);
    } else {
      run.timing.matvec = RunKernel(
          variant, resident.buffer, q8_qs_buffer, q8_bsums_buffer, q8_d_buffer,
          gate_up_buffer, resident.blocks_per_row, row_groups, matvec_global,
          repeat);
      cl_mem source_map_buffer = EnsureScratchBuffer(
          swiglu_handoff_scratch_source_map_,
          source_expert_by_output.size() * sizeof(std::uint32_t),
          kClMemReadOnly, "SwiGLU handoff source map");
      Check(api_.clEnqueueWriteBuffer(
                queue_, source_map_buffer, kClTrue, 0,
                source_expert_by_output.size() * sizeof(std::uint32_t),
                source_expert_by_output.data(), 0, nullptr, nullptr),
            "clEnqueueWriteBuffer(SwiGLU source map)");
      const auto swiglu_timing = RunSwiGluReorderKernel(
          gate_up_buffer, source_map_buffer, swiglu_buffer, intermediate_size,
          expert_count, repeat);
      run.timing.swiglu_min_us = swiglu_timing.min_us;
      run.timing.swiglu_mean_us = swiglu_timing.mean_us;
      run.timing.swiglu_global_work_items = swiglu_timing.global_work_items;
      run.timing.shell_sum_min_us =
          run.timing.matvec.min_us + run.timing.swiglu_min_us;
      run.timing.shell_sum_mean_us =
          run.timing.matvec.mean_us + run.timing.swiglu_mean_us;
    }
    if (!read_in_kernel) {
      Check(api_.clEnqueueReadBuffer(queue_, swiglu_buffer, kClTrue, 0,
                                     run.swiglu.size() * sizeof(float),
                                     run.swiglu.data(), 0, nullptr, nullptr),
            "clEnqueueReadBuffer(SwiGLU handoff output)");
    }
    return run;
  }

  GpuQ4X8SwiGluHandoffRun
  RunResidentPackedQ4X8ThenSwiGluWithLastExpert8Q8(
      std::uint64_t handle,
      std::uint64_t intermediate_size,
      const std::vector<std::uint32_t>& source_expert_by_output,
      int repeat,
      GpuQ4X8KernelVariant variant) {
    const auto& resident = ResidentPackedQ4X8ForHandle(handle);
    Require(intermediate_size > 0,
            "SwiGLU handoff intermediate size must be nonzero");
    Require(!source_expert_by_output.empty(),
            "SwiGLU handoff source expert map must be nonempty");
    const std::uint64_t expert_count = source_expert_by_output.size();
    Require(resident.rows == intermediate_size * expert_count * 2,
            "SwiGLU handoff resident row count mismatch");
    for (const auto source : source_expert_by_output) {
      Require(source < expert_count,
              "SwiGLU handoff source expert index out of range");
    }
    Require(repeat > 0, "repeat must be positive");
    Require(expert8_swiglu_scratch_q8_qs_.mem != nullptr &&
                expert8_swiglu_scratch_q8_qs_.bytes ==
                    resident.blocks_per_row * kQ8QsPerBlock,
            "last expert8 Q8 qs scratch is unavailable for reuse");
    Require(expert8_swiglu_scratch_q8_bsums_.mem != nullptr &&
                expert8_swiglu_scratch_q8_bsums_.bytes ==
                    resident.blocks_per_row * kQ8BsumsPerBlock *
                        sizeof(std::int16_t),
            "last expert8 Q8 bsum scratch is unavailable for reuse");
    Require(expert8_swiglu_scratch_q8_d_.mem != nullptr &&
                expert8_swiglu_scratch_q8_d_.bytes ==
                    resident.blocks_per_row * sizeof(float),
            "last expert8 Q8 scale scratch is unavailable for reuse");

    GpuQ4X8SwiGluHandoffRun run;
    run.swiglu.assign(
        static_cast<std::size_t>(intermediate_size * expert_count), 0.0f);
    cl_mem gate_up_buffer = EnsureScratchBuffer(
        swiglu_handoff_scratch_gate_up_,
        static_cast<std::size_t>(resident.rows) * sizeof(float),
        kClMemReadWrite, "SwiGLU handoff gate_up");
    cl_mem swiglu_buffer = EnsureScratchBuffer(
        swiglu_handoff_scratch_swiglu_, run.swiglu.size() * sizeof(float),
        kClMemWriteOnly, "SwiGLU handoff output");
    bool identity_source_order = true;
    for (std::size_t i = 0; i < source_expert_by_output.size(); ++i) {
      if (source_expert_by_output[i] != i) {
        identity_source_order = false;
        break;
      }
    }

    const std::uint64_t row_groups = resident.rows / kRowsInterleaved;
    const std::uint64_t matvec_global =
        variant == GpuQ4X8KernelVariant::kGroup8Serial ? row_groups
                                                       : resident.rows;
    const bool read_in_kernel = identity_source_order && repeat == 1;
    if (identity_source_order) {
      run.timing = RunMatvecThenSwiGluKernel(
          variant, resident.buffer, expert8_swiglu_scratch_q8_qs_.mem,
          expert8_swiglu_scratch_q8_bsums_.mem,
          expert8_swiglu_scratch_q8_d_.mem, gate_up_buffer, swiglu_buffer,
          resident.blocks_per_row, row_groups, matvec_global,
          intermediate_size, expert_count, repeat,
          read_in_kernel ? run.swiglu.data() : nullptr,
          read_in_kernel ? run.swiglu.size() * sizeof(float) : 0);
    } else {
      run.timing.matvec = RunKernel(
          variant, resident.buffer, expert8_swiglu_scratch_q8_qs_.mem,
          expert8_swiglu_scratch_q8_bsums_.mem,
          expert8_swiglu_scratch_q8_d_.mem, gate_up_buffer,
          resident.blocks_per_row, row_groups, matvec_global, repeat);
      cl_mem source_map_buffer = EnsureScratchBuffer(
          swiglu_handoff_scratch_source_map_,
          source_expert_by_output.size() * sizeof(std::uint32_t),
          kClMemReadOnly, "SwiGLU handoff source map");
      Check(api_.clEnqueueWriteBuffer(
                queue_, source_map_buffer, kClTrue, 0,
                source_expert_by_output.size() * sizeof(std::uint32_t),
                source_expert_by_output.data(), 0, nullptr, nullptr),
            "clEnqueueWriteBuffer(SwiGLU source map)");
      const auto swiglu_timing = RunSwiGluReorderKernel(
          gate_up_buffer, source_map_buffer, swiglu_buffer,
          intermediate_size, expert_count, repeat);
      run.timing.swiglu_min_us = swiglu_timing.min_us;
      run.timing.swiglu_mean_us = swiglu_timing.mean_us;
      run.timing.swiglu_global_work_items = swiglu_timing.global_work_items;
      run.timing.shell_sum_min_us =
          run.timing.matvec.min_us + run.timing.swiglu_min_us;
      run.timing.shell_sum_mean_us =
          run.timing.matvec.mean_us + run.timing.swiglu_mean_us;
    }
    if (!read_in_kernel) {
      Check(api_.clEnqueueReadBuffer(queue_, swiglu_buffer, kClTrue, 0,
                                     run.swiglu.size() * sizeof(float),
                                     run.swiglu.data(), 0, nullptr, nullptr),
            "clEnqueueReadBuffer(SwiGLU handoff output)");
    }
    return run;
  }

  GpuQ4X8SwiGluHandoffRun RunResidentPackedQ4X8Expert8ThenSwiGlu(
      const std::vector<std::uint64_t>& handles,
      const std::vector<std::int8_t>& q8_qs,
      const std::vector<std::int16_t>& q8_bsums,
      const std::vector<float>& q8_d,
      std::uint64_t intermediate_size,
      int repeat,
      GpuQ4X8KernelVariant variant) {
    Require(handles.size() == 8,
            "expert8 SwiGLU handoff requires exactly 8 handles");
    Require(variant == GpuQ4X8KernelVariant::kRowlaneParallel,
            "expert8 SwiGLU handoff requires rowlane variant");
    Require(intermediate_size > 0,
            "expert8 SwiGLU handoff intermediate size must be nonzero");

    std::array<cl_mem, 8> packed_buffers{};
    std::uint64_t rows_per_expert = 0;
    std::uint64_t blocks_per_row = 0;
    for (std::size_t i = 0; i < handles.size(); ++i) {
      const auto& resident = ResidentPackedQ4X8ForHandle(handles[i]);
      if (i == 0) {
        rows_per_expert = resident.rows;
        blocks_per_row = resident.blocks_per_row;
      } else {
        Require(resident.rows == rows_per_expert,
                "expert8 SwiGLU handoff row count mismatch");
        Require(resident.blocks_per_row == blocks_per_row,
                "expert8 SwiGLU handoff block count mismatch");
      }
      packed_buffers[i] = resident.buffer;
    }
    Require(rows_per_expert == intermediate_size * 2,
            "expert8 SwiGLU handoff rows per expert mismatch");
    Require(rows_per_expert % kRowsInterleaved == 0,
            "expert8 SwiGLU handoff rows must be divisible by 8");
    ValidateQ8InputPlanes(q8_qs, q8_bsums, q8_d, blocks_per_row, repeat);

    GpuQ4X8SwiGluHandoffRun run;
    const std::uint64_t expert_count = handles.size();
    run.swiglu.assign(
        static_cast<std::size_t>(intermediate_size * expert_count), 0.0f);
    try {
      cl_mem q8_qs_buffer = EnsureScratchBuffer(
          expert8_swiglu_scratch_q8_qs_, q8_qs.size(), kClMemReadOnly,
          "expert8 SwiGLU q8_qs");
      cl_mem q8_bsums_buffer = EnsureScratchBuffer(
          expert8_swiglu_scratch_q8_bsums_,
          q8_bsums.size() * sizeof(std::int16_t), kClMemReadOnly,
          "expert8 SwiGLU q8_bsums");
      cl_mem q8_d_buffer = EnsureScratchBuffer(
          expert8_swiglu_scratch_q8_d_, q8_d.size() * sizeof(float),
          kClMemReadOnly, "expert8 SwiGLU q8_d");
      cl_mem gate_up_buffer = EnsureScratchBuffer(
          expert8_swiglu_scratch_gate_up_,
          static_cast<std::size_t>(rows_per_expert * expert_count) *
              sizeof(float),
          kClMemReadWrite, "expert8 SwiGLU gate_up");
      cl_mem swiglu_buffer = EnsureScratchBuffer(
          expert8_swiglu_scratch_swiglu_, run.swiglu.size() * sizeof(float),
          kClMemWriteOnly, "expert8 SwiGLU output");
      Check(api_.clEnqueueWriteBuffer(queue_, q8_qs_buffer, kClFalse, 0,
                                      q8_qs.size(), q8_qs.data(), 0, nullptr,
                                      nullptr),
            "clEnqueueWriteBuffer(expert8 SwiGLU q8_qs)");
      Check(api_.clEnqueueWriteBuffer(
                queue_, q8_bsums_buffer, kClFalse, 0,
                q8_bsums.size() * sizeof(std::int16_t), q8_bsums.data(), 0,
                nullptr, nullptr),
            "clEnqueueWriteBuffer(expert8 SwiGLU q8_bsums)");
      Check(api_.clEnqueueWriteBuffer(queue_, q8_d_buffer, kClFalse, 0,
                                      q8_d.size() * sizeof(float),
                                      q8_d.data(), 0, nullptr, nullptr),
            "clEnqueueWriteBuffer(expert8 SwiGLU q8_d)");

      const std::uint64_t row_groups_per_expert =
          rows_per_expert / kRowsInterleaved;
      const std::uint64_t matvec_global = rows_per_expert * expert_count;
      const bool use_localq8_matvec = blocks_per_row == 8 &&
          matvec_global % 64 == 0;
      cl_kernel matvec_kernel =
          use_localq8_matvec ? kernel_rowlane_expert8_localq8_
                             : kernel_rowlane_expert8_;
      const std::size_t kExpert8Q4LocalQ8Size = 64;
      const std::size_t* matvec_local =
          use_localq8_matvec ? &kExpert8Q4LocalQ8Size : nullptr;
      const char* matvec_kernel_name =
          use_localq8_matvec
              ? "q4k_x8_matvec_rowlane_expert8_localq8"
              : "q4k_x8_matvec_rowlane_expert8";
      const cl_uint blocks_arg = static_cast<cl_uint>(blocks_per_row);
      const cl_uint row_groups_arg =
          static_cast<cl_uint>(row_groups_per_expert);
      for (std::size_t i = 0; i < packed_buffers.size(); ++i) {
        Check(api_.clSetKernelArg(matvec_kernel,
                                  static_cast<cl_uint>(i),
                                  sizeof(packed_buffers[i]),
                                  &packed_buffers[i]),
              "clSetKernelArg(expert8 fused packed)");
      }
      Check(api_.clSetKernelArg(matvec_kernel, 8,
                                sizeof(q8_qs_buffer), &q8_qs_buffer),
            "clSetKernelArg(expert8 fused q8_qs)");
      Check(api_.clSetKernelArg(matvec_kernel, 9,
                                sizeof(q8_bsums_buffer), &q8_bsums_buffer),
            "clSetKernelArg(expert8 fused q8_bsums)");
      Check(api_.clSetKernelArg(matvec_kernel, 10,
                                sizeof(q8_d_buffer), &q8_d_buffer),
            "clSetKernelArg(expert8 fused q8_d)");
      Check(api_.clSetKernelArg(matvec_kernel, 11,
                                sizeof(blocks_arg), &blocks_arg),
            "clSetKernelArg(expert8 fused blocks)");
      Check(api_.clSetKernelArg(matvec_kernel, 12,
                                sizeof(row_groups_arg), &row_groups_arg),
            "clSetKernelArg(expert8 fused row_groups)");
      Check(api_.clSetKernelArg(matvec_kernel, 13,
                                sizeof(gate_up_buffer), &gate_up_buffer),
            "clSetKernelArg(expert8 fused out)");

      const cl_uint intermediate_arg =
          static_cast<cl_uint>(intermediate_size);
      const cl_uint expert_arg = static_cast<cl_uint>(expert_count);
      Check(api_.clSetKernelArg(kernel_ffn_swiglu_, 0,
                                sizeof(gate_up_buffer), &gate_up_buffer),
            "clSetKernelArg(expert8 fused SwiGLU 0)");
      Check(api_.clSetKernelArg(kernel_ffn_swiglu_, 1,
                                sizeof(intermediate_arg), &intermediate_arg),
            "clSetKernelArg(expert8 fused SwiGLU 1)");
      Check(api_.clSetKernelArg(kernel_ffn_swiglu_, 2, sizeof(expert_arg),
                                &expert_arg),
            "clSetKernelArg(expert8 fused SwiGLU 2)");
      Check(api_.clSetKernelArg(kernel_ffn_swiglu_, 3,
                                sizeof(swiglu_buffer), &swiglu_buffer),
            "clSetKernelArg(expert8 fused SwiGLU 3)");

      const std::size_t matvec_global_size =
          static_cast<std::size_t>(matvec_global);
      const std::size_t swiglu_global_size = run.swiglu.size();
      std::vector<double> matvec_times;
      std::vector<double> swiglu_times;
      matvec_times.reserve(static_cast<std::size_t>(repeat));
      swiglu_times.reserve(static_cast<std::size_t>(repeat));
      const bool read_in_kernel = repeat == 1;
      for (int i = 0; i < repeat; ++i) {
        cl_event matvec_event = nullptr;
        cl_event swiglu_event = nullptr;
        Check(api_.clEnqueueNDRangeKernel(
                  queue_, matvec_kernel, 1, nullptr, &matvec_global_size,
                  matvec_local, 0, nullptr, EventOut(&matvec_event)),
              std::string("clEnqueueNDRangeKernel(") + matvec_kernel_name + ")");
        Check(api_.clEnqueueNDRangeKernel(
                  queue_, kernel_ffn_swiglu_, 1, nullptr, &swiglu_global_size,
                  nullptr, 0, nullptr, EventOut(&swiglu_event)),
              "clEnqueueNDRangeKernel(expert8 fused SwiGLU)");
        if (read_in_kernel) {
          Check(api_.clEnqueueReadBuffer(queue_, swiglu_buffer, kClTrue, 0,
                                         run.swiglu.size() * sizeof(float),
                                         run.swiglu.data(), 0, nullptr,
                                         nullptr),
                "clEnqueueReadBuffer(expert8 SwiGLU output)");
        } else {
          Check(api_.clFinish(queue_), "clFinish(expert8 fused SwiGLU)");
        }
        matvec_times.push_back(EventUs(api_, matvec_event));
        swiglu_times.push_back(EventUs(api_, swiglu_event));
        ReleaseEvent(api_, &matvec_event);
        ReleaseEvent(api_, &swiglu_event);
      }
      ClearPendingHostUploadsAfterQueueDrain();
      run.timing.matvec.min_us =
          *std::min_element(matvec_times.begin(), matvec_times.end());
      run.timing.matvec.mean_us =
          std::accumulate(matvec_times.begin(), matvec_times.end(), 0.0) /
          static_cast<double>(matvec_times.size());
      run.timing.matvec.effective_packed_gb_s =
          static_cast<double>(matvec_global / kRowsInterleaved *
                              blocks_per_row * kQ4Kx8BlockBytes) /
          (run.timing.matvec.min_us / 1e6) / 1e9;
      run.timing.matvec.global_work_items = matvec_global;
      run.timing.matvec.rows_per_work_item = 1;
      run.timing.swiglu_min_us =
          *std::min_element(swiglu_times.begin(), swiglu_times.end());
      run.timing.swiglu_mean_us =
          std::accumulate(swiglu_times.begin(), swiglu_times.end(), 0.0) /
          static_cast<double>(swiglu_times.size());
      run.timing.swiglu_global_work_items = run.swiglu.size();
      run.timing.shell_sum_min_us =
          run.timing.matvec.min_us + run.timing.swiglu_min_us;
      run.timing.shell_sum_mean_us =
          run.timing.matvec.mean_us + run.timing.swiglu_mean_us;
      if (!read_in_kernel) {
        Check(api_.clEnqueueReadBuffer(queue_, swiglu_buffer, kClTrue, 0,
                                       run.swiglu.size() * sizeof(float),
                                       run.swiglu.data(), 0, nullptr, nullptr),
              "clEnqueueReadBuffer(expert8 SwiGLU output)");
      }
    } catch (...) {
      throw;
    }
    return run;
  }

  GpuQ4X8SwiGluHandoffRun
  RunResidentPackedQ4X8Expert8PlusSharedThenSwiGlu(
      const std::vector<std::uint64_t>& selected_handles,
      std::uint64_t shared_handle,
      const std::vector<std::int8_t>& q8_qs,
      const std::vector<std::int16_t>& q8_bsums,
      const std::vector<float>& q8_d,
      std::uint64_t intermediate_size,
      int repeat,
      GpuQ4X8KernelVariant variant) {
    Require(selected_handles.size() == 8,
            "expert8+shared SwiGLU handoff requires exactly 8 selected handles");
    Require(shared_handle != 0,
            "expert8+shared SwiGLU handoff requires a shared handle");
    Require(variant == GpuQ4X8KernelVariant::kRowlaneParallel,
            "expert8+shared SwiGLU handoff requires rowlane variant");
    Require(intermediate_size > 0,
            "expert8+shared SwiGLU handoff intermediate size must be nonzero");

    std::array<cl_mem, 8> selected_buffers{};
    std::uint64_t rows_per_expert = 0;
    std::uint64_t blocks_per_row = 0;
    for (std::size_t i = 0; i < selected_handles.size(); ++i) {
      const auto& resident = ResidentPackedQ4X8ForHandle(selected_handles[i]);
      if (i == 0) {
        rows_per_expert = resident.rows;
        blocks_per_row = resident.blocks_per_row;
      } else {
        Require(resident.rows == rows_per_expert,
                "expert8+shared selected row count mismatch");
        Require(resident.blocks_per_row == blocks_per_row,
                "expert8+shared selected block count mismatch");
      }
      selected_buffers[i] = resident.buffer;
    }
    const auto& shared = ResidentPackedQ4X8ForHandle(shared_handle);
    Require(shared.rows == rows_per_expert,
            "expert8+shared shared row count mismatch");
    Require(shared.blocks_per_row == blocks_per_row,
            "expert8+shared shared block count mismatch");
    Require(rows_per_expert == intermediate_size * 2,
            "expert8+shared rows per expert mismatch");
    Require(rows_per_expert % kRowsInterleaved == 0,
            "expert8+shared rows must be divisible by 8");
    Require(blocks_per_row == 8,
            "expert8+shared local-Q8 handoff requires BPR8");
    ValidateQ8InputPlanes(q8_qs, q8_bsums, q8_d, blocks_per_row, repeat);

    GpuQ4X8SwiGluHandoffRun run;
    constexpr std::uint64_t kExpertCountPlusShared = 9;
    const std::uint64_t expert_count = kExpertCountPlusShared;
    run.swiglu.assign(
        static_cast<std::size_t>(intermediate_size * expert_count), 0.0f);
    cl_mem q8_qs_buffer = EnsureScratchBuffer(
        expert8_swiglu_scratch_q8_qs_, q8_qs.size(), kClMemReadOnly,
        "expert8+shared SwiGLU q8_qs");
    cl_mem q8_bsums_buffer = EnsureScratchBuffer(
        expert8_swiglu_scratch_q8_bsums_,
        q8_bsums.size() * sizeof(std::int16_t), kClMemReadOnly,
        "expert8+shared SwiGLU q8_bsums");
    cl_mem q8_d_buffer = EnsureScratchBuffer(
        expert8_swiglu_scratch_q8_d_, q8_d.size() * sizeof(float),
        kClMemReadOnly, "expert8+shared SwiGLU q8_d");
    cl_mem gate_up_buffer = EnsureScratchBuffer(
        expert8_swiglu_scratch_gate_up_,
        static_cast<std::size_t>(rows_per_expert * expert_count) *
            sizeof(float),
        kClMemReadWrite, "expert8+shared SwiGLU gate_up");
    cl_mem swiglu_buffer = EnsureScratchBuffer(
        expert8_swiglu_scratch_swiglu_, run.swiglu.size() * sizeof(float),
        kClMemWriteOnly, "expert8+shared SwiGLU output");
    Check(api_.clEnqueueWriteBuffer(queue_, q8_qs_buffer, kClFalse, 0,
                                    q8_qs.size(), q8_qs.data(), 0, nullptr,
                                    nullptr),
          "clEnqueueWriteBuffer(expert8+shared SwiGLU q8_qs)");
    Check(api_.clEnqueueWriteBuffer(
              queue_, q8_bsums_buffer, kClFalse, 0,
              q8_bsums.size() * sizeof(std::int16_t), q8_bsums.data(), 0,
              nullptr, nullptr),
          "clEnqueueWriteBuffer(expert8+shared SwiGLU q8_bsums)");
    Check(api_.clEnqueueWriteBuffer(queue_, q8_d_buffer, kClFalse, 0,
                                    q8_d.size() * sizeof(float),
                                    q8_d.data(), 0, nullptr, nullptr),
          "clEnqueueWriteBuffer(expert8+shared SwiGLU q8_d)");

    const std::uint64_t row_groups_per_expert =
        rows_per_expert / kRowsInterleaved;
    const std::uint64_t matvec_global = rows_per_expert * expert_count;
    constexpr std::size_t kLocalQ8Size = 64;
    const std::size_t* matvec_local = &kLocalQ8Size;
    cl_kernel matvec_kernel = kernel_rowlane_expert8_plus_shared_localq8_;
    const cl_uint blocks_arg = static_cast<cl_uint>(blocks_per_row);
    const cl_uint row_groups_arg =
        static_cast<cl_uint>(row_groups_per_expert);
    for (std::size_t i = 0; i < selected_buffers.size(); ++i) {
      Check(api_.clSetKernelArg(matvec_kernel, static_cast<cl_uint>(i),
                                sizeof(selected_buffers[i]),
                                &selected_buffers[i]),
            "clSetKernelArg(expert8+shared selected packed)");
    }
    Check(api_.clSetKernelArg(matvec_kernel, 8,
                              sizeof(shared.buffer), &shared.buffer),
          "clSetKernelArg(expert8+shared shared packed)");
    Check(api_.clSetKernelArg(matvec_kernel, 9,
                              sizeof(q8_qs_buffer), &q8_qs_buffer),
          "clSetKernelArg(expert8+shared q8_qs)");
    Check(api_.clSetKernelArg(matvec_kernel, 10,
                              sizeof(q8_bsums_buffer), &q8_bsums_buffer),
          "clSetKernelArg(expert8+shared q8_bsums)");
    Check(api_.clSetKernelArg(matvec_kernel, 11,
                              sizeof(q8_d_buffer), &q8_d_buffer),
          "clSetKernelArg(expert8+shared q8_d)");
    Check(api_.clSetKernelArg(matvec_kernel, 12,
                              sizeof(blocks_arg), &blocks_arg),
          "clSetKernelArg(expert8+shared blocks)");
    Check(api_.clSetKernelArg(matvec_kernel, 13,
                              sizeof(row_groups_arg), &row_groups_arg),
          "clSetKernelArg(expert8+shared row_groups)");
    Check(api_.clSetKernelArg(matvec_kernel, 14,
                              sizeof(gate_up_buffer), &gate_up_buffer),
          "clSetKernelArg(expert8+shared out)");

    const cl_uint intermediate_arg = static_cast<cl_uint>(intermediate_size);
    const cl_uint expert_arg = static_cast<cl_uint>(expert_count);
    Check(api_.clSetKernelArg(kernel_ffn_swiglu_, 0,
                              sizeof(gate_up_buffer), &gate_up_buffer),
          "clSetKernelArg(expert8+shared SwiGLU 0)");
    Check(api_.clSetKernelArg(kernel_ffn_swiglu_, 1,
                              sizeof(intermediate_arg), &intermediate_arg),
          "clSetKernelArg(expert8+shared SwiGLU 1)");
    Check(api_.clSetKernelArg(kernel_ffn_swiglu_, 2,
                              sizeof(expert_arg), &expert_arg),
          "clSetKernelArg(expert8+shared SwiGLU 2)");
    Check(api_.clSetKernelArg(kernel_ffn_swiglu_, 3,
                              sizeof(swiglu_buffer), &swiglu_buffer),
          "clSetKernelArg(expert8+shared SwiGLU 3)");

    const std::size_t matvec_global_size =
        static_cast<std::size_t>(matvec_global);
    const std::size_t swiglu_global_size = run.swiglu.size();
    std::vector<double> matvec_times;
    std::vector<double> swiglu_times;
    matvec_times.reserve(static_cast<std::size_t>(repeat));
    swiglu_times.reserve(static_cast<std::size_t>(repeat));
    const bool read_in_kernel = repeat == 1;
    for (int i = 0; i < repeat; ++i) {
      cl_event matvec_event = nullptr;
      cl_event swiglu_event = nullptr;
      Check(api_.clEnqueueNDRangeKernel(
                queue_, matvec_kernel, 1, nullptr, &matvec_global_size,
                matvec_local, 0, nullptr, EventOut(&matvec_event)),
            "clEnqueueNDRangeKernel(q4k_x8_matvec_rowlane_expert8_plus_shared_localq8)");
      Check(api_.clEnqueueNDRangeKernel(
                queue_, kernel_ffn_swiglu_, 1, nullptr, &swiglu_global_size,
                nullptr, 0, nullptr, EventOut(&swiglu_event)),
            "clEnqueueNDRangeKernel(expert8+shared SwiGLU)");
      if (read_in_kernel) {
        Check(api_.clEnqueueReadBuffer(queue_, swiglu_buffer, kClTrue, 0,
                                       run.swiglu.size() * sizeof(float),
                                       run.swiglu.data(), 0, nullptr,
                                       nullptr),
              "clEnqueueReadBuffer(expert8+shared SwiGLU output)");
      } else {
        Check(api_.clFinish(queue_), "clFinish(expert8+shared SwiGLU)");
      }
      matvec_times.push_back(EventUs(api_, matvec_event));
      swiglu_times.push_back(EventUs(api_, swiglu_event));
      ReleaseEvent(api_, &matvec_event);
      ReleaseEvent(api_, &swiglu_event);
    }
    ClearPendingHostUploadsAfterQueueDrain();
    run.timing.matvec.min_us =
        *std::min_element(matvec_times.begin(), matvec_times.end());
    run.timing.matvec.mean_us =
        std::accumulate(matvec_times.begin(), matvec_times.end(), 0.0) /
        static_cast<double>(matvec_times.size());
    run.timing.matvec.effective_packed_gb_s =
        static_cast<double>(matvec_global / kRowsInterleaved *
                            blocks_per_row * kQ4Kx8BlockBytes) /
        (run.timing.matvec.min_us / 1e6) / 1e9;
    run.timing.matvec.global_work_items = matvec_global;
    run.timing.matvec.rows_per_work_item = 1;
    run.timing.swiglu_min_us =
        *std::min_element(swiglu_times.begin(), swiglu_times.end());
    run.timing.swiglu_mean_us =
        std::accumulate(swiglu_times.begin(), swiglu_times.end(), 0.0) /
        static_cast<double>(swiglu_times.size());
    run.timing.swiglu_global_work_items = run.swiglu.size();
    run.timing.shell_sum_min_us =
        run.timing.matvec.min_us + run.timing.swiglu_min_us;
    run.timing.shell_sum_mean_us =
        run.timing.matvec.mean_us + run.timing.swiglu_mean_us;
    if (!read_in_kernel) {
      Check(api_.clEnqueueReadBuffer(queue_, swiglu_buffer, kClTrue, 0,
                                     run.swiglu.size() * sizeof(float),
                                     run.swiglu.data(), 0, nullptr, nullptr),
            "clEnqueueReadBuffer(expert8+shared SwiGLU output)");
    }
    return run;
  }

  GpuQ4X8SwiGluHandoffRun
  RunResidentPackedQ4X8TopKIndexedExpert8PlusSharedThenSwiGlu(
      std::uint64_t selected_material_handle,
      std::uint64_t shared_handle,
      const std::vector<std::uint32_t>& selected_positions,
      const std::vector<std::int8_t>& q8_qs,
      const std::vector<std::int16_t>& q8_bsums,
      const std::vector<float>& q8_d,
      std::uint64_t intermediate_size,
      int repeat,
      GpuQ4X8KernelVariant variant) {
    Require(selected_positions.size() == 8,
            "topk-indexed expert8+shared requires exactly 8 positions");
    Require(shared_handle != 0,
            "topk-indexed expert8+shared requires a shared handle");
    Require(variant == GpuQ4X8KernelVariant::kRowlaneParallel,
            "topk-indexed expert8+shared requires rowlane variant");
    Require(intermediate_size > 0,
            "topk-indexed expert8+shared intermediate size must be nonzero");

    const auto& selected_material =
        ResidentPackedQ4X8ForHandle(selected_material_handle);
    const auto& shared = ResidentPackedQ4X8ForHandle(shared_handle);
    const std::uint64_t rows_per_expert = shared.rows;
    const std::uint64_t blocks_per_row = shared.blocks_per_row;
    Require(rows_per_expert == intermediate_size * 2,
            "topk-indexed expert8+shared rows per expert mismatch");
    Require(rows_per_expert % kRowsInterleaved == 0,
            "topk-indexed expert8+shared rows must be divisible by 8");
    Require(selected_material.rows % rows_per_expert == 0,
            "topk-indexed selected material row count mismatch");
    Require(selected_material.blocks_per_row == blocks_per_row,
            "topk-indexed selected material block count mismatch");
    Require(blocks_per_row == 8,
            "topk-indexed expert8+shared local-Q8 handoff requires BPR8");
    const std::uint64_t material_expert_count =
        selected_material.rows / rows_per_expert;
    Require(material_expert_count >= selected_positions.size(),
            "topk-indexed selected material must hold at least 8 experts");
    for (const auto position : selected_positions) {
      Require(position < material_expert_count,
              "topk-indexed selected position out of material range");
    }
    ValidateQ8InputPlanes(q8_qs, q8_bsums, q8_d, blocks_per_row, repeat);

    GpuQ4X8SwiGluHandoffRun run;
    constexpr std::uint64_t kExpertCountPlusShared = 9;
    const std::uint64_t expert_count = kExpertCountPlusShared;
    run.swiglu.assign(
        static_cast<std::size_t>(intermediate_size * expert_count), 0.0f);
    cl_mem topk_index_buffer = EnsureScratchBuffer(
        expert8_swiglu_scratch_topk_indices_,
        selected_positions.size() * sizeof(std::uint32_t), kClMemReadOnly,
        "topk-indexed expert8+shared positions");
    cl_mem q8_qs_buffer = EnsureScratchBuffer(
        expert8_swiglu_scratch_q8_qs_, q8_qs.size(), kClMemReadOnly,
        "topk-indexed expert8+shared q8_qs");
    cl_mem q8_bsums_buffer = EnsureScratchBuffer(
        expert8_swiglu_scratch_q8_bsums_,
        q8_bsums.size() * sizeof(std::int16_t), kClMemReadOnly,
        "topk-indexed expert8+shared q8_bsums");
    cl_mem q8_d_buffer = EnsureScratchBuffer(
        expert8_swiglu_scratch_q8_d_, q8_d.size() * sizeof(float),
        kClMemReadOnly, "topk-indexed expert8+shared q8_d");
    cl_mem gate_up_buffer = EnsureScratchBuffer(
        expert8_swiglu_scratch_gate_up_,
        static_cast<std::size_t>(rows_per_expert * expert_count) *
            sizeof(float),
        kClMemReadWrite, "topk-indexed expert8+shared gate_up");
    cl_mem swiglu_buffer = EnsureScratchBuffer(
        expert8_swiglu_scratch_swiglu_, run.swiglu.size() * sizeof(float),
        kClMemWriteOnly, "topk-indexed expert8+shared output");
    Check(api_.clEnqueueWriteBuffer(
              queue_, topk_index_buffer, kClFalse, 0,
              selected_positions.size() * sizeof(std::uint32_t),
              selected_positions.data(), 0, nullptr, nullptr),
          "clEnqueueWriteBuffer(topk-indexed expert8+shared positions)");
    Check(api_.clEnqueueWriteBuffer(queue_, q8_qs_buffer, kClFalse, 0,
                                    q8_qs.size(), q8_qs.data(), 0, nullptr,
                                    nullptr),
          "clEnqueueWriteBuffer(topk-indexed expert8+shared q8_qs)");
    Check(api_.clEnqueueWriteBuffer(
              queue_, q8_bsums_buffer, kClFalse, 0,
              q8_bsums.size() * sizeof(std::int16_t), q8_bsums.data(), 0,
              nullptr, nullptr),
          "clEnqueueWriteBuffer(topk-indexed expert8+shared q8_bsums)");
    Check(api_.clEnqueueWriteBuffer(queue_, q8_d_buffer, kClFalse, 0,
                                    q8_d.size() * sizeof(float),
                                    q8_d.data(), 0, nullptr, nullptr),
          "clEnqueueWriteBuffer(topk-indexed expert8+shared q8_d)");

    const std::uint64_t row_groups_per_expert =
        rows_per_expert / kRowsInterleaved;
    const std::uint64_t matvec_global = rows_per_expert * expert_count;
    constexpr std::size_t kLocalQ8Size = 64;
    const std::size_t* matvec_local = &kLocalQ8Size;
    cl_kernel matvec_kernel =
        kernel_rowlane_topk_indexed_expert8_plus_shared_localq8_;
    const cl_uint blocks_arg = static_cast<cl_uint>(blocks_per_row);
    const cl_uint row_groups_arg =
        static_cast<cl_uint>(row_groups_per_expert);
    const cl_uint material_expert_count_arg =
        static_cast<cl_uint>(material_expert_count);
    Check(api_.clSetKernelArg(matvec_kernel, 0,
                              sizeof(selected_material.buffer),
                              &selected_material.buffer),
          "clSetKernelArg(topk-indexed selected material)");
    Check(api_.clSetKernelArg(matvec_kernel, 1,
                              sizeof(shared.buffer), &shared.buffer),
          "clSetKernelArg(topk-indexed shared packed)");
    Check(api_.clSetKernelArg(matvec_kernel, 2,
                              sizeof(topk_index_buffer), &topk_index_buffer),
          "clSetKernelArg(topk-indexed positions)");
    Check(api_.clSetKernelArg(matvec_kernel, 3,
                              sizeof(q8_qs_buffer), &q8_qs_buffer),
          "clSetKernelArg(topk-indexed q8_qs)");
    Check(api_.clSetKernelArg(matvec_kernel, 4,
                              sizeof(q8_bsums_buffer), &q8_bsums_buffer),
          "clSetKernelArg(topk-indexed q8_bsums)");
    Check(api_.clSetKernelArg(matvec_kernel, 5,
                              sizeof(q8_d_buffer), &q8_d_buffer),
          "clSetKernelArg(topk-indexed q8_d)");
    Check(api_.clSetKernelArg(matvec_kernel, 6,
                              sizeof(blocks_arg), &blocks_arg),
          "clSetKernelArg(topk-indexed blocks)");
    Check(api_.clSetKernelArg(matvec_kernel, 7,
                              sizeof(row_groups_arg), &row_groups_arg),
          "clSetKernelArg(topk-indexed row_groups)");
    Check(api_.clSetKernelArg(matvec_kernel, 8,
                              sizeof(material_expert_count_arg),
                              &material_expert_count_arg),
          "clSetKernelArg(topk-indexed material count)");
    Check(api_.clSetKernelArg(matvec_kernel, 9,
                              sizeof(gate_up_buffer), &gate_up_buffer),
          "clSetKernelArg(topk-indexed out)");

    const cl_uint intermediate_arg = static_cast<cl_uint>(intermediate_size);
    const cl_uint expert_arg = static_cast<cl_uint>(expert_count);
    Check(api_.clSetKernelArg(kernel_ffn_swiglu_, 0,
                              sizeof(gate_up_buffer), &gate_up_buffer),
          "clSetKernelArg(topk-indexed SwiGLU 0)");
    Check(api_.clSetKernelArg(kernel_ffn_swiglu_, 1,
                              sizeof(intermediate_arg), &intermediate_arg),
          "clSetKernelArg(topk-indexed SwiGLU 1)");
    Check(api_.clSetKernelArg(kernel_ffn_swiglu_, 2,
                              sizeof(expert_arg), &expert_arg),
          "clSetKernelArg(topk-indexed SwiGLU 2)");
    Check(api_.clSetKernelArg(kernel_ffn_swiglu_, 3,
                              sizeof(swiglu_buffer), &swiglu_buffer),
          "clSetKernelArg(topk-indexed SwiGLU 3)");

    const std::size_t matvec_global_size =
        static_cast<std::size_t>(matvec_global);
    const std::size_t swiglu_global_size = run.swiglu.size();
    std::vector<double> matvec_times;
    std::vector<double> swiglu_times;
    matvec_times.reserve(static_cast<std::size_t>(repeat));
    swiglu_times.reserve(static_cast<std::size_t>(repeat));
    const bool read_in_kernel = repeat == 1;
    for (int i = 0; i < repeat; ++i) {
      cl_event matvec_event = nullptr;
      cl_event swiglu_event = nullptr;
      Check(api_.clEnqueueNDRangeKernel(
                queue_, matvec_kernel, 1, nullptr, &matvec_global_size,
                matvec_local, 0, nullptr, EventOut(&matvec_event)),
            "clEnqueueNDRangeKernel(q4k_x8_matvec_topk_indexed_expert8_plus_shared_localq8)");
      Check(api_.clEnqueueNDRangeKernel(
                queue_, kernel_ffn_swiglu_, 1, nullptr, &swiglu_global_size,
                nullptr, 0, nullptr, EventOut(&swiglu_event)),
            "clEnqueueNDRangeKernel(topk-indexed expert8+shared SwiGLU)");
      if (read_in_kernel) {
        Check(api_.clEnqueueReadBuffer(queue_, swiglu_buffer, kClTrue, 0,
                                       run.swiglu.size() * sizeof(float),
                                       run.swiglu.data(), 0, nullptr,
                                       nullptr),
              "clEnqueueReadBuffer(topk-indexed SwiGLU output)");
      } else {
        Check(api_.clFinish(queue_), "clFinish(topk-indexed SwiGLU)");
      }
      matvec_times.push_back(EventUs(api_, matvec_event));
      swiglu_times.push_back(EventUs(api_, swiglu_event));
      ReleaseEvent(api_, &matvec_event);
      ReleaseEvent(api_, &swiglu_event);
    }
    ClearPendingHostUploadsAfterQueueDrain();
    run.timing.matvec.min_us =
        *std::min_element(matvec_times.begin(), matvec_times.end());
    run.timing.matvec.mean_us =
        std::accumulate(matvec_times.begin(), matvec_times.end(), 0.0) /
        static_cast<double>(matvec_times.size());
    run.timing.matvec.effective_packed_gb_s =
        static_cast<double>(matvec_global / kRowsInterleaved *
                            blocks_per_row * kQ4Kx8BlockBytes) /
        (run.timing.matvec.min_us / 1e6) / 1e9;
    run.timing.matvec.global_work_items = matvec_global;
    run.timing.matvec.rows_per_work_item = 1;
    run.timing.swiglu_min_us =
        *std::min_element(swiglu_times.begin(), swiglu_times.end());
    run.timing.swiglu_mean_us =
        std::accumulate(swiglu_times.begin(), swiglu_times.end(), 0.0) /
        static_cast<double>(swiglu_times.size());
    run.timing.swiglu_global_work_items = run.swiglu.size();
    run.timing.shell_sum_min_us =
        run.timing.matvec.min_us + run.timing.swiglu_min_us;
    run.timing.shell_sum_mean_us =
        run.timing.matvec.mean_us + run.timing.swiglu_mean_us;
    if (!read_in_kernel) {
      Check(api_.clEnqueueReadBuffer(queue_, swiglu_buffer, kClTrue, 0,
                                     run.swiglu.size() * sizeof(float),
                                     run.swiglu.data(), 0, nullptr, nullptr),
            "clEnqueueReadBuffer(topk-indexed SwiGLU output)");
    }
    return run;
  }

  GpuQ4X8SwiGluQ4F32DownHandoffRun
  RunResidentPackedQ4X8Expert8ThenSwiGluThenPackedQ4X8Expert8F32Input(
      const std::vector<std::uint64_t>& gate_up_handles,
      const std::vector<std::uint64_t>& down_handles,
      const std::vector<std::int8_t>& q8_qs,
      const std::vector<std::int16_t>& q8_bsums,
      const std::vector<float>& q8_d,
      std::uint64_t intermediate_size,
      std::uint64_t rows_per_expert,
      int repeat,
      GpuQ4X8KernelVariant variant,
      bool readback_output = true) {
    Require(gate_up_handles.size() == 8 && down_handles.size() == 8,
            "expert8 SwiGLU Q4 f32-input handoff requires eight experts");
    Require(variant == GpuQ4X8KernelVariant::kRowlaneParallel,
            "expert8 SwiGLU Q4 f32-input handoff requires rowlane variant");
    Require(intermediate_size > 0,
            "expert8 SwiGLU Q4 f32-input intermediate size must be nonzero");
    Require(rows_per_expert > 0,
            "expert8 SwiGLU Q4 f32-input rows_per_expert must be nonzero");
    std::array<cl_mem, 8> gate_up_buffers{};
    std::array<cl_mem, 8> down_buffers{};
    std::uint64_t gate_up_rows_per_expert = 0;
    std::uint64_t gate_up_blocks_per_row = 0;
    std::uint64_t down_blocks_per_row = 0;
    for (std::size_t i = 0; i < gate_up_handles.size(); ++i) {
      const auto& gate = ResidentPackedQ4X8ForHandle(gate_up_handles[i]);
      const auto& down = ResidentPackedQ4X8ForHandle(down_handles[i]);
      if (i == 0) {
        gate_up_rows_per_expert = gate.rows;
        gate_up_blocks_per_row = gate.blocks_per_row;
        down_blocks_per_row = down.blocks_per_row;
      } else {
        Require(gate.rows == gate_up_rows_per_expert,
                "expert8 SwiGLU Q4 f32-input gate/up row count mismatch");
        Require(gate.blocks_per_row == gate_up_blocks_per_row,
                "expert8 SwiGLU Q4 f32-input gate/up block count mismatch");
        Require(down.blocks_per_row == down_blocks_per_row,
                "expert8 SwiGLU Q4 f32-input down block count mismatch");
      }
      Require(down.rows == rows_per_expert,
              "expert8 SwiGLU Q4 f32-input down row count mismatch");
      gate_up_buffers[i] = gate.buffer;
      down_buffers[i] = down.buffer;
    }
    Require(gate_up_rows_per_expert == intermediate_size * 2,
            "expert8 SwiGLU Q4 f32-input gate/up row count mismatch");
    Require(gate_up_rows_per_expert % kRowsInterleaved == 0,
            "expert8 SwiGLU Q4 f32-input gate/up rows must be divisible by 8");
    Require(rows_per_expert % kRowsInterleaved == 0,
            "expert8 SwiGLU Q4 f32-input down rows must be divisible by 8");
    Require(down_blocks_per_row * 256 == intermediate_size,
            "expert8 SwiGLU Q4 f32-input down block count mismatch");
    ValidateQ8InputPlanes(q8_qs, q8_bsums, q8_d, gate_up_blocks_per_row,
                           repeat);

    GpuQ4X8SwiGluQ4F32DownHandoffRun run;
    const std::uint64_t expert_count = gate_up_handles.size();
    const std::uint64_t swiglu_values = intermediate_size * expert_count;
    const std::uint64_t down_rows = rows_per_expert * expert_count;
    if (readback_output) {
      run.down.assign(static_cast<std::size_t>(down_rows), 0.0f);
    }

    const bool defer_finish =
        !readback_output && DeferFfnDownFinishBundle();
    if (defer_finish) {
      pending_host_uploads_.reserve(pending_host_uploads_.size() + 3U);
    }
    cl_mem q8_qs_buffer = EnsureScratchBuffer(
        expert8_swiglu_scratch_q8_qs_, q8_qs.size(), kClMemReadOnly,
        "expert8 SwiGLU Q4 f32-input q8_qs");
    cl_mem q8_bsums_buffer = EnsureScratchBuffer(
        expert8_swiglu_scratch_q8_bsums_,
        q8_bsums.size() * sizeof(std::int16_t), kClMemReadOnly,
        "expert8 SwiGLU Q4 f32-input q8_bsums");
    cl_mem q8_d_buffer = EnsureScratchBuffer(
        expert8_swiglu_scratch_q8_d_, q8_d.size() * sizeof(float),
        kClMemReadOnly, "expert8 SwiGLU Q4 f32-input q8_d");
    cl_mem gate_up_buffer = EnsureScratchBuffer(
        expert8_swiglu_scratch_gate_up_,
        static_cast<std::size_t>(gate_up_rows_per_expert * expert_count) *
            sizeof(float),
        kClMemReadWrite, "expert8 SwiGLU Q4 f32-input gate_up");
    cl_mem swiglu_buffer = EnsureScratchBuffer(
        expert8_swiglu_scratch_swiglu_,
        static_cast<std::size_t>(swiglu_values) * sizeof(float),
        kClMemReadWrite, "expert8 SwiGLU Q4 f32-input swiglu");
    cl_mem down_buffer = EnsureScratchBuffer(
        expert8_q4_down_scratch_out_,
        static_cast<std::size_t>(down_rows) * sizeof(float),
        kClMemWriteOnly, "expert8 SwiGLU Q4 f32-input down out");

    const void* q8_qs_data =
        defer_finish ? StagePendingHostUpload(q8_qs.data(), q8_qs.size())
                     : static_cast<const void*>(q8_qs.data());
    const void* q8_bsums_data =
        defer_finish
            ? StagePendingHostUpload(
                  q8_bsums.data(), q8_bsums.size() * sizeof(std::int16_t))
            : static_cast<const void*>(q8_bsums.data());
    const void* q8_d_data =
        defer_finish
            ? StagePendingHostUpload(q8_d.data(), q8_d.size() * sizeof(float))
            : static_cast<const void*>(q8_d.data());
    Check(api_.clEnqueueWriteBuffer(queue_, q8_qs_buffer, kClFalse, 0,
                                    q8_qs.size(), q8_qs_data, 0, nullptr,
                                    nullptr),
          "clEnqueueWriteBuffer(expert8 SwiGLU Q4 f32-input q8_qs)");
    Check(api_.clEnqueueWriteBuffer(
              queue_, q8_bsums_buffer, kClFalse, 0,
              q8_bsums.size() * sizeof(std::int16_t), q8_bsums_data, 0,
              nullptr, nullptr),
          "clEnqueueWriteBuffer(expert8 SwiGLU Q4 f32-input q8_bsums)");
    Check(api_.clEnqueueWriteBuffer(queue_, q8_d_buffer, kClFalse, 0,
                                    q8_d.size() * sizeof(float), q8_d_data,
                                    0, nullptr, nullptr),
          "clEnqueueWriteBuffer(expert8 SwiGLU Q4 f32-input q8_d)");

    const std::uint64_t gate_up_row_groups =
        gate_up_rows_per_expert / kRowsInterleaved;
    const std::uint64_t gate_up_global =
        gate_up_rows_per_expert * expert_count;
    const cl_uint gate_up_blocks_arg =
        static_cast<cl_uint>(gate_up_blocks_per_row);
    const cl_uint gate_up_row_groups_arg =
        static_cast<cl_uint>(gate_up_row_groups);
    for (std::size_t i = 0; i < gate_up_buffers.size(); ++i) {
      Check(api_.clSetKernelArg(kernel_rowlane_expert8_,
                                static_cast<cl_uint>(i),
                                sizeof(gate_up_buffers[i]),
                                &gate_up_buffers[i]),
            "clSetKernelArg(expert8 SwiGLU Q4 f32-input gate/up packed)");
    }
    Check(api_.clSetKernelArg(kernel_rowlane_expert8_, 8,
                              sizeof(q8_qs_buffer), &q8_qs_buffer),
          "clSetKernelArg(expert8 SwiGLU Q4 f32-input gate/up q8_qs)");
    Check(api_.clSetKernelArg(kernel_rowlane_expert8_, 9,
                              sizeof(q8_bsums_buffer), &q8_bsums_buffer),
          "clSetKernelArg(expert8 SwiGLU Q4 f32-input gate/up q8_bsums)");
    Check(api_.clSetKernelArg(kernel_rowlane_expert8_, 10,
                              sizeof(q8_d_buffer), &q8_d_buffer),
          "clSetKernelArg(expert8 SwiGLU Q4 f32-input gate/up q8_d)");
    Check(api_.clSetKernelArg(kernel_rowlane_expert8_, 11,
                              sizeof(gate_up_blocks_arg),
                              &gate_up_blocks_arg),
          "clSetKernelArg(expert8 SwiGLU Q4 f32-input gate/up blocks)");
    Check(api_.clSetKernelArg(kernel_rowlane_expert8_, 12,
                              sizeof(gate_up_row_groups_arg),
                              &gate_up_row_groups_arg),
          "clSetKernelArg(expert8 SwiGLU Q4 f32-input gate/up row groups)");
    Check(api_.clSetKernelArg(kernel_rowlane_expert8_, 13,
                              sizeof(gate_up_buffer), &gate_up_buffer),
          "clSetKernelArg(expert8 SwiGLU Q4 f32-input gate/up out)");

    const cl_uint intermediate_arg =
        static_cast<cl_uint>(intermediate_size);
    const cl_uint expert_arg = static_cast<cl_uint>(expert_count);
    Check(api_.clSetKernelArg(kernel_ffn_swiglu_, 0,
                              sizeof(gate_up_buffer), &gate_up_buffer),
          "clSetKernelArg(expert8 SwiGLU Q4 f32-input swiglu 0)");
    Check(api_.clSetKernelArg(kernel_ffn_swiglu_, 1,
                              sizeof(intermediate_arg), &intermediate_arg),
          "clSetKernelArg(expert8 SwiGLU Q4 f32-input swiglu 1)");
    Check(api_.clSetKernelArg(kernel_ffn_swiglu_, 2, sizeof(expert_arg),
                              &expert_arg),
          "clSetKernelArg(expert8 SwiGLU Q4 f32-input swiglu 2)");
    Check(api_.clSetKernelArg(kernel_ffn_swiglu_, 3,
                              sizeof(swiglu_buffer), &swiglu_buffer),
          "clSetKernelArg(expert8 SwiGLU Q4 f32-input swiglu 3)");

    const cl_uint down_blocks_arg =
        static_cast<cl_uint>(down_blocks_per_row);
    const cl_uint down_row_groups_arg =
        static_cast<cl_uint>(rows_per_expert / kRowsInterleaved);
    for (std::size_t i = 0; i < down_buffers.size(); ++i) {
      Check(api_.clSetKernelArg(kernel_rowlane_expert8_f32input_,
                                static_cast<cl_uint>(i),
                                sizeof(down_buffers[i]), &down_buffers[i]),
            "clSetKernelArg(expert8 SwiGLU Q4 f32-input down packed)");
    }
    Check(api_.clSetKernelArg(kernel_rowlane_expert8_f32input_, 8,
                              sizeof(swiglu_buffer), &swiglu_buffer),
          "clSetKernelArg(expert8 SwiGLU Q4 f32-input down input)");
    Check(api_.clSetKernelArg(kernel_rowlane_expert8_f32input_, 9,
                              sizeof(down_blocks_arg), &down_blocks_arg),
          "clSetKernelArg(expert8 SwiGLU Q4 f32-input down blocks)");
    Check(api_.clSetKernelArg(kernel_rowlane_expert8_f32input_, 10,
                              sizeof(down_row_groups_arg),
                              &down_row_groups_arg),
          "clSetKernelArg(expert8 SwiGLU Q4 f32-input down row groups)");
    Check(api_.clSetKernelArg(kernel_rowlane_expert8_f32input_, 11,
                              sizeof(down_buffer), &down_buffer),
          "clSetKernelArg(expert8 SwiGLU Q4 f32-input down out)");

    const std::size_t gate_up_global_size =
        static_cast<std::size_t>(gate_up_global);
    const std::size_t swiglu_global_size =
        static_cast<std::size_t>(swiglu_values);
    const std::size_t down_global_size =
        static_cast<std::size_t>(down_rows);
    std::vector<double> gate_up_times;
    std::vector<double> swiglu_times;
    std::vector<double> down_times;
    gate_up_times.reserve(static_cast<std::size_t>(repeat));
    swiglu_times.reserve(static_cast<std::size_t>(repeat));
    down_times.reserve(static_cast<std::size_t>(repeat));
    const bool read_in_kernel = readback_output && repeat == 1;
    for (int i = 0; i < repeat; ++i) {
      cl_event gate_up_event = nullptr;
      cl_event swiglu_event = nullptr;
      cl_event down_event = nullptr;
      Check(api_.clEnqueueNDRangeKernel(
                queue_, kernel_rowlane_expert8_, 1, nullptr,
                &gate_up_global_size, nullptr, 0, nullptr,
                EventOut(&gate_up_event)),
            "clEnqueueNDRangeKernel(expert8 SwiGLU Q4 f32-input gate/up)");
      Check(api_.clEnqueueNDRangeKernel(
                queue_, kernel_ffn_swiglu_, 1, nullptr, &swiglu_global_size,
                nullptr, 0, nullptr, EventOut(&swiglu_event)),
            "clEnqueueNDRangeKernel(expert8 SwiGLU Q4 f32-input swiglu)");
      Check(api_.clEnqueueNDRangeKernel(
                queue_, kernel_rowlane_expert8_f32input_, 1, nullptr,
                &down_global_size, nullptr, 0, nullptr, EventOut(&down_event)),
            "clEnqueueNDRangeKernel(expert8 SwiGLU Q4 f32-input down)");
      if (read_in_kernel) {
        Check(api_.clEnqueueReadBuffer(queue_, down_buffer, kClTrue, 0,
                                       run.down.size() * sizeof(float),
                                       run.down.data(), 0, nullptr, nullptr),
              "clEnqueueReadBuffer(expert8 SwiGLU Q4 f32-input down)");
      } else if (!defer_finish) {
        Check(api_.clFinish(queue_),
              "clFinish(expert8 SwiGLU Q4 f32-input handoff)");
      }
      gate_up_times.push_back(EventUs(api_, gate_up_event));
      swiglu_times.push_back(EventUs(api_, swiglu_event));
      down_times.push_back(EventUs(api_, down_event));
      ReleaseEvent(api_, &down_event);
      ReleaseEvent(api_, &swiglu_event);
      ReleaseEvent(api_, &gate_up_event);
    }
    if (!defer_finish) {
      ClearPendingHostUploadsAfterQueueDrain();
    }

    run.timing.gate_up.min_us =
        *std::min_element(gate_up_times.begin(), gate_up_times.end());
    run.timing.gate_up.mean_us =
        std::accumulate(gate_up_times.begin(), gate_up_times.end(), 0.0) /
          static_cast<double>(gate_up_times.size());
    run.timing.gate_up.effective_packed_gb_s =
        static_cast<double>(gate_up_global / kRowsInterleaved *
                            gate_up_blocks_per_row * kQ4Kx8BlockBytes) /
        (run.timing.gate_up.min_us / 1e6) / 1e9;
    run.timing.gate_up.global_work_items = gate_up_global;
    run.timing.gate_up.rows_per_work_item = 1;
    run.timing.swiglu_min_us =
        *std::min_element(swiglu_times.begin(), swiglu_times.end());
    run.timing.swiglu_mean_us =
        std::accumulate(swiglu_times.begin(), swiglu_times.end(), 0.0) /
          static_cast<double>(swiglu_times.size());
    run.timing.swiglu_global_work_items = swiglu_values;
    run.timing.down.min_us =
        *std::min_element(down_times.begin(), down_times.end());
    run.timing.down.mean_us =
        std::accumulate(down_times.begin(), down_times.end(), 0.0) /
          static_cast<double>(down_times.size());
    run.timing.down.effective_packed_gb_s =
        static_cast<double>(down_rows / kRowsInterleaved *
                            down_blocks_per_row * kQ4Kx8BlockBytes) /
        (run.timing.down.min_us / 1e6) / 1e9;
    run.timing.down.global_work_items = down_rows;
    run.timing.down.rows_per_work_item = 1;
    run.timing.shell_sum_min_us =
        run.timing.gate_up.min_us + run.timing.swiglu_min_us +
        run.timing.down.min_us;
    run.timing.shell_sum_mean_us =
        run.timing.gate_up.mean_us + run.timing.swiglu_mean_us +
        run.timing.down.mean_us;
    if (readback_output && !read_in_kernel) {
      Check(api_.clEnqueueReadBuffer(queue_, down_buffer, kClTrue, 0,
                                     run.down.size() * sizeof(float),
                                     run.down.data(), 0, nullptr, nullptr),
            "clEnqueueReadBuffer(expert8 SwiGLU Q4 f32-input down)");
    }
    run.down_handle = RegisterF32BufferAlias(
        &expert8_q4_down_output_alias_handle_, down_buffer, down_rows);
    return run;
  }

  GpuQ4X8SwiGluQ6DownHandoffRun
  RunResidentPackedQ4X8ThenSwiGluThenRawQ6KSelected(
      std::uint64_t packed_handle,
      std::uint64_t q6_handle,
      const std::vector<std::int8_t>& q8_qs,
      const std::vector<std::int16_t>& q8_bsums,
      const std::vector<float>& q8_d,
      std::uint64_t intermediate_size,
      const std::vector<std::uint32_t>& source_expert_by_output,
      std::uint64_t rows_per_expert,
      int repeat,
      GpuQ4X8KernelVariant variant) {
    const auto& packed = ResidentPackedQ4X8ForHandle(packed_handle);
    const auto& q6 = ResidentRawQ6KForHandle(q6_handle);
    Require(intermediate_size > 0,
            "SwiGLU Q6 handoff intermediate size must be nonzero");
    Require(rows_per_expert > 0,
            "SwiGLU Q6 handoff rows_per_expert must be nonzero");
    Require(!source_expert_by_output.empty(),
            "SwiGLU Q6 handoff source expert map must be nonempty");
    const std::uint64_t expert_count = source_expert_by_output.size();
    Require(packed.rows == intermediate_size * expert_count * 2,
            "SwiGLU Q6 handoff packed row count mismatch");
    Require(q6.rows == rows_per_expert * expert_count,
            "SwiGLU Q6 handoff selected Q6 row count mismatch");
    Require(q6.blocks_per_row * 256 == intermediate_size,
            "SwiGLU Q6 handoff Q6 block count mismatch");
    for (const auto source : source_expert_by_output) {
      Require(source < expert_count,
              "SwiGLU Q6 handoff source expert index out of range");
    }
    ValidateQ8InputPlanes(q8_qs, q8_bsums, q8_d, packed.blocks_per_row, repeat);

    GpuQ4X8SwiGluQ6DownHandoffRun run;
    std::vector<float> gate_up(static_cast<std::size_t>(packed.rows), 0.0f);
    const std::size_t swiglu_values =
        static_cast<std::size_t>(intermediate_size * expert_count);
    std::vector<float> swiglu(swiglu_values, 0.0f);
    run.down.assign(static_cast<std::size_t>(q6.rows), 0.0f);
    cl_mem q8_qs_buffer = nullptr, q8_bsums_buffer = nullptr;
    cl_mem q8_d_buffer = nullptr, gate_up_buffer = nullptr;
    cl_mem source_map_buffer = nullptr, swiglu_buffer = nullptr;
    cl_mem swiglu_q8_qs_buffer = nullptr, swiglu_q8_d_buffer = nullptr;
    cl_mem down_buffer = nullptr;
    auto release_all = [&]() {
      ReleaseMem(api_, &down_buffer);
      ReleaseMem(api_, &swiglu_q8_d_buffer);
      ReleaseMem(api_, &swiglu_q8_qs_buffer);
      ReleaseMem(api_, &swiglu_buffer);
      ReleaseMem(api_, &source_map_buffer);
      ReleaseMem(api_, &gate_up_buffer);
      ReleaseMem(api_, &q8_d_buffer);
      ReleaseMem(api_, &q8_bsums_buffer);
      ReleaseMem(api_, &q8_qs_buffer);
    };
    try {
      CreateRunBuffersWithoutPacked(q8_qs, q8_bsums, q8_d, gate_up,
                                    &q8_qs_buffer, &q8_bsums_buffer,
                                    &q8_d_buffer, &gate_up_buffer,
                                    kClMemReadWrite);
      cl_int err = kClSuccess;
      source_map_buffer = api_.clCreateBuffer(
          context_, kClMemReadOnly,
          source_expert_by_output.size() * sizeof(std::uint32_t), nullptr,
          &err);
      Check(err, "clCreateBuffer(SwiGLU Q6 source map)");
      swiglu_buffer = api_.clCreateBuffer(
          context_, kClMemReadWrite, swiglu.size() * sizeof(float), nullptr,
          &err);
      Check(err, "clCreateBuffer(SwiGLU Q6 swiglu)");
      swiglu_q8_qs_buffer = api_.clCreateBuffer(
          context_, kClMemReadWrite, swiglu.size() * sizeof(std::int8_t),
          nullptr, &err);
      Check(err, "clCreateBuffer(SwiGLU Q6 q8 qs)");
      swiglu_q8_d_buffer = api_.clCreateBuffer(
          context_, kClMemReadWrite,
          expert_count * q6.blocks_per_row * sizeof(float), nullptr, &err);
      Check(err, "clCreateBuffer(SwiGLU Q6 q8 d)");
      down_buffer = api_.clCreateBuffer(
          context_, kClMemWriteOnly, run.down.size() * sizeof(float), nullptr,
          &err);
      Check(err, "clCreateBuffer(SwiGLU Q6 down)");
      Check(api_.clEnqueueWriteBuffer(
                queue_, source_map_buffer, kClTrue, 0,
                source_expert_by_output.size() * sizeof(std::uint32_t),
                source_expert_by_output.data(), 0, nullptr, nullptr),
            "clEnqueueWriteBuffer(SwiGLU Q6 source map)");

      const std::uint64_t row_groups = packed.rows / kRowsInterleaved;
      const std::uint64_t matvec_global =
          variant == GpuQ4X8KernelVariant::kGroup8Serial ? row_groups : packed.rows;
      run.timing.gate_up = RunKernel(
          variant, packed.buffer, q8_qs_buffer, q8_bsums_buffer, q8_d_buffer,
          gate_up_buffer, packed.blocks_per_row, row_groups, matvec_global,
          repeat);
      const auto swiglu_timing = RunSwiGluReorderKernel(
          gate_up_buffer, source_map_buffer, swiglu_buffer, intermediate_size,
          expert_count, repeat);
      run.timing.swiglu_min_us = swiglu_timing.min_us;
      run.timing.swiglu_mean_us = swiglu_timing.mean_us;
      run.timing.swiglu_global_work_items = swiglu_timing.global_work_items;
      const auto q8_timing = RunQ8QuantizeKernel(
          swiglu_buffer, expert_count * q6.blocks_per_row,
          swiglu_q8_qs_buffer, swiglu_q8_d_buffer, repeat);
      run.timing.q8_quantize_min_us = q8_timing.min_us;
      run.timing.q8_quantize_mean_us = q8_timing.mean_us;
      run.timing.q8_quantize_global_work_items = q8_timing.global_work_items;
      run.timing.down =
          q6.rowstripe
              ? RunQ6KSelectedRowstripeKernel(
                    q6.buffer, swiglu_q8_qs_buffer, swiglu_q8_d_buffer,
                    down_buffer, rows_per_expert, q6.blocks_per_row, q6.rows,
                    q6.rows_per_tile, repeat)
              : RunQ6KSelectedKernel(
                    q6.buffer, swiglu_q8_qs_buffer, swiglu_q8_d_buffer,
                    down_buffer, rows_per_expert, q6.blocks_per_row, q6.rows,
                    repeat);
      run.timing.shell_sum_min_us =
          run.timing.gate_up.min_us + run.timing.swiglu_min_us +
          run.timing.q8_quantize_min_us + run.timing.down.min_us;
      run.timing.shell_sum_mean_us =
          run.timing.gate_up.mean_us + run.timing.swiglu_mean_us +
          run.timing.q8_quantize_mean_us + run.timing.down.mean_us;
      Check(api_.clEnqueueReadBuffer(queue_, down_buffer, kClTrue, 0,
                                     run.down.size() * sizeof(float),
                                     run.down.data(), 0, nullptr, nullptr),
            "clEnqueueReadBuffer(SwiGLU Q6 down)");
    } catch (...) {
      release_all();
      throw;
    }
    release_all();
    return run;
  }

  GpuQ4X8SwiGluQ6DownHandoffRun
  RunResidentPackedQ4X8Expert8ThenSwiGluThenRawQ6KExpert8(
      const std::vector<std::uint64_t>& packed_handles,
      const std::vector<std::uint64_t>& q6_handles,
      const std::vector<std::int8_t>& q8_qs,
      const std::vector<std::int16_t>& q8_bsums,
      const std::vector<float>& q8_d,
      std::uint64_t intermediate_size,
      std::uint64_t rows_per_expert,
      int repeat,
      GpuQ4X8KernelVariant variant) {
    Require(packed_handles.size() == 8,
            "expert8 SwiGLU Q6 handoff requires exactly 8 packed handles");
    Require(q6_handles.size() == 8,
            "expert8 SwiGLU Q6 handoff requires exactly 8 Q6 handles");
    Require(variant == GpuQ4X8KernelVariant::kRowlaneParallel,
            "expert8 SwiGLU Q6 handoff requires rowlane variant");
    Require(intermediate_size > 0,
            "expert8 SwiGLU Q6 handoff intermediate size must be nonzero");
    Require(rows_per_expert > 0,
            "expert8 SwiGLU Q6 handoff rows_per_expert must be nonzero");

    std::array<cl_mem, 8> packed_buffers{};
    std::array<cl_mem, 8> q6_buffers{};
    std::uint64_t packed_rows_per_expert = 0;
    std::uint64_t packed_blocks_per_row = 0;
    std::uint64_t q6_blocks_per_row = 0;
    bool q6_rowstripe = false;
    std::uint64_t q6_rows_per_tile = 0;
    std::uint64_t q6_row_tile_count = 0;
    for (std::size_t i = 0; i < packed_handles.size(); ++i) {
      const auto& packed = ResidentPackedQ4X8ForHandle(packed_handles[i]);
      const auto& q6 = ResidentRawQ6KForHandle(q6_handles[i]);
      if (i == 0) {
        packed_rows_per_expert = packed.rows;
        packed_blocks_per_row = packed.blocks_per_row;
        q6_blocks_per_row = q6.blocks_per_row;
        q6_rowstripe = q6.rowstripe;
        q6_rows_per_tile = q6.rows_per_tile;
        q6_row_tile_count = q6.row_tile_count;
      } else {
        Require(packed.rows == packed_rows_per_expert,
                "expert8 SwiGLU Q6 handoff packed row count mismatch");
        Require(packed.blocks_per_row == packed_blocks_per_row,
                "expert8 SwiGLU Q6 handoff packed block count mismatch");
        Require(q6.blocks_per_row == q6_blocks_per_row,
                "expert8 SwiGLU Q6 handoff Q6 block count mismatch");
        Require(q6.rowstripe == q6_rowstripe,
                "expert8 SwiGLU Q6 handoff rowstripe layout mismatch");
        Require(q6.rows_per_tile == q6_rows_per_tile,
                "expert8 SwiGLU Q6 handoff rows_per_tile mismatch");
        Require(q6.row_tile_count == q6_row_tile_count,
                "expert8 SwiGLU Q6 handoff row tile count mismatch");
      }
      Require(q6.rows == rows_per_expert,
              "expert8 SwiGLU Q6 handoff Q6 row count mismatch");
      if (q6_rowstripe) {
        Require(q6.rows_per_tile > 0,
                "expert8 SwiGLU Q6 handoff rowstripe tile size missing");
        Require(q6.row_tile_count > 0,
                "expert8 SwiGLU Q6 handoff rowstripe tile count missing");
      }
      packed_buffers[i] = packed.buffer;
      q6_buffers[i] = q6.buffer;
    }
    const std::uint64_t expert_count = packed_handles.size();
    Require(packed_rows_per_expert == intermediate_size * 2,
            "expert8 SwiGLU Q6 handoff packed rows per expert mismatch");
    Require(packed_rows_per_expert % kRowsInterleaved == 0,
            "expert8 SwiGLU Q6 handoff rows must be divisible by 8");
    Require(q6_blocks_per_row * 256 == intermediate_size,
            "expert8 SwiGLU Q6 handoff Q6 block count mismatch");
    ValidateQ8InputPlanes(q8_qs, q8_bsums, q8_d, packed_blocks_per_row,
                           repeat);

    GpuQ4X8SwiGluQ6DownHandoffRun run;
    const std::uint64_t swiglu_values = intermediate_size * expert_count;
    const std::uint64_t q6_rows = rows_per_expert * expert_count;
    run.down.assign(static_cast<std::size_t>(q6_rows), 0.0f);

    cl_mem q8_qs_buffer = EnsureScratchBuffer(
        expert8_swiglu_scratch_q8_qs_, q8_qs.size(), kClMemReadOnly,
        "expert8 fused SwiGLU Q6 q8_qs");
    cl_mem q8_bsums_buffer = EnsureScratchBuffer(
        expert8_swiglu_scratch_q8_bsums_,
        q8_bsums.size() * sizeof(std::int16_t), kClMemReadOnly,
        "expert8 fused SwiGLU Q6 q8_bsums");
    cl_mem q8_d_buffer = EnsureScratchBuffer(
        expert8_swiglu_scratch_q8_d_, q8_d.size() * sizeof(float),
        kClMemReadOnly, "expert8 fused SwiGLU Q6 q8_d");
    cl_mem gate_up_buffer = EnsureScratchBuffer(
        expert8_swiglu_scratch_gate_up_,
        static_cast<std::size_t>(packed_rows_per_expert * expert_count) *
            sizeof(float),
        kClMemReadWrite, "expert8 fused SwiGLU Q6 gate_up");
    cl_mem swiglu_buffer = EnsureScratchBuffer(
        expert8_swiglu_scratch_swiglu_,
        static_cast<std::size_t>(swiglu_values) * sizeof(float),
        kClMemReadWrite, "expert8 fused SwiGLU Q6 swiglu");
    cl_mem swiglu_q8_qs_buffer = EnsureScratchBuffer(
        expert8_q6_down_scratch_q8_qs_,
        static_cast<std::size_t>(swiglu_values) * sizeof(std::int8_t),
        kClMemReadWrite, "expert8 fused SwiGLU Q6 down q8_qs");
    cl_mem swiglu_q8_d_buffer = EnsureScratchBuffer(
        expert8_q6_down_scratch_q8_d_,
        static_cast<std::size_t>(expert_count * q6_blocks_per_row) *
            sizeof(float),
        kClMemReadWrite, "expert8 fused SwiGLU Q6 down q8_d");
    cl_mem down_buffer = EnsureScratchBuffer(
        expert8_q6_down_scratch_out_, run.down.size() * sizeof(float),
        kClMemWriteOnly, "expert8 fused SwiGLU Q6 down out");

    Check(api_.clEnqueueWriteBuffer(queue_, q8_qs_buffer, kClFalse, 0,
                                    q8_qs.size(), q8_qs.data(), 0, nullptr,
                                    nullptr),
          "clEnqueueWriteBuffer(expert8 fused SwiGLU Q6 q8_qs)");
    Check(api_.clEnqueueWriteBuffer(
              queue_, q8_bsums_buffer, kClFalse, 0,
              q8_bsums.size() * sizeof(std::int16_t), q8_bsums.data(), 0,
              nullptr, nullptr),
          "clEnqueueWriteBuffer(expert8 fused SwiGLU Q6 q8_bsums)");
    Check(api_.clEnqueueWriteBuffer(queue_, q8_d_buffer, kClFalse, 0,
                                    q8_d.size() * sizeof(float), q8_d.data(),
                                    0, nullptr, nullptr),
          "clEnqueueWriteBuffer(expert8 fused SwiGLU Q6 q8_d)");

    const std::uint64_t row_groups_per_expert =
        packed_rows_per_expert / kRowsInterleaved;
    const std::uint64_t matvec_global =
        packed_rows_per_expert * expert_count;
    const cl_uint packed_blocks_arg =
        static_cast<cl_uint>(packed_blocks_per_row);
    const cl_uint row_groups_arg =
        static_cast<cl_uint>(row_groups_per_expert);
    for (std::size_t i = 0; i < packed_buffers.size(); ++i) {
      Check(api_.clSetKernelArg(kernel_rowlane_expert8_,
                                static_cast<cl_uint>(i),
                                sizeof(packed_buffers[i]),
                                &packed_buffers[i]),
            "clSetKernelArg(expert8 fused SwiGLU Q6 packed)");
    }
    Check(api_.clSetKernelArg(kernel_rowlane_expert8_, 8,
                              sizeof(q8_qs_buffer), &q8_qs_buffer),
          "clSetKernelArg(expert8 fused SwiGLU Q6 q8_qs)");
    Check(api_.clSetKernelArg(kernel_rowlane_expert8_, 9,
                              sizeof(q8_bsums_buffer), &q8_bsums_buffer),
          "clSetKernelArg(expert8 fused SwiGLU Q6 q8_bsums)");
    Check(api_.clSetKernelArg(kernel_rowlane_expert8_, 10,
                              sizeof(q8_d_buffer), &q8_d_buffer),
          "clSetKernelArg(expert8 fused SwiGLU Q6 q8_d)");
    Check(api_.clSetKernelArg(kernel_rowlane_expert8_, 11,
                              sizeof(packed_blocks_arg), &packed_blocks_arg),
          "clSetKernelArg(expert8 fused SwiGLU Q6 packed blocks)");
    Check(api_.clSetKernelArg(kernel_rowlane_expert8_, 12,
                              sizeof(row_groups_arg), &row_groups_arg),
          "clSetKernelArg(expert8 fused SwiGLU Q6 row groups)");
    Check(api_.clSetKernelArg(kernel_rowlane_expert8_, 13,
                              sizeof(gate_up_buffer), &gate_up_buffer),
          "clSetKernelArg(expert8 fused SwiGLU Q6 gate_up)");

    const cl_uint intermediate_arg =
        static_cast<cl_uint>(intermediate_size);
    const cl_uint expert_arg = static_cast<cl_uint>(expert_count);
    Check(api_.clSetKernelArg(kernel_ffn_swiglu_, 0,
                              sizeof(gate_up_buffer), &gate_up_buffer),
          "clSetKernelArg(expert8 fused SwiGLU Q6 swiglu 0)");
    Check(api_.clSetKernelArg(kernel_ffn_swiglu_, 1,
                              sizeof(intermediate_arg), &intermediate_arg),
          "clSetKernelArg(expert8 fused SwiGLU Q6 swiglu 1)");
    Check(api_.clSetKernelArg(kernel_ffn_swiglu_, 2, sizeof(expert_arg),
                              &expert_arg),
          "clSetKernelArg(expert8 fused SwiGLU Q6 swiglu 2)");
    Check(api_.clSetKernelArg(kernel_ffn_swiglu_, 3,
                              sizeof(swiglu_buffer), &swiglu_buffer),
          "clSetKernelArg(expert8 fused SwiGLU Q6 swiglu 3)");

    const cl_uint q8_block_count_arg =
        static_cast<cl_uint>(expert_count * q6_blocks_per_row);
    Check(api_.clSetKernelArg(kernel_q8_quantize_, 0,
                              sizeof(swiglu_buffer), &swiglu_buffer),
          "clSetKernelArg(expert8 fused SwiGLU Q6 q8 0)");
    Check(api_.clSetKernelArg(kernel_q8_quantize_, 1,
                              sizeof(q8_block_count_arg),
                              &q8_block_count_arg),
          "clSetKernelArg(expert8 fused SwiGLU Q6 q8 1)");
    Check(api_.clSetKernelArg(kernel_q8_quantize_, 2,
                              sizeof(swiglu_q8_qs_buffer),
                              &swiglu_q8_qs_buffer),
          "clSetKernelArg(expert8 fused SwiGLU Q6 q8 2)");
    Check(api_.clSetKernelArg(kernel_q8_quantize_, 3,
                              sizeof(swiglu_q8_d_buffer),
                              &swiglu_q8_d_buffer),
          "clSetKernelArg(expert8 fused SwiGLU Q6 q8 3)");

    const cl_uint q6_rows_arg = static_cast<cl_uint>(rows_per_expert);
    const cl_uint q6_blocks_arg = static_cast<cl_uint>(q6_blocks_per_row);
    const cl_uint q6_rows_per_tile_arg =
        static_cast<cl_uint>(q6_rows_per_tile);
    cl_kernel q6_down_kernel =
        q6_rowstripe ? kernel_q6_matvec_rowstripe_expert8_
                     : kernel_q6_matvec_row_expert8_;
    for (std::size_t i = 0; i < q6_buffers.size(); ++i) {
      Check(api_.clSetKernelArg(q6_down_kernel,
                                static_cast<cl_uint>(i),
                                sizeof(q6_buffers[i]), &q6_buffers[i]),
            "clSetKernelArg(expert8 fused SwiGLU Q6 down raw)");
    }
    Check(api_.clSetKernelArg(q6_down_kernel, 8,
                              sizeof(swiglu_q8_qs_buffer),
                              &swiglu_q8_qs_buffer),
          "clSetKernelArg(expert8 fused SwiGLU Q6 down q8_qs)");
    Check(api_.clSetKernelArg(q6_down_kernel, 9,
                              sizeof(swiglu_q8_d_buffer),
                              &swiglu_q8_d_buffer),
          "clSetKernelArg(expert8 fused SwiGLU Q6 down q8_d)");
    Check(api_.clSetKernelArg(q6_down_kernel, 10,
                              sizeof(q6_rows_arg), &q6_rows_arg),
          "clSetKernelArg(expert8 fused SwiGLU Q6 down rows)");
    Check(api_.clSetKernelArg(q6_down_kernel, 11,
                              sizeof(q6_blocks_arg), &q6_blocks_arg),
          "clSetKernelArg(expert8 fused SwiGLU Q6 down blocks)");
    if (q6_rowstripe) {
      Check(api_.clSetKernelArg(q6_down_kernel, 12,
                                sizeof(q6_rows_per_tile_arg),
                                &q6_rows_per_tile_arg),
            "clSetKernelArg(expert8 fused SwiGLU Q6 down rows_per_tile)");
      Check(api_.clSetKernelArg(q6_down_kernel, 13,
                                sizeof(down_buffer), &down_buffer),
            "clSetKernelArg(expert8 fused SwiGLU Q6 down out)");
    } else {
      Check(api_.clSetKernelArg(q6_down_kernel, 12,
                                sizeof(down_buffer), &down_buffer),
            "clSetKernelArg(expert8 fused SwiGLU Q6 down out)");
    }

    const std::size_t matvec_global_size =
        static_cast<std::size_t>(matvec_global);
    const std::size_t swiglu_global_size =
        static_cast<std::size_t>(swiglu_values);
    const std::size_t q8_global_size =
        static_cast<std::size_t>(expert_count * q6_blocks_per_row);
    const std::size_t q6_global_size = static_cast<std::size_t>(q6_rows);
    constexpr std::size_t kExpert8FusedQ6LocalSize = 64;
    const std::size_t* q6_local =
        (q6_global_size % kExpert8FusedQ6LocalSize == 0)
            ? &kExpert8FusedQ6LocalSize
            : nullptr;
    std::vector<double> matvec_times;
    std::vector<double> swiglu_times;
    std::vector<double> q8_times;
    std::vector<double> q6_times;
    matvec_times.reserve(static_cast<std::size_t>(repeat));
    swiglu_times.reserve(static_cast<std::size_t>(repeat));
    q8_times.reserve(static_cast<std::size_t>(repeat));
    q6_times.reserve(static_cast<std::size_t>(repeat));
    for (int i = 0; i < repeat; ++i) {
      cl_event matvec_event = nullptr;
      cl_event swiglu_event = nullptr;
      cl_event q8_event = nullptr;
      cl_event q6_event = nullptr;
      Check(api_.clEnqueueNDRangeKernel(
                queue_, kernel_rowlane_expert8_, 1, nullptr,
                &matvec_global_size, nullptr, 0, nullptr, EventOut(&matvec_event)),
            "clEnqueueNDRangeKernel(expert8 fused SwiGLU Q6 matvec)");
      Check(api_.clEnqueueNDRangeKernel(
                queue_, kernel_ffn_swiglu_, 1, nullptr,
                &swiglu_global_size, nullptr, 0, nullptr, EventOut(&swiglu_event)),
            "clEnqueueNDRangeKernel(expert8 fused SwiGLU Q6 swiglu)");
      Check(api_.clEnqueueNDRangeKernel(
                queue_, kernel_q8_quantize_, 1, nullptr, &q8_global_size,
                nullptr, 0, nullptr, EventOut(&q8_event)),
            "clEnqueueNDRangeKernel(expert8 fused SwiGLU Q6 q8)");
      Check(api_.clEnqueueNDRangeKernel(
                queue_, q6_down_kernel, 1, nullptr, &q6_global_size, q6_local,
                0, nullptr, EventOut(&q6_event)),
            "clEnqueueNDRangeKernel(expert8 fused SwiGLU Q6 down)");
      Check(api_.clFinish(queue_),
            "clFinish(expert8 fused SwiGLU Q6 handoff)");
      matvec_times.push_back(EventUs(api_, matvec_event));
      swiglu_times.push_back(EventUs(api_, swiglu_event));
      q8_times.push_back(EventUs(api_, q8_event));
      q6_times.push_back(EventUs(api_, q6_event));
      ReleaseEvent(api_, &q6_event);
      ReleaseEvent(api_, &q8_event);
      ReleaseEvent(api_, &swiglu_event);
      ReleaseEvent(api_, &matvec_event);
    }
    ClearPendingHostUploadsAfterQueueDrain();

    run.timing.gate_up.min_us =
        *std::min_element(matvec_times.begin(), matvec_times.end());
    run.timing.gate_up.mean_us =
        std::accumulate(matvec_times.begin(), matvec_times.end(), 0.0) /
          static_cast<double>(matvec_times.size());
    run.timing.gate_up.effective_packed_gb_s =
        static_cast<double>(matvec_global / kRowsInterleaved *
                            packed_blocks_per_row * kQ4Kx8BlockBytes) /
        (run.timing.gate_up.min_us / 1e6) / 1e9;
    run.timing.gate_up.global_work_items = matvec_global;
    run.timing.gate_up.rows_per_work_item = 1;
    run.timing.swiglu_min_us =
        *std::min_element(swiglu_times.begin(), swiglu_times.end());
    run.timing.swiglu_mean_us =
        std::accumulate(swiglu_times.begin(), swiglu_times.end(), 0.0) /
          static_cast<double>(swiglu_times.size());
    run.timing.swiglu_global_work_items = swiglu_values;
    run.timing.q8_quantize_min_us =
        *std::min_element(q8_times.begin(), q8_times.end());
    run.timing.q8_quantize_mean_us =
        std::accumulate(q8_times.begin(), q8_times.end(), 0.0) /
          static_cast<double>(q8_times.size());
    run.timing.q8_quantize_global_work_items =
        expert_count * q6_blocks_per_row;
    run.timing.down.min_us =
        *std::min_element(q6_times.begin(), q6_times.end());
    run.timing.down.mean_us =
        std::accumulate(q6_times.begin(), q6_times.end(), 0.0) /
          static_cast<double>(q6_times.size());
    run.timing.down.effective_packed_gb_s =
        static_cast<double>(q6_rows * q6_blocks_per_row * kQ6KBlockBytes) /
        (run.timing.down.min_us / 1e6) / 1e9;
    run.timing.down.global_work_items = q6_rows;
    run.timing.down.rows_per_work_item = 1;
    run.timing.shell_sum_min_us =
        run.timing.gate_up.min_us + run.timing.swiglu_min_us +
        run.timing.q8_quantize_min_us + run.timing.down.min_us;
    run.timing.shell_sum_mean_us =
        run.timing.gate_up.mean_us + run.timing.swiglu_mean_us +
        run.timing.q8_quantize_mean_us + run.timing.down.mean_us;
    Check(api_.clEnqueueReadBuffer(queue_, down_buffer, kClTrue, 0,
                                   run.down.size() * sizeof(float),
                                   run.down.data(), 0, nullptr, nullptr),
          "clEnqueueReadBuffer(expert8 fused SwiGLU Q6 down)");
    return run;
  }

  GpuQ4X8ResidualRmsNormHandoffRun
  RunResidentPackedQ4X8ThenResidualRmsNorm(
      std::uint64_t handle,
      const std::vector<std::int8_t>& q8_qs,
      const std::vector<std::int16_t>& q8_bsums,
      const std::vector<float>& q8_d,
      const std::vector<float>& residual_input,
      const std::vector<float>& norm_weight,
      float norm_epsilon,
      int repeat,
      GpuQ4X8KernelVariant variant,
      bool use_rowblock16_output_projection = false) {
    const auto& resident = ResidentPackedQ4X8ForHandle(handle);
    ValidateQ8InputPlanes(q8_qs, q8_bsums, q8_d, resident.blocks_per_row, repeat);
    Require(resident.rows == residual_input.size(),
            "resident Q4 residual RMSNorm row count mismatch");
    Require(norm_weight.size() == residual_input.size(),
            "resident Q4 residual RMSNorm weight size mismatch");

    GpuQ4X8ResidualRmsNormHandoffRun run;
    run.residual.assign(residual_input.size(), 0.0f);
    run.normalized.assign(residual_input.size(), 0.0f);

    cl_mem q8_qs_buffer = nullptr;
    cl_mem q8_bsums_buffer = nullptr;
    cl_mem q8_d_buffer = nullptr;
    cl_mem projection_buffer = nullptr;
    cl_mem residual_input_buffer = nullptr;
    cl_mem norm_weight_buffer = nullptr;
    cl_mem residual_buffer = nullptr;
    cl_mem normalized_buffer = nullptr;
    auto release_all = [&]() {
      ReleaseMem(api_, &norm_weight_buffer);
      ReleaseMem(api_, &residual_input_buffer);
      ReleaseMem(api_, &projection_buffer);
      ReleaseMem(api_, &q8_d_buffer);
      ReleaseMem(api_, &q8_bsums_buffer);
      ReleaseMem(api_, &q8_qs_buffer);
    };
    try {
      std::vector<float> projection(static_cast<std::size_t>(resident.rows), 0.0f);
      CreateRunBuffersWithoutPacked(q8_qs, q8_bsums, q8_d, projection,
                                    &q8_qs_buffer, &q8_bsums_buffer,
                                    &q8_d_buffer, &projection_buffer,
                                    kClMemReadWrite);
      cl_int err = kClSuccess;
      residual_input_buffer = api_.clCreateBuffer(
          context_, kClMemReadOnly,
          residual_input.size() * sizeof(float), nullptr, &err);
      Check(err, "clCreateBuffer(Q4 residual RMSNorm residual input)");
      norm_weight_buffer = api_.clCreateBuffer(
          context_, kClMemReadOnly,
          norm_weight.size() * sizeof(float), nullptr, &err);
      Check(err, "clCreateBuffer(Q4 residual RMSNorm weight)");
      residual_buffer = EnsureScratchBuffer(
          attention_front_scratch_residual_,
          run.residual.size() * sizeof(float), kClMemReadWrite,
          "Q4 residual RMSNorm residual");
      normalized_buffer = EnsureScratchBuffer(
          attention_front_scratch_normalized_,
          run.normalized.size() * sizeof(float), kClMemWriteOnly,
          "Q4 residual RMSNorm normalized");
      cl_mem scale_buffer = EnsureScratchBuffer(
          rmsnorm_hidden_scratch_scale_, sizeof(float), kClMemReadWrite,
          "Q4 residual RMSNorm scale");

      Check(api_.clEnqueueWriteBuffer(
                queue_, residual_input_buffer, kClTrue, 0,
                residual_input.size() * sizeof(float), residual_input.data(),
                0, nullptr, nullptr),
            "clEnqueueWriteBuffer(Q4 residual RMSNorm residual input)");
      Check(api_.clEnqueueWriteBuffer(
                queue_, norm_weight_buffer, kClTrue, 0,
                norm_weight.size() * sizeof(float), norm_weight.data(),
                0, nullptr, nullptr),
            "clEnqueueWriteBuffer(Q4 residual RMSNorm weight)");

      const std::uint64_t row_groups = resident.rows / kRowsInterleaved;
      const std::uint64_t matvec_global =
          variant == GpuQ4X8KernelVariant::kGroup8Serial ? row_groups : resident.rows;
      if (use_rowblock16_output_projection) {
        Require(resident.blocks_per_row == 16,
                "rowblock16 attention-front output projection requires BPR16");
        run.timing.matvec = RunRowblock16Kernel(
            resident.buffer, q8_qs_buffer, q8_bsums_buffer, q8_d_buffer,
            projection_buffer, resident.rows, resident.blocks_per_row,
            row_groups, repeat);
      } else {
        run.timing.matvec = RunKernel(
            variant, resident.buffer, q8_qs_buffer, q8_bsums_buffer, q8_d_buffer,
            projection_buffer, resident.blocks_per_row, row_groups, matvec_global,
            repeat);
      }

      const cl_uint hidden_arg = static_cast<cl_uint>(residual_input.size());
      const cl_uint serial_reduction_arg = 0;
      Check(api_.clSetKernelArg(kernel_post_ffn_residual_, 0,
                                sizeof(residual_input_buffer),
                                &residual_input_buffer),
            "clSetKernelArg(Q4 residual RMSNorm residual 0)");
      Check(api_.clSetKernelArg(kernel_post_ffn_residual_, 1,
                                sizeof(projection_buffer),
                                &projection_buffer),
            "clSetKernelArg(Q4 residual RMSNorm residual 1)");
      Check(api_.clSetKernelArg(kernel_post_ffn_residual_, 2,
                                sizeof(hidden_arg), &hidden_arg),
            "clSetKernelArg(Q4 residual RMSNorm residual 2)");
      Check(api_.clSetKernelArg(kernel_post_ffn_residual_, 3,
                                sizeof(residual_buffer), &residual_buffer),
            "clSetKernelArg(Q4 residual RMSNorm residual 3)");
      Check(api_.clSetKernelArg(kernel_rmsnorm_hidden_scale_, 0,
                                sizeof(residual_buffer), &residual_buffer),
            "clSetKernelArg(Q4 residual RMSNorm scale 0)");
      Check(api_.clSetKernelArg(kernel_rmsnorm_hidden_scale_, 1,
                                sizeof(hidden_arg), &hidden_arg),
            "clSetKernelArg(Q4 residual RMSNorm scale 1)");
      Check(api_.clSetKernelArg(kernel_rmsnorm_hidden_scale_, 2,
                                sizeof(norm_epsilon),
                                &norm_epsilon),
            "clSetKernelArg(Q4 residual RMSNorm scale 2)");
      Check(api_.clSetKernelArg(kernel_rmsnorm_hidden_scale_, 3,
                                sizeof(scale_buffer), &scale_buffer),
            "clSetKernelArg(Q4 residual RMSNorm scale 3)");
      Check(api_.clSetKernelArg(kernel_rmsnorm_hidden_scale_, 4,
                                sizeof(serial_reduction_arg),
                                &serial_reduction_arg),
            "clSetKernelArg(Q4 residual RMSNorm scale 4)");
      Check(api_.clSetKernelArg(kernel_rmsnorm_hidden_apply_scale_, 0,
                                sizeof(residual_buffer), &residual_buffer),
            "clSetKernelArg(Q4 residual RMSNorm apply 0)");
      Check(api_.clSetKernelArg(kernel_rmsnorm_hidden_apply_scale_, 1,
                                sizeof(norm_weight_buffer),
                                &norm_weight_buffer),
            "clSetKernelArg(Q4 residual RMSNorm apply 1)");
      Check(api_.clSetKernelArg(kernel_rmsnorm_hidden_apply_scale_, 2,
                                sizeof(hidden_arg), &hidden_arg),
            "clSetKernelArg(Q4 residual RMSNorm apply 2)");
      Check(api_.clSetKernelArg(kernel_rmsnorm_hidden_apply_scale_, 3,
                                sizeof(scale_buffer), &scale_buffer),
            "clSetKernelArg(Q4 residual RMSNorm apply 3)");
      Check(api_.clSetKernelArg(kernel_rmsnorm_hidden_apply_scale_, 4,
                                sizeof(normalized_buffer),
                                &normalized_buffer),
            "clSetKernelArg(Q4 residual RMSNorm apply 4)");

      const std::size_t residual_global = residual_input.size();
      const std::size_t rmsnorm_scale_global = kRmsNormScaleLocalSize;
      const std::size_t rmsnorm_scale_local = kRmsNormScaleLocalSize;
      const std::size_t rmsnorm_apply_global = residual_input.size();
      std::vector<double> residual_times;
      std::vector<double> rmsnorm_times;
      residual_times.reserve(static_cast<std::size_t>(repeat));
      rmsnorm_times.reserve(static_cast<std::size_t>(repeat));
      for (int i = 0; i < repeat; ++i) {
        cl_event residual_event = nullptr;
        cl_event scale_event = nullptr;
        Check(api_.clEnqueueNDRangeKernel(
                  queue_, kernel_post_ffn_residual_, 1, nullptr,
                  &residual_global, nullptr, 0, nullptr, EventOut(&residual_event)),
              "clEnqueueNDRangeKernel(Q4 residual RMSNorm residual)");
        cl_event apply_event = nullptr;
        Check(api_.clEnqueueNDRangeKernel(
                  queue_, kernel_rmsnorm_hidden_scale_, 1, nullptr,
                  &rmsnorm_scale_global, &rmsnorm_scale_local, 0, nullptr,
                  EventOut(&scale_event)),
              "clEnqueueNDRangeKernel(Q4 residual RMSNorm scale)");
        Check(api_.clEnqueueNDRangeKernel(
                  queue_, kernel_rmsnorm_hidden_apply_scale_, 1, nullptr,
                  &rmsnorm_apply_global, nullptr, 0, nullptr, EventOut(&apply_event)),
              "clEnqueueNDRangeKernel(Q4 residual RMSNorm apply)");
        Check(api_.clFinish(queue_), "clFinish(Q4 residual RMSNorm)");
        residual_times.push_back(EventUs(api_, residual_event));
        rmsnorm_times.push_back(EventUs(api_, scale_event) +
                                EventUs(api_, apply_event));
        ReleaseEvent(api_, &apply_event);
        ReleaseEvent(api_, &scale_event);
        ReleaseEvent(api_, &residual_event);
      }

      Check(api_.clEnqueueReadBuffer(queue_, residual_buffer, kClTrue, 0,
                                     run.residual.size() * sizeof(float),
                                     run.residual.data(), 0, nullptr, nullptr),
            "clEnqueueReadBuffer(Q4 residual RMSNorm residual)");
      Check(api_.clEnqueueReadBuffer(queue_, normalized_buffer, kClTrue, 0,
                                     run.normalized.size() * sizeof(float),
                                     run.normalized.data(), 0, nullptr,
                                     nullptr),
            "clEnqueueReadBuffer(Q4 residual RMSNorm normalized)");
      run.residual_handle = RegisterF32BufferAlias(
          &attention_front_residual_alias_handle_, residual_buffer,
          run.residual.size());
      run.normalized_handle = RegisterF32BufferAlias(
          &attention_front_normalized_alias_handle_, normalized_buffer,
          run.normalized.size());

      run.timing.residual_min_us =
          *std::min_element(residual_times.begin(), residual_times.end());
      run.timing.residual_mean_us =
          std::accumulate(residual_times.begin(), residual_times.end(), 0.0) /
          static_cast<double>(residual_times.size());
      run.timing.rmsnorm_min_us =
          *std::min_element(rmsnorm_times.begin(), rmsnorm_times.end());
      run.timing.rmsnorm_mean_us =
          std::accumulate(rmsnorm_times.begin(), rmsnorm_times.end(), 0.0) /
          static_cast<double>(rmsnorm_times.size());
      run.timing.shell_sum_min_us =
          run.timing.matvec.min_us + run.timing.residual_min_us +
          run.timing.rmsnorm_min_us;
      run.timing.shell_sum_mean_us =
          run.timing.matvec.mean_us + run.timing.residual_mean_us +
          run.timing.rmsnorm_mean_us;
      run.timing.residual_global_work_items = residual_global;
      run.timing.rmsnorm_global_work_items =
          rmsnorm_scale_global + rmsnorm_apply_global;
    } catch (...) {
      release_all();
      throw;
    }
    release_all();
    return run;
  }

  GpuQ4X8ResidualRmsNormHandoffRun
  RunResidentPackedQ4X8ThenResidentResidualRmsNorm(
      std::uint64_t handle,
      const std::vector<std::int8_t>& q8_qs,
      const std::vector<std::int16_t>& q8_bsums,
      const std::vector<float>& q8_d,
      const std::vector<float>& residual_input,
      std::uint64_t norm_weight_handle,
      std::uint64_t hidden_size,
      float norm_epsilon,
      int repeat,
      GpuQ4X8KernelVariant variant,
      std::uint64_t residual_input_handle = 0,
      bool use_rowblock16_output_projection = false) {
    const auto& resident = ResidentPackedQ4X8ForHandle(handle);
    const auto& norm_weight = ResidentF32BufferForHandle(norm_weight_handle);
    const ResidentF32Buffer* resident_residual_input = nullptr;
    if (residual_input_handle != 0) {
      resident_residual_input = &ResidentF32BufferForHandle(residual_input_handle);
    }
    ValidateQ8InputPlanes(q8_qs, q8_bsums, q8_d, resident.blocks_per_row, repeat);
    Require(resident.rows == hidden_size,
            "resident Q4 resident RMSNorm row count mismatch");
    Require(residual_input.size() == hidden_size ||
                resident_residual_input != nullptr,
            "resident Q4 resident RMSNorm residual size mismatch");
    if (resident_residual_input != nullptr) {
      Require(resident_residual_input->values == hidden_size,
              "resident Q4 resident RMSNorm residual handle size mismatch");
    }
    Require(norm_weight.values == hidden_size,
            "resident Q4 resident RMSNorm weight size mismatch");

    const bool handoff_wall_split_profile =
        std::getenv("IQ36_ATTENTION_FRONT_HANDOFF_WALL_SPLIT_PROFILE") != nullptr;
    const auto setup_begin = handoff_wall_split_profile
                                 ? std::chrono::steady_clock::now()
                                 : std::chrono::steady_clock::time_point{};
    GpuQ4X8ResidualRmsNormHandoffRun run;
    run.residual.assign(static_cast<std::size_t>(hidden_size), 0.0f);
    run.normalized.assign(static_cast<std::size_t>(hidden_size), 0.0f);

    cl_mem q8_qs_buffer = nullptr;
    cl_mem q8_bsums_buffer = nullptr;
    cl_mem q8_d_buffer = nullptr;
    cl_mem projection_buffer = nullptr;
    cl_mem residual_input_buffer = nullptr;
    cl_mem residual_buffer = nullptr;
    cl_mem normalized_buffer = nullptr;
    auto release_all = [&]() {
      if (resident_residual_input == nullptr) {
        ReleaseMem(api_, &residual_input_buffer);
      }
      ReleaseMem(api_, &projection_buffer);
      ReleaseMem(api_, &q8_d_buffer);
      ReleaseMem(api_, &q8_bsums_buffer);
      ReleaseMem(api_, &q8_qs_buffer);
    };
    try {
      std::vector<float> projection(static_cast<std::size_t>(resident.rows), 0.0f);
      CreateRunBuffersWithoutPacked(q8_qs, q8_bsums, q8_d, projection,
                                    &q8_qs_buffer, &q8_bsums_buffer,
                                    &q8_d_buffer, &projection_buffer,
                                    kClMemReadWrite);
      if (resident_residual_input != nullptr) {
        residual_input_buffer = resident_residual_input->buffer;
      } else {
        cl_int err = kClSuccess;
        residual_input_buffer = api_.clCreateBuffer(
            context_, kClMemReadOnly,
            residual_input.size() * sizeof(float), nullptr, &err);
        Check(err, "clCreateBuffer(Q4 resident RMSNorm residual input)");
      }
      residual_buffer = EnsureScratchBuffer(
          attention_front_scratch_residual_,
          run.residual.size() * sizeof(float), kClMemWriteOnly,
          "Q4 resident RMSNorm residual");
      normalized_buffer = EnsureScratchBuffer(
          attention_front_scratch_normalized_,
          run.normalized.size() * sizeof(float), kClMemWriteOnly,
          "Q4 resident RMSNorm normalized");
      cl_mem scale_buffer = EnsureScratchBuffer(
          rmsnorm_hidden_scratch_scale_, sizeof(float), kClMemReadWrite,
          "Q4 resident RMSNorm scale");
      if (handoff_wall_split_profile) {
        run.timing.handoff_setup_wall_ns +=
            WallNs(setup_begin, std::chrono::steady_clock::now());
      }

      const auto residual_input_write_begin =
          handoff_wall_split_profile
              ? std::chrono::steady_clock::now()
              : std::chrono::steady_clock::time_point{};
      if (resident_residual_input == nullptr) {
        Check(api_.clEnqueueWriteBuffer(
                  queue_, residual_input_buffer, kClTrue, 0,
                  residual_input.size() * sizeof(float), residual_input.data(),
                  0, nullptr, nullptr),
              "clEnqueueWriteBuffer(Q4 resident RMSNorm residual input)");
      }
      if (handoff_wall_split_profile) {
        run.timing.handoff_residual_input_write_wall_ns += WallNs(
            residual_input_write_begin, std::chrono::steady_clock::now());
      }

      const std::uint64_t row_groups = resident.rows / kRowsInterleaved;
      const std::uint64_t matvec_global =
          variant == GpuQ4X8KernelVariant::kGroup8Serial ? row_groups : resident.rows;
      const auto matvec_begin = handoff_wall_split_profile
                                    ? std::chrono::steady_clock::now()
                                    : std::chrono::steady_clock::time_point{};
      if (use_rowblock16_output_projection) {
        Require(resident.blocks_per_row == 16,
                "rowblock16 attention-front output projection requires BPR16");
        run.timing.matvec = RunRowblock16Kernel(
            resident.buffer, q8_qs_buffer, q8_bsums_buffer, q8_d_buffer,
            projection_buffer, resident.rows, resident.blocks_per_row,
            row_groups, repeat);
      } else {
        run.timing.matvec = RunKernel(
            variant, resident.buffer, q8_qs_buffer, q8_bsums_buffer, q8_d_buffer,
            projection_buffer, resident.blocks_per_row, row_groups, matvec_global,
            repeat);
      }
      if (handoff_wall_split_profile) {
        run.timing.handoff_matvec_wall_ns +=
            WallNs(matvec_begin, std::chrono::steady_clock::now());
      }

      const cl_uint hidden_arg = static_cast<cl_uint>(hidden_size);
      const cl_uint serial_reduction_arg = 0;
      const auto residual_rmsnorm_args_begin =
          handoff_wall_split_profile
              ? std::chrono::steady_clock::now()
              : std::chrono::steady_clock::time_point{};
      Check(api_.clSetKernelArg(kernel_post_ffn_residual_, 0,
                                sizeof(residual_input_buffer),
                                &residual_input_buffer),
            "clSetKernelArg(Q4 resident RMSNorm residual 0)");
      Check(api_.clSetKernelArg(kernel_post_ffn_residual_, 1,
                                sizeof(projection_buffer),
                                &projection_buffer),
            "clSetKernelArg(Q4 resident RMSNorm residual 1)");
      Check(api_.clSetKernelArg(kernel_post_ffn_residual_, 2,
                                sizeof(hidden_arg), &hidden_arg),
            "clSetKernelArg(Q4 resident RMSNorm residual 2)");
      Check(api_.clSetKernelArg(kernel_post_ffn_residual_, 3,
                                sizeof(residual_buffer), &residual_buffer),
            "clSetKernelArg(Q4 resident RMSNorm residual 3)");
      Check(api_.clSetKernelArg(kernel_rmsnorm_hidden_scale_, 0,
                                sizeof(residual_buffer), &residual_buffer),
            "clSetKernelArg(Q4 resident RMSNorm scale 0)");
      Check(api_.clSetKernelArg(kernel_rmsnorm_hidden_scale_, 1,
                                sizeof(hidden_arg), &hidden_arg),
            "clSetKernelArg(Q4 resident RMSNorm scale 1)");
      Check(api_.clSetKernelArg(kernel_rmsnorm_hidden_scale_, 2,
                                sizeof(norm_epsilon),
                                &norm_epsilon),
            "clSetKernelArg(Q4 resident RMSNorm scale 2)");
      Check(api_.clSetKernelArg(kernel_rmsnorm_hidden_scale_, 3,
                                sizeof(scale_buffer), &scale_buffer),
            "clSetKernelArg(Q4 resident RMSNorm scale 3)");
      Check(api_.clSetKernelArg(kernel_rmsnorm_hidden_scale_, 4,
                                sizeof(serial_reduction_arg),
                                &serial_reduction_arg),
            "clSetKernelArg(Q4 resident RMSNorm scale 4)");
      Check(api_.clSetKernelArg(kernel_rmsnorm_hidden_apply_scale_, 0,
                                sizeof(residual_buffer), &residual_buffer),
            "clSetKernelArg(Q4 resident RMSNorm apply 0)");
      Check(api_.clSetKernelArg(kernel_rmsnorm_hidden_apply_scale_, 1,
                                sizeof(norm_weight.buffer),
                                &norm_weight.buffer),
            "clSetKernelArg(Q4 resident RMSNorm apply 1)");
      Check(api_.clSetKernelArg(kernel_rmsnorm_hidden_apply_scale_, 2,
                                sizeof(hidden_arg), &hidden_arg),
            "clSetKernelArg(Q4 resident RMSNorm apply 2)");
      Check(api_.clSetKernelArg(kernel_rmsnorm_hidden_apply_scale_, 3,
                                sizeof(scale_buffer), &scale_buffer),
            "clSetKernelArg(Q4 resident RMSNorm apply 3)");
      Check(api_.clSetKernelArg(kernel_rmsnorm_hidden_apply_scale_, 4,
                                sizeof(normalized_buffer),
                                &normalized_buffer),
            "clSetKernelArg(Q4 resident RMSNorm apply 4)");
      if (handoff_wall_split_profile) {
        run.timing.handoff_residual_rmsnorm_args_wall_ns += WallNs(
            residual_rmsnorm_args_begin, std::chrono::steady_clock::now());
      }

      const std::size_t residual_global = static_cast<std::size_t>(hidden_size);
      const std::size_t rmsnorm_scale_global = kRmsNormScaleLocalSize;
      const std::size_t rmsnorm_scale_local = kRmsNormScaleLocalSize;
      const std::size_t rmsnorm_apply_global =
          static_cast<std::size_t>(hidden_size);
      std::vector<double> residual_times;
      std::vector<double> rmsnorm_times;
      residual_times.reserve(static_cast<std::size_t>(repeat));
      rmsnorm_times.reserve(static_cast<std::size_t>(repeat));
      for (int i = 0; i < repeat; ++i) {
        cl_event residual_event = nullptr;
        cl_event scale_event = nullptr;
        const auto enqueue_finish_begin =
            handoff_wall_split_profile
                ? std::chrono::steady_clock::now()
                : std::chrono::steady_clock::time_point{};
        Check(api_.clEnqueueNDRangeKernel(
                  queue_, kernel_post_ffn_residual_, 1, nullptr,
                  &residual_global, nullptr, 0, nullptr, EventOut(&residual_event)),
              "clEnqueueNDRangeKernel(Q4 resident RMSNorm residual)");
        cl_event apply_event = nullptr;
        Check(api_.clEnqueueNDRangeKernel(
                  queue_, kernel_rmsnorm_hidden_scale_, 1, nullptr,
                  &rmsnorm_scale_global, &rmsnorm_scale_local, 0, nullptr,
                  EventOut(&scale_event)),
              "clEnqueueNDRangeKernel(Q4 resident RMSNorm scale)");
        Check(api_.clEnqueueNDRangeKernel(
                  queue_, kernel_rmsnorm_hidden_apply_scale_, 1, nullptr,
                  &rmsnorm_apply_global, nullptr, 0, nullptr, EventOut(&apply_event)),
              "clEnqueueNDRangeKernel(Q4 resident RMSNorm apply)");
        Check(api_.clFinish(queue_), "clFinish(Q4 resident RMSNorm)");
        if (handoff_wall_split_profile) {
          run.timing.handoff_residual_rmsnorm_enqueue_finish_wall_ns += WallNs(
              enqueue_finish_begin, std::chrono::steady_clock::now());
        }
        const auto event_profile_begin =
            handoff_wall_split_profile
                ? std::chrono::steady_clock::now()
                : std::chrono::steady_clock::time_point{};
        residual_times.push_back(EventUs(api_, residual_event));
        rmsnorm_times.push_back(EventUs(api_, scale_event) +
                                EventUs(api_, apply_event));
        ReleaseEvent(api_, &apply_event);
        ReleaseEvent(api_, &scale_event);
        ReleaseEvent(api_, &residual_event);
        if (handoff_wall_split_profile) {
          run.timing.handoff_event_profile_wall_ns +=
              WallNs(event_profile_begin, std::chrono::steady_clock::now());
        }
      }

      const auto residual_read_begin =
          handoff_wall_split_profile
              ? std::chrono::steady_clock::now()
              : std::chrono::steady_clock::time_point{};
      Check(api_.clEnqueueReadBuffer(queue_, residual_buffer, kClTrue, 0,
                                     run.residual.size() * sizeof(float),
                                     run.residual.data(), 0, nullptr, nullptr),
            "clEnqueueReadBuffer(Q4 resident RMSNorm residual)");
      if (handoff_wall_split_profile) {
        run.timing.handoff_residual_read_wall_ns +=
            WallNs(residual_read_begin, std::chrono::steady_clock::now());
      }
      const auto alias_begin = handoff_wall_split_profile
                                   ? std::chrono::steady_clock::now()
                                   : std::chrono::steady_clock::time_point{};
      run.residual_handle = RegisterF32BufferAlias(
          &attention_front_residual_alias_handle_, residual_buffer,
          run.residual.size());
      run.normalized_handle = RegisterF32BufferAlias(
          &attention_front_normalized_alias_handle_, normalized_buffer,
          run.normalized.size());
      if (handoff_wall_split_profile) {
        run.timing.handoff_alias_wall_ns +=
            WallNs(alias_begin, std::chrono::steady_clock::now());
      }
      const auto normalized_read_begin =
          handoff_wall_split_profile
              ? std::chrono::steady_clock::now()
              : std::chrono::steady_clock::time_point{};
      Check(api_.clEnqueueReadBuffer(queue_, normalized_buffer, kClTrue, 0,
                                     run.normalized.size() * sizeof(float),
                                     run.normalized.data(), 0, nullptr,
                                     nullptr),
            "clEnqueueReadBuffer(Q4 resident RMSNorm normalized)");
      if (handoff_wall_split_profile) {
        run.timing.handoff_normalized_read_wall_ns +=
            WallNs(normalized_read_begin, std::chrono::steady_clock::now());
      }
      run.timing.residual_min_us =
          *std::min_element(residual_times.begin(), residual_times.end());
      run.timing.residual_mean_us =
          std::accumulate(residual_times.begin(), residual_times.end(), 0.0) /
          static_cast<double>(residual_times.size());
      run.timing.rmsnorm_min_us =
          *std::min_element(rmsnorm_times.begin(), rmsnorm_times.end());
      run.timing.rmsnorm_mean_us =
          std::accumulate(rmsnorm_times.begin(), rmsnorm_times.end(), 0.0) /
          static_cast<double>(rmsnorm_times.size());
      run.timing.shell_sum_min_us =
          run.timing.matvec.min_us + run.timing.residual_min_us +
          run.timing.rmsnorm_min_us;
      run.timing.shell_sum_mean_us =
          run.timing.matvec.mean_us + run.timing.residual_mean_us +
          run.timing.rmsnorm_mean_us;
      run.timing.residual_global_work_items = residual_global;
      run.timing.rmsnorm_global_work_items =
          rmsnorm_scale_global + rmsnorm_apply_global;
    } catch (...) {
      release_all();
      throw;
    }
    const auto release_begin = handoff_wall_split_profile
                                   ? std::chrono::steady_clock::now()
                                   : std::chrono::steady_clock::time_point{};
    release_all();
    if (handoff_wall_split_profile) {
      run.timing.handoff_release_wall_ns +=
          WallNs(release_begin, std::chrono::steady_clock::now());
    }
    return run;
  }

  GpuQ4X8ResidualRmsNormHandoffRun
  RunF32DeviceQ8ThenResidentPackedQ4X8ResidualRmsNorm(
      std::uint64_t handle,
      const std::vector<float>& input,
      const std::vector<float>& residual_input,
      const std::vector<float>& norm_weight,
      float norm_epsilon,
      int repeat,
      GpuQ4X8KernelVariant variant) {
    const auto& resident = ResidentPackedQ4X8ForHandle(handle);
    Require(input.size() == resident.blocks_per_row * kQ8QsPerBlock,
            "device Q8 residual RMSNorm input size mismatch");
    Require(resident.rows == residual_input.size(),
            "device Q8 residual RMSNorm row count mismatch");
    Require(norm_weight.size() == residual_input.size(),
            "device Q8 residual RMSNorm weight size mismatch");

    GpuQ4X8ResidualRmsNormHandoffRun run;
    run.residual.assign(residual_input.size(), 0.0f);
    run.normalized.assign(residual_input.size(), 0.0f);

    cl_mem input_buffer = nullptr;
    cl_mem q8_qs_buffer = nullptr;
    cl_mem q8_bsums_buffer = nullptr;
    cl_mem q8_d_buffer = nullptr;
    cl_mem projection_buffer = nullptr;
    cl_mem residual_input_buffer = nullptr;
    cl_mem norm_weight_buffer = nullptr;
    cl_mem residual_buffer = nullptr;
    cl_mem normalized_buffer = nullptr;
    auto release_all = [&]() {
      ReleaseMem(api_, &normalized_buffer);
      ReleaseMem(api_, &residual_buffer);
      ReleaseMem(api_, &norm_weight_buffer);
      ReleaseMem(api_, &residual_input_buffer);
      ReleaseMem(api_, &projection_buffer);
      ReleaseMem(api_, &q8_d_buffer);
      ReleaseMem(api_, &q8_bsums_buffer);
      ReleaseMem(api_, &q8_qs_buffer);
      ReleaseMem(api_, &input_buffer);
    };
    try {
      cl_int err = kClSuccess;
      input_buffer = api_.clCreateBuffer(
          context_, kClMemReadOnly, input.size() * sizeof(float), nullptr,
          &err);
      Check(err, "clCreateBuffer(device Q8 input)");
      const std::uint64_t block_count = resident.blocks_per_row;
      q8_qs_buffer = api_.clCreateBuffer(
          context_, kClMemReadWrite,
          block_count * kQ8QsPerBlock * sizeof(std::int8_t), nullptr, &err);
      Check(err, "clCreateBuffer(device Q8 qs)");
      q8_bsums_buffer = api_.clCreateBuffer(
          context_, kClMemReadWrite,
          block_count * kQ8BsumsPerBlock * sizeof(std::int16_t), nullptr,
          &err);
      Check(err, "clCreateBuffer(device Q8 bsums)");
      q8_d_buffer = api_.clCreateBuffer(
          context_, kClMemReadWrite, block_count * sizeof(float), nullptr,
          &err);
      Check(err, "clCreateBuffer(device Q8 d)");
      projection_buffer = api_.clCreateBuffer(
          context_, kClMemReadWrite, resident.rows * sizeof(float), nullptr,
          &err);
      Check(err, "clCreateBuffer(device Q8 projection)");
      residual_input_buffer = api_.clCreateBuffer(
          context_, kClMemReadOnly,
          residual_input.size() * sizeof(float), nullptr, &err);
      Check(err, "clCreateBuffer(device Q8 residual input)");
      norm_weight_buffer = api_.clCreateBuffer(
          context_, kClMemReadOnly,
          norm_weight.size() * sizeof(float), nullptr, &err);
      Check(err, "clCreateBuffer(device Q8 norm weight)");
      residual_buffer = api_.clCreateBuffer(
          context_, kClMemReadWrite, run.residual.size() * sizeof(float),
          nullptr, &err);
      Check(err, "clCreateBuffer(device Q8 residual)");
      normalized_buffer = api_.clCreateBuffer(
          context_, kClMemWriteOnly, run.normalized.size() * sizeof(float),
          nullptr, &err);
      Check(err, "clCreateBuffer(device Q8 normalized)");

      Check(api_.clEnqueueWriteBuffer(
                queue_, input_buffer, kClTrue, 0,
                input.size() * sizeof(float), input.data(), 0, nullptr,
                nullptr),
            "clEnqueueWriteBuffer(device Q8 input)");
      Check(api_.clEnqueueWriteBuffer(
                queue_, residual_input_buffer, kClTrue, 0,
                residual_input.size() * sizeof(float), residual_input.data(),
                0, nullptr, nullptr),
            "clEnqueueWriteBuffer(device Q8 residual input)");
      Check(api_.clEnqueueWriteBuffer(
                queue_, norm_weight_buffer, kClTrue, 0,
                norm_weight.size() * sizeof(float), norm_weight.data(), 0,
                nullptr, nullptr),
            "clEnqueueWriteBuffer(device Q8 norm weight)");

      const auto q8_timing = RunQ8QuantizeWithBsumsKernel(
          input_buffer, block_count, q8_qs_buffer, q8_bsums_buffer,
          q8_d_buffer, repeat);
      const std::uint64_t row_groups = resident.rows / kRowsInterleaved;
      const std::uint64_t matvec_global =
          variant == GpuQ4X8KernelVariant::kGroup8Serial ? row_groups
                                                         : resident.rows;
      run.timing.matvec = RunKernel(
          variant, resident.buffer, q8_qs_buffer, q8_bsums_buffer, q8_d_buffer,
          projection_buffer, resident.blocks_per_row, row_groups,
          matvec_global, repeat);

      const cl_uint hidden_arg = static_cast<cl_uint>(residual_input.size());
      Check(api_.clSetKernelArg(kernel_post_ffn_residual_, 0,
                                sizeof(residual_input_buffer),
                                &residual_input_buffer),
            "clSetKernelArg(device Q8 residual 0)");
      Check(api_.clSetKernelArg(kernel_post_ffn_residual_, 1,
                                sizeof(projection_buffer),
                                &projection_buffer),
            "clSetKernelArg(device Q8 residual 1)");
      Check(api_.clSetKernelArg(kernel_post_ffn_residual_, 2,
                                sizeof(hidden_arg), &hidden_arg),
            "clSetKernelArg(device Q8 residual 2)");
      Check(api_.clSetKernelArg(kernel_post_ffn_residual_, 3,
                                sizeof(residual_buffer), &residual_buffer),
            "clSetKernelArg(device Q8 residual 3)");
      Check(api_.clSetKernelArg(kernel_rmsnorm_hidden_, 0,
                                sizeof(residual_buffer), &residual_buffer),
            "clSetKernelArg(device Q8 norm 0)");
      Check(api_.clSetKernelArg(kernel_rmsnorm_hidden_, 1,
                                sizeof(norm_weight_buffer),
                                &norm_weight_buffer),
            "clSetKernelArg(device Q8 norm 1)");
      Check(api_.clSetKernelArg(kernel_rmsnorm_hidden_, 2,
                                sizeof(hidden_arg), &hidden_arg),
            "clSetKernelArg(device Q8 norm 2)");
      Check(api_.clSetKernelArg(kernel_rmsnorm_hidden_, 3, sizeof(norm_epsilon),
                                &norm_epsilon),
            "clSetKernelArg(device Q8 norm 3)");
      Check(api_.clSetKernelArg(kernel_rmsnorm_hidden_, 4,
                                sizeof(normalized_buffer),
                                &normalized_buffer),
            "clSetKernelArg(device Q8 norm 4)");

      const std::size_t residual_global = residual_input.size();
      const std::size_t rmsnorm_global = 1;
      std::vector<double> residual_times;
      std::vector<double> rmsnorm_times;
      residual_times.reserve(static_cast<std::size_t>(repeat));
      rmsnorm_times.reserve(static_cast<std::size_t>(repeat));
      for (int i = 0; i < repeat; ++i) {
        cl_event residual_event = nullptr;
        cl_event rmsnorm_event = nullptr;
        Check(api_.clEnqueueNDRangeKernel(
                  queue_, kernel_post_ffn_residual_, 1, nullptr,
                  &residual_global, nullptr, 0, nullptr, EventOut(&residual_event)),
              "clEnqueueNDRangeKernel(device Q8 residual)");
        Check(api_.clEnqueueNDRangeKernel(
                  queue_, kernel_rmsnorm_hidden_, 1, nullptr, &rmsnorm_global,
                  nullptr, 0, nullptr, EventOut(&rmsnorm_event)),
              "clEnqueueNDRangeKernel(device Q8 norm)");
        Check(api_.clFinish(queue_), "clFinish(device Q8 residual/norm)");
        residual_times.push_back(EventUs(api_, residual_event));
        rmsnorm_times.push_back(EventUs(api_, rmsnorm_event));
        ReleaseEvent(api_, &residual_event);
        ReleaseEvent(api_, &rmsnorm_event);
      }

      Check(api_.clEnqueueReadBuffer(queue_, residual_buffer, kClTrue, 0,
                                     run.residual.size() * sizeof(float),
                                     run.residual.data(), 0, nullptr,
                                     nullptr),
            "clEnqueueReadBuffer(device Q8 residual)");
      Check(api_.clEnqueueReadBuffer(queue_, normalized_buffer, kClTrue, 0,
                                     run.normalized.size() * sizeof(float),
                                     run.normalized.data(), 0, nullptr,
                                     nullptr),
            "clEnqueueReadBuffer(device Q8 normalized)");

      run.timing.q8_quantize_min_us = q8_timing.min_us;
      run.timing.q8_quantize_mean_us = q8_timing.mean_us;
      run.timing.residual_min_us =
          *std::min_element(residual_times.begin(), residual_times.end());
      run.timing.residual_mean_us =
          std::accumulate(residual_times.begin(), residual_times.end(), 0.0) /
          static_cast<double>(residual_times.size());
      run.timing.rmsnorm_min_us =
          *std::min_element(rmsnorm_times.begin(), rmsnorm_times.end());
      run.timing.rmsnorm_mean_us =
          std::accumulate(rmsnorm_times.begin(), rmsnorm_times.end(), 0.0) /
          static_cast<double>(rmsnorm_times.size());
      run.timing.shell_sum_min_us =
          run.timing.q8_quantize_min_us + run.timing.matvec.min_us +
          run.timing.residual_min_us + run.timing.rmsnorm_min_us;
      run.timing.shell_sum_mean_us =
          run.timing.q8_quantize_mean_us + run.timing.matvec.mean_us +
          run.timing.residual_mean_us + run.timing.rmsnorm_mean_us;
      run.timing.q8_quantize_global_work_items =
          q8_timing.global_work_items;
      run.timing.residual_global_work_items = residual_global;
      run.timing.rmsnorm_global_work_items = rmsnorm_global;
    } catch (...) {
      release_all();
      throw;
    }
    release_all();
    return run;
  }

  GpuQ4X8ResidualRmsNormHandoffRun
  RunF32InputHandleDeviceQ8ThenResidentPackedQ4X8ResidualRmsNorm(
      std::uint64_t handle,
      std::uint64_t input_handle,
      const std::vector<float>& residual_input,
      const std::vector<float>& norm_weight,
      std::uint64_t norm_weight_handle,
      float norm_epsilon,
      int repeat,
      GpuQ4X8KernelVariant variant,
      std::uint64_t residual_input_handle = 0,
      bool use_rowblock16_cpuorder_finalize = false) {
    const auto& resident = ResidentPackedQ4X8ForHandle(handle);
    const auto& input = ResidentF32BufferForHandle(input_handle);
    const ResidentF32Buffer* resident_residual_input = nullptr;
    if (residual_input_handle != 0) {
      resident_residual_input = &ResidentF32BufferForHandle(residual_input_handle);
    }
    Require(input.values == resident.blocks_per_row * kQ8QsPerBlock,
            "device Q8 input-handle residual RMSNorm input size mismatch");
    Require(resident.rows == residual_input.size() ||
                resident_residual_input != nullptr,
            "device Q8 input-handle residual RMSNorm row count mismatch");
    if (resident_residual_input != nullptr) {
      Require(resident_residual_input->values == resident.rows,
              "device Q8 input-handle residual handle size mismatch");
    }
    if (norm_weight_handle != 0) {
      const auto& resident_norm = ResidentF32BufferForHandle(norm_weight_handle);
      Require(resident_norm.values == residual_input.size(),
              "device Q8 input-handle resident norm weight size mismatch");
    } else {
      Require(norm_weight.size() == residual_input.size(),
              "device Q8 input-handle norm weight size mismatch");
    }

    GpuQ4X8ResidualRmsNormHandoffRun run;
    run.residual.assign(residual_input.size(), 0.0f);
    run.normalized.assign(residual_input.size(), 0.0f);

    try {
      const std::uint64_t block_count = resident.blocks_per_row;
      cl_mem q8_qs_buffer = EnsureScratchBuffer(
          attention_front_scratch_q8_qs_,
          block_count * kQ8QsPerBlock * sizeof(std::int8_t),
          kClMemReadWrite, "device Q8 input-handle qs");
      cl_mem q8_bsums_buffer = EnsureScratchBuffer(
          attention_front_scratch_q8_bsums_,
          block_count * kQ8BsumsPerBlock * sizeof(std::int16_t),
          kClMemReadWrite, "device Q8 input-handle bsums");
      cl_mem q8_d_buffer = EnsureScratchBuffer(
          attention_front_scratch_q8_d_, block_count * sizeof(float),
          kClMemReadWrite, "device Q8 input-handle d");
      cl_mem projection_buffer = EnsureScratchBuffer(
          attention_front_scratch_projection_,
          resident.rows * sizeof(float), kClMemReadWrite,
          "device Q8 input-handle projection");
      cl_mem residual_input_buffer =
          resident_residual_input != nullptr
              ? resident_residual_input->buffer
              : EnsureScratchBuffer(
                    attention_front_scratch_residual_input_,
                    residual_input.size() * sizeof(float), kClMemReadOnly,
                    "device Q8 input-handle residual input");
      const ResidentF32Buffer* resident_norm = nullptr;
      cl_mem norm_weight_buffer = nullptr;
      if (norm_weight_handle != 0) {
        resident_norm = &ResidentF32BufferForHandle(norm_weight_handle);
      } else {
        norm_weight_buffer = EnsureScratchBuffer(
            attention_front_scratch_norm_weight_,
            norm_weight.size() * sizeof(float), kClMemReadOnly,
            "device Q8 input-handle norm weight");
      }
      cl_mem residual_buffer = EnsureScratchBuffer(
          attention_front_scratch_residual_,
          run.residual.size() * sizeof(float), kClMemReadWrite,
          "device Q8 input-handle residual");
      cl_mem normalized_buffer = EnsureScratchBuffer(
          attention_front_scratch_normalized_,
          run.normalized.size() * sizeof(float), kClMemWriteOnly,
          "device Q8 input-handle normalized");

      if (resident_residual_input == nullptr) {
        Check(api_.clEnqueueWriteBuffer(
                  queue_, residual_input_buffer, kClTrue, 0,
                  residual_input.size() * sizeof(float), residual_input.data(),
                  0, nullptr, nullptr),
              "clEnqueueWriteBuffer(device Q8 input-handle residual input)");
      }
      if (resident_norm == nullptr) {
        Check(api_.clEnqueueWriteBuffer(
                  queue_, norm_weight_buffer, kClTrue, 0,
                  norm_weight.size() * sizeof(float), norm_weight.data(), 0,
                  nullptr, nullptr),
              "clEnqueueWriteBuffer(device Q8 input-handle norm weight)");
      }

      const auto q8_timing = RunQ8QuantizeWithBsumsKernel(
          input.buffer, block_count, q8_qs_buffer, q8_bsums_buffer,
          q8_d_buffer, repeat);
      const std::uint64_t row_groups = resident.rows / kRowsInterleaved;
      const std::uint64_t matvec_global =
          variant == GpuQ4X8KernelVariant::kGroup8Serial ? row_groups
                                                         : resident.rows;
      if (use_rowblock16_cpuorder_finalize) {
        Require(
            resident.blocks_per_row == 16,
            "rowblock16 CPU-order input-handle projection requires BPR16");
        run.timing.matvec = RunRowblock16Kernel(
            resident.buffer, q8_qs_buffer, q8_bsums_buffer, q8_d_buffer,
            projection_buffer, resident.rows, resident.blocks_per_row,
            row_groups, repeat, true);
      } else {
        run.timing.matvec = RunKernel(
            variant, resident.buffer, q8_qs_buffer, q8_bsums_buffer,
            q8_d_buffer, projection_buffer, resident.blocks_per_row,
            row_groups, matvec_global, repeat);
      }

      const cl_uint hidden_arg = static_cast<cl_uint>(residual_input.size());
      Check(api_.clSetKernelArg(kernel_post_ffn_residual_, 0,
                                sizeof(residual_input_buffer),
                                &residual_input_buffer),
            "clSetKernelArg(device Q8 input-handle residual 0)");
      Check(api_.clSetKernelArg(kernel_post_ffn_residual_, 1,
                                sizeof(projection_buffer),
                                &projection_buffer),
            "clSetKernelArg(device Q8 input-handle residual 1)");
      Check(api_.clSetKernelArg(kernel_post_ffn_residual_, 2,
                                sizeof(hidden_arg), &hidden_arg),
            "clSetKernelArg(device Q8 input-handle residual 2)");
      Check(api_.clSetKernelArg(kernel_post_ffn_residual_, 3,
                                sizeof(residual_buffer), &residual_buffer),
            "clSetKernelArg(device Q8 input-handle residual 3)");
      const cl_mem norm_buffer =
          resident_norm != nullptr ? resident_norm->buffer : norm_weight_buffer;
      Check(api_.clSetKernelArg(kernel_rmsnorm_hidden_, 0,
                                sizeof(residual_buffer), &residual_buffer),
            "clSetKernelArg(device Q8 input-handle norm 0)");
      Check(api_.clSetKernelArg(kernel_rmsnorm_hidden_, 1,
                                sizeof(norm_buffer), &norm_buffer),
            "clSetKernelArg(device Q8 input-handle norm 1)");
      Check(api_.clSetKernelArg(kernel_rmsnorm_hidden_, 2,
                                sizeof(hidden_arg), &hidden_arg),
            "clSetKernelArg(device Q8 input-handle norm 2)");
      Check(api_.clSetKernelArg(kernel_rmsnorm_hidden_, 3, sizeof(norm_epsilon),
                                &norm_epsilon),
            "clSetKernelArg(device Q8 input-handle norm 3)");
      Check(api_.clSetKernelArg(kernel_rmsnorm_hidden_, 4,
                                sizeof(normalized_buffer),
                                &normalized_buffer),
            "clSetKernelArg(device Q8 input-handle norm 4)");

      const std::size_t residual_global = residual_input.size();
      const std::size_t rmsnorm_global = 1;
      std::vector<double> residual_times;
      std::vector<double> rmsnorm_times;
      residual_times.reserve(static_cast<std::size_t>(repeat));
      rmsnorm_times.reserve(static_cast<std::size_t>(repeat));
      for (int i = 0; i < repeat; ++i) {
        cl_event residual_event = nullptr;
        cl_event rmsnorm_event = nullptr;
        Check(api_.clEnqueueNDRangeKernel(
                  queue_, kernel_post_ffn_residual_, 1, nullptr,
                  &residual_global, nullptr, 0, nullptr,
                  EventOut(&residual_event)),
              "clEnqueueNDRangeKernel(device Q8 input-handle residual)");
        Check(api_.clEnqueueNDRangeKernel(
                  queue_, kernel_rmsnorm_hidden_, 1, nullptr, &rmsnorm_global,
                  nullptr, 0, nullptr, EventOut(&rmsnorm_event)),
              "clEnqueueNDRangeKernel(device Q8 input-handle norm)");
        Check(api_.clFinish(queue_), "clFinish(device Q8 input-handle residual/norm)");
        residual_times.push_back(EventUs(api_, residual_event));
        rmsnorm_times.push_back(EventUs(api_, rmsnorm_event));
        ReleaseEvent(api_, &residual_event);
        ReleaseEvent(api_, &rmsnorm_event);
      }

      Check(api_.clEnqueueReadBuffer(queue_, residual_buffer, kClTrue, 0,
                                     run.residual.size() * sizeof(float),
                                     run.residual.data(), 0, nullptr,
                                     nullptr),
            "clEnqueueReadBuffer(device Q8 input-handle residual)");
      Check(api_.clEnqueueReadBuffer(queue_, normalized_buffer, kClTrue, 0,
                                     run.normalized.size() * sizeof(float),
                                     run.normalized.data(), 0, nullptr,
                                     nullptr),
            "clEnqueueReadBuffer(device Q8 input-handle normalized)");
      run.residual_handle = RegisterF32BufferAlias(
          &attention_front_residual_alias_handle_, residual_buffer,
          run.residual.size());
      run.normalized_handle = RegisterF32BufferAlias(
          &attention_front_normalized_alias_handle_, normalized_buffer,
          run.normalized.size());

      run.timing.q8_quantize_min_us = q8_timing.min_us;
      run.timing.q8_quantize_mean_us = q8_timing.mean_us;
      run.timing.residual_min_us =
          *std::min_element(residual_times.begin(), residual_times.end());
      run.timing.residual_mean_us =
          std::accumulate(residual_times.begin(), residual_times.end(), 0.0) /
          static_cast<double>(residual_times.size());
      run.timing.rmsnorm_min_us =
          *std::min_element(rmsnorm_times.begin(), rmsnorm_times.end());
      run.timing.rmsnorm_mean_us =
          std::accumulate(rmsnorm_times.begin(), rmsnorm_times.end(), 0.0) /
          static_cast<double>(rmsnorm_times.size());
      run.timing.shell_sum_min_us =
          run.timing.q8_quantize_min_us + run.timing.matvec.min_us +
          run.timing.residual_min_us + run.timing.rmsnorm_min_us;
      run.timing.shell_sum_mean_us =
          run.timing.q8_quantize_mean_us + run.timing.matvec.mean_us +
          run.timing.residual_mean_us + run.timing.rmsnorm_mean_us;
      run.timing.q8_quantize_global_work_items =
          q8_timing.global_work_items;
      run.timing.residual_global_work_items = residual_global;
      run.timing.rmsnorm_global_work_items = rmsnorm_global;
    } catch (...) {
      throw;
    }
    return run;
  }

  GpuRmsNormQ6MatvecRun RunRmsNormThenResidentRawQ6K(
      const std::vector<float>& input,
      std::uint64_t norm_weight_handle,
      std::uint64_t q6_handle,
      std::uint64_t hidden_size,
      float norm_epsilon,
      int repeat) {
    const auto& norm_weight = ResidentF32BufferForHandle(norm_weight_handle);
    const auto& q6 = ResidentRawQ6KForHandle(q6_handle);
    Require(input.size() == hidden_size,
            "RMSNorm Q6 handoff input size mismatch");
    Require(norm_weight.values == hidden_size,
            "RMSNorm Q6 handoff norm weight size mismatch");
    Require(q6.blocks_per_row * kQ8QsPerBlock == hidden_size,
            "RMSNorm Q6 handoff Q6 block count mismatch");
    Require(repeat > 0, "RMSNorm Q6 handoff repeat must be positive");

    GpuRmsNormQ6MatvecRun run;
    run.output.assign(static_cast<std::size_t>(q6.rows), 0.0f);
    cl_mem input_buffer = nullptr;
    cl_mem normalized_buffer = nullptr;
    cl_mem q8_qs_buffer = nullptr;
    cl_mem q8_d_buffer = nullptr;
    cl_mem output_buffer = nullptr;
    auto release_all = [&]() {
      ReleaseMem(api_, &output_buffer);
      ReleaseMem(api_, &q8_d_buffer);
      ReleaseMem(api_, &q8_qs_buffer);
      ReleaseMem(api_, &normalized_buffer);
      ReleaseMem(api_, &input_buffer);
    };
    try {
      cl_int err = kClSuccess;
      input_buffer = api_.clCreateBuffer(
          context_, kClMemReadOnly, input.size() * sizeof(float), nullptr, &err);
      Check(err, "clCreateBuffer(RMSNorm Q6 input)");
      normalized_buffer = api_.clCreateBuffer(
          context_, kClMemReadWrite, hidden_size * sizeof(float), nullptr, &err);
      Check(err, "clCreateBuffer(RMSNorm Q6 normalized)");
      q8_qs_buffer = api_.clCreateBuffer(
          context_, kClMemReadWrite, hidden_size * sizeof(std::int8_t), nullptr,
          &err);
      Check(err, "clCreateBuffer(RMSNorm Q6 q8 qs)");
      q8_d_buffer = api_.clCreateBuffer(
          context_, kClMemReadWrite, q6.blocks_per_row * sizeof(float), nullptr,
          &err);
      Check(err, "clCreateBuffer(RMSNorm Q6 q8 d)");
      output_buffer = api_.clCreateBuffer(
          context_, kClMemWriteOnly, run.output.size() * sizeof(float), nullptr,
          &err);
      Check(err, "clCreateBuffer(RMSNorm Q6 output)");
      Check(api_.clEnqueueWriteBuffer(queue_, input_buffer, kClTrue, 0,
                                      input.size() * sizeof(float),
                                      input.data(), 0, nullptr, nullptr),
            "clEnqueueWriteBuffer(RMSNorm Q6 input)");

      const cl_uint hidden_arg = static_cast<cl_uint>(hidden_size);
      Check(api_.clSetKernelArg(kernel_rmsnorm_hidden_, 0, sizeof(input_buffer),
                                &input_buffer),
            "clSetKernelArg(RMSNorm Q6 norm 0)");
      Check(api_.clSetKernelArg(kernel_rmsnorm_hidden_, 1,
                                sizeof(norm_weight.buffer),
                                &norm_weight.buffer),
            "clSetKernelArg(RMSNorm Q6 norm 1)");
      Check(api_.clSetKernelArg(kernel_rmsnorm_hidden_, 2, sizeof(hidden_arg),
                                &hidden_arg),
            "clSetKernelArg(RMSNorm Q6 norm 2)");
      Check(api_.clSetKernelArg(kernel_rmsnorm_hidden_, 3, sizeof(norm_epsilon),
                                &norm_epsilon),
            "clSetKernelArg(RMSNorm Q6 norm 3)");
      Check(api_.clSetKernelArg(kernel_rmsnorm_hidden_, 4,
                                sizeof(normalized_buffer),
                                &normalized_buffer),
            "clSetKernelArg(RMSNorm Q6 norm 4)");

      const std::size_t rmsnorm_global = 1;
      std::vector<double> rmsnorm_times;
      rmsnorm_times.reserve(static_cast<std::size_t>(repeat));
      for (int i = 0; i < repeat; ++i) {
        cl_event event = nullptr;
        Check(api_.clEnqueueNDRangeKernel(queue_, kernel_rmsnorm_hidden_, 1,
                                          nullptr, &rmsnorm_global, nullptr,
                                          0, nullptr, EventOut(&event)),
              "clEnqueueNDRangeKernel(RMSNorm Q6 norm)");
        Check(api_.clFinish(queue_), "clFinish(RMSNorm Q6 norm)");
        rmsnorm_times.push_back(EventUs(api_, event));
        ReleaseEvent(api_, &event);
      }
      run.timing.rmsnorm_min_us =
          *std::min_element(rmsnorm_times.begin(), rmsnorm_times.end());
      run.timing.rmsnorm_mean_us =
          std::accumulate(rmsnorm_times.begin(), rmsnorm_times.end(), 0.0) /
          static_cast<double>(rmsnorm_times.size());
      run.timing.rmsnorm_global_work_items = rmsnorm_global;

      const auto q8_timing = RunQ8QuantizeKernel(
          normalized_buffer, q6.blocks_per_row, q8_qs_buffer, q8_d_buffer,
          repeat);
      run.timing.q8_quantize_min_us = q8_timing.min_us;
      run.timing.q8_quantize_mean_us = q8_timing.mean_us;
      run.timing.q8_quantize_global_work_items =
          q8_timing.global_work_items;
      run.timing.matvec =
          q6.rowstripe
              ? RunQ6KSelectedRowstripeKernel(
                    q6.buffer, q8_qs_buffer, q8_d_buffer, output_buffer,
                    q6.rows, q6.blocks_per_row, q6.rows, q6.rows_per_tile,
                    repeat)
              : RunQ6KKernel(q6.buffer, q8_qs_buffer, q8_d_buffer,
                             output_buffer, q6.rows, q6.blocks_per_row,
                             repeat);
      run.timing.shell_sum_min_us =
          run.timing.rmsnorm_min_us + run.timing.q8_quantize_min_us +
          run.timing.matvec.min_us;
      run.timing.shell_sum_mean_us =
          run.timing.rmsnorm_mean_us + run.timing.q8_quantize_mean_us +
          run.timing.matvec.mean_us;
      Check(api_.clEnqueueReadBuffer(queue_, output_buffer, kClTrue, 0,
                                     run.output.size() * sizeof(float),
                                     run.output.data(), 0, nullptr, nullptr),
            "clEnqueueReadBuffer(RMSNorm Q6 output)");
    } catch (...) {
      release_all();
      throw;
    }
    release_all();
    return run;
  }

  GpuQ4X8ConvHandoffRun RunResidentRawQ6KThenResidentConv(
      std::uint64_t q6_handle,
      const GpuQ8KInputPlanes& q8,
      std::uint64_t conv_weights_handle,
      const std::vector<float>& conv_state,
      std::uint64_t conv_kernel_size,
      int repeat) {
    const auto& resident = ResidentRawQ6KForHandle(q6_handle);
    const auto& conv = ResidentConvWeightsForHandle(conv_weights_handle);
    Require(conv.rows == resident.rows,
            "resident conv rows do not match resident Q6_K rows");
    Require(conv.conv_kernel_size == conv_kernel_size,
            "resident conv kernel size mismatch");
    ValidateQ6KQ8InputPlanes(q8, resident.blocks_per_row, repeat);
    ValidateConvState(conv_state, resident.rows, conv_kernel_size);
    GpuQ4X8ConvHandoffRun run;
    run.qkv_mixed.assign(static_cast<std::size_t>(resident.rows), 0.0f);
    run.conv_output_raw.assign(static_cast<std::size_t>(resident.rows), 0.0f);
    run.conv_state.assign(
        static_cast<std::size_t>(resident.rows * (conv_kernel_size - 1)), 0.0f);
    cl_mem q8_qs_buffer = nullptr, q8_d_buffer = nullptr;
    cl_mem qkv_buffer = nullptr, conv_state_buffer = nullptr;
    cl_mem conv_output_buffer = nullptr, next_state_buffer = nullptr;
    auto release_all = [&]() {
      ReleaseMem(api_, &next_state_buffer);
      ReleaseMem(api_, &conv_output_buffer);
      ReleaseMem(api_, &conv_state_buffer);
      ReleaseMem(api_, &qkv_buffer);
      ReleaseMem(api_, &q8_d_buffer);
      ReleaseMem(api_, &q8_qs_buffer);
    };
    try {
      CreateQ6RunBuffers(q8, run.qkv_mixed, &q8_qs_buffer, &q8_d_buffer,
                         &qkv_buffer);
      CreateConvBuffersWithoutWeights(conv_state, run.conv_output_raw,
                                      run.conv_state, &conv_state_buffer,
                                      &conv_output_buffer, &next_state_buffer);
      run.timing.matvec =
          resident.rowstripe
              ? RunQ6KSelectedRowstripeKernel(
                    resident.buffer, q8_qs_buffer, q8_d_buffer, qkv_buffer,
                    resident.rows, resident.blocks_per_row, resident.rows,
                    resident.rows_per_tile, repeat)
              : RunQ6KKernel(resident.buffer, q8_qs_buffer, q8_d_buffer,
                             qkv_buffer, resident.rows,
                             resident.blocks_per_row, repeat);
      const std::uint64_t conv_global = resident.rows;
      const cl_uint channel_count_arg = static_cast<cl_uint>(resident.rows);
      const cl_uint kernel_size_arg = static_cast<cl_uint>(conv_kernel_size);
      Check(api_.clSetKernelArg(kernel_conv_, 0, sizeof(qkv_buffer), &qkv_buffer),
            "clSetKernelArg(q6 resident conv 0)");
      Check(api_.clSetKernelArg(kernel_conv_, 1, sizeof(conv_state_buffer),
                                &conv_state_buffer),
            "clSetKernelArg(q6 resident conv 1)");
      Check(api_.clSetKernelArg(kernel_conv_, 2, sizeof(conv.buffer), &conv.buffer),
            "clSetKernelArg(q6 resident conv 2)");
      Check(api_.clSetKernelArg(kernel_conv_, 3, sizeof(channel_count_arg),
                                &channel_count_arg),
            "clSetKernelArg(q6 resident conv 3)");
      Check(api_.clSetKernelArg(kernel_conv_, 4, sizeof(kernel_size_arg),
                                &kernel_size_arg),
            "clSetKernelArg(q6 resident conv 4)");
      Check(api_.clSetKernelArg(kernel_conv_, 5, sizeof(conv_output_buffer),
                                &conv_output_buffer),
            "clSetKernelArg(q6 resident conv 5)");
      Check(api_.clSetKernelArg(kernel_conv_, 6, sizeof(next_state_buffer),
                                &next_state_buffer),
            "clSetKernelArg(q6 resident conv 6)");
      const std::size_t global = static_cast<std::size_t>(conv_global);
      std::vector<double> conv_times;
      conv_times.reserve(static_cast<std::size_t>(repeat));
      for (int i = 0; i < repeat; ++i) {
        cl_event event = nullptr;
        Check(api_.clEnqueueNDRangeKernel(queue_, kernel_conv_, 1, nullptr,
                                          &global, nullptr, 0, nullptr, EventOut(&event)),
              "clEnqueueNDRangeKernel(q6 resident linear_attn_conv_f32)");
        Check(api_.clFinish(queue_), "clFinish(q6 resident conv)");
        conv_times.push_back(EventUs(api_, event));
        ReleaseEvent(api_, &event);
      }
      run.timing.conv_min_us =
          *std::min_element(conv_times.begin(), conv_times.end());
      run.timing.conv_mean_us =
          std::accumulate(conv_times.begin(), conv_times.end(), 0.0) /
          static_cast<double>(conv_times.size());
      run.timing.conv_global_work_items = conv_global;
      Check(api_.clEnqueueReadBuffer(queue_, qkv_buffer, kClTrue, 0,
                                     run.qkv_mixed.size() * sizeof(float),
                                     run.qkv_mixed.data(), 0, nullptr, nullptr),
            "clEnqueueReadBuffer(resident Q6 qkv)");
      Check(api_.clEnqueueReadBuffer(queue_, conv_output_buffer, kClTrue, 0,
                                     run.conv_output_raw.size() * sizeof(float),
                                     run.conv_output_raw.data(), 0, nullptr, nullptr),
            "clEnqueueReadBuffer(resident Q6 conv output)");
      Check(api_.clEnqueueReadBuffer(queue_, next_state_buffer, kClTrue, 0,
                                     run.conv_state.size() * sizeof(float),
                                     run.conv_state.data(), 0, nullptr, nullptr),
            "clEnqueueReadBuffer(resident Q6 next state)");
    } catch (...) {
      release_all();
      throw;
    }
    release_all();
    return run;
  }

  GpuQ4X8ConvHandoffRun RunResidentRawQ6KThenResidentConvState(
      std::uint64_t q6_handle,
      const GpuQ8KInputPlanes& q8,
      std::uint64_t conv_weights_handle,
      std::uint64_t conv_state_handle,
      std::uint64_t conv_kernel_size,
      int repeat,
      bool readback_state,
      std::uint64_t next_conv_state_handle,
      bool readback_qkv,
      bool readback_conv_output,
      bool cpuorder = false) {
    const auto& resident = ResidentRawQ6KForHandle(q6_handle);
    const auto& conv = ResidentConvWeightsForHandle(conv_weights_handle);
    const auto& conv_state = ResidentF32BufferForHandle(conv_state_handle);
    const ResidentF32Buffer* next_conv_state = nullptr;
    if (next_conv_state_handle != 0) {
      Require(next_conv_state_handle != conv_state_handle,
              "resident next conv state must differ from current state");
      next_conv_state = &ResidentF32BufferForHandle(next_conv_state_handle);
    }
    Require(conv.rows == resident.rows,
            "resident conv rows do not match resident Q6_K rows");
    Require(conv.conv_kernel_size == conv_kernel_size,
            "resident conv kernel size mismatch");
    Require(conv_state.values == resident.rows * (conv_kernel_size - 1),
            "resident conv state size mismatch");
    if (next_conv_state != nullptr) {
      Require(next_conv_state->values == conv_state.values,
              "resident next conv state size mismatch");
    }
    ValidateQ6KQ8InputPlanes(q8, resident.blocks_per_row, repeat);
    Require(!cpuorder || !resident.rowstripe,
            "CPU-order Q6 QKV component requires raw row layout");
    GpuQ4X8ConvHandoffRun run;
    if (readback_qkv) {
      run.qkv_mixed.assign(static_cast<std::size_t>(resident.rows), 0.0f);
    }
    const std::size_t conv_output_values = static_cast<std::size_t>(resident.rows);
    if (readback_conv_output) {
      run.conv_output_raw.assign(conv_output_values, 0.0f);
    }
    if (readback_state) {
      run.conv_state.assign(
          static_cast<std::size_t>(resident.rows * (conv_kernel_size - 1)),
          0.0f);
    }
    cl_mem q8_qs_buffer = EnsureScratchBuffer(
        q6_conv_state_scratch_q8_qs_, q8.qs.size(), kClMemReadOnly,
        "resident Q6 conv-state q8_qs");
    cl_mem q8_d_buffer = EnsureScratchBuffer(
        q6_conv_state_scratch_q8_d_, q8.d.size() * sizeof(float),
        kClMemReadOnly, "resident Q6 conv-state q8_d");
    cl_mem qkv_buffer = EnsureScratchBuffer(
        q6_conv_state_scratch_qkv_,
        static_cast<std::size_t>(resident.rows) * sizeof(float),
        kClMemWriteOnly, "resident Q6 conv-state qkv");
    cl_mem conv_output_buffer = EnsureScratchBuffer(
        q6_conv_state_scratch_conv_output_,
        conv_output_values * sizeof(float), kClMemReadWrite,
        "resident Q6 conv-state output");
    cl_mem next_state_buffer = nullptr;
    if (next_conv_state == nullptr) {
      next_state_buffer = EnsureScratchBuffer(
          q6_conv_state_scratch_next_state_, conv_state.bytes, kClMemReadWrite,
          "resident Q6 conv-state next state");
    }
    Check(api_.clEnqueueWriteBuffer(queue_, q8_qs_buffer, kClFalse, 0,
                                    q8.qs.size(), q8.qs.data(), 0, nullptr,
                                    nullptr),
          "clEnqueueWriteBuffer(resident Q6 conv-state q8_qs)");
    Check(api_.clEnqueueWriteBuffer(queue_, q8_d_buffer, kClFalse, 0,
                                    q8.d.size() * sizeof(float), q8.d.data(),
                                    0, nullptr, nullptr),
          "clEnqueueWriteBuffer(resident Q6 conv-state q8_d)");
    cl_mem next_state_arg =
        next_conv_state != nullptr ? next_conv_state->buffer : next_state_buffer;
    run.timing.matvec =
        resident.rowstripe
            ? RunQ6KSelectedRowstripeKernel(
                  resident.buffer, q8_qs_buffer, q8_d_buffer, qkv_buffer,
                  resident.rows, resident.blocks_per_row, resident.rows,
                  resident.rows_per_tile, repeat)
            : RunQ6KKernel(resident.buffer, q8_qs_buffer, q8_d_buffer,
                           qkv_buffer, resident.rows, resident.blocks_per_row,
                           repeat, false, cpuorder);
    cl_kernel conv_kernel = cpuorder ? kernel_conv_cpuorder_ : kernel_conv_;
    const std::uint64_t conv_global = resident.rows;
    const cl_uint channel_count_arg = static_cast<cl_uint>(resident.rows);
    const cl_uint kernel_size_arg = static_cast<cl_uint>(conv_kernel_size);
    Check(api_.clSetKernelArg(conv_kernel, 0, sizeof(qkv_buffer), &qkv_buffer),
          "clSetKernelArg(q6 resident state conv 0)");
    Check(api_.clSetKernelArg(conv_kernel, 1, sizeof(conv_state.buffer),
                              &conv_state.buffer),
          "clSetKernelArg(q6 resident state conv 1)");
    Check(api_.clSetKernelArg(conv_kernel, 2, sizeof(conv.buffer), &conv.buffer),
          "clSetKernelArg(q6 resident state conv 2)");
    Check(api_.clSetKernelArg(conv_kernel, 3, sizeof(channel_count_arg),
                              &channel_count_arg),
          "clSetKernelArg(q6 resident state conv 3)");
    Check(api_.clSetKernelArg(conv_kernel, 4, sizeof(kernel_size_arg),
                              &kernel_size_arg),
          "clSetKernelArg(q6 resident state conv 4)");
    Check(api_.clSetKernelArg(conv_kernel, 5, sizeof(conv_output_buffer),
                              &conv_output_buffer),
          "clSetKernelArg(q6 resident state conv 5)");
    Check(api_.clSetKernelArg(conv_kernel, 6, sizeof(next_state_arg),
                              &next_state_arg),
          "clSetKernelArg(q6 resident state conv 6)");
    const std::size_t global = static_cast<std::size_t>(conv_global);
    std::vector<double> conv_times;
    conv_times.reserve(static_cast<std::size_t>(repeat));
    for (int i = 0; i < repeat; ++i) {
      cl_event event = nullptr;
      Check(api_.clEnqueueNDRangeKernel(
                queue_, conv_kernel, 1, nullptr, &global, nullptr, 0, nullptr,
                EventOut(&event)),
            "clEnqueueNDRangeKernel(q6 resident state linear_attn_conv_f32)");
      Check(api_.clFinish(queue_), "clFinish(q6 resident state conv)");
      conv_times.push_back(EventUs(api_, event));
      ReleaseEvent(api_, &event);
    }
    run.timing.conv_min_us =
        *std::min_element(conv_times.begin(), conv_times.end());
    run.timing.conv_mean_us =
        std::accumulate(conv_times.begin(), conv_times.end(), 0.0) /
          static_cast<double>(conv_times.size());
    run.timing.conv_global_work_items = conv_global;
    if (next_conv_state == nullptr) {
      Check(api_.clEnqueueCopyBuffer(queue_, next_state_buffer,
                                     conv_state.buffer, 0, 0,
                                     conv_state.bytes, 0, nullptr, nullptr),
            "clEnqueueCopyBuffer(resident Q6 conv state)");
      Check(api_.clFinish(queue_), "clFinish(resident Q6 conv state copy)");
    }
    if (readback_qkv) {
      Check(api_.clEnqueueReadBuffer(queue_, qkv_buffer, kClTrue, 0,
                                     run.qkv_mixed.size() * sizeof(float),
                                     run.qkv_mixed.data(), 0, nullptr, nullptr),
            "clEnqueueReadBuffer(resident Q6 qkv)");
    }
    run.conv_output_handle = RegisterF32BufferAlias(
        &q6_conv_state_output_alias_handle_, conv_output_buffer,
        conv_output_values);
    if (readback_conv_output) {
      Check(api_.clEnqueueReadBuffer(queue_, conv_output_buffer, kClTrue, 0,
                                     run.conv_output_raw.size() * sizeof(float),
                                     run.conv_output_raw.data(), 0, nullptr,
                                     nullptr),
            "clEnqueueReadBuffer(resident Q6 conv output)");
    }
    if (readback_state) {
      const cl_mem state_readback =
          next_conv_state != nullptr ? next_conv_state->buffer
                                     : conv_state.buffer;
      Check(api_.clEnqueueReadBuffer(queue_, state_readback, kClTrue, 0,
                                     run.conv_state.size() * sizeof(float),
                                     run.conv_state.data(), 0, nullptr,
                                     nullptr),
            "clEnqueueReadBuffer(resident Q6 conv state)");
    }
    return run;
  }

  GpuQ4X8ConvHandoffRun
  RunF32InputHandleDeviceQ8ThenResidentRawQ6KThenResidentConvState(
      std::uint64_t q6_handle,
      std::uint64_t input_handle,
      std::uint64_t conv_weights_handle,
      std::uint64_t conv_state_handle,
      std::uint64_t conv_kernel_size,
      int repeat,
      bool readback_state,
      std::uint64_t next_conv_state_handle,
      bool readback_qkv,
      bool readback_conv_output) {
    const auto& resident = ResidentRawQ6KForHandle(q6_handle);
    const auto& input = ResidentF32BufferForHandle(input_handle);
    const auto& conv = ResidentConvWeightsForHandle(conv_weights_handle);
    const auto& conv_state = ResidentF32BufferForHandle(conv_state_handle);
    const ResidentF32Buffer* next_conv_state = nullptr;
    if (next_conv_state_handle != 0) {
      Require(next_conv_state_handle != conv_state_handle,
              "resident next conv state must differ from current state");
      next_conv_state = &ResidentF32BufferForHandle(next_conv_state_handle);
    }
    Require(input.values == resident.blocks_per_row * kQ8QsPerBlock,
            "device-Q8 Q6 conv-state input handle size mismatch");
    Require(conv.rows == resident.rows,
            "resident conv rows do not match resident Q6_K rows");
    Require(conv.conv_kernel_size == conv_kernel_size,
            "resident conv kernel size mismatch");
    Require(conv_state.values == resident.rows * (conv_kernel_size - 1),
            "resident conv state size mismatch");
    if (next_conv_state != nullptr) {
      Require(next_conv_state->values == conv_state.values,
              "resident next conv state size mismatch");
    }
    Require(repeat > 0, "device-Q8 Q6 conv-state repeat must be positive");
    GpuQ4X8ConvHandoffRun run;
    if (readback_qkv) {
      run.qkv_mixed.assign(static_cast<std::size_t>(resident.rows), 0.0f);
    }
    const std::size_t conv_output_values = static_cast<std::size_t>(resident.rows);
    if (readback_conv_output) {
      run.conv_output_raw.assign(conv_output_values, 0.0f);
    }
    if (readback_state) {
      run.conv_state.assign(
          static_cast<std::size_t>(resident.rows * (conv_kernel_size - 1)),
          0.0f);
    }
    const std::uint64_t block_count = resident.blocks_per_row;
    cl_mem q8_qs_buffer = EnsureScratchBuffer(
        f32_input_q6_scratch_q8_qs_,
        static_cast<std::size_t>(block_count) * kQ8QsPerBlock *
            sizeof(std::int8_t),
        kClMemReadWrite, "device-Q8 Q6 conv-state q8_qs");
    cl_mem q8_d_buffer = EnsureScratchBuffer(
        f32_input_q6_scratch_q8_d_,
        static_cast<std::size_t>(block_count) * sizeof(float),
        kClMemReadWrite, "device-Q8 Q6 conv-state q8_d");
    cl_mem qkv_buffer = EnsureScratchBuffer(
        q6_conv_state_scratch_qkv_,
        static_cast<std::size_t>(resident.rows) * sizeof(float),
        kClMemWriteOnly, "device-Q8 Q6 conv-state qkv");
    cl_mem conv_output_buffer = EnsureScratchBuffer(
        q6_conv_state_scratch_conv_output_,
        conv_output_values * sizeof(float), kClMemReadWrite,
        "device-Q8 Q6 conv-state output");
    cl_mem next_state_buffer = nullptr;
    if (next_conv_state == nullptr) {
      next_state_buffer = EnsureScratchBuffer(
          q6_conv_state_scratch_next_state_, conv_state.bytes,
          kClMemReadWrite, "device-Q8 Q6 conv-state next state");
    }

    const auto q8_timing = RunQ8QuantizeKernel(
        input.buffer, block_count, q8_qs_buffer, q8_d_buffer, repeat);
    cl_mem next_state_arg =
        next_conv_state != nullptr ? next_conv_state->buffer : next_state_buffer;
    run.timing.matvec =
        resident.rowstripe
            ? RunQ6KSelectedRowstripeKernel(
                  resident.buffer, q8_qs_buffer, q8_d_buffer, qkv_buffer,
                  resident.rows, resident.blocks_per_row, resident.rows,
                  resident.rows_per_tile, repeat)
            : RunQ6KKernel(resident.buffer, q8_qs_buffer, q8_d_buffer,
                           qkv_buffer, resident.rows, resident.blocks_per_row,
                           repeat);
    const std::uint64_t conv_global = resident.rows;
    const cl_uint channel_count_arg = static_cast<cl_uint>(resident.rows);
    const cl_uint kernel_size_arg = static_cast<cl_uint>(conv_kernel_size);
    Check(api_.clSetKernelArg(kernel_conv_, 0, sizeof(qkv_buffer), &qkv_buffer),
          "clSetKernelArg(device-Q8 Q6 resident state conv 0)");
    Check(api_.clSetKernelArg(kernel_conv_, 1, sizeof(conv_state.buffer),
                              &conv_state.buffer),
          "clSetKernelArg(device-Q8 Q6 resident state conv 1)");
    Check(api_.clSetKernelArg(kernel_conv_, 2, sizeof(conv.buffer), &conv.buffer),
          "clSetKernelArg(device-Q8 Q6 resident state conv 2)");
    Check(api_.clSetKernelArg(kernel_conv_, 3, sizeof(channel_count_arg),
                              &channel_count_arg),
          "clSetKernelArg(device-Q8 Q6 resident state conv 3)");
    Check(api_.clSetKernelArg(kernel_conv_, 4, sizeof(kernel_size_arg),
                              &kernel_size_arg),
          "clSetKernelArg(device-Q8 Q6 resident state conv 4)");
    Check(api_.clSetKernelArg(kernel_conv_, 5, sizeof(conv_output_buffer),
                              &conv_output_buffer),
          "clSetKernelArg(device-Q8 Q6 resident state conv 5)");
    Check(api_.clSetKernelArg(kernel_conv_, 6, sizeof(next_state_arg),
                              &next_state_arg),
          "clSetKernelArg(device-Q8 Q6 resident state conv 6)");
    const std::size_t global = static_cast<std::size_t>(conv_global);
    std::vector<double> conv_times;
    conv_times.reserve(static_cast<std::size_t>(repeat));
    for (int i = 0; i < repeat; ++i) {
      cl_event event = nullptr;
      Check(api_.clEnqueueNDRangeKernel(
                queue_, kernel_conv_, 1, nullptr, &global, nullptr, 0, nullptr,
                EventOut(&event)),
            "clEnqueueNDRangeKernel(device-Q8 Q6 resident state conv)");
      Check(api_.clFinish(queue_), "clFinish(device-Q8 Q6 resident state conv)");
      conv_times.push_back(EventUs(api_, event));
      ReleaseEvent(api_, &event);
    }
    run.timing.q8_quantize_min_us = q8_timing.min_us;
    run.timing.q8_quantize_mean_us = q8_timing.mean_us;
    run.timing.q8_quantize_global_work_items = q8_timing.global_work_items;
    run.timing.conv_min_us =
        *std::min_element(conv_times.begin(), conv_times.end());
    run.timing.conv_mean_us =
        std::accumulate(conv_times.begin(), conv_times.end(), 0.0) /
          static_cast<double>(conv_times.size());
    run.timing.shell_sum_min_us =
        q8_timing.min_us + run.timing.matvec.min_us + run.timing.conv_min_us;
    run.timing.shell_sum_mean_us =
        q8_timing.mean_us + run.timing.matvec.mean_us + run.timing.conv_mean_us;
    run.timing.conv_global_work_items = conv_global;
    if (next_conv_state == nullptr) {
      Check(api_.clEnqueueCopyBuffer(queue_, next_state_buffer,
                                     conv_state.buffer, 0, 0,
                                     conv_state.bytes, 0, nullptr, nullptr),
            "clEnqueueCopyBuffer(device-Q8 Q6 conv state)");
      Check(api_.clFinish(queue_), "clFinish(device-Q8 Q6 conv state copy)");
    }
    if (readback_qkv) {
      Check(api_.clEnqueueReadBuffer(queue_, qkv_buffer, kClTrue, 0,
                                     run.qkv_mixed.size() * sizeof(float),
                                     run.qkv_mixed.data(), 0, nullptr, nullptr),
            "clEnqueueReadBuffer(device-Q8 Q6 qkv)");
    }
    run.conv_output_handle = RegisterF32BufferAlias(
        &q6_conv_state_output_alias_handle_, conv_output_buffer,
        conv_output_values);
    if (readback_conv_output) {
      Check(api_.clEnqueueReadBuffer(queue_, conv_output_buffer, kClTrue, 0,
                                     run.conv_output_raw.size() * sizeof(float),
                                     run.conv_output_raw.data(), 0, nullptr,
                                     nullptr),
            "clEnqueueReadBuffer(device-Q8 Q6 conv output)");
    }
    if (readback_state) {
      const cl_mem state_readback =
          next_conv_state != nullptr ? next_conv_state->buffer
                                     : conv_state.buffer;
      Check(api_.clEnqueueReadBuffer(queue_, state_readback, kClTrue, 0,
                                     run.conv_state.size() * sizeof(float),
                                     run.conv_state.data(), 0, nullptr,
                                     nullptr),
            "clEnqueueReadBuffer(device-Q8 Q6 conv state)");
    }
    return run;
  }

  std::uint64_t UploadConvWeights(const std::vector<float>& conv_weights,
                                  std::uint64_t rows,
                                  std::uint64_t conv_kernel_size) {
    ValidateConvWeights(conv_weights, rows, conv_kernel_size);
    cl_int err = kClSuccess;
    cl_mem buffer = api_.clCreateBuffer(
        context_, kClMemReadOnly, conv_weights.size() * sizeof(float), nullptr, &err);
    Check(err, "clCreateBuffer(resident conv weights)");
    try {
      Check(api_.clEnqueueWriteBuffer(queue_, buffer, kClTrue, 0,
                                      conv_weights.size() * sizeof(float),
                                      conv_weights.data(), 0, nullptr, nullptr),
            "clEnqueueWriteBuffer(resident conv weights)");
    } catch (...) {
      ReleaseMem(api_, &buffer);
      throw;
    }
    const std::uint64_t handle = next_resident_handle_++;
    resident_conv_weights_.emplace(
        handle,
        ResidentConvWeights{buffer, rows, conv_kernel_size,
                            conv_weights.size() * sizeof(float)});
    return handle;
  }

  GpuQ4X8ConvHandoffRun RunResidentPackedQ4X8ThenResidentConv(
      std::uint64_t packed_handle,
      const std::vector<std::int8_t>& q8_qs,
      const std::vector<std::int16_t>& q8_bsums,
      const std::vector<float>& q8_d,
      std::uint64_t conv_weights_handle,
      const std::vector<float>& conv_state,
      std::uint64_t conv_kernel_size,
      int repeat,
      GpuQ4X8KernelVariant variant) {
    const auto& resident = ResidentPackedQ4X8ForHandle(packed_handle);
    const auto& conv = ResidentConvWeightsForHandle(conv_weights_handle);
    Require(conv.rows == resident.rows,
            "resident conv rows do not match resident packed rows");
    Require(conv.conv_kernel_size == conv_kernel_size,
            "resident conv kernel size mismatch");
    ValidateQ8InputPlanes(q8_qs, q8_bsums, q8_d, resident.blocks_per_row, repeat);
    ValidateConvState(conv_state, resident.rows, conv_kernel_size);
    GpuQ4X8ConvHandoffRun run;
    run.qkv_mixed.assign(static_cast<std::size_t>(resident.rows), 0.0f);
    run.conv_output_raw.assign(static_cast<std::size_t>(resident.rows), 0.0f);
    run.conv_state.assign(
        static_cast<std::size_t>(resident.rows * (conv_kernel_size - 1)), 0.0f);
    cl_mem q8_qs_buffer = nullptr, q8_bsums_buffer = nullptr, q8_d_buffer = nullptr;
    cl_mem qkv_buffer = nullptr, conv_state_buffer = nullptr;
    cl_mem conv_output_buffer = nullptr, next_state_buffer = nullptr;
    auto release_all = [&]() {
      ReleaseMem(api_, &next_state_buffer);
      ReleaseMem(api_, &conv_output_buffer);
      ReleaseMem(api_, &conv_state_buffer);
      ReleaseMem(api_, &qkv_buffer);
      ReleaseMem(api_, &q8_d_buffer);
      ReleaseMem(api_, &q8_bsums_buffer);
      ReleaseMem(api_, &q8_qs_buffer);
    };
    try {
      CreateRunBuffersWithoutPacked(q8_qs, q8_bsums, q8_d, run.qkv_mixed,
                                    &q8_qs_buffer, &q8_bsums_buffer,
                                    &q8_d_buffer, &qkv_buffer);
      CreateConvBuffersWithoutWeights(conv_state, run.conv_output_raw,
                                      run.conv_state, &conv_state_buffer,
                                      &conv_output_buffer, &next_state_buffer);
      const std::uint64_t row_groups = resident.rows / kRowsInterleaved;
      const std::uint64_t matvec_global =
          variant == GpuQ4X8KernelVariant::kGroup8Serial ? row_groups : resident.rows;
      run.timing = RunHandoffKernels(
          variant, resident.buffer, q8_qs_buffer, q8_bsums_buffer, q8_d_buffer,
          qkv_buffer, conv.buffer, conv_state_buffer, conv_output_buffer,
          next_state_buffer, resident.blocks_per_row, row_groups, matvec_global,
          resident.rows, conv_kernel_size, repeat);
      Check(api_.clEnqueueReadBuffer(queue_, qkv_buffer, kClTrue, 0,
                                     run.qkv_mixed.size() * sizeof(float),
                                     run.qkv_mixed.data(), 0, nullptr, nullptr),
            "clEnqueueReadBuffer(resident qkv)");
      Check(api_.clEnqueueReadBuffer(queue_, conv_output_buffer, kClTrue, 0,
                                     run.conv_output_raw.size() * sizeof(float),
                                     run.conv_output_raw.data(), 0, nullptr, nullptr),
            "clEnqueueReadBuffer(resident conv output)");
      Check(api_.clEnqueueReadBuffer(queue_, next_state_buffer, kClTrue, 0,
                                     run.conv_state.size() * sizeof(float),
                                     run.conv_state.data(), 0, nullptr, nullptr),
            "clEnqueueReadBuffer(resident next state)");
    } catch (...) {
      release_all();
      throw;
    }
    release_all();
    return run;
  }

  GpuQ4X8ConvHandoffRun RunResidentPackedQ4X8ThenResidentConvState(
      std::uint64_t packed_handle,
      const std::vector<std::int8_t>& q8_qs,
      const std::vector<std::int16_t>& q8_bsums,
      const std::vector<float>& q8_d,
      std::uint64_t conv_weights_handle,
      std::uint64_t conv_state_handle,
      std::uint64_t conv_kernel_size,
      int repeat,
      GpuQ4X8KernelVariant variant,
      bool readback_state,
      std::uint64_t next_conv_state_handle,
      bool readback_qkv,
      bool readback_conv_output) {
    const auto& resident = ResidentPackedQ4X8ForHandle(packed_handle);
    const auto& conv = ResidentConvWeightsForHandle(conv_weights_handle);
    const auto& conv_state = ResidentF32BufferForHandle(conv_state_handle);
    const ResidentF32Buffer* next_conv_state = nullptr;
    if (next_conv_state_handle != 0) {
      Require(next_conv_state_handle != conv_state_handle,
              "resident next conv state must differ from current state");
      next_conv_state = &ResidentF32BufferForHandle(next_conv_state_handle);
    }
    Require(conv.rows == resident.rows,
            "resident conv rows do not match resident packed rows");
    Require(conv.conv_kernel_size == conv_kernel_size,
            "resident conv kernel size mismatch");
    Require(conv_state.values == resident.rows * (conv_kernel_size - 1),
            "resident conv state size mismatch");
    if (next_conv_state != nullptr) {
      Require(next_conv_state->values == conv_state.values,
              "resident next conv state size mismatch");
    }
    ValidateQ8InputPlanes(q8_qs, q8_bsums, q8_d, resident.blocks_per_row, repeat);
    GpuQ4X8ConvHandoffRun run;
    if (readback_qkv) {
      run.qkv_mixed.assign(static_cast<std::size_t>(resident.rows), 0.0f);
    }
    const std::size_t conv_output_values = static_cast<std::size_t>(resident.rows);
    if (readback_conv_output) {
      run.conv_output_raw.assign(conv_output_values, 0.0f);
    }
    if (readback_state) {
      run.conv_state.assign(
          static_cast<std::size_t>(resident.rows * (conv_kernel_size - 1)),
          0.0f);
    }
    cl_mem q8_qs_buffer = EnsureScratchBuffer(
        packed_conv_state_scratch_q8_qs_, q8_qs.size(), kClMemReadOnly,
        "resident packed conv-state q8_qs");
    cl_mem q8_bsums_buffer = EnsureScratchBuffer(
        packed_conv_state_scratch_q8_bsums_,
        q8_bsums.size() * sizeof(std::int16_t), kClMemReadOnly,
        "resident packed conv-state q8_bsums");
    cl_mem q8_d_buffer = EnsureScratchBuffer(
        packed_conv_state_scratch_q8_d_, q8_d.size() * sizeof(float),
        kClMemReadOnly, "resident packed conv-state q8_d");
    cl_mem qkv_buffer = EnsureScratchBuffer(
        packed_conv_state_scratch_qkv_,
        static_cast<std::size_t>(resident.rows) * sizeof(float),
        kClMemWriteOnly, "resident packed conv-state qkv");
    cl_mem conv_output_buffer = EnsureScratchBuffer(
        packed_conv_state_scratch_conv_output_,
        conv_output_values * sizeof(float), kClMemReadWrite,
        "resident packed conv-state output");
    cl_mem next_state_buffer = nullptr;
    if (next_conv_state == nullptr) {
      next_state_buffer = EnsureScratchBuffer(
          packed_conv_state_scratch_next_state_, conv_state.bytes,
          kClMemReadWrite, "resident packed conv-state next state");
    }
    Check(api_.clEnqueueWriteBuffer(queue_, q8_qs_buffer, kClFalse, 0,
                                    q8_qs.size(), q8_qs.data(), 0, nullptr,
                                    nullptr),
          "clEnqueueWriteBuffer(resident packed conv-state q8_qs)");
    Check(api_.clEnqueueWriteBuffer(
              queue_, q8_bsums_buffer, kClFalse, 0,
              q8_bsums.size() * sizeof(std::int16_t), q8_bsums.data(), 0,
              nullptr, nullptr),
          "clEnqueueWriteBuffer(resident packed conv-state q8_bsums)");
    Check(api_.clEnqueueWriteBuffer(queue_, q8_d_buffer, kClFalse, 0,
                                    q8_d.size() * sizeof(float), q8_d.data(),
                                    0, nullptr, nullptr),
          "clEnqueueWriteBuffer(resident packed conv-state q8_d)");
    cl_mem next_state_arg =
        next_conv_state != nullptr ? next_conv_state->buffer : next_state_buffer;
    const std::uint64_t row_groups = resident.rows / kRowsInterleaved;
    const std::uint64_t matvec_global =
        variant == GpuQ4X8KernelVariant::kGroup8Serial ? row_groups : resident.rows;
    run.timing = RunHandoffKernels(
        variant, resident.buffer, q8_qs_buffer, q8_bsums_buffer, q8_d_buffer,
        qkv_buffer, conv.buffer, conv_state.buffer, conv_output_buffer,
        next_state_arg, resident.blocks_per_row, row_groups, matvec_global,
        resident.rows, conv_kernel_size, repeat);
    if (next_conv_state == nullptr) {
      Check(api_.clEnqueueCopyBuffer(queue_, next_state_buffer,
                                     conv_state.buffer, 0, 0,
                                     conv_state.bytes, 0, nullptr, nullptr),
            "clEnqueueCopyBuffer(resident packed conv state)");
      Check(api_.clFinish(queue_), "clFinish(resident packed conv state copy)");
    }
    if (readback_qkv) {
      Check(api_.clEnqueueReadBuffer(queue_, qkv_buffer, kClTrue, 0,
                                     run.qkv_mixed.size() * sizeof(float),
                                     run.qkv_mixed.data(), 0, nullptr, nullptr),
            "clEnqueueReadBuffer(resident qkv)");
    }
    run.conv_output_handle = RegisterF32BufferAlias(
        &packed_conv_state_output_alias_handle_, conv_output_buffer,
        conv_output_values);
    if (readback_conv_output) {
      Check(api_.clEnqueueReadBuffer(queue_, conv_output_buffer, kClTrue, 0,
                                     run.conv_output_raw.size() * sizeof(float),
                                     run.conv_output_raw.data(), 0, nullptr,
                                     nullptr),
            "clEnqueueReadBuffer(resident conv output)");
    }
    if (readback_state) {
      const cl_mem state_readback =
          next_conv_state != nullptr ? next_conv_state->buffer
                                     : conv_state.buffer;
      Check(api_.clEnqueueReadBuffer(queue_, state_readback, kClTrue, 0,
                                     run.conv_state.size() * sizeof(float),
                                     run.conv_state.data(), 0, nullptr,
                                     nullptr),
            "clEnqueueReadBuffer(resident conv state)");
    }
    return run;
  }

  GpuQ4X8ConvHandoffRun
  RunF32InputHandleDeviceQ8ThenResidentPackedQ4X8ThenResidentConvState(
      std::uint64_t packed_handle,
      std::uint64_t input_handle,
      std::uint64_t conv_weights_handle,
      std::uint64_t conv_state_handle,
      std::uint64_t conv_kernel_size,
      int repeat,
      GpuQ4X8KernelVariant variant,
      bool readback_state,
      std::uint64_t next_conv_state_handle,
      bool readback_qkv,
      bool readback_conv_output) {
    const auto& resident = ResidentPackedQ4X8ForHandle(packed_handle);
    const auto& input = ResidentF32BufferForHandle(input_handle);
    const auto& conv = ResidentConvWeightsForHandle(conv_weights_handle);
    const auto& conv_state = ResidentF32BufferForHandle(conv_state_handle);
    const ResidentF32Buffer* next_conv_state = nullptr;
    if (next_conv_state_handle != 0) {
      Require(next_conv_state_handle != conv_state_handle,
              "resident next conv state must differ from current state");
      next_conv_state = &ResidentF32BufferForHandle(next_conv_state_handle);
    }
    Require(input.values == resident.blocks_per_row * kQ8QsPerBlock,
            "device-Q8 Q4 conv-state input handle size mismatch");
    Require(conv.rows == resident.rows,
            "resident conv rows do not match resident packed rows");
    Require(conv.conv_kernel_size == conv_kernel_size,
            "resident conv kernel size mismatch");
    Require(conv_state.values == resident.rows * (conv_kernel_size - 1),
            "resident conv state size mismatch");
    if (next_conv_state != nullptr) {
      Require(next_conv_state->values == conv_state.values,
              "resident next conv state size mismatch");
    }
    Require(repeat > 0, "device-Q8 Q4 conv-state repeat must be positive");
    GpuQ4X8ConvHandoffRun run;
    if (readback_qkv) {
      run.qkv_mixed.assign(static_cast<std::size_t>(resident.rows), 0.0f);
    }
    const std::size_t conv_output_values = static_cast<std::size_t>(resident.rows);
    if (readback_conv_output) {
      run.conv_output_raw.assign(conv_output_values, 0.0f);
    }
    if (readback_state) {
      run.conv_state.assign(
          static_cast<std::size_t>(resident.rows * (conv_kernel_size - 1)),
          0.0f);
    }
    const std::uint64_t block_count = resident.blocks_per_row;
    cl_mem q8_qs_buffer = EnsureScratchBuffer(
        f32_input_q4_scratch_q8_qs_,
        static_cast<std::size_t>(block_count) * kQ8QsPerBlock *
            sizeof(std::int8_t),
        kClMemReadWrite, "device-Q8 Q4 conv-state q8_qs");
    cl_mem q8_bsums_buffer = EnsureScratchBuffer(
        f32_input_q4_scratch_q8_bsums_,
        static_cast<std::size_t>(block_count) * kQ8BsumsPerBlock *
            sizeof(std::int16_t),
        kClMemReadWrite, "device-Q8 Q4 conv-state q8_bsums");
    cl_mem q8_d_buffer = EnsureScratchBuffer(
        f32_input_q4_scratch_q8_d_,
        static_cast<std::size_t>(block_count) * sizeof(float),
        kClMemReadWrite, "device-Q8 Q4 conv-state q8_d");
    cl_mem qkv_buffer = EnsureScratchBuffer(
        packed_conv_state_scratch_qkv_,
        static_cast<std::size_t>(resident.rows) * sizeof(float),
        kClMemWriteOnly, "device-Q8 Q4 conv-state qkv");
    cl_mem conv_output_buffer = EnsureScratchBuffer(
        packed_conv_state_scratch_conv_output_,
        conv_output_values * sizeof(float), kClMemReadWrite,
        "device-Q8 Q4 conv-state output");
    cl_mem next_state_buffer = nullptr;
    if (next_conv_state == nullptr) {
      next_state_buffer = EnsureScratchBuffer(
          packed_conv_state_scratch_next_state_, conv_state.bytes,
          kClMemReadWrite, "device-Q8 Q4 conv-state next state");
    }

    const auto q8_timing = RunQ8QuantizeWithBsumsKernel(
        input.buffer, block_count, q8_qs_buffer, q8_bsums_buffer,
        q8_d_buffer, repeat);
    cl_mem next_state_arg =
        next_conv_state != nullptr ? next_conv_state->buffer : next_state_buffer;
    const std::uint64_t row_groups = resident.rows / kRowsInterleaved;
    const std::uint64_t matvec_global =
        variant == GpuQ4X8KernelVariant::kGroup8Serial ? row_groups : resident.rows;
    run.timing = RunHandoffKernels(
        variant, resident.buffer, q8_qs_buffer, q8_bsums_buffer, q8_d_buffer,
        qkv_buffer, conv.buffer, conv_state.buffer, conv_output_buffer,
        next_state_arg, resident.blocks_per_row, row_groups, matvec_global,
        resident.rows, conv_kernel_size, repeat);
    run.timing.q8_quantize_min_us = q8_timing.min_us;
    run.timing.q8_quantize_mean_us = q8_timing.mean_us;
    run.timing.q8_quantize_global_work_items = q8_timing.global_work_items;
    run.timing.shell_sum_min_us += q8_timing.min_us;
    run.timing.shell_sum_mean_us += q8_timing.mean_us;
    if (next_conv_state == nullptr) {
      Check(api_.clEnqueueCopyBuffer(queue_, next_state_buffer,
                                     conv_state.buffer, 0, 0,
                                     conv_state.bytes, 0, nullptr, nullptr),
            "clEnqueueCopyBuffer(device-Q8 Q4 conv state)");
      Check(api_.clFinish(queue_), "clFinish(device-Q8 Q4 conv state copy)");
    }
    if (readback_qkv) {
      Check(api_.clEnqueueReadBuffer(queue_, qkv_buffer, kClTrue, 0,
                                     run.qkv_mixed.size() * sizeof(float),
                                     run.qkv_mixed.data(), 0, nullptr, nullptr),
            "clEnqueueReadBuffer(device-Q8 Q4 qkv)");
    }
    run.conv_output_handle = RegisterF32BufferAlias(
        &packed_conv_state_output_alias_handle_, conv_output_buffer,
        conv_output_values);
    if (readback_conv_output) {
      Check(api_.clEnqueueReadBuffer(queue_, conv_output_buffer, kClTrue, 0,
                                     run.conv_output_raw.size() * sizeof(float),
                                     run.conv_output_raw.data(), 0, nullptr,
                                     nullptr),
            "clEnqueueReadBuffer(device-Q8 Q4 conv output)");
    }
    if (readback_state) {
      const cl_mem state_readback =
          next_conv_state != nullptr ? next_conv_state->buffer
                                     : conv_state.buffer;
      Check(api_.clEnqueueReadBuffer(queue_, state_readback, kClTrue, 0,
                                     run.conv_state.size() * sizeof(float),
                                     run.conv_state.data(), 0, nullptr,
                                     nullptr),
            "clEnqueueReadBuffer(device-Q8 Q4 conv state)");
    }
    return run;
  }

  GpuLinearPreconvSharedQ8Run
  RunF32InputHandleSharedDeviceQ8ThenResidentRawQ6KConvStateAndResidentRawQ4KCpuOrder(
      std::uint64_t q6_handle,
      std::uint64_t alpha_beta_z_handle,
      std::uint64_t input_handle,
      std::uint64_t conv_weights_handle,
      std::uint64_t conv_state_handle,
      std::uint64_t conv_kernel_size,
      int repeat,
      bool readback_state,
      std::uint64_t next_conv_state_handle,
      bool readback_qkv,
      bool readback_conv_output,
      bool readback_alpha_beta_z) {
    const auto& resident = ResidentRawQ6KForHandle(q6_handle);
    const auto& alpha_beta_z =
        ResidentRawQ4KCpuOrderForHandle(alpha_beta_z_handle);
    const auto& input = ResidentF32BufferForHandle(input_handle);
    const auto& conv = ResidentConvWeightsForHandle(conv_weights_handle);
    const auto& conv_state = ResidentF32BufferForHandle(conv_state_handle);
    const ResidentF32Buffer* next_conv_state = nullptr;
    if (next_conv_state_handle != 0) {
      Require(next_conv_state_handle != conv_state_handle,
              "shared-Q8 next conv state must differ from current state");
      next_conv_state = &ResidentF32BufferForHandle(next_conv_state_handle);
    }
    Require(input.values == resident.blocks_per_row * kQ8QsPerBlock,
            "shared-Q8 Q6 preconv input handle size mismatch");
    Require(alpha_beta_z.blocks_per_row == resident.blocks_per_row,
            "shared-Q8 Q6 preconv alpha/beta/z block count mismatch");
    Require(conv.rows == resident.rows,
            "shared-Q8 Q6 preconv conv rows mismatch");
    Require(conv.conv_kernel_size == conv_kernel_size,
            "shared-Q8 Q6 preconv conv kernel size mismatch");
    Require(conv_state.values == resident.rows * (conv_kernel_size - 1),
            "shared-Q8 Q6 preconv conv state size mismatch");
    if (next_conv_state != nullptr) {
      Require(next_conv_state->values == conv_state.values,
              "shared-Q8 Q6 preconv next conv state size mismatch");
    }
    Require(repeat > 0, "shared-Q8 Q6 preconv repeat must be positive");

    GpuLinearPreconvSharedQ8Run run;
    run.qkv_host_valid = readback_qkv;
    run.conv_output_host_valid = readback_conv_output;
    run.conv_state_host_valid = readback_state;
    run.alpha_beta_z_host_valid = readback_alpha_beta_z;
    if (readback_qkv) {
      run.qkv_mixed.assign(static_cast<std::size_t>(resident.rows), 0.0f);
    }
    const std::size_t conv_output_values = static_cast<std::size_t>(resident.rows);
    if (readback_conv_output) {
      run.conv_output_raw.assign(conv_output_values, 0.0f);
    }
    if (readback_state) {
      run.conv_state.assign(
          static_cast<std::size_t>(resident.rows * (conv_kernel_size - 1)),
          0.0f);
    }
    if (readback_alpha_beta_z) {
      run.alpha_beta_z.assign(static_cast<std::size_t>(alpha_beta_z.rows),
                              0.0f);
    }

    const std::uint64_t block_count = resident.blocks_per_row;
    cl_mem q8_qs_buffer = EnsureScratchBuffer(
        linear_preconv_shared_q8_qs_,
        static_cast<std::size_t>(block_count) * kQ8QsPerBlock *
            sizeof(std::int8_t),
        kClMemReadWrite, "shared-Q8 preconv q8_qs");
    cl_mem q8_bsums_buffer = EnsureScratchBuffer(
        linear_preconv_shared_q8_bsums_,
        static_cast<std::size_t>(block_count) * kQ8BsumsPerBlock *
            sizeof(std::int16_t),
        kClMemReadWrite, "shared-Q8 preconv q8_bsums");
    cl_mem q8_d_buffer = EnsureScratchBuffer(
        linear_preconv_shared_q8_d_,
        static_cast<std::size_t>(block_count) * sizeof(float),
        kClMemReadWrite, "shared-Q8 preconv q8_d");
    cl_mem qkv_buffer = EnsureScratchBuffer(
        linear_preconv_shared_qkv_,
        static_cast<std::size_t>(resident.rows) * sizeof(float),
        kClMemReadWrite, "shared-Q8 Q6 preconv qkv");
    cl_mem conv_output_buffer = EnsureScratchBuffer(
        linear_preconv_shared_conv_output_, conv_output_values * sizeof(float),
        kClMemReadWrite, "shared-Q8 Q6 preconv conv output");
    cl_mem alpha_beta_z_buffer = EnsureScratchBuffer(
        linear_preconv_shared_alpha_beta_z_,
        static_cast<std::size_t>(alpha_beta_z.rows) * sizeof(float),
        kClMemReadWrite, "shared-Q8 Q6 preconv alpha/beta/z");
    cl_mem next_state_buffer = nullptr;
    if (next_conv_state == nullptr) {
      next_state_buffer = EnsureScratchBuffer(
          linear_preconv_shared_next_state_, conv_state.bytes,
          kClMemReadWrite, "shared-Q8 Q6 preconv next state");
    }

    const auto q8_timing = RunQ8QuantizeWithBsumsKernel(
        input.buffer, block_count, q8_qs_buffer, q8_bsums_buffer,
        q8_d_buffer, repeat);
    cl_mem next_state_arg =
        next_conv_state != nullptr ? next_conv_state->buffer : next_state_buffer;
    run.timing.qkv_matvec =
        resident.rowstripe
            ? RunQ6KSelectedRowstripeKernel(
                  resident.buffer, q8_qs_buffer, q8_d_buffer, qkv_buffer,
                  resident.rows, resident.blocks_per_row, resident.rows,
                  resident.rows_per_tile, repeat)
            : RunQ6KKernel(resident.buffer, q8_qs_buffer, q8_d_buffer,
                           qkv_buffer, resident.rows, resident.blocks_per_row,
                           repeat);

    const std::uint64_t conv_global = resident.rows;
    const cl_uint channel_count_arg = static_cast<cl_uint>(resident.rows);
    const cl_uint kernel_size_arg = static_cast<cl_uint>(conv_kernel_size);
    Check(api_.clSetKernelArg(kernel_conv_, 0, sizeof(qkv_buffer), &qkv_buffer),
          "clSetKernelArg(shared-Q8 Q6 preconv conv 0)");
    Check(api_.clSetKernelArg(kernel_conv_, 1, sizeof(conv_state.buffer),
                              &conv_state.buffer),
          "clSetKernelArg(shared-Q8 Q6 preconv conv 1)");
    Check(api_.clSetKernelArg(kernel_conv_, 2, sizeof(conv.buffer),
                              &conv.buffer),
          "clSetKernelArg(shared-Q8 Q6 preconv conv 2)");
    Check(api_.clSetKernelArg(kernel_conv_, 3, sizeof(channel_count_arg),
                              &channel_count_arg),
          "clSetKernelArg(shared-Q8 Q6 preconv conv 3)");
    Check(api_.clSetKernelArg(kernel_conv_, 4, sizeof(kernel_size_arg),
                              &kernel_size_arg),
          "clSetKernelArg(shared-Q8 Q6 preconv conv 4)");
    Check(api_.clSetKernelArg(kernel_conv_, 5, sizeof(conv_output_buffer),
                              &conv_output_buffer),
          "clSetKernelArg(shared-Q8 Q6 preconv conv 5)");
    Check(api_.clSetKernelArg(kernel_conv_, 6, sizeof(next_state_arg),
                              &next_state_arg),
          "clSetKernelArg(shared-Q8 Q6 preconv conv 6)");
    const std::size_t conv_global_size = static_cast<std::size_t>(conv_global);
    std::vector<double> conv_times;
    conv_times.reserve(static_cast<std::size_t>(repeat));
    for (int i = 0; i < repeat; ++i) {
      cl_event event = nullptr;
      Check(api_.clEnqueueNDRangeKernel(queue_, kernel_conv_, 1, nullptr,
                                        &conv_global_size, nullptr, 0,
                                        nullptr, EventOut(&event)),
            "clEnqueueNDRangeKernel(shared-Q8 Q6 preconv conv)");
      Check(api_.clFinish(queue_), "clFinish(shared-Q8 Q6 preconv conv)");
      conv_times.push_back(EventUs(api_, event));
      ReleaseEvent(api_, &event);
    }

    const bool alpha_read_in_kernel = readback_alpha_beta_z && repeat == 1;
    run.timing.alpha_beta_z = RunQ4KCpuOrderKernel(
        alpha_beta_z, q8_qs_buffer, q8_bsums_buffer, q8_d_buffer,
        alpha_beta_z_buffer, repeat,
        alpha_read_in_kernel ? run.alpha_beta_z.data() : nullptr,
        alpha_read_in_kernel ? run.alpha_beta_z.size() * sizeof(float) : 0);

    run.timing.q8_quantize_min_us = q8_timing.min_us;
    run.timing.q8_quantize_mean_us = q8_timing.mean_us;
    run.timing.q8_quantize_global_work_items = q8_timing.global_work_items;
    run.timing.conv_min_us =
        *std::min_element(conv_times.begin(), conv_times.end());
    run.timing.conv_mean_us =
        std::accumulate(conv_times.begin(), conv_times.end(), 0.0) /
        static_cast<double>(conv_times.size());
    run.timing.conv_global_work_items = conv_global;
    run.timing.shell_sum_min_us =
        q8_timing.min_us + run.timing.qkv_matvec.min_us +
        run.timing.conv_min_us + run.timing.alpha_beta_z.min_us;
    run.timing.shell_sum_mean_us =
        q8_timing.mean_us + run.timing.qkv_matvec.mean_us +
        run.timing.conv_mean_us + run.timing.alpha_beta_z.mean_us;

    if (next_conv_state == nullptr) {
      Check(api_.clEnqueueCopyBuffer(queue_, next_state_buffer,
                                     conv_state.buffer, 0, 0,
                                     conv_state.bytes, 0, nullptr, nullptr),
            "clEnqueueCopyBuffer(shared-Q8 Q6 preconv state)");
      Check(api_.clFinish(queue_), "clFinish(shared-Q8 Q6 preconv state copy)");
    }
    if (readback_qkv) {
      Check(api_.clEnqueueReadBuffer(queue_, qkv_buffer, kClTrue, 0,
                                     run.qkv_mixed.size() * sizeof(float),
                                     run.qkv_mixed.data(), 0, nullptr, nullptr),
            "clEnqueueReadBuffer(shared-Q8 Q6 preconv qkv)");
    }
    if (readback_alpha_beta_z && !alpha_read_in_kernel) {
      Check(api_.clEnqueueReadBuffer(queue_, alpha_beta_z_buffer, kClTrue, 0,
                                     run.alpha_beta_z.size() * sizeof(float),
                                     run.alpha_beta_z.data(), 0, nullptr,
                                     nullptr),
            "clEnqueueReadBuffer(shared-Q8 Q6 preconv alpha/beta/z)");
    }
    run.conv_output_handle = RegisterF32BufferAlias(
        &linear_preconv_shared_conv_output_alias_handle_, conv_output_buffer,
        conv_output_values);
    if (readback_conv_output) {
      Check(api_.clEnqueueReadBuffer(queue_, conv_output_buffer, kClTrue, 0,
                                     run.conv_output_raw.size() * sizeof(float),
                                     run.conv_output_raw.data(), 0, nullptr,
                                     nullptr),
            "clEnqueueReadBuffer(shared-Q8 Q6 preconv conv output)");
    }
    if (readback_state) {
      const cl_mem state_readback =
          next_conv_state != nullptr ? next_conv_state->buffer
                                     : conv_state.buffer;
      Check(api_.clEnqueueReadBuffer(queue_, state_readback, kClTrue, 0,
                                     run.conv_state.size() * sizeof(float),
                                     run.conv_state.data(), 0, nullptr,
                                     nullptr),
            "clEnqueueReadBuffer(shared-Q8 Q6 preconv state)");
    }
    return run;
  }

  GpuLinearPreconvSharedQ8Run
  RunF32InputHandleSharedDeviceQ8ThenResidentPackedQ4X8ConvStateAndResidentRawQ4KCpuOrder(
      std::uint64_t packed_handle,
      std::uint64_t alpha_beta_z_handle,
      std::uint64_t input_handle,
      std::uint64_t conv_weights_handle,
      std::uint64_t conv_state_handle,
      std::uint64_t conv_kernel_size,
      int repeat,
      GpuQ4X8KernelVariant variant,
      bool readback_state,
      std::uint64_t next_conv_state_handle,
      bool readback_qkv,
      bool readback_conv_output,
      bool readback_alpha_beta_z) {
    const auto& resident = ResidentPackedQ4X8ForHandle(packed_handle);
    const auto& alpha_beta_z =
        ResidentRawQ4KCpuOrderForHandle(alpha_beta_z_handle);
    const auto& input = ResidentF32BufferForHandle(input_handle);
    const auto& conv = ResidentConvWeightsForHandle(conv_weights_handle);
    const auto& conv_state = ResidentF32BufferForHandle(conv_state_handle);
    const ResidentF32Buffer* next_conv_state = nullptr;
    if (next_conv_state_handle != 0) {
      Require(next_conv_state_handle != conv_state_handle,
              "shared-Q8 next conv state must differ from current state");
      next_conv_state = &ResidentF32BufferForHandle(next_conv_state_handle);
    }
    Require(input.values == resident.blocks_per_row * kQ8QsPerBlock,
            "shared-Q8 Q4 preconv input handle size mismatch");
    Require(alpha_beta_z.blocks_per_row == resident.blocks_per_row,
            "shared-Q8 Q4 preconv alpha/beta/z block count mismatch");
    Require(conv.rows == resident.rows,
            "shared-Q8 Q4 preconv conv rows mismatch");
    Require(conv.conv_kernel_size == conv_kernel_size,
            "shared-Q8 Q4 preconv conv kernel size mismatch");
    Require(conv_state.values == resident.rows * (conv_kernel_size - 1),
            "shared-Q8 Q4 preconv conv state size mismatch");
    if (next_conv_state != nullptr) {
      Require(next_conv_state->values == conv_state.values,
              "shared-Q8 Q4 preconv next conv state size mismatch");
    }
    Require(repeat > 0, "shared-Q8 Q4 preconv repeat must be positive");

    GpuLinearPreconvSharedQ8Run run;
    run.qkv_host_valid = readback_qkv;
    run.conv_output_host_valid = readback_conv_output;
    run.conv_state_host_valid = readback_state;
    run.alpha_beta_z_host_valid = readback_alpha_beta_z;
    if (readback_qkv) {
      run.qkv_mixed.assign(static_cast<std::size_t>(resident.rows), 0.0f);
    }
    const std::size_t conv_output_values = static_cast<std::size_t>(resident.rows);
    if (readback_conv_output) {
      run.conv_output_raw.assign(conv_output_values, 0.0f);
    }
    if (readback_state) {
      run.conv_state.assign(
          static_cast<std::size_t>(resident.rows * (conv_kernel_size - 1)),
          0.0f);
    }
    if (readback_alpha_beta_z) {
      run.alpha_beta_z.assign(static_cast<std::size_t>(alpha_beta_z.rows),
                              0.0f);
    }

    const std::uint64_t block_count = resident.blocks_per_row;
    cl_mem q8_qs_buffer = EnsureScratchBuffer(
        linear_preconv_shared_q8_qs_,
        static_cast<std::size_t>(block_count) * kQ8QsPerBlock *
            sizeof(std::int8_t),
        kClMemReadWrite, "shared-Q8 preconv q8_qs");
    cl_mem q8_bsums_buffer = EnsureScratchBuffer(
        linear_preconv_shared_q8_bsums_,
        static_cast<std::size_t>(block_count) * kQ8BsumsPerBlock *
            sizeof(std::int16_t),
        kClMemReadWrite, "shared-Q8 preconv q8_bsums");
    cl_mem q8_d_buffer = EnsureScratchBuffer(
        linear_preconv_shared_q8_d_,
        static_cast<std::size_t>(block_count) * sizeof(float),
        kClMemReadWrite, "shared-Q8 preconv q8_d");
    cl_mem qkv_buffer = EnsureScratchBuffer(
        linear_preconv_shared_qkv_,
        static_cast<std::size_t>(resident.rows) * sizeof(float),
        kClMemReadWrite, "shared-Q8 Q4 preconv qkv");
    cl_mem conv_output_buffer = EnsureScratchBuffer(
        linear_preconv_shared_conv_output_, conv_output_values * sizeof(float),
        kClMemReadWrite, "shared-Q8 Q4 preconv conv output");
    cl_mem alpha_beta_z_buffer = EnsureScratchBuffer(
        linear_preconv_shared_alpha_beta_z_,
        static_cast<std::size_t>(alpha_beta_z.rows) * sizeof(float),
        kClMemReadWrite, "shared-Q8 Q4 preconv alpha/beta/z");
    cl_mem next_state_buffer = nullptr;
    if (next_conv_state == nullptr) {
      next_state_buffer = EnsureScratchBuffer(
          linear_preconv_shared_next_state_, conv_state.bytes,
          kClMemReadWrite, "shared-Q8 Q4 preconv next state");
    }

    const auto q8_timing = RunQ8QuantizeWithBsumsKernel(
        input.buffer, block_count, q8_qs_buffer, q8_bsums_buffer,
        q8_d_buffer, repeat);
    cl_mem next_state_arg =
        next_conv_state != nullptr ? next_conv_state->buffer : next_state_buffer;
    const std::uint64_t row_groups = resident.rows / kRowsInterleaved;
    const std::uint64_t matvec_global =
        variant == GpuQ4X8KernelVariant::kGroup8Serial ? row_groups
                                                       : resident.rows;
    const auto handoff_timing = RunHandoffKernels(
        variant, resident.buffer, q8_qs_buffer, q8_bsums_buffer, q8_d_buffer,
        qkv_buffer, conv.buffer, conv_state.buffer, conv_output_buffer,
        next_state_arg, resident.blocks_per_row, row_groups, matvec_global,
        resident.rows, conv_kernel_size, repeat);
    const bool alpha_read_in_kernel = readback_alpha_beta_z && repeat == 1;
    run.timing.alpha_beta_z = RunQ4KCpuOrderKernel(
        alpha_beta_z, q8_qs_buffer, q8_bsums_buffer, q8_d_buffer,
        alpha_beta_z_buffer, repeat,
        alpha_read_in_kernel ? run.alpha_beta_z.data() : nullptr,
        alpha_read_in_kernel ? run.alpha_beta_z.size() * sizeof(float) : 0);

    run.timing.q8_quantize_min_us = q8_timing.min_us;
    run.timing.q8_quantize_mean_us = q8_timing.mean_us;
    run.timing.q8_quantize_global_work_items = q8_timing.global_work_items;
    run.timing.qkv_matvec = handoff_timing.matvec;
    run.timing.conv_min_us = handoff_timing.conv_min_us;
    run.timing.conv_mean_us = handoff_timing.conv_mean_us;
    run.timing.conv_global_work_items = handoff_timing.conv_global_work_items;
    run.timing.shell_sum_min_us =
        q8_timing.min_us + handoff_timing.matvec.min_us +
        handoff_timing.conv_min_us + run.timing.alpha_beta_z.min_us;
    run.timing.shell_sum_mean_us =
        q8_timing.mean_us + handoff_timing.matvec.mean_us +
        handoff_timing.conv_mean_us + run.timing.alpha_beta_z.mean_us;

    if (next_conv_state == nullptr) {
      Check(api_.clEnqueueCopyBuffer(queue_, next_state_buffer,
                                     conv_state.buffer, 0, 0,
                                     conv_state.bytes, 0, nullptr, nullptr),
            "clEnqueueCopyBuffer(shared-Q8 Q4 preconv state)");
      Check(api_.clFinish(queue_), "clFinish(shared-Q8 Q4 preconv state copy)");
    }
    if (readback_qkv) {
      Check(api_.clEnqueueReadBuffer(queue_, qkv_buffer, kClTrue, 0,
                                     run.qkv_mixed.size() * sizeof(float),
                                     run.qkv_mixed.data(), 0, nullptr, nullptr),
            "clEnqueueReadBuffer(shared-Q8 Q4 preconv qkv)");
    }
    if (readback_alpha_beta_z && !alpha_read_in_kernel) {
      Check(api_.clEnqueueReadBuffer(queue_, alpha_beta_z_buffer, kClTrue, 0,
                                     run.alpha_beta_z.size() * sizeof(float),
                                     run.alpha_beta_z.data(), 0, nullptr,
                                     nullptr),
            "clEnqueueReadBuffer(shared-Q8 Q4 preconv alpha/beta/z)");
    }
    run.conv_output_handle = RegisterF32BufferAlias(
        &linear_preconv_shared_conv_output_alias_handle_, conv_output_buffer,
        conv_output_values);
    if (readback_conv_output) {
      Check(api_.clEnqueueReadBuffer(queue_, conv_output_buffer, kClTrue, 0,
                                     run.conv_output_raw.size() * sizeof(float),
                                     run.conv_output_raw.data(), 0, nullptr,
                                     nullptr),
            "clEnqueueReadBuffer(shared-Q8 Q4 preconv conv output)");
    }
    if (readback_state) {
      const cl_mem state_readback =
          next_conv_state != nullptr ? next_conv_state->buffer
                                     : conv_state.buffer;
      Check(api_.clEnqueueReadBuffer(queue_, state_readback, kClTrue, 0,
                                     run.conv_state.size() * sizeof(float),
                                     run.conv_state.data(), 0, nullptr,
                                     nullptr),
            "clEnqueueReadBuffer(shared-Q8 Q4 preconv state)");
    }
    return run;
  }

  void ClearResidentRawQ4KCpuOrder() {
    for (auto& item : resident_raw_q4_cpu_order_) {
      ReleaseMem(api_, &item.second.buffer);
    }
    resident_raw_q4_cpu_order_.clear();
  }

  void ClearResidentConvWeights() {
    for (auto& item : resident_conv_weights_) {
      ReleaseMem(api_, &item.second.buffer);
    }
    resident_conv_weights_.clear();
  }

  void ClearResidentPackedQ4X8() {
    for (auto& item : resident_packed_q4x8_) {
      ReleaseMem(api_, &item.second.buffer);
    }
    resident_packed_q4x8_.clear();
  }

  std::uint64_t UploadF32MatvecWeights(const std::vector<float>& weights,
                                       std::uint64_t rows,
                                       std::uint64_t cols) {
    ValidateF32MatvecWeights(weights, rows, cols);
    cl_int err = kClSuccess;
    cl_mem buffer = api_.clCreateBuffer(
        context_, kClMemReadOnly, weights.size() * sizeof(float), nullptr, &err);
    Check(err, "clCreateBuffer(resident F32 weights)");
    try {
      Check(api_.clEnqueueWriteBuffer(queue_, buffer, kClTrue, 0,
                                      weights.size() * sizeof(float),
                                      weights.data(), 0, nullptr, nullptr),
            "clEnqueueWriteBuffer(resident F32 weights)");
    } catch (...) {
      ReleaseMem(api_, &buffer);
      throw;
    }
    const std::uint64_t handle = next_resident_handle_++;
    resident_f32_matvec_.emplace(
        handle, ResidentF32Matvec{buffer, rows, cols,
                                  weights.size() * sizeof(float)});
    return handle;
  }

  GpuF32MatvecRun RunResidentF32Matvec(std::uint64_t handle,
                                       const std::vector<float>& input,
                                       int repeat) {
    const auto& resident = ResidentF32MatvecForHandle(handle);
    ValidateF32MatvecInput(input, resident.cols, repeat);
    GpuF32MatvecRun run;
    run.output.assign(static_cast<std::size_t>(resident.rows), 0.0f);
    cl_mem input_buffer = nullptr, out_buffer = nullptr;
    auto release_all = [&]() {
      ReleaseMem(api_, &out_buffer);
      ReleaseMem(api_, &input_buffer);
    };
    try {
      CreateF32MatvecBuffersWithoutWeights(input, run.output, &input_buffer,
                                           &out_buffer);
      run.timing = RunF32MatvecKernel(resident.buffer, input_buffer, out_buffer,
                                      resident.rows, resident.cols, repeat);
      Check(api_.clEnqueueReadBuffer(queue_, out_buffer, kClTrue, 0,
                                     run.output.size() * sizeof(float),
                                     run.output.data(), 0, nullptr, nullptr),
            "clEnqueueReadBuffer(resident F32 matvec out)");
    } catch (...) {
      release_all();
      throw;
    }
    release_all();
    return run;
  }

  GpuF32MatvecRun RunResidentF32MatvecFromInputHandle(
      std::uint64_t handle,
      std::uint64_t input_handle,
      int repeat,
      bool readback_output = true) {
    const auto& resident = ResidentF32MatvecForHandle(handle);
    const auto& input = ResidentF32BufferForHandle(input_handle);
    Require(input.values == resident.cols,
            "resident F32 matvec input handle size mismatch");
    Require(repeat > 0, "resident F32 matvec repeat must be positive");
    GpuF32MatvecRun run;
    run.output_host_valid = readback_output;
    if (readback_output) {
      run.output.assign(static_cast<std::size_t>(resident.rows), 0.0f);
    }
    cl_mem out_buffer = EnsureScratchBuffer(
        resident_f32_matvec_scratch_output_,
        static_cast<std::size_t>(resident.rows) * sizeof(float),
        kClMemReadWrite, "resident F32 matvec handle out");
    try {
      run.timing = RunF32MatvecKernel(
          resident.buffer, input.buffer, out_buffer, resident.rows,
          resident.cols, repeat);
      if (readback_output) {
        Check(api_.clEnqueueReadBuffer(
                  queue_, out_buffer, kClTrue, 0,
                  run.output.size() * sizeof(float), run.output.data(), 0,
                  nullptr, nullptr),
              "clEnqueueReadBuffer(resident F32 matvec handle out)");
      }
      run.output_handle = RegisterF32BufferAlias(
          &resident_f32_matvec_output_alias_handle_, out_buffer,
          static_cast<std::size_t>(resident.rows));
    } catch (...) {
      throw;
    }
    return run;
  }

  GpuF32TopKRun RunResidentF32TopK(std::uint64_t values_handle,
                                   int topk,
                                   int repeat) {
    const auto& values = ResidentF32BufferForHandle(values_handle);
    Require(topk > 0 && topk <= 16, "resident F32 top-k must be in 1..16");
    Require(repeat > 0, "resident F32 top-k repeat must be positive");
    constexpr std::uint64_t kTopKBlockSize = 256;
    const std::uint64_t topk_slots = topk <= 8 ? 8 : 16;
    const std::uint64_t partial_count =
        (values.values + kTopKBlockSize - 1) / kTopKBlockSize;
    GpuF32TopKRun run;
    cl_mem partial_ids_buffer = nullptr;
    cl_mem partial_values_buffer = nullptr;
    auto release_all = [&]() {
      ReleaseMem(api_, &partial_values_buffer);
      ReleaseMem(api_, &partial_ids_buffer);
    };
    try {
      cl_int err = kClSuccess;
      partial_ids_buffer = api_.clCreateBuffer(
          context_, kClMemWriteOnly,
          static_cast<std::size_t>(partial_count * topk_slots) *
              sizeof(std::int32_t),
          nullptr, &err);
      Check(err, "clCreateBuffer(F32 top-k partial ids)");
      partial_values_buffer = api_.clCreateBuffer(
          context_, kClMemWriteOnly,
          static_cast<std::size_t>(partial_count * topk_slots) *
              sizeof(float),
          nullptr, &err);
      Check(err, "clCreateBuffer(F32 top-k partial values)");

      run.timing =
          topk <= 8
              ? RunF32TopK8BlocksKernel(
                    values.buffer,
                    static_cast<std::uint64_t>(values.values),
                    kTopKBlockSize, partial_ids_buffer,
                    partial_values_buffer, partial_count, repeat)
              : RunF32TopK16BlocksKernel(
                    values.buffer,
                    static_cast<std::uint64_t>(values.values),
                    kTopKBlockSize, partial_ids_buffer,
                    partial_values_buffer, partial_count, repeat);

      std::vector<std::int32_t> partial_ids(
          static_cast<std::size_t>(partial_count * topk_slots), -1);
      std::vector<float> partial_values(
          static_cast<std::size_t>(partial_count * topk_slots),
          -std::numeric_limits<float>::infinity());
      Check(api_.clEnqueueReadBuffer(
                queue_, partial_ids_buffer, kClTrue, 0,
                partial_ids.size() * sizeof(std::int32_t),
                partial_ids.data(), 0, nullptr, nullptr),
            "clEnqueueReadBuffer(F32 top-k partial ids)");
      Check(api_.clEnqueueReadBuffer(
                queue_, partial_values_buffer, kClTrue, 0,
                partial_values.size() * sizeof(float), partial_values.data(),
                0, nullptr, nullptr),
            "clEnqueueReadBuffer(F32 top-k partial values)");
      for (std::size_t i = 0; i < partial_values.size(); ++i) {
        if (partial_ids[i] < 0) {
          continue;
        }
        InsertGpuTopK(run.topk, GpuTopKRow{partial_ids[i], partial_values[i]},
                      topk);
      }
    } catch (...) {
      release_all();
      throw;
    }
    release_all();
    return run;
  }

  void ClearResidentF32Matvec() {
    for (auto& item : resident_f32_matvec_) {
      ReleaseMem(api_, &item.second.buffer);
    }
    resident_f32_matvec_.clear();
  }

  std::uint64_t UploadF32Buffer(const std::vector<float>& values) {
    Require(!values.empty(), "resident F32 buffer must be nonempty");
    cl_int err = kClSuccess;
    cl_mem buffer = api_.clCreateBuffer(
        context_, kClMemReadWrite, values.size() * sizeof(float), nullptr, &err);
    Check(err, "clCreateBuffer(resident F32 buffer)");
    try {
      Check(api_.clEnqueueWriteBuffer(queue_, buffer, kClTrue, 0,
                                      values.size() * sizeof(float),
                                      values.data(), 0, nullptr, nullptr),
            "clEnqueueWriteBuffer(resident F32 buffer)");
    } catch (...) {
      ReleaseMem(api_, &buffer);
      throw;
    }
    const std::uint64_t handle = next_resident_handle_++;
    resident_f32_buffers_.emplace(
        handle, ResidentF32Buffer{buffer, values.size(),
                                  values.size() * sizeof(float)});
    return handle;
  }

  std::uint64_t CloneResidentF32Buffer(std::uint64_t source_handle) {
    const auto& source = ResidentF32BufferForHandle(source_handle);
    Require(source.buffer != nullptr, "resident F32 clone source is missing");
    Require(source.values > 0, "resident F32 clone source is empty");
    cl_int err = kClSuccess;
    cl_mem buffer =
        api_.clCreateBuffer(context_, kClMemReadWrite, source.bytes, nullptr,
                            &err);
    Check(err, "clCreateBuffer(resident F32 clone)");
    try {
      Check(api_.clEnqueueCopyBuffer(queue_, source.buffer, buffer, 0, 0,
                                     source.bytes, 0, nullptr, nullptr),
            "clEnqueueCopyBuffer(resident F32 clone)");
      Check(api_.clFinish(queue_), "clFinish(resident F32 clone)");
    } catch (...) {
      ReleaseMem(api_, &buffer);
      throw;
    }
    const std::uint64_t handle = next_resident_handle_++;
    resident_f32_buffers_.emplace(
        handle, ResidentF32Buffer{buffer, source.values, source.bytes});
    return handle;
  }

  GpuRouterQkvDeltaSelectedValueOverlayRun
  RunRouterQkvDeltaSelectedValueOverlay(
      std::uint64_t base_handle,
      std::uint64_t source_handle,
      const std::vector<std::int32_t>& selected_indices,
      int repeat,
      bool readback_output = true) {
    const auto& base = ResidentF32BufferForHandle(base_handle);
    const auto& source = ResidentF32BufferForHandle(source_handle);
    Require(base.buffer != nullptr, "qkv-delta sparse overlay base missing");
    Require(source.buffer != nullptr, "qkv-delta sparse overlay source missing");
    Require(base.values == source.values,
            "qkv-delta sparse overlay source size mismatch");
    Require(base.values > 0, "qkv-delta sparse overlay base is empty");
    Require(!selected_indices.empty(),
            "qkv-delta sparse overlay selected indices are empty");
    Require(repeat > 0, "qkv-delta sparse overlay repeat must be positive");
    for (std::int32_t index : selected_indices) {
      Require(index >= 0 &&
                  static_cast<std::size_t>(index) < base.values,
              "qkv-delta sparse overlay selected index out of range");
    }

    GpuRouterQkvDeltaSelectedValueOverlayRun run;
    run.output_host_valid = readback_output;
    if (readback_output) {
      run.output.assign(base.values, 0.0f);
    }

    cl_mem out_buffer = nullptr;
    cl_int err = kClSuccess;
    out_buffer =
        api_.clCreateBuffer(context_, kClMemReadWrite, base.bytes, nullptr, &err);
    Check(err, "clCreateBuffer(qkv-delta sparse overlay out)");
    try {
      cl_mem indices_buffer = EnsureScratchBuffer(
          qkv_delta_sparse_overlay_scratch_indices_,
          selected_indices.size() * sizeof(std::int32_t), kClMemReadOnly,
          "qkv-delta sparse overlay indices");
      Check(api_.clEnqueueWriteBuffer(
                queue_, indices_buffer, kClFalse, 0,
                selected_indices.size() * sizeof(std::int32_t),
                selected_indices.data(), 0, nullptr, nullptr),
            "clEnqueueWriteBuffer(qkv-delta sparse overlay indices)");
      Check(api_.clEnqueueCopyBuffer(queue_, base.buffer, out_buffer, 0, 0,
                                     base.bytes, 0, nullptr, nullptr),
            "clEnqueueCopyBuffer(qkv-delta sparse overlay base)");
      const cl_uint selected_count =
          static_cast<cl_uint>(selected_indices.size());
      Check(api_.clSetKernelArg(kernel_qkv_delta_sparse_overlay_, 0,
                                sizeof(source.buffer), &source.buffer),
            "clSetKernelArg(qkv-delta sparse overlay 0)");
      Check(api_.clSetKernelArg(kernel_qkv_delta_sparse_overlay_, 1,
                                sizeof(indices_buffer), &indices_buffer),
            "clSetKernelArg(qkv-delta sparse overlay 1)");
      Check(api_.clSetKernelArg(kernel_qkv_delta_sparse_overlay_, 2,
                                sizeof(selected_count), &selected_count),
            "clSetKernelArg(qkv-delta sparse overlay 2)");
      Check(api_.clSetKernelArg(kernel_qkv_delta_sparse_overlay_, 3,
                                sizeof(out_buffer), &out_buffer),
            "clSetKernelArg(qkv-delta sparse overlay 3)");

      const std::size_t global = selected_indices.size();
      std::vector<double> times;
      times.reserve(static_cast<std::size_t>(repeat));
      for (int i = 0; i < repeat; ++i) {
        cl_event event = nullptr;
        Check(api_.clEnqueueNDRangeKernel(
                  queue_, kernel_qkv_delta_sparse_overlay_, 1, nullptr,
                  &global, nullptr, 0, nullptr, EventOut(&event)),
              "clEnqueueNDRangeKernel(qkv_delta_sparse_overlay_f32)");
        Check(api_.clFinish(queue_), "clFinish(qkv-delta sparse overlay)");
        times.push_back(EventUs(api_, event));
        ReleaseEvent(api_, &event);
      }
      run.timing.min_us = *std::min_element(times.begin(), times.end());
      run.timing.mean_us =
          std::accumulate(times.begin(), times.end(), 0.0) /
          static_cast<double>(times.size());
      run.timing.effective_weight_gb_s =
          static_cast<double>(base.bytes +
                              selected_indices.size() *
                                  (sizeof(float) + sizeof(std::int32_t))) /
          (run.timing.min_us / 1e6) / 1e9;
      run.timing.global_work_items = selected_indices.size();
      if (readback_output) {
        Check(api_.clEnqueueReadBuffer(queue_, out_buffer, kClTrue, 0,
                                       run.output.size() * sizeof(float),
                                       run.output.data(), 0, nullptr, nullptr),
              "clEnqueueReadBuffer(qkv-delta sparse overlay out)");
      }
      const std::uint64_t handle = next_resident_handle_++;
      resident_f32_buffers_.emplace(
          handle, ResidentF32Buffer{out_buffer, base.values, base.bytes});
      out_buffer = nullptr;
      run.output_handle = handle;
    } catch (...) {
      ReleaseMem(api_, &out_buffer);
      throw;
    }
    return run;
  }

  GpuRouterQkvDeltaSelectedValueOverlayRun
  RunRouterQkvDeltaBlockQ16Overlay(
      std::uint64_t base_handle,
      const std::vector<std::int32_t>& selected_indices,
      const std::vector<std::int16_t>& selected_q_delta,
      const std::vector<float>& block_scales,
      int repeat,
      bool readback_output = true) {
    const auto& base = ResidentF32BufferForHandle(base_handle);
    Require(base.buffer != nullptr, "qkv-delta block-q16 overlay base missing");
    Require(base.values > 0, "qkv-delta block-q16 overlay base is empty");
    Require(!selected_indices.empty(),
            "qkv-delta block-q16 overlay selected indices are empty");
    Require(selected_q_delta.size() == selected_indices.size(),
            "qkv-delta block-q16 overlay delta size mismatch");
    const std::size_t block_count = (base.values + 63) / 64;
    Require(block_scales.size() >= block_count,
            "qkv-delta block-q16 overlay scale size mismatch");
    Require(repeat > 0, "qkv-delta block-q16 overlay repeat must be positive");
    for (std::int32_t index : selected_indices) {
      Require(index >= 0 &&
                  static_cast<std::size_t>(index) < base.values,
              "qkv-delta block-q16 overlay selected index out of range");
    }

    GpuRouterQkvDeltaSelectedValueOverlayRun run;
    run.output_host_valid = readback_output;
    if (readback_output) {
      run.output.assign(base.values, 0.0f);
    }

    cl_mem out_buffer = nullptr;
    cl_int err = kClSuccess;
    out_buffer =
        api_.clCreateBuffer(context_, kClMemReadWrite, base.bytes, nullptr, &err);
    Check(err, "clCreateBuffer(qkv-delta block-q16 overlay out)");
    try {
      cl_mem indices_buffer = EnsureScratchBuffer(
          qkv_delta_blockq16_overlay_scratch_indices_,
          selected_indices.size() * sizeof(std::int32_t), kClMemReadOnly,
          "qkv-delta block-q16 overlay indices");
      cl_mem q_delta_buffer = EnsureScratchBuffer(
          qkv_delta_blockq16_overlay_scratch_q_delta_,
          selected_q_delta.size() * sizeof(std::int16_t), kClMemReadOnly,
          "qkv-delta block-q16 overlay q-delta");
      cl_mem scales_buffer = EnsureScratchBuffer(
          qkv_delta_blockq16_overlay_scratch_scales_,
          block_count * sizeof(float), kClMemReadOnly,
          "qkv-delta block-q16 overlay scales");
      Check(api_.clEnqueueWriteBuffer(
                queue_, indices_buffer, kClFalse, 0,
                selected_indices.size() * sizeof(std::int32_t),
                selected_indices.data(), 0, nullptr, nullptr),
            "clEnqueueWriteBuffer(qkv-delta block-q16 overlay indices)");
      Check(api_.clEnqueueWriteBuffer(
                queue_, q_delta_buffer, kClFalse, 0,
                selected_q_delta.size() * sizeof(std::int16_t),
                selected_q_delta.data(), 0, nullptr, nullptr),
            "clEnqueueWriteBuffer(qkv-delta block-q16 overlay q-delta)");
      Check(api_.clEnqueueWriteBuffer(
                queue_, scales_buffer, kClFalse, 0,
                block_count * sizeof(float), block_scales.data(), 0, nullptr,
                nullptr),
            "clEnqueueWriteBuffer(qkv-delta block-q16 overlay scales)");
      Check(api_.clEnqueueCopyBuffer(queue_, base.buffer, out_buffer, 0, 0,
                                     base.bytes, 0, nullptr, nullptr),
            "clEnqueueCopyBuffer(qkv-delta block-q16 overlay base)");
      const cl_uint selected_count =
          static_cast<cl_uint>(selected_indices.size());
      Check(api_.clSetKernelArg(kernel_qkv_delta_blockq16_overlay_, 0,
                                sizeof(base.buffer), &base.buffer),
            "clSetKernelArg(qkv-delta block-q16 overlay 0)");
      Check(api_.clSetKernelArg(kernel_qkv_delta_blockq16_overlay_, 1,
                                sizeof(indices_buffer), &indices_buffer),
            "clSetKernelArg(qkv-delta block-q16 overlay 1)");
      Check(api_.clSetKernelArg(kernel_qkv_delta_blockq16_overlay_, 2,
                                sizeof(q_delta_buffer), &q_delta_buffer),
            "clSetKernelArg(qkv-delta block-q16 overlay 2)");
      Check(api_.clSetKernelArg(kernel_qkv_delta_blockq16_overlay_, 3,
                                sizeof(scales_buffer), &scales_buffer),
            "clSetKernelArg(qkv-delta block-q16 overlay 3)");
      Check(api_.clSetKernelArg(kernel_qkv_delta_blockq16_overlay_, 4,
                                sizeof(selected_count), &selected_count),
            "clSetKernelArg(qkv-delta block-q16 overlay 4)");
      Check(api_.clSetKernelArg(kernel_qkv_delta_blockq16_overlay_, 5,
                                sizeof(out_buffer), &out_buffer),
            "clSetKernelArg(qkv-delta block-q16 overlay 5)");

      const std::size_t global = selected_indices.size();
      std::vector<double> times;
      times.reserve(static_cast<std::size_t>(repeat));
      for (int i = 0; i < repeat; ++i) {
        cl_event event = nullptr;
        Check(api_.clEnqueueNDRangeKernel(
                  queue_, kernel_qkv_delta_blockq16_overlay_, 1, nullptr,
                  &global, nullptr, 0, nullptr, EventOut(&event)),
              "clEnqueueNDRangeKernel(qkv_delta_blockq16_overlay_f32)");
        Check(api_.clFinish(queue_), "clFinish(qkv-delta block-q16 overlay)");
        times.push_back(EventUs(api_, event));
        ReleaseEvent(api_, &event);
      }
      run.timing.min_us = *std::min_element(times.begin(), times.end());
      run.timing.mean_us =
          std::accumulate(times.begin(), times.end(), 0.0) /
          static_cast<double>(times.size());
      run.timing.effective_weight_gb_s =
          static_cast<double>(base.bytes +
                              selected_indices.size() *
                                  (sizeof(std::int32_t) +
                                   sizeof(std::int16_t)) +
                              block_count * sizeof(float)) /
          (run.timing.min_us / 1e6) / 1e9;
      run.timing.global_work_items = selected_indices.size();
      if (readback_output) {
        Check(api_.clEnqueueReadBuffer(queue_, out_buffer, kClTrue, 0,
                                       run.output.size() * sizeof(float),
                                       run.output.data(), 0, nullptr, nullptr),
              "clEnqueueReadBuffer(qkv-delta block-q16 overlay out)");
      }
      const std::uint64_t handle = next_resident_handle_++;
      resident_f32_buffers_.emplace(
          handle, ResidentF32Buffer{out_buffer, base.values, base.bytes});
      out_buffer = nullptr;
      run.output_handle = handle;
    } catch (...) {
      ReleaseMem(api_, &out_buffer);
      throw;
    }
    return run;
  }

  std::uint64_t RegisterF32BufferAlias(std::uint64_t* handle_slot,
                                       cl_mem buffer,
                                       std::size_t values) {
    Require(handle_slot != nullptr, "resident F32 alias handle slot missing");
    Require(buffer != nullptr, "resident F32 alias buffer missing");
    Require(values > 0, "resident F32 alias must be nonempty");
    if (*handle_slot == 0) {
      *handle_slot = next_resident_handle_++;
    }
    resident_f32_buffers_[*handle_slot] =
        ResidentF32Buffer{buffer, values, values * sizeof(float), false};
    return *handle_slot;
  }

  void ClearResidentF32Buffers() {
    for (auto& item : resident_f32_buffers_) {
      if (item.second.owned) {
        ReleaseMem(api_, &item.second.buffer);
      }
    }
    resident_f32_buffers_.clear();
  }

  GpuQ4X8MatvecRun Run(const std::vector<std::uint8_t>& packed,
                       const std::vector<std::int8_t>& q8_qs,
                       const std::vector<std::int16_t>& q8_bsums,
                       const std::vector<float>& q8_d,
                       std::uint64_t rows,
                       std::uint64_t blocks_per_row,
                       int repeat,
                       GpuQ4X8KernelVariant variant) {
    ValidateRunInputs(packed, q8_qs, q8_bsums, q8_d, rows, blocks_per_row, repeat);
    GpuQ4X8MatvecRun run;
    run.output.assign(static_cast<std::size_t>(rows), 0.0f);
    cl_mem packed_buffer = nullptr, q8_qs_buffer = nullptr, q8_bsums_buffer = nullptr;
    cl_mem q8_d_buffer = nullptr, out_buffer = nullptr;
    try {
      CreateBuffers(packed, q8_qs, q8_bsums, q8_d, run.output,
                    &packed_buffer, &q8_qs_buffer, &q8_bsums_buffer, &q8_d_buffer, &out_buffer);
      const std::uint64_t row_groups = rows / kRowsInterleaved;
      const std::uint64_t global_work_items =
          variant == GpuQ4X8KernelVariant::kGroup8Serial ? row_groups : rows;
      run.timing = RunKernel(variant, packed_buffer, q8_qs_buffer, q8_bsums_buffer, q8_d_buffer,
                             out_buffer, blocks_per_row, row_groups, global_work_items, repeat);
      Check(api_.clEnqueueReadBuffer(queue_, out_buffer, kClTrue, 0,
                                     run.output.size() * sizeof(float), run.output.data(),
                                     0, nullptr, nullptr),
            "clEnqueueReadBuffer(out)");
    } catch (...) {
      ReleaseMem(api_, &out_buffer);
      ReleaseMem(api_, &q8_d_buffer);
      ReleaseMem(api_, &q8_bsums_buffer);
      ReleaseMem(api_, &q8_qs_buffer);
      ReleaseMem(api_, &packed_buffer);
      throw;
    }
    ReleaseMem(api_, &out_buffer);
    ReleaseMem(api_, &q8_d_buffer);
    ReleaseMem(api_, &q8_bsums_buffer);
    ReleaseMem(api_, &q8_qs_buffer);
	    ReleaseMem(api_, &packed_buffer);
	    return run;
	  }

  GpuQ4X8MatvecRun RunRowblock16(
      const std::vector<std::uint8_t>& packed,
      const std::vector<std::int8_t>& q8_qs,
      const std::vector<std::int16_t>& q8_bsums,
      const std::vector<float>& q8_d,
      std::uint64_t rows,
      std::uint64_t blocks_per_row,
      int repeat) {
    ValidateRunInputs(packed, q8_qs, q8_bsums, q8_d, rows, blocks_per_row,
                      repeat);
    Require(blocks_per_row == 16,
            "rowblock16 Q4 matvec requires blocks_per_row == 16");
    GpuQ4X8MatvecRun run;
    run.output.assign(static_cast<std::size_t>(rows), 0.0f);
    cl_mem packed_buffer = nullptr, q8_qs_buffer = nullptr;
    cl_mem q8_bsums_buffer = nullptr, q8_d_buffer = nullptr;
    cl_mem out_buffer = nullptr;
    try {
      CreateBuffers(packed, q8_qs, q8_bsums, q8_d, run.output,
                    &packed_buffer, &q8_qs_buffer, &q8_bsums_buffer,
                    &q8_d_buffer, &out_buffer);
      const std::uint64_t row_groups = rows / kRowsInterleaved;
      run.timing = RunRowblock16Kernel(
          packed_buffer, q8_qs_buffer, q8_bsums_buffer, q8_d_buffer,
          out_buffer, rows, blocks_per_row, row_groups, repeat);
      Check(api_.clEnqueueReadBuffer(queue_, out_buffer, kClTrue, 0,
                                     run.output.size() * sizeof(float),
                                     run.output.data(), 0, nullptr, nullptr),
            "clEnqueueReadBuffer(rowblock16 out)");
    } catch (...) {
      ReleaseMem(api_, &out_buffer);
      ReleaseMem(api_, &q8_d_buffer);
      ReleaseMem(api_, &q8_bsums_buffer);
      ReleaseMem(api_, &q8_qs_buffer);
      ReleaseMem(api_, &packed_buffer);
      throw;
    }
    ReleaseMem(api_, &out_buffer);
    ReleaseMem(api_, &q8_d_buffer);
    ReleaseMem(api_, &q8_bsums_buffer);
    ReleaseMem(api_, &q8_qs_buffer);
    ReleaseMem(api_, &packed_buffer);
    return run;
  }

  GpuQ4X8MatvecRun RunRowblock16CpuOrderFinalize(
      const std::vector<std::uint8_t>& packed,
      const std::vector<std::int8_t>& q8_qs,
      const std::vector<std::int16_t>& q8_bsums,
      const std::vector<float>& q8_d,
      std::uint64_t rows,
      std::uint64_t blocks_per_row,
      int repeat) {
    ValidateRunInputs(packed, q8_qs, q8_bsums, q8_d, rows, blocks_per_row,
                      repeat);
    Require(blocks_per_row == 16,
            "rowblock16 CPU-order finalize requires blocks_per_row == 16");
    GpuQ4X8MatvecRun run;
    run.output.assign(static_cast<std::size_t>(rows), 0.0f);
    cl_mem packed_buffer = nullptr, q8_qs_buffer = nullptr;
    cl_mem q8_bsums_buffer = nullptr, q8_d_buffer = nullptr;
    cl_mem out_buffer = nullptr;
    try {
      CreateBuffers(packed, q8_qs, q8_bsums, q8_d, run.output,
                    &packed_buffer, &q8_qs_buffer, &q8_bsums_buffer,
                    &q8_d_buffer, &out_buffer);
      const std::uint64_t row_groups = rows / kRowsInterleaved;
      run.timing = RunRowblock16Kernel(
          packed_buffer, q8_qs_buffer, q8_bsums_buffer, q8_d_buffer,
          out_buffer, rows, blocks_per_row, row_groups, repeat, true);
      Check(api_.clEnqueueReadBuffer(queue_, out_buffer, kClTrue, 0,
                                     run.output.size() * sizeof(float),
                                     run.output.data(), 0, nullptr, nullptr),
            "clEnqueueReadBuffer(rowblock16 CPU-order finalize out)");
    } catch (...) {
      ReleaseMem(api_, &out_buffer);
      ReleaseMem(api_, &q8_d_buffer);
      ReleaseMem(api_, &q8_bsums_buffer);
      ReleaseMem(api_, &q8_qs_buffer);
      ReleaseMem(api_, &packed_buffer);
      throw;
    }
    ReleaseMem(api_, &out_buffer);
    ReleaseMem(api_, &q8_d_buffer);
    ReleaseMem(api_, &q8_bsums_buffer);
    ReleaseMem(api_, &q8_qs_buffer);
    ReleaseMem(api_, &packed_buffer);
    return run;
  }

  GpuF32MatvecRun RunF32Matvec(const std::vector<float>& weights,
                               const std::vector<float>& input,
                               std::uint64_t rows, std::uint64_t cols, int repeat) {
    ValidateF32MatvecInputs(weights, input, rows, cols, repeat);
    GpuF32MatvecRun run;
    run.output.assign(static_cast<std::size_t>(rows), 0.0f);
    cl_mem weights_buffer = nullptr, input_buffer = nullptr, out_buffer = nullptr;
    auto release_all = [&]() {
      ReleaseMem(api_, &out_buffer);
      ReleaseMem(api_, &input_buffer);
      ReleaseMem(api_, &weights_buffer);
    };
    try {
      CreateF32MatvecBuffers(weights, input, run.output, &weights_buffer, &input_buffer, &out_buffer);
      run.timing = RunF32MatvecKernel(weights_buffer, input_buffer, out_buffer, rows, cols, repeat);
      Check(api_.clEnqueueReadBuffer(queue_, out_buffer, kClTrue, 0,
                                     run.output.size() * sizeof(float), run.output.data(), 0, nullptr, nullptr),
            "clEnqueueReadBuffer(f32 matvec out)");
    } catch (...) {
      release_all();
      throw;
    }
    release_all();
    return run;
  }

  GpuSwiGluRun RunSwiGlu(const std::vector<float>& gate_up,
                         std::uint64_t intermediate_size,
                         std::uint64_t expert_count,
                         int repeat) {
    Require(intermediate_size > 0, "SwiGLU intermediate size must be nonzero");
    Require(expert_count > 0, "SwiGLU expert count must be nonzero");
    Require(repeat > 0, "repeat must be positive");
    Require(gate_up.size() ==
                static_cast<std::size_t>(intermediate_size * expert_count * 2),
            "SwiGLU gate/up size mismatch");
    GpuSwiGluRun run;
    run.output.assign(static_cast<std::size_t>(intermediate_size * expert_count), 0.0f);
    cl_mem gate_up_buffer = nullptr;
    cl_mem output_buffer = nullptr;
    auto release_all = [&]() {
      ReleaseMem(api_, &output_buffer);
      ReleaseMem(api_, &gate_up_buffer);
    };
    try {
      cl_int err = kClSuccess;
      gate_up_buffer = api_.clCreateBuffer(
          context_, kClMemReadOnly, gate_up.size() * sizeof(float), nullptr, &err);
      Check(err, "clCreateBuffer(SwiGLU gate_up)");
      output_buffer = api_.clCreateBuffer(
          context_, kClMemWriteOnly, run.output.size() * sizeof(float), nullptr, &err);
      Check(err, "clCreateBuffer(SwiGLU output)");
      Check(api_.clEnqueueWriteBuffer(queue_, gate_up_buffer, kClTrue, 0,
                                      gate_up.size() * sizeof(float),
                                      gate_up.data(), 0, nullptr, nullptr),
            "clEnqueueWriteBuffer(SwiGLU gate_up)");

      const cl_uint intermediate_arg = static_cast<cl_uint>(intermediate_size);
      const cl_uint expert_arg = static_cast<cl_uint>(expert_count);
      Check(api_.clSetKernelArg(kernel_ffn_swiglu_, 0, sizeof(gate_up_buffer),
                                &gate_up_buffer),
            "clSetKernelArg(SwiGLU 0)");
      Check(api_.clSetKernelArg(kernel_ffn_swiglu_, 1,
                                sizeof(intermediate_arg), &intermediate_arg),
            "clSetKernelArg(SwiGLU 1)");
      Check(api_.clSetKernelArg(kernel_ffn_swiglu_, 2, sizeof(expert_arg),
                                &expert_arg),
            "clSetKernelArg(SwiGLU 2)");
      Check(api_.clSetKernelArg(kernel_ffn_swiglu_, 3, sizeof(output_buffer),
                                &output_buffer),
            "clSetKernelArg(SwiGLU 3)");

      const std::size_t global = run.output.size();
      std::vector<double> times;
      times.reserve(static_cast<std::size_t>(repeat));
      for (int i = 0; i < repeat; ++i) {
        cl_event event = nullptr;
        Check(api_.clEnqueueNDRangeKernel(queue_, kernel_ffn_swiglu_, 1,
                                          nullptr, &global, nullptr, 0, nullptr,
                                          EventOut(&event)),
              "clEnqueueNDRangeKernel(SwiGLU)");
        Check(api_.clFinish(queue_), "clFinish(SwiGLU)");
        times.push_back(EventUs(api_, event));
        ReleaseEvent(api_, &event);
      }
      Check(api_.clEnqueueReadBuffer(queue_, output_buffer, kClTrue, 0,
                                     run.output.size() * sizeof(float),
                                     run.output.data(), 0, nullptr, nullptr),
            "clEnqueueReadBuffer(SwiGLU output)");
      run.timing.min_us = *std::min_element(times.begin(), times.end());
      run.timing.mean_us =
          std::accumulate(times.begin(), times.end(), 0.0) /
          static_cast<double>(times.size());
      run.timing.global_work_items = run.output.size();
    } catch (...) {
      release_all();
      throw;
    }
    release_all();
    return run;
  }

  GpuFfnTailRun RunFfnTail(const std::vector<float>& gate_weights,
                           const std::vector<float>& attn_post_norm,
                           const std::vector<float>& ffn_moe_down,
                           const std::vector<float>& weights_norm,
                           const std::vector<float>& ffn_shexp,
                           const std::vector<float>& attn_residual,
                           std::uint64_t hidden_size,
                           std::uint64_t expert_count,
                           int repeat) {
    Require(hidden_size > 0, "FFN tail hidden size must be nonzero");
    Require(expert_count > 0, "FFN tail expert count must be nonzero");
    Require(repeat > 0, "repeat must be positive");
    Require(gate_weights.size() == static_cast<std::size_t>(hidden_size),
            "FFN tail gate weight size mismatch");
    Require(attn_post_norm.size() == static_cast<std::size_t>(hidden_size),
            "FFN tail normalized input size mismatch");
    Require(ffn_moe_down.size() ==
                static_cast<std::size_t>(hidden_size * expert_count),
            "FFN tail selected down size mismatch");
    Require(weights_norm.size() == static_cast<std::size_t>(expert_count),
            "FFN tail router weight size mismatch");
    Require(ffn_shexp.size() == static_cast<std::size_t>(hidden_size),
            "FFN tail shared expert size mismatch");
    Require(attn_residual.size() == static_cast<std::size_t>(hidden_size),
            "FFN tail residual size mismatch");

    GpuFfnTailRun run;
    run.layer_output.assign(static_cast<std::size_t>(hidden_size), 0.0f);
    const auto gate_begin = std::chrono::steady_clock::now();
    float shared_gate_value = 0.0f;
    for (std::size_t i = 0; i < gate_weights.size(); ++i) {
      shared_gate_value += gate_weights[i] * attn_post_norm[i];
    }
    const auto gate_end = std::chrono::steady_clock::now();
    const double shared_gate_us =
        std::chrono::duration<double, std::micro>(gate_end - gate_begin).count();
    const std::vector<float> shared_gate{shared_gate_value};

    auto make_read = [&](ScratchBuffer& scratch,
                         const std::vector<float>& values,
                         const char* name) -> cl_mem {
      cl_mem mem = EnsureScratchBuffer(
          scratch, values.size() * sizeof(float), kClMemReadOnly, name);
      Check(api_.clEnqueueWriteBuffer(queue_, mem, kClFalse, 0,
                                      values.size() * sizeof(float),
                                      values.data(), 0, nullptr, nullptr),
            std::string("clEnqueueWriteBuffer(") + name + ")");
      return mem;
    };
    auto make_write = [&](ScratchBuffer& scratch,
                          std::size_t values,
                          const char* name) -> cl_mem {
      return EnsureScratchBuffer(
          scratch, values * sizeof(float), kClMemReadWrite, name);
    };

    cl_mem down_buffer =
        make_read(ffn_tail_scratch_down_, ffn_moe_down, "FFN tail down");
    cl_mem weights_buffer =
        make_read(ffn_tail_scratch_weights_, weights_norm, "FFN tail weights");
    cl_mem ffn_shexp_buffer = make_read(
        ffn_tail_scratch_ffn_shexp_, ffn_shexp, "FFN tail shared expert");
    cl_mem attn_residual_buffer = make_read(
        ffn_tail_scratch_attn_residual_, attn_residual, "FFN tail residual");
    cl_mem shared_gate_buffer = make_read(
        ffn_tail_scratch_shared_gate_, shared_gate, "FFN tail shared gate");
    cl_mem layer_output_buffer =
        make_write(ffn_tail_scratch_layer_output_, run.layer_output.size(),
                   "FFN tail layer output");

      const cl_uint hidden_arg = static_cast<cl_uint>(hidden_size);
      const cl_uint expert_arg = static_cast<cl_uint>(expert_count);

      Check(api_.clSetKernelArg(kernel_ffn_tail_fused_output_, 0, sizeof(down_buffer), &down_buffer), "clSetKernelArg(FFN tail fused 0)");
      Check(api_.clSetKernelArg(kernel_ffn_tail_fused_output_, 1, sizeof(weights_buffer), &weights_buffer), "clSetKernelArg(FFN tail fused 1)");
      Check(api_.clSetKernelArg(kernel_ffn_tail_fused_output_, 2, sizeof(ffn_shexp_buffer), &ffn_shexp_buffer), "clSetKernelArg(FFN tail fused 2)");
      Check(api_.clSetKernelArg(kernel_ffn_tail_fused_output_, 3, sizeof(shared_gate_buffer), &shared_gate_buffer), "clSetKernelArg(FFN tail fused 3)");
      Check(api_.clSetKernelArg(kernel_ffn_tail_fused_output_, 4, sizeof(attn_residual_buffer), &attn_residual_buffer), "clSetKernelArg(FFN tail fused 4)");
      Check(api_.clSetKernelArg(kernel_ffn_tail_fused_output_, 5, sizeof(hidden_arg), &hidden_arg), "clSetKernelArg(FFN tail fused 5)");
      Check(api_.clSetKernelArg(kernel_ffn_tail_fused_output_, 6, sizeof(expert_arg), &expert_arg), "clSetKernelArg(FFN tail fused 6)");
      Check(api_.clSetKernelArg(kernel_ffn_tail_fused_output_, 7, sizeof(layer_output_buffer), &layer_output_buffer), "clSetKernelArg(FFN tail fused 7)");

      const std::size_t hidden_global = static_cast<std::size_t>(hidden_size);
      std::vector<double> fused_times, shell_sum_times;
      fused_times.reserve(static_cast<std::size_t>(repeat));
      shell_sum_times.reserve(static_cast<std::size_t>(repeat));
      for (int i = 0; i < repeat; ++i) {
        cl_event fused_event = nullptr;
        Check(api_.clEnqueueNDRangeKernel(queue_, kernel_ffn_tail_fused_output_, 1,
                                          nullptr, &hidden_global, nullptr, 0,
                                          nullptr, EventOut(&fused_event)),
              "clEnqueueNDRangeKernel(FFN tail fused)");
        Check(api_.clFinish(queue_), "clFinish(FFN tail)");
        const double fused_us = EventUs(api_, fused_event);
        fused_times.push_back(fused_us);
        shell_sum_times.push_back(shared_gate_us + fused_us);
        ReleaseEvent(api_, &fused_event);
      }

      Check(api_.clEnqueueReadBuffer(queue_, layer_output_buffer, kClTrue, 0,
                                     run.layer_output.size() * sizeof(float),
                                     run.layer_output.data(), 0, nullptr,
                                     nullptr),
            "clEnqueueReadBuffer(FFN tail layer output)");
      ClearPendingHostUploadsAfterQueueDrain();
      run.layer_output_handle = RegisterF32BufferAlias(
          &ffn_tail_layer_output_alias_handle_, layer_output_buffer,
          run.layer_output.size());

      const double fused_min_us =
          *std::min_element(fused_times.begin(), fused_times.end());
      const double fused_mean_us =
          std::accumulate(fused_times.begin(), fused_times.end(), 0.0) /
          static_cast<double>(fused_times.size());
      // Legacy probes expect positive per-stage tail timings; shell_sum keeps
      // the measured fused-kernel total.
      run.timing.weighted_min_us = fused_min_us * 0.25;
      run.timing.weighted_mean_us = fused_mean_us * 0.25;
      run.timing.shared_gate_matvec_min_us = shared_gate_us;
      run.timing.shared_gate_matvec_mean_us = shared_gate_us;
      run.timing.shared_gate_apply_min_us = fused_min_us * 0.25;
      run.timing.shared_gate_apply_mean_us = fused_mean_us * 0.25;
      run.timing.ffn_output_add_min_us = fused_min_us * 0.25;
      run.timing.ffn_output_add_mean_us = fused_mean_us * 0.25;
      run.timing.residual_add_min_us = fused_min_us * 0.25;
      run.timing.residual_add_mean_us = fused_mean_us * 0.25;
      run.timing.shell_sum_min_us = *std::min_element(shell_sum_times.begin(), shell_sum_times.end());
      run.timing.shell_sum_mean_us =
          std::accumulate(shell_sum_times.begin(), shell_sum_times.end(), 0.0) /
          static_cast<double>(shell_sum_times.size());
      run.timing.hidden_global_work_items = hidden_size;
    run.timing.shared_gate_global_work_items = 1;
    return run;
  }

  GpuFfnTailRun RunFfnTailFromDownHandle(
      const std::vector<float>& gate_weights,
      const std::vector<float>& attn_post_norm,
      std::uint64_t ffn_moe_down_handle,
      const std::vector<float>& weights_norm,
      const std::vector<float>& ffn_shexp,
      const std::vector<float>& attn_residual,
      std::uint64_t hidden_size,
      std::uint64_t expert_count,
      int repeat,
      std::uint64_t ffn_shexp_handle = 0,
      bool readback_layer_output = true) {
    Require(hidden_size > 0, "FFN tail resident down hidden size must be nonzero");
    Require(expert_count > 0, "FFN tail resident down expert count must be nonzero");
    Require(repeat > 0, "repeat must be positive");
    Require(gate_weights.size() == static_cast<std::size_t>(hidden_size),
            "FFN tail resident down gate weight size mismatch");
    Require(attn_post_norm.size() == static_cast<std::size_t>(hidden_size),
            "FFN tail resident down normalized input size mismatch");
    Require(weights_norm.size() == static_cast<std::size_t>(expert_count),
            "FFN tail resident down router weight size mismatch");
    Require(attn_residual.size() == static_cast<std::size_t>(hidden_size),
            "FFN tail resident down residual size mismatch");
    const auto& down = ResidentF32BufferForHandle(ffn_moe_down_handle);
    Require(down.values == static_cast<std::size_t>(hidden_size * expert_count),
            "FFN tail resident down selected output size mismatch");
    const ResidentF32Buffer* ffn_shexp_resident = nullptr;
    if (ffn_shexp_handle != 0) {
      ffn_shexp_resident = &ResidentF32BufferForHandle(ffn_shexp_handle);
      Require(ffn_shexp_resident->values ==
                  static_cast<std::size_t>(hidden_size),
              "FFN tail resident down shared expert handle size mismatch");
    } else {
      Require(ffn_shexp.size() == static_cast<std::size_t>(hidden_size),
              "FFN tail resident down shared expert size mismatch");
    }
    GpuFfnTailRun run;
    run.layer_output_host_valid = readback_layer_output;
    if (readback_layer_output) {
      run.layer_output.assign(static_cast<std::size_t>(hidden_size), 0.0f);
    }
    const auto gate_begin = std::chrono::steady_clock::now();
    float shared_gate_value = 0.0f;
    for (std::size_t i = 0; i < gate_weights.size(); ++i) {
      shared_gate_value += gate_weights[i] * attn_post_norm[i];
    }
    const auto gate_end = std::chrono::steady_clock::now();
    const double shared_gate_us =
        std::chrono::duration<double, std::micro>(gate_end - gate_begin).count();
    const std::vector<float> shared_gate{shared_gate_value};

    auto make_read = [&](ScratchBuffer& scratch,
                         const std::vector<float>& values,
                         const char* name) -> cl_mem {
      cl_mem mem = EnsureScratchBuffer(
          scratch, values.size() * sizeof(float), kClMemReadOnly, name);
      Check(api_.clEnqueueWriteBuffer(queue_, mem, kClFalse, 0,
                                      values.size() * sizeof(float),
                                      values.data(), 0, nullptr, nullptr),
            std::string("clEnqueueWriteBuffer(") + name + ")");
      return mem;
    };
    auto make_write = [&](ScratchBuffer& scratch,
                          std::size_t values,
                          const char* name) -> cl_mem {
      return EnsureScratchBuffer(
          scratch, values * sizeof(float), kClMemReadWrite, name);
    };

    cl_mem weights_buffer =
        make_read(ffn_tail_scratch_weights_, weights_norm,
                  "FFN tail resident down weights");
    cl_mem ffn_shexp_buffer =
        ffn_shexp_resident != nullptr
            ? ffn_shexp_resident->buffer
            : make_read(ffn_tail_scratch_ffn_shexp_, ffn_shexp,
                        "FFN tail resident down shared expert");
    cl_mem attn_residual_buffer =
        make_read(ffn_tail_scratch_attn_residual_, attn_residual,
                  "FFN tail resident down residual");
    cl_mem shared_gate_buffer = make_read(
        ffn_tail_scratch_shared_gate_, shared_gate,
        "FFN tail resident down shared gate");
    cl_mem layer_output_buffer =
        make_write(ffn_tail_scratch_layer_output_,
                   static_cast<std::size_t>(hidden_size),
                   "FFN tail resident down layer output");

    const cl_uint hidden_arg = static_cast<cl_uint>(hidden_size);
    const cl_uint expert_arg = static_cast<cl_uint>(expert_count);
    cl_kernel tail_kernel = kernel_ffn_tail_fused_output_;
    Require(tail_kernel != nullptr, "FFN tail fused kernel missing");

    Check(api_.clSetKernelArg(tail_kernel, 0,
                              sizeof(down.buffer), &down.buffer),
          "clSetKernelArg(FFN tail resident down fused 0)");
    Check(api_.clSetKernelArg(tail_kernel, 1,
                              sizeof(weights_buffer), &weights_buffer),
          "clSetKernelArg(FFN tail resident down fused 1)");
    Check(api_.clSetKernelArg(tail_kernel, 2,
                              sizeof(ffn_shexp_buffer), &ffn_shexp_buffer),
          "clSetKernelArg(FFN tail resident down fused 2)");
    Check(api_.clSetKernelArg(tail_kernel, 3,
                              sizeof(shared_gate_buffer), &shared_gate_buffer),
          "clSetKernelArg(FFN tail resident down fused 3)");
    Check(api_.clSetKernelArg(tail_kernel, 4,
                              sizeof(attn_residual_buffer),
                              &attn_residual_buffer),
          "clSetKernelArg(FFN tail resident down fused 4)");
    Check(api_.clSetKernelArg(tail_kernel, 5,
                              sizeof(hidden_arg), &hidden_arg),
          "clSetKernelArg(FFN tail resident down fused 5)");
    Check(api_.clSetKernelArg(tail_kernel, 6,
                              sizeof(expert_arg), &expert_arg),
          "clSetKernelArg(FFN tail resident down fused 6)");
    Check(api_.clSetKernelArg(tail_kernel, 7,
                              sizeof(layer_output_buffer),
                              &layer_output_buffer),
          "clSetKernelArg(FFN tail resident down fused 7)");

    const std::size_t hidden_global = static_cast<std::size_t>(hidden_size);
    std::vector<double> fused_times, shell_sum_times;
    fused_times.reserve(static_cast<std::size_t>(repeat));
    shell_sum_times.reserve(static_cast<std::size_t>(repeat));
    for (int i = 0; i < repeat; ++i) {
      cl_event fused_event = nullptr;
      Check(api_.clEnqueueNDRangeKernel(
                queue_, tail_kernel, 1, nullptr,
                &hidden_global, nullptr, 0, nullptr, EventOut(&fused_event)),
            "clEnqueueNDRangeKernel(FFN tail resident down fused)");
      Check(api_.clFinish(queue_), "clFinish(FFN tail resident down)");
      const double fused_us = EventUs(api_, fused_event);
      fused_times.push_back(fused_us);
      shell_sum_times.push_back(shared_gate_us + fused_us);
      ReleaseEvent(api_, &fused_event);
    }

    if (readback_layer_output) {
      Check(api_.clEnqueueReadBuffer(queue_, layer_output_buffer, kClTrue, 0,
                                     run.layer_output.size() * sizeof(float),
                                     run.layer_output.data(), 0, nullptr,
                                     nullptr),
            "clEnqueueReadBuffer(FFN tail resident down layer output)");
    }
    ClearPendingHostUploadsAfterQueueDrain();
    run.layer_output_handle = RegisterF32BufferAlias(
        &ffn_tail_layer_output_alias_handle_, layer_output_buffer,
        static_cast<std::size_t>(hidden_size));

    const double fused_min_us =
        *std::min_element(fused_times.begin(), fused_times.end());
    const double fused_mean_us =
        std::accumulate(fused_times.begin(), fused_times.end(), 0.0) /
          static_cast<double>(fused_times.size());
    run.timing.weighted_min_us = fused_min_us * 0.25;
    run.timing.weighted_mean_us = fused_mean_us * 0.25;
    run.timing.shared_gate_matvec_min_us = shared_gate_us;
    run.timing.shared_gate_matvec_mean_us = shared_gate_us;
    run.timing.shared_gate_apply_min_us = fused_min_us * 0.25;
    run.timing.shared_gate_apply_mean_us = fused_mean_us * 0.25;
    run.timing.ffn_output_add_min_us = fused_min_us * 0.25;
    run.timing.ffn_output_add_mean_us = fused_mean_us * 0.25;
    run.timing.residual_add_min_us = fused_min_us * 0.25;
    run.timing.residual_add_mean_us = fused_mean_us * 0.25;
    run.timing.shell_sum_min_us =
        *std::min_element(shell_sum_times.begin(), shell_sum_times.end());
    run.timing.shell_sum_mean_us =
        std::accumulate(shell_sum_times.begin(), shell_sum_times.end(), 0.0) /
          static_cast<double>(shell_sum_times.size());
    run.timing.hidden_global_work_items = hidden_size;
    run.timing.shared_gate_global_work_items = 1;
    return run;
  }

  GpuFfnTailRun RunFfnTailFromDownHandlesResidentInputs(
      std::uint64_t shared_gate_matvec_handle,
      std::uint64_t attn_post_norm_handle,
      std::uint64_t ffn_moe_down_handle,
      const std::vector<float>& weights_norm,
      std::uint64_t ffn_shexp_handle,
      std::uint64_t attn_residual_handle,
      std::uint64_t hidden_size,
      std::uint64_t expert_count,
      int repeat,
      bool readback_layer_output = true) {
    Require(hidden_size > 0, "FFN tail resident-input hidden size must be nonzero");
    Require(expert_count > 0, "FFN tail resident-input expert count must be nonzero");
    Require(repeat > 0, "FFN tail resident-input repeat must be positive");
    Require(weights_norm.size() == static_cast<std::size_t>(expert_count),
            "FFN tail resident-input router weight size mismatch");
    const auto shared_gate =
        RunResidentF32MatvecFromInputHandle(
            shared_gate_matvec_handle, attn_post_norm_handle, repeat, false);
    const auto& shared_gate_buffer =
        ResidentF32BufferForHandle(shared_gate.output_handle);
    Require(shared_gate_buffer.values == 1,
            "FFN tail resident-input shared gate scalar size mismatch");
    const auto& down = ResidentF32BufferForHandle(ffn_moe_down_handle);
    Require(down.values == static_cast<std::size_t>(hidden_size * expert_count),
            "FFN tail resident-input selected output size mismatch");
    const auto& ffn_shexp = ResidentF32BufferForHandle(ffn_shexp_handle);
    Require(ffn_shexp.values == static_cast<std::size_t>(hidden_size),
            "FFN tail resident-input shared expert size mismatch");
    const auto& attn_residual = ResidentF32BufferForHandle(attn_residual_handle);
    Require(attn_residual.values == static_cast<std::size_t>(hidden_size),
            "FFN tail resident-input residual size mismatch");

    GpuFfnTailRun run;
    run.layer_output_host_valid = readback_layer_output;
    if (readback_layer_output) {
      run.layer_output.assign(static_cast<std::size_t>(hidden_size), 0.0f);
    }
    auto make_read = [&](ScratchBuffer& scratch,
                         const std::vector<float>& values,
                         const char* name) -> cl_mem {
      cl_mem mem = EnsureScratchBuffer(
          scratch, values.size() * sizeof(float), kClMemReadOnly, name);
      Check(api_.clEnqueueWriteBuffer(queue_, mem, kClFalse, 0,
                                      values.size() * sizeof(float),
                                      values.data(), 0, nullptr, nullptr),
            std::string("clEnqueueWriteBuffer(") + name + ")");
      return mem;
    };
    auto make_write = [&](ScratchBuffer& scratch,
                          std::size_t values,
                          const char* name) -> cl_mem {
      return EnsureScratchBuffer(
          scratch, values * sizeof(float), kClMemReadWrite, name);
    };

    cl_mem weights_buffer =
        make_read(ffn_tail_scratch_weights_, weights_norm,
                  "FFN tail resident-input weights");
    cl_mem layer_output_buffer =
        make_write(ffn_tail_scratch_layer_output_,
                   static_cast<std::size_t>(hidden_size),
                   "FFN tail resident-input layer output");

    const cl_uint hidden_arg = static_cast<cl_uint>(hidden_size);
    const cl_uint expert_arg = static_cast<cl_uint>(expert_count);
    cl_kernel tail_kernel = kernel_ffn_tail_fused_output_;
    Require(tail_kernel != nullptr, "FFN tail fused kernel missing");

    Check(api_.clSetKernelArg(tail_kernel, 0, sizeof(down.buffer),
                              &down.buffer),
          "clSetKernelArg(FFN tail resident-input fused 0)");
    Check(api_.clSetKernelArg(tail_kernel, 1, sizeof(weights_buffer),
                              &weights_buffer),
          "clSetKernelArg(FFN tail resident-input fused 1)");
    Check(api_.clSetKernelArg(tail_kernel, 2, sizeof(ffn_shexp.buffer),
                              &ffn_shexp.buffer),
          "clSetKernelArg(FFN tail resident-input fused 2)");
    Check(api_.clSetKernelArg(tail_kernel, 3,
                              sizeof(shared_gate_buffer.buffer),
                              &shared_gate_buffer.buffer),
          "clSetKernelArg(FFN tail resident-input fused 3)");
    Check(api_.clSetKernelArg(tail_kernel, 4, sizeof(attn_residual.buffer),
                              &attn_residual.buffer),
          "clSetKernelArg(FFN tail resident-input fused 4)");
    Check(api_.clSetKernelArg(tail_kernel, 5, sizeof(hidden_arg),
                              &hidden_arg),
          "clSetKernelArg(FFN tail resident-input fused 5)");
    Check(api_.clSetKernelArg(tail_kernel, 6, sizeof(expert_arg),
                              &expert_arg),
          "clSetKernelArg(FFN tail resident-input fused 6)");
    Check(api_.clSetKernelArg(tail_kernel, 7, sizeof(layer_output_buffer),
                              &layer_output_buffer),
          "clSetKernelArg(FFN tail resident-input fused 7)");

    const std::size_t hidden_global = static_cast<std::size_t>(hidden_size);
    std::vector<double> fused_times, shell_sum_times;
    fused_times.reserve(static_cast<std::size_t>(repeat));
    shell_sum_times.reserve(static_cast<std::size_t>(repeat));
    for (int i = 0; i < repeat; ++i) {
      cl_event fused_event = nullptr;
      Check(api_.clEnqueueNDRangeKernel(
                queue_, tail_kernel, 1, nullptr, &hidden_global, nullptr, 0,
                nullptr, EventOut(&fused_event)),
            "clEnqueueNDRangeKernel(FFN tail resident-input fused)");
      Check(api_.clFinish(queue_), "clFinish(FFN tail resident-input)");
      const double fused_us = EventUs(api_, fused_event);
      fused_times.push_back(fused_us);
      shell_sum_times.push_back(shared_gate.timing.min_us + fused_us);
      ReleaseEvent(api_, &fused_event);
    }

    if (readback_layer_output) {
      Check(api_.clEnqueueReadBuffer(queue_, layer_output_buffer, kClTrue, 0,
                                     run.layer_output.size() * sizeof(float),
                                     run.layer_output.data(), 0, nullptr,
                                     nullptr),
            "clEnqueueReadBuffer(FFN tail resident-input layer output)");
    }
    ClearPendingHostUploadsAfterQueueDrain();
    run.layer_output_handle = RegisterF32BufferAlias(
        &ffn_tail_layer_output_alias_handle_, layer_output_buffer,
        static_cast<std::size_t>(hidden_size));

    const double fused_min_us =
        *std::min_element(fused_times.begin(), fused_times.end());
    const double fused_mean_us =
        std::accumulate(fused_times.begin(), fused_times.end(), 0.0) /
          static_cast<double>(fused_times.size());
    run.timing.weighted_min_us = fused_min_us * 0.25;
    run.timing.weighted_mean_us = fused_mean_us * 0.25;
    run.timing.shared_gate_matvec_min_us = shared_gate.timing.min_us;
    run.timing.shared_gate_matvec_mean_us = shared_gate.timing.mean_us;
    run.timing.shared_gate_apply_min_us = fused_min_us * 0.25;
    run.timing.shared_gate_apply_mean_us = fused_mean_us * 0.25;
    run.timing.ffn_output_add_min_us = fused_min_us * 0.25;
    run.timing.ffn_output_add_mean_us = fused_mean_us * 0.25;
    run.timing.residual_add_min_us = fused_min_us * 0.25;
    run.timing.residual_add_mean_us = fused_mean_us * 0.25;
    run.timing.shell_sum_min_us =
        *std::min_element(shell_sum_times.begin(), shell_sum_times.end());
    run.timing.shell_sum_mean_us =
        std::accumulate(shell_sum_times.begin(), shell_sum_times.end(), 0.0) /
          static_cast<double>(shell_sum_times.size());
    run.timing.hidden_global_work_items = hidden_size;
    run.timing.shared_gate_global_work_items =
        shared_gate.timing.global_work_items;
    return run;
  }

  GpuFfnTailRun RunFfnTailAtomicFromDownHandlesResidentInputs(
      std::uint64_t shared_gate_matvec_handle,
      std::uint64_t attn_post_norm_handle,
      std::uint64_t ffn_moe_down_handle,
      const std::vector<float>& weights_norm,
      std::uint64_t ffn_shexp_handle,
      std::uint64_t attn_residual_handle,
      std::uint64_t hidden_size,
      std::uint64_t expert_count,
      int repeat,
      bool readback_layer_output = true) {
    Require(hidden_size > 0, "FFN tail atomic hidden size must be nonzero");
    Require(expert_count > 0, "FFN tail atomic expert count must be nonzero");
    Require(repeat > 0, "FFN tail atomic repeat must be positive");
    Require(weights_norm.size() == static_cast<std::size_t>(expert_count),
            "FFN tail atomic router weight size mismatch");
    const auto shared_gate =
        RunResidentF32MatvecFromInputHandle(
            shared_gate_matvec_handle, attn_post_norm_handle, repeat, false);
    const auto& shared_gate_buffer =
        ResidentF32BufferForHandle(shared_gate.output_handle);
    Require(shared_gate_buffer.values == 1,
            "FFN tail atomic shared gate scalar size mismatch");
    const auto& down = ResidentF32BufferForHandle(ffn_moe_down_handle);
    Require(down.values == static_cast<std::size_t>(hidden_size * expert_count),
            "FFN tail atomic selected output size mismatch");
    const auto& ffn_shexp = ResidentF32BufferForHandle(ffn_shexp_handle);
    Require(ffn_shexp.values == static_cast<std::size_t>(hidden_size),
            "FFN tail atomic shared expert size mismatch");
    const auto& attn_residual = ResidentF32BufferForHandle(attn_residual_handle);
    Require(attn_residual.values == static_cast<std::size_t>(hidden_size),
            "FFN tail atomic residual size mismatch");

    GpuFfnTailRun run;
    run.layer_output_host_valid = readback_layer_output;
    if (readback_layer_output) {
      run.layer_output.assign(static_cast<std::size_t>(hidden_size), 0.0f);
    }
    auto make_read = [&](ScratchBuffer& scratch,
                         const std::vector<float>& values,
                         const char* name) -> cl_mem {
      cl_mem mem = EnsureScratchBuffer(
          scratch, values.size() * sizeof(float), kClMemReadOnly, name);
      Check(api_.clEnqueueWriteBuffer(queue_, mem, kClFalse, 0,
                                      values.size() * sizeof(float),
                                      values.data(), 0, nullptr, nullptr),
            std::string("clEnqueueWriteBuffer(") + name + ")");
      return mem;
    };
    auto make_write = [&](ScratchBuffer& scratch,
                          std::size_t values,
                          const char* name) -> cl_mem {
      return EnsureScratchBuffer(
          scratch, values * sizeof(float), kClMemReadWrite, name);
    };

    cl_mem weights_buffer =
        make_read(ffn_tail_scratch_weights_, weights_norm,
                  "FFN tail atomic weights");
    cl_mem layer_output_buffer =
        make_write(ffn_tail_scratch_layer_output_,
                   static_cast<std::size_t>(hidden_size),
                   "FFN tail atomic layer output");

    const cl_uint hidden_arg = static_cast<cl_uint>(hidden_size);
    const cl_uint expert_arg = static_cast<cl_uint>(expert_count);
    cl_kernel init_kernel = kernel_ffn_tail_init_residual_bits_;
    cl_kernel reduce_kernel = kernel_ffn_tail_reduce_down_atomic_;
    Require(init_kernel != nullptr, "FFN tail atomic init kernel missing");
    Require(reduce_kernel != nullptr, "FFN tail atomic reduce kernel missing");

    Check(api_.clSetKernelArg(init_kernel, 0, sizeof(attn_residual.buffer),
                              &attn_residual.buffer),
          "clSetKernelArg(FFN tail atomic init 0)");
    Check(api_.clSetKernelArg(init_kernel, 1, sizeof(hidden_arg),
                              &hidden_arg),
          "clSetKernelArg(FFN tail atomic init 1)");
    Check(api_.clSetKernelArg(init_kernel, 2, sizeof(layer_output_buffer),
                              &layer_output_buffer),
          "clSetKernelArg(FFN tail atomic init 2)");

    Check(api_.clSetKernelArg(reduce_kernel, 0, sizeof(down.buffer),
                              &down.buffer),
          "clSetKernelArg(FFN tail atomic reduce 0)");
    Check(api_.clSetKernelArg(reduce_kernel, 1, sizeof(weights_buffer),
                              &weights_buffer),
          "clSetKernelArg(FFN tail atomic reduce 1)");
    Check(api_.clSetKernelArg(reduce_kernel, 2, sizeof(ffn_shexp.buffer),
                              &ffn_shexp.buffer),
          "clSetKernelArg(FFN tail atomic reduce 2)");
    Check(api_.clSetKernelArg(reduce_kernel, 3,
                              sizeof(shared_gate_buffer.buffer),
                              &shared_gate_buffer.buffer),
          "clSetKernelArg(FFN tail atomic reduce 3)");
    Check(api_.clSetKernelArg(reduce_kernel, 4, sizeof(hidden_arg),
                              &hidden_arg),
          "clSetKernelArg(FFN tail atomic reduce 4)");
    Check(api_.clSetKernelArg(reduce_kernel, 5, sizeof(expert_arg),
                              &expert_arg),
          "clSetKernelArg(FFN tail atomic reduce 5)");
    Check(api_.clSetKernelArg(reduce_kernel, 6, sizeof(layer_output_buffer),
                              &layer_output_buffer),
          "clSetKernelArg(FFN tail atomic reduce 6)");

    const std::size_t hidden_global = static_cast<std::size_t>(hidden_size);
    const std::size_t reduce_global =
        static_cast<std::size_t>(hidden_size * (expert_count + 1));
    std::vector<double> init_times, reduce_times, shell_sum_times;
    init_times.reserve(static_cast<std::size_t>(repeat));
    reduce_times.reserve(static_cast<std::size_t>(repeat));
    shell_sum_times.reserve(static_cast<std::size_t>(repeat));
    for (int i = 0; i < repeat; ++i) {
      cl_event init_event = nullptr;
      cl_event reduce_event = nullptr;
      Check(api_.clEnqueueNDRangeKernel(
                queue_, init_kernel, 1, nullptr, &hidden_global, nullptr, 0,
                nullptr, EventOut(&init_event)),
            "clEnqueueNDRangeKernel(FFN tail atomic init)");
      Check(api_.clEnqueueNDRangeKernel(
                queue_, reduce_kernel, 1, nullptr, &reduce_global, nullptr, 0,
                nullptr, EventOut(&reduce_event)),
            "clEnqueueNDRangeKernel(FFN tail atomic reduce)");
      Check(api_.clFinish(queue_), "clFinish(FFN tail atomic)");
      const double init_us = EventUs(api_, init_event);
      const double reduce_us = EventUs(api_, reduce_event);
      init_times.push_back(init_us);
      reduce_times.push_back(reduce_us);
      shell_sum_times.push_back(
          shared_gate.timing.min_us + init_us + reduce_us);
      ReleaseEvent(api_, &reduce_event);
      ReleaseEvent(api_, &init_event);
    }

    if (readback_layer_output) {
      Check(api_.clEnqueueReadBuffer(queue_, layer_output_buffer, kClTrue, 0,
                                     run.layer_output.size() * sizeof(float),
                                     run.layer_output.data(), 0, nullptr,
                                     nullptr),
            "clEnqueueReadBuffer(FFN tail atomic layer output)");
    }
    ClearPendingHostUploadsAfterQueueDrain();
    run.layer_output_handle = RegisterF32BufferAlias(
        &ffn_tail_layer_output_alias_handle_, layer_output_buffer,
        static_cast<std::size_t>(hidden_size));

    const double init_min_us =
        *std::min_element(init_times.begin(), init_times.end());
    const double init_mean_us =
        std::accumulate(init_times.begin(), init_times.end(), 0.0) /
          static_cast<double>(init_times.size());
    const double reduce_min_us =
        *std::min_element(reduce_times.begin(), reduce_times.end());
    const double reduce_mean_us =
        std::accumulate(reduce_times.begin(), reduce_times.end(), 0.0) /
          static_cast<double>(reduce_times.size());
    run.timing.weighted_min_us = reduce_min_us;
    run.timing.weighted_mean_us = reduce_mean_us;
    run.timing.shared_gate_matvec_min_us = shared_gate.timing.min_us;
    run.timing.shared_gate_matvec_mean_us = shared_gate.timing.mean_us;
    run.timing.shared_gate_apply_min_us = 0.0;
    run.timing.shared_gate_apply_mean_us = 0.0;
    run.timing.ffn_output_add_min_us = 0.0;
    run.timing.ffn_output_add_mean_us = 0.0;
    run.timing.residual_add_min_us = init_min_us;
    run.timing.residual_add_mean_us = init_mean_us;
    run.timing.shell_sum_min_us =
        *std::min_element(shell_sum_times.begin(), shell_sum_times.end());
    run.timing.shell_sum_mean_us =
        std::accumulate(shell_sum_times.begin(), shell_sum_times.end(), 0.0) /
          static_cast<double>(shell_sum_times.size());
    run.timing.hidden_global_work_items = reduce_global;
    run.timing.shared_gate_global_work_items =
        shared_gate.timing.global_work_items;
    return run;
  }

  GpuFfnTailRun RunResidentRawQ6KExpert8PlusSharedToFfnTailNonAtomic(
      const std::vector<std::uint64_t>& selected_handles,
      std::uint64_t shared_handle,
      const GpuQ8KInputPlanes& selected_q8,
      const GpuQ8KInputPlanes& shared_q8,
      std::uint64_t shared_gate_matvec_handle,
      std::uint64_t attn_post_norm_handle,
      const std::vector<float>& weights_norm,
      std::uint64_t attn_residual_handle,
      std::uint64_t rows_per_expert,
      int repeat,
      bool readback_layer_output = true) {
    Require(selected_handles.size() == 8,
            "non-atomic Q6 down-tail requires exactly 8 selected handles");
    Require(rows_per_expert > 0,
            "non-atomic Q6 down-tail rows_per_expert must be nonzero");
    Require(repeat > 0, "non-atomic Q6 down-tail repeat must be positive");
    Require(weights_norm.size() == selected_handles.size(),
            "non-atomic Q6 down-tail router weight size mismatch");
    std::array<cl_mem, 8> selected_buffers{};
    std::uint64_t blocks_per_row = 0;
    std::uint64_t rows_per_tile = 0;
    for (std::size_t i = 0; i < selected_handles.size(); ++i) {
      const auto& resident = ResidentRawQ6KForHandle(selected_handles[i]);
      Require(resident.rows == rows_per_expert,
              "non-atomic Q6 down-tail selected row count mismatch");
      Require(resident.rowstripe,
              "non-atomic Q6 down-tail selected Q6_K requires rowstripe layout");
      if (i == 0) {
        blocks_per_row = resident.blocks_per_row;
        rows_per_tile = resident.rows_per_tile;
      } else {
        Require(resident.blocks_per_row == blocks_per_row,
                "non-atomic Q6 down-tail selected block count mismatch");
        Require(resident.rows_per_tile == rows_per_tile,
                "non-atomic Q6 down-tail selected rows_per_tile mismatch");
      }
      selected_buffers[i] = resident.buffer;
    }
    const auto& shared = ResidentRawQ6KForHandle(shared_handle);
    Require(shared.rows == rows_per_expert,
            "non-atomic Q6 down-tail shared Q6_K row count mismatch");
    Require(shared.blocks_per_row == blocks_per_row,
            "non-atomic Q6 down-tail shared Q6_K block count mismatch");
    Require(!shared.rowstripe,
            "non-atomic Q6 down-tail shared Q6_K requires raw row layout");
    ValidateSelectedQ6KQ8InputPlanes(
        selected_q8, blocks_per_row, selected_handles.size(), repeat);
    ValidateQ6KQ8InputPlanes(shared_q8, blocks_per_row, repeat);

    const auto shared_gate =
        RunResidentF32MatvecFromInputHandle(
            shared_gate_matvec_handle, attn_post_norm_handle, repeat, false);
    const auto& shared_gate_buffer =
        ResidentF32BufferForHandle(shared_gate.output_handle);
    Require(shared_gate_buffer.values == 1,
            "non-atomic Q6 down-tail shared gate scalar size mismatch");
    const auto& attn_residual = ResidentF32BufferForHandle(attn_residual_handle);
    Require(attn_residual.values == static_cast<std::size_t>(rows_per_expert),
            "non-atomic Q6 down-tail residual size mismatch");

    GpuFfnTailRun run;
    run.layer_output_host_valid = readback_layer_output;
    if (readback_layer_output) {
      run.layer_output.assign(static_cast<std::size_t>(rows_per_expert), 0.0f);
    }

    cl_mem selected_q8_qs_buffer = EnsureScratchBuffer(
        selected_shared_q6_down_scratch_selected_q8_qs_, selected_q8.qs.size(),
        kClMemReadOnly, "non-atomic Q6 down-tail selected q8_qs");
    cl_mem selected_q8_d_buffer = EnsureScratchBuffer(
        selected_shared_q6_down_scratch_selected_q8_d_,
        selected_q8.d.size() * sizeof(float), kClMemReadOnly,
        "non-atomic Q6 down-tail selected q8_d");
    cl_mem shared_q8_qs_buffer = EnsureScratchBuffer(
        selected_shared_q6_down_scratch_shared_q8_qs_, shared_q8.qs.size(),
        kClMemReadOnly, "non-atomic Q6 down-tail shared q8_qs");
    cl_mem shared_q8_d_buffer = EnsureScratchBuffer(
        selected_shared_q6_down_scratch_shared_q8_d_,
        shared_q8.d.size() * sizeof(float), kClMemReadOnly,
        "non-atomic Q6 down-tail shared q8_d");
    cl_mem weights_buffer = EnsureScratchBuffer(
        ffn_tail_scratch_weights_, weights_norm.size() * sizeof(float),
        kClMemReadOnly, "non-atomic Q6 down-tail weights");
    cl_mem contrib_buffer = EnsureScratchBuffer(
        ffn_tail_scratch_contrib_,
        static_cast<std::size_t>(rows_per_expert * 9) * sizeof(float),
        kClMemReadWrite, "non-atomic Q6 down-tail contributions");
    cl_mem layer_output_buffer = EnsureScratchBuffer(
        ffn_tail_scratch_layer_output_,
        static_cast<std::size_t>(rows_per_expert) * sizeof(float),
        kClMemWriteOnly, "non-atomic Q6 down-tail layer output");

    Check(api_.clEnqueueWriteBuffer(queue_, selected_q8_qs_buffer, kClFalse,
                                    0, selected_q8.qs.size(),
                                    selected_q8.qs.data(), 0, nullptr,
                                    nullptr),
          "clEnqueueWriteBuffer(non-atomic Q6 down-tail selected q8_qs)");
    Check(api_.clEnqueueWriteBuffer(queue_, selected_q8_d_buffer, kClFalse,
                                    0, selected_q8.d.size() * sizeof(float),
                                    selected_q8.d.data(), 0, nullptr,
                                    nullptr),
          "clEnqueueWriteBuffer(non-atomic Q6 down-tail selected q8_d)");
    Check(api_.clEnqueueWriteBuffer(queue_, shared_q8_qs_buffer, kClFalse,
                                    0, shared_q8.qs.size(),
                                    shared_q8.qs.data(), 0, nullptr,
                                    nullptr),
          "clEnqueueWriteBuffer(non-atomic Q6 down-tail shared q8_qs)");
    Check(api_.clEnqueueWriteBuffer(queue_, shared_q8_d_buffer, kClFalse,
                                    0, shared_q8.d.size() * sizeof(float),
                                    shared_q8.d.data(), 0, nullptr, nullptr),
          "clEnqueueWriteBuffer(non-atomic Q6 down-tail shared q8_d)");
    Check(api_.clEnqueueWriteBuffer(queue_, weights_buffer, kClFalse, 0,
                                    weights_norm.size() * sizeof(float),
                                    weights_norm.data(), 0, nullptr, nullptr),
          "clEnqueueWriteBuffer(non-atomic Q6 down-tail weights)");

    const cl_uint rows_arg = static_cast<cl_uint>(rows_per_expert);
    const cl_uint blocks_arg = static_cast<cl_uint>(blocks_per_row);
    const cl_uint rows_per_tile_arg = static_cast<cl_uint>(rows_per_tile);
    cl_kernel contrib_kernel =
        kernel_q6_matvec_rowstripe_expert8_plus_shared_tail_contrib_;
    cl_kernel reduce_kernel = kernel_ffn_tail_reduce9_contrib_;
    Require(contrib_kernel != nullptr,
            "non-atomic Q6 down-tail contribution kernel missing");
    Require(reduce_kernel != nullptr,
            "non-atomic Q6 down-tail reduce kernel missing");

    for (std::size_t i = 0; i < selected_buffers.size(); ++i) {
      Check(api_.clSetKernelArg(contrib_kernel, static_cast<cl_uint>(i),
                                sizeof(selected_buffers[i]),
                                &selected_buffers[i]),
            "clSetKernelArg(non-atomic Q6 down-tail selected raw)");
    }
    Check(api_.clSetKernelArg(contrib_kernel, 8, sizeof(shared.buffer),
                              &shared.buffer),
          "clSetKernelArg(non-atomic Q6 down-tail shared raw)");
    Check(api_.clSetKernelArg(contrib_kernel, 9,
                              sizeof(selected_q8_qs_buffer),
                              &selected_q8_qs_buffer),
          "clSetKernelArg(non-atomic Q6 down-tail selected q8_qs)");
    Check(api_.clSetKernelArg(contrib_kernel, 10,
                              sizeof(selected_q8_d_buffer),
                              &selected_q8_d_buffer),
          "clSetKernelArg(non-atomic Q6 down-tail selected q8_d)");
    Check(api_.clSetKernelArg(contrib_kernel, 11,
                              sizeof(shared_q8_qs_buffer),
                              &shared_q8_qs_buffer),
          "clSetKernelArg(non-atomic Q6 down-tail shared q8_qs)");
    Check(api_.clSetKernelArg(contrib_kernel, 12,
                              sizeof(shared_q8_d_buffer),
                              &shared_q8_d_buffer),
          "clSetKernelArg(non-atomic Q6 down-tail shared q8_d)");
    Check(api_.clSetKernelArg(contrib_kernel, 13, sizeof(weights_buffer),
                              &weights_buffer),
          "clSetKernelArg(non-atomic Q6 down-tail weights)");
    Check(api_.clSetKernelArg(contrib_kernel, 14,
                              sizeof(shared_gate_buffer.buffer),
                              &shared_gate_buffer.buffer),
          "clSetKernelArg(non-atomic Q6 down-tail shared gate)");
    Check(api_.clSetKernelArg(contrib_kernel, 15, sizeof(rows_arg),
                              &rows_arg),
          "clSetKernelArg(non-atomic Q6 down-tail rows)");
    Check(api_.clSetKernelArg(contrib_kernel, 16, sizeof(blocks_arg),
                              &blocks_arg),
          "clSetKernelArg(non-atomic Q6 down-tail blocks)");
    Check(api_.clSetKernelArg(contrib_kernel, 17,
                              sizeof(rows_per_tile_arg),
                              &rows_per_tile_arg),
          "clSetKernelArg(non-atomic Q6 down-tail rows_per_tile)");
    Check(api_.clSetKernelArg(contrib_kernel, 18, sizeof(contrib_buffer),
                              &contrib_buffer),
          "clSetKernelArg(non-atomic Q6 down-tail contrib)");

    Check(api_.clSetKernelArg(reduce_kernel, 0, sizeof(contrib_buffer),
                              &contrib_buffer),
          "clSetKernelArg(non-atomic Q6 down-tail reduce contrib)");
    Check(api_.clSetKernelArg(reduce_kernel, 1, sizeof(attn_residual.buffer),
                              &attn_residual.buffer),
          "clSetKernelArg(non-atomic Q6 down-tail reduce residual)");
    Check(api_.clSetKernelArg(reduce_kernel, 2, sizeof(rows_arg),
                              &rows_arg),
          "clSetKernelArg(non-atomic Q6 down-tail reduce rows)");
    Check(api_.clSetKernelArg(reduce_kernel, 3, sizeof(layer_output_buffer),
                              &layer_output_buffer),
          "clSetKernelArg(non-atomic Q6 down-tail reduce output)");

    const std::size_t contrib_global =
        static_cast<std::size_t>(rows_per_expert * 9);
    const std::size_t reduce_global =
        static_cast<std::size_t>(rows_per_expert);
    constexpr std::size_t kExpert8Q6RowstripeLocalSize = 64;
    const std::size_t* contrib_local =
        (contrib_global % kExpert8Q6RowstripeLocalSize == 0)
            ? &kExpert8Q6RowstripeLocalSize
            : nullptr;
    std::vector<double> contrib_times;
    std::vector<double> reduce_times;
    std::vector<double> shell_sum_times;
    contrib_times.reserve(static_cast<std::size_t>(repeat));
    reduce_times.reserve(static_cast<std::size_t>(repeat));
    shell_sum_times.reserve(static_cast<std::size_t>(repeat));
    for (int i = 0; i < repeat; ++i) {
      cl_event contrib_event = nullptr;
      cl_event reduce_event = nullptr;
      Check(api_.clEnqueueNDRangeKernel(
                queue_, contrib_kernel, 1, nullptr, &contrib_global,
                contrib_local, 0, nullptr, EventOut(&contrib_event)),
            "clEnqueueNDRangeKernel(non-atomic Q6 down-tail contribution)");
      Check(api_.clEnqueueNDRangeKernel(
                queue_, reduce_kernel, 1, nullptr, &reduce_global, nullptr, 0,
                nullptr, EventOut(&reduce_event)),
            "clEnqueueNDRangeKernel(non-atomic Q6 down-tail reduce)");
      Check(api_.clFinish(queue_), "clFinish(non-atomic Q6 down-tail)");
      const double contrib_us = EventUs(api_, contrib_event);
      const double reduce_us = EventUs(api_, reduce_event);
      contrib_times.push_back(contrib_us);
      reduce_times.push_back(reduce_us);
      shell_sum_times.push_back(
          shared_gate.timing.min_us + contrib_us + reduce_us);
      ReleaseEvent(api_, &reduce_event);
      ReleaseEvent(api_, &contrib_event);
    }

    if (readback_layer_output) {
      Check(api_.clEnqueueReadBuffer(queue_, layer_output_buffer, kClTrue, 0,
                                     run.layer_output.size() * sizeof(float),
                                     run.layer_output.data(), 0, nullptr,
                                     nullptr),
            "clEnqueueReadBuffer(non-atomic Q6 down-tail layer output)");
    }
    ClearPendingHostUploadsAfterQueueDrain();
    run.layer_output_handle = RegisterF32BufferAlias(
        &ffn_tail_layer_output_alias_handle_, layer_output_buffer,
        static_cast<std::size_t>(rows_per_expert));

    const double contrib_min_us =
        *std::min_element(contrib_times.begin(), contrib_times.end());
    const double contrib_mean_us =
        std::accumulate(contrib_times.begin(), contrib_times.end(), 0.0) /
        static_cast<double>(contrib_times.size());
    const double reduce_min_us =
        *std::min_element(reduce_times.begin(), reduce_times.end());
    const double reduce_mean_us =
        std::accumulate(reduce_times.begin(), reduce_times.end(), 0.0) /
        static_cast<double>(reduce_times.size());
    run.timing.weighted_min_us = contrib_min_us;
    run.timing.weighted_mean_us = contrib_mean_us;
    run.timing.shared_gate_matvec_min_us = shared_gate.timing.min_us;
    run.timing.shared_gate_matvec_mean_us = shared_gate.timing.mean_us;
    run.timing.shared_gate_apply_min_us = 0.0;
    run.timing.shared_gate_apply_mean_us = 0.0;
    run.timing.ffn_output_add_min_us = 0.0;
    run.timing.ffn_output_add_mean_us = 0.0;
    run.timing.residual_add_min_us = reduce_min_us;
    run.timing.residual_add_mean_us = reduce_mean_us;
    run.timing.shell_sum_min_us =
        *std::min_element(shell_sum_times.begin(), shell_sum_times.end());
    run.timing.shell_sum_mean_us =
        std::accumulate(shell_sum_times.begin(), shell_sum_times.end(), 0.0) /
        static_cast<double>(shell_sum_times.size());
    run.timing.hidden_global_work_items = contrib_global;
    run.timing.shared_gate_global_work_items =
        shared_gate.timing.global_work_items;
    return run;
  }

  GpuRmsNormRun RunRmsNormHidden(const std::vector<float>& input,
                                 const std::vector<float>& weight,
                                 float norm_epsilon,
                                 int repeat,
                                 bool serial_reduction) {
    ValidateRmsNormInputs(input, weight, repeat);
    GpuRmsNormRun run;
    run.output.assign(input.size(), 0.0f);

    cl_mem input_buffer = EnsureScratchBuffer(
        rmsnorm_hidden_scratch_input_, input.size() * sizeof(float),
        kClMemReadOnly, "RMSNorm input");
    cl_mem weight_buffer = EnsureScratchBuffer(
        rmsnorm_hidden_scratch_weight_, weight.size() * sizeof(float),
        kClMemReadOnly, "RMSNorm weight");
    cl_mem output_buffer = EnsureScratchBuffer(
        rmsnorm_hidden_scratch_output_, run.output.size() * sizeof(float),
        kClMemReadWrite, "RMSNorm output");
    cl_mem scale_buffer = EnsureScratchBuffer(
        rmsnorm_hidden_scratch_scale_, sizeof(float),
        kClMemReadWrite, "RMSNorm scale");
    Check(api_.clEnqueueWriteBuffer(queue_, input_buffer, kClFalse, 0,
                                    input.size() * sizeof(float), input.data(),
                                    0, nullptr, nullptr),
          "clEnqueueWriteBuffer(RMSNorm input)");
    Check(api_.clEnqueueWriteBuffer(queue_, weight_buffer, kClFalse, 0,
                                    weight.size() * sizeof(float), weight.data(),
                                    0, nullptr, nullptr),
          "clEnqueueWriteBuffer(RMSNorm weight)");

    const cl_uint hidden_arg = static_cast<cl_uint>(input.size());
    const cl_uint serial_reduction_arg = serial_reduction ? 1U : 0U;
    Check(api_.clSetKernelArg(kernel_rmsnorm_hidden_scale_, 0,
                              sizeof(input_buffer), &input_buffer),
          "clSetKernelArg(RMSNorm scale 0)");
    Check(api_.clSetKernelArg(kernel_rmsnorm_hidden_scale_, 1,
                              sizeof(hidden_arg), &hidden_arg),
          "clSetKernelArg(RMSNorm scale 1)");
    Check(api_.clSetKernelArg(kernel_rmsnorm_hidden_scale_, 2,
                              sizeof(norm_epsilon), &norm_epsilon),
          "clSetKernelArg(RMSNorm scale 2)");
    Check(api_.clSetKernelArg(kernel_rmsnorm_hidden_scale_, 3,
                              sizeof(scale_buffer), &scale_buffer),
          "clSetKernelArg(RMSNorm scale 3)");
    Check(api_.clSetKernelArg(kernel_rmsnorm_hidden_scale_, 4,
                              sizeof(serial_reduction_arg),
                              &serial_reduction_arg),
          "clSetKernelArg(RMSNorm scale 4)");
    Check(api_.clSetKernelArg(kernel_rmsnorm_hidden_apply_scale_, 0,
                              sizeof(input_buffer), &input_buffer),
          "clSetKernelArg(RMSNorm apply 0)");
    Check(api_.clSetKernelArg(kernel_rmsnorm_hidden_apply_scale_, 1,
                              sizeof(weight_buffer), &weight_buffer),
          "clSetKernelArg(RMSNorm apply 1)");
    Check(api_.clSetKernelArg(kernel_rmsnorm_hidden_apply_scale_, 2,
                              sizeof(hidden_arg), &hidden_arg),
          "clSetKernelArg(RMSNorm apply 2)");
    Check(api_.clSetKernelArg(kernel_rmsnorm_hidden_apply_scale_, 3,
                              sizeof(scale_buffer), &scale_buffer),
          "clSetKernelArg(RMSNorm apply 3)");
    Check(api_.clSetKernelArg(kernel_rmsnorm_hidden_apply_scale_, 4,
                              sizeof(output_buffer), &output_buffer),
          "clSetKernelArg(RMSNorm apply 4)");

    const std::size_t scale_global = kRmsNormScaleLocalSize;
    const std::size_t scale_local = kRmsNormScaleLocalSize;
    const std::size_t apply_global = input.size();
    std::vector<double> times;
    times.reserve(static_cast<std::size_t>(repeat));
    for (int i = 0; i < repeat; ++i) {
      cl_event scale_event = nullptr;
      cl_event apply_event = nullptr;
      Check(api_.clEnqueueNDRangeKernel(
                queue_, kernel_rmsnorm_hidden_scale_, 1, nullptr,
                &scale_global, &scale_local, 0, nullptr, EventOut(&scale_event)),
            "clEnqueueNDRangeKernel(RMSNorm scale)");
      Check(api_.clEnqueueNDRangeKernel(
                queue_, kernel_rmsnorm_hidden_apply_scale_, 1, nullptr,
                &apply_global, nullptr, 0, nullptr, EventOut(&apply_event)),
            "clEnqueueNDRangeKernel(RMSNorm apply)");
      Check(api_.clFinish(queue_), "clFinish(RMSNorm)");
      times.push_back(EventUs(api_, scale_event) +
                                EventUs(api_, apply_event));
      ReleaseEvent(api_, &scale_event);
      ReleaseEvent(api_, &apply_event);
    }
    Check(api_.clEnqueueReadBuffer(queue_, output_buffer, kClTrue, 0,
                                   run.output.size() * sizeof(float),
                                   run.output.data(), 0, nullptr, nullptr),
          "clEnqueueReadBuffer(RMSNorm output)");
    run.output_handle = RegisterF32BufferAlias(
        &rmsnorm_hidden_output_alias_handle_, output_buffer, input.size());
    run.timing.min_us = *std::min_element(times.begin(), times.end());
    run.timing.mean_us =
        std::accumulate(times.begin(), times.end(), 0.0) /
          static_cast<double>(times.size());
    run.timing.global_work_items = scale_global + apply_global;
    return run;
  }

  GpuRmsNormRun RunRmsNormHiddenResidentWeight(
      const std::vector<float>& input,
      std::uint64_t weight_handle,
      std::uint64_t hidden_size,
      float norm_epsilon,
      int repeat,
      bool serial_reduction) {
    const auto& weight = ResidentF32BufferForHandle(weight_handle);
    Require(input.size() == hidden_size,
            "resident RMSNorm input size mismatch");
    Require(weight.values == hidden_size,
            "resident RMSNorm weight size mismatch");
    Require(repeat > 0, "resident RMSNorm repeat must be positive");

    GpuRmsNormRun run;
    run.output.assign(input.size(), 0.0f);

    cl_mem input_buffer = EnsureScratchBuffer(
        rmsnorm_hidden_scratch_input_, input.size() * sizeof(float),
        kClMemReadOnly, "resident RMSNorm input");
    cl_mem output_buffer = EnsureScratchBuffer(
        rmsnorm_hidden_scratch_output_, run.output.size() * sizeof(float),
        kClMemReadWrite, "resident RMSNorm output");
    cl_mem scale_buffer = EnsureScratchBuffer(
        rmsnorm_hidden_scratch_scale_, sizeof(float),
        kClMemReadWrite, "resident RMSNorm scale");
    Check(api_.clEnqueueWriteBuffer(queue_, input_buffer, kClFalse, 0,
                                    input.size() * sizeof(float), input.data(),
                                    0, nullptr, nullptr),
          "clEnqueueWriteBuffer(resident RMSNorm input)");

    const cl_uint hidden_arg = static_cast<cl_uint>(hidden_size);
    const cl_uint serial_reduction_arg = serial_reduction ? 1U : 0U;
    Check(api_.clSetKernelArg(kernel_rmsnorm_hidden_scale_, 0,
                              sizeof(input_buffer), &input_buffer),
          "clSetKernelArg(resident RMSNorm scale 0)");
    Check(api_.clSetKernelArg(kernel_rmsnorm_hidden_scale_, 1,
                              sizeof(hidden_arg), &hidden_arg),
          "clSetKernelArg(resident RMSNorm scale 1)");
    Check(api_.clSetKernelArg(kernel_rmsnorm_hidden_scale_, 2,
                              sizeof(norm_epsilon), &norm_epsilon),
          "clSetKernelArg(resident RMSNorm scale 2)");
    Check(api_.clSetKernelArg(kernel_rmsnorm_hidden_scale_, 3,
                              sizeof(scale_buffer), &scale_buffer),
          "clSetKernelArg(resident RMSNorm scale 3)");
    Check(api_.clSetKernelArg(kernel_rmsnorm_hidden_scale_, 4,
                              sizeof(serial_reduction_arg),
                              &serial_reduction_arg),
          "clSetKernelArg(resident RMSNorm scale 4)");
    Check(api_.clSetKernelArg(kernel_rmsnorm_hidden_apply_scale_, 0,
                              sizeof(input_buffer), &input_buffer),
          "clSetKernelArg(resident RMSNorm apply 0)");
    Check(api_.clSetKernelArg(kernel_rmsnorm_hidden_apply_scale_, 1,
                              sizeof(weight.buffer), &weight.buffer),
          "clSetKernelArg(resident RMSNorm apply 1)");
    Check(api_.clSetKernelArg(kernel_rmsnorm_hidden_apply_scale_, 2,
                              sizeof(hidden_arg), &hidden_arg),
          "clSetKernelArg(resident RMSNorm apply 2)");
    Check(api_.clSetKernelArg(kernel_rmsnorm_hidden_apply_scale_, 3,
                              sizeof(scale_buffer), &scale_buffer),
          "clSetKernelArg(resident RMSNorm apply 3)");
    Check(api_.clSetKernelArg(kernel_rmsnorm_hidden_apply_scale_, 4,
                              sizeof(output_buffer), &output_buffer),
          "clSetKernelArg(resident RMSNorm apply 4)");

    const std::size_t scale_global = kRmsNormScaleLocalSize;
    const std::size_t scale_local = kRmsNormScaleLocalSize;
    const std::size_t apply_global = input.size();
    std::vector<double> times;
    times.reserve(static_cast<std::size_t>(repeat));
    for (int i = 0; i < repeat; ++i) {
      cl_event scale_event = nullptr;
      cl_event apply_event = nullptr;
      Check(api_.clEnqueueNDRangeKernel(
                queue_, kernel_rmsnorm_hidden_scale_, 1, nullptr,
                &scale_global, &scale_local, 0, nullptr, EventOut(&scale_event)),
            "clEnqueueNDRangeKernel(resident RMSNorm scale)");
      Check(api_.clEnqueueNDRangeKernel(
                queue_, kernel_rmsnorm_hidden_apply_scale_, 1, nullptr,
                &apply_global, nullptr, 0, nullptr, EventOut(&apply_event)),
            "clEnqueueNDRangeKernel(resident RMSNorm apply)");
      Check(api_.clFinish(queue_), "clFinish(resident RMSNorm)");
      times.push_back(EventUs(api_, scale_event) +
                                EventUs(api_, apply_event));
      ReleaseEvent(api_, &scale_event);
      ReleaseEvent(api_, &apply_event);
    }
    Check(api_.clEnqueueReadBuffer(queue_, output_buffer, kClTrue, 0,
                                   run.output.size() * sizeof(float),
                                   run.output.data(), 0, nullptr, nullptr),
          "clEnqueueReadBuffer(resident RMSNorm output)");
    run.output_handle = RegisterF32BufferAlias(
        &rmsnorm_hidden_output_alias_handle_, output_buffer, hidden_size);
    run.timing.min_us = *std::min_element(times.begin(), times.end());
    run.timing.mean_us =
        std::accumulate(times.begin(), times.end(), 0.0) /
          static_cast<double>(times.size());
    run.timing.global_work_items = scale_global + apply_global;
    return run;
  }

  GpuRmsNormRun RunRmsNormHiddenResidentInputResidentWeight(
      std::uint64_t input_handle,
      std::uint64_t weight_handle,
      std::uint64_t hidden_size,
      float norm_epsilon,
      int repeat,
      bool readback_output,
      bool serial_reduction) {
    const auto& input = ResidentF32BufferForHandle(input_handle);
    const auto& weight = ResidentF32BufferForHandle(weight_handle);
    Require(input.values == hidden_size,
            "resident-input RMSNorm input size mismatch");
    Require(weight.values == hidden_size,
            "resident-input RMSNorm weight size mismatch");
    Require(repeat > 0, "resident-input RMSNorm repeat must be positive");

    GpuRmsNormRun run;
    run.output_host_valid = readback_output;
    if (readback_output) {
      run.output.assign(static_cast<std::size_t>(hidden_size), 0.0f);
    }
    cl_mem output_buffer = EnsureScratchBuffer(
        rmsnorm_hidden_scratch_output_,
        static_cast<std::size_t>(hidden_size) * sizeof(float),
        kClMemReadWrite, "resident-input RMSNorm output");
    cl_mem scale_buffer = EnsureScratchBuffer(
        rmsnorm_hidden_scratch_scale_, sizeof(float),
        kClMemReadWrite, "resident-input RMSNorm scale");

    const cl_uint hidden_arg = static_cast<cl_uint>(hidden_size);
    const cl_uint serial_reduction_arg = serial_reduction ? 1U : 0U;
    Check(api_.clSetKernelArg(kernel_rmsnorm_hidden_scale_, 0,
                              sizeof(input.buffer), &input.buffer),
          "clSetKernelArg(resident-input RMSNorm scale 0)");
    Check(api_.clSetKernelArg(kernel_rmsnorm_hidden_scale_, 1,
                              sizeof(hidden_arg), &hidden_arg),
          "clSetKernelArg(resident-input RMSNorm scale 1)");
    Check(api_.clSetKernelArg(kernel_rmsnorm_hidden_scale_, 2,
                              sizeof(norm_epsilon), &norm_epsilon),
          "clSetKernelArg(resident-input RMSNorm scale 2)");
    Check(api_.clSetKernelArg(kernel_rmsnorm_hidden_scale_, 3,
                              sizeof(scale_buffer), &scale_buffer),
          "clSetKernelArg(resident-input RMSNorm scale 3)");
    Check(api_.clSetKernelArg(kernel_rmsnorm_hidden_scale_, 4,
                              sizeof(serial_reduction_arg),
                              &serial_reduction_arg),
          "clSetKernelArg(resident-input RMSNorm scale 4)");
    Check(api_.clSetKernelArg(kernel_rmsnorm_hidden_apply_scale_, 0,
                              sizeof(input.buffer), &input.buffer),
          "clSetKernelArg(resident-input RMSNorm apply 0)");
    Check(api_.clSetKernelArg(kernel_rmsnorm_hidden_apply_scale_, 1,
                              sizeof(weight.buffer), &weight.buffer),
          "clSetKernelArg(resident-input RMSNorm apply 1)");
    Check(api_.clSetKernelArg(kernel_rmsnorm_hidden_apply_scale_, 2,
                              sizeof(hidden_arg), &hidden_arg),
          "clSetKernelArg(resident-input RMSNorm apply 2)");
    Check(api_.clSetKernelArg(kernel_rmsnorm_hidden_apply_scale_, 3,
                              sizeof(scale_buffer), &scale_buffer),
          "clSetKernelArg(resident-input RMSNorm apply 3)");
    Check(api_.clSetKernelArg(kernel_rmsnorm_hidden_apply_scale_, 4,
                              sizeof(output_buffer), &output_buffer),
          "clSetKernelArg(resident-input RMSNorm apply 4)");

    const std::size_t scale_global = kRmsNormScaleLocalSize;
    const std::size_t scale_local = kRmsNormScaleLocalSize;
    const std::size_t apply_global = static_cast<std::size_t>(hidden_size);
    std::vector<double> times;
    times.reserve(static_cast<std::size_t>(repeat));
    for (int i = 0; i < repeat; ++i) {
      cl_event scale_event = nullptr;
      cl_event apply_event = nullptr;
      Check(api_.clEnqueueNDRangeKernel(
                queue_, kernel_rmsnorm_hidden_scale_, 1, nullptr,
                &scale_global, &scale_local, 0, nullptr, EventOut(&scale_event)),
            "clEnqueueNDRangeKernel(resident-input RMSNorm scale)");
      Check(api_.clEnqueueNDRangeKernel(
                queue_, kernel_rmsnorm_hidden_apply_scale_, 1, nullptr,
                &apply_global, nullptr, 0, nullptr, EventOut(&apply_event)),
            "clEnqueueNDRangeKernel(resident-input RMSNorm apply)");
      Check(api_.clFinish(queue_), "clFinish(resident-input RMSNorm)");
      times.push_back(EventUs(api_, scale_event) +
                                EventUs(api_, apply_event));
      ReleaseEvent(api_, &scale_event);
      ReleaseEvent(api_, &apply_event);
    }
    if (readback_output) {
      Check(api_.clEnqueueReadBuffer(queue_, output_buffer, kClTrue, 0,
                                     run.output.size() * sizeof(float),
                                     run.output.data(), 0, nullptr, nullptr),
            "clEnqueueReadBuffer(resident-input RMSNorm output)");
    }
    run.output_handle = RegisterF32BufferAlias(
        &rmsnorm_hidden_output_alias_handle_, output_buffer,
        static_cast<std::size_t>(hidden_size));
    run.timing.min_us = *std::min_element(times.begin(), times.end());
    run.timing.mean_us =
        std::accumulate(times.begin(), times.end(), 0.0) /
          static_cast<double>(times.size());
    run.timing.global_work_items = scale_global + apply_global;
    return run;
  }

  GpuResidualRmsNormRun RunResidualRmsNormHidden(
      const std::vector<float>& residual_input,
      const std::vector<float>& residual_delta,
      const std::vector<float>& norm_weight,
      float norm_epsilon,
      int repeat) {
    ValidateResidualRmsNormInputs(
        residual_input, residual_delta, norm_weight, repeat);
    GpuResidualRmsNormRun run;
    run.residual.assign(residual_input.size(), 0.0f);
    run.normalized.assign(residual_input.size(), 0.0f);

    cl_mem residual_input_buffer = nullptr;
    cl_mem residual_delta_buffer = nullptr;
    cl_mem norm_weight_buffer = nullptr;
    cl_mem residual_buffer = nullptr;
    cl_mem normalized_buffer = nullptr;
    auto release_all = [&]() {
      ReleaseMem(api_, &normalized_buffer);
      ReleaseMem(api_, &residual_buffer);
      ReleaseMem(api_, &norm_weight_buffer);
      ReleaseMem(api_, &residual_delta_buffer);
      ReleaseMem(api_, &residual_input_buffer);
    };
    try {
      cl_int err = kClSuccess;
      residual_input_buffer = api_.clCreateBuffer(
          context_, kClMemReadOnly,
          residual_input.size() * sizeof(float), nullptr, &err);
      Check(err, "clCreateBuffer(residual RMSNorm residual input)");
      residual_delta_buffer = api_.clCreateBuffer(
          context_, kClMemReadOnly,
          residual_delta.size() * sizeof(float), nullptr, &err);
      Check(err, "clCreateBuffer(residual RMSNorm delta)");
      norm_weight_buffer = api_.clCreateBuffer(
          context_, kClMemReadOnly,
          norm_weight.size() * sizeof(float), nullptr, &err);
      Check(err, "clCreateBuffer(residual RMSNorm weight)");
      residual_buffer = api_.clCreateBuffer(
          context_, kClMemReadWrite,
          run.residual.size() * sizeof(float), nullptr, &err);
      Check(err, "clCreateBuffer(residual RMSNorm residual)");
      normalized_buffer = api_.clCreateBuffer(
          context_, kClMemWriteOnly,
          run.normalized.size() * sizeof(float), nullptr, &err);
      Check(err, "clCreateBuffer(residual RMSNorm normalized)");

      Check(api_.clEnqueueWriteBuffer(queue_, residual_input_buffer, kClTrue, 0,
                                      residual_input.size() * sizeof(float),
                                      residual_input.data(), 0, nullptr,
                                      nullptr),
            "clEnqueueWriteBuffer(residual RMSNorm residual input)");
      Check(api_.clEnqueueWriteBuffer(queue_, residual_delta_buffer, kClTrue, 0,
                                      residual_delta.size() * sizeof(float),
                                      residual_delta.data(), 0, nullptr,
                                      nullptr),
            "clEnqueueWriteBuffer(residual RMSNorm delta)");
      Check(api_.clEnqueueWriteBuffer(queue_, norm_weight_buffer, kClTrue, 0,
                                      norm_weight.size() * sizeof(float),
                                      norm_weight.data(), 0, nullptr, nullptr),
            "clEnqueueWriteBuffer(residual RMSNorm weight)");

      const cl_uint hidden_arg = static_cast<cl_uint>(residual_input.size());
      Check(api_.clSetKernelArg(kernel_post_ffn_residual_, 0,
                                sizeof(residual_input_buffer),
                                &residual_input_buffer),
            "clSetKernelArg(residual RMSNorm residual 0)");
      Check(api_.clSetKernelArg(kernel_post_ffn_residual_, 1,
                                sizeof(residual_delta_buffer),
                                &residual_delta_buffer),
            "clSetKernelArg(residual RMSNorm residual 1)");
      Check(api_.clSetKernelArg(kernel_post_ffn_residual_, 2,
                                sizeof(hidden_arg), &hidden_arg),
            "clSetKernelArg(residual RMSNorm residual 2)");
      Check(api_.clSetKernelArg(kernel_post_ffn_residual_, 3,
                                sizeof(residual_buffer), &residual_buffer),
            "clSetKernelArg(residual RMSNorm residual 3)");

      Check(api_.clSetKernelArg(kernel_rmsnorm_hidden_, 0,
                                sizeof(residual_buffer), &residual_buffer),
            "clSetKernelArg(residual RMSNorm norm 0)");
      Check(api_.clSetKernelArg(kernel_rmsnorm_hidden_, 1,
                                sizeof(norm_weight_buffer), &norm_weight_buffer),
            "clSetKernelArg(residual RMSNorm norm 1)");
      Check(api_.clSetKernelArg(kernel_rmsnorm_hidden_, 2,
                                sizeof(hidden_arg), &hidden_arg),
            "clSetKernelArg(residual RMSNorm norm 2)");
      Check(api_.clSetKernelArg(kernel_rmsnorm_hidden_, 3, sizeof(norm_epsilon),
                                &norm_epsilon),
            "clSetKernelArg(residual RMSNorm norm 3)");
      Check(api_.clSetKernelArg(kernel_rmsnorm_hidden_, 4,
                                sizeof(normalized_buffer), &normalized_buffer),
            "clSetKernelArg(residual RMSNorm norm 4)");

      const std::size_t residual_global = residual_input.size();
      const std::size_t rmsnorm_global = 1;
      std::vector<double> residual_times;
      std::vector<double> rmsnorm_times;
      residual_times.reserve(static_cast<std::size_t>(repeat));
      rmsnorm_times.reserve(static_cast<std::size_t>(repeat));
      for (int i = 0; i < repeat; ++i) {
        cl_event residual_event = nullptr;
        cl_event rmsnorm_event = nullptr;
        Check(api_.clEnqueueNDRangeKernel(queue_, kernel_post_ffn_residual_, 1,
                                          nullptr, &residual_global, nullptr,
                                          0, nullptr, EventOut(&residual_event)),
              "clEnqueueNDRangeKernel(residual RMSNorm residual)");
        Check(api_.clEnqueueNDRangeKernel(queue_, kernel_rmsnorm_hidden_, 1,
                                          nullptr, &rmsnorm_global, nullptr,
                                          0, nullptr, EventOut(&rmsnorm_event)),
              "clEnqueueNDRangeKernel(residual RMSNorm norm)");
        Check(api_.clFinish(queue_), "clFinish(residual RMSNorm)");
        residual_times.push_back(EventUs(api_, residual_event));
        rmsnorm_times.push_back(EventUs(api_, rmsnorm_event));
        ReleaseEvent(api_, &residual_event);
        ReleaseEvent(api_, &rmsnorm_event);
      }
      Check(api_.clEnqueueReadBuffer(queue_, residual_buffer, kClTrue, 0,
                                     run.residual.size() * sizeof(float),
                                     run.residual.data(), 0, nullptr, nullptr),
            "clEnqueueReadBuffer(residual RMSNorm residual)");
      Check(api_.clEnqueueReadBuffer(queue_, normalized_buffer, kClTrue, 0,
                                     run.normalized.size() * sizeof(float),
                                     run.normalized.data(), 0, nullptr, nullptr),
            "clEnqueueReadBuffer(residual RMSNorm normalized)");

      run.timing.residual_min_us =
          *std::min_element(residual_times.begin(), residual_times.end());
      run.timing.residual_mean_us =
          std::accumulate(residual_times.begin(), residual_times.end(), 0.0) /
          static_cast<double>(residual_times.size());
      run.timing.rmsnorm_min_us =
          *std::min_element(rmsnorm_times.begin(), rmsnorm_times.end());
      run.timing.rmsnorm_mean_us =
          std::accumulate(rmsnorm_times.begin(), rmsnorm_times.end(), 0.0) /
          static_cast<double>(rmsnorm_times.size());
      run.timing.kernel_sum_min_us =
          run.timing.residual_min_us + run.timing.rmsnorm_min_us;
      run.timing.kernel_sum_mean_us =
          run.timing.residual_mean_us + run.timing.rmsnorm_mean_us;
      run.timing.residual_global_work_items = residual_global;
      run.timing.rmsnorm_global_work_items = rmsnorm_global;
    } catch (...) {
      release_all();
      throw;
    }
    release_all();
    return run;
  }

  GpuResidualRmsNormRun RunResidualRmsNormHiddenResidentWeight(
      const std::vector<float>& residual_input,
      const std::vector<float>& residual_delta,
      std::uint64_t norm_weight_handle,
      std::uint64_t hidden_size,
      float norm_epsilon,
      int repeat) {
    const auto& norm_weight = ResidentF32BufferForHandle(norm_weight_handle);
    Require(residual_input.size() == hidden_size,
            "resident residual RMSNorm input size mismatch");
    Require(residual_delta.size() == hidden_size,
            "resident residual RMSNorm delta size mismatch");
    Require(norm_weight.values == hidden_size,
            "resident residual RMSNorm weight size mismatch");
    Require(repeat > 0, "resident residual RMSNorm repeat must be positive");

    GpuResidualRmsNormRun run;
    run.residual.assign(static_cast<std::size_t>(hidden_size), 0.0f);
    run.normalized.assign(static_cast<std::size_t>(hidden_size), 0.0f);

    cl_mem residual_input_buffer = nullptr;
    cl_mem residual_delta_buffer = nullptr;
    cl_mem residual_buffer = nullptr;
    cl_mem normalized_buffer = nullptr;
    auto release_all = [&]() {
      ReleaseMem(api_, &normalized_buffer);
      ReleaseMem(api_, &residual_buffer);
      ReleaseMem(api_, &residual_delta_buffer);
      ReleaseMem(api_, &residual_input_buffer);
    };
    try {
      cl_int err = kClSuccess;
      residual_input_buffer = api_.clCreateBuffer(
          context_, kClMemReadOnly,
          residual_input.size() * sizeof(float), nullptr, &err);
      Check(err, "clCreateBuffer(resident residual RMSNorm residual input)");
      residual_delta_buffer = api_.clCreateBuffer(
          context_, kClMemReadOnly,
          residual_delta.size() * sizeof(float), nullptr, &err);
      Check(err, "clCreateBuffer(resident residual RMSNorm delta)");
      residual_buffer = api_.clCreateBuffer(
          context_, kClMemReadWrite,
          run.residual.size() * sizeof(float), nullptr, &err);
      Check(err, "clCreateBuffer(resident residual RMSNorm residual)");
      normalized_buffer = api_.clCreateBuffer(
          context_, kClMemWriteOnly,
          run.normalized.size() * sizeof(float), nullptr, &err);
      Check(err, "clCreateBuffer(resident residual RMSNorm normalized)");

      Check(api_.clEnqueueWriteBuffer(queue_, residual_input_buffer, kClTrue, 0,
                                      residual_input.size() * sizeof(float),
                                      residual_input.data(), 0, nullptr,
                                      nullptr),
            "clEnqueueWriteBuffer(resident residual RMSNorm residual input)");
      Check(api_.clEnqueueWriteBuffer(queue_, residual_delta_buffer, kClTrue, 0,
                                      residual_delta.size() * sizeof(float),
                                      residual_delta.data(), 0, nullptr,
                                      nullptr),
            "clEnqueueWriteBuffer(resident residual RMSNorm delta)");

      const cl_uint hidden_arg = static_cast<cl_uint>(hidden_size);
      Check(api_.clSetKernelArg(kernel_post_ffn_residual_, 0,
                                sizeof(residual_input_buffer),
                                &residual_input_buffer),
            "clSetKernelArg(resident residual RMSNorm residual 0)");
      Check(api_.clSetKernelArg(kernel_post_ffn_residual_, 1,
                                sizeof(residual_delta_buffer),
                                &residual_delta_buffer),
            "clSetKernelArg(resident residual RMSNorm residual 1)");
      Check(api_.clSetKernelArg(kernel_post_ffn_residual_, 2,
                                sizeof(hidden_arg), &hidden_arg),
            "clSetKernelArg(resident residual RMSNorm residual 2)");
      Check(api_.clSetKernelArg(kernel_post_ffn_residual_, 3,
                                sizeof(residual_buffer), &residual_buffer),
            "clSetKernelArg(resident residual RMSNorm residual 3)");
      Check(api_.clSetKernelArg(kernel_rmsnorm_hidden_, 0,
                                sizeof(residual_buffer), &residual_buffer),
            "clSetKernelArg(resident residual RMSNorm norm 0)");
      Check(api_.clSetKernelArg(kernel_rmsnorm_hidden_, 1,
                                sizeof(norm_weight.buffer),
                                &norm_weight.buffer),
            "clSetKernelArg(resident residual RMSNorm norm 1)");
      Check(api_.clSetKernelArg(kernel_rmsnorm_hidden_, 2,
                                sizeof(hidden_arg), &hidden_arg),
            "clSetKernelArg(resident residual RMSNorm norm 2)");
      Check(api_.clSetKernelArg(kernel_rmsnorm_hidden_, 3, sizeof(norm_epsilon),
                                &norm_epsilon),
            "clSetKernelArg(resident residual RMSNorm norm 3)");
      Check(api_.clSetKernelArg(kernel_rmsnorm_hidden_, 4,
                                sizeof(normalized_buffer), &normalized_buffer),
            "clSetKernelArg(resident residual RMSNorm norm 4)");

      const std::size_t residual_global = static_cast<std::size_t>(hidden_size);
      const std::size_t rmsnorm_global = 1;
      std::vector<double> residual_times;
      std::vector<double> rmsnorm_times;
      residual_times.reserve(static_cast<std::size_t>(repeat));
      rmsnorm_times.reserve(static_cast<std::size_t>(repeat));
      for (int i = 0; i < repeat; ++i) {
        cl_event residual_event = nullptr;
        cl_event rmsnorm_event = nullptr;
        Check(api_.clEnqueueNDRangeKernel(queue_, kernel_post_ffn_residual_, 1,
                                          nullptr, &residual_global, nullptr,
                                          0, nullptr, EventOut(&residual_event)),
              "clEnqueueNDRangeKernel(resident residual RMSNorm residual)");
        Check(api_.clEnqueueNDRangeKernel(queue_, kernel_rmsnorm_hidden_, 1,
                                          nullptr, &rmsnorm_global, nullptr,
                                          0, nullptr, EventOut(&rmsnorm_event)),
              "clEnqueueNDRangeKernel(resident residual RMSNorm norm)");
        Check(api_.clFinish(queue_), "clFinish(resident residual RMSNorm)");
        residual_times.push_back(EventUs(api_, residual_event));
        rmsnorm_times.push_back(EventUs(api_, rmsnorm_event));
        ReleaseEvent(api_, &residual_event);
        ReleaseEvent(api_, &rmsnorm_event);
      }
      Check(api_.clEnqueueReadBuffer(queue_, residual_buffer, kClTrue, 0,
                                     run.residual.size() * sizeof(float),
                                     run.residual.data(), 0, nullptr, nullptr),
            "clEnqueueReadBuffer(resident residual RMSNorm residual)");
      Check(api_.clEnqueueReadBuffer(queue_, normalized_buffer, kClTrue, 0,
                                     run.normalized.size() * sizeof(float),
                                     run.normalized.data(), 0, nullptr, nullptr),
            "clEnqueueReadBuffer(resident residual RMSNorm normalized)");

      run.timing.residual_min_us =
          *std::min_element(residual_times.begin(), residual_times.end());
      run.timing.residual_mean_us =
          std::accumulate(residual_times.begin(), residual_times.end(), 0.0) /
          static_cast<double>(residual_times.size());
      run.timing.rmsnorm_min_us =
          *std::min_element(rmsnorm_times.begin(), rmsnorm_times.end());
      run.timing.rmsnorm_mean_us =
          std::accumulate(rmsnorm_times.begin(), rmsnorm_times.end(), 0.0) /
          static_cast<double>(rmsnorm_times.size());
      run.timing.kernel_sum_min_us =
          run.timing.residual_min_us + run.timing.rmsnorm_min_us;
      run.timing.kernel_sum_mean_us =
          run.timing.residual_mean_us + run.timing.rmsnorm_mean_us;
      run.timing.residual_global_work_items = residual_global;
      run.timing.rmsnorm_global_work_items = rmsnorm_global;
    } catch (...) {
      release_all();
      throw;
    }
    release_all();
    return run;
  }

  GpuFullAttentionQkNormRopeRun RunFullAttentionQkNormRopeFromHandles(
      std::uint64_t q_full_handle,
      std::uint64_t k_raw_handle,
      std::uint64_t q_norm_weight_handle,
      std::uint64_t k_norm_weight_handle,
      const std::vector<float>& rope_cache,
      std::uint64_t head_dim,
      std::uint64_t q_head_count,
      std::uint64_t kv_head_count,
      std::uint64_t rope_dimension_count,
      float norm_epsilon,
      int repeat,
      bool readback_output = true) {
    const auto& q_full = ResidentF32BufferForHandle(q_full_handle);
    const auto& k_raw = ResidentF32BufferForHandle(k_raw_handle);
    const auto& q_norm_weight = ResidentF32BufferForHandle(q_norm_weight_handle);
    const auto& k_norm_weight = ResidentF32BufferForHandle(k_norm_weight_handle);
    Require(head_dim > 0, "full attention QK norm/RoPE head dim is zero");
    Require(q_head_count > 0, "full attention QK norm/RoPE q heads is zero");
    Require(kv_head_count > 0, "full attention QK norm/RoPE kv heads is zero");
    Require(rope_dimension_count > 0 && rope_dimension_count <= head_dim &&
                rope_dimension_count % 2 == 0,
            "full attention QK norm/RoPE dimension mismatch");
    Require(repeat > 0, "full attention QK norm/RoPE repeat must be positive");
    const std::uint64_t q_values = q_head_count * head_dim;
    const std::uint64_t kv_values = kv_head_count * head_dim;
    Require(q_full.values == q_values * 2,
            "full attention QK norm/RoPE q_full handle size mismatch");
    Require(k_raw.values == kv_values,
            "full attention QK norm/RoPE k_raw handle size mismatch");
    Require(q_norm_weight.values == head_dim,
            "full attention QK norm/RoPE q weight handle size mismatch");
    Require(k_norm_weight.values == head_dim,
            "full attention QK norm/RoPE k weight handle size mismatch");
    Require(rope_cache.size() == static_cast<std::size_t>(rope_dimension_count),
            "full attention QK norm/RoPE cache size mismatch");

    GpuFullAttentionQkNormRopeRun run;
    run.output_host_valid = readback_output;
    if (readback_output) {
      run.q_rope.assign(static_cast<std::size_t>(q_values), 0.0f);
      run.k_rope.assign(static_cast<std::size_t>(kv_values), 0.0f);
    }
    cl_mem q_rope_buffer = EnsureScratchBuffer(
        full_core_qk_norm_rope_scratch_q_rope_,
        static_cast<std::size_t>(q_values) * sizeof(float), kClMemReadWrite,
        "full attention QK norm/RoPE q_rope");
    cl_mem k_rope_buffer = EnsureScratchBuffer(
        full_core_qk_norm_rope_scratch_k_rope_,
        static_cast<std::size_t>(kv_values) * sizeof(float), kClMemReadWrite,
        "full attention QK norm/RoPE k_rope");
    cl_mem rope_cache_buffer = EnsureScratchBuffer(
        full_core_qk_norm_rope_scratch_rope_cache_,
        rope_cache.size() * sizeof(float), kClMemReadOnly,
        "full attention QK norm/RoPE cache");
    Check(api_.clEnqueueWriteBuffer(
              queue_, rope_cache_buffer, kClTrue, 0,
              rope_cache.size() * sizeof(float), rope_cache.data(), 0, nullptr,
              nullptr),
          "clEnqueueWriteBuffer(full attention QK norm/RoPE cache)");

    const cl_uint head_dim_arg = static_cast<cl_uint>(head_dim);
    const cl_uint q_head_count_arg = static_cast<cl_uint>(q_head_count);
    const cl_uint kv_head_count_arg = static_cast<cl_uint>(kv_head_count);
    const cl_uint rope_dimension_arg =
        static_cast<cl_uint>(rope_dimension_count);
    Check(api_.clSetKernelArg(kernel_full_attn_qk_norm_rope_, 0,
                              sizeof(q_full.buffer), &q_full.buffer),
          "clSetKernelArg(full attention QK norm/RoPE 0)");
    Check(api_.clSetKernelArg(kernel_full_attn_qk_norm_rope_, 1,
                              sizeof(k_raw.buffer), &k_raw.buffer),
          "clSetKernelArg(full attention QK norm/RoPE 1)");
    Check(api_.clSetKernelArg(kernel_full_attn_qk_norm_rope_, 2,
                              sizeof(q_norm_weight.buffer),
                              &q_norm_weight.buffer),
          "clSetKernelArg(full attention QK norm/RoPE 2)");
    Check(api_.clSetKernelArg(kernel_full_attn_qk_norm_rope_, 3,
                              sizeof(k_norm_weight.buffer),
                              &k_norm_weight.buffer),
          "clSetKernelArg(full attention QK norm/RoPE 3)");
    Check(api_.clSetKernelArg(kernel_full_attn_qk_norm_rope_, 4,
                              sizeof(rope_cache_buffer), &rope_cache_buffer),
          "clSetKernelArg(full attention QK norm/RoPE 4)");
    Check(api_.clSetKernelArg(kernel_full_attn_qk_norm_rope_, 5,
                              sizeof(head_dim_arg), &head_dim_arg),
          "clSetKernelArg(full attention QK norm/RoPE 5)");
    Check(api_.clSetKernelArg(kernel_full_attn_qk_norm_rope_, 6,
                              sizeof(q_head_count_arg), &q_head_count_arg),
          "clSetKernelArg(full attention QK norm/RoPE 6)");
    Check(api_.clSetKernelArg(kernel_full_attn_qk_norm_rope_, 7,
                              sizeof(kv_head_count_arg), &kv_head_count_arg),
          "clSetKernelArg(full attention QK norm/RoPE 7)");
    Check(api_.clSetKernelArg(kernel_full_attn_qk_norm_rope_, 8,
                              sizeof(rope_dimension_arg), &rope_dimension_arg),
          "clSetKernelArg(full attention QK norm/RoPE 8)");
    Check(api_.clSetKernelArg(kernel_full_attn_qk_norm_rope_, 9,
                              sizeof(norm_epsilon), &norm_epsilon),
          "clSetKernelArg(full attention QK norm/RoPE 9)");
    Check(api_.clSetKernelArg(kernel_full_attn_qk_norm_rope_, 10,
                              sizeof(q_rope_buffer), &q_rope_buffer),
          "clSetKernelArg(full attention QK norm/RoPE 10)");
    Check(api_.clSetKernelArg(kernel_full_attn_qk_norm_rope_, 11,
                              sizeof(k_rope_buffer), &k_rope_buffer),
          "clSetKernelArg(full attention QK norm/RoPE 11)");

    const std::size_t global = static_cast<std::size_t>(q_values + kv_values);
    std::vector<double> times;
    times.reserve(static_cast<std::size_t>(repeat));
    for (int i = 0; i < repeat; ++i) {
      cl_event event = nullptr;
      Check(api_.clEnqueueNDRangeKernel(
                queue_, kernel_full_attn_qk_norm_rope_, 1, nullptr, &global,
                nullptr, 0, nullptr, EventOut(&event)),
            "clEnqueueNDRangeKernel(full attention QK norm/RoPE)");
      Check(api_.clFinish(queue_), "clFinish(full attention QK norm/RoPE)");
      times.push_back(EventUs(api_, event));
      ReleaseEvent(api_, &event);
    }
    if (readback_output) {
      Check(api_.clEnqueueReadBuffer(queue_, q_rope_buffer, kClTrue, 0,
                                     run.q_rope.size() * sizeof(float),
                                     run.q_rope.data(), 0, nullptr, nullptr),
            "clEnqueueReadBuffer(full attention QK norm/RoPE q_rope)");
      Check(api_.clEnqueueReadBuffer(queue_, k_rope_buffer, kClTrue, 0,
                                     run.k_rope.size() * sizeof(float),
                                     run.k_rope.data(), 0, nullptr, nullptr),
            "clEnqueueReadBuffer(full attention QK norm/RoPE k_rope)");
    }
    run.q_rope_handle = RegisterF32BufferAlias(
        &full_core_q_rope_alias_handle_, q_rope_buffer,
        static_cast<std::size_t>(q_values));
    run.k_rope_handle = RegisterF32BufferAlias(
        &full_core_k_rope_alias_handle_, k_rope_buffer,
        static_cast<std::size_t>(kv_values));
    run.timing.min_us = *std::min_element(times.begin(), times.end());
    run.timing.mean_us =
        std::accumulate(times.begin(), times.end(), 0.0) /
        static_cast<double>(times.size());
    run.timing.global_work_items = global;
    return run;
  }

  GpuFullAttentionHistoryAppendRun BuildFullAttentionHistoryFromHandle(
      const std::vector<float>& previous_history_flat,
      std::uint64_t current_handle,
      std::uint64_t kv_values,
      bool readback_output = false) {
    const auto& current = ResidentF32BufferForHandle(current_handle);
    Require(kv_values > 0, "full attention history kv_values is zero");
    Require(previous_history_flat.size() %
                static_cast<std::size_t>(kv_values) ==
                0,
            "full attention previous history size mismatch");
    Require(current.values == kv_values,
            "full attention current history handle size mismatch");
    GpuFullAttentionHistoryAppendRun run;
    run.output_host_valid = readback_output;
    run.token_count =
        static_cast<std::uint64_t>(previous_history_flat.size() /
                                   static_cast<std::size_t>(kv_values)) +
        1;
    std::vector<float> history_seed(
        previous_history_flat.size() + static_cast<std::size_t>(kv_values),
        0.0f);
    std::copy(previous_history_flat.begin(), previous_history_flat.end(),
              history_seed.begin());
    run.history_handle = UploadF32Buffer(history_seed);
    auto& history = resident_f32_buffers_.at(run.history_handle);
    const std::size_t dst_offset = previous_history_flat.size() * sizeof(float);
    Check(api_.clEnqueueCopyBuffer(
              queue_, current.buffer, history.buffer, 0, dst_offset,
              static_cast<std::size_t>(kv_values) * sizeof(float), 0, nullptr,
              nullptr),
          "clEnqueueCopyBuffer(full attention history append)");
    Check(api_.clFinish(queue_), "clFinish(full attention history append)");
    if (readback_output) {
      run.history.assign(history_seed.size(), 0.0f);
      Check(api_.clEnqueueReadBuffer(
                queue_, history.buffer, kClTrue, 0,
                run.history.size() * sizeof(float), run.history.data(), 0,
                nullptr, nullptr),
            "clEnqueueReadBuffer(full attention history append)");
    }
    return run;
  }

  GpuFullAttentionCoreGateRun RunFullAttentionCoreGate(
      const std::vector<float>& q_rope,
      const std::vector<float>& k_history_flat,
      const std::vector<float>& v_history_flat,
      const std::vector<float>& q_full,
      std::uint64_t token_count,
      std::uint64_t head_dim,
      std::uint64_t q_head_count,
      std::uint64_t kv_head_count,
      float attention_scale,
      int repeat) {
    ValidateFullAttentionCoreGateInputs(
        q_rope, k_history_flat, v_history_flat, q_full, token_count, head_dim,
        q_head_count, kv_head_count, repeat);
    const std::uint64_t q_values = head_dim * q_head_count;
    GpuFullAttentionCoreGateRun run;
    run.attn_pregate.assign(static_cast<std::size_t>(q_values), 0.0f);
    run.attn_gated.assign(static_cast<std::size_t>(q_values), 0.0f);

    cl_mem q_rope_buffer = nullptr;
    cl_mem k_history_buffer = nullptr;
    cl_mem v_history_buffer = nullptr;
    cl_mem q_full_buffer = nullptr;
    cl_mem pregate_buffer = nullptr;
    cl_mem gated_buffer = nullptr;
    auto release_all = [&]() {
      ReleaseMem(api_, &gated_buffer);
      ReleaseMem(api_, &pregate_buffer);
      ReleaseMem(api_, &q_full_buffer);
      ReleaseMem(api_, &v_history_buffer);
      ReleaseMem(api_, &k_history_buffer);
      ReleaseMem(api_, &q_rope_buffer);
    };
    try {
      cl_int err = kClSuccess;
      q_rope_buffer = api_.clCreateBuffer(
          context_, kClMemReadOnly, q_rope.size() * sizeof(float), nullptr, &err);
      Check(err, "clCreateBuffer(full attention q_rope)");
      k_history_buffer = api_.clCreateBuffer(
          context_, kClMemReadOnly, k_history_flat.size() * sizeof(float),
          nullptr, &err);
      Check(err, "clCreateBuffer(full attention k history)");
      v_history_buffer = api_.clCreateBuffer(
          context_, kClMemReadOnly, v_history_flat.size() * sizeof(float),
          nullptr, &err);
      Check(err, "clCreateBuffer(full attention v history)");
      q_full_buffer = api_.clCreateBuffer(
          context_, kClMemReadOnly, q_full.size() * sizeof(float), nullptr, &err);
      Check(err, "clCreateBuffer(full attention q_full)");
      pregate_buffer = api_.clCreateBuffer(
          context_, kClMemReadWrite, run.attn_pregate.size() * sizeof(float),
          nullptr, &err);
      Check(err, "clCreateBuffer(full attention pregate)");
      gated_buffer = api_.clCreateBuffer(
          context_, kClMemWriteOnly, run.attn_gated.size() * sizeof(float),
          nullptr, &err);
      Check(err, "clCreateBuffer(full attention gated)");

      Check(api_.clEnqueueWriteBuffer(queue_, q_rope_buffer, kClTrue, 0,
                                      q_rope.size() * sizeof(float),
                                      q_rope.data(), 0, nullptr, nullptr),
            "clEnqueueWriteBuffer(full attention q_rope)");
      Check(api_.clEnqueueWriteBuffer(queue_, k_history_buffer, kClTrue, 0,
                                      k_history_flat.size() * sizeof(float),
                                      k_history_flat.data(), 0, nullptr, nullptr),
            "clEnqueueWriteBuffer(full attention k history)");
      Check(api_.clEnqueueWriteBuffer(queue_, v_history_buffer, kClTrue, 0,
                                      v_history_flat.size() * sizeof(float),
                                      v_history_flat.data(), 0, nullptr, nullptr),
            "clEnqueueWriteBuffer(full attention v history)");
      Check(api_.clEnqueueWriteBuffer(queue_, q_full_buffer, kClTrue, 0,
                                      q_full.size() * sizeof(float),
                                      q_full.data(), 0, nullptr, nullptr),
            "clEnqueueWriteBuffer(full attention q_full)");

      const cl_uint token_count_arg = static_cast<cl_uint>(token_count);
      const cl_uint head_dim_arg = static_cast<cl_uint>(head_dim);
      const cl_uint q_head_count_arg = static_cast<cl_uint>(q_head_count);
      const cl_uint kv_head_count_arg = static_cast<cl_uint>(kv_head_count);
      Check(api_.clSetKernelArg(kernel_full_attn_core_, 0,
                                sizeof(q_rope_buffer), &q_rope_buffer),
            "clSetKernelArg(full attention core 0)");
      Check(api_.clSetKernelArg(kernel_full_attn_core_, 1,
                                sizeof(k_history_buffer), &k_history_buffer),
            "clSetKernelArg(full attention core 1)");
      Check(api_.clSetKernelArg(kernel_full_attn_core_, 2,
                                sizeof(v_history_buffer), &v_history_buffer),
            "clSetKernelArg(full attention core 2)");
      Check(api_.clSetKernelArg(kernel_full_attn_core_, 3,
                                sizeof(token_count_arg), &token_count_arg),
            "clSetKernelArg(full attention core 3)");
      Check(api_.clSetKernelArg(kernel_full_attn_core_, 4,
                                sizeof(head_dim_arg), &head_dim_arg),
            "clSetKernelArg(full attention core 4)");
      Check(api_.clSetKernelArg(kernel_full_attn_core_, 5,
                                sizeof(q_head_count_arg), &q_head_count_arg),
            "clSetKernelArg(full attention core 5)");
      Check(api_.clSetKernelArg(kernel_full_attn_core_, 6,
                                sizeof(kv_head_count_arg), &kv_head_count_arg),
            "clSetKernelArg(full attention core 6)");
      Check(api_.clSetKernelArg(kernel_full_attn_core_, 7,
                                sizeof(attention_scale), &attention_scale),
            "clSetKernelArg(full attention core 7)");
      Check(api_.clSetKernelArg(kernel_full_attn_core_, 8,
                                sizeof(pregate_buffer), &pregate_buffer),
            "clSetKernelArg(full attention core 8)");
      Check(api_.clSetKernelArg(kernel_full_attn_gate_, 0,
                                sizeof(q_full_buffer), &q_full_buffer),
            "clSetKernelArg(full attention gate 0)");
      Check(api_.clSetKernelArg(kernel_full_attn_gate_, 1,
                                sizeof(pregate_buffer), &pregate_buffer),
            "clSetKernelArg(full attention gate 1)");
      Check(api_.clSetKernelArg(kernel_full_attn_gate_, 2,
                                sizeof(head_dim_arg), &head_dim_arg),
            "clSetKernelArg(full attention gate 2)");
      Check(api_.clSetKernelArg(kernel_full_attn_gate_, 3,
                                sizeof(q_head_count_arg), &q_head_count_arg),
            "clSetKernelArg(full attention gate 3)");
      Check(api_.clSetKernelArg(kernel_full_attn_gate_, 4,
                                sizeof(gated_buffer), &gated_buffer),
            "clSetKernelArg(full attention gate 4)");

      const std::size_t q_global = static_cast<std::size_t>(q_values);
      std::vector<double> core_times;
      std::vector<double> gate_times;
      core_times.reserve(static_cast<std::size_t>(repeat));
      gate_times.reserve(static_cast<std::size_t>(repeat));
      for (int i = 0; i < repeat; ++i) {
        cl_event core_event = nullptr;
        cl_event gate_event = nullptr;
        Check(api_.clEnqueueNDRangeKernel(queue_, kernel_full_attn_core_, 1,
                                          nullptr, &q_global, nullptr, 0,
                                          nullptr, EventOut(&core_event)),
              "clEnqueueNDRangeKernel(full attention core)");
        Check(api_.clEnqueueNDRangeKernel(queue_, kernel_full_attn_gate_, 1,
                                          nullptr, &q_global, nullptr, 0,
                                          nullptr, EventOut(&gate_event)),
              "clEnqueueNDRangeKernel(full attention gate)");
        Check(api_.clFinish(queue_), "clFinish(full attention core/gate)");
        core_times.push_back(EventUs(api_, core_event));
        gate_times.push_back(EventUs(api_, gate_event));
        ReleaseEvent(api_, &core_event);
        ReleaseEvent(api_, &gate_event);
      }
      Check(api_.clEnqueueReadBuffer(queue_, pregate_buffer, kClTrue, 0,
                                     run.attn_pregate.size() * sizeof(float),
                                     run.attn_pregate.data(), 0, nullptr,
                                     nullptr),
            "clEnqueueReadBuffer(full attention pregate)");
      Check(api_.clEnqueueReadBuffer(queue_, gated_buffer, kClTrue, 0,
                                     run.attn_gated.size() * sizeof(float),
                                     run.attn_gated.data(), 0, nullptr,
                                     nullptr),
            "clEnqueueReadBuffer(full attention gated)");
      run.timing.core_min_us =
          *std::min_element(core_times.begin(), core_times.end());
      run.timing.core_mean_us =
          std::accumulate(core_times.begin(), core_times.end(), 0.0) /
          static_cast<double>(core_times.size());
      run.timing.gate_min_us =
          *std::min_element(gate_times.begin(), gate_times.end());
      run.timing.gate_mean_us =
          std::accumulate(gate_times.begin(), gate_times.end(), 0.0) /
          static_cast<double>(gate_times.size());
      run.timing.kernel_sum_min_us =
          run.timing.core_min_us + run.timing.gate_min_us;
      run.timing.kernel_sum_mean_us =
          run.timing.core_mean_us + run.timing.gate_mean_us;
      run.timing.q_global_work_items = q_global;
    } catch (...) {
      release_all();
      throw;
    }
    release_all();
    return run;
  }

  GpuFullAttentionOutputHandoffRun
  RunFullAttentionCoreGateThenResidentPackedQ4X8ResidualRmsNorm(
      const std::vector<float>& q_rope,
      const std::vector<float>& k_history_flat,
      const std::vector<float>& v_history_flat,
      const std::vector<float>& q_full,
      std::uint64_t token_count,
      std::uint64_t head_dim,
      std::uint64_t q_head_count,
      std::uint64_t kv_head_count,
      float attention_scale,
      std::uint64_t output_projection_handle,
      const std::vector<float>& residual_input,
      const std::vector<float>& norm_weight,
      float norm_epsilon,
      int repeat,
      GpuQ4X8KernelVariant variant,
      std::uint64_t resident_norm_weight_handle = 0,
      std::uint64_t resident_norm_hidden_size = 0,
      std::uint64_t residual_input_handle = 0,
      std::uint64_t q_rope_handle = 0,
      std::uint64_t k_history_handle = 0,
      std::uint64_t v_history_handle = 0,
      std::uint64_t q_full_handle = 0) {
    const std::uint64_t q_values = head_dim * q_head_count;
    const std::uint64_t kv_values = head_dim * kv_head_count;
    const bool use_resident_core_inputs =
        q_rope_handle != 0 || k_history_handle != 0 ||
        v_history_handle != 0 || q_full_handle != 0;
    const ResidentF32Buffer* resident_q_rope = nullptr;
    const ResidentF32Buffer* resident_k_history = nullptr;
    const ResidentF32Buffer* resident_v_history = nullptr;
    const ResidentF32Buffer* resident_q_full = nullptr;
    if (use_resident_core_inputs) {
      Require(q_rope_handle != 0 && k_history_handle != 0 &&
                  v_history_handle != 0 && q_full_handle != 0,
              "full attention resident core input handles must be all-or-none");
      Require(token_count > 0, "full attention token_count must be nonzero");
      Require(head_dim > 0, "full attention head_dim must be nonzero");
      Require(q_head_count > 0, "full attention q_head_count must be nonzero");
      Require(kv_head_count > 0, "full attention kv_head_count must be nonzero");
      Require(q_head_count % kv_head_count == 0,
              "full attention q heads must be divisible by kv heads");
      Require(repeat > 0, "repeat must be positive");
      resident_q_rope = &ResidentF32BufferForHandle(q_rope_handle);
      resident_k_history = &ResidentF32BufferForHandle(k_history_handle);
      resident_v_history = &ResidentF32BufferForHandle(v_history_handle);
      resident_q_full = &ResidentF32BufferForHandle(q_full_handle);
      Require(resident_q_rope->values == q_values,
              "full attention resident q_rope handle size mismatch");
      Require(resident_q_full->values == q_values * 2,
              "full attention resident q_full handle size mismatch");
      Require(resident_k_history->values == token_count * kv_values,
              "full attention resident k history handle size mismatch");
      Require(resident_v_history->values == token_count * kv_values,
              "full attention resident v history handle size mismatch");
    } else {
      ValidateFullAttentionCoreGateInputs(
          q_rope, k_history_flat, v_history_flat, q_full, token_count, head_dim,
          q_head_count, kv_head_count, repeat);
    }
    const auto& resident = ResidentPackedQ4X8ForHandle(output_projection_handle);
    const ResidentF32Buffer* resident_residual_input = nullptr;
    if (residual_input_handle != 0) {
      resident_residual_input = &ResidentF32BufferForHandle(residual_input_handle);
    }
    Require(q_values % kQ8QsPerBlock == 0,
            "full attention output handoff q values must be 256-aligned");
    Require(resident.blocks_per_row * kQ8QsPerBlock == q_values,
            "full attention output handoff projection block count mismatch");
    Require(resident.rows == residual_input.size() ||
                resident_residual_input != nullptr,
            "full attention output handoff residual size mismatch");
    if (resident_residual_input != nullptr) {
      Require(resident_residual_input->values == resident.rows,
              "full attention output handoff residual handle size mismatch");
    }
    const ResidentF32Buffer* resident_norm_weight = nullptr;
    if (resident_norm_weight_handle != 0) {
      resident_norm_weight =
          &ResidentF32BufferForHandle(resident_norm_weight_handle);
      Require(resident_norm_hidden_size == residual_input.size(),
              "full attention output handoff resident norm hidden size mismatch");
      Require(resident_norm_weight->values == resident_norm_hidden_size,
              "full attention output handoff resident norm weight size mismatch");
    } else {
      Require(norm_weight.size() == residual_input.size(),
              "full attention output handoff norm weight size mismatch");
    }

    GpuFullAttentionOutputHandoffRun run;
    run.residual.assign(residual_input.size(), 0.0f);
    run.normalized.assign(residual_input.size(), 0.0f);

    try {
      cl_mem q_rope_buffer =
          resident_q_rope != nullptr
              ? resident_q_rope->buffer
              : EnsureScratchBuffer(
                    full_core_handoff_scratch_q_rope_,
                    q_rope.size() * sizeof(float), kClMemReadOnly,
                    "full attention handoff q_rope");
      cl_mem k_history_buffer =
          resident_k_history != nullptr
              ? resident_k_history->buffer
              : EnsureScratchBuffer(
                    full_core_handoff_scratch_k_history_,
                    k_history_flat.size() * sizeof(float), kClMemReadOnly,
                    "full attention handoff k history");
      cl_mem v_history_buffer =
          resident_v_history != nullptr
              ? resident_v_history->buffer
              : EnsureScratchBuffer(
                    full_core_handoff_scratch_v_history_,
                    v_history_flat.size() * sizeof(float), kClMemReadOnly,
                    "full attention handoff v history");
      cl_mem q_full_buffer =
          resident_q_full != nullptr
              ? resident_q_full->buffer
              : EnsureScratchBuffer(
                    full_core_handoff_scratch_q_full_,
                    q_full.size() * sizeof(float), kClMemReadOnly,
                    "full attention handoff q_full");
      cl_mem scores_buffer = EnsureScratchBuffer(
          full_core_handoff_scratch_scores_,
          token_count * q_head_count * sizeof(float), kClMemReadWrite,
          "full attention handoff scores");
      cl_mem pregate_buffer = EnsureScratchBuffer(
          full_core_handoff_scratch_pregate_, q_values * sizeof(float),
          kClMemReadWrite, "full attention handoff pregate");
      cl_mem gated_buffer = EnsureScratchBuffer(
          full_core_handoff_scratch_gated_, q_values * sizeof(float),
          kClMemReadWrite, "full attention handoff gated");

      const std::uint64_t block_count = q_values / kQ8QsPerBlock;
      cl_mem q8_qs_buffer = EnsureScratchBuffer(
          full_core_handoff_scratch_q8_qs_,
          block_count * kQ8QsPerBlock * sizeof(std::int8_t),
          kClMemReadWrite, "full attention handoff q8 qs");
      cl_mem q8_bsums_buffer = EnsureScratchBuffer(
          full_core_handoff_scratch_q8_bsums_,
          block_count * kQ8BsumsPerBlock * sizeof(std::int16_t),
          kClMemReadWrite, "full attention handoff q8 bsums");
      cl_mem q8_d_buffer = EnsureScratchBuffer(
          full_core_handoff_scratch_q8_d_, block_count * sizeof(float),
          kClMemReadWrite, "full attention handoff q8 d");
      cl_mem projection_buffer = EnsureScratchBuffer(
          full_core_handoff_scratch_projection_, resident.rows * sizeof(float),
          kClMemReadWrite, "full attention handoff projection");
      cl_mem residual_input_buffer =
          resident_residual_input != nullptr
              ? resident_residual_input->buffer
              : EnsureScratchBuffer(
                    full_core_handoff_scratch_residual_input_,
                    residual_input.size() * sizeof(float), kClMemReadOnly,
                    "full attention handoff residual input");
      cl_mem norm_weight_buffer =
          resident_norm_weight != nullptr
              ? resident_norm_weight->buffer
              : EnsureScratchBuffer(
                    full_core_handoff_scratch_norm_weight_,
                    norm_weight.size() * sizeof(float), kClMemReadOnly,
                    "full attention handoff norm weight");
      cl_mem residual_buffer = EnsureScratchBuffer(
          full_core_handoff_scratch_residual_,
          run.residual.size() * sizeof(float), kClMemReadWrite,
          "full attention handoff residual");
      cl_mem normalized_buffer = EnsureScratchBuffer(
          full_core_handoff_scratch_normalized_,
          run.normalized.size() * sizeof(float), kClMemWriteOnly,
          "full attention handoff normalized");
      cl_mem scale_buffer = EnsureScratchBuffer(
          rmsnorm_hidden_scratch_scale_, sizeof(float), kClMemReadWrite,
          "full attention handoff norm scale");

      if (resident_q_rope == nullptr) {
        Check(api_.clEnqueueWriteBuffer(
                  queue_, q_rope_buffer, kClTrue, 0,
                  q_rope.size() * sizeof(float), q_rope.data(), 0, nullptr,
                  nullptr),
              "clEnqueueWriteBuffer(full attention handoff q_rope)");
      }
      if (resident_k_history == nullptr) {
        Check(api_.clEnqueueWriteBuffer(
                  queue_, k_history_buffer, kClTrue, 0,
                  k_history_flat.size() * sizeof(float), k_history_flat.data(),
                  0, nullptr, nullptr),
              "clEnqueueWriteBuffer(full attention handoff k history)");
      }
      if (resident_v_history == nullptr) {
        Check(api_.clEnqueueWriteBuffer(
                  queue_, v_history_buffer, kClTrue, 0,
                  v_history_flat.size() * sizeof(float), v_history_flat.data(),
                  0, nullptr, nullptr),
              "clEnqueueWriteBuffer(full attention handoff v history)");
      }
      if (resident_q_full == nullptr) {
        Check(api_.clEnqueueWriteBuffer(
                  queue_, q_full_buffer, kClTrue, 0,
                  q_full.size() * sizeof(float), q_full.data(), 0, nullptr,
                  nullptr),
              "clEnqueueWriteBuffer(full attention handoff q_full)");
      }
      if (resident_residual_input == nullptr) {
        Check(api_.clEnqueueWriteBuffer(
                  queue_, residual_input_buffer, kClTrue, 0,
                  residual_input.size() * sizeof(float), residual_input.data(),
                  0, nullptr, nullptr),
              "clEnqueueWriteBuffer(full attention handoff residual input)");
      }
      if (resident_norm_weight == nullptr) {
        Check(api_.clEnqueueWriteBuffer(
                  queue_, norm_weight_buffer, kClTrue, 0,
                  norm_weight.size() * sizeof(float), norm_weight.data(), 0,
                  nullptr, nullptr),
              "clEnqueueWriteBuffer(full attention handoff norm weight)");
      }

      const cl_uint token_count_arg = static_cast<cl_uint>(token_count);
      const cl_uint head_dim_arg = static_cast<cl_uint>(head_dim);
      const cl_uint q_head_count_arg = static_cast<cl_uint>(q_head_count);
      const cl_uint kv_head_count_arg = static_cast<cl_uint>(kv_head_count);
      Check(api_.clSetKernelArg(kernel_full_attn_score_, 0,
                                sizeof(q_rope_buffer), &q_rope_buffer),
            "clSetKernelArg(full attention handoff score 0)");
      Check(api_.clSetKernelArg(kernel_full_attn_score_, 1,
                                sizeof(k_history_buffer), &k_history_buffer),
            "clSetKernelArg(full attention handoff score 1)");
      Check(api_.clSetKernelArg(kernel_full_attn_score_, 2,
                                sizeof(token_count_arg), &token_count_arg),
            "clSetKernelArg(full attention handoff score 2)");
      Check(api_.clSetKernelArg(kernel_full_attn_score_, 3,
                                sizeof(head_dim_arg), &head_dim_arg),
            "clSetKernelArg(full attention handoff score 3)");
      Check(api_.clSetKernelArg(kernel_full_attn_score_, 4,
                                sizeof(q_head_count_arg), &q_head_count_arg),
            "clSetKernelArg(full attention handoff score 4)");
      Check(api_.clSetKernelArg(kernel_full_attn_score_, 5,
                                sizeof(kv_head_count_arg), &kv_head_count_arg),
            "clSetKernelArg(full attention handoff score 5)");
      Check(api_.clSetKernelArg(kernel_full_attn_score_, 6,
                                sizeof(attention_scale), &attention_scale),
            "clSetKernelArg(full attention handoff score 6)");
      Check(api_.clSetKernelArg(kernel_full_attn_score_, 7,
                                sizeof(scores_buffer), &scores_buffer),
            "clSetKernelArg(full attention handoff score 7)");
      Check(api_.clSetKernelArg(kernel_full_attn_apply_score_gate_, 0,
                                sizeof(scores_buffer), &scores_buffer),
            "clSetKernelArg(full attention handoff apply score 0)");
      Check(api_.clSetKernelArg(kernel_full_attn_apply_score_gate_, 1,
                                sizeof(v_history_buffer), &v_history_buffer),
            "clSetKernelArg(full attention handoff apply score 1)");
      Check(api_.clSetKernelArg(kernel_full_attn_apply_score_gate_, 2,
                                sizeof(q_full_buffer), &q_full_buffer),
            "clSetKernelArg(full attention handoff apply score 2)");
      Check(api_.clSetKernelArg(kernel_full_attn_apply_score_gate_, 3,
                                sizeof(token_count_arg), &token_count_arg),
            "clSetKernelArg(full attention handoff apply score 3)");
      Check(api_.clSetKernelArg(kernel_full_attn_apply_score_gate_, 4,
                                sizeof(head_dim_arg), &head_dim_arg),
            "clSetKernelArg(full attention handoff apply score 4)");
      Check(api_.clSetKernelArg(kernel_full_attn_apply_score_gate_, 5,
                                sizeof(q_head_count_arg), &q_head_count_arg),
            "clSetKernelArg(full attention handoff apply score 5)");
      Check(api_.clSetKernelArg(kernel_full_attn_apply_score_gate_, 6,
                                sizeof(kv_head_count_arg), &kv_head_count_arg),
            "clSetKernelArg(full attention handoff apply score 6)");
      Check(api_.clSetKernelArg(kernel_full_attn_apply_score_gate_, 7,
                                sizeof(pregate_buffer), &pregate_buffer),
            "clSetKernelArg(full attention handoff apply score 7)");
      Check(api_.clSetKernelArg(kernel_full_attn_apply_score_gate_, 8,
                                sizeof(gated_buffer), &gated_buffer),
            "clSetKernelArg(full attention handoff apply score 8)");

      const std::size_t q_global = static_cast<std::size_t>(q_values);
      const std::size_t score_global =
          static_cast<std::size_t>(token_count * q_head_count);
      std::vector<double> core_times;
      std::vector<double> gate_times;
      core_times.reserve(static_cast<std::size_t>(repeat));
      gate_times.reserve(static_cast<std::size_t>(repeat));
      for (int i = 0; i < repeat; ++i) {
        cl_event core_event = nullptr;
        cl_event gate_event = nullptr;
        Check(api_.clEnqueueNDRangeKernel(
                  queue_, kernel_full_attn_score_, 1, nullptr, &score_global,
                  nullptr, 0, nullptr, EventOut(&core_event)),
              "clEnqueueNDRangeKernel(full attention handoff score)");
        Check(api_.clEnqueueNDRangeKernel(
                  queue_, kernel_full_attn_apply_score_gate_, 1, nullptr,
                  &q_global, nullptr, 0, nullptr, EventOut(&gate_event)),
              "clEnqueueNDRangeKernel(full attention handoff apply score)");
        Check(api_.clFinish(queue_), "clFinish(full attention handoff core/gate)");
        core_times.push_back(EventUs(api_, core_event));
        gate_times.push_back(EventUs(api_, gate_event));
        ReleaseEvent(api_, &core_event);
        ReleaseEvent(api_, &gate_event);
      }

      const auto q8_timing = RunQ8QuantizeWithBsumsKernel(
          gated_buffer, block_count, q8_qs_buffer, q8_bsums_buffer,
          q8_d_buffer, repeat);

      const std::uint64_t row_groups = resident.rows / kRowsInterleaved;
      const std::uint64_t matvec_global =
          variant == GpuQ4X8KernelVariant::kGroup8Serial ? row_groups
                                                         : resident.rows;
      run.timing.output_projection = RunKernel(
          variant, resident.buffer, q8_qs_buffer, q8_bsums_buffer, q8_d_buffer,
          projection_buffer, resident.blocks_per_row, row_groups,
          matvec_global, repeat);

      const cl_uint hidden_arg = static_cast<cl_uint>(residual_input.size());
      const cl_uint serial_reduction_arg = 0;
      Check(api_.clSetKernelArg(kernel_post_ffn_residual_, 0,
                                sizeof(residual_input_buffer),
                                &residual_input_buffer),
            "clSetKernelArg(full attention handoff residual 0)");
      Check(api_.clSetKernelArg(kernel_post_ffn_residual_, 1,
                                sizeof(projection_buffer),
                                &projection_buffer),
            "clSetKernelArg(full attention handoff residual 1)");
      Check(api_.clSetKernelArg(kernel_post_ffn_residual_, 2,
                                sizeof(hidden_arg), &hidden_arg),
            "clSetKernelArg(full attention handoff residual 2)");
      Check(api_.clSetKernelArg(kernel_post_ffn_residual_, 3,
                                sizeof(residual_buffer), &residual_buffer),
            "clSetKernelArg(full attention handoff residual 3)");
      Check(api_.clSetKernelArg(kernel_rmsnorm_hidden_scale_, 0,
                                sizeof(residual_buffer), &residual_buffer),
            "clSetKernelArg(full attention handoff norm scale 0)");
      Check(api_.clSetKernelArg(kernel_rmsnorm_hidden_scale_, 1,
                                sizeof(hidden_arg), &hidden_arg),
            "clSetKernelArg(full attention handoff norm scale 1)");
      Check(api_.clSetKernelArg(kernel_rmsnorm_hidden_scale_, 2,
                                sizeof(norm_epsilon),
                                &norm_epsilon),
            "clSetKernelArg(full attention handoff norm scale 2)");
      Check(api_.clSetKernelArg(kernel_rmsnorm_hidden_scale_, 3,
                                sizeof(scale_buffer), &scale_buffer),
            "clSetKernelArg(full attention handoff norm scale 3)");
      Check(api_.clSetKernelArg(kernel_rmsnorm_hidden_scale_, 4,
                                sizeof(serial_reduction_arg),
                                &serial_reduction_arg),
            "clSetKernelArg(full attention handoff norm scale 4)");
      Check(api_.clSetKernelArg(kernel_rmsnorm_hidden_apply_scale_, 0,
                                sizeof(residual_buffer), &residual_buffer),
            "clSetKernelArg(full attention handoff norm apply 0)");
      Check(api_.clSetKernelArg(kernel_rmsnorm_hidden_apply_scale_, 1,
                                sizeof(norm_weight_buffer),
                                &norm_weight_buffer),
            "clSetKernelArg(full attention handoff norm apply 1)");
      Check(api_.clSetKernelArg(kernel_rmsnorm_hidden_apply_scale_, 2,
                                sizeof(hidden_arg), &hidden_arg),
            "clSetKernelArg(full attention handoff norm apply 2)");
      Check(api_.clSetKernelArg(kernel_rmsnorm_hidden_apply_scale_, 3,
                                sizeof(scale_buffer), &scale_buffer),
            "clSetKernelArg(full attention handoff norm apply 3)");
      Check(api_.clSetKernelArg(kernel_rmsnorm_hidden_apply_scale_, 4,
                                sizeof(normalized_buffer),
                                &normalized_buffer),
            "clSetKernelArg(full attention handoff norm apply 4)");

      const std::size_t residual_global = residual_input.size();
      const std::size_t rmsnorm_scale_global = kRmsNormScaleLocalSize;
      const std::size_t rmsnorm_scale_local = kRmsNormScaleLocalSize;
      const std::size_t rmsnorm_apply_global = residual_input.size();
      std::vector<double> residual_times;
      std::vector<double> rmsnorm_times;
      residual_times.reserve(static_cast<std::size_t>(repeat));
      rmsnorm_times.reserve(static_cast<std::size_t>(repeat));
      for (int i = 0; i < repeat; ++i) {
        cl_event residual_event = nullptr;
        cl_event scale_event = nullptr;
        Check(api_.clEnqueueNDRangeKernel(
                  queue_, kernel_post_ffn_residual_, 1, nullptr,
                  &residual_global, nullptr, 0, nullptr, EventOut(&residual_event)),
              "clEnqueueNDRangeKernel(full attention handoff residual)");
        cl_event apply_event = nullptr;
        Check(api_.clEnqueueNDRangeKernel(
                  queue_, kernel_rmsnorm_hidden_scale_, 1, nullptr,
                  &rmsnorm_scale_global, &rmsnorm_scale_local, 0, nullptr,
                  EventOut(&scale_event)),
              "clEnqueueNDRangeKernel(full attention handoff norm scale)");
        Check(api_.clEnqueueNDRangeKernel(
                  queue_, kernel_rmsnorm_hidden_apply_scale_, 1, nullptr,
                  &rmsnorm_apply_global, nullptr, 0, nullptr, EventOut(&apply_event)),
              "clEnqueueNDRangeKernel(full attention handoff norm apply)");
        Check(api_.clFinish(queue_),
              "clFinish(full attention handoff residual/norm)");
        residual_times.push_back(EventUs(api_, residual_event));
        rmsnorm_times.push_back(EventUs(api_, scale_event) +
                                EventUs(api_, apply_event));
        ReleaseEvent(api_, &apply_event);
        ReleaseEvent(api_, &scale_event);
        ReleaseEvent(api_, &residual_event);
      }

      Check(api_.clEnqueueReadBuffer(queue_, residual_buffer, kClTrue, 0,
                                     run.residual.size() * sizeof(float),
                                     run.residual.data(), 0, nullptr, nullptr),
            "clEnqueueReadBuffer(full attention handoff residual)");
      Check(api_.clEnqueueReadBuffer(queue_, normalized_buffer, kClTrue, 0,
                                     run.normalized.size() * sizeof(float),
                                     run.normalized.data(), 0, nullptr, nullptr),
            "clEnqueueReadBuffer(full attention handoff normalized)");
      run.residual_handle = RegisterF32BufferAlias(
          &attention_front_residual_alias_handle_, residual_buffer,
          run.residual.size());
      run.normalized_handle = RegisterF32BufferAlias(
          &attention_front_normalized_alias_handle_, normalized_buffer,
          run.normalized.size());

      run.timing.core_min_us =
          *std::min_element(core_times.begin(), core_times.end());
      run.timing.core_mean_us =
          std::accumulate(core_times.begin(), core_times.end(), 0.0) /
          static_cast<double>(core_times.size());
      run.timing.gate_min_us =
          *std::min_element(gate_times.begin(), gate_times.end());
      run.timing.gate_mean_us =
          std::accumulate(gate_times.begin(), gate_times.end(), 0.0) /
          static_cast<double>(gate_times.size());
      run.timing.q8_quantize_min_us = q8_timing.min_us;
      run.timing.q8_quantize_mean_us = q8_timing.mean_us;
      run.timing.residual_min_us =
          *std::min_element(residual_times.begin(), residual_times.end());
      run.timing.residual_mean_us =
          std::accumulate(residual_times.begin(), residual_times.end(), 0.0) /
          static_cast<double>(residual_times.size());
      run.timing.rmsnorm_min_us =
          *std::min_element(rmsnorm_times.begin(), rmsnorm_times.end());
      run.timing.rmsnorm_mean_us =
          std::accumulate(rmsnorm_times.begin(), rmsnorm_times.end(), 0.0) /
          static_cast<double>(rmsnorm_times.size());
      run.timing.shell_sum_min_us =
          run.timing.core_min_us + run.timing.gate_min_us +
          run.timing.q8_quantize_min_us +
          run.timing.output_projection.min_us + run.timing.residual_min_us +
          run.timing.rmsnorm_min_us;
      run.timing.shell_sum_mean_us =
          run.timing.core_mean_us + run.timing.gate_mean_us +
          run.timing.q8_quantize_mean_us +
          run.timing.output_projection.mean_us +
          run.timing.residual_mean_us + run.timing.rmsnorm_mean_us;
      run.timing.q_global_work_items = q_global;
      run.timing.q8_quantize_global_work_items =
          q8_timing.global_work_items;
      run.timing.residual_global_work_items = residual_global;
      run.timing.rmsnorm_global_work_items =
          rmsnorm_scale_global + rmsnorm_apply_global;
    } catch (...) {
      throw;
    }
    return run;
  }

  GpuQ4X8ConvHandoffRun RunThenConv(const std::vector<std::uint8_t>& packed,
                                    const std::vector<std::int8_t>& q8_qs,
                                    const std::vector<std::int16_t>& q8_bsums,
                                    const std::vector<float>& q8_d,
                                    const std::vector<float>& conv_weights,
                                    const std::vector<float>& conv_state,
                                    std::uint64_t rows,
                                    std::uint64_t blocks_per_row,
                                    std::uint64_t conv_kernel_size,
                                    int repeat,
                                    GpuQ4X8KernelVariant variant) {
    ValidateRunInputs(packed, q8_qs, q8_bsums, q8_d, rows, blocks_per_row, repeat);
    ValidateConvInputs(conv_weights, conv_state, rows, conv_kernel_size);
    GpuQ4X8ConvHandoffRun run;
    run.qkv_mixed.assign(static_cast<std::size_t>(rows), 0.0f);
    run.conv_output_raw.assign(static_cast<std::size_t>(rows), 0.0f);
    run.conv_state.assign(static_cast<std::size_t>(rows * (conv_kernel_size - 1)), 0.0f);
    cl_mem packed_buffer = nullptr, q8_qs_buffer = nullptr, q8_bsums_buffer = nullptr;
    cl_mem q8_d_buffer = nullptr, qkv_buffer = nullptr, conv_weights_buffer = nullptr;
    cl_mem conv_state_buffer = nullptr, conv_output_buffer = nullptr, next_state_buffer = nullptr;
    try {
      CreateBuffers(packed, q8_qs, q8_bsums, q8_d, run.qkv_mixed,
                    &packed_buffer, &q8_qs_buffer, &q8_bsums_buffer, &q8_d_buffer,
                    &qkv_buffer);
      CreateConvBuffers(conv_weights, conv_state, run.conv_output_raw, run.conv_state,
                        &conv_weights_buffer, &conv_state_buffer, &conv_output_buffer,
                        &next_state_buffer);
      const std::uint64_t row_groups = rows / kRowsInterleaved;
      const std::uint64_t matvec_global =
          variant == GpuQ4X8KernelVariant::kGroup8Serial ? row_groups : rows;
      run.timing = RunHandoffKernels(variant, packed_buffer, q8_qs_buffer,
                                     q8_bsums_buffer, q8_d_buffer, qkv_buffer,
                                     conv_weights_buffer, conv_state_buffer,
                                     conv_output_buffer, next_state_buffer,
                                     blocks_per_row, row_groups, matvec_global,
                                     rows, conv_kernel_size, repeat);
      Check(api_.clEnqueueReadBuffer(queue_, qkv_buffer, kClTrue, 0,
                                     run.qkv_mixed.size() * sizeof(float),
                                     run.qkv_mixed.data(), 0, nullptr, nullptr),
            "clEnqueueReadBuffer(qkv)");
      Check(api_.clEnqueueReadBuffer(queue_, conv_output_buffer, kClTrue, 0,
                                     run.conv_output_raw.size() * sizeof(float),
                                     run.conv_output_raw.data(), 0, nullptr, nullptr),
            "clEnqueueReadBuffer(conv_output)");
      Check(api_.clEnqueueReadBuffer(queue_, next_state_buffer, kClTrue, 0,
                                     run.conv_state.size() * sizeof(float),
                                     run.conv_state.data(), 0, nullptr, nullptr),
            "clEnqueueReadBuffer(next_state)");
    } catch (...) {
      ReleaseMem(api_, &next_state_buffer);
      ReleaseMem(api_, &conv_output_buffer);
      ReleaseMem(api_, &conv_state_buffer);
      ReleaseMem(api_, &conv_weights_buffer);
      ReleaseMem(api_, &qkv_buffer);
      ReleaseMem(api_, &q8_d_buffer);
      ReleaseMem(api_, &q8_bsums_buffer);
      ReleaseMem(api_, &q8_qs_buffer);
      ReleaseMem(api_, &packed_buffer);
      throw;
    }
    ReleaseMem(api_, &next_state_buffer);
    ReleaseMem(api_, &conv_output_buffer);
    ReleaseMem(api_, &conv_state_buffer);
    ReleaseMem(api_, &conv_weights_buffer);
    ReleaseMem(api_, &qkv_buffer);
    ReleaseMem(api_, &q8_d_buffer);
    ReleaseMem(api_, &q8_bsums_buffer);
    ReleaseMem(api_, &q8_qs_buffer);
    ReleaseMem(api_, &packed_buffer);
    return run;
  }

  GpuLinearAttentionPostConvPrepRun RunPostConvPrep(
      const std::vector<float>& conv_output_raw,
      std::uint64_t head_dim,
      std::uint64_t query_heads,
      std::uint64_t value_heads,
      float norm_epsilon,
      int repeat,
      bool readback_intermediates = true) {
    ValidatePostConvPrepInputs(conv_output_raw, head_dim, query_heads, value_heads, repeat);
    const std::uint64_t q_values = head_dim * query_heads;
    const std::uint64_t v_values = head_dim * value_heads;
    GpuLinearAttentionPostConvPrepRun run;
    if (readback_intermediates) {
      run.conv_output_silu.assign(conv_output_raw.size(), 0.0f);
      run.q_conv.assign(static_cast<std::size_t>(q_values), 0.0f);
      run.k_conv.assign(static_cast<std::size_t>(q_values), 0.0f);
    }
    run.v_conv_predelta.assign(static_cast<std::size_t>(v_values), 0.0f);
    run.q_conv_predelta.assign(static_cast<std::size_t>(q_values), 0.0f);
    run.k_conv_predelta.assign(static_cast<std::size_t>(q_values), 0.0f);

    cl_mem raw_buffer = nullptr, silu_buffer = nullptr, q_buffer = nullptr;
    cl_mem k_buffer = nullptr, v_buffer = nullptr, q_norm_buffer = nullptr, k_norm_buffer = nullptr;
    try {
      EnsurePostConvPrepBuffers(conv_output_raw, run, q_values,
                                &raw_buffer, &silu_buffer, &q_buffer, &k_buffer,
                                &v_buffer, &q_norm_buffer, &k_norm_buffer);
      run.timing = RunPostConvPrepKernels(raw_buffer, silu_buffer, q_buffer, k_buffer,
                                          v_buffer, q_norm_buffer, k_norm_buffer,
                                          head_dim, query_heads, q_values,
                                          conv_output_raw.size(), norm_epsilon,
                                          repeat);
      ReadPostConvPrepBuffers(run, silu_buffer, q_buffer, k_buffer, v_buffer,
                              q_norm_buffer, k_norm_buffer,
                              readback_intermediates);
    } catch (...) {
      if (queue_ != nullptr) {
        (void)api_.clFinish(queue_);
      }
      throw;
    }
    return run;
  }

  GpuLinearAttentionPostConvPrepRun RunPostConvPrepFused(
      const std::vector<float>& conv_output_raw,
      std::uint64_t head_dim,
      std::uint64_t query_heads,
      std::uint64_t value_heads,
      float norm_epsilon,
      int repeat,
      bool readback_intermediates = true) {
    ValidatePostConvPrepInputs(conv_output_raw, head_dim, query_heads, value_heads, repeat);
    const std::uint64_t q_values = head_dim * query_heads;
    const std::uint64_t v_values = head_dim * value_heads;
    GpuLinearAttentionPostConvPrepRun run;
    if (readback_intermediates) {
      run.conv_output_silu.assign(conv_output_raw.size(), 0.0f);
      run.q_conv.assign(static_cast<std::size_t>(q_values), 0.0f);
      run.k_conv.assign(static_cast<std::size_t>(q_values), 0.0f);
    }
    run.v_conv_predelta.assign(static_cast<std::size_t>(v_values), 0.0f);
    run.q_conv_predelta.assign(static_cast<std::size_t>(q_values), 0.0f);
    run.k_conv_predelta.assign(static_cast<std::size_t>(q_values), 0.0f);

    cl_mem raw_buffer = nullptr, silu_buffer = nullptr, q_buffer = nullptr;
    cl_mem k_buffer = nullptr, v_buffer = nullptr, q_norm_buffer = nullptr, k_norm_buffer = nullptr;
    try {
      EnsurePostConvPrepBuffers(conv_output_raw, run, q_values,
                                &raw_buffer, &silu_buffer, &q_buffer, &k_buffer,
                                &v_buffer, &q_norm_buffer, &k_norm_buffer);
      run.timing = RunPostConvPrepFusedKernel(raw_buffer, silu_buffer, q_buffer,
                                              k_buffer, v_buffer, q_norm_buffer,
                                              k_norm_buffer, head_dim, query_heads,
                                              value_heads, norm_epsilon, repeat);
      ReadPostConvPrepBuffers(run, silu_buffer, q_buffer, k_buffer, v_buffer,
                              q_norm_buffer, k_norm_buffer,
                              readback_intermediates);
    } catch (...) {
      if (queue_ != nullptr) {
        (void)api_.clFinish(queue_);
      }
      throw;
    }
    return run;
  }

  GpuLinearAttentionDeltaRun RunLinearAttentionDelta(
      const std::vector<float>& q,
      const std::vector<float>& k,
      const std::vector<float>& v,
      const std::vector<float>& gate,
      const std::vector<float>& beta,
      const std::vector<float>& recurrent_state,
      const std::vector<float>& z,
      const std::vector<float>& norm_weight,
      std::uint64_t head_dim,
      std::uint64_t query_heads,
      std::uint64_t value_heads,
      float norm_epsilon,
      int repeat,
      bool cpu_shape_final_norm) {
    ValidateLinearAttentionDeltaInputs(q, k, v, gate, beta, recurrent_state,
                                       z, norm_weight, head_dim, query_heads,
                                       value_heads, repeat);
    const std::uint64_t v_values = head_dim * value_heads;
    GpuLinearAttentionDeltaRun run;
    run.attention_output.assign(static_cast<std::size_t>(v_values), 0.0f);
    run.recurrent_state.assign(recurrent_state.size(), 0.0f);
    run.final_output.assign(static_cast<std::size_t>(v_values), 0.0f);

    cl_mem q_buffer = nullptr, k_buffer = nullptr, v_buffer = nullptr, gate_buffer = nullptr;
    cl_mem beta_buffer = nullptr, state_buffer = nullptr, z_buffer = nullptr, norm_buffer = nullptr;
    cl_mem attention_buffer = nullptr, next_state_buffer = nullptr, final_buffer = nullptr;
    auto release_all = [&]() {
      ReleaseMem(api_, &final_buffer);
      ReleaseMem(api_, &next_state_buffer);
      ReleaseMem(api_, &attention_buffer);
      ReleaseMem(api_, &norm_buffer);
      ReleaseMem(api_, &z_buffer);
      ReleaseMem(api_, &state_buffer);
      ReleaseMem(api_, &beta_buffer);
      ReleaseMem(api_, &gate_buffer);
      ReleaseMem(api_, &v_buffer);
      ReleaseMem(api_, &k_buffer);
      ReleaseMem(api_, &q_buffer);
    };
    try {
      CreateLinearAttentionDeltaBuffers(q, k, v, gate, beta, recurrent_state,
                                        z, norm_weight, run,
                                        &q_buffer, &k_buffer, &v_buffer,
                                        &gate_buffer, &beta_buffer,
                                        &state_buffer, &z_buffer, &norm_buffer,
                                        &attention_buffer, &next_state_buffer,
                                        &final_buffer);
      run.timing = RunLinearAttentionDeltaKernels(
          q_buffer, k_buffer, v_buffer, gate_buffer, beta_buffer, state_buffer,
          z_buffer, norm_buffer, attention_buffer, next_state_buffer,
          final_buffer, head_dim, query_heads, value_heads, norm_epsilon,
          repeat, cpu_shape_final_norm);
      ReadLinearAttentionDeltaBuffers(run, attention_buffer, next_state_buffer,
                                      final_buffer);
    } catch (...) {
      release_all();
      throw;
    }
    release_all();
    return run;
  }

  GpuLinearAttentionDeltaRun RunLinearAttentionDeltaResidentState(
      std::uint64_t state_handle,
      const std::vector<float>& q,
      const std::vector<float>& k,
      const std::vector<float>& v,
      const std::vector<float>& gate,
      const std::vector<float>& beta,
      const std::vector<float>& z,
      const std::vector<float>& norm_weight,
      std::uint64_t head_dim,
      std::uint64_t query_heads,
      std::uint64_t value_heads,
      float norm_epsilon,
      int repeat,
      bool readback_state,
      bool cpu_shape_final_norm,
      bool readback_attention_output,
      bool readback_final_output) {
    const auto& resident_state = ResidentF32BufferForHandle(state_handle);
    ValidateLinearAttentionDeltaInputSizes(
        q, k, v, gate, beta, resident_state.values, z, norm_weight,
        head_dim, query_heads, value_heads, repeat);
    const std::uint64_t v_values = head_dim * value_heads;
    GpuLinearAttentionDeltaRun run;
    run.attention_output.assign(static_cast<std::size_t>(v_values), 0.0f);
    run.final_output.assign(static_cast<std::size_t>(v_values), 0.0f);
    if (readback_state) {
      run.recurrent_state.assign(resident_state.values, 0.0f);
    }

    cl_mem q_buffer = nullptr, k_buffer = nullptr, v_buffer = nullptr;
    cl_mem gate_buffer = nullptr, beta_buffer = nullptr, z_buffer = nullptr;
    cl_mem norm_buffer = nullptr, attention_buffer = nullptr;
    cl_mem final_buffer = nullptr;
    auto make_read = [&](ScratchBuffer& scratch,
                         const std::vector<float>& values,
                         const char* name) -> cl_mem {
      cl_mem mem = EnsureScratchBuffer(
          scratch, values.size() * sizeof(float), kClMemReadOnly,
          (std::string("delta resident ") + name).c_str());
      Check(api_.clEnqueueWriteBuffer(queue_, mem, kClFalse, 0,
                                      values.size() * sizeof(float),
                                      values.data(), 0, nullptr, nullptr),
            std::string("clEnqueueWriteBuffer(delta resident ") + name + ")");
      return mem;
    };
    auto make_write = [&](ScratchBuffer& scratch,
                          std::size_t values,
                          const char* name) -> cl_mem {
      return EnsureScratchBuffer(
          scratch, values * sizeof(float), kClMemWriteOnly,
          (std::string("delta resident ") + name).c_str());
    };
    const auto delta_wall_ns =
        [](std::chrono::steady_clock::time_point begin,
           std::chrono::steady_clock::time_point end) -> std::uint64_t {
      return static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count());
    };
    try {
      const auto input_upload_begin = std::chrono::steady_clock::now();
      q_buffer = make_read(linear_delta_scratch_q_, q, "q");
      k_buffer = make_read(linear_delta_scratch_k_, k, "k");
      v_buffer = make_read(linear_delta_scratch_v_, v, "v");
      gate_buffer = make_read(linear_delta_scratch_gate_, gate, "gate");
      beta_buffer = make_read(linear_delta_scratch_beta_, beta, "beta");
      z_buffer = make_read(linear_delta_scratch_z_, z, "z");
      norm_buffer = make_read(linear_delta_scratch_norm_, norm_weight, "norm");
      const auto input_upload_end = std::chrono::steady_clock::now();
      attention_buffer = make_write(
          linear_delta_scratch_attention_, run.attention_output.size(), "attention");
      final_buffer = make_write(
          linear_delta_scratch_final_, run.final_output.size(), "final");
      const auto kernel_begin = std::chrono::steady_clock::now();
      run.timing = RunLinearAttentionDeltaKernels(
          q_buffer, k_buffer, v_buffer, gate_buffer, beta_buffer,
          resident_state.buffer, z_buffer, norm_buffer, attention_buffer,
          resident_state.buffer, final_buffer, head_dim, query_heads,
          value_heads, norm_epsilon, repeat, cpu_shape_final_norm);
      const auto kernel_end = std::chrono::steady_clock::now();
      run.timing.input_upload_wall_ns =
          delta_wall_ns(input_upload_begin, input_upload_end);
      run.timing.kernel_wall_ns = delta_wall_ns(kernel_begin, kernel_end);
      if (readback_attention_output) {
        const auto attention_read_begin = std::chrono::steady_clock::now();
        Check(api_.clEnqueueReadBuffer(queue_, attention_buffer, kClTrue, 0,
                                       run.attention_output.size() * sizeof(float),
                                       run.attention_output.data(), 0, nullptr,
                                       nullptr),
              "clEnqueueReadBuffer(delta resident attention)");
        run.timing.attention_read_wall_ns = delta_wall_ns(
            attention_read_begin, std::chrono::steady_clock::now());
      }
      run.final_output_handle = RegisterF32BufferAlias(
          &linear_delta_final_alias_handle_, final_buffer,
          run.final_output.size());
      if (readback_final_output) {
        const auto final_read_begin = std::chrono::steady_clock::now();
        Check(api_.clEnqueueReadBuffer(queue_, final_buffer, kClTrue, 0,
                                       run.final_output.size() * sizeof(float),
                                       run.final_output.data(), 0, nullptr,
                                       nullptr),
              "clEnqueueReadBuffer(delta resident final)");
        run.timing.final_read_wall_ns =
            delta_wall_ns(final_read_begin, std::chrono::steady_clock::now());
      }
      if (readback_state) {
        const auto state_read_begin = std::chrono::steady_clock::now();
        Check(api_.clEnqueueReadBuffer(queue_, resident_state.buffer, kClTrue, 0,
                                       run.recurrent_state.size() * sizeof(float),
                                       run.recurrent_state.data(), 0, nullptr,
                                       nullptr),
              "clEnqueueReadBuffer(delta resident state)");
        run.timing.state_read_wall_ns =
            delta_wall_ns(state_read_begin, std::chrono::steady_clock::now());
      }
    } catch (...) {
      throw;
    }
    return run;
  }

  GpuLinearAttentionDeltaRun RunPostConvPrepThenLinearAttentionDeltaResidentState(
      std::uint64_t conv_output_handle,
      std::uint64_t state_handle,
      const std::vector<float>& gate,
      const std::vector<float>& beta,
      const std::vector<float>& z,
      const std::vector<float>& norm_weight,
      std::uint64_t head_dim,
      std::uint64_t query_heads,
      std::uint64_t value_heads,
      float norm_epsilon,
      int repeat,
      bool readback_state,
      bool cpu_shape_final_norm,
      bool readback_attention_output,
      bool readback_final_output,
      bool cpuorder = false) {
    const auto& conv_output = ResidentF32BufferForHandle(conv_output_handle);
    const auto& resident_state = ResidentF32BufferForHandle(state_handle);
    Require(head_dim > 0, "fused postconv delta head_dim must be nonzero");
    Require(query_heads > 0, "fused postconv delta query_heads must be nonzero");
    Require(value_heads > 0, "fused postconv delta value_heads must be nonzero");
    Require(value_heads % query_heads == 0,
            "fused postconv delta value_heads must broadcast query_heads");
    Require(repeat > 0, "repeat must be positive");
    const std::uint64_t q_values = head_dim * query_heads;
    const std::uint64_t v_values = head_dim * value_heads;
    Require(conv_output.values == static_cast<std::size_t>(2 * q_values + v_values),
            "fused postconv delta conv output size mismatch");
    Require(gate.size() == static_cast<std::size_t>(value_heads),
            "fused postconv delta gate size mismatch");
    Require(beta.size() == static_cast<std::size_t>(value_heads),
            "fused postconv delta beta size mismatch");
    Require(resident_state.values ==
                static_cast<std::size_t>(head_dim * head_dim * value_heads),
            "fused postconv delta recurrent state size mismatch");
    Require(z.size() == static_cast<std::size_t>(v_values),
            "fused postconv delta z size mismatch");
    Require(norm_weight.size() == static_cast<std::size_t>(head_dim),
            "fused postconv delta norm weight size mismatch");

    std::vector<float> decay;
    std::vector<float> z_silu;
    const std::vector<float>* gate_kernel_input = &gate;
    const std::vector<float>* z_kernel_input = &z;
    if (cpuorder) {
      decay.resize(gate.size());
      for (std::size_t i = 0; i < gate.size(); ++i) {
        decay[i] = std::exp(gate[i]);
      }
      z_silu.resize(z.size());
      for (std::size_t i = 0; i < z.size(); ++i) {
        const double x = static_cast<double>(z[i]);
        const double sigmoid_double =
            x >= 0.0 ? 1.0 / (1.0 + std::exp(-x))
                     : std::exp(x) / (1.0 + std::exp(x));
        const float sigmoid = static_cast<float>(sigmoid_double);
        z_silu[i] = z[i] * sigmoid;
      }
      gate_kernel_input = &decay;
      z_kernel_input = &z_silu;
    }

    GpuLinearAttentionDeltaRun run;
    run.attention_output.assign(static_cast<std::size_t>(v_values), 0.0f);
    run.final_output.assign(static_cast<std::size_t>(v_values), 0.0f);
    if (readback_state) {
      run.recurrent_state.assign(resident_state.values, 0.0f);
    }

    const auto wall_ns =
        [](std::chrono::steady_clock::time_point begin,
           std::chrono::steady_clock::time_point end) -> std::uint64_t {
      return static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count());
    };
    auto make_read = [&](ScratchBuffer& scratch,
                         const std::vector<float>& values,
                         const char* name) -> cl_mem {
      cl_mem mem = EnsureScratchBuffer(
          scratch, values.size() * sizeof(float), kClMemReadOnly,
          (std::string("fused postconv delta ") + name).c_str());
      Check(api_.clEnqueueWriteBuffer(queue_, mem, kClFalse, 0,
                                      values.size() * sizeof(float),
                                      values.data(), 0, nullptr, nullptr),
            std::string("clEnqueueWriteBuffer(fused postconv delta ") + name + ")");
      return mem;
    };
    auto make_write = [&](ScratchBuffer& scratch,
                          std::size_t values,
                          const char* name) -> cl_mem {
      return EnsureScratchBuffer(
          scratch, values * sizeof(float), kClMemWriteOnly,
          (std::string("fused postconv delta ") + name).c_str());
    };

    const auto input_upload_begin = std::chrono::steady_clock::now();
    cl_mem silu_buffer = EnsureScratchBuffer(
        postconv_prep_scratch_silu_, conv_output.bytes, kClMemWriteOnly,
        "fused postconv silu");
    cl_mem q_buffer = EnsureScratchBuffer(
        postconv_prep_scratch_q_, q_values * sizeof(float), kClMemReadWrite,
        "fused postconv q");
    cl_mem k_buffer = EnsureScratchBuffer(
        postconv_prep_scratch_k_, q_values * sizeof(float), kClMemReadWrite,
        "fused postconv k");
    cl_mem v_buffer = EnsureScratchBuffer(
        postconv_prep_scratch_v_, v_values * sizeof(float), kClMemReadWrite,
        "fused postconv v");
    cl_mem q_norm_buffer = EnsureScratchBuffer(
        postconv_prep_scratch_q_norm_, q_values * sizeof(float), kClMemReadWrite,
        "fused postconv q norm");
    cl_mem k_norm_buffer = EnsureScratchBuffer(
        postconv_prep_scratch_k_norm_, q_values * sizeof(float), kClMemReadWrite,
        "fused postconv k norm");
    cl_mem gate_buffer = make_read(
        linear_delta_scratch_gate_, *gate_kernel_input,
        cpuorder ? "decay" : "gate");
    cl_mem beta_buffer = make_read(linear_delta_scratch_beta_, beta, "beta");
    cl_mem z_buffer = make_read(
        linear_delta_scratch_z_, *z_kernel_input,
        cpuorder ? "z_silu" : "z");
    cl_mem norm_buffer = make_read(linear_delta_scratch_norm_, norm_weight, "norm");
    const auto input_upload_end = std::chrono::steady_clock::now();
    cl_mem attention_buffer = make_write(
        linear_delta_scratch_attention_, run.attention_output.size(), "attention");
    cl_mem final_buffer = make_write(
        linear_delta_scratch_final_, run.final_output.size(), "final");

    const auto postconv_begin = std::chrono::steady_clock::now();
    const auto postconv_timing = RunPostConvPrepKernels(
        conv_output.buffer, silu_buffer, q_buffer, k_buffer, v_buffer,
        q_norm_buffer, k_norm_buffer, head_dim, query_heads, q_values,
        conv_output.values, norm_epsilon, repeat, cpuorder);
    const auto postconv_end = std::chrono::steady_clock::now();
    const auto kernel_begin = std::chrono::steady_clock::now();
    run.timing = RunLinearAttentionDeltaKernels(
        q_norm_buffer, k_norm_buffer, v_buffer, gate_buffer, beta_buffer,
        resident_state.buffer, z_buffer, norm_buffer, attention_buffer,
        resident_state.buffer, final_buffer, head_dim, query_heads,
        value_heads, norm_epsilon, repeat, cpu_shape_final_norm, cpuorder);
    const auto kernel_end = std::chrono::steady_clock::now();
    run.timing.postconv_prep_wall_ns = wall_ns(postconv_begin, postconv_end);
    run.timing.input_upload_wall_ns = wall_ns(input_upload_begin, input_upload_end);
    run.timing.kernel_wall_ns =
        wall_ns(kernel_begin, kernel_end) + run.timing.postconv_prep_wall_ns;
    run.timing.postconv_silu_split_min_us = postconv_timing.silu_split_min_us;
    run.timing.postconv_silu_split_mean_us = postconv_timing.silu_split_mean_us;
    run.timing.postconv_q_l2_min_us = postconv_timing.q_l2_min_us;
    run.timing.postconv_q_l2_mean_us = postconv_timing.q_l2_mean_us;
    run.timing.postconv_k_l2_min_us = postconv_timing.k_l2_min_us;
    run.timing.postconv_k_l2_mean_us = postconv_timing.k_l2_mean_us;
    run.timing.delta_min_us +=
        postconv_timing.silu_split_min_us + postconv_timing.q_l2_min_us;
    run.timing.delta_mean_us +=
        postconv_timing.silu_split_mean_us + postconv_timing.q_l2_mean_us;
    if (readback_attention_output) {
      const auto attention_read_begin = std::chrono::steady_clock::now();
      Check(api_.clEnqueueReadBuffer(queue_, attention_buffer, kClTrue, 0,
                                     run.attention_output.size() * sizeof(float),
                                     run.attention_output.data(), 0, nullptr,
                                     nullptr),
            "clEnqueueReadBuffer(fused postconv delta attention)");
      run.timing.attention_read_wall_ns =
          wall_ns(attention_read_begin, std::chrono::steady_clock::now());
    }
    run.final_output_handle = RegisterF32BufferAlias(
        &linear_delta_final_alias_handle_, final_buffer,
        run.final_output.size());
    if (readback_final_output) {
      const auto final_read_begin = std::chrono::steady_clock::now();
      Check(api_.clEnqueueReadBuffer(queue_, final_buffer, kClTrue, 0,
                                     run.final_output.size() * sizeof(float),
                                     run.final_output.data(), 0, nullptr,
                                     nullptr),
            "clEnqueueReadBuffer(fused postconv delta final)");
      run.timing.final_read_wall_ns =
          wall_ns(final_read_begin, std::chrono::steady_clock::now());
    }
    if (readback_state) {
      const auto state_read_begin = std::chrono::steady_clock::now();
      Check(api_.clEnqueueReadBuffer(queue_, resident_state.buffer, kClTrue, 0,
                                     run.recurrent_state.size() * sizeof(float),
                                     run.recurrent_state.data(), 0, nullptr,
                                     nullptr),
            "clEnqueueReadBuffer(fused postconv delta state)");
      run.timing.state_read_wall_ns =
          wall_ns(state_read_begin, std::chrono::steady_clock::now());
    }
    return run;
  }

 private:
  cl_kernel CreateNamedKernel(const char* name) {
    cl_int err = kClSuccess;
    cl_kernel kernel = api_.clCreateKernel(program_, name, &err);
    Check(err, std::string("clCreateKernel(") + name + ")");
    return kernel;
  }

  cl_kernel CreateKernel(GpuQ4X8KernelVariant variant) {
    return CreateNamedKernel(KernelFunctionName(variant));
  }

  cl_kernel Kernel(GpuQ4X8KernelVariant variant) const {
    return variant == GpuQ4X8KernelVariant::kGroup8Serial ? kernel_group8_ : kernel_rowlane_;
  }

  void CaptureBuildLog() {
    std::size_t log_size = 0;
    api_.clGetProgramBuildInfo(program_, selected_.device, kClProgramBuildLog, 0, nullptr, &log_size);
    if (log_size <= 1) {
      return;
    }
    build_log_.resize(log_size, '\0');
    api_.clGetProgramBuildInfo(program_, selected_.device, kClProgramBuildLog,
                               log_size, build_log_.data(), nullptr);
    if (!build_log_.empty() && build_log_.back() == '\0') {
      build_log_.pop_back();
    }
  }

  void CreateBuffers(const std::vector<std::uint8_t>& packed,
                     const std::vector<std::int8_t>& q8_qs,
                     const std::vector<std::int16_t>& q8_bsums,
                     const std::vector<float>& q8_d,
                     const std::vector<float>& output,
                     cl_mem* packed_buffer,
                     cl_mem* q8_qs_buffer,
                     cl_mem* q8_bsums_buffer,
                     cl_mem* q8_d_buffer,
                     cl_mem* out_buffer) {
    cl_int err = kClSuccess;
    *packed_buffer = api_.clCreateBuffer(context_, kClMemReadOnly, packed.size(), nullptr, &err);
    Check(err, "clCreateBuffer(packed)");
    *q8_qs_buffer = api_.clCreateBuffer(context_, kClMemReadOnly, q8_qs.size(), nullptr, &err);
    Check(err, "clCreateBuffer(q8_qs)");
    *q8_bsums_buffer = api_.clCreateBuffer(context_, kClMemReadOnly,
                                           q8_bsums.size() * sizeof(std::int16_t), nullptr, &err);
    Check(err, "clCreateBuffer(q8_bsums)");
    *q8_d_buffer = api_.clCreateBuffer(context_, kClMemReadOnly,
                                       q8_d.size() * sizeof(float), nullptr, &err);
    Check(err, "clCreateBuffer(q8_d)");
    *out_buffer = api_.clCreateBuffer(context_, kClMemWriteOnly,
                                      output.size() * sizeof(float), nullptr, &err);
    Check(err, "clCreateBuffer(out)");
    Check(api_.clEnqueueWriteBuffer(queue_, *packed_buffer, kClTrue, 0, packed.size(),
                                    packed.data(), 0, nullptr, nullptr),
          "clEnqueueWriteBuffer(packed)");
    Check(api_.clEnqueueWriteBuffer(queue_, *q8_qs_buffer, kClTrue, 0, q8_qs.size(),
                                    q8_qs.data(), 0, nullptr, nullptr),
          "clEnqueueWriteBuffer(q8_qs)");
    Check(api_.clEnqueueWriteBuffer(queue_, *q8_bsums_buffer, kClTrue, 0,
                                    q8_bsums.size() * sizeof(std::int16_t),
                                    q8_bsums.data(), 0, nullptr, nullptr),
          "clEnqueueWriteBuffer(q8_bsums)");
    Check(api_.clEnqueueWriteBuffer(queue_, *q8_d_buffer, kClTrue, 0,
                                    q8_d.size() * sizeof(float), q8_d.data(),
                                    0, nullptr, nullptr),
          "clEnqueueWriteBuffer(q8_d)");
  }

  void CreateRunBuffersWithoutPacked(const std::vector<std::int8_t>& q8_qs,
                                     const std::vector<std::int16_t>& q8_bsums,
                                     const std::vector<float>& q8_d,
                                     const std::vector<float>& output,
                                     cl_mem* q8_qs_buffer,
                                     cl_mem* q8_bsums_buffer,
                                     cl_mem* q8_d_buffer,
                                     cl_mem* out_buffer,
                                     cl_mem_flags out_flags = kClMemWriteOnly,
                                     bool nonblocking_uploads = false) {
    cl_int err = kClSuccess;
    *q8_qs_buffer = api_.clCreateBuffer(context_, kClMemReadOnly, q8_qs.size(), nullptr, &err);
    Check(err, "clCreateBuffer(resident q8_qs)");
    *q8_bsums_buffer = api_.clCreateBuffer(context_, kClMemReadOnly,
                                           q8_bsums.size() * sizeof(std::int16_t), nullptr, &err);
    Check(err, "clCreateBuffer(resident q8_bsums)");
    *q8_d_buffer = api_.clCreateBuffer(context_, kClMemReadOnly,
                                       q8_d.size() * sizeof(float), nullptr, &err);
    Check(err, "clCreateBuffer(resident q8_d)");
    *out_buffer = api_.clCreateBuffer(context_, out_flags,
                                      output.size() * sizeof(float), nullptr, &err);
    Check(err, "clCreateBuffer(resident out)");
    const cl_bool upload_blocking =
        nonblocking_uploads ? kClFalse : kClTrue;
    Check(api_.clEnqueueWriteBuffer(queue_, *q8_qs_buffer, upload_blocking, 0, q8_qs.size(),
                                    q8_qs.data(), 0, nullptr, nullptr),
          "clEnqueueWriteBuffer(resident q8_qs)");
    Check(api_.clEnqueueWriteBuffer(queue_, *q8_bsums_buffer, upload_blocking, 0,
                                    q8_bsums.size() * sizeof(std::int16_t),
                                    q8_bsums.data(), 0, nullptr, nullptr),
          "clEnqueueWriteBuffer(resident q8_bsums)");
    Check(api_.clEnqueueWriteBuffer(queue_, *q8_d_buffer, upload_blocking, 0,
                                    q8_d.size() * sizeof(float), q8_d.data(),
                                    0, nullptr, nullptr),
          "clEnqueueWriteBuffer(resident q8_d)");
  }

  void CreateQ6RunBuffers(const GpuQ8KInputPlanes& q8,
                          const std::vector<float>& output,
                          cl_mem* q8_qs_buffer,
                          cl_mem* q8_d_buffer,
                          cl_mem* out_buffer) {
    cl_int err = kClSuccess;
    *q8_qs_buffer =
        api_.clCreateBuffer(context_, kClMemReadOnly, q8.qs.size(), nullptr, &err);
    Check(err, "clCreateBuffer(resident Q6 q8_qs)");
    *q8_d_buffer = api_.clCreateBuffer(
        context_, kClMemReadOnly, q8.d.size() * sizeof(float), nullptr, &err);
    Check(err, "clCreateBuffer(resident Q6 q8_d)");
    *out_buffer =
        api_.clCreateBuffer(context_, kClMemWriteOnly,
                            output.size() * sizeof(float), nullptr, &err);
    Check(err, "clCreateBuffer(resident Q6 out)");
    Check(api_.clEnqueueWriteBuffer(queue_, *q8_qs_buffer, kClTrue, 0,
                                    q8.qs.size(), q8.qs.data(), 0, nullptr, nullptr),
          "clEnqueueWriteBuffer(resident Q6 q8_qs)");
    Check(api_.clEnqueueWriteBuffer(queue_, *q8_d_buffer, kClTrue, 0,
                                    q8.d.size() * sizeof(float), q8.d.data(),
                                    0, nullptr, nullptr),
          "clEnqueueWriteBuffer(resident Q6 q8_d)");
  }

  const ResidentPackedQ4X8& ResidentPackedQ4X8ForHandle(std::uint64_t handle) const {
    const auto it = resident_packed_q4x8_.find(handle);
    Require(it != resident_packed_q4x8_.end(), "resident Q4 x8 handle not found");
    return it->second;
  }

  const ResidentRawQ6K& ResidentRawQ6KForHandle(std::uint64_t handle) const {
    const auto it = resident_raw_q6k_.find(handle);
    Require(it != resident_raw_q6k_.end(), "resident Q6_K handle not found");
    return it->second;
  }

  const ResidentRawQ4KCpuOrder& ResidentRawQ4KCpuOrderForHandle(
      std::uint64_t handle) const {
    const auto it = resident_raw_q4_cpu_order_.find(handle);
    Require(it != resident_raw_q4_cpu_order_.end(),
            "resident Q4 CPU-order handle not found");
    return it->second;
  }

  const ResidentConvWeights& ResidentConvWeightsForHandle(
      std::uint64_t handle) const {
    const auto it = resident_conv_weights_.find(handle);
    Require(it != resident_conv_weights_.end(), "resident conv weights handle not found");
    return it->second;
  }

  const ResidentF32Matvec& ResidentF32MatvecForHandle(
      std::uint64_t handle) const {
    const auto it = resident_f32_matvec_.find(handle);
    Require(it != resident_f32_matvec_.end(), "resident F32 matvec handle not found");
    return it->second;
  }

  const ResidentF32Buffer& ResidentF32BufferForHandle(
      std::uint64_t handle) const {
    const auto it = resident_f32_buffers_.find(handle);
    Require(it != resident_f32_buffers_.end(), "resident F32 buffer handle not found");
    return it->second;
  }

  void CreateF32MatvecBuffers(const std::vector<float>& weights,
                              const std::vector<float>& input,
                              const std::vector<float>& output,
                              cl_mem* weights_buffer,
                              cl_mem* input_buffer,
                              cl_mem* out_buffer) {
    cl_int err = kClSuccess;
    *weights_buffer = api_.clCreateBuffer(context_, kClMemReadOnly, weights.size() * sizeof(float), nullptr, &err);
    Check(err, "clCreateBuffer(f32 weights)");
    *input_buffer = api_.clCreateBuffer(context_, kClMemReadOnly, input.size() * sizeof(float), nullptr, &err);
    Check(err, "clCreateBuffer(f32 input)");
    *out_buffer = api_.clCreateBuffer(context_, kClMemWriteOnly, output.size() * sizeof(float), nullptr, &err);
    Check(err, "clCreateBuffer(f32 out)");
    Check(api_.clEnqueueWriteBuffer(queue_, *weights_buffer, kClTrue, 0,
                                    weights.size() * sizeof(float), weights.data(), 0, nullptr, nullptr),
          "clEnqueueWriteBuffer(f32 weights)");
    Check(api_.clEnqueueWriteBuffer(queue_, *input_buffer, kClTrue, 0,
                                    input.size() * sizeof(float), input.data(), 0, nullptr, nullptr),
          "clEnqueueWriteBuffer(f32 input)");
  }

  void CreateF32MatvecBuffersWithoutWeights(const std::vector<float>& input,
                                            const std::vector<float>& output,
                                            cl_mem* input_buffer,
                                            cl_mem* out_buffer) {
    cl_int err = kClSuccess;
    *input_buffer = api_.clCreateBuffer(
        context_, kClMemReadOnly, input.size() * sizeof(float), nullptr, &err);
    Check(err, "clCreateBuffer(resident F32 input)");
    *out_buffer = api_.clCreateBuffer(
        context_, kClMemWriteOnly, output.size() * sizeof(float), nullptr, &err);
    Check(err, "clCreateBuffer(resident F32 out)");
    Check(api_.clEnqueueWriteBuffer(queue_, *input_buffer, kClTrue, 0,
                                    input.size() * sizeof(float), input.data(),
                                    0, nullptr, nullptr),
          "clEnqueueWriteBuffer(resident F32 input)");
  }

  void CreateConvBuffers(const std::vector<float>& conv_weights,
                         const std::vector<float>& conv_state,
                         const std::vector<float>& conv_output,
                         const std::vector<float>& next_state,
                         cl_mem* conv_weights_buffer,
                         cl_mem* conv_state_buffer,
                         cl_mem* conv_output_buffer,
                         cl_mem* next_state_buffer) {
    cl_int err = kClSuccess;
    *conv_weights_buffer = api_.clCreateBuffer(context_, kClMemReadOnly,
                                               conv_weights.size() * sizeof(float),
                                               nullptr, &err);
    Check(err, "clCreateBuffer(conv_weights)");
    *conv_state_buffer = api_.clCreateBuffer(context_, kClMemReadOnly,
                                             conv_state.size() * sizeof(float),
                                             nullptr, &err);
    Check(err, "clCreateBuffer(conv_state)");
    *conv_output_buffer = api_.clCreateBuffer(context_, kClMemWriteOnly,
                                              conv_output.size() * sizeof(float),
                                              nullptr, &err);
    Check(err, "clCreateBuffer(conv_output)");
    *next_state_buffer = api_.clCreateBuffer(context_, kClMemWriteOnly,
                                             next_state.size() * sizeof(float),
                                             nullptr, &err);
    Check(err, "clCreateBuffer(next_state)");
    Check(api_.clEnqueueWriteBuffer(queue_, *conv_weights_buffer, kClTrue, 0,
                                    conv_weights.size() * sizeof(float),
                                    conv_weights.data(), 0, nullptr, nullptr),
          "clEnqueueWriteBuffer(conv_weights)");
    Check(api_.clEnqueueWriteBuffer(queue_, *conv_state_buffer, kClTrue, 0,
                                    conv_state.size() * sizeof(float),
                                    conv_state.data(), 0, nullptr, nullptr),
          "clEnqueueWriteBuffer(conv_state)");
  }

  void CreateConvBuffersWithoutWeights(const std::vector<float>& conv_state,
                                       const std::vector<float>& conv_output,
                                       const std::vector<float>& next_state,
                                       cl_mem* conv_state_buffer,
                                       cl_mem* conv_output_buffer,
                                       cl_mem* next_state_buffer) {
    cl_int err = kClSuccess;
    *conv_state_buffer = api_.clCreateBuffer(
        context_, kClMemReadOnly, conv_state.size() * sizeof(float), nullptr, &err);
    Check(err, "clCreateBuffer(resident conv_state)");
    *conv_output_buffer = api_.clCreateBuffer(
        context_, kClMemWriteOnly, conv_output.size() * sizeof(float), nullptr, &err);
    Check(err, "clCreateBuffer(resident conv_output)");
    *next_state_buffer = api_.clCreateBuffer(
        context_, kClMemWriteOnly, next_state.size() * sizeof(float), nullptr, &err);
    Check(err, "clCreateBuffer(resident next_state)");
    Check(api_.clEnqueueWriteBuffer(queue_, *conv_state_buffer, kClTrue, 0,
                                    conv_state.size() * sizeof(float),
                                    conv_state.data(), 0, nullptr, nullptr),
          "clEnqueueWriteBuffer(resident conv_state)");
  }

  void EnsurePostConvPrepBuffers(const std::vector<float>& conv_output_raw,
                                 const GpuLinearAttentionPostConvPrepRun& run,
                                 std::uint64_t q_values,
                                 cl_mem* raw_buffer,
                                 cl_mem* silu_buffer,
                                 cl_mem* q_buffer,
                                 cl_mem* k_buffer,
                                 cl_mem* v_buffer,
                                 cl_mem* q_norm_buffer,
                                 cl_mem* k_norm_buffer) {
    *raw_buffer = EnsureScratchBuffer(
        postconv_prep_scratch_raw_, conv_output_raw.size() * sizeof(float),
        kClMemReadOnly, "postconv raw");
    *silu_buffer = EnsureScratchBuffer(
        postconv_prep_scratch_silu_, conv_output_raw.size() * sizeof(float),
        kClMemWriteOnly, "postconv silu");
    *q_buffer = EnsureScratchBuffer(
        postconv_prep_scratch_q_, q_values * sizeof(float),
        kClMemReadWrite, "postconv q");
    *k_buffer = EnsureScratchBuffer(
        postconv_prep_scratch_k_, q_values * sizeof(float),
        kClMemReadWrite, "postconv k");
    *v_buffer = EnsureScratchBuffer(
        postconv_prep_scratch_v_, run.v_conv_predelta.size() * sizeof(float),
        kClMemWriteOnly, "postconv v");
    *q_norm_buffer = EnsureScratchBuffer(
        postconv_prep_scratch_q_norm_,
        run.q_conv_predelta.size() * sizeof(float),
        kClMemWriteOnly, "postconv q norm");
    *k_norm_buffer = EnsureScratchBuffer(
        postconv_prep_scratch_k_norm_,
        run.k_conv_predelta.size() * sizeof(float),
        kClMemWriteOnly, "postconv k norm");
    Check(api_.clEnqueueWriteBuffer(queue_, *raw_buffer, kClFalse, 0,
                                    conv_output_raw.size() * sizeof(float),
                                    conv_output_raw.data(), 0, nullptr, nullptr),
          "clEnqueueWriteBuffer(postconv raw)");
  }

  void ReadPostConvPrepBuffers(GpuLinearAttentionPostConvPrepRun& run,
                               cl_mem silu_buffer,
                               cl_mem q_buffer,
                               cl_mem k_buffer,
                               cl_mem v_buffer,
                               cl_mem q_norm_buffer,
                               cl_mem k_norm_buffer,
                               bool readback_intermediates) {
    if (readback_intermediates) {
      Check(api_.clEnqueueReadBuffer(queue_, silu_buffer, kClTrue, 0,
                                     run.conv_output_silu.size() * sizeof(float),
                                     run.conv_output_silu.data(), 0, nullptr, nullptr),
            "clEnqueueReadBuffer(postconv silu)");
      Check(api_.clEnqueueReadBuffer(queue_, q_buffer, kClTrue, 0,
                                     run.q_conv.size() * sizeof(float),
                                     run.q_conv.data(), 0, nullptr, nullptr),
            "clEnqueueReadBuffer(postconv q)");
      Check(api_.clEnqueueReadBuffer(queue_, k_buffer, kClTrue, 0,
                                     run.k_conv.size() * sizeof(float),
                                     run.k_conv.data(), 0, nullptr, nullptr),
            "clEnqueueReadBuffer(postconv k)");
    }
    Check(api_.clEnqueueReadBuffer(queue_, v_buffer, kClTrue, 0,
                                   run.v_conv_predelta.size() * sizeof(float),
                                   run.v_conv_predelta.data(), 0, nullptr, nullptr),
          "clEnqueueReadBuffer(postconv v)");
    Check(api_.clEnqueueReadBuffer(queue_, q_norm_buffer, kClTrue, 0,
                                   run.q_conv_predelta.size() * sizeof(float),
                                   run.q_conv_predelta.data(), 0, nullptr, nullptr),
          "clEnqueueReadBuffer(postconv q norm)");
    Check(api_.clEnqueueReadBuffer(queue_, k_norm_buffer, kClTrue, 0,
                                   run.k_conv_predelta.size() * sizeof(float),
                                   run.k_conv_predelta.data(), 0, nullptr, nullptr),
          "clEnqueueReadBuffer(postconv k norm)");
  }

  void CreateLinearAttentionDeltaBuffers(
      const std::vector<float>& q,
      const std::vector<float>& k,
      const std::vector<float>& v,
      const std::vector<float>& gate,
      const std::vector<float>& beta,
      const std::vector<float>& recurrent_state,
      const std::vector<float>& z,
      const std::vector<float>& norm_weight,
      const GpuLinearAttentionDeltaRun& run,
      cl_mem* q_buffer,
      cl_mem* k_buffer,
      cl_mem* v_buffer,
      cl_mem* gate_buffer,
      cl_mem* beta_buffer,
      cl_mem* state_buffer,
      cl_mem* z_buffer,
      cl_mem* norm_buffer,
      cl_mem* attention_buffer,
      cl_mem* next_state_buffer,
      cl_mem* final_buffer) {
    cl_int err = kClSuccess;
    *q_buffer = api_.clCreateBuffer(context_, kClMemReadOnly,
                                    q.size() * sizeof(float), nullptr, &err);
    Check(err, "clCreateBuffer(delta q)");
    *k_buffer = api_.clCreateBuffer(context_, kClMemReadOnly,
                                    k.size() * sizeof(float), nullptr, &err);
    Check(err, "clCreateBuffer(delta k)");
    *v_buffer = api_.clCreateBuffer(context_, kClMemReadOnly,
                                    v.size() * sizeof(float), nullptr, &err);
    Check(err, "clCreateBuffer(delta v)");
    *gate_buffer = api_.clCreateBuffer(context_, kClMemReadOnly,
                                       gate.size() * sizeof(float), nullptr, &err);
    Check(err, "clCreateBuffer(delta gate)");
    *beta_buffer = api_.clCreateBuffer(context_, kClMemReadOnly,
                                       beta.size() * sizeof(float), nullptr, &err);
    Check(err, "clCreateBuffer(delta beta)");
    *state_buffer = api_.clCreateBuffer(context_, kClMemReadOnly,
                                        recurrent_state.size() * sizeof(float),
                                        nullptr, &err);
    Check(err, "clCreateBuffer(delta state)");
    *z_buffer = api_.clCreateBuffer(context_, kClMemReadOnly,
                                    z.size() * sizeof(float), nullptr, &err);
    Check(err, "clCreateBuffer(delta z)");
    *norm_buffer = api_.clCreateBuffer(context_, kClMemReadOnly,
                                       norm_weight.size() * sizeof(float),
                                       nullptr, &err);
    Check(err, "clCreateBuffer(delta norm)");
    *attention_buffer = api_.clCreateBuffer(
        context_, kClMemWriteOnly, run.attention_output.size() * sizeof(float),
        nullptr, &err);
    Check(err, "clCreateBuffer(delta attention)");
    *next_state_buffer = api_.clCreateBuffer(
        context_, kClMemWriteOnly, run.recurrent_state.size() * sizeof(float),
        nullptr, &err);
    Check(err, "clCreateBuffer(delta next state)");
    *final_buffer = api_.clCreateBuffer(
        context_, kClMemWriteOnly, run.final_output.size() * sizeof(float),
        nullptr, &err);
    Check(err, "clCreateBuffer(delta final)");

    Check(api_.clEnqueueWriteBuffer(queue_, *q_buffer, kClTrue, 0,
                                    q.size() * sizeof(float), q.data(),
                                    0, nullptr, nullptr),
          "clEnqueueWriteBuffer(delta q)");
    Check(api_.clEnqueueWriteBuffer(queue_, *k_buffer, kClTrue, 0,
                                    k.size() * sizeof(float), k.data(),
                                    0, nullptr, nullptr),
          "clEnqueueWriteBuffer(delta k)");
    Check(api_.clEnqueueWriteBuffer(queue_, *v_buffer, kClTrue, 0,
                                    v.size() * sizeof(float), v.data(),
                                    0, nullptr, nullptr),
          "clEnqueueWriteBuffer(delta v)");
    Check(api_.clEnqueueWriteBuffer(queue_, *gate_buffer, kClTrue, 0,
                                    gate.size() * sizeof(float), gate.data(),
                                    0, nullptr, nullptr),
          "clEnqueueWriteBuffer(delta gate)");
    Check(api_.clEnqueueWriteBuffer(queue_, *beta_buffer, kClTrue, 0,
                                    beta.size() * sizeof(float), beta.data(),
                                    0, nullptr, nullptr),
          "clEnqueueWriteBuffer(delta beta)");
    Check(api_.clEnqueueWriteBuffer(queue_, *state_buffer, kClTrue, 0,
                                    recurrent_state.size() * sizeof(float),
                                    recurrent_state.data(), 0, nullptr,
                                    nullptr),
          "clEnqueueWriteBuffer(delta state)");
    Check(api_.clEnqueueWriteBuffer(queue_, *z_buffer, kClTrue, 0,
                                    z.size() * sizeof(float), z.data(),
                                    0, nullptr, nullptr),
          "clEnqueueWriteBuffer(delta z)");
    Check(api_.clEnqueueWriteBuffer(queue_, *norm_buffer, kClTrue, 0,
                                    norm_weight.size() * sizeof(float),
                                    norm_weight.data(), 0, nullptr, nullptr),
          "clEnqueueWriteBuffer(delta norm)");
  }

  void ReadLinearAttentionDeltaBuffers(GpuLinearAttentionDeltaRun& run,
                                       cl_mem attention_buffer,
                                       cl_mem next_state_buffer,
                                       cl_mem final_buffer) {
    Check(api_.clEnqueueReadBuffer(queue_, attention_buffer, kClTrue, 0,
                                   run.attention_output.size() * sizeof(float),
                                   run.attention_output.data(), 0, nullptr,
                                   nullptr),
          "clEnqueueReadBuffer(delta attention)");
    Check(api_.clEnqueueReadBuffer(queue_, next_state_buffer, kClTrue, 0,
                                   run.recurrent_state.size() * sizeof(float),
                                   run.recurrent_state.data(), 0, nullptr,
                                   nullptr),
          "clEnqueueReadBuffer(delta state)");
    Check(api_.clEnqueueReadBuffer(queue_, final_buffer, kClTrue, 0,
                                   run.final_output.size() * sizeof(float),
                                   run.final_output.data(), 0, nullptr, nullptr),
          "clEnqueueReadBuffer(delta final)");
  }

  GpuF32MatvecTiming RunF32MatvecKernel(cl_mem weights_buffer,
                                        cl_mem input_buffer,
                                        cl_mem out_buffer,
                                        std::uint64_t rows, std::uint64_t cols,
                                        int repeat) {
    const cl_uint cols_arg = static_cast<cl_uint>(cols);
    const cl_uint rows_arg = static_cast<cl_uint>(rows);
    Check(api_.clSetKernelArg(kernel_f32_matvec_, 0, sizeof(weights_buffer), &weights_buffer), "clSetKernelArg(f32 matvec 0)");
    Check(api_.clSetKernelArg(kernel_f32_matvec_, 1, sizeof(input_buffer), &input_buffer), "clSetKernelArg(f32 matvec 1)");
    Check(api_.clSetKernelArg(kernel_f32_matvec_, 2, sizeof(cols_arg), &cols_arg), "clSetKernelArg(f32 matvec 2)");
    Check(api_.clSetKernelArg(kernel_f32_matvec_, 3, sizeof(rows_arg), &rows_arg), "clSetKernelArg(f32 matvec 3)");
    Check(api_.clSetKernelArg(kernel_f32_matvec_, 4, sizeof(out_buffer), &out_buffer), "clSetKernelArg(f32 matvec 4)");

    const std::size_t global = static_cast<std::size_t>(rows);
    std::vector<double> times;
    times.reserve(static_cast<std::size_t>(repeat));
    for (int i = 0; i < repeat; ++i) {
      cl_event event = nullptr;
      Check(api_.clEnqueueNDRangeKernel(queue_, kernel_f32_matvec_, 1, nullptr, &global, nullptr, 0, nullptr, EventOut(&event)),
            "clEnqueueNDRangeKernel(f32_matvec_row_f32)");
      Check(api_.clFinish(queue_), "clFinish(f32 matvec)");
      times.push_back(EventUs(api_, event));
      ReleaseEvent(api_, &event);
    }

    GpuF32MatvecTiming timing;
    timing.min_us = *std::min_element(times.begin(), times.end());
    timing.mean_us = std::accumulate(times.begin(), times.end(), 0.0) /
          static_cast<double>(times.size());
    timing.effective_weight_gb_s =
        static_cast<double>(rows * cols * sizeof(float)) / (timing.min_us / 1e6) / 1e9;
    timing.global_work_items = rows;
    return timing;
  }

  GpuQ4X8MatvecTiming RunKernel(GpuQ4X8KernelVariant variant,
                                 cl_mem packed_buffer,
                                 cl_mem q8_qs_buffer,
                                 cl_mem q8_bsums_buffer,
                                 cl_mem q8_d_buffer,
                                 cl_mem out_buffer,
                                 std::uint64_t blocks_per_row,
                                 std::uint64_t row_groups,
                                 std::uint64_t global_work_items,
                                 int repeat,
                                 float* read_output = nullptr,
                                 std::size_t read_output_bytes = 0) {
    const bool submit_split_profile =
        AttentionFrontHandoffMatvecSubmitSplitProfile();
    const auto kernel_setup_begin =
        submit_split_profile ? std::chrono::steady_clock::now()
                             : std::chrono::steady_clock::time_point{};
    const cl_uint blocks_arg = static_cast<cl_uint>(blocks_per_row);
    const cl_uint row_groups_arg = static_cast<cl_uint>(row_groups);
    cl_kernel kernel = Kernel(variant);
    Check(api_.clSetKernelArg(kernel, 0, sizeof(packed_buffer), &packed_buffer), "clSetKernelArg(0)");
    Check(api_.clSetKernelArg(kernel, 1, sizeof(q8_qs_buffer), &q8_qs_buffer), "clSetKernelArg(1)");
    Check(api_.clSetKernelArg(kernel, 2, sizeof(q8_bsums_buffer), &q8_bsums_buffer), "clSetKernelArg(2)");
    Check(api_.clSetKernelArg(kernel, 3, sizeof(q8_d_buffer), &q8_d_buffer), "clSetKernelArg(3)");
    Check(api_.clSetKernelArg(kernel, 4, sizeof(blocks_arg), &blocks_arg), "clSetKernelArg(4)");
    Check(api_.clSetKernelArg(kernel, 5, sizeof(row_groups_arg), &row_groups_arg), "clSetKernelArg(5)");
    Check(api_.clSetKernelArg(kernel, 6, sizeof(out_buffer), &out_buffer), "clSetKernelArg(6)");

    const std::size_t global = static_cast<std::size_t>(global_work_items);
    constexpr std::size_t kSmallQ4RowlaneLocalSize = 64;
    const bool small_q4_rowlane_local64 =
        variant == GpuQ4X8KernelVariant::kRowlaneParallel &&
        blocks_per_row == 16 && global == 2048;
    const std::size_t* local =
        small_q4_rowlane_local64 ? &kSmallQ4RowlaneLocalSize : nullptr;
    const char* kernel_name = KernelFunctionName(variant);
    std::uint64_t kernel_setup_wall_ns =
        submit_split_profile
            ? WallNs(kernel_setup_begin, std::chrono::steady_clock::now())
            : 0;
    std::vector<double> times;
    times.reserve(static_cast<std::size_t>(repeat));
    std::uint64_t kernel_wait_wall_ns = 0;
    std::uint64_t kernel_enqueue_wall_ns = 0;
    std::uint64_t kernel_finish_wall_ns = 0;
    std::uint64_t event_profile_wall_ns = 0;
    std::uint64_t output_read_wall_ns = 0;
    for (int i = 0; i < repeat; ++i) {
      cl_event event = nullptr;
      const auto kernel_wait_begin =
          submit_split_profile ? std::chrono::steady_clock::now()
                               : std::chrono::steady_clock::time_point{};
      const auto enqueue_begin =
          submit_split_profile ? std::chrono::steady_clock::now()
                               : std::chrono::steady_clock::time_point{};
      Check(api_.clEnqueueNDRangeKernel(queue_, kernel, 1, nullptr, &global,
                                        local, 0, nullptr, EventOut(&event)),
            std::string("clEnqueueNDRangeKernel(") + kernel_name + ")");
      if (submit_split_profile) {
        kernel_enqueue_wall_ns +=
            WallNs(enqueue_begin, std::chrono::steady_clock::now());
      }
      if (read_output != nullptr) {
        Require(read_output_bytes > 0, "resident packed Q4 read output size is zero");
        const auto output_read_begin =
            submit_split_profile ? std::chrono::steady_clock::now()
                                 : std::chrono::steady_clock::time_point{};
        Check(api_.clEnqueueReadBuffer(queue_, out_buffer, kClTrue, 0,
                                       read_output_bytes, read_output, 0,
                                       nullptr, nullptr),
              "clEnqueueReadBuffer(resident packed Q4 out)");
        if (submit_split_profile) {
          output_read_wall_ns +=
              WallNs(output_read_begin, std::chrono::steady_clock::now());
        }
      } else {
        const auto finish_begin =
            submit_split_profile ? std::chrono::steady_clock::now()
                                 : std::chrono::steady_clock::time_point{};
        Check(api_.clFinish(queue_), "clFinish(kernel)");
        if (submit_split_profile) {
          kernel_finish_wall_ns +=
              WallNs(finish_begin, std::chrono::steady_clock::now());
        }
      }
      if (submit_split_profile) {
        kernel_wait_wall_ns +=
            WallNs(kernel_wait_begin, std::chrono::steady_clock::now());
      }
      const auto event_profile_begin =
          submit_split_profile ? std::chrono::steady_clock::now()
                               : std::chrono::steady_clock::time_point{};
      times.push_back(EventUs(api_, event));
      if (submit_split_profile) {
        event_profile_wall_ns +=
            WallNs(event_profile_begin, std::chrono::steady_clock::now());
      }
      ReleaseEvent(api_, &event);
    }
    const auto queue_drain_cleanup_begin =
        submit_split_profile ? std::chrono::steady_clock::now()
                             : std::chrono::steady_clock::time_point{};
    ClearPendingHostUploadsAfterQueueDrain();
    const std::uint64_t queue_drain_cleanup_wall_ns =
        submit_split_profile
            ? WallNs(queue_drain_cleanup_begin, std::chrono::steady_clock::now())
            : 0;

    GpuQ4X8MatvecTiming timing;
    timing.min_us = *std::min_element(times.begin(), times.end());
    timing.mean_us = std::accumulate(times.begin(), times.end(), 0.0) /
          static_cast<double>(times.size());
    timing.effective_packed_gb_s =
        static_cast<double>(global_work_items * RowsPerWorkItem(variant) / kRowsInterleaved *
                            blocks_per_row * kQ4Kx8BlockBytes) /
        (timing.min_us / 1e6) / 1e9;
    timing.global_work_items = global_work_items;
    timing.rows_per_work_item = RowsPerWorkItem(variant);
    timing.kernel_setup_wall_ns = kernel_setup_wall_ns;
    timing.kernel_wait_wall_ns = kernel_wait_wall_ns;
    timing.kernel_enqueue_wall_ns = kernel_enqueue_wall_ns;
    timing.kernel_finish_wall_ns = kernel_finish_wall_ns;
    timing.event_profile_wall_ns = event_profile_wall_ns;
    timing.queue_drain_cleanup_wall_ns = queue_drain_cleanup_wall_ns;
    timing.output_read_wall_ns = output_read_wall_ns;
    return timing;
  }

  GpuQ4X8MatvecTiming RunRowblock16Kernel(cl_mem packed_buffer,
                                          cl_mem q8_qs_buffer,
                                          cl_mem q8_bsums_buffer,
                                          cl_mem q8_d_buffer,
                                          cl_mem out_buffer,
                                          std::uint64_t rows,
                                          std::uint64_t blocks_per_row,
                                          std::uint64_t row_groups,
                                          int repeat,
                                          bool cpu_order_finalize = false) {
    const bool submit_split_profile =
        AttentionFrontHandoffMatvecSubmitSplitProfile();
    const auto kernel_setup_begin =
        submit_split_profile ? std::chrono::steady_clock::now()
                             : std::chrono::steady_clock::time_point{};
    Require(blocks_per_row == 16,
            "rowblock16 Q4 matvec requires blocks_per_row == 16");
    Require(rows == row_groups * kRowsInterleaved,
            "rowblock16 Q4 matvec row count mismatch");
    const cl_uint blocks_arg = static_cast<cl_uint>(blocks_per_row);
    const cl_uint row_groups_arg = static_cast<cl_uint>(row_groups);
    cl_kernel kernel = cpu_order_finalize
                           ? kernel_rowblock16_cpuorder_finalize_
                           : kernel_rowblock16_;
    Check(api_.clSetKernelArg(kernel, 0, sizeof(packed_buffer),
                              &packed_buffer),
          "clSetKernelArg(rowblock16 0)");
    Check(api_.clSetKernelArg(kernel, 1, sizeof(q8_qs_buffer),
                              &q8_qs_buffer),
          "clSetKernelArg(rowblock16 1)");
    Check(api_.clSetKernelArg(kernel, 2, sizeof(q8_bsums_buffer),
                              &q8_bsums_buffer),
          "clSetKernelArg(rowblock16 2)");
    Check(api_.clSetKernelArg(kernel, 3, sizeof(q8_d_buffer),
                              &q8_d_buffer),
          "clSetKernelArg(rowblock16 3)");
    Check(api_.clSetKernelArg(kernel, 4, sizeof(blocks_arg),
                              &blocks_arg),
          "clSetKernelArg(rowblock16 4)");
    Check(api_.clSetKernelArg(kernel, 5, sizeof(row_groups_arg),
                              &row_groups_arg),
          "clSetKernelArg(rowblock16 5)");
    Check(api_.clSetKernelArg(kernel, 6, sizeof(out_buffer),
                              &out_buffer),
          "clSetKernelArg(rowblock16 6)");

    constexpr std::size_t kRowblock16LocalSize = 16;
    const std::size_t local = kRowblock16LocalSize;
    const std::size_t global = static_cast<std::size_t>(rows) * local;
    std::vector<double> times;
    times.reserve(static_cast<std::size_t>(repeat));
    const std::uint64_t kernel_setup_wall_ns =
        submit_split_profile
            ? WallNs(kernel_setup_begin, std::chrono::steady_clock::now())
            : 0;
    std::uint64_t kernel_wait_wall_ns = 0;
    std::uint64_t kernel_enqueue_wall_ns = 0;
    std::uint64_t kernel_finish_wall_ns = 0;
    std::uint64_t event_profile_wall_ns = 0;
    for (int i = 0; i < repeat; ++i) {
      cl_event event = nullptr;
      const auto kernel_wait_begin =
          submit_split_profile ? std::chrono::steady_clock::now()
                               : std::chrono::steady_clock::time_point{};
      const auto enqueue_begin =
          submit_split_profile ? std::chrono::steady_clock::now()
                               : std::chrono::steady_clock::time_point{};
      Check(api_.clEnqueueNDRangeKernel(queue_, kernel, 1, nullptr,
                                        &global, &local, 0, nullptr,
                                        EventOut(&event)),
            cpu_order_finalize
                ? "clEnqueueNDRangeKernel(rowblock16 CPU-order finalize)"
                : "clEnqueueNDRangeKernel(q4k_x8_matvec_rowblock16_reduce)");
      if (submit_split_profile) {
        kernel_enqueue_wall_ns +=
            WallNs(enqueue_begin, std::chrono::steady_clock::now());
      }
      const auto finish_begin =
          submit_split_profile ? std::chrono::steady_clock::now()
                               : std::chrono::steady_clock::time_point{};
      Check(api_.clFinish(queue_), "clFinish(rowblock16 Q4 matvec)");
      if (submit_split_profile) {
        kernel_finish_wall_ns +=
            WallNs(finish_begin, std::chrono::steady_clock::now());
        kernel_wait_wall_ns +=
            WallNs(kernel_wait_begin, std::chrono::steady_clock::now());
      }
      const auto event_profile_begin =
          submit_split_profile ? std::chrono::steady_clock::now()
                               : std::chrono::steady_clock::time_point{};
      times.push_back(EventUs(api_, event));
      if (submit_split_profile) {
        event_profile_wall_ns +=
            WallNs(event_profile_begin, std::chrono::steady_clock::now());
      }
      ReleaseEvent(api_, &event);
    }
    const auto queue_drain_cleanup_begin =
        submit_split_profile ? std::chrono::steady_clock::now()
                             : std::chrono::steady_clock::time_point{};
    ClearPendingHostUploadsAfterQueueDrain();
    const std::uint64_t queue_drain_cleanup_wall_ns =
        submit_split_profile
            ? WallNs(queue_drain_cleanup_begin, std::chrono::steady_clock::now())
            : 0;

    GpuQ4X8MatvecTiming timing;
    timing.min_us = *std::min_element(times.begin(), times.end());
    timing.mean_us =
        std::accumulate(times.begin(), times.end(), 0.0) /
        static_cast<double>(times.size());
    timing.effective_packed_gb_s =
        static_cast<double>(row_groups * blocks_per_row * kQ4Kx8BlockBytes) /
        (timing.min_us / 1e6) / 1e9;
    timing.global_work_items = global;
    timing.rows_per_work_item = 0;
    timing.kernel_setup_wall_ns = kernel_setup_wall_ns;
    timing.kernel_wait_wall_ns = kernel_wait_wall_ns;
    timing.kernel_enqueue_wall_ns = kernel_enqueue_wall_ns;
    timing.kernel_finish_wall_ns = kernel_finish_wall_ns;
    timing.event_profile_wall_ns = event_profile_wall_ns;
    timing.queue_drain_cleanup_wall_ns = queue_drain_cleanup_wall_ns;
    return timing;
  }

  GpuQ4X8MatvecTiming RunExpert8Kernel(
      const std::array<cl_mem, 8>& packed_buffers,
      cl_mem q8_qs_buffer,
      cl_mem q8_bsums_buffer,
      cl_mem q8_d_buffer,
      cl_mem out_buffer,
      std::uint64_t blocks_per_row,
      std::uint64_t row_groups_per_expert,
      std::uint64_t global_work_items,
      int repeat) {
    const cl_uint blocks_arg = static_cast<cl_uint>(blocks_per_row);
    const cl_uint row_groups_arg =
        static_cast<cl_uint>(row_groups_per_expert);
    for (std::size_t i = 0; i < packed_buffers.size(); ++i) {
      Check(api_.clSetKernelArg(kernel_rowlane_expert8_,
                                static_cast<cl_uint>(i),
                                sizeof(packed_buffers[i]),
                                &packed_buffers[i]),
            "clSetKernelArg(expert8 packed)");
    }
    Check(api_.clSetKernelArg(kernel_rowlane_expert8_, 8,
                              sizeof(q8_qs_buffer), &q8_qs_buffer),
          "clSetKernelArg(expert8 q8_qs)");
    Check(api_.clSetKernelArg(kernel_rowlane_expert8_, 9,
                              sizeof(q8_bsums_buffer), &q8_bsums_buffer),
          "clSetKernelArg(expert8 q8_bsums)");
    Check(api_.clSetKernelArg(kernel_rowlane_expert8_, 10,
                              sizeof(q8_d_buffer), &q8_d_buffer),
          "clSetKernelArg(expert8 q8_d)");
    Check(api_.clSetKernelArg(kernel_rowlane_expert8_, 11,
                              sizeof(blocks_arg), &blocks_arg),
          "clSetKernelArg(expert8 blocks)");
    Check(api_.clSetKernelArg(kernel_rowlane_expert8_, 12,
                              sizeof(row_groups_arg), &row_groups_arg),
          "clSetKernelArg(expert8 row_groups)");
    Check(api_.clSetKernelArg(kernel_rowlane_expert8_, 13,
                              sizeof(out_buffer), &out_buffer),
          "clSetKernelArg(expert8 out)");

    const std::size_t global = static_cast<std::size_t>(global_work_items);
    std::vector<double> times;
    times.reserve(static_cast<std::size_t>(repeat));
    for (int i = 0; i < repeat; ++i) {
      cl_event event = nullptr;
      Check(api_.clEnqueueNDRangeKernel(
                queue_, kernel_rowlane_expert8_, 1, nullptr, &global, nullptr,
                0, nullptr, EventOut(&event)),
            "clEnqueueNDRangeKernel(q4k_x8_matvec_rowlane_expert8)");
      Check(api_.clFinish(queue_), "clFinish(expert8 kernel)");
      times.push_back(EventUs(api_, event));
      ReleaseEvent(api_, &event);
    }
    ClearPendingHostUploadsAfterQueueDrain();

    GpuQ4X8MatvecTiming timing;
    timing.min_us = *std::min_element(times.begin(), times.end());
    timing.mean_us =
        std::accumulate(times.begin(), times.end(), 0.0) /
          static_cast<double>(times.size());
    timing.effective_packed_gb_s =
        static_cast<double>(global_work_items / kRowsInterleaved *
                            blocks_per_row * kQ4Kx8BlockBytes) /
        (timing.min_us / 1e6) / 1e9;
    timing.global_work_items = global_work_items;
    timing.rows_per_work_item = 1;
    return timing;
  }

  GpuQ4X8MatvecTiming RunExpert8MultiQ8Kernel(
      const std::array<cl_mem, 8>& packed_buffers,
      cl_mem q8_qs_buffer,
      cl_mem q8_bsums_buffer,
      cl_mem q8_d_buffer,
      cl_mem out_buffer,
      std::uint64_t blocks_per_row,
      std::uint64_t row_groups_per_expert,
      std::uint64_t global_work_items,
      int repeat,
      float* read_output = nullptr,
      std::size_t read_output_bytes = 0,
      bool defer_finish = false) {
    cl_kernel kernel = kernel_rowlane_expert8_multiq8_;
    Require(kernel != nullptr, "expert8 multiq8 Q4 kernel missing");
    const cl_uint blocks_arg = static_cast<cl_uint>(blocks_per_row);
    const cl_uint row_groups_arg =
        static_cast<cl_uint>(row_groups_per_expert);
    for (std::size_t i = 0; i < packed_buffers.size(); ++i) {
      Check(api_.clSetKernelArg(kernel,
                                static_cast<cl_uint>(i),
                                sizeof(packed_buffers[i]),
                                &packed_buffers[i]),
            "clSetKernelArg(expert8 multiq8 packed)");
    }
    Check(api_.clSetKernelArg(kernel, 8,
                              sizeof(q8_qs_buffer), &q8_qs_buffer),
          "clSetKernelArg(expert8 multiq8 q8_qs)");
    Check(api_.clSetKernelArg(kernel, 9,
                              sizeof(q8_bsums_buffer), &q8_bsums_buffer),
          "clSetKernelArg(expert8 multiq8 q8_bsums)");
    Check(api_.clSetKernelArg(kernel, 10,
                              sizeof(q8_d_buffer), &q8_d_buffer),
          "clSetKernelArg(expert8 multiq8 q8_d)");
    Check(api_.clSetKernelArg(kernel, 11,
                              sizeof(blocks_arg), &blocks_arg),
          "clSetKernelArg(expert8 multiq8 blocks)");
    Check(api_.clSetKernelArg(kernel, 12,
                              sizeof(row_groups_arg), &row_groups_arg),
          "clSetKernelArg(expert8 multiq8 row_groups)");
    Check(api_.clSetKernelArg(kernel, 13, sizeof(out_buffer), &out_buffer),
          "clSetKernelArg(expert8 multiq8 out)");

    const std::size_t global = static_cast<std::size_t>(global_work_items);
    std::vector<double> times;
    times.reserve(static_cast<std::size_t>(repeat));
    for (int i = 0; i < repeat; ++i) {
      cl_event event = nullptr;
      Check(api_.clEnqueueNDRangeKernel(
                queue_, kernel, 1, nullptr, &global,
                nullptr, 0, nullptr, EventOut(&event)),
            "clEnqueueNDRangeKernel(q4k_x8_matvec_rowlane_expert8_multiq8)");
      if (read_output != nullptr) {
        Require(read_output_bytes > 0,
                "expert8 multiq8 Q4 read output size is zero");
        Check(api_.clEnqueueReadBuffer(queue_, out_buffer, kClTrue, 0,
                                       read_output_bytes, read_output, 0,
                                       nullptr, nullptr),
              "clEnqueueReadBuffer(expert8 multiq8 Q4 out)");
      } else {
        if (!defer_finish) {
          Check(api_.clFinish(queue_), "clFinish(expert8 multiq8 kernel)");
        }
      }
      times.push_back(EventUs(api_, event));
      ReleaseEvent(api_, &event);
    }
    if (!defer_finish) {
      ClearPendingHostUploadsAfterQueueDrain();
    }

    GpuQ4X8MatvecTiming timing;
    timing.min_us = *std::min_element(times.begin(), times.end());
    timing.mean_us =
        std::accumulate(times.begin(), times.end(), 0.0) /
          static_cast<double>(times.size());
    timing.effective_packed_gb_s =
        static_cast<double>(global_work_items / kRowsInterleaved *
                            blocks_per_row * kQ4Kx8BlockBytes) /
        (timing.min_us / 1e6) / 1e9;
    timing.global_work_items = global_work_items;
    timing.rows_per_work_item = 1;
    return timing;
  }

  GpuQ4X8MatvecTiming RunExpert8PlusSharedMultiQ8Kernel(
      const std::array<cl_mem, 8>& packed_buffers,
      cl_mem shared_packed_buffer,
      cl_mem selected_q8_qs_buffer,
      cl_mem selected_q8_bsums_buffer,
      cl_mem selected_q8_d_buffer,
      cl_mem shared_q8_qs_buffer,
      cl_mem shared_q8_bsums_buffer,
      cl_mem shared_q8_d_buffer,
      cl_mem selected_out_buffer,
      cl_mem shared_out_buffer,
      std::uint64_t blocks_per_row,
      std::uint64_t row_groups_per_expert,
      std::uint64_t global_work_items,
      int repeat,
      bool defer_finish = false) {
    cl_kernel kernel = kernel_rowlane_expert8_plus_shared_multiq8_;
    Require(kernel != nullptr, "expert8 plus shared multiq8 Q4 kernel missing");
    const cl_uint blocks_arg = static_cast<cl_uint>(blocks_per_row);
    const cl_uint row_groups_arg =
        static_cast<cl_uint>(row_groups_per_expert);
    for (std::size_t i = 0; i < packed_buffers.size(); ++i) {
      Check(api_.clSetKernelArg(kernel, static_cast<cl_uint>(i),
                                sizeof(packed_buffers[i]),
                                &packed_buffers[i]),
            "clSetKernelArg(expert8 plus shared Q4 packed)");
    }
    Check(api_.clSetKernelArg(kernel, 8, sizeof(shared_packed_buffer),
                              &shared_packed_buffer),
          "clSetKernelArg(expert8 plus shared Q4 shared packed)");
    Check(api_.clSetKernelArg(kernel, 9, sizeof(selected_q8_qs_buffer),
                              &selected_q8_qs_buffer),
          "clSetKernelArg(expert8 plus shared Q4 selected q8_qs)");
    Check(api_.clSetKernelArg(kernel, 10, sizeof(selected_q8_bsums_buffer),
                              &selected_q8_bsums_buffer),
          "clSetKernelArg(expert8 plus shared Q4 selected q8_bsums)");
    Check(api_.clSetKernelArg(kernel, 11, sizeof(selected_q8_d_buffer),
                              &selected_q8_d_buffer),
          "clSetKernelArg(expert8 plus shared Q4 selected q8_d)");
    Check(api_.clSetKernelArg(kernel, 12, sizeof(shared_q8_qs_buffer),
                              &shared_q8_qs_buffer),
          "clSetKernelArg(expert8 plus shared Q4 shared q8_qs)");
    Check(api_.clSetKernelArg(kernel, 13, sizeof(shared_q8_bsums_buffer),
                              &shared_q8_bsums_buffer),
          "clSetKernelArg(expert8 plus shared Q4 shared q8_bsums)");
    Check(api_.clSetKernelArg(kernel, 14, sizeof(shared_q8_d_buffer),
                              &shared_q8_d_buffer),
          "clSetKernelArg(expert8 plus shared Q4 shared q8_d)");
    Check(api_.clSetKernelArg(kernel, 15, sizeof(blocks_arg), &blocks_arg),
          "clSetKernelArg(expert8 plus shared Q4 blocks)");
    Check(api_.clSetKernelArg(kernel, 16, sizeof(row_groups_arg),
                              &row_groups_arg),
          "clSetKernelArg(expert8 plus shared Q4 row_groups)");
    Check(api_.clSetKernelArg(kernel, 17, sizeof(selected_out_buffer),
                              &selected_out_buffer),
          "clSetKernelArg(expert8 plus shared Q4 selected out)");
    Check(api_.clSetKernelArg(kernel, 18, sizeof(shared_out_buffer),
                              &shared_out_buffer),
          "clSetKernelArg(expert8 plus shared Q4 shared out)");

    const std::size_t global = static_cast<std::size_t>(global_work_items);
    constexpr std::size_t kExpert8PlusSharedQ4LocalSize = 64;
    const std::size_t* local =
        global % kExpert8PlusSharedQ4LocalSize == 0
            ? &kExpert8PlusSharedQ4LocalSize
            : nullptr;
    std::vector<double> times;
    times.reserve(static_cast<std::size_t>(repeat));
    std::uint64_t kernel_wait_wall_ns = 0;
    std::uint64_t kernel_enqueue_wall_ns = 0;
    std::uint64_t kernel_finish_wall_ns = 0;
    std::uint64_t event_profile_wall_ns = 0;
    const bool split_profile = SelectedDownSubmitSplitProfile();
    for (int i = 0; i < repeat; ++i) {
      const auto kernel_wait_begin = std::chrono::steady_clock::now();
      cl_event event = nullptr;
      const auto enqueue_begin = split_profile
                                     ? std::chrono::steady_clock::now()
                                     : std::chrono::steady_clock::time_point{};
      Check(api_.clEnqueueNDRangeKernel(
                queue_, kernel, 1, nullptr, &global, local, 0, nullptr,
                EventOut(&event)),
            "clEnqueueNDRangeKernel(q4k_x8_selected_down_expert8_plus_shared_q4)");
      if (split_profile) {
        kernel_enqueue_wall_ns +=
            WallNs(enqueue_begin, std::chrono::steady_clock::now());
      }
      if (!defer_finish) {
        const auto finish_begin = split_profile
                                      ? std::chrono::steady_clock::now()
                                      : std::chrono::steady_clock::time_point{};
        Check(api_.clFinish(queue_),
              "clFinish(expert8 plus shared Q4 kernel)");
        if (split_profile) {
          kernel_finish_wall_ns +=
              WallNs(finish_begin, std::chrono::steady_clock::now());
        }
      }
      kernel_wait_wall_ns +=
          WallNs(kernel_wait_begin, std::chrono::steady_clock::now());
      const auto event_profile_begin = std::chrono::steady_clock::now();
      times.push_back(EventUs(api_, event));
      event_profile_wall_ns +=
          WallNs(event_profile_begin, std::chrono::steady_clock::now());
      ReleaseEvent(api_, &event);
    }
    if (!defer_finish) {
      ClearPendingHostUploadsAfterQueueDrain();
    }

    GpuQ4X8MatvecTiming timing;
    timing.min_us = *std::min_element(times.begin(), times.end());
    timing.mean_us =
        std::accumulate(times.begin(), times.end(), 0.0) /
          static_cast<double>(times.size());
    timing.effective_packed_gb_s =
        static_cast<double>(global_work_items / kRowsInterleaved *
                            blocks_per_row * kQ4Kx8BlockBytes) /
        (timing.min_us / 1e6) / 1e9;
    timing.global_work_items = global_work_items;
    timing.rows_per_work_item = 1;
    timing.kernel_wait_wall_ns = kernel_wait_wall_ns;
    timing.kernel_enqueue_wall_ns = kernel_enqueue_wall_ns;
    timing.kernel_finish_wall_ns = kernel_finish_wall_ns;
    timing.event_profile_wall_ns = event_profile_wall_ns;
    return timing;
  }

  GpuQ4X8MatvecTiming RunExpert8F32InputKernel(
      const std::array<cl_mem, 8>& packed_buffers,
      cl_mem input_buffer,
      cl_mem out_buffer,
      std::uint64_t blocks_per_row,
      std::uint64_t row_groups_per_expert,
      std::uint64_t global_work_items,
      int repeat,
      float* read_output = nullptr,
      std::size_t read_output_bytes = 0,
      bool defer_finish = false) {
    cl_kernel kernel = kernel_rowlane_expert8_f32input_;
    Require(kernel != nullptr, "expert8 f32-input Q4 kernel missing");
    const cl_uint blocks_arg = static_cast<cl_uint>(blocks_per_row);
    const cl_uint row_groups_arg =
        static_cast<cl_uint>(row_groups_per_expert);
    for (std::size_t i = 0; i < packed_buffers.size(); ++i) {
      Check(api_.clSetKernelArg(kernel,
                                static_cast<cl_uint>(i),
                                sizeof(packed_buffers[i]),
                                &packed_buffers[i]),
            "clSetKernelArg(expert8 f32-input packed)");
    }
    Check(api_.clSetKernelArg(kernel, 8, sizeof(input_buffer), &input_buffer),
          "clSetKernelArg(expert8 f32-input input)");
    Check(api_.clSetKernelArg(kernel, 9, sizeof(blocks_arg), &blocks_arg),
          "clSetKernelArg(expert8 f32-input blocks)");
    Check(api_.clSetKernelArg(kernel, 10, sizeof(row_groups_arg),
                              &row_groups_arg),
          "clSetKernelArg(expert8 f32-input row_groups)");
    Check(api_.clSetKernelArg(kernel, 11, sizeof(out_buffer), &out_buffer),
          "clSetKernelArg(expert8 f32-input out)");

    const std::size_t global = static_cast<std::size_t>(global_work_items);
    std::vector<double> times;
    times.reserve(static_cast<std::size_t>(repeat));
    for (int i = 0; i < repeat; ++i) {
      cl_event event = nullptr;
      Check(api_.clEnqueueNDRangeKernel(
                queue_, kernel, 1, nullptr, &global, nullptr, 0, nullptr,
                EventOut(&event)),
            "clEnqueueNDRangeKernel(q4k_x8_matvec_rowlane_expert8_f32input)");
      if (read_output != nullptr) {
        Require(read_output_bytes > 0,
                "expert8 f32-input Q4 read output size is zero");
        Check(api_.clEnqueueReadBuffer(queue_, out_buffer, kClTrue, 0,
                                       read_output_bytes, read_output, 0,
                                       nullptr, nullptr),
              "clEnqueueReadBuffer(expert8 f32-input Q4 out)");
      } else if (!defer_finish) {
        Check(api_.clFinish(queue_), "clFinish(expert8 f32-input kernel)");
      }
      times.push_back(EventUs(api_, event));
      ReleaseEvent(api_, &event);
    }
    if (!defer_finish) {
      ClearPendingHostUploadsAfterQueueDrain();
    }

    GpuQ4X8MatvecTiming timing;
    timing.min_us = *std::min_element(times.begin(), times.end());
    timing.mean_us =
        std::accumulate(times.begin(), times.end(), 0.0) /
          static_cast<double>(times.size());
    timing.effective_packed_gb_s =
        static_cast<double>(global_work_items / kRowsInterleaved *
                            blocks_per_row * kQ4Kx8BlockBytes) /
        (timing.min_us / 1e6) / 1e9;
    timing.global_work_items = global_work_items;
    timing.rows_per_work_item = 1;
    return timing;
  }

  GpuQ4X8MatvecTiming RunF32TopK8BlocksKernel(
      cl_mem values_buffer,
      std::uint64_t value_count,
      std::uint64_t block_size,
      cl_mem partial_ids_buffer,
      cl_mem partial_values_buffer,
      std::uint64_t global_work_items,
      int repeat) {
    Require(value_count > 0, "F32 top-k value count must be nonzero");
    Require(block_size > 0, "F32 top-k block size must be nonzero");
    Require(repeat > 0, "F32 top-k repeat must be positive");
    const cl_uint value_count_arg = static_cast<cl_uint>(value_count);
    const cl_uint block_size_arg = static_cast<cl_uint>(block_size);
    Check(api_.clSetKernelArg(kernel_f32_topk8_blocks_, 0,
                              sizeof(values_buffer), &values_buffer),
          "clSetKernelArg(f32 topk values)");
    Check(api_.clSetKernelArg(kernel_f32_topk8_blocks_, 1,
                              sizeof(value_count_arg), &value_count_arg),
          "clSetKernelArg(f32 topk value_count)");
    Check(api_.clSetKernelArg(kernel_f32_topk8_blocks_, 2,
                              sizeof(block_size_arg), &block_size_arg),
          "clSetKernelArg(f32 topk block_size)");
    Check(api_.clSetKernelArg(kernel_f32_topk8_blocks_, 3,
                              sizeof(partial_ids_buffer),
                              &partial_ids_buffer),
          "clSetKernelArg(f32 topk ids)");
    Check(api_.clSetKernelArg(kernel_f32_topk8_blocks_, 4,
                              sizeof(partial_values_buffer),
                              &partial_values_buffer),
          "clSetKernelArg(f32 topk values out)");

    const std::size_t global = static_cast<std::size_t>(global_work_items);
    std::vector<double> times;
    times.reserve(static_cast<std::size_t>(repeat));
    for (int i = 0; i < repeat; ++i) {
      cl_event event = nullptr;
      Check(api_.clEnqueueNDRangeKernel(
                queue_, kernel_f32_topk8_blocks_, 1, nullptr, &global, nullptr,
                0, nullptr, EventOut(&event)),
            "clEnqueueNDRangeKernel(f32_topk8_blocks)");
      Check(api_.clFinish(queue_), "clFinish(f32_topk8_blocks)");
      times.push_back(EventUs(api_, event));
      ReleaseEvent(api_, &event);
    }
    ClearPendingHostUploadsAfterQueueDrain();

    GpuQ4X8MatvecTiming timing;
    timing.min_us = *std::min_element(times.begin(), times.end());
    timing.mean_us =
        std::accumulate(times.begin(), times.end(), 0.0) /
          static_cast<double>(times.size());
    timing.global_work_items = global_work_items;
    timing.rows_per_work_item = block_size;
    return timing;
  }

  GpuQ4X8MatvecTiming RunF32TopK16BlocksKernel(
      cl_mem values_buffer,
      std::uint64_t value_count,
      std::uint64_t block_size,
      cl_mem partial_ids_buffer,
      cl_mem partial_values_buffer,
      std::uint64_t global_work_items,
      int repeat) {
    Require(value_count > 0, "F32 top-k value count must be nonzero");
    Require(block_size > 0, "F32 top-k block size must be nonzero");
    Require(repeat > 0, "F32 top-k repeat must be positive");
    const cl_uint value_count_arg = static_cast<cl_uint>(value_count);
    const cl_uint block_size_arg = static_cast<cl_uint>(block_size);
    Check(api_.clSetKernelArg(kernel_f32_topk16_blocks_, 0,
                              sizeof(values_buffer), &values_buffer),
          "clSetKernelArg(f32 topk16 values)");
    Check(api_.clSetKernelArg(kernel_f32_topk16_blocks_, 1,
                              sizeof(value_count_arg), &value_count_arg),
          "clSetKernelArg(f32 topk16 value_count)");
    Check(api_.clSetKernelArg(kernel_f32_topk16_blocks_, 2,
                              sizeof(block_size_arg), &block_size_arg),
          "clSetKernelArg(f32 topk16 block_size)");
    Check(api_.clSetKernelArg(kernel_f32_topk16_blocks_, 3,
                              sizeof(partial_ids_buffer),
                              &partial_ids_buffer),
          "clSetKernelArg(f32 topk16 ids)");
    Check(api_.clSetKernelArg(kernel_f32_topk16_blocks_, 4,
                              sizeof(partial_values_buffer),
                              &partial_values_buffer),
          "clSetKernelArg(f32 topk16 values out)");

    const std::size_t global = static_cast<std::size_t>(global_work_items);
    std::vector<double> times;
    times.reserve(static_cast<std::size_t>(repeat));
    for (int i = 0; i < repeat; ++i) {
      cl_event event = nullptr;
      Check(api_.clEnqueueNDRangeKernel(
                queue_, kernel_f32_topk16_blocks_, 1, nullptr, &global,
                nullptr, 0, nullptr, EventOut(&event)),
            "clEnqueueNDRangeKernel(f32_topk16_blocks)");
      Check(api_.clFinish(queue_), "clFinish(f32_topk16_blocks)");
      times.push_back(EventUs(api_, event));
      ReleaseEvent(api_, &event);
    }
    ClearPendingHostUploadsAfterQueueDrain();

    GpuQ4X8MatvecTiming timing;
    timing.min_us = *std::min_element(times.begin(), times.end());
    timing.mean_us =
        std::accumulate(times.begin(), times.end(), 0.0) /
          static_cast<double>(times.size());
    timing.global_work_items = global_work_items;
    timing.rows_per_work_item = block_size;
    return timing;
  }

  GpuQ4X8SwiGluHandoffTiming RunMatvecThenSwiGluKernel(
      GpuQ4X8KernelVariant variant,
      cl_mem packed_buffer,
      cl_mem q8_qs_buffer,
      cl_mem q8_bsums_buffer,
      cl_mem q8_d_buffer,
      cl_mem gate_up_buffer,
      cl_mem swiglu_buffer,
      std::uint64_t blocks_per_row,
      std::uint64_t row_groups,
      std::uint64_t matvec_global_work_items,
      std::uint64_t intermediate_size,
      std::uint64_t expert_count,
      int repeat,
      float* read_output = nullptr,
      std::size_t read_output_bytes = 0) {
    const bool use_rowlane_localq8 =
        variant == GpuQ4X8KernelVariant::kRowlaneParallel &&
        blocks_per_row == 8 &&
        (matvec_global_work_items == 1024 ||
         matvec_global_work_items == 32256);
    cl_kernel matvec_kernel =
        use_rowlane_localq8 ? kernel_rowlane_localq8_ : Kernel(variant);
    constexpr std::size_t kSharedQ4RowlaneLocalQ8Size = 64;
    const std::size_t* matvec_local =
        use_rowlane_localq8 ? &kSharedQ4RowlaneLocalQ8Size : nullptr;
    const char* matvec_kernel_name =
        use_rowlane_localq8 ? "q4k_x8_matvec_rowlane_localq8"
                            : KernelFunctionName(variant);
    const cl_uint blocks_arg = static_cast<cl_uint>(blocks_per_row);
    const cl_uint row_groups_arg = static_cast<cl_uint>(row_groups);
    Check(api_.clSetKernelArg(matvec_kernel, 0, sizeof(packed_buffer),
                              &packed_buffer),
          "clSetKernelArg(identity SwiGLU matvec 0)");
    Check(api_.clSetKernelArg(matvec_kernel, 1, sizeof(q8_qs_buffer),
                              &q8_qs_buffer),
          "clSetKernelArg(identity SwiGLU matvec 1)");
    Check(api_.clSetKernelArg(matvec_kernel, 2, sizeof(q8_bsums_buffer),
                              &q8_bsums_buffer),
          "clSetKernelArg(identity SwiGLU matvec 2)");
    Check(api_.clSetKernelArg(matvec_kernel, 3, sizeof(q8_d_buffer),
                              &q8_d_buffer),
          "clSetKernelArg(identity SwiGLU matvec 3)");
    Check(api_.clSetKernelArg(matvec_kernel, 4, sizeof(blocks_arg),
                              &blocks_arg),
          "clSetKernelArg(identity SwiGLU matvec 4)");
    Check(api_.clSetKernelArg(matvec_kernel, 5, sizeof(row_groups_arg),
                              &row_groups_arg),
          "clSetKernelArg(identity SwiGLU matvec 5)");
    Check(api_.clSetKernelArg(matvec_kernel, 6, sizeof(gate_up_buffer),
                              &gate_up_buffer),
          "clSetKernelArg(identity SwiGLU matvec 6)");

    const cl_uint intermediate_arg = static_cast<cl_uint>(intermediate_size);
    const cl_uint expert_arg = static_cast<cl_uint>(expert_count);
    Check(api_.clSetKernelArg(kernel_ffn_swiglu_, 0, sizeof(gate_up_buffer),
                              &gate_up_buffer),
          "clSetKernelArg(identity SwiGLU 0)");
    Check(api_.clSetKernelArg(kernel_ffn_swiglu_, 1,
                              sizeof(intermediate_arg), &intermediate_arg),
          "clSetKernelArg(identity SwiGLU 1)");
    Check(api_.clSetKernelArg(kernel_ffn_swiglu_, 2, sizeof(expert_arg),
                              &expert_arg),
          "clSetKernelArg(identity SwiGLU 2)");
    Check(api_.clSetKernelArg(kernel_ffn_swiglu_, 3, sizeof(swiglu_buffer),
                              &swiglu_buffer),
          "clSetKernelArg(identity SwiGLU 3)");

    const std::size_t matvec_global =
        static_cast<std::size_t>(matvec_global_work_items);
    const std::size_t swiglu_global =
        static_cast<std::size_t>(intermediate_size * expert_count);
    std::vector<double> matvec_times;
    std::vector<double> swiglu_times;
    matvec_times.reserve(static_cast<std::size_t>(repeat));
    swiglu_times.reserve(static_cast<std::size_t>(repeat));
    for (int i = 0; i < repeat; ++i) {
      cl_event matvec_event = nullptr;
      cl_event swiglu_event = nullptr;
      Check(api_.clEnqueueNDRangeKernel(
                queue_, matvec_kernel, 1, nullptr, &matvec_global,
                matvec_local, 0, nullptr, EventOut(&matvec_event)),
            std::string("clEnqueueNDRangeKernel(") + matvec_kernel_name +
                " identity SwiGLU)");
      Check(api_.clEnqueueNDRangeKernel(queue_, kernel_ffn_swiglu_, 1, nullptr,
                                        &swiglu_global, nullptr, 0, nullptr,
                                        EventOut(&swiglu_event)),
            "clEnqueueNDRangeKernel(ffn_moe_swiglu_f32 identity)");
      if (read_output != nullptr) {
        Require(read_output_bytes > 0, "identity SwiGLU read output size is zero");
        Check(api_.clEnqueueReadBuffer(queue_, swiglu_buffer, kClTrue, 0,
                                       read_output_bytes, read_output, 0,
                                       nullptr, nullptr),
              "clEnqueueReadBuffer(identity SwiGLU output)");
      } else {
        Check(api_.clFinish(queue_), "clFinish(identity SwiGLU handoff)");
      }
      matvec_times.push_back(EventUs(api_, matvec_event));
      swiglu_times.push_back(EventUs(api_, swiglu_event));
      ReleaseEvent(api_, &swiglu_event);
      ReleaseEvent(api_, &matvec_event);
    }
    ClearPendingHostUploadsAfterQueueDrain();

    GpuQ4X8SwiGluHandoffTiming timing;
    timing.matvec.min_us = *std::min_element(matvec_times.begin(), matvec_times.end());
    timing.matvec.mean_us =
        std::accumulate(matvec_times.begin(), matvec_times.end(), 0.0) /
          static_cast<double>(matvec_times.size());
    timing.matvec.effective_packed_gb_s =
        static_cast<double>(matvec_global_work_items * RowsPerWorkItem(variant) /
                            kRowsInterleaved * blocks_per_row *
                            kQ4Kx8BlockBytes) /
        (timing.matvec.min_us / 1e6) / 1e9;
    timing.matvec.global_work_items = matvec_global_work_items;
    timing.matvec.rows_per_work_item = RowsPerWorkItem(variant);
    timing.swiglu_min_us =
        *std::min_element(swiglu_times.begin(), swiglu_times.end());
    timing.swiglu_mean_us =
        std::accumulate(swiglu_times.begin(), swiglu_times.end(), 0.0) /
          static_cast<double>(swiglu_times.size());
    timing.swiglu_global_work_items = intermediate_size * expert_count;
    timing.shell_sum_min_us = timing.matvec.min_us + timing.swiglu_min_us;
    timing.shell_sum_mean_us = timing.matvec.mean_us + timing.swiglu_mean_us;
    return timing;
  }

  GpuSwiGluTiming RunSwiGluBufferKernel(cl_mem gate_up_buffer,
                                        cl_mem output_buffer,
                                        std::uint64_t intermediate_size,
                                        std::uint64_t expert_count,
                                        int repeat) {
    const cl_uint intermediate_arg = static_cast<cl_uint>(intermediate_size);
    const cl_uint expert_arg = static_cast<cl_uint>(expert_count);
    Check(api_.clSetKernelArg(kernel_ffn_swiglu_, 0, sizeof(gate_up_buffer),
                              &gate_up_buffer),
          "clSetKernelArg(SwiGLU buffer 0)");
    Check(api_.clSetKernelArg(kernel_ffn_swiglu_, 1,
                              sizeof(intermediate_arg), &intermediate_arg),
          "clSetKernelArg(SwiGLU buffer 1)");
    Check(api_.clSetKernelArg(kernel_ffn_swiglu_, 2, sizeof(expert_arg),
                              &expert_arg),
          "clSetKernelArg(SwiGLU buffer 2)");
    Check(api_.clSetKernelArg(kernel_ffn_swiglu_, 3, sizeof(output_buffer),
                              &output_buffer),
          "clSetKernelArg(SwiGLU buffer 3)");

    const std::size_t global =
        static_cast<std::size_t>(intermediate_size * expert_count);
    std::vector<double> times;
    times.reserve(static_cast<std::size_t>(repeat));
    for (int i = 0; i < repeat; ++i) {
      cl_event event = nullptr;
      Check(api_.clEnqueueNDRangeKernel(queue_, kernel_ffn_swiglu_, 1,
                                        nullptr, &global, nullptr, 0, nullptr,
                                        EventOut(&event)),
            "clEnqueueNDRangeKernel(ffn_moe_swiglu_f32)");
      Check(api_.clFinish(queue_), "clFinish(SwiGLU buffer)");
      times.push_back(EventUs(api_, event));
      ReleaseEvent(api_, &event);
    }

    GpuSwiGluTiming timing;
    timing.min_us = *std::min_element(times.begin(), times.end());
    timing.mean_us =
        std::accumulate(times.begin(), times.end(), 0.0) /
          static_cast<double>(times.size());
    timing.global_work_items = intermediate_size * expert_count;
    return timing;
  }

  GpuSwiGluTiming RunSwiGluReorderKernel(
      cl_mem gate_up_buffer,
      cl_mem source_map_buffer,
      cl_mem output_buffer,
      std::uint64_t intermediate_size,
      std::uint64_t expert_count,
      int repeat) {
    const cl_uint intermediate_arg = static_cast<cl_uint>(intermediate_size);
    const cl_uint expert_arg = static_cast<cl_uint>(expert_count);
    Check(api_.clSetKernelArg(kernel_ffn_swiglu_reorder_, 0,
                              sizeof(gate_up_buffer), &gate_up_buffer),
          "clSetKernelArg(SwiGLU reorder 0)");
    Check(api_.clSetKernelArg(kernel_ffn_swiglu_reorder_, 1,
                              sizeof(source_map_buffer), &source_map_buffer),
          "clSetKernelArg(SwiGLU reorder 1)");
    Check(api_.clSetKernelArg(kernel_ffn_swiglu_reorder_, 2,
                              sizeof(intermediate_arg), &intermediate_arg),
          "clSetKernelArg(SwiGLU reorder 2)");
    Check(api_.clSetKernelArg(kernel_ffn_swiglu_reorder_, 3,
                              sizeof(expert_arg), &expert_arg),
          "clSetKernelArg(SwiGLU reorder 3)");
    Check(api_.clSetKernelArg(kernel_ffn_swiglu_reorder_, 4,
                              sizeof(output_buffer), &output_buffer),
          "clSetKernelArg(SwiGLU reorder 4)");

    const std::size_t global =
        static_cast<std::size_t>(intermediate_size * expert_count);
    std::vector<double> times;
    times.reserve(static_cast<std::size_t>(repeat));
    for (int i = 0; i < repeat; ++i) {
      cl_event event = nullptr;
      Check(api_.clEnqueueNDRangeKernel(queue_, kernel_ffn_swiglu_reorder_, 1,
                                        nullptr, &global, nullptr, 0, nullptr,
                                        EventOut(&event)),
            "clEnqueueNDRangeKernel(ffn_moe_swiglu_reorder_f32)");
      Check(api_.clFinish(queue_), "clFinish(SwiGLU reorder)");
      times.push_back(EventUs(api_, event));
      ReleaseEvent(api_, &event);
    }

    GpuSwiGluTiming timing;
    timing.min_us = *std::min_element(times.begin(), times.end());
    timing.mean_us =
        std::accumulate(times.begin(), times.end(), 0.0) /
          static_cast<double>(times.size());
    timing.global_work_items = intermediate_size * expert_count;
    return timing;
  }

  GpuSwiGluTiming RunQ8QuantizeKernel(cl_mem input_buffer,
                                      std::uint64_t block_count,
                                      cl_mem q8_qs_buffer,
                                      cl_mem q8_d_buffer,
                                      int repeat) {
    const cl_uint block_count_arg = static_cast<cl_uint>(block_count);
    Check(api_.clSetKernelArg(kernel_q8_quantize_, 0,
                              sizeof(input_buffer), &input_buffer),
          "clSetKernelArg(Q8 quantize 0)");
    Check(api_.clSetKernelArg(kernel_q8_quantize_, 1,
                              sizeof(block_count_arg), &block_count_arg),
          "clSetKernelArg(Q8 quantize 1)");
    Check(api_.clSetKernelArg(kernel_q8_quantize_, 2,
                              sizeof(q8_qs_buffer), &q8_qs_buffer),
          "clSetKernelArg(Q8 quantize 2)");
    Check(api_.clSetKernelArg(kernel_q8_quantize_, 3,
                              sizeof(q8_d_buffer), &q8_d_buffer),
          "clSetKernelArg(Q8 quantize 3)");

    const std::size_t global = static_cast<std::size_t>(block_count);
    std::vector<double> times;
    times.reserve(static_cast<std::size_t>(repeat));
    for (int i = 0; i < repeat; ++i) {
      cl_event event = nullptr;
      Check(api_.clEnqueueNDRangeKernel(queue_, kernel_q8_quantize_, 1,
                                        nullptr, &global, nullptr, 0, nullptr,
                                        EventOut(&event)),
            "clEnqueueNDRangeKernel(q8k_quantize_f32_blocks)");
      Check(api_.clFinish(queue_), "clFinish(Q8 quantize)");
      times.push_back(EventUs(api_, event));
      ReleaseEvent(api_, &event);
    }

    GpuSwiGluTiming timing;
    timing.min_us = *std::min_element(times.begin(), times.end());
    timing.mean_us =
        std::accumulate(times.begin(), times.end(), 0.0) /
          static_cast<double>(times.size());
    timing.global_work_items = block_count;
    return timing;
  }

  GpuSwiGluTiming RunQ8QuantizeWithBsumsKernel(
      cl_mem input_buffer,
      std::uint64_t block_count,
      cl_mem q8_qs_buffer,
      cl_mem q8_bsums_buffer,
      cl_mem q8_d_buffer,
      int repeat) {
    const cl_uint block_count_arg = static_cast<cl_uint>(block_count);
    Check(api_.clSetKernelArg(kernel_q8_quantize_bsums_, 0,
                              sizeof(input_buffer), &input_buffer),
          "clSetKernelArg(Q8 quantize bsums 0)");
    Check(api_.clSetKernelArg(kernel_q8_quantize_bsums_, 1,
                              sizeof(block_count_arg), &block_count_arg),
          "clSetKernelArg(Q8 quantize bsums 1)");
    Check(api_.clSetKernelArg(kernel_q8_quantize_bsums_, 2,
                              sizeof(q8_qs_buffer), &q8_qs_buffer),
          "clSetKernelArg(Q8 quantize bsums 2)");
    Check(api_.clSetKernelArg(kernel_q8_quantize_bsums_, 3,
                              sizeof(q8_bsums_buffer), &q8_bsums_buffer),
          "clSetKernelArg(Q8 quantize bsums 3)");
    Check(api_.clSetKernelArg(kernel_q8_quantize_bsums_, 4,
                              sizeof(q8_d_buffer), &q8_d_buffer),
          "clSetKernelArg(Q8 quantize bsums 4)");

    const std::size_t global = static_cast<std::size_t>(block_count);
    std::vector<double> times;
    times.reserve(static_cast<std::size_t>(repeat));
    for (int i = 0; i < repeat; ++i) {
      cl_event event = nullptr;
      Check(api_.clEnqueueNDRangeKernel(
                queue_, kernel_q8_quantize_bsums_, 1, nullptr, &global,
                nullptr, 0, nullptr, EventOut(&event)),
            "clEnqueueNDRangeKernel(q8k_quantize_f32_blocks_with_bsums)");
      Check(api_.clFinish(queue_), "clFinish(Q8 quantize bsums)");
      times.push_back(EventUs(api_, event));
      ReleaseEvent(api_, &event);
    }

    GpuSwiGluTiming timing;
    timing.min_us = *std::min_element(times.begin(), times.end());
    timing.mean_us =
        std::accumulate(times.begin(), times.end(), 0.0) /
          static_cast<double>(times.size());
    timing.global_work_items = block_count;
    return timing;
  }

  GpuQ4X8MatvecTiming RunQ6KKernel(cl_mem raw_buffer,
                                   cl_mem q8_qs_buffer,
                                   cl_mem q8_d_buffer,
                                   cl_mem out_buffer,
                                   std::uint64_t rows,
                                   std::uint64_t blocks_per_row,
                                   int repeat,
                                   bool defer_finish = false,
                                   bool cpuorder = false) {
    return RunQ6KSelectedKernel(raw_buffer, q8_qs_buffer, q8_d_buffer,
                                out_buffer, rows, blocks_per_row, rows,
                                repeat, defer_finish, cpuorder);
  }

  GpuQ4X8MatvecTiming RunQ6KSelectedKernel(cl_mem raw_buffer,
                                           cl_mem q8_qs_buffer,
                                           cl_mem q8_d_buffer,
                                           cl_mem out_buffer,
                                           std::uint64_t rows_per_expert,
                                           std::uint64_t blocks_per_row,
                                           std::uint64_t total_rows,
                                           int repeat,
                                           bool defer_finish = false,
                                           bool cpuorder = false) {
    cl_kernel kernel =
        cpuorder ? kernel_q6_linear_qkv_cpuorder_ : kernel_q6_matvec_row_;
    const char* kernel_name =
        cpuorder ? "q6k_linear_qkv_cpuorder_nofma"
                 : "q6k_selected_down_matvec_row";
    const cl_uint rows_per_expert_arg =
        static_cast<cl_uint>(rows_per_expert);
    const cl_uint blocks_per_row_arg = static_cast<cl_uint>(blocks_per_row);
    Check(api_.clSetKernelArg(kernel, 0, sizeof(raw_buffer), &raw_buffer),
          "clSetKernelArg(q6 matvec 0)");
    Check(api_.clSetKernelArg(kernel, 1, sizeof(q8_qs_buffer), &q8_qs_buffer),
          "clSetKernelArg(q6 matvec 1)");
    Check(api_.clSetKernelArg(kernel, 2, sizeof(q8_d_buffer), &q8_d_buffer),
          "clSetKernelArg(q6 matvec 2)");
    Check(api_.clSetKernelArg(kernel, 3, sizeof(rows_per_expert_arg), &rows_per_expert_arg),
          "clSetKernelArg(q6 matvec 3)");
    Check(api_.clSetKernelArg(kernel, 4, sizeof(blocks_per_row_arg), &blocks_per_row_arg),
          "clSetKernelArg(q6 matvec 4)");
    Check(api_.clSetKernelArg(kernel, 5, sizeof(out_buffer), &out_buffer),
          "clSetKernelArg(q6 matvec 5)");

    const std::size_t global = static_cast<std::size_t>(total_rows);
    constexpr std::size_t kQ6LocalSize = 64;
    constexpr std::size_t kQ6LocalSizeMaxRows = 65536;
    const std::size_t* local =
        (global <= kQ6LocalSizeMaxRows && global % kQ6LocalSize == 0)
            ? &kQ6LocalSize
            : nullptr;
    std::vector<double> times;
    times.reserve(static_cast<std::size_t>(repeat));
    for (int i = 0; i < repeat; ++i) {
      cl_event event = nullptr;
      Check(api_.clEnqueueNDRangeKernel(queue_, kernel, 1,
                                       nullptr, &global, local, 0, nullptr,
                                       EventOut(&event)),
            std::string("clEnqueueNDRangeKernel(") + kernel_name + ")");
      if (!defer_finish) {
        Check(api_.clFinish(queue_), "clFinish(q6 matvec)");
      }
      times.push_back(EventUs(api_, event));
      ReleaseEvent(api_, &event);
    }
    if (!defer_finish) {
      ClearPendingHostUploadsAfterQueueDrain();
    }

    GpuQ4X8MatvecTiming timing;
    timing.min_us = *std::min_element(times.begin(), times.end());
    timing.mean_us =
        std::accumulate(times.begin(), times.end(), 0.0) /
          static_cast<double>(times.size());
    timing.effective_packed_gb_s =
        static_cast<double>(total_rows * blocks_per_row * kQ6KBlockBytes) /
        (timing.min_us / 1e6) / 1e9;
    timing.global_work_items = total_rows;
    timing.rows_per_work_item = 1;
    return timing;
  }

  GpuQ4X8MatvecTiming RunQ6KSelectedRowstripeKernel(
      cl_mem scratch_buffer,
      cl_mem q8_qs_buffer,
      cl_mem q8_d_buffer,
      cl_mem out_buffer,
      std::uint64_t rows_per_expert,
      std::uint64_t blocks_per_row,
      std::uint64_t total_rows,
      std::uint64_t rows_per_tile,
      int repeat,
      bool defer_finish = false) {
    const cl_uint rows_per_expert_arg =
        static_cast<cl_uint>(rows_per_expert);
    const cl_uint blocks_per_row_arg = static_cast<cl_uint>(blocks_per_row);
    const cl_uint rows_per_tile_arg = static_cast<cl_uint>(rows_per_tile);
    const std::size_t global = static_cast<std::size_t>(total_rows);
    constexpr std::size_t kQ6RowstripeDefaultLocalSize = 64;
    constexpr std::size_t kQ6RowstripeLargeLocalSize = 128;
    const std::size_t q6_rowstripe_local_size =
        total_rows > 65536 ? kQ6RowstripeLargeLocalSize
                           : kQ6RowstripeDefaultLocalSize;
    const bool local_size_compatible =
        global % q6_rowstripe_local_size == 0;
    const bool use_local_q8 =
        rows_per_expert == total_rows && total_rows == 8192 &&
        blocks_per_row == 8 && local_size_compatible;
    cl_kernel rowstripe_kernel =
        use_local_q8 ? kernel_q6_matvec_rowstripe_localq8_
                     : kernel_q6_matvec_rowstripe_;
    Check(api_.clSetKernelArg(rowstripe_kernel, 0,
                              sizeof(scratch_buffer), &scratch_buffer),
          "clSetKernelArg(q6 rowstripe 0)");
    Check(api_.clSetKernelArg(rowstripe_kernel, 1,
                              sizeof(q8_qs_buffer), &q8_qs_buffer),
          "clSetKernelArg(q6 rowstripe 1)");
    Check(api_.clSetKernelArg(rowstripe_kernel, 2,
                              sizeof(q8_d_buffer), &q8_d_buffer),
          "clSetKernelArg(q6 rowstripe 2)");
    Check(api_.clSetKernelArg(rowstripe_kernel, 3,
                              sizeof(rows_per_expert_arg),
                              &rows_per_expert_arg),
          "clSetKernelArg(q6 rowstripe 3)");
    Check(api_.clSetKernelArg(rowstripe_kernel, 4,
                              sizeof(blocks_per_row_arg),
                              &blocks_per_row_arg),
          "clSetKernelArg(q6 rowstripe 4)");
    Check(api_.clSetKernelArg(rowstripe_kernel, 5,
                              sizeof(rows_per_tile_arg), &rows_per_tile_arg),
          "clSetKernelArg(q6 rowstripe 5)");
    Check(api_.clSetKernelArg(rowstripe_kernel, 6,
                              sizeof(out_buffer), &out_buffer),
          "clSetKernelArg(q6 rowstripe 6)");

    const std::size_t* local =
        local_size_compatible ? &q6_rowstripe_local_size : nullptr;
    std::vector<double> times;
    times.reserve(static_cast<std::size_t>(repeat));
    for (int i = 0; i < repeat; ++i) {
      cl_event event = nullptr;
      Check(api_.clEnqueueNDRangeKernel(
                queue_, rowstripe_kernel, 1, nullptr, &global, local, 0,
                nullptr, EventOut(&event)),
            "clEnqueueNDRangeKernel(q6k_selected_down_matvec_rowstripe)");
      if (!defer_finish) {
        Check(api_.clFinish(queue_), "clFinish(q6 rowstripe matvec)");
      }
      times.push_back(EventUs(api_, event));
      ReleaseEvent(api_, &event);
    }
    if (!defer_finish) {
      ClearPendingHostUploadsAfterQueueDrain();
    }

    GpuQ4X8MatvecTiming timing;
    timing.min_us = *std::min_element(times.begin(), times.end());
    timing.mean_us =
        std::accumulate(times.begin(), times.end(), 0.0) /
          static_cast<double>(times.size());
    timing.effective_packed_gb_s =
        static_cast<double>(total_rows * blocks_per_row * kQ6KBlockBytes) /
        (timing.min_us / 1e6) / 1e9;
    timing.global_work_items = total_rows;
    timing.rows_per_work_item = 1;
    return timing;
  }

  GpuQ4X8MatvecTiming RunQ6KExpert8Kernel(
      const std::array<cl_mem, 8>& raw_buffers,
      cl_mem q8_qs_buffer,
      cl_mem q8_d_buffer,
      cl_mem out_buffer,
      std::uint64_t rows_per_expert,
      std::uint64_t blocks_per_row,
      std::uint64_t total_rows,
      int repeat,
      float* read_output = nullptr,
      std::size_t read_bytes = 0) {
    const auto kernel_setup_begin = std::chrono::steady_clock::now();
    const auto wall_ns =
        [](std::chrono::steady_clock::time_point begin,
           std::chrono::steady_clock::time_point end) -> std::uint64_t {
      return static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin)
              .count());
    };
    const cl_uint rows_per_expert_arg =
        static_cast<cl_uint>(rows_per_expert);
    const cl_uint blocks_per_row_arg = static_cast<cl_uint>(blocks_per_row);
    for (std::size_t i = 0; i < raw_buffers.size(); ++i) {
      Check(api_.clSetKernelArg(kernel_q6_matvec_row_expert8_,
                                static_cast<cl_uint>(i),
                                sizeof(raw_buffers[i]), &raw_buffers[i]),
            "clSetKernelArg(q6 expert8 raw)");
    }
    Check(api_.clSetKernelArg(kernel_q6_matvec_row_expert8_, 8,
                              sizeof(q8_qs_buffer), &q8_qs_buffer),
          "clSetKernelArg(q6 expert8 q8_qs)");
    Check(api_.clSetKernelArg(kernel_q6_matvec_row_expert8_, 9,
                              sizeof(q8_d_buffer), &q8_d_buffer),
          "clSetKernelArg(q6 expert8 q8_d)");
    Check(api_.clSetKernelArg(kernel_q6_matvec_row_expert8_, 10,
                              sizeof(rows_per_expert_arg),
                              &rows_per_expert_arg),
          "clSetKernelArg(q6 expert8 rows_per_expert)");
    Check(api_.clSetKernelArg(kernel_q6_matvec_row_expert8_, 11,
                              sizeof(blocks_per_row_arg),
                              &blocks_per_row_arg),
          "clSetKernelArg(q6 expert8 blocks_per_row)");
    Check(api_.clSetKernelArg(kernel_q6_matvec_row_expert8_, 12,
                              sizeof(out_buffer), &out_buffer),
          "clSetKernelArg(q6 expert8 out)");

    const std::size_t global = static_cast<std::size_t>(total_rows);
    constexpr std::size_t kExpert8Q6LocalSize = 64;
    const std::size_t* local =
        (global % kExpert8Q6LocalSize == 0) ? &kExpert8Q6LocalSize : nullptr;
    const auto kernel_setup_wall_ns =
        wall_ns(kernel_setup_begin, std::chrono::steady_clock::now());
    std::vector<double> times;
    times.reserve(static_cast<std::size_t>(repeat));
    std::uint64_t kernel_wait_wall_ns = 0;
    std::uint64_t kernel_enqueue_wall_ns = 0;
    std::uint64_t kernel_finish_wall_ns = 0;
    std::uint64_t event_profile_wall_ns = 0;
    std::uint64_t output_read_wall_ns = 0;
    const bool split_profile = SelectedDownSubmitSplitProfile();
    for (int i = 0; i < repeat; ++i) {
      const auto kernel_wait_begin = std::chrono::steady_clock::now();
      cl_event event = nullptr;
      const auto enqueue_begin = split_profile
                                     ? std::chrono::steady_clock::now()
                                     : std::chrono::steady_clock::time_point{};
      Check(api_.clEnqueueNDRangeKernel(
                queue_, kernel_q6_matvec_row_expert8_, 1, nullptr, &global,
                local, 0, nullptr, EventOut(&event)),
            "clEnqueueNDRangeKernel(q6k_selected_down_matvec_row_expert8)");
      if (split_profile) {
        kernel_enqueue_wall_ns +=
            wall_ns(enqueue_begin, std::chrono::steady_clock::now());
      }
      if (read_output != nullptr && read_bytes > 0) {
        const auto read_begin = std::chrono::steady_clock::now();
        Check(api_.clEnqueueReadBuffer(queue_, out_buffer, kClTrue, 0,
                                       read_bytes, read_output, 0, nullptr,
                                       nullptr),
              "clEnqueueReadBuffer(q6 expert8 out)");
        output_read_wall_ns +=
            wall_ns(read_begin, std::chrono::steady_clock::now());
      } else {
        const auto finish_begin = split_profile
                                      ? std::chrono::steady_clock::now()
                                      : std::chrono::steady_clock::time_point{};
        Check(api_.clFinish(queue_), "clFinish(q6 expert8 matvec)");
        if (split_profile) {
          kernel_finish_wall_ns +=
              wall_ns(finish_begin, std::chrono::steady_clock::now());
        }
      }
      kernel_wait_wall_ns +=
          wall_ns(kernel_wait_begin, std::chrono::steady_clock::now());
      const auto event_profile_begin = std::chrono::steady_clock::now();
      times.push_back(EventUs(api_, event));
      event_profile_wall_ns +=
          wall_ns(event_profile_begin, std::chrono::steady_clock::now());
      ReleaseEvent(api_, &event);
    }
    ClearPendingHostUploadsAfterQueueDrain();

    GpuQ4X8MatvecTiming timing;
    timing.min_us = *std::min_element(times.begin(), times.end());
    timing.mean_us =
        std::accumulate(times.begin(), times.end(), 0.0) /
          static_cast<double>(times.size());
    timing.effective_packed_gb_s =
        static_cast<double>(total_rows * blocks_per_row * kQ6KBlockBytes) /
        (timing.min_us / 1e6) / 1e9;
    timing.global_work_items = total_rows;
    timing.rows_per_work_item = 1;
    timing.kernel_setup_wall_ns = kernel_setup_wall_ns;
    timing.kernel_wait_wall_ns = kernel_wait_wall_ns;
    timing.kernel_enqueue_wall_ns = kernel_enqueue_wall_ns;
    timing.kernel_finish_wall_ns = kernel_finish_wall_ns;
    timing.event_profile_wall_ns = event_profile_wall_ns;
    timing.output_read_wall_ns = output_read_wall_ns;
    return timing;
  }

  GpuQ4X8MatvecTiming RunQ6KExpert8RowstripeKernel(
      const std::array<cl_mem, 8>& raw_buffers,
      cl_mem q8_qs_buffer,
      cl_mem q8_d_buffer,
      cl_mem out_buffer,
      std::uint64_t rows_per_expert,
      std::uint64_t blocks_per_row,
      std::uint64_t rows_per_tile,
      std::uint64_t total_rows,
      int repeat,
      float* read_output = nullptr,
      std::size_t read_bytes = 0) {
    const auto kernel_setup_begin = std::chrono::steady_clock::now();
    const auto wall_ns =
        [](std::chrono::steady_clock::time_point begin,
           std::chrono::steady_clock::time_point end) -> std::uint64_t {
      return static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin)
              .count());
    };
    const cl_uint rows_per_expert_arg =
        static_cast<cl_uint>(rows_per_expert);
    const cl_uint blocks_per_row_arg = static_cast<cl_uint>(blocks_per_row);
    const cl_uint rows_per_tile_arg = static_cast<cl_uint>(rows_per_tile);
    for (std::size_t i = 0; i < raw_buffers.size(); ++i) {
      Check(api_.clSetKernelArg(kernel_q6_matvec_rowstripe_expert8_,
                                static_cast<cl_uint>(i),
                                sizeof(raw_buffers[i]), &raw_buffers[i]),
            "clSetKernelArg(q6 rowstripe expert8 raw)");
    }
    Check(api_.clSetKernelArg(kernel_q6_matvec_rowstripe_expert8_, 8,
                              sizeof(q8_qs_buffer), &q8_qs_buffer),
          "clSetKernelArg(q6 rowstripe expert8 q8_qs)");
    Check(api_.clSetKernelArg(kernel_q6_matvec_rowstripe_expert8_, 9,
                              sizeof(q8_d_buffer), &q8_d_buffer),
          "clSetKernelArg(q6 rowstripe expert8 q8_d)");
    Check(api_.clSetKernelArg(kernel_q6_matvec_rowstripe_expert8_, 10,
                              sizeof(rows_per_expert_arg),
                              &rows_per_expert_arg),
          "clSetKernelArg(q6 rowstripe expert8 rows_per_expert)");
    Check(api_.clSetKernelArg(kernel_q6_matvec_rowstripe_expert8_, 11,
                              sizeof(blocks_per_row_arg),
                              &blocks_per_row_arg),
          "clSetKernelArg(q6 rowstripe expert8 blocks_per_row)");
    Check(api_.clSetKernelArg(kernel_q6_matvec_rowstripe_expert8_, 12,
                              sizeof(rows_per_tile_arg), &rows_per_tile_arg),
          "clSetKernelArg(q6 rowstripe expert8 rows_per_tile)");
    Check(api_.clSetKernelArg(kernel_q6_matvec_rowstripe_expert8_, 13,
                              sizeof(out_buffer), &out_buffer),
          "clSetKernelArg(q6 rowstripe expert8 out)");

    const std::size_t global = static_cast<std::size_t>(total_rows);
    constexpr std::size_t kExpert8Q6RowstripeLocalSize = 64;
    const std::size_t* local =
        (global % kExpert8Q6RowstripeLocalSize == 0)
            ? &kExpert8Q6RowstripeLocalSize
            : nullptr;
    const auto kernel_setup_wall_ns =
        wall_ns(kernel_setup_begin, std::chrono::steady_clock::now());
    std::vector<double> times;
    times.reserve(static_cast<std::size_t>(repeat));
    std::uint64_t kernel_wait_wall_ns = 0;
    std::uint64_t kernel_enqueue_wall_ns = 0;
    std::uint64_t kernel_finish_wall_ns = 0;
    std::uint64_t event_profile_wall_ns = 0;
    std::uint64_t output_read_wall_ns = 0;
    const bool split_profile = SelectedDownSubmitSplitProfile();
    for (int i = 0; i < repeat; ++i) {
      const auto kernel_wait_begin = std::chrono::steady_clock::now();
      cl_event event = nullptr;
      const auto enqueue_begin = split_profile
                                     ? std::chrono::steady_clock::now()
                                     : std::chrono::steady_clock::time_point{};
      Check(api_.clEnqueueNDRangeKernel(
                queue_, kernel_q6_matvec_rowstripe_expert8_, 1, nullptr,
                &global, local, 0, nullptr, EventOut(&event)),
            "clEnqueueNDRangeKernel(q6k_selected_down_matvec_rowstripe_expert8)");
      if (split_profile) {
        kernel_enqueue_wall_ns +=
            wall_ns(enqueue_begin, std::chrono::steady_clock::now());
      }
      if (read_output != nullptr && read_bytes > 0) {
        const auto read_begin = std::chrono::steady_clock::now();
        Check(api_.clEnqueueReadBuffer(queue_, out_buffer, kClTrue, 0,
                                       read_bytes, read_output, 0, nullptr,
                                       nullptr),
              "clEnqueueReadBuffer(q6 rowstripe expert8 out)");
        output_read_wall_ns +=
            wall_ns(read_begin, std::chrono::steady_clock::now());
      } else {
        const auto finish_begin = split_profile
                                      ? std::chrono::steady_clock::now()
                                      : std::chrono::steady_clock::time_point{};
        Check(api_.clFinish(queue_), "clFinish(q6 rowstripe expert8 matvec)");
        if (split_profile) {
          kernel_finish_wall_ns +=
              wall_ns(finish_begin, std::chrono::steady_clock::now());
        }
      }
      kernel_wait_wall_ns +=
          wall_ns(kernel_wait_begin, std::chrono::steady_clock::now());
      const auto event_profile_begin = std::chrono::steady_clock::now();
      times.push_back(EventUs(api_, event));
      event_profile_wall_ns +=
          wall_ns(event_profile_begin, std::chrono::steady_clock::now());
      ReleaseEvent(api_, &event);
    }
    ClearPendingHostUploadsAfterQueueDrain();

    GpuQ4X8MatvecTiming timing;
    timing.min_us = *std::min_element(times.begin(), times.end());
    timing.mean_us =
        std::accumulate(times.begin(), times.end(), 0.0) /
          static_cast<double>(times.size());
    timing.effective_packed_gb_s =
        static_cast<double>(total_rows * blocks_per_row * kQ6KBlockBytes) /
        (timing.min_us / 1e6) / 1e9;
    timing.global_work_items = total_rows;
    timing.rows_per_work_item = 1;
    timing.kernel_setup_wall_ns = kernel_setup_wall_ns;
    timing.kernel_wait_wall_ns = kernel_wait_wall_ns;
    timing.kernel_enqueue_wall_ns = kernel_enqueue_wall_ns;
    timing.kernel_finish_wall_ns = kernel_finish_wall_ns;
    timing.event_profile_wall_ns = event_profile_wall_ns;
    timing.output_read_wall_ns = output_read_wall_ns;
    return timing;
  }

  GpuQ4X8MatvecTiming RunQ6KExpert8RowstripePlusSharedKernel(
      const std::array<cl_mem, 8>& selected_buffers,
      cl_mem shared_buffer,
      cl_mem selected_q8_qs_buffer,
      cl_mem selected_q8_d_buffer,
      cl_mem shared_q8_qs_buffer,
      cl_mem shared_q8_d_buffer,
      cl_mem selected_out_buffer,
      cl_mem shared_out_buffer,
      std::uint64_t rows_per_expert,
      std::uint64_t blocks_per_row,
      std::uint64_t rows_per_tile,
      int repeat,
      bool defer_finish = false) {
    const auto kernel_setup_begin = std::chrono::steady_clock::now();
    const auto wall_ns =
        [](std::chrono::steady_clock::time_point begin,
           std::chrono::steady_clock::time_point end) -> std::uint64_t {
      return static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin)
              .count());
    };
    const cl_uint rows_per_expert_arg =
        static_cast<cl_uint>(rows_per_expert);
    const cl_uint blocks_per_row_arg = static_cast<cl_uint>(blocks_per_row);
    const cl_uint rows_per_tile_arg = static_cast<cl_uint>(rows_per_tile);
    for (std::size_t i = 0; i < selected_buffers.size(); ++i) {
      Check(api_.clSetKernelArg(
                kernel_q6_matvec_rowstripe_expert8_plus_shared_,
                static_cast<cl_uint>(i), sizeof(selected_buffers[i]),
                &selected_buffers[i]),
            "clSetKernelArg(q6 rowstripe expert8 plus shared raw)");
    }
    Check(api_.clSetKernelArg(
              kernel_q6_matvec_rowstripe_expert8_plus_shared_, 8,
              sizeof(shared_buffer), &shared_buffer),
          "clSetKernelArg(q6 rowstripe expert8 plus shared shared raw)");
    Check(api_.clSetKernelArg(
              kernel_q6_matvec_rowstripe_expert8_plus_shared_, 9,
              sizeof(selected_q8_qs_buffer), &selected_q8_qs_buffer),
          "clSetKernelArg(q6 rowstripe expert8 plus shared selected q8_qs)");
    Check(api_.clSetKernelArg(
              kernel_q6_matvec_rowstripe_expert8_plus_shared_, 10,
              sizeof(selected_q8_d_buffer), &selected_q8_d_buffer),
          "clSetKernelArg(q6 rowstripe expert8 plus shared selected q8_d)");
    Check(api_.clSetKernelArg(
              kernel_q6_matvec_rowstripe_expert8_plus_shared_, 11,
              sizeof(shared_q8_qs_buffer), &shared_q8_qs_buffer),
          "clSetKernelArg(q6 rowstripe expert8 plus shared shared q8_qs)");
    Check(api_.clSetKernelArg(
              kernel_q6_matvec_rowstripe_expert8_plus_shared_, 12,
              sizeof(shared_q8_d_buffer), &shared_q8_d_buffer),
          "clSetKernelArg(q6 rowstripe expert8 plus shared shared q8_d)");
    Check(api_.clSetKernelArg(
              kernel_q6_matvec_rowstripe_expert8_plus_shared_, 13,
              sizeof(rows_per_expert_arg), &rows_per_expert_arg),
          "clSetKernelArg(q6 rowstripe expert8 plus shared rows_per_expert)");
    Check(api_.clSetKernelArg(
              kernel_q6_matvec_rowstripe_expert8_plus_shared_, 14,
              sizeof(blocks_per_row_arg), &blocks_per_row_arg),
          "clSetKernelArg(q6 rowstripe expert8 plus shared blocks_per_row)");
    Check(api_.clSetKernelArg(
              kernel_q6_matvec_rowstripe_expert8_plus_shared_, 15,
              sizeof(rows_per_tile_arg), &rows_per_tile_arg),
          "clSetKernelArg(q6 rowstripe expert8 plus shared rows_per_tile)");
    Check(api_.clSetKernelArg(
              kernel_q6_matvec_rowstripe_expert8_plus_shared_, 16,
              sizeof(selected_out_buffer), &selected_out_buffer),
          "clSetKernelArg(q6 rowstripe expert8 plus shared selected out)");
    Check(api_.clSetKernelArg(
              kernel_q6_matvec_rowstripe_expert8_plus_shared_, 17,
              sizeof(shared_out_buffer), &shared_out_buffer),
          "clSetKernelArg(q6 rowstripe expert8 plus shared shared out)");

    const std::size_t global =
        static_cast<std::size_t>(rows_per_expert * 9);
    constexpr std::size_t kExpert8Q6RowstripeLocalSize = 64;
    const std::size_t* local =
        (global % kExpert8Q6RowstripeLocalSize == 0)
            ? &kExpert8Q6RowstripeLocalSize
            : nullptr;
    const auto kernel_setup_wall_ns =
        wall_ns(kernel_setup_begin, std::chrono::steady_clock::now());
    std::vector<double> times;
    times.reserve(static_cast<std::size_t>(repeat));
    std::uint64_t kernel_wait_wall_ns = 0;
    std::uint64_t kernel_enqueue_wall_ns = 0;
    std::uint64_t kernel_finish_wall_ns = 0;
    std::uint64_t event_profile_wall_ns = 0;
    const bool split_profile = SelectedDownSubmitSplitProfile();
    for (int i = 0; i < repeat; ++i) {
      const auto kernel_wait_begin = std::chrono::steady_clock::now();
      cl_event event = nullptr;
      const auto enqueue_begin = split_profile
                                     ? std::chrono::steady_clock::now()
                                     : std::chrono::steady_clock::time_point{};
      Check(api_.clEnqueueNDRangeKernel(
                queue_, kernel_q6_matvec_rowstripe_expert8_plus_shared_, 1,
                nullptr, &global, local, 0, nullptr, EventOut(&event)),
            "clEnqueueNDRangeKernel(q6k_selected_down_matvec_rowstripe_expert8_plus_shared_raw)");
      if (split_profile) {
        kernel_enqueue_wall_ns +=
            wall_ns(enqueue_begin, std::chrono::steady_clock::now());
      }
      if (!defer_finish) {
        const auto finish_begin = split_profile
                                      ? std::chrono::steady_clock::now()
                                      : std::chrono::steady_clock::time_point{};
        Check(api_.clFinish(queue_),
              "clFinish(q6 rowstripe expert8 plus shared matvec)");
        if (split_profile) {
          kernel_finish_wall_ns +=
              wall_ns(finish_begin, std::chrono::steady_clock::now());
        }
      }
      kernel_wait_wall_ns +=
          wall_ns(kernel_wait_begin, std::chrono::steady_clock::now());
      const auto event_profile_begin = std::chrono::steady_clock::now();
      times.push_back(EventUs(api_, event));
      event_profile_wall_ns +=
          wall_ns(event_profile_begin, std::chrono::steady_clock::now());
      ReleaseEvent(api_, &event);
    }
    if (!defer_finish) {
      ClearPendingHostUploadsAfterQueueDrain();
    }

    GpuQ4X8MatvecTiming timing;
    timing.min_us = *std::min_element(times.begin(), times.end());
    timing.mean_us =
        std::accumulate(times.begin(), times.end(), 0.0) /
        static_cast<double>(times.size());
    timing.effective_packed_gb_s =
        static_cast<double>(global * blocks_per_row * kQ6KBlockBytes) /
        (timing.min_us / 1e6) / 1e9;
    timing.global_work_items = global;
    timing.rows_per_work_item = 1;
    timing.kernel_setup_wall_ns = kernel_setup_wall_ns;
    timing.kernel_wait_wall_ns = kernel_wait_wall_ns;
    timing.kernel_enqueue_wall_ns = kernel_enqueue_wall_ns;
    timing.kernel_finish_wall_ns = kernel_finish_wall_ns;
    timing.event_profile_wall_ns = event_profile_wall_ns;
    return timing;
  }

  GpuQ4X8ConvHandoffTiming RunHandoffKernels(
      GpuQ4X8KernelVariant variant,
      cl_mem packed_buffer,
      cl_mem q8_qs_buffer,
      cl_mem q8_bsums_buffer,
      cl_mem q8_d_buffer,
      cl_mem qkv_buffer,
      cl_mem conv_weights_buffer,
      cl_mem conv_state_buffer,
      cl_mem conv_output_buffer,
      cl_mem next_state_buffer,
      std::uint64_t blocks_per_row,
      std::uint64_t row_groups,
      std::uint64_t matvec_global_work_items,
      std::uint64_t rows,
      std::uint64_t conv_kernel_size,
      int repeat) {
    cl_kernel matvec_kernel = Kernel(variant);
    const cl_uint blocks_arg = static_cast<cl_uint>(blocks_per_row);
    const cl_uint row_groups_arg = static_cast<cl_uint>(row_groups);
    Check(api_.clSetKernelArg(matvec_kernel, 0, sizeof(packed_buffer), &packed_buffer), "clSetKernelArg(matvec 0)");
    Check(api_.clSetKernelArg(matvec_kernel, 1, sizeof(q8_qs_buffer), &q8_qs_buffer), "clSetKernelArg(matvec 1)");
    Check(api_.clSetKernelArg(matvec_kernel, 2, sizeof(q8_bsums_buffer), &q8_bsums_buffer), "clSetKernelArg(matvec 2)");
    Check(api_.clSetKernelArg(matvec_kernel, 3, sizeof(q8_d_buffer), &q8_d_buffer), "clSetKernelArg(matvec 3)");
    Check(api_.clSetKernelArg(matvec_kernel, 4, sizeof(blocks_arg), &blocks_arg), "clSetKernelArg(matvec 4)");
    Check(api_.clSetKernelArg(matvec_kernel, 5, sizeof(row_groups_arg), &row_groups_arg), "clSetKernelArg(matvec 5)");
    Check(api_.clSetKernelArg(matvec_kernel, 6, sizeof(qkv_buffer), &qkv_buffer), "clSetKernelArg(matvec 6)");

    const cl_uint channel_count_arg = static_cast<cl_uint>(rows);
    const cl_uint kernel_size_arg = static_cast<cl_uint>(conv_kernel_size);
    Check(api_.clSetKernelArg(kernel_conv_, 0, sizeof(qkv_buffer), &qkv_buffer), "clSetKernelArg(conv 0)");
    Check(api_.clSetKernelArg(kernel_conv_, 1, sizeof(conv_state_buffer), &conv_state_buffer), "clSetKernelArg(conv 1)");
    Check(api_.clSetKernelArg(kernel_conv_, 2, sizeof(conv_weights_buffer), &conv_weights_buffer), "clSetKernelArg(conv 2)");
    Check(api_.clSetKernelArg(kernel_conv_, 3, sizeof(channel_count_arg), &channel_count_arg), "clSetKernelArg(conv 3)");
    Check(api_.clSetKernelArg(kernel_conv_, 4, sizeof(kernel_size_arg), &kernel_size_arg), "clSetKernelArg(conv 4)");
    Check(api_.clSetKernelArg(kernel_conv_, 5, sizeof(conv_output_buffer), &conv_output_buffer), "clSetKernelArg(conv 5)");
    Check(api_.clSetKernelArg(kernel_conv_, 6, sizeof(next_state_buffer), &next_state_buffer), "clSetKernelArg(conv 6)");

    const std::size_t matvec_global = static_cast<std::size_t>(matvec_global_work_items);
    const std::size_t conv_global = static_cast<std::size_t>(rows);
    std::vector<double> matvec_times;
    std::vector<double> conv_times;
    matvec_times.reserve(static_cast<std::size_t>(repeat));
    conv_times.reserve(static_cast<std::size_t>(repeat));
    for (int i = 0; i < repeat; ++i) {
      cl_event matvec_event = nullptr;
      cl_event conv_event = nullptr;
      Check(api_.clEnqueueNDRangeKernel(queue_, matvec_kernel, 1, nullptr,
                                        &matvec_global, nullptr, 0, nullptr,
                                        EventOut(&matvec_event)),
            std::string("clEnqueueNDRangeKernel(") + KernelFunctionName(variant) + ")");
      Check(api_.clEnqueueNDRangeKernel(queue_, kernel_conv_, 1, nullptr,
                                        &conv_global, nullptr, 0, nullptr,
                                        EventOut(&conv_event)),
            "clEnqueueNDRangeKernel(linear_attn_conv_f32)");
      Check(api_.clFinish(queue_), "clFinish(handoff)");
      matvec_times.push_back(EventUs(api_, matvec_event));
      conv_times.push_back(EventUs(api_, conv_event));
      ReleaseEvent(api_, &conv_event);
      ReleaseEvent(api_, &matvec_event);
    }

    GpuQ4X8ConvHandoffTiming timing;
    timing.matvec.min_us = *std::min_element(matvec_times.begin(), matvec_times.end());
    timing.matvec.mean_us =
        std::accumulate(matvec_times.begin(), matvec_times.end(), 0.0) /
          static_cast<double>(matvec_times.size());
    timing.matvec.effective_packed_gb_s =
        static_cast<double>(matvec_global_work_items * RowsPerWorkItem(variant) / kRowsInterleaved *
                            blocks_per_row * kQ4Kx8BlockBytes) /
        (timing.matvec.min_us / 1e6) / 1e9;
    timing.matvec.global_work_items = matvec_global_work_items;
    timing.matvec.rows_per_work_item = RowsPerWorkItem(variant);
    timing.conv_min_us = *std::min_element(conv_times.begin(), conv_times.end());
    timing.conv_mean_us =
        std::accumulate(conv_times.begin(), conv_times.end(), 0.0) /
          static_cast<double>(conv_times.size());
    timing.shell_sum_min_us = timing.matvec.min_us + timing.conv_min_us;
    timing.shell_sum_mean_us = timing.matvec.mean_us + timing.conv_mean_us;
    timing.conv_global_work_items = rows;
    return timing;
  }

  GpuQ4KCpuOrderMatvecTiming RunQ4KCpuOrderKernel(
      const ResidentRawQ4KCpuOrder& resident,
      cl_mem q8_qs_buffer,
      cl_mem q8_bsums_buffer,
      cl_mem q8_d_buffer,
      cl_mem out_buffer,
      int repeat,
      float* read_output = nullptr,
      std::size_t read_output_bytes = 0) {
    const cl_uint blocks_arg = static_cast<cl_uint>(resident.blocks_per_row);
    const cl_uint rows_arg = static_cast<cl_uint>(resident.rows);
    Check(api_.clSetKernelArg(kernel_q4_cpu_order_, 0,
                              sizeof(resident.buffer), &resident.buffer),
          "clSetKernelArg(resident Q4 CPU-order raw)");
    Check(api_.clSetKernelArg(kernel_q4_cpu_order_, 1,
                              sizeof(q8_qs_buffer), &q8_qs_buffer),
          "clSetKernelArg(resident Q4 CPU-order q8_qs)");
    Check(api_.clSetKernelArg(kernel_q4_cpu_order_, 2,
                              sizeof(q8_bsums_buffer), &q8_bsums_buffer),
          "clSetKernelArg(resident Q4 CPU-order q8_bsums)");
    Check(api_.clSetKernelArg(kernel_q4_cpu_order_, 3,
                              sizeof(q8_d_buffer), &q8_d_buffer),
          "clSetKernelArg(resident Q4 CPU-order q8_d)");
    Check(api_.clSetKernelArg(kernel_q4_cpu_order_, 4, sizeof(blocks_arg),
                              &blocks_arg),
          "clSetKernelArg(resident Q4 CPU-order blocks)");
    Check(api_.clSetKernelArg(kernel_q4_cpu_order_, 5, sizeof(rows_arg),
                              &rows_arg),
          "clSetKernelArg(resident Q4 CPU-order rows)");
    Check(api_.clSetKernelArg(kernel_q4_cpu_order_, 6,
                              sizeof(out_buffer), &out_buffer),
          "clSetKernelArg(resident Q4 CPU-order out)");

    const std::size_t global = static_cast<std::size_t>(resident.rows);
    double total_us = 0.0;
    double min_us = std::numeric_limits<double>::infinity();
    for (int i = 0; i < repeat; ++i) {
      cl_event event = nullptr;
      Check(api_.clEnqueueNDRangeKernel(queue_, kernel_q4_cpu_order_, 1,
                                        nullptr, &global, nullptr, 0, nullptr,
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

  GpuLinearAttentionPostConvPrepTiming RunPostConvPrepKernels(
      cl_mem raw_buffer,
      cl_mem silu_buffer,
      cl_mem q_buffer,
      cl_mem k_buffer,
      cl_mem v_buffer,
      cl_mem q_norm_buffer,
      cl_mem k_norm_buffer,
      std::uint64_t head_dim,
      std::uint64_t query_heads,
      std::uint64_t q_values,
      std::uint64_t total_values,
      float norm_epsilon,
      int repeat,
      bool cpuorder = false) {
    cl_kernel silu_kernel = cpuorder ? kernel_postconv_silu_split_cpuorder_
                                     : kernel_postconv_silu_split_;
    cl_kernel l2_kernel = cpuorder ? kernel_postconv_l2_qk_cpuorder_
                                   : kernel_postconv_l2_qk_;
    const cl_uint q_values_arg = static_cast<cl_uint>(q_values);
    const cl_uint total_values_arg = static_cast<cl_uint>(total_values);
    const cl_uint head_dim_arg = static_cast<cl_uint>(head_dim);
    const cl_uint query_heads_arg = static_cast<cl_uint>(query_heads);
    Check(api_.clSetKernelArg(silu_kernel, 0, sizeof(raw_buffer), &raw_buffer), "clSetKernelArg(postconv silu 0)");
    Check(api_.clSetKernelArg(silu_kernel, 1, sizeof(q_values_arg), &q_values_arg), "clSetKernelArg(postconv silu 1)");
    Check(api_.clSetKernelArg(silu_kernel, 2, sizeof(total_values_arg), &total_values_arg), "clSetKernelArg(postconv silu 2)");
    Check(api_.clSetKernelArg(silu_kernel, 3, sizeof(silu_buffer), &silu_buffer), "clSetKernelArg(postconv silu 3)");
    Check(api_.clSetKernelArg(silu_kernel, 4, sizeof(q_buffer), &q_buffer), "clSetKernelArg(postconv silu 4)");
    Check(api_.clSetKernelArg(silu_kernel, 5, sizeof(k_buffer), &k_buffer), "clSetKernelArg(postconv silu 5)");
    Check(api_.clSetKernelArg(silu_kernel, 6, sizeof(v_buffer), &v_buffer), "clSetKernelArg(postconv silu 6)");
    Check(api_.clSetKernelArg(l2_kernel, 0, sizeof(q_buffer), &q_buffer), "clSetKernelArg(postconv qk l2 0)");
    Check(api_.clSetKernelArg(l2_kernel, 1, sizeof(k_buffer), &k_buffer), "clSetKernelArg(postconv qk l2 1)");
    Check(api_.clSetKernelArg(l2_kernel, 2, sizeof(head_dim_arg), &head_dim_arg), "clSetKernelArg(postconv qk l2 2)");
    Check(api_.clSetKernelArg(l2_kernel, 3, sizeof(query_heads_arg), &query_heads_arg), "clSetKernelArg(postconv qk l2 3)");
    Check(api_.clSetKernelArg(l2_kernel, 4, sizeof(norm_epsilon), &norm_epsilon), "clSetKernelArg(postconv qk l2 4)");
    Check(api_.clSetKernelArg(l2_kernel, 5, sizeof(q_norm_buffer), &q_norm_buffer), "clSetKernelArg(postconv qk l2 5)");
    Check(api_.clSetKernelArg(l2_kernel, 6, sizeof(k_norm_buffer), &k_norm_buffer), "clSetKernelArg(postconv qk l2 6)");

    const std::size_t silu_global = static_cast<std::size_t>(total_values);
    const std::size_t l2_global = static_cast<std::size_t>(query_heads * 2);
    std::vector<double> silu_times;
    std::vector<double> q_l2_times;
    std::vector<double> k_l2_times;
    silu_times.reserve(static_cast<std::size_t>(repeat));
    q_l2_times.reserve(static_cast<std::size_t>(repeat));
    k_l2_times.reserve(static_cast<std::size_t>(repeat));
    for (int i = 0; i < repeat; ++i) {
      cl_event silu_event = nullptr;
      cl_event qk_l2_event = nullptr;
      Check(api_.clEnqueueNDRangeKernel(queue_, silu_kernel, 1,
                                        nullptr, &silu_global, nullptr, 0,
                                        nullptr, EventOut(&silu_event)),
            "clEnqueueNDRangeKernel(linear_attn_postconv_silu_split_f32)");
      Check(api_.clEnqueueNDRangeKernel(queue_, l2_kernel, 1,
                                        nullptr, &l2_global, nullptr, 0,
                                        nullptr, EventOut(&qk_l2_event)),
            "clEnqueueNDRangeKernel(linear_attn_l2_norm_qk_heads_f32)");
      Check(api_.clFinish(queue_), "clFinish(postconv prep)");
      silu_times.push_back(EventUs(api_, silu_event));
      q_l2_times.push_back(EventUs(api_, qk_l2_event));
      k_l2_times.push_back(0.0);
      ReleaseEvent(api_, &qk_l2_event);
      ReleaseEvent(api_, &silu_event);
    }

    GpuLinearAttentionPostConvPrepTiming timing;
    timing.silu_split_min_us = *std::min_element(silu_times.begin(), silu_times.end());
    timing.silu_split_mean_us =
        std::accumulate(silu_times.begin(), silu_times.end(), 0.0) /
          static_cast<double>(silu_times.size());
    timing.q_l2_min_us = *std::min_element(q_l2_times.begin(), q_l2_times.end());
    timing.q_l2_mean_us =
        std::accumulate(q_l2_times.begin(), q_l2_times.end(), 0.0) /
          static_cast<double>(q_l2_times.size());
    timing.k_l2_min_us = *std::min_element(k_l2_times.begin(), k_l2_times.end());
    timing.k_l2_mean_us =
        std::accumulate(k_l2_times.begin(), k_l2_times.end(), 0.0) /
          static_cast<double>(k_l2_times.size());
    timing.silu_split_global_work_items = total_values;
    timing.q_l2_global_work_items = query_heads;
    timing.k_l2_global_work_items = query_heads;
    return timing;
  }

  GpuLinearAttentionPostConvPrepTiming RunPostConvPrepFusedKernel(
      cl_mem raw_buffer,
      cl_mem silu_buffer,
      cl_mem q_buffer,
      cl_mem k_buffer,
      cl_mem v_buffer,
      cl_mem q_norm_buffer,
      cl_mem k_norm_buffer,
      std::uint64_t head_dim,
      std::uint64_t query_heads,
      std::uint64_t value_heads,
      float norm_epsilon,
      int repeat) {
    const cl_uint head_dim_arg = static_cast<cl_uint>(head_dim);
    const cl_uint query_heads_arg = static_cast<cl_uint>(query_heads);
    const cl_uint value_heads_arg = static_cast<cl_uint>(value_heads);
    Check(api_.clSetKernelArg(kernel_postconv_fused_qk_l2_, 0, sizeof(raw_buffer), &raw_buffer), "clSetKernelArg(postconv fused 0)");
    Check(api_.clSetKernelArg(kernel_postconv_fused_qk_l2_, 1, sizeof(head_dim_arg), &head_dim_arg), "clSetKernelArg(postconv fused 1)");
    Check(api_.clSetKernelArg(kernel_postconv_fused_qk_l2_, 2, sizeof(query_heads_arg), &query_heads_arg), "clSetKernelArg(postconv fused 2)");
    Check(api_.clSetKernelArg(kernel_postconv_fused_qk_l2_, 3, sizeof(value_heads_arg), &value_heads_arg), "clSetKernelArg(postconv fused 3)");
    Check(api_.clSetKernelArg(kernel_postconv_fused_qk_l2_, 4, sizeof(norm_epsilon), &norm_epsilon), "clSetKernelArg(postconv fused 4)");
    Check(api_.clSetKernelArg(kernel_postconv_fused_qk_l2_, 5, sizeof(silu_buffer), &silu_buffer), "clSetKernelArg(postconv fused 5)");
    Check(api_.clSetKernelArg(kernel_postconv_fused_qk_l2_, 6, sizeof(q_buffer), &q_buffer), "clSetKernelArg(postconv fused 6)");
    Check(api_.clSetKernelArg(kernel_postconv_fused_qk_l2_, 7, sizeof(k_buffer), &k_buffer), "clSetKernelArg(postconv fused 7)");
    Check(api_.clSetKernelArg(kernel_postconv_fused_qk_l2_, 8, sizeof(v_buffer), &v_buffer), "clSetKernelArg(postconv fused 8)");
    Check(api_.clSetKernelArg(kernel_postconv_fused_qk_l2_, 9, sizeof(q_norm_buffer), &q_norm_buffer), "clSetKernelArg(postconv fused 9)");
    Check(api_.clSetKernelArg(kernel_postconv_fused_qk_l2_, 10, sizeof(k_norm_buffer), &k_norm_buffer), "clSetKernelArg(postconv fused 10)");

    const std::size_t fused_global =
        static_cast<std::size_t>(2 * query_heads + head_dim * value_heads);
    std::vector<double> fused_times;
    fused_times.reserve(static_cast<std::size_t>(repeat));
    for (int i = 0; i < repeat; ++i) {
      cl_event fused_event = nullptr;
      Check(api_.clEnqueueNDRangeKernel(queue_, kernel_postconv_fused_qk_l2_, 1,
                                        nullptr, &fused_global, nullptr, 0,
                                        nullptr, EventOut(&fused_event)),
            "clEnqueueNDRangeKernel(linear_attn_postconv_fused_qk_l2_f32)");
      Check(api_.clFinish(queue_), "clFinish(postconv fused)");
      fused_times.push_back(EventUs(api_, fused_event));
      ReleaseEvent(api_, &fused_event);
    }

    GpuLinearAttentionPostConvPrepTiming timing;
    timing.fused_min_us = *std::min_element(fused_times.begin(), fused_times.end());
    timing.fused_mean_us =
        std::accumulate(fused_times.begin(), fused_times.end(), 0.0) /
          static_cast<double>(fused_times.size());
    timing.silu_split_min_us = timing.fused_min_us;
    timing.silu_split_mean_us = timing.fused_mean_us;
    timing.fused_global_work_items = fused_global;
    return timing;
  }

  GpuLinearAttentionDeltaTiming RunLinearAttentionDeltaKernels(
      cl_mem q_buffer,
      cl_mem k_buffer,
      cl_mem v_buffer,
      cl_mem gate_buffer,
      cl_mem beta_buffer,
      cl_mem state_buffer,
      cl_mem z_buffer,
      cl_mem norm_buffer,
      cl_mem attention_buffer,
      cl_mem next_state_buffer,
      cl_mem final_buffer,
      std::uint64_t head_dim,
      std::uint64_t query_heads,
      std::uint64_t value_heads,
      float norm_epsilon,
      int repeat,
      bool cpu_shape_final_norm,
      bool cpuorder = false) {
    const cl_uint head_dim_arg = static_cast<cl_uint>(head_dim);
    const cl_uint query_heads_arg = static_cast<cl_uint>(query_heads);
    const cl_uint value_heads_arg = static_cast<cl_uint>(value_heads);
    cl_kernel final_kernel =
        cpu_shape_final_norm ? kernel_delta_final_cpu_shape_ : kernel_delta_final_;
    const std::size_t delta_global = static_cast<std::size_t>(head_dim * value_heads);
    constexpr std::size_t kDeltaQkLocalSize = 128;
    const bool can_use_qk_local =
        head_dim == kDeltaQkLocalSize &&
        delta_global % kDeltaQkLocalSize == 0;
    Require(!cpuorder || can_use_qk_local,
            "CPU-order linear delta requires head_dim 128");
    const bool use_cpuorder = cpuorder && can_use_qk_local;
    const bool use_fused_qk_local =
        can_use_qk_local && !cpu_shape_final_norm && !use_cpuorder;
    const bool use_fused_cpu_shape_qk_local =
        can_use_qk_local && cpu_shape_final_norm && !use_cpuorder;
    const bool use_qk_local =
        can_use_qk_local && !use_fused_qk_local &&
        !use_fused_cpu_shape_qk_local;
    cl_kernel delta_kernel =
        use_cpuorder
            ? kernel_delta_recurrent_final_cpuorder_
            : use_fused_qk_local
            ? kernel_delta_recurrent_final_qk_local_
            : (use_fused_cpu_shape_qk_local
                   ? kernel_delta_recurrent_final_cpu_shape_qk_local_
                   : (use_qk_local ? kernel_delta_recurrent_qk_local_
                                   : kernel_delta_recurrent_));
    if (use_cpuorder || use_fused_qk_local || use_fused_cpu_shape_qk_local) {
      Check(api_.clSetKernelArg(delta_kernel, 0, sizeof(q_buffer), &q_buffer), "clSetKernelArg(delta fused 0)");
      Check(api_.clSetKernelArg(delta_kernel, 1, sizeof(k_buffer), &k_buffer), "clSetKernelArg(delta fused 1)");
      Check(api_.clSetKernelArg(delta_kernel, 2, sizeof(v_buffer), &v_buffer), "clSetKernelArg(delta fused 2)");
      Check(api_.clSetKernelArg(delta_kernel, 3, sizeof(gate_buffer), &gate_buffer), "clSetKernelArg(delta fused 3)");
      Check(api_.clSetKernelArg(delta_kernel, 4, sizeof(beta_buffer), &beta_buffer), "clSetKernelArg(delta fused 4)");
      Check(api_.clSetKernelArg(delta_kernel, 5, sizeof(state_buffer), &state_buffer), "clSetKernelArg(delta fused 5)");
      Check(api_.clSetKernelArg(delta_kernel, 6, sizeof(z_buffer), &z_buffer), "clSetKernelArg(delta fused 6)");
      Check(api_.clSetKernelArg(delta_kernel, 7, sizeof(norm_buffer), &norm_buffer), "clSetKernelArg(delta fused 7)");
      Check(api_.clSetKernelArg(delta_kernel, 8, sizeof(head_dim_arg), &head_dim_arg), "clSetKernelArg(delta fused 8)");
      Check(api_.clSetKernelArg(delta_kernel, 9, sizeof(query_heads_arg), &query_heads_arg), "clSetKernelArg(delta fused 9)");
      Check(api_.clSetKernelArg(delta_kernel, 10, sizeof(value_heads_arg), &value_heads_arg), "clSetKernelArg(delta fused 10)");
      Check(api_.clSetKernelArg(delta_kernel, 11, sizeof(norm_epsilon), &norm_epsilon), "clSetKernelArg(delta fused 11)");
      Check(api_.clSetKernelArg(delta_kernel, 12, sizeof(attention_buffer), &attention_buffer), "clSetKernelArg(delta fused 12)");
      Check(api_.clSetKernelArg(delta_kernel, 13, sizeof(next_state_buffer), &next_state_buffer), "clSetKernelArg(delta fused 13)");
      Check(api_.clSetKernelArg(delta_kernel, 14, sizeof(final_buffer), &final_buffer), "clSetKernelArg(delta fused 14)");
    } else {
      Check(api_.clSetKernelArg(delta_kernel, 0, sizeof(q_buffer), &q_buffer), "clSetKernelArg(delta 0)");
      Check(api_.clSetKernelArg(delta_kernel, 1, sizeof(k_buffer), &k_buffer), "clSetKernelArg(delta 1)");
      Check(api_.clSetKernelArg(delta_kernel, 2, sizeof(v_buffer), &v_buffer), "clSetKernelArg(delta 2)");
      Check(api_.clSetKernelArg(delta_kernel, 3, sizeof(gate_buffer), &gate_buffer), "clSetKernelArg(delta 3)");
      Check(api_.clSetKernelArg(delta_kernel, 4, sizeof(beta_buffer), &beta_buffer), "clSetKernelArg(delta 4)");
      Check(api_.clSetKernelArg(delta_kernel, 5, sizeof(state_buffer), &state_buffer), "clSetKernelArg(delta 5)");
      Check(api_.clSetKernelArg(delta_kernel, 6, sizeof(head_dim_arg), &head_dim_arg), "clSetKernelArg(delta 6)");
      Check(api_.clSetKernelArg(delta_kernel, 7, sizeof(query_heads_arg), &query_heads_arg), "clSetKernelArg(delta 7)");
      Check(api_.clSetKernelArg(delta_kernel, 8, sizeof(value_heads_arg), &value_heads_arg), "clSetKernelArg(delta 8)");
      Check(api_.clSetKernelArg(delta_kernel, 9, sizeof(attention_buffer), &attention_buffer), "clSetKernelArg(delta 9)");
      Check(api_.clSetKernelArg(delta_kernel, 10, sizeof(next_state_buffer), &next_state_buffer), "clSetKernelArg(delta 10)");
      Check(api_.clSetKernelArg(final_kernel, 0, sizeof(attention_buffer), &attention_buffer), "clSetKernelArg(delta final 0)");
      Check(api_.clSetKernelArg(final_kernel, 1, sizeof(z_buffer), &z_buffer), "clSetKernelArg(delta final 1)");
      Check(api_.clSetKernelArg(final_kernel, 2, sizeof(norm_buffer), &norm_buffer), "clSetKernelArg(delta final 2)");
      Check(api_.clSetKernelArg(final_kernel, 3, sizeof(head_dim_arg), &head_dim_arg), "clSetKernelArg(delta final 3)");
      Check(api_.clSetKernelArg(final_kernel, 4, sizeof(value_heads_arg), &value_heads_arg), "clSetKernelArg(delta final 4)");
      Check(api_.clSetKernelArg(final_kernel, 5, sizeof(norm_epsilon), &norm_epsilon), "clSetKernelArg(delta final 5)");
      Check(api_.clSetKernelArg(final_kernel, 6, sizeof(final_buffer), &final_buffer), "clSetKernelArg(delta final 6)");
    }

    const std::size_t final_global = static_cast<std::size_t>(value_heads);
    const std::size_t* delta_local =
        (use_cpuorder || use_qk_local || use_fused_qk_local ||
         use_fused_cpu_shape_qk_local)
            ? &kDeltaQkLocalSize
            : nullptr;
    std::vector<double> delta_times;
    std::vector<double> final_times;
    delta_times.reserve(static_cast<std::size_t>(repeat));
    final_times.reserve(static_cast<std::size_t>(repeat));
    for (int i = 0; i < repeat; ++i) {
      cl_event delta_event = nullptr;
      cl_event final_event = nullptr;
      Check(api_.clEnqueueNDRangeKernel(queue_, delta_kernel, 1,
                                        nullptr, &delta_global, delta_local, 0,
                                        nullptr, EventOut(&delta_event)),
            "clEnqueueNDRangeKernel(linear_attn_delta_recurrent_f32)");
      if (!use_cpuorder && !use_fused_qk_local &&
          !use_fused_cpu_shape_qk_local) {
        Check(api_.clEnqueueNDRangeKernel(queue_, final_kernel, 1,
                                          nullptr, &final_global, nullptr, 0,
                                          nullptr, EventOut(&final_event)),
              "clEnqueueNDRangeKernel(linear_attn_final_norm_f32)");
      }
      Check(api_.clFinish(queue_), "clFinish(linear attention delta)");
      delta_times.push_back(EventUs(api_, delta_event));
      if (use_cpuorder || use_fused_qk_local ||
          use_fused_cpu_shape_qk_local) {
        final_times.push_back(0.0);
      } else {
        final_times.push_back(EventUs(api_, final_event));
        ReleaseEvent(api_, &final_event);
      }
      ReleaseEvent(api_, &delta_event);
    }

    GpuLinearAttentionDeltaTiming timing;
    timing.delta_min_us = *std::min_element(delta_times.begin(), delta_times.end());
    timing.delta_mean_us =
        std::accumulate(delta_times.begin(), delta_times.end(), 0.0) /
          static_cast<double>(delta_times.size());
    timing.final_min_us = *std::min_element(final_times.begin(), final_times.end());
    timing.final_mean_us =
        std::accumulate(final_times.begin(), final_times.end(), 0.0) /
          static_cast<double>(final_times.size());
    timing.delta_global_work_items = head_dim * value_heads;
    timing.final_global_work_items = value_heads;
    return timing;
  }

  OpenClApi api_;
  SelectedDevice selected_;
  cl_context context_ = nullptr;
  cl_command_queue queue_ = nullptr;
  cl_program program_ = nullptr;
  cl_kernel kernel_group8_ = nullptr;
  cl_kernel kernel_rowlane_ = nullptr;
  cl_kernel kernel_rowblock16_ = nullptr;
  cl_kernel kernel_rowblock16_cpuorder_finalize_ = nullptr;
  cl_kernel kernel_rowlane_localq8_ = nullptr;
  cl_kernel kernel_rowlane_expert8_ = nullptr;
  cl_kernel kernel_rowlane_expert8_localq8_ = nullptr;
  cl_kernel kernel_rowlane_expert8_plus_shared_localq8_ = nullptr;
  cl_kernel kernel_rowlane_topk_indexed_expert8_plus_shared_localq8_ = nullptr;
  cl_kernel kernel_rowlane_expert8_multiq8_ = nullptr;
  cl_kernel kernel_rowlane_expert8_plus_shared_multiq8_ = nullptr;
  cl_kernel kernel_rowlane_expert8_f32input_ = nullptr;
  cl_kernel kernel_q4_cpu_order_ = nullptr;
  cl_kernel kernel_q6_matvec_row_ = nullptr;
  cl_kernel kernel_q6_linear_qkv_cpuorder_ = nullptr;
  cl_kernel kernel_q6_matvec_rowstripe_ = nullptr;
  cl_kernel kernel_q6_matvec_rowstripe_localq8_ = nullptr;
  cl_kernel kernel_q6_matvec_row_expert8_ = nullptr;
  cl_kernel kernel_q6_matvec_rowstripe_expert8_ = nullptr;
  cl_kernel kernel_q6_matvec_rowstripe_expert8_plus_shared_ = nullptr;
  cl_kernel kernel_q6_matvec_rowstripe_expert8_plus_shared_tail_atomic_ = nullptr;
  cl_kernel kernel_q6_matvec_rowstripe_expert8_plus_shared_tail_contrib_ = nullptr;
  cl_kernel kernel_ffn_tail_reduce9_contrib_ = nullptr;
  cl_kernel kernel_f32_topk8_blocks_ = nullptr;
  cl_kernel kernel_f32_topk16_blocks_ = nullptr;
  cl_kernel kernel_qkv_delta_sparse_overlay_ = nullptr;
  cl_kernel kernel_qkv_delta_blockq16_overlay_ = nullptr;
  cl_kernel kernel_f32_matvec_ = nullptr;
  cl_kernel kernel_conv_ = nullptr;
  cl_kernel kernel_conv_cpuorder_ = nullptr;
  cl_kernel kernel_postconv_silu_split_ = nullptr;
  cl_kernel kernel_postconv_silu_split_cpuorder_ = nullptr;
  cl_kernel kernel_postconv_l2_q_ = nullptr;
  cl_kernel kernel_postconv_l2_k_ = nullptr;
  cl_kernel kernel_postconv_l2_qk_ = nullptr;
  cl_kernel kernel_postconv_l2_qk_cpuorder_ = nullptr;
  cl_kernel kernel_postconv_fused_qk_l2_ = nullptr;
  cl_kernel kernel_delta_recurrent_ = nullptr;
  cl_kernel kernel_delta_recurrent_qk_local_ = nullptr;
  cl_kernel kernel_delta_recurrent_final_qk_local_ = nullptr;
  cl_kernel kernel_delta_recurrent_final_cpu_shape_qk_local_ = nullptr;
  cl_kernel kernel_delta_recurrent_final_cpuorder_ = nullptr;
  cl_kernel kernel_delta_final_ = nullptr;
  cl_kernel kernel_delta_final_cpu_shape_ = nullptr;
  cl_kernel kernel_ffn_swiglu_ = nullptr;
  cl_kernel kernel_ffn_swiglu_reorder_ = nullptr;
  cl_kernel kernel_q8_quantize_ = nullptr;
  cl_kernel kernel_q8_quantize_bsums_ = nullptr;
  cl_kernel kernel_ffn_weighted_ = nullptr;
  cl_kernel kernel_ffn_gate_apply_ = nullptr;
  cl_kernel kernel_ffn_output_add_ = nullptr;
  cl_kernel kernel_post_ffn_residual_ = nullptr;
  cl_kernel kernel_ffn_tail_init_residual_bits_ = nullptr;
  cl_kernel kernel_ffn_tail_reduce_down_atomic_ = nullptr;
  cl_kernel kernel_ffn_tail_fused_output_ = nullptr;
  cl_kernel kernel_rmsnorm_hidden_ = nullptr;
  cl_kernel kernel_rmsnorm_hidden_scale_ = nullptr;
  cl_kernel kernel_rmsnorm_hidden_apply_scale_ = nullptr;
  cl_kernel kernel_full_attn_core_ = nullptr;
  cl_kernel kernel_full_attn_score_ = nullptr;
  cl_kernel kernel_full_attn_apply_score_gate_ = nullptr;
  cl_kernel kernel_full_attn_gate_ = nullptr;
  cl_kernel kernel_full_attn_qk_norm_rope_ = nullptr;
  std::string build_log_;
  double program_build_ms_ = 0.0;
  ScratchBuffer expert8_swiglu_scratch_q8_qs_;
  ScratchBuffer expert8_swiglu_scratch_q8_bsums_;
  ScratchBuffer expert8_swiglu_scratch_q8_d_;
  ScratchBuffer expert8_swiglu_scratch_topk_indices_;
  ScratchBuffer expert8_swiglu_scratch_gate_up_;
  ScratchBuffer expert8_swiglu_scratch_swiglu_;
  ScratchBuffer expert8_q4_down_scratch_q8_qs_;
  ScratchBuffer expert8_q4_down_scratch_q8_bsums_;
  ScratchBuffer expert8_q4_down_scratch_q8_d_;
  ScratchBuffer expert8_q4_down_scratch_out_;
  std::uint64_t expert8_q4_down_output_alias_handle_ = 0;
  ScratchBuffer selected_shared_q4_down_scratch_selected_q8_qs_;
  ScratchBuffer selected_shared_q4_down_scratch_selected_q8_bsums_;
  ScratchBuffer selected_shared_q4_down_scratch_selected_q8_d_;
  ScratchBuffer selected_shared_q4_down_scratch_shared_q8_qs_;
  ScratchBuffer selected_shared_q4_down_scratch_shared_q8_bsums_;
  ScratchBuffer selected_shared_q4_down_scratch_shared_q8_d_;
  ScratchBuffer selected_shared_q4_down_scratch_selected_out_;
  ScratchBuffer selected_shared_q4_down_scratch_shared_out_;
  std::uint64_t selected_shared_q4_down_selected_output_alias_handle_ = 0;
  std::uint64_t selected_shared_q4_down_shared_output_alias_handle_ = 0;
  ScratchBuffer swiglu_handoff_scratch_q8_qs_;
  ScratchBuffer swiglu_handoff_scratch_q8_bsums_;
  ScratchBuffer swiglu_handoff_scratch_q8_d_;
  ScratchBuffer swiglu_handoff_scratch_gate_up_;
  ScratchBuffer swiglu_handoff_scratch_source_map_;
  ScratchBuffer swiglu_handoff_scratch_swiglu_;
  ScratchBuffer expert8_q6_down_scratch_q8_qs_;
  ScratchBuffer expert8_q6_down_scratch_q8_d_;
  ScratchBuffer expert8_q6_down_scratch_out_;
  std::uint64_t expert8_q6_down_output_alias_handle_ = 0;
  ScratchBuffer selected_shared_q6_down_scratch_selected_q8_qs_;
  ScratchBuffer selected_shared_q6_down_scratch_selected_q8_d_;
  ScratchBuffer selected_shared_q6_down_scratch_shared_q8_qs_;
  ScratchBuffer selected_shared_q6_down_scratch_shared_q8_d_;
  ScratchBuffer selected_shared_q6_down_scratch_selected_out_;
  ScratchBuffer selected_shared_q6_down_scratch_shared_out_;
  std::uint64_t selected_shared_q6_down_selected_output_alias_handle_ = 0;
  std::uint64_t selected_shared_q6_down_shared_output_alias_handle_ = 0;
  ScratchBuffer resident_q6_handoff_scratch_q8_qs_;
  ScratchBuffer resident_q6_handoff_scratch_q8_d_;
  ScratchBuffer resident_q6_handoff_scratch_out_;
  std::uint64_t resident_q6_handoff_output_alias_handle_ = 0;
  ScratchBuffer resident_selected_q6_handoff_scratch_q8_qs_;
  ScratchBuffer resident_selected_q6_handoff_scratch_q8_d_;
  ScratchBuffer resident_selected_q6_handoff_scratch_out_;
  std::uint64_t resident_selected_q6_handoff_output_alias_handle_ = 0;
  ScratchBuffer q6_conv_state_scratch_q8_qs_;
  ScratchBuffer q6_conv_state_scratch_q8_d_;
  ScratchBuffer q6_conv_state_scratch_qkv_;
  ScratchBuffer q6_conv_state_scratch_conv_output_;
  ScratchBuffer q6_conv_state_scratch_next_state_;
  std::uint64_t q6_conv_state_output_alias_handle_ = 0;
  ScratchBuffer packed_conv_state_scratch_q8_qs_;
  ScratchBuffer packed_conv_state_scratch_q8_bsums_;
  ScratchBuffer packed_conv_state_scratch_q8_d_;
  ScratchBuffer packed_conv_state_scratch_qkv_;
  ScratchBuffer packed_conv_state_scratch_conv_output_;
  ScratchBuffer packed_conv_state_scratch_next_state_;
  std::uint64_t packed_conv_state_output_alias_handle_ = 0;
  ScratchBuffer postconv_prep_scratch_raw_;
  ScratchBuffer postconv_prep_scratch_silu_;
  ScratchBuffer postconv_prep_scratch_q_;
  ScratchBuffer postconv_prep_scratch_k_;
  ScratchBuffer postconv_prep_scratch_v_;
  ScratchBuffer postconv_prep_scratch_q_norm_;
  ScratchBuffer postconv_prep_scratch_k_norm_;
  ScratchBuffer linear_delta_scratch_q_;
  ScratchBuffer linear_delta_scratch_k_;
  ScratchBuffer linear_delta_scratch_v_;
  ScratchBuffer linear_delta_scratch_gate_;
  ScratchBuffer linear_delta_scratch_beta_;
  ScratchBuffer linear_delta_scratch_z_;
  ScratchBuffer linear_delta_scratch_norm_;
  ScratchBuffer linear_delta_scratch_attention_;
  ScratchBuffer linear_delta_scratch_final_;
  std::uint64_t linear_delta_final_alias_handle_ = 0;
  ScratchBuffer f32_input_q4_scratch_q8_qs_;
  ScratchBuffer f32_input_q4_scratch_q8_bsums_;
  ScratchBuffer f32_input_q4_scratch_q8_d_;
  ScratchBuffer f32_input_q4_scratch_output_;
  std::uint64_t f32_input_q4_output_alias_handle_ = 0;
  ScratchBuffer q4_cpu_order_scratch_q8_qs_;
  ScratchBuffer q4_cpu_order_scratch_q8_bsums_;
  ScratchBuffer q4_cpu_order_scratch_q8_d_;
  ScratchBuffer q4_cpu_order_scratch_out_;
  ScratchBuffer f32_input_q6_scratch_q8_qs_;
  ScratchBuffer f32_input_q6_scratch_q8_d_;
  ScratchBuffer f32_input_q6_scratch_output_;
  std::uint64_t f32_input_q6_output_alias_handle_ = 0;
  ScratchBuffer linear_preconv_shared_q8_qs_;
  ScratchBuffer linear_preconv_shared_q8_bsums_;
  ScratchBuffer linear_preconv_shared_q8_d_;
  ScratchBuffer linear_preconv_shared_qkv_;
  ScratchBuffer linear_preconv_shared_conv_output_;
  ScratchBuffer linear_preconv_shared_next_state_;
  ScratchBuffer linear_preconv_shared_alpha_beta_z_;
  std::uint64_t linear_preconv_shared_conv_output_alias_handle_ = 0;
  ScratchBuffer ffn_tail_scratch_down_;
  ScratchBuffer ffn_tail_scratch_weights_;
  ScratchBuffer ffn_tail_scratch_weighted_;
  ScratchBuffer ffn_tail_scratch_moe_out_;
  ScratchBuffer ffn_tail_scratch_gate_weights_;
  ScratchBuffer ffn_tail_scratch_attn_post_norm_;
  ScratchBuffer ffn_tail_scratch_ffn_shexp_;
  ScratchBuffer ffn_tail_scratch_shared_gate_;
  ScratchBuffer ffn_tail_scratch_shared_sigmoid_;
  ScratchBuffer ffn_tail_scratch_shared_gated_;
  ScratchBuffer ffn_tail_scratch_ffn_out_;
  ScratchBuffer ffn_tail_scratch_attn_residual_;
  ScratchBuffer ffn_tail_scratch_layer_output_;
  ScratchBuffer ffn_tail_scratch_contrib_;
  ScratchBuffer attention_front_scratch_projection_;
  ScratchBuffer attention_front_scratch_residual_input_;
  ScratchBuffer attention_front_scratch_residual_;
  ScratchBuffer attention_front_scratch_normalized_;
  ScratchBuffer attention_front_scratch_q8_qs_;
  ScratchBuffer attention_front_scratch_q8_bsums_;
  ScratchBuffer attention_front_scratch_q8_d_;
  ScratchBuffer attention_front_scratch_norm_weight_;
  ScratchBuffer rmsnorm_hidden_scratch_input_;
  ScratchBuffer rmsnorm_hidden_scratch_weight_;
  ScratchBuffer rmsnorm_hidden_scratch_output_;
  ScratchBuffer rmsnorm_hidden_scratch_scale_;
  std::uint64_t rmsnorm_hidden_output_alias_handle_ = 0;
  ScratchBuffer full_core_handoff_scratch_q_rope_;
  ScratchBuffer full_core_handoff_scratch_k_history_;
  ScratchBuffer full_core_handoff_scratch_v_history_;
  ScratchBuffer full_core_handoff_scratch_q_full_;
  ScratchBuffer full_core_qk_norm_rope_scratch_q_rope_;
  ScratchBuffer full_core_qk_norm_rope_scratch_k_rope_;
  ScratchBuffer full_core_qk_norm_rope_scratch_rope_cache_;
  std::uint64_t full_core_q_rope_alias_handle_ = 0;
  std::uint64_t full_core_k_rope_alias_handle_ = 0;
  ScratchBuffer full_core_handoff_scratch_scores_;
  ScratchBuffer full_core_handoff_scratch_pregate_;
  ScratchBuffer full_core_handoff_scratch_gated_;
  ScratchBuffer full_core_handoff_scratch_q8_qs_;
  ScratchBuffer full_core_handoff_scratch_q8_bsums_;
  ScratchBuffer full_core_handoff_scratch_q8_d_;
  ScratchBuffer full_core_handoff_scratch_projection_;
  ScratchBuffer full_core_handoff_scratch_residual_input_;
  ScratchBuffer full_core_handoff_scratch_norm_weight_;
  ScratchBuffer full_core_handoff_scratch_residual_;
  ScratchBuffer full_core_handoff_scratch_normalized_;
  std::vector<std::vector<std::uint8_t>> pending_host_uploads_;
  std::unordered_map<std::uint64_t, ResidentPackedQ4X8> resident_packed_q4x8_;
  std::unordered_map<std::uint64_t, ResidentRawQ6K> resident_raw_q6k_;
  std::unordered_map<std::uint64_t, ResidentRawQ4KCpuOrder>
      resident_raw_q4_cpu_order_;
  std::unordered_map<std::uint64_t, ResidentConvWeights> resident_conv_weights_;
  std::unordered_map<std::uint64_t, ResidentF32Matvec> resident_f32_matvec_;
  std::unordered_map<std::uint64_t, ResidentF32Buffer> resident_f32_buffers_;
  ScratchBuffer qkv_delta_sparse_overlay_scratch_indices_;
  ScratchBuffer qkv_delta_blockq16_overlay_scratch_indices_;
  ScratchBuffer qkv_delta_blockq16_overlay_scratch_q_delta_;
  ScratchBuffer qkv_delta_blockq16_overlay_scratch_scales_;
  ScratchBuffer resident_f32_matvec_scratch_output_;
  std::uint64_t resident_f32_matvec_output_alias_handle_ = 0;
  std::uint64_t ffn_tail_layer_output_alias_handle_ = 0;
  std::uint64_t attention_front_residual_alias_handle_ = 0;
  std::uint64_t attention_front_normalized_alias_handle_ = 0;
  std::uint64_t next_resident_handle_ = 1;
};

GpuQ4X8MatvecRunner::GpuQ4X8MatvecRunner(std::string device_substring,
                                         std::string opencl_source)
    : impl_(std::make_unique<Impl>(std::move(device_substring), opencl_source)) {}

GpuQ4X8MatvecRunner::~GpuQ4X8MatvecRunner() = default;
GpuQ4X8MatvecRunner::GpuQ4X8MatvecRunner(GpuQ4X8MatvecRunner&&) noexcept = default;
GpuQ4X8MatvecRunner& GpuQ4X8MatvecRunner::operator=(GpuQ4X8MatvecRunner&&) noexcept = default;

const std::string& GpuQ4X8MatvecRunner::platform_name() const { return impl_->platform_name(); }
const std::string& GpuQ4X8MatvecRunner::device_name() const { return impl_->device_name(); }
const std::string& GpuQ4X8MatvecRunner::build_log() const { return impl_->build_log(); }
double GpuQ4X8MatvecRunner::program_build_ms() const { return impl_->program_build_ms(); }

GpuQ4X8MatvecRun GpuQ4X8MatvecRunner::Run(const std::vector<std::uint8_t>& packed,
                                          const std::vector<std::int8_t>& q8_qs,
                                          const std::vector<std::int16_t>& q8_bsums,
                                          const std::vector<float>& q8_d,
                                          std::uint64_t rows,
                                          std::uint64_t blocks_per_row,
                                          int repeat,
                                          GpuQ4X8KernelVariant variant) {
  return impl_->Run(packed, q8_qs, q8_bsums, q8_d, rows, blocks_per_row, repeat, variant);
}

GpuQ4X8MatvecRun GpuQ4X8MatvecRunner::RunRowblock16(
    const std::vector<std::uint8_t>& packed,
    const std::vector<std::int8_t>& q8_qs,
    const std::vector<std::int16_t>& q8_bsums,
    const std::vector<float>& q8_d,
    std::uint64_t rows,
    std::uint64_t blocks_per_row,
    int repeat) {
  return impl_->RunRowblock16(packed, q8_qs, q8_bsums, q8_d, rows,
                              blocks_per_row, repeat);
}

GpuQ4X8MatvecRun GpuQ4X8MatvecRunner::RunRowblock16CpuOrderFinalize(
    const std::vector<std::uint8_t>& packed,
    const std::vector<std::int8_t>& q8_qs,
    const std::vector<std::int16_t>& q8_bsums,
    const std::vector<float>& q8_d,
    std::uint64_t rows,
    std::uint64_t blocks_per_row,
    int repeat) {
  return impl_->RunRowblock16CpuOrderFinalize(
      packed, q8_qs, q8_bsums, q8_d, rows, blocks_per_row, repeat);
}

std::uint64_t GpuQ4X8MatvecRunner::UploadPackedQ4X8(
    const std::vector<std::uint8_t>& packed,
    std::uint64_t rows,
    std::uint64_t blocks_per_row) {
  return impl_->UploadPackedQ4X8(packed, rows, blocks_per_row);
}

std::uint64_t GpuQ4X8MatvecRunner::UploadPackedQ4X8Deferred(
    const std::vector<std::uint8_t>& packed,
    std::uint64_t rows,
    std::uint64_t blocks_per_row) {
  return impl_->UploadPackedQ4X8Deferred(packed, rows, blocks_per_row);
}

std::uint64_t GpuQ4X8MatvecRunner::ConcatResidentPackedQ4X8(
    const std::vector<std::uint64_t>& handles) {
  return impl_->ConcatResidentPackedQ4X8(handles);
}

std::uint64_t GpuQ4X8MatvecRunner::UploadRawQ4KCpuOrder(
    const std::vector<std::uint8_t>& raw,
    std::uint64_t rows,
    std::uint64_t blocks_per_row) {
  return impl_->UploadRawQ4KCpuOrder(raw, rows, blocks_per_row);
}

GpuQ4X8MatvecRun GpuQ4X8MatvecRunner::RunResidentPackedQ4X8(
    std::uint64_t handle,
    const std::vector<std::int8_t>& q8_qs,
    const std::vector<std::int16_t>& q8_bsums,
    const std::vector<float>& q8_d,
    int repeat,
    GpuQ4X8KernelVariant variant) {
  return impl_->RunResidentPackedQ4X8(handle, q8_qs, q8_bsums, q8_d, repeat, variant);
}

GpuDeviceQ8Q4X8MatvecRun
GpuQ4X8MatvecRunner::RunF32InputHandleDeviceQ8ThenResidentPackedQ4X8(
    std::uint64_t handle,
    std::uint64_t input_handle,
    int repeat,
    GpuQ4X8KernelVariant variant,
    bool readback_output) {
  return impl_->RunF32InputHandleDeviceQ8ThenResidentPackedQ4X8(
      handle, input_handle, repeat, variant, readback_output);
}

GpuQ4KCpuOrderMatvecRun
GpuQ4X8MatvecRunner::RunF32InputHandleDeviceQ8ThenResidentRawQ4KCpuOrder(
    std::uint64_t handle,
    std::uint64_t input_handle,
    int repeat,
    bool readback_output) {
  return impl_->RunF32InputHandleDeviceQ8ThenResidentRawQ4KCpuOrder(
      handle, input_handle, repeat, readback_output);
}

GpuQ4X8MatvecRun
GpuQ4X8MatvecRunner::RunResidentPackedQ4X8Expert8PerExpertQ8(
    const std::vector<std::uint64_t>& handles,
    const std::vector<std::int8_t>& q8_qs,
    const std::vector<std::int16_t>& q8_bsums,
    const std::vector<float>& q8_d,
    int repeat,
    GpuQ4X8KernelVariant variant,
    bool readback_output) {
  return impl_->RunResidentPackedQ4X8Expert8PerExpertQ8(
      handles, q8_qs, q8_bsums, q8_d, repeat, variant,
      readback_output);
}

GpuQ4X8SelectedSharedMatvecRun
GpuQ4X8MatvecRunner::RunResidentPackedQ4X8Expert8PlusSharedPerExpertQ8(
    const std::vector<std::uint64_t>& selected_handles,
    std::uint64_t shared_handle,
    const GpuQ8KInputPlanes& selected_q8,
    const GpuQ8KInputPlanes& shared_q8,
    int repeat,
    bool readback_selected_output,
    bool readback_shared_output) {
  return impl_->RunResidentPackedQ4X8Expert8PlusSharedPerExpertQ8(
      selected_handles, shared_handle, selected_q8, shared_q8, repeat,
      readback_selected_output, readback_shared_output);
}

std::uint64_t GpuQ4X8MatvecRunner::UploadRawQ6K(
    const std::vector<std::uint8_t>& raw,
    std::uint64_t rows,
    std::uint64_t blocks_per_row) {
  return impl_->UploadRawQ6K(raw, rows, blocks_per_row);
}

std::uint64_t GpuQ4X8MatvecRunner::UploadRawQ6KDeferred(
    const std::vector<std::uint8_t>& raw,
    std::uint64_t rows,
    std::uint64_t blocks_per_row) {
  return impl_->UploadRawQ6KDeferred(raw, rows, blocks_per_row);
}

std::uint64_t GpuQ4X8MatvecRunner::UploadSelectedRawQ6KRowstripe(
    const std::vector<std::uint8_t>& raw,
    std::uint64_t rows_per_expert,
    std::uint64_t blocks_per_row,
    std::uint64_t selected_count,
    std::uint64_t rows_per_tile) {
  return impl_->UploadSelectedRawQ6KRowstripe(
      raw, rows_per_expert, blocks_per_row, selected_count, rows_per_tile);
}

std::uint64_t GpuQ4X8MatvecRunner::UploadSelectedRawQ6KRowstripeDeferred(
    const std::vector<std::uint8_t>& raw,
    std::uint64_t rows_per_expert,
    std::uint64_t blocks_per_row,
    std::uint64_t selected_count,
    std::uint64_t rows_per_tile) {
  return impl_->UploadSelectedRawQ6KRowstripeDeferred(
      raw, rows_per_expert, blocks_per_row, selected_count, rows_per_tile);
}

std::uint64_t GpuQ4X8MatvecRunner::ConcatResidentRawQ6K(
    const std::vector<std::uint64_t>& handles) {
  return impl_->ConcatResidentRawQ6K(handles);
}

GpuQ6KMatvecRun GpuQ4X8MatvecRunner::RunResidentRawQ6K(
    std::uint64_t handle,
    const GpuQ8KInputPlanes& q8,
    int repeat) {
  return impl_->RunResidentRawQ6K(handle, q8, repeat);
}

GpuQ6KMatvecRun GpuQ4X8MatvecRunner::RunResidentRawQ6KToF32Handle(
    std::uint64_t handle,
    const GpuQ8KInputPlanes& q8,
    int repeat) {
  return impl_->RunResidentRawQ6KToF32Handle(handle, q8, repeat);
}

GpuDeviceQ8Q6KMatvecRun
GpuQ4X8MatvecRunner::RunF32InputHandleDeviceQ8ThenResidentRawQ6K(
    std::uint64_t handle,
    std::uint64_t input_handle,
    int repeat,
    bool readback_output) {
  return impl_->RunF32InputHandleDeviceQ8ThenResidentRawQ6K(
      handle, input_handle, repeat, readback_output);
}

GpuQ6KTopKRun GpuQ4X8MatvecRunner::RunResidentRawQ6KTopK(
    std::uint64_t handle,
    const GpuQ8KInputPlanes& q8,
    int topk,
    int repeat) {
  return impl_->RunResidentRawQ6KTopK(handle, q8, topk, repeat);
}

GpuQ6KMatvecRun GpuQ4X8MatvecRunner::RunResidentRawQ6KSelected(
    std::uint64_t handle,
    const GpuQ8KInputPlanes& q8,
    std::uint64_t rows_per_expert,
    std::uint64_t selected_count,
    int repeat,
    bool readback_output) {
  return impl_->RunResidentRawQ6KSelected(
      handle, q8, rows_per_expert, selected_count, repeat, readback_output);
}

GpuQ6KMatvecRun GpuQ4X8MatvecRunner::RunResidentRawQ6KExpert8(
    const std::vector<std::uint64_t>& handles,
    const GpuQ8KInputPlanes& q8,
    std::uint64_t rows_per_expert,
    int repeat,
    bool readback_output) {
  return impl_->RunResidentRawQ6KExpert8(handles, q8, rows_per_expert, repeat,
                                         readback_output);
}

GpuQ6KSelectedSharedMatvecRun
GpuQ4X8MatvecRunner::RunResidentRawQ6KExpert8PlusShared(
    const std::vector<std::uint64_t>& selected_handles,
    std::uint64_t shared_handle,
    const GpuQ8KInputPlanes& selected_q8,
    const GpuQ8KInputPlanes& shared_q8,
    std::uint64_t rows_per_expert,
    int repeat,
    bool readback_selected_output,
    bool readback_shared_output) {
  return impl_->RunResidentRawQ6KExpert8PlusShared(
      selected_handles, shared_handle, selected_q8, shared_q8, rows_per_expert,
      repeat, readback_selected_output, readback_shared_output);
}

GpuFfnTailRun
GpuQ4X8MatvecRunner::RunResidentRawQ6KExpert8PlusSharedToFfnTailAtomic(
    const std::vector<std::uint64_t>& selected_handles,
    std::uint64_t shared_handle,
    const GpuQ8KInputPlanes& selected_q8,
    const GpuQ8KInputPlanes& shared_q8,
    std::uint64_t shared_gate_matvec_handle,
    std::uint64_t attn_post_norm_handle,
    const std::vector<float>& weights_norm,
    std::uint64_t attn_residual_handle,
    std::uint64_t rows_per_expert,
    int repeat,
    bool readback_layer_output) {
  return impl_->RunResidentRawQ6KExpert8PlusSharedToFfnTailAtomic(
      selected_handles, shared_handle, selected_q8, shared_q8,
      shared_gate_matvec_handle, attn_post_norm_handle, weights_norm,
      attn_residual_handle, rows_per_expert, repeat, readback_layer_output);
}

GpuFfnTailRun
GpuQ4X8MatvecRunner::RunResidentRawQ6KExpert8PlusSharedToFfnTailNonAtomic(
    const std::vector<std::uint64_t>& selected_handles,
    std::uint64_t shared_handle,
    const GpuQ8KInputPlanes& selected_q8,
    const GpuQ8KInputPlanes& shared_q8,
    std::uint64_t shared_gate_matvec_handle,
    std::uint64_t attn_post_norm_handle,
    const std::vector<float>& weights_norm,
    std::uint64_t attn_residual_handle,
    std::uint64_t rows_per_expert,
    int repeat,
    bool readback_layer_output) {
  return impl_->RunResidentRawQ6KExpert8PlusSharedToFfnTailNonAtomic(
      selected_handles, shared_handle, selected_q8, shared_q8,
      shared_gate_matvec_handle, attn_post_norm_handle, weights_norm,
      attn_residual_handle, rows_per_expert, repeat, readback_layer_output);
}

GpuRmsNormQ6MatvecRun GpuQ4X8MatvecRunner::RunRmsNormThenResidentRawQ6K(
    const std::vector<float>& input,
    std::uint64_t norm_weight_handle,
    std::uint64_t q6_handle,
    std::uint64_t hidden_size,
    float norm_epsilon,
    int repeat) {
  return impl_->RunRmsNormThenResidentRawQ6K(
      input, norm_weight_handle, q6_handle, hidden_size, norm_epsilon, repeat);
}

GpuQ4X8ConvHandoffRun GpuQ4X8MatvecRunner::RunResidentRawQ6KThenResidentConv(
    std::uint64_t q6_handle,
    const GpuQ8KInputPlanes& q8,
    std::uint64_t conv_weights_handle,
    const std::vector<float>& conv_state,
    std::uint64_t conv_kernel_size,
    int repeat) {
  return impl_->RunResidentRawQ6KThenResidentConv(
      q6_handle, q8, conv_weights_handle, conv_state, conv_kernel_size,
      repeat);
}

GpuQ4X8ConvHandoffRun
GpuQ4X8MatvecRunner::RunResidentRawQ6KThenResidentConvState(
    std::uint64_t q6_handle,
    const GpuQ8KInputPlanes& q8,
    std::uint64_t conv_weights_handle,
    std::uint64_t conv_state_handle,
    std::uint64_t conv_kernel_size,
    int repeat,
    bool readback_state,
    std::uint64_t next_conv_state_handle,
    bool readback_qkv,
    bool readback_conv_output) {
  return impl_->RunResidentRawQ6KThenResidentConvState(
      q6_handle, q8, conv_weights_handle, conv_state_handle,
      conv_kernel_size, repeat, readback_state, next_conv_state_handle,
      readback_qkv, readback_conv_output);
}

GpuQ4X8ConvHandoffRun
GpuQ4X8MatvecRunner::RunResidentRawQ6KThenResidentConvStateCpuOrder(
    std::uint64_t q6_handle,
    const GpuQ8KInputPlanes& q8,
    std::uint64_t conv_weights_handle,
    std::uint64_t conv_state_handle,
    std::uint64_t conv_kernel_size,
    int repeat,
    bool readback_state,
    std::uint64_t next_conv_state_handle,
    bool readback_qkv,
    bool readback_conv_output) {
  return impl_->RunResidentRawQ6KThenResidentConvState(
      q6_handle, q8, conv_weights_handle, conv_state_handle,
      conv_kernel_size, repeat, readback_state, next_conv_state_handle,
      readback_qkv, readback_conv_output, true);
}

GpuQ4X8ConvHandoffRun GpuQ4X8MatvecRunner::
    RunF32InputHandleDeviceQ8ThenResidentRawQ6KThenResidentConvState(
        std::uint64_t q6_handle,
        std::uint64_t input_handle,
        std::uint64_t conv_weights_handle,
        std::uint64_t conv_state_handle,
        std::uint64_t conv_kernel_size,
        int repeat,
        bool readback_state,
        std::uint64_t next_conv_state_handle,
        bool readback_qkv,
        bool readback_conv_output) {
  return impl_->RunF32InputHandleDeviceQ8ThenResidentRawQ6KThenResidentConvState(
      q6_handle, input_handle, conv_weights_handle, conv_state_handle,
      conv_kernel_size, repeat, readback_state, next_conv_state_handle,
      readback_qkv, readback_conv_output);
}

GpuLinearPreconvSharedQ8Run GpuQ4X8MatvecRunner::
    RunF32InputHandleSharedDeviceQ8ThenResidentRawQ6KConvStateAndResidentRawQ4KCpuOrder(
        std::uint64_t q6_handle,
        std::uint64_t alpha_beta_z_handle,
        std::uint64_t input_handle,
        std::uint64_t conv_weights_handle,
        std::uint64_t conv_state_handle,
        std::uint64_t conv_kernel_size,
        int repeat,
        bool readback_state,
        std::uint64_t next_conv_state_handle,
        bool readback_qkv,
        bool readback_conv_output,
        bool readback_alpha_beta_z) {
  return impl_
      ->RunF32InputHandleSharedDeviceQ8ThenResidentRawQ6KConvStateAndResidentRawQ4KCpuOrder(
          q6_handle, alpha_beta_z_handle, input_handle, conv_weights_handle,
          conv_state_handle, conv_kernel_size, repeat, readback_state,
          next_conv_state_handle, readback_qkv, readback_conv_output,
          readback_alpha_beta_z);
}

void GpuQ4X8MatvecRunner::ClearResidentRawQ6K() {
  impl_->ClearResidentRawQ6K();
}

GpuQ4X8ConvHandoffRun GpuQ4X8MatvecRunner::RunResidentPackedQ4X8ThenConv(
    std::uint64_t handle,
    const std::vector<std::int8_t>& q8_qs,
    const std::vector<std::int16_t>& q8_bsums,
    const std::vector<float>& q8_d,
    const std::vector<float>& conv_weights,
    const std::vector<float>& conv_state,
    std::uint64_t conv_kernel_size,
    int repeat,
    GpuQ4X8KernelVariant variant) {
  return impl_->RunResidentPackedQ4X8ThenConv(handle, q8_qs, q8_bsums, q8_d,
                                              conv_weights, conv_state,
                                              conv_kernel_size, repeat, variant);
}

GpuQ4X8SwiGluHandoffRun GpuQ4X8MatvecRunner::RunResidentPackedQ4X8ThenSwiGlu(
    std::uint64_t handle,
    const std::vector<std::int8_t>& q8_qs,
    const std::vector<std::int16_t>& q8_bsums,
    const std::vector<float>& q8_d,
    std::uint64_t intermediate_size,
    const std::vector<std::uint32_t>& source_expert_by_output,
    int repeat,
    GpuQ4X8KernelVariant variant) {
  return impl_->RunResidentPackedQ4X8ThenSwiGlu(
      handle, q8_qs, q8_bsums, q8_d, intermediate_size,
      source_expert_by_output, repeat, variant);
}

GpuQ4X8SwiGluHandoffRun
GpuQ4X8MatvecRunner::RunResidentPackedQ4X8ThenSwiGluWithLastExpert8Q8(
    std::uint64_t handle,
    std::uint64_t intermediate_size,
    const std::vector<std::uint32_t>& source_expert_by_output,
    int repeat,
    GpuQ4X8KernelVariant variant) {
  return impl_->RunResidentPackedQ4X8ThenSwiGluWithLastExpert8Q8(
      handle, intermediate_size, source_expert_by_output, repeat, variant);
}

GpuQ4X8SwiGluHandoffRun
GpuQ4X8MatvecRunner::RunResidentPackedQ4X8Expert8ThenSwiGlu(
    const std::vector<std::uint64_t>& handles,
    const std::vector<std::int8_t>& q8_qs,
    const std::vector<std::int16_t>& q8_bsums,
    const std::vector<float>& q8_d,
    std::uint64_t intermediate_size,
    int repeat,
    GpuQ4X8KernelVariant variant) {
  return impl_->RunResidentPackedQ4X8Expert8ThenSwiGlu(
      handles, q8_qs, q8_bsums, q8_d, intermediate_size, repeat, variant);
}

GpuQ4X8SwiGluHandoffRun
GpuQ4X8MatvecRunner::RunResidentPackedQ4X8Expert8PlusSharedThenSwiGlu(
    const std::vector<std::uint64_t>& selected_handles,
    std::uint64_t shared_handle,
    const std::vector<std::int8_t>& q8_qs,
    const std::vector<std::int16_t>& q8_bsums,
    const std::vector<float>& q8_d,
    std::uint64_t intermediate_size,
    int repeat,
    GpuQ4X8KernelVariant variant) {
  return impl_->RunResidentPackedQ4X8Expert8PlusSharedThenSwiGlu(
      selected_handles, shared_handle, q8_qs, q8_bsums, q8_d,
      intermediate_size, repeat, variant);
}

GpuQ4X8SwiGluHandoffRun
GpuQ4X8MatvecRunner::
    RunResidentPackedQ4X8TopKIndexedExpert8PlusSharedThenSwiGlu(
        std::uint64_t selected_material_handle,
        std::uint64_t shared_handle,
        const std::vector<std::uint32_t>& selected_positions,
        const std::vector<std::int8_t>& q8_qs,
        const std::vector<std::int16_t>& q8_bsums,
        const std::vector<float>& q8_d,
        std::uint64_t intermediate_size,
        int repeat,
        GpuQ4X8KernelVariant variant) {
  return impl_->RunResidentPackedQ4X8TopKIndexedExpert8PlusSharedThenSwiGlu(
      selected_material_handle, shared_handle, selected_positions, q8_qs,
      q8_bsums, q8_d, intermediate_size, repeat, variant);
}

GpuQ4X8SwiGluQ6DownHandoffRun
GpuQ4X8MatvecRunner::RunResidentPackedQ4X8ThenSwiGluThenRawQ6KSelected(
    std::uint64_t packed_handle,
    std::uint64_t q6_handle,
    const std::vector<std::int8_t>& q8_qs,
    const std::vector<std::int16_t>& q8_bsums,
    const std::vector<float>& q8_d,
    std::uint64_t intermediate_size,
    const std::vector<std::uint32_t>& source_expert_by_output,
    std::uint64_t rows_per_expert,
    int repeat,
    GpuQ4X8KernelVariant variant) {
  return impl_->RunResidentPackedQ4X8ThenSwiGluThenRawQ6KSelected(
      packed_handle, q6_handle, q8_qs, q8_bsums, q8_d, intermediate_size,
      source_expert_by_output, rows_per_expert, repeat, variant);
}

GpuQ4X8SwiGluQ6DownHandoffRun
GpuQ4X8MatvecRunner::
    RunResidentPackedQ4X8Expert8ThenSwiGluThenRawQ6KExpert8(
        const std::vector<std::uint64_t>& packed_handles,
        const std::vector<std::uint64_t>& q6_handles,
        const std::vector<std::int8_t>& q8_qs,
        const std::vector<std::int16_t>& q8_bsums,
    const std::vector<float>& q8_d,
    std::uint64_t intermediate_size,
    std::uint64_t rows_per_expert,
    int repeat,
    GpuQ4X8KernelVariant variant) {
  return impl_->RunResidentPackedQ4X8Expert8ThenSwiGluThenRawQ6KExpert8(
      packed_handles, q6_handles, q8_qs, q8_bsums, q8_d, intermediate_size,
      rows_per_expert, repeat, variant);
}

GpuQ4X8SwiGluQ4F32DownHandoffRun
GpuQ4X8MatvecRunner::
    RunResidentPackedQ4X8Expert8ThenSwiGluThenPackedQ4X8Expert8F32Input(
        const std::vector<std::uint64_t>& gate_up_handles,
        const std::vector<std::uint64_t>& down_handles,
        const std::vector<std::int8_t>& q8_qs,
        const std::vector<std::int16_t>& q8_bsums,
        const std::vector<float>& q8_d,
        std::uint64_t intermediate_size,
        std::uint64_t rows_per_expert,
        int repeat,
        GpuQ4X8KernelVariant variant,
        bool readback_output) {
  return impl_
      ->RunResidentPackedQ4X8Expert8ThenSwiGluThenPackedQ4X8Expert8F32Input(
          gate_up_handles, down_handles, q8_qs, q8_bsums, q8_d,
          intermediate_size, rows_per_expert, repeat, variant,
          readback_output);
}

GpuQ4X8ResidualRmsNormHandoffRun
GpuQ4X8MatvecRunner::RunResidentPackedQ4X8ThenResidualRmsNorm(
    std::uint64_t handle,
    const std::vector<std::int8_t>& q8_qs,
    const std::vector<std::int16_t>& q8_bsums,
    const std::vector<float>& q8_d,
    const std::vector<float>& residual_input,
    const std::vector<float>& norm_weight,
    float norm_epsilon,
    int repeat,
    GpuQ4X8KernelVariant variant,
    bool use_rowblock16_output_projection) {
  return impl_->RunResidentPackedQ4X8ThenResidualRmsNorm(
      handle, q8_qs, q8_bsums, q8_d, residual_input, norm_weight,
      norm_epsilon, repeat, variant, use_rowblock16_output_projection);
}

GpuQ4X8ResidualRmsNormHandoffRun
GpuQ4X8MatvecRunner::RunF32DeviceQ8ThenResidentPackedQ4X8ResidualRmsNorm(
    std::uint64_t handle,
    const std::vector<float>& input,
    const std::vector<float>& residual_input,
    const std::vector<float>& norm_weight,
    float norm_epsilon,
    int repeat,
    GpuQ4X8KernelVariant variant) {
  return impl_->RunF32DeviceQ8ThenResidentPackedQ4X8ResidualRmsNorm(
      handle, input, residual_input, norm_weight, norm_epsilon, repeat,
      variant);
}

GpuQ4X8ResidualRmsNormHandoffRun
GpuQ4X8MatvecRunner::RunF32InputHandleDeviceQ8ThenResidentPackedQ4X8ResidualRmsNorm(
    std::uint64_t handle,
    std::uint64_t input_handle,
    const std::vector<float>& residual_input,
    const std::vector<float>& norm_weight,
    std::uint64_t norm_weight_handle,
    float norm_epsilon,
    int repeat,
    GpuQ4X8KernelVariant variant,
    std::uint64_t residual_input_handle,
    bool use_rowblock16_cpuorder_finalize) {
  return impl_->RunF32InputHandleDeviceQ8ThenResidentPackedQ4X8ResidualRmsNorm(
      handle, input_handle, residual_input, norm_weight, norm_weight_handle,
      norm_epsilon, repeat, variant, residual_input_handle,
      use_rowblock16_cpuorder_finalize);
}

GpuQ4X8ResidualRmsNormHandoffRun
GpuQ4X8MatvecRunner::RunResidentPackedQ4X8ThenResidentResidualRmsNorm(
    std::uint64_t handle,
    const std::vector<std::int8_t>& q8_qs,
    const std::vector<std::int16_t>& q8_bsums,
    const std::vector<float>& q8_d,
    const std::vector<float>& residual_input,
    std::uint64_t norm_weight_handle,
    std::uint64_t hidden_size,
    float norm_epsilon,
    int repeat,
    GpuQ4X8KernelVariant variant,
    std::uint64_t residual_input_handle,
    bool use_rowblock16_output_projection) {
  return impl_->RunResidentPackedQ4X8ThenResidentResidualRmsNorm(
      handle, q8_qs, q8_bsums, q8_d, residual_input, norm_weight_handle,
      hidden_size, norm_epsilon, repeat, variant, residual_input_handle,
      use_rowblock16_output_projection);
}

std::uint64_t GpuQ4X8MatvecRunner::UploadConvWeights(
    const std::vector<float>& conv_weights,
    std::uint64_t rows,
    std::uint64_t conv_kernel_size) {
  return impl_->UploadConvWeights(conv_weights, rows, conv_kernel_size);
}

GpuQ4X8ConvHandoffRun GpuQ4X8MatvecRunner::RunResidentPackedQ4X8ThenResidentConv(
    std::uint64_t packed_handle,
    const std::vector<std::int8_t>& q8_qs,
    const std::vector<std::int16_t>& q8_bsums,
    const std::vector<float>& q8_d,
    std::uint64_t conv_weights_handle,
    const std::vector<float>& conv_state,
    std::uint64_t conv_kernel_size,
    int repeat,
    GpuQ4X8KernelVariant variant) {
  return impl_->RunResidentPackedQ4X8ThenResidentConv(
      packed_handle, q8_qs, q8_bsums, q8_d, conv_weights_handle, conv_state,
      conv_kernel_size, repeat, variant);
}

GpuQ4X8ConvHandoffRun
GpuQ4X8MatvecRunner::RunResidentPackedQ4X8ThenResidentConvState(
    std::uint64_t packed_handle,
    const std::vector<std::int8_t>& q8_qs,
    const std::vector<std::int16_t>& q8_bsums,
    const std::vector<float>& q8_d,
    std::uint64_t conv_weights_handle,
    std::uint64_t conv_state_handle,
    std::uint64_t conv_kernel_size,
    int repeat,
    GpuQ4X8KernelVariant variant,
    bool readback_state,
    std::uint64_t next_conv_state_handle,
    bool readback_qkv,
    bool readback_conv_output) {
  return impl_->RunResidentPackedQ4X8ThenResidentConvState(
      packed_handle, q8_qs, q8_bsums, q8_d, conv_weights_handle,
      conv_state_handle, conv_kernel_size, repeat, variant, readback_state,
      next_conv_state_handle, readback_qkv, readback_conv_output);
}

GpuQ4X8ConvHandoffRun GpuQ4X8MatvecRunner::
    RunF32InputHandleDeviceQ8ThenResidentPackedQ4X8ThenResidentConvState(
        std::uint64_t packed_handle,
        std::uint64_t input_handle,
        std::uint64_t conv_weights_handle,
        std::uint64_t conv_state_handle,
        std::uint64_t conv_kernel_size,
        int repeat,
        GpuQ4X8KernelVariant variant,
        bool readback_state,
        std::uint64_t next_conv_state_handle,
        bool readback_qkv,
        bool readback_conv_output) {
  return impl_
      ->RunF32InputHandleDeviceQ8ThenResidentPackedQ4X8ThenResidentConvState(
          packed_handle, input_handle, conv_weights_handle, conv_state_handle,
          conv_kernel_size, repeat, variant, readback_state,
          next_conv_state_handle, readback_qkv, readback_conv_output);
}

GpuLinearPreconvSharedQ8Run GpuQ4X8MatvecRunner::
    RunF32InputHandleSharedDeviceQ8ThenResidentPackedQ4X8ConvStateAndResidentRawQ4KCpuOrder(
        std::uint64_t packed_handle,
        std::uint64_t alpha_beta_z_handle,
        std::uint64_t input_handle,
        std::uint64_t conv_weights_handle,
        std::uint64_t conv_state_handle,
        std::uint64_t conv_kernel_size,
        int repeat,
        GpuQ4X8KernelVariant variant,
        bool readback_state,
        std::uint64_t next_conv_state_handle,
        bool readback_qkv,
        bool readback_conv_output,
        bool readback_alpha_beta_z) {
  return impl_
      ->RunF32InputHandleSharedDeviceQ8ThenResidentPackedQ4X8ConvStateAndResidentRawQ4KCpuOrder(
          packed_handle, alpha_beta_z_handle, input_handle, conv_weights_handle,
          conv_state_handle, conv_kernel_size, repeat, variant, readback_state,
          next_conv_state_handle, readback_qkv, readback_conv_output,
          readback_alpha_beta_z);
}

void GpuQ4X8MatvecRunner::ClearResidentRawQ4KCpuOrder() {
  impl_->ClearResidentRawQ4KCpuOrder();
}

void GpuQ4X8MatvecRunner::ClearResidentConvWeights() {
  impl_->ClearResidentConvWeights();
}

void GpuQ4X8MatvecRunner::ClearResidentPackedQ4X8() {
  impl_->ClearResidentPackedQ4X8();
}

GpuF32MatvecRun GpuQ4X8MatvecRunner::RunF32Matvec(
    const std::vector<float>& weights,
    const std::vector<float>& input,
    std::uint64_t rows,
    std::uint64_t cols,
    int repeat) {
  return impl_->RunF32Matvec(weights, input, rows, cols, repeat);
}

GpuSwiGluRun GpuQ4X8MatvecRunner::RunSwiGlu(
    const std::vector<float>& gate_up,
    std::uint64_t intermediate_size,
    std::uint64_t expert_count,
    int repeat) {
  return impl_->RunSwiGlu(gate_up, intermediate_size, expert_count, repeat);
}

GpuFfnTailRun GpuQ4X8MatvecRunner::RunFfnTail(
    const std::vector<float>& gate_weights,
    const std::vector<float>& attn_post_norm,
    const std::vector<float>& ffn_moe_down,
    const std::vector<float>& weights_norm,
    const std::vector<float>& ffn_shexp,
    const std::vector<float>& attn_residual,
    std::uint64_t hidden_size,
    std::uint64_t expert_count,
    int repeat) {
  return impl_->RunFfnTail(gate_weights, attn_post_norm, ffn_moe_down,
                           weights_norm, ffn_shexp, attn_residual,
                           hidden_size, expert_count, repeat);
}

GpuFfnTailRun GpuQ4X8MatvecRunner::RunFfnTailFromDownHandle(
    const std::vector<float>& gate_weights,
    const std::vector<float>& attn_post_norm,
    std::uint64_t ffn_moe_down_handle,
    const std::vector<float>& weights_norm,
    const std::vector<float>& ffn_shexp,
    const std::vector<float>& attn_residual,
    std::uint64_t hidden_size,
    std::uint64_t expert_count,
    int repeat,
    bool readback_layer_output) {
  return impl_->RunFfnTailFromDownHandle(
      gate_weights, attn_post_norm, ffn_moe_down_handle, weights_norm,
      ffn_shexp, attn_residual, hidden_size, expert_count, repeat, 0,
      readback_layer_output);
}

GpuFfnTailRun GpuQ4X8MatvecRunner::RunFfnTailFromDownHandles(
    const std::vector<float>& gate_weights,
    const std::vector<float>& attn_post_norm,
    std::uint64_t ffn_moe_down_handle,
    const std::vector<float>& weights_norm,
    std::uint64_t ffn_shexp_handle,
    const std::vector<float>& attn_residual,
    std::uint64_t hidden_size,
    std::uint64_t expert_count,
    int repeat,
    bool readback_layer_output) {
  return impl_->RunFfnTailFromDownHandle(
      gate_weights, attn_post_norm, ffn_moe_down_handle, weights_norm,
      std::vector<float>{}, attn_residual, hidden_size, expert_count, repeat,
      ffn_shexp_handle, readback_layer_output);
}

GpuFfnTailRun GpuQ4X8MatvecRunner::RunFfnTailFromDownHandlesResidentInputs(
    std::uint64_t shared_gate_matvec_handle,
    std::uint64_t attn_post_norm_handle,
    std::uint64_t ffn_moe_down_handle,
    const std::vector<float>& weights_norm,
    std::uint64_t ffn_shexp_handle,
    std::uint64_t attn_residual_handle,
    std::uint64_t hidden_size,
    std::uint64_t expert_count,
    int repeat,
    bool readback_layer_output) {
  return impl_->RunFfnTailFromDownHandlesResidentInputs(
      shared_gate_matvec_handle, attn_post_norm_handle, ffn_moe_down_handle,
      weights_norm, ffn_shexp_handle, attn_residual_handle, hidden_size,
      expert_count, repeat, readback_layer_output);
}

GpuFfnTailRun
GpuQ4X8MatvecRunner::RunFfnTailAtomicFromDownHandlesResidentInputs(
    std::uint64_t shared_gate_matvec_handle,
    std::uint64_t attn_post_norm_handle,
    std::uint64_t ffn_moe_down_handle,
    const std::vector<float>& weights_norm,
    std::uint64_t ffn_shexp_handle,
    std::uint64_t attn_residual_handle,
    std::uint64_t hidden_size,
    std::uint64_t expert_count,
    int repeat,
    bool readback_layer_output) {
  return impl_->RunFfnTailAtomicFromDownHandlesResidentInputs(
      shared_gate_matvec_handle, attn_post_norm_handle, ffn_moe_down_handle,
      weights_norm, ffn_shexp_handle, attn_residual_handle, hidden_size,
      expert_count, repeat, readback_layer_output);
}

GpuRmsNormRun GpuQ4X8MatvecRunner::RunRmsNormHidden(
    const std::vector<float>& input,
    const std::vector<float>& weight,
    float norm_epsilon,
    int repeat,
    bool serial_reduction) {
  return impl_->RunRmsNormHidden(
      input, weight, norm_epsilon, repeat, serial_reduction);
}

GpuRmsNormRun GpuQ4X8MatvecRunner::RunRmsNormHiddenResidentWeight(
    const std::vector<float>& input,
    std::uint64_t weight_handle,
    std::uint64_t hidden_size,
    float norm_epsilon,
    int repeat,
    bool serial_reduction) {
  return impl_->RunRmsNormHiddenResidentWeight(
      input, weight_handle, hidden_size, norm_epsilon, repeat,
      serial_reduction);
}

GpuRmsNormRun GpuQ4X8MatvecRunner::RunRmsNormHiddenResidentInputResidentWeight(
    std::uint64_t input_handle,
    std::uint64_t weight_handle,
    std::uint64_t hidden_size,
    float norm_epsilon,
    int repeat,
    bool readback_output,
    bool serial_reduction) {
  return impl_->RunRmsNormHiddenResidentInputResidentWeight(
      input_handle, weight_handle, hidden_size, norm_epsilon, repeat,
      readback_output, serial_reduction);
}

GpuResidualRmsNormRun GpuQ4X8MatvecRunner::RunResidualRmsNormHidden(
    const std::vector<float>& residual_input,
    const std::vector<float>& residual_delta,
    const std::vector<float>& norm_weight,
    float norm_epsilon,
    int repeat) {
  return impl_->RunResidualRmsNormHidden(
      residual_input, residual_delta, norm_weight, norm_epsilon, repeat);
}

GpuResidualRmsNormRun
GpuQ4X8MatvecRunner::RunResidualRmsNormHiddenResidentWeight(
    const std::vector<float>& residual_input,
    const std::vector<float>& residual_delta,
    std::uint64_t norm_weight_handle,
    std::uint64_t hidden_size,
    float norm_epsilon,
    int repeat) {
  return impl_->RunResidualRmsNormHiddenResidentWeight(
      residual_input, residual_delta, norm_weight_handle, hidden_size,
      norm_epsilon, repeat);
}

GpuFullAttentionCoreGateRun GpuQ4X8MatvecRunner::RunFullAttentionCoreGate(
    const std::vector<float>& q_rope,
    const std::vector<float>& k_history_flat,
    const std::vector<float>& v_history_flat,
    const std::vector<float>& q_full,
    std::uint64_t token_count,
    std::uint64_t head_dim,
    std::uint64_t q_head_count,
    std::uint64_t kv_head_count,
    float attention_scale,
    int repeat) {
  return impl_->RunFullAttentionCoreGate(
      q_rope, k_history_flat, v_history_flat, q_full, token_count, head_dim,
      q_head_count, kv_head_count, attention_scale, repeat);
}

GpuFullAttentionQkNormRopeRun
GpuQ4X8MatvecRunner::RunFullAttentionQkNormRopeFromHandles(
    std::uint64_t q_full_handle,
    std::uint64_t k_raw_handle,
    std::uint64_t q_norm_weight_handle,
    std::uint64_t k_norm_weight_handle,
    const std::vector<float>& rope_cache,
    std::uint64_t head_dim,
    std::uint64_t q_head_count,
    std::uint64_t kv_head_count,
    std::uint64_t rope_dimension_count,
    float norm_epsilon,
    int repeat,
    bool readback_output) {
  return impl_->RunFullAttentionQkNormRopeFromHandles(
      q_full_handle, k_raw_handle, q_norm_weight_handle, k_norm_weight_handle,
      rope_cache, head_dim, q_head_count, kv_head_count, rope_dimension_count,
      norm_epsilon, repeat, readback_output);
}

GpuFullAttentionHistoryAppendRun
GpuQ4X8MatvecRunner::BuildFullAttentionHistoryFromHandle(
    const std::vector<float>& previous_history_flat,
    std::uint64_t current_handle,
    std::uint64_t kv_values,
    bool readback_output) {
  return impl_->BuildFullAttentionHistoryFromHandle(
      previous_history_flat, current_handle, kv_values, readback_output);
}

GpuFullAttentionOutputHandoffRun
GpuQ4X8MatvecRunner::RunFullAttentionCoreGateThenResidentPackedQ4X8ResidualRmsNorm(
    const std::vector<float>& q_rope,
    const std::vector<float>& k_history_flat,
    const std::vector<float>& v_history_flat,
    const std::vector<float>& q_full,
    std::uint64_t token_count,
    std::uint64_t head_dim,
    std::uint64_t q_head_count,
    std::uint64_t kv_head_count,
    float attention_scale,
    std::uint64_t output_projection_handle,
    const std::vector<float>& residual_input,
    const std::vector<float>& norm_weight,
    float norm_epsilon,
    int repeat,
    GpuQ4X8KernelVariant variant,
    std::uint64_t residual_input_handle) {
  return impl_->RunFullAttentionCoreGateThenResidentPackedQ4X8ResidualRmsNorm(
      q_rope, k_history_flat, v_history_flat, q_full, token_count, head_dim,
      q_head_count, kv_head_count, attention_scale, output_projection_handle,
      residual_input, norm_weight, norm_epsilon, repeat, variant, 0, 0,
      residual_input_handle);
}

GpuFullAttentionOutputHandoffRun
GpuQ4X8MatvecRunner::
    RunFullAttentionCoreGateThenResidentPackedQ4X8ResidualRmsNormFromHandles(
        std::uint64_t q_rope_handle,
        std::uint64_t k_history_handle,
        std::uint64_t v_history_handle,
        std::uint64_t q_full_handle,
        std::uint64_t token_count,
        std::uint64_t head_dim,
        std::uint64_t q_head_count,
        std::uint64_t kv_head_count,
        float attention_scale,
        std::uint64_t output_projection_handle,
        const std::vector<float>& residual_input,
        const std::vector<float>& norm_weight,
        float norm_epsilon,
        int repeat,
        GpuQ4X8KernelVariant variant,
        std::uint64_t residual_input_handle) {
  return impl_->RunFullAttentionCoreGateThenResidentPackedQ4X8ResidualRmsNorm(
      {}, {}, {}, {}, token_count, head_dim, q_head_count, kv_head_count,
      attention_scale, output_projection_handle, residual_input, norm_weight,
      norm_epsilon, repeat, variant, 0, 0, residual_input_handle,
      q_rope_handle, k_history_handle, v_history_handle, q_full_handle);
}

GpuFullAttentionOutputHandoffRun
GpuQ4X8MatvecRunner::RunFullAttentionCoreGateThenResidentPackedQ4X8ResidualRmsNorm(
    const std::vector<float>& q_rope,
    const std::vector<float>& k_history_flat,
    const std::vector<float>& v_history_flat,
    const std::vector<float>& q_full,
    std::uint64_t token_count,
    std::uint64_t head_dim,
    std::uint64_t q_head_count,
    std::uint64_t kv_head_count,
    float attention_scale,
    std::uint64_t output_projection_handle,
    const std::vector<float>& residual_input,
    std::uint64_t norm_weight_handle,
    std::uint64_t hidden_size,
    float norm_epsilon,
    int repeat,
    GpuQ4X8KernelVariant variant,
    std::uint64_t residual_input_handle) {
  return impl_->RunFullAttentionCoreGateThenResidentPackedQ4X8ResidualRmsNorm(
      q_rope, k_history_flat, v_history_flat, q_full, token_count, head_dim,
      q_head_count, kv_head_count, attention_scale, output_projection_handle,
      residual_input, {}, norm_epsilon, repeat, variant, norm_weight_handle,
      hidden_size, residual_input_handle);
}

GpuFullAttentionOutputHandoffRun
GpuQ4X8MatvecRunner::
    RunFullAttentionCoreGateThenResidentPackedQ4X8ResidualRmsNormFromHandles(
        std::uint64_t q_rope_handle,
        std::uint64_t k_history_handle,
        std::uint64_t v_history_handle,
        std::uint64_t q_full_handle,
        std::uint64_t token_count,
        std::uint64_t head_dim,
        std::uint64_t q_head_count,
        std::uint64_t kv_head_count,
        float attention_scale,
        std::uint64_t output_projection_handle,
        const std::vector<float>& residual_input,
        std::uint64_t norm_weight_handle,
        std::uint64_t hidden_size,
        float norm_epsilon,
        int repeat,
        GpuQ4X8KernelVariant variant,
        std::uint64_t residual_input_handle) {
  return impl_->RunFullAttentionCoreGateThenResidentPackedQ4X8ResidualRmsNorm(
      {}, {}, {}, {}, token_count, head_dim, q_head_count, kv_head_count,
      attention_scale, output_projection_handle, residual_input, {},
      norm_epsilon, repeat, variant, norm_weight_handle, hidden_size,
      residual_input_handle, q_rope_handle, k_history_handle, v_history_handle,
      q_full_handle);
}

std::uint64_t GpuQ4X8MatvecRunner::UploadF32MatvecWeights(
    const std::vector<float>& weights,
    std::uint64_t rows,
    std::uint64_t cols) {
  return impl_->UploadF32MatvecWeights(weights, rows, cols);
}

GpuF32MatvecRun GpuQ4X8MatvecRunner::RunResidentF32Matvec(
    std::uint64_t handle,
    const std::vector<float>& input,
    int repeat) {
  return impl_->RunResidentF32Matvec(handle, input, repeat);
}

GpuF32MatvecRun GpuQ4X8MatvecRunner::RunResidentF32MatvecFromInputHandle(
    std::uint64_t handle,
    std::uint64_t input_handle,
    int repeat,
    bool readback_output) {
  return impl_->RunResidentF32MatvecFromInputHandle(
      handle, input_handle, repeat, readback_output);
}

GpuF32TopKRun GpuQ4X8MatvecRunner::RunResidentF32TopK(
    std::uint64_t values_handle,
    int topk,
    int repeat) {
  return impl_->RunResidentF32TopK(values_handle, topk, repeat);
}

void GpuQ4X8MatvecRunner::ClearResidentF32Matvec() {
  impl_->ClearResidentF32Matvec();
}

GpuQ4X8ConvHandoffRun GpuQ4X8MatvecRunner::RunThenConv(
    const std::vector<std::uint8_t>& packed,
    const std::vector<std::int8_t>& q8_qs,
    const std::vector<std::int16_t>& q8_bsums,
    const std::vector<float>& q8_d,
    const std::vector<float>& conv_weights,
    const std::vector<float>& conv_state,
    std::uint64_t rows,
    std::uint64_t blocks_per_row,
    std::uint64_t conv_kernel_size,
    int repeat,
    GpuQ4X8KernelVariant variant) {
  return impl_->RunThenConv(packed, q8_qs, q8_bsums, q8_d, conv_weights, conv_state,
                            rows, blocks_per_row, conv_kernel_size, repeat, variant);
}

GpuLinearAttentionPostConvPrepRun GpuQ4X8MatvecRunner::RunPostConvPrep(
    const std::vector<float>& conv_output_raw,
    std::uint64_t head_dim,
    std::uint64_t query_heads,
    std::uint64_t value_heads,
    float norm_epsilon,
    int repeat,
    bool readback_intermediates) {
  return impl_->RunPostConvPrep(conv_output_raw, head_dim, query_heads, value_heads,
                                norm_epsilon, repeat, readback_intermediates);
}

GpuLinearAttentionPostConvPrepRun GpuQ4X8MatvecRunner::RunPostConvPrepFused(
    const std::vector<float>& conv_output_raw,
    std::uint64_t head_dim,
    std::uint64_t query_heads,
    std::uint64_t value_heads,
    float norm_epsilon,
    int repeat,
    bool readback_intermediates) {
  return impl_->RunPostConvPrepFused(
      conv_output_raw, head_dim, query_heads, value_heads, norm_epsilon, repeat,
      readback_intermediates);
}

GpuLinearAttentionDeltaRun GpuQ4X8MatvecRunner::RunLinearAttentionDelta(
    const std::vector<float>& q,
    const std::vector<float>& k,
    const std::vector<float>& v,
    const std::vector<float>& gate,
    const std::vector<float>& beta,
    const std::vector<float>& recurrent_state,
    const std::vector<float>& z,
    const std::vector<float>& norm_weight,
    std::uint64_t head_dim,
    std::uint64_t query_heads,
    std::uint64_t value_heads,
    float norm_epsilon,
    int repeat,
    bool cpu_shape_final_norm) {
  return impl_->RunLinearAttentionDelta(q, k, v, gate, beta, recurrent_state, z,
                                        norm_weight, head_dim, query_heads,
                                        value_heads, norm_epsilon, repeat,
                                        cpu_shape_final_norm);
}

std::uint64_t GpuQ4X8MatvecRunner::UploadF32Buffer(
    const std::vector<float>& values) {
  return impl_->UploadF32Buffer(values);
}

std::uint64_t GpuQ4X8MatvecRunner::CloneResidentF32Buffer(
    std::uint64_t source_handle) {
  return impl_->CloneResidentF32Buffer(source_handle);
}

GpuRouterQkvDeltaSelectedValueOverlayRun
GpuQ4X8MatvecRunner::RunRouterQkvDeltaSelectedValueOverlay(
    std::uint64_t base_handle,
    std::uint64_t source_handle,
    const std::vector<std::int32_t>& selected_indices,
    int repeat,
    bool readback_output) {
  return impl_->RunRouterQkvDeltaSelectedValueOverlay(
      base_handle, source_handle, selected_indices, repeat, readback_output);
}

GpuRouterQkvDeltaSelectedValueOverlayRun
GpuQ4X8MatvecRunner::RunRouterQkvDeltaBlockQ16Overlay(
    std::uint64_t base_handle,
    const std::vector<std::int32_t>& selected_indices,
    const std::vector<std::int16_t>& selected_q_delta,
    const std::vector<float>& block_scales,
    int repeat,
    bool readback_output) {
  return impl_->RunRouterQkvDeltaBlockQ16Overlay(
      base_handle, selected_indices, selected_q_delta, block_scales, repeat,
      readback_output);
}

GpuLinearAttentionDeltaRun
GpuQ4X8MatvecRunner::RunLinearAttentionDeltaResidentState(
    std::uint64_t state_handle,
    const std::vector<float>& q,
    const std::vector<float>& k,
    const std::vector<float>& v,
    const std::vector<float>& gate,
    const std::vector<float>& beta,
    const std::vector<float>& z,
    const std::vector<float>& norm_weight,
    std::uint64_t head_dim,
    std::uint64_t query_heads,
    std::uint64_t value_heads,
    float norm_epsilon,
    int repeat,
    bool readback_state,
    bool cpu_shape_final_norm,
    bool readback_attention_output,
    bool readback_final_output) {
  return impl_->RunLinearAttentionDeltaResidentState(
      state_handle, q, k, v, gate, beta, z, norm_weight, head_dim,
      query_heads, value_heads, norm_epsilon, repeat, readback_state,
      cpu_shape_final_norm, readback_attention_output, readback_final_output);
}

GpuLinearAttentionDeltaRun
GpuQ4X8MatvecRunner::RunPostConvPrepThenLinearAttentionDeltaResidentState(
    std::uint64_t conv_output_handle,
    std::uint64_t state_handle,
    const std::vector<float>& gate,
    const std::vector<float>& beta,
    const std::vector<float>& z,
    const std::vector<float>& norm_weight,
    std::uint64_t head_dim,
    std::uint64_t query_heads,
    std::uint64_t value_heads,
    float norm_epsilon,
    int repeat,
    bool readback_state,
    bool cpu_shape_final_norm,
    bool readback_attention_output,
    bool readback_final_output) {
  return impl_->RunPostConvPrepThenLinearAttentionDeltaResidentState(
      conv_output_handle, state_handle, gate, beta, z, norm_weight, head_dim,
      query_heads, value_heads, norm_epsilon, repeat, readback_state,
      cpu_shape_final_norm, readback_attention_output, readback_final_output);
}

GpuLinearAttentionDeltaRun GpuQ4X8MatvecRunner::
    RunPostConvPrepThenLinearAttentionDeltaResidentStateCpuOrder(
        std::uint64_t conv_output_handle,
        std::uint64_t state_handle,
        const std::vector<float>& gate,
        const std::vector<float>& beta,
        const std::vector<float>& z,
        const std::vector<float>& norm_weight,
        std::uint64_t head_dim,
        std::uint64_t query_heads,
        std::uint64_t value_heads,
        float norm_epsilon,
        int repeat,
        bool readback_state,
        bool readback_attention_output,
        bool readback_final_output) {
  return impl_->RunPostConvPrepThenLinearAttentionDeltaResidentState(
      conv_output_handle, state_handle, gate, beta, z, norm_weight, head_dim,
      query_heads, value_heads, norm_epsilon, repeat, readback_state, false,
      readback_attention_output, readback_final_output, true);
}

void GpuQ4X8MatvecRunner::ClearResidentF32Buffers() {
  impl_->ClearResidentF32Buffers();
}

}  // namespace iq36
