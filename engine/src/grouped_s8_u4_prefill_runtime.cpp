#include <CL/cl.h>
#include <CL/cl_ext.h>

#include "intel_qwen36/grouped_s8_u4_prefill_runtime.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr std::size_t kTokenCount = 1024;
constexpr std::size_t kHiddenSize = 2048;
constexpr std::size_t kAssignments = 8192;
constexpr std::size_t kExpertCount = 256;
constexpr std::size_t kScheduleMetadataWords = 7;
constexpr std::size_t kNativeGateTaskCapacity = kAssignments;
constexpr std::size_t kNativeDownTaskCapacity = kAssignments * 2;
constexpr std::size_t kPersistentWorkgroupCount = 96;
constexpr std::size_t kRouterWeightBytes =
    kExpertCount * kHiddenSize * sizeof(float);
constexpr std::size_t kRouterStatusBytes = 96 * 64;
constexpr std::size_t kIntermediateSize = 512;
constexpr std::size_t kGateUpSize = 1024;
constexpr std::size_t kGateGroups = 64;
constexpr std::size_t kGateBlocks = 8;
constexpr std::size_t kDownGroups = 16;
constexpr std::size_t kExactQ6DownGroups = 32;
constexpr std::size_t kDownBlocks = 2;
constexpr double kMismatchThreshold = 5e-3;
constexpr std::size_t kGateUpWeightBytes =
    kExpertCount * kGateUpSize * kHiddenSize / 2;
constexpr std::size_t kGateUpScaleBytes =
    kExpertCount * kGateGroups * kGateUpSize * sizeof(float);
constexpr std::size_t kGateUpMinCodeBytes =
    kExpertCount * kGateUpSize * kGateBlocks * 8;
constexpr std::size_t kGateUpDminBytes =
    kExpertCount * kGateUpSize * kGateBlocks * sizeof(float);
constexpr std::size_t kGateUpExactScaleCodeBytes =
    kExpertCount * kGateUpSize * kGateBlocks * 8;
constexpr std::size_t kGateUpExactBlockScaleBytes =
    kExpertCount * kGateUpSize * kGateBlocks * sizeof(float);
constexpr std::size_t kDownWeightBytes =
    kExpertCount * kHiddenSize * kIntermediateSize / 2;
constexpr std::size_t kDownScaleBytes =
    kExpertCount * kDownGroups * kHiddenSize * sizeof(float);
constexpr std::size_t kDownMinCodeBytes =
    kExpertCount * kHiddenSize * kDownBlocks * 8;
constexpr std::size_t kDownDminBytes =
    kExpertCount * kHiddenSize * kDownBlocks * sizeof(float);
constexpr std::size_t kDownExactScaleCodeBytes =
    kExpertCount * kHiddenSize * kDownBlocks * 8;
constexpr std::size_t kDownExactBlockScaleBytes =
    kExpertCount * kHiddenSize * kDownBlocks * sizeof(float);
constexpr std::size_t kGateUpResidentWeightBytes =
    kGateUpWeightBytes + kGateUpScaleBytes + kGateUpMinCodeBytes +
    kGateUpDminBytes;
constexpr std::size_t kExactGateUpResidentWeightBytes =
    kGateUpWeightBytes + kGateUpExactScaleCodeBytes +
    kGateUpMinCodeBytes + kGateUpExactBlockScaleBytes +
    kGateUpDminBytes;
constexpr std::size_t kQ4DownResidentWeightBytes =
    kDownWeightBytes + kDownScaleBytes + kDownMinCodeBytes + kDownDminBytes;
constexpr std::size_t kExactQ4DownResidentWeightBytes =
    kDownWeightBytes + kDownExactScaleCodeBytes + kDownMinCodeBytes +
    kDownExactBlockScaleBytes + kDownDminBytes;
constexpr std::size_t kQ6DownWeightBytes =
    kExpertCount * kHiddenSize * kIntermediateSize;
constexpr std::size_t kQ6DownScaleBytes =
    kExpertCount * kHiddenSize * kDownGroups * sizeof(std::uint16_t);
constexpr std::size_t kExactQ6DownScaleBytes =
    kExpertCount * kHiddenSize * kExactQ6DownGroups * sizeof(float);
constexpr std::size_t kExactBlockQ6DownIntegerScaleBytes =
    kExpertCount * kHiddenSize * kExactQ6DownGroups;
constexpr std::size_t kExactBlockQ6DownBlockScaleBytes =
    kExpertCount * kHiddenSize * 2 * sizeof(float);
using Args = iq36::GroupedS8U4PrefillConfig;

bool IsQ6DownKind(iq36::GroupedPrefillDownKind kind) {
  return kind == iq36::GroupedPrefillDownKind::kQ6U8Surrogate ||
      kind == iq36::GroupedPrefillDownKind::kQ6U8ExactPer16 ||
      kind == iq36::GroupedPrefillDownKind::kQ6U8ExactBlock;
}

bool UsesF32Contributions(iq36::GroupedPrefillDownKind kind) {
  return kind == iq36::GroupedPrefillDownKind::kQ4U4F32Contribution ||
      kind == iq36::GroupedPrefillDownKind::kQ4U4ExactBlock ||
      kind == iq36::GroupedPrefillDownKind::kQ6U8ExactPer16 ||
      kind == iq36::GroupedPrefillDownKind::kQ6U8ExactBlock;
}

bool UsesExactQ4Down(iq36::GroupedPrefillDownKind kind) {
  return kind == iq36::GroupedPrefillDownKind::kQ4U4ExactBlock;
}

[[noreturn]] void Fail(const std::string& message) {
  throw std::runtime_error(message);
}

void Require(bool condition, const std::string& message) {
  if (!condition) Fail(message);
}

void Check(cl_int status, const std::string& where) {
  if (status != CL_SUCCESS) {
    std::ostringstream stream;
    stream << where << " failed with OpenCL error " << status;
    Fail(stream.str());
  }
}

Args ParseArgs(int argc, char** argv) {
  Args args;
  for (int index = 1; index < argc; ++index) {
    const std::string option = argv[index];
    const auto value = [&]() -> std::string {
      if (++index >= argc) Fail(option + " requires a value");
      return argv[index];
    };
    if (option == "--prep-dir") args.prep_dir = value();
    else if (option == "--gateup-binary") args.gateup_binary = value();
    else if (option == "--down-binary") args.down_binary = value();
    else if (option == "--kernels") args.kernels = value();
    else if (option == "--input") args.input = value();
    else if (option == "--topk") args.topk = value();
    else if (option == "--topk-stride") {
      args.topk_stride = std::stoull(value());
    }
    else if (option == "--oracle") args.oracle = value();
    else if (option == "--router-weights") args.router_weights = value();
    else if (option == "--down-oracle") args.down_oracle = value();
    else if (option == "--moe-oracle") args.moe_oracle = value();
    else if (option == "--warmup") args.warmup = std::stoi(value());
    else if (option == "--repeat") args.repeat = std::stoi(value());
    else if (option == "--kernel-cap-us") {
      args.kernel_cap_us = std::stod(value());
    } else if (option == "--schedule-probe-only") {
      args.schedule_probe_only = true;
    } else if (option == "--m8-source-preflight") {
      args.m8_source_preflight = true;
    } else {
      Fail("unknown option: " + option);
    }
  }
  return args;
}

void ValidateArgs(const Args& args) {
  Require(!args.topk.empty(), "top-k input is required");
  Require(args.topk_stride >= 8 * sizeof(std::int32_t),
          "top-k stride is too small");
  if (!args.schedule_probe_only) {
    Require(!args.prep_dir.empty() && !args.gateup_binary.empty() &&
                !args.down_binary.empty() && !args.kernels.empty() &&
                !args.input.empty() && !args.oracle.empty() &&
                !args.router_weights.empty() && !args.down_oracle.empty() &&
                !args.moe_oracle.empty(),
            "prep, binaries, kernels, input, and all oracles are required");
    Require(args.warmup > 0 && args.repeat > 0 && args.kernel_cap_us > 0.0,
            "warmup, repeat, and kernel cap must be positive");
  }
}

std::string JoinPath(const std::string& directory, const char* name) {
  return directory + (directory.empty() || directory.back() == '/' ? "" : "/") +
      name;
}

std::vector<std::uint8_t> ReadBytes(const std::string& path,
                                    std::size_t expected_bytes = 0) {
  std::ifstream input(path, std::ios::binary);
  Require(static_cast<bool>(input), "could not open input: " + path);
  input.seekg(0, std::ios::end);
  const auto size = input.tellg();
  Require(size >= 0, "could not determine input size: " + path);
  if (expected_bytes != 0) {
    Require(static_cast<std::size_t>(size) == expected_bytes,
            "input byte size mismatch: " + path);
  }
  input.seekg(0, std::ios::beg);
  std::vector<std::uint8_t> values(static_cast<std::size_t>(size));
  input.read(reinterpret_cast<char*>(values.data()),
             static_cast<std::streamsize>(values.size()));
  Require(static_cast<bool>(input), "could not read input: " + path);
  return values;
}

template <typename Value>
std::vector<Value> ReadVector(const std::string& path, std::size_t count) {
  const auto bytes = ReadBytes(path, count * sizeof(Value));
  std::vector<Value> values(count);
  std::memcpy(values.data(), bytes.data(), bytes.size());
  return values;
}

std::string ReadText(const std::string& path) {
  const auto bytes = ReadBytes(path);
  return std::string(bytes.begin(), bytes.end());
}

struct GroupedSchedule {
  std::vector<std::int32_t> offsets;
  std::vector<std::int32_t> token_map;
  std::vector<std::int32_t> inverse_map;
  std::vector<float> compact_router_weights;
  std::vector<std::uint32_t> gateup_task_coordinates;
  std::vector<std::uint32_t> down_task_coordinates;
  std::vector<std::uint32_t> native_m8_gateup_task_coordinates;
  std::vector<std::uint32_t> native_m8_down_task_coordinates;
  std::size_t gateup_task_count = 0;
  std::size_t down_task_count = 0;
  std::size_t active_experts = 0;
  std::size_t max_group = 0;
};

GroupedSchedule BuildGroupedSchedule(
    const std::vector<std::uint8_t>& topk, std::size_t stride,
    const std::vector<float>& router_weights, bool build_gateup_tasks,
    bool build_down_tasks, bool build_native_m8_tasks = false) {
  Require(topk.size() >= (kTokenCount - 1) * stride +
              8 * sizeof(std::int32_t),
          "top-k payload is truncated");
  Require(router_weights.size() == kAssignments,
          "router weight count mismatch");
  std::array<std::int32_t, kExpertCount> counts{};
  std::array<std::int32_t, kAssignments> expert_ids{};
  for (std::size_t token = 0; token < kTokenCount; ++token) {
    for (std::size_t rank = 0; rank < 8; ++rank) {
      std::int32_t expert = -1;
      std::memcpy(&expert,
                  topk.data() + token * stride +
                      rank * sizeof(std::int32_t),
                  sizeof(expert));
      if (expert < 0 ||
          expert >= static_cast<std::int32_t>(kExpertCount)) {
        Fail("top-k expert is out of range");
      }
      for (std::size_t prior = 0; prior < rank; ++prior) {
        if (expert_ids[token * 8 + prior] == expert) {
          Fail("top-k expert is duplicated within a token");
        }
      }
      expert_ids[token * 8 + rank] = expert;
      ++counts[static_cast<std::size_t>(expert)];
    }
  }

  GroupedSchedule schedule;
  schedule.offsets.resize(kExpertCount);
  schedule.token_map.resize(kAssignments);
  schedule.inverse_map.assign(kAssignments, -1);
  schedule.compact_router_weights.resize(kAssignments);
  if (build_gateup_tasks) {
    schedule.gateup_task_coordinates.reserve(kExpertCount * 64);
  }
  if (build_down_tasks) {
    schedule.down_task_coordinates.reserve(kExpertCount * 128);
  }
  if (build_native_m8_tasks) {
    schedule.native_m8_gateup_task_coordinates.reserve(kAssignments * 5);
    schedule.native_m8_down_task_coordinates.reserve(kAssignments * 5);
  }
  std::array<std::int32_t, kExpertCount> cursors{};
  std::int32_t cumulative = 0;
  for (std::size_t expert = 0; expert < kExpertCount; ++expert) {
    const std::int32_t begin = cumulative;
    cursors[expert] = cumulative;
    schedule.active_experts += counts[expert] != 0;
    schedule.max_group = std::max(
        schedule.max_group, static_cast<std::size_t>(counts[expert]));
    cumulative += counts[expert];
    schedule.offsets[expert] = cumulative;
    const std::uint32_t m_tiles = static_cast<std::uint32_t>(
        (cumulative - begin + 15) / 16);
    if (build_gateup_tasks) {
      for (std::uint32_t output_tile = 0; output_tile < 64U;
           ++output_tile) {
        for (std::uint32_t m_tile = 0; m_tile < m_tiles; ++m_tile) {
          schedule.gateup_task_coordinates.push_back(
              static_cast<std::uint32_t>(expert) | (output_tile << 8) |
              (m_tile << 15));
        }
      }
    }
    if (build_down_tasks) {
      for (std::uint32_t output_tile = 0; output_tile < 128U;
           ++output_tile) {
        for (std::uint32_t m_tile = 0; m_tile < m_tiles; ++m_tile) {
          schedule.down_task_coordinates.push_back(
              static_cast<std::uint32_t>(expert) | (output_tile << 8) |
              (m_tile << 15));
        }
      }
    }
    if (build_native_m8_tasks) {
      const std::uint32_t m8_tiles = static_cast<std::uint32_t>(
          (cumulative - begin + 7) / 8);
      for (std::uint32_t output_tile = 0; output_tile < 32U;
           ++output_tile) {
        for (std::uint32_t m_tile = 0; m_tile < m8_tiles; ++m_tile) {
          const std::uint32_t coordinate =
              static_cast<std::uint32_t>(expert) | (output_tile << 8) |
              (m_tile << 15);
          schedule.native_m8_gateup_task_coordinates.push_back(coordinate);
          schedule.native_m8_down_task_coordinates.push_back(coordinate);
        }
      }
    }
  }
  schedule.gateup_task_count = schedule.gateup_task_coordinates.size();
  schedule.down_task_count = schedule.down_task_coordinates.size();
  Require(cumulative == static_cast<std::int32_t>(kAssignments),
          "grouped assignment count mismatch");
  for (std::size_t token = 0; token < kTokenCount; ++token) {
    for (std::size_t rank = 0; rank < 8; ++rank) {
      const std::int32_t expert = expert_ids[token * 8 + rank];
      const std::int32_t row =
          cursors[static_cast<std::size_t>(expert)]++;
      schedule.token_map[static_cast<std::size_t>(row)] =
          static_cast<std::int32_t>(token);
      schedule.inverse_map[token * 8 + rank] = row;
      schedule.compact_router_weights[static_cast<std::size_t>(row)] =
          router_weights[token * 8 + rank];
    }
  }
  Require(std::none_of(schedule.inverse_map.begin(),
                       schedule.inverse_map.end(),
                       [](std::int32_t value) { return value < 0; }),
          "grouped inverse map is incomplete");
  return schedule;
}

std::string ProgramLog(cl_program program, cl_device_id device) {
  std::size_t size = 0;
  Check(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0,
                              nullptr, &size),
        "clGetProgramBuildInfo size");
  std::string log(size, '\0');
  if (size != 0) {
    Check(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, size,
                                log.data(), nullptr),
          "clGetProgramBuildInfo log");
  }
  return log;
}

struct OpenClState {
  cl_platform_id platform = nullptr;
  cl_device_id device = nullptr;
  cl_context context = nullptr;
  cl_command_queue queue = nullptr;
  clMemFreeINTEL_fn mem_free = nullptr;
  clSetKernelArgMemPointerINTEL_fn set_usm_arg = nullptr;
  clEnqueueMemFillINTEL_fn fill_usm = nullptr;
  std::vector<void*> usm_allocations;
  std::vector<cl_mem> memories;
  std::vector<cl_kernel> kernels;
  std::vector<cl_program> programs;

  OpenClState() = default;
  OpenClState(const OpenClState&) = delete;
  OpenClState& operator=(const OpenClState&) = delete;
  OpenClState(OpenClState&& other) noexcept
      : platform(other.platform),
        device(other.device),
        context(other.context),
        queue(other.queue),
        mem_free(other.mem_free),
        set_usm_arg(other.set_usm_arg),
        fill_usm(other.fill_usm),
        usm_allocations(std::move(other.usm_allocations)),
        memories(std::move(other.memories)),
        kernels(std::move(other.kernels)),
        programs(std::move(other.programs)) {
    other.context = nullptr;
    other.queue = nullptr;
    other.mem_free = nullptr;
    other.set_usm_arg = nullptr;
    other.fill_usm = nullptr;
    other.usm_allocations.clear();
    other.memories.clear();
    other.kernels.clear();
    other.programs.clear();
  }

  ~OpenClState() {
    for (cl_kernel value : kernels) clReleaseKernel(value);
    for (cl_program value : programs) clReleaseProgram(value);
    for (cl_mem value : memories) clReleaseMemObject(value);
    if (mem_free != nullptr) {
      for (void* value : usm_allocations) mem_free(context, value);
    }
    if (queue != nullptr) clReleaseCommandQueue(queue);
    if (context != nullptr) clReleaseContext(context);
  }
};

OpenClState CreateOpenCl(bool profiling = false) {
  OpenClState state;
  cl_uint platform_count = 0;
  Check(clGetPlatformIDs(0, nullptr, &platform_count), "clGetPlatformIDs count");
  std::vector<cl_platform_id> platforms(platform_count);
  Check(clGetPlatformIDs(platform_count, platforms.data(), nullptr),
        "clGetPlatformIDs");
  for (cl_platform_id platform : platforms) {
    cl_uint device_count = 0;
    if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr,
                       &device_count) != CL_SUCCESS) {
      continue;
    }
    std::vector<cl_device_id> devices(device_count);
    Check(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, device_count,
                         devices.data(), nullptr),
          "clGetDeviceIDs");
    for (cl_device_id device : devices) {
      char name[256] = {};
      Check(clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name), name, nullptr),
            "clGetDeviceInfo name");
      if (std::string(name).find("B390") != std::string::npos) {
        state.platform = platform;
        state.device = device;
        break;
      }
    }
    if (state.device != nullptr) break;
  }
  Require(state.device != nullptr, "Intel Arc B390 OpenCL device not found");
  cl_int status = CL_SUCCESS;
  state.context = clCreateContext(nullptr, 1, &state.device, nullptr, nullptr,
                                  &status);
  Check(status, "clCreateContext");
  const cl_queue_properties properties[] = {
      CL_QUEUE_PROPERTIES,
      static_cast<cl_queue_properties>(
          profiling ? CL_QUEUE_PROFILING_ENABLE : 0),
      0};
  state.queue = clCreateCommandQueueWithProperties(
      state.context, state.device, properties, &status);
  Check(status, "clCreateCommandQueueWithProperties");
  return state;
}

cl_mem CreateCopied(OpenClState& state, const std::string& path,
                    std::size_t bytes, cl_mem_flags flags = CL_MEM_READ_ONLY) {
  const auto values = ReadBytes(path, bytes);
  cl_int status = CL_SUCCESS;
  cl_mem memory = clCreateBuffer(state.context,
      flags | CL_MEM_COPY_HOST_PTR, values.size(),
      const_cast<std::uint8_t*>(values.data()), &status);
  Check(status, "clCreateBuffer copied " + path);
  state.memories.push_back(memory);
  return memory;
}

template <typename Value>
cl_mem CreateCopiedVector(OpenClState& state,
                          const std::vector<Value>& values,
                          cl_mem_flags flags = CL_MEM_READ_ONLY) {
  cl_int status = CL_SUCCESS;
  cl_mem memory = clCreateBuffer(state.context,
      flags | CL_MEM_COPY_HOST_PTR, values.size() * sizeof(Value),
      const_cast<Value*>(values.data()), &status);
  Check(status, "clCreateBuffer copied vector");
  state.memories.push_back(memory);
  return memory;
}

cl_mem CreateEmpty(OpenClState& state, std::size_t bytes) {
  cl_int status = CL_SUCCESS;
  cl_mem memory = clCreateBuffer(
      state.context, CL_MEM_READ_WRITE, bytes, nullptr, &status);
  Check(status, "clCreateBuffer empty");
  state.memories.push_back(memory);
  return memory;
}

void* CreateZeroedDeviceUsm(OpenClState& state, std::size_t bytes,
                            std::size_t alignment) {
  const auto allocate = reinterpret_cast<clDeviceMemAllocINTEL_fn>(
      clGetExtensionFunctionAddressForPlatform(
          state.platform, "clDeviceMemAllocINTEL"));
  state.mem_free = reinterpret_cast<clMemFreeINTEL_fn>(
      clGetExtensionFunctionAddressForPlatform(
          state.platform, "clMemFreeINTEL"));
  state.set_usm_arg = reinterpret_cast<clSetKernelArgMemPointerINTEL_fn>(
      clGetExtensionFunctionAddressForPlatform(
          state.platform, "clSetKernelArgMemPointerINTEL"));
  state.fill_usm = reinterpret_cast<clEnqueueMemFillINTEL_fn>(
      clGetExtensionFunctionAddressForPlatform(
          state.platform, "clEnqueueMemFillINTEL"));
  Require(allocate != nullptr && state.mem_free != nullptr &&
              state.set_usm_arg != nullptr && state.fill_usm != nullptr,
          "native router requires Intel USM extensions");
  cl_int status = CL_SUCCESS;
  void* pointer = allocate(
      state.context, state.device, nullptr, bytes, alignment, &status);
  Check(status, "allocate native router status USM");
  Require(pointer != nullptr, "native router status USM is null");
  state.usm_allocations.push_back(pointer);
  const std::uint32_t zero = 0;
  Check(state.fill_usm(state.queue, pointer, &zero, sizeof(zero), bytes, 0,
                       nullptr, nullptr),
        "clear native router status USM");
  Check(clFinish(state.queue), "finish native router status clear");
  return pointer;
}

cl_program LoadBinaryProgram(OpenClState& state, const std::string& path) {
  const auto binary = ReadBytes(path);
  const std::size_t size = binary.size();
  const unsigned char* data = binary.data();
  cl_int binary_status = CL_SUCCESS;
  cl_int status = CL_SUCCESS;
  cl_program program = clCreateProgramWithBinary(state.context, 1,
      &state.device, &size, &data, &binary_status, &status);
  Check(status, "clCreateProgramWithBinary " + path);
  Check(binary_status, "binary status " + path);
  status = clBuildProgram(program, 1, &state.device, "", nullptr, nullptr);
  if (status != CL_SUCCESS) {
    const std::string log = ProgramLog(program, state.device);
    clReleaseProgram(program);
    Fail("native program build failed: " + log);
  }
  state.programs.push_back(program);
  return program;
}

cl_program BuildSourceProgram(OpenClState& state, const std::string& path) {
  const std::string source = ReadText(path);
  const char* data = source.data();
  const std::size_t size = source.size();
  cl_int status = CL_SUCCESS;
  cl_program program = clCreateProgramWithSource(
      state.context, 1, &data, &size, &status);
  Check(status, "clCreateProgramWithSource");
  status = clBuildProgram(
      program, 1, &state.device,
      "-cl-std=CL2.0 -cl-fp32-correctly-rounded-divide-sqrt",
      nullptr, nullptr);
  if (status != CL_SUCCESS) {
    const std::string log = ProgramLog(program, state.device);
    clReleaseProgram(program);
    Fail("support program build failed: " + log);
  }
  state.programs.push_back(program);
  return program;
}

cl_kernel CreateKernel(OpenClState& state, cl_program program,
                       const char* name) {
  cl_int status = CL_SUCCESS;
  cl_kernel kernel = clCreateKernel(program, name, &status);
  Check(status, std::string("clCreateKernel ") + name);
  state.kernels.push_back(kernel);
  return kernel;
}

void SetMem(cl_kernel kernel, cl_uint index, cl_mem memory,
            const char* label) {
  Check(clSetKernelArg(kernel, index, sizeof(memory), &memory), label);
}

void SetI64(cl_kernel kernel, cl_uint index, std::int64_t value,
            const char* label) {
  Check(clSetKernelArg(kernel, index, sizeof(value), &value), label);
}

void SetNativeCommon(cl_kernel kernel, cl_mem source, cl_mem weights,
                     cl_mem destination, cl_mem offsets, cl_mem source_scales,
                     cl_mem sum_low, cl_mem weight_scales, cl_mem min_codes,
                     cl_mem sum_high, cl_mem dmins, cl_mem extra,
                     cl_mem dispatch_coordinates,
                     cl_mem dispatch_metadata, std::int64_t k,
                     std::int64_t n, std::int64_t ldsrcq) {
  const std::int64_t ldsrc = k;
  const std::int64_t lddst = n == static_cast<std::int64_t>(kGateUpSize)
      ? static_cast<std::int64_t>(kIntermediateSize) : n;
  const std::int64_t ldweiq = n;
  const std::array<std::int64_t, 4> strides = {k * n, 1, k, 0};
  SetMem(kernel, 0, source, "set native source");
  SetI64(kernel, 1, ldsrc, "set native ldsrc");
  SetMem(kernel, 2, weights, "set native weights");
  Check(clSetKernelArg(kernel, 3, sizeof(strides), strides.data()),
        "set native strides");
  SetMem(kernel, 4, destination, "set native destination");
  SetI64(kernel, 5, lddst, "set native lddst");
  SetMem(kernel, 6, offsets, "set native offsets");
  SetMem(kernel, 7, dispatch_coordinates,
         "set native dispatch coordinates");
  SetMem(kernel, 8, source_scales, "set native source scales");
  SetMem(kernel, 9, sum_low, "set native low sums");
  SetI64(kernel, 10, ldsrcq, "set native ldsrcq");
  SetMem(kernel, 11, weight_scales, "set native weight scales");
  SetMem(kernel, 12, min_codes, "set native min codes");
  SetI64(kernel, 13, ldweiq, "set native ldweiq");
  SetI64(kernel, 14, n, "set native n");
  SetI64(kernel, 15, k, "set native k");
  if (n == static_cast<std::int64_t>(kGateUpSize)) {
    SetMem(kernel, 16, sum_high, "set native high sums");
    SetMem(kernel, 17, dmins, "set native dmins");
    SetMem(kernel, 18, extra, "set native F32 SwiGLU output");
    SetMem(kernel, 19, dispatch_metadata,
           "set native dispatch metadata");
  } else {
    SetMem(kernel, 16, extra, "set native extra");
    SetMem(kernel, 17, sum_high, "set native high sums");
    SetMem(kernel, 18, dmins, "set native dmins");
    SetMem(kernel, 19, dispatch_metadata,
           "set native final dispatch metadata");
  }
}

float HalfToFloat(std::uint16_t value) {
  const std::uint32_t sign = static_cast<std::uint32_t>(value & 0x8000U) << 16;
  std::uint32_t exponent = (value >> 10) & 0x1fU;
  std::uint32_t mantissa = value & 0x03ffU;
  std::uint32_t bits = 0;
  if (exponent == 0) {
    if (mantissa == 0) {
      bits = sign;
    } else {
      int shift = 0;
      while ((mantissa & 0x0400U) == 0) {
        mantissa <<= 1;
        ++shift;
      }
      mantissa &= 0x03ffU;
      bits = sign | static_cast<std::uint32_t>(127 - 15 - shift) << 23 |
          mantissa << 13;
    }
  } else if (exponent == 31) {
    bits = sign | 0x7f800000U | mantissa << 13;
  } else {
    bits = sign | (exponent + 112U) << 23 | mantissa << 13;
  }
  float result = 0.0f;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

struct CompareStats {
  std::size_t count = 0;
  std::size_t mismatch_count = 0;
  bool finite = true;
  double max_abs_diff = 0.0;
  double mean_abs_diff = 0.0;
  double rmse = 0.0;
  double cosine = 0.0;
  double relative_l2 = 0.0;
};

CompareStats Compare(const std::vector<float>& observed,
                     const std::vector<float>& expected) {
  Require(observed.size() == expected.size(), "comparison size mismatch");
  CompareStats stats;
  stats.count = observed.size();
  double abs_sum = 0.0;
  double squared_sum = 0.0;
  double dot = 0.0;
  double observed_norm = 0.0;
  double expected_norm = 0.0;
  for (std::size_t index = 0; index < observed.size(); ++index) {
    const double actual = observed[index];
    const double reference = expected[index];
    stats.finite = stats.finite && std::isfinite(actual);
    const double difference = std::abs(actual - reference);
    stats.max_abs_diff = std::max(stats.max_abs_diff, difference);
    abs_sum += difference;
    squared_sum += difference * difference;
    stats.mismatch_count += difference > kMismatchThreshold;
    dot += actual * reference;
    observed_norm += actual * actual;
    expected_norm += reference * reference;
  }
  if (stats.count != 0) {
    stats.mean_abs_diff = abs_sum / stats.count;
    stats.rmse = std::sqrt(squared_sum / stats.count);
  }
  if (observed_norm != 0.0 && expected_norm != 0.0) {
    stats.cosine = dot / std::sqrt(observed_norm * expected_norm);
  }
  if (expected_norm != 0.0) {
    stats.relative_l2 = std::sqrt(squared_sum / expected_norm);
  }
  return stats;
}

void PrintCompare(std::ostream& output, const char* name,
                  const CompareStats& stats) {
  output << "\"" << name << "\":{";
  output << "\"compared_value_count\":" << stats.count << ",";
  output << "\"cosine\":" << stats.cosine << ",";
  output << "\"finite\":" << std::boolalpha << stats.finite << ",";
  output << "\"max_abs_diff\":" << stats.max_abs_diff << ",";
  output << "\"mean_abs_diff\":" << stats.mean_abs_diff << ",";
  output << "\"mismatch_count\":" << stats.mismatch_count << ",";
  output << "\"relative_l2\":" << stats.relative_l2 << ",";
  output << "\"rmse\":" << stats.rmse << "},";
}

bool ComponentAccuracyPass(const CompareStats& stats) {
  return stats.finite && stats.cosine >= 0.999 &&
      stats.relative_l2 <= 0.002;
}

bool MapsAreNativeOnly() {
  std::ifstream maps("/proc/self/maps");
  std::string line;
  while (std::getline(maps, line)) {
    std::string lower = line;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char value) { return std::tolower(value); });
    if (lower.find("libdnnl") != std::string::npos ||
        lower.find("openvino") != std::string::npos) {
      return false;
    }
  }
  return true;
}

}  // namespace

namespace iq36 {

class GroupedS8U4PrefillRuntime::Impl {
 public:
  explicit Impl(const GroupedS8U4PrefillProgramConfig& config)
      : state_(CreateOpenCl(config.persistent_dispatch)),
        persistent_dispatch_(config.persistent_dispatch) {
    Require(!config.gateup_binary.empty() && !config.down_binary.empty() &&
                !config.kernels.empty(),
            "grouped prefill program paths are required");
    char device_name[256] = {};
    char driver_version[256] = {};
    Check(clGetDeviceInfo(state_.device, CL_DEVICE_NAME, sizeof(device_name),
                          device_name, nullptr),
          "clGetDeviceInfo resident name");
    Check(clGetDeviceInfo(state_.device, CL_DRIVER_VERSION,
                          sizeof(driver_version), driver_version, nullptr),
          "clGetDeviceInfo resident driver");
    device_name_ = device_name;
    driver_version_ = driver_version;

    const std::uint32_t zero = 0;
    dummy_ = CreateCopiedVector(state_, std::vector<std::uint32_t>{zero});
    offsets_ = CreateEmpty(state_, kExpertCount * sizeof(std::int32_t));
    token_map_ = CreateEmpty(state_, kAssignments * sizeof(std::int32_t));
    inverse_map_ = CreateEmpty(state_, kAssignments * sizeof(std::int32_t));
    compact_router_ = CreateEmpty(state_, kAssignments * sizeof(float));
    schedule_topk_ = CreateEmpty(
        state_, kAssignments * sizeof(std::int32_t));
    schedule_router_ = CreateEmpty(state_, kAssignments * sizeof(float));
    schedule_metadata_ = CreateEmpty(
        state_, kScheduleMetadataWords * sizeof(std::uint32_t));
    schedule_counts_ = CreateEmpty(
        state_, kExpertCount * sizeof(std::uint32_t));
    schedule_begins_ = CreateEmpty(
        state_, kExpertCount * sizeof(std::uint32_t));
    schedule_cursors_ = CreateEmpty(
        state_, kExpertCount * sizeof(std::uint32_t));
    schedule_gateup_bases_ = CreateEmpty(
        state_, kExpertCount * sizeof(std::uint32_t));
    schedule_down_bases_ = CreateEmpty(
        state_, kExpertCount * sizeof(std::uint32_t));
    schedule_native_gateup_bases_ = CreateEmpty(
        state_, kExpertCount * sizeof(std::uint32_t));
    schedule_native_down_bases_ = CreateEmpty(
        state_, kExpertCount * sizeof(std::uint32_t));
    schedule_partial_counts_ = CreateEmpty(
        state_, 32 * kExpertCount * sizeof(std::uint32_t));
    schedule_chunk_bases_ = CreateEmpty(
        state_, 32 * kExpertCount * sizeof(std::uint32_t));
    if (!config.router_binary.empty()) {
      router_logits_ = CreateEmpty(
          state_, kTokenCount * kExpertCount * sizeof(float));
      router_status_ = CreateZeroedDeviceUsm(
          state_, kRouterStatusBytes, 64);
    }
    input_ = CreateEmpty(
        state_, kTokenCount * kHiddenSize * sizeof(float));
    token_q8_ = CreateEmpty(state_, kTokenCount * kHiddenSize);
    token_scales_ = CreateEmpty(
        state_, kTokenCount * 8 * sizeof(float));
    token_sum_low_ = CreateEmpty(state_, kTokenCount * 64);
    token_sum_high_ = CreateEmpty(state_, kTokenCount * 64);
    grouped_q8_ = CreateEmpty(state_, kAssignments * kHiddenSize);
    grouped_scales_ = CreateEmpty(
        state_, kAssignments * 8 * sizeof(float));
    grouped_sum_low_ = CreateEmpty(state_, kAssignments * 64);
    grouped_sum_high_ = CreateEmpty(state_, kAssignments * 64);
    swiglu_ = CreateEmpty(
        state_, kAssignments * kIntermediateSize * sizeof(std::uint16_t));
    swiglu_f32_ = CreateEmpty(
        state_, kAssignments * kIntermediateSize * sizeof(float));
    down_q8_ = CreateEmpty(state_, kAssignments * kIntermediateSize);
    down_source_scales_ = CreateEmpty(
        state_, kAssignments * 2 * sizeof(float));
    down_sum_low_ = CreateEmpty(state_, kAssignments * 16);
    down_sum_high_ = CreateEmpty(state_, kAssignments * 16);
    q6_sums_ = CreateEmpty(
        state_, kAssignments * kExactQ6DownGroups * sizeof(std::int16_t));
    gateup_task_coordinates_ = CreateEmpty(
        state_, kAssignments * 8 * sizeof(std::uint32_t));
    down_task_coordinates_ = CreateEmpty(
        state_, kAssignments * 12 * sizeof(std::uint32_t));
    native_gateup_task_coordinates_ = CreateEmpty(
        state_, kNativeGateTaskCapacity * sizeof(std::uint32_t));
    native_down_task_coordinates_ = CreateEmpty(
        state_, kNativeDownTaskCapacity * sizeof(std::uint32_t));
    contributions_ = CreateEmpty(
        state_, kAssignments * kHiddenSize * sizeof(std::uint16_t));
    q6_contributions_f32_ = CreateEmpty(
        state_, kAssignments * kHiddenSize * sizeof(float));
    output_ = CreateEmpty(
        state_, kTokenCount * kHiddenSize * sizeof(float));

    cl_program support_program = BuildSourceProgram(state_, config.kernels);
    cl_program gateup_program =
        LoadBinaryProgram(state_, config.gateup_binary);
    cl_program down_program = LoadBinaryProgram(state_, config.down_binary);
    cl_program router_program = config.router_binary.empty()
        ? nullptr : LoadBinaryProgram(state_, config.router_binary);
    quantize_tokens_ = CreateKernel(
        state_, support_program, "iq36_quantize_tokens_q8");
    gather_ = CreateKernel(
        state_, support_program, "iq36_gather_quantized_q8");
    quantize_down_ = CreateKernel(
        state_, support_program, "iq36_quantize_swiglu_f32_q8");
    scatter_ = CreateKernel(
        state_, support_program, "iq36_scatter_f16_contributions");
    scatter_f32_ = CreateKernel(
        state_, support_program, "iq36_scatter_f32_contributions");
    schedule_reset_ = CreateKernel(
        state_, support_program, "iq36_grouped_schedule_reset_1024");
    schedule_count_ = CreateKernel(
        state_, support_program, "iq36_grouped_schedule_count_1024");
    schedule_prefix_ = CreateKernel(
        state_, support_program, "iq36_grouped_schedule_prefix_1024");
    schedule_scatter_ = CreateKernel(
        state_, support_program, "iq36_grouped_schedule_scatter_1024");
    schedule_tasks_ = CreateKernel(
        state_, support_program, "iq36_grouped_schedule_tasks_1024");
    schedule_native_tasks_ = CreateKernel(
        state_, support_program,
        "iq36_grouped_schedule_native_tasks_1024");
    router_topk_ = CreateKernel(
        state_, support_program, "iq36_router_topk8_1024");
    q4_exact_gateup_ = CreateKernel(
        state_, support_program, "iq36_grouped_q4_exact_block_gateup_m16");
    q4_exact_down_ = CreateKernel(
        state_, support_program, "iq36_grouped_q4_exact_block_down_m16_f32");
    gateup_ = CreateKernel(state_, gateup_program, "grouped_micro_gemm");
    down_ = CreateKernel(state_, down_program, "grouped_micro_gemm");
    if (router_program != nullptr) {
      native_router_ = CreateKernel(state_, router_program, "gemm_kernel");
      SetMem(router_topk_, 0, router_logits_, "set native router logits");
      SetMem(router_topk_, 1, schedule_topk_, "set native router top-k IDs");
      SetMem(router_topk_, 2, schedule_router_,
             "set native router normalized weights");
    }
    if (!config.q6_down_kernels.empty()) {
      cl_program q6_program = BuildSourceProgram(
          state_, config.q6_down_kernels);
      q6_combine_sums_ = CreateKernel(
          state_, support_program, "iq36_combine_q8_sums16");
      q6_exact_sums_ = CreateKernel(
          state_, support_program, "iq36_sum_q8_k16");
      q6_surrogate_down_ = CreateKernel(
          state_, q6_program,
          "iq36_grouped_s8_u8_q6_surrogate_down_m16_f16scale_compact");
      q6_exact_down_ = CreateKernel(
          state_, q6_program,
          "iq36_grouped_s8_u8_q6_exact_down_m16_compact_f32");
      q6_exact_block_down_ = CreateKernel(
          state_, q6_program,
          "iq36_grouped_s8_u8_q6_exact_block_down_m16_compact_f32");
    }
    BindDeviceScheduleKernels();
    stats_.context_create_count = 1;
    stats_.program_load_count = 3 + (q6_surrogate_down_ != nullptr ? 1 : 0) +
        (native_router_ != nullptr ? 1 : 0);
  }

  std::uint64_t LoadLayer(const GroupedS8U4PrefillLayerConfig& config) {
    Require(config.layer_index >= 0 && config.layer_index < 40,
            "grouped prefill layer index must be in [0, 40)");
    Require(!config.prep_dir.empty(),
            "grouped prefill layer prepack directory is required");
    for (const auto& layer : layers_) {
      Require(layer.layer_index != config.layer_index,
              "grouped prefill layer index is already loaded");
    }
    LayerResources layer;
    layer.handle = layers_.size() + 1;
    layer.layer_index = config.layer_index;
    layer.prep_dir = config.prep_dir;
    layer.exact_q4_gateup = config.exact_q4_gateup;
    layer.down_kind = config.down_kind;
    if (!config.router_weights.empty()) {
      Require(native_router_ != nullptr,
              "layer router weights require the native router program");
      layer.router_weights = CreateCopied(
          state_, config.router_weights, kRouterWeightBytes);
    }
    layer.gateup_weights = CreateCopied(
        state_, JoinPath(config.prep_dir, "gateup-weights.bin"),
        kGateUpWeightBytes);
    if (config.exact_q4_gateup) {
      layer.gateup_scale_codes = CreateCopied(
          state_, JoinPath(config.prep_dir, "gateup-scale-codes.bin"),
          kGateUpExactScaleCodeBytes);
      layer.gateup_block_scales = CreateCopied(
          state_, JoinPath(config.prep_dir, "gateup-block-ds.bin"),
          kGateUpExactBlockScaleBytes);
    } else {
      layer.gateup_scales = CreateCopied(
          state_, JoinPath(config.prep_dir, "gateup-scales.bin"),
          kGateUpScaleBytes);
    }
    layer.gateup_min_codes = CreateCopied(
        state_, JoinPath(config.prep_dir, "gateup-min-codes.bin"),
        kGateUpMinCodeBytes);
    layer.gateup_dmins = CreateCopied(
        state_, JoinPath(config.prep_dir, "gateup-dmins.bin"),
        kGateUpDminBytes);
    if (IsQ6DownKind(config.down_kind)) {
      Require(q6_surrogate_down_ != nullptr && q6_exact_down_ != nullptr &&
                  q6_exact_block_down_ != nullptr,
              "Q6 layer requires the Q6 down program");
      layer.down_weights = CreateCopied(
          state_, JoinPath(
              config.prep_dir,
              config.down_kind == GroupedPrefillDownKind::kQ6U8ExactPer16 ||
                      config.down_kind ==
                          GroupedPrefillDownKind::kQ6U8ExactBlock
                  ? "q6-down-exact-per16-values-u8.bin"
                  : "q6-down-weights-u8.bin"),
          kQ6DownWeightBytes);
      if (config.down_kind == GroupedPrefillDownKind::kQ6U8ExactBlock) {
        layer.down_scales = CreateCopied(
            state_, JoinPath(
                config.prep_dir, "q6-down-exact-block-scales-i8.bin"),
            kExactBlockQ6DownIntegerScaleBytes);
        layer.down_block_scales = CreateCopied(
            state_, JoinPath(
                config.prep_dir, "q6-down-exact-block-d-f32.bin"),
            kExactBlockQ6DownBlockScaleBytes);
      } else {
        layer.down_scales = CreateCopied(
            state_, JoinPath(
                config.prep_dir,
                config.down_kind == GroupedPrefillDownKind::kQ6U8ExactPer16
                    ? "q6-down-exact-per16-scales-f32.bin"
                    : "q6-down-scales-f16.bin"),
            config.down_kind == GroupedPrefillDownKind::kQ6U8ExactPer16
                ? kExactQ6DownScaleBytes
                : kQ6DownScaleBytes);
      }
    } else if (UsesExactQ4Down(config.down_kind)) {
      layer.down_weights = CreateCopied(
          state_, JoinPath(config.prep_dir, "down-weights.bin"),
          kDownWeightBytes);
      layer.down_scale_codes = CreateCopied(
          state_, JoinPath(config.prep_dir, "down-scale-codes.bin"),
          kDownExactScaleCodeBytes);
      layer.down_block_scales = CreateCopied(
          state_, JoinPath(config.prep_dir, "down-block-ds.bin"),
          kDownExactBlockScaleBytes);
      layer.down_min_codes = CreateCopied(
          state_, JoinPath(config.prep_dir, "down-min-codes.bin"),
          kDownMinCodeBytes);
      layer.down_dmins = CreateCopied(
          state_, JoinPath(config.prep_dir, "down-dmins.bin"),
          kDownDminBytes);
    } else {
      layer.down_weights = CreateCopied(
          state_, JoinPath(config.prep_dir, "down-weights.bin"),
          kDownWeightBytes);
      layer.down_scales = CreateCopied(
          state_, JoinPath(config.prep_dir, "down-scales.bin"),
          kDownScaleBytes);
      layer.down_min_codes = CreateCopied(
          state_, JoinPath(config.prep_dir, "down-min-codes.bin"),
          kDownMinCodeBytes);
      layer.down_dmins = CreateCopied(
          state_, JoinPath(config.prep_dir, "down-dmins.bin"),
          kDownDminBytes);
    }
    layers_.push_back(std::move(layer));
    ++stats_.layer_load_count;
    stats_.layer_count = layers_.size();
    const std::uint64_t gateup_bytes = config.exact_q4_gateup
        ? kExactGateUpResidentWeightBytes : kGateUpResidentWeightBytes;
    const std::uint64_t down_bytes =
        config.down_kind == GroupedPrefillDownKind::kQ6U8ExactBlock
            ? kQ6DownWeightBytes + kExactBlockQ6DownIntegerScaleBytes +
                  kExactBlockQ6DownBlockScaleBytes
            : (config.down_kind == GroupedPrefillDownKind::kQ6U8ExactPer16
                   ? kQ6DownWeightBytes + kExactQ6DownScaleBytes
                   : (config.down_kind == GroupedPrefillDownKind::kQ6U8Surrogate
                          ? kQ6DownWeightBytes + kQ6DownScaleBytes
                          : (UsesExactQ4Down(config.down_kind)
                                 ? kExactQ4DownResidentWeightBytes
                                 : kQ4DownResidentWeightBytes)));
    stats_.resident_weight_bytes += gateup_bytes + down_bytes +
        (layers_.back().router_weights != nullptr ? kRouterWeightBytes : 0);
    return layers_.back().handle;
  }

  GroupedS8U4PrefillRun RunLayer(
      std::uint64_t layer_handle,
      const GroupedS8U4PrefillInput& input) {
    const LayerResources& layer = LayerForHandle(layer_handle);
    Require(input.hidden_states.size() == kTokenCount * kHiddenSize,
            "grouped prefill hidden-state count mismatch");
    Require(!input.native_router || input.device_schedule,
            "native router requires the device schedule");
    if (persistent_dispatch_) {
      Require(input.native_router && input.device_schedule,
              "persistent dispatch requires the native router schedule");
      Require(!layer.exact_q4_gateup && !IsQ6DownKind(layer.down_kind) &&
                  !UsesExactQ4Down(layer.down_kind),
              "persistent dispatch requires native Q4 gate/up and down");
    }
    if (input.native_router) {
      Require(native_router_ != nullptr && layer.router_weights != nullptr,
              "native router program or layer weights are unavailable");
    } else {
      Require(input.topk_stride >= 8 * sizeof(std::int32_t),
              "grouped prefill top-k stride is too small");
      Require(input.router_weights.size() == kAssignments,
              "grouped prefill router weight count mismatch");
    }
    Require(input.swiglu_override_source_order.empty() ||
            input.swiglu_override_source_order.size() ==
                kAssignments * kIntermediateSize,
            "grouped prefill SwiGLU override count mismatch");
    Require(input.down_override_source_order.empty() ||
            input.down_override_source_order.size() ==
                kAssignments * kHiddenSize,
            "grouped prefill down override count mismatch");
    Require(input.down_override_source_order.empty() ||
            UsesF32Contributions(layer.down_kind),
            "grouped prefill down override requires F32 contributions");
    Require(input.warmup >= 0 && input.repeat > 0,
            "grouped prefill warmup/repeat are invalid");

    GroupedS8U4PrefillRun run;
    run.down_kind = layer.down_kind;
    run.device_schedule = input.device_schedule;
    run.native_router = input.native_router;
    run.persistent_dispatch = persistent_dispatch_;
    run.persistent_workgroup_count =
        persistent_dispatch_ ? kPersistentWorkgroupCount : 0;
    const auto input_upload_begin = std::chrono::steady_clock::now();
    WriteBuffer(input_, input.hidden_states.data(),
                input.hidden_states.size() * sizeof(float), true,
                "write resident grouped input");
    run.timing.input_upload_us = ElapsedUs(input_upload_begin);
    std::vector<float> grouped_swiglu_override;
    std::vector<float> grouped_down_override;
    cl_event schedule_start_event = nullptr;
    cl_event schedule_end_event = nullptr;
    if (input.device_schedule) {
      Require(input.swiglu_override_source_order.empty() &&
                  input.down_override_source_order.empty(),
              "device schedule does not support diagnostic overrides");
      std::size_t host_schedule_upload_bytes = 0;
      std::chrono::steady_clock::time_point device_begin;
      if (input.native_router) {
        BindNativeRouter(layer);
        device_begin = std::chrono::steady_clock::now();
        constexpr std::array<std::size_t, 3> router_global = {1536, 4, 1};
        constexpr std::array<std::size_t, 3> router_local = {32, 4, 1};
        constexpr std::size_t topk_global = kTokenCount * kExpertCount;
        constexpr std::size_t topk_local = 256;
        Check(clEnqueueNDRangeKernel(
                  state_.queue, native_router_, 3, nullptr,
                  router_global.data(), router_local.data(), 0, nullptr,
                  persistent_dispatch_ ? &schedule_start_event : nullptr),
              "enqueue resident native router");
        Check(clEnqueueNDRangeKernel(
                  state_.queue, router_topk_, 1, nullptr, &topk_global,
                  &topk_local, 0, nullptr, nullptr),
              "enqueue resident native router top-k");
        ++stats_.native_router_run_count;
      } else {
        const auto prepare_begin = std::chrono::steady_clock::now();
        std::vector<std::int32_t> topk_ids(kAssignments);
        Require(input.topk.size() >=
                    (kTokenCount - 1) * input.topk_stride +
                        8 * sizeof(std::int32_t),
                "device schedule top-k payload is truncated");
        for (std::size_t token = 0; token < kTokenCount; ++token) {
          std::memcpy(topk_ids.data() + token * 8,
                      input.topk.data() + token * input.topk_stride,
                      8 * sizeof(std::int32_t));
        }
        run.timing.schedule_prepare_us = ElapsedUs(prepare_begin);
        device_begin = std::chrono::steady_clock::now();
        WriteBuffer(schedule_topk_, topk_ids.data(),
                    topk_ids.size() * sizeof(std::int32_t), false,
                    "write device schedule top-k");
        WriteBuffer(schedule_router_, input.router_weights.data(),
                    input.router_weights.size() * sizeof(float), false,
                    "write device schedule router weights");
        host_schedule_upload_bytes =
            topk_ids.size() * sizeof(std::int32_t) +
            input.router_weights.size() * sizeof(float);
      }
      constexpr std::size_t schedule_local = 256;
      constexpr std::size_t reset_global = 256;
      constexpr std::size_t assignment_global = kAssignments;
      constexpr std::size_t prefix_global = 256;
      constexpr std::size_t task_global = kExpertCount * 256;
      Check(clEnqueueNDRangeKernel(
                state_.queue, schedule_reset_, 1, nullptr, &reset_global,
                &schedule_local, 0, nullptr, nullptr),
            "enqueue device schedule reset");
      Check(clEnqueueNDRangeKernel(
                state_.queue, schedule_count_, 1, nullptr, &assignment_global,
                &schedule_local, 0, nullptr, nullptr),
            "enqueue device schedule count");
      Check(clEnqueueNDRangeKernel(
                state_.queue, schedule_prefix_, 1, nullptr, &prefix_global,
                &schedule_local, 0, nullptr, nullptr),
            "enqueue device schedule prefix");
      Check(clEnqueueNDRangeKernel(
                state_.queue, schedule_scatter_, 1, nullptr,
                &assignment_global, &schedule_local, 0, nullptr, nullptr),
            "enqueue device schedule scatter");
      if (persistent_dispatch_) {
        Check(clEnqueueNDRangeKernel(
                  state_.queue, schedule_native_tasks_, 1, nullptr,
                  &task_global, &schedule_local, 0, nullptr,
                  &schedule_end_event),
              "enqueue native device schedule tasks");
        ++stats_.persistent_dispatch_run_count;
      } else {
        Check(clEnqueueNDRangeKernel(
                  state_.queue, schedule_tasks_, 1, nullptr, &task_global,
                  &schedule_local, 0, nullptr, nullptr),
              "enqueue device schedule tasks");
        std::array<std::uint32_t, 5> metadata{};
        ReadBuffer(schedule_metadata_, metadata.data(),
                   metadata.size() * sizeof(std::uint32_t),
                   "read device schedule metadata");
        Require(metadata[4] == 0,
                "device schedule metadata reports an error");
        run.timing.device_schedule_us = ElapsedUs(device_begin);
        run.timing.schedule_upload_us = run.timing.device_schedule_us;
        run.timing.schedule_setup_us = run.timing.schedule_prepare_us +
            run.timing.schedule_upload_us;
        run.gateup_work_tile_count = metadata[0];
        run.q6_work_tile_count = metadata[1];
        run.active_experts = metadata[2];
        run.max_group_size = metadata[3];
        run.native_global_y = ((run.max_group_size + 31) / 32) * 4;
        stats_.device_schedule_host_read_bytes +=
            metadata.size() * sizeof(std::uint32_t);
      }
      if (input.capture_intermediates) {
        run.inverse_map.resize(kAssignments);
        ReadBuffer(inverse_map_, run.inverse_map.data(),
                   run.inverse_map.size() * sizeof(std::int32_t),
                   "read device schedule inverse map");
        stats_.device_schedule_host_read_bytes +=
            run.inverse_map.size() * sizeof(std::int32_t);
      }
      ++stats_.device_schedule_run_count;
      stats_.device_schedule_host_upload_bytes += host_schedule_upload_bytes;
    } else {
      const auto schedule_begin = std::chrono::steady_clock::now();
      const GroupedSchedule schedule = BuildGroupedSchedule(
          input.topk, input.topk_stride, input.router_weights,
          layer.exact_q4_gateup,
          IsQ6DownKind(layer.down_kind) || UsesExactQ4Down(layer.down_kind));
      run.timing.schedule_prepare_us = ElapsedUs(schedule_begin);
      run.active_experts = schedule.active_experts;
      run.max_group_size = schedule.max_group;
      run.native_global_y = ((schedule.max_group + 31) / 32) * 4;
      run.gateup_work_tile_count = schedule.gateup_task_count;
      run.q6_work_tile_count = schedule.down_task_count;
      run.inverse_map = schedule.inverse_map;
      if (!input.swiglu_override_source_order.empty()) {
        grouped_swiglu_override.resize(kAssignments * kIntermediateSize);
        for (std::size_t source = 0; source < kAssignments; ++source) {
          const std::size_t grouped = static_cast<std::size_t>(
              schedule.inverse_map[source]);
          std::copy_n(
              input.swiglu_override_source_order.data() +
                  source * kIntermediateSize,
              kIntermediateSize,
              grouped_swiglu_override.data() + grouped * kIntermediateSize);
        }
      }
      if (!input.down_override_source_order.empty()) {
        grouped_down_override.resize(kAssignments * kHiddenSize);
        for (std::size_t source = 0; source < kAssignments; ++source) {
          const std::size_t grouped = static_cast<std::size_t>(
              schedule.inverse_map[source]);
          const float router = input.router_weights[source];
          for (std::size_t hidden = 0; hidden < kHiddenSize; ++hidden) {
            grouped_down_override[grouped * kHiddenSize + hidden] =
                input.down_override_source_order[
                    source * kHiddenSize + hidden] * router;
          }
        }
      }
      const auto schedule_upload_begin = std::chrono::steady_clock::now();
      WriteBuffer(offsets_, schedule.offsets.data(),
                  schedule.offsets.size() * sizeof(std::int32_t), false,
                  "write resident grouped offsets");
      WriteBuffer(token_map_, schedule.token_map.data(),
                  schedule.token_map.size() * sizeof(std::int32_t), false,
                  "write resident grouped token map");
      WriteBuffer(inverse_map_, schedule.inverse_map.data(),
                  schedule.inverse_map.size() * sizeof(std::int32_t), false,
                  "write resident grouped inverse map");
      WriteBuffer(compact_router_, schedule.compact_router_weights.data(),
                  schedule.compact_router_weights.size() * sizeof(float),
                  false, "write resident grouped router weights");
      if (layer.exact_q4_gateup) {
        WriteBuffer(gateup_task_coordinates_,
                    schedule.gateup_task_coordinates.data(),
                    schedule.gateup_task_coordinates.size() *
                        sizeof(schedule.gateup_task_coordinates.front()),
                    false, "write resident exact-Q4 gate/up coordinates");
      }
      if (IsQ6DownKind(layer.down_kind) || UsesExactQ4Down(layer.down_kind)) {
        WriteBuffer(
            down_task_coordinates_, schedule.down_task_coordinates.data(),
            schedule.down_task_coordinates.size() *
                sizeof(schedule.down_task_coordinates.front()),
            false, "write resident grouped down coordinates");
      }
      Check(clFinish(state_.queue),
            "finish resident grouped schedule upload");
      run.timing.schedule_upload_us = ElapsedUs(schedule_upload_begin);
      run.timing.schedule_setup_us = run.timing.schedule_prepare_us +
          run.timing.schedule_upload_us;
    }

    BindKernels(layer);
    constexpr std::size_t local = 256;
    const std::size_t token_quant_global = kTokenCount * 8 * local;
    const std::size_t gather_global = kAssignments * kHiddenSize;
    const std::size_t down_quant_global = kAssignments * 2 * local;
    const std::size_t scatter_global = kTokenCount * kHiddenSize;
    constexpr std::array<std::size_t, 3> native_local = {32, 4, 1};
    const std::array<std::size_t, 3> gateup_global = {
        512, run.native_global_y, kExpertCount};
    const std::array<std::size_t, 3> down_global = {
        1024, run.native_global_y, kExpertCount};
    constexpr std::array<std::size_t, 3> persistent_global = {
        32, 4, kPersistentWorkgroupCount};
    const std::size_t q6_down_local = 16;
    const std::size_t q6_down_global =
        run.q6_work_tile_count * q6_down_local;
    const std::size_t q4_gateup_local = 16;
    const std::size_t q4_gateup_global =
        run.gateup_work_tile_count * q4_gateup_local;

    const auto EnqueueGather = [&]() {
      Check(clEnqueueNDRangeKernel(state_.queue, quantize_tokens_, 1, nullptr,
                                   &token_quant_global, &local, 0, nullptr,
                                   nullptr),
            "enqueue resident token quantize");
      Check(clEnqueueNDRangeKernel(state_.queue, gather_, 1, nullptr,
                                   &gather_global, &local, 0, nullptr, nullptr),
            "enqueue resident grouped gather");
    };
    const auto EnqueueGateUp = [&]() {
      if (layer.exact_q4_gateup) {
        Check(clEnqueueNDRangeKernel(
                  state_.queue, q4_exact_gateup_, 1, nullptr,
                  &q4_gateup_global, &q4_gateup_local, 0, nullptr, nullptr),
              "enqueue resident exact-block Q4 gate/up");
      } else {
        const auto& global = persistent_dispatch_
            ? persistent_global : gateup_global;
        Check(clEnqueueNDRangeKernel(
                  state_.queue, gateup_, 3, nullptr, global.data(),
                  native_local.data(), 0, nullptr, nullptr),
              "enqueue resident native gate/up");
      }
      if (!grouped_swiglu_override.empty()) {
        WriteBuffer(swiglu_f32_, grouped_swiglu_override.data(),
                    grouped_swiglu_override.size() * sizeof(float), false,
                    "write diagnostic grouped SwiGLU override");
      }
    };
    const auto EnqueueDownQuantize = [&]() {
      Check(clEnqueueNDRangeKernel(state_.queue, quantize_down_, 1, nullptr,
                                   &down_quant_global, &local, 0, nullptr,
                                   nullptr),
            "enqueue resident down quantize");
      if (layer.down_kind == GroupedPrefillDownKind::kQ6U8Surrogate) {
        const std::size_t sums_global = kAssignments * kDownGroups;
        Check(clEnqueueNDRangeKernel(state_.queue, q6_combine_sums_, 1,
                                     nullptr, &sums_global, &local, 0,
                                     nullptr, nullptr),
              "enqueue resident Q6 sum combine");
      } else if (layer.down_kind ==
                     GroupedPrefillDownKind::kQ6U8ExactPer16 ||
                 layer.down_kind ==
                     GroupedPrefillDownKind::kQ6U8ExactBlock) {
        const std::size_t sums_global =
            kAssignments * kExactQ6DownGroups;
        Check(clEnqueueNDRangeKernel(state_.queue, q6_exact_sums_, 1,
                                     nullptr, &sums_global, &local, 0,
                                     nullptr, nullptr),
              "enqueue resident exact-Q6 K16 sums");
      }
    };
    const auto EnqueueDown = [&]() {
      if (IsQ6DownKind(layer.down_kind)) {
        cl_kernel q6_kernel =
            layer.down_kind == GroupedPrefillDownKind::kQ6U8ExactBlock
                ? q6_exact_block_down_
                : (layer.down_kind == GroupedPrefillDownKind::kQ6U8ExactPer16
                       ? q6_exact_down_
                       : q6_surrogate_down_);
        Check(clEnqueueNDRangeKernel(
                  state_.queue, q6_kernel, 1, nullptr, &q6_down_global,
                  &q6_down_local, 0, nullptr, nullptr),
              "enqueue resident Q6 down");
      } else if (UsesExactQ4Down(layer.down_kind)) {
        Check(clEnqueueNDRangeKernel(
                  state_.queue, q4_exact_down_, 1, nullptr, &q6_down_global,
                  &q6_down_local, 0, nullptr, nullptr),
              "enqueue resident exact-block Q4 down");
      } else {
        const auto& global = persistent_dispatch_
            ? persistent_global : down_global;
        Check(clEnqueueNDRangeKernel(
                  state_.queue, down_, 3, nullptr, global.data(),
                  native_local.data(), 0, nullptr, nullptr),
              "enqueue resident native down");
      }
      if (!grouped_down_override.empty()) {
        WriteBuffer(q6_contributions_f32_, grouped_down_override.data(),
                    grouped_down_override.size() * sizeof(float), false,
                    "write diagnostic grouped down override");
      }
    };
    const auto EnqueueScatter = [&]() {
      cl_kernel scatter_kernel =
          UsesF32Contributions(layer.down_kind) ? scatter_f32_ : scatter_;
      Check(clEnqueueNDRangeKernel(state_.queue, scatter_kernel, 1, nullptr,
                                   &scatter_global, &local, 0, nullptr,
                                   nullptr),
            "enqueue resident scatter");
    };
    const auto Execute = [&]() {
      EnqueueGather();
      EnqueueGateUp();
      EnqueueDownQuantize();
      if (input.execute_down) EnqueueDown();
      EnqueueScatter();
      Check(clFinish(state_.queue), "finish resident grouped component");
    };
    const auto Timed = [&](const auto& enqueue) {
      const auto begin = std::chrono::steady_clock::now();
      enqueue();
      Check(clFinish(state_.queue), "finish resident grouped stage");
      return ElapsedUs(begin);
    };

    for (int iteration = 0; iteration < input.warmup; ++iteration) Execute();
    run.timing.stage_us = {Timed(EnqueueGather), Timed(EnqueueGateUp),
        Timed(EnqueueDownQuantize),
        input.execute_down ? Timed(EnqueueDown) : 0.0,
        Timed(EnqueueScatter)};
    if (persistent_dispatch_) {
      Require(schedule_start_event != nullptr && schedule_end_event != nullptr,
              "persistent schedule profiling events are unavailable");
      cl_ulong device_start = 0;
      cl_ulong device_end = 0;
      Check(clGetEventProfilingInfo(
                schedule_start_event, CL_PROFILING_COMMAND_START,
                sizeof(device_start), &device_start, nullptr),
            "read persistent schedule profile start");
      Check(clGetEventProfilingInfo(
                schedule_end_event, CL_PROFILING_COMMAND_END,
                sizeof(device_end), &device_end, nullptr),
            "read persistent schedule profile end");
      Require(device_end >= device_start,
              "persistent schedule profile interval is invalid");
      run.timing.device_schedule_us =
          static_cast<double>(device_end - device_start) / 1000.0;
      run.timing.schedule_upload_us = run.timing.device_schedule_us;
      run.timing.schedule_setup_us = run.timing.schedule_prepare_us +
          run.timing.schedule_upload_us;
      Check(clReleaseEvent(schedule_start_event),
            "release persistent schedule start event");
      Check(clReleaseEvent(schedule_end_event),
            "release persistent schedule end event");
    }
    run.timing.samples_us.reserve(input.repeat);
    for (int iteration = 0; iteration < input.repeat; ++iteration) {
      const auto begin = std::chrono::steady_clock::now();
      Execute();
      run.timing.samples_us.push_back(ElapsedUs(begin));
    }
    std::vector<double> sorted = run.timing.samples_us;
    std::sort(sorted.begin(), sorted.end());
    run.timing.minimum_us = sorted.front();
    run.timing.median_us = sorted[sorted.size() / 2];
    run.timing.mean_us = std::accumulate(
        run.timing.samples_us.begin(), run.timing.samples_us.end(), 0.0) /
        run.timing.samples_us.size();
    run.timing.complete_minimum_us =
        run.timing.minimum_us + run.timing.schedule_setup_us;

    run.output.resize(kTokenCount * kHiddenSize);
    ReadBuffer(output_, run.output.data(), run.output.size() * sizeof(float),
               "read resident grouped output");
    if (input.capture_intermediates) {
      run.grouped_swiglu_f16.resize(kAssignments * kIntermediateSize);
      ReadBuffer(swiglu_, run.grouped_swiglu_f16.data(),
                 run.grouped_swiglu_f16.size() * sizeof(std::uint16_t),
                 "read resident grouped swiglu");
      run.grouped_swiglu_f32.resize(kAssignments * kIntermediateSize);
      ReadBuffer(swiglu_f32_, run.grouped_swiglu_f32.data(),
                 run.grouped_swiglu_f32.size() * sizeof(float),
                 "read resident grouped F32 swiglu");
      run.grouped_down_q8.resize(kAssignments * kIntermediateSize);
      ReadBuffer(down_q8_, run.grouped_down_q8.data(),
                 run.grouped_down_q8.size() * sizeof(std::int8_t),
                 "read resident grouped down Q8");
      run.grouped_down_scales.resize(kAssignments * kDownBlocks);
      ReadBuffer(down_source_scales_, run.grouped_down_scales.data(),
                 run.grouped_down_scales.size() * sizeof(float),
                 "read resident grouped down Q8 scales");
      if (UsesF32Contributions(layer.down_kind)) {
        run.grouped_contributions_f32.resize(kAssignments * kHiddenSize);
        ReadBuffer(q6_contributions_f32_,
                   run.grouped_contributions_f32.data(),
                   run.grouped_contributions_f32.size() * sizeof(float),
                   "read resident exact-Q6 F32 contributions");
      } else {
        run.grouped_contributions_f16.resize(kAssignments * kHiddenSize);
        ReadBuffer(contributions_, run.grouped_contributions_f16.data(),
                   run.grouped_contributions_f16.size() *
                       sizeof(std::uint16_t),
                   "read resident grouped contributions");
      }
    }
    run.maps_native_only = MapsAreNativeOnly();
    ++stats_.run_count;
    return run;
  }

  const std::string& device_name() const { return device_name_; }
  const std::string& driver_version() const { return driver_version_; }
  GroupedS8U4PrefillRuntimeStats stats() const { return stats_; }

 private:
  struct LayerResources {
    std::uint64_t handle = 0;
    int layer_index = -1;
    std::string prep_dir;
    bool exact_q4_gateup = false;
    GroupedPrefillDownKind down_kind = GroupedPrefillDownKind::kQ4U4;
    cl_mem router_weights = nullptr;
    cl_mem gateup_weights = nullptr;
    cl_mem gateup_scales = nullptr;
    cl_mem gateup_scale_codes = nullptr;
    cl_mem gateup_block_scales = nullptr;
    cl_mem gateup_min_codes = nullptr;
    cl_mem gateup_dmins = nullptr;
    cl_mem down_weights = nullptr;
    cl_mem down_scales = nullptr;
    cl_mem down_scale_codes = nullptr;
    cl_mem down_block_scales = nullptr;
    cl_mem down_min_codes = nullptr;
    cl_mem down_dmins = nullptr;
  };

  static double ElapsedUs(
      const std::chrono::steady_clock::time_point& begin) {
    return std::chrono::duration<double, std::micro>(
               std::chrono::steady_clock::now() - begin)
        .count();
  }

  void WriteBuffer(cl_mem buffer, const void* data, std::size_t bytes,
                   bool blocking, const char* label) {
    Check(clEnqueueWriteBuffer(state_.queue, buffer,
                               blocking ? CL_TRUE : CL_FALSE, 0, bytes, data,
                               0, nullptr, nullptr),
          label);
  }

  void ReadBuffer(cl_mem buffer, void* data, std::size_t bytes,
                  const char* label) {
    Check(clEnqueueReadBuffer(state_.queue, buffer, CL_TRUE, 0, bytes, data,
                              0, nullptr, nullptr),
          label);
  }

  const LayerResources& LayerForHandle(std::uint64_t handle) const {
    Require(handle > 0 && handle <= layers_.size(),
            "grouped prefill layer handle is invalid");
    return layers_[static_cast<std::size_t>(handle - 1)];
  }

  void BindNativeRouter(const LayerResources& layer) {
    const auto SetValue = [&](cl_uint index, const auto& value,
                              const char* label) {
      Check(clSetKernelArg(native_router_, index, sizeof(value), &value),
            label);
    };
    SetMem(native_router_, 0, layer.router_weights,
           "set native router weights");
    SetMem(native_router_, 1, input_, "set native router input");
    SetMem(native_router_, 2, router_logits_, "set native router output");
    const std::int64_t zero_offset = 0;
    SetValue(3, zero_offset, "set native router weight offset");
    SetValue(4, zero_offset, "set native router input offset");
    SetValue(5, zero_offset, "set native router output offset");
    const std::array<std::int32_t, 6> dimensions = {
        2048, 2048, 256, 256, 1024, 2048};
    for (cl_uint index = 0; index < dimensions.size(); ++index) {
      SetValue(6 + index, dimensions[index],
               "set native router matrix dimension");
    }
    const float alpha = 1.0f;
    const float beta = 0.0f;
    SetValue(12, alpha, "set native router alpha");
    SetValue(13, beta, "set native router beta");
    const std::uint32_t flags = 0;
    SetValue(14, flags, "set native router flags");
    Check(state_.set_usm_arg(native_router_, 15, router_status_),
          "set native router status USM");
    const std::array<std::uint32_t, 7> dispatch = {
        4U, 16U, 0U, 2776U, 64U, 4228890877U, 48U};
    for (cl_uint index = 0; index < dispatch.size(); ++index) {
      SetValue(16 + index, dispatch[index],
               "set native router dispatch specialization");
    }
    Check(clSetKernelArg(native_router_, 23, 0, nullptr),
          "set native router local memory");
  }

  void BindDeviceScheduleKernels() {
    const std::array<cl_mem, 3> reset_args = {
        schedule_counts_, schedule_cursors_, schedule_metadata_};
    for (cl_uint index = 0; index < reset_args.size(); ++index) {
      SetMem(schedule_reset_, index, reset_args[index],
             "set device schedule reset arg");
    }
    const std::array<cl_mem, 3> count_args = {
        schedule_topk_, schedule_partial_counts_, schedule_metadata_};
    for (cl_uint index = 0; index < count_args.size(); ++index) {
      SetMem(schedule_count_, index, count_args[index],
             "set device schedule count arg");
    }
    const std::array<cl_mem, 11> prefix_args = {
        schedule_partial_counts_, schedule_counts_, schedule_begins_,
        schedule_cursors_, offsets_, schedule_gateup_bases_,
        schedule_down_bases_, schedule_native_gateup_bases_,
        schedule_native_down_bases_, schedule_chunk_bases_,
        schedule_metadata_};
    for (cl_uint index = 0; index < prefix_args.size(); ++index) {
      SetMem(schedule_prefix_, index, prefix_args[index],
             "set device schedule prefix arg");
    }
    const std::array<cl_mem, 6> scatter_args = {
        schedule_topk_, schedule_router_, schedule_chunk_bases_, token_map_,
        inverse_map_, compact_router_};
    for (cl_uint index = 0; index < scatter_args.size(); ++index) {
      SetMem(schedule_scatter_, index, scatter_args[index],
             "set device schedule scatter arg");
    }
    const std::array<cl_mem, 8> task_args = {
        schedule_counts_, schedule_cursors_, offsets_,
        schedule_gateup_bases_, schedule_down_bases_,
        gateup_task_coordinates_, down_task_coordinates_, schedule_metadata_};
    for (cl_uint index = 0; index < task_args.size(); ++index) {
      SetMem(schedule_tasks_, index, task_args[index],
             "set device schedule task arg");
    }
    const std::array<cl_mem, 8> native_task_args = {
        schedule_counts_, schedule_cursors_, offsets_,
        schedule_native_gateup_bases_, schedule_native_down_bases_,
        native_gateup_task_coordinates_, native_down_task_coordinates_,
        schedule_metadata_};
    for (cl_uint index = 0; index < native_task_args.size(); ++index) {
      SetMem(schedule_native_tasks_, index, native_task_args[index],
             "set native device schedule task arg");
    }
  }

  void BindKernels(const LayerResources& layer) {
    const cl_uint token_rows = kTokenCount;
    const std::array<cl_mem, 5> quantize_args = {input_, token_q8_,
        token_scales_, token_sum_low_, token_sum_high_};
    for (cl_uint index = 0; index < quantize_args.size(); ++index) {
      SetMem(quantize_tokens_, index, quantize_args[index],
             "set resident token quantize arg");
    }
    Check(clSetKernelArg(quantize_tokens_, 5, sizeof(token_rows), &token_rows),
          "set resident token row count");
    const cl_uint assignment_rows = kAssignments;
    const std::array<cl_mem, 9> gather_args = {token_map_, token_q8_,
        token_scales_, token_sum_low_, token_sum_high_, grouped_q8_,
        grouped_scales_, grouped_sum_low_, grouped_sum_high_};
    for (cl_uint index = 0; index < gather_args.size(); ++index) {
      SetMem(gather_, index, gather_args[index],
             "set resident grouped gather arg");
    }
    Check(clSetKernelArg(gather_, 9, sizeof(assignment_rows), &assignment_rows),
          "set resident grouped row count");
    const std::array<cl_mem, 5> down_quantize_args = {swiglu_f32_, down_q8_,
        down_source_scales_, down_sum_low_, down_sum_high_};
    for (cl_uint index = 0; index < down_quantize_args.size(); ++index) {
      SetMem(quantize_down_, index, down_quantize_args[index],
             "set resident down quantize arg");
    }
    Check(clSetKernelArg(
              quantize_down_, 5, sizeof(assignment_rows), &assignment_rows),
          "set resident down row count");
    SetMem(scatter_, 0, contributions_, "set resident scatter contributions");
    SetMem(scatter_, 1, inverse_map_, "set resident scatter inverse map");
    SetMem(scatter_, 2, output_, "set resident scatter output");
    SetMem(scatter_f32_, 0, q6_contributions_f32_,
           "set exact-Q6 F32 scatter contributions");
    SetMem(scatter_f32_, 1, inverse_map_,
           "set exact-Q6 F32 scatter inverse map");
    SetMem(scatter_f32_, 2, output_,
           "set exact-Q6 F32 scatter output");
    const cl_mem gateup_dispatch_coordinates = persistent_dispatch_
        ? native_gateup_task_coordinates_ : dummy_;
    const cl_mem down_dispatch_coordinates = persistent_dispatch_
        ? native_down_task_coordinates_ : dummy_;
    const cl_mem dispatch_metadata = persistent_dispatch_
        ? schedule_metadata_ : dummy_;
    if (layer.exact_q4_gateup) {
      const std::array<cl_mem, 13> q4_gateup_args = {
          layer.gateup_weights, layer.gateup_scale_codes,
          layer.gateup_min_codes, layer.gateup_block_scales,
          layer.gateup_dmins, offsets_, gateup_task_coordinates_, grouped_q8_,
          grouped_scales_, grouped_sum_low_, grouped_sum_high_, swiglu_,
          swiglu_f32_};
      for (cl_uint index = 0; index < q4_gateup_args.size(); ++index) {
        SetMem(q4_exact_gateup_, index, q4_gateup_args[index],
               "set resident exact-block Q4 gate/up arg");
      }
    } else {
      SetNativeCommon(gateup_, grouped_q8_, layer.gateup_weights, swiglu_,
          offsets_, grouped_scales_, grouped_sum_low_, layer.gateup_scales,
          layer.gateup_min_codes, grouped_sum_high_, layer.gateup_dmins,
          swiglu_f32_, gateup_dispatch_coordinates, dispatch_metadata,
          kHiddenSize, kGateUpSize, 8);
    }
    if (IsQ6DownKind(layer.down_kind)) {
      SetMem(q6_combine_sums_, 0, down_sum_low_,
             "set Q6 combine low sums");
      SetMem(q6_combine_sums_, 1, down_sum_high_,
             "set Q6 combine high sums");
      SetMem(q6_combine_sums_, 2, q6_sums_, "set Q6 combined sums");
      SetMem(q6_exact_sums_, 0, down_q8_, "set exact-Q6 sum source");
      SetMem(q6_exact_sums_, 1, q6_sums_, "set exact-Q6 K16 sums");
      const cl_mem q6_contributions = UsesF32Contributions(layer.down_kind)
          ? q6_contributions_f32_ : contributions_;
      if (layer.down_kind == GroupedPrefillDownKind::kQ6U8ExactBlock) {
        const std::array<cl_mem, 10> q6_args = {
            layer.down_weights, layer.down_scales, layer.down_block_scales,
            offsets_, down_task_coordinates_, down_q8_, down_source_scales_,
            q6_sums_, compact_router_, q6_contributions};
        for (cl_uint index = 0; index < q6_args.size(); ++index) {
          SetMem(q6_exact_block_down_, index, q6_args[index],
                 "set resident exact-block Q6 down arg");
        }
      } else {
        const std::array<cl_mem, 9> q6_args = {
            layer.down_weights, layer.down_scales, offsets_,
            down_task_coordinates_, down_q8_, down_source_scales_, q6_sums_,
            compact_router_, q6_contributions};
        cl_kernel q6_kernel =
            layer.down_kind == GroupedPrefillDownKind::kQ6U8ExactPer16
                ? q6_exact_down_ : q6_surrogate_down_;
        for (cl_uint index = 0; index < q6_args.size(); ++index) {
          SetMem(q6_kernel, index, q6_args[index],
                 "set resident Q6 down arg");
        }
      }
    } else if (UsesExactQ4Down(layer.down_kind)) {
      const std::array<cl_mem, 13> q4_down_args = {
          layer.down_weights, layer.down_scale_codes, layer.down_min_codes,
          layer.down_block_scales, layer.down_dmins, offsets_,
          down_task_coordinates_, down_q8_, down_source_scales_, down_sum_low_,
          down_sum_high_, compact_router_, q6_contributions_f32_};
      for (cl_uint index = 0; index < q4_down_args.size(); ++index) {
        SetMem(q4_exact_down_, index, q4_down_args[index],
               "set resident exact-block Q4 down arg");
      }
    } else {
      SetNativeCommon(down_, down_q8_, layer.down_weights,
          UsesF32Contributions(layer.down_kind)
              ? q6_contributions_f32_ : contributions_,
          offsets_, down_source_scales_, down_sum_low_, layer.down_scales,
          layer.down_min_codes, down_sum_high_, layer.down_dmins,
          compact_router_, down_dispatch_coordinates, dispatch_metadata,
          kIntermediateSize, kHiddenSize, 2);
    }
  }

  OpenClState state_;
  std::string device_name_;
  std::string driver_version_;
  bool persistent_dispatch_ = false;
  std::vector<LayerResources> layers_;
  GroupedS8U4PrefillRuntimeStats stats_;
  cl_mem dummy_ = nullptr;
  cl_mem offsets_ = nullptr;
  cl_mem token_map_ = nullptr;
  cl_mem inverse_map_ = nullptr;
  cl_mem compact_router_ = nullptr;
  cl_mem schedule_topk_ = nullptr;
  cl_mem schedule_router_ = nullptr;
  cl_mem schedule_metadata_ = nullptr;
  cl_mem schedule_counts_ = nullptr;
  cl_mem schedule_begins_ = nullptr;
  cl_mem schedule_cursors_ = nullptr;
  cl_mem schedule_gateup_bases_ = nullptr;
  cl_mem schedule_down_bases_ = nullptr;
  cl_mem schedule_native_gateup_bases_ = nullptr;
  cl_mem schedule_native_down_bases_ = nullptr;
  cl_mem schedule_partial_counts_ = nullptr;
  cl_mem schedule_chunk_bases_ = nullptr;
  cl_mem router_logits_ = nullptr;
  void* router_status_ = nullptr;
  cl_mem input_ = nullptr;
  cl_mem token_q8_ = nullptr;
  cl_mem token_scales_ = nullptr;
  cl_mem token_sum_low_ = nullptr;
  cl_mem token_sum_high_ = nullptr;
  cl_mem grouped_q8_ = nullptr;
  cl_mem grouped_scales_ = nullptr;
  cl_mem grouped_sum_low_ = nullptr;
  cl_mem grouped_sum_high_ = nullptr;
  cl_mem swiglu_ = nullptr;
  cl_mem swiglu_f32_ = nullptr;
  cl_mem down_q8_ = nullptr;
  cl_mem down_source_scales_ = nullptr;
  cl_mem down_sum_low_ = nullptr;
  cl_mem down_sum_high_ = nullptr;
  cl_mem q6_sums_ = nullptr;
  cl_mem gateup_task_coordinates_ = nullptr;
  cl_mem down_task_coordinates_ = nullptr;
  cl_mem native_gateup_task_coordinates_ = nullptr;
  cl_mem native_down_task_coordinates_ = nullptr;
  cl_mem contributions_ = nullptr;
  cl_mem q6_contributions_f32_ = nullptr;
  cl_mem output_ = nullptr;
  cl_kernel quantize_tokens_ = nullptr;
  cl_kernel gather_ = nullptr;
  cl_kernel quantize_down_ = nullptr;
  cl_kernel scatter_ = nullptr;
  cl_kernel scatter_f32_ = nullptr;
  cl_kernel schedule_reset_ = nullptr;
  cl_kernel schedule_count_ = nullptr;
  cl_kernel schedule_prefix_ = nullptr;
  cl_kernel schedule_scatter_ = nullptr;
  cl_kernel schedule_tasks_ = nullptr;
  cl_kernel schedule_native_tasks_ = nullptr;
  cl_kernel native_router_ = nullptr;
  cl_kernel router_topk_ = nullptr;
  cl_kernel q4_exact_gateup_ = nullptr;
  cl_kernel q4_exact_down_ = nullptr;
  cl_kernel gateup_ = nullptr;
  cl_kernel down_ = nullptr;
  cl_kernel q6_combine_sums_ = nullptr;
  cl_kernel q6_exact_sums_ = nullptr;
  cl_kernel q6_surrogate_down_ = nullptr;
  cl_kernel q6_exact_down_ = nullptr;
  cl_kernel q6_exact_block_down_ = nullptr;
};

GroupedS8U4PrefillRuntime::GroupedS8U4PrefillRuntime(
    const GroupedS8U4PrefillProgramConfig& config)
    : impl_(std::make_unique<Impl>(config)) {}

GroupedS8U4PrefillRuntime::~GroupedS8U4PrefillRuntime() = default;
GroupedS8U4PrefillRuntime::GroupedS8U4PrefillRuntime(
    GroupedS8U4PrefillRuntime&&) noexcept = default;
GroupedS8U4PrefillRuntime& GroupedS8U4PrefillRuntime::operator=(
    GroupedS8U4PrefillRuntime&&) noexcept = default;

std::uint64_t GroupedS8U4PrefillRuntime::LoadLayer(
    const GroupedS8U4PrefillLayerConfig& config) {
  return impl_->LoadLayer(config);
}

GroupedS8U4PrefillRun GroupedS8U4PrefillRuntime::RunLayer(
    std::uint64_t layer_handle,
    const GroupedS8U4PrefillInput& input) {
  return impl_->RunLayer(layer_handle, input);
}

const std::string& GroupedS8U4PrefillRuntime::device_name() const {
  return impl_->device_name();
}

const std::string& GroupedS8U4PrefillRuntime::driver_version() const {
  return impl_->driver_version();
}

GroupedS8U4PrefillRuntimeStats GroupedS8U4PrefillRuntime::stats() const {
  return impl_->stats();
}

int RunGroupedS8U4Prefill(const GroupedS8U4PrefillConfig& args,
                          std::ostream& json_output,
                          std::ostream& error_output) {
  try {
    ValidateArgs(args);
    const auto topk = ReadBytes(args.topk);
    const auto router_weights = args.schedule_probe_only
        ? std::vector<float>(kAssignments, 0.0f)
        : ReadVector<float>(args.router_weights, kAssignments);
    const auto schedule_begin = std::chrono::steady_clock::now();
    const GroupedSchedule schedule = BuildGroupedSchedule(
        topk, args.topk_stride, router_weights, false, false,
        args.m8_source_preflight);
    const double schedule_prepare_us =
        std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - schedule_begin)
            .count();
    const std::size_t native_global_y =
        ((schedule.max_group + 31) / 32) * 4;
    if (args.schedule_probe_only) {
      json_output << std::boolalpha << std::setprecision(12) << "{"
             << "\"active_experts\":" << schedule.active_experts << ","
             << "\"assignment_count\":" << kAssignments << ","
             << "\"dynamic_router_schedule\":true,"
             << "\"max_group_size\":" << schedule.max_group << ","
             << "\"native_global_y\":" << native_global_y << ","
             << "\"schedule_prepare_us\":" << schedule_prepare_us << "}"
             << std::endl;
      return 0;
    }
    OpenClState state = CreateOpenCl();
    char device_name[256] = {};
    char driver_version[256] = {};
    Check(clGetDeviceInfo(state.device, CL_DEVICE_NAME, sizeof(device_name),
                          device_name, nullptr),
          "clGetDeviceInfo name");
    Check(clGetDeviceInfo(state.device, CL_DRIVER_VERSION,
                          sizeof(driver_version), driver_version, nullptr),
          "clGetDeviceInfo driver");

    constexpr std::size_t gateup_weight_bytes =
        kExpertCount * kGateUpSize * kHiddenSize / 2;
    constexpr std::size_t gateup_scale_bytes =
        kExpertCount * kGateGroups * kGateUpSize * sizeof(float);
    constexpr std::size_t gateup_min_code_bytes =
        kExpertCount * kGateUpSize * kGateBlocks * 8;
    constexpr std::size_t gateup_dmin_bytes =
        kExpertCount * kGateUpSize * kGateBlocks * sizeof(float);
    constexpr std::size_t down_weight_bytes =
        kExpertCount * kHiddenSize * kIntermediateSize / 2;
    constexpr std::size_t down_scale_bytes =
        kExpertCount * kDownGroups * kHiddenSize * sizeof(float);
    constexpr std::size_t down_min_code_bytes =
        kExpertCount * kHiddenSize * kDownBlocks * 8;
    constexpr std::size_t down_dmin_bytes =
        kExpertCount * kHiddenSize * kDownBlocks * sizeof(float);

    cl_mem gateup_weights = CreateCopied(state,
        JoinPath(args.prep_dir, "gateup-weights.bin"), gateup_weight_bytes);
    cl_mem gateup_scales = CreateCopied(state,
        JoinPath(args.prep_dir, "gateup-scales.bin"), gateup_scale_bytes);
    cl_mem gateup_min_codes = CreateCopied(state,
        JoinPath(args.prep_dir, "gateup-min-codes.bin"),
        gateup_min_code_bytes);
    cl_mem gateup_dmins = CreateCopied(state,
        JoinPath(args.prep_dir, "gateup-dmins.bin"), gateup_dmin_bytes);
    cl_mem down_weights = CreateCopied(state,
        JoinPath(args.prep_dir, "down-weights.bin"), down_weight_bytes);
    cl_mem down_scales = CreateCopied(state,
        JoinPath(args.prep_dir, "down-scales.bin"), down_scale_bytes);
    cl_mem down_min_codes = CreateCopied(state,
        JoinPath(args.prep_dir, "down-min-codes.bin"), down_min_code_bytes);
    cl_mem down_dmins = CreateCopied(state,
        JoinPath(args.prep_dir, "down-dmins.bin"), down_dmin_bytes);

    const auto schedule_upload_begin = std::chrono::steady_clock::now();
    cl_mem offsets = CreateCopiedVector(state, schedule.offsets);
    cl_mem token_map = CreateCopiedVector(state, schedule.token_map);
    cl_mem inverse_map = CreateCopiedVector(state, schedule.inverse_map);
    cl_mem compact_router = CreateCopiedVector(
        state, schedule.compact_router_weights);
    const double schedule_upload_us =
        std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - schedule_upload_begin)
            .count();
    const double schedule_setup_us = schedule_prepare_us + schedule_upload_us;
    const std::uint32_t zero = 0;
    cl_mem dummy = CreateCopiedVector(state, std::vector<std::uint32_t>{zero});
    cl_mem gateup_dispatch = dummy;
    cl_mem down_dispatch = dummy;
    cl_mem dispatch_metadata = dummy;
    if (args.m8_source_preflight) {
      Require(schedule.active_experts == 222 && schedule.max_group == 361,
              "fixed M8 preflight requires the locked layer-27 histogram");
      Require(schedule.native_m8_gateup_task_coordinates.size() == 36480 &&
                  schedule.native_m8_down_task_coordinates.size() == 36480,
              "fixed M8 preflight logical task count changed");
      gateup_dispatch = CreateCopiedVector(
          state, schedule.native_m8_gateup_task_coordinates);
      down_dispatch = CreateCopiedVector(
          state, schedule.native_m8_down_task_coordinates);
      std::vector<std::uint32_t> metadata(kScheduleMetadataWords, 0U);
      metadata[5] = static_cast<std::uint32_t>(
          schedule.native_m8_gateup_task_coordinates.size());
      metadata[6] = static_cast<std::uint32_t>(
          schedule.native_m8_down_task_coordinates.size());
      dispatch_metadata = CreateCopiedVector(state, metadata);
    }

    const auto input = ReadVector<float>(
        args.input, kTokenCount * kHiddenSize);
    const auto swiglu_oracle = ReadVector<float>(
        args.oracle, kAssignments * kIntermediateSize);
    const auto down_oracle = ReadVector<float>(
        args.down_oracle, kAssignments * kHiddenSize);
    const auto moe_oracle = ReadVector<float>(
        args.moe_oracle, kTokenCount * kHiddenSize);
    cl_mem input_memory = CreateCopiedVector(state, input);

    cl_mem token_q8 = CreateEmpty(state, kTokenCount * kHiddenSize);
    cl_mem token_scales = CreateEmpty(state, kTokenCount * 8 * sizeof(float));
    cl_mem token_sum_low = CreateEmpty(state, kTokenCount * 64);
    cl_mem token_sum_high = CreateEmpty(state, kTokenCount * 64);
    cl_mem grouped_q8 = CreateEmpty(state, kAssignments * kHiddenSize);
    cl_mem grouped_scales = CreateEmpty(
        state, kAssignments * 8 * sizeof(float));
    cl_mem grouped_sum_low = CreateEmpty(state, kAssignments * 64);
    cl_mem grouped_sum_high = CreateEmpty(state, kAssignments * 64);
    cl_mem swiglu = CreateEmpty(
        state, kAssignments * kIntermediateSize * sizeof(std::uint16_t));
    cl_mem swiglu_f32 = CreateEmpty(
        state, kAssignments * kIntermediateSize * sizeof(float));
    cl_mem down_q8 = CreateEmpty(state, kAssignments * kIntermediateSize);
    cl_mem down_source_scales = CreateEmpty(
        state, kAssignments * 2 * sizeof(float));
    cl_mem down_sum_low = CreateEmpty(state, kAssignments * 16);
    cl_mem down_sum_high = CreateEmpty(state, kAssignments * 16);
    cl_mem contributions = CreateEmpty(
        state, kAssignments * kHiddenSize * sizeof(std::uint16_t));
    cl_mem output = CreateEmpty(
        state, kTokenCount * kHiddenSize * sizeof(float));

    cl_program support_program = BuildSourceProgram(state, args.kernels);
    cl_program gateup_program = LoadBinaryProgram(state, args.gateup_binary);
    cl_program down_program = LoadBinaryProgram(state, args.down_binary);
    cl_kernel quantize_tokens = CreateKernel(
        state, support_program, "iq36_quantize_tokens_q8");
    cl_kernel gather = CreateKernel(
        state, support_program, "iq36_gather_quantized_q8");
    cl_kernel quantize_down = CreateKernel(
        state, support_program, "iq36_quantize_swiglu_f32_q8");
    cl_kernel scatter = CreateKernel(
        state, support_program, "iq36_scatter_f16_contributions");
    cl_kernel gateup = CreateKernel(state, gateup_program, "grouped_micro_gemm");
    cl_kernel down = CreateKernel(state, down_program, "grouped_micro_gemm");

    const cl_uint token_rows = kTokenCount;
    const std::array<cl_mem, 5> quantize_args = {input_memory, token_q8,
        token_scales, token_sum_low, token_sum_high};
    for (cl_uint index = 0; index < quantize_args.size(); ++index) {
      SetMem(quantize_tokens, index, quantize_args[index],
             "set token quantize arg");
    }
    Check(clSetKernelArg(
              quantize_tokens, 5, sizeof(token_rows), &token_rows),
          "set token row count");
    const cl_uint assignment_rows = kAssignments;
    const std::array<cl_mem, 9> gather_args = {token_map, token_q8,
        token_scales, token_sum_low, token_sum_high, grouped_q8,
        grouped_scales, grouped_sum_low, grouped_sum_high};
    for (cl_uint index = 0; index < gather_args.size(); ++index) {
      SetMem(gather, index, gather_args[index], "set grouped gather arg");
    }
    Check(clSetKernelArg(gather, 9, sizeof(assignment_rows), &assignment_rows),
          "set grouped row count");
    const std::array<cl_mem, 5> down_quantize_args = {swiglu_f32, down_q8,
        down_source_scales, down_sum_low, down_sum_high};
    for (cl_uint index = 0; index < down_quantize_args.size(); ++index) {
      SetMem(quantize_down, index, down_quantize_args[index],
             "set down quantize arg");
    }
    Check(clSetKernelArg(
              quantize_down, 5, sizeof(assignment_rows), &assignment_rows),
          "set down row count");
    SetMem(scatter, 0, contributions, "set scatter contributions");
    SetMem(scatter, 1, inverse_map, "set scatter inverse map");
    SetMem(scatter, 2, output, "set scatter output");

    SetNativeCommon(gateup, grouped_q8, gateup_weights, swiglu, offsets,
        grouped_scales, grouped_sum_low, gateup_scales, gateup_min_codes,
        grouped_sum_high, gateup_dmins, swiglu_f32, gateup_dispatch,
        dispatch_metadata,
        kHiddenSize, kGateUpSize, 8);
    SetNativeCommon(down, down_q8, down_weights, contributions, offsets,
        down_source_scales, down_sum_low, down_scales, down_min_codes,
        down_sum_high, down_dmins, compact_router, down_dispatch,
        dispatch_metadata,
        kIntermediateSize, kHiddenSize, 2);

    constexpr std::size_t local = 256;
    const std::size_t token_quant_global = kTokenCount * 8 * local;
    const std::size_t gather_global = kAssignments * kHiddenSize;
    const std::size_t down_quant_global = kAssignments * 2 * local;
    const std::size_t scatter_global = kTokenCount * kHiddenSize;
    constexpr std::array<std::size_t, 3> native_local = {32, 4, 1};
    constexpr std::array<std::size_t, 3> m8_gateup_local = {16, 8, 1};
    constexpr std::array<std::size_t, 3> m8_down_local = {32, 4, 1};
    const std::array<std::size_t, 3> gateup_global = {
        512, native_global_y, kExpertCount};
    const std::array<std::size_t, 3> down_global = {
        1024, native_global_y, kExpertCount};
    constexpr std::array<std::size_t, 3> m8_gateup_global = {
        16, 8, kPersistentWorkgroupCount};
    constexpr std::array<std::size_t, 3> m8_down_global = {
        32, 4, kPersistentWorkgroupCount};

    const auto EnqueueGather = [&]() {
      Check(clEnqueueNDRangeKernel(state.queue, quantize_tokens, 1, nullptr,
                                   &token_quant_global, &local, 0, nullptr,
                                   nullptr),
            "enqueue token quantize");
      Check(clEnqueueNDRangeKernel(state.queue, gather, 1, nullptr,
                                   &gather_global, &local, 0, nullptr, nullptr),
            "enqueue grouped gather");
    };
    const auto EnqueueGateUp = [&]() {
      const auto& global = args.m8_source_preflight
          ? m8_gateup_global : gateup_global;
      const auto& local_size = args.m8_source_preflight
          ? m8_gateup_local : native_local;
      Check(clEnqueueNDRangeKernel(state.queue, gateup, 3, nullptr,
                                   global.data(), local_size.data(), 0,
                                   nullptr, nullptr),
            "enqueue native gate/up");
    };
    const auto EnqueueDownQuantize = [&]() {
      Check(clEnqueueNDRangeKernel(state.queue, quantize_down, 1, nullptr,
                                   &down_quant_global, &local, 0, nullptr,
                                   nullptr),
            "enqueue down quantize");
    };
    const auto EnqueueDown = [&]() {
      const auto& global = args.m8_source_preflight
          ? m8_down_global : down_global;
      const auto& local_size = args.m8_source_preflight
          ? m8_down_local : native_local;
      Check(clEnqueueNDRangeKernel(state.queue, down, 3, nullptr,
                                   global.data(), local_size.data(), 0,
                                   nullptr, nullptr),
            "enqueue native down");
    };
    const auto EnqueueScatter = [&]() {
      Check(clEnqueueNDRangeKernel(state.queue, scatter, 1, nullptr,
                                   &scatter_global, &local, 0, nullptr,
                                   nullptr),
            "enqueue scatter");
    };
    const auto Execute = [&]() {
      EnqueueGather();
      EnqueueGateUp();
      EnqueueDownQuantize();
      EnqueueDown();
      EnqueueScatter();
      Check(clFinish(state.queue), "clFinish component");
    };
    const auto Timed = [&](const auto& enqueue) {
      const auto begin = std::chrono::steady_clock::now();
      enqueue();
      Check(clFinish(state.queue), "clFinish stage");
      return std::chrono::duration<double, std::micro>(
                 std::chrono::steady_clock::now() - begin)
          .count();
    };

    for (int iteration = 0; iteration < args.warmup; ++iteration) Execute();
    const std::array<double, 5> stage_us = {Timed(EnqueueGather),
        Timed(EnqueueGateUp), Timed(EnqueueDownQuantize), Timed(EnqueueDown),
        Timed(EnqueueScatter)};
    std::vector<double> matrix_samples_us;
    if (args.m8_source_preflight) {
      matrix_samples_us.reserve(args.repeat);
      for (int iteration = 0; iteration < args.repeat; ++iteration) {
        const double gateup_us = Timed(EnqueueGateUp);
        Timed(EnqueueDownQuantize);
        const double down_us = Timed(EnqueueDown);
        matrix_samples_us.push_back(gateup_us + down_us);
      }
    }
    std::vector<double> samples;
    for (int iteration = 0; iteration < args.repeat; ++iteration) {
      const auto begin = std::chrono::steady_clock::now();
      Execute();
      samples.push_back(std::chrono::duration<double, std::micro>(
                            std::chrono::steady_clock::now() - begin)
                            .count());
    }

    std::vector<std::uint16_t> swiglu_half(
        kAssignments * kIntermediateSize);
    std::vector<std::uint16_t> contribution_half(
        kAssignments * kHiddenSize);
    std::vector<float> routed(kTokenCount * kHiddenSize);
    Check(clEnqueueReadBuffer(state.queue, swiglu, CL_TRUE, 0,
                              swiglu_half.size() * sizeof(std::uint16_t),
                              swiglu_half.data(), 0, nullptr, nullptr),
          "read swiglu");
    Check(clEnqueueReadBuffer(state.queue, contributions, CL_TRUE, 0,
                              contribution_half.size() * sizeof(std::uint16_t),
                              contribution_half.data(), 0, nullptr, nullptr),
          "read contributions");
    Check(clEnqueueReadBuffer(state.queue, output, CL_TRUE, 0,
                              routed.size() * sizeof(float), routed.data(), 0,
                              nullptr, nullptr),
          "read routed output");

    std::vector<float> observed_swiglu(swiglu_oracle.size());
    std::vector<float> observed_weighted(down_oracle.size());
    std::vector<float> expected_weighted(down_oracle.size());
    for (std::size_t source = 0; source < kAssignments; ++source) {
      const std::size_t row =
          static_cast<std::size_t>(schedule.inverse_map[source]);
      for (std::size_t inner = 0; inner < kIntermediateSize; ++inner) {
        observed_swiglu[source * kIntermediateSize + inner] = HalfToFloat(
            swiglu_half[row * kIntermediateSize + inner]);
      }
      for (std::size_t hidden = 0; hidden < kHiddenSize; ++hidden) {
        observed_weighted[source * kHiddenSize + hidden] = HalfToFloat(
            contribution_half[row * kHiddenSize + hidden]);
        expected_weighted[source * kHiddenSize + hidden] =
            down_oracle[source * kHiddenSize + hidden] * router_weights[source];
      }
    }
    const CompareStats swiglu_compare =
        Compare(observed_swiglu, swiglu_oracle);
    const CompareStats weighted_compare =
        Compare(observed_weighted, expected_weighted);
    const CompareStats routed_compare = Compare(routed, moe_oracle);
    const bool correctness_pass = ComponentAccuracyPass(swiglu_compare) &&
        ComponentAccuracyPass(weighted_compare) &&
        ComponentAccuracyPass(routed_compare);
    std::vector<double> sorted = samples;
    std::sort(sorted.begin(), sorted.end());
    const double minimum = sorted.front();
    const double median = sorted[sorted.size() / 2];
    const double mean =
        std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
    const double complete_minimum_us = minimum + schedule_setup_us;
    constexpr std::uint64_t m8_padded_matrix_macs = 28689039360ULL;
    std::vector<double> sorted_matrix_samples = matrix_samples_us;
    std::sort(sorted_matrix_samples.begin(), sorted_matrix_samples.end());
    const double matrix_minimum_us = sorted_matrix_samples.empty()
        ? 0.0 : sorted_matrix_samples.front();
    const double matrix_median_us = sorted_matrix_samples.empty()
        ? 0.0
        : sorted_matrix_samples[sorted_matrix_samples.size() / 2];
    const double matrix_rate_tmac_s = matrix_minimum_us == 0.0
        ? 0.0
        : static_cast<double>(m8_padded_matrix_macs) /
            matrix_minimum_us / 1.0e6;
    const bool performance_pass = args.m8_source_preflight
        ? matrix_minimum_us <= args.kernel_cap_us &&
            matrix_rate_tmac_s >= 5.4
        : complete_minimum_us <= args.kernel_cap_us;
    const bool maps_native_only = MapsAreNativeOnly();
    const std::uint64_t resident_bytes = gateup_weight_bytes +
        gateup_scale_bytes + gateup_min_code_bytes + gateup_dmin_bytes +
        down_weight_bytes + down_scale_bytes + down_min_code_bytes +
        down_dmin_bytes;

    json_output << std::boolalpha << std::setprecision(12) << "{";
    json_output << "\"active_experts\":" << schedule.active_experts << ",";
    json_output << "\"assignment_count\":" << kAssignments << ",";
    PrintCompare(json_output, "compare", swiglu_compare);
    json_output << "\"complete_minimum_us\":" << complete_minimum_us << ",";
    json_output << "\"correctness_pass\":" << correctness_pass << ",";
    json_output << "\"device_name\":\"" << device_name << "\",";
    json_output << "\"driver_version\":\"" << driver_version << "\",";
    json_output << "\"dynamic_router_schedule\":true,";
    json_output << "\"kernel_cap_us\":" << args.kernel_cap_us << ",";
    json_output << "\"maps_native_only\":" << maps_native_only << ",";
    json_output << "\"max_group_size\":" << schedule.max_group << ",";
    if (args.m8_source_preflight) {
      json_output << "\"m8_padded_rows\":9120,";
      json_output << "\"matrix_median_us\":" << matrix_median_us << ",";
      json_output << "\"matrix_minimum_us\":" << matrix_minimum_us << ",";
      json_output << "\"matrix_padded_macs\":"
                  << m8_padded_matrix_macs << ",";
      json_output << "\"matrix_rate_tmac_s\":" << matrix_rate_tmac_s
                  << ",";
      json_output << "\"matrix_samples_us\":[";
      for (std::size_t index = 0; index < matrix_samples_us.size(); ++index) {
        if (index != 0) json_output << ',';
        json_output << matrix_samples_us[index];
      }
      json_output << "],";
      json_output << "\"persistent_workgroups\":96,";
      json_output << "\"preflight_logical_tasks\":"
                  << schedule.native_m8_gateup_task_coordinates.size()
                  << ",";
    }
    json_output << "\"mean_us\":" << mean << ",";
    json_output << "\"median_us\":" << median << ",";
    json_output << "\"minimum_us\":" << minimum << ",";
    json_output << "\"mode\":\""
                << (args.m8_source_preflight
                        ? "fixed_m8_persistent_source_preflight"
                        : "parameterized_grouped_s8_u4_f32_swiglu_"
                          "f16_contribution")
                << "\",";
    PrintCompare(json_output, "moe_compare", routed_compare);
    json_output << "\"performance_pass\":" << performance_pass << ",";
    json_output << "\"resident_weight_bytes\":" << resident_bytes << ",";
    json_output << "\"schedule_prepare_us\":" << schedule_prepare_us << ",";
    json_output << "\"schedule_setup_us\":" << schedule_setup_us << ",";
    json_output << "\"schedule_upload_us\":" << schedule_upload_us << ",";
    json_output << "\"samples_us\":[";
    for (std::size_t index = 0; index < samples.size(); ++index) {
      if (index != 0) json_output << ',';
      json_output << samples[index];
    }
    json_output << "],\"stage_us\":{";
    json_output << "\"gather\":" << stage_us[0] << ",";
    json_output << "\"gateup\":" << stage_us[1] << ",";
    json_output << "\"down_quantize\":" << stage_us[2] << ",";
    json_output << "\"down\":" << stage_us[3] << ",";
    json_output << "\"scatter\":" << stage_us[4] << "},";
    PrintCompare(json_output, "weighted_down_compare", weighted_compare);
    json_output << "\"workgroup\":"
                << (args.m8_source_preflight
                        ? "{\"gateup\":[16,8,1],\"down\":[32,4,1]}"
                        : "[32,4,1]")
                << "}" << std::endl;
    return correctness_pass && performance_pass && maps_native_only ? 0 : 2;
  } catch (const std::exception& exception) {
    error_output << exception.what() << '\n';
    return 4;
  }
}

int RunGroupedS8U4PrefillCommandLine(int argc,
                                     char** argv,
                                     std::ostream& output,
                                     std::ostream& error_output) {
  try {
    return RunGroupedS8U4Prefill(
        ParseArgs(argc, argv), output, error_output);
  } catch (const std::exception& exception) {
    error_output << exception.what() << '\n';
    return 4;
  }
}

}  // namespace iq36
