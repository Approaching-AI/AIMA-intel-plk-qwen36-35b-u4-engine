#include <CL/cl.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr std::size_t kTokens = 1024;
constexpr std::size_t kTopK = 8;
constexpr std::size_t kRoutedRows = kTokens * kTopK;
constexpr std::size_t kCompleteRows = kRoutedRows + kTokens;
constexpr std::size_t kRoutedExperts = 256;
constexpr std::size_t kCombinedExperts = 257;
constexpr std::size_t kHidden = 2048;
constexpr std::size_t kIntermediate = 512;

[[noreturn]] void Fail(const std::string& message) {
  throw std::runtime_error(message);
}

void Require(bool condition, const std::string& message) {
  if (!condition) Fail(message);
}

void Check(cl_int status, const std::string& operation) {
  if (status != CL_SUCCESS) {
    Fail(operation + " failed with OpenCL status " + std::to_string(status));
  }
}

struct Args {
  std::string prep_dir;
  std::string gate_binary;
  std::string up_binary;
  std::string down_binary;
  std::string support_source;
  std::string input;
  std::string topk;
  std::string router_weights;
  std::string oracle;
  std::string swiglu_oracle;
  std::string weighted_down_oracle;
  std::string routed_oracle;
  std::string debug_dir;
  std::size_t topk_stride = 1024;
  int warmup = 5;
  int repeat = 11;
  double cap_us = 6250.0;
};

Args ParseArgs(int argc, char** argv) {
  Args args;
  for (int index = 1; index < argc; ++index) {
    const std::string option = argv[index];
    auto value = [&]() {
      if (++index >= argc) Fail(option + " requires a value");
      return std::string(argv[index]);
    };
    if (option == "--prep-dir") args.prep_dir = value();
    else if (option == "--gate-binary") args.gate_binary = value();
    else if (option == "--up-binary") args.up_binary = value();
    else if (option == "--down-binary") args.down_binary = value();
    else if (option == "--support-source") args.support_source = value();
    else if (option == "--input") args.input = value();
    else if (option == "--topk") args.topk = value();
    else if (option == "--router-weights") args.router_weights = value();
    else if (option == "--oracle") args.oracle = value();
    else if (option == "--swiglu-oracle") args.swiglu_oracle = value();
    else if (option == "--weighted-down-oracle") {
      args.weighted_down_oracle = value();
    } else if (option == "--routed-oracle") {
      args.routed_oracle = value();
    } else if (option == "--debug-dir") {
      args.debug_dir = value();
    } else if (option == "--topk-stride") {
      args.topk_stride = std::stoull(value());
    } else if (option == "--warmup") {
      args.warmup = std::stoi(value());
    } else if (option == "--repeat") {
      args.repeat = std::stoi(value());
    } else if (option == "--cap-us") {
      args.cap_us = std::stod(value());
    } else {
      Fail("unknown option: " + option);
    }
  }
  Require(!args.prep_dir.empty() && !args.gate_binary.empty() &&
              !args.up_binary.empty() && !args.down_binary.empty() &&
              !args.support_source.empty() && !args.input.empty() &&
              !args.topk.empty() && !args.router_weights.empty() &&
              !args.oracle.empty(),
          "all runtime paths are required");
  Require(args.topk_stride >= kTopK * sizeof(std::int32_t),
          "top-k stride is too small");
  Require(args.warmup > 0 && args.repeat >= 5 && args.cap_us > 0.0,
          "warmup/repeat/cap arguments are invalid");
  return args;
}

std::vector<std::uint8_t> ReadBytes(
    const std::string& path, std::size_t expected = 0) {
  std::ifstream input(path, std::ios::binary);
  Require(static_cast<bool>(input), "could not open " + path);
  input.seekg(0, std::ios::end);
  const auto size = input.tellg();
  Require(size >= 0, "could not size " + path);
  if (expected != 0) {
    Require(static_cast<std::size_t>(size) == expected,
            "byte-size mismatch for " + path);
  }
  input.seekg(0, std::ios::beg);
  std::vector<std::uint8_t> data(static_cast<std::size_t>(size));
  input.read(reinterpret_cast<char*>(data.data()),
             static_cast<std::streamsize>(data.size()));
  Require(static_cast<bool>(input), "could not read " + path);
  return data;
}

template <typename Value>
std::vector<Value> ReadVector(const std::string& path, std::size_t count) {
  const auto bytes = ReadBytes(path, count * sizeof(Value));
  std::vector<Value> values(count);
  std::memcpy(values.data(), bytes.data(), bytes.size());
  return values;
}

template <typename Value>
void WriteVector(const std::string& path, const std::vector<Value>& values) {
  std::ofstream output(path, std::ios::binary);
  Require(static_cast<bool>(output), "could not open debug output " + path);
  output.write(reinterpret_cast<const char*>(values.data()),
               static_cast<std::streamsize>(values.size() * sizeof(Value)));
  Require(static_cast<bool>(output), "could not write debug output " + path);
}

std::string ReadText(const std::string& path) {
  const auto bytes = ReadBytes(path);
  return std::string(bytes.begin(), bytes.end());
}

struct Schedule {
  std::vector<std::int32_t> active_experts;
  std::vector<std::int32_t> starts;
  std::vector<std::int32_t> lengths;
  std::vector<std::int32_t> token_map;
  std::vector<std::int32_t> inverse_map;
  std::vector<std::int32_t> row_expert;
  std::vector<float> compact_router_weights;
  std::size_t routed_active_experts = 0;
  std::size_t max_routed_group = 0;
};

Schedule BuildSchedule(const std::vector<std::uint8_t>& topk,
                       std::size_t stride,
                       const std::vector<float>& router_weights) {
  Require(topk.size() >= (kTokens - 1) * stride +
              kTopK * sizeof(std::int32_t),
          "top-k tensor is truncated");
  Require(router_weights.size() == kRoutedRows,
          "router weight count mismatch");
  std::array<std::vector<std::pair<std::int32_t, std::int32_t>>,
             kRoutedExperts> assignments;
  for (std::size_t token = 0; token < kTokens; ++token) {
    std::array<std::int32_t, kTopK> selected{};
    for (std::size_t rank = 0; rank < kTopK; ++rank) {
      std::memcpy(&selected[rank],
                  topk.data() + token * stride + rank * sizeof(std::int32_t),
                  sizeof(std::int32_t));
      Require(selected[rank] >= 0 &&
                  selected[rank] < static_cast<std::int32_t>(kRoutedExperts),
              "top-k expert is out of range");
      for (std::size_t prior = 0; prior < rank; ++prior) {
        Require(selected[rank] != selected[prior],
                "top-k expert is duplicated");
      }
      assignments[static_cast<std::size_t>(selected[rank])].push_back(
          {static_cast<std::int32_t>(token),
           static_cast<std::int32_t>(rank)});
    }
  }

  Schedule schedule;
  schedule.token_map.reserve(kCompleteRows);
  schedule.inverse_map.assign(kRoutedRows, -1);
  schedule.row_expert.reserve(kCompleteRows);
  schedule.compact_router_weights.reserve(kRoutedRows);
  for (std::size_t expert = 0; expert < kRoutedExperts; ++expert) {
    const auto& rows = assignments[expert];
    if (rows.empty()) continue;
    schedule.active_experts.push_back(static_cast<std::int32_t>(expert));
    schedule.starts.push_back(
        static_cast<std::int32_t>(schedule.token_map.size()));
    schedule.lengths.push_back(static_cast<std::int32_t>(rows.size()));
    schedule.max_routed_group = std::max(
        schedule.max_routed_group, rows.size());
    for (const auto& [token, rank] : rows) {
      const std::int32_t compact =
          static_cast<std::int32_t>(schedule.token_map.size());
      schedule.token_map.push_back(token);
      schedule.inverse_map[static_cast<std::size_t>(token) * kTopK +
                           static_cast<std::size_t>(rank)] = compact;
      schedule.row_expert.push_back(static_cast<std::int32_t>(expert));
      schedule.compact_router_weights.push_back(
          router_weights[static_cast<std::size_t>(token) * kTopK +
                         static_cast<std::size_t>(rank)]);
    }
  }
  schedule.routed_active_experts = schedule.active_experts.size();
  Require(schedule.token_map.size() == kRoutedRows,
          "routed assignment count mismatch");
  Require(schedule.routed_active_experts == 222 &&
              schedule.max_routed_group == 361,
          "locked layer-27 routing histogram changed");
  Require(std::none_of(schedule.inverse_map.begin(), schedule.inverse_map.end(),
                       [](std::int32_t value) { return value < 0; }),
          "inverse route map is incomplete");

  schedule.active_experts.push_back(
      static_cast<std::int32_t>(kRoutedExperts));
  schedule.starts.push_back(static_cast<std::int32_t>(kRoutedRows));
  schedule.lengths.push_back(static_cast<std::int32_t>(kTokens));
  for (std::size_t token = 0; token < kTokens; ++token) {
    schedule.token_map.push_back(static_cast<std::int32_t>(token));
    schedule.row_expert.push_back(
        static_cast<std::int32_t>(kRoutedExperts));
  }
  Require(schedule.token_map.size() == kCompleteRows &&
              schedule.row_expert.size() == kCompleteRows &&
              schedule.active_experts.size() == 223,
          "complete FFN schedule shape mismatch");
  return schedule;
}

std::string ProgramLog(cl_program program, cl_device_id device) {
  std::size_t size = 0;
  Check(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                              0, nullptr, &size),
        "clGetProgramBuildInfo size");
  std::string log(size, '\0');
  if (size != 0) {
    Check(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                                size, log.data(), nullptr),
          "clGetProgramBuildInfo log");
  }
  return log;
}

struct OpenClState {
  cl_device_id device = nullptr;
  cl_context context = nullptr;
  cl_command_queue queue = nullptr;
  std::vector<cl_mem> memories;
  std::vector<cl_program> programs;
  std::vector<cl_kernel> kernels;

  OpenClState() = default;
  OpenClState(const OpenClState&) = delete;
  OpenClState& operator=(const OpenClState&) = delete;
  OpenClState(OpenClState&& other) noexcept
      : device(other.device), context(other.context), queue(other.queue),
        memories(std::move(other.memories)),
        programs(std::move(other.programs)), kernels(std::move(other.kernels)) {
    other.device = nullptr;
    other.context = nullptr;
    other.queue = nullptr;
    other.memories.clear();
    other.programs.clear();
    other.kernels.clear();
  }

  ~OpenClState() {
    for (cl_kernel kernel : kernels) clReleaseKernel(kernel);
    for (cl_program program : programs) clReleaseProgram(program);
    for (cl_mem memory : memories) clReleaseMemObject(memory);
    if (queue != nullptr) clReleaseCommandQueue(queue);
    if (context != nullptr) clReleaseContext(context);
  }
};

OpenClState CreateOpenCl() {
  OpenClState state;
  cl_uint platform_count = 0;
  Check(clGetPlatformIDs(0, nullptr, &platform_count),
        "clGetPlatformIDs count");
  std::vector<cl_platform_id> platforms(platform_count);
  Check(clGetPlatformIDs(platform_count, platforms.data(), nullptr),
        "clGetPlatformIDs");
  for (cl_platform_id platform : platforms) {
    cl_uint count = 0;
    if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &count) !=
        CL_SUCCESS) {
      continue;
    }
    std::vector<cl_device_id> devices(count);
    Check(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, count,
                         devices.data(), nullptr),
          "clGetDeviceIDs");
    for (cl_device_id candidate : devices) {
      char name[256] = {};
      Check(clGetDeviceInfo(candidate, CL_DEVICE_NAME, sizeof(name),
                            name, nullptr),
            "clGetDeviceInfo name");
      if (std::string(name).find("B390") != std::string::npos) {
        state.device = candidate;
        break;
      }
    }
    if (state.device != nullptr) break;
  }
  Require(state.device != nullptr, "Intel Arc B390 OpenCL device not found");
  cl_int status = CL_SUCCESS;
  state.context = clCreateContext(
      nullptr, 1, &state.device, nullptr, nullptr, &status);
  Check(status, "clCreateContext");
  const cl_queue_properties properties[] = {
      CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
  state.queue = clCreateCommandQueueWithProperties(
      state.context, state.device, properties, &status);
  Check(status, "clCreateCommandQueueWithProperties");
  return state;
}

cl_mem CreateCopied(OpenClState& state, const void* data,
                    std::size_t bytes) {
  cl_int status = CL_SUCCESS;
  cl_mem memory = clCreateBuffer(
      state.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      bytes, const_cast<void*>(data), &status);
  Check(status, "clCreateBuffer copied");
  state.memories.push_back(memory);
  return memory;
}

cl_mem CreateCopiedFile(OpenClState& state, const std::string& path,
                        std::size_t bytes) {
  const auto data = ReadBytes(path, bytes);
  return CreateCopied(state, data.data(), data.size());
}

template <typename Value>
cl_mem CreateCopiedVector(OpenClState& state,
                          const std::vector<Value>& values) {
  return CreateCopied(
      state, values.data(), values.size() * sizeof(Value));
}

cl_mem CreateEmpty(OpenClState& state, std::size_t bytes) {
  cl_int status = CL_SUCCESS;
  cl_mem memory = clCreateBuffer(
      state.context, CL_MEM_READ_WRITE, bytes, nullptr, &status);
  Check(status, "clCreateBuffer empty");
  state.memories.push_back(memory);
  return memory;
}

cl_program LoadBinary(OpenClState& state, const std::string& path) {
  const auto binary = ReadBytes(path);
  const std::size_t size = binary.size();
  const unsigned char* data = binary.data();
  cl_int binary_status = CL_SUCCESS;
  cl_int status = CL_SUCCESS;
  cl_program program = clCreateProgramWithBinary(
      state.context, 1, &state.device, &size, &data,
      &binary_status, &status);
  Check(status, "clCreateProgramWithBinary");
  Check(binary_status, "binary status");
  status = clBuildProgram(program, 1, &state.device, "", nullptr, nullptr);
  if (status != CL_SUCCESS) {
    const std::string log = ProgramLog(program, state.device);
    clReleaseProgram(program);
    Fail("binary program build failed: " + log);
  }
  state.programs.push_back(program);
  return program;
}

cl_program BuildSource(OpenClState& state, const std::string& path) {
  const std::string source = ReadText(path);
  const char* data = source.data();
  const std::size_t size = source.size();
  cl_int status = CL_SUCCESS;
  cl_program program = clCreateProgramWithSource(
      state.context, 1, &data, &size, &status);
  Check(status, "clCreateProgramWithSource support");
  status = clBuildProgram(
      program, 1, &state.device,
      "-cl-std=CL2.0 -cl-fp32-correctly-rounded-divide-sqrt",
      nullptr, nullptr);
  if (status != CL_SUCCESS) {
    const std::string log = ProgramLog(program, state.device);
    clReleaseProgram(program);
    Fail("support source build failed: " + log);
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

void SetMem(cl_kernel kernel, cl_uint index, cl_mem value,
            const char* label) {
  Check(clSetKernelArg(kernel, index, sizeof(value), &value), label);
}

void SetI32(cl_kernel kernel, cl_uint index, std::int32_t value,
            const char* label) {
  Check(clSetKernelArg(kernel, index, sizeof(value), &value), label);
}

void SetMicroArgs(cl_kernel kernel, cl_mem input, cl_mem weights,
                  cl_mem output, cl_mem active_experts, cl_mem starts,
                  cl_mem lengths, std::int32_t m, std::int32_t k,
                  cl_mem scales, cl_mem zps) {
  SetMem(kernel, 0, input, "set micro input");
  SetMem(kernel, 1, weights, "set micro weights");
  SetMem(kernel, 2, output, "set micro output");
  SetMem(kernel, 3, active_experts, "set micro active experts");
  SetMem(kernel, 4, starts, "set micro starts");
  SetMem(kernel, 5, lengths, "set micro lengths");
  SetI32(kernel, 6, m, "set micro m");
  SetI32(kernel, 7, k, "set micro k");
  SetMem(kernel, 8, scales, "set micro scales");
  SetMem(kernel, 9, zps, "set micro zero points");
}

struct Kernels {
  cl_kernel gather = nullptr;
  cl_kernel up = nullptr;
  cl_kernel gate = nullptr;
  cl_kernel swiglu = nullptr;
  cl_kernel down = nullptr;
  cl_kernel down_residual = nullptr;
  cl_kernel scalar_gate = nullptr;
  cl_kernel final = nullptr;
};

struct Events {
  std::vector<std::string> names;
  std::vector<cl_event> values;

  ~Events() {
    for (cl_event value : values) clReleaseEvent(value);
  }
};

void Enqueue(OpenClState& state, Events& events, const char* name,
             cl_kernel kernel, const std::array<std::size_t, 3>& global,
             const std::array<std::size_t, 3>& local) {
  cl_event event = nullptr;
  Check(clEnqueueNDRangeKernel(
      state.queue, kernel, 3, nullptr, global.data(), local.data(),
      0, nullptr, &event), std::string("enqueue ") + name);
  events.names.emplace_back(name);
  events.values.push_back(event);
}

struct Sample {
  double device_span_us = 0.0;
  double wall_us = 0.0;
  std::vector<double> stage_us;
};

Sample RunOnce(OpenClState& state, const Kernels& kernels,
               std::size_t active_experts) {
  Events events;
  const auto wall_start = std::chrono::steady_clock::now();
  Enqueue(state, events, "gather", kernels.gather,
          {kCompleteRows * 8 * 256, 1, 1}, {256, 1, 1});
  Enqueue(state, events, "up", kernels.up,
          {256, 24, active_experts}, {128, 4, 1});
  Enqueue(state, events, "gate", kernels.gate,
          {256, 24, active_experts}, {128, 4, 1});
  Enqueue(state, events, "swiglu_residual", kernels.swiglu,
          {kCompleteRows * 2 * 256, 1, 1}, {256, 1, 1});
  Enqueue(state, events, "down", kernels.down,
          {1024, 24, active_experts}, {128, 4, 1});
  Enqueue(state, events, "down_residual", kernels.down_residual,
          {kCompleteRows * 8 * 256, 1, 1}, {256, 1, 1});
  Enqueue(state, events, "shared_scalar_gate", kernels.scalar_gate,
          {kTokens * 256, 1, 1}, {256, 1, 1});
  Enqueue(state, events, "scatter_add", kernels.final,
          {kTokens * kHidden, 1, 1}, {256, 1, 1});
  Check(clWaitForEvents(1, &events.values.back()), "clWaitForEvents final");
  const auto wall_end = std::chrono::steady_clock::now();

  Sample sample;
  sample.wall_us = std::chrono::duration<double, std::micro>(
      wall_end - wall_start).count();
  sample.stage_us.reserve(events.values.size());
  cl_ulong first_start = 0;
  cl_ulong last_end = 0;
  for (std::size_t index = 0; index < events.values.size(); ++index) {
    cl_ulong start = 0;
    cl_ulong end = 0;
    Check(clGetEventProfilingInfo(
        events.values[index], CL_PROFILING_COMMAND_START,
        sizeof(start), &start, nullptr), "profile start");
    Check(clGetEventProfilingInfo(
        events.values[index], CL_PROFILING_COMMAND_END,
        sizeof(end), &end, nullptr), "profile end");
    if (index == 0) first_start = start;
    last_end = end;
    sample.stage_us.push_back(static_cast<double>(end - start) / 1000.0);
  }
  sample.device_span_us = static_cast<double>(last_end - first_start) / 1000.0;
  return sample;
}

double Median(std::vector<double> values) {
  Require(!values.empty(), "median input is empty");
  std::sort(values.begin(), values.end());
  const std::size_t middle = values.size() / 2;
  return values.size() % 2 == 0
      ? (values[middle - 1] + values[middle]) * 0.5
      : values[middle];
}

struct CompareStats {
  bool finite = true;
  double cosine = 0.0;
  double relative_l2 = 0.0;
  double max_abs_diff = 0.0;
  double rmse = 0.0;
};

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
      bits = sign |
          (static_cast<std::uint32_t>(127 - 15 - shift) << 23) |
          (mantissa << 13);
    }
  } else if (exponent == 31) {
    bits = sign | 0x7f800000U | (mantissa << 13);
  } else {
    bits = sign | ((exponent + 112U) << 23) | (mantissa << 13);
  }
  float result = 0.0f;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

CompareStats Compare(const std::vector<float>& actual,
                     const std::vector<float>& reference) {
  Require(actual.size() == reference.size(), "comparison size mismatch");
  long double dot = 0.0;
  long double actual_norm = 0.0;
  long double reference_norm = 0.0;
  long double error_norm = 0.0;
  CompareStats stats;
  for (std::size_t index = 0; index < actual.size(); ++index) {
    const double lhs = actual[index];
    const double rhs = reference[index];
    const double delta = lhs - rhs;
    stats.finite = stats.finite && std::isfinite(lhs) && std::isfinite(rhs);
    stats.max_abs_diff = std::max(stats.max_abs_diff, std::abs(delta));
    dot += lhs * rhs;
    actual_norm += lhs * lhs;
    reference_norm += rhs * rhs;
    error_norm += delta * delta;
  }
  stats.cosine = static_cast<double>(
      dot / std::sqrt(actual_norm * reference_norm));
  stats.relative_l2 = static_cast<double>(
      std::sqrt(error_norm / reference_norm));
  stats.rmse = std::sqrt(
      static_cast<double>(error_norm / actual.size()));
  return stats;
}

std::string DeviceString(cl_device_id device, cl_device_info key) {
  std::size_t size = 0;
  Check(clGetDeviceInfo(device, key, 0, nullptr, &size),
        "clGetDeviceInfo string size");
  std::string value(size, '\0');
  Check(clGetDeviceInfo(device, key, size, value.data(), nullptr),
        "clGetDeviceInfo string");
  while (!value.empty() && value.back() == '\0') value.pop_back();
  return value;
}

std::string Join(const std::string& directory, const char* name) {
  return directory + "/" + name;
}

int Main(int argc, char** argv) {
  const Args args = ParseArgs(argc, argv);
  const auto input = ReadVector<float>(args.input, kTokens * kHidden);
  const auto topk = ReadBytes(args.topk);
  const auto router = ReadVector<float>(args.router_weights, kRoutedRows);
  const auto oracle = ReadVector<float>(args.oracle, kTokens * kHidden);
  const Schedule schedule = BuildSchedule(topk, args.topk_stride, router);

  OpenClState state = CreateOpenCl();
  const cl_program gate_program = LoadBinary(state, args.gate_binary);
  const cl_program up_program = LoadBinary(state, args.up_binary);
  const cl_program down_program = LoadBinary(state, args.down_binary);
  const cl_program support_program = BuildSource(state, args.support_source);
  Kernels kernels;
  kernels.gather = CreateKernel(
      state, support_program, "iq36_micro_gather_f16_sums32");
  kernels.up = CreateKernel(state, up_program, "iq36_moe_micro_up");
  kernels.gate = CreateKernel(state, gate_program, "iq36_moe_micro_gate");
  kernels.swiglu = CreateKernel(
      state, support_program,
      "iq36_micro_q4k_residual_swiglu_sums32");
  kernels.down = CreateKernel(state, down_program, "iq36_moe_micro_down");
  kernels.down_residual = CreateKernel(
      state, support_program, "iq36_micro_q4k_down_residual_weight");
  kernels.scalar_gate = CreateKernel(
      state, support_program, "iq36_micro_shared_scalar_gate");
  kernels.final = CreateKernel(
      state, support_program, "iq36_micro_complete_scatter_add");

  const cl_mem input_mem = CreateCopiedVector(state, input);
  const cl_mem active_mem = CreateCopiedVector(state, schedule.active_experts);
  const cl_mem starts_mem = CreateCopiedVector(state, schedule.starts);
  const cl_mem lengths_mem = CreateCopiedVector(state, schedule.lengths);
  const cl_mem token_map_mem = CreateCopiedVector(state, schedule.token_map);
  const cl_mem inverse_mem = CreateCopiedVector(state, schedule.inverse_map);
  const cl_mem row_expert_mem = CreateCopiedVector(state, schedule.row_expert);
  const cl_mem router_mem = CreateCopiedVector(
      state, schedule.compact_router_weights);
  const cl_mem scalar_weight = CreateCopiedFile(
      state, Join(args.prep_dir, "shared-scalar-gate.f32"),
      kHidden * sizeof(float));

  const std::size_t gu_weight_bytes =
      kCombinedExperts * kIntermediate * kHidden / 2;
  const std::size_t gu_scale_bytes =
      kCombinedExperts * (kHidden / 32) * kIntermediate * sizeof(std::uint16_t);
  const std::size_t gu_zp_bytes =
      kCombinedExperts * (kHidden / 32) * kIntermediate / 2;
  const std::size_t gu_min_bytes =
      kCombinedExperts * (kHidden / 32) * kIntermediate * sizeof(float);
  const std::size_t down_weight_bytes =
      kCombinedExperts * kHidden * kIntermediate / 2;
  const std::size_t down_scale_bytes =
      kCombinedExperts * (kIntermediate / 32) * kHidden * sizeof(std::uint16_t);
  const std::size_t down_zp_bytes =
      kCombinedExperts * (kIntermediate / 32) * kHidden / 2;
  const std::size_t down_min_bytes =
      kCombinedExperts * (kIntermediate / 32) * kHidden * sizeof(float);

  const cl_mem gate_weights = CreateCopiedFile(
      state, Join(args.prep_dir, "gate-weights.u4"), gu_weight_bytes);
  const cl_mem gate_scales = CreateCopiedFile(
      state, Join(args.prep_dir, "gate-scales.f16"), gu_scale_bytes);
  const cl_mem gate_zps = CreateCopiedFile(
      state, Join(args.prep_dir, "gate-zps.u4"), gu_zp_bytes);
  const cl_mem gate_mins = CreateCopiedFile(
      state, Join(args.prep_dir, "gate-mins.f32"), gu_min_bytes);
  const cl_mem up_weights = CreateCopiedFile(
      state, Join(args.prep_dir, "up-weights.u4"), gu_weight_bytes);
  const cl_mem up_scales = CreateCopiedFile(
      state, Join(args.prep_dir, "up-scales.f16"), gu_scale_bytes);
  const cl_mem up_zps = CreateCopiedFile(
      state, Join(args.prep_dir, "up-zps.u4"), gu_zp_bytes);
  const cl_mem up_mins = CreateCopiedFile(
      state, Join(args.prep_dir, "up-mins.f32"), gu_min_bytes);
  const cl_mem down_weights = CreateCopiedFile(
      state, Join(args.prep_dir, "down-weights.u4"), down_weight_bytes);
  const cl_mem down_scales = CreateCopiedFile(
      state, Join(args.prep_dir, "down-scales.f16"), down_scale_bytes);
  const cl_mem down_zps = CreateCopiedFile(
      state, Join(args.prep_dir, "down-zps.u4"), down_zp_bytes);
  const cl_mem down_mins = CreateCopiedFile(
      state, Join(args.prep_dir, "down-mins.f32"), down_min_bytes);

  const cl_mem grouped_input = CreateEmpty(
      state, kCompleteRows * kHidden * sizeof(std::uint16_t));
  const cl_mem input_sums = CreateEmpty(
      state, kCompleteRows * (kHidden / 32) * sizeof(float));
  const cl_mem gate_main = CreateEmpty(
      state, kCompleteRows * kIntermediate * sizeof(std::uint16_t));
  const cl_mem up_main = CreateEmpty(
      state, kCompleteRows * kIntermediate * sizeof(std::uint16_t));
  const cl_mem swiglu = CreateEmpty(
      state, kCompleteRows * kIntermediate * sizeof(std::uint16_t));
  const cl_mem down_sums = CreateEmpty(
      state, kCompleteRows * (kIntermediate / 32) * sizeof(float));
  const cl_mem down_main = CreateEmpty(
      state, kCompleteRows * kHidden * sizeof(std::uint16_t));
  const cl_mem contributions = CreateEmpty(
      state, kCompleteRows * kHidden * sizeof(float));
  const cl_mem shared_gate = CreateEmpty(state, kTokens * sizeof(float));
  const cl_mem output = CreateEmpty(
      state, kTokens * kHidden * sizeof(float));

  SetMem(kernels.gather, 0, input_mem, "set gather input");
  SetMem(kernels.gather, 1, token_map_mem, "set gather map");
  SetMem(kernels.gather, 2, grouped_input, "set gather output");
  SetMem(kernels.gather, 3, input_sums, "set gather sums");
  SetMicroArgs(kernels.up, grouped_input, up_weights, up_main,
               active_mem, starts_mem, lengths_mem,
               kIntermediate, kHidden, up_scales, up_zps);
  SetMicroArgs(kernels.gate, grouped_input, gate_weights, gate_main,
               active_mem, starts_mem, lengths_mem,
               kIntermediate, kHidden, gate_scales, gate_zps);
  SetMem(kernels.swiglu, 0, gate_main, "set swiglu gate");
  SetMem(kernels.swiglu, 1, up_main, "set swiglu up");
  SetMem(kernels.swiglu, 2, input_sums, "set swiglu sums");
  SetMem(kernels.swiglu, 3, gate_mins, "set swiglu gate mins");
  SetMem(kernels.swiglu, 4, up_mins, "set swiglu up mins");
  SetMem(kernels.swiglu, 5, row_expert_mem, "set swiglu experts");
  SetMem(kernels.swiglu, 6, swiglu, "set swiglu output");
  SetMem(kernels.swiglu, 7, down_sums, "set down sums");
  SetMicroArgs(kernels.down, swiglu, down_weights, down_main,
               active_mem, starts_mem, lengths_mem,
               kHidden, kIntermediate, down_scales, down_zps);
  SetMem(kernels.down_residual, 0, down_main, "set down main");
  SetMem(kernels.down_residual, 1, down_sums, "set down input sums");
  SetMem(kernels.down_residual, 2, down_mins, "set down mins");
  SetMem(kernels.down_residual, 3, row_expert_mem, "set down experts");
  SetMem(kernels.down_residual, 4, router_mem, "set down router weights");
  SetMem(kernels.down_residual, 5, contributions,
         "set complete contributions");
  SetMem(kernels.scalar_gate, 0, input_mem, "set scalar input");
  SetMem(kernels.scalar_gate, 1, scalar_weight, "set scalar weight");
  SetMem(kernels.scalar_gate, 2, shared_gate, "set scalar output");
  SetMem(kernels.final, 0, contributions, "set final contributions");
  SetMem(kernels.final, 1, inverse_mem, "set final inverse map");
  SetMem(kernels.final, 2, shared_gate, "set final shared gate");
  SetMem(kernels.final, 3, output, "set final output");

  Check(clFinish(state.queue), "finish setup");
  for (int index = 0; index < args.warmup; ++index) {
    (void)RunOnce(state, kernels, schedule.active_experts.size());
  }
  std::vector<Sample> samples;
  samples.reserve(static_cast<std::size_t>(args.repeat));
  for (int index = 0; index < args.repeat; ++index) {
    samples.push_back(
        RunOnce(state, kernels, schedule.active_experts.size()));
  }

  std::vector<float> observed(kTokens * kHidden);
  Check(clEnqueueReadBuffer(
      state.queue, output, CL_TRUE, 0, observed.size() * sizeof(float),
      observed.data(), 0, nullptr, nullptr), "read final output");
  const CompareStats comparison = Compare(observed, oracle);
  if (!args.debug_dir.empty()) {
    std::filesystem::create_directories(args.debug_dir);
    const auto dump_half_row = [&](const char* name, cl_mem memory,
                                   std::size_t count) {
      std::vector<std::uint16_t> values(count);
      Check(clEnqueueReadBuffer(
          state.queue, memory, CL_TRUE, 0, count * sizeof(std::uint16_t),
          values.data(), 0, nullptr, nullptr),
          std::string("read debug ") + name);
      WriteVector(args.debug_dir + "/" + name, values);
    };
    dump_half_row("grouped-row0.f16", grouped_input, kHidden);
    dump_half_row("gate-row0.f16", gate_main, kIntermediate);
    dump_half_row("up-row0.f16", up_main, kIntermediate);
    dump_half_row("swiglu-row0.f16", swiglu, kIntermediate);
    dump_half_row("down-row0.f16", down_main, kHidden);
  }
  CompareStats swiglu_comparison;
  CompareStats weighted_down_comparison;
  CompareStats routed_comparison;
  const bool diagnostics = !args.swiglu_oracle.empty() &&
      !args.weighted_down_oracle.empty() && !args.routed_oracle.empty();
  if (diagnostics) {
    std::vector<std::uint16_t> swiglu_half(kRoutedRows * kIntermediate);
    Check(clEnqueueReadBuffer(
        state.queue, swiglu, CL_TRUE, 0,
        swiglu_half.size() * sizeof(std::uint16_t), swiglu_half.data(),
        0, nullptr, nullptr), "read swiglu diagnostic");
    std::vector<float> swiglu_float(swiglu_half.size());
    std::transform(swiglu_half.begin(), swiglu_half.end(),
                   swiglu_float.begin(), HalfToFloat);
    swiglu_comparison = Compare(
        swiglu_float, ReadVector<float>(
            args.swiglu_oracle, kRoutedRows * kIntermediate));

    std::vector<float> weighted_down(kRoutedRows * kHidden);
    Check(clEnqueueReadBuffer(
        state.queue, contributions, CL_TRUE, 0,
        weighted_down.size() * sizeof(float), weighted_down.data(),
        0, nullptr, nullptr), "read weighted down diagnostic");
    weighted_down_comparison = Compare(
        weighted_down, ReadVector<float>(
            args.weighted_down_oracle, kRoutedRows * kHidden));
    std::vector<float> routed(kTokens * kHidden, 0.0f);
    for (std::size_t token = 0; token < kTokens; ++token) {
      for (std::size_t hidden = 0; hidden < kHidden; ++hidden) {
        float sum = 0.0f;
        for (std::size_t rank = 0; rank < kTopK; ++rank) {
          const std::int32_t row =
              schedule.inverse_map[token * kTopK + rank];
          sum += weighted_down[static_cast<std::size_t>(row) * kHidden +
                               hidden];
        }
        routed[token * kHidden + hidden] = sum;
      }
    }
    routed_comparison = Compare(
        routed, ReadVector<float>(
            args.routed_oracle, kTokens * kHidden));
  }
  std::vector<double> device_samples;
  std::vector<double> wall_samples;
  for (const Sample& sample : samples) {
    device_samples.push_back(sample.device_span_us);
    wall_samples.push_back(sample.wall_us);
  }
  const double device_median = Median(device_samples);
  const double wall_median = Median(wall_samples);
  std::vector<double> stage_medians(samples.front().stage_us.size());
  for (std::size_t stage = 0; stage < stage_medians.size(); ++stage) {
    std::vector<double> values;
    for (const Sample& sample : samples) values.push_back(sample.stage_us[stage]);
    stage_medians[stage] = Median(values);
  }
  const bool correctness = comparison.finite &&
      comparison.cosine >= 0.999 && comparison.relative_l2 <= 0.002;
  const bool performance = device_median <= args.cap_us;

  std::cout.precision(12);
  std::cout << "{\"schema_version\":\"intel-qwen36-openvino-moe-micro-runtime-v0\"";
  std::cout << ",\"device_name\":\"" << DeviceString(state.device, CL_DEVICE_NAME) << "\"";
  std::cout << ",\"driver_version\":\"" << DeviceString(state.device, CL_DRIVER_VERSION) << "\"";
  std::cout << ",\"routed_active_experts\":"
            << schedule.routed_active_experts;
  std::cout << ",\"complete_active_experts\":"
            << schedule.active_experts.size();
  std::cout << ",\"max_routed_group\":" << schedule.max_routed_group;
  std::cout << ",\"maps_native_only\":true";
  std::cout << ",\"timed_host_upload_bytes\":0,\"timed_host_readback_bytes\":0";
  std::cout << ",\"cap_us\":" << args.cap_us;
  std::cout << ",\"device_span_median_us\":" << device_median;
  std::cout << ",\"wall_median_us\":" << wall_median;
  std::cout << ",\"device_samples_us\":[";
  for (std::size_t index = 0; index < device_samples.size(); ++index) {
    if (index != 0) std::cout << ',';
    std::cout << device_samples[index];
  }
  std::cout << "]";
  const std::array<const char*, 8> stage_names = {
      "gather", "up", "gate", "swiglu_residual", "down",
      "down_residual", "shared_scalar_gate", "scatter_add"};
  std::cout << ",\"stage_median_us\":{";
  for (std::size_t index = 0; index < stage_names.size(); ++index) {
    if (index != 0) std::cout << ',';
    std::cout << '\"' << stage_names[index] << "\":" << stage_medians[index];
  }
  std::cout << "}";
  std::cout << ",\"compare\":{";
  std::cout << "\"finite\":" << (comparison.finite ? "true" : "false");
  std::cout << ",\"cosine\":" << comparison.cosine;
  std::cout << ",\"relative_l2\":" << comparison.relative_l2;
  std::cout << ",\"max_abs_diff\":" << comparison.max_abs_diff;
  std::cout << ",\"rmse\":" << comparison.rmse << "}";
  if (diagnostics) {
    const auto write_compare = [](const char* name, const CompareStats& value) {
      std::cout << ",\"" << name << "\":{";
      std::cout << "\"finite\":" << (value.finite ? "true" : "false");
      std::cout << ",\"cosine\":" << value.cosine;
      std::cout << ",\"relative_l2\":" << value.relative_l2;
      std::cout << ",\"max_abs_diff\":" << value.max_abs_diff;
      std::cout << ",\"rmse\":" << value.rmse << "}";
    };
    write_compare("swiglu_compare", swiglu_comparison);
    write_compare("weighted_down_compare", weighted_down_comparison);
    write_compare("routed_compare", routed_comparison);
  }
  std::cout << ",\"correctness_pass\":"
            << (correctness ? "true" : "false");
  std::cout << ",\"performance_pass\":"
            << (performance ? "true" : "false");
  std::cout << ",\"required_checks_passed\":"
            << (correctness && performance ? "true" : "false") << "}\n";
  return correctness && performance ? 0 : 2;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return Main(argc, argv);
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
