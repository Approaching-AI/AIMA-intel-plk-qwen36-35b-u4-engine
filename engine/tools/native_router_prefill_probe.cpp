#include <CL/cl.h>
#include <CL/cl_ext.h>

#include "intel_qwen36/gguf_loader.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kTokenCount = 1024;
constexpr std::size_t kTopK = 8;
constexpr std::size_t kAssignmentCount = kTokenCount * kTopK;
constexpr std::size_t kExpertCount = 256;

void Check(cl_int status, const char* where) {
  if (status != CL_SUCCESS) {
    throw std::runtime_error(std::string(where) + " failed: " +
                             std::to_string(status));
  }
}

template <typename Value>
std::vector<Value> ReadVector(const std::string& path,
                              std::size_t expected_count = 0) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("could not open input: " + path);
  input.seekg(0, std::ios::end);
  const auto size = input.tellg();
  if (size < 0 || size % static_cast<std::streamoff>(sizeof(Value)) != 0) {
    throw std::runtime_error("invalid input size: " + path);
  }
  const auto count = static_cast<std::size_t>(size) / sizeof(Value);
  if (expected_count != 0 && count != expected_count) {
    throw std::runtime_error("input count mismatch: " + path);
  }
  input.seekg(0, std::ios::beg);
  std::vector<Value> values(count);
  input.read(reinterpret_cast<char*>(values.data()),
             static_cast<std::streamsize>(size));
  if (!input) throw std::runtime_error("could not read input: " + path);
  return values;
}

std::string ReadText(const std::string& path) {
  const auto bytes = ReadVector<char>(path);
  return std::string(bytes.begin(), bytes.end());
}

struct OpenClState {
  cl_context context = nullptr;
  cl_command_queue queue = nullptr;
  cl_program program = nullptr;
  clMemFreeINTEL_fn mem_free = nullptr;
  std::vector<void*> usm_allocations;
  std::vector<cl_program> extra_programs;
  std::vector<cl_kernel> kernels;
  std::vector<cl_mem> buffers;

  ~OpenClState() {
    for (auto buffer : buffers) clReleaseMemObject(buffer);
    if (mem_free != nullptr) {
      for (auto allocation : usm_allocations) {
        mem_free(context, allocation);
      }
    }
    for (auto kernel : kernels) clReleaseKernel(kernel);
    for (auto program_value : extra_programs) {
      clReleaseProgram(program_value);
    }
    if (program) clReleaseProgram(program);
    if (queue) clReleaseCommandQueue(queue);
    if (context) clReleaseContext(context);
  }
};

struct DeviceSelection {
  cl_platform_id platform = nullptr;
  cl_device_id device = nullptr;
  std::string name;
};

DeviceSelection SelectDevice() {
  cl_uint platform_count = 0;
  Check(clGetPlatformIDs(0, nullptr, &platform_count),
        "clGetPlatformIDs count");
  std::vector<cl_platform_id> platforms(platform_count);
  Check(clGetPlatformIDs(platform_count, platforms.data(), nullptr),
        "clGetPlatformIDs");
  DeviceSelection fallback;
  for (auto platform : platforms) {
    cl_uint count = 0;
    if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &count) !=
            CL_SUCCESS ||
        count == 0) {
      continue;
    }
    std::vector<cl_device_id> devices(count);
    Check(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, count, devices.data(),
                         nullptr),
          "clGetDeviceIDs");
    for (auto device : devices) {
      std::array<char, 256> name{};
      Check(clGetDeviceInfo(device, CL_DEVICE_NAME, name.size(), name.data(),
                            nullptr),
            "clGetDeviceInfo name");
      DeviceSelection selected{platform, device, name.data()};
      if (!fallback.device) fallback = selected;
      if (selected.name.find("Arc") != std::string::npos) return selected;
    }
  }
  if (!fallback.device) throw std::runtime_error("no OpenCL GPU found");
  return fallback;
}

cl_mem CreateBuffer(OpenClState& state, std::size_t bytes, cl_mem_flags flags,
                    void* host = nullptr) {
  cl_int status = CL_SUCCESS;
  auto buffer = clCreateBuffer(state.context, flags, bytes, host, &status);
  Check(status, "clCreateBuffer");
  state.buffers.push_back(buffer);
  return buffer;
}

template <typename Value>
cl_mem CreateInput(OpenClState& state, const std::vector<Value>& values) {
  return CreateBuffer(state, values.size() * sizeof(Value),
                      CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                      const_cast<Value*>(values.data()));
}

double Median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

void PrintSamples(const std::vector<double>& values) {
  std::cout << '[';
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0) std::cout << ',';
    std::cout << values[index];
  }
  std::cout << ']';
}

int RunRouterProbe(int argc, char** argv) {
  if (argc < 9 || argc > 11) {
    throw std::invalid_argument(
        "usage: native-router-prefill-probe --router MODEL LAYER "
        "NATIVE_PROGRAM KERNEL HIDDEN TOPK ROUTER_WEIGHTS "
        "[WARMUP] [REPEAT]");
  }
  const std::string model_path = argv[2];
  const int layer = std::stoi(argv[3]);
  if (layer < 0 || layer >= 40) {
    throw std::invalid_argument("router layer is out of range");
  }
  const int warmup = argc >= 10 ? std::stoi(argv[9]) : 5;
  const int repeat = argc >= 11 ? std::stoi(argv[10]) : 21;
  if (warmup < 0 || repeat < 3) {
    throw std::invalid_argument("router warmup/repeat are invalid");
  }
  constexpr std::size_t kHiddenSize = 2048;
  constexpr double kRouterCapUs = 500.0;
  constexpr double kWeightTolerance = 0.002;
  const auto hidden = ReadVector<float>(
      argv[6], kTokenCount * kHiddenSize);
  const auto reference_ids = ReadVector<std::int32_t>(
      argv[7], kAssignmentCount);
  const auto reference_weights = ReadVector<float>(
      argv[8], kAssignmentCount);

  const auto index = iq36::parse_gguf_model_index(model_path);
  const std::string tensor_name = "blk." + std::to_string(layer) +
      ".ffn_gate_inp.weight";
  const auto* tensor = iq36::find_tensor(index, tensor_name);
  if (tensor == nullptr || tensor->type != 0 ||
      tensor->dims != std::vector<std::uint64_t>{2048, 256}) {
    throw std::runtime_error("router tensor contract mismatch");
  }
  std::vector<float> weights(kExpertCount * kHiddenSize);
  for (std::size_t expert = 0; expert < kExpertCount; ++expert) {
    const auto row = iq36::decode_tensor_row(
        model_path, index, tensor_name, expert);
    if (row.size() != kHiddenSize) {
      throw std::runtime_error("decoded router row size mismatch");
    }
    for (std::size_t column = 0; column < kHiddenSize; ++column) {
      weights[expert * kHiddenSize + column] = row[column];
    }
  }

  const auto selected = SelectDevice();
  OpenClState state;
  cl_int status = CL_SUCCESS;
  state.context = clCreateContext(nullptr, 1, &selected.device, nullptr,
                                  nullptr, &status);
  Check(status, "router clCreateContext");
  state.queue = clCreateCommandQueue(state.context, selected.device,
                                     CL_QUEUE_PROFILING_ENABLE, &status);
  Check(status, "router clCreateCommandQueue");
  const auto source_text = ReadText(argv[5]);
  const char* source = source_text.c_str();
  const std::size_t source_size = source_text.size();
  state.program = clCreateProgramWithSource(
      state.context, 1, &source, &source_size, &status);
  Check(status, "router clCreateProgramWithSource");
  status = clBuildProgram(
      state.program, 1, &selected.device, "", nullptr, nullptr);
  if (status != CL_SUCCESS) {
    std::size_t log_size = 0;
    clGetProgramBuildInfo(state.program, selected.device,
                          CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
    std::string log(log_size, '\0');
    clGetProgramBuildInfo(state.program, selected.device,
                          CL_PROGRAM_BUILD_LOG, log.size(), log.data(),
                          nullptr);
    throw std::runtime_error("router OpenCL build failed: " + log);
  }
  const auto native_program_bytes = ReadVector<std::uint8_t>(argv[4]);
  const std::size_t native_program_size = native_program_bytes.size();
  const unsigned char* native_program_data = native_program_bytes.data();
  cl_int binary_status = CL_SUCCESS;
  auto native_program = clCreateProgramWithBinary(
      state.context, 1, &selected.device, &native_program_size,
      &native_program_data, &binary_status, &status);
  Check(status, "router clCreateProgramWithBinary");
  Check(binary_status, "router binary status");
  state.extra_programs.push_back(native_program);
  Check(clBuildProgram(native_program, 1, &selected.device, "", nullptr,
                       nullptr),
        "router native clBuildProgram");
  const auto CreateKernel = [&](cl_program program, const char* name) {
    auto kernel = clCreateKernel(program, name, &status);
    Check(status, "router clCreateKernel");
    state.kernels.push_back(kernel);
    return kernel;
  };
  const auto router_kernel = CreateKernel(native_program, "gemm_kernel");
  const auto topk_kernel = CreateKernel(
      state.program, "iq36_router_topk8_1024");

  auto hidden_buffer = CreateInput(state, hidden);
  auto weight_buffer = CreateInput(state, weights);
  auto logits_buffer = CreateBuffer(
      state, kTokenCount * kExpertCount * sizeof(float), CL_MEM_READ_WRITE);
  auto id_buffer = CreateBuffer(
      state, kAssignmentCount * sizeof(std::int32_t), CL_MEM_WRITE_ONLY);
  auto router_weight_buffer = CreateBuffer(
      state, kAssignmentCount * sizeof(float), CL_MEM_WRITE_ONLY);
  // The pinned 1024x2048x256 PTL JIT launch has at most 96 K-sliced groups;
  // each group owns one 64-byte status line. The status carrier must be USM
  // because the generated kernel uses a stateless atomic pointer.
  constexpr std::size_t kRouterStatusBytes = 96 * 64;
  const auto DeviceAllocate = reinterpret_cast<clDeviceMemAllocINTEL_fn>(
      clGetExtensionFunctionAddressForPlatform(
          selected.platform, "clDeviceMemAllocINTEL"));
  state.mem_free = reinterpret_cast<clMemFreeINTEL_fn>(
      clGetExtensionFunctionAddressForPlatform(
          selected.platform, "clMemFreeINTEL"));
  const auto SetUsmArgument =
      reinterpret_cast<clSetKernelArgMemPointerINTEL_fn>(
          clGetExtensionFunctionAddressForPlatform(
              selected.platform, "clSetKernelArgMemPointerINTEL"));
  const auto FillUsm = reinterpret_cast<clEnqueueMemFillINTEL_fn>(
      clGetExtensionFunctionAddressForPlatform(
          selected.platform, "clEnqueueMemFillINTEL"));
  if (DeviceAllocate == nullptr || state.mem_free == nullptr ||
      SetUsmArgument == nullptr || FillUsm == nullptr) {
    throw std::runtime_error("router USM extension is unavailable");
  }
  auto status_pointer = DeviceAllocate(
      state.context, selected.device, nullptr, kRouterStatusBytes, 64,
      &status);
  Check(status, "allocate router USM status");
  if (status_pointer == nullptr) {
    throw std::runtime_error("router USM status allocation returned null");
  }
  state.usm_allocations.push_back(status_pointer);
  const auto SetMemoryArgument = [&](cl_kernel kernel, cl_uint arg,
                                     cl_mem value) {
    Check(clSetKernelArg(kernel, arg, sizeof(value), &value),
          "router clSetKernelArg memory");
  };
  const auto SetValueArgument = [&](cl_kernel kernel, cl_uint arg,
                                    const auto& value) {
    Check(clSetKernelArg(kernel, arg, sizeof(value), &value),
          "router clSetKernelArg value");
  };
  const auto SetMemoryArguments = [&](cl_kernel kernel,
                                      const std::vector<cl_mem>& arguments) {
    for (cl_uint arg = 0; arg < arguments.size(); ++arg) {
      SetMemoryArgument(kernel, arg, arguments[arg]);
    }
  };
  SetMemoryArgument(router_kernel, 0, weight_buffer);
  SetMemoryArgument(router_kernel, 1, hidden_buffer);
  SetMemoryArgument(router_kernel, 2, logits_buffer);
  const std::int64_t zero_offset = 0;
  SetValueArgument(router_kernel, 3, zero_offset);
  SetValueArgument(router_kernel, 4, zero_offset);
  SetValueArgument(router_kernel, 5, zero_offset);
  const std::int32_t lda = 2048;
  const std::int32_t ldb = 2048;
  const std::int32_t ldc = 256;
  const std::int32_t m = 256;
  const std::int32_t n = 1024;
  const std::int32_t k = 2048;
  SetValueArgument(router_kernel, 6, lda);
  SetValueArgument(router_kernel, 7, ldb);
  SetValueArgument(router_kernel, 8, ldc);
  SetValueArgument(router_kernel, 9, m);
  SetValueArgument(router_kernel, 10, n);
  SetValueArgument(router_kernel, 11, k);
  const float alpha = 1.0f;
  const float beta = 0.0f;
  SetValueArgument(router_kernel, 12, alpha);
  SetValueArgument(router_kernel, 13, beta);
  const std::uint32_t flags = 0;
  SetValueArgument(router_kernel, 14, flags);
  Check(SetUsmArgument(router_kernel, 15, status_pointer),
        "router clSetKernelArgMemPointerINTEL");
  // These specialization constants and the 3-D range are extracted by the
  // clean codegen gate from the pinned oneDNN commit. This runtime loads only
  // the resulting native OpenCL program and does not link oneDNN. Regenerate
  // and re-gate them together if the exact shape or code generator changes.
  const std::array<std::uint32_t, 7> dispatch_arguments = {
      4U, 16U, 0U, 2776U, 64U, 4228890877U, 48U};
  for (cl_uint index_value = 0; index_value < dispatch_arguments.size();
       ++index_value) {
    SetValueArgument(
        router_kernel, 16 + index_value, dispatch_arguments[index_value]);
  }
  Check(clSetKernelArg(router_kernel, 23, 0, nullptr),
        "router clSetKernelArg local");
  SetMemoryArguments(topk_kernel,
                     {logits_buffer, id_buffer, router_weight_buffer});

  const std::array<std::size_t, 3> router_global = {1536, 4, 1};
  const std::array<std::size_t, 3> router_local = {32, 4, 1};
  constexpr std::size_t topk_global = kTokenCount * kExpertCount;
  constexpr std::size_t topk_local = 256;
  const std::uint32_t zero = 0;
  Check(FillUsm(state.queue, status_pointer, &zero, sizeof(zero),
                kRouterStatusBytes, 0, nullptr, nullptr),
        "clear router USM status");
  Check(clFinish(state.queue), "finish router status clear");
  const auto Enqueue = [&](std::array<cl_event, 2>* events) {
    auto Event = [&](std::size_t stage) -> cl_event* {
      return events == nullptr ? nullptr : &(*events)[stage];
    };
    Check(clEnqueueNDRangeKernel(
              state.queue, router_kernel, 3, nullptr, router_global.data(),
              router_local.data(), 0, nullptr, Event(0)),
          "enqueue router matmul");
    Check(clEnqueueNDRangeKernel(
              state.queue, topk_kernel, 1, nullptr, &topk_global,
              &topk_local, 0, nullptr, Event(1)),
          "enqueue router top-k");
  };
  for (int iteration = 0; iteration < warmup; ++iteration) Enqueue(nullptr);
  Check(clFinish(state.queue), "finish router warmup");
  std::vector<double> samples;
  std::array<double, 2> stage_sums{};
  samples.reserve(repeat);
  for (int iteration = 0; iteration < repeat; ++iteration) {
    std::array<cl_event, 2> events{};
    Enqueue(&events);
    Check(clWaitForEvents(1, &events.back()), "wait router probe");
    cl_ulong first_start = 0;
    cl_ulong last_end = 0;
    Check(clGetEventProfilingInfo(
              events.front(), CL_PROFILING_COMMAND_START,
              sizeof(first_start), &first_start, nullptr),
          "read router start");
    Check(clGetEventProfilingInfo(
              events.back(), CL_PROFILING_COMMAND_END,
              sizeof(last_end), &last_end, nullptr),
          "read router end");
    for (std::size_t stage = 0; stage < events.size(); ++stage) {
      cl_ulong begin = 0;
      cl_ulong end = 0;
      Check(clGetEventProfilingInfo(
                events[stage], CL_PROFILING_COMMAND_START,
                sizeof(begin), &begin, nullptr),
            "read router stage start");
      Check(clGetEventProfilingInfo(
                events[stage], CL_PROFILING_COMMAND_END,
                sizeof(end), &end, nullptr),
            "read router stage end");
      stage_sums[stage] += static_cast<double>(end - begin) / 1000.0;
      clReleaseEvent(events[stage]);
    }
    samples.push_back(static_cast<double>(last_end - first_start) / 1000.0);
  }

  std::vector<std::int32_t> observed_ids(kAssignmentCount);
  std::vector<float> observed_weights(kAssignmentCount);
  Check(clEnqueueReadBuffer(
            state.queue, id_buffer, CL_TRUE, 0,
            observed_ids.size() * sizeof(std::int32_t), observed_ids.data(),
            0, nullptr, nullptr),
        "read router ids");
  Check(clEnqueueReadBuffer(
            state.queue, router_weight_buffer, CL_TRUE, 0,
            observed_weights.size() * sizeof(float), observed_weights.data(),
            0, nullptr, nullptr),
        "read router weights");
  std::size_t ordered_match_rows = 0;
  std::size_t set_match_rows = 0;
  std::size_t missing_experts = 0;
  double maximum_router_weight_diff = 0.0;
  for (std::size_t token = 0; token < kTokenCount; ++token) {
    bool ordered = true;
    std::array<bool, kExpertCount> reference_set{};
    std::array<bool, kExpertCount> observed_set{};
    std::array<float, kExpertCount> reference_by_id{};
    for (std::size_t rank = 0; rank < kTopK; ++rank) {
      const auto source = token * kTopK + rank;
      const int reference_id = reference_ids[source];
      const int observed_id = observed_ids[source];
      if (reference_id < 0 || reference_id >= 256 ||
          observed_id < 0 || observed_id >= 256) {
        throw std::runtime_error("router top-k id is out of range");
      }
      ordered = ordered && reference_id == observed_id;
      reference_set[static_cast<std::size_t>(reference_id)] = true;
      observed_set[static_cast<std::size_t>(observed_id)] = true;
      reference_by_id[static_cast<std::size_t>(reference_id)] =
          reference_weights[source];
    }
    ordered_match_rows += ordered;
    bool set_match = true;
    for (std::size_t expert = 0; expert < kExpertCount; ++expert) {
      if (reference_set[expert] != observed_set[expert]) {
        set_match = false;
        missing_experts += reference_set[expert] && !observed_set[expert];
      }
    }
    set_match_rows += set_match;
    if (set_match) {
      for (std::size_t rank = 0; rank < kTopK; ++rank) {
        const auto source = token * kTopK + rank;
        const auto expert = static_cast<std::size_t>(observed_ids[source]);
        maximum_router_weight_diff = std::max(
            maximum_router_weight_diff,
            std::abs(static_cast<double>(observed_weights[source]) -
                     reference_by_id[expert]));
      }
    }
  }
  const double median = Median(samples);
  const bool accuracy = set_match_rows == kTokenCount &&
      maximum_router_weight_diff <= kWeightTolerance;
  const bool performance = median <= kRouterCapUs;
  const bool passed = accuracy && performance;
  std::cout << std::boolalpha << std::setprecision(12) << '{'
            << "\"device_name\":\"" << selected.name << "\","
            << "\"kernel_samples_us\":";
  PrintSamples(samples);
  std::cout << ',' << "\"maximum_router_weight_abs_diff\":"
            << maximum_router_weight_diff << ','
            << "\"missing_expert_count\":" << missing_experts << ','
            << "\"ordered_top8_match_rows\":" << ordered_match_rows << ','
            << "\"performance_pass\":" << performance << ','
            << "\"required_checks_passed\":" << passed << ','
            << "\"router_cap_us\":" << kRouterCapUs << ','
            << "\"router_median_us\":" << median << ','
            << "\"router_representation\":"
               "\"native_jit_f32_weight_f32_input\","
            << "\"set_top8_match_rows\":" << set_match_rows << ','
            << "\"stage_mean_us\":[";
  for (std::size_t stage = 0; stage < stage_sums.size(); ++stage) {
    if (stage != 0) std::cout << ',';
    std::cout << stage_sums[stage] / repeat;
  }
  std::cout << "],\"weight_tolerance\":" << kWeightTolerance << '}'
            << std::endl;
  return passed ? 0 : 2;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return RunRouterProbe(argc, argv);
  } catch (const std::exception& exception) {
    std::cerr << exception.what() << '\n';
    return 4;
  }
}
