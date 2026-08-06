#include <CL/cl.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kTokenCount = 1024;
constexpr std::size_t kTopK = 8;
constexpr std::size_t kAssignmentCount = kTokenCount * kTopK;
constexpr std::size_t kExpertCount = 256;
constexpr std::size_t kGateTaskCapacity = kAssignmentCount * 8;
constexpr std::size_t kDownTaskCapacity = kAssignmentCount * 12;
constexpr std::size_t kNativeGateTaskCapacity = kAssignmentCount;
constexpr std::size_t kNativeDownTaskCapacity = kAssignmentCount * 2;
constexpr double kDeviceScheduleCapUs = 60.0;

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

struct Schedule {
  std::array<std::int32_t, kExpertCount> offsets{};
  std::array<std::int32_t, kAssignmentCount> token_map{};
  std::array<std::int32_t, kAssignmentCount> inverse_map{};
  std::array<float, kAssignmentCount> router_weights{};
  std::vector<std::uint32_t> gate_tasks;
  std::vector<std::uint32_t> down_tasks;
  std::vector<std::uint32_t> native_gate_tasks;
  std::vector<std::uint32_t> native_down_tasks;
  std::uint32_t active_experts = 0;
  std::uint32_t max_group = 0;
};

Schedule BuildReference(const std::vector<std::int32_t>& topk,
                        const std::vector<float>& router_weights) {
  std::array<std::uint32_t, kExpertCount> counts{};
  for (std::size_t token = 0; token < kTokenCount; ++token) {
    for (std::size_t rank = 0; rank < kTopK; ++rank) {
      const auto source = token * kTopK + rank;
      const auto expert = topk[source];
      if (expert < 0 || expert >= static_cast<std::int32_t>(kExpertCount)) {
        throw std::runtime_error("top-k expert is out of range");
      }
      for (std::size_t prior = 0; prior < rank; ++prior) {
        if (expert == topk[token * kTopK + prior]) {
          throw std::runtime_error("top-k expert is duplicated");
        }
      }
      ++counts[static_cast<std::size_t>(expert)];
    }
  }

  Schedule result;
  std::array<std::uint32_t, kExpertCount> cursors{};
  std::uint32_t cumulative = 0;
  for (std::size_t expert = 0; expert < kExpertCount; ++expert) {
    const auto begin = cumulative;
    cursors[expert] = begin;
    cumulative += counts[expert];
    result.offsets[expert] = static_cast<std::int32_t>(cumulative);
    result.active_experts += counts[expert] != 0;
    result.max_group = std::max(result.max_group, counts[expert]);
    const auto m_tiles = (counts[expert] + 15U) / 16U;
    for (std::uint32_t output_tile = 0; output_tile < 64U; ++output_tile) {
      for (std::uint32_t m_tile = 0; m_tile < m_tiles; ++m_tile) {
        result.gate_tasks.push_back(static_cast<std::uint32_t>(expert) |
                                    (output_tile << 8U) |
                                    (m_tile << 15U));
      }
    }
    for (std::uint32_t output_tile = 0; output_tile < 128U; ++output_tile) {
      for (std::uint32_t m_tile = 0; m_tile < m_tiles; ++m_tile) {
        result.down_tasks.push_back(static_cast<std::uint32_t>(expert) |
                                    (output_tile << 8U) |
                                    (m_tile << 15U));
      }
    }
    const auto native_m_tiles = (counts[expert] + 31U) / 32U;
    for (std::uint32_t output_tile = 0; output_tile < 16U; ++output_tile) {
      for (std::uint32_t m_tile = 0; m_tile < native_m_tiles; ++m_tile) {
        result.native_gate_tasks.push_back(
            static_cast<std::uint32_t>(expert) | (output_tile << 8U) |
            (m_tile << 15U));
      }
    }
    for (std::uint32_t output_tile = 0; output_tile < 32U; ++output_tile) {
      for (std::uint32_t m_tile = 0; m_tile < native_m_tiles; ++m_tile) {
        result.native_down_tasks.push_back(
            static_cast<std::uint32_t>(expert) | (output_tile << 8U) |
            (m_tile << 15U));
      }
    }
  }
  if (cumulative != kAssignmentCount) {
    throw std::runtime_error("reference assignment count mismatch");
  }
  for (std::size_t source = 0; source < kAssignmentCount; ++source) {
    const auto expert = static_cast<std::size_t>(topk[source]);
    const auto row = cursors[expert]++;
    result.token_map[row] = static_cast<std::int32_t>(source / kTopK);
    result.inverse_map[source] = static_cast<std::int32_t>(row);
    result.router_weights[row] = router_weights[source];
  }
  return result;
}

struct OpenClState {
  cl_context context = nullptr;
  cl_command_queue queue = nullptr;
  cl_program program = nullptr;
  std::vector<cl_kernel> kernels;
  std::vector<cl_mem> buffers;

  ~OpenClState() {
    for (auto buffer : buffers) clReleaseMemObject(buffer);
    for (auto kernel : kernels) clReleaseKernel(kernel);
    if (program) clReleaseProgram(program);
    if (queue) clReleaseCommandQueue(queue);
    if (context) clReleaseContext(context);
  }
};

struct DeviceSelection {
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
      DeviceSelection selected{device, name.data()};
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

template <typename Value, std::size_t Count>
std::size_t MismatchCount(const std::array<Value, Count>& expected,
                          const std::vector<Value>& observed) {
  if (observed.size() != Count) return Count;
  std::size_t mismatches = 0;
  for (std::size_t index = 0; index < Count; ++index) {
    mismatches += std::memcmp(&expected[index], &observed[index],
                              sizeof(Value)) != 0;
  }
  return mismatches;
}

template <typename Value>
std::size_t MismatchCount(const std::vector<Value>& expected,
                          const std::vector<Value>& observed,
                          std::size_t count) {
  if (expected.size() < count || observed.size() < count) return count;
  std::size_t mismatches = 0;
  for (std::size_t index = 0; index < count; ++index) {
    mismatches += expected[index] != observed[index];
  }
  return mismatches;
}

double Median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

std::size_t SemanticMismatchCount(
    const std::vector<std::int32_t>& topk,
    const std::vector<float>& router,
    const std::vector<std::int32_t>& offsets,
    const std::vector<std::int32_t>& token_map,
    const std::vector<std::int32_t>& inverse_map,
    const std::vector<float>& compact_router) {
  std::vector<std::uint8_t> used(kAssignmentCount, 0);
  std::size_t mismatches = 0;
  for (std::size_t source = 0; source < kAssignmentCount; ++source) {
    const auto row_value = inverse_map[source];
    if (row_value < 0 ||
        row_value >= static_cast<std::int32_t>(kAssignmentCount)) {
      ++mismatches;
      continue;
    }
    const auto row = static_cast<std::size_t>(row_value);
    mismatches += used[row] != 0;
    used[row] = 1;
    const auto expert_it = std::upper_bound(
        offsets.begin(), offsets.end(), static_cast<std::int32_t>(row));
    if (expert_it == offsets.end()) {
      ++mismatches;
      continue;
    }
    const auto expert = static_cast<std::int32_t>(
        std::distance(offsets.begin(), expert_it));
    mismatches += expert != topk[source];
    mismatches += token_map[row] !=
        static_cast<std::int32_t>(source / kTopK);
    mismatches += std::memcmp(&compact_router[row], &router[source],
                              sizeof(float)) != 0;
  }
  mismatches += std::count(used.begin(), used.end(), 0);
  return mismatches;
}

void PrintSamples(const std::vector<double>& values) {
  std::cout << '[';
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0) std::cout << ',';
    std::cout << values[index];
  }
  std::cout << ']';
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 4 || argc > 6) {
      throw std::invalid_argument(
          "usage: grouped-prefill-device-schedule-probe KERNEL TOPK ROUTER "
          "[WARMUP] [REPEAT]");
    }
    const int warmup = argc >= 5 ? std::stoi(argv[4]) : 5;
    const int repeat = argc >= 6 ? std::stoi(argv[5]) : 21;
    if (warmup < 0 || repeat < 3) {
      throw std::invalid_argument("warmup/repeat are invalid");
    }
    const auto topk = ReadVector<std::int32_t>(argv[2], kAssignmentCount);
    const auto router = ReadVector<float>(argv[3], kAssignmentCount);
    const auto reference = BuildReference(topk, router);

    std::vector<double> cpu_samples;
    cpu_samples.reserve(repeat);
    volatile std::uint64_t cpu_checksum = 0;
    for (int iteration = 0; iteration < repeat; ++iteration) {
      const auto begin = std::chrono::steady_clock::now();
      const auto row = BuildReference(topk, router);
      const auto end = std::chrono::steady_clock::now();
      cpu_checksum += row.gate_tasks.size() + row.down_tasks.size();
      cpu_samples.push_back(std::chrono::duration<double, std::micro>(
                                end - begin)
                                .count());
    }

    const auto selected = SelectDevice();
    OpenClState state;
    cl_int status = CL_SUCCESS;
    state.context = clCreateContext(nullptr, 1, &selected.device, nullptr,
                                    nullptr, &status);
    Check(status, "clCreateContext");
    state.queue = clCreateCommandQueue(state.context, selected.device,
                                       CL_QUEUE_PROFILING_ENABLE, &status);
    Check(status, "clCreateCommandQueue");
    const auto source_text = ReadText(argv[1]);
    const char* source = source_text.c_str();
    const std::size_t source_size = source_text.size();
    state.program = clCreateProgramWithSource(state.context, 1, &source,
                                              &source_size, &status);
    Check(status, "clCreateProgramWithSource");
    status = clBuildProgram(state.program, 1, &selected.device, "", nullptr,
                            nullptr);
    if (status != CL_SUCCESS) {
      std::size_t log_size = 0;
      clGetProgramBuildInfo(state.program, selected.device,
                            CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
      std::string log(log_size, '\0');
      clGetProgramBuildInfo(state.program, selected.device,
                            CL_PROGRAM_BUILD_LOG, log.size(), log.data(),
                            nullptr);
      throw std::runtime_error("OpenCL build failed: " + log);
    }
    const auto CreateKernel = [&](const char* name) {
      auto kernel = clCreateKernel(state.program, name, &status);
      Check(status, "clCreateKernel");
      state.kernels.push_back(kernel);
      return kernel;
    };
    const auto reset_kernel = CreateKernel(
        "iq36_grouped_schedule_reset_1024");
    const auto count_kernel = CreateKernel(
        "iq36_grouped_schedule_count_1024");
    const auto prefix_kernel = CreateKernel(
        "iq36_grouped_schedule_prefix_1024");
    const auto scatter_kernel = CreateKernel(
        "iq36_grouped_schedule_scatter_1024");
    const auto tasks_kernel = CreateKernel(
        "iq36_grouped_schedule_tasks_1024");
    const auto native_tasks_kernel = CreateKernel(
        "iq36_grouped_schedule_native_tasks_1024");

    auto topk_buffer = CreateInput(state, topk);
    auto router_buffer = CreateInput(state, router);
    auto offsets_buffer = CreateBuffer(
        state, kExpertCount * sizeof(std::int32_t), CL_MEM_WRITE_ONLY);
    auto token_map_buffer = CreateBuffer(
        state, kAssignmentCount * sizeof(std::int32_t), CL_MEM_WRITE_ONLY);
    auto inverse_map_buffer = CreateBuffer(
        state, kAssignmentCount * sizeof(std::int32_t), CL_MEM_WRITE_ONLY);
    auto compact_router_buffer = CreateBuffer(
        state, kAssignmentCount * sizeof(float), CL_MEM_WRITE_ONLY);
    auto gate_tasks_buffer = CreateBuffer(
        state, kGateTaskCapacity * sizeof(std::uint32_t), CL_MEM_WRITE_ONLY);
    auto down_tasks_buffer = CreateBuffer(
        state, kDownTaskCapacity * sizeof(std::uint32_t), CL_MEM_WRITE_ONLY);
    auto native_gate_tasks_buffer = CreateBuffer(
        state, kNativeGateTaskCapacity * sizeof(std::uint32_t),
        CL_MEM_WRITE_ONLY);
    auto native_down_tasks_buffer = CreateBuffer(
        state, kNativeDownTaskCapacity * sizeof(std::uint32_t),
        CL_MEM_WRITE_ONLY);
    auto metadata_buffer = CreateBuffer(
        state, 7 * sizeof(std::uint32_t), CL_MEM_READ_WRITE);
    auto counts_buffer = CreateBuffer(
        state, kExpertCount * sizeof(std::uint32_t), CL_MEM_READ_WRITE);
    auto begins_buffer = CreateBuffer(
        state, kExpertCount * sizeof(std::uint32_t), CL_MEM_READ_WRITE);
    auto cursors_buffer = CreateBuffer(
        state, kExpertCount * sizeof(std::uint32_t), CL_MEM_READ_WRITE);
    auto gate_bases_buffer = CreateBuffer(
        state, kExpertCount * sizeof(std::uint32_t), CL_MEM_READ_WRITE);
    auto down_bases_buffer = CreateBuffer(
        state, kExpertCount * sizeof(std::uint32_t), CL_MEM_READ_WRITE);
    auto native_gate_bases_buffer = CreateBuffer(
        state, kExpertCount * sizeof(std::uint32_t), CL_MEM_READ_WRITE);
    auto native_down_bases_buffer = CreateBuffer(
        state, kExpertCount * sizeof(std::uint32_t), CL_MEM_READ_WRITE);
    auto partial_counts_buffer = CreateBuffer(
        state, 32 * kExpertCount * sizeof(std::uint32_t), CL_MEM_READ_WRITE);
    auto chunk_bases_buffer = CreateBuffer(
        state, 32 * kExpertCount * sizeof(std::uint32_t), CL_MEM_READ_WRITE);
    const auto SetArguments = [&](cl_kernel kernel,
                                  const std::vector<cl_mem>& arguments) {
      for (cl_uint index = 0; index < arguments.size(); ++index) {
        Check(clSetKernelArg(kernel, index, sizeof(cl_mem), &arguments[index]),
              "clSetKernelArg");
      }
    };
    SetArguments(reset_kernel,
                 {counts_buffer, cursors_buffer, metadata_buffer});
    SetArguments(count_kernel,
                 {topk_buffer, partial_counts_buffer, metadata_buffer});
    SetArguments(prefix_kernel,
                 {partial_counts_buffer, counts_buffer, begins_buffer,
                  cursors_buffer, offsets_buffer, gate_bases_buffer,
                  down_bases_buffer, native_gate_bases_buffer,
                  native_down_bases_buffer, chunk_bases_buffer,
                  metadata_buffer});
    SetArguments(scatter_kernel,
                 {topk_buffer, router_buffer, chunk_bases_buffer,
                  token_map_buffer, inverse_map_buffer,
                  compact_router_buffer});
    SetArguments(tasks_kernel,
                 {counts_buffer, cursors_buffer, offsets_buffer,
                  gate_bases_buffer, down_bases_buffer, gate_tasks_buffer,
                  down_tasks_buffer, metadata_buffer});
    SetArguments(native_tasks_kernel,
                 {counts_buffer, cursors_buffer, offsets_buffer,
                  native_gate_bases_buffer, native_down_bases_buffer,
                  native_gate_tasks_buffer, native_down_tasks_buffer,
                  metadata_buffer});

    constexpr std::size_t reset_global = 256;
    constexpr std::size_t count_global = kAssignmentCount;
    constexpr std::size_t prefix_global = 256;
    constexpr std::size_t scatter_global = kAssignmentCount;
    constexpr std::size_t tasks_global = kExpertCount * 256;
    constexpr std::size_t local = 256;
    const auto EnqueueSchedule = [&](std::array<cl_event, 6>* events) {
      auto Event = [&](std::size_t index) -> cl_event* {
        return events == nullptr ? nullptr : &(*events)[index];
      };
      Check(clEnqueueNDRangeKernel(state.queue, reset_kernel, 1, nullptr,
                                   &reset_global, &local, 0, nullptr,
                                   Event(0)),
            "enqueue schedule reset");
      Check(clEnqueueNDRangeKernel(state.queue, count_kernel, 1, nullptr,
                                   &count_global, &local, 0, nullptr, Event(1)),
            "enqueue schedule count");
      Check(clEnqueueNDRangeKernel(state.queue, prefix_kernel, 1, nullptr,
                                   &prefix_global, &local, 0, nullptr,
                                   Event(2)),
            "enqueue schedule prefix");
      Check(clEnqueueNDRangeKernel(state.queue, scatter_kernel, 1, nullptr,
                                   &scatter_global, &local, 0, nullptr,
                                   Event(3)),
            "enqueue schedule scatter");
      Check(clEnqueueNDRangeKernel(state.queue, tasks_kernel, 1, nullptr,
                                   &tasks_global, &local, 0, nullptr,
                                   Event(4)),
            "enqueue schedule tasks");
      Check(clEnqueueNDRangeKernel(state.queue, native_tasks_kernel, 1, nullptr,
                                   &tasks_global, &local, 0, nullptr,
                                   Event(5)),
            "enqueue native schedule tasks");
    };
    for (int iteration = 0; iteration < warmup; ++iteration) {
      EnqueueSchedule(nullptr);
    }
    Check(clFinish(state.queue), "finish warmup");
    std::vector<double> device_samples;
    device_samples.reserve(repeat);
    std::array<double, 6> stage_sums{};
    for (int iteration = 0; iteration < repeat; ++iteration) {
      std::array<cl_event, 6> events{};
      EnqueueSchedule(&events);
      Check(clWaitForEvents(1, &events.back()), "wait measured schedule");
      cl_ulong start = 0;
      cl_ulong end = 0;
      Check(clGetEventProfilingInfo(events.front(), CL_PROFILING_COMMAND_START,
                                    sizeof(start), &start, nullptr),
            "read profile start");
      Check(clGetEventProfilingInfo(events.back(), CL_PROFILING_COMMAND_END,
                                    sizeof(end), &end, nullptr),
            "read profile end");
      for (std::size_t stage = 0; stage < events.size(); ++stage) {
        cl_ulong stage_start = 0;
        cl_ulong stage_end = 0;
        Check(clGetEventProfilingInfo(events[stage],
                                      CL_PROFILING_COMMAND_START,
                                      sizeof(stage_start), &stage_start,
                                      nullptr),
              "read stage profile start");
        Check(clGetEventProfilingInfo(events[stage],
                                      CL_PROFILING_COMMAND_END,
                                      sizeof(stage_end), &stage_end, nullptr),
              "read stage profile end");
        stage_sums[stage] +=
            static_cast<double>(stage_end - stage_start) / 1000.0;
        clReleaseEvent(events[stage]);
      }
      device_samples.push_back(static_cast<double>(end - start) / 1000.0);
    }

    std::vector<std::int32_t> offsets(kExpertCount);
    std::vector<std::int32_t> token_map(kAssignmentCount);
    std::vector<std::int32_t> inverse_map(kAssignmentCount);
    std::vector<float> compact_router(kAssignmentCount);
    std::vector<std::uint32_t> gate_tasks(kGateTaskCapacity);
    std::vector<std::uint32_t> down_tasks(kDownTaskCapacity);
    std::vector<std::uint32_t> native_gate_tasks(kNativeGateTaskCapacity);
    std::vector<std::uint32_t> native_down_tasks(kNativeDownTaskCapacity);
    std::array<std::uint32_t, 7> metadata{};
    const auto Read = [&](cl_mem buffer, void* destination,
                          std::size_t bytes, const char* label) {
      Check(clEnqueueReadBuffer(state.queue, buffer, CL_TRUE, 0, bytes,
                                destination, 0, nullptr, nullptr),
            label);
    };
    Read(offsets_buffer, offsets.data(), offsets.size() * sizeof(offsets[0]),
         "read offsets");
    Read(token_map_buffer, token_map.data(),
         token_map.size() * sizeof(token_map[0]), "read token map");
    Read(inverse_map_buffer, inverse_map.data(),
         inverse_map.size() * sizeof(inverse_map[0]), "read inverse map");
    Read(compact_router_buffer, compact_router.data(),
         compact_router.size() * sizeof(compact_router[0]),
         "read compact router");
    Read(metadata_buffer, metadata.data(), metadata.size() * sizeof(metadata[0]),
         "read metadata");
    if (metadata[0] > gate_tasks.size() || metadata[1] > down_tasks.size() ||
        metadata[5] > native_gate_tasks.size() ||
        metadata[6] > native_down_tasks.size()) {
      throw std::runtime_error("device task count exceeds capacity");
    }
    Read(gate_tasks_buffer, gate_tasks.data(),
         metadata[0] * sizeof(gate_tasks[0]), "read gate tasks");
    Read(down_tasks_buffer, down_tasks.data(),
         metadata[1] * sizeof(down_tasks[0]), "read down tasks");
    Read(native_gate_tasks_buffer, native_gate_tasks.data(),
         metadata[5] * sizeof(native_gate_tasks[0]),
         "read native gate tasks");
    Read(native_down_tasks_buffer, native_down_tasks.data(),
         metadata[6] * sizeof(native_down_tasks[0]),
         "read native down tasks");

    const auto offsets_mismatch =
        MismatchCount(reference.offsets, offsets);
    const auto token_map_mismatch =
        MismatchCount(reference.token_map, token_map);
    const auto inverse_map_mismatch =
        MismatchCount(reference.inverse_map, inverse_map);
    const auto router_mismatch =
        MismatchCount(reference.router_weights, compact_router);
    const auto gate_mismatch =
        MismatchCount(reference.gate_tasks, gate_tasks, metadata[0]);
    const auto down_mismatch =
        MismatchCount(reference.down_tasks, down_tasks, metadata[1]);
    const auto native_gate_mismatch = MismatchCount(
        reference.native_gate_tasks, native_gate_tasks, metadata[5]);
    const auto native_down_mismatch = MismatchCount(
        reference.native_down_tasks, native_down_tasks, metadata[6]);
    const auto semantic_mismatch = SemanticMismatchCount(
        topk, router, offsets, token_map, inverse_map, compact_router);
    const bool metadata_match =
        metadata[0] == reference.gate_tasks.size() &&
        metadata[1] == reference.down_tasks.size() &&
        metadata[2] == reference.active_experts &&
        metadata[3] == reference.max_group && metadata[4] == 0 &&
        metadata[5] == reference.native_gate_tasks.size() &&
        metadata[6] == reference.native_down_tasks.size();
    const bool exact = metadata_match && offsets_mismatch == 0 &&
        semantic_mismatch == 0 && gate_mismatch == 0 && down_mismatch == 0 &&
        native_gate_mismatch == 0 && native_down_mismatch == 0;
    const double device_median = Median(device_samples);
    const double cpu_median = Median(cpu_samples);
    const bool performance = device_median <= kDeviceScheduleCapUs;
    const bool passed = exact && performance;

    std::cout << std::boolalpha << std::setprecision(12) << '{'
              << "\"active_experts\":" << metadata[2] << ','
              << "\"cpu_checksum\":" << cpu_checksum << ','
              << "\"cpu_median_us\":" << cpu_median << ','
              << "\"device_median_us\":" << device_median << ','
              << "\"device_name\":\"" << selected.name << "\","
              << "\"device_schedule_cap_us\":" << kDeviceScheduleCapUs
              << ',' << "\"down_task_count\":" << metadata[1] << ','
              << "\"down_task_mismatch_count\":" << down_mismatch << ','
              << "\"exact_schedule_match\":" << exact << ','
              << "\"gateup_task_count\":" << metadata[0] << ','
              << "\"gateup_task_mismatch_count\":" << gate_mismatch << ','
              << "\"inverse_map_mismatch_count\":"
              << inverse_map_mismatch << ','
              << "\"kernel_samples_us\":";
    PrintSamples(device_samples);
    std::cout << ',' << "\"max_group_size\":" << metadata[3] << ','
              << "\"metadata_error_bitmap\":" << metadata[4] << ','
              << "\"metadata_match\":" << metadata_match << ','
              << "\"native_down_task_count\":" << metadata[6] << ','
              << "\"native_down_task_mismatch_count\":"
              << native_down_mismatch << ','
              << "\"native_gateup_task_count\":" << metadata[5] << ','
              << "\"native_gateup_task_mismatch_count\":"
              << native_gate_mismatch << ','
              << "\"offset_mismatch_count\":" << offsets_mismatch << ','
              << "\"performance_pass\":" << performance << ','
              << "\"required_checks_passed\":" << passed << ','
              << "\"router_weight_mismatch_count\":" << router_mismatch
              << ',' << "\"semantic_mismatch_count\":"
              << semantic_mismatch << ','
              << "\"speedup_vs_cpu_schedule\":"
              << cpu_median / device_median << ','
              << "\"stage_mean_us\":[";
    for (std::size_t stage = 0; stage < stage_sums.size(); ++stage) {
      if (stage != 0) std::cout << ',';
      std::cout << stage_sums[stage] / repeat;
    }
    std::cout << "],"
              << "\"token_map_mismatch_count\":" << token_map_mismatch
              << '}' << std::endl;
    return passed ? 0 : 2;
  } catch (const std::exception& exception) {
    std::cerr << exception.what() << '\n';
    return 4;
  }
}
