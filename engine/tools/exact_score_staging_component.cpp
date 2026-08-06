#include <CL/cl.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef CL_KERNEL_REGISTER_COUNT_INTEL
#define CL_KERNEL_REGISTER_COUNT_INTEL 0x425B
#endif

#ifndef CL_KERNEL_SPILL_MEM_SIZE_INTEL
#define CL_KERNEL_SPILL_MEM_SIZE_INTEL 0x4109
#endif

namespace {

constexpr std::size_t kContextTokens = 131072;
constexpr std::size_t kHeadDim = 256;
constexpr std::size_t kQHeads = 16;
constexpr std::size_t kKvHeads = 2;
constexpr std::size_t kGqaGroup = 8;
constexpr std::size_t kKqColumns = 16;
constexpr std::size_t kKeyBlock = 256;
constexpr std::size_t kBlocksPerKv = kContextTokens / kKeyBlock;
constexpr std::size_t kSubgroupItems = 16;
constexpr std::size_t kTrafficSubgroups = 48;
constexpr std::size_t kTrafficWorkgroupItems =
    kSubgroupItems * kTrafficSubgroups;
constexpr std::size_t kTrafficChecksumVectorsPerHead =
    2 * kTrafficWorkgroupItems;
constexpr std::size_t kTrafficChecksumWords =
    kKvHeads * kTrafficChecksumVectorsPerHead * 8;
constexpr int kWarmups = 3;
constexpr int kTrafficWarmups = 12;
constexpr int kSamples = 20;

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

std::vector<std::uint8_t> ReadBytes(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  Require(static_cast<bool>(input), "could not open program binary");
  input.seekg(0, std::ios::end);
  const auto size = input.tellg();
  Require(size > 0, "program binary is empty");
  input.seekg(0, std::ios::beg);
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
  input.read(reinterpret_cast<char*>(bytes.data()),
             static_cast<std::streamsize>(bytes.size()));
  Require(static_cast<bool>(input), "could not read program binary");
  return bytes;
}

std::string DeviceString(cl_device_id device, cl_device_info key) {
  std::size_t bytes = 0;
  Check(clGetDeviceInfo(device, key, 0, nullptr, &bytes),
        "clGetDeviceInfo(size)");
  std::string value(bytes, '\0');
  Check(clGetDeviceInfo(device, key, bytes, value.data(), nullptr),
        "clGetDeviceInfo(value)");
  while (!value.empty() && value.back() == '\0') value.pop_back();
  return value;
}

cl_device_id SelectGpu() {
  cl_uint platform_count = 0;
  Check(clGetPlatformIDs(0, nullptr, &platform_count),
        "clGetPlatformIDs(count)");
  std::vector<cl_platform_id> platforms(platform_count);
  Check(clGetPlatformIDs(platform_count, platforms.data(), nullptr),
        "clGetPlatformIDs(values)");
  for (cl_platform_id platform : platforms) {
    cl_uint device_count = 0;
    const cl_int status = clGetDeviceIDs(
        platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &device_count);
    if (status == CL_DEVICE_NOT_FOUND || device_count == 0) continue;
    Check(status, "clGetDeviceIDs(count)");
    std::vector<cl_device_id> devices(device_count);
    Check(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, device_count,
                         devices.data(), nullptr),
          "clGetDeviceIDs(values)");
    for (cl_device_id device : devices) {
      if (DeviceString(device, CL_DEVICE_NAME).find("B390") !=
          std::string::npos) {
        return device;
      }
    }
  }
  Fail("Intel Arc B390 OpenCL device not found");
}

std::string ProgramLog(cl_program program, cl_device_id device) {
  std::size_t bytes = 0;
  Check(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                              0, nullptr, &bytes),
        "clGetProgramBuildInfo(size)");
  std::string log(bytes, '\0');
  if (bytes != 0) {
    Check(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                                bytes, log.data(), nullptr),
          "clGetProgramBuildInfo(value)");
  }
  return log;
}

cl_program LoadProgram(
    cl_context context, cl_device_id device, const std::string& path) {
  const std::vector<std::uint8_t> binary = ReadBytes(path);
  const unsigned char* binary_data = binary.data();
  const std::size_t binary_size = binary.size();
  cl_int binary_status = CL_SUCCESS;
  cl_int status = CL_SUCCESS;
  cl_program program = clCreateProgramWithBinary(
      context, 1, &device, &binary_size, &binary_data,
      &binary_status, &status);
  Check(status, "clCreateProgramWithBinary");
  Check(binary_status, "clCreateProgramWithBinary(binary status)");
  status = clBuildProgram(program, 1, &device, "", nullptr, nullptr);
  if (status != CL_SUCCESS) {
    const std::string log = ProgramLog(program, device);
    clReleaseProgram(program);
    Fail("program binary build failed: " + log);
  }
  return program;
}

cl_mem Buffer(cl_context context, std::size_t bytes) {
  cl_int status = CL_SUCCESS;
  cl_mem result = clCreateBuffer(
      context, CL_MEM_READ_WRITE, bytes, nullptr, &status);
  Check(status, "clCreateBuffer");
  return result;
}

template <typename T>
void Arg(cl_kernel kernel, cl_uint index, const T& value) {
  Check(clSetKernelArg(kernel, index, sizeof(T), &value), "clSetKernelArg");
}

void Enqueue1d(cl_command_queue queue, cl_kernel kernel,
               std::size_t global, std::size_t local) {
  Check(clEnqueueNDRangeKernel(
      queue, kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr),
      "clEnqueueNDRangeKernel(1d)");
}

cl_event Enqueue2d(cl_command_queue queue, cl_kernel kernel,
                   const std::array<std::size_t, 2>& global,
                   const std::array<std::size_t, 2>& local) {
  cl_event event = nullptr;
  Check(clEnqueueNDRangeKernel(
      queue, kernel, 2, nullptr, global.data(), local.data(),
      0, nullptr, &event), "clEnqueueNDRangeKernel(2d)");
  return event;
}

struct EventRange {
  cl_ulong start = 0;
  cl_ulong end = 0;
};

EventRange Range(cl_event event) {
  EventRange result;
  Check(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START,
                                sizeof(result.start), &result.start, nullptr),
        "clGetEventProfilingInfo(start)");
  Check(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,
                                sizeof(result.end), &result.end, nullptr),
        "clGetEventProfilingInfo(end)");
  return result;
}

double Milliseconds(cl_ulong nanoseconds) {
  return static_cast<double>(nanoseconds) / 1.0e6;
}

struct FusedTiming {
  double total_ms = 0.0;
};

struct StagedTiming {
  double kq_ms = 0.0;
  double owner_ms = 0.0;
  double total_ms = 0.0;
};

struct KernelResources {
  cl_uint register_count = 0;
  cl_ulong spill_memory_bytes = 0;
  cl_ulong local_memory_bytes = 0;
  std::size_t maximum_workgroup_items = 0;
  std::size_t preferred_workgroup_multiple = 0;
};

KernelResources Resources(cl_kernel kernel, cl_device_id device) {
  KernelResources result;
  Check(clGetKernelWorkGroupInfo(
            kernel, device, CL_KERNEL_REGISTER_COUNT_INTEL,
            sizeof(result.register_count), &result.register_count, nullptr),
        "clGetKernelWorkGroupInfo(register count)");
  Check(clGetKernelWorkGroupInfo(
            kernel, device, CL_KERNEL_SPILL_MEM_SIZE_INTEL,
            sizeof(result.spill_memory_bytes), &result.spill_memory_bytes,
            nullptr),
        "clGetKernelWorkGroupInfo(spill memory)");
  Check(clGetKernelWorkGroupInfo(
            kernel, device, CL_KERNEL_LOCAL_MEM_SIZE,
            sizeof(result.local_memory_bytes), &result.local_memory_bytes,
            nullptr),
        "clGetKernelWorkGroupInfo(local memory)");
  Check(clGetKernelWorkGroupInfo(
            kernel, device, CL_KERNEL_WORK_GROUP_SIZE,
            sizeof(result.maximum_workgroup_items),
            &result.maximum_workgroup_items, nullptr),
        "clGetKernelWorkGroupInfo(maximum workgroup size)");
  Check(clGetKernelWorkGroupInfo(
            kernel, device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
            sizeof(result.preferred_workgroup_multiple),
            &result.preferred_workgroup_multiple, nullptr),
        "clGetKernelWorkGroupInfo(preferred workgroup multiple)");
  return result;
}

void PrintResources(const KernelResources& resources) {
  std::cout << "{\"register_count\":" << resources.register_count
            << ",\"spill_memory_bytes\":" << resources.spill_memory_bytes
            << ",\"local_memory_bytes\":" << resources.local_memory_bytes
            << ",\"maximum_workgroup_items\":"
            << resources.maximum_workgroup_items
            << ",\"preferred_workgroup_multiple\":"
            << resources.preferred_workgroup_multiple << '}';
}

FusedTiming RunFused(cl_command_queue queue, cl_kernel kernel) {
  const std::array<std::size_t, 2> global = {16, 16 * kKvHeads};
  const std::array<std::size_t, 2> local = {16, 16};
  cl_event event = Enqueue2d(queue, kernel, global, local);
  Check(clWaitForEvents(1, &event), "clWaitForEvents(fused)");
  const EventRange range = Range(event);
  clReleaseEvent(event);
  return {Milliseconds(range.end - range.start)};
}

StagedTiming RunStaged(cl_command_queue queue, cl_kernel kq,
                       cl_kernel owner) {
  const std::array<std::size_t, 2> kq_global = {
      16, 16 * kKvHeads * kBlocksPerKv};
  const std::array<std::size_t, 2> owner_global = {16, 16 * kKvHeads};
  const std::array<std::size_t, 2> local = {16, 16};
  cl_event kq_event = Enqueue2d(queue, kq, kq_global, local);
  cl_event owner_event = Enqueue2d(queue, owner, owner_global, local);
  Check(clWaitForEvents(1, &owner_event), "clWaitForEvents(staged)");
  const EventRange kq_range = Range(kq_event);
  const EventRange owner_range = Range(owner_event);
  clReleaseEvent(kq_event);
  clReleaseEvent(owner_event);
  StagedTiming result;
  result.kq_ms = Milliseconds(kq_range.end - kq_range.start);
  result.owner_ms = Milliseconds(owner_range.end - owner_range.start);
  result.total_ms = Milliseconds(owner_range.end - kq_range.start);
  return result;
}

struct ThreeStageTiming {
  double kq_ms = 0.0;
  double softmax_ms = 0.0;
  double vs_ms = 0.0;
  double total_ms = 0.0;
};

double RunDualControl(cl_command_queue queue, cl_kernel kernel) {
  const std::array<std::size_t, 2> global = {16, 32 * kKvHeads};
  const std::array<std::size_t, 2> local = {16, 32};
  cl_event event = Enqueue2d(queue, kernel, global, local);
  Check(clWaitForEvents(1, &event), "clWaitForEvents(dual control)");
  const EventRange range = Range(event);
  clReleaseEvent(event);
  return Milliseconds(range.end - range.start);
}

double RunTripleCohort(cl_command_queue queue, cl_kernel kernel) {
  const std::array<std::size_t, 2> global = {16, 48 * kKvHeads};
  const std::array<std::size_t, 2> local = {16, 48};
  cl_event event = Enqueue2d(queue, kernel, global, local);
  Check(clWaitForEvents(1, &event), "clWaitForEvents(triple cohort)");
  const EventRange range = Range(event);
  clReleaseEvent(event);
  return Milliseconds(range.end - range.start);
}

double RunDenseTraffic(cl_command_queue queue, cl_kernel kernel) {
  const std::array<std::size_t, 2> global = {
      kSubgroupItems, kTrafficSubgroups * kKvHeads};
  const std::array<std::size_t, 2> local = {
      kSubgroupItems, kTrafficSubgroups};
  cl_event event = Enqueue2d(queue, kernel, global, local);
  Check(clWaitForEvents(1, &event), "clWaitForEvents(dense traffic)");
  const EventRange range = Range(event);
  clReleaseEvent(event);
  return Milliseconds(range.end - range.start);
}

ThreeStageTiming RunThreeStages(
    cl_command_queue queue, cl_kernel kq, cl_kernel softmax, cl_kernel vs) {
  const std::array<std::size_t, 2> kq_global = {
      16, 16 * kKvHeads * kBlocksPerKv};
  const std::array<std::size_t, 2> owner_global = {16, 16 * kKvHeads};
  const std::array<std::size_t, 2> local = {16, 16};
  cl_event kq_event = Enqueue2d(queue, kq, kq_global, local);
  cl_event softmax_event = Enqueue2d(
      queue, softmax, owner_global, local);
  cl_event vs_event = Enqueue2d(queue, vs, owner_global, local);
  Check(clWaitForEvents(1, &vs_event), "clWaitForEvents(three stage)");
  const EventRange kq_range = Range(kq_event);
  const EventRange softmax_range = Range(softmax_event);
  const EventRange vs_range = Range(vs_event);
  clReleaseEvent(kq_event);
  clReleaseEvent(softmax_event);
  clReleaseEvent(vs_event);
  ThreeStageTiming result;
  result.kq_ms = Milliseconds(kq_range.end - kq_range.start);
  result.softmax_ms =
      Milliseconds(softmax_range.end - softmax_range.start);
  result.vs_ms = Milliseconds(vs_range.end - vs_range.start);
  result.total_ms = Milliseconds(vs_range.end - kq_range.start);
  return result;
}

template <typename T>
std::size_t Mismatches(const std::vector<T>& left,
                       const std::vector<T>& right) {
  Require(left.size() == right.size(), "comparison shape mismatch");
  std::size_t mismatches = 0;
  for (std::size_t index = 0; index < left.size(); ++index) {
    mismatches += left[index] != right[index];
  }
  return mismatches;
}

std::uint64_t HashWords(const std::vector<std::uint32_t>& words) {
  std::uint64_t hash = 1469598103934665603ULL;
  for (std::uint32_t word : words) {
    for (unsigned int shift = 0; shift < 32; shift += 8) {
      hash ^= (word >> shift) & 0xffU;
      hash *= 1099511628211ULL;
    }
  }
  return hash;
}

int RunDenseTrafficCeiling(char** argv) {
  const cl_device_id device = SelectGpu();
  cl_int status = CL_SUCCESS;
  cl_context context = clCreateContext(
      nullptr, 1, &device, nullptr, nullptr, &status);
  Check(status, "clCreateContext");
  const cl_queue_properties properties[] = {
      CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
  cl_command_queue queue = clCreateCommandQueueWithProperties(
      context, device, properties, &status);
  Check(status, "clCreateCommandQueueWithProperties");

  cl_program capture_program = LoadProgram(context, device, argv[1]);
  cl_program traffic_program = LoadProgram(context, device, argv[2]);
  auto Kernel = [&](cl_program program, const char* name) {
    cl_kernel kernel = clCreateKernel(program, name, &status);
    Check(status, std::string("clCreateKernel(") + name + ")");
    return kernel;
  };
  cl_kernel init_history =
      Kernel(capture_program, "iq36_exact_score_init_history");
  cl_kernel traffic = Kernel(
      traffic_program, "iq36_exact_attention_dense_traffic_ceiling");
  const KernelResources resources = Resources(traffic, device);

  const std::size_t history_values =
      kKvHeads * kContextTokens * kHeadDim;
  const std::size_t payload_bytes =
      2 * history_values * sizeof(std::uint16_t);
  cl_mem key = Buffer(
      context, history_values * sizeof(std::uint16_t));
  cl_mem value = Buffer(
      context, history_values * sizeof(std::uint16_t));
  cl_mem checksums = Buffer(
      context, kTrafficChecksumWords * sizeof(std::uint32_t));

  Arg(init_history, 0, key);
  Arg(init_history, 1, value);
  Enqueue1d(queue, init_history, history_values, 256);
  Check(clFinish(queue), "clFinish(initialization)");
  Arg(traffic, 0, key);
  Arg(traffic, 1, value);
  Arg(traffic, 2, checksums);

  (void)RunDenseTraffic(queue, traffic);
  std::vector<std::uint32_t> first_checksums(kTrafficChecksumWords);
  Check(clEnqueueReadBuffer(
            queue, checksums, CL_TRUE, 0,
            first_checksums.size() * sizeof(first_checksums[0]),
            first_checksums.data(), 0, nullptr, nullptr),
        "clEnqueueReadBuffer(first dense traffic checksums)");

  for (int warmup = 0; warmup < kTrafficWarmups; ++warmup) {
    (void)RunDenseTraffic(queue, traffic);
  }
  std::vector<double> samples;
  samples.reserve(kSamples);
  for (int sample = 0; sample < kSamples; ++sample) {
    samples.push_back(RunDenseTraffic(queue, traffic));
  }

  std::vector<std::uint32_t> final_checksums(kTrafficChecksumWords);
  Check(clEnqueueReadBuffer(
            queue, checksums, CL_TRUE, 0,
            final_checksums.size() * sizeof(final_checksums[0]),
            final_checksums.data(), 0, nullptr, nullptr),
        "clEnqueueReadBuffer(final dense traffic checksums)");
  const std::size_t checksum_mismatches =
      Mismatches(first_checksums, final_checksums);
  const std::size_t checksum_nonzero_words = static_cast<std::size_t>(
      std::count_if(
          final_checksums.begin(), final_checksums.end(),
          [](std::uint32_t word) { return word != 0; }));
  const std::uint64_t first_hash = HashWords(first_checksums);
  const std::uint64_t final_hash = HashWords(final_checksums);
  const bool numeric_pass =
      checksum_mismatches == 0 && checksum_nonzero_words != 0
      && first_hash == final_hash;

  std::cout << std::boolalpha << std::setprecision(12)
            << "{\"schema_version\":"
               "\"intel-qwen36-exact-attention-dense-traffic-ceiling-v1\""
            << ",\"algorithm\":"
               "\"two_workgroup_full_kv_uint8_dense_read\""
            << ",\"context_tokens\":" << kContextTokens
            << ",\"head_dim\":" << kHeadDim
            << ",\"kv_heads\":" << kKvHeads
            << ",\"traffic_workgroups\":" << kKvHeads
            << ",\"traffic_subgroups_per_workgroup\":"
            << kTrafficSubgroups
            << ",\"traffic_workitems_per_workgroup\":"
            << kTrafficWorkgroupItems
            << ",\"read_vector_bytes\":32"
            << ",\"read_vectors_per_head\":"
            << (kContextTokens * kHeadDim * sizeof(std::uint16_t)) / 32
            << ",\"mandatory_key_value_payload_bytes\":"
            << payload_bytes
            << ",\"checksum_output_bytes\":"
            << kTrafficChecksumWords * sizeof(std::uint32_t)
            << ",\"checksum_compared_words\":"
            << kTrafficChecksumWords
            << ",\"checksum_mismatch_count\":" << checksum_mismatches
            << ",\"checksum_nonzero_words\":" << checksum_nonzero_words
            << ",\"checksum_hash\":" << final_hash
            << ",\"numeric_pass\":" << numeric_pass
            << ",\"warmup_count\":" << kTrafficWarmups
            << ",\"sample_count\":" << samples.size()
            << ",\"schedule\":"
               "\"single_dense_traffic_ceiling_after_twelve_warmups\""
            << ",\"device\":\"" << DeviceString(device, CL_DEVICE_NAME)
            << "\",\"resources\":";
  PrintResources(resources);
  std::cout << ",\"samples\":[";
  for (std::size_t index = 0; index < samples.size(); ++index) {
    if (index != 0) std::cout << ',';
    const double latency_ms = samples[index];
    std::cout << "{\"sample\":" << index
              << ",\"latency_ms\":" << latency_ms
              << ",\"bandwidth_gb_s\":"
              << static_cast<double>(payload_bytes) /
                     (latency_ms * 1.0e6)
              << '}';
  }
  std::cout << "]}\n";

  for (cl_mem memory : {key, value, checksums}) {
    clReleaseMemObject(memory);
  }
  for (cl_kernel kernel : {init_history, traffic}) {
    clReleaseKernel(kernel);
  }
  for (cl_program program : {capture_program, traffic_program}) {
    clReleaseProgram(program);
  }
  clReleaseCommandQueue(queue);
  clReleaseContext(context);
  return numeric_pass && samples.size() == kSamples ? 0 : 2;
}

int RunPipelinedCohortComponent(
    char** argv, bool normalized_dual_cohort) {
  const cl_device_id device = SelectGpu();
  cl_int status = CL_SUCCESS;
  cl_context context = clCreateContext(
      nullptr, 1, &device, nullptr, nullptr, &status);
  Check(status, "clCreateContext");
  const cl_queue_properties properties[] = {
      CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
  cl_command_queue queue = clCreateCommandQueueWithProperties(
      context, device, properties, &status);
  Check(status, "clCreateCommandQueueWithProperties");

  cl_program capture_program = LoadProgram(context, device, argv[1]);
  cl_program dual_program = LoadProgram(context, device, argv[2]);
  cl_program candidate_program = LoadProgram(context, device, argv[3]);
  auto Kernel = [&](cl_program program, const char* name) {
    cl_kernel kernel = clCreateKernel(program, name, &status);
    Check(status, std::string("clCreateKernel(") + name + ")");
    return kernel;
  };
  cl_kernel init_query =
      Kernel(capture_program, "iq36_exact_score_init_query");
  cl_kernel init_history =
      Kernel(capture_program, "iq36_exact_score_init_history");
  cl_kernel dual =
      Kernel(dual_program, "iq36_exact_score_dual_cohort");
  cl_kernel candidate = Kernel(
      candidate_program,
      normalized_dual_cohort
          ? "iq36_exact_score_normalized_dual_cohort"
          : "iq36_exact_score_triple_cohort");
  const KernelResources dual_resources = Resources(dual, device);
  const KernelResources candidate_resources =
      Resources(candidate, device);

  const std::size_t query_values = kQHeads * kHeadDim;
  const std::size_t history_values =
      kKvHeads * kContextTokens * kHeadDim;
  cl_mem query = Buffer(context, query_values * sizeof(std::uint16_t));
  cl_mem key = Buffer(context, history_values * sizeof(std::uint16_t));
  cl_mem value = Buffer(context, history_values * sizeof(std::uint16_t));
  cl_mem dual_output =
      Buffer(context, query_values * sizeof(std::uint16_t));
  cl_mem candidate_output =
      Buffer(context, query_values * sizeof(std::uint16_t));

  Arg(init_query, 0, query);
  Arg(init_history, 0, key);
  Arg(init_history, 1, value);
  Enqueue1d(queue, init_query, query_values, 256);
  Enqueue1d(queue, init_history, history_values, 256);
  Check(clFinish(queue), "clFinish(initialization)");
  Arg(dual, 0, query);
  Arg(dual, 1, key);
  Arg(dual, 2, value);
  Arg(dual, 3, dual_output);
  Arg(candidate, 0, query);
  Arg(candidate, 1, key);
  Arg(candidate, 2, value);
  Arg(candidate, 3, candidate_output);
  auto RunCandidate = [&]() {
    return normalized_dual_cohort
        ? RunDualControl(queue, candidate)
        : RunTripleCohort(queue, candidate);
  };

  (void)RunDualControl(queue, dual);
  (void)RunCandidate();
  std::vector<std::uint16_t> dual_values(query_values);
  std::vector<std::uint16_t> candidate_values(query_values);
  Check(clEnqueueReadBuffer(
            queue, dual_output, CL_TRUE, 0,
            dual_values.size() * sizeof(dual_values[0]),
            dual_values.data(), 0, nullptr, nullptr),
        "clEnqueueReadBuffer(dual output)");
  Check(clEnqueueReadBuffer(
            queue, candidate_output, CL_TRUE, 0,
            candidate_values.size() * sizeof(candidate_values[0]),
            candidate_values.data(), 0, nullptr, nullptr),
        "clEnqueueReadBuffer(candidate output)");
  const std::size_t output_mismatches =
      Mismatches(dual_values, candidate_values);

  for (int warmup = 0; warmup < kWarmups; ++warmup) {
    if ((warmup & 1) == 0) {
      (void)RunDualControl(queue, dual);
      (void)RunCandidate();
    } else {
      (void)RunCandidate();
      (void)RunDualControl(queue, dual);
    }
  }
  struct Sample {
    const char* order = nullptr;
    double dual_ms = 0.0;
    double candidate_ms = 0.0;
  };
  std::vector<Sample> samples;
  samples.reserve(kSamples);
  for (int sample = 0; sample < kSamples; ++sample) {
    Sample row;
    if ((sample & 1) == 0) {
      row.order = normalized_dual_cohort
          ? "dual_normalized" : "dual_triple";
      row.dual_ms = RunDualControl(queue, dual);
      row.candidate_ms = RunCandidate();
    } else {
      row.order = normalized_dual_cohort
          ? "normalized_dual" : "triple_dual";
      row.candidate_ms = RunCandidate();
      row.dual_ms = RunDualControl(queue, dual);
    }
    samples.push_back(row);
  }

  const bool numeric_pass = output_mismatches == 0;
  const char* schema = normalized_dual_cohort
      ? "intel-qwen36-exact-attention-normalized-dual-cohort-component-v1"
      : "intel-qwen36-exact-attention-triple-cohort-component-v1";
  const char* algorithm = normalized_dual_cohort
      ? "generated_m256_n16_onchip_kq_softmax_producer_vs_consumer"
      : "generated_m256_n16_onchip_kq_softmax_vs_pipeline";
  const char* schedule = normalized_dual_cohort
      ? "interleaved_dual_normalized_normalized_dual"
      : "interleaved_dual_triple_triple_dual";
  const char* candidate_label =
      normalized_dual_cohort ? "normalized" : "triple";
  std::cout << std::boolalpha << std::setprecision(12)
            << "{\"schema_version\":\"" << schema << '"'
            << ",\"algorithm\":\"" << algorithm << '"'
            << ",\"context_tokens\":" << kContextTokens
            << ",\"head_dim\":" << kHeadDim
            << ",\"query_heads\":" << kQHeads
            << ",\"kv_heads\":" << kKvHeads
            << ",\"gqa_group\":" << kGqaGroup
            << ",\"key_block\":" << kKeyBlock
            << ",\"dual_subgroups\":32";
  if (normalized_dual_cohort) {
    std::cout << ",\"normalized_producer_subgroups\":16"
              << ",\"normalized_consumer_subgroups\":16"
              << ",\"normalized_total_subgroups\":32"
              << ",\"normalized_workgroup_items\":512";
  } else {
    std::cout << ",\"triple_kq_subgroups\":16"
              << ",\"triple_softmax_subgroups\":16"
              << ",\"triple_vs_subgroups\":16"
              << ",\"triple_total_subgroups\":48"
              << ",\"triple_workgroup_items\":768";
  }
  std::cout
            << ",\"mandatory_key_value_payload_bytes\":"
            << 2 * history_values * sizeof(std::uint16_t)
            << ",\"output_compared_values\":" << query_values
            << ",\"output_mismatch_count\":" << output_mismatches
            << ",\"numeric_pass\":" << numeric_pass
            << ",\"sample_count\":" << samples.size()
            << ",\"schedule\":\"" << schedule << '"'
            << ",\"device\":\"" << DeviceString(device, CL_DEVICE_NAME)
            << "\",\"resources\":{\"dual\":";
  PrintResources(dual_resources);
  std::cout << ",\"" << candidate_label << "\":";
  PrintResources(candidate_resources);
  std::cout << "},\"paired_samples\":[";
  for (std::size_t index = 0; index < samples.size(); ++index) {
    if (index != 0) std::cout << ',';
    const Sample& row = samples[index];
    std::cout << "{\"sample\":" << index
              << ",\"order\":\"" << row.order << "\""
              << ",\"dual_ms\":" << row.dual_ms
              << ",\"" << candidate_label << "_ms\":"
              << row.candidate_ms
              << ",\"delta_ms\":" << row.candidate_ms - row.dual_ms
              << ",\"speedup_ratio\":"
              << row.dual_ms / row.candidate_ms << '}';
  }
  std::cout << "]}\n";

  for (cl_mem memory : {
           query, key, value, dual_output, candidate_output}) {
    clReleaseMemObject(memory);
  }
  for (cl_kernel kernel : {
           init_query, init_history, dual, candidate}) {
    clReleaseKernel(kernel);
  }
  for (cl_program program : {
           capture_program, dual_program, candidate_program}) {
    clReleaseProgram(program);
  }
  clReleaseCommandQueue(queue);
  clReleaseContext(context);
  return numeric_pass && samples.size() == kSamples ? 0 : 2;
}

int RunTripleCohortComponent(char** argv) {
  return RunPipelinedCohortComponent(argv, false);
}

int RunNormalizedDualCohortComponent(char** argv) {
  return RunPipelinedCohortComponent(argv, true);
}

int RunThreeStageComponent(char** argv) {
  const cl_device_id device = SelectGpu();
  cl_int status = CL_SUCCESS;
  cl_context context = clCreateContext(
      nullptr, 1, &device, nullptr, nullptr, &status);
  Check(status, "clCreateContext");
  const cl_queue_properties properties[] = {
      CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
  cl_command_queue queue = clCreateCommandQueueWithProperties(
      context, device, properties, &status);
  Check(status, "clCreateCommandQueueWithProperties");

  cl_program capture_program = LoadProgram(context, device, argv[1]);
  cl_program dual_program = LoadProgram(context, device, argv[2]);
  cl_program staged_program = LoadProgram(context, device, argv[3]);
  cl_program softmax_program = LoadProgram(context, device, argv[4]);
  cl_program traffic_program = LoadProgram(context, device, argv[5]);
  cl_program vs_program = LoadProgram(context, device, argv[6]);
  auto Kernel = [&](cl_program program, const char* name) {
    cl_kernel kernel = clCreateKernel(program, name, &status);
    Check(status, std::string("clCreateKernel(") + name + ")");
    return kernel;
  };
  cl_kernel init_query =
      Kernel(capture_program, "iq36_exact_score_init_query");
  cl_kernel init_history =
      Kernel(capture_program, "iq36_exact_score_init_history");
  cl_kernel dual =
      Kernel(dual_program, "iq36_exact_score_dual_cohort");
  cl_kernel kq =
      Kernel(staged_program, "iq36_exact_score_kq_stage");
  cl_kernel owner =
      Kernel(staged_program, "iq36_exact_score_owner_stage");
  cl_kernel softmax =
      Kernel(softmax_program, "iq36_exact_score_softmax_stage");
  cl_kernel traffic =
      Kernel(traffic_program, "iq36_exact_score_softmax_traffic");
  cl_kernel vs =
      Kernel(vs_program, "iq36_exact_score_vs_stage");

  const KernelResources dual_resources = Resources(dual, device);
  const KernelResources kq_resources = Resources(kq, device);
  const KernelResources owner_resources = Resources(owner, device);
  const KernelResources softmax_resources = Resources(softmax, device);
  const KernelResources traffic_resources = Resources(traffic, device);
  const KernelResources vs_resources = Resources(vs, device);
  const std::size_t query_values = kQHeads * kHeadDim;
  const std::size_t history_values =
      kKvHeads * kContextTokens * kHeadDim;
  const std::size_t score_values =
      kKvHeads * kKqColumns * kContextTokens;
  const std::size_t state_values =
      kKvHeads * kBlocksPerKv * kKqColumns * kKeyBlock / 16;
  const std::size_t sum_values =
      kKvHeads * kKqColumns * kKeyBlock / 16;
  cl_mem query = Buffer(context, query_values * sizeof(std::uint16_t));
  cl_mem key = Buffer(context, history_values * sizeof(std::uint16_t));
  cl_mem value = Buffer(context, history_values * sizeof(std::uint16_t));
  cl_mem raw_score = Buffer(context, score_values * sizeof(std::uint32_t));
  cl_mem normalized_score =
      Buffer(context, score_values * sizeof(std::uint16_t));
  cl_mem accumulator_rescale =
      Buffer(context, state_values * sizeof(float));
  cl_mem final_sum = Buffer(context, sum_values * sizeof(float));
  cl_mem dual_output =
      Buffer(context, query_values * sizeof(std::uint16_t));
  cl_mem owner_output =
      Buffer(context, query_values * sizeof(std::uint16_t));
  cl_mem staged_output =
      Buffer(context, query_values * sizeof(std::uint16_t));

  Arg(init_query, 0, query);
  Arg(init_history, 0, key);
  Arg(init_history, 1, value);
  Enqueue1d(queue, init_query, query_values, 256);
  Enqueue1d(queue, init_history, history_values, 256);
  Check(clFinish(queue), "clFinish(initialization)");

  Arg(dual, 0, query);
  Arg(dual, 1, key);
  Arg(dual, 2, value);
  Arg(dual, 3, dual_output);
  Arg(kq, 0, query);
  Arg(kq, 1, key);
  Arg(kq, 2, raw_score);
  Arg(owner, 0, raw_score);
  Arg(owner, 1, value);
  Arg(owner, 2, owner_output);
  Arg(softmax, 0, raw_score);
  Arg(softmax, 1, normalized_score);
  Arg(softmax, 2, accumulator_rescale);
  Arg(softmax, 3, final_sum);
  Arg(traffic, 0, raw_score);
  Arg(traffic, 1, normalized_score);
  Arg(traffic, 2, accumulator_rescale);
  Arg(traffic, 3, final_sum);
  Arg(vs, 0, normalized_score);
  Arg(vs, 1, accumulator_rescale);
  Arg(vs, 2, final_sum);
  Arg(vs, 3, value);
  Arg(vs, 4, staged_output);

  (void)RunDualControl(queue, dual);
  (void)RunThreeStages(queue, kq, softmax, vs);
  (void)RunFused(queue, owner);
  std::vector<std::uint16_t> dual_values(query_values);
  std::vector<std::uint16_t> owner_values(query_values);
  std::vector<std::uint16_t> staged_values(query_values);
  Check(clEnqueueReadBuffer(
            queue, dual_output, CL_TRUE, 0,
            dual_values.size() * sizeof(dual_values[0]),
            dual_values.data(), 0, nullptr, nullptr),
        "clEnqueueReadBuffer(dual output)");
  Check(clEnqueueReadBuffer(
            queue, owner_output, CL_TRUE, 0,
            owner_values.size() * sizeof(owner_values[0]),
            owner_values.data(), 0, nullptr, nullptr),
        "clEnqueueReadBuffer(owner output)");
  Check(clEnqueueReadBuffer(
            queue, staged_output, CL_TRUE, 0,
            staged_values.size() * sizeof(staged_values[0]),
            staged_values.data(), 0, nullptr, nullptr),
        "clEnqueueReadBuffer(three-stage output)");
  const std::size_t output_mismatches =
      Mismatches(dual_values, staged_values);
  const std::size_t owner_output_mismatches =
      Mismatches(dual_values, owner_values);

  for (int warmup = 0; warmup < kWarmups; ++warmup) {
    if ((warmup & 1) == 0) {
      (void)RunDualControl(queue, dual);
      (void)RunThreeStages(queue, kq, softmax, vs);
      (void)RunFused(queue, owner);
      (void)RunFused(queue, traffic);
    } else {
      (void)RunFused(queue, traffic);
      (void)RunFused(queue, owner);
      (void)RunThreeStages(queue, kq, softmax, vs);
      (void)RunDualControl(queue, dual);
    }
  }

  struct Sample {
    const char* order = nullptr;
    double dual_ms = 0.0;
    double owner_ms = 0.0;
    double softmax_traffic_ms = 0.0;
    ThreeStageTiming staged;
  };
  std::vector<Sample> samples;
  samples.reserve(kSamples);
  for (int sample = 0; sample < kSamples; ++sample) {
    Sample row;
    if ((sample & 1) == 0) {
      row.order = "dual_three_stage";
      row.dual_ms = RunDualControl(queue, dual);
      row.staged = RunThreeStages(queue, kq, softmax, vs);
      row.owner_ms = RunFused(queue, owner).total_ms;
      row.softmax_traffic_ms = RunFused(queue, traffic).total_ms;
    } else {
      row.order = "three_stage_dual";
      row.softmax_traffic_ms = RunFused(queue, traffic).total_ms;
      row.owner_ms = RunFused(queue, owner).total_ms;
      row.staged = RunThreeStages(queue, kq, softmax, vs);
      row.dual_ms = RunDualControl(queue, dual);
    }
    samples.push_back(row);
  }

  const bool numeric_pass =
      output_mismatches == 0 && owner_output_mismatches == 0;
  const std::size_t raw_score_bytes =
      score_values * sizeof(std::uint32_t);
  const std::size_t normalized_score_bytes =
      score_values * sizeof(std::uint16_t);
  const std::size_t rescale_bytes = state_values * sizeof(float);
  const std::size_t final_sum_bytes = sum_values * sizeof(float);
  std::cout << std::boolalpha << std::setprecision(12)
            << "{\"schema_version\":"
               "\"intel-qwen36-exact-attention-three-stage-component-v1\""
            << ",\"algorithm\":"
               "\"generated_m256_n16_kq_softmax_vs_decomposition\""
            << ",\"context_tokens\":" << kContextTokens
            << ",\"head_dim\":" << kHeadDim
            << ",\"query_heads\":" << kQHeads
            << ",\"kv_heads\":" << kKvHeads
            << ",\"gqa_group\":" << kGqaGroup
            << ",\"key_block\":" << kKeyBlock
            << ",\"dual_useful_groups\":" << kKvHeads
            << ",\"kq_useful_groups\":" << kKvHeads * kBlocksPerKv
            << ",\"softmax_useful_groups\":" << kKvHeads
            << ",\"vs_useful_groups\":" << kKvHeads
            << ",\"raw_score_bytes\":" << raw_score_bytes
            << ",\"normalized_score_bytes\":" << normalized_score_bytes
            << ",\"accumulator_rescale_bytes\":" << rescale_bytes
            << ",\"final_sum_bytes\":" << final_sum_bytes
            << ",\"global_intermediate_round_trip_bytes\":"
            << 2 * (raw_score_bytes + normalized_score_bytes +
                    rescale_bytes + final_sum_bytes)
            << ",\"mandatory_key_value_payload_bytes\":"
            << 2 * history_values * sizeof(std::uint16_t)
            << ",\"output_compared_values\":" << query_values
            << ",\"output_mismatch_count\":" << output_mismatches
            << ",\"owner_output_mismatch_count\":"
            << owner_output_mismatches
            << ",\"numeric_pass\":" << numeric_pass
            << ",\"sample_count\":" << samples.size()
            << ",\"schedule\":"
               "\"interleaved_dual_three_stage_three_stage_dual\""
            << ",\"device\":\"" << DeviceString(device, CL_DEVICE_NAME)
            << "\",\"resources\":{\"dual\":";
  PrintResources(dual_resources);
  std::cout << ",\"kq\":";
  PrintResources(kq_resources);
  std::cout << ",\"owner\":";
  PrintResources(owner_resources);
  std::cout << ",\"softmax\":";
  PrintResources(softmax_resources);
  std::cout << ",\"softmax_traffic\":";
  PrintResources(traffic_resources);
  std::cout << ",\"vs\":";
  PrintResources(vs_resources);
  std::cout << "},\"paired_samples\":[";
  for (std::size_t index = 0; index < samples.size(); ++index) {
    if (index != 0) std::cout << ',';
    const Sample& row = samples[index];
    std::cout << "{\"sample\":" << index
              << ",\"order\":\"" << row.order << "\""
              << ",\"dual_ms\":" << row.dual_ms
              << ",\"owner_ms\":" << row.owner_ms
              << ",\"softmax_traffic_ms\":" << row.softmax_traffic_ms
              << ",\"three_stage\":{\"kq_ms\":" << row.staged.kq_ms
              << ",\"softmax_ms\":" << row.staged.softmax_ms
              << ",\"vs_ms\":" << row.staged.vs_ms
              << ",\"total_ms\":" << row.staged.total_ms
              << ",\"global_staged_bottleneck_ms\":"
              << std::max(
                     row.staged.kq_ms,
                     std::max(row.staged.softmax_ms, row.staged.vs_ms))
              << ",\"owner_residual_proxy_ms\":"
              << row.owner_ms - row.staged.softmax_ms
              << ",\"softmax_arithmetic_ms\":"
              << row.staged.softmax_ms - row.softmax_traffic_ms
              << ",\"projected_onchip_bottleneck_proxy_ms\":"
              << std::max(
                     row.staged.kq_ms,
                     std::max(
                         row.staged.softmax_ms,
                         row.owner_ms - row.staged.softmax_ms))
              << "},\"global_staging_penalty_ms\":"
              << row.staged.total_ms - row.dual_ms
              << ",\"projected_onchip_saving_proxy_ms\":"
              << row.dual_ms -
                     std::max(
                         row.staged.kq_ms,
                         std::max(
                             row.staged.softmax_ms,
                             row.owner_ms - row.staged.softmax_ms))
              << '}';
  }
  std::cout << "]}\n";

  for (cl_mem memory : {
           query, key, value, raw_score, normalized_score,
           accumulator_rescale, final_sum, dual_output, owner_output,
           staged_output}) {
    clReleaseMemObject(memory);
  }
  for (cl_kernel kernel : {
           init_query, init_history, dual, kq, owner, softmax, traffic, vs}) {
    clReleaseKernel(kernel);
  }
  for (cl_program program : {
           capture_program, dual_program, staged_program,
           softmax_program, traffic_program, vs_program}) {
    clReleaseProgram(program);
  }
  clReleaseCommandQueue(queue);
  clReleaseContext(context);
  return numeric_pass && samples.size() == kSamples ? 0 : 2;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const bool dense_traffic =
        argc == 4 && std::string(argv[3]) == "--dense-traffic";
    if (dense_traffic) {
      return RunDenseTrafficCeiling(argv);
    }
    const bool normalized_dual_cohort =
        argc == 5 &&
        std::string(argv[4]) == "--normalized-dual-cohort";
    if (normalized_dual_cohort) {
      return RunNormalizedDualCohortComponent(argv);
    }
    const bool triple_cohort =
        argc == 5 && std::string(argv[4]) == "--triple-cohort";
    if (triple_cohort) {
      return RunTripleCohortComponent(argv);
    }
    const bool three_stage =
        argc == 8 && std::string(argv[7]) == "--three-stage";
    if (three_stage) {
      return RunThreeStageComponent(argv);
    }
    if (argc != 4) {
      throw std::invalid_argument(
          "usage: iq36-exact-score-staging-component "
          "CAPTURE_PROGRAM FUSED_PROGRAM STAGED_PROGRAM, or "
          "CAPTURE_PROGRAM DUAL_PROGRAM STAGED_PROGRAM SOFTMAX_PROGRAM "
          "SOFTMAX_TRAFFIC_PROGRAM VS_PROGRAM --three-stage, or "
          "CAPTURE_PROGRAM DUAL_PROGRAM TRIPLE_PROGRAM --triple-cohort, or "
          "CAPTURE_PROGRAM DUAL_PROGRAM NORMALIZED_PROGRAM "
          "--normalized-dual-cohort, or "
          "CAPTURE_PROGRAM TRAFFIC_PROGRAM --dense-traffic");
    }
    const cl_device_id device = SelectGpu();
    cl_int status = CL_SUCCESS;
    cl_context context = clCreateContext(
        nullptr, 1, &device, nullptr, nullptr, &status);
    Check(status, "clCreateContext");
    const cl_queue_properties properties[] = {
        CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
    cl_command_queue queue = clCreateCommandQueueWithProperties(
        context, device, properties, &status);
    Check(status, "clCreateCommandQueueWithProperties");

    cl_program capture_program = LoadProgram(context, device, argv[1]);
    cl_program fused_program = LoadProgram(context, device, argv[2]);
    cl_program staged_program = LoadProgram(context, device, argv[3]);
    auto Kernel = [&](cl_program program, const char* name) {
      cl_kernel kernel = clCreateKernel(program, name, &status);
      Check(status, std::string("clCreateKernel(") + name + ")");
      return kernel;
    };
    cl_kernel init_query =
        Kernel(capture_program, "iq36_exact_score_init_query");
    cl_kernel init_history =
        Kernel(capture_program, "iq36_exact_score_init_history");
    cl_kernel serial_capture =
        Kernel(capture_program, "iq36_exact_score_serial_capture");
    cl_kernel fused =
        Kernel(fused_program, "iq36_exact_score_fused");
    cl_kernel kq =
        Kernel(staged_program, "iq36_exact_score_kq_stage");
    cl_kernel owner =
        Kernel(staged_program, "iq36_exact_score_owner_stage");

    const std::size_t query_values = kQHeads * kHeadDim;
    const std::size_t history_values =
        kKvHeads * kContextTokens * kHeadDim;
    const std::size_t score_values =
        kKvHeads * kKqColumns * kContextTokens;
    const std::size_t output_values = query_values;
    cl_mem query = Buffer(context, query_values * sizeof(std::uint16_t));
    cl_mem key = Buffer(context, history_values * sizeof(std::uint16_t));
    cl_mem value = Buffer(context, history_values * sizeof(std::uint16_t));
    cl_mem serial_score = Buffer(context, score_values * sizeof(std::uint32_t));
    cl_mem staged_score = Buffer(context, score_values * sizeof(std::uint32_t));
    cl_mem fused_output = Buffer(context, output_values * sizeof(std::uint16_t));
    cl_mem staged_output = Buffer(context, output_values * sizeof(std::uint16_t));

    Arg(init_query, 0, query);
    Arg(init_history, 0, key);
    Arg(init_history, 1, value);
    Enqueue1d(queue, init_query, query_values, 256);
    Enqueue1d(queue, init_history, history_values, 256);
    Check(clFinish(queue), "clFinish(initialization)");

    Arg(serial_capture, 0, query);
    Arg(serial_capture, 1, key);
    Arg(serial_capture, 2, serial_score);
    Arg(fused, 0, query);
    Arg(fused, 1, key);
    Arg(fused, 2, value);
    Arg(fused, 3, fused_output);
    Arg(kq, 0, query);
    Arg(kq, 1, key);
    Arg(kq, 2, staged_score);
    Arg(owner, 0, staged_score);
    Arg(owner, 1, value);
    Arg(owner, 2, staged_output);

    const std::array<std::size_t, 2> capture_global = {
        16, 16 * kKvHeads};
    const std::array<std::size_t, 2> local = {16, 16};
    cl_event capture_event =
        Enqueue2d(queue, serial_capture, capture_global, local);
    Check(clWaitForEvents(1, &capture_event),
          "clWaitForEvents(serial capture)");
    clReleaseEvent(capture_event);

    std::vector<std::uint32_t> serial_scores(score_values);
    std::vector<std::uint32_t> staged_scores(score_values);
    Check(clEnqueueReadBuffer(
        queue, serial_score, CL_TRUE, 0,
        serial_scores.size() * sizeof(serial_scores[0]),
        serial_scores.data(), 0, nullptr, nullptr),
        "clEnqueueReadBuffer(serial scores)");
    (void)RunStaged(queue, kq, owner);
    Check(clEnqueueReadBuffer(
        queue, staged_score, CL_TRUE, 0,
        staged_scores.size() * sizeof(staged_scores[0]),
        staged_scores.data(), 0, nullptr, nullptr),
        "clEnqueueReadBuffer(staged scores)");
    const std::size_t raw_score_mismatches =
        Mismatches(serial_scores, staged_scores);
    serial_scores.clear();
    staged_scores.clear();
    serial_scores.shrink_to_fit();
    staged_scores.shrink_to_fit();

    (void)RunFused(queue, fused);
    (void)RunStaged(queue, kq, owner);
    std::vector<std::uint16_t> fused_values(output_values);
    std::vector<std::uint16_t> staged_values(output_values);
    Check(clEnqueueReadBuffer(
        queue, fused_output, CL_TRUE, 0,
        fused_values.size() * sizeof(fused_values[0]),
        fused_values.data(), 0, nullptr, nullptr),
        "clEnqueueReadBuffer(fused output)");
    Check(clEnqueueReadBuffer(
        queue, staged_output, CL_TRUE, 0,
        staged_values.size() * sizeof(staged_values[0]),
        staged_values.data(), 0, nullptr, nullptr),
        "clEnqueueReadBuffer(staged output)");
    const std::size_t output_mismatches =
        Mismatches(fused_values, staged_values);

    for (int warmup = 0; warmup < kWarmups; ++warmup) {
      if ((warmup & 1) == 0) {
        (void)RunFused(queue, fused);
        (void)RunStaged(queue, kq, owner);
      } else {
        (void)RunStaged(queue, kq, owner);
        (void)RunFused(queue, fused);
      }
    }

    struct Sample {
      const char* order = nullptr;
      FusedTiming fused;
      StagedTiming staged;
    };
    std::vector<Sample> samples;
    samples.reserve(kSamples);
    for (int sample = 0; sample < kSamples; ++sample) {
      Sample row;
      if ((sample & 1) == 0) {
        row.order = "fused_staged";
        row.fused = RunFused(queue, fused);
        row.staged = RunStaged(queue, kq, owner);
      } else {
        row.order = "staged_fused";
        row.staged = RunStaged(queue, kq, owner);
        row.fused = RunFused(queue, fused);
      }
      samples.push_back(row);
    }

    const bool numeric_pass =
        raw_score_mismatches == 0 && output_mismatches == 0;
    std::cout << std::boolalpha << std::setprecision(12)
              << "{\"schema_version\":"
                 "\"intel-qwen36-exact-score-staging-component-v1\""
              << ",\"algorithm\":"
                 "\"generated_m256_n16_full_raw_f32_staging\""
              << ",\"context_tokens\":" << kContextTokens
              << ",\"head_dim\":" << kHeadDim
              << ",\"query_heads\":" << kQHeads
              << ",\"kv_heads\":" << kKvHeads
              << ",\"gqa_group\":" << kGqaGroup
              << ",\"kq_columns_per_kv\":" << kKqColumns
              << ",\"key_block\":" << kKeyBlock
              << ",\"fused_useful_groups\":" << kKvHeads
              << ",\"staged_kq_useful_groups\":"
              << kKvHeads * kBlocksPerKv
              << ",\"staged_owner_useful_groups\":" << kKvHeads
              << ",\"raw_score_bytes\":"
              << score_values * sizeof(std::uint32_t)
              << ",\"raw_score_round_trip_bytes\":"
              << 2 * score_values * sizeof(std::uint32_t)
              << ",\"raw_score_compared_values\":" << score_values
              << ",\"raw_score_mismatch_count\":" << raw_score_mismatches
              << ",\"output_compared_values\":" << output_values
              << ",\"output_mismatch_count\":" << output_mismatches
              << ",\"numeric_pass\":" << numeric_pass
              << ",\"sample_count\":" << samples.size()
              << ",\"schedule\":\"interleaved_fused_staged_staged_fused\""
              << ",\"device\":\"" << DeviceString(device, CL_DEVICE_NAME)
              << "\",\"paired_samples\":[";
    for (std::size_t index = 0; index < samples.size(); ++index) {
      if (index != 0) std::cout << ',';
      const Sample& row = samples[index];
      std::cout << "{\"sample\":" << index
                << ",\"order\":\"" << row.order << "\""
                << ",\"fused\":{\"total_ms\":"
                << row.fused.total_ms << "}"
                << ",\"staged\":{\"kq_ms\":" << row.staged.kq_ms
                << ",\"owner_ms\":" << row.staged.owner_ms
                << ",\"total_ms\":" << row.staged.total_ms << "}"
                << ",\"differential_ms\":"
                << row.staged.total_ms - row.fused.total_ms << '}';
    }
    std::cout << "]}\n";

    for (cl_mem memory : {
             query, key, value, serial_score, staged_score,
             fused_output, staged_output}) {
      clReleaseMemObject(memory);
    }
    for (cl_kernel kernel : {
             init_query, init_history, serial_capture, fused, kq, owner}) {
      clReleaseKernel(kernel);
    }
    clReleaseProgram(capture_program);
    clReleaseProgram(fused_program);
    clReleaseProgram(staged_program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return numeric_pass && samples.size() == kSamples ? 0 : 2;
  } catch (const std::exception& error) {
    std::cerr << "iq36-exact-score-staging-component: "
              << error.what() << '\n';
    return 4;
  }
}
