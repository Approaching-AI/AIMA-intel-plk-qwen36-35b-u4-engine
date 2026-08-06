#include <CL/cl.h>

#include <array>
#include <cstdint>
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
constexpr std::size_t kControlLocalY = 16;
constexpr std::size_t kDualCohortLocalY = 32;
constexpr int kWarmups = 3;
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

cl_event Enqueue2d(
    cl_command_queue queue, cl_kernel kernel, std::size_t local_y) {
  const std::array<std::size_t, 2> global = {16, local_y * kKvHeads};
  const std::array<std::size_t, 2> local = {16, local_y};
  cl_event event = nullptr;
  Check(clEnqueueNDRangeKernel(
      queue, kernel, 2, nullptr, global.data(), local.data(),
      0, nullptr, &event), "clEnqueueNDRangeKernel(2d)");
  return event;
}

double Run(
    cl_command_queue queue, cl_kernel kernel, std::size_t local_y) {
  cl_event event = Enqueue2d(queue, kernel, local_y);
  Check(clWaitForEvents(1, &event), "clWaitForEvents(attention)");
  cl_ulong start = 0;
  cl_ulong end = 0;
  Check(clGetEventProfilingInfo(
      event, CL_PROFILING_COMMAND_START, sizeof(start), &start, nullptr),
      "clGetEventProfilingInfo(start)");
  Check(clGetEventProfilingInfo(
      event, CL_PROFILING_COMMAND_END, sizeof(end), &end, nullptr),
      "clGetEventProfilingInfo(end)");
  clReleaseEvent(event);
  return static_cast<double>(end - start) / 1.0e6;
}

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
      sizeof(result.spill_memory_bytes), &result.spill_memory_bytes, nullptr),
      "clGetKernelWorkGroupInfo(spill memory)");
  Check(clGetKernelWorkGroupInfo(
      kernel, device, CL_KERNEL_LOCAL_MEM_SIZE,
      sizeof(result.local_memory_bytes), &result.local_memory_bytes, nullptr),
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

template <typename T>
std::size_t Mismatches(
    const std::vector<T>& left, const std::vector<T>& right) {
  Require(left.size() == right.size(), "comparison shape mismatch");
  std::size_t mismatches = 0;
  for (std::size_t index = 0; index < left.size(); ++index) {
    mismatches += left[index] != right[index];
  }
  return mismatches;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const bool dual_cohort =
        argc == 5 && std::string(argv[4]) == "--dual-cohort";
    if (argc != 4 && !dual_cohort) {
      throw std::invalid_argument(
          "usage: iq36-exact-attention-vrt160-component "
          "CAPTURE_PROGRAM CONTROL128_PROGRAM CANDIDATE_PROGRAM "
          "[--dual-cohort]");
    }
    const std::size_t candidate_local_y =
        dual_cohort ? kDualCohortLocalY : kControlLocalY;
    const char* candidate_kernel_name =
        dual_cohort ? "iq36_exact_score_dual_cohort"
                    : "iq36_exact_score_fused";
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
    cl_program control_program = LoadProgram(context, device, argv[2]);
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
    cl_kernel control =
        Kernel(control_program, "iq36_exact_score_fused");
    cl_kernel candidate =
        Kernel(candidate_program, candidate_kernel_name);

    const KernelResources control_resources = Resources(control, device);
    const KernelResources candidate_resources = Resources(candidate, device);
    const std::size_t query_values = kQHeads * kHeadDim;
    const std::size_t history_values =
        kKvHeads * kContextTokens * kHeadDim;
    cl_mem query = Buffer(context, query_values * sizeof(std::uint16_t));
    cl_mem key = Buffer(context, history_values * sizeof(std::uint16_t));
    cl_mem value = Buffer(context, history_values * sizeof(std::uint16_t));
    cl_mem control_output =
        Buffer(context, query_values * sizeof(std::uint16_t));
    cl_mem candidate_output =
        Buffer(context, query_values * sizeof(std::uint16_t));

    Arg(init_query, 0, query);
    Arg(init_history, 0, key);
    Arg(init_history, 1, value);
    Enqueue1d(queue, init_query, query_values, 256);
    Enqueue1d(queue, init_history, history_values, 256);
    Check(clFinish(queue), "clFinish(initialization)");

    for (cl_kernel kernel : {control, candidate}) {
      Arg(kernel, 0, query);
      Arg(kernel, 1, key);
      Arg(kernel, 2, value);
    }
    Arg(control, 3, control_output);
    Arg(candidate, 3, candidate_output);

    (void)Run(queue, control, kControlLocalY);
    (void)Run(queue, candidate, candidate_local_y);
    std::vector<std::uint16_t> control_values(query_values);
    std::vector<std::uint16_t> candidate_values(query_values);
    Check(clEnqueueReadBuffer(
        queue, control_output, CL_TRUE, 0,
        control_values.size() * sizeof(control_values[0]),
        control_values.data(), 0, nullptr, nullptr),
        "clEnqueueReadBuffer(control output)");
    Check(clEnqueueReadBuffer(
        queue, candidate_output, CL_TRUE, 0,
        candidate_values.size() * sizeof(candidate_values[0]),
        candidate_values.data(), 0, nullptr, nullptr),
        "clEnqueueReadBuffer(candidate output)");
    const std::size_t output_mismatches =
        Mismatches(control_values, candidate_values);

    for (int warmup = 0; warmup < kWarmups; ++warmup) {
      if ((warmup & 1) == 0) {
        (void)Run(queue, control, kControlLocalY);
        (void)Run(queue, candidate, candidate_local_y);
      } else {
        (void)Run(queue, candidate, candidate_local_y);
        (void)Run(queue, control, kControlLocalY);
      }
    }

    struct Sample {
      const char* order = nullptr;
      double control_ms = 0.0;
      double candidate_ms = 0.0;
    };
    std::vector<Sample> samples;
    samples.reserve(kSamples);
    for (int sample = 0; sample < kSamples; ++sample) {
      Sample row;
      if ((sample & 1) == 0) {
        row.order = "control_candidate";
        row.control_ms = Run(queue, control, kControlLocalY);
        row.candidate_ms = Run(queue, candidate, candidate_local_y);
      } else {
        row.order = "candidate_control";
        row.candidate_ms = Run(queue, candidate, candidate_local_y);
        row.control_ms = Run(queue, control, kControlLocalY);
      }
      samples.push_back(row);
    }

    const bool numeric_pass = output_mismatches == 0;
    std::cout << std::boolalpha << std::setprecision(12)
              << "{\"schema_version\":"
              << (dual_cohort
                      ? "\"intel-qwen36-exact-attention-dual-cohort-"
                        "component-v1\""
                      : "\"intel-qwen36-exact-attention-vrt160-"
                        "component-v1\"")
              << ",\"algorithm\":"
              << (dual_cohort
                      ? "\"generated_m256_n16_dual_cohort_pipeline\""
                      : "\"generated_m256_n16_fused_vrt160\"")
              << ",\"context_tokens\":" << kContextTokens
              << ",\"head_dim\":" << kHeadDim
              << ",\"query_heads\":" << kQHeads
              << ",\"kv_heads\":" << kKvHeads
              << ",\"gqa_group\":" << kGqaGroup
              << ",\"useful_workgroups\":" << kKvHeads
              << ",\"output_compared_values\":" << query_values
              << ",\"output_mismatch_count\":" << output_mismatches
              << ",\"numeric_pass\":" << numeric_pass
              << ",\"control_register_count\":"
              << control_resources.register_count
              << ",\"candidate_register_count\":"
              << candidate_resources.register_count
              << ",\"control_spill_memory_bytes\":"
              << control_resources.spill_memory_bytes
              << ",\"candidate_spill_memory_bytes\":"
              << candidate_resources.spill_memory_bytes
              << ",\"control_local_memory_bytes\":"
              << control_resources.local_memory_bytes
              << ",\"candidate_local_memory_bytes\":"
              << candidate_resources.local_memory_bytes
              << ",\"control_maximum_workgroup_items\":"
              << control_resources.maximum_workgroup_items
              << ",\"candidate_maximum_workgroup_items\":"
              << candidate_resources.maximum_workgroup_items
              << ",\"control_preferred_workgroup_multiple\":"
              << control_resources.preferred_workgroup_multiple
              << ",\"candidate_preferred_workgroup_multiple\":"
              << candidate_resources.preferred_workgroup_multiple
              << ",\"control_local_workgroup_items\":"
              << 16 * kControlLocalY
              << ",\"candidate_local_workgroup_items\":"
              << 16 * candidate_local_y
              << ",\"sample_count\":" << samples.size()
              << ",\"schedule\":"
                 "\"interleaved_control_candidate_candidate_control\""
              << ",\"device\":\"" << DeviceString(device, CL_DEVICE_NAME)
              << "\",\"paired_samples\":[";
    for (std::size_t index = 0; index < samples.size(); ++index) {
      if (index != 0) std::cout << ',';
      const Sample& row = samples[index];
      std::cout << "{\"sample\":" << index
                << ",\"order\":\"" << row.order << "\""
                << ",\"control_ms\":" << row.control_ms
                << ",\"candidate_ms\":" << row.candidate_ms
                << ",\"differential_ms\":"
                << row.candidate_ms - row.control_ms << '}';
    }
    std::cout << "]}\n";

    for (cl_mem memory : {
             query, key, value, control_output, candidate_output}) {
      clReleaseMemObject(memory);
    }
    for (cl_kernel kernel : {
             init_query, init_history, control, candidate}) {
      clReleaseKernel(kernel);
    }
    clReleaseProgram(capture_program);
    clReleaseProgram(control_program);
    clReleaseProgram(candidate_program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return numeric_pass && samples.size() == kSamples ? 0 : 2;
  } catch (const std::exception& error) {
    std::cerr << "iq36-exact-attention-vrt160-component: "
              << error.what() << '\n';
    return 4;
  }
}
