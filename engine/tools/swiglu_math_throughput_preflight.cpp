#include <CL/cl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kValueCount = 4'194'304;
constexpr std::size_t kValuesPerWorkItem = 8;
constexpr std::size_t kGlobalSize = kValueCount / kValuesPerWorkItem;
constexpr std::size_t kLocalSize = 16;
constexpr std::size_t kResidualFmaCount = 805'306'368;

constexpr const char* kSource = R"CLC(
__attribute__((reqd_work_group_size(16, 1, 1)))
__kernel void swiglu_math_throughput(__global float *output) {
  const uint index = get_global_id(0);
  const float base = ((float)((int)(index & 1023U) - 512)) * 0.005f;
  const float8 gate = base + (float8)(
      0.00f, 0.01f, 0.02f, 0.03f, 0.04f, 0.05f, 0.06f, 0.07f);
  const float8 up = 0.75f + base * 0.125f + (float8)(
      0.08f, 0.07f, 0.06f, 0.05f, 0.04f, 0.03f, 0.02f, 0.01f);
  const float8 value = gate * (1.0f / (1.0f + exp(-gate))) * up;
  output[index] = value.s0 + value.s1 + value.s2 + value.s3 +
      value.s4 + value.s5 + value.s6 + value.s7;
}

__attribute__((reqd_work_group_size(16, 1, 1)))
__kernel void q4k_residual_fma_throughput(__global float *output) {
  const uint index = get_global_id(0);
  const float base = ((float)((int)(index & 1023U) - 512)) * 0.0001f;
  float8 acc0 = base + (float8)(0.01f);
  float8 acc1 = base + (float8)(0.02f);
  float8 acc2 = base + (float8)(0.03f);
  float8 acc3 = base + (float8)(0.04f);
#pragma unroll 48
  for (uint iteration = 0; iteration < 48U; ++iteration) {
    const float scale = 0.9999f + (float)iteration * 0.000001f;
    const float8 coefficient = (float8)(scale);
    acc0 = fma(acc0, coefficient, (float8)(0.00001f));
    acc1 = fma(acc1, coefficient, (float8)(0.00002f));
    acc2 = fma(acc2, coefficient, (float8)(0.00003f));
    acc3 = fma(acc3, coefficient, (float8)(0.00004f));
  }
  const float8 value = acc0 + acc1 + acc2 + acc3;
  output[index] = value.s0 + value.s1 + value.s2 + value.s3 +
      value.s4 + value.s5 + value.s6 + value.s7;
}
)CLC";

void Check(cl_int status, const char* operation) {
  if (status != CL_SUCCESS) {
    throw std::runtime_error(
        std::string(operation) + " failed: " + std::to_string(status));
  }
}

std::string DeviceString(cl_device_id device, cl_device_info key) {
  std::size_t size = 0;
  Check(clGetDeviceInfo(device, key, 0, nullptr, &size), "clGetDeviceInfo size");
  std::string value(size, '\0');
  Check(clGetDeviceInfo(device, key, size, value.data(), nullptr),
        "clGetDeviceInfo value");
  if (!value.empty() && value.back() == '\0') value.pop_back();
  return value;
}

cl_device_id FindDevice() {
  cl_uint platform_count = 0;
  Check(clGetPlatformIDs(0, nullptr, &platform_count), "clGetPlatformIDs count");
  std::vector<cl_platform_id> platforms(platform_count);
  Check(clGetPlatformIDs(platform_count, platforms.data(), nullptr),
        "clGetPlatformIDs");
  for (cl_platform_id platform : platforms) {
    cl_uint device_count = 0;
    cl_int status = clGetDeviceIDs(
        platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &device_count);
    if (status == CL_DEVICE_NOT_FOUND) continue;
    Check(status, "clGetDeviceIDs count");
    std::vector<cl_device_id> devices(device_count);
    Check(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, device_count,
                         devices.data(), nullptr),
          "clGetDeviceIDs");
    for (cl_device_id device : devices) {
      if (DeviceString(device, CL_DEVICE_NAME).find("B390") !=
          std::string::npos) {
        return device;
      }
    }
  }
  throw std::runtime_error("Intel Arc B390 GPU not found");
}

double EventMicroseconds(cl_event event) {
  cl_ulong begin = 0;
  cl_ulong end = 0;
  Check(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START,
                                sizeof(begin), &begin, nullptr),
        "clGetEventProfilingInfo start");
  Check(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,
                                sizeof(end), &end, nullptr),
        "clGetEventProfilingInfo end");
  return static_cast<double>(end - begin) / 1000.0;
}

struct Stats {
  double minimum = 0.0;
  double median = 0.0;
  double mean = 0.0;
  std::vector<double> samples;
};

Stats Summarize(std::vector<double> samples) {
  std::vector<double> sorted = samples;
  std::sort(sorted.begin(), sorted.end());
  return {
      sorted.front(),
      sorted[sorted.size() / 2],
      std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size(),
      std::move(samples),
  };
}

void PrintStats(const char* name, const Stats& stats) {
  std::cout << "\"" << name << "\":{\"mean_us\":" << stats.mean
            << ",\"median_us\":" << stats.median
            << ",\"minimum_us\":" << stats.minimum << ",\"samples_us\":[";
  for (std::size_t index = 0; index < stats.samples.size(); ++index) {
    if (index != 0) std::cout << ',';
    std::cout << stats.samples[index];
  }
  std::cout << "]}";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int warmup = 3;
    int repeat = 11;
    for (int index = 1; index < argc; ++index) {
      const std::string option = argv[index];
      if (++index >= argc) throw std::runtime_error(option + " needs a value");
      const int value = std::stoi(argv[index]);
      if (value <= 0) throw std::runtime_error(option + " must be positive");
      if (option == "--warmup") warmup = value;
      else if (option == "--repeat") repeat = value;
      else throw std::runtime_error("unknown option: " + option);
    }

    const cl_device_id device = FindDevice();
    cl_int status = CL_SUCCESS;
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr,
                                         &status);
    Check(status, "clCreateContext");
    const cl_queue_properties properties[] = {
        CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
    cl_command_queue queue = clCreateCommandQueueWithProperties(
        context, device, properties, &status);
    Check(status, "clCreateCommandQueueWithProperties");
    const std::size_t source_size = std::strlen(kSource);
    const char* source = kSource;
    cl_program program = clCreateProgramWithSource(
        context, 1, &source, &source_size, &status);
    Check(status, "clCreateProgramWithSource");
    status = clBuildProgram(program, 1, &device, "-cl-std=CL2.0", nullptr,
                            nullptr);
    if (status != CL_SUCCESS) {
      std::size_t log_size = 0;
      clGetProgramBuildInfo(
          program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
      std::string log(log_size, '\0');
      clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size,
                            log.data(), nullptr);
      throw std::runtime_error("clBuildProgram failed: " + log);
    }
    cl_kernel swiglu_kernel =
        clCreateKernel(program, "swiglu_math_throughput", &status);
    Check(status, "clCreateKernel swiglu");
    cl_kernel residual_kernel =
        clCreateKernel(program, "q4k_residual_fma_throughput", &status);
    Check(status, "clCreateKernel residual");
    cl_mem output = clCreateBuffer(
        context, CL_MEM_WRITE_ONLY, kGlobalSize * sizeof(float), nullptr,
        &status);
    Check(status, "clCreateBuffer");
    Check(clSetKernelArg(swiglu_kernel, 0, sizeof(output), &output),
          "clSetKernelArg swiglu");
    Check(clSetKernelArg(residual_kernel, 0, sizeof(output), &output),
          "clSetKernelArg residual");

    const auto Execute = [&](cl_kernel kernel, bool profile) {
      cl_event event = nullptr;
      Check(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &kGlobalSize,
                                   &kLocalSize, 0, nullptr,
                                   profile ? &event : nullptr),
            "clEnqueueNDRangeKernel");
      Check(clFinish(queue), "clFinish");
      if (!profile) return 0.0;
      const double elapsed = EventMicroseconds(event);
      clReleaseEvent(event);
      return elapsed;
    };
    for (int iteration = 0; iteration < warmup; ++iteration) {
      Execute(swiglu_kernel, false);
      Execute(residual_kernel, false);
    }
    std::vector<double> swiglu_samples;
    std::vector<double> residual_samples;
    swiglu_samples.reserve(repeat);
    residual_samples.reserve(repeat);
    for (int iteration = 0; iteration < repeat; ++iteration) {
      swiglu_samples.push_back(Execute(swiglu_kernel, true));
      residual_samples.push_back(Execute(residual_kernel, true));
    }
    std::vector<float> values(kGlobalSize);
    Check(clEnqueueReadBuffer(queue, output, CL_TRUE, 0,
                              values.size() * sizeof(float), values.data(), 0,
                              nullptr, nullptr),
          "clEnqueueReadBuffer");
    const bool finite = std::all_of(
        values.begin(), values.end(), [](float value) { return std::isfinite(value); });
    const Stats swiglu_stats = Summarize(std::move(swiglu_samples));
    const Stats residual_stats = Summarize(std::move(residual_samples));
    std::cout << std::boolalpha << std::setprecision(12) << "{"
              << "\"device_name\":\"" << DeviceString(device, CL_DEVICE_NAME)
              << "\",\"driver_version\":\""
              << DeviceString(device, CL_DRIVER_VERSION) << "\","
              << "\"finite\":" << finite << ","
              << "\"global_work_items\":" << kGlobalSize << ","
              << "\"residual_fma_count\":" << kResidualFmaCount << ",";
    PrintStats("residual_fma", residual_stats);
    std::cout << ',';
    PrintStats("swiglu", swiglu_stats);
    std::cout << ",\"value_count\":" << kValueCount << "}\n";

    clReleaseMemObject(output);
    clReleaseKernel(residual_kernel);
    clReleaseKernel(swiglu_kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return finite ? 0 : 2;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 3;
  }
}
