#include "intel_qwen36/gguf_loader.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

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
constexpr cl_bool kClTrue = 1;
constexpr cl_device_type kClDeviceTypeGpu = 1ULL << 2;
constexpr cl_mem_flags kClMemReadOnly = 1ULL << 2;
constexpr cl_mem_flags kClMemWriteOnly = 1ULL << 1;
constexpr cl_mem_flags kClMemCopyHostPtr = 1ULL << 5;
constexpr cl_command_queue_properties kClQueueProfilingEnable = 1ULL << 1;
constexpr cl_platform_info kClPlatformName = 0x0902;
constexpr cl_device_info kClDeviceName = 0x102B;
constexpr cl_program_build_info kClProgramBuildLog = 0x1183;
constexpr cl_profiling_info kClProfilingCommandStart = 0x1282;
constexpr cl_profiling_info kClProfilingCommandEnd = 0x1283;
constexpr std::size_t kQ6BlockBytes = 210;
constexpr std::size_t kQk = 256;

[[noreturn]] void Die(const std::string& message) {
  throw std::runtime_error(message);
}

void Require(bool condition, const std::string& message) {
  if (!condition) Die(message);
}

void Check(cl_int status, const std::string& where) {
  if (status != kClSuccess) {
    std::ostringstream message;
    message << where << " failed with OpenCL error " << status;
    Die(message.str());
  }
}

template <typename Fn>
Fn LoadSymbol(void* library, const char* name) {
  void* symbol = dlsym(library, name);
  if (symbol == nullptr) Die(std::string("missing OpenCL symbol: ") + name);
  return reinterpret_cast<Fn>(symbol);
}

struct OpenClApi {
  void* library = nullptr;
  cl_int (*clGetPlatformIDs)(cl_uint, cl_platform_id*, cl_uint*) = nullptr;
  cl_int (*clGetPlatformInfo)(cl_platform_id, cl_platform_info, std::size_t,
                              void*, std::size_t*) = nullptr;
  cl_int (*clGetDeviceIDs)(cl_platform_id, cl_device_type, cl_uint,
                           cl_device_id*, cl_uint*) = nullptr;
  cl_int (*clGetDeviceInfo)(cl_device_id, cl_device_info, std::size_t, void*,
                            std::size_t*) = nullptr;
  cl_context (*clCreateContext)(const cl_context_properties*, cl_uint,
                                const cl_device_id*, void*, void*, cl_int*) = nullptr;
  cl_int (*clReleaseContext)(cl_context) = nullptr;
  cl_command_queue (*clCreateCommandQueue)(cl_context, cl_device_id,
                                            cl_command_queue_properties,
                                            cl_int*) = nullptr;
  cl_int (*clReleaseCommandQueue)(cl_command_queue) = nullptr;
  cl_mem (*clCreateBuffer)(cl_context, cl_mem_flags, std::size_t, void*,
                           cl_int*) = nullptr;
  cl_int (*clReleaseMemObject)(cl_mem) = nullptr;
  cl_program (*clCreateProgramWithSource)(cl_context, cl_uint, const char**,
                                          const std::size_t*, cl_int*) = nullptr;
  cl_int (*clBuildProgram)(cl_program, cl_uint, const cl_device_id*,
                           const char*, void*, void*) = nullptr;
  cl_int (*clGetProgramBuildInfo)(cl_program, cl_device_id,
                                  cl_program_build_info, std::size_t, void*,
                                  std::size_t*) = nullptr;
  cl_int (*clReleaseProgram)(cl_program) = nullptr;
  cl_kernel (*clCreateKernel)(cl_program, const char*, cl_int*) = nullptr;
  cl_int (*clSetKernelArg)(cl_kernel, cl_uint, std::size_t, const void*) = nullptr;
  cl_int (*clReleaseKernel)(cl_kernel) = nullptr;
  cl_int (*clEnqueueReadBuffer)(cl_command_queue, cl_mem, cl_bool, std::size_t,
                                std::size_t, void*, cl_uint, const cl_event*,
                                cl_event*) = nullptr;
  cl_int (*clEnqueueNDRangeKernel)(cl_command_queue, cl_kernel, cl_uint,
                                   const std::size_t*, const std::size_t*,
                                   const std::size_t*, cl_uint, const cl_event*,
                                   cl_event*) = nullptr;
  cl_int (*clFinish)(cl_command_queue) = nullptr;
  cl_int (*clGetEventProfilingInfo)(cl_event, cl_profiling_info, std::size_t,
                                    void*, std::size_t*) = nullptr;
  cl_int (*clReleaseEvent)(cl_event) = nullptr;

  OpenClApi() {
    library = dlopen("libOpenCL.so.1", RTLD_NOW | RTLD_LOCAL);
    if (library == nullptr) Die(std::string("dlopen OpenCL failed: ") + dlerror());
#define IQ36_LOAD(name) name = LoadSymbol<decltype(name)>(library, #name)
    IQ36_LOAD(clGetPlatformIDs);
    IQ36_LOAD(clGetPlatformInfo);
    IQ36_LOAD(clGetDeviceIDs);
    IQ36_LOAD(clGetDeviceInfo);
    IQ36_LOAD(clCreateContext);
    IQ36_LOAD(clReleaseContext);
    IQ36_LOAD(clCreateCommandQueue);
    IQ36_LOAD(clReleaseCommandQueue);
    IQ36_LOAD(clCreateBuffer);
    IQ36_LOAD(clReleaseMemObject);
    IQ36_LOAD(clCreateProgramWithSource);
    IQ36_LOAD(clBuildProgram);
    IQ36_LOAD(clGetProgramBuildInfo);
    IQ36_LOAD(clReleaseProgram);
    IQ36_LOAD(clCreateKernel);
    IQ36_LOAD(clSetKernelArg);
    IQ36_LOAD(clReleaseKernel);
    IQ36_LOAD(clEnqueueReadBuffer);
    IQ36_LOAD(clEnqueueNDRangeKernel);
    IQ36_LOAD(clFinish);
    IQ36_LOAD(clGetEventProfilingInfo);
    IQ36_LOAD(clReleaseEvent);
#undef IQ36_LOAD
  }

  ~OpenClApi() {
    if (library != nullptr) dlclose(library);
  }
};

std::string InfoString(OpenClApi& api, cl_platform_id platform,
                       cl_platform_info key) {
  std::size_t size = 0;
  Check(api.clGetPlatformInfo(platform, key, 0, nullptr, &size),
        "clGetPlatformInfo(size)");
  std::string value(size, '\0');
  Check(api.clGetPlatformInfo(platform, key, size, value.data(), nullptr),
        "clGetPlatformInfo(value)");
  while (!value.empty() && value.back() == '\0') value.pop_back();
  return value;
}

std::string InfoString(OpenClApi& api, cl_device_id device,
                       cl_device_info key) {
  std::size_t size = 0;
  Check(api.clGetDeviceInfo(device, key, 0, nullptr, &size),
        "clGetDeviceInfo(size)");
  std::string value(size, '\0');
  Check(api.clGetDeviceInfo(device, key, size, value.data(), nullptr),
        "clGetDeviceInfo(value)");
  while (!value.empty() && value.back() == '\0') value.pop_back();
  return value;
}

struct Device {
  cl_platform_id platform = nullptr;
  cl_device_id device = nullptr;
  std::string platform_name;
  std::string device_name;
};

Device SelectDevice(OpenClApi& api, const std::string& substring) {
  cl_uint platform_count = 0;
  Check(api.clGetPlatformIDs(0, nullptr, &platform_count),
        "clGetPlatformIDs(count)");
  std::vector<cl_platform_id> platforms(platform_count);
  Check(api.clGetPlatformIDs(platform_count, platforms.data(), nullptr),
        "clGetPlatformIDs(list)");
  for (auto platform : platforms) {
    cl_uint device_count = 0;
    if (api.clGetDeviceIDs(platform, kClDeviceTypeGpu, 0, nullptr,
                           &device_count) != kClSuccess ||
        device_count == 0) {
      continue;
    }
    std::vector<cl_device_id> devices(device_count);
    Check(api.clGetDeviceIDs(platform, kClDeviceTypeGpu, device_count,
                             devices.data(), nullptr),
          "clGetDeviceIDs(list)");
    for (auto device : devices) {
      const auto name = InfoString(api, device, kClDeviceName);
      if (substring.empty() || name.find(substring) != std::string::npos) {
        return {platform, device, InfoString(api, platform, kClPlatformName), name};
      }
    }
  }
  Die("no matching GPU: " + substring);
}

std::string ReadText(const std::string& path) {
  std::ifstream input(path);
  Require(static_cast<bool>(input), "kernel source could not be opened");
  std::ostringstream buffer;
  buffer << input.rdbuf();
  return buffer.str();
}

std::vector<std::uint8_t> ReadTensor(const std::string& model_path,
                                     const iq36::GgufTensorInfo& tensor) {
  std::ifstream input(model_path, std::ios::binary);
  Require(static_cast<bool>(input), "model could not be opened");
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(tensor.nbytes));
  input.seekg(static_cast<std::streamoff>(tensor.absolute_offset), std::ios::beg);
  Require(static_cast<bool>(input), "tensor seek failed");
  input.read(reinterpret_cast<char*>(bytes.data()),
             static_cast<std::streamsize>(bytes.size()));
  Require(input.gcount() == static_cast<std::streamsize>(bytes.size()),
          "tensor read failed");
  return bytes;
}

std::uint16_t LoadU16(const std::uint8_t* bytes) {
  return std::uint16_t(bytes[0]) | (std::uint16_t(bytes[1]) << 8);
}

float HalfToFloat(std::uint16_t value) {
  const std::uint32_t sign = (std::uint32_t(value) & 0x8000U) << 16;
  std::uint32_t exponent = (value >> 10) & 0x1FU;
  std::uint32_t mantissa = value & 0x03FFU;
  std::uint32_t bits = 0;
  if (exponent == 0) {
    if (mantissa == 0) {
      bits = sign;
    } else {
      std::uint32_t shift = 0;
      while ((mantissa & 0x0400U) == 0) {
        mantissa <<= 1;
        ++shift;
      }
      mantissa &= 0x03FFU;
      bits = sign | ((127U - 14U - shift) << 23) | (mantissa << 13);
    }
  } else if (exponent == 0x1FU) {
    bits = sign | 0x7F800000U | (mantissa << 13);
  } else {
    bits = sign | ((exponent + 112U) << 23) | (mantissa << 13);
  }
  float result = 0.0f;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

int NearestInt(float value) {
  const float shifted = value + 12582912.0f;
  int bits = 0;
  std::memcpy(&bits, &shifted, sizeof(bits));
  return (bits & 0x007fffff) - 0x00400000;
}

struct Q8Input {
  std::vector<std::int8_t> values;
  std::vector<std::int16_t> sums;
  std::vector<float> scales;
};

Q8Input QuantizeInput(std::size_t columns) {
  Require(columns % kQk == 0, "Q8 input columns must be 256-aligned");
  std::vector<float> input(columns);
  for (std::size_t i = 0; i < columns; ++i) {
    input[i] = std::sin(float(i + 1) * 0.013f) * 0.75f +
               std::cos(float((i % 17) + 3) * 0.11f) * 0.15f;
  }
  const std::size_t blocks = columns / kQk;
  Q8Input result;
  result.values.resize(columns);
  result.sums.resize(blocks * 16);
  result.scales.resize(blocks);
  for (std::size_t block = 0; block < blocks; ++block) {
    float maximum = 0.0f;
    float absolute_maximum = 0.0f;
    for (std::size_t i = 0; i < kQk; ++i) {
      const float value = input[block * kQk + i];
      if (std::abs(value) > absolute_maximum) {
        absolute_maximum = std::abs(value);
        maximum = value;
      }
    }
    if (absolute_maximum == 0.0f) continue;
    const float inverse_scale = -127.0f / maximum;
    result.scales[block] = 1.0f / inverse_scale;
    for (std::size_t i = 0; i < kQk; ++i) {
      const int quantized =
          std::min(127, NearestInt(inverse_scale * input[block * kQk + i]));
      result.values[block * kQk + i] = static_cast<std::int8_t>(quantized);
    }
    for (std::size_t group = 0; group < 16; ++group) {
      int sum = 0;
      for (std::size_t i = 0; i < 16; ++i) {
        sum += result.values[block * kQk + group * 16 + i];
      }
      result.sums[block * 16 + group] = static_cast<std::int16_t>(sum);
    }
  }
  return result;
}

int Q6Value(const std::uint8_t* block, std::size_t index) {
  const std::size_t half = index / 128;
  const std::size_t within = index % 128;
  const std::size_t quadrant = within / 32;
  const std::size_t lane = within % 32;
  const auto high = block[128 + half * 32 + lane];
  int low = 0;
  int high_bits = 0;
  if (quadrant == 0) {
    low = block[half * 64 + lane] & 15;
    high_bits = (high >> 0) & 3;
  } else if (quadrant == 1) {
    low = block[half * 64 + 32 + lane] & 15;
    high_bits = (high >> 2) & 3;
  } else if (quadrant == 2) {
    low = block[half * 64 + lane] >> 4;
    high_bits = (high >> 4) & 3;
  } else {
    low = block[half * 64 + 32 + lane] >> 4;
    high_bits = (high >> 6) & 3;
  }
  return (low | (high_bits << 4)) - 32;
}

std::vector<float> CpuReference(const std::vector<std::uint8_t>& raw,
                                std::size_t rows,
                                std::size_t blocks_per_row,
                                const Q8Input& input) {
  std::vector<float> output(rows);
  const std::size_t row_bytes = blocks_per_row * kQ6BlockBytes;
  for (std::size_t row = 0; row < rows; ++row) {
    const auto* row_raw = raw.data() + row * row_bytes;
    float total = 0.0f;
    for (std::size_t block_index = 0; block_index < blocks_per_row;
         ++block_index) {
      const auto* block = row_raw + block_index * kQ6BlockBytes;
      std::array<std::int32_t, 8> lanes{};
      const auto* scales = reinterpret_cast<const std::int8_t*>(block + 192);
      for (std::size_t k = 0; k < kQk; ++k) {
        const int group = static_cast<int>(k / 16);
        lanes[k & 7] += int(scales[group]) *
                        int(input.values[block_index * kQk + k]) *
                        Q6Value(block, k);
      }
      const float scale =
          HalfToFloat(LoadU16(block + 208)) * input.scales[block_index];
      for (std::int32_t lane : lanes) total += scale * float(lane);
    }
    output[row] = total;
  }
  return output;
}

struct Comparison {
  double maximum_absolute_difference = 0.0;
  std::size_t maximum_difference_index = 0;
  double relative_l2 = 0.0;
  double cosine = 0.0;
  double rmse = 0.0;
  bool passed = false;
};

Comparison Compare(const std::vector<float>& candidate,
                   const std::vector<float>& reference) {
  Require(candidate.size() == reference.size(), "comparison size mismatch");
  Comparison result;
  double delta_squared = 0.0;
  double candidate_squared = 0.0;
  double reference_squared = 0.0;
  double dot = 0.0;
  for (std::size_t i = 0; i < candidate.size(); ++i) {
    Require(std::isfinite(candidate[i]) && std::isfinite(reference[i]),
            "comparison contains non-finite output");
    const double delta = double(candidate[i]) - reference[i];
    const double absolute = std::abs(delta);
    if (absolute > result.maximum_absolute_difference) {
      result.maximum_absolute_difference = absolute;
      result.maximum_difference_index = i;
    }
    delta_squared += delta * delta;
    candidate_squared += double(candidate[i]) * candidate[i];
    reference_squared += double(reference[i]) * reference[i];
    dot += double(candidate[i]) * reference[i];
  }
  result.relative_l2 = std::sqrt(delta_squared / reference_squared);
  result.cosine = dot / std::sqrt(candidate_squared * reference_squared);
  result.rmse = std::sqrt(delta_squared / candidate.size());
  result.passed = result.relative_l2 <= 2e-6 && result.cosine >= 0.999999;
  return result;
}

std::string JsonEscape(const std::string& value) {
  std::string result;
  for (char character : value) {
    if (character == '\\') result += "\\\\";
    else if (character == '"') result += "\\\"";
    else if (character == '\n') result += "\\n";
    else if (character == '\r') result += "\\r";
    else if (character == '\t') result += "\\t";
    else result += character;
  }
  return result;
}

struct Args {
  std::string model;
  std::string tensor = "blk.7.ffn_down_exps.weight";
  std::string kernel_source;
  std::string device_substring = "B390";
  int repeat = 7;
};

Args ParseArgs(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    auto value = [&]() {
      Require(i + 1 < argc, "missing value after " + key);
      return std::string(argv[++i]);
    };
    if (key == "--model") args.model = value();
    else if (key == "--tensor") args.tensor = value();
    else if (key == "--kernel-source") args.kernel_source = value();
    else if (key == "--device-substring") args.device_substring = value();
    else if (key == "--repeat") args.repeat = std::stoi(value());
    else Die("unknown argument: " + key);
  }
  Require(!args.model.empty(), "--model is required");
  Require(!args.kernel_source.empty(), "--kernel-source is required");
  Require(args.repeat > 0, "--repeat must be positive");
  return args;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto args = ParseArgs(argc, argv);
    const auto index = iq36::parse_gguf_model_index(args.model);
    const auto* tensor = iq36::find_tensor(index, args.tensor);
    Require(tensor != nullptr && tensor->type == 14 && tensor->dims.size() >= 2,
            "selected tensor is not a Q6_K matrix");
    const std::size_t columns = static_cast<std::size_t>(tensor->dims[0]);
    std::size_t rows = 1;
    for (std::size_t i = 1; i < tensor->dims.size(); ++i) {
      rows *= static_cast<std::size_t>(tensor->dims[i]);
    }
    Require(columns % kQk == 0 && rows % 16 == 0,
            "selected tensor shape does not satisfy kernel contract");
    const std::size_t blocks_per_row = columns / kQk;
    Require(tensor->nbytes == rows * blocks_per_row * kQ6BlockBytes,
            "Q6 tensor byte count mismatch");
    const auto raw = ReadTensor(args.model, *tensor);
    const auto q8 = QuantizeInput(columns);
    const auto reference = CpuReference(raw, rows, blocks_per_row, q8);

    OpenClApi api;
    const auto selected = SelectDevice(api, args.device_substring);
    cl_int status = kClSuccess;
    cl_context context = api.clCreateContext(
        nullptr, 1, &selected.device, nullptr, nullptr, &status);
    Check(status, "clCreateContext");
    cl_command_queue queue = api.clCreateCommandQueue(
        context, selected.device, kClQueueProfilingEnable, &status);
    Check(status, "clCreateCommandQueue");
    const auto source = ReadText(args.kernel_source);
    const char* source_pointer = source.c_str();
    const std::size_t source_size = source.size();
    cl_program program = api.clCreateProgramWithSource(
        context, 1, &source_pointer, &source_size, &status);
    Check(status, "clCreateProgramWithSource");
    status = api.clBuildProgram(program, 1, &selected.device,
                                "-cl-std=CL3.0", nullptr, nullptr);
    std::size_t log_size = 0;
    api.clGetProgramBuildInfo(program, selected.device, kClProgramBuildLog,
                              0, nullptr, &log_size);
    std::string build_log(log_size, '\0');
    if (log_size > 0) {
      api.clGetProgramBuildInfo(program, selected.device, kClProgramBuildLog,
                                log_size, build_log.data(), nullptr);
      while (!build_log.empty() && build_log.back() == '\0') build_log.pop_back();
    }
    Check(status, "clBuildProgram");
    cl_kernel kernel = api.clCreateKernel(
        program, "q6k_splitplane_dpas_rowtile16", &status);
    Check(status, "clCreateKernel");

    auto create = [&](cl_mem_flags flags, std::size_t bytes, void* data,
                      const char* label) {
      cl_mem buffer = api.clCreateBuffer(context, flags, bytes, data, &status);
      Check(status, label);
      return buffer;
    };
    cl_mem raw_buffer = create(kClMemReadOnly | kClMemCopyHostPtr,
                               raw.size(), const_cast<std::uint8_t*>(raw.data()),
                               "clCreateBuffer(raw)");
    cl_mem values_buffer = create(
        kClMemReadOnly | kClMemCopyHostPtr, q8.values.size(),
        const_cast<std::int8_t*>(q8.values.data()), "clCreateBuffer(q8 values)");
    cl_mem sums_buffer = create(
        kClMemReadOnly | kClMemCopyHostPtr,
        q8.sums.size() * sizeof(std::int16_t),
        const_cast<std::int16_t*>(q8.sums.data()), "clCreateBuffer(q8 sums)");
    cl_mem scales_buffer = create(
        kClMemReadOnly | kClMemCopyHostPtr,
        q8.scales.size() * sizeof(float),
        const_cast<float*>(q8.scales.data()), "clCreateBuffer(q8 scales)");
    std::vector<float> output(rows);
    cl_mem output_buffer = create(kClMemWriteOnly,
                                  output.size() * sizeof(float), nullptr,
                                  "clCreateBuffer(output)");
    const cl_uint rows_arg = static_cast<cl_uint>(rows);
    const cl_uint blocks_arg = static_cast<cl_uint>(blocks_per_row);
    Check(api.clSetKernelArg(kernel, 0, sizeof(raw_buffer), &raw_buffer), "arg0");
    Check(api.clSetKernelArg(kernel, 1, sizeof(values_buffer), &values_buffer), "arg1");
    Check(api.clSetKernelArg(kernel, 2, sizeof(sums_buffer), &sums_buffer), "arg2");
    Check(api.clSetKernelArg(kernel, 3, sizeof(scales_buffer), &scales_buffer), "arg3");
    Check(api.clSetKernelArg(kernel, 4, sizeof(rows_arg), &rows_arg), "arg4");
    Check(api.clSetKernelArg(kernel, 5, sizeof(blocks_arg), &blocks_arg), "arg5");
    Check(api.clSetKernelArg(kernel, 6, sizeof(output_buffer), &output_buffer), "arg6");

    const std::size_t global = rows;
    const std::size_t local = 16;
    std::vector<double> samples;
    for (int iteration = 0; iteration < args.repeat + 2; ++iteration) {
      cl_event event = nullptr;
      Check(api.clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global,
                                       &local, 0, nullptr, &event),
            "clEnqueueNDRangeKernel");
      Check(api.clFinish(queue), "clFinish");
      cl_ulong begin = 0;
      cl_ulong end = 0;
      Check(api.clGetEventProfilingInfo(event, kClProfilingCommandStart,
                                        sizeof(begin), &begin, nullptr),
            "clGetEventProfilingInfo(start)");
      Check(api.clGetEventProfilingInfo(event, kClProfilingCommandEnd,
                                        sizeof(end), &end, nullptr),
            "clGetEventProfilingInfo(end)");
      api.clReleaseEvent(event);
      if (iteration >= 2) samples.push_back(double(end - begin) / 1000.0);
    }
    Check(api.clEnqueueReadBuffer(queue, output_buffer, kClTrue, 0,
                                  output.size() * sizeof(float), output.data(),
                                  0, nullptr, nullptr),
          "clEnqueueReadBuffer");
    const auto comparison = Compare(output, reference);
    const double minimum_us = *std::min_element(samples.begin(), samples.end());
    double mean_us = 0.0;
    for (double sample : samples) mean_us += sample;
    mean_us /= samples.size();
    const double effective_gb_s = double(raw.size()) / minimum_us / 1000.0;

    api.clReleaseMemObject(output_buffer);
    api.clReleaseMemObject(scales_buffer);
    api.clReleaseMemObject(sums_buffer);
    api.clReleaseMemObject(values_buffer);
    api.clReleaseMemObject(raw_buffer);
    api.clReleaseKernel(kernel);
    api.clReleaseProgram(program);
    api.clReleaseCommandQueue(queue);
    api.clReleaseContext(context);

    std::cout << std::setprecision(17);
    std::cout << "{";
    std::cout << "\"schema_version\":\"intel-qwen36-q6-splitplane-dpas-component-v0\",";
    std::cout << "\"platform_name\":\"" << JsonEscape(selected.platform_name) << "\",";
    std::cout << "\"device_name\":\"" << JsonEscape(selected.device_name) << "\",";
    std::cout << "\"build_log\":\"" << JsonEscape(build_log) << "\",";
    std::cout << "\"tensor_name\":\"" << JsonEscape(args.tensor) << "\",";
    std::cout << "\"rows\":" << rows << ",";
    std::cout << "\"columns\":" << columns << ",";
    std::cout << "\"blocks_per_row\":" << blocks_per_row << ",";
    std::cout << "\"payload_bytes\":" << raw.size() << ",";
    std::cout << "\"persistent_expansion_bytes\":0,";
    std::cout << "\"repeat\":" << args.repeat << ",";
    std::cout << "\"kernel_min_us\":" << minimum_us << ",";
    std::cout << "\"kernel_mean_us\":" << mean_us << ",";
    std::cout << "\"effective_source_gb_s\":" << effective_gb_s << ",";
    std::cout << "\"comparison\":{";
    std::cout << "\"maximum_absolute_difference\":"
              << comparison.maximum_absolute_difference << ",";
    std::cout << "\"maximum_difference_index\":"
              << comparison.maximum_difference_index << ",";
    std::cout << "\"relative_l2\":" << comparison.relative_l2 << ",";
    std::cout << "\"cosine\":" << comparison.cosine << ",";
    std::cout << "\"rmse\":" << comparison.rmse << ",";
    std::cout << "\"passed\":" << (comparison.passed ? "true" : "false") << "}";
    std::cout << "}" << std::endl;
    return comparison.passed ? 0 : 2;
  } catch (const std::exception& error) {
    std::cerr << "q6-splitplane-dpas-gate: " << error.what() << "\n";
    return 1;
  }
}
