#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
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
  std::string binary;
  std::string input;
  std::string weights;
  std::string scales;
  std::string zps;
  std::string oracle;
  std::string actual;
  std::string kernel = "iq36_moe_micro_layer0_qkv";
  int m = 8192;
  int n = 1;
  int k = 2048;
  int quant_group = 64;
  int sg_per_wg_m = 1;
  int sg_per_wg_n = 1;
  int sg_per_wg_k = 1;
  int wg_tile_m = 1;
  int wg_tile_n = 1;
  int warmup = 8;
  int repeat = 31;
  double minimum_gbps = 94.2;
  double minimum_tmac_s = 0.0;
  bool performance_only = false;
};

Args ParseArgs(int argc, char** argv) {
  Args args;
  for (int index = 1; index < argc; ++index) {
    const std::string option = argv[index];
    auto value = [&]() {
      if (++index >= argc) Fail(option + " requires a value");
      return std::string(argv[index]);
    };
    if (option == "--performance-only") {
      args.performance_only = true;
    } else if (option == "--binary") args.binary = value();
    else if (option == "--input") args.input = value();
    else if (option == "--weights") args.weights = value();
    else if (option == "--scales") args.scales = value();
    else if (option == "--zps") args.zps = value();
    else if (option == "--oracle") args.oracle = value();
    else if (option == "--actual") args.actual = value();
    else if (option == "--kernel") args.kernel = value();
    else if (option == "--m") args.m = std::stoi(value());
    else if (option == "--n") args.n = std::stoi(value());
    else if (option == "--k") args.k = std::stoi(value());
    else if (option == "--quant-group") args.quant_group = std::stoi(value());
    else if (option == "--sg-per-wg-m") args.sg_per_wg_m = std::stoi(value());
    else if (option == "--sg-per-wg-n") args.sg_per_wg_n = std::stoi(value());
    else if (option == "--sg-per-wg-k") args.sg_per_wg_k = std::stoi(value());
    else if (option == "--wg-tile-m") args.wg_tile_m = std::stoi(value());
    else if (option == "--wg-tile-n") args.wg_tile_n = std::stoi(value());
    else if (option == "--warmup") args.warmup = std::stoi(value());
    else if (option == "--repeat") args.repeat = std::stoi(value());
    else if (option == "--minimum-gbps") {
      args.minimum_gbps = std::stod(value());
    } else if (option == "--minimum-tmac-s") {
      args.minimum_tmac_s = std::stod(value());
    } else {
      Fail("unknown option: " + option);
    }
  }
  Require(!args.binary.empty() && !args.input.empty() &&
              !args.weights.empty() && !args.scales.empty() &&
              !args.zps.empty() &&
              (args.performance_only || !args.oracle.empty()),
          "binary, input, weight, scale, zero-point, and numeric oracle paths are required");
  Require(args.m > 0 && args.n > 0 && args.k > 0 && args.quant_group > 0 &&
              args.k % args.quant_group == 0 && args.sg_per_wg_m > 0 &&
              args.sg_per_wg_n > 0 && args.sg_per_wg_k > 0 &&
              args.wg_tile_m > 0 && args.wg_tile_n > 0 &&
              args.warmup > 0 && args.repeat >= 5 &&
              args.minimum_gbps > 0.0 && args.minimum_tmac_s >= 0.0,
          "numeric arguments are invalid");
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
            "byte-size mismatch for " + path + ": observed " +
                std::to_string(size) + ", expected " +
                std::to_string(expected));
  }
  input.seekg(0, std::ios::beg);
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
  input.read(reinterpret_cast<char*>(bytes.data()),
             static_cast<std::streamsize>(bytes.size()));
  Require(static_cast<bool>(input), "could not read " + path);
  return bytes;
}

void WriteBytes(const std::string& path,
                const std::vector<std::uint8_t>& bytes) {
  if (path.empty()) return;
  std::ofstream output(path, std::ios::binary);
  Require(static_cast<bool>(output), "could not create " + path);
  output.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
  Require(static_cast<bool>(output), "could not write " + path);
}

std::string ProgramLog(cl_program program, cl_device_id device) {
  std::size_t size = 0;
  Check(clGetProgramBuildInfo(
            program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &size),
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
  cl_program program = nullptr;
  cl_kernel kernel = nullptr;
  std::vector<cl_mem> memories;

  ~OpenClState() {
    for (cl_mem memory : memories) clReleaseMemObject(memory);
    if (kernel != nullptr) clReleaseKernel(kernel);
    if (program != nullptr) clReleaseProgram(program);
    if (queue != nullptr) clReleaseCommandQueue(queue);
    if (context != nullptr) clReleaseContext(context);
  }
};

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

OpenClState CreateOpenCl(const Args& args) {
  OpenClState state;
  cl_uint platform_count = 0;
  Check(clGetPlatformIDs(0, nullptr, &platform_count),
        "clGetPlatformIDs count");
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
      if (DeviceString(device, CL_DEVICE_NAME).find("B390") !=
          std::string::npos) {
        state.device = device;
        break;
      }
    }
    if (state.device != nullptr) break;
  }
  Require(state.device != nullptr, "B390 OpenCL device not found");
  cl_int status = CL_SUCCESS;
  state.context = clCreateContext(
      nullptr, 1, &state.device, nullptr, nullptr, &status);
  Check(status, "clCreateContext");
  state.queue = clCreateCommandQueue(
      state.context, state.device, CL_QUEUE_PROFILING_ENABLE, &status);
  Check(status, "clCreateCommandQueue");

  const auto binary = ReadBytes(args.binary);
  const std::size_t binary_size = binary.size();
  const unsigned char* binary_data = binary.data();
  cl_int binary_status = CL_SUCCESS;
  state.program = clCreateProgramWithBinary(
      state.context, 1, &state.device, &binary_size, &binary_data,
      &binary_status, &status);
  Check(status, "clCreateProgramWithBinary");
  Check(binary_status, "program binary status");
  status = clBuildProgram(state.program, 1, &state.device, "", nullptr, nullptr);
  if (status != CL_SUCCESS) {
    Fail("clBuildProgram failed: " + ProgramLog(state.program, state.device));
  }
  state.kernel = clCreateKernel(state.program, args.kernel.c_str(), &status);
  Check(status, "clCreateKernel");
  return state;
}

cl_mem CreateCopied(OpenClState& state,
                    const std::vector<std::uint8_t>& bytes) {
  cl_int status = CL_SUCCESS;
  cl_mem memory = clCreateBuffer(
      state.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes.size(),
      const_cast<std::uint8_t*>(bytes.data()), &status);
  Check(status, "clCreateBuffer copied");
  state.memories.push_back(memory);
  return memory;
}

cl_mem CreateOutput(OpenClState& state, std::size_t bytes) {
  cl_int status = CL_SUCCESS;
  cl_mem memory = clCreateBuffer(
      state.context, CL_MEM_WRITE_ONLY, bytes, nullptr, &status);
  Check(status, "clCreateBuffer output");
  state.memories.push_back(memory);
  return memory;
}

void SetMem(cl_kernel kernel, cl_uint index, cl_mem memory) {
  Check(clSetKernelArg(kernel, index, sizeof(memory), &memory),
        "clSetKernelArg memory " + std::to_string(index));
}

double Median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  const std::size_t middle = values.size() / 2;
  return values.size() % 2 == 0
      ? (values[middle - 1] + values[middle]) * 0.5
      : values[middle];
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

struct Comparison {
  bool finite = true;
  std::size_t exact = 0;
  double cosine = 0.0;
  double relative_l2 = 0.0;
  double max_abs = 0.0;
  double rmse = 0.0;
};

Comparison Compare(const std::vector<std::uint8_t>& actual,
                   const std::vector<std::uint8_t>& reference) {
  Require(actual.size() == reference.size() && actual.size() % 2 == 0,
          "output comparison size mismatch");
  Comparison result;
  long double dot = 0.0;
  long double actual_norm = 0.0;
  long double reference_norm = 0.0;
  long double error_norm = 0.0;
  for (std::size_t offset = 0; offset < actual.size(); offset += 2) {
    std::uint16_t lhs_bits = 0;
    std::uint16_t rhs_bits = 0;
    std::memcpy(&lhs_bits, actual.data() + offset, sizeof(lhs_bits));
    std::memcpy(&rhs_bits, reference.data() + offset, sizeof(rhs_bits));
    if (lhs_bits == rhs_bits) ++result.exact;
    const double lhs = HalfToFloat(lhs_bits);
    const double rhs = HalfToFloat(rhs_bits);
    const double delta = lhs - rhs;
    result.finite = result.finite && std::isfinite(lhs) && std::isfinite(rhs);
    result.max_abs = std::max(result.max_abs, std::abs(delta));
    dot += lhs * rhs;
    actual_norm += lhs * lhs;
    reference_norm += rhs * rhs;
    error_norm += delta * delta;
  }
  if (reference_norm == 0.0L) {
    result.cosine = actual_norm == 0.0L ? 1.0 : 0.0;
    result.relative_l2 = error_norm == 0.0L
        ? 0.0
        : std::numeric_limits<double>::infinity();
  } else {
    result.cosine = static_cast<double>(
        dot / std::sqrt(actual_norm * reference_norm));
    result.relative_l2 = static_cast<double>(
        std::sqrt(error_norm / reference_norm));
  }
  result.rmse = std::sqrt(static_cast<double>(
      error_norm / (actual.size() / sizeof(std::uint16_t))));
  return result;
}

int Main(int argc, char** argv) {
  const Args args = ParseArgs(argc, argv);
  const std::size_t input_bytes =
      static_cast<std::size_t>(args.n) * args.k * sizeof(std::uint16_t);
  const std::size_t weight_bytes =
      static_cast<std::size_t>(args.m) * args.k / 2;
  const std::size_t scale_bytes =
      static_cast<std::size_t>(args.m) * (args.k / args.quant_group) *
      sizeof(std::uint16_t);
  const std::size_t zp_bytes = scale_bytes / 4;
  const std::size_t output_bytes =
      static_cast<std::size_t>(args.n) * args.m * sizeof(std::uint16_t);

  const auto input = ReadBytes(args.input, input_bytes);
  const auto weights = ReadBytes(args.weights, weight_bytes);
  const auto scales = ReadBytes(args.scales, scale_bytes);
  const auto zps = ReadBytes(args.zps, zp_bytes);
  const auto oracle = args.performance_only
      ? std::vector<std::uint8_t>{}
      : ReadBytes(args.oracle, output_bytes);
  OpenClState state = CreateOpenCl(args);
  cl_uint register_count = 0;
  cl_ulong spill_memory_bytes = 0;
  Check(clGetKernelWorkGroupInfo(
            state.kernel, state.device, CL_KERNEL_REGISTER_COUNT_INTEL,
            sizeof(register_count), &register_count, nullptr),
        "clGetKernelWorkGroupInfo register count");
  Check(clGetKernelWorkGroupInfo(
            state.kernel, state.device, CL_KERNEL_SPILL_MEM_SIZE_INTEL,
            sizeof(spill_memory_bytes), &spill_memory_bytes, nullptr),
        "clGetKernelWorkGroupInfo spill memory");
  const cl_mem input_mem = CreateCopied(state, input);
  const cl_mem weight_mem = CreateCopied(state, weights);
  const cl_mem output_mem = CreateOutput(state, output_bytes);
  const cl_mem scale_mem = CreateCopied(state, scales);
  const cl_mem zp_mem = CreateCopied(state, zps);
  SetMem(state.kernel, 0, input_mem);
  SetMem(state.kernel, 1, weight_mem);
  SetMem(state.kernel, 2, output_mem);
  SetMem(state.kernel, 3, scale_mem);
  SetMem(state.kernel, 4, zp_mem);
  Check(clSetKernelArg(state.kernel, 5, sizeof(args.m), &args.m),
        "clSetKernelArg m");
  Check(clSetKernelArg(state.kernel, 6, sizeof(args.k), &args.k),
        "clSetKernelArg k");
  Check(clSetKernelArg(state.kernel, 7, sizeof(args.n), &args.n),
        "clSetKernelArg n");

  const std::size_t local[3] = {
      static_cast<std::size_t>(args.sg_per_wg_m) * 16,
      static_cast<std::size_t>(args.sg_per_wg_n),
      static_cast<std::size_t>(args.sg_per_wg_k),
  };
  const std::size_t group_m =
      (static_cast<std::size_t>(args.m) + args.wg_tile_m - 1) /
      args.wg_tile_m;
  const std::size_t group_n =
      (static_cast<std::size_t>(args.n) + args.wg_tile_n - 1) /
      args.wg_tile_n;
  const std::size_t global[3] = {
      group_m * local[0], group_n * local[1], local[2],
  };

  auto enqueue = [&]() {
    cl_event event = nullptr;
    Check(clEnqueueNDRangeKernel(state.queue, state.kernel, 3, nullptr,
                                 global, local, 0, nullptr, &event),
          "clEnqueueNDRangeKernel");
    Check(clWaitForEvents(1, &event), "clWaitForEvents");
    return event;
  };
  for (int index = 0; index < args.warmup; ++index) {
    cl_event event = enqueue();
    clReleaseEvent(event);
  }
  std::vector<double> samples;
  samples.reserve(static_cast<std::size_t>(args.repeat));
  for (int index = 0; index < args.repeat; ++index) {
    cl_event event = enqueue();
    cl_ulong start = 0;
    cl_ulong end = 0;
    Check(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START,
                                  sizeof(start), &start, nullptr),
          "clGetEventProfilingInfo start");
    Check(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,
                                  sizeof(end), &end, nullptr),
          "clGetEventProfilingInfo end");
    Require(end >= start, "negative profiling interval");
    samples.push_back(static_cast<double>(end - start) / 1000.0);
    clReleaseEvent(event);
  }
  std::vector<std::uint8_t> actual(output_bytes);
  Check(clEnqueueReadBuffer(state.queue, output_mem, CL_TRUE, 0,
                            actual.size(), actual.data(), 0, nullptr, nullptr),
        "clEnqueueReadBuffer output");
  WriteBytes(args.actual, actual);
  const Comparison comparison = args.performance_only
      ? Comparison{}
      : Compare(actual, oracle);
  const double median_us = Median(samples);
  const std::size_t parameter_bytes = weight_bytes + scale_bytes + zp_bytes;
  const double parameter_gbps = parameter_bytes / (median_us * 1000.0);
  const double tmac_s =
      (static_cast<double>(args.m) * args.n * args.k) /
      (median_us * 1.0e6);
  const bool correctness = args.performance_only ||
      (comparison.finite && comparison.cosine >= 0.999 &&
       comparison.relative_l2 <= 0.002);
  const bool performance = args.minimum_tmac_s > 0.0
      ? tmac_s >= args.minimum_tmac_s
      : parameter_gbps >= args.minimum_gbps;

  std::cout.precision(12);
  std::cout << "{\"schema_version\":\"intel-qwen36-openvino-fc-micro-runtime-v0\"";
  std::cout << ",\"device_name\":\""
            << DeviceString(state.device, CL_DEVICE_NAME) << "\"";
  std::cout << ",\"driver_version\":\""
            << DeviceString(state.device, CL_DRIVER_VERSION) << "\"";
  std::cout << ",\"m\":" << args.m << ",\"n\":" << args.n
            << ",\"k\":" << args.k;
  std::cout << ",\"quant_group_size\":" << args.quant_group;
  std::cout << ",\"register_count\":" << register_count;
  std::cout << ",\"spill_memory_bytes\":" << spill_memory_bytes;
  std::cout << ",\"parameter_bytes\":" << parameter_bytes;
  std::cout << ",\"kernel_median_us\":" << median_us;
  std::cout << ",\"parameter_gbps\":" << parameter_gbps;
  std::cout << ",\"tmac_s\":" << tmac_s;
  std::cout << ",\"minimum_gbps\":" << args.minimum_gbps;
  std::cout << ",\"minimum_tmac_s\":" << args.minimum_tmac_s;
  std::cout << ",\"global\":[" << global[0] << ',' << global[1] << ','
            << global[2] << "]";
  std::cout << ",\"local\":[" << local[0] << ',' << local[1] << ','
            << local[2] << "]";
  std::cout << ",\"samples_us\":[";
  for (std::size_t index = 0; index < samples.size(); ++index) {
    if (index != 0) std::cout << ',';
    std::cout << samples[index];
  }
  const std::size_t values = output_bytes / sizeof(std::uint16_t);
  std::cout << "],\"correctness_evaluated\":"
            << (args.performance_only ? "false" : "true");
  std::cout << ",\"compare\":{";
  std::cout << "\"finite\":" << (comparison.finite ? "true" : "false");
  std::cout << ",\"exact_values\":" << comparison.exact;
  std::cout << ",\"exact_rate\":"
            << (args.performance_only ? 0.0 :
                static_cast<double>(comparison.exact) / values);
  std::cout << ",\"cosine\":" << comparison.cosine;
  std::cout << ",\"relative_l2\":" << comparison.relative_l2;
  std::cout << ",\"max_abs_diff\":" << comparison.max_abs;
  std::cout << ",\"rmse\":" << comparison.rmse << '}';
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
