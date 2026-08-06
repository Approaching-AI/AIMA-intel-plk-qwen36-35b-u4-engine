#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
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

std::vector<std::string> Split(const std::string& value) {
  std::vector<std::string> result;
  std::size_t begin = 0;
  while (begin <= value.size()) {
    const std::size_t end = value.find(',', begin);
    const std::string part = value.substr(
        begin, end == std::string::npos ? std::string::npos : end - begin);
    Require(!part.empty(), "comma-separated argument contains an empty value");
    result.push_back(part);
    if (end == std::string::npos) break;
    begin = end + 1;
  }
  return result;
}

std::vector<int> SplitInts(const std::string& value) {
  std::vector<int> result;
  for (const std::string& part : Split(value)) result.push_back(std::stoi(part));
  return result;
}

struct Args {
  std::string baseline_binary;
  std::string candidate_binary;
  std::string input;
  std::string baseline_weights;
  std::string baseline_scales;
  std::string baseline_zps;
  std::vector<std::string> weights;
  std::vector<std::string> scales;
  std::vector<std::string> zps;
  std::vector<int> widths;
  std::string reference;
  std::string actual_prefix;
  std::string kernel;
  int k = 2048;
  int n = 1;
  int quant_group = 64;
  int sg_per_wg_m = 2;
  int sg_per_wg_n = 1;
  int sg_per_wg_k = 8;
  int wg_tile_m = 64;
  int wg_tile_n = 8;
  int warmup = 512;
  int repeat = 31;
  int blocks = 8;
  bool allow_baseline_difference = false;
  bool candidate_single = false;
  bool baseline_zp_u8 = false;
  bool candidate_zp_u8 = false;
};

Args ParseArgs(int argc, char** argv) {
  Args args;
  for (int index = 1; index < argc; ++index) {
    const std::string option = argv[index];
    auto value = [&]() {
      if (++index >= argc) Fail(option + " requires a value");
      return std::string(argv[index]);
    };
    if (option == "--allow-baseline-difference") {
      args.allow_baseline_difference = true;
    } else if (option == "--candidate-single") {
      args.candidate_single = true;
    } else if (option == "--baseline-zp-u8") {
      args.baseline_zp_u8 = true;
    } else if (option == "--candidate-zp-u8") {
      args.candidate_zp_u8 = true;
    } else if (option == "--baseline-binary") args.baseline_binary = value();
    else if (option == "--candidate-binary") args.candidate_binary = value();
    else if (option == "--input") args.input = value();
    else if (option == "--baseline-weights") args.baseline_weights = value();
    else if (option == "--baseline-scales") args.baseline_scales = value();
    else if (option == "--baseline-zps") args.baseline_zps = value();
    else if (option == "--weights") args.weights = Split(value());
    else if (option == "--scales") args.scales = Split(value());
    else if (option == "--zps") args.zps = Split(value());
    else if (option == "--widths") args.widths = SplitInts(value());
    else if (option == "--reference") args.reference = value();
    else if (option == "--actual-prefix") args.actual_prefix = value();
    else if (option == "--kernel") args.kernel = value();
    else if (option == "--k") args.k = std::stoi(value());
    else if (option == "--n") args.n = std::stoi(value());
    else if (option == "--quant-group") args.quant_group = std::stoi(value());
    else if (option == "--sg-per-wg-m") args.sg_per_wg_m = std::stoi(value());
    else if (option == "--sg-per-wg-n") args.sg_per_wg_n = std::stoi(value());
    else if (option == "--sg-per-wg-k") args.sg_per_wg_k = std::stoi(value());
    else if (option == "--wg-tile-m") args.wg_tile_m = std::stoi(value());
    else if (option == "--wg-tile-n") args.wg_tile_n = std::stoi(value());
    else if (option == "--warmup") args.warmup = std::stoi(value());
    else if (option == "--repeat") args.repeat = std::stoi(value());
    else if (option == "--blocks") args.blocks = std::stoi(value());
    else Fail("unknown option: " + option);
  }
  Require(!args.baseline_binary.empty() && !args.candidate_binary.empty() &&
              !args.input.empty() && !args.baseline_weights.empty() &&
              !args.baseline_scales.empty() && !args.baseline_zps.empty() &&
              !args.kernel.empty(),
          "binary, input, fused stream, and kernel arguments are required");
  Require(!args.widths.empty() && args.widths.size() <= 4 &&
              args.weights.size() == args.widths.size() &&
              args.scales.size() == args.widths.size() &&
              args.zps.size() == args.widths.size(),
          "width and per-projection stream counts must match in [1,4]");
  Require(!args.candidate_single || args.widths.size() == 1,
          "candidate-single requires exactly one fused width");
  Require(!args.candidate_zp_u8 || args.candidate_single,
          "candidate-zp-u8 requires candidate-single");
  Require(std::all_of(args.widths.begin(), args.widths.end(),
                      [](int width) { return width > 0; }) &&
              args.k > 0 && args.quant_group > 0 &&
              args.n > 0 && args.k % args.quant_group == 0 &&
              args.sg_per_wg_m > 0 &&
              args.sg_per_wg_n > 0 && args.sg_per_wg_k > 0 &&
              args.wg_tile_m > 0 && args.wg_tile_n > 0 &&
              args.warmup > 0 && args.repeat >= 5 && args.blocks >= 8,
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
  std::vector<cl_program> programs;
  std::vector<cl_kernel> kernels;
  std::vector<cl_mem> memories;

  ~OpenClState() {
    for (cl_mem memory : memories) clReleaseMemObject(memory);
    for (cl_kernel kernel : kernels) clReleaseKernel(kernel);
    for (cl_program program : programs) clReleaseProgram(program);
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

void InitializeOpenCl(OpenClState& state) {
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
}

cl_kernel LoadKernel(OpenClState& state, const std::string& binary_path,
                     const std::string& kernel_name) {
  const auto binary = ReadBytes(binary_path);
  const std::size_t binary_size = binary.size();
  const unsigned char* binary_data = binary.data();
  cl_int status = CL_SUCCESS;
  cl_int binary_status = CL_SUCCESS;
  cl_program program = clCreateProgramWithBinary(
      state.context, 1, &state.device, &binary_size, &binary_data,
      &binary_status, &status);
  Check(status, "clCreateProgramWithBinary");
  Check(binary_status, "program binary status");
  status = clBuildProgram(program, 1, &state.device, "", nullptr, nullptr);
  if (status != CL_SUCCESS) {
    const std::string log = ProgramLog(program, state.device);
    clReleaseProgram(program);
    Fail("clBuildProgram failed: " + log);
  }
  cl_kernel kernel = clCreateKernel(program, kernel_name.c_str(), &status);
  if (status != CL_SUCCESS) {
    clReleaseProgram(program);
    Check(status, "clCreateKernel");
  }
  state.programs.push_back(program);
  state.kernels.push_back(kernel);
  return kernel;
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

void PrintComparison(const Comparison& comparison, std::size_t values) {
  std::cout << "{\"finite\":" << (comparison.finite ? "true" : "false");
  std::cout << ",\"exact_values\":" << comparison.exact;
  std::cout << ",\"exact_rate\":"
            << static_cast<double>(comparison.exact) / values;
  std::cout << ",\"cosine\":" << comparison.cosine;
  std::cout << ",\"relative_l2\":" << comparison.relative_l2;
  std::cout << ",\"max_abs_diff\":" << comparison.max_abs;
  std::cout << ",\"rmse\":" << comparison.rmse << '}';
}

struct KernelInfo {
  cl_uint registers = 0;
  cl_ulong spill_bytes = 0;
};

KernelInfo QueryKernel(cl_kernel kernel, cl_device_id device) {
  KernelInfo result;
  Check(clGetKernelWorkGroupInfo(
            kernel, device, CL_KERNEL_REGISTER_COUNT_INTEL,
            sizeof(result.registers), &result.registers, nullptr),
        "clGetKernelWorkGroupInfo register count");
  Check(clGetKernelWorkGroupInfo(
            kernel, device, CL_KERNEL_SPILL_MEM_SIZE_INTEL,
            sizeof(result.spill_bytes), &result.spill_bytes, nullptr),
        "clGetKernelWorkGroupInfo spill memory");
  return result;
}

double ProfileOnce(OpenClState& state, cl_kernel kernel,
                   const std::array<std::size_t, 3>& global,
                   const std::array<std::size_t, 3>& local) {
  cl_event event = nullptr;
  Check(clEnqueueNDRangeKernel(state.queue, kernel, 3, nullptr,
                               global.data(), local.data(), 0, nullptr, &event),
        "clEnqueueNDRangeKernel");
  Check(clWaitForEvents(1, &event), "clWaitForEvents");
  cl_ulong start = 0;
  cl_ulong end = 0;
  Check(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START,
                                sizeof(start), &start, nullptr),
        "clGetEventProfilingInfo start");
  Check(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,
                                sizeof(end), &end, nullptr),
        "clGetEventProfilingInfo end");
  clReleaseEvent(event);
  Require(end >= start, "negative profiling interval");
  return static_cast<double>(end - start) / 1000.0;
}

std::vector<double> RunLeg(OpenClState& state, cl_kernel kernel,
                           const std::array<std::size_t, 3>& global,
                           const std::array<std::size_t, 3>& local,
                           int repeat) {
  std::vector<double> result;
  result.reserve(static_cast<std::size_t>(repeat));
  for (int index = 0; index < repeat; ++index) {
    result.push_back(ProfileOnce(state, kernel, global, local));
  }
  return result;
}

struct Block {
  std::array<double, 4> leg_medians{};
  double baseline_us = 0.0;
  double candidate_us = 0.0;
  double delta_us = 0.0;
  bool reversed = false;
};

int Main(int argc, char** argv) {
  const Args args = ParseArgs(argc, argv);
  int total_m = 0;
  for (int width : args.widths) total_m += width;
  const std::size_t input_bytes =
      static_cast<std::size_t>(args.n) * args.k * sizeof(std::uint16_t);
  const std::size_t weight_bytes =
      static_cast<std::size_t>(total_m) * args.k / 2;
  const std::size_t scale_bytes =
      static_cast<std::size_t>(total_m) * (args.k / args.quant_group) *
      sizeof(std::uint16_t);
  const std::size_t zp_elements =
      static_cast<std::size_t>(total_m) * (args.k / args.quant_group);
  const std::size_t baseline_zp_bytes =
      args.baseline_zp_u8 ? zp_elements : zp_elements / 2;
  const std::size_t candidate_zp_bytes =
      args.candidate_zp_u8 ? zp_elements : zp_elements / 2;
  const std::size_t output_bytes =
      static_cast<std::size_t>(args.n) * total_m * sizeof(std::uint16_t);

  const auto input = ReadBytes(args.input, input_bytes);
  const auto baseline_weights = ReadBytes(args.baseline_weights, weight_bytes);
  const auto baseline_scales = ReadBytes(args.baseline_scales, scale_bytes);
  const auto baseline_zps = ReadBytes(
      args.baseline_zps, baseline_zp_bytes);
  std::vector<std::vector<std::uint8_t>> weights;
  std::vector<std::vector<std::uint8_t>> scales;
  std::vector<std::vector<std::uint8_t>> zps;
  for (std::size_t index = 0; index < args.widths.size(); ++index) {
    const std::size_t width = static_cast<std::size_t>(args.widths[index]);
    weights.push_back(ReadBytes(args.weights[index], width * args.k / 2));
    scales.push_back(ReadBytes(
        args.scales[index], width * (args.k / args.quant_group) * 2));
    const std::size_t zp_elements =
        width * static_cast<std::size_t>(args.k / args.quant_group);
    zps.push_back(ReadBytes(
        args.zps[index],
        args.candidate_zp_u8 ? zp_elements : zp_elements / 2));
  }

  OpenClState state;
  InitializeOpenCl(state);
  const cl_kernel baseline_kernel = LoadKernel(
      state, args.baseline_binary, args.kernel);
  const cl_kernel candidate_kernel = LoadKernel(
      state, args.candidate_binary, args.kernel);
  const KernelInfo baseline_info = QueryKernel(baseline_kernel, state.device);
  const KernelInfo candidate_info = QueryKernel(candidate_kernel, state.device);

  const cl_mem input_mem = CreateCopied(state, input);
  const cl_mem baseline_weight_mem = CreateCopied(state, baseline_weights);
  const cl_mem baseline_output_mem = CreateOutput(state, output_bytes);
  const cl_mem baseline_scale_mem = CreateCopied(state, baseline_scales);
  const cl_mem baseline_zp_mem = CreateCopied(state, baseline_zps);
  SetMem(baseline_kernel, 0, input_mem);
  SetMem(baseline_kernel, 1, baseline_weight_mem);
  SetMem(baseline_kernel, 2, baseline_output_mem);
  SetMem(baseline_kernel, 3, baseline_scale_mem);
  SetMem(baseline_kernel, 4, baseline_zp_mem);
  Check(clSetKernelArg(baseline_kernel, 5, sizeof(total_m), &total_m),
        "clSetKernelArg baseline m");
  Check(clSetKernelArg(baseline_kernel, 6, sizeof(args.k), &args.k),
        "clSetKernelArg baseline k");
  Check(clSetKernelArg(baseline_kernel, 7, sizeof(args.n), &args.n),
        "clSetKernelArg baseline n");

  std::vector<cl_mem> weight_memories;
  std::vector<cl_mem> output_memories;
  std::vector<cl_mem> scale_memories;
  std::vector<cl_mem> zp_memories;
  for (std::size_t index = 0; index < args.widths.size(); ++index) {
    weight_memories.push_back(CreateCopied(state, weights[index]));
    output_memories.push_back(CreateOutput(
        state, static_cast<std::size_t>(args.n) * args.widths[index] * 2));
    scale_memories.push_back(CreateCopied(state, scales[index]));
    zp_memories.push_back(CreateCopied(state, zps[index]));
  }
  SetMem(candidate_kernel, 0, input_mem);
  if (args.candidate_single) {
    SetMem(candidate_kernel, 1, weight_memories[0]);
    SetMem(candidate_kernel, 2, output_memories[0]);
    SetMem(candidate_kernel, 3, scale_memories[0]);
    SetMem(candidate_kernel, 4, zp_memories[0]);
    Check(clSetKernelArg(candidate_kernel, 5, sizeof(total_m), &total_m),
          "clSetKernelArg candidate m");
    Check(clSetKernelArg(candidate_kernel, 6, sizeof(args.k), &args.k),
          "clSetKernelArg candidate k");
    Check(clSetKernelArg(candidate_kernel, 7, sizeof(args.n), &args.n),
          "clSetKernelArg candidate n");
  } else {
    for (std::size_t slot = 0; slot < 4; ++slot) {
      const std::size_t source = slot < args.widths.size() ? slot : 0;
      const cl_uint base = static_cast<cl_uint>(1 + slot * 4);
      SetMem(candidate_kernel, base, weight_memories[source]);
      SetMem(candidate_kernel, base + 1, output_memories[source]);
      SetMem(candidate_kernel, base + 2, scale_memories[source]);
      SetMem(candidate_kernel, base + 3, zp_memories[source]);
    }
    Check(clSetKernelArg(candidate_kernel, 17, sizeof(args.k), &args.k),
          "clSetKernelArg candidate k");
    Check(clSetKernelArg(candidate_kernel, 18, sizeof(args.n), &args.n),
          "clSetKernelArg candidate n");
  }

  const std::array<std::size_t, 3> local = {
      static_cast<std::size_t>(args.sg_per_wg_m) * 16,
      static_cast<std::size_t>(args.sg_per_wg_n),
      static_cast<std::size_t>(args.sg_per_wg_k),
  };
  const std::size_t baseline_groups =
      (static_cast<std::size_t>(total_m) + args.wg_tile_m - 1) /
      args.wg_tile_m;
  std::size_t candidate_groups = 0;
  for (int width : args.widths) {
    candidate_groups +=
        (static_cast<std::size_t>(width) + args.wg_tile_m - 1) /
        args.wg_tile_m;
  }
  const std::size_t group_n =
      (static_cast<std::size_t>(args.n) + args.wg_tile_n - 1) /
      args.wg_tile_n;
  const std::array<std::size_t, 3> baseline_global = {
      baseline_groups * local[0], group_n * local[1], local[2]};
  const std::array<std::size_t, 3> candidate_global = {
      candidate_groups * local[0], group_n * local[1], local[2]};

  for (int index = 0; index < args.warmup; ++index) {
    if ((index & 1) == 0) {
      ProfileOnce(state, baseline_kernel, baseline_global, local);
      ProfileOnce(state, candidate_kernel, candidate_global, local);
    } else {
      ProfileOnce(state, candidate_kernel, candidate_global, local);
      ProfileOnce(state, baseline_kernel, baseline_global, local);
    }
  }

  std::vector<Block> blocks;
  blocks.reserve(static_cast<std::size_t>(args.blocks));
  for (int block_index = 0; block_index < args.blocks; ++block_index) {
    Block block;
    block.reversed = (block_index & 1) != 0;
    const std::array<int, 4> order = block.reversed
        ? std::array<int, 4>{1, 0, 0, 1}
        : std::array<int, 4>{0, 1, 1, 0};
    std::vector<double> baseline_samples;
    std::vector<double> candidate_samples;
    for (std::size_t leg = 0; leg < order.size(); ++leg) {
      const bool candidate = order[leg] == 1;
      std::vector<double> samples = RunLeg(
          state, candidate ? candidate_kernel : baseline_kernel,
          candidate ? candidate_global : baseline_global, local, args.repeat);
      block.leg_medians[leg] = Median(samples);
      std::vector<double>& destination =
          candidate ? candidate_samples : baseline_samples;
      destination.insert(destination.end(), samples.begin(), samples.end());
    }
    block.baseline_us = Median(baseline_samples);
    block.candidate_us = Median(candidate_samples);
    block.delta_us = block.candidate_us - block.baseline_us;
    blocks.push_back(block);
  }

  std::vector<std::uint8_t> baseline_token_major(output_bytes);
  Check(clEnqueueReadBuffer(state.queue, baseline_output_mem, CL_TRUE, 0,
                            baseline_token_major.size(),
                            baseline_token_major.data(),
                            0, nullptr, nullptr),
        "clEnqueueReadBuffer baseline output");
  std::vector<std::uint8_t> baseline_actual;
  baseline_actual.reserve(output_bytes);
  std::size_t width_begin = 0;
  for (int width : args.widths) {
    const std::size_t projection_bytes =
        static_cast<std::size_t>(args.n) * width * 2;
    const std::size_t old_size = baseline_actual.size();
    baseline_actual.resize(old_size + projection_bytes);
    for (int token = 0; token < args.n; ++token) {
      const std::size_t source =
          (static_cast<std::size_t>(token) * total_m + width_begin) * 2;
      const std::size_t destination =
          old_size + static_cast<std::size_t>(token) * width * 2;
      std::copy_n(baseline_token_major.begin() + source,
                  static_cast<std::size_t>(width) * 2,
                  baseline_actual.begin() + destination);
    }
    width_begin += static_cast<std::size_t>(width);
  }
  std::vector<std::uint8_t> candidate_actual;
  candidate_actual.reserve(output_bytes);
  for (std::size_t index = 0; index < output_memories.size(); ++index) {
    std::vector<std::uint8_t> output(
        static_cast<std::size_t>(args.n) * args.widths[index] * 2);
    Check(clEnqueueReadBuffer(state.queue, output_memories[index], CL_TRUE, 0,
                              output.size(), output.data(), 0, nullptr, nullptr),
          "clEnqueueReadBuffer candidate output");
    candidate_actual.insert(
        candidate_actual.end(), output.begin(), output.end());
  }
  WriteBytes(args.actual_prefix.empty()
                 ? std::string()
                 : args.actual_prefix + ".baseline.f16",
             baseline_actual);
  WriteBytes(args.actual_prefix.empty()
                 ? std::string()
                 : args.actual_prefix + ".candidate.f16",
             candidate_actual);

  const Comparison equivalence = Compare(candidate_actual, baseline_actual);
  const bool equivalent = equivalence.finite &&
      equivalence.exact == static_cast<std::size_t>(args.n) * total_m;
  bool reference_pass = true;
  Comparison baseline_reference;
  Comparison candidate_reference;
  if (!args.reference.empty()) {
    const auto reference = ReadBytes(args.reference, output_bytes);
    baseline_reference = Compare(baseline_actual, reference);
    candidate_reference = Compare(candidate_actual, reference);
    reference_pass = candidate_reference.finite &&
        candidate_reference.cosine >= 0.999 &&
        candidate_reference.relative_l2 <= 0.002 &&
        (args.allow_baseline_difference ||
         (baseline_reference.finite && baseline_reference.cosine >= 0.999 &&
          baseline_reference.relative_l2 <= 0.002));
  }
  const bool correctness =
      (equivalent || args.allow_baseline_difference) && reference_pass;
  std::vector<double> baseline_block_us;
  std::vector<double> candidate_block_us;
  for (const Block& block : blocks) {
    baseline_block_us.push_back(block.baseline_us);
    candidate_block_us.push_back(block.candidate_us);
  }
  const double baseline_median_us = Median(baseline_block_us);
  const double candidate_median_us = Median(candidate_block_us);
  const std::size_t baseline_parameter_bytes =
      weight_bytes + scale_bytes + baseline_zp_bytes;
  const std::size_t candidate_parameter_bytes =
      weight_bytes + scale_bytes + candidate_zp_bytes;

  std::cout.precision(12);
  std::cout << "{\"schema_version\":\"intel-qwen36-openvino-fc-multi-output-runtime-v0\"";
  std::cout << ",\"device_name\":\""
            << DeviceString(state.device, CL_DEVICE_NAME) << "\"";
  std::cout << ",\"driver_version\":\""
            << DeviceString(state.device, CL_DRIVER_VERSION) << "\"";
  std::cout << ",\"m\":" << total_m << ",\"n\":" << args.n
            << ",\"k\":" << args.k;
  std::cout << ",\"widths\":[";
  for (std::size_t index = 0; index < args.widths.size(); ++index) {
    if (index != 0) std::cout << ',';
    std::cout << args.widths[index];
  }
  std::cout << "]";
  std::cout << ",\"quant_group_size\":" << args.quant_group;
  std::cout << ",\"parameter_bytes\":" << baseline_parameter_bytes;
  std::cout << ",\"candidate_parameter_bytes\":"
            << candidate_parameter_bytes;
  std::cout << ",\"baseline_register_count\":" << baseline_info.registers;
  std::cout << ",\"candidate_register_count\":" << candidate_info.registers;
  std::cout << ",\"baseline_spill_memory_bytes\":"
            << baseline_info.spill_bytes;
  std::cout << ",\"candidate_spill_memory_bytes\":"
            << candidate_info.spill_bytes;
  std::cout << ",\"baseline_kernel_median_us\":" << baseline_median_us;
  std::cout << ",\"candidate_kernel_median_us\":" << candidate_median_us;
  std::cout << ",\"baseline_parameter_gbps\":"
            << baseline_parameter_bytes / (baseline_median_us * 1000.0);
  std::cout << ",\"candidate_parameter_gbps\":"
            << candidate_parameter_bytes / (candidate_median_us * 1000.0);
  std::cout << ",\"baseline_global\":[" << baseline_global[0] << ','
            << baseline_global[1] << ',' << baseline_global[2] << ']';
  std::cout << ",\"candidate_global\":[" << candidate_global[0] << ','
            << candidate_global[1] << ',' << candidate_global[2] << ']';
  std::cout << ",\"local\":[" << local[0] << ',' << local[1] << ','
            << local[2] << ']';
  std::cout << ",\"baseline_equivalence_required\":"
            << (args.allow_baseline_difference ? "false" : "true");
  std::cout << ",\"candidate_single\":"
            << (args.candidate_single ? "true" : "false");
  std::cout << ",\"baseline_zero_point_type\":\""
            << (args.baseline_zp_u8 ? "u8" : "u4") << "\"";
  std::cout << ",\"candidate_zero_point_type\":\""
            << (args.candidate_zp_u8 ? "u8" : "u4") << "\"";
  std::cout << ",\"blocks\":[";
  for (std::size_t index = 0; index < blocks.size(); ++index) {
    if (index != 0) std::cout << ',';
    const Block& block = blocks[index];
    std::cout << "{\"index\":" << index;
    std::cout << ",\"schedule\":\""
              << (block.reversed ? "BAAB" : "ABBA") << "\"";
    std::cout << ",\"leg_medians_us\":[";
    for (std::size_t leg = 0; leg < block.leg_medians.size(); ++leg) {
      if (leg != 0) std::cout << ',';
      std::cout << block.leg_medians[leg];
    }
    std::cout << "]";
    std::cout << ",\"baseline_us\":" << block.baseline_us;
    std::cout << ",\"candidate_us\":" << block.candidate_us;
    std::cout << ",\"delta_us\":" << block.delta_us << '}';
  }
  std::cout << "],\"baseline_candidate_compare\":";
  PrintComparison(equivalence, static_cast<std::size_t>(args.n) * total_m);
  if (!args.reference.empty()) {
    std::cout << ",\"baseline_reference_compare\":";
    PrintComparison(
        baseline_reference, static_cast<std::size_t>(args.n) * total_m);
    std::cout << ",\"candidate_reference_compare\":";
    PrintComparison(
        candidate_reference, static_cast<std::size_t>(args.n) * total_m);
  }
  std::cout << ",\"correctness_pass\":"
            << (correctness ? "true" : "false") << "}\n";
  return correctness ? 0 : 2;
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
