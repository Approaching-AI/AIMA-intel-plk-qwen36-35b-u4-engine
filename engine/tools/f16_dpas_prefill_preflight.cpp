#include <CL/cl.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kTokenCount = 1024;
constexpr std::size_t kHeadDim = 128;
constexpr std::size_t kKeyHeads = 16;
constexpr std::size_t kTileCount = 65536;
constexpr std::size_t kM = 8;
constexpr std::size_t kN = 16;
constexpr std::size_t kK = 128;
constexpr std::uint64_t kMacs = kTileCount * kM * kN * kK;
constexpr double kMinimumTmacPerSecond = 4.0;
constexpr double kCosineMinimum = 0.999;
constexpr double kRelativeL2Maximum = 0.002;

struct Args {
  std::filesystem::path kernel;
  std::filesystem::path capture;
  std::filesystem::path dump_binary;
  int warmup = 20;
  int repeat = 21;
  double minimum_tmac_per_second = kMinimumTmacPerSecond;
};

[[noreturn]] void Fail(const std::string& message) {
  throw std::runtime_error(message);
}

void Require(bool condition, const std::string& message) {
  if (!condition) Fail(message);
}

void Check(cl_int status, const std::string& where) {
  if (status != CL_SUCCESS) {
    Fail(where + " failed with OpenCL status " + std::to_string(status));
  }
}

Args ParseArgs(int argc, char** argv) {
  Args args;
  for (int index = 1; index < argc; ++index) {
    const std::string option = argv[index];
    const auto Value = [&]() -> std::string {
      if (++index >= argc) Fail(option + " requires a value");
      return argv[index];
    };
    if (option == "--kernel") args.kernel = Value();
    else if (option == "--capture") args.capture = Value();
    else if (option == "--dump-binary") args.dump_binary = Value();
    else if (option == "--warmup") args.warmup = std::stoi(Value());
    else if (option == "--repeat") args.repeat = std::stoi(Value());
    else if (option == "--minimum-tmac-per-second") {
      args.minimum_tmac_per_second = std::stod(Value());
    } else {
      Fail("unknown option: " + option);
    }
  }
  Require(!args.kernel.empty() && !args.capture.empty(),
          "kernel and capture paths are required");
  Require(args.warmup > 0 && args.repeat > 0 &&
              args.minimum_tmac_per_second > 0.0,
          "warmup, repeat, and minimum rate must be positive");
  return args;
}

std::string ReadText(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  Require(static_cast<bool>(input), "could not open " + path.string());
  input.seekg(0, std::ios::end);
  const auto size = input.tellg();
  Require(size >= 0, "could not size " + path.string());
  input.seekg(0, std::ios::beg);
  std::string text(static_cast<std::size_t>(size), '\0');
  input.read(text.data(), static_cast<std::streamsize>(text.size()));
  Require(static_cast<bool>(input), "could not read " + path.string());
  return text;
}

std::string JsonStringField(const std::string& line,
                            const std::string& field) {
  const std::string marker = "\"" + field + "\":\"";
  const auto begin = line.find(marker);
  if (begin == std::string::npos) return {};
  const auto value_begin = begin + marker.size();
  const auto end = line.find('"', value_begin);
  return end == std::string::npos
      ? std::string{} : line.substr(value_begin, end - value_begin);
}

std::filesystem::path FindPayload(const std::filesystem::path& capture,
                                  const std::string& name) {
  std::ifstream input(capture / "tensor-dumps.jsonl");
  Require(static_cast<bool>(input), "could not open capture index");
  std::string line;
  while (std::getline(input, line)) {
    if (JsonStringField(line, "tensor_name") == name) {
      const auto relative = JsonStringField(line, "payload_path");
      Require(!relative.empty(), "capture payload path is empty");
      return capture / relative;
    }
  }
  Fail("capture is missing " + name);
}

std::vector<float> ReadFloats(const std::filesystem::path& path,
                              std::size_t count) {
  std::ifstream input(path, std::ios::binary);
  Require(static_cast<bool>(input), "could not open " + path.string());
  std::vector<float> values(count);
  input.read(reinterpret_cast<char*>(values.data()),
             static_cast<std::streamsize>(values.size() * sizeof(float)));
  Require(static_cast<bool>(input), "could not read " + path.string());
  input.peek();
  Require(input.eof(), "payload size mismatch for " + path.string());
  return values;
}

std::uint16_t FloatToHalf(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const std::uint32_t sign = (bits >> 16U) & 0x8000U;
  const std::uint32_t exponent_bits = (bits >> 23U) & 0xffU;
  std::uint32_t mantissa = bits & 0x7fffffU;
  if (exponent_bits == 0xffU) {
    return static_cast<std::uint16_t>(
        sign | 0x7c00U | (mantissa == 0U ? 0U : 0x0200U));
  }
  int exponent = static_cast<int>(exponent_bits) - 127 + 15;
  if (exponent <= 0) {
    if (exponent < -10) return static_cast<std::uint16_t>(sign);
    mantissa |= 0x800000U;
    const int shift = 14 - exponent;
    const std::uint32_t rounded =
        (mantissa + ((UINT32_C(1) << (shift - 1)) - 1U) +
         ((mantissa >> shift) & 1U)) >> shift;
    return static_cast<std::uint16_t>(sign | rounded);
  }
  if (exponent >= 31) return static_cast<std::uint16_t>(sign | 0x7c00U);
  mantissa += 0x00000fffU + ((mantissa >> 13U) & 1U);
  if ((mantissa & 0x00800000U) != 0U) {
    mantissa = 0U;
    ++exponent;
    if (exponent >= 31) {
      return static_cast<std::uint16_t>(sign | 0x7c00U);
    }
  }
  return static_cast<std::uint16_t>(
      sign | (static_cast<std::uint32_t>(exponent) << 10U) |
      (mantissa >> 13U));
}

struct DeviceSelection {
  cl_device_id device = nullptr;
  std::string name;
  std::string driver;
};

std::string DeviceString(cl_device_id device, cl_device_info field) {
  std::size_t bytes = 0;
  Check(clGetDeviceInfo(device, field, 0, nullptr, &bytes),
        "clGetDeviceInfo size");
  std::string value(bytes, '\0');
  Check(clGetDeviceInfo(device, field, bytes, value.data(), nullptr),
        "clGetDeviceInfo value");
  if (!value.empty() && value.back() == '\0') value.pop_back();
  return value;
}

DeviceSelection SelectDevice() {
  cl_uint platform_count = 0;
  Check(clGetPlatformIDs(0, nullptr, &platform_count),
        "clGetPlatformIDs count");
  std::vector<cl_platform_id> platforms(platform_count);
  Check(clGetPlatformIDs(platform_count, platforms.data(), nullptr),
        "clGetPlatformIDs");
  for (const auto platform : platforms) {
    cl_uint count = 0;
    if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &count) !=
            CL_SUCCESS || count == 0) continue;
    std::vector<cl_device_id> devices(count);
    Check(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, count, devices.data(),
                         nullptr), "clGetDeviceIDs");
    for (const auto device : devices) {
      DeviceSelection result{device, DeviceString(device, CL_DEVICE_NAME),
                             DeviceString(device, CL_DRIVER_VERSION)};
      if (result.name.find("Arc") != std::string::npos) return result;
    }
  }
  Fail("no Arc OpenCL GPU found");
}

struct OpenClState {
  cl_context context = nullptr;
  cl_command_queue queue = nullptr;
  cl_program program = nullptr;
  cl_kernel kernel = nullptr;
  std::vector<cl_mem> buffers;
  ~OpenClState() {
    for (const auto buffer : buffers) clReleaseMemObject(buffer);
    if (kernel) clReleaseKernel(kernel);
    if (program) clReleaseProgram(program);
    if (queue) clReleaseCommandQueue(queue);
    if (context) clReleaseContext(context);
  }
};

cl_mem Buffer(OpenClState& state, std::size_t bytes, cl_mem_flags flags,
              void* host = nullptr) {
  cl_int status = CL_SUCCESS;
  const auto result =
      clCreateBuffer(state.context, flags, bytes, host, &status);
  Check(status, "clCreateBuffer");
  state.buffers.push_back(result);
  return result;
}

void SetMem(cl_kernel kernel, cl_uint index, cl_mem value) {
  Check(clSetKernelArg(kernel, index, sizeof(value), &value),
        "clSetKernelArg");
}

double EventUs(cl_event event, cl_profiling_info field) {
  cl_ulong value = 0;
  Check(clGetEventProfilingInfo(event, field, sizeof(value), &value, nullptr),
        "clGetEventProfilingInfo");
  return static_cast<double>(value) / 1000.0;
}

double Median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

void DumpProgramBinary(cl_program program,
                       const std::filesystem::path& path) {
  if (path.empty()) return;
  std::size_t size = 0;
  Check(clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(size),
                         &size, nullptr), "clGetProgramInfo size");
  std::vector<unsigned char> binary(size);
  unsigned char* pointer = binary.data();
  Check(clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof(pointer),
                         &pointer, nullptr), "clGetProgramInfo binary");
  std::ofstream output(path, std::ios::binary);
  output.write(reinterpret_cast<const char*>(binary.data()),
               static_cast<std::streamsize>(binary.size()));
  Require(static_cast<bool>(output), "could not write program binary");
}

bool MapsForbiddenRuntime() {
  std::ifstream maps("/proc/self/maps");
  std::string line;
  while (std::getline(maps, line)) {
    std::transform(line.begin(), line.end(), line.begin(),
                   [](unsigned char value) {
                     return static_cast<char>(std::tolower(value));
                   });
    if (line.find("openvino") != std::string::npos ||
        line.find("libdnnl") != std::string::npos ||
        line.find("onednn") != std::string::npos) return true;
  }
  return false;
}

}  // namespace

int main(int argc, char** argv) {
  std::cout << std::setprecision(12);
  try {
    const auto args = ParseArgs(argc, argv);
    const auto q = ReadFloats(FindPayload(args.capture, "q_conv_predelta-0"),
                              kTokenCount * kKeyHeads * kHeadDim);
    const auto k = ReadFloats(FindPayload(args.capture, "k_conv_predelta-0"),
                              kTokenCount * kKeyHeads * kHeadDim);
    std::vector<std::uint16_t> a(kM * kK);
    std::vector<std::uint16_t> b(kK * kN);
    std::vector<float> reference(kM * kN, 0.0f);
    for (std::size_t row = 0; row < kM; ++row) {
      for (std::size_t depth = 0; depth < kK; ++depth) {
        a[row * kK + depth] =
            FloatToHalf(q[(row * kKeyHeads) * kHeadDim + depth]);
      }
    }
    for (std::size_t depth = 0; depth < kK; ++depth) {
      for (std::size_t column = 0; column < kN; ++column) {
        b[depth * kN + column] = FloatToHalf(
            k[(column * kKeyHeads) * kHeadDim + depth]);
      }
    }
    for (std::size_t row = 0; row < kM; ++row) {
      for (std::size_t column = 0; column < kN; ++column) {
        float sum = 0.0f;
        for (std::size_t depth = 0; depth < kK; ++depth) {
          sum = std::fma(q[(row * kKeyHeads) * kHeadDim + depth],
                         k[(column * kKeyHeads) * kHeadDim + depth], sum);
        }
        reference[row * kN + column] = sum;
      }
    }

    const auto device = SelectDevice();
    OpenClState state;
    cl_int status = CL_SUCCESS;
    state.context =
        clCreateContext(nullptr, 1, &device.device, nullptr, nullptr, &status);
    Check(status, "clCreateContext");
    const cl_queue_properties properties[] = {
        CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
    state.queue = clCreateCommandQueueWithProperties(
        state.context, device.device, properties, &status);
    Check(status, "clCreateCommandQueueWithProperties");
    const auto source = ReadText(args.kernel);
    const char* source_pointer = source.c_str();
    const std::size_t source_size = source.size();
    state.program = clCreateProgramWithSource(
        state.context, 1, &source_pointer, &source_size, &status);
    Check(status, "clCreateProgramWithSource");
    const char* options = "-cl-std=CL3.0 -cl-fast-relaxed-math";
    status = clBuildProgram(state.program, 1, &device.device, options,
                            nullptr, nullptr);
    if (status != CL_SUCCESS) {
      std::size_t log_size = 0;
      clGetProgramBuildInfo(state.program, device.device, CL_PROGRAM_BUILD_LOG,
                            0, nullptr, &log_size);
      std::string log(log_size, '\0');
      clGetProgramBuildInfo(state.program, device.device, CL_PROGRAM_BUILD_LOG,
                            log.size(), log.data(), nullptr);
      Fail("OpenCL build failed: " + log);
    }
    DumpProgramBinary(state.program, args.dump_binary);
    state.kernel = clCreateKernel(
        state.program, "iq36_f16_dpas_8x16x128_preflight", &status);
    Check(status, "clCreateKernel");
    const auto a_buffer = Buffer(
        state, a.size() * sizeof(a[0]),
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, a.data());
    const auto b_buffer = Buffer(
        state, b.size() * sizeof(b[0]),
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, b.data());
    const auto checksum_buffer = Buffer(
        state, kTileCount * sizeof(float), CL_MEM_WRITE_ONLY);
    const auto output_buffer = Buffer(
        state, kM * kN * sizeof(float), CL_MEM_WRITE_ONLY);
    SetMem(state.kernel, 0, a_buffer);
    SetMem(state.kernel, 1, b_buffer);
    SetMem(state.kernel, 2, checksum_buffer);
    SetMem(state.kernel, 3, output_buffer);

    std::vector<double> samples_us;
    const std::size_t global = kTileCount * kN;
    const std::size_t local = kN;
    for (int iteration = -args.warmup; iteration < args.repeat; ++iteration) {
      cl_event event = nullptr;
      Check(clEnqueueNDRangeKernel(state.queue, state.kernel, 1, nullptr,
                                   &global, &local, 0, nullptr, &event),
            "clEnqueueNDRangeKernel");
      Check(clFinish(state.queue), "clFinish");
      if (iteration >= 0) {
        samples_us.push_back(EventUs(event, CL_PROFILING_COMMAND_END) -
                             EventUs(event, CL_PROFILING_COMMAND_START));
      }
      clReleaseEvent(event);
    }
    std::vector<float> observed(kM * kN);
    Check(clEnqueueReadBuffer(state.queue, output_buffer, CL_TRUE, 0,
                              observed.size() * sizeof(float), observed.data(),
                              0, nullptr, nullptr), "clEnqueueReadBuffer");

    long double diff_sq = 0.0L;
    long double observed_sq = 0.0L;
    long double reference_sq = 0.0L;
    long double dot = 0.0L;
    double max_abs = 0.0;
    for (std::size_t index = 0; index < observed.size(); ++index) {
      const double lhs = observed[index];
      const double rhs = reference[index];
      const double difference = lhs - rhs;
      max_abs = std::max(max_abs, std::abs(difference));
      diff_sq += difference * difference;
      observed_sq += lhs * lhs;
      reference_sq += rhs * rhs;
      dot += lhs * rhs;
    }
    const double relative_l2 =
        std::sqrt(static_cast<double>(diff_sq / reference_sq));
    const double cosine = static_cast<double>(
        dot / std::sqrt(observed_sq * reference_sq));
    const double median_us = Median(samples_us);
    const double tmac_per_second =
        static_cast<double>(kMacs) / (median_us * 1.0e6);
    const bool forbidden_runtime_mapped = MapsForbiddenRuntime();
    const bool numeric_passes = relative_l2 <= kRelativeL2Maximum &&
                                cosine >= kCosineMinimum;
    const bool passed = numeric_passes &&
                        tmac_per_second >= args.minimum_tmac_per_second &&
                        !forbidden_runtime_mapped;

    std::cout << "{\"schema_version\":\"intel-qwen36-f16-dpas-prefill-"
                 "preflight-v0\""
              << ",\"device_name\":\"" << device.name << "\""
              << ",\"driver_version\":\"" << device.driver << "\""
              << ",\"tile_count\":" << kTileCount
              << ",\"m\":" << kM << ",\"n\":" << kN
              << ",\"k\":" << kK << ",\"macs\":" << kMacs
              << ",\"timed_host_upload_bytes\":0"
              << ",\"timed_host_read_bytes\":0"
              << ",\"warmup\":" << args.warmup
              << ",\"repeat\":" << args.repeat
              << ",\"minimum_tmac_per_second\":"
              << args.minimum_tmac_per_second
              << ",\"samples_us\":[";
    for (std::size_t index = 0; index < samples_us.size(); ++index) {
      if (index) std::cout << ',';
      std::cout << samples_us[index];
    }
    std::cout << "]"
              << ",\"median_us\":" << median_us
              << ",\"tmac_per_second\":" << tmac_per_second
              << ",\"comparison\":{\"count\":" << observed.size()
              << ",\"max_abs\":" << max_abs
              << ",\"relative_l2\":" << relative_l2
              << ",\"cosine\":" << cosine
              << ",\"passes\":"
              << (numeric_passes ? "true" : "false") << "}"
              << ",\"forbidden_runtime_mapped\":"
              << (forbidden_runtime_mapped ? "true" : "false")
              << ",\"required_checks_passed\":"
              << (passed ? "true" : "false") << "}\n";
    return passed ? 0 : 2;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
