#include <CL/cl.h>

#include "intel_qwen36/gguf_loader.hpp"

#include <algorithm>
#include <array>
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
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kTokenCount = 1024;
constexpr std::size_t kHeadDim = 128;
constexpr std::size_t kKeyHeads = 16;
constexpr std::size_t kValueHeads = 32;
constexpr std::size_t kQkValues = kTokenCount * kKeyHeads * kHeadDim;
constexpr std::size_t kValueValues =
    kTokenCount * kValueHeads * kHeadDim;
constexpr std::size_t kControlValues = kTokenCount * kValueHeads;
constexpr std::size_t kStateValues =
    kValueHeads * kHeadDim * kHeadDim;
constexpr std::size_t kRecurrentWorkgroups = kValueHeads * 8;
constexpr std::size_t kF16RecurrentWorkgroups = kValueHeads * 16;
constexpr std::size_t kChunkSize = 64;
constexpr std::size_t kChunkCount = kTokenCount / kChunkSize;
constexpr std::size_t kChunkHeadWorkgroups = kChunkCount * kValueHeads;
constexpr std::size_t kChunkStateValues = kChunkCount * kStateValues;
constexpr std::size_t kNormWorkgroups = kTokenCount * kValueHeads;
constexpr double kNormEpsilon = 1.0e-6;
constexpr double kCosineMinimum = 0.999;
constexpr double kRelativeL2Maximum = 0.002;

struct Args {
  std::filesystem::path model;
  std::filesystem::path kernel;
  std::filesystem::path capture;
  std::filesystem::path dump_binary;
  std::string storage = "f32";
  int layer = 0;
  int warmup = 20;
  int repeat = 21;
  double cap_us = 407.0;
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
    if (option == "--model") args.model = Value();
    else if (option == "--kernel") args.kernel = Value();
    else if (option == "--capture") args.capture = Value();
    else if (option == "--dump-binary") args.dump_binary = Value();
    else if (option == "--storage") args.storage = Value();
    else if (option == "--layer") args.layer = std::stoi(Value());
    else if (option == "--warmup") args.warmup = std::stoi(Value());
    else if (option == "--repeat") args.repeat = std::stoi(Value());
    else if (option == "--cap-us") args.cap_us = std::stod(Value());
    else Fail("unknown option: " + option);
  }
  Require(!args.model.empty() && !args.kernel.empty() && !args.capture.empty(),
          "model, kernel, and capture paths are required");
  Require(args.layer >= 0 && args.layer < 40,
          "layer must be in [0, 39]");
  Require(args.storage == "f32" || args.storage == "f16" ||
              args.storage == "chunk64",
          "storage must be f32, f16, or chunk64");
  Require(args.warmup > 0 && args.repeat > 0 && args.cap_us > 0.0,
          "warmup, repeat, and cap must be positive");
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
  if (end == std::string::npos) return {};
  return line.substr(value_begin, end - value_begin);
}

std::map<std::string, std::filesystem::path> CapturePayloads(
    const std::filesystem::path& capture) {
  const auto index_path = capture / "tensor-dumps.jsonl";
  std::ifstream input(index_path);
  Require(static_cast<bool>(input),
          "could not open capture index " + index_path.string());
  std::map<std::string, std::filesystem::path> paths;
  std::string line;
  while (std::getline(input, line)) {
    const auto name = JsonStringField(line, "tensor_name");
    const auto relative = JsonStringField(line, "payload_path");
    if (!name.empty() && !relative.empty()) {
      paths.emplace(name, capture / relative);
    }
  }
  return paths;
}

template <typename Value>
std::vector<Value> ReadVector(const std::filesystem::path& path,
                              std::size_t expected_count) {
  std::ifstream input(path, std::ios::binary);
  Require(static_cast<bool>(input), "could not open " + path.string());
  input.seekg(0, std::ios::end);
  const auto bytes = input.tellg();
  Require(bytes == static_cast<std::streamoff>(
                       expected_count * sizeof(Value)),
          "payload size mismatch for " + path.string());
  input.seekg(0, std::ios::beg);
  std::vector<Value> values(expected_count);
  input.read(reinterpret_cast<char*>(values.data()),
             static_cast<std::streamsize>(bytes));
  Require(static_cast<bool>(input), "could not read " + path.string());
  return values;
}

std::vector<float> ReadStridedV(const std::filesystem::path& path) {
  constexpr std::size_t kSourceTokenStride = 8192;
  constexpr std::size_t kValuesPerToken = kValueHeads * kHeadDim;
  constexpr std::size_t kSourceCount =
      (kTokenCount - 1) * kSourceTokenStride + kValuesPerToken;
  const auto source = ReadVector<float>(path, kSourceCount);
  std::vector<float> compact(kValueValues);
  for (std::size_t token = 0; token < kTokenCount; ++token) {
    std::copy_n(source.begin() + token * kSourceTokenStride,
                kValuesPerToken,
                compact.begin() + token * kValuesPerToken);
  }
  return compact;
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
  if (exponent >= 31) {
    return static_cast<std::uint16_t>(sign | 0x7c00U);
  }
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

float HalfToFloat(std::uint16_t half) {
  const std::uint32_t sign =
      static_cast<std::uint32_t>(half & 0x8000U) << 16U;
  std::uint32_t exponent = (half >> 10U) & 0x1fU;
  std::uint32_t mantissa = half & 0x03ffU;
  std::uint32_t bits = 0;
  if (exponent == 0U) {
    if (mantissa == 0U) {
      bits = sign;
    } else {
      exponent = 1U;
      while ((mantissa & 0x0400U) == 0U) {
        mantissa <<= 1U;
        --exponent;
      }
      mantissa &= 0x03ffU;
      bits = sign | ((exponent + 112U) << 23U) | (mantissa << 13U);
    }
  } else if (exponent == 31U) {
    bits = sign | 0x7f800000U | (mantissa << 13U);
  } else {
    bits = sign | ((exponent + 112U) << 23U) | (mantissa << 13U);
  }
  float value = 0.0f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

std::vector<std::uint16_t> ToHalf(const std::vector<float>& values) {
  std::vector<std::uint16_t> result(values.size());
  std::transform(values.begin(), values.end(), result.begin(), FloatToHalf);
  return result;
}

std::vector<float> ToFloat(const std::vector<std::uint16_t>& values) {
  std::vector<float> result(values.size());
  std::transform(values.begin(), values.end(), result.begin(), HalfToFloat);
  return result;
}

const std::filesystem::path& FindPayload(
    const std::map<std::string, std::filesystem::path>& paths,
    const std::string& name) {
  const auto found = paths.find(name);
  Require(found != paths.end(), "capture is missing " + name);
  return found->second;
}

struct Comparison {
  std::size_t count = 0;
  std::size_t finite_pairs = 0;
  double max_abs = 0.0;
  double mean_abs = 0.0;
  double relative_l2 = std::numeric_limits<double>::infinity();
  double cosine = 0.0;
  bool finite = false;
  bool passes = false;
};

Comparison Compare(const std::vector<float>& observed,
                   const std::vector<float>& reference) {
  Require(observed.size() == reference.size(), "comparison size mismatch");
  Comparison result;
  result.count = observed.size();
  long double abs_sum = 0.0L;
  long double diff_sq = 0.0L;
  long double observed_sq = 0.0L;
  long double reference_sq = 0.0L;
  long double dot = 0.0L;
  for (std::size_t index = 0; index < observed.size(); ++index) {
    const double lhs = observed[index];
    const double rhs = reference[index];
    if (!std::isfinite(lhs) || !std::isfinite(rhs)) continue;
    ++result.finite_pairs;
    const double difference = lhs - rhs;
    const double absolute = std::abs(difference);
    result.max_abs = std::max(result.max_abs, absolute);
    abs_sum += absolute;
    diff_sq += difference * difference;
    observed_sq += lhs * lhs;
    reference_sq += rhs * rhs;
    dot += lhs * rhs;
  }
  result.finite = result.finite_pairs == result.count;
  if (result.count != 0) {
    result.mean_abs = static_cast<double>(abs_sum / result.count);
  }
  if (reference_sq > 0.0L) {
    result.relative_l2 =
        std::sqrt(static_cast<double>(diff_sq / reference_sq));
  }
  if (observed_sq > 0.0L && reference_sq > 0.0L) {
    result.cosine = static_cast<double>(
        dot / std::sqrt(observed_sq * reference_sq));
  }
  result.passes = result.finite && result.cosine >= kCosineMinimum &&
                  result.relative_l2 <= kRelativeL2Maximum;
  return result;
}

double Median(std::vector<double> values) {
  Require(!values.empty(), "cannot take median of no samples");
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
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
  DeviceSelection fallback;
  for (const auto platform : platforms) {
    cl_uint device_count = 0;
    if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr,
                       &device_count) != CL_SUCCESS ||
        device_count == 0) {
      continue;
    }
    std::vector<cl_device_id> devices(device_count);
    Check(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, device_count,
                         devices.data(), nullptr),
          "clGetDeviceIDs");
    for (const auto device : devices) {
      DeviceSelection candidate{device, DeviceString(device, CL_DEVICE_NAME),
                                DeviceString(device, CL_DRIVER_VERSION)};
      if (!fallback.device) fallback = candidate;
      if (candidate.name.find("Arc") != std::string::npos) return candidate;
    }
  }
  Require(fallback.device != nullptr, "no OpenCL GPU found");
  return fallback;
}

struct OpenClState {
  cl_context context = nullptr;
  cl_command_queue queue = nullptr;
  cl_program program = nullptr;
  cl_kernel prepare = nullptr;
  cl_kernel recurrence = nullptr;
  cl_kernel scan = nullptr;
  cl_kernel output = nullptr;
  cl_kernel norm = nullptr;
  std::vector<cl_mem> buffers;

  ~OpenClState() {
    for (const auto buffer : buffers) clReleaseMemObject(buffer);
    if (norm) clReleaseKernel(norm);
    if (output) clReleaseKernel(output);
    if (scan) clReleaseKernel(scan);
    if (recurrence) clReleaseKernel(recurrence);
    if (prepare) clReleaseKernel(prepare);
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

void DumpProgramBinary(cl_program program,
                       const std::filesystem::path& path) {
  if (path.empty()) return;
  std::size_t binary_size = 0;
  Check(clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES,
                         sizeof(binary_size), &binary_size, nullptr),
        "clGetProgramInfo binary size");
  Require(binary_size != 0, "OpenCL program binary is empty");
  std::vector<unsigned char> binary(binary_size);
  unsigned char* binary_pointer = binary.data();
  Check(clGetProgramInfo(program, CL_PROGRAM_BINARIES,
                         sizeof(binary_pointer), &binary_pointer, nullptr),
        "clGetProgramInfo binary");
  std::ofstream output(path, std::ios::binary);
  Require(static_cast<bool>(output),
          "could not create program binary " + path.string());
  output.write(reinterpret_cast<const char*>(binary.data()),
               static_cast<std::streamsize>(binary.size()));
  Require(static_cast<bool>(output),
          "could not write program binary " + path.string());
}

template <typename Value>
cl_mem Input(OpenClState& state, const std::vector<Value>& values) {
  return Buffer(state, values.size() * sizeof(Value),
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                const_cast<Value*>(values.data()));
}

void SetMem(cl_kernel kernel, cl_uint index, cl_mem value) {
  Check(clSetKernelArg(kernel, index, sizeof(value), &value),
        "clSetKernelArg mem");
}

double EventUs(cl_event event, cl_profiling_info field) {
  cl_ulong value = 0;
  Check(clGetEventProfilingInfo(event, field, sizeof(value), &value, nullptr),
        "clGetEventProfilingInfo");
  return static_cast<double>(value) / 1000.0;
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
        line.find("onednn") != std::string::npos) {
      return true;
    }
  }
  return false;
}

void WriteComparison(std::ostream& output, const Comparison& value) {
  output << "{\"count\":" << value.count
         << ",\"finite_pairs\":" << value.finite_pairs
         << ",\"finite\":" << (value.finite ? "true" : "false")
         << ",\"max_abs\":" << value.max_abs
         << ",\"mean_abs\":" << value.mean_abs
         << ",\"relative_l2\":" << value.relative_l2
         << ",\"cosine\":" << value.cosine
         << ",\"passes\":" << (value.passes ? "true" : "false")
         << "}";
}

}  // namespace

int main(int argc, char** argv) {
  std::cout << std::setprecision(12);
  try {
    const auto args = ParseArgs(argc, argv);
    const auto paths = CapturePayloads(args.capture);
    const auto suffix = std::to_string(args.layer);
    const auto q = ReadVector<float>(
        FindPayload(paths, "q_conv_predelta-" + suffix), kQkValues);
    const auto k = ReadVector<float>(
        FindPayload(paths, "k_conv_predelta-" + suffix), kQkValues);
    const auto v = ReadStridedV(
        FindPayload(paths, "v_conv_predelta-" + suffix));
    const auto gate = ReadVector<float>(
        FindPayload(paths, "gate-" + suffix), kControlValues);
    const auto beta = ReadVector<float>(
        FindPayload(paths, "beta_sigmoid-" + suffix), kControlValues);
    const auto state_in = ReadVector<float>(
        FindPayload(paths, "state_predelta-" + suffix), kStateValues);
    const auto attention_reference = ReadVector<float>(
        FindPayload(paths, "attn_output-" + suffix), kValueValues);
    const auto state_reference = ReadVector<float>(
        FindPayload(paths, "new_state-" + suffix), kStateValues);
    const auto z = ReadVector<float>(
        FindPayload(paths, "z-" + suffix), kValueValues);
    const auto final_reference = ReadVector<float>(
        FindPayload(paths, "final_output-" + suffix), kValueValues);

    const auto model_index = iq36::parse_gguf_model_index(args.model.string());
    const auto norm_weight = iq36::decode_tensor_row(
        args.model.string(), model_index,
        "blk." + suffix + ".ssm_norm.weight", 0);
    Require(norm_weight.size() == kHeadDim,
            "linear-attention norm weight shape mismatch");
    const bool use_f16 = args.storage == "f16";
    const bool use_chunk64 = args.storage == "chunk64";
    const auto q_f16 = use_f16 ? ToHalf(q) : std::vector<std::uint16_t>{};
    const auto k_f16 = use_f16 ? ToHalf(k) : std::vector<std::uint16_t>{};
    const auto v_f16 = use_f16 ? ToHalf(v) : std::vector<std::uint16_t>{};
    const auto gate_f16 =
        use_f16 ? ToHalf(gate) : std::vector<std::uint16_t>{};
    const auto beta_f16 =
        use_f16 ? ToHalf(beta) : std::vector<std::uint16_t>{};
    const auto state_in_f16 =
        use_f16 ? ToHalf(state_in) : std::vector<std::uint16_t>{};
    const auto z_f16 = use_f16 ? ToHalf(z) : std::vector<std::uint16_t>{};
    const auto norm_weight_f16 =
        use_f16 ? ToHalf(norm_weight) : std::vector<std::uint16_t>{};

    const auto device = SelectDevice();
    OpenClState state;
    cl_int status = CL_SUCCESS;
    state.context =
        clCreateContext(nullptr, 1, &device.device, nullptr, nullptr, &status);
    Check(status, "clCreateContext");
    const cl_queue_properties queue_properties[] = {
        CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
    state.queue = clCreateCommandQueueWithProperties(
        state.context, device.device, queue_properties, &status);
    Check(status, "clCreateCommandQueueWithProperties");
    const auto source = ReadText(args.kernel);
    const char* source_pointer = source.c_str();
    const std::size_t source_size = source.size();
    state.program = clCreateProgramWithSource(
        state.context, 1, &source_pointer, &source_size, &status);
    Check(status, "clCreateProgramWithSource");
    const char* build_options =
        "-cl-std=CL3.0 -cl-mad-enable -cl-unsafe-math-optimizations "
        "-cl-finite-math-only -cl-fast-relaxed-math";
    status = clBuildProgram(state.program, 1, &device.device, build_options,
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
    const char* recurrence_name = use_f16
        ? "iq36_linear_prefill_recurrent_f16"
        : "iq36_linear_prefill_recurrent_f32";
    const char* norm_name = use_f16
        ? "iq36_linear_prefill_norm_gate_f16"
        : "iq36_linear_prefill_norm_gate_f32";
    if (use_chunk64) {
      state.prepare = clCreateKernel(
          state.program, "iq36_linear_prefill_chunk64_prepare_f32", &status);
      Check(status, "clCreateKernel chunk64 prepare");
      state.scan = clCreateKernel(
          state.program, "iq36_linear_prefill_chunk64_scan_f32", &status);
      Check(status, "clCreateKernel chunk64 scan");
      state.output = clCreateKernel(
          state.program, "iq36_linear_prefill_chunk64_output_f32", &status);
      Check(status, "clCreateKernel chunk64 output");
    } else {
      state.recurrence = clCreateKernel(
          state.program, recurrence_name, &status);
      Check(status, "clCreateKernel recurrence");
    }
    state.norm = clCreateKernel(state.program, norm_name, &status);
    Check(status, "clCreateKernel norm");

    const auto StorageInput = [&](const std::vector<float>& f32,
                                  const std::vector<std::uint16_t>& f16) {
      return use_f16 ? Input(state, f16) : Input(state, f32);
    };
    const auto q_buffer = StorageInput(q, q_f16);
    const auto k_buffer = StorageInput(k, k_f16);
    const auto v_buffer = StorageInput(v, v_f16);
    const auto gate_buffer = StorageInput(gate, gate_f16);
    const auto beta_buffer = StorageInput(beta, beta_f16);
    const auto state_in_buffer = StorageInput(state_in, state_in_f16);
    const auto z_buffer = StorageInput(z, z_f16);
    const auto norm_weight_buffer =
        StorageInput(norm_weight, norm_weight_f16);
    const std::size_t storage_bytes =
        use_f16 ? sizeof(std::uint16_t) : sizeof(float);
    const auto attention_buffer = Buffer(
        state, kValueValues * storage_bytes, CL_MEM_READ_WRITE);
    const auto state_out_buffer = Buffer(
        state, kStateValues * storage_bytes, CL_MEM_WRITE_ONLY);
    const auto final_buffer = Buffer(
        state, kValueValues * sizeof(float), CL_MEM_WRITE_ONLY);
    cl_mem cumulative_gate_buffer = nullptr;
    cl_mem w_buffer = nullptr;
    cl_mem u_buffer = nullptr;
    cl_mem v_new_buffer = nullptr;
    cl_mem chunk_state_buffer = nullptr;
    if (use_chunk64) {
      cumulative_gate_buffer = Buffer(
          state, kControlValues * sizeof(float), CL_MEM_READ_WRITE);
      w_buffer = Buffer(
          state, kValueValues * sizeof(float), CL_MEM_READ_WRITE);
      u_buffer = Buffer(
          state, kValueValues * sizeof(float), CL_MEM_READ_WRITE);
      v_new_buffer = Buffer(
          state, kValueValues * sizeof(float), CL_MEM_READ_WRITE);
      chunk_state_buffer = Buffer(
          state, kChunkStateValues * sizeof(float), CL_MEM_READ_WRITE);
    }

    if (use_chunk64) {
      SetMem(state.prepare, 0, k_buffer);
      SetMem(state.prepare, 1, v_buffer);
      SetMem(state.prepare, 2, gate_buffer);
      SetMem(state.prepare, 3, beta_buffer);
      SetMem(state.prepare, 4, cumulative_gate_buffer);
      SetMem(state.prepare, 5, w_buffer);
      SetMem(state.prepare, 6, u_buffer);
      SetMem(state.scan, 0, k_buffer);
      SetMem(state.scan, 1, cumulative_gate_buffer);
      SetMem(state.scan, 2, w_buffer);
      SetMem(state.scan, 3, u_buffer);
      SetMem(state.scan, 4, state_in_buffer);
      SetMem(state.scan, 5, v_new_buffer);
      SetMem(state.scan, 6, chunk_state_buffer);
      SetMem(state.scan, 7, state_out_buffer);
      SetMem(state.output, 0, q_buffer);
      SetMem(state.output, 1, k_buffer);
      SetMem(state.output, 2, cumulative_gate_buffer);
      SetMem(state.output, 3, v_new_buffer);
      SetMem(state.output, 4, chunk_state_buffer);
      SetMem(state.output, 5, attention_buffer);
    } else {
      SetMem(state.recurrence, 0, q_buffer);
      SetMem(state.recurrence, 1, k_buffer);
      SetMem(state.recurrence, 2, v_buffer);
      SetMem(state.recurrence, 3, gate_buffer);
      SetMem(state.recurrence, 4, beta_buffer);
      SetMem(state.recurrence, 5, state_in_buffer);
      SetMem(state.recurrence, 6, attention_buffer);
      SetMem(state.recurrence, 7, state_out_buffer);
    }
    SetMem(state.norm, 0, attention_buffer);
    SetMem(state.norm, 1, z_buffer);
    SetMem(state.norm, 2, norm_weight_buffer);
    const float epsilon = static_cast<float>(kNormEpsilon);
    Check(clSetKernelArg(state.norm, 3, sizeof(epsilon), &epsilon),
          "clSetKernelArg epsilon");
    SetMem(state.norm, 4, final_buffer);

    std::vector<double> prepare_samples;
    std::vector<double> recurrence_samples;
    std::vector<double> scan_samples;
    std::vector<double> output_samples;
    std::vector<double> state_core_samples;
    std::vector<double> norm_samples;
    std::vector<double> complete_samples;
    const std::size_t recurrent_workgroups = use_chunk64
        ? kChunkHeadWorkgroups
        : (use_f16 ? kF16RecurrentWorkgroups : kRecurrentWorkgroups);
    const std::size_t recurrence_local = use_f16 ? 16 : 32;
    const std::size_t recurrence_global =
        recurrent_workgroups * recurrence_local;
    const std::size_t chunk_head_global = kChunkHeadWorkgroups * kChunkSize;
    const std::size_t chunk_head_local = kChunkSize;
    const std::size_t chunk_scan_global = kRecurrentWorkgroups * 32;
    const std::size_t chunk_scan_local = 32;
    const std::size_t norm_local = use_f16 ? 16 : 32;
    const std::size_t norm_global = kNormWorkgroups * norm_local;
    for (int iteration = -args.warmup; iteration < args.repeat; ++iteration) {
      cl_event prepare_event = nullptr;
      cl_event recurrence_event = nullptr;
      cl_event scan_event = nullptr;
      cl_event output_event = nullptr;
      cl_event norm_event = nullptr;
      if (use_chunk64) {
        Check(clEnqueueNDRangeKernel(state.queue, state.prepare, 1, nullptr,
                                     &chunk_head_global, &chunk_head_local,
                                     0, nullptr, &prepare_event),
              "clEnqueueNDRangeKernel chunk64 prepare");
        Check(clEnqueueNDRangeKernel(state.queue, state.scan, 1, nullptr,
                                     &chunk_scan_global, &chunk_scan_local,
                                     0, nullptr, &scan_event),
              "clEnqueueNDRangeKernel chunk64 scan");
        Check(clEnqueueNDRangeKernel(state.queue, state.output, 1, nullptr,
                                     &chunk_head_global, &chunk_head_local,
                                     0, nullptr, &output_event),
              "clEnqueueNDRangeKernel chunk64 output");
      } else {
        Check(clEnqueueNDRangeKernel(
                  state.queue, state.recurrence, 1, nullptr,
                  &recurrence_global, &recurrence_local, 0, nullptr,
                  &recurrence_event),
              "clEnqueueNDRangeKernel recurrence");
      }
      Check(clEnqueueNDRangeKernel(state.queue, state.norm, 1, nullptr,
                                   &norm_global, &norm_local, 0, nullptr,
                                   &norm_event),
            "clEnqueueNDRangeKernel norm");
      Check(clFinish(state.queue), "clFinish");
      if (iteration >= 0) {
        double state_core_start = 0.0;
        double state_core_end = 0.0;
        if (use_chunk64) {
          const double prepare_start =
              EventUs(prepare_event, CL_PROFILING_COMMAND_START);
          const double prepare_end =
              EventUs(prepare_event, CL_PROFILING_COMMAND_END);
          const double scan_start =
              EventUs(scan_event, CL_PROFILING_COMMAND_START);
          const double scan_end =
              EventUs(scan_event, CL_PROFILING_COMMAND_END);
          const double output_start =
              EventUs(output_event, CL_PROFILING_COMMAND_START);
          const double output_end =
              EventUs(output_event, CL_PROFILING_COMMAND_END);
          prepare_samples.push_back(prepare_end - prepare_start);
          scan_samples.push_back(scan_end - scan_start);
          output_samples.push_back(output_end - output_start);
          recurrence_samples.push_back(scan_end - scan_start);
          state_core_start = prepare_start;
          state_core_end = output_end;
        } else {
          const double recurrence_start =
              EventUs(recurrence_event, CL_PROFILING_COMMAND_START);
          const double recurrence_end =
              EventUs(recurrence_event, CL_PROFILING_COMMAND_END);
          recurrence_samples.push_back(recurrence_end - recurrence_start);
          state_core_start = recurrence_start;
          state_core_end = recurrence_end;
        }
        const double norm_start = EventUs(norm_event, CL_PROFILING_COMMAND_START);
        const double norm_end = EventUs(norm_event, CL_PROFILING_COMMAND_END);
        state_core_samples.push_back(state_core_end - state_core_start);
        norm_samples.push_back(norm_end - norm_start);
        complete_samples.push_back(norm_end - state_core_start);
      }
      clReleaseEvent(norm_event);
      if (output_event) clReleaseEvent(output_event);
      if (scan_event) clReleaseEvent(scan_event);
      if (recurrence_event) clReleaseEvent(recurrence_event);
      if (prepare_event) clReleaseEvent(prepare_event);
    }

    std::vector<float> attention;
    std::vector<float> state_out;
    std::vector<float> final(kValueValues);
    if (use_f16) {
      std::vector<std::uint16_t> attention_f16(kValueValues);
      std::vector<std::uint16_t> state_out_f16(kStateValues);
      Check(clEnqueueReadBuffer(state.queue, attention_buffer, CL_TRUE, 0,
                                attention_f16.size() * sizeof(std::uint16_t),
                                attention_f16.data(), 0, nullptr, nullptr),
            "clEnqueueReadBuffer attention f16");
      Check(clEnqueueReadBuffer(state.queue, state_out_buffer, CL_TRUE, 0,
                                state_out_f16.size() * sizeof(std::uint16_t),
                                state_out_f16.data(), 0, nullptr, nullptr),
            "clEnqueueReadBuffer state f16");
      attention = ToFloat(attention_f16);
      state_out = ToFloat(state_out_f16);
    } else {
      attention.resize(kValueValues);
      state_out.resize(kStateValues);
      Check(clEnqueueReadBuffer(state.queue, attention_buffer, CL_TRUE, 0,
                                attention.size() * sizeof(float),
                                attention.data(), 0, nullptr, nullptr),
            "clEnqueueReadBuffer attention f32");
      Check(clEnqueueReadBuffer(state.queue, state_out_buffer, CL_TRUE, 0,
                                state_out.size() * sizeof(float),
                                state_out.data(), 0, nullptr, nullptr),
            "clEnqueueReadBuffer state f32");
    }
    Check(clEnqueueReadBuffer(state.queue, final_buffer, CL_TRUE, 0,
                              final.size() * sizeof(float), final.data(), 0,
                              nullptr, nullptr),
          "clEnqueueReadBuffer final");

    const auto attention_comparison =
        Compare(attention, attention_reference);
    const auto state_comparison = Compare(state_out, state_reference);
    const auto final_comparison = Compare(final, final_reference);
    const double prepare_median =
        use_chunk64 ? Median(prepare_samples) : 0.0;
    const double recurrence_median = Median(recurrence_samples);
    const double scan_median = use_chunk64 ? Median(scan_samples) : 0.0;
    const double output_median = use_chunk64 ? Median(output_samples) : 0.0;
    const double state_core_median = Median(state_core_samples);
    const double norm_median = Median(norm_samples);
    const double complete_median = Median(complete_samples);
    const bool forbidden_runtime_mapped = MapsForbiddenRuntime();
    const bool passed = attention_comparison.passes &&
                        state_comparison.passes && final_comparison.passes &&
                        state_core_median <= args.cap_us &&
                        !forbidden_runtime_mapped;

    std::cout << "{\"schema_version\":"
              << "\"intel-qwen36-linear-attention-prefill-state-probe-v0\""
              << ",\"device_name\":\"" << device.name << "\""
              << ",\"driver_version\":\"" << device.driver << "\""
              << ",\"storage\":\"" << args.storage << "\""
              << ",\"layer\":" << args.layer
              << ",\"token_count\":" << kTokenCount
              << ",\"head_dim\":" << kHeadDim
              << ",\"key_heads\":" << kKeyHeads
              << ",\"value_heads\":" << kValueHeads
              << ",\"chunk_size\":" << (use_chunk64 ? kChunkSize : 0)
              << ",\"chunk_count\":" << (use_chunk64 ? kChunkCount : 0)
              << ",\"recurrent_workgroups\":" << recurrent_workgroups
              << ",\"chunk_scan_workgroups\":"
              << (use_chunk64 ? kRecurrentWorkgroups : 0)
              << ",\"norm_workgroups\":" << kNormWorkgroups
              << ",\"timed_host_upload_bytes\":0"
              << ",\"timed_host_read_bytes\":0"
              << ",\"warmup\":" << args.warmup
              << ",\"repeat\":" << args.repeat
              << ",\"cap_us\":" << args.cap_us
              << ",\"prepare_samples_us\":[";
    for (std::size_t index = 0; index < prepare_samples.size(); ++index) {
      if (index) std::cout << ',';
      std::cout << prepare_samples[index];
    }
    std::cout << "],\"recurrence_samples_us\":[";
    for (std::size_t index = 0; index < recurrence_samples.size(); ++index) {
      if (index) std::cout << ',';
      std::cout << recurrence_samples[index];
    }
    std::cout << "],\"scan_samples_us\":[";
    for (std::size_t index = 0; index < scan_samples.size(); ++index) {
      if (index) std::cout << ',';
      std::cout << scan_samples[index];
    }
    std::cout << "],\"output_samples_us\":[";
    for (std::size_t index = 0; index < output_samples.size(); ++index) {
      if (index) std::cout << ',';
      std::cout << output_samples[index];
    }
    std::cout << "],\"state_core_samples_us\":[";
    for (std::size_t index = 0; index < state_core_samples.size(); ++index) {
      if (index) std::cout << ',';
      std::cout << state_core_samples[index];
    }
    std::cout << "],\"norm_samples_us\":[";
    for (std::size_t index = 0; index < norm_samples.size(); ++index) {
      if (index) std::cout << ',';
      std::cout << norm_samples[index];
    }
    std::cout << "],\"complete_samples_us\":[";
    for (std::size_t index = 0; index < complete_samples.size(); ++index) {
      if (index) std::cout << ',';
      std::cout << complete_samples[index];
    }
    std::cout << "]"
              << ",\"prepare_median_us\":" << prepare_median
              << ",\"recurrence_median_us\":" << recurrence_median
              << ",\"scan_median_us\":" << scan_median
              << ",\"output_median_us\":" << output_median
              << ",\"state_core_median_us\":" << state_core_median
              << ",\"norm_median_us\":" << norm_median
              << ",\"complete_median_us\":" << complete_median
              << ",\"attention_comparison\":";
    WriteComparison(std::cout, attention_comparison);
    std::cout << ",\"state_comparison\":";
    WriteComparison(std::cout, state_comparison);
    std::cout << ",\"final_comparison\":";
    WriteComparison(std::cout, final_comparison);
    std::cout << ",\"forbidden_runtime_mapped\":"
              << (forbidden_runtime_mapped ? "true" : "false")
              << ",\"required_checks_passed\":"
              << (passed ? "true" : "false") << "}\n";
    return passed ? 0 : 2;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
