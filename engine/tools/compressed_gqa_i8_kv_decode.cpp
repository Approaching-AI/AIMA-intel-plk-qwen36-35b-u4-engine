#include <CL/cl.h>

#include <algorithm>
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

constexpr cl_uint kContextTokens = 131072;
constexpr cl_uint kHeadDim = 256;
constexpr cl_uint kQHeads = 16;
constexpr cl_uint kKvHeads = 2;
constexpr cl_uint kGqaGroup = 8;
constexpr cl_uint kQuantGroup = 32;
constexpr cl_uint kScaleGroups = kHeadDim / kQuantGroup;
constexpr cl_uint kChunkTokens = 256;
constexpr cl_uint kLocalSize = 256;
constexpr cl_uint kQuantLocalSize = 32;
constexpr float kAttentionScale = 0.0625f;
constexpr double kComponentCapMs = 2.825;

[[noreturn]] void Die(const std::string& message) {
  throw std::runtime_error(message);
}

void Require(bool condition, const std::string& message) {
  if (!condition) Die(message);
}

void Check(cl_int status, const std::string& operation) {
  if (status != CL_SUCCESS) {
    Die(operation + " failed with OpenCL status " + std::to_string(status));
  }
}

std::string ReadText(const std::string& path) {
  std::ifstream input(path);
  Require(static_cast<bool>(input), "failed to open OpenCL source");
  return std::string(
      std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
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
  Check(clGetPlatformIDs(0, nullptr, &platform_count), "clGetPlatformIDs(count)");
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
      if (DeviceString(device, CL_DEVICE_NAME).find("Intel") !=
          std::string::npos) {
        return device;
      }
    }
  }
  Die("Intel GPU OpenCL device not found");
}

cl_mem Buffer(cl_context context, cl_mem_flags flags, std::size_t bytes,
              void* host = nullptr) {
  cl_int status = CL_SUCCESS;
  cl_mem result = clCreateBuffer(context, flags, bytes, host, &status);
  Check(status, "clCreateBuffer");
  return result;
}

template <typename T>
void Arg(cl_kernel kernel, cl_uint index, const T& value) {
  Check(clSetKernelArg(kernel, index, sizeof(T), &value), "clSetKernelArg");
}

double EventMs(cl_event event) {
  cl_ulong begin = 0;
  cl_ulong end = 0;
  Check(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START,
                                sizeof(begin), &begin, nullptr),
        "clGetEventProfilingInfo(start)");
  Check(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,
                                sizeof(end), &end, nullptr),
        "clGetEventProfilingInfo(end)");
  return static_cast<double>(end - begin) / 1.0e6;
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
  if (exponent >= 31) {
    return static_cast<std::uint16_t>(sign | 0x7c00U);
  }
  if (exponent <= 0) {
    if (exponent < -10) return static_cast<std::uint16_t>(sign);
    mantissa |= 0x800000U;
    const unsigned shift = static_cast<unsigned>(14 - exponent);
    const std::uint32_t rounded =
        (mantissa + (1U << (shift - 1U)) - 1U +
         ((mantissa >> shift) & 1U)) >> shift;
    return static_cast<std::uint16_t>(sign | rounded);
  }
  mantissa += 0x0fffU + ((mantissa >> 13U) & 1U);
  if ((mantissa & 0x800000U) != 0U) {
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

void CompressPattern(const std::vector<float>& input,
                     std::vector<std::uint8_t>* values,
                     std::vector<std::uint16_t>* scales) {
  Require(input.size() % (kKvHeads * kHeadDim) == 0,
          "pattern shape mismatch");
  const std::size_t tokens = input.size() / (kKvHeads * kHeadDim);
  values->resize(input.size());
  scales->resize(tokens * kKvHeads * kScaleGroups);
  for (std::size_t token = 0; token < tokens; ++token) {
    for (std::size_t head = 0; head < kKvHeads; ++head) {
      for (std::size_t group = 0; group < kScaleGroups; ++group) {
        const std::size_t value_base =
            (token * kKvHeads + head) * kHeadDim + group * kQuantGroup;
        float maximum = 0.0f;
        for (std::size_t lane = 0; lane < kQuantGroup; ++lane) {
          maximum = std::max(maximum, std::abs(input[value_base + lane]));
        }
        const float scale = maximum == 0.0f ? 1.0f : maximum / 127.0f;
        const std::size_t scale_index =
            (token * kKvHeads + head) * kScaleGroups + group;
        (*scales)[scale_index] = FloatToHalf(scale);
        for (std::size_t lane = 0; lane < kQuantGroup; ++lane) {
          const int quantized = std::clamp(
              static_cast<int>(std::nearbyint(input[value_base + lane] / scale)),
              -127, 127);
          (*values)[value_base + lane] =
              static_cast<std::uint8_t>(static_cast<std::int8_t>(quantized));
        }
      }
    }
  }
}

std::uint8_t EncodeE4M3(float value) {
  const std::uint8_t sign = value < 0.0f ? 128U : 0U;
  const float magnitude = std::abs(value);
  std::uint32_t code = 0U;
  if (magnitude < 0.015625f) {
    code = std::min<std::uint32_t>(
        static_cast<std::uint32_t>(std::nearbyint(magnitude * 512.0f)), 7U);
  } else {
    const std::uint16_t half_bits = FloatToHalf(magnitude);
    std::uint32_t exponent = (half_bits >> 10U) - 8U;
    const std::uint32_t half_mantissa = half_bits & 1023U;
    std::uint32_t mantissa = half_mantissa >> 7U;
    const std::uint32_t discarded = half_mantissa & 127U;
    if (discarded > 64U ||
        (discarded == 64U && (mantissa & 1U) != 0U)) {
      ++mantissa;
      if (mantissa == 8U) {
        mantissa = 0U;
        ++exponent;
      }
    }
    code = std::min<std::uint32_t>((exponent << 3U) | mantissa, 119U);
  }
  return static_cast<std::uint8_t>(sign | code);
}

void CompressE4M3Pattern(const std::vector<float>& input,
                         std::vector<std::uint8_t>* values,
                         std::vector<std::uint16_t>* scales) {
  Require(input.size() % (kKvHeads * kHeadDim) == 0,
          "pattern shape mismatch");
  const std::size_t tokens = input.size() / (kKvHeads * kHeadDim);
  values->resize(input.size());
  scales->resize(tokens * kKvHeads * kScaleGroups);
  for (std::size_t token = 0; token < tokens; ++token) {
    for (std::size_t head = 0; head < kKvHeads; ++head) {
      for (std::size_t group = 0; group < kScaleGroups; ++group) {
        const std::size_t value_base =
            (token * kKvHeads + head) * kHeadDim + group * kQuantGroup;
        float maximum = 0.0f;
        for (std::size_t lane = 0; lane < kQuantGroup; ++lane) {
          maximum = std::max(maximum, std::abs(input[value_base + lane]));
        }
        const float scale = maximum == 0.0f ? 1.0f : maximum;
        const std::size_t scale_index =
            (token * kKvHeads + head) * kScaleGroups + group;
        (*scales)[scale_index] = FloatToHalf(scale);
        for (std::size_t lane = 0; lane < kQuantGroup; ++lane) {
          (*values)[value_base + lane] =
              EncodeE4M3(input[value_base + lane] / scale);
        }
      }
    }
  }
}

struct TimedRun {
  double quantize_ms = 0.0;
  double partial_ms = 0.0;
  double reduce_ms = 0.0;
  double total_ms = 0.0;
};

TimedRun RunCompressed(cl_command_queue queue, cl_kernel quantize,
                       cl_kernel partial, cl_kernel reduce,
                       std::size_t quantize_global,
                       std::size_t partial_global,
                       std::size_t reduce_global) {
  const std::size_t quantize_local = kQuantLocalSize;
  const std::size_t local = kLocalSize;
  cl_event quantize_event = nullptr;
  cl_event partial_event = nullptr;
  cl_event reduce_event = nullptr;
  Check(clEnqueueNDRangeKernel(queue, quantize, 1, nullptr, &quantize_global,
                               &quantize_local, 0, nullptr, &quantize_event),
        "clEnqueueNDRangeKernel(quantize)");
  Check(clEnqueueNDRangeKernel(queue, partial, 1, nullptr, &partial_global,
                               &local, 0, nullptr, &partial_event),
        "clEnqueueNDRangeKernel(partial)");
  Check(clEnqueueNDRangeKernel(queue, reduce, 1, nullptr, &reduce_global,
                               &local, 0, nullptr, &reduce_event),
        "clEnqueueNDRangeKernel(reduce)");
  Check(clFinish(queue), "clFinish(compressed)");
  TimedRun result;
  result.quantize_ms = EventMs(quantize_event);
  result.partial_ms = EventMs(partial_event);
  result.reduce_ms = EventMs(reduce_event);
  result.total_ms =
      result.quantize_ms + result.partial_ms + result.reduce_ms;
  clReleaseEvent(quantize_event);
  clReleaseEvent(partial_event);
  clReleaseEvent(reduce_event);
  return result;
}

TimedRun RunDistribution(cl_command_queue queue, cl_kernel quantize,
                         cl_kernel partial, cl_kernel reduce,
                         std::size_t quantize_global,
                         std::size_t partial_global,
                         std::size_t reduce_global,
                         std::vector<TimedRun>* samples) {
  constexpr int kSamples = 10;
  samples->clear();
  samples->reserve(kSamples);
  for (int sample = 0; sample < kSamples; ++sample) {
    samples->push_back(RunCompressed(
        queue, quantize, partial, reduce, quantize_global, partial_global,
        reduce_global));
  }
  std::vector<TimedRun> ordered = *samples;
  std::sort(ordered.begin(), ordered.end(),
            [](const TimedRun& left, const TimedRun& right) {
              return left.total_ms < right.total_ms;
            });
  return ordered[ordered.size() / 2U];
}

struct Numeric {
  double cosine = 0.0;
  double relative_l2 = 0.0;
  double rmse = 0.0;
  double max_abs = 0.0;
  bool finite = true;
};

Numeric Compare(const std::vector<float>& reference,
                const std::vector<float>& candidate) {
  Require(reference.size() == candidate.size() && !reference.empty(),
          "numeric comparison shape mismatch");
  double dot = 0.0;
  double ref_l2 = 0.0;
  double cand_l2 = 0.0;
  double diff_l2 = 0.0;
  double max_abs = 0.0;
  bool finite = true;
  for (std::size_t i = 0; i < reference.size(); ++i) {
    const double ref = reference[i];
    const double cand = candidate[i];
    finite = finite && std::isfinite(ref) && std::isfinite(cand);
    const double diff = cand - ref;
    dot += ref * cand;
    ref_l2 += ref * ref;
    cand_l2 += cand * cand;
    diff_l2 += diff * diff;
    max_abs = std::max(max_abs, std::abs(diff));
  }
  Numeric result;
  result.cosine = dot / (std::sqrt(ref_l2) * std::sqrt(cand_l2));
  result.relative_l2 = std::sqrt(diff_l2 / ref_l2);
  result.rmse = std::sqrt(diff_l2 / static_cast<double>(reference.size()));
  result.max_abs = max_abs;
  result.finite = finite;
  return result;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 2 && argc != 3) {
      throw std::invalid_argument(
          "usage: iq36-compressed-gqa-i8-kv-decode OPENCL_SOURCE [e4m3]");
    }
    const bool e4m3 = argc == 3 && std::string(argv[2]) == "e4m3";
    Require(argc == 2 || e4m3, "unsupported compressed KV mode");
    const auto device = SelectGpu();
    cl_int status = CL_SUCCESS;
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr,
                                         &status);
    Check(status, "clCreateContext");
    cl_command_queue queue = clCreateCommandQueue(
        context, device, CL_QUEUE_PROFILING_ENABLE, &status);
    Check(status, "clCreateCommandQueue");
    const std::string source = ReadText(argv[1]);
    const char* source_data = source.data();
    const std::size_t source_size = source.size();
    cl_program program = clCreateProgramWithSource(
        context, 1, &source_data, &source_size, &status);
    Check(status, "clCreateProgramWithSource");
    status = clBuildProgram(program, 1, &device, "-cl-std=CL3.0", nullptr,
                            nullptr);
    if (status != CL_SUCCESS) {
      std::size_t log_bytes = 0;
      clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr,
                            &log_bytes);
      std::string log(log_bytes, '\0');
      clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_bytes,
                            log.data(), nullptr);
      std::cerr << log << '\n';
      Check(status, "clBuildProgram");
    }
    auto Kernel = [&](const char* name) {
      cl_kernel kernel = clCreateKernel(program, name, &status);
      Check(status, std::string("clCreateKernel(") + name + ")");
      return kernel;
    };
    cl_kernel quantize = Kernel(e4m3
        ? "iq36_encode_current_scaled_e4m3_block32"
        : "iq36_quantize_current_i8_block32");
    cl_kernel partial = Kernel(e4m3
        ? "iq36_scaled_e4m3_gqa_partial"
        : "iq36_compressed_gqa_i8_partial");
    cl_kernel reduce = Kernel(e4m3
        ? "iq36_scaled_e4m3_gqa_partial_reduce"
        : "iq36_compressed_gqa_partial_reduce");
    cl_kernel reference_score = Kernel("iq36_reference_score_f32");
    cl_kernel reference_apply = Kernel("iq36_reference_apply_f32");

    const std::size_t history_values =
        static_cast<std::size_t>(kContextTokens) * kKvHeads * kHeadDim;
    const std::size_t scale_values =
        static_cast<std::size_t>(kContextTokens) * kKvHeads * kScaleGroups;
    const std::size_t output_values =
        static_cast<std::size_t>(kQHeads) * kHeadDim;
    const cl_uint chunk_count =
        (kContextTokens + kChunkTokens - 1U) / kChunkTokens;
    const std::size_t group_count =
        static_cast<std::size_t>(kKvHeads) * chunk_count;
    const std::size_t meta_values = group_count * kGqaGroup;
    const std::size_t partial_values = meta_values * kHeadDim;

    std::vector<float> q(output_values);
    std::vector<float> gate(output_values);
    for (std::size_t i = 0; i < output_values; ++i) {
      q[i] = static_cast<float>(static_cast<int>((i * 29U + 17U) & 255U) - 128)
          / 1024.0f;
      gate[i] = static_cast<float>(static_cast<int>((i * 11U + 3U) & 127U) - 64)
          / 64.0f;
    }
    constexpr std::size_t kPatternTokens = 256;
    const std::size_t pattern_values = kPatternTokens * kKvHeads * kHeadDim;
    std::vector<float> k_pattern(pattern_values);
    std::vector<float> v_pattern(pattern_values);
    for (std::size_t token = 0; token < kPatternTokens; ++token) {
      for (std::size_t dim = 0; dim < kKvHeads * kHeadDim; ++dim) {
        const std::size_t index = token * kKvHeads * kHeadDim + dim;
        k_pattern[index] = static_cast<float>(
            static_cast<int>((token * 13U + dim * 7U + 5U) & 255U) - 128)
            / 2048.0f;
        v_pattern[index] = 0.02f + static_cast<float>(
            static_cast<int>((token * 3U + dim * 19U + 9U) & 255U) - 128)
            / 4096.0f;
      }
    }
    std::vector<std::uint8_t> k_i8_pattern;
    std::vector<std::uint8_t> v_i8_pattern;
    std::vector<std::uint16_t> k_scale_pattern;
    std::vector<std::uint16_t> v_scale_pattern;
    if (e4m3) {
      CompressE4M3Pattern(k_pattern, &k_i8_pattern, &k_scale_pattern);
      CompressE4M3Pattern(v_pattern, &v_i8_pattern, &v_scale_pattern);
    } else {
      CompressPattern(k_pattern, &k_i8_pattern, &k_scale_pattern);
      CompressPattern(v_pattern, &v_i8_pattern, &v_scale_pattern);
    }
    std::vector<float> current_k(kKvHeads * kHeadDim);
    std::vector<float> current_v(kKvHeads * kHeadDim);
    std::copy(k_pattern.end() - current_k.size(), k_pattern.end(),
              current_k.begin());
    std::copy(v_pattern.end() - current_v.size(), v_pattern.end(),
              current_v.begin());

    std::vector<float> k_history(history_values);
    std::vector<float> v_history(history_values);
    std::vector<std::uint8_t> k_i8(history_values);
    std::vector<std::uint8_t> v_i8(history_values);
    std::vector<std::uint16_t> k_scales(scale_values);
    std::vector<std::uint16_t> v_scales(scale_values);
    const std::size_t pattern_scales = k_scale_pattern.size();
    for (std::size_t token = 0; token < kContextTokens;
         token += kPatternTokens) {
      const std::size_t value_offset = token * kKvHeads * kHeadDim;
      const std::size_t scale_offset = token * kKvHeads * kScaleGroups;
      std::memcpy(k_history.data() + value_offset, k_pattern.data(),
                  pattern_values * sizeof(float));
      std::memcpy(v_history.data() + value_offset, v_pattern.data(),
                  pattern_values * sizeof(float));
      std::memcpy(k_i8.data() + value_offset, k_i8_pattern.data(),
                  pattern_values * sizeof(std::uint8_t));
      std::memcpy(v_i8.data() + value_offset, v_i8_pattern.data(),
                  pattern_values * sizeof(std::uint8_t));
      std::memcpy(k_scales.data() + scale_offset, k_scale_pattern.data(),
                  pattern_scales * sizeof(std::uint16_t));
      std::memcpy(v_scales.data() + scale_offset, v_scale_pattern.data(),
                  pattern_scales * sizeof(std::uint16_t));
    }

    cl_mem q_buffer = Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                             q.size() * sizeof(float), q.data());
    cl_mem gate_buffer = Buffer(
        context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        gate.size() * sizeof(float), gate.data());
    cl_mem current_k_buffer = Buffer(
        context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        current_k.size() * sizeof(float), current_k.data());
    cl_mem current_v_buffer = Buffer(
        context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        current_v.size() * sizeof(float), current_v.data());
    cl_mem k_f32 = Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                          k_history.size() * sizeof(float), k_history.data());
    cl_mem v_f32 = Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                          v_history.size() * sizeof(float), v_history.data());
    cl_mem k_i8_buffer = Buffer(
        context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        k_i8.size() * sizeof(std::uint8_t), k_i8.data());
    cl_mem v_i8_buffer = Buffer(
        context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        v_i8.size() * sizeof(std::uint8_t), v_i8.data());
    cl_mem k_scale_buffer = Buffer(
        context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        k_scales.size() * sizeof(std::uint16_t), k_scales.data());
    cl_mem v_scale_buffer = Buffer(
        context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        v_scales.size() * sizeof(std::uint16_t), v_scales.data());
    cl_mem scores = Buffer(context, CL_MEM_READ_WRITE,
                           static_cast<std::size_t>(kQHeads) * kContextTokens *
                               sizeof(float));
    cl_mem reference_output = Buffer(context, CL_MEM_READ_WRITE,
                                     output_values * sizeof(float));
    cl_mem compressed_output = Buffer(context, CL_MEM_READ_WRITE,
                                      output_values * sizeof(float));
    cl_mem partial_max = Buffer(context, CL_MEM_READ_WRITE,
                                meta_values * sizeof(float));
    cl_mem partial_sum = Buffer(context, CL_MEM_READ_WRITE,
                                meta_values * sizeof(float));
    cl_mem partial_output = Buffer(context, CL_MEM_READ_WRITE,
                                   partial_values * sizeof(float));
    k_history.clear();
    v_history.clear();
    k_i8.clear();
    v_i8.clear();
    k_scales.clear();
    v_scales.clear();

    const std::size_t local = kLocalSize;
    Arg(reference_score, 0, q_buffer);
    Arg(reference_score, 1, k_f32);
    Arg(reference_score, 2, kContextTokens);
    Arg(reference_score, 3, kAttentionScale);
    Arg(reference_score, 4, scores);
    const std::size_t score_items =
        static_cast<std::size_t>(kQHeads) * kContextTokens;
    const std::size_t score_global =
        ((score_items + local - 1U) / local) * local;
    Check(clEnqueueNDRangeKernel(queue, reference_score, 1, nullptr,
                                 &score_global, &local, 0, nullptr, nullptr),
          "clEnqueueNDRangeKernel(reference score)");
    Arg(reference_apply, 0, scores);
    Arg(reference_apply, 1, v_f32);
    Arg(reference_apply, 2, gate_buffer);
    Arg(reference_apply, 3, kContextTokens);
    Arg(reference_apply, 4, reference_output);
    const std::size_t output_global = output_values;
    Check(clEnqueueNDRangeKernel(queue, reference_apply, 1, nullptr,
                                 &output_global, &local, 0, nullptr, nullptr),
          "clEnqueueNDRangeKernel(reference apply)");
    Check(clFinish(queue), "clFinish(reference)");

    Arg(quantize, 0, current_k_buffer);
    Arg(quantize, 1, current_v_buffer);
    Arg(quantize, 2, k_i8_buffer);
    Arg(quantize, 3, v_i8_buffer);
    Arg(quantize, 4, k_scale_buffer);
    Arg(quantize, 5, v_scale_buffer);
    Arg(quantize, 6, kContextTokens);
    Arg(partial, 0, q_buffer);
    Arg(partial, 1, k_i8_buffer);
    Arg(partial, 2, v_i8_buffer);
    Arg(partial, 3, k_scale_buffer);
    Arg(partial, 4, v_scale_buffer);
    Arg(partial, 5, kContextTokens);
    Arg(partial, 6, kAttentionScale);
    Arg(partial, 7, partial_max);
    Arg(partial, 8, partial_sum);
    Arg(partial, 9, partial_output);
    Arg(reduce, 0, partial_max);
    Arg(reduce, 1, partial_sum);
    Arg(reduce, 2, partial_output);
    Arg(reduce, 3, gate_buffer);
    Arg(reduce, 4, kContextTokens);
    Arg(reduce, 5, compressed_output);
    const std::size_t quantize_global =
        2U * kKvHeads * kScaleGroups * kQuantLocalSize;
    const std::size_t partial_global = group_count * local;
    const std::size_t reduce_global = output_values;
    for (int warmup = 0; warmup < 5; ++warmup) {
      (void)RunCompressed(queue, quantize, partial, reduce, quantize_global,
                          partial_global, reduce_global);
    }
    std::vector<TimedRun> repeat_samples;
    std::vector<TimedRun> confirm_samples;
    const auto repeat = RunDistribution(
        queue, quantize, partial, reduce, quantize_global, partial_global,
        reduce_global, &repeat_samples);
    const auto confirm = RunDistribution(
        queue, quantize, partial, reduce, quantize_global, partial_global,
        reduce_global, &confirm_samples);

    std::vector<float> reference(output_values);
    std::vector<float> candidate(output_values);
    Check(clEnqueueReadBuffer(queue, reference_output, CL_TRUE, 0,
                              reference.size() * sizeof(float),
                              reference.data(), 0, nullptr, nullptr),
          "clEnqueueReadBuffer(reference)");
    Check(clEnqueueReadBuffer(queue, compressed_output, CL_TRUE, 0,
                              candidate.size() * sizeof(float),
                              candidate.data(), 0, nullptr, nullptr),
          "clEnqueueReadBuffer(candidate)");
    const auto numeric = Compare(reference, candidate);
    const double spread =
        std::abs(repeat.total_ms - confirm.total_ms) /
        std::max(repeat.total_ms, confirm.total_ms);
    const bool numeric_pass = numeric.finite && numeric.cosine >= 0.999 &&
        numeric.relative_l2 <= 0.002;
    // The wrapper performs the promotion decision from the one-sided 95%
    // confidence bound over all timed samples.  Repeat/confirm spread remains
    // telemetry only; it is not a rate gate.
    const bool point_estimate_rate_pass = repeat.total_ms <= kComponentCapMs &&
        confirm.total_ms <= kComponentCapMs;
    // The process-level wrapper owns the confidence-bound rate decision.
    const bool pass = numeric_pass;
    const std::size_t compressed_kv_bytes =
        2U * (history_values * sizeof(std::uint8_t) +
              scale_values * sizeof(std::uint16_t));

    auto EmitTiming = [](const TimedRun& timing) {
      std::cout << "{\"partial_ms\":" << timing.partial_ms
                << ",\"quantize_ms\":" << timing.quantize_ms
                << ",\"reduce_ms\":" << timing.reduce_ms
                << ",\"total_ms\":" << timing.total_ms << "}";
    };
    auto EmitDistribution = [&](const std::vector<TimedRun>& samples) {
      std::cout << "[";
      for (std::size_t index = 0; index < samples.size(); ++index) {
        if (index != 0U) std::cout << ",";
        EmitTiming(samples[index]);
      }
      std::cout << "]";
    };
    std::cout << std::boolalpha << std::setprecision(12) << "{"
              << "\"algorithm\":\""
              << (e4m3 ? "scaled_e4m3_block32_gqa_fused"
                        : "int8_block32_gqa_fused") << "\","
              << "\"chunk_tokens\":" << kChunkTokens << ","
              << "\"component_cap_ms\":" << kComponentCapMs << ","
              << "\"compressed_kv_bytes\":" << compressed_kv_bytes << ","
              << "\"confirm\":";
    EmitTiming(confirm);
    std::cout << ",\"context_tokens\":" << kContextTokens << ","
              << "\"decode_table_entries\":" << (e4m3 ? 256 : 0) << ","
              << "\"device\":\"" << DeviceString(device, CL_DEVICE_NAME)
              << "\","
              << "\"finite\":" << numeric.finite << ","
              << "\"fp8_format\":\"" << (e4m3 ? "e4m3_bias7" : "none")
              << "\","
              << "\"gqa_group\":" << kGqaGroup << ","
              << "\"head_dim\":" << kHeadDim << ","
              << "\"kv_dtype\":\""
              << (e4m3 ? "e4m3_block32_fp16_scale"
                        : "int8_block32_fp16_scale") << "\","
              << "\"kv_head_count\":" << kKvHeads << ","
              << "\"max_abs\":" << numeric.max_abs << ","
              << "\"numeric_pass\":" << numeric_pass << ","
              << "\"output_cosine\":" << numeric.cosine << ","
              << "\"output_relative_l2\":" << numeric.relative_l2 << ","
              << "\"output_rmse\":" << numeric.rmse << ","
              << "\"partial_bytes\":"
              << partial_values * sizeof(float) +
                     2U * meta_values * sizeof(float)
              << ","
              << "\"q_head_count\":" << kQHeads << ","
              << "\"quant_group\":" << kQuantGroup << ","
              << "\"quantization_included\":true,"
              << "\"quantization_rounding\":\""
              << (e4m3 ? "rne_e4m3_finite" : "rne_clamp_-127_127")
              << "\","
              << "\"repeat\":";
    EmitTiming(repeat);
    std::cout << ",\"repeat_samples\":";
    EmitDistribution(repeat_samples);
    std::cout << ",\"confirm_samples\":";
    EmitDistribution(confirm_samples);
    std::cout << ",\"point_estimate_rate_pass\":"
              << point_estimate_rate_pass << ","
              << "\"required_checks_passed\":" << pass << ","
              << "\"scale_dtype\":\"fp16\","
              << "\"spread\":" << spread << ","
              << "\"subgroup_size\":32,"
              << "\"timing_pass\":" << point_estimate_rate_pass << "}"
              << std::endl;

    for (cl_mem buffer : {
             q_buffer, gate_buffer, current_k_buffer, current_v_buffer, k_f32,
             v_f32, k_i8_buffer, v_i8_buffer, k_scale_buffer, v_scale_buffer,
             scores, reference_output, compressed_output, partial_max,
             partial_sum, partial_output}) {
      clReleaseMemObject(buffer);
    }
    for (cl_kernel kernel : {
             quantize, partial, reduce, reference_score, reference_apply}) {
      clReleaseKernel(kernel);
    }
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return pass ? 0 : 2;
  } catch (const std::exception& exception) {
    std::cerr << "iq36-compressed-gqa-i8-kv-decode: "
              << exception.what() << '\n';
    return 4;
  }
}
