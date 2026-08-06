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

#ifndef IQ36_ADAPTIVE_COMPONENT_TOPK
#error "IQ36_ADAPTIVE_COMPONENT_TOPK is required"
#endif

namespace {

constexpr cl_uint kBaseContext = 32768;
constexpr cl_uint kTargetContext = 65536;
constexpr cl_uint kHotTokens = 16384;
constexpr cl_uint kHeadDim = 256;
constexpr cl_uint kQHeads = 16;
constexpr cl_uint kKvHeads = 2;
constexpr cl_uint kGqaGroup = 8;
constexpr cl_uint kQuantGroup = 32;
constexpr cl_uint kScaleGroups = kHeadDim / kQuantGroup;
constexpr cl_uint kKeyPackWords = kHeadDim / 4U;
constexpr cl_uint kTokenTile = 16;
constexpr cl_uint kChunkTokens = 512;
constexpr cl_uint kLocalTopK = 64;
constexpr cl_uint kTopK = IQ36_ADAPTIVE_COMPONENT_TOPK;
constexpr cl_uint kPartialLocal = 128;
constexpr cl_uint kSelectLocal = 256;
constexpr cl_uint kCorrectLocal = 128;
constexpr cl_uint kUpdateLocal = 32;
constexpr int kWarmups = 4;
constexpr int kSamples = 20;
static_assert(kTopK == 256U || kTopK == 512U);

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
  Check(clGetPlatformIDs(0, nullptr, &platform_count),
        "clGetPlatformIDs(count)");
  std::vector<cl_platform_id> platforms(platform_count);
  Check(clGetPlatformIDs(platform_count, platforms.data(), nullptr),
        "clGetPlatformIDs(values)");
  for (cl_platform_id platform : platforms) {
    cl_uint device_count = 0;
    const cl_int count_status = clGetDeviceIDs(
        platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &device_count);
    if (count_status == CL_DEVICE_NOT_FOUND || device_count == 0) continue;
    Check(count_status, "clGetDeviceIDs(count)");
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

float HalfToFloat(std::uint16_t value) {
  const std::uint32_t sign = static_cast<std::uint32_t>(value & 0x8000U) << 16U;
  std::uint32_t exponent = (value >> 10U) & 0x1fU;
  std::uint32_t mantissa = value & 0x03ffU;
  std::uint32_t bits = 0;
  if (exponent == 0U) {
    if (mantissa == 0U) {
      bits = sign;
    } else {
      int shift = 0;
      while ((mantissa & 0x0400U) == 0U) {
        mantissa <<= 1U;
        ++shift;
      }
      mantissa &= 0x03ffU;
      const std::uint32_t f32_exponent =
          static_cast<std::uint32_t>(127 - 15 - shift + 1);
      bits = sign | (f32_exponent << 23U) | (mantissa << 13U);
    }
  } else if (exponent == 0x1fU) {
    bits = sign | 0x7f800000U | (mantissa << 13U);
  } else {
    exponent += 127U - 15U;
    bits = sign | (exponent << 23U) | (mantissa << 13U);
  }
  float result = 0.0f;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

std::uint16_t OrderedHalf(std::uint16_t bits) {
  return (bits & 0x8000U) != 0U
      ? static_cast<std::uint16_t>(~bits)
      : static_cast<std::uint16_t>(bits ^ 0x8000U);
}

std::uint32_t Mix(std::uint32_t value) {
  value ^= value >> 16U;
  value *= 0x7feb352dU;
  value ^= value >> 15U;
  value *= 0x846ca68bU;
  value ^= value >> 16U;
  return value;
}

float UnitValue(cl_uint token, cl_uint flat_dim, std::uint32_t salt) {
  const std::uint32_t hash = Mix(
      token * 0x9e3779b9U ^ flat_dim * 0x85ebca6bU ^ salt);
  return static_cast<float>(static_cast<int>(hash & 0xffffU) - 32768) /
      32768.0f;
}

float KeyValue(cl_uint token, cl_uint kv_head, cl_uint dim) {
  return UnitValue(token, kv_head * kHeadDim + dim, 0x51a7d3e1U);
}

float ValueValue(cl_uint token, cl_uint kv_head, cl_uint dim) {
  return 0.1f + 0.5f * UnitValue(
      token, kv_head * kHeadDim + dim, 0x7f4a7c15U);
}

std::size_t DenseIndex(
    cl_uint token, cl_uint kv_head, cl_uint dim) {
  return (static_cast<std::size_t>(token) * kKvHeads + kv_head) *
      kHeadDim + dim;
}

struct Timing {
  double scan_ms = 0.0;
  double select_ms = 0.0;
  double correct_ms = 0.0;
  double update_ms = 0.0;
  double total_ms = 0.0;
};

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
  double reference_l2 = 0.0;
  double candidate_l2 = 0.0;
  double difference_l2 = 0.0;
  double max_abs = 0.0;
  bool finite = true;
  for (std::size_t index = 0; index < reference.size(); ++index) {
    const double ref = reference[index];
    const double cand = candidate[index];
    finite = finite && std::isfinite(ref) && std::isfinite(cand);
    const double difference = cand - ref;
    dot += ref * cand;
    reference_l2 += ref * ref;
    candidate_l2 += cand * cand;
    difference_l2 += difference * difference;
    max_abs = std::max(max_abs, std::abs(difference));
  }
  return {
      dot / (std::sqrt(reference_l2) * std::sqrt(candidate_l2)),
      std::sqrt(difference_l2 / reference_l2),
      std::sqrt(difference_l2 / static_cast<double>(reference.size())),
      max_abs,
      finite,
  };
}

struct Component {
  cl_uint context_tokens = 0;
  cl_uint cold_tokens = 0;
  cl_uint chunk_count = 0;
  cl_uint cold_chunk_count = 0;
  cl_uint correction_partitions = 0;
  cl_uint union_words = 0;
  cl_program program = nullptr;
  cl_kernel partial = nullptr;
  cl_kernel select = nullptr;
  cl_kernel correct = nullptr;
  cl_kernel update = nullptr;
  cl_kernel reference_score = nullptr;
  cl_kernel reference_apply = nullptr;
  cl_mem query = nullptr;
  cl_mem evicted_k = nullptr;
  cl_mem evicted_v = nullptr;
  cl_mem current_k = nullptr;
  cl_mem current_v = nullptr;
  cl_mem cold_k = nullptr;
  cl_mem cold_v = nullptr;
  cl_mem cold_k_scales = nullptr;
  cl_mem cold_v_scales = nullptr;
  cl_mem hot_k = nullptr;
  cl_mem hot_v = nullptr;
  cl_mem exact_cold_k = nullptr;
  cl_mem exact_cold_v = nullptr;
  cl_mem exact_key = nullptr;
  cl_mem exact_value = nullptr;
  cl_mem partial_max = nullptr;
  cl_mem partial_sum = nullptr;
  cl_mem partial_output = nullptr;
  cl_mem scan_cold_score = nullptr;
  cl_mem local_candidates = nullptr;
  cl_mem union_bits = nullptr;
  cl_mem aggregate_max = nullptr;
  cl_mem aggregate_sum = nullptr;
  cl_mem aggregate_numerator = nullptr;
  cl_mem correction_partial_max = nullptr;
  cl_mem correction_partial_sum = nullptr;
  cl_mem correction_partial_numerator = nullptr;
  cl_mem correction_completion = nullptr;
  cl_mem output = nullptr;
  cl_mem approximate_score = nullptr;
  cl_mem exact_score = nullptr;
  cl_mem adaptive_reference_output = nullptr;
  cl_mem exact_reference_output = nullptr;
  std::vector<cl_mem> owned_buffers;
};

cl_program BuildProgram(cl_context context, cl_device_id device,
                        const std::string& source, cl_uint context_tokens) {
  cl_int status = CL_SUCCESS;
  const char* source_data = source.data();
  const std::size_t source_size = source.size();
  cl_program program = clCreateProgramWithSource(
      context, 1, &source_data, &source_size, &status);
  Check(status, "clCreateProgramWithSource");
  const std::string options =
      "-cl-std=CL3.0 -DIQ36_ADAPTIVE_ATTENTION=1 "
      "-DIQ36_CONTEXT_TOKENS=" + std::to_string(context_tokens) +
      " -DIQ36_HOT_TOKENS=" + std::to_string(kHotTokens) +
      " -DIQ36_QUANT_GROUP=32 -DIQ36_ADAPTIVE_TOPK=" +
      std::to_string(kTopK);
  status = clBuildProgram(
      program, 1, &device, options.c_str(), nullptr, nullptr);
  if (status != CL_SUCCESS) {
    std::size_t log_bytes = 0;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                          0, nullptr, &log_bytes);
    std::string log(log_bytes, '\0');
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                          log_bytes, log.data(), nullptr);
    std::cerr << log << '\n';
    Check(status, "clBuildProgram");
  }
  return program;
}

cl_kernel Kernel(cl_program program, const char* name) {
  cl_int status = CL_SUCCESS;
  cl_kernel result = clCreateKernel(program, name, &status);
  Check(status, std::string("clCreateKernel(") + name + ")");
  return result;
}

template <typename T>
cl_mem HostBuffer(cl_context context, cl_mem_flags flags,
                  std::vector<T>& values) {
  return Buffer(context, flags | CL_MEM_COPY_HOST_PTR,
                values.size() * sizeof(T), values.data());
}

Component CreateComponent(
    cl_context context, cl_device_id device, cl_mem query_buffer,
    const std::string& source, cl_uint context_tokens) {
  Component component;
  component.context_tokens = context_tokens;
  component.cold_tokens = context_tokens - kHotTokens;
  component.chunk_count = context_tokens / kChunkTokens;
  component.cold_chunk_count = component.cold_tokens / kChunkTokens;
  component.correction_partitions = component.cold_chunk_count;
  component.union_words = (component.cold_tokens + 31U) / 32U;
  component.query = query_buffer;
  component.program = BuildProgram(context, device, source, context_tokens);
  component.partial = Kernel(
      component.program, "iq36_direct_i8_hotcold_partial");
  component.select = Kernel(
      component.program, "iq36_adaptive_select_reduce_union");
  component.correct = Kernel(
      component.program, "iq36_adaptive_correct_normalize");
  component.update = Kernel(component.program, "iq36_direct_i8_update_state");
  component.reference_score = Kernel(
      component.program, "iq36_adaptive_reference_score");
  component.reference_apply = Kernel(
      component.program, "iq36_adaptive_reference_apply");

  const std::size_t history_values =
      static_cast<std::size_t>(context_tokens) * kKvHeads * kHeadDim;
  const std::size_t cold_values =
      static_cast<std::size_t>(component.cold_tokens) * kKvHeads * kHeadDim;
  const std::size_t hot_values =
      static_cast<std::size_t>(kHotTokens) * kKvHeads * kHeadDim;
  const std::size_t cold_scale_values =
      static_cast<std::size_t>(component.cold_tokens) * kKvHeads *
      kScaleGroups;
  const std::size_t cold_k_words = cold_values / 4U;
  const std::size_t hot_k_words = hot_values / 2U;
  const std::size_t output_values =
      static_cast<std::size_t>(kQHeads) * kHeadDim;
  const std::size_t group_count =
      static_cast<std::size_t>(kKvHeads) * component.chunk_count;
  const std::size_t meta_values = group_count * kGqaGroup;
  const std::size_t partial_values = meta_values * kHeadDim;
  const std::size_t candidate_values =
      static_cast<std::size_t>(kQHeads) * component.cold_chunk_count *
      kLocalTopK;
  const std::size_t cold_score_values =
      static_cast<std::size_t>(kQHeads) * component.cold_tokens;
  const std::size_t union_values =
      static_cast<std::size_t>(kKvHeads) * component.union_words;
  const std::size_t correction_meta_values =
      static_cast<std::size_t>(kKvHeads) * component.correction_partitions *
      kGqaGroup;

  std::vector<std::uint16_t> exact_key(history_values);
  std::vector<std::uint16_t> exact_value(history_values);
  for (cl_uint token = 0; token < context_tokens; ++token) {
    for (cl_uint kv_head = 0; kv_head < kKvHeads; ++kv_head) {
      for (cl_uint dim = 0; dim < kHeadDim; ++dim) {
        const std::size_t index = DenseIndex(token, kv_head, dim);
        exact_key[index] = FloatToHalf(KeyValue(token, kv_head, dim));
        exact_value[index] = FloatToHalf(ValueValue(token, kv_head, dim));
      }
    }
  }

  std::vector<std::uint16_t> exact_cold_k(cold_values);
  std::vector<std::uint16_t> exact_cold_v(cold_values);
  std::vector<std::uint32_t> cold_k(cold_k_words, 0U);
  std::vector<std::int8_t> cold_v(cold_values);
  std::vector<std::uint16_t> cold_k_scales(cold_scale_values);
  std::vector<std::uint16_t> cold_v_scales(cold_scale_values);
  auto* cold_k_bytes = reinterpret_cast<std::int8_t*>(cold_k.data());
  const std::size_t cold_k_words_per_head =
      cold_values / 4U / kKvHeads;
  for (cl_uint token = 0; token < component.cold_tokens; ++token) {
    const cl_uint token_block = token / kTokenTile;
    const cl_uint token_lane = token & (kTokenTile - 1U);
    for (cl_uint kv_head = 0; kv_head < kKvHeads; ++kv_head) {
      for (cl_uint group = 0; group < kScaleGroups; ++group) {
        float key_max = 0.0f;
        float value_max = 0.0f;
        for (cl_uint lane = 0; lane < kQuantGroup; ++lane) {
          const cl_uint dim = group * kQuantGroup + lane;
          const std::size_t dense = DenseIndex(token, kv_head, dim);
          key_max = std::max(key_max, std::abs(HalfToFloat(exact_key[dense])));
          value_max = std::max(
              value_max, std::abs(HalfToFloat(exact_value[dense])));
        }
        const float key_scale = key_max == 0.0f ? 1.0f : key_max / 127.0f;
        const float value_scale =
            value_max == 0.0f ? 1.0f : value_max / 127.0f;
        const std::size_t scale_index =
            (static_cast<std::size_t>(kv_head) * kScaleGroups + group) *
                component.cold_tokens + token;
        cold_k_scales[scale_index] = FloatToHalf(key_scale);
        cold_v_scales[scale_index] = FloatToHalf(value_scale);
        for (cl_uint lane = 0; lane < kQuantGroup; ++lane) {
          const cl_uint dim = group * kQuantGroup + lane;
          const std::size_t dense = DenseIndex(token, kv_head, dim);
          const float key = HalfToFloat(exact_key[dense]);
          const float value = HalfToFloat(exact_value[dense]);
          const auto quantized_key = static_cast<std::int8_t>(std::clamp(
              static_cast<int>(std::nearbyint(key / key_scale)), -127, 127));
          const auto quantized_value = static_cast<std::int8_t>(std::clamp(
              static_cast<int>(std::nearbyint(value / value_scale)),
              -127, 127));
          const std::size_t word =
              static_cast<std::size_t>(kv_head) * cold_k_words_per_head +
              ((static_cast<std::size_t>(token_block) * kKeyPackWords +
                dim / 4U) * kTokenTile + token_lane);
          cold_k_bytes[word * 4U + dim % 4U] = quantized_key;
          cold_v[(static_cast<std::size_t>(kv_head) * kHeadDim + dim) *
              component.cold_tokens + token] = quantized_value;
          const std::size_t exact_cold =
              (static_cast<std::size_t>(kv_head) * component.cold_tokens +
               token) * kHeadDim + dim;
          exact_cold_k[exact_cold] = exact_key[dense];
          exact_cold_v[exact_cold] = exact_value[dense];
        }
      }
    }
  }

  std::vector<std::uint32_t> hot_k(hot_k_words, 0U);
  std::vector<std::uint16_t> hot_v(hot_values);
  const std::size_t hot_words_per_head =
      static_cast<std::size_t>(kHotTokens) * kHeadDim / 2U;
  for (cl_uint hot_token = 0; hot_token < kHotTokens; ++hot_token) {
    const cl_uint token = component.cold_tokens + hot_token;
    const cl_uint token_block = hot_token / kTokenTile;
    const cl_uint token_lane = hot_token & (kTokenTile - 1U);
    for (cl_uint kv_head = 0; kv_head < kKvHeads; ++kv_head) {
      for (cl_uint pair = 0; pair < kHeadDim / 2U; ++pair) {
        const cl_uint dim = pair * 2U;
        const std::size_t word =
            static_cast<std::size_t>(kv_head) * hot_words_per_head +
            ((static_cast<std::size_t>(token_block) * (kHeadDim / 2U) +
              pair) * kTokenTile + token_lane);
        hot_k[word] = exact_key[DenseIndex(token, kv_head, dim)] |
            (static_cast<std::uint32_t>(
                exact_key[DenseIndex(token, kv_head, dim + 1U)]) << 16U);
      }
      for (cl_uint dim = 0; dim < kHeadDim; ++dim) {
        hot_v[(static_cast<std::size_t>(kv_head) * kHeadDim + dim) *
            kHotTokens + hot_token] = exact_value[
                DenseIndex(token, kv_head, dim)];
      }
    }
  }

  std::vector<std::uint16_t> evicted_k(kKvHeads * kHeadDim);
  std::vector<std::uint16_t> evicted_v(kKvHeads * kHeadDim);
  std::vector<std::uint16_t> current_k(kKvHeads * kHeadDim);
  std::vector<std::uint16_t> current_v(kKvHeads * kHeadDim);
  for (cl_uint kv_head = 0; kv_head < kKvHeads; ++kv_head) {
    for (cl_uint dim = 0; dim < kHeadDim; ++dim) {
      const std::size_t index = kv_head * kHeadDim + dim;
      evicted_k[index] = exact_key[
          DenseIndex(component.cold_tokens - 1U, kv_head, dim)];
      evicted_v[index] = exact_value[
          DenseIndex(component.cold_tokens - 1U, kv_head, dim)];
      current_k[index] = exact_key[
          DenseIndex(context_tokens - 1U, kv_head, dim)];
      current_v[index] = exact_value[
          DenseIndex(context_tokens - 1U, kv_head, dim)];
    }
  }

  auto Own = [&](cl_mem buffer) {
    component.owned_buffers.push_back(buffer);
    return buffer;
  };
  component.evicted_k = Own(HostBuffer(
      context, CL_MEM_READ_ONLY, evicted_k));
  component.evicted_v = Own(HostBuffer(
      context, CL_MEM_READ_ONLY, evicted_v));
  component.current_k = Own(HostBuffer(
      context, CL_MEM_READ_ONLY, current_k));
  component.current_v = Own(HostBuffer(
      context, CL_MEM_READ_ONLY, current_v));
  component.cold_k = Own(HostBuffer(
      context, CL_MEM_READ_WRITE, cold_k));
  component.cold_v = Own(HostBuffer(
      context, CL_MEM_READ_WRITE, cold_v));
  component.cold_k_scales = Own(HostBuffer(
      context, CL_MEM_READ_WRITE, cold_k_scales));
  component.cold_v_scales = Own(HostBuffer(
      context, CL_MEM_READ_WRITE, cold_v_scales));
  component.hot_k = Own(HostBuffer(
      context, CL_MEM_READ_WRITE, hot_k));
  component.hot_v = Own(HostBuffer(
      context, CL_MEM_READ_WRITE, hot_v));
  component.exact_cold_k = Own(HostBuffer(
      context, CL_MEM_READ_WRITE, exact_cold_k));
  component.exact_cold_v = Own(HostBuffer(
      context, CL_MEM_READ_WRITE, exact_cold_v));
  component.exact_key = Own(HostBuffer(
      context, CL_MEM_READ_ONLY, exact_key));
  component.exact_value = Own(HostBuffer(
      context, CL_MEM_READ_ONLY, exact_value));
  component.partial_max = Own(Buffer(
      context, CL_MEM_READ_WRITE, meta_values * sizeof(float)));
  component.partial_sum = Own(Buffer(
      context, CL_MEM_READ_WRITE, meta_values * sizeof(float)));
  component.partial_output = Own(Buffer(
      context, CL_MEM_READ_WRITE, partial_values * sizeof(float)));
  component.scan_cold_score = Own(Buffer(
      context, CL_MEM_READ_WRITE,
      cold_score_values * sizeof(std::uint16_t)));
  component.local_candidates = Own(Buffer(
      context, CL_MEM_READ_WRITE, candidate_values * sizeof(std::uint32_t)));
  component.union_bits = Own(Buffer(
      context, CL_MEM_READ_WRITE, union_values * sizeof(std::uint32_t)));
  component.aggregate_max = Own(Buffer(
      context, CL_MEM_READ_WRITE, kQHeads * sizeof(float)));
  component.aggregate_sum = Own(Buffer(
      context, CL_MEM_READ_WRITE, kQHeads * sizeof(float)));
  component.aggregate_numerator = Own(Buffer(
      context, CL_MEM_READ_WRITE, output_values * sizeof(float)));
  component.correction_partial_max = Own(Buffer(
      context, CL_MEM_READ_WRITE,
      correction_meta_values * sizeof(float)));
  component.correction_partial_sum = Own(Buffer(
      context, CL_MEM_READ_WRITE,
      correction_meta_values * sizeof(float)));
  component.correction_partial_numerator = Own(Buffer(
      context, CL_MEM_READ_WRITE,
      correction_meta_values * kHeadDim * sizeof(float)));
  component.correction_completion = Own(Buffer(
      context, CL_MEM_READ_WRITE, kKvHeads * sizeof(std::uint32_t)));
  component.output = Own(Buffer(
      context, CL_MEM_READ_WRITE, output_values * sizeof(float)));
  component.approximate_score = Own(Buffer(
      context, CL_MEM_READ_WRITE,
      static_cast<std::size_t>(kQHeads) * context_tokens * sizeof(float)));
  component.exact_score = Own(Buffer(
      context, CL_MEM_READ_WRITE,
      static_cast<std::size_t>(kQHeads) * context_tokens * sizeof(float)));
  component.adaptive_reference_output = Own(Buffer(
      context, CL_MEM_READ_WRITE, output_values * sizeof(float)));
  component.exact_reference_output = Own(Buffer(
      context, CL_MEM_READ_WRITE, output_values * sizeof(float)));

  Arg(component.partial, 0, component.query);
  Arg(component.partial, 1, component.cold_k);
  Arg(component.partial, 2, component.cold_v);
  Arg(component.partial, 3, component.cold_k_scales);
  Arg(component.partial, 4, component.cold_v_scales);
  Arg(component.partial, 5, component.hot_k);
  Arg(component.partial, 6, component.hot_v);
  Arg(component.partial, 7, component.partial_max);
  Arg(component.partial, 8, component.partial_sum);
  Arg(component.partial, 9, component.partial_output);
  Arg(component.partial, 10, component.scan_cold_score);
  Arg(component.partial, 11, component.local_candidates);
  Arg(component.partial, 12, component.union_bits);
  Arg(component.partial, 13, component.correction_completion);
  Arg(component.select, 0, component.partial_max);
  Arg(component.select, 1, component.partial_sum);
  Arg(component.select, 2, component.partial_output);
  Arg(component.select, 3, component.local_candidates);
  Arg(component.select, 4, component.union_bits);
  Arg(component.select, 5, component.aggregate_max);
  Arg(component.select, 6, component.aggregate_sum);
  Arg(component.select, 7, component.aggregate_numerator);
  Arg(component.correct, 0, component.query);
  Arg(component.correct, 1, component.cold_k);
  Arg(component.correct, 2, component.cold_v);
  Arg(component.correct, 3, component.cold_k_scales);
  Arg(component.correct, 4, component.cold_v_scales);
  Arg(component.correct, 5, component.exact_cold_k);
  Arg(component.correct, 6, component.exact_cold_v);
  Arg(component.correct, 7, component.scan_cold_score);
  Arg(component.correct, 8, component.union_bits);
  Arg(component.correct, 9, component.aggregate_max);
  Arg(component.correct, 10, component.aggregate_sum);
  Arg(component.correct, 11, component.aggregate_numerator);
  Arg(component.correct, 12, component.correction_partial_max);
  Arg(component.correct, 13, component.correction_partial_sum);
  Arg(component.correct, 14, component.correction_partial_numerator);
  Arg(component.correct, 15, component.correction_completion);
  Arg(component.correct, 16, component.output);
  Arg(component.update, 0, component.evicted_k);
  Arg(component.update, 1, component.evicted_v);
  Arg(component.update, 2, component.current_k);
  Arg(component.update, 3, component.current_v);
  Arg(component.update, 4, component.cold_k);
  Arg(component.update, 5, component.cold_v);
  Arg(component.update, 6, component.cold_k_scales);
  Arg(component.update, 7, component.cold_v_scales);
  Arg(component.update, 8, component.hot_k);
  Arg(component.update, 9, component.hot_v);
  Arg(component.update, 10, component.exact_cold_k);
  Arg(component.update, 11, component.exact_cold_v);
  Arg(component.reference_score, 0, component.query);
  Arg(component.reference_score, 1, component.cold_k);
  Arg(component.reference_score, 2, component.cold_k_scales);
  Arg(component.reference_score, 3, component.hot_k);
  Arg(component.reference_score, 4, component.exact_key);
  Arg(component.reference_score, 5, component.approximate_score);
  Arg(component.reference_score, 6, component.exact_score);
  Arg(component.reference_apply, 0, component.approximate_score);
  Arg(component.reference_apply, 1, component.exact_score);
  Arg(component.reference_apply, 2, component.cold_v);
  Arg(component.reference_apply, 3, component.cold_v_scales);
  Arg(component.reference_apply, 4, component.hot_v);
  Arg(component.reference_apply, 5, component.exact_value);
  Arg(component.reference_apply, 6, component.union_bits);
  Arg(component.reference_apply, 7, component.adaptive_reference_output);
  Arg(component.reference_apply, 8, component.exact_reference_output);
  return component;
}

Timing Run(cl_command_queue queue, Component& component) {
  const std::size_t partial_global =
      static_cast<std::size_t>(kKvHeads) * component.chunk_count *
      kPartialLocal;
  const std::size_t select_global =
      static_cast<std::size_t>(kQHeads) * kSelectLocal;
  const std::size_t correct_global =
      static_cast<std::size_t>(kKvHeads) * component.correction_partitions *
      kCorrectLocal;
  const std::size_t update_global =
      static_cast<std::size_t>(kKvHeads) * (kScaleGroups + kScaleGroups) *
      kUpdateLocal;
  const std::size_t partial_local = kPartialLocal;
  const std::size_t select_local = kSelectLocal;
  const std::size_t correct_local = kCorrectLocal;
  const std::size_t update_local = kUpdateLocal;
  cl_event partial_event = nullptr;
  cl_event select_event = nullptr;
  cl_event correct_event = nullptr;
  cl_event update_event = nullptr;
  Check(clEnqueueNDRangeKernel(
      queue, component.partial, 1, nullptr, &partial_global, &partial_local,
      0, nullptr, &partial_event), "clEnqueueNDRangeKernel(adaptive partial)");
  Check(clEnqueueNDRangeKernel(
      queue, component.select, 1, nullptr, &select_global, &select_local,
      0, nullptr, &select_event), "clEnqueueNDRangeKernel(adaptive select)");
  Check(clEnqueueNDRangeKernel(
      queue, component.correct, 1, nullptr, &correct_global, &correct_local,
      0, nullptr, &correct_event), "clEnqueueNDRangeKernel(adaptive correct)");
  Check(clEnqueueNDRangeKernel(
      queue, component.update, 1, nullptr, &update_global, &update_local,
      0, nullptr, &update_event), "clEnqueueNDRangeKernel(adaptive update)");
  Check(clFinish(queue), "clFinish(adaptive component)");
  Timing result;
  result.scan_ms = EventMs(partial_event);
  result.select_ms = EventMs(select_event);
  result.correct_ms = EventMs(correct_event);
  result.update_ms = EventMs(update_event);
  result.total_ms = result.scan_ms + result.select_ms +
      result.correct_ms + result.update_ms;
  clReleaseEvent(partial_event);
  clReleaseEvent(select_event);
  clReleaseEvent(correct_event);
  clReleaseEvent(update_event);
  return result;
}

template <typename T>
std::vector<T> ReadBuffer(
    cl_command_queue queue, cl_mem buffer, std::size_t count) {
  std::vector<T> values(count);
  Check(clEnqueueReadBuffer(
      queue, buffer, CL_TRUE, 0, count * sizeof(T), values.data(),
      0, nullptr, nullptr), "clEnqueueReadBuffer");
  return values;
}

struct Validation {
  Numeric implementation_vs_adaptive;
  Numeric adaptive_vs_exact;
  bool candidate_shape_pass = false;
  bool union_exact = false;
  bool union_deterministic = false;
  std::vector<unsigned> union_rows;
};

std::vector<std::uint32_t> ExpectedUnion(
    const Component& component,
    const std::vector<std::uint32_t>& candidates,
    bool* shape_pass) {
  std::vector<std::uint32_t> result(
      static_cast<std::size_t>(kKvHeads) * component.union_words, 0U);
  bool shape = true;
  const std::size_t per_head =
      static_cast<std::size_t>(component.cold_chunk_count) * kLocalTopK;
  auto Better = [](std::uint32_t left, std::uint32_t right) {
    const auto left_score = OrderedHalf(
        static_cast<std::uint16_t>(left >> 16U));
    const auto right_score = OrderedHalf(
        static_cast<std::uint16_t>(right >> 16U));
    if (left_score != right_score) return left_score > right_score;
    return static_cast<std::uint16_t>(left) <
        static_cast<std::uint16_t>(right);
  };
  for (cl_uint q_head = 0; q_head < kQHeads; ++q_head) {
    std::vector<std::uint32_t> pool;
    pool.reserve(per_head);
    for (cl_uint chunk = 0; chunk < component.cold_chunk_count; ++chunk) {
      std::vector<bool> seen(kChunkTokens, false);
      for (cl_uint row = 0; row < kLocalTopK; ++row) {
        const std::size_t offset =
            static_cast<std::size_t>(q_head) * per_head +
            static_cast<std::size_t>(chunk) * kLocalTopK + row;
        const std::uint32_t record = candidates[offset];
        const cl_uint token = record & 0xffffU;
        const bool inside = token >= chunk * kChunkTokens &&
            token < (chunk + 1U) * kChunkTokens;
        if (!inside || (inside && seen[token - chunk * kChunkTokens])) {
          shape = false;
        }
        if (inside) seen[token - chunk * kChunkTokens] = true;
        pool.push_back(record);
      }
    }
    std::sort(pool.begin(), pool.end(), Better);
    const cl_uint kv_head = q_head / kGqaGroup;
    for (cl_uint row = 0; row < kTopK; ++row) {
      const cl_uint token = pool[row] & 0xffffU;
      result[static_cast<std::size_t>(kv_head) * component.union_words +
          token / 32U] |= 1U << (token & 31U);
    }
  }
  *shape_pass = shape;
  return result;
}

Validation Validate(cl_command_queue queue, Component& component) {
  const std::size_t score_global =
      static_cast<std::size_t>(kQHeads) * component.context_tokens;
  const std::size_t score_local = 256U;
  const std::size_t apply_global =
      static_cast<std::size_t>(kQHeads) * kHeadDim;
  const std::size_t apply_local = 256U;
  Check(clEnqueueNDRangeKernel(
      queue, component.reference_score, 1, nullptr, &score_global,
      &score_local, 0, nullptr, nullptr),
      "clEnqueueNDRangeKernel(adaptive reference score)");
  Check(clEnqueueNDRangeKernel(
      queue, component.reference_apply, 1, nullptr, &apply_global,
      &apply_local, 0, nullptr, nullptr),
      "clEnqueueNDRangeKernel(adaptive reference apply)");
  Check(clFinish(queue), "clFinish(adaptive reference)");

  const std::size_t output_values =
      static_cast<std::size_t>(kQHeads) * kHeadDim;
  const std::size_t candidate_values =
      static_cast<std::size_t>(kQHeads) * component.cold_chunk_count *
      kLocalTopK;
  const std::size_t union_values =
      static_cast<std::size_t>(kKvHeads) * component.union_words;
  const auto output = ReadBuffer<float>(queue, component.output, output_values);
  const auto adaptive = ReadBuffer<float>(
      queue, component.adaptive_reference_output, output_values);
  const auto exact = ReadBuffer<float>(
      queue, component.exact_reference_output, output_values);
  const auto candidates = ReadBuffer<std::uint32_t>(
      queue, component.local_candidates, candidate_values);
  const auto union_first = ReadBuffer<std::uint32_t>(
      queue, component.union_bits, union_values);
  bool shape_pass = false;
  const auto expected = ExpectedUnion(component, candidates, &shape_pass);
  (void)Run(queue, component);
  const auto union_second = ReadBuffer<std::uint32_t>(
      queue, component.union_bits, union_values);
  Validation result;
  result.implementation_vs_adaptive = Compare(adaptive, output);
  result.adaptive_vs_exact = Compare(exact, adaptive);
  result.candidate_shape_pass = shape_pass;
  result.union_exact = expected == union_first;
  result.union_deterministic = union_first == union_second;
  for (cl_uint kv_head = 0; kv_head < kKvHeads; ++kv_head) {
    unsigned count = 0;
    for (cl_uint word = 0; word < component.union_words; ++word) {
      count += static_cast<unsigned>(__builtin_popcount(
          union_first[static_cast<std::size_t>(kv_head) *
              component.union_words + word]));
    }
    result.union_rows.push_back(count);
  }
  return result;
}

double MedianTotal(const std::vector<Timing>& values) {
  std::vector<double> totals;
  totals.reserve(values.size());
  for (const auto& value : values) totals.push_back(value.total_ms);
  std::sort(totals.begin(), totals.end());
  return (totals[totals.size() / 2U - 1U] +
          totals[totals.size() / 2U]) / 2.0;
}

void Release(Component& component) {
  for (cl_mem buffer : component.owned_buffers) clReleaseMemObject(buffer);
  for (cl_kernel kernel : {
           component.partial, component.select, component.correct,
           component.update, component.reference_score,
           component.reference_apply}) {
    if (kernel != nullptr) clReleaseKernel(kernel);
  }
  if (component.program != nullptr) clReleaseProgram(component.program);
}

void EmitTiming(const Timing& value) {
  std::cout << "{\"scan_ms\":" << value.scan_ms
            << ",\"select_ms\":" << value.select_ms
            << ",\"correct_ms\":" << value.correct_ms
            << ",\"update_ms\":" << value.update_ms
            << ",\"total_ms\":" << value.total_ms << "}";
}

void EmitNumeric(const Numeric& value) {
  std::cout << "{\"cosine\":" << value.cosine
            << ",\"relative_l2\":" << value.relative_l2
            << ",\"rmse\":" << value.rmse
            << ",\"max_abs\":" << value.max_abs
            << ",\"finite\":" << value.finite << "}";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 2) {
      throw std::invalid_argument(
          "usage: iq36-adaptive-i8-hotcold-gqa-decode OPENCL_SOURCE");
    }
    const auto device = SelectGpu();
    cl_int status = CL_SUCCESS;
    cl_context context = clCreateContext(
        nullptr, 1, &device, nullptr, nullptr, &status);
    Check(status, "clCreateContext");
    cl_command_queue queue = clCreateCommandQueue(
        context, device, CL_QUEUE_PROFILING_ENABLE, &status);
    Check(status, "clCreateCommandQueue");
    std::vector<std::uint16_t> query(kQHeads * kHeadDim);
    for (cl_uint q_head = 0; q_head < kQHeads; ++q_head) {
      for (cl_uint dim = 0; dim < kHeadDim; ++dim) {
        query[q_head * kHeadDim + dim] = FloatToHalf(UnitValue(
            q_head, dim, 0x3c6ef372U));
      }
    }
    cl_mem query_buffer = HostBuffer(context, CL_MEM_READ_ONLY, query);
    const std::string source = ReadText(argv[1]);
    Component base = CreateComponent(
        context, device, query_buffer, source, kBaseContext);
    Component target = CreateComponent(
        context, device, query_buffer, source, kTargetContext);

    for (int warmup = 0; warmup < kWarmups; ++warmup) {
      (void)Run(queue, base);
      (void)Run(queue, target);
    }
    std::vector<Timing> base_samples;
    std::vector<Timing> target_samples;
    std::vector<std::string> orders;
    base_samples.reserve(kSamples);
    target_samples.reserve(kSamples);
    for (int sample = 0; sample < kSamples; ++sample) {
      if ((sample & 1) == 0) {
        orders.emplace_back("base_target");
        base_samples.push_back(Run(queue, base));
        target_samples.push_back(Run(queue, target));
      } else {
        orders.emplace_back("target_base");
        target_samples.push_back(Run(queue, target));
        base_samples.push_back(Run(queue, base));
      }
    }
    const Validation base_validation = Validate(queue, base);
    const Validation target_validation = Validate(queue, target);
    const bool numeric_pass =
        base_validation.implementation_vs_adaptive.finite &&
        target_validation.implementation_vs_adaptive.finite &&
        base_validation.implementation_vs_adaptive.relative_l2 <= 0.002 &&
        target_validation.implementation_vs_adaptive.relative_l2 <= 0.002 &&
        base_validation.adaptive_vs_exact.relative_l2 <= 0.005 &&
        target_validation.adaptive_vs_exact.relative_l2 <= 0.005;
    const bool selection_pass =
        base_validation.candidate_shape_pass &&
        target_validation.candidate_shape_pass &&
        base_validation.union_exact && target_validation.union_exact &&
        base_validation.union_deterministic &&
        target_validation.union_deterministic;
    const bool required = numeric_pass && selection_pass;

    std::cout << std::boolalpha << std::setprecision(12) << "{"
              << "\"algorithm\":\"adaptive_block32_i8_exact_f16_correction\","
              << "\"device\":\"" << DeviceString(device, CL_DEVICE_NAME)
              << "\","
              << "\"topk_per_query\":" << kTopK << ","
              << "\"local_topk_per_chunk\":" << kLocalTopK << ","
              << "\"chunk_tokens\":" << kChunkTokens << ","
              << "\"hot_tokens\":" << kHotTokens << ","
              << "\"base_context_tokens\":" << kBaseContext << ","
              << "\"target_context_tokens\":" << kTargetContext << ","
              << "\"sample_count\":" << kSamples << ","
              << "\"schedule\":\"interleaved_base_target_target_base\","
              << "\"dispatches_per_context\":4,"
              << "\"timed_scope\":\"scan_local_select_global_select_reduce_union_correct_normalize_ordered_update\","
              << "\"base_median_ms\":" << MedianTotal(base_samples) << ","
              << "\"target_median_ms\":" << MedianTotal(target_samples) << ","
              << "\"base_validation\":{"
              << "\"implementation_vs_adaptive\":";
    EmitNumeric(base_validation.implementation_vs_adaptive);
    std::cout << ",\"adaptive_vs_exact\":";
    EmitNumeric(base_validation.adaptive_vs_exact);
    std::cout << ",\"candidate_shape_pass\":"
              << base_validation.candidate_shape_pass
              << ",\"union_exact\":" << base_validation.union_exact
              << ",\"union_deterministic\":"
              << base_validation.union_deterministic
              << ",\"union_rows\":[";
    for (std::size_t index = 0; index < base_validation.union_rows.size();
         ++index) {
      if (index != 0U) std::cout << ",";
      std::cout << base_validation.union_rows[index];
    }
    std::cout << "]},\"target_validation\":{"
              << "\"implementation_vs_adaptive\":";
    EmitNumeric(target_validation.implementation_vs_adaptive);
    std::cout << ",\"adaptive_vs_exact\":";
    EmitNumeric(target_validation.adaptive_vs_exact);
    std::cout << ",\"candidate_shape_pass\":"
              << target_validation.candidate_shape_pass
              << ",\"union_exact\":" << target_validation.union_exact
              << ",\"union_deterministic\":"
              << target_validation.union_deterministic
              << ",\"union_rows\":[";
    for (std::size_t index = 0; index < target_validation.union_rows.size();
         ++index) {
      if (index != 0U) std::cout << ",";
      std::cout << target_validation.union_rows[index];
    }
    std::cout << "]},\"numeric_pass\":" << numeric_pass
              << ",\"selection_pass\":" << selection_pass
              << ",\"required_checks_passed\":" << required
              << ",\"paired_samples\":[";
    for (int index = 0; index < kSamples; ++index) {
      if (index != 0) std::cout << ",";
      std::cout << "{\"order\":\"" << orders[index]
                << "\",\"base\":";
      EmitTiming(base_samples[index]);
      std::cout << ",\"target\":";
      EmitTiming(target_samples[index]);
      std::cout << ",\"differential_ms\":"
                << target_samples[index].total_ms -
                       base_samples[index].total_ms
                << "}";
    }
    std::cout << "]}" << std::endl;

    Release(base);
    Release(target);
    clReleaseMemObject(query_buffer);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return required ? 0 : 2;
  } catch (const std::exception& exception) {
    std::cerr << "iq36-adaptive-i8-hotcold-gqa-decode: "
              << exception.what() << '\n';
    return 4;
  }
}
