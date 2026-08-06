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

#ifndef IQ36_COMPONENT_QUANT_GROUP
#define IQ36_COMPONENT_QUANT_GROUP 32
#endif
#ifndef IQ36_COMPONENT_KEY_QUANT_GROUP
#define IQ36_COMPONENT_KEY_QUANT_GROUP IQ36_COMPONENT_QUANT_GROUP
#endif
#ifndef IQ36_COMPONENT_VALUE_QUANT_GROUP
#define IQ36_COMPONENT_VALUE_QUANT_GROUP IQ36_COMPONENT_QUANT_GROUP
#endif
#ifndef IQ36_COMPONENT_HOT_TOKENS
#define IQ36_COMPONENT_HOT_TOKENS 8192
#endif
#ifndef IQ36_COMPONENT_UPDATE_AFTER_ATTENTION
#define IQ36_COMPONENT_UPDATE_AFTER_ATTENTION 0
#endif
#ifndef IQ36_COMPONENT_STORAGE_SPECIALIZED
#define IQ36_COMPONENT_STORAGE_SPECIALIZED 0
#endif

namespace {

constexpr cl_uint kContextTokens = 32768;
constexpr cl_uint kHotTokens = IQ36_COMPONENT_HOT_TOKENS;
constexpr cl_uint kColdTokens = kContextTokens - kHotTokens;
constexpr cl_uint kHeadDim = 256;
constexpr cl_uint kQHeads = 16;
constexpr cl_uint kKvHeads = 2;
constexpr cl_uint kGqaGroup = 8;
constexpr cl_uint kKeyQuantGroup = IQ36_COMPONENT_KEY_QUANT_GROUP;
constexpr cl_uint kValueQuantGroup = IQ36_COMPONENT_VALUE_QUANT_GROUP;
constexpr cl_uint kKeyScaleGroups = kHeadDim / kKeyQuantGroup;
constexpr cl_uint kValueScaleGroups = kHeadDim / kValueQuantGroup;
constexpr cl_uint kKeyPackWords = kHeadDim / 4U;
constexpr cl_uint kTokenTile = 16;
constexpr cl_uint kChunkTokens = 512;
constexpr cl_uint kChunkCount = kContextTokens / kChunkTokens;
constexpr cl_uint kColdChunkCount = kColdTokens / kChunkTokens;
constexpr cl_uint kHotChunkCount = kHotTokens / kChunkTokens;
constexpr cl_uint kUpdateLocal = 32;
constexpr cl_uint kPartialLocal = 128;
constexpr cl_uint kReduceLocal = 256;
constexpr int kWarmups = 5;
constexpr int kSamples = 20;
constexpr double kComponentCapMs = 0.5618915;
static_assert(kKeyQuantGroup == 32U || kKeyQuantGroup == 4U ||
              kKeyQuantGroup == 2U,
              "the admitted key quantization groups are 32, 4, and 2");
static_assert(kValueQuantGroup == 32U || kValueQuantGroup == 4U,
              "the admitted value quantization groups are 32 and 4");
static_assert(kHeadDim % kKeyQuantGroup == 0U);
static_assert(kHeadDim % kValueQuantGroup == 0U);
static_assert(kKeyQuantGroup != 2U || kValueQuantGroup == 4U,
              "the only admitted asymmetric codec is K2/V4");
static_assert(kHotTokens == 8192U || kHotTokens == 16384U,
              "the admitted hot windows are 8192 and 16384 tokens");
static_assert(kHotTokens % kChunkTokens == 0U);
static_assert(IQ36_COMPONENT_UPDATE_AFTER_ATTENTION == 0 ||
              IQ36_COMPONENT_UPDATE_AFTER_ATTENTION == 1);
static_assert(IQ36_COMPONENT_STORAGE_SPECIALIZED == 0 ||
              IQ36_COMPONENT_STORAGE_SPECIALIZED == 1);

constexpr bool kHybridK2V4 =
    kKeyQuantGroup == 2U && kValueQuantGroup == 4U;
constexpr bool kSplitStateOwnerHot16K =
    kHybridK2V4 && kHotTokens == 16384U &&
    IQ36_COMPONENT_UPDATE_AFTER_ATTENTION == 1;
constexpr bool kStorageSpecialized =
    kSplitStateOwnerHot16K && IQ36_COMPONENT_STORAGE_SPECIALIZED == 1;
static_assert(IQ36_COMPONENT_STORAGE_SPECIALIZED == 0 ||
              kSplitStateOwnerHot16K,
              "storage specialization is admitted only for split hot16k K2/V4");
constexpr const char* kAlgorithm = kStorageSpecialized
    ? "direct_i8_hybrid_k2_v4_hot16384_storage_specialized_dpas"
    : kSplitStateOwnerHot16K
    ? "direct_i8_hybrid_k2_v4_hot16384_split_state_owner_dpas"
    : kHybridK2V4
    ? "direct_i8_hybrid_k2_v4_full_cold_hot8192_f16_dpas"
    : kKeyQuantGroup == 32U
        ? "direct_i8_block32_hot8192_f16_dpas"
        : "direct_i8_group4_full_cold_hot8192_f16_dpas";
constexpr const char* kColdKLayout = kHybridK2V4
    ? "token16_dim4_packed_i8_group2_fp16_scale"
    : kKeyQuantGroup == 32U
        ? "token16_block32_packed_i8"
        : "token16_group4_packed_i8";
constexpr const char* kColdVLayout = kHybridK2V4
    ? "dimension_major_token16_i8_group4_fp16_scale"
    : "dimension_major_token16_i8";
constexpr const char* kKvDtype = kHybridK2V4
    ? "hot_f16_cold_int8_key_group2_value_group4_fp16_scale"
    : kKeyQuantGroup == 32U
        ? "hot_f16_cold_int8_block32_fp16_scale"
        : "hot_f16_cold_int8_group4_fp16_scale";

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

std::size_t DenseIndex(cl_uint token, cl_uint kv_head, cl_uint dim) {
  return (static_cast<std::size_t>(token) * kKvHeads + kv_head) * kHeadDim +
      dim;
}

float KeyValue(cl_uint token, cl_uint kv_head, cl_uint dim) {
  const cl_uint flat_dim = kv_head * kHeadDim + dim;
  return static_cast<float>(
      static_cast<int>((token * 13U + flat_dim * 7U + 5U) & 255U) - 128) /
      2048.0f;
}

float ValueValue(cl_uint token, cl_uint kv_head, cl_uint dim) {
  const cl_uint flat_dim = kv_head * kHeadDim + dim;
  return 0.02f + static_cast<float>(
      static_cast<int>((token * 3U + flat_dim * 19U + 9U) & 255U) - 128) /
      4096.0f;
}

struct Timing {
  double update_ms = 0.0;
  double partial_ms = 0.0;
  double reduce_ms = 0.0;
  double total_ms = 0.0;
};

Timing Run(cl_command_queue queue, cl_kernel update, cl_kernel partial,
           cl_kernel hot_partial, cl_kernel reduce) {
  const std::size_t update_global =
      kKvHeads * (kKeyScaleGroups + kValueScaleGroups) * kUpdateLocal;
  const std::size_t update_local = kUpdateLocal;
  const std::size_t partial_global =
      static_cast<std::size_t>(kKvHeads) * kChunkCount * kPartialLocal;
  const std::size_t cold_partial_global =
      static_cast<std::size_t>(kKvHeads) * kColdChunkCount * kPartialLocal;
  const std::size_t hot_partial_global =
      static_cast<std::size_t>(kKvHeads) * kHotChunkCount * kPartialLocal;
  const std::size_t partial_local = kPartialLocal;
  const std::size_t reduce_global =
      static_cast<std::size_t>(kQHeads) * kReduceLocal;
  const std::size_t reduce_local = kReduceLocal;
  cl_event update_event = nullptr;
  cl_event partial_event = nullptr;
  cl_event hot_partial_event = nullptr;
  cl_event reduce_event = nullptr;
#if IQ36_COMPONENT_UPDATE_AFTER_ATTENTION == 0
  Check(clEnqueueNDRangeKernel(queue, update, 1, nullptr, &update_global,
                               &update_local, 0, nullptr, &update_event),
        "clEnqueueNDRangeKernel(update)");
#endif
#if IQ36_COMPONENT_STORAGE_SPECIALIZED == 1
  Check(clEnqueueNDRangeKernel(
      queue, partial, 1, nullptr, &cold_partial_global,
      &partial_local, 0, nullptr, &partial_event),
      "clEnqueueNDRangeKernel(cold partial)");
  Check(clEnqueueNDRangeKernel(
      queue, hot_partial, 1, nullptr, &hot_partial_global,
      &partial_local, 0, nullptr, &hot_partial_event),
      "clEnqueueNDRangeKernel(hot partial)");
#else
  (void)cold_partial_global;
  (void)hot_partial_global;
  (void)hot_partial;
  Check(clEnqueueNDRangeKernel(queue, partial, 1, nullptr, &partial_global,
                               &partial_local, 0, nullptr, &partial_event),
        "clEnqueueNDRangeKernel(partial)");
#endif
  Check(clEnqueueNDRangeKernel(queue, reduce, 1, nullptr, &reduce_global,
                               &reduce_local, 0, nullptr, &reduce_event),
        "clEnqueueNDRangeKernel(reduce)");
#if IQ36_COMPONENT_UPDATE_AFTER_ATTENTION == 1
  Check(clEnqueueNDRangeKernel(queue, update, 1, nullptr, &update_global,
                               &update_local, 0, nullptr, &update_event),
        "clEnqueueNDRangeKernel(update)");
#endif
  Check(clFinish(queue), "clFinish(candidate)");
  Timing result;
  result.update_ms = EventMs(update_event);
#if IQ36_COMPONENT_STORAGE_SPECIALIZED == 1
  result.partial_ms = EventMs(partial_event) + EventMs(hot_partial_event);
#else
  result.partial_ms = EventMs(partial_event);
#endif
  result.reduce_ms = EventMs(reduce_event);
  result.total_ms = result.update_ms + result.partial_ms + result.reduce_ms;
  clReleaseEvent(update_event);
  clReleaseEvent(partial_event);
#if IQ36_COMPONENT_STORAGE_SPECIALIZED == 1
  clReleaseEvent(hot_partial_event);
#endif
  clReleaseEvent(reduce_event);
  return result;
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

double Median(const std::vector<Timing>& samples) {
  std::vector<double> values;
  values.reserve(samples.size());
  for (const auto& sample : samples) values.push_back(sample.total_ms);
  std::sort(values.begin(), values.end());
  return (values[values.size() / 2U - 1U] + values[values.size() / 2U]) / 2.0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 2) {
      throw std::invalid_argument(
          "usage: iq36-direct-i8-hotcold-gqa-decode OPENCL_SOURCE");
    }

    const std::size_t history_values =
        static_cast<std::size_t>(kContextTokens) * kKvHeads * kHeadDim;
    const std::size_t cold_values =
        static_cast<std::size_t>(kColdTokens) * kKvHeads * kHeadDim;
    const std::size_t hot_values =
        static_cast<std::size_t>(kHotTokens) * kKvHeads * kHeadDim;
    const std::size_t cold_key_scale_values =
        static_cast<std::size_t>(kColdTokens) * kKvHeads * kKeyScaleGroups;
    const std::size_t cold_value_scale_values =
        static_cast<std::size_t>(kColdTokens) * kKvHeads * kValueScaleGroups;
    const std::size_t output_values =
        static_cast<std::size_t>(kQHeads) * kHeadDim;
    const std::size_t group_count =
        static_cast<std::size_t>(kKvHeads) * kChunkCount;
    const std::size_t meta_values = group_count * kGqaGroup;
    const std::size_t partial_values = meta_values * kHeadDim;
    const std::size_t cold_k_words = cold_values / 4U;
    const std::size_t hot_k_words = hot_values / 2U;

    std::vector<float> dense_k(history_values);
    std::vector<float> dense_v(history_values);
    for (cl_uint token = 0; token < kContextTokens; ++token) {
      for (cl_uint kv_head = 0; kv_head < kKvHeads; ++kv_head) {
        for (cl_uint dim = 0; dim < kHeadDim; ++dim) {
          const std::size_t index = DenseIndex(token, kv_head, dim);
          dense_k[index] = KeyValue(token, kv_head, dim);
          dense_v[index] = ValueValue(token, kv_head, dim);
        }
      }
    }

    std::vector<std::uint16_t> query(output_values);
    for (std::size_t index = 0; index < output_values; ++index) {
      const float value = static_cast<float>(
          static_cast<int>((index * 29U + 17U) & 255U) - 128) / 1024.0f;
      query[index] = FloatToHalf(value);
    }

    std::vector<std::uint32_t> cold_k(cold_k_words, 0U);
    std::vector<std::int8_t> cold_v(cold_values);
    std::vector<std::uint16_t> cold_k_scales(cold_key_scale_values);
    std::vector<std::uint16_t> cold_v_scales(cold_value_scale_values);
    auto* cold_k_bytes = reinterpret_cast<std::int8_t*>(cold_k.data());
    const std::size_t cold_k_words_per_head =
        cold_values / 4U / kKvHeads;
    for (cl_uint token = 0; token < kColdTokens; ++token) {
      const cl_uint token_block = token / kTokenTile;
      const cl_uint token_lane = token & (kTokenTile - 1U);
      for (cl_uint kv_head = 0; kv_head < kKvHeads; ++kv_head) {
        for (cl_uint group = 0; group < kKeyScaleGroups; ++group) {
          float k_max = 0.0f;
          for (cl_uint lane = 0; lane < kKeyQuantGroup; ++lane) {
            const cl_uint dim = group * kKeyQuantGroup + lane;
            k_max = std::max(
                k_max, std::abs(dense_k[DenseIndex(token, kv_head, dim)]));
          }
          const float k_scale = k_max == 0.0f ? 1.0f : k_max / 127.0f;
          const std::size_t scale_index =
              (static_cast<std::size_t>(kv_head) * kKeyScaleGroups + group) *
                  kColdTokens + token;
          cold_k_scales[scale_index] = FloatToHalf(k_scale);
          for (cl_uint lane = 0; lane < kKeyQuantGroup; ++lane) {
            const cl_uint dim = group * kKeyQuantGroup + lane;
            const auto quantized = static_cast<std::int8_t>(std::clamp(
                static_cast<int>(std::nearbyint(
                    dense_k[DenseIndex(token, kv_head, dim)] / k_scale)),
                -127, 127));
            const std::size_t packed_index =
                static_cast<std::size_t>(kv_head) * cold_k_words_per_head +
                ((static_cast<std::size_t>(token_block) * kKeyPackWords +
                  dim / 4U) * kTokenTile + token_lane);
            cold_k_bytes[packed_index * 4U + dim % 4U] = quantized;
          }
        }
        for (cl_uint group = 0; group < kValueScaleGroups; ++group) {
          float v_max = 0.0f;
          for (cl_uint lane = 0; lane < kValueQuantGroup; ++lane) {
            const cl_uint dim = group * kValueQuantGroup + lane;
            v_max = std::max(
                v_max, std::abs(dense_v[DenseIndex(token, kv_head, dim)]));
          }
          const float v_scale = v_max == 0.0f ? 1.0f : v_max / 127.0f;
          const std::size_t scale_index =
              (static_cast<std::size_t>(kv_head) * kValueScaleGroups +
               group) * kColdTokens + token;
          cold_v_scales[scale_index] = FloatToHalf(v_scale);
          for (cl_uint lane = 0; lane < kValueQuantGroup; ++lane) {
            const cl_uint dim = group * kValueQuantGroup + lane;
            const auto quantized = static_cast<std::int8_t>(std::clamp(
                static_cast<int>(std::nearbyint(
                    dense_v[DenseIndex(token, kv_head, dim)] / v_scale)),
                -127, 127));
            cold_v[(static_cast<std::size_t>(kv_head) * kHeadDim + dim) *
                kColdTokens + token] = quantized;
          }
        }
      }
    }

    std::vector<std::uint32_t> hot_k(hot_k_words, 0U);
    std::vector<std::uint16_t> hot_v(hot_values);
    const std::size_t hot_k_words_per_head =
        static_cast<std::size_t>(kHotTokens) * kHeadDim / 2U;
    for (cl_uint hot_token = 0; hot_token < kHotTokens; ++hot_token) {
      const cl_uint token = kColdTokens + hot_token;
      const cl_uint token_block = hot_token / kTokenTile;
      const cl_uint token_lane = hot_token & (kTokenTile - 1U);
      for (cl_uint kv_head = 0; kv_head < kKvHeads; ++kv_head) {
        for (cl_uint pair = 0; pair < kHeadDim / 2U; ++pair) {
          const cl_uint dim = pair * 2U;
          const std::uint16_t low = FloatToHalf(
              dense_k[DenseIndex(token, kv_head, dim)]);
          const std::uint16_t high = FloatToHalf(
              dense_k[DenseIndex(token, kv_head, dim + 1U)]);
          const std::size_t packed_index =
              static_cast<std::size_t>(kv_head) * hot_k_words_per_head +
              ((static_cast<std::size_t>(token_block) * (kHeadDim / 2U) +
                pair) * kTokenTile + token_lane);
          hot_k[packed_index] = static_cast<std::uint32_t>(low) |
              (static_cast<std::uint32_t>(high) << 16U);
        }
        for (cl_uint dim = 0; dim < kHeadDim; ++dim) {
          hot_v[(static_cast<std::size_t>(kv_head) * kHeadDim + dim) *
              kHotTokens + hot_token] = FloatToHalf(
                  dense_v[DenseIndex(token, kv_head, dim)]);
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
        evicted_k[index] = FloatToHalf(
            dense_k[DenseIndex(kColdTokens - 1U, kv_head, dim)]);
        evicted_v[index] = FloatToHalf(
            dense_v[DenseIndex(kColdTokens - 1U, kv_head, dim)]);
        current_k[index] = FloatToHalf(
            dense_k[DenseIndex(kContextTokens - 1U, kv_head, dim)]);
        current_v[index] = FloatToHalf(
            dense_v[DenseIndex(kContextTokens - 1U, kv_head, dim)]);
      }
    }

    const auto device = SelectGpu();
    cl_int status = CL_SUCCESS;
    cl_context context = clCreateContext(
        nullptr, 1, &device, nullptr, nullptr, &status);
    Check(status, "clCreateContext");
    cl_command_queue queue = clCreateCommandQueue(
        context, device, CL_QUEUE_PROFILING_ENABLE, &status);
    Check(status, "clCreateCommandQueue");
    const std::string source = ReadText(argv[1]);
    const char* source_data = source.data();
    const std::size_t source_size = source.size();
    const std::string base_build_options =
        "-cl-std=CL3.0 -DIQ36_KEY_QUANT_GROUP=" +
        std::to_string(kKeyQuantGroup) +
        " -DIQ36_VALUE_QUANT_GROUP=" + std::to_string(kValueQuantGroup) +
        " -DIQ36_HOT_TOKENS=" + std::to_string(kHotTokens);
    auto BuildProgram = [&](const std::string& build_options) {
      cl_program result = clCreateProgramWithSource(
          context, 1, &source_data, &source_size, &status);
      Check(status, "clCreateProgramWithSource");
      status = clBuildProgram(
          result, 1, &device, build_options.c_str(), nullptr, nullptr);
      if (status != CL_SUCCESS) {
        std::size_t log_bytes = 0;
        clGetProgramBuildInfo(result, device, CL_PROGRAM_BUILD_LOG,
                              0, nullptr, &log_bytes);
        std::string log(log_bytes, '\0');
        clGetProgramBuildInfo(result, device, CL_PROGRAM_BUILD_LOG,
                              log_bytes, log.data(), nullptr);
        std::cerr << log << '\n';
        Check(status, "clBuildProgram");
      }
      return result;
    };
#if IQ36_COMPONENT_STORAGE_SPECIALIZED == 1
    cl_program program = BuildProgram(
        base_build_options + " -DIQ36_PARTIAL_STORAGE_CLASS=1");
    cl_program hot_program = BuildProgram(
        base_build_options + " -DIQ36_PARTIAL_STORAGE_CLASS=2");
#else
    cl_program program = BuildProgram(base_build_options);
    cl_program hot_program = nullptr;
#endif
    auto Kernel = [&](cl_program owner, const char* name) {
      cl_kernel kernel = clCreateKernel(owner, name, &status);
      Check(status, std::string("clCreateKernel(") + name + ")");
      return kernel;
    };
    cl_kernel update = Kernel(program, "iq36_direct_i8_update_state");
#if IQ36_COMPONENT_STORAGE_SPECIALIZED == 1
    cl_kernel partial = Kernel(program, "iq36_direct_i8_cold_partial");
    cl_kernel hot_partial = Kernel(
        hot_program, "iq36_direct_f16_hot_partial");
#else
    cl_kernel partial = Kernel(program, "iq36_direct_i8_hotcold_partial");
    cl_kernel hot_partial = nullptr;
#endif
    cl_kernel reduce = Kernel(program, "iq36_direct_i8_hotcold_reduce");
    cl_kernel reference_score = Kernel(
        program, "iq36_direct_i8_reference_score");
    cl_kernel reference_apply = Kernel(
        program, "iq36_direct_i8_reference_apply");

    cl_mem query_buffer = Buffer(
        context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        query.size() * sizeof(query[0]), query.data());
    cl_mem evicted_k_buffer = Buffer(
        context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        evicted_k.size() * sizeof(evicted_k[0]), evicted_k.data());
    cl_mem evicted_v_buffer = Buffer(
        context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        evicted_v.size() * sizeof(evicted_v[0]), evicted_v.data());
    cl_mem current_k_buffer = Buffer(
        context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        current_k.size() * sizeof(current_k[0]), current_k.data());
    cl_mem current_v_buffer = Buffer(
        context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        current_v.size() * sizeof(current_v[0]), current_v.data());
    cl_mem cold_k_buffer = Buffer(
        context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        cold_k.size() * sizeof(cold_k[0]), cold_k.data());
    cl_mem cold_v_buffer = Buffer(
        context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        cold_v.size() * sizeof(cold_v[0]), cold_v.data());
    cl_mem cold_k_scale_buffer = Buffer(
        context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        cold_k_scales.size() * sizeof(cold_k_scales[0]),
        cold_k_scales.data());
    cl_mem cold_v_scale_buffer = Buffer(
        context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        cold_v_scales.size() * sizeof(cold_v_scales[0]),
        cold_v_scales.data());
    cl_mem hot_k_buffer = Buffer(
        context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        hot_k.size() * sizeof(hot_k[0]), hot_k.data());
    cl_mem hot_v_buffer = Buffer(
        context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        hot_v.size() * sizeof(hot_v[0]), hot_v.data());
    cl_mem dense_k_buffer = Buffer(
        context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        dense_k.size() * sizeof(dense_k[0]), dense_k.data());
    cl_mem dense_v_buffer = Buffer(
        context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        dense_v.size() * sizeof(dense_v[0]), dense_v.data());
    cl_mem score_buffer = Buffer(
        context, CL_MEM_READ_WRITE,
        static_cast<std::size_t>(kQHeads) * kContextTokens * sizeof(float));
    cl_mem reference_output_buffer = Buffer(
        context, CL_MEM_READ_WRITE, output_values * sizeof(float));
    cl_mem candidate_output_buffer = Buffer(
        context, CL_MEM_READ_WRITE, output_values * sizeof(float));
    cl_mem partial_max_buffer = Buffer(
        context, CL_MEM_READ_WRITE, meta_values * sizeof(float));
    cl_mem partial_sum_buffer = Buffer(
        context, CL_MEM_READ_WRITE, meta_values * sizeof(float));
    cl_mem partial_output_buffer = Buffer(
        context, CL_MEM_READ_WRITE, partial_values * sizeof(float));

    Arg(reference_score, 0, query_buffer);
    Arg(reference_score, 1, dense_k_buffer);
    Arg(reference_score, 2, score_buffer);
    Arg(reference_apply, 0, score_buffer);
    Arg(reference_apply, 1, dense_v_buffer);
    Arg(reference_apply, 2, reference_output_buffer);
    const std::size_t reference_score_global =
        static_cast<std::size_t>(kQHeads) * kContextTokens;
    const std::size_t reference_apply_global = output_values;
    const std::size_t reference_local = 256U;
    Check(clEnqueueNDRangeKernel(
        queue, reference_score, 1, nullptr, &reference_score_global,
        &reference_local, 0, nullptr, nullptr),
        "clEnqueueNDRangeKernel(reference score)");
    Check(clEnqueueNDRangeKernel(
        queue, reference_apply, 1, nullptr, &reference_apply_global,
        &reference_local, 0, nullptr, nullptr),
        "clEnqueueNDRangeKernel(reference apply)");
    Check(clFinish(queue), "clFinish(reference)");

    Arg(update, 0, evicted_k_buffer);
    Arg(update, 1, evicted_v_buffer);
    Arg(update, 2, current_k_buffer);
    Arg(update, 3, current_v_buffer);
    Arg(update, 4, cold_k_buffer);
    Arg(update, 5, cold_v_buffer);
    Arg(update, 6, cold_k_scale_buffer);
    Arg(update, 7, cold_v_scale_buffer);
    Arg(update, 8, hot_k_buffer);
    Arg(update, 9, hot_v_buffer);
#if IQ36_COMPONENT_STORAGE_SPECIALIZED == 1
    Arg(partial, 0, query_buffer);
    Arg(partial, 1, cold_k_buffer);
    Arg(partial, 2, cold_v_buffer);
    Arg(partial, 3, cold_k_scale_buffer);
    Arg(partial, 4, cold_v_scale_buffer);
    Arg(partial, 5, partial_max_buffer);
    Arg(partial, 6, partial_sum_buffer);
    Arg(partial, 7, partial_output_buffer);
    Arg(hot_partial, 0, query_buffer);
    Arg(hot_partial, 1, hot_k_buffer);
    Arg(hot_partial, 2, hot_v_buffer);
    Arg(hot_partial, 3, partial_max_buffer);
    Arg(hot_partial, 4, partial_sum_buffer);
    Arg(hot_partial, 5, partial_output_buffer);
#else
    Arg(partial, 0, query_buffer);
    Arg(partial, 1, cold_k_buffer);
    Arg(partial, 2, cold_v_buffer);
    Arg(partial, 3, cold_k_scale_buffer);
    Arg(partial, 4, cold_v_scale_buffer);
    Arg(partial, 5, hot_k_buffer);
    Arg(partial, 6, hot_v_buffer);
    Arg(partial, 7, partial_max_buffer);
    Arg(partial, 8, partial_sum_buffer);
    Arg(partial, 9, partial_output_buffer);
#endif
    Arg(reduce, 0, partial_max_buffer);
    Arg(reduce, 1, partial_sum_buffer);
    Arg(reduce, 2, partial_output_buffer);
    Arg(reduce, 3, candidate_output_buffer);

    dense_k.clear();
    dense_v.clear();
    cold_k.clear();
    cold_v.clear();
    cold_k_scales.clear();
    cold_v_scales.clear();
    hot_k.clear();
    hot_v.clear();
    for (int warmup = 0; warmup < kWarmups; ++warmup) {
      (void)Run(queue, update, partial, hot_partial, reduce);
    }
    std::vector<Timing> samples;
    samples.reserve(kSamples);
    for (int sample = 0; sample < kSamples; ++sample) {
      samples.push_back(Run(queue, update, partial, hot_partial, reduce));
    }

    std::vector<float> reference(output_values);
    std::vector<float> candidate(output_values);
    Check(clEnqueueReadBuffer(
        queue, reference_output_buffer, CL_TRUE, 0,
        reference.size() * sizeof(float), reference.data(), 0, nullptr,
        nullptr),
        "clEnqueueReadBuffer(reference)");
    Check(clEnqueueReadBuffer(
        queue, candidate_output_buffer, CL_TRUE, 0,
        candidate.size() * sizeof(float), candidate.data(), 0, nullptr,
        nullptr),
        "clEnqueueReadBuffer(candidate)");
    const Numeric numeric = Compare(reference, candidate);
    const bool numeric_pass = numeric.finite && numeric.cosine >= 0.999 &&
        numeric.relative_l2 <= 0.002;
    const double median_ms = Median(samples);
    const bool point_estimate_rate_pass = median_ms <= kComponentCapMs;
    const std::size_t state_bytes =
        2U * cold_values * sizeof(std::int8_t) +
        cold_key_scale_values * sizeof(std::uint16_t) +
        cold_value_scale_values * sizeof(std::uint16_t) +
        2U * hot_values * sizeof(std::uint16_t);

    auto EmitTiming = [](const Timing& timing) {
      std::cout << "{\"update_ms\":" << timing.update_ms
                << ",\"partial_ms\":" << timing.partial_ms
                << ",\"reduce_ms\":" << timing.reduce_ms
                << ",\"total_ms\":" << timing.total_ms << "}";
    };
    std::cout << std::boolalpha << std::setprecision(12) << "{"
              << "\"algorithm\":\"" << kAlgorithm << "\","
              << "\"chunk_tokens\":" << kChunkTokens << ","
              << "\"cold_k_layout\":\"" << kColdKLayout << "\","
              << "\"cold_v_layout\":\"" << kColdVLayout << "\","
              << "\"component_cap_ms\":" << kComponentCapMs << ","
              << "\"context_tokens\":" << kContextTokens << ","
              << "\"device\":\"" << DeviceString(device, CL_DEVICE_NAME)
              << "\","
              << "\"finite\":" << numeric.finite << ","
              << "\"head_dim\":" << kHeadDim << ","
              << "\"hot_k_layout\":\"token16_dim2_packed_f16\","
              << "\"hot_tokens\":" << kHotTokens << ","
              << "\"hot_v_layout\":\"dimension_major_token16_f16\","
              << "\"cold_tokens\":" << kColdTokens << ","
              << "\"kv_dtype\":\"" << kKvDtype << "\","
              << "\"kv_head_count\":" << kKvHeads << ","
              << "\"max_abs\":" << numeric.max_abs << ","
              << "\"minimum_samples\":" << kSamples << ","
              << "\"numeric_pass\":" << numeric_pass << ","
              << "\"output_cosine\":" << numeric.cosine << ","
              << "\"output_relative_l2\":" << numeric.relative_l2 << ","
              << "\"output_rmse\":" << numeric.rmse << ","
              << "\"partial_bytes\":"
              << partial_values * sizeof(float) +
                     2U * meta_values * sizeof(float)
              << ","
              << "\"partial_dispatches\":"
              << (kStorageSpecialized ? 2 : 1) << ","
              << "\"point_estimate_ms\":" << median_ms << ","
              << "\"point_estimate_rate_pass\":"
              << point_estimate_rate_pass << ","
              << "\"q_head_count\":" << kQHeads << ","
              << "\"quant_group\":" << kKeyQuantGroup << ","
              << "\"key_quant_group\":" << kKeyQuantGroup << ","
              << "\"value_quant_group\":" << kValueQuantGroup << ","
              << "\"key_pack_dimensions\":4,"
              << "\"execution_order\":\""
              << (IQ36_COMPONENT_UPDATE_AFTER_ATTENTION == 1
                      ? "partial_then_reduce_then_update"
                      : "update_then_partial_then_reduce")
              << "\","
              << "\"required_checks_passed\":" << numeric_pass << ","
              << "\"samples\":[";
    for (std::size_t index = 0; index < samples.size(); ++index) {
      if (index != 0U) std::cout << ",";
      EmitTiming(samples[index]);
    }
    std::cout << "],"
              << "\"scale_dtype\":\"fp16\","
              << "\"state_bytes\":" << state_bytes << ","
              << "\"storage_specialized\":" << kStorageSpecialized << ","
              << "\"subgroup_size\":16,"
              << "\"timed_scope\":\"append_quantize_qk_softmax_pv_workspace_reduce\"}"
              << std::endl;

    for (cl_mem buffer : {
             query_buffer, evicted_k_buffer, evicted_v_buffer,
             current_k_buffer, current_v_buffer, cold_k_buffer, cold_v_buffer,
             cold_k_scale_buffer, cold_v_scale_buffer, hot_k_buffer,
             hot_v_buffer, dense_k_buffer, dense_v_buffer, score_buffer,
             reference_output_buffer, candidate_output_buffer,
             partial_max_buffer, partial_sum_buffer, partial_output_buffer}) {
      clReleaseMemObject(buffer);
    }
    for (cl_kernel kernel : {
             update, partial, hot_partial, reduce,
             reference_score, reference_apply}) {
      if (kernel == nullptr) continue;
      clReleaseKernel(kernel);
    }
    clReleaseProgram(program);
    if (hot_program != nullptr) clReleaseProgram(hot_program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return numeric_pass ? 0 : 2;
  } catch (const std::exception& exception) {
    std::cerr << "iq36-direct-i8-hotcold-gqa-decode: "
              << exception.what() << '\n';
    return 4;
  }
}
