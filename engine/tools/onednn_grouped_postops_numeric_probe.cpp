#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_ocl.hpp>

#include <CL/cl.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

#ifndef IQ36_PROBE_EXPERTS
#define IQ36_PROBE_EXPERTS 16
#endif
#ifndef IQ36_PROBE_ROWS
#define IQ36_PROBE_ROWS 64
#endif

constexpr std::size_t kExperts = IQ36_PROBE_EXPERTS;
constexpr std::size_t kRows = IQ36_PROBE_ROWS;
constexpr std::size_t kInput = 2048;
constexpr std::size_t kOutput = 512;
constexpr std::size_t kWeightGroup = 64;
constexpr std::size_t kWeightGroups = kInput / kWeightGroup;
constexpr std::size_t kValues = kRows * kOutput;

[[noreturn]] void Fail(const std::string& message) {
  throw std::runtime_error(message);
}

void Require(bool condition, const std::string& message) {
  if (!condition) Fail(message);
}

std::string JsonString(const std::string& value) {
  std::string result = "\"";
  for (char character : value) {
    switch (character) {
      case '\\':
        result += "\\\\";
        break;
      case '"':
        result += "\\\"";
        break;
      case '\n':
        result += "\\n";
        break;
      case '\r':
        result += "\\r";
        break;
      case '\t':
        result += "\\t";
        break;
      default:
        result += character;
        break;
    }
  }
  result += '"';
  return result;
}

float HalfToFloat(std::uint16_t value) {
  const std::uint32_t sign = (std::uint32_t(value) & 0x8000U) << 16;
  std::uint32_t exponent = (value >> 10) & 0x1fU;
  std::uint32_t mantissa = value & 0x03ffU;
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
      mantissa &= 0x03ffU;
      bits = sign | ((127U - 14U - shift) << 23) | (mantissa << 13);
    }
  } else if (exponent == 0x1fU) {
    bits = sign | 0x7f800000U | (mantissa << 13);
  } else {
    bits = sign | ((exponent + 112U) << 23) | (mantissa << 13);
  }
  float result = 0.0f;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

float FloatBitsToFloat(std::uint32_t bits) {
  float result = 0.0f;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

std::uint32_t OrderedFloatBits(std::uint32_t bits) {
  return (bits & 0x80000000U) != 0 ? ~bits : (bits ^ 0x80000000U);
}

template <typename Value>
void WriteMemory(const std::vector<Value>& values, dnnl::memory& memory,
                 int handle_index = 0) {
  Require(memory.get_desc().get_size(handle_index)
              == values.size() * sizeof(Value),
          "oneDNN memory size mismatch");
  Value* mapped = memory.map_data<Value>(handle_index);
  Require(mapped != nullptr, "oneDNN map returned null");
  std::copy(values.begin(), values.end(), mapped);
  memory.unmap_data(mapped, handle_index);
}

template <typename Value>
std::vector<Value> ReadMemory(dnnl::memory& memory, int handle_index = 0) {
  const std::size_t count
      = memory.get_desc().get_size(handle_index) / sizeof(Value);
  Value* mapped = memory.map_data<Value>(handle_index);
  Require(mapped != nullptr, "oneDNN map returned null");
  std::vector<Value> result(mapped, mapped + count);
  memory.unmap_data(mapped, handle_index);
  return result;
}

void WriteOffsets(dnnl::memory& memory,
                  const std::vector<std::int32_t>& offsets) {
  WriteMemory(offsets, memory, 1);
}

void CheckCl(cl_int status, const std::string& operation) {
  if (status != CL_SUCCESS) {
    Fail(operation + " failed with OpenCL status "
         + std::to_string(status));
  }
}

std::string ProgramBuildLog(cl_program program, cl_device_id device) {
  std::size_t bytes = 0;
  clGetProgramBuildInfo(
      program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &bytes);
  std::string result(bytes, '\0');
  if (bytes != 0) {
    clGetProgramBuildInfo(
        program, device, CL_PROGRAM_BUILD_LOG, bytes, result.data(), nullptr);
  }
  return result;
}

std::string Implementation(const dnnl::matmul& primitive) {
  const char* name = nullptr;
  const dnnl_status_t status = dnnl_primitive_desc_query(
      primitive.get_primitive_desc(), dnnl_query_impl_info_str, 0, &name);
  Require(status == dnnl_success && name != nullptr,
          "could not query oneDNN implementation name");
  return name;
}

std::vector<std::uint8_t> MakeWeights(std::uint32_t salt = 0) {
  const std::size_t logical = kExperts * kInput * kOutput;
  std::vector<std::uint8_t> packed((logical + 1) / 2);
  auto code = [](std::size_t index) {
    std::uint32_t value = static_cast<std::uint32_t>(index);
    value ^= value >> 15;
    value *= 0x2c1b3c6dU;
    value ^= value >> 12;
    value *= 0x297a2d39U;
    value ^= value >> 15;
    return static_cast<std::uint8_t>(value & 15U);
  };
  for (std::size_t byte = 0; byte < packed.size(); ++byte) {
    const std::uint8_t low = code((byte * 2) ^ salt);
    const std::uint8_t high = code((byte * 2 + 1) ^ salt);
    packed[byte] = static_cast<std::uint8_t>(low | (high << 4));
  }
  return packed;
}

std::vector<std::uint16_t> MakeSource() {
  constexpr std::array<std::uint16_t, 8> values = {
      0xbc00,  // -1
      0xba00,  // -0.75
      0xb800,  // -0.5
      0xb400,  // -0.25
      0x3400,  // 0.25
      0x3800,  // 0.5
      0x3a00,  // 0.75
      0x3c00,  // 1
  };
  std::vector<std::uint16_t> result(kRows * kInput);
  for (std::size_t row = 0; row < kRows; ++row) {
    for (std::size_t inner = 0; inner < kInput; ++inner) {
      result[row * kInput + inner]
          = values[(row * 17 + inner * 13 + (inner >> 5)) % values.size()];
    }
  }
  return result;
}

std::vector<std::uint16_t> MakeScales() {
  constexpr std::array<std::uint16_t, 4> values = {
      0x1c00,  // 1 / 256
      0x2000,  // 2 / 256
      0x2200,  // 3 / 256
      0x2400,  // 4 / 256
  };
  std::vector<std::uint16_t> result(
      kExperts * kWeightGroups * kOutput);
  for (std::size_t expert = 0; expert < kExperts; ++expert) {
    for (std::size_t group = 0; group < kWeightGroups; ++group) {
      for (std::size_t output = 0; output < kOutput; ++output) {
        const std::size_t index
            = (expert * kWeightGroups + group) * kOutput + output;
        result[index]
            = values[(expert * 3 + group * 5 + output * 7) % values.size()];
      }
    }
  }
  return result;
}

std::vector<std::uint16_t> MakeBinary() {
  constexpr std::array<std::uint16_t, 8> values = {
      0xc000,  // -2
      0xbc00,  // -1
      0xb800,  // -0.5
      0x3400,  // 0.25
      0x3800,  // 0.5
      0x3c00,  // 1
      0x3e00,  // 1.5
      0x4000,  // 2
  };
  std::vector<std::uint16_t> result(kValues);
  for (std::size_t row = 0; row < kRows; ++row) {
    for (std::size_t output = 0; output < kOutput; ++output) {
      result[row * kOutput + output]
          = values[(row * 29 + output * 11 + (output >> 4)) % values.size()];
    }
  }
  return result;
}

void SetPackedU4(std::vector<std::uint8_t>& values,
                 std::size_t index, std::uint8_t value) {
  Require(value < 16, "U4 value out of range");
  const std::size_t byte = index / 2;
  const std::uint8_t shift = static_cast<std::uint8_t>((index % 2) * 4);
  values[byte] = static_cast<std::uint8_t>(
      (values[byte] & ~(0x0fU << shift)) | (value << shift));
}

std::vector<std::uint8_t> MakeIdentityWeights() {
  const std::size_t logical = kExperts * kInput * kOutput;
  std::vector<std::uint8_t> result((logical + 1) / 2, 0x88);
  for (std::size_t expert = 0; expert < kExperts; ++expert) {
    for (std::size_t output = 0; output < kOutput; ++output) {
      const std::size_t physical
          = (expert * kOutput + output) * kInput + output;
      SetPackedU4(result, physical, 9);
    }
  }
  return result;
}

std::vector<std::uint16_t> MakeUnitScales() {
  return std::vector<std::uint16_t>(
      kExperts * kWeightGroups * kOutput, 0x3c00);
}

std::vector<std::uint16_t> FiniteHalfValues() {
  std::vector<std::uint16_t> result;
  result.reserve(63488);
  for (std::uint32_t bits = 0; bits <= 0xffffU; ++bits) {
    if ((bits & 0x7c00U) != 0x7c00U) {
      result.push_back(static_cast<std::uint16_t>(bits));
    }
  }
  Require(result.size() == 63488, "finite F16 census size mismatch");
  return result;
}

std::vector<std::uint16_t> MakeActivationCensusSource(
    const std::vector<std::uint16_t>& finite_values,
    std::size_t offset) {
  std::vector<std::uint16_t> result(kRows * kInput, 0);
  const std::size_t count
      = std::min(kValues, finite_values.size() - offset);
  for (std::size_t index = 0; index < count; ++index) {
    const std::size_t row = index / kOutput;
    const std::size_t output = index % kOutput;
    result[row * kInput + output] = finite_values[offset + index];
  }
  return result;
}

std::vector<std::int32_t> PrefixOffsets(
    const std::array<std::int32_t, kExperts>& counts) {
  std::vector<std::int32_t> offsets(kExperts);
  std::partial_sum(counts.begin(), counts.end(), offsets.begin());
  Require(offsets.back() == static_cast<std::int32_t>(kRows),
          "scenario row count mismatch");
  return offsets;
}

std::vector<std::pair<std::string, std::vector<std::int32_t>>> Scenarios() {
  Require(kExperts >= 16 && kRows % 64 == 0,
          "scenario geometry must preserve the locked 16-expert ratios");
  std::vector<std::pair<std::string, std::vector<std::int32_t>>> result;
  {
    std::array<std::int32_t, kExperts> counts{};
    Require(kRows % kExperts == 0,
            "dense scenario rows must divide across experts");
    counts.fill(static_cast<std::int32_t>(kRows / kExperts));
    result.emplace_back("dense_four_rows_per_expert", PrefixOffsets(counts));
  }
  {
    std::array<std::int32_t, kExperts> counts{};
    const std::array<std::size_t, 4> active = {
        0, 1, kExperts - 2, kExperts - 1};
    for (std::size_t expert : active) {
      counts[expert] = static_cast<std::int32_t>(kRows / 4);
    }
    result.emplace_back("sparse_four_zero_gap", PrefixOffsets(counts));
  }
  {
    std::array<std::int32_t, kExperts> counts{};
    const std::array<std::size_t, 4> active = {
        2, kExperts * 5 / 16, kExperts * 9 / 16, kExperts - 1};
    const std::int32_t scale = static_cast<std::int32_t>(kRows / 64);
    const std::array<std::int32_t, 4> sizes = {
        scale, 7 * scale, 17 * scale, 39 * scale};
    for (std::size_t index = 0; index < active.size(); ++index) {
      counts[active[index]] = sizes[index];
    }
    result.emplace_back("sparse_skewed_offsets", PrefixOffsets(counts));
  }
  return result;
}

enum class PostOpsKind { kNone, kBinary, kSwish, kSwishBinary };

dnnl::primitive_attr MakeAttributes(
    PostOpsKind kind, const dnnl::memory::desc& binary_desc) {
  dnnl::primitive_attr attributes;
  attributes.set_fpmath_mode(dnnl::fpmath_mode::f16, true);
  attributes.set_scales(
      DNNL_ARG_WEIGHTS, 7, {static_cast<std::int64_t>(kWeightGroup), 1},
      dnnl::memory::data_type::f16);
  attributes.set_zero_points(
      DNNL_ARG_WEIGHTS, 7, {static_cast<std::int64_t>(kWeightGroup), 1},
      dnnl::memory::data_type::u4);
  dnnl::post_ops post_ops;
  if (kind == PostOpsKind::kSwish
      || kind == PostOpsKind::kSwishBinary) {
    post_ops.append_eltwise(dnnl::algorithm::eltwise_swish, 1.0f, 0.0f);
  }
  if (kind == PostOpsKind::kBinary
      || kind == PostOpsKind::kSwishBinary) {
    post_ops.append_binary(dnnl::algorithm::binary_mul, binary_desc);
  }
  if (kind != PostOpsKind::kNone) attributes.set_post_ops(post_ops);
  return attributes;
}

struct PrimitiveCase {
  dnnl::memory output;
  dnnl::matmul primitive;
  std::unordered_map<int, dnnl::memory> arguments;
  std::string implementation;
  int binary_index;

  PrimitiveCase(
      const dnnl::engine& engine, const dnnl::memory::desc& source_desc,
      const dnnl::memory::desc& weights_desc,
      const dnnl::memory::desc& output_desc,
      const dnnl::memory::desc& binary_desc, PostOpsKind kind,
      dnnl::memory& source, dnnl::memory& weights, dnnl::memory& scales,
      dnnl::memory& zero_points, dnnl::memory& binary,
      dnnl::memory& max_group_hint)
      : output(output_desc, engine),
        primitive(dnnl::matmul::primitive_desc(
            engine, source_desc, weights_desc, output_desc,
            MakeAttributes(kind, binary_desc))),
        arguments{
            {DNNL_ARG_SRC, source},
            {DNNL_ARG_WEIGHTS, weights},
            {DNNL_ARG_DST, output},
            {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, scales},
            {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS, zero_points},
            {DNNL_ARG_HINT_MAX_GROUP_SIZE, max_group_hint},
        },
        implementation(Implementation(primitive)),
        binary_index(kind == PostOpsKind::kBinary
                         ? 0
                         : (kind == PostOpsKind::kSwishBinary ? 1 : -1)) {
    if (binary_index >= 0) {
      arguments.emplace(
          DNNL_ARG_ATTR_MULTIPLE_POST_OP(binary_index) | DNNL_ARG_SRC_1,
          binary);
    }
  }
};

struct Comparison {
  std::size_t mismatch_count = 0;
  std::size_t value_count = 0;
  double max_abs = 0.0;
  double mean_abs = 0.0;
  std::size_t first_index = std::numeric_limits<std::size_t>::max();
  std::uint16_t first_observed = 0;
  std::uint16_t first_expected = 0;
  std::vector<std::size_t> mismatch_indices;
  std::vector<std::uint16_t> mismatch_observed;
  std::vector<std::uint16_t> mismatch_expected;
};

Comparison Compare(const std::vector<std::uint16_t>& observed,
                   const std::vector<std::uint16_t>& expected) {
  Require(observed.size() == expected.size(), "comparison size mismatch");
  Comparison result;
  result.value_count = observed.size();
  double sum_abs = 0.0;
  for (std::size_t index = 0; index < observed.size(); ++index) {
    const double difference = std::abs(
        static_cast<double>(HalfToFloat(observed[index]))
        - static_cast<double>(HalfToFloat(expected[index])));
    sum_abs += difference;
    result.max_abs = std::max(result.max_abs, difference);
    if (observed[index] != expected[index]) {
      if (result.mismatch_count == 0) {
        result.first_index = index;
        result.first_observed = observed[index];
        result.first_expected = expected[index];
      }
      if (result.mismatch_indices.size() < 64) {
        result.mismatch_indices.push_back(index);
        result.mismatch_observed.push_back(observed[index]);
        result.mismatch_expected.push_back(expected[index]);
      }
      ++result.mismatch_count;
    }
  }
  result.mean_abs = sum_abs / static_cast<double>(observed.size());
  return result;
}

void PrintComparison(const Comparison& comparison) {
  std::cout << "{\"mismatch_count\":" << comparison.mismatch_count
            << ",\"value_count\":" << comparison.value_count
            << ",\"max_abs\":" << std::setprecision(17)
            << comparison.max_abs << ",\"mean_abs\":"
            << comparison.mean_abs;
  if (comparison.first_index != std::numeric_limits<std::size_t>::max()) {
    std::cout << ",\"first\":{\"index\":" << comparison.first_index
              << ",\"row\":" << comparison.first_index / kOutput
              << ",\"column\":" << comparison.first_index % kOutput
              << ",\"observed_bits\":" << comparison.first_observed
              << ",\"expected_bits\":" << comparison.first_expected
              << ",\"observed\":" << HalfToFloat(comparison.first_observed)
              << ",\"expected\":" << HalfToFloat(comparison.first_expected)
              << "}";
  }
  if (!comparison.mismatch_indices.empty()) {
    std::cout << ",\"samples\":[";
    for (std::size_t sample = 0;
         sample < comparison.mismatch_indices.size(); ++sample) {
      if (sample != 0) std::cout << ",";
      const std::size_t index = comparison.mismatch_indices[sample];
      std::cout << "{\"index\":" << index
                << ",\"row\":" << index / kOutput
                << ",\"column\":" << index % kOutput
                << ",\"observed_bits\":"
                << comparison.mismatch_observed[sample]
                << ",\"expected_bits\":"
                << comparison.mismatch_expected[sample]
                << ",\"observed\":"
                << HalfToFloat(comparison.mismatch_observed[sample])
                << ",\"expected\":"
                << HalfToFloat(comparison.mismatch_expected[sample])
                << "}";
    }
    std::cout << "]";
  }
  std::cout << "}";
}

struct FloatBitsComparison {
  std::size_t mismatch_count = 0;
  std::size_t value_count = 0;
  std::uint64_t max_ulp = 0;
  std::size_t observed_below = 0;
  std::size_t observed_above = 0;
  std::vector<std::size_t> mismatch_indices;
  std::vector<std::uint32_t> mismatch_observed;
  std::vector<std::uint32_t> mismatch_expected;
};

FloatBitsComparison CompareFloatBits(
    const std::vector<std::uint32_t>& observed,
    const std::vector<std::uint32_t>& expected,
    std::size_t count = std::numeric_limits<std::size_t>::max()) {
  Require(observed.size() == expected.size(),
          "float comparison size mismatch");
  count = std::min(count, observed.size());
  FloatBitsComparison result;
  result.value_count = count;
  for (std::size_t index = 0; index < count; ++index) {
    if (observed[index] == expected[index]) continue;
    const std::uint32_t observed_ordered
        = OrderedFloatBits(observed[index]);
    const std::uint32_t expected_ordered
        = OrderedFloatBits(expected[index]);
    const std::uint64_t ulp
        = observed_ordered >= expected_ordered
        ? static_cast<std::uint64_t>(observed_ordered - expected_ordered)
        : static_cast<std::uint64_t>(expected_ordered - observed_ordered);
    result.max_ulp = std::max(result.max_ulp, ulp);
    result.observed_below += observed_ordered < expected_ordered ? 1 : 0;
    result.observed_above += observed_ordered > expected_ordered ? 1 : 0;
    if (result.mismatch_indices.size() < 64) {
      result.mismatch_indices.push_back(index);
      result.mismatch_observed.push_back(observed[index]);
      result.mismatch_expected.push_back(expected[index]);
    }
    ++result.mismatch_count;
  }
  return result;
}

void PrintFloatBitsComparison(const FloatBitsComparison& comparison) {
  std::cout << "{\"mismatch_count\":" << comparison.mismatch_count
            << ",\"value_count\":" << comparison.value_count
            << ",\"max_ulp\":" << comparison.max_ulp
            << ",\"observed_below\":" << comparison.observed_below
            << ",\"observed_above\":" << comparison.observed_above;
  if (!comparison.mismatch_indices.empty()) {
    std::cout << ",\"samples\":[";
    for (std::size_t sample = 0;
         sample < comparison.mismatch_indices.size(); ++sample) {
      if (sample != 0) std::cout << ",";
      const std::size_t index = comparison.mismatch_indices[sample];
      std::cout << "{\"index\":" << index
                << ",\"row\":" << index / kOutput
                << ",\"column\":" << index % kOutput
                << ",\"observed_bits\":"
                << comparison.mismatch_observed[sample]
                << ",\"expected_bits\":"
                << comparison.mismatch_expected[sample]
                << ",\"observed\":"
                << FloatBitsToFloat(comparison.mismatch_observed[sample])
                << ",\"expected\":"
                << FloatBitsToFloat(comparison.mismatch_expected[sample])
                << "}";
    }
    std::cout << "]";
  }
  std::cout << "}";
}

class StockOracle {
 public:
  StockOracle(const dnnl::engine& engine, const dnnl::stream& stream,
              const char* build_options)
      : queue_(dnnl::ocl_interop::get_command_queue(stream)) {
    const cl_context context = dnnl::ocl_interop::get_context(engine);
    const cl_device_id device = dnnl::ocl_interop::get_device(engine);
    const char* source = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
inline float iq36_exp_polynomial(float x) {
  if (x > 88.0f) return INFINITY;
  if (x < -104.0f) return 0.0f;
  const int exponent = convert_int_rte(x * 1.4426950408889634f);
  const float remainder
      = x - convert_float(exponent) * 0.6931471805599453f;
  const float polynomial
      = 1.0f
      + remainder
          * (1.0f
              + remainder
                  * (0.5f
                      + remainder
                          * (0.16666666666666666f
                              + remainder
                                  * (0.041666666666666664f
                                      + remainder
                                          * (0.008333333333333333f
                                              + remainder
                                                  * (0.001388888888888889f
                                                      + remainder
                                                          * (0.0001984126984126984f
                                                              + remainder
                                                                  * 0.0000248015873015873f)))))));
  return ldexp(polynomial, exponent);
}
__attribute__((intel_reqd_sub_group_size(32)))
__kernel void iq36_stock_swiglu(
    __global const half* gate, __global const half* binary,
    __global half* output, uint row_width, int mode) {
  const uint row = get_global_id(1);
  const uint column = get_global_id(0);
  const uint subgroup_lane = get_sub_group_local_id();
  const uint block_offset = row * row_width + column - subgroup_lane;
  const float gate_value = as_half(intel_sub_group_block_read_us(
      (const __global ushort*)(gate + block_offset)));
  const float binary_value = as_half(intel_sub_group_block_read_us(
      (const __global ushort*)(binary + block_offset)));
  float value = gate_value;
  if (mode == 1 || mode == 2)
    value /= (1.0f + native_exp(-1.0f * value));
  if (mode == 4 || mode == 5)
    value = gate_value / (1.0f + exp(-1.0f * gate_value));
  if (mode == 6 || mode == 7)
    value = gate_value
        / (1.0f + iq36_exp_polynomial(-1.0f * gate_value));
  if (mode == 0 || mode == 2 || mode == 5 || mode == 7)
    value *= binary_value;
  const half result = value;
  intel_sub_group_block_write_us(
      (__global ushort*)(output + block_offset), as_ushort(result));
}
__attribute__((intel_reqd_sub_group_size(32)))
__kernel void iq36_stock_swish_f32(
    __global const half* gate, __global float* output, uint row_width) {
  const uint row = get_global_id(1);
  const uint column = get_global_id(0);
  const uint subgroup_lane = get_sub_group_local_id();
  const uint block_offset = row * row_width + column - subgroup_lane;
  float value = as_half(intel_sub_group_block_read_us(
      (const __global ushort*)(gate + block_offset)));
  value /= (1.0f + native_exp(-1.0f * value));
  intel_sub_group_block_write(
      (__global uint*)(output + block_offset), as_uint(value));
}
)CLC";
    const std::size_t source_length = std::strlen(source);
    cl_int status = CL_SUCCESS;
    program_ = clCreateProgramWithSource(
        context, 1, &source, &source_length, &status);
    CheckCl(status, "clCreateProgramWithSource");
    status = clBuildProgram(
        program_, 1, &device, build_options, nullptr, nullptr);
    if (status != CL_SUCCESS) {
      Fail("oracle OpenCL build failed: " + ProgramBuildLog(program_, device));
    }
    kernel_ = clCreateKernel(program_, "iq36_stock_swiglu", &status);
    CheckCl(status, "clCreateKernel");
    f32_kernel_ = clCreateKernel(
        program_, "iq36_stock_swish_f32", &status);
    CheckCl(status, "clCreateKernel f32");
  }

  StockOracle(const StockOracle&) = delete;
  StockOracle& operator=(const StockOracle&) = delete;

  ~StockOracle() {
    if (f32_kernel_ != nullptr) clReleaseKernel(f32_kernel_);
    if (kernel_ != nullptr) clReleaseKernel(kernel_);
    if (program_ != nullptr) clReleaseProgram(program_);
  }

  void Run(dnnl::memory& gate, dnnl::memory& binary,
           dnnl::memory& output, int mode) {
    cl_mem gate_memory = dnnl::ocl_interop::get_mem_object(gate);
    cl_mem binary_memory = dnnl::ocl_interop::get_mem_object(binary);
    cl_mem output_memory = dnnl::ocl_interop::get_mem_object(output);
    const cl_uint row_width = static_cast<cl_uint>(kOutput);
    const cl_int oracle_mode = mode;
    CheckCl(clSetKernelArg(
                kernel_, 0, sizeof(gate_memory), &gate_memory),
            "clSetKernelArg gate");
    CheckCl(clSetKernelArg(
                kernel_, 1, sizeof(binary_memory), &binary_memory),
            "clSetKernelArg binary");
    CheckCl(clSetKernelArg(
                kernel_, 2, sizeof(output_memory), &output_memory),
            "clSetKernelArg output");
    CheckCl(clSetKernelArg(
                kernel_, 3, sizeof(row_width), &row_width),
            "clSetKernelArg row_width");
    CheckCl(clSetKernelArg(
                kernel_, 4, sizeof(oracle_mode), &oracle_mode),
            "clSetKernelArg mode");
    constexpr std::array<std::size_t, 2> local = {32, 1};
    constexpr std::array<std::size_t, 2> global = {kOutput, kRows};
    CheckCl(clEnqueueNDRangeKernel(
                queue_, kernel_, 2, nullptr, global.data(), local.data(),
                0, nullptr, nullptr),
            "clEnqueueNDRangeKernel oracle");
    CheckCl(clFinish(queue_), "clFinish oracle");
  }

  void RunF32(dnnl::memory& gate, dnnl::memory& output) {
    cl_mem gate_memory = dnnl::ocl_interop::get_mem_object(gate);
    cl_mem output_memory = dnnl::ocl_interop::get_mem_object(output);
    const cl_uint row_width = static_cast<cl_uint>(kOutput);
    CheckCl(clSetKernelArg(
                f32_kernel_, 0, sizeof(gate_memory), &gate_memory),
            "clSetKernelArg f32 gate");
    CheckCl(clSetKernelArg(
                f32_kernel_, 1, sizeof(output_memory), &output_memory),
            "clSetKernelArg f32 output");
    CheckCl(clSetKernelArg(
                f32_kernel_, 2, sizeof(row_width), &row_width),
            "clSetKernelArg f32 row_width");
    constexpr std::array<std::size_t, 2> local = {32, 1};
    constexpr std::array<std::size_t, 2> global = {kOutput, kRows};
    CheckCl(clEnqueueNDRangeKernel(
                queue_, f32_kernel_, 2, nullptr, global.data(), local.data(),
                0, nullptr, nullptr),
            "clEnqueueNDRangeKernel f32 oracle");
    CheckCl(clFinish(queue_), "clFinish f32 oracle");
  }

 private:
  cl_command_queue queue_ = nullptr;
  cl_program program_ = nullptr;
  cl_kernel kernel_ = nullptr;
  cl_kernel f32_kernel_ = nullptr;
};

}  // namespace

int main() {
  try {
    using data_type = dnnl::memory::data_type;
    using format_tag = dnnl::memory::format_tag;
    const dnnl::engine engine(dnnl::engine::kind::gpu, 0);
    dnnl::stream stream(engine);

    const auto source_desc = dnnl::memory::desc::grouped(
        {static_cast<int>(kRows), static_cast<int>(kInput)},
        data_type::f16, 0, kExperts, data_type::s32);
    const auto weights_desc = dnnl::memory::desc(
        {static_cast<int>(kExperts), static_cast<int>(kInput),
         static_cast<int>(kOutput)},
        data_type::u4, format_tag::acb);
    const auto output_desc = dnnl::memory::desc::grouped(
        {static_cast<int>(kRows), static_cast<int>(kOutput)},
        data_type::f16, 0, kExperts, data_type::s32);
    const auto output_f32_desc = dnnl::memory::desc::grouped(
        {static_cast<int>(kRows), static_cast<int>(kOutput)},
        data_type::f32, 0, kExperts, data_type::s32);
    const auto scale_desc = dnnl::memory::desc(
        {static_cast<int>(kExperts), static_cast<int>(kWeightGroups),
         static_cast<int>(kOutput)},
        data_type::f16, format_tag::abc);
    const auto zero_point_desc = dnnl::memory::desc(
        {static_cast<int>(kExperts), static_cast<int>(kWeightGroups),
         static_cast<int>(kOutput)},
        data_type::u4, format_tag::abc);

    dnnl::memory source(source_desc, engine);
    dnnl::memory weights(weights_desc, engine);
    dnnl::memory up_weights(weights_desc, engine);
    dnnl::memory scales(scale_desc, engine);
    dnnl::memory zero_points(zero_point_desc, engine);
    dnnl::memory binary(output_desc, engine);
    dnnl::memory max_group_hint(
        dnnl::memory::desc::host_scalar(data_type::s32),
        static_cast<std::int32_t>(kRows));
    WriteMemory(MakeSource(), source);
    WriteMemory(MakeWeights(), weights);
    WriteMemory(MakeWeights(0x9e3779b9U), up_weights);
    WriteMemory(MakeScales(), scales);
    WriteMemory(
        std::vector<std::uint8_t>(
            zero_point_desc.get_size(), static_cast<std::uint8_t>(0x88)),
        zero_points);
    WriteMemory(MakeBinary(), binary);

    PrimitiveCase base(
        engine, source_desc, weights_desc, output_desc, output_desc,
        PostOpsKind::kNone, source, weights, scales, zero_points, binary,
        max_group_hint);
    PrimitiveCase binary_case(
        engine, source_desc, weights_desc, output_desc, output_desc,
        PostOpsKind::kBinary, source, weights, scales, zero_points, binary,
        max_group_hint);
    PrimitiveCase swish_case(
        engine, source_desc, weights_desc, output_desc, output_desc,
        PostOpsKind::kSwish, source, weights, scales, zero_points, binary,
        max_group_hint);
    PrimitiveCase swish_f32_case(
        engine, source_desc, weights_desc, output_f32_desc, output_desc,
        PostOpsKind::kSwish, source, weights, scales, zero_points, binary,
        max_group_hint);
    PrimitiveCase full_case(
        engine, source_desc, weights_desc, output_desc, output_desc,
        PostOpsKind::kSwishBinary, source, weights, scales, zero_points,
        binary, max_group_hint);
    PrimitiveCase producer_up(
        engine, source_desc, weights_desc, output_desc, output_desc,
        PostOpsKind::kNone, source, up_weights, scales, zero_points, binary,
        max_group_hint);
    PrimitiveCase producer_full(
        engine, source_desc, weights_desc, output_desc, output_desc,
        PostOpsKind::kSwishBinary, source, weights, scales, zero_points,
        producer_up.output, max_group_hint);
    dnnl::memory producer_snapshot(output_desc, engine);
    PrimitiveCase snapshot_full(
        engine, source_desc, weights_desc, output_desc, output_desc,
        PostOpsKind::kSwishBinary, source, weights, scales, zero_points,
        producer_snapshot, max_group_hint);
    dnnl::memory oracle_output(output_desc, engine);
    dnnl::memory grf256_oracle_output(output_desc, engine);
    dnnl::memory oracle_f32_output(output_f32_desc, engine);
    dnnl::memory correctly_rounded_f32_output(output_f32_desc, engine);
    StockOracle oracle(
        engine, stream, "-cl-std=CL3.0 -cl-mad-enable");
    StockOracle grf256_oracle(
        engine, stream,
        "-cl-std=CL3.0 -cl-mad-enable "
        "-cl-intel-256-GRF-per-thread");
    StockOracle correctly_rounded_oracle(
        engine, stream,
        "-cl-std=CL3.0 -cl-fp32-correctly-rounded-divide-sqrt "
        "-cl-intel-256-GRF-per-thread");

    std::cout << "{\"schema_version\":"
              << JsonString("iq36-onednn-grouped-postops-numeric-probe-v0")
              << ",\"shape\":{\"experts\":" << kExperts
              << ",\"rows\":" << kRows << ",\"input\":" << kInput
              << ",\"output\":" << kOutput
              << ",\"weight_group\":" << kWeightGroup
              << ",\"src_dt\":\"f16\",\"weights_dt\":\"u4\","
                 "\"dst_dt\":\"f16\",\"weight_zero_points\":true}"
              << ",\"providers\":{\"base\":"
              << JsonString(base.implementation)
              << ",\"binary\":" << JsonString(binary_case.implementation)
              << ",\"swish\":" << JsonString(swish_case.implementation)
              << ",\"swish_f32\":"
              << JsonString(swish_f32_case.implementation)
              << ",\"swish_binary\":"
              << JsonString(full_case.implementation)
              << ",\"producer_up\":"
              << JsonString(producer_up.implementation)
              << ",\"producer_swish_binary\":"
              << JsonString(producer_full.implementation)
              << ",\"snapshot_swish_binary\":"
              << JsonString(snapshot_full.implementation)
              << "},\"oracle\":"
              << JsonString(
                     "openvino_grouped_prefill_swiglu_subgroup32_block_io")
              << ",\"scenarios\":[";

    bool first_scenario = true;
    for (const auto& [name, offsets] : Scenarios()) {
      const std::int32_t max_group = [&]() {
        std::int32_t maximum = offsets.front();
        for (std::size_t index = 1; index < offsets.size(); ++index) {
          maximum = std::max(maximum, offsets[index] - offsets[index - 1]);
        }
        return maximum;
      }();
      const std::size_t active_groups = [&]() {
        std::size_t count = offsets.front() > 0 ? 1 : 0;
        for (std::size_t index = 1; index < offsets.size(); ++index) {
          count += offsets[index] > offsets[index - 1] ? 1 : 0;
        }
        return count;
      }();
      WriteOffsets(source, offsets);
      WriteOffsets(binary, offsets);
      WriteOffsets(base.output, offsets);
      WriteOffsets(binary_case.output, offsets);
      WriteOffsets(swish_case.output, offsets);
      WriteOffsets(swish_f32_case.output, offsets);
      WriteOffsets(full_case.output, offsets);
      WriteOffsets(producer_up.output, offsets);
      WriteOffsets(producer_full.output, offsets);
      WriteOffsets(producer_snapshot, offsets);
      WriteOffsets(snapshot_full.output, offsets);
      WriteOffsets(oracle_output, offsets);
      WriteOffsets(grf256_oracle_output, offsets);
      WriteOffsets(oracle_f32_output, offsets);
      WriteOffsets(correctly_rounded_f32_output, offsets);

      base.primitive.execute(stream, base.arguments);
      binary_case.primitive.execute(stream, binary_case.arguments);
      swish_case.primitive.execute(stream, swish_case.arguments);
      swish_f32_case.primitive.execute(stream, swish_f32_case.arguments);
      full_case.primitive.execute(stream, full_case.arguments);
      producer_up.primitive.execute(stream, producer_up.arguments);
      producer_full.primitive.execute(stream, producer_full.arguments);
      stream.wait();
      const auto first_full = ReadMemory<std::uint16_t>(full_case.output);
      const auto observed_swish_f32
          = ReadMemory<std::uint32_t>(swish_f32_case.output);
      const auto first_producer_full
          = ReadMemory<std::uint16_t>(producer_full.output);
      const auto producer_values
          = ReadMemory<std::uint16_t>(producer_up.output);
      WriteMemory(producer_values, producer_snapshot);
      WriteOffsets(producer_snapshot, offsets);
      snapshot_full.primitive.execute(stream, snapshot_full.arguments);
      stream.wait();
      const auto snapshot_values
          = ReadMemory<std::uint16_t>(producer_snapshot);
      const auto first_snapshot_full
          = ReadMemory<std::uint16_t>(snapshot_full.output);
      full_case.primitive.execute(stream, full_case.arguments);
      producer_full.primitive.execute(stream, producer_full.arguments);
      stream.wait();
      const auto second_full = ReadMemory<std::uint16_t>(full_case.output);
      const auto second_producer_full
          = ReadMemory<std::uint16_t>(producer_full.output);

      oracle.Run(base.output, binary, oracle_output, 0);
      const auto oracle_binary = ReadMemory<std::uint16_t>(oracle_output);
      oracle.Run(base.output, binary, oracle_output, 1);
      const auto oracle_swish = ReadMemory<std::uint16_t>(oracle_output);
      oracle.Run(base.output, binary, oracle_output, 2);
      const auto oracle_full = ReadMemory<std::uint16_t>(oracle_output);
      oracle.Run(base.output, binary, oracle_output, 3);
      const auto oracle_identity = ReadMemory<std::uint16_t>(oracle_output);
      oracle.Run(base.output, binary, oracle_output, 4);
      const auto oracle_exp_swish
          = ReadMemory<std::uint16_t>(oracle_output);
      oracle.Run(base.output, binary, oracle_output, 5);
      const auto oracle_exp_full
          = ReadMemory<std::uint16_t>(oracle_output);
      oracle.Run(base.output, binary, oracle_output, 6);
      const auto oracle_polynomial_swish
          = ReadMemory<std::uint16_t>(oracle_output);
      oracle.Run(base.output, binary, oracle_output, 7);
      const auto oracle_polynomial_full
          = ReadMemory<std::uint16_t>(oracle_output);
      grf256_oracle.Run(
          base.output, binary, grf256_oracle_output, 1);
      const auto grf256_oracle_swish
          = ReadMemory<std::uint16_t>(grf256_oracle_output);
      grf256_oracle.Run(
          base.output, binary, grf256_oracle_output, 2);
      const auto grf256_oracle_full
          = ReadMemory<std::uint16_t>(grf256_oracle_output);
      oracle.Run(
          base.output, producer_up.output, oracle_output, 2);
      const auto oracle_producer_full
          = ReadMemory<std::uint16_t>(oracle_output);
      grf256_oracle.Run(
          base.output, producer_up.output, grf256_oracle_output, 2);
      const auto grf256_oracle_producer_full
          = ReadMemory<std::uint16_t>(grf256_oracle_output);
      oracle.RunF32(base.output, oracle_f32_output);
      const auto oracle_swish_f32
          = ReadMemory<std::uint32_t>(oracle_f32_output);
      correctly_rounded_oracle.RunF32(
          base.output, correctly_rounded_f32_output);
      const auto correctly_rounded_swish_f32
          = ReadMemory<std::uint32_t>(correctly_rounded_f32_output);

      const auto stored_gate = ReadMemory<std::uint16_t>(base.output);
      const auto observed_binary
          = ReadMemory<std::uint16_t>(binary_case.output);
      const auto observed_swish
          = ReadMemory<std::uint16_t>(swish_case.output);
      const Comparison binary_comparison
          = Compare(observed_binary, oracle_binary);
      const Comparison swish_comparison
          = Compare(observed_swish, oracle_swish);
      const FloatBitsComparison swish_f32_comparison
          = CompareFloatBits(observed_swish_f32, oracle_swish_f32);
      const FloatBitsComparison swish_f32_vs_correctly_rounded
          = CompareFloatBits(
              observed_swish_f32, correctly_rounded_swish_f32);
      const FloatBitsComparison stock_f32_vs_correctly_rounded
          = CompareFloatBits(
              oracle_swish_f32, correctly_rounded_swish_f32);
      const Comparison full_comparison = Compare(first_full, oracle_full);
      const Comparison determinism = Compare(first_full, second_full);
      const Comparison producer_full_comparison
          = Compare(first_producer_full, oracle_producer_full);
      const Comparison producer_full_determinism
          = Compare(first_producer_full, second_producer_full);
      const Comparison producer_snapshot_transport
          = Compare(snapshot_values, producer_values);
      const Comparison snapshot_full_comparison
          = Compare(first_snapshot_full, oracle_producer_full);
      const Comparison producer_full_vs_grf256_oracle
          = Compare(first_producer_full, grf256_oracle_producer_full);
      const Comparison stock_vs_grf256_producer_full
          = Compare(oracle_producer_full, grf256_oracle_producer_full);
      const Comparison oracle_identity_comparison
          = Compare(oracle_identity, stored_gate);
      const Comparison swish_output_vs_stored_gate
          = Compare(observed_swish, stored_gate);
      const Comparison full_output_vs_stock_binary
          = Compare(first_full, oracle_binary);
      const Comparison swish_vs_grf256_oracle
          = Compare(observed_swish, grf256_oracle_swish);
      const Comparison full_vs_grf256_oracle
          = Compare(first_full, grf256_oracle_full);
      const Comparison swish_vs_exp_oracle
          = Compare(observed_swish, oracle_exp_swish);
      const Comparison full_vs_exp_oracle
          = Compare(first_full, oracle_exp_full);
      const Comparison stock_vs_polynomial_swish
          = Compare(oracle_polynomial_swish, oracle_swish);
      const Comparison stock_vs_polynomial_full
          = Compare(oracle_polynomial_full, oracle_full);
      const Comparison fused_vs_polynomial_swish
          = Compare(observed_swish, oracle_polynomial_swish);
      const Comparison fused_vs_polynomial_full
          = Compare(first_full, oracle_polynomial_full);

      if (!first_scenario) std::cout << ",";
      first_scenario = false;
      std::cout << "{\"name\":" << JsonString(name)
                << ",\"max_group\":" << max_group
                << ",\"active_groups\":" << active_groups
                << ",\"binary_vs_stored_gate_oracle\":";
      PrintComparison(binary_comparison);
      std::cout << ",\"swish_vs_stored_gate_oracle\":";
      PrintComparison(swish_comparison);
      std::cout << ",\"swish_f32_vs_stock_f32\":";
      PrintFloatBitsComparison(swish_f32_comparison);
      std::cout << ",\"diagnostic_swish_f32_vs_correctly_rounded\":";
      PrintFloatBitsComparison(swish_f32_vs_correctly_rounded);
      std::cout << ",\"diagnostic_stock_f32_vs_correctly_rounded\":";
      PrintFloatBitsComparison(stock_f32_vs_correctly_rounded);
      std::cout << ",\"swish_binary_vs_stock_pipeline\":";
      PrintComparison(full_comparison);
      std::cout << ",\"swish_binary_repeat_determinism\":";
      PrintComparison(determinism);
      std::cout << ",\"producer_fed_swiglu_vs_stock_pipeline\":";
      PrintComparison(producer_full_comparison);
      std::cout << ",\"producer_fed_swiglu_repeat_determinism\":";
      PrintComparison(producer_full_determinism);
      std::cout << ",\"producer_snapshot_transport\":";
      PrintComparison(producer_snapshot_transport);
      std::cout << ",\"snapshot_fed_swiglu_vs_stock_pipeline\":";
      PrintComparison(snapshot_full_comparison);
      std::cout << ",\"diagnostic_producer_fed_vs_grf256_oracle\":";
      PrintComparison(producer_full_vs_grf256_oracle);
      std::cout << ",\"diagnostic_stock_vs_grf256_producer_full\":";
      PrintComparison(stock_vs_grf256_producer_full);
      std::cout << ",\"stock_oracle_identity_io\":";
      PrintComparison(oracle_identity_comparison);
      std::cout << ",\"diagnostic_swish_output_vs_stored_gate\":";
      PrintComparison(swish_output_vs_stored_gate);
      std::cout << ",\"diagnostic_full_output_vs_stock_binary\":";
      PrintComparison(full_output_vs_stock_binary);
      std::cout << ",\"diagnostic_swish_vs_grf256_oracle\":";
      PrintComparison(swish_vs_grf256_oracle);
      std::cout << ",\"diagnostic_full_vs_grf256_oracle\":";
      PrintComparison(full_vs_grf256_oracle);
      std::cout << ",\"diagnostic_swish_vs_exp_oracle\":";
      PrintComparison(swish_vs_exp_oracle);
      std::cout << ",\"diagnostic_full_vs_exp_oracle\":";
      PrintComparison(full_vs_exp_oracle);
      std::cout << ",\"diagnostic_stock_vs_polynomial_swish\":";
      PrintComparison(stock_vs_polynomial_swish);
      std::cout << ",\"diagnostic_stock_vs_polynomial_full\":";
      PrintComparison(stock_vs_polynomial_full);
      std::cout << ",\"diagnostic_fused_vs_polynomial_swish\":";
      PrintComparison(fused_vs_polynomial_swish);
      std::cout << ",\"diagnostic_fused_vs_polynomial_full\":";
      PrintComparison(fused_vs_polynomial_full);
      if (!producer_full_comparison.mismatch_indices.empty()) {
        std::cout << ",\"producer_activation_witnesses\":[";
        for (std::size_t sample = 0;
             sample < producer_full_comparison.mismatch_indices.size();
             ++sample) {
          if (sample != 0) std::cout << ",";
          const std::size_t sample_index
              = producer_full_comparison.mismatch_indices[sample];
          std::cout << "{\"index\":" << sample_index
                    << ",\"row\":" << sample_index / kOutput
                    << ",\"column\":" << sample_index % kOutput
                    << ",\"stored_gate_bits\":"
                    << stored_gate[sample_index]
                    << ",\"stored_gate\":"
                    << HalfToFloat(stored_gate[sample_index])
                    << ",\"producer_up_bits\":"
                    << producer_values[sample_index]
                    << ",\"producer_up\":"
                    << HalfToFloat(producer_values[sample_index])
                    << ",\"direct_fused_bits\":"
                    << first_producer_full[sample_index]
                    << ",\"snapshot_fused_bits\":"
                    << first_snapshot_full[sample_index]
                    << ",\"stock_bits\":"
                    << oracle_producer_full[sample_index]
                    << "}";
        }
        std::cout << "]";
      }
      if (swish_comparison.first_index
          != std::numeric_limits<std::size_t>::max()) {
        const std::size_t witness = swish_comparison.first_index;
        const auto binary_values = ReadMemory<std::uint16_t>(binary);
        std::cout << ",\"activation_witness\":{\"index\":" << witness
                  << ",\"stored_gate_bits\":" << stored_gate[witness]
                  << ",\"stored_gate\":"
                  << HalfToFloat(stored_gate[witness])
                  << ",\"binary_bits\":" << binary_values[witness]
                  << ",\"binary\":" << HalfToFloat(binary_values[witness])
                  << ",\"fused_bits\":" << observed_swish[witness]
                  << ",\"stock_native_bits\":" << oracle_swish[witness]
                  << ",\"stock_exp_bits\":" << oracle_exp_swish[witness]
                  << "}";
        std::cout << ",\"activation_witnesses\":[";
        for (std::size_t sample = 0;
             sample < swish_comparison.mismatch_indices.size(); ++sample) {
          if (sample != 0) std::cout << ",";
          const std::size_t sample_index
              = swish_comparison.mismatch_indices[sample];
          std::cout << "{\"index\":" << sample_index
                    << ",\"row\":" << sample_index / kOutput
                    << ",\"column\":" << sample_index % kOutput
                    << ",\"stored_gate_bits\":"
                    << stored_gate[sample_index]
                    << ",\"stored_gate\":"
                    << HalfToFloat(stored_gate[sample_index])
                    << ",\"binary_bits\":" << binary_values[sample_index]
                    << ",\"binary\":"
                    << HalfToFloat(binary_values[sample_index])
                    << ",\"fused_bits\":" << observed_swish[sample_index]
                    << ",\"stock_native_bits\":"
                    << oracle_swish[sample_index]
                    << ",\"stock_exp_bits\":"
                    << oracle_exp_swish[sample_index]
                    << "}";
        }
        std::cout << "]";
      }
      std::cout << "}";
    }

    const auto finite_half_values = FiniteHalfValues();
    dnnl::memory census_source(source_desc, engine);
    dnnl::memory census_weights(weights_desc, engine);
    dnnl::memory census_scales(scale_desc, engine);
    dnnl::memory census_zero_points(zero_point_desc, engine);
    WriteMemory(MakeIdentityWeights(), census_weights);
    WriteMemory(MakeUnitScales(), census_scales);
    WriteMemory(
        std::vector<std::uint8_t>(
            zero_point_desc.get_size(), static_cast<std::uint8_t>(0x88)),
        census_zero_points);
    PrimitiveCase census_base(
        engine, source_desc, weights_desc, output_desc, output_desc,
        PostOpsKind::kNone, census_source, census_weights, census_scales,
        census_zero_points, binary, max_group_hint);
    PrimitiveCase census_swish(
        engine, source_desc, weights_desc, output_desc, output_desc,
        PostOpsKind::kSwish, census_source, census_weights, census_scales,
        census_zero_points, binary, max_group_hint);
    PrimitiveCase census_swish_f32(
        engine, source_desc, weights_desc, output_f32_desc, output_desc,
        PostOpsKind::kSwish, census_source, census_weights, census_scales,
        census_zero_points, binary, max_group_hint);
    std::array<std::int32_t, kExperts> census_counts{};
    census_counts.fill(static_cast<std::int32_t>(kRows / kExperts));
    const auto census_offsets = PrefixOffsets(census_counts);
    WriteOffsets(census_source, census_offsets);
    WriteOffsets(census_base.output, census_offsets);
    WriteOffsets(census_swish.output, census_offsets);
    WriteOffsets(census_swish_f32.output, census_offsets);
    WriteOffsets(oracle_output, census_offsets);
    WriteOffsets(oracle_f32_output, census_offsets);
    WriteOffsets(correctly_rounded_f32_output, census_offsets);
    std::vector<std::uint16_t> intended_gates;
    std::vector<std::uint16_t> stored_gates;
    std::vector<std::uint16_t> fused_swish;
    std::vector<std::uint16_t> stock_swish;
    std::vector<std::uint32_t> fused_swish_f32;
    std::vector<std::uint32_t> stock_swish_f32;
    std::vector<std::uint32_t> correctly_rounded_swish_f32;
    intended_gates.reserve(finite_half_values.size());
    stored_gates.reserve(finite_half_values.size());
    fused_swish.reserve(finite_half_values.size());
    stock_swish.reserve(finite_half_values.size());
    fused_swish_f32.reserve(finite_half_values.size());
    stock_swish_f32.reserve(finite_half_values.size());
    correctly_rounded_swish_f32.reserve(finite_half_values.size());
    std::size_t page_count = 0;
    for (std::size_t offset = 0; offset < finite_half_values.size();
         offset += kValues) {
      ++page_count;
      WriteMemory(
          MakeActivationCensusSource(finite_half_values, offset),
          census_source);
      WriteOffsets(census_source, census_offsets);
      census_base.primitive.execute(
          stream, census_base.arguments);
      census_swish.primitive.execute(
          stream, census_swish.arguments);
      census_swish_f32.primitive.execute(
          stream, census_swish_f32.arguments);
      stream.wait();
      const auto page_stored
          = ReadMemory<std::uint16_t>(census_base.output);
      const auto page_fused
          = ReadMemory<std::uint16_t>(census_swish.output);
      const auto page_fused_f32
          = ReadMemory<std::uint32_t>(census_swish_f32.output);
      oracle.Run(census_base.output, binary, oracle_output, 1);
      const auto page_stock
          = ReadMemory<std::uint16_t>(oracle_output);
      oracle.RunF32(census_base.output, oracle_f32_output);
      const auto page_stock_f32
          = ReadMemory<std::uint32_t>(oracle_f32_output);
      correctly_rounded_oracle.RunF32(
          census_base.output, correctly_rounded_f32_output);
      const auto page_correctly_rounded_f32
          = ReadMemory<std::uint32_t>(correctly_rounded_f32_output);
      const std::size_t count
          = std::min(kValues, finite_half_values.size() - offset);
      intended_gates.insert(
          intended_gates.end(), finite_half_values.begin() + offset,
          finite_half_values.begin() + offset + count);
      stored_gates.insert(
          stored_gates.end(), page_stored.begin(),
          page_stored.begin() + count);
      fused_swish.insert(
          fused_swish.end(), page_fused.begin(),
          page_fused.begin() + count);
      stock_swish.insert(
          stock_swish.end(), page_stock.begin(),
          page_stock.begin() + count);
      fused_swish_f32.insert(
          fused_swish_f32.end(), page_fused_f32.begin(),
          page_fused_f32.begin() + count);
      stock_swish_f32.insert(
          stock_swish_f32.end(), page_stock_f32.begin(),
          page_stock_f32.begin() + count);
      correctly_rounded_swish_f32.insert(
          correctly_rounded_swish_f32.end(),
          page_correctly_rounded_f32.begin(),
          page_correctly_rounded_f32.begin() + count);
    }
    const Comparison census_transport
        = Compare(stored_gates, intended_gates);
    const Comparison census_swish_comparison
        = Compare(fused_swish, stock_swish);
    const FloatBitsComparison census_swish_f32_comparison
        = CompareFloatBits(fused_swish_f32, stock_swish_f32);
    const FloatBitsComparison
        census_swish_f32_vs_correctly_rounded
        = CompareFloatBits(
            fused_swish_f32, correctly_rounded_swish_f32);
    const FloatBitsComparison
        census_stock_f32_vs_correctly_rounded
        = CompareFloatBits(
            stock_swish_f32, correctly_rounded_swish_f32);
    std::array<bool, 1U << 16> observed_gate_bits{};
    for (std::uint16_t bits : stored_gates) observed_gate_bits[bits] = true;
    const std::size_t unique_stored_gate_bits = static_cast<std::size_t>(
        std::count(
            observed_gate_bits.begin(), observed_gate_bits.end(), true));
    std::cout << "],\"exhaustive_finite_f16_activation\":{"
              << "\"input_value_count\":" << finite_half_values.size()
              << ",\"page_count\":" << page_count
              << ",\"unique_stored_gate_bits\":"
              << unique_stored_gate_bits
              << ",\"base_provider\":"
              << JsonString(census_base.implementation)
              << ",\"swish_provider\":"
              << JsonString(census_swish.implementation)
              << ",\"swish_f32_provider\":"
              << JsonString(census_swish_f32.implementation)
              << ",\"identity_transport\":";
    PrintComparison(census_transport);
    std::cout << ",\"swish_vs_stock_oracle\":";
    PrintComparison(census_swish_comparison);
    std::cout << ",\"swish_f32_vs_stock_f32\":";
    PrintFloatBitsComparison(census_swish_f32_comparison);
    std::cout << ",\"swish_f32_vs_correctly_rounded\":";
    PrintFloatBitsComparison(
        census_swish_f32_vs_correctly_rounded);
    std::cout << ",\"stock_f32_vs_correctly_rounded\":";
    PrintFloatBitsComparison(
        census_stock_f32_vs_correctly_rounded);
    if (!census_swish_comparison.mismatch_indices.empty()) {
      std::cout << ",\"activation_witnesses\":[";
      for (std::size_t sample = 0;
           sample < census_swish_comparison.mismatch_indices.size();
           ++sample) {
        if (sample != 0) std::cout << ",";
        const std::size_t index
            = census_swish_comparison.mismatch_indices[sample];
        std::cout << "{\"index\":" << index
                  << ",\"stored_gate_bits\":" << stored_gates[index]
                  << ",\"stored_gate\":"
                  << HalfToFloat(stored_gates[index])
                  << ",\"fused_bits\":" << fused_swish[index]
                  << ",\"stock_native_bits\":" << stock_swish[index]
                  << "}";
      }
      std::cout << "]";
    }
    if (!census_swish_f32_comparison.mismatch_indices.empty()) {
      std::cout << ",\"f32_activation_witnesses\":[";
      for (std::size_t sample = 0;
           sample < census_swish_f32_comparison.mismatch_indices.size();
           ++sample) {
        if (sample != 0) std::cout << ",";
        const std::size_t index
            = census_swish_f32_comparison.mismatch_indices[sample];
        std::cout << "{\"index\":" << index
                  << ",\"stored_gate_bits\":" << stored_gates[index]
                  << ",\"stored_gate\":"
                  << HalfToFloat(stored_gates[index])
                  << ",\"fused_bits\":" << fused_swish_f32[index]
                  << ",\"fused\":"
                  << FloatBitsToFloat(fused_swish_f32[index])
                  << ",\"stock_bits\":" << stock_swish_f32[index]
                  << ",\"stock\":"
                  << FloatBitsToFloat(stock_swish_f32[index])
                  << "}";
      }
      std::cout << "]";
    }
    std::cout << "},\"speedup_claims_allowed\":false}\n";
    return 0;
  } catch (const dnnl::error& error) {
    std::cerr << "oneDNN error status=" << error.status
              << ": " << error.what() << "\n";
    return 2;
  } catch (const std::exception& error) {
    std::cerr << error.what() << "\n";
    return 1;
  }
}
