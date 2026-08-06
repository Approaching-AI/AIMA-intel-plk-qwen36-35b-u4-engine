#include "intel_qwen36/gguf_loader.hpp"

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kExpertCount = 256;
constexpr std::size_t kTokenCount = 1024;
constexpr std::size_t kSelectedCount = 8;
constexpr std::size_t kAssignmentCount = kTokenCount * kSelectedCount;
constexpr std::size_t kInputSize = 512;
constexpr std::size_t kOutputSize = 2048;
constexpr std::size_t kQ6BlockValues = 256;
constexpr std::size_t kQ6BlockBytes = 210;
constexpr std::size_t kGroupsPerRow = kInputSize / 32;
constexpr std::size_t kExactGroupsPerRow = kInputSize / 16;
constexpr std::size_t kOutputTiles = kOutputSize / 16;

[[noreturn]] void Fail(const std::string& message) {
  throw std::runtime_error(message);
}

void Require(bool condition, const std::string& message) {
  if (!condition) Fail(message);
}

void Check(cl_int status, const std::string& label) {
  if (status != CL_SUCCESS) {
    Fail(label + " failed with OpenCL error " + std::to_string(status));
  }
}

std::vector<std::uint8_t> ReadBytes(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  Require(static_cast<bool>(input), "could not open " + path);
  input.seekg(0, std::ios::end);
  const auto size = input.tellg();
  Require(size >= 0, "could not size " + path);
  input.seekg(0, std::ios::beg);
  std::vector<std::uint8_t> values(static_cast<std::size_t>(size));
  input.read(reinterpret_cast<char*>(values.data()),
             static_cast<std::streamsize>(values.size()));
  Require(static_cast<bool>(input), "could not read " + path);
  return values;
}

template <typename Value>
std::vector<Value> ReadVector(const std::string& path,
                              std::size_t expected_count) {
  const auto bytes = ReadBytes(path);
  Require(bytes.size() == expected_count * sizeof(Value),
          "input size mismatch: " + path);
  std::vector<Value> values(expected_count);
  std::memcpy(values.data(), bytes.data(), bytes.size());
  return values;
}

template <typename Value>
void WriteVector(const std::filesystem::path& path,
                 const std::vector<Value>& values) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  Require(static_cast<bool>(output), "could not create " + path.string());
  output.write(reinterpret_cast<const char*>(values.data()),
               static_cast<std::streamsize>(values.size() * sizeof(Value)));
  Require(static_cast<bool>(output), "could not write " + path.string());
}

std::string ReadText(const std::string& path) {
  const auto bytes = ReadBytes(path);
  return std::string(bytes.begin(), bytes.end());
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

std::vector<std::uint8_t> ReadTensor(
    const std::string& model_path,
    const iq36::GgufTensorInfo& tensor) {
  std::ifstream input(model_path, std::ios::binary);
  Require(static_cast<bool>(input), "could not open model");
  input.seekg(static_cast<std::streamoff>(tensor.absolute_offset),
              std::ios::beg);
  Require(static_cast<bool>(input), "could not seek model tensor");
  std::vector<std::uint8_t> values(static_cast<std::size_t>(tensor.nbytes));
  input.read(reinterpret_cast<char*>(values.data()),
             static_cast<std::streamsize>(values.size()));
  Require(input.gcount() == static_cast<std::streamsize>(values.size()),
          "could not read model tensor");
  return values;
}

std::uint16_t LoadU16(const std::uint8_t* bytes) {
  return std::uint16_t(bytes[0]) | (std::uint16_t(bytes[1]) << 8);
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

std::uint16_t FloatToHalf(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const std::uint32_t sign = (bits >> 16) & 0x8000U;
  const std::uint32_t exponent = (bits >> 23) & 0xffU;
  const std::uint32_t mantissa = bits & 0x7fffffU;
  if (exponent == 0xffU) {
    return static_cast<std::uint16_t>(
        sign | (mantissa == 0 ? 0x7c00U : 0x7e00U));
  }
  const int half_exponent = static_cast<int>(exponent) - 127 + 15;
  if (half_exponent >= 31) {
    return static_cast<std::uint16_t>(sign | 0x7c00U);
  }
  if (half_exponent <= 0) {
    if (half_exponent < -10) return static_cast<std::uint16_t>(sign);
    std::uint32_t normalized = mantissa | 0x800000U;
    const int shift = 14 - half_exponent;
    std::uint32_t half_mantissa = normalized >> shift;
    const std::uint32_t remainder = normalized & ((1U << shift) - 1U);
    const std::uint32_t halfway = 1U << (shift - 1);
    if (remainder > halfway ||
        (remainder == halfway && (half_mantissa & 1U) != 0)) {
      ++half_mantissa;
    }
    return static_cast<std::uint16_t>(sign | half_mantissa);
  }
  std::uint32_t half_mantissa = mantissa >> 13;
  const std::uint32_t remainder = mantissa & 0x1fffU;
  if (remainder > 0x1000U ||
      (remainder == 0x1000U && (half_mantissa & 1U) != 0)) {
    ++half_mantissa;
    if (half_mantissa == 0x400U) {
      half_mantissa = 0;
      if (half_exponent + 1 >= 31) {
        return static_cast<std::uint16_t>(sign | 0x7c00U);
      }
      return static_cast<std::uint16_t>(
          sign | static_cast<std::uint32_t>(half_exponent + 1) << 10);
    }
  }
  return static_cast<std::uint16_t>(
      sign | static_cast<std::uint32_t>(half_exponent) << 10 |
      half_mantissa);
}

int NearestInt(float value) {
  const float shifted = value + 12582912.0f;
  int bits = 0;
  std::memcpy(&bits, &shifted, sizeof(bits));
  return (bits & 0x007fffff) - 0x00400000;
}

int Q6Value(const std::uint8_t* block, std::size_t index) {
  const std::size_t half = index / 128;
  const std::size_t within = index % 128;
  const std::size_t quadrant = within / 32;
  const std::size_t lane = within % 32;
  const std::uint8_t high = block[128 + half * 32 + lane];
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

struct PrepackedWeights {
  std::vector<std::uint8_t> values;
  std::vector<float> scales;
  std::vector<std::int8_t> integer_scales;
  std::vector<float> block_scales;
  double maximum_absolute_error = 0.0;
  double relative_l2_error = 0.0;
};

PrepackedWeights PrepackQ6Surrogate(
    const std::vector<std::uint8_t>& raw) {
  const std::size_t row_bytes = 2 * kQ6BlockBytes;
  Require(raw.size() == kExpertCount * kOutputSize * row_bytes,
          "Q6 tensor size mismatch");
  PrepackedWeights result;
  result.values.resize(kExpertCount * kOutputSize * kInputSize);
  result.scales.resize(kExpertCount * kOutputSize * kGroupsPerRow);
  double maximum_error = 0.0;
  double error_squared = 0.0;
  double reference_squared = 0.0;
#pragma omp parallel for schedule(static) reduction(max : maximum_error) \
    reduction(+ : error_squared, reference_squared)
  for (std::int64_t expert_signed = 0;
       expert_signed < static_cast<std::int64_t>(kExpertCount);
       ++expert_signed) {
    const std::size_t expert = static_cast<std::size_t>(expert_signed);
    for (std::size_t output = 0; output < kOutputSize; ++output) {
      const auto* row = raw.data() +
          (expert * kOutputSize + output) * row_bytes;
      for (std::size_t group = 0; group < kGroupsPerRow; ++group) {
        const std::size_t block_index = group / 8;
        const std::size_t group_in_block = group % 8;
        const auto* block = row + block_index * kQ6BlockBytes;
        const float d = HalfToFloat(LoadU16(block + 208));
        const auto* q6_scales =
            reinterpret_cast<const std::int8_t*>(block + 192);
        std::array<float, 32> decoded{};
        float maximum = 0.0f;
        for (std::size_t within = 0; within < 32; ++within) {
          const std::size_t k = group_in_block * 32 + within;
          const std::size_t scale_group = k / 16;
          decoded[within] = d * static_cast<float>(q6_scales[scale_group]) *
              static_cast<float>(Q6Value(block, k));
          maximum = std::max(maximum, std::abs(decoded[within]));
        }
        const float scale = maximum == 0.0f ? 0.0f : maximum / 127.0f;
        const std::size_t output_tile = output / 16;
        const std::size_t lane = output % 16;
        const std::size_t scale_index =
            (((expert * kOutputTiles + output_tile) * kGroupsPerRow + group) *
                 16 + lane);
        result.scales[scale_index] = scale;
        for (std::size_t within = 0; within < 32; ++within) {
          const int quantized = scale == 0.0f ? 0 : std::clamp(
              NearestInt(decoded[within] / scale), -127, 127);
          const float restored = scale * static_cast<float>(quantized);
          const double error = static_cast<double>(restored) - decoded[within];
          maximum_error = std::max(maximum_error, std::abs(error));
          error_squared += error * error;
          reference_squared +=
              static_cast<double>(decoded[within]) * decoded[within];
          const std::size_t value_index =
              ((((expert * kOutputTiles + output_tile) * kGroupsPerRow +
                  group) * 16 + lane) * 32 + within);
          result.values[value_index] =
              static_cast<std::uint8_t>(quantized + 128);
        }
      }
    }
  }
  result.maximum_absolute_error = maximum_error;
  result.relative_l2_error = std::sqrt(error_squared / reference_squared);
  return result;
}

PrepackedWeights PrepackQ6ExactPer16(
    const std::vector<std::uint8_t>& raw) {
  const std::size_t row_bytes = 2 * kQ6BlockBytes;
  Require(raw.size() == kExpertCount * kOutputSize * row_bytes,
          "Q6 tensor size mismatch");
  PrepackedWeights result;
  result.values.resize(kExpertCount * kOutputSize * kInputSize);
  result.scales.resize(kExpertCount * kOutputSize * kExactGroupsPerRow);
  result.integer_scales.resize(
      kExpertCount * kOutputSize * kExactGroupsPerRow);
  result.block_scales.resize(kExpertCount * kOutputSize * 2);
  double maximum_error = 0.0;
  double error_squared = 0.0;
  double reference_squared = 0.0;
#pragma omp parallel for schedule(static) reduction(max : maximum_error) \
    reduction(+ : error_squared, reference_squared)
  for (std::int64_t expert_signed = 0;
       expert_signed < static_cast<std::int64_t>(kExpertCount);
       ++expert_signed) {
    const std::size_t expert = static_cast<std::size_t>(expert_signed);
    for (std::size_t output = 0; output < kOutputSize; ++output) {
      const auto* row = raw.data() +
          (expert * kOutputSize + output) * row_bytes;
      for (std::size_t group = 0; group < kExactGroupsPerRow; ++group) {
        const std::size_t block_index = group / 16;
        const std::size_t group_in_block = group % 16;
        const auto* block = row + block_index * kQ6BlockBytes;
        const float d = HalfToFloat(LoadU16(block + 208));
        const auto* q6_scales =
            reinterpret_cast<const std::int8_t*>(block + 192);
        const float scale =
            d * static_cast<float>(q6_scales[group_in_block]);
        const std::size_t output_tile = output / 16;
        const std::size_t lane = output % 16;
        const std::size_t scale_index =
            (((expert * kOutputTiles + output_tile) * kExactGroupsPerRow +
              group) * 16 + lane);
        result.scales[scale_index] = scale;
        result.integer_scales[scale_index] = q6_scales[group_in_block];
        if (group_in_block == 0) {
          const std::size_t block_scale_index =
              (((expert * kOutputTiles + output_tile) * 2 + block_index) *
                   16 + lane);
          result.block_scales[block_scale_index] = d;
        }
        for (std::size_t within = 0; within < 16; ++within) {
          const std::size_t k = group_in_block * 16 + within;
          const int quantized = Q6Value(block, k);
          const float decoded = scale * static_cast<float>(quantized);
          const float restored = scale * static_cast<float>(quantized);
          const double error = static_cast<double>(restored) - decoded;
          maximum_error = std::max(maximum_error, std::abs(error));
          error_squared += error * error;
          reference_squared +=
              static_cast<double>(decoded) * decoded;
          const std::size_t value_index =
              ((((expert * kOutputTiles + output_tile) *
                  kExactGroupsPerRow + group) * 16 + lane) * 16 + within);
          result.values[value_index] =
              static_cast<std::uint8_t>(quantized + 128);
        }
      }
    }
  }
  result.maximum_absolute_error = maximum_error;
  result.relative_l2_error = std::sqrt(error_squared / reference_squared);
  return result;
}

struct WorkTile {
  std::int32_t expert;
  std::int32_t row_start;
  std::int32_t row_end;
  std::int32_t unused;
};

struct GroupedInput {
  std::vector<std::int8_t> q8;
  std::vector<float> scales;
  std::vector<std::int16_t> sums;
  std::vector<float> router_weights;
  std::vector<std::int32_t> inverse_map;
  std::vector<WorkTile> work_tiles;
  std::vector<WorkTile> shared64_tiles;
  std::vector<std::int32_t> expert_offsets;
  std::vector<WorkTile> flat_work_tiles;
  std::vector<std::uint32_t> task_coordinates;
  std::size_t flat_task_count = 0;
  std::size_t active_experts = 0;
  std::size_t maximum_group = 0;
};

GroupedInput MakeGroupedInput(
    const std::vector<float>& swiglu,
    const std::vector<std::uint8_t>& topk,
    std::size_t topk_stride,
    const std::vector<float>& router_weights,
    std::size_t rows_per_work_tile,
    bool hybrid_shared64,
    bool exact_per16) {
  Require(swiglu.size() == kAssignmentCount * kInputSize,
          "SwiGLU input count mismatch");
  Require(topk.size() >= (kTokenCount - 1) * topk_stride +
              kSelectedCount * sizeof(std::int32_t),
          "top-k input is truncated");
  Require(router_weights.size() == kAssignmentCount,
          "router weight count mismatch");
  Require(rows_per_work_tile == 16 || rows_per_work_tile == 24 ||
              rows_per_work_tile == 32 ||
              rows_per_work_tile == 48 ||
              rows_per_work_tile == 64,
          "work tile must be M16, M24, M32, M48, or M64");
  Require(!hybrid_shared64 || rows_per_work_tile == 32,
          "hybrid shared64 requires M32 tails");
  std::array<std::int32_t, kExpertCount> counts{};
  std::array<std::int32_t, kAssignmentCount> expert_ids{};
  for (std::size_t token = 0; token < kTokenCount; ++token) {
    for (std::size_t rank = 0; rank < kSelectedCount; ++rank) {
      std::int32_t expert = -1;
      std::memcpy(&expert,
                  topk.data() + token * topk_stride +
                      rank * sizeof(std::int32_t),
                  sizeof(expert));
      Require(expert >= 0 && expert < static_cast<std::int32_t>(kExpertCount),
              "top-k expert is out of range");
      expert_ids[token * kSelectedCount + rank] = expert;
      ++counts[static_cast<std::size_t>(expert)];
    }
  }
  std::array<std::int32_t, kExpertCount> starts{};
  std::array<std::int32_t, kExpertCount> cursors{};
  std::int32_t cumulative = 0;
  GroupedInput result;
  result.expert_offsets.resize(kExpertCount);
  for (std::size_t expert = 0; expert < kExpertCount; ++expert) {
    starts[expert] = cumulative;
    cursors[expert] = cumulative;
    cumulative += counts[expert];
    result.expert_offsets[expert] = cumulative;
    const std::uint32_t m_tiles = static_cast<std::uint32_t>(
        (counts[expert] + static_cast<std::int32_t>(rows_per_work_tile) - 1) /
        static_cast<std::int32_t>(rows_per_work_tile));
    for (std::uint32_t output_tile = 0;
         output_tile < kOutputTiles; ++output_tile) {
      for (std::uint32_t m_tile = 0; m_tile < m_tiles; ++m_tile) {
        result.flat_work_tiles.push_back({
            static_cast<std::int32_t>(expert),
            starts[expert] + static_cast<std::int32_t>(
                m_tile * rows_per_work_tile),
            cumulative, static_cast<std::int32_t>(output_tile)});
        result.task_coordinates.push_back(
            static_cast<std::uint32_t>(expert) | (output_tile << 8) |
            (m_tile << 15));
      }
    }
    result.active_experts += counts[expert] != 0;
    result.maximum_group = std::max(
        result.maximum_group, static_cast<std::size_t>(counts[expert]));
    std::int32_t row = starts[expert];
    if (hybrid_shared64) {
      while (row + 64 <= cumulative) {
        result.shared64_tiles.push_back({
            static_cast<std::int32_t>(expert), row, row + 64, 0});
        row += 64;
      }
    }
    for (; row < cumulative;
         row += static_cast<std::int32_t>(rows_per_work_tile)) {
      result.work_tiles.push_back({static_cast<std::int32_t>(expert), row,
                                   cumulative, 0});
    }
  }
  Require(cumulative == static_cast<std::int32_t>(kAssignmentCount),
          "assignment count mismatch");
  result.flat_task_count = result.flat_work_tiles.size();
  result.q8.resize(kAssignmentCount * kInputSize);
  result.scales.resize(kAssignmentCount * 2);
  const std::size_t sum_groups =
      exact_per16 ? kExactGroupsPerRow : kGroupsPerRow;
  const std::size_t sum_group_values = exact_per16 ? 16 : 32;
  result.sums.resize(kAssignmentCount * sum_groups);
  result.router_weights.resize(kAssignmentCount);
  result.inverse_map.resize(kAssignmentCount, -1);

#pragma omp parallel for schedule(static)
  for (std::int64_t source_signed = 0;
       source_signed < static_cast<std::int64_t>(kAssignmentCount);
       ++source_signed) {
    const std::size_t source = static_cast<std::size_t>(source_signed);
    const std::size_t expert =
        static_cast<std::size_t>(expert_ids[source]);
    std::int32_t row = 0;
#pragma omp critical(iq36_grouped_cursor)
    {
      row = cursors[expert]++;
      result.inverse_map[source] = row;
      result.router_weights[static_cast<std::size_t>(row)] =
          router_weights[source];
    }
    const auto* input = swiglu.data() + source * kInputSize;
    for (std::size_t block = 0; block < 2; ++block) {
      float maximum = 0.0f;
      float signed_maximum = 0.0f;
      for (std::size_t index = 0; index < kQ6BlockValues; ++index) {
        const float value = input[block * kQ6BlockValues + index];
        if (std::abs(value) > maximum) {
          maximum = std::abs(value);
          signed_maximum = value;
        }
      }
      const float inverse = maximum == 0.0f ? 0.0f : -127.0f / signed_maximum;
      const float scale = inverse == 0.0f ? 0.0f : 1.0f / inverse;
      result.scales[static_cast<std::size_t>(row) * 2 + block] = scale;
      for (std::size_t index = 0; index < kQ6BlockValues; ++index) {
        const int quantized = inverse == 0.0f ? 0 : std::min(
            127, NearestInt(inverse *
                input[block * kQ6BlockValues + index]));
        result.q8[static_cast<std::size_t>(row) * kInputSize +
                  block * kQ6BlockValues + index] =
            static_cast<std::int8_t>(quantized);
      }
    }
    for (std::size_t group = 0; group < sum_groups; ++group) {
      std::int32_t sum = 0;
      for (std::size_t index = 0; index < sum_group_values; ++index) {
        sum += result.q8[static_cast<std::size_t>(row) * kInputSize +
                         group * sum_group_values + index];
      }
      result.sums[static_cast<std::size_t>(row) * sum_groups + group] =
          static_cast<std::int16_t>(sum);
    }
  }
  Require(std::none_of(result.inverse_map.begin(), result.inverse_map.end(),
                       [](std::int32_t value) { return value < 0; }),
          "inverse map is incomplete");
  return result;
}

struct Args {
  std::string model;
  std::string tensor = "blk.7.ffn_down_exps.weight";
  std::string kernel;
  std::string swiglu;
  std::string topk;
  std::size_t topk_stride = 0;
  std::string router_weights;
  std::string oracle;
  int warmup = 3;
  int repeat = 7;
  double kernel_cap_us = 3400.0;
  std::size_t m_tile = 32;
  bool hybrid_shared64 = false;
  bool f16_weight_scales = false;
  std::string dump_prepacked_dir;
  std::size_t workgroup_subgroups = 1;
  bool flatten_output_tasks = false;
  bool prepack_only = false;
  bool exact_per16 = false;
  bool exact_block_accum = false;
  bool round_swiglu_f16 = false;
};

Args ParseArgs(int argc, char** argv) {
  Args args;
  for (int index = 1; index < argc; ++index) {
    const std::string key = argv[index];
    auto value = [&]() {
      Require(index + 1 < argc, "missing value after " + key);
      return std::string(argv[++index]);
    };
    if (key == "--model") args.model = value();
    else if (key == "--tensor") args.tensor = value();
    else if (key == "--kernel") args.kernel = value();
    else if (key == "--swiglu") args.swiglu = value();
    else if (key == "--topk") args.topk = value();
    else if (key == "--topk-stride") args.topk_stride = std::stoull(value());
    else if (key == "--router-weights") args.router_weights = value();
    else if (key == "--oracle") args.oracle = value();
    else if (key == "--warmup") args.warmup = std::stoi(value());
    else if (key == "--repeat") args.repeat = std::stoi(value());
    else if (key == "--kernel-cap-us") args.kernel_cap_us = std::stod(value());
    else if (key == "--m-tile") args.m_tile = std::stoull(value());
    else if (key == "--hybrid-shared64") args.hybrid_shared64 = true;
    else if (key == "--f16-weight-scales") args.f16_weight_scales = true;
    else if (key == "--dump-prepacked-dir") args.dump_prepacked_dir = value();
    else if (key == "--workgroup-subgroups") {
      args.workgroup_subgroups = std::stoull(value());
    }
    else if (key == "--flatten-output-tasks") {
      args.flatten_output_tasks = true;
    }
    else if (key == "--prepack-only") args.prepack_only = true;
    else if (key == "--exact-per16") args.exact_per16 = true;
    else if (key == "--exact-block-accum") {
      args.exact_per16 = true;
      args.exact_block_accum = true;
    }
    else if (key == "--round-swiglu-f16") args.round_swiglu_f16 = true;
    else Fail("unknown argument: " + key);
  }
  Require(!args.model.empty(), "model is required");
  if (args.prepack_only) {
    Require(!args.dump_prepacked_dir.empty(),
            "--prepack-only requires --dump-prepacked-dir");
  } else {
    Require(!args.kernel.empty() && !args.swiglu.empty() &&
                !args.topk.empty() && !args.router_weights.empty() &&
                !args.oracle.empty(),
            "kernel, swiglu, topk, router-weights, and oracle are required");
  }
  if (args.prepack_only) return args;
  Require(args.topk_stride >= kSelectedCount * sizeof(std::int32_t),
          "top-k stride is too small");
  Require(args.warmup >= 0 && args.repeat > 0 && args.kernel_cap_us > 0.0,
          "warmup, repeat, or cap is invalid");
  Require(args.m_tile == 16 || args.m_tile == 24 || args.m_tile == 32 ||
              args.m_tile == 48 || args.m_tile == 64,
          "--m-tile must be 16, 24, 32, 48, or 64");
  Require(!args.hybrid_shared64 || args.m_tile == 32,
          "--hybrid-shared64 requires --m-tile 32");
  Require(args.workgroup_subgroups == 1 || args.workgroup_subgroups == 4,
          "--workgroup-subgroups must be 1 or 4");
  Require(!args.flatten_output_tasks || args.exact_per16 ||
              ((args.m_tile == 16 || args.m_tile == 24 ||
                args.m_tile == 32) &&
               args.f16_weight_scales &&
               !args.hybrid_shared64 && args.workgroup_subgroups == 1),
          "flattened tasks require plain M16/M24/M32 with F16 scales");
  Require(!args.exact_per16 ||
              (args.m_tile == 16 && args.flatten_output_tasks &&
               !args.hybrid_shared64 && args.workgroup_subgroups == 1 &&
               !args.f16_weight_scales),
          "exact per-16 requires flat M16 with F32 weight scales");
  Require(!args.exact_block_accum || args.exact_per16,
          "exact block accumulation requires exact per-16");
  return args;
}

std::string DeviceString(cl_device_id device, cl_device_info key) {
  std::size_t size = 0;
  Check(clGetDeviceInfo(device, key, 0, nullptr, &size),
        "clGetDeviceInfo size");
  std::string value(size, '\0');
  Check(clGetDeviceInfo(device, key, size, value.data(), nullptr),
        "clGetDeviceInfo value");
  while (!value.empty() && value.back() == '\0') value.pop_back();
  return value;
}

cl_device_id SelectGpu() {
  cl_uint platform_count = 0;
  Check(clGetPlatformIDs(0, nullptr, &platform_count),
        "clGetPlatformIDs count");
  std::vector<cl_platform_id> platforms(platform_count);
  Check(clGetPlatformIDs(platform_count, platforms.data(), nullptr),
        "clGetPlatformIDs list");
  for (cl_platform_id platform : platforms) {
    cl_uint device_count = 0;
    if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr,
                       &device_count) != CL_SUCCESS || device_count == 0) {
      continue;
    }
    std::vector<cl_device_id> devices(device_count);
    Check(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, device_count,
                         devices.data(), nullptr),
          "clGetDeviceIDs list");
    for (cl_device_id device : devices) {
      if (DeviceString(device, CL_DEVICE_NAME).find("B390") !=
          std::string::npos) {
        return device;
      }
    }
  }
  Fail("Arc B390 GPU was not found");
}

std::string ProgramLog(cl_program program, cl_device_id device) {
  std::size_t size = 0;
  clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr,
                        &size);
  std::string log(size, '\0');
  if (size != 0) {
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, size,
                          log.data(), nullptr);
  }
  while (!log.empty() && log.back() == '\0') log.pop_back();
  return log;
}

template <typename Value>
cl_mem CreateCopied(cl_context context, cl_mem_flags flags,
                    std::vector<Value>& values, const char* label) {
  cl_int status = CL_SUCCESS;
  cl_mem buffer = clCreateBuffer(
      context, flags | CL_MEM_COPY_HOST_PTR,
      values.size() * sizeof(Value), values.data(), &status);
  Check(status, label);
  return buffer;
}

bool MapsAreNativeOnly() {
  std::ifstream maps("/proc/self/maps");
  std::string line;
  while (std::getline(maps, line)) {
    std::transform(line.begin(), line.end(), line.begin(),
                   [](unsigned char value) { return std::tolower(value); });
    if (line.find("libdnnl") != std::string::npos ||
        line.find("openvino") != std::string::npos) {
      return false;
    }
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args args = ParseArgs(argc, argv);
    const auto index = iq36::parse_gguf_model_index(args.model);
    const auto* tensor = iq36::find_tensor(index, args.tensor);
    Require(tensor != nullptr && tensor->type == 14 &&
                tensor->dims == std::vector<std::uint64_t>(
                    {kInputSize, kOutputSize, kExpertCount}),
            "selected tensor is not the locked Q6_K expert down matrix");
    auto raw = ReadTensor(args.model, *tensor);
    auto prepacked = args.exact_per16
        ? PrepackQ6ExactPer16(raw)
        : PrepackQ6Surrogate(raw);
    raw.clear();
    raw.shrink_to_fit();
    if (args.prepack_only) {
      if (args.exact_per16) {
        const std::filesystem::path dump_dir(args.dump_prepacked_dir);
        std::filesystem::create_directories(dump_dir);
        WriteVector(dump_dir / "q6-down-exact-per16-values-u8.bin",
                    prepacked.values);
        if (args.exact_block_accum) {
          WriteVector(dump_dir / "q6-down-exact-block-scales-i8.bin",
                      prepacked.integer_scales);
          WriteVector(dump_dir / "q6-down-exact-block-d-f32.bin",
                      prepacked.block_scales);
        } else {
          WriteVector(dump_dir / "q6-down-exact-per16-scales-f32.bin",
                      prepacked.scales);
        }
        std::cout << std::boolalpha << std::setprecision(12) << "{";
        std::cout << "\"exact_per16\":true,";
        std::cout << "\"exact_block_accum\":"
                  << args.exact_block_accum << ",";
        std::cout << "\"max_abs_weight_error\":"
                  << prepacked.maximum_absolute_error << ",";
        std::cout << "\"prepack_only\":true,";
        std::cout << "\"relative_l2_weight_error\":"
                  << prepacked.relative_l2_error << ",";
        std::cout << "\"resident_bytes\":"
                  << prepacked.values.size() +
                         (args.exact_block_accum
                              ? prepacked.integer_scales.size() +
                                    prepacked.block_scales.size() *
                                        sizeof(float)
                              : prepacked.scales.size() * sizeof(float))
                  << ",";
        std::cout << "\"tensor\":\"" << args.tensor << "\"}"
                  << std::endl;
        return 0;
      }
      std::vector<std::uint16_t> scales(prepacked.scales.size());
#pragma omp parallel for schedule(static)
      for (std::int64_t scale = 0;
           scale < static_cast<std::int64_t>(prepacked.scales.size());
           ++scale) {
        scales[static_cast<std::size_t>(scale)] = FloatToHalf(
            prepacked.scales[static_cast<std::size_t>(scale)]);
      }
      const std::filesystem::path dump_dir(args.dump_prepacked_dir);
      std::filesystem::create_directories(dump_dir);
      WriteVector(dump_dir / "q6-down-weights-u8.bin", prepacked.values);
      WriteVector(dump_dir / "q6-down-scales-f16.bin", scales);
      std::cout << std::boolalpha << std::setprecision(12) << "{";
      std::cout << "\"max_abs_weight_error\":"
                << prepacked.maximum_absolute_error << ",";
      std::cout << "\"prepack_only\":true,";
      std::cout << "\"relative_l2_weight_error\":"
                << prepacked.relative_l2_error << ",";
      std::cout << "\"resident_bytes\":"
                << prepacked.values.size() +
                       scales.size() * sizeof(std::uint16_t) << ",";
      std::cout << "\"tensor\":\"" << args.tensor << "\"}"
                << std::endl;
      return 0;
    }
    auto swiglu = ReadVector<float>(
        args.swiglu, kAssignmentCount * kInputSize);
    if (args.round_swiglu_f16) {
#pragma omp parallel for schedule(static)
      for (std::int64_t index = 0;
           index < static_cast<std::int64_t>(swiglu.size()); ++index) {
        swiglu[static_cast<std::size_t>(index)] = HalfToFloat(
            FloatToHalf(swiglu[static_cast<std::size_t>(index)]));
      }
    }
    const auto topk = ReadBytes(args.topk);
    const auto router = ReadVector<float>(
        args.router_weights, kAssignmentCount);
    auto grouped = MakeGroupedInput(
        swiglu, topk, args.topk_stride, router, args.m_tile,
        args.hybrid_shared64, args.exact_per16);
    const auto oracle = ReadVector<float>(
        args.oracle, kAssignmentCount * kOutputSize);

    cl_device_id device = SelectGpu();
    cl_int status = CL_SUCCESS;
    cl_context context = clCreateContext(
        nullptr, 1, &device, nullptr, nullptr, &status);
    Check(status, "clCreateContext");
    cl_command_queue queue = clCreateCommandQueueWithProperties(
        context, device,
        (const cl_queue_properties[]){CL_QUEUE_PROPERTIES,
                                      CL_QUEUE_PROFILING_ENABLE, 0},
        &status);
    Check(status, "clCreateCommandQueueWithProperties");
    const std::string source = ReadText(args.kernel);
    const char* source_pointer = source.data();
    const std::size_t source_size = source.size();
    cl_program program = clCreateProgramWithSource(
        context, 1, &source_pointer, &source_size, &status);
    Check(status, "clCreateProgramWithSource");
    status = clBuildProgram(
        program, 1, &device, "-cl-std=CL3.0", nullptr, nullptr);
    const std::string build_log = ProgramLog(program, device);
    if (status != CL_SUCCESS) Fail("kernel build failed: " + build_log);
    const char* kernel_name =
        args.exact_per16
            ? (args.exact_block_accum
                   ? "iq36_grouped_s8_u8_q6_exact_block_down_m16_compact_f32"
                   : "iq36_grouped_s8_u8_q6_exact_down_m16_compact")
            : (args.f16_weight_scales
            ? (args.flatten_output_tasks
                   ? (args.m_tile == 16
                          ? "iq36_grouped_s8_u8_q6_surrogate_down_m16_f16scale_compact"
                          : (args.m_tile == 24
                                 ? "iq36_grouped_s8_u8_q6_surrogate_down_m24_f16scale_table"
                                 : "iq36_grouped_s8_u8_q6_surrogate_down_m32_f16scale_table"))
                   : (args.workgroup_subgroups == 4
                   ? "iq36_grouped_s8_u8_q6_surrogate_down_m32_f16scale_wg64"
                   : "iq36_grouped_s8_u8_q6_surrogate_down_m32_f16scale"))
            : (args.m_tile == 64
            ? "iq36_grouped_s8_u8_q6_surrogate_down_m64"
            : (args.m_tile == 48
                   ? "iq36_grouped_s8_u8_q6_surrogate_down_m48"
                   : "iq36_grouped_s8_u8_q6_surrogate_down_m32")));
    Require(!args.f16_weight_scales ||
                ((args.m_tile == 16 || args.m_tile == 24 ||
                  args.m_tile == 32) &&
                 !args.hybrid_shared64),
            "F16 weight scales currently require plain M16/M24/M32");
    Require(args.workgroup_subgroups == 1 ||
                (args.f16_weight_scales && args.m_tile == 32 &&
                 !args.hybrid_shared64),
            "WG64 currently requires plain M32 with F16 scales");
    cl_kernel kernel = clCreateKernel(program, kernel_name, &status);
    Check(status, "clCreateKernel");
    cl_kernel shared64_kernel = nullptr;
    if (args.hybrid_shared64) {
      shared64_kernel = clCreateKernel(
          program, "iq36_grouped_s8_u8_q6_surrogate_down_m64_shared",
          &status);
      Check(status, "clCreateKernel shared64");
    }

    cl_mem weights_buffer = CreateCopied(
        context, CL_MEM_READ_ONLY, prepacked.values, "create weights");
    std::vector<std::uint16_t> f16_weight_scales;
    cl_mem weight_scales_buffer = nullptr;
    cl_mem weight_block_scales_buffer = nullptr;
    if (args.exact_block_accum) {
      weight_scales_buffer = CreateCopied(
          context, CL_MEM_READ_ONLY, prepacked.integer_scales,
          "create integer weight scales");
      weight_block_scales_buffer = CreateCopied(
          context, CL_MEM_READ_ONLY, prepacked.block_scales,
          "create block weight scales");
    } else if (args.f16_weight_scales) {
      f16_weight_scales.resize(prepacked.scales.size());
#pragma omp parallel for schedule(static)
      for (std::int64_t index = 0;
           index < static_cast<std::int64_t>(prepacked.scales.size());
           ++index) {
        f16_weight_scales[static_cast<std::size_t>(index)] = FloatToHalf(
            prepacked.scales[static_cast<std::size_t>(index)]);
      }
      if (!args.dump_prepacked_dir.empty()) {
        const std::filesystem::path dump_dir(args.dump_prepacked_dir);
        std::filesystem::create_directories(dump_dir);
        WriteVector(dump_dir / "q6-down-weights-u8.bin", prepacked.values);
        WriteVector(dump_dir / "q6-down-scales-f16.bin", f16_weight_scales);
      }
      weight_scales_buffer = CreateCopied(
          context, CL_MEM_READ_ONLY, f16_weight_scales,
          "create F16 weight scales");
    } else {
      weight_scales_buffer = CreateCopied(
          context, CL_MEM_READ_ONLY, prepacked.scales,
          "create F32 weight scales");
    }
    cl_mem tiles_buffer = CreateCopied(
        context, CL_MEM_READ_ONLY, grouped.work_tiles, "create work tiles");
    cl_mem expert_offsets_buffer = nullptr;
    cl_mem flat_work_tiles_buffer = nullptr;
    cl_mem task_coordinates_buffer = nullptr;
    if (args.flatten_output_tasks) {
      expert_offsets_buffer = CreateCopied(
          context, CL_MEM_READ_ONLY, grouped.expert_offsets,
          "create expert offsets");
      if (args.m_tile == 16) {
        task_coordinates_buffer = CreateCopied(
            context, CL_MEM_READ_ONLY, grouped.task_coordinates,
            "create task coordinates");
      } else {
        flat_work_tiles_buffer = CreateCopied(
            context, CL_MEM_READ_ONLY, grouped.flat_work_tiles,
            "create flat work tiles");
      }
    }
    cl_mem shared64_tiles_buffer = nullptr;
    if (args.hybrid_shared64) {
      shared64_tiles_buffer = CreateCopied(
          context, CL_MEM_READ_ONLY, grouped.shared64_tiles,
          "create shared64 work tiles");
    }
    cl_mem source_buffer = CreateCopied(
        context, CL_MEM_READ_ONLY, grouped.q8, "create source");
    cl_mem source_scales_buffer = CreateCopied(
        context, CL_MEM_READ_ONLY, grouped.scales, "create source scales");
    cl_mem source_sums_buffer = CreateCopied(
        context, CL_MEM_READ_ONLY, grouped.sums, "create source sums");
    cl_mem router_buffer = CreateCopied(
        context, CL_MEM_READ_ONLY, grouped.router_weights, "create router");
    cl_mem output_buffer = clCreateBuffer(
        context, CL_MEM_WRITE_ONLY,
        kAssignmentCount * kOutputSize *
            (args.exact_block_accum ? sizeof(float) : sizeof(std::uint16_t)),
        nullptr, &status);
    Check(status, "create output");
    if (args.flatten_output_tasks) {
      if (args.m_tile == 16) {
        if (args.exact_block_accum) {
          const std::array<cl_mem, 10> arguments = {
              weights_buffer, weight_scales_buffer,
              weight_block_scales_buffer, expert_offsets_buffer,
              task_coordinates_buffer, source_buffer, source_scales_buffer,
              source_sums_buffer, router_buffer, output_buffer};
          for (cl_uint argument = 0; argument < arguments.size(); ++argument) {
            Check(clSetKernelArg(kernel, argument, sizeof(cl_mem),
                                 &arguments[argument]),
                  "clSetKernelArg exact block compact");
          }
        } else {
          const std::array<cl_mem, 9> arguments = {
              weights_buffer, weight_scales_buffer, expert_offsets_buffer,
              task_coordinates_buffer, source_buffer, source_scales_buffer,
              source_sums_buffer, router_buffer, output_buffer};
          for (cl_uint argument = 0; argument < arguments.size(); ++argument) {
            Check(clSetKernelArg(kernel, argument, sizeof(cl_mem),
                                 &arguments[argument]),
                  "clSetKernelArg compact");
          }
        }
      } else {
        const std::array<cl_mem, 8> arguments = {
            weights_buffer, weight_scales_buffer, flat_work_tiles_buffer,
            source_buffer, source_scales_buffer, source_sums_buffer,
            router_buffer, output_buffer};
        for (cl_uint argument = 0; argument < arguments.size(); ++argument) {
          Check(clSetKernelArg(kernel, argument, sizeof(cl_mem),
                               &arguments[argument]),
                "clSetKernelArg table");
        }
      }
    } else {
      const std::array<cl_mem, 8> arguments = {
          weights_buffer, weight_scales_buffer, tiles_buffer, source_buffer,
          source_scales_buffer, source_sums_buffer, router_buffer,
          output_buffer};
      for (cl_uint argument = 0; argument < arguments.size(); ++argument) {
        Check(clSetKernelArg(kernel, argument, sizeof(cl_mem),
                             &arguments[argument]),
              "clSetKernelArg");
      }
    }
    if (args.hybrid_shared64) {
      const std::array<cl_mem, 8> shared_arguments = {
          weights_buffer, weight_scales_buffer, shared64_tiles_buffer,
          source_buffer, source_scales_buffer, source_sums_buffer,
          router_buffer, output_buffer};
      for (cl_uint argument = 0; argument < shared_arguments.size();
           ++argument) {
        Check(clSetKernelArg(shared64_kernel, argument, sizeof(cl_mem),
                             &shared_arguments[argument]),
              "clSetKernelArg shared64");
      }
    }
    const std::array<std::size_t, 2> local = {
        16 * args.workgroup_subgroups, 1};
    const std::array<std::size_t, 2> global = {
        args.flatten_output_tasks
            ? grouped.flat_task_count * local[0]
            : kOutputTiles * local[0],
        args.flatten_output_tasks ? 1 : grouped.work_tiles.size()};
    const std::array<std::size_t, 2> shared_local = {32, 1};
    const std::array<std::size_t, 2> shared_global = {
        kOutputTiles * shared_local[0], grouped.shared64_tiles.size()};
    auto enqueue = [&](cl_event* first_event, cl_event* last_event) {
      if (args.hybrid_shared64) {
        Check(clEnqueueNDRangeKernel(
                  queue, shared64_kernel, 2, nullptr, shared_global.data(),
                  shared_local.data(), 0, nullptr, first_event),
              "clEnqueueNDRangeKernel shared64");
        Check(clEnqueueNDRangeKernel(queue, kernel, 2, nullptr,
                                     global.data(), local.data(), 0, nullptr,
                                     last_event),
              "clEnqueueNDRangeKernel tail32");
      } else {
        if (args.flatten_output_tasks) {
          Check(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr,
                                       global.data(), local.data(), 0,
                                       nullptr, first_event),
                "clEnqueueNDRangeKernel flat");
        } else {
          Check(clEnqueueNDRangeKernel(queue, kernel, 2, nullptr,
                                       global.data(), local.data(), 0,
                                       nullptr, first_event),
                "clEnqueueNDRangeKernel");
        }
      }
    };
    for (int iteration = 0; iteration < args.warmup; ++iteration) {
      enqueue(nullptr, nullptr);
      Check(clFinish(queue), "warmup finish");
    }
    std::vector<double> samples_us;
    samples_us.reserve(args.repeat);
    for (int iteration = 0; iteration < args.repeat; ++iteration) {
      cl_event first_event = nullptr;
      cl_event last_event = nullptr;
      enqueue(&first_event,
              args.hybrid_shared64 ? &last_event : nullptr);
      Check(clFinish(queue), "timed finish");
      cl_ulong begin = 0;
      cl_ulong end = 0;
      Check(clGetEventProfilingInfo(first_event, CL_PROFILING_COMMAND_START,
                                    sizeof(begin), &begin, nullptr),
            "profile start");
      Check(clGetEventProfilingInfo(
                args.hybrid_shared64 ? last_event : first_event,
                CL_PROFILING_COMMAND_END,
                                    sizeof(end), &end, nullptr),
            "profile end");
      samples_us.push_back(static_cast<double>(end - begin) / 1000.0);
      clReleaseEvent(first_event);
      if (last_event != nullptr) clReleaseEvent(last_event);
    }
    std::vector<std::uint16_t> output(
        args.exact_block_accum ? 0 : kAssignmentCount * kOutputSize);
    std::vector<float> output_f32(
        args.exact_block_accum ? kAssignmentCount * kOutputSize : 0);
    void* output_data = args.exact_block_accum
        ? static_cast<void*>(output_f32.data())
        : static_cast<void*>(output.data());
    const std::size_t output_bytes = args.exact_block_accum
        ? output_f32.size() * sizeof(float)
        : output.size() * sizeof(std::uint16_t);
    Check(clEnqueueReadBuffer(queue, output_buffer, CL_TRUE, 0,
                              output_bytes, output_data, 0, nullptr, nullptr),
          "read output");

    double maximum_absolute_difference = 0.0;
    double error_squared = 0.0;
    double candidate_squared = 0.0;
    double reference_squared = 0.0;
    double dot = 0.0;
    std::uint64_t mismatch_count = 0;
    bool finite = true;
#pragma omp parallel for schedule(static) reduction(max : maximum_absolute_difference) \
    reduction(+ : error_squared, candidate_squared, reference_squared, dot, mismatch_count) \
    reduction(&& : finite)
    for (std::int64_t source_signed = 0;
         source_signed < static_cast<std::int64_t>(kAssignmentCount);
         ++source_signed) {
      const std::size_t source_index =
          static_cast<std::size_t>(source_signed);
      const std::size_t grouped_row = static_cast<std::size_t>(
          grouped.inverse_map[source_index]);
      const float router_weight = router[source_index];
      for (std::size_t hidden = 0; hidden < kOutputSize; ++hidden) {
        const std::size_t output_index = grouped_row * kOutputSize + hidden;
        const float candidate = args.exact_block_accum
            ? output_f32[output_index] : HalfToFloat(output[output_index]);
        const float reference =
            oracle[source_index * kOutputSize + hidden] * router_weight;
        const double difference =
            static_cast<double>(candidate) - reference;
        maximum_absolute_difference = std::max(
            maximum_absolute_difference, std::abs(difference));
        mismatch_count += std::abs(difference) > 5e-3;
        finite = finite && std::isfinite(candidate) && std::isfinite(reference);
        error_squared += difference * difference;
        candidate_squared += static_cast<double>(candidate) * candidate;
        reference_squared += static_cast<double>(reference) * reference;
        dot += static_cast<double>(candidate) * reference;
      }
    }
    std::sort(samples_us.begin(), samples_us.end());
    const double minimum_us = samples_us.front();
    const double median_us = samples_us[samples_us.size() / 2];
    const double relative_l2 = std::sqrt(error_squared / reference_squared);
    const double cosine = dot /
        std::sqrt(candidate_squared * reference_squared);
    const double rmse = std::sqrt(
        error_squared /
            static_cast<double>(kAssignmentCount * kOutputSize));
    const bool correctness_pass = finite && cosine >= 0.999 &&
        relative_l2 <= 0.002;
    const bool performance_pass = minimum_us <= args.kernel_cap_us;
    const std::uint64_t scale_bytes = args.exact_block_accum
        ? sizeof(std::int8_t)
        : (args.f16_weight_scales ? sizeof(std::uint16_t) : sizeof(float));
    const std::uint64_t scale_groups =
        args.exact_per16 ? kExactGroupsPerRow : kGroupsPerRow;
    const std::uint64_t expanded_bytes_per_expert =
        kOutputSize * kInputSize +
        kOutputSize * scale_groups * scale_bytes +
        (args.exact_block_accum ? kOutputSize * 2 * sizeof(float) : 0);
    const std::uint64_t raw_bytes_per_expert =
        kOutputSize * 2 * kQ6BlockBytes;
    const std::uint64_t active_expanded_bytes =
        grouped.active_experts * expanded_bytes_per_expert;
    const std::uint64_t active_raw_bytes =
        grouped.active_experts * raw_bytes_per_expert;

    std::cout << std::boolalpha << std::setprecision(12) << "{";
    std::cout << "\"active_experts\":" << grouped.active_experts << ",";
    std::cout << "\"active_expanded_weight_bytes\":"
              << active_expanded_bytes << ",";
    std::cout << "\"active_raw_q6_bytes\":" << active_raw_bytes << ",";
    std::cout << "\"build_log\":\"" << JsonEscape(build_log) << "\",";
    std::cout << "\"comparison\":{";
    std::cout << "\"compared_value_count\":"
              << kAssignmentCount * kOutputSize << ",";
    std::cout << "\"cosine\":" << cosine << ",";
    std::cout << "\"finite\":" << finite << ",";
    std::cout << "\"max_abs_diff\":" << maximum_absolute_difference << ",";
    std::cout << "\"mismatch_count\":" << mismatch_count << ",";
    std::cout << "\"relative_l2\":" << relative_l2 << ",";
    std::cout << "\"rmse\":" << rmse << "},";
    std::cout << "\"correctness_pass\":" << correctness_pass << ",";
    std::cout << "\"device_name\":\"" << DeviceString(device, CL_DEVICE_NAME)
              << "\",";
    std::cout << "\"driver_version\":\""
              << DeviceString(device, CL_DRIVER_VERSION) << "\",";
    std::cout << "\"expanded_effective_gb_s\":"
              << static_cast<double>(active_expanded_bytes) / minimum_us /
                     1000.0
              << ",";
    std::cout << "\"kernel_cap_us\":" << args.kernel_cap_us << ",";
    std::cout << "\"f16_weight_scales\":" << args.f16_weight_scales << ",";
    std::cout << "\"exact_per16\":" << args.exact_per16 << ",";
    std::cout << "\"exact_block_accum\":" << args.exact_block_accum << ",";
    std::cout << "\"flatten_output_tasks\":"
              << args.flatten_output_tasks << ",";
    std::cout << "\"kernel_median_us\":" << median_us << ",";
    std::cout << "\"kernel_min_us\":" << minimum_us << ",";
    std::cout << "\"hybrid_shared64\":" << args.hybrid_shared64 << ",";
    std::cout << "\"maps_native_only\":" << MapsAreNativeOnly() << ",";
    std::cout << "\"max_group_size\":" << grouped.maximum_group << ",";
    std::cout << "\"m_tile\":" << args.m_tile << ",";
    std::cout << "\"performance_pass\":" << performance_pass << ",";
    std::cout << "\"prepack\":{";
    std::cout << "\"max_abs_weight_error\":"
              << prepacked.maximum_absolute_error << ",";
    std::cout << "\"relative_l2_weight_error\":"
              << prepacked.relative_l2_error << ",";
    std::cout << "\"resident_bytes\":"
              << prepacked.values.size() +
                     (args.exact_block_accum
                          ? prepacked.integer_scales.size() +
                                prepacked.block_scales.size() * sizeof(float)
                          : prepacked.scales.size() * scale_bytes)
              << "},";
    std::cout << "\"raw_q6_effective_gb_s\":"
              << static_cast<double>(active_raw_bytes) / minimum_us / 1000.0
              << ",";
    std::cout << "\"round_swiglu_f16\":" << args.round_swiglu_f16 << ",";
    std::cout << "\"samples_us\":[";
    for (std::size_t index = 0; index < samples_us.size(); ++index) {
      if (index != 0) std::cout << ",";
      std::cout << samples_us[index];
    }
    std::cout << "],";
    std::cout << "\"schema_version\":"
                 "\"iq36-grouped-s8-u8-q6-surrogate-down-v1\",";
    std::cout << "\"shared64_work_tile_count\":"
              << grouped.shared64_tiles.size() << ",";
    std::cout << "\"work_tile_count\":"
              << (args.flatten_output_tasks
                      ? grouped.flat_task_count
                      : grouped.work_tiles.size() +
                            grouped.shared64_tiles.size());
    std::cout << ",\"workgroup_subgroups\":"
              << args.workgroup_subgroups;
    std::cout << "}" << std::endl;

    clReleaseMemObject(output_buffer);
    clReleaseMemObject(router_buffer);
    clReleaseMemObject(source_sums_buffer);
    clReleaseMemObject(source_scales_buffer);
    clReleaseMemObject(source_buffer);
    clReleaseMemObject(tiles_buffer);
    if (flat_work_tiles_buffer != nullptr) {
      clReleaseMemObject(flat_work_tiles_buffer);
    }
    if (task_coordinates_buffer != nullptr) {
      clReleaseMemObject(task_coordinates_buffer);
    }
    if (expert_offsets_buffer != nullptr) {
      clReleaseMemObject(expert_offsets_buffer);
    }
    if (shared64_tiles_buffer != nullptr) {
      clReleaseMemObject(shared64_tiles_buffer);
    }
    clReleaseMemObject(weight_scales_buffer);
    if (weight_block_scales_buffer != nullptr) {
      clReleaseMemObject(weight_block_scales_buffer);
    }
    clReleaseMemObject(weights_buffer);
    clReleaseKernel(kernel);
    if (shared64_kernel != nullptr) clReleaseKernel(shared64_kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return correctness_pass && performance_pass ? 0 : 2;
  } catch (const std::exception& exception) {
    std::cerr << "grouped-s8-u8-q6-surrogate-down: " << exception.what()
              << '\n';
    return 4;
  }
}
