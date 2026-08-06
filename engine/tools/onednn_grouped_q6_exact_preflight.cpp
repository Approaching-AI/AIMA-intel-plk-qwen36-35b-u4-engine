#include "intel_qwen36/gguf_loader.hpp"

#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_ocl.hpp>

#include <CL/cl.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

constexpr std::size_t kExperts = 256;
constexpr std::size_t kTokens = 1024;
constexpr std::size_t kSelected = 8;
constexpr std::size_t kRows = kTokens * kSelected;
constexpr std::size_t kInput = 512;
constexpr std::size_t kOutput = 2048;
constexpr std::size_t kBlockValues = 256;
constexpr std::size_t kBlockBytes = 210;
constexpr std::size_t kExactGroupValues = 16;

[[noreturn]] void Fail(const std::string& message) {
  throw std::runtime_error(message);
}

void Require(bool condition, const std::string& message) {
  if (!condition) Fail(message);
}

std::vector<std::uint8_t> ReadBytes(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  Require(static_cast<bool>(input), "could not open " + path);
  input.seekg(0, std::ios::end);
  const auto size = input.tellg();
  Require(size >= 0, "could not size " + path);
  input.seekg(0, std::ios::beg);
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
  input.read(reinterpret_cast<char*>(bytes.data()),
             static_cast<std::streamsize>(bytes.size()));
  Require(static_cast<bool>(input), "could not read " + path);
  return bytes;
}

template <typename Value>
std::vector<Value> ReadVector(const std::string& path, std::size_t count) {
  const auto bytes = ReadBytes(path);
  Require(bytes.size() == count * sizeof(Value), "input size mismatch");
  std::vector<Value> values(count);
  std::memcpy(values.data(), bytes.data(), bytes.size());
  return values;
}

std::vector<std::uint8_t> ReadTensor(
    const std::string& model, const iq36::GgufTensorInfo& tensor) {
  std::ifstream input(model, std::ios::binary);
  Require(static_cast<bool>(input), "could not open model");
  input.seekg(static_cast<std::streamoff>(tensor.absolute_offset));
  Require(static_cast<bool>(input), "could not seek model tensor");
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(tensor.nbytes));
  input.read(reinterpret_cast<char*>(bytes.data()),
             static_cast<std::streamsize>(bytes.size()));
  Require(input.gcount() == static_cast<std::streamsize>(bytes.size()),
          "could not read model tensor");
  return bytes;
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
    high_bits = high & 3;
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

template <typename Value>
void WriteMemory(const std::vector<Value>& values, dnnl::memory& memory,
                 int index = 0) {
  Require(memory.get_desc().get_size(index) ==
              values.size() * sizeof(Value),
          "oneDNN memory size mismatch");
  Value* mapped = memory.map_data<Value>(index);
  Require(mapped != nullptr, "oneDNN map returned null");
  std::copy(values.begin(), values.end(), mapped);
  memory.unmap_data(mapped, index);
}

void WriteOffsets(dnnl::memory& memory,
                  const std::vector<std::int32_t>& offsets) {
  WriteMemory(offsets, memory, 1);
}

void CheckCl(cl_int status, const std::string& operation) {
  if (status != CL_SUCCESS) {
    Fail(operation + " failed with OpenCL status " + std::to_string(status));
  }
}

std::string ProgramBuildLog(cl_program program, cl_device_id device) {
  std::size_t bytes = 0;
  clGetProgramBuildInfo(
      program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &bytes);
  std::string result(bytes, '\0');
  if (bytes != 0) {
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                          bytes, result.data(), nullptr);
  }
  return result;
}

void SetS4(std::vector<std::uint8_t>& packed, std::size_t logical_index,
           int value) {
  Require(value >= -7 && value <= 7, "S4 residual code out of range");
  const std::uint8_t code = static_cast<std::uint8_t>(value) & 15U;
  std::uint8_t& byte = packed[logical_index >> 1];
  if ((logical_index & 1U) == 0) {
    byte = static_cast<std::uint8_t>((byte & 0xf0U) | code);
  } else {
    byte = static_cast<std::uint8_t>((byte & 0x0fU) | (code << 4));
  }
}

int GetS4(const std::vector<std::uint8_t>& packed,
          std::size_t logical_index) {
  const std::uint8_t byte = packed[logical_index >> 1];
  int value = (logical_index & 1U) == 0 ? byte & 15U : byte >> 4;
  if (value >= 8) value -= 16;
  return value;
}

struct Schedule {
  std::vector<std::int32_t> offsets;
  std::vector<std::int32_t> inverse;
  std::vector<std::int32_t> expert_ids;
  std::size_t active_experts = 0;
  std::size_t max_group = 0;
};

Schedule MakeSchedule(const std::vector<std::uint8_t>& topk,
                      std::size_t stride) {
  Require(topk.size() >= (kTokens - 1) * stride + kSelected * 4,
          "top-k payload is truncated");
  std::array<std::int32_t, kExperts> counts{};
  std::array<std::int32_t, kRows> source_experts{};
  for (std::size_t token = 0; token < kTokens; ++token) {
    for (std::size_t rank = 0; rank < kSelected; ++rank) {
      std::int32_t expert = -1;
      std::memcpy(&expert, topk.data() + token * stride + rank * 4, 4);
      Require(expert >= 0 && expert < static_cast<std::int32_t>(kExperts),
              "top-k expert out of range");
      source_experts[token * kSelected + rank] = expert;
      ++counts[static_cast<std::size_t>(expert)];
    }
  }
  Schedule result;
  result.offsets.resize(kExperts);
  result.inverse.resize(kRows, -1);
  result.expert_ids.resize(kRows, -1);
  std::array<std::int32_t, kExperts> cursors{};
  std::int32_t total = 0;
  for (std::size_t expert = 0; expert < kExperts; ++expert) {
    cursors[expert] = total;
    total += counts[expert];
    result.offsets[expert] = total;
    result.active_experts += counts[expert] != 0;
    result.max_group = std::max(
        result.max_group, static_cast<std::size_t>(counts[expert]));
  }
  Require(total == static_cast<std::int32_t>(kRows),
          "assignment count mismatch");
  for (std::size_t source = 0; source < kRows; ++source) {
    const std::size_t expert =
        static_cast<std::size_t>(source_experts[source]);
    const std::size_t grouped =
        static_cast<std::size_t>(cursors[expert]++);
    result.inverse[source] = static_cast<std::int32_t>(grouped);
    result.expert_ids[grouped] = static_cast<std::int32_t>(expert);
  }
  return result;
}

struct Args {
  std::string model;
  std::string tensor = "blk.39.ffn_down_exps.weight";
  std::string swiglu;
  std::string topk;
  std::size_t topk_stride = 1024;
  std::string oracle;
  std::string encoding = "s8";
  std::string representation = "exact-per16";
  int warmup = 3;
  int repeat = 7;
  double cap_us = 4316.404;
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
    else if (key == "--swiglu") args.swiglu = value();
    else if (key == "--topk") args.topk = value();
    else if (key == "--topk-stride") args.topk_stride = std::stoull(value());
    else if (key == "--oracle") args.oracle = value();
    else if (key == "--encoding") args.encoding = value();
    else if (key == "--representation") args.representation = value();
    else if (key == "--warmup") args.warmup = std::stoi(value());
    else if (key == "--repeat") args.repeat = std::stoi(value());
    else if (key == "--cap-us") args.cap_us = std::stod(value());
    else Fail("unknown argument: " + key);
  }
  Require(!args.model.empty() && !args.swiglu.empty() &&
              !args.topk.empty() && !args.oracle.empty(),
          "model, swiglu, topk, and oracle are required");
  Require(args.warmup >= 0 && args.repeat > 0 && args.cap_us > 0,
          "invalid warmup/repeat/cap");
  Require(args.encoding == "s8" || args.encoding == "u8-zp32" ||
              args.encoding == "u8-affine",
          "unsupported encoding");
  Require(args.representation == "exact-per16" ||
              args.representation == "requant-s8-per32" ||
              args.representation == "requant-s8-per32-s4-residual" ||
              args.representation ==
                  "requant-s8-per32-ternary-residual" ||
              args.representation == "requant-u8-affine-per32" ||
              args.representation ==
                  "requant-s8-affine-per32-external-zp" ||
              args.representation == "requant-s8-per32-activation-lsq" ||
              args.representation ==
                  "requant-s8-per32-full-gram-lsq",
          "unsupported representation");
  Require((args.representation == "requant-u8-affine-per32") ==
              (args.encoding == "u8-affine"),
          "affine representation and encoding must be selected together");
  Require(args.representation == "exact-per16" ||
              args.representation == "requant-u8-affine-per32" ||
              args.encoding == "s8",
          "S8 requant representations require s8 encoding");
  return args;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args args = ParseArgs(argc, argv);
    const auto index = iq36::parse_gguf_model_index(args.model);
    const auto* tensor = iq36::find_tensor(index, args.tensor);
    Require(tensor != nullptr && tensor->type == 14 &&
                tensor->dims == std::vector<std::uint64_t>(
                    {kInput, kOutput, kExperts}),
            "tensor is not the expected Q6_K expert down matrix");
    const auto raw = ReadTensor(args.model, *tensor);
    const auto swiglu = ReadVector<float>(args.swiglu, kRows * kInput);
    const auto topk = ReadBytes(args.topk);
    const auto oracle = ReadVector<float>(args.oracle, kRows * kOutput);
    const Schedule schedule = MakeSchedule(topk, args.topk_stride);

    std::vector<std::int8_t> source(kRows * kInput);
    std::vector<float> source_scales(kRows * 2);
#pragma omp parallel for schedule(static)
    for (std::int64_t source_signed = 0;
         source_signed < static_cast<std::int64_t>(kRows); ++source_signed) {
      const std::size_t source_row =
          static_cast<std::size_t>(source_signed);
      const std::size_t grouped =
          static_cast<std::size_t>(schedule.inverse[source_row]);
      for (std::size_t block = 0; block < 2; ++block) {
        float maximum = 0.0f;
        float signed_maximum = 0.0f;
        for (std::size_t inner = 0; inner < kBlockValues; ++inner) {
          const float value =
              swiglu[source_row * kInput + block * kBlockValues + inner];
          if (std::abs(value) > maximum) {
            maximum = std::abs(value);
            signed_maximum = value;
          }
        }
        const float inverse =
            maximum == 0.0f ? 0.0f : -127.0f / signed_maximum;
        const float scale = inverse == 0.0f ? 0.0f : 1.0f / inverse;
        source_scales[grouped * 2 + block] = scale;
        for (std::size_t inner = 0; inner < kBlockValues; ++inner) {
          const float value =
              swiglu[source_row * kInput + block * kBlockValues + inner];
          const int quantized = inverse == 0.0f
              ? 0 : std::min(127, NearestInt(inverse * value));
          source[grouped * kInput + block * kBlockValues + inner] =
              static_cast<std::int8_t>(quantized);
        }
      }
    }

    const std::size_t weight_group_values =
        args.representation == "exact-per16" ? 16 : 32;
    const std::size_t weight_group_count = kInput / weight_group_values;
    const bool with_s4_residual =
        args.representation == "requant-s8-per32-s4-residual";
    const bool with_ternary_residual =
        args.representation == "requant-s8-per32-ternary-residual";
    const bool affine_u8 =
        args.representation == "requant-u8-affine-per32";
    const bool external_affine =
        args.representation == "requant-s8-affine-per32-external-zp";
    const bool affine_codec = affine_u8 || external_affine;
    const bool activation_lsq =
        args.representation == "requant-s8-per32-activation-lsq";
    const bool full_gram_lsq =
        args.representation == "requant-s8-per32-full-gram-lsq";
    std::vector<double> activation_grams(
        activation_lsq ? 16 * 32 * 32 : 0, 0.0);
    if (activation_lsq) {
#pragma omp parallel
      {
        std::vector<double> local(16 * 32 * 32, 0.0);
#pragma omp for schedule(static)
        for (std::int64_t row_signed = 0;
             row_signed < static_cast<std::int64_t>(kRows); ++row_signed) {
          const std::size_t row = static_cast<std::size_t>(row_signed);
          for (std::size_t group = 0; group < 16; ++group) {
            const double scale = source_scales[row * 2 + group / 8];
            for (std::size_t left = 0; left < 32; ++left) {
              const double x_left =
                  source[row * kInput + group * 32 + left] * scale;
              for (std::size_t right = 0; right < 32; ++right) {
                const double x_right =
                    source[row * kInput + group * 32 + right] * scale;
                local[(group * 32 + left) * 32 + right] +=
                    x_left * x_right;
              }
            }
          }
        }
#pragma omp critical
        for (std::size_t index = 0; index < local.size(); ++index) {
          activation_grams[index] += local[index];
        }
      }
    } else if (full_gram_lsq) {
      activation_grams.assign(kInput * kInput, 0.0);
#pragma omp parallel
      {
        std::vector<double> local(kInput * kInput, 0.0);
#pragma omp for schedule(static)
        for (std::int64_t row_signed = 0;
             row_signed < static_cast<std::int64_t>(kRows); ++row_signed) {
          const std::size_t row = static_cast<std::size_t>(row_signed);
          for (std::size_t left = 0; left < kInput; ++left) {
            const double x_left = source[row * kInput + left] *
                source_scales[row * 2 + left / kBlockValues];
            for (std::size_t right = 0; right < kInput; ++right) {
              const double x_right = source[row * kInput + right] *
                  source_scales[row * 2 + right / kBlockValues];
              local[left * kInput + right] += x_left * x_right;
            }
          }
        }
#pragma omp critical
        for (std::size_t index = 0; index < local.size(); ++index) {
          activation_grams[index] += local[index];
        }
      }
    }
    const bool with_residual =
        with_s4_residual || with_ternary_residual;
    std::vector<std::int8_t> weights(kExperts * kOutput * kInput);
    std::vector<float> weight_scales(
        kExperts * weight_group_count * kOutput);
    std::vector<std::uint8_t> weight_zero_points(
        args.encoding == "s8" && !external_affine
            ? 0 : kExperts * weight_group_count * kOutput, 0);
    std::vector<std::uint8_t> residual_weights(
        with_s4_residual ? kExperts * kOutput * kInput / 2 : 0, 0);
    std::vector<std::int8_t> ternary_weights(
        with_ternary_residual ? kExperts * kOutput * kInput : 0, 0);
    std::vector<float> residual_scales(
        with_residual
            ? kExperts * weight_group_count * kOutput : 0, 0.0f);
    double weight_error_squared = 0.0;
    double weight_reference_squared = 0.0;
    double weight_max_abs = 0.0;
    double corrected_weight_error_squared = 0.0;
    double corrected_weight_max_abs = 0.0;
    std::uint64_t residual_nonzero_count = 0;
    std::uint64_t lsq_changed_code_count = 0;
    std::uint64_t lsq_total_sweeps = 0;
    std::uint64_t lsq_group_count = 0;
    std::uint64_t lsq_max_sweeps = 0;
#pragma omp parallel for schedule(static) reduction(+ : weight_error_squared, weight_reference_squared, corrected_weight_error_squared, residual_nonzero_count, lsq_changed_code_count, lsq_total_sweeps, lsq_group_count) reduction(max : weight_max_abs, corrected_weight_max_abs, lsq_max_sweeps)
    for (std::int64_t expert_signed = 0;
         expert_signed < static_cast<std::int64_t>(kExperts);
         ++expert_signed) {
      const std::size_t expert = static_cast<std::size_t>(expert_signed);
      for (std::size_t output = 0; output < kOutput; ++output) {
        const auto* row = raw.data() +
            (expert * kOutput + output) * 2 * kBlockBytes;
        for (std::size_t group = 0; group < weight_group_count; ++group) {
          std::array<float, 32> reference{};
          float maximum = 0.0f;
          float minimum_value = std::numeric_limits<float>::infinity();
          float maximum_value = -std::numeric_limits<float>::infinity();
          for (std::size_t within = 0;
               within < weight_group_values; ++within) {
            const std::size_t k = group * weight_group_values + within;
            const std::size_t block_index = k / kBlockValues;
            const std::size_t within_block = k % kBlockValues;
            const auto* block = row + block_index * kBlockBytes;
            const float d = HalfToFloat(LoadU16(block + 208));
            const auto* scales =
                reinterpret_cast<const std::int8_t*>(block + 192);
            const float value = d * static_cast<float>(
                scales[within_block / kExactGroupValues]) *
                static_cast<float>(Q6Value(block, within_block));
            reference[within] = value;
            maximum = std::max(maximum, std::abs(value));
            minimum_value = std::min(minimum_value, value);
            maximum_value = std::max(maximum_value, value);
          }
          float stored_scale = maximum == 0.0f
              ? 0.0f : maximum / 127.0f;
          int zero_point = 0;
          if (args.representation == "exact-per16") {
            const std::size_t k = group * weight_group_values;
            const std::size_t block_index = k / kBlockValues;
            const std::size_t within_block = k % kBlockValues;
            const auto* block = row + block_index * kBlockBytes;
            const float d = HalfToFloat(LoadU16(block + 208));
            const auto* scales =
                reinterpret_cast<const std::int8_t*>(block + 192);
            stored_scale = std::abs(
                d * static_cast<float>(
                    scales[within_block / kExactGroupValues]));
            zero_point = args.encoding == "u8-zp32" ? 32 : 0;
          } else if (affine_codec) {
            stored_scale = maximum_value == minimum_value
                ? 0.0f : (maximum_value - minimum_value) / 255.0f;
            zero_point = stored_scale == 0.0f
                ? 0 : NearestInt(-minimum_value / stored_scale);
            zero_point = std::max(0, std::min(255, zero_point));
          }
          const std::size_t coefficient =
              (expert * weight_group_count + group) * kOutput + output;
          weight_scales[coefficient] = stored_scale;
          if (args.encoding != "s8" || external_affine) {
            weight_zero_points[coefficient] =
                static_cast<std::uint8_t>(zero_point);
          }
          std::array<int, 32> quantized_codes{};
          for (std::size_t within = 0;
               within < weight_group_values; ++within) {
            int quantized = stored_scale == 0.0f
                ? 0 : NearestInt(reference[within] / stored_scale);
            if (!affine_codec && args.representation != "exact-per16") {
              quantized = std::max(-127, std::min(127, quantized));
            }
            quantized_codes[within] = quantized;
          }
          if (activation_lsq && stored_scale != 0.0f) {
            const auto initial_codes = quantized_codes;
            std::array<double, 32> error{};
            std::array<double, 32> gradient{};
            for (std::size_t left = 0; left < 32; ++left) {
              error[left] = quantized_codes[left] * stored_scale -
                  reference[left];
              for (std::size_t right = 0; right < 32; ++right) {
                gradient[left] +=
                    activation_grams[(group * 32 + left) * 32 + right] *
                    (quantized_codes[right] * stored_scale -
                     reference[right]);
              }
            }
            std::uint64_t sweeps = 0;
            bool changed = false;
            do {
              changed = false;
              ++sweeps;
              for (std::size_t coordinate = 0;
                   coordinate < 32; ++coordinate) {
                const double diagonal = activation_grams[
                    (group * 32 + coordinate) * 32 + coordinate];
                if (diagonal == 0.0) continue;
                const double optimum_error =
                    error[coordinate] - gradient[coordinate] / diagonal;
                int candidate = NearestInt(static_cast<float>(
                    (reference[coordinate] + optimum_error) / stored_scale));
                candidate = std::max(-127, std::min(127, candidate));
                const double delta =
                    (candidate - quantized_codes[coordinate]) * stored_scale;
                if (delta == 0.0) continue;
                const double objective_change =
                    2.0 * delta * gradient[coordinate] +
                    delta * delta * diagonal;
                if (objective_change >= 0.0) continue;
                quantized_codes[coordinate] = candidate;
                error[coordinate] += delta;
                for (std::size_t row_index = 0;
                     row_index < 32; ++row_index) {
                  gradient[row_index] += activation_grams[
                      (group * 32 + row_index) * 32 + coordinate] * delta;
                }
                changed = true;
              }
            } while (changed);
            for (std::size_t within = 0; within < 32; ++within) {
              lsq_changed_code_count +=
                  quantized_codes[within] != initial_codes[within];
            }
            lsq_total_sweeps += sweeps;
            ++lsq_group_count;
            lsq_max_sweeps = std::max(lsq_max_sweeps, sweeps);
          }
          std::array<float, 32> residual{};
          float residual_maximum = 0.0f;
          for (std::size_t within = 0;
               within < weight_group_values; ++within) {
            int centered_quantized = quantized_codes[within];
            int stored = centered_quantized;
            if (affine_codec) {
              stored = centered_quantized + zero_point;
              stored = std::max(0, std::min(255, stored));
              centered_quantized = stored - zero_point;
              if (external_affine) {
                stored -= 128;
              }
            } else {
              if (args.representation != "exact-per16") {
                centered_quantized = std::max(
                    -127, std::min(127, centered_quantized));
              }
              stored = args.encoding == "s8"
                  ? centered_quantized : centered_quantized + zero_point;
            }
            const std::size_t k = group * weight_group_values + within;
            weights[(expert * kOutput + output) * kInput + k] =
                static_cast<std::int8_t>(
                    static_cast<std::uint8_t>(stored));
            const double reconstructed =
                static_cast<double>(centered_quantized) * stored_scale;
            const double difference = reconstructed - reference[within];
            residual[within] = static_cast<float>(-difference);
            residual_maximum = std::max(
                residual_maximum, std::abs(residual[within]));
            weight_error_squared += difference * difference;
            weight_reference_squared +=
                static_cast<double>(reference[within]) * reference[within];
            weight_max_abs = std::max(weight_max_abs, std::abs(difference));
          }
          if (with_residual) {
            float residual_scale = residual_maximum == 0.0f
                ? 0.0f : residual_maximum / 7.0f;
            std::array<bool, 32> ternary_selected{};
            if (with_ternary_residual && residual_maximum != 0.0f) {
              std::array<std::size_t, 32> order{};
              std::iota(order.begin(), order.end(), 0);
              std::stable_sort(
                  order.begin(), order.begin() + weight_group_values,
                  [&](std::size_t left, std::size_t right) {
                    return std::abs(residual[left]) >
                        std::abs(residual[right]);
                  });
              double prefix = 0.0;
              double best_score = 0.0;
              std::size_t best_count = 0;
              for (std::size_t count = 1;
                   count <= weight_group_values; ++count) {
                prefix += std::abs(residual[order[count - 1]]);
                const double score = prefix * prefix / count;
                if (score > best_score) {
                  best_score = score;
                  best_count = count;
                  residual_scale = static_cast<float>(prefix / count);
                }
              }
              for (std::size_t rank = 0; rank < best_count; ++rank) {
                ternary_selected[order[rank]] = true;
              }
            }
            residual_scales[
                (expert * weight_group_count + group) * kOutput + output] =
                residual_scale;
            for (std::size_t within = 0;
                 within < weight_group_values; ++within) {
              int quantized = 0;
              if (with_s4_residual) {
                quantized = residual_scale == 0.0f
                    ? 0 : NearestInt(residual[within] / residual_scale);
                quantized = std::max(-7, std::min(7, quantized));
              } else if (ternary_selected[within]) {
                quantized = residual[within] < 0.0f ? -1 : 1;
              }
              const std::size_t k = group * weight_group_values + within;
              const std::size_t logical =
                  (expert * kOutput + output) * kInput + k;
              if (with_s4_residual) {
                SetS4(residual_weights, logical, quantized);
              } else {
                ternary_weights[logical] =
                    static_cast<std::int8_t>(quantized);
              }
              residual_nonzero_count += quantized != 0;
              const double corrected =
                  static_cast<double>(quantized) * residual_scale;
              const double difference = corrected - residual[within];
              corrected_weight_error_squared += difference * difference;
              corrected_weight_max_abs = std::max(
                  corrected_weight_max_abs, std::abs(difference));
            }
          }
        }
      }
    }
    if (!with_residual) {
      corrected_weight_error_squared = weight_error_squared;
      corrected_weight_max_abs = weight_max_abs;
    }
    if (full_gram_lsq) {
      weight_error_squared = 0.0;
      weight_reference_squared = 0.0;
      weight_max_abs = 0.0;
      corrected_weight_error_squared = 0.0;
      corrected_weight_max_abs = 0.0;
      lsq_changed_code_count = 0;
      lsq_total_sweeps = 0;
      lsq_group_count = 0;
      lsq_max_sweeps = 0;
#pragma omp parallel for schedule(static) reduction(+ : weight_error_squared, weight_reference_squared, corrected_weight_error_squared, lsq_changed_code_count, lsq_total_sweeps, lsq_group_count) reduction(max : weight_max_abs, corrected_weight_max_abs, lsq_max_sweeps)
      for (std::int64_t row_signed = 0;
           row_signed < static_cast<std::int64_t>(kExperts * kOutput);
           ++row_signed) {
        const std::size_t weight_row =
            static_cast<std::size_t>(row_signed);
        const std::size_t expert = weight_row / kOutput;
        const std::size_t output = weight_row % kOutput;
        const auto* row = raw.data() + weight_row * 2 * kBlockBytes;
        std::array<float, kInput> reference{};
        std::array<int, kInput> codes{};
        std::array<int, kInput> initial_codes{};
        std::array<double, kInput> error{};
        std::array<double, kInput> gradient{};
        for (std::size_t k = 0; k < kInput; ++k) {
          const std::size_t block_index = k / kBlockValues;
          const std::size_t within_block = k % kBlockValues;
          const auto* block = row + block_index * kBlockBytes;
          const float d = HalfToFloat(LoadU16(block + 208));
          const auto* scales =
              reinterpret_cast<const std::int8_t*>(block + 192);
          reference[k] = d * static_cast<float>(
              scales[within_block / kExactGroupValues]) *
              static_cast<float>(Q6Value(block, within_block));
          codes[k] = weights[weight_row * kInput + k];
          initial_codes[k] = codes[k];
          const float scale = weight_scales[
              (expert * weight_group_count + k / 32) * kOutput + output];
          error[k] = codes[k] * scale - reference[k];
        }
        for (std::size_t left = 0; left < kInput; ++left) {
          for (std::size_t right = 0; right < kInput; ++right) {
            gradient[left] += activation_grams[left * kInput + right] *
                error[right];
          }
        }
        std::uint64_t sweeps = 0;
        bool changed = false;
        do {
          changed = false;
          ++sweeps;
          for (std::size_t coordinate = 0;
               coordinate < kInput; ++coordinate) {
            const double diagonal = activation_grams[
                coordinate * kInput + coordinate];
            if (diagonal == 0.0) continue;
            const float scale = weight_scales[
                (expert * weight_group_count + coordinate / 32) *
                    kOutput + output];
            if (scale == 0.0f) continue;
            const double optimum_error =
                error[coordinate] - gradient[coordinate] / diagonal;
            int candidate = NearestInt(static_cast<float>(
                (reference[coordinate] + optimum_error) / scale));
            candidate = std::max(-127, std::min(127, candidate));
            const double delta = (candidate - codes[coordinate]) * scale;
            if (delta == 0.0) continue;
            const double objective_change =
                2.0 * delta * gradient[coordinate] +
                delta * delta * diagonal;
            if (objective_change >= 0.0) continue;
            codes[coordinate] = candidate;
            error[coordinate] += delta;
            for (std::size_t row_index = 0;
                 row_index < kInput; ++row_index) {
              gradient[row_index] += activation_grams[
                  row_index * kInput + coordinate] * delta;
            }
            changed = true;
          }
        } while (changed);
        for (std::size_t k = 0; k < kInput; ++k) {
          weights[weight_row * kInput + k] =
              static_cast<std::int8_t>(codes[k]);
          lsq_changed_code_count += codes[k] != initial_codes[k];
          const float scale = weight_scales[
              (expert * weight_group_count + k / 32) * kOutput + output];
          const double reconstructed = codes[k] * scale;
          const double difference = reconstructed - reference[k];
          weight_error_squared += difference * difference;
          weight_reference_squared +=
              static_cast<double>(reference[k]) * reference[k];
          weight_max_abs = std::max(weight_max_abs, std::abs(difference));
        }
        lsq_total_sweeps += sweeps;
        ++lsq_group_count;
        lsq_max_sweeps = std::max(lsq_max_sweeps, sweeps);
      }
      corrected_weight_error_squared = weight_error_squared;
      corrected_weight_max_abs = weight_max_abs;
    }

    using dt = dnnl::memory::data_type;
    using tag = dnnl::memory::format_tag;
    dnnl::engine engine(dnnl::engine::kind::gpu, 0);
    dnnl::stream stream(engine);
    dnnl::memory source_memory(
        dnnl::memory::desc::grouped(
            {static_cast<int>(kRows), static_cast<int>(kInput)},
            dt::s8, 0, kExperts), engine);
    dnnl::memory weights_memory(
        dnnl::memory::desc(
            {static_cast<int>(kExperts), static_cast<int>(kInput),
             static_cast<int>(kOutput)},
            args.encoding == "s8" ? dt::s8 : dt::u8, tag::acb), engine);
    dnnl::memory output_memory(
        dnnl::memory::desc::grouped(
            {static_cast<int>(kRows), static_cast<int>(kOutput)},
            dt::f32, 0, kExperts), engine);
    dnnl::memory source_scale_memory(
        dnnl::memory::desc(
            {static_cast<int>(kRows), 2}, dt::f32, tag::ab), engine);
    dnnl::memory weight_scale_memory(
        dnnl::memory::desc(
            {static_cast<int>(kExperts), static_cast<int>(weight_group_count),
             static_cast<int>(kOutput)}, dt::f32, tag::abc), engine);
    dnnl::memory weight_zero_point_memory;
    WriteOffsets(source_memory, schedule.offsets);
    WriteOffsets(output_memory, schedule.offsets);
    WriteMemory(source, source_memory);
    WriteMemory(weights, weights_memory);
    WriteMemory(source_scales, source_scale_memory);
    WriteMemory(weight_scales, weight_scale_memory);

    dnnl::primitive_attr attributes;
    attributes.set_scales(DNNL_ARG_SRC, 3, {1, 256}, dt::f32);
    attributes.set_scales(
        DNNL_ARG_WEIGHTS, 7,
        {static_cast<std::int64_t>(weight_group_values), 1}, dt::f32);
    if (args.encoding != "s8") {
      attributes.set_zero_points(
          DNNL_ARG_WEIGHTS, 7,
          {static_cast<std::int64_t>(weight_group_values), 1}, dt::u8);
      weight_zero_point_memory = dnnl::memory(
          dnnl::memory::desc(
              {static_cast<int>(kExperts),
               static_cast<int>(weight_group_count),
               static_cast<int>(kOutput)}, dt::u8, tag::abc), engine);
      WriteMemory(
          weight_zero_points,
          weight_zero_point_memory);
    }
    dnnl::matmul::primitive_desc descriptor(
        engine, source_memory.get_desc(), weights_memory.get_desc(),
        output_memory.get_desc(), attributes);
    dnnl::matmul primitive(descriptor);
    dnnl::memory max_group_hint(
        dnnl::memory::desc::host_scalar(dt::s32),
        static_cast<std::int32_t>(schedule.max_group));
    std::unordered_map<int, dnnl::memory> arguments = {
        {DNNL_ARG_SRC, source_memory},
        {DNNL_ARG_WEIGHTS, weights_memory},
        {DNNL_ARG_DST, output_memory},
        {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, source_scale_memory},
        {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, weight_scale_memory},
        {DNNL_ARG_HINT_MAX_GROUP_SIZE, max_group_hint},
    };
    if (args.encoding != "s8") {
      arguments.emplace(
          DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS,
          weight_zero_point_memory);
    }
    std::unique_ptr<dnnl::memory> residual_weight_memory;
    std::unique_ptr<dnnl::memory> residual_scale_memory;
    std::unique_ptr<dnnl::memory> residual_output_memory;
    std::unique_ptr<dnnl::matmul::primitive_desc> residual_descriptor;
    std::unique_ptr<dnnl::matmul> residual_primitive;
    std::unordered_map<int, dnnl::memory> residual_arguments;
    cl_program add_program = nullptr;
    cl_kernel add_kernel = nullptr;
    cl_command_queue queue = nullptr;
    std::size_t add_global = 0;
    const std::size_t add_local = 256;
    std::unique_ptr<dnnl::memory> external_zero_point_memory;
    std::unique_ptr<dnnl::memory> external_expert_memory;
    std::unique_ptr<dnnl::memory> external_group_sum_memory;
    cl_program affine_program = nullptr;
    cl_kernel affine_group_sum_kernel = nullptr;
    cl_kernel affine_correction_kernel = nullptr;
    std::size_t affine_group_sum_global = 0;
    std::size_t affine_correction_global = 0;
    if (with_residual) {
      residual_weight_memory = std::make_unique<dnnl::memory>(
          dnnl::memory::desc(
              {static_cast<int>(kExperts), static_cast<int>(kInput),
               static_cast<int>(kOutput)},
              with_s4_residual ? dt::s4 : dt::s8, tag::acb), engine);
      residual_scale_memory = std::make_unique<dnnl::memory>(
          dnnl::memory::desc(
              {static_cast<int>(kExperts),
               static_cast<int>(weight_group_count),
               static_cast<int>(kOutput)}, dt::f32, tag::abc), engine);
      residual_output_memory = std::make_unique<dnnl::memory>(
          dnnl::memory::desc::grouped(
              {static_cast<int>(kRows), static_cast<int>(kOutput)},
              dt::f32, 0, kExperts), engine);
      WriteOffsets(*residual_output_memory, schedule.offsets);
      if (with_s4_residual) {
        WriteMemory(residual_weights, *residual_weight_memory);
      } else {
        WriteMemory(ternary_weights, *residual_weight_memory);
      }
      WriteMemory(residual_scales, *residual_scale_memory);
      dnnl::primitive_attr residual_attributes;
      residual_attributes.set_scales(
          DNNL_ARG_SRC, 3, {1, 256}, dt::f32);
      residual_attributes.set_scales(
          DNNL_ARG_WEIGHTS, 7, {32, 1}, dt::f32);
      residual_descriptor =
          std::make_unique<dnnl::matmul::primitive_desc>(
              engine, source_memory.get_desc(),
              residual_weight_memory->get_desc(),
              residual_output_memory->get_desc(), residual_attributes);
      residual_primitive =
          std::make_unique<dnnl::matmul>(*residual_descriptor);
      residual_arguments = {
          {DNNL_ARG_SRC, source_memory},
          {DNNL_ARG_WEIGHTS, *residual_weight_memory},
          {DNNL_ARG_DST, *residual_output_memory},
          {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, source_scale_memory},
          {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS,
           *residual_scale_memory},
          {DNNL_ARG_HINT_MAX_GROUP_SIZE, max_group_hint},
      };

      const cl_context context = dnnl::ocl_interop::get_context(engine);
      const cl_device_id device = dnnl::ocl_interop::get_device(engine);
      queue = dnnl::ocl_interop::get_command_queue(stream);
      const char* source_text = R"CLC(
__kernel void iq36_add_residual(__global float* main_values,
                                __global const float* residual_values,
                                ulong count) {
  const ulong index = get_global_id(0);
  if (index < count) main_values[index] += residual_values[index];
}
)CLC";
      const std::size_t source_length = std::strlen(source_text);
      cl_int status = CL_SUCCESS;
      add_program = clCreateProgramWithSource(
          context, 1, &source_text, &source_length, &status);
      CheckCl(status, "clCreateProgramWithSource residual add");
      status = clBuildProgram(
          add_program, 1, &device, "-cl-std=CL3.0", nullptr, nullptr);
      if (status != CL_SUCCESS) {
        Fail("residual add build failed: " +
             ProgramBuildLog(add_program, device));
      }
      add_kernel = clCreateKernel(
          add_program, "iq36_add_residual", &status);
      CheckCl(status, "clCreateKernel residual add");
      cl_mem main_values = dnnl::ocl_interop::get_mem_object(output_memory);
      cl_mem residual_values =
          dnnl::ocl_interop::get_mem_object(*residual_output_memory);
      const cl_ulong count = static_cast<cl_ulong>(kRows * kOutput);
      CheckCl(clSetKernelArg(
                  add_kernel, 0, sizeof(main_values), &main_values),
              "clSetKernelArg residual main");
      CheckCl(clSetKernelArg(
                  add_kernel, 1, sizeof(residual_values), &residual_values),
              "clSetKernelArg residual values");
      CheckCl(clSetKernelArg(add_kernel, 2, sizeof(count), &count),
              "clSetKernelArg residual count");
      add_global = ((kRows * kOutput + add_local - 1) / add_local) * add_local;
    }
    if (external_affine) {
      external_zero_point_memory = std::make_unique<dnnl::memory>(
          dnnl::memory::desc(
              {static_cast<int>(kExperts),
               static_cast<int>(weight_group_count),
               static_cast<int>(kOutput)}, dt::u8, tag::abc), engine);
      external_expert_memory = std::make_unique<dnnl::memory>(
          dnnl::memory::desc(
              {static_cast<int>(kRows)}, dt::s32, tag::a), engine);
      external_group_sum_memory = std::make_unique<dnnl::memory>(
          dnnl::memory::desc(
              {static_cast<int>(kRows),
               static_cast<int>(weight_group_count)}, dt::s32, tag::ab),
          engine);
      WriteMemory(weight_zero_points, *external_zero_point_memory);
      WriteMemory(schedule.expert_ids, *external_expert_memory);

      const cl_context context = dnnl::ocl_interop::get_context(engine);
      const cl_device_id device = dnnl::ocl_interop::get_device(engine);
      queue = dnnl::ocl_interop::get_command_queue(stream);
      const char* source_text = R"CLC(
__kernel void iq36_affine_group_sums(
    __global const char* source, __global int* group_sums,
    ulong group_count) {
  const ulong index = get_global_id(0);
  if (index >= group_count) return;
  const ulong row = index / 16;
  const ulong group = index - row * 16;
  int sum = 0;
#pragma unroll
  for (ulong inner = 0; inner < 32; ++inner) {
    sum += source[row * 512 + group * 32 + inner];
  }
  group_sums[index] = sum;
}

__kernel void iq36_affine_zero_point_correction(
    __global float* output, __global const int* group_sums,
    __global const float* weight_scales,
    __global const uchar* weight_zero_points,
    __global const float* source_scales,
    __global const int* expert_ids, ulong value_count) {
  const ulong index = get_global_id(0);
  if (index >= value_count) return;
  const ulong row = index / 2048;
  const ulong column = index - row * 2048;
  const ulong expert = expert_ids[row];
  float correction = 0.0f;
#pragma unroll
  for (ulong group = 0; group < 16; ++group) {
    const ulong coefficient = (expert * 16 + group) * 2048 + column;
    const float affine =
        (128 - (int)weight_zero_points[coefficient]) *
        weight_scales[coefficient];
    correction += affine * group_sums[row * 16 + group] *
        source_scales[row * 2 + group / 8];
  }
  output[index] += correction;
}
)CLC";
      const std::size_t source_length = std::strlen(source_text);
      cl_int status = CL_SUCCESS;
      affine_program = clCreateProgramWithSource(
          context, 1, &source_text, &source_length, &status);
      CheckCl(status, "clCreateProgramWithSource affine correction");
      status = clBuildProgram(
          affine_program, 1, &device, "-cl-std=CL3.0", nullptr, nullptr);
      if (status != CL_SUCCESS) {
        Fail("affine correction build failed: " +
             ProgramBuildLog(affine_program, device));
      }
      affine_group_sum_kernel = clCreateKernel(
          affine_program, "iq36_affine_group_sums", &status);
      CheckCl(status, "clCreateKernel affine group sums");
      affine_correction_kernel = clCreateKernel(
          affine_program, "iq36_affine_zero_point_correction", &status);
      CheckCl(status, "clCreateKernel affine correction");

      cl_mem source_values =
          dnnl::ocl_interop::get_mem_object(source_memory);
      cl_mem group_sums =
          dnnl::ocl_interop::get_mem_object(*external_group_sum_memory);
      const cl_ulong group_count =
          static_cast<cl_ulong>(kRows * weight_group_count);
      CheckCl(clSetKernelArg(
                  affine_group_sum_kernel, 0,
                  sizeof(source_values), &source_values),
              "clSetKernelArg affine source");
      CheckCl(clSetKernelArg(
                  affine_group_sum_kernel, 1,
                  sizeof(group_sums), &group_sums),
              "clSetKernelArg affine sums");
      CheckCl(clSetKernelArg(
                  affine_group_sum_kernel, 2,
                  sizeof(group_count), &group_count),
              "clSetKernelArg affine group count");

      cl_mem output_values =
          dnnl::ocl_interop::get_mem_object(output_memory);
      cl_mem scale_values =
          dnnl::ocl_interop::get_mem_object(weight_scale_memory);
      cl_mem zero_point_values =
          dnnl::ocl_interop::get_mem_object(*external_zero_point_memory);
      cl_mem source_scale_values =
          dnnl::ocl_interop::get_mem_object(source_scale_memory);
      cl_mem expert_values =
          dnnl::ocl_interop::get_mem_object(*external_expert_memory);
      const cl_ulong value_count =
          static_cast<cl_ulong>(kRows * kOutput);
      CheckCl(clSetKernelArg(
                  affine_correction_kernel, 0,
                  sizeof(output_values), &output_values),
              "clSetKernelArg affine output");
      CheckCl(clSetKernelArg(
                  affine_correction_kernel, 1,
                  sizeof(group_sums), &group_sums),
              "clSetKernelArg affine correction sums");
      CheckCl(clSetKernelArg(
                  affine_correction_kernel, 2,
                  sizeof(scale_values), &scale_values),
              "clSetKernelArg affine scales");
      CheckCl(clSetKernelArg(
                  affine_correction_kernel, 3,
                  sizeof(zero_point_values), &zero_point_values),
              "clSetKernelArg affine zero points");
      CheckCl(clSetKernelArg(
                  affine_correction_kernel, 4,
                  sizeof(source_scale_values), &source_scale_values),
              "clSetKernelArg affine source scales");
      CheckCl(clSetKernelArg(
                  affine_correction_kernel, 5,
                  sizeof(expert_values), &expert_values),
              "clSetKernelArg affine experts");
      CheckCl(clSetKernelArg(
                  affine_correction_kernel, 6,
                  sizeof(value_count), &value_count),
              "clSetKernelArg affine value count");
      affine_group_sum_global =
          ((kRows * weight_group_count + add_local - 1) / add_local) *
          add_local;
      affine_correction_global =
          ((kRows * kOutput + add_local - 1) / add_local) * add_local;
    }
    auto execute = [&]() {
      primitive.execute(stream, arguments);
      if (with_residual) {
        residual_primitive->execute(stream, residual_arguments);
        CheckCl(clEnqueueNDRangeKernel(
                    queue, add_kernel, 1, nullptr, &add_global, &add_local,
                    0, nullptr, nullptr),
                "enqueue residual add");
      }
      if (external_affine) {
        CheckCl(clEnqueueNDRangeKernel(
                    queue, affine_group_sum_kernel, 1, nullptr,
                    &affine_group_sum_global, &add_local,
                    0, nullptr, nullptr),
                "enqueue affine group sums");
        CheckCl(clEnqueueNDRangeKernel(
                    queue, affine_correction_kernel, 1, nullptr,
                    &affine_correction_global, &add_local,
                    0, nullptr, nullptr),
                "enqueue affine correction");
      }
    };
    for (int iteration = 0; iteration < args.warmup; ++iteration) {
      execute();
      stream.wait();
    }
    std::vector<double> samples;
    for (int iteration = 0; iteration < args.repeat; ++iteration) {
      const auto begin = std::chrono::steady_clock::now();
      execute();
      stream.wait();
      samples.push_back(std::chrono::duration<double, std::micro>(
          std::chrono::steady_clock::now() - begin).count());
    }
    std::sort(samples.begin(), samples.end());
    const double minimum_us = samples.front();

    const auto* output_values = output_memory.map_data<float>();
    Require(output_values != nullptr, "could not map output");
    constexpr std::size_t kScaleProbeOutputs = 64;
    double host_probe_gpu_vs_repacked_max_abs = 0.0;
    double host_probe_gpu_vs_effective_max_abs = 0.0;
    double host_probe_repacked_vs_oracle_max_abs = 0.0;
    const std::size_t scale_probe_source_row = 0;
    const std::size_t scale_probe_grouped = static_cast<std::size_t>(
        schedule.inverse[scale_probe_source_row]);
    const std::size_t scale_probe_expert = static_cast<std::size_t>(
        schedule.expert_ids[scale_probe_grouped]);
    for (std::size_t output = 0; output < kScaleProbeOutputs; ++output) {
      double repacked = 0.0;
      double effective = 0.0;
      for (std::size_t group = 0; group < weight_group_count; ++group) {
        int raw_dot = 0;
        int residual_dot = 0;
        for (std::size_t inner = 0;
             inner < weight_group_values; ++inner) {
          const std::size_t logical =
              (scale_probe_expert * kOutput + output) * kInput +
              group * weight_group_values + inner;
          const int stored_weight = args.encoding == "s8"
              ? weights[logical]
              : static_cast<std::uint8_t>(weights[logical]);
          int zero_point = 0;
          if (args.encoding != "s8" || external_affine) {
            zero_point = weight_zero_points[
                (scale_probe_expert * weight_group_count + group) *
                    kOutput + output];
            if (external_affine) zero_point -= 128;
          }
          const int weight = stored_weight - zero_point;
          raw_dot += source[
              scale_probe_grouped * kInput +
              group * weight_group_values + inner] * weight;
          if (with_residual) {
            residual_dot += source[
                scale_probe_grouped * kInput +
                group * weight_group_values + inner] *
                (with_s4_residual
                     ? GetS4(residual_weights, logical)
                     : ternary_weights[logical]);
          }
        }
        const double source_scale = source_scales[
            scale_probe_grouped * 2 +
            (group * weight_group_values) / kBlockValues];
        repacked += static_cast<double>(raw_dot) * source_scale *
            weight_scales[(scale_probe_expert * weight_group_count + group) *
                              kOutput + output];
        const std::size_t effective_group =
            args.representation == "exact-per16"
            ? (group & ~std::size_t{1}) : group;
        effective += static_cast<double>(raw_dot) * source_scale *
            weight_scales[(scale_probe_expert * weight_group_count +
                            effective_group) * kOutput + output];
        if (with_residual) {
          const double residual =
              static_cast<double>(residual_dot) * source_scale *
              residual_scales[(scale_probe_expert * weight_group_count +
                               group) * kOutput + output];
          repacked += residual;
          effective += residual;
        }
      }
      const double gpu = output_values[
          scale_probe_grouped * kOutput + output];
      host_probe_gpu_vs_repacked_max_abs = std::max(
          host_probe_gpu_vs_repacked_max_abs, std::abs(gpu - repacked));
      host_probe_gpu_vs_effective_max_abs = std::max(
          host_probe_gpu_vs_effective_max_abs, std::abs(gpu - effective));
      host_probe_repacked_vs_oracle_max_abs = std::max(
          host_probe_repacked_vs_oracle_max_abs,
          std::abs(repacked -
                   oracle[scale_probe_source_row * kOutput + output]));
    }
    double error_squared = 0.0;
    double candidate_squared = 0.0;
    double reference_squared = 0.0;
    double dot = 0.0;
    double max_abs = 0.0;
    bool finite = true;
#pragma omp parallel for schedule(static) reduction(+ : error_squared, candidate_squared, reference_squared, dot) reduction(max : max_abs) reduction(&& : finite)
    for (std::int64_t source_signed = 0;
         source_signed < static_cast<std::int64_t>(kRows); ++source_signed) {
      const std::size_t source_row =
          static_cast<std::size_t>(source_signed);
      const std::size_t grouped =
          static_cast<std::size_t>(schedule.inverse[source_row]);
      for (std::size_t output = 0; output < kOutput; ++output) {
        const double candidate =
            output_values[grouped * kOutput + output];
        const double reference = oracle[source_row * kOutput + output];
        const double difference = candidate - reference;
        finite = finite && std::isfinite(candidate) && std::isfinite(reference);
        max_abs = std::max(max_abs, std::abs(difference));
        error_squared += difference * difference;
        candidate_squared += candidate * candidate;
        reference_squared += reference * reference;
        dot += candidate * reference;
      }
    }
    output_memory.unmap_data(const_cast<float*>(output_values));
    const double relative_l2 = std::sqrt(error_squared / reference_squared);
    const double cosine = dot /
        std::sqrt(candidate_squared * reference_squared);
    const double weight_relative_l2 =
        std::sqrt(weight_error_squared / weight_reference_squared);
    const double corrected_weight_relative_l2 =
        std::sqrt(corrected_weight_error_squared / weight_reference_squared);
    const bool correctness_pass =
        finite && cosine >= 0.999 && relative_l2 <= 0.002;
    const bool performance_pass = minimum_us <= args.cap_us;
    std::cout << std::boolalpha << std::setprecision(12) << "{";
    std::cout << "\"active_experts\":" << schedule.active_experts << ",";
    std::cout << "\"cap_us\":" << args.cap_us << ",";
    std::cout << "\"comparison\":{\"cosine\":" << cosine << ",";
    std::cout << "\"compared_value_count\":" << kRows * kOutput << ",";
    std::cout << "\"finite\":" << finite << ",";
    std::cout << "\"max_abs_diff\":" << max_abs << ",";
    std::cout << "\"relative_l2\":" << relative_l2 << "},";
    std::cout << "\"correctness_pass\":" << correctness_pass << ",";
    std::cout << "\"encoding\":\"" << args.encoding << "\",";
    std::cout << "\"host_repacked_probe\":{";
    std::cout << "\"gpu_vs_effective_max_abs\":"
              << host_probe_gpu_vs_effective_max_abs << ",";
    std::cout << "\"gpu_vs_repacked_max_abs\":"
              << host_probe_gpu_vs_repacked_max_abs << ",";
    std::cout << "\"repacked_vs_oracle_max_abs\":"
              << host_probe_repacked_vs_oracle_max_abs << ",";
    std::cout << "\"sample_value_count\":" << kScaleProbeOutputs << "},";
    std::cout << "\"implementation\":\""
              << descriptor.impl_info_str() << "\",";
    if (with_residual) {
      std::cout << "\"residual_implementation\":\""
                << residual_descriptor->impl_info_str() << "\",";
    }
    std::cout << "\"max_group_size\":" << schedule.max_group << ",";
    std::cout << "\"minimum_us\":" << minimum_us << ",";
    std::cout << "\"performance_pass\":" << performance_pass << ",";
    std::cout << "\"raw_core_only\":true,";
    std::cout << "\"representation\":\"" << args.representation << "\",";
    std::cout << "\"external_affine_correction\":"
              << external_affine << ",";
    if (activation_lsq || full_gram_lsq) {
      std::cout << "\"activation_lsq\":{";
      std::cout << "\"changed_code_count\":"
                << lsq_changed_code_count << ",";
      std::cout << "\"changed_code_fraction\":"
                << static_cast<double>(lsq_changed_code_count) /
                    (kExperts * kOutput * kInput) << ",";
      std::cout << "\"gram_row_count\":" << kRows << ",";
      std::cout << "\"gram_width\":"
                << (full_gram_lsq ? kInput : 32) << ",";
      std::cout << "\"group_count\":" << lsq_group_count << ",";
      std::cout << "\"max_sweeps\":" << lsq_max_sweeps << ",";
      std::cout << "\"mean_sweeps\":"
                << static_cast<double>(lsq_total_sweeps) /
                    lsq_group_count << "},";
    }
    if (args.representation == "exact-per16") {
      std::cout << "\"scale_granularity_probe\":{";
      std::cout << "\"exact_weight_group_values\":16,";
      std::cout << "\"gpu_vs_effective_per32_max_abs\":"
                << host_probe_gpu_vs_effective_max_abs << ",";
      std::cout << "\"gpu_vs_exact_per16_max_abs\":"
                << host_probe_gpu_vs_repacked_max_abs << ",";
      std::cout << "\"exact_per16_vs_oracle_max_abs\":"
                << host_probe_repacked_vs_oracle_max_abs << ",";
      std::cout << "\"sample_value_count\":" << kScaleProbeOutputs << "},";
    }
    std::cout << "\"weight_requantization\":{";
    std::cout << "\"group_values\":" << weight_group_values << ",";
    if (affine_codec) {
      const auto zero_point_range = std::minmax_element(
          weight_zero_points.begin(), weight_zero_points.end());
      std::cout << "\"zero_point_max\":"
                << static_cast<int>(*zero_point_range.second) << ",";
      std::cout << "\"zero_point_min\":"
                << static_cast<int>(*zero_point_range.first) << ",";
    }
    std::cout << "\"max_abs_diff\":" << weight_max_abs << ",";
    std::cout << "\"relative_l2\":" << weight_relative_l2 << "},";
    if (with_residual) {
      std::cout << "\"residual_correction\":{";
      std::cout << "\"code_max\":"
                << (with_s4_residual ? 7 : 1) << ",";
      std::cout << "\"group_values\":32,";
      std::cout << "\"max_abs_diff\":"
                << corrected_weight_max_abs << ",";
      std::cout << "\"relative_l2\":"
                << corrected_weight_relative_l2 << ",";
      std::cout << "\"nonzero_count\":" << residual_nonzero_count << ",";
      std::cout << "\"nonzero_density\":"
                << static_cast<double>(residual_nonzero_count) /
                    (kExperts * kOutput * kInput) << "},";
    }
    std::cout << "\"samples_us\":[";
    for (std::size_t i = 0; i < samples.size(); ++i) {
      if (i != 0) std::cout << ',';
      std::cout << samples[i];
    }
    std::cout << "],\"schema_version\":"
                 "\"iq36-onednn-grouped-q6-exact-preflight-v0\"}"
              << std::endl;
    if (add_kernel != nullptr) clReleaseKernel(add_kernel);
    if (add_program != nullptr) clReleaseProgram(add_program);
    if (affine_group_sum_kernel != nullptr) {
      clReleaseKernel(affine_group_sum_kernel);
    }
    if (affine_correction_kernel != nullptr) {
      clReleaseKernel(affine_correction_kernel);
    }
    if (affine_program != nullptr) clReleaseProgram(affine_program);
    return correctness_pass && performance_pass ? 0 : 2;
  } catch (const std::exception& exception) {
    std::cerr << "onednn-grouped-q6-exact-preflight: "
              << exception.what() << '\n';
    return 4;
  }
}
