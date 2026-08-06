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
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

constexpr std::size_t kTokenCount = 1024;
constexpr std::size_t kHiddenSize = 2048;
constexpr std::size_t kExpertCount = 256;
constexpr std::size_t kSelectedExperts = 8;
constexpr std::size_t kIntermediateSize = 512;
constexpr std::size_t kGateUpSize = 1024;
constexpr std::size_t kAssignments = kTokenCount * kSelectedExperts;
constexpr std::size_t kQ4BlockValues = 256;
constexpr std::size_t kQ4BlockBytes = 144;
constexpr std::size_t kBlocksPerRow = kHiddenSize / kQ4BlockValues;
constexpr std::size_t kRowsPerExpert = kGateUpSize;
constexpr std::size_t kExpertWeightBytes =
    kRowsPerExpert * kBlocksPerRow * kQ4BlockBytes;
constexpr std::size_t kExpectedWeightBytes =
    kExpertCount * kExpertWeightBytes;
constexpr std::size_t kQ8ScaleGroups = kBlocksPerRow;
constexpr std::size_t kQ4ScaleGroups = kBlocksPerRow * 8;
constexpr std::size_t kDownInputSize = 512;
constexpr std::size_t kDownOutputSize = 2048;
constexpr std::size_t kDownBlocksPerRow = kDownInputSize / kQ4BlockValues;
constexpr std::size_t kDownScaleGroups = kDownBlocksPerRow * 8;
constexpr std::size_t kDownExpertWeightBytes =
    kDownOutputSize * kDownBlocksPerRow * kQ4BlockBytes;
constexpr std::size_t kExpectedDownWeightBytes =
    kExpertCount * kDownExpertWeightBytes;
constexpr double kMismatchThreshold = 5e-3;
constexpr std::array<int, 7> kLockedBucketM = {8, 16, 32, 64, 128, 256, 512};
constexpr std::array<int, 7> kLockedBucketExperts = {86, 41, 28, 28, 19, 17, 3};

constexpr const char* kSwiGluSource = R"CLC(
float swiglu(float gate, float up) {
  const float sigmoid =
      gate >= 0.0f ? 1.0f / (1.0f + exp(-gate))
                   : exp(gate) / (1.0f + exp(gate));
  return gate * sigmoid * up;
}

__kernel void q4k_compensate_swiglu(
    __global const float * main_term,
    __global const float * min_term,
    __global float * output,
    uint row_count) {
  const uint index = get_global_id(0);
  const uint output_count = row_count * 512U;
  if (index >= output_count) return;
  const uint row = index / 512U;
  const uint inner = index - row * 512U;
  const uint base = row * 1024U;
  const float gate = main_term[base + inner] - min_term[base + inner];
  const float up = main_term[base + 512U + inner] -
                   min_term[base + 512U + inner];
  output[index] = swiglu(gate, up);
}

int nearest_int(float value) {
  const float shifted = value + 12582912.0f;
  return (as_int(shifted) & 0x007fffff) - 0x00400000;
}

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void q4k_gather_quantize_input(
    __global const float * input,
    __global const int * token_map,
    __global char * q8,
    __global float * scales,
    __global float * sums32_scaled,
    uint row_count) {
  const uint lane = get_local_id(0);
  const uint group = get_group_id(0);
  const uint row = group >> 3;
  const uint block = group & 7U;
  if (row >= row_count) return;
  const int token = token_map[row];
  if (token < 0) {
    q8[row * 2048U + block * 256U + lane] = (char)0;
    if (lane == 0U) scales[row * 8U + block] = 0.0f;
    if (lane < 8U) sums32_scaled[row * 64U + block * 8U + lane] = 0.0f;
    return;
  }
  __local float values[256];
  __local float maxima[256];
  __local uint indices[256];
  __local int quantized[256];
  const float value = input[(uint)token * 2048U + block * 256U + lane];
  values[lane] = value;
  maxima[lane] = fabs(value);
  indices[lane] = lane;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint step = 128U; step > 0U; step >>= 1U) {
    if (lane < step) {
      const float rhs = maxima[lane + step];
      const uint rhs_index = indices[lane + step];
      if (rhs > maxima[lane] ||
          (rhs == maxima[lane] && rhs_index < indices[lane])) {
        maxima[lane] = rhs;
        indices[lane] = rhs_index;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float max_value = values[indices[0]];
  const float inverse_scale = maxima[0] == 0.0f ? 0.0f : -127.0f / max_value;
  const float scale = inverse_scale == 0.0f ? 0.0f : 1.0f / inverse_scale;
  const int q = inverse_scale == 0.0f
      ? 0 : min(127, nearest_int(inverse_scale * value));
  quantized[lane] = q;
  q8[row * 2048U + block * 256U + lane] = (char)q;
  if (lane == 0U) scales[row * 8U + block] = scale;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lane < 8U) {
    int sum = 0;
    for (uint index = 0U; index < 32U; ++index) {
      sum += quantized[lane * 32U + index];
    }
    sums32_scaled[row * 64U + block * 8U + lane] =
        scale * (float)sum;
  }
}

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void q4k_compensate_swiglu_quantize(
    __global const float * main_term,
    __global const float * min_term,
    __global float * swiglu_output,
    __global char * q8,
    __global float * scales,
    __global float * sums32_scaled,
    uint row_count) {
  const uint lane = get_local_id(0);
  const uint group = get_group_id(0);
  const uint row = group >> 1;
  const uint block = group & 1U;
  if (row >= row_count) return;
  const uint inner = block * 256U + lane;
  const uint base = row * 1024U;
  const float gate = main_term[base + inner] - min_term[base + inner];
  const float up = main_term[base + 512U + inner] -
                   min_term[base + 512U + inner];
  const float value = swiglu(gate, up);
  swiglu_output[row * 512U + inner] = value;
  __local float values[256];
  __local float maxima[256];
  __local uint indices[256];
  __local int quantized[256];
  values[lane] = value;
  maxima[lane] = fabs(value);
  indices[lane] = lane;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint step = 128U; step > 0U; step >>= 1U) {
    if (lane < step) {
      const float rhs = maxima[lane + step];
      const uint rhs_index = indices[lane + step];
      if (rhs > maxima[lane] ||
          (rhs == maxima[lane] && rhs_index < indices[lane])) {
        maxima[lane] = rhs;
        indices[lane] = rhs_index;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float max_value = values[indices[0]];
  const float inverse_scale = maxima[0] == 0.0f ? 0.0f : -127.0f / max_value;
  const float scale = inverse_scale == 0.0f ? 0.0f : 1.0f / inverse_scale;
  const int q = inverse_scale == 0.0f
      ? 0 : min(127, nearest_int(inverse_scale * value));
  quantized[lane] = q;
  q8[row * 512U + inner] = (char)q;
  if (lane == 0U) scales[row * 2U + block] = scale;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lane < 8U) {
    int sum = 0;
    for (uint index = 0U; index < 32U; ++index) {
      sum += quantized[lane * 32U + index];
    }
    sums32_scaled[row * 16U + block * 8U + lane] =
        scale * (float)sum;
  }
}

__kernel void q4k_down_compensate_weight(
    __global const float * main_term,
    __global const float * min_term,
    __global const int * bucket_map,
    __global const float * bucket_weights,
    __global float * contributions,
    uint row_count) {
  const uint index = get_global_id(0);
  const uint count = row_count * 2048U;
  if (index >= count) return;
  const uint row = index / 2048U;
  const uint output = index - row * 2048U;
  const int bucket = bucket_map[row];
  if (bucket < 0) return;
  contributions[(uint)bucket * 2048U + output] =
      (main_term[index] - min_term[index]) * bucket_weights[(uint)bucket];
}

__kernel void q4k_scatter_routed_output(
    __global const float * contributions,
    __global const int * token_rank_to_bucket,
    __global float * output) {
  const uint index = get_global_id(0);
  if (index >= 1024U * 2048U) return;
  const uint token = index / 2048U;
  const uint hidden = index - token * 2048U;
  float sum = 0.0f;
  for (uint rank = 0U; rank < 8U; ++rank) {
    const int bucket = token_rank_to_bucket[token * 8U + rank];
    sum += contributions[(uint)bucket * 2048U + hidden];
  }
  output[index] = sum;
}
)CLC";

[[noreturn]] void Fail(const std::string& message) {
  throw std::runtime_error(message);
}

void Require(bool condition, const std::string& message) {
  if (!condition) Fail(message);
}

void CheckCl(cl_int status, const std::string& where) {
  if (status != CL_SUCCESS) {
    std::ostringstream stream;
    stream << where << " failed with OpenCL error " << status;
    Fail(stream.str());
  }
}

struct Args {
  std::string model;
  std::string input;
  std::string topk;
  std::string oracle;
  std::string router_weights;
  std::string down_oracle;
  std::string moe_oracle;
  std::string grouped_gateup_binary;
  std::string grouped_down_binary;
  std::string dump_prepacked_dir;
  bool prepack_only = false;
  std::uint64_t weight_offset = 0;
  std::uint64_t weight_bytes = 0;
  std::uint64_t down_weight_offset = 0;
  std::uint64_t down_weight_bytes = 0;
  std::size_t topk_stride = 0;
  int warmup = 3;
  int repeat = 11;
  double kernel_cap_us = 0.0;
};

Args ParseArgs(int argc, char** argv) {
  Args args;
  for (int index = 1; index < argc; ++index) {
    const std::string option = argv[index];
    const auto value = [&]() -> std::string {
      if (++index >= argc) Fail(option + " requires a value");
      return argv[index];
    };
    if (option == "--model") args.model = value();
    else if (option == "--input") args.input = value();
    else if (option == "--topk") args.topk = value();
    else if (option == "--oracle") args.oracle = value();
    else if (option == "--router-weights") args.router_weights = value();
    else if (option == "--down-oracle") args.down_oracle = value();
    else if (option == "--moe-oracle") args.moe_oracle = value();
    else if (option == "--grouped-gateup-binary") {
      args.grouped_gateup_binary = value();
    } else if (option == "--grouped-down-binary") {
      args.grouped_down_binary = value();
    } else if (option == "--dump-prepacked-dir") {
      args.dump_prepacked_dir = value();
    } else if (option == "--prepack-only") {
      args.prepack_only = true;
    } else if (option == "--weight-offset") args.weight_offset = std::stoull(value());
    else if (option == "--weight-bytes") args.weight_bytes = std::stoull(value());
    else if (option == "--down-weight-offset") {
      args.down_weight_offset = std::stoull(value());
    } else if (option == "--down-weight-bytes") {
      args.down_weight_bytes = std::stoull(value());
    }
    else if (option == "--topk-stride") args.topk_stride = std::stoull(value());
    else if (option == "--warmup") args.warmup = std::stoi(value());
    else if (option == "--repeat") args.repeat = std::stoi(value());
    else if (option == "--kernel-cap-us") args.kernel_cap_us = std::stod(value());
    else Fail("unknown option: " + option);
  }
  Require(!args.model.empty(), "model is required");
  Require(args.weight_bytes == kExpectedWeightBytes,
          "weight byte count does not match the locked Q4_K tensor");
  const bool routed = args.down_weight_bytes != 0;
  Require(!routed || args.down_weight_bytes == kExpectedDownWeightBytes,
          "routed mode requires locked down weights");
  if (args.prepack_only) {
    Require(routed && !args.dump_prepacked_dir.empty(),
            "prepack-only mode requires routed weights and a dump directory");
  } else {
    Require(!args.input.empty() && !args.topk.empty() && !args.oracle.empty(),
            "input, topk, and oracle are required");
    Require(!routed ||
                (!args.router_weights.empty() && !args.down_oracle.empty() &&
                 !args.moe_oracle.empty()),
            "routed mode requires router/down/MoE oracles");
    Require(args.topk_stride >= kSelectedExperts * sizeof(std::int32_t),
            "top-k stride is too small");
    Require(args.warmup > 0 && args.repeat > 0 && args.kernel_cap_us > 0.0,
            "warmup, repeat, and kernel cap must be positive");
  }
  Require(args.grouped_gateup_binary.empty() ==
              args.grouped_down_binary.empty(),
          "grouped fused mode requires both gate/up and down binaries");
  return args;
}

bool RoutedMode(const Args& args) { return args.down_weight_bytes != 0; }

template <typename Value>
std::vector<Value> ReadVector(const std::string& path,
                              std::size_t expected_count) {
  std::ifstream input(path, std::ios::binary);
  Require(static_cast<bool>(input), "could not open input: " + path);
  input.seekg(0, std::ios::end);
  const auto size = input.tellg();
  Require(size >= 0 && static_cast<std::uint64_t>(size) ==
                           expected_count * sizeof(Value),
          "input size mismatch: " + path);
  input.seekg(0, std::ios::beg);
  std::vector<Value> values(expected_count);
  input.read(reinterpret_cast<char*>(values.data()),
             static_cast<std::streamsize>(values.size() * sizeof(Value)));
  Require(static_cast<bool>(input), "could not read input: " + path);
  return values;
}

std::vector<std::uint8_t> ReadBytes(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  Require(static_cast<bool>(input), "could not open input: " + path);
  input.seekg(0, std::ios::end);
  const auto size = input.tellg();
  Require(size >= 0, "could not determine input size: " + path);
  input.seekg(0, std::ios::beg);
  std::vector<std::uint8_t> values(static_cast<std::size_t>(size));
  input.read(reinterpret_cast<char*>(values.data()),
             static_cast<std::streamsize>(values.size()));
  Require(static_cast<bool>(input), "could not read input: " + path);
  return values;
}

std::vector<std::uint8_t> ReadModelSlice(
    const std::string& model_path, std::uint64_t offset, std::uint64_t bytes) {
  std::ifstream model(model_path, std::ios::binary);
  Require(static_cast<bool>(model), "could not open model");
  model.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
  Require(static_cast<bool>(model), "could not seek model weight");
  std::vector<std::uint8_t> weight(static_cast<std::size_t>(bytes));
  model.read(reinterpret_cast<char*>(weight.data()),
             static_cast<std::streamsize>(weight.size()));
  Require(model.gcount() == static_cast<std::streamsize>(weight.size()),
          "could not read complete model weight");
  return weight;
}

float HalfToFloat(std::uint16_t value) {
  const std::uint32_t sign = (value & 0x8000U) << 16;
  std::uint32_t exponent = (value >> 10) & 0x1fU;
  std::uint32_t mantissa = value & 0x3ffU;
  std::uint32_t bits = 0;
  if (exponent == 0) {
    if (mantissa == 0) {
      bits = sign;
    } else {
      std::uint32_t shift = 0;
      while ((mantissa & 0x400U) == 0) {
        mantissa <<= 1;
        ++shift;
      }
      mantissa &= 0x3ffU;
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

std::uint16_t LoadU16(const std::uint8_t* data) {
  return static_cast<std::uint16_t>(data[0]) |
         static_cast<std::uint16_t>(data[1] << 8);
}

std::uint8_t GetScale(int index, const std::uint8_t* scales) {
  if (index < 4) return scales[index] & 63U;
  return static_cast<std::uint8_t>(
      (scales[index + 4] & 0x0fU) | ((scales[index - 4] >> 6) << 4));
}

std::uint8_t GetMinimum(int index, const std::uint8_t* scales) {
  if (index < 4) return scales[index + 4] & 63U;
  return static_cast<std::uint8_t>(
      (scales[index + 4] >> 4) | ((scales[index] >> 6) << 4));
}

int NearestInt(float value) {
  const float shifted = value + 12582912.0f;
  int bits = 0;
  std::memcpy(&bits, &shifted, sizeof(bits));
  return (bits & 0x007fffff) - 0x00400000;
}

struct Q8Data {
  std::vector<std::int8_t> values;
  std::vector<float> scales;
  std::vector<float> sums32_scaled;
};

Q8Data QuantizeQ8K(const std::vector<float>& input) {
  Q8Data q8;
  q8.values.resize(kTokenCount * kHiddenSize, 0);
  q8.scales.resize(kTokenCount * kQ8ScaleGroups, 0.0f);
  q8.sums32_scaled.resize(kTokenCount * kQ4ScaleGroups, 0.0f);
  for (std::size_t token = 0; token < kTokenCount; ++token) {
    for (std::size_t block = 0; block < kBlocksPerRow; ++block) {
      const float* source =
          input.data() + token * kHiddenSize + block * kQ4BlockValues;
      float max_value = 0.0f;
      float absolute_max = 0.0f;
      for (std::size_t index = 0; index < kQ4BlockValues; ++index) {
        const float absolute = std::fabs(source[index]);
        if (absolute > absolute_max) {
          absolute_max = absolute;
          max_value = source[index];
        }
      }
      if (absolute_max == 0.0f) continue;
      const float inverse_scale = -127.0f / max_value;
      const float scale = 1.0f / inverse_scale;
      q8.scales[token * kQ8ScaleGroups + block] = scale;
      std::array<int, 8> sums32{};
      for (std::size_t index = 0; index < kQ4BlockValues; ++index) {
        const int value =
            std::min(127, NearestInt(inverse_scale * source[index]));
        q8.values[token * kHiddenSize + block * kQ4BlockValues + index] =
            static_cast<std::int8_t>(value);
        sums32[index / 32] += value;
      }
      for (std::size_t group = 0; group < 8; ++group) {
        q8.sums32_scaled[token * kQ4ScaleGroups + block * 8 + group] =
            scale * static_cast<float>(sums32[group]);
      }
    }
  }
  return q8;
}

struct Assignment {
  std::uint32_t token = 0;
  std::uint32_t rank = 0;
  std::uint32_t bucket_index = 0;
};

struct ExpertGroup {
  std::uint32_t expert = 0;
  int bucket_m = 0;
  std::vector<Assignment> assignments;
};

struct BucketPlan {
  std::array<std::vector<ExpertGroup>, kLockedBucketM.size()> buckets;
  std::vector<std::uint32_t> bucket_token;
  std::vector<std::uint32_t> bucket_rank;
  std::size_t active_experts = 0;
  std::size_t padded_assignments = 0;
  std::size_t max_group_m = 0;
};

std::size_t BucketIndex(std::size_t group_m) {
  for (std::size_t index = 0; index < kLockedBucketM.size(); ++index) {
    if (group_m <= static_cast<std::size_t>(kLockedBucketM[index])) return index;
  }
  Fail("expert group exceeds the locked M=512 ceiling");
}

BucketPlan BuildPlan(const std::vector<std::uint8_t>& topk,
                     std::size_t stride,
                     bool enforce_locked_layer27 = true) {
  std::array<std::vector<std::pair<std::uint32_t, std::uint32_t>>,
             kExpertCount>
      grouped;
  for (std::size_t token = 0; token < kTokenCount; ++token) {
    std::array<bool, kExpertCount> seen{};
    for (std::size_t rank = 0; rank < kSelectedExperts; ++rank) {
      const std::size_t offset = token * stride + rank * sizeof(std::int32_t);
      Require(offset + sizeof(std::int32_t) <= topk.size(),
              "top-k payload is truncated");
      std::int32_t expert = -1;
      std::memcpy(&expert, topk.data() + offset, sizeof(expert));
      Require(expert >= 0 && expert < static_cast<std::int32_t>(kExpertCount),
              "top-k expert is out of range");
      Require(!seen[expert], "top-k expert is duplicated within a token");
      seen[expert] = true;
      grouped[expert].push_back(
          {static_cast<std::uint32_t>(token), static_cast<std::uint32_t>(rank)});
    }
  }

  BucketPlan plan;
  plan.bucket_token.reserve(kAssignments);
  plan.bucket_rank.reserve(kAssignments);
  for (std::size_t expert = 0; expert < kExpertCount; ++expert) {
    const auto& source = grouped[expert];
    if (source.empty()) continue;
    const std::size_t bucket_index = BucketIndex(source.size());
    ExpertGroup group;
    group.expert = static_cast<std::uint32_t>(expert);
    group.bucket_m = kLockedBucketM[bucket_index];
    group.assignments.reserve(source.size());
    for (const auto& [token, rank] : source) {
      const std::uint32_t output_index =
          static_cast<std::uint32_t>(plan.bucket_token.size());
      plan.bucket_token.push_back(token);
      plan.bucket_rank.push_back(rank);
      group.assignments.push_back({token, rank, output_index});
    }
    plan.buckets[bucket_index].push_back(std::move(group));
    ++plan.active_experts;
    plan.padded_assignments += kLockedBucketM[bucket_index];
    plan.max_group_m = std::max(plan.max_group_m, source.size());
  }
  Require(plan.bucket_token.size() == kAssignments,
          "bucket assignment count mismatch");
  if (enforce_locked_layer27) {
    Require(plan.active_experts == 222 && plan.padded_assignments == 12352,
            "locked layer-27 expert schedule changed");
    for (std::size_t index = 0; index < plan.buckets.size(); ++index) {
      Require(plan.buckets[index].size() ==
                  static_cast<std::size_t>(kLockedBucketExperts[index]),
              "locked layer-27 bucket histogram changed");
    }
  }
  return plan;
}

template <typename Value>
void WriteMemory(const std::vector<Value>& source,
                 const dnnl::memory& destination) {
  Require(source.size() * sizeof(Value) == destination.get_desc().get_size(),
          "oneDNN memory size does not match host payload");
  void* mapped = destination.map_data();
  Require(mapped != nullptr, "oneDNN returned a null mapped pointer");
  std::memcpy(mapped, source.data(), source.size() * sizeof(Value));
  destination.unmap_data(mapped);
}

dnnl::matmul::primitive_desc MainPrimitiveDescriptor(
    const dnnl::engine& engine, const dnnl::memory::desc& source,
    const dnnl::memory::desc& weights, const dnnl::memory::desc& destination) {
  dnnl::primitive_attr attributes;
  attributes.set_scales(DNNL_ARG_SRC, 7, {1, 256},
                        dnnl::memory::data_type::f32);
  attributes.set_scales(DNNL_ARG_WEIGHTS, 7, {32, 1},
                        dnnl::memory::data_type::f32);
  return dnnl::matmul::primitive_desc(
      engine, source, weights, destination, attributes);
}

const std::uint8_t* Q4BlockLayout(
    const std::vector<std::uint8_t>& weight, std::size_t expert,
    std::size_t output, std::size_t block, std::size_t rows_per_expert,
    std::size_t blocks_per_row) {
  const std::size_t row = expert * rows_per_expert + output;
  return weight.data() +
         (row * blocks_per_row + block) * kQ4BlockBytes;
}

const std::uint8_t* Q4Block(const std::vector<std::uint8_t>& weight,
                            std::size_t expert, std::size_t output,
                            std::size_t block) {
  return Q4BlockLayout(weight, expert, output, block, kRowsPerExpert,
                       kBlocksPerRow);
}

struct Job {
  int m = 0;
  int experts = 0;
  std::vector<std::uint32_t> slot_to_bucket;
  std::vector<std::int32_t> slot_to_token;
  std::unique_ptr<dnnl::memory> source;
  std::unique_ptr<dnnl::memory> weights;
  std::unique_ptr<dnnl::memory> main_destination;
  std::unique_ptr<dnnl::memory> source_scales;
  std::unique_ptr<dnnl::memory> weight_scales;
  std::unique_ptr<dnnl::memory> min_source;
  std::unique_ptr<dnnl::memory> min_weights;
  std::unique_ptr<dnnl::memory> min_destination;
  std::unique_ptr<dnnl::memory> output;
  std::unique_ptr<dnnl::memory> bucket_map;
  std::unique_ptr<dnnl::memory> token_map;
  std::unique_ptr<dnnl::matmul> main_primitive;
  std::unique_ptr<dnnl::matmul> min_primitive;
  std::unordered_map<int, dnnl::memory> main_arguments;
  std::unordered_map<int, dnnl::memory> min_arguments;
  std::string main_implementation;
  std::string min_implementation;
  cl_kernel swiglu_kernel = nullptr;
  cl_kernel gather_kernel = nullptr;
  cl_kernel swiglu_quantize_kernel = nullptr;
  cl_kernel down_finalize_kernel = nullptr;
  std::uint64_t repacked_q4_code_count = 0;
  std::uint64_t repack_mismatch_count = 0;

  Job(const dnnl::engine& engine, int bucket_m,
      const std::vector<ExpertGroup>& groups, const Q8Data& q8,
      const std::vector<std::uint8_t>& q4, bool preload_q8 = true)
      : m(bucket_m), experts(static_cast<int>(groups.size())) {
    Require(experts > 0, "empty bucket job");
    using data_type = dnnl::memory::data_type;
    using format_tag = dnnl::memory::format_tag;
    const auto source_desc = dnnl::memory::desc(
        {experts, m, static_cast<int>(kHiddenSize)}, data_type::s8,
        format_tag::abc);
    const auto weights_desc = dnnl::memory::desc(
        {experts, static_cast<int>(kHiddenSize), static_cast<int>(kGateUpSize)},
        data_type::u4, format_tag::abc);
    const auto main_desc = dnnl::memory::desc(
        {experts, m, static_cast<int>(kGateUpSize)}, data_type::f32,
        format_tag::abc);
    const auto source_scales_desc = dnnl::memory::desc(
        {experts, m, static_cast<int>(kQ8ScaleGroups)}, data_type::f32,
        format_tag::abc);
    const auto weight_scales_desc = dnnl::memory::desc(
        {experts, static_cast<int>(kQ4ScaleGroups),
         static_cast<int>(kGateUpSize)},
        data_type::f32, format_tag::abc);
    const auto min_source_desc = dnnl::memory::desc(
        {experts, m, static_cast<int>(kQ4ScaleGroups)}, data_type::f32,
        format_tag::abc);
    const auto output_desc = dnnl::memory::desc(
        {experts, m, static_cast<int>(kIntermediateSize)}, data_type::f32,
        format_tag::abc);
    const auto map_desc = dnnl::memory::desc(
        {experts * m}, data_type::s32, format_tag::a);

    source = std::make_unique<dnnl::memory>(source_desc, engine);
    weights = std::make_unique<dnnl::memory>(weights_desc, engine);
    main_destination = std::make_unique<dnnl::memory>(main_desc, engine);
    source_scales = std::make_unique<dnnl::memory>(source_scales_desc, engine);
    weight_scales = std::make_unique<dnnl::memory>(weight_scales_desc, engine);
    min_source = std::make_unique<dnnl::memory>(min_source_desc, engine);
    min_weights = std::make_unique<dnnl::memory>(weight_scales_desc, engine);
    min_destination = std::make_unique<dnnl::memory>(main_desc, engine);
    output = std::make_unique<dnnl::memory>(output_desc, engine);
    bucket_map = std::make_unique<dnnl::memory>(map_desc, engine);
    token_map = std::make_unique<dnnl::memory>(map_desc, engine);

    const auto main_pd = MainPrimitiveDescriptor(
        engine, source_desc, weights_desc, main_desc);
    const auto min_pd = dnnl::matmul::primitive_desc(
        engine, min_source_desc, weight_scales_desc, main_desc);
    main_implementation = main_pd.impl_info_str();
    min_implementation = min_pd.impl_info_str();
    main_primitive = std::make_unique<dnnl::matmul>(main_pd);
    min_primitive = std::make_unique<dnnl::matmul>(min_pd);

    main_arguments = {
        {DNNL_ARG_SRC, *source},
        {DNNL_ARG_WEIGHTS, *weights},
        {DNNL_ARG_DST, *main_destination},
        {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, *source_scales},
        {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, *weight_scales},
    };
    min_arguments = {
        {DNNL_ARG_SRC, *min_source},
        {DNNL_ARG_WEIGHTS, *min_weights},
        {DNNL_ARG_DST, *min_destination},
    };

    std::vector<std::int8_t> source_host(
        static_cast<std::size_t>(experts) * m * kHiddenSize, 0);
    std::vector<float> source_scales_host(
        static_cast<std::size_t>(experts) * m * kQ8ScaleGroups, 0.0f);
    std::vector<float> min_source_host(
        static_cast<std::size_t>(experts) * m * kQ4ScaleGroups, 0.0f);
    slot_to_bucket.assign(
        static_cast<std::size_t>(experts) * m,
        std::numeric_limits<std::uint32_t>::max());
    slot_to_token.assign(static_cast<std::size_t>(experts) * m, -1);
    for (int batch = 0; batch < experts; ++batch) {
      const ExpertGroup& group = groups[batch];
      for (std::size_t row = 0; row < group.assignments.size(); ++row) {
        const Assignment& assignment = group.assignments[row];
        const std::size_t slot = static_cast<std::size_t>(batch) * m + row;
        slot_to_bucket[slot] = assignment.bucket_index;
        slot_to_token[slot] = static_cast<std::int32_t>(assignment.token);
        if (preload_q8) {
          std::memcpy(source_host.data() + slot * kHiddenSize,
                      q8.values.data() + assignment.token * kHiddenSize,
                      kHiddenSize * sizeof(std::int8_t));
          std::memcpy(source_scales_host.data() + slot * kQ8ScaleGroups,
                      q8.scales.data() + assignment.token * kQ8ScaleGroups,
                      kQ8ScaleGroups * sizeof(float));
          std::memcpy(min_source_host.data() + slot * kQ4ScaleGroups,
                      q8.sums32_scaled.data() +
                          assignment.token * kQ4ScaleGroups,
                      kQ4ScaleGroups * sizeof(float));
        }
      }
    }

    const std::size_t logical_weight_values =
        static_cast<std::size_t>(experts) * kHiddenSize * kGateUpSize;
    std::vector<std::uint8_t> weights_host(logical_weight_values / 2, 0);
    std::vector<float> weight_scales_host(
        static_cast<std::size_t>(experts) * kQ4ScaleGroups * kGateUpSize,
        0.0f);
    std::vector<float> min_weights_host(weight_scales_host.size(), 0.0f);
    for (int batch = 0; batch < experts; ++batch) {
      const std::size_t expert = groups[batch].expert;
      for (std::size_t output_pair = 0; output_pair < kGateUpSize;
           output_pair += 2) {
        for (std::size_t block = 0; block < kBlocksPerRow; ++block) {
          const std::uint8_t* block0 = Q4Block(q4, expert, output_pair, block);
          const std::uint8_t* block1 =
              Q4Block(q4, expert, output_pair + 1, block);
          const std::uint8_t* qs0 = block0 + 16;
          const std::uint8_t* qs1 = block1 + 16;
          for (std::size_t segment = 0; segment < 4; ++segment) {
            for (std::size_t offset = 0; offset < 32; ++offset) {
              const std::uint8_t packed0 = qs0[segment * 32 + offset];
              const std::uint8_t packed1 = qs1[segment * 32 + offset];
              const std::size_t k0 = block * 256 + segment * 64 + offset;
              const std::size_t k1 = k0 + 32;
              const std::size_t destination0 =
                  ((static_cast<std::size_t>(batch) * kHiddenSize + k0) *
                       kGateUpSize +
                   output_pair) /
                  2;
              const std::size_t destination1 =
                  ((static_cast<std::size_t>(batch) * kHiddenSize + k1) *
                       kGateUpSize +
                   output_pair) /
                  2;
              const std::uint8_t repacked0 =
                  static_cast<std::uint8_t>((packed0 & 0x0fU) |
                                            ((packed1 & 0x0fU) << 4));
              const std::uint8_t repacked1 =
                  static_cast<std::uint8_t>((packed0 >> 4) |
                                            ((packed1 >> 4) << 4));
              weights_host[destination0] = repacked0;
              weights_host[destination1] = repacked1;
              repacked_q4_code_count += 4;
              repack_mismatch_count +=
                  static_cast<std::uint64_t>((repacked0 & 0x0fU) !=
                                             (packed0 & 0x0fU));
              repack_mismatch_count +=
                  static_cast<std::uint64_t>((repacked0 >> 4) !=
                                             (packed1 & 0x0fU));
              repack_mismatch_count +=
                  static_cast<std::uint64_t>((repacked1 & 0x0fU) !=
                                             (packed0 >> 4));
              repack_mismatch_count +=
                  static_cast<std::uint64_t>((repacked1 >> 4) !=
                                             (packed1 >> 4));
            }
          }
        }
      }
      for (std::size_t output_index = 0; output_index < kGateUpSize;
           ++output_index) {
        for (std::size_t block = 0; block < kBlocksPerRow; ++block) {
          const std::uint8_t* q4_block =
              Q4Block(q4, expert, output_index, block);
          const float d = HalfToFloat(LoadU16(q4_block));
          const float dmin = HalfToFloat(LoadU16(q4_block + 2));
          const std::uint8_t* scales = q4_block + 4;
          for (std::size_t group = 0; group < 8; ++group) {
            const std::size_t scale_group = block * 8 + group;
            const std::size_t destination =
                (static_cast<std::size_t>(batch) * kQ4ScaleGroups +
                 scale_group) *
                    kGateUpSize +
                output_index;
            weight_scales_host[destination] =
                d * static_cast<float>(GetScale(group, scales));
            min_weights_host[destination] =
                dmin * static_cast<float>(GetMinimum(group, scales));
          }
        }
      }
    }

    WriteMemory(source_host, *source);
    WriteMemory(weights_host, *weights);
    WriteMemory(source_scales_host, *source_scales);
    WriteMemory(weight_scales_host, *weight_scales);
    WriteMemory(min_source_host, *min_source);
    WriteMemory(min_weights_host, *min_weights);
    std::vector<float> zeros_main(
        static_cast<std::size_t>(experts) * m * kGateUpSize, 0.0f);
    std::vector<float> zeros_output(
        static_cast<std::size_t>(experts) * m * kIntermediateSize, 0.0f);
    WriteMemory(zeros_main, *main_destination);
    WriteMemory(zeros_main, *min_destination);
    WriteMemory(zeros_output, *output);
    std::vector<std::int32_t> bucket_map_host(slot_to_bucket.size(), -1);
    for (std::size_t index = 0; index < slot_to_bucket.size(); ++index) {
      if (slot_to_bucket[index] != std::numeric_limits<std::uint32_t>::max()) {
        bucket_map_host[index] = static_cast<std::int32_t>(slot_to_bucket[index]);
      }
    }
    WriteMemory(bucket_map_host, *bucket_map);
    WriteMemory(slot_to_token, *token_map);
  }

  ~Job() {
    if (swiglu_kernel != nullptr) clReleaseKernel(swiglu_kernel);
    if (gather_kernel != nullptr) clReleaseKernel(gather_kernel);
    if (swiglu_quantize_kernel != nullptr) {
      clReleaseKernel(swiglu_quantize_kernel);
    }
    if (down_finalize_kernel != nullptr) clReleaseKernel(down_finalize_kernel);
  }

  std::size_t RowCount() const {
    return static_cast<std::size_t>(experts) * m;
  }
};

struct DownJob {
  int m;
  int experts;
  std::unique_ptr<dnnl::memory> source;
  std::unique_ptr<dnnl::memory> weights;
  std::unique_ptr<dnnl::memory> main_destination;
  std::unique_ptr<dnnl::memory> source_scales;
  std::unique_ptr<dnnl::memory> weight_scales;
  std::unique_ptr<dnnl::memory> min_source;
  std::unique_ptr<dnnl::memory> min_weights;
  std::unique_ptr<dnnl::memory> min_destination;
  std::unique_ptr<dnnl::matmul> main_primitive;
  std::unique_ptr<dnnl::matmul> min_primitive;
  std::unordered_map<int, dnnl::memory> main_arguments;
  std::unordered_map<int, dnnl::memory> min_arguments;
  std::string main_implementation;
  std::string min_implementation;
  std::uint64_t repacked_q4_code_count = 0;
  std::uint64_t repack_mismatch_count = 0;

  DownJob(const dnnl::engine& engine, int bucket_m,
          const std::vector<ExpertGroup>& groups,
          const std::vector<std::uint8_t>& q4)
      : m(bucket_m), experts(static_cast<int>(groups.size())) {
    using dt = dnnl::memory::data_type;
    using tag = dnnl::memory::format_tag;
    const auto src_md = dnnl::memory::desc(
        {experts, m, static_cast<int>(kDownInputSize)}, dt::s8, tag::abc);
    const auto wei_md = dnnl::memory::desc(
        {experts, static_cast<int>(kDownInputSize),
         static_cast<int>(kDownOutputSize)}, dt::u4, tag::abc);
    const auto dst_md = dnnl::memory::desc(
        {experts, m, static_cast<int>(kDownOutputSize)}, dt::f32, tag::abc);
    const auto src_scale_md = dnnl::memory::desc(
        {experts, m, static_cast<int>(kDownBlocksPerRow)}, dt::f32, tag::abc);
    const auto wei_scale_md = dnnl::memory::desc(
        {experts, static_cast<int>(kDownScaleGroups),
         static_cast<int>(kDownOutputSize)}, dt::f32, tag::abc);
    const auto min_src_md = dnnl::memory::desc(
        {experts, m, static_cast<int>(kDownScaleGroups)}, dt::f32, tag::abc);
    source = std::make_unique<dnnl::memory>(src_md, engine);
    weights = std::make_unique<dnnl::memory>(wei_md, engine);
    main_destination = std::make_unique<dnnl::memory>(dst_md, engine);
    source_scales = std::make_unique<dnnl::memory>(src_scale_md, engine);
    weight_scales = std::make_unique<dnnl::memory>(wei_scale_md, engine);
    min_source = std::make_unique<dnnl::memory>(min_src_md, engine);
    min_weights = std::make_unique<dnnl::memory>(wei_scale_md, engine);
    min_destination = std::make_unique<dnnl::memory>(dst_md, engine);
    const auto main_pd = MainPrimitiveDescriptor(engine, src_md, wei_md, dst_md);
    const auto min_pd = dnnl::matmul::primitive_desc(
        engine, min_src_md, wei_scale_md, dst_md);
    main_implementation = main_pd.impl_info_str();
    min_implementation = min_pd.impl_info_str();
    main_primitive = std::make_unique<dnnl::matmul>(main_pd);
    min_primitive = std::make_unique<dnnl::matmul>(min_pd);
    main_arguments = {{DNNL_ARG_SRC, *source},
                      {DNNL_ARG_WEIGHTS, *weights},
                      {DNNL_ARG_DST, *main_destination},
                      {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, *source_scales},
                      {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, *weight_scales}};
    min_arguments = {{DNNL_ARG_SRC, *min_source},
                     {DNNL_ARG_WEIGHTS, *min_weights},
                     {DNNL_ARG_DST, *min_destination}};

    const std::size_t logical =
        static_cast<std::size_t>(experts) * kDownInputSize * kDownOutputSize;
    std::vector<std::uint8_t> packed(logical / 2, 0);
    std::vector<float> scales(
        static_cast<std::size_t>(experts) * kDownScaleGroups * kDownOutputSize);
    std::vector<float> mins(scales.size());
    for (int batch = 0; batch < experts; ++batch) {
      const std::size_t expert = groups[batch].expert;
      for (std::size_t out = 0; out < kDownOutputSize; out += 2) {
        for (std::size_t block = 0; block < kDownBlocksPerRow; ++block) {
          const auto* b0 = Q4BlockLayout(
              q4, expert, out, block, kDownOutputSize, kDownBlocksPerRow);
          const auto* b1 = Q4BlockLayout(
              q4, expert, out + 1, block, kDownOutputSize, kDownBlocksPerRow);
          for (std::size_t segment = 0; segment < 4; ++segment) {
            for (std::size_t offset = 0; offset < 32; ++offset) {
              const std::uint8_t p0 = b0[16 + segment * 32 + offset];
              const std::uint8_t p1 = b1[16 + segment * 32 + offset];
              for (std::size_t high = 0; high < 2; ++high) {
                const std::size_t k =
                    block * 256 + segment * 64 + offset + high * 32;
                const std::size_t dst =
                    ((static_cast<std::size_t>(batch) * kDownInputSize + k) *
                         kDownOutputSize + out) / 2;
                const std::uint8_t value = high == 0
                    ? static_cast<std::uint8_t>((p0 & 15U) | ((p1 & 15U) << 4))
                    : static_cast<std::uint8_t>((p0 >> 4) | ((p1 >> 4) << 4));
                packed[dst] = value;
                repacked_q4_code_count += 2;
                const std::uint8_t expected0 =
                    high == 0 ? p0 & 15U : p0 >> 4;
                const std::uint8_t expected1 =
                    high == 0 ? p1 & 15U : p1 >> 4;
                repack_mismatch_count += static_cast<std::uint64_t>(
                    (value & 15U) != expected0);
                repack_mismatch_count += static_cast<std::uint64_t>(
                    (value >> 4) != expected1);
              }
            }
          }
        }
      }
      for (std::size_t out = 0; out < kDownOutputSize; ++out) {
        for (std::size_t block = 0; block < kDownBlocksPerRow; ++block) {
          const auto* q = Q4BlockLayout(
              q4, expert, out, block, kDownOutputSize, kDownBlocksPerRow);
          const float d = HalfToFloat(LoadU16(q));
          const float dmin = HalfToFloat(LoadU16(q + 2));
          for (std::size_t group = 0; group < 8; ++group) {
            const std::size_t index =
                ((static_cast<std::size_t>(batch) * kDownScaleGroups +
                  block * 8 + group) * kDownOutputSize) + out;
            scales[index] = d * static_cast<float>(GetScale(group, q + 4));
            mins[index] = dmin * static_cast<float>(GetMinimum(group, q + 4));
          }
        }
      }
    }
    WriteMemory(packed, *weights);
    WriteMemory(scales, *weight_scales);
    WriteMemory(mins, *min_weights);
    WriteMemory(std::vector<std::int8_t>(
                    static_cast<std::size_t>(experts) * m * kDownInputSize),
                *source);
    WriteMemory(std::vector<float>(
                    static_cast<std::size_t>(experts) * m * kDownBlocksPerRow),
                *source_scales);
    WriteMemory(std::vector<float>(
                    static_cast<std::size_t>(experts) * m * kDownScaleGroups),
                *min_source);
    const std::vector<float> zeros(
        static_cast<std::size_t>(experts) * m * kDownOutputSize);
    WriteMemory(zeros, *main_destination);
    WriteMemory(zeros, *min_destination);
  }

  std::size_t RowCount() const {
    return static_cast<std::size_t>(experts) * m;
  }
};

std::string ProgramBuildLog(cl_program program, cl_device_id device) {
  std::size_t size = 0;
  clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &size);
  std::string log(size, '\0');
  if (size != 0) {
    clGetProgramBuildInfo(
        program, device, CL_PROGRAM_BUILD_LOG, size, log.data(), nullptr);
  }
  while (!log.empty() && log.back() == '\0') log.pop_back();
  return log;
}

cl_program BuildProgram(cl_context context, cl_device_id device,
                        std::string* build_log) {
  const char* source = kSwiGluSource;
  const std::size_t length = std::strlen(source);
  cl_int status = CL_SUCCESS;
  cl_program program =
      clCreateProgramWithSource(context, 1, &source, &length, &status);
  CheckCl(status, "clCreateProgramWithSource");
  status = clBuildProgram(program, 1, &device, "-cl-std=CL2.0", nullptr, nullptr);
  *build_log = ProgramBuildLog(program, device);
  if (status != CL_SUCCESS) {
    clReleaseProgram(program);
    Fail("clBuildProgram failed: " + *build_log);
  }
  return program;
}

void BindSwiGluKernel(Job& job, cl_program program) {
  cl_int status = CL_SUCCESS;
  job.swiglu_kernel =
      clCreateKernel(program, "q4k_compensate_swiglu", &status);
  CheckCl(status, "clCreateKernel q4k_compensate_swiglu");
  cl_mem main_buffer = dnnl::ocl_interop::get_mem_object(*job.main_destination);
  cl_mem min_buffer = dnnl::ocl_interop::get_mem_object(*job.min_destination);
  cl_mem output_buffer = dnnl::ocl_interop::get_mem_object(*job.output);
  const cl_uint row_count = static_cast<cl_uint>(job.RowCount());
  CheckCl(clSetKernelArg(job.swiglu_kernel, 0, sizeof(main_buffer), &main_buffer),
          "clSetKernelArg main");
  CheckCl(clSetKernelArg(job.swiglu_kernel, 1, sizeof(min_buffer), &min_buffer),
          "clSetKernelArg min");
  CheckCl(clSetKernelArg(job.swiglu_kernel, 2, sizeof(output_buffer), &output_buffer),
          "clSetKernelArg output");
  CheckCl(clSetKernelArg(job.swiglu_kernel, 3, sizeof(row_count), &row_count),
          "clSetKernelArg rows");
}

cl_kernel CreateKernel(cl_program program, const char* name) {
  cl_int status = CL_SUCCESS;
  cl_kernel kernel = clCreateKernel(program, name, &status);
  CheckCl(status, std::string("clCreateKernel ") + name);
  return kernel;
}

void BindRoutedKernels(
    Job& gate, DownJob& down, cl_program program,
    const dnnl::memory& input, const dnnl::memory& bucket_weights,
    const dnnl::memory& contributions) {
  const auto mem = [](const dnnl::memory& value) {
    return dnnl::ocl_interop::get_mem_object(value);
  };
  const cl_uint rows = static_cast<cl_uint>(gate.RowCount());
  gate.gather_kernel = CreateKernel(program, "q4k_gather_quantize_input");
  std::array<cl_mem, 5> gather = {
      mem(input), mem(*gate.token_map), mem(*gate.source),
      mem(*gate.source_scales), mem(*gate.min_source)};
  for (cl_uint index = 0; index < gather.size(); ++index) {
    CheckCl(clSetKernelArg(gate.gather_kernel, index, sizeof(cl_mem),
                           &gather[index]), "clSetKernelArg gather");
  }
  CheckCl(clSetKernelArg(gate.gather_kernel, 5, sizeof(rows), &rows),
          "clSetKernelArg gather rows");

  gate.swiglu_quantize_kernel =
      CreateKernel(program, "q4k_compensate_swiglu_quantize");
  std::array<cl_mem, 6> swiglu = {
      mem(*gate.main_destination), mem(*gate.min_destination),
      mem(*gate.output), mem(*down.source), mem(*down.source_scales),
      mem(*down.min_source)};
  for (cl_uint index = 0; index < swiglu.size(); ++index) {
    CheckCl(clSetKernelArg(gate.swiglu_quantize_kernel, index, sizeof(cl_mem),
                           &swiglu[index]), "clSetKernelArg swiglu quantize");
  }
  CheckCl(clSetKernelArg(gate.swiglu_quantize_kernel, 6, sizeof(rows), &rows),
          "clSetKernelArg swiglu quantize rows");

  gate.down_finalize_kernel =
      CreateKernel(program, "q4k_down_compensate_weight");
  std::array<cl_mem, 5> finalize = {
      mem(*down.main_destination), mem(*down.min_destination),
      mem(*gate.bucket_map), mem(bucket_weights), mem(contributions)};
  for (cl_uint index = 0; index < finalize.size(); ++index) {
    CheckCl(clSetKernelArg(gate.down_finalize_kernel, index, sizeof(cl_mem),
                           &finalize[index]), "clSetKernelArg down finalize");
  }
  CheckCl(clSetKernelArg(gate.down_finalize_kernel, 5, sizeof(rows), &rows),
          "clSetKernelArg down finalize rows");
}

void ExecuteJobs(std::vector<std::unique_ptr<Job>>& jobs,
                 dnnl::stream& stream, cl_command_queue queue) {
  constexpr std::size_t local = 256;
  for (auto& job : jobs) {
    job->main_primitive->execute(stream, job->main_arguments);
    job->min_primitive->execute(stream, job->min_arguments);
    const std::size_t values = job->RowCount() * kIntermediateSize;
    const std::size_t global = (values + local - 1) / local * local;
    CheckCl(clEnqueueNDRangeKernel(queue, job->swiglu_kernel, 1, nullptr,
                                   &global, &local, 0, nullptr, nullptr),
            "clEnqueueNDRangeKernel q4k_compensate_swiglu");
  }
  CheckCl(clFinish(queue), "clFinish component");
}

void ExecuteRouted(
    std::vector<std::unique_ptr<Job>>& gates,
    std::vector<std::unique_ptr<DownJob>>& downs, dnnl::stream& stream,
    cl_command_queue queue, cl_kernel scatter_kernel) {
  constexpr std::size_t local = 256;
  for (std::size_t index = 0; index < gates.size(); ++index) {
    Job& gate = *gates[index];
    DownJob& down = *downs[index];
    const std::size_t rows = gate.RowCount();
    std::size_t global = rows * 8 * local;
    CheckCl(clEnqueueNDRangeKernel(queue, gate.gather_kernel, 1, nullptr,
                                   &global, &local, 0, nullptr, nullptr),
            "clEnqueueNDRangeKernel gather");
    gate.main_primitive->execute(stream, gate.main_arguments);
    gate.min_primitive->execute(stream, gate.min_arguments);
    global = rows * 2 * local;
    CheckCl(clEnqueueNDRangeKernel(queue, gate.swiglu_quantize_kernel, 1,
                                   nullptr, &global, &local, 0, nullptr,
                                   nullptr), "clEnqueueNDRangeKernel swiglu quantize");
    down.main_primitive->execute(stream, down.main_arguments);
    down.min_primitive->execute(stream, down.min_arguments);
    global = (rows * kDownOutputSize + local - 1) / local * local;
    CheckCl(clEnqueueNDRangeKernel(queue, gate.down_finalize_kernel, 1,
                                   nullptr, &global, &local, 0, nullptr,
                                   nullptr), "clEnqueueNDRangeKernel down finalize");
  }
  const std::size_t global = kTokenCount * kHiddenSize;
  CheckCl(clEnqueueNDRangeKernel(queue, scatter_kernel, 1, nullptr, &global,
                                 &local, 0, nullptr, nullptr),
          "clEnqueueNDRangeKernel scatter");
  CheckCl(clFinish(queue), "clFinish routed component");
}

struct CompareStats {
  bool finite = true;
  std::size_t count = 0;
  std::size_t mismatch_count = 0;
  double max_abs = 0.0;
  double mean_abs = 0.0;
  double rmse = 0.0;
  double cosine = 0.0;
};

CompareStats Compare(const std::vector<float>& output,
                     const std::vector<float>& oracle,
                     const BucketPlan& plan) {
  Require(output.size() == kAssignments * kIntermediateSize,
          "output size mismatch");
  long double absolute_sum = 0.0L;
  long double square_sum = 0.0L;
  long double dot = 0.0L;
  long double norm_output = 0.0L;
  long double norm_oracle = 0.0L;
  CompareStats stats;
  stats.count = output.size();
  for (std::size_t bucket = 0; bucket < kAssignments; ++bucket) {
    const std::size_t token = plan.bucket_token[bucket];
    const std::size_t rank = plan.bucket_rank[bucket];
    const std::size_t oracle_base =
        (token * kSelectedExperts + rank) * kIntermediateSize;
    const std::size_t output_base = bucket * kIntermediateSize;
    for (std::size_t inner = 0; inner < kIntermediateSize; ++inner) {
      const double actual = output[output_base + inner];
      const double expected = oracle[oracle_base + inner];
      if (!std::isfinite(actual) || !std::isfinite(expected)) stats.finite = false;
      const double difference = actual - expected;
      const double absolute = std::fabs(difference);
      stats.max_abs = std::max(stats.max_abs, absolute);
      absolute_sum += absolute;
      square_sum += difference * difference;
      dot += actual * expected;
      norm_output += actual * actual;
      norm_oracle += expected * expected;
      if (absolute > kMismatchThreshold) ++stats.mismatch_count;
    }
  }
  stats.mean_abs = static_cast<double>(absolute_sum / stats.count);
  stats.rmse = std::sqrt(static_cast<double>(square_sum / stats.count));
  const long double norm = std::sqrt(norm_output * norm_oracle);
  stats.cosine = norm == 0.0L ? 1.0 : static_cast<double>(dot / norm);
  return stats;
}

CompareStats CompareFlat(const std::vector<float>& output,
                         const std::vector<float>& oracle) {
  Require(output.size() == oracle.size(), "flat comparison size mismatch");
  long double absolute_sum = 0.0L;
  long double square_sum = 0.0L;
  long double dot = 0.0L;
  long double norm_output = 0.0L;
  long double norm_oracle = 0.0L;
  CompareStats stats;
  stats.count = output.size();
  for (std::size_t index = 0; index < output.size(); ++index) {
    const double actual = output[index];
    const double expected = oracle[index];
    if (!std::isfinite(actual) || !std::isfinite(expected)) stats.finite = false;
    const double difference = actual - expected;
    const double absolute = std::fabs(difference);
    stats.max_abs = std::max(stats.max_abs, absolute);
    absolute_sum += absolute;
    square_sum += difference * difference;
    dot += actual * expected;
    norm_output += actual * actual;
    norm_oracle += expected * expected;
    if (absolute > kMismatchThreshold) ++stats.mismatch_count;
  }
  stats.mean_abs = static_cast<double>(absolute_sum / stats.count);
  stats.rmse = std::sqrt(static_cast<double>(square_sum / stats.count));
  const long double norm = std::sqrt(norm_output * norm_oracle);
  stats.cosine = norm == 0.0L ? 1.0 : static_cast<double>(dot / norm);
  return stats;
}

bool ComparePass(const CompareStats& stats) {
  return stats.finite && stats.mismatch_count == 0 &&
         stats.max_abs <= kMismatchThreshold && stats.rmse <= 5e-4 &&
         stats.cosine >= 0.999;
}

void PrintCompare(const char* name, const CompareStats& stats) {
  std::cout << "\"" << name << "\":{\"compared_value_count\":"
            << stats.count << ",\"cosine\":" << stats.cosine
            << ",\"finite\":" << stats.finite
            << ",\"max_abs_diff\":" << stats.max_abs
            << ",\"mean_abs_diff\":" << stats.mean_abs
            << ",\"mismatch_count\":" << stats.mismatch_count
            << ",\"rmse\":" << stats.rmse << "},";
}

std::string JsonEscape(const std::string& value) {
  std::string escaped;
  for (const char character : value) {
    if (character == '\\' || character == '"') escaped.push_back('\\');
    if (character == '\n') escaped += "\\n";
    else if (character != '\r') escaped.push_back(character);
  }
  return escaped;
}

}  // namespace

#ifndef IQ36_Q4K_COMPONENT_NO_MAIN
int main(int argc, char** argv) {
  try {
    const Args args = ParseArgs(argc, argv);
    const bool routed = RoutedMode(args);
    const auto input = ReadVector<float>(args.input, kTokenCount * kHiddenSize);
    const auto topk = ReadBytes(args.topk);
    const auto oracle =
        ReadVector<float>(args.oracle, kAssignments * kIntermediateSize);
    const auto q4 = ReadModelSlice(
        args.model, args.weight_offset, args.weight_bytes);
    const auto down_q4 = routed
        ? ReadModelSlice(args.model, args.down_weight_offset,
                         args.down_weight_bytes)
        : std::vector<std::uint8_t>{};
    const auto router_weights = routed
        ? ReadVector<float>(args.router_weights, kAssignments)
        : std::vector<float>{};
    const auto down_oracle = routed
        ? ReadVector<float>(args.down_oracle, kAssignments * kDownOutputSize)
        : std::vector<float>{};
    const auto moe_oracle = routed
        ? ReadVector<float>(args.moe_oracle, kTokenCount * kHiddenSize)
        : std::vector<float>{};
    const Q8Data q8 = routed ? Q8Data{} : QuantizeQ8K(input);
    const BucketPlan plan = BuildPlan(topk, args.topk_stride);

    dnnl::engine engine(dnnl::engine::kind::gpu, 0);
    dnnl::stream stream(engine);
    const cl_context context = dnnl::ocl_interop::get_context(engine);
    const cl_device_id device = dnnl::ocl_interop::get_device(engine);
    const cl_command_queue queue = dnnl::ocl_interop::get_command_queue(stream);

    std::vector<std::unique_ptr<Job>> jobs;
    jobs.reserve(kLockedBucketM.size());
    for (std::size_t index = 0; index < kLockedBucketM.size(); ++index) {
      jobs.emplace_back(std::make_unique<Job>(
          engine, kLockedBucketM[index], plan.buckets[index], q8, q4,
          !routed));
    }
    std::vector<std::unique_ptr<DownJob>> down_jobs;
    if (routed) {
      down_jobs.reserve(kLockedBucketM.size());
      for (std::size_t index = 0; index < kLockedBucketM.size(); ++index) {
        down_jobs.emplace_back(std::make_unique<DownJob>(
            engine, kLockedBucketM[index], plan.buckets[index], down_q4));
      }
    }

    using dt = dnnl::memory::data_type;
    using tag = dnnl::memory::format_tag;
    std::unique_ptr<dnnl::memory> input_memory;
    std::unique_ptr<dnnl::memory> bucket_weight_memory;
    std::unique_ptr<dnnl::memory> inverse_map_memory;
    std::unique_ptr<dnnl::memory> contribution_memory;
    std::unique_ptr<dnnl::memory> routed_output_memory;
    if (routed) {
      input_memory = std::make_unique<dnnl::memory>(dnnl::memory::desc(
          {static_cast<int>(kTokenCount), static_cast<int>(kHiddenSize)},
          dt::f32, tag::ab), engine);
      bucket_weight_memory = std::make_unique<dnnl::memory>(
          dnnl::memory::desc({static_cast<int>(kAssignments)}, dt::f32, tag::a),
          engine);
      inverse_map_memory = std::make_unique<dnnl::memory>(
          dnnl::memory::desc({static_cast<int>(kAssignments)}, dt::s32, tag::a),
          engine);
      contribution_memory = std::make_unique<dnnl::memory>(dnnl::memory::desc(
          {static_cast<int>(kAssignments), static_cast<int>(kHiddenSize)},
          dt::f32, tag::ab), engine);
      routed_output_memory = std::make_unique<dnnl::memory>(dnnl::memory::desc(
          {static_cast<int>(kTokenCount), static_cast<int>(kHiddenSize)},
          dt::f32, tag::ab), engine);
      std::vector<float> bucket_weights(kAssignments);
      std::vector<std::int32_t> inverse_map(kAssignments, -1);
      for (std::size_t bucket = 0; bucket < kAssignments; ++bucket) {
        const std::size_t token = plan.bucket_token[bucket];
        const std::size_t rank = plan.bucket_rank[bucket];
        bucket_weights[bucket] =
            router_weights[token * kSelectedExperts + rank];
        inverse_map[token * kSelectedExperts + rank] =
            static_cast<std::int32_t>(bucket);
      }
      Require(std::none_of(inverse_map.begin(), inverse_map.end(),
                           [](std::int32_t value) { return value < 0; }),
              "token/rank inverse map is incomplete");
      WriteMemory(input, *input_memory);
      WriteMemory(bucket_weights, *bucket_weight_memory);
      WriteMemory(inverse_map, *inverse_map_memory);
      WriteMemory(std::vector<float>(kAssignments * kHiddenSize, 0.0f),
                  *contribution_memory);
      WriteMemory(std::vector<float>(kTokenCount * kHiddenSize, 0.0f),
                  *routed_output_memory);
    }

    std::string build_log;
    cl_program program = BuildProgram(context, device, &build_log);
    cl_kernel scatter_kernel = nullptr;
    if (routed) {
      for (std::size_t index = 0; index < jobs.size(); ++index) {
        BindRoutedKernels(*jobs[index], *down_jobs[index], program,
                          *input_memory, *bucket_weight_memory,
                          *contribution_memory);
      }
      scatter_kernel = CreateKernel(program, "q4k_scatter_routed_output");
      std::array<cl_mem, 3> scatter = {
          dnnl::ocl_interop::get_mem_object(*contribution_memory),
          dnnl::ocl_interop::get_mem_object(*inverse_map_memory),
          dnnl::ocl_interop::get_mem_object(*routed_output_memory)};
      for (cl_uint index = 0; index < scatter.size(); ++index) {
        CheckCl(clSetKernelArg(scatter_kernel, index, sizeof(cl_mem),
                               &scatter[index]), "clSetKernelArg scatter");
      }
    } else {
      for (auto& job : jobs) BindSwiGluKernel(*job, program);
    }

    const auto execute = [&]() {
      if (routed) ExecuteRouted(jobs, down_jobs, stream, queue, scatter_kernel);
      else ExecuteJobs(jobs, stream, queue);
    };

    for (int iteration = 0; iteration < args.warmup; ++iteration) {
      execute();
    }
    std::vector<double> samples_us;
    samples_us.reserve(args.repeat);
    for (int iteration = 0; iteration < args.repeat; ++iteration) {
      const auto begin = std::chrono::steady_clock::now();
      execute();
      const auto end = std::chrono::steady_clock::now();
      samples_us.push_back(
          std::chrono::duration<double, std::micro>(end - begin).count());
    }

    std::vector<float> output(kAssignments * kIntermediateSize, 0.0f);
    for (const auto& job : jobs) {
      const float* mapped = job->output->map_data<float>();
      Require(mapped != nullptr, "could not map component output");
      for (std::size_t slot = 0; slot < job->slot_to_bucket.size(); ++slot) {
        const std::uint32_t bucket = job->slot_to_bucket[slot];
        if (bucket == std::numeric_limits<std::uint32_t>::max()) continue;
        std::memcpy(output.data() +
                        static_cast<std::size_t>(bucket) * kIntermediateSize,
                    mapped + slot * kIntermediateSize,
                    kIntermediateSize * sizeof(float));
      }
      job->output->unmap_data(const_cast<float*>(mapped));
    }
    const CompareStats compare = Compare(output, oracle, plan);
    CompareStats down_compare;
    CompareStats moe_compare;
    if (routed) {
      std::vector<float> weighted_oracle(
          kAssignments * kDownOutputSize, 0.0f);
      for (std::size_t bucket = 0; bucket < kAssignments; ++bucket) {
        const std::size_t token = plan.bucket_token[bucket];
        const std::size_t rank = plan.bucket_rank[bucket];
        const std::size_t source = token * kSelectedExperts + rank;
        const float weight = router_weights[source];
        for (std::size_t hidden = 0; hidden < kDownOutputSize; ++hidden) {
          weighted_oracle[bucket * kDownOutputSize + hidden] =
              down_oracle[source * kDownOutputSize + hidden] * weight;
        }
      }
      const float* contributions = contribution_memory->map_data<float>();
      Require(contributions != nullptr, "could not map weighted down output");
      std::vector<float> weighted_output(
          contributions, contributions + kAssignments * kDownOutputSize);
      contribution_memory->unmap_data(const_cast<float*>(contributions));
      down_compare = CompareFlat(weighted_output, weighted_oracle);
      const float* mapped = routed_output_memory->map_data<float>();
      Require(mapped != nullptr, "could not map routed output");
      std::vector<float> routed_output(
          mapped, mapped + kTokenCount * kHiddenSize);
      routed_output_memory->unmap_data(const_cast<float*>(mapped));
      moe_compare = CompareFlat(routed_output, moe_oracle);
    }
    std::vector<double> sorted = samples_us;
    std::sort(sorted.begin(), sorted.end());
    const double minimum_us = sorted.front();
    const double median_us = sorted[sorted.size() / 2];
    const double mean_us =
        std::accumulate(samples_us.begin(), samples_us.end(), 0.0) /
        samples_us.size();
    bool implementations_pass = std::all_of(
        jobs.begin(), jobs.end(), [](const std::unique_ptr<Job>& job) {
          return job->main_implementation.find("jit:gemm") != std::string::npos &&
                 job->min_implementation.find("jit:gemm") != std::string::npos;
        });
    if (routed) {
      implementations_pass = implementations_pass && std::all_of(
          down_jobs.begin(), down_jobs.end(),
          [](const std::unique_ptr<DownJob>& job) {
            return job->main_implementation.find("jit:gemm") !=
                       std::string::npos &&
                   job->min_implementation.find("jit:gemm") !=
                       std::string::npos;
          });
    }
    bool correctness_pass = ComparePass(compare) &&
        (!routed || (ComparePass(down_compare) && ComparePass(moe_compare)));
    const bool performance_pass = minimum_us <= args.kernel_cap_us;
    std::uint64_t repacked_q4_code_count = std::accumulate(
        jobs.begin(), jobs.end(), std::uint64_t{0},
        [](std::uint64_t sum, const std::unique_ptr<Job>& job) {
          return sum + job->repacked_q4_code_count;
        });
    std::uint64_t repack_mismatch_count = std::accumulate(
        jobs.begin(), jobs.end(), std::uint64_t{0},
        [](std::uint64_t sum, const std::unique_ptr<Job>& job) {
          return sum + job->repack_mismatch_count;
        });
    if (routed) {
      for (const auto& job : down_jobs) {
        repacked_q4_code_count += job->repacked_q4_code_count;
        repack_mismatch_count += job->repack_mismatch_count;
      }
    }
    const std::uint64_t expected_repacked_q4_code_count = plan.active_experts *
        (kGateUpSize * kHiddenSize +
         (routed ? kDownInputSize * kDownOutputSize : 0));
    const bool repack_pass =
        repacked_q4_code_count == expected_repacked_q4_code_count &&
        repack_mismatch_count == 0;
    correctness_pass = correctness_pass && repack_pass;
    const dnnl_version_t* version = dnnl_version();
    char device_name[256] = {};
    char driver_version[256] = {};
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), device_name, nullptr);
    clGetDeviceInfo(
        device, CL_DRIVER_VERSION, sizeof(driver_version), driver_version, nullptr);

    std::cout << std::boolalpha << std::setprecision(12) << "{";
    std::cout << "\"active_experts\":" << plan.active_experts << ",";
    std::cout << "\"assignment_count\":" << kAssignments << ",";
    std::cout << "\"buckets\":[";
    for (std::size_t index = 0; index < jobs.size(); ++index) {
      if (index != 0) std::cout << ",";
      const Job& job = *jobs[index];
      std::cout << "{\"experts\":" << job.experts
                << ",\"m\":" << job.m
                << ",\"main_implementation\":\""
                << JsonEscape(job.main_implementation)
                << "\",\"min_implementation\":\""
                << JsonEscape(job.min_implementation) << "\"";
      if (routed) {
        std::cout << ",\"down_main_implementation\":\""
                  << JsonEscape(down_jobs[index]->main_implementation)
                  << "\",\"down_min_implementation\":\""
                  << JsonEscape(down_jobs[index]->min_implementation) << "\"";
      }
      std::cout << "}";
    }
    std::cout << "],";
    std::cout << "\"build_log\":\"" << JsonEscape(build_log) << "\",";
    PrintCompare("compare", compare);
    if (routed) {
      PrintCompare("weighted_down_compare", down_compare);
      PrintCompare("moe_compare", moe_compare);
    }
    std::cout << "\"correctness_pass\":" << correctness_pass << ",";
    std::cout << "\"device_name\":\"" << JsonEscape(device_name) << "\",";
    std::cout << "\"driver_version\":\"" << JsonEscape(driver_version)
              << "\",";
    std::cout << "\"implementations_pass\":" << implementations_pass << ",";
    std::cout << "\"kernel_cap_us\":" << args.kernel_cap_us << ",";
    std::cout << "\"mean_us\":" << mean_us << ",";
    std::cout << "\"median_us\":" << median_us << ",";
    std::cout << "\"minimum_us\":" << minimum_us << ",";
    std::cout << "\"mode\":\"" << (routed ? "routed_moe" : "gate_up")
              << "\",";
    std::cout << "\"onednn_version\":{\"hash\":\""
              << JsonEscape(version->hash == nullptr ? "" : version->hash)
              << "\",\"major\":" << version->major
              << ",\"minor\":" << version->minor
              << ",\"patch\":" << version->patch << "},";
    std::cout << "\"padded_assignments\":" << plan.padded_assignments << ",";
    std::cout << "\"performance_pass\":" << performance_pass << ",";
    std::cout << "\"repack_pass\":" << repack_pass << ",";
    std::cout << "\"repack_mismatch_count\":" << repack_mismatch_count << ",";
    std::cout << "\"repacked_q4_code_count\":" << repacked_q4_code_count << ",";
    std::cout << "\"samples_us\":[";
    for (std::size_t index = 0; index < samples_us.size(); ++index) {
      if (index != 0) std::cout << ",";
      std::cout << samples_us[index];
    }
    std::cout << "]}" << "\n";
    if (scatter_kernel != nullptr) clReleaseKernel(scatter_kernel);
    clReleaseProgram(program);
    return implementations_pass && correctness_pass && performance_pass ? 0 : 2;
  } catch (const dnnl::error& error) {
    std::cerr << "oneDNN status " << error.status << ": " << error.what()
              << "\n";
    return 3;
  } catch (const std::exception& error) {
    std::cerr << error.what() << "\n";
    return 4;
  }
}
#endif
