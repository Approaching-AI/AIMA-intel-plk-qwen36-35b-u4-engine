#include "intel_qwen36/packed_token_level_zero_backend.hpp"

#include "intel_qwen36/gguf_loader.hpp"
#include "intel_qwen36/gpu_q4x8_matvec.hpp"

#include <level_zero/ze_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

constexpr std::uint32_t kIntelVendorId = 0x8086U;
constexpr std::uint32_t kPtlDeviceId = 0xB080U;
constexpr std::uint32_t kHiddenSize = 2048;
constexpr std::uint32_t kLinearQkvValues = 8192;
constexpr std::uint32_t kLinearProjectionValues = 4096;
constexpr std::uint32_t kLinearCoefficientValues = 4096 + 32 + 32;
constexpr std::uint32_t kLinearHeadDim = 128;
constexpr std::uint32_t kLinearQueryHeads = 16;
constexpr std::uint32_t kLinearValueHeads = 32;
constexpr std::uint32_t kLinearConvStateValues = 3 * 8192;
constexpr std::uint32_t kLinearRecurrentStateValues = 32 * 128 * 128;
constexpr std::uint32_t kFullQValues = 8192;
constexpr std::uint32_t kFullKvValues = 512;
constexpr std::uint32_t kFullKvScaleGroupsPerToken = 2 * 8;
constexpr std::uint32_t kFullAttentionChunkTokens = 256;
constexpr std::uint32_t kFullKvHotWindowTokens = 8192;
constexpr std::uint32_t kSelectedRows = 8 * 2048;
constexpr std::uint32_t kSharedRows = 2048;
constexpr std::uint32_t kFfnGateUpValues = 9 * 1024;
constexpr std::uint32_t kVocabularySize = 248320;
constexpr std::uint32_t kLmHeadBlockSize = 256;
constexpr std::uint32_t kLmHeadBlockCount =
    (kVocabularySize + kLmHeadBlockSize - 1) / kLmHeadBlockSize;
constexpr std::size_t kMaxProfileTimestamps = 1024;
constexpr std::uint32_t kSymmetricQ4Group128Type = 100;
constexpr std::uint32_t kSymmetricQ4Group64Type = 102;
constexpr std::uint32_t kSymmetricQ4Group32Type = 104;
constexpr std::uint64_t kQkvGroup32LayerMask = UINT64_C(0x17);
constexpr std::uint64_t kDownExactLayerMask = UINT64_C(0x8000000000);
constexpr std::uint64_t kDownGroup32LayerMask = UINT64_C(0x0400000000);
constexpr std::uint64_t kVExactLayerMask = UINT64_C(0x80000);
constexpr std::uint64_t kVGroup32LayerMask = UINT64_C(0x88);

[[noreturn]] void Die(const std::string& message) {
  throw std::runtime_error(message);
}

void Require(bool condition, const std::string& message) {
  if (!condition) Die(message);
}

void Check(ze_result_t result, const char* where) {
  if (result != ZE_RESULT_SUCCESS) {
    Die(std::string(where) + " failed with ze_result_t " +
        std::to_string(static_cast<unsigned int>(result)));
  }
}

std::vector<std::uint8_t> ReadBinary(const std::string& path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  Require(static_cast<bool>(input), "failed to open native module");
  const auto size = input.tellg();
  Require(size > 0, "native module is empty");
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
  input.seekg(0, std::ios::beg);
  input.read(reinterpret_cast<char*>(bytes.data()), size);
  Require(static_cast<bool>(input), "failed to read native module");
  return bytes;
}

std::vector<std::uint8_t> ReadTensorBytes(
    const std::string& model_path,
    const iq36::GgufTensorInfo& tensor) {
  std::ifstream input(model_path, std::ios::binary);
  Require(static_cast<bool>(input), "failed to open locked GGUF");
  input.seekg(static_cast<std::streamoff>(tensor.absolute_offset));
  Require(static_cast<bool>(input), "failed to seek locked GGUF tensor");
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(tensor.nbytes));
  input.read(reinterpret_cast<char*>(bytes.data()),
             static_cast<std::streamsize>(bytes.size()));
  Require(static_cast<bool>(input), "failed to read locked GGUF tensor");
  return bytes;
}

std::uint64_t ElementCount(const iq36::GgufTensorInfo& tensor) {
  std::uint64_t count = 1;
  for (const auto dim : tensor.dims) {
    Require(dim > 0 && count <= std::numeric_limits<std::uint64_t>::max() / dim,
            "tensor element count overflow");
    count *= dim;
  }
  return count;
}

std::uint32_t TensorRows(const iq36::GgufTensorInfo& tensor) {
  Require(!tensor.dims.empty() && tensor.dims[0] > 0,
          "tensor has no row dimension");
  const auto rows = ElementCount(tensor) / tensor.dims[0];
  Require(rows <= std::numeric_limits<std::uint32_t>::max(),
          "tensor row count exceeds uint32");
  return static_cast<std::uint32_t>(rows);
}

std::uint32_t TensorBlocks(const iq36::GgufTensorInfo& tensor) {
  Require(!tensor.dims.empty() && tensor.dims[0] % 256 == 0,
          "quantized tensor columns are not block aligned");
  return static_cast<std::uint32_t>(tensor.dims[0] / 256);
}

std::string LayerTensorName(int layer, const char* suffix) {
  return "blk." + std::to_string(layer) + "." + suffix;
}

bool IsFullAttentionLayer(int layer) {
  return (layer + 1) % 4 == 0;
}

std::uint64_t TimestampDelta(std::uint64_t start,
                             std::uint64_t end,
                             std::uint32_t valid_bits) {
  if (valid_bits == 0 || valid_bits >= 64) return end - start;
  const std::uint64_t mask = (std::uint64_t{1} << valid_bits) - 1;
  return (end - start) & mask;
}

void* Offset(void* pointer, std::size_t bytes) {
  return static_cast<void*>(static_cast<std::uint8_t*>(pointer) + bytes);
}

std::vector<std::uint8_t> PackQ4KBlockStripe16(
    const std::vector<std::uint8_t>& raw, std::uint32_t rows) {
  constexpr std::uint32_t kBlockCount = 16;
  constexpr std::uint32_t kBlockBytes = 144;
  Require(rows % 8U == 0U &&
              raw.size() == static_cast<std::size_t>(rows) * kBlockCount *
                                kBlockBytes,
          "Q4_K blockstripe16 source shape mismatch");
  std::vector<std::uint8_t> packed(raw.size());
  for (std::uint32_t row_group = 0; row_group < rows / 8U; ++row_group) {
    for (std::uint32_t row_lane = 0; row_lane < 8U; ++row_lane) {
      const auto row = row_group * 8U + row_lane;
      const auto destination_base =
          (static_cast<std::size_t>(row_group) * 8U + row_lane) *
          kBlockBytes * kBlockCount;
      for (std::uint32_t word = 0; word < kBlockBytes / 4U; ++word) {
        for (std::uint32_t block = 0; block < kBlockCount; ++block) {
          const auto source =
              (static_cast<std::size_t>(row) * kBlockCount + block) *
                  kBlockBytes +
              word * 4U;
          const auto destination =
              destination_base +
              (static_cast<std::size_t>(word) * kBlockCount + block) * 4U;
          std::memcpy(packed.data() + destination, raw.data() + source, 4U);
        }
      }
    }
  }
  return packed;
}

float HalfToFloat(std::uint16_t half) {
  const std::uint32_t sign = (half & 0x8000U) << 16U;
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

struct CompressedBlock32Kv {
  std::vector<std::int8_t> values;
  std::vector<std::uint16_t> scales;
};

CompressedBlock32Kv CompressBlock32Kv(const std::vector<float>& input) {
  constexpr std::size_t kHeadDim = 256;
  constexpr std::size_t kKvHeads = 2;
  constexpr std::size_t kGroup = 32;
  constexpr std::size_t kScaleGroups = kHeadDim / kGroup;
  Require(input.size() % (kKvHeads * kHeadDim) == 0,
          "INT8 block32 KV input shape mismatch");
  const auto tokens = input.size() / (kKvHeads * kHeadDim);
  CompressedBlock32Kv result;
  result.values.resize(input.size());
  result.scales.resize(tokens * kKvHeads * kScaleGroups);
  for (std::size_t token = 0; token < tokens; ++token) {
    for (std::size_t head = 0; head < kKvHeads; ++head) {
      for (std::size_t group = 0; group < kScaleGroups; ++group) {
        const auto value_base =
            (token * kKvHeads + head) * kHeadDim + group * kGroup;
        float maximum = 0.0f;
        for (std::size_t lane = 0; lane < kGroup; ++lane) {
          maximum = std::max(maximum, std::abs(input[value_base + lane]));
        }
        const float scale = maximum == 0.0f ? 1.0f : maximum / 127.0f;
        result.scales[(token * kKvHeads + head) * kScaleGroups + group] =
            FloatToHalf(scale);
        for (std::size_t lane = 0; lane < kGroup; ++lane) {
          const int quantized = std::clamp(
              static_cast<int>(std::nearbyint(
                  input[value_base + lane] / scale)),
              -127, 127);
          result.values[value_base + lane] =
              static_cast<std::int8_t>(quantized);
        }
      }
    }
  }
  return result;
}

std::vector<float> DecompressBlock32Kv(
    const std::vector<std::int8_t>& values,
    const std::vector<std::uint16_t>& scales) {
  constexpr std::size_t kHeadDim = 256;
  constexpr std::size_t kKvHeads = 2;
  constexpr std::size_t kGroup = 32;
  constexpr std::size_t kScaleGroups = kHeadDim / kGroup;
  Require(values.size() % (kKvHeads * kHeadDim) == 0,
          "INT8 block32 KV value shape mismatch");
  const auto tokens = values.size() / (kKvHeads * kHeadDim);
  Require(scales.size() == tokens * kKvHeads * kScaleGroups,
          "INT8 block32 KV scale shape mismatch");
  std::vector<float> result(values.size());
  for (std::size_t token = 0; token < tokens; ++token) {
    for (std::size_t head = 0; head < kKvHeads; ++head) {
      for (std::size_t group = 0; group < kScaleGroups; ++group) {
        const auto value_base =
            (token * kKvHeads + head) * kHeadDim + group * kGroup;
        const float scale = HalfToFloat(
            scales[(token * kKvHeads + head) * kScaleGroups + group]);
        for (std::size_t lane = 0; lane < kGroup; ++lane) {
          result[value_base + lane] =
              static_cast<float>(values[value_base + lane]) * scale;
        }
      }
    }
  }
  return result;
}

void DecodeQ6KBlock(const std::uint8_t* block, float* values) {
  const auto* ql = block;
  const auto* qh = block + 128;
  const auto* scales = reinterpret_cast<const std::int8_t*>(block + 192);
  const std::uint16_t d_bits = static_cast<std::uint16_t>(block[208]) |
      (static_cast<std::uint16_t>(block[209]) << 8U);
  const float d = HalfToFloat(d_bits);
  for (int half = 0; half < 2; ++half) {
    const int base = half * 128;
    for (int lane = 0; lane < 32; ++lane) {
      const int scale_group = lane / 16;
      const auto high = qh[half * 32 + lane];
      const auto low0 = ql[half * 64 + lane];
      const auto low1 = ql[half * 64 + 32 + lane];
      values[base + lane] = d * scales[half * 8 + scale_group] *
          (static_cast<int>((low0 & 15U) | (((high >> 0U) & 3U) << 4U)) - 32);
      values[base + 32 + lane] = d * scales[half * 8 + scale_group + 2] *
          (static_cast<int>((low1 & 15U) | (((high >> 2U) & 3U) << 4U)) - 32);
      values[base + 64 + lane] = d * scales[half * 8 + scale_group + 4] *
          (static_cast<int>((low0 >> 4U) | (((high >> 4U) & 3U) << 4U)) - 32);
      values[base + 96 + lane] = d * scales[half * 8 + scale_group + 6] *
          (static_cast<int>((low1 >> 4U) | (((high >> 6U) & 3U) << 4U)) - 32);
    }
  }
}

int QuantizeSymmetricQ4(float value, float scale) {
  if (scale == 0.0f) return 8;
  const int signed_value = std::clamp(
      static_cast<int>(std::nearbyint(value / scale)), -8, 7);
  return signed_value + 8;
}

std::vector<std::uint8_t> StripeSymmetricQ4Rows8(
    const std::vector<std::uint8_t>& row_major, std::uint32_t rows,
    std::uint32_t group_count, std::uint32_t code_bytes,
    std::uint32_t scale_bytes) {
  constexpr std::uint32_t kRowsPerStripe = 8;
  const std::uint32_t group_bytes = code_bytes + scale_bytes;
  Require(rows % kRowsPerStripe == 0U &&
              row_major.size() == static_cast<std::size_t>(rows) *
                                      group_count * group_bytes,
          "symmetric-Q4 rowstripe8 source shape mismatch");
  std::vector<std::uint8_t> striped(row_major.size());
  for (std::uint32_t row_group = 0; row_group < rows / kRowsPerStripe;
       ++row_group) {
    for (std::uint32_t group = 0; group < group_count; ++group) {
      auto* destination = striped.data() +
          (static_cast<std::size_t>(row_group) * group_count + group) *
              group_bytes * kRowsPerStripe;
      for (std::uint32_t code = 0; code < code_bytes; code += 4U) {
        for (std::uint32_t lane = 0; lane < kRowsPerStripe; ++lane) {
          const auto* source = row_major.data() +
              (static_cast<std::size_t>(row_group * kRowsPerStripe + lane) *
                   group_count +
               group) *
                  group_bytes +
              code;
          std::memcpy(destination +
                          static_cast<std::size_t>(code) * kRowsPerStripe +
                          lane * 4U,
                      source, 4U);
        }
      }
      for (std::uint32_t lane = 0; lane < kRowsPerStripe; ++lane) {
        const auto* source = row_major.data() +
            (static_cast<std::size_t>(row_group * kRowsPerStripe + lane) *
                 group_count +
             group) *
                group_bytes +
            code_bytes;
        std::memcpy(destination +
                        static_cast<std::size_t>(code_bytes) * kRowsPerStripe +
                        lane * scale_bytes,
                    source, scale_bytes);
      }
    }
  }
  return striped;
}

std::vector<std::uint8_t> PackQ6KAsSymmetricQ4Group128(
    const std::vector<std::uint8_t>& raw, std::uint32_t rows,
    std::uint32_t blocks_per_row) {
  constexpr std::size_t kQ6BlockBytes = 210;
  constexpr std::size_t kPackedGroupBytes = 66;
  Require(raw.size() == static_cast<std::size_t>(rows) * blocks_per_row *
                            kQ6BlockBytes,
          "Q6_K symmetric-Q4 source shape mismatch");
  std::vector<std::uint8_t> packed(
      static_cast<std::size_t>(rows) * blocks_per_row * 2U *
      kPackedGroupBytes);
  std::array<float, 256> values{};
  for (std::uint32_t row = 0; row < rows; ++row) {
    for (std::uint32_t block = 0; block < blocks_per_row; ++block) {
      DecodeQ6KBlock(raw.data() +
                         (static_cast<std::size_t>(row) * blocks_per_row + block) *
                             kQ6BlockBytes,
                     values.data());
      for (std::uint32_t group = 0; group < 2U; ++group) {
        const auto* source = values.data() + group * 128U;
        float max_value = 0.0f;
        float max_abs = 0.0f;
        for (std::uint32_t index = 0; index < 128U; ++index) {
          const float absolute = std::fabs(source[index]);
          if (absolute > max_abs) {
            max_abs = absolute;
            max_value = source[index];
          }
        }
        float scale = max_abs == 0.0f ? 0.0f : -max_value / 8.0f;
        const auto scale_f16 = FloatToHalf(scale);
        scale = HalfToFloat(scale_f16);
        auto* destination = packed.data() +
            ((static_cast<std::size_t>(row) * blocks_per_row + block) * 2U +
             group) * kPackedGroupBytes;
        for (std::uint32_t index = 0; index < 64U; ++index) {
          const int low = QuantizeSymmetricQ4(source[index], scale);
          const int high = QuantizeSymmetricQ4(
              source[index + 64U], scale);
          destination[index] = static_cast<std::uint8_t>(low | (high << 4));
        }
        std::memcpy(destination + 64, &scale_f16, sizeof(scale_f16));
      }
    }
  }
  return StripeSymmetricQ4Rows8(
      packed, rows, blocks_per_row * 2U, 64U, sizeof(std::uint16_t));
}

std::vector<std::uint8_t> PackQ6KAsSymmetricQ4Group64(
    const std::vector<std::uint8_t>& raw, std::uint32_t rows,
    std::uint32_t blocks_per_row) {
  constexpr std::size_t kQ6BlockBytes = 210;
  constexpr std::size_t kPackedGroupBytes = 36;
  Require(raw.size() == static_cast<std::size_t>(rows) * blocks_per_row *
                            kQ6BlockBytes,
          "Q6_K symmetric-Q4 group64 source shape mismatch");
  std::vector<std::uint8_t> packed(
      static_cast<std::size_t>(rows) * blocks_per_row * 4U *
      kPackedGroupBytes);
  std::array<float, 256> values{};
  for (std::uint32_t row = 0; row < rows; ++row) {
    for (std::uint32_t block = 0; block < blocks_per_row; ++block) {
      DecodeQ6KBlock(raw.data() +
                         (static_cast<std::size_t>(row) * blocks_per_row + block) *
                             kQ6BlockBytes,
                     values.data());
      for (std::uint32_t group = 0; group < 4U; ++group) {
        const auto* source = values.data() + group * 64U;
        float max_value = 0.0f;
        float max_abs = 0.0f;
        for (std::uint32_t index = 0; index < 64U; ++index) {
          const float absolute = std::fabs(source[index]);
          if (absolute > max_abs) {
            max_abs = absolute;
            max_value = source[index];
          }
        }
        const float scale = max_abs == 0.0f ? 0.0f : -max_value / 8.0f;
        auto* destination = packed.data() +
            ((static_cast<std::size_t>(row) * blocks_per_row + block) * 4U +
             group) * kPackedGroupBytes;
        for (std::uint32_t index = 0; index < 32U; ++index) {
          const int low = QuantizeSymmetricQ4(source[index], scale);
          const int high = QuantizeSymmetricQ4(
              source[index + 32U], scale);
          destination[index] = static_cast<std::uint8_t>(low | (high << 4));
        }
        std::memcpy(destination + 32, &scale, sizeof(scale));
      }
    }
  }
  return StripeSymmetricQ4Rows8(
      packed, rows, blocks_per_row * 4U, 32U, sizeof(float));
}

std::vector<std::uint8_t> PackQ6KAsSymmetricQ4Group32(
    const std::vector<std::uint8_t>& raw, std::uint32_t rows,
    std::uint32_t blocks_per_row) {
  constexpr std::size_t kQ6BlockBytes = 210;
  constexpr std::size_t kPackedGroupBytes = 20;
  Require(raw.size() == static_cast<std::size_t>(rows) * blocks_per_row *
                            kQ6BlockBytes,
          "Q6_K symmetric-Q4 group32 source shape mismatch");
  std::vector<std::uint8_t> packed(
      static_cast<std::size_t>(rows) * blocks_per_row * 8U *
      kPackedGroupBytes);
  std::array<float, 256> values{};
  for (std::uint32_t row = 0; row < rows; ++row) {
    for (std::uint32_t block = 0; block < blocks_per_row; ++block) {
      DecodeQ6KBlock(raw.data() +
                         (static_cast<std::size_t>(row) * blocks_per_row + block) *
                             kQ6BlockBytes,
                     values.data());
      for (std::uint32_t group = 0; group < 8U; ++group) {
        const auto* source = values.data() + group * 32U;
        float max_value = 0.0f;
        float max_abs = 0.0f;
        for (std::uint32_t index = 0; index < 32U; ++index) {
          const float absolute = std::fabs(source[index]);
          if (absolute > max_abs) {
            max_abs = absolute;
            max_value = source[index];
          }
        }
        const float scale = max_abs == 0.0f ? 0.0f : -max_value / 8.0f;
        auto* destination = packed.data() +
            ((static_cast<std::size_t>(row) * blocks_per_row + block) * 8U +
             group) * kPackedGroupBytes;
        for (std::uint32_t index = 0; index < 16U; ++index) {
          const int low = QuantizeSymmetricQ4(source[index], scale);
          const int high = QuantizeSymmetricQ4(source[index + 16U], scale);
          destination[index] = static_cast<std::uint8_t>(low | (high << 4));
        }
        std::memcpy(destination + 16U, &scale, sizeof(scale));
      }
    }
  }
  return StripeSymmetricQ4Rows8(
      packed, rows, blocks_per_row * 8U, 16U, sizeof(float));
}

}  // namespace

namespace iq36 {

class PackedTokenLevelZeroBackend::Impl {
 public:
  Impl(std::string model_path,
       std::string module_path,
       PackedTokenLevelZeroConfig config)
      : model_path_(std::move(model_path)),
        module_path_(std::move(module_path)),
        config_(std::move(config)),
        index_(parse_gguf_model_index(model_path_)) {
    Require(validate_qwen36_load_map(index_).ready,
            "Level Zero backend requires the locked Qwen3.6 load map");
    Require(config_.state_capacity_tokens > 0,
            "state capacity must be nonzero");
    if (config_.use_int8_block32_kv_gqa) {
      Require(config_.state_capacity_tokens <=
                  std::numeric_limits<std::uint32_t>::max(),
              "INT8 KV capacity exceeds uint32");
      Require(config_.full_head_dim == 256 &&
                  config_.full_q_head_count == 16 &&
                  config_.full_kv_head_count == 2,
              "INT8 block32 GQA requires locked Hq16/Hkv2/D256");
    }
    InitializeDevice();
    InitializeRuntime();
  }

  ~Impl() { Cleanup(); }

  void LoadState(const PackedTokenStateSnapshot& state) {
    initial_state_ = state;
    if (compiled_) UploadState();
  }

  PackedTokenStateSnapshot ReadState() const {
    Require(compiled_, "backend state requested before compile");
    PackedTokenStateSnapshot result;
    for (int layer = 0; layer < kPackedTokenLayerCount; ++layer) {
      if (IsFullAttentionLayer(layer)) {
        const std::size_t values = static_cast<std::size_t>(
            std::min<std::uint64_t>(last_token_position_ + 1,
                                    config_.state_capacity_tokens) *
            kFullKvValues);
        if (config_.use_int8_block32_kv_gqa) {
          const std::size_t tokens = values / kFullKvValues;
          const std::size_t scale_values =
              tokens * kFullKvScaleGroupsPerToken;
          std::vector<std::int8_t> k_values(values);
          std::vector<std::int8_t> v_values(values);
          std::vector<std::uint16_t> k_scales(scale_values);
          std::vector<std::uint16_t> v_scales(scale_values);
          CopyFromDevice(full_state_[layer].k, k_values.data(), values);
          CopyFromDevice(full_state_[layer].v, v_values.data(), values);
          CopyFromDevice(full_state_[layer].k_scales, k_scales.data(),
                         scale_values * sizeof(std::uint16_t));
          CopyFromDevice(full_state_[layer].v_scales, v_scales.data(),
                         scale_values * sizeof(std::uint16_t));
          result.full_k_history[layer] =
              DecompressBlock32Kv(k_values, k_scales);
          result.full_v_history[layer] =
              DecompressBlock32Kv(v_values, v_scales);
          const auto hot_values = static_cast<std::size_t>(
              kFullKvHotWindowTokens * kFullKvValues);
          std::vector<float> hot_k(hot_values);
          std::vector<float> hot_v(hot_values);
          CopyFromDevice(full_state_[layer].hot_k, hot_k.data(),
                         hot_values * sizeof(float));
          CopyFromDevice(full_state_[layer].hot_v, hot_v.data(),
                         hot_values * sizeof(float));
          const auto hot_count = std::min<std::size_t>(
              tokens, kFullKvHotWindowTokens);
          for (std::size_t token = tokens - hot_count; token < tokens;
               ++token) {
            const auto source =
                (token % kFullKvHotWindowTokens) * kFullKvValues;
            const auto destination = token * kFullKvValues;
            std::copy_n(hot_k.data() + source, kFullKvValues,
                        result.full_k_history[layer].data() + destination);
            std::copy_n(hot_v.data() + source, kFullKvValues,
                        result.full_v_history[layer].data() + destination);
          }
        } else {
          result.full_k_history[layer].resize(values);
          result.full_v_history[layer].resize(values);
          CopyFromDevice(full_state_[layer].k,
                         result.full_k_history[layer].data(),
                         values * sizeof(float));
          CopyFromDevice(full_state_[layer].v,
                         result.full_v_history[layer].data(),
                         values * sizeof(float));
        }
      } else {
        result.linear_conv[layer].resize(kLinearConvStateValues);
        result.linear_recurrent[layer].resize(kLinearRecurrentStateValues);
        CopyFromDevice(linear_state_[layer].conv,
                       result.linear_conv[layer].data(),
                       kLinearConvStateValues * sizeof(float));
        CopyFromDevice(linear_state_[layer].recurrent,
                       result.linear_recurrent[layer].data(),
                       kLinearRecurrentStateValues * sizeof(float));
      }
    }
    return result;
  }

  std::vector<float> ReadLogits() const {
    Require(compiled_, "backend logits requested before compile");
    std::vector<float> result(kVocabularySize);
    CopyFromDevice(logits_, result.data(), result.size() * sizeof(float));
    return result;
  }

  void Compile(const PackedTokenProgram& program) {
    Require(!compiled_, "Level Zero backend may only compile once");
    Require(program.context_tokens <= config_.state_capacity_tokens,
            "program context exceeds backend state capacity");
    const auto validation = ValidatePackedTokenProgram(index_, program);
    Require(validation.passed, "packed token program validation failed");
    program_ = program;
    LoadWeights();
    AllocateScratch();
    AllocateState();
    UploadState();
    RecordCommandList();
    compiled_ = true;
  }

  std::vector<PackedTokenTopKRow> SubmitToken(
      const PackedTokenSubmission& submission) {
    Require(compiled_, "token submitted before backend compile");
    Require(submission.top_k > 0 && submission.top_k <= 8,
            "Level Zero top-k must be in 1..8");
    Require(submission.token_position < config_.state_capacity_tokens,
            "token position exceeds state capacity");
    token_control_[0] = submission.token_id;
    token_control_[1] = submission.token_position;
    const auto rope = build_qwen36_rope_cache(
        static_cast<std::int32_t>(submission.token_position),
        config_.rope_dimension_count, config_.rope_sections,
        config_.rope_context_length, config_.rope_freq_base,
        config_.rope_freq_scale, config_.rope_ext_factor,
        config_.rope_attn_factor, config_.rope_beta_fast,
        config_.rope_beta_slow);
    Require(rope.size() == config_.rope_dimension_count,
            "RoPE cache size mismatch");
    std::memcpy(rope_cache_, rope.data(), rope.size() * sizeof(float));
    const auto wall_begin = std::chrono::steady_clock::now();
    Check(zeCommandQueueExecuteCommandLists(queue_, 1, &command_list_, nullptr),
          "zeCommandQueueExecuteCommandLists(token)");
    const auto submit_end = std::chrono::steady_clock::now();
    Check(zeCommandQueueSynchronize(queue_, UINT64_MAX),
          "zeCommandQueueSynchronize(token)");
    const auto wall_end = std::chrono::steady_clock::now();
    const auto ticks = TimestampDelta(timestamps_[0],
                                      timestamps_[end_timestamp_index_],
                                      properties_.kernelTimestampValidBits);
    timing_.device_ms = ticks * timestamp_ns_per_tick_ / 1.0e6;
    timing_.kernel_profile.clear();
    if (config_.profile_kernel_times) {
      Require(profile_kernel_names_.size() == end_timestamp_index_,
              "kernel profile timestamp/name count mismatch");
      timing_.kernel_profile.reserve(profile_kernel_names_.size());
      for (std::size_t index = 0; index < profile_kernel_names_.size(); ++index) {
        const auto kernel_ticks = TimestampDelta(
            timestamps_[index], timestamps_[index + 1],
            properties_.kernelTimestampValidBits);
        timing_.kernel_profile.push_back({
            profile_kernel_names_[index],
            kernel_ticks * timestamp_ns_per_tick_ / 1.0e6});
      }
    }
    timing_.host_submit_ms =
        std::chrono::duration<double, std::milli>(submit_end - wall_begin)
            .count();
    timing_.wall_ms =
        std::chrono::duration<double, std::milli>(wall_end - wall_begin).count();
    last_token_position_ = submission.token_position;
    std::vector<PackedTokenTopKRow> result;
    result.reserve(submission.top_k);
    for (std::size_t i = 0; i < submission.top_k; ++i) {
      result.push_back({top_ids_[i], top_values_[i]});
    }
    return result;
  }

  PackedTokenLevelZeroTiming timing() const { return timing_; }
  const std::string& device_name() const { return device_name_; }

 private:
  struct DeviceTensor {
    void* pointer = nullptr;
    std::uint64_t bytes = 0;
    std::uint32_t type = 0;
    std::uint32_t rows = 0;
    std::uint32_t cols = 0;
    std::uint32_t blocks_per_row = 0;
    std::uint32_t rows_per_tile = 0;
  };

  struct LinearState {
    void* conv = nullptr;
    void* recurrent = nullptr;
  };

  struct FullState {
    void* k = nullptr;
    void* v = nullptr;
    void* k_scales = nullptr;
    void* v_scales = nullptr;
    void* hot_k = nullptr;
    void* hot_v = nullptr;
  };

  void InitializeDevice() {
    Check(zeInit(ZE_INIT_FLAG_GPU_ONLY), "zeInit");
    std::uint32_t driver_count = 0;
    Check(zeDriverGet(&driver_count, nullptr), "zeDriverGet(count)");
    std::vector<ze_driver_handle_t> drivers(driver_count);
    Check(zeDriverGet(&driver_count, drivers.data()), "zeDriverGet(list)");
    for (auto driver : drivers) {
      std::uint32_t device_count = 0;
      Check(zeDeviceGet(driver, &device_count, nullptr), "zeDeviceGet(count)");
      std::vector<ze_device_handle_t> devices(device_count);
      Check(zeDeviceGet(driver, &device_count, devices.data()),
            "zeDeviceGet(list)");
      for (auto device : devices) {
        ze_device_properties_t properties{ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES};
        Check(zeDeviceGetProperties(device, &properties),
              "zeDeviceGetProperties");
        if (properties.vendorId != kIntelVendorId ||
            properties.deviceId != kPtlDeviceId) continue;
        std::uint32_t group_count = 0;
        Check(zeDeviceGetCommandQueueGroupProperties(device, &group_count,
                                                      nullptr),
              "zeDeviceGetCommandQueueGroupProperties(count)");
        std::vector<ze_command_queue_group_properties_t> groups(group_count);
        for (auto& group : groups) {
          group.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES;
        }
        Check(zeDeviceGetCommandQueueGroupProperties(device, &group_count,
                                                      groups.data()),
              "zeDeviceGetCommandQueueGroupProperties(list)");
        for (std::uint32_t ordinal = 0; ordinal < group_count; ++ordinal) {
          if ((groups[ordinal].flags &
               ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) != 0U) {
            driver_ = driver;
            device_ = device;
            queue_ordinal_ = ordinal;
            properties_ = properties;
            device_name_ = properties.name;
            break;
          }
        }
        if (device_ != nullptr) break;
      }
      if (device_ != nullptr) break;
    }
    Require(device_ != nullptr, "PTL Level Zero device not found");
    std::uint64_t host0 = 0;
    std::uint64_t device0 = 0;
    std::uint64_t host1 = 0;
    std::uint64_t device1 = 0;
    Check(zeDeviceGetGlobalTimestamps(device_, &host0, &device0),
          "zeDeviceGetGlobalTimestamps(start)");
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    Check(zeDeviceGetGlobalTimestamps(device_, &host1, &device1),
          "zeDeviceGetGlobalTimestamps(end)");
    const auto ticks = TimestampDelta(device0, device1,
                                      properties_.kernelTimestampValidBits);
    Require(host1 > host0 && ticks > 0, "timestamp calibration failed");
    timestamp_ns_per_tick_ =
        static_cast<double>(host1 - host0) / static_cast<double>(ticks);
  }

  void InitializeRuntime() {
    ze_context_desc_t context_desc{ZE_STRUCTURE_TYPE_CONTEXT_DESC};
    Check(zeContextCreate(driver_, &context_desc, &context_),
          "zeContextCreate");
    ze_command_queue_desc_t queue_desc{ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC};
    queue_desc.ordinal = queue_ordinal_;
    queue_desc.index = 0;
    queue_desc.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
    Check(zeCommandQueueCreate(context_, device_, &queue_desc, &queue_),
          "zeCommandQueueCreate");
    ze_command_list_desc_t list_desc{ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC};
    list_desc.commandQueueGroupOrdinal = queue_ordinal_;
    list_desc.flags = ZE_COMMAND_LIST_FLAG_IN_ORDER |
                      ZE_COMMAND_LIST_FLAG_MAXIMIZE_THROUGHPUT;
    Check(zeCommandListCreate(context_, device_, &list_desc, &command_list_),
          "zeCommandListCreate(run)");
    queue_desc.mode = ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS;
    Check(zeCommandListCreateImmediate(context_, device_, &queue_desc,
                                       &immediate_list_),
          "zeCommandListCreateImmediate");
    module_bytes_ = ReadBinary(module_path_);
    ze_module_desc_t module_desc{ZE_STRUCTURE_TYPE_MODULE_DESC};
    module_desc.format = ZE_MODULE_FORMAT_NATIVE;
    module_desc.inputSize = module_bytes_.size();
    module_desc.pInputModule = module_bytes_.data();
    module_desc.pBuildFlags = "";
    ze_module_build_log_handle_t log = nullptr;
    const auto result = zeModuleCreate(context_, device_, &module_desc,
                                       &module_, &log);
    if (log != nullptr) zeModuleBuildLogDestroy(log);
    Check(result, "zeModuleCreate");
  }

  void* AllocateDevice(std::size_t bytes, bool zero = true) {
    Require(bytes > 0, "zero-size device allocation");
    ze_device_mem_alloc_desc_t desc{ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC};
    void* pointer = nullptr;
    Check(zeMemAllocDevice(context_, &desc, bytes, 64, device_, &pointer),
          "zeMemAllocDevice");
    device_allocations_.push_back(pointer);
    if (zero) {
      const std::uint32_t pattern = 0;
      Check(zeCommandListAppendMemoryFill(immediate_list_, pointer, &pattern,
                                          sizeof(pattern), bytes, nullptr,
                                          0, nullptr),
            "zeCommandListAppendMemoryFill");
    }
    return pointer;
  }

  void* AllocateShared(std::size_t bytes) {
    ze_device_mem_alloc_desc_t device_desc{
        ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC};
    ze_host_mem_alloc_desc_t host_desc{ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC};
    void* pointer = nullptr;
    Check(zeMemAllocShared(context_, &device_desc, &host_desc, bytes, 64,
                           device_, &pointer),
          "zeMemAllocShared");
    std::memset(pointer, 0, bytes);
    shared_allocations_.push_back(pointer);
    return pointer;
  }

  void ZeroDevice(void* pointer, std::size_t bytes) {
    Require(pointer != nullptr && bytes > 0, "invalid device zero fill");
    const std::uint32_t pattern = 0;
    Check(zeCommandListAppendMemoryFill(immediate_list_, pointer, &pattern,
                                        sizeof(pattern), bytes, nullptr,
                                        0, nullptr),
          "zeCommandListAppendMemoryFill(reset)");
  }

  void CopyToDevice(void* destination, const void* source, std::size_t bytes) {
    Require(destination != nullptr && source != nullptr && bytes > 0,
            "invalid device upload");
    ze_host_mem_alloc_desc_t host_desc{ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC};
    void* staging = nullptr;
    Check(zeMemAllocHost(context_, &host_desc, bytes, 64, &staging),
          "zeMemAllocHost(upload)");
    std::memcpy(staging, source, bytes);
    Check(zeCommandListAppendMemoryCopy(immediate_list_, destination, staging,
                                        bytes, nullptr, 0, nullptr),
          "zeCommandListAppendMemoryCopy(upload)");
    Check(zeMemFree(context_, staging), "zeMemFree(upload staging)");
  }

  void CopyFromDevice(void* source, void* destination, std::size_t bytes) const {
    Require(source != nullptr && destination != nullptr && bytes > 0,
            "invalid device readback");
    ze_host_mem_alloc_desc_t host_desc{ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC};
    void* staging = nullptr;
    Check(zeMemAllocHost(context_, &host_desc, bytes, 64, &staging),
          "zeMemAllocHost(readback)");
    Check(zeCommandListAppendMemoryCopy(immediate_list_, staging, source,
                                        bytes, nullptr, 0, nullptr),
          "zeCommandListAppendMemoryCopy(readback)");
    std::memcpy(destination, staging, bytes);
    Check(zeMemFree(context_, staging), "zeMemFree(readback staging)");
  }

  DeviceTensor UploadTensor(const GgufTensorInfo& tensor) {
    auto raw = ReadTensorBytes(model_path_, tensor);
    std::vector<std::uint8_t> packed;
    std::uint32_t rows_per_tile = 0;
    const auto rows = TensorRows(tensor);
    std::uint32_t uploaded_type = tensor.type;
    std::uint32_t blocks = 0;
    if (tensor.type == 14 && tensor.name == "output.weight") {
      const auto exact_blocks = TensorBlocks(tensor);
      auto exact_layout = PackQ6KRowstripe(
          raw, rows, exact_blocks, 1, 16);
      lm_head_exact_ = AllocateDevice(exact_layout.bytes.size(), false);
      CopyToDevice(lm_head_exact_, exact_layout.bytes.data(),
                   exact_layout.bytes.size());
      lm_head_exact_rows_per_tile_ = static_cast<std::uint32_t>(
          exact_layout.rows_per_tile);
      timing_.resident_weight_bytes += exact_layout.bytes.size();
    }
    if (tensor.type == 12) {
      blocks = TensorBlocks(tensor);
      if (tensor.name == "token_embd.weight") {
        packed = std::move(raw);
      } else if (blocks == 16U) {
        packed = PackQ4KBlockStripe16(raw, rows);
      } else {
        packed = PackQ4Kx8(raw, rows, blocks);
      }
    } else if (tensor.type == 14) {
      const bool repack_lm_head = tensor.name == "output.weight";
      const bool repack_qkv = tensor.suffix == "attn_qkv.weight";
      const bool repack_down =
          (tensor.suffix == "ffn_down_exps.weight" ||
           tensor.suffix == "ffn_down_shexp.weight") &&
          (tensor.layer_index < 0 ||
           (kDownExactLayerMask &
            (UINT64_C(1) << tensor.layer_index)) == 0U);
      const bool repack_v =
          tensor.suffix == "attn_v.weight" &&
          (tensor.layer_index < 0 ||
           (kVExactLayerMask &
            (UINT64_C(1) << tensor.layer_index)) == 0U);
      const bool qkv_group32 =
          repack_qkv && tensor.layer_index >= 0 &&
          (kQkvGroup32LayerMask &
           (UINT64_C(1) << tensor.layer_index)) != 0U;
      const bool down_group32 =
          repack_down && tensor.layer_index >= 0 &&
          (kDownGroup32LayerMask &
           (UINT64_C(1) << tensor.layer_index)) != 0U;
      const bool v_group32 =
          repack_v && tensor.layer_index >= 0 &&
          (kVGroup32LayerMask &
           (UINT64_C(1) << tensor.layer_index)) != 0U;
      if (qkv_group32 || down_group32 || v_group32) {
        blocks = TensorBlocks(tensor);
        packed = PackQ6KAsSymmetricQ4Group32(raw, rows, blocks);
        uploaded_type = kSymmetricQ4Group32Type;
      } else if (repack_qkv || repack_v) {
        blocks = TensorBlocks(tensor);
        packed = PackQ6KAsSymmetricQ4Group64(raw, rows, blocks);
        uploaded_type = kSymmetricQ4Group64Type;
      } else if (repack_lm_head || repack_down) {
        blocks = TensorBlocks(tensor);
        packed = PackQ6KAsSymmetricQ4Group128(raw, rows, blocks);
        uploaded_type = kSymmetricQ4Group128Type;
      } else {
        blocks = TensorBlocks(tensor);
        PackedQ6KRowstripe layout;
        if (tensor.suffix == "ffn_down_exps.weight") {
          layout = PackQ6KRowstripeCoalesced(raw, 2048, blocks, 256, 32);
        } else if (tensor.suffix == "ffn_down_shexp.weight") {
          layout = PackQ6KRowstripeCoalesced(raw, 2048, blocks, 1, 32);
        } else {
          layout = PackQ6KRowstripe(raw, rows, blocks, 1,
                                    rows <= 8192 ? 32 : 16);
        }
        rows_per_tile = static_cast<std::uint32_t>(layout.rows_per_tile);
        packed = std::move(layout.bytes);
      }
    } else if (tensor.type == 0) {
      packed = std::move(raw);
    } else {
      Die("unsupported backend tensor type: " + tensor.name);
    }
    auto* pointer = AllocateDevice(packed.size(), false);
    CopyToDevice(pointer, packed.data(), packed.size());
    timing_.resident_weight_bytes += packed.size();
    return {pointer, packed.size(), uploaded_type, rows,
            static_cast<std::uint32_t>(tensor.dims[0]), blocks,
            rows_per_tile};
  }

  void LoadWeights() {
    for (const auto& tensor : index_.tensors) {
      if (tensor.suffix == "ffn_gate_shexp.weight" ||
          tensor.suffix == "ffn_up_shexp.weight" ||
          tensor.suffix == "ffn_gate_inp.weight" ||
          tensor.suffix == "ffn_gate_inp_shexp.weight" ||
          tensor.suffix == "attn_gate.weight" ||
          tensor.suffix == "ssm_alpha.weight" ||
          tensor.suffix == "ssm_beta.weight" ||
          (tensor.suffix == "attn_qkv.weight" && tensor.type == 12) ||
          tensor.suffix == "attn_q.weight" ||
          tensor.suffix == "attn_k.weight" ||
          (tensor.suffix == "attn_v.weight" && tensor.type == 12)) {
        continue;
      }
      weights_.emplace(tensor.name, UploadTensor(tensor));
    }
    for (int layer = 0; layer < kPackedTokenLayerCount; ++layer) {
      const auto gate_name = LayerTensorName(layer, "ffn_gate_shexp.weight");
      const auto up_name = LayerTensorName(layer, "ffn_up_shexp.weight");
      const auto* gate = find_tensor(index_, gate_name);
      const auto* up = find_tensor(index_, up_name);
      Require(gate != nullptr && up != nullptr, "shared gate/up tensor missing");
      auto gate_raw = ReadTensorBytes(model_path_, *gate);
      auto up_raw = ReadTensorBytes(model_path_, *up);
      auto gate_packed = PackQ4Kx8(gate_raw, 512, 8);
      auto up_packed = PackQ4Kx8(up_raw, 512, 8);
      gate_packed.insert(gate_packed.end(), up_packed.begin(), up_packed.end());
      auto* pointer = AllocateDevice(gate_packed.size(), false);
      CopyToDevice(pointer, gate_packed.data(), gate_packed.size());
      timing_.resident_weight_bytes += gate_packed.size();
      const auto key = LayerTensorName(layer, "ffn_gate_up_shexp.combined");
      weights_.emplace(key, DeviceTensor{
          pointer, gate_packed.size(), 12, 1024, 2048, 8, 0});

      const auto router_name = LayerTensorName(layer, "ffn_gate_inp.weight");
      const auto shared_router_name =
          LayerTensorName(layer, "ffn_gate_inp_shexp.weight");
      const auto* router = find_tensor(index_, router_name);
      const auto* shared_router = find_tensor(index_, shared_router_name);
      Require(router != nullptr && shared_router != nullptr,
              "router tensor missing");
      auto router_raw = ReadTensorBytes(model_path_, *router);
      auto shared_router_raw = ReadTensorBytes(model_path_, *shared_router);
      router_raw.insert(router_raw.end(), shared_router_raw.begin(),
                        shared_router_raw.end());
      Require(router_raw.size() == 257U * kHiddenSize * sizeof(float),
              "router tensor size is not 257x2048 F32");
      constexpr std::uint32_t kRouterGroupSize = 32;
      constexpr std::uint32_t kRouterGroups =
          kHiddenSize / kRouterGroupSize;
      std::vector<std::int8_t> router_q8(257U * kHiddenSize);
      std::vector<float> router_scales(257U * kRouterGroups);
      for (std::uint32_t row = 0; row < 257U; ++row) {
        for (std::uint32_t group = 0; group < kRouterGroups; ++group) {
          float values[kRouterGroupSize];
          float maximum = 0.0f;
          for (std::uint32_t lane = 0; lane < kRouterGroupSize; ++lane) {
            const auto index =
                static_cast<std::size_t>(row) * kHiddenSize +
                group * kRouterGroupSize + lane;
            std::memcpy(&values[lane],
                        router_raw.data() + index * sizeof(float),
                        sizeof(float));
            maximum = std::max(maximum, std::fabs(values[lane]));
          }
          const float scale = maximum == 0.0f ? 0.0f : maximum / 127.0f;
          router_scales[static_cast<std::size_t>(row) * kRouterGroups +
                        group] = scale;
          for (std::uint32_t lane = 0; lane < kRouterGroupSize; ++lane) {
            const auto quantized = scale == 0.0f
                ? 0L
                : std::lround(values[lane] / scale);
            const auto index =
                static_cast<std::size_t>(row) * kHiddenSize +
                group * kRouterGroupSize + lane;
            router_q8[index] = static_cast<std::int8_t>(
                std::clamp(quantized, -127L, 127L));
          }
        }
      }
      const auto router_bytes = router_q8.size() +
          router_scales.size() * sizeof(float);
      std::vector<std::uint8_t> router_packed(router_bytes);
      std::memcpy(router_packed.data(), router_q8.data(), router_q8.size());
      std::memcpy(router_packed.data() + router_q8.size(),
                  router_scales.data(),
                  router_scales.size() * sizeof(float));
      auto* router_pointer = AllocateDevice(router_bytes, false);
      CopyToDevice(router_pointer, router_packed.data(), router_bytes);
      timing_.resident_weight_bytes += router_bytes;
      weights_.emplace(
          LayerTensorName(layer, "ffn_router_shared.combined"),
          DeviceTensor{router_pointer, router_bytes, 15, 257, 2048, 0, 0});

      if (!IsFullAttentionLayer(layer)) {
        const auto qkv_name = LayerTensorName(layer, "attn_qkv.weight");
        const auto* qkv = find_tensor(index_, qkv_name);
        Require(qkv != nullptr, "linear QKV tensor missing");
        std::vector<std::uint8_t> front_packed;
        std::uint32_t front_rows = kLinearCoefficientValues;
        if (qkv->type == 12) {
          front_packed = PackQ4Kx8(ReadTensorBytes(model_path_, *qkv),
                                   kLinearQkvValues, TensorBlocks(*qkv));
          front_rows += kLinearQkvValues;
        }
        for (const char* suffix : {"attn_gate.weight", "ssm_alpha.weight",
                                   "ssm_beta.weight"}) {
          const auto name = LayerTensorName(layer, suffix);
          const auto* tensor = find_tensor(index_, name);
          Require(tensor != nullptr && tensor->type == 12,
                  "linear coefficient tensor missing or non-Q4");
          const auto rows = TensorRows(*tensor);
          auto packed = PackQ4Kx8(ReadTensorBytes(model_path_, *tensor), rows,
                                  TensorBlocks(*tensor));
          front_packed.insert(front_packed.end(), packed.begin(), packed.end());
        }
        auto* front_pointer = AllocateDevice(front_packed.size(), false);
        CopyToDevice(front_pointer, front_packed.data(), front_packed.size());
        timing_.resident_weight_bytes += front_packed.size();
        weights_.emplace(
            LayerTensorName(layer, qkv->type == 12
                                       ? "linear_qkv_coefficients.combined"
                                       : "linear_coefficients.combined"),
            DeviceTensor{front_pointer, front_packed.size(), 12,
                         front_rows, 2048, 8, 0});
      } else {
        const auto q_name = LayerTensorName(layer, "attn_q.weight");
        const auto k_name = LayerTensorName(layer, "attn_k.weight");
        const auto v_name = LayerTensorName(layer, "attn_v.weight");
        const auto* q = find_tensor(index_, q_name);
        const auto* k = find_tensor(index_, k_name);
        const auto* v = find_tensor(index_, v_name);
        Require(q != nullptr && k != nullptr && v != nullptr &&
                    q->type == 12 && k->type == 12,
                "full-attention Q/K/V tensor contract missing");
        std::vector<std::uint8_t> full_front;
        std::uint32_t rows = 0;
        for (const auto* tensor : {q, k}) {
          const auto tensor_rows = TensorRows(*tensor);
          auto packed = PackQ4Kx8(ReadTensorBytes(model_path_, *tensor),
                                  tensor_rows, TensorBlocks(*tensor));
          full_front.insert(full_front.end(), packed.begin(), packed.end());
          rows += tensor_rows;
        }
        if (v->type == 12) {
          const auto tensor_rows = TensorRows(*v);
          auto packed = PackQ4Kx8(ReadTensorBytes(model_path_, *v),
                                  tensor_rows, TensorBlocks(*v));
          full_front.insert(full_front.end(), packed.begin(), packed.end());
          rows += tensor_rows;
        }
        auto* full_front_pointer = AllocateDevice(full_front.size(), false);
        CopyToDevice(full_front_pointer, full_front.data(), full_front.size());
        timing_.resident_weight_bytes += full_front.size();
        weights_.emplace(
            LayerTensorName(layer, v->type == 12
                                       ? "full_qkv.combined"
                                       : "full_qk.combined"),
            DeviceTensor{full_front_pointer, full_front.size(), 12, rows,
                         2048, 8, 0});
      }
    }
  }

  void AllocateScratch() {
    token_control_ = static_cast<std::uint64_t*>(
        AllocateShared(2 * sizeof(std::uint64_t)));
    rope_cache_ = static_cast<float*>(AllocateShared(
        config_.rope_dimension_count * sizeof(float)));
    const auto timestamp_count = config_.profile_kernel_times
        ? kMaxProfileTimestamps
        : 2;
    timestamps_ = static_cast<std::uint64_t*>(
        AllocateShared(timestamp_count * sizeof(std::uint64_t)));
    top_ids_ = static_cast<std::int32_t*>(AllocateShared(8 * sizeof(int)));
    top_values_ = static_cast<float*>(AllocateShared(8 * sizeof(float)));
    hidden_a_ = AllocateDevice(kHiddenSize * sizeof(float));
    hidden_b_ = AllocateDevice(kHiddenSize * sizeof(float));
    norm_ = AllocateDevice(kHiddenSize * sizeof(float));
    rms_scale_ = AllocateDevice(sizeof(float));
    attention_residual_ = AllocateDevice(kHiddenSize * sizeof(float));
    q8_qs_ = AllocateDevice(kLinearQkvValues * sizeof(std::int8_t));
    q8_bsums_ = AllocateDevice(512 * sizeof(std::int16_t));
    q8_d_ = AllocateDevice(32 * sizeof(float));
    linear_front_ = AllocateDevice(
        (kLinearQkvValues + kLinearCoefficientValues) * sizeof(float));
    qkv_ = linear_front_;
    linear_coefficients_ =
        Offset(linear_front_, kLinearQkvValues * sizeof(float));
    z_ = linear_coefficients_;
    alpha_ = Offset(linear_coefficients_, 4096 * sizeof(float));
    beta_ = Offset(linear_coefficients_, (4096 + 32) * sizeof(float));
    linear_gate_ = AllocateDevice(32 * sizeof(float));
    beta_sigmoid_ = AllocateDevice(32 * sizeof(float));
    conv_output_ = AllocateDevice(kLinearQkvValues * sizeof(float));
    q_predelta_ = AllocateDevice(2048 * sizeof(float));
    k_predelta_ = AllocateDevice(2048 * sizeof(float));
    v_predelta_ = AllocateDevice(4096 * sizeof(float));
    linear_attention_ = AllocateDevice(4096 * sizeof(float));
    linear_final_ = AllocateDevice(4096 * sizeof(float));
    projection_ = AllocateDevice(kHiddenSize * sizeof(float));
    full_front_ = AllocateDevice(
        (kFullQValues + 2 * kFullKvValues) * sizeof(float));
    full_q_ = full_front_;
    full_k_ = Offset(full_front_, kFullQValues * sizeof(float));
    full_v_ = Offset(full_front_,
                     (kFullQValues + kFullKvValues) * sizeof(float));
    full_q_rope_ = AllocateDevice(4096 * sizeof(float));
    full_k_rope_ = AllocateDevice(kFullKvValues * sizeof(float));
    full_pregate_ = AllocateDevice(4096 * sizeof(float));
    full_gated_ = AllocateDevice(4096 * sizeof(float));
    if (config_.use_int8_block32_kv_gqa) {
      const auto chunks =
          (config_.state_capacity_tokens + kFullAttentionChunkTokens - 1U) /
          kFullAttentionChunkTokens;
      const auto meta_values = config_.full_kv_head_count * chunks * 8U;
      full_partial_max_ = AllocateDevice(meta_values * sizeof(float));
      full_partial_sum_ = AllocateDevice(meta_values * sizeof(float));
      full_partial_output_ = AllocateDevice(
          meta_values * config_.full_head_dim * sizeof(float));
    } else {
      full_scores_ = AllocateDevice(
          config_.full_q_head_count * config_.state_capacity_tokens *
          sizeof(float));
    }
    router_and_shared_ = AllocateDevice(257 * sizeof(float));
    router_logits_ = router_and_shared_;
    shared_gate_ = Offset(router_and_shared_, 256 * sizeof(float));
    selected_positions_ = AllocateDevice(8 * sizeof(std::uint32_t));
    router_weights_ = AllocateDevice(8 * sizeof(float));
    gate_up_ = AllocateDevice(kFfnGateUpValues * sizeof(float));
    selected_down_ = AllocateDevice(kSelectedRows * sizeof(float));
    shared_down_ = AllocateDevice(kSharedRows * sizeof(float));
    logits_ = AllocateDevice(kVocabularySize * sizeof(float));
    partial_top_ids_ = AllocateDevice(kLmHeadBlockCount * 8 * sizeof(int));
    partial_top_values_ =
        AllocateDevice(kLmHeadBlockCount * 8 * sizeof(float));
  }

  void AllocateState() {
    for (int layer = 0; layer < kPackedTokenLayerCount; ++layer) {
      if (IsFullAttentionLayer(layer)) {
        const auto value_count = static_cast<std::size_t>(
            config_.state_capacity_tokens * kFullKvValues);
        if (config_.use_int8_block32_kv_gqa) {
          const auto value_bytes = value_count * sizeof(std::int8_t);
          const auto scale_bytes = static_cast<std::size_t>(
              config_.state_capacity_tokens *
              kFullKvScaleGroupsPerToken * sizeof(std::uint16_t));
          const auto hot_values = static_cast<std::size_t>(
              kFullKvHotWindowTokens * kFullKvValues);
          const auto hot_k_bytes = hot_values * sizeof(float);
          const auto hot_v_bytes = hot_values * sizeof(float);
          full_state_[layer].k = AllocateDevice(value_bytes);
          full_state_[layer].v = AllocateDevice(value_bytes);
          full_state_[layer].k_scales = AllocateDevice(scale_bytes);
          full_state_[layer].v_scales = AllocateDevice(scale_bytes);
          full_state_[layer].hot_k = AllocateDevice(hot_k_bytes);
          full_state_[layer].hot_v = AllocateDevice(hot_v_bytes);
          timing_.resident_state_bytes +=
              2 * (value_bytes + scale_bytes) + hot_k_bytes + hot_v_bytes;
        } else {
          const auto bytes = value_count * sizeof(float);
          full_state_[layer].k = AllocateDevice(bytes);
          full_state_[layer].v = AllocateDevice(bytes);
          timing_.resident_state_bytes += 2 * bytes;
        }
      } else {
        const auto conv_bytes =
            static_cast<std::size_t>(kLinearConvStateValues) * sizeof(float);
        const auto recurrent_bytes =
            static_cast<std::size_t>(kLinearRecurrentStateValues) * sizeof(float);
        linear_state_[layer].conv = AllocateDevice(conv_bytes);
        linear_state_[layer].recurrent = AllocateDevice(recurrent_bytes);
        timing_.resident_state_bytes += conv_bytes + recurrent_bytes;
      }
    }
  }

  void UploadState() {
    for (int layer = 0; layer < kPackedTokenLayerCount; ++layer) {
      if (IsFullAttentionLayer(layer)) {
        const auto& k = initial_state_.full_k_history[layer];
        const auto& v = initial_state_.full_v_history[layer];
        Require(k.size() == v.size(), "full-attention K/V state size mismatch");
        Require(k.size() <= config_.state_capacity_tokens * kFullKvValues,
                "full-attention initial state exceeds capacity");
        if (config_.use_int8_block32_kv_gqa) {
          const auto value_bytes = static_cast<std::size_t>(
              config_.state_capacity_tokens * kFullKvValues);
          const auto scale_bytes = static_cast<std::size_t>(
              config_.state_capacity_tokens *
              kFullKvScaleGroupsPerToken * sizeof(std::uint16_t));
          const auto hot_values = static_cast<std::size_t>(
              kFullKvHotWindowTokens * kFullKvValues);
          const auto hot_k_bytes = hot_values * sizeof(float);
          const auto hot_v_bytes = hot_values * sizeof(float);
          ZeroDevice(full_state_[layer].k, value_bytes);
          ZeroDevice(full_state_[layer].v, value_bytes);
          ZeroDevice(full_state_[layer].k_scales, scale_bytes);
          ZeroDevice(full_state_[layer].v_scales, scale_bytes);
          ZeroDevice(full_state_[layer].hot_k, hot_k_bytes);
          ZeroDevice(full_state_[layer].hot_v, hot_v_bytes);
          if (!k.empty()) {
            const auto compressed_k = CompressBlock32Kv(k);
            const auto compressed_v = CompressBlock32Kv(v);
            CopyToDevice(full_state_[layer].k, compressed_k.values.data(),
                         compressed_k.values.size());
            CopyToDevice(full_state_[layer].v, compressed_v.values.data(),
                         compressed_v.values.size());
            CopyToDevice(full_state_[layer].k_scales,
                         compressed_k.scales.data(),
                         compressed_k.scales.size() * sizeof(std::uint16_t));
            CopyToDevice(full_state_[layer].v_scales,
                         compressed_v.scales.data(),
                         compressed_v.scales.size() * sizeof(std::uint16_t));
            const auto token_count = k.size() / kFullKvValues;
            const auto hot_count = std::min<std::size_t>(
                token_count, kFullKvHotWindowTokens);
            const auto hot_begin = token_count - hot_count;
            std::vector<float> hot_k(hot_values, 0.0f);
            std::vector<float> hot_v(hot_values, 0.0f);
            for (std::size_t token = hot_begin; token < token_count; ++token) {
              const auto source = token * kFullKvValues;
              const auto destination =
                  (token % kFullKvHotWindowTokens) * kFullKvValues;
              std::copy_n(k.data() + source, kFullKvValues,
                          hot_k.data() + destination);
              std::copy_n(v.data() + source, kFullKvValues,
                          hot_v.data() + destination);
            }
            CopyToDevice(full_state_[layer].hot_k, hot_k.data(), hot_k_bytes);
            CopyToDevice(full_state_[layer].hot_v, hot_v.data(), hot_v_bytes);
          }
        } else if (!k.empty()) {
          CopyToDevice(full_state_[layer].k, k.data(),
                       k.size() * sizeof(float));
          CopyToDevice(full_state_[layer].v, v.data(),
                       v.size() * sizeof(float));
        }
      } else {
        const auto& conv = initial_state_.linear_conv[layer];
        const auto& recurrent = initial_state_.linear_recurrent[layer];
        if (!conv.empty()) {
          Require(conv.size() == kLinearConvStateValues,
                  "linear conv initial state size mismatch");
          CopyToDevice(linear_state_[layer].conv, conv.data(),
                       conv.size() * sizeof(float));
        }
        if (!recurrent.empty()) {
          Require(recurrent.size() == kLinearRecurrentStateValues,
                  "linear recurrent initial state size mismatch");
          CopyToDevice(linear_state_[layer].recurrent, recurrent.data(),
                       recurrent.size() * sizeof(float));
        }
      }
    }
  }

  const DeviceTensor& Weight(const std::string& name) const {
    const auto it = weights_.find(name);
    if (it == weights_.end()) Die("backend weight missing: " + name);
    return it->second;
  }

  ze_kernel_handle_t Kernel(const char* name) {
    ze_kernel_desc_t desc{ZE_STRUCTURE_TYPE_KERNEL_DESC};
    desc.pKernelName = name;
    ze_kernel_handle_t kernel = nullptr;
    Check(zeKernelCreate(module_, &desc, &kernel), "zeKernelCreate");
    kernels_.push_back(kernel);
    kernel_names_.emplace(kernel, name);
    return kernel;
  }

  void SetPointer(ze_kernel_handle_t kernel, std::uint32_t index,
                  void* pointer) {
    Check(zeKernelSetArgumentValue(kernel, index, sizeof(pointer), &pointer),
          "zeKernelSetArgumentValue(pointer)");
  }

  template <typename Value>
  void SetValue(ze_kernel_handle_t kernel, std::uint32_t index,
                const Value& value) {
    Check(zeKernelSetArgumentValue(kernel, index, sizeof(value), &value),
          "zeKernelSetArgumentValue(value)");
  }

  void Launch(ze_kernel_handle_t kernel, std::uint32_t global,
              std::uint32_t local) {
    Require(global > 0 && local > 0 && global % local == 0,
            "invalid kernel launch shape");
    Check(zeKernelSetGroupSize(kernel, local, 1, 1),
          "zeKernelSetGroupSize");
    ze_group_count_t groups{global / local, 1, 1};
    Check(zeCommandListAppendLaunchKernel(command_list_, kernel, &groups,
                                          nullptr, 0, nullptr),
          "zeCommandListAppendLaunchKernel");
    ++timing_.kernel_count;
    if (config_.profile_kernel_times) {
      Require(profile_timestamp_count_ + 1 < kMaxProfileTimestamps,
              "kernel profile timestamp capacity exceeded");
      ++profile_timestamp_count_;
      Check(zeCommandListAppendWriteGlobalTimestamp(
                command_list_, timestamps_ + profile_timestamp_count_,
                nullptr, 0, nullptr),
            "zeCommandListAppendWriteGlobalTimestamp(profile)");
      const auto name = kernel_names_.find(kernel);
      Require(name != kernel_names_.end(), "profiled kernel name missing");
      profile_kernel_names_.push_back(name->second);
    }
  }

  std::uint32_t SuggestedLocal(ze_kernel_handle_t kernel,
                               std::uint32_t global) const {
    std::uint32_t x = 0;
    std::uint32_t y = 0;
    std::uint32_t z = 0;
    Check(zeKernelSuggestGroupSize(kernel, global, 1, 1, &x, &y, &z),
          "zeKernelSuggestGroupSize");
    Require(x > 0 && y == 1 && z == 1 && global % x == 0,
            "suggested kernel group size is incompatible");
    return x;
  }

  void AppendRmsScale(void* input) {
    auto kernel = Kernel("rms_norm_hidden_scale_f64_parallel");
    SetPointer(kernel, 0, input);
    SetValue(kernel, 1, kHiddenSize);
    SetValue(kernel, 2, config_.rms_norm_epsilon);
    SetPointer(kernel, 3, rms_scale_);
    Launch(kernel, 256, 256);
  }

  void AppendRmsNorm(void* input, const std::string& weight_name, void* output,
                     bool scale_ready = false) {
    if (!scale_ready) AppendRmsScale(input);
    {
      auto kernel = Kernel("rms_norm_hidden_apply_q8_f32");
      SetPointer(kernel, 0, input);
      SetPointer(kernel, 1, Weight(weight_name).pointer);
      SetValue(kernel, 2, kHiddenSize);
      SetPointer(kernel, 3, rms_scale_);
      SetPointer(kernel, 4, output);
      SetPointer(kernel, 5, q8_qs_);
      SetPointer(kernel, 6, q8_bsums_);
      SetPointer(kernel, 7, q8_d_);
      Launch(kernel, kHiddenSize, 256);
    }
  }

  void AppendQ8(void* input, std::uint32_t values) {
    Require(values % 256 == 0, "Q8 input is not block aligned");
    auto kernel = Kernel("q8k_quantize_f32_blocks_with_bsums_parallel");
    SetPointer(kernel, 0, input);
    SetValue(kernel, 1, values / 256);
    SetPointer(kernel, 2, q8_qs_);
    SetPointer(kernel, 3, q8_bsums_);
    SetPointer(kernel, 4, q8_d_);
    Launch(kernel, values, 256);
  }

  void AppendMatvec(const std::string& weight_name, void* output) {
    const auto& weight = Weight(weight_name);
    if (weight.type == kSymmetricQ4Group32Type) {
      auto kernel = Kernel("q4s_group32_matvec_f32");
      kernel_names_[kernel] += "/" + weight_name + "/" +
          std::to_string(weight.rows) + "x" + std::to_string(weight.cols);
      SetPointer(kernel, 0, weight.pointer);
      SetPointer(kernel, 1, q8_qs_);
      SetPointer(kernel, 2, q8_d_);
      SetValue(kernel, 3, weight.rows);
      SetValue(kernel, 4, weight.cols);
      SetPointer(kernel, 5, output);
      Launch(kernel, weight.rows, 128);
      return;
    }
    if (weight.type == kSymmetricQ4Group64Type) {
      auto kernel = Kernel("q4s_group64_matvec_f32");
      kernel_names_[kernel] += "/" + weight_name + "/" +
          std::to_string(weight.rows) + "x" + std::to_string(weight.cols);
      SetPointer(kernel, 0, weight.pointer);
      SetPointer(kernel, 1, q8_qs_);
      SetPointer(kernel, 2, q8_d_);
      SetValue(kernel, 3, weight.rows);
      SetValue(kernel, 4, weight.cols);
      SetPointer(kernel, 5, output);
      Launch(kernel, weight.rows, 128);
      return;
    }
    if (weight.type == kSymmetricQ4Group128Type) {
      auto kernel = Kernel("q4s_group128_matvec_f32");
      kernel_names_[kernel] += "/" + weight_name + "/" +
          std::to_string(weight.rows) + "x" + std::to_string(weight.cols);
      SetPointer(kernel, 0, weight.pointer);
      SetPointer(kernel, 1, q8_qs_);
      SetPointer(kernel, 2, q8_d_);
      SetValue(kernel, 3, weight.rows);
      SetValue(kernel, 4, weight.cols);
      SetPointer(kernel, 5, output);
      Launch(kernel, weight.rows, 128);
      return;
    }
    if (weight.type == 12) {
      const bool rowblock16 = weight.blocks_per_row == 16;
      auto kernel = Kernel(rowblock16
                               ? "q4k_blockstripe16_matvec_group_subgroups"
                               : "q4k_x8_matvec_rowlane");
      const auto* tensor = find_tensor(index_, weight_name);
      const auto shape = std::to_string(weight.rows) + "x" +
          std::to_string(weight.blocks_per_row);
      kernel_names_[kernel] += "/" +
          (tensor != nullptr && !tensor->suffix.empty()
               ? tensor->suffix
               : weight_name) +
          "/" + shape;
      SetPointer(kernel, 0, weight.pointer);
      SetPointer(kernel, 1, q8_qs_);
      SetPointer(kernel, 2, q8_bsums_);
      SetPointer(kernel, 3, q8_d_);
      SetValue(kernel, 4, weight.blocks_per_row);
      const std::uint32_t row_groups = weight.rows / 8;
      SetValue(kernel, 5, row_groups);
      SetPointer(kernel, 6, output);
      if (rowblock16) {
        Launch(kernel, row_groups * 128, 128);
        return;
      }
      const std::uint32_t local = weight.rows >= 1024 && weight.rows % 1024 == 0
          ? 1024
          : SuggestedLocal(kernel, weight.rows);
      Launch(kernel, weight.rows, local);
      return;
    }
    if (weight.type == 14) {
      const bool local_q8 = weight.blocks_per_row == 8;
      auto kernel = Kernel(local_q8
                               ? "q6k_selected_down_matvec_rowstripe_localq8"
                               : "q6k_selected_down_matvec_rowstripe");
      const auto* tensor = find_tensor(index_, weight_name);
      const auto shape = std::to_string(weight.rows) + "x" +
          std::to_string(weight.blocks_per_row);
      kernel_names_[kernel] += "/" +
          (tensor != nullptr && !tensor->suffix.empty()
               ? tensor->suffix
               : weight_name) +
          "/" + shape;
      SetPointer(kernel, 0, weight.pointer);
      SetPointer(kernel, 1, q8_qs_);
      SetPointer(kernel, 2, q8_d_);
      SetValue(kernel, 3, weight.rows);
      SetValue(kernel, 4, weight.blocks_per_row);
      SetValue(kernel, 5, weight.rows_per_tile);
      SetPointer(kernel, 6, output);
      Launch(kernel, weight.rows, 128);
      return;
    }
    Die("quantized matvec requested for non-quantized weight");
  }

  void AppendF32Matvec(const std::string& weight_name, void* input,
                       void* output) {
    const auto& weight = Weight(weight_name);
    Require(weight.type == 0, "F32 matvec requires F32 weight");
    Require(weight.cols > 0, "F32 matvec tensor shape missing");
    auto kernel = Kernel("f32_matvec_row_f32");
    SetPointer(kernel, 0, weight.pointer);
    SetPointer(kernel, 1, input);
    SetValue(kernel, 2, weight.cols);
    SetValue(kernel, 3, weight.rows);
    SetPointer(kernel, 4, output);
    const std::uint32_t local = weight.rows >= 64 ? 64 : 1;
    const std::uint32_t global =
        ((weight.rows + local - 1) / local) * local;
    Launch(kernel, global, local);
  }

  void AppendVectorAddRmsScale(void* lhs, void* rhs, void* output) {
    auto kernel = Kernel("vector_add_rms_scale_f64_parallel");
    SetPointer(kernel, 0, lhs);
    SetPointer(kernel, 1, rhs);
    SetValue(kernel, 2, kHiddenSize);
    SetValue(kernel, 3, config_.rms_norm_epsilon);
    SetPointer(kernel, 4, output);
    SetPointer(kernel, 5, rms_scale_);
    Launch(kernel, 256, 256);
  }

  void RecordLinearLayer(int layer, void* hidden_in, void* hidden_out) {
    AppendRmsNorm(hidden_in, LayerTensorName(layer, "attn_norm.weight"), norm_);
    const auto* qkv_tensor =
        find_tensor(index_, LayerTensorName(layer, "attn_qkv.weight"));
    Require(qkv_tensor != nullptr, "linear QKV tensor missing");
    if (qkv_tensor->type == 12) {
      AppendMatvec(LayerTensorName(layer, "linear_qkv_coefficients.combined"),
                   linear_front_);
    } else {
      AppendMatvec(LayerTensorName(layer, "attn_qkv.weight"), qkv_);
      AppendMatvec(LayerTensorName(layer, "linear_coefficients.combined"),
                   linear_coefficients_);
    }
    {
      auto kernel = Kernel("linear_preconv_alpha_beta_conv_f32");
      SetPointer(kernel, 0, alpha_);
      SetPointer(kernel, 1, beta_);
      SetPointer(kernel, 2, Weight(LayerTensorName(layer, "ssm_dt.bias")).pointer);
      SetPointer(kernel, 3, Weight(LayerTensorName(layer, "ssm_a")).pointer);
      SetPointer(kernel, 4, linear_gate_);
      SetPointer(kernel, 5, beta_sigmoid_);
      SetPointer(kernel, 6, qkv_);
      SetPointer(kernel, 7, linear_state_[layer].conv);
      SetPointer(kernel, 8,
                 Weight(LayerTensorName(layer, "ssm_conv1d.weight")).pointer);
      SetValue(kernel, 9, kLinearQkvValues);
      const std::uint32_t kernel_size = 4;
      SetValue(kernel, 10, kernel_size);
      SetPointer(kernel, 11, conv_output_);
      SetPointer(kernel, 12, linear_state_[layer].conv);
      Launch(kernel, kLinearQkvValues, 256);
    }
    {
      auto kernel = Kernel(
          "linear_attn_postconv_fused_qk_l2_parallel_f32");
      SetPointer(kernel, 0, conv_output_);
      SetValue(kernel, 1, kLinearHeadDim);
      SetValue(kernel, 2, kLinearQueryHeads);
      SetValue(kernel, 3, kLinearValueHeads);
      SetValue(kernel, 4, config_.rms_norm_epsilon);
      SetPointer(kernel, 5, qkv_);
      SetPointer(kernel, 6, q_predelta_);
      SetPointer(kernel, 7, k_predelta_);
      SetPointer(kernel, 8, v_predelta_);
      SetPointer(kernel, 9, q_predelta_);
      SetPointer(kernel, 10, k_predelta_);
      Launch(kernel, 64 * 128, 128);
    }
    {
      auto kernel = Kernel(
          "linear_attn_delta_recurrent_final_cpu_shape_qk_local_f32");
      SetPointer(kernel, 0, q_predelta_);
      SetPointer(kernel, 1, k_predelta_);
      SetPointer(kernel, 2, v_predelta_);
      SetPointer(kernel, 3, linear_gate_);
      SetPointer(kernel, 4, beta_sigmoid_);
      SetPointer(kernel, 5, linear_state_[layer].recurrent);
      SetPointer(kernel, 6, z_);
      SetPointer(kernel, 7, Weight(LayerTensorName(layer, "ssm_norm.weight")).pointer);
      SetValue(kernel, 8, kLinearHeadDim);
      SetValue(kernel, 9, kLinearQueryHeads);
      SetValue(kernel, 10, kLinearValueHeads);
      SetValue(kernel, 11, config_.rms_norm_epsilon);
      SetPointer(kernel, 12, linear_attention_);
      SetPointer(kernel, 13, linear_state_[layer].recurrent);
      SetPointer(kernel, 14, linear_final_);
      Launch(kernel, 4096, 128);
    }
    AppendQ8(linear_final_, kLinearProjectionValues);
    AppendMatvec(LayerTensorName(layer, "ssm_out.weight"), projection_);
    AppendVectorAddRmsScale(hidden_in, projection_, attention_residual_);
    RecordFfn(layer, attention_residual_, hidden_out);
  }

  void RecordFullLayer(int layer, void* hidden_in, void* hidden_out) {
    AppendRmsNorm(hidden_in, LayerTensorName(layer, "attn_norm.weight"), norm_);
    const auto* v_tensor =
        find_tensor(index_, LayerTensorName(layer, "attn_v.weight"));
    Require(v_tensor != nullptr, "full-attention V tensor missing");
    if (v_tensor->type == 12) {
      AppendMatvec(LayerTensorName(layer, "full_qkv.combined"), full_front_);
    } else {
      AppendMatvec(LayerTensorName(layer, "full_qk.combined"), full_front_);
      if (Weight(LayerTensorName(layer, "attn_v.weight")).type == 0) {
        AppendF32Matvec(LayerTensorName(layer, "attn_v.weight"), norm_,
                        full_v_);
      } else {
        AppendMatvec(LayerTensorName(layer, "attn_v.weight"), full_v_);
      }
    }
    {
      auto kernel = Kernel("full_attn_qk_norm_rope_f32");
      SetPointer(kernel, 0, full_q_);
      SetPointer(kernel, 1, full_k_);
      SetPointer(kernel, 2,
                 Weight(LayerTensorName(layer, "attn_q_norm.weight")).pointer);
      SetPointer(kernel, 3,
                 Weight(LayerTensorName(layer, "attn_k_norm.weight")).pointer);
      SetPointer(kernel, 4, rope_cache_);
      SetValue(kernel, 5, static_cast<std::uint32_t>(config_.full_head_dim));
      SetValue(kernel, 6, static_cast<std::uint32_t>(config_.full_q_head_count));
      SetValue(kernel, 7, static_cast<std::uint32_t>(config_.full_kv_head_count));
      SetValue(kernel, 8, static_cast<std::uint32_t>(config_.rope_dimension_count));
      SetValue(kernel, 9, config_.rms_norm_epsilon);
      SetPointer(kernel, 10, full_q_rope_);
      SetPointer(kernel, 11, full_k_rope_);
      Launch(kernel, 4608, 128);
    }
    if (config_.use_int8_block32_kv_gqa) {
      {
        auto kernel = Kernel("full_attn_kv_append_i8_block32_control");
        SetPointer(kernel, 0, full_k_rope_);
        SetPointer(kernel, 1, full_v_);
        SetPointer(kernel, 2, token_control_);
        SetValue(kernel, 3,
                 static_cast<std::uint32_t>(config_.state_capacity_tokens));
        SetPointer(kernel, 4, full_state_[layer].k);
        SetPointer(kernel, 5, full_state_[layer].v);
        SetPointer(kernel, 6, full_state_[layer].k_scales);
        SetPointer(kernel, 7, full_state_[layer].v_scales);
        SetPointer(kernel, 8, full_state_[layer].hot_k);
        SetPointer(kernel, 9, full_state_[layer].hot_v);
        Launch(kernel, 2 * 2 * 8 * 32, 32);
      }
      {
        auto kernel = Kernel("full_attn_i8_block32_gqa_partial_control");
        SetPointer(kernel, 0, full_q_rope_);
        SetPointer(kernel, 1, full_state_[layer].k);
        SetPointer(kernel, 2, full_state_[layer].v);
        SetPointer(kernel, 3, full_state_[layer].k_scales);
        SetPointer(kernel, 4, full_state_[layer].v_scales);
        SetPointer(kernel, 5, full_state_[layer].hot_k);
        SetPointer(kernel, 6, full_state_[layer].hot_v);
        SetPointer(kernel, 7, token_control_);
        SetValue(kernel, 8,
                 static_cast<std::uint32_t>(config_.state_capacity_tokens));
        SetValue(kernel, 9, config_.attention_scale);
        SetPointer(kernel, 10, full_partial_max_);
        SetPointer(kernel, 11, full_partial_sum_);
        SetPointer(kernel, 12, full_partial_output_);
        const auto chunks = static_cast<std::uint32_t>(
            (config_.state_capacity_tokens + kFullAttentionChunkTokens - 1U) /
            kFullAttentionChunkTokens);
        Launch(kernel, config_.full_kv_head_count * chunks * 256U, 256);
      }
      {
        auto kernel = Kernel(
            "full_attn_i8_block32_gqa_reduce_gate_control");
        SetPointer(kernel, 0, full_partial_max_);
        SetPointer(kernel, 1, full_partial_sum_);
        SetPointer(kernel, 2, full_partial_output_);
        SetPointer(kernel, 3, full_q_);
        SetPointer(kernel, 4, token_control_);
        SetValue(kernel, 5,
                 static_cast<std::uint32_t>(config_.state_capacity_tokens));
        SetPointer(kernel, 6, full_pregate_);
        SetPointer(kernel, 7, full_gated_);
        Launch(kernel, 4096, 256);
      }
    } else {
      {
        auto kernel = Kernel("full_attn_kv_append_f32");
        SetPointer(kernel, 0, full_k_rope_);
        SetPointer(kernel, 1, full_v_);
        SetPointer(kernel, 2, token_control_);
        SetValue(kernel, 3,
                 static_cast<std::uint32_t>(config_.state_capacity_tokens));
        SetPointer(kernel, 4, full_state_[layer].k);
        SetPointer(kernel, 5, full_state_[layer].v);
        Launch(kernel, 512, 128);
      }
      {
        auto kernel = Kernel("full_attn_score_control_f32");
        SetPointer(kernel, 0, full_q_rope_);
        SetPointer(kernel, 1, full_state_[layer].k);
        SetPointer(kernel, 2, token_control_);
        SetValue(kernel, 3,
                 static_cast<std::uint32_t>(config_.state_capacity_tokens));
        SetValue(kernel, 4,
                 static_cast<std::uint32_t>(config_.full_head_dim));
        SetValue(kernel, 5,
                 static_cast<std::uint32_t>(config_.full_q_head_count));
        SetValue(kernel, 6,
                 static_cast<std::uint32_t>(config_.full_kv_head_count));
        SetValue(kernel, 7, config_.attention_scale);
        SetPointer(kernel, 8, full_scores_);
        const auto score_count = static_cast<std::uint32_t>(
            config_.full_q_head_count * config_.state_capacity_tokens);
        const std::uint32_t local = 128;
        const auto global = ((score_count + local - 1) / local) * local;
        Launch(kernel, global, local);
      }
      {
        auto kernel = Kernel("full_attn_apply_score_gate_control_f32");
        SetPointer(kernel, 0, full_scores_);
        SetPointer(kernel, 1, full_state_[layer].v);
        SetPointer(kernel, 2, full_q_);
        SetPointer(kernel, 3, token_control_);
        SetValue(kernel, 4,
                 static_cast<std::uint32_t>(config_.state_capacity_tokens));
        SetValue(kernel, 5,
                 static_cast<std::uint32_t>(config_.full_head_dim));
        SetValue(kernel, 6,
                 static_cast<std::uint32_t>(config_.full_q_head_count));
        SetValue(kernel, 7,
                 static_cast<std::uint32_t>(config_.full_kv_head_count));
        SetPointer(kernel, 8, full_pregate_);
        SetPointer(kernel, 9, full_gated_);
        Launch(kernel, 4096, 128);
      }
    }
    AppendQ8(full_gated_, 4096);
    AppendMatvec(LayerTensorName(layer, "attn_output.weight"), projection_);
    AppendVectorAddRmsScale(hidden_in, projection_, attention_residual_);
    RecordFfn(layer, attention_residual_, hidden_out);
  }

  void RecordFfn(int layer, void* residual, void* output) {
    AppendRmsNorm(residual, LayerTensorName(layer, "post_attention_norm.weight"),
                  norm_, true);
    {
      const auto& router =
          Weight(LayerTensorName(layer, "ffn_router_shared.combined"));
      auto kernel = Kernel("q8_router_matvec_group32_rows_parallel_f32_input");
      SetPointer(kernel, 0, router.pointer);
      SetPointer(kernel, 1, Offset(
          router.pointer, 257U * kHiddenSize * sizeof(std::int8_t)));
      SetPointer(kernel, 2, norm_);
      SetValue(kernel, 3, router.cols);
      SetValue(kernel, 4, std::uint32_t{32});
      SetPointer(kernel, 5, router_logits_);
      Launch(kernel, 257 * 64, 64);
    }
    {
      auto kernel = Kernel("router_logits_topk8_f32");
      SetPointer(kernel, 0, router_logits_);
      SetPointer(kernel, 1, shared_gate_);
      SetPointer(kernel, 2, selected_positions_);
      SetPointer(kernel, 3, router_weights_);
      Launch(kernel, 256, 256);
    }
    {
      auto kernel = Kernel(
          "q4k_x8_matvec_topk_indexed_expert8_plus_shared_localq8");
      SetPointer(kernel, 0,
                 Weight(LayerTensorName(layer, "ffn_gate_up_exps.weight")).pointer);
      SetPointer(kernel, 1,
                 Weight(LayerTensorName(layer,
                                        "ffn_gate_up_shexp.combined")).pointer);
      SetPointer(kernel, 2, selected_positions_);
      SetPointer(kernel, 3, q8_qs_);
      SetPointer(kernel, 4, q8_bsums_);
      SetPointer(kernel, 5, q8_d_);
      const std::uint32_t blocks = 8;
      const std::uint32_t row_groups = 128;
      const std::uint32_t experts = 256;
      SetValue(kernel, 6, blocks);
      SetValue(kernel, 7, row_groups);
      SetValue(kernel, 8, experts);
      SetPointer(kernel, 9, gate_up_);
      Launch(kernel, kFfnGateUpValues, 1024);
    }
    {
      auto kernel = Kernel("ffn_swiglu_q8_blocks_with_bsums_parallel");
      SetPointer(kernel, 0, gate_up_);
      SetPointer(kernel, 1, q8_qs_);
      SetPointer(kernel, 2, q8_bsums_);
      SetPointer(kernel, 3, q8_d_);
      Launch(kernel, 18 * 256, 256);
    }
    const auto down_name = LayerTensorName(layer, "ffn_down_exps.weight");
    const auto& down = Weight(down_name);
    if (down.type == kSymmetricQ4Group32Type ||
        down.type == kSymmetricQ4Group128Type) {
      const char* kernel_name =
          down.type == kSymmetricQ4Group32Type
              ? "q4s_group32_all_expert_down_topk8_plus_shared"
              : "q4s_group128_all_expert_down_topk8_plus_shared";
      auto kernel = Kernel(kernel_name);
      SetPointer(kernel, 0, down.pointer);
      SetPointer(kernel, 1,
                 Weight(LayerTensorName(layer, "ffn_down_shexp.weight")).pointer);
      SetPointer(kernel, 2, selected_positions_);
      SetPointer(kernel, 3, q8_qs_);
      SetPointer(kernel, 4, q8_d_);
      SetPointer(kernel, 5, Offset(q8_qs_, 4096));
      SetPointer(kernel, 6, Offset(q8_d_, 16 * sizeof(float)));
      SetPointer(kernel, 7, selected_down_);
      SetPointer(kernel, 8, shared_down_);
      Launch(kernel, kSelectedRows + kSharedRows, 128);
    } else if (down.type == 14) {
      auto kernel = Kernel(
          "q6k_all_expert_rowstripe_coalesced_topk8_plus_shared");
      SetPointer(kernel, 0, down.pointer);
      SetPointer(kernel, 1,
                 Weight(LayerTensorName(layer, "ffn_down_shexp.weight")).pointer);
      SetPointer(kernel, 2, selected_positions_);
      SetPointer(kernel, 3, q8_qs_);
      SetPointer(kernel, 4, q8_d_);
      SetPointer(kernel, 5, Offset(q8_qs_, 4096));
      SetPointer(kernel, 6, Offset(q8_d_, 16 * sizeof(float)));
      SetPointer(kernel, 7, selected_down_);
      SetPointer(kernel, 8, shared_down_);
      Launch(kernel, kSelectedRows + kSharedRows, 64);
    } else {
      auto kernel = Kernel("q4k_x8_all_expert_down_topk8_plus_shared");
      SetPointer(kernel, 0, down.pointer);
      SetPointer(kernel, 1,
                 Weight(LayerTensorName(layer, "ffn_down_shexp.weight")).pointer);
      SetPointer(kernel, 2, selected_positions_);
      SetPointer(kernel, 3, q8_qs_);
      SetPointer(kernel, 4, q8_bsums_);
      SetPointer(kernel, 5, q8_d_);
      SetPointer(kernel, 6, Offset(q8_qs_, 4096));
      SetPointer(kernel, 7, Offset(q8_bsums_, 256 * sizeof(std::int16_t)));
      SetPointer(kernel, 8, Offset(q8_d_, 16 * sizeof(float)));
      SetPointer(kernel, 9, selected_down_);
      SetPointer(kernel, 10, shared_down_);
      Launch(kernel, kSelectedRows + kSharedRows, 1024);
    }
    {
      auto kernel = Kernel("ffn_tail_fused_output_f32");
      SetPointer(kernel, 0, selected_down_);
      SetPointer(kernel, 1, router_weights_);
      SetPointer(kernel, 2, shared_down_);
      SetPointer(kernel, 3, shared_gate_);
      SetPointer(kernel, 4, residual);
      SetValue(kernel, 5, kHiddenSize);
      const std::uint32_t experts = 8;
      SetValue(kernel, 6, experts);
      SetPointer(kernel, 7, output);
      Launch(kernel, kHiddenSize, 256);
    }
  }

  void RecordCommandList() {
    profile_timestamp_count_ = 0;
    profile_kernel_names_.clear();
    Check(zeCommandListAppendWriteGlobalTimestamp(command_list_, timestamps_,
                                                   nullptr, 0, nullptr),
          "zeCommandListAppendWriteGlobalTimestamp(start)");
    {
      auto kernel = Kernel("q4k_embedding_row_decode_f32");
      SetPointer(kernel, 0, Weight("token_embd.weight").pointer);
      SetPointer(kernel, 1, token_control_);
      SetValue(kernel, 2, kVocabularySize);
      SetPointer(kernel, 3, hidden_a_);
      Launch(kernel, kHiddenSize, 256);
    }
    for (int layer = 0; layer < kPackedTokenLayerCount; ++layer) {
      void* input = layer % 2 == 0 ? hidden_a_ : hidden_b_;
      void* output = layer % 2 == 0 ? hidden_b_ : hidden_a_;
      if (IsFullAttentionLayer(layer)) {
        RecordFullLayer(layer, input, output);
      } else {
        RecordLinearLayer(layer, input, output);
      }
    }
    AppendRmsNorm(hidden_a_, "output_norm.weight", norm_);
    {
      const auto& weight = Weight("output.weight");
      Require(weight.type == kSymmetricQ4Group128Type,
              "LM head must use the locked group-128 layout");
      auto kernel = Kernel("q4s_group128_lm_head_topk8_f32");
      SetPointer(kernel, 0, weight.pointer);
      SetPointer(kernel, 1, q8_qs_);
      SetPointer(kernel, 2, q8_d_);
      SetValue(kernel, 3, weight.rows);
      SetValue(kernel, 4, weight.cols);
      SetPointer(kernel, 5, logits_);
      SetPointer(kernel, 6, partial_top_ids_);
      SetPointer(kernel, 7, partial_top_values_);
      Launch(kernel, weight.rows, 256);
    }
    Require(lm_head_exact_ != nullptr && lm_head_exact_rows_per_tile_ != 0U,
            "sparse exact LM head storage is unavailable");
    {
      auto kernel = Kernel("q6k_lm_head_sparse_exact_candidates");
      SetPointer(kernel, 0, lm_head_exact_);
      SetPointer(kernel, 1, q8_qs_);
      SetPointer(kernel, 2, q8_d_);
      SetPointer(kernel, 3, partial_top_ids_);
      constexpr std::uint32_t kCandidatesPerBlock = 2;
      SetValue(kernel, 4, kLmHeadBlockCount);
      SetValue(kernel, 5, kCandidatesPerBlock);
      SetValue(kernel, 6, Weight("output.weight").blocks_per_row);
      SetValue(kernel, 7, lm_head_exact_rows_per_tile_);
      SetPointer(kernel, 8, logits_);
      SetPointer(kernel, 9, partial_top_values_);
      SetValue(kernel, 10, 10.0f);
      constexpr std::uint32_t kCandidateLocalSize = 32;
      const std::uint32_t candidate_count =
          kLmHeadBlockCount * kCandidatesPerBlock;
      const auto candidate_global =
          ((candidate_count + kCandidateLocalSize - 1U) /
           kCandidateLocalSize) * kCandidateLocalSize;
      Launch(kernel, candidate_global, kCandidateLocalSize);
    }
    {
      auto kernel = Kernel("sort_partial_top8_blocks_in_place_f32");
      SetPointer(kernel, 0, partial_top_ids_);
      SetPointer(kernel, 1, partial_top_values_);
      SetValue(kernel, 2, kLmHeadBlockCount);
      Launch(kernel, kLmHeadBlockCount, 1);
    }
    {
      auto kernel = Kernel("f32_topk8_merge_blocks_parallel");
      SetPointer(kernel, 0, partial_top_ids_);
      SetPointer(kernel, 1, partial_top_values_);
      SetValue(kernel, 2, kLmHeadBlockCount);
      const std::uint32_t output_count = 8;
      SetValue(kernel, 3, output_count);
      SetPointer(kernel, 4, top_ids_);
      SetPointer(kernel, 5, top_values_);
      Launch(kernel, 128, 128);
    }
    if (config_.profile_kernel_times) {
      end_timestamp_index_ = profile_timestamp_count_;
    } else {
      end_timestamp_index_ = 1;
      Check(zeCommandListAppendWriteGlobalTimestamp(
                command_list_, timestamps_ + end_timestamp_index_,
                nullptr, 0, nullptr),
            "zeCommandListAppendWriteGlobalTimestamp(end)");
    }
    Check(zeCommandListClose(command_list_), "zeCommandListClose(run)");
    timing_.command_list_record_count = 1;
  }

  void Cleanup() {
    if (queue_ != nullptr) zeCommandQueueSynchronize(queue_, UINT64_MAX);
    for (auto kernel : kernels_) zeKernelDestroy(kernel);
    if (command_list_ != nullptr) zeCommandListDestroy(command_list_);
    if (immediate_list_ != nullptr) zeCommandListDestroy(immediate_list_);
    if (module_ != nullptr) zeModuleDestroy(module_);
    for (void* pointer : shared_allocations_) zeMemFree(context_, pointer);
    for (void* pointer : device_allocations_) zeMemFree(context_, pointer);
    if (queue_ != nullptr) zeCommandQueueDestroy(queue_);
    if (context_ != nullptr) zeContextDestroy(context_);
  }

  std::string model_path_;
  std::string module_path_;
  PackedTokenLevelZeroConfig config_;
  GgufModelIndex index_;
  PackedTokenProgram program_;
  PackedTokenStateSnapshot initial_state_;
  std::array<LinearState, kPackedTokenLayerCount> linear_state_{};
  std::array<FullState, kPackedTokenLayerCount> full_state_{};
  std::unordered_map<std::string, DeviceTensor> weights_;
  ze_driver_handle_t driver_ = nullptr;
  ze_device_handle_t device_ = nullptr;
  ze_context_handle_t context_ = nullptr;
  ze_command_queue_handle_t queue_ = nullptr;
  ze_command_list_handle_t command_list_ = nullptr;
  ze_command_list_handle_t immediate_list_ = nullptr;
  ze_module_handle_t module_ = nullptr;
  ze_device_properties_t properties_{ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES};
  std::vector<std::uint8_t> module_bytes_;
  std::vector<ze_kernel_handle_t> kernels_;
  std::unordered_map<ze_kernel_handle_t, std::string> kernel_names_;
  std::vector<void*> device_allocations_;
  std::vector<void*> shared_allocations_;
  std::uint32_t queue_ordinal_ = UINT32_MAX;
  std::string device_name_;
  double timestamp_ns_per_tick_ = 0.0;
  bool compiled_ = false;
  std::uint64_t last_token_position_ = 0;
  std::size_t profile_timestamp_count_ = 0;
  std::size_t end_timestamp_index_ = 1;
  std::vector<std::string> profile_kernel_names_;
  PackedTokenLevelZeroTiming timing_;
  std::uint64_t* token_control_ = nullptr;
  float* rope_cache_ = nullptr;
  std::uint64_t* timestamps_ = nullptr;
  std::int32_t* top_ids_ = nullptr;
  float* top_values_ = nullptr;
  void* hidden_a_ = nullptr;
  void* hidden_b_ = nullptr;
  void* norm_ = nullptr;
  void* rms_scale_ = nullptr;
  void* attention_residual_ = nullptr;
  void* q8_qs_ = nullptr;
  void* q8_bsums_ = nullptr;
  void* q8_d_ = nullptr;
  void* qkv_ = nullptr;
  void* linear_front_ = nullptr;
  void* linear_coefficients_ = nullptr;
  void* alpha_ = nullptr;
  void* beta_ = nullptr;
  void* linear_gate_ = nullptr;
  void* beta_sigmoid_ = nullptr;
  void* z_ = nullptr;
  void* conv_output_ = nullptr;
  void* q_predelta_ = nullptr;
  void* k_predelta_ = nullptr;
  void* v_predelta_ = nullptr;
  void* linear_attention_ = nullptr;
  void* linear_final_ = nullptr;
  void* projection_ = nullptr;
  void* full_front_ = nullptr;
  void* full_q_ = nullptr;
  void* full_k_ = nullptr;
  void* full_v_ = nullptr;
  void* full_q_rope_ = nullptr;
  void* full_k_rope_ = nullptr;
  void* full_pregate_ = nullptr;
  void* full_gated_ = nullptr;
  void* full_scores_ = nullptr;
  void* full_partial_max_ = nullptr;
  void* full_partial_sum_ = nullptr;
  void* full_partial_output_ = nullptr;
  void* router_logits_ = nullptr;
  void* router_and_shared_ = nullptr;
  void* selected_positions_ = nullptr;
  void* router_weights_ = nullptr;
  void* shared_gate_ = nullptr;
  void* gate_up_ = nullptr;
  void* selected_down_ = nullptr;
  void* shared_down_ = nullptr;
  void* logits_ = nullptr;
  void* lm_head_exact_ = nullptr;
  std::uint32_t lm_head_exact_rows_per_tile_ = 0;
  void* partial_top_ids_ = nullptr;
  void* partial_top_values_ = nullptr;
};

PackedTokenLevelZeroBackend::PackedTokenLevelZeroBackend(
    std::string model_path,
    std::string native_module_path,
    PackedTokenLevelZeroConfig config)
    : impl_(std::make_unique<Impl>(std::move(model_path),
                                  std::move(native_module_path),
                                  std::move(config))) {}

PackedTokenLevelZeroBackend::~PackedTokenLevelZeroBackend() = default;
PackedTokenLevelZeroBackend::PackedTokenLevelZeroBackend(
    PackedTokenLevelZeroBackend&&) noexcept = default;
PackedTokenLevelZeroBackend& PackedTokenLevelZeroBackend::operator=(
    PackedTokenLevelZeroBackend&&) noexcept = default;

void PackedTokenLevelZeroBackend::LoadState(
    const PackedTokenStateSnapshot& state) {
  impl_->LoadState(state);
}

PackedTokenStateSnapshot PackedTokenLevelZeroBackend::ReadState() const {
  return impl_->ReadState();
}

std::vector<float> PackedTokenLevelZeroBackend::ReadLogits() const {
  return impl_->ReadLogits();
}

void PackedTokenLevelZeroBackend::Compile(const PackedTokenProgram& program) {
  impl_->Compile(program);
}

std::vector<PackedTokenTopKRow> PackedTokenLevelZeroBackend::SubmitToken(
    const PackedTokenSubmission& submission) {
  return impl_->SubmitToken(submission);
}

PackedTokenLevelZeroTiming PackedTokenLevelZeroBackend::last_timing() const {
  return impl_->timing();
}

const std::string& PackedTokenLevelZeroBackend::device_name() const {
  return impl_->device_name();
}

}  // namespace iq36
