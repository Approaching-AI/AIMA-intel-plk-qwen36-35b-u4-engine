#include "intel_qwen36/gguf_loader.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kLayerIndex = 3;
constexpr int kSourceTokenPosition = 15;
constexpr std::uint64_t kHistoryTokenCount = 16;
constexpr std::uint64_t kHeadDim = 256;
constexpr std::uint64_t kQHeadCount = 16;
constexpr std::uint64_t kKvHeadCount = 2;
constexpr std::uint64_t kQSize = kQHeadCount * kHeadDim;
constexpr std::uint64_t kKvSize = kKvHeadCount * kHeadDim;
constexpr float kAttentionScale = 0.0625f;
constexpr double kMismatchThreshold = 5e-3;
constexpr double kMaxAbsDiffThreshold = 5e-3;
constexpr double kRmseThreshold = 1e-3;
constexpr double kMinCosine = 0.99999;

struct ValueStats {
  std::uint64_t count = 0;
  double min = 0.0;
  double max = 0.0;
  double abs_sum = 0.0;
  double l2 = 0.0;
  bool finite = false;
  bool nonzero = false;
};

struct ModeResult {
  std::string name;
  iq36::Qwen36FullAttentionCoreResult native;
  iq36::VectorCompareStats comparison;
  ValueStats native_stats;
  bool passed = false;
};

void require(bool ok, const char* message) {
  if (!ok) {
    throw std::runtime_error(message);
  }
}

std::string json_escape(const std::string& value) {
  std::string out;
  out.reserve(value.size() + 8);
  for (const char ch : value) {
    switch (ch) {
      case '\\':
        out += "\\\\";
        break;
      case '"':
        out += "\\\"";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        out += ch;
        break;
    }
  }
  return out;
}

std::string join_path(const std::string& dir, const std::string& name) {
  if (dir.empty() || dir.back() == '/') {
    return dir + name;
  }
  return dir + "/" + name;
}

std::string two_digit(std::uint64_t value) {
  if (value < 10) {
    return "0" + std::to_string(value);
  }
  return std::to_string(value);
}

std::uint64_t metadata_uint(const iq36::GgufModelIndex& index,
                            const std::string& key,
                            std::uint64_t fallback) {
  const auto found = index.metadata.find(key);
  if (found == index.metadata.end()) {
    return fallback;
  }
  const auto& value = found->second;
  if (value.kind == iq36::GgufMetadataValue::Kind::kUInt) {
    return value.uint_value;
  }
  if (value.kind == iq36::GgufMetadataValue::Kind::kInt &&
      value.int_value >= 0) {
    return static_cast<std::uint64_t>(value.int_value);
  }
  return fallback;
}

ValueStats stats_from_values(const std::vector<float>& values) {
  ValueStats stats;
  stats.count = values.size();
  stats.finite = !values.empty();
  stats.min = std::numeric_limits<double>::infinity();
  stats.max = -std::numeric_limits<double>::infinity();
  for (const auto value : values) {
    if (!std::isfinite(value)) {
      stats.finite = false;
      continue;
    }
    const double as_double = value;
    stats.min = std::min(stats.min, as_double);
    stats.max = std::max(stats.max, as_double);
    stats.abs_sum += std::abs(as_double);
    stats.l2 += as_double * as_double;
  }
  if (values.empty()) {
    stats.min = 0.0;
    stats.max = 0.0;
  }
  stats.nonzero = stats.abs_sum > 0.0;
  return stats;
}

std::vector<float> flatten_history(
    const std::vector<std::vector<float>>& history) {
  std::vector<float> flat;
  for (const auto& item : history) {
    flat.insert(flat.end(), item.begin(), item.end());
  }
  return flat;
}

bool vector_compare_passed(const iq36::VectorCompareStats& stats) {
  return stats.same_size && stats.finite && stats.mismatch_count == 0 &&
         stats.max_abs_diff <= kMaxAbsDiffThreshold &&
         stats.rmse <= kRmseThreshold && stats.cosine >= kMinCosine;
}

std::uint32_t float_bits(float value) {
  std::uint32_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value), "unexpected float size");
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

float bits_float(std::uint32_t bits) {
  float value = 0.0f;
  static_assert(sizeof(bits) == sizeof(value), "unexpected float size");
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

std::uint16_t float_to_half_bits(float value) {
  const std::uint32_t bits = float_bits(value);
  const std::uint16_t sign = static_cast<std::uint16_t>((bits >> 16) & 0x8000u);
  const std::uint32_t exponent = (bits >> 23) & 0xffu;
  std::uint32_t mantissa = bits & 0x7fffffu;

  if (exponent == 0xffu) {
    return static_cast<std::uint16_t>(
        sign | (mantissa == 0 ? 0x7c00u : 0x7e00u));
  }

  const int half_exponent = static_cast<int>(exponent) - 127 + 15;
  if (half_exponent >= 31) {
    return static_cast<std::uint16_t>(sign | 0x7c00u);
  }
  if (half_exponent <= 0) {
    if (half_exponent < -10) {
      return sign;
    }
    mantissa |= 0x800000u;
    const int shift = 14 - half_exponent;
    std::uint32_t half_mantissa = mantissa >> shift;
    const std::uint32_t round_bit = (mantissa >> (shift - 1)) & 1u;
    const std::uint32_t sticky_mask = (1u << (shift - 1)) - 1u;
    const std::uint32_t sticky = mantissa & sticky_mask;
    if (round_bit != 0 && (sticky != 0 || (half_mantissa & 1u) != 0)) {
      ++half_mantissa;
    }
    return static_cast<std::uint16_t>(sign | half_mantissa);
  }

  std::uint32_t half_mantissa = mantissa >> 13;
  std::uint32_t half_exp_bits =
      static_cast<std::uint32_t>(half_exponent) << 10;
  const std::uint32_t round_bits = mantissa & 0x1fffu;
  if (round_bits > 0x1000u ||
      (round_bits == 0x1000u && (half_mantissa & 1u) != 0)) {
    ++half_mantissa;
    if (half_mantissa == 0x400u) {
      half_mantissa = 0;
      half_exp_bits += 0x400u;
      if (half_exp_bits >= 0x7c00u) {
        return static_cast<std::uint16_t>(sign | 0x7c00u);
      }
    }
  }
  return static_cast<std::uint16_t>(sign | half_exp_bits | half_mantissa);
}

float half_bits_to_float(std::uint16_t half) {
  const std::uint32_t sign =
      static_cast<std::uint32_t>(half & 0x8000u) << 16;
  std::uint32_t exponent = (half >> 10) & 0x1fu;
  std::uint32_t mantissa = half & 0x03ffu;
  std::uint32_t bits = 0;
  if (exponent == 0) {
    if (mantissa == 0) {
      bits = sign;
    } else {
      int normalized_exponent = 1;
      while ((mantissa & 0x0400u) == 0) {
        mantissa <<= 1;
        --normalized_exponent;
      }
      mantissa &= 0x03ffu;
      const auto exponent_bits =
          static_cast<std::uint32_t>(normalized_exponent + 127 - 15);
      bits = sign | (exponent_bits << 23) | (mantissa << 13);
    }
  } else if (exponent == 31) {
    bits = sign | 0x7f800000u | (mantissa << 13);
  } else {
    exponent = exponent + 127 - 15;
    bits = sign | (exponent << 23) | (mantissa << 13);
  }
  return bits_float(bits);
}

float round_to_fp16(float value) {
  return half_bits_to_float(float_to_half_bits(value));
}

std::vector<float> round_vector_to_fp16(const std::vector<float>& values) {
  std::vector<float> rounded;
  rounded.reserve(values.size());
  for (const auto value : values) {
    rounded.push_back(round_to_fp16(value));
  }
  return rounded;
}

std::vector<std::vector<float>> round_history_to_fp16(
    const std::vector<std::vector<float>>& history) {
  std::vector<std::vector<float>> rounded;
  rounded.reserve(history.size());
  for (const auto& item : history) {
    rounded.push_back(round_vector_to_fp16(item));
  }
  return rounded;
}

void write_u64_vector(const std::vector<std::uint64_t>& values) {
  std::cout << "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0) {
      std::cout << ",";
    }
    std::cout << values[i];
  }
  std::cout << "]";
}

void write_value_stats(const ValueStats& stats) {
  std::cout << "{";
  std::cout << "\"abs_sum\":" << stats.abs_sum << ",";
  std::cout << "\"count\":" << stats.count << ",";
  std::cout << "\"finite\":" << (stats.finite ? "true" : "false") << ",";
  std::cout << "\"l2\":" << stats.l2 << ",";
  std::cout << "\"max\":" << stats.max << ",";
  std::cout << "\"min\":" << stats.min << ",";
  std::cout << "\"nonzero\":" << (stats.nonzero ? "true" : "false");
  std::cout << "}";
}

void write_compare_stats(const iq36::VectorCompareStats& stats) {
  std::cout << "{";
  std::cout << "\"compared_value_count\":" << stats.compared_value_count << ",";
  std::cout << "\"cosine\":" << stats.cosine << ",";
  std::cout << "\"finite\":" << (stats.finite ? "true" : "false") << ",";
  std::cout << "\"finite_pair_count\":" << stats.finite_pair_count << ",";
  std::cout << "\"lhs_l2\":" << stats.lhs_l2 << ",";
  std::cout << "\"lhs_value_count\":" << stats.lhs_value_count << ",";
  std::cout << "\"max_abs_diff\":" << stats.max_abs_diff << ",";
  std::cout << "\"mean_abs_diff\":" << stats.mean_abs_diff << ",";
  std::cout << "\"mismatch_count\":" << stats.mismatch_count << ",";
  std::cout << "\"rhs_l2\":" << stats.rhs_l2 << ",";
  std::cout << "\"rhs_value_count\":" << stats.rhs_value_count << ",";
  std::cout << "\"rmse\":" << stats.rmse << ",";
  std::cout << "\"same_size\":" << (stats.same_size ? "true" : "false");
  std::cout << "}";
}

void write_mode_result(const ModeResult& mode) {
  std::cout << "{";
  std::cout << "\"comparison\":";
  write_compare_stats(mode.comparison);
  std::cout << ",\"native_vector\":";
  write_value_stats(mode.native_stats);
  std::cout << ",\"passed\":" << (mode.passed ? "true" : "false");
  std::cout << "}";
}

ModeResult run_mode(const std::string& name,
                    const std::vector<float>& q_rope,
                    const std::vector<std::vector<float>>& k_history,
                    const std::vector<std::vector<float>>& v_history,
                    const std::vector<float>& oracle_attn_pregate) {
  ModeResult mode;
  mode.name = name;
  mode.native = iq36::run_qwen36_full_attention_core(
      q_rope, k_history, v_history, kHeadDim, kQHeadCount, kKvHeadCount,
      kAttentionScale);
  mode.comparison = iq36::compare_vectors(
      mode.native.attn_pregate, oracle_attn_pregate, kMismatchThreshold);
  mode.native_stats = stats_from_values(mode.native.attn_pregate);
  mode.passed = vector_compare_passed(mode.comparison);
  return mode;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    require(argc == 3,
            "usage: iq36-full-attn-core-compare "
            "<model.gguf> <history-payload-dir>");
    const std::string model_path = argv[1];
    const std::string payload_dir = argv[2];

    const auto index = iq36::parse_gguf_model_index(model_path);
    const auto load_map = iq36::validate_qwen36_load_map(index);
    const auto full_attention_interval = metadata_uint(
        index, "qwen35moe.full_attention_interval", 0);
    const auto q_head_count = metadata_uint(
        index, "qwen35moe.attention.head_count", 0);
    const auto kv_head_count = metadata_uint(
        index, "qwen35moe.attention.head_count_kv", 0);
    const auto key_length = metadata_uint(
        index, "qwen35moe.attention.key_length", 0);
    const auto value_length = metadata_uint(
        index, "qwen35moe.attention.value_length", 0);

    const auto prefix = std::string("blk.") + std::to_string(kLayerIndex) + ".";
    const auto* q_tensor = iq36::find_tensor(index, prefix + "attn_q.weight");
    const auto* k_tensor = iq36::find_tensor(index, prefix + "attn_k.weight");
    const auto* v_tensor = iq36::find_tensor(index, prefix + "attn_v.weight");
    require(q_tensor != nullptr && k_tensor != nullptr && v_tensor != nullptr,
            "L3 full-attention q/k/v tensor set is incomplete");
    const bool tensor_shapes_ok =
        q_tensor->type == 12 &&
        q_tensor->dims == std::vector<std::uint64_t>{2048, 8192} &&
        k_tensor->type == 12 &&
        k_tensor->dims == std::vector<std::uint64_t>{2048, kKvSize} &&
        v_tensor->dims == std::vector<std::uint64_t>{2048, kKvSize} &&
        (v_tensor->type == 12 || v_tensor->type == 14);
    const bool metadata_ok =
        full_attention_interval == 4 &&
        q_head_count == kQHeadCount &&
        kv_head_count == kKvHeadCount &&
        key_length == kHeadDim &&
        value_length == kHeadDim;

    const auto q_rope = iq36::read_f32_vector_file(
        join_path(payload_dir, "tok15_q_rope.bin"));
    const auto oracle_attn_pregate = iq36::read_f32_vector_file(
        join_path(payload_dir, "tok15_attn_pregate.bin"));
    std::vector<std::vector<float>> k_history;
    std::vector<std::vector<float>> v_history;
    k_history.reserve(kHistoryTokenCount);
    v_history.reserve(kHistoryTokenCount);
    for (std::uint64_t token = 0; token < kHistoryTokenCount; ++token) {
      const auto token_prefix = std::string("tok") + two_digit(token);
      k_history.push_back(iq36::read_f32_vector_file(
          join_path(payload_dir, token_prefix + "_k_rope.bin")));
      v_history.push_back(iq36::read_f32_vector_file(
          join_path(payload_dir, token_prefix + "_v.bin")));
    }

    const auto k_history_flat = flatten_history(k_history);
    const auto v_history_flat = flatten_history(v_history);
    const auto q_stats = stats_from_values(q_rope);
    const auto k_stats = stats_from_values(k_history_flat);
    const auto v_stats = stats_from_values(v_history_flat);
    const auto oracle_stats = stats_from_values(oracle_attn_pregate);
    const bool counts_ok =
        q_rope.size() == kQSize &&
        oracle_attn_pregate.size() == kQSize &&
        k_history.size() == kHistoryTokenCount &&
        v_history.size() == kHistoryTokenCount &&
        std::all_of(k_history.begin(), k_history.end(), [](const auto& item) {
          return item.size() == kKvSize;
        }) &&
        std::all_of(v_history.begin(), v_history.end(), [](const auto& item) {
          return item.size() == kKvSize;
        });
    const bool stats_ok =
        q_stats.finite && q_stats.nonzero &&
        k_stats.finite && k_stats.nonzero &&
        v_stats.finite && v_stats.nonzero &&
        oracle_stats.finite && oracle_stats.nonzero;

    const auto f32_mode = run_mode(
        "f32_source_payload", q_rope, k_history, v_history,
        oracle_attn_pregate);
    const auto k_history_fp16 = round_history_to_fp16(k_history);
    const auto v_history_fp16 = round_history_to_fp16(v_history);
    const auto fp16_mode = run_mode(
        "fp16_kv_cache", q_rope, k_history_fp16, v_history_fp16,
        oracle_attn_pregate);
    const ModeResult* selected = &f32_mode;
    if (!selected->passed && fp16_mode.passed) {
      selected = &fp16_mode;
    } else if (!selected->passed &&
               fp16_mode.comparison.rmse < f32_mode.comparison.rmse) {
      selected = &fp16_mode;
    }

    const bool passed =
        load_map.ready && tensor_shapes_ok && metadata_ok && counts_ok &&
        stats_ok && selected->passed;

    std::cout << std::setprecision(17);
    std::cout << "{";
    std::cout << "\"attention_parameters\":{";
    std::cout << "\"attention_scale\":" << kAttentionScale << ",";
    std::cout << "\"gqa_group\":" << (kQHeadCount / kKvHeadCount) << ",";
    std::cout << "\"head_dim\":" << kHeadDim << ",";
    std::cout << "\"history_token_count\":" << kHistoryTokenCount << ",";
    std::cout << "\"kv_head_count\":" << kKvHeadCount << ",";
    std::cout << "\"q_head_count\":" << kQHeadCount;
    std::cout << "}";
    std::cout << ",\"history_vectors\":{";
    std::cout << "\"k_history\":";
    write_value_stats(k_stats);
    std::cout << ",\"q_rope\":";
    write_value_stats(q_stats);
    std::cout << ",\"v_history\":";
    write_value_stats(v_stats);
    std::cout << "}";
    std::cout << ",\"layer_index\":" << kLayerIndex;
    std::cout << ",\"load_map_ready\":" << (load_map.ready ? "true" : "false");
    std::cout << ",\"metadata\":{";
    std::cout << "\"full_attention_interval\":" << full_attention_interval << ",";
    std::cout << "\"head_count\":" << q_head_count << ",";
    std::cout << "\"head_count_kv\":" << kv_head_count << ",";
    std::cout << "\"key_length\":" << key_length << ",";
    std::cout << "\"ok\":" << (metadata_ok ? "true" : "false") << ",";
    std::cout << "\"value_length\":" << value_length;
    std::cout << "}";
    std::cout << ",\"model_path\":\"" << json_escape(model_path) << "\"";
    std::cout << ",\"modes\":{";
    std::cout << "\"f32_source_payload\":";
    write_mode_result(f32_mode);
    std::cout << ",\"fp16_kv_cache\":";
    write_mode_result(fp16_mode);
    std::cout << "}";
    std::cout << ",\"oracle_vector\":";
    write_value_stats(oracle_stats);
    std::cout << ",\"passed\":" << (passed ? "true" : "false");
    std::cout << ",\"prompt_case_id\":\"short_math_001\"";
    std::cout << ",\"schema_version\":\"intel-qwen36-engine-full-attn-core-compare-v0\"";
    std::cout << ",\"selected_mode\":\"" << json_escape(selected->name) << "\"";
    std::cout << ",\"source_token_position\":" << kSourceTokenPosition;
    std::cout << ",\"tensors\":{";
    std::cout << "\"k\":{\"dims\":";
    write_u64_vector(k_tensor->dims);
    std::cout << ",\"name\":\"" << json_escape(k_tensor->name)
              << "\",\"type_name\":\"" << iq36::ggml_type_name(k_tensor->type)
              << "\"},";
    std::cout << "\"q\":{\"dims\":";
    write_u64_vector(q_tensor->dims);
    std::cout << ",\"name\":\"" << json_escape(q_tensor->name)
              << "\",\"type_name\":\"" << iq36::ggml_type_name(q_tensor->type)
              << "\"},";
    std::cout << "\"shape_ok\":" << (tensor_shapes_ok ? "true" : "false") << ",";
    std::cout << "\"v\":{\"dims\":";
    write_u64_vector(v_tensor->dims);
    std::cout << ",\"name\":\"" << json_escape(v_tensor->name)
              << "\",\"type_name\":\"" << iq36::ggml_type_name(v_tensor->type)
              << "\"}";
    std::cout << "}";
    std::cout << ",\"thresholds\":{";
    std::cout << "\"max_abs_diff\":" << kMaxAbsDiffThreshold << ",";
    std::cout << "\"min_cosine\":" << kMinCosine << ",";
    std::cout << "\"mismatch_abs_diff\":" << kMismatchThreshold << ",";
    std::cout << "\"rmse\":" << kRmseThreshold;
    std::cout << "}";
    std::cout << "}\n";
    return passed ? 0 : 2;
  } catch (const std::exception& exc) {
    std::cerr << "iq36-full-attn-core-compare failed: " << exc.what() << "\n";
    return 1;
  }
}
