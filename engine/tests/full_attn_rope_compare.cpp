#include "intel_qwen36/gguf_loader.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kLayerIndex = 3;
constexpr int kQSize = 4096;
constexpr int kKvSize = 512;
constexpr int kHeadDim = 256;
constexpr int kSourceTokenPosition = 15;
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

float metadata_float(const iq36::GgufModelIndex& index,
                     const std::string& key,
                     float fallback) {
  const auto found = index.metadata.find(key);
  if (found == index.metadata.end()) {
    return fallback;
  }
  const auto& value = found->second;
  if (value.kind == iq36::GgufMetadataValue::Kind::kFloat) {
    return static_cast<float>(value.float_value);
  }
  return fallback;
}

std::vector<std::int64_t> metadata_int_array(
    const iq36::GgufModelIndex& index,
    const std::string& key,
    const std::vector<std::int64_t>& fallback) {
  const auto found = index.metadata.find(key);
  if (found == index.metadata.end()) {
    return fallback;
  }
  const auto& value = found->second;
  if (value.kind == iq36::GgufMetadataValue::Kind::kArray &&
      !value.int_array.empty()) {
    return value.int_array;
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

bool vector_compare_passed(const iq36::VectorCompareStats& stats) {
  return stats.same_size && stats.finite && stats.mismatch_count == 0 &&
         stats.max_abs_diff <= kMaxAbsDiffThreshold &&
         stats.rmse <= kRmseThreshold && stats.cosine >= kMinCosine;
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

void write_i64_vector(const std::vector<std::int64_t>& values) {
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

void write_named_stats(const std::vector<std::pair<std::string, ValueStats>>& values) {
  std::cout << "{";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0) {
      std::cout << ",";
    }
    std::cout << "\"" << json_escape(values[i].first) << "\":";
    write_value_stats(values[i].second);
  }
  std::cout << "}";
}

void write_named_comparisons(
    const std::vector<std::pair<std::string, iq36::VectorCompareStats>>& values) {
  std::cout << "{";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0) {
      std::cout << ",";
    }
    std::cout << "\"" << json_escape(values[i].first) << "\":";
    write_compare_stats(values[i].second);
  }
  std::cout << "}";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    require(argc == 3,
            "usage: iq36-full-attn-rope-compare "
            "<model.gguf> <oracle-payload-dir>");
    const std::string model_path = argv[1];
    const std::string payload_dir = argv[2];

    const auto index = iq36::parse_gguf_model_index(model_path);
    const auto load_map = iq36::validate_qwen36_load_map(index);
    const auto rope_dimension_count = metadata_uint(
        index, "qwen35moe.rope.dimension_count", 64);
    const auto rope_context_length = metadata_uint(
        index, "qwen35moe.context_length", 262144);
    const auto head_dim = metadata_uint(
        index, "qwen35moe.attention.key_length", kHeadDim);
    const auto rope_sections = metadata_int_array(
        index, "qwen35moe.rope.dimension_sections", {11, 11, 10, 0});
    const float rope_freq_base = metadata_float(
        index, "qwen35moe.rope.freq_base", 10000000.0f);
    constexpr float kRopeFreqScale = 1.0f;
    constexpr float kRopeExtFactor = 0.0f;
    constexpr float kRopeAttnFactor = 1.0f;
    constexpr float kRopeBetaFast = 32.0f;
    constexpr float kRopeBetaSlow = 1.0f;

    const auto oracle_q_normed =
        iq36::read_f32_vector_file(join_path(payload_dir, "q_normed.bin"));
    const auto oracle_k_normed =
        iq36::read_f32_vector_file(join_path(payload_dir, "k_normed.bin"));
    const auto oracle_q_rope =
        iq36::read_f32_vector_file(join_path(payload_dir, "q_rope.bin"));
    const auto oracle_k_rope =
        iq36::read_f32_vector_file(join_path(payload_dir, "k_rope.bin"));

    const auto prefix = std::string("blk.") + std::to_string(kLayerIndex) + ".";
    const auto* q_norm_tensor = iq36::find_tensor(index, prefix + "attn_q_norm.weight");
    const auto* k_norm_tensor = iq36::find_tensor(index, prefix + "attn_k_norm.weight");
    require(q_norm_tensor != nullptr && k_norm_tensor != nullptr,
            "full attention norm tensor set is incomplete");
    const bool tensors_shape_ok =
        q_norm_tensor->type == 0 &&
        q_norm_tensor->dims == std::vector<std::uint64_t>{kHeadDim} &&
        k_norm_tensor->type == 0 &&
        k_norm_tensor->dims == std::vector<std::uint64_t>{kHeadDim};

    const auto native = iq36::run_qwen36_full_attention_rope(
        oracle_q_normed,
        oracle_k_normed,
        kSourceTokenPosition,
        head_dim,
        rope_dimension_count,
        rope_sections,
        rope_context_length,
        rope_freq_base,
        kRopeFreqScale,
        kRopeExtFactor,
        kRopeAttnFactor,
        kRopeBetaFast,
        kRopeBetaSlow);

    const std::vector<std::pair<std::string, iq36::VectorCompareStats>> comparisons = {
        {"q_rope", iq36::compare_vectors(native.q_rope, oracle_q_rope, kMismatchThreshold)},
        {"k_rope", iq36::compare_vectors(native.k_rope, oracle_k_rope, kMismatchThreshold)},
    };

    bool comparisons_ok = true;
    for (const auto& item : comparisons) {
      comparisons_ok = comparisons_ok && vector_compare_passed(item.second);
    }

    const std::vector<std::pair<std::string, ValueStats>> native_stats = {
        {"q_normed", stats_from_values(oracle_q_normed)},
        {"k_normed", stats_from_values(oracle_k_normed)},
        {"q_rope", stats_from_values(native.q_rope)},
        {"k_rope", stats_from_values(native.k_rope)},
    };
    const bool stats_ok = std::all_of(
        native_stats.begin(), native_stats.end(), [](const auto& item) {
          return item.second.finite && item.second.nonzero;
        });
    const bool counts_ok =
        oracle_q_normed.size() == kQSize &&
        oracle_k_normed.size() == kKvSize &&
        oracle_q_rope.size() == kQSize &&
        oracle_k_rope.size() == kKvSize &&
        native.q_rope.size() == kQSize &&
        native.k_rope.size() == kKvSize;
    const bool rope_metadata_ok =
        head_dim == kHeadDim &&
        rope_dimension_count == 64 &&
        rope_context_length == 262144 &&
        rope_sections == std::vector<std::int64_t>({11, 11, 10, 0}) &&
        rope_freq_base == 10000000.0f;

    const bool passed =
        load_map.ready && tensors_shape_ok && counts_ok && stats_ok &&
        rope_metadata_ok && comparisons_ok;

    std::cout << std::setprecision(17);
    std::cout << "{";
    std::cout << "\"comparisons\":";
    write_named_comparisons(comparisons);
    std::cout << ",\"layer_index\":" << kLayerIndex;
    std::cout << ",\"load_map_ready\":" << (load_map.ready ? "true" : "false");
    std::cout << ",\"model_path\":\"" << json_escape(model_path) << "\"";
    std::cout << ",\"native_vectors\":";
    write_named_stats(native_stats);
    std::cout << ",\"passed\":" << (passed ? "true" : "false");
    std::cout << ",\"prompt_case_id\":\"short_math_001\"";
    std::cout << ",\"rope_parameters\":{";
    std::cout << "\"context_length\":" << rope_context_length << ",";
    std::cout << "\"head_dim\":" << head_dim << ",";
    std::cout << "\"position_ids\":[15,15,15,0],";
    std::cout << "\"rope_attn_factor\":" << kRopeAttnFactor << ",";
    std::cout << "\"rope_beta_fast\":" << kRopeBetaFast << ",";
    std::cout << "\"rope_beta_slow\":" << kRopeBetaSlow << ",";
    std::cout << "\"rope_dimension_count\":" << rope_dimension_count << ",";
    std::cout << "\"rope_dimension_sections\":";
    write_i64_vector(rope_sections);
    std::cout << ",\"rope_ext_factor\":" << kRopeExtFactor << ",";
    std::cout << "\"rope_freq_base\":" << rope_freq_base << ",";
    std::cout << "\"rope_freq_scale\":" << kRopeFreqScale << ",";
    std::cout << "\"rope_type\":\"imrope\"";
    std::cout << "}";
    std::cout << ",\"schema_version\":\"intel-qwen36-engine-full-attn-rope-compare-v0\"";
    std::cout << ",\"source_token_position\":" << kSourceTokenPosition;
    std::cout << ",\"tensors\":{";
    std::cout << "\"q_norm\":{\"dims\":";
    write_u64_vector(q_norm_tensor->dims);
    std::cout << ",\"name\":\"" << json_escape(q_norm_tensor->name)
              << "\",\"type_name\":\"" << iq36::ggml_type_name(q_norm_tensor->type)
              << "\"},";
    std::cout << "\"k_norm\":{\"dims\":";
    write_u64_vector(k_norm_tensor->dims);
    std::cout << ",\"name\":\"" << json_escape(k_norm_tensor->name)
              << "\",\"type_name\":\"" << iq36::ggml_type_name(k_norm_tensor->type)
              << "\"},";
    std::cout << "\"shape_ok\":" << (tensors_shape_ok ? "true" : "false");
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
    std::cerr << "iq36-full-attn-rope-compare failed: " << exc.what() << "\n";
    return 1;
  }
}
