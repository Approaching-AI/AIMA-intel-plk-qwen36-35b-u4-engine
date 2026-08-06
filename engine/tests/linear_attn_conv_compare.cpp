#include "intel_qwen36/gguf_loader.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kLayerIndex = 0;
constexpr int kHiddenSize = 2048;
constexpr int kQkvMixedSize = 8192;
constexpr int kConvKernelSize = 4;
constexpr int kSourceTokenPosition = 15;
constexpr double kMismatchThreshold = 5e-4;
constexpr double kMaxAbsDiffThreshold = 5e-4;
constexpr double kRmseThreshold = 5e-5;
constexpr double kMinCosine = 0.99999;

constexpr std::array<std::int64_t, 16> kPromptTokenIds = {
    15666, 303, 799, 2716, 11316, 25, 1092, 369,
    220,   16,  22,  5346, 220,   17, 20,   30,
};

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

bool compare_passed(const iq36::VectorCompareStats& stats) {
  return stats.same_size && stats.finite && stats.mismatch_count == 0 &&
         stats.max_abs_diff <= kMaxAbsDiffThreshold &&
         stats.rmse <= kRmseThreshold && stats.cosine >= kMinCosine;
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

}  // namespace

int main(int argc, char** argv) {
  try {
    require(argc == 3,
            "usage: iq36-linear-attn-conv-compare <model.gguf> <oracle-payload-dir>");
    const std::string model_path = argv[1];
    const std::string payload_dir = argv[2];

    const auto index = iq36::parse_gguf_model_index(model_path);
    const auto load_map = iq36::validate_qwen36_load_map(index);
    const float epsilon = metadata_float(
        index,
        "qwen35moe.attention.layer_norm_rms_epsilon",
        1e-6f);

    const auto oracle_embedding =
        iq36::read_f32_vector_file(join_path(payload_dir, "model_input_embed.bin"));
    const auto oracle_attention_norm =
        iq36::read_f32_vector_file(join_path(payload_dir, "attn_norm.bin"));
    const auto oracle_qkv_mixed =
        iq36::read_f32_vector_file(join_path(payload_dir, "linear_attn_qkv_mixed.bin"));
    const auto oracle_conv_output =
        iq36::read_f32_vector_file(join_path(payload_dir, "conv_output_raw.bin"));

    const auto* embed_tensor = iq36::find_tensor(index, "token_embd.weight");
    const auto* norm_tensor = iq36::find_tensor(index, "blk.0.attn_norm.weight");
    const auto* qkv_tensor = iq36::find_tensor(index, "blk.0.attn_qkv.weight");
    const auto* conv_tensor = iq36::find_tensor(index, "blk.0.ssm_conv1d.weight");
    require(embed_tensor != nullptr && norm_tensor != nullptr &&
                qkv_tensor != nullptr && conv_tensor != nullptr,
            "linear attention conv required tensor missing");
    const bool tensors_shape_ok =
        embed_tensor->dims == std::vector<std::uint64_t>{kHiddenSize, 248320} &&
        norm_tensor->type == 0 &&
        norm_tensor->dims == std::vector<std::uint64_t>{kHiddenSize} &&
        (qkv_tensor->type == 12 || qkv_tensor->type == 14) &&
        qkv_tensor->dims == std::vector<std::uint64_t>{kHiddenSize, kQkvMixedSize} &&
        conv_tensor->type == 0 &&
        conv_tensor->dims ==
            std::vector<std::uint64_t>{kConvKernelSize, kQkvMixedSize};

    const auto norm_weight =
        iq36::decode_tensor_row(model_path, index, "blk.0.attn_norm.weight", 0);
    std::vector<float> conv_state(
        static_cast<std::size_t>((kConvKernelSize - 1) * kQkvMixedSize),
        0.0f);

    std::vector<float> native_embedding;
    std::vector<float> native_attention_norm;
    std::vector<float> native_qkv_mixed;
    iq36::Qwen36LinearAttentionConvResult native_conv;
    for (std::size_t pos = 0; pos < kPromptTokenIds.size(); ++pos) {
      const auto token_id = static_cast<std::uint64_t>(kPromptTokenIds[pos]);
      const auto embedding =
          iq36::decode_tensor_row(model_path, index, "token_embd.weight", token_id);
      const auto attention_norm =
          iq36::apply_rms_norm(embedding, norm_weight, epsilon);
      const auto preconv = iq36::run_qwen36_linear_attention_preconv_core(
          model_path, index, kLayerIndex, attention_norm);
      native_conv = iq36::run_qwen36_linear_attention_conv_core(
          model_path, index, kLayerIndex, preconv.qkv_mixed, conv_state);
      conv_state = native_conv.conv_state;

      if (static_cast<int>(pos) == kSourceTokenPosition) {
        native_embedding = embedding;
        native_attention_norm = attention_norm;
        native_qkv_mixed = preconv.qkv_mixed;
      }
    }

    const std::vector<std::pair<std::string, iq36::VectorCompareStats>> comparisons = {
        {"model_input_embed", iq36::compare_vectors(native_embedding, oracle_embedding, kMismatchThreshold)},
        {"attention_norm", iq36::compare_vectors(native_attention_norm, oracle_attention_norm, kMismatchThreshold)},
        {"linear_attn_qkv_mixed", iq36::compare_vectors(native_qkv_mixed, oracle_qkv_mixed, kMismatchThreshold)},
        {"conv_output_raw", iq36::compare_vectors(native_conv.conv_output_raw, oracle_conv_output, kMismatchThreshold)},
    };

    bool comparisons_ok = true;
    for (const auto& item : comparisons) {
      comparisons_ok = comparisons_ok && compare_passed(item.second);
    }
    const bool counts_ok =
        native_embedding.size() == kHiddenSize &&
        native_attention_norm.size() == kHiddenSize &&
        native_qkv_mixed.size() == kQkvMixedSize &&
        native_conv.conv_output_raw.size() == kQkvMixedSize &&
        native_conv.conv_state.size() ==
            static_cast<std::size_t>((kConvKernelSize - 1) * kQkvMixedSize);

    const std::vector<std::pair<std::string, ValueStats>> native_stats = {
        {"model_input_embed", stats_from_values(native_embedding)},
        {"attention_norm", stats_from_values(native_attention_norm)},
        {"linear_attn_qkv_mixed", stats_from_values(native_qkv_mixed)},
        {"conv_output_raw", stats_from_values(native_conv.conv_output_raw)},
        {"conv_state_after", stats_from_values(native_conv.conv_state)},
    };
    const bool stats_ok = std::all_of(
        native_stats.begin(), native_stats.end(), [](const auto& item) {
          return item.second.finite && item.second.nonzero;
        });
    const bool passed =
        load_map.ready && tensors_shape_ok && counts_ok && stats_ok && comparisons_ok;

    std::cout << std::setprecision(17);
    std::cout << "{";
    std::cout << "\"comparisons\":";
    write_named_comparisons(comparisons);
    std::cout << ",\"conv_kernel_size\":" << kConvKernelSize;
    std::cout << ",\"epsilon\":" << epsilon;
    std::cout << ",\"layer_index\":" << kLayerIndex;
    std::cout << ",\"load_map_ready\":" << (load_map.ready ? "true" : "false");
    std::cout << ",\"model_path\":\"" << json_escape(model_path) << "\"";
    std::cout << ",\"native_vectors\":";
    write_named_stats(native_stats);
    std::cout << ",\"passed\":" << (passed ? "true" : "false");
    std::cout << ",\"prompt_case_id\":\"short_math_001\"";
    std::cout << ",\"prompt_token_count\":" << kPromptTokenIds.size();
    std::cout << ",\"schema_version\":\"intel-qwen36-engine-linear-attn-conv-compare-v0\"";
    std::cout << ",\"source_token_position\":" << kSourceTokenPosition;
    std::cout << ",\"tensors\":{";
    std::cout << "\"embedding\":{\"dims\":";
    write_u64_vector(embed_tensor->dims);
    std::cout << ",\"name\":\"" << embed_tensor->name << "\",\"type_name\":\""
              << iq36::ggml_type_name(embed_tensor->type) << "\"},";
    std::cout << "\"qkv\":{\"dims\":";
    write_u64_vector(qkv_tensor->dims);
    std::cout << ",\"name\":\"" << qkv_tensor->name << "\",\"type_name\":\""
              << iq36::ggml_type_name(qkv_tensor->type) << "\"},";
    std::cout << "\"conv\":{\"dims\":";
    write_u64_vector(conv_tensor->dims);
    std::cout << ",\"name\":\"" << conv_tensor->name << "\",\"type_name\":\""
              << iq36::ggml_type_name(conv_tensor->type) << "\"},";
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
    std::cerr << "iq36-linear-attn-conv-compare failed: " << exc.what() << "\n";
    return 1;
  }
}
