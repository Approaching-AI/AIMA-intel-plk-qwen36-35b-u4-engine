#include "intel_qwen36/gguf_loader.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kLayerCount = 2;
constexpr int kHiddenSize = 2048;
constexpr int kQkvMixedSize = 8192;
constexpr int kConvKernelSize = 4;
constexpr int kConvStateSize = (kConvKernelSize - 1) * kQkvMixedSize;
constexpr int kAttentionStateSize = 4096;
constexpr int kHeadDim = 128;
constexpr int kValueHeads = 32;
constexpr int kRecurrentStateSize = kHeadDim * kHeadDim * kValueHeads;
constexpr int kExpertUsedCount = 8;
constexpr int kSourceTokenPosition = 15;
constexpr double kMismatchThreshold = 5e-4;
constexpr double kMaxAbsDiffThreshold = 5e-4;
constexpr double kRmseThreshold = 5e-5;
constexpr double kResidualMismatchThreshold = 5e-6;
constexpr double kResidualMaxAbsDiffThreshold = 5e-6;
constexpr double kResidualRmseThreshold = 1e-6;
constexpr double kWeightsMismatchThreshold = 2e-5;
constexpr double kWeightsMaxAbsDiffThreshold = 2e-5;
constexpr double kWeightsRmseThreshold = 1e-6;
constexpr double kMinCosine = 0.999;

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

struct IntCompareStats {
  std::uint64_t lhs_value_count = 0;
  std::uint64_t rhs_value_count = 0;
  std::uint64_t compared_value_count = 0;
  std::uint64_t mismatch_count = 0;
  bool same_size = false;
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

std::int32_t read_le_i32(const std::vector<std::uint8_t>& bytes,
                         std::size_t offset) {
  const std::uint32_t value =
      static_cast<std::uint32_t>(bytes[offset]) |
      (static_cast<std::uint32_t>(bytes[offset + 1]) << 8) |
      (static_cast<std::uint32_t>(bytes[offset + 2]) << 16) |
      (static_cast<std::uint32_t>(bytes[offset + 3]) << 24);
  return static_cast<std::int32_t>(value);
}

std::vector<std::int32_t> read_i32_vector_file(const std::string& path) {
  const auto file_size = std::filesystem::file_size(path);
  if (file_size % sizeof(std::int32_t) != 0) {
    throw std::invalid_argument("i32 vector file size is not divisible by 4");
  }
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::invalid_argument("i32 vector file could not be opened");
  }
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(file_size));
  input.read(reinterpret_cast<char*>(bytes.data()),
             static_cast<std::streamsize>(bytes.size()));
  if (!input) {
    throw std::runtime_error("i32 vector file read failed");
  }
  std::vector<std::int32_t> values;
  values.reserve(bytes.size() / sizeof(std::int32_t));
  for (std::size_t i = 0; i < bytes.size(); i += sizeof(std::int32_t)) {
    values.push_back(read_le_i32(bytes, i));
  }
  return values;
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

IntCompareStats compare_i32_vectors(const std::vector<std::int32_t>& lhs,
                                    const std::vector<std::int32_t>& rhs) {
  IntCompareStats stats;
  stats.lhs_value_count = lhs.size();
  stats.rhs_value_count = rhs.size();
  stats.compared_value_count = std::min(lhs.size(), rhs.size());
  stats.same_size = lhs.size() == rhs.size();
  if (!stats.same_size) {
    stats.mismatch_count +=
        static_cast<std::uint64_t>(
            lhs.size() > rhs.size() ? lhs.size() - rhs.size()
                                    : rhs.size() - lhs.size());
  }
  for (std::size_t i = 0; i < stats.compared_value_count; ++i) {
    if (lhs[i] != rhs[i]) {
      ++stats.mismatch_count;
    }
  }
  return stats;
}

bool vector_compare_passed(const std::string& name,
                           const iq36::VectorCompareStats& stats) {
  double max_abs = kMaxAbsDiffThreshold;
  double rmse = kRmseThreshold;
  if (name == "layer1_attention_residual") {
    max_abs = kResidualMaxAbsDiffThreshold;
    rmse = kResidualRmseThreshold;
  } else if (name == "layer1_weights_norm") {
    max_abs = kWeightsMaxAbsDiffThreshold;
    rmse = kWeightsRmseThreshold;
  }
  return stats.same_size && stats.finite && stats.mismatch_count == 0 &&
         stats.max_abs_diff <= max_abs && stats.rmse <= rmse &&
         stats.cosine >= kMinCosine;
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

void write_int_compare_stats(const IntCompareStats& stats) {
  std::cout << "{";
  std::cout << "\"compared_value_count\":" << stats.compared_value_count << ",";
  std::cout << "\"lhs_value_count\":" << stats.lhs_value_count << ",";
  std::cout << "\"mismatch_count\":" << stats.mismatch_count << ",";
  std::cout << "\"rhs_value_count\":" << stats.rhs_value_count << ",";
  std::cout << "\"same_size\":" << (stats.same_size ? "true" : "false");
  std::cout << "}";
}

void write_i32_vector(const std::vector<std::int32_t>& values) {
  std::cout << "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0) {
      std::cout << ",";
    }
    std::cout << values[i];
  }
  std::cout << "]";
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
            "usage: iq36-two-linear-layers-stateful-compare "
            "<model.gguf> <oracle-payload-dir>");
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
    const auto oracle_layer0_output =
        iq36::read_f32_vector_file(join_path(payload_dir, "layer0_output.bin"));
    const auto oracle_layer1_attention_norm =
        iq36::read_f32_vector_file(join_path(payload_dir, "layer1_attention_norm.bin"));
    const auto oracle_layer1_qkv_mixed =
        iq36::read_f32_vector_file(join_path(payload_dir, "layer1_linear_attn_qkv_mixed.bin"));
    const auto oracle_layer1_conv_output =
        iq36::read_f32_vector_file(join_path(payload_dir, "layer1_conv_output_raw.bin"));
    const auto oracle_layer1_state_predelta =
        iq36::read_f32_vector_file(join_path(payload_dir, "layer1_state_predelta.bin"));
    const auto oracle_layer1_final_output =
        iq36::read_f32_vector_file(join_path(payload_dir, "layer1_final_output.bin"));
    const auto oracle_layer1_linear_attention_out =
        iq36::read_f32_vector_file(join_path(payload_dir, "layer1_linear_attention_out.bin"));
    const auto oracle_layer1_attention_residual =
        iq36::read_f32_vector_file(join_path(payload_dir, "layer1_attention_residual.bin"));
    const auto oracle_layer1_topk =
        read_i32_vector_file(join_path(payload_dir, "layer1_topk.bin"));
    const auto oracle_layer1_weights_norm =
        iq36::read_f32_vector_file(join_path(payload_dir, "layer1_weights_norm.bin"));
    const auto oracle_layer1_ffn_out =
        iq36::read_f32_vector_file(join_path(payload_dir, "layer1_ffn_out.bin"));
    const auto oracle_layer1_output =
        iq36::read_f32_vector_file(join_path(payload_dir, "layer1_output.bin"));

    const auto* embed_tensor = iq36::find_tensor(index, "token_embd.weight");
    require(embed_tensor != nullptr, "embedding tensor missing");
    bool tensors_shape_ok =
        embed_tensor->dims == std::vector<std::uint64_t>{kHiddenSize, 248320};
    for (int layer = 0; layer < kLayerCount; ++layer) {
      const auto prefix = "blk." + std::to_string(layer) + ".";
      const auto* norm_tensor = iq36::find_tensor(index, prefix + "attn_norm.weight");
      const auto* qkv_tensor = iq36::find_tensor(index, prefix + "attn_qkv.weight");
      const auto* conv_tensor = iq36::find_tensor(index, prefix + "ssm_conv1d.weight");
      const auto* ssm_out_tensor = iq36::find_tensor(index, prefix + "ssm_out.weight");
      require(norm_tensor != nullptr && qkv_tensor != nullptr &&
                  conv_tensor != nullptr && ssm_out_tensor != nullptr,
              "two linear layer tensor set is incomplete");
      tensors_shape_ok =
          tensors_shape_ok &&
          norm_tensor->type == 0 &&
          norm_tensor->dims == std::vector<std::uint64_t>{kHiddenSize} &&
          (qkv_tensor->type == 12 || qkv_tensor->type == 14) &&
          qkv_tensor->dims ==
              std::vector<std::uint64_t>{kHiddenSize, kQkvMixedSize} &&
          conv_tensor->type == 0 &&
          conv_tensor->dims ==
              std::vector<std::uint64_t>{kConvKernelSize, kQkvMixedSize} &&
          ssm_out_tensor->type == 12 &&
          ssm_out_tensor->dims ==
              std::vector<std::uint64_t>{kAttentionStateSize, kHiddenSize};
    }

    std::vector<std::vector<float>> conv_states(
        kLayerCount, std::vector<float>(kConvStateSize, 0.0f));
    std::vector<std::vector<float>> recurrent_states(
        kLayerCount, std::vector<float>(kRecurrentStateSize, 0.0f));

    std::vector<float> native_embedding;
    iq36::Qwen36StatefulLinearAttentionLayerResult native_layer0;
    iq36::Qwen36StatefulLinearAttentionLayerResult native_layer1;

    for (std::size_t pos = 0; pos < kPromptTokenIds.size(); ++pos) {
      const auto token_id = static_cast<std::uint64_t>(kPromptTokenIds[pos]);
      auto residual =
          iq36::decode_tensor_row(model_path, index, "token_embd.weight", token_id);
      auto layer0 = iq36::run_qwen36_stateful_linear_attention_layer(
          model_path,
          index,
          0,
          residual,
          conv_states[0],
          recurrent_states[0],
          epsilon);
      conv_states[0] = layer0.conv.conv_state;
      recurrent_states[0] = layer0.attention.recurrent_state;
      residual = layer0.residual;

      auto layer1 = iq36::run_qwen36_stateful_linear_attention_layer(
          model_path,
          index,
          1,
          residual,
          conv_states[1],
          recurrent_states[1],
          epsilon);
      conv_states[1] = layer1.conv.conv_state;
      recurrent_states[1] = layer1.attention.recurrent_state;

      if (static_cast<int>(pos) == kSourceTokenPosition) {
        native_embedding =
            iq36::decode_tensor_row(model_path, index, "token_embd.weight", token_id);
        native_layer0 = std::move(layer0);
        native_layer1 = std::move(layer1);
      }
    }

    const std::vector<std::pair<std::string, iq36::VectorCompareStats>> comparisons = {
        {"model_input_embed", iq36::compare_vectors(native_embedding, oracle_embedding, kMismatchThreshold)},
        {"layer0_output", iq36::compare_vectors(native_layer0.residual, oracle_layer0_output, kMismatchThreshold)},
        {"layer1_attention_norm", iq36::compare_vectors(native_layer1.attention_norm, oracle_layer1_attention_norm, kMismatchThreshold)},
        {"layer1_linear_attn_qkv_mixed", iq36::compare_vectors(native_layer1.preconv.qkv_mixed, oracle_layer1_qkv_mixed, kMismatchThreshold)},
        {"layer1_conv_output_raw", iq36::compare_vectors(native_layer1.conv.conv_output_raw, oracle_layer1_conv_output, kMismatchThreshold)},
        {"layer1_state_predelta", iq36::compare_vectors(native_layer1.state_predelta, oracle_layer1_state_predelta, kMismatchThreshold)},
        {"layer1_final_output", iq36::compare_vectors(native_layer1.attention.final_output, oracle_layer1_final_output, kMismatchThreshold)},
        {"layer1_linear_attention_out", iq36::compare_vectors(native_layer1.linear_attention_out, oracle_layer1_linear_attention_out, kMismatchThreshold)},
        {"layer1_attention_residual", iq36::compare_vectors(native_layer1.attention_residual, oracle_layer1_attention_residual, kResidualMismatchThreshold)},
        {"layer1_weights_norm", iq36::compare_vectors(native_layer1.ffn.router.normalized_weights, oracle_layer1_weights_norm, kWeightsMismatchThreshold)},
        {"layer1_ffn_out", iq36::compare_vectors(native_layer1.ffn.ffn_out, oracle_layer1_ffn_out, kMismatchThreshold)},
        {"layer1_output", iq36::compare_vectors(native_layer1.residual, oracle_layer1_output, kMismatchThreshold)},
    };
    const auto topk_compare =
        compare_i32_vectors(native_layer1.ffn.router.expert_ids, oracle_layer1_topk);

    bool comparisons_ok = true;
    for (const auto& item : comparisons) {
      comparisons_ok =
          comparisons_ok && vector_compare_passed(item.first, item.second);
    }

    const bool counts_ok =
        native_embedding.size() == kHiddenSize &&
        native_layer0.residual.size() == kHiddenSize &&
        native_layer1.attention_norm.size() == kHiddenSize &&
        native_layer1.preconv.qkv_mixed.size() == kQkvMixedSize &&
        native_layer1.conv.conv_output_raw.size() == kQkvMixedSize &&
        native_layer1.state_predelta.size() == static_cast<std::size_t>(kRecurrentStateSize) &&
        native_layer1.attention.final_output.size() == kAttentionStateSize &&
        native_layer1.linear_attention_out.size() == kHiddenSize &&
        native_layer1.attention_residual.size() == kHiddenSize &&
        native_layer1.ffn.ffn_norm.size() == kHiddenSize &&
        native_layer1.ffn.router_logits.size() == 256 &&
        native_layer1.ffn.router.expert_ids.size() == kExpertUsedCount &&
        native_layer1.ffn.router.normalized_weights.size() == kExpertUsedCount &&
        native_layer1.ffn.ffn_out.size() == kHiddenSize &&
        native_layer1.residual.size() == kHiddenSize &&
        conv_states[0].size() == kConvStateSize &&
        conv_states[1].size() == kConvStateSize &&
        recurrent_states[0].size() == static_cast<std::size_t>(kRecurrentStateSize) &&
        recurrent_states[1].size() == static_cast<std::size_t>(kRecurrentStateSize);

    const std::vector<std::pair<std::string, ValueStats>> native_stats = {
        {"model_input_embed", stats_from_values(native_embedding)},
        {"layer0_output", stats_from_values(native_layer0.residual)},
        {"layer1_attention_norm", stats_from_values(native_layer1.attention_norm)},
        {"layer1_linear_attn_qkv_mixed", stats_from_values(native_layer1.preconv.qkv_mixed)},
        {"layer1_conv_output_raw", stats_from_values(native_layer1.conv.conv_output_raw)},
        {"layer1_state_predelta", stats_from_values(native_layer1.state_predelta)},
        {"layer1_final_output", stats_from_values(native_layer1.attention.final_output)},
        {"layer1_linear_attention_out", stats_from_values(native_layer1.linear_attention_out)},
        {"layer1_attention_residual", stats_from_values(native_layer1.attention_residual)},
        {"layer1_ffn_norm", stats_from_values(native_layer1.ffn.ffn_norm)},
        {"layer1_router_logits", stats_from_values(native_layer1.ffn.router_logits)},
        {"layer1_ffn_out", stats_from_values(native_layer1.ffn.ffn_out)},
        {"layer1_output", stats_from_values(native_layer1.residual)},
        {"layer0_conv_state_after", stats_from_values(conv_states[0])},
        {"layer1_conv_state_after", stats_from_values(conv_states[1])},
        {"layer0_recurrent_state_after", stats_from_values(recurrent_states[0])},
        {"layer1_recurrent_state_after", stats_from_values(recurrent_states[1])},
    };
    const bool stats_ok = std::all_of(
        native_stats.begin(), native_stats.end(), [](const auto& item) {
          return item.second.finite && item.second.nonzero;
        });

    const bool passed =
        load_map.ready && tensors_shape_ok && counts_ok && stats_ok &&
        comparisons_ok && topk_compare.same_size &&
        topk_compare.mismatch_count == 0;

    std::cout << std::setprecision(17);
    std::cout << "{";
    std::cout << "\"comparisons\":";
    write_named_comparisons(comparisons);
    std::cout << ",\"conv_kernel_size\":" << kConvKernelSize;
    std::cout << ",\"epsilon\":" << epsilon;
    std::cout << ",\"layer_count\":" << kLayerCount;
    std::cout << ",\"load_map_ready\":" << (load_map.ready ? "true" : "false");
    std::cout << ",\"model_path\":\"" << json_escape(model_path) << "\"";
    std::cout << ",\"native_layer1_topk\":";
    write_i32_vector(native_layer1.ffn.router.expert_ids);
    std::cout << ",\"native_vectors\":";
    write_named_stats(native_stats);
    std::cout << ",\"oracle_layer1_topk\":";
    write_i32_vector(oracle_layer1_topk);
    std::cout << ",\"passed\":" << (passed ? "true" : "false");
    std::cout << ",\"prompt_case_id\":\"short_math_001\"";
    std::cout << ",\"prompt_token_count\":" << kPromptTokenIds.size();
    std::cout << ",\"schema_version\":\"intel-qwen36-engine-two-linear-layers-stateful-compare-v0\"";
    std::cout << ",\"source_token_position\":" << kSourceTokenPosition;
    std::cout << ",\"tensors\":{";
    std::cout << "\"embedding\":{\"dims\":";
    write_u64_vector(embed_tensor->dims);
    std::cout << ",\"name\":\"" << embed_tensor->name << "\",\"type_name\":\""
              << iq36::ggml_type_name(embed_tensor->type) << "\"},";
    std::cout << "\"shape_ok\":" << (tensors_shape_ok ? "true" : "false");
    std::cout << "}";
    std::cout << ",\"thresholds\":{";
    std::cout << "\"max_abs_diff\":" << kMaxAbsDiffThreshold << ",";
    std::cout << "\"min_cosine\":" << kMinCosine << ",";
    std::cout << "\"mismatch_abs_diff\":" << kMismatchThreshold << ",";
    std::cout << "\"residual_max_abs_diff\":" << kResidualMaxAbsDiffThreshold << ",";
    std::cout << "\"residual_rmse\":" << kResidualRmseThreshold << ",";
    std::cout << "\"rmse\":" << kRmseThreshold << ",";
    std::cout << "\"weights_max_abs_diff\":" << kWeightsMaxAbsDiffThreshold << ",";
    std::cout << "\"weights_rmse\":" << kWeightsRmseThreshold;
    std::cout << "}";
    std::cout << ",\"topk_comparison\":";
    write_int_compare_stats(topk_compare);
    std::cout << "}\n";
    return passed ? 0 : 2;
  } catch (const std::exception& exc) {
    std::cerr << "iq36-two-linear-layers-stateful-compare failed: "
              << exc.what() << "\n";
    return 1;
  }
}
