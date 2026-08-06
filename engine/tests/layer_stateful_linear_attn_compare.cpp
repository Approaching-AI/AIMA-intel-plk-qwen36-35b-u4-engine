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

constexpr int kLayerIndex = 0;
constexpr int kHiddenSize = 2048;
constexpr int kQkvMixedSize = 8192;
constexpr int kConvKernelSize = 4;
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

bool vector_compare_passed(const iq36::VectorCompareStats& stats,
                           double max_abs_threshold,
                           double rmse_threshold) {
  return stats.same_size && stats.finite && stats.mismatch_count == 0 &&
         stats.max_abs_diff <= max_abs_threshold &&
         stats.rmse <= rmse_threshold && stats.cosine >= kMinCosine;
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
            "usage: iq36-layer-stateful-linear-attn-compare "
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
    const auto oracle_attention_norm =
        iq36::read_f32_vector_file(join_path(payload_dir, "attention_norm.bin"));
    const auto oracle_qkv_mixed =
        iq36::read_f32_vector_file(join_path(payload_dir, "linear_attn_qkv_mixed.bin"));
    const auto oracle_conv_output =
        iq36::read_f32_vector_file(join_path(payload_dir, "conv_output_raw.bin"));
    const auto oracle_state_predelta =
        iq36::read_f32_vector_file(join_path(payload_dir, "state_predelta.bin"));
    const auto oracle_final_output =
        iq36::read_f32_vector_file(join_path(payload_dir, "final_output.bin"));
    const auto oracle_linear_attention_out =
        iq36::read_f32_vector_file(join_path(payload_dir, "linear_attention_out.bin"));
    const auto oracle_attention_residual =
        iq36::read_f32_vector_file(join_path(payload_dir, "attention_residual.bin"));
    const auto oracle_topk =
        read_i32_vector_file(join_path(payload_dir, "topk.bin"));
    const auto oracle_weights_norm =
        iq36::read_f32_vector_file(join_path(payload_dir, "weights_norm.bin"));
    const auto oracle_ffn_out =
        iq36::read_f32_vector_file(join_path(payload_dir, "ffn_out.bin"));
    const auto oracle_layer_output =
        iq36::read_f32_vector_file(join_path(payload_dir, "layer_output.bin"));

    const auto* embed_tensor = iq36::find_tensor(index, "token_embd.weight");
    const auto* norm_tensor = iq36::find_tensor(index, "blk.0.attn_norm.weight");
    const auto* qkv_tensor = iq36::find_tensor(index, "blk.0.attn_qkv.weight");
    const auto* conv_tensor = iq36::find_tensor(index, "blk.0.ssm_conv1d.weight");
    const auto* ssm_out_tensor = iq36::find_tensor(index, "blk.0.ssm_out.weight");
    require(embed_tensor != nullptr && norm_tensor != nullptr &&
                qkv_tensor != nullptr && conv_tensor != nullptr &&
                ssm_out_tensor != nullptr,
            "stateful linear attention required tensor missing");
    const bool tensors_shape_ok =
        embed_tensor->dims == std::vector<std::uint64_t>{kHiddenSize, 248320} &&
        norm_tensor->type == 0 &&
        norm_tensor->dims == std::vector<std::uint64_t>{kHiddenSize} &&
        (qkv_tensor->type == 12 || qkv_tensor->type == 14) &&
        qkv_tensor->dims == std::vector<std::uint64_t>{kHiddenSize, kQkvMixedSize} &&
        conv_tensor->type == 0 &&
        conv_tensor->dims ==
            std::vector<std::uint64_t>{kConvKernelSize, kQkvMixedSize} &&
        ssm_out_tensor->type == 12 &&
        ssm_out_tensor->dims ==
            std::vector<std::uint64_t>{kAttentionStateSize, kHiddenSize};

    const auto attention_norm_weight =
        iq36::decode_tensor_row(model_path, index, "blk.0.attn_norm.weight", 0);
    const auto ssm_norm_weight =
        iq36::decode_tensor_row(model_path, index, "blk.0.ssm_norm.weight", 0);

    std::vector<float> conv_state(
        static_cast<std::size_t>((kConvKernelSize - 1) * kQkvMixedSize),
        0.0f);
    std::vector<float> recurrent_state(kRecurrentStateSize, 0.0f);

    std::vector<float> native_embedding;
    std::vector<float> native_attention_norm;
    std::vector<float> native_qkv_mixed;
    std::vector<float> native_conv_output_raw;
    std::vector<float> native_state_predelta;
    std::vector<float> native_final_output;
    std::vector<float> native_linear_attention_out;
    std::vector<float> native_attention_residual;
    iq36::Qwen36MoeFfnLayerResult native_ffn;

    for (std::size_t pos = 0; pos < kPromptTokenIds.size(); ++pos) {
      const auto token_id = static_cast<std::uint64_t>(kPromptTokenIds[pos]);
      const auto embedding =
          iq36::decode_tensor_row(model_path, index, "token_embd.weight", token_id);
      const auto attention_norm =
          iq36::apply_rms_norm(embedding, attention_norm_weight, epsilon);
      const auto preconv = iq36::run_qwen36_linear_attention_preconv_core(
          model_path, index, kLayerIndex, attention_norm);
      const auto conv = iq36::run_qwen36_linear_attention_conv_core(
          model_path, index, kLayerIndex, preconv.qkv_mixed, conv_state);
      const auto state_predelta = recurrent_state;
      const auto attention = iq36::run_qwen36_linear_attention_postconv_core(
          conv.conv_output_raw,
          preconv.gate,
          preconv.beta_sigmoid,
          recurrent_state,
          preconv.z,
          ssm_norm_weight,
          epsilon);

      conv_state = conv.conv_state;
      recurrent_state = attention.recurrent_state;

      if (static_cast<int>(pos) == kSourceTokenPosition) {
        native_embedding = embedding;
        native_attention_norm = attention_norm;
        native_qkv_mixed = preconv.qkv_mixed;
        native_conv_output_raw = conv.conv_output_raw;
        native_state_predelta = state_predelta;
        native_final_output = attention.final_output;
        native_linear_attention_out = iq36::matvec_tensor(
            model_path,
            index,
            "blk.0.ssm_out.weight",
            native_final_output);
        native_attention_residual =
            iq36::add_vectors(embedding, native_linear_attention_out);
        native_ffn = iq36::run_qwen36_moe_ffn_layer(
            model_path,
            index,
            kLayerIndex,
            native_attention_residual,
            epsilon);
      }
    }

    const std::vector<std::pair<std::string, iq36::VectorCompareStats>> comparisons = {
        {"model_input_embed", iq36::compare_vectors(native_embedding, oracle_embedding, kMismatchThreshold)},
        {"attention_norm", iq36::compare_vectors(native_attention_norm, oracle_attention_norm, kMismatchThreshold)},
        {"linear_attn_qkv_mixed", iq36::compare_vectors(native_qkv_mixed, oracle_qkv_mixed, kMismatchThreshold)},
        {"conv_output_raw", iq36::compare_vectors(native_conv_output_raw, oracle_conv_output, kMismatchThreshold)},
        {"state_predelta", iq36::compare_vectors(native_state_predelta, oracle_state_predelta, kMismatchThreshold)},
        {"final_output", iq36::compare_vectors(native_final_output, oracle_final_output, kMismatchThreshold)},
        {"linear_attention_out", iq36::compare_vectors(native_linear_attention_out, oracle_linear_attention_out, kMismatchThreshold)},
        {"attention_residual", iq36::compare_vectors(native_attention_residual, oracle_attention_residual, kResidualMismatchThreshold)},
        {"weights_norm", iq36::compare_vectors(native_ffn.router.normalized_weights, oracle_weights_norm, kWeightsMismatchThreshold)},
        {"ffn_out", iq36::compare_vectors(native_ffn.ffn_out, oracle_ffn_out, kMismatchThreshold)},
        {"layer_output", iq36::compare_vectors(native_ffn.residual, oracle_layer_output, kMismatchThreshold)},
    };
    const auto topk_compare =
        compare_i32_vectors(native_ffn.router.expert_ids, oracle_topk);

    bool comparisons_ok = true;
    for (const auto& item : comparisons) {
      double max_abs = kMaxAbsDiffThreshold;
      double rmse = kRmseThreshold;
      if (item.first == "attention_residual") {
        max_abs = kResidualMaxAbsDiffThreshold;
        rmse = kResidualRmseThreshold;
      } else if (item.first == "weights_norm") {
        max_abs = kWeightsMaxAbsDiffThreshold;
        rmse = kWeightsRmseThreshold;
      }
      comparisons_ok =
          comparisons_ok && vector_compare_passed(item.second, max_abs, rmse);
    }

    const bool counts_ok =
        native_embedding.size() == kHiddenSize &&
        native_attention_norm.size() == kHiddenSize &&
        native_qkv_mixed.size() == kQkvMixedSize &&
        native_conv_output_raw.size() == kQkvMixedSize &&
        native_state_predelta.size() == static_cast<std::size_t>(kRecurrentStateSize) &&
        native_final_output.size() == kAttentionStateSize &&
        native_linear_attention_out.size() == kHiddenSize &&
        native_attention_residual.size() == kHiddenSize &&
        native_ffn.ffn_norm.size() == kHiddenSize &&
        native_ffn.router_logits.size() == 256 &&
        native_ffn.router.expert_ids.size() == kExpertUsedCount &&
        native_ffn.router.normalized_weights.size() == kExpertUsedCount &&
        native_ffn.ffn_out.size() == kHiddenSize &&
        native_ffn.residual.size() == kHiddenSize &&
        conv_state.size() ==
            static_cast<std::size_t>((kConvKernelSize - 1) * kQkvMixedSize) &&
        recurrent_state.size() == static_cast<std::size_t>(kRecurrentStateSize);

    const std::vector<std::pair<std::string, ValueStats>> native_stats = {
        {"model_input_embed", stats_from_values(native_embedding)},
        {"attention_norm", stats_from_values(native_attention_norm)},
        {"linear_attn_qkv_mixed", stats_from_values(native_qkv_mixed)},
        {"conv_output_raw", stats_from_values(native_conv_output_raw)},
        {"state_predelta", stats_from_values(native_state_predelta)},
        {"final_output", stats_from_values(native_final_output)},
        {"linear_attention_out", stats_from_values(native_linear_attention_out)},
        {"attention_residual", stats_from_values(native_attention_residual)},
        {"ffn_norm", stats_from_values(native_ffn.ffn_norm)},
        {"router_logits", stats_from_values(native_ffn.router_logits)},
        {"ffn_out", stats_from_values(native_ffn.ffn_out)},
        {"layer_output", stats_from_values(native_ffn.residual)},
        {"conv_state_after", stats_from_values(conv_state)},
        {"recurrent_state_after", stats_from_values(recurrent_state)},
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
    std::cout << ",\"layer_index\":" << kLayerIndex;
    std::cout << ",\"load_map_ready\":" << (load_map.ready ? "true" : "false");
    std::cout << ",\"model_path\":\"" << json_escape(model_path) << "\"";
    std::cout << ",\"native_topk\":";
    write_i32_vector(native_ffn.router.expert_ids);
    std::cout << ",\"native_vectors\":";
    write_named_stats(native_stats);
    std::cout << ",\"oracle_topk\":";
    write_i32_vector(oracle_topk);
    std::cout << ",\"passed\":" << (passed ? "true" : "false");
    std::cout << ",\"prompt_case_id\":\"short_math_001\"";
    std::cout << ",\"prompt_token_count\":" << kPromptTokenIds.size();
    std::cout << ",\"schema_version\":\"intel-qwen36-engine-layer-stateful-linear-attn-compare-v0\"";
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
    std::cout << "\"ssm_out\":{\"dims\":";
    write_u64_vector(ssm_out_tensor->dims);
    std::cout << ",\"name\":\"" << ssm_out_tensor->name << "\",\"type_name\":\""
              << iq36::ggml_type_name(ssm_out_tensor->type) << "\"},";
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
    std::cerr << "iq36-layer-stateful-linear-attn-compare failed: "
              << exc.what() << "\n";
    return 1;
  }
}
