#include "intel_qwen36/gguf_loader.hpp"

#include <algorithm>
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

constexpr double kResidualMismatchThreshold = 5e-6;
constexpr double kResidualMaxAbsDiffThreshold = 5e-6;
constexpr double kResidualRmseThreshold = 1e-6;
constexpr double kVectorMismatchThreshold = 5e-4;
constexpr double kVectorMaxAbsDiffThreshold = 5e-4;
constexpr double kVectorRmseThreshold = 5e-5;
constexpr double kWeightsMismatchThreshold = 2e-5;
constexpr double kWeightsMaxAbsDiffThreshold = 2e-5;
constexpr double kWeightsRmseThreshold = 1e-6;
constexpr double kMinCosine = 0.999;
constexpr float kRmsNormEpsilon = 9.999999974752427e-07f;
constexpr int kHiddenSize = 2048;
constexpr int kConvSize = 8192;
constexpr int kAttentionStateSize = 4096;
constexpr int kRecurrentStateSize = 524288;
constexpr int kExpertUsedCount = 8;

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
         stats.rmse <= rmse_threshold &&
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

}  // namespace

int main(int argc, char** argv) {
  try {
    require(argc == 13 || argc == 14,
            "usage: iq36-layer-postconv-compare <model.gguf> "
            "<residual-input-f32> <conv-output-raw-f32> <state-predelta-f32> "
            "<oracle-attn-norm-f32> <oracle-final-output-f32> "
            "<oracle-linear-attn-out-f32> <oracle-attention-residual-f32> "
            "<oracle-topk-i32> <oracle-weights-norm-f32> "
            "<oracle-ffn-out-f32> <oracle-layer-output-f32> [layer-index]");
    const std::string model_path = argv[1];
    const std::string residual_input_path = argv[2];
    const std::string conv_output_raw_path = argv[3];
    const std::string state_predelta_path = argv[4];
    const std::string oracle_attn_norm_path = argv[5];
    const std::string oracle_final_output_path = argv[6];
    const std::string oracle_linear_attn_out_path = argv[7];
    const std::string oracle_attention_residual_path = argv[8];
    const std::string oracle_topk_path = argv[9];
    const std::string oracle_weights_norm_path = argv[10];
    const std::string oracle_ffn_out_path = argv[11];
    const std::string oracle_layer_output_path = argv[12];
    const int layer_index = argc == 14 ? std::stoi(argv[13]) : 0;

    const auto index = iq36::parse_gguf_model_index(model_path);
    const auto load_map = iq36::validate_qwen36_load_map(index);
    const auto residual_input = iq36::read_f32_vector_file(residual_input_path);
    const auto conv_output_raw = iq36::read_f32_vector_file(conv_output_raw_path);
    const auto state_predelta = iq36::read_f32_vector_file(state_predelta_path);
    const auto oracle_attn_norm =
        iq36::read_f32_vector_file(oracle_attn_norm_path);
    const auto oracle_final_output =
        iq36::read_f32_vector_file(oracle_final_output_path);
    const auto oracle_linear_attn_out =
        iq36::read_f32_vector_file(oracle_linear_attn_out_path);
    const auto oracle_attention_residual =
        iq36::read_f32_vector_file(oracle_attention_residual_path);
    const auto oracle_topk = read_i32_vector_file(oracle_topk_path);
    const auto oracle_weights_norm =
        iq36::read_f32_vector_file(oracle_weights_norm_path);
    const auto oracle_ffn_out =
        iq36::read_f32_vector_file(oracle_ffn_out_path);
    const auto oracle_layer_output =
        iq36::read_f32_vector_file(oracle_layer_output_path);

    const auto result = iq36::run_qwen36_layer_with_external_conv_output(
        model_path,
        index,
        layer_index,
        residual_input,
        conv_output_raw,
        state_predelta,
        kRmsNormEpsilon);

    const auto attn_norm_compare = iq36::compare_vectors(
        result.attention_norm, oracle_attn_norm, kVectorMismatchThreshold);
    const auto final_output_compare = iq36::compare_vectors(
        result.attention.final_output,
        oracle_final_output,
        kVectorMismatchThreshold);
    const auto linear_attention_out_compare = iq36::compare_vectors(
        result.linear_attention_out,
        oracle_linear_attn_out,
        kVectorMismatchThreshold);
    const auto attention_residual_compare = iq36::compare_vectors(
        result.attention_residual,
        oracle_attention_residual,
        kResidualMismatchThreshold);
    const auto topk_compare =
        compare_i32_vectors(result.ffn.router.expert_ids, oracle_topk);
    const auto weights_norm_compare = iq36::compare_vectors(
        result.ffn.router.normalized_weights,
        oracle_weights_norm,
        kWeightsMismatchThreshold);
    const auto ffn_out_compare =
        iq36::compare_vectors(result.ffn.ffn_out, oracle_ffn_out, kVectorMismatchThreshold);
    const auto layer_output_compare =
        iq36::compare_vectors(result.residual, oracle_layer_output, kVectorMismatchThreshold);

    const auto residual_input_stats = stats_from_values(residual_input);
    const auto conv_output_raw_stats = stats_from_values(conv_output_raw);
    const auto state_predelta_stats = stats_from_values(state_predelta);
    const auto attn_norm_stats = stats_from_values(result.attention_norm);
    const auto final_output_stats = stats_from_values(result.attention.final_output);
    const auto linear_attention_out_stats =
        stats_from_values(result.linear_attention_out);
    const auto attention_residual_stats =
        stats_from_values(result.attention_residual);
    const auto ffn_norm_stats = stats_from_values(result.ffn.ffn_norm);
    const auto router_logits_stats = stats_from_values(result.ffn.router_logits);
    const auto ffn_out_stats = stats_from_values(result.ffn.ffn_out);
    const auto layer_output_stats = stats_from_values(result.residual);
    const auto oracle_layer_output_stats = stats_from_values(oracle_layer_output);

    const bool counts_ok =
        residual_input_stats.count == kHiddenSize &&
        conv_output_raw_stats.count == kConvSize &&
        state_predelta_stats.count == kRecurrentStateSize &&
        attn_norm_stats.count == kHiddenSize &&
        result.attention.final_output.size() == kAttentionStateSize &&
        linear_attention_out_stats.count == kHiddenSize &&
        attention_residual_stats.count == kHiddenSize &&
        ffn_norm_stats.count == kHiddenSize &&
        router_logits_stats.count == 256 &&
        result.ffn.router.expert_ids.size() == kExpertUsedCount &&
        result.ffn.router.normalized_weights.size() == kExpertUsedCount &&
        ffn_out_stats.count == kHiddenSize &&
        layer_output_stats.count == kHiddenSize &&
        oracle_layer_output_stats.count == kHiddenSize;

    const bool stats_ok =
        residual_input_stats.finite &&
        conv_output_raw_stats.finite &&
        state_predelta_stats.finite &&
        attn_norm_stats.finite &&
        final_output_stats.finite &&
        linear_attention_out_stats.finite &&
        attention_residual_stats.finite &&
        ffn_norm_stats.finite &&
        router_logits_stats.finite &&
        ffn_out_stats.finite &&
        layer_output_stats.finite &&
        oracle_layer_output_stats.finite &&
        residual_input_stats.nonzero &&
        conv_output_raw_stats.nonzero &&
        state_predelta_stats.nonzero &&
        attn_norm_stats.nonzero &&
        final_output_stats.nonzero &&
        linear_attention_out_stats.nonzero &&
        attention_residual_stats.nonzero &&
        ffn_norm_stats.nonzero &&
        router_logits_stats.nonzero &&
        ffn_out_stats.nonzero &&
        layer_output_stats.nonzero &&
        oracle_layer_output_stats.nonzero;

    const bool passed =
        load_map.ready &&
        counts_ok &&
        stats_ok &&
        vector_compare_passed(
            attn_norm_compare, kVectorMaxAbsDiffThreshold,
            kVectorRmseThreshold) &&
        vector_compare_passed(
            final_output_compare, kVectorMaxAbsDiffThreshold,
            kVectorRmseThreshold) &&
        vector_compare_passed(
            linear_attention_out_compare, kVectorMaxAbsDiffThreshold,
            kVectorRmseThreshold) &&
        vector_compare_passed(
            attention_residual_compare, kResidualMaxAbsDiffThreshold,
            kResidualRmseThreshold) &&
        topk_compare.same_size &&
        topk_compare.mismatch_count == 0 &&
        vector_compare_passed(
            weights_norm_compare, kWeightsMaxAbsDiffThreshold,
            kWeightsRmseThreshold) &&
        vector_compare_passed(
            ffn_out_compare, kVectorMaxAbsDiffThreshold,
            kVectorRmseThreshold) &&
        vector_compare_passed(
            layer_output_compare, kVectorMaxAbsDiffThreshold,
            kVectorRmseThreshold);

    std::cout << std::setprecision(17);
    std::cout << "{";
    std::cout << "\"attention_norm_comparison\":";
    write_compare_stats(attn_norm_compare);
    std::cout << ",";
    std::cout << "\"attention_norm_vector\":";
    write_value_stats(attn_norm_stats);
    std::cout << ",";
    std::cout << "\"attention_residual_comparison\":";
    write_compare_stats(attention_residual_compare);
    std::cout << ",";
    std::cout << "\"attention_residual_vector\":";
    write_value_stats(attention_residual_stats);
    std::cout << ",";
    std::cout << "\"conv_output_raw_payload_path\":\""
              << json_escape(conv_output_raw_path) << "\",";
    std::cout << "\"conv_output_raw_vector\":";
    write_value_stats(conv_output_raw_stats);
    std::cout << ",";
    std::cout << "\"ffn_norm_vector\":";
    write_value_stats(ffn_norm_stats);
    std::cout << ",";
    std::cout << "\"ffn_out_comparison\":";
    write_compare_stats(ffn_out_compare);
    std::cout << ",";
    std::cout << "\"ffn_out_vector\":";
    write_value_stats(ffn_out_stats);
    std::cout << ",";
    std::cout << "\"final_output_comparison\":";
    write_compare_stats(final_output_compare);
    std::cout << ",";
    std::cout << "\"final_output_vector\":";
    write_value_stats(final_output_stats);
    std::cout << ",";
    std::cout << "\"layer_output_comparison\":";
    write_compare_stats(layer_output_compare);
    std::cout << ",";
    std::cout << "\"layer_output_vector\":";
    write_value_stats(layer_output_stats);
    std::cout << ",";
    std::cout << "\"linear_attention_out_comparison\":";
    write_compare_stats(linear_attention_out_compare);
    std::cout << ",";
    std::cout << "\"linear_attention_out_vector\":";
    write_value_stats(linear_attention_out_stats);
    std::cout << ",";
    std::cout << "\"layer_index\":" << layer_index << ",";
    std::cout << "\"load_map_ready\":" << (load_map.ready ? "true" : "false") << ",";
    std::cout << "\"model_path\":\"" << json_escape(model_path) << "\",";
    std::cout << "\"native_topk\":";
    write_i32_vector(result.ffn.router.expert_ids);
    std::cout << ",";
    std::cout << "\"oracle_layer_output_vector\":";
    write_value_stats(oracle_layer_output_stats);
    std::cout << ",";
    std::cout << "\"oracle_topk\":";
    write_i32_vector(oracle_topk);
    std::cout << ",";
    std::cout << "\"passed\":" << (passed ? "true" : "false") << ",";
    std::cout << "\"recurrent_state_vector\":";
    write_value_stats(state_predelta_stats);
    std::cout << ",";
    std::cout << "\"residual_input_payload_path\":\""
              << json_escape(residual_input_path) << "\",";
    std::cout << "\"residual_input_vector\":";
    write_value_stats(residual_input_stats);
    std::cout << ",";
    std::cout << "\"router_logits_vector\":";
    write_value_stats(router_logits_stats);
    std::cout << ",";
    std::cout << "\"schema_version\":\"intel-qwen36-engine-layer-postconv-compare-v0\",";
    std::cout << "\"topk_comparison\":";
    write_int_compare_stats(topk_compare);
    std::cout << ",";
    std::cout << "\"weights_norm_comparison\":";
    write_compare_stats(weights_norm_compare);
    std::cout << "}";
    return passed ? 0 : 1;
  } catch (const std::exception& exc) {
    std::cerr << "iq36-layer-postconv-compare failed: " << exc.what() << "\n";
    return 2;
  }
}
