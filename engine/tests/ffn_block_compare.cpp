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

constexpr double kVectorMismatchThreshold = 5e-3;
constexpr double kVectorMaxAbsDiffThreshold = 5e-3;
constexpr double kVectorRmseThreshold = 5e-4;
constexpr double kWeightsMismatchThreshold = 2e-5;
constexpr double kWeightsMaxAbsDiffThreshold = 2e-5;
constexpr double kWeightsRmseThreshold = 1e-6;
constexpr double kMinCosine = 0.999;
constexpr float kRmsNormEpsilon = 9.999999974752427e-07f;
constexpr int kLayerIndex = 0;
constexpr int kHiddenSize = 2048;
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

}  // namespace

int main(int argc, char** argv) {
  try {
    require(argc == 7,
            "usage: iq36-ffn-block-compare <model.gguf> "
            "<attn-residual-f32> <oracle-topk-i32> "
            "<oracle-weights-norm-f32> <oracle-ffn-out-f32> "
            "<oracle-moe-residual-f32>");
    const std::string model_path = argv[1];
    const std::string attn_residual_path = argv[2];
    const std::string oracle_topk_path = argv[3];
    const std::string oracle_weights_norm_path = argv[4];
    const std::string oracle_ffn_out_path = argv[5];
    const std::string oracle_moe_residual_path = argv[6];

    const auto index = iq36::parse_gguf_model_index(model_path);
    const auto load_map = iq36::validate_qwen36_load_map(index);
    const auto attn_residual = iq36::read_f32_vector_file(attn_residual_path);
    const auto oracle_topk = read_i32_vector_file(oracle_topk_path);
    const auto oracle_weights_norm =
        iq36::read_f32_vector_file(oracle_weights_norm_path);
    const auto oracle_ffn_out =
        iq36::read_f32_vector_file(oracle_ffn_out_path);
    const auto oracle_moe_residual =
        iq36::read_f32_vector_file(oracle_moe_residual_path);

    const auto result = iq36::run_qwen36_moe_ffn_layer(
        model_path, index, kLayerIndex, attn_residual, kRmsNormEpsilon);

    const auto topk_compare =
        compare_i32_vectors(result.router.expert_ids, oracle_topk);
    const auto weights_norm_compare = iq36::compare_vectors(
        result.router.normalized_weights,
        oracle_weights_norm,
        kWeightsMismatchThreshold);
    const auto ffn_out_compare =
        iq36::compare_vectors(result.ffn_out, oracle_ffn_out, kVectorMismatchThreshold);
    const auto moe_residual_compare = iq36::compare_vectors(
        result.residual, oracle_moe_residual, kVectorMismatchThreshold);

    const auto attn_residual_stats = stats_from_values(attn_residual);
    const auto ffn_norm_stats = stats_from_values(result.ffn_norm);
    const auto router_logits_stats = stats_from_values(result.router_logits);
    const auto selected_gate_up_stats = stats_from_values(result.selected_gate_up);
    const auto selected_swiglu_stats = stats_from_values(result.selected_swiglu);
    const auto selected_down_stats = stats_from_values(result.selected_down);
    const auto weighted_selected_down_stats =
        stats_from_values(result.weighted_selected_down);
    const auto moe_out_stats = stats_from_values(result.moe_out);
    const auto shared_swiglu_stats = stats_from_values(result.shared_swiglu);
    const auto shared_down_stats = stats_from_values(result.shared_down);
    const auto shared_gate_stats = stats_from_values(result.shared_gate);
    const auto shared_gate_sigmoid_stats =
        stats_from_values(result.shared_gate_sigmoid);
    const auto shared_gated_stats = stats_from_values(result.shared_gated);
    const auto ffn_out_stats = stats_from_values(result.ffn_out);
    const auto moe_residual_stats = stats_from_values(result.residual);
    const auto oracle_ffn_out_stats = stats_from_values(oracle_ffn_out);
    const auto oracle_moe_residual_stats = stats_from_values(oracle_moe_residual);

    const bool passed =
        load_map.ready &&
        attn_residual_stats.count == kHiddenSize &&
        ffn_norm_stats.count == kHiddenSize &&
        router_logits_stats.count == 256 &&
        result.router.expert_ids.size() == kExpertUsedCount &&
        result.router.normalized_weights.size() == kExpertUsedCount &&
        selected_gate_up_stats.count == 8192 &&
        selected_swiglu_stats.count == 4096 &&
        selected_down_stats.count == 16384 &&
        weighted_selected_down_stats.count == 16384 &&
        moe_out_stats.count == kHiddenSize &&
        shared_swiglu_stats.count == 512 &&
        shared_down_stats.count == kHiddenSize &&
        shared_gate_stats.count == 1 &&
        shared_gate_sigmoid_stats.count == 1 &&
        shared_gated_stats.count == kHiddenSize &&
        ffn_out_stats.count == kHiddenSize &&
        moe_residual_stats.count == kHiddenSize &&
        oracle_ffn_out_stats.count == kHiddenSize &&
        oracle_moe_residual_stats.count == kHiddenSize &&
        attn_residual_stats.finite &&
        ffn_norm_stats.finite &&
        router_logits_stats.finite &&
        selected_gate_up_stats.finite &&
        selected_swiglu_stats.finite &&
        selected_down_stats.finite &&
        weighted_selected_down_stats.finite &&
        moe_out_stats.finite &&
        shared_swiglu_stats.finite &&
        shared_down_stats.finite &&
        shared_gate_stats.finite &&
        shared_gate_sigmoid_stats.finite &&
        shared_gated_stats.finite &&
        ffn_out_stats.finite &&
        moe_residual_stats.finite &&
        oracle_ffn_out_stats.finite &&
        oracle_moe_residual_stats.finite &&
        attn_residual_stats.nonzero &&
        ffn_norm_stats.nonzero &&
        router_logits_stats.nonzero &&
        selected_gate_up_stats.nonzero &&
        selected_swiglu_stats.nonzero &&
        selected_down_stats.nonzero &&
        weighted_selected_down_stats.nonzero &&
        moe_out_stats.nonzero &&
        shared_swiglu_stats.nonzero &&
        shared_down_stats.nonzero &&
        shared_gate_stats.nonzero &&
        shared_gate_sigmoid_stats.nonzero &&
        shared_gated_stats.nonzero &&
        ffn_out_stats.nonzero &&
        moe_residual_stats.nonzero &&
        oracle_ffn_out_stats.nonzero &&
        oracle_moe_residual_stats.nonzero &&
        topk_compare.same_size &&
        topk_compare.mismatch_count == 0 &&
        vector_compare_passed(
            weights_norm_compare, kWeightsMaxAbsDiffThreshold,
            kWeightsRmseThreshold) &&
        vector_compare_passed(
            ffn_out_compare, kVectorMaxAbsDiffThreshold,
            kVectorRmseThreshold) &&
        vector_compare_passed(
            moe_residual_compare, kVectorMaxAbsDiffThreshold,
            kVectorRmseThreshold);

    std::cout << std::setprecision(17);
    std::cout << "{";
    std::cout << "\"attn_residual_payload_path\":\""
              << json_escape(attn_residual_path) << "\",";
    std::cout << "\"attn_residual_vector\":";
    write_value_stats(attn_residual_stats);
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
    std::cout << "\"load_map_ready\":" << (load_map.ready ? "true" : "false") << ",";
    std::cout << "\"model_path\":\"" << json_escape(model_path) << "\",";
    std::cout << "\"moe_out_vector\":";
    write_value_stats(moe_out_stats);
    std::cout << ",";
    std::cout << "\"moe_residual_comparison\":";
    write_compare_stats(moe_residual_compare);
    std::cout << ",";
    std::cout << "\"moe_residual_vector\":";
    write_value_stats(moe_residual_stats);
    std::cout << ",";
    std::cout << "\"native_topk\":";
    write_i32_vector(result.router.expert_ids);
    std::cout << ",";
    std::cout << "\"oracle_ffn_out_payload_path\":\""
              << json_escape(oracle_ffn_out_path) << "\",";
    std::cout << "\"oracle_ffn_out_vector\":";
    write_value_stats(oracle_ffn_out_stats);
    std::cout << ",";
    std::cout << "\"oracle_moe_residual_payload_path\":\""
              << json_escape(oracle_moe_residual_path) << "\",";
    std::cout << "\"oracle_moe_residual_vector\":";
    write_value_stats(oracle_moe_residual_stats);
    std::cout << ",";
    std::cout << "\"oracle_topk\":";
    write_i32_vector(oracle_topk);
    std::cout << ",";
    std::cout << "\"oracle_topk_payload_path\":\""
              << json_escape(oracle_topk_path) << "\",";
    std::cout << "\"oracle_weights_norm_payload_path\":\""
              << json_escape(oracle_weights_norm_path) << "\",";
    std::cout << "\"passed\":" << (passed ? "true" : "false") << ",";
    std::cout << "\"router_logits_vector\":";
    write_value_stats(router_logits_stats);
    std::cout << ",";
    std::cout << "\"schema_version\":\"intel-qwen36-engine-ffn-block-compare-v0\",";
    std::cout << "\"selected_down_vector\":";
    write_value_stats(selected_down_stats);
    std::cout << ",";
    std::cout << "\"selected_gate_up_vector\":";
    write_value_stats(selected_gate_up_stats);
    std::cout << ",";
    std::cout << "\"selected_swiglu_vector\":";
    write_value_stats(selected_swiglu_stats);
    std::cout << ",";
    std::cout << "\"shared_down_vector\":";
    write_value_stats(shared_down_stats);
    std::cout << ",";
    std::cout << "\"shared_gate_sigmoid_vector\":";
    write_value_stats(shared_gate_sigmoid_stats);
    std::cout << ",";
    std::cout << "\"shared_gate_vector\":";
    write_value_stats(shared_gate_stats);
    std::cout << ",";
    std::cout << "\"shared_gated_vector\":";
    write_value_stats(shared_gated_stats);
    std::cout << ",";
    std::cout << "\"shared_swiglu_vector\":";
    write_value_stats(shared_swiglu_stats);
    std::cout << ",";
    std::cout << "\"thresholds\":{";
    std::cout << "\"min_cosine\":" << kMinCosine << ",";
    std::cout << "\"vector_max_abs_diff\":" << kVectorMaxAbsDiffThreshold << ",";
    std::cout << "\"vector_mismatch_abs_diff\":" << kVectorMismatchThreshold << ",";
    std::cout << "\"vector_rmse\":" << kVectorRmseThreshold << ",";
    std::cout << "\"weights_max_abs_diff\":" << kWeightsMaxAbsDiffThreshold << ",";
    std::cout << "\"weights_mismatch_abs_diff\":" << kWeightsMismatchThreshold << ",";
    std::cout << "\"weights_rmse\":" << kWeightsRmseThreshold;
    std::cout << "},";
    std::cout << "\"topk_comparison\":";
    write_int_compare_stats(topk_compare);
    std::cout << ",";
    std::cout << "\"weighted_selected_down_vector\":";
    write_value_stats(weighted_selected_down_stats);
    std::cout << ",";
    std::cout << "\"weights_norm_comparison\":";
    write_compare_stats(weights_norm_compare);
    std::cout << "}\n";
    return passed ? 0 : 2;
  } catch (const std::exception& exc) {
    std::cerr << "iq36-ffn-block-compare: " << exc.what() << "\n";
    return 1;
  }
}
