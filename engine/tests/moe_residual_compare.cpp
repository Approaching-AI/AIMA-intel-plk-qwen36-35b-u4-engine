#include "intel_qwen36/gguf_loader.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr const char* kSharedGateTensorName = "blk.0.ffn_gate_inp_shexp.weight";
constexpr double kMismatchThreshold = 5e-3;
constexpr double kMaxAbsDiffThreshold = 5e-3;
constexpr double kRmseThreshold = 5e-4;
constexpr double kMinCosine = 0.999;
constexpr int kExpertUsedCount = 8;
constexpr int kHiddenSize = 2048;
constexpr int kWeightedValueCount = kExpertUsedCount * kHiddenSize;

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
         stats.rmse <= kRmseThreshold &&
         stats.cosine >= kMinCosine;
}

float sigmoid(float value) {
  const double x = value;
  const double result =
      x >= 0.0 ? 1.0 / (1.0 + std::exp(-x))
               : std::exp(x) / (1.0 + std::exp(x));
  return static_cast<float>(result);
}

std::vector<float> apply_expert_weights(const std::vector<float>& down,
                                        const std::vector<float>& weights) {
  if (down.size() != kWeightedValueCount) {
    throw std::invalid_argument("MoE down vector size mismatch");
  }
  if (weights.size() != kExpertUsedCount) {
    throw std::invalid_argument("MoE normalized weights size mismatch");
  }
  std::vector<float> output;
  output.reserve(down.size());
  for (int expert = 0; expert < kExpertUsedCount; ++expert) {
    const std::size_t base = static_cast<std::size_t>(expert * kHiddenSize);
    for (int i = 0; i < kHiddenSize; ++i) {
      output.push_back(down[base + static_cast<std::size_t>(i)] *
                       weights[static_cast<std::size_t>(expert)]);
    }
  }
  return output;
}

std::vector<float> aggregate_experts(const std::vector<float>& weighted) {
  if (weighted.size() != kWeightedValueCount) {
    throw std::invalid_argument("weighted MoE vector size mismatch");
  }
  std::vector<float> output;
  output.reserve(kHiddenSize);
  for (int i = 0; i < kHiddenSize; ++i) {
    float acc = weighted[static_cast<std::size_t>(i)];
    for (int expert = 1; expert < kExpertUsedCount; ++expert) {
      acc += weighted[static_cast<std::size_t>(expert * kHiddenSize + i)];
    }
    output.push_back(acc);
  }
  return output;
}

std::vector<float> multiply_by_scalar(const std::vector<float>& input,
                                      float scalar) {
  std::vector<float> output;
  output.reserve(input.size());
  for (const auto value : input) {
    output.push_back(value * scalar);
  }
  return output;
}

std::vector<float> add_float_vectors(const std::vector<float>& lhs,
                                     const std::vector<float>& rhs) {
  if (lhs.size() != rhs.size()) {
    throw std::invalid_argument("float vector add size mismatch");
  }
  std::vector<float> output;
  output.reserve(lhs.size());
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    output.push_back(lhs[i] + rhs[i]);
  }
  return output;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    require(argc == 13,
            "usage: iq36-moe-residual-compare <model.gguf> "
            "<attn-post-norm-f32> <ffn-moe-down-f32> "
            "<ffn-moe-weights-norm-f32> <ffn-moe-weighted-f32> "
            "<ffn-shexp-f32> <shared-gate-f32> <shared-gate-sigmoid-f32> "
            "<ffn-shexp-gated-f32> <ffn-out-f32> <attn-residual-f32> "
            "<moe-residual-f32>");
    const std::string model_path = argv[1];
    const std::string attn_post_norm_path = argv[2];
    const std::string ffn_moe_down_path = argv[3];
    const std::string weights_norm_path = argv[4];
    const std::string ffn_moe_weighted_path = argv[5];
    const std::string ffn_shexp_path = argv[6];
    const std::string shared_gate_path = argv[7];
    const std::string shared_gate_sigmoid_path = argv[8];
    const std::string ffn_shexp_gated_path = argv[9];
    const std::string ffn_out_path = argv[10];
    const std::string attn_residual_path = argv[11];
    const std::string moe_residual_path = argv[12];

    const auto index = iq36::parse_gguf_model_index(model_path);
    const auto load_map = iq36::validate_qwen36_load_map(index);
    const auto* shared_gate_tensor =
        iq36::find_tensor(index, kSharedGateTensorName);
    require(shared_gate_tensor != nullptr, "L0 shared expert gate tensor missing");
    const bool tensor_shape_ok =
        shared_gate_tensor->type == 0 &&
        shared_gate_tensor->dims == std::vector<std::uint64_t>{2048};

    const auto attn_post_norm = iq36::read_f32_vector_file(attn_post_norm_path);
    const auto ffn_moe_down = iq36::read_f32_vector_file(ffn_moe_down_path);
    const auto weights_norm = iq36::read_f32_vector_file(weights_norm_path);
    const auto oracle_weighted = iq36::read_f32_vector_file(ffn_moe_weighted_path);
    const auto ffn_shexp = iq36::read_f32_vector_file(ffn_shexp_path);
    const auto oracle_shared_gate = iq36::read_f32_vector_file(shared_gate_path);
    const auto oracle_shared_gate_sigmoid =
        iq36::read_f32_vector_file(shared_gate_sigmoid_path);
    const auto oracle_shexp_gated =
        iq36::read_f32_vector_file(ffn_shexp_gated_path);
    const auto oracle_ffn_out = iq36::read_f32_vector_file(ffn_out_path);
    const auto attn_residual = iq36::read_f32_vector_file(attn_residual_path);
    const auto oracle_moe_residual =
        iq36::read_f32_vector_file(moe_residual_path);

    const auto native_shared_gate = iq36::matvec_tensor(
        model_path, index, kSharedGateTensorName, attn_post_norm);
    require(native_shared_gate.size() == 1, "shared gate output size mismatch");
    const std::vector<float> native_shared_gate_sigmoid{
        sigmoid(native_shared_gate[0])};
    const auto native_shexp_gated =
        multiply_by_scalar(ffn_shexp, native_shared_gate_sigmoid[0]);
    const auto native_weighted =
        apply_expert_weights(ffn_moe_down, weights_norm);
    const auto native_moe_out = aggregate_experts(native_weighted);
    const auto native_ffn_out =
        add_float_vectors(native_moe_out, native_shexp_gated);
    const auto native_moe_residual =
        add_float_vectors(attn_residual, native_ffn_out);

    const auto weighted_compare =
        iq36::compare_vectors(native_weighted, oracle_weighted, kMismatchThreshold);
    const auto shared_gate_compare = iq36::compare_vectors(
        native_shared_gate, oracle_shared_gate, kMismatchThreshold);
    const auto shared_gate_sigmoid_compare = iq36::compare_vectors(
        native_shared_gate_sigmoid, oracle_shared_gate_sigmoid, kMismatchThreshold);
    const auto shexp_gated_compare = iq36::compare_vectors(
        native_shexp_gated, oracle_shexp_gated, kMismatchThreshold);
    const auto ffn_out_compare =
        iq36::compare_vectors(native_ffn_out, oracle_ffn_out, kMismatchThreshold);
    const auto moe_residual_compare = iq36::compare_vectors(
        native_moe_residual, oracle_moe_residual, kMismatchThreshold);

    const auto attn_post_norm_stats = stats_from_values(attn_post_norm);
    const auto ffn_moe_down_stats = stats_from_values(ffn_moe_down);
    const auto weights_norm_stats = stats_from_values(weights_norm);
    const auto native_weighted_stats = stats_from_values(native_weighted);
    const auto ffn_shexp_stats = stats_from_values(ffn_shexp);
    const auto native_shared_gate_stats = stats_from_values(native_shared_gate);
    const auto native_shared_gate_sigmoid_stats =
        stats_from_values(native_shared_gate_sigmoid);
    const auto native_shexp_gated_stats = stats_from_values(native_shexp_gated);
    const auto native_ffn_out_stats = stats_from_values(native_ffn_out);
    const auto attn_residual_stats = stats_from_values(attn_residual);
    const auto native_moe_residual_stats = stats_from_values(native_moe_residual);
    const auto oracle_moe_residual_stats = stats_from_values(oracle_moe_residual);

    const bool passed =
        load_map.ready &&
        tensor_shape_ok &&
        attn_post_norm_stats.count == kHiddenSize &&
        ffn_moe_down_stats.count == kWeightedValueCount &&
        weights_norm_stats.count == kExpertUsedCount &&
        native_weighted_stats.count == kWeightedValueCount &&
        ffn_shexp_stats.count == kHiddenSize &&
        native_shared_gate_stats.count == 1 &&
        native_shared_gate_sigmoid_stats.count == 1 &&
        native_shexp_gated_stats.count == kHiddenSize &&
        native_ffn_out_stats.count == kHiddenSize &&
        attn_residual_stats.count == kHiddenSize &&
        native_moe_residual_stats.count == kHiddenSize &&
        oracle_moe_residual_stats.count == kHiddenSize &&
        attn_post_norm_stats.finite &&
        ffn_moe_down_stats.finite &&
        weights_norm_stats.finite &&
        native_weighted_stats.finite &&
        ffn_shexp_stats.finite &&
        native_shared_gate_stats.finite &&
        native_shared_gate_sigmoid_stats.finite &&
        native_shexp_gated_stats.finite &&
        native_ffn_out_stats.finite &&
        attn_residual_stats.finite &&
        native_moe_residual_stats.finite &&
        oracle_moe_residual_stats.finite &&
        attn_post_norm_stats.nonzero &&
        ffn_moe_down_stats.nonzero &&
        weights_norm_stats.nonzero &&
        native_weighted_stats.nonzero &&
        ffn_shexp_stats.nonzero &&
        native_shared_gate_stats.nonzero &&
        native_shared_gate_sigmoid_stats.nonzero &&
        native_shexp_gated_stats.nonzero &&
        native_ffn_out_stats.nonzero &&
        attn_residual_stats.nonzero &&
        native_moe_residual_stats.nonzero &&
        oracle_moe_residual_stats.nonzero &&
        compare_passed(weighted_compare) &&
        compare_passed(shared_gate_compare) &&
        compare_passed(shared_gate_sigmoid_compare) &&
        compare_passed(shexp_gated_compare) &&
        compare_passed(ffn_out_compare) &&
        compare_passed(moe_residual_compare);

    std::cout << std::setprecision(17);
    std::cout << "{";
    std::cout << "\"attn_post_norm_vector\":";
    write_value_stats(attn_post_norm_stats);
    std::cout << ",";
    std::cout << "\"attn_residual_vector\":";
    write_value_stats(attn_residual_stats);
    std::cout << ",";
    std::cout << "\"ffn_moe_down_vector\":";
    write_value_stats(ffn_moe_down_stats);
    std::cout << ",";
    std::cout << "\"ffn_moe_weighted_comparison\":";
    write_compare_stats(weighted_compare);
    std::cout << ",";
    std::cout << "\"ffn_moe_weighted_vector\":";
    write_value_stats(native_weighted_stats);
    std::cout << ",";
    std::cout << "\"ffn_out_comparison\":";
    write_compare_stats(ffn_out_compare);
    std::cout << ",";
    std::cout << "\"ffn_out_vector\":";
    write_value_stats(native_ffn_out_stats);
    std::cout << ",";
    std::cout << "\"ffn_shexp_gated_comparison\":";
    write_compare_stats(shexp_gated_compare);
    std::cout << ",";
    std::cout << "\"ffn_shexp_gated_vector\":";
    write_value_stats(native_shexp_gated_stats);
    std::cout << ",";
    std::cout << "\"ffn_shexp_vector\":";
    write_value_stats(ffn_shexp_stats);
    std::cout << ",";
    std::cout << "\"load_map_ready\":" << (load_map.ready ? "true" : "false") << ",";
    std::cout << "\"model_path\":\"" << json_escape(model_path) << "\",";
    std::cout << "\"moe_residual_comparison\":";
    write_compare_stats(moe_residual_compare);
    std::cout << ",";
    std::cout << "\"moe_residual_vector\":";
    write_value_stats(native_moe_residual_stats);
    std::cout << ",";
    std::cout << "\"oracle_moe_residual_vector\":";
    write_value_stats(oracle_moe_residual_stats);
    std::cout << ",";
    std::cout << "\"passed\":" << (passed ? "true" : "false") << ",";
    std::cout << "\"schema_version\":\"intel-qwen36-engine-moe-residual-compare-v0\",";
    std::cout << "\"shared_gate_comparison\":";
    write_compare_stats(shared_gate_compare);
    std::cout << ",";
    std::cout << "\"shared_gate_sigmoid_comparison\":";
    write_compare_stats(shared_gate_sigmoid_compare);
    std::cout << ",";
    std::cout << "\"shared_gate_sigmoid_vector\":";
    write_value_stats(native_shared_gate_sigmoid_stats);
    std::cout << ",";
    std::cout << "\"shared_gate_vector\":";
    write_value_stats(native_shared_gate_stats);
    std::cout << ",";
    std::cout << "\"tensor\":{";
    std::cout << "\"dims\":";
    write_u64_vector(shared_gate_tensor->dims);
    std::cout << ",";
    std::cout << "\"name\":\"" << json_escape(shared_gate_tensor->name) << "\",";
    std::cout << "\"nbytes\":" << shared_gate_tensor->nbytes << ",";
    std::cout << "\"shape_ok\":" << (tensor_shape_ok ? "true" : "false") << ",";
    std::cout << "\"type_name\":\"" << iq36::ggml_type_name(shared_gate_tensor->type) << "\"";
    std::cout << "},";
    std::cout << "\"thresholds\":{";
    std::cout << "\"max_abs_diff\":" << kMaxAbsDiffThreshold << ",";
    std::cout << "\"min_cosine\":" << kMinCosine << ",";
    std::cout << "\"mismatch_abs_diff\":" << kMismatchThreshold << ",";
    std::cout << "\"rmse\":" << kRmseThreshold;
    std::cout << "},";
    std::cout << "\"weights_norm_vector\":";
    write_value_stats(weights_norm_stats);
    std::cout << "}\n";
    return passed ? 0 : 2;
  } catch (const std::exception& exc) {
    std::cerr << "iq36-moe-residual-compare: " << exc.what() << "\n";
    return 1;
  }
}
