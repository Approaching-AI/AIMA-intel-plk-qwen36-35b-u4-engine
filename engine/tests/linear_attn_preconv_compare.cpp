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

constexpr int kLayerIndex = 0;
constexpr int kHiddenSize = 2048;
constexpr int kQkvMixedSize = 8192;
constexpr int kConvSize = 8192;
constexpr int kValueHeads = 32;
constexpr double kMismatchThreshold = 5e-4;
constexpr double kMaxAbsDiffThreshold = 5e-4;
constexpr double kRmseThreshold = 5e-5;
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

}  // namespace

int main(int argc, char** argv) {
  try {
    require(argc == 3,
            "usage: iq36-linear-attn-preconv-compare <model.gguf> <oracle-payload-dir>");
    const std::string model_path = argv[1];
    const std::string payload_dir = argv[2];

    const auto index = iq36::parse_gguf_model_index(model_path);
    const auto load_map = iq36::validate_qwen36_load_map(index);

    const auto attn_norm =
        iq36::read_f32_vector_file(join_path(payload_dir, "attn_norm.bin"));
    const auto oracle_qkv_mixed =
        iq36::read_f32_vector_file(join_path(payload_dir, "linear_attn_qkv_mixed.bin"));
    const auto oracle_alpha =
        iq36::read_f32_vector_file(join_path(payload_dir, "alpha.bin"));
    const auto oracle_a_softplus =
        iq36::read_f32_vector_file(join_path(payload_dir, "a_softplus.bin"));
    const auto oracle_gate =
        iq36::read_f32_vector_file(join_path(payload_dir, "gate.bin"));
    const auto oracle_beta =
        iq36::read_f32_vector_file(join_path(payload_dir, "beta.bin"));
    const auto oracle_beta_sigmoid =
        iq36::read_f32_vector_file(join_path(payload_dir, "beta_sigmoid.bin"));
    const auto oracle_z =
        iq36::read_f32_vector_file(join_path(payload_dir, "z.bin"));

    const auto* qkv_tensor = iq36::find_tensor(index, "blk.0.attn_qkv.weight");
    const auto* alpha_tensor = iq36::find_tensor(index, "blk.0.ssm_alpha.weight");
    const auto* beta_tensor = iq36::find_tensor(index, "blk.0.ssm_beta.weight");
    const auto* z_tensor = iq36::find_tensor(index, "blk.0.attn_gate.weight");
    require(qkv_tensor != nullptr && alpha_tensor != nullptr &&
                beta_tensor != nullptr && z_tensor != nullptr,
            "linear attention preconv required tensor missing");
    const bool tensors_shape_ok =
        (qkv_tensor->type == 12 || qkv_tensor->type == 14) &&
        qkv_tensor->dims == std::vector<std::uint64_t>{kHiddenSize, kQkvMixedSize} &&
        alpha_tensor->type == 12 &&
        alpha_tensor->dims == std::vector<std::uint64_t>{kHiddenSize, kValueHeads} &&
        beta_tensor->type == 12 &&
        beta_tensor->dims == std::vector<std::uint64_t>{kHiddenSize, kValueHeads} &&
        z_tensor->type == 12 &&
        z_tensor->dims == std::vector<std::uint64_t>{kHiddenSize, kConvSize / 2};

    const auto native = iq36::run_qwen36_linear_attention_preconv_core(
        model_path, index, kLayerIndex, attn_norm);

    const std::vector<std::pair<std::string, iq36::VectorCompareStats>> comparisons = {
        {"linear_attn_qkv_mixed", iq36::compare_vectors(native.qkv_mixed, oracle_qkv_mixed, kMismatchThreshold)},
        {"alpha", iq36::compare_vectors(native.alpha, oracle_alpha, kMismatchThreshold)},
        {"a_softplus", iq36::compare_vectors(native.alpha_softplus, oracle_a_softplus, kMismatchThreshold)},
        {"gate", iq36::compare_vectors(native.gate, oracle_gate, kMismatchThreshold)},
        {"beta", iq36::compare_vectors(native.beta, oracle_beta, kMismatchThreshold)},
        {"beta_sigmoid", iq36::compare_vectors(native.beta_sigmoid, oracle_beta_sigmoid, kMismatchThreshold)},
        {"z", iq36::compare_vectors(native.z, oracle_z, kMismatchThreshold)},
    };

    bool comparisons_ok = true;
    for (const auto& item : comparisons) {
      comparisons_ok = comparisons_ok && compare_passed(item.second);
    }
    const bool counts_ok =
        attn_norm.size() == kHiddenSize &&
        native.qkv_mixed.size() == kQkvMixedSize &&
        native.alpha.size() == kValueHeads &&
        native.alpha_softplus.size() == kValueHeads &&
        native.gate.size() == kValueHeads &&
        native.beta.size() == kValueHeads &&
        native.beta_sigmoid.size() == kValueHeads &&
        native.z.size() == kConvSize / 2;

    const std::vector<std::pair<std::string, ValueStats>> input_stats = {
        {"attn_norm", stats_from_values(attn_norm)},
    };
    const std::vector<std::pair<std::string, ValueStats>> native_stats = {
        {"linear_attn_qkv_mixed", stats_from_values(native.qkv_mixed)},
        {"alpha", stats_from_values(native.alpha)},
        {"a_softplus", stats_from_values(native.alpha_softplus)},
        {"gate", stats_from_values(native.gate)},
        {"beta", stats_from_values(native.beta)},
        {"beta_sigmoid", stats_from_values(native.beta_sigmoid)},
        {"z", stats_from_values(native.z)},
    };
    const bool stats_ok =
        std::all_of(input_stats.begin(), input_stats.end(), [](const auto& item) {
          return item.second.finite && item.second.nonzero;
        }) &&
        std::all_of(native_stats.begin(), native_stats.end(), [](const auto& item) {
          return item.second.finite && item.second.nonzero;
        });
    const bool passed =
        load_map.ready && tensors_shape_ok && counts_ok && stats_ok && comparisons_ok;

    std::cout << std::setprecision(17);
    std::cout << "{";
    std::cout << "\"comparisons\":";
    write_named_comparisons(comparisons);
    std::cout << ",\"input_vectors\":";
    write_named_stats(input_stats);
    std::cout << ",\"layer_index\":" << kLayerIndex;
    std::cout << ",\"load_map_ready\":" << (load_map.ready ? "true" : "false");
    std::cout << ",\"model_path\":\"" << json_escape(model_path) << "\"";
    std::cout << ",\"native_vectors\":";
    write_named_stats(native_stats);
    std::cout << ",\"passed\":" << (passed ? "true" : "false");
    std::cout << ",\"schema_version\":\"intel-qwen36-engine-linear-attn-preconv-compare-v0\"";
    std::cout << ",\"tensors\":{";
    std::cout << "\"qkv\":{\"dims\":";
    write_u64_vector(qkv_tensor->dims);
    std::cout << ",\"name\":\"" << qkv_tensor->name << "\",\"type_name\":\""
              << iq36::ggml_type_name(qkv_tensor->type) << "\"},";
    std::cout << "\"alpha\":{\"dims\":";
    write_u64_vector(alpha_tensor->dims);
    std::cout << ",\"name\":\"" << alpha_tensor->name << "\",\"type_name\":\""
              << iq36::ggml_type_name(alpha_tensor->type) << "\"},";
    std::cout << "\"beta\":{\"dims\":";
    write_u64_vector(beta_tensor->dims);
    std::cout << ",\"name\":\"" << beta_tensor->name << "\",\"type_name\":\""
              << iq36::ggml_type_name(beta_tensor->type) << "\"},";
    std::cout << "\"z\":{\"dims\":";
    write_u64_vector(z_tensor->dims);
    std::cout << ",\"name\":\"" << z_tensor->name << "\",\"type_name\":\""
              << iq36::ggml_type_name(z_tensor->type) << "\"},";
    std::cout << "\"shape_ok\":" << (tensors_shape_ok ? "true" : "false");
    std::cout << "}}";
    return passed ? 0 : 1;
  } catch (const std::exception& exc) {
    std::cerr << "iq36-linear-attn-preconv-compare failed: " << exc.what() << "\n";
    return 2;
  }
}
