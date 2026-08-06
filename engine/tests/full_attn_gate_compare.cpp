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

constexpr int kLayerIndex = 3;
constexpr int kQFullSize = 8192;
constexpr int kAttentionSize = 4096;
constexpr int kHeadDim = 256;
constexpr int kSourceTokenPosition = 15;
constexpr const char* kQTensorName = "blk.3.attn_q.weight";
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

}  // namespace

int main(int argc, char** argv) {
  try {
    require(argc == 3,
            "usage: iq36-full-attn-gate-compare "
            "<model.gguf> <oracle-payload-dir>");
    const std::string model_path = argv[1];
    const std::string payload_dir = argv[2];

    const auto index = iq36::parse_gguf_model_index(model_path);
    const auto load_map = iq36::validate_qwen36_load_map(index);
    const auto* q_tensor = iq36::find_tensor(index, kQTensorName);
    require(q_tensor != nullptr, "L3 full-attention q tensor missing");
    const bool q_tensor_shape_ok =
        q_tensor->type == 12 &&
        q_tensor->dims == std::vector<std::uint64_t>{2048, kQFullSize};

    const auto q_full =
        iq36::read_f32_vector_file(join_path(payload_dir, "q_full.bin"));
    const auto attn_pregate =
        iq36::read_f32_vector_file(join_path(payload_dir, "attn_pregate.bin"));
    const auto oracle_attn_gated =
        iq36::read_f32_vector_file(join_path(payload_dir, "attn_gated.bin"));
    const auto native =
        iq36::run_qwen36_full_attention_gate(q_full, attn_pregate, kHeadDim);

    const auto comparison =
        iq36::compare_vectors(native.attn_gated, oracle_attn_gated, kMismatchThreshold);
    const std::vector<std::pair<std::string, ValueStats>> vectors = {
        {"q_full", stats_from_values(q_full)},
        {"attn_pregate", stats_from_values(attn_pregate)},
        {"q_gate", stats_from_values(native.q_gate)},
        {"gate_sigmoid", stats_from_values(native.gate_sigmoid)},
        {"native_attn_gated", stats_from_values(native.attn_gated)},
        {"oracle_attn_gated", stats_from_values(oracle_attn_gated)},
    };

    bool counts_ok = true;
    bool stats_ok = true;
    for (const auto& item : vectors) {
      const auto expected_count =
          item.first == "q_full" ? kQFullSize : kAttentionSize;
      counts_ok = counts_ok && item.second.count == expected_count;
      stats_ok = stats_ok && item.second.finite && item.second.nonzero;
    }
    const bool passed =
        load_map.ready && q_tensor_shape_ok && counts_ok && stats_ok &&
        vector_compare_passed(comparison);

    std::cout << std::setprecision(17);
    std::cout << "{";
    std::cout << "\"comparison\":";
    write_compare_stats(comparison);
    std::cout << ",\"gate_layout\":{";
    std::cout << "\"head_dim\":" << kHeadDim << ",";
    std::cout << "\"q_full_layout\":\"per_head_query_then_gate\",";
    std::cout << "\"query_value_count\":" << kAttentionSize << ",";
    std::cout << "\"gate_value_count\":" << kAttentionSize;
    std::cout << "}";
    std::cout << ",\"layer_index\":" << kLayerIndex;
    std::cout << ",\"load_map_ready\":" << (load_map.ready ? "true" : "false");
    std::cout << ",\"model_path\":\"" << json_escape(model_path) << "\"";
    std::cout << ",\"passed\":" << (passed ? "true" : "false");
    std::cout << ",\"prompt_case_id\":\"short_math_001\"";
    std::cout << ",\"q_source_tensor\":{";
    std::cout << "\"dims\":";
    write_u64_vector(q_tensor->dims);
    std::cout << ",\"name\":\"" << json_escape(q_tensor->name) << "\"";
    std::cout << ",\"nbytes\":" << q_tensor->nbytes;
    std::cout << ",\"shape_ok\":" << (q_tensor_shape_ok ? "true" : "false");
    std::cout << ",\"type_name\":\"" << iq36::ggml_type_name(q_tensor->type) << "\"";
    std::cout << "}";
    std::cout << ",\"schema_version\":\"intel-qwen36-engine-full-attn-gate-compare-v0\"";
    std::cout << ",\"source_token_position\":" << kSourceTokenPosition;
    std::cout << ",\"thresholds\":{";
    std::cout << "\"max_abs_diff\":" << kMaxAbsDiffThreshold << ",";
    std::cout << "\"min_cosine\":" << kMinCosine << ",";
    std::cout << "\"mismatch_abs_diff\":" << kMismatchThreshold << ",";
    std::cout << "\"rmse\":" << kRmseThreshold;
    std::cout << "}";
    std::cout << ",\"vectors\":";
    write_named_stats(vectors);
    std::cout << "}\n";
    return passed ? 0 : 1;
  } catch (const std::exception& exc) {
    std::cerr << "iq36-full-attn-gate-compare failed: " << exc.what() << "\n";
    return 1;
  }
}
