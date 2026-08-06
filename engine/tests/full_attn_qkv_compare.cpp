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
constexpr int kHiddenSize = 2048;
constexpr int kQFullSize = 8192;
constexpr int kQSize = 4096;
constexpr int kKvSize = 512;
constexpr int kHeadNormSize = 256;
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
            "usage: iq36-full-attn-qkv-compare "
            "<model.gguf> <oracle-payload-dir>");
    const std::string model_path = argv[1];
    const std::string payload_dir = argv[2];

    const auto index = iq36::parse_gguf_model_index(model_path);
    const auto load_map = iq36::validate_qwen36_load_map(index);
    const float epsilon = metadata_float(
        index,
        "qwen35moe.attention.layer_norm_rms_epsilon",
        1e-6f);

    const auto oracle_input =
        iq36::read_f32_vector_file(join_path(payload_dir, "layer_input.bin"));
    const auto oracle_attention_norm =
        iq36::read_f32_vector_file(join_path(payload_dir, "attention_norm.bin"));
    const auto oracle_q_full =
        iq36::read_f32_vector_file(join_path(payload_dir, "q_full.bin"));
    const auto oracle_q_normed =
        iq36::read_f32_vector_file(join_path(payload_dir, "q_normed.bin"));
    const auto oracle_k_raw =
        iq36::read_f32_vector_file(join_path(payload_dir, "k_raw.bin"));
    const auto oracle_k_normed =
        iq36::read_f32_vector_file(join_path(payload_dir, "k_normed.bin"));
    const auto oracle_v =
        iq36::read_f32_vector_file(join_path(payload_dir, "v.bin"));

    const auto prefix = std::string("blk.") + std::to_string(kLayerIndex) + ".";
    const auto* norm_tensor = iq36::find_tensor(index, prefix + "attn_norm.weight");
    const auto* q_tensor = iq36::find_tensor(index, prefix + "attn_q.weight");
    const auto* q_norm_tensor = iq36::find_tensor(index, prefix + "attn_q_norm.weight");
    const auto* k_tensor = iq36::find_tensor(index, prefix + "attn_k.weight");
    const auto* k_norm_tensor = iq36::find_tensor(index, prefix + "attn_k_norm.weight");
    const auto* v_tensor = iq36::find_tensor(index, prefix + "attn_v.weight");
    require(norm_tensor != nullptr && q_tensor != nullptr &&
                q_norm_tensor != nullptr && k_tensor != nullptr &&
                k_norm_tensor != nullptr && v_tensor != nullptr,
            "full attention qkv tensor set is incomplete");
    const bool tensors_shape_ok =
        norm_tensor->type == 0 &&
        norm_tensor->dims == std::vector<std::uint64_t>{kHiddenSize} &&
        q_tensor->type == 12 &&
        q_tensor->dims == std::vector<std::uint64_t>{kHiddenSize, kQFullSize} &&
        q_norm_tensor->type == 0 &&
        q_norm_tensor->dims == std::vector<std::uint64_t>{kHeadNormSize} &&
        k_tensor->type == 12 &&
        k_tensor->dims == std::vector<std::uint64_t>{kHiddenSize, kKvSize} &&
        k_norm_tensor->type == 0 &&
        k_norm_tensor->dims == std::vector<std::uint64_t>{kHeadNormSize} &&
        (v_tensor->type == 12 || v_tensor->type == 14) &&
        v_tensor->dims == std::vector<std::uint64_t>{kHiddenSize, kKvSize};

    const auto native = iq36::run_qwen36_full_attention_qkv_projection(
        model_path, index, kLayerIndex, oracle_input, epsilon);

    const std::vector<std::pair<std::string, iq36::VectorCompareStats>> comparisons = {
        {"attention_norm", iq36::compare_vectors(native.attention_norm, oracle_attention_norm, kMismatchThreshold)},
        {"q_full", iq36::compare_vectors(native.q_full, oracle_q_full, kMismatchThreshold)},
        {"q_normed", iq36::compare_vectors(native.q_normed, oracle_q_normed, kMismatchThreshold)},
        {"k_raw", iq36::compare_vectors(native.k_raw, oracle_k_raw, kMismatchThreshold)},
        {"k_normed", iq36::compare_vectors(native.k_normed, oracle_k_normed, kMismatchThreshold)},
        {"v", iq36::compare_vectors(native.v, oracle_v, kMismatchThreshold)},
    };

    bool comparisons_ok = true;
    for (const auto& item : comparisons) {
      comparisons_ok = comparisons_ok && vector_compare_passed(item.second);
    }

    const std::vector<std::pair<std::string, ValueStats>> native_stats = {
        {"attention_norm", stats_from_values(native.attention_norm)},
        {"q_full", stats_from_values(native.q_full)},
        {"q_raw", stats_from_values(native.q_raw)},
        {"q_gate", stats_from_values(native.q_gate)},
        {"q_normed", stats_from_values(native.q_normed)},
        {"k_raw", stats_from_values(native.k_raw)},
        {"k_normed", stats_from_values(native.k_normed)},
        {"v", stats_from_values(native.v)},
    };
    const bool stats_ok = std::all_of(
        native_stats.begin(), native_stats.end(), [](const auto& item) {
          return item.second.finite && item.second.nonzero;
        });

    const bool counts_ok =
        oracle_input.size() == kHiddenSize &&
        native.attention_norm.size() == kHiddenSize &&
        native.q_full.size() == kQFullSize &&
        native.q_raw.size() == kQSize &&
        native.q_gate.size() == kQSize &&
        native.q_normed.size() == kQSize &&
        native.k_raw.size() == kKvSize &&
        native.k_normed.size() == kKvSize &&
        native.v.size() == kKvSize &&
        oracle_attention_norm.size() == kHiddenSize &&
        oracle_q_full.size() == kQFullSize &&
        oracle_q_normed.size() == kQSize &&
        oracle_k_raw.size() == kKvSize &&
        oracle_k_normed.size() == kKvSize &&
        oracle_v.size() == kKvSize;

    const bool passed =
        load_map.ready && tensors_shape_ok && counts_ok && stats_ok &&
        comparisons_ok;

    std::cout << std::setprecision(17);
    std::cout << "{";
    std::cout << "\"comparisons\":";
    write_named_comparisons(comparisons);
    std::cout << ",\"epsilon\":" << epsilon;
    std::cout << ",\"layer_index\":" << kLayerIndex;
    std::cout << ",\"load_map_ready\":" << (load_map.ready ? "true" : "false");
    std::cout << ",\"model_path\":\"" << json_escape(model_path) << "\"";
    std::cout << ",\"native_vectors\":";
    write_named_stats(native_stats);
    std::cout << ",\"passed\":" << (passed ? "true" : "false");
    std::cout << ",\"prompt_case_id\":\"short_math_001\"";
    std::cout << ",\"schema_version\":\"intel-qwen36-engine-full-attn-qkv-compare-v0\"";
    std::cout << ",\"source_token_position\":" << kSourceTokenPosition;
    std::cout << ",\"tensors\":{";
    std::cout << "\"attention_norm\":{\"dims\":";
    write_u64_vector(norm_tensor->dims);
    std::cout << ",\"name\":\"" << json_escape(norm_tensor->name)
              << "\",\"type_name\":\"" << iq36::ggml_type_name(norm_tensor->type)
              << "\"},";
    std::cout << "\"q\":{\"dims\":";
    write_u64_vector(q_tensor->dims);
    std::cout << ",\"name\":\"" << json_escape(q_tensor->name)
              << "\",\"type_name\":\"" << iq36::ggml_type_name(q_tensor->type)
              << "\"},";
    std::cout << "\"q_norm\":{\"dims\":";
    write_u64_vector(q_norm_tensor->dims);
    std::cout << ",\"name\":\"" << json_escape(q_norm_tensor->name)
              << "\",\"type_name\":\"" << iq36::ggml_type_name(q_norm_tensor->type)
              << "\"},";
    std::cout << "\"k\":{\"dims\":";
    write_u64_vector(k_tensor->dims);
    std::cout << ",\"name\":\"" << json_escape(k_tensor->name)
              << "\",\"type_name\":\"" << iq36::ggml_type_name(k_tensor->type)
              << "\"},";
    std::cout << "\"k_norm\":{\"dims\":";
    write_u64_vector(k_norm_tensor->dims);
    std::cout << ",\"name\":\"" << json_escape(k_norm_tensor->name)
              << "\",\"type_name\":\"" << iq36::ggml_type_name(k_norm_tensor->type)
              << "\"},";
    std::cout << "\"v\":{\"dims\":";
    write_u64_vector(v_tensor->dims);
    std::cout << ",\"name\":\"" << json_escape(v_tensor->name)
              << "\",\"type_name\":\"" << iq36::ggml_type_name(v_tensor->type)
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
    std::cerr << "iq36-full-attn-qkv-compare failed: " << exc.what() << "\n";
    return 1;
  }
}
