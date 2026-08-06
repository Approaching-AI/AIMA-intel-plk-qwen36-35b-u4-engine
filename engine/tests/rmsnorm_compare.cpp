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

constexpr const char* kInputTensorName = "token_embd.weight";
constexpr const char* kWeightTensorName = "blk.0.attn_norm.weight";
constexpr double kMismatchThreshold = 2e-5;
constexpr double kMaxAbsDiffThreshold = 2e-5;
constexpr double kRmseThreshold = 1e-6;
constexpr double kMinCosine = 0.999999;

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

std::uint64_t parse_u64(const std::string& value) {
  std::size_t consumed = 0;
  const auto parsed = std::stoull(value, &consumed, 10);
  if (consumed != value.size()) {
    throw std::invalid_argument("invalid unsigned integer argument");
  }
  return parsed;
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

}  // namespace

int main(int argc, char** argv) {
  try {
    require(argc == 4,
            "usage: iq36-rmsnorm-compare <model.gguf> <token_id> <oracle-f32-payload>");
    const std::string model_path = argv[1];
    const auto token_id = parse_u64(argv[2]);
    const std::string oracle_payload_path = argv[3];

    const auto index = iq36::parse_gguf_model_index(model_path);
    const auto load_map = iq36::validate_qwen36_load_map(index);
    const auto* input_tensor = iq36::find_tensor(index, kInputTensorName);
    const auto* weight_tensor = iq36::find_tensor(index, kWeightTensorName);
    require(input_tensor != nullptr, "token embedding tensor missing");
    require(weight_tensor != nullptr, "L0 attention norm tensor missing");
    const bool input_shape_ok =
        input_tensor->type == 12 &&
        input_tensor->dims == std::vector<std::uint64_t>{2048, 248320};
    const bool weight_shape_ok =
        weight_tensor->type == 0 &&
        weight_tensor->dims == std::vector<std::uint64_t>{2048};

    const float epsilon = metadata_float(
        index,
        "qwen35moe.attention.layer_norm_rms_epsilon",
        1e-6f);
    const auto input = iq36::decode_tensor_row(model_path, index, kInputTensorName, token_id);
    const auto weight = iq36::decode_tensor_row(model_path, index, kWeightTensorName, 0);
    const auto native = iq36::apply_rms_norm(input, weight, epsilon);
    const auto oracle = iq36::read_f32_vector_file(oracle_payload_path);

    const auto input_stats = stats_from_values(input);
    const auto weight_stats = stats_from_values(weight);
    const auto native_stats = stats_from_values(native);
    const auto oracle_stats = stats_from_values(oracle);
    const auto compare = iq36::compare_vectors(native, oracle, kMismatchThreshold);

    const bool passed =
        load_map.ready &&
        input_shape_ok &&
        weight_shape_ok &&
        input_stats.count == 2048 &&
        weight_stats.count == 2048 &&
        native_stats.count == 2048 &&
        oracle_stats.count == 2048 &&
        input_stats.finite &&
        weight_stats.finite &&
        native_stats.finite &&
        oracle_stats.finite &&
        input_stats.nonzero &&
        weight_stats.nonzero &&
        native_stats.nonzero &&
        oracle_stats.nonzero &&
        compare.same_size &&
        compare.finite &&
        compare.mismatch_count == 0 &&
        compare.max_abs_diff <= kMaxAbsDiffThreshold &&
        compare.rmse <= kRmseThreshold &&
        compare.cosine >= kMinCosine;

    std::cout << std::setprecision(17);
    std::cout << "{";
    std::cout << "\"comparison\":";
    write_compare_stats(compare);
    std::cout << ",";
    std::cout << "\"epsilon\":" << epsilon << ",";
    std::cout << "\"input_tensor\":{";
    std::cout << "\"dims\":";
    write_u64_vector(input_tensor->dims);
    std::cout << ",";
    std::cout << "\"name\":\"" << json_escape(input_tensor->name) << "\",";
    std::cout << "\"shape_ok\":" << (input_shape_ok ? "true" : "false") << ",";
    std::cout << "\"type_name\":\"" << iq36::ggml_type_name(input_tensor->type) << "\"";
    std::cout << "},";
    std::cout << "\"input_vector\":";
    write_value_stats(input_stats);
    std::cout << ",";
    std::cout << "\"load_map_ready\":" << (load_map.ready ? "true" : "false") << ",";
    std::cout << "\"model_path\":\"" << json_escape(model_path) << "\",";
    std::cout << "\"native_vector\":";
    write_value_stats(native_stats);
    std::cout << ",";
    std::cout << "\"oracle_payload_path\":\"" << json_escape(oracle_payload_path) << "\",";
    std::cout << "\"oracle_vector\":";
    write_value_stats(oracle_stats);
    std::cout << ",";
    std::cout << "\"passed\":" << (passed ? "true" : "false") << ",";
    std::cout << "\"schema_version\":\"intel-qwen36-engine-rmsnorm-compare-v0\",";
    std::cout << "\"thresholds\":{";
    std::cout << "\"max_abs_diff\":" << kMaxAbsDiffThreshold << ",";
    std::cout << "\"min_cosine\":" << kMinCosine << ",";
    std::cout << "\"mismatch_abs_diff\":" << kMismatchThreshold << ",";
    std::cout << "\"rmse\":" << kRmseThreshold;
    std::cout << "},";
    std::cout << "\"token_id\":" << token_id << ",";
    std::cout << "\"weight_tensor\":{";
    std::cout << "\"dims\":";
    write_u64_vector(weight_tensor->dims);
    std::cout << ",";
    std::cout << "\"name\":\"" << json_escape(weight_tensor->name) << "\",";
    std::cout << "\"shape_ok\":" << (weight_shape_ok ? "true" : "false") << ",";
    std::cout << "\"type_name\":\"" << iq36::ggml_type_name(weight_tensor->type) << "\"";
    std::cout << "},";
    std::cout << "\"weight_vector\":";
    write_value_stats(weight_stats);
    std::cout << "}\n";
    return passed ? 0 : 2;
  } catch (const std::exception& exc) {
    std::cerr << "iq36-rmsnorm-compare: " << exc.what() << "\n";
    return 1;
  }
}
