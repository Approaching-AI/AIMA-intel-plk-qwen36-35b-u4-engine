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

constexpr const char* kWeightTensorName = "blk.0.ffn_gate_inp.weight";
constexpr double kMismatchThreshold = 1e-4;
constexpr double kMaxAbsDiffThreshold = 1e-4;
constexpr double kRmseThreshold = 1e-5;
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

}  // namespace

int main(int argc, char** argv) {
  try {
    require(argc == 4,
            "usage: iq36-router-logits-compare <model.gguf> <oracle-input-f32> <oracle-output-f32>");
    const std::string model_path = argv[1];
    const std::string oracle_input_path = argv[2];
    const std::string oracle_output_path = argv[3];

    const auto index = iq36::parse_gguf_model_index(model_path);
    const auto load_map = iq36::validate_qwen36_load_map(index);
    const auto* tensor = iq36::find_tensor(index, kWeightTensorName);
    require(tensor != nullptr, "L0 router logits tensor missing");
    const bool tensor_shape_ok =
        tensor->type == 0 &&
        tensor->dims == std::vector<std::uint64_t>{2048, 256};

    const auto input = iq36::read_f32_vector_file(oracle_input_path);
    const auto native = iq36::matvec_tensor(model_path, index, kWeightTensorName, input);
    const auto oracle = iq36::read_f32_vector_file(oracle_output_path);

    const auto input_stats = stats_from_values(input);
    const auto native_stats = stats_from_values(native);
    const auto oracle_stats = stats_from_values(oracle);
    const auto compare = iq36::compare_vectors(native, oracle, kMismatchThreshold);

    const bool passed =
        load_map.ready &&
        tensor_shape_ok &&
        input_stats.count == 2048 &&
        native_stats.count == 256 &&
        oracle_stats.count == 256 &&
        input_stats.finite &&
        native_stats.finite &&
        oracle_stats.finite &&
        input_stats.nonzero &&
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
    std::cout << "\"input_payload_path\":\"" << json_escape(oracle_input_path) << "\",";
    std::cout << "\"input_vector\":";
    write_value_stats(input_stats);
    std::cout << ",";
    std::cout << "\"load_map_ready\":" << (load_map.ready ? "true" : "false") << ",";
    std::cout << "\"model_path\":\"" << json_escape(model_path) << "\",";
    std::cout << "\"native_vector\":";
    write_value_stats(native_stats);
    std::cout << ",";
    std::cout << "\"oracle_payload_path\":\"" << json_escape(oracle_output_path) << "\",";
    std::cout << "\"oracle_vector\":";
    write_value_stats(oracle_stats);
    std::cout << ",";
    std::cout << "\"passed\":" << (passed ? "true" : "false") << ",";
    std::cout << "\"schema_version\":\"intel-qwen36-engine-router-logits-compare-v0\",";
    std::cout << "\"tensor\":{";
    std::cout << "\"dims\":";
    write_u64_vector(tensor->dims);
    std::cout << ",";
    std::cout << "\"name\":\"" << json_escape(tensor->name) << "\",";
    std::cout << "\"nbytes\":" << tensor->nbytes << ",";
    std::cout << "\"shape_ok\":" << (tensor_shape_ok ? "true" : "false") << ",";
    std::cout << "\"type_name\":\"" << iq36::ggml_type_name(tensor->type) << "\"";
    std::cout << "},";
    std::cout << "\"thresholds\":{";
    std::cout << "\"max_abs_diff\":" << kMaxAbsDiffThreshold << ",";
    std::cout << "\"min_cosine\":" << kMinCosine << ",";
    std::cout << "\"mismatch_abs_diff\":" << kMismatchThreshold << ",";
    std::cout << "\"rmse\":" << kRmseThreshold;
    std::cout << "}";
    std::cout << "}\n";
    return passed ? 0 : 2;
  } catch (const std::exception& exc) {
    std::cerr << "iq36-router-logits-compare: " << exc.what() << "\n";
    return 1;
  }
}
