#include "intel_qwen36/gguf_loader.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr const char* kWeightTensorName = "blk.0.ffn_gate_inp.weight";
constexpr double kLogitsMismatchThreshold = 1e-4;
constexpr double kLogitsMaxAbsDiffThreshold = 1e-4;
constexpr double kLogitsRmseThreshold = 1e-5;
constexpr double kWeightsMismatchThreshold = 2e-5;
constexpr double kWeightsMaxAbsDiffThreshold = 2e-5;
constexpr double kWeightsRmseThreshold = 1e-6;
constexpr double kMinCosine = 0.999999;
constexpr int kExpertCount = 256;
constexpr int kExpertUsedCount = 8;
constexpr float kMinWeightSum = 6.103515625e-5f;

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

void write_int_compare_stats(const IntCompareStats& stats) {
  std::cout << "{";
  std::cout << "\"compared_value_count\":" << stats.compared_value_count << ",";
  std::cout << "\"lhs_value_count\":" << stats.lhs_value_count << ",";
  std::cout << "\"mismatch_count\":" << stats.mismatch_count << ",";
  std::cout << "\"rhs_value_count\":" << stats.rhs_value_count << ",";
  std::cout << "\"same_size\":" << (stats.same_size ? "true" : "false");
  std::cout << "}";
}

std::vector<float> softmax(const std::vector<float>& logits) {
  if (logits.empty()) {
    throw std::invalid_argument("softmax input is empty");
  }
  const auto max_it = std::max_element(logits.begin(), logits.end());
  double sum = 0.0;
  std::vector<double> exp_values;
  exp_values.reserve(logits.size());
  for (const auto value : logits) {
    const double exp_value = std::exp(static_cast<double>(value) -
                                      static_cast<double>(*max_it));
    exp_values.push_back(exp_value);
    sum += exp_value;
  }
  std::vector<float> probs;
  probs.reserve(logits.size());
  for (const auto value : exp_values) {
    probs.push_back(static_cast<float>(value / sum));
  }
  return probs;
}

std::vector<std::int32_t> top_k_indices(const std::vector<float>& values,
                                        int k) {
  std::vector<std::int32_t> indexes(values.size());
  std::iota(indexes.begin(), indexes.end(), 0);
  std::partial_sort(
      indexes.begin(),
      indexes.begin() + k,
      indexes.end(),
      [&values](std::int32_t lhs, std::int32_t rhs) {
        if (values[static_cast<std::size_t>(lhs)] ==
            values[static_cast<std::size_t>(rhs)]) {
          return lhs < rhs;
        }
        return values[static_cast<std::size_t>(lhs)] >
               values[static_cast<std::size_t>(rhs)];
      });
  indexes.resize(static_cast<std::size_t>(k));
  return indexes;
}

std::vector<float> gather_weights(const std::vector<float>& probs,
                                  const std::vector<std::int32_t>& indexes) {
  std::vector<float> weights;
  weights.reserve(indexes.size());
  for (const auto index : indexes) {
    if (index < 0 || static_cast<std::size_t>(index) >= probs.size()) {
      throw std::out_of_range("top-k expert index out of range");
    }
    weights.push_back(probs[static_cast<std::size_t>(index)]);
  }
  return weights;
}

std::vector<float> normalize_weights(const std::vector<float>& weights) {
  float sum = 0.0f;
  for (const auto value : weights) {
    sum += value;
  }
  const float denominator = std::max(sum, kMinWeightSum);
  std::vector<float> normalized;
  normalized.reserve(weights.size());
  for (const auto value : weights) {
    normalized.push_back(value / denominator);
  }
  return normalized;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    require(argc == 7,
            "usage: iq36-router-topk-compare <model.gguf> <oracle-input-f32> <oracle-logits-f32> <oracle-topk-i32> <oracle-weights-f32> <oracle-weights-norm-f32>");
    const std::string model_path = argv[1];
    const std::string oracle_input_path = argv[2];
    const std::string oracle_logits_path = argv[3];
    const std::string oracle_topk_path = argv[4];
    const std::string oracle_weights_path = argv[5];
    const std::string oracle_weights_norm_path = argv[6];

    const auto index = iq36::parse_gguf_model_index(model_path);
    const auto load_map = iq36::validate_qwen36_load_map(index);
    const auto* tensor = iq36::find_tensor(index, kWeightTensorName);
    require(tensor != nullptr, "L0 router logits tensor missing");
    const bool tensor_shape_ok =
        tensor->type == 0 &&
        tensor->dims == std::vector<std::uint64_t>{2048, 256};

    const auto input = iq36::read_f32_vector_file(oracle_input_path);
    const auto native_logits =
        iq36::matvec_tensor(model_path, index, kWeightTensorName, input);
    const auto native_probs = softmax(native_logits);
    const auto native_topk = top_k_indices(native_probs, kExpertUsedCount);
    const auto native_weights = gather_weights(native_probs, native_topk);
    const auto native_weights_norm = normalize_weights(native_weights);

    const auto oracle_logits = iq36::read_f32_vector_file(oracle_logits_path);
    const auto oracle_topk = read_i32_vector_file(oracle_topk_path);
    const auto oracle_weights = iq36::read_f32_vector_file(oracle_weights_path);
    const auto oracle_weights_norm =
        iq36::read_f32_vector_file(oracle_weights_norm_path);

    const auto input_stats = stats_from_values(input);
    const auto logits_stats = stats_from_values(native_logits);
    const auto weights_stats = stats_from_values(native_weights);
    const auto weights_norm_stats = stats_from_values(native_weights_norm);
    const auto oracle_logits_stats = stats_from_values(oracle_logits);
    const auto oracle_weights_stats = stats_from_values(oracle_weights);
    const auto oracle_weights_norm_stats = stats_from_values(oracle_weights_norm);
    const auto logits_compare =
        iq36::compare_vectors(native_logits, oracle_logits, kLogitsMismatchThreshold);
    const auto topk_compare = compare_i32_vectors(native_topk, oracle_topk);
    const auto weights_compare =
        iq36::compare_vectors(native_weights, oracle_weights, kWeightsMismatchThreshold);
    const auto weights_norm_compare =
        iq36::compare_vectors(
            native_weights_norm,
            oracle_weights_norm,
            kWeightsMismatchThreshold);

    const bool passed =
        load_map.ready &&
        tensor_shape_ok &&
        input_stats.count == 2048 &&
        logits_stats.count == kExpertCount &&
        oracle_logits_stats.count == kExpertCount &&
        native_topk.size() == kExpertUsedCount &&
        oracle_topk.size() == kExpertUsedCount &&
        weights_stats.count == kExpertUsedCount &&
        oracle_weights_stats.count == kExpertUsedCount &&
        weights_norm_stats.count == kExpertUsedCount &&
        oracle_weights_norm_stats.count == kExpertUsedCount &&
        input_stats.finite &&
        logits_stats.finite &&
        oracle_logits_stats.finite &&
        weights_stats.finite &&
        oracle_weights_stats.finite &&
        weights_norm_stats.finite &&
        oracle_weights_norm_stats.finite &&
        input_stats.nonzero &&
        logits_stats.nonzero &&
        oracle_logits_stats.nonzero &&
        weights_stats.nonzero &&
        oracle_weights_stats.nonzero &&
        weights_norm_stats.nonzero &&
        oracle_weights_norm_stats.nonzero &&
        logits_compare.same_size &&
        logits_compare.finite &&
        logits_compare.mismatch_count == 0 &&
        logits_compare.max_abs_diff <= kLogitsMaxAbsDiffThreshold &&
        logits_compare.rmse <= kLogitsRmseThreshold &&
        logits_compare.cosine >= kMinCosine &&
        topk_compare.same_size &&
        topk_compare.mismatch_count == 0 &&
        weights_compare.same_size &&
        weights_compare.finite &&
        weights_compare.mismatch_count == 0 &&
        weights_compare.max_abs_diff <= kWeightsMaxAbsDiffThreshold &&
        weights_compare.rmse <= kWeightsRmseThreshold &&
        weights_compare.cosine >= kMinCosine &&
        weights_norm_compare.same_size &&
        weights_norm_compare.finite &&
        weights_norm_compare.mismatch_count == 0 &&
        weights_norm_compare.max_abs_diff <= kWeightsMaxAbsDiffThreshold &&
        weights_norm_compare.rmse <= kWeightsRmseThreshold &&
        weights_norm_compare.cosine >= kMinCosine;

    std::cout << std::setprecision(17);
    std::cout << "{";
    std::cout << "\"comparison_logits\":";
    write_compare_stats(logits_compare);
    std::cout << ",";
    std::cout << "\"comparison_topk\":";
    write_int_compare_stats(topk_compare);
    std::cout << ",";
    std::cout << "\"comparison_weights\":";
    write_compare_stats(weights_compare);
    std::cout << ",";
    std::cout << "\"comparison_weights_norm\":";
    write_compare_stats(weights_norm_compare);
    std::cout << ",";
    std::cout << "\"input_payload_path\":\"" << json_escape(oracle_input_path) << "\",";
    std::cout << "\"input_vector\":";
    write_value_stats(input_stats);
    std::cout << ",";
    std::cout << "\"load_map_ready\":" << (load_map.ready ? "true" : "false") << ",";
    std::cout << "\"logits_payload_path\":\"" << json_escape(oracle_logits_path) << "\",";
    std::cout << "\"model_path\":\"" << json_escape(model_path) << "\",";
    std::cout << "\"native_logits_vector\":";
    write_value_stats(logits_stats);
    std::cout << ",";
    std::cout << "\"native_topk\":";
    write_i32_vector(native_topk);
    std::cout << ",";
    std::cout << "\"native_weights_norm_vector\":";
    write_value_stats(weights_norm_stats);
    std::cout << ",";
    std::cout << "\"native_weights_vector\":";
    write_value_stats(weights_stats);
    std::cout << ",";
    std::cout << "\"oracle_logits_vector\":";
    write_value_stats(oracle_logits_stats);
    std::cout << ",";
    std::cout << "\"oracle_topk\":";
    write_i32_vector(oracle_topk);
    std::cout << ",";
    std::cout << "\"oracle_topk_payload_path\":\"" << json_escape(oracle_topk_path) << "\",";
    std::cout << "\"oracle_weights_norm_payload_path\":\""
              << json_escape(oracle_weights_norm_path) << "\",";
    std::cout << "\"oracle_weights_norm_vector\":";
    write_value_stats(oracle_weights_norm_stats);
    std::cout << ",";
    std::cout << "\"oracle_weights_payload_path\":\""
              << json_escape(oracle_weights_path) << "\",";
    std::cout << "\"oracle_weights_vector\":";
    write_value_stats(oracle_weights_stats);
    std::cout << ",";
    std::cout << "\"passed\":" << (passed ? "true" : "false") << ",";
    std::cout << "\"schema_version\":\"intel-qwen36-engine-router-topk-compare-v0\",";
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
    std::cout << "\"logits_max_abs_diff\":" << kLogitsMaxAbsDiffThreshold << ",";
    std::cout << "\"logits_mismatch_abs_diff\":" << kLogitsMismatchThreshold << ",";
    std::cout << "\"logits_rmse\":" << kLogitsRmseThreshold << ",";
    std::cout << "\"min_cosine\":" << kMinCosine << ",";
    std::cout << "\"weights_max_abs_diff\":" << kWeightsMaxAbsDiffThreshold << ",";
    std::cout << "\"weights_mismatch_abs_diff\":" << kWeightsMismatchThreshold << ",";
    std::cout << "\"weights_rmse\":" << kWeightsRmseThreshold;
    std::cout << "}";
    std::cout << "}\n";
    return passed ? 0 : 2;
  } catch (const std::exception& exc) {
    std::cerr << "iq36-router-topk-compare: " << exc.what() << "\n";
    return 1;
  }
}
