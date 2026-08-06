#include "intel_qwen36/gguf_loader.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <regex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr int kLayerCount = 40;
constexpr int kHiddenSize = 2048;
constexpr int kProjectionInputSize = 4096;
constexpr int kVocabSize = 248320;
constexpr int kTopK = 8;
constexpr double kVectorMismatchThreshold = 5e-3;
constexpr double kVectorMaxAbsDiffThreshold = 5e-3;
constexpr double kVectorRmseThreshold = 5e-4;
constexpr double kFinalNormMismatchThreshold = 2e-5;
constexpr double kFinalNormMaxAbsDiffThreshold = 2e-5;
constexpr double kFinalNormRmseThreshold = 2e-6;
constexpr double kLogitsMismatchThreshold = 5e-3;
constexpr double kLogitsMaxAbsDiffThreshold = 5e-3;
constexpr double kLogitsRmseThreshold = 1e-3;
constexpr double kMinCosine = 0.999;

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

struct TopKRow {
  std::int32_t token_id = 0;
  float logit = 0.0f;
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

std::string two_digit(int value) {
  if (value < 0 || value >= 100) {
    throw std::runtime_error("two_digit value out of range");
  }
  std::string out;
  out.push_back(static_cast<char>('0' + (value / 10)));
  out.push_back(static_cast<char>('0' + (value % 10)));
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

std::vector<std::int32_t> read_i32_vector_file(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("failed to open i32 vector: " + path);
  }
  input.seekg(0, std::ios::end);
  const auto bytes = input.tellg();
  if (bytes < 0 || bytes % static_cast<std::streamoff>(sizeof(std::int32_t)) != 0) {
    throw std::runtime_error("i32 vector byte size mismatch: " + path);
  }
  input.seekg(0, std::ios::beg);
  std::vector<std::int32_t> values(
      static_cast<std::size_t>(bytes / static_cast<std::streamoff>(sizeof(std::int32_t))));
  input.read(reinterpret_cast<char*>(values.data()), bytes);
  if (!input) {
    throw std::runtime_error("failed to read i32 vector: " + path);
  }
  return values;
}

std::string read_text_file(const std::string& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("failed to open text file: " + path);
  }
  return std::string(
      std::istreambuf_iterator<char>(input),
      std::istreambuf_iterator<char>());
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
  stats.same_size = lhs.size() == rhs.size();
  stats.compared_value_count = std::min(lhs.size(), rhs.size());
  for (std::size_t i = 0; i < stats.compared_value_count; ++i) {
    if (lhs[i] != rhs[i]) {
      ++stats.mismatch_count;
    }
  }
  if (!stats.same_size) {
    stats.mismatch_count +=
        static_cast<std::uint64_t>(std::max(lhs.size(), rhs.size()) -
                                   stats.compared_value_count);
  }
  return stats;
}

bool comparison_passed(const iq36::VectorCompareStats& stats,
                       double max_abs_diff,
                       double rmse) {
  return stats.same_size &&
         stats.finite &&
         stats.mismatch_count == 0 &&
         stats.max_abs_diff <= max_abs_diff &&
         stats.rmse <= rmse &&
         stats.cosine >= kMinCosine;
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

std::vector<TopKRow> top_k_logits(const std::vector<float>& logits, int k) {
  std::vector<std::int32_t> indexes(logits.size());
  for (std::size_t i = 0; i < logits.size(); ++i) {
    indexes[i] = static_cast<std::int32_t>(i);
  }
  const auto limit = std::min<std::size_t>(static_cast<std::size_t>(k), indexes.size());
  std::partial_sort(
      indexes.begin(),
      indexes.begin() + static_cast<std::ptrdiff_t>(limit),
      indexes.end(),
      [&logits](std::int32_t lhs, std::int32_t rhs) {
        const float lhs_value = logits[static_cast<std::size_t>(lhs)];
        const float rhs_value = logits[static_cast<std::size_t>(rhs)];
        if (lhs_value == rhs_value) {
          return lhs < rhs;
        }
        return lhs_value > rhs_value;
      });
  std::vector<TopKRow> rows;
  rows.reserve(limit);
  for (std::size_t i = 0; i < limit; ++i) {
    const auto token_id = indexes[i];
    rows.push_back({token_id, logits[static_cast<std::size_t>(token_id)]});
  }
  return rows;
}

std::vector<TopKRow> read_sampler_topk_json(const std::string& path) {
  const std::string text = read_text_file(path);
  std::regex row_regex(
      "\\{\\s*\"token_id\"\\s*:\\s*(-?[0-9]+)\\s*,\\s*\"logit\"\\s*:\\s*(-?[0-9]+(?:\\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)");
  std::vector<TopKRow> rows;
  for (auto it = std::sregex_iterator(text.begin(), text.end(), row_regex);
       it != std::sregex_iterator();
       ++it) {
    rows.push_back({
        static_cast<std::int32_t>(std::stoi((*it)[1].str())),
        std::stof((*it)[2].str()),
    });
  }
  if (rows.empty()) {
    throw std::runtime_error("sampler JSON top_k rows missing");
  }
  return rows;
}

void write_top_k_rows(const std::vector<TopKRow>& rows) {
  std::cout << "[";
  for (std::size_t i = 0; i < rows.size(); ++i) {
    if (i != 0) {
      std::cout << ",";
    }
    std::cout << "{";
    std::cout << "\"logit\":" << rows[i].logit << ",";
    std::cout << "\"token_id\":" << rows[i].token_id;
    std::cout << "}";
  }
  std::cout << "]";
}

IntCompareStats compare_top_k_tokens(const std::vector<TopKRow>& lhs,
                                     const std::vector<TopKRow>& rhs) {
  std::vector<std::int32_t> lhs_ids;
  std::vector<std::int32_t> rhs_ids;
  lhs_ids.reserve(lhs.size());
  rhs_ids.reserve(rhs.size());
  for (const auto& row : lhs) {
    lhs_ids.push_back(row.token_id);
  }
  for (const auto& row : rhs) {
    rhs_ids.push_back(row.token_id);
  }
  return compare_i32_vectors(lhs_ids, rhs_ids);
}

std::filesystem::path layer_path(const std::filesystem::path& dir,
                                 const std::string& prefix,
                                 int layer) {
  return dir / (prefix + "_" + two_digit(layer) + ".bin");
}

}  // namespace

int main(int argc, char** argv) {
  try {
    require(argc == 4 || argc == 5,
            "usage: iq36-loop-shell-compare <model.gguf> <payload-dir> <sampler-topk-json> [--teacher-forced-residuals]");
    const std::string model_path = argv[1];
    const std::filesystem::path payload_dir = argv[2];
    const std::string sampler_topk_path = argv[3];
    const bool teacher_forced_residuals =
        argc == 5 && std::string(argv[4]) == "--teacher-forced-residuals";
    if (argc == 5 && !teacher_forced_residuals) {
      throw std::runtime_error("unsupported loop shell mode flag");
    }

    const auto index = iq36::parse_gguf_model_index(model_path);
    const auto load_map = iq36::validate_qwen36_load_map(index);
    const float epsilon = metadata_float(
        index,
        "qwen35moe.attention.layer_norm_rms_epsilon",
        1e-6f);

    std::vector<std::vector<float>> attention_projection_inputs;
    attention_projection_inputs.reserve(kLayerCount);
    for (int layer = 0; layer < kLayerCount; ++layer) {
      attention_projection_inputs.push_back(iq36::read_f32_vector_file(
          layer_path(payload_dir, "attention_projection_input", layer).string()));
    }
    const auto residual_input =
        iq36::read_f32_vector_file((payload_dir / "residual_input.bin").string());
    std::vector<iq36::Qwen36LayerShellResult> native_layers;
    std::vector<float> native_final_norm;
    std::vector<float> native_logits;
    if (teacher_forced_residuals) {
      native_layers.reserve(kLayerCount);
      for (int layer = 0; layer < kLayerCount; ++layer) {
        const auto layer_residual_input = iq36::read_f32_vector_file(
            layer_path(payload_dir, "residual_input", layer).string());
        native_layers.push_back(iq36::run_qwen36_layer_with_external_attention_state(
            model_path,
            index,
            layer,
            layer_residual_input,
            attention_projection_inputs[static_cast<std::size_t>(layer)],
            epsilon));
      }
      const auto norm_weight =
          iq36::decode_tensor_row(model_path, index, "output_norm.weight", 0);
      native_final_norm = iq36::apply_rms_norm(
          native_layers.back().residual, norm_weight, epsilon);
      native_logits =
          iq36::matvec_tensor(model_path, index, "output.weight", native_final_norm);
    } else {
      const auto result = iq36::run_qwen36_loop_with_external_attention_states(
          model_path, index, residual_input, attention_projection_inputs, epsilon);
      native_layers = std::move(result.layers);
      native_final_norm = std::move(result.final_norm);
      native_logits = std::move(result.logits);
    }

    std::vector<std::vector<float>> oracle_attention_outputs;
    std::vector<std::vector<float>> oracle_attention_residuals;
    std::vector<std::vector<float>> oracle_layer_outputs;
    std::vector<std::vector<std::int32_t>> oracle_topks;
    oracle_attention_outputs.reserve(kLayerCount);
    oracle_attention_residuals.reserve(kLayerCount);
    oracle_layer_outputs.reserve(kLayerCount);
    oracle_topks.reserve(kLayerCount);
    for (int layer = 0; layer < kLayerCount; ++layer) {
      oracle_attention_outputs.push_back(iq36::read_f32_vector_file(
          layer_path(payload_dir, "attention_output", layer).string()));
      oracle_attention_residuals.push_back(iq36::read_f32_vector_file(
          layer_path(payload_dir, "attention_residual", layer).string()));
      oracle_layer_outputs.push_back(iq36::read_f32_vector_file(
          layer_path(payload_dir, "layer_output", layer).string()));
      oracle_topks.push_back(read_i32_vector_file(
          layer_path(payload_dir, "topk", layer).string()));
    }
    const auto oracle_final_norm =
        iq36::read_f32_vector_file((payload_dir / "result_norm.bin").string());
    const auto oracle_logits =
        iq36::read_f32_vector_file((payload_dir / "result_output.bin").string());
    const auto expected_sampler_topk = read_sampler_topk_json(sampler_topk_path);
    const auto native_sampler_topk = top_k_logits(native_logits, kTopK);

    require(native_layers.size() == kLayerCount, "loop layer count mismatch");
    bool vectors_ok = stats_from_values(residual_input).count == kHiddenSize;
    bool passed = load_map.ready && vectors_ok;
    std::uint64_t topk_mismatch_total = 0;
    double max_attention_output_abs_diff = 0.0;
    double max_attention_residual_abs_diff = 0.0;
    double max_layer_output_abs_diff = 0.0;

    std::vector<iq36::VectorCompareStats> attention_output_compares;
    std::vector<iq36::VectorCompareStats> attention_residual_compares;
    std::vector<iq36::VectorCompareStats> layer_output_compares;
    std::vector<IntCompareStats> topk_compares;
    attention_output_compares.reserve(kLayerCount);
    attention_residual_compares.reserve(kLayerCount);
    layer_output_compares.reserve(kLayerCount);
    topk_compares.reserve(kLayerCount);

    for (int layer = 0; layer < kLayerCount; ++layer) {
      const auto& layer_result = native_layers[static_cast<std::size_t>(layer)];
      attention_output_compares.push_back(iq36::compare_vectors(
          layer_result.attention_output,
          oracle_attention_outputs[static_cast<std::size_t>(layer)],
          kVectorMismatchThreshold));
      attention_residual_compares.push_back(iq36::compare_vectors(
          layer_result.attention_residual,
          oracle_attention_residuals[static_cast<std::size_t>(layer)],
          kVectorMismatchThreshold));
      layer_output_compares.push_back(iq36::compare_vectors(
          layer_result.residual,
          oracle_layer_outputs[static_cast<std::size_t>(layer)],
          kVectorMismatchThreshold));
      topk_compares.push_back(compare_i32_vectors(
          layer_result.ffn.router.expert_ids,
          oracle_topks[static_cast<std::size_t>(layer)]));

      const auto& attention_output_compare =
          attention_output_compares[static_cast<std::size_t>(layer)];
      const auto& attention_residual_compare =
          attention_residual_compares[static_cast<std::size_t>(layer)];
      const auto& layer_output_compare =
          layer_output_compares[static_cast<std::size_t>(layer)];
      const auto& topk_compare = topk_compares[static_cast<std::size_t>(layer)];
      max_attention_output_abs_diff = std::max(
          max_attention_output_abs_diff, attention_output_compare.max_abs_diff);
      max_attention_residual_abs_diff = std::max(
          max_attention_residual_abs_diff, attention_residual_compare.max_abs_diff);
      max_layer_output_abs_diff = std::max(
          max_layer_output_abs_diff, layer_output_compare.max_abs_diff);
      topk_mismatch_total += topk_compare.mismatch_count;
      passed = passed &&
               attention_projection_inputs[static_cast<std::size_t>(layer)].size() ==
                   kProjectionInputSize &&
               layer_result.attention_output.size() == kHiddenSize &&
               layer_result.attention_residual.size() == kHiddenSize &&
               layer_result.residual.size() == kHiddenSize &&
               comparison_passed(
                   attention_output_compare,
                   kVectorMaxAbsDiffThreshold,
                   kVectorRmseThreshold) &&
               comparison_passed(
                   attention_residual_compare,
                   kVectorMaxAbsDiffThreshold,
                   kVectorRmseThreshold) &&
               comparison_passed(
                   layer_output_compare,
                   kVectorMaxAbsDiffThreshold,
                   kVectorRmseThreshold) &&
               topk_compare.same_size &&
               topk_compare.mismatch_count == 0;
    }

    const auto final_norm_compare = iq36::compare_vectors(
        native_final_norm, oracle_final_norm, kFinalNormMismatchThreshold);
    const auto logits_compare = iq36::compare_vectors(
        native_logits, oracle_logits, kLogitsMismatchThreshold);
    const auto sampler_compare =
        compare_top_k_tokens(native_sampler_topk, expected_sampler_topk);
    passed = passed &&
             native_final_norm.size() == kHiddenSize &&
             native_logits.size() == kVocabSize &&
             comparison_passed(
                 final_norm_compare,
                 kFinalNormMaxAbsDiffThreshold,
                 kFinalNormRmseThreshold) &&
             comparison_passed(
                 logits_compare,
                 kLogitsMaxAbsDiffThreshold,
                 kLogitsRmseThreshold) &&
             sampler_compare.same_size &&
             sampler_compare.mismatch_count == 0;

    std::cout << std::setprecision(17);
    std::cout << "{";
    std::cout << "\"epsilon\":" << epsilon << ",";
    std::cout << "\"expected_sampler_topk\":";
    write_top_k_rows(expected_sampler_topk);
    std::cout << ",";
    std::cout << "\"final_norm_comparison\":";
    write_compare_stats(final_norm_compare);
    std::cout << ",";
    std::cout << "\"final_norm_vector\":";
    write_value_stats(stats_from_values(native_final_norm));
    std::cout << ",";
    std::cout << "\"layer_count\":" << kLayerCount << ",";
    std::cout << "\"layers\":[";
    for (int layer = 0; layer < kLayerCount; ++layer) {
      if (layer != 0) {
        std::cout << ",";
      }
      const auto& layer_result = native_layers[static_cast<std::size_t>(layer)];
      std::cout << "{";
      std::cout << "\"attention_output_comparison\":";
      write_compare_stats(attention_output_compares[static_cast<std::size_t>(layer)]);
      std::cout << ",";
      std::cout << "\"attention_residual_comparison\":";
      write_compare_stats(attention_residual_compares[static_cast<std::size_t>(layer)]);
      std::cout << ",";
      std::cout << "\"layer_index\":" << layer << ",";
      std::cout << "\"layer_output_comparison\":";
      write_compare_stats(layer_output_compares[static_cast<std::size_t>(layer)]);
      std::cout << ",";
      std::cout << "\"native_topk\":";
      write_i32_vector(layer_result.ffn.router.expert_ids);
      std::cout << ",";
      std::cout << "\"oracle_topk\":";
      write_i32_vector(oracle_topks[static_cast<std::size_t>(layer)]);
      std::cout << ",";
      std::cout << "\"projection_input_vector\":";
      write_value_stats(stats_from_values(
          attention_projection_inputs[static_cast<std::size_t>(layer)]));
      std::cout << ",";
      std::cout << "\"topk_comparison\":";
      write_int_compare_stats(topk_compares[static_cast<std::size_t>(layer)]);
      std::cout << "}";
    }
    std::cout << "],";
    std::cout << "\"load_map_ready\":" << (load_map.ready ? "true" : "false") << ",";
    std::cout << "\"logits_comparison\":";
    write_compare_stats(logits_compare);
    std::cout << ",";
    std::cout << "\"logits_vector\":";
    write_value_stats(stats_from_values(native_logits));
    std::cout << ",";
    std::cout << "\"max_attention_output_abs_diff\":"
              << max_attention_output_abs_diff << ",";
    std::cout << "\"max_attention_residual_abs_diff\":"
              << max_attention_residual_abs_diff << ",";
    std::cout << "\"max_layer_output_abs_diff\":" << max_layer_output_abs_diff << ",";
    std::cout << "\"model_path\":\"" << json_escape(model_path) << "\",";
    std::cout << "\"native_sampler_topk\":";
    write_top_k_rows(native_sampler_topk);
    std::cout << ",";
    std::cout << "\"passed\":" << (passed ? "true" : "false") << ",";
    std::cout << "\"residual_mode\":\""
              << (teacher_forced_residuals ? "teacher_forced_oracle" : "sequential_native_shell")
              << "\",";
    std::cout << "\"residual_input_vector\":";
    write_value_stats(stats_from_values(residual_input));
    std::cout << ",";
    std::cout << "\"sampler_comparison\":";
    write_int_compare_stats(sampler_compare);
    std::cout << ",";
    std::cout << "\"schema_version\":\"intel-qwen36-engine-loop-shell-compare-v0\",";
    std::cout << "\"thresholds\":{";
    std::cout << "\"final_norm_max_abs_diff\":"
              << kFinalNormMaxAbsDiffThreshold << ",";
    std::cout << "\"final_norm_mismatch_abs_diff\":"
              << kFinalNormMismatchThreshold << ",";
    std::cout << "\"final_norm_rmse\":" << kFinalNormRmseThreshold << ",";
    std::cout << "\"logits_max_abs_diff\":" << kLogitsMaxAbsDiffThreshold << ",";
    std::cout << "\"logits_mismatch_abs_diff\":"
              << kLogitsMismatchThreshold << ",";
    std::cout << "\"logits_rmse\":" << kLogitsRmseThreshold << ",";
    std::cout << "\"min_cosine\":" << kMinCosine << ",";
    std::cout << "\"vector_max_abs_diff\":" << kVectorMaxAbsDiffThreshold << ",";
    std::cout << "\"vector_mismatch_abs_diff\":"
              << kVectorMismatchThreshold << ",";
    std::cout << "\"vector_rmse\":" << kVectorRmseThreshold;
    std::cout << "},";
    std::cout << "\"topk_mismatch_total\":" << topk_mismatch_total;
    std::cout << "}\n";
    return passed ? 0 : 2;
  } catch (const std::exception& exc) {
    std::cerr << "iq36-loop-shell-compare: " << exc.what() << "\n";
    return 1;
  }
}
