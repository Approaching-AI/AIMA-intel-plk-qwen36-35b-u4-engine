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

constexpr int kVocabSize = 248320;
constexpr int kTopK = 8;
constexpr double kLogitAbsDiffThreshold = 5e-4;

struct ValueStats {
  std::uint64_t count = 0;
  double min = 0.0;
  double max = 0.0;
  double abs_sum = 0.0;
  double l2 = 0.0;
  bool finite = false;
  bool nonzero = false;
};

struct TopKRow {
  std::int32_t token_id = 0;
  float logit = 0.0f;
};

struct SamplerTopKJson {
  std::string case_id;
  int source_token_position = -1;
  int prompt_token_count = 0;
  bool logits_present = false;
  std::vector<TopKRow> top_k;
};

struct TopKCompareStats {
  std::uint64_t compared_count = 0;
  std::uint64_t token_id_mismatch_count = 0;
  double max_abs_logit_diff = 0.0;
  double mean_abs_logit_diff = 0.0;
  bool same_size = false;
  bool finite = false;
  bool logits_within_threshold = false;
  bool top1_matches = false;
};

void require(bool ok, const char* message) {
  if (!ok) {
    throw std::runtime_error(message);
  }
}

std::string read_text_file(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::invalid_argument("text file could not be opened");
  }
  return std::string(
      std::istreambuf_iterator<char>(input),
      std::istreambuf_iterator<char>());
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

std::string extract_string(const std::string& json,
                           const std::string& key) {
  const std::regex pattern(
      "\"" + key + "\"\\s*:\\s*\"([^\"]*)\"");
  std::smatch match;
  if (!std::regex_search(json, match, pattern)) {
    throw std::runtime_error("sampler JSON missing string key: " + key);
  }
  return match[1].str();
}

int extract_int(const std::string& json, const std::string& key) {
  const std::regex pattern("\"" + key + "\"\\s*:\\s*(-?[0-9]+)");
  std::smatch match;
  if (!std::regex_search(json, match, pattern)) {
    throw std::runtime_error("sampler JSON missing int key: " + key);
  }
  return std::stoi(match[1].str());
}

bool extract_bool(const std::string& json, const std::string& key) {
  const std::regex pattern("\"" + key + "\"\\s*:\\s*(true|false)");
  std::smatch match;
  if (!std::regex_search(json, match, pattern)) {
    throw std::runtime_error("sampler JSON missing bool key: " + key);
  }
  return match[1].str() == "true";
}

SamplerTopKJson read_sampler_topk_json(const std::string& path) {
  const auto json = read_text_file(path);
  SamplerTopKJson result;
  result.case_id = extract_string(json, "case_id");
  result.source_token_position = extract_int(json, "source_token_position");
  result.prompt_token_count = extract_int(json, "prompt_token_count");
  result.logits_present = extract_bool(json, "logits_present");

  const std::regex row_pattern(
      "\\{\\s*\"token_id\"\\s*:\\s*(-?[0-9]+)\\s*,\\s*\"logit\"\\s*:\\s*([-+0-9.eE]+)\\s*\\}");
  auto begin = std::sregex_iterator(json.begin(), json.end(), row_pattern);
  auto end = std::sregex_iterator();
  for (auto iter = begin; iter != end; ++iter) {
    TopKRow row;
    row.token_id = std::stoi((*iter)[1].str());
    row.logit = std::stof((*iter)[2].str());
    result.top_k.push_back(row);
  }
  if (result.top_k.empty()) {
    throw std::runtime_error("sampler JSON top_k is empty");
  }
  return result;
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

std::vector<TopKRow> top_k_logits(const std::vector<float>& logits, int k) {
  std::vector<TopKRow> rows;
  rows.reserve(logits.size());
  for (std::size_t i = 0; i < logits.size(); ++i) {
    rows.push_back(TopKRow{static_cast<std::int32_t>(i), logits[i]});
  }
  const auto kk = static_cast<std::size_t>(std::min<int>(k, rows.size()));
  std::partial_sort(
      rows.begin(),
      rows.begin() + static_cast<std::ptrdiff_t>(kk),
      rows.end(),
      [](const TopKRow& lhs, const TopKRow& rhs) {
        return lhs.logit > rhs.logit;
      });
  rows.resize(kk);
  return rows;
}

TopKCompareStats compare_top_k(const std::vector<TopKRow>& native,
                               const std::vector<TopKRow>& expected) {
  TopKCompareStats stats;
  stats.same_size = native.size() == expected.size();
  stats.compared_count = std::min(native.size(), expected.size());
  stats.finite = stats.compared_count > 0;
  double diff_sum = 0.0;
  for (std::size_t i = 0; i < stats.compared_count; ++i) {
    if (native[i].token_id != expected[i].token_id) {
      ++stats.token_id_mismatch_count;
    }
    if (!std::isfinite(native[i].logit) || !std::isfinite(expected[i].logit)) {
      stats.finite = false;
      continue;
    }
    const double diff = std::abs(
        static_cast<double>(native[i].logit) -
        static_cast<double>(expected[i].logit));
    stats.max_abs_logit_diff = std::max(stats.max_abs_logit_diff, diff);
    diff_sum += diff;
  }
  if (stats.compared_count > 0) {
    stats.mean_abs_logit_diff = diff_sum /
                                static_cast<double>(stats.compared_count);
    stats.top1_matches = native[0].token_id == expected[0].token_id;
  }
  stats.logits_within_threshold =
      stats.finite && stats.max_abs_logit_diff <= kLogitAbsDiffThreshold;
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

void write_compare_stats(const TopKCompareStats& stats) {
  std::cout << "{";
  std::cout << "\"compared_count\":" << stats.compared_count << ",";
  std::cout << "\"finite\":" << (stats.finite ? "true" : "false") << ",";
  std::cout << "\"logits_within_threshold\":"
            << (stats.logits_within_threshold ? "true" : "false") << ",";
  std::cout << "\"max_abs_logit_diff\":" << stats.max_abs_logit_diff << ",";
  std::cout << "\"mean_abs_logit_diff\":" << stats.mean_abs_logit_diff << ",";
  std::cout << "\"same_size\":" << (stats.same_size ? "true" : "false") << ",";
  std::cout << "\"token_id_mismatch_count\":"
            << stats.token_id_mismatch_count << ",";
  std::cout << "\"top1_matches\":" << (stats.top1_matches ? "true" : "false");
  std::cout << "}";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    require(argc == 4,
            "usage: iq36-sampler-compare <model.gguf> <oracle-logits-f32> <sampler-topk-json>");
    const std::string model_path = argv[1];
    const std::string logits_path = argv[2];
    const std::string sampler_topk_path = argv[3];

    const auto index = iq36::parse_gguf_model_index(model_path);
    const auto load_map = iq36::validate_qwen36_load_map(index);
    const auto logits = iq36::read_f32_vector_file(logits_path);
    const auto expected = read_sampler_topk_json(sampler_topk_path);
    const auto native_top_k = top_k_logits(logits, kTopK);
    const auto compare = compare_top_k(native_top_k, expected.top_k);
    const auto logits_stats = stats_from_values(logits);

    const bool passed =
        load_map.ready &&
        logits_stats.count == kVocabSize &&
        logits_stats.finite &&
        logits_stats.nonzero &&
        expected.case_id == "short_math_001" &&
        expected.source_token_position == 15 &&
        expected.prompt_token_count == 16 &&
        expected.logits_present &&
        expected.top_k.size() == kTopK &&
        native_top_k.size() == kTopK &&
        compare.same_size &&
        compare.finite &&
        compare.top1_matches &&
        compare.token_id_mismatch_count == 0 &&
        compare.logits_within_threshold;

    std::cout << std::setprecision(17);
    std::cout << "{";
    std::cout << "\"comparison\":";
    write_compare_stats(compare);
    std::cout << ",";
    std::cout << "\"expected_top_k\":";
    write_top_k_rows(expected.top_k);
    std::cout << ",";
    std::cout << "\"load_map_ready\":" << (load_map.ready ? "true" : "false") << ",";
    std::cout << "\"logits_payload_path\":\"" << json_escape(logits_path) << "\",";
    std::cout << "\"logits_vector\":";
    write_value_stats(logits_stats);
    std::cout << ",";
    std::cout << "\"model_path\":\"" << json_escape(model_path) << "\",";
    std::cout << "\"native_top_k\":";
    write_top_k_rows(native_top_k);
    std::cout << ",";
    std::cout << "\"passed\":" << (passed ? "true" : "false") << ",";
    std::cout << "\"sampler_json\":{";
    std::cout << "\"case_id\":\"" << json_escape(expected.case_id) << "\",";
    std::cout << "\"logits_present\":"
              << (expected.logits_present ? "true" : "false") << ",";
    std::cout << "\"prompt_token_count\":" << expected.prompt_token_count << ",";
    std::cout << "\"source_token_position\":"
              << expected.source_token_position;
    std::cout << "},";
    std::cout << "\"sampler_topk_path\":\""
              << json_escape(sampler_topk_path) << "\",";
    std::cout << "\"schema_version\":\"intel-qwen36-engine-sampler-compare-v0\",";
    std::cout << "\"thresholds\":{";
    std::cout << "\"logit_abs_diff\":" << kLogitAbsDiffThreshold;
    std::cout << "}";
    std::cout << "}\n";
    return passed ? 0 : 2;
  } catch (const std::exception& exc) {
    std::cerr << "iq36-sampler-compare: " << exc.what() << "\n";
    return 1;
  }
}
