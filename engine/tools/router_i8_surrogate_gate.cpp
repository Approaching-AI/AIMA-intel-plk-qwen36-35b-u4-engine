#include "intel_qwen36/gguf_loader.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr std::size_t kHiddenSize = 2048;
constexpr std::size_t kExpertCount = 256;
constexpr std::size_t kSelectedExperts = 8;
constexpr std::size_t kBlockSize = 32;

[[noreturn]] void Die(const std::string& message) {
  throw std::runtime_error(message);
}

void Require(bool condition, const std::string& message) {
  if (!condition) Die(message);
}

struct CorpusRow {
  int layer = -1;
  std::string vector_path;
  std::string oracle_logits_path;
  std::string sha256;
};

std::vector<std::string> SplitTabs(const std::string& line) {
  std::vector<std::string> fields;
  std::size_t begin = 0;
  while (begin <= line.size()) {
    const auto end = line.find('\t', begin);
    fields.push_back(line.substr(
        begin, end == std::string::npos ? std::string::npos : end - begin));
    if (end == std::string::npos) break;
    begin = end + 1;
  }
  return fields;
}

std::vector<CorpusRow> ReadManifest(const std::string& path) {
  std::ifstream input(path);
  Require(static_cast<bool>(input), "corpus manifest could not be opened");
  std::vector<CorpusRow> rows;
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty() || line[0] == '#') continue;
    const auto fields = SplitTabs(line);
    Require(fields.size() == 4, "corpus manifest row must have four fields");
    CorpusRow row;
    row.layer = std::stoi(fields[0]);
    row.vector_path = fields[1];
    row.oracle_logits_path = fields[2];
    row.sha256 = fields[3];
    Require(row.layer >= 0 && row.layer < 40, "corpus layer is out of range");
    rows.push_back(std::move(row));
  }
  Require(!rows.empty(), "corpus manifest is empty");
  return rows;
}

std::vector<float> ReadF32(const std::string& path, std::size_t expected) {
  std::ifstream input(path, std::ios::binary);
  Require(static_cast<bool>(input), "float payload could not be opened: " + path);
  input.seekg(0, std::ios::end);
  const auto size = input.tellg();
  Require(size >= 0 && static_cast<std::size_t>(size) == expected * sizeof(float),
          "float payload size mismatch: " + path);
  input.seekg(0, std::ios::beg);
  std::vector<float> values(expected);
  input.read(reinterpret_cast<char*>(values.data()),
             static_cast<std::streamsize>(values.size() * sizeof(float)));
  Require(static_cast<bool>(input), "float payload read failed: " + path);
  for (float value : values) {
    Require(std::isfinite(value), "float payload contains non-finite value: " + path);
  }
  return values;
}

std::vector<std::size_t> SortIds(const std::vector<float>& values) {
  std::vector<std::size_t> ids(values.size());
  std::iota(ids.begin(), ids.end(), 0);
  std::sort(ids.begin(), ids.end(), [&](std::size_t lhs, std::size_t rhs) {
    return values[lhs] > values[rhs] ||
           (values[lhs] == values[rhs] && lhs < rhs);
  });
  return ids;
}

std::vector<double> SelectedWeights(const std::vector<float>& logits,
                                    const std::vector<std::size_t>& ids) {
  Require(!ids.empty(), "selected expert set is empty");
  double max_value = -std::numeric_limits<double>::infinity();
  for (std::size_t id : ids) max_value = std::max(max_value, double(logits[id]));
  std::vector<double> weights;
  weights.reserve(ids.size());
  double denominator = 0.0;
  for (std::size_t id : ids) {
    const double value = std::exp(double(logits[id]) - max_value);
    weights.push_back(value);
    denominator += value;
  }
  for (double& value : weights) value /= denominator;
  return weights;
}

std::int8_t Quantize(float value, float scale) {
  if (scale == 0.0f) return 0;
  const long rounded = std::lround(double(value) / double(scale));
  return static_cast<std::int8_t>(std::max(-127L, std::min(127L, rounded)));
}

struct QuantizedInput {
  std::vector<std::int8_t> values;
  std::vector<float> scales;
};

QuantizedInput QuantizeInput(const std::vector<float>& input) {
  Require(input.size() == kHiddenSize, "router input size mismatch");
  QuantizedInput result;
  result.values.resize(input.size());
  result.scales.resize(kHiddenSize / kBlockSize);
  for (std::size_t block = 0; block < result.scales.size(); ++block) {
    float maximum = 0.0f;
    for (std::size_t i = 0; i < kBlockSize; ++i) {
      maximum = std::max(maximum, std::abs(input[block * kBlockSize + i]));
    }
    const float scale = maximum == 0.0f ? 0.0f : maximum / 127.0f;
    result.scales[block] = scale;
    for (std::size_t i = 0; i < kBlockSize; ++i) {
      const auto index = block * kBlockSize + i;
      result.values[index] = Quantize(input[index], scale);
    }
  }
  return result;
}

enum class ScaleGranularity { kRow, kBlock32 };

struct QuantizedRouter {
  ScaleGranularity granularity = ScaleGranularity::kRow;
  std::vector<std::int8_t> values;
  std::vector<float> scales;
  std::uint64_t resident_bytes = 0;
};

QuantizedRouter QuantizeRouter(const std::vector<float>& weights,
                               ScaleGranularity granularity) {
  Require(weights.size() == kExpertCount * kHiddenSize,
          "router weight size mismatch");
  QuantizedRouter result;
  result.granularity = granularity;
  result.values.resize(weights.size());
  const std::size_t scales_per_row =
      granularity == ScaleGranularity::kRow ? 1 : kHiddenSize / kBlockSize;
  result.scales.resize(kExpertCount * scales_per_row);
  for (std::size_t row = 0; row < kExpertCount; ++row) {
    for (std::size_t scale_index = 0; scale_index < scales_per_row;
         ++scale_index) {
      const std::size_t begin =
          row * kHiddenSize +
          (granularity == ScaleGranularity::kRow ? 0 : scale_index * kBlockSize);
      const std::size_t count =
          granularity == ScaleGranularity::kRow ? kHiddenSize : kBlockSize;
      float maximum = 0.0f;
      for (std::size_t i = 0; i < count; ++i) {
        maximum = std::max(maximum, std::abs(weights[begin + i]));
      }
      const float scale = maximum == 0.0f ? 0.0f : maximum / 127.0f;
      result.scales[row * scales_per_row + scale_index] = scale;
      for (std::size_t i = 0; i < count; ++i) {
        result.values[begin + i] = Quantize(weights[begin + i], scale);
      }
    }
  }
  result.resident_bytes = result.values.size() + result.scales.size() * 4ULL;
  return result;
}

std::vector<float> SurrogateLogits(const QuantizedRouter& router,
                                   const QuantizedInput& input) {
  const std::size_t input_blocks = kHiddenSize / kBlockSize;
  const std::size_t weight_scales_per_row =
      router.granularity == ScaleGranularity::kRow ? 1 : input_blocks;
  std::vector<float> logits(kExpertCount, 0.0f);
  for (std::size_t row = 0; row < kExpertCount; ++row) {
    float sum = 0.0f;
    for (std::size_t block = 0; block < input_blocks; ++block) {
      std::int32_t dot = 0;
      const auto base = row * kHiddenSize + block * kBlockSize;
      const auto input_base = block * kBlockSize;
      for (std::size_t i = 0; i < kBlockSize; ++i) {
        dot += std::int32_t(router.values[base + i]) *
               std::int32_t(input.values[input_base + i]);
      }
      const auto scale_index =
          row * weight_scales_per_row +
          (router.granularity == ScaleGranularity::kRow ? 0 : block);
      sum = std::fma(router.scales[scale_index] * input.scales[block],
                     float(dot), sum);
    }
    logits[row] = sum;
  }
  return logits;
}

struct SchemeStats {
  std::string name;
  std::uint64_t resident_bytes_per_layer = 0;
  std::size_t maximum_true_top8_rank = 0;
  std::map<std::size_t, std::size_t> top8_match_rows_by_cap;
  std::map<std::size_t, double> maximum_weight_abs_diff_by_cap;
  double maximum_surrogate_logit_abs_diff = 0.0;
  std::string worst_sha256;
  int worst_layer = -1;
};

void WriteScheme(const SchemeStats& stats, std::size_t row_count) {
  std::cout << "{";
  std::cout << "\"name\":\"" << stats.name << "\",";
  std::cout << "\"resident_bytes_per_layer\":"
            << stats.resident_bytes_per_layer << ",";
  std::cout << "\"maximum_true_top8_rank\":"
            << stats.maximum_true_top8_rank << ",";
  std::cout << "\"maximum_surrogate_logit_abs_diff\":"
            << stats.maximum_surrogate_logit_abs_diff << ",";
  std::cout << "\"worst_layer\":" << stats.worst_layer << ",";
  std::cout << "\"worst_sha256\":\"" << stats.worst_sha256 << "\",";
  std::cout << "\"caps\":{";
  bool first = true;
  for (const auto& item : stats.top8_match_rows_by_cap) {
    if (!first) std::cout << ",";
    first = false;
    std::cout << "\"" << item.first << "\":{";
    std::cout << "\"top8_match_count\":" << item.second << ",";
    std::cout << "\"top8_match_rate\":"
              << double(item.second) / double(row_count) << ",";
    std::cout << "\"maximum_normalized_weight_abs_diff\":"
              << stats.maximum_weight_abs_diff_by_cap.at(item.first);
    std::cout << "}";
  }
  std::cout << "}}";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Require(argc == 3,
            "usage: router_i8_surrogate_gate <model.gguf> <corpus.tsv>");
    const std::string model_path = argv[1];
    const auto corpus = ReadManifest(argv[2]);
    const auto index = iq36::parse_gguf_model_index(model_path);
    const auto load_map = iq36::validate_qwen36_load_map(index);
    Require(load_map.ready, "locked model load map is not ready");
    iq36::set_resident_tensor_cache_enabled(true);
    iq36::set_dense_matvec_enabled(true);
    iq36::set_dense_matvec_min_rows(1);
    iq36::set_dense_matvec_thread_count(16);
    iq36::set_dense_matvec_payload_cache_enabled(true);

    const std::vector<std::size_t> caps{16, 32, 64, 128};
    std::vector<SchemeStats> stats{
        {"row_i8_x_block32_i8"}, {"block32_i8_x_block32_i8"}};
    for (auto& scheme : stats) {
      for (std::size_t cap : caps) {
        scheme.top8_match_rows_by_cap[cap] = 0;
        scheme.maximum_weight_abs_diff_by_cap[cap] = 0.0;
      }
    }
    std::map<std::pair<int, int>, QuantizedRouter> quantized;
    std::size_t oracle_rows = 0;
    std::size_t oracle_top8_matches = 0;
    double oracle_max_abs_diff = 0.0;
    std::map<int, std::size_t> rows_by_layer;

    for (const auto& row : corpus) {
      const std::string tensor_name =
          "blk." + std::to_string(row.layer) + ".ffn_gate_inp.weight";
      const auto* tensor = iq36::find_tensor(index, tensor_name);
      Require(tensor != nullptr && tensor->type == 0 &&
                  tensor->dims == std::vector<std::uint64_t>{2048, 256},
              "router tensor contract mismatch: " + tensor_name);
      const auto input = ReadF32(row.vector_path, kHiddenSize);
      const auto exact = iq36::matvec_tensor(model_path, index, tensor_name, input);
      Require(exact.size() == kExpertCount, "exact router logit size mismatch");
      const auto exact_order = SortIds(exact);
      std::vector<std::size_t> exact_top8(exact_order.begin(),
                                          exact_order.begin() + kSelectedExperts);
      const auto exact_weights = SelectedWeights(exact, exact_top8);
      ++rows_by_layer[row.layer];

      if (row.oracle_logits_path != "-") {
        const auto oracle = ReadF32(row.oracle_logits_path, kExpertCount);
        const auto oracle_order = SortIds(oracle);
        ++oracle_rows;
        if (std::equal(exact_top8.begin(), exact_top8.end(),
                       oracle_order.begin())) {
          ++oracle_top8_matches;
        }
        for (std::size_t i = 0; i < kExpertCount; ++i) {
          oracle_max_abs_diff =
              std::max(oracle_max_abs_diff, std::abs(double(exact[i]) - oracle[i]));
        }
      }

      std::vector<float> decoded_weights;
      for (int scheme_index = 0; scheme_index < 2; ++scheme_index) {
        const auto key = std::make_pair(row.layer, scheme_index);
        auto found = quantized.find(key);
        if (found == quantized.end()) {
          if (decoded_weights.empty()) {
            decoded_weights.reserve(kExpertCount * kHiddenSize);
            for (std::size_t expert = 0; expert < kExpertCount; ++expert) {
              const auto values = iq36::decode_tensor_row(
                  model_path, index, tensor_name, expert);
              Require(values.size() == kHiddenSize,
                      "decoded router row size mismatch");
              decoded_weights.insert(decoded_weights.end(), values.begin(), values.end());
            }
          }
          const auto granularity = scheme_index == 0
                                       ? ScaleGranularity::kRow
                                       : ScaleGranularity::kBlock32;
          found = quantized.emplace(key,
                                    QuantizeRouter(decoded_weights, granularity)).first;
          stats[scheme_index].resident_bytes_per_layer =
              found->second.resident_bytes;
        }
        const auto input_q = QuantizeInput(input);
        const auto surrogate = SurrogateLogits(found->second, input_q);
        const auto surrogate_order = SortIds(surrogate);
        auto& scheme = stats[scheme_index];
        for (std::size_t i = 0; i < kExpertCount; ++i) {
          const double diff = std::abs(double(exact[i]) - surrogate[i]);
          if (diff > scheme.maximum_surrogate_logit_abs_diff) {
            scheme.maximum_surrogate_logit_abs_diff = diff;
            scheme.worst_sha256 = row.sha256;
            scheme.worst_layer = row.layer;
          }
        }
        std::vector<std::size_t> inverse(kExpertCount);
        for (std::size_t rank = 0; rank < surrogate_order.size(); ++rank) {
          inverse[surrogate_order[rank]] = rank + 1;
        }
        for (std::size_t id : exact_top8) {
          scheme.maximum_true_top8_rank =
              std::max(scheme.maximum_true_top8_rank, inverse[id]);
        }
        for (std::size_t cap : caps) {
          std::vector<std::size_t> candidates(surrogate_order.begin(),
                                               surrogate_order.begin() + cap);
          std::sort(candidates.begin(), candidates.end(),
                    [&](std::size_t lhs, std::size_t rhs) {
            return exact[lhs] > exact[rhs] ||
                   (exact[lhs] == exact[rhs] && lhs < rhs);
          });
          candidates.resize(kSelectedExperts);
          if (candidates == exact_top8) {
            ++scheme.top8_match_rows_by_cap[cap];
          }
          const auto hybrid_weights = SelectedWeights(exact, candidates);
          if (candidates == exact_top8) {
            for (std::size_t i = 0; i < kSelectedExperts; ++i) {
              scheme.maximum_weight_abs_diff_by_cap[cap] = std::max(
                  scheme.maximum_weight_abs_diff_by_cap[cap],
                  std::abs(exact_weights[i] - hybrid_weights[i]));
            }
          } else {
            scheme.maximum_weight_abs_diff_by_cap[cap] =
                std::numeric_limits<double>::max();
          }
        }
      }
    }

    std::cout << std::setprecision(17);
    std::cout << "{";
    std::cout << "\"schema_version\":\"intel-qwen36-router-i8-surrogate-component-v0\",";
    std::cout << "\"corpus_row_count\":" << corpus.size() << ",";
    std::cout << "\"covered_layer_count\":" << rows_by_layer.size() << ",";
    std::cout << "\"oracle_row_count\":" << oracle_rows << ",";
    std::cout << "\"oracle_top8_match_count\":" << oracle_top8_matches << ",";
    std::cout << "\"oracle_max_abs_logit_diff\":" << oracle_max_abs_diff << ",";
    std::cout << "\"schemes\":[";
    for (std::size_t i = 0; i < stats.size(); ++i) {
      if (i) std::cout << ",";
      WriteScheme(stats[i], corpus.size());
    }
    std::cout << "]}" << std::endl;
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "router-i8-surrogate-gate: " << error.what() << "\n";
    return 1;
  }
}
