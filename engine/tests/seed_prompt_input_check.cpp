#include "intel_qwen36/gguf_loader.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

constexpr const char* kTensorName = "token_embd.weight";
constexpr int kExpectedCaseCount = 6;
constexpr int kHiddenSize = 2048;
constexpr int kVocabSize = 248320;
constexpr double kMismatchThreshold = 1e-6;
constexpr double kMaxAbsDiffThreshold = 1e-6;
constexpr double kRmseThreshold = 1e-7;
constexpr double kMinCosine = 0.999999;
constexpr std::uint64_t kFnvOffset = 14695981039346656037ull;
constexpr std::uint64_t kFnvPrime = 1099511628211ull;

struct ValueStats {
  std::uint64_t count = 0;
  double min = 0.0;
  double max = 0.0;
  double abs_sum = 0.0;
  double l2 = 0.0;
  bool finite = false;
  bool nonzero = false;
};

struct CaseSpec {
  std::string case_id;
  std::uint64_t expected_count = 0;
  std::uint64_t expected_fnv64 = 0;
  std::uint64_t expected_first = 0;
  std::uint64_t expected_last = 0;
  std::string token_file;
};

struct CaseResult {
  CaseSpec spec;
  std::vector<std::uint32_t> token_ids;
  std::uint64_t actual_fnv64 = 0;
  bool token_count_ok = false;
  bool token_hash_ok = false;
  bool first_token_ok = false;
  bool last_token_ok = false;
  bool token_ids_in_vocab = false;
  bool embeddings_ok = false;
  bool passed = false;
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

std::vector<std::string> split_tabs(const std::string& line) {
  std::vector<std::string> fields;
  std::string field;
  std::istringstream stream(line);
  while (std::getline(stream, field, '\t')) {
    fields.push_back(field);
  }
  return fields;
}

std::uint64_t parse_u64(const std::string& value) {
  std::size_t consumed = 0;
  const auto parsed = std::stoull(value, &consumed, 10);
  if (consumed != value.size()) {
    throw std::invalid_argument("invalid unsigned integer");
  }
  return parsed;
}

std::uint64_t parse_hex_u64(const std::string& value) {
  std::size_t consumed = 0;
  const auto parsed = std::stoull(value, &consumed, 16);
  if (consumed != value.size()) {
    throw std::invalid_argument("invalid hex unsigned integer");
  }
  return parsed;
}

std::string hex_u64(std::uint64_t value) {
  std::ostringstream out;
  out << std::hex << std::setfill('0') << std::setw(16) << value;
  return out.str();
}

std::uint64_t fnv64_bytes(const std::vector<unsigned char>& bytes) {
  std::uint64_t hash = kFnvOffset;
  for (const auto byte : bytes) {
    hash ^= byte;
    hash *= kFnvPrime;
  }
  return hash;
}

std::vector<CaseSpec> read_case_specs(const std::string& token_dir) {
  std::ifstream input(join_path(token_dir, "cases.tsv"));
  require(input.good(), "cases.tsv missing");
  std::vector<CaseSpec> cases;
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty()) {
      continue;
    }
    const auto fields = split_tabs(line);
    require(fields.size() == 6, "cases.tsv row must have 6 tab-separated fields");
    CaseSpec spec;
    spec.case_id = fields[0];
    spec.expected_count = parse_u64(fields[1]);
    spec.expected_fnv64 = parse_hex_u64(fields[2]);
    spec.expected_first = parse_u64(fields[3]);
    spec.expected_last = parse_u64(fields[4]);
    spec.token_file = fields[5];
    cases.push_back(spec);
  }
  return cases;
}

std::vector<std::uint32_t> read_token_file(const std::string& path,
                                           std::uint64_t* fnv64) {
  std::ifstream input(path, std::ios::binary);
  require(input.good(), "token file missing");
  const std::vector<unsigned char> bytes(
      (std::istreambuf_iterator<char>(input)),
      std::istreambuf_iterator<char>());
  require(bytes.size() % 4 == 0, "token file size is not u32-aligned");
  *fnv64 = fnv64_bytes(bytes);
  std::vector<std::uint32_t> tokens;
  tokens.reserve(bytes.size() / 4);
  for (std::size_t i = 0; i < bytes.size(); i += 4) {
    const std::uint32_t value =
        static_cast<std::uint32_t>(bytes[i]) |
        (static_cast<std::uint32_t>(bytes[i + 1]) << 8) |
        (static_cast<std::uint32_t>(bytes[i + 2]) << 16) |
        (static_cast<std::uint32_t>(bytes[i + 3]) << 24);
    tokens.push_back(value);
  }
  return tokens;
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

bool stats_ok(const ValueStats& stats) {
  return stats.count == kHiddenSize && stats.finite && stats.nonzero;
}

bool compare_passed(const iq36::VectorCompareStats& stats) {
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

void write_case_result(const CaseResult& result) {
  std::cout << "{";
  std::cout << "\"actual_fnv64\":\"" << hex_u64(result.actual_fnv64) << "\",";
  std::cout << "\"embedding_rows_decoded\":" << result.token_ids.size() << ",";
  std::cout << "\"embeddings_ok\":"
            << (result.embeddings_ok ? "true" : "false") << ",";
  std::cout << "\"expected_fnv64\":\""
            << hex_u64(result.spec.expected_fnv64) << "\",";
  std::cout << "\"first_token_id\":"
            << (result.token_ids.empty() ? 0 : result.token_ids.front()) << ",";
  std::cout << "\"first_token_ok\":"
            << (result.first_token_ok ? "true" : "false") << ",";
  std::cout << "\"last_token_id\":"
            << (result.token_ids.empty() ? 0 : result.token_ids.back()) << ",";
  std::cout << "\"last_token_ok\":"
            << (result.last_token_ok ? "true" : "false") << ",";
  std::cout << "\"passed\":" << (result.passed ? "true" : "false") << ",";
  std::cout << "\"token_count\":" << result.token_ids.size() << ",";
  std::cout << "\"token_count_ok\":"
            << (result.token_count_ok ? "true" : "false") << ",";
  std::cout << "\"token_file\":\""
            << json_escape(result.spec.token_file) << "\",";
  std::cout << "\"token_hash_ok\":"
            << (result.token_hash_ok ? "true" : "false") << ",";
  std::cout << "\"token_ids_in_vocab\":"
            << (result.token_ids_in_vocab ? "true" : "false");
  std::cout << "}";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    require(argc == 4,
            "usage: iq36-seed-prompt-input-check <model.gguf> "
            "<token-input-dir> <short-math-oracle-embedding.bin>");
    const std::string model_path = argv[1];
    const std::string token_dir = argv[2];
    const std::string short_math_oracle_path = argv[3];

    const auto index = iq36::parse_gguf_model_index(model_path);
    const auto load_map = iq36::validate_qwen36_load_map(index);
    const auto* tensor = iq36::find_tensor(index, kTensorName);
    require(tensor != nullptr, "token embedding tensor missing");
    const bool tensor_shape_ok =
        tensor->type == 12 &&
        tensor->dims == std::vector<std::uint64_t>{kHiddenSize, kVocabSize};

    const auto specs = read_case_specs(token_dir);
    const std::set<std::string> expected_case_ids = {
        "router_code_reason_002",
        "router_instruction_003",
        "router_math_reason_001",
        "short_factual_002",
        "short_math_001",
        "short_transform_003",
    };
    std::set<std::string> actual_case_ids;
    std::set<std::uint32_t> unique_token_ids;
    std::vector<CaseResult> results;
    results.reserve(specs.size());

    for (const auto& spec : specs) {
      CaseResult result;
      result.spec = spec;
      actual_case_ids.insert(spec.case_id);
      result.token_ids = read_token_file(join_path(token_dir, spec.token_file),
                                         &result.actual_fnv64);
      result.token_count_ok = result.token_ids.size() == spec.expected_count;
      result.token_hash_ok = result.actual_fnv64 == spec.expected_fnv64;
      result.first_token_ok =
          !result.token_ids.empty() && result.token_ids.front() == spec.expected_first;
      result.last_token_ok =
          !result.token_ids.empty() && result.token_ids.back() == spec.expected_last;
      result.token_ids_in_vocab =
          std::all_of(result.token_ids.begin(), result.token_ids.end(),
                      [](std::uint32_t token_id) {
                        return token_id < static_cast<std::uint32_t>(kVocabSize);
                      });
      for (const auto token_id : result.token_ids) {
        unique_token_ids.insert(token_id);
      }
      results.push_back(std::move(result));
    }

    std::unordered_map<std::uint32_t, ValueStats> embedding_stats;
    embedding_stats.reserve(unique_token_ids.size());
    bool unique_embeddings_ok = true;
    for (const auto token_id : unique_token_ids) {
      const auto embedding =
          iq36::decode_tensor_row(model_path, index, kTensorName, token_id);
      const auto stats = stats_from_values(embedding);
      unique_embeddings_ok = unique_embeddings_ok && stats_ok(stats);
      embedding_stats.emplace(token_id, stats);
    }

    for (auto& result : results) {
      result.embeddings_ok = result.token_ids_in_vocab;
      for (const auto token_id : result.token_ids) {
        const auto found = embedding_stats.find(token_id);
        result.embeddings_ok =
            result.embeddings_ok && found != embedding_stats.end() &&
            stats_ok(found->second);
      }
      result.passed =
          result.token_count_ok && result.token_hash_ok &&
          result.first_token_ok && result.last_token_ok &&
          result.token_ids_in_vocab && result.embeddings_ok;
    }

    const auto short_math = std::find_if(
        results.begin(), results.end(), [](const CaseResult& result) {
          return result.spec.case_id == "short_math_001";
        });
    require(short_math != results.end(), "short_math_001 case missing");
    require(!short_math->token_ids.empty(), "short_math_001 token list empty");
    const auto short_math_native = iq36::decode_tensor_row(
        model_path, index, kTensorName, short_math->token_ids.back());
    const auto short_math_oracle =
        iq36::read_f32_vector_file(short_math_oracle_path);
    const auto short_math_compare = iq36::compare_vectors(
        short_math_native, short_math_oracle, kMismatchThreshold);

    std::uint64_t total_prompt_tokens = 0;
    bool cases_ok = specs.size() == kExpectedCaseCount &&
                    actual_case_ids == expected_case_ids;
    for (const auto& result : results) {
      total_prompt_tokens += result.token_ids.size();
      cases_ok = cases_ok && result.passed;
    }

    const bool passed =
        load_map.ready && tensor_shape_ok && cases_ok && unique_embeddings_ok &&
        compare_passed(short_math_compare);

    std::cout << std::setprecision(17);
    std::cout << "{";
    std::cout << "\"case_count\":" << results.size() << ",";
    std::cout << "\"case_ids_ok\":"
              << (actual_case_ids == expected_case_ids ? "true" : "false")
              << ",";
    std::cout << "\"cases\":{";
    for (std::size_t i = 0; i < results.size(); ++i) {
      if (i != 0) {
        std::cout << ",";
      }
      std::cout << "\"" << json_escape(results[i].spec.case_id) << "\":";
      write_case_result(results[i]);
    }
    std::cout << "},";
    std::cout << "\"cases_ok\":" << (cases_ok ? "true" : "false") << ",";
    std::cout << "\"load_map_ready\":"
              << (load_map.ready ? "true" : "false") << ",";
    std::cout << "\"model_path\":\"" << json_escape(model_path) << "\",";
    std::cout << "\"passed\":" << (passed ? "true" : "false") << ",";
    std::cout << "\"schema_version\":\"intel-qwen36-engine-seed-prompt-input-check-v0\",";
    std::cout << "\"short_math_oracle_embedding_compare\":";
    write_compare_stats(short_math_compare);
    std::cout << ",";
    std::cout << "\"source\":\"seed_prompt_token_ids_u32le\",";
    std::cout << "\"tensor\":{";
    std::cout << "\"absolute_offset\":" << tensor->absolute_offset << ",";
    std::cout << "\"dims\":";
    write_u64_vector(tensor->dims);
    std::cout << ",\"name\":\"" << json_escape(tensor->name) << "\",";
    std::cout << "\"nbytes\":" << tensor->nbytes << ",";
    std::cout << "\"shape_ok\":" << (tensor_shape_ok ? "true" : "false")
              << ",";
    std::cout << "\"type_name\":\"" << iq36::ggml_type_name(tensor->type)
              << "\"";
    std::cout << "},";
    std::cout << "\"thresholds\":{";
    std::cout << "\"max_abs_diff\":" << kMaxAbsDiffThreshold << ",";
    std::cout << "\"min_cosine\":" << kMinCosine << ",";
    std::cout << "\"mismatch_abs_diff\":" << kMismatchThreshold << ",";
    std::cout << "\"rmse\":" << kRmseThreshold;
    std::cout << "},";
    std::cout << "\"total_prompt_tokens\":" << total_prompt_tokens << ",";
    std::cout << "\"unique_embedding_rows_decoded\":"
              << embedding_stats.size() << ",";
    std::cout << "\"unique_embeddings_ok\":"
              << (unique_embeddings_ok ? "true" : "false");
    std::cout << "}\n";
    return passed ? 0 : 2;
  } catch (const std::exception& exc) {
    std::cerr << "iq36-seed-prompt-input-check: " << exc.what() << "\n";
    return 1;
  }
}
