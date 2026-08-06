#ifndef INTEL_QWEN36_TESTS_COMPARE_HARNESS_HPP
#define INTEL_QWEN36_TESTS_COMPARE_HARNESS_HPP

// Shared boundary-compare harness.
//
// Every engine compare test (engine/tests/*_compare.cpp) historically
// re-implemented the same argument parsing, JSON writing, and vector statistics
// (~120 lines, ~85% identical across files).  This header is the single source
// of truth for that boilerplate: a compare test #includes it and writes only
// its boundary-specific main().  Boundaries are registered in
// engine/boundaries.json (which drives CMake and tools/iq36-ladder.py).

#include "intel_qwen36/gguf_loader.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace iq36 {
namespace harness {

inline void require(bool ok, const char* message) {
  if (!ok) {
    throw std::runtime_error(message);
  }
}

inline std::uint64_t parse_u64(const std::string& value) {
  std::size_t consumed = 0;
  const auto parsed = std::stoull(value, &consumed, 10);
  if (consumed != value.size()) {
    throw std::invalid_argument("invalid unsigned integer argument");
  }
  return parsed;
}

inline std::string json_escape(const std::string& value) {
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

inline void write_u64_vector(const std::vector<std::uint64_t>& values) {
  std::cout << "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0) {
      std::cout << ",";
    }
    std::cout << values[i];
  }
  std::cout << "]";
}

struct ValueStats {
  std::uint64_t count = 0;
  double min = 0.0;
  double max = 0.0;
  double abs_sum = 0.0;
  double l2 = 0.0;
  bool finite = false;
  bool nonzero = false;
};

inline ValueStats stats_from_values(const std::vector<float>& values) {
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

inline void write_value_stats(const ValueStats& stats) {
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

inline void write_compare_stats(const iq36::VectorCompareStats& stats) {
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

}  // namespace harness
}  // namespace iq36

#endif  // INTEL_QWEN36_TESTS_COMPARE_HARNESS_HPP
