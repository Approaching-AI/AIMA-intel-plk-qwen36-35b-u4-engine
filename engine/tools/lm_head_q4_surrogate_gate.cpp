#include "intel_qwen36/gguf_loader.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kHiddenSize = 2048;
constexpr std::size_t kVocabSize = 248320;

void Require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

std::vector<float> ReadVectors(const std::string& path, std::size_t count) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  Require(static_cast<bool>(input), "cannot open vector payload");
  const auto size = input.tellg();
  const auto expected = static_cast<std::streamoff>(
      count * kHiddenSize * sizeof(float));
  Require(size == expected, "vector payload size mismatch");
  input.seekg(0);
  std::vector<float> values(count * kHiddenSize);
  input.read(reinterpret_cast<char*>(values.data()), size);
  Require(static_cast<bool>(input), "cannot read vector payload");
  return values;
}

std::size_t Top1(const std::vector<float>& values) {
  return static_cast<std::size_t>(
      std::max_element(values.begin(), values.end()) - values.begin());
}

std::size_t RankOf(const std::vector<float>& values, std::size_t target) {
  const float target_value = values[target];
  std::size_t rank = 1;
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (values[i] > target_value ||
        (values[i] == target_value && i < target)) {
      ++rank;
    }
  }
  return rank;
}

double KldWithExactCandidates(const std::vector<float>& exact,
                              const std::vector<float>& surrogate,
                              const std::vector<std::size_t>& order_rank,
                              std::size_t candidate_count) {
  const auto hybrid_value = [&](std::size_t i) {
    return order_rank[i] < candidate_count ? exact[i] : surrogate[i];
  };
  const double exact_max = *std::max_element(exact.begin(), exact.end());
  double hybrid_max = -std::numeric_limits<double>::infinity();
  for (std::size_t i = 0; i < exact.size(); ++i) {
    hybrid_max = std::max(hybrid_max, static_cast<double>(hybrid_value(i)));
  }
  double exact_sum = 0.0;
  double hybrid_sum = 0.0;
  for (std::size_t i = 0; i < exact.size(); ++i) {
    exact_sum += std::exp(static_cast<double>(exact[i]) - exact_max);
    hybrid_sum += std::exp(static_cast<double>(hybrid_value(i)) - hybrid_max);
  }
  const double exact_logz = exact_max + std::log(exact_sum);
  const double hybrid_logz = hybrid_max + std::log(hybrid_sum);
  double kld = 0.0;
  for (std::size_t i = 0; i < exact.size(); ++i) {
    const double probability =
        std::exp(static_cast<double>(exact[i]) - exact_logz);
    kld += probability *
        (static_cast<double>(exact[i]) - static_cast<double>(hybrid_value(i)) +
         hybrid_logz - exact_logz);
  }
  return std::max(0.0, kld);
}

double ElapsedMs(std::chrono::steady_clock::time_point begin) {
  return std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - begin).count();
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Require(argc == 5,
            "usage: gate <exact.gguf> <surrogate.gguf> <vectors.f32> <count>");
    const std::string exact_path = argv[1];
    const std::string surrogate_path = argv[2];
    const std::string vectors_path = argv[3];
    const auto vector_count = static_cast<std::size_t>(std::stoull(argv[4]));
    Require(vector_count > 0, "vector count must be positive");

    const auto exact_index = iq36::parse_gguf_model_index(exact_path);
    const auto surrogate_index = iq36::parse_gguf_model_index(surrogate_path);
    const auto* exact_tensor = iq36::find_tensor(exact_index, "output.weight");
    const auto* surrogate_tensor =
        iq36::find_tensor(surrogate_index, "output.weight");
    Require(exact_tensor != nullptr && surrogate_tensor != nullptr,
            "output.weight missing");
    Require(exact_tensor->type == 14, "exact output.weight must be Q6_K");
    Require(surrogate_tensor->type == 12,
            "surrogate output.weight must be Q4_K");
    const std::vector<std::uint64_t> expected_dims{kHiddenSize, kVocabSize};
    Require(exact_tensor->dims == expected_dims &&
                surrogate_tensor->dims == expected_dims,
            "output.weight dims mismatch");

    iq36::set_resident_tensor_cache_enabled(true);
    iq36::set_dense_matvec_enabled(true);
    iq36::set_dense_matvec_min_rows(1);
    iq36::set_dense_matvec_thread_count(16);
    iq36::set_dense_matvec_payload_cache_enabled(true);
    iq36::set_dense_q4_direct_dot_enabled(true);
    iq36::set_dense_q6_direct_dot_enabled(true);

    const auto vectors = ReadVectors(vectors_path, vector_count);
    std::vector<std::size_t> ranks;
    std::vector<std::size_t> exact_top1;
    std::vector<std::size_t> surrogate_top1;
    const std::vector<std::size_t> caps{16, 64, 256, 1024, 4096};
    std::vector<std::vector<double>> hybrid_kld(caps.size());
    std::vector<double> surrogate_kld;
    ranks.reserve(vector_count);
    exact_top1.reserve(vector_count);
    surrogate_top1.reserve(vector_count);
    surrogate_kld.reserve(vector_count);
    for (auto& values : hybrid_kld) values.reserve(vector_count);
    double exact_ms = 0.0;
    double surrogate_ms = 0.0;

    for (std::size_t row = 0; row < vector_count; ++row) {
      const auto begin = vectors.begin() + row * kHiddenSize;
      const std::vector<float> hidden(begin, begin + kHiddenSize);

      auto started = std::chrono::steady_clock::now();
      const auto exact =
          iq36::matvec_tensor(exact_path, exact_index, "output.weight", hidden);
      exact_ms += ElapsedMs(started);
      started = std::chrono::steady_clock::now();
      const auto surrogate = iq36::matvec_tensor(
          surrogate_path, surrogate_index, "output.weight", hidden);
      surrogate_ms += ElapsedMs(started);
      Require(exact.size() == kVocabSize && surrogate.size() == kVocabSize,
              "logit vector size mismatch");

      const auto winner = Top1(exact);
      exact_top1.push_back(winner);
      surrogate_top1.push_back(Top1(surrogate));
      ranks.push_back(RankOf(surrogate, winner));

      std::vector<std::size_t> ids(kVocabSize);
      for (std::size_t i = 0; i < ids.size(); ++i) ids[i] = i;
      const auto max_cap = caps.back();
      std::partial_sort(
          ids.begin(), ids.begin() + max_cap, ids.end(),
          [&](std::size_t lhs, std::size_t rhs) {
            return surrogate[lhs] > surrogate[rhs] ||
                (surrogate[lhs] == surrogate[rhs] && lhs < rhs);
          });
      std::vector<std::size_t> order_rank(kVocabSize, kVocabSize);
      for (std::size_t i = 0; i < max_cap; ++i) order_rank[ids[i]] = i;
      surrogate_kld.push_back(
          KldWithExactCandidates(exact, surrogate, order_rank, 0));
      for (std::size_t cap_index = 0; cap_index < caps.size(); ++cap_index) {
        hybrid_kld[cap_index].push_back(KldWithExactCandidates(
            exact, surrogate, order_rank, caps[cap_index]));
      }
    }

    std::cout << std::setprecision(17);
    std::cout << "{";
    std::cout << "\"exact_head_bytes\":" << exact_tensor->nbytes << ",";
    std::cout << "\"exact_ms\":" << exact_ms << ",";
    std::cout << "\"exact_top1\":[";
    for (std::size_t i = 0; i < exact_top1.size(); ++i) {
      if (i) std::cout << ",";
      std::cout << exact_top1[i];
    }
    std::cout << "],";
    std::cout << "\"caps\":[";
    for (std::size_t i = 0; i < caps.size(); ++i) {
      if (i) std::cout << ",";
      std::cout << caps[i];
    }
    std::cout << "],";
    std::cout << "\"hybrid_kld\":[";
    for (std::size_t cap_index = 0; cap_index < caps.size(); ++cap_index) {
      if (cap_index) std::cout << ",";
      std::cout << "[";
      for (std::size_t i = 0; i < hybrid_kld[cap_index].size(); ++i) {
        if (i) std::cout << ",";
        std::cout << hybrid_kld[cap_index][i];
      }
      std::cout << "]";
    }
    std::cout << "],";
    std::cout << "\"ranks\":[";
    for (std::size_t i = 0; i < ranks.size(); ++i) {
      if (i) std::cout << ",";
      std::cout << ranks[i];
    }
    std::cout << "],";
    std::cout << "\"schema_version\":\"intel-qwen36-lm-head-q4-surrogate-component-v0\",";
    std::cout << "\"surrogate_head_bytes\":" << surrogate_tensor->nbytes << ",";
    std::cout << "\"surrogate_ms\":" << surrogate_ms << ",";
    std::cout << "\"surrogate_kld\":[";
    for (std::size_t i = 0; i < surrogate_kld.size(); ++i) {
      if (i) std::cout << ",";
      std::cout << surrogate_kld[i];
    }
    std::cout << "],";
    std::cout << "\"surrogate_top1\":[";
    for (std::size_t i = 0; i < surrogate_top1.size(); ++i) {
      if (i) std::cout << ",";
      std::cout << surrogate_top1[i];
    }
    std::cout << "],";
    std::cout << "\"vector_count\":" << vector_count;
    std::cout << "}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "lm-head-q4-surrogate-gate: " << error.what() << "\n";
    return 1;
  }
}
