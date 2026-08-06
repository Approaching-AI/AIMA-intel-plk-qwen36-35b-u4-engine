#include <oneapi/dnnl/dnnl.hpp>

#include "intel_qwen36/gguf_loader.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

constexpr std::size_t kTokens = 1024;
constexpr std::size_t kHidden = 2048;
constexpr std::size_t kExperts = 256;
constexpr std::size_t kTopK = 8;
constexpr std::size_t kAssignments = kTokens * kTopK;

template <typename Value>
std::vector<Value> ReadVector(const std::string& path,
                              std::size_t expected_count) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) throw std::runtime_error("could not open input: " + path);
  const auto size = input.tellg();
  if (size != static_cast<std::streamoff>(
                  expected_count * sizeof(Value))) {
    throw std::runtime_error("input size mismatch: " + path);
  }
  input.seekg(0);
  std::vector<Value> values(expected_count);
  input.read(reinterpret_cast<char*>(values.data()), size);
  if (!input) throw std::runtime_error("could not read input: " + path);
  return values;
}

template <typename Value>
void WriteMemory(const std::vector<Value>& values, dnnl::memory& memory) {
  if (memory.get_desc().get_size() != values.size() * sizeof(Value)) {
    throw std::runtime_error("oneDNN memory size mismatch");
  }
  void* mapped = memory.map_data();
  if (mapped == nullptr) throw std::runtime_error("oneDNN map returned null");
  std::memcpy(mapped, values.data(), values.size() * sizeof(Value));
  memory.unmap_data(mapped);
}

template <typename Value>
std::vector<Value> ReadMemory(dnnl::memory& memory, std::size_t count) {
  if (memory.get_desc().get_size() != count * sizeof(Value)) {
    throw std::runtime_error("oneDNN output size mismatch");
  }
  void* mapped = memory.map_data();
  if (mapped == nullptr) throw std::runtime_error("oneDNN map returned null");
  std::vector<Value> values(count);
  std::memcpy(values.data(), mapped, values.size() * sizeof(Value));
  memory.unmap_data(mapped);
  return values;
}

double Median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

void PrintSamples(const std::vector<double>& values) {
  std::cout << '[';
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0) std::cout << ',';
    std::cout << values[index];
  }
  std::cout << ']';
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 7 || argc > 10) {
      throw std::invalid_argument(
          "usage: onednn-router-prefill-probe MODEL LAYER HIDDEN TOPK "
          "ROUTER_WEIGHTS CAP_US [WARMUP] [REPEAT] [WEIGHTS_OUT]");
    }
    const std::string model_path = argv[1];
    const int layer = std::stoi(argv[2]);
    const double cap_us = std::stod(argv[6]);
    const int warmup = argc >= 8 ? std::stoi(argv[7]) : 5;
    const int repeat = argc >= 9 ? std::stoi(argv[8]) : 21;
    if (layer < 0 || layer >= 40 || cap_us <= 0.0 || warmup < 0 ||
        repeat < 3) {
      throw std::invalid_argument("router probe argument is invalid");
    }
    const auto hidden = ReadVector<float>(argv[3], kTokens * kHidden);
    const auto reference_ids = ReadVector<std::int32_t>(
        argv[4], kAssignments);
    const auto reference_weights = ReadVector<float>(
        argv[5], kAssignments);

    const auto index = iq36::parse_gguf_model_index(model_path);
    const std::string tensor_name = "blk." + std::to_string(layer) +
        ".ffn_gate_inp.weight";
    const auto* tensor = iq36::find_tensor(index, tensor_name);
    if (tensor == nullptr || tensor->type != 0 ||
        tensor->dims != std::vector<std::uint64_t>{kHidden, kExperts}) {
      throw std::runtime_error("router tensor contract mismatch");
    }
    std::vector<float> weights(kExperts * kHidden);
    for (std::size_t expert = 0; expert < kExperts; ++expert) {
      const auto row = iq36::decode_tensor_row(
          model_path, index, tensor_name, expert);
      if (row.size() != kHidden) {
        throw std::runtime_error("decoded router row size mismatch");
      }
      std::copy(row.begin(), row.end(), weights.begin() + expert * kHidden);
    }
    if (argc >= 10) {
      std::ofstream output(argv[9], std::ios::binary | std::ios::trunc);
      if (!output) {
        throw std::runtime_error("could not create router weights output");
      }
      output.write(
          reinterpret_cast<const char*>(weights.data()),
          static_cast<std::streamsize>(weights.size() * sizeof(float)));
      if (!output) {
        throw std::runtime_error("could not write router weights output");
      }
    }

    using data_type = dnnl::memory::data_type;
    using format_tag = dnnl::memory::format_tag;
    dnnl::engine engine(dnnl::engine::kind::gpu, 0);
    dnnl::stream stream(engine);
    dnnl::memory source(
        dnnl::memory::desc({kTokens, kHidden}, data_type::f32,
                           format_tag::ab),
        engine);
    dnnl::memory weight(
        dnnl::memory::desc({kHidden, kExperts}, data_type::f32,
                           format_tag::ba),
        engine);
    dnnl::memory destination(
        dnnl::memory::desc({kTokens, kExperts}, data_type::f32,
                           format_tag::ab),
        engine);
    WriteMemory(hidden, source);
    WriteMemory(weights, weight);
    const dnnl::matmul::primitive_desc descriptor(
        engine, source.get_desc(), weight.get_desc(), destination.get_desc());
    const dnnl::matmul primitive(descriptor);
    const std::unordered_map<int, dnnl::memory> arguments = {
        {DNNL_ARG_SRC, source},
        {DNNL_ARG_WEIGHTS, weight},
        {DNNL_ARG_DST, destination},
    };
    for (int iteration = 0; iteration < warmup; ++iteration) {
      primitive.execute(stream, arguments);
      stream.wait();
    }
    std::vector<double> samples;
    samples.reserve(repeat);
    for (int iteration = 0; iteration < repeat; ++iteration) {
      const auto begin = std::chrono::steady_clock::now();
      primitive.execute(stream, arguments);
      stream.wait();
      const auto end = std::chrono::steady_clock::now();
      samples.push_back(std::chrono::duration<double, std::micro>(
                            end - begin).count());
    }
    auto logits = ReadMemory<float>(destination, kTokens * kExperts);

    std::size_t ordered_match_rows = 0;
    std::size_t set_match_rows = 0;
    std::size_t missing_experts = 0;
    double maximum_weight_diff = 0.0;
    for (std::size_t token = 0; token < kTokens; ++token) {
      std::array<std::size_t, kExperts> order{};
      for (std::size_t expert = 0; expert < kExperts; ++expert) {
        order[expert] = expert;
      }
      const auto compare = [&](std::size_t lhs, std::size_t rhs) {
        const float lhs_value = logits[token * kExperts + lhs];
        const float rhs_value = logits[token * kExperts + rhs];
        return lhs_value != rhs_value ? lhs_value > rhs_value : lhs < rhs;
      };
      std::partial_sort(order.begin(), order.begin() + kTopK, order.end(),
                        compare);
      std::array<bool, kExperts> reference_set{};
      std::array<bool, kExperts> observed_set{};
      std::array<float, kExperts> reference_by_id{};
      bool ordered = true;
      for (std::size_t rank = 0; rank < kTopK; ++rank) {
        const auto source_index = token * kTopK + rank;
        const int reference_id = reference_ids[source_index];
        if (reference_id < 0 || reference_id >= 256) {
          throw std::runtime_error("reference expert is out of range");
        }
        ordered = ordered &&
            static_cast<std::size_t>(reference_id) == order[rank];
        reference_set[static_cast<std::size_t>(reference_id)] = true;
        observed_set[order[rank]] = true;
        reference_by_id[static_cast<std::size_t>(reference_id)] =
            reference_weights[source_index];
      }
      ordered_match_rows += ordered;
      bool set_match = true;
      for (std::size_t expert = 0; expert < kExperts; ++expert) {
        if (reference_set[expert] != observed_set[expert]) {
          set_match = false;
          missing_experts += reference_set[expert] && !observed_set[expert];
        }
      }
      set_match_rows += set_match;
      if (set_match) {
        const float maximum = logits[token * kExperts + order[0]];
        std::array<double, kTopK> exponentials{};
        double denominator = 0.0;
        for (std::size_t rank = 0; rank < kTopK; ++rank) {
          exponentials[rank] = std::exp(static_cast<double>(
              logits[token * kExperts + order[rank]] - maximum));
          denominator += exponentials[rank];
        }
        for (std::size_t rank = 0; rank < kTopK; ++rank) {
          maximum_weight_diff = std::max(
              maximum_weight_diff,
              std::abs(exponentials[rank] / denominator -
                       reference_by_id[order[rank]]));
        }
      }
    }
    const double median_us = Median(samples);
    const bool passed = set_match_rows == kTokens &&
        maximum_weight_diff <= 0.002 && median_us <= cap_us;
    std::cout << std::boolalpha << std::setprecision(12) << '{'
              << "\"implementation_info\":\""
              << descriptor.impl_info_str() << "\","
              << "\"kernel_samples_us\":";
    PrintSamples(samples);
    std::cout << ",\"maximum_router_weight_abs_diff\":"
              << maximum_weight_diff
              << ",\"missing_expert_count\":" << missing_experts
              << ",\"ordered_top8_match_rows\":" << ordered_match_rows
              << ",\"required_checks_passed\":" << passed
              << ",\"router_cap_us\":" << cap_us
              << ",\"router_median_us\":" << median_us
              << ",\"router_representation\":\"f32_weight_f32_input\""
              << ",\"set_top8_match_rows\":" << set_match_rows
              << '}' << std::endl;
    return passed ? 0 : 2;
  } catch (const dnnl::error& error) {
    std::cerr << "oneDNN error " << error.status << ": " << error.what()
              << std::endl;
    return 1;
  } catch (const std::exception& error) {
    std::cerr << error.what() << std::endl;
    return 1;
  }
}
