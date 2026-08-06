#include "intel_qwen36/grouped_s8_u4_prefill_runtime.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr double kComponentCosineMin = 0.999;
constexpr double kComponentRelativeL2Max = 0.002;

constexpr std::array<int, 20> kQ6Layers = {
    0, 1, 2, 3, 4, 7, 10, 13, 16, 19,
    22, 25, 28, 31, 34, 35, 36, 37, 38, 39};

bool IsQ6Layer(int layer) {
  return std::find(kQ6Layers.begin(), kQ6Layers.end(), layer) !=
      kQ6Layers.end();
}

std::vector<std::uint8_t> ReadBytes(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("could not open " + path.string());
  input.seekg(0, std::ios::end);
  const auto size = input.tellg();
  if (size < 0) throw std::runtime_error("could not size " + path.string());
  input.seekg(0, std::ios::beg);
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
  input.read(reinterpret_cast<char*>(bytes.data()),
             static_cast<std::streamsize>(bytes.size()));
  if (!input) throw std::runtime_error("could not read " + path.string());
  return bytes;
}

template <typename Value>
std::vector<Value> ReadVector(const std::filesystem::path& path,
                              std::size_t count) {
  const auto bytes = ReadBytes(path);
  if (bytes.size() != count * sizeof(Value)) {
    throw std::runtime_error("input size mismatch: " + path.string());
  }
  std::vector<Value> values(count);
  std::memcpy(values.data(), bytes.data(), bytes.size());
  return values;
}

std::filesystem::path Payload(const std::filesystem::path& directory,
                              const std::string& stem) {
  const std::string prefix = stem + "__tok1023__ord";
  std::filesystem::path result;
  for (const auto& entry : std::filesystem::directory_iterator(directory)) {
    const std::string name = entry.path().filename().string();
    if (name.rfind(prefix, 0) == 0 && entry.path().extension() == ".bin") {
      if (!result.empty()) {
        throw std::runtime_error("duplicate payload for " + stem);
      }
      result = entry.path();
    }
  }
  if (result.empty()) throw std::runtime_error("missing payload for " + stem);
  return result;
}

float HalfToFloat(std::uint16_t value) {
  const std::uint32_t sign = (std::uint32_t(value) & 0x8000U) << 16;
  std::uint32_t exponent = (value >> 10) & 0x1fU;
  std::uint32_t mantissa = value & 0x03ffU;
  std::uint32_t bits = 0;
  if (exponent == 0) {
    if (mantissa == 0) {
      bits = sign;
    } else {
      std::uint32_t shift = 0;
      while ((mantissa & 0x0400U) == 0) {
        mantissa <<= 1;
        ++shift;
      }
      mantissa &= 0x03ffU;
      bits = sign | ((127U - 14U - shift) << 23) | (mantissa << 13);
    }
  } else if (exponent == 0x1fU) {
    bits = sign | 0x7f800000U | (mantissa << 13);
  } else {
    bits = sign | ((exponent + 112U) << 23) | (mantissa << 13);
  }
  float result = 0.0f;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

struct Comparison {
  std::uint64_t count = 0;
  std::uint64_t mismatch_count = 0;
  double max_abs = 0.0;
  long double error_squared = 0.0;
  long double candidate_squared = 0.0;
  long double reference_squared = 0.0;
  long double dot = 0.0;
  bool finite = true;

  void Add(float candidate, float reference) {
    const double difference = static_cast<double>(candidate) - reference;
    ++count;
    mismatch_count += std::abs(difference) > 5e-3;
    max_abs = std::max(max_abs, std::abs(difference));
    error_squared += difference * difference;
    candidate_squared += static_cast<double>(candidate) * candidate;
    reference_squared += static_cast<double>(reference) * reference;
    dot += static_cast<double>(candidate) * reference;
    finite = finite && std::isfinite(candidate) && std::isfinite(reference);
  }

  bool pass() const {
    return finite && cosine() >= kComponentCosineMin &&
        relative_l2() <= kComponentRelativeL2Max;
  }

  double cosine() const {
    return static_cast<double>(
        dot / std::sqrt(candidate_squared * reference_squared));
  }

  double relative_l2() const {
    return static_cast<double>(std::sqrt(error_squared / reference_squared));
  }
};

int NearestInt(float value) {
  const float shifted = value + 12582912.0f;
  std::int32_t bits = 0;
  std::memcpy(&bits, &shifted, sizeof(bits));
  return (bits & 0x007fffff) - 0x00400000;
}

struct Q8Parity {
  std::uint64_t value_count = 0;
  std::uint64_t value_mismatch_count = 0;
  std::uint64_t scale_count = 0;
  std::uint64_t scale_bit_mismatch_count = 0;
  double max_scale_abs_diff = 0.0;

  void AddRow(const std::int8_t* candidate, const float* candidate_scales,
              const float* reference) {
    for (std::size_t block = 0; block < 2; ++block) {
      float maximum = 0.0f;
      float absolute_maximum = 0.0f;
      for (std::size_t inner = 0; inner < 256; ++inner) {
        const float value = reference[block * 256 + inner];
        const float absolute = std::fabs(value);
        if (absolute > absolute_maximum) {
          absolute_maximum = absolute;
          maximum = value;
        }
      }
      const float inverse_scale = absolute_maximum == 0.0f
          ? 0.0f : -127.0f / maximum;
      const float scale = inverse_scale == 0.0f
          ? 0.0f : 1.0f / inverse_scale;
      std::uint32_t candidate_bits = 0;
      std::uint32_t reference_bits = 0;
      std::memcpy(&candidate_bits, candidate_scales + block,
                  sizeof(candidate_bits));
      std::memcpy(&reference_bits, &scale, sizeof(reference_bits));
      ++scale_count;
      scale_bit_mismatch_count += candidate_bits != reference_bits;
      max_scale_abs_diff = std::max(
          max_scale_abs_diff,
          std::abs(static_cast<double>(candidate_scales[block]) - scale));
      for (std::size_t inner = 0; inner < 256; ++inner) {
        const std::size_t index = block * 256 + inner;
        const int expected = inverse_scale == 0.0f
            ? 0 : std::min(127, NearestInt(inverse_scale * reference[index]));
        ++value_count;
        value_mismatch_count += candidate[index] != expected;
      }
    }
  }
};

struct LayerRow {
  int layer = -1;
  const char* codec = "";
  std::size_t active_experts = 0;
  std::size_t max_group_size = 0;
  double complete_minimum_us = 0.0;
  Comparison swiglu;
  Q8Parity down_q8;
  Comparison weighted_down;
  Comparison routed_output;
};

void PrintComparison(const Comparison& value) {
  std::cout << "{\"compared_value_count\":" << value.count << ",";
  std::cout << "\"cosine\":" << value.cosine() << ",";
  std::cout << "\"finite\":" << value.finite << ",";
  std::cout << "\"max_abs_diff\":" << value.max_abs << ",";
  std::cout << "\"mismatch_count\":" << value.mismatch_count << ",";
  std::cout << "\"relative_l2\":" << value.relative_l2() << "}";
}

void PrintQ8Parity(const Q8Parity& value) {
  std::cout << "{\"max_scale_abs_diff\":" << value.max_scale_abs_diff
            << ",\"scale_bit_mismatch_count\":"
            << value.scale_bit_mismatch_count
            << ",\"scale_count\":" << value.scale_count
            << ",\"value_count\":" << value.value_count
            << ",\"value_mismatch_count\":"
            << value.value_mismatch_count << "}";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 8 && argc != 9) {
      throw std::invalid_argument(
          "usage: mixed-all-layer-compare GATEUP DOWN SUPPORT Q6_KERNEL "
          "PREP_ROOT CAPTURE_DIR REPEAT [--reference-swiglu]");
    }
    constexpr std::size_t kTokenCount = 1024;
    constexpr std::size_t kHiddenSize = 2048;
    constexpr std::size_t kIntermediateSize = 512;
    constexpr std::size_t kAssignments = 8192;
    const int repeat = std::stoi(argv[7]);
    const bool reference_swiglu = argc == 9 &&
        std::string(argv[8]) == "--reference-swiglu";
    if (argc == 9 && !reference_swiglu) {
      throw std::invalid_argument("unknown all-layer compare option");
    }
    if (repeat <= 0) throw std::invalid_argument("repeat must be positive");

    iq36::GroupedS8U4PrefillProgramConfig program;
    program.gateup_binary = argv[1];
    program.down_binary = argv[2];
    program.kernels = argv[3];
    program.q6_down_kernels = argv[4];
    iq36::GroupedS8U4PrefillRuntime runtime(program);
    const std::filesystem::path prep_root = argv[5];
    const std::filesystem::path payloads =
        std::filesystem::path(argv[6]) / "payloads";
    std::array<std::uint64_t, 40> handles{};
    for (int layer = 0; layer < 40; ++layer) {
      std::ostringstream name;
      name << "layer-" << std::setfill('0') << std::setw(2) << layer;
      iq36::GroupedS8U4PrefillLayerConfig config;
      config.layer_index = layer;
      config.prep_dir = (prep_root / name.str()).string();
      config.exact_q4_gateup = true;
      if (IsQ6Layer(layer)) {
        config.down_kind = iq36::GroupedPrefillDownKind::kQ6U8ExactBlock;
      } else {
        config.down_kind = iq36::GroupedPrefillDownKind::kQ4U4ExactBlock;
      }
      handles[static_cast<std::size_t>(layer)] = runtime.LoadLayer(config);
    }

    std::vector<LayerRow> rows;
    rows.reserve(40);
    Comparison all_swiglu;
    Q8Parity all_down_q8;
    Comparison all_weighted_down;
    Comparison all_routed_output;
    for (int layer = 0; layer < 40; ++layer) {
      iq36::GroupedS8U4PrefillInput input;
      input.hidden_states = ReadVector<float>(
          Payload(payloads, "attn_post_norm-" + std::to_string(layer)),
          kTokenCount * kHiddenSize);
      input.topk = ReadBytes(
          Payload(payloads, "ffn_moe_topk-" + std::to_string(layer)));
      input.topk_stride = 1024;
      input.router_weights = ReadVector<float>(
          Payload(payloads,
                  "ffn_moe_weights_norm-" + std::to_string(layer)),
          kAssignments);
      const auto swiglu_oracle = ReadVector<float>(
          Payload(payloads, "ffn_moe_swiglu-" + std::to_string(layer)),
          kAssignments * kIntermediateSize);
      if (reference_swiglu) {
        input.swiglu_override_source_order = swiglu_oracle;
      }
      input.warmup = 0;
      input.repeat = repeat;
      input.capture_intermediates = true;
      const auto run = runtime.RunLayer(
          handles[static_cast<std::size_t>(layer)], input);
      const auto down_oracle = ReadVector<float>(
          Payload(payloads, "ffn_moe_down-" + std::to_string(layer)),
          kAssignments * kHiddenSize);
      const auto output_oracle = ReadVector<float>(
          Payload(payloads, "ffn_moe_out-" + std::to_string(layer)),
          kTokenCount * kHiddenSize);
      LayerRow row;
      row.layer = layer;
      row.codec = IsQ6Layer(layer)
          ? "Q6_K_EXACT_BLOCK" : "Q4_K_EXACT_BLOCK";
      row.active_experts = run.active_experts;
      row.max_group_size = run.max_group_size;
      row.complete_minimum_us = run.timing.complete_minimum_us;
      for (std::size_t source = 0; source < kAssignments; ++source) {
        const std::size_t grouped = static_cast<std::size_t>(
            run.inverse_map[source]);
        for (std::size_t inner = 0; inner < kIntermediateSize; ++inner) {
          const float candidate = run.grouped_swiglu_f32[
              grouped * kIntermediateSize + inner];
          const float reference =
              swiglu_oracle[source * kIntermediateSize + inner];
          row.swiglu.Add(candidate, reference);
          all_swiglu.Add(candidate, reference);
        }
        row.down_q8.AddRow(
            run.grouped_down_q8.data() + grouped * kIntermediateSize,
            run.grouped_down_scales.data() + grouped * 2,
            swiglu_oracle.data() + source * kIntermediateSize);
        all_down_q8.AddRow(
            run.grouped_down_q8.data() + grouped * kIntermediateSize,
            run.grouped_down_scales.data() + grouped * 2,
            swiglu_oracle.data() + source * kIntermediateSize);
        const float router = input.router_weights[source];
        for (std::size_t hidden = 0; hidden < kHiddenSize; ++hidden) {
          const float candidate = run.grouped_contributions_f32[
              grouped * kHiddenSize + hidden];
          const float reference =
              down_oracle[source * kHiddenSize + hidden] * router;
          row.weighted_down.Add(candidate, reference);
          all_weighted_down.Add(candidate, reference);
        }
      }
      for (std::size_t index = 0; index < run.output.size(); ++index) {
        row.routed_output.Add(run.output[index], output_oracle[index]);
        all_routed_output.Add(run.output[index], output_oracle[index]);
      }
      rows.push_back(std::move(row));
    }

    const auto stats = runtime.stats();
    const bool all_rows_pass = std::all_of(
        rows.begin(), rows.end(), [](const LayerRow& row) {
          return row.swiglu.pass() && row.weighted_down.pass() &&
              row.routed_output.pass();
        });
    const bool pass = all_rows_pass && all_swiglu.pass() &&
        all_weighted_down.pass() && all_routed_output.pass() &&
        stats.context_create_count == 1 && stats.program_load_count == 4 &&
        stats.layer_count == 40 && stats.layer_load_count == 40 &&
        stats.run_count == 40 &&
        stats.resident_weight_bytes == 21726494720ULL;
    std::cout << std::boolalpha << std::setprecision(12) << "{";
    std::cout << "\"all_layer_compare_pass\":" << pass << ",";
    std::cout << "\"all_routed_output_compare\":";
    PrintComparison(all_routed_output);
    std::cout << ",\"all_swiglu_compare\":";
    PrintComparison(all_swiglu);
    std::cout << ",\"all_down_q8_parity\":";
    PrintQ8Parity(all_down_q8);
    std::cout << ",\"all_weighted_down_compare\":";
    PrintComparison(all_weighted_down);
    std::cout << ",\"context_create_count\":"
              << stats.context_create_count << ",";
    std::cout << "\"device_name\":\"" << runtime.device_name() << "\",";
    std::cout << "\"layer_count\":" << rows.size() << ",";
    std::cout << "\"per_layer\":[";
    for (std::size_t index = 0; index < rows.size(); ++index) {
      if (index != 0) std::cout << ",";
      const auto& row = rows[index];
      std::cout << "{\"active_experts\":" << row.active_experts << ",";
      std::cout << "\"codec\":\"" << row.codec << "\",";
      std::cout << "\"complete_minimum_us\":"
                << row.complete_minimum_us << ",";
      std::cout << "\"layer\":" << row.layer << ",";
      std::cout << "\"max_group_size\":" << row.max_group_size << ",";
      std::cout << "\"routed_output_compare\":";
      PrintComparison(row.routed_output);
      std::cout << ",\"swiglu_compare\":";
      PrintComparison(row.swiglu);
      std::cout << ",\"down_q8_parity\":";
      PrintQ8Parity(row.down_q8);
      std::cout << ",\"weighted_down_compare\":";
      PrintComparison(row.weighted_down);
      std::cout << "}";
    }
    std::cout << "],\"program_load_count\":"
              << stats.program_load_count << ",";
    std::cout << "\"reference_swiglu\":" << reference_swiglu << ",";
    std::cout << "\"resident_weight_bytes\":"
              << stats.resident_weight_bytes << ",";
    std::cout << "\"run_count\":" << stats.run_count << "}"
              << std::endl;
    return pass ? 0 : 2;
  } catch (const std::exception& exception) {
    std::cerr << "mixed-prefill-all-layer-compare: " << exception.what()
              << '\n';
    return 4;
  }
}
