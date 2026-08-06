#include "intel_qwen36/grouped_s8_u4_prefill_runtime.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<std::uint8_t> ReadBytes(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("could not open " + path);
  input.seekg(0, std::ios::end);
  const auto size = input.tellg();
  if (size < 0) throw std::runtime_error("could not size " + path);
  input.seekg(0, std::ios::beg);
  std::vector<std::uint8_t> values(static_cast<std::size_t>(size));
  input.read(reinterpret_cast<char*>(values.data()),
             static_cast<std::streamsize>(values.size()));
  if (!input) throw std::runtime_error("could not read " + path);
  return values;
}

template <typename Value>
std::vector<Value> ReadVector(const std::string& path,
                              std::size_t expected_count) {
  const auto bytes = ReadBytes(path);
  if (bytes.size() != expected_count * sizeof(Value)) {
    throw std::runtime_error("input size mismatch: " + path);
  }
  std::vector<Value> values(expected_count);
  std::memcpy(values.data(), bytes.data(), bytes.size());
  return values;
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
  double maximum_absolute_difference = 0.0;
  double error_squared = 0.0;
  double candidate_squared = 0.0;
  double reference_squared = 0.0;
  double dot = 0.0;
  bool finite = true;

  void Add(float candidate, float reference) {
    const double difference = static_cast<double>(candidate) - reference;
    ++count;
    mismatch_count += std::abs(difference) > 5e-3;
    maximum_absolute_difference = std::max(
        maximum_absolute_difference, std::abs(difference));
    error_squared += difference * difference;
    candidate_squared += static_cast<double>(candidate) * candidate;
    reference_squared += static_cast<double>(reference) * reference;
    dot += static_cast<double>(candidate) * reference;
    finite = finite && std::isfinite(candidate) && std::isfinite(reference);
  }

  bool pass() const {
    return finite && mismatch_count == 0 &&
        maximum_absolute_difference <= 5e-3;
  }
};

void PrintComparison(const char* name, const Comparison& comparison) {
  const double cosine = comparison.dot /
      std::sqrt(comparison.candidate_squared * comparison.reference_squared);
  const double relative_l2 = std::sqrt(
      comparison.error_squared / comparison.reference_squared);
  const double rmse = std::sqrt(
      comparison.error_squared / static_cast<double>(comparison.count));
  std::cout << "\"" << name << "\":{";
  std::cout << "\"compared_value_count\":" << comparison.count << ",";
  std::cout << "\"cosine\":" << cosine << ",";
  std::cout << "\"finite\":" << comparison.finite << ",";
  std::cout << "\"max_abs_diff\":"
            << comparison.maximum_absolute_difference << ",";
  std::cout << "\"mismatch_count\":" << comparison.mismatch_count << ",";
  std::cout << "\"relative_l2\":" << relative_l2 << ",";
  std::cout << "\"rmse\":" << rmse << "},";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 14) {
      throw std::invalid_argument(
          "usage: q6-prefill-resident PREP GATEUP DOWN SUPPORT Q6_KERNEL "
          "INPUT TOPK TOPK_STRIDE ROUTER SWIGLU_ORACLE DOWN_ORACLE "
          "MOE_ORACLE REPEAT");
    }
    constexpr std::size_t kTokenCount = 1024;
    constexpr std::size_t kHiddenSize = 2048;
    constexpr std::size_t kIntermediateSize = 512;
    constexpr std::size_t kAssignments = 8192;
    const int repeat = std::stoi(argv[13]);
    if (repeat <= 0) throw std::invalid_argument("repeat must be positive");

    iq36::GroupedS8U4PrefillProgramConfig program;
    program.gateup_binary = argv[2];
    program.down_binary = argv[3];
    program.kernels = argv[4];
    program.q6_down_kernels = argv[5];
    iq36::GroupedS8U4PrefillRuntime runtime(program);
    iq36::GroupedS8U4PrefillLayerConfig layer;
    layer.layer_index = 7;
    layer.prep_dir = argv[1];
    layer.down_kind = iq36::GroupedPrefillDownKind::kQ6U8Surrogate;
    const auto handle = runtime.LoadLayer(layer);

    iq36::GroupedS8U4PrefillInput input;
    input.hidden_states = ReadVector<float>(
        argv[6], kTokenCount * kHiddenSize);
    input.topk = ReadBytes(argv[7]);
    input.topk_stride = std::stoull(argv[8]);
    input.router_weights = ReadVector<float>(argv[9], kAssignments);
    input.warmup = 3;
    input.repeat = repeat;
    input.capture_intermediates = true;
    const auto run = runtime.RunLayer(handle, input);

    const auto swiglu_oracle = ReadVector<float>(
        argv[10], kAssignments * kIntermediateSize);
    const auto down_oracle = ReadVector<float>(
        argv[11], kAssignments * kHiddenSize);
    const auto moe_oracle = ReadVector<float>(
        argv[12], kTokenCount * kHiddenSize);
    Comparison swiglu;
    Comparison weighted_down;
    Comparison routed_output;
    for (std::size_t source = 0; source < kAssignments; ++source) {
      const std::size_t row = static_cast<std::size_t>(run.inverse_map[source]);
      for (std::size_t inner = 0; inner < kIntermediateSize; ++inner) {
        swiglu.Add(
            HalfToFloat(run.grouped_swiglu_f16[
                row * kIntermediateSize + inner]),
            swiglu_oracle[source * kIntermediateSize + inner]);
      }
      const float router_weight = input.router_weights[source];
      for (std::size_t hidden = 0; hidden < kHiddenSize; ++hidden) {
        weighted_down.Add(
            HalfToFloat(run.grouped_contributions_f16[
                row * kHiddenSize + hidden]),
            down_oracle[source * kHiddenSize + hidden] * router_weight);
      }
    }
    for (std::size_t index = 0; index < run.output.size(); ++index) {
      routed_output.Add(run.output[index], moe_oracle[index]);
    }
    const auto stats = runtime.stats();
    const bool component_cap_pass =
        run.timing.complete_minimum_us <= 9526.177;
    const bool pass = swiglu.pass() && weighted_down.pass() &&
        routed_output.pass() && run.maps_native_only && handle != 0 &&
        run.down_kind == iq36::GroupedPrefillDownKind::kQ6U8Surrogate &&
        run.q6_work_tile_count > 0 && stats.context_create_count == 1 &&
        stats.program_load_count == 4 && stats.layer_load_count == 1 &&
        stats.layer_count == 1 && stats.run_count == 1 &&
        stats.resident_weight_bytes == 645922816;

    std::cout << std::boolalpha << std::setprecision(12) << "{";
    std::cout << "\"active_experts\":" << run.active_experts << ",";
    std::cout << "\"component_cap_pass\":" << component_cap_pass << ",";
    std::cout << "\"complete_minimum_us\":"
              << run.timing.complete_minimum_us << ",";
    std::cout << "\"context_create_count\":"
              << stats.context_create_count << ",";
    std::cout << "\"device_name\":\"" << runtime.device_name() << "\",";
    std::cout << "\"integration_pass\":" << pass << ",";
    std::cout << "\"kernel_minimum_us\":" << run.timing.minimum_us << ",";
    std::cout << "\"maps_native_only\":" << run.maps_native_only << ",";
    std::cout << "\"max_group_size\":" << run.max_group_size << ",";
    PrintComparison("moe_compare", routed_output);
    std::cout << "\"program_load_count\":" << stats.program_load_count
              << ",";
    std::cout << "\"q6_work_tile_count\":" << run.q6_work_tile_count << ",";
    std::cout << "\"resident_weight_bytes\":"
              << stats.resident_weight_bytes << ",";
    std::cout << "\"schedule_setup_us\":"
              << run.timing.schedule_setup_us << ",";
    std::cout << "\"stage_us\":{";
    std::cout << "\"down\":" << run.timing.stage_us[3] << ",";
    std::cout << "\"down_quantize\":" << run.timing.stage_us[2] << ",";
    std::cout << "\"gateup\":" << run.timing.stage_us[1] << ",";
    std::cout << "\"gather\":" << run.timing.stage_us[0] << ",";
    std::cout << "\"scatter\":" << run.timing.stage_us[4] << "},";
    PrintComparison("swiglu_compare", swiglu);
    PrintComparison("weighted_down_compare", weighted_down);
    std::cout << "\"workgroup\":[16,1,1]}" << std::endl;
    return pass ? 0 : 2;
  } catch (const std::exception& exception) {
    std::cerr << "grouped-s8-u8-q6-prefill-resident: " << exception.what()
              << '\n';
    return 4;
  }
}
