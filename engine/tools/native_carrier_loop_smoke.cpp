#include "intel_qwen36/gguf_loader.hpp"
#include "intel_qwen36/native_carrier_loop.hpp"
#include "intel_qwen36/resident_harness.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<std::uint8_t> ReadBytes(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("could not open input: " + path);
  input.seekg(0, std::ios::end);
  const auto size = input.tellg();
  if (size < 0) throw std::runtime_error("could not size input: " + path);
  input.seekg(0, std::ios::beg);
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
  input.read(reinterpret_cast<char*>(bytes.data()),
             static_cast<std::streamsize>(bytes.size()));
  if (!input) throw std::runtime_error("could not read input: " + path);
  return bytes;
}

std::vector<std::uint8_t> ReadTensorBytes(
    const std::string& model_path,
    const iq36::GgufTensorInfo& tensor) {
  std::ifstream input(model_path, std::ios::binary);
  if (!input) throw std::runtime_error("could not open GGUF model");
  input.seekg(static_cast<std::streamoff>(tensor.absolute_offset));
  if (!input) throw std::runtime_error("could not seek GGUF tensor");
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(tensor.nbytes));
  input.read(reinterpret_cast<char*>(bytes.data()),
             static_cast<std::streamsize>(bytes.size()));
  if (!input) throw std::runtime_error("could not read GGUF tensor");
  return bytes;
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

bool MapsAreNativeOnly() {
  std::ifstream maps("/proc/self/maps");
  std::string line;
  while (std::getline(maps, line)) {
    std::string lower = line;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char value) { return std::tolower(value); });
    if (lower.find("libdnnl") != std::string::npos ||
        lower.find("openvino") != std::string::npos) {
      return false;
    }
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 12) {
      throw std::invalid_argument(
          "usage: native-carrier-loop-smoke MODEL PREP GATEUP DOWN "
          "GROUPED_KERNEL Q6_KERNEL INPUT TOPK TOPK_STRIDE ROUTER_WEIGHTS "
          "MOE_ORACLE");
    }
    constexpr std::size_t kTokenCount = 1024;
    constexpr std::size_t kHiddenSize = 2048;
    constexpr std::size_t kAssignmentCount = 8192;
    constexpr char kQ6Tensor[] = "blk.7.ffn_down_exps.weight";

    const auto index = iq36::parse_gguf_model_index(argv[1]);
    const auto* tensor = iq36::find_tensor(index, kQ6Tensor);
    if (tensor == nullptr || tensor->dims.size() != 3 ||
        tensor->dims[0] != 512 || tensor->dims[1] != 2048 ||
        tensor->dims[2] != 256 || tensor->nbytes != 220200960) {
      throw std::runtime_error("locked Q6 carrier tensor shape mismatch");
    }

    iq36::NativeCarrierProgramConfig program;
    program.grouped_prefill.gateup_binary = argv[3];
    program.grouped_prefill.down_binary = argv[4];
    program.grouped_prefill.kernels = argv[5];
    program.q6_device_substring = "B390";
    program.q6_opencl_source_path = argv[6];
    iq36::NativeCarrierLayerRuntime runtime(program);
    runtime.LoadGroupedPrefillLayer({27, argv[2]});
    iq36::NativeQ6CarrierLayerConfig q6_layer;
    q6_layer.layer_index = 7;
    q6_layer.raw_weights = ReadTensorBytes(argv[1], *tensor);
    q6_layer.rows_per_expert = tensor->dims[1];
    q6_layer.blocks_per_row = tensor->dims[0] / 256;
    q6_layer.expert_count = tensor->dims[2];
    q6_layer.rows_per_tile = 16;
    runtime.LoadQ6Layer(q6_layer);
    q6_layer.raw_weights.clear();
    q6_layer.raw_weights.shrink_to_fit();

    auto hidden_states = ReadVector<float>(
        argv[7], kTokenCount * kHiddenSize);
    std::vector<float> q6_source(
        hidden_states.begin(), hidden_states.begin() + 512);
    iq36::NativeGroupedPrefillStep prefill_step;
    prefill_step.layer_index = 27;
    prefill_step.input.hidden_states = std::move(hidden_states);
    prefill_step.input.topk = ReadBytes(argv[8]);
    prefill_step.input.topk_stride = std::stoull(argv[9]);
    prefill_step.input.router_weights =
        ReadVector<float>(argv[10], kAssignmentCount);
    prefill_step.input.warmup = 3;
    prefill_step.input.repeat = 3;

    iq36::NativeQ6CarrierStep q6_step;
    q6_step.layer_index = 7;
    q6_step.input = iq36::QuantizeQ8KInputPlanes(q6_source);
    q6_step.repeat = 5;
    iq36::NativeCarrierLoop loop(runtime);
    std::vector<iq36::NativeGroupedPrefillStep> prefill_steps;
    prefill_steps.push_back(std::move(prefill_step));
    std::vector<iq36::NativeQ6CarrierStep> q6_steps;
    q6_steps.push_back(std::move(q6_step));
    const auto prefill = loop.RunGroupedPrefill(prefill_steps);
    const auto q6 = loop.RunQ6(q6_steps);
    const auto moe_oracle = ReadVector<float>(
        argv[11], kTokenCount * kHiddenSize);

    std::size_t prefill_mismatch_count = 0;
    double prefill_max_abs_diff = 0.0;
    bool finite = true;
    for (std::size_t i = 0; i < moe_oracle.size(); ++i) {
      const double observed = prefill.runs[0].output[i];
      const double difference = std::abs(observed - moe_oracle[i]);
      finite = finite && std::isfinite(observed);
      prefill_max_abs_diff = std::max(prefill_max_abs_diff, difference);
      prefill_mismatch_count += difference > 5e-3;
    }

    std::vector<std::int32_t> all_experts(256);
    std::iota(all_experts.begin(), all_experts.end(), 0);
    iq36::set_expert_slice_matvec_enabled(true);
    iq36::set_expert_slice_matvec_thread_count(16);
    const auto q6_oracle = iq36::matvec_expert_tensor(
        argv[1], index, kQ6Tensor, q6_source, all_experts);
    const auto q6_compare = iq36::compare_vectors(
        q6.runs[0].output, q6_oracle, 1e-4);
    const auto stats = runtime.stats();
    const bool maps_native_only = MapsAreNativeOnly();
    const bool pass = finite && prefill_mismatch_count == 0 &&
        prefill.runs.size() == 1 && prefill.layer_indices ==
            std::vector<int>{27} &&
        prefill.runs[0].timing.complete_minimum_us <= 9526.177 &&
        q6.runs.size() == 1 && q6.layer_indices == std::vector<int>{7} &&
        q6_compare.same_size && q6_compare.finite &&
        q6_compare.mismatch_count == 0 &&
        q6.runs[0].timing.effective_packed_gb_s >= 58.0 &&
        stats.grouped_prefill.context_create_count == 1 &&
        stats.grouped_prefill.program_load_count == 3 &&
        stats.grouped_prefill.layer_count == 1 &&
        stats.grouped_prefill.run_count == 1 &&
        stats.q6_context_create_count == 1 &&
        stats.q6_layer_load_count == 1 && stats.q6_layer_count == 1 &&
        stats.q6_run_count == 1 &&
        stats.q6_resident_weight_bytes == 220200960 && maps_native_only;

    std::cout << std::boolalpha << std::setprecision(12) << "{"
              << "\"grouped_device_name\":\""
              << runtime.grouped_prefill_device_name() << "\","
              << "\"grouped_context_create_count\":"
              << stats.grouped_prefill.context_create_count << ","
              << "\"grouped_layer_count\":"
              << stats.grouped_prefill.layer_count << ","
              << "\"grouped_program_load_count\":"
              << stats.grouped_prefill.program_load_count << ","
              << "\"grouped_run_count\":"
              << stats.grouped_prefill.run_count << ","
              << "\"maps_native_only\":" << maps_native_only << ","
              << "\"parameterized_layer_count\":"
              << iq36::parameterized_layer_count() << ","
              << "\"prefill_complete_minimum_us\":"
              << prefill.runs[0].timing.complete_minimum_us << ","
              << "\"prefill_max_abs_diff\":" << prefill_max_abs_diff << ","
              << "\"prefill_mismatch_count\":"
              << prefill_mismatch_count << ","
              << "\"q6_context_create_count\":"
              << stats.q6_context_create_count << ","
              << "\"q6_device_name\":\"" << runtime.q6_device_name()
              << "\","
              << "\"q6_effective_packed_gb_s\":"
              << q6.runs[0].timing.effective_packed_gb_s << ","
              << "\"q6_layer_count\":" << stats.q6_layer_count << ","
              << "\"q6_max_abs_diff\":" << q6_compare.max_abs_diff << ","
              << "\"q6_mismatch_count\":" << q6_compare.mismatch_count << ","
              << "\"q6_resident_weight_bytes\":"
              << stats.q6_resident_weight_bytes << ","
              << "\"q6_run_count\":" << stats.q6_run_count << ","
              << "\"required_checks_passed\":" << pass << "}" << std::endl;
    return pass ? 0 : 2;
  } catch (const std::exception& exception) {
    std::cerr << exception.what() << '\n';
    return 4;
  }
}
