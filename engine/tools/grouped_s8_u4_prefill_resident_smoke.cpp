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

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 10 || argc > 15) {
      throw std::invalid_argument(
          "usage: resident-smoke PREP GATEUP DOWN KERNEL INPUT TOPK "
          "TOPK_STRIDE ROUTER_WEIGHTS MOE_ORACLE "
          "[--exact-block] [--device-schedule] "
          "[--native-router PROGRAM WEIGHTS] [--persistent-dispatch]");
    }
    bool exact_block = false;
    bool device_schedule = false;
    bool native_router = false;
    bool persistent_dispatch = false;
    std::string router_program;
    std::string router_weights;
    for (int index = 10; index < argc; ++index) {
      const std::string option = argv[index];
      if (option == "--exact-block") {
        exact_block = true;
      } else if (option == "--device-schedule") {
        device_schedule = true;
      } else if (option == "--native-router") {
        if (index + 2 >= argc) {
          throw std::invalid_argument(
              "--native-router requires PROGRAM and WEIGHTS");
        }
        native_router = true;
        device_schedule = true;
        router_program = argv[++index];
        router_weights = argv[++index];
      } else if (option == "--persistent-dispatch") {
        persistent_dispatch = true;
      } else {
        throw std::invalid_argument("unknown resident-smoke option");
      }
    }
    constexpr std::size_t kTokenCount = 1024;
    constexpr std::size_t kHiddenSize = 2048;
    constexpr std::size_t kAssignmentCount = 8192;
    iq36::GroupedS8U4PrefillProgramConfig program;
    program.gateup_binary = argv[2];
    program.down_binary = argv[3];
    program.router_binary = router_program;
    program.kernels = argv[4];
    program.persistent_dispatch = persistent_dispatch;
    iq36::GroupedS8U4PrefillRuntime runtime(program);
    iq36::GroupedS8U4PrefillLayerConfig layer;
    layer.layer_index = 27;
    layer.prep_dir = argv[1];
    layer.router_weights = router_weights;
    layer.exact_q4_gateup = exact_block;
    if (exact_block) {
      layer.down_kind = iq36::GroupedPrefillDownKind::kQ4U4ExactBlock;
    }
    const auto handle = runtime.LoadLayer(layer);

    iq36::GroupedS8U4PrefillInput input;
    input.hidden_states = ReadVector<float>(
        argv[5], kTokenCount * kHiddenSize);
    input.topk = ReadBytes(argv[6]);
    input.topk_stride = std::stoull(argv[7]);
    input.router_weights = ReadVector<float>(argv[8], kAssignmentCount);
    input.device_schedule = device_schedule;
    input.native_router = native_router;
    input.warmup = 1;
    input.repeat = 2;
    const auto first = runtime.RunLayer(handle, input);
    input.warmup = 0;
    input.repeat = 1;
    const auto second = runtime.RunLayer(handle, input);
    const auto oracle = ReadVector<float>(
        argv[9], kTokenCount * kHiddenSize);

    std::size_t oracle_mismatch_count = 0;
    std::size_t deterministic_mismatch_count = 0;
    double max_abs_diff = 0.0;
    bool finite = true;
    for (std::size_t index = 0; index < oracle.size(); ++index) {
      const double observed = first.output[index];
      const double difference = std::abs(observed - oracle[index]);
      finite = finite && std::isfinite(observed);
      max_abs_diff = std::max(max_abs_diff, difference);
      oracle_mismatch_count += difference > 5e-3;
      deterministic_mismatch_count +=
          first.output[index] != second.output[index];
    }
    const auto stats = runtime.stats();
    const double steady_state_cap_us =
        9526.177 + (native_router ? 500.0 : 0.0);
    const bool steady_state_cap_pass =
        second.timing.complete_minimum_us <= steady_state_cap_us;
    const bool reuse_pass = finite && oracle_mismatch_count == 0 &&
        deterministic_mismatch_count == 0 && first.maps_native_only &&
        second.maps_native_only && stats.context_create_count == 1 &&
        stats.program_load_count == (native_router ? 4U : 3U) &&
        stats.layer_load_count == 1 &&
        stats.layer_count == 1 && stats.run_count == 2 &&
        stats.device_schedule_run_count == (device_schedule ? 2U : 0U) &&
        stats.native_router_run_count == (native_router ? 2U : 0U) &&
        stats.persistent_dispatch_run_count ==
            (persistent_dispatch ? 2U : 0U) &&
        stats.device_schedule_host_upload_bytes ==
            (device_schedule ? (native_router
                                   ? 0U
                                   : 2U * kAssignmentCount *
                                         (sizeof(std::int32_t) + sizeof(float)))
                             : 0U) &&
        stats.device_schedule_host_read_bytes ==
            (device_schedule && !persistent_dispatch
                 ? 2U * 5U * sizeof(std::uint32_t) : 0U) &&
        stats.resident_weight_bytes ==
            (exact_block ? 478150656ULL : 541065216ULL) +
                (native_router ? 2097152ULL : 0ULL) &&
        steady_state_cap_pass;

    std::cout << std::boolalpha << std::setprecision(12) << "{"
              << "\"context_create_count\":"
              << stats.context_create_count << ","
              << "\"deterministic_mismatch_count\":"
              << deterministic_mismatch_count << ","
              << "\"device_name\":\"" << runtime.device_name() << "\","
              << "\"device_schedule\":" << device_schedule << ","
              << "\"device_schedule_host_read_bytes\":"
              << stats.device_schedule_host_read_bytes << ","
              << "\"device_schedule_host_upload_bytes\":"
              << stats.device_schedule_host_upload_bytes << ","
              << "\"device_schedule_run_count\":"
              << stats.device_schedule_run_count << ","
              << "\"device_schedule_us\":"
              << second.timing.device_schedule_us << ","
              << "\"exact_block\":" << exact_block << ","
              << "\"first_complete_minimum_us\":"
              << first.timing.complete_minimum_us << ","
              << "\"first_stage_us\":["
              << first.timing.stage_us[0] << ","
              << first.timing.stage_us[1] << ","
              << first.timing.stage_us[2] << ","
              << first.timing.stage_us[3] << ","
              << first.timing.stage_us[4] << "],"
              << "\"layer_count\":" << stats.layer_count << ","
              << "\"layer_load_count\":" << stats.layer_load_count << ","
              << "\"maps_native_only\":"
              << (first.maps_native_only && second.maps_native_only) << ","
              << "\"max_abs_diff\":" << max_abs_diff << ","
              << "\"native_router\":" << native_router << ","
              << "\"native_router_run_count\":"
              << stats.native_router_run_count << ","
              << "\"oracle_mismatch_count\":" << oracle_mismatch_count << ","
              << "\"persistent_dispatch\":" << persistent_dispatch << ","
              << "\"persistent_dispatch_run_count\":"
              << stats.persistent_dispatch_run_count << ","
              << "\"persistent_workgroup_count\":"
              << second.persistent_workgroup_count << ","
              << "\"program_load_count\":" << stats.program_load_count << ","
              << "\"resident_reuse_pass\":" << reuse_pass << ","
              << "\"resident_weight_bytes\":"
              << stats.resident_weight_bytes << ","
              << "\"run_count\":" << stats.run_count << ","
              << "\"second_complete_minimum_us\":"
              << second.timing.complete_minimum_us << ","
              << "\"steady_state_cap_pass\":"
              << steady_state_cap_pass << ","
              << "\"steady_state_cap_us\":" << steady_state_cap_us
              << "}" << std::endl;
    return reuse_pass ? 0 : 2;
  } catch (const std::exception& exception) {
    std::cerr << exception.what() << '\n';
    return 4;
  }
}
