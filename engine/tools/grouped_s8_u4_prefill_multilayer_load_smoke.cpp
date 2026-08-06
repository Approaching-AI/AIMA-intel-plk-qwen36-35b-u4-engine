#include "intel_qwen36/grouped_s8_u4_prefill_runtime.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

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
    if (argc != 6) {
      throw std::invalid_argument(
          "usage: multilayer-load-smoke GATEUP DOWN KERNEL LAYER5 LAYER27");
    }
    iq36::GroupedS8U4PrefillProgramConfig program;
    program.gateup_binary = argv[1];
    program.down_binary = argv[2];
    program.kernels = argv[3];
    iq36::GroupedS8U4PrefillRuntime runtime(program);
    const auto layer5 = runtime.LoadLayer({5, argv[4]});
    const auto layer27 = runtime.LoadLayer({27, argv[5]});
    const auto stats = runtime.stats();
    const bool maps_native_only = MapsAreNativeOnly();
    const bool pass = layer5 != 0 && layer27 != 0 && layer5 != layer27 &&
        stats.context_create_count == 1 && stats.program_load_count == 3 &&
        stats.layer_load_count == 2 && stats.layer_count == 2 &&
        stats.run_count == 0 &&
        stats.resident_weight_bytes == 2ULL * 541065216ULL &&
        maps_native_only;
    std::cout << std::boolalpha << "{"
              << "\"context_create_count\":"
              << stats.context_create_count << ","
              << "\"device_name\":\"" << runtime.device_name() << "\","
              << "\"layer_count\":" << stats.layer_count << ","
              << "\"layer_load_count\":" << stats.layer_load_count << ","
              << "\"maps_native_only\":" << maps_native_only << ","
              << "\"multilayer_load_pass\":" << pass << ","
              << "\"program_load_count\":" << stats.program_load_count << ","
              << "\"resident_weight_bytes\":"
              << stats.resident_weight_bytes << ","
              << "\"run_count\":" << stats.run_count << "}" << std::endl;
    return pass ? 0 : 2;
  } catch (const std::exception& exception) {
    std::cerr << exception.what() << '\n';
    return 4;
  }
}
