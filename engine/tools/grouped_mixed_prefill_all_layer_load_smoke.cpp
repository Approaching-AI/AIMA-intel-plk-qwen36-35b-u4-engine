#include "intel_qwen36/grouped_s8_u4_prefill_runtime.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

constexpr std::array<int, 20> kQ6Layers = {
    0, 1, 2, 3, 4, 7, 10, 13, 16, 19,
    22, 25, 28, 31, 34, 35, 36, 37, 38, 39};
constexpr std::uint64_t kQ4LayerBytes = 478150656;
constexpr std::uint64_t kQ6LayerBytes = 608174080;
constexpr std::uint64_t kExpectedResidentBytes =
    20 * kQ4LayerBytes + 20 * kQ6LayerBytes;

bool IsQ6Layer(int layer) {
  return std::find(kQ6Layers.begin(), kQ6Layers.end(), layer) !=
      kQ6Layers.end();
}

bool MapsAreNativeOnly() {
  std::ifstream maps("/proc/self/maps");
  std::string line;
  while (std::getline(maps, line)) {
    std::transform(line.begin(), line.end(), line.begin(),
                   [](unsigned char value) { return std::tolower(value); });
    if (line.find("libdnnl") != std::string::npos ||
        line.find("openvino") != std::string::npos) {
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
          "usage: mixed-all-layer-load GATEUP DOWN SUPPORT Q6_KERNEL ROOT");
    }
    iq36::GroupedS8U4PrefillProgramConfig program;
    program.gateup_binary = argv[1];
    program.down_binary = argv[2];
    program.kernels = argv[3];
    program.q6_down_kernels = argv[4];
    iq36::GroupedS8U4PrefillRuntime runtime(program);
    const std::filesystem::path root = argv[5];
    std::array<std::uint64_t, 40> handles{};
    for (int layer_index = 0; layer_index < 40; ++layer_index) {
      iq36::GroupedS8U4PrefillLayerConfig layer;
      layer.layer_index = layer_index;
      layer.exact_q4_gateup = true;
      std::ostringstream name;
      name << "layer-" << std::setfill('0') << std::setw(2) << layer_index;
      layer.prep_dir = (root / name.str()).string();
      if (IsQ6Layer(layer_index)) {
        layer.down_kind = iq36::GroupedPrefillDownKind::kQ6U8ExactBlock;
      } else {
        layer.down_kind = iq36::GroupedPrefillDownKind::kQ4U4ExactBlock;
      }
      handles[static_cast<std::size_t>(layer_index)] =
          runtime.LoadLayer(layer);
    }
    const auto stats = runtime.stats();
    const bool handles_pass = [&]() {
      for (std::size_t index = 0; index < handles.size(); ++index) {
        if (handles[index] != index + 1) return false;
      }
      return true;
    }();
    const bool maps_native_only = MapsAreNativeOnly();
    const bool pass = handles_pass && maps_native_only &&
        stats.context_create_count == 1 && stats.program_load_count == 4 &&
        stats.layer_load_count == 40 && stats.layer_count == 40 &&
        stats.run_count == 0 &&
        stats.resident_weight_bytes == kExpectedResidentBytes;
    std::cout << std::boolalpha << "{";
    std::cout << "\"all_layer_load_pass\":" << pass << ",";
    std::cout << "\"context_create_count\":"
              << stats.context_create_count << ",";
    std::cout << "\"device_name\":\"" << runtime.device_name() << "\",";
    std::cout << "\"handles_sequential\":" << handles_pass << ",";
    std::cout << "\"layer_count\":" << stats.layer_count << ",";
    std::cout << "\"layer_load_count\":" << stats.layer_load_count << ",";
    std::cout << "\"maps_native_only\":" << maps_native_only << ",";
    std::cout << "\"program_load_count\":" << stats.program_load_count
              << ",";
    std::cout << "\"q4_layer_count\":20,";
    std::cout << "\"q6_exact_block_layer_count\":20,";
    std::cout << "\"resident_weight_bytes\":"
              << stats.resident_weight_bytes << ",";
    std::cout << "\"run_count\":" << stats.run_count << "}"
              << std::endl;
    return pass ? 0 : 2;
  } catch (const std::exception& exception) {
    std::cerr << "mixed-prefill-all-layer-load: " << exception.what()
              << '\n';
    return 4;
  }
}
