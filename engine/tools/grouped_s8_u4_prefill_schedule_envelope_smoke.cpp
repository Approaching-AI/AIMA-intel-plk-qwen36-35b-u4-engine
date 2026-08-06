#include "intel_qwen36/grouped_s8_u4_prefill_runtime.hpp"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

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

struct LayerRow {
  int layer = -1;
  std::size_t active_experts = 0;
  std::size_t max_group_size = 0;
  double full_complete_minimum_us = 0.0;
  double full_kernel_minimum_us = 0.0;
  double shell_complete_minimum_us = 0.0;
  double shell_kernel_minimum_us = 0.0;
};

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 9) {
      throw std::invalid_argument(
          "usage: schedule-envelope PREP GATEUP DOWN KERNEL INPUT "
          "SCHEDULE_DIR ROUTER_WEIGHTS REPEAT");
    }
    constexpr std::size_t kTokenCount = 1024;
    constexpr std::size_t kHiddenSize = 2048;
    constexpr std::size_t kAssignmentCount = 8192;
    const int repeat = std::stoi(argv[8]);
    if (repeat <= 0) throw std::invalid_argument("repeat must be positive");

    iq36::GroupedS8U4PrefillProgramConfig program;
    program.gateup_binary = argv[2];
    program.down_binary = argv[3];
    program.kernels = argv[4];
    iq36::GroupedS8U4PrefillRuntime runtime(program);
    const auto handle = runtime.LoadLayer({27, argv[1]});

    iq36::GroupedS8U4PrefillInput input;
    input.hidden_states = ReadVector<float>(
        argv[5], kTokenCount * kHiddenSize);
    input.router_weights = ReadVector<float>(argv[7], kAssignmentCount);
    input.topk_stride = 8 * sizeof(std::int32_t);
    input.warmup = 1;
    input.repeat = repeat;
    const std::filesystem::path schedule_dir = argv[6];
    std::vector<LayerRow> rows;
    rows.reserve(40);
    for (int layer = 0; layer < 40; ++layer) {
      input.topk = ReadBytes(
          schedule_dir / ("layer-" + std::to_string(layer) + ".topk.i32"));
      input.execute_down = true;
      const auto full = runtime.RunLayer(handle, input);
      input.execute_down = false;
      const auto shell = runtime.RunLayer(handle, input);
      if (full.active_experts != shell.active_experts ||
          full.max_group_size != shell.max_group_size) {
        throw std::runtime_error("paired schedule shape changed");
      }
      rows.push_back({layer, full.active_experts, full.max_group_size,
                      full.timing.complete_minimum_us,
                      full.timing.minimum_us,
                      shell.timing.complete_minimum_us,
                      shell.timing.minimum_us});
    }
    const auto stats = runtime.stats();
    const bool maps_native_only = MapsAreNativeOnly();
    const bool pass = maps_native_only && rows.size() == 40 &&
        stats.context_create_count == 1 && stats.program_load_count == 3 &&
        stats.layer_load_count == 1 && stats.layer_count == 1 &&
        stats.run_count == 80 && stats.resident_weight_bytes == 541065216;
    std::cout << std::boolalpha << std::setprecision(12) << "{";
    std::cout << "\"context_create_count\":"
              << stats.context_create_count << ",";
    std::cout << "\"device_name\":\"" << runtime.device_name() << "\",";
    std::cout << "\"layer_load_count\":" << stats.layer_load_count << ",";
    std::cout << "\"maps_native_only\":" << maps_native_only << ",";
    std::cout << "\"paired_layer_count\":" << rows.size() << ",";
    std::cout << "\"per_layer\":[";
    for (std::size_t index = 0; index < rows.size(); ++index) {
      if (index != 0) std::cout << ",";
      const auto& row = rows[index];
      std::cout << "{\"active_experts\":" << row.active_experts << ",";
      std::cout << "\"full_complete_minimum_us\":"
                << row.full_complete_minimum_us << ",";
      std::cout << "\"full_kernel_minimum_us\":"
                << row.full_kernel_minimum_us << ",";
      std::cout << "\"layer\":" << row.layer << ",";
      std::cout << "\"max_group_size\":" << row.max_group_size << ",";
      std::cout << "\"shell_complete_minimum_us\":"
                << row.shell_complete_minimum_us << ",";
      std::cout << "\"shell_kernel_minimum_us\":"
                << row.shell_kernel_minimum_us << "}";
    }
    std::cout << "],";
    std::cout << "\"program_load_count\":" << stats.program_load_count
              << ",";
    std::cout << "\"resident_weight_bytes\":"
              << stats.resident_weight_bytes << ",";
    std::cout << "\"run_count\":" << stats.run_count << ",";
    std::cout << "\"schedule_envelope_pass\":" << pass << "}"
              << std::endl;
    return pass ? 0 : 2;
  } catch (const std::exception& exception) {
    std::cerr << "grouped-prefill-schedule-envelope: " << exception.what()
              << '\n';
    return 4;
  }
}
