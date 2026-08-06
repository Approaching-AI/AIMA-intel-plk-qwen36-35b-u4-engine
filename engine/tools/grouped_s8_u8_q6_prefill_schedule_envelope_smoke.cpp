#include "intel_qwen36/grouped_s8_u4_prefill_runtime.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::array<int, 20> kQ6Layers = {
    0, 1, 2, 3, 4, 7, 10, 13, 16, 19,
    22, 25, 28, 31, 34, 35, 36, 37, 38, 39};

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
  std::size_t task_count = 0;
  double complete_minimum_us = 0.0;
  double kernel_minimum_us = 0.0;
  double schedule_prepare_us = 0.0;
  double schedule_upload_us = 0.0;
  double schedule_setup_us = 0.0;
  std::array<double, 5> stage_us{};
};

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 10) {
      throw std::invalid_argument(
          "usage: q6-schedule-envelope PREP GATEUP DOWN SUPPORT Q6_KERNEL "
          "INPUT SCHEDULE_DIR ROUTER_WEIGHTS REPEAT");
    }
    constexpr std::size_t kTokenCount = 1024;
    constexpr std::size_t kHiddenSize = 2048;
    constexpr std::size_t kAssignmentCount = 8192;
    const int repeat = std::stoi(argv[9]);
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
    input.router_weights = ReadVector<float>(argv[8], kAssignmentCount);
    input.topk_stride = 8 * sizeof(std::int32_t);
    input.warmup = 1;
    input.repeat = repeat;
    const std::filesystem::path schedule_dir = argv[7];
    std::vector<LayerRow> rows;
    rows.reserve(kQ6Layers.size());
    for (const int schedule_layer : kQ6Layers) {
      input.topk = ReadBytes(schedule_dir /
          ("layer-" + std::to_string(schedule_layer) + ".topk.i32"));
      const auto run = runtime.RunLayer(handle, input);
      rows.push_back({schedule_layer, run.active_experts,
          run.max_group_size, run.q6_work_tile_count,
          run.timing.complete_minimum_us, run.timing.minimum_us,
          run.timing.schedule_prepare_us, run.timing.schedule_upload_us,
          run.timing.schedule_setup_us, run.timing.stage_us});
    }
    const auto stats = runtime.stats();
    const bool maps_native_only = MapsAreNativeOnly();
    const bool pass = maps_native_only && rows.size() == kQ6Layers.size() &&
        stats.context_create_count == 1 && stats.program_load_count == 4 &&
        stats.layer_load_count == 1 && stats.layer_count == 1 &&
        stats.run_count == kQ6Layers.size() &&
        stats.resident_weight_bytes == 645922816;
    const double complete_sum_us = std::accumulate(
        rows.begin(), rows.end(), 0.0,
        [](double sum, const LayerRow& row) {
          return sum + row.complete_minimum_us;
        });
    const double kernel_sum_us = std::accumulate(
        rows.begin(), rows.end(), 0.0,
        [](double sum, const LayerRow& row) {
          return sum + row.kernel_minimum_us;
        });

    std::cout << std::boolalpha << std::setprecision(12) << "{";
    std::cout << "\"complete_sum_us\":" << complete_sum_us << ",";
    std::cout << "\"context_create_count\":"
              << stats.context_create_count << ",";
    std::cout << "\"device_name\":\"" << runtime.device_name() << "\",";
    std::cout << "\"kernel_sum_us\":" << kernel_sum_us << ",";
    std::cout << "\"maps_native_only\":" << maps_native_only << ",";
    std::cout << "\"per_layer\":[";
    for (std::size_t index = 0; index < rows.size(); ++index) {
      if (index != 0) std::cout << ",";
      const auto& row = rows[index];
      std::cout << "{\"active_experts\":" << row.active_experts << ",";
      std::cout << "\"complete_minimum_us\":"
                << row.complete_minimum_us << ",";
      std::cout << "\"kernel_minimum_us\":" << row.kernel_minimum_us
                << ",";
      std::cout << "\"layer\":" << row.layer << ",";
      std::cout << "\"max_group_size\":" << row.max_group_size << ",";
      std::cout << "\"schedule_prepare_us\":"
                << row.schedule_prepare_us << ",";
      std::cout << "\"schedule_setup_us\":" << row.schedule_setup_us
                << ",";
      std::cout << "\"schedule_upload_us\":" << row.schedule_upload_us
                << ",";
      std::cout << "\"stage_us\":{";
      std::cout << "\"down\":" << row.stage_us[3] << ",";
      std::cout << "\"down_quantize\":" << row.stage_us[2] << ",";
      std::cout << "\"gateup\":" << row.stage_us[1] << ",";
      std::cout << "\"gather\":" << row.stage_us[0] << ",";
      std::cout << "\"scatter\":" << row.stage_us[4] << "},";
      std::cout << "\"task_count\":" << row.task_count << "}";
    }
    std::cout << "],";
    std::cout << "\"program_load_count\":" << stats.program_load_count
              << ",";
    std::cout << "\"q6_layer_count\":" << rows.size() << ",";
    std::cout << "\"q6_schedule_envelope_pass\":" << pass << ",";
    std::cout << "\"resident_weight_bytes\":"
              << stats.resident_weight_bytes << ",";
    std::cout << "\"run_count\":" << stats.run_count << "}"
              << std::endl;
    return pass ? 0 : 2;
  } catch (const std::exception& exception) {
    std::cerr << "q6-prefill-schedule-envelope: " << exception.what()
              << '\n';
    return 4;
  }
}
