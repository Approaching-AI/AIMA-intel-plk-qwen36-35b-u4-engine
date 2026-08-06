#include "intel_qwen36/native_carrier_loop.hpp"
#include "intel_qwen36/resident_harness.hpp"

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

std::string ReadText(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("could not open OpenCL source: " + path);
  input.seekg(0, std::ios::end);
  const auto size = input.tellg();
  if (size < 0) throw std::runtime_error("could not size OpenCL source: " + path);
  input.seekg(0, std::ios::beg);
  std::string text(static_cast<std::size_t>(size), '\0');
  input.read(text.data(), static_cast<std::streamsize>(text.size()));
  if (!input) throw std::runtime_error("could not read OpenCL source: " + path);
  return text;
}

void RequireLayerIndex(int layer_index) {
  if (layer_index < 0 || layer_index >= 40) {
    throw std::invalid_argument("native carrier layer index must be in [0, 40)");
  }
}

}  // namespace

namespace iq36 {

class NativeCarrierLayerRuntime::Impl {
 public:
  explicit Impl(const NativeCarrierProgramConfig& config)
      : grouped_(config.grouped_prefill),
        q6_(config.q6_device_substring,
            ReadText(config.q6_opencl_source_path)) {
    if (config.q6_device_substring.empty() ||
        config.q6_opencl_source_path.empty()) {
      throw std::invalid_argument(
          "native Q6 device and OpenCL source path are required");
    }
    stats_.q6_context_create_count = 1;
  }

  void LoadGroupedPrefillLayer(
      const GroupedS8U4PrefillLayerConfig& config) {
    RequireLayerIndex(config.layer_index);
    for (const auto& layer : grouped_layers_) {
      if (layer.layer_index == config.layer_index) {
        throw std::invalid_argument(
            "native grouped prefill layer is already loaded");
      }
    }
    const auto handle = grouped_.LoadLayer(config);
    grouped_layers_.push_back({config.layer_index, handle});
  }

  void LoadQ6Layer(const NativeQ6CarrierLayerConfig& config) {
    RequireLayerIndex(config.layer_index);
    if (config.rows_per_expert == 0 || config.blocks_per_row == 0 ||
        config.expert_count == 0 || config.rows_per_tile == 0) {
      throw std::invalid_argument("native Q6 carrier shape is required");
    }
    for (const auto& layer : q6_layers_) {
      if (layer.layer_index == config.layer_index) {
        throw std::invalid_argument("native Q6 layer is already loaded");
      }
    }
    const auto handle = q6_.UploadSelectedRawQ6KRowstripe(
        config.raw_weights, config.rows_per_expert, config.blocks_per_row,
        config.expert_count, config.rows_per_tile);
    q6_layers_.push_back({config.layer_index, handle});
    ++stats_.q6_layer_load_count;
    stats_.q6_layer_count = q6_layers_.size();
    stats_.q6_resident_weight_bytes += config.raw_weights.size();
  }

  GroupedS8U4PrefillRun RunGroupedPrefillLayer(
      int layer_index,
      const GroupedS8U4PrefillInput& input) {
    return grouped_.RunLayer(GroupedHandle(layer_index), input);
  }

  GpuQ6KMatvecRun RunQ6Layer(int layer_index,
                             const GpuQ8KInputPlanes& input,
                             int repeat) {
    if (repeat <= 0) {
      throw std::invalid_argument("native Q6 repeat must be positive");
    }
    auto run = q6_.RunResidentRawQ6K(Q6Handle(layer_index), input, repeat);
    ++stats_.q6_run_count;
    return run;
  }

  const std::string& grouped_prefill_device_name() const {
    return grouped_.device_name();
  }
  const std::string& q6_device_name() const { return q6_.device_name(); }

  NativeCarrierRuntimeStats stats() const {
    auto result = stats_;
    result.grouped_prefill = grouped_.stats();
    return result;
  }

 private:
  struct LayerHandle {
    int layer_index = -1;
    std::uint64_t handle = 0;
  };

  std::uint64_t GroupedHandle(int layer_index) const {
    RequireLayerIndex(layer_index);
    for (const auto& layer : grouped_layers_) {
      if (layer.layer_index == layer_index) return layer.handle;
    }
    throw std::invalid_argument("native grouped prefill layer is not loaded");
  }

  std::uint64_t Q6Handle(int layer_index) const {
    RequireLayerIndex(layer_index);
    for (const auto& layer : q6_layers_) {
      if (layer.layer_index == layer_index) return layer.handle;
    }
    throw std::invalid_argument("native Q6 layer is not loaded");
  }

  GroupedS8U4PrefillRuntime grouped_;
  GpuQ4X8MatvecRunner q6_;
  std::vector<LayerHandle> grouped_layers_;
  std::vector<LayerHandle> q6_layers_;
  NativeCarrierRuntimeStats stats_;
};

NativeCarrierLayerRuntime::NativeCarrierLayerRuntime(
    const NativeCarrierProgramConfig& config)
    : impl_(std::make_unique<Impl>(config)) {}

NativeCarrierLayerRuntime::~NativeCarrierLayerRuntime() = default;
NativeCarrierLayerRuntime::NativeCarrierLayerRuntime(
    NativeCarrierLayerRuntime&&) noexcept = default;
NativeCarrierLayerRuntime& NativeCarrierLayerRuntime::operator=(
    NativeCarrierLayerRuntime&&) noexcept = default;

void NativeCarrierLayerRuntime::LoadGroupedPrefillLayer(
    const GroupedS8U4PrefillLayerConfig& config) {
  impl_->LoadGroupedPrefillLayer(config);
}

void NativeCarrierLayerRuntime::LoadQ6Layer(
    const NativeQ6CarrierLayerConfig& config) {
  impl_->LoadQ6Layer(config);
}

GroupedS8U4PrefillRun NativeCarrierLayerRuntime::RunGroupedPrefillLayer(
    int layer_index,
    const GroupedS8U4PrefillInput& input) {
  return impl_->RunGroupedPrefillLayer(layer_index, input);
}

GpuQ6KMatvecRun NativeCarrierLayerRuntime::RunQ6Layer(
    int layer_index,
    const GpuQ8KInputPlanes& input,
    int repeat) {
  return impl_->RunQ6Layer(layer_index, input, repeat);
}

const std::string& NativeCarrierLayerRuntime::grouped_prefill_device_name()
    const {
  return impl_->grouped_prefill_device_name();
}

const std::string& NativeCarrierLayerRuntime::q6_device_name() const {
  return impl_->q6_device_name();
}

NativeCarrierRuntimeStats NativeCarrierLayerRuntime::stats() const {
  return impl_->stats();
}

int parameterized_layer_count() {
  return model_contract().layers;
}

}  // namespace iq36
