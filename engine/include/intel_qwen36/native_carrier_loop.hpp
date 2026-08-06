#pragma once

#include "intel_qwen36/gpu_q4x8_matvec.hpp"
#include "intel_qwen36/grouped_s8_u4_prefill_runtime.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace iq36 {

struct NativeCarrierProgramConfig {
  GroupedS8U4PrefillProgramConfig grouped_prefill;
  std::string q6_device_substring;
  std::string q6_opencl_source_path;
};

struct NativeQ6CarrierLayerConfig {
  int layer_index = -1;
  std::vector<std::uint8_t> raw_weights;
  std::uint64_t rows_per_expert = 0;
  std::uint64_t blocks_per_row = 0;
  std::uint64_t expert_count = 0;
  std::uint64_t rows_per_tile = 16;
};

struct NativeCarrierRuntimeStats {
  GroupedS8U4PrefillRuntimeStats grouped_prefill;
  std::uint64_t q6_context_create_count = 0;
  std::uint64_t q6_layer_load_count = 0;
  std::uint64_t q6_layer_count = 0;
  std::uint64_t q6_run_count = 0;
  std::uint64_t q6_resident_weight_bytes = 0;
};

class NativeCarrierLayerRuntime {
 public:
  explicit NativeCarrierLayerRuntime(const NativeCarrierProgramConfig& config);
  ~NativeCarrierLayerRuntime();

  NativeCarrierLayerRuntime(const NativeCarrierLayerRuntime&) = delete;
  NativeCarrierLayerRuntime& operator=(const NativeCarrierLayerRuntime&) =
      delete;
  NativeCarrierLayerRuntime(NativeCarrierLayerRuntime&&) noexcept;
  NativeCarrierLayerRuntime& operator=(NativeCarrierLayerRuntime&&) noexcept;

  void LoadGroupedPrefillLayer(
      const GroupedS8U4PrefillLayerConfig& config);
  void LoadQ6Layer(const NativeQ6CarrierLayerConfig& config);
  GroupedS8U4PrefillRun RunGroupedPrefillLayer(
      int layer_index,
      const GroupedS8U4PrefillInput& input);
  GpuQ6KMatvecRun RunQ6Layer(int layer_index,
                             const GpuQ8KInputPlanes& input,
                             int repeat);

  const std::string& grouped_prefill_device_name() const;
  const std::string& q6_device_name() const;
  NativeCarrierRuntimeStats stats() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

struct NativeGroupedPrefillStep {
  int layer_index = -1;
  GroupedS8U4PrefillInput input;
};

struct NativeQ6CarrierStep {
  int layer_index = -1;
  GpuQ8KInputPlanes input;
  int repeat = 1;
};

struct NativeGroupedPrefillSequenceResult {
  std::vector<int> layer_indices;
  std::vector<GroupedS8U4PrefillRun> runs;
};

struct NativeQ6CarrierSequenceResult {
  std::vector<int> layer_indices;
  std::vector<GpuQ6KMatvecRun> runs;
};

class NativeCarrierLoop {
 public:
  explicit NativeCarrierLoop(NativeCarrierLayerRuntime& runtime);

  NativeGroupedPrefillSequenceResult RunGroupedPrefill(
      const std::vector<NativeGroupedPrefillStep>& steps);
  NativeQ6CarrierSequenceResult RunQ6(
      const std::vector<NativeQ6CarrierStep>& steps);

 private:
  NativeCarrierLayerRuntime& runtime_;
};

}  // namespace iq36
