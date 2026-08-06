#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <memory>
#include <string>
#include <vector>

namespace iq36 {

enum class GroupedPrefillDownKind {
  kQ4U4,
  kQ4U4F32Contribution,
  kQ4U4ExactBlock,
  kQ6U8Surrogate,
  kQ6U8ExactPer16,
  kQ6U8ExactBlock,
};

struct GroupedS8U4PrefillConfig {
  std::string prep_dir;
  std::string gateup_binary;
  std::string down_binary;
  std::string kernels;
  std::string input;
  std::string topk;
  std::size_t topk_stride = 0;
  std::string oracle;
  std::string router_weights;
  std::string down_oracle;
  std::string moe_oracle;
  int warmup = 3;
  int repeat = 11;
  double kernel_cap_us = 9526.177;
  bool schedule_probe_only = false;
  // Fixed route-control mode for the admitted M8/N32+N64 persistent source.
  // This is not a tunable product option.
  bool m8_source_preflight = false;
};

struct GroupedS8U4PrefillProgramConfig {
  std::string gateup_binary;
  std::string down_binary;
  std::string router_binary;
  std::string kernels;
  std::string q6_down_kernels;
  // The gate/up and down binaries consume device-built logical task
  // coordinates from a fixed physical workgroup grid.
  bool persistent_dispatch = false;
};

struct GroupedS8U4PrefillLayerConfig {
  int layer_index = -1;
  std::string prep_dir;
  std::string router_weights;
  bool exact_q4_gateup = false;
  GroupedPrefillDownKind down_kind = GroupedPrefillDownKind::kQ4U4;
};

struct GroupedS8U4PrefillInput {
  std::vector<float> hidden_states;
  std::vector<std::uint8_t> topk;
  std::size_t topk_stride = 0;
  std::vector<float> router_weights;
  // Diagnostic-only source-order SwiGLU oracle. Empty in native product runs.
  std::vector<float> swiglu_override_source_order;
  // Diagnostic-only unweighted source-order down oracle.
  std::vector<float> down_override_source_order;
  int warmup = 0;
  int repeat = 1;
  bool capture_intermediates = false;
  bool execute_down = true;
  // Build offsets/maps/task coordinates from the supplied top-8/router rows
  // on device. This removes CPU schedule construction and bulk schedule
  // uploads. With a persistent-dispatch program, grouped kernels consume the
  // resulting logical task list directly and no metadata is read by the host.
  bool device_schedule = false;
  // Produce top-8 IDs and normalized weights from hidden_states with the
  // resident layer router. Requires device_schedule, router_binary, and the
  // layer router_weights path; topk/router_weights inputs are then ignored.
  bool native_router = false;
};

struct GroupedS8U4PrefillTiming {
  double input_upload_us = 0.0;
  double schedule_prepare_us = 0.0;
  double schedule_upload_us = 0.0;
  double schedule_setup_us = 0.0;
  double device_schedule_us = 0.0;
  std::array<double, 5> stage_us{};
  std::vector<double> samples_us;
  double minimum_us = 0.0;
  double median_us = 0.0;
  double mean_us = 0.0;
  double complete_minimum_us = 0.0;
};

struct GroupedS8U4PrefillRun {
  std::vector<float> output;
  std::vector<std::uint16_t> grouped_swiglu_f16;
  std::vector<float> grouped_swiglu_f32;
  std::vector<std::int8_t> grouped_down_q8;
  std::vector<float> grouped_down_scales;
  std::vector<std::uint16_t> grouped_contributions_f16;
  std::vector<float> grouped_contributions_f32;
  std::vector<std::int32_t> inverse_map;
  std::size_t active_experts = 0;
  std::size_t max_group_size = 0;
  std::size_t native_global_y = 0;
  std::size_t gateup_work_tile_count = 0;
  std::size_t q6_work_tile_count = 0;
  GroupedPrefillDownKind down_kind = GroupedPrefillDownKind::kQ4U4;
  bool device_schedule = false;
  bool native_router = false;
  bool persistent_dispatch = false;
  std::size_t persistent_workgroup_count = 0;
  bool maps_native_only = false;
  GroupedS8U4PrefillTiming timing;
};

struct GroupedS8U4PrefillRuntimeStats {
  std::uint64_t context_create_count = 0;
  std::uint64_t program_load_count = 0;
  std::uint64_t layer_load_count = 0;
  std::uint64_t layer_count = 0;
  std::uint64_t run_count = 0;
  std::uint64_t resident_weight_bytes = 0;
  std::uint64_t device_schedule_run_count = 0;
  std::uint64_t native_router_run_count = 0;
  std::uint64_t persistent_dispatch_run_count = 0;
  std::uint64_t device_schedule_host_upload_bytes = 0;
  std::uint64_t device_schedule_host_read_bytes = 0;
};

class GroupedS8U4PrefillRuntime {
 public:
  explicit GroupedS8U4PrefillRuntime(
      const GroupedS8U4PrefillProgramConfig& config);
  ~GroupedS8U4PrefillRuntime();

  GroupedS8U4PrefillRuntime(const GroupedS8U4PrefillRuntime&) = delete;
  GroupedS8U4PrefillRuntime& operator=(
      const GroupedS8U4PrefillRuntime&) = delete;
  GroupedS8U4PrefillRuntime(GroupedS8U4PrefillRuntime&&) noexcept;
  GroupedS8U4PrefillRuntime& operator=(
      GroupedS8U4PrefillRuntime&&) noexcept;

  std::uint64_t LoadLayer(const GroupedS8U4PrefillLayerConfig& config);
  GroupedS8U4PrefillRun RunLayer(
      std::uint64_t layer_handle,
      const GroupedS8U4PrefillInput& input);

  const std::string& device_name() const;
  const std::string& driver_version() const;
  GroupedS8U4PrefillRuntimeStats stats() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

int RunGroupedS8U4Prefill(const GroupedS8U4PrefillConfig& config,
                          std::ostream& output,
                          std::ostream& error);

int RunGroupedS8U4PrefillCommandLine(int argc,
                                     char** argv,
                                     std::ostream& output,
                                     std::ostream& error);

}  // namespace iq36
