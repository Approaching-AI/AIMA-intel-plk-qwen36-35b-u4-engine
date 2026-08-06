#pragma once

#include "intel_qwen36/packed_token_schedule.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace iq36 {

struct PackedTokenStateSnapshot {
  std::array<std::vector<float>, kPackedTokenLayerCount> linear_conv;
  std::array<std::vector<float>, kPackedTokenLayerCount> linear_recurrent;
  std::array<std::vector<float>, kPackedTokenLayerCount> full_k_history;
  std::array<std::vector<float>, kPackedTokenLayerCount> full_v_history;
};

struct PackedTokenLevelZeroProfileRow {
  std::string kernel;
  double device_ms = 0.0;
};

struct PackedTokenLevelZeroTiming {
  double device_ms = 0.0;
  double host_submit_ms = 0.0;
  double wall_ms = 0.0;
  std::uint64_t command_list_record_count = 0;
  std::uint64_t kernel_count = 0;
  std::uint64_t barrier_count = 0;
  std::uint64_t resident_weight_bytes = 0;
  std::uint64_t resident_state_bytes = 0;
  std::vector<PackedTokenLevelZeroProfileRow> kernel_profile;
};

struct PackedTokenLevelZeroConfig {
  std::uint64_t state_capacity_tokens = kPackedTokenAdmissionContextTokens + 32;
  float rms_norm_epsilon = 1.0e-6f;
  std::uint64_t full_head_dim = 256;
  std::uint64_t full_q_head_count = 16;
  std::uint64_t full_kv_head_count = 2;
  std::uint64_t rope_dimension_count = 64;
  std::vector<std::int64_t> rope_sections{11, 11, 10, 0};
  std::uint64_t rope_context_length = 262144;
  float rope_freq_base = 10000000.0f;
  float rope_freq_scale = 1.0f;
  float rope_ext_factor = 0.0f;
  float rope_attn_factor = 1.0f;
  float rope_beta_fast = 32.0f;
  float rope_beta_slow = 1.0f;
  float attention_scale = 0.0625f;
  bool use_int8_block32_kv_gqa = false;
  bool profile_kernel_times = false;
};

class PackedTokenLevelZeroBackend final : public PackedTokenBackend {
 public:
  PackedTokenLevelZeroBackend(std::string model_path,
                              std::string native_module_path,
                              PackedTokenLevelZeroConfig config = {});
  ~PackedTokenLevelZeroBackend() override;

  PackedTokenLevelZeroBackend(const PackedTokenLevelZeroBackend&) = delete;
  PackedTokenLevelZeroBackend& operator=(
      const PackedTokenLevelZeroBackend&) = delete;
  PackedTokenLevelZeroBackend(PackedTokenLevelZeroBackend&&) noexcept;
  PackedTokenLevelZeroBackend& operator=(
      PackedTokenLevelZeroBackend&&) noexcept;

  void LoadState(const PackedTokenStateSnapshot& state);
  PackedTokenStateSnapshot ReadState() const;
  std::vector<float> ReadLogits() const;

  void Compile(const PackedTokenProgram& program) override;
  std::vector<PackedTokenTopKRow> SubmitToken(
      const PackedTokenSubmission& submission) override;

  PackedTokenLevelZeroTiming last_timing() const;
  const std::string& device_name() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace iq36
