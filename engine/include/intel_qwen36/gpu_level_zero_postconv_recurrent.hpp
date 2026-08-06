#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace iq36 {

struct GpuLevelZeroPostconvRecurrentInput {
  std::vector<float> conv_output_raw;
  std::vector<float> decay;
  std::vector<float> beta;
  std::vector<float> recurrent_state;
  std::vector<float> z_silu;
  std::vector<float> norm_weight;
  float norm_epsilon = 1.0e-6f;
  float attention_scale = 0.0f;
};

struct GpuLevelZeroPostconvRecurrentRun {
  std::vector<float> q_conv_predelta;
  std::vector<float> k_conv_predelta;
  std::vector<float> v_conv_predelta;
  std::vector<float> attention_output;
  std::vector<float> recurrent_state;
  std::vector<float> final_output;
  std::vector<double> sample_wall_us;
};

class GpuLevelZeroPostconvRecurrentRunner {
 public:
  explicit GpuLevelZeroPostconvRecurrentRunner(
      const std::string& native_module_path,
      std::uint32_t device_id = 0xB080U);
  ~GpuLevelZeroPostconvRecurrentRunner();

  GpuLevelZeroPostconvRecurrentRunner(
      GpuLevelZeroPostconvRecurrentRunner&&) noexcept;
  GpuLevelZeroPostconvRecurrentRunner& operator=(
      GpuLevelZeroPostconvRecurrentRunner&&) noexcept;

  GpuLevelZeroPostconvRecurrentRunner(
      const GpuLevelZeroPostconvRecurrentRunner&) = delete;
  GpuLevelZeroPostconvRecurrentRunner& operator=(
      const GpuLevelZeroPostconvRecurrentRunner&) = delete;

  const std::string& device_name() const;
  GpuLevelZeroPostconvRecurrentRun Run(
      const GpuLevelZeroPostconvRecurrentInput& input,
      int samples = 1);

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace iq36
