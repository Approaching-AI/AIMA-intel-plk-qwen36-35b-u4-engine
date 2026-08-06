#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace iq36 {

struct GpuVulkanPostconvRecurrentInput {
  std::vector<float> conv_output_raw;
  std::vector<float> decay;
  std::vector<float> beta;
  std::vector<float> recurrent_state;
  std::vector<float> z_silu;
  std::vector<float> norm_weight;
  float norm_epsilon = 1.0e-6f;
  float attention_scale = 0.0f;
};

struct GpuVulkanPostconvRecurrentRun {
  std::vector<float> q_conv_predelta;
  std::vector<float> k_conv_predelta;
  std::vector<float> v_conv_predelta;
  std::vector<float> attention_output;
  std::vector<float> recurrent_state;
  std::vector<float> final_output;
  std::vector<double> sample_wall_us;
};

class GpuVulkanPostconvRecurrentRunner {
 public:
  GpuVulkanPostconvRecurrentRunner(
      const std::string& postconv_spirv_path,
      const std::string& recurrent_spirv_path,
      const std::string& device_substring = "PTL");
  ~GpuVulkanPostconvRecurrentRunner();

  GpuVulkanPostconvRecurrentRunner(
      GpuVulkanPostconvRecurrentRunner&&) noexcept;
  GpuVulkanPostconvRecurrentRunner& operator=(
      GpuVulkanPostconvRecurrentRunner&&) noexcept;

  GpuVulkanPostconvRecurrentRunner(
      const GpuVulkanPostconvRecurrentRunner&) = delete;
  GpuVulkanPostconvRecurrentRunner& operator=(
      const GpuVulkanPostconvRecurrentRunner&) = delete;

  const std::string& device_name() const;
  GpuVulkanPostconvRecurrentRun Run(
      const GpuVulkanPostconvRecurrentInput& input,
      int samples = 1);

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace iq36
