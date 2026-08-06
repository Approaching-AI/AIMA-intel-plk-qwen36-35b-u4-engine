#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace iq36 {

enum class GpuQ4X8KernelVariant {
  kGroup8Serial,
  kRowlaneParallel,
};

struct GpuQ4X8MatvecTiming {
  double min_us = 0.0;
  double mean_us = 0.0;
  double effective_packed_gb_s = 0.0;
  std::uint64_t global_work_items = 0;
  std::uint64_t rows_per_work_item = 0;
  std::uint64_t input_setup_wall_ns = 0;
  std::uint64_t input_write_wall_ns = 0;
  std::uint64_t kernel_setup_wall_ns = 0;
  std::uint64_t kernel_wait_wall_ns = 0;
  std::uint64_t kernel_enqueue_wall_ns = 0;
  std::uint64_t kernel_finish_wall_ns = 0;
  std::uint64_t event_profile_wall_ns = 0;
  std::uint64_t queue_drain_cleanup_wall_ns = 0;
  std::uint64_t output_read_wall_ns = 0;
};

struct GpuQ4X8MatvecRun {
  std::vector<float> output;
  std::uint64_t output_handle = 0;
  GpuQ4X8MatvecTiming timing;
};

struct GpuDeviceQ8Q4X8MatvecTiming {
  double q8_quantize_min_us = 0.0;
  double q8_quantize_mean_us = 0.0;
  GpuQ4X8MatvecTiming matvec;
  double shell_sum_min_us = 0.0;
  double shell_sum_mean_us = 0.0;
  std::uint64_t q8_quantize_global_work_items = 0;
};

struct GpuDeviceQ8Q4X8MatvecRun {
  std::vector<float> output;
  std::uint64_t output_handle = 0;
  bool output_host_valid = true;
  GpuDeviceQ8Q4X8MatvecTiming timing;
};

struct GpuQ6KMatvecRun {
  std::vector<float> output;
  std::uint64_t output_handle = 0;
  GpuQ4X8MatvecTiming timing;
};

struct GpuDeviceQ8Q6KMatvecTiming {
  double q8_quantize_min_us = 0.0;
  double q8_quantize_mean_us = 0.0;
  GpuQ4X8MatvecTiming matvec;
  double shell_sum_min_us = 0.0;
  double shell_sum_mean_us = 0.0;
  std::uint64_t q8_quantize_global_work_items = 0;
};

struct GpuDeviceQ8Q6KMatvecRun {
  std::vector<float> output;
  std::uint64_t output_handle = 0;
  bool output_host_valid = true;
  GpuDeviceQ8Q6KMatvecTiming timing;
};

struct GpuQ6KSelectedSharedMatvecRun {
  std::vector<float> selected_output;
  std::vector<float> shared_output;
  std::uint64_t selected_output_handle = 0;
  std::uint64_t shared_output_handle = 0;
  GpuQ4X8MatvecTiming timing;
};

struct GpuQ4X8SelectedSharedMatvecRun {
  std::vector<float> selected_output;
  std::vector<float> shared_output;
  std::uint64_t selected_output_handle = 0;
  std::uint64_t shared_output_handle = 0;
  GpuQ4X8MatvecTiming timing;
};

struct GpuTopKRow {
  std::int32_t token_id = 0;
  float value = 0.0f;
};

struct GpuQ6KTopKTiming {
  GpuQ4X8MatvecTiming matvec;
  double partial_topk_min_us = 0.0;
  double partial_topk_mean_us = 0.0;
  double shell_sum_min_us = 0.0;
  double shell_sum_mean_us = 0.0;
  std::uint64_t partial_topk_global_work_items = 0;
};

struct GpuQ6KTopKRun {
  std::vector<GpuTopKRow> topk;
  GpuQ6KTopKTiming timing;
};

struct GpuF32TopKRun {
  std::vector<GpuTopKRow> topk;
  GpuQ4X8MatvecTiming timing;
};

struct GpuQ4X8ConvHandoffTiming {
  double q8_quantize_min_us = 0.0;
  double q8_quantize_mean_us = 0.0;
  GpuQ4X8MatvecTiming matvec;
  double conv_min_us = 0.0;
  double conv_mean_us = 0.0;
  double shell_sum_min_us = 0.0;
  double shell_sum_mean_us = 0.0;
  std::uint64_t q8_quantize_global_work_items = 0;
  std::uint64_t conv_global_work_items = 0;
};

struct GpuQ4X8ConvHandoffRun {
  std::vector<float> qkv_mixed;
  std::vector<float> conv_output_raw;
  std::uint64_t conv_output_handle = 0;
  std::vector<float> conv_state;
  GpuQ4X8ConvHandoffTiming timing;
};

struct GpuQ4X8SwiGluHandoffTiming {
  GpuQ4X8MatvecTiming matvec;
  double swiglu_min_us = 0.0;
  double swiglu_mean_us = 0.0;
  double shell_sum_min_us = 0.0;
  double shell_sum_mean_us = 0.0;
  std::uint64_t swiglu_global_work_items = 0;
};

struct GpuQ4X8SwiGluHandoffRun {
  std::vector<float> swiglu;
  GpuQ4X8SwiGluHandoffTiming timing;
};

struct GpuQ4X8SwiGluQ6DownHandoffTiming {
  GpuQ4X8MatvecTiming gate_up;
  double swiglu_min_us = 0.0;
  double swiglu_mean_us = 0.0;
  double q8_quantize_min_us = 0.0;
  double q8_quantize_mean_us = 0.0;
  GpuQ4X8MatvecTiming down;
  double shell_sum_min_us = 0.0;
  double shell_sum_mean_us = 0.0;
  std::uint64_t swiglu_global_work_items = 0;
  std::uint64_t q8_quantize_global_work_items = 0;
};

struct GpuQ4X8SwiGluQ6DownHandoffRun {
  std::vector<float> down;
  GpuQ4X8SwiGluQ6DownHandoffTiming timing;
};

struct GpuQ4X8SwiGluQ4F32DownHandoffTiming {
  GpuQ4X8MatvecTiming gate_up;
  double swiglu_min_us = 0.0;
  double swiglu_mean_us = 0.0;
  GpuQ4X8MatvecTiming down;
  double shell_sum_min_us = 0.0;
  double shell_sum_mean_us = 0.0;
  std::uint64_t swiglu_global_work_items = 0;
};

struct GpuQ4X8SwiGluQ4F32DownHandoffRun {
  std::vector<float> down;
  std::uint64_t down_handle = 0;
  GpuQ4X8SwiGluQ4F32DownHandoffTiming timing;
};

struct GpuQ4X8ResidualRmsNormHandoffTiming {
  GpuQ4X8MatvecTiming matvec;
  double q8_quantize_min_us = 0.0;
  double q8_quantize_mean_us = 0.0;
  double residual_min_us = 0.0;
  double residual_mean_us = 0.0;
  double rmsnorm_min_us = 0.0;
  double rmsnorm_mean_us = 0.0;
  double shell_sum_min_us = 0.0;
  double shell_sum_mean_us = 0.0;
  std::uint64_t handoff_setup_wall_ns = 0;
  std::uint64_t handoff_residual_input_write_wall_ns = 0;
  std::uint64_t handoff_matvec_wall_ns = 0;
  std::uint64_t handoff_residual_rmsnorm_args_wall_ns = 0;
  std::uint64_t handoff_residual_rmsnorm_enqueue_finish_wall_ns = 0;
  std::uint64_t handoff_event_profile_wall_ns = 0;
  std::uint64_t handoff_residual_read_wall_ns = 0;
  std::uint64_t handoff_normalized_read_wall_ns = 0;
  std::uint64_t handoff_alias_wall_ns = 0;
  std::uint64_t handoff_release_wall_ns = 0;
  std::uint64_t q8_quantize_global_work_items = 0;
  std::uint64_t residual_global_work_items = 0;
  std::uint64_t rmsnorm_global_work_items = 0;
};

struct GpuQ4X8ResidualRmsNormHandoffRun {
  std::vector<float> residual;
  std::vector<float> normalized;
  std::uint64_t residual_handle = 0;
  std::uint64_t normalized_handle = 0;
  GpuQ4X8ResidualRmsNormHandoffTiming timing;
};

struct GpuLinearAttentionPostConvPrepTiming {
  double silu_split_min_us = 0.0;
  double silu_split_mean_us = 0.0;
  double q_l2_min_us = 0.0;
  double q_l2_mean_us = 0.0;
  double k_l2_min_us = 0.0;
  double k_l2_mean_us = 0.0;
  double fused_min_us = 0.0;
  double fused_mean_us = 0.0;
  std::uint64_t silu_split_global_work_items = 0;
  std::uint64_t q_l2_global_work_items = 0;
  std::uint64_t k_l2_global_work_items = 0;
  std::uint64_t fused_global_work_items = 0;
};

struct GpuLinearAttentionPostConvPrepRun {
  std::vector<float> conv_output_silu;
  std::vector<float> q_conv;
  std::vector<float> k_conv;
  std::vector<float> v_conv_predelta;
  std::vector<float> q_conv_predelta;
  std::vector<float> k_conv_predelta;
  GpuLinearAttentionPostConvPrepTiming timing;
};

struct GpuLinearAttentionDeltaTiming {
  double delta_min_us = 0.0;
  double delta_mean_us = 0.0;
  double final_min_us = 0.0;
  double final_mean_us = 0.0;
  double postconv_silu_split_min_us = 0.0;
  double postconv_silu_split_mean_us = 0.0;
  double postconv_q_l2_min_us = 0.0;
  double postconv_q_l2_mean_us = 0.0;
  double postconv_k_l2_min_us = 0.0;
  double postconv_k_l2_mean_us = 0.0;
  std::uint64_t input_upload_wall_ns = 0;
  std::uint64_t postconv_prep_wall_ns = 0;
  std::uint64_t kernel_wall_ns = 0;
  std::uint64_t attention_read_wall_ns = 0;
  std::uint64_t final_read_wall_ns = 0;
  std::uint64_t state_read_wall_ns = 0;
  std::uint64_t delta_global_work_items = 0;
  std::uint64_t final_global_work_items = 0;
};

struct GpuLinearAttentionDeltaRun {
  std::vector<float> attention_output;
  std::vector<float> recurrent_state;
  std::vector<float> final_output;
  std::uint64_t final_output_handle = 0;
  GpuLinearAttentionDeltaTiming timing;
};

struct GpuF32MatvecTiming {
  double min_us = 0.0;
  double mean_us = 0.0;
  double effective_weight_gb_s = 0.0;
  std::uint64_t global_work_items = 0;
};

struct GpuF32MatvecRun {
  std::vector<float> output;
  std::uint64_t output_handle = 0;
  bool output_host_valid = true;
  GpuF32MatvecTiming timing;
};

struct GpuRouterQkvDeltaSelectedValueOverlayRun {
  std::vector<float> output;
  std::uint64_t output_handle = 0;
  bool output_host_valid = true;
  GpuF32MatvecTiming timing;
};

struct GpuFfnTailTiming {
  double weighted_min_us = 0.0;
  double weighted_mean_us = 0.0;
  double shared_gate_matvec_min_us = 0.0;
  double shared_gate_matvec_mean_us = 0.0;
  double shared_gate_apply_min_us = 0.0;
  double shared_gate_apply_mean_us = 0.0;
  double ffn_output_add_min_us = 0.0;
  double ffn_output_add_mean_us = 0.0;
  double residual_add_min_us = 0.0;
  double residual_add_mean_us = 0.0;
  double shell_sum_min_us = 0.0;
  double shell_sum_mean_us = 0.0;
  std::uint64_t hidden_global_work_items = 0;
  std::uint64_t shared_gate_global_work_items = 0;
};

struct GpuFfnTailRun {
  std::vector<float> layer_output;
  std::uint64_t layer_output_handle = 0;
  bool layer_output_host_valid = true;
  GpuFfnTailTiming timing;
};

struct GpuSwiGluTiming {
  double min_us = 0.0;
  double mean_us = 0.0;
  std::uint64_t global_work_items = 0;
};

struct GpuSwiGluRun {
  std::vector<float> output;
  GpuSwiGluTiming timing;
};

struct GpuRmsNormTiming {
  double min_us = 0.0;
  double mean_us = 0.0;
  std::uint64_t global_work_items = 0;
};

struct GpuRmsNormRun {
  std::vector<float> output;
  std::uint64_t output_handle = 0;
  bool output_host_valid = true;
  GpuRmsNormTiming timing;
};

struct GpuResidualRmsNormTiming {
  double residual_min_us = 0.0;
  double residual_mean_us = 0.0;
  double rmsnorm_min_us = 0.0;
  double rmsnorm_mean_us = 0.0;
  double kernel_sum_min_us = 0.0;
  double kernel_sum_mean_us = 0.0;
  std::uint64_t residual_global_work_items = 0;
  std::uint64_t rmsnorm_global_work_items = 0;
};

struct GpuResidualRmsNormRun {
  std::vector<float> residual;
  std::vector<float> normalized;
  GpuResidualRmsNormTiming timing;
};

struct GpuRmsNormQ6MatvecTiming {
  double rmsnorm_min_us = 0.0;
  double rmsnorm_mean_us = 0.0;
  double q8_quantize_min_us = 0.0;
  double q8_quantize_mean_us = 0.0;
  GpuQ4X8MatvecTiming matvec;
  double shell_sum_min_us = 0.0;
  double shell_sum_mean_us = 0.0;
  std::uint64_t rmsnorm_global_work_items = 0;
  std::uint64_t q8_quantize_global_work_items = 0;
};

struct GpuRmsNormQ6MatvecRun {
  std::vector<float> output;
  GpuRmsNormQ6MatvecTiming timing;
};

struct GpuFullAttentionCoreGateTiming {
  double core_min_us = 0.0;
  double core_mean_us = 0.0;
  double gate_min_us = 0.0;
  double gate_mean_us = 0.0;
  double kernel_sum_min_us = 0.0;
  double kernel_sum_mean_us = 0.0;
  std::uint64_t q_global_work_items = 0;
};

struct GpuFullAttentionCoreGateRun {
  std::vector<float> attn_pregate;
  std::vector<float> attn_gated;
  GpuFullAttentionCoreGateTiming timing;
};

struct GpuFullAttentionQkNormRopeTiming {
  double min_us = 0.0;
  double mean_us = 0.0;
  std::uint64_t global_work_items = 0;
};

struct GpuFullAttentionQkNormRopeRun {
  std::vector<float> q_rope;
  std::vector<float> k_rope;
  std::uint64_t q_rope_handle = 0;
  std::uint64_t k_rope_handle = 0;
  bool output_host_valid = true;
  GpuFullAttentionQkNormRopeTiming timing;
};

struct GpuFullAttentionHistoryAppendRun {
  std::vector<float> history;
  std::uint64_t history_handle = 0;
  std::uint64_t token_count = 0;
  bool output_host_valid = true;
};

struct GpuFullAttentionOutputHandoffTiming {
  double core_min_us = 0.0;
  double core_mean_us = 0.0;
  double gate_min_us = 0.0;
  double gate_mean_us = 0.0;
  double q8_quantize_min_us = 0.0;
  double q8_quantize_mean_us = 0.0;
  GpuQ4X8MatvecTiming output_projection;
  double residual_min_us = 0.0;
  double residual_mean_us = 0.0;
  double rmsnorm_min_us = 0.0;
  double rmsnorm_mean_us = 0.0;
  double shell_sum_min_us = 0.0;
  double shell_sum_mean_us = 0.0;
  std::uint64_t q_global_work_items = 0;
  std::uint64_t q8_quantize_global_work_items = 0;
  std::uint64_t residual_global_work_items = 0;
  std::uint64_t rmsnorm_global_work_items = 0;
};

struct GpuFullAttentionOutputHandoffRun {
  std::vector<float> residual;
  std::vector<float> normalized;
  std::uint64_t residual_handle = 0;
  std::uint64_t normalized_handle = 0;
  GpuFullAttentionOutputHandoffTiming timing;
};

struct GpuQ8KInputPlanes {
  std::vector<std::int8_t> qs;
  std::vector<std::int16_t> bsums;
  std::vector<float> d;
};

struct GpuQ4KCpuOrderMatvecTiming {
  double min_us = 0.0;
  double mean_us = 0.0;
  double effective_raw_gb_s = 0.0;
  std::uint64_t global_work_items = 0;
};

struct GpuQ4KCpuOrderMatvecRun {
  std::vector<float> output;
  GpuQ4KCpuOrderMatvecTiming timing;
  std::string platform_name;
  std::string device_name;
  std::string build_log;
  double program_build_ms = 0.0;
};

struct GpuLinearPreconvSharedQ8Timing {
  double q8_quantize_min_us = 0.0;
  double q8_quantize_mean_us = 0.0;
  GpuQ4X8MatvecTiming qkv_matvec;
  double conv_min_us = 0.0;
  double conv_mean_us = 0.0;
  GpuQ4KCpuOrderMatvecTiming alpha_beta_z;
  double shell_sum_min_us = 0.0;
  double shell_sum_mean_us = 0.0;
  std::uint64_t q8_quantize_global_work_items = 0;
  std::uint64_t conv_global_work_items = 0;
};

struct GpuLinearPreconvSharedQ8Run {
  std::vector<float> qkv_mixed;
  std::vector<float> conv_output_raw;
  std::vector<float> conv_state;
  std::vector<float> alpha_beta_z;
  std::uint64_t conv_output_handle = 0;
  bool qkv_host_valid = true;
  bool conv_output_host_valid = true;
  bool conv_state_host_valid = true;
  bool alpha_beta_z_host_valid = true;
  GpuLinearPreconvSharedQ8Timing timing;
};

class GpuQ4X8MatvecRunner {
 public:
  GpuQ4X8MatvecRunner(std::string device_substring, std::string opencl_source);
  ~GpuQ4X8MatvecRunner();

  GpuQ4X8MatvecRunner(const GpuQ4X8MatvecRunner&) = delete;
  GpuQ4X8MatvecRunner& operator=(const GpuQ4X8MatvecRunner&) = delete;
  GpuQ4X8MatvecRunner(GpuQ4X8MatvecRunner&&) noexcept;
  GpuQ4X8MatvecRunner& operator=(GpuQ4X8MatvecRunner&&) noexcept;

  const std::string& platform_name() const;
  const std::string& device_name() const;
  const std::string& build_log() const;
  double program_build_ms() const;

  GpuQ4X8MatvecRun Run(const std::vector<std::uint8_t>& packed,
                       const std::vector<std::int8_t>& q8_qs,
                       const std::vector<std::int16_t>& q8_bsums,
                       const std::vector<float>& q8_d,
                       std::uint64_t rows,
                       std::uint64_t blocks_per_row,
                       int repeat,
                       GpuQ4X8KernelVariant variant);

  GpuQ4X8MatvecRun RunRowblock16(
      const std::vector<std::uint8_t>& packed,
      const std::vector<std::int8_t>& q8_qs,
      const std::vector<std::int16_t>& q8_bsums,
      const std::vector<float>& q8_d,
      std::uint64_t rows,
      std::uint64_t blocks_per_row,
      int repeat);

  GpuQ4X8MatvecRun RunRowblock16CpuOrderFinalize(
      const std::vector<std::uint8_t>& packed,
      const std::vector<std::int8_t>& q8_qs,
      const std::vector<std::int16_t>& q8_bsums,
      const std::vector<float>& q8_d,
      std::uint64_t rows,
      std::uint64_t blocks_per_row,
      int repeat);

  std::uint64_t UploadPackedQ4X8(const std::vector<std::uint8_t>& packed,
                                 std::uint64_t rows,
                                 std::uint64_t blocks_per_row);
  std::uint64_t UploadPackedQ4X8Deferred(
      const std::vector<std::uint8_t>& packed,
      std::uint64_t rows,
      std::uint64_t blocks_per_row);

  std::uint64_t ConcatResidentPackedQ4X8(
      const std::vector<std::uint64_t>& handles);

  std::uint64_t UploadRawQ4KCpuOrder(const std::vector<std::uint8_t>& raw,
                                     std::uint64_t rows,
                                     std::uint64_t blocks_per_row);

  GpuQ4X8MatvecRun RunResidentPackedQ4X8(
      std::uint64_t handle,
      const std::vector<std::int8_t>& q8_qs,
      const std::vector<std::int16_t>& q8_bsums,
      const std::vector<float>& q8_d,
      int repeat,
      GpuQ4X8KernelVariant variant);

  GpuDeviceQ8Q4X8MatvecRun
  RunF32InputHandleDeviceQ8ThenResidentPackedQ4X8(
      std::uint64_t handle,
      std::uint64_t input_handle,
      int repeat,
      GpuQ4X8KernelVariant variant,
      bool readback_output = true);

  GpuQ4KCpuOrderMatvecRun
  RunF32InputHandleDeviceQ8ThenResidentRawQ4KCpuOrder(
      std::uint64_t handle,
      std::uint64_t input_handle,
      int repeat,
      bool readback_output = true);

  GpuQ4X8MatvecRun RunResidentPackedQ4X8Expert8PerExpertQ8(
      const std::vector<std::uint64_t>& handles,
      const std::vector<std::int8_t>& q8_qs,
      const std::vector<std::int16_t>& q8_bsums,
      const std::vector<float>& q8_d,
      int repeat,
      GpuQ4X8KernelVariant variant,
      bool readback_output = true);

  GpuQ4X8SelectedSharedMatvecRun
  RunResidentPackedQ4X8Expert8PlusSharedPerExpertQ8(
      const std::vector<std::uint64_t>& selected_handles,
      std::uint64_t shared_handle,
      const GpuQ8KInputPlanes& selected_q8,
      const GpuQ8KInputPlanes& shared_q8,
      int repeat,
      bool readback_selected_output = true,
      bool readback_shared_output = true);

  std::uint64_t UploadRawQ6K(const std::vector<std::uint8_t>& raw,
                             std::uint64_t rows,
                             std::uint64_t blocks_per_row);
  std::uint64_t UploadRawQ6KDeferred(const std::vector<std::uint8_t>& raw,
                                     std::uint64_t rows,
                                     std::uint64_t blocks_per_row);
  std::uint64_t UploadSelectedRawQ6KRowstripe(
      const std::vector<std::uint8_t>& raw,
      std::uint64_t rows_per_expert,
      std::uint64_t blocks_per_row,
      std::uint64_t selected_count,
      std::uint64_t rows_per_tile);
  std::uint64_t UploadSelectedRawQ6KRowstripeDeferred(
      const std::vector<std::uint8_t>& raw,
      std::uint64_t rows_per_expert,
      std::uint64_t blocks_per_row,
      std::uint64_t selected_count,
      std::uint64_t rows_per_tile);

  std::uint64_t ConcatResidentRawQ6K(
      const std::vector<std::uint64_t>& handles);

  GpuQ6KMatvecRun RunResidentRawQ6K(std::uint64_t handle,
                                    const GpuQ8KInputPlanes& q8,
                                    int repeat);

  GpuQ6KMatvecRun RunResidentRawQ6KToF32Handle(std::uint64_t handle,
                                                const GpuQ8KInputPlanes& q8,
                                                int repeat);

  GpuDeviceQ8Q6KMatvecRun RunF32InputHandleDeviceQ8ThenResidentRawQ6K(
      std::uint64_t handle,
      std::uint64_t input_handle,
      int repeat,
      bool readback_output = true);

  GpuQ6KTopKRun RunResidentRawQ6KTopK(std::uint64_t handle,
                                      const GpuQ8KInputPlanes& q8,
                                      int topk,
                                      int repeat);

  GpuQ6KMatvecRun RunResidentRawQ6KSelected(std::uint64_t handle,
                                            const GpuQ8KInputPlanes& q8,
                                            std::uint64_t rows_per_expert,
                                            std::uint64_t selected_count,
                                            int repeat,
                                            bool readback_output = true);

  GpuQ6KMatvecRun RunResidentRawQ6KExpert8(
      const std::vector<std::uint64_t>& handles,
      const GpuQ8KInputPlanes& q8,
      std::uint64_t rows_per_expert,
      int repeat,
      bool readback_output = true);

  GpuQ6KSelectedSharedMatvecRun RunResidentRawQ6KExpert8PlusShared(
      const std::vector<std::uint64_t>& selected_handles,
      std::uint64_t shared_handle,
      const GpuQ8KInputPlanes& selected_q8,
      const GpuQ8KInputPlanes& shared_q8,
      std::uint64_t rows_per_expert,
      int repeat,
      bool readback_selected_output = true,
      bool readback_shared_output = true);

  GpuFfnTailRun RunResidentRawQ6KExpert8PlusSharedToFfnTailAtomic(
      const std::vector<std::uint64_t>& selected_handles,
      std::uint64_t shared_handle,
      const GpuQ8KInputPlanes& selected_q8,
      const GpuQ8KInputPlanes& shared_q8,
      std::uint64_t shared_gate_matvec_handle,
      std::uint64_t attn_post_norm_handle,
      const std::vector<float>& weights_norm,
      std::uint64_t attn_residual_handle,
      std::uint64_t rows_per_expert,
      int repeat,
      bool readback_layer_output = true);

  GpuFfnTailRun RunResidentRawQ6KExpert8PlusSharedToFfnTailNonAtomic(
      const std::vector<std::uint64_t>& selected_handles,
      std::uint64_t shared_handle,
      const GpuQ8KInputPlanes& selected_q8,
      const GpuQ8KInputPlanes& shared_q8,
      std::uint64_t shared_gate_matvec_handle,
      std::uint64_t attn_post_norm_handle,
      const std::vector<float>& weights_norm,
      std::uint64_t attn_residual_handle,
      std::uint64_t rows_per_expert,
      int repeat,
      bool readback_layer_output = true);

  GpuRmsNormQ6MatvecRun RunRmsNormThenResidentRawQ6K(
      const std::vector<float>& input,
      std::uint64_t norm_weight_handle,
      std::uint64_t q6_handle,
      std::uint64_t hidden_size,
      float norm_epsilon,
      int repeat);

  GpuQ4X8ConvHandoffRun RunResidentRawQ6KThenResidentConv(
      std::uint64_t q6_handle,
      const GpuQ8KInputPlanes& q8,
      std::uint64_t conv_weights_handle,
      const std::vector<float>& conv_state,
      std::uint64_t conv_kernel_size,
      int repeat);

  GpuQ4X8ConvHandoffRun RunResidentRawQ6KThenResidentConvState(
      std::uint64_t q6_handle,
      const GpuQ8KInputPlanes& q8,
      std::uint64_t conv_weights_handle,
      std::uint64_t conv_state_handle,
      std::uint64_t conv_kernel_size,
      int repeat,
      bool readback_state = false,
      std::uint64_t next_conv_state_handle = 0,
      bool readback_qkv = true,
      bool readback_conv_output = true);

  GpuQ4X8ConvHandoffRun RunResidentRawQ6KThenResidentConvStateCpuOrder(
      std::uint64_t q6_handle,
      const GpuQ8KInputPlanes& q8,
      std::uint64_t conv_weights_handle,
      std::uint64_t conv_state_handle,
      std::uint64_t conv_kernel_size,
      int repeat,
      bool readback_state = false,
      std::uint64_t next_conv_state_handle = 0,
      bool readback_qkv = true,
      bool readback_conv_output = true);

  GpuQ4X8ConvHandoffRun
  RunF32InputHandleDeviceQ8ThenResidentRawQ6KThenResidentConvState(
      std::uint64_t q6_handle,
      std::uint64_t input_handle,
      std::uint64_t conv_weights_handle,
      std::uint64_t conv_state_handle,
      std::uint64_t conv_kernel_size,
      int repeat,
      bool readback_state = false,
      std::uint64_t next_conv_state_handle = 0,
      bool readback_qkv = true,
      bool readback_conv_output = true);

  GpuLinearPreconvSharedQ8Run
  RunF32InputHandleSharedDeviceQ8ThenResidentRawQ6KConvStateAndResidentRawQ4KCpuOrder(
      std::uint64_t q6_handle,
      std::uint64_t alpha_beta_z_handle,
      std::uint64_t input_handle,
      std::uint64_t conv_weights_handle,
      std::uint64_t conv_state_handle,
      std::uint64_t conv_kernel_size,
      int repeat,
      bool readback_state = false,
      std::uint64_t next_conv_state_handle = 0,
      bool readback_qkv = true,
      bool readback_conv_output = true,
      bool readback_alpha_beta_z = true);

  void ClearResidentRawQ6K();

  GpuQ4X8ConvHandoffRun RunResidentPackedQ4X8ThenConv(
      std::uint64_t handle,
      const std::vector<std::int8_t>& q8_qs,
      const std::vector<std::int16_t>& q8_bsums,
      const std::vector<float>& q8_d,
      const std::vector<float>& conv_weights,
      const std::vector<float>& conv_state,
      std::uint64_t conv_kernel_size,
      int repeat,
      GpuQ4X8KernelVariant variant);

  GpuQ4X8SwiGluHandoffRun RunResidentPackedQ4X8ThenSwiGlu(
      std::uint64_t handle,
      const std::vector<std::int8_t>& q8_qs,
      const std::vector<std::int16_t>& q8_bsums,
      const std::vector<float>& q8_d,
      std::uint64_t intermediate_size,
      const std::vector<std::uint32_t>& source_expert_by_output,
      int repeat,
      GpuQ4X8KernelVariant variant);

  GpuQ4X8SwiGluHandoffRun
  RunResidentPackedQ4X8ThenSwiGluWithLastExpert8Q8(
      std::uint64_t handle,
      std::uint64_t intermediate_size,
      const std::vector<std::uint32_t>& source_expert_by_output,
      int repeat,
      GpuQ4X8KernelVariant variant);

  GpuQ4X8SwiGluHandoffRun RunResidentPackedQ4X8Expert8ThenSwiGlu(
      const std::vector<std::uint64_t>& handles,
      const std::vector<std::int8_t>& q8_qs,
      const std::vector<std::int16_t>& q8_bsums,
      const std::vector<float>& q8_d,
      std::uint64_t intermediate_size,
      int repeat,
      GpuQ4X8KernelVariant variant);

  GpuQ4X8SwiGluHandoffRun
  RunResidentPackedQ4X8Expert8PlusSharedThenSwiGlu(
      const std::vector<std::uint64_t>& selected_handles,
      std::uint64_t shared_handle,
      const std::vector<std::int8_t>& q8_qs,
      const std::vector<std::int16_t>& q8_bsums,
      const std::vector<float>& q8_d,
      std::uint64_t intermediate_size,
      int repeat,
      GpuQ4X8KernelVariant variant);

  GpuQ4X8SwiGluHandoffRun
  RunResidentPackedQ4X8TopKIndexedExpert8PlusSharedThenSwiGlu(
      std::uint64_t selected_material_handle,
      std::uint64_t shared_handle,
      const std::vector<std::uint32_t>& selected_positions,
      const std::vector<std::int8_t>& q8_qs,
      const std::vector<std::int16_t>& q8_bsums,
      const std::vector<float>& q8_d,
      std::uint64_t intermediate_size,
      int repeat,
      GpuQ4X8KernelVariant variant);

  GpuQ4X8SwiGluQ6DownHandoffRun
  RunResidentPackedQ4X8ThenSwiGluThenRawQ6KSelected(
      std::uint64_t packed_handle,
      std::uint64_t q6_handle,
      const std::vector<std::int8_t>& q8_qs,
      const std::vector<std::int16_t>& q8_bsums,
      const std::vector<float>& q8_d,
      std::uint64_t intermediate_size,
      const std::vector<std::uint32_t>& source_expert_by_output,
      std::uint64_t rows_per_expert,
      int repeat,
      GpuQ4X8KernelVariant variant);

  GpuQ4X8SwiGluQ6DownHandoffRun
  RunResidentPackedQ4X8Expert8ThenSwiGluThenRawQ6KExpert8(
      const std::vector<std::uint64_t>& packed_handles,
      const std::vector<std::uint64_t>& q6_handles,
      const std::vector<std::int8_t>& q8_qs,
      const std::vector<std::int16_t>& q8_bsums,
      const std::vector<float>& q8_d,
      std::uint64_t intermediate_size,
      std::uint64_t rows_per_expert,
      int repeat,
      GpuQ4X8KernelVariant variant);

  GpuQ4X8SwiGluQ4F32DownHandoffRun
  RunResidentPackedQ4X8Expert8ThenSwiGluThenPackedQ4X8Expert8F32Input(
      const std::vector<std::uint64_t>& gate_up_handles,
      const std::vector<std::uint64_t>& down_handles,
      const std::vector<std::int8_t>& q8_qs,
      const std::vector<std::int16_t>& q8_bsums,
      const std::vector<float>& q8_d,
      std::uint64_t intermediate_size,
      std::uint64_t rows_per_expert,
      int repeat,
      GpuQ4X8KernelVariant variant,
      bool readback_output = true);

  GpuQ4X8ResidualRmsNormHandoffRun RunResidentPackedQ4X8ThenResidualRmsNorm(
      std::uint64_t handle,
      const std::vector<std::int8_t>& q8_qs,
      const std::vector<std::int16_t>& q8_bsums,
      const std::vector<float>& q8_d,
      const std::vector<float>& residual_input,
      const std::vector<float>& norm_weight,
      float norm_epsilon,
      int repeat,
      GpuQ4X8KernelVariant variant,
      bool use_rowblock16_output_projection = false);

  GpuQ4X8ResidualRmsNormHandoffRun
  RunF32DeviceQ8ThenResidentPackedQ4X8ResidualRmsNorm(
      std::uint64_t handle,
      const std::vector<float>& input,
      const std::vector<float>& residual_input,
      const std::vector<float>& norm_weight,
      float norm_epsilon,
      int repeat,
      GpuQ4X8KernelVariant variant);

  GpuQ4X8ResidualRmsNormHandoffRun
  RunF32InputHandleDeviceQ8ThenResidentPackedQ4X8ResidualRmsNorm(
      std::uint64_t handle,
      std::uint64_t input_handle,
      const std::vector<float>& residual_input,
      const std::vector<float>& norm_weight,
      std::uint64_t norm_weight_handle,
      float norm_epsilon,
      int repeat,
      GpuQ4X8KernelVariant variant,
      std::uint64_t residual_input_handle = 0,
      bool use_rowblock16_cpuorder_finalize = false);

  GpuQ4X8ResidualRmsNormHandoffRun
  RunResidentPackedQ4X8ThenResidentResidualRmsNorm(
      std::uint64_t handle,
      const std::vector<std::int8_t>& q8_qs,
      const std::vector<std::int16_t>& q8_bsums,
      const std::vector<float>& q8_d,
      const std::vector<float>& residual_input,
      std::uint64_t norm_weight_handle,
      std::uint64_t hidden_size,
      float norm_epsilon,
      int repeat,
      GpuQ4X8KernelVariant variant,
      std::uint64_t residual_input_handle = 0,
      bool use_rowblock16_output_projection = false);

  std::uint64_t UploadConvWeights(const std::vector<float>& conv_weights,
                                  std::uint64_t rows,
                                  std::uint64_t conv_kernel_size);

  GpuQ4X8ConvHandoffRun RunResidentPackedQ4X8ThenResidentConv(
      std::uint64_t packed_handle,
      const std::vector<std::int8_t>& q8_qs,
      const std::vector<std::int16_t>& q8_bsums,
      const std::vector<float>& q8_d,
      std::uint64_t conv_weights_handle,
      const std::vector<float>& conv_state,
      std::uint64_t conv_kernel_size,
      int repeat,
      GpuQ4X8KernelVariant variant);

  GpuQ4X8ConvHandoffRun RunResidentPackedQ4X8ThenResidentConvState(
      std::uint64_t packed_handle,
      const std::vector<std::int8_t>& q8_qs,
      const std::vector<std::int16_t>& q8_bsums,
      const std::vector<float>& q8_d,
      std::uint64_t conv_weights_handle,
      std::uint64_t conv_state_handle,
      std::uint64_t conv_kernel_size,
      int repeat,
      GpuQ4X8KernelVariant variant,
      bool readback_state = false,
      std::uint64_t next_conv_state_handle = 0,
      bool readback_qkv = true,
      bool readback_conv_output = true);

  GpuQ4X8ConvHandoffRun
  RunF32InputHandleDeviceQ8ThenResidentPackedQ4X8ThenResidentConvState(
      std::uint64_t packed_handle,
      std::uint64_t input_handle,
      std::uint64_t conv_weights_handle,
      std::uint64_t conv_state_handle,
      std::uint64_t conv_kernel_size,
      int repeat,
      GpuQ4X8KernelVariant variant,
      bool readback_state = false,
      std::uint64_t next_conv_state_handle = 0,
      bool readback_qkv = true,
      bool readback_conv_output = true);

  GpuLinearPreconvSharedQ8Run
  RunF32InputHandleSharedDeviceQ8ThenResidentPackedQ4X8ConvStateAndResidentRawQ4KCpuOrder(
      std::uint64_t packed_handle,
      std::uint64_t alpha_beta_z_handle,
      std::uint64_t input_handle,
      std::uint64_t conv_weights_handle,
      std::uint64_t conv_state_handle,
      std::uint64_t conv_kernel_size,
      int repeat,
      GpuQ4X8KernelVariant variant,
      bool readback_state = false,
      std::uint64_t next_conv_state_handle = 0,
      bool readback_qkv = true,
      bool readback_conv_output = true,
      bool readback_alpha_beta_z = true);

  void ClearResidentRawQ4KCpuOrder();

  void ClearResidentConvWeights();

  void ClearResidentPackedQ4X8();

  GpuQ4X8ConvHandoffRun RunThenConv(const std::vector<std::uint8_t>& packed,
                                    const std::vector<std::int8_t>& q8_qs,
                                    const std::vector<std::int16_t>& q8_bsums,
                                    const std::vector<float>& q8_d,
                                    const std::vector<float>& conv_weights,
                                    const std::vector<float>& conv_state,
                                    std::uint64_t rows,
                                    std::uint64_t blocks_per_row,
                                    std::uint64_t conv_kernel_size,
                                    int repeat,
                                    GpuQ4X8KernelVariant variant);

  GpuLinearAttentionPostConvPrepRun RunPostConvPrep(
      const std::vector<float>& conv_output_raw,
      std::uint64_t head_dim,
      std::uint64_t query_heads,
      std::uint64_t value_heads,
      float norm_epsilon,
      int repeat,
      bool readback_intermediates = true);

  GpuLinearAttentionPostConvPrepRun RunPostConvPrepFused(
      const std::vector<float>& conv_output_raw,
      std::uint64_t head_dim,
      std::uint64_t query_heads,
      std::uint64_t value_heads,
      float norm_epsilon,
      int repeat,
      bool readback_intermediates = true);

  GpuLinearAttentionDeltaRun RunLinearAttentionDelta(
      const std::vector<float>& q,
      const std::vector<float>& k,
      const std::vector<float>& v,
      const std::vector<float>& gate,
      const std::vector<float>& beta,
      const std::vector<float>& recurrent_state,
      const std::vector<float>& z,
      const std::vector<float>& norm_weight,
      std::uint64_t head_dim,
      std::uint64_t query_heads,
      std::uint64_t value_heads,
      float norm_epsilon,
      int repeat,
      bool cpu_shape_final_norm = false);

  std::uint64_t UploadF32Buffer(const std::vector<float>& values);

  std::uint64_t CloneResidentF32Buffer(std::uint64_t source_handle);

  GpuRouterQkvDeltaSelectedValueOverlayRun
  RunRouterQkvDeltaSelectedValueOverlay(
      std::uint64_t base_handle,
      std::uint64_t source_handle,
      const std::vector<std::int32_t>& selected_indices,
      int repeat,
      bool readback_output = true);

  GpuRouterQkvDeltaSelectedValueOverlayRun
  RunRouterQkvDeltaBlockQ16Overlay(
      std::uint64_t base_handle,
      const std::vector<std::int32_t>& selected_indices,
      const std::vector<std::int16_t>& selected_q_delta,
      const std::vector<float>& block_scales,
      int repeat,
      bool readback_output = true);

  GpuLinearAttentionDeltaRun RunLinearAttentionDeltaResidentState(
      std::uint64_t state_handle,
      const std::vector<float>& q,
      const std::vector<float>& k,
      const std::vector<float>& v,
      const std::vector<float>& gate,
      const std::vector<float>& beta,
      const std::vector<float>& z,
      const std::vector<float>& norm_weight,
      std::uint64_t head_dim,
      std::uint64_t query_heads,
      std::uint64_t value_heads,
      float norm_epsilon,
      int repeat,
      bool readback_state = false,
      bool cpu_shape_final_norm = false,
      bool readback_attention_output = true,
      bool readback_final_output = true);

  GpuLinearAttentionDeltaRun RunPostConvPrepThenLinearAttentionDeltaResidentState(
      std::uint64_t conv_output_handle,
      std::uint64_t state_handle,
      const std::vector<float>& gate,
      const std::vector<float>& beta,
      const std::vector<float>& z,
      const std::vector<float>& norm_weight,
      std::uint64_t head_dim,
      std::uint64_t query_heads,
      std::uint64_t value_heads,
      float norm_epsilon,
      int repeat,
      bool readback_state = false,
      bool cpu_shape_final_norm = false,
      bool readback_attention_output = true,
      bool readback_final_output = true);

  GpuLinearAttentionDeltaRun
  RunPostConvPrepThenLinearAttentionDeltaResidentStateCpuOrder(
      std::uint64_t conv_output_handle,
      std::uint64_t state_handle,
      const std::vector<float>& gate,
      const std::vector<float>& beta,
      const std::vector<float>& z,
      const std::vector<float>& norm_weight,
      std::uint64_t head_dim,
      std::uint64_t query_heads,
      std::uint64_t value_heads,
      float norm_epsilon,
      int repeat,
      bool readback_state = false,
      bool readback_attention_output = true,
      bool readback_final_output = true);

  void ClearResidentF32Buffers();

  GpuF32MatvecRun RunF32Matvec(const std::vector<float>& weights,
                               const std::vector<float>& input,
                               std::uint64_t rows,
                               std::uint64_t cols,
                               int repeat);

  GpuSwiGluRun RunSwiGlu(const std::vector<float>& gate_up,
                         std::uint64_t intermediate_size,
                         std::uint64_t expert_count,
                         int repeat);

  GpuFfnTailRun RunFfnTail(const std::vector<float>& gate_weights,
                           const std::vector<float>& attn_post_norm,
                           const std::vector<float>& ffn_moe_down,
                           const std::vector<float>& weights_norm,
                           const std::vector<float>& ffn_shexp,
                           const std::vector<float>& attn_residual,
                           std::uint64_t hidden_size,
                           std::uint64_t expert_count,
                           int repeat);

  GpuFfnTailRun RunFfnTailFromDownHandle(
      const std::vector<float>& gate_weights,
      const std::vector<float>& attn_post_norm,
      std::uint64_t ffn_moe_down_handle,
      const std::vector<float>& weights_norm,
      const std::vector<float>& ffn_shexp,
      const std::vector<float>& attn_residual,
      std::uint64_t hidden_size,
      std::uint64_t expert_count,
      int repeat,
      bool readback_layer_output = true);

  GpuFfnTailRun RunFfnTailFromDownHandles(
      const std::vector<float>& gate_weights,
      const std::vector<float>& attn_post_norm,
      std::uint64_t ffn_moe_down_handle,
      const std::vector<float>& weights_norm,
      std::uint64_t ffn_shexp_handle,
      const std::vector<float>& attn_residual,
      std::uint64_t hidden_size,
      std::uint64_t expert_count,
      int repeat,
      bool readback_layer_output = true);

  GpuFfnTailRun RunFfnTailFromDownHandlesResidentInputs(
      std::uint64_t shared_gate_matvec_handle,
      std::uint64_t attn_post_norm_handle,
      std::uint64_t ffn_moe_down_handle,
      const std::vector<float>& weights_norm,
      std::uint64_t ffn_shexp_handle,
      std::uint64_t attn_residual_handle,
      std::uint64_t hidden_size,
      std::uint64_t expert_count,
      int repeat,
      bool readback_layer_output = true);

  GpuFfnTailRun RunFfnTailAtomicFromDownHandlesResidentInputs(
      std::uint64_t shared_gate_matvec_handle,
      std::uint64_t attn_post_norm_handle,
      std::uint64_t ffn_moe_down_handle,
      const std::vector<float>& weights_norm,
      std::uint64_t ffn_shexp_handle,
      std::uint64_t attn_residual_handle,
      std::uint64_t hidden_size,
      std::uint64_t expert_count,
      int repeat,
      bool readback_layer_output = true);

  GpuRmsNormRun RunRmsNormHidden(const std::vector<float>& input,
                                 const std::vector<float>& weight,
                                 float norm_epsilon,
                                 int repeat,
                                 bool serial_reduction = false);

  GpuRmsNormRun RunRmsNormHiddenResidentWeight(
      const std::vector<float>& input,
      std::uint64_t weight_handle,
      std::uint64_t hidden_size,
      float norm_epsilon,
      int repeat,
      bool serial_reduction = false);

  GpuRmsNormRun RunRmsNormHiddenResidentInputResidentWeight(
      std::uint64_t input_handle,
      std::uint64_t weight_handle,
      std::uint64_t hidden_size,
      float norm_epsilon,
      int repeat,
      bool readback_output = true,
      bool serial_reduction = false);

  GpuResidualRmsNormRun RunResidualRmsNormHidden(
      const std::vector<float>& residual_input,
      const std::vector<float>& residual_delta,
      const std::vector<float>& norm_weight,
      float norm_epsilon,
      int repeat);

  GpuResidualRmsNormRun RunResidualRmsNormHiddenResidentWeight(
      const std::vector<float>& residual_input,
      const std::vector<float>& residual_delta,
      std::uint64_t norm_weight_handle,
      std::uint64_t hidden_size,
      float norm_epsilon,
      int repeat);

  GpuFullAttentionCoreGateRun RunFullAttentionCoreGate(
      const std::vector<float>& q_rope,
      const std::vector<float>& k_history_flat,
      const std::vector<float>& v_history_flat,
      const std::vector<float>& q_full,
      std::uint64_t token_count,
      std::uint64_t head_dim,
      std::uint64_t q_head_count,
      std::uint64_t kv_head_count,
      float attention_scale,
      int repeat);

  GpuFullAttentionQkNormRopeRun RunFullAttentionQkNormRopeFromHandles(
      std::uint64_t q_full_handle,
      std::uint64_t k_raw_handle,
      std::uint64_t q_norm_weight_handle,
      std::uint64_t k_norm_weight_handle,
      const std::vector<float>& rope_cache,
      std::uint64_t head_dim,
      std::uint64_t q_head_count,
      std::uint64_t kv_head_count,
      std::uint64_t rope_dimension_count,
      float norm_epsilon,
      int repeat,
      bool readback_output = true);

  GpuFullAttentionHistoryAppendRun BuildFullAttentionHistoryFromHandle(
      const std::vector<float>& previous_history_flat,
      std::uint64_t current_handle,
      std::uint64_t kv_values,
      bool readback_output = false);

  GpuFullAttentionOutputHandoffRun
  RunFullAttentionCoreGateThenResidentPackedQ4X8ResidualRmsNorm(
      const std::vector<float>& q_rope,
      const std::vector<float>& k_history_flat,
      const std::vector<float>& v_history_flat,
      const std::vector<float>& q_full,
      std::uint64_t token_count,
      std::uint64_t head_dim,
      std::uint64_t q_head_count,
      std::uint64_t kv_head_count,
      float attention_scale,
      std::uint64_t output_projection_handle,
      const std::vector<float>& residual_input,
      const std::vector<float>& norm_weight,
      float norm_epsilon,
      int repeat,
      GpuQ4X8KernelVariant variant,
      std::uint64_t residual_input_handle = 0);

  GpuFullAttentionOutputHandoffRun
  RunFullAttentionCoreGateThenResidentPackedQ4X8ResidualRmsNormFromHandles(
      std::uint64_t q_rope_handle,
      std::uint64_t k_history_handle,
      std::uint64_t v_history_handle,
      std::uint64_t q_full_handle,
      std::uint64_t token_count,
      std::uint64_t head_dim,
      std::uint64_t q_head_count,
      std::uint64_t kv_head_count,
      float attention_scale,
      std::uint64_t output_projection_handle,
      const std::vector<float>& residual_input,
      const std::vector<float>& norm_weight,
      float norm_epsilon,
      int repeat,
      GpuQ4X8KernelVariant variant,
      std::uint64_t residual_input_handle = 0);

  GpuFullAttentionOutputHandoffRun
  RunFullAttentionCoreGateThenResidentPackedQ4X8ResidualRmsNorm(
      const std::vector<float>& q_rope,
      const std::vector<float>& k_history_flat,
      const std::vector<float>& v_history_flat,
      const std::vector<float>& q_full,
      std::uint64_t token_count,
      std::uint64_t head_dim,
      std::uint64_t q_head_count,
      std::uint64_t kv_head_count,
      float attention_scale,
      std::uint64_t output_projection_handle,
      const std::vector<float>& residual_input,
      std::uint64_t norm_weight_handle,
      std::uint64_t hidden_size,
      float norm_epsilon,
      int repeat,
      GpuQ4X8KernelVariant variant,
      std::uint64_t residual_input_handle = 0);

  GpuFullAttentionOutputHandoffRun
  RunFullAttentionCoreGateThenResidentPackedQ4X8ResidualRmsNormFromHandles(
      std::uint64_t q_rope_handle,
      std::uint64_t k_history_handle,
      std::uint64_t v_history_handle,
      std::uint64_t q_full_handle,
      std::uint64_t token_count,
      std::uint64_t head_dim,
      std::uint64_t q_head_count,
      std::uint64_t kv_head_count,
      float attention_scale,
      std::uint64_t output_projection_handle,
      const std::vector<float>& residual_input,
      std::uint64_t norm_weight_handle,
      std::uint64_t hidden_size,
      float norm_epsilon,
      int repeat,
      GpuQ4X8KernelVariant variant,
      std::uint64_t residual_input_handle = 0);

  std::uint64_t UploadF32MatvecWeights(const std::vector<float>& weights,
                                       std::uint64_t rows,
                                       std::uint64_t cols);

  GpuF32MatvecRun RunResidentF32Matvec(std::uint64_t handle,
                                       const std::vector<float>& input,
                                       int repeat);
  GpuF32MatvecRun RunResidentF32MatvecFromInputHandle(
      std::uint64_t handle,
      std::uint64_t input_handle,
      int repeat,
      bool readback_output = true);
  GpuF32TopKRun RunResidentF32TopK(std::uint64_t values_handle,
                                   int topk,
                                   int repeat);

  void ClearResidentF32Matvec();

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

const char* KernelVariantName(GpuQ4X8KernelVariant variant);
const char* KernelFunctionName(GpuQ4X8KernelVariant variant);
std::uint64_t RowsPerWorkItem(GpuQ4X8KernelVariant variant);
GpuQ8KInputPlanes QuantizeQ8KInputPlanes(const std::vector<float>& input);
std::vector<std::uint8_t> PackQ4Kx8(const std::vector<std::uint8_t>& raw,
                                    std::uint64_t rows,
                                    std::uint64_t blocks_per_row);

struct PackedQ6KRowstripe {
  std::vector<std::uint8_t> bytes;
  std::uint64_t rows_per_tile = 0;
  std::uint64_t row_tile_count = 0;
};

PackedQ6KRowstripe PackQ6KRowstripe(
    const std::vector<std::uint8_t>& raw,
    std::uint64_t rows_per_expert,
    std::uint64_t blocks_per_row,
    std::uint64_t expert_count,
    std::uint64_t rows_per_tile = 16);

PackedQ6KRowstripe PackQ6KRowstripeCoalesced(
    const std::vector<std::uint8_t>& raw,
    std::uint64_t rows_per_expert,
    std::uint64_t blocks_per_row,
    std::uint64_t expert_count,
    std::uint64_t rows_per_tile = 16);

class GpuQ4KCpuOrderMatvecRunner {
 public:
  explicit GpuQ4KCpuOrderMatvecRunner(std::string device_substring);
  ~GpuQ4KCpuOrderMatvecRunner();

  GpuQ4KCpuOrderMatvecRunner(const GpuQ4KCpuOrderMatvecRunner&) = delete;
  GpuQ4KCpuOrderMatvecRunner& operator=(const GpuQ4KCpuOrderMatvecRunner&) = delete;
  GpuQ4KCpuOrderMatvecRunner(GpuQ4KCpuOrderMatvecRunner&&) noexcept;
  GpuQ4KCpuOrderMatvecRunner& operator=(GpuQ4KCpuOrderMatvecRunner&&) noexcept;

  const std::string& platform_name() const;
  const std::string& device_name() const;
  const std::string& build_log() const;
  double program_build_ms() const;

  std::uint64_t UploadRawQ4KCpuOrder(const std::vector<std::uint8_t>& raw,
                                     std::uint64_t rows,
                                     std::uint64_t blocks_per_row);

  GpuQ4KCpuOrderMatvecRun RunResidentRawQ4KCpuOrder(
      std::uint64_t handle,
      const GpuQ8KInputPlanes& q8,
      int repeat);

  void ClearResidentRawQ4KCpuOrder();

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

GpuQ4KCpuOrderMatvecRun RunQ4KCpuOrderMatvec(
    const std::vector<std::uint8_t>& raw,
    const GpuQ8KInputPlanes& q8,
    std::uint64_t rows,
    std::uint64_t blocks_per_row,
    const std::string& device_substring,
    int repeat);

}  // namespace iq36
