#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace iq36 {

struct GgufMetadataValue {
  enum class Kind {
    kArray,
    kBool,
    kFloat,
    kInt,
    kString,
    kUInt,
    kUnknown,
  };

  Kind kind = Kind::kUnknown;
  bool bool_value = false;
  double float_value = 0.0;
  std::int64_t int_value = 0;
  std::uint64_t uint_value = 0;
  std::string string_value;
  std::uint32_t array_element_type = 0;
  std::vector<std::int64_t> int_array;
  std::vector<std::uint64_t> uint_array;
  std::vector<double> float_array;
  std::vector<std::string> string_array;
};

struct GgufTensorInfo {
  std::string name;
  std::vector<std::uint64_t> dims;
  std::uint32_t type = 0;
  std::uint64_t offset = 0;
  std::uint64_t absolute_offset = 0;
  std::uint64_t nbytes = 0;
  int layer_index = -1;
  std::string suffix;
};

struct GgufModelIndex {
  std::uint32_t version = 0;
  std::uint64_t tensor_count = 0;
  std::uint64_t metadata_kv_count = 0;
  std::uint64_t data_section_offset = 0;
  std::uint64_t file_size_bytes = 0;
  std::unordered_map<std::string, GgufMetadataValue> metadata;
  std::vector<GgufTensorInfo> tensors;
};

struct GgufLayerSummary {
  int layer_index = 0;
  std::string kind;
  int tensor_count = 0;
};

struct GgufLoadMapSummary {
  bool ready = false;
  int tensor_count = 0;
  int metadata_kv_count = 0;
  int linear_ssm_layer_count = 0;
  int full_attention_layer_count = 0;
  std::vector<int> full_attention_layers;
  std::unordered_map<std::string, int> tensor_type_counts;
  std::vector<GgufLayerSummary> layer_summaries;
  std::vector<std::string> failed_checks;
};

struct ResidentTensorCacheStats {
  bool enabled = false;
  std::uint64_t decoded_row_hits = 0;
  std::uint64_t decoded_row_misses = 0;
  std::uint64_t decoded_row_cached_values = 0;
  std::uint64_t decoded_row_cached_bytes = 0;
  std::uint64_t tensor_payload_hits = 0;
  std::uint64_t tensor_payload_misses = 0;
  std::uint64_t tensor_payload_cached_bytes = 0;
  std::uint64_t q4_plane_hits = 0;
  std::uint64_t q4_plane_misses = 0;
  std::uint64_t q4_plane_cached_bytes = 0;
  std::uint64_t q4_plane_repack_ns = 0;
  std::uint64_t expert_slice_hits = 0;
  std::uint64_t expert_slice_misses = 0;
  std::uint64_t expert_slice_cached_bytes = 0;
};

struct MatvecProfileRow {
  std::string op;
  std::string tensor_name;
  std::uint64_t call_count = 0;
  std::uint64_t total_ns = 0;
  std::uint64_t max_ns = 0;
  std::uint64_t input_value_count = 0;
  std::uint64_t output_value_count = 0;
  std::uint64_t row_count = 0;
};

struct MatvecTopKRow {
  std::int32_t token_id = 0;
  float value = 0.0f;
};

struct TensorPayloadStats {
  std::string name;
  std::string type_name;
  std::uint64_t absolute_offset = 0;
  std::uint64_t nbytes = 0;
  std::uint64_t decoded_values = 0;
  double min = 0.0;
  double max = 0.0;
  double sum = 0.0;
  double abs_sum = 0.0;
  double l2 = 0.0;
  bool finite = false;
  bool nonzero = false;
};

struct VectorCompareStats {
  std::uint64_t lhs_value_count = 0;
  std::uint64_t rhs_value_count = 0;
  std::uint64_t compared_value_count = 0;
  std::uint64_t finite_pair_count = 0;
  std::uint64_t mismatch_count = 0;
  double max_abs_diff = 0.0;
  double mean_abs_diff = 0.0;
  double rmse = 0.0;
  double cosine = 0.0;
  double lhs_l2 = 0.0;
  double rhs_l2 = 0.0;
  bool same_size = false;
  bool finite = false;
};

struct RouterTopKSelection {
  std::vector<std::int32_t> expert_ids;
  std::vector<float> weights;
  std::vector<float> normalized_weights;
};

struct Qwen36MoeFfnLayerResult {
  std::vector<float> ffn_norm;
  std::vector<float> router_logits;
  RouterTopKSelection router;
  std::vector<float> selected_gate_up;
  std::vector<float> selected_swiglu;
  std::vector<float> selected_down;
  std::vector<float> weighted_selected_down;
  std::vector<float> moe_out;
  std::vector<float> shared_gate;
  std::vector<float> shared_gate_sigmoid;
  std::vector<float> shared_gate_up;
  std::vector<float> shared_swiglu;
  std::vector<float> shared_down;
  std::vector<float> shared_gated;
  std::vector<float> ffn_out;
  std::vector<float> residual;
};

struct Qwen36LayerShellResult {
  std::vector<float> attention_output;
  std::vector<float> attention_residual;
  Qwen36MoeFfnLayerResult ffn;
  std::vector<float> residual;
};

struct Qwen36LoopShellResult {
  std::vector<Qwen36LayerShellResult> layers;
  std::vector<float> final_norm;
  std::vector<float> logits;
};

struct Qwen36LinearAttentionDeltaResult {
  std::vector<float> attention_output;
  std::vector<float> recurrent_state;
  std::vector<float> final_output;
};

struct Qwen36LinearAttentionPreConvResult {
  std::vector<float> qkv_mixed;
  std::vector<float> alpha;
  std::vector<float> alpha_softplus;
  std::vector<float> gate;
  std::vector<float> beta;
  std::vector<float> beta_sigmoid;
  std::vector<float> z;
};

struct Qwen36LinearAttentionConvResult {
  std::vector<float> conv_output_raw;
  std::vector<float> conv_state;
};

struct Qwen36LinearAttentionPostConvResult {
  std::vector<float> conv_output_silu;
  std::vector<float> q_conv;
  std::vector<float> k_conv;
  std::vector<float> v_conv_predelta;
  std::vector<float> q_conv_predelta;
  std::vector<float> k_conv_predelta;
  std::vector<float> attention_output;
  std::vector<float> recurrent_state;
  std::vector<float> final_output;
};

struct Qwen36LayerPostConvResult {
  std::vector<float> attention_norm;
  Qwen36LinearAttentionPostConvResult attention;
  std::vector<float> linear_attention_out;
  std::vector<float> attention_residual;
  Qwen36MoeFfnLayerResult ffn;
  std::vector<float> residual;
};

struct Qwen36StatefulLinearAttentionLayerResult {
  std::vector<float> attention_norm;
  Qwen36LinearAttentionPreConvResult preconv;
  Qwen36LinearAttentionConvResult conv;
  std::vector<float> state_predelta;
  Qwen36LinearAttentionPostConvResult attention;
  std::vector<float> linear_attention_out;
  std::vector<float> attention_residual;
  Qwen36MoeFfnLayerResult ffn;
  std::vector<float> residual;
};

struct Qwen36FullAttentionQkvProjectionResult {
  std::vector<float> attention_norm;
  std::vector<float> q_full;
  std::vector<float> q_raw;
  std::vector<float> q_gate;
  std::vector<float> q_normed;
  std::vector<float> k_raw;
  std::vector<float> k_normed;
  std::vector<float> v;
};

struct Qwen36FullAttentionRopeResult {
  std::vector<float> q_rope;
  std::vector<float> k_rope;
};

struct Qwen36FullAttentionGateResult {
  std::vector<float> q_gate;
  std::vector<float> gate_sigmoid;
  std::vector<float> attn_gated;
};

struct Qwen36FullAttentionCoreResult {
  std::vector<float> attention_weights;
  std::vector<float> attn_pregate;
};

struct Qwen36StatefulFullAttentionLayerResult {
  Qwen36FullAttentionQkvProjectionResult qkv;
  Qwen36FullAttentionRopeResult rope;
  std::vector<std::vector<float>> k_history;
  std::vector<std::vector<float>> v_history;
  Qwen36FullAttentionCoreResult core;
  Qwen36FullAttentionGateResult gate;
  std::vector<float> attention_output;
};

GgufModelIndex parse_gguf_model_index(const std::string& path);
void set_resident_tensor_cache_enabled(bool enabled);
void reset_resident_tensor_cache();
ResidentTensorCacheStats resident_tensor_cache_stats();
void set_matvec_profile_enabled(bool enabled);
void reset_matvec_profile();
std::vector<MatvecProfileRow> matvec_profile_rows();
void set_expert_slice_matvec_enabled(bool enabled);
bool expert_slice_matvec_enabled();
void set_expert_slice_matvec_thread_count(int thread_count);
int expert_slice_matvec_thread_count();
void set_dense_matvec_enabled(bool enabled);
bool dense_matvec_enabled();
void set_dense_matvec_thread_count(int thread_count);
int dense_matvec_thread_count();
void set_dense_matvec_min_rows(std::uint64_t min_rows);
std::uint64_t dense_matvec_min_rows();
void set_dense_matvec_payload_cache_enabled(bool enabled);
bool dense_matvec_payload_cache_enabled();
void set_dense_q4_direct_dot_enabled(bool enabled);
bool dense_q4_direct_dot_enabled();
void set_dense_q4_pair_dot_enabled(bool enabled);
bool dense_q4_pair_dot_enabled();
void set_dense_q6_direct_dot_enabled(bool enabled);
bool dense_q6_direct_dot_enabled();
void set_dense_q6_pair_dot_enabled(bool enabled);
bool dense_q6_pair_dot_enabled();
void set_lm_head_q6_pair_dot_enabled(bool enabled);
bool lm_head_q6_pair_dot_enabled();
void set_q4_direct_minsum_pair_enabled(bool enabled);
bool q4_direct_minsum_pair_enabled();
void set_q4_block_meta_cache_enabled(bool enabled);
bool q4_block_meta_cache_enabled();
void set_q4_plane_layout_enabled(bool enabled);
bool q4_plane_layout_enabled();
void set_dense_q4_plane_pair_dot_enabled(bool enabled);
bool dense_q4_plane_pair_dot_enabled();
void set_small_q4_direct_dot_enabled(bool enabled);
bool small_q4_direct_dot_enabled();
void set_matvec_q8_input_reuse_enabled(bool enabled);
bool matvec_q8_input_reuse_enabled();
void set_shared_parallel_executor_enabled(bool enabled);
bool shared_parallel_executor_enabled();
void set_shared_expert_gate_up_fused_enabled(bool enabled);
bool shared_expert_gate_up_fused_enabled();
void set_selected_expert_ffn_enabled(bool enabled);
bool selected_expert_ffn_enabled();
void set_selected_expert_ffn_thread_count(int thread_count);
int selected_expert_ffn_thread_count();
void set_selected_expert_minimal_outputs_enabled(bool enabled);
bool selected_expert_minimal_outputs_enabled();
void set_selected_expert_slice_cache_enabled(bool enabled);
bool selected_expert_slice_cache_enabled();
void set_selected_expert_down_slice_cache_enabled(bool enabled);
bool selected_expert_down_slice_cache_enabled();
void set_selected_expert_down_expert_major_enabled(bool enabled);
bool selected_expert_down_expert_major_enabled();
void set_selected_expert_down_q4_pair_dot_enabled(bool enabled);
bool selected_expert_down_q4_pair_dot_enabled();
void set_selected_expert_down_q6_pair_dot_enabled(bool enabled);
bool selected_expert_down_q6_pair_dot_enabled();
void set_selected_gate_q4_direct_dot_enabled(bool enabled);
bool selected_gate_q4_direct_dot_enabled();
void set_selected_gate_q4_pair_dot_enabled(bool enabled);
bool selected_gate_q4_pair_dot_enabled();
void set_selected_gate_q4_pair_sum_dot_enabled(bool enabled);
bool selected_gate_q4_pair_sum_dot_enabled();
void set_selected_gate_q4_plane_pair_dot_enabled(bool enabled);
bool selected_gate_q4_plane_pair_dot_enabled();
std::uint64_t ggml_tensor_nbytes(std::uint32_t type,
                                 const std::vector<std::uint64_t>& dims);
std::string ggml_type_name(std::uint32_t type);
GgufLoadMapSummary validate_qwen36_load_map(const GgufModelIndex& index);
const GgufTensorInfo* find_tensor(const GgufModelIndex& index,
                                  const std::string& name);
TensorPayloadStats smoke_tensor_payload(const std::string& path,
                                        const GgufModelIndex& index,
                                        const std::string& tensor_name);
std::vector<float> decode_tensor_row(const std::string& path,
                                     const GgufModelIndex& index,
                                     const std::string& tensor_name,
                                     std::uint64_t row_index);
std::vector<float> read_f32_vector_file(const std::string& path);
VectorCompareStats compare_vectors(const std::vector<float>& lhs,
                                   const std::vector<float>& rhs,
                                   double mismatch_threshold);
std::vector<float> apply_rms_norm(const std::vector<float>& input,
                                  const std::vector<float>& weight,
                                  float epsilon);
std::vector<float> add_vectors(const std::vector<float>& lhs,
                               const std::vector<float>& rhs);
float sigmoid_scalar(float value);
std::vector<float> softmax(const std::vector<float>& logits);
std::vector<std::int32_t> top_k_indices(const std::vector<float>& values,
                                        int k);
std::vector<float> gather_values(const std::vector<float>& values,
                                 const std::vector<std::int32_t>& indexes);
std::vector<float> normalize_weights(const std::vector<float>& weights,
                                     float min_weight_sum);
RouterTopKSelection select_router_topk(const std::vector<float>& logits,
                                       int expert_used_count,
                                       float min_weight_sum);
std::vector<float> apply_swiglu_pair(const std::vector<float>& gate,
                                     const std::vector<float>& up);
std::vector<float> multiply_by_scalar(const std::vector<float>& input,
                                      float scalar);
std::vector<float> apply_expert_weights(const std::vector<float>& expert_down,
                                        const std::vector<float>& weights,
                                        std::uint64_t hidden_size);
std::vector<float> aggregate_experts(const std::vector<float>& weighted,
                                     std::uint64_t expert_count,
                                     std::uint64_t hidden_size);
std::vector<float> matvec_tensor(const std::string& path,
                                 const GgufModelIndex& index,
                                 const std::string& tensor_name,
                                 const std::vector<float>& input);
std::vector<MatvecTopKRow> top_k_matvec_tensor(
    const std::string& path,
    const GgufModelIndex& index,
    const std::string& tensor_name,
    const std::vector<float>& input,
    int k,
    int thread_count);
std::vector<float> matvec_expert_tensor(
    const std::string& path,
    const GgufModelIndex& index,
    const std::string& tensor_name,
    const std::vector<float>& input,
    const std::vector<std::int32_t>& expert_ids);
std::vector<float> matvec_expert_tensor_per_expert_input(
    const std::string& path,
    const GgufModelIndex& index,
    const std::string& tensor_name,
    const std::vector<float>& input,
    const std::vector<std::int32_t>& expert_ids);
std::vector<float> apply_swiglu_from_gate_up(
    const std::vector<float>& gate_up,
    std::uint64_t intermediate_size,
    std::uint64_t expert_count);
Qwen36LinearAttentionPreConvResult run_qwen36_linear_attention_preconv_core(
    const std::string& path,
    const GgufModelIndex& index,
    int layer_index,
    const std::vector<float>& attention_norm);
Qwen36LinearAttentionConvResult run_qwen36_linear_attention_conv_core(
    const std::string& path,
    const GgufModelIndex& index,
    int layer_index,
    const std::vector<float>& qkv_mixed,
    const std::vector<float>& conv_state);
Qwen36LinearAttentionDeltaResult run_qwen36_linear_attention_delta_core(
    const std::vector<float>& q,
    const std::vector<float>& k,
    const std::vector<float>& v,
    const std::vector<float>& gate,
    const std::vector<float>& beta,
    const std::vector<float>& recurrent_state,
    const std::vector<float>& z,
    const std::vector<float>& norm_weight,
    float rms_norm_epsilon);
Qwen36LinearAttentionPostConvResult run_qwen36_linear_attention_postconv_core(
    const std::vector<float>& conv_output_raw,
    const std::vector<float>& gate,
    const std::vector<float>& beta,
    const std::vector<float>& recurrent_state,
    const std::vector<float>& z,
    const std::vector<float>& norm_weight,
    float norm_epsilon);
Qwen36LayerPostConvResult run_qwen36_layer_with_external_conv_output(
    const std::string& path,
    const GgufModelIndex& index,
    int layer_index,
    const std::vector<float>& residual_input,
    const std::vector<float>& conv_output_raw,
    const std::vector<float>& recurrent_state,
    float rms_norm_epsilon);
Qwen36StatefulLinearAttentionLayerResult
run_qwen36_stateful_linear_attention_layer(
    const std::string& path,
    const GgufModelIndex& index,
    int layer_index,
    const std::vector<float>& residual_input,
    const std::vector<float>& conv_state,
    const std::vector<float>& recurrent_state,
    float rms_norm_epsilon);
Qwen36FullAttentionQkvProjectionResult
run_qwen36_full_attention_qkv_projection(
    const std::string& path,
    const GgufModelIndex& index,
    int layer_index,
    const std::vector<float>& residual_input,
    float rms_norm_epsilon);
Qwen36FullAttentionRopeResult run_qwen36_full_attention_rope(
    const std::vector<float>& q_normed,
    const std::vector<float>& k_normed,
    std::int32_t token_position,
    std::uint64_t head_dim,
    std::uint64_t rope_dimension_count,
    const std::vector<std::int64_t>& rope_sections,
    std::uint64_t rope_context_length,
    float rope_freq_base,
    float rope_freq_scale,
    float rope_ext_factor,
    float rope_attn_factor,
    float rope_beta_fast,
    float rope_beta_slow);
std::vector<float> build_qwen36_rope_cache(
    std::int32_t token_position,
    std::uint64_t rope_dimension_count,
    const std::vector<std::int64_t>& rope_sections,
    std::uint64_t rope_context_length,
    float rope_freq_base,
    float rope_freq_scale,
    float rope_ext_factor,
    float rope_attn_factor,
    float rope_beta_fast,
    float rope_beta_slow);
Qwen36FullAttentionGateResult run_qwen36_full_attention_gate(
    const std::vector<float>& q_full,
    const std::vector<float>& attn_pregate,
    std::uint64_t head_dim);
Qwen36FullAttentionCoreResult run_qwen36_full_attention_core(
    const std::vector<float>& q_rope,
    const std::vector<std::vector<float>>& k_history,
    const std::vector<std::vector<float>>& v_history,
    std::uint64_t head_dim,
    std::uint64_t q_head_count,
    std::uint64_t kv_head_count,
    float attention_scale);
Qwen36StatefulFullAttentionLayerResult
run_qwen36_stateful_full_attention_layer(
    const std::string& path,
    const GgufModelIndex& index,
    int layer_index,
    const std::vector<float>& residual_input,
    const std::vector<std::vector<float>>& k_history,
    const std::vector<std::vector<float>>& v_history,
    std::int32_t token_position,
    std::uint64_t head_dim,
    std::uint64_t q_head_count,
    std::uint64_t kv_head_count,
    std::uint64_t rope_dimension_count,
    const std::vector<std::int64_t>& rope_sections,
    std::uint64_t rope_context_length,
    float rope_freq_base,
    float rope_freq_scale,
    float rope_ext_factor,
    float rope_attn_factor,
    float rope_beta_fast,
    float rope_beta_slow,
    float attention_scale,
    float rms_norm_epsilon);
Qwen36MoeFfnLayerResult run_qwen36_moe_ffn_layer(
    const std::string& path,
    const GgufModelIndex& index,
    int layer_index,
    const std::vector<float>& residual_input,
    float rms_norm_epsilon);
Qwen36LayerShellResult run_qwen36_layer_with_external_attention_state(
    const std::string& path,
    const GgufModelIndex& index,
    int layer_index,
    const std::vector<float>& residual_input,
    const std::vector<float>& attention_projection_input,
    float rms_norm_epsilon);
Qwen36LoopShellResult run_qwen36_loop_with_external_attention_states(
    const std::string& path,
    const GgufModelIndex& index,
    const std::vector<float>& residual_input,
    const std::vector<std::vector<float>>& attention_projection_inputs,
    float rms_norm_epsilon);

}  // namespace iq36
