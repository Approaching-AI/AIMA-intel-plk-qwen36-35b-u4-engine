#pragma once

#include "intel_qwen36/gguf_loader.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace iq36 {

constexpr int kPackedTokenLayerCount = 40;
constexpr int kPackedTokenLinearLayerCount = 30;
constexpr int kPackedTokenFullAttentionLayerCount = 10;
constexpr int kPackedTokenActiveExpertCount = 8;
constexpr std::uint64_t kPackedTokenAdmissionContextTokens = 1024;
constexpr std::uint64_t kPackedTokenActiveWeightBytes = 1'975'676'544ULL;
constexpr std::uint64_t kPackedTokenKvHistoryBytesAtAdmission = 20'971'520ULL;
constexpr std::uint64_t kPackedTokenResidentStateReadBytesAtAdmission =
    86'835'200ULL;
constexpr std::uint64_t kPackedTokenResidentStateWriteBytesAtAdmission =
    65'884'160ULL;
constexpr std::uint64_t kPackedTokenStrictStreamBytesAtAdmission =
    2'128'395'904ULL;

struct PackedTokenAdmission {
  double decode_tokens_per_second_min = 49.8;
  double wall_ms_per_token_max = 20.080321285140563;
  double kernel_schedule_ms_per_token_max = 19.980321285140562;
  double host_submit_ms_per_token_max = 0.1;
  double strict_stream_bandwidth_gb_s_min = 105.99411601919999;
};

enum class PackedTokenLayerKind {
  kLinearSsm,
  kFullAttention,
};

enum class PackedTokenStageKind {
  kEmbedding,
  kLinearPreconv,
  kLinearRecurrent,
  kAttentionFront,
  kFullAttentionCore,
  kAttentionProjection,
  kFfnRouter,
  kSelectedFfn,
  kSharedFfn,
  kLayerResidual,
  kLmHead,
};

enum class PackedTokenBufferSlot {
  kTokenId,
  kHiddenA,
  kHiddenB,
  kAttentionScratch,
  kLinearState,
  kKvCache,
  kFfnNorm,
  kRouterSelection,
  kMoeScratch,
  kTopK,
};

enum class PackedTokenHostBoundary {
  kNone,
  kTokenInput,
  kTopKOutput,
};

struct PackedTokenTensorStream {
  std::string tensor_name;
  std::vector<std::uint64_t> dims;
  std::uint32_t ggml_type = 0;
  std::uint64_t absolute_offset = 0;
  std::uint64_t source_nbytes = 0;
  std::uint64_t active_nbytes_per_token = 0;
  std::uint64_t expert_stride_nbytes = 0;
  int selected_expert_count = 0;
};

struct PackedTokenCommand {
  PackedTokenStageKind stage = PackedTokenStageKind::kEmbedding;
  PackedTokenLayerKind layer_kind = PackedTokenLayerKind::kLinearSsm;
  int layer_index = -1;
  std::vector<PackedTokenBufferSlot> inputs;
  std::vector<PackedTokenBufferSlot> outputs;
  std::vector<PackedTokenTensorStream> streams;
  std::uint64_t resident_state_read_bytes = 0;
  std::uint64_t resident_state_write_bytes = 0;
  PackedTokenHostBoundary host_boundary = PackedTokenHostBoundary::kNone;
};

struct PackedTokenProgram {
  std::uint64_t context_tokens = 0;
  PackedTokenAdmission admission;
  std::vector<PackedTokenCommand> commands;
  std::uint64_t active_weight_bytes_per_token = 0;
  std::uint64_t kv_history_read_bytes_per_token = 0;
  std::uint64_t resident_state_read_bytes_per_token = 0;
  std::uint64_t resident_state_write_bytes_per_token = 0;
  std::uint64_t strict_stream_bytes_per_token = 0;
  std::uint64_t q4_stream_bytes_per_token = 0;
  std::uint64_t q6_stream_bytes_per_token = 0;
  std::uint64_t f32_stream_bytes_per_token = 0;
  int linear_layer_count = 0;
  int full_attention_layer_count = 0;
  int covered_tensor_count = 0;
  int token_input_boundary_count = 0;
  int topk_output_boundary_count = 0;
};

struct PackedTokenProgramValidation {
  bool passed = false;
  std::vector<std::string> failed_checks;
};

PackedTokenProgram BuildPackedTokenProgram(
    const GgufModelIndex& index,
    std::uint64_t context_tokens = kPackedTokenAdmissionContextTokens);
PackedTokenProgramValidation ValidatePackedTokenProgram(
    const GgufModelIndex& index,
    const PackedTokenProgram& program);

const char* PackedTokenStageName(PackedTokenStageKind stage);
const char* PackedTokenLayerKindName(PackedTokenLayerKind kind);
const char* PackedTokenBufferSlotName(PackedTokenBufferSlot slot);

struct PackedTokenSubmission {
  std::uint32_t token_id = 0;
  std::uint64_t token_position = 0;
  std::size_t top_k = 5;
};

struct PackedTokenTopKRow {
  std::int32_t token_id = 0;
  float logit = 0.0f;
};

class PackedTokenBackend {
 public:
  virtual ~PackedTokenBackend() = default;

  // Compile is the only schedule-build boundary. Implementations retain all
  // weights, scratch buffers, recurrent state, KV state, and the command list.
  virtual void Compile(const PackedTokenProgram& program) = 0;

  // One call is one host submission for the whole token. No API exists for an
  // intermediate stage read or per-layer host callback.
  virtual std::vector<PackedTokenTopKRow> SubmitToken(
      const PackedTokenSubmission& submission) = 0;
};

struct PackedTokenRuntimeStats {
  std::uint64_t compile_count = 0;
  std::uint64_t token_submission_count = 0;
  std::uint64_t host_input_boundary_count = 0;
  std::uint64_t host_output_boundary_count = 0;
  std::uint64_t intermediate_host_read_count = 0;
};

class PackedTokenRuntime {
 public:
  PackedTokenRuntime(PackedTokenProgram program,
                     std::unique_ptr<PackedTokenBackend> backend);
  ~PackedTokenRuntime();

  PackedTokenRuntime(const PackedTokenRuntime&) = delete;
  PackedTokenRuntime& operator=(const PackedTokenRuntime&) = delete;
  PackedTokenRuntime(PackedTokenRuntime&&) noexcept;
  PackedTokenRuntime& operator=(PackedTokenRuntime&&) noexcept;

  std::vector<PackedTokenTopKRow> SubmitToken(
      const PackedTokenSubmission& submission);
  const PackedTokenProgram& program() const;
  PackedTokenRuntimeStats stats() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace iq36
