#include "intel_qwen36/packed_token_schedule.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace {

using iq36::GgufModelIndex;
using iq36::GgufTensorInfo;
using iq36::PackedTokenBufferSlot;
using iq36::PackedTokenCommand;
using iq36::PackedTokenHostBoundary;
using iq36::PackedTokenLayerKind;
using iq36::PackedTokenProgram;
using iq36::PackedTokenStageKind;

constexpr std::uint32_t kGgmlTypeF32 = 0;
constexpr std::uint32_t kGgmlTypeQ4K = 12;
constexpr std::uint32_t kGgmlTypeQ6K = 14;
constexpr int kExpertCount = 256;
constexpr std::uint64_t kFp16Bytes = 2;
constexpr std::uint64_t kF32Bytes = 4;
constexpr std::uint64_t kLinearConvStateValues = 3 * 8192;
constexpr std::uint64_t kLinearRecurrentStateValues = 32 * 128 * 128;

using LayerTensorMap =
    std::array<std::unordered_map<std::string, const GgufTensorInfo*>, 40>;

const GgufTensorInfo& RequireTensor(
    const std::unordered_map<std::string, const GgufTensorInfo*>& tensors,
    const char* name) {
  const auto it = tensors.find(name);
  if (it == tensors.end() || it->second == nullptr) {
    throw std::invalid_argument(
        std::string("packed token schedule tensor missing: ") + name);
  }
  return *it->second;
}

const GgufTensorInfo& RequireLayerTensor(
    const LayerTensorMap& layers,
    int layer_index,
    const char* suffix) {
  if (layer_index < 0 || layer_index >= iq36::kPackedTokenLayerCount) {
    throw std::invalid_argument("packed token schedule layer index out of range");
  }
  const auto& layer = layers[static_cast<std::size_t>(layer_index)];
  const auto it = layer.find(std::string(suffix));
  if (it == layer.end() || it->second == nullptr) {
    throw std::invalid_argument(
        "packed token schedule layer tensor missing: blk." +
        std::to_string(layer_index) + "." + suffix);
  }
  return *it->second;
}

void AddStream(PackedTokenCommand* command,
               const GgufTensorInfo& tensor,
               std::uint64_t active_nbytes,
               int selected_expert_count = 0) {
  if (command == nullptr || active_nbytes == 0 ||
      active_nbytes > tensor.nbytes) {
    throw std::invalid_argument("invalid packed token tensor stream");
  }
  const std::uint64_t expert_stride_nbytes =
      selected_expert_count > 0 ? tensor.nbytes / kExpertCount : 0;
  command->streams.push_back(
      {tensor.name, tensor.dims, tensor.type, tensor.absolute_offset,
       tensor.nbytes, active_nbytes, expert_stride_nbytes,
       selected_expert_count});
}

void AddWholeTensor(PackedTokenCommand* command,
                    const GgufTensorInfo& tensor) {
  AddStream(command, tensor, tensor.nbytes);
}

void AddSelectedExpertTensor(PackedTokenCommand* command,
                             const GgufTensorInfo& tensor) {
  if (tensor.dims.size() != 3 || tensor.dims[2] != kExpertCount ||
      tensor.nbytes % kExpertCount != 0) {
    throw std::invalid_argument(
        "selected-expert tensor does not have the locked expert layout");
  }
  AddStream(command, tensor,
            tensor.nbytes / kExpertCount * iq36::kPackedTokenActiveExpertCount,
            iq36::kPackedTokenActiveExpertCount);
}

PackedTokenCommand MakeCommand(
    PackedTokenStageKind stage,
    PackedTokenLayerKind layer_kind,
    int layer_index,
    std::initializer_list<PackedTokenBufferSlot> inputs,
    std::initializer_list<PackedTokenBufferSlot> outputs) {
  PackedTokenCommand command;
  command.stage = stage;
  command.layer_kind = layer_kind;
  command.layer_index = layer_index;
  command.inputs.assign(inputs.begin(), inputs.end());
  command.outputs.assign(outputs.begin(), outputs.end());
  return command;
}

bool IsFullAttentionLayer(int layer_index) {
  return (layer_index + 1) % 4 == 0;
}

PackedTokenBufferSlot CurrentHidden(int layer_index) {
  return layer_index % 2 == 0 ? PackedTokenBufferSlot::kHiddenA
                              : PackedTokenBufferSlot::kHiddenB;
}

PackedTokenBufferSlot NextHidden(int layer_index) {
  return layer_index % 2 == 0 ? PackedTokenBufferSlot::kHiddenB
                              : PackedTokenBufferSlot::kHiddenA;
}

void AddCommonFfnCommands(
    PackedTokenProgram* program,
    const LayerTensorMap& layers,
    int layer_index,
    PackedTokenLayerKind layer_kind) {
  const auto hidden = CurrentHidden(layer_index);

  auto router = MakeCommand(
      PackedTokenStageKind::kFfnRouter, layer_kind, layer_index,
      {hidden, PackedTokenBufferSlot::kAttentionScratch},
      {PackedTokenBufferSlot::kFfnNorm,
       PackedTokenBufferSlot::kRouterSelection});
  AddWholeTensor(&router, RequireLayerTensor(
                              layers, layer_index,
                              "post_attention_norm.weight"));
  AddWholeTensor(&router, RequireLayerTensor(
                              layers, layer_index, "ffn_gate_inp.weight"));
  AddWholeTensor(&router, RequireLayerTensor(
                              layers, layer_index,
                              "ffn_gate_inp_shexp.weight"));
  program->commands.push_back(std::move(router));

  auto selected = MakeCommand(
      PackedTokenStageKind::kSelectedFfn, layer_kind, layer_index,
      {PackedTokenBufferSlot::kFfnNorm,
       PackedTokenBufferSlot::kRouterSelection},
      {PackedTokenBufferSlot::kMoeScratch});
  AddSelectedExpertTensor(
      &selected, RequireLayerTensor(
                     layers, layer_index, "ffn_gate_up_exps.weight"));
  AddSelectedExpertTensor(
      &selected, RequireLayerTensor(
                     layers, layer_index, "ffn_down_exps.weight"));
  program->commands.push_back(std::move(selected));

  auto shared = MakeCommand(
      PackedTokenStageKind::kSharedFfn, layer_kind, layer_index,
      {PackedTokenBufferSlot::kFfnNorm,
       PackedTokenBufferSlot::kMoeScratch},
      {PackedTokenBufferSlot::kMoeScratch});
  AddWholeTensor(&shared, RequireLayerTensor(
                              layers, layer_index,
                              "ffn_gate_shexp.weight"));
  AddWholeTensor(&shared, RequireLayerTensor(
                              layers, layer_index,
                              "ffn_up_shexp.weight"));
  AddWholeTensor(&shared, RequireLayerTensor(
                              layers, layer_index,
                              "ffn_down_shexp.weight"));
  program->commands.push_back(std::move(shared));

  program->commands.push_back(MakeCommand(
      PackedTokenStageKind::kLayerResidual, layer_kind, layer_index,
      {hidden, PackedTokenBufferSlot::kAttentionScratch,
       PackedTokenBufferSlot::kMoeScratch},
      {NextHidden(layer_index)}));
}

void AddLinearLayerCommands(PackedTokenProgram* program,
                            const LayerTensorMap& layers,
                            int layer_index) {
  const auto hidden = CurrentHidden(layer_index);
  auto preconv = MakeCommand(
      PackedTokenStageKind::kLinearPreconv,
      PackedTokenLayerKind::kLinearSsm, layer_index, {hidden},
      {PackedTokenBufferSlot::kAttentionScratch});
  for (const char* suffix : {"attn_norm.weight", "attn_qkv.weight",
                             "ssm_alpha.weight", "ssm_beta.weight",
                             "attn_gate.weight", "ssm_conv1d.weight"}) {
    AddWholeTensor(&preconv, RequireLayerTensor(layers, layer_index, suffix));
  }
  preconv.resident_state_read_bytes = kLinearConvStateValues * kF32Bytes;
  preconv.resident_state_write_bytes = kLinearConvStateValues * kF32Bytes;
  program->commands.push_back(std::move(preconv));

  auto recurrent = MakeCommand(
      PackedTokenStageKind::kLinearRecurrent,
      PackedTokenLayerKind::kLinearSsm, layer_index,
      {PackedTokenBufferSlot::kAttentionScratch,
       PackedTokenBufferSlot::kLinearState},
      {PackedTokenBufferSlot::kAttentionScratch,
       PackedTokenBufferSlot::kLinearState});
  for (const char* suffix : {"ssm_a", "ssm_dt.bias", "ssm_norm.weight",
                             "ssm_out.weight"}) {
    AddWholeTensor(&recurrent, RequireLayerTensor(layers, layer_index, suffix));
  }
  recurrent.resident_state_read_bytes =
      kLinearRecurrentStateValues * kF32Bytes;
  recurrent.resident_state_write_bytes =
      kLinearRecurrentStateValues * kF32Bytes;
  program->commands.push_back(std::move(recurrent));

  AddCommonFfnCommands(program, layers, layer_index,
                       PackedTokenLayerKind::kLinearSsm);
}

void AddFullAttentionLayerCommands(PackedTokenProgram* program,
                                   const LayerTensorMap& layers,
                                   int layer_index,
                                   std::uint64_t context_tokens) {
  const auto hidden = CurrentHidden(layer_index);
  auto front = MakeCommand(
      PackedTokenStageKind::kAttentionFront,
      PackedTokenLayerKind::kFullAttention, layer_index, {hidden},
      {PackedTokenBufferSlot::kAttentionScratch});
  for (const char* suffix : {"attn_norm.weight", "attn_q.weight",
                             "attn_k.weight", "attn_v.weight",
                             "attn_q_norm.weight", "attn_k_norm.weight"}) {
    AddWholeTensor(&front, RequireLayerTensor(layers, layer_index, suffix));
  }
  program->commands.push_back(std::move(front));

  const auto& k = RequireLayerTensor(layers, layer_index, "attn_k.weight");
  const auto& v = RequireLayerTensor(layers, layer_index, "attn_v.weight");
  if (k.dims.size() != 2 || v.dims.size() != 2 ||
      k.dims[1] != 512 || v.dims[1] != 512) {
    throw std::invalid_argument("locked full-attention KV shape mismatch");
  }
  auto core = MakeCommand(
      PackedTokenStageKind::kFullAttentionCore,
      PackedTokenLayerKind::kFullAttention, layer_index,
      {PackedTokenBufferSlot::kAttentionScratch,
       PackedTokenBufferSlot::kKvCache},
      {PackedTokenBufferSlot::kAttentionScratch,
       PackedTokenBufferSlot::kKvCache});
  core.resident_state_read_bytes =
      context_tokens * (k.dims[1] + v.dims[1]) * kFp16Bytes;
  core.resident_state_write_bytes =
      (k.dims[1] + v.dims[1]) * kFp16Bytes;
  program->commands.push_back(std::move(core));

  auto projection = MakeCommand(
      PackedTokenStageKind::kAttentionProjection,
      PackedTokenLayerKind::kFullAttention, layer_index,
      {PackedTokenBufferSlot::kAttentionScratch},
      {PackedTokenBufferSlot::kAttentionScratch});
  AddWholeTensor(&projection, RequireLayerTensor(
                                  layers, layer_index,
                                  "attn_output.weight"));
  program->commands.push_back(std::move(projection));

  AddCommonFfnCommands(program, layers, layer_index,
                       PackedTokenLayerKind::kFullAttention);
}

bool HasSlot(const std::vector<PackedTokenBufferSlot>& slots,
             PackedTokenBufferSlot expected) {
  return std::find(slots.begin(), slots.end(), expected) != slots.end();
}

void RecordFailure(iq36::PackedTokenProgramValidation* result,
                   bool passed,
                   const char* name) {
  if (!passed) result->failed_checks.emplace_back(name);
}

}  // namespace

namespace iq36 {

const char* PackedTokenStageName(PackedTokenStageKind stage) {
  switch (stage) {
    case PackedTokenStageKind::kEmbedding:
      return "embedding";
    case PackedTokenStageKind::kLinearPreconv:
      return "linear_preconv";
    case PackedTokenStageKind::kLinearRecurrent:
      return "linear_recurrent";
    case PackedTokenStageKind::kAttentionFront:
      return "attention_front";
    case PackedTokenStageKind::kFullAttentionCore:
      return "full_attention_core";
    case PackedTokenStageKind::kAttentionProjection:
      return "attention_projection";
    case PackedTokenStageKind::kFfnRouter:
      return "ffn_router";
    case PackedTokenStageKind::kSelectedFfn:
      return "selected_ffn";
    case PackedTokenStageKind::kSharedFfn:
      return "shared_ffn";
    case PackedTokenStageKind::kLayerResidual:
      return "layer_residual";
    case PackedTokenStageKind::kLmHead:
      return "lm_head";
  }
  return "unknown";
}

const char* PackedTokenLayerKindName(PackedTokenLayerKind kind) {
  switch (kind) {
    case PackedTokenLayerKind::kLinearSsm:
      return "linear_ssm";
    case PackedTokenLayerKind::kFullAttention:
      return "full_attention";
  }
  return "unknown";
}

const char* PackedTokenBufferSlotName(PackedTokenBufferSlot slot) {
  switch (slot) {
    case PackedTokenBufferSlot::kTokenId:
      return "token_id";
    case PackedTokenBufferSlot::kHiddenA:
      return "hidden_a";
    case PackedTokenBufferSlot::kHiddenB:
      return "hidden_b";
    case PackedTokenBufferSlot::kAttentionScratch:
      return "attention_scratch";
    case PackedTokenBufferSlot::kLinearState:
      return "linear_state";
    case PackedTokenBufferSlot::kKvCache:
      return "kv_cache";
    case PackedTokenBufferSlot::kFfnNorm:
      return "ffn_norm";
    case PackedTokenBufferSlot::kRouterSelection:
      return "router_selection";
    case PackedTokenBufferSlot::kMoeScratch:
      return "moe_scratch";
    case PackedTokenBufferSlot::kTopK:
      return "top_k";
  }
  return "unknown";
}

PackedTokenProgram BuildPackedTokenProgram(const GgufModelIndex& index,
                                           std::uint64_t context_tokens) {
  if (context_tokens == 0) {
    throw std::invalid_argument("packed token context must be nonzero");
  }
  const auto load_map = validate_qwen36_load_map(index);
  if (!load_map.ready) {
    throw std::invalid_argument("packed token schedule requires locked load map");
  }

  LayerTensorMap layers;
  std::unordered_map<std::string, const GgufTensorInfo*> non_layer;
  for (const auto& tensor : index.tensors) {
    if (tensor.layer_index >= 0 &&
        tensor.layer_index < kPackedTokenLayerCount) {
      layers[static_cast<std::size_t>(tensor.layer_index)]
          .emplace(tensor.suffix, &tensor);
    } else {
      non_layer.emplace(tensor.name, &tensor);
    }
  }

  PackedTokenProgram program;
  program.context_tokens = context_tokens;
  auto embedding = MakeCommand(
      PackedTokenStageKind::kEmbedding, PackedTokenLayerKind::kLinearSsm, -1,
      {PackedTokenBufferSlot::kTokenId}, {PackedTokenBufferSlot::kHiddenA});
  const auto& embedding_tensor = RequireTensor(non_layer, "token_embd.weight");
  if (embedding_tensor.dims.size() != 2 ||
      embedding_tensor.dims[1] == 0 ||
      embedding_tensor.nbytes % embedding_tensor.dims[1] != 0) {
    throw std::invalid_argument("locked embedding row layout mismatch");
  }
  AddStream(&embedding, embedding_tensor,
            embedding_tensor.nbytes / embedding_tensor.dims[1]);
  embedding.host_boundary = PackedTokenHostBoundary::kTokenInput;
  program.commands.push_back(std::move(embedding));

  for (int layer_index = 0; layer_index < kPackedTokenLayerCount;
       ++layer_index) {
    if (IsFullAttentionLayer(layer_index)) {
      AddFullAttentionLayerCommands(
          &program, layers, layer_index, context_tokens);
      ++program.full_attention_layer_count;
    } else {
      AddLinearLayerCommands(&program, layers, layer_index);
      ++program.linear_layer_count;
    }
  }

  auto lm_head = MakeCommand(
      PackedTokenStageKind::kLmHead, PackedTokenLayerKind::kFullAttention, -1,
      {PackedTokenBufferSlot::kHiddenA}, {PackedTokenBufferSlot::kTopK});
  AddWholeTensor(&lm_head, RequireTensor(non_layer, "output_norm.weight"));
  AddWholeTensor(&lm_head, RequireTensor(non_layer, "output.weight"));
  lm_head.host_boundary = PackedTokenHostBoundary::kTopKOutput;
  program.commands.push_back(std::move(lm_head));

  std::unordered_set<std::string> covered;
  for (const auto& command : program.commands) {
    if (command.host_boundary == PackedTokenHostBoundary::kTokenInput) {
      ++program.token_input_boundary_count;
    } else if (command.host_boundary == PackedTokenHostBoundary::kTopKOutput) {
      ++program.topk_output_boundary_count;
    }
    program.resident_state_read_bytes_per_token +=
        command.resident_state_read_bytes;
    program.resident_state_write_bytes_per_token +=
        command.resident_state_write_bytes;
    if (command.stage == PackedTokenStageKind::kFullAttentionCore) {
      program.kv_history_read_bytes_per_token +=
          command.resident_state_read_bytes;
    }
    for (const auto& stream : command.streams) {
      if (!covered.insert(stream.tensor_name).second) {
        throw std::invalid_argument(
            "packed token tensor assigned to multiple stages: " +
            stream.tensor_name);
      }
      program.active_weight_bytes_per_token +=
          stream.active_nbytes_per_token;
      if (stream.ggml_type == kGgmlTypeQ4K) {
        program.q4_stream_bytes_per_token += stream.active_nbytes_per_token;
      } else if (stream.ggml_type == kGgmlTypeQ6K) {
        program.q6_stream_bytes_per_token += stream.active_nbytes_per_token;
      } else if (stream.ggml_type == kGgmlTypeF32) {
        program.f32_stream_bytes_per_token += stream.active_nbytes_per_token;
      }
    }
  }
  program.covered_tensor_count = static_cast<int>(covered.size());
  program.strict_stream_bytes_per_token =
      program.active_weight_bytes_per_token +
      program.resident_state_read_bytes_per_token +
      program.resident_state_write_bytes_per_token;
  program.admission.strict_stream_bandwidth_gb_s_min =
      static_cast<double>(program.strict_stream_bytes_per_token) / 1e9 *
      program.admission.decode_tokens_per_second_min;

  const auto validation = ValidatePackedTokenProgram(index, program);
  if (!validation.passed) {
    throw std::runtime_error(
        "packed token schedule validation failed: " +
        validation.failed_checks.front());
  }
  return program;
}

PackedTokenProgramValidation ValidatePackedTokenProgram(
    const GgufModelIndex& index,
    const PackedTokenProgram& program) {
  PackedTokenProgramValidation result;
  const auto load_map = validate_qwen36_load_map(index);
  RecordFailure(&result, load_map.ready, "locked_load_map");
  RecordFailure(&result, program.context_tokens > 0, "nonzero_context");
  RecordFailure(&result,
                program.linear_layer_count == kPackedTokenLinearLayerCount,
                "linear_layer_count");
  RecordFailure(
      &result,
      program.full_attention_layer_count ==
          kPackedTokenFullAttentionLayerCount,
      "full_attention_layer_count");
  RecordFailure(&result,
                program.covered_tensor_count ==
                    static_cast<int>(index.tensor_count),
                "tensor_coverage");
  RecordFailure(&result, program.token_input_boundary_count == 1,
                "single_token_input_boundary");
  RecordFailure(&result, program.topk_output_boundary_count == 1,
                "single_topk_output_boundary");

  std::array<int, 11> stage_counts{};
  std::unordered_set<std::string> tensor_names;
  int unexpected_host_boundaries = 0;
  bool selected_streams_are_sliced = true;
  for (const auto& command : program.commands) {
    const auto stage_index = static_cast<std::size_t>(command.stage);
    if (stage_index < stage_counts.size()) ++stage_counts[stage_index];
    if (command.host_boundary != PackedTokenHostBoundary::kNone &&
        command.stage != PackedTokenStageKind::kEmbedding &&
        command.stage != PackedTokenStageKind::kLmHead) {
      ++unexpected_host_boundaries;
    }
    for (const auto& stream : command.streams) {
      tensor_names.insert(stream.tensor_name);
      if (command.stage == PackedTokenStageKind::kSelectedFfn) {
        selected_streams_are_sliced =
            selected_streams_are_sliced &&
            stream.selected_expert_count == kPackedTokenActiveExpertCount &&
            stream.active_nbytes_per_token * kExpertCount ==
                stream.source_nbytes * kPackedTokenActiveExpertCount;
      }
    }
  }
  RecordFailure(&result, unexpected_host_boundaries == 0,
                "no_intermediate_host_boundary");
  RecordFailure(&result,
                tensor_names.size() == static_cast<std::size_t>(
                                           program.covered_tensor_count),
                "unique_tensor_assignment");
  RecordFailure(&result, selected_streams_are_sliced,
                "selected_expert_stream_slicing");
  RecordFailure(
      &result,
      stage_counts[static_cast<std::size_t>(
          PackedTokenStageKind::kLinearPreconv)] ==
          kPackedTokenLinearLayerCount,
      "linear_preconv_coverage");
  RecordFailure(
      &result,
      stage_counts[static_cast<std::size_t>(
          PackedTokenStageKind::kAttentionFront)] ==
          kPackedTokenFullAttentionLayerCount,
      "attention_front_coverage");
  RecordFailure(
      &result,
      stage_counts[static_cast<std::size_t>(
          PackedTokenStageKind::kSelectedFfn)] == kPackedTokenLayerCount,
      "selected_ffn_coverage");

  bool hidden_chain_ok = !program.commands.empty() &&
      program.commands.front().stage == PackedTokenStageKind::kEmbedding &&
      HasSlot(program.commands.front().outputs,
              PackedTokenBufferSlot::kHiddenA);
  for (int layer_index = 0;
       hidden_chain_ok && layer_index < kPackedTokenLayerCount;
       ++layer_index) {
    const auto it = std::find_if(
        program.commands.begin(), program.commands.end(),
        [layer_index](const PackedTokenCommand& command) {
          return command.layer_index == layer_index &&
                 command.stage == PackedTokenStageKind::kLayerResidual;
        });
    hidden_chain_ok = it != program.commands.end() &&
        HasSlot(it->inputs, CurrentHidden(layer_index)) &&
        HasSlot(it->outputs, NextHidden(layer_index));
  }
  RecordFailure(&result, hidden_chain_ok, "resident_hidden_state_chain");

  if (program.context_tokens == kPackedTokenAdmissionContextTokens) {
    RecordFailure(&result,
                  program.active_weight_bytes_per_token ==
                      kPackedTokenActiveWeightBytes,
                  "active_weight_bytes");
    RecordFailure(&result,
                  program.kv_history_read_bytes_per_token ==
                      kPackedTokenKvHistoryBytesAtAdmission,
                  "kv_history_bytes");
    RecordFailure(&result,
                  program.resident_state_read_bytes_per_token ==
                      kPackedTokenResidentStateReadBytesAtAdmission,
                  "resident_state_read_bytes");
    RecordFailure(&result,
                  program.resident_state_write_bytes_per_token ==
                      kPackedTokenResidentStateWriteBytesAtAdmission,
                  "resident_state_write_bytes");
    RecordFailure(&result,
                  program.strict_stream_bytes_per_token ==
                      kPackedTokenStrictStreamBytesAtAdmission,
                  "strict_stream_bytes");
  }
  RecordFailure(
      &result,
      std::abs(program.admission.wall_ms_per_token_max -
               1000.0 / program.admission.decode_tokens_per_second_min) <
          1e-9,
      "wall_budget_matches_floor");
  RecordFailure(
      &result,
      std::abs(program.admission.kernel_schedule_ms_per_token_max +
                   program.admission.host_submit_ms_per_token_max -
               program.admission.wall_ms_per_token_max) < 1e-9,
      "kernel_plus_submit_budget");
  RecordFailure(
      &result,
      std::abs(program.admission.strict_stream_bandwidth_gb_s_min -
               static_cast<double>(program.strict_stream_bytes_per_token) /
                   1e9 * program.admission.decode_tokens_per_second_min) <
          1e-9,
      "strict_bandwidth_matches_bytes");
  result.passed = result.failed_checks.empty();
  return result;
}

class PackedTokenRuntime::Impl {
 public:
  Impl(PackedTokenProgram program,
       std::unique_ptr<PackedTokenBackend> backend)
      : program_(std::move(program)), backend_(std::move(backend)) {
    if (!backend_) {
      throw std::invalid_argument("packed token backend is null");
    }
    backend_->Compile(program_);
    stats_.compile_count = 1;
  }

  std::vector<PackedTokenTopKRow> SubmitToken(
      const PackedTokenSubmission& submission) {
    if (submission.top_k == 0) {
      throw std::invalid_argument("packed token top_k must be nonzero");
    }
    auto rows = backend_->SubmitToken(submission);
    if (rows.size() != submission.top_k) {
      throw std::runtime_error("packed token backend top-k size mismatch");
    }
    for (const auto& row : rows) {
      if (row.token_id < 0 || !std::isfinite(row.logit)) {
        throw std::runtime_error("packed token backend returned invalid top-k");
      }
    }
    ++stats_.token_submission_count;
    ++stats_.host_input_boundary_count;
    ++stats_.host_output_boundary_count;
    return rows;
  }

  PackedTokenProgram program_;
  std::unique_ptr<PackedTokenBackend> backend_;
  PackedTokenRuntimeStats stats_;
};

PackedTokenRuntime::PackedTokenRuntime(
    PackedTokenProgram program,
    std::unique_ptr<PackedTokenBackend> backend)
    : impl_(std::make_unique<Impl>(std::move(program), std::move(backend))) {}

PackedTokenRuntime::~PackedTokenRuntime() = default;
PackedTokenRuntime::PackedTokenRuntime(PackedTokenRuntime&&) noexcept = default;
PackedTokenRuntime& PackedTokenRuntime::operator=(
    PackedTokenRuntime&&) noexcept = default;

std::vector<PackedTokenTopKRow> PackedTokenRuntime::SubmitToken(
    const PackedTokenSubmission& submission) {
  return impl_->SubmitToken(submission);
}

const PackedTokenProgram& PackedTokenRuntime::program() const {
  return impl_->program_;
}

PackedTokenRuntimeStats PackedTokenRuntime::stats() const {
  return impl_->stats_;
}

}  // namespace iq36
