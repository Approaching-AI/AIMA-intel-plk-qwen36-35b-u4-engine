#include "intel_qwen36/gguf_loader.hpp"
#include "intel_qwen36/resident_harness.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

constexpr int kLayerCount = 40;
constexpr int kTopK = 5;
constexpr int kHiddenSize = 2048;
constexpr int kConvKernelSize = 4;
constexpr int kConvChannelCount = 8192;
constexpr int kLinearHeadDim = 128;
constexpr int kLinearValueHeads = 32;
constexpr int kLinearRecurrentStateSize =
    kLinearHeadDim * kLinearHeadDim * kLinearValueHeads;
constexpr int kFullHeadDim = 256;
constexpr int kFullQHeadCount = 16;
constexpr int kFullKvHeadCount = 2;
constexpr float kAttentionScale = 0.0625f;
constexpr int kEosTokenId = 248044;

struct CaseSpec {
  std::string case_id;
  std::string token_file;
};

struct TopKRow {
  std::int32_t token_id = 0;
  float logit = 0.0f;
};

struct TokenResult {
  std::vector<TopKRow> topk;
};

struct LoadedCaseSpec {
  std::string case_id;
  std::vector<std::uint32_t> prompt_token_ids;
};

struct CaseRunResult {
  std::string case_id;
  std::vector<std::uint32_t> prompt_token_ids;
  std::vector<TopKRow> first_topk;
  std::vector<std::uint32_t> generated_token_ids;
  std::uint64_t prompt_prefill_ns = 0;
  std::uint64_t decode_continuation_ns = 0;
  std::uint64_t case_total_ns = 0;
};

struct TimedRunResult {
  int run_index = 0;
  std::uint64_t total_ns = 0;
  std::vector<CaseRunResult> cases;
};

struct WarmupRunResult {
  int run_index = 0;
  std::uint64_t total_ns = 0;
  std::uint64_t case_count = 0;
  std::uint64_t prompt_token_count = 0;
  std::uint64_t generated_token_count = 0;
};

struct SequenceState {
  std::vector<std::vector<float>> linear_conv;
  std::vector<std::vector<float>> linear_recurrent;
  std::vector<std::vector<std::vector<float>>> full_k;
  std::vector<std::vector<std::vector<float>>> full_v;
};

using Clock = std::chrono::steady_clock;

using StreamTokenCallback = std::function<void(
    const std::string& case_id,
    int run_index,
    const std::string& phase,
    std::size_t generated_index,
    std::uint64_t predicted_token_position,
    const std::vector<TopKRow>& topk,
    std::uint64_t elapsed_ns)>;

std::uint64_t elapsed_ns(Clock::time_point start, Clock::time_point end) {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
}

void require(bool ok, const char* message) {
  if (!ok) {
    throw std::runtime_error(message);
  }
}

std::string json_escape(const std::string& value) {
  std::string out;
  out.reserve(value.size() + 8);
  for (const char ch : value) {
    switch (ch) {
      case '\\':
        out += "\\\\";
        break;
      case '"':
        out += "\\\"";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        out += ch;
        break;
    }
  }
  return out;
}

std::string join_path(const std::string& dir, const std::string& name) {
  if (dir.empty() || dir.back() == '/') {
    return dir + name;
  }
  return dir + "/" + name;
}

std::vector<std::string> split_tabs(const std::string& line) {
  std::vector<std::string> fields;
  std::string field;
  std::istringstream stream(line);
  while (std::getline(stream, field, '\t')) {
    fields.push_back(field);
  }
  return fields;
}

std::vector<CaseSpec> read_case_specs(const std::string& token_dir) {
  std::ifstream input(join_path(token_dir, "cases.tsv"));
  require(input.good(), "cases.tsv missing");
  std::vector<CaseSpec> cases;
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty()) {
      continue;
    }
    const auto fields = split_tabs(line);
    require(fields.size() == 6, "cases.tsv row must have 6 fields");
    cases.push_back(CaseSpec{fields[0], fields[5]});
  }
  return cases;
}

std::vector<std::uint32_t> read_token_file(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  require(input.good(), "token file missing");
  const std::vector<unsigned char> bytes(
      (std::istreambuf_iterator<char>(input)),
      std::istreambuf_iterator<char>());
  require(bytes.size() % 4 == 0, "token file size is not u32-aligned");
  std::vector<std::uint32_t> tokens;
  tokens.reserve(bytes.size() / 4);
  for (std::size_t i = 0; i < bytes.size(); i += 4) {
    const std::uint32_t value =
        static_cast<std::uint32_t>(bytes[i]) |
        (static_cast<std::uint32_t>(bytes[i + 1]) << 8) |
        (static_cast<std::uint32_t>(bytes[i + 2]) << 16) |
        (static_cast<std::uint32_t>(bytes[i + 3]) << 24);
    tokens.push_back(value);
  }
  return tokens;
}

std::vector<LoadedCaseSpec> load_case_inputs(
    const std::string& token_dir,
    const std::set<std::string>& selected_cases) {
  const auto cases = read_case_specs(token_dir);
  std::vector<LoadedCaseSpec> out;
  for (const auto& item : cases) {
    if (!selected_cases.empty() && !selected_cases.count(item.case_id)) {
      continue;
    }
    auto prompt_token_ids = read_token_file(join_path(token_dir, item.token_file));
    require(!prompt_token_ids.empty(), "prompt token ids empty");
    out.push_back(LoadedCaseSpec{item.case_id, std::move(prompt_token_ids)});
  }
  return out;
}

std::uint64_t metadata_uint(const iq36::GgufModelIndex& index,
                            const std::string& key,
                            std::uint64_t fallback) {
  const auto found = index.metadata.find(key);
  if (found == index.metadata.end()) {
    return fallback;
  }
  const auto& value = found->second;
  if (value.kind == iq36::GgufMetadataValue::Kind::kUInt) {
    return value.uint_value;
  }
  if (value.kind == iq36::GgufMetadataValue::Kind::kInt &&
      value.int_value >= 0) {
    return static_cast<std::uint64_t>(value.int_value);
  }
  return fallback;
}

float metadata_float(const iq36::GgufModelIndex& index,
                     const std::string& key,
                     float fallback) {
  const auto found = index.metadata.find(key);
  if (found == index.metadata.end()) {
    return fallback;
  }
  const auto& value = found->second;
  if (value.kind == iq36::GgufMetadataValue::Kind::kFloat) {
    return static_cast<float>(value.float_value);
  }
  return fallback;
}

std::vector<std::int64_t> metadata_int_array(
    const iq36::GgufModelIndex& index,
    const std::string& key,
    const std::vector<std::int64_t>& fallback) {
  const auto found = index.metadata.find(key);
  if (found == index.metadata.end()) {
    return fallback;
  }
  const auto& value = found->second;
  if (value.kind == iq36::GgufMetadataValue::Kind::kArray &&
      !value.int_array.empty()) {
    return value.int_array;
  }
  return fallback;
}

bool has_tensor(const iq36::GgufModelIndex& index, const std::string& name) {
  return iq36::find_tensor(index, name) != nullptr;
}

std::string layer_tensor_name(int layer_index, const std::string& suffix) {
  return "blk." + std::to_string(layer_index) + "." + suffix;
}

bool is_linear_layer(const iq36::GgufModelIndex& index, int layer_index) {
  return has_tensor(index, layer_tensor_name(layer_index, "ssm_out.weight"));
}

bool is_full_attention_layer(const iq36::GgufModelIndex& index, int layer_index) {
  return has_tensor(index, layer_tensor_name(layer_index, "attn_output.weight"));
}

SequenceState make_state(const iq36::GgufModelIndex& index) {
  SequenceState state;
  state.linear_conv.resize(kLayerCount);
  state.linear_recurrent.resize(kLayerCount);
  state.full_k.resize(kLayerCount);
  state.full_v.resize(kLayerCount);
  for (int layer = 0; layer < kLayerCount; ++layer) {
    if (!is_linear_layer(index, layer)) {
      continue;
    }
    const auto* conv =
        iq36::find_tensor(index, layer_tensor_name(layer, "ssm_conv1d.weight"));
    require(conv != nullptr, "linear layer conv tensor missing");
    require(conv->dims.size() == 2, "linear conv tensor rank mismatch");
    const auto kernel = static_cast<std::size_t>(conv->dims[0]);
    const auto channels = static_cast<std::size_t>(conv->dims[1]);
    require(kernel == kConvKernelSize, "linear conv kernel size mismatch");
    require(channels == kConvChannelCount, "linear conv channel mismatch");
    state.linear_conv[layer].assign((kernel - 1) * channels, 0.0f);
    state.linear_recurrent[layer].assign(kLinearRecurrentStateSize, 0.0f);
  }
  return state;
}

std::vector<TopKRow> top_k_logits(const std::vector<float>& logits, int k) {
  std::vector<std::int32_t> indexes(logits.size());
  for (std::size_t i = 0; i < logits.size(); ++i) {
    indexes[i] = static_cast<std::int32_t>(i);
  }
  const auto limit = std::min<std::size_t>(static_cast<std::size_t>(k), indexes.size());
  std::partial_sort(
      indexes.begin(),
      indexes.begin() + static_cast<std::ptrdiff_t>(limit),
      indexes.end(),
      [&logits](std::int32_t lhs, std::int32_t rhs) {
        const float lhs_value = logits[static_cast<std::size_t>(lhs)];
        const float rhs_value = logits[static_cast<std::size_t>(rhs)];
        if (lhs_value == rhs_value) {
          return lhs < rhs;
        }
        return lhs_value > rhs_value;
      });
  std::vector<TopKRow> rows;
  rows.reserve(limit);
  for (std::size_t i = 0; i < limit; ++i) {
    const auto token_id = indexes[i];
    rows.push_back({token_id, logits[static_cast<std::size_t>(token_id)]});
  }
  return rows;
}

std::vector<float> run_full_attention_layer_inplace_history(
    const std::string& model_path,
    const iq36::GgufModelIndex& index,
    SequenceState* state,
    int layer,
    const std::vector<float>& residual,
    int token_position,
    float rms_norm_epsilon,
    std::uint64_t full_head_dim,
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
    float rope_beta_slow) {
  auto qkv = iq36::run_qwen36_full_attention_qkv_projection(
      model_path, index, layer, residual, rms_norm_epsilon);
  auto rope = iq36::run_qwen36_full_attention_rope(
      qkv.q_normed,
      qkv.k_normed,
      token_position,
      full_head_dim,
      rope_dimension_count,
      rope_sections,
      rope_context_length,
      rope_freq_base,
      rope_freq_scale,
      rope_ext_factor,
      rope_attn_factor,
      rope_beta_fast,
      rope_beta_slow);
  state->full_k[layer].push_back(std::move(rope.k_rope));
  state->full_v[layer].push_back(std::move(qkv.v));
  auto core = iq36::run_qwen36_full_attention_core(
      rope.q_rope,
      state->full_k[layer],
      state->full_v[layer],
      full_head_dim,
      q_head_count,
      kv_head_count,
      kAttentionScale);
  auto gate =
      iq36::run_qwen36_full_attention_gate(qkv.q_full, core.attn_pregate, full_head_dim);
  auto attention_output = iq36::matvec_tensor(
      model_path,
      index,
      layer_tensor_name(layer, "attn_output.weight"),
      gate.attn_gated);
  const auto attention_residual = iq36::add_vectors(residual, attention_output);
  auto ffn = iq36::run_qwen36_moe_ffn_layer(
      model_path, index, layer, attention_residual, rms_norm_epsilon);
  return std::move(ffn.residual);
}

TokenResult run_token(const std::string& model_path,
                      const iq36::GgufModelIndex& index,
                      SequenceState* state,
                      std::uint32_t token_id,
                      int token_position,
                      float rms_norm_epsilon,
                      std::uint64_t full_head_dim,
                      std::uint64_t q_head_count,
                      std::uint64_t kv_head_count,
                      std::uint64_t rope_dimension_count,
                      const std::vector<std::int64_t>& rope_sections,
                      std::uint64_t rope_context_length,
                      float rope_freq_base,
                      bool lm_head_top_k_enabled,
                      int lm_head_threads,
                      bool emit_topk,
                      int top_k_count,
                      bool full_attention_inplace_history_enabled) {
  constexpr float kRopeFreqScale = 1.0f;
  constexpr float kRopeExtFactor = 0.0f;
  constexpr float kRopeAttnFactor = 1.0f;
  constexpr float kRopeBetaFast = 32.0f;
  constexpr float kRopeBetaSlow = 1.0f;

  std::vector<float> residual =
      iq36::decode_tensor_row(model_path, index, "token_embd.weight", token_id);
  for (int layer = 0; layer < kLayerCount; ++layer) {
    if (is_linear_layer(index, layer)) {
      auto layer_result = iq36::run_qwen36_stateful_linear_attention_layer(
          model_path,
          index,
          layer,
          residual,
          state->linear_conv[layer],
          state->linear_recurrent[layer],
          rms_norm_epsilon);
      state->linear_conv[layer] = std::move(layer_result.conv.conv_state);
      state->linear_recurrent[layer] =
          std::move(layer_result.attention.recurrent_state);
      residual = std::move(layer_result.residual);
    } else if (is_full_attention_layer(index, layer)) {
      if (full_attention_inplace_history_enabled) {
        residual = run_full_attention_layer_inplace_history(
            model_path,
            index,
            state,
            layer,
            residual,
            token_position,
            rms_norm_epsilon,
            full_head_dim,
            q_head_count,
            kv_head_count,
            rope_dimension_count,
            rope_sections,
            rope_context_length,
            rope_freq_base,
            kRopeFreqScale,
            kRopeExtFactor,
            kRopeAttnFactor,
            kRopeBetaFast,
            kRopeBetaSlow);
      } else {
        auto attention = iq36::run_qwen36_stateful_full_attention_layer(
            model_path,
            index,
            layer,
            residual,
            state->full_k[layer],
            state->full_v[layer],
            token_position,
            full_head_dim,
            q_head_count,
            kv_head_count,
            rope_dimension_count,
            rope_sections,
            rope_context_length,
            rope_freq_base,
            kRopeFreqScale,
            kRopeExtFactor,
            kRopeAttnFactor,
            kRopeBetaFast,
            kRopeBetaSlow,
            kAttentionScale,
            rms_norm_epsilon);
        state->full_k[layer] = std::move(attention.k_history);
        state->full_v[layer] = std::move(attention.v_history);
        const auto attention_residual =
            iq36::add_vectors(residual, attention.attention_output);
        auto ffn = iq36::run_qwen36_moe_ffn_layer(
            model_path, index, layer, attention_residual, rms_norm_epsilon);
        residual = std::move(ffn.residual);
      }
    } else {
      throw std::runtime_error("layer has neither linear nor full attention tensors");
    }
  }

  if (!emit_topk) {
    return TokenResult{};
  }
  const auto norm_weight =
      iq36::decode_tensor_row(model_path, index, "output_norm.weight", 0);
  const auto final_norm =
      iq36::apply_rms_norm(residual, norm_weight, rms_norm_epsilon);
  if (lm_head_top_k_enabled) {
    const auto top_rows = iq36::top_k_matvec_tensor(
        model_path,
        index,
        "output.weight",
        final_norm,
        top_k_count,
        lm_head_threads);
    std::vector<TopKRow> topk;
    topk.reserve(top_rows.size());
    for (const auto& row : top_rows) {
      topk.push_back(TopKRow{row.token_id, row.value});
    }
    return TokenResult{std::move(topk)};
  }
  const auto logits = iq36::matvec_tensor(model_path, index, "output.weight", final_norm);
  return TokenResult{top_k_logits(logits, top_k_count)};
}

CaseRunResult run_case(const std::string& model_path,
                       const iq36::GgufModelIndex& index,
                       const LoadedCaseSpec& item,
                       int max_new_tokens,
                       float rms_norm_epsilon,
                       std::uint64_t full_head_dim,
                       std::uint64_t q_head_count,
                       std::uint64_t kv_head_count,
                       std::uint64_t rope_dimension_count,
                       const std::vector<std::int64_t>& rope_sections,
                       std::uint64_t rope_context_length,
                       float rope_freq_base,
                       bool lm_head_top_k_enabled,
                       int lm_head_threads,
                       bool prefill_final_logits_only_enabled,
                       bool decode_top1_only_enabled,
                       bool full_attention_inplace_history_enabled,
                       bool ignore_eos_enabled,
                       int run_index,
                       const StreamTokenCallback& stream_token_callback) {
  const auto total_begin = Clock::now();
  auto state = make_state(index);
  TokenResult next;

  const auto prefill_begin = Clock::now();
  for (std::size_t pos = 0; pos < item.prompt_token_ids.size(); ++pos) {
    const bool emit_topk =
        !prefill_final_logits_only_enabled ||
        pos + 1 == item.prompt_token_ids.size();
    next = run_token(
        model_path,
        index,
        &state,
        item.prompt_token_ids[pos],
        static_cast<int>(pos),
        rms_norm_epsilon,
        full_head_dim,
        q_head_count,
        kv_head_count,
        rope_dimension_count,
        rope_sections,
        rope_context_length,
        rope_freq_base,
        lm_head_top_k_enabled,
        lm_head_threads,
        emit_topk,
        kTopK,
        full_attention_inplace_history_enabled);
  }
  const auto prefill_end = Clock::now();
  require(!next.topk.empty(), "first token top-k is empty");
  const auto first_topk = next.topk;
  if (stream_token_callback) {
    stream_token_callback(
        item.case_id,
        run_index,
        "prefill_first",
        0,
        static_cast<std::uint64_t>(item.prompt_token_ids.size()),
        first_topk,
        elapsed_ns(prefill_begin, prefill_end));
  }

  std::vector<std::uint32_t> generated;
  generated.push_back(static_cast<std::uint32_t>(first_topk[0].token_id));
  const auto decode_begin = Clock::now();
  while (generated.size() < static_cast<std::size_t>(max_new_tokens) &&
         (ignore_eos_enabled ||
          static_cast<int>(generated.back()) != kEosTokenId)) {
    const int token_position =
        static_cast<int>(item.prompt_token_ids.size() + generated.size() - 1);
    const auto decode_token_begin = Clock::now();
    next = run_token(
        model_path,
        index,
        &state,
        generated.back(),
        token_position,
        rms_norm_epsilon,
        full_head_dim,
        q_head_count,
        kv_head_count,
        rope_dimension_count,
        rope_sections,
        rope_context_length,
        rope_freq_base,
        lm_head_top_k_enabled,
        lm_head_threads,
        true,
        decode_top1_only_enabled ? 1 : kTopK,
        full_attention_inplace_history_enabled);
    const auto decode_token_end = Clock::now();
    require(!next.topk.empty(), "decode top-k is empty");
    if (stream_token_callback) {
      stream_token_callback(
          item.case_id,
          run_index,
          "decode",
          generated.size(),
          static_cast<std::uint64_t>(item.prompt_token_ids.size() +
                                     generated.size()),
          next.topk,
          elapsed_ns(decode_token_begin, decode_token_end));
    }
    generated.push_back(static_cast<std::uint32_t>(next.topk[0].token_id));
  }
  const auto decode_end = Clock::now();
  const auto total_end = Clock::now();

  CaseRunResult result;
  result.case_id = item.case_id;
  result.prompt_token_ids = item.prompt_token_ids;
  result.first_topk = first_topk;
  result.generated_token_ids = std::move(generated);
  result.prompt_prefill_ns = elapsed_ns(prefill_begin, prefill_end);
  result.decode_continuation_ns = elapsed_ns(decode_begin, decode_end);
  result.case_total_ns = elapsed_ns(total_begin, total_end);
  return result;
}

void write_int_array(const std::vector<std::uint32_t>& values) {
  std::cout << "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0) {
      std::cout << ",";
    }
    std::cout << values[i];
  }
  std::cout << "]";
}

void write_int32_array(const std::vector<std::int32_t>& values) {
  std::cout << "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0) {
      std::cout << ",";
    }
    std::cout << values[i];
  }
  std::cout << "]";
}

void write_topk(const std::vector<TopKRow>& rows) {
  std::cout << "[";
  for (std::size_t i = 0; i < rows.size(); ++i) {
    if (i != 0) {
      std::cout << ",";
    }
    std::cout << "{\"logit\":" << rows[i].logit
              << ",\"token_id\":" << rows[i].token_id << "}";
  }
  std::cout << "]";
}

void write_topk_ids(const std::vector<TopKRow>& rows) {
  std::vector<std::int32_t> ids;
  ids.reserve(rows.size());
  for (const auto& row : rows) {
    ids.push_back(row.token_id);
  }
  write_int32_array(ids);
}

void write_oracle_bundle_stats(const iq36::OracleBundleStats& stats) {
  std::cout << "{";
  std::cout << "\"boundary_input_rows\":" << stats.boundary_input_rows << ",";
  std::cout << "\"boundary_output_rows\":" << stats.boundary_output_rows << ",";
  std::cout << "\"teacher_forced_distribution_rows\":"
            << stats.teacher_forced_distribution_rows << ",";
  std::cout << "\"token_topk_rows\":" << stats.token_topk_rows;
  std::cout << "}";
}

std::vector<iq36::ResidentTopKRow> resident_topk_rows(
    const std::vector<TopKRow>& rows) {
  std::vector<iq36::ResidentTopKRow> out;
  out.reserve(rows.size());
  for (const auto& row : rows) {
    out.push_back(iq36::ResidentTopKRow{row.token_id, row.logit});
  }
  return out;
}

iq36::ResidentTokenEvent make_resident_token_event(
    const std::string& case_id,
    int run_index,
    const std::string& phase,
    std::size_t generated_index,
    std::uint64_t predicted_token_position,
    const std::vector<TopKRow>& topk,
    std::uint64_t elapsed_ns) {
  iq36::ResidentTokenEvent event;
  event.case_id = case_id;
  event.run_index = run_index;
  event.phase = phase;
  event.generated_index = generated_index;
  event.predicted_token_position = predicted_token_position;
  event.topk = resident_topk_rows(topk);
  event.elapsed_ns = elapsed_ns;
  return event;
}

void write_case_result(const CaseRunResult& result) {
  std::cout << "{";
  std::cout << "\"case_id\":\"" << json_escape(result.case_id) << "\",";
  std::cout << "\"first_token_top_k\":";
  write_topk(result.first_topk);
  std::cout << ",\"first_token_top_logprob_id_signature\":";
  write_topk_ids(result.first_topk);
  std::cout << ",\"generated_token_ids\":";
  write_int_array(result.generated_token_ids);
  std::cout << ",\"prompt_token_count\":" << result.prompt_token_ids.size() << ",";
  std::cout << "\"prompt_token_ids\":";
  write_int_array(result.prompt_token_ids);
  std::cout << ",\"timing_ns\":{";
  std::cout << "\"case_total\":" << result.case_total_ns << ",";
  std::cout << "\"decode_continuation\":" << result.decode_continuation_ns << ",";
  std::cout << "\"prompt_prefill\":" << result.prompt_prefill_ns;
  std::cout << "}";
  std::cout << "}";
}

void write_case_results(const std::vector<CaseRunResult>& cases) {
  std::cout << "[";
  for (std::size_t i = 0; i < cases.size(); ++i) {
    if (i != 0) {
      std::cout << ",";
    }
    write_case_result(cases[i]);
  }
  std::cout << "]";
}

void write_warmup_runs(const std::vector<WarmupRunResult>& runs) {
  std::cout << "[";
  for (std::size_t i = 0; i < runs.size(); ++i) {
    if (i != 0) {
      std::cout << ",";
    }
    std::cout << "{";
    std::cout << "\"case_count\":" << runs[i].case_count << ",";
    std::cout << "\"generated_token_count\":" << runs[i].generated_token_count << ",";
    std::cout << "\"prompt_token_count\":" << runs[i].prompt_token_count << ",";
    std::cout << "\"run_index\":" << runs[i].run_index << ",";
    std::cout << "\"total_ns\":" << runs[i].total_ns;
    std::cout << "}";
  }
  std::cout << "]";
}

void write_timed_runs(const std::vector<TimedRunResult>& runs) {
  std::cout << "[";
  for (std::size_t i = 0; i < runs.size(); ++i) {
    if (i != 0) {
      std::cout << ",";
    }
    std::cout << "{";
    std::cout << "\"cases\":";
    write_case_results(runs[i].cases);
    std::cout << ",\"run_index\":" << runs[i].run_index << ",";
    std::cout << "\"total_ns\":" << runs[i].total_ns;
    std::cout << "}";
  }
  std::cout << "]";
}

iq36::ResidentCaseResult make_resident_case_result(
    const CaseRunResult& result) {
  iq36::ResidentCaseResult out;
  out.case_id = result.case_id;
  out.prompt_token_ids = result.prompt_token_ids;
  out.first_topk = resident_topk_rows(result.first_topk);
  out.generated_token_ids = result.generated_token_ids;
  out.prompt_prefill_ns = result.prompt_prefill_ns;
  out.decode_continuation_ns = result.decode_continuation_ns;
  out.case_total_ns = result.case_total_ns;
  return out;
}

iq36::ResidentDoneEvent make_resident_done_event(
    const std::vector<CaseRunResult>& emitted_cases,
    std::uint64_t max_new_tokens,
    std::uint64_t process_total_ns,
    bool q4_plane_layout_enabled,
    bool selected_expert_down_q6_pair_dot_enabled,
    bool dense_q6_pair_dot_enabled,
    std::uint64_t resident_harness_load_ns) {
  iq36::ResidentDoneEvent event;
  event.cases.reserve(emitted_cases.size());
  for (const auto& emitted_case : emitted_cases) {
    event.cases.push_back(make_resident_case_result(emitted_case));
  }
  event.max_new_tokens = max_new_tokens;
  event.process_total_ns = process_total_ns;
  event.q4_plane_layout_enabled = q4_plane_layout_enabled;
  event.selected_expert_down_q6_pair_dot_enabled =
      selected_expert_down_q6_pair_dot_enabled;
  event.dense_q6_pair_dot_enabled = dense_q6_pair_dot_enabled;
  event.resident_harness_load_ns = resident_harness_load_ns;
  return event;
}

void write_cache_stats(const iq36::ResidentTensorCacheStats& stats) {
  std::cout << "{";
  std::cout << "\"decoded_row_cached_bytes\":" << stats.decoded_row_cached_bytes << ",";
  std::cout << "\"decoded_row_cached_values\":" << stats.decoded_row_cached_values << ",";
  std::cout << "\"decoded_row_hits\":" << stats.decoded_row_hits << ",";
  std::cout << "\"decoded_row_misses\":" << stats.decoded_row_misses << ",";
  std::cout << "\"enabled\":" << (stats.enabled ? "true" : "false") << ",";
  std::cout << "\"expert_slice_cached_bytes\":"
            << stats.expert_slice_cached_bytes << ",";
  std::cout << "\"expert_slice_hits\":" << stats.expert_slice_hits << ",";
  std::cout << "\"expert_slice_misses\":" << stats.expert_slice_misses << ",";
  std::cout << "\"q4_plane_cached_bytes\":"
            << stats.q4_plane_cached_bytes << ",";
  std::cout << "\"q4_plane_hits\":" << stats.q4_plane_hits << ",";
  std::cout << "\"q4_plane_misses\":" << stats.q4_plane_misses << ",";
  std::cout << "\"q4_plane_repack_ns\":"
            << stats.q4_plane_repack_ns << ",";
  std::cout << "\"tensor_payload_cached_bytes\":"
            << stats.tensor_payload_cached_bytes << ",";
  std::cout << "\"tensor_payload_hits\":" << stats.tensor_payload_hits << ",";
  std::cout << "\"tensor_payload_misses\":" << stats.tensor_payload_misses;
  std::cout << "}";
}

void write_matvec_profile(const std::vector<iq36::MatvecProfileRow>& rows) {
  std::cout << "[";
  for (std::size_t i = 0; i < rows.size(); ++i) {
    if (i != 0) {
      std::cout << ",";
    }
    const auto& row = rows[i];
    const auto average_ns =
        row.call_count == 0 ? 0 : row.total_ns / row.call_count;
    std::cout << "{";
    std::cout << "\"average_ns\":" << average_ns << ",";
    std::cout << "\"call_count\":" << row.call_count << ",";
    std::cout << "\"input_value_count\":" << row.input_value_count << ",";
    std::cout << "\"max_ns\":" << row.max_ns << ",";
    std::cout << "\"op\":\"" << json_escape(row.op) << "\",";
    std::cout << "\"output_value_count\":" << row.output_value_count << ",";
    std::cout << "\"row_count\":" << row.row_count << ",";
    std::cout << "\"tensor_name\":\"" << json_escape(row.tensor_name) << "\",";
    std::cout << "\"total_ns\":" << row.total_ns;
    std::cout << "}";
  }
  std::cout << "]";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    require(argc >= 4,
            "usage: iq36-native-candidate-jsonl <model.gguf> "
            "<token-input-dir> <max-new-tokens> "
            "[--warmup-runs N] [--timed-runs N] "
            "[--resident-cache] [--profile-matvec] "
            "[--stream-sse-events] "
            "[--resident-harness-bundle PATH] "
            "[--prefill-final-logits-only] "
            "[--decode-top1-only] "
            "[--ignore-eos] "
            "[--full-attention-inplace-history] "
            "[--lm-head-top-k] [--lm-head-threads N] "
            "[--lm-head-q6-pair-dot] "
            "[--expert-slice-matvec] [--expert-slice-threads N] "
            "[--dense-matvec] [--dense-matvec-threads N] "
            "[--dense-matvec-min-rows N] [--dense-matvec-payload-cache] "
            "[--dense-q4-direct-dot] [--dense-q4-pair-dot] "
            "[--dense-q6-direct-dot] "
            "[--q4-block-meta-cache] [--q4-plane-layout] "
            "[--small-q4-direct-dot] "
            "[--matvec-q8-input-reuse] "
            "[--shared-parallel-executor] "
            "[--shared-expert-gate-up-fused] "
            "[--selected-expert-ffn] [--selected-expert-ffn-threads N] "
            "[--selected-expert-minimal-outputs] "
            "[--selected-expert-slice-cache] "
            "[--selected-expert-down-slice-cache] "
            "[--selected-expert-down-q4-pair-dot] "
            "[--selected-expert-down-q6-pair-dot] "
            "[--selected-gate-q4-direct-dot] [--selected-gate-q4-pair-dot] "
            "[--selected-gate-q4-pair-sum-dot] "
            "[--selected-gate-q4-plane-pair-dot] "
            "[case_id...]");
    const std::string model_path = argv[1];
    const std::string token_dir = argv[2];
    const int max_new_tokens = std::stoi(argv[3]);
    require(max_new_tokens >= 1 && max_new_tokens <= 512,
            "max-new-tokens must be 1..512");

    int warmup_runs = 0;
    int timed_runs_requested = 1;
    bool resident_cache_enabled = false;
    bool matvec_profile_enabled = false;
    bool stream_sse_events_enabled = false;
    bool prefill_final_logits_only_enabled = false;
    bool decode_top1_only_enabled = false;
    bool ignore_eos_enabled = false;
    bool full_attention_inplace_history_enabled = false;
    bool lm_head_top_k_enabled = false;
    bool lm_head_q6_pair_dot_enabled = false;
    bool expert_slice_matvec_enabled = false;
    bool dense_matvec_enabled = false;
    bool dense_matvec_payload_cache_enabled = false;
    bool dense_q4_direct_dot_enabled = false;
    bool dense_q4_pair_dot_enabled = false;
    bool dense_q6_direct_dot_enabled = false;
    bool dense_q6_pair_dot_enabled = false;
    bool q4_direct_minsum_pair_enabled = false;
    bool q4_block_meta_cache_enabled = false;
    bool q4_plane_layout_enabled = false;
    bool dense_q4_plane_pair_dot_enabled = false;
    bool small_q4_direct_dot_enabled = false;
    bool matvec_q8_input_reuse_enabled = false;
    bool shared_parallel_executor_enabled = false;
    bool shared_expert_gate_up_fused_enabled = false;
    bool selected_expert_ffn_enabled = false;
    bool selected_expert_minimal_outputs_enabled = false;
    bool selected_expert_slice_cache_enabled = false;
    bool selected_expert_down_slice_cache_enabled = false;
    bool selected_expert_down_expert_major_enabled = false;
    bool selected_expert_down_q4_pair_dot_enabled = false;
    bool selected_expert_down_q6_pair_dot_enabled = false;
    bool selected_gate_q4_direct_dot_enabled = false;
    bool selected_gate_q4_pair_dot_enabled = false;
    bool selected_gate_q4_pair_sum_dot_enabled = false;
    bool selected_gate_q4_plane_pair_dot_enabled = false;
    int lm_head_threads = 1;
    int expert_slice_threads = 1;
    int dense_matvec_threads = 1;
    int selected_expert_ffn_threads = 1;
    std::uint64_t dense_matvec_min_rows = 1024;
    std::string resident_harness_bundle_path;
    std::set<std::string> selected_cases;
    for (int i = 4; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--warmup-runs") {
        require(i + 1 < argc, "--warmup-runs requires a value");
        warmup_runs = std::stoi(argv[++i]);
      } else if (arg == "--timed-runs") {
        require(i + 1 < argc, "--timed-runs requires a value");
        timed_runs_requested = std::stoi(argv[++i]);
      } else if (arg == "--case-id") {
        require(i + 1 < argc, "--case-id requires a value");
        selected_cases.insert(argv[++i]);
      } else if (arg == "--resident-cache") {
        resident_cache_enabled = true;
      } else if (arg == "--profile-matvec") {
        matvec_profile_enabled = true;
      } else if (arg == "--stream-sse-events") {
        stream_sse_events_enabled = true;
      } else if (arg == "--resident-harness-bundle") {
        require(i + 1 < argc, "--resident-harness-bundle requires a value");
        resident_harness_bundle_path = argv[++i];
      } else if (arg == "--prefill-final-logits-only") {
        prefill_final_logits_only_enabled = true;
      } else if (arg == "--decode-top1-only") {
        decode_top1_only_enabled = true;
      } else if (arg == "--ignore-eos") {
        ignore_eos_enabled = true;
      } else if (arg == "--full-attention-inplace-history") {
        full_attention_inplace_history_enabled = true;
      } else if (arg == "--lm-head-top-k") {
        lm_head_top_k_enabled = true;
      } else if (arg == "--lm-head-threads") {
        require(i + 1 < argc, "--lm-head-threads requires a value");
        lm_head_threads = std::stoi(argv[++i]);
      } else if (arg == "--lm-head-q6-pair-dot") {
        lm_head_q6_pair_dot_enabled = true;
      } else if (arg == "--expert-slice-matvec") {
        expert_slice_matvec_enabled = true;
      } else if (arg == "--expert-slice-threads") {
        require(i + 1 < argc, "--expert-slice-threads requires a value");
        expert_slice_threads = std::stoi(argv[++i]);
      } else if (arg == "--dense-matvec") {
        dense_matvec_enabled = true;
      } else if (arg == "--dense-matvec-threads") {
        require(i + 1 < argc, "--dense-matvec-threads requires a value");
        dense_matvec_threads = std::stoi(argv[++i]);
      } else if (arg == "--dense-matvec-min-rows") {
        require(i + 1 < argc, "--dense-matvec-min-rows requires a value");
        dense_matvec_min_rows = std::stoull(argv[++i]);
      } else if (arg == "--dense-matvec-payload-cache") {
        dense_matvec_payload_cache_enabled = true;
      } else if (arg == "--dense-q4-direct-dot") {
        dense_q4_direct_dot_enabled = true;
      } else if (arg == "--dense-q4-pair-dot") {
        dense_q4_pair_dot_enabled = true;
      } else if (arg == "--dense-q6-direct-dot") {
        dense_q6_direct_dot_enabled = true;
      } else if (arg == "--dense-q6-pair-dot") {
        dense_q6_pair_dot_enabled = true;
      } else if (arg == "--q4-direct-minsum-pair") {
        q4_direct_minsum_pair_enabled = true;
      } else if (arg == "--q4-block-meta-cache") {
        q4_block_meta_cache_enabled = true;
      } else if (arg == "--q4-plane-layout") {
        q4_plane_layout_enabled = true;
      } else if (arg == "--dense-q4-plane-pair-dot") {
        dense_q4_plane_pair_dot_enabled = true;
      } else if (arg == "--small-q4-direct-dot") {
        small_q4_direct_dot_enabled = true;
      } else if (arg == "--matvec-q8-input-reuse") {
        matvec_q8_input_reuse_enabled = true;
      } else if (arg == "--shared-parallel-executor") {
        shared_parallel_executor_enabled = true;
      } else if (arg == "--shared-expert-gate-up-fused") {
        shared_expert_gate_up_fused_enabled = true;
      } else if (arg == "--selected-expert-ffn") {
        selected_expert_ffn_enabled = true;
      } else if (arg == "--selected-expert-ffn-threads") {
        require(i + 1 < argc, "--selected-expert-ffn-threads requires a value");
        selected_expert_ffn_threads = std::stoi(argv[++i]);
      } else if (arg == "--selected-expert-minimal-outputs") {
        selected_expert_minimal_outputs_enabled = true;
      } else if (arg == "--selected-expert-slice-cache") {
        selected_expert_slice_cache_enabled = true;
      } else if (arg == "--selected-expert-down-slice-cache") {
        selected_expert_down_slice_cache_enabled = true;
      } else if (arg == "--selected-expert-down-expert-major") {
        selected_expert_down_expert_major_enabled = true;
      } else if (arg == "--selected-expert-down-q4-pair-dot") {
        selected_expert_down_q4_pair_dot_enabled = true;
      } else if (arg == "--selected-expert-down-q6-pair-dot") {
        selected_expert_down_q6_pair_dot_enabled = true;
      } else if (arg == "--selected-gate-q4-direct-dot") {
        selected_gate_q4_direct_dot_enabled = true;
      } else if (arg == "--selected-gate-q4-pair-dot") {
        selected_gate_q4_pair_dot_enabled = true;
      } else if (arg == "--selected-gate-q4-pair-sum-dot") {
        selected_gate_q4_pair_sum_dot_enabled = true;
      } else if (arg == "--selected-gate-q4-plane-pair-dot") {
        selected_gate_q4_plane_pair_dot_enabled = true;
      } else {
        selected_cases.insert(arg);
      }
    }
    require(warmup_runs >= 0 && warmup_runs <= 8, "warmup-runs must be 0..8");
    require(timed_runs_requested >= 1 && timed_runs_requested <= 8,
            "timed-runs must be 1..8");
    require(!stream_sse_events_enabled || timed_runs_requested == 1,
            "--stream-sse-events requires timed-runs=1");
    require(lm_head_threads >= 1 && lm_head_threads <= 256,
            "lm-head-threads must be 1..256");
    require(expert_slice_threads >= 1 && expert_slice_threads <= 256,
            "expert-slice-threads must be 1..256");
    require(dense_matvec_threads >= 1 && dense_matvec_threads <= 256,
            "dense-matvec-threads must be 1..256");
    require(dense_matvec_min_rows >= 1 && dense_matvec_min_rows <= 1048576,
            "dense-matvec-min-rows must be 1..1048576");
    require(selected_expert_ffn_threads >= 1 && selected_expert_ffn_threads <= 256,
            "selected-expert-ffn-threads must be 1..256");
    require(!q4_plane_layout_enabled || resident_cache_enabled,
            "--q4-plane-layout requires --resident-cache");
    require(!dense_q4_plane_pair_dot_enabled || q4_plane_layout_enabled,
            "--dense-q4-plane-pair-dot requires --q4-plane-layout");
    require(!dense_q4_plane_pair_dot_enabled || dense_q4_direct_dot_enabled,
            "--dense-q4-plane-pair-dot requires --dense-q4-direct-dot");
    require(!selected_gate_q4_plane_pair_dot_enabled || q4_plane_layout_enabled,
            "--selected-gate-q4-plane-pair-dot requires --q4-plane-layout");
    require(
        !selected_gate_q4_plane_pair_dot_enabled ||
            selected_gate_q4_direct_dot_enabled ||
            selected_gate_q4_pair_dot_enabled ||
            selected_gate_q4_pair_sum_dot_enabled,
        "--selected-gate-q4-plane-pair-dot requires a selected-gate Q4 route");
    iq36::set_resident_tensor_cache_enabled(resident_cache_enabled);
    iq36::reset_resident_tensor_cache();
    iq36::set_matvec_profile_enabled(matvec_profile_enabled);
    iq36::reset_matvec_profile();
    iq36::set_expert_slice_matvec_enabled(expert_slice_matvec_enabled);
    iq36::set_expert_slice_matvec_thread_count(expert_slice_threads);
    iq36::set_dense_matvec_enabled(dense_matvec_enabled);
    iq36::set_dense_matvec_thread_count(dense_matvec_threads);
    iq36::set_dense_matvec_min_rows(dense_matvec_min_rows);
    iq36::set_dense_matvec_payload_cache_enabled(
        dense_matvec_payload_cache_enabled);
    iq36::set_dense_q4_direct_dot_enabled(dense_q4_direct_dot_enabled);
    iq36::set_dense_q4_pair_dot_enabled(dense_q4_pair_dot_enabled);
    iq36::set_dense_q6_direct_dot_enabled(dense_q6_direct_dot_enabled);
    iq36::set_dense_q6_pair_dot_enabled(dense_q6_pair_dot_enabled);
    iq36::set_q4_direct_minsum_pair_enabled(q4_direct_minsum_pair_enabled);
    iq36::set_q4_block_meta_cache_enabled(q4_block_meta_cache_enabled);
    iq36::set_q4_plane_layout_enabled(q4_plane_layout_enabled);
    iq36::set_dense_q4_plane_pair_dot_enabled(dense_q4_plane_pair_dot_enabled);
    iq36::set_small_q4_direct_dot_enabled(small_q4_direct_dot_enabled);
    iq36::set_matvec_q8_input_reuse_enabled(matvec_q8_input_reuse_enabled);
    iq36::set_shared_parallel_executor_enabled(shared_parallel_executor_enabled);
    iq36::set_shared_expert_gate_up_fused_enabled(
        shared_expert_gate_up_fused_enabled);
    iq36::set_selected_expert_ffn_enabled(selected_expert_ffn_enabled);
    iq36::set_selected_expert_ffn_thread_count(selected_expert_ffn_threads);
    iq36::set_selected_expert_minimal_outputs_enabled(
        selected_expert_minimal_outputs_enabled);
    iq36::set_selected_expert_slice_cache_enabled(
        selected_expert_slice_cache_enabled);
    iq36::set_selected_expert_down_slice_cache_enabled(
        selected_expert_down_slice_cache_enabled);
    iq36::set_selected_expert_down_expert_major_enabled(
        selected_expert_down_expert_major_enabled);
    iq36::set_selected_expert_down_q4_pair_dot_enabled(
        selected_expert_down_q4_pair_dot_enabled);
    iq36::set_selected_expert_down_q6_pair_dot_enabled(
        selected_expert_down_q6_pair_dot_enabled);
    iq36::set_selected_gate_q4_direct_dot_enabled(
        selected_gate_q4_direct_dot_enabled);
    iq36::set_selected_gate_q4_pair_dot_enabled(
        selected_gate_q4_pair_dot_enabled);
    iq36::set_selected_gate_q4_pair_sum_dot_enabled(
        selected_gate_q4_pair_sum_dot_enabled);
    iq36::set_selected_gate_q4_plane_pair_dot_enabled(
        selected_gate_q4_plane_pair_dot_enabled);
    iq36::set_lm_head_q6_pair_dot_enabled(lm_head_q6_pair_dot_enabled);

    const auto process_begin = Clock::now();
    iq36::ResidentHarness resident_harness;
    std::uint64_t resident_harness_load_ns = 0;
    if (!resident_harness_bundle_path.empty()) {
      const auto resident_harness_begin = Clock::now();
      resident_harness.load(model_path, resident_harness_bundle_path);
      resident_harness_load_ns =
          elapsed_ns(resident_harness_begin, Clock::now());
    }

    const auto model_load_begin = Clock::now();
    const auto index = iq36::parse_gguf_model_index(model_path);
    const auto load_map = iq36::validate_qwen36_load_map(index);
    require(load_map.ready, "GGUF load map is not ready");
    const auto model_load_end = Clock::now();

    const auto rms_norm_epsilon = metadata_float(
        index, "qwen35moe.attention.layer_norm_rms_epsilon", 1e-6f);
    const auto head_dim = metadata_uint(
        index, "qwen35moe.attention.key_length", kFullHeadDim);
    const auto value_length = metadata_uint(
        index, "qwen35moe.attention.value_length", kFullHeadDim);
    const auto q_head_count = metadata_uint(
        index, "qwen35moe.attention.head_count", kFullQHeadCount);
    const auto kv_head_count = metadata_uint(
        index, "qwen35moe.attention.head_count_kv", kFullKvHeadCount);
    const auto rope_dimension_count = metadata_uint(
        index, "qwen35moe.rope.dimension_count", 64);
    const auto rope_context_length = metadata_uint(
        index, "qwen35moe.context_length", 262144);
    const auto rope_sections = metadata_int_array(
        index, "qwen35moe.rope.dimension_sections", {11, 11, 10, 0});
    const float rope_freq_base = metadata_float(
        index, "qwen35moe.rope.freq_base", 10000000.0f);
    require(head_dim == value_length, "full-attention key/value length mismatch");

    const auto token_load_begin = Clock::now();
    const auto cases = load_case_inputs(token_dir, selected_cases);
    const auto token_load_end = Clock::now();
    require(!cases.empty(), "no selected cases");

    std::vector<WarmupRunResult> warmup_results;
    warmup_results.reserve(static_cast<std::size_t>(warmup_runs));
    for (int run_index = 0; run_index < warmup_runs; ++run_index) {
      const auto warmup_begin = Clock::now();
      WarmupRunResult row;
      row.run_index = run_index;
      for (const auto& item : cases) {
        auto result = run_case(
            model_path,
            index,
            item,
            max_new_tokens,
            rms_norm_epsilon,
            head_dim,
            q_head_count,
            kv_head_count,
            rope_dimension_count,
            rope_sections,
            rope_context_length,
            rope_freq_base,
            lm_head_top_k_enabled,
            lm_head_threads,
            prefill_final_logits_only_enabled,
            decode_top1_only_enabled,
            full_attention_inplace_history_enabled,
            ignore_eos_enabled,
            run_index,
            StreamTokenCallback{});
        ++row.case_count;
        row.prompt_token_count += result.prompt_token_ids.size();
        row.generated_token_count += result.generated_token_ids.size();
      }
      row.total_ns = elapsed_ns(warmup_begin, Clock::now());
      warmup_results.push_back(row);
    }

    std::vector<TimedRunResult> timed_results;
    timed_results.reserve(static_cast<std::size_t>(timed_runs_requested));
    for (int run_index = 0; run_index < timed_runs_requested; ++run_index) {
      const auto run_begin = Clock::now();
      TimedRunResult timed;
      timed.run_index = run_index;
      timed.cases.reserve(cases.size());
      if (stream_sse_events_enabled) {
        iq36::ResidentStreamingSessionConfig session_config;
        session_config.session_id =
            "native-candidate-run-" + std::to_string(run_index);
        session_config.run_index = run_index;
        session_config.max_new_tokens =
            static_cast<std::uint64_t>(max_new_tokens);
        session_config.expected_case_count = cases.size();
        resident_harness.begin_streaming_session(session_config);
      }
      for (const auto& item : cases) {
        auto stream_callback =
            stream_sse_events_enabled
                ? StreamTokenCallback(
                      [&resident_harness](
                          const std::string& case_id,
                          int run_index,
                          const std::string& phase,
                          std::size_t generated_index,
                          std::uint64_t predicted_token_position,
                          const std::vector<TopKRow>& topk,
                          std::uint64_t elapsed_ns) {
                        resident_harness.emit_sse_session_token_event(
                            std::cout,
                            make_resident_token_event(
                                case_id,
                                run_index,
                                phase,
                                generated_index,
                                predicted_token_position,
                                topk,
                                elapsed_ns));
                      })
                : StreamTokenCallback{};
        timed.cases.push_back(run_case(
            model_path,
            index,
            item,
            max_new_tokens,
            rms_norm_epsilon,
            head_dim,
            q_head_count,
            kv_head_count,
            rope_dimension_count,
            rope_sections,
            rope_context_length,
            rope_freq_base,
            lm_head_top_k_enabled,
            lm_head_threads,
            prefill_final_logits_only_enabled,
            decode_top1_only_enabled,
            full_attention_inplace_history_enabled,
            ignore_eos_enabled,
            run_index,
            stream_callback));
      }
      timed.total_ns = elapsed_ns(run_begin, Clock::now());
      timed_results.push_back(std::move(timed));
    }

    const auto process_end = Clock::now();
    const auto& emitted_cases = timed_results.back().cases;
    std::cout << std::setprecision(17);
    if (stream_sse_events_enabled) {
      resident_harness.emit_sse_session_done_event(
          std::cout,
          make_resident_done_event(
              emitted_cases,
              static_cast<std::uint64_t>(max_new_tokens),
              elapsed_ns(process_begin, process_end),
              q4_plane_layout_enabled,
              selected_expert_down_q6_pair_dot_enabled,
              dense_q6_pair_dot_enabled,
              resident_harness_load_ns));
      return !emitted_cases.empty() ? 0 : 2;
    }
    std::cout << "{";
    std::cout << "\"cache_state\":\"";
    std::cout << (warmup_runs > 0 ? "single_process_hot_after_internal_warmup"
                                  : "single_process_no_internal_warmup");
    std::cout << "\",";
    std::cout << "\"cases\":";
    write_case_results(emitted_cases);
    std::cout << ",\"dense_matvec_enabled\":"
              << (dense_matvec_enabled ? "true" : "false") << ",";
    std::cout << "\"decode_top1_only_enabled\":"
              << (decode_top1_only_enabled ? "true" : "false") << ",";
    std::cout << "\"dense_matvec_min_rows\":" << dense_matvec_min_rows << ",";
    std::cout << "\"dense_matvec_payload_cache_enabled\":"
              << (dense_matvec_payload_cache_enabled ? "true" : "false")
              << ",";
    std::cout << "\"dense_q4_direct_dot_enabled\":"
              << (dense_q4_direct_dot_enabled ? "true" : "false") << ",";
    std::cout << "\"dense_q4_pair_dot_enabled\":"
              << (dense_q4_pair_dot_enabled ? "true" : "false") << ",";
    std::cout << "\"dense_q6_direct_dot_enabled\":"
              << (dense_q6_direct_dot_enabled ? "true" : "false") << ",";
    std::cout << "\"dense_q6_pair_dot_enabled\":"
              << (dense_q6_pair_dot_enabled ? "true" : "false") << ",";
    std::cout << "\"q4_direct_minsum_pair_enabled\":"
              << (q4_direct_minsum_pair_enabled ? "true" : "false") << ",";
    std::cout << "\"q4_block_meta_cache_enabled\":"
              << (q4_block_meta_cache_enabled ? "true" : "false") << ",";
    std::cout << "\"q4_plane_layout_enabled\":"
              << (q4_plane_layout_enabled ? "true" : "false") << ",";
    std::cout << "\"dense_q4_plane_pair_dot_enabled\":"
              << (dense_q4_plane_pair_dot_enabled ? "true" : "false") << ",";
    std::cout << "\"small_q4_direct_dot_enabled\":"
              << (small_q4_direct_dot_enabled ? "true" : "false") << ",";
    std::cout << "\"dense_matvec_threads\":" << dense_matvec_threads << ",";
    std::cout << "\"emitted_case_count\":" << emitted_cases.size() << ",";
    std::cout << "\"expert_slice_matvec_enabled\":"
              << (expert_slice_matvec_enabled ? "true" : "false") << ",";
    std::cout << "\"expert_slice_threads\":" << expert_slice_threads << ",";
    std::cout << "\"full_attention_inplace_history_enabled\":"
              << (full_attention_inplace_history_enabled ? "true" : "false")
              << ",";
    std::cout << "\"lm_head_threads\":" << lm_head_threads << ",";
    std::cout << "\"ignore_eos_enabled\":"
              << (ignore_eos_enabled ? "true" : "false") << ",";
    std::cout << "\"lm_head_top_k_enabled\":"
              << (lm_head_top_k_enabled ? "true" : "false") << ",";
    std::cout << "\"lm_head_q6_pair_dot_enabled\":"
              << (lm_head_q6_pair_dot_enabled ? "true" : "false") << ",";
    std::cout << "\"load_map_ready\":true,";
    std::cout << "\"max_new_tokens\":" << max_new_tokens << ",";
    std::cout << "\"matvec_profile_enabled\":"
              << (matvec_profile_enabled ? "true" : "false") << ",";
    std::cout << "\"matvec_q8_input_reuse_enabled\":"
              << (matvec_q8_input_reuse_enabled ? "true" : "false") << ",";
    std::cout << "\"matvec_profile\":";
    write_matvec_profile(iq36::matvec_profile_rows());
    std::cout << ",";
    std::cout << "\"model_path\":\"" << json_escape(model_path) << "\",";
    std::cout << "\"process_timing_ns\":{";
    std::cout << "\"model_parse_and_validate\":"
              << elapsed_ns(model_load_begin, model_load_end) << ",";
    std::cout << "\"process_total\":" << elapsed_ns(process_begin, process_end) << ",";
    std::cout << "\"resident_harness_load\":"
              << resident_harness_load_ns << ",";
    std::cout << "\"token_input_load\":"
              << elapsed_ns(token_load_begin, token_load_end);
    std::cout << "},";
    std::cout << "\"prefill_final_logits_only_enabled\":"
              << (prefill_final_logits_only_enabled ? "true" : "false")
              << ",";
    std::cout << "\"resident_tensor_cache_enabled\":"
              << (resident_cache_enabled ? "true" : "false") << ",";
    std::cout << "\"resident_tensor_cache_stats\":";
    write_cache_stats(iq36::resident_tensor_cache_stats());
    std::cout << ",";
    std::cout << "\"resident_harness_loaded\":"
              << (resident_harness.loaded() ? "true" : "false") << ",";
    std::cout << "\"resident_harness_oracle_bundle_stats\":";
    write_oracle_bundle_stats(resident_harness.oracle_bundle_stats());
    std::cout << ",";
    std::cout << "\"schema_version\":\"intel-qwen36-engine-native-candidate-jsonl-v0\",";
    std::cout << "\"selected_expert_ffn_enabled\":"
              << (selected_expert_ffn_enabled ? "true" : "false") << ",";
    std::cout << "\"selected_expert_ffn_threads\":"
              << selected_expert_ffn_threads << ",";
    std::cout << "\"selected_expert_minimal_outputs_enabled\":"
              << (selected_expert_minimal_outputs_enabled ? "true" : "false")
              << ",";
    std::cout << "\"selected_expert_slice_cache_enabled\":"
              << (selected_expert_slice_cache_enabled ? "true" : "false")
              << ",";
    std::cout << "\"selected_expert_down_slice_cache_enabled\":"
              << (selected_expert_down_slice_cache_enabled ? "true" : "false")
              << ",";
    std::cout << "\"selected_expert_down_expert_major_enabled\":"
              << (selected_expert_down_expert_major_enabled ? "true" : "false")
              << ",";
    std::cout << "\"selected_expert_down_q4_pair_dot_enabled\":"
              << (selected_expert_down_q4_pair_dot_enabled ? "true" : "false")
              << ",";
    std::cout << "\"selected_expert_down_q6_pair_dot_enabled\":"
              << (selected_expert_down_q6_pair_dot_enabled ? "true" : "false")
              << ",";
    std::cout << "\"selected_gate_q4_direct_dot_enabled\":"
              << (selected_gate_q4_direct_dot_enabled ? "true" : "false") << ",";
    std::cout << "\"selected_gate_q4_pair_dot_enabled\":"
              << (selected_gate_q4_pair_dot_enabled ? "true" : "false") << ",";
    std::cout << "\"selected_gate_q4_pair_sum_dot_enabled\":"
              << (selected_gate_q4_pair_sum_dot_enabled ? "true" : "false")
              << ",";
    std::cout << "\"selected_gate_q4_plane_pair_dot_enabled\":"
              << (selected_gate_q4_plane_pair_dot_enabled ? "true" : "false")
              << ",";
    std::cout << "\"shared_parallel_executor_enabled\":"
              << (shared_parallel_executor_enabled ? "true" : "false") << ",";
    std::cout << "\"shared_expert_gate_up_fused_enabled\":"
              << (shared_expert_gate_up_fused_enabled ? "true" : "false")
              << ",";
    std::cout << "\"timed_run_count\":" << timed_results.size() << ",";
    std::cout << "\"timed_runs\":";
    write_timed_runs(timed_results);
    std::cout << ",\"timing_schema_version\":\"intel-qwen36-engine-timing-v0\",";
    std::cout << "\"warmup_run_count\":" << warmup_results.size() << ",";
    std::cout << "\"warmup_runs\":";
    write_warmup_runs(warmup_results);
    std::cout << "}\n";
    return !emitted_cases.empty() ? 0 : 2;
  } catch (const std::exception& exc) {
    std::cerr << "iq36-native-candidate-jsonl: " << exc.what() << "\n";
    return 1;
  }
}
