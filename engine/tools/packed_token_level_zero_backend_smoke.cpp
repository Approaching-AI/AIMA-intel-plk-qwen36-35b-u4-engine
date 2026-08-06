#include "intel_qwen36/gguf_loader.hpp"
#include "intel_qwen36/packed_token_level_zero_backend.hpp"
#include "intel_qwen36/packed_token_schedule.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

constexpr int kLayerCount = 40;
constexpr std::size_t kLinearConvStateValues = 3 * 8192;
constexpr std::size_t kLinearRecurrentStateValues = 32 * 128 * 128;
constexpr std::size_t kFullKvValues = 512;
constexpr std::size_t kVocabularySize = 248320;

[[noreturn]] void Die(const std::string& message) {
  throw std::runtime_error(message);
}

void Require(bool condition, const std::string& message) {
  if (!condition) Die(message);
}

bool IsFullAttentionLayer(int layer) {
  return (layer + 1) % 4 == 0;
}

bool UseInt8Block32KvGqa() {
  const char* value = std::getenv("IQ36_INT8_BLOCK32_KV_GQA");
  return value != nullptr && std::string(value) != "0";
}

bool SequentialCpuDistributionCheck() {
  const char* value =
      std::getenv("IQ36_SEQUENTIAL_CPU_DISTRIBUTION_CHECK");
  return value != nullptr && std::string(value) != "0";
}

std::vector<std::uint32_t> ReadTokens(const std::string& path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  Require(static_cast<bool>(input), "failed to open token input");
  const auto bytes = static_cast<std::size_t>(input.tellg());
  Require(bytes >= 2 * sizeof(std::uint32_t) &&
              bytes % sizeof(std::uint32_t) == 0,
          "token input size is invalid");
  std::vector<std::uint32_t> tokens(bytes / sizeof(std::uint32_t));
  input.seekg(0, std::ios::beg);
  input.read(reinterpret_cast<char*>(tokens.data()),
             static_cast<std::streamsize>(bytes));
  Require(static_cast<bool>(input), "failed to read token input");
  return tokens;
}

struct CpuState {
  std::array<std::vector<float>, kLayerCount> linear_conv;
  std::array<std::vector<float>, kLayerCount> linear_recurrent;
  std::array<std::vector<std::vector<float>>, kLayerCount> full_k;
  std::array<std::vector<std::vector<float>>, kLayerCount> full_v;
};

CpuState MakeCpuState() {
  CpuState state;
  for (int layer = 0; layer < kLayerCount; ++layer) {
    if (!IsFullAttentionLayer(layer)) {
      state.linear_conv[layer].assign(kLinearConvStateValues, 0.0f);
      state.linear_recurrent[layer].assign(
          kLinearRecurrentStateValues, 0.0f);
    }
  }
  return state;
}

iq36::PackedTokenStateSnapshot Snapshot(const CpuState& state) {
  iq36::PackedTokenStateSnapshot result;
  for (int layer = 0; layer < kLayerCount; ++layer) {
    if (IsFullAttentionLayer(layer)) {
      for (const auto& token : state.full_k[layer]) {
        Require(token.size() == kFullKvValues, "CPU K history size mismatch");
        result.full_k_history[layer].insert(
            result.full_k_history[layer].end(), token.begin(), token.end());
      }
      for (const auto& token : state.full_v[layer]) {
        Require(token.size() == kFullKvValues, "CPU V history size mismatch");
        result.full_v_history[layer].insert(
            result.full_v_history[layer].end(), token.begin(), token.end());
      }
    } else {
      result.linear_conv[layer] = state.linear_conv[layer];
      result.linear_recurrent[layer] = state.linear_recurrent[layer];
    }
  }
  return result;
}

std::vector<iq36::MatvecTopKRow> CpuToken(
    const std::string& model,
    const iq36::GgufModelIndex& index,
    CpuState* state,
    std::uint32_t token,
    std::uint32_t position,
    const iq36::PackedTokenLevelZeroConfig& config,
    std::vector<float>* full_logits = nullptr) {
  auto residual = iq36::decode_tensor_row(
      model, index, "token_embd.weight", token);
  for (int layer = 0; layer < kLayerCount; ++layer) {
    if (!IsFullAttentionLayer(layer)) {
      auto result = iq36::run_qwen36_stateful_linear_attention_layer(
          model, index, layer, residual, state->linear_conv[layer],
          state->linear_recurrent[layer], config.rms_norm_epsilon);
      state->linear_conv[layer] = std::move(result.conv.conv_state);
      state->linear_recurrent[layer] =
          std::move(result.attention.recurrent_state);
      residual = std::move(result.residual);
    } else {
      auto attention = iq36::run_qwen36_stateful_full_attention_layer(
          model, index, layer, residual, state->full_k[layer],
          state->full_v[layer], static_cast<std::int32_t>(position),
          config.full_head_dim, config.full_q_head_count,
          config.full_kv_head_count, config.rope_dimension_count,
          config.rope_sections, config.rope_context_length,
          config.rope_freq_base, config.rope_freq_scale,
          config.rope_ext_factor, config.rope_attn_factor,
          config.rope_beta_fast, config.rope_beta_slow,
          config.attention_scale, config.rms_norm_epsilon);
      state->full_k[layer] = std::move(attention.k_history);
      state->full_v[layer] = std::move(attention.v_history);
      const auto attention_residual =
          iq36::add_vectors(residual, attention.attention_output);
      auto ffn = iq36::run_qwen36_moe_ffn_layer(
          model, index, layer, attention_residual,
          config.rms_norm_epsilon);
      residual = std::move(ffn.residual);
    }
  }
  const auto norm_weight = iq36::decode_tensor_row(
      model, index, "output_norm.weight", 0);
  const auto final_norm = iq36::apply_rms_norm(
      residual, norm_weight, config.rms_norm_epsilon);
  if (full_logits != nullptr) {
    *full_logits = iq36::matvec_tensor(
        model, index, "output.weight", final_norm);
  }
  return iq36::top_k_matvec_tensor(
      model, index, "output.weight", final_norm, 8, 16);
}

struct DistributionSummary {
  double kld = 0.0;
  double logits_cosine = 0.0;
  bool top1_matches = false;
};

struct DistributionLadderSummary {
  double max_kld = 0.0;
  double mean_kld = 0.0;
  double min_logits_cosine = 1.0;
  double top1_rate = 0.0;
  bool required_checks_passed = false;
};

struct VectorComparison {
  std::size_t value_count = 0;
  double cosine = 0.0;
  double relative_l2 = 0.0;
  double max_abs = 0.0;
  bool finite = false;
};

struct StateComparisonRow {
  int layer = -1;
  std::string kind;
  VectorComparison numeric;
};

DistributionSummary CompareDistribution(
    const std::vector<float>& reference,
    const std::vector<float>& candidate) {
  Require(!reference.empty() && reference.size() == candidate.size(),
          "distribution logit size mismatch");
  const auto logsumexp = [](const std::vector<float>& values) {
    const double maximum = *std::max_element(values.begin(), values.end());
    double sum = 0.0;
    for (const float value : values) {
      Require(std::isfinite(value), "distribution logit is not finite");
      sum += std::exp(static_cast<double>(value) - maximum);
    }
    return maximum + std::log(sum);
  };
  const double reference_lse = logsumexp(reference);
  const double candidate_lse = logsumexp(candidate);
  double kld = 0.0;
  double dot = 0.0;
  double reference_l2 = 0.0;
  double candidate_l2 = 0.0;
  for (std::size_t index = 0; index < reference.size(); ++index) {
    const double reference_value = reference[index];
    const double candidate_value = candidate[index];
    const double reference_logp = reference_value - reference_lse;
    kld += std::exp(reference_logp) *
        (reference_logp - (candidate_value - candidate_lse));
    dot += reference_value * candidate_value;
    reference_l2 += reference_value * reference_value;
    candidate_l2 += candidate_value * candidate_value;
  }
  const auto reference_top1 = static_cast<std::size_t>(std::distance(
      reference.begin(), std::max_element(reference.begin(), reference.end())));
  const auto candidate_top1 = static_cast<std::size_t>(std::distance(
      candidate.begin(), std::max_element(candidate.begin(), candidate.end())));
  return {
      kld,
      dot / (std::sqrt(reference_l2) * std::sqrt(candidate_l2)),
      reference_top1 == candidate_top1,
  };
}

DistributionLadderSummary SummarizeDistributionLadder(
    const std::vector<DistributionSummary>& steps) {
  Require(!steps.empty(), "distribution ladder is empty");
  DistributionLadderSummary summary;
  double kld_sum = 0.0;
  std::size_t top1_matches = 0;
  for (const auto& step : steps) {
    summary.max_kld = std::max(summary.max_kld, step.kld);
    summary.min_logits_cosine =
        std::min(summary.min_logits_cosine, step.logits_cosine);
    kld_sum += step.kld;
    top1_matches += step.top1_matches;
  }
  summary.mean_kld = kld_sum / static_cast<double>(steps.size());
  summary.top1_rate =
      static_cast<double>(top1_matches) / static_cast<double>(steps.size());
  summary.required_checks_passed =
      summary.max_kld < 0.005 && summary.top1_rate >= 0.99;
  return summary;
}

bool DistributionPass(const DistributionSummary& summary) {
  return summary.kld <= 0.005 && summary.top1_matches;
}

VectorComparison CompareVectors(const std::vector<float>& reference,
                                const std::vector<float>& candidate) {
  Require(!reference.empty() && reference.size() == candidate.size(),
          "state vector size mismatch");
  double dot = 0.0;
  double reference_l2 = 0.0;
  double candidate_l2 = 0.0;
  double difference_l2 = 0.0;
  double max_abs = 0.0;
  bool finite = true;
  for (std::size_t index = 0; index < reference.size(); ++index) {
    const double lhs = reference[index];
    const double rhs = candidate[index];
    finite = finite && std::isfinite(lhs) && std::isfinite(rhs);
    dot += lhs * rhs;
    reference_l2 += lhs * lhs;
    candidate_l2 += rhs * rhs;
    const double difference = lhs - rhs;
    difference_l2 += difference * difference;
    max_abs = std::max(max_abs, std::abs(difference));
  }
  const double denominator = std::sqrt(reference_l2 * candidate_l2);
  const double cosine = denominator > 0.0
      ? dot / denominator
      : (reference_l2 == 0.0 && candidate_l2 == 0.0 ? 1.0 : 0.0);
  const double relative_l2 = reference_l2 > 0.0
      ? std::sqrt(difference_l2 / reference_l2)
      : (candidate_l2 == 0.0 ? 0.0 : std::numeric_limits<double>::infinity());
  return {reference.size(), cosine, relative_l2, max_abs, finite};
}

void WriteDistributionComparison(const DistributionSummary& summary) {
  std::cout << "{\"kld\":" << summary.kld
            << ",\"logits_cosine\":" << summary.logits_cosine
            << ",\"required_checks_passed\":"
            << DistributionPass(summary)
            << ",\"top1_matches\":" << summary.top1_matches << "}";
}

void WriteVectorComparison(const VectorComparison& summary) {
  std::cout << "{\"cosine\":" << summary.cosine
            << ",\"finite\":" << summary.finite
            << ",\"max_abs\":" << summary.max_abs
            << ",\"relative_l2\":" << summary.relative_l2
            << ",\"value_count\":" << summary.value_count << "}";
}

std::vector<float> ReadFloatFile(const std::filesystem::path& path,
                                 std::size_t expected_values) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  Require(static_cast<bool>(input), "failed to open float file: " + path.string());
  const auto bytes = static_cast<std::size_t>(input.tellg());
  Require(bytes == expected_values * sizeof(float),
          "float file size mismatch: " + path.string());
  std::vector<float> result(expected_values);
  input.seekg(0, std::ios::beg);
  input.read(reinterpret_cast<char*>(result.data()),
             static_cast<std::streamsize>(bytes));
  Require(static_cast<bool>(input), "failed to read float file: " + path.string());
  return result;
}

iq36::PackedTokenStateSnapshot ReadImportedState(
    const std::filesystem::path& directory, std::size_t prefix_tokens) {
  iq36::PackedTokenStateSnapshot state;
  for (int layer = 0; layer < kLayerCount; ++layer) {
    const auto suffix = std::to_string(layer) + ".f32";
    if (IsFullAttentionLayer(layer)) {
      const auto values = prefix_tokens * kFullKvValues;
      state.full_k_history[layer] = ReadFloatFile(
          directory / ("full_k_" + suffix), values);
      state.full_v_history[layer] = ReadFloatFile(
          directory / ("full_v_" + suffix), values);
    } else {
      state.linear_conv[layer] = ReadFloatFile(
          directory / ("linear_conv_" + suffix), kLinearConvStateValues);
      state.linear_recurrent[layer] = ReadFloatFile(
          directory / ("linear_recurrent_" + suffix),
          kLinearRecurrentStateValues);
    }
  }
  return state;
}

void WriteCpuTopK(const std::vector<iq36::MatvecTopKRow>& rows) {
  std::cout << "[";
  for (std::size_t i = 0; i < rows.size(); ++i) {
    if (i != 0) std::cout << ",";
    std::cout << "{\"id\":" << rows[i].token_id
              << ",\"value\":" << rows[i].value << "}";
  }
  std::cout << "]";
}

void WriteGpuTopK(const std::vector<iq36::PackedTokenTopKRow>& rows) {
  std::cout << "[";
  for (std::size_t i = 0; i < rows.size(); ++i) {
    if (i != 0) std::cout << ",";
    std::cout << "{\"id\":" << rows[i].token_id
              << ",\"value\":" << rows[i].logit << "}";
  }
  std::cout << "]";
}

void WriteKernelProfile(
    const std::vector<iq36::PackedTokenLevelZeroProfileRow>& rows) {
  std::unordered_map<std::string, std::pair<double, std::size_t>> totals;
  for (const auto& row : rows) {
    auto& total = totals[row.kernel];
    total.first += row.device_ms;
    ++total.second;
  }
  std::vector<std::pair<std::string, std::pair<double, std::size_t>>> sorted(
      totals.begin(), totals.end());
  std::sort(sorted.begin(), sorted.end(), [](const auto& lhs, const auto& rhs) {
    return lhs.second.first > rhs.second.first;
  });
  std::cout << "[";
  for (std::size_t index = 0; index < sorted.size(); ++index) {
    if (index != 0) std::cout << ",";
    std::cout << "{\"count\":" << sorted[index].second.second
              << ",\"device_ms\":" << sorted[index].second.first
              << ",\"kernel\":\"" << sorted[index].first << "\"}";
  }
  std::cout << "]";
}

void WriteDoubleArray(const std::vector<double>& values) {
  std::cout << "[";
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0) std::cout << ",";
    std::cout << values[index];
  }
  std::cout << "]";
}

void WriteIntArray(const std::vector<std::int32_t>& values) {
  std::cout << "[";
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0) std::cout << ",";
    std::cout << values[index];
  }
  std::cout << "]";
}

double Median(std::vector<double> values) {
  Require(!values.empty(), "median input is empty");
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

double Percentile(std::vector<double> values, double percentile) {
  Require(!values.empty(), "percentile input is empty");
  Require(percentile >= 0.0 && percentile <= 1.0,
          "percentile must be in [0, 1]");
  std::sort(values.begin(), values.end());
  const auto index = static_cast<std::size_t>(std::ceil(
      percentile * static_cast<double>(values.size()))) - 1U;
  return values[std::min(index, values.size() - 1U)];
}

int RunSyntheticContextDiagnostic(
    const std::string& model, const std::string& module,
    const iq36::GgufModelIndex& index,
    const std::vector<std::uint32_t>& tokens,
    std::uint64_t context_tokens, std::uint32_t sample_tokens) {
  constexpr std::uint64_t kReservedOutputTokens = 512;
  Require(context_tokens >= 2048 && context_tokens <= 131072,
          "synthetic context must be in [2048, 131072]");
  Require(sample_tokens >= 1 && sample_tokens <= kReservedOutputTokens,
          "sample token count must be in [1, 512]");
  Require(context_tokens + kReservedOutputTokens <= 262144,
          "synthetic context plus reserved output exceeds model context");

  iq36::PackedTokenLevelZeroConfig config;
  config.state_capacity_tokens = context_tokens + kReservedOutputTokens;
  config.use_int8_block32_kv_gqa = UseInt8Block32KvGqa();
  config.profile_kernel_times = std::getenv("IQ36_PROFILE_KERNELS") != nullptr;
  const auto zero_state = MakeCpuState();
  iq36::PackedTokenLevelZeroBackend backend(model, module, config);
  backend.LoadState(Snapshot(zero_state));
  backend.Compile(iq36::BuildPackedTokenProgram(
      index, config.state_capacity_tokens));

  const auto seed_token = tokens.back();
  (void)backend.SubmitToken({seed_token, context_tokens - 1U, 8});
  backend.LoadState(Snapshot(zero_state));

  std::vector<double> device_samples;
  std::vector<double> host_samples;
  std::vector<double> wall_samples;
  device_samples.reserve(sample_tokens);
  host_samples.reserve(sample_tokens);
  wall_samples.reserve(sample_tokens);
  std::uint32_t feed = seed_token;
  std::int32_t first_generated = -1;
  std::int32_t last_generated = -1;
  for (std::uint32_t step = 0; step < sample_tokens; ++step) {
    const auto rows = backend.SubmitToken(
        {feed, context_tokens - 1U + step, 8});
    Require(!rows.empty(), "synthetic context top-k is empty");
    feed = static_cast<std::uint32_t>(rows.front().token_id);
    if (step == 0U) first_generated = rows.front().token_id;
    last_generated = rows.front().token_id;
    const auto timing = backend.last_timing();
    device_samples.push_back(timing.device_ms);
    host_samples.push_back(timing.host_submit_ms);
    wall_samples.push_back(timing.wall_ms);
  }
  const auto timing = backend.last_timing();
  const double wall_total_ms = std::accumulate(
      wall_samples.begin(), wall_samples.end(), 0.0);
  const double device_total_ms = std::accumulate(
      device_samples.begin(), device_samples.end(), 0.0);
  const double wall_median_ms = Median(wall_samples);
  const bool pass =
      timing.command_list_record_count == 1 && timing.kernel_count > 252 &&
      std::all_of(device_samples.begin(), device_samples.end(),
                  [](double value) { return std::isfinite(value) && value > 0.0; }) &&
      std::all_of(host_samples.begin(), host_samples.end(),
                  [](double value) { return std::isfinite(value) && value > 0.0; }) &&
      std::all_of(wall_samples.begin(), wall_samples.end(),
                  [](double value) { return std::isfinite(value) && value > 0.0; });

  std::cout << std::boolalpha << std::setprecision(12) << "{"
            << "\"context_tokens\":" << context_tokens
            << ",\"correctness_applicable\":false"
            << ",\"device_ms_median\":" << Median(device_samples)
            << ",\"device_ms_samples\":";
  WriteDoubleArray(device_samples);
  std::cout
            << ",\"device_ms_total\":" << device_total_ms
            << ",\"device_name\":\"" << backend.device_name() << "\""
            << ",\"first_generated_token_id\":" << first_generated
            << ",\"host_submit_ms_max\":"
            << *std::max_element(host_samples.begin(), host_samples.end())
            << ",\"host_submit_ms_median\":" << Median(host_samples)
            << ",\"host_submit_ms_p95\":" << Percentile(host_samples, 0.95)
            << ",\"host_submit_ms_samples\":";
  WriteDoubleArray(host_samples);
  std::cout
            << ",\"kernel_count\":" << timing.kernel_count
            << ",\"kernel_profile\":";
  WriteKernelProfile(timing.kernel_profile);
  std::cout << ",\"last_generated_token_id\":" << last_generated
            << ",\"full_kv_dtype\":\""
            << (config.use_int8_block32_kv_gqa
                    ? "int8_block32_fp16_scale_f32_hot8192" : "f32")
            << "\""
            << ",\"output_512_projected_wall_s\":"
            << wall_median_ms * kReservedOutputTokens / 1000.0
            << ",\"required_checks_passed\":" << pass
            << ",\"reserved_output_tokens\":" << kReservedOutputTokens
            << ",\"resident_state_bytes\":" << timing.resident_state_bytes
            << ",\"resident_weight_bytes\":" << timing.resident_weight_bytes
            << ",\"sample_tokens\":" << sample_tokens
            << ",\"speedup_claims_allowed\":false"
            << ",\"state_semantics\":\"zero_initialized_performance_only\""
            << ",\"wall_ms_median\":" << wall_median_ms
            << ",\"wall_ms_samples\":";
  WriteDoubleArray(wall_samples);
  std::cout
            << ",\"wall_ms_p95\":" << Percentile(wall_samples, 0.95)
            << ",\"wall_ms_total\":" << wall_total_ms
            << ",\"wall_tokens_s\":"
            << static_cast<double>(sample_tokens) * 1000.0 / wall_total_ms
            << "}" << std::endl;
  return pass ? 0 : 2;
}

int RunImportedStateDiagnostic(
    const std::string& model, const std::string& module,
    const iq36::GgufModelIndex& index,
    const std::vector<std::uint32_t>& tokens,
    const std::filesystem::path& state_directory) {
  Require(tokens.size() >= 2, "imported-state prompt must contain two tokens");
  const auto prefix_tokens = tokens.size() - 1U;
  const auto imported_state = ReadImportedState(state_directory, prefix_tokens);

  iq36::PackedTokenLevelZeroConfig config;
  config.state_capacity_tokens = std::max<std::uint64_t>(
      1024U + 32U, static_cast<std::uint64_t>(tokens.size() + 32U));
  config.use_int8_block32_kv_gqa = UseInt8Block32KvGqa();
  config.profile_kernel_times = std::getenv("IQ36_PROFILE_KERNELS") != nullptr;

  auto cpu_state = MakeCpuState();
  for (std::size_t position = 0; position < prefix_tokens; ++position) {
    (void)CpuToken(model, index, &cpu_state, tokens[position],
                   static_cast<std::uint32_t>(position), config);
  }
  const auto cpu_prefix_snapshot = Snapshot(cpu_state);
  std::vector<StateComparisonRow> state_rows;
  state_rows.reserve(80);
  for (int layer = 0; layer < kLayerCount; ++layer) {
    if (IsFullAttentionLayer(layer)) {
      state_rows.push_back({
          layer, "full_k", CompareVectors(
              cpu_prefix_snapshot.full_k_history[layer],
              imported_state.full_k_history[layer])});
      state_rows.push_back({
          layer, "full_v", CompareVectors(
              cpu_prefix_snapshot.full_v_history[layer],
              imported_state.full_v_history[layer])});
    } else {
      state_rows.push_back({
          layer, "linear_conv", CompareVectors(
              cpu_prefix_snapshot.linear_conv[layer],
              imported_state.linear_conv[layer])});
      state_rows.push_back({
          layer, "linear_recurrent", CompareVectors(
              cpu_prefix_snapshot.linear_recurrent[layer],
              imported_state.linear_recurrent[layer])});
    }
  }

  auto cpu_reference_state = cpu_state;
  std::vector<float> cpu_logits;
  const auto position = static_cast<std::uint32_t>(prefix_tokens);
  const auto cpu_topk = CpuToken(
      model, index, &cpu_reference_state, tokens.back(), position, config,
      &cpu_logits);
  Require(!cpu_topk.empty(), "imported-state CPU top-k is empty");
  const auto openvino_logits = ReadFloatFile(
      state_directory / "openvino_logits.f32", kVocabularySize);

  iq36::PackedTokenLevelZeroBackend backend(model, module, config);
  backend.LoadState(imported_state);
  backend.Compile(iq36::BuildPackedTokenProgram(
      index, config.state_capacity_tokens));
  for (int warmup = 0; warmup < 2; ++warmup) {
    (void)backend.SubmitToken({tokens.back(), position, 8});
    backend.LoadState(imported_state);
  }
  const auto gpu_topk = backend.SubmitToken({tokens.back(), position, 8});
  Require(!gpu_topk.empty(), "imported-state GPU top-k is empty");
  const auto native_logits = backend.ReadLogits();
  const auto timing = backend.last_timing();

  const auto cpu_vs_openvino = CompareDistribution(cpu_logits, openvino_logits);
  const auto cpu_vs_native = CompareDistribution(cpu_logits, native_logits);
  const auto openvino_vs_native = CompareDistribution(
      openvino_logits, native_logits);
  const bool state_finite = std::all_of(
      state_rows.begin(), state_rows.end(),
      [](const StateComparisonRow& row) { return row.numeric.finite; });
  const bool pass = config.use_int8_block32_kv_gqa && state_finite &&
      DistributionPass(cpu_vs_openvino) &&
      DistributionPass(cpu_vs_native) &&
      DistributionPass(openvino_vs_native) &&
      timing.command_list_record_count == 1 && timing.kernel_count > 252 &&
      std::isfinite(timing.host_submit_ms) && timing.host_submit_ms > 0.0;

  std::cout << std::boolalpha << std::setprecision(12) << "{"
            << "\"correctness_applicable\":true"
            << ",\"cpu_topk\":";
  WriteCpuTopK(cpu_topk);
  std::cout << ",\"cpu_vs_imported_native\":";
  WriteDistributionComparison(cpu_vs_native);
  std::cout << ",\"cpu_vs_openvino\":";
  WriteDistributionComparison(cpu_vs_openvino);
  std::cout << ",\"device_ms\":" << timing.device_ms
            << ",\"device_name\":\"" << backend.device_name() << "\""
            << ",\"full_kv_dtype\":\"int8_block32_fp16_scale_f32_hot8192\""
            << ",\"gpu_topk\":";
  WriteGpuTopK(gpu_topk);
  std::cout << ",\"host_submit_ms\":" << timing.host_submit_ms
            << ",\"kernel_count\":" << timing.kernel_count
            << ",\"openvino_vs_imported_native\":";
  WriteDistributionComparison(openvino_vs_native);
  std::cout << ",\"prefix_tokens\":" << prefix_tokens
            << ",\"required_checks_passed\":" << pass
            << ",\"speedup_claims_allowed\":false"
            << ",\"state_comparisons\":[";
  for (std::size_t row_index = 0; row_index < state_rows.size(); ++row_index) {
    if (row_index != 0U) std::cout << ",";
    const auto& row = state_rows[row_index];
    std::cout << "{\"kind\":\"" << row.kind << "\",\"layer\":"
              << row.layer << ",\"numeric\":";
    WriteVectorComparison(row.numeric);
    std::cout << "}";
  }
  std::cout << "]"
            << ",\"state_semantics\":\"openvino_reference_import\""
            << ",\"teacher_token_id\":" << tokens.back()
            << ",\"token_position\":" << position
            << ",\"wall_ms\":" << timing.wall_ms
            << "}" << std::endl;
  return pass ? 0 : 2;
}

int RunSequentialPromptDiagnostic(
    const std::string& model, const std::string& module,
    const iq36::GgufModelIndex& index,
    const std::vector<std::uint32_t>& prompt_tokens,
    const std::vector<std::uint32_t>& reference_tokens) {
  Require(!prompt_tokens.empty(), "sequential prompt is empty");
  Require(reference_tokens.size() >= 2,
          "sequential reference must contain at least two tokens");
  Require(UseInt8Block32KvGqa(),
          "sequential prompt diagnostic requires the accepted INT8/F32-hot "
          "state representation");
  const auto state_capacity = static_cast<std::uint64_t>(
      prompt_tokens.size() + reference_tokens.size() + 32U);
  Require(state_capacity <= 262144U,
          "sequential prompt plus reference exceeds model context");

  iq36::PackedTokenLevelZeroConfig config;
  config.state_capacity_tokens = state_capacity;
  config.use_int8_block32_kv_gqa = true;
  config.profile_kernel_times = std::getenv("IQ36_PROFILE_KERNELS") != nullptr;
  const auto zero_state = MakeCpuState();
  iq36::PackedTokenLevelZeroBackend backend(model, module, config);
  backend.LoadState(Snapshot(zero_state));
  backend.Compile(iq36::BuildPackedTokenProgram(index, state_capacity));

  const bool cpu_distribution_check = SequentialCpuDistributionCheck();
  auto cpu_state = MakeCpuState();
  std::vector<DistributionSummary> distribution_steps;
  std::vector<std::int32_t> cpu_top1_ids;
  if (cpu_distribution_check) {
    distribution_steps.reserve(reference_tokens.size());
    cpu_top1_ids.reserve(reference_tokens.size());
  }

  std::vector<double> prefix_device_samples;
  std::vector<double> prefix_host_samples;
  std::vector<double> prefix_wall_samples;
  prefix_device_samples.reserve(prompt_tokens.size());
  prefix_host_samples.reserve(prompt_tokens.size());
  prefix_wall_samples.reserve(prompt_tokens.size());
  std::vector<std::int32_t> candidate_top1_ids;
  candidate_top1_ids.reserve(reference_tokens.size());

  for (std::size_t position = 0; position < prompt_tokens.size(); ++position) {
    std::vector<float> cpu_logits;
    std::vector<iq36::MatvecTopKRow> cpu_rows;
    if (cpu_distribution_check) {
      cpu_rows = CpuToken(
          model, index, &cpu_state, prompt_tokens[position],
          static_cast<std::uint32_t>(position), config,
          position + 1U == prompt_tokens.size() ? &cpu_logits : nullptr);
    }
    const auto gpu_rows = backend.SubmitToken({
        prompt_tokens[position], static_cast<std::uint32_t>(position), 8});
    Require(!gpu_rows.empty(), "sequential prompt GPU top-k is empty");
    const auto timing = backend.last_timing();
    prefix_device_samples.push_back(timing.device_ms);
    prefix_host_samples.push_back(timing.host_submit_ms);
    prefix_wall_samples.push_back(timing.wall_ms);
    if (position + 1U == prompt_tokens.size()) {
      candidate_top1_ids.push_back(gpu_rows.front().token_id);
      if (cpu_distribution_check) {
        Require(!cpu_rows.empty(), "sequential prompt CPU top-k is empty");
        cpu_top1_ids.push_back(cpu_rows.front().token_id);
        distribution_steps.push_back(
            CompareDistribution(cpu_logits, backend.ReadLogits()));
      }
    }
  }

  std::vector<double> decode_device_samples;
  std::vector<double> decode_host_samples;
  std::vector<double> decode_wall_samples;
  decode_device_samples.reserve(reference_tokens.size() - 1U);
  decode_host_samples.reserve(reference_tokens.size() - 1U);
  decode_wall_samples.reserve(reference_tokens.size() - 1U);
  for (std::size_t reference_index = 0;
       reference_index + 1U < reference_tokens.size(); ++reference_index) {
    const auto position = static_cast<std::uint32_t>(
        prompt_tokens.size() + reference_index);
    std::vector<float> cpu_logits;
    std::vector<iq36::MatvecTopKRow> cpu_rows;
    if (cpu_distribution_check) {
      cpu_rows = CpuToken(model, index, &cpu_state,
                          reference_tokens[reference_index], position, config,
                          &cpu_logits);
      Require(!cpu_rows.empty(), "sequential decode CPU top-k is empty");
    }
    const auto gpu_rows = backend.SubmitToken(
        {reference_tokens[reference_index], position, 8});
    Require(!gpu_rows.empty(), "sequential decode GPU top-k is empty");
    candidate_top1_ids.push_back(gpu_rows.front().token_id);
    if (cpu_distribution_check) {
      cpu_top1_ids.push_back(cpu_rows.front().token_id);
      distribution_steps.push_back(
          CompareDistribution(cpu_logits, backend.ReadLogits()));
    }
    const auto timing = backend.last_timing();
    decode_device_samples.push_back(timing.device_ms);
    decode_host_samples.push_back(timing.host_submit_ms);
    decode_wall_samples.push_back(timing.wall_ms);
  }

  Require(candidate_top1_ids.size() == reference_tokens.size(),
          "sequential prediction count mismatch");
  std::size_t matching_reference_ids = 0;
  std::size_t first_divergence = reference_tokens.size();
  std::vector<std::int32_t> reference_ids;
  reference_ids.reserve(reference_tokens.size());
  for (std::size_t index_value = 0; index_value < reference_tokens.size();
       ++index_value) {
    Require(reference_tokens[index_value] <=
                static_cast<std::uint32_t>(
                    std::numeric_limits<std::int32_t>::max()),
            "reference token id exceeds int32");
    reference_ids.push_back(
        static_cast<std::int32_t>(reference_tokens[index_value]));
    if (candidate_top1_ids[index_value] == reference_ids.back()) {
      ++matching_reference_ids;
    } else if (first_divergence == reference_tokens.size()) {
      first_divergence = index_value;
    }
  }
  const bool exact_reference_ids =
      matching_reference_ids == reference_tokens.size();
  const auto distribution = cpu_distribution_check
      ? SummarizeDistributionLadder(distribution_steps)
      : DistributionLadderSummary{};
  const double prefix_wall_total_ms = std::accumulate(
      prefix_wall_samples.begin(), prefix_wall_samples.end(), 0.0);
  const double prefix_device_total_ms = std::accumulate(
      prefix_device_samples.begin(), prefix_device_samples.end(), 0.0);
  const double decode_wall_total_ms = std::accumulate(
      decode_wall_samples.begin(), decode_wall_samples.end(), 0.0);
  const double decode_device_total_ms = std::accumulate(
      decode_device_samples.begin(), decode_device_samples.end(), 0.0);
  const auto timing = backend.last_timing();
  const bool timings_finite =
      std::all_of(prefix_wall_samples.begin(), prefix_wall_samples.end(),
                  [](double value) {
                    return std::isfinite(value) && value > 0.0;
                  }) &&
      std::all_of(prefix_host_samples.begin(), prefix_host_samples.end(),
                  [](double value) {
                    return std::isfinite(value) && value > 0.0;
                  }) &&
      std::all_of(decode_wall_samples.begin(), decode_wall_samples.end(),
                  [](double value) {
                    return std::isfinite(value) && value > 0.0;
                  }) &&
      std::all_of(decode_host_samples.begin(), decode_host_samples.end(),
                  [](double value) {
                    return std::isfinite(value) && value > 0.0;
                  });
  const bool pass = exact_reference_ids && timings_finite &&
      (!cpu_distribution_check || distribution.required_checks_passed) &&
      timing.command_list_record_count == 1 && timing.kernel_count > 252;

  std::cout << std::boolalpha << std::setprecision(12) << "{"
            << "\"candidate_top1_ids\":";
  WriteIntArray(candidate_top1_ids);
  std::cout << ",\"correctness_applicable\":true"
            << ",\"cpu_distribution_check\":"
            << cpu_distribution_check;
  if (cpu_distribution_check) {
    std::cout << ",\"cpu_distribution_ladder\":{"
              << "\"logit_scope\":\"full_vocab\","
              << "\"max_kld\":" << distribution.max_kld << ","
              << "\"mean_kld\":" << distribution.mean_kld << ","
              << "\"min_logits_cosine\":"
              << distribution.min_logits_cosine << ","
              << "\"position_count\":" << distribution_steps.size() << ","
              << "\"required_checks_passed\":"
              << distribution.required_checks_passed << ","
              << "\"steps\":[";
    for (std::size_t index_value = 0;
         index_value < distribution_steps.size(); ++index_value) {
      if (index_value != 0U) std::cout << ",";
      std::cout << "{\"kld\":" << distribution_steps[index_value].kld
                << ",\"logits_cosine\":"
                << distribution_steps[index_value].logits_cosine
                << ",\"top1_matches\":"
                << distribution_steps[index_value].top1_matches << "}";
    }
    std::cout << "],\"teacher_forced\":true,"
              << "\"thresholds\":{\"kld_max\":0.005,"
              << "\"top1_min\":0.99},"
              << "\"top1_rate\":" << distribution.top1_rate << "}"
              << ",\"cpu_top1_ids\":";
    WriteIntArray(cpu_top1_ids);
  }
  std::cout << ",\"decode_device_ms_median\":"
            << Median(decode_device_samples)
            << ",\"decode_device_ms_samples\":";
  WriteDoubleArray(decode_device_samples);
  std::cout << ",\"decode_device_ms_total\":" << decode_device_total_ms
            << ",\"decode_host_submit_ms_median\":"
            << Median(decode_host_samples)
            << ",\"decode_host_submit_ms_p95\":"
            << Percentile(decode_host_samples, 0.95)
            << ",\"decode_host_submit_ms_samples\":";
  WriteDoubleArray(decode_host_samples);
  std::cout << ",\"decode_sample_tokens\":"
            << decode_wall_samples.size()
            << ",\"decode_wall_ms_median\":"
            << Median(decode_wall_samples)
            << ",\"decode_wall_ms_p95\":"
            << Percentile(decode_wall_samples, 0.95)
            << ",\"decode_wall_ms_samples\":";
  WriteDoubleArray(decode_wall_samples);
  std::cout << ",\"decode_wall_ms_total\":" << decode_wall_total_ms
            << ",\"decode_wall_tokens_s\":"
            << static_cast<double>(decode_wall_samples.size()) * 1000.0 /
                   decode_wall_total_ms
            << ",\"deterministic_greedy_exact_match_proved_by_induction\":"
            << exact_reference_ids
            << ",\"device_name\":\"" << backend.device_name() << "\""
            << ",\"exact_reference_ids\":" << exact_reference_ids
            << ",\"first_divergence_index\":";
  if (first_divergence == reference_tokens.size()) {
    std::cout << "null";
  } else {
    std::cout << first_divergence;
  }
  std::cout << ",\"full_kv_dtype\":"
            << "\"int8_block32_fp16_scale_f32_hot8192\""
            << ",\"kernel_count\":" << timing.kernel_count
            << ",\"matching_reference_ids\":" << matching_reference_ids
            << ",\"prefill_product_claim_allowed\":false"
            << ",\"prompt_conditioned\":true"
            << ",\"prompt_tokens\":" << prompt_tokens.size()
            << ",\"reference_ids\":";
  WriteIntArray(reference_ids);
  std::cout << ",\"reference_prediction_count\":"
            << reference_tokens.size()
            << ",\"required_checks_passed\":" << pass
            << ",\"resident_state_bytes\":" << timing.resident_state_bytes
            << ",\"resident_weight_bytes\":" << timing.resident_weight_bytes
            << ",\"sequential_state_build_device_ms_total\":"
            << prefix_device_total_ms
            << ",\"sequential_state_build_host_submit_ms_median\":"
            << Median(prefix_host_samples)
            << ",\"sequential_state_build_host_submit_ms_p95\":"
            << Percentile(prefix_host_samples, 0.95)
            << ",\"sequential_state_build_wall_ms_median\":"
            << Median(prefix_wall_samples)
            << ",\"sequential_state_build_wall_ms_p95\":"
            << Percentile(prefix_wall_samples, 0.95)
            << ",\"sequential_state_build_wall_ms_total\":"
            << prefix_wall_total_ms
            << ",\"sequential_state_build_wall_tokens_s\":"
            << static_cast<double>(prompt_tokens.size()) * 1000.0 /
                   prefix_wall_total_ms
            << ",\"speedup_claims_allowed\":false"
            << ",\"state_capacity_tokens\":" << state_capacity
            << ",\"state_semantics\":"
            << "\"native_sequential_locked_gguf\""
            << "}" << std::endl;
  return pass ? 0 : 2;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 4 && argc != 6) {
      throw std::invalid_argument(
          "usage: iq36-packed-token-level-zero-backend-smoke "
          "MODEL MODULE TOKENS_U32 [CONTEXT_TOKENS SAMPLE_TOKENS | "
          "--import-state STATE_DIRECTORY | "
          "--sequential-prompt REFERENCE_TOKENS_U32]");
    }
    const std::string model = argv[1];
    const auto index = iq36::parse_gguf_model_index(model);
    const auto tokens = ReadTokens(argv[3]);
    iq36::set_resident_tensor_cache_enabled(true);
    iq36::set_dense_matvec_enabled(true);
    iq36::set_dense_matvec_thread_count(16);
    iq36::set_dense_matvec_min_rows(256);
    iq36::set_selected_expert_ffn_enabled(true);
    iq36::set_selected_expert_ffn_thread_count(16);
    if (argc == 6) {
      if (std::string(argv[4]) == "--import-state") {
        return RunImportedStateDiagnostic(
            model, argv[2], index, tokens, argv[5]);
      }
      if (std::string(argv[4]) == "--sequential-prompt") {
        return RunSequentialPromptDiagnostic(
            model, argv[2], index, tokens, ReadTokens(argv[5]));
      }
      return RunSyntheticContextDiagnostic(
          model, argv[2], index, tokens,
          static_cast<std::uint64_t>(std::stoull(argv[4])),
          static_cast<std::uint32_t>(std::stoul(argv[5])));
    }
    iq36::PackedTokenLevelZeroConfig config;
    config.state_capacity_tokens = 1024 + 32;
    config.use_int8_block32_kv_gqa = UseInt8Block32KvGqa();
    config.profile_kernel_times = std::getenv("IQ36_PROFILE_KERNELS") != nullptr;
    auto cpu_state = MakeCpuState();
    for (std::size_t position = 0; position + 1 < tokens.size(); ++position) {
      (void)CpuToken(model, index, &cpu_state, tokens[position],
                     static_cast<std::uint32_t>(position), config);
    }
    iq36::PackedTokenLevelZeroBackend backend(model, argv[2], config);
    backend.LoadState(Snapshot(cpu_state));
    const auto program = iq36::BuildPackedTokenProgram(index, 1024);
    backend.Compile(program);
    auto cpu_reference_state = cpu_state;
    const auto position = static_cast<std::uint32_t>(tokens.size() - 1);
    const bool distribution_check =
        std::getenv("IQ36_DISTRIBUTION_CHECK") != nullptr;
    std::vector<std::vector<float>> cpu_logits;
    cpu_logits.resize(distribution_check ? 9U : 0U);
    const auto cpu = CpuToken(model, index, &cpu_reference_state,
                              tokens.back(), position, config,
                              distribution_check ? &cpu_logits[0] : nullptr);
    Require(!cpu.empty(), "CPU seed top-k is empty");
    std::vector<std::int32_t> cpu_generated;
    auto cpu_feed = static_cast<std::uint32_t>(cpu.front().token_id);
    for (std::uint32_t step = 0; step < 8; ++step) {
      const auto rows = CpuToken(model, index, &cpu_reference_state, cpu_feed,
                                 position + 1 + step, config,
                                 distribution_check
                                     ? &cpu_logits[step + 1U]
                                     : nullptr);
      Require(!rows.empty(), "CPU decode top-k is empty");
      cpu_generated.push_back(rows.front().token_id);
      cpu_feed = static_cast<std::uint32_t>(rows.front().token_id);
    }
    for (int warmup = 0; warmup < 5; ++warmup) {
      (void)backend.SubmitToken(
          {tokens.back(), static_cast<std::uint32_t>(tokens.size() - 1), 8});
      backend.LoadState(Snapshot(cpu_state));
    }
    const auto gpu = backend.SubmitToken({tokens.back(), position, 8});
    Require(!gpu.empty(), "GPU seed top-k is empty");
    std::vector<DistributionSummary> distribution_steps;
    if (distribution_check) {
      distribution_steps.push_back(
          CompareDistribution(cpu_logits[0], backend.ReadLogits()));
    }
    std::vector<std::int32_t> gpu_generated;
    std::vector<double> device_samples;
    std::vector<double> host_samples;
    std::vector<double> wall_samples;
    auto gpu_feed = static_cast<std::uint32_t>(
        distribution_check ? cpu.front().token_id : gpu.front().token_id);
    for (std::uint32_t step = 0; step < 8; ++step) {
      const auto rows = backend.SubmitToken(
          {gpu_feed, position + 1 + step, 8});
      Require(!rows.empty(), "GPU decode top-k is empty");
      gpu_generated.push_back(rows.front().token_id);
      if (distribution_check) {
        distribution_steps.push_back(CompareDistribution(
            cpu_logits[step + 1U], backend.ReadLogits()));
        gpu_feed = static_cast<std::uint32_t>(cpu_generated[step]);
      } else {
        gpu_feed = static_cast<std::uint32_t>(rows.front().token_id);
      }
      const auto sample = backend.last_timing();
      device_samples.push_back(sample.device_ms);
      host_samples.push_back(sample.host_submit_ms);
      wall_samples.push_back(sample.wall_ms);
    }
    const auto timing = backend.last_timing();
    const bool same_top1 = !cpu.empty() && !gpu.empty() &&
        cpu[0].token_id == gpu[0].token_id;
    std::size_t matching_ids = 0;
    for (std::size_t i = 0; i < std::min(cpu.size(), gpu.size()); ++i) {
      matching_ids += cpu[i].token_id == gpu[i].token_id;
    }
    const bool exact_generated = gpu_generated == cpu_generated;
    const auto distribution = distribution_check
        ? SummarizeDistributionLadder(distribution_steps)
        : DistributionLadderSummary{};
    const bool pass = same_top1 && exact_generated &&
        (!distribution_check || distribution.required_checks_passed) &&
        timing.command_list_record_count == 1 &&
        timing.kernel_count > 252 &&
        std::all_of(host_samples.begin(), host_samples.end(),
                    [](double value) {
                      return std::isfinite(value) && value > 0.0;
                    });
    std::cout << std::boolalpha << std::setprecision(12) << "{"
              << "\"cpu_topk\":";
    WriteCpuTopK(cpu);
    std::cout << ",\"cpu_generated_ids\":";
    WriteIntArray(cpu_generated);
    if (distribution_check) {
      std::cout << ",\"distribution_ladder\":{"
                << "\"logit_scope\":\"full_vocab\","
                << "\"max_kld\":" << distribution.max_kld << ","
                << "\"mean_kld\":" << distribution.mean_kld << ","
                << "\"min_logits_cosine\":"
                << distribution.min_logits_cosine << ","
                << "\"position_count\":" << distribution_steps.size() << ","
                << "\"required_checks_passed\":"
                << distribution.required_checks_passed << ","
                << "\"steps\":[";
      for (std::size_t index = 0; index < distribution_steps.size(); ++index) {
        if (index != 0U) std::cout << ",";
        std::cout << "{\"kld\":" << distribution_steps[index].kld
                  << ",\"logits_cosine\":"
                  << distribution_steps[index].logits_cosine
                  << ",\"top1_matches\":"
                  << distribution_steps[index].top1_matches << "}";
      }
      std::cout << "],"
                << "\"teacher_forced\":true,"
                << "\"thresholds\":{\"kld_max\":0.005,\"top1_min\":0.99},"
                << "\"top1_rate\":" << distribution.top1_rate << "}";
    }
    std::cout << ",\"device_ms_median\":" << Median(device_samples)
              << ",\"device_name\":\"" << backend.device_name() << "\""
              << ",\"exact_generated_ids\":" << exact_generated
              << ",\"full_kv_dtype\":\""
              << (config.use_int8_block32_kv_gqa
                      ? "int8_block32_fp16_scale_f32_hot8192" : "f32")
              << "\""
              << ",\"gpu_generated_ids\":";
    WriteIntArray(gpu_generated);
    std::cout << ",\"gpu_topk\":";
    WriteGpuTopK(gpu);
    std::cout << ",\"host_submit_ms_max\":"
              << *std::max_element(host_samples.begin(), host_samples.end())
              << ",\"host_submit_ms_median\":" << Median(host_samples)
              << ",\"host_submit_ms_p95\":" << Percentile(host_samples, 0.95)
              << ",\"host_submit_ms_samples\":";
    WriteDoubleArray(host_samples);
    std::cout
              << ",\"kernel_profile\":";
    WriteKernelProfile(timing.kernel_profile);
    std::cout << ",\"kernel_count\":" << timing.kernel_count
              << ",\"matching_topk_ids\":" << matching_ids
              << ",\"required_checks_passed\":" << pass
              << ",\"resident_state_bytes\":"
              << timing.resident_state_bytes
              << ",\"resident_weight_bytes\":"
              << timing.resident_weight_bytes
              << ",\"same_top1\":" << same_top1
              << ",\"wall_ms_median\":" << Median(wall_samples)
              << "}" << std::endl;
    return pass ? 0 : 2;
  } catch (const std::exception& exception) {
    std::cerr << "iq36-packed-token-level-zero-backend-smoke: "
              << exception.what() << '\n';
    return 4;
  }
}
