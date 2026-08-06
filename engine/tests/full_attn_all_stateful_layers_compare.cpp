#include "intel_qwen36/gguf_loader.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::array<int, 10> kLayerIndices = {3, 7, 11, 15, 19,
                                               23, 27, 31, 35, 39};
constexpr int kSourceTokenPosition = 15;
constexpr std::uint64_t kInputHistoryTokenCount = 15;
constexpr std::uint64_t kUpdatedHistoryTokenCount = 16;
constexpr std::uint64_t kHiddenSize = 2048;
constexpr std::uint64_t kHeadDim = 256;
constexpr std::uint64_t kQHeadCount = 16;
constexpr std::uint64_t kKvHeadCount = 2;
constexpr std::uint64_t kQFullSize = 8192;
constexpr std::uint64_t kQSize = kQHeadCount * kHeadDim;
constexpr std::uint64_t kKvSize = kKvHeadCount * kHeadDim;
constexpr float kAttentionScale = 0.0625f;
constexpr double kMismatchThreshold = 1.25e-2;
constexpr double kMaxAbsDiffThreshold = 1.25e-2;
constexpr double kRmseThreshold = 1e-3;
constexpr double kMinCosine = 0.99998;

struct ValueStats {
  std::uint64_t count = 0;
  double min = 0.0;
  double max = 0.0;
  double abs_sum = 0.0;
  double l2 = 0.0;
  bool finite = false;
  bool nonzero = false;
};

struct TensorSet {
  const iq36::GgufTensorInfo* q = nullptr;
  const iq36::GgufTensorInfo* k = nullptr;
  const iq36::GgufTensorInfo* v = nullptr;
  const iq36::GgufTensorInfo* output = nullptr;
  bool shape_ok = false;
};

struct LayerResult {
  int layer_index = 0;
  bool counts_ok = false;
  bool stats_ok = false;
  bool comparisons_ok = false;
  bool passed = false;
  TensorSet tensors;
  std::vector<std::pair<std::string, iq36::VectorCompareStats>> comparisons;
  std::vector<std::pair<std::string, ValueStats>> vectors;
};

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

std::string two_digit(int value) {
  if (value >= 0 && value < 10) {
    return "0" + std::to_string(value);
  }
  return std::to_string(value);
}

std::string layer_prefix(int layer_index) {
  return "l" + two_digit(layer_index);
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

ValueStats stats_from_values(const std::vector<float>& values) {
  ValueStats stats;
  stats.count = values.size();
  stats.finite = !values.empty();
  stats.min = std::numeric_limits<double>::infinity();
  stats.max = -std::numeric_limits<double>::infinity();
  for (const auto value : values) {
    if (!std::isfinite(value)) {
      stats.finite = false;
      continue;
    }
    const double as_double = value;
    stats.min = std::min(stats.min, as_double);
    stats.max = std::max(stats.max, as_double);
    stats.abs_sum += std::abs(as_double);
    stats.l2 += as_double * as_double;
  }
  if (values.empty()) {
    stats.min = 0.0;
    stats.max = 0.0;
  }
  stats.nonzero = stats.abs_sum > 0.0;
  return stats;
}

bool vector_compare_passed(const iq36::VectorCompareStats& stats) {
  return stats.same_size && stats.finite && stats.mismatch_count == 0 &&
         stats.max_abs_diff <= kMaxAbsDiffThreshold &&
         stats.rmse <= kRmseThreshold && stats.cosine >= kMinCosine;
}

std::vector<float> flatten_history(
    const std::vector<std::vector<float>>& history) {
  std::vector<float> flat;
  for (const auto& item : history) {
    flat.insert(flat.end(), item.begin(), item.end());
  }
  return flat;
}

void write_u64_vector(const std::vector<std::uint64_t>& values) {
  std::cout << "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0) {
      std::cout << ",";
    }
    std::cout << values[i];
  }
  std::cout << "]";
}

void write_i64_vector(const std::vector<std::int64_t>& values) {
  std::cout << "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0) {
      std::cout << ",";
    }
    std::cout << values[i];
  }
  std::cout << "]";
}

void write_layer_index_array() {
  std::cout << "[";
  for (std::size_t i = 0; i < kLayerIndices.size(); ++i) {
    if (i != 0) {
      std::cout << ",";
    }
    std::cout << kLayerIndices[i];
  }
  std::cout << "]";
}

void write_value_stats(const ValueStats& stats) {
  std::cout << "{";
  std::cout << "\"abs_sum\":" << stats.abs_sum << ",";
  std::cout << "\"count\":" << stats.count << ",";
  std::cout << "\"finite\":" << (stats.finite ? "true" : "false") << ",";
  std::cout << "\"l2\":" << stats.l2 << ",";
  std::cout << "\"max\":" << stats.max << ",";
  std::cout << "\"min\":" << stats.min << ",";
  std::cout << "\"nonzero\":" << (stats.nonzero ? "true" : "false");
  std::cout << "}";
}

void write_compare_stats(const iq36::VectorCompareStats& stats) {
  std::cout << "{";
  std::cout << "\"compared_value_count\":" << stats.compared_value_count << ",";
  std::cout << "\"cosine\":" << stats.cosine << ",";
  std::cout << "\"finite\":" << (stats.finite ? "true" : "false") << ",";
  std::cout << "\"finite_pair_count\":" << stats.finite_pair_count << ",";
  std::cout << "\"lhs_l2\":" << stats.lhs_l2 << ",";
  std::cout << "\"lhs_value_count\":" << stats.lhs_value_count << ",";
  std::cout << "\"max_abs_diff\":" << stats.max_abs_diff << ",";
  std::cout << "\"mean_abs_diff\":" << stats.mean_abs_diff << ",";
  std::cout << "\"mismatch_count\":" << stats.mismatch_count << ",";
  std::cout << "\"rhs_l2\":" << stats.rhs_l2 << ",";
  std::cout << "\"rhs_value_count\":" << stats.rhs_value_count << ",";
  std::cout << "\"rmse\":" << stats.rmse << ",";
  std::cout << "\"same_size\":" << (stats.same_size ? "true" : "false");
  std::cout << "}";
}

void write_named_stats(
    const std::vector<std::pair<std::string, ValueStats>>& values) {
  std::cout << "{";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0) {
      std::cout << ",";
    }
    std::cout << "\"" << json_escape(values[i].first) << "\":";
    write_value_stats(values[i].second);
  }
  std::cout << "}";
}

void write_named_comparisons(
    const std::vector<std::pair<std::string, iq36::VectorCompareStats>>& values) {
  std::cout << "{";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0) {
      std::cout << ",";
    }
    std::cout << "\"" << json_escape(values[i].first) << "\":";
    write_compare_stats(values[i].second);
  }
  std::cout << "}";
}

void write_tensor_info(const iq36::GgufTensorInfo* tensor) {
  if (tensor == nullptr) {
    std::cout << "null";
    return;
  }
  std::cout << "{";
  std::cout << "\"dims\":";
  write_u64_vector(tensor->dims);
  std::cout << ",\"name\":\"" << json_escape(tensor->name) << "\"";
  std::cout << ",\"type_name\":\"" << iq36::ggml_type_name(tensor->type)
            << "\"";
  std::cout << "}";
}

TensorSet load_tensor_set(const iq36::GgufModelIndex& index,
                          int layer_index) {
  const auto prefix = std::string("blk.") + std::to_string(layer_index) + ".";
  TensorSet tensors;
  tensors.q = iq36::find_tensor(index, prefix + "attn_q.weight");
  tensors.k = iq36::find_tensor(index, prefix + "attn_k.weight");
  tensors.v = iq36::find_tensor(index, prefix + "attn_v.weight");
  tensors.output = iq36::find_tensor(index, prefix + "attn_output.weight");
  tensors.shape_ok =
      tensors.q != nullptr && tensors.k != nullptr && tensors.v != nullptr &&
      tensors.output != nullptr && tensors.q->type == 12 &&
      tensors.q->dims == std::vector<std::uint64_t>{kHiddenSize, kQFullSize} &&
      tensors.k->type == 12 &&
      tensors.k->dims == std::vector<std::uint64_t>{kHiddenSize, kKvSize} &&
      tensors.v->dims == std::vector<std::uint64_t>{kHiddenSize, kKvSize} &&
      (tensors.v->type == 12 || tensors.v->type == 14) &&
      tensors.output->type == 12 &&
      tensors.output->dims == std::vector<std::uint64_t>{kQSize, kHiddenSize};
  return tensors;
}

LayerResult compare_layer(const std::string& model_path,
                          const iq36::GgufModelIndex& index,
                          int layer_index,
                          const std::string& payload_dir,
                          std::uint64_t head_dim,
                          std::uint64_t q_head_count,
                          std::uint64_t kv_head_count,
                          std::uint64_t rope_dimension_count,
                          const std::vector<std::int64_t>& rope_sections,
                          std::uint64_t rope_context_length,
                          float rope_freq_base,
                          float rms_norm_epsilon) {
  constexpr float kRopeFreqScale = 1.0f;
  constexpr float kRopeExtFactor = 0.0f;
  constexpr float kRopeAttnFactor = 1.0f;
  constexpr float kRopeBetaFast = 32.0f;
  constexpr float kRopeBetaSlow = 1.0f;

  LayerResult result;
  result.layer_index = layer_index;
  result.tensors = load_tensor_set(index, layer_index);

  const auto prefix = layer_prefix(layer_index);
  std::vector<std::vector<float>> k_history;
  std::vector<std::vector<float>> v_history;
  k_history.reserve(kInputHistoryTokenCount);
  v_history.reserve(kInputHistoryTokenCount);
  for (std::uint64_t token = 0; token < kInputHistoryTokenCount; ++token) {
    const auto token_prefix =
        std::string("tok") + two_digit(static_cast<int>(token));
    k_history.push_back(iq36::read_f32_vector_file(join_path(
        payload_dir, token_prefix + "_" + prefix + "_k_rope.bin")));
    v_history.push_back(iq36::read_f32_vector_file(join_path(
        payload_dir, token_prefix + "_" + prefix + "_v.bin")));
  }

  const auto residual_input = iq36::read_f32_vector_file(
      join_path(payload_dir, prefix + "_layer_input.bin"));
  const auto oracle_attention_norm = iq36::read_f32_vector_file(
      join_path(payload_dir, prefix + "_attn_norm.bin"));
  const auto oracle_q_full = iq36::read_f32_vector_file(
      join_path(payload_dir, prefix + "_q_full.bin"));
  const auto oracle_q_rope = iq36::read_f32_vector_file(
      join_path(payload_dir, prefix + "_q_rope.bin"));
  const auto oracle_k_rope = iq36::read_f32_vector_file(
      join_path(payload_dir, prefix + "_k_rope.bin"));
  const auto oracle_v = iq36::read_f32_vector_file(
      join_path(payload_dir, prefix + "_v.bin"));
  const auto oracle_attn_pregate = iq36::read_f32_vector_file(
      join_path(payload_dir, prefix + "_attn_pregate.bin"));
  const auto oracle_attn_gated = iq36::read_f32_vector_file(
      join_path(payload_dir, prefix + "_attn_gated.bin"));
  const auto oracle_attn_output = iq36::read_f32_vector_file(
      join_path(payload_dir, prefix + "_attn_output.bin"));

  const auto native = iq36::run_qwen36_stateful_full_attention_layer(
      model_path,
      index,
      layer_index,
      residual_input,
      k_history,
      v_history,
      kSourceTokenPosition,
      head_dim,
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

  require(native.k_history.size() == kUpdatedHistoryTokenCount,
          "updated K history token count mismatch");
  require(native.v_history.size() == kUpdatedHistoryTokenCount,
          "updated V history token count mismatch");

  result.comparisons = {
      {"attention_norm", iq36::compare_vectors(
                             native.qkv.attention_norm,
                             oracle_attention_norm,
                             kMismatchThreshold)},
      {"q_full", iq36::compare_vectors(
                     native.qkv.q_full, oracle_q_full, kMismatchThreshold)},
      {"q_rope", iq36::compare_vectors(
                     native.rope.q_rope, oracle_q_rope, kMismatchThreshold)},
      {"k_rope_appended", iq36::compare_vectors(
                              native.k_history.back(), oracle_k_rope,
                              kMismatchThreshold)},
      {"v_appended", iq36::compare_vectors(
                         native.v_history.back(), oracle_v,
                         kMismatchThreshold)},
      {"attn_pregate", iq36::compare_vectors(
                            native.core.attn_pregate,
                            oracle_attn_pregate,
                            kMismatchThreshold)},
      {"attn_gated", iq36::compare_vectors(
                         native.gate.attn_gated,
                         oracle_attn_gated,
                         kMismatchThreshold)},
      {"attn_output", iq36::compare_vectors(
                          native.attention_output,
                          oracle_attn_output,
                          kMismatchThreshold)},
  };

  result.comparisons_ok = true;
  for (const auto& item : result.comparisons) {
    result.comparisons_ok =
        result.comparisons_ok && vector_compare_passed(item.second);
  }

  result.vectors = {
      {"residual_input", stats_from_values(residual_input)},
      {"input_k_history", stats_from_values(flatten_history(k_history))},
      {"input_v_history", stats_from_values(flatten_history(v_history))},
      {"attention_norm", stats_from_values(native.qkv.attention_norm)},
      {"q_full", stats_from_values(native.qkv.q_full)},
      {"q_rope", stats_from_values(native.rope.q_rope)},
      {"k_rope_appended", stats_from_values(native.k_history.back())},
      {"v_appended", stats_from_values(native.v_history.back())},
      {"attn_pregate", stats_from_values(native.core.attn_pregate)},
      {"attn_gated", stats_from_values(native.gate.attn_gated)},
      {"attn_output", stats_from_values(native.attention_output)},
  };
  result.stats_ok = true;
  for (const auto& item : result.vectors) {
    result.stats_ok =
        result.stats_ok && item.second.finite && item.second.nonzero;
  }

  result.counts_ok =
      residual_input.size() == kHiddenSize &&
      oracle_attention_norm.size() == kHiddenSize &&
      oracle_q_full.size() == kQFullSize &&
      oracle_q_rope.size() == kQSize &&
      oracle_k_rope.size() == kKvSize &&
      oracle_v.size() == kKvSize &&
      oracle_attn_pregate.size() == kQSize &&
      oracle_attn_gated.size() == kQSize &&
      oracle_attn_output.size() == kHiddenSize &&
      native.qkv.attention_norm.size() == kHiddenSize &&
      native.qkv.q_full.size() == kQFullSize &&
      native.rope.q_rope.size() == kQSize &&
      native.k_history.back().size() == kKvSize &&
      native.v_history.back().size() == kKvSize &&
      native.core.attn_pregate.size() == kQSize &&
      native.gate.attn_gated.size() == kQSize &&
      native.attention_output.size() == kHiddenSize &&
      std::all_of(k_history.begin(), k_history.end(), [](const auto& item) {
        return item.size() == kKvSize;
      }) &&
      std::all_of(v_history.begin(), v_history.end(), [](const auto& item) {
        return item.size() == kKvSize;
      });

  result.passed = result.tensors.shape_ok && result.counts_ok &&
                  result.stats_ok && result.comparisons_ok;
  return result;
}

void write_layer_result(const LayerResult& result) {
  std::cout << "{";
  std::cout << "\"comparisons\":";
  write_named_comparisons(result.comparisons);
  std::cout << ",\"comparisons_ok\":"
            << (result.comparisons_ok ? "true" : "false");
  std::cout << ",\"counts_ok\":" << (result.counts_ok ? "true" : "false");
  std::cout << ",\"kv_update\":{";
  std::cout << "\"appended_token_position\":" << kSourceTokenPosition << ",";
  std::cout << "\"input_history_token_count\":" << kInputHistoryTokenCount << ",";
  std::cout << "\"k_history_token_count\":" << kUpdatedHistoryTokenCount << ",";
  std::cout << "\"v_history_token_count\":" << kUpdatedHistoryTokenCount;
  std::cout << "}";
  std::cout << ",\"layer_index\":" << result.layer_index;
  std::cout << ",\"native_vectors\":";
  write_named_stats(result.vectors);
  std::cout << ",\"passed\":" << (result.passed ? "true" : "false");
  std::cout << ",\"stats_ok\":" << (result.stats_ok ? "true" : "false");
  std::cout << ",\"tensors\":{";
  std::cout << "\"k\":";
  write_tensor_info(result.tensors.k);
  std::cout << ",\"output\":";
  write_tensor_info(result.tensors.output);
  std::cout << ",\"q\":";
  write_tensor_info(result.tensors.q);
  std::cout << ",\"shape_ok\":"
            << (result.tensors.shape_ok ? "true" : "false") << ",";
  std::cout << "\"v\":";
  write_tensor_info(result.tensors.v);
  std::cout << "}";
  std::cout << "}";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    require(argc == 3,
            "usage: iq36-full-attn-all-stateful-layers-compare "
            "<model.gguf> <payload-dir>");
    const std::string model_path = argv[1];
    const std::string payload_dir = argv[2];

    const auto index = iq36::parse_gguf_model_index(model_path);
    const auto load_map = iq36::validate_qwen36_load_map(index);
    const auto rms_norm_epsilon = metadata_float(
        index, "qwen35moe.attention.layer_norm_rms_epsilon", 1e-6f);
    const auto head_dim = metadata_uint(
        index, "qwen35moe.attention.key_length", kHeadDim);
    const auto value_length = metadata_uint(
        index, "qwen35moe.attention.value_length", kHeadDim);
    const auto q_head_count = metadata_uint(
        index, "qwen35moe.attention.head_count", kQHeadCount);
    const auto kv_head_count = metadata_uint(
        index, "qwen35moe.attention.head_count_kv", kKvHeadCount);
    const auto full_attention_interval = metadata_uint(
        index, "qwen35moe.full_attention_interval", 4);
    const auto rope_dimension_count = metadata_uint(
        index, "qwen35moe.rope.dimension_count", 64);
    const auto rope_context_length = metadata_uint(
        index, "qwen35moe.context_length", 262144);
    const auto rope_sections = metadata_int_array(
        index, "qwen35moe.rope.dimension_sections", {11, 11, 10, 0});
    const float rope_freq_base = metadata_float(
        index, "qwen35moe.rope.freq_base", 10000000.0f);

    const bool metadata_ok =
        full_attention_interval == 4 &&
        head_dim == kHeadDim &&
        value_length == kHeadDim &&
        q_head_count == kQHeadCount &&
        kv_head_count == kKvHeadCount &&
        rope_dimension_count == 64 &&
        rope_context_length == 262144 &&
        rope_sections == std::vector<std::int64_t>({11, 11, 10, 0}) &&
        rope_freq_base == 10000000.0f;

    std::vector<LayerResult> layer_results;
    layer_results.reserve(kLayerIndices.size());
    for (const auto layer_index : kLayerIndices) {
      layer_results.push_back(compare_layer(
          model_path,
          index,
          layer_index,
          payload_dir,
          head_dim,
          q_head_count,
          kv_head_count,
          rope_dimension_count,
          rope_sections,
          rope_context_length,
          rope_freq_base,
          rms_norm_epsilon));
    }

    bool layers_ok = true;
    for (const auto& layer : layer_results) {
      layers_ok = layers_ok && layer.passed;
    }
    const bool passed = load_map.ready && metadata_ok && layers_ok;

    std::cout << std::setprecision(17);
    std::cout << "{";
    std::cout << "\"attention_parameters\":{";
    std::cout << "\"attention_scale\":" << kAttentionScale << ",";
    std::cout << "\"gqa_group\":" << (kQHeadCount / kKvHeadCount) << ",";
    std::cout << "\"head_dim\":" << head_dim << ",";
    std::cout << "\"input_history_token_count\":" << kInputHistoryTokenCount << ",";
    std::cout << "\"kv_head_count\":" << kv_head_count << ",";
    std::cout << "\"q_head_count\":" << q_head_count << ",";
    std::cout << "\"updated_history_token_count\":" << kUpdatedHistoryTokenCount;
    std::cout << "}";
    std::cout << ",\"full_attention_layers\":";
    write_layer_index_array();
    std::cout << ",\"layer_count\":" << layer_results.size();
    std::cout << ",\"layers\":{";
    for (std::size_t i = 0; i < layer_results.size(); ++i) {
      if (i != 0) {
        std::cout << ",";
      }
      std::cout << "\"" << layer_results[i].layer_index << "\":";
      write_layer_result(layer_results[i]);
    }
    std::cout << "}";
    std::cout << ",\"layers_ok\":" << (layers_ok ? "true" : "false");
    std::cout << ",\"load_map_ready\":" << (load_map.ready ? "true" : "false");
    std::cout << ",\"metadata\":{";
    std::cout << "\"full_attention_interval\":" << full_attention_interval << ",";
    std::cout << "\"head_count\":" << q_head_count << ",";
    std::cout << "\"head_count_kv\":" << kv_head_count << ",";
    std::cout << "\"key_length\":" << head_dim << ",";
    std::cout << "\"ok\":" << (metadata_ok ? "true" : "false") << ",";
    std::cout << "\"value_length\":" << value_length;
    std::cout << "}";
    std::cout << ",\"model_path\":\"" << json_escape(model_path) << "\"";
    std::cout << ",\"passed\":" << (passed ? "true" : "false");
    std::cout << ",\"prompt_case_id\":\"short_math_001\"";
    std::cout << ",\"rope_parameters\":{";
    std::cout << "\"context_length\":" << rope_context_length << ",";
    std::cout << "\"position_ids\":[15,15,15,0],";
    std::cout << "\"rope_attn_factor\":1,";
    std::cout << "\"rope_beta_fast\":32,";
    std::cout << "\"rope_beta_slow\":1,";
    std::cout << "\"rope_dimension_count\":" << rope_dimension_count << ",";
    std::cout << "\"rope_dimension_sections\":";
    write_i64_vector(rope_sections);
    std::cout << ",\"rope_ext_factor\":0,";
    std::cout << "\"rope_freq_base\":" << rope_freq_base << ",";
    std::cout << "\"rope_freq_scale\":1";
    std::cout << "}";
    std::cout << ",\"schema_version\":\"intel-qwen36-engine-full-attn-all-stateful-layers-compare-v0\"";
    std::cout << ",\"source_token_position\":" << kSourceTokenPosition;
    std::cout << ",\"thresholds\":{";
    std::cout << "\"max_abs_diff\":" << kMaxAbsDiffThreshold << ",";
    std::cout << "\"min_cosine\":" << kMinCosine << ",";
    std::cout << "\"mismatch_abs_diff\":" << kMismatchThreshold << ",";
    std::cout << "\"rmse\":" << kRmseThreshold;
    std::cout << "}";
    std::cout << "}\n";
    return passed ? 0 : 2;
  } catch (const std::exception& exc) {
    std::cerr << "iq36-full-attn-all-stateful-layers-compare failed: "
              << exc.what() << "\n";
    return 1;
  }
}
