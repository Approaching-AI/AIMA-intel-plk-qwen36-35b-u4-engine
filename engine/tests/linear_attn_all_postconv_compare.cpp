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

constexpr std::array<int, 30> kLinearLayers = {
    0,  1,  2,  4,  5,  6,  8,  9,  10, 12,
    13, 14, 16, 17, 18, 20, 21, 22, 24, 25,
    26, 28, 29, 30, 32, 33, 34, 36, 37, 38};
constexpr int kHiddenSize = 2048;
constexpr int kConvSize = 8192;
constexpr int kHeadDim = 128;
constexpr int kQueryHeads = 16;
constexpr int kValueHeads = 32;
constexpr double kMismatchThreshold = 5e-4;
constexpr double kMaxAbsDiffThreshold = 5e-4;
constexpr double kRmseThreshold = 5e-5;
constexpr double kMinCosine = 0.99999;

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
  const iq36::GgufTensorInfo* alpha = nullptr;
  const iq36::GgufTensorInfo* beta = nullptr;
  const iq36::GgufTensorInfo* z = nullptr;
  const iq36::GgufTensorInfo* norm = nullptr;
  const iq36::GgufTensorInfo* out = nullptr;
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
  std::vector<std::pair<std::string, ValueStats>> input_stats;
  std::vector<std::pair<std::string, ValueStats>> native_stats;
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

std::string layer_tensor_name(int layer_index, const std::string& suffix) {
  return "blk." + std::to_string(layer_index) + "." + suffix;
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

float softplus_scalar(float value) {
  return value > 20.0f ? value : std::log(1.0f + std::exp(value));
}

std::vector<float> sigmoid_vector(const std::vector<float>& input) {
  std::vector<float> output;
  output.reserve(input.size());
  for (const auto value : input) {
    output.push_back(iq36::sigmoid_scalar(value));
  }
  return output;
}

std::vector<float> add_vectors_checked(const std::vector<float>& lhs,
                                       const std::vector<float>& rhs) {
  require(lhs.size() == rhs.size(), "vector add size mismatch");
  std::vector<float> output;
  output.reserve(lhs.size());
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    output.push_back(lhs[i] + rhs[i]);
  }
  return output;
}

std::vector<float> multiply_vectors_checked(const std::vector<float>& lhs,
                                            const std::vector<float>& rhs) {
  require(lhs.size() == rhs.size(), "vector multiply size mismatch");
  std::vector<float> output;
  output.reserve(lhs.size());
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    output.push_back(lhs[i] * rhs[i]);
  }
  return output;
}

std::vector<float> softplus_vector(const std::vector<float>& input) {
  std::vector<float> output;
  output.reserve(input.size());
  for (const auto value : input) {
    output.push_back(softplus_scalar(value));
  }
  return output;
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

bool compare_passed(const iq36::VectorCompareStats& stats) {
  return stats.same_size && stats.finite && stats.mismatch_count == 0 &&
         stats.max_abs_diff <= kMaxAbsDiffThreshold &&
         stats.rmse <= kRmseThreshold && stats.cosine >= kMinCosine;
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

void write_layer_array() {
  std::cout << "[";
  for (std::size_t i = 0; i < kLinearLayers.size(); ++i) {
    if (i != 0) {
      std::cout << ",";
    }
    std::cout << kLinearLayers[i];
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

void write_named_stats(const std::vector<std::pair<std::string, ValueStats>>& values) {
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
  TensorSet tensors;
  tensors.alpha =
      iq36::find_tensor(index, layer_tensor_name(layer_index, "ssm_alpha.weight"));
  tensors.beta =
      iq36::find_tensor(index, layer_tensor_name(layer_index, "ssm_beta.weight"));
  tensors.z =
      iq36::find_tensor(index, layer_tensor_name(layer_index, "attn_gate.weight"));
  tensors.norm =
      iq36::find_tensor(index, layer_tensor_name(layer_index, "ssm_norm.weight"));
  tensors.out =
      iq36::find_tensor(index, layer_tensor_name(layer_index, "ssm_out.weight"));
  tensors.shape_ok =
      tensors.alpha != nullptr && tensors.beta != nullptr &&
      tensors.z != nullptr && tensors.norm != nullptr && tensors.out != nullptr &&
      tensors.alpha->type == 12 &&
      tensors.alpha->dims == std::vector<std::uint64_t>{kHiddenSize, kValueHeads} &&
      tensors.beta->type == 12 &&
      tensors.beta->dims == std::vector<std::uint64_t>{kHiddenSize, kValueHeads} &&
      tensors.z->type == 12 &&
      tensors.z->dims == std::vector<std::uint64_t>{kHiddenSize, kConvSize / 2} &&
      tensors.norm->type == 0 &&
      tensors.norm->dims == std::vector<std::uint64_t>{kHeadDim} &&
      tensors.out->type == 12 &&
      tensors.out->dims == std::vector<std::uint64_t>{kConvSize / 2, kHiddenSize};
  return tensors;
}

LayerResult compare_layer(const std::string& model_path,
                          const iq36::GgufModelIndex& index,
                          int layer_index,
                          const std::string& payload_dir,
                          float epsilon) {
  LayerResult result;
  result.layer_index = layer_index;
  result.tensors = load_tensor_set(index, layer_index);
  const auto prefix = layer_prefix(layer_index);

  const auto attn_norm =
      iq36::read_f32_vector_file(join_path(payload_dir, prefix + "_attn_norm.bin"));
  const auto conv_output_raw =
      iq36::read_f32_vector_file(join_path(payload_dir, prefix + "_conv_output_raw.bin"));
  const auto state_predelta =
      iq36::read_f32_vector_file(join_path(payload_dir, prefix + "_state_predelta.bin"));

  const auto oracle_conv_silu =
      iq36::read_f32_vector_file(join_path(payload_dir, prefix + "_conv_output_silu.bin"));
  const auto oracle_q_conv =
      iq36::read_f32_vector_file(join_path(payload_dir, prefix + "_q_conv.bin"));
  const auto oracle_q_predelta =
      iq36::read_f32_vector_file(join_path(payload_dir, prefix + "_q_conv_predelta.bin"));
  const auto oracle_k_conv =
      iq36::read_f32_vector_file(join_path(payload_dir, prefix + "_k_conv.bin"));
  const auto oracle_k_predelta =
      iq36::read_f32_vector_file(join_path(payload_dir, prefix + "_k_conv_predelta.bin"));
  const auto oracle_v_predelta =
      iq36::read_f32_vector_file(join_path(payload_dir, prefix + "_v_conv_predelta.bin"));
  const auto oracle_alpha =
      iq36::read_f32_vector_file(join_path(payload_dir, prefix + "_alpha.bin"));
  const auto oracle_a_softplus =
      iq36::read_f32_vector_file(join_path(payload_dir, prefix + "_a_softplus.bin"));
  const auto oracle_gate =
      iq36::read_f32_vector_file(join_path(payload_dir, prefix + "_gate.bin"));
  const auto oracle_beta =
      iq36::read_f32_vector_file(join_path(payload_dir, prefix + "_beta.bin"));
  const auto oracle_beta_sigmoid =
      iq36::read_f32_vector_file(join_path(payload_dir, prefix + "_beta_sigmoid.bin"));
  const auto oracle_z =
      iq36::read_f32_vector_file(join_path(payload_dir, prefix + "_z.bin"));
  const auto oracle_attn =
      iq36::read_f32_vector_file(join_path(payload_dir, prefix + "_attention_output.bin"));
  const auto oracle_final =
      iq36::read_f32_vector_file(join_path(payload_dir, prefix + "_final_output.bin"));
  const auto oracle_linear_out =
      iq36::read_f32_vector_file(join_path(payload_dir, prefix + "_linear_attn_out.bin"));

  const auto ssm_a = iq36::decode_tensor_row(
      model_path, index, layer_tensor_name(layer_index, "ssm_a"), 0);
  const auto ssm_dt = iq36::decode_tensor_row(
      model_path, index, layer_tensor_name(layer_index, "ssm_dt.bias"), 0);
  const auto norm_weight = iq36::decode_tensor_row(
      model_path, index, layer_tensor_name(layer_index, "ssm_norm.weight"), 0);

  const auto native_alpha = iq36::matvec_tensor(
      model_path, index, layer_tensor_name(layer_index, "ssm_alpha.weight"), attn_norm);
  const auto native_alpha_biased = add_vectors_checked(native_alpha, ssm_dt);
  const auto native_a_softplus = softplus_vector(native_alpha_biased);
  const auto native_gate = multiply_vectors_checked(native_a_softplus, ssm_a);
  const auto native_beta = iq36::matvec_tensor(
      model_path, index, layer_tensor_name(layer_index, "ssm_beta.weight"), attn_norm);
  const auto native_beta_sigmoid = sigmoid_vector(native_beta);
  const auto native_z = iq36::matvec_tensor(
      model_path, index, layer_tensor_name(layer_index, "attn_gate.weight"), attn_norm);

  const auto native = iq36::run_qwen36_linear_attention_postconv_core(
      conv_output_raw,
      native_gate,
      native_beta_sigmoid,
      state_predelta,
      native_z,
      norm_weight,
      epsilon);
  const auto native_linear_out = iq36::matvec_tensor(
      model_path,
      index,
      layer_tensor_name(layer_index, "ssm_out.weight"),
      native.final_output);

  result.comparisons = {
      {"alpha", iq36::compare_vectors(native_alpha, oracle_alpha, kMismatchThreshold)},
      {"a_softplus", iq36::compare_vectors(native_a_softplus, oracle_a_softplus, kMismatchThreshold)},
      {"gate", iq36::compare_vectors(native_gate, oracle_gate, kMismatchThreshold)},
      {"beta", iq36::compare_vectors(native_beta, oracle_beta, kMismatchThreshold)},
      {"beta_sigmoid", iq36::compare_vectors(native_beta_sigmoid, oracle_beta_sigmoid, kMismatchThreshold)},
      {"z", iq36::compare_vectors(native_z, oracle_z, kMismatchThreshold)},
      {"conv_output_silu", iq36::compare_vectors(native.conv_output_silu, oracle_conv_silu, kMismatchThreshold)},
      {"q_conv", iq36::compare_vectors(native.q_conv, oracle_q_conv, kMismatchThreshold)},
      {"q_conv_predelta", iq36::compare_vectors(native.q_conv_predelta, oracle_q_predelta, kMismatchThreshold)},
      {"k_conv", iq36::compare_vectors(native.k_conv, oracle_k_conv, kMismatchThreshold)},
      {"k_conv_predelta", iq36::compare_vectors(native.k_conv_predelta, oracle_k_predelta, kMismatchThreshold)},
      {"v_conv_predelta", iq36::compare_vectors(native.v_conv_predelta, oracle_v_predelta, kMismatchThreshold)},
      {"attention_output", iq36::compare_vectors(native.attention_output, oracle_attn, kMismatchThreshold)},
      {"final_output", iq36::compare_vectors(native.final_output, oracle_final, kMismatchThreshold)},
      {"linear_attn_out", iq36::compare_vectors(native_linear_out, oracle_linear_out, kMismatchThreshold)},
  };
  result.comparisons_ok = true;
  for (const auto& item : result.comparisons) {
    result.comparisons_ok = result.comparisons_ok && compare_passed(item.second);
  }

  result.counts_ok =
      attn_norm.size() == kHiddenSize &&
      conv_output_raw.size() == kConvSize &&
      state_predelta.size() == kHeadDim * kHeadDim * kValueHeads &&
      native_alpha.size() == kValueHeads &&
      native_beta.size() == kValueHeads &&
      native_z.size() == kConvSize / 2 &&
      native.conv_output_silu.size() == kConvSize &&
      native.q_conv.size() == kHeadDim * kQueryHeads &&
      native.k_conv.size() == kHeadDim * kQueryHeads &&
      native.v_conv_predelta.size() == kHeadDim * kValueHeads &&
      native.attention_output.size() == kHeadDim * kValueHeads &&
      native.final_output.size() == kHeadDim * kValueHeads &&
      native_linear_out.size() == kHiddenSize;

  result.input_stats = {
      {"attn_norm", stats_from_values(attn_norm)},
      {"conv_output_raw", stats_from_values(conv_output_raw)},
      {"state_predelta", stats_from_values(state_predelta)},
  };
  result.native_stats = {
      {"alpha", stats_from_values(native_alpha)},
      {"a_softplus", stats_from_values(native_a_softplus)},
      {"gate", stats_from_values(native_gate)},
      {"beta", stats_from_values(native_beta)},
      {"beta_sigmoid", stats_from_values(native_beta_sigmoid)},
      {"z", stats_from_values(native_z)},
      {"conv_output_silu", stats_from_values(native.conv_output_silu)},
      {"q_conv", stats_from_values(native.q_conv)},
      {"q_conv_predelta", stats_from_values(native.q_conv_predelta)},
      {"k_conv", stats_from_values(native.k_conv)},
      {"k_conv_predelta", stats_from_values(native.k_conv_predelta)},
      {"v_conv_predelta", stats_from_values(native.v_conv_predelta)},
      {"attention_output", stats_from_values(native.attention_output)},
      {"recurrent_state", stats_from_values(native.recurrent_state)},
      {"final_output", stats_from_values(native.final_output)},
      {"linear_attn_out", stats_from_values(native_linear_out)},
  };
  result.stats_ok =
      std::all_of(result.input_stats.begin(), result.input_stats.end(),
                  [](const auto& item) {
                    return item.second.finite && item.second.nonzero;
                  }) &&
      std::all_of(result.native_stats.begin(), result.native_stats.end(),
                  [](const auto& item) {
                    return item.second.finite && item.second.nonzero;
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
  std::cout << ",\"input_vectors\":";
  write_named_stats(result.input_stats);
  std::cout << ",\"layer_index\":" << result.layer_index;
  std::cout << ",\"native_vectors\":";
  write_named_stats(result.native_stats);
  std::cout << ",\"passed\":" << (result.passed ? "true" : "false");
  std::cout << ",\"stats_ok\":" << (result.stats_ok ? "true" : "false");
  std::cout << ",\"tensors\":{";
  std::cout << "\"alpha\":";
  write_tensor_info(result.tensors.alpha);
  std::cout << ",\"beta\":";
  write_tensor_info(result.tensors.beta);
  std::cout << ",\"norm\":";
  write_tensor_info(result.tensors.norm);
  std::cout << ",\"out\":";
  write_tensor_info(result.tensors.out);
  std::cout << ",\"shape_ok\":"
            << (result.tensors.shape_ok ? "true" : "false") << ",";
  std::cout << "\"z\":";
  write_tensor_info(result.tensors.z);
  std::cout << "}";
  std::cout << "}";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    require(argc == 3,
            "usage: iq36-linear-attn-all-postconv-compare "
            "<model.gguf> <oracle-payload-dir>");
    const std::string model_path = argv[1];
    const std::string payload_dir = argv[2];

    const auto index = iq36::parse_gguf_model_index(model_path);
    const auto load_map = iq36::validate_qwen36_load_map(index);
    const float epsilon = metadata_float(
        index,
        "qwen35moe.attention.layer_norm_rms_epsilon",
        1e-6f);

    std::vector<LayerResult> layer_results;
    layer_results.reserve(kLinearLayers.size());
    for (const auto layer_index : kLinearLayers) {
      layer_results.push_back(
          compare_layer(model_path, index, layer_index, payload_dir, epsilon));
    }
    bool layers_ok = true;
    for (const auto& layer : layer_results) {
      layers_ok = layers_ok && layer.passed;
    }
    const bool passed = load_map.ready && layers_ok;

    std::cout << std::setprecision(17);
    std::cout << "{";
    std::cout << "\"epsilon\":" << epsilon;
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
    std::cout << ",\"linear_layers\":";
    write_layer_array();
    std::cout << ",\"load_map_ready\":" << (load_map.ready ? "true" : "false");
    std::cout << ",\"model_path\":\"" << json_escape(model_path) << "\"";
    std::cout << ",\"passed\":" << (passed ? "true" : "false");
    std::cout << ",\"prompt_case_id\":\"short_math_001\"";
    std::cout << ",\"schema_version\":\"intel-qwen36-engine-linear-attn-all-postconv-compare-v0\"";
    std::cout << ",\"source_token_position\":15";
    std::cout << ",\"thresholds\":{";
    std::cout << "\"max_abs_diff\":" << kMaxAbsDiffThreshold << ",";
    std::cout << "\"min_cosine\":" << kMinCosine << ",";
    std::cout << "\"mismatch_abs_diff\":" << kMismatchThreshold << ",";
    std::cout << "\"rmse\":" << kRmseThreshold;
    std::cout << "}";
    std::cout << "}\n";
    return passed ? 0 : 2;
  } catch (const std::exception& exc) {
    std::cerr << "iq36-linear-attn-all-postconv-compare: "
              << exc.what() << "\n";
    return 1;
  }
}
