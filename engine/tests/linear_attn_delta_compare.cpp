#include "intel_qwen36/gguf_loader.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr const char* kNormTensorName = "blk.0.ssm_norm.weight";
constexpr double kMismatchThreshold = 5e-5;
constexpr double kMaxAbsDiffThreshold = 5e-5;
constexpr double kRmseThreshold = 5e-6;
constexpr double kMinCosine = 0.999999;
constexpr int kLayerIndex = 0;
constexpr int kHeadDim = 128;
constexpr int kQueryHeads = 16;
constexpr int kValueHeads = 32;

struct ValueStats {
  std::uint64_t count = 0;
  double min = 0.0;
  double max = 0.0;
  double abs_sum = 0.0;
  double l2 = 0.0;
  bool finite = false;
  bool nonzero = false;
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

bool compare_passed(const iq36::VectorCompareStats& stats) {
  return stats.same_size && stats.finite && stats.mismatch_count == 0 &&
         stats.max_abs_diff <= kMaxAbsDiffThreshold &&
         stats.rmse <= kRmseThreshold && stats.cosine >= kMinCosine;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    require(argc == 3,
            "usage: iq36-linear-attn-delta-compare <model.gguf> <oracle-payload-dir>");
    const std::string model_path = argv[1];
    const std::string payload_dir = argv[2];

    const auto index = iq36::parse_gguf_model_index(model_path);
    const auto load_map = iq36::validate_qwen36_load_map(index);
    const auto* norm_tensor = iq36::find_tensor(index, kNormTensorName);
    require(norm_tensor != nullptr, "L0 ssm_norm tensor missing");
    const bool norm_shape_ok =
        norm_tensor->type == 0 &&
        norm_tensor->dims == std::vector<std::uint64_t>{kHeadDim};

    const float epsilon = metadata_float(
        index,
        "qwen35moe.attention.layer_norm_rms_epsilon",
        1e-6f);
    const auto q =
        iq36::read_f32_vector_file(join_path(payload_dir, "q_conv_predelta.bin"));
    const auto k =
        iq36::read_f32_vector_file(join_path(payload_dir, "k_conv_predelta.bin"));
    const auto v =
        iq36::read_f32_vector_file(join_path(payload_dir, "v_conv_predelta.bin"));
    const auto gate =
        iq36::read_f32_vector_file(join_path(payload_dir, "gate.bin"));
    const auto beta =
        iq36::read_f32_vector_file(join_path(payload_dir, "beta_sigmoid.bin"));
    const auto state =
        iq36::read_f32_vector_file(join_path(payload_dir, "state_predelta.bin"));
    const auto z = iq36::read_f32_vector_file(join_path(payload_dir, "z.bin"));
    const auto oracle_attn =
        iq36::read_f32_vector_file(join_path(payload_dir, "attn_output.bin"));
    const auto oracle_final =
        iq36::read_f32_vector_file(join_path(payload_dir, "final_output.bin"));
    const auto norm_weight =
        iq36::decode_tensor_row(model_path, index, kNormTensorName, 0);

    const auto native = iq36::run_qwen36_linear_attention_delta_core(
        q,
        k,
        v,
        gate,
        beta,
        state,
        z,
        norm_weight,
        epsilon);

    const auto q_stats = stats_from_values(q);
    const auto k_stats = stats_from_values(k);
    const auto v_stats = stats_from_values(v);
    const auto gate_stats = stats_from_values(gate);
    const auto beta_stats = stats_from_values(beta);
    const auto state_stats = stats_from_values(state);
    const auto z_stats = stats_from_values(z);
    const auto norm_stats = stats_from_values(norm_weight);
    const auto native_attn_stats = stats_from_values(native.attention_output);
    const auto native_state_stats = stats_from_values(native.recurrent_state);
    const auto native_final_stats = stats_from_values(native.final_output);
    const auto oracle_attn_stats = stats_from_values(oracle_attn);
    const auto oracle_final_stats = stats_from_values(oracle_final);
    const auto attn_compare =
        iq36::compare_vectors(native.attention_output, oracle_attn, kMismatchThreshold);
    const auto final_compare =
        iq36::compare_vectors(native.final_output, oracle_final, kMismatchThreshold);

    const bool counts_ok =
        q_stats.count == kHeadDim * kQueryHeads &&
        k_stats.count == kHeadDim * kQueryHeads &&
        v_stats.count == kHeadDim * kValueHeads &&
        gate_stats.count == kValueHeads &&
        beta_stats.count == kValueHeads &&
        state_stats.count == kHeadDim * kHeadDim * kValueHeads &&
        z_stats.count == kHeadDim * kValueHeads &&
        norm_stats.count == kHeadDim &&
        native_attn_stats.count == kHeadDim * kValueHeads &&
        native_state_stats.count == kHeadDim * kHeadDim * kValueHeads &&
        native_final_stats.count == kHeadDim * kValueHeads &&
        oracle_attn_stats.count == kHeadDim * kValueHeads &&
        oracle_final_stats.count == kHeadDim * kValueHeads;
    const bool finite_ok =
        q_stats.finite && k_stats.finite && v_stats.finite &&
        gate_stats.finite && beta_stats.finite && state_stats.finite &&
        z_stats.finite && norm_stats.finite && native_attn_stats.finite &&
        native_state_stats.finite && native_final_stats.finite &&
        oracle_attn_stats.finite && oracle_final_stats.finite;
    const bool nonzero_ok =
        q_stats.nonzero && k_stats.nonzero && v_stats.nonzero &&
        gate_stats.nonzero && beta_stats.nonzero && state_stats.nonzero &&
        z_stats.nonzero && norm_stats.nonzero && native_attn_stats.nonzero &&
        native_state_stats.nonzero && native_final_stats.nonzero &&
        oracle_attn_stats.nonzero && oracle_final_stats.nonzero;
    const bool passed =
        load_map.ready && norm_shape_ok && counts_ok && finite_ok && nonzero_ok &&
        compare_passed(attn_compare) && compare_passed(final_compare);

    std::cout << std::setprecision(17);
    std::cout << "{";
    std::cout << "\"comparisons\":{";
    std::cout << "\"attention_output\":";
    write_compare_stats(attn_compare);
    std::cout << ",\"final_output\":";
    write_compare_stats(final_compare);
    std::cout << "},";
    std::cout << "\"epsilon\":" << epsilon << ",";
    std::cout << "\"input_vectors\":{";
    std::cout << "\"beta_sigmoid\":";
    write_value_stats(beta_stats);
    std::cout << ",\"gate\":";
    write_value_stats(gate_stats);
    std::cout << ",\"k_conv_predelta\":";
    write_value_stats(k_stats);
    std::cout << ",\"q_conv_predelta\":";
    write_value_stats(q_stats);
    std::cout << ",\"state_predelta\":";
    write_value_stats(state_stats);
    std::cout << ",\"v_conv_predelta\":";
    write_value_stats(v_stats);
    std::cout << ",\"z\":";
    write_value_stats(z_stats);
    std::cout << "},";
    std::cout << "\"layer_index\":" << kLayerIndex << ",";
    std::cout << "\"load_map_ready\":" << (load_map.ready ? "true" : "false") << ",";
    std::cout << "\"model_path\":\"" << json_escape(model_path) << "\",";
    std::cout << "\"native_vectors\":{";
    std::cout << "\"attention_output\":";
    write_value_stats(native_attn_stats);
    std::cout << ",\"final_output\":";
    write_value_stats(native_final_stats);
    std::cout << ",\"recurrent_state\":";
    write_value_stats(native_state_stats);
    std::cout << "},";
    std::cout << "\"norm_tensor\":{";
    std::cout << "\"dims\":";
    write_u64_vector(norm_tensor->dims);
    std::cout << ",\"name\":\"" << json_escape(norm_tensor->name) << "\",";
    std::cout << "\"nbytes\":" << norm_tensor->nbytes << ",";
    std::cout << "\"shape_ok\":" << (norm_shape_ok ? "true" : "false") << ",";
    std::cout << "\"type_name\":\"" << iq36::ggml_type_name(norm_tensor->type) << "\"";
    std::cout << "},";
    std::cout << "\"norm_weight\":";
    write_value_stats(norm_stats);
    std::cout << ",";
    std::cout << "\"oracle_payload_dir\":\"" << json_escape(payload_dir) << "\",";
    std::cout << "\"oracle_vectors\":{";
    std::cout << "\"attention_output\":";
    write_value_stats(oracle_attn_stats);
    std::cout << ",\"final_output\":";
    write_value_stats(oracle_final_stats);
    std::cout << "},";
    std::cout << "\"passed\":" << (passed ? "true" : "false") << ",";
    std::cout << "\"schema_version\":\"intel-qwen36-engine-linear-attn-delta-compare-v0\",";
    std::cout << "\"thresholds\":{";
    std::cout << "\"max_abs_diff\":" << kMaxAbsDiffThreshold << ",";
    std::cout << "\"min_cosine\":" << kMinCosine << ",";
    std::cout << "\"mismatch_abs_diff\":" << kMismatchThreshold << ",";
    std::cout << "\"rmse\":" << kRmseThreshold;
    std::cout << "}";
    std::cout << "}\n";
    return passed ? 0 : 2;
  } catch (const std::exception& exc) {
    std::cerr << "iq36-linear-attn-delta-compare: " << exc.what() << "\n";
    return 1;
  }
}
