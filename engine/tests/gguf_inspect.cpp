#include "intel_qwen36/gguf_loader.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

namespace {

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

void write_int_vector(const std::vector<int>& values) {
  std::cout << "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0) {
      std::cout << ",";
    }
    std::cout << values[i];
  }
  std::cout << "]";
}

void write_payload_stats(const iq36::TensorPayloadStats& stats) {
  std::cout << "{";
  std::cout << "\"absolute_offset\":" << stats.absolute_offset << ",";
  std::cout << "\"abs_sum\":" << stats.abs_sum << ",";
  std::cout << "\"decoded_values\":" << stats.decoded_values << ",";
  std::cout << "\"finite\":" << (stats.finite ? "true" : "false") << ",";
  std::cout << "\"l2\":" << stats.l2 << ",";
  std::cout << "\"max\":" << stats.max << ",";
  std::cout << "\"min\":" << stats.min << ",";
  std::cout << "\"name\":\"" << json_escape(stats.name) << "\",";
  std::cout << "\"nbytes\":" << stats.nbytes << ",";
  std::cout << "\"nonzero\":" << (stats.nonzero ? "true" : "false") << ",";
  std::cout << "\"sum\":" << stats.sum << ",";
  std::cout << "\"type_name\":\"" << json_escape(stats.type_name) << "\"";
  std::cout << "}";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    require(argc == 2, "usage: iq36-gguf-inspect <model.gguf>");
    const std::string model_path = argv[1];
    const auto index = iq36::parse_gguf_model_index(model_path);
    const auto summary = iq36::validate_qwen36_load_map(index);
    const std::vector<iq36::TensorPayloadStats> payload_stats{
        iq36::smoke_tensor_payload(model_path, index, "output_norm.weight"),
        iq36::smoke_tensor_payload(model_path, index, "token_embd.weight"),
        iq36::smoke_tensor_payload(model_path, index, "output.weight")};
    bool payload_smoke_passed = true;
    for (const auto& stats : payload_stats) {
      payload_smoke_passed = payload_smoke_passed &&
                             stats.finite &&
                             stats.nonzero &&
                             stats.decoded_values > 0;
    }

    std::cout << "{";
    std::cout << "\"data_section_offset\":" << index.data_section_offset << ",";
    std::cout << "\"file_size_bytes\":" << index.file_size_bytes << ",";
    std::cout << "\"header\":{";
    std::cout << "\"metadata_kv_count\":" << index.metadata_kv_count << ",";
    std::cout << "\"tensor_count\":" << index.tensor_count << ",";
    std::cout << "\"version\":" << index.version << "},";
    std::cout << "\"model_path\":\"" << json_escape(model_path) << "\",";
    std::cout << "\"native_gguf_load_map\":{";
    std::cout << "\"failed_check_count\":" << summary.failed_checks.size() << ",";
    std::cout << "\"full_attention_layer_count\":" << summary.full_attention_layer_count << ",";
    std::cout << "\"full_attention_layers\":";
    write_int_vector(summary.full_attention_layers);
    std::cout << ",";
    std::cout << "\"linear_ssm_layer_count\":" << summary.linear_ssm_layer_count << ",";
    std::cout << "\"native_gguf_load_map_ready\":"
              << (summary.ready ? "true" : "false") << ",";
    std::cout << "\"tensor_count\":" << summary.tensor_count << ",";
    std::cout << "\"tensor_type_counts\":{";
    std::cout << "\"F32\":" << summary.tensor_type_counts.at("F32") << ",";
    std::cout << "\"Q4_K\":" << summary.tensor_type_counts.at("Q4_K") << ",";
    std::cout << "\"Q6_K\":" << summary.tensor_type_counts.at("Q6_K") << "}";
    std::cout << "},";
    std::cout << "\"payload_smoke\":{";
    std::cout << "\"decoded_tensor_count\":" << payload_stats.size() << ",";
    std::cout << "\"passed\":" << (payload_smoke_passed ? "true" : "false") << ",";
    std::cout << "\"tensors\":[";
    for (std::size_t i = 0; i < payload_stats.size(); ++i) {
      if (i != 0) {
        std::cout << ",";
      }
      write_payload_stats(payload_stats[i]);
    }
    std::cout << "]},";
    std::cout << "\"schema_version\":\"intel-qwen36-engine-gguf-inspect-v0\"";
    std::cout << "}\n";
    return summary.ready && payload_smoke_passed ? 0 : 2;
  } catch (const std::exception& exc) {
    std::cerr << "iq36-gguf-inspect: " << exc.what() << "\n";
    return 1;
  }
}
