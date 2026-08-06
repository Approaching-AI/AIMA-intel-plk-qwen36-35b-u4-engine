#include "intel_qwen36/gguf_loader.hpp"

#include <cstdint>
#include <iostream>
#include <map>
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

}  // namespace

int main(int argc, char** argv) {
  try {
    require(argc == 2, "usage: iq36-layout-inventory <model.gguf>");
    const std::string model_path = argv[1];
    const auto index = iq36::parse_gguf_model_index(model_path);
    const auto summary = iq36::validate_qwen36_load_map(index);
    std::map<std::string, std::uint64_t> bytes_by_type;
    std::map<std::string, int> tensor_count_by_type;
    for (const auto& tensor : index.tensors) {
      const auto type_name = iq36::ggml_type_name(tensor.type);
      bytes_by_type[type_name] += tensor.nbytes;
      ++tensor_count_by_type[type_name];
    }

    std::cout << "{";
    std::cout << "\"schema_version\":\"intel-qwen36-r3-layout-inventory-v0\",";
    std::cout << "\"model_path\":\"" << json_escape(model_path) << "\",";
    std::cout << "\"file_size_bytes\":" << index.file_size_bytes << ",";
    std::cout << "\"tensor_count\":" << index.tensor_count << ",";
    std::cout << "\"native_gguf_load_map_ready\":"
              << (summary.ready ? "true" : "false") << ",";
    std::cout << "\"bytes_by_type\":{";
    bool first = true;
    for (const auto& item : bytes_by_type) {
      if (!first) {
        std::cout << ",";
      }
      first = false;
      std::cout << "\"" << json_escape(item.first) << "\":" << item.second;
    }
    std::cout << "},";
    std::cout << "\"tensor_count_by_type\":{";
    first = true;
    for (const auto& item : tensor_count_by_type) {
      if (!first) {
        std::cout << ",";
      }
      first = false;
      std::cout << "\"" << json_escape(item.first) << "\":" << item.second;
    }
    std::cout << "},";
    std::cout << "\"quantized_tensors\":[";
    first = true;
    for (const auto& tensor : index.tensors) {
      const auto type_name = iq36::ggml_type_name(tensor.type);
      if (type_name != "Q4_K" && type_name != "Q6_K") {
        continue;
      }
      if (!first) {
        std::cout << ",";
      }
      first = false;
      std::cout << "{";
      std::cout << "\"absolute_offset\":" << tensor.absolute_offset << ",";
      std::cout << "\"dims\":";
      write_u64_vector(tensor.dims);
      std::cout << ",";
      std::cout << "\"layer_index\":" << tensor.layer_index << ",";
      std::cout << "\"name\":\"" << json_escape(tensor.name) << "\",";
      std::cout << "\"nbytes\":" << tensor.nbytes << ",";
      std::cout << "\"offset\":" << tensor.offset << ",";
      std::cout << "\"suffix\":\"" << json_escape(tensor.suffix) << "\",";
      std::cout << "\"type\":" << tensor.type << ",";
      std::cout << "\"type_name\":\"" << json_escape(type_name) << "\"";
      std::cout << "}";
    }
    std::cout << "]}";
    std::cout << "\n";
    return summary.ready ? 0 : 2;
  } catch (const std::exception& exc) {
    std::cerr << "iq36-layout-inventory: " << exc.what() << "\n";
    return 1;
  }
}
