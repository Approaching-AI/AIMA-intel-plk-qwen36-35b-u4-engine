#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr std::size_t kQ6BlockBytes = 210;
constexpr std::size_t kQ6Codes = 256;
constexpr std::size_t kBaseCodeBytes = 128;
constexpr std::size_t kScaleBytes = 18;
constexpr std::size_t kBlockHeaderBytes = 2;
constexpr std::size_t kBlockDirectoryBytes = 4;

struct Args {
  std::string model;
  std::string manifest;
};

struct ManifestRow {
  std::string name;
  std::uint64_t offset = 0;
  std::uint64_t bytes = 0;
  std::string suffix;
  std::size_t expert_count = 1;
  std::size_t selected_count = 1;
};

struct ExpertStats {
  std::uint64_t encoded_bytes = 0;
  std::uint64_t exception_count = 0;
};

struct TensorStats {
  ManifestRow manifest;
  std::uint64_t block_count = 0;
  std::uint64_t encoded_bytes = 0;
  std::uint64_t exception_count = 0;
  std::uint64_t correction_abs_sum = 0;
  std::uint64_t reconstruction_mismatches = 0;
  std::uint64_t active_block_count = 0;
  std::uint64_t active_source_bytes = 0;
  std::uint64_t active_encoded_bytes = 0;
  std::uint64_t active_exception_count = 0;
  std::array<std::uint64_t, kQ6Codes + 1> exception_histogram{};
  std::array<std::uint64_t, 49> window_start_histogram{};
  std::vector<std::size_t> active_expert_indices;
};

std::string JsonEscape(const std::string& value) {
  std::string output;
  output.reserve(value.size() + 8);
  for (const unsigned char character : value) {
    switch (character) {
      case '\\': output += "\\\\"; break;
      case '"': output += "\\\""; break;
      case '\n': output += "\\n"; break;
      case '\r': output += "\\r"; break;
      case '\t': output += "\\t"; break;
      default:
        if (character < 0x20) {
          char buffer[8];
          std::snprintf(buffer, sizeof(buffer), "\\u%04x", character);
          output += buffer;
        } else {
          output.push_back(static_cast<char>(character));
        }
    }
  }
  return output;
}

Args ParseArgs(int argc, char** argv) {
  Args args;
  for (int index = 1; index < argc; ++index) {
    const std::string option = argv[index];
    const auto Value = [&](const char* name) -> std::string {
      if (index + 1 >= argc) {
        throw std::runtime_error(std::string("missing value for ") + name);
      }
      return argv[++index];
    };
    if (option == "--model") {
      args.model = Value(option.c_str());
    } else if (option == "--manifest") {
      args.manifest = Value(option.c_str());
    } else if (option == "-h" || option == "--help") {
      std::cout << "usage: " << argv[0]
                << " --model MODEL --manifest TENSOR_TSV\n";
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument: " + option);
    }
  }
  if (args.model.empty() || args.manifest.empty()) {
    throw std::runtime_error("--model and --manifest are required");
  }
  return args;
}

std::vector<std::string> SplitTabs(const std::string& line) {
  std::vector<std::string> fields;
  std::size_t start = 0;
  while (true) {
    const std::size_t end = line.find('\t', start);
    fields.push_back(line.substr(start, end - start));
    if (end == std::string::npos) break;
    start = end + 1;
  }
  return fields;
}

std::vector<ManifestRow> ReadManifest(const std::string& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("failed to open manifest: " + path);
  }
  std::vector<ManifestRow> rows;
  std::string line;
  std::size_t line_number = 0;
  while (std::getline(input, line)) {
    ++line_number;
    if (line.empty()) continue;
    const auto fields = SplitTabs(line);
    if (fields.size() != 6) {
      throw std::runtime_error("manifest line does not have six fields: " +
                               std::to_string(line_number));
    }
    ManifestRow row;
    row.name = fields[0];
    row.offset = std::stoull(fields[1]);
    row.bytes = std::stoull(fields[2]);
    row.suffix = fields[3];
    row.expert_count = std::stoull(fields[4]);
    row.selected_count = std::stoull(fields[5]);
    if (row.bytes == 0 || row.bytes % kQ6BlockBytes != 0) {
      throw std::runtime_error("tensor is not Q6 block aligned: " + row.name);
    }
    if (row.expert_count == 0 || row.selected_count == 0 ||
        row.selected_count > row.expert_count ||
        (row.bytes / kQ6BlockBytes) % row.expert_count != 0) {
      throw std::runtime_error("invalid expert shape: " + row.name);
    }
    rows.push_back(std::move(row));
  }
  if (rows.empty()) {
    throw std::runtime_error("manifest is empty");
  }
  return rows;
}

int Q6Value(const std::uint8_t* block, std::size_t index) {
  const std::size_t half = index / 128;
  const std::size_t within = index % 128;
  const std::size_t quadrant = within / 32;
  const std::size_t lane = within % 32;
  const std::uint8_t high = block[128 + half * 32 + lane];
  int low = 0;
  int high_bits = 0;
  if (quadrant == 0) {
    low = block[half * 64 + lane] & 15;
    high_bits = (high >> 0) & 3;
  } else if (quadrant == 1) {
    low = block[half * 64 + 32 + lane] & 15;
    high_bits = (high >> 2) & 3;
  } else if (quadrant == 2) {
    low = block[half * 64 + lane] >> 4;
    high_bits = (high >> 4) & 3;
  } else {
    low = block[half * 64 + 32 + lane] >> 4;
    high_bits = (high >> 6) & 3;
  }
  return (low | (high_bits << 4)) - 32;
}

struct EncodedBlockStats {
  std::size_t encoded_bytes = 0;
  std::size_t exceptions = 0;
  std::size_t correction_abs_sum = 0;
  std::size_t mismatches = 0;
  int window_start = 0;
};

EncodedBlockStats EncodeAndVerify(const std::uint8_t* source) {
  std::array<int, kQ6Codes> values{};
  std::array<int, 64> frequencies{};
  for (std::size_t index = 0; index < values.size(); ++index) {
    values[index] = Q6Value(source, index);
    ++frequencies[static_cast<std::size_t>(values[index] + 32)];
  }

  int best_start = -32;
  int inside = 0;
  for (int bin = 0; bin < 16; ++bin) inside += frequencies[bin];
  int best_inside = inside;
  for (int first_bin = 1; first_bin <= 48; ++first_bin) {
    inside -= frequencies[first_bin - 1];
    inside += frequencies[first_bin + 15];
    if (inside > best_inside) {
      best_start = first_bin - 32;
      best_inside = inside;
    }
  }

  std::array<std::uint8_t, kBaseCodeBytes> packed{};
  std::array<std::uint8_t, kQ6Codes> exception_positions{};
  std::array<std::int8_t, kQ6Codes> exception_values{};
  std::size_t exception_count = 0;
  std::size_t correction_abs_sum = 0;
  for (std::size_t index = 0; index < values.size(); ++index) {
    const int base = std::clamp(values[index], best_start, best_start + 15);
    const std::uint8_t code = static_cast<std::uint8_t>(base - best_start);
    if ((index & 1) == 0) {
      packed[index / 2] = code;
    } else {
      packed[index / 2] |= static_cast<std::uint8_t>(code << 4);
    }
    if (values[index] != base) {
      exception_positions[exception_count] = static_cast<std::uint8_t>(index);
      exception_values[exception_count] = static_cast<std::int8_t>(values[index]);
      ++exception_count;
      correction_abs_sum += static_cast<std::size_t>(
          std::abs(values[index] - base));
    }
  }

  std::array<int, kQ6Codes> reconstructed{};
  for (std::size_t index = 0; index < reconstructed.size(); ++index) {
    const std::uint8_t byte = packed[index / 2];
    const int code = (index & 1) == 0 ? byte & 15 : byte >> 4;
    reconstructed[index] = best_start + code;
  }
  for (std::size_t index = 0; index < exception_count; ++index) {
    reconstructed[exception_positions[index]] = exception_values[index];
  }
  std::size_t mismatches = 0;
  for (std::size_t index = 0; index < values.size(); ++index) {
    mismatches += values[index] != reconstructed[index];
  }

  const std::size_t payload_bytes =
      kBaseCodeBytes + kScaleBytes + kBlockHeaderBytes +
      exception_count * 2;
  const std::size_t aligned_payload_bytes = (payload_bytes + 3) & ~std::size_t(3);
  return {
      kBlockDirectoryBytes + aligned_payload_bytes,
      exception_count,
      correction_abs_sum,
      mismatches,
      best_start,
  };
}

TensorStats ProcessTensor(std::ifstream& model, const ManifestRow& manifest) {
  TensorStats stats;
  stats.manifest = manifest;
  stats.block_count = manifest.bytes / kQ6BlockBytes;
  const std::uint64_t blocks_per_expert = stats.block_count / manifest.expert_count;
  std::vector<ExpertStats> experts(manifest.expert_count);
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(manifest.bytes));
  model.clear();
  model.seekg(static_cast<std::streamoff>(manifest.offset));
  model.read(reinterpret_cast<char*>(bytes.data()),
             static_cast<std::streamsize>(bytes.size()));
  if (!model || static_cast<std::size_t>(model.gcount()) != bytes.size()) {
    throw std::runtime_error("short model payload read: " + manifest.name);
  }

  for (std::uint64_t block_index = 0; block_index < stats.block_count;
       ++block_index) {
    const auto block = EncodeAndVerify(
        bytes.data() + block_index * kQ6BlockBytes);
    stats.encoded_bytes += block.encoded_bytes;
    stats.exception_count += block.exceptions;
    stats.correction_abs_sum += block.correction_abs_sum;
    stats.reconstruction_mismatches += block.mismatches;
    ++stats.exception_histogram[block.exceptions];
    ++stats.window_start_histogram[static_cast<std::size_t>(
        block.window_start + 32)];
    const std::size_t expert = static_cast<std::size_t>(
        block_index / blocks_per_expert);
    experts[expert].encoded_bytes += block.encoded_bytes;
    experts[expert].exception_count += block.exceptions;
  }

  std::vector<std::size_t> expert_order(manifest.expert_count);
  for (std::size_t index = 0; index < expert_order.size(); ++index) {
    expert_order[index] = index;
  }
  std::sort(
      expert_order.begin(), expert_order.end(),
      [&](std::size_t left, std::size_t right) {
        if (experts[left].encoded_bytes != experts[right].encoded_bytes) {
          return experts[left].encoded_bytes > experts[right].encoded_bytes;
        }
        return experts[left].exception_count > experts[right].exception_count;
      });
  for (std::size_t index = 0; index < manifest.selected_count; ++index) {
    const std::size_t expert = expert_order[index];
    stats.active_expert_indices.push_back(expert);
    stats.active_encoded_bytes += experts[expert].encoded_bytes;
    stats.active_exception_count += experts[expert].exception_count;
  }
  stats.active_block_count = blocks_per_expert * manifest.selected_count;
  stats.active_source_bytes = stats.active_block_count * kQ6BlockBytes;
  return stats;
}

std::size_t HistogramPercentile(
    const std::array<std::uint64_t, kQ6Codes + 1>& histogram,
    std::uint64_t count, double percentile) {
  const std::uint64_t target = static_cast<std::uint64_t>(
      std::ceil(percentile * static_cast<double>(count)));
  std::uint64_t cumulative = 0;
  for (std::size_t index = 0; index < histogram.size(); ++index) {
    cumulative += histogram[index];
    if (cumulative >= target) return index;
  }
  return histogram.size() - 1;
}

template <typename Values>
void WriteU64Array(const Values& values) {
  std::cout << "[";
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0) std::cout << ",";
    std::cout << values[index];
  }
  std::cout << "]";
}

void WriteTensorJson(const TensorStats& stats) {
  const double mean_exceptions = static_cast<double>(stats.exception_count) /
                                 static_cast<double>(stats.block_count);
  std::cout << "{\"name\":\"" << JsonEscape(stats.manifest.name) << "\","
            << "\"suffix\":\"" << JsonEscape(stats.manifest.suffix) << "\","
            << "\"source_bytes\":" << stats.manifest.bytes << ","
            << "\"block_count\":" << stats.block_count << ","
            << "\"encoded_bytes\":" << stats.encoded_bytes << ","
            << "\"encoded_over_source_ratio\":" << std::setprecision(12)
            << static_cast<double>(stats.encoded_bytes) /
                   static_cast<double>(stats.manifest.bytes) << ","
            << "\"exception_count\":" << stats.exception_count << ","
            << "\"exceptions_per_block_mean\":" << mean_exceptions << ","
            << "\"exceptions_per_block_p50\":"
            << HistogramPercentile(stats.exception_histogram, stats.block_count, 0.50)
            << ",\"exceptions_per_block_p90\":"
            << HistogramPercentile(stats.exception_histogram, stats.block_count, 0.90)
            << ",\"exceptions_per_block_p99\":"
            << HistogramPercentile(stats.exception_histogram, stats.block_count, 0.99)
            << ",\"correction_abs_sum\":" << stats.correction_abs_sum << ","
            << "\"reconstruction_mismatch_count\":"
            << stats.reconstruction_mismatches << ","
            << "\"expert_count\":" << stats.manifest.expert_count << ","
            << "\"selected_count\":" << stats.manifest.selected_count << ","
            << "\"active_block_count\":" << stats.active_block_count << ","
            << "\"active_source_bytes\":" << stats.active_source_bytes << ","
            << "\"active_encoded_bytes\":" << stats.active_encoded_bytes << ","
            << "\"active_exception_count\":"
            << stats.active_exception_count << ","
            << "\"active_expert_indices\":";
  WriteU64Array(stats.active_expert_indices);
  std::cout << ",\"window_start_histogram\":";
  WriteU64Array(stats.window_start_histogram);
  std::cout << "}";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args args = ParseArgs(argc, argv);
    const auto manifest = ReadManifest(args.manifest);
    std::ifstream model(args.model, std::ios::binary);
    if (!model) {
      throw std::runtime_error("failed to open model: " + args.model);
    }
    std::vector<TensorStats> tensor_stats;
    tensor_stats.reserve(manifest.size());
    for (const auto& row : manifest) {
      tensor_stats.push_back(ProcessTensor(model, row));
    }

    std::uint64_t source_bytes = 0;
    std::uint64_t blocks = 0;
    std::uint64_t encoded_bytes = 0;
    std::uint64_t exceptions = 0;
    std::uint64_t mismatches = 0;
    std::uint64_t active_source_bytes = 0;
    std::uint64_t active_blocks = 0;
    std::uint64_t active_encoded_bytes = 0;
    std::uint64_t active_exceptions = 0;
    for (const auto& stats : tensor_stats) {
      source_bytes += stats.manifest.bytes;
      blocks += stats.block_count;
      encoded_bytes += stats.encoded_bytes;
      exceptions += stats.exception_count;
      mismatches += stats.reconstruction_mismatches;
      active_source_bytes += stats.active_source_bytes;
      active_blocks += stats.active_block_count;
      active_encoded_bytes += stats.active_encoded_bytes;
      active_exceptions += stats.active_exception_count;
    }

    std::cout << "{\"schema_version\":"
              << "\"intel-qwen36-q6-sparse-exception-feasibility-core-v1\","
              << "\"format\":{\"base_bits\":4,\"base_window_width\":16,"
              << "\"base_code_bytes_per_block\":" << kBaseCodeBytes << ","
              << "\"scale_bytes_per_block\":" << kScaleBytes << ","
              << "\"header_bytes_per_block\":" << kBlockHeaderBytes << ","
              << "\"directory_bytes_per_block\":" << kBlockDirectoryBytes << ","
              << "\"exception_bytes\":2,\"payload_alignment_bytes\":4},"
              << "\"tensor_rows\":[";
    for (std::size_t index = 0; index < tensor_stats.size(); ++index) {
      if (index != 0) std::cout << ",";
      WriteTensorJson(tensor_stats[index]);
    }
    std::cout << "],\"aggregate\":{"
              << "\"tensor_count\":" << tensor_stats.size() << ","
              << "\"source_bytes\":" << source_bytes << ","
              << "\"block_count\":" << blocks << ","
              << "\"encoded_bytes\":" << encoded_bytes << ","
              << "\"exception_count\":" << exceptions << ","
              << "\"exceptions_per_block_mean\":" << std::setprecision(12)
              << static_cast<double>(exceptions) / static_cast<double>(blocks)
              << ",\"reconstruction_mismatch_count\":" << mismatches << ","
              << "\"active_source_bytes\":" << active_source_bytes << ","
              << "\"active_block_count\":" << active_blocks << ","
              << "\"active_encoded_bytes\":" << active_encoded_bytes << ","
              << "\"active_exception_count\":" << active_exceptions << ","
              << "\"active_exceptions_per_block_mean\":"
              << static_cast<double>(active_exceptions) /
                     static_cast<double>(active_blocks)
              << "}}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << "\n";
    return 1;
  }
}
