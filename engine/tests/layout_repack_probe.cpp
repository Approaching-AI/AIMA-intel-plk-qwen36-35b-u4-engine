#include "intel_qwen36/gguf_loader.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::uint64_t kFnvOffset = 14695981039346656037ULL;
constexpr std::uint64_t kFnvPrime = 1099511628211ULL;

volatile std::uint64_t g_stream_sink = 0;

struct LaneSpec {
  std::string suffix;
  std::string type_name;
};

struct RepackedTensor {
  std::string layout;
  std::uint64_t block_count = 0;
  std::vector<std::uint8_t> q4;
  std::vector<std::uint8_t> ql;
  std::vector<std::uint8_t> qh;
  std::vector<std::uint8_t> scales;
  std::vector<std::uint8_t> mins;
  std::vector<std::uint8_t> d_values;
  std::vector<std::uint8_t> dmin_values;
};

struct StreamStats {
  std::uint64_t bytes_per_iteration = 0;
  std::uint64_t total_bytes = 0;
  std::uint64_t ns = 0;
  double gb_s = 0.0;
  std::uint64_t checksum = 0;
};

struct ProbeRow {
  std::uint64_t absolute_offset = 0;
  std::uint64_t block_count = 0;
  std::vector<std::uint64_t> dims;
  std::uint64_t layer_index = 0;
  std::string layout;
  std::string name;
  std::uint64_t raw_bytes = 0;
  std::uint64_t raw_checksum = 0;
  StreamStats raw_stream;
  std::uint64_t repack_ns = 0;
  std::uint64_t repacked_bytes = 0;
  std::uint64_t repacked_checksum = 0;
  StreamStats repacked_stream;
  StreamStats repacked_quant_only_stream;
  std::string selected_by_lane;
  std::string suffix;
  std::string type_name;
  std::uint64_t q4_bytes = 0;
  std::uint64_t ql_bytes = 0;
  std::uint64_t qh_bytes = 0;
  std::uint64_t scale_bytes = 0;
  std::uint64_t min_bytes = 0;
  std::uint64_t d_bytes = 0;
  std::uint64_t dmin_bytes = 0;
};

void require(bool ok, const std::string& message) {
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

std::uint64_t fnv64_update(std::uint64_t hash,
                           const std::uint8_t* data,
                           std::size_t size) {
  for (std::size_t i = 0; i < size; ++i) {
    hash ^= static_cast<std::uint64_t>(data[i]);
    hash *= kFnvPrime;
  }
  return hash;
}

std::uint64_t fnv64(const std::vector<std::uint8_t>& data) {
  return fnv64_update(kFnvOffset, data.data(), data.size());
}

std::uint64_t fnv64_buffers(
    const std::vector<const std::vector<std::uint8_t>*>& buffers) {
  std::uint64_t hash = kFnvOffset;
  for (const auto* buffer : buffers) {
    const auto size = static_cast<std::uint64_t>(buffer->size());
    hash = fnv64_update(hash,
                        reinterpret_cast<const std::uint8_t*>(&size),
                        sizeof(size));
    hash = fnv64_update(hash, buffer->data(), buffer->size());
  }
  return hash;
}

std::uint64_t vector_sample_checksum(const std::vector<std::uint8_t>& data) {
  if (data.empty()) {
    return 0;
  }
  std::uint64_t value = static_cast<std::uint64_t>(data.front());
  value = (value << 8) ^ static_cast<std::uint64_t>(data[data.size() / 2]);
  value = (value << 8) ^ static_cast<std::uint64_t>(data.back());
  value ^= static_cast<std::uint64_t>(data.size());
  return value;
}

std::uint64_t total_bytes(
    const std::vector<const std::vector<std::uint8_t>*>& buffers) {
  std::uint64_t total = 0;
  for (const auto* buffer : buffers) {
    total += static_cast<std::uint64_t>(buffer->size());
  }
  return total;
}

StreamStats measure_stream(
    const std::vector<const std::vector<std::uint8_t>*>& buffers,
    int iterations) {
  require(iterations > 0, "stream iterations must be positive");
  std::vector<std::vector<std::uint8_t>> destinations;
  destinations.reserve(buffers.size());
  for (const auto* buffer : buffers) {
    destinations.emplace_back(buffer->size());
  }

  const auto begin = std::chrono::steady_clock::now();
  std::uint64_t checksum = 0;
  for (int iteration = 0; iteration < iterations; ++iteration) {
    for (std::size_t i = 0; i < buffers.size(); ++i) {
      const auto& source = *buffers[i];
      if (source.empty()) {
        continue;
      }
      auto& destination = destinations[i];
      std::memcpy(destination.data(), source.data(), source.size());
      checksum = (checksum << 7) ^ (checksum >> 3) ^
                 vector_sample_checksum(destination) ^
                 static_cast<std::uint64_t>(iteration + 1);
    }
  }
  const auto end = std::chrono::steady_clock::now();
  auto ns = static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count());
  if (ns == 0) {
    ns = 1;
  }

  const auto bytes_per_iteration = total_bytes(buffers);
  const auto total_streamed =
      bytes_per_iteration * static_cast<std::uint64_t>(iterations);
  g_stream_sink ^= checksum;

  StreamStats stats;
  stats.bytes_per_iteration = bytes_per_iteration;
  stats.total_bytes = total_streamed;
  stats.ns = ns;
  stats.gb_s = static_cast<double>(total_streamed) / static_cast<double>(ns);
  stats.checksum = checksum;
  return stats;
}

std::array<std::uint8_t, 8> unpack_q4_k_scale_plane(
    const std::uint8_t* block,
    bool mins) {
  constexpr std::uint32_t kMask1 = 0x3f3f3f3f;
  constexpr std::uint32_t kMask2 = 0x0f0f0f0f;
  constexpr std::uint32_t kMask3 = 0x03030303;
  std::array<std::uint32_t, 4> unpacked{};
  std::memcpy(unpacked.data(), block + 4, 12);
  unpacked[3] = ((unpacked[2] >> 4) & kMask2) |
                (((unpacked[1] >> 6) & kMask3) << 4);
  const std::uint32_t aux_scales = unpacked[1] & kMask1;
  unpacked[1] = (unpacked[2] & kMask2) |
                (((unpacked[0] >> 6) & kMask3) << 4);
  unpacked[2] = aux_scales;
  unpacked[0] &= kMask1;

  std::array<std::uint8_t, 8> out{};
  const auto* base = reinterpret_cast<const std::uint8_t*>(unpacked.data());
  std::memcpy(out.data(), base + (mins ? 8 : 0), out.size());
  return out;
}

RepackedTensor repack_q4_k(const std::vector<std::uint8_t>& raw) {
  constexpr std::uint64_t kBlockBytes = 144;
  require(raw.size() % kBlockBytes == 0, "Q4_K tensor is not block-aligned");
  RepackedTensor out;
  out.layout = "q4k_plane_v0";
  out.block_count = static_cast<std::uint64_t>(raw.size()) / kBlockBytes;
  out.q4.resize(out.block_count * 128);
  out.scales.resize(out.block_count * 8);
  out.mins.resize(out.block_count * 8);
  out.d_values.resize(out.block_count * 2);
  out.dmin_values.resize(out.block_count * 2);
  for (std::uint64_t block_index = 0; block_index < out.block_count; ++block_index) {
    const auto* block = raw.data() + block_index * kBlockBytes;
    std::memcpy(out.d_values.data() + block_index * 2, block, 2);
    std::memcpy(out.dmin_values.data() + block_index * 2, block + 2, 2);
    const auto scales = unpack_q4_k_scale_plane(block, false);
    const auto mins = unpack_q4_k_scale_plane(block, true);
    std::memcpy(out.scales.data() + block_index * 8, scales.data(), scales.size());
    std::memcpy(out.mins.data() + block_index * 8, mins.data(), mins.size());
    std::memcpy(out.q4.data() + block_index * 128, block + 16, 128);
  }
  return out;
}

RepackedTensor repack_q6_k(const std::vector<std::uint8_t>& raw) {
  constexpr std::uint64_t kBlockBytes = 210;
  require(raw.size() % kBlockBytes == 0, "Q6_K tensor is not block-aligned");
  RepackedTensor out;
  out.layout = "q6k_plane_v0";
  out.block_count = static_cast<std::uint64_t>(raw.size()) / kBlockBytes;
  out.ql.resize(out.block_count * 128);
  out.qh.resize(out.block_count * 64);
  out.scales.resize(out.block_count * 16);
  out.d_values.resize(out.block_count * 2);
  for (std::uint64_t block_index = 0; block_index < out.block_count; ++block_index) {
    const auto* block = raw.data() + block_index * kBlockBytes;
    std::memcpy(out.ql.data() + block_index * 128, block, 128);
    std::memcpy(out.qh.data() + block_index * 64, block + 128, 64);
    std::memcpy(out.scales.data() + block_index * 16, block + 192, 16);
    std::memcpy(out.d_values.data() + block_index * 2, block + 208, 2);
  }
  return out;
}

std::vector<const std::vector<std::uint8_t>*> repacked_stream_buffers(
    const RepackedTensor& repacked) {
  std::vector<const std::vector<std::uint8_t>*> buffers;
  for (const auto* buffer : {
           &repacked.q4,
           &repacked.ql,
           &repacked.qh,
           &repacked.scales,
           &repacked.mins,
           &repacked.d_values,
           &repacked.dmin_values,
       }) {
    if (!buffer->empty()) {
      buffers.push_back(buffer);
    }
  }
  return buffers;
}

std::vector<const std::vector<std::uint8_t>*> repacked_quant_buffers(
    const RepackedTensor& repacked) {
  std::vector<const std::vector<std::uint8_t>*> buffers;
  for (const auto* buffer : {&repacked.q4, &repacked.ql, &repacked.qh}) {
    if (!buffer->empty()) {
      buffers.push_back(buffer);
    }
  }
  return buffers;
}

std::vector<std::uint8_t> read_tensor_bytes(
    std::ifstream& in,
    const iq36::GgufTensorInfo& tensor) {
  require(tensor.nbytes <= static_cast<std::uint64_t>(
                              std::numeric_limits<std::size_t>::max()),
          "tensor too large for this host size_t");
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(tensor.nbytes));
  in.clear();
  in.seekg(static_cast<std::streamoff>(tensor.absolute_offset), std::ios::beg);
  require(static_cast<bool>(in), "failed to seek to tensor payload");
  in.read(reinterpret_cast<char*>(bytes.data()),
          static_cast<std::streamsize>(bytes.size()));
  require(in.gcount() == static_cast<std::streamsize>(bytes.size()),
          "failed to read full tensor payload");
  return bytes;
}

std::int64_t parse_i64(const std::string& value, const std::string& name) {
  std::size_t consumed = 0;
  const auto parsed = std::stoll(value, &consumed, 10);
  require(consumed == value.size(), name + " must be an integer");
  return parsed;
}

LaneSpec parse_lane(const std::string& value) {
  const auto split = value.rfind(':');
  require(split != std::string::npos && split > 0 && split + 1 < value.size(),
          "lane must use suffix:quant format");
  return LaneSpec{value.substr(0, split), value.substr(split + 1)};
}

std::string lane_key(const LaneSpec& lane) {
  return lane.suffix + ":" + lane.type_name;
}

bool tensor_matches_lane(const iq36::GgufTensorInfo& tensor,
                         const std::string& tensor_type_name,
                         const LaneSpec& lane) {
  const auto suffix = tensor.suffix.empty() ? tensor.name : tensor.suffix;
  return suffix == lane.suffix && tensor_type_name == lane.type_name;
}

ProbeRow probe_tensor(std::ifstream& in,
                      const iq36::GgufTensorInfo& tensor,
                      const LaneSpec& lane,
                      int iterations) {
  const auto type_name = iq36::ggml_type_name(tensor.type);
  const auto raw = read_tensor_bytes(in, tensor);
  const std::vector<const std::vector<std::uint8_t>*> raw_buffers{&raw};
  const auto raw_checksum = fnv64(raw);
  const auto raw_stream = measure_stream(raw_buffers, iterations);

  const auto repack_begin = std::chrono::steady_clock::now();
  const RepackedTensor repacked =
      type_name == "Q4_K" ? repack_q4_k(raw) : repack_q6_k(raw);
  const auto repack_end = std::chrono::steady_clock::now();
  auto repack_ns = static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          repack_end - repack_begin)
          .count());
  if (repack_ns == 0) {
    repack_ns = 1;
  }

  const auto repacked_buffers = repacked_stream_buffers(repacked);
  const auto repacked_quant_only_buffers = repacked_quant_buffers(repacked);
  const auto repacked_bytes = total_bytes(repacked_buffers);
  const auto repacked_checksum = fnv64_buffers(repacked_buffers);
  const auto repacked_stream = measure_stream(repacked_buffers, iterations);
  const auto repacked_quant_only_stream =
      measure_stream(repacked_quant_only_buffers, iterations);

  ProbeRow row;
  row.absolute_offset = tensor.absolute_offset;
  row.block_count = repacked.block_count;
  row.dims = tensor.dims;
  row.layer_index = static_cast<std::uint64_t>(std::max(tensor.layer_index, 0));
  row.layout = repacked.layout;
  row.name = tensor.name;
  row.raw_bytes = static_cast<std::uint64_t>(raw.size());
  row.raw_checksum = raw_checksum;
  row.raw_stream = raw_stream;
  row.repack_ns = repack_ns;
  row.repacked_bytes = repacked_bytes;
  row.repacked_checksum = repacked_checksum;
  row.repacked_stream = repacked_stream;
  row.repacked_quant_only_stream = repacked_quant_only_stream;
  row.selected_by_lane = lane_key(lane);
  row.suffix = tensor.suffix.empty() ? tensor.name : tensor.suffix;
  row.type_name = type_name;
  row.q4_bytes = static_cast<std::uint64_t>(repacked.q4.size());
  row.ql_bytes = static_cast<std::uint64_t>(repacked.ql.size());
  row.qh_bytes = static_cast<std::uint64_t>(repacked.qh.size());
  row.scale_bytes = static_cast<std::uint64_t>(repacked.scales.size());
  row.min_bytes = static_cast<std::uint64_t>(repacked.mins.size());
  row.d_bytes = static_cast<std::uint64_t>(repacked.d_values.size());
  row.dmin_bytes = static_cast<std::uint64_t>(repacked.dmin_values.size());
  return row;
}

void write_stream_stats(const StreamStats& stats) {
  std::cout << "{";
  std::cout << "\"bytes_per_iteration\":" << stats.bytes_per_iteration << ",";
  std::cout << "\"checksum\":" << stats.checksum << ",";
  std::cout << "\"gb_s\":" << stats.gb_s << ",";
  std::cout << "\"ns\":" << stats.ns << ",";
  std::cout << "\"total_bytes\":" << stats.total_bytes;
  std::cout << "}";
}

void write_probe_row(const ProbeRow& row) {
  const double overhead =
      row.raw_bytes == 0
          ? 0.0
          : static_cast<double>(row.repacked_bytes) /
                static_cast<double>(row.raw_bytes);
  const double repack_gb_s =
      static_cast<double>(row.raw_bytes) / static_cast<double>(row.repack_ns);
  std::cout << "{";
  std::cout << "\"absolute_offset\":" << row.absolute_offset << ",";
  std::cout << "\"block_count\":" << row.block_count << ",";
  std::cout << "\"dims\":";
  write_u64_vector(row.dims);
  std::cout << ",";
  std::cout << "\"layer_index\":" << row.layer_index << ",";
  std::cout << "\"layout\":\"" << json_escape(row.layout) << "\",";
  std::cout << "\"name\":\"" << json_escape(row.name) << "\",";
  std::cout << "\"plane_bytes\":{";
  std::cout << "\"d\":" << row.d_bytes << ",";
  std::cout << "\"dmin\":" << row.dmin_bytes << ",";
  std::cout << "\"mins\":" << row.min_bytes << ",";
  std::cout << "\"q4\":" << row.q4_bytes << ",";
  std::cout << "\"qh\":" << row.qh_bytes << ",";
  std::cout << "\"ql\":" << row.ql_bytes << ",";
  std::cout << "\"scales\":" << row.scale_bytes;
  std::cout << "},";
  std::cout << "\"raw_bytes\":" << row.raw_bytes << ",";
  std::cout << "\"raw_checksum\":" << row.raw_checksum << ",";
  std::cout << "\"raw_stream\":";
  write_stream_stats(row.raw_stream);
  std::cout << ",";
  std::cout << "\"repack_gb_s\":" << repack_gb_s << ",";
  std::cout << "\"repack_ns\":" << row.repack_ns << ",";
  std::cout << "\"repacked_bytes\":" << row.repacked_bytes << ",";
  std::cout << "\"repacked_checksum\":" << row.repacked_checksum << ",";
  std::cout << "\"repacked_overhead_ratio\":" << overhead << ",";
  std::cout << "\"repacked_quant_only_stream\":";
  write_stream_stats(row.repacked_quant_only_stream);
  std::cout << ",";
  std::cout << "\"repacked_stream\":";
  write_stream_stats(row.repacked_stream);
  std::cout << ",";
  std::cout << "\"selected_by_lane\":\"" << json_escape(row.selected_by_lane)
            << "\",";
  std::cout << "\"suffix\":\"" << json_escape(row.suffix) << "\",";
  std::cout << "\"type_name\":\"" << json_escape(row.type_name) << "\"";
  std::cout << "}";
}

void usage() {
  std::cerr
      << "usage: iq36-layout-repack-probe <model.gguf> "
      << "[--lane suffix:quant] [--iterations N] "
      << "[--max-tensors-per-lane N]\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 2) {
      usage();
      return 1;
    }
    const std::string model_path = argv[1];
    std::vector<LaneSpec> lanes;
    int iterations = 5;
    int max_tensors_per_lane = 1;

    for (int i = 2; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--lane") {
        require(i + 1 < argc, "--lane requires a value");
        lanes.push_back(parse_lane(argv[++i]));
      } else if (arg.rfind("--lane=", 0) == 0) {
        lanes.push_back(parse_lane(arg.substr(std::string("--lane=").size())));
      } else if (arg == "--iterations") {
        require(i + 1 < argc, "--iterations requires a value");
        iterations = static_cast<int>(parse_i64(argv[++i], "--iterations"));
      } else if (arg.rfind("--iterations=", 0) == 0) {
        iterations = static_cast<int>(
            parse_i64(arg.substr(std::string("--iterations=").size()),
                      "--iterations"));
      } else if (arg == "--max-tensors-per-lane") {
        require(i + 1 < argc, "--max-tensors-per-lane requires a value");
        max_tensors_per_lane =
            static_cast<int>(parse_i64(argv[++i], "--max-tensors-per-lane"));
      } else if (arg.rfind("--max-tensors-per-lane=", 0) == 0) {
        max_tensors_per_lane = static_cast<int>(
            parse_i64(arg.substr(std::string("--max-tensors-per-lane=").size()),
                      "--max-tensors-per-lane"));
      } else {
        throw std::runtime_error("unknown argument: " + arg);
      }
    }

    if (lanes.empty()) {
      lanes.push_back(parse_lane("attn_qkv.weight:Q4_K"));
      lanes.push_back(parse_lane("ffn_gate_up_exps.weight:Q4_K"));
    }
    require(iterations > 0, "--iterations must be positive");
    require(max_tensors_per_lane > 0, "--max-tensors-per-lane must be positive");
    for (const auto& lane : lanes) {
      require(lane.type_name == "Q4_K" || lane.type_name == "Q6_K",
              "only Q4_K and Q6_K lanes are supported");
    }

    const auto index = iq36::parse_gguf_model_index(model_path);
    const auto summary = iq36::validate_qwen36_load_map(index);
    std::ifstream in(model_path, std::ios::binary);
    require(static_cast<bool>(in), "failed to open model");

    std::map<std::string, int> selected_counts;
    std::vector<ProbeRow> rows;
    for (const auto& tensor : index.tensors) {
      const auto type_name = iq36::ggml_type_name(tensor.type);
      for (const auto& lane : lanes) {
        const auto key = lane_key(lane);
        if (selected_counts[key] >= max_tensors_per_lane) {
          continue;
        }
        if (!tensor_matches_lane(tensor, type_name, lane)) {
          continue;
        }
        rows.push_back(probe_tensor(in, tensor, lane, iterations));
        ++selected_counts[key];
        break;
      }
    }

    std::uint64_t aggregate_raw_bytes = 0;
    std::uint64_t aggregate_repacked_bytes = 0;
    std::uint64_t aggregate_repack_ns = 0;
    std::uint64_t aggregate_raw_stream_bytes = 0;
    std::uint64_t aggregate_raw_stream_ns = 0;
    std::uint64_t aggregate_repacked_stream_bytes = 0;
    std::uint64_t aggregate_repacked_stream_ns = 0;
    std::uint64_t aggregate_quant_stream_bytes = 0;
    std::uint64_t aggregate_quant_stream_ns = 0;
    for (const auto& row : rows) {
      aggregate_raw_bytes += row.raw_bytes;
      aggregate_repacked_bytes += row.repacked_bytes;
      aggregate_repack_ns += row.repack_ns;
      aggregate_raw_stream_bytes += row.raw_stream.total_bytes;
      aggregate_raw_stream_ns += row.raw_stream.ns;
      aggregate_repacked_stream_bytes += row.repacked_stream.total_bytes;
      aggregate_repacked_stream_ns += row.repacked_stream.ns;
      aggregate_quant_stream_bytes += row.repacked_quant_only_stream.total_bytes;
      aggregate_quant_stream_ns += row.repacked_quant_only_stream.ns;
    }

    std::cout << std::setprecision(10);
    std::cout << "{";
    std::cout
        << "\"schema_version\":\"intel-qwen36-r3-repack-source-stream-probe-v0\",";
    std::cout << "\"model_path\":\"" << json_escape(model_path) << "\",";
    std::cout << "\"file_size_bytes\":" << index.file_size_bytes << ",";
    std::cout << "\"iterations\":" << iterations << ",";
    std::cout << "\"max_tensors_per_lane\":" << max_tensors_per_lane << ",";
    std::cout << "\"native_gguf_load_map_ready\":"
              << (summary.ready ? "true" : "false") << ",";
    std::cout << "\"lanes\":[";
    for (std::size_t i = 0; i < lanes.size(); ++i) {
      if (i != 0) {
        std::cout << ",";
      }
      std::cout << "{";
      std::cout << "\"suffix\":\"" << json_escape(lanes[i].suffix) << "\",";
      std::cout << "\"type_name\":\"" << json_escape(lanes[i].type_name)
                << "\"";
      std::cout << "}";
    }
    std::cout << "],";
    std::cout << "\"rows\":[";
    for (std::size_t i = 0; i < rows.size(); ++i) {
      if (i != 0) {
        std::cout << ",";
      }
      write_probe_row(rows[i]);
    }
    std::cout << "],";
    std::cout << "\"aggregate\":{";
    std::cout << "\"raw_bytes\":" << aggregate_raw_bytes << ",";
    std::cout << "\"raw_stream_gb_s_weighted\":"
              << (aggregate_raw_stream_ns == 0
                      ? 0.0
                      : static_cast<double>(aggregate_raw_stream_bytes) /
                            static_cast<double>(aggregate_raw_stream_ns))
              << ",";
    std::cout << "\"raw_stream_ns\":" << aggregate_raw_stream_ns << ",";
    std::cout << "\"repack_gb_s\":"
              << (aggregate_repack_ns == 0
                      ? 0.0
                      : static_cast<double>(aggregate_raw_bytes) /
                            static_cast<double>(aggregate_repack_ns))
              << ",";
    std::cout << "\"repack_ns\":" << aggregate_repack_ns << ",";
    std::cout << "\"repacked_bytes\":" << aggregate_repacked_bytes << ",";
    std::cout << "\"repacked_overhead_ratio\":"
              << (aggregate_raw_bytes == 0
                      ? 0.0
                      : static_cast<double>(aggregate_repacked_bytes) /
                            static_cast<double>(aggregate_raw_bytes))
              << ",";
    std::cout << "\"repacked_quant_only_stream_gb_s_weighted\":"
              << (aggregate_quant_stream_ns == 0
                      ? 0.0
                      : static_cast<double>(aggregate_quant_stream_bytes) /
                            static_cast<double>(aggregate_quant_stream_ns))
              << ",";
    std::cout << "\"repacked_quant_only_stream_ns\":"
              << aggregate_quant_stream_ns << ",";
    std::cout << "\"repacked_stream_gb_s_weighted\":"
              << (aggregate_repacked_stream_ns == 0
                      ? 0.0
                      : static_cast<double>(aggregate_repacked_stream_bytes) /
                            static_cast<double>(aggregate_repacked_stream_ns))
              << ",";
    std::cout << "\"repacked_stream_ns\":" << aggregate_repacked_stream_ns
              << ",";
    std::cout << "\"selected_tensor_count\":" << rows.size() << ",";
    std::cout << "\"stream_sink\":" << g_stream_sink;
    std::cout << "}";
    std::cout << "}\n";
    return rows.empty() ? 2 : 0;
  } catch (const std::exception& exc) {
    std::cerr << "iq36-layout-repack-probe: " << exc.what() << "\n";
    return 1;
  }
}
