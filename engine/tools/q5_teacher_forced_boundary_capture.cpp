#include "llama.h"

#include "ggml.h"
#include "ggml-backend.h"

#ifdef IQ36_GROUPED_LIVE_INJECTION
#include "intel_qwen36/grouped_s8_u4_prefill_runtime.hpp"
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cerrno>
#include <climits>
#include <clocale>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

struct Args {
  std::string model;
  std::string token_ids_file;
  std::string out_dir;
  std::string case_id = "router_math_reason_001";
  int threads = 2;
  int n_ctx = 64;
  int n_gpu_layers = 0;
  int top_k = 16;
  int predicts_generated_position = -1;
  int token_offset = 0;
  int token_count = 0;
  int component_layer = -1;
  int linear_component_layer = -1;
  bool binary_u32_token_file = false;
  bool batch_all = false;
  bool component_all_layers = false;
  bool component_through_down = false;
  bool component_through_ffn_out = false;
  bool router_only = false;
  std::vector<int> watch_tokens;
#ifdef IQ36_GROUPED_LIVE_INJECTION
  bool live_injection_boundaries = false;
  bool no_tensor_dumps = false;
  bool full_logits = false;
  int generate_count = 0;
  std::vector<int> generate_teacher_tokens;
  std::string inject_prep_root;
  std::string inject_gateup_binary;
  std::string inject_down_binary;
  std::string inject_support_kernel;
  std::string inject_q6_kernel;
  std::string inject_apply_codec = "all";
  bool inject_q4_f32_contributions = false;
  bool inject_q4_exact_block = false;
  int inject_layer_start = 0;
  int inject_layer_end = 40;
  bool inject_reference_swiglu = false;
  bool inject_reference_down = false;
#endif
};

void Usage(const char* argv0) {
  std::fprintf(
      stderr,
      "usage: %s --model MODEL --token-ids-file FILE --out-dir DIR [options]\n"
      "  --case-id ID       metadata case id\n"
      "  --threads N        CPU decode threads (default 2)\n"
      "  --n-ctx N          context size (default 64)\n"
      "  --ngl N            GPU layers (default 0)\n"
      "  --top-k N          logits rows to record (default 16)\n"
      "  --predicts-generated-position N  oracle position predicted by the prefix\n"
      "  --binary-u32-token-file  read little-endian uint32 token ids\n"
      "  --token-offset N   first token index to read (default 0)\n"
      "  --token-count N    number of token ids to read (default all)\n"
      "  --batch-all        evaluate selected tokens in one logical batch\n"
      "  --router-only      capture only ffn_moe_topk layer tensors\n"
      "  --component-layer N  capture FFN input/top-k/SwiGLU for one layer\n"
      "  --linear-component-layer N  capture the 1024-token linear-attention "
      "state boundary for one layer\n"
      "  --component-all-layers  capture FFN inputs/outputs for all layers\n"
      "  --component-through-down  also capture router weights/down/MoE output\n"
      "  --component-through-ffn-out  also capture shared expert/final add\n"
      "  --watch-token ID   extra token logit to record; repeatable\n",
      argv0);
#ifdef IQ36_GROUPED_LIVE_INJECTION
  std::fprintf(
      stderr,
      "  --live-injection-boundaries  request only the four routed-MoE "
      "injection boundaries\n"
      "  --no-tensor-dumps  run callbacks without writing boundary payloads\n"
      "  --full-logits      write final full-logits.f32.bin\n"
      "  --generate-count N  greedily continue N tokens after the prompt\n"
      "  --generate-teacher-token ID  feed this fixed continuation token; "
      "repeat once per generated position\n"
      "  --inject-prep-root DIR --inject-gateup-binary FILE\n"
      "  --inject-down-binary FILE --inject-support-kernel FILE\n"
      "  --inject-q6-kernel FILE  enable all-40 resident routed-MoE injection\n");
  std::fprintf(
      stderr,
      "  --inject-apply-codec all|q4|q6  compare all layers but overwrite "
      "only the selected codec\n"
      "  --inject-q4-f32-contributions  interpret the Q4 down binary and "
      "scatter plane as F32\n"
      "  --inject-q4-exact-block  use exact-block Q4 gate/up and down\n"
      "  --inject-layer-start N --inject-layer-end N  overwrite only the "
      "half-open layer range [start,end)\n"
      "  --inject-reference-swiglu  diagnostic: feed captured reference "
      "SwiGLU into native down\n"
      "  --inject-reference-down  diagnostic: feed captured reference down "
      "through native router scatter\n");
#endif
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
    if (option == "-h" || option == "--help") {
      Usage(argv[0]);
      std::exit(0);
    } else if (option == "-m" || option == "--model") {
      args.model = Value(option.c_str());
    } else if (option == "--token-ids-file") {
      args.token_ids_file = Value(option.c_str());
    } else if (option == "--out-dir") {
      args.out_dir = Value(option.c_str());
    } else if (option == "--case-id") {
      args.case_id = Value(option.c_str());
    } else if (option == "--threads") {
      args.threads = std::stoi(Value(option.c_str()));
    } else if (option == "--n-ctx") {
      args.n_ctx = std::stoi(Value(option.c_str()));
    } else if (option == "--ngl") {
      args.n_gpu_layers = std::stoi(Value(option.c_str()));
    } else if (option == "--top-k") {
      args.top_k = std::stoi(Value(option.c_str()));
    } else if (option == "--predicts-generated-position") {
      args.predicts_generated_position = std::stoi(Value(option.c_str()));
    } else if (option == "--binary-u32-token-file") {
      args.binary_u32_token_file = true;
    } else if (option == "--token-offset") {
      args.token_offset = std::stoi(Value(option.c_str()));
    } else if (option == "--token-count") {
      args.token_count = std::stoi(Value(option.c_str()));
    } else if (option == "--batch-all") {
      args.batch_all = true;
    } else if (option == "--router-only") {
      args.router_only = true;
    } else if (option == "--component-layer") {
      args.component_layer = std::stoi(Value(option.c_str()));
    } else if (option == "--linear-component-layer") {
      args.linear_component_layer = std::stoi(Value(option.c_str()));
    } else if (option == "--component-all-layers") {
      args.component_all_layers = true;
    } else if (option == "--component-through-down") {
      args.component_through_down = true;
    } else if (option == "--component-through-ffn-out") {
      args.component_through_ffn_out = true;
    } else if (option == "--watch-token") {
      args.watch_tokens.push_back(std::stoi(Value(option.c_str())));
#ifdef IQ36_GROUPED_LIVE_INJECTION
    } else if (option == "--live-injection-boundaries") {
      args.live_injection_boundaries = true;
    } else if (option == "--no-tensor-dumps") {
      args.no_tensor_dumps = true;
    } else if (option == "--full-logits") {
      args.full_logits = true;
    } else if (option == "--generate-count") {
      args.generate_count = std::stoi(Value(option.c_str()));
    } else if (option == "--generate-teacher-token") {
      args.generate_teacher_tokens.push_back(
          std::stoi(Value(option.c_str())));
    } else if (option == "--inject-prep-root") {
      args.inject_prep_root = Value(option.c_str());
    } else if (option == "--inject-gateup-binary") {
      args.inject_gateup_binary = Value(option.c_str());
    } else if (option == "--inject-down-binary") {
      args.inject_down_binary = Value(option.c_str());
    } else if (option == "--inject-support-kernel") {
      args.inject_support_kernel = Value(option.c_str());
    } else if (option == "--inject-q6-kernel") {
      args.inject_q6_kernel = Value(option.c_str());
    } else if (option == "--inject-apply-codec") {
      args.inject_apply_codec = Value(option.c_str());
    } else if (option == "--inject-q4-f32-contributions") {
      args.inject_q4_f32_contributions = true;
    } else if (option == "--inject-q4-exact-block") {
      args.inject_q4_exact_block = true;
    } else if (option == "--inject-layer-start") {
      args.inject_layer_start = std::stoi(Value(option.c_str()));
    } else if (option == "--inject-layer-end") {
      args.inject_layer_end = std::stoi(Value(option.c_str()));
    } else if (option == "--inject-reference-swiglu") {
      args.inject_reference_swiglu = true;
    } else if (option == "--inject-reference-down") {
      args.inject_reference_down = true;
#endif
    } else {
      throw std::runtime_error("unknown argument: " + option);
    }
  }
  if (args.model.empty() || args.token_ids_file.empty() || args.out_dir.empty()) {
    Usage(argv[0]);
    throw std::runtime_error("missing required model, token-id file, or output directory");
  }
  if (args.threads <= 0 || args.n_ctx <= 0 || args.top_k <= 0 ||
      args.predicts_generated_position < 0 || args.token_offset < 0 ||
      args.token_count < 0) {
    throw std::runtime_error("numeric arguments are outside their valid range");
  }
  if (args.component_layer < -1 || args.component_layer >= 40) {
    throw std::runtime_error("component layer must be in [0, 39]");
  }
  if (args.linear_component_layer < -1 ||
      args.linear_component_layer >= 40) {
    throw std::runtime_error(
        "linear component layer must be in [0, 39]");
  }
  if (args.linear_component_layer >= 0 &&
      (args.component_layer >= 0 || args.component_all_layers ||
       args.router_only)) {
    throw std::runtime_error(
        "linear-component-layer is mutually exclusive with FFN/router capture");
  }
  if (args.component_all_layers && args.component_layer >= 0) {
    throw std::runtime_error(
        "component-all-layers and component-layer are mutually exclusive");
  }
  if (args.router_only &&
      (args.component_layer >= 0 || args.component_all_layers)) {
    throw std::runtime_error(
        "router-only and component capture are mutually exclusive");
  }
  if (args.component_through_down && args.component_layer < 0 &&
      !args.component_all_layers) {
    throw std::runtime_error(
        "component-through-down requires component-layer or all-layers");
  }
  if (args.component_through_ffn_out && args.component_layer < 0 &&
      !args.component_all_layers) {
    throw std::runtime_error(
        "component-through-ffn-out requires component-layer or all-layers");
  }
#ifdef IQ36_GROUPED_LIVE_INJECTION
  const std::array<bool, 5> injection_paths = {
      !args.inject_prep_root.empty(), !args.inject_gateup_binary.empty(),
      !args.inject_down_binary.empty(), !args.inject_support_kernel.empty(),
      !args.inject_q6_kernel.empty()};
  const bool any_injection_path = std::any_of(
      injection_paths.begin(), injection_paths.end(), [](bool value) {
        return value;
      });
  const bool all_injection_paths = std::all_of(
      injection_paths.begin(), injection_paths.end(), [](bool value) {
        return value;
      });
  if (any_injection_path && !all_injection_paths) {
    throw std::runtime_error("all five injection paths are required together");
  }
  if (any_injection_path && !args.live_injection_boundaries) {
    throw std::runtime_error(
        "resident injection requires --live-injection-boundaries");
  }
  if (args.generate_count < 0) {
    throw std::runtime_error("generate count must be nonnegative");
  }
  if (!args.generate_teacher_tokens.empty() &&
      args.generate_teacher_tokens.size() !=
          static_cast<std::size_t>(args.generate_count)) {
    throw std::runtime_error(
        "generate teacher token count must equal generate-count");
  }
  if (std::any_of(args.generate_teacher_tokens.begin(),
                  args.generate_teacher_tokens.end(),
                  [](int token) { return token < 0; })) {
    throw std::runtime_error("generate teacher tokens must be nonnegative");
  }
  if (args.inject_apply_codec != "all" && args.inject_apply_codec != "q4" &&
      args.inject_apply_codec != "q6") {
    throw std::runtime_error("inject apply codec must be all, q4, or q6");
  }
  if (args.inject_layer_start < 0 ||
      args.inject_layer_start > args.inject_layer_end ||
      args.inject_layer_end > 40) {
    throw std::runtime_error("inject layer range must satisfy 0 <= start <= end <= 40");
  }
#endif
  return args;
}

std::vector<llama_token> ReadTokenIds(const std::string& path, bool binary_u32) {
  std::ifstream input(
      path, binary_u32 ? std::ios::binary | std::ios::in : std::ios::in);
  if (!input) {
    throw std::runtime_error("failed to open token-id file: " + path + ": " +
                             std::strerror(errno));
  }
  std::vector<llama_token> tokens;
  if (binary_u32) {
    std::array<std::uint8_t, 4> bytes{};
    while (input.read(reinterpret_cast<char*>(bytes.data()), bytes.size())) {
      const std::uint32_t value =
          std::uint32_t(bytes[0]) | (std::uint32_t(bytes[1]) << 8) |
          (std::uint32_t(bytes[2]) << 16) | (std::uint32_t(bytes[3]) << 24);
      if (value > static_cast<std::uint32_t>(INT32_MAX)) {
        throw std::runtime_error("token id is outside int32 range");
      }
      tokens.push_back(static_cast<llama_token>(value));
    }
    if (!input.eof() || input.gcount() != 0) {
      throw std::runtime_error("binary token-id file is not uint32 aligned");
    }
    if (tokens.empty()) {
      throw std::runtime_error("token-id file is empty");
    }
    return tokens;
  }
  long long value = 0;
  while (input >> value) {
    if (value < 0 || value > INT32_MAX) {
      throw std::runtime_error("token id is outside int32 range");
    }
    tokens.push_back(static_cast<llama_token>(value));
  }
  if (!input.eof()) {
    throw std::runtime_error("invalid token-id file: " + path);
  }
  if (tokens.empty()) {
    throw std::runtime_error("token-id file is empty");
  }
  return tokens;
}

std::string JsonEscape(const std::string& value) {
  std::string result;
  result.reserve(value.size() + 8);
  for (const unsigned char character : value) {
    switch (character) {
      case '\\': result += "\\\\"; break;
      case '"': result += "\\\""; break;
      case '\n': result += "\\n"; break;
      case '\r': result += "\\r"; break;
      case '\t': result += "\\t"; break;
      default:
        if (character < 0x20) {
          char buffer[8];
          std::snprintf(buffer, sizeof(buffer), "\\u%04x", character);
          result += buffer;
        } else {
          result.push_back(static_cast<char>(character));
        }
    }
  }
  return result;
}

std::string SafeName(const std::string& name) {
  std::string result;
  result.reserve(name.size());
  for (const char character : name) {
    if ((character >= 'a' && character <= 'z') ||
        (character >= 'A' && character <= 'Z') ||
        (character >= '0' && character <= '9') || character == '-' ||
        character == '_' || character == '.') {
      result.push_back(character);
    } else {
      result.push_back('_');
    }
  }
  return result.empty() ? "unnamed" : result;
}

#ifdef IQ36_GROUPED_LIVE_INJECTION

constexpr std::size_t kInjectionTokenCount = 1024;
constexpr std::size_t kInjectionHiddenSize = 2048;
constexpr std::size_t kInjectionIntermediateSize = 512;
constexpr std::size_t kInjectionAssignments = 8192;
constexpr std::uint64_t kInjectionLegacyResidentBytes = 23823646720ULL;
constexpr std::uint64_t kInjectionExactQ4ResidentBytes = 21726494720ULL;

constexpr std::array<int, 20> kInjectionQ6Layers = {
    0, 1, 2, 3, 4, 7, 10, 13, 16, 19,
    22, 25, 28, 31, 34, 35, 36, 37, 38, 39};

bool IsInjectionQ6Layer(int layer) {
  return std::find(kInjectionQ6Layers.begin(), kInjectionQ6Layers.end(),
                   layer) != kInjectionQ6Layers.end();
}

bool InjectionEnabled(const Args& args) {
  return !args.inject_prep_root.empty();
}

bool InjectionMapsExcludeDenominators() {
  std::ifstream maps("/proc/self/maps");
  std::string line;
  while (std::getline(maps, line)) {
    std::transform(line.begin(), line.end(), line.begin(),
                   [](unsigned char value) { return std::tolower(value); });
    if (line.find("libdnnl") != std::string::npos ||
        line.find("openvino") != std::string::npos) {
      return false;
    }
  }
  return true;
}

int LayerSuffix(const char* name, const char* prefix) {
  if (name == nullptr) return -1;
  const std::size_t prefix_size = std::strlen(prefix);
  if (std::strncmp(name, prefix, prefix_size) != 0) return -1;
  const char* suffix = name + prefix_size;
  if (*suffix == '\0') return -1;
  char* end = nullptr;
  const long layer = std::strtol(suffix, &end, 10);
  return end != nullptr && *end == '\0' && layer >= 0 && layer < 40
      ? static_cast<int>(layer)
      : -1;
}

std::vector<std::uint8_t> TensorBytes(const ggml_tensor* tensor) {
  if (tensor == nullptr || tensor->buffer == nullptr) {
    throw std::runtime_error("live-injection tensor has no backend buffer");
  }
  std::vector<std::uint8_t> bytes(ggml_nbytes(tensor));
  ggml_backend_tensor_get(tensor, bytes.data(), 0, bytes.size());
  return bytes;
}

std::vector<float> TensorFloats(const ggml_tensor* tensor,
                                std::size_t expected_count) {
  if (tensor == nullptr || tensor->type != GGML_TYPE_F32 ||
      ggml_nbytes(tensor) != expected_count * sizeof(float)) {
    throw std::runtime_error("live-injection F32 tensor shape mismatch");
  }
  const auto bytes = TensorBytes(tensor);
  std::vector<float> values(expected_count);
  std::memcpy(values.data(), bytes.data(), bytes.size());
  return values;
}

struct InjectionComparison {
  std::uint64_t count = 0;
  std::uint64_t mismatch_count = 0;
  long double error_squared = 0.0;
  long double candidate_squared = 0.0;
  long double reference_squared = 0.0;
  long double dot = 0.0;
  double max_abs_diff = 0.0;
  bool finite = true;

  void Add(float candidate, float reference) {
    const double difference = static_cast<double>(candidate) - reference;
    ++count;
    mismatch_count += std::abs(difference) > 5e-3;
    max_abs_diff = std::max(max_abs_diff, std::abs(difference));
    error_squared += difference * difference;
    candidate_squared += static_cast<double>(candidate) * candidate;
    reference_squared += static_cast<double>(reference) * reference;
    dot += static_cast<double>(candidate) * reference;
    finite = finite && std::isfinite(candidate) && std::isfinite(reference);
  }

  double Cosine() const {
    const long double denominator =
        std::sqrt(candidate_squared * reference_squared);
    return denominator == 0.0 ? 0.0
                              : static_cast<double>(dot / denominator);
  }

  double RelativeL2() const {
    return reference_squared == 0.0
        ? 0.0
        : static_cast<double>(std::sqrt(error_squared / reference_squared));
  }

  bool Pass() const {
    return finite && count == kInjectionTokenCount * kInjectionHiddenSize &&
        Cosine() >= 0.999 && RelativeL2() <= 0.002;
  }

  void Print(std::ostream& output) const {
    output << "{\"compared_value_count\":" << count
           << ",\"cosine\":" << Cosine()
           << ",\"finite\":" << std::boolalpha << finite
           << ",\"max_abs_diff\":" << max_abs_diff
           << ",\"mismatch_count\":" << mismatch_count
           << ",\"relative_l2\":" << RelativeL2() << "}";
  }
};

class LiveGroupedInjector {
 public:
  explicit LiveGroupedInjector(const Args& args)
      : runtime_(ProgramConfig(args)), apply_codec_(args.inject_apply_codec),
        layer_start_(args.inject_layer_start),
        layer_end_(args.inject_layer_end),
        expected_resident_bytes_(args.inject_q4_exact_block
            ? kInjectionExactQ4ResidentBytes
            : kInjectionLegacyResidentBytes),
        reference_swiglu_(args.inject_reference_swiglu),
        reference_down_(args.inject_reference_down) {
    const std::filesystem::path root(args.inject_prep_root);
    for (int layer = 0; layer < 40; ++layer) {
      std::ostringstream name;
      name << "layer-" << std::setfill('0') << std::setw(2) << layer;
      iq36::GroupedS8U4PrefillLayerConfig config;
      config.layer_index = layer;
      config.prep_dir = (root / name.str()).string();
      config.exact_q4_gateup = args.inject_q4_exact_block;
      if (IsInjectionQ6Layer(layer)) {
        config.down_kind = iq36::GroupedPrefillDownKind::kQ6U8ExactBlock;
      } else if (args.inject_q4_exact_block) {
        config.down_kind = iq36::GroupedPrefillDownKind::kQ4U4ExactBlock;
      } else if (args.inject_q4_f32_contributions) {
        config.down_kind =
            iq36::GroupedPrefillDownKind::kQ4U4F32Contribution;
      }
      handles_[static_cast<std::size_t>(layer)] = runtime_.LoadLayer(config);
    }
  }

  void Observe(ggml_tensor* tensor) {
    const char* name = tensor == nullptr ? nullptr : tensor->name;
    int layer = LayerSuffix(name, "attn_post_norm-");
    if (layer >= 0) {
      Pending& pending = pending_[static_cast<std::size_t>(layer)];
      pending.hidden_states = TensorFloats(
          tensor, kInjectionTokenCount * kInjectionHiddenSize);
      pending.has_hidden_states = true;
      return;
    }
    layer = LayerSuffix(name, "ffn_moe_topk-");
    if (layer >= 0) {
      if (tensor->type != GGML_TYPE_I32 || tensor->ne[0] != 8 ||
          tensor->ne[1] != static_cast<std::int64_t>(kInjectionTokenCount)) {
        throw std::runtime_error("live-injection top-k tensor mismatch");
      }
      Pending& pending = pending_[static_cast<std::size_t>(layer)];
      pending.topk = TensorBytes(tensor);
      pending.topk_stride = tensor->nb[1];
      pending.has_topk = true;
      return;
    }
    layer = LayerSuffix(name, "ffn_moe_weights_norm-");
    if (layer >= 0) {
      Pending& pending = pending_[static_cast<std::size_t>(layer)];
      pending.router_weights = TensorFloats(tensor, kInjectionAssignments);
      pending.has_router_weights = true;
      return;
    }
    layer = LayerSuffix(name, "ffn_moe_swiglu-");
    if (layer >= 0 && reference_swiglu_) {
      Pending& pending = pending_[static_cast<std::size_t>(layer)];
      pending.swiglu = TensorFloats(
          tensor, kInjectionAssignments * kInjectionIntermediateSize);
      pending.has_swiglu = true;
      return;
    }
    layer = LayerSuffix(name, "ffn_moe_down-");
    if (layer >= 0 && reference_down_) {
      Pending& pending = pending_[static_cast<std::size_t>(layer)];
      pending.down = TensorFloats(
          tensor, kInjectionAssignments * kInjectionHiddenSize);
      pending.has_down = true;
      return;
    }
    layer = LayerSuffix(name, "ffn_moe_out-");
    if (layer >= 0) Inject(layer, tensor);
  }

  bool WriteSummary(const std::filesystem::path& path) const {
    const auto stats = runtime_.stats();
    int expected_applied = 0;
    for (int layer = layer_start_; layer < layer_end_; ++layer) {
      expected_applied += ShouldApply(layer);
    }
    const bool rows_pass = injected_count_ == 40 &&
        std::all_of(rows_.begin(), rows_.end(), [](const Row& row) {
          return row.comparison.Pass();
        });
    const bool pass = rows_pass && aggregate_.finite &&
        aggregate_.Cosine() >= 0.999 && aggregate_.RelativeL2() <= 0.002 &&
        stats.context_create_count == 1 && stats.program_load_count == 4 &&
        stats.layer_count == 40 && stats.layer_load_count == 40 &&
        stats.run_count == 40 &&
        stats.resident_weight_bytes == expected_resident_bytes_ &&
        applied_count_ == expected_applied &&
        InjectionMapsExcludeDenominators();
    std::ofstream output(path);
    if (!output) {
      throw std::runtime_error("failed to create live-injection summary");
    }
    output << std::boolalpha << std::setprecision(12) << "{";
    output << "\"aggregate_routed_output_compare\":";
    aggregate_.Print(output);
    output << ",\"applied_codec\":\"" << apply_codec_ << "\"";
    output << ",\"applied_layer_count\":" << applied_count_;
    output << ",\"context_create_count\":" << stats.context_create_count;
    output << ",\"injection_count\":" << injected_count_;
    output << ",\"applied_layer_start\":" << layer_start_;
    output << ",\"applied_layer_end\":" << layer_end_;
    output << ",\"reference_swiglu\":" << reference_swiglu_;
    output << ",\"reference_down\":" << reference_down_;
    output << ",\"layer_count\":" << stats.layer_count;
    output << ",\"maps_exclude_onednn_openvino\":"
           << InjectionMapsExcludeDenominators();
    output << ",\"per_layer\":[";
    for (std::size_t index = 0; index < rows_.size(); ++index) {
      if (index != 0) output << ",";
      output << "{\"complete_minimum_us\":"
             << rows_[index].complete_minimum_us << ",\"layer\":" << index
             << ",\"routed_output_compare\":";
      rows_[index].comparison.Print(output);
      output << "}";
    }
    output << "],\"program_load_count\":" << stats.program_load_count;
    output << ",\"required_checks_passed\":" << pass;
    output << ",\"resident_weight_bytes\":"
           << stats.resident_weight_bytes;
    output << ",\"run_count\":" << stats.run_count << "}\n";
    return pass;
  }

 private:
  struct Pending {
    std::vector<float> hidden_states;
    std::vector<std::uint8_t> topk;
    std::size_t topk_stride = 0;
    std::vector<float> router_weights;
    std::vector<float> swiglu;
    std::vector<float> down;
    bool has_hidden_states = false;
    bool has_topk = false;
    bool has_router_weights = false;
    bool has_swiglu = false;
    bool has_down = false;
  };

  struct Row {
    InjectionComparison comparison;
    double complete_minimum_us = 0.0;
  };

  static iq36::GroupedS8U4PrefillProgramConfig ProgramConfig(
      const Args& args) {
    iq36::GroupedS8U4PrefillProgramConfig config;
    config.gateup_binary = args.inject_gateup_binary;
    config.down_binary = args.inject_down_binary;
    config.kernels = args.inject_support_kernel;
    config.q6_down_kernels = args.inject_q6_kernel;
    return config;
  }

  bool ShouldApply(int layer) const {
    const bool codec_selected = apply_codec_ == "all" ||
        (apply_codec_ == "q4" && !IsInjectionQ6Layer(layer)) ||
        (apply_codec_ == "q6" && IsInjectionQ6Layer(layer));
    return codec_selected && layer >= layer_start_ && layer < layer_end_;
  }

  void Inject(int layer, ggml_tensor* tensor) {
    if (layer != injected_count_) {
      throw std::runtime_error("live-injection layer order mismatch");
    }
    Pending& pending = pending_[static_cast<std::size_t>(layer)];
    const bool apply = ShouldApply(layer);
    if (!pending.has_hidden_states || !pending.has_topk ||
        !pending.has_router_weights ||
        (apply && reference_swiglu_ && !pending.has_swiglu) ||
        (apply && reference_down_ && !pending.has_down)) {
      throw std::runtime_error("live-injection boundary input is incomplete");
    }
    const auto reference = TensorFloats(
        tensor, kInjectionTokenCount * kInjectionHiddenSize);
    iq36::GroupedS8U4PrefillInput input;
    input.hidden_states = std::move(pending.hidden_states);
    input.topk = std::move(pending.topk);
    input.topk_stride = pending.topk_stride;
    input.router_weights = std::move(pending.router_weights);
    if (apply && reference_swiglu_) {
      input.swiglu_override_source_order = std::move(pending.swiglu);
    }
    if (apply && reference_down_) {
      input.down_override_source_order = std::move(pending.down);
    }
    input.warmup = 0;
    input.repeat = 1;
    const auto run = runtime_.RunLayer(
        handles_[static_cast<std::size_t>(layer)], input);
    if (run.output.size() != reference.size()) {
      throw std::runtime_error("live-injection output size mismatch");
    }
    Row& row = rows_[static_cast<std::size_t>(layer)];
    row.complete_minimum_us = run.timing.complete_minimum_us;
    for (std::size_t index = 0; index < reference.size(); ++index) {
      row.comparison.Add(run.output[index], reference[index]);
      aggregate_.Add(run.output[index], reference[index]);
    }
    if (apply) {
      ggml_backend_tensor_set(
          tensor, run.output.data(), 0, run.output.size() * sizeof(float));
      ++applied_count_;
    }
    pending = Pending{};
    ++injected_count_;
  }

  iq36::GroupedS8U4PrefillRuntime runtime_;
  std::array<std::uint64_t, 40> handles_{};
  std::array<Pending, 40> pending_{};
  std::array<Row, 40> rows_{};
  InjectionComparison aggregate_;
  std::string apply_codec_;
  int layer_start_ = 0;
  int layer_end_ = 40;
  std::uint64_t expected_resident_bytes_ = 0;
  bool reference_swiglu_ = false;
  bool reference_down_ = false;
  int injected_count_ = 0;
  int applied_count_ = 0;
};

#endif

struct CaptureState {
  std::filesystem::path out_dir;
  std::ofstream tensor_jsonl;
  std::vector<std::regex> filters;
  std::string case_id;
  int token_position = -1;
  int captured = 0;
  bool enabled = false;
#ifdef IQ36_GROUPED_LIVE_INJECTION
  bool write_tensors = true;
  LiveGroupedInjector* injector = nullptr;
#endif
};

bool Matches(const CaptureState& state, const char* name) {
  if (name == nullptr || name[0] == '\0') {
    return false;
  }
  return std::any_of(
      state.filters.begin(), state.filters.end(),
      [&](const std::regex& filter) { return std::regex_match(name, filter); });
}

void WriteTensor(CaptureState& state, const ggml_tensor* tensor) {
  if (tensor == nullptr || tensor->buffer == nullptr) {
    return;
  }
  const std::size_t byte_count = ggml_nbytes(tensor);
  std::vector<std::uint8_t> bytes(byte_count);
  ggml_backend_tensor_get(tensor, bytes.data(), 0, byte_count);
  const std::string name = tensor->name[0] ? tensor->name : "unnamed";
  const std::string file_name = SafeName(name) + "__tok" +
                                std::to_string(state.token_position) + "__ord" +
                                std::to_string(state.captured) + ".bin";
  const auto relative_path = std::filesystem::path("payloads") / file_name;
  std::ofstream payload(state.out_dir / relative_path, std::ios::binary);
  if (!payload) {
    throw std::runtime_error("failed to create tensor payload: " + file_name);
  }
  payload.write(reinterpret_cast<const char*>(bytes.data()),
                static_cast<std::streamsize>(bytes.size()));

  state.tensor_jsonl << "{"
                     << "\"case_id\":\"" << JsonEscape(state.case_id) << "\","
                     << "\"token_position\":" << state.token_position << ","
                     << "\"tensor_name\":\"" << JsonEscape(name) << "\","
                     << "\"tensor_type\":\"" << ggml_type_name(tensor->type)
                     << "\","
                     << "\"tensor_op\":\"" << ggml_op_desc(tensor) << "\","
                     << "\"ne\":[" << tensor->ne[0] << "," << tensor->ne[1]
                     << "," << tensor->ne[2] << "," << tensor->ne[3] << "],"
                     << "\"nb\":[" << tensor->nb[0] << "," << tensor->nb[1]
                     << "," << tensor->nb[2] << "," << tensor->nb[3] << "],"
                     << "\"nbytes\":" << byte_count << ","
                     << "\"payload_path\":\""
                     << JsonEscape(relative_path.generic_string()) << "\"}"
                     << "\n";
  ++state.captured;
}

bool EvalCallback(ggml_tensor* tensor, bool ask, void* user_data) {
  auto* state = static_cast<CaptureState*>(user_data);
  if (state == nullptr || !state->enabled) {
    return ask ? false : true;
  }
  const bool match = Matches(*state, tensor ? tensor->name : nullptr);
  if (ask) {
    return match;
  }
  if (match) {
#ifdef IQ36_GROUPED_LIVE_INJECTION
    if (state->injector != nullptr) state->injector->Observe(tensor);
    if (state->write_tensors) WriteTensor(*state, tensor);
#else
    WriteTensor(*state, tensor);
#endif
  }
  return true;
}

std::vector<std::pair<int, float>> TopK(const float* logits, int vocabulary,
                                        int count) {
  std::vector<std::pair<int, float>> values;
  values.reserve(static_cast<std::size_t>(vocabulary));
  for (int token = 0; token < vocabulary; ++token) {
    values.emplace_back(token, logits[token]);
  }
  const int selected = std::min(count, vocabulary);
  std::partial_sort(
      values.begin(), values.begin() + selected, values.end(),
      [](const auto& left, const auto& right) { return left.second > right.second; });
  values.resize(static_cast<std::size_t>(selected));
  return values;
}

}  // namespace

int main(int argc, char** argv) {
  std::setlocale(LC_NUMERIC, "C");
  try {
    const Args args = ParseArgs(argc, argv);
    const auto all_tokens = ReadTokenIds(
        args.token_ids_file, args.binary_u32_token_file);
    if (static_cast<std::size_t>(args.token_offset) >= all_tokens.size()) {
      throw std::runtime_error("token offset is outside token-id file");
    }
    const std::size_t available = all_tokens.size() - args.token_offset;
    const std::size_t selected = args.token_count == 0
        ? available
        : std::min<std::size_t>(args.token_count, available);
    const std::vector<llama_token> tokens(
        all_tokens.begin() + args.token_offset,
        all_tokens.begin() + args.token_offset + selected);
#ifdef IQ36_GROUPED_LIVE_INJECTION
    if (InjectionEnabled(args) &&
        (!args.batch_all || tokens.size() != kInjectionTokenCount)) {
      throw std::runtime_error(
          "live injection requires one batch of exactly 1024 tokens");
    }
#endif
    if (static_cast<int>(tokens.size()) >= args.n_ctx) {
      throw std::runtime_error("token count must be smaller than n-ctx");
    }
    std::filesystem::create_directories(
        std::filesystem::path(args.out_dir) / "payloads");

    CaptureState state;
    state.out_dir = args.out_dir;
    state.case_id = args.case_id;
    state.token_position = args.token_offset + static_cast<int>(tokens.size()) - 1;
#ifdef IQ36_GROUPED_LIVE_INJECTION
    state.write_tensors = !args.no_tensor_dumps;
#endif
    state.tensor_jsonl.open(state.out_dir / "tensor-dumps.jsonl");
    if (!state.tensor_jsonl) {
      throw std::runtime_error("failed to create tensor-dumps.jsonl");
    }
    const std::vector<std::string> full_filters = {
        R"(^model\.input_embed$)",
        R"(^attn_norm-[0-9]+$)",
        R"(^linear_attn_qkv_mixed-[0-9]+$)",
        R"(^Vcur-[0-9]+$)",
        R"(^attn_residual-[0-9]+$)",
        R"(^attn_post_norm-[0-9]+$)",
        R"(^ffn_moe_logits-[0-9]+$)",
        R"(^ffn_moe_topk-[0-9]+$)",
        R"(^ffn_moe_weights_norm-[0-9]+$)",
        R"(^ffn_moe_swiglu-[0-9]+$)",
        R"(^ffn_moe_down-[0-9]+$)",
        R"(^ffn_swiglu-[0-9]+$)",
        R"(^ffn_down-[0-9]+$)",
        R"(^ffn_shexp-[0-9]+$)",
        R"(^shared_expert_gate-[0-9]+$)",
        R"(^shared_expert_gate_sigmoid-[0-9]+$)",
        R"(^ffn_shexp_gated-[0-9]+$)",
        R"(^ffn_out-[0-9]+$)",
        R"(^post_moe-[0-9]+$)",
        R"(^l_out-[0-9]+$)",
        R"(^result_norm$)",
        R"(^result_output$)",
    };
    const std::vector<std::string> router_filters = {
        R"(^ffn_moe_topk-[0-9]+$)",
    };
    std::vector<std::string> component_filters;
    if (args.component_layer >= 0) {
      const std::string layer = std::to_string(args.component_layer);
      component_filters = {
          "^attn_post_norm-" + layer + "$",
          "^ffn_moe_topk-" + layer + "$",
          "^ffn_moe_swiglu-" + layer + "$",
      };
      if (args.component_through_down || args.component_through_ffn_out) {
        component_filters.push_back("^ffn_moe_weights_norm-" + layer + "$");
        component_filters.push_back("^ffn_moe_down-" + layer + "$");
        component_filters.push_back("^ffn_moe_out-" + layer + "$");
      }
      if (args.component_through_ffn_out) {
        component_filters.push_back("^ffn_shexp-" + layer + "$");
        component_filters.push_back("^shared_expert_gate-" + layer + "$");
        component_filters.push_back(
            "^shared_expert_gate_sigmoid-" + layer + "$");
        component_filters.push_back("^ffn_shexp_gated-" + layer + "$");
        component_filters.push_back("^ffn_out-" + layer + "$");
      }
    }
    if (args.component_all_layers) {
      component_filters = {
          R"(^attn_post_norm-[0-9]+$)",
          R"(^ffn_moe_topk-[0-9]+$)",
          R"(^ffn_moe_swiglu-[0-9]+$)",
      };
      if (args.component_through_down || args.component_through_ffn_out) {
        component_filters.push_back(R"(^ffn_moe_weights_norm-[0-9]+$)");
        component_filters.push_back(R"(^ffn_moe_down-[0-9]+$)");
        component_filters.push_back(R"(^ffn_moe_out-[0-9]+$)");
      }
      if (args.component_through_ffn_out) {
        component_filters.push_back(R"(^ffn_shexp-[0-9]+$)");
        component_filters.push_back(R"(^shared_expert_gate-[0-9]+$)");
        component_filters.push_back(
            R"(^shared_expert_gate_sigmoid-[0-9]+$)");
        component_filters.push_back(R"(^ffn_shexp_gated-[0-9]+$)");
        component_filters.push_back(R"(^ffn_out-[0-9]+$)");
      }
    }
    std::vector<std::string> linear_component_filters;
    if (args.linear_component_layer >= 0) {
      const std::string layer = std::to_string(args.linear_component_layer);
      linear_component_filters = {
          "^attn_norm-" + layer + "$",
          "^linear_attn_qkv_mixed-" + layer + "$",
          "^z-" + layer + "$",
          "^linear_attn_mixed_ba-" + layer + "$",
          "^beta-" + layer + "$",
          "^beta_sigmoid-" + layer + "$",
          "^alpha-" + layer + "$",
          "^a_softplus-" + layer + "$",
          "^gate-" + layer + "$",
          "^conv_states_reshaped-" + layer + "$",
          "^conv_output_raw-" + layer + "$",
          "^q_conv_predelta-" + layer + "$",
          "^k_conv_predelta-" + layer + "$",
          "^v_conv_predelta-" + layer + "$",
          "^state_predelta-" + layer + "$",
          "^attn_output-" + layer + "$",
          "^new_state-" + layer + "$",
          "^output_state-" + layer + "$",
          "^final_output-" + layer + "$",
          "^linear_attn_out-" + layer + "$",
      };
    }
    std::vector<std::string> selected_filters =
        args.linear_component_layer >= 0
        ? linear_component_filters
        : ((args.component_layer >= 0 || args.component_all_layers)
           ? component_filters
           : (args.router_only ? router_filters : full_filters));
#ifdef IQ36_GROUPED_LIVE_INJECTION
    if (args.live_injection_boundaries) {
      selected_filters = {
          R"(^attn_post_norm-[0-9]+$)",
          R"(^ffn_moe_topk-[0-9]+$)",
          R"(^ffn_moe_weights_norm-[0-9]+$)",
          R"(^ffn_moe_out-[0-9]+$)",
      };
      if (args.inject_reference_swiglu) {
        selected_filters.push_back(R"(^ffn_moe_swiglu-[0-9]+$)");
      }
      if (args.inject_reference_down) {
        selected_filters.push_back(R"(^ffn_moe_down-[0-9]+$)");
      }
    }
#endif
    for (const auto& filter : selected_filters) {
      state.filters.emplace_back(filter, std::regex::optimize);
    }

#ifdef IQ36_GROUPED_LIVE_INJECTION
    std::unique_ptr<LiveGroupedInjector> injector;
    if (InjectionEnabled(args)) {
      injector = std::make_unique<LiveGroupedInjector>(args);
      state.injector = injector.get();
    }
#endif

    ggml_backend_load_all();
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = args.n_gpu_layers;
    llama_model* model = llama_model_load_from_file(args.model.c_str(), model_params);
    if (model == nullptr) {
      throw std::runtime_error("failed to load model");
    }
    const llama_vocab* vocabulary = llama_model_get_vocab(model);
    const int vocabulary_size = llama_vocab_n_tokens(vocabulary);
    for (const llama_token token : tokens) {
      if (token < 0 || token >= vocabulary_size) {
        llama_model_free(model);
        throw std::runtime_error("token id outside model vocabulary");
      }
    }
#ifdef IQ36_GROUPED_LIVE_INJECTION
    for (const int token : args.generate_teacher_tokens) {
      if (token >= vocabulary_size) {
        llama_model_free(model);
        throw std::runtime_error(
            "generate teacher token is outside model vocabulary");
      }
    }
#endif

    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = static_cast<std::uint32_t>(args.n_ctx);
    context_params.n_batch = args.batch_all
        ? static_cast<std::uint32_t>(tokens.size()) : 1;
    context_params.n_ubatch = args.batch_all
        ? static_cast<std::uint32_t>(tokens.size()) : 1;
    context_params.n_seq_max = 1;
    context_params.n_threads = args.threads;
    context_params.n_threads_batch = args.threads;
    context_params.no_perf = true;
    context_params.cb_eval = EvalCallback;
    context_params.cb_eval_user_data = &state;
    llama_context* context = llama_init_from_model(model, context_params);
    if (context == nullptr) {
      llama_model_free(model);
      throw std::runtime_error("failed to create context");
    }

    llama_batch batch = llama_batch_init(
        args.batch_all ? static_cast<int>(tokens.size()) : 1, 0, 1);
    if (args.batch_all) {
      state.enabled = true;
      batch.n_tokens = static_cast<int>(tokens.size());
      for (int index = 0; index < batch.n_tokens; ++index) {
        batch.token[index] = tokens[static_cast<std::size_t>(index)];
        batch.pos[index] = args.token_offset + index;
        batch.n_seq_id[index] = 1;
        batch.seq_id[index][0] = 0;
        batch.logits[index] = index + 1 == batch.n_tokens ? 1 : 0;
      }
      const int return_code = llama_decode(context, batch);
      if (return_code != 0) {
        throw std::runtime_error(
            "batched llama_decode failed rc=" + std::to_string(return_code));
      }
    } else {
      for (int position = 0; position < static_cast<int>(tokens.size()); ++position) {
        state.enabled = position + 1 == static_cast<int>(tokens.size());
        batch.n_tokens = 1;
        batch.token[0] = tokens[static_cast<std::size_t>(position)];
        batch.pos[0] = args.token_offset + position;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = state.enabled ? 1 : 0;
        const int return_code = llama_decode(context, batch);
        if (return_code != 0) {
          throw std::runtime_error("llama_decode failed at position " +
                                   std::to_string(position) + " rc=" +
                                   std::to_string(return_code));
        }
      }
    }
    llama_synchronize(context);
    const float* logits = llama_get_logits_ith(context, -1);
    if (logits == nullptr) {
      throw std::runtime_error("final logits are unavailable");
    }
#ifdef IQ36_GROUPED_LIVE_INJECTION
    if (args.full_logits) {
      std::ofstream full_logits(
          state.out_dir / "full-logits.f32.bin", std::ios::binary);
      if (!full_logits) {
        throw std::runtime_error("failed to create full-logits.f32.bin");
      }
      full_logits.write(
          reinterpret_cast<const char*>(logits),
          static_cast<std::streamsize>(
              vocabulary_size * static_cast<int>(sizeof(float))));
    }
#endif

    std::ofstream topk_file(state.out_dir / "sampler-topk.json");
    if (!topk_file) {
      throw std::runtime_error("failed to create sampler-topk.json");
    }
    const auto topk = TopK(logits, vocabulary_size, args.top_k);
    topk_file << "{\"case_id\":\"" << JsonEscape(args.case_id)
              << "\",\"token_count\":" << tokens.size()
              << ",\"predicts_generated_position\":"
              << args.predicts_generated_position << ",\"top_k\":[";
    for (std::size_t index = 0; index < topk.size(); ++index) {
      if (index != 0) {
        topk_file << ",";
      }
      topk_file << "{\"token_id\":" << topk[index].first
                << ",\"logit\":" << topk[index].second << "}";
    }
    topk_file << "],\"watched\":[";
    for (std::size_t index = 0; index < args.watch_tokens.size(); ++index) {
      if (index != 0) {
        topk_file << ",";
      }
      const int token = args.watch_tokens[index];
      if (token < 0 || token >= vocabulary_size) {
        throw std::runtime_error("watched token is outside model vocabulary");
      }
      topk_file << "{\"token_id\":" << token << ",\"logit\":"
                << logits[token] << "}";
    }
    topk_file << "]}\n";

#ifdef IQ36_GROUPED_LIVE_INJECTION
    std::vector<llama_token> generated_tokens;
    std::vector<llama_token> fed_tokens;
    if (args.generate_count > 0) {
      generated_tokens.reserve(static_cast<std::size_t>(args.generate_count));
      fed_tokens.reserve(static_cast<std::size_t>(args.generate_count));
      state.enabled = false;
      state.injector = nullptr;
      const float* step_logits = logits;
      for (int step = 0; step < args.generate_count; ++step) {
        if (!args.generate_teacher_tokens.empty()) {
          std::ostringstream name;
          name << "continuation-logits-" << std::setfill('0')
               << std::setw(3) << step << ".f32.bin";
          std::ofstream step_logits_file(
              state.out_dir / name.str(), std::ios::binary);
          if (!step_logits_file) {
            throw std::runtime_error(
                "failed to create continuation full-logits file");
          }
          step_logits_file.write(
              reinterpret_cast<const char*>(step_logits),
              static_cast<std::streamsize>(
                  vocabulary_size * static_cast<int>(sizeof(float))));
        }
        const auto best = TopK(step_logits, vocabulary_size, 1);
        generated_tokens.push_back(best.front().first);
        const llama_token token = args.generate_teacher_tokens.empty()
            ? best.front().first
            : args.generate_teacher_tokens[static_cast<std::size_t>(step)];
        fed_tokens.push_back(token);
        batch.n_tokens = 1;
        batch.token[0] = token;
        batch.pos[0] = args.token_offset +
            static_cast<int>(tokens.size()) + step;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = 1;
        const int return_code = llama_decode(context, batch);
        if (return_code != 0) {
          throw std::runtime_error(
              "greedy continuation failed at step " +
              std::to_string(step) + " rc=" +
              std::to_string(return_code));
        }
        llama_synchronize(context);
        step_logits = llama_get_logits_ith(context, -1);
        if (step_logits == nullptr) {
          throw std::runtime_error("continuation logits are unavailable");
        }
      }
      std::ofstream generated_file(
          state.out_dir / "generated-tokens.json");
      if (!generated_file) {
        throw std::runtime_error("failed to create generated-tokens.json");
      }
      generated_file << "{\"count\":" << generated_tokens.size()
                     << ",\"teacher_forced\":"
                     << std::boolalpha
                     << !args.generate_teacher_tokens.empty()
                     << ",\"token_ids\":[";
      for (std::size_t index = 0; index < generated_tokens.size(); ++index) {
        if (index != 0) generated_file << ",";
        generated_file << generated_tokens[index];
      }
      generated_file << "],\"fed_token_ids\":[";
      for (std::size_t index = 0; index < fed_tokens.size(); ++index) {
        if (index != 0) generated_file << ",";
        generated_file << fed_tokens[index];
      }
      generated_file << "]}\n";
    }
#endif

    std::ofstream summary(state.out_dir / "capture-summary.json");
    if (!summary) {
      throw std::runtime_error("failed to create capture-summary.json");
    }
    summary << "{\"case_id\":\"" << JsonEscape(args.case_id)
            << "\",\"token_count\":" << tokens.size()
            << ",\"capture_token_position\":" << state.token_position
            << ",\"predicts_generated_position\":"
            << args.predicts_generated_position
            << ",\"captured_tensor_count\":" << state.captured
            << ",\"batch_all\":" << (args.batch_all ? "true" : "false")
            << ",\"router_only\":" << (args.router_only ? "true" : "false")
            << ",\"component_layer\":" << args.component_layer
            << ",\"linear_component_layer\":"
            << args.linear_component_layer
            << ",\"component_all_layers\":"
            << (args.component_all_layers ? "true" : "false")
            << ",\"component_through_down\":"
            << (args.component_through_down ? "true" : "false")
            << ",\"token_offset\":" << args.token_offset
            << ",\"n_ctx\":" << args.n_ctx
            << ",\"n_gpu_layers\":" << args.n_gpu_layers << "}\n";

#ifdef IQ36_GROUPED_LIVE_INJECTION
    bool injection_pass = true;
    if (injector != nullptr) {
      injection_pass = injector->WriteSummary(
          state.out_dir / "live-injection-summary.json");
    }
#endif

    llama_batch_free(batch);
    llama_free(context);
    llama_model_free(model);
    std::fprintf(stdout, "captured_tensor_count=%d\n", state.captured);
    std::fprintf(stdout, "token_count=%zu\n", tokens.size());
    std::fprintf(stdout, "top1_token=%d\n", topk.front().first);
#ifdef IQ36_GROUPED_LIVE_INJECTION
    if (!injection_pass) return 2;
#endif
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "error: %s\n", error.what());
    return 1;
  }
}
