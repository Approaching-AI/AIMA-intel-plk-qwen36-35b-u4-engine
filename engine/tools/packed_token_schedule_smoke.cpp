#include "intel_qwen36/gguf_loader.hpp"
#include "intel_qwen36/packed_token_schedule.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

class ContractBackend final : public iq36::PackedTokenBackend {
 public:
  void Compile(const iq36::PackedTokenProgram& program) override {
    if (compile_count != 0) {
      throw std::runtime_error("packed token backend compiled twice");
    }
    ++compile_count;
    compiled_command_count = program.commands.size();
    compiled_stream_bytes = program.strict_stream_bytes_per_token;
  }

  std::vector<iq36::PackedTokenTopKRow> SubmitToken(
      const iq36::PackedTokenSubmission& submission) override {
    if (compile_count != 1) {
      throw std::runtime_error("packed token backend was not compiled");
    }
    ++submit_count;
    last_token_id = submission.token_id;
    last_token_position = submission.token_position;
    std::vector<iq36::PackedTokenTopKRow> rows;
    rows.reserve(submission.top_k);
    for (std::size_t i = 0; i < submission.top_k; ++i) {
      rows.push_back({static_cast<std::int32_t>(100 + i),
                      static_cast<float>(10.0 - i)});
    }
    return rows;
  }

  std::uint64_t compile_count = 0;
  std::uint64_t submit_count = 0;
  std::size_t compiled_command_count = 0;
  std::uint64_t compiled_stream_bytes = 0;
  std::uint32_t last_token_id = 0;
  std::uint64_t last_token_position = 0;
};

bool MapsAreNativeOnly() {
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

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 2) {
      throw std::invalid_argument(
          "usage: iq36-packed-token-schedule-smoke MODEL.gguf");
    }
    const auto index = iq36::parse_gguf_model_index(argv[1]);
    auto program = iq36::BuildPackedTokenProgram(index);
    const auto validation = iq36::ValidatePackedTokenProgram(index, program);

    std::map<std::string, int> stage_counts;
    std::uint64_t resident_state_write_bytes = 0;
    for (const auto& command : program.commands) {
      ++stage_counts[iq36::PackedTokenStageName(command.stage)];
      resident_state_write_bytes += command.resident_state_write_bytes;
    }

    auto backend = std::make_unique<ContractBackend>();
    auto* backend_observer = backend.get();
    iq36::PackedTokenRuntime runtime(std::move(program), std::move(backend));
    iq36::PackedTokenSubmission submission;
    submission.token_id = 42;
    submission.token_position = iq36::kPackedTokenAdmissionContextTokens;
    submission.top_k = 5;
    const auto topk = runtime.SubmitToken(submission);
    const auto stats = runtime.stats();
    const auto& compiled = runtime.program();
    const double kernel_stream_bandwidth_gb_s =
        static_cast<double>(compiled.strict_stream_bytes_per_token) / 1e6 /
        compiled.admission.kernel_schedule_ms_per_token_max;
    const bool maps_native_only = MapsAreNativeOnly();
    const bool pass =
        validation.passed && compiled.context_tokens == 1024 &&
        compiled.commands.size() == 252 &&
        compiled.linear_layer_count == 30 &&
        compiled.full_attention_layer_count == 10 &&
        compiled.covered_tensor_count == 693 &&
        compiled.active_weight_bytes_per_token == 1'975'676'544ULL &&
        compiled.kv_history_read_bytes_per_token == 20'971'520ULL &&
        compiled.resident_state_read_bytes_per_token == 86'835'200ULL &&
        compiled.resident_state_write_bytes_per_token == 65'884'160ULL &&
        compiled.strict_stream_bytes_per_token == 2'128'395'904ULL &&
        compiled.q4_stream_bytes_per_token == 1'116'980'352ULL &&
        compiled.q6_stream_bytes_per_token == 769'843'200ULL &&
        compiled.f32_stream_bytes_per_token == 88'852'992ULL &&
        compiled.q4_stream_bytes_per_token +
                compiled.q6_stream_bytes_per_token +
                compiled.f32_stream_bytes_per_token ==
            compiled.active_weight_bytes_per_token &&
        stage_counts["linear_preconv"] == 30 &&
        stage_counts["attention_front"] == 10 &&
        stage_counts["selected_ffn"] == 40 &&
        compiled.token_input_boundary_count == 1 &&
        compiled.topk_output_boundary_count == 1 &&
        resident_state_write_bytes == 65'884'160ULL &&
        stats.compile_count == 1 && stats.token_submission_count == 1 &&
        stats.host_input_boundary_count == 1 &&
        stats.host_output_boundary_count == 1 &&
        stats.intermediate_host_read_count == 0 &&
        backend_observer->compile_count == 1 &&
        backend_observer->submit_count == 1 &&
        backend_observer->compiled_command_count == 252 &&
        backend_observer->compiled_stream_bytes == 2'128'395'904ULL &&
        backend_observer->last_token_id == 42 &&
        backend_observer->last_token_position == 1024 &&
        topk.size() == 5 && topk.front().token_id == 100 &&
        std::isfinite(kernel_stream_bandwidth_gb_s) &&
        kernel_stream_bandwidth_gb_s >
            compiled.admission.strict_stream_bandwidth_gb_s_min &&
        maps_native_only;

    std::cout << std::boolalpha << std::setprecision(12) << "{"
              << "\"active_weight_bytes_per_token\":"
              << compiled.active_weight_bytes_per_token << ","
              << "\"attention_front_command_count\":"
              << stage_counts["attention_front"] << ","
              << "\"command_count\":" << compiled.commands.size() << ","
              << "\"compile_count\":" << stats.compile_count << ","
              << "\"context_tokens\":" << compiled.context_tokens << ","
              << "\"covered_tensor_count\":"
              << compiled.covered_tensor_count << ","
              << "\"f32_stream_bytes_per_token\":"
              << compiled.f32_stream_bytes_per_token << ","
              << "\"full_attention_layer_count\":"
              << compiled.full_attention_layer_count << ","
              << "\"host_input_boundary_count\":"
              << stats.host_input_boundary_count << ","
              << "\"host_submit_ms_per_token_max\":"
              << compiled.admission.host_submit_ms_per_token_max << ","
              << "\"host_output_boundary_count\":"
              << stats.host_output_boundary_count << ","
              << "\"intermediate_host_read_count\":"
              << stats.intermediate_host_read_count << ","
              << "\"kernel_schedule_ms_per_token_max\":"
              << compiled.admission.kernel_schedule_ms_per_token_max << ","
              << "\"kernel_stream_bandwidth_gb_s_min\":"
              << kernel_stream_bandwidth_gb_s << ","
              << "\"kv_history_read_bytes_per_token\":"
              << compiled.kv_history_read_bytes_per_token << ","
              << "\"linear_layer_count\":"
              << compiled.linear_layer_count << ","
              << "\"linear_preconv_command_count\":"
              << stage_counts["linear_preconv"] << ","
              << "\"maps_native_only\":" << maps_native_only << ","
              << "\"q4_stream_bytes_per_token\":"
              << compiled.q4_stream_bytes_per_token << ","
              << "\"q6_stream_bytes_per_token\":"
              << compiled.q6_stream_bytes_per_token << ","
              << "\"required_checks_passed\":" << pass << ","
              << "\"resident_state_write_bytes_per_token\":"
              << resident_state_write_bytes << ","
              << "\"resident_state_read_bytes_per_token\":"
              << compiled.resident_state_read_bytes_per_token << ","
              << "\"selected_ffn_command_count\":"
              << stage_counts["selected_ffn"] << ","
              << "\"strict_stream_bandwidth_gb_s_min\":"
              << compiled.admission.strict_stream_bandwidth_gb_s_min << ","
              << "\"strict_stream_bytes_per_token\":"
              << compiled.strict_stream_bytes_per_token << ","
              << "\"token_submission_count\":"
              << stats.token_submission_count << ","
              << "\"topk_count\":" << topk.size() << ","
              << "\"validation_failure_count\":"
              << validation.failed_checks.size() << ","
              << "\"wall_ms_per_token_max\":"
              << compiled.admission.wall_ms_per_token_max << "}"
              << std::endl;
    return pass ? 0 : 2;
  } catch (const std::exception& exception) {
    std::cerr << "iq36-packed-token-schedule-smoke: "
              << exception.what() << '\n';
    return 4;
  }
}
