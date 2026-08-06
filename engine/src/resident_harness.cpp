#include "intel_qwen36/resident_harness.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <utility>

namespace iq36 {

const ModelContract& model_contract() {
  static const ModelContract contract{
      "intel-qwen36-35b-a3b-gguf-q4km",
      "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf",
      "d42becf903ed7093d438c0d9f44afc136756b6b8f766121e9066fb888a2dc36e",
      40,
      2048,
      256,
      8,
      262144};
  return contract;
}

const TargetContract& target_contract() {
  static const TargetContract contract{
      "local",
      "Intel(R) Core(TM) Ultra X7 358H",
      "Intel(R) Arc(TM) B390 GPU"};
  return contract;
}

std::vector<std::string> boundary_types() {
  return {
      "embedding",
      "layer_input_rmsnorm",
      "qkv_projection",
      "rope",
      "attention",
      "attention_output_projection",
      "post_attention_residual",
      "ffn_rmsnorm",
      "router_topk",
      "selected_expert_gate_up",
      "swiglu",
      "selected_expert_down",
      "shared_expert",
      "moe_residual",
      "final_norm",
      "lm_head",
      "sampler"};
}

std::vector<std::string> required_oracle_bundle_paths() {
  return {
      "manifest.json",
      "correctness.json",
      "token-topk-references.jsonl",
      "teacher-forced-distribution-references.jsonl",
      "boundary-references/inputs.jsonl",
      "boundary-references/outputs.jsonl"};
}

namespace {

std::size_t count_nonempty_lines(const std::filesystem::path& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::invalid_argument("oracle bundle file could not be opened");
  }
  std::size_t count = 0;
  std::string line;
  while (std::getline(input, line)) {
    if (!line.empty()) {
      ++count;
    }
  }
  return count;
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

void write_uint32_array(std::ostream& output,
                        const std::vector<std::uint32_t>& values) {
  output << "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0) {
      output << ",";
    }
    output << values[i];
  }
  output << "]";
}

void write_topk(std::ostream& output,
                const std::vector<ResidentTopKRow>& rows) {
  output << "[";
  for (std::size_t i = 0; i < rows.size(); ++i) {
    if (i != 0) {
      output << ",";
    }
    output << "{\"logit\":" << rows[i].logit
           << ",\"token_id\":" << rows[i].token_id << "}";
  }
  output << "]";
}

void write_topk_ids(std::ostream& output,
                    const std::vector<ResidentTopKRow>& rows) {
  output << "[";
  for (std::size_t i = 0; i < rows.size(); ++i) {
    if (i != 0) {
      output << ",";
    }
    output << rows[i].token_id;
  }
  output << "]";
}

void write_oracle_bundle_stats(std::ostream& output,
                               const OracleBundleStats& stats) {
  output << "{";
  output << "\"boundary_input_rows\":" << stats.boundary_input_rows << ",";
  output << "\"boundary_output_rows\":" << stats.boundary_output_rows << ",";
  output << "\"teacher_forced_distribution_rows\":"
         << stats.teacher_forced_distribution_rows << ",";
  output << "\"token_topk_rows\":" << stats.token_topk_rows;
  output << "}";
}

void write_case_result(std::ostream& output,
                       const ResidentCaseResult& result) {
  output << "{";
  output << "\"case_id\":\"" << json_escape(result.case_id) << "\",";
  output << "\"first_token_top_k\":";
  write_topk(output, result.first_topk);
  output << ",\"first_token_top_logprob_id_signature\":";
  write_topk_ids(output, result.first_topk);
  output << ",\"generated_token_ids\":";
  write_uint32_array(output, result.generated_token_ids);
  output << ",\"prompt_token_count\":" << result.prompt_token_ids.size() << ",";
  output << "\"prompt_token_ids\":";
  write_uint32_array(output, result.prompt_token_ids);
  output << ",\"timing_ns\":{";
  output << "\"case_total\":" << result.case_total_ns << ",";
  output << "\"decode_continuation\":" << result.decode_continuation_ns << ",";
  output << "\"prompt_prefill\":" << result.prompt_prefill_ns;
  output << "}";
  output << "}";
}

void write_case_results(std::ostream& output,
                        const std::vector<ResidentCaseResult>& cases) {
  output << "[";
  for (std::size_t i = 0; i < cases.size(); ++i) {
    if (i != 0) {
      output << ",";
    }
    write_case_result(output, cases[i]);
  }
  output << "]";
}

std::uint64_t generated_token_count(
    const std::vector<ResidentCaseResult>& cases) {
  std::uint64_t count = 0;
  for (const auto& item : cases) {
    count += static_cast<std::uint64_t>(item.generated_token_ids.size());
  }
  return count;
}

}  // namespace

void ResidentHarness::load(std::string model_path,
                           std::string oracle_bundle_path) {
  if (model_path.empty()) {
    throw std::invalid_argument("model_path is required");
  }
  if (model_path != model_contract().model_path) {
    throw std::invalid_argument("model_path must match locked model contract");
  }
  if (oracle_bundle_path.empty()) {
    throw std::invalid_argument("oracle_bundle_path is required");
  }
  if (oracle_bundle_path.find("placeholder") != std::string::npos) {
    throw std::invalid_argument("oracle_bundle_path must not be a placeholder");
  }
  const auto bundle_path = std::filesystem::path(oracle_bundle_path);
  if (!std::filesystem::exists(bundle_path)) {
    throw std::invalid_argument("oracle_bundle_path must exist");
  }
  if (!std::filesystem::is_directory(bundle_path)) {
    throw std::invalid_argument("oracle_bundle_path must be a directory");
  }
  for (const auto& relative_path : required_oracle_bundle_paths()) {
    const auto required_path = bundle_path / relative_path;
    if (!std::filesystem::is_regular_file(required_path)) {
      throw std::invalid_argument("oracle_bundle_path missing required bundle file");
    }
  }
  OracleBundleStats stats{
      count_nonempty_lines(bundle_path / "token-topk-references.jsonl"),
      count_nonempty_lines(bundle_path / "teacher-forced-distribution-references.jsonl"),
      count_nonempty_lines(bundle_path / "boundary-references/inputs.jsonl"),
      count_nonempty_lines(bundle_path / "boundary-references/outputs.jsonl")};
  if (stats.token_topk_rows == 0 ||
      stats.teacher_forced_distribution_rows == 0 ||
      stats.boundary_input_rows == 0 ||
      stats.boundary_output_rows == 0) {
    throw std::invalid_argument("oracle_bundle_path contains empty reference files");
  }
  oracle_bundle_stats_ = stats;
  model_path_ = std::move(model_path);
  oracle_bundle_path_ = std::move(oracle_bundle_path);
  loaded_ = true;
}

void ResidentHarness::swap_kernel(std::string boundary_id,
                                  std::string implementation_id) {
  if (!loaded_) {
    throw std::logic_error("load must run before swap_kernel");
  }
  if (boundary_id.empty() || implementation_id.empty()) {
    throw std::invalid_argument("boundary_id and implementation_id required");
  }
  last_boundary_id_ = std::move(boundary_id);
  last_implementation_id_ = std::move(implementation_id);
}

BoundaryResult ResidentHarness::run_boundary(
    std::string_view boundary_id) const {
  if (!loaded_) {
    throw std::logic_error("load must run before run_boundary");
  }
  if (boundary_id.empty()) {
    throw std::invalid_argument("boundary_id is required");
  }
  // STUB: a real teacher-forced comparison requires the target adapter (load the
  // 35B model + oracle resident on PTL, run only this one box, compare its output
  // to the oracle reference). That is not wired on this host. Returning a "passing"
  // cosine=1.0 here would be a fake pass — exactly the wrong-but-fast failure the
  // methodology's cheap-judge gate exists to catch (ch.3 §3.3). Return a
  // not-evaluated sentinel (NaN) instead: NaN can never satisfy a `cosine >= 0.999`
  // gate, so nothing downstream can mistake the unimplemented stub for a real pass.
  // Replace this body with the adapter call; see the resident-harness design.
  const double kNotEvaluated = std::numeric_limits<double>::quiet_NaN();
  return BoundaryResult{
      std::string(boundary_id),
      kNotEvaluated,   // cosine
      kNotEvaluated,   // relative_l2
      kNotEvaluated,   // kl_divergence
      kNotEvaluated,   // top1
      0.0,             // elapsed_us
      false};          // promoted
}

bool ResidentHarness::promote(std::string_view boundary_id) const {
  if (!loaded_) {
    throw std::logic_error("load must run before promote");
  }
  if (boundary_id.empty()) {
    throw std::invalid_argument("boundary_id is required");
  }
  return false;
}

bool ResidentHarness::loaded() const {
  return loaded_;
}

const OracleBundleStats& ResidentHarness::oracle_bundle_stats() const {
  return oracle_bundle_stats_;
}

void ResidentHarness::begin_streaming_session(
    const ResidentStreamingSessionConfig& config) {
  if (streaming_session_.active) {
    throw std::logic_error("resident streaming session is already active");
  }
  if (config.session_id.empty()) {
    throw std::invalid_argument("resident streaming session_id is required");
  }
  if (config.max_new_tokens == 0) {
    throw std::invalid_argument("resident streaming max_new_tokens is required");
  }
  if (config.expected_case_count == 0) {
    throw std::invalid_argument("resident streaming expected_case_count is required");
  }
  streaming_session_.active = true;
  streaming_session_.session_id = config.session_id;
  streaming_session_.run_index = config.run_index;
  streaming_session_.max_new_tokens = config.max_new_tokens;
  streaming_session_.expected_case_count = config.expected_case_count;
  streaming_session_.emitted_token_count = 0;
}

bool ResidentHarness::streaming_session_active() const {
  return streaming_session_.active;
}

void ResidentHarness::emit_sse_token_event(
    std::ostream& output,
    const ResidentTokenEvent& event) const {
  if (event.topk.empty()) {
    throw std::invalid_argument("resident token event top-k is empty");
  }
  output << "event: token\n";
  output << "data: {";
  output << "\"case_id\":\"" << json_escape(event.case_id) << "\",";
  output << "\"elapsed_ns\":" << event.elapsed_ns << ",";
  output << "\"generated_index\":" << event.generated_index << ",";
  output << "\"phase\":\"" << json_escape(event.phase) << "\",";
  output << "\"predicted_token_position\":"
         << event.predicted_token_position << ",";
  output << "\"resident_event_api\":\"ResidentHarness\",";
  output << "\"resident_session_event_index\":"
         << event.resident_session_event_index << ",";
  output << "\"resident_session_id\":\""
         << json_escape(event.resident_session_id) << "\",";
  output << "\"run_index\":" << event.run_index << ",";
  output << "\"top_logprob_id_signature\":";
  write_topk_ids(output, event.topk);
  output << ",\"token_id\":" << event.topk[0].token_id;
  output << "}\n\n" << std::flush;
}

void ResidentHarness::emit_sse_session_token_event(
    std::ostream& output,
    ResidentTokenEvent event) {
  if (!streaming_session_.active) {
    throw std::logic_error("resident streaming session is not active");
  }
  if (event.run_index != streaming_session_.run_index) {
    throw std::invalid_argument("resident token event run_index mismatch");
  }
  if (event.generated_index >= streaming_session_.max_new_tokens) {
    throw std::invalid_argument("resident token event generated_index exceeds max");
  }
  event.resident_session_id = streaming_session_.session_id;
  event.resident_session_event_index =
      streaming_session_.emitted_token_count;
  emit_sse_token_event(output, event);
  ++streaming_session_.emitted_token_count;
}

void ResidentHarness::emit_sse_done_event(
    std::ostream& output,
    const ResidentDoneEvent& event) const {
  output << "event: done\n";
  output << "data: {";
  output << "\"cases\":";
  write_case_results(output, event.cases);
  output << ",\"dense_q6_pair_dot_enabled\":"
         << (event.dense_q6_pair_dot_enabled ? "true" : "false") << ",";
  output << "\"emitted_case_count\":" << event.cases.size() << ",";
  output << "\"max_new_tokens\":" << event.max_new_tokens << ",";
  output << "\"process_total_ns\":" << event.process_total_ns << ",";
  output << "\"q4_plane_layout_enabled\":"
         << (event.q4_plane_layout_enabled ? "true" : "false") << ",";
  output << "\"resident_event_api\":\"ResidentHarness\",";
  output << "\"resident_harness_load_ns\":"
         << event.resident_harness_load_ns << ",";
  output << "\"resident_harness_loaded\":"
         << (loaded_ ? "true" : "false") << ",";
  output << "\"resident_harness_oracle_bundle_stats\":";
  write_oracle_bundle_stats(output, oracle_bundle_stats_);
  output << ",";
  output << "\"resident_session_id\":\""
         << json_escape(event.resident_session_id) << "\",";
  output << "\"resident_session_token_count\":"
         << event.resident_session_token_count << ",";
  output << "\"schema_version\":\"intel-qwen36-native-sse-events-v0\",";
  output << "\"selected_expert_down_q6_pair_dot_enabled\":"
         << (event.selected_expert_down_q6_pair_dot_enabled ? "true" : "false");
  output << "}\n\n" << std::flush;
}

void ResidentHarness::emit_sse_session_done_event(
    std::ostream& output,
    ResidentDoneEvent event) {
  if (!streaming_session_.active) {
    throw std::logic_error("resident streaming session is not active");
  }
  if (event.max_new_tokens != streaming_session_.max_new_tokens) {
    throw std::invalid_argument("resident done event max_new_tokens mismatch");
  }
  if (event.cases.size() != streaming_session_.expected_case_count) {
    throw std::invalid_argument("resident done event case count mismatch");
  }
  const auto generated_count = generated_token_count(event.cases);
  if (generated_count != streaming_session_.emitted_token_count) {
    throw std::invalid_argument("resident done event token count mismatch");
  }
  event.resident_session_id = streaming_session_.session_id;
  event.resident_session_token_count =
      streaming_session_.emitted_token_count;
  emit_sse_done_event(output, event);
  streaming_session_ = StreamingSessionState{};
}

ResidentDecodeLoop::ResidentDecodeLoop(ResidentHarness& harness)
    : harness_(harness) {}

ResidentGpuHotDecodeLoop::ResidentGpuHotDecodeLoop(ResidentHarness& harness)
    : harness_(harness) {}

ResidentDecodeLoopResult ResidentDecodeLoop::run(
    std::ostream& output,
    const ResidentDecodeLoopConfig& config,
    const ResidentDecodeTokenCallback& token_cb,
    const ResidentDecodeDoneCallback& done_cb) {
  if (config.session_id_prefix.empty()) {
    throw std::invalid_argument("resident decode session_id_prefix is required");
  }
  if (config.max_new_tokens == 0) {
    throw std::invalid_argument("resident decode max_new_tokens is required");
  }
  if (config.session_count == 0) {
    throw std::invalid_argument("resident decode session_count is required");
  }
  if (config.expected_case_count == 0) {
    throw std::invalid_argument("resident decode expected_case_count is required");
  }
  if (!config.teacher_forced_token_ids.empty() &&
      config.teacher_forced_token_ids.size() <
          static_cast<std::size_t>(config.max_new_tokens)) {
    throw std::invalid_argument(
        "resident decode teacher-forced token list is shorter than max_new_tokens");
  }
  if (!token_cb) {
    throw std::invalid_argument("resident decode token callback is required");
  }
  if (!done_cb) {
    throw std::invalid_argument("resident decode done callback is required");
  }

  ResidentDecodeLoopResult result;
  result.session_count = config.session_count;
  const auto process_begin = std::chrono::steady_clock::now();
  for (std::size_t session_index = 0;
       session_index < config.session_count;
       ++session_index) {
    const int run_index =
        config.run_index_base + static_cast<int>(session_index);
    const std::string session_id =
        config.session_id_prefix + "-" + std::to_string(session_index);
    if (config.emit_sse_events) {
      ResidentStreamingSessionConfig session_config;
      session_config.session_id = session_id;
      session_config.run_index = run_index;
      session_config.max_new_tokens = config.max_new_tokens;
      session_config.expected_case_count = config.expected_case_count;
      harness_.begin_streaming_session(session_config);
    }
    const auto session_begin = std::chrono::steady_clock::now();
    std::uint32_t input_token_id = config.initial_input_token_id;
    std::uint64_t top1_match_count = 0;
    std::uint64_t topk_match_count = 0;
    std::vector<std::uint32_t> input_token_ids;
    std::vector<std::uint32_t> generated_token_ids;
    input_token_ids.reserve(static_cast<std::size_t>(config.max_new_tokens));
    generated_token_ids.reserve(static_cast<std::size_t>(config.max_new_tokens));
    for (std::uint64_t token_index = 0;
         token_index < config.max_new_tokens;
         ++token_index) {
      ResidentDecodeStepContext step;
      step.session_index = session_index;
      step.token_index = static_cast<std::size_t>(token_index);
      step.run_index = run_index;
      step.input_token_id = input_token_id;
      step.max_new_tokens = config.max_new_tokens;
      ResidentDecodeTokenResult token_result = token_cb(step);
      ResidentTokenEvent event = std::move(token_result.event);
      if (event.topk.empty()) {
        throw std::invalid_argument("resident decode token event top-k is empty");
      }
      const auto generated_token_id = event.topk[0].token_id;
      if (generated_token_id < 0) {
        throw std::invalid_argument(
            "resident decode generated token id is negative");
      }
      input_token_ids.push_back(input_token_id);
      generated_token_ids.push_back(
          static_cast<std::uint32_t>(generated_token_id));
      if (token_result.top1_matches) {
        ++top1_match_count;
      }
      if (token_result.topk_matches) {
        ++topk_match_count;
      }
      event.run_index = run_index;
      event.generated_index = static_cast<std::size_t>(token_index);
      if (config.emit_sse_events) {
        harness_.emit_sse_session_token_event(output, event);
      }
      ++result.emitted_token_count;
      input_token_id =
          config.teacher_forced_token_ids.empty()
              ? static_cast<std::uint32_t>(generated_token_id)
              : config.teacher_forced_token_ids[static_cast<std::size_t>(
                    token_index)];
    }
    const auto session_end = std::chrono::steady_clock::now();
    ResidentDecodeSessionContext done_context;
    done_context.session_index = session_index;
    done_context.run_index = run_index;
    done_context.session_id = session_id;
    done_context.max_new_tokens = config.max_new_tokens;
    done_context.emitted_token_count = config.max_new_tokens;
    done_context.top1_match_count = top1_match_count;
    done_context.topk_match_count = topk_match_count;
    done_context.input_token_ids = std::move(input_token_ids);
    done_context.generated_token_ids = std::move(generated_token_ids);
    done_context.session_elapsed_ns =
        static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                session_end - session_begin).count());
    ResidentDoneEvent done = done_cb(done_context);
    done.max_new_tokens = config.max_new_tokens;
    done.process_total_ns = done_context.session_elapsed_ns;
    if (config.emit_sse_events) {
      harness_.emit_sse_session_done_event(output, std::move(done));
    }
    result.input_token_count += done_context.emitted_token_count;
    result.generated_token_count += done_context.emitted_token_count;
  }
  const auto process_end = std::chrono::steady_clock::now();
  result.process_total_ns =
      static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              process_end - process_begin).count());
  return result;
}

}  // namespace iq36
