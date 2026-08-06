#include "intel_qwen36/resident_harness.hpp"
#include "intel_qwen36/gguf_loader.hpp"

#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace {

void require(bool ok, const char* message) {
  if (!ok) {
    throw std::runtime_error(message);
  }
}

template <typename Fn>
void require_throws(Fn&& fn, const char* message) {
  try {
    fn();
  } catch (const std::invalid_argument&) {
    return;
  }
  throw std::runtime_error(message);
}

void write_file(const std::filesystem::path& path, const char* contents) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream out(path);
  out << contents;
}

void write_u32(std::ofstream& out, std::uint32_t value) {
  for (int shift = 0; shift < 32; shift += 8) {
    out.put(static_cast<char>((value >> shift) & 0xff));
  }
}

void write_u64(std::ofstream& out, std::uint64_t value) {
  for (int shift = 0; shift < 64; shift += 8) {
    out.put(static_cast<char>((value >> shift) & 0xff));
  }
}

void write_string(std::ofstream& out, const std::string& value) {
  write_u64(out, value.size());
  out.write(value.data(), static_cast<std::streamsize>(value.size()));
}

void write_synthetic_gguf(const std::filesystem::path& path) {
  std::ofstream out(path, std::ios::binary);
  out.write("GGUF", 4);
  write_u32(out, 3);
  write_u64(out, 1);
  write_u64(out, 3);

  write_string(out, "general.architecture");
  write_u32(out, 8);
  write_string(out, "qwen35moe");

  write_string(out, "general.alignment");
  write_u32(out, 4);
  write_u32(out, 32);

  write_string(out, "unit.sections");
  write_u32(out, 9);
  write_u32(out, 5);
  write_u64(out, 4);
  write_u32(out, 11);
  write_u32(out, 11);
  write_u32(out, 10);
  write_u32(out, 0);

  write_string(out, "unit.weight");
  write_u32(out, 1);
  write_u64(out, 256);
  write_u32(out, 12);
  write_u64(out, 0);
}

void write_synthetic_f32_matrix_gguf(const std::filesystem::path& path) {
  std::ofstream out(path, std::ios::binary);
  out.write("GGUF", 4);
  write_u32(out, 3);
  write_u64(out, 1);
  write_u64(out, 2);

  write_string(out, "general.architecture");
  write_u32(out, 8);
  write_string(out, "unit");

  write_string(out, "general.alignment");
  write_u32(out, 4);
  write_u32(out, 32);

  write_string(out, "unit.matrix");
  write_u32(out, 2);
  write_u64(out, 2);
  write_u64(out, 3);
  write_u32(out, 0);
  write_u64(out, 0);

  const auto end = static_cast<std::uint64_t>(out.tellp());
  const auto aligned = ((end + 31) / 32) * 32;
  for (std::uint64_t i = end; i < aligned; ++i) {
    out.put('\0');
  }
  for (const auto value : std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    write_u32(out, bits);
  }
}

void write_synthetic_f32_expert_gguf(const std::filesystem::path& path) {
  std::ofstream out(path, std::ios::binary);
  out.write("GGUF", 4);
  write_u32(out, 3);
  write_u64(out, 1);
  write_u64(out, 2);

  write_string(out, "general.architecture");
  write_u32(out, 8);
  write_string(out, "unit");

  write_string(out, "general.alignment");
  write_u32(out, 4);
  write_u32(out, 32);

  write_string(out, "unit.experts");
  write_u32(out, 3);
  write_u64(out, 2);
  write_u64(out, 3);
  write_u64(out, 2);
  write_u32(out, 0);
  write_u64(out, 0);

  const auto end = static_cast<std::uint64_t>(out.tellp());
  const auto aligned = ((end + 31) / 32) * 32;
  for (std::uint64_t i = end; i < aligned; ++i) {
    out.put('\0');
  }
  for (const auto value : std::vector<float>{
           1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
           7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f}) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    write_u32(out, bits);
  }
}

void write_f32_vector(const std::filesystem::path& path,
                      const std::vector<float>& values) {
  std::ofstream out(path, std::ios::binary);
  for (const auto value : values) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    write_u32(out, bits);
  }
}

}  // namespace

int main() {
  const auto& model = iq36::model_contract();
  require(model.workstream == "intel-qwen36-35b-a3b-gguf-q4km",
          "unexpected workstream");
  require(model.layers == 40, "unexpected layer count");
  require(model.hidden_size == 2048, "unexpected hidden size");
  require(model.experts == 256, "unexpected expert count");
  require(model.active_experts == 8, "unexpected active expert count");
  require(model.context_length == 262144, "unexpected context length");
  require(iq36::parameterized_layer_count() == 40,
          "parameterized layer count mismatch");
  require(iq36::boundary_types().size() >= 10,
          "boundary list is too small");
  require(iq36::required_oracle_bundle_paths().size() == 6,
          "unexpected oracle bundle required path count");

  const auto& target = iq36::target_contract();
  require(target.host_alias == "local",
          "unexpected target alias");

  iq36::ResidentHarness harness;
  require(!harness.loaded(), "harness should start unloaded");
  require_throws(
      [&] { harness.load(model.model_path, "oracle/qwen36-placeholder.bundle"); },
      "placeholder oracle bundle should be rejected");
  require(!harness.loaded(), "failed load must not mark harness loaded");

  const auto file_bundle_path =
      std::filesystem::temp_directory_path() / "iq36-unit-oracle-file.bundle";
  {
    std::ofstream out(file_bundle_path);
    out << "unit-test oracle marker\n";
  }
  require_throws([&] { harness.load(model.model_path, file_bundle_path.string()); },
                 "file oracle bundle should be rejected");
  std::filesystem::remove(file_bundle_path);

  const auto bundle_path =
      std::filesystem::temp_directory_path() / "iq36-unit-oracle.bundle.d";
  std::filesystem::remove_all(bundle_path);
  for (const auto& relative_path : iq36::required_oracle_bundle_paths()) {
    write_file(bundle_path / relative_path, "unit-test oracle marker\n");
  }
  harness.load(model.model_path, bundle_path.string());
  std::filesystem::remove_all(bundle_path);
  require(harness.loaded(), "harness did not enter loaded state");
  require(harness.oracle_bundle_stats().token_topk_rows == 1,
          "harness did not count token/top-k rows");
  require(harness.oracle_bundle_stats().teacher_forced_distribution_rows == 1,
          "harness did not count distribution rows");
  require(harness.oracle_bundle_stats().boundary_input_rows == 1,
          "harness did not count boundary input rows");
  require(harness.oracle_bundle_stats().boundary_output_rows == 1,
          "harness did not count boundary output rows");
  harness.swap_kernel("router_topk", "stub");
  const auto result = harness.run_boundary("router_topk");
  require(std::isnan(result.cosine),
          "stub run_boundary must return a not-evaluated sentinel (NaN), never a "
          "fake cosine=1.0 pass");
  require(!result.promoted, "stub boundary must not promote");
  require(!harness.promote("router_topk"), "stub promote must be false");

  iq36::ResidentTokenEvent token_event;
  token_event.case_id = "unit_case";
  token_event.run_index = 3;
  token_event.phase = "decode";
  token_event.generated_index = 2;
  token_event.predicted_token_position = 1026;
  token_event.topk = {
      iq36::ResidentTopKRow{271, 20.5f},
      iq36::ResidentTopKRow{198, 17.25f}};
  token_event.elapsed_ns = 123456;
  std::ostringstream token_sse;
  harness.emit_sse_token_event(token_sse, token_event);
  const auto token_sse_text = token_sse.str();
  require(token_sse_text.find("event: token\n") != std::string::npos,
          "resident token SSE event header missing");
  require(token_sse_text.find("\"resident_event_api\":\"ResidentHarness\"") !=
              std::string::npos,
          "resident token SSE API marker missing");
  require(token_sse_text.find("\"top_logprob_id_signature\":[271,198]") !=
              std::string::npos,
          "resident token SSE top-k signature mismatch");
  require(token_sse_text.find("\"token_id\":271") != std::string::npos,
          "resident token SSE token id mismatch");
  std::ostringstream bad_token_sse;
  require_throws(
      [&] {
        iq36::ResidentTokenEvent bad_event;
        bad_event.case_id = "unit_case";
        harness.emit_sse_token_event(bad_token_sse, bad_event);
      },
      "resident token SSE should reject empty top-k");

  iq36::ResidentCaseResult resident_case;
  resident_case.case_id = "unit_case";
  resident_case.prompt_token_ids = {1, 2, 3};
  resident_case.first_topk = token_event.topk;
  resident_case.generated_token_ids = {271};
  resident_case.prompt_prefill_ns = 10;
  resident_case.decode_continuation_ns = 20;
  resident_case.case_total_ns = 30;
  iq36::ResidentDoneEvent done_event;
  done_event.cases = {resident_case};
  done_event.max_new_tokens = 1;
  done_event.process_total_ns = 40;
  done_event.q4_plane_layout_enabled = true;
  done_event.selected_expert_down_q6_pair_dot_enabled = true;
  done_event.dense_q6_pair_dot_enabled = true;
  done_event.resident_harness_load_ns = 50;
  std::ostringstream done_sse;
  harness.emit_sse_done_event(done_sse, done_event);
  const auto done_sse_text = done_sse.str();
  require(done_sse_text.find("event: done\n") != std::string::npos,
          "resident done SSE event header missing");
  require(done_sse_text.find("\"resident_event_api\":\"ResidentHarness\"") !=
              std::string::npos,
          "resident done SSE API marker missing");
  require(done_sse_text.find("\"resident_harness_loaded\":true") !=
              std::string::npos,
          "resident done SSE loaded marker mismatch");
  require(done_sse_text.find("\"resident_harness_load_ns\":50") !=
              std::string::npos,
          "resident done SSE load timing mismatch");
  require(done_sse_text.find("\"q4_plane_layout_enabled\":true") !=
              std::string::npos,
          "resident done SSE q4-plane flag mismatch");
  require(done_sse_text.find("\"generated_token_ids\":[271]") !=
              std::string::npos,
          "resident done SSE generated tokens mismatch");
  require(done_sse_text.find("\"token_topk_rows\":1") != std::string::npos,
          "resident done SSE oracle stats mismatch");

  iq36::ResidentStreamingSessionConfig session_config;
  session_config.session_id = "unit-session";
  session_config.run_index = 7;
  session_config.max_new_tokens = 2;
  session_config.expected_case_count = 1;
  harness.begin_streaming_session(session_config);
  require(harness.streaming_session_active(),
          "resident streaming session did not become active");
  token_event.run_index = 7;
  token_event.generated_index = 0;
  std::ostringstream session_token_sse;
  harness.emit_sse_session_token_event(session_token_sse, token_event);
  const auto session_token_sse_text = session_token_sse.str();
  require(session_token_sse_text.find("\"resident_session_id\":\"unit-session\"") !=
              std::string::npos,
          "resident session token id missing");
  require(session_token_sse_text.find("\"resident_session_event_index\":0") !=
              std::string::npos,
          "resident session token index mismatch");
  done_event.max_new_tokens = 2;
  done_event.cases[0].generated_token_ids = {271};
  std::ostringstream session_done_sse;
  harness.emit_sse_session_done_event(session_done_sse, done_event);
  const auto session_done_sse_text = session_done_sse.str();
  require(!harness.streaming_session_active(),
          "resident streaming session did not close after done");
  require(session_done_sse_text.find("\"resident_session_id\":\"unit-session\"") !=
              std::string::npos,
          "resident session done id missing");
  require(session_done_sse_text.find("\"resident_session_token_count\":1") !=
              std::string::npos,
          "resident session done token count mismatch");

  harness.begin_streaming_session(session_config);
  iq36::ResidentTokenEvent bad_session_token = token_event;
  bad_session_token.run_index = 8;
  std::ostringstream bad_session_token_sse;
  require_throws(
      [&] {
        harness.emit_sse_session_token_event(
            bad_session_token_sse, bad_session_token);
      },
      "resident session should reject run index mismatch");
  done_event.cases[0].generated_token_ids = {271};
  std::ostringstream bad_session_done_sse;
  require_throws(
      [&] {
        harness.emit_sse_session_done_event(
            bad_session_done_sse, done_event);
      },
      "resident session should reject token count mismatch");

  iq36::ResidentHarness loop_harness;
  iq36::ResidentDecodeLoopConfig loop_config;
  loop_config.session_id_prefix = "loop-session";
  loop_config.run_index_base = 10;
  loop_config.initial_input_token_id = 700;
  loop_config.teacher_forced_token_ids = {900, 901};
  loop_config.max_new_tokens = 2;
  loop_config.session_count = 2;
  loop_config.expected_case_count = 1;
  loop_config.emit_sse_events = true;
  iq36::ResidentDecodeLoop loop(loop_harness);
  std::ostringstream loop_sse;
  const auto loop_result = loop.run(
      loop_sse, loop_config,
      [&](const iq36::ResidentDecodeStepContext& context) {
        require(context.max_new_tokens == 2,
                "resident decode loop max token context mismatch");
        const std::uint32_t expected_input =
            context.token_index == 0 ? 700u : 900u;
        require(context.input_token_id == expected_input,
                "resident decode loop input token context mismatch");
        const auto token_id =
            static_cast<std::uint32_t>(
                1000 + context.session_index * 10 + context.token_index);
        iq36::ResidentTokenEvent event;
        event.case_id = "loop_case";
        event.phase = "decode";
        event.predicted_token_position = 200 + context.token_index;
        event.topk = {iq36::ResidentTopKRow{
            static_cast<std::int32_t>(token_id), 1.0f}};
        event.elapsed_ns = 100 + context.token_index;
        iq36::ResidentDecodeTokenResult result;
        result.event = std::move(event);
        result.top1_matches = true;
        result.topk_matches = context.token_index == 0;
        return result;
      },
      [&](const iq36::ResidentDecodeSessionContext& context) {
        require(context.emitted_token_count == 2,
                "resident decode loop session token count mismatch");
        require(context.top1_match_count == 2,
                "resident decode loop top1 count mismatch");
        require(context.topk_match_count == 1,
                "resident decode loop topk count mismatch");
        require((context.input_token_ids == std::vector<std::uint32_t>{700, 900}),
                "resident decode loop owned input tokens mismatch");
        const std::vector<std::uint32_t> expected_generated = {
            static_cast<std::uint32_t>(1000 + context.session_index * 10),
            static_cast<std::uint32_t>(1001 + context.session_index * 10)};
        require(context.generated_token_ids == expected_generated,
                "resident decode loop owned generated tokens mismatch");
        iq36::ResidentCaseResult loop_case;
        loop_case.case_id = "loop_case";
        loop_case.prompt_token_ids = {1, 2};
        loop_case.first_topk = {iq36::ResidentTopKRow{7, 2.0f}};
        loop_case.generated_token_ids = context.generated_token_ids;
        loop_case.decode_continuation_ns = context.session_elapsed_ns;
        loop_case.case_total_ns = context.session_elapsed_ns;
        iq36::ResidentDoneEvent loop_done;
        loop_done.cases = {loop_case};
        return loop_done;
      });
  const auto loop_sse_text = loop_sse.str();
  require(loop_result.session_count == 2,
          "resident decode loop session count mismatch");
  require(loop_result.emitted_token_count == 4,
          "resident decode loop emitted token count mismatch");
  require(loop_result.input_token_count == 4,
          "resident decode loop input token count mismatch");
  require(loop_result.generated_token_count == 4,
          "resident decode loop generated token count mismatch");
  require(!loop_harness.streaming_session_active(),
          "resident decode loop left a streaming session active");
  require(loop_sse_text.find("\"resident_session_id\":\"loop-session-0\"") !=
              std::string::npos,
          "resident decode loop first session id missing");
  require(loop_sse_text.find("\"resident_session_id\":\"loop-session-1\"") !=
              std::string::npos,
          "resident decode loop second session id missing");
  require(loop_sse_text.find("\"run_index\":10") != std::string::npos,
          "resident decode loop first run index missing");
  require(loop_sse_text.find("\"run_index\":11") != std::string::npos,
          "resident decode loop second run index missing");
  require(loop_sse_text.find("\"resident_session_token_count\":2") !=
              std::string::npos,
          "resident decode loop done token count missing");

  iq36::ResidentHarness hot_loop_harness;
  iq36::ResidentDecodeLoopConfig hot_loop_config;
  hot_loop_config.session_id_prefix = "hot-loop-session";
  hot_loop_config.initial_input_token_id = 42;
  hot_loop_config.teacher_forced_token_ids = {43, 44};
  hot_loop_config.max_new_tokens = 2;
  auto hot_token_fn =
      [&](const iq36::ResidentDecodeStepContext& context) {
        require(context.input_token_id == (context.token_index == 0 ? 42u : 43u),
                "resident GPU hot loop input token mismatch");
        iq36::ResidentTokenEvent event;
        event.case_id = "hot_loop_case";
        event.phase = "decode";
        event.predicted_token_position = 300 + context.token_index;
        event.topk = {iq36::ResidentTopKRow{
            static_cast<std::int32_t>(7000 + context.token_index), 1.0f}};
        iq36::ResidentDecodeTokenResult result;
        result.event = std::move(event);
        result.top1_matches = true;
        result.topk_matches = true;
        return result;
      };
  auto hot_done_fn =
      [&](const iq36::ResidentDecodeSessionContext& context) {
        require((context.input_token_ids == std::vector<std::uint32_t>{42, 43}),
                "resident GPU hot loop owned input tokens mismatch");
        require((context.generated_token_ids ==
                 std::vector<std::uint32_t>{7000, 7001}),
                "resident GPU hot loop owned generated tokens mismatch");
        iq36::ResidentCaseResult loop_case;
        loop_case.case_id = "hot_loop_case";
        loop_case.generated_token_ids = context.generated_token_ids;
        iq36::ResidentDoneEvent loop_done;
        loop_done.cases = {loop_case};
        return loop_done;
      };
  auto hot_runtime = iq36::make_resident_gpu_hot_decode_loop_runtime(
      hot_token_fn, hot_done_fn);
  iq36::ResidentGpuHotDecodeLoop hot_loop(hot_loop_harness);
  std::ostringstream hot_loop_sse;
  const auto hot_loop_result = hot_loop.run(
      hot_loop_sse, hot_loop_config, hot_runtime);
  require(hot_loop_result.session_count == 1,
          "resident GPU hot loop session count mismatch");
  require(hot_loop_result.emitted_token_count == 2,
          "resident GPU hot loop emitted token count mismatch");
  require(hot_loop_result.input_token_count == 2,
          "resident GPU hot loop input token count mismatch");
  require(hot_loop_result.generated_token_count == 2,
          "resident GPU hot loop generated token count mismatch");

  struct UnitDecodeState {
    int token = 0;
    std::vector<int> history;
  };
  iq36::ResidentDecodeStateBank<UnitDecodeState> state_bank(
      UnitDecodeState{7, {1}}, 2);
  require(state_bank.session_count() == 2,
          "resident decode state bank session count mismatch");
  require(state_bank.reset_count() == 0,
          "resident decode state bank initial reset count mismatch");
  state_bank.session_state(1).token = 9;
  require(state_bank.session_state(0).token == 7,
          "resident decode state bank sessions alias unexpectedly");
  require(state_bank.session_state(1).token == 9,
          "resident decode state bank session mutation missing");
  state_bank.reset_session(1, UnitDecodeState{11, {2, 3}});
  require(state_bank.reset_count() == 1,
          "resident decode state bank reset count mismatch");
  require(state_bank.session_state(1).token == 11,
          "resident decode state bank reset token mismatch");
  require((state_bank.session_state(1).history == std::vector<int>{2, 3}),
          "resident decode state bank reset history mismatch");
  require_throws(
      [&] { (void)state_bank.session_state(2); },
      "resident decode state bank should reject out-of-range session");

  iq36::ResidentDeviceStateHandleBank handle_bank;
  handle_bank.reset(3);
  require(handle_bank.layer_count() == 3,
          "resident device state handle bank layer count mismatch");
  handle_bank.set_layer_handles(1, 10, 20, 30, 120);
  require(handle_bank.handle_count() == 3,
          "resident device state handle bank handle count mismatch");
  require(handle_bank.uploaded_bytes() == 120,
          "resident device state handle bank bytes mismatch");
  require(handle_bank.misses() == 3,
          "resident device state handle bank miss count mismatch");
  require(handle_bank.recurrent_handle(1) == 10,
          "resident device state handle bank recurrent handle mismatch");
  require(handle_bank.conv_handle(1) == 20,
          "resident device state handle bank conv handle mismatch");
  require(handle_bank.conv_next_handle(1) == 30,
          "resident device state handle bank next conv handle mismatch");
  require(handle_bank.hits() == 3,
          "resident device state handle bank hit count mismatch");
  handle_bank.swap_conv_handles(1);
  require(handle_bank.conv_handle(1) == 30,
          "resident device state handle bank swap conv mismatch");
  require(handle_bank.conv_next_handle(1) == 20,
          "resident device state handle bank swap next mismatch");
  handle_bank.set_layer_recurrent_handle(1, 40, 16);
  require(handle_bank.recurrent_handle(1) == 40,
          "resident device state handle bank recurrent update mismatch");
  handle_bank.set_layer_conv_handles(1, 50, 60, 32);
  require(handle_bank.conv_handle(1) == 50,
          "resident device state handle bank conv update mismatch");
  require(handle_bank.conv_next_handle(1) == 60,
          "resident device state handle bank next conv update mismatch");
  handle_bank.reset_hit_miss_counters();
  require(handle_bank.hits() == 0 && handle_bank.misses() == 0,
          "resident device state handle bank counter reset mismatch");
  require_throws(
      [&] { (void)handle_bank.recurrent_handle(2); },
      "resident device state handle bank should reject missing handle");
  require_throws(
      [&] { handle_bank.set_layer_handles(3, 1, 2, 3, 4); },
      "resident device state handle bank should reject out-of-range layer");

  iq36::ResidentSmallTensorStore small_tensor_store;
  small_tensor_store.insert("norm.values.0", 101, 64);
  small_tensor_store.insert("layer.0.attn_norm", 101, 0);
  small_tensor_store.insert("norm.values.1", 202, 128);
  require(small_tensor_store.handle_count() == 2,
          "resident small tensor store handle count mismatch");
  require(small_tensor_store.key_count() == 3,
          "resident small tensor store key count mismatch");
  require(small_tensor_store.uploaded_bytes() == 192,
          "resident small tensor store bytes mismatch");
  require(small_tensor_store.lookup_handle("layer.0.attn_norm") == 101,
          "resident small tensor store alias lookup mismatch");
  require(small_tensor_store.find_handle("norm.values.1") == 202,
          "resident small tensor store value lookup mismatch");
  require(small_tensor_store.find_handle("missing") == 0,
          "resident small tensor store missing lookup mismatch");
  require(small_tensor_store.hits() == 2 && small_tensor_store.misses() == 1,
          "resident small tensor store counter mismatch");
  small_tensor_store.reset_hit_miss_counters();
  require(small_tensor_store.hits() == 0 && small_tensor_store.misses() == 0,
          "resident small tensor store counter reset mismatch");
  require_throws(
      [&] { small_tensor_store.insert("", 1, 0); },
      "resident small tensor store should reject empty key");
  require_throws(
      [&] { small_tensor_store.insert("bad", 0, 0); },
      "resident small tensor store should reject zero handle");
  require_throws(
      [&] { small_tensor_store.insert("norm.values.0", 303, 0); },
      "resident small tensor store should reject handle change");
  require_throws(
      [&] { (void)small_tensor_store.lookup_handle("missing"); },
      "resident small tensor store should reject missing handle");
  small_tensor_store.clear();
  require(small_tensor_store.handle_count() == 0 &&
              small_tensor_store.uploaded_bytes() == 0,
          "resident small tensor store clear mismatch");

  iq36::ResidentF32MatvecWeightStore f32_weight_store;
  f32_weight_store.insert("blk.0.ffn_gate_inp.weight", 1001, 256, 2048, 2048);
  f32_weight_store.insert("blk.1.ffn_gate_inp.weight", 1002, 256, 2048, 2048);
  require(f32_weight_store.handle_count() == 2,
          "resident F32 matvec weight store handle count mismatch");
  require(f32_weight_store.uploaded_bytes() == 4096,
          "resident F32 matvec weight store bytes mismatch");
  require(f32_weight_store.lookup_handle(
              "blk.0.ffn_gate_inp.weight", 256, 2048) == 1001,
          "resident F32 matvec weight store lookup mismatch");
  require(f32_weight_store.find_handle(
              "blk.1.ffn_gate_inp.weight", 256, 2048) == 1002,
          "resident F32 matvec weight store find mismatch");
  require(f32_weight_store.find_handle("missing", 256, 2048) == 0,
          "resident F32 matvec weight store missing lookup mismatch");
  require(f32_weight_store.hits() == 2 && f32_weight_store.misses() == 1,
          "resident F32 matvec weight store counter mismatch");
  f32_weight_store.reset_hit_miss_counters();
  require(f32_weight_store.hits() == 0 && f32_weight_store.misses() == 0,
          "resident F32 matvec weight store counter reset mismatch");
  require_throws(
      [&] { f32_weight_store.insert("", 1, 1, 1, 0); },
      "resident F32 matvec weight store should reject empty key");
  require_throws(
      [&] { f32_weight_store.insert("bad", 0, 1, 1, 0); },
      "resident F32 matvec weight store should reject zero handle");
  require_throws(
      [&] { f32_weight_store.insert("bad-shape", 1, 0, 1, 0); },
      "resident F32 matvec weight store should reject zero rows");
  require_throws(
      [&] {
        f32_weight_store.insert(
            "blk.0.ffn_gate_inp.weight", 1003, 256, 2048, 0);
      },
      "resident F32 matvec weight store should reject handle change");
  require_throws(
      [&] {
        (void)f32_weight_store.find_handle(
            "blk.0.ffn_gate_inp.weight", 128, 2048);
      },
      "resident F32 matvec weight store should reject shape mismatch");
  require_throws(
      [&] {
        (void)f32_weight_store.lookup_handle("missing", 256, 2048);
      },
      "resident F32 matvec weight store should reject missing handle");
  f32_weight_store.clear();
  require(f32_weight_store.handle_count() == 0 &&
              f32_weight_store.uploaded_bytes() == 0,
          "resident F32 matvec weight store clear mismatch");

  iq36::ResidentQ6RowstripeWeightStore q6_weight_store;
  q6_weight_store.insert("blk.0.attn_qkv.weight", 2001, 8192, 256, 8, 4096);
  q6_weight_store.insert("blk.1.attn_qkv.weight", 2002, 8192, 256, 8, 4096);
  require(q6_weight_store.handle_count() == 2,
          "resident Q6 rowstripe weight store handle count mismatch");
  require(q6_weight_store.uploaded_bytes() == 8192,
          "resident Q6 rowstripe weight store bytes mismatch");
  require(q6_weight_store.lookup_handle(
              "blk.0.attn_qkv.weight", 8192, 256, 8) == 2001,
          "resident Q6 rowstripe weight store lookup mismatch");
  require(q6_weight_store.find_handle(
              "blk.1.attn_qkv.weight", 8192, 256, 8) == 2002,
          "resident Q6 rowstripe weight store find mismatch");
  require(q6_weight_store.find_handle("missing", 8192, 256, 8) == 0,
          "resident Q6 rowstripe weight store missing lookup mismatch");
  require(q6_weight_store.hits() == 2 && q6_weight_store.misses() == 1,
          "resident Q6 rowstripe weight store counter mismatch");
  q6_weight_store.reset_hit_miss_counters();
  require(q6_weight_store.hits() == 0 && q6_weight_store.misses() == 0,
          "resident Q6 rowstripe weight store counter reset mismatch");
  require_throws(
      [&] { q6_weight_store.insert("", 1, 1, 1, 1, 0); },
      "resident Q6 rowstripe weight store should reject empty key");
  require_throws(
      [&] { q6_weight_store.insert("bad", 0, 1, 1, 1, 0); },
      "resident Q6 rowstripe weight store should reject zero handle");
  require_throws(
      [&] { q6_weight_store.insert("bad-shape", 1, 0, 1, 1, 0); },
      "resident Q6 rowstripe weight store should reject zero rows");
  require_throws(
      [&] {
        q6_weight_store.insert("blk.0.attn_qkv.weight", 2003, 8192, 256, 8, 0);
      },
      "resident Q6 rowstripe weight store should reject handle change");
  require_throws(
      [&] {
        (void)q6_weight_store.find_handle(
            "blk.0.attn_qkv.weight", 8192, 512, 8);
      },
      "resident Q6 rowstripe weight store should reject shape mismatch");
  require_throws(
      [&] {
        (void)q6_weight_store.lookup_handle("missing", 8192, 256, 8);
      },
      "resident Q6 rowstripe weight store should reject missing handle");
  q6_weight_store.clear();
  require(q6_weight_store.handle_count() == 0 &&
              q6_weight_store.uploaded_bytes() == 0,
          "resident Q6 rowstripe weight store clear mismatch");

  iq36::ResidentSelectedQ6WeightStore selected_q6_weight_store;
  selected_q6_weight_store.insert(
      "blk.0.ffn_down.weight#selected#1,2", 2051, 2048, 256, 8, 2,
      true, 4096);
  selected_q6_weight_store.insert(
      "blk.0.ffn_down.weight#expert#7", 2052, 2048, 256, 8, 1,
      true, 4096);
  require(selected_q6_weight_store.handle_count() == 2,
          "resident selected Q6 weight store handle count mismatch");
  require(selected_q6_weight_store.uploaded_bytes() == 8192,
          "resident selected Q6 weight store bytes mismatch");
  require(selected_q6_weight_store.lookup_handle(
              "blk.0.ffn_down.weight#selected#1,2", 2048, 256, 8, 2,
              true) == 2051,
          "resident selected Q6 weight store lookup mismatch");
  require(selected_q6_weight_store.find_handle(
              "blk.0.ffn_down.weight#expert#7", 2048, 256, 8, 1,
              true) == 2052,
          "resident selected Q6 weight store find mismatch");
  require(selected_q6_weight_store.find_handle(
              "missing", 2048, 256, 8, 1, true) == 0,
          "resident selected Q6 weight store missing lookup mismatch");
  require(selected_q6_weight_store.hits() == 2 &&
              selected_q6_weight_store.misses() == 1,
          "resident selected Q6 weight store counter mismatch");
  selected_q6_weight_store.reset_hit_miss_counters();
  require(selected_q6_weight_store.hits() == 0 &&
              selected_q6_weight_store.misses() == 0,
          "resident selected Q6 weight store counter reset mismatch");
  require_throws(
      [&] {
        selected_q6_weight_store.insert("", 1, 1, 1, 1, 1, false, 0);
      },
      "resident selected Q6 weight store should reject empty key");
  require_throws(
      [&] {
        selected_q6_weight_store.insert("bad", 0, 1, 1, 1, 1, false, 0);
      },
      "resident selected Q6 weight store should reject zero handle");
  require_throws(
      [&] {
        selected_q6_weight_store.insert(
            "bad-shape", 1, 0, 1, 1, 1, false, 0);
      },
      "resident selected Q6 weight store should reject zero rows");
  require_throws(
      [&] {
        selected_q6_weight_store.insert(
            "blk.0.ffn_down.weight#selected#1,2", 2053, 2048, 256, 8,
            2, true, 0);
      },
      "resident selected Q6 weight store should reject handle change");
  require_throws(
      [&] {
        (void)selected_q6_weight_store.find_handle(
            "blk.0.ffn_down.weight#selected#1,2", 2048, 256, 8, 2,
            false);
      },
      "resident selected Q6 weight store should reject shape mismatch");
  require_throws(
      [&] {
        (void)selected_q6_weight_store.lookup_handle(
            "missing", 2048, 256, 8, 1, true);
      },
      "resident selected Q6 weight store should reject missing handle");
  selected_q6_weight_store.clear();
  require(selected_q6_weight_store.handle_count() == 0 &&
              selected_q6_weight_store.uploaded_bytes() == 0,
          "resident selected Q6 weight store clear mismatch");

  iq36::ResidentRawQ6WeightStore raw_q6_weight_store;
  raw_q6_weight_store.insert("blk.0.ffn_down_shexp.weight", 2101, 4096, 256, 8, 4096);
  raw_q6_weight_store.insert("blk.1.attn_v.weight", 2102, 2048, 256, 8, 4096);
  require(raw_q6_weight_store.handle_count() == 2,
          "resident raw Q6 weight store handle count mismatch");
  require(raw_q6_weight_store.uploaded_bytes() == 8192,
          "resident raw Q6 weight store bytes mismatch");
  require(raw_q6_weight_store.lookup_handle(
              "blk.0.ffn_down_shexp.weight", 4096, 256, 8) == 2101,
          "resident raw Q6 weight store lookup mismatch");
  require(raw_q6_weight_store.find_handle(
              "blk.1.attn_v.weight", 2048, 256, 8) == 2102,
          "resident raw Q6 weight store find mismatch");
  require(raw_q6_weight_store.find_handle("missing", 2048, 256, 8) == 0,
          "resident raw Q6 weight store missing lookup mismatch");
  require(raw_q6_weight_store.hits() == 2 && raw_q6_weight_store.misses() == 1,
          "resident raw Q6 weight store counter mismatch");
  raw_q6_weight_store.reset_hit_miss_counters();
  require(raw_q6_weight_store.hits() == 0 &&
              raw_q6_weight_store.misses() == 0,
          "resident raw Q6 weight store counter reset mismatch");
  require_throws(
      [&] { raw_q6_weight_store.insert("", 1, 1, 1, 1, 0); },
      "resident raw Q6 weight store should reject empty key");
  require_throws(
      [&] { raw_q6_weight_store.insert("bad", 0, 1, 1, 1, 0); },
      "resident raw Q6 weight store should reject zero handle");
  require_throws(
      [&] { raw_q6_weight_store.insert("bad-shape", 1, 0, 1, 1, 0); },
      "resident raw Q6 weight store should reject zero rows");
  require_throws(
      [&] {
        raw_q6_weight_store.insert(
            "blk.0.ffn_down_shexp.weight", 2103, 4096, 256, 8, 0);
      },
      "resident raw Q6 weight store should reject handle change");
  require_throws(
      [&] {
        (void)raw_q6_weight_store.find_handle(
            "blk.0.ffn_down_shexp.weight", 4096, 512, 8);
      },
      "resident raw Q6 weight store should reject shape mismatch");
  require_throws(
      [&] {
        (void)raw_q6_weight_store.lookup_handle("missing", 2048, 256, 8);
      },
      "resident raw Q6 weight store should reject missing handle");
  raw_q6_weight_store.clear();
  require(raw_q6_weight_store.handle_count() == 0 &&
              raw_q6_weight_store.uploaded_bytes() == 0,
          "resident raw Q6 weight store clear mismatch");

  iq36::ResidentPackedQ4WeightStore q4_weight_store;
  q4_weight_store.insert("blk.0.ffn_gate.weight", 3001, 16384, 8, 4096);
  q4_weight_store.insert("blk.0.ffn_down.weight.expert7", 3002, 2048, 8, 4096);
  require(q4_weight_store.handle_count() == 2,
          "resident packed Q4 weight store handle count mismatch");
  require(q4_weight_store.uploaded_bytes() == 8192,
          "resident packed Q4 weight store bytes mismatch");
  require(q4_weight_store.lookup_handle(
              "blk.0.ffn_gate.weight", 16384, 8) == 3001,
          "resident packed Q4 weight store lookup mismatch");
  require(q4_weight_store.find_handle(
              "blk.0.ffn_down.weight.expert7", 2048, 8) == 3002,
          "resident packed Q4 weight store find mismatch");
  require(q4_weight_store.find_handle("missing", 2048, 8) == 0,
          "resident packed Q4 weight store missing lookup mismatch");
  require(q4_weight_store.hits() == 2 && q4_weight_store.misses() == 1,
          "resident packed Q4 weight store counter mismatch");
  q4_weight_store.reset_hit_miss_counters();
  require(q4_weight_store.hits() == 0 && q4_weight_store.misses() == 0,
          "resident packed Q4 weight store counter reset mismatch");
  require_throws(
      [&] { q4_weight_store.insert("", 1, 1, 1, 0); },
      "resident packed Q4 weight store should reject empty key");
  require_throws(
      [&] { q4_weight_store.insert("bad", 0, 1, 1, 0); },
      "resident packed Q4 weight store should reject zero handle");
  require_throws(
      [&] { q4_weight_store.insert("bad-shape", 1, 0, 1, 0); },
      "resident packed Q4 weight store should reject zero rows");
  require_throws(
      [&] {
        q4_weight_store.insert("blk.0.ffn_gate.weight", 3003, 16384, 8, 0);
      },
      "resident packed Q4 weight store should reject handle change");
  require_throws(
      [&] {
        (void)q4_weight_store.find_handle("blk.0.ffn_gate.weight", 8192, 8);
      },
      "resident packed Q4 weight store should reject shape mismatch");
  require_throws(
      [&] { (void)q4_weight_store.lookup_handle("missing", 2048, 8); },
      "resident packed Q4 weight store should reject missing handle");
  q4_weight_store.clear();
  require(q4_weight_store.handle_count() == 0 &&
              q4_weight_store.uploaded_bytes() == 0,
          "resident packed Q4 weight store clear mismatch");

  iq36::ResidentQ4CpuOrderWeightStore q4_cpu_order_weight_store;
  q4_cpu_order_weight_store.insert("blk.0.attn_z.weight", 3501, 2048, 8, 4096);
  q4_cpu_order_weight_store.insert(
      "blk.0.attn_alpha.beta.z.weight", 3502, 6144, 8, 4096);
  require(q4_cpu_order_weight_store.handle_count() == 2,
          "resident Q4 CPU-order weight store handle count mismatch");
  require(q4_cpu_order_weight_store.uploaded_bytes() == 8192,
          "resident Q4 CPU-order weight store bytes mismatch");
  require(q4_cpu_order_weight_store.lookup_handle(
              "blk.0.attn_z.weight", 2048, 8) == 3501,
          "resident Q4 CPU-order weight store lookup mismatch");
  require(q4_cpu_order_weight_store.find_handle(
              "blk.0.attn_alpha.beta.z.weight", 6144, 8) == 3502,
          "resident Q4 CPU-order weight store find mismatch");
  require(q4_cpu_order_weight_store.find_handle("missing", 2048, 8) == 0,
          "resident Q4 CPU-order weight store missing lookup mismatch");
  require(q4_cpu_order_weight_store.hits() == 2 &&
              q4_cpu_order_weight_store.misses() == 1,
          "resident Q4 CPU-order weight store counter mismatch");
  q4_cpu_order_weight_store.reset_hit_miss_counters();
  require(q4_cpu_order_weight_store.hits() == 0 &&
              q4_cpu_order_weight_store.misses() == 0,
          "resident Q4 CPU-order weight store counter reset mismatch");
  require_throws(
      [&] { q4_cpu_order_weight_store.insert("", 1, 1, 1, 0); },
      "resident Q4 CPU-order weight store should reject empty key");
  require_throws(
      [&] { q4_cpu_order_weight_store.insert("bad", 0, 1, 1, 0); },
      "resident Q4 CPU-order weight store should reject zero handle");
  require_throws(
      [&] { q4_cpu_order_weight_store.insert("bad-shape", 1, 0, 1, 0); },
      "resident Q4 CPU-order weight store should reject zero rows");
  require_throws(
      [&] {
        q4_cpu_order_weight_store.insert(
            "blk.0.attn_z.weight", 3503, 2048, 8, 0);
      },
      "resident Q4 CPU-order weight store should reject handle change");
  require_throws(
      [&] {
        (void)q4_cpu_order_weight_store.find_handle(
            "blk.0.attn_z.weight", 4096, 8);
      },
      "resident Q4 CPU-order weight store should reject shape mismatch");
  require_throws(
      [&] {
        (void)q4_cpu_order_weight_store.lookup_handle("missing", 2048, 8);
      },
      "resident Q4 CPU-order weight store should reject missing handle");
  q4_cpu_order_weight_store.clear();
  require(q4_cpu_order_weight_store.handle_count() == 0 &&
              q4_cpu_order_weight_store.uploaded_bytes() == 0,
          "resident Q4 CPU-order weight store clear mismatch");

  iq36::ResidentLmHeadWeightStore lm_head_weight_store;
  lm_head_weight_store.insert_q6("output.weight", 4501, 151936, 8, 4096);
  lm_head_weight_store.insert_norm("output_norm.weight", 4502, 2048, 8192);
  require(lm_head_weight_store.handle_count() == 2,
          "resident LM-head weight store handle count mismatch");
  require(lm_head_weight_store.q6_handle_count() == 1 &&
              lm_head_weight_store.norm_handle_count() == 1,
          "resident LM-head weight store split count mismatch");
  require(lm_head_weight_store.uploaded_bytes() == 12288 &&
              lm_head_weight_store.q6_uploaded_bytes() == 4096 &&
              lm_head_weight_store.norm_uploaded_bytes() == 8192,
          "resident LM-head weight store bytes mismatch");
  require(lm_head_weight_store.lookup_q6("output.weight", 151936, 8) == 4501,
          "resident LM-head Q6 lookup mismatch");
  require(lm_head_weight_store.lookup_norm("output_norm.weight", 2048) == 4502,
          "resident LM-head norm lookup mismatch");
  require(lm_head_weight_store.find_q6("missing", 151936, 8) == 0 &&
              lm_head_weight_store.find_norm("missing", 2048) == 0,
          "resident LM-head missing lookup mismatch");
  require(lm_head_weight_store.q6_hits() == 1 &&
              lm_head_weight_store.q6_misses() == 1 &&
              lm_head_weight_store.norm_hits() == 1 &&
              lm_head_weight_store.norm_misses() == 1 &&
              lm_head_weight_store.hits() == 2 &&
              lm_head_weight_store.misses() == 2,
          "resident LM-head weight store counter mismatch");
  lm_head_weight_store.reset_hit_miss_counters();
  require(lm_head_weight_store.hits() == 0 &&
              lm_head_weight_store.misses() == 0,
          "resident LM-head weight store counter reset mismatch");
  require_throws(
      [&] { lm_head_weight_store.insert_q6("", 1, 1, 1, 0); },
      "resident LM-head weight store should reject empty Q6 key");
  require_throws(
      [&] { lm_head_weight_store.insert_q6("bad", 0, 1, 1, 0); },
      "resident LM-head weight store should reject zero Q6 handle");
  require_throws(
      [&] { lm_head_weight_store.insert_q6("bad-shape", 1, 0, 1, 0); },
      "resident LM-head weight store should reject zero Q6 rows");
  require_throws(
      [&] { lm_head_weight_store.insert_norm("", 1, 1, 0); },
      "resident LM-head weight store should reject empty norm key");
  require_throws(
      [&] { lm_head_weight_store.insert_norm("bad", 0, 1, 0); },
      "resident LM-head weight store should reject zero norm handle");
  require_throws(
      [&] { lm_head_weight_store.insert_norm("bad-shape", 1, 0, 0); },
      "resident LM-head weight store should reject zero norm elements");
  require_throws(
      [&] { lm_head_weight_store.insert_q6("output.weight", 4503, 151936, 8, 0); },
      "resident LM-head weight store should reject Q6 handle change");
  require_throws(
      [&] {
        lm_head_weight_store.insert_norm("output_norm.weight", 4503, 2048, 0);
      },
      "resident LM-head weight store should reject norm handle change");
  require_throws(
      [&] {
        (void)lm_head_weight_store.find_q6("output.weight", 151936, 16);
      },
      "resident LM-head weight store should reject Q6 shape mismatch");
  require_throws(
      [&] {
        (void)lm_head_weight_store.find_norm("output_norm.weight", 4096);
      },
      "resident LM-head weight store should reject norm shape mismatch");
  require_throws(
      [&] {
        (void)lm_head_weight_store.lookup_q6("missing", 151936, 8);
      },
      "resident LM-head weight store should reject missing Q6 handle");
  require_throws(
      [&] {
        (void)lm_head_weight_store.lookup_norm("missing", 2048);
      },
      "resident LM-head weight store should reject missing norm handle");
  lm_head_weight_store.clear();
  require(lm_head_weight_store.handle_count() == 0 &&
              lm_head_weight_store.uploaded_bytes() == 0,
          "resident LM-head weight store clear mismatch");

  iq36::ResidentF32ConvWeightStore conv_weight_store;
  conv_weight_store.insert("blk.0.conv1d.weight", 4001, 8192, 4, 4096);
  conv_weight_store.insert("blk.1.conv1d.weight", 4002, 8192, 4, 4096);
  require(conv_weight_store.handle_count() == 2,
          "resident F32 conv weight store handle count mismatch");
  require(conv_weight_store.uploaded_bytes() == 8192,
          "resident F32 conv weight store bytes mismatch");
  require(conv_weight_store.lookup_handle(
              "blk.0.conv1d.weight", 8192, 4) == 4001,
          "resident F32 conv weight store lookup mismatch");
  require(conv_weight_store.find_handle(
              "blk.1.conv1d.weight", 8192, 4) == 4002,
          "resident F32 conv weight store find mismatch");
  require(conv_weight_store.find_handle("missing", 8192, 4) == 0,
          "resident F32 conv weight store missing lookup mismatch");
  require(conv_weight_store.hits() == 2 && conv_weight_store.misses() == 1,
          "resident F32 conv weight store counter mismatch");
  conv_weight_store.reset_hit_miss_counters();
  require(conv_weight_store.hits() == 0 && conv_weight_store.misses() == 0,
          "resident F32 conv weight store counter reset mismatch");
  require_throws(
      [&] { conv_weight_store.insert("", 1, 1, 1, 0); },
      "resident F32 conv weight store should reject empty key");
  require_throws(
      [&] { conv_weight_store.insert("bad", 0, 1, 1, 0); },
      "resident F32 conv weight store should reject zero handle");
  require_throws(
      [&] { conv_weight_store.insert("bad-shape", 1, 0, 1, 0); },
      "resident F32 conv weight store should reject zero rows");
  require_throws(
      [&] {
        conv_weight_store.insert("blk.0.conv1d.weight", 4003, 8192, 4, 0);
      },
      "resident F32 conv weight store should reject handle change");
  require_throws(
      [&] {
        (void)conv_weight_store.find_handle("blk.0.conv1d.weight", 8192, 8);
      },
      "resident F32 conv weight store should reject shape mismatch");
  require_throws(
      [&] { (void)conv_weight_store.lookup_handle("missing", 8192, 4); },
      "resident F32 conv weight store should reject missing handle");
  conv_weight_store.clear();
  require(conv_weight_store.handle_count() == 0 &&
              conv_weight_store.uploaded_bytes() == 0,
          "resident F32 conv weight store clear mismatch");

  const auto gguf_path =
      std::filesystem::temp_directory_path() / "iq36-unit-synthetic.gguf";
  write_synthetic_gguf(gguf_path);
  const auto gguf = iq36::parse_gguf_model_index(gguf_path.string());
  std::filesystem::remove(gguf_path);
  require(gguf.version == 3, "synthetic GGUF version mismatch");
  require(gguf.tensor_count == 1, "synthetic GGUF tensor count mismatch");
  require(gguf.metadata_kv_count == 3, "synthetic GGUF metadata count mismatch");
  require(gguf.tensors.size() == 1, "synthetic GGUF tensor table missing");
  require(gguf.tensors[0].name == "unit.weight", "synthetic GGUF tensor name mismatch");
  require(gguf.tensors[0].nbytes == 144, "synthetic GGUF Q4_K byte size mismatch");
  const auto sections_it = gguf.metadata.find("unit.sections");
  require(sections_it != gguf.metadata.end(), "synthetic GGUF array missing");
  require(sections_it->second.kind == iq36::GgufMetadataValue::Kind::kArray,
          "synthetic GGUF array kind mismatch");
  require(sections_it->second.array_element_type == 5,
          "synthetic GGUF array element type mismatch");
  require(sections_it->second.int_array == std::vector<std::int64_t>({11, 11, 10, 0}),
          "synthetic GGUF int array mismatch");
  require(iq36::ggml_type_name(12) == "Q4_K", "GGML type name mismatch");

  const auto vector_path =
      std::filesystem::temp_directory_path() / "iq36-unit-f32-vector.bin";
  const std::vector<float> expected_vector{1.0f, -2.0f, 0.25f};
  write_f32_vector(vector_path, expected_vector);
  const auto read_vector = iq36::read_f32_vector_file(vector_path.string());
  std::filesystem::remove(vector_path);
  require(read_vector == expected_vector, "f32 vector read mismatch");
  const auto compare = iq36::compare_vectors(read_vector, expected_vector, 0.0);
  require(compare.same_size, "vector compare size mismatch");
  require(compare.finite, "vector compare finite mismatch");
  require(compare.mismatch_count == 0, "vector compare unexpected mismatch");
  require(compare.max_abs_diff == 0.0, "vector compare max diff mismatch");
  const auto rms = iq36::apply_rms_norm(
      std::vector<float>{3.0f, 4.0f},
      std::vector<float>{1.0f, 2.0f},
      0.0f);
  require(rms.size() == 2, "RMSNorm size mismatch");
  require(std::abs(rms[0] - 0.84852815f) < 1e-6f, "RMSNorm value 0 mismatch");
  require(std::abs(rms[1] - 2.2627418f) < 1e-6f, "RMSNorm value 1 mismatch");

  const auto matrix_path =
      std::filesystem::temp_directory_path() / "iq36-unit-f32-matrix.gguf";
  write_synthetic_f32_matrix_gguf(matrix_path);
  const auto matrix_index = iq36::parse_gguf_model_index(matrix_path.string());
  const auto matvec = iq36::matvec_tensor(
      matrix_path.string(),
      matrix_index,
      "unit.matrix",
      std::vector<float>{10.0f, 20.0f});
  std::filesystem::remove(matrix_path);
  require(matvec == std::vector<float>({50.0f, 110.0f, 170.0f}),
          "matvec orientation mismatch");

  const auto expert_path =
      std::filesystem::temp_directory_path() / "iq36-unit-f32-experts.gguf";
  write_synthetic_f32_expert_gguf(expert_path);
  const auto expert_index = iq36::parse_gguf_model_index(expert_path.string());
  const auto expert_matvec = iq36::matvec_expert_tensor(
      expert_path.string(),
      expert_index,
      "unit.experts",
      std::vector<float>{10.0f, 20.0f},
      std::vector<std::int32_t>{1, 0});
  std::filesystem::remove(expert_path);
  require(expert_matvec == std::vector<float>({
                              230.0f, 290.0f, 350.0f,
                              50.0f, 110.0f, 170.0f}),
          "expert matvec orientation mismatch");
  const auto expert_path_per_input =
      std::filesystem::temp_directory_path() / "iq36-unit-f32-experts-per-input.gguf";
  write_synthetic_f32_expert_gguf(expert_path_per_input);
  const auto expert_per_input_index =
      iq36::parse_gguf_model_index(expert_path_per_input.string());
  const auto expert_per_input_matvec =
      iq36::matvec_expert_tensor_per_expert_input(
          expert_path_per_input.string(),
          expert_per_input_index,
          "unit.experts",
          std::vector<float>{10.0f, 20.0f, 1.0f, 1.0f},
          std::vector<std::int32_t>{1, 0});
  std::filesystem::remove(expert_path_per_input);
  require(expert_per_input_matvec == std::vector<float>({
                                      230.0f, 290.0f, 350.0f,
                                      3.0f, 7.0f, 11.0f}),
          "per-expert input matvec orientation mismatch");

  const auto swiglu = iq36::apply_swiglu_from_gate_up(
      std::vector<float>{1.0f, -2.0f, 3.0f, 4.0f, -1.0f, 2.0f, 0.5f, -0.5f},
      2,
      2);
  require(swiglu.size() == 4, "SwiGLU size mismatch");
  require(std::abs(swiglu[0] - 2.1931758f) < 1e-6f, "SwiGLU value 0 mismatch");
  require(std::abs(swiglu[1] - -0.9536231f) < 1e-6f, "SwiGLU value 1 mismatch");
  require(std::abs(swiglu[2] - -0.13447072f) < 1e-6f, "SwiGLU value 2 mismatch");
  require(std::abs(swiglu[3] - -0.8807971f) < 1e-6f, "SwiGLU value 3 mismatch");

  require(std::abs(iq36::sigmoid_scalar(0.0f) - 0.5f) < 1e-7f,
          "sigmoid scalar zero mismatch");
  const auto full_attn_gate = iq36::run_qwen36_full_attention_gate(
      std::vector<float>{10.0f, 20.0f, 0.0f, 2.0f,
                         30.0f, 40.0f, -2.0f, 1.0f},
      std::vector<float>{4.0f, 5.0f, 6.0f, 7.0f},
      2);
  require(full_attn_gate.q_gate ==
              std::vector<float>({0.0f, 2.0f, -2.0f, 1.0f}),
          "full attention gate split mismatch");
  require(std::abs(full_attn_gate.gate_sigmoid[0] - 0.5f) < 1e-7f &&
              std::abs(full_attn_gate.gate_sigmoid[1] - 0.8807971f) < 1e-6f &&
              std::abs(full_attn_gate.gate_sigmoid[2] - 0.1192029f) < 1e-6f &&
              std::abs(full_attn_gate.gate_sigmoid[3] - 0.7310586f) < 1e-6f,
          "full attention sigmoid gate mismatch");
  require(full_attn_gate.attn_gated.size() == 4,
          "full attention gated output size mismatch");
  require(std::abs(full_attn_gate.attn_gated[0] - 2.0f) < 1e-6f &&
              std::abs(full_attn_gate.attn_gated[1] - 4.4039855f) < 1e-6f &&
              std::abs(full_attn_gate.attn_gated[2] - 0.7152175f) < 1e-6f &&
              std::abs(full_attn_gate.attn_gated[3] - 5.1174102f) < 1e-6f,
          "full attention gated output mismatch");
  const auto full_attn_core = iq36::run_qwen36_full_attention_core(
      std::vector<float>{1.0f, 0.0f, 0.0f, 1.0f},
      std::vector<std::vector<float>>{
          {1.0f, 0.0f},
          {0.0f, 1.0f},
      },
      std::vector<std::vector<float>>{
          {10.0f, 20.0f},
          {30.0f, 40.0f},
      },
      2,
      2,
      1,
      1.0f);
  require(full_attn_core.attention_weights.size() == 4,
          "full attention core weight size mismatch");
  require(full_attn_core.attn_pregate.size() == 4,
          "full attention core output size mismatch");
  require(std::abs(full_attn_core.attention_weights[0] - 0.7310586f) <
              1e-6f &&
              std::abs(full_attn_core.attention_weights[1] - 0.2689414f) <
                  1e-6f &&
              std::abs(full_attn_core.attention_weights[2] - 0.2689414f) <
                  1e-6f &&
              std::abs(full_attn_core.attention_weights[3] - 0.7310586f) <
                  1e-6f,
          "full attention core softmax mismatch");
  require(std::abs(full_attn_core.attn_pregate[0] - 15.378828f) < 1e-5f &&
              std::abs(full_attn_core.attn_pregate[1] - 25.378828f) <
                  1e-5f &&
              std::abs(full_attn_core.attn_pregate[2] - 24.621172f) <
                  1e-5f &&
              std::abs(full_attn_core.attn_pregate[3] - 34.621172f) <
                  1e-5f,
          "full attention core output mismatch");
  const auto probs = iq36::softmax(std::vector<float>{1.0f, 2.0f});
  require(probs.size() == 2, "softmax size mismatch");
  require(std::abs((probs[0] + probs[1]) - 1.0f) < 1e-6f,
          "softmax normalization mismatch");
  require(probs[1] > probs[0], "softmax ordering mismatch");
  const auto topk = iq36::top_k_indices(
      std::vector<float>{0.1f, 0.3f, 0.3f, 0.2f}, 2);
  require(topk == std::vector<std::int32_t>({1, 2}),
          "top-k stable tie ordering mismatch");
  const auto gathered = iq36::gather_values(
      std::vector<float>{10.0f, 20.0f, 30.0f}, std::vector<std::int32_t>{2, 0});
  require(gathered == std::vector<float>({30.0f, 10.0f}),
          "gather values mismatch");
  const auto normalized =
      iq36::normalize_weights(std::vector<float>{0.25f, 0.75f}, 1e-6f);
  require(normalized == std::vector<float>({0.25f, 0.75f}),
          "normal weight normalization mismatch");
  const auto clamped =
      iq36::normalize_weights(std::vector<float>{1e-8f, 1e-8f}, 1e-4f);
  require(std::abs(clamped[0] - 1e-4f) < 1e-10f &&
              std::abs(clamped[1] - 1e-4f) < 1e-10f,
          "minimum weight normalization mismatch");
  const auto selection =
      iq36::select_router_topk(std::vector<float>{0.0f, 1.0f, 2.0f}, 2, 1e-6f);
  require(selection.expert_ids == std::vector<std::int32_t>({2, 1}),
          "router top-k selection mismatch");
  require(selection.weights.size() == 2 &&
              selection.normalized_weights.size() == 2,
          "router selection weight size mismatch");
  require(std::abs((selection.normalized_weights[0] +
                    selection.normalized_weights[1]) -
                   1.0f) < 1e-6f,
          "router selection normalized sum mismatch");
  const auto swiglu_pair = iq36::apply_swiglu_pair(
      std::vector<float>{1.0f, -2.0f}, std::vector<float>{3.0f, 4.0f});
  require(swiglu_pair.size() == 2, "SwiGLU pair size mismatch");
  require(std::abs(swiglu_pair[0] - 2.1931758f) < 1e-6f,
          "SwiGLU pair value 0 mismatch");
  require(std::abs(swiglu_pair[1] - -0.9536231f) < 1e-6f,
          "SwiGLU pair value 1 mismatch");
  const auto weighted = iq36::apply_expert_weights(
      std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f},
      std::vector<float>{0.25f, 0.5f},
      2);
  require(weighted == std::vector<float>({0.25f, 0.5f, 1.5f, 2.0f}),
          "expert weight application mismatch");
  const auto aggregated = iq36::aggregate_experts(weighted, 2, 2);
  require(aggregated == std::vector<float>({1.75f, 2.5f}),
          "expert aggregation mismatch");
  const auto scaled =
      iq36::multiply_by_scalar(std::vector<float>{2.0f, -3.0f}, 0.5f);
  require(scaled == std::vector<float>({1.0f, -1.5f}),
          "scalar multiply mismatch");
  const auto delta = iq36::run_qwen36_linear_attention_delta_core(
      std::vector<float>{0.5f, 1.0f},
      std::vector<float>{1.0f, -1.0f},
      std::vector<float>{2.0f, 3.0f},
      std::vector<float>{0.0f},
      std::vector<float>{0.5f},
      std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f},
      std::vector<float>{0.0f, 0.0f},
      std::vector<float>{1.0f, 1.0f},
      0.0f);
  require(delta.attention_output.size() == 2,
          "linear attention output size mismatch");
  require(delta.recurrent_state == std::vector<float>({2.5f, 0.5f, 5.0f, 2.0f}),
          "linear attention state update mismatch");
  require(std::abs(delta.attention_output[0] - 1.2374369f) < 1e-6f,
          "linear attention output value 0 mismatch");
  require(std::abs(delta.attention_output[1] - 3.1819806f) < 1e-6f,
          "linear attention output value 1 mismatch");
  require(delta.final_output == std::vector<float>({0.0f, 0.0f}),
          "linear attention gated norm mismatch");
  const auto postconv = iq36::run_qwen36_linear_attention_postconv_core(
      std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
      std::vector<float>{0.0f},
      std::vector<float>{0.5f},
      std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f},
      std::vector<float>{0.0f, 0.0f},
      std::vector<float>{1.0f, 1.0f},
      0.0f);
  require(postconv.conv_output_silu.size() == 6,
          "linear attention postconv silu size mismatch");
  require(std::abs(postconv.q_conv[0] - 0.7310586f) < 1e-6f,
          "linear attention postconv q split mismatch");
  const float q_l2 = std::sqrt(
      postconv.q_conv_predelta[0] * postconv.q_conv_predelta[0] +
      postconv.q_conv_predelta[1] * postconv.q_conv_predelta[1]);
  require(std::abs(q_l2 - 1.0f) < 1e-6f,
          "linear attention postconv q L2 norm mismatch");
  require(postconv.v_conv_predelta.size() == 2,
          "linear attention postconv v split size mismatch");
  require(postconv.final_output == std::vector<float>({0.0f, 0.0f}),
          "linear attention postconv gated norm mismatch");

  std::cout << "iq36-self-test ok\n";
  return 0;
}
