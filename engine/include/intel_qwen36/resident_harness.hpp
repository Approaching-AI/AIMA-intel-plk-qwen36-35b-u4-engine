#pragma once

#include <cstddef>
#include <cstdint>
#include <chrono>
#include <functional>
#include <iosfwd>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace iq36 {

struct ModelContract {
  std::string workstream;
  std::string model_path;
  std::string model_sha256;
  int layers;
  int hidden_size;
  int experts;
  int active_experts;
  int context_length;
};

struct TargetContract {
  std::string host_alias;
  std::string cpu_model;
  std::string opencl_device;
};

// run_boundary returns this. NOTE: until the target adapter is wired (load the
// 35B model + oracle resident on PTL and run one box vs the oracle), run_boundary
// is a stub that returns a not-evaluated sentinel: cosine/relative_l2/
// kl_divergence/top1 are NaN. A NaN can never satisfy a `cosine >= 0.999` gate,
// so the unimplemented stub cannot be mistaken for a real teacher-forced pass.
struct BoundaryResult {
  std::string boundary_id;
  double cosine;
  double relative_l2;
  double kl_divergence;
  double top1;
  double elapsed_us;
  bool promoted;
};

struct OracleBundleStats {
  std::size_t token_topk_rows;
  std::size_t teacher_forced_distribution_rows;
  std::size_t boundary_input_rows;
  std::size_t boundary_output_rows;
};

struct ResidentTopKRow {
  std::int32_t token_id = 0;
  float logit = 0.0f;
};

struct ResidentTokenEvent {
  std::string case_id;
  std::string resident_session_id;
  int run_index = 0;
  std::string phase;
  std::size_t generated_index = 0;
  std::uint64_t resident_session_event_index = 0;
  std::uint64_t predicted_token_position = 0;
  std::vector<ResidentTopKRow> topk;
  std::uint64_t elapsed_ns = 0;
};

struct ResidentCaseResult {
  std::string case_id;
  std::vector<std::uint32_t> prompt_token_ids;
  std::vector<ResidentTopKRow> first_topk;
  std::vector<std::uint32_t> generated_token_ids;
  std::uint64_t prompt_prefill_ns = 0;
  std::uint64_t decode_continuation_ns = 0;
  std::uint64_t case_total_ns = 0;
};

struct ResidentDoneEvent {
  std::string resident_session_id;
  std::vector<ResidentCaseResult> cases;
  std::uint64_t max_new_tokens = 0;
  std::uint64_t process_total_ns = 0;
  std::uint64_t resident_session_token_count = 0;
  bool q4_plane_layout_enabled = false;
  bool selected_expert_down_q6_pair_dot_enabled = false;
  bool dense_q6_pair_dot_enabled = false;
  std::uint64_t resident_harness_load_ns = 0;
};

struct ResidentStreamingSessionConfig {
  std::string session_id;
  int run_index = 0;
  std::uint64_t max_new_tokens = 0;
  std::size_t expected_case_count = 0;
};

struct ResidentDecodeLoopConfig {
  std::string session_id_prefix;
  int run_index_base = 0;
  std::uint32_t initial_input_token_id = 0;
  std::vector<std::uint32_t> teacher_forced_token_ids;
  std::uint64_t max_new_tokens = 0;
  std::size_t session_count = 1;
  std::size_t expected_case_count = 1;
  bool emit_sse_events = false;
};

struct ResidentDecodeStepContext {
  std::size_t session_index = 0;
  std::size_t token_index = 0;
  int run_index = 0;
  std::uint32_t input_token_id = 0;
  std::uint64_t max_new_tokens = 0;
};

struct ResidentDecodeSessionContext {
  std::size_t session_index = 0;
  int run_index = 0;
  std::string session_id;
  std::uint64_t max_new_tokens = 0;
  std::uint64_t emitted_token_count = 0;
  std::uint64_t top1_match_count = 0;
  std::uint64_t topk_match_count = 0;
  std::vector<std::uint32_t> input_token_ids;
  std::vector<std::uint32_t> generated_token_ids;
  std::uint64_t session_elapsed_ns = 0;
};

struct ResidentDecodeLoopResult {
  std::size_t session_count = 0;
  std::uint64_t emitted_token_count = 0;
  std::uint64_t input_token_count = 0;
  std::uint64_t generated_token_count = 0;
  std::uint64_t process_total_ns = 0;
};

struct ResidentDecodeTokenResult {
  ResidentTokenEvent event;
  bool top1_matches = false;
  bool topk_matches = false;
};

using ResidentDecodeTokenCallback =
    std::function<ResidentDecodeTokenResult(const ResidentDecodeStepContext&)>;
using ResidentDecodeDoneCallback =
    std::function<ResidentDoneEvent(const ResidentDecodeSessionContext&)>;

const ModelContract& model_contract();
const TargetContract& target_contract();
std::vector<std::string> boundary_types();
std::vector<std::string> required_oracle_bundle_paths();
int parameterized_layer_count();

class ResidentHarness {
 public:
  void load(std::string model_path, std::string oracle_bundle_path);
  void swap_kernel(std::string boundary_id, std::string implementation_id);
  BoundaryResult run_boundary(std::string_view boundary_id) const;
  bool promote(std::string_view boundary_id) const;
  bool loaded() const;
  const OracleBundleStats& oracle_bundle_stats() const;
  void begin_streaming_session(const ResidentStreamingSessionConfig& config);
  bool streaming_session_active() const;
  void emit_sse_token_event(std::ostream& output,
                            const ResidentTokenEvent& event) const;
  void emit_sse_session_token_event(std::ostream& output,
                                    ResidentTokenEvent event);
  void emit_sse_done_event(std::ostream& output,
                           const ResidentDoneEvent& event) const;
  void emit_sse_session_done_event(std::ostream& output,
                                   ResidentDoneEvent event);

 private:
  struct StreamingSessionState {
    bool active = false;
    std::string session_id;
    int run_index = 0;
    std::uint64_t max_new_tokens = 0;
    std::size_t expected_case_count = 0;
    std::uint64_t emitted_token_count = 0;
  };

  bool loaded_ = false;
  OracleBundleStats oracle_bundle_stats_{};
  std::string model_path_;
  std::string oracle_bundle_path_;
  std::string last_boundary_id_;
  std::string last_implementation_id_;
  StreamingSessionState streaming_session_;
};

class ResidentDecodeLoop {
 public:
  explicit ResidentDecodeLoop(ResidentHarness& harness);
  ResidentDecodeLoopResult run(std::ostream& output,
                               const ResidentDecodeLoopConfig& config,
                               const ResidentDecodeTokenCallback& token_cb,
                               const ResidentDecodeDoneCallback& done_cb);

 private:
  ResidentHarness& harness_;
};

class ResidentGpuHotDecodeLoop {
 public:
  explicit ResidentGpuHotDecodeLoop(ResidentHarness& harness);

  template <typename Runtime>
  ResidentDecodeLoopResult run(std::ostream& output,
                               const ResidentDecodeLoopConfig& config,
                               Runtime& runtime) {
    if (config.session_id_prefix.empty()) {
      throw std::invalid_argument(
          "resident GPU hot decode session_id_prefix is required");
    }
    if (config.max_new_tokens == 0) {
      throw std::invalid_argument(
          "resident GPU hot decode max_new_tokens is required");
    }
    if (config.session_count == 0) {
      throw std::invalid_argument(
          "resident GPU hot decode session_count is required");
    }
    if (config.expected_case_count == 0) {
      throw std::invalid_argument(
          "resident GPU hot decode expected_case_count is required");
    }
    if (!config.teacher_forced_token_ids.empty() &&
        config.teacher_forced_token_ids.size() <
            static_cast<std::size_t>(config.max_new_tokens)) {
      throw std::invalid_argument(
          "resident GPU hot decode teacher-forced token list is shorter than "
          "max_new_tokens");
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
      generated_token_ids.reserve(
          static_cast<std::size_t>(config.max_new_tokens));
      for (std::uint64_t token_index = 0;
           token_index < config.max_new_tokens;
           ++token_index) {
        ResidentDecodeStepContext step;
        step.session_index = session_index;
        step.token_index = static_cast<std::size_t>(token_index);
        step.run_index = run_index;
        step.input_token_id = input_token_id;
        step.max_new_tokens = config.max_new_tokens;
        ResidentDecodeTokenResult token_result = runtime.decode_token(step);
        ResidentTokenEvent event = std::move(token_result.event);
        if (event.topk.empty()) {
          throw std::invalid_argument(
              "resident GPU hot decode token event top-k is empty");
        }
        const auto generated_token_id = event.topk[0].token_id;
        if (generated_token_id < 0) {
          throw std::invalid_argument(
              "resident GPU hot decode generated token id is negative");
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
      ResidentDoneEvent done = runtime.finish_session(done_context);
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

 private:
  ResidentHarness& harness_;
};

template <typename TokenFn, typename DoneFn>
class ResidentGpuHotDecodeLoopRuntimeAdapter {
 public:
  ResidentGpuHotDecodeLoopRuntimeAdapter(TokenFn& token_fn, DoneFn& done_fn)
      : token_fn_(token_fn), done_fn_(done_fn) {}

  ResidentDecodeTokenResult decode_token(
      const ResidentDecodeStepContext& context) {
    return token_fn_(context);
  }

  ResidentDoneEvent finish_session(
      const ResidentDecodeSessionContext& context) {
    return done_fn_(context);
  }

 private:
  TokenFn& token_fn_;
  DoneFn& done_fn_;
};

template <typename TokenFn, typename DoneFn>
ResidentGpuHotDecodeLoopRuntimeAdapter<TokenFn, DoneFn>
make_resident_gpu_hot_decode_loop_runtime(TokenFn& token_fn, DoneFn& done_fn) {
  return ResidentGpuHotDecodeLoopRuntimeAdapter<TokenFn, DoneFn>{
      token_fn, done_fn};
}

template <typename State>
class ResidentDecodeStateBank {
 public:
  ResidentDecodeStateBank(const State& initial_state,
                          std::size_t session_count)
      : states_(session_count, initial_state) {
    if (session_count == 0) {
      throw std::invalid_argument(
          "resident decode state bank session_count is required");
    }
  }

  std::size_t session_count() const { return states_.size(); }
  std::uint64_t reset_count() const { return reset_count_; }

  State& session_state(std::size_t session_index) {
    if (session_index >= states_.size()) {
      throw std::invalid_argument(
          "resident decode state bank session out of range");
    }
    return states_[session_index];
  }

  const State& session_state(std::size_t session_index) const {
    if (session_index >= states_.size()) {
      throw std::invalid_argument(
          "resident decode state bank session out of range");
    }
    return states_[session_index];
  }

  void reset_session(std::size_t session_index, const State& state) {
    session_state(session_index) = state;
    ++reset_count_;
  }

  void reset_all(const State& state) {
    for (auto& session_state : states_) {
      session_state = state;
    }
    reset_count_ += static_cast<std::uint64_t>(states_.size());
  }

 private:
  std::vector<State> states_;
  std::uint64_t reset_count_ = 0;
};

class ResidentSmallTensorStore {
 public:
  void clear() {
    handles_.clear();
    unique_handles_.clear();
    uploaded_bytes_ = 0;
    hits_ = 0;
    misses_ = 0;
  }

  std::uint64_t handle_count() const {
    return static_cast<std::uint64_t>(unique_handles_.size());
  }
  std::uint64_t key_count() const {
    return static_cast<std::uint64_t>(handles_.size());
  }
  std::uint64_t uploaded_bytes() const { return uploaded_bytes_; }
  std::uint64_t hits() const { return hits_; }
  std::uint64_t misses() const { return misses_; }

  void reset_hit_miss_counters() {
    hits_ = 0;
    misses_ = 0;
  }

  void insert(std::string key,
              std::uint64_t handle,
              std::uint64_t uploaded_bytes) {
    if (key.empty()) {
      throw std::invalid_argument("resident small tensor key is required");
    }
    if (handle == 0) {
      throw std::invalid_argument(
          "resident small tensor handle must be nonzero");
    }
    const auto found = handles_.find(key);
    if (found != handles_.end()) {
      if (found->second != handle) {
        throw std::invalid_argument(
            "resident small tensor key changed handles");
      }
      return;
    }
    handles_.emplace(std::move(key), handle);
    if (!has_unique_handle(handle)) {
      unique_handles_.push_back(handle);
    }
    uploaded_bytes_ += uploaded_bytes;
  }

  std::uint64_t find_handle(const std::string& key) {
    const auto found = handles_.find(key);
    if (found == handles_.end()) {
      ++misses_;
      return 0;
    }
    ++hits_;
    return found->second;
  }

  std::uint64_t lookup_handle(const std::string& key) {
    const auto handle = find_handle(key);
    if (handle == 0) {
      throw std::invalid_argument("resident small tensor handle missing");
    }
    return handle;
  }

 private:
  bool has_unique_handle(std::uint64_t handle) const {
    for (const auto stored : unique_handles_) {
      if (stored == handle) {
        return true;
      }
    }
    return false;
  }

  std::unordered_map<std::string, std::uint64_t> handles_;
  std::vector<std::uint64_t> unique_handles_;
  std::uint64_t uploaded_bytes_ = 0;
  std::uint64_t hits_ = 0;
  std::uint64_t misses_ = 0;
};

struct ResidentF32MatvecWeightHandle {
  std::uint64_t handle = 0;
  std::uint64_t rows = 0;
  std::uint64_t cols = 0;
};

class ResidentF32MatvecWeightStore {
 public:
  void clear() {
    handles_.clear();
    uploaded_bytes_ = 0;
    hits_ = 0;
    misses_ = 0;
  }

  std::uint64_t handle_count() const {
    return static_cast<std::uint64_t>(handles_.size());
  }
  std::uint64_t uploaded_bytes() const { return uploaded_bytes_; }
  std::uint64_t hits() const { return hits_; }
  std::uint64_t misses() const { return misses_; }

  void reset_hit_miss_counters() {
    hits_ = 0;
    misses_ = 0;
  }

  void insert(std::string key,
              std::uint64_t handle,
              std::uint64_t rows,
              std::uint64_t cols,
              std::uint64_t uploaded_bytes) {
    if (key.empty()) {
      throw std::invalid_argument(
          "resident F32 matvec weight key is required");
    }
    if (handle == 0) {
      throw std::invalid_argument(
          "resident F32 matvec weight handle must be nonzero");
    }
    if (rows == 0 || cols == 0) {
      throw std::invalid_argument(
          "resident F32 matvec weight dimensions are required");
    }
    const auto found = handles_.find(key);
    if (found != handles_.end()) {
      if (found->second.handle != handle || found->second.rows != rows ||
          found->second.cols != cols) {
        throw std::invalid_argument(
            "resident F32 matvec weight key changed handles or shape");
      }
      return;
    }
    handles_.emplace(
        std::move(key), ResidentF32MatvecWeightHandle{handle, rows, cols});
    uploaded_bytes_ += uploaded_bytes;
  }

  std::uint64_t find_handle(const std::string& key,
                            std::uint64_t rows,
                            std::uint64_t cols) {
    const auto found = handles_.find(key);
    if (found == handles_.end()) {
      ++misses_;
      return 0;
    }
    if (found->second.rows != rows || found->second.cols != cols) {
      throw std::invalid_argument(
          "resident F32 matvec weight shape mismatch");
    }
    ++hits_;
    return found->second.handle;
  }

  std::uint64_t lookup_handle(const std::string& key,
                              std::uint64_t rows,
                              std::uint64_t cols) {
    const auto handle = find_handle(key, rows, cols);
    if (handle == 0) {
      throw std::invalid_argument(
          "resident F32 matvec weight handle missing");
    }
    return handle;
  }

 private:
  std::unordered_map<std::string, ResidentF32MatvecWeightHandle> handles_;
  std::uint64_t uploaded_bytes_ = 0;
  std::uint64_t hits_ = 0;
  std::uint64_t misses_ = 0;
};

struct ResidentQ6RowstripeWeightHandle {
  std::uint64_t handle = 0;
  std::uint64_t rows = 0;
  std::uint64_t row_nbytes = 0;
  std::uint64_t blocks_per_row = 0;
};

class ResidentQ6RowstripeWeightStore {
 public:
  void clear() {
    handles_.clear();
    uploaded_bytes_ = 0;
    hits_ = 0;
    misses_ = 0;
  }

  std::uint64_t handle_count() const {
    return static_cast<std::uint64_t>(handles_.size());
  }
  std::uint64_t uploaded_bytes() const { return uploaded_bytes_; }
  std::uint64_t hits() const { return hits_; }
  std::uint64_t misses() const { return misses_; }

  void reset_hit_miss_counters() {
    hits_ = 0;
    misses_ = 0;
  }

  void insert(std::string key,
              std::uint64_t handle,
              std::uint64_t rows,
              std::uint64_t row_nbytes,
              std::uint64_t blocks_per_row,
              std::uint64_t uploaded_bytes) {
    if (key.empty()) {
      throw std::invalid_argument(
          "resident Q6 rowstripe weight key is required");
    }
    if (handle == 0) {
      throw std::invalid_argument(
          "resident Q6 rowstripe weight handle must be nonzero");
    }
    if (rows == 0 || row_nbytes == 0 || blocks_per_row == 0) {
      throw std::invalid_argument(
          "resident Q6 rowstripe weight shape is required");
    }
    const auto found = handles_.find(key);
    if (found != handles_.end()) {
      if (found->second.handle != handle || found->second.rows != rows ||
          found->second.row_nbytes != row_nbytes ||
          found->second.blocks_per_row != blocks_per_row) {
        throw std::invalid_argument(
            "resident Q6 rowstripe weight key changed handles or shape");
      }
      return;
    }
    handles_.emplace(
        std::move(key),
        ResidentQ6RowstripeWeightHandle{
            handle, rows, row_nbytes, blocks_per_row});
    uploaded_bytes_ += uploaded_bytes;
  }

  std::uint64_t find_handle(const std::string& key,
                            std::uint64_t rows,
                            std::uint64_t row_nbytes,
                            std::uint64_t blocks_per_row) {
    const auto found = handles_.find(key);
    if (found == handles_.end()) {
      ++misses_;
      return 0;
    }
    if (found->second.rows != rows ||
        found->second.row_nbytes != row_nbytes ||
        found->second.blocks_per_row != blocks_per_row) {
      throw std::invalid_argument(
          "resident Q6 rowstripe weight shape mismatch");
    }
    ++hits_;
    return found->second.handle;
  }

  std::uint64_t lookup_handle(const std::string& key,
                              std::uint64_t rows,
                              std::uint64_t row_nbytes,
                              std::uint64_t blocks_per_row) {
    const auto handle = find_handle(key, rows, row_nbytes, blocks_per_row);
    if (handle == 0) {
      throw std::invalid_argument(
          "resident Q6 rowstripe weight handle missing");
    }
    return handle;
  }

 private:
  std::unordered_map<std::string, ResidentQ6RowstripeWeightHandle> handles_;
  std::uint64_t uploaded_bytes_ = 0;
  std::uint64_t hits_ = 0;
  std::uint64_t misses_ = 0;
};

struct ResidentSelectedQ6WeightHandle {
  std::uint64_t handle = 0;
  std::uint64_t rows_per_expert = 0;
  std::uint64_t row_nbytes = 0;
  std::uint64_t blocks_per_row = 0;
  std::uint64_t expert_count = 0;
  bool rowstripe = false;
};

class ResidentSelectedQ6WeightStore {
 public:
  void clear() {
    handles_.clear();
    uploaded_bytes_ = 0;
    hits_ = 0;
    misses_ = 0;
  }

  std::uint64_t handle_count() const {
    return static_cast<std::uint64_t>(handles_.size());
  }
  std::uint64_t uploaded_bytes() const { return uploaded_bytes_; }
  std::uint64_t hits() const { return hits_; }
  std::uint64_t misses() const { return misses_; }

  void reset_hit_miss_counters() {
    hits_ = 0;
    misses_ = 0;
  }

  void insert(std::string key,
              std::uint64_t handle,
              std::uint64_t rows_per_expert,
              std::uint64_t row_nbytes,
              std::uint64_t blocks_per_row,
              std::uint64_t expert_count,
              bool rowstripe,
              std::uint64_t uploaded_bytes) {
    if (key.empty()) {
      throw std::invalid_argument(
          "resident selected Q6 weight key is required");
    }
    if (handle == 0) {
      throw std::invalid_argument(
          "resident selected Q6 weight handle must be nonzero");
    }
    if (rows_per_expert == 0 || row_nbytes == 0 ||
        blocks_per_row == 0 || expert_count == 0) {
      throw std::invalid_argument(
          "resident selected Q6 weight shape is required");
    }
    const auto found = handles_.find(key);
    if (found != handles_.end()) {
      if (found->second.handle != handle ||
          found->second.rows_per_expert != rows_per_expert ||
          found->second.row_nbytes != row_nbytes ||
          found->second.blocks_per_row != blocks_per_row ||
          found->second.expert_count != expert_count ||
          found->second.rowstripe != rowstripe) {
        throw std::invalid_argument(
            "resident selected Q6 weight key changed handles or shape");
      }
      return;
    }
    handles_.emplace(
        std::move(key),
        ResidentSelectedQ6WeightHandle{
            handle, rows_per_expert, row_nbytes, blocks_per_row,
            expert_count, rowstripe});
    uploaded_bytes_ += uploaded_bytes;
  }

  std::uint64_t find_handle(const std::string& key,
                            std::uint64_t rows_per_expert,
                            std::uint64_t row_nbytes,
                            std::uint64_t blocks_per_row,
                            std::uint64_t expert_count,
                            bool rowstripe) {
    const auto found = handles_.find(key);
    if (found == handles_.end()) {
      ++misses_;
      return 0;
    }
    if (found->second.rows_per_expert != rows_per_expert ||
        found->second.row_nbytes != row_nbytes ||
        found->second.blocks_per_row != blocks_per_row ||
        found->second.expert_count != expert_count ||
        found->second.rowstripe != rowstripe) {
      throw std::invalid_argument(
          "resident selected Q6 weight shape mismatch");
    }
    ++hits_;
    return found->second.handle;
  }

  std::uint64_t lookup_handle(const std::string& key,
                              std::uint64_t rows_per_expert,
                              std::uint64_t row_nbytes,
                              std::uint64_t blocks_per_row,
                              std::uint64_t expert_count,
                              bool rowstripe) {
    const auto handle = find_handle(
        key, rows_per_expert, row_nbytes, blocks_per_row, expert_count,
        rowstripe);
    if (handle == 0) {
      throw std::invalid_argument(
          "resident selected Q6 weight handle missing");
    }
    return handle;
  }

 private:
  std::unordered_map<std::string, ResidentSelectedQ6WeightHandle> handles_;
  std::uint64_t uploaded_bytes_ = 0;
  std::uint64_t hits_ = 0;
  std::uint64_t misses_ = 0;
};

struct ResidentRawQ6WeightHandle {
  std::uint64_t handle = 0;
  std::uint64_t rows = 0;
  std::uint64_t row_nbytes = 0;
  std::uint64_t blocks_per_row = 0;
};

class ResidentRawQ6WeightStore {
 public:
  void clear() {
    handles_.clear();
    uploaded_bytes_ = 0;
    hits_ = 0;
    misses_ = 0;
  }

  std::uint64_t handle_count() const {
    return static_cast<std::uint64_t>(handles_.size());
  }
  std::uint64_t uploaded_bytes() const { return uploaded_bytes_; }
  std::uint64_t hits() const { return hits_; }
  std::uint64_t misses() const { return misses_; }

  void reset_hit_miss_counters() {
    hits_ = 0;
    misses_ = 0;
  }

  void insert(std::string key,
              std::uint64_t handle,
              std::uint64_t rows,
              std::uint64_t row_nbytes,
              std::uint64_t blocks_per_row,
              std::uint64_t uploaded_bytes) {
    if (key.empty()) {
      throw std::invalid_argument("resident raw Q6 weight key is required");
    }
    if (handle == 0) {
      throw std::invalid_argument(
          "resident raw Q6 weight handle must be nonzero");
    }
    if (rows == 0 || row_nbytes == 0 || blocks_per_row == 0) {
      throw std::invalid_argument(
          "resident raw Q6 weight shape is required");
    }
    const auto found = handles_.find(key);
    if (found != handles_.end()) {
      if (found->second.handle != handle || found->second.rows != rows ||
          found->second.row_nbytes != row_nbytes ||
          found->second.blocks_per_row != blocks_per_row) {
        throw std::invalid_argument(
            "resident raw Q6 weight key changed handles or shape");
      }
      return;
    }
    handles_.emplace(
        std::move(key),
        ResidentRawQ6WeightHandle{handle, rows, row_nbytes, blocks_per_row});
    uploaded_bytes_ += uploaded_bytes;
  }

  std::uint64_t find_handle(const std::string& key,
                            std::uint64_t rows,
                            std::uint64_t row_nbytes,
                            std::uint64_t blocks_per_row) {
    const auto found = handles_.find(key);
    if (found == handles_.end()) {
      ++misses_;
      return 0;
    }
    if (found->second.rows != rows ||
        found->second.row_nbytes != row_nbytes ||
        found->second.blocks_per_row != blocks_per_row) {
      throw std::invalid_argument(
          "resident raw Q6 weight shape mismatch");
    }
    ++hits_;
    return found->second.handle;
  }

  std::uint64_t lookup_handle(const std::string& key,
                              std::uint64_t rows,
                              std::uint64_t row_nbytes,
                              std::uint64_t blocks_per_row) {
    const auto handle = find_handle(key, rows, row_nbytes, blocks_per_row);
    if (handle == 0) {
      throw std::invalid_argument("resident raw Q6 weight handle missing");
    }
    return handle;
  }

 private:
  std::unordered_map<std::string, ResidentRawQ6WeightHandle> handles_;
  std::uint64_t uploaded_bytes_ = 0;
  std::uint64_t hits_ = 0;
  std::uint64_t misses_ = 0;
};

struct ResidentPackedQ4WeightHandle {
  std::uint64_t handle = 0;
  std::uint64_t rows = 0;
  std::uint64_t blocks_per_row = 0;
};

class ResidentPackedQ4WeightStore {
 public:
  void clear() {
    handles_.clear();
    uploaded_bytes_ = 0;
    hits_ = 0;
    misses_ = 0;
  }

  std::uint64_t handle_count() const {
    return static_cast<std::uint64_t>(handles_.size());
  }
  std::uint64_t uploaded_bytes() const { return uploaded_bytes_; }
  std::uint64_t hits() const { return hits_; }
  std::uint64_t misses() const { return misses_; }

  void reset_hit_miss_counters() {
    hits_ = 0;
    misses_ = 0;
  }

  void insert(std::string key,
              std::uint64_t handle,
              std::uint64_t rows,
              std::uint64_t blocks_per_row,
              std::uint64_t uploaded_bytes) {
    if (key.empty()) {
      throw std::invalid_argument(
          "resident packed Q4 weight key is required");
    }
    if (handle == 0) {
      throw std::invalid_argument(
          "resident packed Q4 weight handle must be nonzero");
    }
    if (rows == 0 || blocks_per_row == 0) {
      throw std::invalid_argument(
          "resident packed Q4 weight shape is required");
    }
    const auto found = handles_.find(key);
    if (found != handles_.end()) {
      if (found->second.handle != handle || found->second.rows != rows ||
          found->second.blocks_per_row != blocks_per_row) {
        throw std::invalid_argument(
            "resident packed Q4 weight key changed handles or shape");
      }
      return;
    }
    handles_.emplace(
        std::move(key),
        ResidentPackedQ4WeightHandle{handle, rows, blocks_per_row});
    uploaded_bytes_ += uploaded_bytes;
  }

  std::uint64_t find_handle(const std::string& key,
                            std::uint64_t rows,
                            std::uint64_t blocks_per_row) {
    const auto found = handles_.find(key);
    if (found == handles_.end()) {
      ++misses_;
      return 0;
    }
    if (found->second.rows != rows ||
        found->second.blocks_per_row != blocks_per_row) {
      throw std::invalid_argument(
          "resident packed Q4 weight shape mismatch");
    }
    ++hits_;
    return found->second.handle;
  }

  std::uint64_t lookup_handle(const std::string& key,
                              std::uint64_t rows,
                              std::uint64_t blocks_per_row) {
    const auto handle = find_handle(key, rows, blocks_per_row);
    if (handle == 0) {
      throw std::invalid_argument(
          "resident packed Q4 weight handle missing");
    }
    return handle;
  }

 private:
  std::unordered_map<std::string, ResidentPackedQ4WeightHandle> handles_;
  std::uint64_t uploaded_bytes_ = 0;
  std::uint64_t hits_ = 0;
  std::uint64_t misses_ = 0;
};

struct ResidentQ4CpuOrderWeightHandle {
  std::uint64_t handle = 0;
  std::uint64_t rows = 0;
  std::uint64_t blocks_per_row = 0;
};

class ResidentQ4CpuOrderWeightStore {
 public:
  void clear() {
    handles_.clear();
    uploaded_bytes_ = 0;
    hits_ = 0;
    misses_ = 0;
  }

  std::uint64_t handle_count() const {
    return static_cast<std::uint64_t>(handles_.size());
  }
  std::uint64_t uploaded_bytes() const { return uploaded_bytes_; }
  std::uint64_t hits() const { return hits_; }
  std::uint64_t misses() const { return misses_; }

  void reset_hit_miss_counters() {
    hits_ = 0;
    misses_ = 0;
  }

  void insert(std::string key,
              std::uint64_t handle,
              std::uint64_t rows,
              std::uint64_t blocks_per_row,
              std::uint64_t uploaded_bytes) {
    if (key.empty()) {
      throw std::invalid_argument(
          "resident Q4 CPU-order weight key is required");
    }
    if (handle == 0) {
      throw std::invalid_argument(
          "resident Q4 CPU-order weight handle must be nonzero");
    }
    if (rows == 0 || blocks_per_row == 0) {
      throw std::invalid_argument(
          "resident Q4 CPU-order weight shape is required");
    }
    const auto found = handles_.find(key);
    if (found != handles_.end()) {
      if (found->second.handle != handle || found->second.rows != rows ||
          found->second.blocks_per_row != blocks_per_row) {
        throw std::invalid_argument(
            "resident Q4 CPU-order weight key changed handles or shape");
      }
      return;
    }
    handles_.emplace(
        std::move(key),
        ResidentQ4CpuOrderWeightHandle{handle, rows, blocks_per_row});
    uploaded_bytes_ += uploaded_bytes;
  }

  std::uint64_t find_handle(const std::string& key,
                            std::uint64_t rows,
                            std::uint64_t blocks_per_row) {
    const auto found = handles_.find(key);
    if (found == handles_.end()) {
      ++misses_;
      return 0;
    }
    if (found->second.rows != rows ||
        found->second.blocks_per_row != blocks_per_row) {
      throw std::invalid_argument(
          "resident Q4 CPU-order weight shape mismatch");
    }
    ++hits_;
    return found->second.handle;
  }

  std::uint64_t lookup_handle(const std::string& key,
                              std::uint64_t rows,
                              std::uint64_t blocks_per_row) {
    const auto handle = find_handle(key, rows, blocks_per_row);
    if (handle == 0) {
      throw std::invalid_argument(
          "resident Q4 CPU-order weight handle missing");
    }
    return handle;
  }

 private:
  std::unordered_map<std::string, ResidentQ4CpuOrderWeightHandle> handles_;
  std::uint64_t uploaded_bytes_ = 0;
  std::uint64_t hits_ = 0;
  std::uint64_t misses_ = 0;
};

struct ResidentLmHeadQ6WeightHandle {
  std::uint64_t handle = 0;
  std::uint64_t rows = 0;
  std::uint64_t blocks_per_row = 0;
};

struct ResidentLmHeadNormHandle {
  std::uint64_t handle = 0;
  std::uint64_t elements = 0;
};

class ResidentLmHeadWeightStore {
 public:
  void clear() {
    q6_handles_.clear();
    norm_handles_.clear();
    q6_uploaded_bytes_ = 0;
    norm_uploaded_bytes_ = 0;
    q6_hits_ = 0;
    q6_misses_ = 0;
    norm_hits_ = 0;
    norm_misses_ = 0;
  }

  std::uint64_t q6_handle_count() const {
    return static_cast<std::uint64_t>(q6_handles_.size());
  }
  std::uint64_t norm_handle_count() const {
    return static_cast<std::uint64_t>(norm_handles_.size());
  }
  std::uint64_t handle_count() const {
    return q6_handle_count() + norm_handle_count();
  }
  std::uint64_t q6_uploaded_bytes() const { return q6_uploaded_bytes_; }
  std::uint64_t norm_uploaded_bytes() const { return norm_uploaded_bytes_; }
  std::uint64_t uploaded_bytes() const {
    return q6_uploaded_bytes_ + norm_uploaded_bytes_;
  }
  std::uint64_t q6_hits() const { return q6_hits_; }
  std::uint64_t q6_misses() const { return q6_misses_; }
  std::uint64_t norm_hits() const { return norm_hits_; }
  std::uint64_t norm_misses() const { return norm_misses_; }
  std::uint64_t hits() const { return q6_hits_ + norm_hits_; }
  std::uint64_t misses() const { return q6_misses_ + norm_misses_; }

  void reset_hit_miss_counters() {
    q6_hits_ = 0;
    q6_misses_ = 0;
    norm_hits_ = 0;
    norm_misses_ = 0;
  }

  void insert_q6(std::string key,
                 std::uint64_t handle,
                 std::uint64_t rows,
                 std::uint64_t blocks_per_row,
                 std::uint64_t uploaded_bytes) {
    if (key.empty()) {
      throw std::invalid_argument("resident LM-head Q6 key is required");
    }
    if (handle == 0) {
      throw std::invalid_argument(
          "resident LM-head Q6 handle must be nonzero");
    }
    if (rows == 0 || blocks_per_row == 0) {
      throw std::invalid_argument("resident LM-head Q6 shape is required");
    }
    const auto found = q6_handles_.find(key);
    if (found != q6_handles_.end()) {
      if (found->second.handle != handle || found->second.rows != rows ||
          found->second.blocks_per_row != blocks_per_row) {
        throw std::invalid_argument(
            "resident LM-head Q6 key changed handles or shape");
      }
      return;
    }
    q6_handles_.emplace(
        std::move(key),
        ResidentLmHeadQ6WeightHandle{handle, rows, blocks_per_row});
    q6_uploaded_bytes_ += uploaded_bytes;
  }

  void insert_norm(std::string key,
                   std::uint64_t handle,
                   std::uint64_t elements,
                   std::uint64_t uploaded_bytes) {
    if (key.empty()) {
      throw std::invalid_argument("resident LM-head norm key is required");
    }
    if (handle == 0) {
      throw std::invalid_argument(
          "resident LM-head norm handle must be nonzero");
    }
    if (elements == 0) {
      throw std::invalid_argument(
          "resident LM-head norm element count is required");
    }
    const auto found = norm_handles_.find(key);
    if (found != norm_handles_.end()) {
      if (found->second.handle != handle ||
          found->second.elements != elements) {
        throw std::invalid_argument(
            "resident LM-head norm key changed handles or shape");
      }
      return;
    }
    norm_handles_.emplace(
        std::move(key), ResidentLmHeadNormHandle{handle, elements});
    norm_uploaded_bytes_ += uploaded_bytes;
  }

  std::uint64_t find_q6(const std::string& key,
                        std::uint64_t rows,
                        std::uint64_t blocks_per_row) {
    const auto found = q6_handles_.find(key);
    if (found == q6_handles_.end()) {
      ++q6_misses_;
      return 0;
    }
    if (found->second.rows != rows ||
        found->second.blocks_per_row != blocks_per_row) {
      throw std::invalid_argument("resident LM-head Q6 shape mismatch");
    }
    ++q6_hits_;
    return found->second.handle;
  }

  std::uint64_t lookup_q6(const std::string& key,
                          std::uint64_t rows,
                          std::uint64_t blocks_per_row) {
    const auto handle = find_q6(key, rows, blocks_per_row);
    if (handle == 0) {
      throw std::invalid_argument("resident LM-head Q6 handle missing");
    }
    return handle;
  }

  std::uint64_t find_norm(const std::string& key, std::uint64_t elements) {
    const auto found = norm_handles_.find(key);
    if (found == norm_handles_.end()) {
      ++norm_misses_;
      return 0;
    }
    if (found->second.elements != elements) {
      throw std::invalid_argument("resident LM-head norm shape mismatch");
    }
    ++norm_hits_;
    return found->second.handle;
  }

  std::uint64_t lookup_norm(const std::string& key, std::uint64_t elements) {
    const auto handle = find_norm(key, elements);
    if (handle == 0) {
      throw std::invalid_argument("resident LM-head norm handle missing");
    }
    return handle;
  }

 private:
  std::unordered_map<std::string, ResidentLmHeadQ6WeightHandle> q6_handles_;
  std::unordered_map<std::string, ResidentLmHeadNormHandle> norm_handles_;
  std::uint64_t q6_uploaded_bytes_ = 0;
  std::uint64_t norm_uploaded_bytes_ = 0;
  std::uint64_t q6_hits_ = 0;
  std::uint64_t q6_misses_ = 0;
  std::uint64_t norm_hits_ = 0;
  std::uint64_t norm_misses_ = 0;
};

struct ResidentF32ConvWeightHandle {
  std::uint64_t handle = 0;
  std::uint64_t rows = 0;
  std::uint64_t kernel_size = 0;
};

class ResidentF32ConvWeightStore {
 public:
  void clear() {
    handles_.clear();
    uploaded_bytes_ = 0;
    hits_ = 0;
    misses_ = 0;
  }

  std::uint64_t handle_count() const {
    return static_cast<std::uint64_t>(handles_.size());
  }
  std::uint64_t uploaded_bytes() const { return uploaded_bytes_; }
  std::uint64_t hits() const { return hits_; }
  std::uint64_t misses() const { return misses_; }

  void reset_hit_miss_counters() {
    hits_ = 0;
    misses_ = 0;
  }

  void insert(std::string key,
              std::uint64_t handle,
              std::uint64_t rows,
              std::uint64_t kernel_size,
              std::uint64_t uploaded_bytes) {
    if (key.empty()) {
      throw std::invalid_argument(
          "resident F32 conv weight key is required");
    }
    if (handle == 0) {
      throw std::invalid_argument(
          "resident F32 conv weight handle must be nonzero");
    }
    if (rows == 0 || kernel_size == 0) {
      throw std::invalid_argument(
          "resident F32 conv weight shape is required");
    }
    const auto found = handles_.find(key);
    if (found != handles_.end()) {
      if (found->second.handle != handle || found->second.rows != rows ||
          found->second.kernel_size != kernel_size) {
        throw std::invalid_argument(
            "resident F32 conv weight key changed handles or shape");
      }
      return;
    }
    handles_.emplace(
        std::move(key),
        ResidentF32ConvWeightHandle{handle, rows, kernel_size});
    uploaded_bytes_ += uploaded_bytes;
  }

  std::uint64_t find_handle(const std::string& key,
                            std::uint64_t rows,
                            std::uint64_t kernel_size) {
    const auto found = handles_.find(key);
    if (found == handles_.end()) {
      ++misses_;
      return 0;
    }
    if (found->second.rows != rows ||
        found->second.kernel_size != kernel_size) {
      throw std::invalid_argument(
          "resident F32 conv weight shape mismatch");
    }
    ++hits_;
    return found->second.handle;
  }

  std::uint64_t lookup_handle(const std::string& key,
                              std::uint64_t rows,
                              std::uint64_t kernel_size) {
    const auto handle = find_handle(key, rows, kernel_size);
    if (handle == 0) {
      throw std::invalid_argument(
          "resident F32 conv weight handle missing");
    }
    return handle;
  }

 private:
  std::unordered_map<std::string, ResidentF32ConvWeightHandle> handles_;
  std::uint64_t uploaded_bytes_ = 0;
  std::uint64_t hits_ = 0;
  std::uint64_t misses_ = 0;
};

class ResidentDeviceStateHandleBank {
 public:
  void reset(std::size_t layer_count) {
    recurrent_handles_.assign(layer_count, 0);
    conv_handles_.assign(layer_count, 0);
    conv_next_handles_.assign(layer_count, 0);
    uploaded_bytes_ = 0;
    hits_ = 0;
    misses_ = 0;
  }

  void clear() {
    recurrent_handles_.clear();
    conv_handles_.clear();
    conv_next_handles_.clear();
    uploaded_bytes_ = 0;
    hits_ = 0;
    misses_ = 0;
  }

  std::size_t layer_count() const { return recurrent_handles_.size(); }
  std::uint64_t uploaded_bytes() const { return uploaded_bytes_; }
  std::uint64_t hits() const { return hits_; }
  std::uint64_t misses() const { return misses_; }

  void reset_hit_miss_counters() {
    hits_ = 0;
    misses_ = 0;
  }

  std::uint64_t handle_count() const {
    return count_nonzero(recurrent_handles_) + count_nonzero(conv_handles_) +
           count_nonzero(conv_next_handles_);
  }

  void set_layer_handles(std::size_t layer,
                         std::uint64_t recurrent_handle,
                         std::uint64_t conv_handle,
                         std::uint64_t conv_next_handle,
                         std::uint64_t uploaded_bytes) {
    check_layer(layer);
    if (recurrent_handle == 0 || conv_handle == 0 || conv_next_handle == 0) {
      throw std::invalid_argument(
          "resident device state handle bank requires nonzero handles");
    }
    recurrent_handles_[layer] = recurrent_handle;
    conv_handles_[layer] = conv_handle;
    conv_next_handles_[layer] = conv_next_handle;
    uploaded_bytes_ += uploaded_bytes;
    misses_ += 3;
  }

  void set_layer_recurrent_handle(std::size_t layer,
                                  std::uint64_t recurrent_handle,
                                  std::uint64_t uploaded_bytes) {
    check_layer(layer);
    if (recurrent_handle == 0) {
      throw std::invalid_argument(
          "resident device state handle bank requires nonzero recurrent handle");
    }
    recurrent_handles_[layer] = recurrent_handle;
    uploaded_bytes_ += uploaded_bytes;
    misses_ += 1;
  }

  void set_layer_conv_handles(std::size_t layer,
                              std::uint64_t conv_handle,
                              std::uint64_t conv_next_handle,
                              std::uint64_t uploaded_bytes) {
    check_layer(layer);
    if (conv_handle == 0 || conv_next_handle == 0) {
      throw std::invalid_argument(
          "resident device state handle bank requires nonzero conv handles");
    }
    conv_handles_[layer] = conv_handle;
    conv_next_handles_[layer] = conv_next_handle;
    uploaded_bytes_ += uploaded_bytes;
    misses_ += 2;
  }

  std::uint64_t recurrent_handle(std::size_t layer) {
    const auto handle = handle_at(recurrent_handles_, layer);
    ++hits_;
    return handle;
  }

  std::uint64_t conv_handle(std::size_t layer) {
    const auto handle = handle_at(conv_handles_, layer);
    ++hits_;
    return handle;
  }

  std::uint64_t conv_next_handle(std::size_t layer) {
    const auto handle = handle_at(conv_next_handles_, layer);
    ++hits_;
    return handle;
  }

  void swap_conv_handles(std::size_t layer) {
    check_layer(layer);
    std::swap(conv_handles_[layer], conv_next_handles_[layer]);
  }

 private:
  static std::uint64_t count_nonzero(
      const std::vector<std::uint64_t>& handles) {
    std::uint64_t count = 0;
    for (const auto handle : handles) {
      if (handle != 0) {
        ++count;
      }
    }
    return count;
  }

  void check_layer(std::size_t layer) const {
    if (layer >= recurrent_handles_.size() ||
        layer >= conv_handles_.size() ||
        layer >= conv_next_handles_.size()) {
      throw std::invalid_argument(
          "resident device state handle bank layer out of range");
    }
  }

  std::uint64_t handle_at(const std::vector<std::uint64_t>& handles,
                          std::size_t layer) const {
    check_layer(layer);
    const auto handle = handles[layer];
    if (handle == 0) {
      throw std::invalid_argument(
          "resident device state handle bank handle missing");
    }
    return handle;
  }

  std::vector<std::uint64_t> recurrent_handles_;
  std::vector<std::uint64_t> conv_handles_;
  std::vector<std::uint64_t> conv_next_handles_;
  std::uint64_t uploaded_bytes_ = 0;
  std::uint64_t hits_ = 0;
  std::uint64_t misses_ = 0;
};

}  // namespace iq36
