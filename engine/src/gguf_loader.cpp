#include "intel_qwen36/gguf_loader.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <numeric>
#include <regex>
#include <stdexcept>
#include <thread>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace iq36 {
namespace {

constexpr std::uint64_t kExpectedModelSize = 21166755168ULL;
constexpr std::uint64_t kMaxStoredMetadataArrayLength = 1024;
constexpr std::uint64_t kMaxDecodedRowCacheBytes = 4ULL * 1024ULL * 1024ULL;

struct Q4KBlockMeta {
  float d = 0.0f;
  float dmin = 0.0f;
  std::array<std::uint8_t, 8> scales{};
  std::array<std::uint8_t, 8> mins{};
};

struct Q4KPlaneRows {
  std::uint64_t row_count = 0;
  std::uint64_t blocks_per_row = 0;
  std::vector<std::uint8_t> q4;
  std::vector<std::uint8_t> scales;
  std::vector<std::uint8_t> mins;
  std::vector<float> d_values;
  std::vector<float> dmin_values;
};

std::uint64_t q4_plane_cached_bytes(const Q4KPlaneRows& plane) {
  return static_cast<std::uint64_t>(plane.q4.size() + plane.scales.size() +
                                    plane.mins.size() +
                                    plane.d_values.size() * sizeof(float) +
                                    plane.dmin_values.size() * sizeof(float));
}

struct ResidentCacheState {
  bool enabled = false;
  std::unordered_map<std::string, std::vector<float>> decoded_rows;
  std::unordered_map<std::string, std::vector<std::uint8_t>> tensor_payloads;
  std::unordered_map<std::string, std::vector<Q4KBlockMeta>> q4_block_meta;
  std::unordered_map<std::string, Q4KPlaneRows> q4_plane_rows;
  std::unordered_map<std::string, std::vector<std::uint8_t>> expert_slices;
  ResidentTensorCacheStats stats;
};

struct MatvecProfileState {
  bool enabled = false;
  std::unordered_map<std::string, MatvecProfileRow> rows;
};

struct ExpertSliceMatvecState {
  bool enabled = false;
  int thread_count = 1;
};

struct DenseMatvecState {
  bool enabled = false;
  int thread_count = 1;
  std::uint64_t min_rows = 1024;
};

struct DenseMatvecPayloadCacheState {
  bool enabled = false;
};

struct DenseQ4DirectDotState {
  bool enabled = false;
};

struct DenseQ4PairDotState {
  bool enabled = false;
};

struct DenseQ6DirectDotState {
  bool enabled = false;
};

struct DenseQ6PairDotState {
  bool enabled = false;
};

struct LmHeadQ6PairDotState {
  bool enabled = false;
};

struct Q4DirectMinsumPairState {
  bool enabled = false;
};

struct Q4BlockMetaCacheState {
  bool enabled = false;
};

struct Q4PlaneLayoutState {
  bool enabled = false;
};

struct DenseQ4PlanePairDotState {
  bool enabled = false;
};

struct SmallQ4DirectDotState {
  bool enabled = false;
};

struct MatvecQ8InputReuseState {
  bool enabled = false;
};

struct SharedParallelExecutorState {
  bool enabled = false;
};

struct SharedExpertGateUpFusedState {
  bool enabled = false;
};

struct SelectedExpertFfnState {
  bool enabled = false;
  int thread_count = 1;
};

struct SelectedExpertMinimalOutputsState {
  bool enabled = false;
};

struct SelectedExpertSliceCacheState {
  bool enabled = false;
};

struct SelectedExpertDownSliceCacheState {
  bool enabled = false;
};

struct SelectedExpertDownExpertMajorState {
  bool enabled = false;
};

struct SelectedExpertDownQ4PairDotState {
  bool enabled = false;
};

struct SelectedExpertDownQ6PairDotState {
  bool enabled = false;
};

struct SelectedGateQ4DirectDotState {
  bool enabled = false;
};

struct SelectedGateQ4PairDotState {
  bool enabled = false;
};

struct SelectedGateQ4PairSumDotState {
  bool enabled = false;
};

struct SelectedGateQ4PlanePairDotState {
  bool enabled = false;
};

struct SelectedExpertFfnRouteResult {
  std::vector<float> selected_gate_up;
  std::vector<float> selected_swiglu;
  std::vector<float> selected_down;
  std::vector<float> weighted_selected_down;
  std::vector<float> moe_out;
};

struct SharedExpertGateUpFusedRouteResult {
  std::vector<float> shared_gate;
  std::vector<float> shared_up;
  std::vector<float> shared_gate_up;
  std::vector<float> shared_swiglu;
};

struct ParallelRowRange {
  std::uint64_t begin = 0;
  std::uint64_t end = 0;
};

using ParallelRangeFn =
    std::function<void(std::uint64_t, std::uint64_t, std::uint64_t)>;

class SharedParallelExecutor {
 public:
  ~SharedParallelExecutor() {
    shutdown();
  }

  SharedParallelExecutor(const SharedParallelExecutor&) = delete;
  SharedParallelExecutor& operator=(const SharedParallelExecutor&) = delete;

  void run(std::uint64_t row_count,
           std::uint64_t thread_count,
           const ParallelRangeFn& fn) {
    if (row_count == 0) {
      return;
    }
    const auto effective_thread_count =
        std::min<std::uint64_t>(row_count, std::max<std::uint64_t>(1, thread_count));
    if (effective_thread_count <= 1) {
      fn(0, row_count, 0);
      return;
    }

    ensure_workers(effective_thread_count - 1);

    {
      std::unique_lock<std::mutex> lock(mutex_);
      if (active_) {
        throw std::runtime_error("shared parallel executor is already active");
      }
      active_ = true;
      job_ = fn;
      worker_exception_ = nullptr;
      ranges_.assign(
          static_cast<std::size_t>(effective_thread_count - 1),
          ParallelRowRange{});
      for (std::uint64_t worker_index = 0;
           worker_index < effective_thread_count - 1;
           ++worker_index) {
        const auto shard_index = worker_index + 1;
        ranges_[static_cast<std::size_t>(worker_index)] = ParallelRowRange{
            row_count * shard_index / effective_thread_count,
            row_count * (shard_index + 1) / effective_thread_count};
      }
      remaining_workers_ = effective_thread_count - 1;
      ++generation_;
    }
    cv_.notify_all();

    std::exception_ptr main_exception;
    try {
      fn(0, row_count / effective_thread_count, 0);
    } catch (...) {
      main_exception = std::current_exception();
    }

    std::exception_ptr worker_exception;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      done_cv_.wait(lock, [&]() { return remaining_workers_ == 0; });
      worker_exception = worker_exception_;
      job_ = nullptr;
      ranges_.clear();
      active_ = false;
    }

    if (main_exception) {
      std::rethrow_exception(main_exception);
    }
    if (worker_exception) {
      std::rethrow_exception(worker_exception);
    }
  }

 private:
  SharedParallelExecutor() = default;

  friend SharedParallelExecutor& shared_parallel_executor();

  void ensure_workers(std::uint64_t worker_count) {
    while (workers_.size() < worker_count) {
      const auto worker_index = workers_.size();
      workers_.emplace_back([this, worker_index]() { worker_loop(worker_index); });
    }
  }

  void shutdown() {
    {
      std::unique_lock<std::mutex> lock(mutex_);
      stop_ = true;
      ++generation_;
    }
    cv_.notify_all();
    for (auto& worker : workers_) {
      if (worker.joinable()) {
        worker.join();
      }
    }
  }

  void worker_loop(std::size_t worker_index) {
    std::uint64_t seen_generation = 0;
    while (true) {
      ParallelRowRange range;
      ParallelRangeFn job;
      std::uint64_t shard_index = 0;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [&]() { return stop_ || generation_ != seen_generation; });
        if (stop_) {
          return;
        }
        seen_generation = generation_;
        if (worker_index >= ranges_.size()) {
          continue;
        }
        range = ranges_[worker_index];
        job = job_;
        shard_index = static_cast<std::uint64_t>(worker_index + 1);
      }

      try {
        if (range.begin < range.end && job) {
          job(range.begin, range.end, shard_index);
        }
      } catch (...) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!worker_exception_) {
          worker_exception_ = std::current_exception();
        }
      }

      {
        std::unique_lock<std::mutex> lock(mutex_);
        if (remaining_workers_ > 0) {
          --remaining_workers_;
        }
        if (remaining_workers_ == 0) {
          done_cv_.notify_one();
        }
      }
    }
  }

  std::mutex mutex_;
  std::condition_variable cv_;
  std::condition_variable done_cv_;
  std::vector<std::thread> workers_;
  std::vector<ParallelRowRange> ranges_;
  ParallelRangeFn job_;
  std::exception_ptr worker_exception_;
  std::uint64_t generation_ = 0;
  std::uint64_t remaining_workers_ = 0;
  bool active_ = false;
  bool stop_ = false;
};

ResidentCacheState& resident_cache() {
  static ResidentCacheState state;
  return state;
}

MatvecProfileState& matvec_profile() {
  static MatvecProfileState state;
  return state;
}

ExpertSliceMatvecState& expert_slice_matvec_state() {
  static ExpertSliceMatvecState state;
  return state;
}

DenseMatvecState& dense_matvec_state() {
  static DenseMatvecState state;
  return state;
}

DenseMatvecPayloadCacheState& dense_matvec_payload_cache_state() {
  static DenseMatvecPayloadCacheState state;
  return state;
}

DenseQ4DirectDotState& dense_q4_direct_dot_state() {
  static DenseQ4DirectDotState state;
  return state;
}

DenseQ4PairDotState& dense_q4_pair_dot_state() {
  static DenseQ4PairDotState state;
  return state;
}

DenseQ6DirectDotState& dense_q6_direct_dot_state() {
  static DenseQ6DirectDotState state;
  return state;
}

DenseQ6PairDotState& dense_q6_pair_dot_state() {
  static DenseQ6PairDotState state;
  return state;
}

LmHeadQ6PairDotState& lm_head_q6_pair_dot_state() {
  static LmHeadQ6PairDotState state;
  return state;
}

Q4DirectMinsumPairState& q4_direct_minsum_pair_state() {
  static Q4DirectMinsumPairState state;
  return state;
}

Q4BlockMetaCacheState& q4_block_meta_cache_state() {
  static Q4BlockMetaCacheState state;
  return state;
}

Q4PlaneLayoutState& q4_plane_layout_state() {
  static Q4PlaneLayoutState state;
  return state;
}

DenseQ4PlanePairDotState& dense_q4_plane_pair_dot_state() {
  static DenseQ4PlanePairDotState state;
  return state;
}

SmallQ4DirectDotState& small_q4_direct_dot_state() {
  static SmallQ4DirectDotState state;
  return state;
}

MatvecQ8InputReuseState& matvec_q8_input_reuse_state() {
  static MatvecQ8InputReuseState state;
  return state;
}

SharedParallelExecutorState& shared_parallel_executor_state() {
  static SharedParallelExecutorState state;
  return state;
}

SharedExpertGateUpFusedState& shared_expert_gate_up_fused_state() {
  static SharedExpertGateUpFusedState state;
  return state;
}

SelectedExpertFfnState& selected_expert_ffn_state() {
  static SelectedExpertFfnState state;
  return state;
}

SelectedExpertMinimalOutputsState& selected_expert_minimal_outputs_state() {
  static SelectedExpertMinimalOutputsState state;
  return state;
}

SelectedExpertSliceCacheState& selected_expert_slice_cache_state() {
  static SelectedExpertSliceCacheState state;
  return state;
}

SelectedExpertDownSliceCacheState& selected_expert_down_slice_cache_state() {
  static SelectedExpertDownSliceCacheState state;
  return state;
}

SelectedExpertDownExpertMajorState& selected_expert_down_expert_major_state() {
  static SelectedExpertDownExpertMajorState state;
  return state;
}

SelectedExpertDownQ4PairDotState& selected_expert_down_q4_pair_dot_state() {
  static SelectedExpertDownQ4PairDotState state;
  return state;
}

SelectedExpertDownQ6PairDotState& selected_expert_down_q6_pair_dot_state() {
  static SelectedExpertDownQ6PairDotState state;
  return state;
}

SelectedGateQ4DirectDotState& selected_gate_q4_direct_dot_state() {
  static SelectedGateQ4DirectDotState state;
  return state;
}

SelectedGateQ4PairDotState& selected_gate_q4_pair_dot_state() {
  static SelectedGateQ4PairDotState state;
  return state;
}

SelectedGateQ4PairSumDotState& selected_gate_q4_pair_sum_dot_state() {
  static SelectedGateQ4PairSumDotState state;
  return state;
}

SelectedGateQ4PlanePairDotState& selected_gate_q4_plane_pair_dot_state() {
  static SelectedGateQ4PlanePairDotState state;
  return state;
}

SharedParallelExecutor& shared_parallel_executor() {
  static SharedParallelExecutor executor;
  return executor;
}

using ProfileClock = std::chrono::steady_clock;

void parallel_for_rows(std::uint64_t row_count,
                       std::uint64_t thread_count,
                       const ParallelRangeFn& fn) {
  if (row_count == 0) {
    return;
  }
  const auto effective_thread_count =
      std::min<std::uint64_t>(row_count, std::max<std::uint64_t>(1, thread_count));
  if (effective_thread_count <= 1) {
    fn(0, row_count, 0);
    return;
  }
  if (shared_parallel_executor_state().enabled) {
    shared_parallel_executor().run(row_count, effective_thread_count, fn);
    return;
  }

  std::vector<std::thread> workers;
  workers.reserve(static_cast<std::size_t>(effective_thread_count));
  for (std::uint64_t shard_index = 0; shard_index < effective_thread_count;
       ++shard_index) {
    const std::uint64_t begin = row_count * shard_index / effective_thread_count;
    const std::uint64_t end =
        row_count * (shard_index + 1) / effective_thread_count;
    workers.emplace_back([&, begin, end, shard_index]() {
      fn(begin, end, shard_index);
    });
  }
  for (auto& worker : workers) {
    worker.join();
  }
}

std::uint64_t profile_elapsed_ns(ProfileClock::time_point start,
                                 ProfileClock::time_point end) {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
}

void record_matvec_profile(const std::string& op,
                           const std::string& tensor_name,
                           std::uint64_t input_value_count,
                           std::uint64_t output_value_count,
                           std::uint64_t row_count,
                           std::uint64_t elapsed_ns) {
  auto& profile = matvec_profile();
  if (!profile.enabled) {
    return;
  }
  const std::string key = op + "\n" + tensor_name;
  auto& row = profile.rows[key];
  row.op = op;
  row.tensor_name = tensor_name;
  ++row.call_count;
  row.total_ns += elapsed_ns;
  row.max_ns = std::max(row.max_ns, elapsed_ns);
  row.input_value_count += input_value_count;
  row.output_value_count += output_value_count;
  row.row_count += row_count;
}

std::string cache_key(const std::string& path, const GgufTensorInfo& tensor) {
  return path + "\n" + tensor.name;
}

std::string expert_slice_cache_key(const std::string& path,
                                   const GgufTensorInfo& tensor,
                                   std::uint64_t rows_per_expert,
                                   std::uint64_t row_nbytes,
                                   std::int32_t expert_id) {
  return cache_key(path, tensor) + "\nexpert=" + std::to_string(expert_id) +
         "\nrows=" + std::to_string(rows_per_expert) +
         "\nrow_nbytes=" + std::to_string(row_nbytes);
}

std::string q4_plane_rows_cache_key(const std::string& path,
                                    const GgufTensorInfo& tensor,
                                    std::uint64_t row_count,
                                    std::uint64_t row_nbytes) {
  return cache_key(path, tensor) + "\nq4k_plane_v0" +
         "\nrows=" + std::to_string(row_count) +
         "\nrow_nbytes=" + std::to_string(row_nbytes);
}

std::string q4_plane_expert_slice_cache_key(const std::string& path,
                                            const GgufTensorInfo& tensor,
                                            std::uint64_t rows_per_expert,
                                            std::uint64_t row_nbytes,
                                            std::int32_t expert_id) {
  return expert_slice_cache_key(path, tensor, rows_per_expert, row_nbytes,
                                expert_id) +
         "\nq4k_plane_v0";
}

std::string cache_key(const std::string& path,
                      const GgufTensorInfo& tensor,
                      std::uint64_t row_index) {
  return cache_key(path, tensor) + "\n" + std::to_string(row_index);
}

std::uint32_t read_u32(std::istream& input) {
  std::array<unsigned char, 4> bytes{};
  input.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
  if (!input) {
    throw std::runtime_error("unexpected EOF reading u32");
  }
  return static_cast<std::uint32_t>(bytes[0]) |
         (static_cast<std::uint32_t>(bytes[1]) << 8) |
         (static_cast<std::uint32_t>(bytes[2]) << 16) |
         (static_cast<std::uint32_t>(bytes[3]) << 24);
}

std::uint64_t read_u64(std::istream& input) {
  std::array<unsigned char, 8> bytes{};
  input.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
  if (!input) {
    throw std::runtime_error("unexpected EOF reading u64");
  }
  std::uint64_t value = 0;
  for (int i = 7; i >= 0; --i) {
    value = (value << 8) | bytes[static_cast<std::size_t>(i)];
  }
  return value;
}

template <typename T>
T read_pod(std::istream& input) {
  T value{};
  input.read(reinterpret_cast<char*>(&value), sizeof(T));
  if (!input) {
    throw std::runtime_error("unexpected EOF reading scalar");
  }
  return value;
}

std::string read_string(std::istream& input) {
  const auto size = read_u64(input);
  if (size > (1ULL << 32)) {
    throw std::runtime_error("GGUF string too large");
  }
  std::string value(static_cast<std::size_t>(size), '\0');
  input.read(value.data(), static_cast<std::streamsize>(value.size()));
  if (!input) {
    throw std::runtime_error("unexpected EOF reading string");
  }
  return value;
}

void skip_bytes(std::istream& input, std::uint64_t count) {
  input.seekg(static_cast<std::streamoff>(count), std::ios::cur);
  if (!input) {
    throw std::runtime_error("unexpected EOF skipping bytes");
  }
}

void skip_scalar(std::istream& input, std::uint32_t value_type) {
  switch (value_type) {
    case 0:
    case 1:
    case 7:
      skip_bytes(input, 1);
      return;
    case 2:
    case 3:
      skip_bytes(input, 2);
      return;
    case 4:
    case 5:
    case 6:
      skip_bytes(input, 4);
      return;
    case 10:
    case 11:
    case 12:
      skip_bytes(input, 8);
      return;
    case 8: {
      const auto size = read_u64(input);
      skip_bytes(input, size);
      return;
    }
    default:
      throw std::runtime_error("unsupported GGUF scalar type");
  }
}

GgufMetadataValue read_metadata_value(std::istream& input,
                                      std::uint32_t value_type) {
  GgufMetadataValue value;
  switch (value_type) {
    case 4:
      value.kind = GgufMetadataValue::Kind::kUInt;
      value.uint_value = read_u32(input);
      return value;
    case 5:
      value.kind = GgufMetadataValue::Kind::kInt;
      value.int_value = read_pod<std::int32_t>(input);
      return value;
    case 6:
      value.kind = GgufMetadataValue::Kind::kFloat;
      value.float_value = read_pod<float>(input);
      return value;
    case 7:
      value.kind = GgufMetadataValue::Kind::kBool;
      value.bool_value = read_pod<bool>(input);
      return value;
    case 8:
      value.kind = GgufMetadataValue::Kind::kString;
      value.string_value = read_string(input);
      return value;
    case 10:
      value.kind = GgufMetadataValue::Kind::kUInt;
      value.uint_value = read_u64(input);
      return value;
    case 11:
      value.kind = GgufMetadataValue::Kind::kInt;
      value.int_value = read_pod<std::int64_t>(input);
      return value;
    case 12:
      value.kind = GgufMetadataValue::Kind::kFloat;
      value.float_value = read_pod<double>(input);
      return value;
    case 9: {
      const auto element_type = read_u32(input);
      const auto length = read_u64(input);
      value.array_element_type = element_type;
      if (length > kMaxStoredMetadataArrayLength) {
        for (std::uint64_t i = 0; i < length; ++i) {
          skip_scalar(input, element_type);
        }
        value.kind = GgufMetadataValue::Kind::kUnknown;
        return value;
      }

      value.kind = GgufMetadataValue::Kind::kArray;
      switch (element_type) {
        case 4:
          value.uint_array.reserve(static_cast<std::size_t>(length));
          for (std::uint64_t i = 0; i < length; ++i) {
            value.uint_array.push_back(read_u32(input));
          }
          return value;
        case 5:
          value.int_array.reserve(static_cast<std::size_t>(length));
          for (std::uint64_t i = 0; i < length; ++i) {
            value.int_array.push_back(read_pod<std::int32_t>(input));
          }
          return value;
        case 6:
          value.float_array.reserve(static_cast<std::size_t>(length));
          for (std::uint64_t i = 0; i < length; ++i) {
            value.float_array.push_back(read_pod<float>(input));
          }
          return value;
        case 8:
          value.string_array.reserve(static_cast<std::size_t>(length));
          for (std::uint64_t i = 0; i < length; ++i) {
            value.string_array.push_back(read_string(input));
          }
          return value;
        case 10:
          value.uint_array.reserve(static_cast<std::size_t>(length));
          for (std::uint64_t i = 0; i < length; ++i) {
            value.uint_array.push_back(read_u64(input));
          }
          return value;
        case 11:
          value.int_array.reserve(static_cast<std::size_t>(length));
          for (std::uint64_t i = 0; i < length; ++i) {
            value.int_array.push_back(read_pod<std::int64_t>(input));
          }
          return value;
        case 12:
          value.float_array.reserve(static_cast<std::size_t>(length));
          for (std::uint64_t i = 0; i < length; ++i) {
            value.float_array.push_back(read_pod<double>(input));
          }
          return value;
        default:
          for (std::uint64_t i = 0; i < length; ++i) {
            skip_scalar(input, element_type);
          }
          value.kind = GgufMetadataValue::Kind::kUnknown;
          return value;
      }
    }
    default:
      throw std::runtime_error("unsupported GGUF metadata type");
  }
}

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment) {
  if (alignment == 0) {
    return value;
  }
  return ((value + alignment - 1) / alignment) * alignment;
}

std::uint64_t metadata_uint(const GgufModelIndex& index,
                            const std::string& key,
                            std::uint64_t fallback = 0) {
  const auto found = index.metadata.find(key);
  if (found == index.metadata.end()) {
    return fallback;
  }
  const auto& value = found->second;
  if (value.kind == GgufMetadataValue::Kind::kUInt) {
    return value.uint_value;
  }
  if (value.kind == GgufMetadataValue::Kind::kInt && value.int_value >= 0) {
    return static_cast<std::uint64_t>(value.int_value);
  }
  return fallback;
}

std::string metadata_string(const GgufModelIndex& index,
                            const std::string& key) {
  const auto found = index.metadata.find(key);
  if (found == index.metadata.end()) {
    return "";
  }
  const auto& value = found->second;
  return value.kind == GgufMetadataValue::Kind::kString ? value.string_value : "";
}

struct TensorSpec {
  std::vector<std::uint64_t> dims;
  std::vector<std::uint32_t> types;
};

using SpecMap = std::map<std::string, TensorSpec>;

SpecMap common_layer_specs() {
  return {
      {"attn_norm.weight", {{2048}, {0}}},
      {"ffn_down_exps.weight", {{512, 2048, 256}, {12, 14}}},
      {"ffn_down_shexp.weight", {{512, 2048}, {12, 14}}},
      {"ffn_gate_inp.weight", {{2048, 256}, {0}}},
      {"ffn_gate_inp_shexp.weight", {{2048}, {0}}},
      {"ffn_gate_shexp.weight", {{2048, 512}, {12}}},
      {"ffn_gate_up_exps.weight", {{2048, 1024, 256}, {12}}},
      {"ffn_up_shexp.weight", {{2048, 512}, {12}}},
      {"post_attention_norm.weight", {{2048}, {0}}},
  };
}

SpecMap linear_ssm_specs() {
  auto specs = common_layer_specs();
  specs.insert({
      {"attn_gate.weight", {{2048, 4096}, {12}}},
      {"attn_qkv.weight", {{2048, 8192}, {12, 14}}},
      {"ssm_a", {{32}, {0}}},
      {"ssm_alpha.weight", {{2048, 32}, {12}}},
      {"ssm_beta.weight", {{2048, 32}, {12}}},
      {"ssm_conv1d.weight", {{4, 8192}, {0}}},
      {"ssm_dt.bias", {{32}, {0}}},
      {"ssm_norm.weight", {{128}, {0}}},
      {"ssm_out.weight", {{4096, 2048}, {12}}},
  });
  return specs;
}

SpecMap full_attention_specs() {
  auto specs = common_layer_specs();
  specs.insert({
      {"attn_k.weight", {{2048, 512}, {12}}},
      {"attn_k_norm.weight", {{256}, {0}}},
      {"attn_output.weight", {{4096, 2048}, {12}}},
      {"attn_q.weight", {{2048, 8192}, {12}}},
      {"attn_q_norm.weight", {{256}, {0}}},
      {"attn_v.weight", {{2048, 512}, {12, 14}}},
  });
  return specs;
}

bool tensor_matches(const GgufTensorInfo* tensor, const TensorSpec& spec) {
  return tensor != nullptr &&
         tensor->dims == spec.dims &&
         std::find(spec.types.begin(), spec.types.end(), tensor->type) !=
             spec.types.end() &&
         tensor->nbytes > 0;
}

std::uint64_t tensor_element_count(const std::vector<std::uint64_t>& dims) {
  if (dims.empty()) {
    return 0;
  }
  std::uint64_t elements = 1;
  for (const auto dim : dims) {
    if (dim == 0 ||
        elements > std::numeric_limits<std::uint64_t>::max() / dim) {
      throw std::runtime_error("tensor dimensions overflow");
    }
    elements *= dim;
  }
  return elements;
}

std::vector<int> expected_full_attention_layers(int block_count, int interval) {
  std::vector<int> layers;
  for (int i = 0; i < block_count; ++i) {
    if ((i + 1) % interval == 0) {
      layers.push_back(i);
    }
  }
  return layers;
}

void record_check(GgufLoadMapSummary& summary,
                  bool ok,
                  const std::string& name) {
  if (!ok) {
    summary.failed_checks.push_back(name);
  }
}

float fp16_to_fp32(std::uint16_t half) {
  const std::uint32_t sign = (half & 0x8000u) << 16;
  std::uint32_t exponent = (half >> 10) & 0x1fu;
  std::uint32_t mantissa = half & 0x03ffu;
  std::uint32_t bits = 0;
  if (exponent == 0) {
    if (mantissa == 0) {
      bits = sign;
    } else {
      exponent = 1;
      while ((mantissa & 0x0400u) == 0) {
        mantissa <<= 1;
        --exponent;
      }
      mantissa &= 0x03ffu;
      bits = sign | ((exponent + 112u) << 23) | (mantissa << 13);
    }
  } else if (exponent == 31) {
    bits = sign | 0x7f800000u | (mantissa << 13);
  } else {
    bits = sign | ((exponent + 112u) << 23) | (mantissa << 13);
  }
  float value = 0.0f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

std::uint16_t read_le_u16(const std::vector<std::uint8_t>& bytes,
                          std::size_t offset) {
  return static_cast<std::uint16_t>(bytes[offset]) |
         (static_cast<std::uint16_t>(bytes[offset + 1]) << 8);
}

float read_le_f32(const std::vector<std::uint8_t>& bytes,
                  std::size_t offset) {
  const std::uint32_t bits =
      static_cast<std::uint32_t>(bytes[offset]) |
      (static_cast<std::uint32_t>(bytes[offset + 1]) << 8) |
      (static_cast<std::uint32_t>(bytes[offset + 2]) << 16) |
      (static_cast<std::uint32_t>(bytes[offset + 3]) << 24);
  float value = 0.0f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

float read_le_f32_ptr(const std::uint8_t* bytes) {
  const std::uint32_t bits =
      static_cast<std::uint32_t>(bytes[0]) |
      (static_cast<std::uint32_t>(bytes[1]) << 8) |
      (static_cast<std::uint32_t>(bytes[2]) << 16) |
      (static_cast<std::uint32_t>(bytes[3]) << 24);
  float value = 0.0f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

void get_scale_min_k4(int index,
                      const std::uint8_t* scales,
                      std::uint8_t& scale,
                      std::uint8_t& min) {
  if (index < 4) {
    scale = scales[index] & 63u;
    min = scales[index + 4] & 63u;
  } else {
    scale = (scales[index + 4] & 0x0fu) |
            static_cast<std::uint8_t>((scales[index - 4] >> 6) << 4);
    min = (scales[index + 4] >> 4) |
          static_cast<std::uint8_t>((scales[index] >> 6) << 4);
  }
}

std::vector<float> decode_f32_values(const std::vector<std::uint8_t>& bytes,
                                     std::size_t max_values) {
  const auto count = std::min(max_values, bytes.size() / sizeof(float));
  std::vector<float> values;
  values.reserve(count);
  for (std::size_t i = 0; i < count; ++i) {
    values.push_back(read_le_f32(bytes, i * sizeof(float)));
  }
  return values;
}

std::vector<float> decode_q4_k_block(const std::vector<std::uint8_t>& bytes) {
  if (bytes.size() < 144) {
    throw std::runtime_error("Q4_K block too small");
  }
  const float d = fp16_to_fp32(read_le_u16(bytes, 0));
  const float dmin = fp16_to_fp32(read_le_u16(bytes, 2));
  const auto* scales = bytes.data() + 4;
  const auto* q = bytes.data() + 16;
  std::vector<float> values;
  values.reserve(256);
  int is = 0;
  for (int group = 0; group < 4; ++group) {
    std::uint8_t sc = 0;
    std::uint8_t m = 0;
    get_scale_min_k4(is + 0, scales, sc, m);
    const float d1 = d * static_cast<float>(sc);
    const float m1 = dmin * static_cast<float>(m);
    get_scale_min_k4(is + 1, scales, sc, m);
    const float d2 = d * static_cast<float>(sc);
    const float m2 = dmin * static_cast<float>(m);
    const auto* group_q = q + group * 32;
    for (int i = 0; i < 32; ++i) {
      values.push_back(d1 * static_cast<float>(group_q[i] & 0x0f) - m1);
    }
    for (int i = 0; i < 32; ++i) {
      values.push_back(d2 * static_cast<float>(group_q[i] >> 4) - m2);
    }
    is += 2;
  }
  return values;
}

std::vector<float> decode_q6_k_block(const std::vector<std::uint8_t>& bytes) {
  if (bytes.size() < 210) {
    throw std::runtime_error("Q6_K block too small");
  }
  const auto* ql = bytes.data();
  const auto* qh = bytes.data() + 128;
  const auto* sc = reinterpret_cast<const std::int8_t*>(bytes.data() + 192);
  const float d = fp16_to_fp32(read_le_u16(bytes, 208));
  std::vector<float> values(256, 0.0f);
  for (int n = 0; n < 256; n += 128) {
    const int base = n;
    for (int l = 0; l < 32; ++l) {
      const int is = l / 16;
      const std::int8_t q1 =
          static_cast<std::int8_t>((ql[l + 0] & 0x0f) |
                                   (((qh[l] >> 0) & 3) << 4)) -
          32;
      const std::int8_t q2 =
          static_cast<std::int8_t>((ql[l + 32] & 0x0f) |
                                   (((qh[l] >> 2) & 3) << 4)) -
          32;
      const std::int8_t q3 =
          static_cast<std::int8_t>((ql[l + 0] >> 4) |
                                   (((qh[l] >> 4) & 3) << 4)) -
          32;
      const std::int8_t q4 =
          static_cast<std::int8_t>((ql[l + 32] >> 4) |
                                   (((qh[l] >> 6) & 3) << 4)) -
          32;
      values[base + l + 0] = d * static_cast<float>(sc[is + 0]) * q1;
      values[base + l + 32] = d * static_cast<float>(sc[is + 2]) * q2;
      values[base + l + 64] = d * static_cast<float>(sc[is + 4]) * q3;
      values[base + l + 96] = d * static_cast<float>(sc[is + 6]) * q4;
    }
    ql += 64;
    qh += 32;
    sc += 8;
  }
  return values;
}

std::uint16_t read_le_u16_ptr(const std::uint8_t* bytes) {
  return static_cast<std::uint16_t>(bytes[0]) |
         (static_cast<std::uint16_t>(bytes[1]) << 8);
}

struct Q8KBlock {
  float d = 0.0f;
  std::array<std::int8_t, 256> qs{};
  std::array<std::int16_t, 16> bsums{};
};

int nearest_int(float value) {
  float shifted = value + 12582912.0f;
  int bits = 0;
  std::memcpy(&bits, &shifted, sizeof(bits));
  return (bits & 0x007fffff) - 0x00400000;
}

std::vector<Q8KBlock> quantize_q8_k_blocks(const std::vector<float>& input) {
  if (input.size() % 256 != 0) {
    throw std::runtime_error("Q8_K activation quantization requires 256-aligned input");
  }
  std::vector<Q8KBlock> blocks(input.size() / 256);
  for (std::size_t block_index = 0; block_index < blocks.size(); ++block_index) {
    const auto* block_input = input.data() + block_index * 256;
    auto& block = blocks[block_index];
    float max = 0.0f;
    float amax = 0.0f;
    for (int i = 0; i < 256; ++i) {
      const float abs_value = std::abs(block_input[i]);
      if (abs_value > amax) {
        amax = abs_value;
        max = block_input[i];
      }
    }
    if (amax == 0.0f) {
      continue;
    }
    const float iscale = -127.0f / max;
    for (int i = 0; i < 256; ++i) {
      const int quantized = std::min(127, nearest_int(iscale * block_input[i]));
      block.qs[static_cast<std::size_t>(i)] = static_cast<std::int8_t>(quantized);
    }
    for (int group = 0; group < 16; ++group) {
      int sum = 0;
      for (int i = 0; i < 16; ++i) {
        sum += block.qs[static_cast<std::size_t>(group * 16 + i)];
      }
      block.bsums[static_cast<std::size_t>(group)] = static_cast<std::int16_t>(sum);
    }
    block.d = 1.0f / iscale;
  }
  return blocks;
}

void accumulate_q4_k_q8_k_block(const std::uint8_t* block,
                                const Q8KBlock& input,
                                float (&sums)[8],
                                float& min_sum) {
  const auto* q4 = block + 16;
  const auto* q8 = input.qs.data();
  std::array<std::int8_t, 256> aux{};
  auto* a = aux.data();
  for (int group = 0; group < 4; ++group) {
    for (int lane = 0; lane < 32; ++lane) {
      a[lane] = static_cast<std::int8_t>(q4[lane] & 0x0f);
    }
    a += 32;
    for (int lane = 0; lane < 32; ++lane) {
      a[lane] = static_cast<std::int8_t>(q4[lane] >> 4);
    }
    a += 32;
    q4 += 32;
  }

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

  const auto* scales = reinterpret_cast<const std::uint8_t*>(unpacked.data());
  const auto* mins = reinterpret_cast<const std::uint8_t*>(unpacked.data() + 2);

  int grouped_min_sum = 0;
  for (int group = 0; group < 16; ++group) {
    grouped_min_sum +=
        static_cast<int>(input.bsums[static_cast<std::size_t>(group)]) *
        static_cast<int>(mins[group / 2]);
  }

  std::array<std::int32_t, 8> lane_sums{};
  a = aux.data();
  int scale_index = 0;
  for (int group = 0; group < 8; ++group) {
    const int scale = scales[scale_index++];
    for (int repeat = 0; repeat < 4; ++repeat) {
      for (int lane = 0; lane < 8; ++lane) {
        lane_sums[static_cast<std::size_t>(lane)] +=
            scale * static_cast<int>(q8[lane] * a[lane]);
      }
      q8 += 8;
      a += 8;
    }
  }

  const float d = fp16_to_fp32(read_le_u16_ptr(block)) * input.d;
  for (int lane = 0; lane < 8; ++lane) {
    sums[lane] += d *
                  static_cast<float>(lane_sums[static_cast<std::size_t>(lane)]);
  }
  const float dmin = fp16_to_fp32(read_le_u16_ptr(block + 2)) * input.d;
  min_sum -= dmin * static_cast<float>(grouped_min_sum);
}

void accumulate_q4_k_q8_k_block_direct(const std::uint8_t* block,
                                       const Q8KBlock& input,
                                       float (&sums)[8],
                                       float& min_sum) {
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

  const auto* scales = reinterpret_cast<const std::uint8_t*>(unpacked.data());
  const auto* mins = reinterpret_cast<const std::uint8_t*>(unpacked.data() + 2);

  int grouped_min_sum = 0;
  for (int group = 0; group < 16; ++group) {
    grouped_min_sum +=
        static_cast<int>(input.bsums[static_cast<std::size_t>(group)]) *
        static_cast<int>(mins[group / 2]);
  }

  std::array<std::int32_t, 8> lane_sums{};
  const auto* q4 = block + 16;
  const auto* q8 = input.qs.data();
  for (int group = 0; group < 4; ++group) {
    const auto* group_q4 = q4 + group * 32;
    const auto* group_q8 = q8 + group * 64;
    const int low_scale = scales[group * 2];
    const int high_scale = scales[group * 2 + 1];
    for (int lane = 0; lane < 32; ++lane) {
      const auto packed = group_q4[lane];
      const int q4_low = static_cast<int>(packed & 0x0f);
      const int q4_high = static_cast<int>(packed >> 4);
      const auto lane_index = static_cast<std::size_t>(lane & 7);
      lane_sums[lane_index] +=
          low_scale *
          (static_cast<int>(group_q8[lane]) * q4_low);
      lane_sums[lane_index] +=
          high_scale *
          (static_cast<int>(group_q8[32 + lane]) * q4_high);
    }
  }

  const float d = fp16_to_fp32(read_le_u16_ptr(block)) * input.d;
  for (int lane = 0; lane < 8; ++lane) {
    sums[lane] += d *
                  static_cast<float>(lane_sums[static_cast<std::size_t>(lane)]);
  }
  const float dmin = fp16_to_fp32(read_le_u16_ptr(block + 2)) * input.d;
  min_sum -= dmin * static_cast<float>(grouped_min_sum);
}

void accumulate_q4_k_q8_k_block_direct_minpair(const std::uint8_t* block,
                                               const Q8KBlock& input,
                                               float (&sums)[8],
                                               float& min_sum) {
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

  const auto* scales = reinterpret_cast<const std::uint8_t*>(unpacked.data());
  const auto* mins = reinterpret_cast<const std::uint8_t*>(unpacked.data() + 2);

  int grouped_min_sum = 0;
  for (int group = 0; group < 8; ++group) {
    const int bsum_pair =
        static_cast<int>(input.bsums[static_cast<std::size_t>(group * 2)]) +
        static_cast<int>(input.bsums[static_cast<std::size_t>(group * 2 + 1)]);
    grouped_min_sum += bsum_pair * static_cast<int>(mins[group]);
  }

  std::array<std::int32_t, 8> lane_sums{};
  const auto* q4 = block + 16;
  const auto* q8 = input.qs.data();
  for (int group = 0; group < 4; ++group) {
    const auto* group_q4 = q4 + group * 32;
    const auto* group_q8 = q8 + group * 64;
    const int low_scale = scales[group * 2];
    const int high_scale = scales[group * 2 + 1];
    for (int lane = 0; lane < 32; ++lane) {
      const auto packed = group_q4[lane];
      const int q4_low = static_cast<int>(packed & 0x0f);
      const int q4_high = static_cast<int>(packed >> 4);
      const auto lane_index = static_cast<std::size_t>(lane & 7);
      lane_sums[lane_index] +=
          low_scale *
          (static_cast<int>(group_q8[lane]) * q4_low);
      lane_sums[lane_index] +=
          high_scale *
          (static_cast<int>(group_q8[32 + lane]) * q4_high);
    }
  }

  const float d = fp16_to_fp32(read_le_u16_ptr(block)) * input.d;
  for (int lane = 0; lane < 8; ++lane) {
    sums[lane] += d *
                  static_cast<float>(lane_sums[static_cast<std::size_t>(lane)]);
  }
  const float dmin = fp16_to_fp32(read_le_u16_ptr(block + 2)) * input.d;
  min_sum -= dmin * static_cast<float>(grouped_min_sum);
}

void unpack_q4_k_scales_mins(const std::uint8_t* block,
                             std::array<std::uint8_t, 8>& scales,
                             std::array<std::uint8_t, 8>& mins) {
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
  std::memcpy(scales.data(), unpacked.data(), scales.size());
  std::memcpy(mins.data(), unpacked.data() + 2, mins.size());
}

Q4KPlaneRows make_q4_plane_rows(const std::uint8_t* payload_data,
                                std::size_t payload_size,
                                std::uint64_t row_count,
                                std::uint64_t row_nbytes) {
  if (row_count == 0 || row_nbytes == 0 || row_nbytes % 144 != 0 ||
      payload_size != row_count * row_nbytes) {
    throw std::runtime_error("Q4 plane layout payload shape mismatch");
  }

  Q4KPlaneRows plane;
  plane.row_count = row_count;
  plane.blocks_per_row = row_nbytes / 144;
  const auto block_count = plane.row_count * plane.blocks_per_row;
  plane.q4.resize(static_cast<std::size_t>(block_count * 128));
  plane.scales.resize(static_cast<std::size_t>(block_count * 8));
  plane.mins.resize(static_cast<std::size_t>(block_count * 8));
  plane.d_values.resize(static_cast<std::size_t>(block_count));
  plane.dmin_values.resize(static_cast<std::size_t>(block_count));

  for (std::uint64_t block_index = 0; block_index < block_count; ++block_index) {
    const auto* block =
        payload_data + static_cast<std::size_t>(block_index * 144);
    std::array<std::uint8_t, 8> scales{};
    std::array<std::uint8_t, 8> mins{};
    unpack_q4_k_scales_mins(block, scales, mins);
    plane.d_values[static_cast<std::size_t>(block_index)] =
        fp16_to_fp32(read_le_u16_ptr(block));
    plane.dmin_values[static_cast<std::size_t>(block_index)] =
        fp16_to_fp32(read_le_u16_ptr(block + 2));
    std::memcpy(plane.scales.data() + static_cast<std::size_t>(block_index * 8),
                scales.data(),
                scales.size());
    std::memcpy(plane.mins.data() + static_cast<std::size_t>(block_index * 8),
                mins.data(),
                mins.size());
    std::memcpy(plane.q4.data() + static_cast<std::size_t>(block_index * 128),
                block + 16,
                128);
  }
  return plane;
}

int q4_k_grouped_min_sum(const std::array<std::uint8_t, 8>& mins,
                         const Q8KBlock& input) {
  int sum = 0;
  for (int group = 0; group < 8; ++group) {
    const int bsum_pair =
        static_cast<int>(input.bsums[static_cast<std::size_t>(group * 2)]) +
        static_cast<int>(input.bsums[static_cast<std::size_t>(group * 2 + 1)]);
    sum += bsum_pair * static_cast<int>(mins[static_cast<std::size_t>(group)]);
  }
  return sum;
}

int q4_k_grouped_min_sum(const std::uint8_t* mins, const Q8KBlock& input) {
  int sum = 0;
  for (int group = 0; group < 8; ++group) {
    const int bsum_pair =
        static_cast<int>(input.bsums[static_cast<std::size_t>(group * 2)]) +
        static_cast<int>(input.bsums[static_cast<std::size_t>(group * 2 + 1)]);
    sum += bsum_pair * static_cast<int>(mins[group]);
  }
  return sum;
}

Q4KBlockMeta decode_q4_k_block_meta(const std::uint8_t* block) {
  Q4KBlockMeta meta;
  meta.d = fp16_to_fp32(read_le_u16_ptr(block));
  meta.dmin = fp16_to_fp32(read_le_u16_ptr(block + 2));
  unpack_q4_k_scales_mins(block, meta.scales, meta.mins);
  return meta;
}

float dot_q4_k_q8_k_row_plane(const Q4KPlaneRows& plane,
                              std::uint64_t row,
                              const std::vector<Q8KBlock>& q8_input) {
  if (row >= plane.row_count ||
      q8_input.size() != static_cast<std::size_t>(plane.blocks_per_row)) {
    throw std::runtime_error("Q4 plane row shape does not match Q8_K input");
  }

  float sums[8] = {};
  float min_sum = 0.0f;
  const auto row_block_base = row * plane.blocks_per_row;
  for (std::uint64_t block = 0; block < plane.blocks_per_row; ++block) {
    const auto block_index = row_block_base + block;
    const auto* q4 =
        plane.q4.data() + static_cast<std::size_t>(block_index * 128);
    const auto* scales =
        plane.scales.data() + static_cast<std::size_t>(block_index * 8);
    const auto* mins =
        plane.mins.data() + static_cast<std::size_t>(block_index * 8);
    const auto& input = q8_input[static_cast<std::size_t>(block)];

    std::array<std::int32_t, 8> lane_sums{};
    const auto* q8 = input.qs.data();
    for (int group = 0; group < 4; ++group) {
      const auto* group_q4 = q4 + group * 32;
      const auto* group_q8 = q8 + group * 64;
      const int low_scale = scales[group * 2];
      const int high_scale = scales[group * 2 + 1];
      for (int lane = 0; lane < 32; ++lane) {
        const auto packed = group_q4[lane];
        const int q4_low = static_cast<int>(packed & 0x0f);
        const int q4_high = static_cast<int>(packed >> 4);
        const auto lane_index = static_cast<std::size_t>(lane & 7);
        lane_sums[lane_index] +=
            low_scale * (static_cast<int>(group_q8[lane]) * q4_low);
        lane_sums[lane_index] +=
            high_scale * (static_cast<int>(group_q8[32 + lane]) * q4_high);
      }
    }

    const float d =
        plane.d_values[static_cast<std::size_t>(block_index)] * input.d;
    for (int lane = 0; lane < 8; ++lane) {
      sums[lane] +=
          d * static_cast<float>(lane_sums[static_cast<std::size_t>(lane)]);
    }
    const float dmin =
        plane.dmin_values[static_cast<std::size_t>(block_index)] * input.d;
    min_sum -= dmin * static_cast<float>(q4_k_grouped_min_sum(mins, input));
  }

  float sum = min_sum;
  for (const float lane_sum : sums) {
    sum += lane_sum;
  }
  return sum;
}

void dot_q4_k_q8_k_row_pair_plane(const Q4KPlaneRows& plane,
                                  std::uint64_t first_row,
                                  std::uint64_t second_row,
                                  const std::vector<Q8KBlock>& q8_input,
                                  float& first,
                                  float& second) {
  if (first_row >= plane.row_count || second_row >= plane.row_count ||
      q8_input.size() != static_cast<std::size_t>(plane.blocks_per_row)) {
    throw std::runtime_error("Q4 plane row-pair shape does not match Q8_K input");
  }

  float first_sums[8] = {};
  float second_sums[8] = {};
  float first_min_sum = 0.0f;
  float second_min_sum = 0.0f;
  const auto first_row_block_base = first_row * plane.blocks_per_row;
  const auto second_row_block_base = second_row * plane.blocks_per_row;
  for (std::uint64_t block = 0; block < plane.blocks_per_row; ++block) {
    const auto first_block_index = first_row_block_base + block;
    const auto second_block_index = second_row_block_base + block;
    const auto* first_q4 =
        plane.q4.data() + static_cast<std::size_t>(first_block_index * 128);
    const auto* second_q4 =
        plane.q4.data() + static_cast<std::size_t>(second_block_index * 128);
    const auto* first_scales =
        plane.scales.data() + static_cast<std::size_t>(first_block_index * 8);
    const auto* second_scales =
        plane.scales.data() + static_cast<std::size_t>(second_block_index * 8);
    const auto* first_mins =
        plane.mins.data() + static_cast<std::size_t>(first_block_index * 8);
    const auto* second_mins =
        plane.mins.data() + static_cast<std::size_t>(second_block_index * 8);
    const auto& input = q8_input[static_cast<std::size_t>(block)];

    std::array<std::int32_t, 8> first_lane_sums{};
    std::array<std::int32_t, 8> second_lane_sums{};
    const auto* q8 = input.qs.data();
    for (int group = 0; group < 4; ++group) {
      const auto* first_group_q4 = first_q4 + group * 32;
      const auto* second_group_q4 = second_q4 + group * 32;
      const auto* group_q8 = q8 + group * 64;
      const int first_low_scale = first_scales[group * 2];
      const int first_high_scale = first_scales[group * 2 + 1];
      const int second_low_scale = second_scales[group * 2];
      const int second_high_scale = second_scales[group * 2 + 1];
      for (int lane = 0; lane < 32; ++lane) {
        const auto first_packed = first_group_q4[lane];
        const auto second_packed = second_group_q4[lane];
        const int first_q4_low = static_cast<int>(first_packed & 0x0f);
        const int first_q4_high = static_cast<int>(first_packed >> 4);
        const int second_q4_low = static_cast<int>(second_packed & 0x0f);
        const int second_q4_high = static_cast<int>(second_packed >> 4);
        const auto lane_index = static_cast<std::size_t>(lane & 7);
        const int q8_low = static_cast<int>(group_q8[lane]);
        const int q8_high = static_cast<int>(group_q8[32 + lane]);
        first_lane_sums[lane_index] +=
            first_low_scale * (q8_low * first_q4_low);
        first_lane_sums[lane_index] +=
            first_high_scale * (q8_high * first_q4_high);
        second_lane_sums[lane_index] +=
            second_low_scale * (q8_low * second_q4_low);
        second_lane_sums[lane_index] +=
            second_high_scale * (q8_high * second_q4_high);
      }
    }

    const float first_d =
        plane.d_values[static_cast<std::size_t>(first_block_index)] * input.d;
    const float second_d =
        plane.d_values[static_cast<std::size_t>(second_block_index)] * input.d;
    for (int lane = 0; lane < 8; ++lane) {
      first_sums[lane] +=
          first_d *
          static_cast<float>(first_lane_sums[static_cast<std::size_t>(lane)]);
      second_sums[lane] +=
          second_d *
          static_cast<float>(second_lane_sums[static_cast<std::size_t>(lane)]);
    }
    const float first_dmin =
        plane.dmin_values[static_cast<std::size_t>(first_block_index)] * input.d;
    const float second_dmin =
        plane.dmin_values[static_cast<std::size_t>(second_block_index)] * input.d;
    first_min_sum -= first_dmin *
                     static_cast<float>(q4_k_grouped_min_sum(first_mins, input));
    second_min_sum -= second_dmin *
                      static_cast<float>(q4_k_grouped_min_sum(second_mins, input));
  }

  first = first_min_sum;
  second = second_min_sum;
  for (const float value : first_sums) {
    first += value;
  }
  for (const float value : second_sums) {
    second += value;
  }
}

void accumulate_q4_k_q8_k_block_direct_meta(
    const std::uint8_t* block,
    const Q4KBlockMeta& meta,
    const Q8KBlock& input,
    float (&sums)[8],
    float& min_sum) {
  const int grouped_min_sum = q4_k_grouped_min_sum(meta.mins, input);

  std::array<std::int32_t, 8> lane_sums{};
  const auto* q4 = block + 16;
  const auto* q8 = input.qs.data();
  for (int group = 0; group < 4; ++group) {
    const auto* group_q4 = q4 + group * 32;
    const auto* group_q8 = q8 + group * 64;
    const int low_scale = meta.scales[static_cast<std::size_t>(group * 2)];
    const int high_scale =
        meta.scales[static_cast<std::size_t>(group * 2 + 1)];
    for (int lane = 0; lane < 32; ++lane) {
      const auto packed = group_q4[lane];
      const int q4_low = static_cast<int>(packed & 0x0f);
      const int q4_high = static_cast<int>(packed >> 4);
      const auto lane_index = static_cast<std::size_t>(lane & 7);
      lane_sums[lane_index] +=
          low_scale * (static_cast<int>(group_q8[lane]) * q4_low);
      lane_sums[lane_index] +=
          high_scale * (static_cast<int>(group_q8[32 + lane]) * q4_high);
    }
  }

  const float d = meta.d * input.d;
  for (int lane = 0; lane < 8; ++lane) {
    sums[lane] += d *
                  static_cast<float>(lane_sums[static_cast<std::size_t>(lane)]);
  }
  const float dmin = meta.dmin * input.d;
  min_sum -= dmin * static_cast<float>(grouped_min_sum);
}

void accumulate_q4_k_q8_k_block_pair_direct(const std::uint8_t* gate_block,
                                            const std::uint8_t* up_block,
                                            const Q8KBlock& input,
                                            float (&gate_sums)[8],
                                            float& gate_min_sum,
                                            float (&up_sums)[8],
                                            float& up_min_sum) {
  std::array<std::uint8_t, 8> gate_scales{};
  std::array<std::uint8_t, 8> gate_mins{};
  std::array<std::uint8_t, 8> up_scales{};
  std::array<std::uint8_t, 8> up_mins{};
  unpack_q4_k_scales_mins(gate_block, gate_scales, gate_mins);
  unpack_q4_k_scales_mins(up_block, up_scales, up_mins);

  const int gate_grouped_min_sum = q4_k_grouped_min_sum(gate_mins, input);
  const int up_grouped_min_sum = q4_k_grouped_min_sum(up_mins, input);

  std::array<std::int32_t, 8> gate_lane_sums{};
  std::array<std::int32_t, 8> up_lane_sums{};
  const auto* gate_q4 = gate_block + 16;
  const auto* up_q4 = up_block + 16;
  const auto* q8 = input.qs.data();
  for (int group = 0; group < 4; ++group) {
    const auto* gate_group_q4 = gate_q4 + group * 32;
    const auto* up_group_q4 = up_q4 + group * 32;
    const auto* group_q8 = q8 + group * 64;
    const int gate_low_scale = gate_scales[static_cast<std::size_t>(group * 2)];
    const int gate_high_scale =
        gate_scales[static_cast<std::size_t>(group * 2 + 1)];
    const int up_low_scale = up_scales[static_cast<std::size_t>(group * 2)];
    const int up_high_scale =
        up_scales[static_cast<std::size_t>(group * 2 + 1)];
    for (int lane = 0; lane < 32; ++lane) {
      const auto lane_index = static_cast<std::size_t>(lane & 7);
      const int low_q8 = static_cast<int>(group_q8[lane]);
      const int high_q8 = static_cast<int>(group_q8[32 + lane]);
      const auto gate_packed = gate_group_q4[lane];
      const auto up_packed = up_group_q4[lane];
      gate_lane_sums[lane_index] +=
          gate_low_scale * low_q8 * static_cast<int>(gate_packed & 0x0f);
      gate_lane_sums[lane_index] +=
          gate_high_scale * high_q8 * static_cast<int>(gate_packed >> 4);
      up_lane_sums[lane_index] +=
          up_low_scale * low_q8 * static_cast<int>(up_packed & 0x0f);
      up_lane_sums[lane_index] +=
          up_high_scale * high_q8 * static_cast<int>(up_packed >> 4);
    }
  }

  const float gate_d = fp16_to_fp32(read_le_u16_ptr(gate_block)) * input.d;
  const float up_d = fp16_to_fp32(read_le_u16_ptr(up_block)) * input.d;
  for (int lane = 0; lane < 8; ++lane) {
    const auto lane_index = static_cast<std::size_t>(lane);
    gate_sums[lane] += gate_d * static_cast<float>(gate_lane_sums[lane_index]);
    up_sums[lane] += up_d * static_cast<float>(up_lane_sums[lane_index]);
  }
  const float gate_dmin = fp16_to_fp32(read_le_u16_ptr(gate_block + 2)) * input.d;
  const float up_dmin = fp16_to_fp32(read_le_u16_ptr(up_block + 2)) * input.d;
  gate_min_sum -= gate_dmin * static_cast<float>(gate_grouped_min_sum);
  up_min_sum -= up_dmin * static_cast<float>(up_grouped_min_sum);
}

void accumulate_q4_k_q8_k_block_pair_sum_direct(
    const std::uint8_t* gate_block,
    const std::uint8_t* up_block,
    const Q8KBlock& input,
    float& gate_scaled_sum,
    float& gate_min_sum,
    float& up_scaled_sum,
    float& up_min_sum) {
  std::array<std::uint8_t, 8> gate_scales{};
  std::array<std::uint8_t, 8> gate_mins{};
  std::array<std::uint8_t, 8> up_scales{};
  std::array<std::uint8_t, 8> up_mins{};
  unpack_q4_k_scales_mins(gate_block, gate_scales, gate_mins);
  unpack_q4_k_scales_mins(up_block, up_scales, up_mins);

  const int gate_grouped_min_sum = q4_k_grouped_min_sum(gate_mins, input);
  const int up_grouped_min_sum = q4_k_grouped_min_sum(up_mins, input);

  std::int32_t gate_block_sum = 0;
  std::int32_t up_block_sum = 0;
  const auto* gate_q4 = gate_block + 16;
  const auto* up_q4 = up_block + 16;
  const auto* q8 = input.qs.data();
  for (int group = 0; group < 4; ++group) {
    const auto* gate_group_q4 = gate_q4 + group * 32;
    const auto* up_group_q4 = up_q4 + group * 32;
    const auto* group_q8 = q8 + group * 64;
    const int gate_low_scale = gate_scales[static_cast<std::size_t>(group * 2)];
    const int gate_high_scale =
        gate_scales[static_cast<std::size_t>(group * 2 + 1)];
    const int up_low_scale = up_scales[static_cast<std::size_t>(group * 2)];
    const int up_high_scale =
        up_scales[static_cast<std::size_t>(group * 2 + 1)];
    for (int lane = 0; lane < 32; ++lane) {
      const int low_q8 = static_cast<int>(group_q8[lane]);
      const int high_q8 = static_cast<int>(group_q8[32 + lane]);
      const auto gate_packed = gate_group_q4[lane];
      const auto up_packed = up_group_q4[lane];
      gate_block_sum +=
          gate_low_scale * low_q8 * static_cast<int>(gate_packed & 0x0f);
      gate_block_sum +=
          gate_high_scale * high_q8 * static_cast<int>(gate_packed >> 4);
      up_block_sum +=
          up_low_scale * low_q8 * static_cast<int>(up_packed & 0x0f);
      up_block_sum +=
          up_high_scale * high_q8 * static_cast<int>(up_packed >> 4);
    }
  }

  const float gate_d = fp16_to_fp32(read_le_u16_ptr(gate_block)) * input.d;
  const float up_d = fp16_to_fp32(read_le_u16_ptr(up_block)) * input.d;
  gate_scaled_sum += gate_d * static_cast<float>(gate_block_sum);
  up_scaled_sum += up_d * static_cast<float>(up_block_sum);
  const float gate_dmin =
      fp16_to_fp32(read_le_u16_ptr(gate_block + 2)) * input.d;
  const float up_dmin = fp16_to_fp32(read_le_u16_ptr(up_block + 2)) * input.d;
  gate_min_sum -= gate_dmin * static_cast<float>(gate_grouped_min_sum);
  up_min_sum -= up_dmin * static_cast<float>(up_grouped_min_sum);
}

float dot_q4_k_q8_k_row(const std::vector<std::uint8_t>& bytes,
                        const std::vector<Q8KBlock>& q8_input) {
  if (bytes.size() != q8_input.size() * 144) {
    throw std::runtime_error("Q4_K row byte count does not match Q8_K input");
  }
  float sums[8] = {};
  float min_sum = 0.0f;
  for (std::size_t block = 0; block < q8_input.size(); ++block) {
    accumulate_q4_k_q8_k_block(bytes.data() + block * 144,
                               q8_input[block],
                               sums,
                               min_sum);
  }
  float sum = min_sum;
  for (const float lane_sum : sums) {
    sum += lane_sum;
  }
  return sum;
}

float dot_q4_k_q8_k_row(const std::uint8_t* bytes,
                        std::size_t byte_count,
                        const std::vector<Q8KBlock>& q8_input) {
  if (byte_count != q8_input.size() * 144) {
    throw std::runtime_error("Q4_K row byte count does not match Q8_K input");
  }
  float sums[8] = {};
  float min_sum = 0.0f;
  for (std::size_t block = 0; block < q8_input.size(); ++block) {
    accumulate_q4_k_q8_k_block(bytes + block * 144,
                               q8_input[block],
                               sums,
                               min_sum);
  }
  float sum = min_sum;
  for (const float lane_sum : sums) {
    sum += lane_sum;
  }
  return sum;
}

void dot_q4_k_q8_k_row_pair_direct(const std::uint8_t* gate_bytes,
                                   const std::uint8_t* up_bytes,
                                   std::size_t byte_count,
                                   const std::vector<Q8KBlock>& q8_input,
                                   float& gate,
                                   float& up) {
  if (byte_count != q8_input.size() * 144) {
    throw std::runtime_error("Q4_K row byte count does not match Q8_K input");
  }
  float gate_sums[8] = {};
  float up_sums[8] = {};
  float gate_min_sum = 0.0f;
  float up_min_sum = 0.0f;
  for (std::size_t block = 0; block < q8_input.size(); ++block) {
    accumulate_q4_k_q8_k_block_pair_direct(
        gate_bytes + block * 144,
        up_bytes + block * 144,
        q8_input[block],
        gate_sums,
        gate_min_sum,
        up_sums,
        up_min_sum);
  }
  gate = gate_min_sum;
  up = up_min_sum;
  for (int lane = 0; lane < 8; ++lane) {
    gate += gate_sums[lane];
    up += up_sums[lane];
  }
}

void dot_q4_k_q8_k_row_pair_sum_direct(const std::uint8_t* gate_bytes,
                                       const std::uint8_t* up_bytes,
                                       std::size_t byte_count,
                                       const std::vector<Q8KBlock>& q8_input,
                                       float& gate,
                                       float& up) {
  if (byte_count != q8_input.size() * 144) {
    throw std::runtime_error("Q4_K row byte count does not match Q8_K input");
  }
  float gate_scaled_sum = 0.0f;
  float up_scaled_sum = 0.0f;
  float gate_min_sum = 0.0f;
  float up_min_sum = 0.0f;
  for (std::size_t block = 0; block < q8_input.size(); ++block) {
    accumulate_q4_k_q8_k_block_pair_sum_direct(
        gate_bytes + block * 144,
        up_bytes + block * 144,
        q8_input[block],
        gate_scaled_sum,
        gate_min_sum,
        up_scaled_sum,
        up_min_sum);
  }
  gate = gate_scaled_sum + gate_min_sum;
  up = up_scaled_sum + up_min_sum;
}

float dot_q4_k_q8_k_row_direct(const std::uint8_t* bytes,
                               std::size_t byte_count,
                               const std::vector<Q8KBlock>& q8_input) {
  if (byte_count != q8_input.size() * 144) {
    throw std::runtime_error("Q4_K row byte count does not match Q8_K input");
  }
  float sums[8] = {};
  float min_sum = 0.0f;
  for (std::size_t block = 0; block < q8_input.size(); ++block) {
    accumulate_q4_k_q8_k_block_direct(bytes + block * 144,
                                      q8_input[block],
                                      sums,
                                      min_sum);
  }
  float sum = min_sum;
  for (const float lane_sum : sums) {
    sum += lane_sum;
  }
  return sum;
}

float dot_q4_k_q8_k_row_direct_meta(
    const std::uint8_t* bytes,
    std::size_t byte_count,
    const std::vector<Q8KBlock>& q8_input,
    const Q4KBlockMeta* meta) {
  if (byte_count != q8_input.size() * 144) {
    throw std::runtime_error("Q4_K row byte count does not match Q8_K input");
  }
  float sums[8] = {};
  float min_sum = 0.0f;
  for (std::size_t block = 0; block < q8_input.size(); ++block) {
    accumulate_q4_k_q8_k_block_direct_meta(bytes + block * 144,
                                           meta[block],
                                           q8_input[block],
                                           sums,
                                           min_sum);
  }
  float sum = min_sum;
  for (const float lane_sum : sums) {
    sum += lane_sum;
  }
  return sum;
}

float dot_q4_k_q8_k_row_direct_minpair(
    const std::uint8_t* bytes,
    std::size_t byte_count,
    const std::vector<Q8KBlock>& q8_input) {
  if (byte_count != q8_input.size() * 144) {
    throw std::runtime_error("Q4_K row byte count does not match Q8_K input");
  }
  float sums[8] = {};
  float min_sum = 0.0f;
  for (std::size_t block = 0; block < q8_input.size(); ++block) {
    accumulate_q4_k_q8_k_block_direct_minpair(bytes + block * 144,
                                              q8_input[block],
                                              sums,
                                              min_sum);
  }
  float sum = min_sum;
  for (const float lane_sum : sums) {
    sum += lane_sum;
  }
  return sum;
}

void accumulate_q6_k_q8_k_block(const std::uint8_t* block,
                                const Q8KBlock& input,
                                float (&sums)[8]) {
  const auto* ql = block;
  const auto* qh = block + 128;
  const auto* scales = reinterpret_cast<const std::int8_t*>(block + 192);
  const float d = fp16_to_fp32(read_le_u16_ptr(block + 208));
  std::array<std::int8_t, 256> aux{};
  auto* a = aux.data();
  for (int n = 0; n < 256; n += 128) {
    for (int l = 0; l < 32; ++l) {
      a[l + 0] =
          static_cast<std::int8_t>((ql[l + 0] & 0x0f) |
                                   (((qh[l] >> 0) & 3) << 4)) -
          32;
      a[l + 32] =
          static_cast<std::int8_t>((ql[l + 32] & 0x0f) |
                                   (((qh[l] >> 2) & 3) << 4)) -
          32;
      a[l + 64] =
          static_cast<std::int8_t>((ql[l + 0] >> 4) |
                                   (((qh[l] >> 4) & 3) << 4)) -
          32;
      a[l + 96] =
          static_cast<std::int8_t>((ql[l + 32] >> 4) |
                                   (((qh[l] >> 6) & 3) << 4)) -
          32;
    }
    a += 128;
    ql += 64;
    qh += 32;
  }
  std::array<std::int32_t, 8> lane_sums{};
  const auto* q8 = input.qs.data();
  a = aux.data();
  int scale_index = 0;
  for (int group = 0; group < 16; ++group) {
    const int scale = scales[scale_index++];
    for (int lane = 0; lane < 8; ++lane) {
      lane_sums[static_cast<std::size_t>(lane)] +=
          scale * static_cast<int>(q8[lane] * a[lane]);
    }
    q8 += 8;
    a += 8;
    for (int lane = 0; lane < 8; ++lane) {
      lane_sums[static_cast<std::size_t>(lane)] +=
          scale * static_cast<int>(q8[lane] * a[lane]);
    }
    q8 += 8;
    a += 8;
  }
  const float combined_scale = d * input.d;
  for (int lane = 0; lane < 8; ++lane) {
    sums[lane] += combined_scale *
                  static_cast<float>(lane_sums[static_cast<std::size_t>(lane)]);
  }
}

void accumulate_q6_k_q8_k_block_direct(const std::uint8_t* block,
                                       const Q8KBlock& input,
                                       float (&sums)[8]) {
  const auto* scales = reinterpret_cast<const std::int8_t*>(block + 192);
  const float combined_scale =
      fp16_to_fp32(read_le_u16_ptr(block + 208)) * input.d;
  std::array<std::int32_t, 8> lane_sums{};
  const auto* q8 = input.qs.data();
  for (int half = 0; half < 2; ++half) {
    const auto* ql = block + half * 64;
    const auto* qh = block + 128 + half * 32;
    const auto* half_scales = scales + half * 8;
    const int base = half * 128;
    for (int scale_group = 0; scale_group < 2; ++scale_group) {
      const int lane_begin = scale_group * 16;
      const int scale_0 = static_cast<int>(half_scales[scale_group]);
      const int scale_1 = static_cast<int>(half_scales[scale_group + 2]);
      const int scale_2 = static_cast<int>(half_scales[scale_group + 4]);
      const int scale_3 = static_cast<int>(half_scales[scale_group + 6]);
      for (int lane = lane_begin; lane < lane_begin + 16; ++lane) {
        const int high = static_cast<int>(qh[lane]);
        const auto lane_index = static_cast<std::size_t>(lane & 7);
        lane_sums[lane_index] +=
            scale_0 *
            static_cast<int>(q8[base + lane]) *
            ((static_cast<int>(ql[lane] & 0x0f) |
              (((high >> 0) & 3) << 4)) -
             32);
        lane_sums[lane_index] +=
            scale_1 *
            static_cast<int>(q8[base + 32 + lane]) *
            ((static_cast<int>(ql[32 + lane] & 0x0f) |
              (((high >> 2) & 3) << 4)) -
             32);
        lane_sums[lane_index] +=
            scale_2 *
            static_cast<int>(q8[base + 64 + lane]) *
            ((static_cast<int>(ql[lane] >> 4) |
              (((high >> 4) & 3) << 4)) -
             32);
        lane_sums[lane_index] +=
            scale_3 *
            static_cast<int>(q8[base + 96 + lane]) *
            ((static_cast<int>(ql[32 + lane] >> 4) |
              (((high >> 6) & 3) << 4)) -
             32);
      }
    }
  }
  for (int lane = 0; lane < 8; ++lane) {
    sums[lane] += combined_scale *
                  static_cast<float>(lane_sums[static_cast<std::size_t>(lane)]);
  }
}

void accumulate_q6_k_q8_k_block_pair_direct(const std::uint8_t* first_block,
                                            const std::uint8_t* second_block,
                                            const Q8KBlock& input,
                                            float (&first_sums)[8],
                                            float (&second_sums)[8]) {
  const auto* first_scales =
      reinterpret_cast<const std::int8_t*>(first_block + 192);
  const auto* second_scales =
      reinterpret_cast<const std::int8_t*>(second_block + 192);
  const float first_combined_scale =
      fp16_to_fp32(read_le_u16_ptr(first_block + 208)) * input.d;
  const float second_combined_scale =
      fp16_to_fp32(read_le_u16_ptr(second_block + 208)) * input.d;
  std::array<std::int32_t, 8> first_lane_sums{};
  std::array<std::int32_t, 8> second_lane_sums{};
  const auto* q8 = input.qs.data();
  for (int half = 0; half < 2; ++half) {
    const auto* first_ql = first_block + half * 64;
    const auto* first_qh = first_block + 128 + half * 32;
    const auto* second_ql = second_block + half * 64;
    const auto* second_qh = second_block + 128 + half * 32;
    const auto* first_half_scales = first_scales + half * 8;
    const auto* second_half_scales = second_scales + half * 8;
    const int base = half * 128;
    for (int scale_group = 0; scale_group < 2; ++scale_group) {
      const int lane_begin = scale_group * 16;
      const int first_scale_0 = static_cast<int>(first_half_scales[scale_group]);
      const int first_scale_1 =
          static_cast<int>(first_half_scales[scale_group + 2]);
      const int first_scale_2 =
          static_cast<int>(first_half_scales[scale_group + 4]);
      const int first_scale_3 =
          static_cast<int>(first_half_scales[scale_group + 6]);
      const int second_scale_0 =
          static_cast<int>(second_half_scales[scale_group]);
      const int second_scale_1 =
          static_cast<int>(second_half_scales[scale_group + 2]);
      const int second_scale_2 =
          static_cast<int>(second_half_scales[scale_group + 4]);
      const int second_scale_3 =
          static_cast<int>(second_half_scales[scale_group + 6]);
      for (int lane = lane_begin; lane < lane_begin + 16; ++lane) {
        const int q8_0 = static_cast<int>(q8[base + lane]);
        const int q8_1 = static_cast<int>(q8[base + 32 + lane]);
        const int q8_2 = static_cast<int>(q8[base + 64 + lane]);
        const int q8_3 = static_cast<int>(q8[base + 96 + lane]);
        const int first_high = static_cast<int>(first_qh[lane]);
        const int second_high = static_cast<int>(second_qh[lane]);
        const auto lane_index = static_cast<std::size_t>(lane & 7);
        first_lane_sums[lane_index] +=
            first_scale_0 * q8_0 *
            ((static_cast<int>(first_ql[lane] & 0x0f) |
              (((first_high >> 0) & 3) << 4)) -
             32);
        first_lane_sums[lane_index] +=
            first_scale_1 * q8_1 *
            ((static_cast<int>(first_ql[32 + lane] & 0x0f) |
              (((first_high >> 2) & 3) << 4)) -
             32);
        first_lane_sums[lane_index] +=
            first_scale_2 * q8_2 *
            ((static_cast<int>(first_ql[lane] >> 4) |
              (((first_high >> 4) & 3) << 4)) -
             32);
        first_lane_sums[lane_index] +=
            first_scale_3 * q8_3 *
            ((static_cast<int>(first_ql[32 + lane] >> 4) |
              (((first_high >> 6) & 3) << 4)) -
             32);
        second_lane_sums[lane_index] +=
            second_scale_0 * q8_0 *
            ((static_cast<int>(second_ql[lane] & 0x0f) |
              (((second_high >> 0) & 3) << 4)) -
             32);
        second_lane_sums[lane_index] +=
            second_scale_1 * q8_1 *
            ((static_cast<int>(second_ql[32 + lane] & 0x0f) |
              (((second_high >> 2) & 3) << 4)) -
             32);
        second_lane_sums[lane_index] +=
            second_scale_2 * q8_2 *
            ((static_cast<int>(second_ql[lane] >> 4) |
              (((second_high >> 4) & 3) << 4)) -
             32);
        second_lane_sums[lane_index] +=
            second_scale_3 * q8_3 *
            ((static_cast<int>(second_ql[32 + lane] >> 4) |
              (((second_high >> 6) & 3) << 4)) -
             32);
      }
    }
  }
  for (int lane = 0; lane < 8; ++lane) {
    const auto lane_index = static_cast<std::size_t>(lane);
    first_sums[lane] +=
        first_combined_scale * static_cast<float>(first_lane_sums[lane_index]);
    second_sums[lane] +=
        second_combined_scale * static_cast<float>(second_lane_sums[lane_index]);
  }
}

float dot_q6_k_q8_k_row(const std::vector<std::uint8_t>& bytes,
                        const std::vector<Q8KBlock>& q8_input) {
  if (bytes.size() != q8_input.size() * 210) {
    throw std::runtime_error("Q6_K row byte count does not match Q8_K input");
  }
  float sums[8] = {};
  for (std::size_t block = 0; block < q8_input.size(); ++block) {
    accumulate_q6_k_q8_k_block(bytes.data() + block * 210,
                               q8_input[block],
                               sums);
  }
  float sum = 0.0f;
  for (const float lane_sum : sums) {
    sum += lane_sum;
  }
  return sum;
}

float dot_q6_k_q8_k_row(const std::uint8_t* bytes,
                        std::size_t byte_count,
                        const std::vector<Q8KBlock>& q8_input) {
  if (byte_count != q8_input.size() * 210) {
    throw std::runtime_error("Q6_K row byte count does not match Q8_K input");
  }
  float sums[8] = {};
  for (std::size_t block = 0; block < q8_input.size(); ++block) {
    accumulate_q6_k_q8_k_block(bytes + block * 210,
                               q8_input[block],
                               sums);
  }
  float sum = 0.0f;
  for (const float lane_sum : sums) {
    sum += lane_sum;
  }
  return sum;
}

float dot_q6_k_q8_k_row_direct(const std::uint8_t* bytes,
                               std::size_t byte_count,
                               const std::vector<Q8KBlock>& q8_input) {
  if (byte_count != q8_input.size() * 210) {
    throw std::runtime_error("Q6_K row byte count does not match Q8_K input");
  }
  float sums[8] = {};
  for (std::size_t block = 0; block < q8_input.size(); ++block) {
    accumulate_q6_k_q8_k_block_direct(bytes + block * 210,
                                      q8_input[block],
                                      sums);
  }
  float sum = 0.0f;
  for (const float lane_sum : sums) {
    sum += lane_sum;
  }
  return sum;
}

void dot_q6_k_q8_k_row_pair_direct(const std::uint8_t* first_bytes,
                                   const std::uint8_t* second_bytes,
                                   std::size_t byte_count,
                                   const std::vector<Q8KBlock>& q8_input,
                                   float& first,
                                   float& second) {
  if (byte_count != q8_input.size() * 210) {
    throw std::runtime_error("Q6_K row byte count does not match Q8_K input");
  }
  float first_sums[8] = {};
  float second_sums[8] = {};
  for (std::size_t block = 0; block < q8_input.size(); ++block) {
    accumulate_q6_k_q8_k_block_pair_direct(
        first_bytes + block * 210,
        second_bytes + block * 210,
        q8_input[block],
        first_sums,
        second_sums);
  }
  first = 0.0f;
  second = 0.0f;
  for (int lane = 0; lane < 8; ++lane) {
    first += first_sums[lane];
    second += second_sums[lane];
  }
}

void append_decoded_blocks(std::vector<float>& values,
                           const std::vector<std::uint8_t>& bytes,
                           std::size_t block_size,
                           const char* type_name) {
  if (bytes.size() % block_size != 0) {
    throw std::runtime_error(std::string(type_name) + " row byte count is not block-aligned");
  }
  const auto block_count = bytes.size() / block_size;
  values.reserve(values.size() + block_count * 256);
  for (std::size_t block = 0; block < block_count; ++block) {
    const auto begin = bytes.begin() +
                       static_cast<std::vector<std::uint8_t>::difference_type>(block * block_size);
    const auto end = begin +
                     static_cast<std::vector<std::uint8_t>::difference_type>(block_size);
    const std::vector<std::uint8_t> block_bytes(begin, end);
    const auto decoded =
        block_size == 144 ? decode_q4_k_block(block_bytes)
                          : decode_q6_k_block(block_bytes);
    values.insert(values.end(), decoded.begin(), decoded.end());
  }
}

std::vector<float> decode_tensor_row_payload(std::uint32_t type,
                                             const std::vector<std::uint8_t>& bytes,
                                             std::uint64_t row_elements) {
  std::vector<float> values;
  if (type == 0) {
    values = decode_f32_values(bytes, static_cast<std::size_t>(row_elements));
  } else if (type == 12) {
    append_decoded_blocks(values, bytes, 144, "Q4_K");
  } else if (type == 14) {
    append_decoded_blocks(values, bytes, 210, "Q6_K");
  } else {
    throw std::invalid_argument("unsupported tensor type for row decode");
  }
  if (values.size() < row_elements) {
    throw std::runtime_error("decoded row has fewer values than expected");
  }
  values.resize(static_cast<std::size_t>(row_elements));
  return values;
}

float dot_product_float(const std::vector<float>& lhs,
                        const std::vector<float>& rhs) {
  if (lhs.size() != rhs.size()) {
    throw std::invalid_argument("dot product vector sizes differ");
  }
  float sum = 0.0f;
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    sum += lhs[i] * rhs[i];
  }
  return sum;
}

float dot_tensor_row_payload(std::uint32_t type,
                             const std::vector<std::uint8_t>& bytes,
                             const std::vector<float>& input,
                             std::uint64_t row_elements,
                             const std::vector<Q8KBlock>& q8_input) {
  if (input.size() != row_elements) {
    throw std::invalid_argument("matvec input size does not match tensor row");
  }
  if (type == 0) {
    const auto values = decode_f32_values(bytes, static_cast<std::size_t>(row_elements));
    if (values.size() < row_elements) {
      throw std::runtime_error("F32 row has fewer values than expected");
    }
    return dot_product_float(values, input);
  }
  if (row_elements % 256 != 0) {
    throw std::runtime_error("quantized matvec row is not block-aligned");
  }
  const auto block_count = static_cast<std::size_t>(row_elements / 256);
  if (type == 12) {
    if (q8_input.size() != block_count) {
      throw std::runtime_error("Q8_K input block count does not match Q4_K row");
    }
    return dot_q4_k_q8_k_row(bytes, q8_input);
  }
  if (type == 14) {
    if (q8_input.size() != block_count) {
      throw std::runtime_error("Q8_K input block count does not match Q6_K row");
    }
    return dot_q6_k_q8_k_row(bytes, q8_input);
  }
  throw std::invalid_argument("unsupported tensor type for matvec");
}

float dot_tensor_row_payload(std::uint32_t type,
                             const std::uint8_t* bytes,
                             std::size_t byte_count,
                             const std::vector<float>& input,
                             std::uint64_t row_elements,
                             const std::vector<Q8KBlock>& q8_input) {
  if (input.size() != row_elements) {
    throw std::invalid_argument("matvec input size does not match tensor row");
  }
  if (type == 0) {
    if (byte_count < row_elements * sizeof(float)) {
      throw std::runtime_error("F32 row has fewer bytes than expected");
    }
    float sum = 0.0f;
    for (std::uint64_t i = 0; i < row_elements; ++i) {
      sum += read_le_f32_ptr(bytes + static_cast<std::size_t>(i * sizeof(float))) *
             input[static_cast<std::size_t>(i)];
    }
    return sum;
  }
  if (row_elements % 256 != 0) {
    throw std::runtime_error("quantized matvec row is not block-aligned");
  }
  const auto block_count = static_cast<std::size_t>(row_elements / 256);
  if (type == 12) {
    if (q8_input.size() != block_count) {
      throw std::runtime_error("Q8_K input block count does not match Q4_K row");
    }
    return dot_q4_k_q8_k_row(bytes, byte_count, q8_input);
  }
  if (type == 14) {
    if (q8_input.size() != block_count) {
      throw std::runtime_error("Q8_K input block count does not match Q6_K row");
    }
    return dot_q6_k_q8_k_row(bytes, byte_count, q8_input);
  }
  throw std::invalid_argument("unsupported tensor type for matvec");
}

TensorPayloadStats stats_from_values(const GgufTensorInfo& tensor,
                                     const std::vector<float>& values) {
  TensorPayloadStats stats;
  stats.name = tensor.name;
  stats.type_name = ggml_type_name(tensor.type);
  stats.absolute_offset = tensor.absolute_offset;
  stats.nbytes = tensor.nbytes;
  stats.decoded_values = values.size();
  stats.min = std::numeric_limits<double>::infinity();
  stats.max = -std::numeric_limits<double>::infinity();
  stats.finite = !values.empty();
  for (const float value : values) {
    if (!std::isfinite(value)) {
      stats.finite = false;
      continue;
    }
    const double as_double = value;
    stats.min = std::min(stats.min, as_double);
    stats.max = std::max(stats.max, as_double);
    stats.sum += as_double;
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

bool should_cache_decoded_row(const GgufTensorInfo& tensor,
                              std::uint64_t row_nbytes) {
  if (tensor.name == "output.weight") {
    return false;
  }
  return row_nbytes > 0 && row_nbytes <= kMaxDecodedRowCacheBytes;
}

bool is_q4_plane_layout_suffix(const std::string& suffix);

bool should_cache_tensor_payload(const GgufTensorInfo& tensor) {
  if (tensor.name == "output.weight") {
    return true;
  }
  if (q4_plane_layout_enabled() && tensor.type == 12 &&
      is_q4_plane_layout_suffix(tensor.suffix)) {
    return true;
  }
  if (!dense_matvec_state().enabled ||
      !dense_matvec_payload_cache_state().enabled) {
    return false;
  }
  if (tensor.suffix == "ffn_gate_inp_shexp.weight") {
    return tensor.type == 0;
  }
  if (tensor.dims.size() != 2) {
    return false;
  }
  if (tensor.suffix == "attn_qkv.weight") {
    return tensor.type == 12 || tensor.type == 14;
  }
  if (tensor.suffix == "ffn_gate_inp.weight") {
    return tensor.type == 0;
  }
  if (tensor.suffix == "ffn_down_shexp.weight") {
    return tensor.type == 12 || tensor.type == 14;
  }
  return (tensor.suffix == "attn_gate.weight" ||
          tensor.suffix == "attn_output.weight" ||
          tensor.suffix == "attn_q.weight" ||
          tensor.suffix == "ffn_gate_shexp.weight" ||
          tensor.suffix == "ffn_up_shexp.weight" ||
          tensor.suffix == "ssm_alpha.weight" ||
          tensor.suffix == "ssm_beta.weight" ||
          tensor.suffix == "ssm_out.weight") &&
         tensor.type == 12;
}

bool is_q4_plane_layout_suffix(const std::string& suffix) {
  return suffix == "attn_gate.weight" ||
         suffix == "attn_k.weight" ||
         suffix == "attn_output.weight" ||
         suffix == "attn_q.weight" ||
         suffix == "attn_qkv.weight" ||
         suffix == "attn_v.weight" ||
         suffix == "ffn_down_exps.weight" ||
         suffix == "ffn_down_shexp.weight" ||
         suffix == "ffn_gate_shexp.weight" ||
         suffix == "ffn_gate_up_exps.weight" ||
         suffix == "ffn_up_shexp.weight" ||
         suffix == "ssm_out.weight";
}

bool should_use_q4_plane_layout_route(const GgufTensorInfo& tensor) {
  if (!q4_plane_layout_enabled() || tensor.type != 12) {
    return false;
  }
  return is_q4_plane_layout_suffix(tensor.suffix);
}

std::vector<std::uint8_t> read_tensor_bytes_uncached(
    const std::string& path,
    const GgufTensorInfo& tensor,
    std::uint64_t relative_offset,
    std::uint64_t byte_count) {
  if (relative_offset > tensor.nbytes || byte_count > tensor.nbytes - relative_offset) {
    throw std::runtime_error("tensor read range exceeds tensor payload");
  }
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::invalid_argument("GGUF model could not be opened for tensor read");
  }
  input.seekg(static_cast<std::streamoff>(tensor.absolute_offset + relative_offset));
  if (!input) {
    throw std::runtime_error("tensor seek failed");
  }
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(byte_count));
  input.read(reinterpret_cast<char*>(bytes.data()),
             static_cast<std::streamsize>(bytes.size()));
  if (!input) {
    throw std::runtime_error("tensor read failed");
  }
  return bytes;
}

const std::vector<std::uint8_t>* cached_tensor_payload(
    const std::string& path,
    const GgufTensorInfo& tensor) {
  auto& cache = resident_cache();
  if (!cache.enabled || !should_cache_tensor_payload(tensor)) {
    return nullptr;
  }
  const auto key = cache_key(path, tensor);
  const auto found = cache.tensor_payloads.find(key);
  if (found != cache.tensor_payloads.end()) {
    ++cache.stats.tensor_payload_hits;
    return &found->second;
  }
  ++cache.stats.tensor_payload_misses;
  auto payload = read_tensor_bytes_uncached(path, tensor, 0, tensor.nbytes);
  cache.stats.tensor_payload_cached_bytes += payload.size();
  const auto inserted =
      cache.tensor_payloads.emplace(key, std::move(payload));
  return &inserted.first->second;
}

const std::vector<Q4KBlockMeta>* cached_q4_block_meta(
    const std::string& path,
    const GgufTensorInfo& tensor,
    const std::uint8_t* payload_data,
    std::size_t payload_size) {
  auto& cache = resident_cache();
  if (!cache.enabled || !q4_block_meta_cache_state().enabled ||
      tensor.type != 12) {
    return nullptr;
  }
  if (payload_size != tensor.nbytes || payload_size % 144 != 0) {
    throw std::runtime_error("Q4 block metadata cache payload size mismatch");
  }
  const auto key = cache_key(path, tensor);
  const auto found = cache.q4_block_meta.find(key);
  if (found != cache.q4_block_meta.end()) {
    return &found->second;
  }

  std::vector<Q4KBlockMeta> meta(payload_size / 144);
  for (std::size_t block = 0; block < meta.size(); ++block) {
    meta[block] = decode_q4_k_block_meta(payload_data + block * 144);
  }
  const auto inserted = cache.q4_block_meta.emplace(key, std::move(meta));
  return &inserted.first->second;
}

const Q4KPlaneRows* cached_q4_plane_rows(
    const std::string& path,
    const GgufTensorInfo& tensor,
    const std::uint8_t* payload_data,
    std::size_t payload_size,
    std::uint64_t row_count,
    std::uint64_t row_nbytes) {
  auto& cache = resident_cache();
  if (!cache.enabled || !should_use_q4_plane_layout_route(tensor)) {
    return nullptr;
  }
  const auto key = q4_plane_rows_cache_key(path, tensor, row_count, row_nbytes);
  const auto found = cache.q4_plane_rows.find(key);
  if (found != cache.q4_plane_rows.end()) {
    ++cache.stats.q4_plane_hits;
    return &found->second;
  }
  ++cache.stats.q4_plane_misses;
  const auto repack_begin = ProfileClock::now();
  auto plane = make_q4_plane_rows(payload_data, payload_size, row_count, row_nbytes);
  cache.stats.q4_plane_repack_ns +=
      profile_elapsed_ns(repack_begin, ProfileClock::now());
  cache.stats.q4_plane_cached_bytes += q4_plane_cached_bytes(plane);
  const auto inserted = cache.q4_plane_rows.emplace(key, std::move(plane));
  return &inserted.first->second;
}

const Q4KPlaneRows* cached_q4_plane_expert_slice(
    const std::string& path,
    const GgufTensorInfo& tensor,
    std::uint64_t rows_per_expert,
    std::uint64_t row_nbytes,
    std::int32_t expert_id,
    const std::uint8_t* payload_data,
    std::size_t payload_size) {
  auto& cache = resident_cache();
  if (!cache.enabled || !should_use_q4_plane_layout_route(tensor)) {
    return nullptr;
  }
  const auto key = q4_plane_expert_slice_cache_key(
      path, tensor, rows_per_expert, row_nbytes, expert_id);
  const auto found = cache.q4_plane_rows.find(key);
  if (found != cache.q4_plane_rows.end()) {
    ++cache.stats.q4_plane_hits;
    return &found->second;
  }
  ++cache.stats.q4_plane_misses;
  const auto repack_begin = ProfileClock::now();
  auto plane =
      make_q4_plane_rows(payload_data, payload_size, rows_per_expert, row_nbytes);
  cache.stats.q4_plane_repack_ns +=
      profile_elapsed_ns(repack_begin, ProfileClock::now());
  cache.stats.q4_plane_cached_bytes += q4_plane_cached_bytes(plane);
  const auto inserted = cache.q4_plane_rows.emplace(key, std::move(plane));
  return &inserted.first->second;
}

const std::vector<std::uint8_t>* cached_expert_slice(
    const std::string& path,
    const GgufTensorInfo& tensor,
    std::uint64_t rows_per_expert,
    std::uint64_t row_nbytes,
    std::int32_t expert_id) {
  auto& cache = resident_cache();
  if (!cache.enabled) {
    return nullptr;
  }
  const auto key =
      expert_slice_cache_key(path, tensor, rows_per_expert, row_nbytes, expert_id);
  const auto found = cache.expert_slices.find(key);
  if (found != cache.expert_slices.end()) {
    ++cache.stats.expert_slice_hits;
    return &found->second;
  }

  const std::uint64_t expert_row_base =
      static_cast<std::uint64_t>(expert_id) * rows_per_expert;
  const std::uint64_t relative_offset = expert_row_base * row_nbytes;
  const std::uint64_t byte_count = rows_per_expert * row_nbytes;
  ++cache.stats.expert_slice_misses;
  auto bytes = read_tensor_bytes_uncached(path, tensor, relative_offset, byte_count);
  cache.stats.expert_slice_cached_bytes += bytes.size();
  const auto inserted = cache.expert_slices.emplace(key, std::move(bytes));
  return &inserted.first->second;
}

std::vector<std::uint8_t> read_tensor_bytes(const std::string& path,
                                            const GgufTensorInfo& tensor,
                                            std::uint64_t relative_offset,
                                            std::uint64_t byte_count) {
  if (relative_offset == 0 && byte_count == tensor.nbytes) {
    const auto* cached = cached_tensor_payload(path, tensor);
    if (cached != nullptr) {
      return *cached;
    }
  }
  return read_tensor_bytes_uncached(path, tensor, relative_offset, byte_count);
}

std::vector<std::uint8_t> read_tensor_prefix(const std::string& path,
                                             const GgufTensorInfo& tensor,
                                             std::uint64_t byte_count) {
  return read_tensor_bytes(path, tensor, 0, byte_count);
}

std::string layer_tensor_name(int layer_index, const std::string& suffix) {
  if (layer_index < 0) {
    throw std::invalid_argument("layer index must be non-negative");
  }
  return "blk." + std::to_string(layer_index) + "." + suffix;
}

float softplus_scalar(float value) {
  return value > 20.0f ? value : std::log(1.0f + std::exp(value));
}

std::vector<float> sigmoid_vector(const std::vector<float>& input) {
  std::vector<float> output;
  output.reserve(input.size());
  for (const auto value : input) {
    output.push_back(sigmoid_scalar(value));
  }
  return output;
}

std::vector<float> softplus_vector(const std::vector<float>& input) {
  std::vector<float> output;
  output.reserve(input.size());
  for (const auto value : input) {
    output.push_back(softplus_scalar(value));
  }
  return output;
}

std::vector<float> multiply_vectors_checked(const std::vector<float>& lhs,
                                            const std::vector<float>& rhs,
                                            const char* message) {
  if (lhs.empty()) {
    throw std::invalid_argument("vector multiply lhs is empty");
  }
  if (lhs.size() != rhs.size()) {
    throw std::invalid_argument(message);
  }

  std::vector<float> output;
  output.reserve(lhs.size());
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    output.push_back(lhs[i] * rhs[i]);
  }
  return output;
}

bool matvec_top_k_better(const MatvecTopKRow& lhs,
                         const MatvecTopKRow& rhs) {
  if (lhs.value == rhs.value) {
    return lhs.token_id < rhs.token_id;
  }
  return lhs.value > rhs.value;
}

void insert_matvec_top_k(std::vector<MatvecTopKRow>& rows,
                         const MatvecTopKRow& candidate,
                         int k) {
  rows.push_back(candidate);
  std::sort(rows.begin(), rows.end(), matvec_top_k_better);
  if (rows.size() > static_cast<std::size_t>(k)) {
    rows.resize(static_cast<std::size_t>(k));
  }
}

bool should_use_dense_matvec_route(const GgufTensorInfo& tensor,
                                   std::uint64_t row_count) {
  const auto& state = dense_matvec_state();
  return state.enabled &&
         state.thread_count > 1 &&
         tensor.name != "output.weight" &&
         row_count >= state.min_rows;
}

bool should_use_dense_q4_direct_dot_route(const GgufTensorInfo& tensor) {
  return dense_matvec_state().enabled &&
         dense_q4_direct_dot_state().enabled &&
         tensor.type == 12;
}

bool should_use_dense_q4_pair_dot_route(const GgufTensorInfo& tensor) {
  return dense_matvec_state().enabled &&
         dense_q4_direct_dot_state().enabled &&
         dense_q4_pair_dot_state().enabled &&
         tensor.type == 12;
}

bool should_use_dense_q6_direct_dot_route(const GgufTensorInfo& tensor) {
  return dense_matvec_state().enabled &&
         dense_q6_direct_dot_state().enabled &&
         tensor.type == 14;
}

bool should_use_dense_q6_pair_dot_route(const GgufTensorInfo& tensor) {
  return dense_matvec_state().enabled &&
         dense_q6_direct_dot_state().enabled &&
         dense_q6_pair_dot_state().enabled &&
         tensor.type == 14;
}

bool should_use_small_q4_direct_dot_route(const GgufTensorInfo& tensor,
                                          std::uint64_t row_count) {
  return small_q4_direct_dot_state().enabled &&
         dense_q4_direct_dot_state().enabled &&
         tensor.type == 12 &&
         tensor.name != "output.weight" &&
         row_count < dense_matvec_state().min_rows;
}

}  // namespace

std::string ggml_type_name(std::uint32_t type) {
  switch (type) {
    case 0:
      return "F32";
    case 12:
      return "Q4_K";
    case 14:
      return "Q6_K";
    default:
      return "UNKNOWN_" + std::to_string(type);
  }
}

std::uint64_t ggml_tensor_nbytes(std::uint32_t type,
                                 const std::vector<std::uint64_t>& dims) {
  const auto elements = tensor_element_count(dims);
  if (type == 0) {
    return elements * 4;
  }
  if (type == 12) {
    return ((elements + 255) / 256) * 144;
  }
  if (type == 14) {
    return ((elements + 255) / 256) * 210;
  }
  return 0;
}

GgufModelIndex parse_gguf_model_index(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::invalid_argument("GGUF model could not be opened");
  }

  char magic[4] = {};
  input.read(magic, 4);
  if (!input || std::string(magic, 4) != "GGUF") {
    throw std::invalid_argument("model is not a GGUF file");
  }

  GgufModelIndex index;
  index.file_size_bytes =
      static_cast<std::uint64_t>(std::filesystem::file_size(path));
  index.version = read_u32(input);
  index.tensor_count = read_u64(input);
  index.metadata_kv_count = read_u64(input);

  for (std::uint64_t i = 0; i < index.metadata_kv_count; ++i) {
    const auto key = read_string(input);
    const auto value_type = read_u32(input);
    index.metadata.emplace(key, read_metadata_value(input, value_type));
  }

  static const std::regex layer_regex("^blk\\.([0-9]+)\\.(.+)$");
  for (std::uint64_t i = 0; i < index.tensor_count; ++i) {
    GgufTensorInfo tensor;
    tensor.name = read_string(input);
    const auto ndims = read_u32(input);
    for (std::uint32_t dim = 0; dim < ndims; ++dim) {
      tensor.dims.push_back(read_u64(input));
    }
    tensor.type = read_u32(input);
    tensor.offset = read_u64(input);

    std::smatch match;
    if (std::regex_match(tensor.name, match, layer_regex)) {
      tensor.layer_index = std::stoi(match[1].str());
      tensor.suffix = match[2].str();
    }
    index.tensors.push_back(std::move(tensor));
  }

  const auto tensor_info_end = static_cast<std::uint64_t>(input.tellg());
  const auto alignment = metadata_uint(index, "general.alignment", 32);
  index.data_section_offset = align_up(tensor_info_end, alignment);
  for (auto& tensor : index.tensors) {
    tensor.absolute_offset = index.data_section_offset + tensor.offset;
    tensor.nbytes = ggml_tensor_nbytes(tensor.type, tensor.dims);
  }
  return index;
}

void set_resident_tensor_cache_enabled(bool enabled) {
  auto& cache = resident_cache();
  cache.enabled = enabled;
  cache.stats.enabled = enabled;
}

void reset_resident_tensor_cache() {
  auto& cache = resident_cache();
  const bool enabled = cache.enabled;
  cache.decoded_rows.clear();
  cache.tensor_payloads.clear();
  cache.q4_block_meta.clear();
  cache.q4_plane_rows.clear();
  cache.expert_slices.clear();
  cache.stats = ResidentTensorCacheStats{};
  cache.stats.enabled = enabled;
}

ResidentTensorCacheStats resident_tensor_cache_stats() {
  auto& cache = resident_cache();
  cache.stats.enabled = cache.enabled;
  return cache.stats;
}

void set_matvec_profile_enabled(bool enabled) {
  matvec_profile().enabled = enabled;
}

void reset_matvec_profile() {
  auto& profile = matvec_profile();
  const bool enabled = profile.enabled;
  profile.rows.clear();
  profile.enabled = enabled;
}

std::vector<MatvecProfileRow> matvec_profile_rows() {
  const auto& profile = matvec_profile();
  std::vector<MatvecProfileRow> rows;
  rows.reserve(profile.rows.size());
  for (const auto& item : profile.rows) {
    rows.push_back(item.second);
  }
  std::sort(rows.begin(), rows.end(), [](const auto& lhs, const auto& rhs) {
    if (lhs.total_ns == rhs.total_ns) {
      return lhs.tensor_name < rhs.tensor_name;
    }
    return lhs.total_ns > rhs.total_ns;
  });
  return rows;
}

void set_expert_slice_matvec_enabled(bool enabled) {
  expert_slice_matvec_state().enabled = enabled;
}

bool expert_slice_matvec_enabled() {
  return expert_slice_matvec_state().enabled;
}

void set_expert_slice_matvec_thread_count(int thread_count) {
  if (thread_count < 1) {
    throw std::invalid_argument("expert-slice matvec thread count must be positive");
  }
  expert_slice_matvec_state().thread_count = thread_count;
}

int expert_slice_matvec_thread_count() {
  return expert_slice_matvec_state().thread_count;
}

void set_dense_matvec_enabled(bool enabled) {
  dense_matvec_state().enabled = enabled;
}

bool dense_matvec_enabled() {
  return dense_matvec_state().enabled;
}

void set_dense_matvec_thread_count(int thread_count) {
  if (thread_count < 1) {
    throw std::invalid_argument("dense matvec thread count must be positive");
  }
  dense_matvec_state().thread_count = thread_count;
}

int dense_matvec_thread_count() {
  return dense_matvec_state().thread_count;
}

void set_dense_matvec_min_rows(std::uint64_t min_rows) {
  if (min_rows == 0) {
    throw std::invalid_argument("dense matvec min rows must be positive");
  }
  dense_matvec_state().min_rows = min_rows;
}

std::uint64_t dense_matvec_min_rows() {
  return dense_matvec_state().min_rows;
}

void set_dense_matvec_payload_cache_enabled(bool enabled) {
  dense_matvec_payload_cache_state().enabled = enabled;
}

bool dense_matvec_payload_cache_enabled() {
  return dense_matvec_payload_cache_state().enabled;
}

void set_dense_q4_direct_dot_enabled(bool enabled) {
  dense_q4_direct_dot_state().enabled = enabled;
}

bool dense_q4_direct_dot_enabled() {
  return dense_q4_direct_dot_state().enabled;
}

void set_dense_q4_pair_dot_enabled(bool enabled) {
  dense_q4_pair_dot_state().enabled = enabled;
}

bool dense_q4_pair_dot_enabled() {
  return dense_q4_pair_dot_state().enabled;
}

void set_dense_q6_direct_dot_enabled(bool enabled) {
  dense_q6_direct_dot_state().enabled = enabled;
}

bool dense_q6_direct_dot_enabled() {
  return dense_q6_direct_dot_state().enabled;
}

void set_dense_q6_pair_dot_enabled(bool enabled) {
  dense_q6_pair_dot_state().enabled = enabled;
}

bool dense_q6_pair_dot_enabled() {
  return dense_q6_pair_dot_state().enabled;
}

void set_lm_head_q6_pair_dot_enabled(bool enabled) {
  lm_head_q6_pair_dot_state().enabled = enabled;
}

bool lm_head_q6_pair_dot_enabled() {
  return lm_head_q6_pair_dot_state().enabled;
}

void set_q4_direct_minsum_pair_enabled(bool enabled) {
  q4_direct_minsum_pair_state().enabled = enabled;
}

bool q4_direct_minsum_pair_enabled() {
  return q4_direct_minsum_pair_state().enabled;
}

void set_q4_block_meta_cache_enabled(bool enabled) {
  q4_block_meta_cache_state().enabled = enabled;
}

bool q4_block_meta_cache_enabled() {
  return q4_block_meta_cache_state().enabled;
}

void set_q4_plane_layout_enabled(bool enabled) {
  q4_plane_layout_state().enabled = enabled;
}

bool q4_plane_layout_enabled() {
  return q4_plane_layout_state().enabled;
}

void set_dense_q4_plane_pair_dot_enabled(bool enabled) {
  dense_q4_plane_pair_dot_state().enabled = enabled;
}

bool dense_q4_plane_pair_dot_enabled() {
  return dense_q4_plane_pair_dot_state().enabled;
}

void set_small_q4_direct_dot_enabled(bool enabled) {
  small_q4_direct_dot_state().enabled = enabled;
}

bool small_q4_direct_dot_enabled() {
  return small_q4_direct_dot_state().enabled;
}

void set_matvec_q8_input_reuse_enabled(bool enabled) {
  matvec_q8_input_reuse_state().enabled = enabled;
}

bool matvec_q8_input_reuse_enabled() {
  return matvec_q8_input_reuse_state().enabled;
}

void set_shared_parallel_executor_enabled(bool enabled) {
  shared_parallel_executor_state().enabled = enabled;
}

bool shared_parallel_executor_enabled() {
  return shared_parallel_executor_state().enabled;
}

void set_shared_expert_gate_up_fused_enabled(bool enabled) {
  shared_expert_gate_up_fused_state().enabled = enabled;
}

bool shared_expert_gate_up_fused_enabled() {
  return shared_expert_gate_up_fused_state().enabled;
}

void set_selected_expert_ffn_enabled(bool enabled) {
  selected_expert_ffn_state().enabled = enabled;
}

bool selected_expert_ffn_enabled() {
  return selected_expert_ffn_state().enabled;
}

void set_selected_expert_ffn_thread_count(int thread_count) {
  if (thread_count < 1) {
    throw std::invalid_argument("selected-expert FFN thread count must be positive");
  }
  selected_expert_ffn_state().thread_count = thread_count;
}

int selected_expert_ffn_thread_count() {
  return selected_expert_ffn_state().thread_count;
}

void set_selected_expert_minimal_outputs_enabled(bool enabled) {
  selected_expert_minimal_outputs_state().enabled = enabled;
}

bool selected_expert_minimal_outputs_enabled() {
  return selected_expert_minimal_outputs_state().enabled;
}

void set_selected_expert_slice_cache_enabled(bool enabled) {
  selected_expert_slice_cache_state().enabled = enabled;
}

bool selected_expert_slice_cache_enabled() {
  return selected_expert_slice_cache_state().enabled;
}

void set_selected_expert_down_slice_cache_enabled(bool enabled) {
  selected_expert_down_slice_cache_state().enabled = enabled;
}

bool selected_expert_down_slice_cache_enabled() {
  return selected_expert_down_slice_cache_state().enabled;
}

void set_selected_expert_down_expert_major_enabled(bool enabled) {
  selected_expert_down_expert_major_state().enabled = enabled;
}

bool selected_expert_down_expert_major_enabled() {
  return selected_expert_down_expert_major_state().enabled;
}

void set_selected_expert_down_q4_pair_dot_enabled(bool enabled) {
  selected_expert_down_q4_pair_dot_state().enabled = enabled;
}

bool selected_expert_down_q4_pair_dot_enabled() {
  return selected_expert_down_q4_pair_dot_state().enabled;
}

void set_selected_expert_down_q6_pair_dot_enabled(bool enabled) {
  selected_expert_down_q6_pair_dot_state().enabled = enabled;
}

bool selected_expert_down_q6_pair_dot_enabled() {
  return selected_expert_down_q6_pair_dot_state().enabled;
}

void set_selected_gate_q4_direct_dot_enabled(bool enabled) {
  selected_gate_q4_direct_dot_state().enabled = enabled;
}

bool selected_gate_q4_direct_dot_enabled() {
  return selected_gate_q4_direct_dot_state().enabled;
}

void set_selected_gate_q4_pair_dot_enabled(bool enabled) {
  selected_gate_q4_pair_dot_state().enabled = enabled;
}

bool selected_gate_q4_pair_dot_enabled() {
  return selected_gate_q4_pair_dot_state().enabled;
}

void set_selected_gate_q4_pair_sum_dot_enabled(bool enabled) {
  selected_gate_q4_pair_sum_dot_state().enabled = enabled;
}

bool selected_gate_q4_pair_sum_dot_enabled() {
  return selected_gate_q4_pair_sum_dot_state().enabled;
}

void set_selected_gate_q4_plane_pair_dot_enabled(bool enabled) {
  selected_gate_q4_plane_pair_dot_state().enabled = enabled;
}

bool selected_gate_q4_plane_pair_dot_enabled() {
  return selected_gate_q4_plane_pair_dot_state().enabled;
}

GgufLoadMapSummary validate_qwen36_load_map(const GgufModelIndex& index) {
  GgufLoadMapSummary summary;
  summary.tensor_count = static_cast<int>(index.tensors.size());
  summary.metadata_kv_count = static_cast<int>(index.metadata_kv_count);

  record_check(summary, index.version == 3, "gguf_v3_header");
  record_check(summary, index.file_size_bytes == kExpectedModelSize,
               "locked_model_size");
  record_check(summary, index.tensor_count == 693 && index.tensors.size() == 693,
               "tensor_count");
  record_check(summary, index.metadata_kv_count == 45, "metadata_kv_count");
  record_check(summary, metadata_string(index, "general.architecture") == "qwen35moe",
               "architecture");
  record_check(summary, metadata_uint(index, "general.file_type") == 15,
               "file_type_q4_k_m");

  const auto block_count = static_cast<int>(metadata_uint(index, "qwen35moe.block_count"));
  const auto interval = static_cast<int>(
      metadata_uint(index, "qwen35moe.full_attention_interval"));
  record_check(summary, block_count == 40, "block_count");
  record_check(summary, interval == 4, "full_attention_interval");
  record_check(summary, metadata_uint(index, "qwen35moe.context_length") == 262144,
               "context_length");
  record_check(summary, metadata_uint(index, "qwen35moe.embedding_length") == 2048,
               "embedding_length");
  record_check(summary, metadata_uint(index, "qwen35moe.expert_count") == 256,
               "expert_count");
  record_check(summary, metadata_uint(index, "qwen35moe.expert_used_count") == 8,
               "expert_used_count");

  std::map<int, std::map<std::string, const GgufTensorInfo*>> layers;
  std::unordered_set<std::string> non_layer;
  for (const auto& tensor : index.tensors) {
    summary.tensor_type_counts[ggml_type_name(tensor.type)] += 1;
    if (tensor.layer_index >= 0) {
      layers[tensor.layer_index][tensor.suffix] = &tensor;
    } else {
      non_layer.insert(tensor.name);
    }
  }

  std::vector<int> layer_indexes;
  for (const auto& item : layers) {
    layer_indexes.push_back(item.first);
  }
  std::vector<int> expected_layers;
  for (int i = 0; i < 40; ++i) {
    expected_layers.push_back(i);
  }
  record_check(summary, layer_indexes == expected_layers, "layer_index_count");

  summary.full_attention_layers = expected_full_attention_layers(block_count, interval);
  record_check(summary,
               summary.full_attention_layers ==
                   std::vector<int>{3, 7, 11, 15, 19, 23, 27, 31, 35, 39},
               "full_attention_layer_indexes");

  record_check(summary, non_layer.count("token_embd.weight") == 1,
               "non_layer_token_embedding");
  record_check(summary, non_layer.count("output_norm.weight") == 1,
               "non_layer_output_norm");
  record_check(summary, non_layer.count("output.weight") == 1,
               "non_layer_output_weight");

  for (int layer = 0; layer < 40; ++layer) {
    const bool full_attention =
        std::find(summary.full_attention_layers.begin(),
                  summary.full_attention_layers.end(),
                  layer) != summary.full_attention_layers.end();
    const auto specs = full_attention ? full_attention_specs() : linear_ssm_specs();
    const auto found_layer = layers.find(layer);
    const auto& tensors = found_layer->second;
    record_check(summary, tensors.size() == specs.size(),
                 "layer_" + std::to_string(layer) + "_suffix_set");
    bool dims_types_ok = true;
    for (const auto& item : specs) {
      const auto found_tensor = tensors.find(item.first);
      dims_types_ok = dims_types_ok &&
                      found_tensor != tensors.end() &&
                      tensor_matches(found_tensor->second, item.second);
    }
    record_check(summary, dims_types_ok,
                 "layer_" + std::to_string(layer) + "_dims_types");
    summary.layer_summaries.push_back(GgufLayerSummary{
        layer,
        full_attention ? "full_attention" : "linear_ssm",
        static_cast<int>(tensors.size())});
    if (full_attention) {
      ++summary.full_attention_layer_count;
    } else {
      ++summary.linear_ssm_layer_count;
    }
  }

  record_check(summary, summary.linear_ssm_layer_count == 30,
               "linear_ssm_layer_count");
  record_check(summary, summary.full_attention_layer_count == 10,
               "full_attention_layer_count");
  record_check(summary,
               summary.tensor_type_counts["F32"] == 301 &&
                   summary.tensor_type_counts["Q4_K"] == 331 &&
                   summary.tensor_type_counts["Q6_K"] == 61,
               "tensor_type_counts");

  summary.ready = summary.failed_checks.empty();
  return summary;
}

const GgufTensorInfo* find_tensor(const GgufModelIndex& index,
                                  const std::string& name) {
  for (const auto& tensor : index.tensors) {
    if (tensor.name == name) {
      return &tensor;
    }
  }
  return nullptr;
}

TensorPayloadStats smoke_tensor_payload(const std::string& path,
                                        const GgufModelIndex& index,
                                        const std::string& tensor_name) {
  const auto* tensor = find_tensor(index, tensor_name);
  if (tensor == nullptr) {
    throw std::invalid_argument("tensor not found");
  }
  std::vector<float> values;
  if (tensor->type == 0) {
    const auto bytes = read_tensor_prefix(path, *tensor, std::min<std::uint64_t>(1024, tensor->nbytes));
    values = decode_f32_values(bytes, 256);
  } else if (tensor->type == 12) {
    const auto bytes = read_tensor_prefix(path, *tensor, 144);
    values = decode_q4_k_block(bytes);
  } else if (tensor->type == 14) {
    const auto bytes = read_tensor_prefix(path, *tensor, 210);
    values = decode_q6_k_block(bytes);
  } else {
    throw std::invalid_argument("unsupported tensor type for smoke decode");
  }
  return stats_from_values(*tensor, values);
}

std::vector<float> decode_tensor_row(const std::string& path,
                                     const GgufModelIndex& index,
                                     const std::string& tensor_name,
                                     std::uint64_t row_index) {
  const auto* tensor = find_tensor(index, tensor_name);
  if (tensor == nullptr) {
    throw std::invalid_argument("tensor not found");
  }
  if (tensor->dims.empty()) {
    throw std::invalid_argument("tensor has no dimensions");
  }

  const std::uint64_t row_elements = tensor->dims[0];
  if (row_elements == 0) {
    throw std::invalid_argument("tensor row has zero elements");
  }
  const auto total_elements = tensor_element_count(tensor->dims);
  const auto row_count = total_elements / row_elements;
  if (row_index >= row_count) {
    throw std::out_of_range("tensor row index out of range");
  }

  const auto row_nbytes =
      ggml_tensor_nbytes(tensor->type, std::vector<std::uint64_t>{row_elements});
  if (row_nbytes == 0) {
    throw std::invalid_argument("unsupported tensor type for row decode");
  }
  if (row_count > 0 && row_nbytes > tensor->nbytes / row_count) {
    throw std::runtime_error("tensor row byte size exceeds tensor payload");
  }

  auto& cache = resident_cache();
  if (cache.enabled && should_cache_decoded_row(*tensor, row_nbytes)) {
    const auto key = cache_key(path, *tensor, row_index);
    const auto found = cache.decoded_rows.find(key);
    if (found != cache.decoded_rows.end()) {
      ++cache.stats.decoded_row_hits;
      return found->second;
    }
    ++cache.stats.decoded_row_misses;
    const auto bytes =
        read_tensor_bytes(path, *tensor, row_index * row_nbytes, row_nbytes);
    auto values = decode_tensor_row_payload(tensor->type, bytes, row_elements);
    cache.stats.decoded_row_cached_values += values.size();
    cache.stats.decoded_row_cached_bytes += values.size() * sizeof(float);
    const auto inserted = cache.decoded_rows.emplace(key, std::move(values));
    return inserted.first->second;
  }

  const auto bytes = read_tensor_bytes(path, *tensor, row_index * row_nbytes, row_nbytes);
  return decode_tensor_row_payload(tensor->type, bytes, row_elements);
}

std::vector<float> read_f32_vector_file(const std::string& path) {
  const auto file_size = std::filesystem::file_size(path);
  if (file_size % sizeof(float) != 0) {
    throw std::invalid_argument("f32 vector file size is not divisible by 4");
  }
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::invalid_argument("f32 vector file could not be opened");
  }
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(file_size));
  input.read(reinterpret_cast<char*>(bytes.data()),
             static_cast<std::streamsize>(bytes.size()));
  if (!input) {
    throw std::runtime_error("f32 vector file read failed");
  }
  return decode_f32_values(bytes, bytes.size() / sizeof(float));
}

VectorCompareStats compare_vectors(const std::vector<float>& lhs,
                                   const std::vector<float>& rhs,
                                   double mismatch_threshold) {
  VectorCompareStats stats;
  stats.lhs_value_count = lhs.size();
  stats.rhs_value_count = rhs.size();
  stats.compared_value_count = std::min(lhs.size(), rhs.size());
  stats.same_size = lhs.size() == rhs.size();
  stats.finite = stats.compared_value_count > 0;
  if (!stats.same_size) {
    stats.mismatch_count +=
        static_cast<std::uint64_t>(
            lhs.size() > rhs.size() ? lhs.size() - rhs.size()
                                    : rhs.size() - lhs.size());
  }

  double diff_sum = 0.0;
  double diff_sq_sum = 0.0;
  double dot = 0.0;
  for (std::size_t i = 0; i < stats.compared_value_count; ++i) {
    const float lhs_value = lhs[i];
    const float rhs_value = rhs[i];
    if (!std::isfinite(lhs_value) || !std::isfinite(rhs_value)) {
      stats.finite = false;
      ++stats.mismatch_count;
      continue;
    }
    ++stats.finite_pair_count;
    const double left = lhs_value;
    const double right = rhs_value;
    const double diff = std::abs(left - right);
    stats.max_abs_diff = std::max(stats.max_abs_diff, diff);
    diff_sum += diff;
    diff_sq_sum += diff * diff;
    dot += left * right;
    stats.lhs_l2 += left * left;
    stats.rhs_l2 += right * right;
    if (diff > mismatch_threshold) {
      ++stats.mismatch_count;
    }
  }

  if (stats.finite_pair_count > 0) {
    stats.mean_abs_diff = diff_sum / static_cast<double>(stats.finite_pair_count);
    stats.rmse = std::sqrt(diff_sq_sum / static_cast<double>(stats.finite_pair_count));
  }
  if (stats.lhs_l2 > 0.0 && stats.rhs_l2 > 0.0) {
    stats.cosine = dot / (std::sqrt(stats.lhs_l2) * std::sqrt(stats.rhs_l2));
  }
  return stats;
}

std::vector<float> apply_rms_norm(const std::vector<float>& input,
                                  const std::vector<float>& weight,
                                  float epsilon) {
  if (input.empty()) {
    throw std::invalid_argument("RMSNorm input is empty");
  }
  if (input.size() != weight.size()) {
    throw std::invalid_argument("RMSNorm input and weight sizes differ");
  }

  float sum_squares = 0.0f;
  for (const auto value : input) {
    sum_squares += value * value;
  }
  const float mean_square = sum_squares / static_cast<float>(input.size());
  const float scale = 1.0f / std::sqrt(mean_square + epsilon);

  std::vector<float> output;
  output.reserve(input.size());
  for (std::size_t i = 0; i < input.size(); ++i) {
    output.push_back(input[i] * scale * weight[i]);
  }
  return output;
}

std::vector<float> add_vectors(const std::vector<float>& lhs,
                               const std::vector<float>& rhs) {
  if (lhs.empty()) {
    throw std::invalid_argument("vector add lhs is empty");
  }
  if (lhs.size() != rhs.size()) {
    throw std::invalid_argument("vector add sizes differ");
  }

  std::vector<float> output;
  output.reserve(lhs.size());
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    output.push_back(lhs[i] + rhs[i]);
  }
  return output;
}

std::vector<float> apply_repeated_rms_norm(const std::vector<float>& input,
                                           const std::vector<float>& weight,
                                           float epsilon) {
  if (input.empty()) {
    throw std::invalid_argument("repeated RMSNorm input is empty");
  }
  if (weight.empty()) {
    throw std::invalid_argument("repeated RMSNorm weight is empty");
  }
  if (input.size() % weight.size() != 0) {
    throw std::invalid_argument("repeated RMSNorm input/weight size mismatch");
  }

  std::vector<float> output;
  output.reserve(input.size());
  for (std::size_t base = 0; base < input.size(); base += weight.size()) {
    float sum_squares = 0.0f;
    for (std::size_t i = 0; i < weight.size(); ++i) {
      const float value = input[base + i];
      sum_squares += value * value;
    }
    const float mean_square =
        sum_squares / static_cast<float>(weight.size());
    const float scale = 1.0f / std::sqrt(mean_square + epsilon);
    for (std::size_t i = 0; i < weight.size(); ++i) {
      output.push_back(input[base + i] * scale * weight[i]);
    }
  }
  return output;
}

namespace {

constexpr float kPi = 3.14159265358979323846f;

float rope_yarn_ramp(float low, float high, std::int64_t i0) {
  const float y = (static_cast<float>(i0) / 2.0f - low) /
                  std::max(0.001f, high - low);
  return 1.0f - std::min(1.0f, std::max(0.0f, y));
}

void rope_yarn(float theta_extrap,
               float freq_scale,
               const std::array<float, 2>& corr_dims,
               std::int64_t i0,
               float ext_factor,
               float mscale,
               float& cos_theta,
               float& sin_theta) {
  const float theta_interp = freq_scale * theta_extrap;
  float theta = theta_interp;
  if (ext_factor != 0.0f) {
    const float ramp_mix =
        rope_yarn_ramp(corr_dims[0], corr_dims[1], i0) * ext_factor;
    theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
    mscale *= 1.0f + 0.1f * std::log(1.0f / freq_scale);
  }
  cos_theta = std::cos(theta) * mscale;
  sin_theta = std::sin(theta) * mscale;
}

float rope_yarn_corr_dim(std::uint64_t n_dims,
                         std::uint64_t n_ctx_orig,
                         float n_rot,
                         float base) {
  return static_cast<float>(n_dims) *
         std::log(static_cast<float>(n_ctx_orig) / (n_rot * 2.0f * kPi)) /
         (2.0f * std::log(base));
}

std::array<float, 2> rope_yarn_corr_dims(std::uint64_t n_dims,
                                         std::uint64_t n_ctx_orig,
                                         float freq_base,
                                         float beta_fast,
                                         float beta_slow) {
  const float start =
      std::floor(rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_base));
  const float end =
      std::ceil(rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_base));
  return {std::max(0.0f, start),
          std::min(static_cast<float>(n_dims - 1), end)};
}

std::array<int, 4> normalize_rope_sections(
    const std::vector<std::int64_t>& rope_sections,
    std::uint64_t rope_dimension_count) {
  if (rope_sections.size() < 4) {
    throw std::invalid_argument("RoPE dimension sections are incomplete");
  }
  std::array<int, 4> sections{};
  int section_sum = 0;
  for (std::size_t i = 0; i < sections.size(); ++i) {
    if (rope_sections[i] < 0 ||
        rope_sections[i] > std::numeric_limits<int>::max()) {
      throw std::invalid_argument("RoPE dimension section is invalid");
    }
    sections[i] = static_cast<int>(rope_sections[i]);
    section_sum += sections[i];
  }
  if (section_sum <= 0 ||
      static_cast<std::uint64_t>(section_sum) > rope_dimension_count) {
    throw std::invalid_argument("RoPE dimension sections mismatch");
  }
  return sections;
}

std::vector<float> build_qwen36_imrope_cache(
    std::int32_t token_position,
    std::uint64_t rope_dimension_count,
    const std::array<int, 4>& sections,
    std::uint64_t rope_context_length,
    float rope_freq_base,
    float rope_freq_scale,
    float rope_ext_factor,
    float rope_attn_factor,
    float rope_beta_fast,
    float rope_beta_slow) {
  const int section_sum =
      sections[0] + sections[1] + sections[2] + sections[3];
  const float theta_scale =
      std::pow(rope_freq_base, -2.0f / static_cast<float>(rope_dimension_count));
  const auto corr_dims = rope_yarn_corr_dims(
      rope_dimension_count,
      rope_context_length,
      rope_freq_base,
      rope_beta_fast,
      rope_beta_slow);

  float theta_t = static_cast<float>(token_position);
  float theta_h = static_cast<float>(token_position);
  float theta_w = static_cast<float>(token_position);
  float theta_e = 0.0f;
  std::vector<float> cache(static_cast<std::size_t>(rope_dimension_count));
  for (std::uint64_t i0 = 0; i0 < rope_dimension_count; i0 += 2) {
    const int sector =
        static_cast<int>((i0 / 2) % static_cast<std::uint64_t>(section_sum));
    float theta = theta_e;
    if (sector % 3 == 1 && sector < 3 * sections[1]) {
      theta = theta_h;
    } else if (sector % 3 == 2 && sector < 3 * sections[2]) {
      theta = theta_w;
    } else if (sector % 3 == 0 && sector < 3 * sections[0]) {
      theta = theta_t;
    }

    rope_yarn(
        theta,
        rope_freq_scale,
        corr_dims,
        static_cast<std::int64_t>(i0),
        rope_ext_factor,
        rope_attn_factor,
        cache[static_cast<std::size_t>(i0)],
        cache[static_cast<std::size_t>(i0 + 1)]);

    theta_t *= theta_scale;
    theta_h *= theta_scale;
    theta_w *= theta_scale;
    theta_e *= theta_scale;
  }
  return cache;
}

std::vector<float> apply_qwen36_imrope(
    const std::vector<float>& input,
    std::int32_t token_position,
    std::uint64_t head_dim,
    std::uint64_t rope_dimension_count,
    const std::array<int, 4>& sections,
    std::uint64_t rope_context_length,
    float rope_freq_base,
    float rope_freq_scale,
    float rope_ext_factor,
    float rope_attn_factor,
    float rope_beta_fast,
    float rope_beta_slow) {
  if (input.empty()) {
    throw std::invalid_argument("RoPE input is empty");
  }
  if (head_dim == 0 || input.size() % head_dim != 0) {
    throw std::invalid_argument("RoPE input/head size mismatch");
  }
  if (rope_dimension_count == 0 || rope_dimension_count % 2 != 0 ||
      rope_dimension_count > head_dim) {
    throw std::invalid_argument("RoPE rotation dimension is invalid");
  }
  if (rope_context_length == 0) {
    throw std::invalid_argument("RoPE context length is invalid");
  }
  if (rope_freq_base <= 0.0f || rope_freq_scale <= 0.0f ||
      rope_beta_fast <= 0.0f || rope_beta_slow <= 0.0f) {
    throw std::invalid_argument("RoPE frequency parameters are invalid");
  }

  const auto cache = build_qwen36_imrope_cache(
      token_position,
      rope_dimension_count,
      sections,
      rope_context_length,
      rope_freq_base,
      rope_freq_scale,
      rope_ext_factor,
      rope_attn_factor,
      rope_beta_fast,
      rope_beta_slow);

  std::vector<float> output = input;
  const std::size_t head_count = input.size() / head_dim;
  const std::size_t rotated_half =
      static_cast<std::size_t>(rope_dimension_count / 2);
  for (std::size_t head = 0; head < head_count; ++head) {
    const std::size_t base = head * static_cast<std::size_t>(head_dim);
    for (std::uint64_t i0 = 0; i0 < rope_dimension_count; i0 += 2) {
      const std::size_t ic = static_cast<std::size_t>(i0 / 2);
      const std::size_t cache_index = static_cast<std::size_t>(i0);
      const float cos_theta = cache[cache_index];
      const float sin_theta = cache[cache_index + 1];
      const float x0 = input[base + ic];
      const float x1 = input[base + ic + rotated_half];
      output[base + ic] = x0 * cos_theta - x1 * sin_theta;
      output[base + ic + rotated_half] = x0 * sin_theta + x1 * cos_theta;
    }
  }
  return output;
}

}  // namespace

float sigmoid_scalar(float value) {
  const double x = value;
  const double result =
      x >= 0.0 ? 1.0 / (1.0 + std::exp(-x))
               : std::exp(x) / (1.0 + std::exp(x));
  return static_cast<float>(result);
}

std::vector<float> softmax(const std::vector<float>& logits) {
  if (logits.empty()) {
    throw std::invalid_argument("softmax input is empty");
  }
  const auto max_it = std::max_element(logits.begin(), logits.end());
  double sum = 0.0;
  std::vector<double> exp_values;
  exp_values.reserve(logits.size());
  for (const auto value : logits) {
    const double exp_value = std::exp(static_cast<double>(value) -
                                      static_cast<double>(*max_it));
    exp_values.push_back(exp_value);
    sum += exp_value;
  }
  if (sum == 0.0 || !std::isfinite(sum)) {
    throw std::runtime_error("softmax denominator is invalid");
  }

  std::vector<float> probabilities;
  probabilities.reserve(logits.size());
  for (const auto value : exp_values) {
    probabilities.push_back(static_cast<float>(value / sum));
  }
  return probabilities;
}

std::vector<std::int32_t> top_k_indices(const std::vector<float>& values,
                                        int k) {
  if (k <= 0) {
    throw std::invalid_argument("top-k count must be positive");
  }
  if (static_cast<std::size_t>(k) > values.size()) {
    throw std::invalid_argument("top-k count exceeds value count");
  }

  std::vector<std::int32_t> indexes(values.size());
  std::iota(indexes.begin(), indexes.end(), 0);
  std::partial_sort(
      indexes.begin(),
      indexes.begin() + k,
      indexes.end(),
      [&values](std::int32_t lhs, std::int32_t rhs) {
        const float left = values[static_cast<std::size_t>(lhs)];
        const float right = values[static_cast<std::size_t>(rhs)];
        if (left == right) {
          return lhs < rhs;
        }
        return left > right;
      });
  indexes.resize(static_cast<std::size_t>(k));
  return indexes;
}

std::vector<float> gather_values(const std::vector<float>& values,
                                 const std::vector<std::int32_t>& indexes) {
  std::vector<float> gathered;
  gathered.reserve(indexes.size());
  for (const auto index : indexes) {
    if (index < 0 || static_cast<std::size_t>(index) >= values.size()) {
      throw std::out_of_range("gather index out of range");
    }
    gathered.push_back(values[static_cast<std::size_t>(index)]);
  }
  return gathered;
}

std::vector<float> normalize_weights(const std::vector<float>& weights,
                                     float min_weight_sum) {
  if (weights.empty()) {
    throw std::invalid_argument("router weights are empty");
  }
  if (!(min_weight_sum > 0.0f)) {
    throw std::invalid_argument("minimum weight sum must be positive");
  }
  float sum = 0.0f;
  for (const auto value : weights) {
    sum += value;
  }
  const float denominator = std::max(sum, min_weight_sum);
  std::vector<float> normalized;
  normalized.reserve(weights.size());
  for (const auto value : weights) {
    normalized.push_back(value / denominator);
  }
  return normalized;
}

RouterTopKSelection select_router_topk(const std::vector<float>& logits,
                                       int expert_used_count,
                                       float min_weight_sum) {
  RouterTopKSelection selection;
  const auto probabilities = softmax(logits);
  selection.expert_ids = top_k_indices(probabilities, expert_used_count);
  selection.weights = gather_values(probabilities, selection.expert_ids);
  selection.normalized_weights =
      normalize_weights(selection.weights, min_weight_sum);
  return selection;
}

std::vector<float> apply_swiglu_pair(const std::vector<float>& gate,
                                     const std::vector<float>& up) {
  if (gate.empty()) {
    throw std::invalid_argument("SwiGLU gate vector is empty");
  }
  if (gate.size() != up.size()) {
    throw std::invalid_argument("SwiGLU gate and up sizes differ");
  }

  std::vector<float> output;
  output.reserve(gate.size());
  for (std::size_t i = 0; i < gate.size(); ++i) {
    const float gate_value = gate[i];
    output.push_back(gate_value * sigmoid_scalar(gate_value) * up[i]);
  }
  return output;
}

std::vector<float> multiply_by_scalar(const std::vector<float>& input,
                                      float scalar) {
  std::vector<float> output;
  output.reserve(input.size());
  for (const auto value : input) {
    output.push_back(value * scalar);
  }
  return output;
}

std::vector<float> apply_expert_weights(const std::vector<float>& expert_down,
                                        const std::vector<float>& weights,
                                        std::uint64_t hidden_size) {
  if (hidden_size == 0) {
    throw std::invalid_argument("hidden size must be nonzero");
  }
  if (weights.empty()) {
    throw std::invalid_argument("expert weights are empty");
  }
  if (expert_down.size() != hidden_size * weights.size()) {
    throw std::invalid_argument("expert output size does not match weights");
  }

  std::vector<float> output;
  output.reserve(expert_down.size());
  for (std::size_t expert = 0; expert < weights.size(); ++expert) {
    const auto base =
        static_cast<std::size_t>(hidden_size) * expert;
    for (std::uint64_t i = 0; i < hidden_size; ++i) {
      output.push_back(
          expert_down[base + static_cast<std::size_t>(i)] * weights[expert]);
    }
  }
  return output;
}

std::vector<float> aggregate_experts(const std::vector<float>& weighted,
                                     std::uint64_t expert_count,
                                     std::uint64_t hidden_size) {
  if (expert_count == 0 || hidden_size == 0) {
    throw std::invalid_argument("expert count and hidden size must be nonzero");
  }
  if (weighted.size() != expert_count * hidden_size) {
    throw std::invalid_argument("weighted expert vector size mismatch");
  }

  std::vector<float> output;
  output.reserve(static_cast<std::size_t>(hidden_size));
  for (std::uint64_t i = 0; i < hidden_size; ++i) {
    float acc = 0.0f;
    for (std::uint64_t expert = 0; expert < expert_count; ++expert) {
      acc += weighted[static_cast<std::size_t>(expert * hidden_size + i)];
    }
    output.push_back(acc);
  }
  return output;
}

std::vector<float> matvec_tensor_impl(
    const std::string& path,
    const GgufModelIndex& index,
    const std::string& tensor_name,
    const std::vector<float>& input,
    const std::vector<Q8KBlock>* q8_input_override) {
  const auto profile_begin = ProfileClock::now();
  const auto* tensor = find_tensor(index, tensor_name);
  if (tensor == nullptr) {
    throw std::invalid_argument("tensor not found");
  }
  if (tensor->dims.empty()) {
    throw std::invalid_argument("tensor has no dimensions");
  }
  const std::uint64_t row_elements = tensor->dims[0];
  if (row_elements == 0) {
    throw std::invalid_argument("tensor row has zero elements");
  }
  if (input.size() != row_elements) {
    throw std::invalid_argument("matvec input size does not match tensor row");
  }
  const auto total_elements = tensor_element_count(tensor->dims);
  const auto row_count = total_elements / row_elements;
  const auto row_nbytes =
      ggml_tensor_nbytes(tensor->type, std::vector<std::uint64_t>{row_elements});
  if (row_nbytes == 0) {
    throw std::invalid_argument("unsupported tensor type for matvec");
  }
  if (row_count > 0 && row_nbytes > tensor->nbytes / row_count) {
    throw std::runtime_error("tensor row byte size exceeds tensor payload");
  }

  std::vector<float> output;
  output.reserve(static_cast<std::size_t>(row_count));
  std::vector<Q8KBlock> owned_q8_input;
  const std::vector<Q8KBlock>* q8_input = q8_input_override;
  static const std::vector<Q8KBlock> empty_q8_input;
  if (tensor->type == 12 || tensor->type == 14) {
    if (q8_input != nullptr) {
      if (input.size() % 256 != 0 || q8_input->size() != input.size() / 256) {
        throw std::runtime_error("reused Q8_K activation block count mismatch");
      }
    } else {
      owned_q8_input = quantize_q8_k_blocks(input);
      q8_input = &owned_q8_input;
    }
  } else {
    q8_input = &empty_q8_input;
  }

  if (should_use_dense_matvec_route(*tensor, row_count)) {
    const bool use_dense_q4_direct_dot =
        should_use_dense_q4_direct_dot_route(*tensor);
    const bool use_q4_direct_minsum_pair =
        q4_direct_minsum_pair_enabled() && tensor->type == 12;
    const bool use_dense_q4_pair_dot =
        should_use_dense_q4_pair_dot_route(*tensor);
    const bool use_dense_q6_direct_dot =
        should_use_dense_q6_direct_dot_route(*tensor);
    const bool use_dense_q6_pair_dot =
        should_use_dense_q6_pair_dot_route(*tensor);
    std::vector<std::uint8_t> payload_storage;
    const auto* resident_payload = cached_tensor_payload(path, *tensor);
    const std::uint8_t* payload_data = nullptr;
    if (resident_payload != nullptr) {
      if (resident_payload->size() != tensor->nbytes) {
        throw std::runtime_error("dense matvec resident payload size mismatch");
      }
      payload_data = resident_payload->data();
    } else {
      payload_storage = read_tensor_bytes_uncached(path, *tensor, 0, tensor->nbytes);
      if (payload_storage.size() != tensor->nbytes) {
        throw std::runtime_error("dense matvec tensor payload size mismatch");
      }
      payload_data = payload_storage.data();
    }
    const bool use_dense_q4_plane =
        use_dense_q4_direct_dot && !use_dense_q4_pair_dot &&
        !use_q4_direct_minsum_pair && should_use_q4_plane_layout_route(*tensor);
    const bool use_dense_q4_plane_pair =
        use_dense_q4_plane && dense_q4_plane_pair_dot_enabled();
    Q4KPlaneRows q4_plane_storage;
    const Q4KPlaneRows* q4_plane_rows = nullptr;
    if (use_dense_q4_plane) {
      q4_plane_rows = cached_q4_plane_rows(
          path,
          *tensor,
          payload_data,
          static_cast<std::size_t>(tensor->nbytes),
          row_count,
          row_nbytes);
      if (q4_plane_rows == nullptr) {
        q4_plane_storage = make_q4_plane_rows(
            payload_data,
            static_cast<std::size_t>(tensor->nbytes),
            row_count,
            row_nbytes);
        q4_plane_rows = &q4_plane_storage;
      }
    }
    const auto* q4_block_meta =
        (use_dense_q4_direct_dot && !use_dense_q4_pair_dot &&
         !use_dense_q4_plane &&
         !use_q4_direct_minsum_pair && resident_payload != nullptr)
            ? cached_q4_block_meta(
                  path, *tensor, payload_data, static_cast<std::size_t>(tensor->nbytes))
            : nullptr;
    const auto blocks_per_row = static_cast<std::size_t>(row_nbytes / 144);
    const bool use_q4_block_meta_dot = q4_block_meta != nullptr;
    output.resize(static_cast<std::size_t>(row_count));
    const auto thread_count = std::min<std::uint64_t>(
        row_count,
        static_cast<std::uint64_t>(dense_matvec_thread_count()));
    auto compute_range = [&](std::uint64_t row_begin, std::uint64_t row_end) {
      std::uint64_t row = row_begin;
      if (use_dense_q4_pair_dot) {
        for (; row + 1 < row_end; row += 2) {
          const auto row_offset = static_cast<std::size_t>(row * row_nbytes);
          float first = 0.0f;
          float second = 0.0f;
          dot_q4_k_q8_k_row_pair_direct(
              payload_data + row_offset,
              payload_data + row_offset + static_cast<std::size_t>(row_nbytes),
              static_cast<std::size_t>(row_nbytes),
              *q8_input,
              first,
              second);
          output[static_cast<std::size_t>(row)] = first;
          output[static_cast<std::size_t>(row + 1)] = second;
        }
      } else if (use_dense_q6_pair_dot) {
        for (; row + 1 < row_end; row += 2) {
          const auto row_offset = static_cast<std::size_t>(row * row_nbytes);
          float first = 0.0f;
          float second = 0.0f;
          dot_q6_k_q8_k_row_pair_direct(
              payload_data + row_offset,
              payload_data + row_offset + static_cast<std::size_t>(row_nbytes),
              static_cast<std::size_t>(row_nbytes),
              *q8_input,
              first,
              second);
          output[static_cast<std::size_t>(row)] = first;
          output[static_cast<std::size_t>(row + 1)] = second;
        }
      } else if (use_dense_q4_plane_pair) {
        for (; row + 1 < row_end; row += 2) {
          float first = 0.0f;
          float second = 0.0f;
          dot_q4_k_q8_k_row_pair_plane(
              *q4_plane_rows, row, row + 1, *q8_input, first, second);
          output[static_cast<std::size_t>(row)] = first;
          output[static_cast<std::size_t>(row + 1)] = second;
        }
      }
      for (; row < row_end; ++row) {
        const auto row_offset = static_cast<std::size_t>(row * row_nbytes);
        output[static_cast<std::size_t>(row)] =
            use_dense_q4_plane
                ? dot_q4_k_q8_k_row_plane(*q4_plane_rows, row, *q8_input)
                : (use_dense_q4_direct_dot
                ? (use_q4_block_meta_dot
                       ? dot_q4_k_q8_k_row_direct_meta(
                             payload_data + row_offset,
                             static_cast<std::size_t>(row_nbytes),
                             *q8_input,
                             q4_block_meta->data() +
                                 static_cast<std::size_t>(row) * blocks_per_row)
                       : (use_q4_direct_minsum_pair
                       ? dot_q4_k_q8_k_row_direct_minpair(
                             payload_data + row_offset,
                             static_cast<std::size_t>(row_nbytes),
                             *q8_input)
                       : dot_q4_k_q8_k_row_direct(
                             payload_data + row_offset,
                             static_cast<std::size_t>(row_nbytes),
                             *q8_input)))
                : (use_dense_q6_direct_dot
                       ? dot_q6_k_q8_k_row_direct(
                             payload_data + row_offset,
                             static_cast<std::size_t>(row_nbytes),
                             *q8_input)
                       : dot_tensor_row_payload(
                             tensor->type,
                             payload_data + row_offset,
                             static_cast<std::size_t>(row_nbytes),
                             input,
                             row_elements,
                             *q8_input)));
      }
    };
    if (thread_count <= 1) {
      compute_range(0, row_count);
    } else {
      parallel_for_rows(row_count, thread_count, [&](std::uint64_t begin,
                                                     std::uint64_t end,
                                                     std::uint64_t) {
        compute_range(begin, end);
      });
    }
    record_matvec_profile(
        use_dense_q4_pair_dot
            ? "matvec_tensor_dense_q4pair"
            : (use_dense_q4_plane
                   ? (use_dense_q4_plane_pair
                          ? "matvec_tensor_dense_q4plane_pair"
                          : "matvec_tensor_dense_q4plane")
                   : (use_dense_q4_direct_dot
                   ? (use_q4_block_meta_dot
                          ? "matvec_tensor_dense_q4directmeta"
                          : (use_q4_direct_minsum_pair
                          ? "matvec_tensor_dense_q4directminpair"
                          : "matvec_tensor_dense_q4direct"))
                   : (use_dense_q6_pair_dot
                          ? "matvec_tensor_dense_q6pair"
                          : (use_dense_q6_direct_dot
                                 ? "matvec_tensor_dense_q6direct"
                                 : "matvec_tensor_dense")))),
        tensor_name,
        input.size(),
        output.size(),
        row_count,
        profile_elapsed_ns(profile_begin, ProfileClock::now()));
    return output;
  }

  const bool use_small_q4_direct_dot =
      should_use_small_q4_direct_dot_route(*tensor, row_count);
  const auto* resident_payload = cached_tensor_payload(path, *tensor);
  if (resident_payload != nullptr) {
    if (resident_payload->size() != tensor->nbytes) {
      throw std::runtime_error("resident tensor payload size mismatch");
    }
    for (std::uint64_t row = 0; row < row_count; ++row) {
      const auto row_offset = static_cast<std::size_t>(row * row_nbytes);
      output.push_back(
          use_small_q4_direct_dot
              ? dot_q4_k_q8_k_row_direct(
                    resident_payload->data() + row_offset,
                    static_cast<std::size_t>(row_nbytes),
                    *q8_input)
              : dot_tensor_row_payload(
                    tensor->type,
                    resident_payload->data() + row_offset,
                    static_cast<std::size_t>(row_nbytes),
                    input,
                    row_elements,
                    *q8_input));
    }
    record_matvec_profile(
        use_small_q4_direct_dot ? "matvec_tensor_small_q4direct"
                                : "matvec_tensor",
        tensor_name,
        input.size(),
        output.size(),
        row_count,
        profile_elapsed_ns(profile_begin, ProfileClock::now()));
    return output;
  }

  std::ifstream model(path, std::ios::binary);
  if (!model) {
    throw std::invalid_argument("GGUF model could not be opened for matvec");
  }
  model.seekg(static_cast<std::streamoff>(tensor->absolute_offset));
  if (!model) {
    throw std::runtime_error("tensor matvec seek failed");
  }

  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(row_nbytes));
  for (std::uint64_t row = 0; row < row_count; ++row) {
    model.read(reinterpret_cast<char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
    if (!model) {
      throw std::runtime_error("tensor matvec row read failed");
    }
    output.push_back(
        use_small_q4_direct_dot
            ? dot_q4_k_q8_k_row_direct(
                  bytes.data(),
                  static_cast<std::size_t>(row_nbytes),
                  *q8_input)
            : dot_tensor_row_payload(
                  tensor->type,
                  bytes,
                  input,
                  row_elements,
                  *q8_input));
  }
  record_matvec_profile(
      use_small_q4_direct_dot ? "matvec_tensor_small_q4direct" : "matvec_tensor",
      tensor_name,
      input.size(),
      output.size(),
      row_count,
      profile_elapsed_ns(profile_begin, ProfileClock::now()));
  return output;
}

std::vector<float> matvec_tensor_with_q8_input(
    const std::string& path,
    const GgufModelIndex& index,
    const std::string& tensor_name,
    const std::vector<float>& input,
    const std::vector<Q8KBlock>& q8_input) {
  return matvec_tensor_impl(path, index, tensor_name, input, &q8_input);
}

std::vector<float> matvec_tensor(const std::string& path,
                                 const GgufModelIndex& index,
                                 const std::string& tensor_name,
                                 const std::vector<float>& input) {
  return matvec_tensor_impl(path, index, tensor_name, input, nullptr);
}

std::vector<MatvecTopKRow> top_k_matvec_tensor(
    const std::string& path,
    const GgufModelIndex& index,
    const std::string& tensor_name,
    const std::vector<float>& input,
    int k,
    int thread_count) {
  const auto profile_begin = ProfileClock::now();
  if (k <= 0) {
    throw std::invalid_argument("top-k count must be positive");
  }
  if (thread_count <= 0) {
    throw std::invalid_argument("top-k matvec thread count must be positive");
  }
  const auto* tensor = find_tensor(index, tensor_name);
  if (tensor == nullptr) {
    throw std::invalid_argument("tensor not found");
  }
  if (tensor->dims.empty()) {
    throw std::invalid_argument("tensor has no dimensions");
  }
  const std::uint64_t row_elements = tensor->dims[0];
  if (row_elements == 0) {
    throw std::invalid_argument("tensor row has zero elements");
  }
  if (input.size() != row_elements) {
    throw std::invalid_argument("matvec input size does not match tensor row");
  }
  const auto total_elements = tensor_element_count(tensor->dims);
  const auto row_count = total_elements / row_elements;
  if (row_count == 0) {
    throw std::invalid_argument("tensor has no rows");
  }
  if (static_cast<std::uint64_t>(k) > row_count) {
    throw std::invalid_argument("top-k count exceeds row count");
  }
  if (row_count > static_cast<std::uint64_t>(
                      std::numeric_limits<std::int32_t>::max())) {
    throw std::runtime_error("top-k matvec row count exceeds int32 token ids");
  }
  const auto row_nbytes =
      ggml_tensor_nbytes(tensor->type, std::vector<std::uint64_t>{row_elements});
  if (row_nbytes == 0) {
    throw std::invalid_argument("unsupported tensor type for matvec");
  }
  if (row_count > 0 && row_nbytes > tensor->nbytes / row_count) {
    throw std::runtime_error("tensor row byte size exceeds tensor payload");
  }

  std::vector<Q8KBlock> q8_input;
  if (tensor->type == 12 || tensor->type == 14) {
    q8_input = quantize_q8_k_blocks(input);
  }

  std::vector<std::uint8_t> owned_payload;
  const std::vector<std::uint8_t>* resident_payload =
      cached_tensor_payload(path, *tensor);
  const std::uint8_t* payload = nullptr;
  if (resident_payload != nullptr) {
    if (resident_payload->size() != tensor->nbytes) {
      throw std::runtime_error("resident tensor payload size mismatch");
    }
    payload = resident_payload->data();
  } else if (thread_count > 1) {
    owned_payload = read_tensor_bytes(path, *tensor, 0, tensor->nbytes);
    payload = owned_payload.data();
  }

  std::vector<MatvecTopKRow> result;
  result.reserve(static_cast<std::size_t>(k));
  if (payload != nullptr) {
    const bool use_q6_pair_dot =
        lm_head_q6_pair_dot_enabled() && tensor->type == 14;
    auto compute_row_value = [&](std::uint64_t row) {
      const auto row_offset = static_cast<std::size_t>(row * row_nbytes);
      return dot_tensor_row_payload(
          tensor->type,
          payload + row_offset,
          static_cast<std::size_t>(row_nbytes),
          input,
          row_elements,
          q8_input);
    };
    auto insert_row_value = [&](std::vector<MatvecTopKRow>& rows,
                                std::uint64_t row,
                                float value) {
      insert_matvec_top_k(
          rows,
          MatvecTopKRow{static_cast<std::int32_t>(row), value},
          k);
    };
    auto compute_range_top_k = [&](std::vector<MatvecTopKRow>& rows,
                                   std::uint64_t row_begin,
                                   std::uint64_t row_end) {
      std::uint64_t row = row_begin;
      if (use_q6_pair_dot) {
        for (; row + 1 < row_end; row += 2) {
          const auto row_offset = static_cast<std::size_t>(row * row_nbytes);
          float first = 0.0f;
          float second = 0.0f;
          dot_q6_k_q8_k_row_pair_direct(
              payload + row_offset,
              payload + row_offset + static_cast<std::size_t>(row_nbytes),
              static_cast<std::size_t>(row_nbytes),
              q8_input,
              first,
              second);
          insert_row_value(rows, row, first);
          insert_row_value(rows, row + 1, second);
        }
      }
      for (; row < row_end; ++row) {
        insert_row_value(rows, row, compute_row_value(row));
      }
    };
    const auto effective_thread_count = static_cast<std::uint64_t>(
        std::min<std::uint64_t>(row_count, static_cast<std::uint64_t>(thread_count)));
    if (effective_thread_count == 1) {
      compute_range_top_k(result, 0, row_count);
    } else {
      std::vector<std::vector<MatvecTopKRow>> local_rows(
          static_cast<std::size_t>(effective_thread_count));
      parallel_for_rows(row_count, effective_thread_count,
                        [&](std::uint64_t row_begin,
                            std::uint64_t row_end,
                            std::uint64_t shard_index) {
          auto& local = local_rows[static_cast<std::size_t>(shard_index)];
          local.reserve(static_cast<std::size_t>(k));
          compute_range_top_k(local, row_begin, row_end);
        });
      for (const auto& local : local_rows) {
        for (const auto& row : local) {
          insert_matvec_top_k(result, row, k);
        }
      }
    }
    record_matvec_profile(
        use_q6_pair_dot ? "top_k_matvec_tensor_q6pair"
                        : "top_k_matvec_tensor",
        tensor_name,
        input.size(),
        result.size(),
        row_count,
        profile_elapsed_ns(profile_begin, ProfileClock::now()));
    return result;
  }

  std::ifstream model(path, std::ios::binary);
  if (!model) {
    throw std::invalid_argument("GGUF model could not be opened for matvec");
  }
  model.seekg(static_cast<std::streamoff>(tensor->absolute_offset));
  if (!model) {
    throw std::runtime_error("tensor matvec seek failed");
  }

  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(row_nbytes));
  for (std::uint64_t row = 0; row < row_count; ++row) {
    model.read(reinterpret_cast<char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
    if (!model) {
      throw std::runtime_error("tensor matvec row read failed");
    }
    const float value =
        dot_tensor_row_payload(tensor->type, bytes, input, row_elements, q8_input);
    insert_matvec_top_k(
        result,
        MatvecTopKRow{static_cast<std::int32_t>(row), value},
        k);
  }
  record_matvec_profile(
      "top_k_matvec_tensor",
      tensor_name,
      input.size(),
      result.size(),
      row_count,
      profile_elapsed_ns(profile_begin, ProfileClock::now()));
  return result;
}

std::vector<float> matvec_expert_tensor(
    const std::string& path,
    const GgufModelIndex& index,
    const std::string& tensor_name,
    const std::vector<float>& input,
    const std::vector<std::int32_t>& expert_ids) {
  const auto profile_begin = ProfileClock::now();
  const auto* tensor = find_tensor(index, tensor_name);
  if (tensor == nullptr) {
    throw std::invalid_argument("expert tensor not found");
  }
  if (tensor->dims.size() != 3) {
    throw std::invalid_argument("expert tensor must be three-dimensional");
  }

  const std::uint64_t row_elements = tensor->dims[0];
  const std::uint64_t rows_per_expert = tensor->dims[1];
  const std::uint64_t expert_count = tensor->dims[2];
  if (row_elements == 0 || rows_per_expert == 0 || expert_count == 0) {
    throw std::invalid_argument("expert tensor has zero-sized dimensions");
  }
  if (input.size() != row_elements) {
    throw std::invalid_argument("expert matvec input size does not match tensor row");
  }

  const auto row_nbytes =
      ggml_tensor_nbytes(tensor->type, std::vector<std::uint64_t>{row_elements});
  if (row_nbytes == 0) {
    throw std::invalid_argument("unsupported tensor type for expert matvec");
  }
  const auto row_count = rows_per_expert * expert_count;
  if (row_count > 0 && row_nbytes > tensor->nbytes / row_count) {
    throw std::runtime_error("expert tensor row byte size exceeds tensor payload");
  }

  std::ifstream model(path, std::ios::binary);
  if (!model) {
    throw std::invalid_argument("GGUF model could not be opened for expert matvec");
  }

  std::vector<Q8KBlock> q8_input;
  if (tensor->type == 12 || tensor->type == 14) {
    q8_input = quantize_q8_k_blocks(input);
  }

  if (expert_slice_matvec_enabled()) {
    std::vector<float> output;
    const auto selected_count = expert_ids.size();
    const auto output_row_count =
        rows_per_expert * static_cast<std::uint64_t>(selected_count);
    output.resize(static_cast<std::size_t>(output_row_count));
    const auto expert_slice_nbytes = rows_per_expert * row_nbytes;
    std::vector<std::vector<std::uint8_t>> expert_slices(
        selected_count,
        std::vector<std::uint8_t>(static_cast<std::size_t>(expert_slice_nbytes)));
    for (std::size_t selected = 0; selected < selected_count; ++selected) {
      const auto expert_id = expert_ids[selected];
      if (expert_id < 0 ||
          static_cast<std::uint64_t>(expert_id) >= expert_count) {
        throw std::out_of_range("expert id out of range");
      }
      const std::uint64_t expert_row_base =
          static_cast<std::uint64_t>(expert_id) * rows_per_expert;
      model.seekg(static_cast<std::streamoff>(
          tensor->absolute_offset + expert_row_base * row_nbytes));
      if (!model) {
        throw std::runtime_error("expert tensor slice seek failed");
      }
      auto& bytes = expert_slices[selected];
      model.read(reinterpret_cast<char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
      if (!model) {
        throw std::runtime_error("expert tensor slice read failed");
      }
    }

    auto compute_range = [&](std::uint64_t begin, std::uint64_t end) {
      for (std::uint64_t flat = begin; flat < end; ++flat) {
        const std::uint64_t selected =
            flat / rows_per_expert;
        const std::uint64_t row = flat % rows_per_expert;
        const auto& bytes = expert_slices[static_cast<std::size_t>(selected)];
        const auto row_offset = static_cast<std::size_t>(row * row_nbytes);
        output[static_cast<std::size_t>(flat)] = dot_tensor_row_payload(
            tensor->type,
            bytes.data() + row_offset,
            static_cast<std::size_t>(row_nbytes),
            input,
            row_elements,
            q8_input);
      }
    };
    const auto thread_count = std::min<std::uint64_t>(
        output_row_count,
        static_cast<std::uint64_t>(expert_slice_matvec_thread_count()));
    if (thread_count <= 1) {
      compute_range(0, output_row_count);
    } else {
      parallel_for_rows(output_row_count, thread_count,
                        [&](std::uint64_t begin,
                            std::uint64_t end,
                            std::uint64_t) {
        compute_range(begin, end);
      });
    }
    record_matvec_profile(
        "matvec_expert_tensor_slice",
        tensor_name,
        input.size(),
        output.size(),
        static_cast<std::uint64_t>(rows_per_expert * expert_ids.size()),
        profile_elapsed_ns(profile_begin, ProfileClock::now()));
    return output;
  }

  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(row_nbytes));
  std::vector<float> output;
  output.reserve(static_cast<std::size_t>(rows_per_expert) * expert_ids.size());
  for (const auto expert_id : expert_ids) {
    if (expert_id < 0 ||
        static_cast<std::uint64_t>(expert_id) >= expert_count) {
      throw std::out_of_range("expert id out of range");
    }
    const std::uint64_t expert_row_base =
        static_cast<std::uint64_t>(expert_id) * rows_per_expert;
    for (std::uint64_t row = 0; row < rows_per_expert; ++row) {
      const std::uint64_t row_index = expert_row_base + row;
      model.seekg(static_cast<std::streamoff>(
          tensor->absolute_offset + row_index * row_nbytes));
      if (!model) {
        throw std::runtime_error("expert tensor matvec seek failed");
      }
      model.read(reinterpret_cast<char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
      if (!model) {
        throw std::runtime_error("expert tensor matvec row read failed");
      }
      output.push_back(
          dot_tensor_row_payload(tensor->type, bytes, input, row_elements, q8_input));
    }
  }
  record_matvec_profile(
      "matvec_expert_tensor",
      tensor_name,
      input.size(),
      output.size(),
      static_cast<std::uint64_t>(rows_per_expert * expert_ids.size()),
      profile_elapsed_ns(profile_begin, ProfileClock::now()));
  return output;
}

std::vector<float> matvec_expert_tensor_per_expert_input(
    const std::string& path,
    const GgufModelIndex& index,
    const std::string& tensor_name,
    const std::vector<float>& input,
    const std::vector<std::int32_t>& expert_ids) {
  const auto profile_begin = ProfileClock::now();
  const auto* tensor = find_tensor(index, tensor_name);
  if (tensor == nullptr) {
    throw std::invalid_argument("expert tensor not found");
  }
  if (tensor->dims.size() != 3) {
    throw std::invalid_argument("expert tensor must be three-dimensional");
  }

  const std::uint64_t row_elements = tensor->dims[0];
  const std::uint64_t rows_per_expert = tensor->dims[1];
  const std::uint64_t expert_count = tensor->dims[2];
  if (row_elements == 0 || rows_per_expert == 0 || expert_count == 0) {
    throw std::invalid_argument("expert tensor has zero-sized dimensions");
  }
  if (expert_ids.empty()) {
    throw std::invalid_argument("expert ids must be nonempty");
  }
  if (input.size() != row_elements * expert_ids.size()) {
    throw std::invalid_argument(
        "per-expert matvec input size does not match selected experts");
  }

  const auto row_nbytes =
      ggml_tensor_nbytes(tensor->type, std::vector<std::uint64_t>{row_elements});
  if (row_nbytes == 0) {
    throw std::invalid_argument("unsupported tensor type for expert matvec");
  }
  const auto row_count = rows_per_expert * expert_count;
  if (row_count > 0 && row_nbytes > tensor->nbytes / row_count) {
    throw std::runtime_error("expert tensor row byte size exceeds tensor payload");
  }

  std::ifstream model(path, std::ios::binary);
  if (!model) {
    throw std::invalid_argument("GGUF model could not be opened for expert matvec");
  }

  if (expert_slice_matvec_enabled()) {
    std::vector<float> output;
    const auto selected_count = expert_ids.size();
    const auto output_row_count =
        rows_per_expert * static_cast<std::uint64_t>(selected_count);
    output.resize(static_cast<std::size_t>(output_row_count));
    const auto expert_slice_nbytes = rows_per_expert * row_nbytes;
    std::vector<std::vector<std::uint8_t>> expert_slices(
        selected_count,
        std::vector<std::uint8_t>(static_cast<std::size_t>(expert_slice_nbytes)));
    std::vector<std::vector<float>> expert_inputs(selected_count);
    std::vector<std::vector<Q8KBlock>> q8_inputs(selected_count);
    for (std::size_t selected = 0; selected < selected_count; ++selected) {
      const auto expert_id = expert_ids[selected];
      if (expert_id < 0 ||
          static_cast<std::uint64_t>(expert_id) >= expert_count) {
        throw std::out_of_range("expert id out of range");
      }
      const auto input_begin =
          input.begin() + static_cast<std::ptrdiff_t>(selected * row_elements);
      const auto input_end =
          input_begin + static_cast<std::ptrdiff_t>(row_elements);
      expert_inputs[selected] = std::vector<float>(input_begin, input_end);
      if (tensor->type == 12 || tensor->type == 14) {
        q8_inputs[selected] = quantize_q8_k_blocks(expert_inputs[selected]);
      }

      const std::uint64_t expert_row_base =
          static_cast<std::uint64_t>(expert_id) * rows_per_expert;
      model.seekg(static_cast<std::streamoff>(
          tensor->absolute_offset + expert_row_base * row_nbytes));
      if (!model) {
        throw std::runtime_error("expert tensor slice seek failed");
      }
      auto& bytes = expert_slices[selected];
      model.read(reinterpret_cast<char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
      if (!model) {
        throw std::runtime_error("expert tensor slice read failed");
      }
    }

    auto compute_range = [&](std::uint64_t begin, std::uint64_t end) {
      for (std::uint64_t flat = begin; flat < end; ++flat) {
        const std::uint64_t selected =
            flat / rows_per_expert;
        const std::uint64_t row = flat % rows_per_expert;
        const auto index = static_cast<std::size_t>(selected);
        const auto& bytes = expert_slices[index];
        const auto row_offset = static_cast<std::size_t>(row * row_nbytes);
        output[static_cast<std::size_t>(flat)] = dot_tensor_row_payload(
            tensor->type,
            bytes.data() + row_offset,
            static_cast<std::size_t>(row_nbytes),
            expert_inputs[index],
            row_elements,
            q8_inputs[index]);
      }
    };
    const auto thread_count = std::min<std::uint64_t>(
        output_row_count,
        static_cast<std::uint64_t>(expert_slice_matvec_thread_count()));
    if (thread_count <= 1) {
      compute_range(0, output_row_count);
    } else {
      parallel_for_rows(output_row_count, thread_count,
                        [&](std::uint64_t begin,
                            std::uint64_t end,
                            std::uint64_t) {
        compute_range(begin, end);
      });
    }
    record_matvec_profile(
        "matvec_expert_tensor_per_expert_input_slice",
        tensor_name,
        input.size(),
        output.size(),
        static_cast<std::uint64_t>(rows_per_expert * expert_ids.size()),
        profile_elapsed_ns(profile_begin, ProfileClock::now()));
    return output;
  }

  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(row_nbytes));
  std::vector<float> output;
  output.reserve(static_cast<std::size_t>(rows_per_expert) * expert_ids.size());
  for (std::size_t selected = 0; selected < expert_ids.size(); ++selected) {
    const auto expert_id = expert_ids[selected];
    if (expert_id < 0 ||
        static_cast<std::uint64_t>(expert_id) >= expert_count) {
      throw std::out_of_range("expert id out of range");
    }
    const auto input_begin =
        input.begin() + static_cast<std::ptrdiff_t>(selected * row_elements);
    const auto input_end =
        input_begin + static_cast<std::ptrdiff_t>(row_elements);
    const std::vector<float> expert_input(input_begin, input_end);
    std::vector<Q8KBlock> q8_input;
    if (tensor->type == 12 || tensor->type == 14) {
      q8_input = quantize_q8_k_blocks(expert_input);
    }

    const std::uint64_t expert_row_base =
        static_cast<std::uint64_t>(expert_id) * rows_per_expert;
    for (std::uint64_t row = 0; row < rows_per_expert; ++row) {
      const std::uint64_t row_index = expert_row_base + row;
      model.seekg(static_cast<std::streamoff>(
          tensor->absolute_offset + row_index * row_nbytes));
      if (!model) {
        throw std::runtime_error("expert tensor matvec seek failed");
      }
      model.read(reinterpret_cast<char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
      if (!model) {
        throw std::runtime_error("expert tensor matvec row read failed");
      }
      output.push_back(dot_tensor_row_payload(
          tensor->type, bytes, expert_input, row_elements, q8_input));
    }
  }
  record_matvec_profile(
      "matvec_expert_tensor_per_expert_input",
      tensor_name,
      input.size(),
      output.size(),
      static_cast<std::uint64_t>(rows_per_expert * expert_ids.size()),
      profile_elapsed_ns(profile_begin, ProfileClock::now()));
  return output;
}

std::vector<float> apply_swiglu_from_gate_up(
    const std::vector<float>& gate_up,
    std::uint64_t intermediate_size,
    std::uint64_t expert_count) {
  if (intermediate_size == 0 || expert_count == 0) {
    throw std::invalid_argument("SwiGLU dimensions must be nonzero");
  }
  const std::uint64_t rows_per_expert = intermediate_size * 2;
  if (gate_up.size() != rows_per_expert * expert_count) {
    throw std::invalid_argument("SwiGLU gate/up size does not match dimensions");
  }

  std::vector<float> output;
  output.reserve(static_cast<std::size_t>(intermediate_size * expert_count));
  for (std::uint64_t expert = 0; expert < expert_count; ++expert) {
    const auto base = static_cast<std::size_t>(expert * rows_per_expert);
    const auto up_base = base + static_cast<std::size_t>(intermediate_size);
    for (std::uint64_t row = 0; row < intermediate_size; ++row) {
      const auto offset = static_cast<std::size_t>(row);
      const double gate = gate_up[base + offset];
      const double up = gate_up[up_base + offset];
      const double sigmoid =
          gate >= 0.0 ? 1.0 / (1.0 + std::exp(-gate))
                      : std::exp(gate) / (1.0 + std::exp(gate));
      output.push_back(static_cast<float>(gate * sigmoid * up));
    }
  }
  return output;
}

Qwen36LinearAttentionDeltaResult run_qwen36_linear_attention_delta_core(
    const std::vector<float>& q,
    const std::vector<float>& k,
    const std::vector<float>& v,
    const std::vector<float>& gate,
    const std::vector<float>& beta,
    const std::vector<float>& recurrent_state,
    const std::vector<float>& z,
    const std::vector<float>& norm_weight,
    float rms_norm_epsilon) {
  if (norm_weight.empty()) {
    throw std::invalid_argument("linear attention norm weight is empty");
  }
  const std::size_t head_dim = norm_weight.size();
  if (q.empty() || k.empty() || v.empty() || recurrent_state.empty()) {
    throw std::invalid_argument("linear attention inputs must be nonempty");
  }
  if (q.size() != k.size() || q.size() % head_dim != 0 ||
      v.size() % head_dim != 0) {
    throw std::invalid_argument("linear attention q/k/v sizes are invalid");
  }
  const std::size_t q_heads = q.size() / head_dim;
  const std::size_t v_heads = v.size() / head_dim;
  if (q_heads == 0 || v_heads == 0) {
    throw std::invalid_argument("linear attention head counts must be nonzero");
  }
  if (v_heads % q_heads != 0) {
    throw std::invalid_argument("linear attention v heads must broadcast q heads");
  }
  if (beta.size() != v_heads) {
    throw std::invalid_argument("linear attention beta size mismatch");
  }
  const bool key_dim_gate = gate.size() == v.size();
  if (!key_dim_gate && gate.size() != v_heads) {
    throw std::invalid_argument("linear attention gate size mismatch");
  }
  if (recurrent_state.size() != head_dim * head_dim * v_heads) {
    throw std::invalid_argument("linear attention state size mismatch");
  }
  if (z.size() != v.size()) {
    throw std::invalid_argument("linear attention z size mismatch");
  }

  Qwen36LinearAttentionDeltaResult result;
  result.attention_output.assign(v.size(), 0.0f);
  result.recurrent_state = recurrent_state;
  result.final_output.assign(v.size(), 0.0f);

  const float attention_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  std::vector<float> delta(head_dim, 0.0f);
  for (std::size_t value_head = 0; value_head < v_heads; ++value_head) {
    const std::size_t query_head = value_head % q_heads;
    const auto* q_head = q.data() + query_head * head_dim;
    const auto* k_head = k.data() + query_head * head_dim;
    const auto* v_head = v.data() + value_head * head_dim;
    auto* state_head =
        result.recurrent_state.data() + value_head * head_dim * head_dim;

    if (key_dim_gate) {
      const auto* gate_head = gate.data() + value_head * head_dim;
      for (std::size_t row = 0; row < head_dim; ++row) {
        auto* state_row = state_head + row * head_dim;
        for (std::size_t col = 0; col < head_dim; ++col) {
          state_row[col] *= std::exp(gate_head[col]);
        }
      }
    } else {
      const float decay = std::exp(gate[value_head]);
      for (std::size_t i = 0; i < head_dim * head_dim; ++i) {
        state_head[i] *= decay;
      }
    }

    for (std::size_t row = 0; row < head_dim; ++row) {
      const auto* state_row = state_head + row * head_dim;
      float sum = 0.0f;
      for (std::size_t col = 0; col < head_dim; ++col) {
        sum += state_row[col] * k_head[col];
      }
      delta[row] = (v_head[row] - sum) * beta[value_head];
    }

    for (std::size_t row = 0; row < head_dim; ++row) {
      auto* state_row = state_head + row * head_dim;
      for (std::size_t col = 0; col < head_dim; ++col) {
        state_row[col] += k_head[col] * delta[row];
      }
    }

    auto* output_head = result.attention_output.data() + value_head * head_dim;
    for (std::size_t row = 0; row < head_dim; ++row) {
      const auto* state_row = state_head + row * head_dim;
      float sum = 0.0f;
      for (std::size_t col = 0; col < head_dim; ++col) {
        sum += state_row[col] * q_head[col];
      }
      output_head[row] = sum * attention_scale;
    }
  }

  for (std::size_t value_head = 0; value_head < v_heads; ++value_head) {
    const auto* output_head =
        result.attention_output.data() + value_head * head_dim;
    const auto* z_head = z.data() + value_head * head_dim;
    float sum_squares = 0.0f;
    for (std::size_t i = 0; i < head_dim; ++i) {
      sum_squares += output_head[i] * output_head[i];
    }
    const float mean_square = sum_squares / static_cast<float>(head_dim);
    const float norm_scale = 1.0f / std::sqrt(mean_square + rms_norm_epsilon);
    auto* final_head = result.final_output.data() + value_head * head_dim;
    for (std::size_t i = 0; i < head_dim; ++i) {
      const float z_value = z_head[i];
      final_head[i] = output_head[i] * norm_scale * norm_weight[i] *
                      (z_value * sigmoid_scalar(z_value));
    }
  }
  return result;
}

Qwen36LinearAttentionPostConvResult run_qwen36_linear_attention_postconv_core(
    const std::vector<float>& conv_output_raw,
    const std::vector<float>& gate,
    const std::vector<float>& beta,
    const std::vector<float>& recurrent_state,
    const std::vector<float>& z,
    const std::vector<float>& norm_weight,
    float norm_epsilon) {
  if (norm_weight.empty()) {
    throw std::invalid_argument("linear attention postconv norm weight is empty");
  }
  if (conv_output_raw.empty()) {
    throw std::invalid_argument("linear attention postconv input is empty");
  }
  const std::size_t head_dim = norm_weight.size();
  const std::size_t v_heads = beta.size();
  if (v_heads == 0) {
    throw std::invalid_argument("linear attention postconv beta is empty");
  }
  const std::size_t v_values = head_dim * v_heads;
  if (conv_output_raw.size() <= v_values ||
      (conv_output_raw.size() - v_values) % 2 != 0) {
    throw std::invalid_argument("linear attention postconv q/k/v split mismatch");
  }
  const std::size_t q_values = (conv_output_raw.size() - v_values) / 2;
  if (q_values == 0 || q_values % head_dim != 0) {
    throw std::invalid_argument("linear attention postconv q/k head mismatch");
  }

  Qwen36LinearAttentionPostConvResult result;
  result.conv_output_silu.reserve(conv_output_raw.size());
  for (const auto value : conv_output_raw) {
    result.conv_output_silu.push_back(value * sigmoid_scalar(value));
  }

  result.q_conv.assign(
      result.conv_output_silu.begin(),
      result.conv_output_silu.begin() + static_cast<std::ptrdiff_t>(q_values));
  result.k_conv.assign(
      result.conv_output_silu.begin() + static_cast<std::ptrdiff_t>(q_values),
      result.conv_output_silu.begin() + static_cast<std::ptrdiff_t>(2 * q_values));
  result.v_conv_predelta.assign(
      result.conv_output_silu.begin() + static_cast<std::ptrdiff_t>(2 * q_values),
      result.conv_output_silu.end());

  auto l2_norm_heads = [head_dim, norm_epsilon](const std::vector<float>& input) {
    if (input.size() % head_dim != 0) {
      throw std::invalid_argument("linear attention L2 input size mismatch");
    }
    std::vector<float> output = input;
    const std::size_t heads = input.size() / head_dim;
    for (std::size_t head = 0; head < heads; ++head) {
      const auto base = head * head_dim;
      double sum = 0.0;
      for (std::size_t i = 0; i < head_dim; ++i) {
        const float value = input[base + i];
        sum += static_cast<double>(value) * static_cast<double>(value);
      }
      const float scale =
          1.0f / std::max(std::sqrt(static_cast<float>(sum)), norm_epsilon);
      for (std::size_t i = 0; i < head_dim; ++i) {
        output[base + i] = input[base + i] * scale;
      }
    }
    return output;
  };
  result.q_conv_predelta = l2_norm_heads(result.q_conv);
  result.k_conv_predelta = l2_norm_heads(result.k_conv);

  const auto delta = run_qwen36_linear_attention_delta_core(
      result.q_conv_predelta,
      result.k_conv_predelta,
      result.v_conv_predelta,
      gate,
      beta,
      recurrent_state,
      z,
      norm_weight,
      norm_epsilon);
  result.attention_output = delta.attention_output;
  result.recurrent_state = delta.recurrent_state;
  result.final_output = delta.final_output;
  return result;
}

Qwen36LinearAttentionPreConvResult run_qwen36_linear_attention_preconv_core(
    const std::string& path,
    const GgufModelIndex& index,
    int layer_index,
    const std::vector<float>& attention_norm) {
  if (attention_norm.empty()) {
    throw std::invalid_argument("linear attention preconv input is empty");
  }

  Qwen36LinearAttentionPreConvResult result;
  std::vector<Q8KBlock> q8_attention_norm;
  const bool reuse_q8_input = matvec_q8_input_reuse_enabled();
  if (reuse_q8_input) {
    q8_attention_norm = quantize_q8_k_blocks(attention_norm);
  }
  auto matvec_attention_norm = [&](const std::string& suffix) {
    const auto tensor_name = layer_tensor_name(layer_index, suffix);
    return reuse_q8_input
               ? matvec_tensor_with_q8_input(
                     path, index, tensor_name, attention_norm, q8_attention_norm)
               : matvec_tensor(path, index, tensor_name, attention_norm);
  };

  result.qkv_mixed = matvec_attention_norm("attn_qkv.weight");
  result.alpha = matvec_attention_norm("ssm_alpha.weight");
  const auto ssm_dt = decode_tensor_row(
      path, index, layer_tensor_name(layer_index, "ssm_dt.bias"), 0);
  result.alpha_softplus = softplus_vector(add_vectors(result.alpha, ssm_dt));
  const auto ssm_a =
      decode_tensor_row(path, index, layer_tensor_name(layer_index, "ssm_a"), 0);
  result.gate = multiply_vectors_checked(
      result.alpha_softplus, ssm_a, "linear attention gate sizes differ");

  result.beta = matvec_attention_norm("ssm_beta.weight");
  result.beta_sigmoid = sigmoid_vector(result.beta);
  result.z = matvec_attention_norm("attn_gate.weight");
  return result;
}

Qwen36LinearAttentionConvResult run_qwen36_linear_attention_conv_core(
    const std::string& path,
    const GgufModelIndex& index,
    int layer_index,
    const std::vector<float>& qkv_mixed,
    const std::vector<float>& conv_state) {
  if (qkv_mixed.empty()) {
    throw std::invalid_argument("linear attention conv qkv input is empty");
  }

  const auto tensor_name = layer_tensor_name(layer_index, "ssm_conv1d.weight");
  const auto* conv_tensor = find_tensor(index, tensor_name);
  if (conv_tensor == nullptr) {
    throw std::invalid_argument("linear attention conv tensor not found");
  }
  if (conv_tensor->type != 0 || conv_tensor->dims.size() != 2) {
    throw std::invalid_argument("linear attention conv tensor shape is invalid");
  }
  const std::size_t kernel_size =
      static_cast<std::size_t>(conv_tensor->dims[0]);
  const std::size_t channel_count =
      static_cast<std::size_t>(conv_tensor->dims[1]);
  if (kernel_size < 2 || channel_count == 0) {
    throw std::invalid_argument("linear attention conv tensor dims are invalid");
  }
  if (qkv_mixed.size() != channel_count) {
    throw std::invalid_argument("linear attention conv qkv size mismatch");
  }
  const std::size_t history_size = (kernel_size - 1) * channel_count;
  if (conv_state.size() != history_size) {
    throw std::invalid_argument("linear attention conv state size mismatch");
  }

  const auto weight_bytes = read_tensor_prefix(path, *conv_tensor, conv_tensor->nbytes);
  const auto weights = decode_f32_values(
      weight_bytes, static_cast<std::size_t>(kernel_size * channel_count));
  if (weights.size() != kernel_size * channel_count) {
    throw std::runtime_error("linear attention conv weight decode mismatch");
  }

  Qwen36LinearAttentionConvResult result;
  result.conv_output_raw.assign(channel_count, 0.0f);
  result.conv_state.assign(history_size, 0.0f);
  for (std::size_t channel = 0; channel < channel_count; ++channel) {
    const std::size_t state_base = channel * (kernel_size - 1);
    const std::size_t weight_base = channel * kernel_size;
    float sum = 0.0f;
    for (std::size_t k = 0; k + 1 < kernel_size; ++k) {
      sum += conv_state[state_base + k] * weights[weight_base + k];
    }
    sum += qkv_mixed[channel] * weights[weight_base + kernel_size - 1];
    result.conv_output_raw[channel] = sum;

    for (std::size_t k = 0; k + 2 < kernel_size; ++k) {
      result.conv_state[state_base + k] = conv_state[state_base + k + 1];
    }
    result.conv_state[state_base + kernel_size - 2] = qkv_mixed[channel];
  }
  return result;
}

SelectedExpertFfnRouteResult run_selected_expert_ffn_route(
    const std::string& path,
    const GgufTensorInfo& gate_up_tensor,
    const GgufTensorInfo& down_tensor,
    const std::vector<float>& ffn_norm,
    const RouterTopKSelection& router) {
  if (gate_up_tensor.dims.size() != 3 || down_tensor.dims.size() != 3) {
    throw std::invalid_argument("selected-expert FFN route tensor ranks are invalid");
  }
  if (router.expert_ids.empty() ||
      router.expert_ids.size() != router.normalized_weights.size()) {
    throw std::invalid_argument("selected-expert FFN route router size mismatch");
  }

  const auto hidden_size = static_cast<std::uint64_t>(ffn_norm.size());
  const std::uint64_t gate_row_elements = gate_up_tensor.dims[0];
  const std::uint64_t gate_rows_per_expert = gate_up_tensor.dims[1];
  const std::uint64_t gate_expert_count = gate_up_tensor.dims[2];
  const std::uint64_t selected_intermediate = gate_rows_per_expert / 2;
  const std::uint64_t down_row_elements = down_tensor.dims[0];
  const std::uint64_t down_rows_per_expert = down_tensor.dims[1];
  const std::uint64_t down_expert_count = down_tensor.dims[2];
  if (hidden_size == 0 || gate_row_elements != hidden_size ||
      gate_rows_per_expert == 0 || gate_rows_per_expert % 2 != 0 ||
      selected_intermediate == 0 || down_row_elements != selected_intermediate ||
      down_rows_per_expert != hidden_size ||
      gate_expert_count != down_expert_count) {
    throw std::invalid_argument("selected-expert FFN route tensor dims mismatch");
  }

  const auto gate_row_nbytes = ggml_tensor_nbytes(
      gate_up_tensor.type, std::vector<std::uint64_t>{gate_row_elements});
  const auto down_row_nbytes = ggml_tensor_nbytes(
      down_tensor.type, std::vector<std::uint64_t>{down_row_elements});
  if (gate_row_nbytes == 0 || down_row_nbytes == 0) {
    throw std::invalid_argument("unsupported selected-expert FFN tensor type");
  }
  if (gate_row_nbytes > gate_up_tensor.nbytes /
                            (gate_rows_per_expert * gate_expert_count) ||
      down_row_nbytes > down_tensor.nbytes /
                           (down_rows_per_expert * down_expert_count)) {
    throw std::runtime_error("selected-expert FFN row byte size exceeds tensor payload");
  }

  std::ifstream model(path, std::ios::binary);
  if (!model) {
    throw std::invalid_argument("GGUF model could not be opened for selected-expert FFN");
  }

  const auto selected_count = router.expert_ids.size();
  const auto gate_slice_nbytes = gate_rows_per_expert * gate_row_nbytes;
  const bool minimal_outputs = selected_expert_minimal_outputs_enabled();

  const auto read_expert_slice =
      [&](const GgufTensorInfo& tensor,
          std::uint64_t rows_per_expert,
          std::uint64_t row_nbytes,
          std::int32_t expert_id,
          bool use_slice_cache,
          std::vector<std::uint8_t>* owned_bytes)
          -> const std::vector<std::uint8_t>* {
        if (expert_id < 0 ||
            static_cast<std::uint64_t>(expert_id) >= gate_expert_count) {
          throw std::out_of_range("expert id out of range");
        }
        if (use_slice_cache) {
          const auto* cached = cached_expert_slice(
              path, tensor, rows_per_expert, row_nbytes, expert_id);
          if (cached != nullptr) {
            return cached;
          }
        }
        const std::uint64_t expert_row_base =
            static_cast<std::uint64_t>(expert_id) * rows_per_expert;
        owned_bytes->resize(static_cast<std::size_t>(rows_per_expert * row_nbytes));
        model.seekg(static_cast<std::streamoff>(
            tensor.absolute_offset + expert_row_base * row_nbytes));
        if (!model) {
          throw std::runtime_error("selected-expert FFN slice seek failed");
        }
        model.read(reinterpret_cast<char*>(owned_bytes->data()),
                   static_cast<std::streamsize>(owned_bytes->size()));
        if (!model) {
          throw std::runtime_error("selected-expert FFN slice read failed");
        }
        return owned_bytes;
      };

  SelectedExpertFfnRouteResult result;
  if (!minimal_outputs) {
    result.selected_gate_up.resize(
        static_cast<std::size_t>(selected_count * gate_rows_per_expert));
  }
  result.selected_swiglu.resize(
      static_cast<std::size_t>(selected_count * selected_intermediate));

  const auto gate_read_profile_begin = ProfileClock::now();
  std::vector<std::vector<std::uint8_t>> gate_up_slice_storage(selected_count);
  std::vector<const std::vector<std::uint8_t>*> gate_up_slices(selected_count);
  for (std::size_t selected = 0; selected < selected_count; ++selected) {
    gate_up_slices[selected] = read_expert_slice(
        gate_up_tensor,
        gate_rows_per_expert,
        gate_row_nbytes,
        router.expert_ids[selected],
        selected_expert_slice_cache_enabled(),
        &gate_up_slice_storage[selected]);
  }
  const auto gate_read_elapsed_ns =
      profile_elapsed_ns(gate_read_profile_begin, ProfileClock::now());
  record_matvec_profile(
      "selected_expert_ffn_gate_up_read",
      gate_up_tensor.name,
      0,
      static_cast<std::uint64_t>(selected_count) * gate_slice_nbytes,
      static_cast<std::uint64_t>(selected_count) * gate_rows_per_expert,
      gate_read_elapsed_ns);

  std::vector<Q8KBlock> q8_ffn_norm;
  if (gate_up_tensor.type == 12 || gate_up_tensor.type == 14) {
    q8_ffn_norm = quantize_q8_k_blocks(ffn_norm);
  }

  const auto gate_compute_profile_begin = ProfileClock::now();
  const auto gate_task_count =
      static_cast<std::uint64_t>(selected_count) * selected_intermediate;
  const auto gate_thread_count = std::min<std::uint64_t>(
      gate_task_count,
      static_cast<std::uint64_t>(selected_expert_ffn_thread_count()));
  const bool use_pair_sum_q4_gate_dot =
      selected_gate_q4_pair_sum_dot_enabled() && gate_up_tensor.type == 12;
  const bool use_pair_q4_gate_dot =
      selected_gate_q4_pair_dot_enabled() && gate_up_tensor.type == 12;
  const bool use_direct_q4_gate_dot =
      selected_gate_q4_direct_dot_enabled() && gate_up_tensor.type == 12;
  const bool use_q4_direct_minsum_pair =
      q4_direct_minsum_pair_enabled();
  const bool use_gate_q4_plane =
      should_use_q4_plane_layout_route(gate_up_tensor) &&
      (use_pair_sum_q4_gate_dot || use_pair_q4_gate_dot ||
       use_direct_q4_gate_dot) &&
      !use_q4_direct_minsum_pair;
  const bool use_gate_q4_plane_pair_dot =
      use_gate_q4_plane && selected_gate_q4_plane_pair_dot_enabled();
  std::vector<Q4KPlaneRows> gate_up_plane_storage(selected_count);
  std::vector<const Q4KPlaneRows*> gate_up_planes(selected_count, nullptr);
  if (use_gate_q4_plane) {
    for (std::size_t selected = 0; selected < selected_count; ++selected) {
      const auto& bytes = *gate_up_slices[selected];
      const auto* cached_plane = cached_q4_plane_expert_slice(
          path,
          gate_up_tensor,
          gate_rows_per_expert,
          gate_row_nbytes,
          router.expert_ids[selected],
          bytes.data(),
          bytes.size());
      if (cached_plane != nullptr) {
        gate_up_planes[selected] = cached_plane;
      } else {
        gate_up_plane_storage[selected] = make_q4_plane_rows(
            bytes.data(),
            bytes.size(),
            gate_rows_per_expert,
            gate_row_nbytes);
        gate_up_planes[selected] = &gate_up_plane_storage[selected];
      }
    }
  }
  auto dot_gate_up_row = [&](const std::uint8_t* bytes) {
    if (use_direct_q4_gate_dot) {
      return use_q4_direct_minsum_pair
                 ? dot_q4_k_q8_k_row_direct_minpair(
                       bytes,
                       static_cast<std::size_t>(gate_row_nbytes),
                       q8_ffn_norm)
                 : dot_q4_k_q8_k_row_direct(
                       bytes,
                       static_cast<std::size_t>(gate_row_nbytes),
                       q8_ffn_norm);
    }
    return dot_tensor_row_payload(
        gate_up_tensor.type,
        bytes,
        static_cast<std::size_t>(gate_row_nbytes),
        ffn_norm,
        gate_row_elements,
        q8_ffn_norm);
  };
  auto compute_gate_range = [&](std::uint64_t begin, std::uint64_t end) {
    for (std::uint64_t flat = begin; flat < end; ++flat) {
      const std::uint64_t selected = flat / selected_intermediate;
      const std::uint64_t row = flat % selected_intermediate;
      const auto selected_index = static_cast<std::size_t>(selected);
      const auto& bytes = *gate_up_slices[selected_index];
      const auto gate_row_offset = static_cast<std::size_t>(row * gate_row_nbytes);
      const auto up_row_offset = static_cast<std::size_t>(
          (selected_intermediate + row) * gate_row_nbytes);
      float gate = 0.0f;
      float up = 0.0f;
      if (use_gate_q4_plane_pair_dot) {
        const auto* plane = gate_up_planes[selected_index];
        dot_q4_k_q8_k_row_pair_plane(
            *plane, row, selected_intermediate + row, q8_ffn_norm, gate, up);
      } else if (use_gate_q4_plane) {
        const auto* plane = gate_up_planes[selected_index];
        gate = dot_q4_k_q8_k_row_plane(*plane, row, q8_ffn_norm);
        up = dot_q4_k_q8_k_row_plane(
            *plane, selected_intermediate + row, q8_ffn_norm);
      } else if (use_pair_sum_q4_gate_dot) {
        dot_q4_k_q8_k_row_pair_sum_direct(
            bytes.data() + gate_row_offset,
            bytes.data() + up_row_offset,
            static_cast<std::size_t>(gate_row_nbytes),
            q8_ffn_norm,
            gate,
            up);
      } else if (use_pair_q4_gate_dot) {
        dot_q4_k_q8_k_row_pair_direct(
            bytes.data() + gate_row_offset,
            bytes.data() + up_row_offset,
            static_cast<std::size_t>(gate_row_nbytes),
            q8_ffn_norm,
            gate,
            up);
      } else {
        gate = dot_gate_up_row(bytes.data() + gate_row_offset);
        up = dot_gate_up_row(bytes.data() + up_row_offset);
      }
      const auto gate_up_base =
          static_cast<std::size_t>(selected * gate_rows_per_expert);
      const auto swiglu_base =
          static_cast<std::size_t>(selected * selected_intermediate);
      const auto row_index = static_cast<std::size_t>(row);
      if (!minimal_outputs) {
        result.selected_gate_up[gate_up_base + row_index] = gate;
        result.selected_gate_up[
            gate_up_base + static_cast<std::size_t>(selected_intermediate) +
            row_index] = up;
      }

      const double gate_double = gate;
      const double up_double = up;
      const double sigmoid =
          gate_double >= 0.0
              ? 1.0 / (1.0 + std::exp(-gate_double))
              : std::exp(gate_double) / (1.0 + std::exp(gate_double));
      result.selected_swiglu[swiglu_base + row_index] =
          static_cast<float>(gate_double * sigmoid * up_double);
    }
  };
  if (gate_thread_count <= 1) {
    compute_gate_range(0, gate_task_count);
  } else {
    parallel_for_rows(gate_task_count, gate_thread_count,
                      [&](std::uint64_t begin,
                          std::uint64_t end,
                          std::uint64_t) {
      compute_gate_range(begin, end);
    });
  }
  record_matvec_profile(
      use_gate_q4_plane_pair_dot
          ? "selected_expert_ffn_gate_swiglu_q4plane_pair"
          : (use_gate_q4_plane
          ? "selected_expert_ffn_gate_swiglu_q4plane"
          : (use_pair_sum_q4_gate_dot
          ? "selected_expert_ffn_gate_swiglu_q4pairsum"
          : (use_pair_q4_gate_dot
                 ? "selected_expert_ffn_gate_swiglu_q4pair"
                 : (use_direct_q4_gate_dot
                        ? "selected_expert_ffn_gate_swiglu_q4direct"
                        : "selected_expert_ffn_gate_swiglu")))),
      gate_up_tensor.name,
      ffn_norm.size(),
      result.selected_swiglu.size(),
      static_cast<std::uint64_t>(selected_count) * gate_rows_per_expert,
      profile_elapsed_ns(gate_compute_profile_begin, ProfileClock::now()));

  const auto down_profile_begin = ProfileClock::now();
  std::vector<std::vector<std::uint8_t>> down_slice_storage(selected_count);
  std::vector<const std::vector<std::uint8_t>*> down_slices(selected_count);
  for (std::size_t selected = 0; selected < selected_count; ++selected) {
    down_slices[selected] = read_expert_slice(
        down_tensor,
        down_rows_per_expert,
        down_row_nbytes,
        router.expert_ids[selected],
        selected_expert_down_slice_cache_enabled(),
        &down_slice_storage[selected]);
  }

  std::vector<std::vector<float>> expert_inputs(selected_count);
  std::vector<std::vector<Q8KBlock>> q8_expert_inputs(selected_count);
  for (std::size_t selected = 0; selected < selected_count; ++selected) {
    const auto input_begin =
        result.selected_swiglu.begin() +
        static_cast<std::ptrdiff_t>(selected * selected_intermediate);
    const auto input_end =
        input_begin + static_cast<std::ptrdiff_t>(selected_intermediate);
    expert_inputs[selected] = std::vector<float>(input_begin, input_end);
    if (down_tensor.type == 12 || down_tensor.type == 14) {
      q8_expert_inputs[selected] = quantize_q8_k_blocks(expert_inputs[selected]);
    }
  }

  if (!minimal_outputs) {
    result.selected_down.resize(
        static_cast<std::size_t>(selected_count * hidden_size));
    result.weighted_selected_down.resize(
        static_cast<std::size_t>(selected_count * hidden_size));
  }
  result.moe_out.assign(static_cast<std::size_t>(hidden_size), 0.0f);

  const auto down_thread_count = std::min<std::uint64_t>(
      hidden_size,
      static_cast<std::uint64_t>(selected_expert_ffn_thread_count()));
  const bool use_direct_q4_down_dot =
      dense_q4_direct_dot_enabled() && down_tensor.type == 12;
  const bool use_direct_q6_down_dot =
      dense_q6_direct_dot_enabled() && down_tensor.type == 14;
  const bool use_q4_down_plane =
      use_direct_q4_down_dot && !use_q4_direct_minsum_pair &&
      should_use_q4_plane_layout_route(down_tensor);
  std::vector<Q4KPlaneRows> down_plane_storage(selected_count);
  std::vector<const Q4KPlaneRows*> down_planes(selected_count, nullptr);
  if (use_q4_down_plane) {
    for (std::size_t selected = 0; selected < selected_count; ++selected) {
      const auto& bytes = *down_slices[selected];
      const auto* cached_plane = cached_q4_plane_expert_slice(
          path,
          down_tensor,
          down_rows_per_expert,
          down_row_nbytes,
          router.expert_ids[selected],
          bytes.data(),
          bytes.size());
      if (cached_plane != nullptr) {
        down_planes[selected] = cached_plane;
      } else {
        down_plane_storage[selected] = make_q4_plane_rows(
            bytes.data(),
            bytes.size(),
            down_rows_per_expert,
            down_row_nbytes);
        down_planes[selected] = &down_plane_storage[selected];
      }
    }
  }
  const bool use_q4_down_pair_dot =
      selected_expert_down_q4_pair_dot_enabled() && use_direct_q4_down_dot &&
      !use_q4_direct_minsum_pair && !use_q4_down_plane;
  const bool use_q6_down_pair_dot =
      selected_expert_down_q6_pair_dot_enabled() && use_direct_q6_down_dot;
  auto dot_down_row = [&](std::size_t selected, std::uint64_t row) {
    const auto& bytes = *down_slices[selected];
    const auto row_offset = static_cast<std::size_t>(row * down_row_nbytes);
    return use_q4_down_plane
               ? dot_q4_k_q8_k_row_plane(
                     *down_planes[selected], row, q8_expert_inputs[selected])
               : (use_direct_q4_down_dot
               ? (use_q4_direct_minsum_pair
                      ? dot_q4_k_q8_k_row_direct_minpair(
                            bytes.data() + row_offset,
                            static_cast<std::size_t>(down_row_nbytes),
                            q8_expert_inputs[selected])
                      : dot_q4_k_q8_k_row_direct(
                            bytes.data() + row_offset,
                            static_cast<std::size_t>(down_row_nbytes),
                            q8_expert_inputs[selected]))
               : (use_direct_q6_down_dot
                      ? dot_q6_k_q8_k_row_direct(
                            bytes.data() + row_offset,
                            static_cast<std::size_t>(down_row_nbytes),
                            q8_expert_inputs[selected])
                      : dot_tensor_row_payload(
                            down_tensor.type,
                            bytes.data() + row_offset,
                            static_cast<std::size_t>(down_row_nbytes),
                            expert_inputs[selected],
                            down_row_elements,
                            q8_expert_inputs[selected])));
  };
  const bool use_expert_major_down =
      selected_expert_down_expert_major_enabled();
  if (use_expert_major_down) {
    std::vector<float> weighted_storage;
    float* weighted_values = nullptr;
    if (minimal_outputs) {
      weighted_storage.resize(
          static_cast<std::size_t>(selected_count * hidden_size));
      weighted_values = weighted_storage.data();
    } else {
      weighted_values = result.weighted_selected_down.data();
    }
    const auto down_task_count =
        static_cast<std::uint64_t>(selected_count) * hidden_size;
    const auto down_expert_major_thread_count = std::min<std::uint64_t>(
        down_task_count,
        static_cast<std::uint64_t>(selected_expert_ffn_thread_count()));
    auto compute_expert_major_range =
        [&](std::uint64_t begin, std::uint64_t end) {
          for (std::uint64_t flat = begin; flat < end; ++flat) {
            const std::size_t selected =
                static_cast<std::size_t>(flat / hidden_size);
            const std::uint64_t row = flat % hidden_size;
            const float down_value = dot_down_row(selected, row);
            const auto output_index =
                static_cast<std::size_t>(selected * hidden_size + row);
            const float weighted =
                down_value * router.normalized_weights[selected];
            if (!minimal_outputs) {
              result.selected_down[output_index] = down_value;
            }
            weighted_values[output_index] = weighted;
          }
        };
    if (down_expert_major_thread_count <= 1) {
      compute_expert_major_range(0, down_task_count);
    } else {
      parallel_for_rows(down_task_count, down_expert_major_thread_count,
                        [&](std::uint64_t begin,
                            std::uint64_t end,
                            std::uint64_t) {
        compute_expert_major_range(begin, end);
      });
    }
    auto aggregate_down_range = [&](std::uint64_t begin, std::uint64_t end) {
      for (std::uint64_t row = begin; row < end; ++row) {
        float acc = 0.0f;
        for (std::size_t selected = 0; selected < selected_count; ++selected) {
          acc += weighted_values[
              static_cast<std::size_t>(selected * hidden_size + row)];
        }
        result.moe_out[static_cast<std::size_t>(row)] = acc;
      }
    };
    if (down_thread_count <= 1) {
      aggregate_down_range(0, hidden_size);
    } else {
      parallel_for_rows(hidden_size, down_thread_count,
                        [&](std::uint64_t begin,
                            std::uint64_t end,
                            std::uint64_t) {
        aggregate_down_range(begin, end);
      });
    }
  } else {
    auto compute_down_range = [&](std::uint64_t begin, std::uint64_t end) {
      std::uint64_t row = begin;
      if (use_q4_down_pair_dot || use_q6_down_pair_dot) {
        for (; row + 1 < end; row += 2) {
          float first_acc = 0.0f;
          float second_acc = 0.0f;
          for (std::size_t selected = 0; selected < selected_count; ++selected) {
            const auto& bytes = *down_slices[selected];
            const auto first_row_offset =
                static_cast<std::size_t>(row * down_row_nbytes);
            const auto second_row_offset =
                static_cast<std::size_t>((row + 1) * down_row_nbytes);
            float first_down_value = 0.0f;
            float second_down_value = 0.0f;
            if (use_q4_down_pair_dot) {
              dot_q4_k_q8_k_row_pair_direct(
                  bytes.data() + first_row_offset,
                  bytes.data() + second_row_offset,
                  static_cast<std::size_t>(down_row_nbytes),
                  q8_expert_inputs[selected],
                  first_down_value,
                  second_down_value);
            } else {
              dot_q6_k_q8_k_row_pair_direct(
                  bytes.data() + first_row_offset,
                  bytes.data() + second_row_offset,
                  static_cast<std::size_t>(down_row_nbytes),
                  q8_expert_inputs[selected],
                  first_down_value,
                  second_down_value);
            }

            const auto first_output_index = static_cast<std::size_t>(
                selected * hidden_size + row);
            const auto second_output_index = static_cast<std::size_t>(
                selected * hidden_size + row + 1);
            const float weight = router.normalized_weights[selected];
            const float first_weighted = first_down_value * weight;
            const float second_weighted = second_down_value * weight;
            if (!minimal_outputs) {
              result.selected_down[first_output_index] = first_down_value;
              result.weighted_selected_down[first_output_index] = first_weighted;
              result.selected_down[second_output_index] = second_down_value;
              result.weighted_selected_down[second_output_index] =
                  second_weighted;
            }
            first_acc += first_weighted;
            second_acc += second_weighted;
          }
          result.moe_out[static_cast<std::size_t>(row)] = first_acc;
          result.moe_out[static_cast<std::size_t>(row + 1)] = second_acc;
        }
      }
      for (; row < end; ++row) {
        float acc = 0.0f;
        for (std::size_t selected = 0; selected < selected_count; ++selected) {
          const float down_value = dot_down_row(selected, row);
          const auto output_index = static_cast<std::size_t>(
              selected * hidden_size + row);
          const float weighted =
              down_value * router.normalized_weights[selected];
          if (!minimal_outputs) {
            result.selected_down[output_index] = down_value;
            result.weighted_selected_down[output_index] = weighted;
          }
          acc += weighted;
        }
        result.moe_out[static_cast<std::size_t>(row)] = acc;
      }
    };
    if (down_thread_count <= 1) {
      compute_down_range(0, hidden_size);
    } else {
      parallel_for_rows(hidden_size, down_thread_count,
                        [&](std::uint64_t begin,
                            std::uint64_t end,
                            std::uint64_t) {
        compute_down_range(begin, end);
      });
    }
  }
  record_matvec_profile(
      use_q4_down_plane
          ? (use_expert_major_down
                 ? "selected_expert_ffn_down_q4plane_expert_major"
                 : "selected_expert_ffn_down_q4plane")
          : (use_expert_major_down
          ? "selected_expert_ffn_down_expert_major"
          : (use_q4_down_pair_dot
                 ? "selected_expert_ffn_down_q4pair"
                 : (use_q6_down_pair_dot
                        ? "selected_expert_ffn_down_q6pair"
                        : "selected_expert_ffn_down_aggregate"))),
      down_tensor.name,
      result.selected_swiglu.size(),
      result.moe_out.size(),
      static_cast<std::uint64_t>(selected_count) * hidden_size,
      profile_elapsed_ns(down_profile_begin, ProfileClock::now()));

  return result;
}

SharedExpertGateUpFusedRouteResult run_shared_expert_gate_up_fused_route(
    const std::string& path,
    const GgufTensorInfo& gate_tensor,
    const GgufTensorInfo& up_tensor,
    const std::vector<float>& ffn_norm) {
  if (gate_tensor.dims.size() != 2 || up_tensor.dims.size() != 2) {
    throw std::invalid_argument("shared expert gate/up fused tensor ranks are invalid");
  }
  if (gate_tensor.type != 12 || up_tensor.type != 12) {
    throw std::invalid_argument("shared expert gate/up fused route requires Q4_K tensors");
  }
  if (gate_tensor.dims != up_tensor.dims) {
    throw std::invalid_argument("shared expert gate/up fused tensor dims differ");
  }
  const std::uint64_t row_elements = gate_tensor.dims[0];
  const std::uint64_t row_count = gate_tensor.dims[1];
  if (row_elements == 0 || row_count == 0 ||
      ffn_norm.size() != row_elements) {
    throw std::invalid_argument("shared expert gate/up fused dims mismatch");
  }
  const auto row_nbytes =
      ggml_tensor_nbytes(gate_tensor.type, std::vector<std::uint64_t>{row_elements});
  if (row_nbytes == 0) {
    throw std::invalid_argument("unsupported shared expert gate/up fused tensor type");
  }
  if (row_nbytes > gate_tensor.nbytes / row_count ||
      row_nbytes > up_tensor.nbytes / row_count) {
    throw std::runtime_error("shared expert gate/up fused row byte size exceeds tensor payload");
  }

  const auto profile_begin = ProfileClock::now();
  std::vector<Q8KBlock> q8_ffn_norm = quantize_q8_k_blocks(ffn_norm);

  std::vector<std::uint8_t> gate_payload_storage;
  std::vector<std::uint8_t> up_payload_storage;
  const auto* gate_payload = cached_tensor_payload(path, gate_tensor);
  const auto* up_payload = cached_tensor_payload(path, up_tensor);
  const std::uint8_t* gate_payload_data = nullptr;
  const std::uint8_t* up_payload_data = nullptr;
  if (gate_payload != nullptr) {
    if (gate_payload->size() != gate_tensor.nbytes) {
      throw std::runtime_error("shared expert gate resident payload size mismatch");
    }
    gate_payload_data = gate_payload->data();
  } else {
    gate_payload_storage =
        read_tensor_bytes_uncached(path, gate_tensor, 0, gate_tensor.nbytes);
    if (gate_payload_storage.size() != gate_tensor.nbytes) {
      throw std::runtime_error("shared expert gate tensor payload size mismatch");
    }
    gate_payload_data = gate_payload_storage.data();
  }
  if (up_payload != nullptr) {
    if (up_payload->size() != up_tensor.nbytes) {
      throw std::runtime_error("shared expert up resident payload size mismatch");
    }
    up_payload_data = up_payload->data();
  } else {
    up_payload_storage =
        read_tensor_bytes_uncached(path, up_tensor, 0, up_tensor.nbytes);
    if (up_payload_storage.size() != up_tensor.nbytes) {
      throw std::runtime_error("shared expert up tensor payload size mismatch");
    }
    up_payload_data = up_payload_storage.data();
  }

  SharedExpertGateUpFusedRouteResult result;
  result.shared_gate.resize(static_cast<std::size_t>(row_count));
  result.shared_up.resize(static_cast<std::size_t>(row_count));
  result.shared_gate_up.resize(static_cast<std::size_t>(row_count * 2));
  result.shared_swiglu.resize(static_cast<std::size_t>(row_count));

  const auto thread_count = std::min<std::uint64_t>(
      row_count,
      static_cast<std::uint64_t>(dense_matvec_thread_count()));
  const bool use_q4_direct_minsum_pair = q4_direct_minsum_pair_enabled();
  auto compute_range = [&](std::uint64_t begin, std::uint64_t end) {
    for (std::uint64_t row = begin; row < end; ++row) {
      const auto row_offset = static_cast<std::size_t>(row * row_nbytes);
      const float gate =
          use_q4_direct_minsum_pair
              ? dot_q4_k_q8_k_row_direct_minpair(
                    gate_payload_data + row_offset,
                    static_cast<std::size_t>(row_nbytes),
                    q8_ffn_norm)
              : dot_q4_k_q8_k_row_direct(
                    gate_payload_data + row_offset,
                    static_cast<std::size_t>(row_nbytes),
                    q8_ffn_norm);
      const float up =
          use_q4_direct_minsum_pair
              ? dot_q4_k_q8_k_row_direct_minpair(
                    up_payload_data + row_offset,
                    static_cast<std::size_t>(row_nbytes),
                    q8_ffn_norm)
              : dot_q4_k_q8_k_row_direct(
                    up_payload_data + row_offset,
                    static_cast<std::size_t>(row_nbytes),
                    q8_ffn_norm);
      const auto row_index = static_cast<std::size_t>(row);
      result.shared_gate[row_index] = gate;
      result.shared_up[row_index] = up;
      result.shared_gate_up[row_index] = gate;
      result.shared_gate_up[static_cast<std::size_t>(row_count) + row_index] = up;
      result.shared_swiglu[row_index] =
          gate * sigmoid_scalar(gate) * up;
    }
  };
  if (thread_count <= 1) {
    compute_range(0, row_count);
  } else {
    parallel_for_rows(row_count, thread_count,
                      [&](std::uint64_t begin,
                          std::uint64_t end,
                          std::uint64_t) {
      compute_range(begin, end);
    });
  }

  record_matvec_profile(
      "shared_expert_gate_up_swiglu_q4direct",
      gate_tensor.name + "+" + up_tensor.name,
      ffn_norm.size(),
      result.shared_swiglu.size(),
      row_count * 2,
      profile_elapsed_ns(profile_begin, ProfileClock::now()));
  return result;
}

Qwen36MoeFfnLayerResult run_qwen36_moe_ffn_layer(
    const std::string& path,
    const GgufModelIndex& index,
    int layer_index,
    const std::vector<float>& residual_input,
    float rms_norm_epsilon) {
  if (residual_input.empty()) {
    throw std::invalid_argument("FFN residual input is empty");
  }

  constexpr int kExpertUsedCount = 8;
  constexpr float kMinWeightSum = 6.103515625e-5f;

  const std::string prefix = "blk." + std::to_string(layer_index) + ".";
  const auto* gate_up_tensor =
      find_tensor(index, prefix + "ffn_gate_up_exps.weight");
  const auto* down_tensor =
      find_tensor(index, prefix + "ffn_down_exps.weight");
  const auto* shared_gate_tensor =
      find_tensor(index, prefix + "ffn_gate_shexp.weight");
  const auto* shared_up_tensor =
      find_tensor(index, prefix + "ffn_up_shexp.weight");
  if (gate_up_tensor == nullptr || down_tensor == nullptr ||
      shared_gate_tensor == nullptr || shared_up_tensor == nullptr) {
    throw std::invalid_argument("FFN layer tensor set is incomplete");
  }
  if (gate_up_tensor->dims.size() != 3 || down_tensor->dims.size() != 3 ||
      shared_gate_tensor->dims.size() != 2 ||
      shared_up_tensor->dims.size() != 2) {
    throw std::invalid_argument("FFN layer tensor ranks are invalid");
  }
  const auto hidden_size = static_cast<std::uint64_t>(residual_input.size());
  if (gate_up_tensor->dims[0] != hidden_size ||
      down_tensor->dims[1] != hidden_size ||
      shared_gate_tensor->dims[0] != hidden_size ||
      shared_up_tensor->dims[0] != hidden_size) {
    throw std::invalid_argument("FFN layer tensor dims do not match hidden size");
  }
  if (gate_up_tensor->dims[1] % 2 != 0) {
    throw std::invalid_argument("selected expert gate/up rows must be even");
  }
  if (shared_gate_tensor->dims[1] != shared_up_tensor->dims[1]) {
    throw std::invalid_argument("shared expert gate/up dims differ");
  }
  const auto selected_intermediate = gate_up_tensor->dims[1] / 2;
  if (down_tensor->dims[0] != selected_intermediate) {
    throw std::invalid_argument("selected expert down dim mismatch");
  }

  Qwen36MoeFfnLayerResult result;
  const auto norm_weight = decode_tensor_row(
      path, index, layer_tensor_name(layer_index, "post_attention_norm.weight"), 0);
  result.ffn_norm =
      apply_rms_norm(residual_input, norm_weight, rms_norm_epsilon);
  result.router_logits = matvec_tensor(
      path, index, layer_tensor_name(layer_index, "ffn_gate_inp.weight"),
      result.ffn_norm);
  result.router =
      select_router_topk(result.router_logits, kExpertUsedCount, kMinWeightSum);
  if (selected_expert_ffn_enabled()) {
    auto selected_route = run_selected_expert_ffn_route(
        path, *gate_up_tensor, *down_tensor, result.ffn_norm, result.router);
    result.selected_gate_up = std::move(selected_route.selected_gate_up);
    result.selected_swiglu = std::move(selected_route.selected_swiglu);
    result.selected_down = std::move(selected_route.selected_down);
    result.weighted_selected_down =
        std::move(selected_route.weighted_selected_down);
    result.moe_out = std::move(selected_route.moe_out);
  } else {
    result.selected_gate_up = matvec_expert_tensor(
        path, index, layer_tensor_name(layer_index, "ffn_gate_up_exps.weight"),
        result.ffn_norm, result.router.expert_ids);
    result.selected_swiglu = apply_swiglu_from_gate_up(
        result.selected_gate_up,
        selected_intermediate,
        result.router.expert_ids.size());
    result.selected_down = matvec_expert_tensor_per_expert_input(
        path, index, layer_tensor_name(layer_index, "ffn_down_exps.weight"),
        result.selected_swiglu, result.router.expert_ids);
    result.weighted_selected_down = apply_expert_weights(
        result.selected_down, result.router.normalized_weights, hidden_size);
    result.moe_out = aggregate_experts(
        result.weighted_selected_down, result.router.expert_ids.size(), hidden_size);
  }

  if (shared_expert_gate_up_fused_enabled()) {
    auto shared_route = run_shared_expert_gate_up_fused_route(
        path, *shared_gate_tensor, *shared_up_tensor, result.ffn_norm);
    result.shared_gate = std::move(shared_route.shared_gate);
    result.shared_gate_up = std::move(shared_route.shared_gate_up);
    result.shared_swiglu = std::move(shared_route.shared_swiglu);
  } else {
    std::vector<Q8KBlock> q8_ffn_norm;
    const bool reuse_q8_input = matvec_q8_input_reuse_enabled();
    if (reuse_q8_input) {
      q8_ffn_norm = quantize_q8_k_blocks(result.ffn_norm);
    }
    auto matvec_ffn_norm = [&](const std::string& suffix) {
      const auto tensor_name = layer_tensor_name(layer_index, suffix);
      return reuse_q8_input
                 ? matvec_tensor_with_q8_input(
                       path, index, tensor_name, result.ffn_norm, q8_ffn_norm)
                 : matvec_tensor(path, index, tensor_name, result.ffn_norm);
    };
    const auto shared_gate = matvec_ffn_norm("ffn_gate_shexp.weight");
    const auto shared_up = matvec_ffn_norm("ffn_up_shexp.weight");
    result.shared_gate_up.reserve(shared_gate.size() + shared_up.size());
    result.shared_gate_up.insert(
        result.shared_gate_up.end(), shared_gate.begin(), shared_gate.end());
    result.shared_gate_up.insert(
        result.shared_gate_up.end(), shared_up.begin(), shared_up.end());
    result.shared_swiglu = apply_swiglu_pair(shared_gate, shared_up);
  }
  result.shared_down = matvec_tensor(
      path, index, layer_tensor_name(layer_index, "ffn_down_shexp.weight"),
      result.shared_swiglu);
  result.shared_gate = matvec_tensor(
      path, index, layer_tensor_name(layer_index, "ffn_gate_inp_shexp.weight"),
      result.ffn_norm);
  if (result.shared_gate.size() != 1) {
    throw std::runtime_error("shared expert gate output size mismatch");
  }
  result.shared_gate_sigmoid = {sigmoid_scalar(result.shared_gate[0])};
  result.shared_gated =
      multiply_by_scalar(result.shared_down, result.shared_gate_sigmoid[0]);
  result.ffn_out = add_vectors(result.moe_out, result.shared_gated);
  result.residual = add_vectors(residual_input, result.ffn_out);
  return result;
}

Qwen36LayerPostConvResult run_qwen36_layer_with_external_conv_output(
    const std::string& path,
    const GgufModelIndex& index,
    int layer_index,
    const std::vector<float>& residual_input,
    const std::vector<float>& conv_output_raw,
    const std::vector<float>& recurrent_state,
    float rms_norm_epsilon) {
  if (residual_input.empty()) {
    throw std::invalid_argument("layer residual input is empty");
  }
  if (conv_output_raw.empty()) {
    throw std::invalid_argument("linear attention conv output is empty");
  }
  if (recurrent_state.empty()) {
    throw std::invalid_argument("linear attention recurrent state is empty");
  }
  if (find_tensor(index, layer_tensor_name(layer_index, "ssm_out.weight")) == nullptr) {
    throw std::invalid_argument("linear attention output tensor not found");
  }

  Qwen36LayerPostConvResult result;
  const auto attention_norm_weight = decode_tensor_row(
      path, index, layer_tensor_name(layer_index, "attn_norm.weight"), 0);
  result.attention_norm =
      apply_rms_norm(residual_input, attention_norm_weight, rms_norm_epsilon);

  const auto preconv = run_qwen36_linear_attention_preconv_core(
      path, index, layer_index, result.attention_norm);
  const auto ssm_norm_weight = decode_tensor_row(
      path, index, layer_tensor_name(layer_index, "ssm_norm.weight"), 0);

  result.attention = run_qwen36_linear_attention_postconv_core(
      conv_output_raw,
      preconv.gate,
      preconv.beta_sigmoid,
      recurrent_state,
      preconv.z,
      ssm_norm_weight,
      rms_norm_epsilon);
  result.linear_attention_out = matvec_tensor(
      path,
      index,
      layer_tensor_name(layer_index, "ssm_out.weight"),
      result.attention.final_output);
  result.attention_residual =
      add_vectors(residual_input, result.linear_attention_out);
  result.ffn = run_qwen36_moe_ffn_layer(
      path, index, layer_index, result.attention_residual, rms_norm_epsilon);
  result.residual = result.ffn.residual;
  return result;
}

Qwen36StatefulLinearAttentionLayerResult
run_qwen36_stateful_linear_attention_layer(
    const std::string& path,
    const GgufModelIndex& index,
    int layer_index,
    const std::vector<float>& residual_input,
    const std::vector<float>& conv_state,
    const std::vector<float>& recurrent_state,
    float rms_norm_epsilon) {
  if (residual_input.empty()) {
    throw std::invalid_argument("stateful linear attention residual input is empty");
  }
  if (conv_state.empty()) {
    throw std::invalid_argument("stateful linear attention conv state is empty");
  }
  if (recurrent_state.empty()) {
    throw std::invalid_argument("stateful linear attention recurrent state is empty");
  }
  if (find_tensor(index, layer_tensor_name(layer_index, "ssm_out.weight")) == nullptr) {
    throw std::invalid_argument("stateful linear attention output tensor not found");
  }

  Qwen36StatefulLinearAttentionLayerResult result;
  const auto attention_norm_weight = decode_tensor_row(
      path, index, layer_tensor_name(layer_index, "attn_norm.weight"), 0);
  result.attention_norm =
      apply_rms_norm(residual_input, attention_norm_weight, rms_norm_epsilon);
  result.preconv = run_qwen36_linear_attention_preconv_core(
      path, index, layer_index, result.attention_norm);
  result.conv = run_qwen36_linear_attention_conv_core(
      path, index, layer_index, result.preconv.qkv_mixed, conv_state);
  result.state_predelta = recurrent_state;

  const auto ssm_norm_weight = decode_tensor_row(
      path, index, layer_tensor_name(layer_index, "ssm_norm.weight"), 0);
  result.attention = run_qwen36_linear_attention_postconv_core(
      result.conv.conv_output_raw,
      result.preconv.gate,
      result.preconv.beta_sigmoid,
      recurrent_state,
      result.preconv.z,
      ssm_norm_weight,
      rms_norm_epsilon);
  result.linear_attention_out = matvec_tensor(
      path,
      index,
      layer_tensor_name(layer_index, "ssm_out.weight"),
      result.attention.final_output);
  result.attention_residual =
      add_vectors(residual_input, result.linear_attention_out);
  result.ffn = run_qwen36_moe_ffn_layer(
      path, index, layer_index, result.attention_residual, rms_norm_epsilon);
  result.residual = result.ffn.residual;
  return result;
}

Qwen36FullAttentionQkvProjectionResult
run_qwen36_full_attention_qkv_projection(
    const std::string& path,
    const GgufModelIndex& index,
    int layer_index,
    const std::vector<float>& residual_input,
    float rms_norm_epsilon) {
  if (residual_input.empty()) {
    throw std::invalid_argument("full attention residual input is empty");
  }
  if (find_tensor(index, layer_tensor_name(layer_index, "attn_q.weight")) == nullptr ||
      find_tensor(index, layer_tensor_name(layer_index, "attn_k.weight")) == nullptr ||
      find_tensor(index, layer_tensor_name(layer_index, "attn_v.weight")) == nullptr) {
    throw std::invalid_argument("full attention qkv tensor set is incomplete");
  }

  Qwen36FullAttentionQkvProjectionResult result;
  const auto attention_norm_weight = decode_tensor_row(
      path, index, layer_tensor_name(layer_index, "attn_norm.weight"), 0);
  result.attention_norm =
      apply_rms_norm(residual_input, attention_norm_weight, rms_norm_epsilon);

  std::vector<Q8KBlock> q8_attention_norm;
  const bool reuse_q8_input = matvec_q8_input_reuse_enabled();
  if (reuse_q8_input) {
    q8_attention_norm = quantize_q8_k_blocks(result.attention_norm);
  }
  auto matvec_attention_norm = [&](const std::string& suffix) {
    const auto tensor_name = layer_tensor_name(layer_index, suffix);
    return reuse_q8_input
               ? matvec_tensor_with_q8_input(
                     path, index, tensor_name, result.attention_norm, q8_attention_norm)
               : matvec_tensor(path, index, tensor_name, result.attention_norm);
  };

  result.q_full = matvec_attention_norm("attn_q.weight");
  const auto q_norm_weight = decode_tensor_row(
      path, index, layer_tensor_name(layer_index, "attn_q_norm.weight"), 0);
  const auto q_head_dim = q_norm_weight.size();
  if (q_head_dim == 0 ||
      result.q_full.size() % (2 * q_head_dim) != 0) {
    throw std::invalid_argument("full attention q projection split is invalid");
  }
  const auto q_head_count = result.q_full.size() / (2 * q_head_dim);
  result.q_raw.reserve(q_head_count * q_head_dim);
  result.q_gate.reserve(q_head_count * q_head_dim);
  for (std::size_t head = 0; head < q_head_count; ++head) {
    const auto base = head * q_head_dim * 2;
    result.q_raw.insert(
        result.q_raw.end(),
        result.q_full.begin() + static_cast<std::ptrdiff_t>(base),
        result.q_full.begin() + static_cast<std::ptrdiff_t>(base + q_head_dim));
    result.q_gate.insert(
        result.q_gate.end(),
        result.q_full.begin() + static_cast<std::ptrdiff_t>(base + q_head_dim),
        result.q_full.begin() +
            static_cast<std::ptrdiff_t>(base + 2 * q_head_dim));
  }
  result.q_normed =
      apply_repeated_rms_norm(result.q_raw, q_norm_weight, rms_norm_epsilon);

  result.k_raw = matvec_attention_norm("attn_k.weight");
  const auto k_norm_weight = decode_tensor_row(
      path, index, layer_tensor_name(layer_index, "attn_k_norm.weight"), 0);
  result.k_normed =
      apply_repeated_rms_norm(result.k_raw, k_norm_weight, rms_norm_epsilon);

  result.v = matvec_attention_norm("attn_v.weight");
  return result;
}

Qwen36FullAttentionRopeResult run_qwen36_full_attention_rope(
    const std::vector<float>& q_normed,
    const std::vector<float>& k_normed,
    std::int32_t token_position,
    std::uint64_t head_dim,
    std::uint64_t rope_dimension_count,
    const std::vector<std::int64_t>& rope_sections,
    std::uint64_t rope_context_length,
    float rope_freq_base,
    float rope_freq_scale,
    float rope_ext_factor,
    float rope_attn_factor,
    float rope_beta_fast,
    float rope_beta_slow) {
  const auto sections =
      normalize_rope_sections(rope_sections, rope_dimension_count);
  Qwen36FullAttentionRopeResult result;
  result.q_rope = apply_qwen36_imrope(
      q_normed,
      token_position,
      head_dim,
      rope_dimension_count,
      sections,
      rope_context_length,
      rope_freq_base,
      rope_freq_scale,
      rope_ext_factor,
      rope_attn_factor,
      rope_beta_fast,
      rope_beta_slow);
  result.k_rope = apply_qwen36_imrope(
      k_normed,
      token_position,
      head_dim,
      rope_dimension_count,
      sections,
      rope_context_length,
      rope_freq_base,
      rope_freq_scale,
      rope_ext_factor,
      rope_attn_factor,
      rope_beta_fast,
      rope_beta_slow);
  return result;
}

std::vector<float> build_qwen36_rope_cache(
    std::int32_t token_position,
    std::uint64_t rope_dimension_count,
    const std::vector<std::int64_t>& rope_sections,
    std::uint64_t rope_context_length,
    float rope_freq_base,
    float rope_freq_scale,
    float rope_ext_factor,
    float rope_attn_factor,
    float rope_beta_fast,
    float rope_beta_slow) {
  return build_qwen36_imrope_cache(
      token_position,
      rope_dimension_count,
      normalize_rope_sections(rope_sections, rope_dimension_count),
      rope_context_length,
      rope_freq_base,
      rope_freq_scale,
      rope_ext_factor,
      rope_attn_factor,
      rope_beta_fast,
      rope_beta_slow);
}

Qwen36FullAttentionGateResult run_qwen36_full_attention_gate(
    const std::vector<float>& q_full,
    const std::vector<float>& attn_pregate,
    std::uint64_t head_dim) {
  if (head_dim == 0) {
    throw std::invalid_argument("full attention gate head dim is zero");
  }
  if (q_full.empty() || q_full.size() % (2 * head_dim) != 0) {
    throw std::invalid_argument("full attention q_full gate layout is invalid");
  }
  const auto head_count = q_full.size() / (2 * head_dim);
  const auto expected_output_size = head_count * head_dim;
  if (attn_pregate.size() != expected_output_size) {
    throw std::invalid_argument("full attention pregate size mismatch");
  }

  Qwen36FullAttentionGateResult result;
  result.q_gate.reserve(expected_output_size);
  result.gate_sigmoid.reserve(expected_output_size);
  result.attn_gated.reserve(expected_output_size);
  for (std::size_t head = 0; head < head_count; ++head) {
    const auto q_base = head * head_dim * 2;
    const auto out_base = head * head_dim;
    for (std::uint64_t i = 0; i < head_dim; ++i) {
      const auto gate_value =
          q_full[q_base + static_cast<std::size_t>(head_dim + i)];
      const auto gate_sigmoid = sigmoid_scalar(gate_value);
      result.q_gate.push_back(gate_value);
      result.gate_sigmoid.push_back(gate_sigmoid);
      result.attn_gated.push_back(
          attn_pregate[out_base + static_cast<std::size_t>(i)] *
          gate_sigmoid);
    }
  }
  return result;
}

Qwen36FullAttentionCoreResult run_qwen36_full_attention_core(
    const std::vector<float>& q_rope,
    const std::vector<std::vector<float>>& k_history,
    const std::vector<std::vector<float>>& v_history,
    std::uint64_t head_dim,
    std::uint64_t q_head_count,
    std::uint64_t kv_head_count,
    float attention_scale) {
  if (head_dim == 0 || q_head_count == 0 || kv_head_count == 0) {
    throw std::invalid_argument("full attention dimensions must be non-zero");
  }
  if (q_head_count % kv_head_count != 0) {
    throw std::invalid_argument("full attention GQA ratio is invalid");
  }
  if (attention_scale <= 0.0f || !std::isfinite(attention_scale)) {
    throw std::invalid_argument("full attention scale is invalid");
  }
  const auto token_count = k_history.size();
  if (token_count == 0 || token_count != v_history.size()) {
    throw std::invalid_argument("full attention history size mismatch");
  }
  const auto q_size = q_head_count * head_dim;
  const auto kv_size = kv_head_count * head_dim;
  if (q_rope.size() != q_size) {
    throw std::invalid_argument("full attention Q size mismatch");
  }
  for (std::size_t token = 0; token < token_count; ++token) {
    if (k_history[token].size() != kv_size ||
        v_history[token].size() != kv_size) {
      throw std::invalid_argument("full attention K/V token size mismatch");
    }
  }

  Qwen36FullAttentionCoreResult result;
  result.attention_weights.assign(q_head_count * token_count, 0.0f);
  result.attn_pregate.assign(q_size, 0.0f);
  const auto gqa_group = q_head_count / kv_head_count;
  std::vector<float> scores(token_count);
  for (std::uint64_t q_head = 0; q_head < q_head_count; ++q_head) {
    const auto kv_head = q_head / gqa_group;
    const auto q_base = static_cast<std::size_t>(q_head * head_dim);
    const auto kv_base = static_cast<std::size_t>(kv_head * head_dim);
    for (std::size_t token = 0; token < token_count; ++token) {
      double dot = 0.0;
      for (std::uint64_t i = 0; i < head_dim; ++i) {
        dot += static_cast<double>(
                   q_rope[q_base + static_cast<std::size_t>(i)]) *
               static_cast<double>(
                   k_history[token][kv_base + static_cast<std::size_t>(i)]);
      }
      scores[token] = static_cast<float>(
          dot * static_cast<double>(attention_scale));
    }
    const auto weights = softmax(scores);
    for (std::size_t token = 0; token < token_count; ++token) {
      result.attention_weights[
          static_cast<std::size_t>(q_head) * token_count + token] =
          weights[token];
      for (std::uint64_t i = 0; i < head_dim; ++i) {
        result.attn_pregate[q_base + static_cast<std::size_t>(i)] +=
            weights[token] *
            v_history[token][kv_base + static_cast<std::size_t>(i)];
      }
    }
  }
  return result;
}

Qwen36StatefulFullAttentionLayerResult
run_qwen36_stateful_full_attention_layer(
    const std::string& path,
    const GgufModelIndex& index,
    int layer_index,
    const std::vector<float>& residual_input,
    const std::vector<std::vector<float>>& k_history,
    const std::vector<std::vector<float>>& v_history,
    std::int32_t token_position,
    std::uint64_t head_dim,
    std::uint64_t q_head_count,
    std::uint64_t kv_head_count,
    std::uint64_t rope_dimension_count,
    const std::vector<std::int64_t>& rope_sections,
    std::uint64_t rope_context_length,
    float rope_freq_base,
    float rope_freq_scale,
    float rope_ext_factor,
    float rope_attn_factor,
    float rope_beta_fast,
    float rope_beta_slow,
    float attention_scale,
    float rms_norm_epsilon) {
  if (token_position < 0) {
    throw std::invalid_argument("full attention token position is negative");
  }
  if (k_history.size() != v_history.size()) {
    throw std::invalid_argument("full attention input history size mismatch");
  }
  Qwen36StatefulFullAttentionLayerResult result;
  result.qkv = run_qwen36_full_attention_qkv_projection(
      path, index, layer_index, residual_input, rms_norm_epsilon);
  result.rope = run_qwen36_full_attention_rope(
      result.qkv.q_normed,
      result.qkv.k_normed,
      token_position,
      head_dim,
      rope_dimension_count,
      rope_sections,
      rope_context_length,
      rope_freq_base,
      rope_freq_scale,
      rope_ext_factor,
      rope_attn_factor,
      rope_beta_fast,
      rope_beta_slow);
  result.k_history = k_history;
  result.v_history = v_history;
  result.k_history.push_back(result.rope.k_rope);
  result.v_history.push_back(result.qkv.v);
  result.core = run_qwen36_full_attention_core(
      result.rope.q_rope,
      result.k_history,
      result.v_history,
      head_dim,
      q_head_count,
      kv_head_count,
      attention_scale);
  result.gate =
      run_qwen36_full_attention_gate(result.qkv.q_full, result.core.attn_pregate, head_dim);
  result.attention_output = matvec_tensor(
      path,
      index,
      layer_tensor_name(layer_index, "attn_output.weight"),
      result.gate.attn_gated);
  return result;
}

Qwen36LayerShellResult run_qwen36_layer_with_external_attention_state(
    const std::string& path,
    const GgufModelIndex& index,
    int layer_index,
    const std::vector<float>& residual_input,
    const std::vector<float>& attention_projection_input,
    float rms_norm_epsilon) {
  if (residual_input.empty()) {
    throw std::invalid_argument("layer residual input is empty");
  }
  if (attention_projection_input.empty()) {
    throw std::invalid_argument("attention projection input is empty");
  }

  const auto linear_ssm_output =
      layer_tensor_name(layer_index, "ssm_out.weight");
  const auto full_attention_output =
      layer_tensor_name(layer_index, "attn_output.weight");
  std::string output_tensor_name;
  if (find_tensor(index, linear_ssm_output) != nullptr) {
    output_tensor_name = linear_ssm_output;
  } else if (find_tensor(index, full_attention_output) != nullptr) {
    output_tensor_name = full_attention_output;
  } else {
    throw std::invalid_argument("layer attention output tensor not found");
  }

  Qwen36LayerShellResult result;
  result.attention_output =
      matvec_tensor(path, index, output_tensor_name, attention_projection_input);
  result.attention_residual = add_vectors(residual_input, result.attention_output);
  result.ffn = run_qwen36_moe_ffn_layer(
      path, index, layer_index, result.attention_residual, rms_norm_epsilon);
  result.residual = result.ffn.residual;
  return result;
}

Qwen36LoopShellResult run_qwen36_loop_with_external_attention_states(
    const std::string& path,
    const GgufModelIndex& index,
    const std::vector<float>& residual_input,
    const std::vector<std::vector<float>>& attention_projection_inputs,
    float rms_norm_epsilon) {
  if (residual_input.empty()) {
    throw std::invalid_argument("loop residual input is empty");
  }
  if (attention_projection_inputs.empty()) {
    throw std::invalid_argument("loop attention projection inputs are empty");
  }

  Qwen36LoopShellResult result;
  result.layers.reserve(attention_projection_inputs.size());
  auto residual = residual_input;
  for (std::size_t layer = 0; layer < attention_projection_inputs.size(); ++layer) {
    auto layer_result = run_qwen36_layer_with_external_attention_state(
        path,
        index,
        static_cast<int>(layer),
        residual,
        attention_projection_inputs[layer],
        rms_norm_epsilon);
    residual = layer_result.residual;
    result.layers.push_back(std::move(layer_result));
  }

  const auto norm_weight = decode_tensor_row(path, index, "output_norm.weight", 0);
  result.final_norm = apply_rms_norm(residual, norm_weight, rms_norm_epsilon);
  result.logits = matvec_tensor(path, index, "output.weight", result.final_norm);
  return result;
}

}  // namespace iq36
