#include <immintrin.h>
#include <pthread.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr std::size_t kContextTokens = 131072;
constexpr std::size_t kPatternTokens = 256;
constexpr std::size_t kHeadDim = 256;
constexpr std::size_t kQHeads = 16;
constexpr std::size_t kKvHeads = 2;
constexpr std::size_t kGqaGroup = 8;
constexpr std::size_t kChunkTokens = 256;
constexpr std::size_t kChunkCount = kContextTokens / kChunkTokens;
constexpr float kAttentionScale = 0.0625f;
constexpr double kComponentCapMs = 2.825;
constexpr int kWorkers = 16;

[[noreturn]] void Die(const std::string& message) {
  throw std::runtime_error(message);
}

void Require(bool condition, const std::string& message) {
  if (!condition) Die(message);
}

std::uint16_t FloatToHalf(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const std::uint32_t sign = (bits >> 16U) & 0x8000U;
  const std::uint32_t exponent_bits = (bits >> 23U) & 0xffU;
  std::uint32_t mantissa = bits & 0x7fffffU;
  if (exponent_bits == 0xffU) {
    return static_cast<std::uint16_t>(
        sign | 0x7c00U | (mantissa == 0U ? 0U : 0x0200U));
  }
  int exponent = static_cast<int>(exponent_bits) - 127 + 15;
  if (exponent >= 31) return static_cast<std::uint16_t>(sign | 0x7c00U);
  if (exponent <= 0) {
    if (exponent < -10) return static_cast<std::uint16_t>(sign);
    mantissa |= 0x800000U;
    const unsigned shift = static_cast<unsigned>(14 - exponent);
    const std::uint32_t rounded =
        (mantissa + (1U << (shift - 1U)) - 1U +
         ((mantissa >> shift) & 1U)) >> shift;
    return static_cast<std::uint16_t>(sign | rounded);
  }
  mantissa += 0x0fffU + ((mantissa >> 13U) & 1U);
  if ((mantissa & 0x800000U) != 0U) {
    mantissa = 0U;
    ++exponent;
    if (exponent >= 31) return static_cast<std::uint16_t>(sign | 0x7c00U);
  }
  return static_cast<std::uint16_t>(
      sign | (static_cast<std::uint32_t>(exponent) << 10U) |
      (mantissa >> 13U));
}

float HalfToFloat(std::uint16_t value) {
  const std::uint32_t sign = static_cast<std::uint32_t>(value & 0x8000U) << 16U;
  const std::uint32_t exponent = (value >> 10U) & 0x1fU;
  std::uint32_t mantissa = value & 0x03ffU;
  std::uint32_t bits = 0;
  if (exponent == 0U) {
    if (mantissa == 0U) {
      bits = sign;
    } else {
      int shift = 0;
      while ((mantissa & 0x0400U) == 0U) {
        mantissa <<= 1U;
        ++shift;
      }
      mantissa &= 0x03ffU;
      bits = sign | (static_cast<std::uint32_t>(127 - 15 - shift) << 23U) |
          (mantissa << 13U);
    }
  } else if (exponent == 31U) {
    bits = sign | 0x7f800000U | (mantissa << 13U);
  } else {
    bits = sign | ((exponent + 112U) << 23U) | (mantissa << 13U);
  }
  float result = 0.0f;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

std::string CpuModel() {
  std::ifstream input("/proc/cpuinfo");
  std::string line;
  while (std::getline(input, line)) {
    constexpr const char* marker = "model name";
    if (line.rfind(marker, 0) == 0) {
      const auto colon = line.find(':');
      return colon == std::string::npos ? line : line.substr(colon + 2U);
    }
  }
  return "unknown";
}

__attribute__((target("avx2,f16c,fma")))
float DotFp16(const float* q, const std::uint16_t* k) {
  __m256 sum = _mm256_setzero_ps();
  for (std::size_t dim = 0; dim < kHeadDim; dim += 8U) {
    const __m128i packed =
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(k + dim));
    const __m256 values = _mm256_cvtph_ps(packed);
    const __m256 query = _mm256_loadu_ps(q + dim);
    sum = _mm256_fmadd_ps(query, values, sum);
  }
  const __m128 low = _mm256_castps256_ps128(sum);
  const __m128 high = _mm256_extractf128_ps(sum, 1);
  __m128 reduced = _mm_add_ps(low, high);
  reduced = _mm_hadd_ps(reduced, reduced);
  reduced = _mm_hadd_ps(reduced, reduced);
  return _mm_cvtss_f32(reduced);
}

__attribute__((target("avx2,f16c,fma")))
void WeightedValueBlock(const std::uint16_t* v_history,
                        std::size_t kv_head, std::size_t begin,
                        const float* weights, std::size_t dim,
                        float* output) {
  __m256 sum = _mm256_setzero_ps();
  constexpr std::size_t stride = kKvHeads * kHeadDim;
  for (std::size_t index = 0; index < kChunkTokens; ++index) {
    const std::uint16_t* values =
        v_history + (begin + index) * stride + kv_head * kHeadDim + dim;
    const __m128i packed =
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(values));
    const __m256 vector = _mm256_cvtph_ps(packed);
    sum = _mm256_fmadd_ps(vector, _mm256_set1_ps(weights[index]), sum);
  }
  _mm256_storeu_ps(output, sum);
}

__attribute__((target("avx2,fma")))
void ReducePartialBlock(const float* partial_output, const float* scales,
                        std::size_t q_head, std::size_t dim, float divisor,
                        const float* gate, float* output) {
  __m256 sum = _mm256_setzero_ps();
  for (std::size_t chunk = 0; chunk < kChunkCount; ++chunk) {
    const float* values = partial_output +
        (q_head * kChunkCount + chunk) * kHeadDim + dim;
    sum = _mm256_fmadd_ps(
        _mm256_loadu_ps(values), _mm256_set1_ps(scales[chunk]), sum);
  }
  alignas(32) float lanes[8];
  _mm256_store_ps(lanes, sum);
  for (std::size_t lane = 0; lane < 8U; ++lane) {
    const std::size_t index = q_head * kHeadDim + dim + lane;
    const float sigmoid = 1.0f / (1.0f + std::exp(-gate[index]));
    output[index] = (lanes[lane] / divisor) * sigmoid;
  }
}

class Component {
 public:
  Component(const std::vector<float>& q, const std::vector<float>& gate,
            std::vector<std::uint16_t>* k_history,
            std::vector<std::uint16_t>* v_history,
            const std::vector<float>& current_k,
            const std::vector<float>& current_v)
      : q_(q), gate_(gate), k_history_(*k_history), v_history_(*v_history),
        current_k_(current_k), current_v_(current_v),
        partial_max_(kQHeads * kChunkCount),
        partial_sum_(kQHeads * kChunkCount),
        partial_output_(kQHeads * kChunkCount * kHeadDim),
        output_(kQHeads * kHeadDim), affinity_ok_(true) {
    for (int worker = 0; worker < kWorkers; ++worker) {
      workers_.emplace_back(&Component::Worker, this, worker);
    }
  }

  ~Component() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
      ++epoch_;
    }
    start_cv_.notify_all();
    for (auto& worker : workers_) worker.join();
  }

  double Run() {
    const auto begin = std::chrono::steady_clock::now();
    const std::size_t last_base =
        (kContextTokens - 1U) * kKvHeads * kHeadDim;
    for (std::size_t index = 0; index < kKvHeads * kHeadDim; ++index) {
      k_history_[last_base + index] = FloatToHalf(current_k_[index]);
      v_history_[last_base + index] = FloatToHalf(current_v_[index]);
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      done_ = 0;
      ++epoch_;
    }
    start_cv_.notify_all();
    {
      std::unique_lock<std::mutex> lock(mutex_);
      done_cv_.wait(lock, [&] { return done_ == kWorkers; });
    }
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - begin).count();
  }

  const std::vector<float>& output() const { return output_; }
  bool affinity_ok() const { return affinity_ok_.load(); }

 private:
  void Worker(int worker_id) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(worker_id, &set);
    if (pthread_setaffinity_np(pthread_self(), sizeof(set), &set) != 0) {
      affinity_ok_.store(false);
    }
    std::uint64_t seen_epoch = 0;
    while (true) {
      {
        std::unique_lock<std::mutex> lock(mutex_);
        start_cv_.wait(lock, [&] { return stop_ || epoch_ != seen_epoch; });
        if (stop_) return;
        seen_epoch = epoch_;
      }
      ComputeHead(static_cast<std::size_t>(worker_id));
      {
        std::lock_guard<std::mutex> lock(mutex_);
        ++done_;
        if (done_ == kWorkers) done_cv_.notify_one();
      }
    }
  }

  void ComputeHead(std::size_t q_head) {
    constexpr std::size_t stride = kKvHeads * kHeadDim;
    const std::size_t kv_head = q_head / kGqaGroup;
    const float* query = q_.data() + q_head * kHeadDim;
    alignas(32) float scores[kChunkTokens];
    alignas(32) float weights[kChunkTokens];
    for (std::size_t chunk = 0; chunk < kChunkCount; ++chunk) {
      const std::size_t begin = chunk * kChunkTokens;
      float maximum = -INFINITY;
      for (std::size_t index = 0; index < kChunkTokens; ++index) {
        const std::uint16_t* key = k_history_.data() +
            (begin + index) * stride + kv_head * kHeadDim;
        scores[index] = DotFp16(query, key) * kAttentionScale;
        maximum = std::max(maximum, scores[index]);
      }
      float sum = 0.0f;
      for (std::size_t index = 0; index < kChunkTokens; ++index) {
        weights[index] = std::exp(scores[index] - maximum);
        sum += weights[index];
      }
      const std::size_t meta = q_head * kChunkCount + chunk;
      partial_max_[meta] = maximum;
      partial_sum_[meta] = sum;
      float* partial = partial_output_.data() + meta * kHeadDim;
      for (std::size_t dim = 0; dim < kHeadDim; dim += 8U) {
        WeightedValueBlock(v_history_.data(), kv_head, begin, weights, dim,
                           partial + dim);
      }
    }

    float global_max = -INFINITY;
    for (std::size_t chunk = 0; chunk < kChunkCount; ++chunk) {
      global_max = std::max(
          global_max, partial_max_[q_head * kChunkCount + chunk]);
    }
    alignas(32) float scales[kChunkCount];
    float divisor = 0.0f;
    for (std::size_t chunk = 0; chunk < kChunkCount; ++chunk) {
      const std::size_t meta = q_head * kChunkCount + chunk;
      scales[chunk] = std::exp(partial_max_[meta] - global_max);
      divisor += partial_sum_[meta] * scales[chunk];
    }
    for (std::size_t dim = 0; dim < kHeadDim; dim += 8U) {
      ReducePartialBlock(partial_output_.data(), scales, q_head, dim, divisor,
                         gate_.data(), output_.data());
    }
  }

  const std::vector<float>& q_;
  const std::vector<float>& gate_;
  std::vector<std::uint16_t>& k_history_;
  std::vector<std::uint16_t>& v_history_;
  const std::vector<float>& current_k_;
  const std::vector<float>& current_v_;
  std::vector<float> partial_max_;
  std::vector<float> partial_sum_;
  std::vector<float> partial_output_;
  std::vector<float> output_;
  std::vector<std::thread> workers_;
  std::mutex mutex_;
  std::condition_variable start_cv_;
  std::condition_variable done_cv_;
  std::uint64_t epoch_ = 0;
  int done_ = 0;
  bool stop_ = false;
  std::atomic<bool> affinity_ok_;
};

std::vector<float> Reference(const std::vector<float>& q,
                             const std::vector<float>& gate,
                             const std::vector<float>& k_pattern,
                             const std::vector<float>& v_pattern) {
  std::vector<float> output(kQHeads * kHeadDim);
  for (std::size_t q_head = 0; q_head < kQHeads; ++q_head) {
    const std::size_t kv_head = q_head / kGqaGroup;
    std::array<double, kPatternTokens> scores{};
    double maximum = -INFINITY;
    for (std::size_t token = 0; token < kPatternTokens; ++token) {
      double dot = 0.0;
      for (std::size_t dim = 0; dim < kHeadDim; ++dim) {
        dot += static_cast<double>(q[q_head * kHeadDim + dim]) *
            k_pattern[(token * kKvHeads + kv_head) * kHeadDim + dim];
      }
      scores[token] = dot * kAttentionScale;
      maximum = std::max(maximum, scores[token]);
    }
    double divisor = 0.0;
    for (double score : scores) divisor += std::exp(score - maximum);
    for (std::size_t dim = 0; dim < kHeadDim; ++dim) {
      double sum = 0.0;
      for (std::size_t token = 0; token < kPatternTokens; ++token) {
        sum += std::exp(scores[token] - maximum) *
            v_pattern[(token * kKvHeads + kv_head) * kHeadDim + dim];
      }
      const std::size_t index = q_head * kHeadDim + dim;
      output[index] = static_cast<float>(sum / divisor) /
          (1.0f + std::exp(-gate[index]));
    }
  }
  return output;
}

struct Numeric {
  double cosine = 0.0;
  double relative_l2 = 0.0;
  double rmse = 0.0;
  double max_abs = 0.0;
  bool finite = true;
};

Numeric Compare(const std::vector<float>& reference,
                const std::vector<float>& candidate) {
  Require(reference.size() == candidate.size(), "comparison shape mismatch");
  double dot = 0.0;
  double ref_l2 = 0.0;
  double cand_l2 = 0.0;
  double diff_l2 = 0.0;
  double max_abs = 0.0;
  bool finite = true;
  for (std::size_t index = 0; index < reference.size(); ++index) {
    const double ref = reference[index];
    const double cand = candidate[index];
    finite = finite && std::isfinite(ref) && std::isfinite(cand);
    const double diff = cand - ref;
    dot += ref * cand;
    ref_l2 += ref * ref;
    cand_l2 += cand * cand;
    diff_l2 += diff * diff;
    max_abs = std::max(max_abs, std::abs(diff));
  }
  return {dot / (std::sqrt(ref_l2) * std::sqrt(cand_l2)),
          std::sqrt(diff_l2 / ref_l2),
          std::sqrt(diff_l2 / static_cast<double>(reference.size())),
          max_abs, finite};
}

double Median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2U];
}

void EmitSamples(const std::vector<double>& samples) {
  std::cout << "[";
  for (std::size_t index = 0; index < samples.size(); ++index) {
    if (index != 0U) std::cout << ",";
    std::cout << samples[index];
  }
  std::cout << "]";
}

}  // namespace

int main() {
  try {
    const std::size_t output_values = kQHeads * kHeadDim;
    const std::size_t pattern_values = kPatternTokens * kKvHeads * kHeadDim;
    const std::size_t history_values = kContextTokens * kKvHeads * kHeadDim;
    std::vector<float> q(output_values);
    std::vector<float> gate(output_values);
    for (std::size_t index = 0; index < output_values; ++index) {
      q[index] = static_cast<float>(
          static_cast<int>((index * 29U + 17U) & 255U) - 128) / 1024.0f;
      gate[index] = static_cast<float>(
          static_cast<int>((index * 11U + 3U) & 127U) - 64) / 64.0f;
    }
    std::vector<float> k_pattern(pattern_values);
    std::vector<float> v_pattern(pattern_values);
    for (std::size_t token = 0; token < kPatternTokens; ++token) {
      for (std::size_t dim = 0; dim < kKvHeads * kHeadDim; ++dim) {
        const std::size_t index = token * kKvHeads * kHeadDim + dim;
        k_pattern[index] = static_cast<float>(
            static_cast<int>((token * 13U + dim * 7U + 5U) & 255U) - 128)
            / 2048.0f;
        v_pattern[index] = 0.02f + static_cast<float>(
            static_cast<int>((token * 3U + dim * 19U + 9U) & 255U) - 128)
            / 4096.0f;
      }
    }
    std::vector<std::uint16_t> k_half_pattern(pattern_values);
    std::vector<std::uint16_t> v_half_pattern(pattern_values);
    for (std::size_t index = 0; index < pattern_values; ++index) {
      k_half_pattern[index] = FloatToHalf(k_pattern[index]);
      v_half_pattern[index] = FloatToHalf(v_pattern[index]);
    }
    std::vector<std::uint16_t> k_history(history_values);
    std::vector<std::uint16_t> v_history(history_values);
    for (std::size_t offset = 0; offset < history_values;
         offset += pattern_values) {
      std::memcpy(k_history.data() + offset, k_half_pattern.data(),
                  pattern_values * sizeof(std::uint16_t));
      std::memcpy(v_history.data() + offset, v_half_pattern.data(),
                  pattern_values * sizeof(std::uint16_t));
    }
    std::vector<float> current_k(kKvHeads * kHeadDim);
    std::vector<float> current_v(kKvHeads * kHeadDim);
    std::copy(k_pattern.end() - current_k.size(), k_pattern.end(),
              current_k.begin());
    std::copy(v_pattern.end() - current_v.size(), v_pattern.end(),
              current_v.begin());
    const auto reference = Reference(q, gate, k_pattern, v_pattern);
    Component component(q, gate, &k_history, &v_history, current_k, current_v);
    for (int warmup = 0; warmup < 5; ++warmup) (void)component.Run();
    std::vector<double> repeat_samples;
    std::vector<double> confirm_samples;
    for (int sample = 0; sample < 7; ++sample) {
      repeat_samples.push_back(component.Run());
    }
    for (int sample = 0; sample < 7; ++sample) {
      confirm_samples.push_back(component.Run());
    }
    const double repeat_ms = Median(repeat_samples);
    const double confirm_ms = Median(confirm_samples);
    const double spread = std::abs(repeat_ms - confirm_ms) /
        std::max(repeat_ms, confirm_ms);
    const auto numeric = Compare(reference, component.output());
    const bool numeric_pass = numeric.finite && numeric.cosine >= 0.999 &&
        numeric.relative_l2 <= 0.002;
    const bool timing_pass = repeat_ms <= kComponentCapMs &&
        confirm_ms <= kComponentCapMs && spread <= 0.005;
    const bool pass = component.affinity_ok() && numeric_pass && timing_pass;

    std::cout << std::boolalpha << std::setprecision(12) << "{"
              << "\"affinity_cpu_ids\":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],"
              << "\"affinity_pass\":" << component.affinity_ok() << ","
              << "\"algorithm\":\"cpu_avx2_fp16_gqa_chunked\","
              << "\"chunk_tokens\":" << kChunkTokens << ","
              << "\"component_cap_ms\":" << kComponentCapMs << ","
              << "\"confirm_ms\":" << confirm_ms << ","
              << "\"confirm_samples_ms\":";
    EmitSamples(confirm_samples);
    std::cout << ",\"context_tokens\":" << kContextTokens << ","
              << "\"cpu_model\":\"" << CpuModel() << "\","
              << "\"finite\":" << numeric.finite << ","
              << "\"gqa_group\":" << kGqaGroup << ","
              << "\"head_dim\":" << kHeadDim << ","
              << "\"isa\":\"avx2_f16c_fma\","
              << "\"kv_bytes\":"
              << 2U * history_values * sizeof(std::uint16_t) << ","
              << "\"kv_dtype\":\"fp16\","
              << "\"kv_head_count\":" << kKvHeads << ","
              << "\"max_abs\":" << numeric.max_abs << ","
              << "\"numeric_pass\":" << numeric_pass << ","
              << "\"output_cosine\":" << numeric.cosine << ","
              << "\"output_relative_l2\":" << numeric.relative_l2 << ","
              << "\"output_rmse\":" << numeric.rmse << ","
              << "\"q_head_count\":" << kQHeads << ","
              << "\"repeat_ms\":" << repeat_ms << ","
              << "\"repeat_samples_ms\":";
    EmitSamples(repeat_samples);
    std::cout << ",\"required_checks_passed\":" << pass << ","
              << "\"spread\":" << spread << ","
              << "\"timed_current_token_conversion\":true,"
              << "\"timed_worker_synchronization\":true,"
              << "\"timing_pass\":" << timing_pass << ","
              << "\"worker_count\":" << kWorkers << "}" << std::endl;
    return pass ? 0 : 2;
  } catch (const std::exception& exception) {
    std::cerr << "iq36-cpu-avx2-fp16-gqa-decode: " << exception.what() << '\n';
    return 4;
  }
}
