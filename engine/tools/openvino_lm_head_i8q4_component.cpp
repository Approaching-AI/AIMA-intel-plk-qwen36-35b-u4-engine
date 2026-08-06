#include <level_zero/ze_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

constexpr std::uint32_t kIntelVendorId = 0x8086U;
constexpr std::uint32_t kPtlDeviceId = 0xB080U;
constexpr std::uint32_t kRows = 248320U;
constexpr std::uint32_t kColumns = 2048U;
constexpr std::uint32_t kRowsPerStripe = 8U;
constexpr std::uint32_t kGroupsPerRow = 16U;
constexpr std::uint32_t kCodeBytesPerGroupStripe = 512U;
constexpr std::uint32_t kCodeBytesPerStripe = 8192U;
constexpr std::uint32_t kStripeBytes = 8224U;
constexpr std::uint32_t kBinaryCodeBytesPerStripe = 2048U;
constexpr std::uint32_t kBinaryStripeBytes = 2128U;
constexpr std::uint64_t kWeightOffset = UINT64_C(18137149498);
constexpr std::uint64_t kWeightBytes = UINT64_C(508559360);
constexpr std::uint64_t kScaleOffset = UINT64_C(18645708858);
constexpr std::uint64_t kScaleBytes = UINT64_C(496640);
constexpr std::uint32_t kTopK8 = 8U;
constexpr std::uint32_t kMatvecBlockCount = kRows / 256U;
constexpr int kFirstDecodePhase = 1;
constexpr int kLastDecodePhase = 17;

[[noreturn]] void Die(const std::string& message) {
  throw std::runtime_error(message);
}

void Require(bool condition, const std::string& message) {
  if (!condition) Die(message);
}

void Check(ze_result_t result, const char* where) {
  if (result != ZE_RESULT_SUCCESS) {
    Die(std::string(where) + " failed with ze_result_t " +
        std::to_string(static_cast<unsigned int>(result)));
  }
}

std::vector<std::uint8_t> ReadRange(
    const std::string& path, std::uint64_t offset, std::uint64_t bytes) {
  std::ifstream input(path, std::ios::binary);
  Require(static_cast<bool>(input), "failed to open model binary");
  input.seekg(static_cast<std::streamoff>(offset));
  Require(static_cast<bool>(input), "failed to seek model binary");
  std::vector<std::uint8_t> values(static_cast<std::size_t>(bytes));
  input.read(reinterpret_cast<char*>(values.data()),
             static_cast<std::streamsize>(values.size()));
  Require(static_cast<bool>(input), "failed to read model binary range");
  return values;
}

std::vector<std::uint8_t> ReadBinary(const std::string& path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  Require(static_cast<bool>(input), "failed to open binary input");
  const auto size = input.tellg();
  Require(size > 0, "binary input is empty");
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
  input.seekg(0, std::ios::beg);
  input.read(reinterpret_cast<char*>(bytes.data()), size);
  Require(static_cast<bool>(input), "failed to read binary input");
  return bytes;
}

std::vector<float> ReadF32(const std::filesystem::path& path,
                           std::size_t count) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  Require(static_cast<bool>(input), "failed to open F32 input " + path.string());
  const auto bytes = input.tellg();
  Require(bytes == static_cast<std::streamoff>(count * sizeof(float)),
          "F32 input size mismatch " + path.string());
  std::vector<float> values(count);
  input.seekg(0, std::ios::beg);
  input.read(reinterpret_cast<char*>(values.data()), bytes);
  Require(static_cast<bool>(input), "failed to read F32 input " + path.string());
  return values;
}

void WriteF32(const std::filesystem::path& path,
              const std::vector<float>& values) {
  std::ofstream output(path, std::ios::binary);
  Require(static_cast<bool>(output), "failed to open F32 output " + path.string());
  output.write(reinterpret_cast<const char*>(values.data()),
               static_cast<std::streamsize>(values.size() * sizeof(float)));
  Require(static_cast<bool>(output), "failed to write F32 output " + path.string());
}

std::vector<std::uint8_t> PackI8Q4Rowstripe8(
    const std::vector<std::uint8_t>& raw,
    const std::vector<std::uint8_t>& scales) {
  Require(raw.size() == kWeightBytes, "locked LM-head I8 byte size mismatch");
  Require(scales.size() == kScaleBytes,
          "locked LM-head scale byte size mismatch");
  std::uint8_t code_lut[256]{};
  for (int value = -128; value <= 127; ++value) {
    const int signed_code = std::clamp(
        static_cast<int>(std::nearbyint(static_cast<float>(value) / 16.0f)),
        -8, 7);
    code_lut[static_cast<std::uint8_t>(value)] =
        static_cast<std::uint8_t>(signed_code + 8);
  }
  const std::size_t stripe_count = kRows / kRowsPerStripe;
  std::vector<std::uint8_t> packed(stripe_count * kStripeBytes, 0U);
  for (std::uint32_t row_group = 0; row_group < stripe_count; ++row_group) {
    auto* stripe = packed.data() +
        static_cast<std::size_t>(row_group) * kStripeBytes;
    for (std::uint32_t group = 0; group < kGroupsPerRow; ++group) {
      auto* block = stripe + group * kCodeBytesPerGroupStripe;
      for (std::uint32_t chunk = 0; chunk < 16U; ++chunk) {
        for (std::uint32_t lane = 0; lane < kRowsPerStripe; ++lane) {
          const std::uint32_t row = row_group * kRowsPerStripe + lane;
          const std::size_t row_offset =
              static_cast<std::size_t>(row) * kColumns;
          for (std::uint32_t element = 0; element < 4U; ++element) {
            const std::size_t low = row_offset + group * 128U +
                chunk * 4U + element;
            const std::size_t high = low + 64U;
            block[chunk * 32U + lane * 4U + element] =
                static_cast<std::uint8_t>(
                    code_lut[raw[low]] | (code_lut[raw[high]] << 4U));
          }
        }
      }
    }
    for (std::uint32_t lane = 0; lane < kRowsPerStripe; ++lane) {
      const std::uint32_t row = row_group * kRowsPerStripe + lane;
      std::memcpy(stripe + kCodeBytesPerStripe + lane * 2U,
                  scales.data() + static_cast<std::size_t>(row) * 2U, 2U);
    }
  }
  return packed;
}

std::vector<std::uint8_t> PackI8Q1Rowstripe8(
    const std::vector<std::uint8_t>& raw,
    const std::vector<std::uint8_t>& scales) {
  Require(raw.size() == kWeightBytes, "locked LM-head I8 byte size mismatch");
  Require(scales.size() == kScaleBytes,
          "locked LM-head scale byte size mismatch");
  const std::size_t stripe_count = kRows / kRowsPerStripe;
  std::vector<std::uint8_t> packed(
      stripe_count * kBinaryStripeBytes, 0U);
  for (std::uint32_t row_group = 0; row_group < stripe_count; ++row_group) {
    auto* stripe = packed.data() +
        static_cast<std::size_t>(row_group) * kBinaryStripeBytes;
    for (std::uint32_t lane = 0; lane < kRowsPerStripe; ++lane) {
      const std::uint32_t row = row_group * kRowsPerStripe + lane;
      const std::size_t row_offset =
          static_cast<std::size_t>(row) * kColumns;
      std::uint32_t histogram[256]{};
      for (std::uint32_t column = 0; column < kColumns; ++column)
        ++histogram[raw[row_offset + column]];
      std::int64_t low_sum = 0;
      std::int64_t high_sum = 0;
      std::uint32_t low_count = 0;
      std::uint32_t high_count = 0;
      for (int value = -128; value <= 127; ++value) {
        const auto count = histogram[static_cast<std::uint8_t>(value)];
        if (value < 0) {
          low_sum += static_cast<std::int64_t>(value) * count;
          low_count += count;
        } else {
          high_sum += static_cast<std::int64_t>(value) * count;
          high_count += count;
        }
      }
      float low = static_cast<float>(low_sum) /
          static_cast<float>(std::max(1U, low_count));
      float high = static_cast<float>(high_sum) /
          static_cast<float>(std::max(1U, high_count));
      for (int iteration = 0; iteration < 5; ++iteration) {
        const float threshold = (low + high) * 0.5f;
        low_sum = 0;
        high_sum = 0;
        low_count = 0;
        high_count = 0;
        for (int value = -128; value <= 127; ++value) {
          const auto count = histogram[static_cast<std::uint8_t>(value)];
          if (static_cast<float>(value) <= threshold) {
            low_sum += static_cast<std::int64_t>(value) * count;
            low_count += count;
          } else {
            high_sum += static_cast<std::int64_t>(value) * count;
            high_count += count;
          }
        }
        if (low_count != 0U)
          low = static_cast<float>(low_sum) / static_cast<float>(low_count);
        if (high_count != 0U)
          high = static_cast<float>(high_sum) / static_cast<float>(high_count);
      }
      const float threshold = (low + high) * 0.5f;
      std::memcpy(
          stripe + kBinaryCodeBytesPerStripe + lane * sizeof(float),
          &low, sizeof(float));
      std::memcpy(
          stripe + kBinaryCodeBytesPerStripe + 8U * sizeof(float) +
              lane * sizeof(float),
          &high, sizeof(float));
      std::memcpy(
          stripe + kBinaryCodeBytesPerStripe + 16U * sizeof(float) +
              lane * 2U,
          scales.data() + static_cast<std::size_t>(row) * 2U, 2U);
      for (std::uint32_t chunk = 0; chunk < kColumns / 32U; ++chunk) {
        for (std::uint32_t byte = 0; byte < 4U; ++byte) {
          std::uint8_t bits = 0U;
          for (std::uint32_t bit = 0; bit < 8U; ++bit) {
            const std::uint32_t column = chunk * 32U + byte * 8U + bit;
            const auto value = static_cast<std::int8_t>(
                raw[row_offset + column]);
            if (static_cast<float>(value) > threshold)
              bits |= static_cast<std::uint8_t>(1U << bit);
          }
          stripe[chunk * 32U + lane * 4U + byte] = bits;
        }
      }
    }
  }
  return packed;
}

double Minimum(const std::vector<double>& values) {
  Require(!values.empty(), "timing vector is empty");
  return *std::min_element(values.begin(), values.end());
}

double Mean(const std::vector<double>& values) {
  Require(!values.empty(), "timing vector is empty");
  double total = 0.0;
  for (double value : values) total += value;
  return total / static_cast<double>(values.size());
}

std::vector<double> AddSamples(const std::vector<double>& lhs,
                               const std::vector<double>& rhs) {
  Require(lhs.size() == rhs.size(), "timing sample count mismatch");
  std::vector<double> result(lhs.size());
  for (std::size_t index = 0; index < lhs.size(); ++index) {
    result[index] = lhs[index] + rhs[index];
  }
  return result;
}

void WriteDoubleArray(const std::vector<double>& values) {
  std::cout << "[";
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0U) std::cout << ",";
    std::cout << values[index];
  }
  std::cout << "]";
}

void WriteIntArray(const std::vector<std::int32_t>& values) {
  std::cout << "[";
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0U) std::cout << ",";
    std::cout << values[index];
  }
  std::cout << "]";
}

std::uint64_t TimestampDelta(std::uint64_t start,
                             std::uint64_t end,
                             std::uint32_t valid_bits) {
  if (valid_bits == 0U || valid_bits >= 64U) return end - start;
  const std::uint64_t mask = (UINT64_C(1) << valid_bits) - 1U;
  return (end - start) & mask;
}

enum class TokenBoundary {
  kHostFullOutput,
  kDeviceFullOutput,
  kDeviceCompactTop3,
  kDeviceCompactDirectTop8,
};

class Runtime {
 public:
  Runtime(const std::string& module_path,
          const std::vector<std::uint8_t>& packed,
          const std::vector<std::uint8_t>& raw,
          const std::vector<std::uint8_t>& scales,
          std::uint32_t topk,
          bool binary,
          TokenBoundary token_boundary = TokenBoundary::kHostFullOutput)
      : packed_bytes_(packed.size()), topk_(topk), binary_(binary),
        token_boundary_(token_boundary) {
    Require(topk_ == 0U || (!binary_ && topk_ == kTopK8) ||
                (binary_ && (topk_ == 2U || topk_ == 12U)),
            "only Q4 top-8 or binary block-local top-2/top-12 is supported");
    Require(token_boundary_ == TokenBoundary::kHostFullOutput ||
                (binary_ && topk_ == 2U),
            "device token boundaries require binary local-top2");
    const bool compact =
        token_boundary_ == TokenBoundary::kDeviceCompactTop3 ||
        token_boundary_ == TokenBoundary::kDeviceCompactDirectTop8;
    InitializeDevice();
    InitializeRuntime(module_path);
    packed_ = Upload(packed.data(), packed.size());
    if (topk_ != 0U) {
      Require(raw.size() == kWeightBytes,
              "top-8 correction raw I8 byte size mismatch");
      Require(scales.size() == kScaleBytes,
              "top-8 correction scale byte size mismatch");
      raw_ = Upload(raw.data(), raw.size());
      scales_ = Upload(scales.data(), scales.size());
      const std::uint32_t partial_topk = compact ? 3U : topk_;
      partial_top_ids_ = AllocateDevice(
          kMatvecBlockCount * partial_topk * sizeof(std::int32_t));
      if (!binary_ || compact) {
        partial_top_values_ = AllocateDevice(
            kMatvecBlockCount * partial_topk * sizeof(float));
      }
      if (!binary_) {
        top_ids_ = static_cast<std::int32_t*>(
            AllocateShared(kTopK8 * sizeof(std::int32_t)));
        top_values_ = static_cast<float*>(
            AllocateShared(kTopK8 * sizeof(float)));
      } else if (
          token_boundary_ == TokenBoundary::kDeviceCompactDirectTop8) {
        top_ids_ = static_cast<std::int32_t*>(
            AllocateDevice(kTopK8 * sizeof(std::int32_t)));
        top_values_ = static_cast<float*>(
            AllocateDevice(kTopK8 * sizeof(float)));
      }
    }
    input_ = static_cast<float*>(AllocateShared(kColumns * sizeof(float)));
    if (!compact)
      output_ = static_cast<float*>(AllocateShared(kRows * sizeof(float)));
    if (token_boundary_ != TokenBoundary::kHostFullOutput) {
      token_ = static_cast<std::int32_t*>(
          AllocateShared(sizeof(std::int32_t)));
    }
    if (token_boundary_ == TokenBoundary::kDeviceFullOutput)
      greedy_partials_ = AllocateDevice(64U * sizeof(std::uint64_t));
    q8_ = AllocateDevice(kColumns * sizeof(std::int8_t));
    q8_d_ = AllocateDevice((kColumns / 256U) * sizeof(float));
    if (binary_)
      q8_sums_ = AllocateDevice((kColumns / 256U) * sizeof(std::int32_t));
    FinishUploads();
    Record();
  }

  ~Runtime() { Cleanup(); }

  struct Run {
    std::vector<double> q8_us;
    std::vector<double> matvec_us;
    std::vector<double> topk_merge_us;
    std::vector<double> correction_us;
    std::vector<double> boundary_partials_us;
    std::vector<double> boundary_merge_us;
    std::vector<double> wall_us;
  };

  Run Execute(const std::vector<float>& input, int warmup, int samples) {
    Require(input.size() == kColumns, "hidden input shape mismatch");
    Require(warmup >= 0 && samples > 0, "invalid timing sample counts");
    std::memcpy(input_, input.data(), kColumns * sizeof(float));
    const bool compact_direct =
        token_boundary_ == TokenBoundary::kDeviceCompactDirectTop8;
    Run run;
    for (int sample = -warmup; sample < samples; ++sample) {
      const auto begin = std::chrono::steady_clock::now();
      Check(zeCommandQueueExecuteCommandLists(
                queue_, 1, &command_list_, nullptr),
            "zeCommandQueueExecuteCommandLists");
      Check(zeCommandQueueSynchronize(queue_, UINT64_MAX),
            "zeCommandQueueSynchronize");
      const auto end = std::chrono::steady_clock::now();
      ze_kernel_timestamp_result_t q8_timestamp{};
      ze_kernel_timestamp_result_t matvec_timestamp{};
      ze_kernel_timestamp_result_t topk_merge_timestamp{};
      ze_kernel_timestamp_result_t correction_timestamp{};
      ze_kernel_timestamp_result_t boundary_partials_timestamp{};
      ze_kernel_timestamp_result_t boundary_merge_timestamp{};
      Check(zeEventQueryKernelTimestamp(q8_event_, &q8_timestamp),
            "zeEventQueryKernelTimestamp(q8)");
      Check(zeEventQueryKernelTimestamp(matvec_event_, &matvec_timestamp),
            "zeEventQueryKernelTimestamp(matvec)");
      if ((!binary_ && topk_ == kTopK8) || compact_direct) {
        Check(zeEventQueryKernelTimestamp(
                  topk_merge_event_, &topk_merge_timestamp),
              "zeEventQueryKernelTimestamp(topk merge)");
      }
      if (topk_ != 0U)
        Check(zeEventQueryKernelTimestamp(
                  correction_event_, &correction_timestamp),
              "zeEventQueryKernelTimestamp(correction)");
      if (token_boundary_ == TokenBoundary::kDeviceFullOutput ||
          compact_direct) {
        Check(zeEventQueryKernelTimestamp(
                  boundary_partials_event_, &boundary_partials_timestamp),
              "zeEventQueryKernelTimestamp(boundary partials)");
      }
      if (token_boundary_ != TokenBoundary::kHostFullOutput) {
        Check(zeEventQueryKernelTimestamp(
                  boundary_merge_event_, &boundary_merge_timestamp),
              "zeEventQueryKernelTimestamp(boundary merge)");
      }
      const auto q8_ticks = TimestampDelta(
          q8_timestamp.context.kernelStart,
          q8_timestamp.context.kernelEnd,
          properties_.kernelTimestampValidBits);
      const auto matvec_ticks = TimestampDelta(
          matvec_timestamp.context.kernelStart,
          matvec_timestamp.context.kernelEnd,
          properties_.kernelTimestampValidBits);
      const auto topk_merge_ticks =
          ((!binary_ && topk_ == kTopK8) || compact_direct)
          ? TimestampDelta(
                topk_merge_timestamp.context.kernelStart,
                topk_merge_timestamp.context.kernelEnd,
                properties_.kernelTimestampValidBits)
          : 0U;
      const auto correction_ticks = topk_ != 0U
          ? TimestampDelta(
                correction_timestamp.context.kernelStart,
                correction_timestamp.context.kernelEnd,
                properties_.kernelTimestampValidBits)
          : 0U;
      const auto boundary_partials_ticks =
          (token_boundary_ == TokenBoundary::kDeviceFullOutput ||
           compact_direct)
          ? TimestampDelta(
                boundary_partials_timestamp.context.kernelStart,
                boundary_partials_timestamp.context.kernelEnd,
                properties_.kernelTimestampValidBits)
          : 0U;
      const auto boundary_merge_ticks =
          token_boundary_ != TokenBoundary::kHostFullOutput
          ? TimestampDelta(
                boundary_merge_timestamp.context.kernelStart,
                boundary_merge_timestamp.context.kernelEnd,
                properties_.kernelTimestampValidBits)
          : 0U;
      Check(zeEventHostReset(q8_event_), "zeEventHostReset(q8)");
      Check(zeEventHostReset(matvec_event_), "zeEventHostReset(matvec)");
      if ((!binary_ && topk_ == kTopK8) || compact_direct) {
        Check(zeEventHostReset(topk_merge_event_),
              "zeEventHostReset(topk merge)");
      }
      if (topk_ != 0U)
        Check(zeEventHostReset(correction_event_),
              "zeEventHostReset(correction)");
      if (token_boundary_ == TokenBoundary::kDeviceFullOutput ||
          compact_direct) {
        Check(zeEventHostReset(boundary_partials_event_),
              "zeEventHostReset(boundary partials)");
      }
      if (token_boundary_ != TokenBoundary::kHostFullOutput) {
        Check(zeEventHostReset(boundary_merge_event_),
              "zeEventHostReset(boundary merge)");
      }
      if (sample < 0) continue;
      run.q8_us.push_back(q8_ticks * timestamp_ns_per_tick_ / 1000.0);
      run.matvec_us.push_back(
          matvec_ticks * timestamp_ns_per_tick_ / 1000.0);
      run.topk_merge_us.push_back(
          topk_merge_ticks * timestamp_ns_per_tick_ / 1000.0);
      run.correction_us.push_back(
          correction_ticks * timestamp_ns_per_tick_ / 1000.0);
      run.boundary_partials_us.push_back(
          boundary_partials_ticks * timestamp_ns_per_tick_ / 1000.0);
      run.boundary_merge_us.push_back(
          boundary_merge_ticks * timestamp_ns_per_tick_ / 1000.0);
      run.wall_us.push_back(std::chrono::duration<double, std::micro>(
                                end - begin).count());
    }
    return run;
  }

  std::vector<float> Output() const {
    Require(output_ != nullptr, "compact token runtime has no full output");
    return {output_, output_ + kRows};
  }

  std::int32_t GreedyToken() const {
    if (token_boundary_ != TokenBoundary::kHostFullOutput) return token_[0];
    std::int32_t best_id = 0;
    float best_value = output_[0];
    for (std::uint32_t row = 1; row < kRows; ++row) {
      if (output_[row] > best_value) {
        best_value = output_[row];
        best_id = static_cast<std::int32_t>(row);
      }
    }
    return best_id;
  }

  std::vector<std::int32_t> SelectedIds() const {
    if (binary_ || topk_ != kTopK8) return {};
    return {top_ids_, top_ids_ + topk_};
  }

  const std::string& device_name() const { return device_name_; }
  std::uint64_t packed_bytes() const { return packed_bytes_; }
  std::uint32_t topk() const { return topk_; }
  double timestamp_ns_per_tick() const { return timestamp_ns_per_tick_; }

 private:
  void InitializeDevice() {
    Check(zeInit(ZE_INIT_FLAG_GPU_ONLY), "zeInit");
    std::uint32_t driver_count = 0;
    Check(zeDriverGet(&driver_count, nullptr), "zeDriverGet(count)");
    std::vector<ze_driver_handle_t> drivers(driver_count);
    Check(zeDriverGet(&driver_count, drivers.data()), "zeDriverGet(list)");
    for (auto driver : drivers) {
      std::uint32_t device_count = 0;
      Check(zeDeviceGet(driver, &device_count, nullptr), "zeDeviceGet(count)");
      std::vector<ze_device_handle_t> devices(device_count);
      Check(zeDeviceGet(driver, &device_count, devices.data()),
            "zeDeviceGet(list)");
      for (auto device : devices) {
        ze_device_properties_t properties{ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES};
        Check(zeDeviceGetProperties(device, &properties),
              "zeDeviceGetProperties");
        if (properties.vendorId != kIntelVendorId ||
            properties.deviceId != kPtlDeviceId) {
          continue;
        }
        std::uint32_t group_count = 0;
        Check(zeDeviceGetCommandQueueGroupProperties(
                  device, &group_count, nullptr),
              "zeDeviceGetCommandQueueGroupProperties(count)");
        std::vector<ze_command_queue_group_properties_t> groups(group_count);
        for (auto& group : groups) {
          group.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES;
        }
        Check(zeDeviceGetCommandQueueGroupProperties(
                  device, &group_count, groups.data()),
              "zeDeviceGetCommandQueueGroupProperties(list)");
        for (std::uint32_t ordinal = 0; ordinal < group_count; ++ordinal) {
          if ((groups[ordinal].flags &
               ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) != 0U) {
            driver_ = driver;
            device_ = device;
            queue_ordinal_ = ordinal;
            properties_ = properties;
            device_name_ = properties.name;
            break;
          }
        }
        if (device_ != nullptr) break;
      }
      if (device_ != nullptr) break;
    }
    Require(device_ != nullptr, "PTL Level Zero device not found");
    std::uint64_t host0 = 0;
    std::uint64_t device0 = 0;
    std::uint64_t host1 = 0;
    std::uint64_t device1 = 0;
    Check(zeDeviceGetGlobalTimestamps(device_, &host0, &device0),
          "zeDeviceGetGlobalTimestamps(start)");
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    Check(zeDeviceGetGlobalTimestamps(device_, &host1, &device1),
          "zeDeviceGetGlobalTimestamps(end)");
    const auto ticks = TimestampDelta(
        device0, device1, properties_.kernelTimestampValidBits);
    Require(host1 > host0 && ticks > 0U, "timestamp calibration failed");
    timestamp_ns_per_tick_ =
        static_cast<double>(host1 - host0) / static_cast<double>(ticks);
  }

  void InitializeRuntime(const std::string& module_path) {
    ze_context_desc_t context_desc{ZE_STRUCTURE_TYPE_CONTEXT_DESC};
    Check(zeContextCreate(driver_, &context_desc, &context_),
          "zeContextCreate");
    ze_command_queue_desc_t queue_desc{
        ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC};
    queue_desc.ordinal = queue_ordinal_;
    queue_desc.index = 0;
    queue_desc.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
    Check(zeCommandQueueCreate(context_, device_, &queue_desc, &queue_),
          "zeCommandQueueCreate");
    ze_command_list_desc_t list_desc{ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC};
    list_desc.commandQueueGroupOrdinal = queue_ordinal_;
    Check(zeCommandListCreate(context_, device_, &list_desc, &upload_list_),
          "zeCommandListCreate(upload)");
    Check(zeCommandListCreate(context_, device_, &list_desc, &command_list_),
          "zeCommandListCreate(run)");
    ze_event_pool_desc_t event_pool_desc{ZE_STRUCTURE_TYPE_EVENT_POOL_DESC};
    event_pool_desc.flags =
        ZE_EVENT_POOL_FLAG_HOST_VISIBLE | ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP;
    event_pool_desc.count =
        token_boundary_ != TokenBoundary::kHostFullOutput ? 6U :
        topk_ != 0U ? 4U : 2U;
    Check(zeEventPoolCreate(
              context_, &event_pool_desc, 1, &device_, &event_pool_),
          "zeEventPoolCreate");
    ze_event_desc_t event_desc{ZE_STRUCTURE_TYPE_EVENT_DESC};
    event_desc.signal = ZE_EVENT_SCOPE_FLAG_HOST;
    event_desc.wait = ZE_EVENT_SCOPE_FLAG_HOST;
    event_desc.index = 0;
    Check(zeEventCreate(event_pool_, &event_desc, &q8_event_),
          "zeEventCreate(q8)");
    event_desc.index = 1;
    Check(zeEventCreate(event_pool_, &event_desc, &matvec_event_),
          "zeEventCreate(matvec)");
    if (topk_ != 0U) {
      event_desc.index = 2;
      Check(zeEventCreate(event_pool_, &event_desc, &topk_merge_event_),
            "zeEventCreate(topk merge)");
      event_desc.index = 3;
      Check(zeEventCreate(event_pool_, &event_desc, &correction_event_),
            "zeEventCreate(correction)");
    }
    if (token_boundary_ == TokenBoundary::kDeviceFullOutput ||
        token_boundary_ == TokenBoundary::kDeviceCompactDirectTop8) {
      event_desc.index = 4;
      Check(zeEventCreate(
                event_pool_, &event_desc, &boundary_partials_event_),
            "zeEventCreate(boundary partials)");
    }
    if (token_boundary_ != TokenBoundary::kHostFullOutput) {
      event_desc.index = 5;
      Check(zeEventCreate(event_pool_, &event_desc, &boundary_merge_event_),
            "zeEventCreate(boundary merge)");
    }
    module_bytes_ = ReadBinary(module_path);
    ze_module_desc_t module_desc{ZE_STRUCTURE_TYPE_MODULE_DESC};
    module_desc.format = ZE_MODULE_FORMAT_NATIVE;
    module_desc.inputSize = module_bytes_.size();
    module_desc.pInputModule = module_bytes_.data();
    module_desc.pBuildFlags = "";
    ze_module_build_log_handle_t log = nullptr;
    const auto result = zeModuleCreate(
        context_, device_, &module_desc, &module_, &log);
    if (log != nullptr) zeModuleBuildLogDestroy(log);
    Check(result, "zeModuleCreate");
  }

  void* AllocateDevice(std::size_t bytes) {
    ze_device_mem_alloc_desc_t desc{ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC};
    void* pointer = nullptr;
    Check(zeMemAllocDevice(context_, &desc, bytes, 64, device_, &pointer),
          "zeMemAllocDevice");
    device_allocations_.push_back(pointer);
    return pointer;
  }

  void* AllocateShared(std::size_t bytes) {
    ze_device_mem_alloc_desc_t device_desc{
        ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC};
    ze_host_mem_alloc_desc_t host_desc{ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC};
    void* pointer = nullptr;
    Check(zeMemAllocShared(context_, &device_desc, &host_desc, bytes, 64,
                           device_, &pointer),
          "zeMemAllocShared");
    std::memset(pointer, 0, bytes);
    shared_allocations_.push_back(pointer);
    return pointer;
  }

  void* Upload(const void* source, std::size_t bytes) {
    Require(source != nullptr && bytes > 0U, "empty upload");
    void* destination = AllocateDevice(bytes);
    ze_host_mem_alloc_desc_t host_desc{ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC};
    void* staging = nullptr;
    Check(zeMemAllocHost(context_, &host_desc, bytes, 64, &staging),
          "zeMemAllocHost");
    std::memcpy(staging, source, bytes);
    staging_allocations_.push_back(staging);
    Check(zeCommandListAppendMemoryCopy(
              upload_list_, destination, staging, bytes, nullptr, 0, nullptr),
          "zeCommandListAppendMemoryCopy");
    return destination;
  }

  void FinishUploads() {
    Check(zeCommandListClose(upload_list_), "zeCommandListClose(upload)");
    Check(zeCommandQueueExecuteCommandLists(queue_, 1, &upload_list_, nullptr),
          "zeCommandQueueExecuteCommandLists(upload)");
    Check(zeCommandQueueSynchronize(queue_, UINT64_MAX),
          "zeCommandQueueSynchronize(upload)");
    for (void* pointer : staging_allocations_) zeMemFree(context_, pointer);
    staging_allocations_.clear();
  }

  ze_kernel_handle_t CreateKernel(const char* name) {
    ze_kernel_desc_t desc{ZE_STRUCTURE_TYPE_KERNEL_DESC};
    desc.pKernelName = name;
    ze_kernel_handle_t kernel = nullptr;
    Check(zeKernelCreate(module_, &desc, &kernel), "zeKernelCreate");
    return kernel;
  }

  void SetPointer(ze_kernel_handle_t kernel,
                  std::uint32_t index,
                  void* pointer) {
    Check(zeKernelSetArgumentValue(kernel, index, sizeof(pointer), &pointer),
          "zeKernelSetArgumentValue(pointer)");
  }

  template <typename Value>
  void SetValue(ze_kernel_handle_t kernel,
                std::uint32_t index,
                const Value& value) {
    Check(zeKernelSetArgumentValue(kernel, index, sizeof(value), &value),
          "zeKernelSetArgumentValue(value)");
  }

  void Record() {
    const bool compact =
        token_boundary_ == TokenBoundary::kDeviceCompactTop3 ||
        token_boundary_ == TokenBoundary::kDeviceCompactDirectTop8;
    const bool compact_direct =
        token_boundary_ == TokenBoundary::kDeviceCompactDirectTop8;
    q8_kernel_ = CreateKernel(binary_
        ? "iq36_lm_head_q8_group256_f16_sums"
        : "iq36_lm_head_q8_group256_f16");
    matvec_kernel_ = CreateKernel(
        compact
        ? "iq36_lm_head_i8q1_rowstripe8_matvec_local_top3_compact_f32"
        : binary_
        ? (topk_ == 12U
            ? "iq36_lm_head_i8q1_rowstripe8_matvec_local_top12_f32"
            : topk_ == 2U
                ? "iq36_lm_head_i8q1_rowstripe8_matvec_local_top2_f32"
                : "iq36_lm_head_i8q1_rowstripe8_matvec_f32")
        : topk_ == kTopK8
            ? "iq36_lm_head_i8q4_rowstripe8_matvec_topk8_f32"
            : "iq36_lm_head_i8q4_rowstripe8_matvec_f32");
    if (topk_ != 0U) {
      if (compact_direct) {
        topk_merge_kernel_ =
            CreateKernel("iq36_lm_head_compact_top3_merge_top8_f32");
      } else if (!binary_) {
        topk_merge_kernel_ =
            CreateKernel("f32_topk8_merge_blocks_parallel");
      }
      correction_kernel_ = CreateKernel(
          compact
          ? "iq36_lm_head_i8_exact_local_top2_compact_correction_f32"
          : "iq36_lm_head_i8_exact_topk8_correction_f32");
    }
    if (token_boundary_ == TokenBoundary::kDeviceFullOutput) {
      boundary_partials_kernel_ =
          CreateKernel("iq36_lm_head_f32_greedy_partials");
      boundary_merge_kernel_ =
          CreateKernel("iq36_lm_head_greedy_merge64");
    } else if (token_boundary_ == TokenBoundary::kDeviceCompactTop3) {
      boundary_merge_kernel_ =
          CreateKernel("iq36_lm_head_compact_top3_greedy");
    } else if (compact_direct) {
      boundary_partials_kernel_ = CreateKernel(
          "iq36_lm_head_i8_direct_compact_top8_correction_f16");
      boundary_merge_kernel_ =
          CreateKernel("iq36_lm_head_top8_greedy");
    }
    Check(zeKernelSetGroupSize(q8_kernel_, 64, 1, 1),
          "zeKernelSetGroupSize(q8)");
    Check(zeKernelSetGroupSize(matvec_kernel_, 256, 1, 1),
          "zeKernelSetGroupSize(matvec)");
    if (topk_ != 0U) {
      if (!binary_ || compact_direct)
        Check(zeKernelSetGroupSize(topk_merge_kernel_, 256, 1, 1),
              "zeKernelSetGroupSize(topk merge)");
      Check(zeKernelSetGroupSize(correction_kernel_, 64, 1, 1),
            "zeKernelSetGroupSize(correction)");
    }
    if (token_boundary_ == TokenBoundary::kDeviceFullOutput) {
      Check(zeKernelSetGroupSize(boundary_partials_kernel_, 256, 1, 1),
            "zeKernelSetGroupSize(boundary partials)");
      Check(zeKernelSetGroupSize(boundary_merge_kernel_, 64, 1, 1),
            "zeKernelSetGroupSize(boundary merge)");
    } else if (token_boundary_ == TokenBoundary::kDeviceCompactTop3) {
      Check(zeKernelSetGroupSize(boundary_merge_kernel_, 256, 1, 1),
            "zeKernelSetGroupSize(compact boundary merge)");
    } else if (compact_direct) {
      Check(zeKernelSetGroupSize(boundary_partials_kernel_, 64, 1, 1),
            "zeKernelSetGroupSize(direct correction)");
      Check(zeKernelSetGroupSize(boundary_merge_kernel_, 64, 1, 1),
            "zeKernelSetGroupSize(direct boundary merge)");
    }
    SetPointer(q8_kernel_, 0, input_);
    SetPointer(q8_kernel_, 1, q8_);
    SetPointer(q8_kernel_, 2, q8_d_);
    if (binary_) SetPointer(q8_kernel_, 3, q8_sums_);
    SetPointer(matvec_kernel_, 0, packed_);
    SetPointer(matvec_kernel_, 1, q8_);
    SetPointer(matvec_kernel_, 2, q8_d_);
    if (binary_) SetPointer(matvec_kernel_, 3, q8_sums_);
    if (compact) {
      SetValue(matvec_kernel_, 4, kRows);
      SetPointer(matvec_kernel_, 5, partial_top_ids_);
      SetPointer(matvec_kernel_, 6, partial_top_values_);
    } else {
      const std::uint32_t row_argument = binary_ ? 4U : 3U;
      const std::uint32_t output_argument = binary_ ? 5U : 4U;
      SetValue(matvec_kernel_, row_argument, kRows);
      SetPointer(matvec_kernel_, output_argument, output_);
      if (binary_ && topk_ != 0U)
        SetPointer(matvec_kernel_, 6, partial_top_ids_);
    }
    if (topk_ != 0U) {
      if (compact_direct) {
        SetPointer(topk_merge_kernel_, 0, partial_top_ids_);
        SetPointer(topk_merge_kernel_, 1, partial_top_values_);
        SetPointer(topk_merge_kernel_, 2, top_ids_);
        SetPointer(topk_merge_kernel_, 3, top_values_);
      } else if (!binary_) {
        SetPointer(matvec_kernel_, 5, partial_top_ids_);
        SetPointer(matvec_kernel_, 6, partial_top_values_);
        SetPointer(topk_merge_kernel_, 0, partial_top_ids_);
        SetPointer(topk_merge_kernel_, 1, partial_top_values_);
        SetValue(topk_merge_kernel_, 2, kMatvecBlockCount);
        SetValue(topk_merge_kernel_, 3, topk_);
        SetPointer(topk_merge_kernel_, 4, top_ids_);
        SetPointer(topk_merge_kernel_, 5, top_values_);
      }
      SetPointer(correction_kernel_, 0, raw_);
      SetPointer(correction_kernel_, 1, scales_);
      SetPointer(correction_kernel_, 2, q8_);
      SetPointer(correction_kernel_, 3, q8_d_);
      if (compact) {
        SetPointer(correction_kernel_, 4, partial_top_ids_);
        SetPointer(correction_kernel_, 5, partial_top_values_);
      } else {
        SetPointer(correction_kernel_, 4,
                   binary_ ? partial_top_ids_ : top_ids_);
        const std::uint32_t correction_rows = binary_
            ? kMatvecBlockCount * topk_ : topk_;
        SetValue(correction_kernel_, 5, correction_rows);
        SetPointer(correction_kernel_, 6, output_);
      }
    }
    if (token_boundary_ == TokenBoundary::kDeviceFullOutput) {
      SetPointer(boundary_partials_kernel_, 0, output_);
      SetPointer(boundary_partials_kernel_, 1, greedy_partials_);
      SetPointer(boundary_merge_kernel_, 0, greedy_partials_);
      SetPointer(boundary_merge_kernel_, 1, token_);
    } else if (token_boundary_ == TokenBoundary::kDeviceCompactTop3) {
      SetPointer(boundary_merge_kernel_, 0, partial_top_ids_);
      SetPointer(boundary_merge_kernel_, 1, partial_top_values_);
      SetPointer(boundary_merge_kernel_, 2, token_);
    } else if (compact_direct) {
      SetPointer(boundary_partials_kernel_, 0, raw_);
      SetPointer(boundary_partials_kernel_, 1, scales_);
      SetPointer(boundary_partials_kernel_, 2, input_);
      SetPointer(boundary_partials_kernel_, 3, top_ids_);
      SetPointer(boundary_partials_kernel_, 4, top_values_);
      SetPointer(boundary_merge_kernel_, 0, top_ids_);
      SetPointer(boundary_merge_kernel_, 1, top_values_);
      SetPointer(boundary_merge_kernel_, 2, token_);
    }
    ze_group_count_t q8_groups{kColumns / 256U, 1, 1};
    Check(zeCommandListAppendLaunchKernel(
              command_list_, q8_kernel_, &q8_groups, q8_event_, 0, nullptr),
          "zeCommandListAppendLaunchKernel(q8)");
    Check(zeCommandListAppendBarrier(command_list_, nullptr, 0, nullptr),
          "zeCommandListAppendBarrier(q8)");
    ze_group_count_t matvec_groups{kRows / 256U, 1, 1};
    Check(zeCommandListAppendLaunchKernel(
              command_list_, matvec_kernel_, &matvec_groups,
              matvec_event_, 0, nullptr),
          "zeCommandListAppendLaunchKernel(matvec)");
    if (topk_ != 0U) {
      Check(zeCommandListAppendBarrier(command_list_, nullptr, 0, nullptr),
            "zeCommandListAppendBarrier(matvec)");
      if (!binary_ && !compact_direct) {
        ze_group_count_t merge_groups{1U, 1U, 1U};
        Check(zeCommandListAppendLaunchKernel(
                  command_list_, topk_merge_kernel_, &merge_groups,
                  topk_merge_event_, 0, nullptr),
              "zeCommandListAppendLaunchKernel(topk merge)");
        Check(zeCommandListAppendBarrier(command_list_, nullptr, 0, nullptr),
              "zeCommandListAppendBarrier(topk merge)");
      }
      ze_group_count_t correction_groups{
          binary_ ? kMatvecBlockCount * topk_ : topk_, 1U, 1U};
      Check(zeCommandListAppendLaunchKernel(
                command_list_, correction_kernel_, &correction_groups,
                correction_event_, 0, nullptr),
            "zeCommandListAppendLaunchKernel(correction)");
    }
    if (token_boundary_ != TokenBoundary::kHostFullOutput) {
      Check(zeCommandListAppendBarrier(command_list_, nullptr, 0, nullptr),
            "zeCommandListAppendBarrier(correction)");
      if (compact_direct) {
        ze_group_count_t merge_groups{1U, 1U, 1U};
        Check(zeCommandListAppendLaunchKernel(
                  command_list_, topk_merge_kernel_, &merge_groups,
                  topk_merge_event_, 0, nullptr),
              "zeCommandListAppendLaunchKernel(compact topk merge)");
        Check(zeCommandListAppendBarrier(command_list_, nullptr, 0, nullptr),
              "zeCommandListAppendBarrier(compact topk merge)");
      }
      if (token_boundary_ == TokenBoundary::kDeviceFullOutput ||
          compact_direct) {
        ze_group_count_t boundary_partials_groups{
            compact_direct ? kTopK8 : 64U, 1U, 1U};
        Check(zeCommandListAppendLaunchKernel(
                  command_list_, boundary_partials_kernel_,
                  &boundary_partials_groups, boundary_partials_event_,
                  0, nullptr),
              "zeCommandListAppendLaunchKernel(boundary partials)");
        Check(zeCommandListAppendBarrier(command_list_, nullptr, 0, nullptr),
              "zeCommandListAppendBarrier(boundary partials)");
      }
      ze_group_count_t boundary_merge_groups{1U, 1U, 1U};
      Check(zeCommandListAppendLaunchKernel(
                command_list_, boundary_merge_kernel_,
                &boundary_merge_groups, boundary_merge_event_, 0, nullptr),
            "zeCommandListAppendLaunchKernel(boundary merge)");
    }
    Check(zeCommandListClose(command_list_), "zeCommandListClose(run)");
  }

  void Cleanup() {
    if (queue_ != nullptr) zeCommandQueueSynchronize(queue_, UINT64_MAX);
    if (boundary_merge_kernel_ != nullptr)
      zeKernelDestroy(boundary_merge_kernel_);
    if (boundary_partials_kernel_ != nullptr)
      zeKernelDestroy(boundary_partials_kernel_);
    if (correction_kernel_ != nullptr) zeKernelDestroy(correction_kernel_);
    if (topk_merge_kernel_ != nullptr) zeKernelDestroy(topk_merge_kernel_);
    if (matvec_kernel_ != nullptr) zeKernelDestroy(matvec_kernel_);
    if (q8_kernel_ != nullptr) zeKernelDestroy(q8_kernel_);
    if (boundary_merge_event_ != nullptr)
      zeEventDestroy(boundary_merge_event_);
    if (boundary_partials_event_ != nullptr)
      zeEventDestroy(boundary_partials_event_);
    if (correction_event_ != nullptr) zeEventDestroy(correction_event_);
    if (topk_merge_event_ != nullptr) zeEventDestroy(topk_merge_event_);
    if (matvec_event_ != nullptr) zeEventDestroy(matvec_event_);
    if (q8_event_ != nullptr) zeEventDestroy(q8_event_);
    if (event_pool_ != nullptr) zeEventPoolDestroy(event_pool_);
    if (command_list_ != nullptr) zeCommandListDestroy(command_list_);
    if (upload_list_ != nullptr) zeCommandListDestroy(upload_list_);
    if (module_ != nullptr) zeModuleDestroy(module_);
    for (void* pointer : shared_allocations_) zeMemFree(context_, pointer);
    for (void* pointer : device_allocations_) zeMemFree(context_, pointer);
    for (void* pointer : staging_allocations_) zeMemFree(context_, pointer);
    if (queue_ != nullptr) zeCommandQueueDestroy(queue_);
    if (context_ != nullptr) zeContextDestroy(context_);
  }

  ze_driver_handle_t driver_ = nullptr;
  ze_device_handle_t device_ = nullptr;
  ze_context_handle_t context_ = nullptr;
  ze_command_queue_handle_t queue_ = nullptr;
  ze_command_list_handle_t upload_list_ = nullptr;
  ze_command_list_handle_t command_list_ = nullptr;
  ze_module_handle_t module_ = nullptr;
  ze_event_pool_handle_t event_pool_ = nullptr;
  ze_event_handle_t q8_event_ = nullptr;
  ze_event_handle_t matvec_event_ = nullptr;
  ze_event_handle_t topk_merge_event_ = nullptr;
  ze_event_handle_t correction_event_ = nullptr;
  ze_event_handle_t boundary_partials_event_ = nullptr;
  ze_event_handle_t boundary_merge_event_ = nullptr;
  ze_kernel_handle_t q8_kernel_ = nullptr;
  ze_kernel_handle_t matvec_kernel_ = nullptr;
  ze_kernel_handle_t topk_merge_kernel_ = nullptr;
  ze_kernel_handle_t correction_kernel_ = nullptr;
  ze_kernel_handle_t boundary_partials_kernel_ = nullptr;
  ze_kernel_handle_t boundary_merge_kernel_ = nullptr;
  ze_device_properties_t properties_{ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES};
  std::vector<std::uint8_t> module_bytes_;
  std::vector<void*> device_allocations_;
  std::vector<void*> shared_allocations_;
  std::vector<void*> staging_allocations_;
  void* packed_ = nullptr;
  void* raw_ = nullptr;
  void* scales_ = nullptr;
  void* partial_top_ids_ = nullptr;
  void* partial_top_values_ = nullptr;
  std::int32_t* top_ids_ = nullptr;
  float* top_values_ = nullptr;
  float* input_ = nullptr;
  void* q8_ = nullptr;
  void* q8_d_ = nullptr;
  void* q8_sums_ = nullptr;
  void* greedy_partials_ = nullptr;
  float* output_ = nullptr;
  std::int32_t* token_ = nullptr;
  std::uint64_t packed_bytes_ = 0;
  std::uint32_t topk_ = 0;
  bool binary_ = false;
  TokenBoundary token_boundary_ = TokenBoundary::kHostFullOutput;
  std::uint32_t queue_ordinal_ = UINT32_MAX;
  std::string device_name_;
  double timestamp_ns_per_tick_ = 0.0;
};

struct PhaseResult {
  int phase = 0;
  Runtime::Run timing;
  std::filesystem::path output;
  std::vector<std::int32_t> selected_ids;
  bool finite = false;
};

double SingleWallUs(const Runtime::Run& run) {
  Require(run.wall_us.size() == 1U, "expected one wall timing sample");
  return run.wall_us[0];
}

double SingleShellUs(const Runtime::Run& run) {
  Require(run.q8_us.size() == 1U && run.matvec_us.size() == 1U &&
              run.topk_merge_us.size() == 1U &&
              run.correction_us.size() == 1U &&
              run.boundary_partials_us.size() == 1U &&
              run.boundary_merge_us.size() == 1U,
          "expected one kernel timing sample");
  return run.q8_us[0] + run.matvec_us[0] +
      run.topk_merge_us[0] + run.correction_us[0] +
      run.boundary_partials_us[0] + run.boundary_merge_us[0];
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 7 && argc != 8) {
      throw std::invalid_argument(
          "usage: iq36-openvino-lm-head-i8q4-component MODEL_BIN MODULE "
          "HIDDEN_DIR OUTPUT_DIR WARMUP SAMPLES [TOPK]");
    }
    const std::string model_bin = argv[1];
    const std::string module = argv[2];
    const std::filesystem::path hidden_dir = argv[3];
    const std::filesystem::path output_dir = argv[4];
    const int warmup = std::stoi(argv[5]);
    const int samples = std::stoi(argv[6]);
    const std::uint32_t topk = argc == 8
        ? static_cast<std::uint32_t>(std::stoul(argv[7]))
        : 0U;
    const char* binary_value = std::getenv("IQ36_LM_HEAD_BINARY");
    const bool binary = binary_value != nullptr &&
        std::string(binary_value) == "1";
    const char* compare_value =
        std::getenv("IQ36_LM_HEAD_COMPARE_LOCAL2");
    const bool compare_local2 = compare_value != nullptr &&
        std::string(compare_value) == "1";
    const char* compare_token_value =
        std::getenv("IQ36_LM_HEAD_COMPARE_TOKEN_ONLY");
    const bool compare_token_only = compare_token_value != nullptr &&
        std::string(compare_token_value) == "1";
    const char* compare_direct_value =
        std::getenv("IQ36_LM_HEAD_COMPARE_COMPACT_DIRECT");
    const bool compare_direct = compare_direct_value != nullptr &&
        std::string(compare_direct_value) == "1";
    const char* probe_direct_value =
        std::getenv("IQ36_LM_HEAD_PROBE_COMPACT_DIRECT");
    const bool probe_direct = probe_direct_value != nullptr &&
        std::string(probe_direct_value) == "1";
    Require(static_cast<int>(compare_local2) +
                static_cast<int>(compare_token_only) +
                static_cast<int>(compare_direct) +
                static_cast<int>(probe_direct) <= 1,
            "select only one LM-head comparison mode");
    const char* first_phase_value = std::getenv("IQ36_LM_HEAD_FIRST_PHASE");
    const int first_phase = first_phase_value != nullptr
        ? std::stoi(first_phase_value) : kFirstDecodePhase;
    const char* last_phase_value = std::getenv("IQ36_LM_HEAD_LAST_PHASE");
    const int last_phase = last_phase_value != nullptr
        ? std::stoi(last_phase_value) : kLastDecodePhase;
    Require(first_phase >= 0 && last_phase >= first_phase &&
                last_phase <= 4096,
            "LM-head phase range must satisfy 0 <= first <= last <= 4096");
    std::filesystem::create_directories(output_dir);

    const auto pack_started = std::chrono::steady_clock::now();
    auto raw = ReadRange(model_bin, kWeightOffset, kWeightBytes);
    auto scales = ReadRange(model_bin, kScaleOffset, kScaleBytes);
    auto packed = binary
        ? PackI8Q1Rowstripe8(raw, scales)
        : PackI8Q4Rowstripe8(raw, scales);
    const auto pack_finished = std::chrono::steady_clock::now();
    const double pack_ms = std::chrono::duration<double, std::milli>(
        pack_finished - pack_started).count();

    if (compare_local2 || compare_token_only ||
        compare_direct || probe_direct) {
      Require(binary && topk == 0U,
              "token comparison requires binary mode and no TOPK argument");
      if (probe_direct) {
        Require(first_phase == last_phase && samples == 20,
                "direct token probe requires one phase and 20 ABBA blocks");
      } else {
        Require(first_phase == 0 && last_phase == 63 && samples == 20,
                "token comparison requires phases 0..63 and 20 ABBA blocks");
      }
      const char* expected_token_value =
          std::getenv("IQ36_LM_HEAD_EXPECT_TOKEN");
      Require(!probe_direct || expected_token_value != nullptr,
              "direct token probe requires IQ36_LM_HEAD_EXPECT_TOKEN");
      const std::int32_t expected_token = probe_direct
          ? static_cast<std::int32_t>(std::stoi(expected_token_value))
          : -1;
      const bool compact_direct = compare_direct || probe_direct;
      Runtime control(
          module, packed, raw, scales, compare_local2 ? 12U : 2U, true,
          compare_token_only ? TokenBoundary::kDeviceFullOutput
          : compact_direct ? TokenBoundary::kDeviceCompactTop3
                             : TokenBoundary::kHostFullOutput);
      Runtime candidate(
          module, packed, raw, scales, 2U, true,
          compact_direct ? TokenBoundary::kDeviceCompactDirectTop8
          : compare_token_only ? TokenBoundary::kDeviceCompactTop3
                             : TokenBoundary::kHostFullOutput);
      packed.clear();
      packed.shrink_to_fit();
      raw.clear();
      raw.shrink_to_fit();
      scales.clear();
      scales.shrink_to_fit();

      std::vector<std::vector<float>> inputs;
      std::vector<std::int32_t> reference_tokens;
      std::vector<std::int32_t> control_tokens;
      std::vector<std::int32_t> candidate_tokens;
      const std::size_t phase_count =
          static_cast<std::size_t>(last_phase - first_phase + 1);
      inputs.reserve(phase_count);
      reference_tokens.reserve(phase_count);
      control_tokens.reserve(phase_count);
      candidate_tokens.reserve(phase_count);
      for (int phase = first_phase; phase <= last_phase; ++phase) {
        std::ostringstream stem;
        stem << "step" << std::setfill('0') << std::setw(4) << phase;
        inputs.push_back(ReadF32(
            hidden_dir / (stem.str() + "-lm-head-input.f32"), kColumns));
        if (probe_direct) {
          reference_tokens.push_back(expected_token);
        } else {
          const auto reference = ReadF32(
              hidden_dir / (stem.str() + "-logits.f32"), kRows);
          reference_tokens.push_back(static_cast<std::int32_t>(
              std::distance(
                  reference.begin(),
                  std::max_element(reference.begin(), reference.end()))));
        }
      }
      control.Execute(inputs[0], warmup, 1);
      candidate.Execute(inputs[0], warmup, 1);
      for (const auto& input : inputs) {
        control.Execute(input, 0, 1);
        candidate.Execute(input, 0, 1);
        control_tokens.push_back(control.GreedyToken());
        candidate_tokens.push_back(candidate.GreedyToken());
      }

      struct Block {
        int block = 0;
        int phase = 0;
        double control_wall_us = 0.0;
        double candidate_wall_us = 0.0;
        double saving_wall_us = 0.0;
        double control_shell_us = 0.0;
        double candidate_shell_us = 0.0;
        double saving_shell_us = 0.0;
      };
      std::vector<Block> blocks;
      blocks.reserve(static_cast<std::size_t>(samples));
      for (int block = 0; block < samples; ++block) {
        const int phase = block % static_cast<int>(inputs.size());
        const auto a1 = control.Execute(inputs[phase], 0, 1);
        const auto b1 = candidate.Execute(inputs[phase], 0, 1);
        const auto b2 = candidate.Execute(inputs[phase], 0, 1);
        const auto a2 = control.Execute(inputs[phase], 0, 1);
        const double control_wall =
            (SingleWallUs(a1) + SingleWallUs(a2)) * 0.5;
        const double candidate_wall =
            (SingleWallUs(b1) + SingleWallUs(b2)) * 0.5;
        const double control_shell =
            (SingleShellUs(a1) + SingleShellUs(a2)) * 0.5;
        const double candidate_shell =
            (SingleShellUs(b1) + SingleShellUs(b2)) * 0.5;
        blocks.push_back({block, phase, control_wall, candidate_wall,
                          control_wall - candidate_wall, control_shell,
                          candidate_shell, control_shell - candidate_shell});
      }
      const bool control_exact = control_tokens == reference_tokens;
      const bool candidate_exact = candidate_tokens == reference_tokens;
      const bool pass = (control_exact || probe_direct) && candidate_exact &&
          control.device_name().find("B390") != std::string::npos &&
          candidate.device_name() == control.device_name();
      std::cout << std::boolalpha << std::setprecision(12) << "{"
                << "\"mode\":\""
                << (probe_direct
                    ? "compact_top3_vs_compact_direct_top8_probe"
                    : compare_direct
                        ? "compact_top3_vs_compact_direct_top8"
                        : compare_token_only
                            ? "binary_local2_full_device_vs_compact_top3"
                            : "binary_local12_vs_token_local2")
                << "\","
                << "\"device_name\":\"" << control.device_name() << "\","
                << "\"pack_ms\":" << pack_ms << ","
                << "\"reference_tokens\":";
      WriteIntArray(reference_tokens);
      std::cout << ",\"control_tokens\":";
      WriteIntArray(control_tokens);
      std::cout << ",\"candidate_tokens\":";
      WriteIntArray(candidate_tokens);
      std::cout << ",\"control_tokens_exact\":" << control_exact
                << ",\"candidate_tokens_exact\":" << candidate_exact
                << ",\"blocks\":[";
      for (std::size_t index = 0; index < blocks.size(); ++index) {
        if (index != 0U) std::cout << ",";
        const auto& block = blocks[index];
        std::cout << "{\"block\":" << block.block
                  << ",\"phase\":" << block.phase
                  << ",\"control_wall_us\":" << block.control_wall_us
                  << ",\"candidate_wall_us\":" << block.candidate_wall_us
                  << ",\"saving_wall_us\":" << block.saving_wall_us
                  << ",\"control_shell_us\":" << block.control_shell_us
                  << ",\"candidate_shell_us\":" << block.candidate_shell_us
                  << ",\"saving_shell_us\":" << block.saving_shell_us
                  << "}";
      }
      std::cout << "],\"required_checks_passed\":" << pass << "}"
                << std::endl;
      return pass ? 0 : 2;
    }

    Runtime runtime(module, packed, raw, scales, topk, binary);
    packed.clear();
    packed.shrink_to_fit();
    raw.clear();
    raw.shrink_to_fit();
    scales.clear();
    scales.shrink_to_fit();
    std::vector<PhaseResult> phases;
    bool all_finite = true;
    for (int phase = first_phase; phase <= last_phase; ++phase) {
      auto input_path = hidden_dir /
          ("phase" + std::to_string(phase) + "-lm-head-input.f32");
      if (!std::filesystem::exists(input_path)) {
        std::ostringstream step_name;
        step_name << "step" << std::setfill('0') << std::setw(4) << phase
                  << "-lm-head-input.f32";
        input_path = hidden_dir / step_name.str();
      }
      const auto output_path = output_dir /
          ("phase" + std::to_string(phase) + "-logits.f32");
      const auto input = ReadF32(input_path, kColumns);
      auto timing = runtime.Execute(input, warmup, samples);
      const auto output = runtime.Output();
      const auto selected_ids = runtime.SelectedIds();
      const bool finite = std::all_of(
          output.begin(), output.end(),
          [](float value) { return std::isfinite(value); });
      WriteF32(output_path, output);
      all_finite = all_finite && finite;
      phases.push_back({phase, std::move(timing), output_path,
                        selected_ids, finite});
    }
    const std::uint64_t expected_packed_bytes =
        static_cast<std::uint64_t>(kRows / kRowsPerStripe) *
        (binary ? kBinaryStripeBytes : kStripeBytes);
    const bool pass = runtime.device_name().find("B390") != std::string::npos &&
        runtime.packed_bytes() == expected_packed_bytes && all_finite &&
        std::all_of(phases.begin(), phases.end(), [](const PhaseResult& row) {
          return Minimum(row.timing.q8_us) > 0.0 &&
              Minimum(row.timing.matvec_us) > 0.0 &&
              Minimum(row.timing.topk_merge_us) >= 0.0 &&
              Minimum(row.timing.correction_us) >= 0.0;
        });

    std::cout << std::boolalpha << std::setprecision(12) << "{"
              << "\"device_name\":\"" << runtime.device_name() << "\","
              << "\"packed_bytes\":" << runtime.packed_bytes() << ","
              << "\"topk\":" << runtime.topk() << ","
              << "\"binary\":" << binary << ","
              << "\"pack_ms\":" << pack_ms << ","
              << "\"timestamp_ns_per_tick\":"
              << runtime.timestamp_ns_per_tick() << ","
              << "\"phases\":[";
    for (std::size_t index = 0; index < phases.size(); ++index) {
      if (index != 0U) std::cout << ",";
      const auto& row = phases[index];
      const auto shell = AddSamples(
          AddSamples(row.timing.q8_us, row.timing.matvec_us),
          AddSamples(row.timing.topk_merge_us, row.timing.correction_us));
      const double matvec_min = Minimum(row.timing.matvec_us);
      std::cout << "{\"phase\":" << row.phase
                << ",\"finite\":" << row.finite
                << ",\"output\":\"" << row.output.string() << "\""
                << ",\"q8_min_us\":" << Minimum(row.timing.q8_us)
                << ",\"q8_mean_us\":" << Mean(row.timing.q8_us)
                << ",\"matvec_min_us\":" << matvec_min
                << ",\"matvec_mean_us\":" << Mean(row.timing.matvec_us)
                << ",\"topk_merge_min_us\":"
                << Minimum(row.timing.topk_merge_us)
                << ",\"topk_merge_mean_us\":"
                << Mean(row.timing.topk_merge_us)
                << ",\"correction_min_us\":"
                << Minimum(row.timing.correction_us)
                << ",\"correction_mean_us\":"
                << Mean(row.timing.correction_us)
                << ",\"shell_min_us\":" << Minimum(shell)
                << ",\"shell_mean_us\":" << Mean(shell)
                << ",\"wall_min_us\":" << Minimum(row.timing.wall_us)
                << ",\"wall_mean_us\":" << Mean(row.timing.wall_us)
                << ",\"effective_packed_gb_s\":"
                << runtime.packed_bytes() / (matvec_min * 1000.0)
                << ",\"selected_ids\":";
      WriteIntArray(row.selected_ids);
      std::cout
                << ",\"q8_samples_us\":";
      WriteDoubleArray(row.timing.q8_us);
      std::cout << ",\"matvec_samples_us\":";
      WriteDoubleArray(row.timing.matvec_us);
      std::cout << ",\"topk_merge_samples_us\":";
      WriteDoubleArray(row.timing.topk_merge_us);
      std::cout << ",\"correction_samples_us\":";
      WriteDoubleArray(row.timing.correction_us);
      std::cout << ",\"shell_samples_us\":";
      WriteDoubleArray(shell);
      std::cout << "}";
    }
    std::cout << "],\"required_checks_passed\":" << pass << "}"
              << std::endl;
    return pass ? 0 : 2;
  } catch (const std::exception& exception) {
    std::cerr << "iq36-openvino-lm-head-i8q4-component: "
              << exception.what() << '\n';
    return 4;
  }
}
