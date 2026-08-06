#include <level_zero/ze_api.h>

#include <algorithm>
#include <cfenv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr std::uint32_t kIntelVendorId = 0x8086U;
constexpr std::uint32_t kPtlDeviceId = 0xB080U;
constexpr std::uint32_t kRows = 248320U;
constexpr std::uint32_t kColumns = 2048U;
constexpr std::uint32_t kGroups = 16U;
constexpr std::uint32_t kGroupSize = 128U;
constexpr std::uint32_t kCapacity = 16812U;
constexpr std::uint32_t kScanWorkgroups = 384U;
constexpr std::uint32_t kViolationWorkgroups = (kRows + 255U) / 256U;
constexpr std::uint64_t kWeightOffset = UINT64_C(18137149498);
constexpr std::uint64_t kWeightBytes =
    static_cast<std::uint64_t>(kRows) * kColumns;
constexpr std::uint64_t kScaleOffset = UINT64_C(18645708858);
constexpr std::uint64_t kScaleBytes =
    static_cast<std::uint64_t>(kRows) * sizeof(std::uint16_t);
constexpr std::uint64_t kCodeBytes = kWeightBytes / 2U;
constexpr std::uint64_t kMinMaxBytes =
    static_cast<std::uint64_t>(kRows) * kGroups * 2U;
constexpr std::uint64_t kResidualBytes =
    static_cast<std::uint64_t>(kRows) * kGroups * sizeof(std::uint16_t);
constexpr std::size_t kEventCount = 2000U;

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
  Require(static_cast<bool>(input), "failed to open model binary " + path);
  input.seekg(static_cast<std::streamoff>(offset));
  Require(static_cast<bool>(input), "failed to seek model binary");
  std::vector<std::uint8_t> values(static_cast<std::size_t>(bytes));
  input.read(reinterpret_cast<char*>(values.data()),
             static_cast<std::streamsize>(values.size()));
  Require(input.gcount() == static_cast<std::streamsize>(values.size()),
          "failed to read complete model range");
  return values;
}

std::vector<std::uint8_t> ReadBinary(const std::string& path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  Require(static_cast<bool>(input), "failed to open " + path);
  const auto size = input.tellg();
  Require(size > 0, "empty input " + path);
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
  input.seekg(0, std::ios::beg);
  input.read(reinterpret_cast<char*>(bytes.data()), size);
  Require(input.gcount() == size, "short read " + path);
  return bytes;
}

template <typename T>
std::vector<T> ReadTyped(
    const std::filesystem::path& path, std::size_t count) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  Require(static_cast<bool>(input), "failed to open " + path.string());
  const auto bytes = input.tellg();
  Require(bytes == static_cast<std::streamoff>(count * sizeof(T)),
          "input size mismatch " + path.string());
  std::vector<T> values(count);
  input.seekg(0, std::ios::beg);
  input.read(reinterpret_cast<char*>(values.data()), bytes);
  Require(input.gcount() == bytes, "short read " + path.string());
  return values;
}

std::uint16_t FloatToHalf(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const std::uint16_t sign =
      static_cast<std::uint16_t>((bits >> 16U) & 0x8000U);
  const std::uint32_t exponent = (bits >> 23U) & 0xFFU;
  const std::uint32_t mantissa = bits & 0x7FFFFFU;
  if (exponent == 0xFFU) {
    if (mantissa == 0U) return static_cast<std::uint16_t>(sign | 0x7C00U);
    return static_cast<std::uint16_t>(
        sign | 0x7C00U | std::max(1U, mantissa >> 13U));
  }
  const int half_exponent = static_cast<int>(exponent) - 127 + 15;
  if (half_exponent >= 31)
    return static_cast<std::uint16_t>(sign | 0x7C00U);
  if (half_exponent <= 0) {
    if (half_exponent < -10) return sign;
    const std::uint32_t normalized = mantissa | 0x800000U;
    const unsigned int shift =
        static_cast<unsigned int>(14 - half_exponent);
    const std::uint32_t halfway = UINT32_C(1) << (shift - 1U);
    const std::uint32_t rounded =
        (normalized + halfway - 1U + ((normalized >> shift) & 1U)) >> shift;
    return static_cast<std::uint16_t>(sign | rounded);
  }
  std::uint32_t rounded = mantissa + 0xFFFU + ((mantissa >> 13U) & 1U);
  std::uint32_t adjusted_exponent =
      static_cast<std::uint32_t>(half_exponent);
  if ((rounded & 0x800000U) != 0U) {
    rounded = 0U;
    ++adjusted_exponent;
    if (adjusted_exponent >= 31U)
      return static_cast<std::uint16_t>(sign | 0x7C00U);
  }
  return static_cast<std::uint16_t>(
      sign | (adjusted_exponent << 10U) | (rounded >> 13U));
}

float HalfToFloat(std::uint16_t value) {
  const std::uint32_t sign =
      static_cast<std::uint32_t>(value & 0x8000U) << 16U;
  std::uint32_t exponent = (value >> 10U) & 0x1FU;
  std::uint32_t mantissa = value & 0x3FFU;
  std::uint32_t bits = 0;
  if (exponent == 0U) {
    if (mantissa == 0U) {
      bits = sign;
    } else {
      int unbiased = -14;
      while ((mantissa & 0x400U) == 0U) {
        mantissa <<= 1U;
        --unbiased;
      }
      mantissa &= 0x3FFU;
      bits = sign |
          (static_cast<std::uint32_t>(unbiased + 127) << 23U) |
          (mantissa << 13U);
    }
  } else if (exponent == 0x1FU) {
    bits = sign | 0x7F800000U | (mantissa << 13U);
  } else {
    exponent = exponent - 15U + 127U;
    bits = sign | (exponent << 23U) | (mantissa << 13U);
  }
  float result = 0.0f;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

std::uint16_t PositiveF16UpperAfterF32(double value) {
  float rounded = static_cast<float>(value);
  rounded = std::nextafter(rounded, std::numeric_limits<float>::infinity());
  std::uint16_t half = FloatToHalf(rounded);
  if (static_cast<double>(HalfToFloat(half)) < static_cast<double>(rounded))
    ++half;
  return half;
}

std::uint64_t TimestampDelta(
    std::uint64_t start, std::uint64_t end, std::uint32_t valid_bits) {
  if (valid_bits == 0U || valid_bits >= 64U) return end - start;
  const std::uint64_t mask = (UINT64_C(1) << valid_bits) - 1U;
  return (end - start) & mask;
}

struct PackedWeights {
  std::vector<std::uint8_t> codes;
  std::vector<std::uint8_t> minmax;
  std::vector<std::uint16_t> residual_norms;
  double seconds = 0.0;
};

PackedWeights PackAffineQ4(const std::vector<std::uint8_t>& bytes) {
  Require(bytes.size() == kWeightBytes, "weight size mismatch");
  Require(std::fesetround(FE_TONEAREST) == 0, "cannot select ties-to-even");
  const auto begin_time = std::chrono::steady_clock::now();
  PackedWeights packed;
  packed.codes.resize(static_cast<std::size_t>(kCodeBytes));
  packed.minmax.resize(static_cast<std::size_t>(kMinMaxBytes));
  packed.residual_norms.resize(
      static_cast<std::size_t>(kRows) * kGroups);
  const auto* weights =
      reinterpret_cast<const std::int8_t*>(bytes.data());
  constexpr double gamma =
      (static_cast<double>(kGroupSize) * 0x1.0p-53) /
      (1.0 - static_cast<double>(kGroupSize) * 0x1.0p-53);
  for (std::uint32_t row = 0; row < kRows; ++row) {
    const std::size_t row_base =
        static_cast<std::size_t>(row) * kColumns;
    const std::size_t code_row =
        static_cast<std::size_t>(row) * (kColumns / 2U);
    for (std::uint32_t group = 0; group < kGroups; ++group) {
      const std::size_t group_base =
          row_base + static_cast<std::size_t>(group) * kGroupSize;
      std::int8_t minimum = weights[group_base];
      std::int8_t maximum = weights[group_base];
      for (std::uint32_t index = 1; index < kGroupSize; ++index) {
        minimum = std::min(minimum, weights[group_base + index]);
        maximum = std::max(maximum, weights[group_base + index]);
      }
      const std::size_t metadata =
          (static_cast<std::size_t>(row) * kGroups + group) * 2U;
      packed.minmax[metadata] =
          static_cast<std::uint8_t>(minimum);
      packed.minmax[metadata + 1U] =
          static_cast<std::uint8_t>(maximum);
      const float step =
          (static_cast<float>(maximum) - static_cast<float>(minimum)) /
          15.0f;
      double residual_square = 0.0;
      for (std::uint32_t index = 0; index < kGroupSize; ++index) {
        const float raw = static_cast<float>(weights[group_base + index]);
        const float scaled = step == 0.0f
            ? 0.0f
            : (raw - static_cast<float>(minimum)) / step;
        int code = static_cast<int>(std::nearbyint(scaled));
        code = std::max(0, std::min(15, code));
        const std::size_t column =
            static_cast<std::size_t>(group) * kGroupSize + index;
        const std::size_t destination = code_row + column / 2U;
        if ((column & 1U) == 0U) {
          packed.codes[destination] = static_cast<std::uint8_t>(code);
        } else {
          packed.codes[destination] |=
              static_cast<std::uint8_t>(code << 4U);
        }
        const float codec = static_cast<float>(minimum) +
            static_cast<float>(code) * step;
        const double residual =
            static_cast<double>(raw) - static_cast<double>(codec);
        residual_square += residual * residual;
      }
      const double norm = std::sqrt(residual_square * (1.0 + gamma));
      packed.residual_norms[
          static_cast<std::size_t>(row) * kGroups + group] =
          PositiveF16UpperAfterF32(norm);
    }
  }
  packed.seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - begin_time).count();
  return packed;
}

struct TimedRun {
  std::vector<double> stages_us;
  double kernel_us = 0.0;
  double wall_us = 0.0;
};

struct CorrectnessRow {
  std::uint32_t candidate_count = 0;
  std::uint32_t violation_count = 0;
  std::int32_t candidate_token = -1;
  std::int32_t reference_token = -1;
};

struct PairRow {
  std::int32_t event = -1;
  std::int32_t block = -1;
  double baseline_kernel_us = 0.0;
  double candidate_kernel_us = 0.0;
  double saving_kernel_us = 0.0;
  double baseline_wall_us = 0.0;
  double candidate_wall_us = 0.0;
  double saving_wall_us = 0.0;
  std::vector<double> baseline_stages_us;
  std::vector<double> candidate_stages_us;
};

class Runtime {
 public:
  Runtime(
      const std::string& module_path,
      const std::vector<std::uint8_t>& weights,
      const std::vector<std::uint8_t>& scales,
      const PackedWeights& packed)
      : host_weights_(weights), host_scales_(scales) {
    Require(weights.size() == kWeightBytes, "weight byte size mismatch");
    Require(scales.size() == kScaleBytes, "scale byte size mismatch");
    Require(packed.codes.size() == kCodeBytes, "code byte size mismatch");
    Require(packed.minmax.size() == kMinMaxBytes, "minmax size mismatch");
    Require(
        packed.residual_norms.size() * sizeof(std::uint16_t) ==
            kResidualBytes,
        "residual norm size mismatch");
    InitializeDevice();
    InitializeRuntime(module_path);
    weights_ = Upload(weights.data(), weights.size());
    scales_ = Upload(scales.data(), scales.size());
    codes_ = Upload(packed.codes.data(), packed.codes.size());
    minmax_ = Upload(packed.minmax.data(), packed.minmax.size());
    residual_norms_ = Upload(
        packed.residual_norms.data(),
        packed.residual_norms.size() * sizeof(std::uint16_t));
    input_ = static_cast<std::uint16_t*>(
        AllocateShared(kColumns * sizeof(std::uint16_t)));
    seed_id_ = static_cast<std::int32_t*>(
        AllocateShared(sizeof(std::int32_t)));
    seed_value_ = static_cast<std::uint16_t*>(
        AllocateShared(sizeof(std::uint16_t)));
    candidate_count_ = static_cast<std::uint32_t*>(
        AllocateShared(sizeof(std::uint32_t)));
    candidate_token_ = static_cast<std::int32_t*>(
        AllocateShared(sizeof(std::int32_t)));
    baseline_token_ = static_cast<std::int32_t*>(
        AllocateShared(sizeof(std::int32_t)));
    reference_token_ = static_cast<std::int32_t*>(
        AllocateShared(sizeof(std::int32_t)));
    violation_count_ = static_cast<std::uint32_t*>(
        AllocateShared(sizeof(std::uint32_t)));
    q8_ = AllocateDevice(kColumns * sizeof(std::int8_t));
    q8_d_ = AllocateDevice(8U * sizeof(float));
    hidden_norms_ = AllocateDevice(kGroups * sizeof(std::uint16_t));
    hidden_delta_norms_ = AllocateDevice(kGroups * sizeof(std::uint16_t));
    upper_output_ = AllocateDevice(kRows * sizeof(std::uint16_t));
    candidate_ids_ = AllocateDevice(kCapacity * sizeof(std::int32_t));
    candidate_values_ = AllocateDevice(kCapacity * sizeof(std::uint16_t));
    baseline_output_ = AllocateDevice(kRows * sizeof(std::uint16_t));
    reference_output_ = AllocateDevice(kRows * sizeof(std::uint16_t));
    FinishUploads();
    Record();
  }

  ~Runtime() { Cleanup(); }

  CorrectnessRow CheckEvent(
      const std::uint16_t* hidden, std::int32_t seed_id) {
    Prepare(hidden, seed_id, SeedValue(hidden, seed_id));
    RunTimed(candidate_list_, 5U);
    RunUntimed(reference_list_);
    CorrectnessRow result{
        *candidate_count_, *violation_count_,
        *candidate_token_, *reference_token_};
    if (result.candidate_count > kCapacity)
      result.candidate_token = result.reference_token;
    return result;
  }

  CorrectnessRow CheckForcedOverflow(
      const std::uint16_t* hidden, std::int32_t seed_id) {
    Prepare(hidden, seed_id, UINT16_C(0xFC00));
    RunTimed(candidate_list_, 5U);
    RunUntimed(reference_list_);
    CorrectnessRow result{
        *candidate_count_, *violation_count_,
        *candidate_token_, *reference_token_};
    if (result.candidate_count > kCapacity)
      result.candidate_token = result.reference_token;
    return result;
  }

  std::vector<PairRow> Compare(
      const std::uint16_t* hidden, std::int32_t seed_id,
      std::int32_t event, int warmup, int blocks) {
    Prepare(hidden, seed_id, SeedValue(hidden, seed_id));
    for (int index = 0; index < warmup; ++index) {
      RunTimed(baseline_list_, 2U);
      RunTimed(candidate_list_, 5U);
    }
    std::vector<PairRow> rows;
    rows.reserve(static_cast<std::size_t>(blocks));
    for (int block = 0; block < blocks; ++block) {
      const TimedRun a1 = RunTimed(baseline_list_, 2U);
      const TimedRun b1 = RunTimed(candidate_list_, 5U);
      const TimedRun b2 = RunTimed(candidate_list_, 5U);
      const TimedRun a2 = RunTimed(baseline_list_, 2U);
      PairRow row;
      row.event = event;
      row.block = block;
      row.baseline_kernel_us = (a1.kernel_us + a2.kernel_us) * 0.5;
      row.candidate_kernel_us = (b1.kernel_us + b2.kernel_us) * 0.5;
      row.saving_kernel_us =
          row.baseline_kernel_us - row.candidate_kernel_us;
      row.baseline_wall_us = (a1.wall_us + a2.wall_us) * 0.5;
      row.candidate_wall_us = (b1.wall_us + b2.wall_us) * 0.5;
      row.saving_wall_us =
          row.baseline_wall_us - row.candidate_wall_us;
      row.baseline_stages_us.resize(2U);
      row.candidate_stages_us.resize(5U);
      for (std::size_t stage = 0; stage < 2U; ++stage) {
        row.baseline_stages_us[stage] =
            (a1.stages_us[stage] + a2.stages_us[stage]) * 0.5;
      }
      for (std::size_t stage = 0; stage < 5U; ++stage) {
        row.candidate_stages_us[stage] =
            (b1.stages_us[stage] + b2.stages_us[stage]) * 0.5;
      }
      rows.push_back(std::move(row));
    }
    return rows;
  }

  const std::string& device_name() const { return device_name_; }
  double timestamp_ns_per_tick() const { return timestamp_ns_per_tick_; }
  const ze_kernel_properties_t& select_properties() const {
    return select_properties_;
  }
  const ze_kernel_properties_t& exact_properties() const {
    return exact_properties_;
  }

 private:
  std::uint16_t SeedValue(
      const std::uint16_t* hidden, std::int32_t seed_id) const {
    Require(seed_id >= 0 && seed_id < static_cast<std::int32_t>(kRows),
            "seed id out of range");
    const auto* weights =
        reinterpret_cast<const std::int8_t*>(host_weights_.data());
    const auto* scales =
        reinterpret_cast<const std::uint16_t*>(host_scales_.data());
    const std::size_t row_base =
        static_cast<std::size_t>(seed_id) * kColumns;
    double value = 0.0;
    for (std::uint32_t column = 0; column < kColumns; ++column) {
      value += static_cast<double>(weights[row_base + column]) *
          static_cast<double>(HalfToFloat(hidden[column]));
    }
    value *= static_cast<double>(HalfToFloat(scales[seed_id]));
    return FloatToHalf(static_cast<float>(value));
  }

  void Prepare(
      const std::uint16_t* hidden, std::int32_t seed_id,
      std::uint16_t seed_value) {
    std::memcpy(
        input_, hidden, kColumns * sizeof(std::uint16_t));
    *seed_id_ = seed_id;
    *seed_value_ = seed_value;
    RunTimed(q8_list_, 1U);
  }

  TimedRun RunTimed(
      ze_command_list_handle_t list, std::uint32_t event_count) {
    const auto begin = std::chrono::steady_clock::now();
    Check(zeCommandQueueExecuteCommandLists(queue_, 1U, &list, nullptr),
          "zeCommandQueueExecuteCommandLists");
    Check(zeCommandQueueSynchronize(queue_, UINT64_MAX),
          "zeCommandQueueSynchronize");
    const auto end = std::chrono::steady_clock::now();
    TimedRun result;
    result.stages_us.resize(event_count);
    for (std::uint32_t index = 0; index < event_count; ++index) {
      ze_kernel_timestamp_result_t timestamp{};
      Check(zeEventQueryKernelTimestamp(events_[index], &timestamp),
            "zeEventQueryKernelTimestamp");
      const std::uint64_t ticks = TimestampDelta(
          timestamp.context.kernelStart, timestamp.context.kernelEnd,
          properties_.kernelTimestampValidBits);
      result.stages_us[index] =
          ticks * timestamp_ns_per_tick_ / 1000.0;
      result.kernel_us += result.stages_us[index];
      Check(zeEventHostReset(events_[index]), "zeEventHostReset");
    }
    result.wall_us = std::chrono::duration<double, std::micro>(
        end - begin).count();
    return result;
  }

  void RunUntimed(ze_command_list_handle_t list) {
    Check(zeCommandQueueExecuteCommandLists(queue_, 1U, &list, nullptr),
          "zeCommandQueueExecuteCommandLists(untimed)");
    Check(zeCommandQueueSynchronize(queue_, UINT64_MAX),
          "zeCommandQueueSynchronize(untimed)");
  }

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
        ze_device_properties_t properties{
            ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES};
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
        for (auto& group : groups)
          group.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES;
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
    Require(device_ != nullptr, "PTL Level Zero device 0xb080 not found");
    std::uint64_t host0 = 0;
    std::uint64_t device0 = 0;
    std::uint64_t host1 = 0;
    std::uint64_t device1 = 0;
    Check(zeDeviceGetGlobalTimestamps(device_, &host0, &device0),
          "zeDeviceGetGlobalTimestamps(start)");
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    Check(zeDeviceGetGlobalTimestamps(device_, &host1, &device1),
          "zeDeviceGetGlobalTimestamps(end)");
    const std::uint64_t ticks = TimestampDelta(
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
    queue_desc.index = 0U;
    queue_desc.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
    Check(zeCommandQueueCreate(context_, device_, &queue_desc, &queue_),
          "zeCommandQueueCreate");
    ze_command_list_desc_t list_desc{ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC};
    list_desc.commandQueueGroupOrdinal = queue_ordinal_;
    Check(zeCommandListCreate(context_, device_, &list_desc, &upload_list_),
          "zeCommandListCreate(upload)");
    Check(zeCommandListCreate(context_, device_, &list_desc, &q8_list_),
          "zeCommandListCreate(q8)");
    Check(zeCommandListCreate(context_, device_, &list_desc, &baseline_list_),
          "zeCommandListCreate(baseline)");
    Check(zeCommandListCreate(context_, device_, &list_desc, &candidate_list_),
          "zeCommandListCreate(candidate)");
    Check(zeCommandListCreate(context_, device_, &list_desc, &reference_list_),
          "zeCommandListCreate(reference)");
    ze_event_pool_desc_t pool_desc{ZE_STRUCTURE_TYPE_EVENT_POOL_DESC};
    pool_desc.flags =
        ZE_EVENT_POOL_FLAG_HOST_VISIBLE | ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP;
    pool_desc.count = 5U;
    Check(zeEventPoolCreate(
              context_, &pool_desc, 1U, &device_, &event_pool_),
          "zeEventPoolCreate");
    for (std::uint32_t index = 0; index < 5U; ++index) {
      ze_event_desc_t event_desc{ZE_STRUCTURE_TYPE_EVENT_DESC};
      event_desc.index = index;
      event_desc.signal = ZE_EVENT_SCOPE_FLAG_HOST;
      event_desc.wait = ZE_EVENT_SCOPE_FLAG_HOST;
      Check(zeEventCreate(event_pool_, &event_desc, &events_[index]),
            "zeEventCreate");
    }
    module_bytes_ = ReadBinary(module_path);
    ze_module_desc_t module_desc{ZE_STRUCTURE_TYPE_MODULE_DESC};
    module_desc.format = ZE_MODULE_FORMAT_NATIVE;
    module_desc.inputSize = module_bytes_.size();
    module_desc.pInputModule = module_bytes_.data();
    module_desc.pBuildFlags = "";
    ze_module_build_log_handle_t log = nullptr;
    const ze_result_t result = zeModuleCreate(
        context_, device_, &module_desc, &module_, &log);
    if (result != ZE_RESULT_SUCCESS && log != nullptr) {
      std::size_t bytes = 0;
      zeModuleBuildLogGetString(log, &bytes, nullptr);
      std::string message(bytes, '\0');
      if (bytes != 0U)
        zeModuleBuildLogGetString(log, &bytes, message.data());
      zeModuleBuildLogDestroy(log);
      Die("zeModuleCreate failed: " + message);
    }
    if (log != nullptr) zeModuleBuildLogDestroy(log);
    Check(result, "zeModuleCreate");
  }

  void* AllocateDevice(std::size_t bytes) {
    ze_device_mem_alloc_desc_t desc{
        ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC};
    void* pointer = nullptr;
    Check(zeMemAllocDevice(context_, &desc, bytes, 64U, device_, &pointer),
          "zeMemAllocDevice");
    device_allocations_.push_back(pointer);
    return pointer;
  }

  void* AllocateShared(std::size_t bytes) {
    ze_device_mem_alloc_desc_t device_desc{
        ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC};
    ze_host_mem_alloc_desc_t host_desc{
        ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC};
    void* pointer = nullptr;
    Check(zeMemAllocShared(
              context_, &device_desc, &host_desc, bytes, 64U, device_,
              &pointer),
          "zeMemAllocShared");
    std::memset(pointer, 0, bytes);
    shared_allocations_.push_back(pointer);
    return pointer;
  }

  void* Upload(const void* source, std::size_t bytes) {
    Require(source != nullptr && bytes > 0U, "empty upload");
    void* destination = AllocateDevice(bytes);
    ze_host_mem_alloc_desc_t host_desc{
        ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC};
    void* staging = nullptr;
    Check(zeMemAllocHost(context_, &host_desc, bytes, 64U, &staging),
          "zeMemAllocHost");
    std::memcpy(staging, source, bytes);
    staging_allocations_.push_back(staging);
    Check(zeCommandListAppendMemoryCopy(
              upload_list_, destination, staging, bytes, nullptr, 0U,
              nullptr),
          "zeCommandListAppendMemoryCopy");
    return destination;
  }

  void FinishUploads() {
    Check(zeCommandListClose(upload_list_), "zeCommandListClose(upload)");
    Check(zeCommandQueueExecuteCommandLists(
              queue_, 1U, &upload_list_, nullptr),
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
    Check(zeKernelCreate(module_, &desc, &kernel), name);
    kernels_.push_back(kernel);
    return kernel;
  }

  void SetPointer(
      ze_kernel_handle_t kernel, std::uint32_t index, void* pointer) {
    Check(zeKernelSetArgumentValue(kernel, index, sizeof(pointer), &pointer),
          "zeKernelSetArgumentValue");
  }

  void Append(
      ze_command_list_handle_t list, ze_kernel_handle_t kernel,
      const ze_group_count_t& groups, ze_event_handle_t event) {
    Check(zeCommandListAppendLaunchKernel(
              list, kernel, &groups, event, 0U, nullptr),
          "zeCommandListAppendLaunchKernel");
    Check(zeCommandListAppendBarrier(list, nullptr, 0U, nullptr),
          "zeCommandListAppendBarrier");
  }

  void Record() {
    q8_kernel_ = CreateKernel("iq36_affine_q4_q8_f16");
    norm_kernel_ = CreateKernel(
        "iq36_affine_q4_hidden_group_norms_f16");
    reset_kernel_ = CreateKernel("iq36_affine_q4_reset");
    select_kernel_ = CreateKernel("iq36_affine_q4_bound_select_f16");
    exact_kernel_ = CreateKernel("iq36_affine_q4_exact_candidates_f16");
    candidate_top_kernel_ = CreateKernel(
        "iq36_affine_q4_candidate_top1_f16");
    baseline_kernel_ = CreateKernel(
        "iq36_affine_q4_full_i8_q8_matvec_f16");
    reference_kernel_ = CreateKernel(
        "iq36_affine_q4_reference_matvec_f16");
    violation_reset_kernel_ = CreateKernel(
        "iq36_affine_q4_violation_reset");
    violation_kernel_ = CreateKernel(
        "iq36_affine_q4_bound_violations_f16");
    full_top_kernel_ = CreateKernel("iq36_affine_q4_full_top1_f16");

    Check(zeKernelSetGroupSize(q8_kernel_, 64U, 1U, 1U), "group(q8)");
    Check(zeKernelSetGroupSize(norm_kernel_, 128U, 1U, 1U), "group(norm)");
    Check(zeKernelSetGroupSize(reset_kernel_, 1U, 1U, 1U), "group(reset)");
    Check(zeKernelSetGroupSize(select_kernel_, 256U, 1U, 1U),
          "group(select)");
    Check(zeKernelSetGroupSize(exact_kernel_, 64U, 1U, 1U), "group(exact)");
    Check(zeKernelSetGroupSize(candidate_top_kernel_, 256U, 1U, 1U),
          "group(candidate top)");
    Check(zeKernelSetGroupSize(baseline_kernel_, 256U, 1U, 1U),
          "group(baseline)");
    Check(zeKernelSetGroupSize(reference_kernel_, 256U, 1U, 1U),
          "group(reference)");
    Check(zeKernelSetGroupSize(violation_reset_kernel_, 1U, 1U, 1U),
          "group(violation reset)");
    Check(zeKernelSetGroupSize(violation_kernel_, 256U, 1U, 1U),
          "group(violation)");
    Check(zeKernelSetGroupSize(full_top_kernel_, 256U, 1U, 1U),
          "group(full top)");
    Check(zeKernelGetProperties(select_kernel_, &select_properties_),
          "properties(select)");
    Check(zeKernelGetProperties(exact_kernel_, &exact_properties_),
          "properties(exact)");

    SetPointer(q8_kernel_, 0U, input_);
    SetPointer(q8_kernel_, 1U, q8_);
    SetPointer(q8_kernel_, 2U, q8_d_);
    SetPointer(norm_kernel_, 0U, input_);
    SetPointer(norm_kernel_, 1U, q8_);
    SetPointer(norm_kernel_, 2U, q8_d_);
    SetPointer(norm_kernel_, 3U, hidden_norms_);
    SetPointer(norm_kernel_, 4U, hidden_delta_norms_);
    SetPointer(reset_kernel_, 0U, candidate_count_);
    SetPointer(select_kernel_, 0U, codes_);
    SetPointer(select_kernel_, 1U, minmax_);
    SetPointer(select_kernel_, 2U, residual_norms_);
    SetPointer(select_kernel_, 3U, scales_);
    SetPointer(select_kernel_, 4U, q8_);
    SetPointer(select_kernel_, 5U, q8_d_);
    SetPointer(select_kernel_, 6U, hidden_norms_);
    SetPointer(select_kernel_, 7U, hidden_delta_norms_);
    SetPointer(select_kernel_, 8U, seed_value_);
    SetPointer(select_kernel_, 9U, candidate_count_);
    SetPointer(select_kernel_, 10U, candidate_ids_);
    SetPointer(select_kernel_, 11U, upper_output_);
    SetPointer(exact_kernel_, 0U, weights_);
    SetPointer(exact_kernel_, 1U, scales_);
    SetPointer(exact_kernel_, 2U, input_);
    SetPointer(exact_kernel_, 3U, candidate_count_);
    SetPointer(exact_kernel_, 4U, candidate_ids_);
    SetPointer(exact_kernel_, 5U, candidate_values_);
    SetPointer(candidate_top_kernel_, 0U, candidate_count_);
    SetPointer(candidate_top_kernel_, 1U, candidate_ids_);
    SetPointer(candidate_top_kernel_, 2U, candidate_values_);
    SetPointer(candidate_top_kernel_, 3U, seed_id_);
    SetPointer(candidate_top_kernel_, 4U, seed_value_);
    SetPointer(candidate_top_kernel_, 5U, candidate_token_);
    SetPointer(baseline_kernel_, 0U, weights_);
    SetPointer(baseline_kernel_, 1U, scales_);
    SetPointer(baseline_kernel_, 2U, q8_);
    SetPointer(baseline_kernel_, 3U, q8_d_);
    SetPointer(baseline_kernel_, 4U, baseline_output_);
    SetPointer(reference_kernel_, 0U, weights_);
    SetPointer(reference_kernel_, 1U, scales_);
    SetPointer(reference_kernel_, 2U, input_);
    SetPointer(reference_kernel_, 3U, reference_output_);
    SetPointer(violation_reset_kernel_, 0U, violation_count_);
    SetPointer(violation_kernel_, 0U, reference_output_);
    SetPointer(violation_kernel_, 1U, upper_output_);
    SetPointer(violation_kernel_, 2U, violation_count_);
    SetPointer(full_top_kernel_, 0U, baseline_output_);
    SetPointer(full_top_kernel_, 1U, baseline_token_);

    const ze_group_count_t q8_groups{8U, 1U, 1U};
    const ze_group_count_t one_group{1U, 1U, 1U};
    const ze_group_count_t scan_groups{kScanWorkgroups, 1U, 1U};
    const ze_group_count_t exact_groups{kCapacity, 1U, 1U};
    const ze_group_count_t violation_groups{
        kViolationWorkgroups, 1U, 1U};
    Append(q8_list_, q8_kernel_, q8_groups, events_[0]);
    Check(zeCommandListClose(q8_list_), "close(q8)");

    Append(baseline_list_, baseline_kernel_, scan_groups, events_[0]);
    Append(baseline_list_, full_top_kernel_, one_group, events_[1]);
    Check(zeCommandListClose(baseline_list_), "close(baseline)");

    Append(candidate_list_, reset_kernel_, one_group, events_[0]);
    Append(candidate_list_, norm_kernel_, one_group, events_[1]);
    Append(candidate_list_, select_kernel_, scan_groups, events_[2]);
    Append(candidate_list_, exact_kernel_, exact_groups, events_[3]);
    Append(candidate_list_, candidate_top_kernel_, one_group, events_[4]);
    Check(zeCommandListClose(candidate_list_), "close(candidate)");

    Append(reference_list_, violation_reset_kernel_, one_group, nullptr);
    Append(reference_list_, reference_kernel_, scan_groups, nullptr);
    Append(reference_list_, violation_kernel_, violation_groups, nullptr);
    SetPointer(full_top_kernel_, 0U, reference_output_);
    SetPointer(full_top_kernel_, 1U, reference_token_);
    Append(reference_list_, full_top_kernel_, one_group, nullptr);
    Check(zeCommandListClose(reference_list_), "close(reference)");
  }

  void Cleanup() noexcept {
    if (queue_ != nullptr) zeCommandQueueSynchronize(queue_, UINT64_MAX);
    for (auto kernel : kernels_) {
      if (kernel != nullptr) zeKernelDestroy(kernel);
    }
    for (auto event : events_) {
      if (event != nullptr) zeEventDestroy(event);
    }
    if (event_pool_ != nullptr) zeEventPoolDestroy(event_pool_);
    if (reference_list_ != nullptr) zeCommandListDestroy(reference_list_);
    if (candidate_list_ != nullptr) zeCommandListDestroy(candidate_list_);
    if (baseline_list_ != nullptr) zeCommandListDestroy(baseline_list_);
    if (q8_list_ != nullptr) zeCommandListDestroy(q8_list_);
    if (upload_list_ != nullptr) zeCommandListDestroy(upload_list_);
    if (module_ != nullptr) zeModuleDestroy(module_);
    for (void* pointer : shared_allocations_) zeMemFree(context_, pointer);
    for (void* pointer : device_allocations_) zeMemFree(context_, pointer);
    for (void* pointer : staging_allocations_) zeMemFree(context_, pointer);
    if (queue_ != nullptr) zeCommandQueueDestroy(queue_);
    if (context_ != nullptr) zeContextDestroy(context_);
  }

  std::vector<std::uint8_t> host_weights_;
  std::vector<std::uint8_t> host_scales_;
  ze_driver_handle_t driver_ = nullptr;
  ze_device_handle_t device_ = nullptr;
  ze_context_handle_t context_ = nullptr;
  ze_command_queue_handle_t queue_ = nullptr;
  ze_command_list_handle_t upload_list_ = nullptr;
  ze_command_list_handle_t q8_list_ = nullptr;
  ze_command_list_handle_t baseline_list_ = nullptr;
  ze_command_list_handle_t candidate_list_ = nullptr;
  ze_command_list_handle_t reference_list_ = nullptr;
  ze_module_handle_t module_ = nullptr;
  ze_event_pool_handle_t event_pool_ = nullptr;
  ze_event_handle_t events_[5]{};
  std::vector<ze_kernel_handle_t> kernels_;
  ze_kernel_handle_t q8_kernel_ = nullptr;
  ze_kernel_handle_t norm_kernel_ = nullptr;
  ze_kernel_handle_t reset_kernel_ = nullptr;
  ze_kernel_handle_t select_kernel_ = nullptr;
  ze_kernel_handle_t exact_kernel_ = nullptr;
  ze_kernel_handle_t candidate_top_kernel_ = nullptr;
  ze_kernel_handle_t baseline_kernel_ = nullptr;
  ze_kernel_handle_t reference_kernel_ = nullptr;
  ze_kernel_handle_t violation_reset_kernel_ = nullptr;
  ze_kernel_handle_t violation_kernel_ = nullptr;
  ze_kernel_handle_t full_top_kernel_ = nullptr;
  ze_kernel_properties_t select_properties_{
      ZE_STRUCTURE_TYPE_KERNEL_PROPERTIES};
  ze_kernel_properties_t exact_properties_{
      ZE_STRUCTURE_TYPE_KERNEL_PROPERTIES};
  ze_device_properties_t properties_{ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES};
  std::vector<std::uint8_t> module_bytes_;
  std::vector<void*> device_allocations_;
  std::vector<void*> shared_allocations_;
  std::vector<void*> staging_allocations_;
  void* weights_ = nullptr;
  void* scales_ = nullptr;
  void* codes_ = nullptr;
  void* minmax_ = nullptr;
  void* residual_norms_ = nullptr;
  std::uint16_t* input_ = nullptr;
  std::int32_t* seed_id_ = nullptr;
  std::uint16_t* seed_value_ = nullptr;
  std::uint32_t* candidate_count_ = nullptr;
  std::int32_t* candidate_token_ = nullptr;
  std::int32_t* baseline_token_ = nullptr;
  std::int32_t* reference_token_ = nullptr;
  std::uint32_t* violation_count_ = nullptr;
  void* q8_ = nullptr;
  void* q8_d_ = nullptr;
  void* hidden_norms_ = nullptr;
  void* hidden_delta_norms_ = nullptr;
  void* upper_output_ = nullptr;
  void* candidate_ids_ = nullptr;
  void* candidate_values_ = nullptr;
  void* baseline_output_ = nullptr;
  void* reference_output_ = nullptr;
  std::uint32_t queue_ordinal_ = UINT32_MAX;
  std::string device_name_;
  double timestamp_ns_per_tick_ = 0.0;
};

void WriteDoubleArray(const std::vector<double>& values) {
  std::cout << "[";
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0U) std::cout << ",";
    std::cout << values[index];
  }
  std::cout << "]";
}

void WriteIntArray(const std::vector<std::uint32_t>& values) {
  std::cout << "[";
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0U) std::cout << ",";
    std::cout << values[index];
  }
  std::cout << "]";
}

void WriteProperties(const ze_kernel_properties_t& value) {
  std::cout << "{\"required_group_size_x\":" << value.requiredGroupSizeX
            << ",\"required_subgroup_size\":"
            << value.requiredSubgroupSize
            << ",\"local_mem_bytes\":" << value.localMemSize
            << ",\"private_mem_bytes\":" << value.privateMemSize
            << ",\"spill_mem_bytes\":" << value.spillMemSize << "}";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 9) {
      throw std::invalid_argument(
          "usage: iq36-openvino-lm-head-affine-q4-group128-component "
          "MODEL_BIN MODULE HIDDEN_F16 SEED_IDS_I32 TIMING_IDS_I32 "
          "WARMUP BLOCKS CORRECTNESS_EVENTS");
    }
    const std::string model = argv[1];
    const std::string module = argv[2];
    const auto hidden_path = std::filesystem::path(argv[3]);
    const auto seed_path = std::filesystem::path(argv[4]);
    const auto timing_path = std::filesystem::path(argv[5]);
    const int warmup = std::stoi(argv[6]);
    const int blocks = std::stoi(argv[7]);
    const int correctness_events = std::stoi(argv[8]);
    Require(warmup >= 0 && blocks > 0, "invalid timing counts");
    Require(
        correctness_events > 0 &&
            correctness_events <= static_cast<int>(kEventCount),
        "invalid correctness event count");

    const auto weights = ReadRange(model, kWeightOffset, kWeightBytes);
    const auto scales = ReadRange(model, kScaleOffset, kScaleBytes);
    const auto hidden = ReadTyped<std::uint16_t>(
        hidden_path, kEventCount * kColumns);
    const auto seed_ids = ReadTyped<std::int32_t>(seed_path, kEventCount);
    std::ifstream timing_input(timing_path, std::ios::binary | std::ios::ate);
    Require(static_cast<bool>(timing_input), "failed to open timing ids");
    const auto timing_bytes = timing_input.tellg();
    Require(
        timing_bytes > 0 &&
            timing_bytes % static_cast<std::streamoff>(
                sizeof(std::int32_t)) == 0,
        "timing id byte size mismatch");
    const auto timing_ids = ReadTyped<std::int32_t>(
        timing_path,
        static_cast<std::size_t>(timing_bytes) / sizeof(std::int32_t));
    for (std::int32_t value : seed_ids) {
      Require(value >= 0 && value < static_cast<std::int32_t>(kRows),
              "seed id out of range");
    }
    for (std::int32_t value : timing_ids) {
      Require(
          value >= 0 && value < correctness_events,
          "timing event out of range");
    }

    const PackedWeights packed = PackAffineQ4(weights);
    Runtime runtime(module, weights, scales, packed);
    std::vector<std::uint32_t> candidate_counts;
    candidate_counts.reserve(static_cast<std::size_t>(correctness_events));
    std::uint64_t violation_count = 0U;
    std::uint32_t overflow_count = 0U;
    std::uint32_t token_mismatch_count = 0U;
    std::uint32_t reference_mismatch_count = 0U;
    for (int event = 0; event < correctness_events; ++event) {
      const auto* row = hidden.data() +
          static_cast<std::size_t>(event) * kColumns;
      const CorrectnessRow result =
          runtime.CheckEvent(row, seed_ids[event]);
      candidate_counts.push_back(result.candidate_count);
      violation_count += result.violation_count;
      if (result.candidate_count > kCapacity) ++overflow_count;
      if (result.candidate_token != seed_ids[event])
        ++token_mismatch_count;
      if (result.reference_token != seed_ids[event])
        ++reference_mismatch_count;
    }
    const CorrectnessRow forced_overflow = runtime.CheckForcedOverflow(
        hidden.data(), seed_ids[0]);

    std::vector<PairRow> pair_rows;
    for (std::int32_t event : timing_ids) {
      const auto* row = hidden.data() +
          static_cast<std::size_t>(event) * kColumns;
      auto rows = runtime.Compare(
          row, seed_ids[event], event, warmup, blocks);
      pair_rows.insert(
          pair_rows.end(),
          std::make_move_iterator(rows.begin()),
          std::make_move_iterator(rows.end()));
    }

    const auto maximum_count = *std::max_element(
        candidate_counts.begin(), candidate_counts.end());
    const bool pass =
        violation_count == 0U &&
        overflow_count == 0U &&
        token_mismatch_count == 0U &&
        reference_mismatch_count == 0U &&
        forced_overflow.candidate_count == kRows &&
        forced_overflow.candidate_token == forced_overflow.reference_token;
    std::cout << std::setprecision(12) << std::boolalpha
              << "{\"schema\":\"iq36-affine-q4-group128-component-v1\""
              << ",\"device\":\"" << runtime.device_name() << "\""
              << ",\"timestamp_ns_per_tick\":"
              << runtime.timestamp_ns_per_tick()
              << ",\"packing_seconds\":" << packed.seconds
              << ",\"packed_bytes\":{"
              << "\"codes\":" << packed.codes.size()
              << ",\"minmax\":" << packed.minmax.size()
              << ",\"residual_norms\":"
              << packed.residual_norms.size() * sizeof(std::uint16_t)
              << ",\"source_scales\":" << scales.size() << "}"
              << ",\"correctness\":{"
              << "\"events\":" << correctness_events
              << ",\"candidate_counts\":";
    WriteIntArray(candidate_counts);
    std::cout << ",\"maximum_candidate_count\":" << maximum_count
              << ",\"capacity\":" << kCapacity
              << ",\"overflow_count\":" << overflow_count
              << ",\"bound_violation_count\":" << violation_count
              << ",\"token_mismatch_count\":" << token_mismatch_count
              << ",\"reference_mismatch_count\":"
              << reference_mismatch_count << "}"
              << ",\"forced_overflow\":{"
              << "\"candidate_count\":"
              << forced_overflow.candidate_count
              << ",\"candidate_token\":"
              << forced_overflow.candidate_token
              << ",\"reference_token\":"
              << forced_overflow.reference_token
              << ",\"fallback_selected\":"
              << (forced_overflow.candidate_count > kCapacity) << "}"
              << ",\"resources\":{\"select\":";
    WriteProperties(runtime.select_properties());
    std::cout << ",\"exact\":";
    WriteProperties(runtime.exact_properties());
    std::cout << "},\"timing\":{\"schedule\":\"ABBA\""
              << ",\"warmup\":" << warmup
              << ",\"blocks_per_event\":" << blocks
              << ",\"baseline_stages\":[\"full_i8_q8\",\"top1\"]"
              << ",\"candidate_stages\":["
              << "\"reset\",\"hidden_norms\",\"bound_select\","
              << "\"exact_candidates\",\"top1\"]"
              << ",\"samples\":[";
    for (std::size_t index = 0; index < pair_rows.size(); ++index) {
      if (index != 0U) std::cout << ",";
      const auto& row = pair_rows[index];
      std::cout << "{\"event\":" << row.event
                << ",\"block\":" << row.block
                << ",\"baseline_kernel_us\":" << row.baseline_kernel_us
                << ",\"candidate_kernel_us\":" << row.candidate_kernel_us
                << ",\"saving_kernel_us\":" << row.saving_kernel_us
                << ",\"baseline_wall_us\":" << row.baseline_wall_us
                << ",\"candidate_wall_us\":" << row.candidate_wall_us
                << ",\"saving_wall_us\":" << row.saving_wall_us
                << ",\"baseline_stages_us\":";
      WriteDoubleArray(row.baseline_stages_us);
      std::cout << ",\"candidate_stages_us\":";
      WriteDoubleArray(row.candidate_stages_us);
      std::cout << "}";
    }
    std::cout << "]},\"required_checks_passed\":" << pass << "}"
              << std::endl;
    return pass ? 0 : 2;
  } catch (const std::exception& error) {
    std::cerr << error.what() << std::endl;
    return 1;
  }
}
