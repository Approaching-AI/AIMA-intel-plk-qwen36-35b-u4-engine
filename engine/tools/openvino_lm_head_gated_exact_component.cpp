#include <level_zero/ze_api.h>

#include <algorithm>
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
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr std::uint32_t kIntelVendorId = 0x8086U;
constexpr std::uint32_t kPtlDeviceId = 0xB080U;
constexpr std::uint32_t kRows = 248320U;
constexpr std::uint32_t kColumns = 2048U;
constexpr std::uint32_t kTopK = 8U;
constexpr std::uint32_t kMatvecWorkgroups = 384U;
constexpr std::uint32_t kBlockRows = 256U;
constexpr std::uint32_t kBlockCount = (kRows + kBlockRows - 1U) / kBlockRows;
constexpr std::uint64_t kWeightOffset = UINT64_C(18137149498);
constexpr std::uint64_t kWeightBytes = UINT64_C(508559360);
constexpr std::uint64_t kScaleOffset = UINT64_C(18645708858);
constexpr std::uint64_t kScaleBytes = UINT64_C(496640);
constexpr std::uint64_t kF16OutputBytes =
    static_cast<std::uint64_t>(kRows) * sizeof(std::uint16_t);
constexpr std::uint64_t kMandatoryMatvecBytes =
    kWeightBytes + kScaleBytes + kF16OutputBytes;

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
          "failed to read complete model binary range");
  return values;
}

std::vector<std::uint8_t> ReadBinary(const std::string& path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  Require(static_cast<bool>(input), "failed to open module " + path);
  const auto size = input.tellg();
  Require(size > 0, "module is empty");
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
  input.seekg(0, std::ios::beg);
  input.read(reinterpret_cast<char*>(bytes.data()), size);
  Require(input.gcount() == size, "failed to read complete module");
  return bytes;
}

std::vector<float> ReadF32(
    const std::filesystem::path& path, std::size_t count) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  Require(static_cast<bool>(input), "failed to open F32 input " + path.string());
  const auto bytes = input.tellg();
  Require(bytes == static_cast<std::streamoff>(count * sizeof(float)),
          "F32 input size mismatch " + path.string());
  std::vector<float> values(count);
  input.seekg(0, std::ios::beg);
  input.read(reinterpret_cast<char*>(values.data()), bytes);
  Require(input.gcount() == bytes,
          "failed to read complete F32 input " + path.string());
  return values;
}

void WriteF32(
    const std::filesystem::path& path, const std::vector<float>& values) {
  std::ofstream output(path, std::ios::binary);
  Require(static_cast<bool>(output), "failed to open output " + path.string());
  output.write(reinterpret_cast<const char*>(values.data()),
               static_cast<std::streamsize>(values.size() * sizeof(float)));
  Require(static_cast<bool>(output), "failed to write output " + path.string());
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
  std::uint32_t adjusted_exponent = static_cast<std::uint32_t>(half_exponent);
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

std::uint64_t TimestampDelta(
    std::uint64_t start, std::uint64_t end, std::uint32_t valid_bits) {
  if (valid_bits == 0U || valid_bits >= 64U) return end - start;
  const std::uint64_t mask = (UINT64_C(1) << valid_bits) - 1U;
  return (end - start) & mask;
}

double Minimum(const std::vector<double>& values) {
  Require(!values.empty(), "empty timing samples");
  return *std::min_element(values.begin(), values.end());
}

double Mean(const std::vector<double>& values) {
  Require(!values.empty(), "empty timing samples");
  double total = 0.0;
  for (double value : values) total += value;
  return total / static_cast<double>(values.size());
}

double Median(std::vector<double> values) {
  Require(!values.empty(), "empty timing samples");
  std::sort(values.begin(), values.end());
  const std::size_t middle = values.size() / 2U;
  if ((values.size() & 1U) != 0U) return values[middle];
  return (values[middle - 1U] + values[middle]) * 0.5;
}

std::vector<double> AddSamples(
    const std::vector<double>& lhs, const std::vector<double>& rhs) {
  Require(lhs.size() == rhs.size(), "timing sample count mismatch");
  std::vector<double> result(lhs.size());
  for (std::size_t index = 0; index < lhs.size(); ++index)
    result[index] = lhs[index] + rhs[index];
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

std::string JsonString(const std::string& value) {
  std::ostringstream output;
  output << '"';
  for (unsigned char character : value) {
    switch (character) {
      case '"': output << "\\\""; break;
      case '\\': output << "\\\\"; break;
      case '\b': output << "\\b"; break;
      case '\f': output << "\\f"; break;
      case '\n': output << "\\n"; break;
      case '\r': output << "\\r"; break;
      case '\t': output << "\\t"; break;
      default:
        if (character < 0x20U) {
          output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                 << static_cast<unsigned int>(character) << std::dec;
        } else {
          output << static_cast<char>(character);
        }
    }
  }
  output << '"';
  return output.str();
}

class Runtime {
 public:
  struct Run {
    std::vector<double> q8_us;
    std::vector<double> matvec_us;
    std::vector<double> block_topk_us;
    std::vector<double> merge_us;
    std::vector<double> correction_us;
    std::vector<double> wall_us;
  };

  struct BlockComparison {
    std::vector<double> baseline_us;
    std::vector<double> candidate_us;
    std::vector<double> saving_us;
    std::vector<double> baseline_wall_us;
    std::vector<double> candidate_wall_us;
    std::vector<double> saving_wall_us;
  };

  Runtime(
      const std::string& module_path,
      const std::vector<std::uint8_t>& weights,
      const std::vector<std::uint8_t>& scales) {
    Require(weights.size() == kWeightBytes, "weight byte size mismatch");
    Require(scales.size() == kScaleBytes, "scale byte size mismatch");
    InitializeDevice();
    InitializeRuntime(module_path);
    weights_ = Upload(weights.data(), weights.size());
    scales_ = Upload(scales.data(), scales.size());
    input_ = static_cast<std::uint16_t*>(
        AllocateShared(kColumns * sizeof(std::uint16_t)));
    output_ = static_cast<std::uint16_t*>(
        AllocateShared(kRows * sizeof(std::uint16_t)));
    q8_ = AllocateDevice(kColumns * sizeof(std::int8_t));
    q8_d_ = AllocateDevice(8U * sizeof(float));
    partial_ids_ = AllocateDevice(
        kBlockCount * kTopK * sizeof(std::int32_t));
    partial_values_ = AllocateDevice(
        kBlockCount * kTopK * sizeof(float));
    top_ids_ = static_cast<std::int32_t*>(
        AllocateShared(kTopK * sizeof(std::int32_t)));
    top_values_ = static_cast<float*>(
        AllocateShared(kTopK * sizeof(float)));
    FinishUploads();
    Record();
  }

  ~Runtime() { Cleanup(); }

  Run Execute(
      const std::vector<float>& input, int warmup, int samples,
      bool parallel_block_topk = false) {
    Require(input.size() == kColumns, "hidden input shape mismatch");
    Require(warmup >= 0 && samples > 0, "invalid timing sample counts");
    for (std::size_t index = 0; index < input.size(); ++index)
      input_[index] = FloatToHalf(input[index]);
    Run run;
    ze_command_list_handle_t list =
        parallel_block_topk ? candidate_command_list_ : command_list_;
    for (int sample = -warmup; sample < samples; ++sample) {
      const auto begin = std::chrono::steady_clock::now();
      Check(zeCommandQueueExecuteCommandLists(
                queue_, 1, &list, nullptr),
            "zeCommandQueueExecuteCommandLists");
      Check(zeCommandQueueSynchronize(queue_, UINT64_MAX),
            "zeCommandQueueSynchronize");
      const auto end = std::chrono::steady_clock::now();
      ze_kernel_timestamp_result_t timestamps[5]{};
      for (std::uint32_t index = 0; index < 5U; ++index) {
        Check(zeEventQueryKernelTimestamp(events_[index], &timestamps[index]),
              "zeEventQueryKernelTimestamp");
      }
      double stage_us[5]{};
      for (std::uint32_t index = 0; index < 5U; ++index) {
        const std::uint64_t ticks = TimestampDelta(
            timestamps[index].context.kernelStart,
            timestamps[index].context.kernelEnd,
            properties_.kernelTimestampValidBits);
        stage_us[index] = ticks * timestamp_ns_per_tick_ / 1000.0;
        Check(zeEventHostReset(events_[index]), "zeEventHostReset");
      }
      if (sample < 0) continue;
      run.q8_us.push_back(stage_us[0]);
      run.matvec_us.push_back(stage_us[1]);
      run.block_topk_us.push_back(stage_us[2]);
      run.merge_us.push_back(stage_us[3]);
      run.correction_us.push_back(stage_us[4]);
      run.wall_us.push_back(
          std::chrono::duration<double, std::micro>(end - begin).count());
    }
    return run;
  }

  BlockComparison CompareBlockTopK(
      const std::vector<float>& input, int warmup, int blocks) {
    Require(input.size() == kColumns, "hidden input shape mismatch");
    Require(warmup >= 0 && blocks > 0, "invalid paired sample counts");
    for (std::size_t index = 0; index < input.size(); ++index)
      input_[index] = FloatToHalf(input[index]);
    PrepareMatvec();
    for (int sample = 0; sample < warmup; ++sample) {
      ExecuteBlock(false);
      ExecuteBlock(true);
    }
    BlockComparison comparison;
    for (int block = 0; block < blocks; ++block) {
      const auto a1 = ExecuteBlock(false);
      const auto b1 = ExecuteBlock(true);
      const auto b2 = ExecuteBlock(true);
      const auto a2 = ExecuteBlock(false);
      const double baseline = (a1.kernel_us + a2.kernel_us) * 0.5;
      const double candidate = (b1.kernel_us + b2.kernel_us) * 0.5;
      const double baseline_wall = (a1.wall_us + a2.wall_us) * 0.5;
      const double candidate_wall = (b1.wall_us + b2.wall_us) * 0.5;
      comparison.baseline_us.push_back(baseline);
      comparison.candidate_us.push_back(candidate);
      comparison.saving_us.push_back(baseline - candidate);
      comparison.baseline_wall_us.push_back(baseline_wall);
      comparison.candidate_wall_us.push_back(candidate_wall);
      comparison.saving_wall_us.push_back(
          baseline_wall - candidate_wall);
    }
    return comparison;
  }

  std::vector<float> Output() const {
    std::vector<float> values(kRows);
    for (std::size_t index = 0; index < values.size(); ++index)
      values[index] = HalfToFloat(output_[index]);
    return values;
  }

  std::vector<std::int32_t> SelectedIds() const {
    return {top_ids_, top_ids_ + kTopK};
  }

  const std::string& device_name() const { return device_name_; }
  double timestamp_ns_per_tick() const { return timestamp_ns_per_tick_; }
  const ze_kernel_properties_t& baseline_block_properties() const {
    return baseline_block_properties_;
  }
  const ze_kernel_properties_t& candidate_block_properties() const {
    return candidate_block_properties_;
  }

 private:
  struct BlockSample {
    double kernel_us = 0.0;
    double wall_us = 0.0;
  };

  void PrepareMatvec() {
    auto list = prepare_command_list_;
    Check(zeCommandQueueExecuteCommandLists(queue_, 1, &list, nullptr),
          "zeCommandQueueExecuteCommandLists(prepare)");
    Check(zeCommandQueueSynchronize(queue_, UINT64_MAX),
          "zeCommandQueueSynchronize(prepare)");
    for (std::uint32_t index = 0; index < 2U; ++index) {
      ze_kernel_timestamp_result_t timestamp{};
      Check(zeEventQueryKernelTimestamp(events_[index], &timestamp),
            "zeEventQueryKernelTimestamp(prepare)");
      const auto ticks = TimestampDelta(
          timestamp.context.kernelStart, timestamp.context.kernelEnd,
          properties_.kernelTimestampValidBits);
      Require(ticks > 0U, "prepare stage timestamp is zero");
      Check(zeEventHostReset(events_[index]), "zeEventHostReset(prepare)");
    }
  }

  BlockSample ExecuteBlock(bool candidate) {
    ze_command_list_handle_t list = candidate
        ? candidate_block_command_list_ : block_command_list_;
    const auto begin = std::chrono::steady_clock::now();
    Check(zeCommandQueueExecuteCommandLists(queue_, 1, &list, nullptr),
          "zeCommandQueueExecuteCommandLists(block)");
    Check(zeCommandQueueSynchronize(queue_, UINT64_MAX),
          "zeCommandQueueSynchronize(block)");
    const auto end = std::chrono::steady_clock::now();
    ze_kernel_timestamp_result_t timestamp{};
    Check(zeEventQueryKernelTimestamp(events_[2], &timestamp),
          "zeEventQueryKernelTimestamp(block)");
    const auto ticks = TimestampDelta(
        timestamp.context.kernelStart, timestamp.context.kernelEnd,
        properties_.kernelTimestampValidBits);
    Check(zeEventHostReset(events_[2]), "zeEventHostReset(block)");
    return {
        ticks * timestamp_ns_per_tick_ / 1000.0,
        std::chrono::duration<double, std::micro>(end - begin).count(),
    };
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
    queue_desc.index = 0;
    queue_desc.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
    Check(zeCommandQueueCreate(context_, device_, &queue_desc, &queue_),
          "zeCommandQueueCreate");
    ze_command_list_desc_t list_desc{ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC};
    list_desc.commandQueueGroupOrdinal = queue_ordinal_;
    Check(zeCommandListCreate(context_, device_, &list_desc, &upload_list_),
          "zeCommandListCreate(upload)");
    Check(zeCommandListCreate(context_, device_, &list_desc, &command_list_),
          "zeCommandListCreate(baseline run)");
    Check(zeCommandListCreate(
              context_, device_, &list_desc, &candidate_command_list_),
          "zeCommandListCreate(candidate run)");
    Check(zeCommandListCreate(
              context_, device_, &list_desc, &prepare_command_list_),
          "zeCommandListCreate(prepare)");
    Check(zeCommandListCreate(
              context_, device_, &list_desc, &block_command_list_),
          "zeCommandListCreate(baseline block)");
    Check(zeCommandListCreate(
              context_, device_, &list_desc, &candidate_block_command_list_),
          "zeCommandListCreate(candidate block)");
    ze_event_pool_desc_t pool_desc{ZE_STRUCTURE_TYPE_EVENT_POOL_DESC};
    pool_desc.flags =
        ZE_EVENT_POOL_FLAG_HOST_VISIBLE | ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP;
    pool_desc.count = 5U;
    Check(zeEventPoolCreate(
              context_, &pool_desc, 1, &device_, &event_pool_),
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
      if (bytes != 0U) zeModuleBuildLogGetString(log, &bytes, message.data());
      zeModuleBuildLogDestroy(log);
      Die("zeModuleCreate failed: " + message);
    }
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
    Check(zeMemAllocShared(
              context_, &device_desc, &host_desc, bytes, 64, device_, &pointer),
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
    kernels_.push_back(kernel);
    return kernel;
  }

  void SetPointer(
      ze_kernel_handle_t kernel, std::uint32_t index, void* pointer) {
    Check(zeKernelSetArgumentValue(kernel, index, sizeof(pointer), &pointer),
          "zeKernelSetArgumentValue(pointer)");
  }

  void Record() {
    q8_kernel_ = CreateKernel(
        "iq36_lm_head_gated_exact_component_q8_f16");
    matvec_kernel_ = CreateKernel(
        "iq36_lm_head_gated_exact_component_matvec_f16");
    block_topk_kernel_ = CreateKernel(
        "iq36_lm_head_gated_exact_component_block_topk8_f16");
    parallel_block_topk_kernel_ = CreateKernel(
        "iq36_lm_head_gated_exact_component_parallel_block_topk8_f16");
    merge_kernel_ = CreateKernel(
        "iq36_lm_head_gated_exact_component_topk8_merge_f32");
    correction_kernel_ = CreateKernel(
        "iq36_lm_head_gated_exact_component_correction_f16");
    Check(zeKernelSetGroupSize(q8_kernel_, 64, 1, 1),
          "zeKernelSetGroupSize(q8)");
    Check(zeKernelSetGroupSize(matvec_kernel_, 256, 1, 1),
          "zeKernelSetGroupSize(matvec)");
    Check(zeKernelSetGroupSize(block_topk_kernel_, 256, 1, 1),
          "zeKernelSetGroupSize(block_topk)");
    Check(zeKernelSetGroupSize(parallel_block_topk_kernel_, 256, 1, 1),
          "zeKernelSetGroupSize(parallel_block_topk)");
    Check(zeKernelSetGroupSize(merge_kernel_, 256, 1, 1),
          "zeKernelSetGroupSize(merge)");
    Check(zeKernelSetGroupSize(correction_kernel_, 64, 1, 1),
          "zeKernelSetGroupSize(correction)");
    Check(zeKernelGetProperties(
              block_topk_kernel_, &baseline_block_properties_),
          "zeKernelGetProperties(block_topk)");
    Check(zeKernelGetProperties(
              parallel_block_topk_kernel_, &candidate_block_properties_),
          "zeKernelGetProperties(parallel_block_topk)");

    SetPointer(q8_kernel_, 0, input_);
    SetPointer(q8_kernel_, 1, q8_);
    SetPointer(q8_kernel_, 2, q8_d_);
    SetPointer(matvec_kernel_, 0, weights_);
    SetPointer(matvec_kernel_, 1, scales_);
    SetPointer(matvec_kernel_, 2, q8_);
    SetPointer(matvec_kernel_, 3, q8_d_);
    SetPointer(matvec_kernel_, 4, output_);
    SetPointer(block_topk_kernel_, 0, output_);
    SetPointer(block_topk_kernel_, 1, partial_ids_);
    SetPointer(block_topk_kernel_, 2, partial_values_);
    SetPointer(parallel_block_topk_kernel_, 0, output_);
    SetPointer(parallel_block_topk_kernel_, 1, partial_ids_);
    SetPointer(parallel_block_topk_kernel_, 2, partial_values_);
    SetPointer(merge_kernel_, 0, partial_ids_);
    SetPointer(merge_kernel_, 1, partial_values_);
    SetPointer(merge_kernel_, 2, top_ids_);
    SetPointer(merge_kernel_, 3, top_values_);
    SetPointer(correction_kernel_, 0, weights_);
    SetPointer(correction_kernel_, 1, scales_);
    SetPointer(correction_kernel_, 2, input_);
    SetPointer(correction_kernel_, 3, top_ids_);
    SetPointer(correction_kernel_, 4, output_);

    const ze_group_count_t group_counts[5] = {
        {8U, 1U, 1U},
        {kMatvecWorkgroups, 1U, 1U},
        {kBlockCount, 1U, 1U},
        {1U, 1U, 1U},
        {kTopK, 1U, 1U},
    };
    const auto record_full = [&](ze_command_list_handle_t list,
                                 ze_kernel_handle_t block_kernel) {
      const ze_kernel_handle_t kernels[5] = {
          q8_kernel_, matvec_kernel_, block_kernel, merge_kernel_,
          correction_kernel_,
      };
      for (std::uint32_t index = 0; index < 5U; ++index) {
        Check(zeCommandListAppendLaunchKernel(
                  list, kernels[index], &group_counts[index],
                  events_[index], 0, nullptr),
              "zeCommandListAppendLaunchKernel(full)");
        Check(zeCommandListAppendBarrier(list, nullptr, 0, nullptr),
              "zeCommandListAppendBarrier(full)");
      }
      Check(zeCommandListClose(list), "zeCommandListClose(full)");
    };
    record_full(command_list_, block_topk_kernel_);
    record_full(candidate_command_list_, parallel_block_topk_kernel_);

    const ze_kernel_handle_t prepare_kernels[2] = {
        q8_kernel_, matvec_kernel_,
    };
    for (std::uint32_t index = 0; index < 2U; ++index) {
      Check(zeCommandListAppendLaunchKernel(
                prepare_command_list_, prepare_kernels[index],
                &group_counts[index], events_[index], 0, nullptr),
            "zeCommandListAppendLaunchKernel(prepare)");
      Check(zeCommandListAppendBarrier(
                prepare_command_list_, nullptr, 0, nullptr),
            "zeCommandListAppendBarrier(prepare)");
    }
    Check(zeCommandListClose(prepare_command_list_),
          "zeCommandListClose(prepare)");

    const auto record_block = [&](ze_command_list_handle_t list,
                                  ze_kernel_handle_t kernel) {
      Check(zeCommandListAppendLaunchKernel(
                list, kernel, &group_counts[2], events_[2], 0, nullptr),
            "zeCommandListAppendLaunchKernel(block)");
      Check(zeCommandListAppendBarrier(list, nullptr, 0, nullptr),
            "zeCommandListAppendBarrier(block)");
      Check(zeCommandListClose(list), "zeCommandListClose(block)");
    };
    record_block(block_command_list_, block_topk_kernel_);
    record_block(candidate_block_command_list_, parallel_block_topk_kernel_);
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
    if (candidate_block_command_list_ != nullptr)
      zeCommandListDestroy(candidate_block_command_list_);
    if (block_command_list_ != nullptr)
      zeCommandListDestroy(block_command_list_);
    if (prepare_command_list_ != nullptr)
      zeCommandListDestroy(prepare_command_list_);
    if (candidate_command_list_ != nullptr)
      zeCommandListDestroy(candidate_command_list_);
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
  ze_command_list_handle_t candidate_command_list_ = nullptr;
  ze_command_list_handle_t prepare_command_list_ = nullptr;
  ze_command_list_handle_t block_command_list_ = nullptr;
  ze_command_list_handle_t candidate_block_command_list_ = nullptr;
  ze_module_handle_t module_ = nullptr;
  ze_event_pool_handle_t event_pool_ = nullptr;
  ze_event_handle_t events_[5]{};
  std::vector<ze_kernel_handle_t> kernels_;
  ze_kernel_handle_t q8_kernel_ = nullptr;
  ze_kernel_handle_t matvec_kernel_ = nullptr;
  ze_kernel_handle_t block_topk_kernel_ = nullptr;
  ze_kernel_handle_t parallel_block_topk_kernel_ = nullptr;
  ze_kernel_handle_t merge_kernel_ = nullptr;
  ze_kernel_handle_t correction_kernel_ = nullptr;
  ze_kernel_properties_t baseline_block_properties_{
      ZE_STRUCTURE_TYPE_KERNEL_PROPERTIES};
  ze_kernel_properties_t candidate_block_properties_{
      ZE_STRUCTURE_TYPE_KERNEL_PROPERTIES};
  ze_device_properties_t properties_{ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES};
  std::vector<std::uint8_t> module_bytes_;
  std::vector<void*> device_allocations_;
  std::vector<void*> shared_allocations_;
  std::vector<void*> staging_allocations_;
  void* weights_ = nullptr;
  void* scales_ = nullptr;
  std::uint16_t* input_ = nullptr;
  std::uint16_t* output_ = nullptr;
  void* q8_ = nullptr;
  void* q8_d_ = nullptr;
  void* partial_ids_ = nullptr;
  void* partial_values_ = nullptr;
  std::int32_t* top_ids_ = nullptr;
  float* top_values_ = nullptr;
  std::uint32_t queue_ordinal_ = UINT32_MAX;
  std::string device_name_;
  double timestamp_ns_per_tick_ = 0.0;
};

struct PhaseResult {
  int phase = 0;
  Runtime::Run timing;
  std::filesystem::path output_path;
  std::vector<std::int32_t> selected_ids;
  bool finite = false;
};

struct PairedPhaseResult {
  int phase = 0;
  Runtime::BlockComparison timing;
  std::filesystem::path output_path;
  std::vector<std::int32_t> selected_ids;
  bool output_bitwise_equal = false;
  bool selected_ids_equal = false;
  bool finite = false;
  double baseline_full_block_topk_us = 0.0;
  double candidate_full_block_topk_us = 0.0;
};

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 9 && argc != 10) {
      throw std::invalid_argument(
          "usage: iq36-openvino-lm-head-gated-exact-component MODEL_BIN "
          "MODULE HIDDEN_DIR OUTPUT_DIR WARMUP SAMPLES FIRST_PHASE LAST_PHASE "
          "[paired-block-topk]");
    }
    const std::string model_bin = argv[1];
    const std::string module = argv[2];
    const std::filesystem::path hidden_dir = argv[3];
    const std::filesystem::path output_dir = argv[4];
    const int warmup = std::stoi(argv[5]);
    const int samples = std::stoi(argv[6]);
    const int first_phase = std::stoi(argv[7]);
    const int last_phase = std::stoi(argv[8]);
    const bool paired_block_topk =
        argc == 10 && std::string(argv[9]) == "paired-block-topk";
    Require(argc == 9 || paired_block_topk,
            "unknown component execution mode");
    Require(warmup >= 0 && samples > 0,
            "warmup must be nonnegative and samples positive");
    Require(first_phase >= 0 && last_phase >= first_phase &&
                last_phase <= 4096,
            "phase range must satisfy 0 <= first <= last <= 4096");
    std::filesystem::create_directories(output_dir);

    auto weights = ReadRange(model_bin, kWeightOffset, kWeightBytes);
    auto scales = ReadRange(model_bin, kScaleOffset, kScaleBytes);
    Runtime runtime(module, weights, scales);
    weights.clear();
    weights.shrink_to_fit();
    scales.clear();
    scales.shrink_to_fit();

    if (paired_block_topk) {
      bool all_outputs_equal = true;
      bool all_selected_ids_equal = true;
      bool all_finite = true;
      bool all_timings_positive = true;
      std::vector<PairedPhaseResult> paired_phases;
      paired_phases.reserve(
          static_cast<std::size_t>(last_phase - first_phase + 1));
      for (int phase = first_phase; phase <= last_phase; ++phase) {
        std::ostringstream stem;
        stem << "step" << std::setfill('0') << std::setw(4) << phase;
        const auto input = ReadF32(
            hidden_dir / (stem.str() + "-lm-head-input.f32"), kColumns);
        const auto output_path =
            output_dir / (stem.str() + "-logits.f32");
        const auto baseline_timing =
            runtime.Execute(input, warmup, 1, false);
        const auto baseline_output = runtime.Output();
        const auto baseline_ids = runtime.SelectedIds();
        const auto candidate_timing =
            runtime.Execute(input, warmup, 1, true);
        const auto candidate_output = runtime.Output();
        const auto candidate_ids = runtime.SelectedIds();
        const bool output_equal =
            baseline_output.size() == candidate_output.size() &&
            std::memcmp(
                baseline_output.data(), candidate_output.data(),
                baseline_output.size() * sizeof(float)) == 0;
        const bool ids_equal = baseline_ids == candidate_ids;
        const bool finite = std::all_of(
            candidate_output.begin(), candidate_output.end(),
            [](float value) { return std::isfinite(value); });
        auto comparison =
            runtime.CompareBlockTopK(input, warmup, samples);
        const bool timings_positive =
            Minimum(baseline_timing.block_topk_us) > 0.0 &&
            Minimum(candidate_timing.block_topk_us) > 0.0 &&
            comparison.baseline_us.size() ==
                static_cast<std::size_t>(samples) &&
            comparison.candidate_us.size() ==
                static_cast<std::size_t>(samples) &&
            Minimum(comparison.baseline_us) > 0.0 &&
            Minimum(comparison.candidate_us) > 0.0;
        WriteF32(output_path, candidate_output);
        all_outputs_equal = all_outputs_equal && output_equal;
        all_selected_ids_equal = all_selected_ids_equal && ids_equal;
        all_finite = all_finite && finite;
        all_timings_positive = all_timings_positive && timings_positive;
        paired_phases.push_back({
            phase, std::move(comparison), output_path, candidate_ids,
            output_equal, ids_equal, finite,
            baseline_timing.block_topk_us[0],
            candidate_timing.block_topk_us[0]});
      }
      const bool pass =
          runtime.device_name().find("B390") != std::string::npos &&
          all_outputs_equal && all_selected_ids_equal &&
          all_finite && all_timings_positive;
      const auto& baseline_properties =
          runtime.baseline_block_properties();
      const auto& candidate_properties =
          runtime.candidate_block_properties();
      std::cout << std::boolalpha << std::setprecision(12) << "{"
                << "\"mode\":\"paired-block-topk\","
                << "\"device_name\":" << JsonString(runtime.device_name())
                << ",\"rows\":" << kRows
                << ",\"columns\":" << kColumns
                << ",\"block_count\":" << kBlockCount
                << ",\"topk\":" << kTopK
                << ",\"paired_blocks_per_hidden\":" << samples
                << ",\"schedule\":\"ABBA\","
                << "\"baseline_block_resources\":{"
                << "\"required_group_size_x\":"
                << baseline_properties.requiredGroupSizeX
                << ",\"required_subgroup_size\":"
                << baseline_properties.requiredSubgroupSize
                << ",\"local_mem_bytes\":"
                << baseline_properties.localMemSize
                << ",\"private_mem_bytes\":"
                << baseline_properties.privateMemSize
                << ",\"spill_mem_bytes\":"
                << baseline_properties.spillMemSize << "},"
                << "\"candidate_block_resources\":{"
                << "\"required_group_size_x\":"
                << candidate_properties.requiredGroupSizeX
                << ",\"required_subgroup_size\":"
                << candidate_properties.requiredSubgroupSize
                << ",\"local_mem_bytes\":"
                << candidate_properties.localMemSize
                << ",\"private_mem_bytes\":"
                << candidate_properties.privateMemSize
                << ",\"spill_mem_bytes\":"
                << candidate_properties.spillMemSize << "},"
                << "\"phases\":[";
      for (std::size_t index = 0; index < paired_phases.size(); ++index) {
        if (index != 0U) std::cout << ",";
        const auto& row = paired_phases[index];
        std::cout << "{\"phase\":" << row.phase
                  << ",\"output\":" << JsonString(row.output_path.string())
                  << ",\"output_bitwise_equal\":"
                  << row.output_bitwise_equal
                  << ",\"selected_ids_equal\":" << row.selected_ids_equal
                  << ",\"finite\":" << row.finite
                  << ",\"selected_ids\":";
        WriteIntArray(row.selected_ids);
        std::cout << ",\"baseline_full_block_topk_us\":"
                  << row.baseline_full_block_topk_us
                  << ",\"candidate_full_block_topk_us\":"
                  << row.candidate_full_block_topk_us
                  << ",\"baseline_block_samples_us\":";
        WriteDoubleArray(row.timing.baseline_us);
        std::cout << ",\"candidate_block_samples_us\":";
        WriteDoubleArray(row.timing.candidate_us);
        std::cout << ",\"saving_block_samples_us\":";
        WriteDoubleArray(row.timing.saving_us);
        std::cout << ",\"baseline_wall_samples_us\":";
        WriteDoubleArray(row.timing.baseline_wall_us);
        std::cout << ",\"candidate_wall_samples_us\":";
        WriteDoubleArray(row.timing.candidate_wall_us);
        std::cout << ",\"saving_wall_samples_us\":";
        WriteDoubleArray(row.timing.saving_wall_us);
        std::cout << "}";
      }
      std::cout << "],\"all_outputs_bitwise_equal\":"
                << all_outputs_equal
                << ",\"all_selected_ids_equal\":"
                << all_selected_ids_equal
                << ",\"all_finite\":" << all_finite
                << ",\"all_timings_positive\":" << all_timings_positive
                << ",\"required_checks_passed\":" << pass << "}"
                << std::endl;
      return pass ? 0 : 2;
    }

    bool all_finite = true;
    bool all_stage_timestamps_positive = true;
    bool all_selected_ids_valid = true;
    std::vector<PhaseResult> phases;
    phases.reserve(static_cast<std::size_t>(last_phase - first_phase + 1));
    for (int phase = first_phase; phase <= last_phase; ++phase) {
      std::ostringstream stem;
      stem << "step" << std::setfill('0') << std::setw(4) << phase;
      const auto input_path =
          hidden_dir / (stem.str() + "-lm-head-input.f32");
      const auto output_path =
          output_dir / (stem.str() + "-logits.f32");
      const auto input = ReadF32(input_path, kColumns);
      auto timing = runtime.Execute(input, warmup, samples);
      const auto output = runtime.Output();
      const auto selected_ids = runtime.SelectedIds();
      const bool finite = std::all_of(
          output.begin(), output.end(),
          [](float value) { return std::isfinite(value); });
      const bool stage_positive =
          Minimum(timing.q8_us) > 0.0 &&
          Minimum(timing.matvec_us) > 0.0 &&
          Minimum(timing.block_topk_us) > 0.0 &&
          Minimum(timing.merge_us) > 0.0 &&
          Minimum(timing.correction_us) > 0.0;
      const bool ids_valid = std::all_of(
          selected_ids.begin(), selected_ids.end(),
          [](std::int32_t id) {
            return id >= 0 && id < static_cast<std::int32_t>(kRows);
          });
      WriteF32(output_path, output);
      all_finite = all_finite && finite;
      all_stage_timestamps_positive =
          all_stage_timestamps_positive && stage_positive;
      all_selected_ids_valid = all_selected_ids_valid && ids_valid;
      phases.push_back({
          phase, std::move(timing), output_path, selected_ids, finite});
    }

    const bool pass =
        runtime.device_name().find("B390") != std::string::npos &&
        all_finite && all_stage_timestamps_positive && all_selected_ids_valid;
    std::cout << std::boolalpha << std::setprecision(12) << "{"
              << "\"device_name\":" << JsonString(runtime.device_name()) << ","
              << "\"weight_bytes\":" << kWeightBytes << ","
              << "\"scale_bytes\":" << kScaleBytes << ","
              << "\"f16_output_bytes\":" << kF16OutputBytes << ","
              << "\"mandatory_matvec_bytes\":" << kMandatoryMatvecBytes << ","
              << "\"rows\":" << kRows << ","
              << "\"columns\":" << kColumns << ","
              << "\"matvec_workgroups\":" << kMatvecWorkgroups << ","
              << "\"block_count\":" << kBlockCount << ","
              << "\"topk\":" << kTopK << ","
              << "\"timestamp_ns_per_tick\":"
              << runtime.timestamp_ns_per_tick() << ","
              << "\"phases\":[";
    for (std::size_t index = 0; index < phases.size(); ++index) {
      if (index != 0U) std::cout << ",";
      const auto& row = phases[index];
      const auto fallback_shell = AddSamples(
          AddSamples(row.timing.matvec_us, row.timing.block_topk_us),
          AddSamples(row.timing.merge_us, row.timing.correction_us));
      const auto full_shell = AddSamples(row.timing.q8_us, fallback_shell);
      const double matvec_median = Median(row.timing.matvec_us);
      std::cout << "{\"phase\":" << row.phase
                << ",\"finite\":" << row.finite
                << ",\"output\":" << JsonString(row.output_path.string())
                << ",\"selected_ids\":";
      WriteIntArray(row.selected_ids);
      std::cout << ",\"q8_min_us\":" << Minimum(row.timing.q8_us)
                << ",\"q8_median_us\":" << Median(row.timing.q8_us)
                << ",\"q8_mean_us\":" << Mean(row.timing.q8_us)
                << ",\"matvec_min_us\":" << Minimum(row.timing.matvec_us)
                << ",\"matvec_median_us\":" << matvec_median
                << ",\"matvec_mean_us\":" << Mean(row.timing.matvec_us)
                << ",\"block_topk_min_us\":"
                << Minimum(row.timing.block_topk_us)
                << ",\"block_topk_median_us\":"
                << Median(row.timing.block_topk_us)
                << ",\"block_topk_mean_us\":"
                << Mean(row.timing.block_topk_us)
                << ",\"merge_min_us\":" << Minimum(row.timing.merge_us)
                << ",\"merge_median_us\":" << Median(row.timing.merge_us)
                << ",\"merge_mean_us\":" << Mean(row.timing.merge_us)
                << ",\"correction_min_us\":"
                << Minimum(row.timing.correction_us)
                << ",\"correction_median_us\":"
                << Median(row.timing.correction_us)
                << ",\"correction_mean_us\":"
                << Mean(row.timing.correction_us)
                << ",\"fallback_shell_min_us\":" << Minimum(fallback_shell)
                << ",\"fallback_shell_median_us\":" << Median(fallback_shell)
                << ",\"fallback_shell_mean_us\":" << Mean(fallback_shell)
                << ",\"full_shell_min_us\":" << Minimum(full_shell)
                << ",\"full_shell_median_us\":" << Median(full_shell)
                << ",\"full_shell_mean_us\":" << Mean(full_shell)
                << ",\"wall_min_us\":" << Minimum(row.timing.wall_us)
                << ",\"wall_median_us\":" << Median(row.timing.wall_us)
                << ",\"wall_mean_us\":" << Mean(row.timing.wall_us)
                << ",\"matvec_median_gb_s\":"
                << kMandatoryMatvecBytes / (matvec_median * 1000.0)
                << ",\"q8_samples_us\":";
      WriteDoubleArray(row.timing.q8_us);
      std::cout << ",\"matvec_samples_us\":";
      WriteDoubleArray(row.timing.matvec_us);
      std::cout << ",\"block_topk_samples_us\":";
      WriteDoubleArray(row.timing.block_topk_us);
      std::cout << ",\"merge_samples_us\":";
      WriteDoubleArray(row.timing.merge_us);
      std::cout << ",\"correction_samples_us\":";
      WriteDoubleArray(row.timing.correction_us);
      std::cout << ",\"fallback_shell_samples_us\":";
      WriteDoubleArray(fallback_shell);
      std::cout << ",\"wall_samples_us\":";
      WriteDoubleArray(row.timing.wall_us);
      std::cout << "}";
    }
    std::cout << "],\"all_finite\":" << all_finite
              << ",\"all_stage_timestamps_positive\":"
              << all_stage_timestamps_positive
              << ",\"all_selected_ids_valid\":" << all_selected_ids_valid
              << ",\"required_checks_passed\":" << pass << "}"
              << std::endl;
    return pass ? 0 : 2;
  } catch (const std::exception& exception) {
    std::cerr << "iq36-openvino-lm-head-gated-exact-component: "
              << exception.what() << '\n';
    return 4;
  }
}
