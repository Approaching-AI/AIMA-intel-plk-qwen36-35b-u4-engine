#include "intel_qwen36/gguf_loader.hpp"
#include "intel_qwen36/gpu_q4x8_matvec.hpp"

#include <level_zero/ze_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr char kSelectedTensorName[] = "blk.7.ffn_down_exps.weight";
constexpr char kSharedTensorName[] = "blk.7.ffn_down_shexp.weight";
constexpr std::uint32_t kIntelVendorId = 0x8086U;
constexpr std::uint32_t kPtlDeviceId = 0xB080U;
constexpr std::uint32_t kRowsPerExpert = 2048;
constexpr std::uint32_t kBlocksPerRow = 2;
constexpr std::uint32_t kRowsPerTile = 32;
constexpr std::uint32_t kMaterialExpertCount = 256;
constexpr std::uint32_t kSelectedCount = 8;
constexpr std::uint32_t kGroupSize = 64;
constexpr std::uint64_t kSelectedActiveBytes =
    static_cast<std::uint64_t>(kSelectedCount) * kRowsPerExpert *
    kBlocksPerRow * 210U;
constexpr std::uint64_t kSharedActiveBytes =
    static_cast<std::uint64_t>(kRowsPerExpert) * kBlocksPerRow * 210U;
constexpr double kKernelRequiredGbS = 106.524608569878;

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

std::vector<std::uint8_t> ReadBinary(const std::string& path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  Require(static_cast<bool>(input), "failed to open native module");
  const auto size = input.tellg();
  Require(size > 0, "native module is empty");
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
  input.seekg(0, std::ios::beg);
  input.read(reinterpret_cast<char*>(bytes.data()), size);
  Require(static_cast<bool>(input), "failed to read native module");
  return bytes;
}

std::vector<std::uint8_t> ReadTensorBytes(
    const std::string& model_path,
    const iq36::GgufTensorInfo& tensor) {
  std::ifstream input(model_path, std::ios::binary);
  Require(static_cast<bool>(input), "failed to open locked GGUF");
  input.seekg(static_cast<std::streamoff>(tensor.absolute_offset));
  Require(static_cast<bool>(input), "failed to seek locked GGUF tensor");
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(tensor.nbytes));
  input.read(reinterpret_cast<char*>(bytes.data()),
             static_cast<std::streamsize>(bytes.size()));
  Require(static_cast<bool>(input), "failed to read locked GGUF tensor");
  return bytes;
}

std::vector<float> MakeInput(std::uint32_t cols, float phase) {
  std::vector<float> input(cols, 0.0f);
  for (std::uint32_t i = 0; i < cols; ++i) {
    const float a = std::sin((static_cast<float>(i) + phase) * 0.013f) * 0.75f;
    const float b =
        std::cos((static_cast<float>(i % 17U) + phase) * 0.11f) * 0.15f;
    input[i] = a + b;
  }
  return input;
}

struct Comparison {
  bool same_size = false;
  bool finite = false;
  double cosine = 0.0;
  double max_abs_diff = 0.0;
  double relative_l2 = 0.0;
};

Comparison Compare(const std::vector<float>& observed,
                   const std::vector<float>& reference) {
  Comparison result;
  result.same_size = observed.size() == reference.size();
  if (!result.same_size || observed.empty()) return result;
  long double dot = 0.0;
  long double observed_l2 = 0.0;
  long double reference_l2 = 0.0;
  long double difference_l2 = 0.0;
  result.finite = true;
  for (std::size_t i = 0; i < observed.size(); ++i) {
    const double lhs = observed[i];
    const double rhs = reference[i];
    result.finite = result.finite && std::isfinite(lhs) && std::isfinite(rhs);
    const double difference = lhs - rhs;
    result.max_abs_diff = std::max(result.max_abs_diff, std::abs(difference));
    dot += static_cast<long double>(lhs) * rhs;
    observed_l2 += static_cast<long double>(lhs) * lhs;
    reference_l2 += static_cast<long double>(rhs) * rhs;
    difference_l2 += static_cast<long double>(difference) * difference;
  }
  if (observed_l2 > 0.0 && reference_l2 > 0.0) {
    result.cosine = static_cast<double>(
        dot / std::sqrt(observed_l2 * reference_l2));
  }
  if (reference_l2 > 0.0) {
    result.relative_l2 = static_cast<double>(
        std::sqrt(difference_l2 / reference_l2));
  }
  return result;
}

void WriteComparison(const Comparison& value) {
  std::cout << "{\"cosine\":" << value.cosine
            << ",\"finite\":" << value.finite
            << ",\"max_abs_diff\":" << value.max_abs_diff
            << ",\"relative_l2\":" << value.relative_l2
            << ",\"same_size\":" << value.same_size << "}";
}

double Minimum(const std::vector<double>& values) {
  Require(!values.empty(), "timing vector is empty");
  return *std::min_element(values.begin(), values.end());
}

double Mean(const std::vector<double>& values) {
  Require(!values.empty(), "timing vector is empty");
  double sum = 0.0;
  for (double value : values) sum += value;
  return sum / static_cast<double>(values.size());
}

void WriteDoubleArray(const std::vector<double>& values) {
  std::cout << "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0) std::cout << ",";
    std::cout << values[i];
  }
  std::cout << "]";
}

std::uint64_t TimestampDelta(std::uint64_t start,
                             std::uint64_t end,
                             std::uint32_t valid_bits) {
  if (valid_bits == 0 || valid_bits >= 64) return end - start;
  const std::uint64_t mask = (std::uint64_t{1} << valid_bits) - 1;
  return (end - start) & mask;
}

template <typename Value>
std::vector<Value> RepeatVector(const std::vector<Value>& values,
                                std::size_t repeat) {
  std::vector<Value> result;
  result.reserve(values.size() * repeat);
  for (std::size_t i = 0; i < repeat; ++i) {
    result.insert(result.end(), values.begin(), values.end());
  }
  return result;
}

class IndexedQ6Runtime {
 public:
  IndexedQ6Runtime(const std::string& module_path,
                   const iq36::PackedQ6KRowstripe& selected_weights,
                   const iq36::PackedQ6KRowstripe& shared_weights,
                   const std::vector<std::uint32_t>& selected_positions,
                   const iq36::GpuQ8KInputPlanes& selected_q8,
                   const iq36::GpuQ8KInputPlanes& shared_q8)
      : selected_resident_bytes_(selected_weights.bytes.size()),
        shared_resident_bytes_(shared_weights.bytes.size()) {
    Require(selected_positions.size() == kSelectedCount,
            "selected position count mismatch");
    Require(selected_weights.rows_per_tile == kRowsPerTile &&
                shared_weights.rows_per_tile == kRowsPerTile,
            "rowstripe tile mismatch");
    InitializeDevice();
    InitializeRuntime(module_path);
    selected_weights_ = Upload(selected_weights.bytes.data(),
                               selected_weights.bytes.size());
    shared_weights_ = Upload(shared_weights.bytes.data(),
                             shared_weights.bytes.size());
    selected_positions_ = Upload(selected_positions.data(),
        selected_positions.size() * sizeof(std::uint32_t));
    const auto repeated_qs = RepeatVector(selected_q8.qs, kSelectedCount);
    const auto repeated_d = RepeatVector(selected_q8.d, kSelectedCount);
    selected_qs_ = Upload(repeated_qs.data(),
                          repeated_qs.size() * sizeof(std::int8_t));
    selected_d_ = Upload(repeated_d.data(),
                         repeated_d.size() * sizeof(float));
    shared_qs_ = Upload(shared_q8.qs.data(),
                        shared_q8.qs.size() * sizeof(std::int8_t));
    shared_d_ = Upload(shared_q8.d.data(),
                       shared_q8.d.size() * sizeof(float));
    selected_output_ = AllocateShared(
        static_cast<std::size_t>(kSelectedCount) * kRowsPerExpert *
        sizeof(float));
    shared_output_ = AllocateShared(
        static_cast<std::size_t>(kRowsPerExpert) * sizeof(float));
    FinishUploads();
    Record();
  }

  ~IndexedQ6Runtime() { Cleanup(); }

  struct Run {
    std::vector<double> kernel_us;
    std::vector<double> submit_us;
    std::vector<double> wall_us;
  };

  Run Execute(int warmup, int samples) {
    Require(warmup >= 0 && samples > 0, "invalid sample counts");
    Run run;
    for (int sample = -warmup; sample < samples; ++sample) {
      const auto begin = std::chrono::steady_clock::now();
      Check(zeCommandQueueExecuteCommandLists(queue_, 1, &command_list_, nullptr),
            "zeCommandQueueExecuteCommandLists");
      const auto submit_end = std::chrono::steady_clock::now();
      Check(zeCommandQueueSynchronize(queue_, UINT64_MAX),
            "zeCommandQueueSynchronize");
      const auto end = std::chrono::steady_clock::now();
      ze_kernel_timestamp_result_t timestamp{};
      Check(zeEventQueryKernelTimestamp(event_, &timestamp),
            "zeEventQueryKernelTimestamp");
      const auto ticks = TimestampDelta(timestamp.context.kernelStart,
                                        timestamp.context.kernelEnd,
                                        properties_.kernelTimestampValidBits);
      Check(zeEventHostReset(event_), "zeEventHostReset");
      if (sample < 0) continue;
      run.kernel_us.push_back(ticks * timestamp_ns_per_tick_ / 1000.0);
      run.submit_us.push_back(std::chrono::duration<double, std::micro>(
          submit_end - begin).count());
      run.wall_us.push_back(std::chrono::duration<double, std::micro>(
          end - begin).count());
    }
    return run;
  }

  std::vector<float> SelectedOutput() const {
    const auto* values = static_cast<const float*>(selected_output_);
    return {values, values + kSelectedCount * kRowsPerExpert};
  }

  std::vector<float> SharedOutput() const {
    const auto* values = static_cast<const float*>(shared_output_);
    return {values, values + kRowsPerExpert};
  }

  const std::string& device_name() const { return device_name_; }
  std::uint64_t selected_resident_bytes() const {
    return selected_resident_bytes_;
  }
  std::uint64_t shared_resident_bytes() const {
    return shared_resident_bytes_;
  }
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
            properties.deviceId != kPtlDeviceId) continue;
        std::uint32_t group_count = 0;
        Check(zeDeviceGetCommandQueueGroupProperties(device, &group_count,
                                                      nullptr),
              "zeDeviceGetCommandQueueGroupProperties(count)");
        std::vector<ze_command_queue_group_properties_t> groups(group_count);
        for (auto& group : groups) {
          group.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES;
        }
        Check(zeDeviceGetCommandQueueGroupProperties(device, &group_count,
                                                      groups.data()),
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
    const auto ticks = TimestampDelta(device0, device1,
                                      properties_.kernelTimestampValidBits);
    Require(host1 > host0 && ticks > 0, "timestamp calibration failed");
    timestamp_ns_per_tick_ =
        static_cast<double>(host1 - host0) / static_cast<double>(ticks);
  }

  void InitializeRuntime(const std::string& module_path) {
    ze_context_desc_t context_desc{ZE_STRUCTURE_TYPE_CONTEXT_DESC};
    Check(zeContextCreate(driver_, &context_desc, &context_),
          "zeContextCreate");
    ze_command_queue_desc_t queue_desc{ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC};
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
    ze_event_pool_desc_t pool_desc{ZE_STRUCTURE_TYPE_EVENT_POOL_DESC};
    pool_desc.flags =
        ZE_EVENT_POOL_FLAG_HOST_VISIBLE | ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP;
    pool_desc.count = 1;
    Check(zeEventPoolCreate(context_, &pool_desc, 1, &device_, &event_pool_),
          "zeEventPoolCreate");
    ze_event_desc_t event_desc{ZE_STRUCTURE_TYPE_EVENT_DESC};
    event_desc.index = 0;
    event_desc.signal = ZE_EVENT_SCOPE_FLAG_HOST;
    event_desc.wait = ZE_EVENT_SCOPE_FLAG_HOST;
    Check(zeEventCreate(event_pool_, &event_desc, &event_), "zeEventCreate");
    module_bytes_ = ReadBinary(module_path);
    ze_module_desc_t module_desc{ZE_STRUCTURE_TYPE_MODULE_DESC};
    module_desc.format = ZE_MODULE_FORMAT_NATIVE;
    module_desc.inputSize = module_bytes_.size();
    module_desc.pInputModule = module_bytes_.data();
    module_desc.pBuildFlags = "";
    ze_module_build_log_handle_t log = nullptr;
    const auto result = zeModuleCreate(context_, device_, &module_desc,
                                       &module_, &log);
    if (log != nullptr) zeModuleBuildLogDestroy(log);
    Check(result, "zeModuleCreate");
  }

  void* Upload(const void* source, std::size_t bytes) {
    Require(source != nullptr && bytes > 0, "empty upload");
    ze_device_mem_alloc_desc_t device_desc{
        ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC};
    void* destination = nullptr;
    Check(zeMemAllocDevice(context_, &device_desc, bytes, 64, device_,
                           &destination),
          "zeMemAllocDevice");
    device_allocations_.push_back(destination);
    ze_host_mem_alloc_desc_t host_desc{ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC};
    void* staging = nullptr;
    Check(zeMemAllocHost(context_, &host_desc, bytes, 64, &staging),
          "zeMemAllocHost");
    std::memcpy(staging, source, bytes);
    staging_allocations_.push_back(staging);
    Check(zeCommandListAppendMemoryCopy(upload_list_, destination, staging,
                                        bytes, nullptr, 0, nullptr),
          "zeCommandListAppendMemoryCopy");
    return destination;
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

  void FinishUploads() {
    Check(zeCommandListClose(upload_list_), "zeCommandListClose(upload)");
    Check(zeCommandQueueExecuteCommandLists(queue_, 1, &upload_list_, nullptr),
          "zeCommandQueueExecuteCommandLists(upload)");
    Check(zeCommandQueueSynchronize(queue_, UINT64_MAX),
          "zeCommandQueueSynchronize(upload)");
    for (void* pointer : staging_allocations_) zeMemFree(context_, pointer);
    staging_allocations_.clear();
  }

  void SetPointerArg(std::uint32_t index, void* pointer) {
    Check(zeKernelSetArgumentValue(kernel_, index, sizeof(pointer), &pointer),
          "zeKernelSetArgumentValue(pointer)");
  }

  template <typename Value>
  void SetValueArg(std::uint32_t index, const Value& value) {
    Check(zeKernelSetArgumentValue(kernel_, index, sizeof(value), &value),
          "zeKernelSetArgumentValue(value)");
  }

  void Record() {
    ze_kernel_desc_t kernel_desc{ZE_STRUCTURE_TYPE_KERNEL_DESC};
    kernel_desc.pKernelName =
        "q6k_all_expert_rowstripe_coalesced_topk8_plus_shared";
    Check(zeKernelCreate(module_, &kernel_desc, &kernel_), "zeKernelCreate");
    Check(zeKernelSetGroupSize(kernel_, kGroupSize, 1, 1),
          "zeKernelSetGroupSize");
    SetPointerArg(0, selected_weights_);
    SetPointerArg(1, shared_weights_);
    SetPointerArg(2, selected_positions_);
    SetPointerArg(3, selected_qs_);
    SetPointerArg(4, selected_d_);
    SetPointerArg(5, shared_qs_);
    SetPointerArg(6, shared_d_);
    SetPointerArg(7, selected_output_);
    SetPointerArg(8, shared_output_);
    constexpr std::uint32_t kTotalRows =
        (kSelectedCount + 1U) * kRowsPerExpert;
    static_assert(kTotalRows % kGroupSize == 0);
    ze_group_count_t groups{kTotalRows / kGroupSize, 1, 1};
    Check(zeCommandListAppendLaunchKernel(command_list_, kernel_, &groups,
                                          event_, 0, nullptr),
          "zeCommandListAppendLaunchKernel");
    Check(zeCommandListClose(command_list_), "zeCommandListClose(run)");
  }

  void Cleanup() {
    if (queue_ != nullptr) zeCommandQueueSynchronize(queue_, UINT64_MAX);
    if (kernel_ != nullptr) zeKernelDestroy(kernel_);
    if (event_ != nullptr) zeEventDestroy(event_);
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
  ze_kernel_handle_t kernel_ = nullptr;
  ze_event_pool_handle_t event_pool_ = nullptr;
  ze_event_handle_t event_ = nullptr;
  ze_device_properties_t properties_{ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES};
  std::vector<std::uint8_t> module_bytes_;
  std::vector<void*> device_allocations_;
  std::vector<void*> shared_allocations_;
  std::vector<void*> staging_allocations_;
  void* selected_weights_ = nullptr;
  void* shared_weights_ = nullptr;
  void* selected_positions_ = nullptr;
  void* selected_qs_ = nullptr;
  void* selected_d_ = nullptr;
  void* shared_qs_ = nullptr;
  void* shared_d_ = nullptr;
  void* selected_output_ = nullptr;
  void* shared_output_ = nullptr;
  std::string device_name_;
  std::uint32_t queue_ordinal_ = UINT32_MAX;
  std::uint64_t selected_resident_bytes_ = 0;
  std::uint64_t shared_resident_bytes_ = 0;
  double timestamp_ns_per_tick_ = 0.0;
};

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 5) {
      throw std::invalid_argument(
          "usage: iq36-level-zero-indexed-q6-down-smoke MODEL MODULE "
          "WARMUP SAMPLES");
    }
    const std::string model_path = argv[1];
    const int warmup = std::stoi(argv[3]);
    const int samples = std::stoi(argv[4]);
    const auto index = iq36::parse_gguf_model_index(model_path);
    const auto* selected_tensor = iq36::find_tensor(index, kSelectedTensorName);
    const auto* shared_tensor = iq36::find_tensor(index, kSharedTensorName);
    Require(selected_tensor != nullptr && shared_tensor != nullptr,
            "locked Q6 down tensor missing");
    Require(selected_tensor->type == 14 &&
                selected_tensor->dims ==
                    std::vector<std::uint64_t>{512, 2048, 256},
            "locked selected Q6 down shape mismatch");
    Require(shared_tensor->type == 14 &&
                shared_tensor->dims ==
                    std::vector<std::uint64_t>{512, 2048},
            "locked shared Q6 down shape mismatch");
    const std::vector<std::uint32_t> selected_positions{
        0, 1, 7, 31, 63, 127, 191, 255};
    std::vector<std::int32_t> oracle_positions(
        selected_positions.begin(), selected_positions.end());
    const auto selected_input = MakeInput(512, 1.0f);
    const auto shared_input = MakeInput(512, 5.0f);
    const auto selected_q8 = iq36::QuantizeQ8KInputPlanes(selected_input);
    const auto shared_q8 = iq36::QuantizeQ8KInputPlanes(shared_input);
    auto selected_raw = ReadTensorBytes(model_path, *selected_tensor);
    auto shared_raw = ReadTensorBytes(model_path, *shared_tensor);
    auto selected_packed = iq36::PackQ6KRowstripeCoalesced(
        selected_raw, kRowsPerExpert, kBlocksPerRow,
        kMaterialExpertCount, kRowsPerTile);
    auto shared_packed = iq36::PackQ6KRowstripeCoalesced(
        shared_raw, kRowsPerExpert, kBlocksPerRow, 1, kRowsPerTile);
    selected_raw.clear();
    selected_raw.shrink_to_fit();
    shared_raw.clear();
    shared_raw.shrink_to_fit();

    IndexedQ6Runtime runtime(argv[2], selected_packed, shared_packed,
                             selected_positions, selected_q8, shared_q8);
    selected_packed.bytes.clear();
    selected_packed.bytes.shrink_to_fit();
    shared_packed.bytes.clear();
    shared_packed.bytes.shrink_to_fit();
    const auto run = runtime.Execute(warmup, samples);
    const auto selected_output = runtime.SelectedOutput();
    const auto shared_output = runtime.SharedOutput();
    iq36::set_expert_slice_matvec_enabled(true);
    iq36::set_expert_slice_matvec_thread_count(16);
    const auto selected_reference = iq36::matvec_expert_tensor(
        model_path, index, kSelectedTensorName, selected_input,
        oracle_positions);
    const auto shared_reference = iq36::matvec_tensor(
        model_path, index, kSharedTensorName, shared_input);
    const auto selected_comparison =
        Compare(selected_output, selected_reference);
    const auto shared_comparison = Compare(shared_output, shared_reference);
    const double kernel_min_us = Minimum(run.kernel_us);
    constexpr std::uint64_t kActiveBytes =
        kSelectedActiveBytes + kSharedActiveBytes;
    const double effective_gb_s = kActiveBytes / (kernel_min_us * 1000.0);
    const bool selected_correct = selected_comparison.same_size &&
        selected_comparison.finite && selected_comparison.cosine >= 0.999 &&
        selected_comparison.relative_l2 <= 0.002;
    const bool shared_correct = shared_comparison.same_size &&
        shared_comparison.finite && shared_comparison.cosine >= 0.999 &&
        shared_comparison.relative_l2 <= 0.002;
    const bool pass = runtime.device_name().find("B390") != std::string::npos &&
        selected_correct && shared_correct &&
        effective_gb_s >= kKernelRequiredGbS &&
        Minimum(run.submit_us) <= 100.0;

    std::cout << std::boolalpha << std::setprecision(12) << "{"
              << "\"active_bytes\":" << kActiveBytes << ","
              << "\"device_name\":\"" << runtime.device_name() << "\","
              << "\"effective_gb_s\":" << effective_gb_s << ","
              << "\"group_size\":" << kGroupSize << ","
              << "\"kernel_mean_us\":" << Mean(run.kernel_us) << ","
              << "\"kernel_min_us\":" << kernel_min_us << ","
              << "\"kernel_required_gb_s\":" << kKernelRequiredGbS << ","
              << "\"kernel_samples_us\":";
    WriteDoubleArray(run.kernel_us);
    std::cout << ",\"required_checks_passed\":" << pass << ","
              << "\"selected_active_bytes\":" << kSelectedActiveBytes << ","
              << "\"selected_comparison\":";
    WriteComparison(selected_comparison);
    std::cout << ",\"selected_positions\":[0,1,7,31,63,127,191,255],"
              << "\"selected_resident_bytes\":"
              << runtime.selected_resident_bytes() << ","
              << "\"shared_active_bytes\":" << kSharedActiveBytes << ","
              << "\"shared_comparison\":";
    WriteComparison(shared_comparison);
    std::cout << ",\"shared_resident_bytes\":"
              << runtime.shared_resident_bytes() << ","
              << "\"submit_mean_us\":" << Mean(run.submit_us) << ","
              << "\"submit_min_us\":" << Minimum(run.submit_us) << ","
              << "\"timestamp_ns_per_tick\":"
              << runtime.timestamp_ns_per_tick() << ","
              << "\"wall_mean_us\":" << Mean(run.wall_us) << ","
              << "\"wall_min_us\":" << Minimum(run.wall_us) << "}"
              << std::endl;
    return pass ? 0 : 2;
  } catch (const std::exception& exception) {
    std::cerr << "iq36-level-zero-indexed-q6-down-smoke: "
              << exception.what() << '\n';
    return 4;
  }
}
