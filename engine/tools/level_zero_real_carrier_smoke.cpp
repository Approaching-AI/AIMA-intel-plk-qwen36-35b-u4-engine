#include "intel_qwen36/gguf_loader.hpp"
#include "intel_qwen36/gpu_q4x8_matvec.hpp"
#include "intel_qwen36/packed_token_schedule.hpp"

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
#include <utility>
#include <vector>

namespace {

constexpr char kQ4TensorName[] = "blk.5.ffn_gate_up_exps.weight";
constexpr char kQ6TensorName[] = "blk.7.ffn_down_exps.weight";
constexpr std::uint32_t kIntelVendorId = 0x8086U;
constexpr std::uint32_t kPtlDeviceId = 0xB080U;

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

std::vector<float> MakeInput(std::uint64_t cols) {
  std::vector<float> input(static_cast<std::size_t>(cols), 0.0f);
  for (std::uint64_t i = 0; i < cols; ++i) {
    const float a = std::sin(static_cast<float>(i + 1) * 0.013f) * 0.75f;
    const float b =
        std::cos(static_cast<float>((i % 17) + 3) * 0.11f) * 0.15f;
    input[static_cast<std::size_t>(i)] = a + b;
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
    result.finite = result.finite &&
                    std::isfinite(lhs) && std::isfinite(rhs);
    const double difference = lhs - rhs;
    result.max_abs_diff =
        std::max(result.max_abs_diff, std::abs(difference));
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

std::vector<double> AddSamples(const std::vector<double>& lhs,
                               const std::vector<double>& rhs) {
  Require(lhs.size() == rhs.size(), "paired timing size mismatch");
  std::vector<double> sums(lhs.size());
  for (std::size_t i = 0; i < lhs.size(); ++i) sums[i] = lhs[i] + rhs[i];
  return sums;
}

std::uint64_t TimestampDelta(std::uint64_t start,
                             std::uint64_t end,
                             std::uint32_t valid_bits) {
  if (valid_bits == 0 || valid_bits >= 64) return end - start;
  const std::uint64_t mask = (std::uint64_t{1} << valid_bits) - 1;
  return (end - start) & mask;
}

class RealCarrierRuntime {
 public:
  RealCarrierRuntime(const std::string& module_path,
                     const std::vector<std::uint8_t>& q4_packed,
                     const iq36::GpuQ8KInputPlanes& q4_q8,
                     std::uint32_t q4_rows,
                     std::uint32_t q4_blocks_per_row,
                     const std::vector<std::uint8_t>& q6_rowstripe,
                     const iq36::GpuQ8KInputPlanes& q6_q8,
                     std::uint32_t q6_rows,
                     std::uint32_t q6_blocks_per_row,
                     std::uint32_t q6_rows_per_tile)
      : q4_bytes_(q4_packed.size()), q6_bytes_(q6_rowstripe.size()),
        q4_rows_(q4_rows), q6_rows_(q6_rows) {
    InitializeDevice();
    InitializeRuntime(module_path);
    q4_weight_ = Upload(q4_packed.data(), q4_packed.size());
    q4_qs_ = Upload(q4_q8.qs.data(), q4_q8.qs.size() * sizeof(std::int8_t));
    q4_bsums_ = Upload(
        q4_q8.bsums.data(), q4_q8.bsums.size() * sizeof(std::int16_t));
    q4_d_ = Upload(q4_q8.d.data(), q4_q8.d.size() * sizeof(float));
    q6_weight_ = Upload(q6_rowstripe.data(), q6_rowstripe.size());
    q6_qs_ = Upload(q6_q8.qs.data(), q6_q8.qs.size() * sizeof(std::int8_t));
    q6_d_ = Upload(q6_q8.d.data(), q6_q8.d.size() * sizeof(float));
    q4_output_ = AllocateShared(
        static_cast<std::size_t>(q4_rows) * sizeof(float));
    q6_output_ = AllocateShared(
        static_cast<std::size_t>(q6_rows) * sizeof(float));
    timestamps_ = static_cast<std::uint64_t*>(
        AllocateShared(3 * sizeof(std::uint64_t)));
    FinishUploads();
    Record(q4_blocks_per_row, q6_blocks_per_row, q6_rows_per_tile);
  }

  ~RealCarrierRuntime() { Cleanup(); }

  struct Run {
    std::vector<double> q4_us;
    std::vector<double> q6_us;
    std::vector<double> submit_us;
    std::vector<double> wall_us;
  };

  Run Execute(int warmup, int samples) {
    Require(warmup >= 0 && samples > 0, "invalid sample counts");
    Run run;
    for (int sample = -warmup; sample < samples; ++sample) {
      const auto begin = std::chrono::steady_clock::now();
      Check(zeCommandQueueExecuteCommandLists(
                queue_, 1, &command_list_, nullptr),
            "zeCommandQueueExecuteCommandLists");
      const auto submit_end = std::chrono::steady_clock::now();
      Check(zeCommandQueueSynchronize(queue_, UINT64_MAX),
            "zeCommandQueueSynchronize");
      const auto end = std::chrono::steady_clock::now();
      ze_kernel_timestamp_result_t q4_timestamp{};
      ze_kernel_timestamp_result_t q6_timestamp{};
      Check(zeEventQueryKernelTimestamp(q4_event_, &q4_timestamp),
            "zeEventQueryKernelTimestamp(q4)");
      Check(zeEventQueryKernelTimestamp(q6_event_, &q6_timestamp),
            "zeEventQueryKernelTimestamp(q6)");
      const auto q4_ticks = TimestampDelta(
          q4_timestamp.context.kernelStart,
          q4_timestamp.context.kernelEnd,
          properties_.kernelTimestampValidBits);
      const auto q6_ticks = TimestampDelta(
          q6_timestamp.context.kernelStart,
          q6_timestamp.context.kernelEnd,
          properties_.kernelTimestampValidBits);
      Check(zeEventHostReset(q4_event_), "zeEventHostReset(q4)");
      Check(zeEventHostReset(q6_event_), "zeEventHostReset(q6)");
      if (sample < 0) continue;
      run.q4_us.push_back(q4_ticks * timestamp_ns_per_tick_ / 1000.0);
      run.q6_us.push_back(q6_ticks * timestamp_ns_per_tick_ / 1000.0);
      run.submit_us.push_back(std::chrono::duration<double, std::micro>(
                                  submit_end - begin).count());
      run.wall_us.push_back(std::chrono::duration<double, std::micro>(
                                end - begin).count());
    }
    return run;
  }

  std::vector<float> Q4Output() const {
    const auto* values = static_cast<const float*>(q4_output_);
    return {values, values + q4_rows_};
  }

  std::vector<float> Q6Output() const {
    const auto* values = static_cast<const float*>(q6_output_);
    return {values, values + q6_rows_};
  }

  const std::string& device_name() const { return device_name_; }
  std::uint32_t q4_group_size() const { return q4_group_size_; }
  std::uint32_t q6_group_size() const { return q6_group_size_; }
  std::uint64_t q4_bytes() const { return q4_bytes_; }
  std::uint64_t q6_bytes() const { return q6_bytes_; }
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
    Require(host1 > host0 && ticks > 0, "timestamp calibration failed");
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
    event_pool_desc.count = 2;
    Check(zeEventPoolCreate(
              context_, &event_pool_desc, 1, &device_, &event_pool_),
          "zeEventPoolCreate");
    ze_event_desc_t event_desc{ZE_STRUCTURE_TYPE_EVENT_DESC};
    event_desc.signal = ZE_EVENT_SCOPE_FLAG_HOST;
    event_desc.wait = ZE_EVENT_SCOPE_FLAG_HOST;
    event_desc.index = 0;
    Check(zeEventCreate(event_pool_, &event_desc, &q4_event_),
          "zeEventCreate(q4)");
    event_desc.index = 1;
    Check(zeEventCreate(event_pool_, &event_desc, &q6_event_),
          "zeEventCreate(q6)");
    module_bytes_ = ReadBinary(module_path);
    ze_module_desc_t desc{ZE_STRUCTURE_TYPE_MODULE_DESC};
    desc.format = ZE_MODULE_FORMAT_NATIVE;
    desc.inputSize = module_bytes_.size();
    desc.pInputModule = module_bytes_.data();
    desc.pBuildFlags = "";
    ze_module_build_log_handle_t log = nullptr;
    const auto result = zeModuleCreate(
        context_, device_, &desc, &module_, &log);
    if (log != nullptr) zeModuleBuildLogDestroy(log);
    Check(result, "zeModuleCreate");
  }

  void* Upload(const void* source, std::size_t bytes) {
    Require(source != nullptr && bytes > 0, "empty carrier upload");
    ze_device_mem_alloc_desc_t device_desc{
        ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC};
    void* device_pointer = nullptr;
    Check(zeMemAllocDevice(context_, &device_desc, bytes, 64, device_,
                           &device_pointer),
          "zeMemAllocDevice");
    device_allocations_.push_back(device_pointer);
    ze_host_mem_alloc_desc_t host_desc{ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC};
    void* staging = nullptr;
    Check(zeMemAllocHost(context_, &host_desc, bytes, 64, &staging),
          "zeMemAllocHost");
    std::memcpy(staging, source, bytes);
    staging_allocations_.push_back(staging);
    Check(zeCommandListAppendMemoryCopy(
              upload_list_, device_pointer, staging, bytes, nullptr, 0, nullptr),
          "zeCommandListAppendMemoryCopy");
    return device_pointer;
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

  ze_kernel_handle_t CreateKernel(const char* name) {
    ze_kernel_desc_t desc{ZE_STRUCTURE_TYPE_KERNEL_DESC};
    desc.pKernelName = name;
    ze_kernel_handle_t kernel = nullptr;
    Check(zeKernelCreate(module_, &desc, &kernel), "zeKernelCreate");
    return kernel;
  }

  void SetPointerArg(ze_kernel_handle_t kernel,
                     std::uint32_t index,
                     void* pointer) {
    Check(zeKernelSetArgumentValue(kernel, index, sizeof(pointer), &pointer),
          "zeKernelSetArgumentValue(pointer)");
  }

  template <typename Value>
  void SetValueArg(ze_kernel_handle_t kernel,
                   std::uint32_t index,
                   const Value& value) {
    Check(zeKernelSetArgumentValue(kernel, index, sizeof(value), &value),
          "zeKernelSetArgumentValue(value)");
  }

  void Record(std::uint32_t q4_blocks_per_row,
              std::uint32_t q6_blocks_per_row,
              std::uint32_t q6_rows_per_tile) {
    q4_kernel_ = CreateKernel("q4k_x8_matvec_rowlane");
    q6_kernel_ = CreateKernel("q6k_selected_down_matvec_rowstripe");
    std::uint32_t suggested_y = 0;
    std::uint32_t suggested_z = 0;
    Check(zeKernelSuggestGroupSize(
              q4_kernel_, q4_rows_, 1, 1, &q4_group_size_,
              &suggested_y, &suggested_z),
          "zeKernelSuggestGroupSize(q4)");
    Require(q4_group_size_ > 0 && q4_rows_ % q4_group_size_ == 0,
            "suggested Q4 group size is incompatible");
    Check(zeKernelSetGroupSize(q4_kernel_, q4_group_size_, 1, 1),
          "zeKernelSetGroupSize(q4)");
    q6_group_size_ = 128;
    Require(q6_rows_ % q6_group_size_ == 0,
            "Q6 group size is incompatible");
    Check(zeKernelSetGroupSize(q6_kernel_, q6_group_size_, 1, 1),
          "zeKernelSetGroupSize(q6)");

    const std::uint32_t q4_row_groups = q4_rows_ / 8;
    SetPointerArg(q4_kernel_, 0, q4_weight_);
    SetPointerArg(q4_kernel_, 1, q4_qs_);
    SetPointerArg(q4_kernel_, 2, q4_bsums_);
    SetPointerArg(q4_kernel_, 3, q4_d_);
    SetValueArg(q4_kernel_, 4, q4_blocks_per_row);
    SetValueArg(q4_kernel_, 5, q4_row_groups);
    SetPointerArg(q4_kernel_, 6, q4_output_);
    SetPointerArg(q6_kernel_, 0, q6_weight_);
    SetPointerArg(q6_kernel_, 1, q6_qs_);
    SetPointerArg(q6_kernel_, 2, q6_d_);
    SetValueArg(q6_kernel_, 3, q6_rows_);
    SetValueArg(q6_kernel_, 4, q6_blocks_per_row);
    SetValueArg(q6_kernel_, 5, q6_rows_per_tile);
    SetPointerArg(q6_kernel_, 6, q6_output_);

    Check(zeCommandListAppendWriteGlobalTimestamp(
              command_list_, timestamps_, nullptr, 0, nullptr),
          "zeCommandListAppendWriteGlobalTimestamp(start)");
    ze_group_count_t q4_groups{q4_rows_ / q4_group_size_, 1, 1};
    Check(zeCommandListAppendLaunchKernel(
              command_list_, q4_kernel_, &q4_groups, q4_event_, 0, nullptr),
          "zeCommandListAppendLaunchKernel(q4)");
    Check(zeCommandListAppendBarrier(command_list_, nullptr, 0, nullptr),
          "zeCommandListAppendBarrier(q4)");
    Check(zeCommandListAppendWriteGlobalTimestamp(
              command_list_, timestamps_ + 1, nullptr, 0, nullptr),
          "zeCommandListAppendWriteGlobalTimestamp(mid)");
    ze_group_count_t q6_groups{q6_rows_ / q6_group_size_, 1, 1};
    Check(zeCommandListAppendLaunchKernel(
              command_list_, q6_kernel_, &q6_groups, q6_event_, 0, nullptr),
          "zeCommandListAppendLaunchKernel(q6)");
    Check(zeCommandListAppendBarrier(command_list_, nullptr, 0, nullptr),
          "zeCommandListAppendBarrier(q6)");
    Check(zeCommandListAppendWriteGlobalTimestamp(
              command_list_, timestamps_ + 2, nullptr, 0, nullptr),
          "zeCommandListAppendWriteGlobalTimestamp(end)");
    Check(zeCommandListClose(command_list_), "zeCommandListClose(run)");
  }

  void Cleanup() {
    if (queue_ != nullptr) zeCommandQueueSynchronize(queue_, UINT64_MAX);
    if (q6_kernel_ != nullptr) zeKernelDestroy(q6_kernel_);
    if (q4_kernel_ != nullptr) zeKernelDestroy(q4_kernel_);
    if (q6_event_ != nullptr) zeEventDestroy(q6_event_);
    if (q4_event_ != nullptr) zeEventDestroy(q4_event_);
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
  ze_event_handle_t q4_event_ = nullptr;
  ze_event_handle_t q6_event_ = nullptr;
  ze_kernel_handle_t q4_kernel_ = nullptr;
  ze_kernel_handle_t q6_kernel_ = nullptr;
  ze_device_properties_t properties_{ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES};
  std::vector<std::uint8_t> module_bytes_;
  std::vector<void*> device_allocations_;
  std::vector<void*> shared_allocations_;
  std::vector<void*> staging_allocations_;
  void* q4_weight_ = nullptr;
  void* q4_qs_ = nullptr;
  void* q4_bsums_ = nullptr;
  void* q4_d_ = nullptr;
  void* q4_output_ = nullptr;
  void* q6_weight_ = nullptr;
  void* q6_qs_ = nullptr;
  void* q6_d_ = nullptr;
  void* q6_output_ = nullptr;
  std::uint64_t* timestamps_ = nullptr;
  std::string device_name_;
  std::uint32_t queue_ordinal_ = UINT32_MAX;
  std::uint32_t q4_group_size_ = 0;
  std::uint32_t q6_group_size_ = 0;
  std::uint64_t q4_bytes_ = 0;
  std::uint64_t q6_bytes_ = 0;
  std::uint32_t q4_rows_ = 0;
  std::uint32_t q6_rows_ = 0;
  double timestamp_ns_per_tick_ = 0.0;
};

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 5) {
      throw std::invalid_argument(
          "usage: iq36-level-zero-real-carrier-smoke MODEL MODULE "
          "WARMUP SAMPLES");
    }
    const std::string model_path = argv[1];
    const int warmup = std::stoi(argv[3]);
    const int samples = std::stoi(argv[4]);
    const auto index = iq36::parse_gguf_model_index(model_path);
    const auto* q4_tensor = iq36::find_tensor(index, kQ4TensorName);
    const auto* q6_tensor = iq36::find_tensor(index, kQ6TensorName);
    Require(q4_tensor != nullptr && q6_tensor != nullptr,
            "locked carrier tensor missing");
    Require(q4_tensor->type == 12 &&
                q4_tensor->dims ==
                    std::vector<std::uint64_t>{2048, 1024, 256},
            "locked Q4 carrier shape mismatch");
    Require(q6_tensor->type == 14 &&
                q6_tensor->dims ==
                    std::vector<std::uint64_t>{512, 2048, 256},
            "locked Q6 carrier shape mismatch");
    const std::uint32_t q4_cols = 2048;
    const std::uint32_t q4_rows = 1024 * 256;
    const std::uint32_t q4_blocks = q4_cols / 256;
    const std::uint32_t q6_cols = 512;
    const std::uint32_t q6_rows = 2048 * 256;
    const std::uint32_t q6_blocks = q6_cols / 256;
    const auto q4_input = MakeInput(q4_cols);
    const auto q6_input = MakeInput(q6_cols);
    const auto q4_q8 = iq36::QuantizeQ8KInputPlanes(q4_input);
    const auto q6_q8 = iq36::QuantizeQ8KInputPlanes(q6_input);
    auto q4_raw = ReadTensorBytes(model_path, *q4_tensor);
    auto q6_raw = ReadTensorBytes(model_path, *q6_tensor);
    auto q4_packed = iq36::PackQ4Kx8(q4_raw, q4_rows, q4_blocks);
    auto q6_packed = iq36::PackQ6KRowstripe(
        q6_raw, q6_rows, q6_blocks, 1, 16);
    q4_raw.clear();
    q4_raw.shrink_to_fit();
    q6_raw.clear();
    q6_raw.shrink_to_fit();

    RealCarrierRuntime runtime(
        argv[2], q4_packed, q4_q8, q4_rows, q4_blocks,
        q6_packed.bytes, q6_q8, q6_rows, q6_blocks,
        static_cast<std::uint32_t>(q6_packed.rows_per_tile));
    q4_packed.clear();
    q4_packed.shrink_to_fit();
    q6_packed.bytes.clear();
    q6_packed.bytes.shrink_to_fit();
    const auto run = runtime.Execute(warmup, samples);
    const auto q4_output = runtime.Q4Output();
    const auto q6_output = runtime.Q6Output();
    const auto q4_reference = iq36::matvec_tensor(
        model_path, index, kQ4TensorName, q4_input);
    const auto q6_reference = iq36::matvec_tensor(
        model_path, index, kQ6TensorName, q6_input);
    const auto q4_comparison = Compare(q4_output, q4_reference);
    const auto q6_comparison = Compare(q6_output, q6_reference);
    const auto combined_kernel_samples = AddSamples(run.q4_us, run.q6_us);
    const double q4_min_us = Minimum(run.q4_us);
    const double q6_min_us = Minimum(run.q6_us);
    const double combined_kernel_min_us = Minimum(combined_kernel_samples);
    const double q4_gb_s = runtime.q4_bytes() / (q4_min_us * 1000.0);
    const double q6_gb_s = runtime.q6_bytes() / (q6_min_us * 1000.0);
    const double combined_gb_s =
        (runtime.q4_bytes() + runtime.q6_bytes()) /
        (combined_kernel_min_us * 1000.0);
    constexpr double kWallRequiredGbS = 105.99411601919999;
    constexpr double kIndividualNoiseFloorGbS = 105.46414543910399;
    constexpr double kKernelRequiredGbS = 106.524608569878;
    const bool q4_correct = q4_comparison.same_size &&
        q4_comparison.finite && q4_comparison.cosine >= 0.999 &&
        q4_comparison.relative_l2 <= 0.002;
    const bool q6_correct = q6_comparison.same_size &&
        q6_comparison.finite && q6_comparison.cosine >= 0.999 &&
        q6_comparison.relative_l2 <= 0.002;
    const bool pass = runtime.device_name().find("B390") != std::string::npos &&
        q4_correct && q6_correct && q4_gb_s >= kIndividualNoiseFloorGbS &&
        q6_gb_s >= kIndividualNoiseFloorGbS &&
        combined_gb_s >= kKernelRequiredGbS &&
        Minimum(run.submit_us) <= 100.0;

    std::cout << std::boolalpha << std::setprecision(12) << "{"
              << "\"combined_effective_gb_s\":" << combined_gb_s << ","
              << "\"combined_kernel_min_us\":"
              << combined_kernel_min_us << ","
              << "\"combined_kernel_samples_us\":";
    WriteDoubleArray(combined_kernel_samples);
    std::cout << ",\"combined_wall_mean_us\":" << Mean(run.wall_us) << ","
              << "\"combined_wall_min_us\":" << Minimum(run.wall_us) << ","
              << "\"device_name\":\"" << runtime.device_name() << "\","
              << "\"q4_bytes\":" << runtime.q4_bytes() << ","
              << "\"q4_comparison\":";
    WriteComparison(q4_comparison);
    std::cout << ",\"q4_effective_gb_s\":" << q4_gb_s << ","
              << "\"q4_group_size\":" << runtime.q4_group_size() << ","
              << "\"q4_mean_us\":" << Mean(run.q4_us) << ","
              << "\"q4_min_us\":" << q4_min_us << ","
              << "\"q4_samples_us\":";
    WriteDoubleArray(run.q4_us);
    std::cout << ",\"q6_bytes\":" << runtime.q6_bytes() << ","
              << "\"q6_comparison\":";
    WriteComparison(q6_comparison);
    std::cout << ",\"q6_effective_gb_s\":" << q6_gb_s << ","
              << "\"q6_group_size\":" << runtime.q6_group_size() << ","
              << "\"q6_mean_us\":" << Mean(run.q6_us) << ","
              << "\"q6_min_us\":" << q6_min_us << ","
              << "\"q6_samples_us\":";
    WriteDoubleArray(run.q6_us);
    std::cout << ",\"required_checks_passed\":" << pass << ","
              << "\"individual_noise_floor_gb_s\":"
              << kIndividualNoiseFloorGbS << ","
              << "\"kernel_required_gb_s\":" << kKernelRequiredGbS << ","
              << "\"wall_required_gb_s\":" << kWallRequiredGbS << ","
              << "\"submit_mean_us\":" << Mean(run.submit_us) << ","
              << "\"submit_min_us\":" << Minimum(run.submit_us) << ","
              << "\"timestamp_ns_per_tick\":"
              << runtime.timestamp_ns_per_tick() << "}" << std::endl;
    return pass ? 0 : 2;
  } catch (const std::exception& exception) {
    std::cerr << "iq36-level-zero-real-carrier-smoke: "
              << exception.what() << '\n';
    return 4;
  }
}
