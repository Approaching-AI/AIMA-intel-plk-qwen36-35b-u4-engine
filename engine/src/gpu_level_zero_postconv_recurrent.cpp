#include "intel_qwen36/gpu_level_zero_postconv_recurrent.hpp"

#include <level_zero/ze_api.h>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace iq36 {
namespace {

constexpr std::size_t kHeadDim = 128;
constexpr std::size_t kQueryHeads = 16;
constexpr std::size_t kValueHeads = 32;
constexpr std::size_t kQValues = kHeadDim * kQueryHeads;
constexpr std::size_t kVValues = kHeadDim * kValueHeads;
constexpr std::size_t kConvValues = 2 * kQValues + kVValues;
constexpr std::size_t kStateValues = kHeadDim * kHeadDim * kValueHeads;

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
  Require(static_cast<bool>(input), "failed to open Level Zero module: " + path);
  const std::streamoff bytes = input.tellg();
  Require(bytes > 0, "empty Level Zero module: " + path);
  std::vector<std::uint8_t> data(static_cast<std::size_t>(bytes));
  input.seekg(0, std::ios::beg);
  input.read(reinterpret_cast<char*>(data.data()), bytes);
  Require(static_cast<bool>(input), "failed to read Level Zero module: " + path);
  return data;
}

template <typename T>
std::size_t Bytes(const std::vector<T>& values) {
  return values.size() * sizeof(T);
}

}  // namespace

class GpuLevelZeroPostconvRecurrentRunner::Impl {
 public:
  Impl() = default;
  ~Impl() { Cleanup(); }

  void Initialize(const std::string& native_module_path,
                  std::uint32_t requested_device_id) {
    Check(zeInit(ZE_INIT_FLAG_GPU_ONLY), "zeInit");
    std::uint32_t driver_count = 0;
    Check(zeDriverGet(&driver_count, nullptr), "zeDriverGet(count)");
    Require(driver_count > 0, "no Level Zero drivers");
    std::vector<ze_driver_handle_t> drivers(driver_count);
    Check(zeDriverGet(&driver_count, drivers.data()), "zeDriverGet(list)");
    for (ze_driver_handle_t driver : drivers) {
      std::uint32_t device_count = 0;
      Check(zeDeviceGet(driver, &device_count, nullptr), "zeDeviceGet(count)");
      std::vector<ze_device_handle_t> devices(device_count);
      Check(zeDeviceGet(driver, &device_count, devices.data()),
            "zeDeviceGet(list)");
      for (ze_device_handle_t device : devices) {
        ze_device_properties_t properties{ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES};
        Check(zeDeviceGetProperties(device, &properties),
              "zeDeviceGetProperties");
        if (properties.vendorId != 0x8086U ||
            properties.deviceId != requested_device_id) {
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
            device_name_ = properties.name;
            break;
          }
        }
        if (device_ != nullptr) break;
      }
      if (device_ != nullptr) break;
    }
    Require(device_ != nullptr, "requested Level Zero device is unavailable");

    ze_context_desc_t context_desc{ZE_STRUCTURE_TYPE_CONTEXT_DESC};
    Check(zeContextCreate(driver_, &context_desc, &context_),
          "zeContextCreate");
    ze_command_queue_desc_t queue_desc{
        ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC};
    queue_desc.ordinal = queue_ordinal_;
    queue_desc.index = 0;
    queue_desc.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
    queue_desc.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
    Check(zeCommandQueueCreate(context_, device_, &queue_desc, &queue_),
          "zeCommandQueueCreate");
    ze_command_list_desc_t list_desc{ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC};
    list_desc.commandQueueGroupOrdinal = queue_ordinal_;
    Check(zeCommandListCreate(context_, device_, &list_desc, &command_list_),
          "zeCommandListCreate");

    const auto module_bytes = ReadBinary(native_module_path);
    ze_module_desc_t module_desc{ZE_STRUCTURE_TYPE_MODULE_DESC};
    module_desc.format = ZE_MODULE_FORMAT_NATIVE;
    module_desc.inputSize = module_bytes.size();
    module_desc.pInputModule = module_bytes.data();
    module_desc.pBuildFlags = "";
    ze_module_build_log_handle_t build_log = nullptr;
    const ze_result_t module_result = zeModuleCreate(
        context_, device_, &module_desc, &module_, &build_log);
    if (build_log != nullptr) zeModuleBuildLogDestroy(build_log);
    Check(module_result, "zeModuleCreate");
    postconv_kernel_ = CreateKernel("iq36_l0_postconv_cpuorder");
    recurrent_kernel_ = CreateKernel("iq36_l0_delta_recurrent_cpuorder");
    Check(zeKernelSetGroupSize(postconv_kernel_, 128, 1, 1),
          "zeKernelSetGroupSize(postconv)");
    Check(zeKernelSetGroupSize(recurrent_kernel_, 128, 1, 1),
          "zeKernelSetGroupSize(recurrent)");
  }

  const std::string& device_name() const { return device_name_; }

  GpuLevelZeroPostconvRecurrentRun Run(
      const GpuLevelZeroPostconvRecurrentInput& input,
      int samples) {
    ValidateInput(input, samples);
    std::vector<void*> allocations;
    allocations.reserve(12);
    auto allocate = [&](std::size_t bytes) {
      void* pointer = AllocateShared(bytes);
      allocations.push_back(pointer);
      return pointer;
    };
    void* conv = allocate(Bytes(input.conv_output_raw));
    void* q = allocate(kQValues * sizeof(float));
    void* k = allocate(kQValues * sizeof(float));
    void* v = allocate(kVValues * sizeof(float));
    void* decay = allocate(Bytes(input.decay));
    void* beta = allocate(Bytes(input.beta));
    void* state_in = allocate(Bytes(input.recurrent_state));
    void* z_silu = allocate(Bytes(input.z_silu));
    void* norm_weight = allocate(Bytes(input.norm_weight));
    void* attention = allocate(kVValues * sizeof(float));
    void* state_out = allocate(kStateValues * sizeof(float));
    void* final_output = allocate(kVValues * sizeof(float));
    auto cleanup_allocations = [&]() {
      for (auto it = allocations.rbegin(); it != allocations.rend(); ++it) {
        zeMemFree(context_, *it);
      }
    };

    try {
      std::memcpy(conv, input.conv_output_raw.data(), Bytes(input.conv_output_raw));
      std::memcpy(decay, input.decay.data(), Bytes(input.decay));
      std::memcpy(beta, input.beta.data(), Bytes(input.beta));
      std::memcpy(state_in, input.recurrent_state.data(),
                  Bytes(input.recurrent_state));
      std::memcpy(z_silu, input.z_silu.data(), Bytes(input.z_silu));
      std::memcpy(norm_weight, input.norm_weight.data(),
                  Bytes(input.norm_weight));

      SetPointerArg(postconv_kernel_, 0, conv);
      SetPointerArg(postconv_kernel_, 1, q);
      SetPointerArg(postconv_kernel_, 2, k);
      SetPointerArg(postconv_kernel_, 3, v);
      Check(zeKernelSetArgumentValue(
                postconv_kernel_, 4, sizeof(input.norm_epsilon),
                &input.norm_epsilon),
            "zeKernelSetArgumentValue(postconv epsilon)");
      SetPointerArg(recurrent_kernel_, 0, q);
      SetPointerArg(recurrent_kernel_, 1, k);
      SetPointerArg(recurrent_kernel_, 2, v);
      SetPointerArg(recurrent_kernel_, 3, decay);
      SetPointerArg(recurrent_kernel_, 4, beta);
      SetPointerArg(recurrent_kernel_, 5, state_in);
      SetPointerArg(recurrent_kernel_, 6, z_silu);
      SetPointerArg(recurrent_kernel_, 7, norm_weight);
      SetPointerArg(recurrent_kernel_, 8, attention);
      SetPointerArg(recurrent_kernel_, 9, state_out);
      SetPointerArg(recurrent_kernel_, 10, final_output);
      Check(zeKernelSetArgumentValue(
                recurrent_kernel_, 11, sizeof(input.norm_epsilon),
                &input.norm_epsilon),
            "zeKernelSetArgumentValue(recurrent epsilon)");
      Check(zeKernelSetArgumentValue(
                recurrent_kernel_, 12, sizeof(input.attention_scale),
                &input.attention_scale),
            "zeKernelSetArgumentValue(attention scale)");

      if (command_list_recorded_) {
        Check(zeCommandListReset(command_list_), "zeCommandListReset");
      }
      ze_group_count_t postconv_groups{64, 1, 1};
      Check(zeCommandListAppendLaunchKernel(
                command_list_, postconv_kernel_, &postconv_groups,
                nullptr, 0, nullptr),
            "zeCommandListAppendLaunchKernel(postconv)");
      Check(zeCommandListAppendBarrier(
                command_list_, nullptr, 0, nullptr),
            "zeCommandListAppendBarrier");
      ze_group_count_t recurrent_groups{32, 1, 1};
      Check(zeCommandListAppendLaunchKernel(
                command_list_, recurrent_kernel_, &recurrent_groups,
                nullptr, 0, nullptr),
            "zeCommandListAppendLaunchKernel(recurrent)");
      Check(zeCommandListClose(command_list_), "zeCommandListClose");
      command_list_recorded_ = true;

      GpuLevelZeroPostconvRecurrentRun result;
      result.sample_wall_us.reserve(static_cast<std::size_t>(samples));
      for (int sample = 0; sample < samples; ++sample) {
        std::memcpy(state_in, input.recurrent_state.data(),
                    Bytes(input.recurrent_state));
        const auto begin = std::chrono::steady_clock::now();
        Check(zeCommandQueueExecuteCommandLists(
                  queue_, 1, &command_list_, nullptr),
              "zeCommandQueueExecuteCommandLists");
        Check(zeCommandQueueSynchronize(queue_, UINT64_MAX),
              "zeCommandQueueSynchronize");
        const auto end = std::chrono::steady_clock::now();
        result.sample_wall_us.push_back(
            std::chrono::duration<double, std::micro>(end - begin).count());
      }

      result.q_conv_predelta = CopyFloats(q, kQValues);
      result.k_conv_predelta = CopyFloats(k, kQValues);
      result.v_conv_predelta = CopyFloats(v, kVValues);
      result.attention_output = CopyFloats(attention, kVValues);
      result.recurrent_state = CopyFloats(state_out, kStateValues);
      result.final_output = CopyFloats(final_output, kVValues);
      cleanup_allocations();
      return result;
    } catch (...) {
      cleanup_allocations();
      throw;
    }
  }

 private:
  void ValidateInput(const GpuLevelZeroPostconvRecurrentInput& input,
                     int samples) const {
    Require(samples > 0, "Level Zero samples must be positive");
    Require(input.conv_output_raw.size() == kConvValues,
            "Level Zero conv output shape mismatch");
    Require(input.decay.size() == kValueHeads,
            "Level Zero decay shape mismatch");
    Require(input.beta.size() == kValueHeads,
            "Level Zero beta shape mismatch");
    Require(input.recurrent_state.size() == kStateValues,
            "Level Zero state shape mismatch");
    Require(input.z_silu.size() == kVValues,
            "Level Zero z_silu shape mismatch");
    Require(input.norm_weight.size() == kHeadDim,
            "Level Zero norm weight shape mismatch");
    Require(std::isfinite(input.norm_epsilon) && input.norm_epsilon > 0.0f,
            "Level Zero norm epsilon is invalid");
    Require(std::isfinite(input.attention_scale) &&
                input.attention_scale > 0.0f,
            "Level Zero attention scale is invalid");
  }

  ze_kernel_handle_t CreateKernel(const char* name) const {
    ze_kernel_desc_t desc{ZE_STRUCTURE_TYPE_KERNEL_DESC};
    desc.pKernelName = name;
    ze_kernel_handle_t kernel = nullptr;
    Check(zeKernelCreate(module_, &desc, &kernel), "zeKernelCreate");
    return kernel;
  }

  void* AllocateShared(std::size_t bytes) const {
    ze_device_mem_alloc_desc_t device_desc{
        ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC};
    ze_host_mem_alloc_desc_t host_desc{
        ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC};
    void* pointer = nullptr;
    Check(zeMemAllocShared(
              context_, &device_desc, &host_desc, bytes, 64, device_, &pointer),
          "zeMemAllocShared");
    Require(pointer != nullptr, "zeMemAllocShared returned null");
    return pointer;
  }

  void SetPointerArg(ze_kernel_handle_t kernel,
                     std::uint32_t index,
                     void* pointer) const {
    Check(zeKernelSetArgumentValue(
              kernel, index, sizeof(pointer), &pointer),
          "zeKernelSetArgumentValue(pointer)");
  }

  std::vector<float> CopyFloats(const void* source,
                                std::size_t count) const {
    std::vector<float> values(count);
    std::memcpy(values.data(), source, count * sizeof(float));
    return values;
  }

  void Cleanup() {
    if (queue_ != nullptr) zeCommandQueueSynchronize(queue_, UINT64_MAX);
    if (command_list_ != nullptr) zeCommandListDestroy(command_list_);
    if (recurrent_kernel_ != nullptr) zeKernelDestroy(recurrent_kernel_);
    if (postconv_kernel_ != nullptr) zeKernelDestroy(postconv_kernel_);
    if (module_ != nullptr) zeModuleDestroy(module_);
    if (queue_ != nullptr) zeCommandQueueDestroy(queue_);
    if (context_ != nullptr) zeContextDestroy(context_);
    recurrent_kernel_ = nullptr;
    postconv_kernel_ = nullptr;
    module_ = nullptr;
    command_list_ = nullptr;
    queue_ = nullptr;
    context_ = nullptr;
  }

  ze_driver_handle_t driver_ = nullptr;
  ze_device_handle_t device_ = nullptr;
  ze_context_handle_t context_ = nullptr;
  ze_command_queue_handle_t queue_ = nullptr;
  ze_command_list_handle_t command_list_ = nullptr;
  ze_module_handle_t module_ = nullptr;
  ze_kernel_handle_t postconv_kernel_ = nullptr;
  ze_kernel_handle_t recurrent_kernel_ = nullptr;
  std::uint32_t queue_ordinal_ = UINT32_MAX;
  bool command_list_recorded_ = false;
  std::string device_name_;
};

GpuLevelZeroPostconvRecurrentRunner::GpuLevelZeroPostconvRecurrentRunner(
    const std::string& native_module_path,
    std::uint32_t device_id)
    : impl_(std::make_unique<Impl>()) {
  impl_->Initialize(native_module_path, device_id);
}

GpuLevelZeroPostconvRecurrentRunner::~GpuLevelZeroPostconvRecurrentRunner() =
    default;

GpuLevelZeroPostconvRecurrentRunner::GpuLevelZeroPostconvRecurrentRunner(
    GpuLevelZeroPostconvRecurrentRunner&&) noexcept = default;

GpuLevelZeroPostconvRecurrentRunner&
GpuLevelZeroPostconvRecurrentRunner::operator=(
    GpuLevelZeroPostconvRecurrentRunner&&) noexcept = default;

const std::string& GpuLevelZeroPostconvRecurrentRunner::device_name() const {
  return impl_->device_name();
}

GpuLevelZeroPostconvRecurrentRun GpuLevelZeroPostconvRecurrentRunner::Run(
    const GpuLevelZeroPostconvRecurrentInput& input,
    int samples) {
  return impl_->Run(input, samples);
}

}  // namespace iq36
