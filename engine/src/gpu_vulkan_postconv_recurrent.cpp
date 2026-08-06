#include "intel_qwen36/gpu_vulkan_postconv_recurrent.hpp"

#include <vulkan/vulkan.h>

#include <algorithm>
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

void Check(VkResult result, const char* where) {
  if (result != VK_SUCCESS) {
    Die(std::string(where) + " failed with VkResult " +
        std::to_string(result));
  }
}

std::vector<std::uint32_t> ReadSpirv(const std::string& path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  Require(static_cast<bool>(input), "failed to open SPIR-V: " + path);
  const auto end = input.tellg();
  Require(end > 0 && end % 4 == 0, "invalid SPIR-V size: " + path);
  std::vector<std::uint32_t> words(
      static_cast<std::size_t>(end) / sizeof(std::uint32_t));
  input.seekg(0, std::ios::beg);
  input.read(reinterpret_cast<char*>(words.data()), end);
  Require(static_cast<bool>(input), "failed to read SPIR-V: " + path);
  Require(words.front() == 0x07230203U, "invalid SPIR-V magic: " + path);
  return words;
}

template <typename T>
VkDeviceSize Bytes(const std::vector<T>& values) {
  return static_cast<VkDeviceSize>(values.size() * sizeof(T));
}

struct Buffer {
  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  void* mapped = nullptr;
  VkDeviceSize bytes = 0;
};

struct PushConstants {
  float norm_epsilon;
  float attention_scale;
};

}  // namespace

class GpuVulkanPostconvRecurrentRunner::Impl {
 public:
  Impl() = default;
  ~Impl() { Cleanup(); }

  void Initialize(const std::string& postconv_spirv_path,
                  const std::string& recurrent_spirv_path,
                  const std::string& device_substring) {
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "intel-qwen36-vulkan-postconv";
    app.applicationVersion = 1;
    app.pEngineName = "intel-qwen36";
    app.engineVersion = 1;
    app.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo instance_create{
        VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    instance_create.pApplicationInfo = &app;
    Check(vkCreateInstance(&instance_create, nullptr, &instance_),
          "vkCreateInstance");

    std::uint32_t device_count = 0;
    Check(vkEnumeratePhysicalDevices(instance_, &device_count, nullptr),
          "vkEnumeratePhysicalDevices(count)");
    Require(device_count > 0, "no Vulkan physical devices");
    std::vector<VkPhysicalDevice> devices(device_count);
    Check(vkEnumeratePhysicalDevices(
              instance_, &device_count, devices.data()),
          "vkEnumeratePhysicalDevices(list)");
    for (VkPhysicalDevice device : devices) {
      VkPhysicalDeviceProperties properties{};
      vkGetPhysicalDeviceProperties(device, &properties);
      if (properties.vendorID != 0x8086U ||
          (!device_substring.empty() &&
           std::string(properties.deviceName).find(device_substring) ==
               std::string::npos)) {
        continue;
      }
      std::uint32_t queue_count = 0;
      vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_count, nullptr);
      std::vector<VkQueueFamilyProperties> queues(queue_count);
      vkGetPhysicalDeviceQueueFamilyProperties(
          device, &queue_count, queues.data());
      for (std::uint32_t index = 0; index < queue_count; ++index) {
        if ((queues[index].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0U) {
          physical_device_ = device;
          queue_family_ = index;
          device_name_ = properties.deviceName;
          break;
        }
      }
      if (physical_device_ != VK_NULL_HANDLE) break;
    }
    Require(physical_device_ != VK_NULL_HANDLE,
            "no matching Intel Vulkan compute device");

    VkPhysicalDeviceFeatures available{};
    vkGetPhysicalDeviceFeatures(physical_device_, &available);
    Require(available.shaderFloat64 == VK_TRUE,
            "Vulkan device lacks shaderFloat64");
    VkPhysicalDeviceFeatures enabled{};
    enabled.shaderFloat64 = VK_TRUE;
    const float priority = 1.0f;
    VkDeviceQueueCreateInfo queue_create{
        VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queue_create.queueFamilyIndex = queue_family_;
    queue_create.queueCount = 1;
    queue_create.pQueuePriorities = &priority;
    VkDeviceCreateInfo device_create{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    device_create.queueCreateInfoCount = 1;
    device_create.pQueueCreateInfos = &queue_create;
    device_create.pEnabledFeatures = &enabled;
    Check(vkCreateDevice(
              physical_device_, &device_create, nullptr, &device_),
          "vkCreateDevice");
    vkGetDeviceQueue(device_, queue_family_, 0, &queue_);
    Require(queue_ != VK_NULL_HANDLE, "vkGetDeviceQueue returned null");
    vkGetPhysicalDeviceMemoryProperties(physical_device_, &memory_properties_);

    postconv_set_layout_ = CreateSetLayout(4);
    recurrent_set_layout_ = CreateSetLayout(11);
    postconv_pipeline_layout_ = CreatePipelineLayout(postconv_set_layout_);
    recurrent_pipeline_layout_ = CreatePipelineLayout(recurrent_set_layout_);
    postconv_pipeline_ = CreatePipeline(
        postconv_spirv_path, postconv_pipeline_layout_);
    recurrent_pipeline_ = CreatePipeline(
        recurrent_spirv_path, recurrent_pipeline_layout_);

    VkCommandPoolCreateInfo pool_create{
        VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pool_create.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool_create.queueFamilyIndex = queue_family_;
    Check(vkCreateCommandPool(
              device_, &pool_create, nullptr, &command_pool_),
          "vkCreateCommandPool");
  }

  const std::string& device_name() const { return device_name_; }

  GpuVulkanPostconvRecurrentRun Run(
      const GpuVulkanPostconvRecurrentInput& input,
      int samples) {
    ValidateInput(input, samples);
    std::vector<Buffer> buffers;
    buffers.reserve(12);
    buffers.push_back(CreateBuffer(Bytes(input.conv_output_raw)));
    buffers.push_back(CreateBuffer(kQValues * sizeof(float)));
    buffers.push_back(CreateBuffer(kQValues * sizeof(float)));
    buffers.push_back(CreateBuffer(kVValues * sizeof(float)));
    buffers.push_back(CreateBuffer(Bytes(input.decay)));
    buffers.push_back(CreateBuffer(Bytes(input.beta)));
    buffers.push_back(CreateBuffer(Bytes(input.recurrent_state)));
    buffers.push_back(CreateBuffer(Bytes(input.z_silu)));
    buffers.push_back(CreateBuffer(Bytes(input.norm_weight)));
    buffers.push_back(CreateBuffer(kVValues * sizeof(float)));
    buffers.push_back(CreateBuffer(kStateValues * sizeof(float)));
    buffers.push_back(CreateBuffer(kVValues * sizeof(float)));

    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSet postconv_set = VK_NULL_HANDLE;
    VkDescriptorSet recurrent_set = VK_NULL_HANDLE;
    VkCommandBuffer command = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    auto cleanup_run = [&]() {
      if (fence != VK_NULL_HANDLE) vkDestroyFence(device_, fence, nullptr);
      if (command != VK_NULL_HANDLE) {
        vkFreeCommandBuffers(device_, command_pool_, 1, &command);
      }
      if (descriptor_pool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(device_, descriptor_pool, nullptr);
      }
      for (auto it = buffers.rbegin(); it != buffers.rend(); ++it) {
        DestroyBuffer(*it);
      }
    };

    try {
      CopyToBuffer(input.conv_output_raw, buffers[0]);
      CopyToBuffer(input.decay, buffers[4]);
      CopyToBuffer(input.beta, buffers[5]);
      CopyToBuffer(input.recurrent_state, buffers[6]);
      CopyToBuffer(input.z_silu, buffers[7]);
      CopyToBuffer(input.norm_weight, buffers[8]);

      VkDescriptorPoolSize pool_size{};
      pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      pool_size.descriptorCount = 15;
      VkDescriptorPoolCreateInfo pool_create{
          VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
      pool_create.maxSets = 2;
      pool_create.poolSizeCount = 1;
      pool_create.pPoolSizes = &pool_size;
      Check(vkCreateDescriptorPool(
                device_, &pool_create, nullptr, &descriptor_pool),
            "vkCreateDescriptorPool");
      postconv_set = AllocateSet(descriptor_pool, postconv_set_layout_);
      recurrent_set = AllocateSet(descriptor_pool, recurrent_set_layout_);
      UpdateSet(postconv_set, buffers, {0, 1, 2, 3});
      UpdateSet(
          recurrent_set, buffers, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});

      VkCommandBufferAllocateInfo allocate{
          VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
      allocate.commandPool = command_pool_;
      allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
      allocate.commandBufferCount = 1;
      Check(vkAllocateCommandBuffers(device_, &allocate, &command),
            "vkAllocateCommandBuffers");
      VkCommandBufferBeginInfo begin{
          VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
      begin.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
      Check(vkBeginCommandBuffer(command, &begin), "vkBeginCommandBuffer");
      const PushConstants push{
          input.norm_epsilon, input.attention_scale};
      vkCmdBindPipeline(
          command, VK_PIPELINE_BIND_POINT_COMPUTE, postconv_pipeline_);
      vkCmdBindDescriptorSets(
          command, VK_PIPELINE_BIND_POINT_COMPUTE,
          postconv_pipeline_layout_, 0, 1, &postconv_set, 0, nullptr);
      vkCmdPushConstants(
          command, postconv_pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT,
          0, sizeof(push), &push);
      vkCmdDispatch(command, 64, 1, 1);
      VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
      barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
      vkCmdPipelineBarrier(
          command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier,
          0, nullptr, 0, nullptr);
      vkCmdBindPipeline(
          command, VK_PIPELINE_BIND_POINT_COMPUTE, recurrent_pipeline_);
      vkCmdBindDescriptorSets(
          command, VK_PIPELINE_BIND_POINT_COMPUTE,
          recurrent_pipeline_layout_, 0, 1, &recurrent_set, 0, nullptr);
      vkCmdPushConstants(
          command, recurrent_pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT,
          0, sizeof(push), &push);
      vkCmdDispatch(command, 32, 1, 1);
      Check(vkEndCommandBuffer(command), "vkEndCommandBuffer");

      VkFenceCreateInfo fence_create{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
      Check(vkCreateFence(device_, &fence_create, nullptr, &fence),
            "vkCreateFence");
      GpuVulkanPostconvRecurrentRun result;
      result.sample_wall_us.reserve(static_cast<std::size_t>(samples));
      for (int sample = 0; sample < samples; ++sample) {
        std::memcpy(
            buffers[6].mapped, input.recurrent_state.data(),
            static_cast<std::size_t>(buffers[6].bytes));
        Check(vkResetFences(device_, 1, &fence), "vkResetFences");
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &command;
        const auto start = std::chrono::steady_clock::now();
        Check(vkQueueSubmit(queue_, 1, &submit, fence), "vkQueueSubmit");
        Check(vkWaitForFences(device_, 1, &fence, VK_TRUE, UINT64_MAX),
              "vkWaitForFences");
        const auto end = std::chrono::steady_clock::now();
        result.sample_wall_us.push_back(
            std::chrono::duration<double, std::micro>(end - start).count());
      }

      result.q_conv_predelta = CopyFromBuffer(buffers[1], kQValues);
      result.k_conv_predelta = CopyFromBuffer(buffers[2], kQValues);
      result.v_conv_predelta = CopyFromBuffer(buffers[3], kVValues);
      result.attention_output = CopyFromBuffer(buffers[9], kVValues);
      result.recurrent_state = CopyFromBuffer(buffers[10], kStateValues);
      result.final_output = CopyFromBuffer(buffers[11], kVValues);
      cleanup_run();
      return result;
    } catch (...) {
      cleanup_run();
      throw;
    }
  }

 private:
  void ValidateInput(const GpuVulkanPostconvRecurrentInput& input,
                     int samples) const {
    Require(samples > 0, "Vulkan component samples must be positive");
    Require(input.conv_output_raw.size() == kConvValues,
            "Vulkan component conv output shape mismatch");
    Require(input.decay.size() == kValueHeads,
            "Vulkan component decay shape mismatch");
    Require(input.beta.size() == kValueHeads,
            "Vulkan component beta shape mismatch");
    Require(input.recurrent_state.size() == kStateValues,
            "Vulkan component state shape mismatch");
    Require(input.z_silu.size() == kVValues,
            "Vulkan component z_silu shape mismatch");
    Require(input.norm_weight.size() == kHeadDim,
            "Vulkan component norm weight shape mismatch");
    Require(std::isfinite(input.norm_epsilon) && input.norm_epsilon > 0.0f,
            "Vulkan component norm epsilon is invalid");
    Require(std::isfinite(input.attention_scale) &&
                input.attention_scale > 0.0f,
            "Vulkan component attention scale is invalid");
  }

  VkDescriptorSetLayout CreateSetLayout(std::uint32_t count) const {
    std::vector<VkDescriptorSetLayoutBinding> bindings(count);
    for (std::uint32_t index = 0; index < count; ++index) {
      bindings[index].binding = index;
      bindings[index].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      bindings[index].descriptorCount = 1;
      bindings[index].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo create{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    create.bindingCount = count;
    create.pBindings = bindings.data();
    VkDescriptorSetLayout layout = VK_NULL_HANDLE;
    Check(vkCreateDescriptorSetLayout(device_, &create, nullptr, &layout),
          "vkCreateDescriptorSetLayout");
    return layout;
  }

  VkPipelineLayout CreatePipelineLayout(VkDescriptorSetLayout set_layout) const {
    VkPushConstantRange push{};
    push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push.offset = 0;
    push.size = sizeof(PushConstants);
    VkPipelineLayoutCreateInfo create{
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    create.setLayoutCount = 1;
    create.pSetLayouts = &set_layout;
    create.pushConstantRangeCount = 1;
    create.pPushConstantRanges = &push;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    Check(vkCreatePipelineLayout(device_, &create, nullptr, &layout),
          "vkCreatePipelineLayout");
    return layout;
  }

  VkPipeline CreatePipeline(const std::string& path,
                            VkPipelineLayout layout) const {
    const auto words = ReadSpirv(path);
    VkShaderModuleCreateInfo shader_create{
        VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    shader_create.codeSize = words.size() * sizeof(std::uint32_t);
    shader_create.pCode = words.data();
    VkShaderModule shader = VK_NULL_HANDLE;
    Check(vkCreateShaderModule(device_, &shader_create, nullptr, &shader),
          "vkCreateShaderModule");
    VkPipelineShaderStageCreateInfo stage{
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = shader;
    stage.pName = "main";
    VkComputePipelineCreateInfo pipeline_create{
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipeline_create.stage = stage;
    pipeline_create.layout = layout;
    VkPipeline pipeline = VK_NULL_HANDLE;
    const VkResult result = vkCreateComputePipelines(
        device_, VK_NULL_HANDLE, 1, &pipeline_create, nullptr, &pipeline);
    vkDestroyShaderModule(device_, shader, nullptr);
    Check(result, "vkCreateComputePipelines");
    return pipeline;
  }

  std::uint32_t FindMemoryType(std::uint32_t type_bits) const {
    const VkMemoryPropertyFlags required =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    for (std::uint32_t pass = 0; pass < 2; ++pass) {
      for (std::uint32_t index = 0;
           index < memory_properties_.memoryTypeCount; ++index) {
        if ((type_bits & (1U << index)) == 0U) continue;
        const auto flags = memory_properties_.memoryTypes[index].propertyFlags;
        if ((flags & required) != required) continue;
        if (pass == 0 &&
            (flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) == 0U) {
          continue;
        }
        return index;
      }
    }
    Die("no host-visible coherent Vulkan memory type");
  }

  Buffer CreateBuffer(VkDeviceSize bytes) const {
    Buffer out;
    out.bytes = bytes;
    VkBufferCreateInfo create{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    create.size = bytes;
    create.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    create.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    Check(vkCreateBuffer(device_, &create, nullptr, &out.buffer),
          "vkCreateBuffer");
    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(device_, out.buffer, &requirements);
    VkMemoryAllocateInfo allocate{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocate.allocationSize = requirements.size;
    allocate.memoryTypeIndex = FindMemoryType(requirements.memoryTypeBits);
    Check(vkAllocateMemory(device_, &allocate, nullptr, &out.memory),
          "vkAllocateMemory");
    Check(vkBindBufferMemory(device_, out.buffer, out.memory, 0),
          "vkBindBufferMemory");
    Check(vkMapMemory(device_, out.memory, 0, bytes, 0, &out.mapped),
          "vkMapMemory");
    return out;
  }

  void DestroyBuffer(Buffer& buffer) const {
    if (buffer.mapped != nullptr) {
      vkUnmapMemory(device_, buffer.memory);
      buffer.mapped = nullptr;
    }
    if (buffer.buffer != VK_NULL_HANDLE) {
      vkDestroyBuffer(device_, buffer.buffer, nullptr);
      buffer.buffer = VK_NULL_HANDLE;
    }
    if (buffer.memory != VK_NULL_HANDLE) {
      vkFreeMemory(device_, buffer.memory, nullptr);
      buffer.memory = VK_NULL_HANDLE;
    }
  }

  void CopyToBuffer(const std::vector<float>& values, Buffer& buffer) const {
    Require(Bytes(values) == buffer.bytes, "Vulkan upload size mismatch");
    std::memcpy(buffer.mapped, values.data(),
                static_cast<std::size_t>(buffer.bytes));
  }

  std::vector<float> CopyFromBuffer(const Buffer& buffer,
                                    std::size_t count) const {
    Require(count * sizeof(float) == buffer.bytes,
            "Vulkan download size mismatch");
    std::vector<float> values(count);
    std::memcpy(values.data(), buffer.mapped,
                static_cast<std::size_t>(buffer.bytes));
    return values;
  }

  VkDescriptorSet AllocateSet(VkDescriptorPool pool,
                              VkDescriptorSetLayout layout) const {
    VkDescriptorSetAllocateInfo allocate{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocate.descriptorPool = pool;
    allocate.descriptorSetCount = 1;
    allocate.pSetLayouts = &layout;
    VkDescriptorSet set = VK_NULL_HANDLE;
    Check(vkAllocateDescriptorSets(device_, &allocate, &set),
          "vkAllocateDescriptorSets");
    return set;
  }

  void UpdateSet(VkDescriptorSet set,
                 const std::vector<Buffer>& buffers,
                 const std::vector<std::size_t>& indices) const {
    std::vector<VkDescriptorBufferInfo> infos(indices.size());
    std::vector<VkWriteDescriptorSet> writes(indices.size());
    for (std::size_t binding = 0; binding < indices.size(); ++binding) {
      const auto& buffer = buffers.at(indices[binding]);
      infos[binding].buffer = buffer.buffer;
      infos[binding].offset = 0;
      infos[binding].range = buffer.bytes;
      writes[binding].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      writes[binding].dstSet = set;
      writes[binding].dstBinding = static_cast<std::uint32_t>(binding);
      writes[binding].descriptorCount = 1;
      writes[binding].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      writes[binding].pBufferInfo = &infos[binding];
    }
    vkUpdateDescriptorSets(
        device_, static_cast<std::uint32_t>(writes.size()), writes.data(),
        0, nullptr);
  }

  void Cleanup() {
    if (device_ != VK_NULL_HANDLE) vkDeviceWaitIdle(device_);
    if (command_pool_ != VK_NULL_HANDLE) {
      vkDestroyCommandPool(device_, command_pool_, nullptr);
    }
    if (recurrent_pipeline_ != VK_NULL_HANDLE) {
      vkDestroyPipeline(device_, recurrent_pipeline_, nullptr);
    }
    if (postconv_pipeline_ != VK_NULL_HANDLE) {
      vkDestroyPipeline(device_, postconv_pipeline_, nullptr);
    }
    if (recurrent_pipeline_layout_ != VK_NULL_HANDLE) {
      vkDestroyPipelineLayout(device_, recurrent_pipeline_layout_, nullptr);
    }
    if (postconv_pipeline_layout_ != VK_NULL_HANDLE) {
      vkDestroyPipelineLayout(device_, postconv_pipeline_layout_, nullptr);
    }
    if (recurrent_set_layout_ != VK_NULL_HANDLE) {
      vkDestroyDescriptorSetLayout(device_, recurrent_set_layout_, nullptr);
    }
    if (postconv_set_layout_ != VK_NULL_HANDLE) {
      vkDestroyDescriptorSetLayout(device_, postconv_set_layout_, nullptr);
    }
    if (device_ != VK_NULL_HANDLE) vkDestroyDevice(device_, nullptr);
    if (instance_ != VK_NULL_HANDLE) vkDestroyInstance(instance_, nullptr);
    command_pool_ = VK_NULL_HANDLE;
    recurrent_pipeline_ = VK_NULL_HANDLE;
    postconv_pipeline_ = VK_NULL_HANDLE;
    recurrent_pipeline_layout_ = VK_NULL_HANDLE;
    postconv_pipeline_layout_ = VK_NULL_HANDLE;
    recurrent_set_layout_ = VK_NULL_HANDLE;
    postconv_set_layout_ = VK_NULL_HANDLE;
    device_ = VK_NULL_HANDLE;
    instance_ = VK_NULL_HANDLE;
  }

  VkInstance instance_ = VK_NULL_HANDLE;
  VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
  VkDevice device_ = VK_NULL_HANDLE;
  VkQueue queue_ = VK_NULL_HANDLE;
  std::uint32_t queue_family_ = UINT32_MAX;
  VkPhysicalDeviceMemoryProperties memory_properties_{};
  VkDescriptorSetLayout postconv_set_layout_ = VK_NULL_HANDLE;
  VkDescriptorSetLayout recurrent_set_layout_ = VK_NULL_HANDLE;
  VkPipelineLayout postconv_pipeline_layout_ = VK_NULL_HANDLE;
  VkPipelineLayout recurrent_pipeline_layout_ = VK_NULL_HANDLE;
  VkPipeline postconv_pipeline_ = VK_NULL_HANDLE;
  VkPipeline recurrent_pipeline_ = VK_NULL_HANDLE;
  VkCommandPool command_pool_ = VK_NULL_HANDLE;
  std::string device_name_;
};

GpuVulkanPostconvRecurrentRunner::GpuVulkanPostconvRecurrentRunner(
    const std::string& postconv_spirv_path,
    const std::string& recurrent_spirv_path,
    const std::string& device_substring)
    : impl_(std::make_unique<Impl>()) {
  impl_->Initialize(
      postconv_spirv_path, recurrent_spirv_path, device_substring);
}

GpuVulkanPostconvRecurrentRunner::~GpuVulkanPostconvRecurrentRunner() = default;

GpuVulkanPostconvRecurrentRunner::GpuVulkanPostconvRecurrentRunner(
    GpuVulkanPostconvRecurrentRunner&&) noexcept = default;

GpuVulkanPostconvRecurrentRunner&
GpuVulkanPostconvRecurrentRunner::operator=(
    GpuVulkanPostconvRecurrentRunner&&) noexcept = default;

const std::string& GpuVulkanPostconvRecurrentRunner::device_name() const {
  return impl_->device_name();
}

GpuVulkanPostconvRecurrentRun GpuVulkanPostconvRecurrentRunner::Run(
    const GpuVulkanPostconvRecurrentInput& input,
    int samples) {
  return impl_->Run(input, samples);
}

}  // namespace iq36
