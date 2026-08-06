#include "intel_qwen36/gguf_loader.hpp"
#include "intel_qwen36/packed_token_schedule.hpp"

#include <level_zero/ze_api.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

constexpr std::uint32_t kIntelVendorId = 0x8086U;
constexpr std::uint32_t kPtlDeviceId = 0xB080U;
constexpr std::uint32_t kLocalSize = 256;
constexpr std::uint32_t kMaximumGroups = 1024;
constexpr std::uint64_t kChecksumGroupsPerCommand = kMaximumGroups;

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
  Require(static_cast<bool>(input), "failed to open Level Zero module");
  const auto size = input.tellg();
  Require(size > 0, "Level Zero module is empty");
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
  input.seekg(0, std::ios::beg);
  input.read(reinterpret_cast<char*>(bytes.data()), size);
  Require(static_cast<bool>(input), "failed to read Level Zero module");
  return bytes;
}

bool MapsAreNativeOnly() {
  std::ifstream maps("/proc/self/maps");
  std::string line;
  while (std::getline(maps, line)) {
    std::transform(line.begin(), line.end(), line.begin(),
                   [](unsigned char value) { return std::tolower(value); });
    if (line.find("libdnnl") != std::string::npos ||
        line.find("openvino") != std::string::npos) {
      return false;
    }
  }
  return true;
}

std::uint64_t CommandStreamBytes(const iq36::PackedTokenCommand& command) {
  std::uint64_t bytes = command.resident_state_read_bytes +
                        command.resident_state_write_bytes;
  for (const auto& stream : command.streams) {
    bytes += stream.active_nbytes_per_token;
  }
  return bytes;
}

std::uint64_t TimestampDelta(std::uint64_t start,
                             std::uint64_t end,
                             std::uint32_t valid_bits) {
  if (valid_bits == 0 || valid_bits >= 64) return end - start;
  const std::uint64_t mask = (std::uint64_t{1} << valid_bits) - 1;
  return (end - start) & mask;
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

void WriteU64Array(const std::vector<std::uint64_t>& values) {
  std::cout << "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0) std::cout << ",";
    std::cout << values[i];
  }
  std::cout << "]";
}

class LevelZeroPackedStreamProbe {
 public:
  LevelZeroPackedStreamProbe(const iq36::PackedTokenProgram& program,
                             const std::string& module_path)
      : program_(program) {
    InitializeDevice();
    InitializeRuntime(module_path);
    AllocateBuffers();
    InitializePayload();
    RecordCommandList();
  }

  ~LevelZeroPackedStreamProbe() { Cleanup(); }

  LevelZeroPackedStreamProbe(const LevelZeroPackedStreamProbe&) = delete;
  LevelZeroPackedStreamProbe& operator=(
      const LevelZeroPackedStreamProbe&) = delete;

  struct Run {
    std::vector<double> device_us;
    std::vector<double> submit_us;
    std::vector<double> wall_us;
    std::vector<std::uint64_t> checksums;
  };

  Run Execute(int warmup, int samples) {
    Require(warmup >= 0 && samples > 0, "invalid sample counts");
    Run run;
    for (int sample = -warmup; sample < samples; ++sample) {
      token_control_[0] = static_cast<std::uint64_t>(1000 + sample + warmup);
      token_control_[1] = program_.context_tokens +
                          static_cast<std::uint64_t>(sample + warmup);
      const auto wall_begin = std::chrono::steady_clock::now();
      const auto submit_begin = wall_begin;
      Check(zeCommandQueueExecuteCommandLists(
                queue_, 1, &command_list_, nullptr),
            "zeCommandQueueExecuteCommandLists");
      const auto submit_end = std::chrono::steady_clock::now();
      Check(zeCommandQueueSynchronize(queue_, UINT64_MAX),
            "zeCommandQueueSynchronize");
      const auto wall_end = std::chrono::steady_clock::now();
      if (sample < 0) continue;
      const std::uint64_t ticks = TimestampDelta(
          timestamps_[0], timestamps_[1], properties_.kernelTimestampValidBits);
      run.device_us.push_back(
          static_cast<double>(ticks) * timestamp_ns_per_tick_ / 1000.0);
      run.submit_us.push_back(std::chrono::duration<double, std::micro>(
                                  submit_end - submit_begin).count());
      run.wall_us.push_back(std::chrono::duration<double, std::micro>(
                                wall_end - wall_begin).count());
      const std::uint64_t final_offset =
          (program_.commands.size() - 1) * kChecksumGroupsPerCommand;
      run.checksums.push_back(command_checksums_[final_offset]);
    }
    return run;
  }

  const std::string& device_name() const { return device_name_; }
  std::uint32_t device_id() const { return properties_.deviceId; }
  std::uint32_t command_queue_ordinal() const { return queue_ordinal_; }
  std::uint32_t command_list_record_count() const { return record_count_; }
  std::uint32_t kernel_count() const {
    return static_cast<std::uint32_t>(kernels_.size());
  }
  std::uint32_t barrier_count() const { return barrier_count_; }
  std::uint64_t maximum_command_bytes() const { return maximum_command_bytes_; }
  std::uint64_t payload_allocation_bytes() const {
    return payload_allocation_bytes_;
  }
  double timestamp_ns_per_tick() const { return timestamp_ns_per_tick_; }
  std::uint64_t timer_resolution() const { return properties_.timerResolution; }
  std::uint32_t timestamp_valid_bits() const {
    return properties_.kernelTimestampValidBits;
  }

 private:
  void InitializeDevice() {
    Check(zeInit(ZE_INIT_FLAG_GPU_ONLY), "zeInit");
    std::uint32_t driver_count = 0;
    Check(zeDriverGet(&driver_count, nullptr), "zeDriverGet(count)");
    Require(driver_count > 0, "no Level Zero driver");
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
            properties_ = properties;
            queue_ordinal_ = ordinal;
            device_name_ = properties.name;
            break;
          }
        }
        if (device_ != nullptr) break;
      }
      if (device_ != nullptr) break;
    }
    Require(device_ != nullptr, "PTL Level Zero device not found");

    std::uint64_t host_start = 0;
    std::uint64_t device_start = 0;
    std::uint64_t host_end = 0;
    std::uint64_t device_end = 0;
    Check(zeDeviceGetGlobalTimestamps(device_, &host_start, &device_start),
          "zeDeviceGetGlobalTimestamps(start)");
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    Check(zeDeviceGetGlobalTimestamps(device_, &host_end, &device_end),
          "zeDeviceGetGlobalTimestamps(end)");
    const auto device_delta = TimestampDelta(
        device_start, device_end, properties_.kernelTimestampValidBits);
    Require(host_end > host_start && device_delta > 0,
            "Level Zero timestamp calibration failed");
    timestamp_ns_per_tick_ =
        static_cast<double>(host_end - host_start) /
        static_cast<double>(device_delta);
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
    queue_desc.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
    Check(zeCommandQueueCreate(context_, device_, &queue_desc, &queue_),
          "zeCommandQueueCreate");
    ze_command_list_desc_t list_desc{ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC};
    list_desc.commandQueueGroupOrdinal = queue_ordinal_;
    Check(zeCommandListCreate(context_, device_, &list_desc, &command_list_),
          "zeCommandListCreate");

    module_bytes_ = ReadBinary(module_path);
    ze_module_desc_t module_desc{ZE_STRUCTURE_TYPE_MODULE_DESC};
    module_desc.format = ZE_MODULE_FORMAT_NATIVE;
    module_desc.inputSize = module_bytes_.size();
    module_desc.pInputModule = module_bytes_.data();
    module_desc.pBuildFlags = "";
    ze_module_build_log_handle_t log = nullptr;
    const ze_result_t result = zeModuleCreate(
        context_, device_, &module_desc, &module_, &log);
    if (log != nullptr) zeModuleBuildLogDestroy(log);
    Check(result, "zeModuleCreate");
  }

  void AllocateBuffers() {
    payload_offsets_.reserve(program_.commands.size());
    for (const auto& command : program_.commands) {
      const std::uint64_t bytes = CommandStreamBytes(command);
      maximum_command_bytes_ = std::max(maximum_command_bytes_, bytes);
      payload_offsets_.push_back(payload_allocation_bytes_);
      payload_allocation_bytes_ += (bytes + 31ULL) / 32ULL * 32ULL;
    }
    Require(payload_allocation_bytes_ >= program_.strict_stream_bytes_per_token,
            "packed stream payload allocation underflow");
    ze_device_mem_alloc_desc_t device_desc{
        ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC};
    Check(zeMemAllocDevice(context_, &device_desc, payload_allocation_bytes_,
                           64, device_, &payload_),
          "zeMemAllocDevice(payload)");
    ze_host_mem_alloc_desc_t host_desc{ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC};
    Check(zeMemAllocShared(context_, &device_desc, &host_desc,
                           2 * sizeof(std::uint64_t), 64, device_,
                           reinterpret_cast<void**>(&token_control_)),
          "zeMemAllocShared(token_control)");
    Check(zeMemAllocShared(context_, &device_desc, &host_desc,
                           2 * sizeof(std::uint64_t), 64, device_,
                           reinterpret_cast<void**>(&timestamps_)),
          "zeMemAllocShared(timestamps)");
    const std::uint64_t checksum_count =
        program_.commands.size() * kChecksumGroupsPerCommand;
    Check(zeMemAllocShared(context_, &device_desc, &host_desc,
                           checksum_count * sizeof(std::uint64_t), 64, device_,
                           reinterpret_cast<void**>(&command_checksums_)),
          "zeMemAllocShared(command_checksums)");
    std::memset(token_control_, 0, 2 * sizeof(std::uint64_t));
    std::memset(timestamps_, 0, 2 * sizeof(std::uint64_t));
    std::memset(command_checksums_, 0,
                checksum_count * sizeof(std::uint64_t));
  }

  void InitializePayload() {
    ze_command_list_handle_t list = nullptr;
    ze_command_list_desc_t list_desc{ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC};
    list_desc.commandQueueGroupOrdinal = queue_ordinal_;
    Check(zeCommandListCreate(context_, device_, &list_desc, &list),
          "zeCommandListCreate(init)");
    const std::uint64_t pattern = 0x6a09e667f3bcc909ULL;
    Check(zeCommandListAppendMemoryFill(
              list, payload_, &pattern, sizeof(pattern),
              payload_allocation_bytes_,
              nullptr, 0, nullptr),
          "zeCommandListAppendMemoryFill");
    Check(zeCommandListClose(list), "zeCommandListClose(init)");
    Check(zeCommandQueueExecuteCommandLists(queue_, 1, &list, nullptr),
          "zeCommandQueueExecuteCommandLists(init)");
    Check(zeCommandQueueSynchronize(queue_, UINT64_MAX),
          "zeCommandQueueSynchronize(init)");
    Check(zeCommandListDestroy(list), "zeCommandListDestroy(init)");
  }

  void RecordCommandList() {
    Check(zeCommandListAppendWriteGlobalTimestamp(
              command_list_, timestamps_, nullptr, 0, nullptr),
          "zeCommandListAppendWriteGlobalTimestamp(start)");
    kernels_.reserve(program_.commands.size());
    for (std::size_t index = 0; index < program_.commands.size(); ++index) {
      ze_kernel_desc_t kernel_desc{ZE_STRUCTURE_TYPE_KERNEL_DESC};
      kernel_desc.pKernelName = "iq36_packed_token_stream_stage";
      ze_kernel_handle_t kernel = nullptr;
      Check(zeKernelCreate(module_, &kernel_desc, &kernel), "zeKernelCreate");
      kernels_.push_back(kernel);
      Check(zeKernelSetGroupSize(kernel, kLocalSize, 1, 1),
            "zeKernelSetGroupSize");
      const std::uint64_t bytes = CommandStreamBytes(program_.commands[index]);
      const std::uint64_t words = (bytes + 31ULL) / 32ULL;
      const std::uint64_t needed_groups =
          (words + kLocalSize - 1) / kLocalSize;
      const std::uint32_t groups = static_cast<std::uint32_t>(
          std::max<std::uint64_t>(
              1, std::min<std::uint64_t>(needed_groups, kMaximumGroups)));
      const std::uint64_t checksum_offset =
          index * kChecksumGroupsPerCommand;
      const std::uint32_t command_index = static_cast<std::uint32_t>(index);
      auto* stream_payload = static_cast<std::uint8_t*>(payload_) +
                             payload_offsets_[index];
      SetPointerArg(kernel, 0, stream_payload);
      SetValueArg(kernel, 1, bytes);
      SetPointerArg(kernel, 2, token_control_);
      SetPointerArg(kernel, 3, command_checksums_);
      SetValueArg(kernel, 4, checksum_offset);
      SetValueArg(kernel, 5, command_index);
      ze_group_count_t group_count{groups, 1, 1};
      Check(zeCommandListAppendLaunchKernel(
                command_list_, kernel, &group_count, nullptr, 0, nullptr),
            "zeCommandListAppendLaunchKernel");
      if (index + 1 != program_.commands.size()) {
        Check(zeCommandListAppendBarrier(
                  command_list_, nullptr, 0, nullptr),
              "zeCommandListAppendBarrier");
        ++barrier_count_;
      }
    }
    Check(zeCommandListAppendWriteGlobalTimestamp(
              command_list_, timestamps_ + 1, nullptr, 0, nullptr),
          "zeCommandListAppendWriteGlobalTimestamp(end)");
    Check(zeCommandListClose(command_list_), "zeCommandListClose");
    ++record_count_;
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

  void Cleanup() {
    if (queue_ != nullptr) zeCommandQueueSynchronize(queue_, UINT64_MAX);
    for (auto it = kernels_.rbegin(); it != kernels_.rend(); ++it) {
      if (*it != nullptr) zeKernelDestroy(*it);
    }
    if (command_list_ != nullptr) zeCommandListDestroy(command_list_);
    if (module_ != nullptr) zeModuleDestroy(module_);
    if (payload_ != nullptr) zeMemFree(context_, payload_);
    if (command_checksums_ != nullptr) zeMemFree(context_, command_checksums_);
    if (timestamps_ != nullptr) zeMemFree(context_, timestamps_);
    if (token_control_ != nullptr) zeMemFree(context_, token_control_);
    if (queue_ != nullptr) zeCommandQueueDestroy(queue_);
    if (context_ != nullptr) zeContextDestroy(context_);
  }

  const iq36::PackedTokenProgram& program_;
  ze_driver_handle_t driver_ = nullptr;
  ze_device_handle_t device_ = nullptr;
  ze_context_handle_t context_ = nullptr;
  ze_command_queue_handle_t queue_ = nullptr;
  ze_command_list_handle_t command_list_ = nullptr;
  ze_module_handle_t module_ = nullptr;
  std::vector<ze_kernel_handle_t> kernels_;
  std::vector<std::uint8_t> module_bytes_;
  std::vector<std::uint64_t> payload_offsets_;
  void* payload_ = nullptr;
  std::uint64_t* token_control_ = nullptr;
  std::uint64_t* timestamps_ = nullptr;
  std::uint64_t* command_checksums_ = nullptr;
  ze_device_properties_t properties_{ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES};
  std::string device_name_;
  std::uint32_t queue_ordinal_ = UINT32_MAX;
  std::uint32_t record_count_ = 0;
  std::uint32_t barrier_count_ = 0;
  std::uint64_t maximum_command_bytes_ = 0;
  std::uint64_t payload_allocation_bytes_ = 0;
  double timestamp_ns_per_tick_ = 0.0;
};

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 5) {
      throw std::invalid_argument(
          "usage: iq36-packed-token-level-zero-probe MODEL MODULE "
          "WARMUP SAMPLES");
    }
    const int warmup = std::stoi(argv[3]);
    const int samples = std::stoi(argv[4]);
    const auto index = iq36::parse_gguf_model_index(argv[1]);
    const auto program = iq36::BuildPackedTokenProgram(index);
    LevelZeroPackedStreamProbe probe(program, argv[2]);
    const auto run = probe.Execute(warmup, samples);
    const double device_min_us = Minimum(run.device_us);
    const double submit_min_us = Minimum(run.submit_us);
    const double wall_min_us = Minimum(run.wall_us);
    const double host_residual_min_us = wall_min_us - device_min_us;
    const double effective_gb_s =
        static_cast<double>(program.strict_stream_bytes_per_token) /
        (device_min_us * 1000.0);
    const bool checksums_change =
        run.checksums.size() > 1 &&
        std::adjacent_find(run.checksums.begin(), run.checksums.end(),
                           std::equal_to<std::uint64_t>()) ==
            run.checksums.end();
    const bool maps_native_only = MapsAreNativeOnly();
    const bool pass = probe.device_id() == kPtlDeviceId &&
        probe.command_list_record_count() == 1 &&
        probe.kernel_count() == program.commands.size() &&
        probe.barrier_count() + 1 == program.commands.size() &&
        program.commands.size() == 252 &&
        program.strict_stream_bytes_per_token == 2'128'395'904ULL &&
        run.device_us.size() == static_cast<std::size_t>(samples) &&
        checksums_change && maps_native_only &&
        std::isfinite(device_min_us) &&
        device_min_us <=
            program.admission.kernel_schedule_ms_per_token_max * 1000.0 &&
        effective_gb_s >=
            program.strict_stream_bytes_per_token / 1e6 /
            program.admission.kernel_schedule_ms_per_token_max &&
        submit_min_us <=
            program.admission.host_submit_ms_per_token_max * 1000.0 &&
        host_residual_min_us <=
            program.admission.host_submit_ms_per_token_max * 1000.0;

    std::cout << std::boolalpha << std::setprecision(12) << "{"
              << "\"barrier_count\":" << probe.barrier_count() << ","
              << "\"checksum_values\":";
    WriteU64Array(run.checksums);
    std::cout << ",\"checksums_change_with_token_control\":"
              << checksums_change << ","
              << "\"command_count\":" << program.commands.size() << ","
              << "\"command_list_record_count\":"
              << probe.command_list_record_count() << ","
              << "\"device_id\":" << probe.device_id() << ","
              << "\"device_name\":\"" << probe.device_name() << "\","
              << "\"device_time_mean_us\":" << Mean(run.device_us) << ","
              << "\"device_time_min_us\":" << device_min_us << ","
              << "\"device_time_samples_us\":";
    WriteDoubleArray(run.device_us);
    std::cout << ",\"effective_stream_gb_s\":" << effective_gb_s << ","
              << "\"host_residual_min_us\":" << host_residual_min_us << ","
              << "\"kernel_count\":" << probe.kernel_count() << ","
              << "\"maps_native_only\":" << maps_native_only << ","
              << "\"maximum_command_bytes\":"
              << probe.maximum_command_bytes() << ","
              << "\"payload_allocation_bytes\":"
              << probe.payload_allocation_bytes() << ","
              << "\"queue_ordinal\":" << probe.command_queue_ordinal() << ","
              << "\"queue_submit_count\":" << warmup + samples << ","
              << "\"required_checks_passed\":" << pass << ","
              << "\"strict_stream_bytes_per_token\":"
              << program.strict_stream_bytes_per_token << ","
              << "\"submit_mean_us\":" << Mean(run.submit_us) << ","
              << "\"submit_min_us\":" << submit_min_us << ","
              << "\"submit_time_samples_us\":";
    WriteDoubleArray(run.submit_us);
    std::cout << ",\"timer_resolution\":" << probe.timer_resolution() << ","
              << "\"timestamp_ns_per_tick\":"
              << probe.timestamp_ns_per_tick() << ","
              << "\"timestamp_valid_bits\":"
              << probe.timestamp_valid_bits() << ","
              << "\"wall_mean_us\":" << Mean(run.wall_us) << ","
              << "\"wall_min_us\":" << wall_min_us << ","
              << "\"wall_time_samples_us\":";
    WriteDoubleArray(run.wall_us);
    std::cout << "}" << std::endl;
    return pass ? 0 : 2;
  } catch (const std::exception& exception) {
    std::cerr << "iq36-packed-token-level-zero-probe: "
              << exception.what() << '\n';
    return 4;
  }
}
