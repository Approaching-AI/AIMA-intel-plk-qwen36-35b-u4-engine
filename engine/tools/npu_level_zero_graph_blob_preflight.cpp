#include "intel_qwen36/npu_level_zero_graph_ext.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

constexpr std::array<char, 8> kIoMagic{'I', 'Q', '3', '6', 'I', 'O', '1', '\0'};

[[noreturn]] void Die(const std::string& message) {
  throw std::runtime_error(message);
}

void Check(ze_result_t result, std::string_view where) {
  if (result != ZE_RESULT_SUCCESS) {
    Die(std::string(where) + " failed with ze_result_t " +
        std::to_string(static_cast<std::uint32_t>(result)));
  }
}

std::vector<std::uint8_t> ReadBytes(const std::filesystem::path& path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) Die("failed to open " + path.string());
  const auto end = stream.tellg();
  if (end < 0) Die("failed to size " + path.string());
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(end));
  stream.seekg(0);
  if (!bytes.empty()) {
    stream.read(reinterpret_cast<char*>(bytes.data()),
                static_cast<std::streamsize>(bytes.size()));
  }
  if (!stream) Die("failed to read " + path.string());
  return bytes;
}

void WriteBytes(const std::filesystem::path& path,
                const std::vector<std::uint8_t>& bytes) {
  std::ofstream stream(path, std::ios::binary);
  if (!stream) Die("failed to create " + path.string());
  if (!bytes.empty()) {
    stream.write(reinterpret_cast<const char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
  }
  if (!stream) Die("failed to write " + path.string());
}

template <typename T>
void AppendScalar(std::vector<std::uint8_t>& out, const T& value) {
  const auto* begin = reinterpret_cast<const std::uint8_t*>(&value);
  out.insert(out.end(), begin, begin + sizeof(T));
}

template <typename T>
T ConsumeScalar(const std::vector<std::uint8_t>& bytes, std::size_t& offset) {
  if (offset + sizeof(T) > bytes.size()) Die("truncated scalar container");
  T value{};
  std::memcpy(&value, bytes.data() + offset, sizeof(T));
  offset += sizeof(T);
  return value;
}

void WriteBuffers(const std::filesystem::path& path,
                  const std::vector<std::vector<std::uint8_t>>& buffers) {
  std::vector<std::uint8_t> bytes(kIoMagic.begin(), kIoMagic.end());
  AppendScalar(bytes, static_cast<std::uint32_t>(buffers.size()));
  for (const auto& buffer : buffers) {
    AppendScalar(bytes, static_cast<std::uint64_t>(buffer.size()));
    bytes.insert(bytes.end(), buffer.begin(), buffer.end());
  }
  WriteBytes(path, bytes);
}

std::vector<std::vector<std::uint8_t>> ReadBuffers(
    const std::filesystem::path& path) {
  const auto bytes = ReadBytes(path);
  if (bytes.size() < kIoMagic.size() ||
      !std::equal(kIoMagic.begin(), kIoMagic.end(), bytes.begin())) {
    Die("invalid IO container " + path.string());
  }
  std::size_t offset = kIoMagic.size();
  const auto count = ConsumeScalar<std::uint32_t>(bytes, offset);
  std::vector<std::vector<std::uint8_t>> result;
  result.reserve(count);
  for (std::uint32_t i = 0; i < count; ++i) {
    const auto size = ConsumeScalar<std::uint64_t>(bytes, offset);
    if (size > bytes.size() - offset) Die("truncated IO buffer");
    result.emplace_back(bytes.begin() + static_cast<std::ptrdiff_t>(offset),
                        bytes.begin() + static_cast<std::ptrdiff_t>(offset + size));
    offset += static_cast<std::size_t>(size);
  }
  if (offset != bytes.size()) Die("trailing IO container bytes");
  return result;
}

std::size_t PrecisionBytes(iq36_ze_graph_argument_precision_t precision) {
  switch (precision) {
    case IQ36_ZE_GRAPH_ARGUMENT_PRECISION_FP64:
    case IQ36_ZE_GRAPH_ARGUMENT_PRECISION_UINT64:
    case IQ36_ZE_GRAPH_ARGUMENT_PRECISION_INT64:
      return 8;
    case IQ36_ZE_GRAPH_ARGUMENT_PRECISION_FP32:
    case IQ36_ZE_GRAPH_ARGUMENT_PRECISION_UINT32:
    case IQ36_ZE_GRAPH_ARGUMENT_PRECISION_INT32:
      return 4;
    case IQ36_ZE_GRAPH_ARGUMENT_PRECISION_FP16:
    case IQ36_ZE_GRAPH_ARGUMENT_PRECISION_BF16:
    case IQ36_ZE_GRAPH_ARGUMENT_PRECISION_UINT16:
    case IQ36_ZE_GRAPH_ARGUMENT_PRECISION_INT16:
      return 2;
    case IQ36_ZE_GRAPH_ARGUMENT_PRECISION_UINT8:
    case IQ36_ZE_GRAPH_ARGUMENT_PRECISION_INT8:
      return 1;
    default:
      Die("unsupported graph argument precision " +
          std::to_string(static_cast<std::uint32_t>(precision)));
  }
}

bool MapsContain(std::string_view needle) {
  std::ifstream maps("/proc/self/maps");
  std::string line;
  while (std::getline(maps, line)) {
    if (line.find(needle) != std::string::npos) return true;
  }
  return false;
}

struct Runtime {
  ze_driver_handle_t driver = nullptr;
  ze_device_handle_t device = nullptr;
  ze_context_handle_t context = nullptr;
  iq36_ze_graph_dditable_ext_t* graph = nullptr;
  std::uint32_t queue_ordinal = 0;
  ze_device_properties_t device_properties{};
  iq36_ze_device_graph_properties_t graph_properties{};

  Runtime() {
    device_properties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
    graph_properties.stype = IQ36_ZE_STRUCTURE_TYPE_DEVICE_GRAPH_PROPERTIES;
    Check(zeInit(ZE_INIT_FLAG_VPU_ONLY), "zeInit");
    std::uint32_t driver_count = 0;
    Check(zeDriverGet(&driver_count, nullptr), "zeDriverGet(count)");
    std::vector<ze_driver_handle_t> drivers(driver_count);
    Check(zeDriverGet(&driver_count, drivers.data()), "zeDriverGet");
    for (auto candidate_driver : drivers) {
      std::uint32_t device_count = 0;
      Check(zeDeviceGet(candidate_driver, &device_count, nullptr),
            "zeDeviceGet(count)");
      std::vector<ze_device_handle_t> devices(device_count);
      Check(zeDeviceGet(candidate_driver, &device_count, devices.data()),
            "zeDeviceGet");
      for (auto candidate_device : devices) {
        ze_device_properties_t props{};
        props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
        Check(zeDeviceGetProperties(candidate_device, &props),
              "zeDeviceGetProperties");
        if (props.type == ZE_DEVICE_TYPE_VPU) {
          driver = candidate_driver;
          device = candidate_device;
          device_properties = props;
          break;
        }
      }
      if (device != nullptr) break;
    }
    if (device == nullptr) Die("no Level Zero VPU/NPU device");

    void* table = nullptr;
    Check(zeDriverGetExtensionFunctionAddress(driver, IQ36_ZE_GRAPH_EXT_NAME,
                                              &table),
          "zeDriverGetExtensionFunctionAddress(graph)");
    graph = static_cast<iq36_ze_graph_dditable_ext_t*>(table);
    if (graph == nullptr) Die("null graph DDI table");
    Check(graph->pfnDeviceGetGraphProperties(device, &graph_properties),
          "pfnDeviceGetGraphProperties");

    ze_context_desc_t context_desc{};
    context_desc.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC;
    Check(zeContextCreate(driver, &context_desc, &context), "zeContextCreate");

    std::uint32_t group_count = 0;
    Check(zeDeviceGetCommandQueueGroupProperties(device, &group_count, nullptr),
          "zeDeviceGetCommandQueueGroupProperties(count)");
    std::vector<ze_command_queue_group_properties_t> groups(group_count);
    for (auto& group : groups) {
      group.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES;
    }
    Check(zeDeviceGetCommandQueueGroupProperties(device, &group_count,
                                                 groups.data()),
          "zeDeviceGetCommandQueueGroupProperties");
    bool found = false;
    for (std::uint32_t i = 0; i < group_count; ++i) {
      if ((groups[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) != 0) {
        queue_ordinal = i;
        found = true;
        break;
      }
    }
    if (!found) Die("no NPU compute command queue group");
  }

  ~Runtime() {
    if (context != nullptr) zeContextDestroy(context);
  }
};

struct Argument {
  std::uint32_t index = 0;
  iq36_ze_graph_argument_properties_t properties{};
  std::size_t bytes = 0;
  void* pointer = nullptr;
};

struct GraphRun {
  Runtime& runtime;
  iq36_ze_graph_handle_t handle = nullptr;
  std::vector<std::uint8_t> source;
  std::vector<Argument> arguments;

  GraphRun(Runtime& runtime_in, iq36_ze_graph_format_t format,
           std::vector<std::uint8_t> input)
      : runtime(runtime_in), source(std::move(input)) {
    iq36_ze_graph_desc_t desc{};
    desc.stype = IQ36_ZE_STRUCTURE_TYPE_GRAPH_DESC;
    desc.format = format;
    desc.inputSize = source.size();
    desc.pInput = source.data();
    Check(runtime.graph->pfnCreate(runtime.context, runtime.device, &desc,
                                   &handle),
          "pfnCreate");
    iq36_ze_graph_properties_t properties{};
    properties.stype = IQ36_ZE_STRUCTURE_TYPE_GRAPH_PROPERTIES;
    Check(runtime.graph->pfnGetProperties(handle, &properties),
          "pfnGetProperties");
    arguments.reserve(properties.numGraphArgs);
    for (std::uint32_t index = 0; index < properties.numGraphArgs; ++index) {
      Argument argument;
      argument.index = index;
      argument.properties.stype =
          IQ36_ZE_STRUCTURE_TYPE_GRAPH_ARGUMENT_PROPERTIES;
      Check(runtime.graph->pfnGetArgumentProperties(
                handle, index, &argument.properties),
            "pfnGetArgumentProperties");
      std::size_t elements = 1;
      for (const auto dim : argument.properties.dims) {
        if (dim == 0) Die("zero graph argument dimension");
        elements *= dim;
      }
      argument.bytes =
          elements * PrecisionBytes(argument.properties.devicePrecision);
      ze_device_mem_alloc_desc_t device_desc{};
      device_desc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
      ze_host_mem_alloc_desc_t host_desc{};
      host_desc.stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC;
      Check(zeMemAllocShared(runtime.context, &device_desc, &host_desc,
                             argument.bytes, 64, runtime.device,
                             &argument.pointer),
            "zeMemAllocShared");
      std::memset(argument.pointer, 0, argument.bytes);
      Check(runtime.graph->pfnSetArgumentValue(handle, index, argument.pointer),
            "pfnSetArgumentValue");
      arguments.push_back(argument);
    }
  }

  ~GraphRun() {
    for (auto& argument : arguments) {
      if (argument.pointer != nullptr) {
        zeMemFree(runtime.context, argument.pointer);
      }
    }
    if (handle != nullptr) runtime.graph->pfnDestroy(handle);
  }

  std::vector<Argument*> Inputs() {
    std::vector<Argument*> result;
    for (auto& argument : arguments) {
      if (argument.properties.type == IQ36_ZE_GRAPH_ARGUMENT_TYPE_INPUT) {
        result.push_back(&argument);
      }
    }
    return result;
  }

  std::vector<Argument*> Outputs() {
    std::vector<Argument*> result;
    for (auto& argument : arguments) {
      if (argument.properties.type == IQ36_ZE_GRAPH_ARGUMENT_TYPE_OUTPUT) {
        result.push_back(&argument);
      }
    }
    return result;
  }

  std::vector<double> InitializeAndExecute(std::uint32_t warmup,
                                           std::uint32_t repeat) {
    if (repeat == 0) Die("repeat must be positive");
    ze_command_queue_desc_t queue_desc{};
    queue_desc.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
    queue_desc.ordinal = runtime.queue_ordinal;
    queue_desc.index = 0;
    queue_desc.mode = ZE_COMMAND_QUEUE_MODE_DEFAULT;
    queue_desc.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
    ze_command_queue_handle_t queue = nullptr;
    Check(zeCommandQueueCreate(runtime.context, runtime.device, &queue_desc,
                               &queue),
          "zeCommandQueueCreate");

    auto make_list = [&](bool initialize) {
      ze_command_list_desc_t list_desc{};
      list_desc.stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC;
      list_desc.commandQueueGroupOrdinal = runtime.queue_ordinal;
      ze_command_list_handle_t list = nullptr;
      Check(zeCommandListCreate(runtime.context, runtime.device, &list_desc,
                                &list),
            "zeCommandListCreate");
      if (initialize) {
        Check(runtime.graph->pfnAppendGraphInitialize(list, handle, nullptr, 0,
                                                      nullptr),
              "pfnAppendGraphInitialize");
      } else {
        Check(runtime.graph->pfnAppendGraphExecute(list, handle, nullptr, nullptr,
                                                   0, nullptr),
              "pfnAppendGraphExecute");
      }
      Check(zeCommandListClose(list), "zeCommandListClose");
      return list;
    };
    auto submit = [&](ze_command_list_handle_t list) {
      Check(zeCommandQueueExecuteCommandLists(queue, 1, &list, nullptr),
            "zeCommandQueueExecuteCommandLists");
      Check(zeCommandQueueSynchronize(queue,
                                      std::numeric_limits<std::uint64_t>::max()),
            "zeCommandQueueSynchronize");
    };

    const auto initialize_list = make_list(true);
    const auto execute_list = make_list(false);
    submit(initialize_list);
    Check(zeCommandListDestroy(initialize_list),
          "zeCommandListDestroy(initialize)");
    for (std::uint32_t i = 0; i < warmup; ++i) submit(execute_list);
    std::vector<double> timings_us;
    timings_us.reserve(repeat);
    for (std::uint32_t i = 0; i < repeat; ++i) {
      const auto started = std::chrono::steady_clock::now();
      submit(execute_list);
      const auto stopped = std::chrono::steady_clock::now();
      timings_us.push_back(
          std::chrono::duration<double, std::micro>(stopped - started).count());
    }
    Check(zeCommandListDestroy(execute_list),
          "zeCommandListDestroy(execute)");
    Check(zeCommandQueueDestroy(queue), "zeCommandQueueDestroy");
    return timings_us;
  }

  std::vector<std::uint8_t> NativeBinary() {
    std::size_t size = 0;
    Check(runtime.graph->pfnGetNativeBinary(handle, &size, nullptr),
          "pfnGetNativeBinary(size)");
    std::vector<std::uint8_t> binary(size);
    Check(runtime.graph->pfnGetNativeBinary(handle, &size, binary.data()),
          "pfnGetNativeBinary(data)");
    binary.resize(size);
    return binary;
  }

  std::vector<std::vector<std::uint8_t>> Capture(
      const std::vector<Argument*>& selected) {
    std::vector<std::vector<std::uint8_t>> result;
    for (const auto* argument : selected) {
      const auto* begin = static_cast<const std::uint8_t*>(argument->pointer);
      result.emplace_back(begin, begin + argument->bytes);
    }
    return result;
  }
};

std::vector<std::uint8_t> NGraphLiteBundle(
    const Runtime& runtime, const std::filesystem::path& xml_path) {
  auto xml = ReadBytes(xml_path);
  auto bin_path = xml_path;
  bin_path.replace_extension(".bin");
  auto bin = ReadBytes(bin_path);
  std::vector<std::uint8_t> bundle;
  AppendScalar(bundle, runtime.graph_properties.compilerVersion);
  AppendScalar(bundle, std::uint32_t{2});
  AppendScalar(bundle, static_cast<std::uint64_t>(xml.size()));
  bundle.insert(bundle.end(), xml.begin(), xml.end());
  AppendScalar(bundle, static_cast<std::uint64_t>(bin.size()));
  bundle.insert(bundle.end(), bin.begin(), bin.end());
  return bundle;
}

void FillDeterministicInputs(GraphRun& graph) {
  std::uint8_t seed = 17;
  for (auto* input : graph.Inputs()) {
    auto* bytes = static_cast<std::uint8_t*>(input->pointer);
    if (input->properties.devicePrecision == IQ36_ZE_GRAPH_ARGUMENT_PRECISION_FP32) {
      auto* values = static_cast<float*>(input->pointer);
      const auto count = input->bytes / sizeof(float);
      for (std::size_t i = 0; i < count; ++i) {
        values[i] = static_cast<float>(static_cast<int>((i + seed) % 31) - 15) /
                    16.0f;
      }
    } else {
      for (std::size_t i = 0; i < input->bytes; ++i) {
        bytes[i] = static_cast<std::uint8_t>(seed + i * 13);
      }
    }
    seed = static_cast<std::uint8_t>(seed + 29);
  }
}

void RestoreInputs(GraphRun& graph,
                   const std::vector<std::vector<std::uint8_t>>& saved) {
  auto inputs = graph.Inputs();
  if (inputs.size() != saved.size()) Die("input count mismatch");
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    if (inputs[i]->bytes != saved[i].size()) Die("input byte-size mismatch");
    std::memcpy(inputs[i]->pointer, saved[i].data(), saved[i].size());
  }
}

std::pair<std::uint64_t, std::uint64_t> Compare(
    const std::vector<std::vector<std::uint8_t>>& actual,
    const std::vector<std::vector<std::uint8_t>>& expected) {
  if (actual.size() != expected.size()) Die("output count mismatch");
  std::uint64_t compared = 0;
  std::uint64_t mismatches = 0;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    if (actual[i].size() != expected[i].size()) Die("output size mismatch");
    compared += actual[i].size();
    for (std::size_t j = 0; j < actual[i].size(); ++j) {
      mismatches += actual[i][j] != expected[i][j];
    }
  }
  return {compared, mismatches};
}

struct Options {
  std::string mode;
  std::filesystem::path xml;
  std::filesystem::path blob;
  std::filesystem::path inputs;
  std::filesystem::path reference;
  std::uint32_t warmup = 0;
  std::uint32_t repeat = 1;
};

std::uint32_t ParseCount(const std::string& text, const char* name,
                         bool allow_zero) {
  std::size_t consumed = 0;
  const auto parsed = std::stoul(text, &consumed);
  if (consumed != text.size() ||
      parsed > std::numeric_limits<std::uint32_t>::max() ||
      (!allow_zero && parsed == 0)) {
    Die(std::string("invalid ") + name + " count " + text);
  }
  return static_cast<std::uint32_t>(parsed);
}

void PrintTimings(const std::vector<double>& timings_us) {
  auto ordered = timings_us;
  std::sort(ordered.begin(), ordered.end());
  const auto middle = ordered.size() / 2;
  const double median = ordered.size() % 2 == 0
                            ? (ordered[middle - 1] + ordered[middle]) / 2.0
                            : ordered[middle];
  std::cout << ",\"execution_us\":[";
  for (std::size_t i = 0; i < timings_us.size(); ++i) {
    if (i != 0) std::cout << ',';
    std::cout << timings_us[i];
  }
  std::cout << "]"
            << ",\"execution_min_us\":" << ordered.front()
            << ",\"execution_median_us\":" << median;
}

Options Parse(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto value = [&](const char* name) -> std::string {
      if (i + 1 >= argc) Die(std::string("missing value for ") + name);
      return argv[++i];
    };
    if (arg == "--mode") options.mode = value("--mode");
    else if (arg == "--xml") options.xml = value("--xml");
    else if (arg == "--blob") options.blob = value("--blob");
    else if (arg == "--inputs") options.inputs = value("--inputs");
    else if (arg == "--reference") options.reference = value("--reference");
    else if (arg == "--warmup") {
      options.warmup = ParseCount(value("--warmup"), "warmup", true);
    } else if (arg == "--repeat") {
      options.repeat = ParseCount(value("--repeat"), "repeat", false);
    }
    else Die("unknown argument " + arg);
  }
  if (options.mode != "compile" && options.mode != "run") {
    Die("--mode must be compile or run");
  }
  if (options.blob.empty() || options.inputs.empty() || options.reference.empty()) {
    Die("--blob, --inputs, and --reference are required");
  }
  if (options.mode == "compile" && options.xml.empty()) {
    Die("--xml is required in compile mode");
  }
  return options;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto options = Parse(argc, argv);
    Runtime runtime;
    if (options.mode == "compile") {
      GraphRun graph(runtime, IQ36_ZE_GRAPH_FORMAT_NGRAPH_LITE,
                     NGraphLiteBundle(runtime, options.xml));
      FillDeterministicInputs(graph);
      const auto timings_us =
          graph.InitializeAndExecute(options.warmup, options.repeat);
      const auto inputs = graph.Capture(graph.Inputs());
      const auto outputs = graph.Capture(graph.Outputs());
      const auto blob = graph.NativeBinary();
      WriteBuffers(options.inputs, inputs);
      WriteBuffers(options.reference, outputs);
      WriteBytes(options.blob, blob);
      std::cout << "{\"mode\":\"compile\",\"device\":\""
                << runtime.device_properties.name << "\",\"graph_extension\":"
                << static_cast<std::uint32_t>(
                       runtime.graph_properties.graphExtensionVersion)
                << ",\"compiler_major\":"
                << runtime.graph_properties.compilerVersion.major
                << ",\"compiler_minor\":"
                << runtime.graph_properties.compilerVersion.minor
                << ",\"input_count\":" << inputs.size()
                << ",\"output_count\":" << outputs.size()
                << ",\"native_blob_bytes\":" << blob.size()
                << ",\"openvino_mapped\":"
                << (MapsContain("openvino") ? "true" : "false")
                << ",\"npu_driver_compiler_mapped\":"
                << (MapsContain("libnpu_driver_compiler") ? "true" : "false");
      PrintTimings(timings_us);
      std::cout << "}\n";
    } else {
      GraphRun graph(runtime, IQ36_ZE_GRAPH_FORMAT_NATIVE,
                     ReadBytes(options.blob));
      RestoreInputs(graph, ReadBuffers(options.inputs));
      const auto timings_us =
          graph.InitializeAndExecute(options.warmup, options.repeat);
      const auto outputs = graph.Capture(graph.Outputs());
      const auto expected = ReadBuffers(options.reference);
      const auto [compared, mismatches] = Compare(outputs, expected);
      std::cout << "{\"mode\":\"run\",\"device\":\""
                << runtime.device_properties.name
                << "\",\"compared_bytes\":" << compared
                << ",\"mismatch_bytes\":" << mismatches
                << ",\"openvino_mapped\":"
                << (MapsContain("openvino") ? "true" : "false")
                << ",\"npu_driver_compiler_mapped\":"
                << (MapsContain("libnpu_driver_compiler") ? "true" : "false");
      PrintTimings(timings_us);
      std::cout << "}\n";
      return mismatches == 0 ? 0 : 2;
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
