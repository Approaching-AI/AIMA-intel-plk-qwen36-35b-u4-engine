#include <oneapi/dnnl/dnnl.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

constexpr int kExperts = 256;
constexpr int kAssignments = 8192;
constexpr int kHidden = 2048;
constexpr int kIntermediate = 512;
constexpr int kQuantGroup = 64;

struct Args {
  std::string offsets;
  int warmup = 3;
  int repeat = 11;
  double cap_us = 0.0;
};

[[noreturn]] void Fail(const std::string& message) {
  throw std::runtime_error(message);
}

int PositiveInteger(const std::string& value, const char* name) {
  std::size_t consumed = 0;
  const long parsed = std::stol(value, &consumed);
  if (consumed != value.size() || parsed <= 0 || parsed > 1000000) {
    Fail(std::string(name) + " must be a positive integer");
  }
  return static_cast<int>(parsed);
}

double PositiveDouble(const std::string& value, const char* name) {
  std::size_t consumed = 0;
  const double parsed = std::stod(value, &consumed);
  if (consumed != value.size() || !std::isfinite(parsed) || parsed <= 0.0) {
    Fail(std::string(name) + " must be positive and finite");
  }
  return parsed;
}

Args ParseArgs(int argc, char** argv) {
  Args args;
  for (int index = 1; index < argc; ++index) {
    const std::string option = argv[index];
    const auto Value = [&]() -> std::string {
      if (++index >= argc) Fail(option + " requires a value");
      return argv[index];
    };
    if (option == "--offsets") args.offsets = Value();
    else if (option == "--warmup") {
      args.warmup = PositiveInteger(Value(), "warmup");
    } else if (option == "--repeat") {
      args.repeat = PositiveInteger(Value(), "repeat");
    } else if (option == "--cap-us") {
      args.cap_us = PositiveDouble(Value(), "cap-us");
    } else {
      Fail("unknown option: " + option);
    }
  }
  if (args.offsets.empty() || args.cap_us <= 0.0) {
    Fail("--offsets and --cap-us are required");
  }
  return args;
}

std::vector<std::int32_t> ReadOffsets(const std::string& path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) Fail("could not open offsets: " + path);
  const std::streamsize size = input.tellg();
  if (size != static_cast<std::streamsize>(
                  kExperts * sizeof(std::int32_t))) {
    Fail("offset file must contain exactly 256 int32 values");
  }
  input.seekg(0);
  std::vector<std::int32_t> offsets(kExperts);
  input.read(reinterpret_cast<char*>(offsets.data()), size);
  if (!input) Fail("could not read offsets: " + path);
  std::int32_t previous = 0;
  for (std::int32_t offset : offsets) {
    if (offset < previous) Fail("offsets must be nondecreasing");
    previous = offset;
  }
  if (offsets.back() != kAssignments) {
    Fail("final offset must equal 8192 assignments");
  }
  return offsets;
}

void Fill(const dnnl::memory& memory, std::uint8_t byte, int index = 0) {
  void* mapped = memory.map_data(index);
  if (mapped == nullptr) Fail("oneDNN returned a null mapped pointer");
  std::memset(mapped, byte, memory.get_desc().get_size(index));
  memory.unmap_data(mapped, index);
}

void WriteOffsets(const dnnl::memory& memory,
                  const std::vector<std::int32_t>& offsets) {
  if (memory.get_desc().get_size(1) !=
      offsets.size() * sizeof(std::int32_t)) {
    Fail("grouped offset buffer size mismatch");
  }
  void* mapped = memory.map_data(1);
  if (mapped == nullptr) Fail("oneDNN returned null grouped offsets");
  std::memcpy(mapped, offsets.data(), offsets.size() * sizeof(std::int32_t));
  memory.unmap_data(mapped, 1);
}

struct Job {
  int k;
  int n;
  dnnl::memory source;
  dnnl::memory weights;
  dnnl::memory destination;
  dnnl::memory weight_scales;
  dnnl::memory weight_zero_points;
  dnnl::memory max_group_hint;
  dnnl::matmul::primitive_desc descriptor;
  dnnl::matmul primitive;
  std::unordered_map<int, dnnl::memory> arguments;
  std::string implementation;

  Job(const dnnl::engine& engine, int input_width, int output_width,
      const std::vector<std::int32_t>& offsets, std::int32_t max_group)
      : k(input_width),
        n(output_width),
        source(dnnl::memory::desc::grouped(
                   {kAssignments, k}, dnnl::memory::data_type::f16, 0,
                   kExperts),
               engine),
        weights(dnnl::memory::desc(
                    {kExperts, k, n}, dnnl::memory::data_type::u4,
                    dnnl::memory::format_tag::acb),
                engine),
        destination(dnnl::memory::desc::grouped(
                        {kAssignments, n}, dnnl::memory::data_type::f16, 0,
                        kExperts),
                    engine),
        weight_scales(dnnl::memory::desc(
                          {kExperts, k / kQuantGroup, n},
                          dnnl::memory::data_type::f16,
                          dnnl::memory::format_tag::abc),
                      engine),
        weight_zero_points(dnnl::memory::desc(
                               {kExperts, k / kQuantGroup, n},
                               dnnl::memory::data_type::u4,
                               dnnl::memory::format_tag::abc),
                           engine),
        max_group_hint(dnnl::memory::desc::host_scalar(
                           dnnl::memory::data_type::s32),
                       max_group),
        descriptor(CreateDescriptor(engine, source.get_desc(),
                                    weights.get_desc(), destination.get_desc())),
        primitive(descriptor),
        implementation(descriptor.impl_info_str()) {
    arguments = {
        {DNNL_ARG_SRC, source},
        {DNNL_ARG_WEIGHTS, weights},
        {DNNL_ARG_DST, destination},
        {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, weight_scales},
        {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS, weight_zero_points},
        {DNNL_ARG_HINT_MAX_GROUP_SIZE, max_group_hint},
    };
    Fill(source, 0, 0);
    WriteOffsets(source, offsets);
    Fill(destination, 0, 0);
    WriteOffsets(destination, offsets);
    Fill(weights, 0x11);
    Fill(weight_scales, 0x3c);  // finite synthetic F16 payload
    Fill(weight_zero_points, 0);
  }

  static dnnl::matmul::primitive_desc CreateDescriptor(
      const dnnl::engine& engine, const dnnl::memory::desc& source,
      const dnnl::memory::desc& weights,
      const dnnl::memory::desc& destination) {
    dnnl::primitive_attr attributes;
    attributes.set_scales(DNNL_ARG_WEIGHTS, 7, {kQuantGroup, 1},
                          dnnl::memory::data_type::f16);
    attributes.set_zero_points(DNNL_ARG_WEIGHTS, 7, {kQuantGroup, 1},
                               dnnl::memory::data_type::u4);
    attributes.set_fpmath_mode(dnnl::fpmath_mode::f16, true);
    return dnnl::matmul::primitive_desc(
        engine, source, weights, destination, attributes);
  }

  std::uint64_t WeightBytes() const {
    return weights.get_desc().get_size() +
           weight_scales.get_desc().get_size() +
           weight_zero_points.get_desc().get_size();
  }
};

std::string Escape(const std::string& value) {
  std::string result;
  for (char character : value) {
    if (character == '\\' || character == '"') result.push_back('\\');
    result.push_back(character);
  }
  return result;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args args = ParseArgs(argc, argv);
    const std::vector<std::int32_t> offsets = ReadOffsets(args.offsets);
    std::int32_t previous = 0;
    std::int32_t max_group = 0;
    int active_experts = 0;
    for (std::int32_t offset : offsets) {
      const std::int32_t size = offset - previous;
      max_group = std::max(max_group, size);
      active_experts += size > 0;
      previous = offset;
    }

    dnnl::engine engine(dnnl::engine::kind::gpu, 0);
    dnnl::stream stream(engine);
    Job gate(engine, kHidden, kIntermediate, offsets, max_group);
    Job up(engine, kHidden, kIntermediate, offsets, max_group);
    Job down(engine, kIntermediate, kHidden, offsets, max_group);
    const bool implementations_pass =
        gate.implementation.find("grouped_gemm:micro") != std::string::npos &&
        up.implementation.find("grouped_gemm:micro") != std::string::npos &&
        down.implementation.find("grouped_gemm:micro") != std::string::npos;

    const auto Execute = [&]() {
      gate.primitive.execute(stream, gate.arguments);
      up.primitive.execute(stream, up.arguments);
      down.primitive.execute(stream, down.arguments);
      stream.wait();
    };
    for (int iteration = 0; iteration < args.warmup; ++iteration) Execute();
    std::vector<double> samples_us;
    samples_us.reserve(args.repeat);
    for (int iteration = 0; iteration < args.repeat; ++iteration) {
      const auto begin = std::chrono::steady_clock::now();
      Execute();
      const auto end = std::chrono::steady_clock::now();
      samples_us.push_back(
          std::chrono::duration<double, std::micro>(end - begin).count());
    }
    std::vector<double> sorted = samples_us;
    std::sort(sorted.begin(), sorted.end());
    const double minimum_us = sorted.front();
    const double median_us = sorted[sorted.size() / 2];
    const double mean_us =
        std::accumulate(samples_us.begin(), samples_us.end(), 0.0) /
        samples_us.size();
    const bool performance_pass = minimum_us <= args.cap_us;
    const std::uint64_t resident_weight_bytes =
        gate.WeightBytes() + up.WeightBytes() + down.WeightBytes();
    const dnnl_version_t* version = dnnl_version();

    std::cout << std::boolalpha << std::setprecision(12) << "{";
    std::cout << "\"active_experts\":" << active_experts << ",";
    std::cout << "\"assignment_count\":" << kAssignments << ",";
    std::cout << "\"cap_us\":" << args.cap_us << ",";
    std::cout << "\"implementations\":[\"" << Escape(gate.implementation)
              << "\",\"" << Escape(up.implementation) << "\",\""
              << Escape(down.implementation) << "\"],";
    std::cout << "\"implementations_pass\":" << implementations_pass << ",";
    std::cout << "\"max_group_size\":" << max_group << ",";
    std::cout << "\"mean_us\":" << mean_us << ",";
    std::cout << "\"median_us\":" << median_us << ",";
    std::cout << "\"minimum_us\":" << minimum_us << ",";
    std::cout << "\"onednn_version\":{\"hash\":\""
              << Escape(version->hash == nullptr ? "" : version->hash)
              << "\",\"major\":" << version->major
              << ",\"minor\":" << version->minor
              << ",\"patch\":" << version->patch << "},";
    std::cout << "\"performance_lower_bound_only\":true,";
    std::cout << "\"performance_pass\":" << performance_pass << ",";
    std::cout << "\"resident_weight_bytes\":" << resident_weight_bytes
              << ",";
    std::cout << "\"samples_us\":[";
    for (std::size_t index = 0; index < samples_us.size(); ++index) {
      if (index != 0) std::cout << ',';
      std::cout << samples_us[index];
    }
    std::cout << "]}" << std::endl;
    return implementations_pass && performance_pass ? 0 : 2;
  } catch (const dnnl::error& error) {
    std::cerr << "oneDNN status " << error.status << ": " << error.what()
              << '\n';
    return 3;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 4;
  }
}
