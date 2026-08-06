#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_ocl.hpp>

#include <CL/cl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
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

constexpr std::int64_t kInputWidth = 2048;
constexpr std::int64_t kGateUpWidth = 1024;

struct Args {
  std::vector<std::pair<int, int>> buckets;
  int actual_assignments = 8192;
  int expected_experts = 222;
  int warmup = 3;
  int repeat = 11;
  double kernel_cap_us = 0.0;
};

[[noreturn]] void Fail(const std::string& message) {
  throw std::runtime_error(message);
}

int ParsePositive(const std::string& value, const char* name) {
  std::size_t consumed = 0;
  const long parsed = std::stol(value, &consumed);
  if (consumed != value.size() || parsed <= 0 || parsed > 1'000'000) {
    Fail(std::string(name) + " must be a positive integer");
  }
  return static_cast<int>(parsed);
}

double ParsePositiveDouble(const std::string& value, const char* name) {
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
    const auto require_value = [&]() -> std::string {
      if (++index >= argc) Fail(option + " requires a value");
      return argv[index];
    };
    if (option == "--bucket") {
      const std::string value = require_value();
      const std::size_t separator = value.find(':');
      if (separator == std::string::npos) Fail("bucket must be M:EXPERTS");
      const int m = ParsePositive(value.substr(0, separator), "bucket M");
      const int experts =
          ParsePositive(value.substr(separator + 1), "bucket experts");
      args.buckets.emplace_back(m, experts);
    } else if (option == "--actual-assignments") {
      args.actual_assignments = ParsePositive(require_value(), option.c_str());
    } else if (option == "--expected-experts") {
      args.expected_experts = ParsePositive(require_value(), option.c_str());
    } else if (option == "--warmup") {
      args.warmup = ParsePositive(require_value(), option.c_str());
    } else if (option == "--repeat") {
      args.repeat = ParsePositive(require_value(), option.c_str());
    } else if (option == "--kernel-cap-us") {
      args.kernel_cap_us =
          ParsePositiveDouble(require_value(), option.c_str());
    } else {
      Fail("unknown option: " + option);
    }
  }
  if (args.buckets.empty()) Fail("at least one --bucket is required");
  if (args.kernel_cap_us <= 0.0) Fail("--kernel-cap-us is required");
  std::sort(args.buckets.begin(), args.buckets.end());
  for (std::size_t index = 0; index < args.buckets.size(); ++index) {
    const int m = args.buckets[index].first;
    if (m < 8 || m > 512 || (m & (m - 1)) != 0) {
      Fail("bucket M must be a power of two in [8, 512]");
    }
    if (index != 0 && args.buckets[index - 1].first == m) {
      Fail("bucket M values must be unique");
    }
  }
  return args;
}

std::string JsonEscape(const std::string& value) {
  std::string escaped;
  for (const unsigned char character : value) {
    switch (character) {
      case '\\': escaped += "\\\\"; break;
      case '"': escaped += "\\\""; break;
      case '\n': escaped += "\\n"; break;
      case '\r': escaped += "\\r"; break;
      case '\t': escaped += "\\t"; break;
      default:
        if (character >= 0x20) escaped += static_cast<char>(character);
    }
  }
  return escaped;
}

std::string DeviceString(cl_device_id device, cl_device_info field) {
  std::size_t size = 0;
  if (clGetDeviceInfo(device, field, 0, nullptr, &size) != CL_SUCCESS) {
    return "unknown";
  }
  std::string value(size, '\0');
  if (clGetDeviceInfo(device, field, size, value.data(), nullptr) != CL_SUCCESS) {
    return "unknown";
  }
  while (!value.empty() && value.back() == '\0') value.pop_back();
  return value;
}

void FillMemory(const dnnl::memory& memory, unsigned char byte) {
  void* mapped = memory.map_data();
  if (mapped == nullptr) Fail("oneDNN returned a null mapped pointer");
  std::memset(mapped, byte, memory.get_desc().get_size());
  memory.unmap_data(mapped);
}

struct Job {
  int m;
  int experts;
  dnnl::memory src;
  dnnl::memory weights;
  dnnl::memory dst;
  dnnl::matmul primitive;
  std::unordered_map<int, dnnl::memory> arguments;
  std::string implementation;

  Job(const dnnl::engine& engine, int m_in, int experts_in)
      : m(m_in),
        experts(experts_in),
        src(dnnl::memory::desc({experts, m, kInputWidth},
                               dnnl::memory::data_type::s8,
                               dnnl::memory::format_tag::abc),
            engine),
        weights(dnnl::memory::desc({experts, kInputWidth, kGateUpWidth},
                                   dnnl::memory::data_type::u4,
                                   dnnl::memory::format_tag::abc),
            engine),
        dst(dnnl::memory::desc({experts, m, kGateUpWidth},
                               dnnl::memory::data_type::f32,
                               dnnl::memory::format_tag::abc),
            engine),
        primitive(dnnl::matmul::primitive_desc(
            engine, src.get_desc(), weights.get_desc(), dst.get_desc())),
        arguments{{DNNL_ARG_SRC, src},
                  {DNNL_ARG_WEIGHTS, weights},
                  {DNNL_ARG_DST, dst}} {
    const char* name = nullptr;
    const dnnl_status_t status = dnnl_primitive_desc_query(
        primitive.get_primitive_desc(), dnnl_query_impl_info_str, 0, &name);
    if (status != dnnl_success || name == nullptr) {
      Fail("could not query oneDNN implementation name");
    }
    implementation = name;
    FillMemory(src, 0x01);
    FillMemory(weights, 0x11);
    FillMemory(dst, 0x00);
  }

  std::uint64_t SourceBytes() const { return src.get_desc().get_size(); }
  std::uint64_t WeightBytes() const { return weights.get_desc().get_size(); }
  std::uint64_t DestinationBytes() const { return dst.get_desc().get_size(); }
};

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args args = ParseArgs(argc, argv);
    const dnnl::engine engine(dnnl::engine::kind::gpu, 0);
    dnnl::stream stream(engine);
    const cl_device_id device = dnnl::ocl_interop::get_device(engine);
    std::vector<std::unique_ptr<Job>> jobs;
    jobs.reserve(args.buckets.size());
    for (const auto& [m, experts] : args.buckets) {
      jobs.emplace_back(std::make_unique<Job>(engine, m, experts));
    }

    const int observed_experts = std::accumulate(
        jobs.begin(), jobs.end(), 0,
        [](int sum, const std::unique_ptr<Job>& job) {
          return sum + job->experts;
        });
    const std::uint64_t padded_assignments = std::accumulate(
        jobs.begin(), jobs.end(), std::uint64_t{0},
        [](std::uint64_t sum, const std::unique_ptr<Job>& job) {
          return sum + static_cast<std::uint64_t>(job->m) * job->experts;
        });
    if (observed_experts != args.expected_experts) {
      Fail("bucket expert total does not match --expected-experts");
    }
    if (padded_assignments < static_cast<std::uint64_t>(args.actual_assignments)) {
      Fail("padded assignments are below actual assignments");
    }

    for (int iteration = 0; iteration < args.warmup; ++iteration) {
      for (const auto& job : jobs) {
        job->primitive.execute(stream, job->arguments);
      }
      stream.wait();
    }

    std::vector<double> samples_us;
    samples_us.reserve(args.repeat);
    for (int iteration = 0; iteration < args.repeat; ++iteration) {
      const auto begin = std::chrono::steady_clock::now();
      for (const auto& job : jobs) {
        job->primitive.execute(stream, job->arguments);
      }
      stream.wait();
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
    const std::uint64_t source_bytes = std::accumulate(
        jobs.begin(), jobs.end(), std::uint64_t{0},
        [](std::uint64_t sum, const std::unique_ptr<Job>& job) {
          return sum + job->SourceBytes();
        });
    const std::uint64_t weight_bytes = std::accumulate(
        jobs.begin(), jobs.end(), std::uint64_t{0},
        [](std::uint64_t sum, const std::unique_ptr<Job>& job) {
          return sum + job->WeightBytes();
        });
    const std::uint64_t destination_bytes = std::accumulate(
        jobs.begin(), jobs.end(), std::uint64_t{0},
        [](std::uint64_t sum, const std::unique_ptr<Job>& job) {
          return sum + job->DestinationBytes();
        });
    const std::uint64_t timed_bytes =
        source_bytes + weight_bytes + destination_bytes;
    const bool implementations_pass = std::all_of(
        jobs.begin(), jobs.end(), [](const std::unique_ptr<Job>& job) {
          return job->implementation.find("jit:gemm") != std::string::npos;
        });
    const bool performance_pass = minimum_us <= args.kernel_cap_us;
    const dnnl_version_t* version = dnnl_version();

    std::cout << std::boolalpha << std::setprecision(12) << "{";
    std::cout << "\"actual_assignments\":" << args.actual_assignments << ",";
    std::cout << "\"buckets\":[";
    for (std::size_t index = 0; index < jobs.size(); ++index) {
      if (index != 0) std::cout << ",";
      const Job& job = *jobs[index];
      std::cout << "{\"experts\":" << job.experts << ",";
      std::cout << "\"implementation\":\""
                << JsonEscape(job.implementation) << "\",";
      std::cout << "\"m\":" << job.m << ",";
      std::cout << "\"source_bytes\":" << job.SourceBytes() << ",";
      std::cout << "\"weight_bytes\":" << job.WeightBytes() << ",";
      std::cout << "\"destination_bytes\":" << job.DestinationBytes()
                << "}";
    }
    std::cout << "],";
    std::cout << "\"destination_bytes\":" << destination_bytes << ",";
    std::cout << "\"device_name\":\""
              << JsonEscape(DeviceString(device, CL_DEVICE_NAME)) << "\",";
    std::cout << "\"driver_version\":\""
              << JsonEscape(DeviceString(device, CL_DRIVER_VERSION)) << "\",";
    std::cout << "\"effective_gb_s\":"
              << timed_bytes / (minimum_us * 1000.0) << ",";
    std::cout << "\"expected_experts\":" << args.expected_experts << ",";
    std::cout << "\"implementations_pass\":" << implementations_pass << ",";
    std::cout << "\"kernel_cap_us\":" << args.kernel_cap_us << ",";
    std::cout << "\"minimum_us\":" << minimum_us << ",";
    std::cout << "\"mean_us\":" << mean_us << ",";
    std::cout << "\"median_us\":" << median_us << ",";
    std::cout << "\"oneapi_version\":\""
              << JsonEscape(DeviceString(device, CL_DEVICE_VERSION)) << "\",";
    std::cout << "\"onednn_version\":{\"hash\":\""
              << JsonEscape(version->hash == nullptr ? "" : version->hash)
              << "\",\"major\":" << version->major
              << ",\"minor\":" << version->minor
              << ",\"patch\":" << version->patch << "},";
    std::cout << "\"padded_assignments\":" << padded_assignments << ",";
    std::cout << "\"padding_ratio\":"
              << static_cast<double>(padded_assignments) /
                         args.actual_assignments -
                     1.0
              << ",";
    std::cout << "\"performance_pass\":" << performance_pass << ",";
    std::cout << "\"raw_u4_core_only\":true,";
    std::cout << "\"samples_us\":[";
    for (std::size_t index = 0; index < samples_us.size(); ++index) {
      if (index != 0) std::cout << ",";
      std::cout << samples_us[index];
    }
    std::cout << "],";
    std::cout << "\"source_bytes\":" << source_bytes << ",";
    std::cout << "\"timed_bytes\":" << timed_bytes << ",";
    std::cout << "\"weight_bytes\":" << weight_bytes;
    std::cout << "}\n";
    return implementations_pass && performance_pass ? 0 : 2;
  } catch (const dnnl::error& error) {
    std::cerr << "oneDNN status " << error.status << ": " << error.what()
              << "\n";
    return 3;
  } catch (const std::exception& error) {
    std::cerr << error.what() << "\n";
    return 4;
  }
}
