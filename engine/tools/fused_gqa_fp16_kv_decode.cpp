#include <CL/cl.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr cl_uint kContextTokens = 131072;
constexpr cl_uint kHeadDim = 256;
constexpr cl_uint kQHeads = 16;
constexpr cl_uint kKvHeads = 2;
constexpr cl_uint kGqaGroup = 8;
constexpr cl_uint kChunkTokens = 256;
constexpr cl_uint kLocalSize = 256;
constexpr float kAttentionScale = 0.0625f;
constexpr double kComponentCapMs = 2.825;

[[noreturn]] void Die(const std::string& message) {
  throw std::runtime_error(message);
}

void Require(bool condition, const std::string& message) {
  if (!condition) Die(message);
}

void Check(cl_int status, const std::string& operation) {
  if (status != CL_SUCCESS) {
    Die(operation + " failed with OpenCL status " + std::to_string(status));
  }
}

std::string ReadText(const std::string& path) {
  std::ifstream input(path);
  Require(static_cast<bool>(input), "failed to open OpenCL source");
  return std::string(
      std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

std::string DeviceString(cl_device_id device, cl_device_info key) {
  std::size_t bytes = 0;
  Check(clGetDeviceInfo(device, key, 0, nullptr, &bytes),
        "clGetDeviceInfo(size)");
  std::string value(bytes, '\0');
  Check(clGetDeviceInfo(device, key, bytes, value.data(), nullptr),
        "clGetDeviceInfo(value)");
  while (!value.empty() && value.back() == '\0') value.pop_back();
  return value;
}

cl_device_id SelectGpu() {
  cl_uint platform_count = 0;
  Check(clGetPlatformIDs(0, nullptr, &platform_count), "clGetPlatformIDs(count)");
  std::vector<cl_platform_id> platforms(platform_count);
  Check(clGetPlatformIDs(platform_count, platforms.data(), nullptr),
        "clGetPlatformIDs(values)");
  for (cl_platform_id platform : platforms) {
    cl_uint device_count = 0;
    const cl_int status = clGetDeviceIDs(
        platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &device_count);
    if (status == CL_DEVICE_NOT_FOUND || device_count == 0) continue;
    Check(status, "clGetDeviceIDs(count)");
    std::vector<cl_device_id> devices(device_count);
    Check(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, device_count,
                         devices.data(), nullptr),
          "clGetDeviceIDs(values)");
    for (cl_device_id device : devices) {
      if (DeviceString(device, CL_DEVICE_NAME).find("Intel") !=
          std::string::npos) {
        return device;
      }
    }
  }
  Die("Intel GPU OpenCL device not found");
}

cl_mem Buffer(cl_context context, cl_mem_flags flags, std::size_t bytes,
              void* host = nullptr) {
  cl_int status = CL_SUCCESS;
  cl_mem result = clCreateBuffer(context, flags, bytes, host, &status);
  Check(status, "clCreateBuffer");
  return result;
}

template <typename T>
void Arg(cl_kernel kernel, cl_uint index, const T& value) {
  Check(clSetKernelArg(kernel, index, sizeof(T), &value), "clSetKernelArg");
}

double EventMs(cl_event event) {
  cl_ulong begin = 0;
  cl_ulong end = 0;
  Check(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START,
                                sizeof(begin), &begin, nullptr),
        "clGetEventProfilingInfo(start)");
  Check(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,
                                sizeof(end), &end, nullptr),
        "clGetEventProfilingInfo(end)");
  return static_cast<double>(end - begin) / 1.0e6;
}

struct TimedPair {
  double partial_ms = 0.0;
  double reduce_ms = 0.0;
  double total_ms = 0.0;
};

TimedPair RunFused(cl_command_queue queue, cl_kernel partial, cl_kernel reduce,
                   std::size_t partial_global, std::size_t reduce_global) {
  const std::size_t local = kLocalSize;
  cl_event partial_event = nullptr;
  cl_event reduce_event = nullptr;
  Check(clEnqueueNDRangeKernel(queue, partial, 1, nullptr, &partial_global,
                               &local, 0, nullptr, &partial_event),
        "clEnqueueNDRangeKernel(partial)");
  Check(clEnqueueNDRangeKernel(queue, reduce, 1, nullptr, &reduce_global,
                               &local, 0, nullptr, &reduce_event),
        "clEnqueueNDRangeKernel(reduce)");
  Check(clFinish(queue), "clFinish(fused)");
  TimedPair result;
  result.partial_ms = EventMs(partial_event);
  result.reduce_ms = EventMs(reduce_event);
  result.total_ms = result.partial_ms + result.reduce_ms;
  clReleaseEvent(partial_event);
  clReleaseEvent(reduce_event);
  return result;
}

struct Numeric {
  double cosine = 0.0;
  double relative_l2 = 0.0;
  double rmse = 0.0;
  double max_abs = 0.0;
  bool finite = true;
};

Numeric Compare(const std::vector<float>& reference,
                const std::vector<float>& candidate) {
  Require(reference.size() == candidate.size() && !reference.empty(),
          "numeric comparison shape mismatch");
  double dot = 0.0;
  double ref_l2 = 0.0;
  double cand_l2 = 0.0;
  double diff_l2 = 0.0;
  double max_abs = 0.0;
  bool finite = true;
  for (std::size_t i = 0; i < reference.size(); ++i) {
    const double ref = reference[i];
    const double cand = candidate[i];
    finite = finite && std::isfinite(ref) && std::isfinite(cand);
    const double diff = cand - ref;
    dot += ref * cand;
    ref_l2 += ref * ref;
    cand_l2 += cand * cand;
    diff_l2 += diff * diff;
    max_abs = std::max(max_abs, std::abs(diff));
  }
  Numeric result;
  result.cosine = dot / (std::sqrt(ref_l2) * std::sqrt(cand_l2));
  result.relative_l2 = std::sqrt(diff_l2 / ref_l2);
  result.rmse = std::sqrt(diff_l2 / static_cast<double>(reference.size()));
  result.max_abs = max_abs;
  result.finite = finite;
  return result;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 2 && argc != 3) {
      throw std::invalid_argument(
          "usage: iq36-fused-gqa-fp16-kv-decode OPENCL_SOURCE [xmx]");
    }
    const bool xmx = argc == 3 && std::string(argv[2]) == "xmx";
    Require(argc == 2 || xmx, "unsupported component mode");
    const auto device = SelectGpu();
    cl_int status = CL_SUCCESS;
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr,
                                         &status);
    Check(status, "clCreateContext");
    cl_command_queue queue = clCreateCommandQueue(
        context, device, CL_QUEUE_PROFILING_ENABLE, &status);
    Check(status, "clCreateCommandQueue");
    const std::string source = ReadText(argv[1]);
    const char* source_data = source.data();
    const std::size_t source_size = source.size();
    cl_program program = clCreateProgramWithSource(
        context, 1, &source_data, &source_size, &status);
    Check(status, "clCreateProgramWithSource");
    status = clBuildProgram(program, 1, &device, "-cl-std=CL3.0", nullptr,
                            nullptr);
    if (status != CL_SUCCESS) {
      std::size_t log_bytes = 0;
      clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr,
                            &log_bytes);
      std::string log(log_bytes, '\0');
      clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_bytes,
                            log.data(), nullptr);
      std::cerr << log << '\n';
      Check(status, "clBuildProgram");
    }
    auto Kernel = [&](const char* name) {
      cl_kernel kernel = clCreateKernel(program, name, &status);
      Check(status, std::string("clCreateKernel(") + name + ")");
      return kernel;
    };
    cl_kernel convert = Kernel("iq36_f32_to_f16");
    cl_kernel pack_k = xmx ? Kernel("iq36_pack_k_dpas16") : nullptr;
    cl_kernel partial = Kernel(xmx ? "iq36_xmx_gqa_fp16_partial"
                                   : "iq36_fused_gqa_fp16_partial");
    cl_kernel reduce = Kernel(xmx ? "iq36_xmx_gqa_partial_reduce"
                                  : "iq36_fused_gqa_partial_reduce");
    cl_kernel reference_score = Kernel("iq36_reference_score_f32");
    cl_kernel reference_apply = Kernel("iq36_reference_apply_f32");

    const std::size_t history_values =
        static_cast<std::size_t>(kContextTokens) * kKvHeads * kHeadDim;
    const std::size_t output_values =
        static_cast<std::size_t>(kQHeads) * kHeadDim;
    const cl_uint chunk_count =
        (kContextTokens + kChunkTokens - 1U) / kChunkTokens;
    const std::size_t group_count =
        static_cast<std::size_t>(kKvHeads) * chunk_count;
    const std::size_t meta_values = group_count * kGqaGroup;
    const std::size_t partial_values = meta_values * kHeadDim;

    std::vector<float> q(output_values);
    std::vector<float> gate(output_values);
    for (std::size_t i = 0; i < output_values; ++i) {
      q[i] = static_cast<float>(static_cast<int>((i * 29U + 17U) & 255U) - 128)
          / 1024.0f;
      gate[i] = static_cast<float>(static_cast<int>((i * 11U + 3U) & 127U) - 64)
          / 64.0f;
    }
    constexpr std::size_t kPatternTokens = 256;
    std::vector<float> k_pattern(kPatternTokens * kKvHeads * kHeadDim);
    std::vector<float> v_pattern(kPatternTokens * kKvHeads * kHeadDim);
    for (std::size_t token = 0; token < kPatternTokens; ++token) {
      for (std::size_t dim = 0; dim < kKvHeads * kHeadDim; ++dim) {
        const std::size_t index = token * kKvHeads * kHeadDim + dim;
        k_pattern[index] = static_cast<float>(
            static_cast<int>((token * 13U + dim * 7U + 5U) & 255U) - 128)
            / 2048.0f;
        v_pattern[index] = 0.02f + static_cast<float>(
            static_cast<int>((token * 3U + dim * 19U + 9U) & 255U) - 128)
            / 4096.0f;
      }
    }
    std::vector<float> k_history(history_values);
    std::vector<float> v_history(history_values);
    const std::size_t pattern_values = k_pattern.size();
    for (std::size_t offset = 0; offset < history_values;
         offset += pattern_values) {
      const std::size_t count =
          std::min(pattern_values, history_values - offset);
      std::memcpy(k_history.data() + offset, k_pattern.data(),
                  count * sizeof(float));
      std::memcpy(v_history.data() + offset, v_pattern.data(),
                  count * sizeof(float));
    }

    cl_mem q_buffer = Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                             q.size() * sizeof(float), q.data());
    cl_mem gate_buffer = Buffer(
        context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        gate.size() * sizeof(float), gate.data());
    cl_mem k_f32 = Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                          k_history.size() * sizeof(float), k_history.data());
    cl_mem v_f32 = Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                          v_history.size() * sizeof(float), v_history.data());
    cl_mem k_f16 = Buffer(context, CL_MEM_READ_WRITE,
                          history_values * sizeof(std::uint16_t));
    cl_mem v_f16 = Buffer(context, CL_MEM_READ_WRITE,
                          history_values * sizeof(std::uint16_t));
    cl_mem q_f16 = Buffer(context, CL_MEM_READ_WRITE,
                          output_values * sizeof(std::uint16_t));
    cl_mem scores = Buffer(context, CL_MEM_READ_WRITE,
                           static_cast<std::size_t>(kQHeads) * kContextTokens *
                               sizeof(float));
    cl_mem reference_output = Buffer(context, CL_MEM_READ_WRITE,
                                     output_values * sizeof(float));
    cl_mem fused_output = Buffer(context, CL_MEM_READ_WRITE,
                                 output_values * sizeof(float));
    cl_mem partial_max = Buffer(context, CL_MEM_READ_WRITE,
                                meta_values * sizeof(float));
    cl_mem partial_sum = Buffer(context, CL_MEM_READ_WRITE,
                                meta_values * sizeof(float));
    cl_mem partial_output = Buffer(context, CL_MEM_READ_WRITE,
                                   partial_values * sizeof(float));
    k_history.clear();
    v_history.clear();
    k_history.shrink_to_fit();
    v_history.shrink_to_fit();

    const cl_uint history_count = static_cast<cl_uint>(history_values);
    const std::size_t convert_global =
        ((history_values + kLocalSize - 1U) / kLocalSize) * kLocalSize;
    const std::size_t local = kLocalSize;
    if (xmx) {
      Arg(pack_k, 0, k_f32);
      Arg(pack_k, 1, kContextTokens);
      Arg(pack_k, 2, k_f16);
      Check(clEnqueueNDRangeKernel(queue, pack_k, 1, nullptr, &convert_global,
                                   &local, 0, nullptr, nullptr),
            "clEnqueueNDRangeKernel(pack K)");
    } else {
      Arg(convert, 0, k_f32);
      Arg(convert, 1, k_f16);
      Arg(convert, 2, history_count);
      Check(clEnqueueNDRangeKernel(queue, convert, 1, nullptr, &convert_global,
                                   &local, 0, nullptr, nullptr),
            "clEnqueueNDRangeKernel(convert K)");
    }
    Arg(convert, 0, v_f32);
    Arg(convert, 1, v_f16);
    Arg(convert, 2, history_count);
    Check(clEnqueueNDRangeKernel(queue, convert, 1, nullptr, &convert_global,
                                 &local, 0, nullptr, nullptr),
          "clEnqueueNDRangeKernel(convert V)");
    const cl_uint output_count = static_cast<cl_uint>(output_values);
    const std::size_t q_global =
        ((output_values + local - 1U) / local) * local;
    Arg(convert, 0, q_buffer);
    Arg(convert, 1, q_f16);
    Arg(convert, 2, output_count);
    Check(clEnqueueNDRangeKernel(queue, convert, 1, nullptr, &q_global,
                                 &local, 0, nullptr, nullptr),
          "clEnqueueNDRangeKernel(convert Q)");
    Check(clFinish(queue), "clFinish(convert)");

    Arg(reference_score, 0, q_buffer);
    Arg(reference_score, 1, k_f32);
    Arg(reference_score, 2, kContextTokens);
    Arg(reference_score, 3, kAttentionScale);
    Arg(reference_score, 4, scores);
    const std::size_t score_items =
        static_cast<std::size_t>(kQHeads) * kContextTokens;
    const std::size_t score_global =
        ((score_items + local - 1U) / local) * local;
    Check(clEnqueueNDRangeKernel(queue, reference_score, 1, nullptr,
                                 &score_global, &local, 0, nullptr, nullptr),
          "clEnqueueNDRangeKernel(reference score)");
    Arg(reference_apply, 0, scores);
    Arg(reference_apply, 1, v_f32);
    Arg(reference_apply, 2, gate_buffer);
    Arg(reference_apply, 3, kContextTokens);
    Arg(reference_apply, 4, reference_output);
    const std::size_t output_global = output_values;
    Check(clEnqueueNDRangeKernel(queue, reference_apply, 1, nullptr,
                                 &output_global, &local, 0, nullptr, nullptr),
          "clEnqueueNDRangeKernel(reference apply)");
    Check(clFinish(queue), "clFinish(reference)");

    Arg(partial, 0, xmx ? q_f16 : q_buffer);
    Arg(partial, 1, k_f16);
    Arg(partial, 2, v_f16);
    Arg(partial, 3, kContextTokens);
    if (xmx) {
      Arg(partial, 4, partial_max);
      Arg(partial, 5, partial_sum);
      Arg(partial, 6, partial_output);
    } else {
      Arg(partial, 4, kAttentionScale);
      Arg(partial, 5, partial_max);
      Arg(partial, 6, partial_sum);
      Arg(partial, 7, partial_output);
    }
    Arg(reduce, 0, partial_max);
    Arg(reduce, 1, partial_sum);
    Arg(reduce, 2, partial_output);
    Arg(reduce, 3, gate_buffer);
    Arg(reduce, 4, kContextTokens);
    Arg(reduce, 5, fused_output);
    const std::size_t partial_global = group_count * local;
    const std::size_t reduce_global = output_values;
    for (int warmup = 0; warmup < 3; ++warmup) {
      (void)RunFused(queue, partial, reduce, partial_global, reduce_global);
    }
    const auto repeat =
        RunFused(queue, partial, reduce, partial_global, reduce_global);
    const auto confirm =
        RunFused(queue, partial, reduce, partial_global, reduce_global);

    std::vector<float> reference(output_values);
    std::vector<float> candidate(output_values);
    Check(clEnqueueReadBuffer(queue, reference_output, CL_TRUE, 0,
                              reference.size() * sizeof(float),
                              reference.data(), 0, nullptr, nullptr),
          "clEnqueueReadBuffer(reference)");
    Check(clEnqueueReadBuffer(queue, fused_output, CL_TRUE, 0,
                              candidate.size() * sizeof(float),
                              candidate.data(), 0, nullptr, nullptr),
          "clEnqueueReadBuffer(candidate)");
    const auto numeric = Compare(reference, candidate);
    const double spread =
        std::abs(repeat.total_ms - confirm.total_ms) /
        std::max(repeat.total_ms, confirm.total_ms);
    const bool numeric_pass = numeric.finite && numeric.cosine >= 0.999 &&
        numeric.relative_l2 <= 0.002;
    const bool timing_pass = repeat.total_ms <= kComponentCapMs &&
        confirm.total_ms <= kComponentCapMs && spread <= 0.005;
    const bool pass = numeric_pass && timing_pass;

    std::cout << std::boolalpha << std::setprecision(12) << "{"
              << "\"algorithm\":\""
              << (xmx ? "xmx_gqa_flash" : "scalar_gqa_fused") << "\","
              << "\"chunk_tokens\":" << kChunkTokens << ","
              << "\"component_cap_ms\":" << kComponentCapMs << ","
              << "\"confirm\":{\"partial_ms\":" << confirm.partial_ms
              << ",\"reduce_ms\":" << confirm.reduce_ms
              << ",\"total_ms\":" << confirm.total_ms << "},"
              << "\"context_tokens\":" << kContextTokens << ","
              << "\"device\":\"" << DeviceString(device, CL_DEVICE_NAME)
              << "\","
              << "\"finite\":" << numeric.finite << ","
              << "\"gqa_group\":" << kGqaGroup << ","
              << "\"head_dim\":" << kHeadDim << ","
              << "\"kv_dtype\":\"fp16\","
              << "\"kv_head_count\":" << kKvHeads << ","
              << "\"max_abs\":" << numeric.max_abs << ","
              << "\"numeric_pass\":" << numeric_pass << ","
              << "\"output_cosine\":" << numeric.cosine << ","
              << "\"output_relative_l2\":" << numeric.relative_l2 << ","
              << "\"output_rmse\":" << numeric.rmse << ","
              << "\"partial_bytes\":"
              << partial_values * sizeof(float) + 2U * meta_values * sizeof(float)
              << ","
              << "\"q_head_count\":" << kQHeads << ","
              << "\"repeat\":{\"partial_ms\":" << repeat.partial_ms
              << ",\"reduce_ms\":" << repeat.reduce_ms
              << ",\"total_ms\":" << repeat.total_ms << "},"
              << "\"required_checks_passed\":" << pass << ","
              << "\"spread\":" << spread << ","
              << "\"subgroup_size\":" << (xmx ? 16 : 32) << ","
              << "\"token_tile\":" << (xmx ? 16 : 0) << ","
              << "\"timing_pass\":" << timing_pass << "}" << std::endl;

    for (cl_mem buffer : {q_buffer, gate_buffer, k_f32, v_f32, k_f16, v_f16,
                          q_f16,
                          scores, reference_output, fused_output, partial_max,
                          partial_sum, partial_output}) {
      clReleaseMemObject(buffer);
    }
    for (cl_kernel kernel : {convert, partial, reduce, reference_score,
                             reference_apply}) {
      clReleaseKernel(kernel);
    }
    if (pack_k != nullptr) clReleaseKernel(pack_k);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return pass ? 0 : 2;
  } catch (const std::exception& exception) {
    std::cerr << "iq36-fused-gqa-fp16-kv-decode: " << exception.what() << '\n';
    return 4;
  }
}
