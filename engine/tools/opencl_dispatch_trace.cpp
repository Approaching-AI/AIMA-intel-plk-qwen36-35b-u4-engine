// Copyright (c) 2026
// Trace fixed OpenCL kernel dispatch metadata for offline native replay gates.

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <CL/cl_ext.h>

#include <dlfcn.h>
#include <fcntl.h>
#include <link.h>
#include <unistd.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

struct ArgValue {
  std::size_t size = 0;
  bool svm = false;
  std::vector<unsigned char> bytes;
};

struct UsmAllocation {
  std::size_t size = 0;
  const char* kind = "unknown";
};

struct ProgramInfo {
  std::size_t source_bytes = 0;
  bool stock_gdn = false;
  bool custom_gdn = false;
  bool stock_sdpa = false;
  bool custom_sdpa = false;
  std::size_t ordinal = 0;
  std::string source;
};

std::mutex g_mutex;
std::unordered_map<cl_kernel, std::vector<ArgValue>> g_args;
std::unordered_map<cl_mem, std::size_t> g_mem_sizes;
std::unordered_map<const void*, UsmAllocation> g_usm_allocations;
std::unordered_map<cl_program, ProgramInfo> g_programs;
std::size_t g_program_ordinal = 0;
bool g_stock_program_dumped = false;
bool g_custom_program_dumped = false;
std::size_t g_stock_sdpa_programs_dumped = 0;
std::size_t g_custom_sdpa_programs_dumped = 0;
bool g_capture_claimed = false;
using GetExtensionFn = void* (CL_API_CALL*)(cl_platform_id, const char*);
GetExtensionFn g_get_extension = nullptr;
clHostMemAllocINTEL_fn g_host_alloc = nullptr;
clDeviceMemAllocINTEL_fn g_device_alloc = nullptr;
clSharedMemAllocINTEL_fn g_shared_alloc = nullptr;
clMemFreeINTEL_fn g_mem_free = nullptr;
clMemBlockingFreeINTEL_fn g_mem_blocking_free = nullptr;
clSetKernelArgMemPointerINTEL_fn g_set_mem_arg = nullptr;

template <typename T>
T Resolve(const char* name) {
  static void* loader = dlopen(
      "/lib/x86_64-linux-gnu/libOpenCL.so.1", RTLD_NOW | RTLD_LOCAL);
  void* symbol = loader == nullptr ? nullptr : ::dlsym(loader, name);
  if (symbol == nullptr) symbol = ::dlsym(RTLD_NEXT, name);
  if (symbol == nullptr) {
    const char prefix[] = "iq36 OpenCL trace: unresolved symbol ";
    write(STDERR_FILENO, prefix, sizeof(prefix) - 1);
    write(STDERR_FILENO, name, std::strlen(name));
    write(STDERR_FILENO, "\n", 1);
    _exit(127);
  }
  return reinterpret_cast<T>(symbol);
}

const char* Env(const char* name) {
  const char* value = std::getenv(name);
  return value == nullptr ? "" : value;
}

std::size_t EnvSize(const char* name, std::size_t fallback) {
  const char* value = Env(name);
  if (value[0] == '\0') return fallback;
  char* end = nullptr;
  const unsigned long long parsed = std::strtoull(value, &end, 10);
  return end != value && end != nullptr && end[0] == '\0'
             ? static_cast<std::size_t>(parsed)
             : fallback;
}

std::string JsonEscape(const std::string& value) {
  std::ostringstream out;
  for (unsigned char ch : value) {
    switch (ch) {
      case '\\': out << "\\\\"; break;
      case '"': out << "\\\""; break;
      case '\n': out << "\\n"; break;
      case '\r': out << "\\r"; break;
      case '\t': out << "\\t"; break;
      default:
        if (ch < 0x20) {
          out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
              << static_cast<unsigned>(ch) << std::dec;
        } else {
          out << static_cast<char>(ch);
        }
    }
  }
  return out.str();
}

std::string ReadMarker() {
  const char* path = Env("IQ36_OPENCL_TRACE_MARKER");
  if (path[0] == '\0') return "active";
  int fd = open(path, O_RDONLY);
  if (fd < 0) return {};
  std::array<char, 128> buffer{};
  const ssize_t count = read(fd, buffer.data(), buffer.size() - 1);
  close(fd);
  if (count <= 0) return "active";
  std::string value(buffer.data(), static_cast<std::size_t>(count));
  while (!value.empty() &&
         (value.back() == '\n' || value.back() == '\r' || value.back() == ' ')) {
    value.pop_back();
  }
  return value;
}

std::string KernelName(cl_kernel kernel) {
  using Fn = cl_int(CL_API_CALL*)(cl_kernel, cl_kernel_info, std::size_t,
                                  void*, std::size_t*);
  static Fn real = Resolve<Fn>("clGetKernelInfo");
  if (real == nullptr) return {};
  std::size_t size = 0;
  if (real(kernel, CL_KERNEL_FUNCTION_NAME, 0, nullptr, &size) != CL_SUCCESS ||
      size == 0 || size > 4096) {
    return {};
  }
  std::string name(size, '\0');
  if (real(kernel, CL_KERNEL_FUNCTION_NAME, size, name.data(), nullptr) !=
      CL_SUCCESS) {
    return {};
  }
  if (!name.empty() && name.back() == '\0') name.pop_back();
  return name;
}

bool Selected(const std::string& name) {
  const char* filter = Env("IQ36_OPENCL_TRACE_FILTER");
  const std::string needles = filter[0] == '\0' ? "sdpa" : filter;
  std::size_t begin = 0;
  while (begin <= needles.size()) {
    const std::size_t end = needles.find(',', begin);
    const std::string needle = needles.substr(
        begin, end == std::string::npos ? std::string::npos : end - begin);
    if (!needle.empty() && name.find(needle) != std::string::npos) return true;
    if (end == std::string::npos) break;
    begin = end + 1;
  }
  return false;
}

bool ContainsAny(const std::string& value, const char* environment_name) {
  const std::string needles = Env(environment_name);
  if (needles.empty()) return true;
  std::size_t begin = 0;
  while (begin <= needles.size()) {
    const std::size_t end = needles.find(',', begin);
    const std::string needle = needles.substr(
        begin, end == std::string::npos ? std::string::npos : end - begin);
    if (!needle.empty() && value.find(needle) != std::string::npos) return true;
    if (end == std::string::npos) break;
    begin = end + 1;
  }
  return false;
}

bool ArgMatchesInt32(const std::vector<ArgValue>& args, std::size_t index,
                     const char* environment_name) {
  const char* expected_text = Env(environment_name);
  if (expected_text[0] == '\0') return true;
  if (index >= args.size() || args[index].bytes.size() != sizeof(std::int32_t)) {
    return false;
  }
  char* end = nullptr;
  const long expected = std::strtol(expected_text, &end, 10);
  if (end == expected_text || end == nullptr || end[0] != '\0') return false;
  std::int32_t observed = 0;
  std::memcpy(&observed, args[index].bytes.data(), sizeof(observed));
  return observed == expected;
}

bool CaptureArgumentsMatch(const std::vector<ArgValue>& args) {
  return ArgMatchesInt32(args, 9, "IQ36_OPENCL_CAPTURE_GEMM_M") &&
         ArgMatchesInt32(args, 10, "IQ36_OPENCL_CAPTURE_GEMM_N") &&
         ArgMatchesInt32(args, 11, "IQ36_OPENCL_CAPTURE_GEMM_K");
}

std::string Hex(
    const std::vector<unsigned char>& bytes, std::size_t max_bytes = 32) {
  std::ostringstream out;
  out << std::hex << std::setfill('0');
  const std::size_t limit = std::min<std::size_t>(bytes.size(), max_bytes);
  for (std::size_t i = 0; i < limit; ++i) {
    out << std::setw(2) << static_cast<unsigned>(bytes[i]);
  }
  return out.str();
}

void AppendLog(const std::string& line) {
  const char* path = Env("IQ36_OPENCL_TRACE_PATH");
  if (path[0] == '\0') return;
  int fd = open(path, O_CREAT | O_WRONLY | O_APPEND, 0644);
  if (fd < 0) return;
  const std::string record = line + "\n";
  const char* cursor = record.data();
  std::size_t remaining = record.size();
  while (remaining != 0) {
    const ssize_t count = write(fd, cursor, remaining);
    if (count <= 0) break;
    cursor += count;
    remaining -= static_cast<std::size_t>(count);
  }
  close(fd);
}

bool WriteBytes(
    const std::string& path, const void* data, std::size_t size) {
  int fd = open(path.c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0644);
  if (fd < 0) return false;
  const char* cursor = static_cast<const char*>(data);
  std::size_t remaining = size;
  while (remaining != 0) {
    const ssize_t count = write(fd, cursor, remaining);
    if (count <= 0) {
      close(fd);
      return false;
    }
    cursor += count;
    remaining -= static_cast<std::size_t>(count);
  }
  close(fd);
  return true;
}

bool ClaimProgramDump(const ProgramInfo& info) {
  if (Env("IQ36_OPENCL_PROGRAM_DUMP_DIR")[0] == '\0') return false;
  std::lock_guard<std::mutex> lock(g_mutex);
  const std::size_t stock_sdpa_limit =
      EnvSize("IQ36_OPENCL_STOCK_SDPA_DUMP_LIMIT", 4);
  const std::size_t custom_sdpa_limit =
      EnvSize("IQ36_OPENCL_CUSTOM_SDPA_DUMP_LIMIT", 4);
  const bool already_dumped =
      (info.stock_gdn && g_stock_program_dumped) ||
      (info.custom_gdn && g_custom_program_dumped) ||
      (info.stock_sdpa &&
       g_stock_sdpa_programs_dumped >= stock_sdpa_limit) ||
      (info.custom_sdpa &&
       g_custom_sdpa_programs_dumped >= custom_sdpa_limit);
  if (already_dumped) return false;
  if (info.stock_gdn) g_stock_program_dumped = true;
  if (info.custom_gdn) g_custom_program_dumped = true;
  if (info.stock_sdpa) ++g_stock_sdpa_programs_dumped;
  if (info.custom_sdpa) ++g_custom_sdpa_programs_dumped;
  return true;
}

void DumpProgramArtifacts(cl_program program, const ProgramInfo& info) {
  const char* directory = Env("IQ36_OPENCL_PROGRAM_DUMP_DIR");
  if (directory[0] == '\0' || !ClaimProgramDump(info)) return;

  const char* kind = info.custom_sdpa
                         ? "custom-sdpa"
                         : (info.stock_sdpa
                                ? "stock-sdpa"
                                : (info.custom_gdn ? "custom-gdn"
                                                   : "stock-gdn"));
  std::ostringstream stem_builder;
  stem_builder << directory << '/' << kind << "-program" << std::setw(3)
               << std::setfill('0') << info.ordinal;
  const std::string stem = stem_builder.str();
  const std::string source_path = stem + ".cl";
  const bool source_written =
      WriteBytes(source_path, info.source.data(), info.source.size());

  using ProgramInfoFn = cl_int(CL_API_CALL*)(
      cl_program, cl_program_info, std::size_t, void*, std::size_t*);
  static ProgramInfoFn get_program_info =
      Resolve<ProgramInfoFn>("clGetProgramInfo");
  cl_uint device_count = 0;
  cl_int status = get_program_info(
      program, CL_PROGRAM_NUM_DEVICES, sizeof(device_count), &device_count,
      nullptr);
  std::vector<std::size_t> sizes(device_count);
  if (status == CL_SUCCESS && device_count != 0) {
    status = get_program_info(
        program, CL_PROGRAM_BINARY_SIZES,
        sizes.size() * sizeof(sizes[0]), sizes.data(), nullptr);
  }
  std::vector<std::vector<unsigned char>> binaries(device_count);
  std::vector<unsigned char*> binary_pointers(device_count, nullptr);
  if (status == CL_SUCCESS) {
    for (std::size_t index = 0; index < device_count; ++index) {
      binaries[index].resize(sizes[index]);
      binary_pointers[index] = binaries[index].data();
    }
    status = get_program_info(
        program, CL_PROGRAM_BINARIES,
        binary_pointers.size() * sizeof(binary_pointers[0]),
        binary_pointers.data(), nullptr);
  }

  std::ostringstream paths;
  paths << '[';
  bool binaries_written = status == CL_SUCCESS && device_count != 0;
  for (std::size_t index = 0; index < device_count; ++index) {
    if (index != 0) paths << ',';
    std::ostringstream binary_path;
    binary_path << stem << "-device" << index << ".bin";
    const bool written =
        status == CL_SUCCESS && !binaries[index].empty() &&
        WriteBytes(binary_path.str(), binaries[index].data(),
                   binaries[index].size());
    binaries_written = binaries_written && written;
    paths << "{\"path\":\"" << JsonEscape(binary_path.str())
          << "\",\"bytes\":" << binaries[index].size()
          << ",\"written\":" << (written ? "true" : "false") << '}';
  }
  paths << ']';

  std::ostringstream record;
  record << "{\"event\":\"program_dump\",\"stock_gdn\":"
         << (info.stock_gdn ? "true" : "false")
         << ",\"custom_gdn\":" << (info.custom_gdn ? "true" : "false")
         << ",\"stock_sdpa\":" << (info.stock_sdpa ? "true" : "false")
         << ",\"custom_sdpa\":" << (info.custom_sdpa ? "true" : "false")
         << ",\"ordinal\":" << info.ordinal
         << ",\"marker\":\"" << JsonEscape(ReadMarker()) << "\""
         << ",\"source_path\":\"" << JsonEscape(source_path)
         << "\",\"source_bytes\":" << info.source.size()
         << ",\"source_written\":" << (source_written ? "true" : "false")
         << ",\"query_status\":" << status
         << ",\"binaries_written\":"
         << (binaries_written ? "true" : "false")
         << ",\"binaries\":" << paths.str() << '}';
  AppendLog(record.str());
}

std::size_t KernelProgramOrdinal(cl_kernel kernel) {
  using Fn = cl_int(CL_API_CALL*)(cl_kernel, cl_kernel_info, std::size_t,
                                  void*, std::size_t*);
  static Fn real = Resolve<Fn>("clGetKernelInfo");
  cl_program program = nullptr;
  if (real(kernel, CL_KERNEL_PROGRAM, sizeof(program), &program, nullptr) !=
          CL_SUCCESS ||
      program == nullptr) {
    return std::numeric_limits<std::size_t>::max();
  }
  std::lock_guard<std::mutex> lock(g_mutex);
  const auto found = g_programs.find(program);
  return found == g_programs.end()
             ? std::numeric_limits<std::size_t>::max()
             : found->second.ordinal;
}

bool ClaimFirstCapture() {
  if (Env("IQ36_OPENCL_CAPTURE_DIR")[0] == '\0') return false;
  std::lock_guard<std::mutex> lock(g_mutex);
  if (g_capture_claimed) return false;
  g_capture_claimed = true;
  return true;
}

struct PointerRange {
  const void* pointer = nullptr;
  std::size_t available = 0;
  const char* kind = "unknown";
};

PointerRange FindPointerRange(
    const ArgValue& arg,
    const std::unordered_map<const void*, UsmAllocation>& allocations) {
  if (!arg.svm || arg.bytes.size() != sizeof(void*)) return {};
  const void* pointer = nullptr;
  std::memcpy(&pointer, arg.bytes.data(), sizeof(pointer));
  const auto address = reinterpret_cast<std::uintptr_t>(pointer);
  for (const auto& allocation : allocations) {
    const auto base = reinterpret_cast<std::uintptr_t>(allocation.first);
    const auto size = allocation.second.size;
    if (address < base || address - base >= size) continue;
    return {pointer, size - (address - base), allocation.second.kind};
  }
  return {};
}

cl_int DumpUsmPointer(
    cl_command_queue queue, const PointerRange& range,
    cl_uint wait_count, const cl_event* wait_list,
    const std::string& path) {
  using QueueInfoFn = cl_int(CL_API_CALL*)(
      cl_command_queue, cl_command_queue_info, std::size_t, void*,
      std::size_t*);
  using DeviceInfoFn = cl_int(CL_API_CALL*)(
      cl_device_id, cl_device_info, std::size_t, void*, std::size_t*);
  static QueueInfoFn queue_info = Resolve<QueueInfoFn>("clGetCommandQueueInfo");
  static DeviceInfoFn device_info = Resolve<DeviceInfoFn>("clGetDeviceInfo");

  cl_context context = nullptr;
  cl_device_id device = nullptr;
  cl_platform_id platform = nullptr;
  cl_int status = queue_info(
      queue, CL_QUEUE_CONTEXT, sizeof(context), &context, nullptr);
  if (status != CL_SUCCESS) return status;
  status = queue_info(
      queue, CL_QUEUE_DEVICE, sizeof(device), &device, nullptr);
  if (status != CL_SUCCESS) return status;
  status = device_info(
      device, CL_DEVICE_PLATFORM, sizeof(platform), &platform, nullptr);
  if (status != CL_SUCCESS) return status;

  GetExtensionFn get_extension = g_get_extension;
  if (get_extension == nullptr) {
    get_extension = Resolve<GetExtensionFn>(
        "clGetExtensionFunctionAddressForPlatform");
  }
  auto host_alloc = reinterpret_cast<clHostMemAllocINTEL_fn>(
      get_extension(platform, "clHostMemAllocINTEL"));
  auto copy = reinterpret_cast<clEnqueueMemcpyINTEL_fn>(
      get_extension(platform, "clEnqueueMemcpyINTEL"));
  auto blocking_free = reinterpret_cast<clMemBlockingFreeINTEL_fn>(
      get_extension(platform, "clMemBlockingFreeINTEL"));
  auto free = reinterpret_cast<clMemFreeINTEL_fn>(
      get_extension(platform, "clMemFreeINTEL"));
  if (host_alloc == nullptr || copy == nullptr ||
      (blocking_free == nullptr && free == nullptr)) {
    return CL_INVALID_OPERATION;
  }

  cl_int alloc_status = CL_SUCCESS;
  void* host = host_alloc(
      context, nullptr, range.available, 0, &alloc_status);
  if (host == nullptr || alloc_status != CL_SUCCESS) return alloc_status;
  status = copy(queue, CL_TRUE, host, range.pointer, range.available,
                wait_count, wait_list, nullptr);
  if (status == CL_SUCCESS) {
    int fd = open(path.c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0644);
    if (fd < 0) {
      status = CL_OUT_OF_HOST_MEMORY;
    } else {
      const char* cursor = static_cast<const char*>(host);
      std::size_t remaining = range.available;
      while (remaining != 0) {
        const ssize_t count = write(fd, cursor, remaining);
        if (count <= 0) {
          status = CL_OUT_OF_HOST_MEMORY;
          break;
        }
        cursor += count;
        remaining -= static_cast<std::size_t>(count);
      }
      close(fd);
    }
  }
  if (blocking_free != nullptr) {
    blocking_free(context, host);
  } else {
    free(context, host);
  }
  return status;
}

void CaptureArguments(
    cl_command_queue queue, const std::vector<ArgValue>& args,
    const std::unordered_map<const void*, UsmAllocation>& allocations,
    const char* phase, std::size_t begin, std::size_t end,
    cl_uint wait_count, const cl_event* wait_list) {
  const char* directory = Env("IQ36_OPENCL_CAPTURE_DIR");
  if (directory[0] == '\0') return;
  end = std::min(end, args.size());
  for (std::size_t index = begin; index < end; ++index) {
    PointerRange range = FindPointerRange(args[index], allocations);
    if (range.pointer == nullptr || range.available == 0) continue;
    range.available = std::min(
        range.available,
        EnvSize("IQ36_OPENCL_CAPTURE_MAX_BYTES", range.available));
    std::ostringstream path;
    path << directory << "/dispatch000-arg" << index << '-' << phase
         << ".bin";
    const cl_int status = DumpUsmPointer(
        queue, range, wait_count, wait_list, path.str());
    std::ostringstream record;
    record << "{\"event\":\"capture\",\"marker\":\""
           << JsonEscape(ReadMarker()) << "\",\"phase\":\"" << phase
           << "\",\"arg_index\":" << index << ",\"status\":" << status
           << ",\"bytes\":" << range.available << ",\"usm_kind\":\""
           << range.kind << "\",\"path\":\""
           << JsonEscape(path.str()) << "\"}";
    AppendLog(record.str());
  }
}

std::string WorkArray(const std::size_t* values, cl_uint dimensions) {
  if (values == nullptr) return "null";
  std::ostringstream out;
  out << '[';
  for (cl_uint i = 0; i < dimensions; ++i) {
    if (i != 0) out << ',';
    out << values[i];
  }
  out << ']';
  return out.str();
}

}  // namespace

extern "C" CL_API_ENTRY cl_program CL_API_CALL clCreateProgramWithSource(
    cl_context context, cl_uint count, const char** strings,
    const std::size_t* lengths, cl_int* errcode_ret) {
  using Fn = decltype(&clCreateProgramWithSource);
  static Fn real = Resolve<Fn>("clCreateProgramWithSource");
  cl_program program = real(context, count, strings, lengths, errcode_ret);
  if (program == nullptr) return program;

  ProgramInfo info;
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    info.ordinal = g_program_ordinal++;
  }
  for (cl_uint index = 0; index < count; ++index) {
    if (strings == nullptr || strings[index] == nullptr) continue;
    const std::size_t size =
        lengths != nullptr && lengths[index] != 0
            ? lengths[index]
            : std::strlen(strings[index]);
    info.source_bytes += size;
    const std::string source(strings[index], size);
    info.source.append(source);
    info.stock_gdn = info.stock_gdn ||
                     source.find("gated_delta_net_ref") != std::string::npos;
    info.custom_gdn = info.custom_gdn ||
                      source.find("iq36_gated_delta_net") != std::string::npos;
    info.stock_sdpa = info.stock_sdpa ||
                      source.find("sdpa_micro") != std::string::npos;
    info.custom_sdpa = info.custom_sdpa ||
                       source.find("iq36_hot_attention_") !=
                           std::string::npos;
  }
  if (info.stock_gdn || info.custom_gdn || info.stock_sdpa ||
      info.custom_sdpa) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_programs[program] = info;
  }
  return program;
}

extern "C" CL_API_ENTRY cl_int CL_API_CALL clBuildProgram(
    cl_program program, cl_uint device_count, const cl_device_id* devices,
    const char* options, void (CL_CALLBACK* callback)(cl_program, void*),
    void* user_data) {
  using Fn = decltype(&clBuildProgram);
  static Fn real = Resolve<Fn>("clBuildProgram");
  const cl_int status = real(
      program, device_count, devices, options, callback, user_data);
  ProgramInfo info;
  bool selected = false;
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    const auto found = g_programs.find(program);
    if (found != g_programs.end()) {
      info = found->second;
      selected = true;
    }
  }
  if (selected) {
    std::string build_log;
    if (status != CL_SUCCESS && device_count != 0 && devices != nullptr) {
      using BuildInfoFn = cl_int(CL_API_CALL*)(
          cl_program, cl_device_id, cl_program_build_info, std::size_t,
          void*, std::size_t*);
      static BuildInfoFn get_build_info =
          Resolve<BuildInfoFn>("clGetProgramBuildInfo");
      std::size_t build_log_size = 0;
      cl_int log_status = get_build_info(
          program, devices[0], CL_PROGRAM_BUILD_LOG, 0, nullptr,
          &build_log_size);
      if (log_status == CL_SUCCESS && build_log_size != 0) {
        build_log.resize(build_log_size);
        log_status = get_build_info(
            program, devices[0], CL_PROGRAM_BUILD_LOG, build_log.size(),
            build_log.data(), nullptr);
        if (log_status != CL_SUCCESS) build_log.clear();
      }
    }
    std::ostringstream out;
    out << "{\"event\":\"build_program\",\"stock_gdn\":"
        << (info.stock_gdn ? "true" : "false")
        << ",\"custom_gdn\":" << (info.custom_gdn ? "true" : "false")
        << ",\"stock_sdpa\":" << (info.stock_sdpa ? "true" : "false")
        << ",\"custom_sdpa\":" << (info.custom_sdpa ? "true" : "false")
        << ",\"ordinal\":" << info.ordinal
        << ",\"marker\":\"" << JsonEscape(ReadMarker()) << "\""
        << ",\"source_bytes\":" << info.source_bytes
        << ",\"options\":\""
        << JsonEscape(options == nullptr ? "" : options)
        << "\",\"status\":" << status
        << ",\"build_log\":\"" << JsonEscape(build_log) << "\"}";
    AppendLog(out.str());
    DumpProgramArtifacts(program, info);
  }
  return status;
}

extern "C" CL_API_ENTRY cl_int CL_API_CALL clReleaseProgram(
    cl_program program) {
  using Fn = decltype(&clReleaseProgram);
  static Fn real = Resolve<Fn>("clReleaseProgram");
  const cl_int status = real(program);
  // Kernels retain their program even after the caller drops its program
  // reference.  Keep this small process-lifetime map so later dispatch rows
  // can bind the kernel back to the exact captured program ordinal.
  return status;
}

extern "C" CL_API_ENTRY cl_mem CL_API_CALL clCreateBuffer(
    cl_context context, cl_mem_flags flags, std::size_t size, void* host_ptr,
    cl_int* errcode_ret) {
  using Fn = decltype(&clCreateBuffer);
  static Fn real = Resolve<Fn>("clCreateBuffer");
  cl_mem memory = real(context, flags, size, host_ptr, errcode_ret);
  if (memory != nullptr) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_mem_sizes[memory] = size;
  }
  return memory;
}

extern "C" CL_API_ENTRY cl_mem CL_API_CALL clCreateSubBuffer(
    cl_mem buffer, cl_mem_flags flags, cl_buffer_create_type create_type,
    const void* create_info, cl_int* errcode_ret) {
  using Fn = decltype(&clCreateSubBuffer);
  static Fn real = Resolve<Fn>("clCreateSubBuffer");
  cl_mem memory = real(buffer, flags, create_type, create_info, errcode_ret);
  if (memory != nullptr && create_type == CL_BUFFER_CREATE_TYPE_REGION &&
      create_info != nullptr) {
    const auto* region = static_cast<const cl_buffer_region*>(create_info);
    std::lock_guard<std::mutex> lock(g_mutex);
    g_mem_sizes[memory] = region->size;
  }
  return memory;
}

extern "C" CL_API_ENTRY cl_int CL_API_CALL clReleaseMemObject(cl_mem memory) {
  using Fn = decltype(&clReleaseMemObject);
  static Fn real = Resolve<Fn>("clReleaseMemObject");
  const cl_int status = real(memory);
  if (status == CL_SUCCESS) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_mem_sizes.erase(memory);
  }
  return status;
}

extern "C" CL_API_ENTRY cl_int CL_API_CALL clSetKernelArg(
    cl_kernel kernel, cl_uint index, std::size_t size, const void* value) {
  using Fn = decltype(&clSetKernelArg);
  static Fn real = Resolve<Fn>("clSetKernelArg");
  const cl_int status = real(kernel, index, size, value);
  if (status == CL_SUCCESS) {
    ArgValue arg;
    arg.size = size;
    if (value != nullptr && size <= 256) {
      const auto* bytes = static_cast<const unsigned char*>(value);
      arg.bytes.assign(bytes, bytes + size);
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    auto& args = g_args[kernel];
    if (args.size() <= index) args.resize(index + 1);
    args[index] = std::move(arg);
  }
  return status;
}

extern "C" CL_API_ENTRY cl_int CL_API_CALL clSetKernelArgSVMPointer(
    cl_kernel kernel, cl_uint index, const void* value) {
  using Fn = decltype(&clSetKernelArgSVMPointer);
  static Fn real = Resolve<Fn>("clSetKernelArgSVMPointer");
  const cl_int status = real(kernel, index, value);
  if (status == CL_SUCCESS) {
    ArgValue arg;
    arg.size = sizeof(value);
    arg.svm = true;
    const auto* bytes = reinterpret_cast<const unsigned char*>(&value);
    arg.bytes.assign(bytes, bytes + sizeof(value));
    std::lock_guard<std::mutex> lock(g_mutex);
    auto& args = g_args[kernel];
    if (args.size() <= index) args.resize(index + 1);
    args[index] = std::move(arg);
  }
  return status;
}

extern "C" CL_API_ENTRY void* CL_API_CALL clHostMemAllocINTEL(
    cl_context context, const cl_mem_properties_intel* properties,
    std::size_t size, cl_uint alignment, cl_int* errcode_ret) {
  void* memory = g_host_alloc(
      context, properties, size, alignment, errcode_ret);
  if (memory != nullptr) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_usm_allocations[memory] = {size, "host"};
  }
  return memory;
}

extern "C" CL_API_ENTRY void* CL_API_CALL clDeviceMemAllocINTEL(
    cl_context context, cl_device_id device,
    const cl_mem_properties_intel* properties, std::size_t size,
    cl_uint alignment, cl_int* errcode_ret) {
  void* memory = g_device_alloc(
      context, device, properties, size, alignment, errcode_ret);
  if (memory != nullptr) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_usm_allocations[memory] = {size, "device"};
  }
  return memory;
}

extern "C" CL_API_ENTRY void* CL_API_CALL clSharedMemAllocINTEL(
    cl_context context, cl_device_id device,
    const cl_mem_properties_intel* properties, std::size_t size,
    cl_uint alignment, cl_int* errcode_ret) {
  void* memory = g_shared_alloc(
      context, device, properties, size, alignment, errcode_ret);
  if (memory != nullptr) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_usm_allocations[memory] = {size, "shared"};
  }
  return memory;
}

extern "C" CL_API_ENTRY cl_int CL_API_CALL clMemFreeINTEL(
    cl_context context, void* memory) {
  const cl_int status = g_mem_free(context, memory);
  if (status == CL_SUCCESS) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_usm_allocations.erase(memory);
  }
  return status;
}

extern "C" CL_API_ENTRY cl_int CL_API_CALL clMemBlockingFreeINTEL(
    cl_context context, void* memory) {
  const cl_int status = g_mem_blocking_free(context, memory);
  if (status == CL_SUCCESS) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_usm_allocations.erase(memory);
  }
  return status;
}

extern "C" CL_API_ENTRY cl_int CL_API_CALL clSetKernelArgMemPointerINTEL(
    cl_kernel kernel, cl_uint index, const void* value) {
  const cl_int status = g_set_mem_arg(kernel, index, value);
  if (status == CL_SUCCESS) {
    ArgValue arg;
    arg.size = sizeof(value);
    arg.svm = true;
    const auto* bytes = reinterpret_cast<const unsigned char*>(&value);
    arg.bytes.assign(bytes, bytes + sizeof(value));
    std::lock_guard<std::mutex> lock(g_mutex);
    auto& args = g_args[kernel];
    if (args.size() <= index) args.resize(index + 1);
    args[index] = std::move(arg);
  }
  return status;
}

extern "C" CL_API_ENTRY void* CL_API_CALL
clGetExtensionFunctionAddressForPlatform(
    cl_platform_id platform, const char* name) {
  if (g_get_extension == nullptr) {
    g_get_extension = Resolve<GetExtensionFn>(
        "clGetExtensionFunctionAddressForPlatform");
  }
  void* function = g_get_extension(platform, name);
  if (function == nullptr || name == nullptr) return function;
  if (std::strcmp(name, "clHostMemAllocINTEL") == 0) {
    g_host_alloc = reinterpret_cast<clHostMemAllocINTEL_fn>(function);
    return reinterpret_cast<void*>(&clHostMemAllocINTEL);
  }
  if (std::strcmp(name, "clDeviceMemAllocINTEL") == 0) {
    g_device_alloc = reinterpret_cast<clDeviceMemAllocINTEL_fn>(function);
    return reinterpret_cast<void*>(&clDeviceMemAllocINTEL);
  }
  if (std::strcmp(name, "clSharedMemAllocINTEL") == 0) {
    g_shared_alloc = reinterpret_cast<clSharedMemAllocINTEL_fn>(function);
    return reinterpret_cast<void*>(&clSharedMemAllocINTEL);
  }
  if (std::strcmp(name, "clMemFreeINTEL") == 0) {
    g_mem_free = reinterpret_cast<clMemFreeINTEL_fn>(function);
    return reinterpret_cast<void*>(&clMemFreeINTEL);
  }
  if (std::strcmp(name, "clMemBlockingFreeINTEL") == 0) {
    g_mem_blocking_free = reinterpret_cast<clMemBlockingFreeINTEL_fn>(function);
    return reinterpret_cast<void*>(&clMemBlockingFreeINTEL);
  }
  if (std::strcmp(name, "clSetKernelArgMemPointerINTEL") == 0) {
    g_set_mem_arg = reinterpret_cast<clSetKernelArgMemPointerINTEL_fn>(function);
    return reinterpret_cast<void*>(&clSetKernelArgMemPointerINTEL);
  }
  return function;
}

extern "C" CL_API_ENTRY cl_int CL_API_CALL clReleaseKernel(cl_kernel kernel) {
  using Fn = decltype(&clReleaseKernel);
  static Fn real = Resolve<Fn>("clReleaseKernel");
  const cl_int status = real(kernel);
  if (status == CL_SUCCESS) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_args.erase(kernel);
  }
  return status;
}

extern "C" CL_API_ENTRY cl_int CL_API_CALL clEnqueueNDRangeKernel(
    cl_command_queue queue, cl_kernel kernel, cl_uint work_dim,
    const std::size_t* global_offset, const std::size_t* global_size,
    const std::size_t* local_size, cl_uint wait_count,
    const cl_event* wait_list, cl_event* event) {
  using Fn = decltype(&clEnqueueNDRangeKernel);
  static Fn real = Resolve<Fn>("clEnqueueNDRangeKernel");
  const std::string marker = ReadMarker();
  const std::string name = marker.empty() ? std::string() : KernelName(kernel);
  const bool selected = !marker.empty() && Selected(name);
  const std::size_t program_ordinal = selected
      ? KernelProgramOrdinal(kernel)
      : std::numeric_limits<std::size_t>::max();
  const bool timing = selected && std::strcmp(Env("IQ36_OPENCL_TRACE_TIMING"), "1") == 0;
  std::vector<ArgValue> args;
  std::unordered_map<cl_mem, std::size_t> mem_sizes;
  std::unordered_map<const void*, UsmAllocation> usm_allocations;
  if (selected) {
    std::lock_guard<std::mutex> lock(g_mutex);
    auto found = g_args.find(kernel);
    if (found != g_args.end()) args = found->second;
    mem_sizes = g_mem_sizes;
    usm_allocations = g_usm_allocations;
  }
  const bool capture = selected &&
      ContainsAny(marker, "IQ36_OPENCL_CAPTURE_MARKER_FILTER") &&
      CaptureArgumentsMatch(args) && ClaimFirstCapture();
  if (capture) {
    CaptureArguments(
        queue, args, usm_allocations, "before",
        EnvSize("IQ36_OPENCL_CAPTURE_BEFORE_BEGIN", 0),
        EnvSize("IQ36_OPENCL_CAPTURE_BEFORE_END", 6),
                     wait_count, wait_list);
  }
  cl_event local_event = nullptr;
  cl_event* event_out = event;
  if ((timing || capture) && event_out == nullptr) event_out = &local_event;
  const cl_int status = real(queue, kernel, work_dim, global_offset, global_size,
                             local_size, wait_count, wait_list, event_out);
  std::uint64_t duration_ns = 0;
  cl_int timing_status = CL_SUCCESS;
  cl_event timing_event = event_out == nullptr ? nullptr : *event_out;
  if (timing && status == CL_SUCCESS && timing_event != nullptr) {
    using WaitFn = cl_int(CL_API_CALL*)(cl_uint, const cl_event*);
    using ProfileFn = cl_int(CL_API_CALL*)(cl_event, cl_profiling_info,
                                           std::size_t, void*, std::size_t*);
    static WaitFn wait = Resolve<WaitFn>("clWaitForEvents");
    static ProfileFn profile = Resolve<ProfileFn>("clGetEventProfilingInfo");
    cl_ulong start = 0;
    cl_ulong end = 0;
    timing_status = wait(1, &timing_event);
    if (timing_status == CL_SUCCESS) {
      timing_status = profile(timing_event, CL_PROFILING_COMMAND_START,
                              sizeof(start), &start, nullptr);
    }
    if (timing_status == CL_SUCCESS) {
      timing_status = profile(timing_event, CL_PROFILING_COMMAND_END,
                              sizeof(end), &end, nullptr);
    }
    if (timing_status == CL_SUCCESS && end >= start) duration_ns = end - start;
  }
  if (!selected) return status;

  if (capture && status == CL_SUCCESS && timing_event != nullptr) {
    CaptureArguments(
        queue, args, usm_allocations, "after",
        EnvSize("IQ36_OPENCL_CAPTURE_AFTER_BEGIN", 6),
        EnvSize("IQ36_OPENCL_CAPTURE_AFTER_END", 8),
                     1, &timing_event);
  }
  std::ostringstream out;
  out << "{\"event\":\"ndrange\",\"marker\":\""
      << JsonEscape(marker) << "\",\"kernel\":\"" << JsonEscape(name)
      << "\",\"program_ordinal\":";
  if (program_ordinal == std::numeric_limits<std::size_t>::max()) {
    out << "null";
  } else {
    out << program_ordinal;
  }
  out << ",\"status\":" << status << ",\"work_dim\":" << work_dim
      << ",\"global_offset\":" << WorkArray(global_offset, work_dim)
      << ",\"global_size\":" << WorkArray(global_size, work_dim)
      << ",\"local_size\":" << WorkArray(local_size, work_dim)
      << ",\"duration_ns\":" << duration_ns
      << ",\"timing_status\":" << timing_status << ",\"args\":[";
  for (std::size_t i = 0; i < args.size(); ++i) {
    if (i != 0) out << ',';
    const ArgValue& arg = args[i];
    out << "{\"index\":" << i << ",\"size\":" << arg.size
        << ",\"svm\":" << (arg.svm ? "true" : "false")
        << ",\"hex\":\"" << Hex(arg.bytes) << "\"";
    if (!arg.svm && arg.bytes.size() == sizeof(cl_mem)) {
      cl_mem memory = nullptr;
      std::memcpy(&memory, arg.bytes.data(), sizeof(memory));
      auto found = mem_sizes.find(memory);
      if (found != mem_sizes.end()) out << ",\"mem_bytes\":" << found->second;
    } else if (arg.svm && arg.bytes.size() == sizeof(void*)) {
      const void* memory = nullptr;
      std::memcpy(&memory, arg.bytes.data(), sizeof(memory));
      const auto address = reinterpret_cast<std::uintptr_t>(memory);
      for (const auto& allocation : usm_allocations) {
        const auto base = reinterpret_cast<std::uintptr_t>(allocation.first);
        const auto size = allocation.second.size;
        if (address < base || address - base >= size) continue;
        const std::size_t offset = address - base;
        out << ",\"mem_bytes\":" << size
            << ",\"mem_offset\":" << offset
            << ",\"usm_kind\":\"" << allocation.second.kind << "\"";
        if (i == 0 && (std::strcmp(allocation.second.kind, "host") == 0 ||
                       std::strcmp(allocation.second.kind, "shared") == 0)) {
          const std::size_t available = size - offset;
          const std::size_t count = std::min<std::size_t>(available, 256);
          std::vector<unsigned char> head(count);
          std::memcpy(head.data(), memory, count);
          out << ",\"head_hex\":\"" << Hex(head, 256) << "\"";
        }
        break;
      }
    }
    out << '}';
  }
  out << "]}";
  AppendLog(out.str());
  if (local_event != nullptr) {
    using ReleaseFn = cl_int(CL_API_CALL*)(cl_event);
    static ReleaseFn release = Resolve<ReleaseFn>("clReleaseEvent");
    release(local_event);
  }
  return status;
}

// OpenVINO/oneDNN resolves OpenCL with dlopen+dlsym, bypassing normal
// LD_PRELOAD binding. The runtime-audit interface observes those bindings
// without replacing dlsym globally and redirects only the calls needed by the
// recorder. Returning the original st_value for everything else keeps the
// compiler/runtime loader untouched.
extern "C" unsigned int la_version(unsigned int) { return LAV_CURRENT; }

extern "C" unsigned int la_objopen(
    struct link_map*, Lmid_t, uintptr_t*) {
  return LA_FLG_BINDTO | LA_FLG_BINDFROM;
}

extern "C" uintptr_t la_symbind64(
    Elf64_Sym* symbol, unsigned int, uintptr_t*, uintptr_t*, unsigned int*,
    const char* name) {
  if (name == nullptr) return symbol->st_value;
  if (std::strcmp(name, "clCreateProgramWithSource") == 0)
    return reinterpret_cast<uintptr_t>(&clCreateProgramWithSource);
  if (std::strcmp(name, "clBuildProgram") == 0)
    return reinterpret_cast<uintptr_t>(&clBuildProgram);
  if (std::strcmp(name, "clReleaseProgram") == 0)
    return reinterpret_cast<uintptr_t>(&clReleaseProgram);
  if (std::strcmp(name, "clCreateBuffer") == 0)
    return reinterpret_cast<uintptr_t>(&clCreateBuffer);
  if (std::strcmp(name, "clCreateSubBuffer") == 0)
    return reinterpret_cast<uintptr_t>(&clCreateSubBuffer);
  if (std::strcmp(name, "clReleaseMemObject") == 0)
    return reinterpret_cast<uintptr_t>(&clReleaseMemObject);
  if (std::strcmp(name, "clSetKernelArg") == 0)
    return reinterpret_cast<uintptr_t>(&clSetKernelArg);
  if (std::strcmp(name, "clSetKernelArgSVMPointer") == 0)
    return reinterpret_cast<uintptr_t>(&clSetKernelArgSVMPointer);
  if (std::strcmp(name, "clReleaseKernel") == 0)
    return reinterpret_cast<uintptr_t>(&clReleaseKernel);
  if (std::strcmp(name, "clEnqueueNDRangeKernel") == 0)
    return reinterpret_cast<uintptr_t>(&clEnqueueNDRangeKernel);
  if (std::strcmp(name, "clGetExtensionFunctionAddressForPlatform") == 0)
    return reinterpret_cast<uintptr_t>(
        &clGetExtensionFunctionAddressForPlatform);
  return symbol->st_value;
}
