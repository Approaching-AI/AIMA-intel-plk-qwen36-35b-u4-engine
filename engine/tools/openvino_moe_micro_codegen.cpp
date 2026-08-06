#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <CL/cl.h>

#include "gemmstone/microkernel/fuser.hpp"
#include "gemmstone/microkernel/shim.hpp"
#include "gemmstone/microkernel_selector.hpp"

#ifndef CL_KERNEL_REGISTER_COUNT_INTEL
#define CL_KERNEL_REGISTER_COUNT_INTEL 0x425B
#endif

#ifndef CL_KERNEL_SPILL_MEM_SIZE_INTEL
#define CL_KERNEL_SPILL_MEM_SIZE_INTEL 0x4109
#endif

namespace {

namespace gs = ::gemmstone;
namespace micro = gs::microkernel;

struct Shape {
  std::string name;
  int m;
  int n;
  int k;
};

int g_quant_group_size = 32;
bool g_decode_fc = false;
bool g_prefill_fc = false;
bool g_attention_nhalf = false;
bool g_row_major_metadata = false;
bool g_u8_zero_point = false;
std::filesystem::path g_existing_shim;
std::string g_existing_kernel_name;
std::string g_host_define;
bool g_exact_attention_vrt160 = false;
bool g_exact_attention_dual_cohort = false;
std::string g_decode_name = "layer0_qkv";
std::string g_provider_commit =
    "20db47e2d3c4df1b66e93bed2e97d30da175512d";
int g_decode_m = 8192;
int g_prefill_n = 32;
int g_decode_k = 2048;
int g_attention_context = 131072;
int g_attention_kq_unroll_m = 16;
int g_register_file_size = 256;
std::filesystem::path g_dump_dir;
std::filesystem::path g_host_source;
std::vector<int> g_projection_widths;

template <typename T, typename = void>
struct HasEfficient64Bit : std::false_type {};

template <typename T>
struct HasEfficient64Bit<
    T, std::void_t<decltype(std::declval<T&>().isEfficient64Bit)>>
    : std::true_type {};

template <typename T>
void SetEfficient64Bit(T& hardware) {
  if constexpr (HasEfficient64Bit<T>::value) {
    // PTL/B390 is Xe3, below the Xe3p efficient-64-bit ABI boundary.
    hardware.isEfficient64Bit = false;
  }
}

std::vector<int> ParseWidths(const std::string& value) {
  std::vector<int> widths;
  std::size_t begin = 0;
  while (begin <= value.size()) {
    const std::size_t end = value.find(',', begin);
    const std::string part = value.substr(
        begin, end == std::string::npos ? std::string::npos : end - begin);
    if (part.empty()) {
      throw std::runtime_error("projection widths contain an empty value");
    }
    widths.push_back(std::stoi(part));
    if (end == std::string::npos) break;
    begin = end + 1;
  }
  return widths;
}

void CheckCl(cl_int status, const char* operation) {
  if (status != CL_SUCCESS) {
    throw std::runtime_error(
        std::string(operation) + " failed with OpenCL status " +
        std::to_string(status));
  }
}

std::string ReadText(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("could not open " + path.string());
  return std::string(std::istreambuf_iterator<char>(input), {});
}

std::string HostDefineDirective(const std::string& value) {
  if (value.empty()) return {};
  const std::size_t equals = value.find('=');
  const std::string name = value.substr(0, equals);
  const std::string replacement =
      equals == std::string::npos ? "1" : value.substr(equals + 1);
  const auto valid_name_character = [](unsigned char character) {
    return std::isalnum(character) || character == '_';
  };
  const auto valid_value_character = [](unsigned char character) {
    return std::isalnum(character) || character == '_';
  };
  if (name.empty() || replacement.empty() ||
      (!std::isalpha(static_cast<unsigned char>(name.front())) &&
       name.front() != '_') ||
      !std::all_of(name.begin(), name.end(), valid_name_character) ||
      !std::all_of(
          replacement.begin(), replacement.end(), valid_value_character)) {
    throw std::runtime_error(
        "--host-define must be a C identifier with an optional "
        "alphanumeric replacement");
  }
  return "#define " + name + " " + replacement + "\n";
}

std::string ProgramLog(cl_program program, cl_device_id device) {
  std::size_t size = 0;
  CheckCl(clGetProgramBuildInfo(
      program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &size),
      "clGetProgramBuildInfo size");
  std::string log(size, '\0');
  if (size != 0) {
    CheckCl(clGetProgramBuildInfo(
        program, device, CL_PROGRAM_BUILD_LOG, size, log.data(), nullptr),
        "clGetProgramBuildInfo log");
  }
  return log;
}

struct OpenClCompiler {
  cl_device_id device = nullptr;
  cl_context context = nullptr;

  OpenClCompiler() {
    cl_uint platform_count = 0;
    CheckCl(clGetPlatformIDs(0, nullptr, &platform_count),
            "clGetPlatformIDs count");
    std::vector<cl_platform_id> platforms(platform_count);
    CheckCl(clGetPlatformIDs(
        platform_count, platforms.data(), nullptr), "clGetPlatformIDs");
    for (cl_platform_id platform : platforms) {
      cl_uint count = 0;
      if (clGetDeviceIDs(
              platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &count) != CL_SUCCESS) {
        continue;
      }
      std::vector<cl_device_id> devices(count);
      CheckCl(clGetDeviceIDs(
          platform, CL_DEVICE_TYPE_GPU, count, devices.data(), nullptr),
          "clGetDeviceIDs");
      for (cl_device_id candidate : devices) {
        char name[256] = {};
        CheckCl(clGetDeviceInfo(
            candidate, CL_DEVICE_NAME, sizeof(name), name, nullptr),
            "clGetDeviceInfo name");
        if (std::string(name).find("B390") != std::string::npos) {
          device = candidate;
          break;
        }
      }
      if (device != nullptr) break;
    }
    if (device == nullptr) throw std::runtime_error("B390 OpenCL device not found");
    cl_int status = CL_SUCCESS;
    context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &status);
    CheckCl(status, "clCreateContext");
  }

  ~OpenClCompiler() {
    if (context != nullptr) clReleaseContext(context);
  }

  std::vector<std::uint8_t> Compile(
      const std::string& source, const std::string& kernel_name,
      int register_file_size, bool exact_attention_host = false,
      cl_uint* register_count = nullptr,
      cl_ulong* spill_memory_bytes = nullptr,
      cl_ulong* local_memory_bytes = nullptr,
      std::size_t* maximum_workgroup_size = nullptr,
      std::size_t* preferred_workgroup_multiple = nullptr) const {
    const char* source_data = source.c_str();
    const std::size_t source_size = source.size();
    cl_int status = CL_SUCCESS;
    cl_program program = clCreateProgramWithSource(
        context, 1, &source_data, &source_size, &status);
    CheckCl(status, "clCreateProgramWithSource");
    std::string options = exact_attention_host
        ? "-cl-std=CL3.0 -cl-fp32-correctly-rounded-divide-sqrt "
        : "-cl-std=CL2.0 ";
    if (register_file_size == 256) {
      options += "-cl-intel-256-GRF-per-thread ";
    } else if (register_file_size != 128) {
      options += "-ze-exp-register-file-size " +
          std::to_string(register_file_size) + " ";
    }
    options +=
        "-Dcl_intel_dot_accumulate "
        "-Dcl_intel_global_float_atomic "
        "-Dcl_intel_subgroup_matrix_multiply_accumulate "
        "-Dcl_intel_subgroup_split_matrix_multiply_accumulate";
    status = clBuildProgram(
        program, 1, &device, options.c_str(), nullptr, nullptr);
    if (status != CL_SUCCESS) {
      const std::string log = ProgramLog(program, device);
      clReleaseProgram(program);
      throw std::runtime_error("host shim build failed: " + log);
    }

    std::size_t binary_size = 0;
    CheckCl(clGetProgramInfo(
        program, CL_PROGRAM_BINARY_SIZES, sizeof(binary_size),
        &binary_size, nullptr), "clGetProgramInfo binary size");
    std::vector<std::uint8_t> binary(binary_size);
    unsigned char* binary_data = binary.data();
    CheckCl(clGetProgramInfo(
        program, CL_PROGRAM_BINARIES, sizeof(binary_data),
        &binary_data, nullptr), "clGetProgramInfo binary");
    clReleaseProgram(program);

    micro::fuse(binary, source.c_str());
    const std::size_t fused_size = binary.size();
    const unsigned char* fused_data = binary.data();
    cl_int binary_status = CL_SUCCESS;
    program = clCreateProgramWithBinary(
        context, 1, &device, &fused_size, &fused_data,
        &binary_status, &status);
    CheckCl(status, "clCreateProgramWithBinary fused");
    CheckCl(binary_status, "fused binary status");
    status = clBuildProgram(program, 1, &device, "", nullptr, nullptr);
    if (status != CL_SUCCESS) {
      const std::string log = ProgramLog(program, device);
      clReleaseProgram(program);
      throw std::runtime_error("fused program build failed: " + log);
    }
    cl_kernel kernel = clCreateKernel(program, kernel_name.c_str(), &status);
    CheckCl(status, "clCreateKernel fused");
    if (register_count != nullptr) {
      CheckCl(clGetKernelWorkGroupInfo(
          kernel, device, CL_KERNEL_REGISTER_COUNT_INTEL,
          sizeof(*register_count), register_count, nullptr),
          "clGetKernelWorkGroupInfo register count");
    }
    if (spill_memory_bytes != nullptr) {
      CheckCl(clGetKernelWorkGroupInfo(
          kernel, device, CL_KERNEL_SPILL_MEM_SIZE_INTEL,
          sizeof(*spill_memory_bytes), spill_memory_bytes, nullptr),
          "clGetKernelWorkGroupInfo spill memory");
    }
    if (local_memory_bytes != nullptr) {
      CheckCl(clGetKernelWorkGroupInfo(
          kernel, device, CL_KERNEL_LOCAL_MEM_SIZE,
          sizeof(*local_memory_bytes), local_memory_bytes, nullptr),
          "clGetKernelWorkGroupInfo local memory");
    }
    if (maximum_workgroup_size != nullptr) {
      CheckCl(clGetKernelWorkGroupInfo(
          kernel, device, CL_KERNEL_WORK_GROUP_SIZE,
          sizeof(*maximum_workgroup_size), maximum_workgroup_size, nullptr),
          "clGetKernelWorkGroupInfo maximum workgroup size");
    }
    if (preferred_workgroup_multiple != nullptr) {
      CheckCl(clGetKernelWorkGroupInfo(
          kernel, device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
          sizeof(*preferred_workgroup_multiple),
          preferred_workgroup_multiple, nullptr),
          "clGetKernelWorkGroupInfo preferred workgroup multiple");
    }
    clReleaseKernel(kernel);

    binary_size = 0;
    CheckCl(clGetProgramInfo(
        program, CL_PROGRAM_BINARY_SIZES, sizeof(binary_size),
        &binary_size, nullptr), "clGetProgramInfo fused binary size");
    binary.assign(binary_size, 0);
    binary_data = binary.data();
    CheckCl(clGetProgramInfo(
        program, CL_PROGRAM_BINARIES, sizeof(binary_data),
        &binary_data, nullptr), "clGetProgramInfo fused binary");
    clReleaseProgram(program);
    return binary;
  }
};

micro::Package Select(const Shape& shape) {
  micro::HWInformation hardware;
  hardware.euCount = 96;
  hardware.gmdid = 0x07800004;
  hardware.systolicAvailable = true;
  SetEfficient64Bit(hardware);

  gs::GEMMProblem problem;
  micro::GEMMOptions options;
  if (g_attention_nhalf) {
    problem.Ta = problem.Ta_ext = gs::Type::f16;
    problem.Tb = problem.Tb_ext = gs::Type::f16;
    problem.Tc = problem.Tc_ext = problem.Ts = gs::Type::f32;
    problem.A.layout = gs::MatrixLayout::N;
    problem.B.layout = gs::MatrixLayout::Pr;
    problem.A.setAlignment(128);
    problem.B.setAlignment(64);
    problem.C.setAlignment(static_cast<int32_t>(problem.Tc.size()));
    options.localB = true;
    options.slmPtr = true;

    gs::SizeParams sizes;
    using Requirement = gs::StrategyRequirement;
    if (shape.name == "attention_kq_nhalf") {
      // Mirror the locked SDPA K^T*Q carrier: sequence x GQA-query-tile x
      // head-size, with Q packed in VNNI form in SLM. N=8 is the recorded
      // occupancy target. M=16 preserves the exact route's 256-key reduction;
      // M=8 is the official XeHPG thin-Q alternate with a 128-key block and
      // therefore needs a separate arithmetic boundary before integration.
      problem.B.crosspack = 2;
      problem.B.tileR = 256;
      problem.B.tileC = 16;
      problem.C.layout = gs::MatrixLayout::T;
      sizes.m = g_attention_context;
      sizes.n = 16;
      sizes.k = 256;
      sizes.batch = 16;
      const std::vector<Requirement> requirements = {
          Requirement::UnrollM == g_attention_kq_unroll_m,
          Requirement::UnrollN == 8,
          Requirement::WGM == 16,
          Requirement::WGN == 1,
      };
      return micro::selectGEMM(
          options, hardware, sizes, problem, requirements);
    }
    if (shape.name == "attention_vs_nhalf") {
      // Mirror the paired V*S carrier for the same N=8 host configuration.
      problem.B.crosspack = 16;
      problem.C.layout = gs::MatrixLayout::N;
      sizes.m = 256;
      sizes.n = 8;
      sizes.k = g_attention_kq_unroll_m * 16;
      sizes.batch = 16;
      const std::vector<Requirement> requirements = {
          Requirement::UnrollM == 16,
          Requirement::UnrollN == 8,
          Requirement::WGM == 16,
          Requirement::WGN == 1,
      };
      const auto enable_dpasw = [](gs::GEMMStrategy& strategy) {
        strategy.dpasw |= strategy.fused;
      };
      return micro::selectGEMM(
          options, hardware, sizes, problem, requirements, enable_dpasw);
    }
    throw std::runtime_error("unknown attention package shape");
  }

  options.slmPtr = true;
  options.kParallelLocal = g_decode_fc;
  options.scaleA = true;
  options.offsetA = true;

  problem.Ta = gs::Type::f16;
  problem.Ta_ext = gs::Type::u4;
  problem.A.setAlignment(
      micro::alignmentForLD(static_cast<int>(shape.k * problem.Ta_ext)));
  problem.Ta_scale = gs::Type::f16;
  problem.A_scale.setAlignment(2);
  problem.A_scale.layout = g_row_major_metadata
      ? gs::MatrixLayout::T : gs::MatrixLayout::N;
  problem.asPtrDims = 2;
  problem.aqGroupM = 1;
  problem.aqGroupK = g_quant_group_size;
  problem.Tao = g_u8_zero_point ? gs::Type::u8 : gs::Type::u4;
  problem.AO.setAlignment(1);
  problem.AO.layout = g_row_major_metadata
      ? gs::MatrixLayout::T : gs::MatrixLayout::N;
  problem.aoPtrDims = 2;
  problem.aOffset = gs::ABOffset::Calc;

  problem.Tb = problem.Tb_ext = gs::Type::f16;
  problem.Tc = problem.Tc_ext = problem.Ts = gs::Type::f32;
  problem.A.layout = gs::MatrixLayout::T;
  problem.B.layout = gs::MatrixLayout::N;
  problem.C.layout = gs::MatrixLayout::N;
  problem.B.setAlignment(
      micro::alignmentForLD(static_cast<int>(shape.k * problem.Tb)));
  problem.C.setAlignment(static_cast<int32_t>(problem.Tc.size()));

  gs::SizeParams sizes;
  sizes.n = static_cast<uint16_t>(shape.n);
  sizes.m = static_cast<uint16_t>(shape.m);
  sizes.k = static_cast<uint16_t>(shape.k);
  sizes.batch = 1;
  if (g_decode_fc) {
    using Requirement = gs::StrategyRequirement;
    const std::vector<Requirement> requirements = {
        Requirement::UnrollM == 32,
        Requirement::UnrollN == 8,
        // 8192 / (32 * 2) = 128 workgroups, enough to occupy all 96 EUs for
        // the single-token boundary; WGM=8 left only 32 workgroups live.
        Requirement::WGM == 2,
        Requirement::WGN == 1,
        // Match the stock oneDNN GEMV carrier observed at the real QKV
        // boundary: eight K subgroups cooperate through a local reduction.
        Requirement::WGK == 8,
    };
    const auto keep_fixed_k_split = [](gs::GEMMStrategy& strategy) {
      strategy.kParallelLocal = true;
      strategy.shrinkWGK = false;
      strategy.forceFixedWGK = true;
      strategy.wg[gs::LoopK] = 8;
    };
    return micro::selectGEMM(
        options, hardware, sizes, problem, requirements, keep_fixed_k_split);
  }
  return micro::selectGEMM(options, hardware, sizes, problem);
}

int Setting(const micro::Package& package, const char* name) {
  return package.getSetting(name);
}

void WritePackage(const Shape& shape, bool first) {
  const micro::Package package = Select(shape);
  if (package.grfMin > g_register_file_size) {
    throw std::runtime_error(
        "selected package needs " + std::to_string(package.grfMin) +
        " GRFs, above requested register file size " +
        std::to_string(g_register_file_size));
  }
  micro::ShimOptions shim_options;
  shim_options.subgroupSize = 16;
  shim_options.useTileOps = true;
  if (g_attention_nhalf) {
    const bool is_vs = shape.name == "attention_vs_nhalf";
    shim_options.decorator = is_vs ? "vs" : "kq";
    shim_options.microkernelID = is_vs ? 1 : 0;
  } else {
    shim_options.decorator = "moe";
  }
  const std::string shim = micro::generateShim(
      package, micro::HostLanguage::OpenCL_C, shim_options);
  std::vector<std::uint8_t> program_binary;

  if (!g_dump_dir.empty()) {
    std::filesystem::create_directories(g_dump_dir);
    const std::filesystem::path stem = g_dump_dir / shape.name;
    std::ofstream shim_file(stem.string() + ".shim.cl", std::ios::binary);
    shim_file.write(shim.data(), static_cast<std::streamsize>(shim.size()));
    if (!shim_file) throw std::runtime_error("could not write shim source");
    std::ofstream binary_file(
        stem.string() + ".micro.bin", std::ios::binary);
    binary_file.write(
        reinterpret_cast<const char*>(package.binary.data()),
        static_cast<std::streamsize>(package.binary.size()));
    if (!binary_file) throw std::runtime_error("could not write micro binary");
    if (!g_host_source.empty()) {
      std::string source = ReadText(g_host_source);
      const std::string marker = "/* IQ36_MICRO_SHIM */";
      const std::size_t marker_offset = source.find(marker);
      if (marker_offset == std::string::npos) {
        throw std::runtime_error("host source is missing the shim marker");
      }
      source.replace(marker_offset, marker.size(), shim);
      const std::string kernel_name =
          std::string("iq36_moe_micro_") + shape.name;
      std::string prefix = "#define IQ36_KERNEL_NAME " + kernel_name + "\n" +
          "#define IQ36_K_PARALLEL_LOCAL " +
          std::string(g_decode_fc ? "1\n" : "0\n");
      if (g_row_major_metadata) {
        prefix += "#define IQ36_ROW_MAJOR_METADATA 1\n";
      }
      source = prefix + source;
      if (!g_projection_widths.empty()) {
        std::array<int, 4> widths = {0, 0, 0, 0};
        for (std::size_t index = 0; index < g_projection_widths.size(); ++index) {
          widths[index] = g_projection_widths[index];
        }
        for (int index = 3; index >= 0; --index) {
          source = "#define IQ36_M" + std::to_string(index) + " " +
              std::to_string(widths[static_cast<std::size_t>(index)]) +
              "\n" + source;
        }
      }
      std::ofstream source_file(
          stem.string() + ".fused.cl", std::ios::binary);
      source_file.write(
          source.data(), static_cast<std::streamsize>(source.size()));
      if (!source_file) {
        throw std::runtime_error("could not write fused host source");
      }
      const OpenClCompiler compiler;
      program_binary = compiler.Compile(
          source, kernel_name, g_register_file_size);
      std::ofstream program_file(
          stem.string() + ".program.bin", std::ios::binary);
      program_file.write(
          reinterpret_cast<const char*>(program_binary.data()),
          static_cast<std::streamsize>(program_binary.size()));
      if (!program_file) {
        throw std::runtime_error("could not write fused program binary");
      }
    }
  }

  if (!first) std::cout << ',';
  std::cout << "{\"kind\":\"" << shape.name << "\"";
  std::cout << ",\"m\":" << shape.m << ",\"n\":" << shape.n
            << ",\"k\":" << shape.k;
  std::cout << ",\"luid\":" << package.luid;
  std::cout << ",\"gmdid_compat\":" << package.gmdidCompat;
  if (!g_attention_nhalf) {
    std::cout << ",\"quant_group_size\":" << g_quant_group_size;
    std::cout << ",\"metadata_layout\":\""
              << (g_row_major_metadata ? "row_major_m_group"
                                       : "group_major_group_m") << "\"";
    std::cout << ",\"zero_point_type\":\""
              << (g_u8_zero_point ? "u8" : "u4") << "\"";
  }
  std::cout << ",\"grf_min\":" << package.grfMin;
  std::cout << ",\"register_file_size\":" << g_register_file_size;
  std::cout << ",\"barrier_count\":" << package.barrierCount;
  std::cout << ",\"systolic\":" << (package.systolic ? "true" : "false");
  std::cout << ",\"binary_bytes\":" << package.binary.size();
  std::cout << ",\"shim_bytes\":" << shim.size();
  std::cout << ",\"program_bytes\":" << program_binary.size();
  if (!g_projection_widths.empty()) {
    std::cout << ",\"projection_widths\":[";
    for (std::size_t index = 0; index < g_projection_widths.size(); ++index) {
      if (index != 0) std::cout << ',';
      std::cout << g_projection_widths[index];
    }
    std::cout << ']';
  }
  std::cout << ",\"settings\":{";
  const std::vector<const char*> names = {
      "sg_per_wg_m", "sg_per_wg_n", "sg_per_wg_k", "wg_tile_m",
      "wg_tile_n", "slm_size"};
  for (std::size_t index = 0; index < names.size(); ++index) {
    if (index != 0) std::cout << ',';
    std::cout << '\"' << names[index] << "\":"
              << Setting(package, names[index]);
  }
  std::cout << "}}";
}

void FuseExistingShim() {
  if (g_dump_dir.empty() || g_host_source.empty() ||
      g_existing_shim.empty() || g_existing_kernel_name.empty()) {
    throw std::runtime_error(
        "existing-shim fusion requires --dump-dir, --host-source, "
        "--fuse-existing-shim, and --kernel-name");
  }
  std::filesystem::create_directories(g_dump_dir);
  std::string source =
      HostDefineDirective(g_host_define) + ReadText(g_host_source);
  const std::string shim = ReadText(g_existing_shim);
  const std::string marker = "/* IQ36_MICRO_SHIM */";
  const std::size_t marker_offset = source.find(marker);
  if (marker_offset == std::string::npos) {
    throw std::runtime_error("host source is missing the shim marker");
  }
  source.replace(marker_offset, marker.size(), shim);
  const std::filesystem::path stem = g_dump_dir / "existing_shim";
  std::ofstream source_file(
      stem.string() + ".fused.cl", std::ios::binary);
  source_file.write(
      source.data(), static_cast<std::streamsize>(source.size()));
  if (!source_file) throw std::runtime_error("could not write fused source");
  const OpenClCompiler compiler;
  cl_uint register_count = 0;
  cl_ulong spill_memory_bytes = 0;
  cl_ulong local_memory_bytes = 0;
  std::size_t maximum_workgroup_size = 0;
  std::size_t preferred_workgroup_multiple = 0;
  const std::vector<std::uint8_t> program_binary = compiler.Compile(
      source, g_existing_kernel_name, g_register_file_size, true,
      &register_count, &spill_memory_bytes, &local_memory_bytes,
      &maximum_workgroup_size, &preferred_workgroup_multiple);
  std::ofstream program_file(
      stem.string() + ".program.bin", std::ios::binary);
  program_file.write(
      reinterpret_cast<const char*>(program_binary.data()),
      static_cast<std::streamsize>(program_binary.size()));
  if (!program_file) throw std::runtime_error("could not write program binary");
  std::cout << "{\"schema_version\":"
               "\"intel-qwen36-openvino-existing-micro-shim-fuse-v0\""
            << ",\"mode\":\"fuse_existing_microkernel_shim\""
            << ",\"openvino_onednn_commit\":\"" << g_provider_commit << "\""
            << ",\"kernel_name\":\"" << g_existing_kernel_name << "\""
            << ",\"host_define\":\"" << g_host_define << "\""
            << ",\"register_file_size\":" << g_register_file_size
            << ",\"exact_attention_vrt160\":"
            << (g_exact_attention_vrt160 ? "true" : "false")
            << ",\"exact_attention_dual_cohort\":"
            << (g_exact_attention_dual_cohort ? "true" : "false")
            << ",\"kernel_register_count\":" << register_count
            << ",\"kernel_spill_memory_bytes\":" << spill_memory_bytes
            << ",\"kernel_local_memory_bytes\":" << local_memory_bytes
            << ",\"kernel_maximum_workgroup_size\":"
            << maximum_workgroup_size
            << ",\"kernel_preferred_workgroup_multiple\":"
            << preferred_workgroup_multiple
            << ",\"shim_bytes\":" << shim.size()
            << ",\"host_source_bytes\":"
            << ReadText(g_host_source).size()
            << ",\"fused_source_bytes\":" << source.size()
            << ",\"program_bytes\":" << program_binary.size() << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    for (int index = 1; index < argc;) {
      const std::string option = argv[index];
      if (option == "--decode-fc" || option == "--decode-qkv") {
        g_decode_fc = true;
        g_quant_group_size = 64;
        ++index;
        continue;
      }
      if (option == "--prefill-fc") {
        g_prefill_fc = true;
        g_quant_group_size = 64;
        ++index;
        continue;
      }
      if (option == "--attention-nhalf") {
        g_attention_nhalf = true;
        ++index;
        continue;
      }
      if (option == "--exact-attention-vrt160") {
        g_exact_attention_vrt160 = true;
        ++index;
        continue;
      }
      if (option == "--exact-attention-dual-cohort") {
        g_exact_attention_dual_cohort = true;
        ++index;
        continue;
      }
      if (option == "--row-major-metadata") {
        g_row_major_metadata = true;
        ++index;
        continue;
      }
      if (option == "--u8-zero-point") {
        g_u8_zero_point = true;
        ++index;
        continue;
      }
      if (index + 1 >= argc) {
        throw std::runtime_error("codegen option is missing its value");
      }
      if (option == "--dump-dir") {
        g_dump_dir = argv[index + 1];
      } else if (option == "--host-source") {
        g_host_source = argv[index + 1];
      } else if (option == "--fuse-existing-shim") {
        g_existing_shim = argv[index + 1];
      } else if (option == "--kernel-name") {
        g_existing_kernel_name = argv[index + 1];
      } else if (option == "--host-define") {
        g_host_define = argv[index + 1];
      } else if (option == "--shape-name") {
        g_decode_name = argv[index + 1];
      } else if (option == "--m") {
        g_decode_m = std::stoi(argv[index + 1]);
      } else if (option == "--n") {
        g_prefill_n = std::stoi(argv[index + 1]);
      } else if (option == "--k") {
        g_decode_k = std::stoi(argv[index + 1]);
      } else if (option == "--context") {
        g_attention_context = std::stoi(argv[index + 1]);
      } else if (option == "--attention-kq-unroll-m") {
        g_attention_kq_unroll_m = std::stoi(argv[index + 1]);
      } else if (option == "--provider-commit") {
        g_provider_commit = argv[index + 1];
      } else if (option == "--register-file-size") {
        g_register_file_size = std::stoi(argv[index + 1]);
      } else if (option == "--projection-widths") {
        g_projection_widths = ParseWidths(argv[index + 1]);
      } else {
        throw std::runtime_error("unknown codegen option: " + option);
      }
      index += 2;
    }
    if (g_exact_attention_vrt160 && g_exact_attention_dual_cohort) {
      throw std::runtime_error(
          "exact-attention VRT160 and dual-cohort modes are mutually exclusive");
    }
    if (!g_host_source.empty() && g_dump_dir.empty()) {
      throw std::runtime_error("--host-source requires --dump-dir");
    }
    if (!g_existing_shim.empty()) {
      const bool native_control =
          !g_exact_attention_vrt160 && !g_exact_attention_dual_cohort &&
          g_register_file_size == 128;
      const bool fixed_vrt160 =
          g_exact_attention_vrt160 && g_register_file_size == 160 &&
          g_existing_kernel_name == "iq36_exact_score_fused" &&
          g_host_define == "IQ36_COMPONENT_PROGRAM=2";
      const bool fixed_dual_cohort =
          g_exact_attention_dual_cohort && g_register_file_size == 128 &&
          g_existing_kernel_name == "iq36_exact_score_dual_cohort" &&
          g_host_define == "IQ36_COMPONENT_PROGRAM=4";
      if (g_decode_fc || g_prefill_fc || g_attention_nhalf ||
          g_row_major_metadata || g_u8_zero_point ||
          (!native_control && !fixed_vrt160 && !fixed_dual_cohort)) {
        throw std::runtime_error(
            "existing-shim fusion is isolated to the native 128-GRF "
            "control or the registered exact-attention fused 160-GRF "
            "and dual-cohort candidates");
      }
      FuseExistingShim();
      return 0;
    }
    if (!g_existing_kernel_name.empty()) {
      throw std::runtime_error("--kernel-name requires --fuse-existing-shim");
    }
    if (!g_host_define.empty()) {
      throw std::runtime_error("--host-define requires --fuse-existing-shim");
    }
    if (g_exact_attention_vrt160) {
      throw std::runtime_error(
          "--exact-attention-vrt160 requires --fuse-existing-shim");
    }
    if (g_exact_attention_dual_cohort) {
      throw std::runtime_error(
          "--exact-attention-dual-cohort requires --fuse-existing-shim");
    }
    if (static_cast<int>(g_decode_fc) + static_cast<int>(g_prefill_fc) +
            static_cast<int>(g_attention_nhalf) >
        1) {
      throw std::runtime_error(
          "--decode-fc, --prefill-fc, and --attention-nhalf are mutually exclusive");
    }
    if (g_row_major_metadata && !(g_decode_fc || g_prefill_fc)) {
      throw std::runtime_error(
          "--row-major-metadata requires --decode-fc or --prefill-fc");
    }
    if (g_u8_zero_point && !(g_decode_fc || g_prefill_fc)) {
      throw std::runtime_error(
          "--u8-zero-point requires --decode-fc or --prefill-fc");
    }
    if (g_attention_nhalf && g_attention_context <= 0) {
      throw std::runtime_error("--context must be positive");
    }
    if (g_attention_nhalf && g_attention_kq_unroll_m != 8 &&
        g_attention_kq_unroll_m != 16) {
      throw std::runtime_error(
          "--attention-kq-unroll-m must be the official 8 alternate or 16 exact control");
    }
    if ((g_decode_fc || g_prefill_fc) &&
        (g_decode_name.empty() || g_decode_m <= 0 || g_decode_k <= 0 ||
         g_prefill_n <= 0 || g_decode_k % g_quant_group_size != 0)) {
      throw std::runtime_error("FC shape arguments are invalid");
    }
    if (!g_projection_widths.empty() &&
        (!(g_decode_fc || g_prefill_fc) ||
         g_projection_widths.size() > 4 ||
         std::any_of(g_projection_widths.begin(), g_projection_widths.end(),
                     [](int width) { return width <= 0; }) ||
         std::accumulate(g_projection_widths.begin(),
                         g_projection_widths.end(), 0) != g_decode_m)) {
      throw std::runtime_error(
          "projection widths must contain 1..4 positive values summing to m");
    }
    if (g_register_file_size != 160 && g_register_file_size != 256) {
      throw std::runtime_error(
          "--register-file-size must be the fixed 160 candidate or 256 control");
    }
    const std::vector<Shape> shapes = g_attention_nhalf
        ? std::vector<Shape>{{"attention_kq_nhalf", g_attention_context, 16, 256},
                             {"attention_vs_nhalf", 256, 8,
                              g_attention_kq_unroll_m * 16}}
        : ((g_decode_fc || g_prefill_fc)
                  ? std::vector<Shape>{{g_decode_name, g_decode_m,
                                        g_decode_fc ? 1 : g_prefill_n,
                                        g_decode_k}}
                  : std::vector<Shape>{{"gate", 512, 32, 2048},
                                       {"up", 512, 32, 2048},
                                       {"down", 2048, 32, 512}});
    std::cout << "{\"schema_version\":\""
              << (g_attention_nhalf
                      ? "intel-qwen36-openvino-attention-micro-capability-v0"
                      : "intel-qwen36-openvino-moe-micro-codegen-v0")
              << "\"";
    std::cout << ",\"openvino_onednn_commit\":\"" << g_provider_commit << "\"";
    const char* mode = !g_projection_widths.empty()
        ? (g_decode_fc ? "decode_fc_multi_output"
                       : "prefill_fc_multi_output")
        : (g_attention_nhalf ? "attention_nhalf_capability"
           : (g_decode_fc ? "decode_fc"
              : (g_prefill_fc ? "prefill_fc" : "moe")));
    std::cout << ",\"mode\":\"" << mode << "\"";
    std::cout << ",\"register_file_size\":" << g_register_file_size;
    if (g_attention_nhalf) {
      std::cout << ",\"attention_kq_unroll_m\":"
                << g_attention_kq_unroll_m;
      std::cout << ",\"attention_key_block\":"
                << g_attention_kq_unroll_m * 16;
    }
    std::cout << ",\"eu_count\":96,\"gmdid\":125829124"
                 ",\"efficient_64bit\":false,\"packages\":[";
    for (std::size_t index = 0; index < shapes.size(); ++index) {
      WritePackage(shapes[index], index == 0);
    }
    std::cout << "]}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
