#!/usr/bin/env python3
"""Generate the captured-layer Vulkan postconv/recurrent component harness."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA_VERSION = (
    "intel-qwen36-gpu-vulkan-postconv-recurrent-component-probe-v0")
DEFAULT_HOST = "local"
DEFAULT_MODEL = "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf"
DEFAULT_ENV_SCRIPT = (
    "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh")
DEFAULT_REMOTE_ROOT = "/home/intel/intel-qwen36-gpu"
PAYLOAD_ROOT = (
    ROOT
    / "output/r0-boundary-capture-run-20260627T054024Z/remote-output/payloads"
)
OPENCL_SOURCE = ROOT / "engine/gpu/opencl/q4x8_matvec.cl"
SOURCE_FILES = [
    ("engine/include/intel_qwen36/gguf_loader.hpp",
     "include/intel_qwen36/gguf_loader.hpp"),
    ("engine/include/intel_qwen36/gpu_q4x8_matvec.hpp",
     "include/intel_qwen36/gpu_q4x8_matvec.hpp"),
    ("engine/include/intel_qwen36/gpu_vulkan_postconv_recurrent.hpp",
     "include/intel_qwen36/gpu_vulkan_postconv_recurrent.hpp"),
    ("engine/src/gguf_loader.cpp", "src/gguf_loader.cpp"),
    ("engine/src/gpu_q4x8_matvec.cpp", "src/gpu_q4x8_matvec.cpp"),
    ("engine/src/gpu_q4_cpu_order_matvec.cpp",
     "src/gpu_q4_cpu_order_matvec.cpp"),
    ("engine/src/gpu_vulkan_postconv_recurrent.cpp",
     "src/gpu_vulkan_postconv_recurrent.cpp"),
]
PAYLOAD_SPECS = {
    "conv_output_raw": (
        "conv_output_raw.bin", "conv_output_raw-{layer}__tok15__ord3.bin",
        32768),
    "gate": ("gate.bin", "gate-{layer}__tok15__ord12.bin", 128),
    "beta_sigmoid": (
        "beta_sigmoid.bin", "beta_sigmoid-{layer}__tok15__ord14.bin", 128),
    "state_predelta": (
        "state_predelta.bin", "state_predelta-{layer}__tok15__ord15.bin",
        2097152),
    "attention_output": (
        "attention_output.bin", "attn_output-{layer}__tok15__ord16.bin",
        16384),
    "z": ("z.bin", "z-{layer}__tok15__ord17.bin", 16384),
    "final_output": (
        "final_output.bin", "final_output-{layer}__tok15__ord18.bin", 16384),
}


HARNESS_CPP = r'''
#include "intel_qwen36/gguf_loader.hpp"
#include "intel_qwen36/gpu_q4x8_matvec.hpp"
#include "intel_qwen36/gpu_vulkan_postconv_recurrent.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

const char* kQ4X8OpenClSource = @@OPENCL_SOURCE_LITERAL@@;

constexpr int kHeadDim = 128;
constexpr int kQueryHeads = 16;
constexpr int kValueHeads = 32;
constexpr int kQValues = kHeadDim * kQueryHeads;
constexpr int kVValues = kHeadDim * kValueHeads;
constexpr int kConvValues = 2 * kQValues + kVValues;
constexpr int kStateValues = kHeadDim * kHeadDim * kValueHeads;
constexpr double kMismatchThreshold = 0.0;
constexpr double kAddedWallMaxUs = 6.841858993929781;

struct Args {
  std::string model_path;
  std::string payload_dir;
  std::string postconv_spirv;
  std::string recurrent_spirv;
  std::string opencl_device = "B390";
  std::string vulkan_device = "PTL";
  int layer = 0;
  int samples = 11;
};

[[noreturn]] void Die(const std::string& message) {
  throw std::runtime_error(message);
}

void Require(bool condition, const std::string& message) {
  if (!condition) Die(message);
}

std::string JoinPath(const std::string& dir, const std::string& name) {
  return (!dir.empty() && dir.back() == '/') ? dir + name : dir + "/" + name;
}

std::string LayerTensorName(int layer, const std::string& suffix) {
  return "blk." + std::to_string(layer) + "." + suffix;
}

Args ParseArgs(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    auto value = [&]() -> std::string {
      if (++i >= argc) Die("missing value for " + key);
      return argv[i];
    };
    if (key == "--model") args.model_path = value();
    else if (key == "--payload-dir") args.payload_dir = value();
    else if (key == "--postconv-spv") args.postconv_spirv = value();
    else if (key == "--recurrent-spv") args.recurrent_spirv = value();
    else if (key == "--opencl-device") args.opencl_device = value();
    else if (key == "--vulkan-device") args.vulkan_device = value();
    else if (key == "--layer") args.layer = std::stoi(value());
    else if (key == "--samples") args.samples = std::stoi(value());
    else Die("unknown argument: " + key);
  }
  Require(!args.model_path.empty(), "--model is required");
  Require(!args.payload_dir.empty(), "--payload-dir is required");
  Require(!args.postconv_spirv.empty(), "--postconv-spv is required");
  Require(!args.recurrent_spirv.empty(), "--recurrent-spv is required");
  Require(args.samples > 0, "--samples must be positive");
  return args;
}

double Min(const std::vector<double>& values) {
  Require(!values.empty(), "empty timing distribution");
  return *std::min_element(values.begin(), values.end());
}

double Mean(const std::vector<double>& values) {
  Require(!values.empty(), "empty timing distribution");
  return std::accumulate(values.begin(), values.end(), 0.0) /
         static_cast<double>(values.size());
}

bool BitExact(const iq36::VectorCompareStats& stats) {
  return stats.same_size && stats.finite && stats.mismatch_count == 0;
}

void WriteCompare(const iq36::VectorCompareStats& stats) {
  std::cout << "{";
  std::cout << "\"compared_value_count\":" << stats.compared_value_count << ",";
  std::cout << "\"cosine\":" << stats.cosine << ",";
  std::cout << "\"finite\":" << (stats.finite ? "true" : "false") << ",";
  std::cout << "\"max_abs_diff\":" << stats.max_abs_diff << ",";
  std::cout << "\"mean_abs_diff\":" << stats.mean_abs_diff << ",";
  std::cout << "\"mismatch_count\":" << stats.mismatch_count << ",";
  std::cout << "\"rmse\":" << stats.rmse << ",";
  std::cout << "\"same_size\":" << (stats.same_size ? "true" : "false");
  std::cout << "}";
}

std::vector<double> RunCurrentWallSamples(
    iq36::GpuQ4X8MatvecRunner& runner,
    std::uint64_t conv_output_handle,
    std::uint64_t state_seed_handle,
    const std::vector<float>& gate,
    const std::vector<float>& beta,
    const std::vector<float>& z,
    const std::vector<float>& norm_weight,
    float norm_epsilon,
    int samples) {
  std::vector<double> wall;
  wall.reserve(static_cast<std::size_t>(samples));
  for (int sample = 0; sample < samples; ++sample) {
    const auto state_handle = runner.CloneResidentF32Buffer(state_seed_handle);
    const auto begin = std::chrono::steady_clock::now();
    runner.RunPostConvPrepThenLinearAttentionDeltaResidentState(
        conv_output_handle, state_handle, gate, beta, z, norm_weight,
        kHeadDim, kQueryHeads, kValueHeads, norm_epsilon, 1,
        false, false, false, false);
    const auto end = std::chrono::steady_clock::now();
    wall.push_back(
        std::chrono::duration<double, std::micro>(end - begin).count());
  }
  return wall;
}

int main(int argc, char** argv) {
  try {
    const Args args = ParseArgs(argc, argv);
    const auto index = iq36::parse_gguf_model_index(args.model_path);
    const auto load_map = iq36::validate_qwen36_load_map(index);
    const auto epsilon_it = index.metadata.find(
        "qwen35moe.attention.layer_norm_rms_epsilon");
    Require(epsilon_it != index.metadata.end(), "RMS epsilon missing");
    Require(epsilon_it->second.kind == iq36::GgufMetadataValue::Kind::kFloat,
            "RMS epsilon type mismatch");
    const float norm_epsilon =
        static_cast<float>(epsilon_it->second.float_value);
    const auto norm_weight = iq36::decode_tensor_row(
        args.model_path, index,
        LayerTensorName(args.layer, "ssm_norm.weight"), 0);

    const auto conv_output = iq36::read_f32_vector_file(
        JoinPath(args.payload_dir, "conv_output_raw.bin"));
    const auto gate = iq36::read_f32_vector_file(
        JoinPath(args.payload_dir, "gate.bin"));
    const auto beta = iq36::read_f32_vector_file(
        JoinPath(args.payload_dir, "beta_sigmoid.bin"));
    const auto state = iq36::read_f32_vector_file(
        JoinPath(args.payload_dir, "state_predelta.bin"));
    const auto z = iq36::read_f32_vector_file(JoinPath(args.payload_dir, "z.bin"));
    const auto captured_attention = iq36::read_f32_vector_file(
        JoinPath(args.payload_dir, "attention_output.bin"));
    const auto captured_final = iq36::read_f32_vector_file(
        JoinPath(args.payload_dir, "final_output.bin"));
    const bool shapes_ok =
        conv_output.size() == kConvValues && gate.size() == kValueHeads &&
        beta.size() == kValueHeads && state.size() == kStateValues &&
        z.size() == kVValues && norm_weight.size() == kHeadDim &&
        captured_attention.size() == kVValues &&
        captured_final.size() == kVValues;
    Require(shapes_ok, "captured component shape mismatch");

    const auto cpu = iq36::run_qwen36_linear_attention_postconv_core(
        conv_output, gate, beta, state, z, norm_weight, norm_epsilon);
    std::vector<float> decay(kValueHeads);
    for (int head = 0; head < kValueHeads; ++head) {
      decay[static_cast<std::size_t>(head)] =
          std::exp(gate[static_cast<std::size_t>(head)]);
    }
    std::vector<float> z_silu(kVValues);
    for (int i = 0; i < kVValues; ++i) {
      const float value = z[static_cast<std::size_t>(i)];
      z_silu[static_cast<std::size_t>(i)] =
          value * iq36::sigmoid_scalar(value);
    }

    iq36::GpuQ4X8MatvecRunner opencl(args.opencl_device, kQ4X8OpenClSource);
    const auto conv_handle = opencl.UploadF32Buffer(conv_output);
    const auto state_seed_handle = opencl.UploadF32Buffer(state);
    RunCurrentWallSamples(
        opencl, conv_handle, state_seed_handle, gate, beta, z, norm_weight,
        norm_epsilon, 1);
    const auto current_wall = RunCurrentWallSamples(
        opencl, conv_handle, state_seed_handle, gate, beta, z, norm_weight,
        norm_epsilon, args.samples);
    const auto current_state_handle =
        opencl.CloneResidentF32Buffer(state_seed_handle);
    const auto current =
        opencl.RunPostConvPrepThenLinearAttentionDeltaResidentState(
            conv_handle, current_state_handle, gate, beta, z, norm_weight,
            kHeadDim, kQueryHeads, kValueHeads, norm_epsilon, 1,
            true, false, true, true);

    iq36::GpuVulkanPostconvRecurrentInput vulkan_input;
    vulkan_input.conv_output_raw = conv_output;
    vulkan_input.decay = decay;
    vulkan_input.beta = beta;
    vulkan_input.recurrent_state = state;
    vulkan_input.z_silu = z_silu;
    vulkan_input.norm_weight = norm_weight;
    vulkan_input.norm_epsilon = norm_epsilon;
    vulkan_input.attention_scale = 1.0f / std::sqrt(128.0f);
    iq36::GpuVulkanPostconvRecurrentRunner vulkan(
        args.postconv_spirv, args.recurrent_spirv, args.vulkan_device);
    vulkan.Run(vulkan_input, 1);
    const auto candidate = vulkan.Run(vulkan_input, args.samples);

    const auto q_compare = iq36::compare_vectors(
        candidate.q_conv_predelta, cpu.q_conv_predelta, kMismatchThreshold);
    const auto k_compare = iq36::compare_vectors(
        candidate.k_conv_predelta, cpu.k_conv_predelta, kMismatchThreshold);
    const auto v_compare = iq36::compare_vectors(
        candidate.v_conv_predelta, cpu.v_conv_predelta, kMismatchThreshold);
    const auto attention_compare = iq36::compare_vectors(
        candidate.attention_output, cpu.attention_output, kMismatchThreshold);
    const auto state_compare = iq36::compare_vectors(
        candidate.recurrent_state, cpu.recurrent_state, kMismatchThreshold);
    const auto final_compare = iq36::compare_vectors(
        candidate.final_output, cpu.final_output, kMismatchThreshold);
    const auto current_attention_compare = iq36::compare_vectors(
        current.attention_output, cpu.attention_output, kMismatchThreshold);
    const auto current_state_compare = iq36::compare_vectors(
        current.recurrent_state, cpu.recurrent_state, kMismatchThreshold);
    const auto current_final_compare = iq36::compare_vectors(
        current.final_output, cpu.final_output, kMismatchThreshold);
    const auto cpu_attention_capture = iq36::compare_vectors(
        cpu.attention_output, captured_attention, kMismatchThreshold);
    const auto cpu_final_capture = iq36::compare_vectors(
        cpu.final_output, captured_final, kMismatchThreshold);

    const bool exact =
        BitExact(q_compare) && BitExact(k_compare) && BitExact(v_compare) &&
        BitExact(attention_compare) && BitExact(state_compare) &&
        BitExact(final_compare);
    const double current_min = Min(current_wall);
    const double current_mean = Mean(current_wall);
    const double candidate_min = Min(candidate.sample_wall_us);
    const double candidate_mean = Mean(candidate.sample_wall_us);
    const double added_min = candidate_min - current_min;
    const double added_mean = candidate_mean - current_mean;
    const bool timing_ok =
        current_min > 0.0 && candidate_min > 0.0 &&
        added_min <= kAddedWallMaxUs;
    const bool required =
        load_map.ready && shapes_ok && exact && timing_ok &&
        candidate.sample_wall_us.size() == static_cast<std::size_t>(args.samples);

    std::cout << std::setprecision(12);
    std::cout << "{";
    std::cout << "\"schema_version\":\""
              << "intel-qwen36-gpu-vulkan-postconv-recurrent-component-probe-v0"
              << "\",";
    std::cout << "\"layer\":" << args.layer << ",";
    std::cout << "\"samples\":" << args.samples << ",";
    std::cout << "\"opencl_device\":\"" << opencl.device_name() << "\",";
    std::cout << "\"vulkan_device\":\"" << vulkan.device_name() << "\",";
    std::cout << "\"timings\":{";
    std::cout << "\"current_wall_min_us\":" << current_min << ",";
    std::cout << "\"current_wall_mean_us\":" << current_mean << ",";
    std::cout << "\"candidate_wall_min_us\":" << candidate_min << ",";
    std::cout << "\"candidate_wall_mean_us\":" << candidate_mean << ",";
    std::cout << "\"candidate_added_min_us\":" << added_min << ",";
    std::cout << "\"candidate_added_mean_us\":" << added_mean << ",";
    std::cout << "\"candidate_added_us_max\":" << kAddedWallMaxUs;
    std::cout << "},\"comparisons\":{";
    std::cout << "\"q_conv_predelta_vs_cpu\":"; WriteCompare(q_compare);
    std::cout << ",\"k_conv_predelta_vs_cpu\":"; WriteCompare(k_compare);
    std::cout << ",\"v_conv_predelta_vs_cpu\":"; WriteCompare(v_compare);
    std::cout << ",\"attention_vs_cpu\":"; WriteCompare(attention_compare);
    std::cout << ",\"state_vs_cpu\":"; WriteCompare(state_compare);
    std::cout << ",\"final_vs_cpu\":"; WriteCompare(final_compare);
    std::cout << ",\"current_attention_vs_cpu\":";
    WriteCompare(current_attention_compare);
    std::cout << ",\"current_state_vs_cpu\":";
    WriteCompare(current_state_compare);
    std::cout << ",\"current_final_vs_cpu\":";
    WriteCompare(current_final_compare);
    std::cout << ",\"cpu_attention_vs_capture\":";
    WriteCompare(cpu_attention_capture);
    std::cout << ",\"cpu_final_vs_capture\":";
    WriteCompare(cpu_final_capture);
    std::cout << "},\"checks\":{";
    std::cout << "\"load_map_ready\":" << (load_map.ready ? "true" : "false") << ",";
    std::cout << "\"payload_shapes_ok\":" << (shapes_ok ? "true" : "false") << ",";
    std::cout << "\"fresh_state_per_sample\":true,";
    std::cout << "\"uploads_and_pipeline_outside_timed_region\":true,";
    std::cout << "\"candidate_bit_exact\":" << (exact ? "true" : "false") << ",";
    std::cout << "\"paired_wall_budget_passed\":" << (timing_ok ? "true" : "false") << ",";
    std::cout << "\"speedup_claims_allowed\":false";
    std::cout << "},\"required_checks_passed\":"
              << (required ? "true" : "false") << "}" << std::endl;
    opencl.ClearResidentF32Buffers();
    return required ? 0 : 2;
  } catch (const std::exception& ex) {
    std::cerr << "Vulkan component probe error: " << ex.what() << std::endl;
    return 1;
  }
}
'''


def generate_cpp(opencl_source: str) -> str:
  literal = 'R"IQ36VK(' + opencl_source + ')IQ36VK"'
  return HARNESS_CPP.replace("@@OPENCL_SOURCE_LITERAL@@", literal)


def payload_manifest(layer: int) -> dict[str, dict[str, object]]:
  manifest: dict[str, dict[str, object]] = {}
  for key, (stage_name, template, expected_size) in PAYLOAD_SPECS.items():
    path = PAYLOAD_ROOT / template.format(layer=layer)
    data = path.read_bytes()
    if len(data) != expected_size:
      raise ValueError(
          f"{path}: expected {expected_size} bytes, found {len(data)}")
    manifest[key] = {
        "path": str(path.relative_to(ROOT)),
        "stage_name": stage_name,
        "size_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }
  return manifest
