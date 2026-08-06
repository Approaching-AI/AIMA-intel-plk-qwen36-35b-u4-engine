#!/usr/bin/env python3
"""Generate the captured-layer exact-preprojection component harness."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA_VERSION = (
    "intel-qwen36-all-linear-preprojection-parity-component-probe-v0"
)
DEFAULT_HOST = "local"
DEFAULT_MODEL = "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf"
DEFAULT_ENV_SCRIPT = "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh"
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
    ("engine/src/gguf_loader.cpp", "src/gguf_loader.cpp"),
    ("engine/src/gpu_q4x8_matvec.cpp", "src/gpu_q4x8_matvec.cpp"),
    ("engine/src/gpu_q4_cpu_order_matvec.cpp",
     "src/gpu_q4_cpu_order_matvec.cpp"),
]
PAYLOAD_SPECS = {
    "attention_norm": ("attention_norm.bin", "attn_norm-{layer}__tok15__ord1.bin", 8192),
    "qkv_mixed": ("qkv_mixed.bin", "linear_attn_qkv_mixed-{layer}__tok15__ord2.bin", 32768),
    "conv_output_raw": ("conv_output_raw.bin", "conv_output_raw-{layer}__tok15__ord3.bin", 32768),
    "gate": ("gate.bin", "gate-{layer}__tok15__ord12.bin", 128),
    "beta_sigmoid": ("beta_sigmoid.bin", "beta_sigmoid-{layer}__tok15__ord14.bin", 128),
    "state_predelta": ("state_predelta.bin", "state_predelta-{layer}__tok15__ord15.bin", 2097152),
    "attention_output": ("attention_output.bin", "attn_output-{layer}__tok15__ord16.bin", 16384),
    "z": ("z.bin", "z-{layer}__tok15__ord17.bin", 16384),
    "final_output": ("final_output.bin", "final_output-{layer}__tok15__ord18.bin", 16384),
    "linear_attn_out": ("linear_attn_out.bin", "linear_attn_out-{layer}__tok15__ord19.bin", 8192),
}


PROBE_CPP = r'''
#include "intel_qwen36/gguf_loader.hpp"
#include "intel_qwen36/gpu_q4x8_matvec.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

const char* kQ4X8OpenClSource = @@OPENCL_SOURCE_LITERAL@@;

constexpr int kLayerCount = 40;
constexpr int kHiddenSize = 2048;
constexpr int kQkvRows = 8192;
constexpr int kConvKernelSize = 4;
constexpr int kHeadDim = 128;
constexpr int kQueryHeads = 16;
constexpr int kValueHeads = 32;
constexpr int kFinalValues = kHeadDim * kValueHeads;
constexpr int kRecurrentStateValues = kHeadDim * kHeadDim * kValueHeads;
constexpr double kMismatchThreshold = 0.0;

struct Args {
  std::string model_path;
  std::string payload_dir;
  int layer = 0;
  int samples = 11;
  std::string device_substring = "B390";
};

struct ComponentSample {
  iq36::GpuQ4X8ConvHandoffRun current_preconv;
  iq36::GpuQ4X8ConvHandoffRun exact_preconv;
  iq36::GpuLinearAttentionDeltaRun current_postconv;
  iq36::GpuLinearAttentionDeltaRun exact_postconv;
  iq36::GpuQ4X8MatvecRun current_projection;
  iq36::GpuQ4X8MatvecRun exact_projection;
  double current_preconv_us = 0.0;
  double exact_preconv_us = 0.0;
  double current_postconv_us = 0.0;
  double exact_postconv_us = 0.0;
  double current_projection_us = 0.0;
  double exact_projection_us = 0.0;
  double current_shell_us = 0.0;
  double exact_shell_us = 0.0;
};

[[noreturn]] void Die(const std::string& message) {
  throw std::runtime_error(message);
}

void Require(bool ok, const std::string& message) {
  if (!ok) Die(message);
}

std::string JsonEscape(const std::string& value) {
  std::string out;
  out.reserve(value.size() + 8);
  for (const char ch : value) {
    switch (ch) {
      case '\\': out += "\\\\"; break;
      case '"': out += "\\\""; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default: out += ch; break;
    }
  }
  return out;
}

std::string JoinPath(const std::string& dir, const std::string& name) {
  return (!dir.empty() && dir.back() == '/') ? dir + name : dir + "/" + name;
}

std::string LayerTensorName(int layer, const std::string& suffix) {
  return "blk." + std::to_string(layer) + "." + suffix;
}

std::vector<std::uint8_t> ReadTensorBytes(
    std::ifstream& in, const iq36::GgufTensorInfo& tensor) {
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(tensor.nbytes));
  in.clear();
  in.seekg(static_cast<std::streamoff>(tensor.absolute_offset), std::ios::beg);
  Require(static_cast<bool>(in), "failed to seek tensor payload");
  in.read(reinterpret_cast<char*>(bytes.data()),
          static_cast<std::streamsize>(bytes.size()));
  Require(in.gcount() == static_cast<std::streamsize>(bytes.size()),
          "failed to read tensor payload");
  return bytes;
}

std::vector<float> ReadF32Tensor(
    std::ifstream& in, const iq36::GgufTensorInfo& tensor,
    std::size_t expected_values) {
  Require(tensor.type == 0, "expected F32 tensor");
  const auto bytes = ReadTensorBytes(in, tensor);
  Require(bytes.size() == expected_values * sizeof(float),
          "F32 tensor payload size mismatch");
  std::vector<float> values(expected_values);
  std::memcpy(values.data(), bytes.data(), bytes.size());
  return values;
}

Args ParseArgs(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    auto value = [&](const char* name) -> std::string {
      Require(i + 1 < argc, std::string("missing value for ") + name);
      return argv[++i];
    };
    if (key == "--model") args.model_path = value("--model");
    else if (key == "--payload-dir") args.payload_dir = value("--payload-dir");
    else if (key == "--layer") args.layer = std::stoi(value("--layer"));
    else if (key == "--samples") args.samples = std::stoi(value("--samples"));
    else if (key == "--device-substring") {
      args.device_substring = value("--device-substring");
    } else {
      Die("unknown argument: " + key);
    }
  }
  Require(!args.model_path.empty(), "--model is required");
  Require(!args.payload_dir.empty(), "--payload-dir is required");
  Require(args.layer >= 0 && args.layer < kLayerCount, "--layer is out of range");
  Require(args.samples > 0, "--samples must be positive");
  return args;
}

bool BitExact(const iq36::VectorCompareStats& stats) {
  return stats.same_size && stats.finite && stats.mismatch_count == 0 &&
         stats.max_abs_diff == 0.0 && stats.rmse == 0.0;
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

double Min(const std::vector<double>& values) {
  Require(!values.empty(), "timing distribution is empty");
  return *std::min_element(values.begin(), values.end());
}

double Mean(const std::vector<double>& values) {
  Require(!values.empty(), "timing distribution is empty");
  return std::accumulate(values.begin(), values.end(), 0.0) /
         static_cast<double>(values.size());
}

double PostConvKernelUs(const iq36::GpuLinearAttentionDeltaRun& run) {
  return run.timing.delta_min_us + run.timing.final_min_us;
}

ComponentSample RunSample(
    iq36::GpuQ4X8MatvecRunner& runner,
    std::uint64_t q6_handle,
    const iq36::GpuQ8KInputPlanes& q8,
    std::uint64_t conv_weights_handle,
    std::uint64_t conv_state_seed_handle,
    std::uint64_t conv_output_seed_handle,
    std::uint64_t recurrent_state_seed_handle,
    const std::vector<float>& gate,
    const std::vector<float>& beta,
    const std::vector<float>& z,
    const std::vector<float>& norm_weight,
    float norm_epsilon,
    const std::vector<std::uint8_t>& projection_packed,
    const iq36::GpuQ8KInputPlanes& projection_q8,
    std::uint64_t projection_rows,
    std::uint64_t projection_blocks_per_row) {
  const auto current_conv_state_handle =
      runner.CloneResidentF32Buffer(conv_state_seed_handle);
  const auto exact_conv_state_handle =
      runner.CloneResidentF32Buffer(conv_state_seed_handle);
  const auto current_recurrent_state_handle =
      runner.CloneResidentF32Buffer(recurrent_state_seed_handle);
  const auto exact_recurrent_state_handle =
      runner.CloneResidentF32Buffer(recurrent_state_seed_handle);

  ComponentSample sample;
  sample.current_preconv = runner.RunResidentRawQ6KThenResidentConvState(
      q6_handle, q8, conv_weights_handle, current_conv_state_handle,
      kConvKernelSize, 1, true, 0, true, true);
  sample.exact_preconv =
      runner.RunResidentRawQ6KThenResidentConvStateCpuOrder(
          q6_handle, q8, conv_weights_handle, exact_conv_state_handle,
          kConvKernelSize, 1, true, 0, true, true);

  sample.current_postconv =
      runner.RunPostConvPrepThenLinearAttentionDeltaResidentState(
          conv_output_seed_handle, current_recurrent_state_handle,
          gate, beta, z, norm_weight, kHeadDim, kQueryHeads, kValueHeads,
          norm_epsilon, 1, true, false, true, true);
  sample.exact_postconv =
      runner.RunPostConvPrepThenLinearAttentionDeltaResidentStateCpuOrder(
          conv_output_seed_handle, exact_recurrent_state_handle,
          gate, beta, z, norm_weight, kHeadDim, kQueryHeads, kValueHeads,
          norm_epsilon, 1, true, true, true);

  sample.current_projection = runner.RunRowblock16(
      projection_packed, projection_q8.qs, projection_q8.bsums,
      projection_q8.d, projection_rows, projection_blocks_per_row, 1);
  sample.exact_projection = runner.RunRowblock16CpuOrderFinalize(
      projection_packed, projection_q8.qs, projection_q8.bsums,
      projection_q8.d, projection_rows, projection_blocks_per_row, 1);

  sample.current_preconv_us = sample.current_preconv.timing.matvec.min_us +
                              sample.current_preconv.timing.conv_min_us;
  sample.exact_preconv_us = sample.exact_preconv.timing.matvec.min_us +
                            sample.exact_preconv.timing.conv_min_us;
  sample.current_postconv_us = PostConvKernelUs(sample.current_postconv);
  sample.exact_postconv_us = PostConvKernelUs(sample.exact_postconv);
  sample.current_projection_us = sample.current_projection.timing.min_us;
  sample.exact_projection_us = sample.exact_projection.timing.min_us;
  sample.current_shell_us = sample.current_preconv_us +
                            sample.current_postconv_us +
                            sample.current_projection_us;
  sample.exact_shell_us = sample.exact_preconv_us +
                          sample.exact_postconv_us +
                          sample.exact_projection_us;
  return sample;
}

int main(int argc, char** argv) {
  try {
    const Args args = ParseArgs(argc, argv);
    const auto index = iq36::parse_gguf_model_index(args.model_path);
    const auto load_map = iq36::validate_qwen36_load_map(index);
    const auto epsilon_it = index.metadata.find(
        "qwen35moe.attention.layer_norm_rms_epsilon");
    Require(epsilon_it != index.metadata.end(), "RMS norm epsilon missing");
    Require(epsilon_it->second.kind == iq36::GgufMetadataValue::Kind::kFloat,
            "RMS norm epsilon metadata type mismatch");
    const float norm_epsilon =
        static_cast<float>(epsilon_it->second.float_value);

    const std::string qkv_name = LayerTensorName(args.layer, "attn_qkv.weight");
    const std::string conv_name = LayerTensorName(args.layer, "ssm_conv1d.weight");
    const std::string norm_name = LayerTensorName(args.layer, "ssm_norm.weight");
    const std::string projection_name = LayerTensorName(args.layer, "ssm_out.weight");
    const auto* qkv_tensor = iq36::find_tensor(index, qkv_name);
    const auto* conv_tensor = iq36::find_tensor(index, conv_name);
    const auto* norm_tensor = iq36::find_tensor(index, norm_name);
    const auto* projection_tensor = iq36::find_tensor(index, projection_name);
    Require(qkv_tensor != nullptr, "Q6 QKV tensor missing");
    Require(conv_tensor != nullptr, "convolution tensor missing");
    Require(norm_tensor != nullptr, "SSM norm tensor missing");
    Require(projection_tensor != nullptr, "output projection tensor missing");
    const bool tensor_shapes_ok =
        qkv_tensor->type == 14 &&
        qkv_tensor->dims == std::vector<std::uint64_t>{kHiddenSize, kQkvRows} &&
        conv_tensor->type == 0 &&
        conv_tensor->dims == std::vector<std::uint64_t>{kConvKernelSize, kQkvRows} &&
        norm_tensor->type == 0 &&
        norm_tensor->dims == std::vector<std::uint64_t>{kHeadDim} &&
        projection_tensor->type == 12 &&
        projection_tensor->dims ==
            std::vector<std::uint64_t>{kFinalValues, kHiddenSize};
    Require(tensor_shapes_ok, "component tensor shapes mismatch");

    const auto attention_norm = iq36::read_f32_vector_file(
        JoinPath(args.payload_dir, "attention_norm.bin"));
    const auto captured_qkv = iq36::read_f32_vector_file(
        JoinPath(args.payload_dir, "qkv_mixed.bin"));
    const auto captured_conv_output = iq36::read_f32_vector_file(
        JoinPath(args.payload_dir, "conv_output_raw.bin"));
    const auto gate = iq36::read_f32_vector_file(
        JoinPath(args.payload_dir, "gate.bin"));
    const auto beta = iq36::read_f32_vector_file(
        JoinPath(args.payload_dir, "beta_sigmoid.bin"));
    const auto recurrent_state = iq36::read_f32_vector_file(
        JoinPath(args.payload_dir, "state_predelta.bin"));
    const auto captured_attention_output = iq36::read_f32_vector_file(
        JoinPath(args.payload_dir, "attention_output.bin"));
    const auto z = iq36::read_f32_vector_file(JoinPath(args.payload_dir, "z.bin"));
    const auto captured_final_output = iq36::read_f32_vector_file(
        JoinPath(args.payload_dir, "final_output.bin"));
    const auto captured_projection = iq36::read_f32_vector_file(
        JoinPath(args.payload_dir, "linear_attn_out.bin"));
    const bool payload_shapes_ok =
        attention_norm.size() == kHiddenSize && captured_qkv.size() == kQkvRows &&
        captured_conv_output.size() == kQkvRows && gate.size() == kValueHeads &&
        beta.size() == kValueHeads && recurrent_state.size() == kRecurrentStateValues &&
        captured_attention_output.size() == kFinalValues && z.size() == kFinalValues &&
        captured_final_output.size() == kFinalValues &&
        captured_projection.size() == kHiddenSize;
    Require(payload_shapes_ok, "captured payload shapes mismatch");

    const std::vector<float> conv_state_seed(
        static_cast<std::size_t>((kConvKernelSize - 1) * kQkvRows), 0.0f);
    const auto cpu_qkv = iq36::matvec_tensor(
        args.model_path, index, qkv_name, attention_norm);
    const auto cpu_conv = iq36::run_qwen36_linear_attention_conv_core(
        args.model_path, index, args.layer, cpu_qkv, conv_state_seed);
    const auto norm_weight = iq36::decode_tensor_row(
        args.model_path, index, norm_name, 0);
    const auto cpu_postconv = iq36::run_qwen36_linear_attention_postconv_core(
        captured_conv_output, gate, beta, recurrent_state, z, norm_weight,
        norm_epsilon);
    const auto cpu_projection = iq36::matvec_tensor(
        args.model_path, index, projection_name, captured_final_output);

    std::ifstream model(args.model_path, std::ios::binary);
    Require(static_cast<bool>(model), "failed to open model");
    const auto qkv_raw = ReadTensorBytes(model, *qkv_tensor);
    const auto conv_weights = ReadF32Tensor(
        model, *conv_tensor,
        static_cast<std::size_t>(kQkvRows * kConvKernelSize));
    const auto projection_raw = ReadTensorBytes(model, *projection_tensor);
    const std::uint64_t qkv_blocks_per_row = kHiddenSize / 256;
    const std::uint64_t projection_blocks_per_row = kFinalValues / 256;
    const auto projection_packed = iq36::PackQ4Kx8(
        projection_raw, kHiddenSize, projection_blocks_per_row);
    const auto q8 = iq36::QuantizeQ8KInputPlanes(attention_norm);
    const auto projection_q8 =
        iq36::QuantizeQ8KInputPlanes(captured_final_output);

    iq36::GpuQ4X8MatvecRunner runner(
        args.device_substring, kQ4X8OpenClSource);
    const auto q6_handle = runner.UploadRawQ6K(
        qkv_raw, kQkvRows, qkv_blocks_per_row);
    const auto conv_weights_handle = runner.UploadConvWeights(
        conv_weights, kQkvRows, kConvKernelSize);
    const auto conv_state_seed_handle = runner.UploadF32Buffer(conv_state_seed);
    const auto conv_output_seed_handle =
        runner.UploadF32Buffer(captured_conv_output);
    const auto recurrent_state_seed_handle =
        runner.UploadF32Buffer(recurrent_state);

    (void)RunSample(
        runner, q6_handle, q8, conv_weights_handle, conv_state_seed_handle,
        conv_output_seed_handle, recurrent_state_seed_handle, gate, beta, z,
        norm_weight, norm_epsilon, projection_packed, projection_q8,
        kHiddenSize, projection_blocks_per_row);

    std::vector<ComponentSample> samples;
    samples.reserve(static_cast<std::size_t>(args.samples));
    for (int i = 0; i < args.samples; ++i) {
      samples.push_back(RunSample(
          runner, q6_handle, q8, conv_weights_handle, conv_state_seed_handle,
          conv_output_seed_handle, recurrent_state_seed_handle, gate, beta, z,
          norm_weight, norm_epsilon, projection_packed, projection_q8,
          kHiddenSize, projection_blocks_per_row));
    }
    const ComponentSample& observed = samples.back();

    std::vector<double> current_preconv_times;
    std::vector<double> exact_preconv_times;
    std::vector<double> current_postconv_times;
    std::vector<double> exact_postconv_times;
    std::vector<double> current_projection_times;
    std::vector<double> exact_projection_times;
    std::vector<double> current_shell_times;
    std::vector<double> exact_shell_times;
    for (const auto& sample : samples) {
      current_preconv_times.push_back(sample.current_preconv_us);
      exact_preconv_times.push_back(sample.exact_preconv_us);
      current_postconv_times.push_back(sample.current_postconv_us);
      exact_postconv_times.push_back(sample.exact_postconv_us);
      current_projection_times.push_back(sample.current_projection_us);
      exact_projection_times.push_back(sample.exact_projection_us);
      current_shell_times.push_back(sample.current_shell_us);
      exact_shell_times.push_back(sample.exact_shell_us);
    }

    const auto cpu_qkv_vs_capture = iq36::compare_vectors(
        cpu_qkv, captured_qkv, kMismatchThreshold);
    const auto cpu_attention_vs_capture = iq36::compare_vectors(
        cpu_postconv.attention_output, captured_attention_output,
        kMismatchThreshold);
    const auto cpu_final_vs_capture = iq36::compare_vectors(
        cpu_postconv.final_output, captured_final_output, kMismatchThreshold);
    const auto cpu_projection_vs_capture = iq36::compare_vectors(
        cpu_projection, captured_projection, kMismatchThreshold);
    const auto exact_qkv_vs_cpu = iq36::compare_vectors(
        observed.exact_preconv.qkv_mixed, cpu_qkv, kMismatchThreshold);
    const auto exact_conv_output_vs_cpu = iq36::compare_vectors(
        observed.exact_preconv.conv_output_raw, cpu_conv.conv_output_raw,
        kMismatchThreshold);
    const auto exact_conv_state_vs_cpu = iq36::compare_vectors(
        observed.exact_preconv.conv_state, cpu_conv.conv_state,
        kMismatchThreshold);
    const auto exact_attention_vs_cpu = iq36::compare_vectors(
        observed.exact_postconv.attention_output,
        cpu_postconv.attention_output, kMismatchThreshold);
    const auto exact_final_vs_cpu = iq36::compare_vectors(
        observed.exact_postconv.final_output, cpu_postconv.final_output,
        kMismatchThreshold);
    const auto exact_recurrent_state_vs_cpu = iq36::compare_vectors(
        observed.exact_postconv.recurrent_state,
        cpu_postconv.recurrent_state, kMismatchThreshold);
    const auto exact_projection_vs_cpu = iq36::compare_vectors(
        observed.exact_projection.output, cpu_projection, kMismatchThreshold);
    const auto current_qkv_vs_cpu = iq36::compare_vectors(
        observed.current_preconv.qkv_mixed, cpu_qkv, kMismatchThreshold);
    const auto current_conv_output_vs_cpu = iq36::compare_vectors(
        observed.current_preconv.conv_output_raw, cpu_conv.conv_output_raw,
        kMismatchThreshold);
    const auto current_final_vs_cpu = iq36::compare_vectors(
        observed.current_postconv.final_output, cpu_postconv.final_output,
        kMismatchThreshold);

    const bool capture_oracles_exact =
        BitExact(cpu_qkv_vs_capture) && BitExact(cpu_attention_vs_capture) &&
        BitExact(cpu_final_vs_capture) && BitExact(cpu_projection_vs_capture);
    const bool exact_component_bit_exact =
        BitExact(exact_qkv_vs_cpu) && BitExact(exact_conv_output_vs_cpu) &&
        BitExact(exact_conv_state_vs_cpu) && BitExact(exact_attention_vs_cpu) &&
        BitExact(exact_final_vs_cpu) && BitExact(exact_recurrent_state_vs_cpu) &&
        BitExact(exact_projection_vs_cpu);
    const bool timing_positive =
        Min(current_shell_times) > 0.0 && Min(exact_shell_times) > 0.0;
    const bool checks_passed =
        load_map.ready && tensor_shapes_ok && payload_shapes_ok &&
        capture_oracles_exact && exact_component_bit_exact && timing_positive &&
        runner.device_name().find(args.device_substring) != std::string::npos;

    std::cout << std::setprecision(12);
    std::cout << "{";
    std::cout << "\"schema_version\":\""
              << "intel-qwen36-all-linear-preprojection-parity-component-probe-v0"
              << "\",";
    std::cout << "\"layer\":" << args.layer << ",";
    std::cout << "\"samples\":" << args.samples << ",";
    std::cout << "\"platform_name\":\"" << JsonEscape(runner.platform_name()) << "\",";
    std::cout << "\"device_name\":\"" << JsonEscape(runner.device_name()) << "\",";
    std::cout << "\"program_build_ms\":" << runner.program_build_ms() << ",";
    std::cout << "\"identical_input_bindings\":{";
    std::cout << "\"q8_planes_shared\":true,";
    std::cout << "\"conv_state_cloned_from_one_seed\":true,";
    std::cout << "\"conv_output_handle_shared\":true,";
    std::cout << "\"recurrent_state_cloned_from_one_seed\":true,";
    std::cout << "\"projection_q8_planes_shared\":true";
    std::cout << "},\"timings\":{";
    std::cout << "\"current_preconv_min_us\":" << Min(current_preconv_times) << ",";
    std::cout << "\"current_preconv_mean_us\":" << Mean(current_preconv_times) << ",";
    std::cout << "\"exact_preconv_min_us\":" << Min(exact_preconv_times) << ",";
    std::cout << "\"exact_preconv_mean_us\":" << Mean(exact_preconv_times) << ",";
    std::cout << "\"current_postconv_min_us\":" << Min(current_postconv_times) << ",";
    std::cout << "\"current_postconv_mean_us\":" << Mean(current_postconv_times) << ",";
    std::cout << "\"exact_postconv_min_us\":" << Min(exact_postconv_times) << ",";
    std::cout << "\"exact_postconv_mean_us\":" << Mean(exact_postconv_times) << ",";
    std::cout << "\"current_projection_min_us\":" << Min(current_projection_times) << ",";
    std::cout << "\"current_projection_mean_us\":" << Mean(current_projection_times) << ",";
    std::cout << "\"exact_projection_min_us\":" << Min(exact_projection_times) << ",";
    std::cout << "\"exact_projection_mean_us\":" << Mean(exact_projection_times) << ",";
    std::cout << "\"current_changed_shell_min_us\":" << Min(current_shell_times) << ",";
    std::cout << "\"current_changed_shell_mean_us\":" << Mean(current_shell_times) << ",";
    std::cout << "\"exact_changed_shell_min_us\":" << Min(exact_shell_times) << ",";
    std::cout << "\"exact_changed_shell_mean_us\":" << Mean(exact_shell_times) << ",";
    std::cout << "\"candidate_added_min_us\":"
              << (Min(exact_shell_times) - Min(current_shell_times)) << ",";
    std::cout << "\"candidate_added_mean_us\":"
              << (Mean(exact_shell_times) - Mean(current_shell_times));
    std::cout << "},\"comparisons\":{";
    std::cout << "\"cpu_qkv_vs_capture\":"; WriteCompare(cpu_qkv_vs_capture);
    std::cout << ",\"cpu_attention_vs_capture\":"; WriteCompare(cpu_attention_vs_capture);
    std::cout << ",\"cpu_final_vs_capture\":"; WriteCompare(cpu_final_vs_capture);
    std::cout << ",\"cpu_projection_vs_capture\":"; WriteCompare(cpu_projection_vs_capture);
    std::cout << ",\"exact_qkv_vs_cpu\":"; WriteCompare(exact_qkv_vs_cpu);
    std::cout << ",\"exact_conv_output_vs_cpu\":"; WriteCompare(exact_conv_output_vs_cpu);
    std::cout << ",\"exact_conv_state_vs_cpu\":"; WriteCompare(exact_conv_state_vs_cpu);
    std::cout << ",\"exact_attention_vs_cpu\":"; WriteCompare(exact_attention_vs_cpu);
    std::cout << ",\"exact_final_vs_cpu\":"; WriteCompare(exact_final_vs_cpu);
    std::cout << ",\"exact_recurrent_state_vs_cpu\":"; WriteCompare(exact_recurrent_state_vs_cpu);
    std::cout << ",\"exact_projection_vs_cpu\":"; WriteCompare(exact_projection_vs_cpu);
    std::cout << ",\"current_qkv_vs_cpu\":"; WriteCompare(current_qkv_vs_cpu);
    std::cout << ",\"current_conv_output_vs_cpu\":"; WriteCompare(current_conv_output_vs_cpu);
    std::cout << ",\"current_final_vs_cpu\":"; WriteCompare(current_final_vs_cpu);
    std::cout << "},\"checks\":{";
    std::cout << "\"load_map_ready\":" << (load_map.ready ? "true" : "false") << ",";
    std::cout << "\"tensor_shapes_ok\":" << (tensor_shapes_ok ? "true" : "false") << ",";
    std::cout << "\"payload_shapes_ok\":" << (payload_shapes_ok ? "true" : "false") << ",";
    std::cout << "\"capture_oracles_bit_exact\":" << (capture_oracles_exact ? "true" : "false") << ",";
    std::cout << "\"exact_component_bit_exact\":" << (exact_component_bit_exact ? "true" : "false") << ",";
    std::cout << "\"gpu_event_timing_positive\":" << (timing_positive ? "true" : "false") << ",";
    std::cout << "\"arc_device_selected\":"
              << (runner.device_name().find(args.device_substring) != std::string::npos ? "true" : "false") << ",";
    std::cout << "\"speedup_claims_allowed\":false";
    std::cout << "},\"required_checks_passed\":"
              << (checks_passed ? "true" : "false") << "}" << std::endl;
    return checks_passed ? 0 : 2;
  } catch (const std::exception& ex) {
    std::cerr << "exact preprojection component probe error: " << ex.what()
              << std::endl;
    return 1;
  }
}
'''


def cpp_raw_string_literal(value: str) -> str:
  delimiter = "IQ36PREPROJ"
  if f"){delimiter}\"" in value:
    raise ValueError(f"OpenCL source contains raw-string delimiter {delimiter}")
  return f'R"{delimiter}({value}){delimiter}"'


def generate_cpp(opencl_source: str) -> str:
  return PROBE_CPP.replace(
      "@@OPENCL_SOURCE_LITERAL@@", cpp_raw_string_literal(opencl_source))


def resolve_payloads(layer: int) -> dict[str, dict[str, Any]]:
  payloads: dict[str, dict[str, Any]] = {}
  for name, (stage_name, pattern, size_bytes) in PAYLOAD_SPECS.items():
    expected = PAYLOAD_ROOT / pattern.format(layer=layer)
    matches = [expected] if expected.exists() else sorted(
        PAYLOAD_ROOT.glob(pattern.format(layer=layer).replace("__ord1.bin", "__ord*.bin"))
    )
    if len(matches) != 1:
      wildcard = pattern.format(layer=layer).split("__ord", maxsplit=1)[0] + "__ord*.bin"
      matches = sorted(PAYLOAD_ROOT.glob(wildcard))
    if len(matches) != 1:
      raise SystemExit(
          f"component payload missing or ambiguous for {name}: {len(matches)} matches")
    path = matches[0].resolve()
    if path.stat().st_size != size_bytes:
      raise SystemExit(f"component payload size mismatch: {path}")
    payloads[name] = {
        "local_path": path,
        "path": str(path.relative_to(ROOT)),
        "stage_name": stage_name,
        "size_bytes": size_bytes,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
  return payloads
