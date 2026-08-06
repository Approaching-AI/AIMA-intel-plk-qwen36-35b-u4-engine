#!/usr/bin/env python3
"""Run the GPU Q4 x8 preconv-to-conv handoff gate."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shlex
from pathlib import Path
from typing import Any

import iq36_local


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA_VERSION = "intel-qwen36-gpu-q4x8-preconv-conv-handoff-probe-v0"
DEFAULT_HOST = "local"
DEFAULT_MODEL = "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf"
DEFAULT_ENV_SCRIPT = "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh"
DEFAULT_REMOTE_ROOT = "/home/intel/intel-qwen36-gpu"
DEFAULT_ORACLE_BUNDLE = ROOT / "oracle/r0-oracle-bundle-20260627T060028Z"
PAYLOAD_ROOT = ROOT / "output/r0-boundary-capture-run-20260627T054024Z/remote-output/payloads"
OPENCL_SOURCE = ROOT / "engine/gpu/opencl/q4x8_matvec.cl"
SOURCE_FILES = [
    ("engine/include/intel_qwen36/gguf_loader.hpp", "include/intel_qwen36/gguf_loader.hpp"),
    ("engine/include/intel_qwen36/gpu_q4x8_matvec.hpp", "include/intel_qwen36/gpu_q4x8_matvec.hpp"),
    ("engine/src/gguf_loader.cpp", "src/gguf_loader.cpp"),
    ("engine/src/gpu_q4x8_matvec.cpp", "src/gpu_q4x8_matvec.cpp"),
]
PAYLOAD_SPECS = {
    "attn_norm": ("attn_norm.bin", "attn_norm-{layer}__tok15__ord189.bin", 8192),
    "linear_attn_qkv_mixed": (
        "linear_attn_qkv_mixed.bin",
        "linear_attn_qkv_mixed-{layer}__tok15__ord190.bin",
        32768,
    ),
    "conv_output_raw": ("conv_output_raw.bin", "conv_output_raw-{layer}__tok15__ord191.bin", 32768),
}


PROBE_CPP = r'''
#include "intel_qwen36/gguf_loader.hpp"
#include "intel_qwen36/gpu_q4x8_matvec.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

const char* kQ4X8OpenClSource = @@OPENCL_SOURCE_LITERAL@@;

constexpr int kLayerCount = 40;
constexpr int kHiddenSize = 2048;
constexpr int kQkvMixedSize = 8192;
constexpr int kConvKernelSize = 4;
constexpr int kLinearHeadDim = 128;
constexpr int kLinearValueHeads = 32;
constexpr int kLinearRecurrentStateSize =
    kLinearHeadDim * kLinearHeadDim * kLinearValueHeads;
constexpr int kFullHeadDim = 256;
constexpr int kFullQHeadCount = 16;
constexpr int kFullKvHeadCount = 2;
constexpr int kSourceTokenPosition = 15;
constexpr float kAttentionScale = 0.0625f;

constexpr std::array<std::int64_t, 16> kPromptTokenIds = {
    15666, 303, 799, 2716, 11316, 25, 1092, 369,
    220,   16,  22,  5346, 220,   17, 20,   30,
};

struct Args {
  std::string model_path;
  std::string payload_dir;
  int layer = 5;
  int repeat = 7;
  std::string device_substring = "B390";
};

struct SequenceState {
  std::vector<std::vector<float>> linear_conv;
  std::vector<std::vector<float>> linear_recurrent;
  std::vector<std::vector<std::vector<float>>> full_k;
  std::vector<std::vector<std::vector<float>>> full_v;
};

struct ReplayFrontier {
  std::vector<float> residual_before_layer;
  std::vector<float> attention_norm;
  std::vector<float> conv_state_before;
  iq36::Qwen36LinearAttentionPreConvResult preconv;
  iq36::Qwen36LinearAttentionConvResult conv;
};

[[noreturn]] void Die(const std::string& message) {
  throw std::runtime_error(message);
}

void Require(bool ok, const std::string& message) {
  if (!ok) {
    Die(message);
  }
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

bool HasTensor(const iq36::GgufModelIndex& index, const std::string& name) {
  return iq36::find_tensor(index, name) != nullptr;
}

bool IsLinearLayer(const iq36::GgufModelIndex& index, int layer) {
  return HasTensor(index, LayerTensorName(layer, "ssm_out.weight"));
}

bool IsFullAttentionLayer(const iq36::GgufModelIndex& index, int layer) {
  return HasTensor(index, LayerTensorName(layer, "attn_output.weight"));
}

std::uint64_t MetadataUInt(const iq36::GgufModelIndex& index,
                           const std::string& key,
                           std::uint64_t fallback) {
  const auto found = index.metadata.find(key);
  if (found == index.metadata.end()) {
    return fallback;
  }
  const auto& value = found->second;
  if (value.kind == iq36::GgufMetadataValue::Kind::kUInt) {
    return value.uint_value;
  }
  if (value.kind == iq36::GgufMetadataValue::Kind::kInt && value.int_value >= 0) {
    return static_cast<std::uint64_t>(value.int_value);
  }
  return fallback;
}

float MetadataFloat(const iq36::GgufModelIndex& index,
                    const std::string& key,
                    float fallback) {
  const auto found = index.metadata.find(key);
  if (found == index.metadata.end()) {
    return fallback;
  }
  const auto& value = found->second;
  if (value.kind == iq36::GgufMetadataValue::Kind::kFloat) {
    return static_cast<float>(value.float_value);
  }
  return fallback;
}

std::vector<std::int64_t> MetadataIntArray(const iq36::GgufModelIndex& index,
                                           const std::string& key,
                                           const std::vector<std::int64_t>& fallback) {
  const auto found = index.metadata.find(key);
  if (found == index.metadata.end()) {
    return fallback;
  }
  const auto& value = found->second;
  if (value.kind == iq36::GgufMetadataValue::Kind::kArray && !value.int_array.empty()) {
    return value.int_array;
  }
  return fallback;
}

std::vector<std::uint8_t> ReadTensorBytes(std::ifstream& in,
                                          const iq36::GgufTensorInfo& tensor) {
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(tensor.nbytes));
  in.clear();
  in.seekg(static_cast<std::streamoff>(tensor.absolute_offset), std::ios::beg);
  Require(static_cast<bool>(in), "failed to seek tensor payload");
  in.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  Require(in.gcount() == static_cast<std::streamsize>(bytes.size()), "failed to read tensor payload");
  return bytes;
}

std::vector<float> ReadF32Tensor(std::ifstream& in,
                                 const iq36::GgufTensorInfo& tensor,
                                 std::size_t expected_values) {
  Require(tensor.type == 0, "tensor is not F32: " + tensor.name);
  Require(tensor.nbytes == expected_values * sizeof(float), "F32 tensor byte size mismatch: " + tensor.name);
  const auto bytes = ReadTensorBytes(in, tensor);
  std::vector<float> values(expected_values, 0.0f);
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
    else if (key == "--repeat") args.repeat = std::stoi(value("--repeat"));
    else if (key == "--device-substring") args.device_substring = value("--device-substring");
    else Die("unknown argument: " + key);
  }
  Require(!args.model_path.empty(), "--model is required");
  Require(!args.payload_dir.empty(), "--payload-dir is required");
  Require(args.layer >= 0 && args.layer < kLayerCount, "--layer is out of range");
  Require(args.repeat > 0, "--repeat must be positive");
  return args;
}

bool ComparePassed(const iq36::VectorCompareStats& stats) {
  return stats.same_size &&
         stats.finite &&
         stats.mismatch_count == 0 &&
         stats.max_abs_diff <= 5e-3 &&
         stats.rmse <= 1e-3 &&
         stats.cosine >= 0.99999;
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

SequenceState MakeState(const iq36::GgufModelIndex& index) {
  SequenceState state;
  state.linear_conv.resize(kLayerCount);
  state.linear_recurrent.resize(kLayerCount);
  state.full_k.resize(kLayerCount);
  state.full_v.resize(kLayerCount);
  for (int layer = 0; layer < kLayerCount; ++layer) {
    if (!IsLinearLayer(index, layer)) {
      continue;
    }
    const auto* conv = iq36::find_tensor(index, LayerTensorName(layer, "ssm_conv1d.weight"));
    Require(conv != nullptr, "linear layer conv tensor missing");
    Require(conv->dims == std::vector<std::uint64_t>{kConvKernelSize, kQkvMixedSize},
            "linear layer conv tensor shape mismatch");
    state.linear_conv[layer].assign(
        static_cast<std::size_t>((kConvKernelSize - 1) * kQkvMixedSize), 0.0f);
    state.linear_recurrent[layer].assign(
        static_cast<std::size_t>(kLinearRecurrentStateSize), 0.0f);
  }
  return state;
}

ReplayFrontier ReplayToFrontier(const std::string& model_path,
                                const iq36::GgufModelIndex& index,
                                int target_layer,
                                float rms_norm_epsilon,
                                std::uint64_t full_head_dim,
                                std::uint64_t q_head_count,
                                std::uint64_t kv_head_count,
                                std::uint64_t rope_dimension_count,
                                const std::vector<std::int64_t>& rope_sections,
                                std::uint64_t rope_context_length,
                                float rope_freq_base) {
  constexpr float kRopeFreqScale = 1.0f;
  constexpr float kRopeExtFactor = 0.0f;
  constexpr float kRopeAttnFactor = 1.0f;
  constexpr float kRopeBetaFast = 32.0f;
  constexpr float kRopeBetaSlow = 1.0f;
  auto state = MakeState(index);
  ReplayFrontier frontier;

  for (std::size_t pos = 0; pos < kPromptTokenIds.size(); ++pos) {
    std::vector<float> residual = iq36::decode_tensor_row(
        model_path, index, "token_embd.weight",
        static_cast<std::uint64_t>(kPromptTokenIds[pos]));
    const int layer_limit =
        static_cast<int>(pos) == kSourceTokenPosition ? target_layer : target_layer + 1;
    for (int layer = 0; layer < layer_limit; ++layer) {
      if (IsLinearLayer(index, layer)) {
        auto layer_result = iq36::run_qwen36_stateful_linear_attention_layer(
            model_path,
            index,
            layer,
            residual,
            state.linear_conv[layer],
            state.linear_recurrent[layer],
            rms_norm_epsilon);
        state.linear_conv[layer] = std::move(layer_result.conv.conv_state);
        state.linear_recurrent[layer] = std::move(layer_result.attention.recurrent_state);
        residual = std::move(layer_result.residual);
      } else if (IsFullAttentionLayer(index, layer)) {
        auto attention = iq36::run_qwen36_stateful_full_attention_layer(
            model_path,
            index,
            layer,
            residual,
            state.full_k[layer],
            state.full_v[layer],
            static_cast<std::int32_t>(pos),
            full_head_dim,
            q_head_count,
            kv_head_count,
            rope_dimension_count,
            rope_sections,
            rope_context_length,
            rope_freq_base,
            kRopeFreqScale,
            kRopeExtFactor,
            kRopeAttnFactor,
            kRopeBetaFast,
            kRopeBetaSlow,
            kAttentionScale,
            rms_norm_epsilon);
        state.full_k[layer] = std::move(attention.k_history);
        state.full_v[layer] = std::move(attention.v_history);
        const auto attention_residual = iq36::add_vectors(residual, attention.attention_output);
        auto ffn = iq36::run_qwen36_moe_ffn_layer(
            model_path, index, layer, attention_residual, rms_norm_epsilon);
        residual = std::move(ffn.residual);
      } else {
        Die("layer has neither linear nor full attention tensors");
      }
    }
    if (static_cast<int>(pos) == kSourceTokenPosition) {
      frontier.residual_before_layer = residual;
      frontier.conv_state_before = state.linear_conv[target_layer];
      const auto norm_weight = iq36::decode_tensor_row(
          model_path, index, LayerTensorName(target_layer, "attn_norm.weight"), 0);
      frontier.attention_norm = iq36::apply_rms_norm(residual, norm_weight, rms_norm_epsilon);
      frontier.preconv = iq36::run_qwen36_linear_attention_preconv_core(
          model_path, index, target_layer, frontier.attention_norm);
      frontier.conv = iq36::run_qwen36_linear_attention_conv_core(
          model_path, index, target_layer, frontier.preconv.qkv_mixed,
          frontier.conv_state_before);
      return frontier;
    }
  }
  Die("source token position was not replayed");
}

int main(int argc, char** argv) {
  try {
    const Args args = ParseArgs(argc, argv);
    const auto index = iq36::parse_gguf_model_index(args.model_path);
    const auto load_map = iq36::validate_qwen36_load_map(index);
    Require(IsLinearLayer(index, args.layer), "target layer is not linear attention");

    const auto oracle_attn_norm = iq36::read_f32_vector_file(JoinPath(args.payload_dir, "attn_norm.bin"));
    const auto oracle_qkv = iq36::read_f32_vector_file(JoinPath(args.payload_dir, "linear_attn_qkv_mixed.bin"));
    const auto oracle_conv = iq36::read_f32_vector_file(JoinPath(args.payload_dir, "conv_output_raw.bin"));
    Require(oracle_attn_norm.size() == kHiddenSize, "oracle attn_norm size mismatch");
    Require(oracle_qkv.size() == kQkvMixedSize, "oracle qkv size mismatch");
    Require(oracle_conv.size() == kQkvMixedSize, "oracle conv size mismatch");

    const float rms_norm_epsilon = MetadataFloat(
        index, "qwen35moe.attention.layer_norm_rms_epsilon", 1e-6f);
    const auto full_head_dim = MetadataUInt(index, "qwen35moe.attention.key_length", kFullHeadDim);
    const auto full_value_length = MetadataUInt(index, "qwen35moe.attention.value_length", kFullHeadDim);
    const auto q_head_count = MetadataUInt(index, "qwen35moe.attention.head_count", kFullQHeadCount);
    const auto kv_head_count = MetadataUInt(index, "qwen35moe.attention.head_count_kv", kFullKvHeadCount);
    const auto rope_dimension_count = MetadataUInt(index, "qwen35moe.rope.dimension_count", 64);
    const auto rope_context_length = MetadataUInt(index, "qwen35moe.context_length", 262144);
    const auto rope_sections = MetadataIntArray(index, "qwen35moe.rope.dimension_sections", {11, 11, 10, 0});
    const float rope_freq_base = MetadataFloat(index, "qwen35moe.rope.freq_base", 10000000.0f);
    Require(full_head_dim == full_value_length, "full attention key/value length mismatch");

    const auto frontier = ReplayToFrontier(
        args.model_path,
        index,
        args.layer,
        rms_norm_epsilon,
        full_head_dim,
        q_head_count,
        kv_head_count,
        rope_dimension_count,
        rope_sections,
        rope_context_length,
        rope_freq_base);
    const auto cpu_preconv = iq36::run_qwen36_linear_attention_preconv_core(
        args.model_path, index, args.layer, oracle_attn_norm);
    const auto cpu_conv = iq36::run_qwen36_linear_attention_conv_core(
        args.model_path, index, args.layer, cpu_preconv.qkv_mixed,
        frontier.conv_state_before);

    const auto* qkv_tensor = iq36::find_tensor(index, LayerTensorName(args.layer, "attn_qkv.weight"));
    const auto* conv_tensor = iq36::find_tensor(index, LayerTensorName(args.layer, "ssm_conv1d.weight"));
    Require(qkv_tensor != nullptr && conv_tensor != nullptr, "required tensors missing");
    Require(qkv_tensor->type == 12, "qkv tensor is not Q4_K");
    Require(qkv_tensor->dims == std::vector<std::uint64_t>{kHiddenSize, kQkvMixedSize},
            "qkv tensor shape mismatch");
    Require(conv_tensor->type == 0, "conv tensor is not F32");
    Require(conv_tensor->dims == std::vector<std::uint64_t>{kConvKernelSize, kQkvMixedSize},
            "conv tensor shape mismatch");

    std::ifstream model(args.model_path, std::ios::binary);
    Require(static_cast<bool>(model), "failed to open model");
    const auto qkv_raw = ReadTensorBytes(model, *qkv_tensor);
    const auto qkv_packed = iq36::PackQ4Kx8(qkv_raw, kQkvMixedSize, kHiddenSize / 256);
    const auto conv_weights = ReadF32Tensor(
        model, *conv_tensor, static_cast<std::size_t>(kQkvMixedSize * kConvKernelSize));
    const auto q8 = iq36::QuantizeQ8KInputPlanes(oracle_attn_norm);

    iq36::GpuQ4X8MatvecRunner runner(args.device_substring, kQ4X8OpenClSource);
    const auto gpu = runner.RunThenConv(
        qkv_packed,
        q8.qs,
        q8.bsums,
        q8.d,
        conv_weights,
        frontier.conv_state_before,
        kQkvMixedSize,
        kHiddenSize / 256,
        kConvKernelSize,
        args.repeat,
        iq36::GpuQ4X8KernelVariant::kRowlaneParallel);

    struct CompareRow {
      std::string name;
      iq36::VectorCompareStats cpu_vs_oracle;
      iq36::VectorCompareStats gpu_vs_cpu;
      iq36::VectorCompareStats gpu_vs_oracle;
    };
    const auto attention_norm_cpu_vs_oracle =
        iq36::compare_vectors(frontier.attention_norm, oracle_attn_norm, 5e-3);
    const auto conv_state_gpu_vs_cpu =
        iq36::compare_vectors(gpu.conv_state, cpu_conv.conv_state, 5e-3);
    const auto conv_output_cpu_vs_oracle_diagnostic =
        iq36::compare_vectors(cpu_conv.conv_output_raw, oracle_conv, 5e-3);
    const auto conv_output_gpu_vs_oracle_diagnostic =
        iq36::compare_vectors(gpu.conv_output_raw, oracle_conv, 5e-3);
    std::vector<CompareRow> rows = {
        {"linear_attn_qkv_mixed",
         iq36::compare_vectors(cpu_preconv.qkv_mixed, oracle_qkv, 5e-3),
         iq36::compare_vectors(gpu.qkv_mixed, cpu_preconv.qkv_mixed, 5e-3),
         iq36::compare_vectors(gpu.qkv_mixed, oracle_qkv, 5e-3)},
        {"conv_output_raw",
         conv_output_cpu_vs_oracle_diagnostic,
         iq36::compare_vectors(gpu.conv_output_raw, cpu_conv.conv_output_raw, 5e-3),
         conv_output_gpu_vs_oracle_diagnostic},
    };
    const bool qkv_oracle_input_matches_oracle =
        ComparePassed(rows[0].cpu_vs_oracle) && ComparePassed(rows[0].gpu_vs_cpu) &&
        ComparePassed(rows[0].gpu_vs_oracle);
    const bool conv_handoff_matches_cpu =
        ComparePassed(rows[1].gpu_vs_cpu) && ComparePassed(conv_state_gpu_vs_cpu);
    const bool comparisons_passed =
        qkv_oracle_input_matches_oracle && conv_handoff_matches_cpu;
    const bool timings_positive =
        gpu.timing.matvec.min_us > 0.0 && gpu.timing.conv_min_us > 0.0;
    const bool checks_passed =
        load_map.ready &&
        runner.device_name().find(args.device_substring) != std::string::npos &&
        frontier.conv_state_before.size() ==
            static_cast<std::size_t>((kConvKernelSize - 1) * kQkvMixedSize) &&
        comparisons_passed &&
        timings_positive;

    std::cout << std::setprecision(10);
    std::cout << "{";
    std::cout << "\"schema_version\":\"intel-qwen36-gpu-q4x8-preconv-conv-handoff-probe-v0\",";
    std::cout << "\"model_path\":\"" << JsonEscape(args.model_path) << "\",";
    std::cout << "\"layer\":" << args.layer << ",";
    std::cout << "\"source_token_position\":" << kSourceTokenPosition << ",";
    std::cout << "\"platform_name\":\"" << JsonEscape(runner.platform_name()) << "\",";
    std::cout << "\"device_name\":\"" << JsonEscape(runner.device_name()) << "\",";
    std::cout << "\"program_build_ms\":" << runner.program_build_ms() << ",";
    std::cout << "\"build_log\":\"" << JsonEscape(runner.build_log()) << "\",";
    std::cout << "\"repeat\":" << args.repeat << ",";
    std::cout << "\"replay\":{";
    std::cout << "\"residual_before_layer_values\":" << frontier.residual_before_layer.size() << ",";
    std::cout << "\"conv_state_before_values\":" << frontier.conv_state_before.size() << ",";
    std::cout << "\"prompt_token_count\":" << kPromptTokenIds.size() << ",";
    std::cout << "\"attention_norm_replay_is_required\":false";
    std::cout << "},";
    std::cout << "\"timings\":{";
    std::cout << "\"qkv_gpu_kernel_min_us\":" << gpu.timing.matvec.min_us << ",";
    std::cout << "\"qkv_gpu_kernel_mean_us\":" << gpu.timing.matvec.mean_us << ",";
    std::cout << "\"qkv_gpu_effective_packed_gb_s\":" << gpu.timing.matvec.effective_packed_gb_s << ",";
    std::cout << "\"qkv_global_work_items\":" << gpu.timing.matvec.global_work_items << ",";
    std::cout << "\"qkv_rows_per_work_item\":" << gpu.timing.matvec.rows_per_work_item << ",";
    std::cout << "\"conv_gpu_kernel_min_us\":" << gpu.timing.conv_min_us << ",";
    std::cout << "\"conv_gpu_kernel_mean_us\":" << gpu.timing.conv_mean_us << ",";
    std::cout << "\"conv_global_work_items\":" << gpu.timing.conv_global_work_items;
    std::cout << "},\"comparisons\":{";
    std::cout << "\"attention_norm_cpu_replay_vs_oracle\":";
    WriteCompare(attention_norm_cpu_vs_oracle);
    std::cout << ",\"conv_state_gpu_vs_cpu\":";
    WriteCompare(conv_state_gpu_vs_cpu);
    std::cout << ",\"conv_output_cpu_vs_captured_oracle_diagnostic\":";
    WriteCompare(conv_output_cpu_vs_oracle_diagnostic);
    std::cout << ",\"conv_output_gpu_vs_captured_oracle_diagnostic\":";
    WriteCompare(conv_output_gpu_vs_oracle_diagnostic);
    for (const auto& row : rows) {
      std::cout << ",\"" << JsonEscape(row.name) << "\":{";
      std::cout << "\"cpu_vs_oracle\":";
      WriteCompare(row.cpu_vs_oracle);
      std::cout << ",\"gpu_vs_cpu\":";
      WriteCompare(row.gpu_vs_cpu);
      std::cout << ",\"gpu_vs_oracle\":";
      WriteCompare(row.gpu_vs_oracle);
      std::cout << "}";
    }
    std::cout << "},\"checks\":{";
    std::cout << "\"load_map_ready\":" << (load_map.ready ? "true" : "false") << ",";
    std::cout << "\"arc_device_selected\":" << (runner.device_name().find(args.device_substring) != std::string::npos ? "true" : "false") << ",";
    std::cout << "\"cpu_replay_frontier_matches_oracle_diagnostic\":" << (ComparePassed(attention_norm_cpu_vs_oracle) ? "true" : "false") << ",";
    std::cout << "\"qkv_oracle_input_matches_oracle\":" << (qkv_oracle_input_matches_oracle ? "true" : "false") << ",";
    std::cout << "\"conv_handoff_matches_cpu\":" << (conv_handoff_matches_cpu ? "true" : "false") << ",";
    std::cout << "\"captured_conv_output_oracle_available\":false,";
    std::cout << "\"comparisons_passed\":" << (comparisons_passed ? "true" : "false") << ",";
    std::cout << "\"gpu_event_timing_positive\":" << (timings_positive ? "true" : "false") << ",";
    std::cout << "\"speedup_claims_allowed\":false";
    std::cout << "},\"required_checks_passed\":" << (checks_passed ? "true" : "false");
    std::cout << "}\n";
    return checks_passed ? 0 : 3;
  } catch (const std::exception& exc) {
    std::cout << "{\"ok\":false,\"error\":\"" << JsonEscape(exc.what()) << "\"}\n";
    return 2;
  }
}
'''


def iso_now() -> str:
  return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_stamp() -> str:
  return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--target", "--host", dest="host", metavar="TARGET", default=DEFAULT_HOST)
  parser.add_argument("--model", default=DEFAULT_MODEL)
  parser.add_argument("--env-script", default=DEFAULT_ENV_SCRIPT)
  parser.add_argument("--staging-root", "--remote-root", dest="remote_root", metavar="PATH", default=DEFAULT_REMOTE_ROOT)
  parser.add_argument("--oracle-bundle", type=Path, default=DEFAULT_ORACLE_BUNDLE)
  parser.add_argument("--layer", type=int, default=5)
  parser.add_argument("--repeat", type=int, default=7)
  parser.add_argument("--device-substring", default="B390")
  parser.add_argument("--timeout-s", type=int, default=1200)
  parser.add_argument("--out-dir", type=Path, default=None)
  return parser.parse_args()


def shell_join(argv: list[str]) -> str:
  return " ".join(shlex.quote(item) for item in argv)


def cpp_raw_string_literal(value: str) -> str:
  delimiter = "IQ36CL"
  if f"){delimiter}\"" in value:
    raise ValueError(f"OpenCL source contains raw-string delimiter {delimiter}")
  return f'R"{delimiter}({value}){delimiter}"'


def parse_probe_stdout(stdout: str) -> dict[str, Any] | None:
  for line in reversed(stdout.splitlines()):
    line = line.strip()
    if not line.startswith("{"):
      continue
    try:
      value = json.loads(line)
    except json.JSONDecodeError:
      continue
    if isinstance(value, dict):
      return value
  return None


def resolve_payloads(layer: int) -> dict[str, dict[str, Any]]:
  payloads: dict[str, dict[str, Any]] = {}
  for name, (stage_name, pattern, size_bytes) in PAYLOAD_SPECS.items():
    path = (PAYLOAD_ROOT / pattern.format(layer=layer)).resolve()
    if not path.exists():
      raise SystemExit(f"preconv-to-conv handoff payload missing: {path}")
    if path.stat().st_size != size_bytes:
      raise SystemExit(f"preconv-to-conv handoff payload size mismatch: {path}")
    payloads[name] = {
        "local_path": path,
        "path": str(path.relative_to(ROOT)),
        "sha256": iq36_local.sha256_file(path),
        "size_bytes": size_bytes,
        "stage_name": stage_name,
    }
  return payloads


def nested_bool(obj: dict[str, Any], *keys: str) -> bool:
  current: Any = obj
  for key in keys:
    if not isinstance(current, dict):
      return False
    current = current.get(key)
  return current is True


def nested_number(obj: dict[str, Any], *keys: str) -> float | None:
  current: Any = obj
  for key in keys:
    if not isinstance(current, dict):
      return None
    current = current.get(key)
  return float(current) if isinstance(current, (int, float)) else None


def write_summary(path: Path, payload: dict[str, Any]) -> None:
  probe = payload.get("probe") if isinstance(payload.get("probe"), dict) else {}
  comparisons = probe.get("comparisons", {}) if isinstance(probe, dict) else {}
  timings = probe.get("timings", {}) if isinstance(probe, dict) else {}
  qkv_cmp = comparisons.get("linear_attn_qkv_mixed", {}) if isinstance(comparisons, dict) else {}
  conv_cmp = comparisons.get("conv_output_raw", {}) if isinstance(comparisons, dict) else {}
  lines = [
      "# GPU Q4-X8 Preconv-to-Conv Handoff Probe",
      "",
      f"- workstream: `{WORKSTREAM}`",
      f"- host: `{payload.get('host')}`",
      f"- model: `{payload.get('model')}`",
      f"- layer: `{payload.get('layer')}`",
      f"- required checks passed: `{str(payload.get('required_checks_passed')).lower()}`",
      "- speedup claims allowed: `false`",
      f"- platform/device: `{probe.get('platform_name')}` / `{probe.get('device_name')}`",
      "",
      "| boundary | min us | required comparison max abs | required comparison RMSE |",
      "|---|---:|---:|---:|",
      "| qkv matvec | "
      f"{timings.get('qkv_gpu_kernel_min_us')} | "
      f"{qkv_cmp.get('gpu_vs_oracle', {}).get('max_abs_diff')} | "
      f"{qkv_cmp.get('gpu_vs_oracle', {}).get('rmse')} |",
      "| conv output | "
      f"{timings.get('conv_gpu_kernel_min_us')} | "
      f"{conv_cmp.get('gpu_vs_cpu', {}).get('max_abs_diff')} | "
      f"{conv_cmp.get('gpu_vs_cpu', {}).get('rmse')} |",
      "",
      "The probe uses captured oracle `attn_norm` for QKV, reconstructs a valid",
      "layer-5 conv state by CPU native prompt replay, then runs Q4 x8 QKV matvec",
      "and F32 depthwise conv back-to-back on the GPU.",
      "Captured `conv_output_raw` is diagnostic only because the capture bundle",
      "does not include the pre-token conv history state.",
      "This is component evidence only; it does not prove decode or model throughput.",
      "",
  ]
  path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  args = parse_args()
  stamp = utc_stamp()
  created_at = iso_now()
  out_dir = args.out_dir or ROOT / f"output/gpu-q4x8-preconv-conv-handoff-probe-{stamp}"
  out_dir.mkdir(parents=True, exist_ok=False)
  raw_dir = out_dir / "raw"
  raw_dir.mkdir()
  payloads = resolve_payloads(args.layer)
  opencl_source = OPENCL_SOURCE.read_text(encoding="utf-8")
  opencl_source_hash = iq36_local.sha256_file(OPENCL_SOURCE)
  local_cpp = out_dir / "gpu_q4x8_preconv_conv_handoff_probe.cpp"
  local_cpp.write_text(
      PROBE_CPP.replace("@@OPENCL_SOURCE_LITERAL@@", cpp_raw_string_literal(opencl_source)),
      encoding="utf-8",
  )

  remote_dir = f"{args.remote_root.rstrip('/')}/gpu-q4x8-preconv-conv-handoff-probe-{stamp}"
  setup = iq36_local.run_target(
      args.host,
      "rm -rf " + shlex.quote(remote_dir) + " && mkdir -p "
      + " ".join(
          shlex.quote(f"{remote_dir}/{subdir}")
          for subdir in ("include/intel_qwen36", "src", "tests", "build", "oracle")
      ),
      args.timeout_s,
  )
  transfers: list[dict[str, Any]] = []
  payload_transfers: dict[str, dict[str, Any]] = {
      name: {"returncode": 1, "stdout": "", "stderr": "stage failed"}
      for name in payloads
  }
  remote_payload_dir = f"{remote_dir}/oracle"
  if setup.get("returncode") == 0:
    for local, remote in SOURCE_FILES:
      transfers.append(iq36_local.copy_to(args.host, ROOT / local, f"{remote_dir}/{remote}", args.timeout_s))
    transfers.append(iq36_local.copy_to(args.host, local_cpp, f"{remote_dir}/tests/gpu_q4x8_preconv_conv_handoff_probe.cpp", args.timeout_s))
    for name, payload in payloads.items():
      payload_transfers[name] = iq36_local.copy_to(
          args.host,
          payload["local_path"],
          f"{remote_payload_dir}/{payload['stage_name']}",
          args.timeout_s,
      )

  compile_cmd = " && ".join([
      f"source {shlex.quote(args.env_script)} >/dev/null 2>&1",
      (
          "g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic "
          f"-I {shlex.quote(remote_dir + '/include')} "
          f"{shlex.quote(remote_dir + '/src/gguf_loader.cpp')} "
          f"{shlex.quote(remote_dir + '/src/gpu_q4x8_matvec.cpp')} "
          f"{shlex.quote(remote_dir + '/tests/gpu_q4x8_preconv_conv_handoff_probe.cpp')} "
          "-ldl -pthread "
          f"-o {shlex.quote(remote_dir + '/build/iq36-gpu-q4x8-preconv-conv-handoff-probe')}"
      ),
  ])
  stage_ok = (
      setup.get("returncode") == 0
      and transfers
      and all(item.get("returncode") == 0 for item in transfers)
      and all(item.get("returncode") == 0 for item in payload_transfers.values())
  )
  compile_result = (
      iq36_local.run_target(args.host, compile_cmd, args.timeout_s)
      if stage_ok
      else {"cmd": ["stage"], "returncode": 1, "stdout": "", "stderr": "stage failed"}
  )
  run_argv = [
      f"{remote_dir}/build/iq36-gpu-q4x8-preconv-conv-handoff-probe",
      "--model", args.model,
      "--payload-dir", remote_payload_dir,
      "--layer", str(args.layer),
      "--repeat", str(args.repeat),
      "--device-substring", args.device_substring,
  ]
  run_result = (
      iq36_local.run_target(
          args.host,
          " && ".join([
              f"source {shlex.quote(args.env_script)} >/dev/null 2>&1",
              shell_join(run_argv),
          ]),
          args.timeout_s,
      )
      if compile_result.get("returncode") == 0
      else {"cmd": run_argv, "returncode": None, "stdout": "", "stderr": "compile skipped run"}
  )
  probe = parse_probe_stdout(run_result.get("stdout", ""))
  iq36_local.write_json(raw_dir / "setup.json", setup)
  iq36_local.write_json(raw_dir / "transfers.json", transfers)
  iq36_local.write_json(raw_dir / "payload-transfers.json", payload_transfers)
  iq36_local.write_json(raw_dir / "compile.json", compile_result)
  iq36_local.write_json(raw_dir / "run.json", run_result)
  if probe is not None:
    iq36_local.write_json(out_dir / "probe-result.json", probe)

  checks = [
      {"name": "remote_dir_created", "pass": setup.get("returncode") == 0},
      {"name": "source_files_transferred", "pass": bool(transfers) and all(item.get("returncode") == 0 for item in transfers)},
      {"name": "oracle_payloads_transferred", "pass": all(item.get("returncode") == 0 for item in payload_transfers.values())},
      {"name": "probe_compiled", "pass": compile_result.get("returncode") == 0},
      {"name": "probe_stdout_json_parsed", "pass": isinstance(probe, dict)},
      {"name": "probe_process_succeeded", "pass": run_result.get("returncode") == 0},
      {"name": "arc_b390_selected", "pass": bool(probe and "B390" in str(probe.get("device_name", "")))},
      {"name": "qkv_oracle_input_matches_oracle", "pass": bool(probe and nested_bool(probe, "checks", "qkv_oracle_input_matches_oracle"))},
      {"name": "conv_handoff_matches_cpu", "pass": bool(probe and nested_bool(probe, "checks", "conv_handoff_matches_cpu"))},
      {"name": "comparisons_passed", "pass": bool(probe and nested_bool(probe, "checks", "comparisons_passed"))},
      {"name": "gpu_event_timing_positive", "pass": bool(probe and nested_bool(probe, "checks", "gpu_event_timing_positive"))},
      {"name": "speedup_claims_forbidden", "pass": True},
  ]
  required_checks_passed = all(item["pass"] for item in checks)
  slim_payloads = {
      name: {key: value for key, value in payload.items() if key != "local_path"}
      for name, payload in payloads.items()
  }
  payload = {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "created_at": created_at,
      "host": args.host,
      "remote_dir": remote_dir,
      "model": args.model,
      "oracle_bundle": str(args.oracle_bundle.resolve().relative_to(ROOT)),
      "payloads": slim_payloads,
      "layer": args.layer,
      "repeat": args.repeat,
      "engine_shim_header": "engine/include/intel_qwen36/gpu_q4x8_matvec.hpp",
      "engine_shim_source": "engine/src/gpu_q4x8_matvec.cpp",
      "opencl_source": str(OPENCL_SOURCE.relative_to(ROOT)),
      "opencl_source_sha256": opencl_source_hash,
      "probe": probe,
      "checks": checks,
      "required_checks_passed": required_checks_passed,
      "speedup_claims_allowed": False,
  }
  manifest = {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "created_at": created_at,
      "tool": "tools/intel-qwen36-gpu-q4x8-preconv-conv-handoff-probe.py",
      "artifact": str(out_dir),
      "host": args.host,
      "remote_dir": remote_dir,
      "layer": args.layer,
      "required_checks_passed": required_checks_passed,
      "speedup_claims_allowed": False,
  }
  correctness = {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "checks": checks,
      "required_checks_passed": required_checks_passed,
      "speedup_claims_allowed": False,
  }
  iq36_local.write_json(out_dir / "probe.json", payload)
  iq36_local.write_json(out_dir / "manifest.json", manifest)
  iq36_local.write_json(out_dir / "correctness.json", correctness)
  aggregate = probe if isinstance(probe, dict) else {}
  timings = aggregate.get("timings", {}) if isinstance(aggregate.get("timings"), dict) else {}
  comparisons = aggregate.get("comparisons", {}) if isinstance(aggregate.get("comparisons"), dict) else {}
  iq36_local.write_metric(
      out_dir / "metrics.jsonl",
      "gpu_q4x8_preconv_conv_handoff_probe",
      [
          ("required_checks_passed", required_checks_passed),
          ("qkv_kernel_min_us", nested_number(timings, "qkv_gpu_kernel_min_us")),
          ("conv_kernel_min_us", nested_number(timings, "conv_gpu_kernel_min_us")),
          ("qkv_gpu_vs_oracle_max_abs_diff", nested_number(comparisons, "linear_attn_qkv_mixed", "gpu_vs_oracle", "max_abs_diff")),
          ("conv_gpu_vs_cpu_max_abs_diff", nested_number(comparisons, "conv_output_raw", "gpu_vs_cpu", "max_abs_diff")),
          ("conv_gpu_vs_captured_oracle_diagnostic_max_abs_diff", nested_number(comparisons, "conv_output_raw", "gpu_vs_oracle", "max_abs_diff")),
          ("conv_state_gpu_vs_cpu_max_abs_diff", nested_number(comparisons, "conv_state_gpu_vs_cpu", "max_abs_diff")),
      ],
  )
  write_summary(out_dir / "summary.md", payload)
  print(out_dir)
  return 0 if required_checks_passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
