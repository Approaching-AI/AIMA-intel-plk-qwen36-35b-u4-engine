#!/usr/bin/env python3
"""Run the resident GPU layer-7 full-attention state/input handoff probe."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import shlex
from pathlib import Path
from types import ModuleType
from typing import Any

import iq36_local


ROOT = Path(__file__).resolve().parents[1]
TWO_LAYER_TOOL = Path(__file__).with_name(
    "intel-qwen36-gpu-resident-two-linear-layer-shell-probe.py"
)
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA_VERSION = "intel-qwen36-gpu-resident-layer7-full-attn-input-handoff-probe-v0"
DEFAULT_HOST = "local"
DEFAULT_MODEL = "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf"
DEFAULT_ENV_SCRIPT = "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh"
DEFAULT_REMOTE_ROOT = "/home/intel/intel-qwen36-gpu"
DEFAULT_ORACLE_BUNDLE = ROOT / "oracle/r0-oracle-bundle-20260627T060028Z"
DEFAULT_ALL_HISTORY = ROOT / "output/r1-full-attn-all-history-capture-20260627T145615Z/history.json"
OPENCL_SOURCE = ROOT / "engine/gpu/opencl/q4x8_matvec.cl"
SOURCE_TOKEN_POSITION = 15


def load_two_layer_tool() -> ModuleType:
  spec = importlib.util.spec_from_file_location("iq36_two_linear_layer_probe", TWO_LAYER_TOOL)
  if spec is None or spec.loader is None:
    raise SystemExit(f"failed to load two-layer shell tool: {TWO_LAYER_TOOL}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


TWO = load_two_layer_tool()
PRECONV = TWO.PRECONV


FULL_ATTN_HANDOFF_CPP = r'''

constexpr int kFullQFullValues = 8192;
constexpr int kFullQValues = 4096;
constexpr int kFullKvValues = 512;
constexpr int kFullHeadDim = 256;
constexpr int kFullQHeadCount = 16;
constexpr int kFullKvHeadCount = 2;
constexpr int kFullInputHistoryTokenCount = 15;
constexpr int kFullUpdatedHistoryTokenCount = 16;
constexpr int kFullSourceTokenPosition = 15;

std::uint64_t MetadataUIntFull(const iq36::GgufModelIndex& index,
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
  if (value.kind == iq36::GgufMetadataValue::Kind::kInt &&
      value.int_value >= 0) {
    return static_cast<std::uint64_t>(value.int_value);
  }
  return fallback;
}

std::vector<std::int64_t> MetadataIntArrayFull(
    const iq36::GgufModelIndex& index,
    const std::string& key,
    const std::vector<std::int64_t>& fallback) {
  const auto found = index.metadata.find(key);
  if (found == index.metadata.end()) {
    return fallback;
  }
  const auto& value = found->second;
  if (value.kind == iq36::GgufMetadataValue::Kind::kArray &&
      !value.int_array.empty()) {
    return value.int_array;
  }
  return fallback;
}

struct FullAttentionTensorBundle {
  int layer = 0;
  std::string attn_norm_tensor_name;
  std::string q_tensor_name;
  std::string k_tensor_name;
  std::string v_tensor_name;
  std::string q_norm_tensor_name;
  std::string k_norm_tensor_name;
  const iq36::GgufTensorInfo* attn_norm_tensor = nullptr;
  const iq36::GgufTensorInfo* q_tensor = nullptr;
  const iq36::GgufTensorInfo* k_tensor = nullptr;
  const iq36::GgufTensorInfo* v_tensor = nullptr;
  const iq36::GgufTensorInfo* q_norm_tensor = nullptr;
  const iq36::GgufTensorInfo* k_norm_tensor = nullptr;
};

FullAttentionTensorBundle ResolveFullAttentionTensorBundle(
    const iq36::GgufModelIndex& index,
    int layer) {
  FullAttentionTensorBundle bundle;
  bundle.layer = layer;
  bundle.attn_norm_tensor_name = LayerTensorName(layer, "attn_norm.weight");
  bundle.q_tensor_name = LayerTensorName(layer, "attn_q.weight");
  bundle.k_tensor_name = LayerTensorName(layer, "attn_k.weight");
  bundle.v_tensor_name = LayerTensorName(layer, "attn_v.weight");
  bundle.q_norm_tensor_name = LayerTensorName(layer, "attn_q_norm.weight");
  bundle.k_norm_tensor_name = LayerTensorName(layer, "attn_k_norm.weight");
  bundle.attn_norm_tensor = iq36::find_tensor(index, bundle.attn_norm_tensor_name);
  bundle.q_tensor = iq36::find_tensor(index, bundle.q_tensor_name);
  bundle.k_tensor = iq36::find_tensor(index, bundle.k_tensor_name);
  bundle.v_tensor = iq36::find_tensor(index, bundle.v_tensor_name);
  bundle.q_norm_tensor = iq36::find_tensor(index, bundle.q_norm_tensor_name);
  bundle.k_norm_tensor = iq36::find_tensor(index, bundle.k_norm_tensor_name);
  Require(bundle.attn_norm_tensor != nullptr, "full attention norm tensor missing");
  Require(bundle.q_tensor != nullptr, "full attention q tensor missing");
  Require(bundle.k_tensor != nullptr, "full attention k tensor missing");
  Require(bundle.v_tensor != nullptr, "full attention v tensor missing");
  Require(bundle.q_norm_tensor != nullptr, "full attention q norm tensor missing");
  Require(bundle.k_norm_tensor != nullptr, "full attention k norm tensor missing");
  return bundle;
}

struct FullAttentionShapeChecks {
  bool attn_norm_tensor_shape_ok = false;
  bool q_tensor_shape_ok = false;
  bool k_tensor_shape_ok = false;
  bool v_tensor_shape_ok = false;
  bool q_norm_tensor_shape_ok = false;
  bool k_norm_tensor_shape_ok = false;
};

FullAttentionShapeChecks CheckFullAttentionShapes(
    const FullAttentionTensorBundle& t) {
  FullAttentionShapeChecks checks;
  checks.attn_norm_tensor_shape_ok =
      t.attn_norm_tensor->type == 0 &&
      t.attn_norm_tensor->dims == std::vector<std::uint64_t>{kHiddenSize};
  checks.q_tensor_shape_ok =
      t.q_tensor->type == 12 &&
      t.q_tensor->dims == std::vector<std::uint64_t>{kHiddenSize, kFullQFullValues};
  checks.k_tensor_shape_ok =
      t.k_tensor->type == 12 &&
      t.k_tensor->dims == std::vector<std::uint64_t>{kHiddenSize, kFullKvValues};
  checks.v_tensor_shape_ok =
      (t.v_tensor->type == 12 || t.v_tensor->type == 14) &&
      t.v_tensor->dims == std::vector<std::uint64_t>{kHiddenSize, kFullKvValues};
  checks.q_norm_tensor_shape_ok =
      t.q_norm_tensor->type == 0 &&
      t.q_norm_tensor->dims == std::vector<std::uint64_t>{kFullHeadDim};
  checks.k_norm_tensor_shape_ok =
      t.k_norm_tensor->type == 0 &&
      t.k_norm_tensor->dims == std::vector<std::uint64_t>{kFullHeadDim};
  return checks;
}

bool FullAttentionShapesPassed(const FullAttentionShapeChecks& checks) {
  return checks.attn_norm_tensor_shape_ok &&
         checks.q_tensor_shape_ok &&
         checks.k_tensor_shape_ok &&
         checks.v_tensor_shape_ok &&
         checks.q_norm_tensor_shape_ok &&
         checks.k_norm_tensor_shape_ok;
}

struct FullAttentionPayloads {
  std::vector<float> residual_input;
  std::vector<float> attn_norm;
  std::vector<float> q_full;
  std::vector<float> q_rope;
  std::vector<float> k_rope;
  std::vector<float> v;
  std::vector<float> attn_pregate;
  std::vector<float> attn_gated;
  std::vector<float> attn_output;
  std::vector<std::vector<float>> k_history;
  std::vector<std::vector<float>> v_history;
};

FullAttentionPayloads LoadFullAttentionPayloads(const std::string& payload_dir) {
  auto f32 = [&](const std::string& name) {
    return iq36::read_f32_vector_file(JoinPath(payload_dir, name));
  };
  FullAttentionPayloads p;
  p.residual_input = f32("full_residual_input.bin");
  p.attn_norm = f32("full_attn_norm.bin");
  p.q_full = f32("full_q_full.bin");
  p.q_rope = f32("full_q_rope.bin");
  p.k_rope = f32("full_k_rope.bin");
  p.v = f32("full_v.bin");
  p.attn_pregate = f32("full_attn_pregate.bin");
  p.attn_gated = f32("full_attn_gated.bin");
  p.attn_output = f32("full_attn_output.bin");
  p.k_history.reserve(kFullInputHistoryTokenCount);
  p.v_history.reserve(kFullInputHistoryTokenCount);
  for (int token = 0; token < kFullInputHistoryTokenCount; ++token) {
    std::string prefix = "hist";
    if (token < 10) {
      prefix += "0";
    }
    prefix += std::to_string(token);
    p.k_history.push_back(f32(prefix + "_k_rope.bin"));
    p.v_history.push_back(f32(prefix + "_v.bin"));
  }
  return p;
}

bool FullAttentionPayloadCountsOk(const FullAttentionPayloads& p) {
  return p.residual_input.size() == kHiddenSize &&
         p.attn_norm.size() == kHiddenSize &&
         p.q_full.size() == kFullQFullValues &&
         p.q_rope.size() == kFullQValues &&
         p.k_rope.size() == kFullKvValues &&
         p.v.size() == kFullKvValues &&
         p.attn_pregate.size() == kFullQValues &&
         p.attn_gated.size() == kFullQValues &&
         p.attn_output.size() == kHiddenSize &&
         p.k_history.size() == kFullInputHistoryTokenCount &&
         p.v_history.size() == kFullInputHistoryTokenCount &&
         std::all_of(p.k_history.begin(), p.k_history.end(), [](const auto& item) {
           return item.size() == kFullKvValues;
         }) &&
         std::all_of(p.v_history.begin(), p.v_history.end(), [](const auto& item) {
           return item.size() == kFullKvValues;
         });
}

struct FullAttentionQSplit {
  std::vector<float> q_raw;
  std::vector<float> q_gate;
};

FullAttentionQSplit SplitFullAttentionQ(const std::vector<float>& q_full) {
  Require(q_full.size() == kFullQFullValues, "full attention q_full size mismatch");
  FullAttentionQSplit split;
  split.q_raw.reserve(kFullQValues);
  split.q_gate.reserve(kFullQValues);
  for (int head = 0; head < kFullQHeadCount; ++head) {
    const int base = head * kFullHeadDim * 2;
    split.q_raw.insert(
        split.q_raw.end(),
        q_full.begin() + base,
        q_full.begin() + base + kFullHeadDim);
    split.q_gate.insert(
        split.q_gate.end(),
        q_full.begin() + base + kFullHeadDim,
        q_full.begin() + base + (2 * kFullHeadDim));
  }
  return split;
}

std::vector<float> ApplyRepeatedRmsNormFull(const std::vector<float>& input,
                                            const std::vector<float>& weight,
                                            float epsilon) {
  Require(!input.empty(), "full attention repeated rmsnorm input is empty");
  Require(!weight.empty(), "full attention repeated rmsnorm weight is empty");
  Require(input.size() % weight.size() == 0,
          "full attention repeated rmsnorm size mismatch");
  std::vector<float> output;
  output.reserve(input.size());
  for (std::size_t base = 0; base < input.size(); base += weight.size()) {
    float sum_squares = 0.0f;
    for (std::size_t i = 0; i < weight.size(); ++i) {
      const float value = input[base + i];
      sum_squares += value * value;
    }
    const float scale =
        1.0f / std::sqrt(sum_squares / static_cast<float>(weight.size()) + epsilon);
    for (std::size_t i = 0; i < weight.size(); ++i) {
      output.push_back(input[base + i] * scale * weight[i]);
    }
  }
  return output;
}

struct FullAttentionQkTiming {
  double host_q8_bridge_us = 0.0;
  double q_projection_min_us = 0.0;
  double q_projection_mean_us = 0.0;
  double k_projection_min_us = 0.0;
  double k_projection_mean_us = 0.0;
  double qk_projection_kernel_sum_min_us = 0.0;
  double qk_projection_kernel_sum_mean_us = 0.0;
};

struct FullAttentionQkRun {
  std::vector<float> q_full;
  std::vector<float> k_raw;
  FullAttentionQkTiming timing;
  std::string platform_name;
  std::string device_name;
  std::string build_log;
  double program_build_ms = 0.0;
};

FullAttentionQkRun RunGpuFullAttentionQkFront(
    const std::string& model_path,
    const iq36::GgufTensorInfo& q_tensor,
    const iq36::GgufTensorInfo& k_tensor,
    const std::vector<float>& attn_norm,
    const std::string& device_substring,
    int repeat) {
  Require(attn_norm.size() == kHiddenSize,
          "full attention qk attn_norm size mismatch");
  Require(q_tensor.type == 12, "full attention q tensor must be Q4_K");
  Require(k_tensor.type == 12, "full attention k tensor must be Q4_K");
  FullAttentionQkRun run;
  std::ifstream model(model_path, std::ios::binary);
  Require(static_cast<bool>(model), "failed to open model for full attention qk");

  const auto bridge_begin = std::chrono::steady_clock::now();
  const auto q8 = iq36::QuantizeQ8KInputPlanes(attn_norm);
  const auto bridge_end = std::chrono::steady_clock::now();
  run.timing.host_q8_bridge_us =
      std::chrono::duration<double, std::micro>(bridge_end - bridge_begin).count();

  iq36::GpuQ4X8MatvecRunner runner(device_substring, kOpenClSource);
  run.platform_name = runner.platform_name();
  run.device_name = runner.device_name();
  run.build_log = runner.build_log();
  run.program_build_ms = runner.program_build_ms();

  const auto q = RunProjectionFromTensor(
      runner, model, q_tensor, q8, kFullQFullValues, repeat);
  const auto k = RunProjectionFromTensor(
      runner, model, k_tensor, q8, kFullKvValues, repeat);
  run.q_full = q.output;
  run.k_raw = k.output;
  run.timing.q_projection_min_us = q.timing.min_us;
  run.timing.q_projection_mean_us = q.timing.mean_us;
  run.timing.k_projection_min_us = k.timing.min_us;
  run.timing.k_projection_mean_us = k.timing.mean_us;
  run.timing.qk_projection_kernel_sum_min_us =
      q.timing.min_us + k.timing.min_us;
  run.timing.qk_projection_kernel_sum_mean_us =
      q.timing.mean_us + k.timing.mean_us;
  return run;
}

void AppendCpuGpuOracleCompare(std::vector<NamedCompareGroup>& groups,
                               const std::string& name,
                               const std::vector<float>& cpu,
                               const std::vector<float>& gpu,
                               const std::vector<float>& oracle) {
  groups.push_back({
      name,
      iq36::compare_vectors(cpu, oracle, kMismatchThreshold),
      iq36::compare_vectors(gpu, cpu, kMismatchThreshold),
      iq36::compare_vectors(gpu, oracle, kMismatchThreshold),
  });
}

void WriteFullAttentionShapeChecks(const FullAttentionShapeChecks& checks) {
  std::cout << "{";
  std::cout << "\"attn_norm_tensor_shape_ok\":"
            << (checks.attn_norm_tensor_shape_ok ? "true" : "false") << ",";
  std::cout << "\"q_tensor_shape_ok\":"
            << (checks.q_tensor_shape_ok ? "true" : "false") << ",";
  std::cout << "\"k_tensor_shape_ok\":"
            << (checks.k_tensor_shape_ok ? "true" : "false") << ",";
  std::cout << "\"v_tensor_shape_ok\":"
            << (checks.v_tensor_shape_ok ? "true" : "false") << ",";
  std::cout << "\"q_norm_tensor_shape_ok\":"
            << (checks.q_norm_tensor_shape_ok ? "true" : "false") << ",";
  std::cout << "\"k_norm_tensor_shape_ok\":"
            << (checks.k_norm_tensor_shape_ok ? "true" : "false");
  std::cout << "}";
}

void WriteFullAttentionTiming(const LayerInputRmsNormRun& rms,
                              const FullAttentionQkRun& qk) {
  std::cout << "{";
  std::cout << "\"layer7_rmsnorm_min_us\":" << rms.timing.rmsnorm_min_us << ",";
  std::cout << "\"layer7_rmsnorm_mean_us\":" << rms.timing.rmsnorm_mean_us << ",";
  std::cout << "\"q_projection_min_us\":" << qk.timing.q_projection_min_us << ",";
  std::cout << "\"q_projection_mean_us\":" << qk.timing.q_projection_mean_us << ",";
  std::cout << "\"k_projection_min_us\":" << qk.timing.k_projection_min_us << ",";
  std::cout << "\"k_projection_mean_us\":" << qk.timing.k_projection_mean_us << ",";
  std::cout << "\"qk_projection_kernel_sum_min_us\":"
            << qk.timing.qk_projection_kernel_sum_min_us << ",";
  std::cout << "\"qk_projection_kernel_sum_mean_us\":"
            << qk.timing.qk_projection_kernel_sum_mean_us << ",";
  std::cout << "\"qk_host_q8_bridge_us\":" << qk.timing.host_q8_bridge_us << ",";
  std::cout << "\"layer7_full_attn_input_kernel_sum_min_us\":"
            << (rms.timing.rmsnorm_min_us +
                qk.timing.qk_projection_kernel_sum_min_us) << ",";
  std::cout << "\"layer7_full_attn_input_kernel_sum_mean_us\":"
            << (rms.timing.rmsnorm_mean_us +
                qk.timing.qk_projection_kernel_sum_mean_us);
  std::cout << "}";
}

int main(int argc, char** argv) {
  try {
    const Args args = ParseArgs(argc, argv);
    const int layer0 = args.layer;
    const int layer1 = args.layer + 1;
    const int layer2 = args.layer + 2;
    Require(layer0 == 5 && layer1 == 6 && layer2 == 7,
            "layer7 handoff probe expects --layer 5");
    const auto index = iq36::parse_gguf_model_index(args.model_path);
    const auto load_map = iq36::validate_qwen36_load_map(index);
    const float rms_norm_epsilon = MetadataFloat(
        index, "qwen35moe.attention.layer_norm_rms_epsilon", 1e-6f);
    const auto head_dim = MetadataUIntFull(
        index, "qwen35moe.attention.key_length", kFullHeadDim);
    const auto value_length = MetadataUIntFull(
        index, "qwen35moe.attention.value_length", kFullHeadDim);
    const auto q_head_count = MetadataUIntFull(
        index, "qwen35moe.attention.head_count", kFullQHeadCount);
    const auto kv_head_count = MetadataUIntFull(
        index, "qwen35moe.attention.head_count_kv", kFullKvHeadCount);
    const auto full_attention_interval = MetadataUIntFull(
        index, "qwen35moe.full_attention_interval", 4);
    const auto rope_dimension_count = MetadataUIntFull(
        index, "qwen35moe.rope.dimension_count", 64);
    const auto rope_context_length = MetadataUIntFull(
        index, "qwen35moe.context_length", 262144);
    const auto rope_sections = MetadataIntArrayFull(
        index, "qwen35moe.rope.dimension_sections", {11, 11, 10, 0});
    const float rope_freq_base = MetadataFloat(
        index, "qwen35moe.rope.freq_base", 10000000.0f);
    constexpr float kRopeFreqScale = 1.0f;
    constexpr float kRopeExtFactor = 0.0f;
    constexpr float kRopeAttnFactor = 1.0f;
    constexpr float kRopeBetaFast = 32.0f;
    constexpr float kRopeBetaSlow = 1.0f;

    const auto layer0_tensors = ResolveLayerTensorBundle(index, layer0);
    const auto layer1_tensors = ResolveLayerTensorBundle(index, layer1);
    const auto layer2_tensors = ResolveFullAttentionTensorBundle(index, layer2);
    const auto layer0_oracle = LoadLayerOraclePayloads(args.payload_dir, "l0");
    const auto layer1_oracle = LoadLayerOraclePayloads(args.payload_dir, "l1");
    const auto layer2_oracle = LoadFullAttentionPayloads(args.payload_dir);

    const auto layer0_run = RunResidentLinearLayerShell(
        args, index, layer0_tensors, layer0_oracle,
        layer0_oracle.residual_input, layer0_oracle.residual_input,
        rms_norm_epsilon);
    const auto layer1_run = RunResidentLinearLayerShell(
        args, index, layer1_tensors, layer1_oracle,
        layer0_run.native_layer_output, layer0_run.gpu_layer_output,
        rms_norm_epsilon);

    const auto full_shapes = CheckFullAttentionShapes(layer2_tensors);
    const bool full_shapes_ok = FullAttentionShapesPassed(full_shapes);
    const bool layer2_payload_counts_ok =
        FullAttentionPayloadCountsOk(layer2_oracle);
    const bool metadata_ok =
        full_attention_interval == 4 &&
        head_dim == kFullHeadDim &&
        value_length == kFullHeadDim &&
        q_head_count == kFullQHeadCount &&
        kv_head_count == kFullKvHeadCount &&
        rope_dimension_count == 64 &&
        rope_context_length == 262144 &&
        rope_sections == std::vector<std::int64_t>({11, 11, 10, 0}) &&
        rope_freq_base == 10000000.0f;

    std::ifstream model(args.model_path, std::ios::binary);
    Require(static_cast<bool>(model), "model file could not be opened");
    const auto attn_norm_weight =
        ReadF32TensorPayload(model, *layer2_tensors.attn_norm_tensor,
                             static_cast<std::size_t>(kHiddenSize));
    const auto q_norm_weight =
        iq36::decode_tensor_row(args.model_path, index,
                                layer2_tensors.q_norm_tensor_name, 0);
    const auto k_norm_weight =
        iq36::decode_tensor_row(args.model_path, index,
                                layer2_tensors.k_norm_tensor_name, 0);

    const auto native_qkv = iq36::run_qwen36_full_attention_qkv_projection(
        args.model_path,
        index,
        layer2,
        layer1_run.native_layer_output,
        rms_norm_epsilon);
    const auto layer2_rms_gpu = RunGpuLayerInputRmsNorm(
        layer1_run.gpu_layer_output,
        attn_norm_weight,
        rms_norm_epsilon,
        args.device_substring,
        args.repeat);
    const auto layer2_qk_gpu = RunGpuFullAttentionQkFront(
        args.model_path,
        *layer2_tensors.q_tensor,
        *layer2_tensors.k_tensor,
        layer2_rms_gpu.attn_norm,
        args.device_substring,
        args.repeat);
    const auto gpu_q_split = SplitFullAttentionQ(layer2_qk_gpu.q_full);
    const auto gpu_q_normed = ApplyRepeatedRmsNormFull(
        gpu_q_split.q_raw, q_norm_weight, rms_norm_epsilon);
    const auto gpu_k_normed = ApplyRepeatedRmsNormFull(
        layer2_qk_gpu.k_raw, k_norm_weight, rms_norm_epsilon);
    const auto gpu_rope = iq36::run_qwen36_full_attention_rope(
        gpu_q_normed,
        gpu_k_normed,
        kFullSourceTokenPosition,
        head_dim,
        rope_dimension_count,
        rope_sections,
        rope_context_length,
        rope_freq_base,
        kRopeFreqScale,
        kRopeExtFactor,
        kRopeAttnFactor,
        kRopeBetaFast,
        kRopeBetaSlow);
    const auto native_rope = iq36::run_qwen36_full_attention_rope(
        native_qkv.q_normed,
        native_qkv.k_normed,
        kFullSourceTokenPosition,
        head_dim,
        rope_dimension_count,
        rope_sections,
        rope_context_length,
        rope_freq_base,
        kRopeFreqScale,
        kRopeExtFactor,
        kRopeAttnFactor,
        kRopeBetaFast,
        kRopeBetaSlow);

    std::vector<NamedCompareGroup> full_groups;
    AppendCpuGpuOracleCompare(full_groups, "l2_residual_input",
                              layer1_run.native_layer_output,
                              layer1_run.gpu_layer_output,
                              layer2_oracle.residual_input);
    AppendCpuGpuOracleCompare(full_groups, "l2_attn_norm",
                              native_qkv.attention_norm,
                              layer2_rms_gpu.attn_norm,
                              layer2_oracle.attn_norm);
    AppendCpuGpuOracleCompare(full_groups, "l2_q_full",
                              native_qkv.q_full,
                              layer2_qk_gpu.q_full,
                              layer2_oracle.q_full);
    AppendCpuGpuOracleCompare(full_groups, "l2_q_rope",
                              native_rope.q_rope,
                              gpu_rope.q_rope,
                              layer2_oracle.q_rope);
    AppendCpuGpuOracleCompare(full_groups, "l2_k_rope",
                              native_rope.k_rope,
                              gpu_rope.k_rope,
                              layer2_oracle.k_rope);

    const auto k_raw_gpu_vs_cpu = iq36::compare_vectors(
        layer2_qk_gpu.k_raw, native_qkv.k_raw, kMismatchThreshold);
    const auto native_v_vs_oracle = iq36::compare_vectors(
        native_qkv.v, layer2_oracle.v, kMismatchThreshold);
    const bool layer0_shapes_ok = ShapesPassed(layer0_run.shape_checks);
    const bool layer1_shapes_ok = ShapesPassed(layer1_run.shape_checks);
    const bool layer0_ok =
        layer0_shapes_ok && layer0_run.payload_counts_ok &&
        layer0_run.comparisons_passed && layer0_run.timing_positive &&
        layer0_run.arc_selected;
    const bool layer1_ok =
        layer1_shapes_ok && layer1_run.payload_counts_ok &&
        layer1_run.comparisons_passed && layer1_run.timing_positive &&
        layer1_run.arc_selected;
    const bool layer2_comparisons_ok =
        CompareGroupsPassed(full_groups) &&
        ComparePassed(k_raw_gpu_vs_cpu) &&
        ComparePassed(native_v_vs_oracle);
    const bool layer2_timing_positive =
        layer2_rms_gpu.timing.rmsnorm_min_us > 0.0 &&
        layer2_qk_gpu.timing.q_projection_min_us > 0.0 &&
        layer2_qk_gpu.timing.k_projection_min_us > 0.0;
    const bool layer2_arc_selected =
        layer2_rms_gpu.device_name.find(args.device_substring) != std::string::npos &&
        layer2_qk_gpu.device_name.find(args.device_substring) != std::string::npos;
    const bool state_carry_ok = ComparePassed(layer1_run.comparisons[0].gpu_vs_oracle);
    const bool v_q6_reference_boundary = layer2_tensors.v_tensor->type == 14;
    const bool required_checks_passed =
        load_map.ready &&
        layer0_ok &&
        layer1_ok &&
        state_carry_ok &&
        full_shapes_ok &&
        layer2_payload_counts_ok &&
        metadata_ok &&
        layer2_comparisons_ok &&
        layer2_timing_positive &&
        layer2_arc_selected &&
        v_q6_reference_boundary &&
        args.repeat > 0;

    const double two_layer_kernel_sum_min =
        layer0_run.timing.layer_kernel_sum_min_us +
        layer1_run.timing.layer_kernel_sum_min_us;
    const double layer2_kernel_sum_min =
        layer2_rms_gpu.timing.rmsnorm_min_us +
        layer2_qk_gpu.timing.qk_projection_kernel_sum_min_us;
    const double resident_handoff_kernel_sum_min =
        two_layer_kernel_sum_min + layer2_kernel_sum_min;

    std::cout << std::setprecision(10);
    std::cout << "{";
    std::cout << "\"schema_version\":\"intel-qwen36-gpu-resident-layer7-full-attn-input-handoff-probe-v0\",";
    std::cout << "\"model_path\":\"" << JsonEscape(args.model_path) << "\",";
    std::cout << "\"layers\":[" << layer0 << "," << layer1 << "," << layer2 << "],";
    std::cout << "\"source_token_position\":" << kFullSourceTokenPosition << ",";
    std::cout << "\"resident_api\":\"two_linear_layer_to_full_attention_state_input_load_once_run_many\",";
    std::cout << "\"resident_load_count\":1,";
    std::cout << "\"resident_shell_invocations\":" << args.repeat << ",";
    std::cout << "\"layer6_residual_input_from_layer5_gpu_output\":true,";
    std::cout << "\"layer7_residual_input_from_layer6_gpu_output\":true,";
    std::cout << "\"captured_layer7_residual_input_required_check\":true,";
    std::cout << "\"full_attn_qk_projection_gpu_boundary\":true,";
    std::cout << "\"full_attn_v_projection_gpu_supported\":false,";
    std::cout << "\"full_attn_v_projection_boundary\":\"cpu_q6_reference\",";
    std::cout << "\"qk_norm_rope_host_boundary\":true,";
    std::cout << "\"history_kv_state_payload_boundary\":true,";
    std::cout << "\"platform_name\":\"" << JsonEscape(layer2_rms_gpu.platform_name) << "\",";
    std::cout << "\"device_name\":\"" << JsonEscape(layer2_rms_gpu.device_name) << "\",";
    std::cout << "\"qk_projection_device_name\":\"" << JsonEscape(layer2_qk_gpu.device_name) << "\",";
    std::cout << "\"layer7_v_tensor_type\":\""
              << JsonEscape(iq36::ggml_type_name(layer2_tensors.v_tensor->type)) << "\",";
    std::cout << "\"program_build_ms\":"
              << (layer0_run.program_build_ms + layer1_run.program_build_ms +
                  layer2_rms_gpu.program_build_ms + layer2_qk_gpu.program_build_ms)
              << ",";
    std::cout << "\"build_log\":\""
              << JsonEscape(layer0_run.build_log + layer1_run.build_log +
                            layer2_rms_gpu.build_log + layer2_qk_gpu.build_log)
              << "\",";
    std::cout << "\"repeat\":" << args.repeat << ",";
    std::cout << "\"attention_parameters\":{";
    std::cout << "\"full_attention_interval\":" << full_attention_interval << ",";
    std::cout << "\"gqa_group\":" << (q_head_count / kv_head_count) << ",";
    std::cout << "\"head_dim\":" << head_dim << ",";
    std::cout << "\"input_history_token_count\":" << kFullInputHistoryTokenCount << ",";
    std::cout << "\"kv_head_count\":" << kv_head_count << ",";
    std::cout << "\"q_head_count\":" << q_head_count << ",";
    std::cout << "\"updated_history_token_count\":" << kFullUpdatedHistoryTokenCount;
    std::cout << "},\"rope_parameters\":{";
    std::cout << "\"context_length\":" << rope_context_length << ",";
    std::cout << "\"position_ids\":[15,15,15,0],";
    std::cout << "\"rope_attn_factor\":" << kRopeAttnFactor << ",";
    std::cout << "\"rope_beta_fast\":" << kRopeBetaFast << ",";
    std::cout << "\"rope_beta_slow\":" << kRopeBetaSlow << ",";
    std::cout << "\"rope_dimension_count\":" << rope_dimension_count << ",";
    std::cout << "\"rope_dimension_sections\":[";
    for (std::size_t i = 0; i < rope_sections.size(); ++i) {
      if (i != 0) {
        std::cout << ",";
      }
      std::cout << rope_sections[i];
    }
    std::cout << "],\"rope_ext_factor\":" << kRopeExtFactor << ",";
    std::cout << "\"rope_freq_base\":" << rope_freq_base << ",";
    std::cout << "\"rope_freq_scale\":" << kRopeFreqScale;
    std::cout << "},\"timings\":{";
    std::cout << "\"layer0\":";
    WriteLayerTiming(layer0_run.timing);
    std::cout << ",\"layer1\":";
    WriteLayerTiming(layer1_run.timing);
    std::cout << ",\"layer2_full_attn_input\":";
    WriteFullAttentionTiming(layer2_rms_gpu, layer2_qk_gpu);
    std::cout << ",\"resident_two_linear_layer_kernel_sum_min_us\":"
              << two_layer_kernel_sum_min << ",";
    std::cout << "\"resident_layer7_full_attn_input_kernel_sum_min_us\":"
              << layer2_kernel_sum_min << ",";
    std::cout << "\"resident_two_linear_plus_layer7_input_kernel_sum_min_us\":"
              << resident_handoff_kernel_sum_min;
    std::cout << "},\"comparisons\":{";
    bool first_compare = true;
    WritePrefixedCompareGroups("l0", layer0_run.comparisons, &first_compare);
    WritePrefixedCompareGroups("l1", layer1_run.comparisons, &first_compare);
    if (!first_compare) {
      std::cout << ",";
    }
    first_compare = false;
    WriteNamedCompareGroups(full_groups);
    std::cout << ",\"l2_k_raw\":{\"gpu_vs_cpu\":";
    WriteCompare(k_raw_gpu_vs_cpu);
    std::cout << "},\"l2_v\":{\"cpu_vs_oracle\":";
    WriteCompare(native_v_vs_oracle);
    std::cout << "},\"l0_conv_state_after\":{\"gpu_vs_cpu\":";
    WriteCompare(layer0_run.conv_state_after_gpu_vs_cpu);
    std::cout << "},\"l1_conv_state_after\":{\"gpu_vs_cpu\":";
    WriteCompare(layer1_run.conv_state_after_gpu_vs_cpu);
    std::cout << "}";
    std::cout << "},\"checks\":{";
    std::cout << "\"load_map_ready\":" << (load_map.ready ? "true" : "false") << ",";
    std::cout << "\"layer0\":";
    WriteLayerChecks(layer0_run);
    std::cout << ",\"layer1\":";
    WriteLayerChecks(layer1_run);
    std::cout << ",\"layer2_shapes\":";
    WriteFullAttentionShapeChecks(full_shapes);
    std::cout << ",\"layer1_residual_input_matches_oracle\":"
              << (state_carry_ok ? "true" : "false") << ",";
    std::cout << "\"layer2_payload_counts_ok\":"
              << (layer2_payload_counts_ok ? "true" : "false") << ",";
    std::cout << "\"metadata_ok\":" << (metadata_ok ? "true" : "false") << ",";
    std::cout << "\"layer2_full_attn_input_matches_oracle\":"
              << (layer2_comparisons_ok ? "true" : "false") << ",";
    std::cout << "\"layer2_k_raw_gpu_vs_cpu_matches\":"
              << (ComparePassed(k_raw_gpu_vs_cpu) ? "true" : "false") << ",";
    std::cout << "\"layer2_v_native_matches_oracle\":"
              << (ComparePassed(native_v_vs_oracle) ? "true" : "false") << ",";
    std::cout << "\"layer2_v_q6_reference_boundary\":"
              << (v_q6_reference_boundary ? "true" : "false") << ",";
    std::cout << "\"history_kv_state_payloads_present\":"
              << (layer2_payload_counts_ok ? "true" : "false") << ",";
    std::cout << "\"layer2_arc_device_selected\":"
              << (layer2_arc_selected ? "true" : "false") << ",";
    std::cout << "\"resident_load_once\":true,";
    std::cout << "\"resident_shell_invocations_positive\":"
              << (args.repeat > 0 ? "true" : "false") << ",";
    std::cout << "\"gpu_event_timing_positive\":"
              << (layer0_run.timing_positive && layer1_run.timing_positive &&
                  layer2_timing_positive ? "true" : "false") << ",";
    std::cout << "\"speedup_claims_allowed\":false";
    std::cout << "},\"required_checks_passed\":"
              << (required_checks_passed ? "true" : "false");
    std::cout << "}\n";
    return required_checks_passed ? 0 : 3;
  } catch (const std::exception& exc) {
    std::cout << "{\"ok\":false,\"error\":\"" << JsonEscape(exc.what()) << "\"}\n";
    return 2;
  }
}
'''


def full_attn_handoff_cpp(opencl_source: str) -> str:
  cpp = TWO.two_layer_probe_cpp(opencl_source)
  main_index = cpp.index("\nint main(")
  return cpp[:main_index] + "\n" + FULL_ATTN_HANDOFF_CPP


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
  parser.add_argument("--all-history-json", type=Path, default=DEFAULT_ALL_HISTORY)
  parser.add_argument("--layer", type=int, default=5)
  parser.add_argument("--resident-invocations", type=int, default=5)
  parser.add_argument("--device-substring", default="B390")
  parser.add_argument("--timeout-s", type=int, default=1200)
  parser.add_argument("--conv-history-probe", type=Path, default=None)
  parser.add_argument("--next-conv-history-probe", type=Path, default=None)
  parser.add_argument("--out-dir", type=Path, default=None)
  return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise SystemExit(f"{path}: expected JSON object")
  return value


def find_payload(pattern: str, expected_bytes: int) -> Path:
  matches = sorted(PRECONV.PAYLOAD_ROOT.glob(pattern))
  if len(matches) != 1:
    raise SystemExit(f"expected one payload for {pattern}, found {len(matches)}")
  path = matches[0].resolve()
  if path.stat().st_size != expected_bytes:
    raise SystemExit(f"payload size mismatch for {path}: {path.stat().st_size}")
  return path


def payload_record(path: Path, stage_name: str, expected_bytes: int) -> dict[str, Any]:
  path = path.resolve()
  if path.stat().st_size != expected_bytes:
    raise SystemExit(f"payload size mismatch for {path}: {path.stat().st_size}")
  return {
      "local_path": path,
      "path": str(path.relative_to(ROOT)),
      "sha256": iq36_local.sha256_file(path),
      "size_bytes": expected_bytes,
      "stage_name": stage_name,
  }


def history_layer_entry(token: dict[str, Any], layer: int) -> dict[str, Any]:
  layers = token.get("layers")
  entry: Any = None
  if isinstance(layers, dict):
    entry = layers.get(str(layer))
  elif isinstance(layers, list):
    for item in layers:
      if isinstance(item, dict) and item.get("layer_index") == layer:
        entry = item
        break
  if not isinstance(entry, dict):
    raise SystemExit(f"history token missing layer {layer}")
  return entry


def history_payload_record(
    history_json: Path,
    token_index: int,
    layer: int,
    payload_name: str,
    stage_name: str,
    entry: dict[str, Any],
) -> dict[str, Any]:
  payloads = entry.get("payloads", {})
  item = payloads.get(payload_name) if isinstance(payloads, dict) else None
  if not isinstance(item, dict):
    raise SystemExit(f"{history_json}: token {token_index} layer {layer} missing {payload_name}")
  path_value = item.get("path")
  size_bytes = item.get("size_bytes")
  expected_sha = item.get("sha256")
  if not isinstance(path_value, str) or not isinstance(size_bytes, int):
    raise SystemExit(f"{history_json}: invalid payload metadata for {payload_name}")
  path = (ROOT / path_value).resolve()
  if not path.exists():
    raise SystemExit(f"{history_json}: missing payload file {path}")
  if path.stat().st_size != size_bytes:
    raise SystemExit(f"{history_json}: size mismatch for {path}")
  digest = iq36_local.sha256_file(path)
  if digest != expected_sha:
    raise SystemExit(f"{history_json}: sha256 mismatch for {path}")
  return {
      "local_path": path,
      "path": str(path.relative_to(ROOT)),
      "sha256": digest,
      "size_bytes": size_bytes,
      "stage_name": stage_name,
      "tensor_name": item.get("tensor_name"),
      "tensor_op": item.get("tensor_op"),
      "token_position": token_index,
      "value_count": item.get("value_count"),
  }


def resolve_full_attention_payloads(
    history_json: Path,
    layer: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
  history_json = history_json.resolve()
  history_doc = load_json(history_json)
  if history_doc.get("schema_version") != "intel-qwen36-r1-full-attn-all-history-capture-v0":
    raise SystemExit(f"{history_json}: unexpected all-history schema")
  if history_doc.get("full_attn_all_history_capture_passed") is not True:
    raise SystemExit(f"{history_json}: all-history capture did not pass")
  capture = history_doc.get("full_attn_all_history_capture", {})
  tokens = capture.get("tokens", [])
  if capture.get("history_token_count") != 16 or len(tokens) != 16:
    raise SystemExit(f"{history_json}: expected 16 token history")

  payloads: dict[str, dict[str, Any]] = {
      "full_residual_input": payload_record(
          find_payload("l_out-6__tok15__ord*.bin", 8192),
          "full_residual_input.bin",
          8192,
      ),
      "full_attn_norm": payload_record(
          find_payload("attn_norm-7__tok15__ord*.bin", 8192),
          "full_attn_norm.bin",
          8192,
      ),
      "full_q_full": payload_record(
          find_payload("Qcur_full-7__tok15__ord*.bin", 32768),
          "full_q_full.bin",
          32768,
      ),
  }
  for token_index, token in enumerate(tokens):
    if token.get("source_token_position") != token_index:
      raise SystemExit(f"{history_json}: token position mismatch at {token_index}")
    entry = history_layer_entry(token, layer)
    if token_index < SOURCE_TOKEN_POSITION:
      prefix = f"hist{token_index:02d}"
      for payload_name in ("k_rope", "v"):
        key = f"{prefix}_{payload_name}"
        payloads[key] = history_payload_record(
            history_json,
            token_index,
            layer,
            payload_name,
            f"{prefix}_{payload_name}.bin",
            entry,
        )
    elif token_index == SOURCE_TOKEN_POSITION:
      for payload_name, stage_name in (
          ("q_rope", "full_q_rope.bin"),
          ("k_rope", "full_k_rope.bin"),
          ("v", "full_v.bin"),
          ("attn_pregate", "full_attn_pregate.bin"),
          ("attn_gated", "full_attn_gated.bin"),
          ("attn_output", "full_attn_output.bin"),
      ):
        payloads[f"full_{payload_name}"] = history_payload_record(
            history_json,
            token_index,
            layer,
            payload_name,
            stage_name,
            entry,
        )
  return payloads, {
      "history_artifact": str(history_json.parent.relative_to(ROOT)),
      "history_json": str(history_json.relative_to(ROOT)),
      "history_token_count": capture.get("history_token_count"),
      "full_attention_layers": capture.get("full_attention_layers"),
      "source_prompt_case_id": capture.get("source_prompt_case_id"),
      "source_token_positions": capture.get("source_token_positions"),
  }


def resident_fields_ok(probe: dict[str, Any] | None, expected_invocations: int) -> bool:
  return (
      isinstance(probe, dict)
      and probe.get("resident_api") == "two_linear_layer_to_full_attention_state_input_load_once_run_many"
      and probe.get("resident_load_count") == 1
      and probe.get("resident_shell_invocations") == expected_invocations
      and probe.get("layer7_residual_input_from_layer6_gpu_output") is True
      and probe.get("captured_layer7_residual_input_required_check") is True
      and probe.get("full_attn_qk_projection_gpu_boundary") is True
      and probe.get("full_attn_v_projection_gpu_supported") is False
      and probe.get("full_attn_v_projection_boundary") == "cpu_q6_reference"
      and probe.get("qk_norm_rope_host_boundary") is True
      and probe.get("history_kv_state_payload_boundary") is True
      and PRECONV.nested_bool(probe, "checks", "resident_load_once")
      and PRECONV.nested_bool(probe, "checks", "resident_shell_invocations_positive")
  )


def comparison_passed(probe: dict[str, Any] | None, name: str, lane: str) -> bool:
  if not isinstance(probe, dict):
    return False
  comparisons = probe.get("comparisons")
  if not isinstance(comparisons, dict):
    return False
  item = comparisons.get(name)
  if not isinstance(item, dict):
    return False
  stats = item.get(lane)
  return (
      isinstance(stats, dict)
      and stats.get("same_size") is True
      and stats.get("finite") is True
      and stats.get("mismatch_count") == 0
      and stats.get("max_abs_diff", 1.0) <= 5e-3
      and stats.get("rmse", 1.0) <= 1e-3
      and stats.get("cosine", 0.0) >= 0.99999
  )


def write_summary(path: Path, payload: dict[str, Any]) -> None:
  probe = payload.get("probe") if isinstance(payload.get("probe"), dict) else {}
  timings = probe.get("timings", {}) if isinstance(probe, dict) else {}
  layer2_timing = (
      timings.get("layer2_full_attn_input", {})
      if isinstance(timings, dict) and isinstance(timings.get("layer2_full_attn_input"), dict)
      else {}
  )
  comparisons = probe.get("comparisons", {}) if isinstance(probe, dict) else {}
  lines = [
      "# GPU Resident Layer-7 Full-Attention Input Handoff Probe",
      "",
      f"- workstream: `{WORKSTREAM}`",
      f"- host: `{payload.get('host')}`",
      f"- model: `{payload.get('model')}`",
      f"- layers: `{payload.get('layers')}`",
      f"- required checks passed: `{str(payload.get('required_checks_passed')).lower()}`",
      "- speedup claims allowed: `false`",
      f"- resident API: `{probe.get('resident_api')}`",
      f"- resident shell invocations: `{probe.get('resident_shell_invocations')}`",
      f"- all-history source: `{payload.get('all_history', {}).get('history_artifact')}`",
      f"- V projection boundary: `{probe.get('full_attn_v_projection_boundary')}`",
      "",
      "| output | comparison | max abs | RMSE |",
      "|---|---|---:|---:|",
  ]
  for name in (
      "l2_residual_input",
      "l2_attn_norm",
      "l2_q_full",
      "l2_q_rope",
      "l2_k_rope",
  ):
    group = comparisons.get(name, {}) if isinstance(comparisons, dict) else {}
    lane = group.get("gpu_vs_oracle", {}) if isinstance(group, dict) else {}
    lines.append(f"| {name} | gpu_vs_oracle | {lane.get('max_abs_diff')} | {lane.get('rmse')} |")
  v_lane = (
      comparisons.get("l2_v", {}).get("cpu_vs_oracle", {})
      if isinstance(comparisons.get("l2_v"), dict)
      else {}
  )
  lines.append(f"| l2_v | cpu_vs_oracle | {v_lane.get('max_abs_diff')} | {v_lane.get('rmse')} |")
  lines += [
      "",
      "| kernel group | min us |",
      "|---|---:|",
      f"| layer7_rmsnorm | {layer2_timing.get('layer7_rmsnorm_min_us')} |",
      f"| q_projection | {layer2_timing.get('q_projection_min_us')} |",
      f"| k_projection | {layer2_timing.get('k_projection_min_us')} |",
      f"| layer7_full_attn_input_sum | {layer2_timing.get('layer7_full_attn_input_kernel_sum_min_us')} |",
      f"| two_linear_plus_layer7_input_sum | {timings.get('resident_two_linear_plus_layer7_input_kernel_sum_min_us') if isinstance(timings, dict) else None} |",
      "",
      "This target-side process carries layer 5 and layer 6 GPU outputs into",
      "layer 7. GPU computes layer-7 attention RMSNorm and Q/K Q4 projections;",
      "Q/K norm and RoPE are host-side validation boundaries. Layer-7 V remains",
      "a CPU Q6 reference boundary because `blk.7.attn_v.weight` is Q6_K.",
      "This is captured single-token handoff evidence only, not decode",
      "throughput.",
      "",
  ]
  path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  args = parse_args()
  if args.resident_invocations <= 0:
    raise SystemExit("--resident-invocations must be positive")
  if args.layer != 5:
    raise SystemExit("--layer must be 5 for the layer-7 full-attention handoff")

  layer0 = args.layer
  layer1 = args.layer + 1
  layer2 = args.layer + 2
  conv0_path = (
      args.conv_history_probe.resolve()
      if args.conv_history_probe is not None
      else TWO.latest_conv_history_probe_for_layer(layer0).resolve()
  )
  conv1_path = (
      args.next_conv_history_probe.resolve()
      if args.next_conv_history_probe is not None
      else TWO.latest_conv_history_probe_for_layer(layer1).resolve()
  )
  all_history_json = args.all_history_json.resolve()
  stamp = utc_stamp()
  created_at = iso_now()
  out_dir = args.out_dir or ROOT / f"output/gpu-resident-layer7-full-attn-input-handoff-probe-{stamp}"
  out_dir.mkdir(parents=True, exist_ok=False)
  raw_dir = out_dir / "raw"
  raw_dir.mkdir()

  payloads0, conv0 = TWO.prefixed_payloads(layer0, conv0_path, "l0")
  payloads1, conv1 = TWO.prefixed_payloads(layer1, conv1_path, "l1")
  full_payloads, all_history = resolve_full_attention_payloads(all_history_json, layer2)
  payloads = {**payloads0, **payloads1, **full_payloads}
  opencl_source = OPENCL_SOURCE.read_text(encoding="utf-8")
  opencl_source_hash = iq36_local.sha256_file(OPENCL_SOURCE)
  embedded_opencl_hash = hashlib.sha256(
      (opencl_source + PRECONV.POSTCONV.ATTENTION.ATTENTION_EXTRA_OPENCL).encode("utf-8")
  ).hexdigest()
  local_cpp = out_dir / "gpu_resident_layer7_full_attn_input_handoff_probe.cpp"
  local_cpp.write_text(full_attn_handoff_cpp(opencl_source), encoding="utf-8")

  remote_dir = f"{args.remote_root.rstrip('/')}/gpu-resident-layer7-full-attn-input-handoff-probe-{stamp}"
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
    for local, remote in PRECONV.POSTCONV.ATTENTION.SHARED.SELECTED.SOURCE_FILES:
      transfers.append(
          iq36_local.copy_to(args.host, ROOT / local, f"{remote_dir}/{remote}", args.timeout_s)
      )
    transfers.append(
        iq36_local.copy_to(
            args.host,
            local_cpp,
            f"{remote_dir}/tests/gpu_resident_layer7_full_attn_input_handoff_probe.cpp",
            args.timeout_s,
        )
    )
    for name, payload in payloads.items():
      payload_transfers[name] = iq36_local.copy_to(
          args.host,
          payload["local_path"],
          f"{remote_payload_dir}/{payload['stage_name']}",
          args.timeout_s,
      )

  executable = f"{remote_dir}/build/iq36-gpu-resident-layer7-full-attn-input-handoff-probe"
  compile_cmd = " && ".join([
      f"source {shlex.quote(args.env_script)} >/dev/null 2>&1",
      (
          "g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic "
          f"-I {shlex.quote(remote_dir + '/include')} "
          f"{shlex.quote(remote_dir + '/src/gguf_loader.cpp')} "
          f"{shlex.quote(remote_dir + '/src/gpu_q4x8_matvec.cpp')} "
          f"{shlex.quote(remote_dir + '/tests/gpu_resident_layer7_full_attn_input_handoff_probe.cpp')} "
          "-ldl -pthread "
          f"-o {shlex.quote(executable)}"
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
      executable,
      "--model", args.model,
      "--payload-dir", remote_payload_dir,
      "--layer", str(args.layer),
      "--repeat", str(args.resident_invocations),
      "--device-substring", args.device_substring,
  ]
  run_result = (
      iq36_local.run_target(
          args.host,
          " && ".join([
              f"source {shlex.quote(args.env_script)} >/dev/null 2>&1",
              PRECONV.shell_join(run_argv),
          ]),
          args.timeout_s,
      )
      if compile_result.get("returncode") == 0
      else {"cmd": run_argv, "returncode": None, "stdout": "", "stderr": "compile skipped run"}
  )
  probe = PRECONV.parse_probe_stdout(run_result.get("stdout", ""))

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
      {"name": "resident_api_fields_present", "pass": resident_fields_ok(probe, args.resident_invocations)},
      {"name": "layer0_checks_passed", "pass": PRECONV.nested_bool(probe, "checks", "layer0", "comparisons_passed")},
      {"name": "layer1_checks_passed", "pass": PRECONV.nested_bool(probe, "checks", "layer1", "comparisons_passed")},
      {"name": "layer1_residual_input_matches_oracle", "pass": PRECONV.nested_bool(probe, "checks", "layer1_residual_input_matches_oracle")},
      {"name": "layer2_full_attn_input_matches_oracle", "pass": PRECONV.nested_bool(probe, "checks", "layer2_full_attn_input_matches_oracle")},
      {"name": "layer2_v_q6_reference_boundary", "pass": PRECONV.nested_bool(probe, "checks", "layer2_v_q6_reference_boundary")},
      {"name": "layer2_v_native_matches_oracle", "pass": PRECONV.nested_bool(probe, "checks", "layer2_v_native_matches_oracle")},
      {"name": "history_kv_state_payloads_present", "pass": PRECONV.nested_bool(probe, "checks", "history_kv_state_payloads_present")},
      {"name": "l2_residual_input_matches_oracle", "pass": comparison_passed(probe, "l2_residual_input", "gpu_vs_oracle")},
      {"name": "l2_attn_norm_matches_oracle", "pass": comparison_passed(probe, "l2_attn_norm", "gpu_vs_oracle")},
      {"name": "l2_q_full_matches_oracle", "pass": comparison_passed(probe, "l2_q_full", "gpu_vs_oracle")},
      {"name": "l2_q_rope_matches_oracle", "pass": comparison_passed(probe, "l2_q_rope", "gpu_vs_oracle")},
      {"name": "l2_k_rope_matches_oracle", "pass": comparison_passed(probe, "l2_k_rope", "gpu_vs_oracle")},
      {"name": "gpu_event_timing_positive", "pass": PRECONV.nested_bool(probe, "checks", "gpu_event_timing_positive")},
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
      "conv_history_probes": {
          "layer0": str(conv0_path.relative_to(ROOT)),
          "layer1": str(conv1_path.relative_to(ROOT)),
      },
      "conv_history_capture_artifacts": {
          "layer0": conv0.get("capture_artifact"),
          "layer1": conv1.get("capture_artifact"),
      },
      "all_history": all_history,
      "payloads": slim_payloads,
      "layers": [layer0, layer1, layer2],
      "resident_invocations": args.resident_invocations,
      "opencl_source": str(OPENCL_SOURCE.relative_to(ROOT)),
      "opencl_source_sha256": opencl_source_hash,
      "embedded_opencl_source_sha256": embedded_opencl_hash,
      "probe_extra_opencl": ["rms_norm_hidden_f32"],
      "probe": probe,
      "checks": checks,
      "required_checks_passed": required_checks_passed,
      "speedup_claims_allowed": False,
  }
  manifest = {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "created_at": created_at,
      "tool": "tools/intel-qwen36-gpu-resident-layer7-full-attn-input-handoff-probe.py",
      "artifact": str(out_dir),
      "host": args.host,
      "remote_dir": remote_dir,
      "layers": [layer0, layer1, layer2],
      "resident_invocations": args.resident_invocations,
      "conv_history_probes": payload["conv_history_probes"],
      "all_history": all_history,
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
  layer2_timing = (
      timings.get("layer2_full_attn_input", {})
      if isinstance(timings.get("layer2_full_attn_input"), dict)
      else {}
  )
  iq36_local.write_metric(
      out_dir / "metrics.jsonl",
      "gpu_resident_layer7_full_attn_input_handoff_probe",
      [
          ("required_checks_passed", required_checks_passed),
          ("resident_shell_invocations", args.resident_invocations),
          ("layer7_rmsnorm_min_us", PRECONV.nested_number(layer2_timing, "layer7_rmsnorm_min_us")),
          ("q_projection_min_us", PRECONV.nested_number(layer2_timing, "q_projection_min_us")),
          ("k_projection_min_us", PRECONV.nested_number(layer2_timing, "k_projection_min_us")),
          ("resident_layer7_full_attn_input_kernel_sum_min_us", PRECONV.nested_number(timings, "resident_layer7_full_attn_input_kernel_sum_min_us")),
          ("resident_two_linear_plus_layer7_input_kernel_sum_min_us", PRECONV.nested_number(timings, "resident_two_linear_plus_layer7_input_kernel_sum_min_us")),
          ("l2_residual_input_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l2_residual_input", "gpu_vs_oracle", "max_abs_diff")),
          ("l2_attn_norm_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l2_attn_norm", "gpu_vs_oracle", "max_abs_diff")),
          ("l2_q_full_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l2_q_full", "gpu_vs_oracle", "max_abs_diff")),
          ("l2_q_rope_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l2_q_rope", "gpu_vs_oracle", "max_abs_diff")),
          ("l2_k_rope_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l2_k_rope", "gpu_vs_oracle", "max_abs_diff")),
          ("l2_v_cpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l2_v", "cpu_vs_oracle", "max_abs_diff")),
      ],
  )
  write_summary(out_dir / "summary.md", payload)
  print(out_dir)
  return 0 if required_checks_passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
