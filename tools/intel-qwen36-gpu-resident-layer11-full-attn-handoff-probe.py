#!/usr/bin/env python3
"""Run the resident GPU layer-11 full-attention handoff probe."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import shlex
from pathlib import Path
from types import ModuleType
from typing import Any

import iq36_local


ROOT = Path(__file__).resolve().parents[1]
L10_TOOL = Path(__file__).with_name(
    "intel-qwen36-gpu-resident-layer10-full-shell-handoff-probe.py"
)
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA_VERSION = "intel-qwen36-gpu-resident-layer11-full-attn-handoff-probe-v0"
DEFAULT_HOST = "local"
DEFAULT_MODEL = "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf"
DEFAULT_ENV_SCRIPT = "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh"
DEFAULT_REMOTE_ROOT = "/home/intel/intel-qwen36-gpu"
DEFAULT_ORACLE_BUNDLE = ROOT / "oracle/r0-oracle-bundle-20260627T060028Z"
DEFAULT_ALL_HISTORY = ROOT / "output/r1-full-attn-all-history-capture-20260627T145615Z/history.json"
OPENCL_SOURCE = ROOT / "engine/gpu/opencl/q4x8_matvec.cl"
SOURCE_TOKEN_POSITION = 15


def load_l10_tool() -> ModuleType:
  spec = importlib.util.spec_from_file_location("iq36_layer10_full_shell_probe", L10_TOOL)
  if spec is None or spec.loader is None:
    raise SystemExit(f"failed to load layer10 full-shell tool: {L10_TOOL}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


L10 = load_l10_tool()
L9 = L10.L9
L8 = L10.L8
CORE = L10.CORE
L7_INPUT = L10.L7_INPUT
TWO = L10.TWO
PRECONV = L10.PRECONV
CANONICAL_PAYLOAD_ROOT = PRECONV.PAYLOAD_ROOT


LAYER11_FULL_ATTN_HELPERS_CPP = r'''

FullAttentionPayloads LoadFullAttentionPayloadsPrefixed(
    const std::string& payload_dir,
    const std::string& prefix) {
  auto f32 = [&](const std::string& name) {
    return iq36::read_f32_vector_file(JoinPath(payload_dir, prefix + name));
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
    std::string token_prefix = "hist";
    if (token < 10) {
      token_prefix += "0";
    }
    token_prefix += std::to_string(token);
    p.k_history.push_back(f32(token_prefix + "_k_rope.bin"));
    p.v_history.push_back(f32(token_prefix + "_v.bin"));
  }
  return p;
}

FullAttentionVQ6Run RunGpuFullAttentionVAny(
    const std::string& model_path,
    const iq36::GgufTensorInfo& v_tensor,
    const std::vector<float>& attn_norm,
    const std::string& device_substring,
    int repeat) {
  Require(v_tensor.type == 12 || v_tensor.type == 14,
          "full attention V tensor must be Q4_K or Q6_K");
  if (v_tensor.type == 14) {
    return RunGpuFullAttentionVQ6(
        model_path, v_tensor, attn_norm, device_substring, repeat);
  }
  Require(v_tensor.dims == std::vector<std::uint64_t>{kHiddenSize, kFullKvValues},
          "full attention V Q4 tensor shape mismatch");
  Require(attn_norm.size() == kHiddenSize,
          "full attention V Q4 attn_norm size mismatch");
  FullAttentionVQ6Run run;
  run.v.assign(kFullKvValues, 0.0f);
  std::ifstream model(model_path, std::ios::binary);
  Require(static_cast<bool>(model), "failed to open model for full attention V Q4");
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
  const auto projected =
      RunProjectionFromTensor(runner, model, v_tensor, q8, kFullKvValues, repeat);
  run.v = projected.output;
  run.timing.v_projection_min_us = projected.timing.min_us;
  run.timing.v_projection_mean_us = projected.timing.mean_us;
  const double raw_bytes = static_cast<double>(v_tensor.nbytes);
  const double io_bytes = raw_bytes +
      static_cast<double>(q8.qs.size() * sizeof(std::int8_t)) +
      static_cast<double>(q8.d.size() * sizeof(float)) +
      static_cast<double>(run.v.size() * sizeof(float));
  run.timing.effective_raw_gb_s =
      raw_bytes / (run.timing.v_projection_min_us / 1e6) / 1e9;
  run.timing.effective_io_gb_s =
      io_bytes / (run.timing.v_projection_min_us / 1e6) / 1e9;
  run.timing.global_work_items = run.v.size();
  run.timing.kernel_launches = 1;
  return run;
}
'''


def replace_once(text: str, old: str, new: str) -> str:
  count = text.count(old)
  if count != 1:
    raise SystemExit(f"expected exactly one source replacement for {old[:100]!r}, found {count}")
  return text.replace(old, new, 1)


def layer11_full_attn_probe_cpp(opencl_source: str) -> str:
  cpp = L10.layer10_full_shell_probe_cpp(opencl_source)
  main_index = cpp.index("\nint main(")
  cpp = cpp[:main_index] + LAYER11_FULL_ATTN_HELPERS_CPP + cpp[main_index:]
  for old, new in {
      L10.SCHEMA_VERSION: SCHEMA_VERSION,
      "two_linear_layer_to_full_attention_v_q6_live_ffn_lout_layer10_full_shell_load_once_run_many":
          "layer5_to_layer11_full_attention_load_once_run_many",
      "layer10 full-shell handoff probe expects --layer 5":
          "layer11 full-attention handoff probe expects --layer 5",
  }.items():
    cpp = replace_once(cpp, old, new)
  cpp = replace_once(
      cpp,
      '''    const int layer5 = args.layer + 5;
    Require(layer0 == 5 && layer1 == 6 && layer2 == 7 && layer3 == 8 && layer4 == 9 && layer5 == 10,
            "layer11 full-attention handoff probe expects --layer 5");
''',
      '''    const int layer5 = args.layer + 5;
    const int layer6 = args.layer + 6;
    Require(layer0 == 5 && layer1 == 6 && layer2 == 7 && layer3 == 8 && layer4 == 9 && layer5 == 10 && layer6 == 11,
            "layer11 full-attention handoff probe expects --layer 5");
''',
  )
  cpp = replace_once(
      cpp,
      '''    const auto layer5_tensors = ResolveLayerTensorBundle(index, layer5);
    const auto layer0_oracle = LoadLayerOraclePayloads(args.payload_dir, "l0");
''',
      '''    const auto layer5_tensors = ResolveLayerTensorBundle(index, layer5);
    const auto layer6_tensors = ResolveFullAttentionTensorBundle(index, layer6);
    const auto layer0_oracle = LoadLayerOraclePayloads(args.payload_dir, "l0");
''',
  )
  cpp = replace_once(
      cpp,
      '''    const auto layer5_oracle = LoadLayerOraclePayloads(args.payload_dir, "l5");
    const auto oracle_attn_residual =
''',
      '''    const auto layer5_oracle = LoadLayerOraclePayloads(args.payload_dir, "l5");
    const auto layer6_oracle = LoadFullAttentionPayloadsPrefixed(args.payload_dir, "l6_");
    const auto oracle_attn_residual =
''',
  )
  cpp = replace_once(
      cpp,
      '''    const auto oracle_layer_output =
        iq36::add_vectors(oracle_attn_residual, oracle_ffn_out);
''',
      '''    const auto oracle_layer_output =
        iq36::add_vectors(oracle_attn_residual, oracle_ffn_out);
    const auto layer6_oracle_attn_residual =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "l6_full_attn_residual.bin"));
    const auto layer6_oracle_attn_post_norm =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "l6_full_attn_post_norm.bin"));
''',
  )
  cpp = replace_once(
      cpp,
      '''    const auto full_shapes = CheckFullAttentionShapes(layer2_tensors);
    const bool full_shapes_ok = FullAttentionShapesPassed(full_shapes);
    const bool layer2_payload_counts_ok =
''',
      '''    const auto full_shapes = CheckFullAttentionShapes(layer2_tensors);
    const bool full_shapes_ok = FullAttentionShapesPassed(full_shapes);
    const auto layer6_full_shapes = CheckFullAttentionShapes(layer6_tensors);
    const bool layer6_full_shapes_ok = FullAttentionShapesPassed(layer6_full_shapes);
    const bool layer2_payload_counts_ok =
''',
  )
  cpp = replace_once(
      cpp,
      '''        oracle_ffn_out.size() == kHiddenSize &&
        oracle_layer_output.size() == kHiddenSize;
    const bool metadata_ok =
''',
      '''        oracle_ffn_out.size() == kHiddenSize &&
        oracle_layer_output.size() == kHiddenSize;
    const bool layer6_payload_counts_ok =
        FullAttentionPayloadCountsOk(layer6_oracle) &&
        layer6_oracle_attn_residual.size() == kHiddenSize &&
        layer6_oracle_attn_post_norm.size() == kHiddenSize;
    const bool metadata_ok =
''',
  )
  cpp = replace_once(
      cpp,
      '''    const auto ffn_norm_weight =
        iq36::decode_tensor_row(args.model_path, index,
                                LayerTensorName(layer2, "post_attention_norm.weight"), 0);
    const std::string selected_gate_up_tensor_name =
''',
      '''    const auto ffn_norm_weight =
        iq36::decode_tensor_row(args.model_path, index,
                                LayerTensorName(layer2, "post_attention_norm.weight"), 0);
    const auto layer6_attn_norm_weight =
        ReadF32TensorPayload(model, *layer6_tensors.attn_norm_tensor,
                             static_cast<std::size_t>(kHiddenSize));
    const auto layer6_q_norm_weight =
        iq36::decode_tensor_row(args.model_path, index,
                                layer6_tensors.q_norm_tensor_name, 0);
    const auto layer6_k_norm_weight =
        iq36::decode_tensor_row(args.model_path, index,
                                layer6_tensors.k_norm_tensor_name, 0);
    const auto layer6_ffn_norm_weight =
        iq36::decode_tensor_row(args.model_path, index,
                                LayerTensorName(layer6, "post_attention_norm.weight"), 0);
    const std::string selected_gate_up_tensor_name =
''',
  )
  cpp = replace_once(
      cpp,
      '''    const auto layer5_run = RunResidentLinearLayerShell(
        args, index, layer5_tensors, layer5_oracle,
        layer4_run.gpu_layer_output, layer4_run.gpu_layer_output, rms_norm_epsilon);

    std::vector<NamedCompareGroup> strict_groups;
''',
      '''    const auto layer5_run = RunResidentLinearLayerShell(
        args, index, layer5_tensors, layer5_oracle,
        layer4_run.gpu_layer_output, layer4_run.gpu_layer_output, rms_norm_epsilon);

    const auto layer6_native_qkv = iq36::run_qwen36_full_attention_qkv_projection(
        args.model_path,
        index,
        layer6,
        layer5_run.gpu_layer_output,
        rms_norm_epsilon);
    const auto layer6_rms_gpu = RunGpuLayerInputRmsNorm(
        layer5_run.gpu_layer_output,
        layer6_attn_norm_weight,
        rms_norm_epsilon,
        args.device_substring,
        args.repeat);
    const auto layer6_qk_gpu = RunGpuFullAttentionQkFront(
        args.model_path,
        *layer6_tensors.q_tensor,
        *layer6_tensors.k_tensor,
        layer6_rms_gpu.attn_norm,
        args.device_substring,
        args.repeat);
    const auto layer6_v_gpu = RunGpuFullAttentionVAny(
        args.model_path,
        *layer6_tensors.v_tensor,
        layer6_rms_gpu.attn_norm,
        args.device_substring,
        args.repeat);
    const auto layer6_gpu_q_split = SplitFullAttentionQ(layer6_qk_gpu.q_full);
    const auto layer6_gpu_q_normed = ApplyRepeatedRmsNormFull(
        layer6_gpu_q_split.q_raw, layer6_q_norm_weight, rms_norm_epsilon);
    const auto layer6_gpu_k_normed = ApplyRepeatedRmsNormFull(
        layer6_qk_gpu.k_raw, layer6_k_norm_weight, rms_norm_epsilon);
    const auto layer6_gpu_rope = iq36::run_qwen36_full_attention_rope(
        layer6_gpu_q_normed,
        layer6_gpu_k_normed,
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
    const auto layer6_native_rope = iq36::run_qwen36_full_attention_rope(
        layer6_native_qkv.q_normed,
        layer6_native_qkv.k_normed,
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
    auto layer6_native_k_history = layer6_oracle.k_history;
    auto layer6_native_v_history = layer6_oracle.v_history;
    layer6_native_k_history.push_back(layer6_oracle.k_rope);
    layer6_native_v_history.push_back(layer6_oracle.v);
    const auto layer6_native_core = iq36::run_qwen36_full_attention_core(
        layer6_oracle.q_rope,
        layer6_native_k_history,
        layer6_native_v_history,
        head_dim,
        q_head_count,
        kv_head_count,
        kAttentionScale);
    const auto layer6_native_gate = iq36::run_qwen36_full_attention_gate(
        layer6_oracle.q_full, layer6_native_core.attn_pregate, head_dim);
    const auto layer6_native_attn_output = iq36::matvec_tensor(
        args.model_path,
        index,
        LayerTensorName(layer6, "attn_output.weight"),
        layer6_native_gate.attn_gated);
    const auto layer6_native_attn_residual =
        iq36::add_vectors(layer5_run.gpu_layer_output, layer6_native_attn_output);
    const auto layer6_native_attn_post_norm =
        iq36::apply_rms_norm(layer6_native_attn_residual,
                             layer6_ffn_norm_weight,
                             rms_norm_epsilon);

    auto layer6_gpu_k_history = layer6_oracle.k_history;
    auto layer6_gpu_v_history = layer6_oracle.v_history;
    layer6_gpu_k_history.push_back(layer6_oracle.k_rope);
    layer6_gpu_v_history.push_back(layer6_v_gpu.v);
    std::vector<float> layer6_gpu_k_history_flat;
    std::vector<float> layer6_gpu_v_history_flat;
    for (const auto& item : layer6_gpu_k_history) {
      layer6_gpu_k_history_flat.insert(layer6_gpu_k_history_flat.end(), item.begin(), item.end());
    }
    for (const auto& item : layer6_gpu_v_history) {
      layer6_gpu_v_history_flat.insert(layer6_gpu_v_history_flat.end(), item.begin(), item.end());
    }
    const auto layer6_core_gate_gpu = RunGpuFullAttentionCoreGate(
        layer6_oracle.q_rope,
        layer6_gpu_k_history_flat,
        layer6_gpu_v_history_flat,
        layer6_qk_gpu.q_full,
        args.device_substring,
        args.repeat);
    const auto layer6_attention_gpu = RunGpuAttentionFront(
        args.model_path,
        *iq36::find_tensor(index, LayerTensorName(layer6, "attn_output.weight")),
        layer6_core_gate_gpu.attn_gated,
        layer5_run.gpu_layer_output,
        layer6_ffn_norm_weight,
        rms_norm_epsilon,
        args.device_substring,
        args.repeat);

    std::vector<NamedCompareGroup> strict_groups;
''',
  )
  cpp = replace_once(
      cpp,
      '''    AppendFullAttentionComponentCompare(full_attention_groups, "l2_attn_post_norm",
                                        native_attn_post_norm,
                                        attention_gpu.attn_post_norm,
                                        oracle_attn_post_norm);

    const auto k_raw_gpu_vs_cpu = iq36::compare_vectors(
''',
      '''    AppendFullAttentionComponentCompare(full_attention_groups, "l2_attn_post_norm",
                                        native_attn_post_norm,
                                        attention_gpu.attn_post_norm,
                                        oracle_attn_post_norm);

    std::vector<NamedCompareGroup> layer6_strict_groups;
    AppendCpuGpuOracleCompare(layer6_strict_groups, "l6_residual_input",
                              layer5_run.gpu_layer_output,
                              layer5_run.gpu_layer_output,
                              layer6_oracle.residual_input);
    AppendCpuGpuOracleCompare(layer6_strict_groups, "l6_attn_norm",
                              layer6_native_qkv.attention_norm,
                              layer6_rms_gpu.attn_norm,
                              layer6_oracle.attn_norm);
    AppendCpuGpuOracleCompare(layer6_strict_groups, "l6_q_full",
                              layer6_native_qkv.q_full,
                              layer6_qk_gpu.q_full,
                              layer6_oracle.q_full);
    AppendCpuGpuOracleCompare(layer6_strict_groups, "l6_q_rope",
                              layer6_native_rope.q_rope,
                              layer6_gpu_rope.q_rope,
                              layer6_oracle.q_rope);
    AppendCpuGpuOracleCompare(layer6_strict_groups, "l6_k_rope",
                              layer6_native_rope.k_rope,
                              layer6_gpu_rope.k_rope,
                              layer6_oracle.k_rope);
    AppendCpuGpuOracleCompare(layer6_strict_groups, "l6_v",
                              layer6_native_qkv.v,
                              layer6_v_gpu.v,
                              layer6_oracle.v);
    std::vector<NamedCompareGroup> layer6_full_attention_groups;
    AppendFullAttentionComponentCompare(layer6_full_attention_groups, "l6_attn_pregate",
                                        layer6_native_core.attn_pregate,
                                        layer6_core_gate_gpu.attn_pregate,
                                        layer6_oracle.attn_pregate);
    AppendFullAttentionComponentCompare(layer6_full_attention_groups, "l6_attn_gated",
                                        layer6_native_gate.attn_gated,
                                        layer6_core_gate_gpu.attn_gated,
                                        layer6_oracle.attn_gated);
    AppendFullAttentionComponentCompare(layer6_full_attention_groups, "l6_attn_output",
                                        layer6_native_attn_output,
                                        layer6_attention_gpu.linear_attn_out,
                                        layer6_oracle.attn_output);
    AppendFullAttentionComponentCompare(layer6_full_attention_groups, "l6_attn_residual",
                                        layer6_native_attn_residual,
                                        layer6_attention_gpu.attn_residual,
                                        layer6_oracle_attn_residual);
    AppendFullAttentionComponentCompare(layer6_full_attention_groups, "l6_attn_post_norm",
                                        layer6_native_attn_post_norm,
                                        layer6_attention_gpu.attn_post_norm,
                                        layer6_oracle_attn_post_norm);

    const auto k_raw_gpu_vs_cpu = iq36::compare_vectors(
''',
  )
  cpp = replace_once(
      cpp,
      '''    const bool layer5_ok =
        layer5_shapes_ok &&
        layer5_run.payload_counts_ok &&
        layer5_run.comparisons_passed &&
        layer5_gpu_cpu_ok &&
        layer5_state_input_oracle_policy_ok &&
        layer5_q6_qkv_boundary &&
        layer5_q6_down_boundary &&
        layer5_run.timing_positive &&
        layer5_run.arc_selected;
    const bool layer2_timing_positive =
''',
      '''    const bool layer5_ok =
        layer5_shapes_ok &&
        layer5_run.payload_counts_ok &&
        layer5_run.comparisons_passed &&
        layer5_gpu_cpu_ok &&
        layer5_state_input_oracle_policy_ok &&
        layer5_q6_qkv_boundary &&
        layer5_q6_down_boundary &&
        layer5_run.timing_positive &&
        layer5_run.arc_selected;
    const auto layer6_k_raw_gpu_vs_cpu = iq36::compare_vectors(
        layer6_qk_gpu.k_raw, layer6_native_qkv.k_raw, kMismatchThreshold);
    const bool layer6_strict_input_ok =
        CompareGroupsPassed(layer6_strict_groups) &&
        ComparePassed(layer6_k_raw_gpu_vs_cpu);
    bool layer6_full_component_ok = true;
    for (const auto& group : layer6_full_attention_groups) {
      layer6_full_component_ok =
          layer6_full_component_ok &&
          ComparePassedFullAttentionComponent(group.gpu_vs_oracle);
    }
    const bool layer6_comparisons_ok =
        layer6_strict_input_ok && layer6_full_component_ok;
    const bool layer6_timing_positive =
        layer6_rms_gpu.timing.rmsnorm_min_us > 0.0 &&
        layer6_qk_gpu.timing.q_projection_min_us > 0.0 &&
        layer6_qk_gpu.timing.k_projection_min_us > 0.0 &&
        layer6_v_gpu.timing.v_projection_min_us > 0.0 &&
        layer6_core_gate_gpu.timing.core_min_us > 0.0 &&
        layer6_core_gate_gpu.timing.gate_min_us > 0.0 &&
        layer6_attention_gpu.timing.output_projection_min_us > 0.0 &&
        layer6_attention_gpu.timing.residual_add_min_us > 0.0 &&
        layer6_attention_gpu.timing.ffn_rmsnorm_min_us > 0.0;
    const bool layer6_arc_selected =
        layer6_rms_gpu.device_name.find(args.device_substring) != std::string::npos &&
        layer6_qk_gpu.device_name.find(args.device_substring) != std::string::npos &&
        layer6_v_gpu.device_name.find(args.device_substring) != std::string::npos &&
        layer6_core_gate_gpu.device_name.find(args.device_substring) != std::string::npos &&
        layer6_attention_gpu.device_name.find(args.device_substring) != std::string::npos;
    const bool layer6_v_gpu_boundary =
        (layer6_tensors.v_tensor->type == 12 || layer6_tensors.v_tensor->type == 14) &&
        layer6_v_gpu.v.size() == kFullKvValues;
    const bool layer6_ffn_q6_boundary =
        iq36::find_tensor(index, LayerTensorName(layer6, "ffn_down_exps.weight"))->type == 14 &&
        iq36::find_tensor(index, LayerTensorName(layer6, "ffn_down_shexp.weight"))->type == 14;
    const bool layer6_ok =
        layer6_full_shapes_ok &&
        layer6_payload_counts_ok &&
        metadata_ok &&
        layer6_comparisons_ok &&
        layer6_timing_positive &&
        layer6_arc_selected &&
        layer6_v_gpu_boundary;
    const bool layer2_timing_positive =
''',
  )
  cpp = replace_once(
      cpp,
      '''        layer5_ok &&
        layer2_timing_positive &&
''',
      '''        layer5_ok &&
        layer6_ok &&
        layer2_timing_positive &&
''',
  )
  cpp = replace_once(
      cpp,
      '''    const double layer5_state_input_sum_min =
        layer5_run.timing.layer_input_rmsnorm_min_us +
        layer5_run.timing.preconv_to_postconv_kernel_sum_min_us;
''',
      '''    const double layer5_state_input_sum_min =
        layer5_run.timing.layer_input_rmsnorm_min_us +
        layer5_run.timing.preconv_to_postconv_kernel_sum_min_us;
    const double layer6_input_sum_min =
        layer6_rms_gpu.timing.rmsnorm_min_us +
        layer6_qk_gpu.timing.qk_projection_kernel_sum_min_us +
        layer6_v_gpu.timing.v_projection_min_us;
    const double layer6_core_output_sum_min =
        layer6_core_gate_gpu.timing.core_gate_kernel_sum_min_us +
        layer6_attention_gpu.timing.attention_front_kernel_sum_min_us;
    const double layer6_attention_sum_min =
        layer6_input_sum_min + layer6_core_output_sum_min;
''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << "\\"layers\\":[" << layer0 << "," << layer1 << "," << layer2 << "," << layer3 << "," << layer4 << "," << layer5 << "],";
''',
      '''    std::cout << "\\"layers\\":[" << layer0 << "," << layer1 << "," << layer2 << "," << layer3 << "," << layer4 << "," << layer5 << "," << layer6 << "],";
''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << "\\"layer10_conv_state_boundary\\":\\"captured_conv_state\\",";
    std::cout << "\\"full_attn_core_input_boundary\\":\\"captured_rope_kv_payloads\\",";''',
      '''    std::cout << "\\"layer10_conv_state_boundary\\":\\"captured_conv_state\\",";
    std::cout << "\\"layer11_residual_input_boundary\\":\\"live_gpu_l_out_10\\",";
    std::cout << "\\"layer11_v_tensor_type\\":\\""
              << JsonEscape(iq36::ggml_type_name(layer6_tensors.v_tensor->type)) << "\\",";
    std::cout << "\\"layer11_v_projection_boundary\\":\\""
              << (layer6_tensors.v_tensor->type == 14 ? "gpu_q6_raw_matvec" : "gpu_q4x8_matvec") << "\\",";
    std::cout << "\\"layer11_ffn_boundary\\":\\"q6_down_pending\\",";
    std::cout << "\\"full_attn_core_input_boundary\\":\\"captured_rope_kv_payloads\\",";''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << "\\"v_projection_device_name\\":\\"" << JsonEscape(layer2_v_gpu.device_name) << "\\",";
    std::cout << "\\"layer7_v_tensor_type\\":\\""
''',
      '''    std::cout << "\\"v_projection_device_name\\":\\"" << JsonEscape(layer2_v_gpu.device_name) << "\\",";
    std::cout << "\\"layer11_core_gate_device_name\\":\\"" << JsonEscape(layer6_core_gate_gpu.device_name) << "\\",";
    std::cout << "\\"layer11_output_projection_device_name\\":\\"" << JsonEscape(layer6_attention_gpu.device_name) << "\\",";
    std::cout << "\\"layer11_v_projection_device_name\\":\\"" << JsonEscape(layer6_v_gpu.device_name) << "\\",";
    std::cout << "\\"layer7_v_tensor_type\\":\\""
''',
  )
  cpp = replace_once(
      cpp,
      '''                  tail_gpu.program_build_ms + layer3_run.program_build_ms +
                  layer4_run.program_build_ms + layer5_run.program_build_ms)
''',
      '''                  tail_gpu.program_build_ms + layer3_run.program_build_ms +
                  layer4_run.program_build_ms + layer5_run.program_build_ms +
                  layer6_rms_gpu.program_build_ms + layer6_qk_gpu.program_build_ms +
                  layer6_v_gpu.program_build_ms +
                  layer6_core_gate_gpu.program_build_ms +
                  layer6_attention_gpu.program_build_ms)
''',
  )
  cpp = replace_once(
      cpp,
      '''                            tail_gpu.build_log + layer3_run.build_log +
                            layer4_run.build_log + layer5_run.build_log)
''',
      '''                            tail_gpu.build_log + layer3_run.build_log +
                            layer4_run.build_log + layer5_run.build_log +
                            layer6_rms_gpu.build_log + layer6_qk_gpu.build_log +
                            layer6_v_gpu.build_log +
                            layer6_core_gate_gpu.build_log +
                            layer6_attention_gpu.build_log)
''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << ",\\"layer2_output_front\\":";
    WriteFullAttentionOutputFrontTiming(attention_gpu.timing);
''',
      '''    std::cout << ",\\"layer2_output_front\\":";
    WriteFullAttentionOutputFrontTiming(attention_gpu.timing);
    std::cout << ",\\"layer6_full_attn_input\\":";
    WriteFullAttentionTiming(layer6_rms_gpu, layer6_qk_gpu);
    std::cout << ",\\"layer6_v_projection\\":";
    WriteFullAttentionVQ6Timing(layer6_v_gpu.timing);
    std::cout << ",\\"layer6_core_gate\\":";
    WriteFullCoreGateTiming(layer6_core_gate_gpu.timing);
    std::cout << ",\\"layer6_output_front\\":";
    WriteFullAttentionOutputFrontTiming(layer6_attention_gpu.timing);
''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << "\\"resident_layer5_6_7_8_9_10_full_shell_kernel_sum_min_us\\":"
              << (two_layer_kernel_sum_min + layer2_attention_sum_min +
                  layer2_ffn_sum_min + layer3_run.timing.layer_kernel_sum_min_us +
                  layer4_run.timing.layer_kernel_sum_min_us +
                  layer5_run.timing.layer_kernel_sum_min_us);
''',
      '''    std::cout << "\\"resident_layer5_6_7_8_9_10_full_shell_kernel_sum_min_us\\":"
              << (two_layer_kernel_sum_min + layer2_attention_sum_min +
                  layer2_ffn_sum_min + layer3_run.timing.layer_kernel_sum_min_us +
                  layer4_run.timing.layer_kernel_sum_min_us +
                  layer5_run.timing.layer_kernel_sum_min_us) << ",";
    std::cout << "\\"resident_layer11_full_attn_input_kernel_sum_min_us\\":"
              << layer6_input_sum_min << ",";
    std::cout << "\\"resident_layer11_full_attn_core_output_kernel_sum_min_us\\":"
              << layer6_core_output_sum_min << ",";
    std::cout << "\\"resident_layer11_full_attention_kernel_sum_min_us\\":"
              << layer6_attention_sum_min << ",";
    std::cout << "\\"resident_layer5_6_7_8_9_10_to_layer11_full_attention_kernel_sum_min_us\\":"
              << (two_layer_kernel_sum_min + layer2_attention_sum_min +
                  layer2_ffn_sum_min + layer3_run.timing.layer_kernel_sum_min_us +
                  layer4_run.timing.layer_kernel_sum_min_us +
                  layer5_run.timing.layer_kernel_sum_min_us +
                  layer6_attention_sum_min);
''',
  )
  cpp = replace_once(
      cpp,
      '''    WritePrefixedCompareGroups("l5", layer5_run.comparisons, &first_compare);
    std::cout << ",\\"l2_k_raw\\":{\\"gpu_vs_cpu\\":";
''',
      '''    WritePrefixedCompareGroups("l5", layer5_run.comparisons, &first_compare);
    if (!first_compare) {
      std::cout << ",";
    }
    first_compare = false;
    WriteNamedCompareGroups(layer6_strict_groups);
    std::cout << ",";
    WriteNamedCompareGroups(layer6_full_attention_groups);
    std::cout << ",\\"l2_k_raw\\":{\\"gpu_vs_cpu\\":";
''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << "},\\"l5_recurrent_state\\":{\\"gpu_vs_cpu\\":";
    WriteCompare(layer5_run.recurrent_state_gpu_vs_cpu);
    std::cout << "}";
    std::cout << "},\\"checks\\":{";
''',
      '''    std::cout << "},\\"l5_recurrent_state\\":{\\"gpu_vs_cpu\\":";
    WriteCompare(layer5_run.recurrent_state_gpu_vs_cpu);
    std::cout << "},\\"l6_k_raw\\":{\\"gpu_vs_cpu\\":";
    WriteCompare(layer6_k_raw_gpu_vs_cpu);
    std::cout << "}";
    std::cout << "},\\"checks\\":{";
''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << ",\\"layer2_shapes\\":";
    WriteFullAttentionShapeChecks(full_shapes);
    std::cout << ",\\"layer3\\":";
''',
      '''    std::cout << ",\\"layer2_shapes\\":";
    WriteFullAttentionShapeChecks(full_shapes);
    std::cout << ",\\"layer6_shapes\\":";
    WriteFullAttentionShapeChecks(layer6_full_shapes);
    std::cout << ",\\"layer3\\":";
''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << "\\"layer10_full_shell_handoff_matches\\":"
              << (layer5_ok ? "true" : "false") << ",";
    std::cout << "\\"history_kv_state_payloads_present\\":"
''',
      '''    std::cout << "\\"layer10_full_shell_handoff_matches\\":"
              << (layer5_ok ? "true" : "false") << ",";
    std::cout << "\\"layer11_residual_input_from_layer10_live_gpu_lout\\":true,";
    std::cout << "\\"layer11_payload_counts_ok\\":"
              << (layer6_payload_counts_ok ? "true" : "false") << ",";
    std::cout << "\\"layer11_full_attn_input_matches_oracle\\":"
              << (layer6_strict_input_ok ? "true" : "false") << ",";
    std::cout << "\\"layer11_full_attn_component_policy_matches_oracle\\":"
              << (layer6_full_component_ok ? "true" : "false") << ",";
    std::cout << "\\"layer11_full_attn_core_output_matches_oracle\\":"
              << (layer6_comparisons_ok ? "true" : "false") << ",";
    std::cout << "\\"layer11_v_gpu_boundary\\":"
              << (layer6_v_gpu_boundary ? "true" : "false") << ",";
    std::cout << "\\"layer11_ffn_q6_boundary\\":"
              << (layer6_ffn_q6_boundary ? "true" : "false") << ",";
    std::cout << "\\"layer11_arc_device_selected\\":"
              << (layer6_arc_selected ? "true" : "false") << ",";
    std::cout << "\\"history_kv_state_payloads_present\\":"
''',
  )
  cpp = replace_once(
      cpp,
      '''                  layer2_timing_positive && layer3_run.timing_positive &&
                  layer4_run.timing_positive && layer5_run.timing_positive ? "true" : "false") << ",";
''',
      '''                  layer2_timing_positive && layer3_run.timing_positive &&
                  layer4_run.timing_positive && layer5_run.timing_positive &&
                  layer6_timing_positive ? "true" : "false") << ",";
''',
  )
  return cpp


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
  parser.add_argument("--layer8-conv-history-probe", type=Path, default=None)
  parser.add_argument("--layer9-conv-history-probe", type=Path, default=None)
  parser.add_argument("--layer10-conv-history-probe", type=Path, default=None)
  parser.add_argument("--out-dir", type=Path, default=None)
  return parser.parse_args()


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


def find_canonical_payload(pattern: str, expected_bytes: int) -> Path:
  matches = sorted(CANONICAL_PAYLOAD_ROOT.glob(pattern))
  if len(matches) != 1:
    raise SystemExit(
        f"expected one canonical payload for {pattern}, found {len(matches)}"
    )
  path = matches[0].resolve()
  if path.stat().st_size != expected_bytes:
    raise SystemExit(f"payload size mismatch for {path}: {path.stat().st_size}")
  return path


def resolve_prefixed_full_attention_payloads(
    history_json: Path,
    layer: int,
    prefix: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
  history_json = history_json.resolve()
  history_doc = L7_INPUT.load_json(history_json)
  if history_doc.get("schema_version") != "intel-qwen36-r1-full-attn-all-history-capture-v0":
    raise SystemExit(f"{history_json}: unexpected all-history schema")
  if history_doc.get("full_attn_all_history_capture_passed") is not True:
    raise SystemExit(f"{history_json}: all-history capture did not pass")
  capture = history_doc.get("full_attn_all_history_capture", {})
  tokens = capture.get("tokens", [])
  if capture.get("history_token_count") != 16 or len(tokens) != 16:
    raise SystemExit(f"{history_json}: expected 16 token history")
  previous_layer = layer - 1
  payloads: dict[str, dict[str, Any]] = {
      f"{prefix}full_residual_input": payload_record(
          find_canonical_payload(f"l_out-{previous_layer}__tok15__ord*.bin", 8192),
          f"{prefix}full_residual_input.bin",
          8192,
      ),
      f"{prefix}full_attn_norm": payload_record(
          find_canonical_payload(f"attn_norm-{layer}__tok15__ord*.bin", 8192),
          f"{prefix}full_attn_norm.bin",
          8192,
      ),
      f"{prefix}full_q_full": payload_record(
          find_canonical_payload(f"Qcur_full-{layer}__tok15__ord*.bin", 32768),
          f"{prefix}full_q_full.bin",
          32768,
      ),
      f"{prefix}full_attn_residual": payload_record(
          find_canonical_payload(f"attn_residual-{layer}__tok15__ord*.bin", 8192),
          f"{prefix}full_attn_residual.bin",
          8192,
      ),
      f"{prefix}full_attn_post_norm": payload_record(
          find_canonical_payload(f"attn_post_norm-{layer}__tok15__ord*.bin", 8192),
          f"{prefix}full_attn_post_norm.bin",
          8192,
      ),
  }
  for token_index, token in enumerate(tokens):
    if token.get("source_token_position") != token_index:
      raise SystemExit(f"{history_json}: token position mismatch at {token_index}")
    entry = L7_INPUT.history_layer_entry(token, layer)
    if token_index < SOURCE_TOKEN_POSITION:
      token_prefix = f"hist{token_index:02d}"
      for payload_name in ("k_rope", "v"):
        key = f"{prefix}{token_prefix}_{payload_name}"
        payloads[key] = L7_INPUT.history_payload_record(
            history_json,
            token_index,
            layer,
            payload_name,
            f"{prefix}{token_prefix}_{payload_name}.bin",
            entry,
        )
    elif token_index == SOURCE_TOKEN_POSITION:
      for payload_name, stage_name in (
          ("q_rope", f"{prefix}full_q_rope.bin"),
          ("k_rope", f"{prefix}full_k_rope.bin"),
          ("v", f"{prefix}full_v.bin"),
          ("attn_pregate", f"{prefix}full_attn_pregate.bin"),
          ("attn_gated", f"{prefix}full_attn_gated.bin"),
          ("attn_output", f"{prefix}full_attn_output.bin"),
      ):
        payloads[f"{prefix}full_{payload_name}"] = L7_INPUT.history_payload_record(
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
      "layer": layer,
      "stage_prefix": prefix,
  }


def resident_fields_ok(probe: dict[str, Any] | None, expected_invocations: int) -> bool:
  return (
      isinstance(probe, dict)
      and probe.get("resident_api") == "layer5_to_layer11_full_attention_load_once_run_many"
      and probe.get("resident_load_count") == 1
      and probe.get("resident_shell_invocations") == expected_invocations
      and probe.get("layer10_residual_input_boundary") == "live_gpu_l_out_9"
      and probe.get("layer11_residual_input_boundary") == "live_gpu_l_out_10"
      and probe.get("layer11_v_tensor_type") in {"Q4_K", "Q6_K"}
      and probe.get("layer11_v_projection_boundary") in {"gpu_q4x8_matvec", "gpu_q6_raw_matvec"}
      and probe.get("layer11_ffn_boundary") == "q6_down_pending"
      and probe.get("full_attn_v_projection_boundary") == "gpu_q6_raw_matvec"
      and probe.get("full_attn_core_gpu_boundary") is True
      and probe.get("full_attn_gate_gpu_boundary") is True
      and probe.get("full_attn_output_projection_gpu_boundary") is True
      and probe.get("full_attn_post_norm_gpu_boundary") is True
      and probe.get("full_attn_ffn_boundary") == "gpu_live_post_norm_to_q6_down"
      and PRECONV.nested_bool(probe, "checks", "resident_load_once")
      and PRECONV.nested_bool(probe, "checks", "resident_shell_invocations_positive")
  )


def write_summary(path: Path, payload: dict[str, Any]) -> None:
  probe = payload.get("probe") if isinstance(payload.get("probe"), dict) else {}
  timings = probe.get("timings", {}) if isinstance(probe, dict) else {}
  comparisons = probe.get("comparisons", {}) if isinstance(probe, dict) else {}
  lines = [
      "# GPU Resident Layer-11 Full-Attention Handoff Probe",
      "",
      f"- workstream: `{WORKSTREAM}`",
      f"- host: `{payload.get('host')}`",
      f"- model: `{payload.get('model')}`",
      f"- layers: `{payload.get('layers')}`",
      f"- required checks passed: `{str(payload.get('required_checks_passed')).lower()}`",
      "- speedup claims allowed: `false`",
      f"- resident API: `{probe.get('resident_api')}`",
      f"- resident shell invocations: `{probe.get('resident_shell_invocations')}`",
      f"- layer 11 residual input boundary: `{probe.get('layer11_residual_input_boundary')}`",
      f"- layer 11 V tensor type: `{probe.get('layer11_v_tensor_type')}`",
      f"- layer 11 FFN boundary: `{probe.get('layer11_ffn_boundary')}`",
      f"- layer 11 V projection boundary: `{probe.get('layer11_v_projection_boundary')}`",
      "",
      "| output | comparison | max abs | RMSE |",
      "|---|---|---:|---:|",
  ]
  for name in (
      "l6_residual_input",
      "l6_attn_norm",
      "l6_q_full",
      "l6_q_rope",
      "l6_k_rope",
      "l6_v",
      "l6_attn_pregate",
      "l6_attn_gated",
      "l6_attn_output",
      "l6_attn_residual",
      "l6_attn_post_norm",
  ):
    group = comparisons.get(name, {}) if isinstance(comparisons, dict) else {}
    lane = group.get("gpu_vs_oracle", {}) if isinstance(group, dict) else {}
    lines.append(
        f"| {name} | gpu_vs_oracle | {lane.get('max_abs_diff')} | {lane.get('rmse')} |"
    )
  lines += [
      "",
      "| kernel group | min us |",
      "|---|---:|",
      f"| layer11_full_attn_input | {timings.get('resident_layer11_full_attn_input_kernel_sum_min_us') if isinstance(timings, dict) else None} |",
      f"| layer11_core_output | {timings.get('resident_layer11_full_attn_core_output_kernel_sum_min_us') if isinstance(timings, dict) else None} |",
      f"| layer11_attention_total | {timings.get('resident_layer11_full_attention_kernel_sum_min_us') if isinstance(timings, dict) else None} |",
      "",
      "This target-side process carries closed live GPU layer outputs through",
      "layer 10, then runs layer-11 full attention from live GPU `l_out-10`.",
      "K/V history and RoPE payloads remain captured teacher-forced boundaries;",
      "this is single-token handoff evidence, not decode throughput.",
      "",
  ]
  path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  args = parse_args()
  if args.resident_invocations <= 0:
    raise SystemExit("--resident-invocations must be positive")
  if args.layer != 5:
    raise SystemExit("--layer must be 5 for the layer-11 full-attention handoff")

  layer0 = args.layer
  layer1 = args.layer + 1
  layer2 = args.layer + 2
  layer3 = args.layer + 3
  layer4 = args.layer + 4
  layer5 = args.layer + 5
  layer6 = args.layer + 6
  conv0_path = (args.conv_history_probe.resolve() if args.conv_history_probe is not None
                else TWO.latest_conv_history_probe_for_layer(layer0).resolve())
  conv1_path = (args.next_conv_history_probe.resolve() if args.next_conv_history_probe is not None
                else TWO.latest_conv_history_probe_for_layer(layer1).resolve())
  conv2_path = (args.layer8_conv_history_probe.resolve() if args.layer8_conv_history_probe is not None
                else TWO.latest_conv_history_probe_for_layer(layer3).resolve())
  conv3_path = (args.layer9_conv_history_probe.resolve() if args.layer9_conv_history_probe is not None
                else TWO.latest_conv_history_probe_for_layer(layer4).resolve())
  conv4_path = (args.layer10_conv_history_probe.resolve() if args.layer10_conv_history_probe is not None
                else TWO.latest_conv_history_probe_for_layer(layer5).resolve())
  all_history_json = args.all_history_json.resolve()
  stamp = utc_stamp()
  created_at = iso_now()
  out_dir = args.out_dir or ROOT / f"output/gpu-resident-layer11-full-attn-handoff-probe-{stamp}"
  out_dir.mkdir(parents=True, exist_ok=False)
  raw_dir = out_dir / "raw"
  raw_dir.mkdir()

  payloads0, conv0 = TWO.prefixed_payloads(layer0, conv0_path, "l0")
  payloads1, conv1 = TWO.prefixed_payloads(layer1, conv1_path, "l1")
  payloads2, conv2 = TWO.prefixed_payloads(layer3, conv2_path, "l3")
  payloads3, conv3 = TWO.prefixed_payloads(layer4, conv3_path, "l4")
  payloads4, conv4 = TWO.prefixed_payloads(layer5, conv4_path, "l5")
  layer7_payloads, layer7_history = L7_INPUT.resolve_full_attention_payloads(
      all_history_json, layer2
  )
  CORE.add_layer7_tail_payloads(layer7_payloads)
  L8.add_layer7_ffn_payloads(layer7_payloads)
  layer11_payloads, layer11_history = resolve_prefixed_full_attention_payloads(
      all_history_json, layer6, "l6_"
  )
  payloads = {
      **payloads0,
      **payloads1,
      **layer7_payloads,
      **payloads2,
      **payloads3,
      **payloads4,
      **layer11_payloads,
  }
  opencl_source = OPENCL_SOURCE.read_text(encoding="utf-8")
  opencl_source_hash = iq36_local.sha256_file(OPENCL_SOURCE)
  embedded_opencl_hash = hashlib.sha256(
      (
          opencl_source
          + PRECONV.POSTCONV.ATTENTION.ATTENTION_EXTRA_OPENCL
          + CORE.FULL_ATTN_CORE_EXTRA_OPENCL
      ).encode("utf-8")
  ).hexdigest()
  local_cpp = out_dir / "gpu_resident_layer11_full_attn_handoff_probe.cpp"
  local_cpp.write_text(layer11_full_attn_probe_cpp(opencl_source), encoding="utf-8")

  remote_dir = f"{args.remote_root.rstrip('/')}/gpu-resident-layer11-full-attn-handoff-probe-{stamp}"
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
            f"{remote_dir}/tests/gpu_resident_layer11_full_attn_handoff_probe.cpp",
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

  executable = f"{remote_dir}/build/iq36-gpu-resident-layer11-full-attn-handoff-probe"
  compile_cmd = " && ".join([
      f"source {shlex.quote(args.env_script)} >/dev/null 2>&1",
      (
          "g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic "
          f"-I {shlex.quote(remote_dir + '/include')} "
          f"{shlex.quote(remote_dir + '/src/gguf_loader.cpp')} "
          f"{shlex.quote(remote_dir + '/src/gpu_q4x8_matvec.cpp')} "
          f"{shlex.quote(remote_dir + '/tests/gpu_resident_layer11_full_attn_handoff_probe.cpp')} "
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
      {"name": "arc_b390_selected", "pass": bool(probe and "B390" in str(probe.get("device_name", "")) and "B390" in str(probe.get("layer11_v_projection_device_name", "")))},
      {"name": "resident_api_fields_present", "pass": resident_fields_ok(probe, args.resident_invocations)},
      {"name": "layer10_full_shell_handoff_matches", "pass": PRECONV.nested_bool(probe, "checks", "layer10_full_shell_handoff_matches")},
      {"name": "layer11_residual_input_from_layer10_live_gpu_lout", "pass": PRECONV.nested_bool(probe, "checks", "layer11_residual_input_from_layer10_live_gpu_lout")},
      {"name": "layer11_payload_counts_ok", "pass": PRECONV.nested_bool(probe, "checks", "layer11_payload_counts_ok")},
      {"name": "layer11_full_attn_input_matches_oracle", "pass": PRECONV.nested_bool(probe, "checks", "layer11_full_attn_input_matches_oracle")},
      {"name": "layer11_full_attn_component_policy_matches_oracle", "pass": PRECONV.nested_bool(probe, "checks", "layer11_full_attn_component_policy_matches_oracle")},
      {"name": "layer11_full_attn_core_output_matches_oracle", "pass": PRECONV.nested_bool(probe, "checks", "layer11_full_attn_core_output_matches_oracle")},
      {"name": "layer11_v_gpu_boundary", "pass": PRECONV.nested_bool(probe, "checks", "layer11_v_gpu_boundary")},
      {"name": "l6_v_matches_oracle", "pass": CORE.comparison_passed(probe, "l6_v", "gpu_vs_oracle")},
      {"name": "l6_attn_pregate_matches_oracle", "pass": CORE.full_attention_comparison_passed(probe, "l6_attn_pregate", "gpu_vs_oracle")},
      {"name": "l6_attn_output_matches_oracle", "pass": CORE.full_attention_comparison_passed(probe, "l6_attn_output", "gpu_vs_oracle")},
      {"name": "l6_attn_post_norm_matches_oracle", "pass": CORE.full_attention_comparison_passed(probe, "l6_attn_post_norm", "gpu_vs_oracle")},
      {"name": "gpu_event_timing_positive", "pass": PRECONV.nested_bool(probe, "checks", "gpu_event_timing_positive")},
      {"name": "speedup_claims_forbidden", "pass": True},
  ]
  required_checks_passed = all(item["pass"] for item in checks)
  slim_payloads = {
      name: {key: value for key, value in payload.items() if key != "local_path"}
      for name, payload in payloads.items()
  }
  comparison_thresholds = {
      "strict_component": CORE.STRICT_COMPARISON_THRESHOLDS,
      "full_attn_component": CORE.FULL_ATTN_COMPARISON_THRESHOLDS,
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
          "layer3": str(conv2_path.relative_to(ROOT)),
          "layer4": str(conv3_path.relative_to(ROOT)),
          "layer5": str(conv4_path.relative_to(ROOT)),
      },
      "conv_history_capture_artifacts": {
          "layer0": conv0.get("capture_artifact"),
          "layer1": conv1.get("capture_artifact"),
          "layer3": conv2.get("capture_artifact"),
          "layer4": conv3.get("capture_artifact"),
          "layer5": conv4.get("capture_artifact"),
      },
      "all_history": {
          "layer7": layer7_history,
          "layer11": layer11_history,
      },
      "payloads": slim_payloads,
      "layers": [layer0, layer1, layer2, layer3, layer4, layer5, layer6],
      "resident_invocations": args.resident_invocations,
      "opencl_source": str(OPENCL_SOURCE.relative_to(ROOT)),
      "opencl_source_sha256": opencl_source_hash,
      "embedded_opencl_source_sha256": embedded_opencl_hash,
      "comparison_thresholds": comparison_thresholds,
      "probe_extra_opencl": [
          "rms_norm_hidden_f32",
          "full_attn_core_f32",
          "full_attn_gate_f32",
          "q6k_selected_down_matvec_row",
      ],
      "probe": probe,
      "checks": checks,
      "required_checks_passed": required_checks_passed,
      "speedup_claims_allowed": False,
  }
  manifest = {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "created_at": created_at,
      "tool": "tools/intel-qwen36-gpu-resident-layer11-full-attn-handoff-probe.py",
      "artifact": str(out_dir),
      "host": args.host,
      "remote_dir": remote_dir,
      "layers": payload["layers"],
      "resident_invocations": args.resident_invocations,
      "conv_history_probes": payload["conv_history_probes"],
      "all_history": payload["all_history"],
      "comparison_thresholds": comparison_thresholds,
      "required_checks_passed": required_checks_passed,
      "speedup_claims_allowed": False,
  }
  correctness = {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "comparison_thresholds": comparison_thresholds,
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
      "gpu_resident_layer11_full_attn_handoff_probe",
      [
          ("required_checks_passed", required_checks_passed),
          ("resident_shell_invocations", args.resident_invocations),
          ("resident_layer11_full_attn_input_kernel_sum_min_us", PRECONV.nested_number(timings, "resident_layer11_full_attn_input_kernel_sum_min_us")),
          ("resident_layer11_full_attn_core_output_kernel_sum_min_us", PRECONV.nested_number(timings, "resident_layer11_full_attn_core_output_kernel_sum_min_us")),
          ("resident_layer11_full_attention_kernel_sum_min_us", PRECONV.nested_number(timings, "resident_layer11_full_attention_kernel_sum_min_us")),
          ("l6_residual_input_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l6_residual_input", "gpu_vs_oracle", "max_abs_diff")),
          ("l6_v_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l6_v", "gpu_vs_oracle", "max_abs_diff")),
          ("l6_attn_output_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l6_attn_output", "gpu_vs_oracle", "max_abs_diff")),
          ("l6_attn_post_norm_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l6_attn_post_norm", "gpu_vs_oracle", "max_abs_diff")),
      ],
  )
  write_summary(out_dir / "summary.md", payload)
  print(out_dir)
  return 0 if required_checks_passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
