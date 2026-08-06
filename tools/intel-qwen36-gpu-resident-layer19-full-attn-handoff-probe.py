#!/usr/bin/env python3
"""Run the resident GPU layer-19 full-attention handoff probe."""

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
L18_TOOL = Path(__file__).with_name(
    "intel-qwen36-gpu-resident-layer18-state-input-handoff-probe.py"
)
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA_VERSION = "intel-qwen36-gpu-resident-layer19-full-attn-handoff-probe-v0"
DEFAULT_HOST = "local"
DEFAULT_MODEL = "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf"
DEFAULT_ENV_SCRIPT = "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh"
DEFAULT_REMOTE_ROOT = "/home/intel/intel-qwen36-gpu"
DEFAULT_ORACLE_BUNDLE = ROOT / "oracle/r0-oracle-bundle-20260627T060028Z"
DEFAULT_ALL_HISTORY = ROOT / "output/r1-full-attn-all-history-capture-20260627T145615Z/history.json"
OPENCL_SOURCE = ROOT / "engine/gpu/opencl/q4x8_matvec.cl"


def load_l18_tool() -> ModuleType:
  spec = importlib.util.spec_from_file_location("iq36_layer18_state_input_probe", L18_TOOL)
  if spec is None or spec.loader is None:
    raise SystemExit(f"failed to load layer18 state/input tool: {L18_TOOL}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


L18 = load_l18_tool()
L17 = L18.L17
L16 = L18.L16
L15 = L18.L15
CORE = L18.CORE
L7_INPUT = L18.L7_INPUT
L8 = L18.L8
L11 = L18.L11
L11_FFN = L18.L11_FFN
L11_FULL = L18.L11_FULL
TWO = L18.TWO
PRECONV = L18.PRECONV


def replace_once(text: str, old: str, new: str) -> str:
  return L18.replace_once(text, old, new)


def layer19_full_attn_probe_cpp(opencl_source: str) -> str:
  cpp = L18.layer18_state_input_probe_cpp(opencl_source)
  for old, new in {
      L18.SCHEMA_VERSION: SCHEMA_VERSION,
      "layer5_to_layer18_state_input_load_once_run_many":
          "layer5_to_layer19_full_attention_load_once_run_many",
      "layer18 state/input handoff probe expects --layer 5":
          "layer19 full-attention handoff probe expects --layer 5",
  }.items():
    cpp = replace_once(cpp, old, new)
  cpp = replace_once(
      cpp,
      '''    const int layer12 = args.layer + 12;
    const int layer13 = args.layer + 13;
    Require(layer0 == 5 && layer1 == 6 && layer2 == 7 && layer3 == 8 && layer4 == 9 && layer5 == 10 && layer6 == 11 && layer7 == 12 && layer8 == 13 && layer9 == 14 && layer10 == 15 && layer11 == 16 && layer12 == 17 && layer13 == 18,
            "layer19 full-attention handoff probe expects --layer 5");
''',
      '''    const int layer12 = args.layer + 12;
    const int layer13 = args.layer + 13;
    const int layer14 = args.layer + 14;
    Require(layer0 == 5 && layer1 == 6 && layer2 == 7 && layer3 == 8 && layer4 == 9 && layer5 == 10 && layer6 == 11 && layer7 == 12 && layer8 == 13 && layer9 == 14 && layer10 == 15 && layer11 == 16 && layer12 == 17 && layer13 == 18 && layer14 == 19,
            "layer19 full-attention handoff probe expects --layer 5");
''',
  )
  cpp = replace_once(
      cpp,
      '''    const auto layer12_tensors = ResolveLayerTensorBundle(index, layer12);
    const auto layer13_tensors = ResolveLayerTensorBundle(index, layer13);
''',
      '''    const auto layer12_tensors = ResolveLayerTensorBundle(index, layer12);
    const auto layer13_tensors = ResolveLayerTensorBundle(index, layer13);
    const auto layer14_tensors = ResolveFullAttentionTensorBundle(index, layer14);
''',
  )
  cpp = replace_once(
      cpp,
      '''    const auto layer12_oracle = LoadLayerOraclePayloads(args.payload_dir, "l17");
    const auto layer13_oracle = LoadLayerOraclePayloads(args.payload_dir, "l18");
''',
      '''    const auto layer12_oracle = LoadLayerOraclePayloads(args.payload_dir, "l17");
    const auto layer13_oracle = LoadLayerOraclePayloads(args.payload_dir, "l18");
    const auto layer14_oracle = LoadFullAttentionPayloadsPrefixed(args.payload_dir, "l19_");
''',
  )
  cpp = replace_once(
      cpp,
      '''    const auto oracle_layer_output =
        iq36::add_vectors(oracle_attn_residual, oracle_ffn_out);
''',
      '''    const auto oracle_layer_output =
        iq36::add_vectors(oracle_attn_residual, oracle_ffn_out);
    const auto layer14_oracle_attn_residual =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "l19_full_attn_residual.bin"));
    const auto layer14_oracle_attn_post_norm =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "l19_full_attn_post_norm.bin"));
    const bool layer14_payload_counts_ok =
        FullAttentionPayloadCountsOk(layer14_oracle) &&
        layer14_oracle_attn_residual.size() == kHiddenSize &&
        layer14_oracle_attn_post_norm.size() == kHiddenSize;
''',
  )
  cpp = replace_once(
      cpp,
      '''    const auto layer10_full_shapes = CheckFullAttentionShapes(layer10_tensors);
    const bool layer10_full_shapes_ok = FullAttentionShapesPassed(layer10_full_shapes);
''',
      '''    const auto layer10_full_shapes = CheckFullAttentionShapes(layer10_tensors);
    const bool layer10_full_shapes_ok = FullAttentionShapesPassed(layer10_full_shapes);
    const auto layer14_full_shapes = CheckFullAttentionShapes(layer14_tensors);
    const bool layer14_full_shapes_ok = FullAttentionShapesPassed(layer14_full_shapes);
''',
  )
  cpp = replace_once(
      cpp,
      '''    const std::string selected_gate_up_tensor_name =
''',
      '''    const auto layer14_attn_norm_weight =
        ReadF32TensorPayload(model, *layer14_tensors.attn_norm_tensor,
                             static_cast<std::size_t>(kHiddenSize));
    const auto layer14_q_norm_weight =
        iq36::decode_tensor_row(args.model_path, index,
                                layer14_tensors.q_norm_tensor_name, 0);
    const auto layer14_k_norm_weight =
        iq36::decode_tensor_row(args.model_path, index,
                                layer14_tensors.k_norm_tensor_name, 0);
    const auto layer14_ffn_norm_weight =
        iq36::decode_tensor_row(args.model_path, index,
                                LayerTensorName(layer14, "post_attention_norm.weight"), 0);
    const std::string selected_gate_up_tensor_name =
''',
  )
  cpp = replace_once(
      cpp,
      '''    const auto layer13_run = RunResidentLinearLayerShell(
        args, index, layer13_tensors, layer13_oracle,
        layer12_run.gpu_layer_output, layer12_run.gpu_layer_output, rms_norm_epsilon);

    std::vector<NamedCompareGroup> strict_groups;
''',
      '''    const auto layer13_run = RunResidentLinearLayerShell(
        args, index, layer13_tensors, layer13_oracle,
        layer12_run.gpu_layer_output, layer12_run.gpu_layer_output, rms_norm_epsilon);

    const auto layer14_native_qkv = iq36::run_qwen36_full_attention_qkv_projection(
        args.model_path,
        index,
        layer14,
        layer13_run.gpu_layer_output,
        rms_norm_epsilon);
    const auto layer14_rms_gpu = RunGpuLayerInputRmsNorm(
        layer13_run.gpu_layer_output,
        layer14_attn_norm_weight,
        rms_norm_epsilon,
        args.device_substring,
        args.repeat);
    const auto layer14_qk_gpu = RunGpuFullAttentionQkFront(
        args.model_path,
        *layer14_tensors.q_tensor,
        *layer14_tensors.k_tensor,
        layer14_rms_gpu.attn_norm,
        args.device_substring,
        args.repeat);
    const auto layer14_v_gpu = RunGpuFullAttentionVAny(
        args.model_path,
        *layer14_tensors.v_tensor,
        layer14_rms_gpu.attn_norm,
        args.device_substring,
        args.repeat);
    const auto layer14_gpu_q_split = SplitFullAttentionQ(layer14_qk_gpu.q_full);
    const auto layer14_gpu_q_normed = ApplyRepeatedRmsNormFull(
        layer14_gpu_q_split.q_raw, layer14_q_norm_weight, rms_norm_epsilon);
    const auto layer14_gpu_k_normed = ApplyRepeatedRmsNormFull(
        layer14_qk_gpu.k_raw, layer14_k_norm_weight, rms_norm_epsilon);
    const auto layer14_gpu_rope = iq36::run_qwen36_full_attention_rope(
        layer14_gpu_q_normed,
        layer14_gpu_k_normed,
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
    const auto layer14_native_rope = iq36::run_qwen36_full_attention_rope(
        layer14_native_qkv.q_normed,
        layer14_native_qkv.k_normed,
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
    auto layer14_native_k_history = layer14_oracle.k_history;
    auto layer14_native_v_history = layer14_oracle.v_history;
    layer14_native_k_history.push_back(layer14_oracle.k_rope);
    layer14_native_v_history.push_back(layer14_oracle.v);
    const auto layer14_native_core = iq36::run_qwen36_full_attention_core(
        layer14_oracle.q_rope,
        layer14_native_k_history,
        layer14_native_v_history,
        head_dim,
        q_head_count,
        kv_head_count,
        kAttentionScale);
    const auto layer14_native_gate = iq36::run_qwen36_full_attention_gate(
        layer14_oracle.q_full, layer14_native_core.attn_pregate, head_dim);
    const auto layer14_native_attn_output = iq36::matvec_tensor(
        args.model_path,
        index,
        LayerTensorName(layer14, "attn_output.weight"),
        layer14_native_gate.attn_gated);
    const auto layer14_native_attn_residual =
        iq36::add_vectors(layer13_run.gpu_layer_output, layer14_native_attn_output);
    const auto layer14_native_attn_post_norm =
        iq36::apply_rms_norm(layer14_native_attn_residual,
                             layer14_ffn_norm_weight,
                             rms_norm_epsilon);

    auto layer14_gpu_k_history = layer14_oracle.k_history;
    auto layer14_gpu_v_history = layer14_oracle.v_history;
    layer14_gpu_k_history.push_back(layer14_oracle.k_rope);
    layer14_gpu_v_history.push_back(layer14_v_gpu.v);
    std::vector<float> layer14_gpu_k_history_flat;
    std::vector<float> layer14_gpu_v_history_flat;
    for (const auto& item : layer14_gpu_k_history) {
      layer14_gpu_k_history_flat.insert(layer14_gpu_k_history_flat.end(), item.begin(), item.end());
    }
    for (const auto& item : layer14_gpu_v_history) {
      layer14_gpu_v_history_flat.insert(layer14_gpu_v_history_flat.end(), item.begin(), item.end());
    }
    const auto layer14_core_gate_gpu = RunGpuFullAttentionCoreGate(
        layer14_oracle.q_rope,
        layer14_gpu_k_history_flat,
        layer14_gpu_v_history_flat,
        layer14_qk_gpu.q_full,
        args.device_substring,
        args.repeat);
    const auto layer14_attention_gpu = RunGpuAttentionFront(
        args.model_path,
        *iq36::find_tensor(index, LayerTensorName(layer14, "attn_output.weight")),
        layer14_core_gate_gpu.attn_gated,
        layer13_run.gpu_layer_output,
        layer14_ffn_norm_weight,
        rms_norm_epsilon,
        args.device_substring,
        args.repeat);

    std::vector<NamedCompareGroup> strict_groups;
''',
  )
  cpp = replace_once(
      cpp,
      '''    const bool layer13_ok =
        layer13_shapes_ok &&
        layer13_run.payload_counts_ok &&
        layer13_state_input_gpu_cpu_ok &&
        layer13_state_input_oracle_policy_ok &&
        layer13_state_input_timing_positive &&
        layer13_run.arc_selected &&
        layer13_qkv_q4_q6_boundary &&
        layer13_down_q4_q6_boundary;
    const bool layer2_timing_positive =
''',
      '''    const bool layer13_ok =
        layer13_shapes_ok &&
        layer13_run.payload_counts_ok &&
        layer13_state_input_gpu_cpu_ok &&
        layer13_state_input_oracle_policy_ok &&
        layer13_state_input_timing_positive &&
        layer13_run.arc_selected &&
        layer13_qkv_q4_q6_boundary &&
        layer13_down_q4_q6_boundary;
    std::vector<NamedCompareGroup> layer14_strict_groups;
    AppendCpuGpuOracleCompare(layer14_strict_groups, "l19_residual_input",
                              layer13_run.gpu_layer_output,
                              layer13_run.gpu_layer_output,
                              layer14_oracle.residual_input);
    AppendCpuGpuOracleCompare(layer14_strict_groups, "l19_attn_norm",
                              layer14_native_qkv.attention_norm,
                              layer14_rms_gpu.attn_norm,
                              layer14_oracle.attn_norm);
    AppendCpuGpuOracleCompare(layer14_strict_groups, "l19_q_full",
                              layer14_native_qkv.q_full,
                              layer14_qk_gpu.q_full,
                              layer14_oracle.q_full);
    AppendCpuGpuOracleCompare(layer14_strict_groups, "l19_q_rope",
                              layer14_native_rope.q_rope,
                              layer14_gpu_rope.q_rope,
                              layer14_oracle.q_rope);
    AppendCpuGpuOracleCompare(layer14_strict_groups, "l19_k_rope",
                              layer14_native_rope.k_rope,
                              layer14_gpu_rope.k_rope,
                              layer14_oracle.k_rope);
    AppendCpuGpuOracleCompare(layer14_strict_groups, "l19_v",
                              layer14_native_qkv.v,
                              layer14_v_gpu.v,
                              layer14_oracle.v);
    std::vector<NamedCompareGroup> layer14_full_attention_groups;
    AppendFullAttentionComponentCompare(layer14_full_attention_groups, "l19_attn_pregate",
                                        layer14_native_core.attn_pregate,
                                        layer14_core_gate_gpu.attn_pregate,
                                        layer14_oracle.attn_pregate);
    AppendFullAttentionComponentCompare(layer14_full_attention_groups, "l19_attn_gated",
                                        layer14_native_gate.attn_gated,
                                        layer14_core_gate_gpu.attn_gated,
                                        layer14_oracle.attn_gated);
    AppendFullAttentionComponentCompare(layer14_full_attention_groups, "l19_attn_output",
                                        layer14_native_attn_output,
                                        layer14_attention_gpu.linear_attn_out,
                                        layer14_oracle.attn_output);
    AppendFullAttentionComponentCompare(layer14_full_attention_groups, "l19_attn_residual",
                                        layer14_native_attn_residual,
                                        layer14_attention_gpu.attn_residual,
                                        layer14_oracle_attn_residual);
    AppendFullAttentionComponentCompare(layer14_full_attention_groups, "l19_attn_post_norm",
                                        layer14_native_attn_post_norm,
                                        layer14_attention_gpu.attn_post_norm,
                                        layer14_oracle_attn_post_norm);
    const auto layer14_k_raw_gpu_vs_cpu = iq36::compare_vectors(
        layer14_qk_gpu.k_raw, layer14_native_qkv.k_raw, kMismatchThreshold);
    const bool layer14_strict_input_ok =
        CompareGroupsPassed(layer14_strict_groups) &&
        ComparePassed(layer14_k_raw_gpu_vs_cpu);
    bool layer14_full_component_ok = true;
    for (const auto& group : layer14_full_attention_groups) {
      layer14_full_component_ok =
          layer14_full_component_ok &&
          ComparePassedFullAttentionComponent(group.gpu_vs_oracle);
    }
    const bool layer14_comparisons_ok =
        layer14_strict_input_ok && layer14_full_component_ok;
    const bool layer14_timing_positive =
        layer14_rms_gpu.timing.rmsnorm_min_us > 0.0 &&
        layer14_qk_gpu.timing.q_projection_min_us > 0.0 &&
        layer14_qk_gpu.timing.k_projection_min_us > 0.0 &&
        layer14_v_gpu.timing.v_projection_min_us > 0.0 &&
        layer14_core_gate_gpu.timing.core_min_us > 0.0 &&
        layer14_core_gate_gpu.timing.gate_min_us > 0.0 &&
        layer14_attention_gpu.timing.output_projection_min_us > 0.0 &&
        layer14_attention_gpu.timing.residual_add_min_us > 0.0 &&
        layer14_attention_gpu.timing.ffn_rmsnorm_min_us > 0.0;
    const bool layer14_arc_selected =
        layer14_rms_gpu.device_name.find(args.device_substring) != std::string::npos &&
        layer14_qk_gpu.device_name.find(args.device_substring) != std::string::npos &&
        layer14_v_gpu.device_name.find(args.device_substring) != std::string::npos &&
        layer14_core_gate_gpu.device_name.find(args.device_substring) != std::string::npos &&
        layer14_attention_gpu.device_name.find(args.device_substring) != std::string::npos;
    const bool layer14_v_gpu_boundary =
        (layer14_tensors.v_tensor->type == 12 || layer14_tensors.v_tensor->type == 14) &&
        layer14_v_gpu.v.size() == kFullKvValues;
    const bool layer14_ffn_q6_boundary =
        iq36::find_tensor(index, LayerTensorName(layer14, "ffn_down_exps.weight"))->type == 14 &&
        iq36::find_tensor(index, LayerTensorName(layer14, "ffn_down_shexp.weight"))->type == 14;
    const bool layer14_ok =
        layer14_full_shapes_ok &&
        layer14_payload_counts_ok &&
        metadata_ok &&
        layer14_comparisons_ok &&
        layer14_timing_positive &&
        layer14_arc_selected &&
        layer14_v_gpu_boundary;
    const bool layer2_timing_positive =
''',
  )
  cpp = replace_once(
      cpp,
      '''        layer11_ok &&
        layer12_ok &&
        layer13_ok &&
        layer2_timing_positive &&
''',
      '''        layer11_ok &&
        layer12_ok &&
        layer13_ok &&
        layer14_ok &&
        layer2_timing_positive &&
''',
  )
  cpp = replace_once(
      cpp,
      '''    const double layer13_state_input_sum_min =
        layer13_run.timing.layer_input_rmsnorm_min_us +
        layer13_run.timing.preconv_to_postconv_kernel_sum_min_us;
''',
      '''    const double layer13_state_input_sum_min =
        layer13_run.timing.layer_input_rmsnorm_min_us +
        layer13_run.timing.preconv_to_postconv_kernel_sum_min_us;
    const double layer14_input_sum_min =
        layer14_rms_gpu.timing.rmsnorm_min_us +
        layer14_qk_gpu.timing.qk_projection_kernel_sum_min_us +
        layer14_v_gpu.timing.v_projection_min_us;
    const double layer14_core_output_sum_min =
        layer14_core_gate_gpu.timing.core_gate_kernel_sum_min_us +
        layer14_attention_gpu.timing.attention_front_kernel_sum_min_us;
    const double layer14_attention_sum_min =
        layer14_input_sum_min + layer14_core_output_sum_min;
''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << "\\"layers\\":[" << layer0 << "," << layer1 << "," << layer2 << "," << layer3 << "," << layer4 << "," << layer5 << "," << layer6 << "," << layer7 << "," << layer8 << "," << layer9 << "," << layer10 << "," << layer11 << "," << layer12 << "," << layer13 << "],";
''',
      '''    std::cout << "\\"layers\\":[" << layer0 << "," << layer1 << "," << layer2 << "," << layer3 << "," << layer4 << "," << layer5 << "," << layer6 << "," << layer7 << "," << layer8 << "," << layer9 << "," << layer10 << "," << layer11 << "," << layer12 << "," << layer13 << "," << layer14 << "],";
''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << "\\"layer18_conv_state_boundary\\":\\"captured_conv_state\\",";
    std::cout << "\\"full_attn_core_input_boundary\\":\\"captured_rope_kv_payloads\\",";''',
      '''    std::cout << "\\"layer18_conv_state_boundary\\":\\"captured_conv_state\\",";
    std::cout << "\\"layer19_residual_input_boundary\\":\\"live_gpu_l_out_18\\",";
    std::cout << "\\"layer19_v_tensor_type\\":\\""
              << JsonEscape(iq36::ggml_type_name(layer14_tensors.v_tensor->type)) << "\\",";
    std::cout << "\\"layer19_v_projection_boundary\\":\\""
              << (layer14_tensors.v_tensor->type == 14 ? "gpu_q6_raw_matvec" : "gpu_q4x8_matvec") << "\\",";
    std::cout << "\\"layer19_ffn_boundary\\":\\"q6_down_pending\\",";
    std::cout << "\\"full_attn_core_input_boundary\\":\\"captured_rope_kv_payloads\\",";''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << "\\"layer18_tail_device_name\\":\\"" << JsonEscape(layer13_run.tail_device_name) << "\\",";
    std::cout << "\\"layer7_v_tensor_type\\":\\""
''',
      '''    std::cout << "\\"layer18_tail_device_name\\":\\"" << JsonEscape(layer13_run.tail_device_name) << "\\",";
    std::cout << "\\"layer19_core_gate_device_name\\":\\"" << JsonEscape(layer14_core_gate_gpu.device_name) << "\\",";
    std::cout << "\\"layer19_output_projection_device_name\\":\\"" << JsonEscape(layer14_attention_gpu.device_name) << "\\",";
    std::cout << "\\"layer19_v_projection_device_name\\":\\"" << JsonEscape(layer14_v_gpu.device_name) << "\\",";
    std::cout << "\\"layer7_v_tensor_type\\":\\""
''',
  )
  cpp = replace_once(
      cpp,
      '''                  layer12_run.program_build_ms +
                  layer13_run.program_build_ms)
''',
      '''                  layer12_run.program_build_ms +
                  layer13_run.program_build_ms +
                  layer14_rms_gpu.program_build_ms + layer14_qk_gpu.program_build_ms +
                  layer14_v_gpu.program_build_ms +
                  layer14_core_gate_gpu.program_build_ms +
                  layer14_attention_gpu.program_build_ms)
''',
  )
  cpp = replace_once(
      cpp,
      '''                            layer12_run.build_log +
                            layer13_run.build_log)
''',
      '''                            layer12_run.build_log +
                            layer13_run.build_log +
                            layer14_rms_gpu.build_log + layer14_qk_gpu.build_log +
                            layer14_v_gpu.build_log +
                            layer14_core_gate_gpu.build_log +
                            layer14_attention_gpu.build_log)
''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << ",\\"layer10_output_front\\":";
    WriteFullAttentionOutputFrontTiming(layer10_attention_gpu.timing);
''',
      '''    std::cout << ",\\"layer10_output_front\\":";
    WriteFullAttentionOutputFrontTiming(layer10_attention_gpu.timing);
    std::cout << ",\\"layer14_full_attn_input\\":";
    WriteFullAttentionTiming(layer14_rms_gpu, layer14_qk_gpu);
    std::cout << ",\\"layer14_v_projection\\":";
    WriteFullAttentionVQ6Timing(layer14_v_gpu.timing);
    std::cout << ",\\"layer14_core_gate\\":";
    WriteFullCoreGateTiming(layer14_core_gate_gpu.timing);
    std::cout << ",\\"layer14_output_front\\":";
    WriteFullAttentionOutputFrontTiming(layer14_attention_gpu.timing);
''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << "\\"resident_layer5_6_7_8_9_10_11_12_13_14_15_16_17_to_layer18_state_input_kernel_sum_min_us\\":"
              << (two_layer_kernel_sum_min + layer2_attention_sum_min +
                  layer2_ffn_sum_min + layer3_run.timing.layer_kernel_sum_min_us +
                  layer4_run.timing.layer_kernel_sum_min_us +
                  layer5_run.timing.layer_kernel_sum_min_us +
                  layer6_attention_sum_min + layer6_ffn_sum_min +
                  layer7_state_input_sum_min + layer8_state_input_sum_min +
                  layer9_state_input_sum_min + layer10_attention_sum_min +
                  layer10_ffn_sum_min + layer11_state_input_sum_min +
                  layer12_state_input_sum_min + layer13_state_input_sum_min);
''',
      '''    std::cout << "\\"resident_layer5_6_7_8_9_10_11_12_13_14_15_16_17_to_layer18_state_input_kernel_sum_min_us\\":"
              << (two_layer_kernel_sum_min + layer2_attention_sum_min +
                  layer2_ffn_sum_min + layer3_run.timing.layer_kernel_sum_min_us +
                  layer4_run.timing.layer_kernel_sum_min_us +
                  layer5_run.timing.layer_kernel_sum_min_us +
                  layer6_attention_sum_min + layer6_ffn_sum_min +
                  layer7_state_input_sum_min + layer8_state_input_sum_min +
                  layer9_state_input_sum_min + layer10_attention_sum_min +
                  layer10_ffn_sum_min + layer11_state_input_sum_min +
                  layer12_state_input_sum_min + layer13_state_input_sum_min) << ",";
    std::cout << "\\"resident_layer19_full_attn_input_kernel_sum_min_us\\":"
              << layer14_input_sum_min << ",";
    std::cout << "\\"resident_layer19_full_attn_core_output_kernel_sum_min_us\\":"
              << layer14_core_output_sum_min << ",";
    std::cout << "\\"resident_layer19_full_attention_kernel_sum_min_us\\":"
              << layer14_attention_sum_min << ",";
    std::cout << "\\"resident_layer5_6_7_8_9_10_11_12_13_14_15_16_17_18_to_layer19_full_attention_kernel_sum_min_us\\":"
              << (two_layer_kernel_sum_min + layer2_attention_sum_min +
                  layer2_ffn_sum_min + layer3_run.timing.layer_kernel_sum_min_us +
                  layer4_run.timing.layer_kernel_sum_min_us +
                  layer5_run.timing.layer_kernel_sum_min_us +
                  layer6_attention_sum_min + layer6_ffn_sum_min +
                  layer7_state_input_sum_min + layer8_state_input_sum_min +
                  layer9_state_input_sum_min + layer10_attention_sum_min +
                  layer10_ffn_sum_min + layer11_state_input_sum_min +
                  layer12_state_input_sum_min + layer13_state_input_sum_min +
                  layer14_attention_sum_min);
''',
  )
  cpp = replace_once(
      cpp,
      '''    WritePrefixedCompareGroups("l18", layer13_run.comparisons, &first_compare);
    std::cout << ",\\"l2_k_raw\\":{\\"gpu_vs_cpu\\":";
''',
      '''    WritePrefixedCompareGroups("l18", layer13_run.comparisons, &first_compare);
    std::cout << ",";
    WriteNamedCompareGroups(layer14_strict_groups);
    std::cout << ",\\"l19_k_raw\\":{\\"gpu_vs_cpu\\":";
    WriteCompare(layer14_k_raw_gpu_vs_cpu);
    std::cout << "},";
    WriteNamedCompareGroups(layer14_full_attention_groups);
    std::cout << ",\\"l2_k_raw\\":{\\"gpu_vs_cpu\\":";
''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << ",\\"layer13\\":";
    WriteLayerChecks(layer13_run);
    std::cout << ",\\"layer1_residual_input_matches_oracle\\":"
''',
      '''    std::cout << ",\\"layer13\\":";
    WriteLayerChecks(layer13_run);
    std::cout << ",\\"layer14_full_attention_shapes\\":";
    WriteFullAttentionShapeChecks(layer14_full_shapes);
    std::cout << ",\\"layer1_residual_input_matches_oracle\\":"
''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << "\\"layer18_state_input_handoff_matches\\":"
              << (layer13_ok ? "true" : "false") << ",";
    std::cout << "\\"history_kv_state_payloads_present\\":"
''',
      '''    std::cout << "\\"layer18_state_input_handoff_matches\\":"
              << (layer13_ok ? "true" : "false") << ",";
    std::cout << "\\"layer19_residual_input_from_layer18_live_gpu_lout\\":true,";
    std::cout << "\\"layer19_payload_counts_ok\\":"
              << (layer14_payload_counts_ok ? "true" : "false") << ",";
    std::cout << "\\"layer19_full_attn_input_matches_oracle\\":"
              << (layer14_strict_input_ok ? "true" : "false") << ",";
    std::cout << "\\"layer19_full_attn_component_policy_matches_oracle\\":"
              << (layer14_full_component_ok ? "true" : "false") << ",";
    std::cout << "\\"layer19_full_attn_core_output_matches_oracle\\":"
              << (layer14_comparisons_ok ? "true" : "false") << ",";
    std::cout << "\\"layer19_v_gpu_boundary\\":"
              << (layer14_v_gpu_boundary ? "true" : "false") << ",";
    std::cout << "\\"layer19_ffn_q6_boundary\\":"
              << (layer14_ffn_q6_boundary ? "true" : "false") << ",";
    std::cout << "\\"layer19_arc_device_selected\\":"
              << (layer14_arc_selected ? "true" : "false") << ",";
    std::cout << "\\"history_kv_state_payloads_present\\":"
''',
  )
  cpp = replace_once(
      cpp,
      '''                  layer12_state_input_timing_positive &&
                  layer13_state_input_timing_positive ? "true" : "false") << ",";
''',
      '''                  layer12_state_input_timing_positive &&
                  layer13_state_input_timing_positive &&
                  layer14_timing_positive ? "true" : "false") << ",";
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
  parser.add_argument("--layer12-conv-history-probe", type=Path, default=None)
  parser.add_argument("--layer13-conv-history-probe", type=Path, default=None)
  parser.add_argument("--layer14-conv-history-probe", type=Path, default=None)
  parser.add_argument("--layer16-conv-history-probe", type=Path, default=None)
  parser.add_argument("--layer17-conv-history-probe", type=Path, default=None)
  parser.add_argument("--layer18-conv-history-probe", type=Path, default=None)
  parser.add_argument("--out-dir", type=Path, default=None)
  return parser.parse_args()


def resident_fields_ok(
    probe: dict[str, Any] | None,
    expected_invocations: int,
    device_substring: str,
) -> bool:
  return (
      isinstance(probe, dict)
      and probe.get("resident_api") == "layer5_to_layer19_full_attention_load_once_run_many"
      and probe.get("resident_load_count") == 1
      and probe.get("resident_shell_invocations") == expected_invocations
      and probe.get("layer18_residual_input_boundary") == "live_gpu_l_out_17"
      and probe.get("layer19_residual_input_boundary") == "live_gpu_l_out_18"
      and probe.get("layer19_v_tensor_type") in {"Q4_K", "Q6_K"}
      and probe.get("layer19_v_projection_boundary") in {"gpu_q4x8_matvec", "gpu_q6_raw_matvec"}
      and probe.get("layer19_ffn_boundary") == "q6_down_pending"
      and device_substring in str(probe.get("layer19_core_gate_device_name", ""))
      and device_substring in str(probe.get("layer19_output_projection_device_name", ""))
      and device_substring in str(probe.get("layer19_v_projection_device_name", ""))
      and PRECONV.nested_bool(probe, "checks", "resident_load_once")
      and PRECONV.nested_bool(probe, "checks", "resident_shell_invocations_positive")
  )


def layer18_predecessor_checks(probe: dict[str, Any] | None) -> list[dict[str, Any]]:
  checks = L18.layer17_predecessor_checks(probe)
  checks.extend([
      {"name": "layer18_residual_input_from_layer17_live_gpu_lout", "pass": PRECONV.nested_bool(probe, "checks", "layer18_residual_input_from_layer17_live_gpu_lout")},
      {"name": "layer18_payload_counts_ok", "pass": PRECONV.nested_bool(probe, "checks", "layer18_payload_counts_ok")},
      {"name": "layer18_shapes_ok", "pass": PRECONV.nested_bool(probe, "checks", "layer18_shapes_ok")},
      {"name": "layer18_qkv_q4_q6_boundary", "pass": PRECONV.nested_bool(probe, "checks", "layer18_qkv_q4_q6_boundary")},
      {"name": "layer18_down_q4_q6_boundary", "pass": PRECONV.nested_bool(probe, "checks", "layer18_down_q4_q6_boundary")},
      {"name": "layer18_conv_state_input_boundary", "pass": PRECONV.nested_bool(probe, "checks", "layer18_conv_state_input_boundary")},
      {"name": "layer18_gpu_cpu_matches_native", "pass": PRECONV.nested_bool(probe, "checks", "layer18_gpu_cpu_matches_native")},
      {"name": "layer18_state_input_oracle_policy_matches", "pass": PRECONV.nested_bool(probe, "checks", "layer18_state_input_oracle_policy_matches")},
      {"name": "layer18_state_input_handoff_matches", "pass": PRECONV.nested_bool(probe, "checks", "layer18_state_input_handoff_matches")},
      {"name": "l18_residual_input_matches_oracle", "pass": CORE.full_attention_comparison_passed(probe, "l18_residual_input", "gpu_vs_oracle")},
      {"name": "l18_attn_norm_matches_oracle", "pass": CORE.full_attention_comparison_passed(probe, "l18_attn_norm", "gpu_vs_oracle")},
      {"name": "l18_qkv_matches_oracle", "pass": CORE.full_attention_comparison_passed(probe, "l18_linear_attn_qkv_mixed", "gpu_vs_oracle")},
      {"name": "l18_conv_output_raw_matches_oracle", "pass": CORE.full_attention_comparison_passed(probe, "l18_conv_output_raw", "gpu_vs_oracle")},
      {"name": "l18_final_output_matches_oracle", "pass": CORE.full_attention_comparison_passed(probe, "l18_final_output", "gpu_vs_oracle")},
      {"name": "l18_layer_output_matches_oracle", "pass": CORE.full_attention_comparison_passed(probe, "l18_layer_output", "gpu_vs_oracle")},
      {"name": "l18_conv_state_after_matches_native", "pass": CORE.comparison_passed(probe, "l18_conv_state_after", "gpu_vs_cpu")},
      {"name": "l18_recurrent_state_matches_native", "pass": CORE.comparison_passed(probe, "l18_recurrent_state", "gpu_vs_cpu")},
      {"name": "layer18_internal_comparisons_passed", "pass": PRECONV.nested_bool(probe, "checks", "layer13", "comparisons_passed")},
  ])
  return checks


def write_summary(path: Path, payload: dict[str, Any]) -> None:
  probe = payload.get("probe") if isinstance(payload.get("probe"), dict) else {}
  timings = probe.get("timings", {}) if isinstance(probe, dict) else {}
  comparisons = probe.get("comparisons", {}) if isinstance(probe, dict) else {}
  lines = [
      "# GPU Resident Layer-19 Full-Attention Handoff Probe",
      "",
      f"- workstream: `{WORKSTREAM}`",
      f"- host: `{payload.get('host')}`",
      f"- model: `{payload.get('model')}`",
      f"- layers: `{payload.get('layers')}`",
      f"- required checks passed: `{str(payload.get('required_checks_passed')).lower()}`",
      "- speedup claims allowed: `false`",
      f"- resident API: `{probe.get('resident_api')}`",
      f"- resident shell invocations: `{probe.get('resident_shell_invocations')}`",
      f"- layer 19 residual input boundary: `{probe.get('layer19_residual_input_boundary')}`",
      f"- layer 19 V tensor type: `{probe.get('layer19_v_tensor_type')}`",
      f"- layer 19 V projection boundary: `{probe.get('layer19_v_projection_boundary')}`",
      f"- layer 19 FFN boundary: `{probe.get('layer19_ffn_boundary')}`",
      "",
      "| output | comparison | max abs | RMSE |",
      "|---|---|---:|---:|",
  ]
  for name in (
      "l19_residual_input",
      "l19_attn_norm",
      "l19_q_full",
      "l19_q_rope",
      "l19_k_rope",
      "l19_v",
      "l19_attn_pregate",
      "l19_attn_output",
      "l19_attn_post_norm",
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
      f"| layer19_full_attn_input | {timings.get('resident_layer19_full_attn_input_kernel_sum_min_us') if isinstance(timings, dict) else None} |",
      f"| layer19_full_attn_core_output | {timings.get('resident_layer19_full_attn_core_output_kernel_sum_min_us') if isinstance(timings, dict) else None} |",
      f"| layer19_full_attention | {timings.get('resident_layer19_full_attention_kernel_sum_min_us') if isinstance(timings, dict) else None} |",
      f"| through_layer19_full_attention | {timings.get('resident_layer5_6_7_8_9_10_11_12_13_14_15_16_17_18_to_layer19_full_attention_kernel_sum_min_us') if isinstance(timings, dict) else None} |",
      "",
      "This target-side process carries closed live GPU layer outputs through",
      "layer 18, then feeds live GPU `l_out-18` into layer 19 RMSNorm, Q/K/V",
      "projection, ROPE, full-attention core/gate, and output projection. This is captured",
      "single-token state/input evidence, not decode throughput.",
      "",
  ]
  path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  args = parse_args()
  if args.resident_invocations <= 0:
    raise SystemExit("--resident-invocations must be positive")
  if args.layer != 5:
    raise SystemExit("--layer must be 5 for the layer-19 full-attention handoff")

  layer0 = args.layer
  layer1 = args.layer + 1
  layer2 = args.layer + 2
  layer3 = args.layer + 3
  layer4 = args.layer + 4
  layer5 = args.layer + 5
  layer6 = args.layer + 6
  layer7 = args.layer + 7
  layer8 = args.layer + 8
  layer9 = args.layer + 9
  layer10 = args.layer + 10
  layer11 = args.layer + 11
  layer12 = args.layer + 12
  layer13 = args.layer + 13
  layer14 = args.layer + 14
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
  conv5_path = (args.layer12_conv_history_probe.resolve() if args.layer12_conv_history_probe is not None
                else TWO.latest_conv_history_probe_for_layer(layer7).resolve())
  conv6_path = (args.layer13_conv_history_probe.resolve() if args.layer13_conv_history_probe is not None
                else TWO.latest_conv_history_probe_for_layer(layer8).resolve())
  conv7_path = (args.layer14_conv_history_probe.resolve() if args.layer14_conv_history_probe is not None
                else TWO.latest_conv_history_probe_for_layer(layer9).resolve())
  conv8_path = (args.layer16_conv_history_probe.resolve() if args.layer16_conv_history_probe is not None
                else TWO.latest_conv_history_probe_for_layer(layer11).resolve())
  conv9_path = (args.layer17_conv_history_probe.resolve() if args.layer17_conv_history_probe is not None
                else TWO.latest_conv_history_probe_for_layer(layer12).resolve())
  conv10_path = (args.layer18_conv_history_probe.resolve() if args.layer18_conv_history_probe is not None
                 else TWO.latest_conv_history_probe_for_layer(layer13).resolve())
  all_history_json = args.all_history_json.resolve()
  stamp = utc_stamp()
  created_at = iso_now()
  out_dir = args.out_dir or ROOT / f"output/gpu-resident-layer19-full-attn-handoff-probe-{stamp}"
  out_dir.mkdir(parents=True, exist_ok=False)
  raw_dir = out_dir / "raw"
  raw_dir.mkdir()

  payloads0, conv0 = TWO.prefixed_payloads(layer0, conv0_path, "l0")
  payloads1, conv1 = TWO.prefixed_payloads(layer1, conv1_path, "l1")
  payloads2, conv2 = TWO.prefixed_payloads(layer3, conv2_path, "l3")
  payloads3, conv3 = TWO.prefixed_payloads(layer4, conv3_path, "l4")
  payloads4, conv4 = TWO.prefixed_payloads(layer5, conv4_path, "l5")
  payloads5, conv5 = TWO.prefixed_payloads(layer7, conv5_path, "l12")
  payloads6, conv6 = TWO.prefixed_payloads(layer8, conv6_path, "l13")
  payloads7, conv7 = TWO.prefixed_payloads(layer9, conv7_path, "l14")
  payloads8, conv8 = TWO.prefixed_payloads(layer11, conv8_path, "l16")
  payloads9, conv9 = TWO.prefixed_payloads(layer12, conv9_path, "l17")
  payloads10, conv10 = TWO.prefixed_payloads(layer13, conv10_path, "l18")
  layer7_payloads, layer7_history = L7_INPUT.resolve_full_attention_payloads(
      all_history_json, layer2
  )
  CORE.add_layer7_tail_payloads(layer7_payloads)
  L8.add_layer7_ffn_payloads(layer7_payloads)
  layer11_payloads, layer11_history = L11.resolve_prefixed_full_attention_payloads(
      all_history_json, layer6, "l6_"
  )
  L11_FFN.add_layer11_ffn_payloads(layer11_payloads)
  layer15_payloads, layer15_history = L11_FULL.resolve_prefixed_full_attention_payloads(
      all_history_json, layer10, "l15_"
  )
  L15.add_layer15_ffn_payloads(layer15_payloads)
  layer19_payloads, layer19_history = L11_FULL.resolve_prefixed_full_attention_payloads(
      all_history_json, layer14, "l19_"
  )
  payloads = {
      **payloads0,
      **payloads1,
      **layer7_payloads,
      **payloads2,
      **payloads3,
      **payloads4,
      **layer11_payloads,
      **payloads5,
      **payloads6,
      **payloads7,
      **layer15_payloads,
      **payloads8,
      **payloads9,
      **payloads10,
      **layer19_payloads,
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
  local_cpp = out_dir / "gpu_resident_layer19_full_attn_handoff_probe.cpp"
  local_cpp.write_text(layer19_full_attn_probe_cpp(opencl_source), encoding="utf-8")

  remote_dir = f"{args.remote_root.rstrip('/')}/gpu-resident-layer19-full-attn-handoff-probe-{stamp}"
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
            f"{remote_dir}/tests/gpu_resident_layer19_full_attn_handoff_probe.cpp",
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

  executable = f"{remote_dir}/build/iq36-gpu-resident-layer19-full-attn-handoff-probe"
  compile_cmd = " && ".join([
      f"source {shlex.quote(args.env_script)} >/dev/null 2>&1",
      (
          "g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic "
          f"-I {shlex.quote(remote_dir + '/include')} "
          f"{shlex.quote(remote_dir + '/src/gguf_loader.cpp')} "
          f"{shlex.quote(remote_dir + '/src/gpu_q4x8_matvec.cpp')} "
          f"{shlex.quote(remote_dir + '/tests/gpu_resident_layer19_full_attn_handoff_probe.cpp')} "
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
      {"name": "arc_b390_selected", "pass": bool(probe and args.device_substring in str(probe.get("device_name", "")) and args.device_substring in str(probe.get("layer19_v_projection_device_name", "")))},
      {"name": "resident_api_fields_present", "pass": resident_fields_ok(probe, args.resident_invocations, args.device_substring)},
  ]
  checks.extend(layer18_predecessor_checks(probe))
  checks.extend([
      {"name": "layer19_residual_input_from_layer18_live_gpu_lout", "pass": PRECONV.nested_bool(probe, "checks", "layer19_residual_input_from_layer18_live_gpu_lout")},
      {"name": "layer19_payload_counts_ok", "pass": PRECONV.nested_bool(probe, "checks", "layer19_payload_counts_ok")},
      {"name": "layer19_full_attn_input_matches_oracle", "pass": PRECONV.nested_bool(probe, "checks", "layer19_full_attn_input_matches_oracle")},
      {"name": "layer19_full_attn_component_policy_matches_oracle", "pass": PRECONV.nested_bool(probe, "checks", "layer19_full_attn_component_policy_matches_oracle")},
      {"name": "layer19_full_attn_core_output_matches_oracle", "pass": PRECONV.nested_bool(probe, "checks", "layer19_full_attn_core_output_matches_oracle")},
      {"name": "layer19_v_gpu_boundary", "pass": PRECONV.nested_bool(probe, "checks", "layer19_v_gpu_boundary")},
      {"name": "layer19_ffn_q6_boundary", "pass": PRECONV.nested_bool(probe, "checks", "layer19_ffn_q6_boundary")},
      {"name": "layer19_arc_device_selected", "pass": PRECONV.nested_bool(probe, "checks", "layer19_arc_device_selected")},
      {"name": "l19_residual_input_matches_oracle", "pass": CORE.full_attention_comparison_passed(probe, "l19_residual_input", "gpu_vs_oracle")},
      {"name": "l19_attn_norm_matches_oracle", "pass": CORE.full_attention_comparison_passed(probe, "l19_attn_norm", "gpu_vs_oracle")},
      {"name": "l19_q_full_matches_oracle", "pass": CORE.full_attention_comparison_passed(probe, "l19_q_full", "gpu_vs_oracle")},
      {"name": "l19_q_rope_matches_oracle", "pass": CORE.full_attention_comparison_passed(probe, "l19_q_rope", "gpu_vs_oracle")},
      {"name": "l19_k_rope_matches_oracle", "pass": CORE.full_attention_comparison_passed(probe, "l19_k_rope", "gpu_vs_oracle")},
      {"name": "l19_v_matches_oracle", "pass": CORE.full_attention_comparison_passed(probe, "l19_v", "gpu_vs_oracle")},
      {"name": "l19_attn_pregate_matches_oracle", "pass": CORE.full_attention_comparison_passed(probe, "l19_attn_pregate", "gpu_vs_oracle")},
      {"name": "l19_attn_output_matches_oracle", "pass": CORE.full_attention_comparison_passed(probe, "l19_attn_output", "gpu_vs_oracle")},
      {"name": "l19_attn_post_norm_matches_oracle", "pass": CORE.full_attention_comparison_passed(probe, "l19_attn_post_norm", "gpu_vs_oracle")},
      {"name": "gpu_event_timing_positive", "pass": PRECONV.nested_bool(probe, "checks", "gpu_event_timing_positive")},
      {"name": "speedup_claims_forbidden", "pass": True},
  ])
  required_checks_passed = all(item["pass"] for item in checks)
  slim_payloads = {
      name: {key: value for key, value in payload.items() if key != "local_path"}
      for name, payload in payloads.items()
  }
  comparison_thresholds = {
      "strict_component": CORE.STRICT_COMPARISON_THRESHOLDS,
      "full_attn_component": CORE.FULL_ATTN_COMPARISON_THRESHOLDS,
      "layer15_full_shell_oracle_policy": CORE.FULL_ATTN_COMPARISON_THRESHOLDS,
      "layer16_state_input_oracle_policy": CORE.FULL_ATTN_COMPARISON_THRESHOLDS,
      "layer17_state_input_oracle_policy": CORE.FULL_ATTN_COMPARISON_THRESHOLDS,
      "layer18_state_input_oracle_policy": CORE.FULL_ATTN_COMPARISON_THRESHOLDS,
      "layer19_full_attention_oracle_policy": CORE.FULL_ATTN_COMPARISON_THRESHOLDS,
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
          "layer7": str(conv5_path.relative_to(ROOT)),
          "layer8": str(conv6_path.relative_to(ROOT)),
          "layer9": str(conv7_path.relative_to(ROOT)),
          "layer11": str(conv8_path.relative_to(ROOT)),
          "layer12": str(conv9_path.relative_to(ROOT)),
          "layer13": str(conv6_path.relative_to(ROOT)),
          "layer14": str(conv7_path.relative_to(ROOT)),
          "layer16": str(conv8_path.relative_to(ROOT)),
          "layer17": str(conv9_path.relative_to(ROOT)),
          "layer18": str(conv10_path.relative_to(ROOT)),
      },
      "conv_history_capture_artifacts": {
          "layer0": conv0.get("capture_artifact"),
          "layer1": conv1.get("capture_artifact"),
          "layer3": conv2.get("capture_artifact"),
          "layer4": conv3.get("capture_artifact"),
          "layer5": conv4.get("capture_artifact"),
          "layer7": conv5.get("capture_artifact"),
          "layer8": conv6.get("capture_artifact"),
          "layer9": conv7.get("capture_artifact"),
          "layer11": conv8.get("capture_artifact"),
          "layer12": conv9.get("capture_artifact"),
          "layer13": conv6.get("capture_artifact"),
          "layer14": conv7.get("capture_artifact"),
          "layer16": conv8.get("capture_artifact"),
          "layer17": conv9.get("capture_artifact"),
          "layer18": conv10.get("capture_artifact"),
      },
      "all_history": {
          "layer7": layer7_history,
          "layer11": layer11_history,
          "layer15": layer15_history,
          "layer19": layer19_history,
      },
      "ffn_payload_root": str(L15.LAYER15_FFN_PAYLOAD_ROOT.relative_to(ROOT)),
      "payloads": slim_payloads,
      "layers": [layer0, layer1, layer2, layer3, layer4, layer5, layer6, layer7, layer8, layer9, layer10, layer11, layer12, layer13, layer14],
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
      "tool": "tools/intel-qwen36-gpu-resident-layer19-full-attn-handoff-probe.py",
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
      "gpu_resident_layer19_full_attn_handoff_probe",
      [
          ("required_checks_passed", required_checks_passed),
          ("resident_shell_invocations", args.resident_invocations),
          ("resident_layer19_full_attn_input_kernel_sum_min_us", PRECONV.nested_number(timings, "resident_layer19_full_attn_input_kernel_sum_min_us")),
          ("resident_layer19_full_attn_core_output_kernel_sum_min_us", PRECONV.nested_number(timings, "resident_layer19_full_attn_core_output_kernel_sum_min_us")),
          ("resident_layer19_full_attention_kernel_sum_min_us", PRECONV.nested_number(timings, "resident_layer19_full_attention_kernel_sum_min_us")),
          ("resident_layer5_6_7_8_9_10_11_12_13_14_15_16_17_18_to_layer19_full_attention_kernel_sum_min_us", PRECONV.nested_number(timings, "resident_layer5_6_7_8_9_10_11_12_13_14_15_16_17_18_to_layer19_full_attention_kernel_sum_min_us")),
          ("layer19_residual_input_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l19_residual_input", "gpu_vs_oracle", "max_abs_diff")),
          ("layer19_attn_norm_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l19_attn_norm", "gpu_vs_oracle", "max_abs_diff")),
          ("layer19_q_full_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l19_q_full", "gpu_vs_oracle", "max_abs_diff")),
          ("layer19_attn_output_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l19_attn_output", "gpu_vs_oracle", "max_abs_diff")),
          ("layer19_attn_post_norm_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l19_attn_post_norm", "gpu_vs_oracle", "max_abs_diff")),
      ],
  )
  write_summary(out_dir / "summary.md", payload)
  print(out_dir)
  return 0 if required_checks_passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
