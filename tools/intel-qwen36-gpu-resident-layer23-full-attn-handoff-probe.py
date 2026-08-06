#!/usr/bin/env python3
"""Run the resident GPU layer-23 full-attention handoff probe."""

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
L21_FULL_TOOL = Path(__file__).with_name(
    "intel-qwen36-gpu-resident-layer21-full-shell-handoff-probe.py"
)
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA_VERSION = "intel-qwen36-gpu-resident-layer23-full-attn-handoff-probe-v0"
L22_API = "layer5_to_layer22_state_input_load_once_run_many"
L23_API = "layer5_to_layer23_full_attention_load_once_run_many"
DEFAULT_HOST = "local"
DEFAULT_MODEL = "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf"
DEFAULT_ENV_SCRIPT = "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh"
DEFAULT_REMOTE_ROOT = "/home/intel/intel-qwen36-gpu"
DEFAULT_ORACLE_BUNDLE = ROOT / "oracle/r0-oracle-bundle-20260627T060028Z"
DEFAULT_ALL_HISTORY = ROOT / "output/r1-full-attn-all-history-capture-20260627T145615Z/history.json"
OPENCL_SOURCE = ROOT / "engine/gpu/opencl/q4x8_matvec.cl"


def load_l21_full_tool() -> ModuleType:
  spec = importlib.util.spec_from_file_location("iq36_layer21_full_shell_probe", L21_FULL_TOOL)
  if spec is None or spec.loader is None:
    raise SystemExit(f"failed to load layer21 full-shell/l_out tool: {L21_FULL_TOOL}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


L21_FULL = load_l21_full_tool()
L21_FULL.install_full_shell_overrides()
L21 = L21_FULL.L21
L20 = L21.L20
L19 = L21.L19
L15 = L19.L15
CORE = L19.CORE
L7_INPUT = L19.L7_INPUT
L8 = L19.L8
L11 = L19.L11
L11_FFN = L19.L11_FFN
L11_FULL = L19.L11_FULL
TWO = L19.TWO
PRECONV = L19.PRECONV
def replace_once(text: str, old: str, new: str) -> str:
  return L21.replace_once(text, old, new)

def layer22_state_input_probe_cpp(opencl_source: str) -> str:
  cpp = L21.layer21_state_input_probe_cpp(opencl_source)
  for old, new in {
      L21_FULL.SCHEMA_VERSION: SCHEMA_VERSION,
      "layer5_to_layer21_full_shell_lout_load_once_run_many": L22_API,
      "layer21 state/input handoff probe expects --layer 5":
          "layer22 state/input handoff probe expects --layer 5",
  }.items():
    cpp = replace_once(cpp, old, new)
  cpp = replace_once(
      cpp,
      '''    const int layer16 = args.layer + 16;
    Require(layer0 == 5 && layer1 == 6 && layer2 == 7 && layer3 == 8 && layer4 == 9 && layer5 == 10 && layer6 == 11 && layer7 == 12 && layer8 == 13 && layer9 == 14 && layer10 == 15 && layer11 == 16 && layer12 == 17 && layer13 == 18 && layer14 == 19 && layer15 == 20 && layer16 == 21,
            "layer22 state/input handoff probe expects --layer 5");
''',
      '''    const int layer16 = args.layer + 16;
    const int layer17 = args.layer + 17;
    Require(layer0 == 5 && layer1 == 6 && layer2 == 7 && layer3 == 8 && layer4 == 9 && layer5 == 10 && layer6 == 11 && layer7 == 12 && layer8 == 13 && layer9 == 14 && layer10 == 15 && layer11 == 16 && layer12 == 17 && layer13 == 18 && layer14 == 19 && layer15 == 20 && layer16 == 21 && layer17 == 22,
            "layer22 state/input handoff probe expects --layer 5");
''',
  )
  cpp = replace_once(
      cpp,
      '''    const auto layer16_tensors = ResolveLayerTensorBundle(index, layer16);
''',
      '''    const auto layer16_tensors = ResolveLayerTensorBundle(index, layer16);
    const auto layer17_tensors = ResolveLayerTensorBundle(index, layer17);
''',
  )
  cpp = replace_once(
      cpp,
      '''    const auto layer16_oracle = LoadLayerOraclePayloads(args.payload_dir, "l21");
''',
      '''    const auto layer16_oracle = LoadLayerOraclePayloads(args.payload_dir, "l21");
    const auto layer17_oracle = LoadLayerOraclePayloads(args.payload_dir, "l22");
''',
  )
  cpp = replace_once(
      cpp,
      '''    const auto layer16_run = RunResidentLinearLayerShell(
        args, index, layer16_tensors, layer16_oracle,
        layer15_run.gpu_layer_output, layer15_run.gpu_layer_output, rms_norm_epsilon);

    std::vector<NamedCompareGroup> strict_groups;
''',
      '''    const auto layer16_run = RunResidentLinearLayerShell(
        args, index, layer16_tensors, layer16_oracle,
        layer15_run.gpu_layer_output, layer15_run.gpu_layer_output, rms_norm_epsilon);

    const auto layer17_run = RunResidentLinearLayerShell(
        args, index, layer17_tensors, layer17_oracle,
        layer16_run.gpu_layer_output, layer16_run.gpu_layer_output, rms_norm_epsilon);

    std::vector<NamedCompareGroup> strict_groups;
''',
  )
  cpp = replace_once(
      cpp,
      '''    const bool layer16_ok =
        layer16_shapes_ok &&
        layer16_run.payload_counts_ok &&
        layer16_state_input_gpu_cpu_ok &&
        layer16_state_input_oracle_policy_ok &&
        layer16_state_input_timing_positive &&
        layer16_run.arc_selected &&
        layer16_qkv_q4_q6_boundary &&
        layer16_down_q4_q6_boundary;
    const bool layer2_timing_positive =
''',
      '''    const bool layer16_ok =
        layer16_shapes_ok &&
        layer16_run.payload_counts_ok &&
        layer16_state_input_gpu_cpu_ok &&
        layer16_state_input_oracle_policy_ok &&
        layer16_state_input_timing_positive &&
        layer16_run.arc_selected &&
        layer16_qkv_q4_q6_boundary &&
        layer16_down_q4_q6_boundary;
    const auto find_l17_group =
        [&](const std::string& name) -> const NamedCompareGroup& {
          const auto found = std::find_if(
              layer17_run.comparisons.begin(),
              layer17_run.comparisons.end(),
              [&](const NamedCompareGroup& group) {
                return group.name == name;
              });
          Require(found != layer17_run.comparisons.end(),
                  "layer22 comparison missing: " + name);
          return *found;
        };
    const auto& layer17_residual_input = find_l17_group("residual_input");
    const auto& layer17_attn_norm = find_l17_group("attn_norm");
    const auto& layer17_qkv = find_l17_group("linear_attn_qkv_mixed");
    const auto& layer17_conv_output_raw = find_l17_group("conv_output_raw");
    const bool layer17_state_input_gpu_cpu_ok =
        ComparePassed(layer17_residual_input.gpu_vs_cpu) &&
        ComparePassed(layer17_attn_norm.gpu_vs_cpu) &&
        ComparePassed(layer17_qkv.gpu_vs_cpu) &&
        ComparePassed(layer17_conv_output_raw.gpu_vs_cpu) &&
        ComparePassed(layer17_run.conv_state_after_gpu_vs_cpu);
    const bool layer17_state_input_oracle_policy_ok =
        ComparePassedFullAttentionComponent(layer17_residual_input.gpu_vs_oracle) &&
        ComparePassedFullAttentionComponent(layer17_attn_norm.gpu_vs_oracle) &&
        ComparePassedFullAttentionComponent(layer17_qkv.gpu_vs_oracle) &&
        ComparePassedFullAttentionComponent(layer17_conv_output_raw.gpu_vs_oracle);
    const bool layer17_shapes_ok = ShapesPassed(layer17_run.shape_checks);
    const bool layer17_qkv_q4_q6_boundary =
        (layer17_tensors.qkv_tensor->type == 12 || layer17_tensors.qkv_tensor->type == 14);
    const bool layer17_down_q4_q6_boundary =
        (layer17_tensors.selected_down_tensor->type == 12 ||
         layer17_tensors.selected_down_tensor->type == 14) &&
        (layer17_tensors.shared_down_tensor->type == 12 ||
         layer17_tensors.shared_down_tensor->type == 14);
    const bool layer17_state_input_timing_positive =
        layer17_run.timing.layer_input_rmsnorm_min_us > 0.0 &&
        layer17_run.timing.preconv_to_postconv_kernel_sum_min_us > 0.0;
    const bool layer17_ok =
        layer17_shapes_ok &&
        layer17_run.payload_counts_ok &&
        layer17_state_input_gpu_cpu_ok &&
        layer17_state_input_oracle_policy_ok &&
        layer17_state_input_timing_positive &&
        layer17_run.arc_selected &&
        layer17_qkv_q4_q6_boundary &&
        layer17_down_q4_q6_boundary;
    const bool layer2_timing_positive =
''',
  )
  cpp = replace_once(
      cpp,
      '''        layer16_ok &&
        layer2_timing_positive &&
''',
      '''        layer16_ok &&
        layer17_ok &&
        layer2_timing_positive &&
''',
  )
  cpp = replace_once(
      cpp,
      '''    const double layer16_state_input_sum_min =
        layer16_run.timing.layer_input_rmsnorm_min_us +
        layer16_run.timing.preconv_to_postconv_kernel_sum_min_us;
''',
      '''    const double layer16_state_input_sum_min =
        layer16_run.timing.layer_input_rmsnorm_min_us +
        layer16_run.timing.preconv_to_postconv_kernel_sum_min_us;
    const double layer17_state_input_sum_min =
        layer17_run.timing.layer_input_rmsnorm_min_us +
        layer17_run.timing.preconv_to_postconv_kernel_sum_min_us;
''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << "\\"layers\\":[" << layer0 << "," << layer1 << "," << layer2 << "," << layer3 << "," << layer4 << "," << layer5 << "," << layer6 << "," << layer7 << "," << layer8 << "," << layer9 << "," << layer10 << "," << layer11 << "," << layer12 << "," << layer13 << "," << layer14 << "," << layer15 << "," << layer16 << "],";
''',
      '''    std::cout << "\\"layers\\":[" << layer0 << "," << layer1 << "," << layer2 << "," << layer3 << "," << layer4 << "," << layer5 << "," << layer6 << "," << layer7 << "," << layer8 << "," << layer9 << "," << layer10 << "," << layer11 << "," << layer12 << "," << layer13 << "," << layer14 << "," << layer15 << "," << layer16 << "," << layer17 << "],";
''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << "\\"layer21_conv_state_boundary\\":\\"captured_conv_state\\",";
    std::cout << "\\"full_attn_core_input_boundary\\":\\"captured_rope_kv_payloads\\",";''',
      '''    std::cout << "\\"layer21_conv_state_boundary\\":\\"captured_conv_state\\",";
    std::cout << "\\"layer22_residual_input_boundary\\":\\"live_gpu_l_out_21\\",";
    std::cout << "\\"layer22_qkv_tensor_type\\":\\""
              << JsonEscape(iq36::ggml_type_name(layer17_tensors.qkv_tensor->type)) << "\\",";
    std::cout << "\\"layer22_selected_down_tensor_type\\":\\""
              << JsonEscape(iq36::ggml_type_name(layer17_tensors.selected_down_tensor->type)) << "\\",";
    std::cout << "\\"layer22_shared_down_tensor_type\\":\\""
              << JsonEscape(iq36::ggml_type_name(layer17_tensors.shared_down_tensor->type)) << "\\",";
    std::cout << "\\"layer22_conv_state_boundary\\":\\"captured_conv_state\\",";
    std::cout << "\\"full_attn_core_input_boundary\\":\\"captured_rope_kv_payloads\\",";''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << "\\"layer21_tail_device_name\\":\\"" << JsonEscape(layer16_run.tail_device_name) << "\\",";
    std::cout << "\\"layer7_v_tensor_type\\":\\""
''',
      '''    std::cout << "\\"layer21_tail_device_name\\":\\"" << JsonEscape(layer16_run.tail_device_name) << "\\",";
    std::cout << "\\"layer22_layer_input_device_name\\":\\"" << JsonEscape(layer17_run.layer_input_device_name) << "\\",";
    std::cout << "\\"layer22_preconv_device_name\\":\\"" << JsonEscape(layer17_run.preconv_device_name) << "\\",";
    std::cout << "\\"layer22_tail_device_name\\":\\"" << JsonEscape(layer17_run.tail_device_name) << "\\",";
    std::cout << "\\"layer7_v_tensor_type\\":\\""
''',
  )
  cpp = replace_once(
      cpp,
      '''                  layer15_run.program_build_ms +
                  layer16_run.program_build_ms)
''',
      '''                  layer15_run.program_build_ms +
                  layer16_run.program_build_ms +
                  layer17_run.program_build_ms)
''',
  )
  cpp = replace_once(
      cpp,
      '''                            layer15_run.build_log +
                            layer16_run.build_log)
''',
      '''                            layer15_run.build_log +
                            layer16_run.build_log +
                            layer17_run.build_log)
''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << "\\"resident_layer5_6_7_8_9_10_11_12_13_14_15_16_17_18_19_20_to_layer21_state_input_kernel_sum_min_us\\":"
              << (two_layer_kernel_sum_min + layer2_attention_sum_min +
                  layer2_ffn_sum_min + layer3_run.timing.layer_kernel_sum_min_us +
                  layer4_run.timing.layer_kernel_sum_min_us +
                  layer5_run.timing.layer_kernel_sum_min_us +
                  layer6_attention_sum_min + layer6_ffn_sum_min +
                  layer7_state_input_sum_min + layer8_state_input_sum_min +
                  layer9_state_input_sum_min + layer10_attention_sum_min +
                  layer10_ffn_sum_min + layer11_state_input_sum_min +
                  layer12_state_input_sum_min + layer13_state_input_sum_min +
                  layer14_attention_sum_min + layer14_ffn_sum_min +
                  layer15_state_input_sum_min +
                  layer16_state_input_sum_min);
''',
      '''    std::cout << "\\"resident_layer5_6_7_8_9_10_11_12_13_14_15_16_17_18_19_20_to_layer21_state_input_kernel_sum_min_us\\":"
              << (two_layer_kernel_sum_min + layer2_attention_sum_min +
                  layer2_ffn_sum_min + layer3_run.timing.layer_kernel_sum_min_us +
                  layer4_run.timing.layer_kernel_sum_min_us +
                  layer5_run.timing.layer_kernel_sum_min_us +
                  layer6_attention_sum_min + layer6_ffn_sum_min +
                  layer7_state_input_sum_min + layer8_state_input_sum_min +
                  layer9_state_input_sum_min + layer10_attention_sum_min +
                  layer10_ffn_sum_min + layer11_state_input_sum_min +
                  layer12_state_input_sum_min + layer13_state_input_sum_min +
                  layer14_attention_sum_min + layer14_ffn_sum_min +
                  layer15_state_input_sum_min +
                  layer16_state_input_sum_min) << ",";
    std::cout << "\\"resident_layer22_state_input_kernel_sum_min_us\\":"
              << layer17_state_input_sum_min << ",";
    std::cout << "\\"resident_layer5_6_7_8_9_10_11_12_13_14_15_16_17_18_19_20_21_to_layer22_state_input_kernel_sum_min_us\\":"
              << (two_layer_kernel_sum_min + layer2_attention_sum_min +
                  layer2_ffn_sum_min + layer3_run.timing.layer_kernel_sum_min_us +
                  layer4_run.timing.layer_kernel_sum_min_us +
                  layer5_run.timing.layer_kernel_sum_min_us +
                  layer6_attention_sum_min + layer6_ffn_sum_min +
                  layer7_state_input_sum_min + layer8_state_input_sum_min +
                  layer9_state_input_sum_min + layer10_attention_sum_min +
                  layer10_ffn_sum_min + layer11_state_input_sum_min +
                  layer12_state_input_sum_min + layer13_state_input_sum_min +
                  layer14_attention_sum_min + layer14_ffn_sum_min +
                  layer15_state_input_sum_min +
                  layer16_state_input_sum_min +
                  layer17_state_input_sum_min);
''',
  )
  cpp = replace_once(
      cpp,
      '''    WritePrefixedCompareGroups("l21", layer16_run.comparisons, &first_compare);
    std::cout << ",\\"l2_k_raw\\":{\\"gpu_vs_cpu\\":";
''',
      '''    WritePrefixedCompareGroups("l21", layer16_run.comparisons, &first_compare);
    WritePrefixedCompareGroups("l22", layer17_run.comparisons, &first_compare);
    std::cout << ",\\"l2_k_raw\\":{\\"gpu_vs_cpu\\":";
''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << "},\\"l21_recurrent_state\\":{\\"gpu_vs_cpu\\":";
    WriteCompare(layer16_run.recurrent_state_gpu_vs_cpu);
    std::cout << "}";
    std::cout << "},\\"checks\\":{";
''',
      '''    std::cout << "},\\"l21_recurrent_state\\":{\\"gpu_vs_cpu\\":";
    WriteCompare(layer16_run.recurrent_state_gpu_vs_cpu);
    std::cout << "},\\"l22_conv_state_after\\":{\\"gpu_vs_cpu\\":";
    WriteCompare(layer17_run.conv_state_after_gpu_vs_cpu);
    std::cout << "},\\"l22_recurrent_state\\":{\\"gpu_vs_cpu\\":";
    WriteCompare(layer17_run.recurrent_state_gpu_vs_cpu);
    std::cout << "}";
    std::cout << "},\\"checks\\":{";
''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << ",\\"layer16\\":";
    WriteLayerChecks(layer16_run);
    std::cout << ",\\"layer1_residual_input_matches_oracle\\":"
''',
      '''    std::cout << ",\\"layer16\\":";
    WriteLayerChecks(layer16_run);
    std::cout << ",\\"layer17\\":";
    WriteLayerChecks(layer17_run);
    std::cout << ",\\"layer1_residual_input_matches_oracle\\":"
''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << "\\"layer21_state_input_handoff_matches\\":"
              << (layer16_ok ? "true" : "false") << ",";
    std::cout << "\\"history_kv_state_payloads_present\\":"
''',
      '''    std::cout << "\\"layer21_state_input_handoff_matches\\":"
              << (layer16_ok ? "true" : "false") << ",";
    std::cout << "\\"layer22_residual_input_from_layer21_live_gpu_lout\\":true,";
    std::cout << "\\"layer22_payload_counts_ok\\":"
              << (layer17_run.payload_counts_ok ? "true" : "false") << ",";
    std::cout << "\\"layer22_shapes_ok\\":"
              << (layer17_shapes_ok ? "true" : "false") << ",";
    std::cout << "\\"layer22_qkv_q4_q6_boundary\\":"
              << (layer17_qkv_q4_q6_boundary ? "true" : "false") << ",";
    std::cout << "\\"layer22_down_q4_q6_boundary\\":"
              << (layer17_down_q4_q6_boundary ? "true" : "false") << ",";
    std::cout << "\\"layer22_conv_state_input_boundary\\":true,";
    std::cout << "\\"layer22_gpu_cpu_matches_native\\":"
              << (layer17_state_input_gpu_cpu_ok ? "true" : "false") << ",";
    std::cout << "\\"layer22_state_input_oracle_policy_matches\\":"
              << (layer17_state_input_oracle_policy_ok ? "true" : "false") << ",";
    std::cout << "\\"layer22_state_input_handoff_matches\\":"
              << (layer17_ok ? "true" : "false") << ",";
    std::cout << "\\"history_kv_state_payloads_present\\":"
''',
  )
  cpp = replace_once(
      cpp,
      '''                  layer14_timing_positive &&
                  layer15_state_input_timing_positive &&
                  layer16_state_input_timing_positive ? "true" : "false") << ",";
''',
      '''                  layer14_timing_positive &&
                  layer15_state_input_timing_positive &&
                  layer16_state_input_timing_positive &&
                  layer17_state_input_timing_positive ? "true" : "false") << ",";
''',
  )
  return cpp

def layer23_full_attn_probe_cpp(opencl_source: str) -> str:
  cpp = layer22_state_input_probe_cpp(opencl_source)
  cpp = replace_once(cpp, L22_API, L23_API)
  cpp = replace_once(
      cpp,
      '''    const int layer17 = args.layer + 17;
    Require(layer0 == 5 && layer1 == 6 && layer2 == 7 && layer3 == 8 && layer4 == 9 && layer5 == 10 && layer6 == 11 && layer7 == 12 && layer8 == 13 && layer9 == 14 && layer10 == 15 && layer11 == 16 && layer12 == 17 && layer13 == 18 && layer14 == 19 && layer15 == 20 && layer16 == 21 && layer17 == 22,
            "layer22 state/input handoff probe expects --layer 5");
''',
      '''    const int layer17 = args.layer + 17;
    const int layer18 = args.layer + 18;
    Require(layer0 == 5 && layer1 == 6 && layer2 == 7 && layer3 == 8 && layer4 == 9 && layer5 == 10 && layer6 == 11 && layer7 == 12 && layer8 == 13 && layer9 == 14 && layer10 == 15 && layer11 == 16 && layer12 == 17 && layer13 == 18 && layer14 == 19 && layer15 == 20 && layer16 == 21 && layer17 == 22 && layer18 == 23,
            "layer23 full-attention handoff probe expects --layer 5");
''',
  )
  cpp = replace_once(
      cpp,
      '''    const auto layer17_tensors = ResolveLayerTensorBundle(index, layer17);
''',
      '''    const auto layer17_tensors = ResolveLayerTensorBundle(index, layer17);
    const auto layer18_tensors = ResolveFullAttentionTensorBundle(index, layer18);
''',
  )
  cpp = replace_once(
      cpp,
      '''    const auto layer17_oracle = LoadLayerOraclePayloads(args.payload_dir, "l22");
''',
      '''    const auto layer17_oracle = LoadLayerOraclePayloads(args.payload_dir, "l22");
    const auto layer18_oracle = LoadFullAttentionPayloadsPrefixed(args.payload_dir, "l23_");
    const auto layer18_oracle_attn_residual =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "l23_full_attn_residual.bin"));
    const auto layer18_oracle_attn_post_norm =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "l23_full_attn_post_norm.bin"));
    const bool layer18_payload_counts_ok =
        FullAttentionPayloadCountsOk(layer18_oracle) &&
        layer18_oracle_attn_residual.size() == kHiddenSize &&
        layer18_oracle_attn_post_norm.size() == kHiddenSize;
''',
  )
  cpp = replace_once(
      cpp,
      '''    const auto layer14_full_shapes = CheckFullAttentionShapes(layer14_tensors);
    const bool layer14_full_shapes_ok = FullAttentionShapesPassed(layer14_full_shapes);
''',
      '''    const auto layer14_full_shapes = CheckFullAttentionShapes(layer14_tensors);
    const bool layer14_full_shapes_ok = FullAttentionShapesPassed(layer14_full_shapes);
    const auto layer18_full_shapes = CheckFullAttentionShapes(layer18_tensors);
    const bool layer18_full_shapes_ok = FullAttentionShapesPassed(layer18_full_shapes);
''',
  )
  cpp = replace_once(
      cpp,
      '''    const auto layer14_ffn_norm_weight =
        iq36::decode_tensor_row(args.model_path, index,
                                LayerTensorName(layer14, "post_attention_norm.weight"), 0);
    const std::string layer14_selected_gate_up_tensor_name =
''',
      '''    const auto layer14_ffn_norm_weight =
        iq36::decode_tensor_row(args.model_path, index,
                                LayerTensorName(layer14, "post_attention_norm.weight"), 0);
    const auto layer18_attn_norm_weight =
        ReadF32TensorPayload(model, *layer18_tensors.attn_norm_tensor,
                             static_cast<std::size_t>(kHiddenSize));
    const auto layer18_q_norm_weight =
        iq36::decode_tensor_row(args.model_path, index,
                                layer18_tensors.q_norm_tensor_name, 0);
    const auto layer18_k_norm_weight =
        iq36::decode_tensor_row(args.model_path, index,
                                layer18_tensors.k_norm_tensor_name, 0);
    const auto layer18_ffn_norm_weight =
        iq36::decode_tensor_row(args.model_path, index,
                                LayerTensorName(layer18, "post_attention_norm.weight"), 0);
    const std::string layer14_selected_gate_up_tensor_name =
''',
  )
  cpp = replace_once(
      cpp,
      '''    const auto layer17_run = RunResidentLinearLayerShell(
        args, index, layer17_tensors, layer17_oracle,
        layer16_run.gpu_layer_output, layer16_run.gpu_layer_output, rms_norm_epsilon);

    std::vector<NamedCompareGroup> strict_groups;
''',
      '''    const auto layer17_run = RunResidentLinearLayerShell(
        args, index, layer17_tensors, layer17_oracle,
        layer16_run.gpu_layer_output, layer16_run.gpu_layer_output, rms_norm_epsilon);

    const auto layer18_native_qkv = iq36::run_qwen36_full_attention_qkv_projection(
        args.model_path,
        index,
        layer18,
        layer17_run.gpu_layer_output,
        rms_norm_epsilon);
    const auto layer18_rms_gpu = RunGpuLayerInputRmsNorm(
        layer17_run.gpu_layer_output,
        layer18_attn_norm_weight,
        rms_norm_epsilon,
        args.device_substring,
        args.repeat);
    const auto layer18_qk_gpu = RunGpuFullAttentionQkFront(
        args.model_path,
        *layer18_tensors.q_tensor,
        *layer18_tensors.k_tensor,
        layer18_rms_gpu.attn_norm,
        args.device_substring,
        args.repeat);
    const auto layer18_v_gpu = RunGpuFullAttentionVAny(
        args.model_path,
        *layer18_tensors.v_tensor,
        layer18_rms_gpu.attn_norm,
        args.device_substring,
        args.repeat);
    const auto layer18_gpu_q_split = SplitFullAttentionQ(layer18_qk_gpu.q_full);
    const auto layer18_gpu_q_normed = ApplyRepeatedRmsNormFull(
        layer18_gpu_q_split.q_raw, layer18_q_norm_weight, rms_norm_epsilon);
    const auto layer18_gpu_k_normed = ApplyRepeatedRmsNormFull(
        layer18_qk_gpu.k_raw, layer18_k_norm_weight, rms_norm_epsilon);
    const auto layer18_gpu_rope = iq36::run_qwen36_full_attention_rope(
        layer18_gpu_q_normed,
        layer18_gpu_k_normed,
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
    const auto layer18_native_rope = iq36::run_qwen36_full_attention_rope(
        layer18_native_qkv.q_normed,
        layer18_native_qkv.k_normed,
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
    auto layer18_native_k_history = layer18_oracle.k_history;
    auto layer18_native_v_history = layer18_oracle.v_history;
    layer18_native_k_history.push_back(layer18_oracle.k_rope);
    layer18_native_v_history.push_back(layer18_oracle.v);
    const auto layer18_native_core = iq36::run_qwen36_full_attention_core(
        layer18_oracle.q_rope,
        layer18_native_k_history,
        layer18_native_v_history,
        head_dim,
        q_head_count,
        kv_head_count,
        kAttentionScale);
    const auto layer18_native_gate = iq36::run_qwen36_full_attention_gate(
        layer18_oracle.q_full, layer18_native_core.attn_pregate, head_dim);
    const auto layer18_native_attn_output = iq36::matvec_tensor(
        args.model_path,
        index,
        LayerTensorName(layer18, "attn_output.weight"),
        layer18_native_gate.attn_gated);
    const auto layer18_native_attn_residual =
        iq36::add_vectors(layer17_run.gpu_layer_output, layer18_native_attn_output);
    const auto layer18_native_attn_post_norm =
        iq36::apply_rms_norm(layer18_native_attn_residual,
                             layer18_ffn_norm_weight,
                             rms_norm_epsilon);

    auto layer18_gpu_k_history = layer18_oracle.k_history;
    auto layer18_gpu_v_history = layer18_oracle.v_history;
    layer18_gpu_k_history.push_back(layer18_oracle.k_rope);
    layer18_gpu_v_history.push_back(layer18_v_gpu.v);
    std::vector<float> layer18_gpu_k_history_flat;
    std::vector<float> layer18_gpu_v_history_flat;
    for (const auto& item : layer18_gpu_k_history) {
      layer18_gpu_k_history_flat.insert(layer18_gpu_k_history_flat.end(), item.begin(), item.end());
    }
    for (const auto& item : layer18_gpu_v_history) {
      layer18_gpu_v_history_flat.insert(layer18_gpu_v_history_flat.end(), item.begin(), item.end());
    }
    const auto layer18_core_gate_gpu = RunGpuFullAttentionCoreGate(
        layer18_oracle.q_rope,
        layer18_gpu_k_history_flat,
        layer18_gpu_v_history_flat,
        layer18_qk_gpu.q_full,
        args.device_substring,
        args.repeat);
    const auto layer18_attention_gpu = RunGpuAttentionFront(
        args.model_path,
        *iq36::find_tensor(index, LayerTensorName(layer18, "attn_output.weight")),
        layer18_core_gate_gpu.attn_gated,
        layer17_run.gpu_layer_output,
        layer18_ffn_norm_weight,
        rms_norm_epsilon,
        args.device_substring,
        args.repeat);

    std::vector<NamedCompareGroup> strict_groups;
''',
  )
  cpp = replace_once(
      cpp,
      '''    const bool layer17_ok =
        layer17_shapes_ok &&
        layer17_run.payload_counts_ok &&
        layer17_state_input_gpu_cpu_ok &&
        layer17_state_input_oracle_policy_ok &&
        layer17_state_input_timing_positive &&
        layer17_run.arc_selected &&
        layer17_qkv_q4_q6_boundary &&
        layer17_down_q4_q6_boundary;
    const bool layer2_timing_positive =
''',
      '''    const bool layer17_ok =
        layer17_shapes_ok &&
        layer17_run.payload_counts_ok &&
        layer17_state_input_gpu_cpu_ok &&
        layer17_state_input_oracle_policy_ok &&
        layer17_state_input_timing_positive &&
        layer17_run.arc_selected &&
        layer17_qkv_q4_q6_boundary &&
        layer17_down_q4_q6_boundary;
    std::vector<NamedCompareGroup> layer18_strict_groups;
    AppendCpuGpuOracleCompare(layer18_strict_groups, "l23_residual_input",
                              layer17_run.gpu_layer_output,
                              layer17_run.gpu_layer_output,
                              layer18_oracle.residual_input);
    AppendCpuGpuOracleCompare(layer18_strict_groups, "l23_attn_norm",
                              layer18_native_qkv.attention_norm,
                              layer18_rms_gpu.attn_norm,
                              layer18_oracle.attn_norm);
    AppendCpuGpuOracleCompare(layer18_strict_groups, "l23_q_full",
                              layer18_native_qkv.q_full,
                              layer18_qk_gpu.q_full,
                              layer18_oracle.q_full);
    AppendCpuGpuOracleCompare(layer18_strict_groups, "l23_q_rope",
                              layer18_native_rope.q_rope,
                              layer18_gpu_rope.q_rope,
                              layer18_oracle.q_rope);
    AppendCpuGpuOracleCompare(layer18_strict_groups, "l23_k_rope",
                              layer18_native_rope.k_rope,
                              layer18_gpu_rope.k_rope,
                              layer18_oracle.k_rope);
    AppendCpuGpuOracleCompare(layer18_strict_groups, "l23_v",
                              layer18_native_qkv.v,
                              layer18_v_gpu.v,
                              layer18_oracle.v);
    std::vector<NamedCompareGroup> layer18_full_attention_groups;
    AppendFullAttentionComponentCompare(layer18_full_attention_groups, "l23_attn_pregate",
                                        layer18_native_core.attn_pregate,
                                        layer18_core_gate_gpu.attn_pregate,
                                        layer18_oracle.attn_pregate);
    AppendFullAttentionComponentCompare(layer18_full_attention_groups, "l23_attn_gated",
                                        layer18_native_gate.attn_gated,
                                        layer18_core_gate_gpu.attn_gated,
                                        layer18_oracle.attn_gated);
    AppendFullAttentionComponentCompare(layer18_full_attention_groups, "l23_attn_output",
                                        layer18_native_attn_output,
                                        layer18_attention_gpu.linear_attn_out,
                                        layer18_oracle.attn_output);
    AppendFullAttentionComponentCompare(layer18_full_attention_groups, "l23_attn_residual",
                                        layer18_native_attn_residual,
                                        layer18_attention_gpu.attn_residual,
                                        layer18_oracle_attn_residual);
    AppendFullAttentionComponentCompare(layer18_full_attention_groups, "l23_attn_post_norm",
                                        layer18_native_attn_post_norm,
                                        layer18_attention_gpu.attn_post_norm,
                                        layer18_oracle_attn_post_norm);
    const auto layer18_k_raw_gpu_vs_cpu = iq36::compare_vectors(
        layer18_qk_gpu.k_raw, layer18_native_qkv.k_raw, kMismatchThreshold);
    const bool layer18_strict_input_ok =
        CompareGroupsPassed(layer18_strict_groups) &&
        ComparePassed(layer18_k_raw_gpu_vs_cpu);
    bool layer18_full_component_ok = true;
    for (const auto& group : layer18_full_attention_groups) {
      layer18_full_component_ok =
          layer18_full_component_ok &&
          ComparePassedFullAttentionComponent(group.gpu_vs_oracle);
    }
    const bool layer18_comparisons_ok =
        layer18_strict_input_ok && layer18_full_component_ok;
    const bool layer18_timing_positive =
        layer18_rms_gpu.timing.rmsnorm_min_us > 0.0 &&
        layer18_qk_gpu.timing.q_projection_min_us > 0.0 &&
        layer18_qk_gpu.timing.k_projection_min_us > 0.0 &&
        layer18_v_gpu.timing.v_projection_min_us > 0.0 &&
        layer18_core_gate_gpu.timing.core_min_us > 0.0 &&
        layer18_core_gate_gpu.timing.gate_min_us > 0.0 &&
        layer18_attention_gpu.timing.output_projection_min_us > 0.0 &&
        layer18_attention_gpu.timing.residual_add_min_us > 0.0 &&
        layer18_attention_gpu.timing.ffn_rmsnorm_min_us > 0.0;
    const bool layer18_arc_selected =
        layer18_rms_gpu.device_name.find(args.device_substring) != std::string::npos &&
        layer18_qk_gpu.device_name.find(args.device_substring) != std::string::npos &&
        layer18_v_gpu.device_name.find(args.device_substring) != std::string::npos &&
        layer18_core_gate_gpu.device_name.find(args.device_substring) != std::string::npos &&
        layer18_attention_gpu.device_name.find(args.device_substring) != std::string::npos;
    const bool layer18_v_gpu_boundary =
        (layer18_tensors.v_tensor->type == 12 || layer18_tensors.v_tensor->type == 14) &&
        layer18_v_gpu.v.size() == kFullKvValues;
    const bool layer18_ok =
        layer18_full_shapes_ok &&
        layer18_payload_counts_ok &&
        metadata_ok &&
        layer18_comparisons_ok &&
        layer18_timing_positive &&
        layer18_arc_selected &&
        layer18_v_gpu_boundary;
    const bool layer2_timing_positive =
''',
  )
  cpp = replace_once(
      cpp,
      '''        layer15_ok &&
        layer16_ok &&
        layer17_ok &&
        layer2_timing_positive &&
''',
      '''        layer15_ok &&
        layer16_ok &&
        layer17_ok &&
        layer18_ok &&
        layer2_timing_positive &&
''',
  )
  cpp = replace_once(
      cpp,
      '''    const double layer17_state_input_sum_min =
        layer17_run.timing.layer_input_rmsnorm_min_us +
        layer17_run.timing.preconv_to_postconv_kernel_sum_min_us;
''',
      '''    const double layer17_state_input_sum_min =
        layer17_run.timing.layer_input_rmsnorm_min_us +
        layer17_run.timing.preconv_to_postconv_kernel_sum_min_us;
    const double layer18_input_sum_min =
        layer18_rms_gpu.timing.rmsnorm_min_us +
        layer18_qk_gpu.timing.qk_projection_kernel_sum_min_us +
        layer18_v_gpu.timing.v_projection_min_us;
    const double layer18_core_output_sum_min =
        layer18_core_gate_gpu.timing.core_gate_kernel_sum_min_us +
        layer18_attention_gpu.timing.attention_front_kernel_sum_min_us;
    const double layer18_attention_sum_min =
        layer18_input_sum_min + layer18_core_output_sum_min;
''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << "\\"layers\\":[" << layer0 << "," << layer1 << "," << layer2 << "," << layer3 << "," << layer4 << "," << layer5 << "," << layer6 << "," << layer7 << "," << layer8 << "," << layer9 << "," << layer10 << "," << layer11 << "," << layer12 << "," << layer13 << "," << layer14 << "," << layer15 << "," << layer16 << "," << layer17 << "],";
''',
      '''    std::cout << "\\"layers\\":[" << layer0 << "," << layer1 << "," << layer2 << "," << layer3 << "," << layer4 << "," << layer5 << "," << layer6 << "," << layer7 << "," << layer8 << "," << layer9 << "," << layer10 << "," << layer11 << "," << layer12 << "," << layer13 << "," << layer14 << "," << layer15 << "," << layer16 << "," << layer17 << "," << layer18 << "],";
''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << "\\"layer22_conv_state_boundary\\":\\"captured_conv_state\\",";
    std::cout << "\\"full_attn_core_input_boundary\\":\\"captured_rope_kv_payloads\\",";''',
      '''    std::cout << "\\"layer22_conv_state_boundary\\":\\"captured_conv_state\\",";
    std::cout << "\\"layer23_residual_input_boundary\\":\\"live_gpu_l_out_22\\",";
    std::cout << "\\"layer23_v_tensor_type\\":\\""
              << JsonEscape(iq36::ggml_type_name(layer18_tensors.v_tensor->type)) << "\\",";
    std::cout << "\\"layer23_v_projection_boundary\\":\\""
              << (layer18_tensors.v_tensor->type == 14 ? "gpu_q6_raw_matvec" : "gpu_q4x8_matvec") << "\\",";
    std::cout << "\\"layer23_ffn_boundary\\":\\"q4_q6_down_pending\\",";
    std::cout << "\\"full_attn_core_input_boundary\\":\\"captured_rope_kv_payloads\\",";''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << "\\"layer22_tail_device_name\\":\\"" << JsonEscape(layer17_run.tail_device_name) << "\\",";
    std::cout << "\\"layer7_v_tensor_type\\":\\""
''',
      '''    std::cout << "\\"layer22_tail_device_name\\":\\"" << JsonEscape(layer17_run.tail_device_name) << "\\",";
    std::cout << "\\"layer23_core_gate_device_name\\":\\"" << JsonEscape(layer18_core_gate_gpu.device_name) << "\\",";
    std::cout << "\\"layer23_output_projection_device_name\\":\\"" << JsonEscape(layer18_attention_gpu.device_name) << "\\",";
    std::cout << "\\"layer23_v_projection_device_name\\":\\"" << JsonEscape(layer18_v_gpu.device_name) << "\\",";
    std::cout << "\\"layer7_v_tensor_type\\":\\""
''',
  )
  cpp = replace_once(
      cpp,
      '''                  layer15_run.program_build_ms +
                  layer16_run.program_build_ms +
                  layer17_run.program_build_ms)
''',
      '''                  layer15_run.program_build_ms +
                  layer16_run.program_build_ms +
                  layer17_run.program_build_ms +
                  layer18_rms_gpu.program_build_ms + layer18_qk_gpu.program_build_ms +
                  layer18_v_gpu.program_build_ms +
                  layer18_core_gate_gpu.program_build_ms +
                  layer18_attention_gpu.program_build_ms)
''',
  )
  cpp = replace_once(
      cpp,
      '''                            layer15_run.build_log +
                            layer16_run.build_log +
                            layer17_run.build_log)
''',
      '''                            layer15_run.build_log +
                            layer16_run.build_log +
                            layer17_run.build_log +
                            layer18_rms_gpu.build_log + layer18_qk_gpu.build_log +
                            layer18_v_gpu.build_log +
                            layer18_core_gate_gpu.build_log +
                            layer18_attention_gpu.build_log)
''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << ",\\"layer14_output_front\\":";
    WriteFullAttentionOutputFrontTiming(layer14_attention_gpu.timing);
''',
      '''    std::cout << ",\\"layer14_output_front\\":";
    WriteFullAttentionOutputFrontTiming(layer14_attention_gpu.timing);
    std::cout << ",\\"layer18_full_attn_input\\":";
    WriteFullAttentionTiming(layer18_rms_gpu, layer18_qk_gpu);
    std::cout << ",\\"layer18_v_projection\\":";
    WriteFullAttentionVQ6Timing(layer18_v_gpu.timing);
    std::cout << ",\\"layer18_core_gate\\":";
    WriteFullCoreGateTiming(layer18_core_gate_gpu.timing);
    std::cout << ",\\"layer18_output_front\\":";
    WriteFullAttentionOutputFrontTiming(layer18_attention_gpu.timing);
''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << "\\"resident_layer22_state_input_kernel_sum_min_us\\":"
              << layer17_state_input_sum_min << ",";
    std::cout << "\\"resident_layer5_6_7_8_9_10_11_12_13_14_15_16_17_18_19_20_21_to_layer22_state_input_kernel_sum_min_us\\":"
              << (two_layer_kernel_sum_min + layer2_attention_sum_min +
                  layer2_ffn_sum_min + layer3_run.timing.layer_kernel_sum_min_us +
                  layer4_run.timing.layer_kernel_sum_min_us +
                  layer5_run.timing.layer_kernel_sum_min_us +
                  layer6_attention_sum_min + layer6_ffn_sum_min +
                  layer7_state_input_sum_min + layer8_state_input_sum_min +
                  layer9_state_input_sum_min + layer10_attention_sum_min +
                  layer10_ffn_sum_min + layer11_state_input_sum_min +
                  layer12_state_input_sum_min + layer13_state_input_sum_min +
                  layer14_attention_sum_min + layer14_ffn_sum_min +
                  layer15_state_input_sum_min +
                  layer16_state_input_sum_min +
                  layer17_state_input_sum_min);
''',
      '''    std::cout << "\\"resident_layer22_state_input_kernel_sum_min_us\\":"
              << layer17_state_input_sum_min << ",";
    std::cout << "\\"resident_layer5_6_7_8_9_10_11_12_13_14_15_16_17_18_19_20_21_to_layer22_state_input_kernel_sum_min_us\\":"
              << (two_layer_kernel_sum_min + layer2_attention_sum_min +
                  layer2_ffn_sum_min + layer3_run.timing.layer_kernel_sum_min_us +
                  layer4_run.timing.layer_kernel_sum_min_us +
                  layer5_run.timing.layer_kernel_sum_min_us +
                  layer6_attention_sum_min + layer6_ffn_sum_min +
                  layer7_state_input_sum_min + layer8_state_input_sum_min +
                  layer9_state_input_sum_min + layer10_attention_sum_min +
                  layer10_ffn_sum_min + layer11_state_input_sum_min +
                  layer12_state_input_sum_min + layer13_state_input_sum_min +
                  layer14_attention_sum_min + layer14_ffn_sum_min +
                  layer15_state_input_sum_min +
                  layer16_state_input_sum_min +
                  layer17_state_input_sum_min) << ",";
    std::cout << "\\"resident_layer23_full_attn_input_kernel_sum_min_us\\":"
              << layer18_input_sum_min << ",";
    std::cout << "\\"resident_layer23_full_attn_core_output_kernel_sum_min_us\\":"
              << layer18_core_output_sum_min << ",";
    std::cout << "\\"resident_layer23_full_attention_kernel_sum_min_us\\":"
              << layer18_attention_sum_min << ",";
    std::cout << "\\"resident_layer5_6_7_8_9_10_11_12_13_14_15_16_17_18_19_20_21_22_to_layer23_full_attention_kernel_sum_min_us\\":"
              << (two_layer_kernel_sum_min + layer2_attention_sum_min +
                  layer2_ffn_sum_min + layer3_run.timing.layer_kernel_sum_min_us +
                  layer4_run.timing.layer_kernel_sum_min_us +
                  layer5_run.timing.layer_kernel_sum_min_us +
                  layer6_attention_sum_min + layer6_ffn_sum_min +
                  layer7_state_input_sum_min + layer8_state_input_sum_min +
                  layer9_state_input_sum_min + layer10_attention_sum_min +
                  layer10_ffn_sum_min + layer11_state_input_sum_min +
                  layer12_state_input_sum_min + layer13_state_input_sum_min +
                  layer14_attention_sum_min + layer14_ffn_sum_min +
                  layer15_state_input_sum_min +
                  layer16_state_input_sum_min +
                  layer17_state_input_sum_min + layer18_attention_sum_min);
''',
  )
  cpp = replace_once(
      cpp,
      '''    WritePrefixedCompareGroups("l22", layer17_run.comparisons, &first_compare);
    std::cout << ",\\"l2_k_raw\\":{\\"gpu_vs_cpu\\":";
''',
      '''    WritePrefixedCompareGroups("l22", layer17_run.comparisons, &first_compare);
    std::cout << ",";
    WriteNamedCompareGroups(layer18_strict_groups);
    std::cout << ",\\"l23_k_raw\\":{\\"gpu_vs_cpu\\":";
    WriteCompare(layer18_k_raw_gpu_vs_cpu);
    std::cout << "},";
    WriteNamedCompareGroups(layer18_full_attention_groups);
    std::cout << ",\\"l2_k_raw\\":{\\"gpu_vs_cpu\\":";
''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << ",\\"layer14_full_attention_shapes\\":";
    WriteFullAttentionShapeChecks(layer14_full_shapes);
''',
      '''    std::cout << ",\\"layer14_full_attention_shapes\\":";
    WriteFullAttentionShapeChecks(layer14_full_shapes);
    std::cout << ",\\"layer18_full_attention_shapes\\":";
    WriteFullAttentionShapeChecks(layer18_full_shapes);
''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << "\\"layer22_state_input_handoff_matches\\":"
              << (layer17_ok ? "true" : "false") << ",";
    std::cout << "\\"history_kv_state_payloads_present\\":"
''',
      '''    std::cout << "\\"layer22_state_input_handoff_matches\\":"
              << (layer17_ok ? "true" : "false") << ",";
    std::cout << "\\"layer23_residual_input_from_layer22_live_gpu_lout\\":true,";
    std::cout << "\\"layer23_payload_counts_ok\\":"
              << (layer18_payload_counts_ok ? "true" : "false") << ",";
    std::cout << "\\"layer23_full_attn_input_matches_oracle\\":"
              << (layer18_strict_input_ok ? "true" : "false") << ",";
    std::cout << "\\"layer23_full_attn_component_policy_matches_oracle\\":"
              << (layer18_full_component_ok ? "true" : "false") << ",";
    std::cout << "\\"layer23_full_attn_core_output_matches_oracle\\":"
              << (layer18_comparisons_ok ? "true" : "false") << ",";
    std::cout << "\\"layer23_v_gpu_boundary\\":"
              << (layer18_v_gpu_boundary ? "true" : "false") << ",";
    std::cout << "\\"layer23_arc_device_selected\\":"
              << (layer18_arc_selected ? "true" : "false") << ",";
    std::cout << "\\"history_kv_state_payloads_present\\":"
''',
  )
  cpp = replace_once(
      cpp,
      '''                  layer15_state_input_timing_positive &&
                  layer16_state_input_timing_positive &&
                  layer17_state_input_timing_positive ? "true" : "false") << ",";
''',
      '''                  layer15_state_input_timing_positive &&
                  layer16_state_input_timing_positive &&
                  layer17_state_input_timing_positive &&
                  layer18_timing_positive ? "true" : "false") << ",";
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
  parser.add_argument("--layer20-conv-history-probe", type=Path, default=None)
  parser.add_argument("--layer21-conv-history-probe", type=Path, default=None)
  parser.add_argument("--layer22-conv-history-probe", type=Path, default=None)
  parser.add_argument("--out-dir", type=Path, default=None)
  return parser.parse_args()


def resident_fields_ok(
    probe: dict[str, Any] | None,
    expected_invocations: int,
    device_substring: str,
) -> bool:
  return (
      isinstance(probe, dict)
      and probe.get("resident_api") == L23_API
      and probe.get("resident_load_count") == 1
      and probe.get("resident_shell_invocations") == expected_invocations
      and probe.get("layer22_residual_input_boundary") == "live_gpu_l_out_21"
      and device_substring in str(probe.get("layer22_tail_device_name", ""))
      and probe.get("layer23_residual_input_boundary") == "live_gpu_l_out_22"
      and probe.get("layer23_v_tensor_type") in {"Q4_K", "Q6_K"}
      and probe.get("layer23_v_projection_boundary") in {"gpu_q4x8_matvec", "gpu_q6_raw_matvec"}
      and device_substring in str(probe.get("layer23_core_gate_device_name", ""))
      and device_substring in str(probe.get("layer23_output_projection_device_name", ""))
      and device_substring in str(probe.get("layer23_v_projection_device_name", ""))
      and PRECONV.nested_bool(probe, "checks", "resident_load_once")
      and PRECONV.nested_bool(probe, "checks", "resident_shell_invocations_positive")
  )


def write_summary(path: Path, payload: dict[str, Any]) -> None:
  probe = payload.get("probe") if isinstance(payload.get("probe"), dict) else {}
  timings = probe.get("timings", {}) if isinstance(probe, dict) else {}
  comparisons = probe.get("comparisons", {}) if isinstance(probe, dict) else {}
  lines = [
      "# GPU Resident Layer-23 Full-Attention Handoff Probe",
      "",
      f"- workstream: `{WORKSTREAM}`",
      f"- host: `{payload.get('host')}`",
      f"- model: `{payload.get('model')}`",
      f"- layers: `{payload.get('layers')}`",
      f"- required checks passed: `{str(payload.get('required_checks_passed')).lower()}`",
      "- speedup claims allowed: `false`",
      f"- resident API: `{probe.get('resident_api')}`",
      f"- resident shell invocations: `{probe.get('resident_shell_invocations')}`",
      f"- layer 23 residual input boundary: `{probe.get('layer23_residual_input_boundary')}`",
      f"- layer 23 V tensor type: `{probe.get('layer23_v_tensor_type')}`",
      f"- layer 23 V projection boundary: `{probe.get('layer23_v_projection_boundary')}`",
      f"- layer 23 FFN boundary: `{probe.get('layer23_ffn_boundary')}`",
      "",
      "| output | comparison | max abs | RMSE |",
      "|---|---|---:|---:|",
  ]
  for name in (
      "l23_residual_input",
      "l23_attn_norm",
      "l23_q_full",
      "l23_q_rope",
      "l23_k_rope",
      "l23_v",
      "l23_attn_pregate",
      "l23_attn_output",
      "l23_attn_post_norm",
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
      f"| layer23_full_attn_input | {timings.get('resident_layer23_full_attn_input_kernel_sum_min_us') if isinstance(timings, dict) else None} |",
      f"| layer23_full_attn_core_output | {timings.get('resident_layer23_full_attn_core_output_kernel_sum_min_us') if isinstance(timings, dict) else None} |",
      f"| layer23_full_attention | {timings.get('resident_layer23_full_attention_kernel_sum_min_us') if isinstance(timings, dict) else None} |",
      f"| through_layer23_full_attention | {timings.get('resident_layer5_6_7_8_9_10_11_12_13_14_15_16_17_18_19_20_21_22_to_layer23_full_attention_kernel_sum_min_us') if isinstance(timings, dict) else None} |",
      "",
      "This target-side process carries closed live GPU layer outputs through",
      "layer 22, then feeds live GPU `l_out-22` into layer 23 full-attention",
      "RMSNorm, Q/K/V projection, ROPE, core/gate, output projection, residual",
      "add, and post-attn RMSNorm. FFN/l_out remains the next gate. This is",
      "captured single-token handoff evidence, not decode throughput.",
      "",
  ]
  path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  args = parse_args()
  if args.resident_invocations <= 0:
    raise SystemExit("--resident-invocations must be positive")
  if args.layer != 5:
    raise SystemExit("--layer must be 5 for the layer-23 full-attention handoff")

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
  layer15 = args.layer + 15
  layer16 = args.layer + 16
  layer17 = args.layer + 17
  layer18 = args.layer + 18
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
  conv11_path = (args.layer20_conv_history_probe.resolve() if args.layer20_conv_history_probe is not None
                 else TWO.latest_conv_history_probe_for_layer(layer15).resolve())
  conv12_path = (args.layer21_conv_history_probe.resolve() if args.layer21_conv_history_probe is not None
                 else TWO.latest_conv_history_probe_for_layer(layer16).resolve())
  conv13_path = (args.layer22_conv_history_probe.resolve() if args.layer22_conv_history_probe is not None
                 else TWO.latest_conv_history_probe_for_layer(layer17).resolve())
  all_history_json = args.all_history_json.resolve()
  stamp = utc_stamp()
  created_at = iso_now()
  out_dir = args.out_dir or ROOT / f"output/gpu-resident-layer23-full-attn-handoff-probe-{stamp}"
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
  L19.add_layer19_ffn_payloads(layer19_payloads)
  layer23_payloads, layer23_history = L11_FULL.resolve_prefixed_full_attention_payloads(
      all_history_json, layer18, "l23_"
  )
  payloads11, conv11 = TWO.prefixed_payloads(layer15, conv11_path, "l20")
  payloads12, conv12 = TWO.prefixed_payloads(layer16, conv12_path, "l21")
  payloads13, conv13 = TWO.prefixed_payloads(layer17, conv13_path, "l22")
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
      **payloads11,
      **payloads12,
      **payloads13,
      **layer23_payloads,
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
  local_cpp = out_dir / "gpu_resident_layer23_full_attn_handoff_probe.cpp"
  local_cpp.write_text(layer23_full_attn_probe_cpp(opencl_source), encoding="utf-8")

  remote_dir = f"{args.remote_root.rstrip('/')}/gpu-resident-layer23-full-attn-handoff-probe-{stamp}"
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
            f"{remote_dir}/tests/gpu_resident_layer23_full_attn_handoff_probe.cpp",
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

  executable = f"{remote_dir}/build/iq36-gpu-resident-layer23-full-attn-handoff-probe"
  compile_cmd = " && ".join([
      f"source {shlex.quote(args.env_script)} >/dev/null 2>&1",
      (
          "g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic "
          f"-I {shlex.quote(remote_dir + '/include')} "
          f"{shlex.quote(remote_dir + '/src/gguf_loader.cpp')} "
          f"{shlex.quote(remote_dir + '/src/gpu_q4_cpu_order_matvec.cpp')} "
          f"{shlex.quote(remote_dir + '/src/gpu_q4x8_matvec.cpp')} "
          f"{shlex.quote(remote_dir + '/tests/gpu_resident_layer23_full_attn_handoff_probe.cpp')} "
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
      {"name": "arc_b390_selected", "pass": bool(probe and args.device_substring in str(probe.get("device_name", "")) and args.device_substring in str(probe.get("layer23_v_projection_device_name", "")))},
      {"name": "resident_api_fields_present", "pass": resident_fields_ok(probe, args.resident_invocations, args.device_substring)},
  ]
  checks.extend([
      {"name": "probe_required_checks_passed", "pass": isinstance(probe, dict) and probe.get("required_checks_passed") is True},
      {"name": "layer23_residual_input_from_layer22_live_gpu_lout", "pass": PRECONV.nested_bool(probe, "checks", "layer23_residual_input_from_layer22_live_gpu_lout")},
      {"name": "layer23_payload_counts_ok", "pass": PRECONV.nested_bool(probe, "checks", "layer23_payload_counts_ok")},
      {"name": "layer23_full_attn_input_matches_oracle", "pass": PRECONV.nested_bool(probe, "checks", "layer23_full_attn_input_matches_oracle")},
      {"name": "layer23_full_attn_component_policy_matches_oracle", "pass": PRECONV.nested_bool(probe, "checks", "layer23_full_attn_component_policy_matches_oracle")},
      {"name": "layer23_full_attn_core_output_matches_oracle", "pass": PRECONV.nested_bool(probe, "checks", "layer23_full_attn_core_output_matches_oracle")},
      {"name": "layer23_v_gpu_boundary", "pass": PRECONV.nested_bool(probe, "checks", "layer23_v_gpu_boundary")},
      {"name": "layer23_arc_device_selected", "pass": PRECONV.nested_bool(probe, "checks", "layer23_arc_device_selected")},
      {"name": "l23_residual_input_matches_oracle", "pass": CORE.full_attention_comparison_passed(probe, "l23_residual_input", "gpu_vs_oracle")},
      {"name": "l23_attn_norm_matches_oracle", "pass": CORE.full_attention_comparison_passed(probe, "l23_attn_norm", "gpu_vs_oracle")},
      {"name": "l23_q_full_matches_oracle", "pass": CORE.full_attention_comparison_passed(probe, "l23_q_full", "gpu_vs_oracle")},
      {"name": "l23_q_rope_matches_oracle", "pass": CORE.full_attention_comparison_passed(probe, "l23_q_rope", "gpu_vs_oracle")},
      {"name": "l23_k_rope_matches_oracle", "pass": CORE.full_attention_comparison_passed(probe, "l23_k_rope", "gpu_vs_oracle")},
      {"name": "l23_v_matches_oracle", "pass": CORE.full_attention_comparison_passed(probe, "l23_v", "gpu_vs_oracle")},
      {"name": "l23_attn_pregate_matches_oracle", "pass": CORE.full_attention_comparison_passed(probe, "l23_attn_pregate", "gpu_vs_oracle")},
      {"name": "l23_attn_output_matches_oracle", "pass": CORE.full_attention_comparison_passed(probe, "l23_attn_output", "gpu_vs_oracle")},
      {"name": "l23_attn_post_norm_matches_oracle", "pass": CORE.full_attention_comparison_passed(probe, "l23_attn_post_norm", "gpu_vs_oracle")},
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
      "layer20_state_input_oracle_policy": CORE.FULL_ATTN_COMPARISON_THRESHOLDS,
      "layer21_state_input_oracle_policy": CORE.FULL_ATTN_COMPARISON_THRESHOLDS,
      "layer22_state_input_oracle_policy": CORE.FULL_ATTN_COMPARISON_THRESHOLDS,
      "layer23_full_attention_oracle_policy": CORE.FULL_ATTN_COMPARISON_THRESHOLDS,
      "live_ffn_oracle_magnitude": {
          "max_abs_diff": CORE.STRICT_COMPARISON_THRESHOLDS["max_abs_diff"],
          "rmse": CORE.STRICT_COMPARISON_THRESHOLDS["rmse"],
          "mismatch_abs_diff": CORE.STRICT_COMPARISON_THRESHOLDS["mismatch_abs_diff"],
      },
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
          "layer20": str(conv11_path.relative_to(ROOT)),
          "layer21": str(conv12_path.relative_to(ROOT)),
          "layer22": str(conv13_path.relative_to(ROOT)),
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
          "layer20": conv11.get("capture_artifact"),
          "layer21": conv12.get("capture_artifact"),
          "layer22": conv13.get("capture_artifact"),
      },
      "all_history": {
          "layer7": layer7_history,
          "layer11": layer11_history,
          "layer15": layer15_history,
          "layer19": layer19_history,
          "layer23": layer23_history,
      },
      "ffn_payload_root": str(L19.LAYER19_FFN_PAYLOAD_ROOT.relative_to(ROOT)),
      "payloads": slim_payloads,
      "layers": [layer0, layer1, layer2, layer3, layer4, layer5, layer6, layer7, layer8, layer9, layer10, layer11, layer12, layer13, layer14, layer15, layer16, layer17, layer18],
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
      "tool": "tools/intel-qwen36-gpu-resident-layer23-full-attn-handoff-probe.py",
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
      "gpu_resident_layer23_full_attn_handoff_probe",
      [
          ("required_checks_passed", required_checks_passed),
          ("resident_shell_invocations", args.resident_invocations),
          ("resident_layer19_full_attn_input_kernel_sum_min_us", PRECONV.nested_number(timings, "resident_layer19_full_attn_input_kernel_sum_min_us")),
          ("resident_layer19_full_attn_core_output_kernel_sum_min_us", PRECONV.nested_number(timings, "resident_layer19_full_attn_core_output_kernel_sum_min_us")),
          ("resident_layer19_full_attention_kernel_sum_min_us", PRECONV.nested_number(timings, "resident_layer19_full_attention_kernel_sum_min_us")),
          ("resident_layer5_6_7_8_9_10_11_12_13_14_15_16_17_18_to_layer19_full_attention_kernel_sum_min_us", PRECONV.nested_number(timings, "resident_layer5_6_7_8_9_10_11_12_13_14_15_16_17_18_to_layer19_full_attention_kernel_sum_min_us")),
          ("resident_layer19_ffn_lout_kernel_sum_min_us", PRECONV.nested_number(timings, "resident_layer19_ffn_lout_kernel_sum_min_us")),
          ("resident_layer5_6_7_8_9_10_11_12_13_14_15_16_17_18_to_layer19_lout_kernel_sum_min_us", PRECONV.nested_number(timings, "resident_layer5_6_7_8_9_10_11_12_13_14_15_16_17_18_to_layer19_lout_kernel_sum_min_us")),
          ("resident_layer20_state_input_kernel_sum_min_us", PRECONV.nested_number(timings, "resident_layer20_state_input_kernel_sum_min_us")),
          ("resident_layer5_6_7_8_9_10_11_12_13_14_15_16_17_18_19_to_layer20_state_input_kernel_sum_min_us", PRECONV.nested_number(timings, "resident_layer5_6_7_8_9_10_11_12_13_14_15_16_17_18_19_to_layer20_state_input_kernel_sum_min_us")),
          ("resident_layer21_state_input_kernel_sum_min_us", PRECONV.nested_number(timings, "resident_layer21_state_input_kernel_sum_min_us")),
          ("resident_layer5_6_7_8_9_10_11_12_13_14_15_16_17_18_19_20_to_layer21_state_input_kernel_sum_min_us", PRECONV.nested_number(timings, "resident_layer5_6_7_8_9_10_11_12_13_14_15_16_17_18_19_20_to_layer21_state_input_kernel_sum_min_us")),
          ("resident_layer22_state_input_kernel_sum_min_us", PRECONV.nested_number(timings, "resident_layer22_state_input_kernel_sum_min_us")),
          ("resident_layer5_6_7_8_9_10_11_12_13_14_15_16_17_18_19_20_21_to_layer22_state_input_kernel_sum_min_us", PRECONV.nested_number(timings, "resident_layer5_6_7_8_9_10_11_12_13_14_15_16_17_18_19_20_21_to_layer22_state_input_kernel_sum_min_us")),
          ("resident_layer23_full_attn_input_kernel_sum_min_us", PRECONV.nested_number(timings, "resident_layer23_full_attn_input_kernel_sum_min_us")),
          ("resident_layer23_full_attn_core_output_kernel_sum_min_us", PRECONV.nested_number(timings, "resident_layer23_full_attn_core_output_kernel_sum_min_us")),
          ("resident_layer23_full_attention_kernel_sum_min_us", PRECONV.nested_number(timings, "resident_layer23_full_attention_kernel_sum_min_us")),
          ("resident_layer5_6_7_8_9_10_11_12_13_14_15_16_17_18_19_20_21_22_to_layer23_full_attention_kernel_sum_min_us", PRECONV.nested_number(timings, "resident_layer5_6_7_8_9_10_11_12_13_14_15_16_17_18_19_20_21_22_to_layer23_full_attention_kernel_sum_min_us")),
          ("layer19_residual_input_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l19_residual_input", "gpu_vs_oracle", "max_abs_diff")),
          ("layer19_attn_norm_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l19_attn_norm", "gpu_vs_oracle", "max_abs_diff")),
          ("layer19_q_full_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l19_q_full", "gpu_vs_oracle", "max_abs_diff")),
          ("layer19_attn_output_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l19_attn_output", "gpu_vs_oracle", "max_abs_diff")),
          ("layer19_attn_post_norm_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l19_attn_post_norm", "gpu_vs_oracle", "max_abs_diff")),
          ("l19_selected_down_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l19_selected_down", "gpu_vs_oracle", "max_abs_diff")),
          ("l19_shared_down_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l19_shared_down", "gpu_vs_oracle", "max_abs_diff")),
          ("l19_ffn_out_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l19_ffn_out", "gpu_vs_oracle", "max_abs_diff")),
          ("l19_layer_output_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l19_layer_output", "gpu_vs_oracle", "max_abs_diff")),
          ("layer20_residual_input_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l20_residual_input", "gpu_vs_oracle", "max_abs_diff")),
          ("layer20_attn_norm_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l20_attn_norm", "gpu_vs_oracle", "max_abs_diff")),
          ("layer20_qkv_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l20_linear_attn_qkv_mixed", "gpu_vs_oracle", "max_abs_diff")),
          ("layer20_conv_output_raw_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l20_conv_output_raw", "gpu_vs_oracle", "max_abs_diff")),
          ("layer21_residual_input_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l21_residual_input", "gpu_vs_oracle", "max_abs_diff")),
          ("layer21_attn_norm_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l21_attn_norm", "gpu_vs_oracle", "max_abs_diff")),
          ("layer21_qkv_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l21_linear_attn_qkv_mixed", "gpu_vs_oracle", "max_abs_diff")),
          ("layer21_conv_output_raw_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l21_conv_output_raw", "gpu_vs_oracle", "max_abs_diff")),
          ("layer22_residual_input_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l22_residual_input", "gpu_vs_oracle", "max_abs_diff")),
          ("layer22_attn_norm_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l22_attn_norm", "gpu_vs_oracle", "max_abs_diff")),
          ("layer22_qkv_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l22_linear_attn_qkv_mixed", "gpu_vs_oracle", "max_abs_diff")),
          ("layer22_conv_output_raw_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l22_conv_output_raw", "gpu_vs_oracle", "max_abs_diff")),
          ("layer23_residual_input_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l23_residual_input", "gpu_vs_oracle", "max_abs_diff")),
          ("layer23_attn_norm_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l23_attn_norm", "gpu_vs_oracle", "max_abs_diff")),
          ("layer23_q_full_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l23_q_full", "gpu_vs_oracle", "max_abs_diff")),
          ("layer23_attn_output_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l23_attn_output", "gpu_vs_oracle", "max_abs_diff")),
          ("layer23_attn_post_norm_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l23_attn_post_norm", "gpu_vs_oracle", "max_abs_diff")),
      ],
  )
  write_summary(out_dir / "summary.md", payload)
  print(out_dir)
  return 0 if required_checks_passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
