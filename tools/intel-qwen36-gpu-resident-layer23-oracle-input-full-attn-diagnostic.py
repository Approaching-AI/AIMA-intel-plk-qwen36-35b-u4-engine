#!/usr/bin/env python3
"""Run a layer-23 full-attention oracle-input diagnostic.

This wraps the layer-23 live-handoff probe and injects an extra C++ lane that
feeds captured/oracle layer-23 residual input directly into the same GPU
full-attention kernels. The live `l_out-22` gate remains owned by the wrapped
tool; this diagnostic only separates inherited layer-22 drift from layer-23
kernel correctness.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASE_TOOL = Path(__file__).with_name(
    "intel-qwen36-gpu-resident-layer23-full-attn-handoff-probe.py"
)
SCHEMA_VERSION = "intel-qwen36-gpu-resident-layer23-oracle-input-full-attn-diagnostic-v0"
DEFAULT_Z_SOURCE = "gpu-q4"
Z_SOURCE = DEFAULT_Z_SOURCE
INCLUDE_FFN_LOUT = False
Z_SOURCE_CHOICES = (
    DEFAULT_Z_SOURCE,
    "native",
    "gpu-f32",
    "gpu-q4-cpu-order",
)
Z_SOURCE_SCHEMAS = {
    DEFAULT_Z_SOURCE: SCHEMA_VERSION,
    "native": "intel-qwen36-gpu-resident-layer23-l22-native-z-correction-diagnostic-v0",
    "gpu-f32": "intel-qwen36-gpu-resident-layer23-l22-gpu-f32-z-correction-diagnostic-v0",
    "gpu-q4-cpu-order": "intel-qwen36-gpu-resident-layer23-l22-gpu-q4-cpu-order-z-correction-diagnostic-v0",
}
Z_SOURCE_APIS = {
    DEFAULT_Z_SOURCE: "layer5_to_layer23_full_attention_load_once_run_many",
    "native": "layer5_to_layer23_l22_native_z_correction_diagnostic",
    "gpu-f32": "layer5_to_layer23_l22_gpu_f32_z_correction_diagnostic",
    "gpu-q4-cpu-order": "layer5_to_layer23_l22_gpu_q4_cpu_order_z_correction_diagnostic",
}
Z_SOURCE_OUTPUT_SLUGS = {
    DEFAULT_Z_SOURCE: "gpu-resident-layer23-oracle-input-full-attn-diagnostic",
    "native": "gpu-resident-layer23-l22-native-z-correction-diagnostic",
    "gpu-f32": "gpu-resident-layer23-l22-gpu-f32-z-correction-diagnostic",
    "gpu-q4-cpu-order": "gpu-resident-layer23-l22-gpu-q4-cpu-order-z-correction-diagnostic",
}
LAYER23_FFN_PAYLOAD_ROOT = BASE_TOOL.parents[1] / "output/r0-boundary-capture-run-20260627T054024Z/remote-output/payloads"
LAYER23_FFN_PAYLOAD_SPECS = {
    "l23_ffn_topk": ("l23_ffn_moe_topk.bin", "ffn_moe_topk-23__tok15__ord*.bin", 32),
    "l23_ffn_weights_norm": (
        "l23_ffn_moe_weights_norm.bin",
        "ffn_moe_weights_norm-23__tok15__ord*.bin",
        32,
    ),
    "l23_ffn_gate_up": (
        "l23_ffn_moe_gate_up.bin",
        "ffn_moe_gate_up-23__tok15__ord*.bin",
        32768,
    ),
    "l23_ffn_swiglu": (
        "l23_ffn_moe_swiglu.bin",
        "ffn_moe_swiglu-23__tok15__ord*.bin",
        16384,
    ),
    "l23_ffn_down": ("l23_ffn_moe_down.bin", "ffn_moe_down-23__tok15__ord*.bin", 65536),
    "l23_ffn_weighted": (
        "l23_ffn_moe_weighted.bin",
        "ffn_moe_weighted-23__tok15__ord*.bin",
        65536,
    ),
    "l23_ffn_moe_out": (
        "l23_ffn_moe_out.bin",
        "ffn_moe_out-23__tok15__ord*.bin",
        8192,
    ),
    "l23_ffn_shexp": ("l23_ffn_shexp.bin", "ffn_shexp-23__tok15__ord*.bin", 8192),
    "l23_shared_gate": (
        "l23_shared_expert_gate.bin",
        "shared_expert_gate-23__tok15__ord*.bin",
        4,
    ),
    "l23_shared_gate_sigmoid": (
        "l23_shared_expert_gate_sigmoid.bin",
        "shared_expert_gate_sigmoid-23__tok15__ord*.bin",
        4,
    ),
    "l23_ffn_shexp_gated": (
        "l23_ffn_shexp_gated.bin",
        "ffn_shexp_gated-23__tok15__ord*.bin",
        8192,
    ),
    "l23_ffn_out": ("l23_ffn_out.bin", "ffn_out-23__tok15__ord*.bin", 8192),
    "l23_l_out": ("l23_l_out.bin", "l_out-23__tok15__ord*.bin", 8192),
}


def load_base_tool() -> ModuleType:
  spec = importlib.util.spec_from_file_location("iq36_layer23_full_attn_probe", BASE_TOOL)
  if spec is None or spec.loader is None:
    raise SystemExit(f"failed to load base layer23 full-attn tool: {BASE_TOOL}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


BASE = load_base_tool()
BASE_CPP = BASE.layer23_full_attn_probe_cpp
BASE_PARSE_ARGS = BASE.parse_args
BASE_WRITE_SUMMARY = BASE.write_summary
BASE_RESOLVE_PREFIXED_FULL_ATTENTION_PAYLOADS = BASE.L11_FULL.resolve_prefixed_full_attention_payloads


def active_schema_version(z_source: str) -> str:
  schema = Z_SOURCE_SCHEMAS[z_source]
  if INCLUDE_FFN_LOUT:
    return schema.removesuffix("-v0") + "-ffn-lout-v0"
  return schema


def active_resident_api(z_source: str) -> str:
  api = Z_SOURCE_APIS[z_source]
  if not INCLUDE_FFN_LOUT:
    return api
  if api.endswith("_diagnostic"):
    return api.removesuffix("_diagnostic") + "_ffn_lout_diagnostic"
  return api.replace("_full_attention_", "_full_attention_ffn_lout_")


def configure_z_source(z_source: str) -> None:
  global Z_SOURCE
  Z_SOURCE = z_source
  BASE.SCHEMA_VERSION = active_schema_version(z_source)
  BASE.L23_API = active_resident_api(z_source)


def replace_once(text: str, old: str, new: str) -> str:
  return BASE.replace_once(text, old, new)


def find_layer23_ffn_payload(pattern: str, expected_bytes: int) -> Path:
  matches = sorted(LAYER23_FFN_PAYLOAD_ROOT.glob(pattern))
  if len(matches) != 1:
    raise SystemExit(f"expected one layer23 FFN payload for {pattern}, found {len(matches)}")
  path = matches[0].resolve()
  if path.stat().st_size != expected_bytes:
    raise SystemExit(f"payload size mismatch for {path}: {path.stat().st_size}")
  return path


def add_layer23_ffn_payloads(payloads: dict[str, dict[str, Any]]) -> None:
  for name, (stage_name, pattern, expected_bytes) in LAYER23_FFN_PAYLOAD_SPECS.items():
    payloads[name] = BASE.L19.payload_record(
        find_layer23_ffn_payload(pattern, expected_bytes),
        stage_name,
        expected_bytes,
    )


def resolve_prefixed_full_attention_payloads(
    history_json: Path,
    layer: int,
    prefix: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
  payloads, history = BASE_RESOLVE_PREFIXED_FULL_ATTENTION_PAYLOADS(
      history_json,
      layer,
      prefix,
  )
  if INCLUDE_FFN_LOUT and prefix == "l23_":
    add_layer23_ffn_payloads(payloads)
  return payloads, history


ORACLE_INPUT_BLOCK = r'''
    const auto layer18_oracle_input_native_qkv =
        iq36::run_qwen36_full_attention_qkv_projection(
            args.model_path, index, layer18, layer18_oracle.residual_input,
            rms_norm_epsilon);
    const auto layer18_oracle_input_rms_gpu = RunGpuLayerInputRmsNorm(
        layer18_oracle.residual_input, layer18_attn_norm_weight,
        rms_norm_epsilon, args.device_substring, args.repeat);
    const auto layer18_oracle_input_qk_gpu = RunGpuFullAttentionQkFront(
        args.model_path, *layer18_tensors.q_tensor, *layer18_tensors.k_tensor,
        layer18_oracle_input_rms_gpu.attn_norm, args.device_substring,
        args.repeat);
    const auto layer18_oracle_input_v_gpu = RunGpuFullAttentionVAny(
        args.model_path, *layer18_tensors.v_tensor,
        layer18_oracle_input_rms_gpu.attn_norm, args.device_substring,
        args.repeat);
    const auto layer18_oracle_input_gpu_q_split =
        SplitFullAttentionQ(layer18_oracle_input_qk_gpu.q_full);
    const auto layer18_oracle_input_gpu_q_normed = ApplyRepeatedRmsNormFull(
        layer18_oracle_input_gpu_q_split.q_raw, layer18_q_norm_weight,
        rms_norm_epsilon);
    const auto layer18_oracle_input_gpu_k_normed = ApplyRepeatedRmsNormFull(
        layer18_oracle_input_qk_gpu.k_raw, layer18_k_norm_weight,
        rms_norm_epsilon);
    const auto layer18_oracle_input_gpu_rope =
        iq36::run_qwen36_full_attention_rope(
            layer18_oracle_input_gpu_q_normed,
            layer18_oracle_input_gpu_k_normed,
            kFullSourceTokenPosition, head_dim, rope_dimension_count,
            rope_sections, rope_context_length, rope_freq_base,
            kRopeFreqScale, kRopeExtFactor, kRopeAttnFactor,
            kRopeBetaFast, kRopeBetaSlow);
    const auto layer18_oracle_input_native_rope =
        iq36::run_qwen36_full_attention_rope(
            layer18_oracle_input_native_qkv.q_normed,
            layer18_oracle_input_native_qkv.k_normed,
            kFullSourceTokenPosition, head_dim, rope_dimension_count,
            rope_sections, rope_context_length, rope_freq_base,
            kRopeFreqScale, kRopeExtFactor, kRopeAttnFactor,
            kRopeBetaFast, kRopeBetaSlow);
    const auto layer18_oracle_input_gpu_k_history_flat =
        FlattenFullAttentionHistory(layer18_oracle.k_history,
                                    layer18_oracle.k_rope);
    const auto layer18_oracle_input_gpu_v_history_flat =
        FlattenFullAttentionHistory(layer18_oracle.v_history,
                                    layer18_oracle_input_v_gpu.v);
    const auto layer18_oracle_input_core_gate_gpu =
        RunGpuFullAttentionCoreGate(
            layer18_oracle.q_rope,
            layer18_oracle_input_gpu_k_history_flat,
            layer18_oracle_input_gpu_v_history_flat,
            layer18_oracle_input_qk_gpu.q_full,
            args.device_substring, args.repeat);
    const auto layer18_oracle_input_attention_gpu = RunGpuAttentionFront(
        args.model_path,
        *iq36::find_tensor(index, LayerTensorName(layer18, "attn_output.weight")),
        layer18_oracle_input_core_gate_gpu.attn_gated,
        layer18_oracle.residual_input,
        layer18_ffn_norm_weight,
        rms_norm_epsilon,
        args.device_substring,
        args.repeat);
    const auto layer18_oracle_input_native_attn_residual =
        iq36::add_vectors(layer18_oracle.residual_input,
                          layer18_native_attn_output);
    const auto layer18_oracle_input_native_attn_post_norm =
        iq36::apply_rms_norm(layer18_oracle_input_native_attn_residual,
                             layer18_ffn_norm_weight, rms_norm_epsilon);

    std::vector<NamedCompareGroup> layer18_oracle_input_strict_groups;
    AppendCpuGpuOracleCompare(layer18_oracle_input_strict_groups,
                              "l23_oracle_input_attn_norm",
                              layer18_oracle_input_native_qkv.attention_norm,
                              layer18_oracle_input_rms_gpu.attn_norm,
                              layer18_oracle.attn_norm);
    AppendCpuGpuOracleCompare(layer18_oracle_input_strict_groups,
                              "l23_oracle_input_q_full",
                              layer18_oracle_input_native_qkv.q_full,
                              layer18_oracle_input_qk_gpu.q_full,
                              layer18_oracle.q_full);
    AppendCpuGpuOracleCompare(layer18_oracle_input_strict_groups,
                              "l23_oracle_input_q_rope",
                              layer18_oracle_input_native_rope.q_rope,
                              layer18_oracle_input_gpu_rope.q_rope,
                              layer18_oracle.q_rope);
    AppendCpuGpuOracleCompare(layer18_oracle_input_strict_groups,
                              "l23_oracle_input_k_rope",
                              layer18_oracle_input_native_rope.k_rope,
                              layer18_oracle_input_gpu_rope.k_rope,
                              layer18_oracle.k_rope);
    AppendCpuGpuOracleCompare(layer18_oracle_input_strict_groups,
                              "l23_oracle_input_v",
                              layer18_oracle_input_native_qkv.v,
                              layer18_oracle_input_v_gpu.v,
                              layer18_oracle.v);
    const auto layer18_oracle_input_k_raw_gpu_vs_cpu = iq36::compare_vectors(
        layer18_oracle_input_qk_gpu.k_raw,
        layer18_oracle_input_native_qkv.k_raw,
        kMismatchThreshold);
    const bool layer18_oracle_input_strict_ok =
        CompareGroupsPassed(layer18_oracle_input_strict_groups) &&
        ComparePassed(layer18_oracle_input_k_raw_gpu_vs_cpu);

    std::vector<NamedCompareGroup> layer18_oracle_input_full_groups;
    AppendFullAttentionComponentCompare(
        layer18_oracle_input_full_groups, "l23_oracle_input_attn_pregate",
        layer18_native_core.attn_pregate,
        layer18_oracle_input_core_gate_gpu.attn_pregate,
        layer18_oracle.attn_pregate);
    AppendFullAttentionComponentCompare(
        layer18_oracle_input_full_groups, "l23_oracle_input_attn_gated",
        layer18_native_gate.attn_gated,
        layer18_oracle_input_core_gate_gpu.attn_gated,
        layer18_oracle.attn_gated);
    AppendFullAttentionComponentCompare(
        layer18_oracle_input_full_groups, "l23_oracle_input_attn_output",
        layer18_native_attn_output,
        layer18_oracle_input_attention_gpu.linear_attn_out,
        layer18_oracle.attn_output);
    AppendFullAttentionComponentCompare(
        layer18_oracle_input_full_groups, "l23_oracle_input_attn_residual",
        layer18_oracle_input_native_attn_residual,
        layer18_oracle_input_attention_gpu.attn_residual,
        layer18_oracle_attn_residual);
    AppendFullAttentionComponentCompare(
        layer18_oracle_input_full_groups, "l23_oracle_input_attn_post_norm",
        layer18_oracle_input_native_attn_post_norm,
        layer18_oracle_input_attention_gpu.attn_post_norm,
        layer18_oracle_attn_post_norm);
    bool layer18_oracle_input_full_ok = true;
    for (const auto& group : layer18_oracle_input_full_groups) {
      layer18_oracle_input_full_ok =
          layer18_oracle_input_full_ok &&
          ComparePassedFullAttentionComponent(group.gpu_vs_oracle);
    }
    const bool layer18_oracle_input_ok =
        layer18_oracle_input_strict_ok && layer18_oracle_input_full_ok;
'''

LAYER23_FFN_RUN_BLOCK = r'''
    const auto& layer18_ffn_input = layer18_attention_gpu.attn_post_norm;
    const auto layer18_native_selected_gate_up =
        iq36::matvec_expert_tensor(args.model_path, index,
                                   layer18_selected_gate_up_tensor_name,
                                   layer18_ffn_input, layer18_oracle_expert_ids);
    const auto layer18_native_selected_swiglu =
        iq36::apply_swiglu_from_gate_up(layer18_native_selected_gate_up,
                                        kIntermediateSize, kExpertUsedCount);
    const auto layer18_native_selected_down =
        iq36::matvec_expert_tensor_per_expert_input(
            args.model_path, index, layer18_selected_down_tensor_name,
            layer18_native_selected_swiglu, layer18_oracle_expert_ids);
    const auto layer18_native_weighted =
        iq36::apply_expert_weights(layer18_native_selected_down,
                                   layer18_oracle_weights_norm, kHiddenSize);
    const auto layer18_native_moe_out =
        iq36::aggregate_experts(layer18_native_weighted, kExpertUsedCount, kHiddenSize);
    const auto layer18_native_shared_gate =
        iq36::matvec_tensor(args.model_path, index,
                            layer18_shared_gate_tensor_name, layer18_ffn_input);
    const auto layer18_native_shared_up =
        iq36::matvec_tensor(args.model_path, index,
                            layer18_shared_up_tensor_name, layer18_ffn_input);
    std::vector<float> layer18_native_shared_gate_up;
    layer18_native_shared_gate_up.reserve(
        layer18_native_shared_gate.size() + layer18_native_shared_up.size());
    layer18_native_shared_gate_up.insert(layer18_native_shared_gate_up.end(),
                                         layer18_native_shared_gate.begin(),
                                         layer18_native_shared_gate.end());
    layer18_native_shared_gate_up.insert(layer18_native_shared_gate_up.end(),
                                         layer18_native_shared_up.begin(),
                                         layer18_native_shared_up.end());
    const auto layer18_native_shared_swiglu =
        iq36::apply_swiglu_from_gate_up(layer18_native_shared_gate_up, kIntermediateSize, 1);
    const auto layer18_native_shared_down =
        iq36::matvec_tensor(args.model_path, index, layer18_shared_down_tensor_name,
                            layer18_native_shared_swiglu);
    const auto layer18_native_shared_input_gate =
        iq36::matvec_tensor(args.model_path, index,
                            layer18_shared_input_gate_tensor_name, layer18_ffn_input);
    Require(layer18_native_shared_input_gate.size() == 1,
            "native layer23 shared input gate size mismatch");
    const std::vector<float> layer18_native_shared_sigmoid{
        iq36::sigmoid_scalar(layer18_native_shared_input_gate[0])};
    const auto layer18_native_shared_gated =
        iq36::multiply_by_scalar(layer18_native_shared_down,
                                 layer18_native_shared_sigmoid[0]);
    const auto layer18_native_ffn_out =
        iq36::add_vectors(layer18_native_moe_out, layer18_native_shared_gated);
    const auto layer18_native_layer_output =
        iq36::add_vectors(layer18_attention_gpu.attn_residual, layer18_native_ffn_out);

    const auto layer18_selected_gpu = RunGpuSelectedFfnShell(
        args.model_path, *layer18_selected_gate_up_tensor, *layer18_selected_down_tensor,
        layer18_ffn_input, layer18_oracle_expert_ids, args.device_substring, args.repeat);
    const auto layer18_shared_gpu = RunGpuSharedFfnShell(
        args.model_path, *layer18_shared_gate_tensor, *layer18_shared_up_tensor,
        *layer18_shared_down_tensor, layer18_ffn_input, args.device_substring, args.repeat);
    const auto layer18_tail_gpu = RunGpuShell(
        layer18_shared_input_gate_weights, layer18_ffn_input,
        layer18_selected_gpu.down, layer18_oracle_weights_norm,
        layer18_shared_gpu.down, layer18_attention_gpu.attn_residual,
        args.device_substring, args.repeat);
'''


def apply_layer23_ffn_lout_patch(cpp: str) -> str:
  if not INCLUDE_FFN_LOUT:
    return cpp

  cpp = replace_once(
      cpp,
      '''    const auto layer18_oracle_attn_post_norm =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "l23_full_attn_post_norm.bin"));
    const bool layer18_payload_counts_ok =
        FullAttentionPayloadCountsOk(layer18_oracle) &&
        layer18_oracle_attn_residual.size() == kHiddenSize &&
        layer18_oracle_attn_post_norm.size() == kHiddenSize;
''',
      '''    const auto layer18_oracle_attn_post_norm =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "l23_full_attn_post_norm.bin"));
    const auto layer18_oracle_expert_ids =
        ReadI32VectorFile(JoinPath(args.payload_dir, "l23_ffn_moe_topk.bin"));
    const auto layer18_oracle_weights_norm =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "l23_ffn_moe_weights_norm.bin"));
    const auto layer18_oracle_selected_gate_up =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "l23_ffn_moe_gate_up.bin"));
    const auto layer18_oracle_selected_swiglu =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "l23_ffn_moe_swiglu.bin"));
    const auto layer18_oracle_selected_down =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "l23_ffn_moe_down.bin"));
    const auto layer18_oracle_weighted =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "l23_ffn_moe_weighted.bin"));
    const auto layer18_oracle_moe_out =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "l23_ffn_moe_out.bin"));
    const auto layer18_oracle_shared_down =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "l23_ffn_shexp.bin"));
    const auto layer18_oracle_shared_gate =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "l23_shared_expert_gate.bin"));
    const auto layer18_oracle_shared_sigmoid =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "l23_shared_expert_gate_sigmoid.bin"));
    const auto layer18_oracle_shared_gated =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "l23_ffn_shexp_gated.bin"));
    const auto layer18_oracle_ffn_out =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "l23_ffn_out.bin"));
    const auto layer18_oracle_layer_output =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "l23_l_out.bin"));
    const bool layer18_payload_counts_ok =
        FullAttentionPayloadCountsOk(layer18_oracle) &&
        layer18_oracle_attn_residual.size() == kHiddenSize &&
        layer18_oracle_attn_post_norm.size() == kHiddenSize &&
        layer18_oracle_expert_ids.size() == kExpertUsedCount &&
        layer18_oracle_weights_norm.size() == kExpertUsedCount &&
        layer18_oracle_selected_gate_up.size() == kGateUpValueCount &&
        layer18_oracle_selected_swiglu.size() == kSwiGluValueCount &&
        layer18_oracle_selected_down.size() == kWeightedValueCount &&
        layer18_oracle_weighted.size() == kWeightedValueCount &&
        layer18_oracle_moe_out.size() == kHiddenSize &&
        layer18_oracle_shared_down.size() == kHiddenSize &&
        layer18_oracle_shared_gate.size() == 1 &&
        layer18_oracle_shared_sigmoid.size() == 1 &&
        layer18_oracle_shared_gated.size() == kHiddenSize &&
        layer18_oracle_ffn_out.size() == kHiddenSize &&
        layer18_oracle_layer_output.size() == kHiddenSize;
''',
  )
  cpp = replace_once(
      cpp,
      '''    const auto layer18_ffn_norm_weight =
        iq36::decode_tensor_row(args.model_path, index,
                                LayerTensorName(layer18, "post_attention_norm.weight"), 0);
    const std::string layer14_selected_gate_up_tensor_name =
''',
      '''    const auto layer18_ffn_norm_weight =
        iq36::decode_tensor_row(args.model_path, index,
                                LayerTensorName(layer18, "post_attention_norm.weight"), 0);
    const std::string layer18_selected_gate_up_tensor_name =
        LayerTensorName(layer18, "ffn_gate_up_exps.weight");
    const std::string layer18_selected_down_tensor_name =
        LayerTensorName(layer18, "ffn_down_exps.weight");
    const std::string layer18_shared_gate_tensor_name =
        LayerTensorName(layer18, "ffn_gate_shexp.weight");
    const std::string layer18_shared_up_tensor_name =
        LayerTensorName(layer18, "ffn_up_shexp.weight");
    const std::string layer18_shared_down_tensor_name =
        LayerTensorName(layer18, "ffn_down_shexp.weight");
    const std::string layer18_shared_input_gate_tensor_name =
        LayerTensorName(layer18, "ffn_gate_inp_shexp.weight");
    const auto* layer18_selected_gate_up_tensor =
        iq36::find_tensor(index, layer18_selected_gate_up_tensor_name);
    const auto* layer18_selected_down_tensor =
        iq36::find_tensor(index, layer18_selected_down_tensor_name);
    const auto* layer18_shared_gate_tensor =
        iq36::find_tensor(index, layer18_shared_gate_tensor_name);
    const auto* layer18_shared_up_tensor =
        iq36::find_tensor(index, layer18_shared_up_tensor_name);
    const auto* layer18_shared_down_tensor =
        iq36::find_tensor(index, layer18_shared_down_tensor_name);
    const auto* layer18_shared_input_gate_tensor =
        iq36::find_tensor(index, layer18_shared_input_gate_tensor_name);
    Require(layer18_selected_gate_up_tensor != nullptr, "layer23 selected gate-up tensor missing");
    Require(layer18_selected_down_tensor != nullptr, "layer23 selected down tensor missing");
    Require(layer18_shared_gate_tensor != nullptr, "layer23 shared gate tensor missing");
    Require(layer18_shared_up_tensor != nullptr, "layer23 shared up tensor missing");
    Require(layer18_shared_down_tensor != nullptr, "layer23 shared down tensor missing");
    Require(layer18_shared_input_gate_tensor != nullptr, "layer23 shared input gate tensor missing");
    const auto layer18_shared_input_gate_weights =
        ReadF32TensorPayload(model, *layer18_shared_input_gate_tensor,
                             static_cast<std::size_t>(kHiddenSize));
    const bool layer18_ffn_tensor_shapes_ok =
        layer18_selected_gate_up_tensor->type == 12 &&
        (layer18_selected_down_tensor->type == 12 || layer18_selected_down_tensor->type == 14) &&
        layer18_shared_gate_tensor->type == 12 &&
        layer18_shared_up_tensor->type == 12 &&
        (layer18_shared_down_tensor->type == 12 || layer18_shared_down_tensor->type == 14) &&
        layer18_shared_input_gate_tensor->type == 0 &&
        layer18_selected_gate_up_tensor->dims ==
            std::vector<std::uint64_t>{kHiddenSize, kGateUpRowsPerExpert, kExpertCount} &&
        layer18_selected_down_tensor->dims ==
            std::vector<std::uint64_t>{kIntermediateSize, kHiddenSize, kExpertCount} &&
        layer18_shared_gate_tensor->dims ==
            std::vector<std::uint64_t>{kHiddenSize, kIntermediateSize} &&
        layer18_shared_up_tensor->dims ==
            std::vector<std::uint64_t>{kHiddenSize, kIntermediateSize} &&
        layer18_shared_down_tensor->dims ==
            std::vector<std::uint64_t>{kIntermediateSize, kHiddenSize} &&
        layer18_shared_input_gate_tensor->dims == std::vector<std::uint64_t>{kHiddenSize};
    const std::string layer14_selected_gate_up_tensor_name =
''',
  )
  cpp = replace_once(
      cpp,
      '''    const bool layer18_oracle_input_ok =
        layer18_oracle_input_strict_ok && layer18_oracle_input_full_ok;

    std::vector<NamedCompareGroup> strict_groups;
''',
      '''    const bool layer18_oracle_input_ok =
        layer18_oracle_input_strict_ok && layer18_oracle_input_full_ok;
''' + LAYER23_FFN_RUN_BLOCK + '''
    std::vector<NamedCompareGroup> strict_groups;
''',
  )
  cpp = replace_once(
      cpp,
      '''    AppendFullAttentionComponentCompare(layer18_full_attention_groups, "l23_attn_post_norm",
                                        layer18_native_attn_post_norm,
                                        layer18_attention_gpu.attn_post_norm,
                                        layer18_oracle_attn_post_norm);
    const auto layer18_k_raw_gpu_vs_cpu = iq36::compare_vectors(
''',
      '''    AppendFullAttentionComponentCompare(layer18_full_attention_groups, "l23_attn_post_norm",
                                        layer18_native_attn_post_norm,
                                        layer18_attention_gpu.attn_post_norm,
                                        layer18_oracle_attn_post_norm);

    std::vector<NamedCompareGroup> layer18_ffn_live_groups;
    AppendCpuGpuOracleCompare(layer18_ffn_live_groups, "l23_selected_gate_up",
                              layer18_native_selected_gate_up,
                              layer18_selected_gpu.gate_up,
                              layer18_oracle_selected_gate_up);
    AppendCpuGpuOracleCompare(layer18_ffn_live_groups, "l23_selected_swiglu",
                              layer18_native_selected_swiglu,
                              layer18_selected_gpu.swiglu,
                              layer18_oracle_selected_swiglu);
    AppendCpuGpuOracleCompare(layer18_ffn_live_groups, "l23_selected_down",
                              layer18_native_selected_down,
                              layer18_selected_gpu.down,
                              layer18_oracle_selected_down);
    AppendCpuGpuOracleCompare(layer18_ffn_live_groups, "l23_ffn_moe_weighted",
                              layer18_native_weighted,
                              layer18_tail_gpu.weighted,
                              layer18_oracle_weighted);
    AppendCpuGpuOracleCompare(layer18_ffn_live_groups, "l23_ffn_moe_out",
                              layer18_native_moe_out,
                              layer18_tail_gpu.moe_out,
                              layer18_oracle_moe_out);
    AppendCpuGpuOracleCompare(layer18_ffn_live_groups, "l23_shared_down",
                              layer18_native_shared_down,
                              layer18_shared_gpu.down,
                              layer18_oracle_shared_down);
    AppendCpuGpuOracleCompare(layer18_ffn_live_groups, "l23_shared_gate",
                              layer18_native_shared_input_gate,
                              layer18_tail_gpu.shared_gate,
                              layer18_oracle_shared_gate);
    AppendCpuGpuOracleCompare(layer18_ffn_live_groups, "l23_shared_gate_sigmoid",
                              layer18_native_shared_sigmoid,
                              layer18_tail_gpu.shared_gate_sigmoid,
                              layer18_oracle_shared_sigmoid);
    AppendCpuGpuOracleCompare(layer18_ffn_live_groups, "l23_ffn_shexp_gated",
                              layer18_native_shared_gated,
                              layer18_tail_gpu.shared_gated,
                              layer18_oracle_shared_gated);
    AppendCpuGpuOracleCompare(layer18_ffn_live_groups, "l23_ffn_out",
                              layer18_native_ffn_out,
                              layer18_tail_gpu.ffn_out,
                              layer18_oracle_ffn_out);
    AppendCpuGpuOracleCompare(layer18_ffn_live_groups, "l23_layer_output",
                              layer18_native_layer_output,
                              layer18_tail_gpu.layer_output,
                              layer18_oracle_layer_output);
    const auto layer18_k_raw_gpu_vs_cpu = iq36::compare_vectors(
''',
  )
  cpp = replace_once(
      cpp,
      '''    const bool layer18_comparisons_ok =
        layer18_strict_input_ok && layer18_full_component_ok;
    const bool layer18_timing_positive =
''',
      '''    const bool layer18_comparisons_ok =
        layer18_strict_input_ok && layer18_full_component_ok;
    bool layer18_live_ffn_gpu_cpu_ok = true;
    for (const auto& group : layer18_ffn_live_groups) {
      layer18_live_ffn_gpu_cpu_ok =
          layer18_live_ffn_gpu_cpu_ok && ComparePassed(group.gpu_vs_cpu);
    }
    const bool layer18_live_ffn_oracle_magnitude_ok =
        live_ffn_oracle_magnitude_passed(layer18_ffn_live_groups[2].gpu_vs_oracle) &&
        live_ffn_oracle_magnitude_passed(layer18_ffn_live_groups[5].gpu_vs_oracle) &&
        live_ffn_oracle_magnitude_passed(layer18_ffn_live_groups[9].gpu_vs_oracle) &&
        live_ffn_oracle_magnitude_passed(layer18_ffn_live_groups[10].gpu_vs_oracle);
    const bool layer18_live_layer_output_full_policy_ok =
        ComparePassedFullAttentionComponent(layer18_ffn_live_groups[10].gpu_vs_oracle);
    const bool layer18_live_ffn_lout_ok =
        layer18_live_ffn_gpu_cpu_ok &&
        layer18_live_ffn_oracle_magnitude_ok &&
        layer18_live_layer_output_full_policy_ok;
    const bool layer18_timing_positive =
''',
  )
  cpp = replace_once(
      cpp,
      '''        layer18_core_gate_gpu.timing.gate_min_us > 0.0 &&
        layer18_attention_gpu.timing.output_projection_min_us > 0.0 &&
        layer18_attention_gpu.timing.residual_add_min_us > 0.0 &&
        layer18_attention_gpu.timing.ffn_rmsnorm_min_us > 0.0;
''',
      '''        layer18_core_gate_gpu.timing.gate_min_us > 0.0 &&
        layer18_attention_gpu.timing.output_projection_min_us > 0.0 &&
        layer18_attention_gpu.timing.residual_add_min_us > 0.0 &&
        layer18_attention_gpu.timing.ffn_rmsnorm_min_us > 0.0 &&
        layer18_selected_gpu.timing.gate_up_min_us > 0.0 &&
        layer18_selected_gpu.timing.swiglu_min_us > 0.0 &&
        layer18_selected_gpu.timing.down_min_us > 0.0 &&
        layer18_shared_gpu.timing.gate_min_us > 0.0 &&
        layer18_shared_gpu.timing.up_min_us > 0.0 &&
        layer18_shared_gpu.timing.swiglu_min_us > 0.0 &&
        layer18_shared_gpu.timing.down_min_us > 0.0 &&
        layer18_tail_gpu.timing.shell_sum_min_us > 0.0;
''',
  )
  cpp = replace_once(
      cpp,
      '''        layer18_v_gpu.device_name.find(args.device_substring) != std::string::npos &&
        layer18_core_gate_gpu.device_name.find(args.device_substring) != std::string::npos &&
        layer18_attention_gpu.device_name.find(args.device_substring) != std::string::npos;
''',
      '''        layer18_v_gpu.device_name.find(args.device_substring) != std::string::npos &&
        layer18_core_gate_gpu.device_name.find(args.device_substring) != std::string::npos &&
        layer18_attention_gpu.device_name.find(args.device_substring) != std::string::npos &&
        layer18_selected_gpu.device_name.find(args.device_substring) != std::string::npos &&
        layer18_shared_gpu.device_name.find(args.device_substring) != std::string::npos &&
        layer18_tail_gpu.device_name.find(args.device_substring) != std::string::npos;
''',
  )
  cpp = replace_once(
      cpp,
      '''    const bool layer18_v_gpu_boundary =
        (layer18_tensors.v_tensor->type == 12 || layer18_tensors.v_tensor->type == 14) &&
        layer18_v_gpu.v.size() == kFullKvValues;
    const bool layer18_ok =
''',
      '''    const bool layer18_v_gpu_boundary =
        (layer18_tensors.v_tensor->type == 12 || layer18_tensors.v_tensor->type == 14) &&
        layer18_v_gpu.v.size() == kFullKvValues;
    const bool layer18_ffn_down_q4_q6_boundary =
        (layer18_selected_down_tensor->type == 12 || layer18_selected_down_tensor->type == 14) &&
        (layer18_shared_down_tensor->type == 12 || layer18_shared_down_tensor->type == 14) &&
        layer18_selected_gpu.down.size() == kWeightedValueCount &&
        layer18_shared_gpu.down.size() == kHiddenSize &&
        layer18_tail_gpu.layer_output.size() == kHiddenSize;
    const bool layer18_ffn_q6_boundary =
        layer18_selected_down_tensor->type == 14 &&
        layer18_shared_down_tensor->type == 14;
    const bool layer18_ok =
''',
  )
  cpp = replace_once(
      cpp,
      '''        metadata_ok &&
        layer18_comparisons_ok &&
        layer18_timing_positive &&
        layer18_arc_selected &&
        layer18_v_gpu_boundary;
''',
      '''        metadata_ok &&
        layer18_comparisons_ok &&
        layer18_live_ffn_lout_ok &&
        layer18_timing_positive &&
        layer18_arc_selected &&
        layer18_v_gpu_boundary &&
        layer18_ffn_tensor_shapes_ok &&
        layer18_ffn_down_q4_q6_boundary;
''',
  )
  cpp = replace_once(
      cpp,
      '''    const double layer18_attention_sum_min =
        layer18_input_sum_min + layer18_core_output_sum_min;
''',
      '''    const double layer18_attention_sum_min =
        layer18_input_sum_min + layer18_core_output_sum_min;
    const double layer18_ffn_sum_min =
        layer18_selected_gpu.timing.selected_ffn_kernel_sum_min_us +
        layer18_shared_gpu.timing.shared_ffn_kernel_sum_min_us +
        layer18_tail_gpu.timing.shell_sum_min_us;
''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << "\\"layer23_v_projection_boundary\\":\\""
              << (layer18_tensors.v_tensor->type == 14 ? "gpu_q6_raw_matvec" : "gpu_q4x8_matvec") << "\\",";
    std::cout << "\\"layer23_ffn_boundary\\":\\"q4_q6_down_pending\\",";
    std::cout << "\\"full_attn_core_input_boundary\\":\\"captured_rope_kv_payloads\\",";''',
      '''    std::cout << "\\"layer23_v_projection_boundary\\":\\""
              << (layer18_tensors.v_tensor->type == 14 ? "gpu_q6_raw_matvec" : "gpu_q4x8_matvec") << "\\",";
    std::cout << "\\"layer23_ffn_boundary\\":\\"gpu_live_post_norm_to_q4_q6_down\\",";
    std::cout << "\\"layer23_ffn_input_boundary\\":\\"live_gpu_layer23_post_attention_norm\\",";
    std::cout << "\\"layer23_layer_output_residual_boundary\\":\\"live_gpu_layer23_attention_residual\\",";
    std::cout << "\\"layer23_lout_boundary\\":\\"live_gpu_l_out_23\\",";
    std::cout << "\\"layer23_selected_down_tensor_type\\":\\""
              << JsonEscape(iq36::ggml_type_name(layer18_selected_down_tensor->type)) << "\\",";
    std::cout << "\\"layer23_shared_down_tensor_type\\":\\""
              << JsonEscape(iq36::ggml_type_name(layer18_shared_down_tensor->type)) << "\\",";
    std::cout << "\\"full_attn_core_input_boundary\\":\\"captured_rope_kv_payloads\\",";''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << "\\"layer23_output_projection_device_name\\":\\"" << JsonEscape(layer18_attention_gpu.device_name) << "\\",";
    std::cout << "\\"layer23_v_projection_device_name\\":\\"" << JsonEscape(layer18_v_gpu.device_name) << "\\",";
    std::cout << "\\"layer7_v_tensor_type\\":\\""
''',
      '''    std::cout << "\\"layer23_output_projection_device_name\\":\\"" << JsonEscape(layer18_attention_gpu.device_name) << "\\",";
    std::cout << "\\"layer23_v_projection_device_name\\":\\"" << JsonEscape(layer18_v_gpu.device_name) << "\\",";
    std::cout << "\\"layer23_selected_ffn_device_name\\":\\"" << JsonEscape(layer18_selected_gpu.device_name) << "\\",";
    std::cout << "\\"layer23_shared_ffn_device_name\\":\\"" << JsonEscape(layer18_shared_gpu.device_name) << "\\",";
    std::cout << "\\"layer23_lout_device_name\\":\\"" << JsonEscape(layer18_tail_gpu.device_name) << "\\",";
    std::cout << "\\"layer7_v_tensor_type\\":\\""
''',
  )
  cpp = replace_once(
      cpp,
      '''                  layer18_v_gpu.program_build_ms +
                  layer18_core_gate_gpu.program_build_ms +
                  layer18_attention_gpu.program_build_ms)
''',
      '''                  layer18_v_gpu.program_build_ms +
                  layer18_core_gate_gpu.program_build_ms +
                  layer18_attention_gpu.program_build_ms +
                  layer18_selected_gpu.program_build_ms +
                  layer18_shared_gpu.program_build_ms +
                  layer18_tail_gpu.program_build_ms)
''',
  )
  cpp = replace_once(
      cpp,
      '''                            layer18_v_gpu.build_log +
                            layer18_core_gate_gpu.build_log +
                            layer18_attention_gpu.build_log)
''',
      '''                            layer18_v_gpu.build_log +
                            layer18_core_gate_gpu.build_log +
                            layer18_attention_gpu.build_log +
                            layer18_selected_gpu.build_log +
                            layer18_shared_gpu.build_log +
                            layer18_tail_gpu.build_log)
''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << "\\"resident_layer23_full_attention_kernel_sum_min_us\\":"
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
      '''    std::cout << "\\"resident_layer23_full_attention_kernel_sum_min_us\\":"
              << layer18_attention_sum_min << ",";
    std::cout << "\\"resident_layer23_ffn_lout_kernel_sum_min_us\\":"
              << layer18_ffn_sum_min << ",";
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
                  layer17_state_input_sum_min + layer18_attention_sum_min) << ",";
    std::cout << "\\"resident_layer5_6_7_8_9_10_11_12_13_14_15_16_17_18_19_20_21_22_to_layer23_lout_kernel_sum_min_us\\":"
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
                  layer17_state_input_sum_min + layer18_attention_sum_min +
                  layer18_ffn_sum_min);
''',
  )
  cpp = replace_once(
      cpp,
      '''    WriteNamedCompareGroups(layer18_full_attention_groups);
    std::cout << ",";
    WriteNamedCompareGroups(layer18_oracle_input_strict_groups);
''',
      '''    WriteNamedCompareGroups(layer18_full_attention_groups);
    std::cout << ",";
    WriteNamedCompareGroups(layer18_ffn_live_groups);
    std::cout << ",";
    WriteNamedCompareGroups(layer18_oracle_input_strict_groups);
''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << "\\"layer23_v_gpu_boundary\\":"
              << (layer18_v_gpu_boundary ? "true" : "false") << ",";
    std::cout << "\\"layer23_arc_device_selected\\":"
              << (layer18_arc_selected ? "true" : "false") << ",";
''',
      '''    std::cout << "\\"layer23_v_gpu_boundary\\":"
              << (layer18_v_gpu_boundary ? "true" : "false") << ",";
    std::cout << "\\"layer23_ffn_tensor_shapes_ok\\":"
              << (layer18_ffn_tensor_shapes_ok ? "true" : "false") << ",";
    std::cout << "\\"layer23_ffn_down_q4_q6_boundary\\":"
              << (layer18_ffn_down_q4_q6_boundary ? "true" : "false") << ",";
    std::cout << "\\"layer23_live_ffn_gpu_cpu_matches_native\\":"
              << (layer18_live_ffn_gpu_cpu_ok ? "true" : "false") << ",";
    std::cout << "\\"layer23_live_ffn_oracle_magnitude_policy_matches\\":"
              << (layer18_live_ffn_oracle_magnitude_ok ? "true" : "false") << ",";
    std::cout << "\\"layer23_live_layer_output_full_policy_matches_oracle\\":"
              << (layer18_live_layer_output_full_policy_ok ? "true" : "false") << ",";
    std::cout << "\\"layer23_ffn_lout_handoff_matches\\":"
              << (layer18_live_ffn_lout_ok ? "true" : "false") << ",";
    std::cout << "\\"layer23_lout_boundary_live_gpu\\":true,";
    std::cout << "\\"layer23_ffn_q6_boundary\\":"
              << (layer18_ffn_q6_boundary ? "true" : "false") << ",";
    std::cout << "\\"layer23_arc_device_selected\\":"
              << (layer18_arc_selected ? "true" : "false") << ",";
''',
  )
  return cpp


def apply_l22_z_source_patch(cpp: str) -> str:
  if Z_SOURCE == DEFAULT_Z_SOURCE:
    return cpp

  if Z_SOURCE == "native":
    cpp = replace_once(
        cpp,
        '''  const auto delta_gpu = RunGpuPostConvDelta(
      preconv_gpu.q_conv_predelta,
      preconv_gpu.k_conv_predelta,
      preconv_gpu.v_conv_predelta,
      preconv_gpu.gate,
      preconv_gpu.beta_sigmoid,
      oracle.state,
      preconv_gpu.z,
      ssm_norm_weight,
      rms_norm_epsilon,
      args.device_substring,
      args.repeat);
''',
        '''  const std::vector<float>& z_for_delta =
      (t.layer == 22)
          ? static_cast<const std::vector<float>&>(native_preconv.z)
          : static_cast<const std::vector<float>&>(preconv_gpu.z);
  const auto delta_gpu = RunGpuPostConvDelta(
      preconv_gpu.q_conv_predelta,
      preconv_gpu.k_conv_predelta,
      preconv_gpu.v_conv_predelta,
      preconv_gpu.gate,
      preconv_gpu.beta_sigmoid,
      oracle.state,
      z_for_delta,
      ssm_norm_weight,
      rms_norm_epsilon,
      args.device_substring,
      args.repeat);
''',
    )
    cpp = replace_once(
        cpp,
        '''  AppendCompare(result.comparisons, "z",
                native_preconv.z, preconv_gpu.z, oracle.z);
''',
        '''  AppendCompare(result.comparisons, "z",
                native_preconv.z, preconv_gpu.z, oracle.z);
  if (t.layer == 22) {
    AppendCompare(result.comparisons, "z_native_correction_input",
                  native_preconv.z, z_for_delta, oracle.z);
  }
''',
    )
    cpp = replace_once(
        cpp,
        '''    std::cout << "\\"layer23_arc_device_selected\\":"
              << (layer18_arc_selected ? "true" : "false") << ",";
    std::cout << "\\"layer23_oracle_input_strict_matches_oracle\\":"
''',
        '''    std::cout << "\\"layer23_arc_device_selected\\":"
              << (layer18_arc_selected ? "true" : "false") << ",";
    std::cout << "\\"layer22_native_z_correction_diagnostic\\":true,";
    std::cout << "\\"layer23_oracle_input_strict_matches_oracle\\":"
''',
    )
    return cpp

  if Z_SOURCE == "gpu-f32":
    cpp = replace_once(
        cpp,
        '''  const auto delta_gpu = RunGpuPostConvDelta(
      preconv_gpu.q_conv_predelta,
      preconv_gpu.k_conv_predelta,
      preconv_gpu.v_conv_predelta,
      preconv_gpu.gate,
      preconv_gpu.beta_sigmoid,
      oracle.state,
      preconv_gpu.z,
      ssm_norm_weight,
      rms_norm_epsilon,
      args.device_substring,
      args.repeat);
''',
        '''  std::vector<float> gpu_f32_z_correction_input;
  if (t.layer == 22) {
    std::vector<float> z_weights_f32;
    z_weights_f32.reserve(static_cast<std::size_t>(kLinearVValues) *
                          static_cast<std::size_t>(kHiddenSize));
    for (int row = 0; row < kLinearVValues; ++row) {
      auto row_values = iq36::decode_tensor_row(
          args.model_path, index, t.z_tensor_name,
          static_cast<std::uint64_t>(row));
      Require(row_values.size() == static_cast<std::size_t>(kHiddenSize),
              "layer22 z F32 decode row width mismatch");
      z_weights_f32.insert(z_weights_f32.end(),
                           row_values.begin(), row_values.end());
    }
    iq36::GpuQ4X8MatvecRunner z_f32_runner(args.device_substring,
                                           kOpenClSource);
    const auto z_q8_input = iq36::QuantizeQ8KInputPlanes(
        layer_input_gpu.attn_norm);
    Require(z_q8_input.qs.size() == static_cast<std::size_t>(kHiddenSize),
            "layer22 z Q8 input qs size mismatch");
    Require(z_q8_input.d.size() == static_cast<std::size_t>(kHiddenSize / 256),
            "layer22 z Q8 input scale size mismatch");
    std::vector<float> z_q8_dequant_input(static_cast<std::size_t>(kHiddenSize),
                                          0.0f);
    for (std::size_t block = 0; block < z_q8_input.d.size(); ++block) {
      for (std::size_t col = 0; col < 256; ++col) {
        const std::size_t offset = block * 256 + col;
        z_q8_dequant_input[offset] =
            static_cast<float>(z_q8_input.qs[offset]) * z_q8_input.d[block];
      }
    }
    const auto z_f32_gpu = z_f32_runner.RunF32Matvec(
        z_weights_f32, z_q8_dequant_input, kLinearVValues,
        kHiddenSize, args.repeat);
    gpu_f32_z_correction_input = z_f32_gpu.output;
    Require(gpu_f32_z_correction_input.size() == preconv_gpu.z.size(),
            "layer22 GPU F32 z correction output size mismatch");
  }
  const std::vector<float>& z_for_delta =
      (t.layer == 22)
          ? static_cast<const std::vector<float>&>(gpu_f32_z_correction_input)
          : static_cast<const std::vector<float>&>(preconv_gpu.z);
  const auto delta_gpu = RunGpuPostConvDelta(
      preconv_gpu.q_conv_predelta,
      preconv_gpu.k_conv_predelta,
      preconv_gpu.v_conv_predelta,
      preconv_gpu.gate,
      preconv_gpu.beta_sigmoid,
      oracle.state,
      z_for_delta,
      ssm_norm_weight,
      rms_norm_epsilon,
      args.device_substring,
      args.repeat);
''',
    )
    cpp = replace_once(
        cpp,
        '''  AppendCompare(result.comparisons, "z",
                native_preconv.z, preconv_gpu.z, oracle.z);
''',
        '''  AppendCompare(result.comparisons, "z",
                native_preconv.z, preconv_gpu.z, oracle.z);
  if (t.layer == 22) {
    AppendCompare(result.comparisons, "z_gpu_f32_correction_input",
                  native_preconv.z, z_for_delta, oracle.z);
  }
''',
    )
    cpp = replace_once(
        cpp,
        '''    std::cout << "\\"layer23_arc_device_selected\\":"
              << (layer18_arc_selected ? "true" : "false") << ",";
    std::cout << "\\"layer23_oracle_input_strict_matches_oracle\\":"
''',
        '''    std::cout << "\\"layer23_arc_device_selected\\":"
              << (layer18_arc_selected ? "true" : "false") << ",";
    std::cout << "\\"layer22_gpu_f32_z_correction_diagnostic\\":true,";
    std::cout << "\\"layer23_oracle_input_strict_matches_oracle\\":"
''',
    )
    return cpp

  if Z_SOURCE == "gpu-q4-cpu-order":
    cpp = replace_once(
        cpp,
        '''  const auto delta_gpu = RunGpuPostConvDelta(
      preconv_gpu.q_conv_predelta,
      preconv_gpu.k_conv_predelta,
      preconv_gpu.v_conv_predelta,
      preconv_gpu.gate,
      preconv_gpu.beta_sigmoid,
      oracle.state,
      preconv_gpu.z,
      ssm_norm_weight,
      rms_norm_epsilon,
      args.device_substring,
      args.repeat);
''',
        '''  std::vector<float> gpu_q4_cpu_order_z_correction_input;
  if (t.layer == 22) {
    const auto z_q8_input = iq36::QuantizeQ8KInputPlanes(
        layer_input_gpu.attn_norm);
    const auto z_raw = ReadTensorBytes(model, *t.z_tensor);
    const auto z_cpu_order_gpu = iq36::RunQ4KCpuOrderMatvec(
        z_raw, z_q8_input, kLinearVValues, kHiddenSize / 256,
        args.device_substring, args.repeat);
    gpu_q4_cpu_order_z_correction_input = z_cpu_order_gpu.output;
    Require(gpu_q4_cpu_order_z_correction_input.size() == preconv_gpu.z.size(),
            "layer22 GPU Q4 CPU-order z correction output size mismatch");
  }
  const std::vector<float>& z_for_delta =
      (t.layer == 22)
          ? static_cast<const std::vector<float>&>(gpu_q4_cpu_order_z_correction_input)
          : static_cast<const std::vector<float>&>(preconv_gpu.z);
  const auto delta_gpu = RunGpuPostConvDelta(
      preconv_gpu.q_conv_predelta,
      preconv_gpu.k_conv_predelta,
      preconv_gpu.v_conv_predelta,
      preconv_gpu.gate,
      preconv_gpu.beta_sigmoid,
      oracle.state,
      z_for_delta,
      ssm_norm_weight,
      rms_norm_epsilon,
      args.device_substring,
      args.repeat);
''',
    )
    cpp = replace_once(
        cpp,
        '''  AppendCompare(result.comparisons, "z",
                native_preconv.z, preconv_gpu.z, oracle.z);
''',
        '''  AppendCompare(result.comparisons, "z",
                native_preconv.z, preconv_gpu.z, oracle.z);
  if (t.layer == 22) {
    AppendCompare(result.comparisons, "z_gpu_q4_cpu_order_correction_input",
                  native_preconv.z, z_for_delta, oracle.z);
  }
''',
    )
    cpp = replace_once(
        cpp,
        '''    std::cout << "\\"layer23_arc_device_selected\\":"
              << (layer18_arc_selected ? "true" : "false") << ",";
    std::cout << "\\"layer23_oracle_input_strict_matches_oracle\\":"
''',
        '''    std::cout << "\\"layer23_arc_device_selected\\":"
              << (layer18_arc_selected ? "true" : "false") << ",";
    std::cout << "\\"layer22_gpu_q4_cpu_order_z_correction_diagnostic\\":true,";
    std::cout << "\\"layer23_oracle_input_strict_matches_oracle\\":"
''',
    )
    return cpp

  raise SystemExit(f"unsupported z source: {Z_SOURCE}")


def diagnostic_cpp(opencl_source: str) -> str:
  cpp = BASE_CPP(opencl_source)
  cpp = replace_once(cpp, BASE.SCHEMA_VERSION, active_schema_version(Z_SOURCE))
  cpp = replace_once(
      cpp,
      '''  const auto delta_gpu = RunGpuPostConvDelta(
      preconv_gpu.q_conv_predelta,
      preconv_gpu.k_conv_predelta,
      preconv_gpu.v_conv_predelta,
      preconv_gpu.gate,
      preconv_gpu.beta_sigmoid,
      oracle.state,
      preconv_gpu.z,
      ssm_norm_weight,
      rms_norm_epsilon,
      args.device_substring,
      args.repeat);
  const auto attention_gpu = RunGpuAttentionFront(
''',
      '''  const auto delta_gpu = RunGpuPostConvDelta(
      preconv_gpu.q_conv_predelta,
      preconv_gpu.k_conv_predelta,
      preconv_gpu.v_conv_predelta,
      preconv_gpu.gate,
      preconv_gpu.beta_sigmoid,
      oracle.state,
      preconv_gpu.z,
      ssm_norm_weight,
      rms_norm_epsilon,
      args.device_substring,
      args.repeat);
  std::vector<float> gpu_input_native_conv_output_raw;
  std::vector<float> gpu_input_native_final_output;
  if (t.layer == 22) {
    const auto gpu_input_native_conv =
        iq36::run_qwen36_linear_attention_conv_core(
            args.model_path, index, t.layer, preconv_gpu.qkv_mixed,
            oracle.conv_state);
    const auto gpu_input_native_postconv =
        iq36::run_qwen36_linear_attention_postconv_core(
            gpu_input_native_conv.conv_output_raw,
            preconv_gpu.gate,
            preconv_gpu.beta_sigmoid,
            oracle.state,
            preconv_gpu.z,
            ssm_norm_weight,
            rms_norm_epsilon);
    gpu_input_native_conv_output_raw = gpu_input_native_conv.conv_output_raw;
    gpu_input_native_final_output = gpu_input_native_postconv.final_output;
  }
  const auto attention_gpu = RunGpuAttentionFront(
''',
  )
  cpp = replace_once(
      cpp,
      '''  const auto attention_gpu = RunGpuAttentionFront(
      args.model_path, *t.output_tensor, delta_gpu.final_output, gpu_residual_input,
      ffn_norm_weight, rms_norm_epsilon, args.device_substring, args.repeat);
  const auto selected_gpu = RunGpuSelectedFfnShell(
''',
      '''  const auto attention_gpu = RunGpuAttentionFront(
      args.model_path, *t.output_tensor, delta_gpu.final_output, gpu_residual_input,
      ffn_norm_weight, rms_norm_epsilon, args.device_substring, args.repeat);
  std::vector<float> gpu_input_native_linear_attn_out;
  std::vector<float> gpu_input_native_attn_residual;
  std::vector<float> gpu_input_native_attn_post_norm;
  if (t.layer == 22) {
    gpu_input_native_linear_attn_out =
        iq36::matvec_tensor(args.model_path, index, t.output_tensor_name,
                            delta_gpu.final_output);
    gpu_input_native_attn_residual =
        iq36::add_vectors(gpu_residual_input, gpu_input_native_linear_attn_out);
    gpu_input_native_attn_post_norm =
        iq36::apply_rms_norm(gpu_input_native_attn_residual,
                             ffn_norm_weight, rms_norm_epsilon);
  }
  const auto selected_gpu = RunGpuSelectedFfnShell(
''',
  )
  cpp = replace_once(
      cpp,
      '''  AppendCompare(result.comparisons, "conv_output_raw",
                native_conv.conv_output_raw, preconv_gpu.conv_output_raw,
                oracle.conv_output_raw);
''',
      '''  AppendCompare(result.comparisons, "conv_output_raw",
                native_conv.conv_output_raw, preconv_gpu.conv_output_raw,
                oracle.conv_output_raw);
  if (t.layer == 22) {
    AppendCompare(result.comparisons, "conv_output_raw_same_gpu_qkv",
                  gpu_input_native_conv_output_raw, preconv_gpu.conv_output_raw,
                  oracle.conv_output_raw);
  }
''',
  )
  cpp = replace_once(
      cpp,
      '''  AppendCompare(result.comparisons, "final_output",
                native_postconv.final_output, delta_gpu.final_output,
                oracle.final_output);
''',
      '''  AppendCompare(result.comparisons, "final_output",
                native_postconv.final_output, delta_gpu.final_output,
                oracle.final_output);
  if (t.layer == 22) {
    AppendCompare(result.comparisons, "final_output_same_gpu_preconv",
                  gpu_input_native_final_output, delta_gpu.final_output,
                  oracle.final_output);
  }
''',
  )
  cpp = replace_once(
      cpp,
      '''  AppendCompare(result.comparisons, "linear_attn_out",
                native_linear_attn_out, attention_gpu.linear_attn_out,
                oracle.linear_attn_out);
  AppendCompare(result.comparisons, "attn_residual",
''',
      '''  AppendCompare(result.comparisons, "linear_attn_out",
                native_linear_attn_out, attention_gpu.linear_attn_out,
                oracle.linear_attn_out);
  if (t.layer == 22) {
    AppendCompare(result.comparisons, "linear_attn_out_same_gpu_input",
                  gpu_input_native_linear_attn_out, attention_gpu.linear_attn_out,
                  oracle.linear_attn_out);
    AppendCompare(result.comparisons, "attn_residual_same_gpu_input",
                  gpu_input_native_attn_residual, attention_gpu.attn_residual,
                  oracle.attn_residual);
    AppendCompare(result.comparisons, "attn_post_norm_same_gpu_input",
                  gpu_input_native_attn_post_norm, attention_gpu.attn_post_norm,
                  oracle.attn_post_norm);
  }
  AppendCompare(result.comparisons, "attn_residual",
''',
  )
  cpp = replace_once(
      cpp,
      '''    const auto layer18_attention_gpu = RunGpuAttentionFront(
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
      '''    const auto layer18_attention_gpu = RunGpuAttentionFront(
        args.model_path,
        *iq36::find_tensor(index, LayerTensorName(layer18, "attn_output.weight")),
        layer18_core_gate_gpu.attn_gated,
        layer17_run.gpu_layer_output,
        layer18_ffn_norm_weight,
        rms_norm_epsilon,
        args.device_substring,
        args.repeat);
''' + ORACLE_INPUT_BLOCK + '''
    std::vector<NamedCompareGroup> strict_groups;
''',
  )
  cpp = replace_once(
      cpp,
      '''    WriteNamedCompareGroups(layer18_full_attention_groups);
    std::cout << ",\\"l2_k_raw\\":{\\"gpu_vs_cpu\\":";
''',
      '''    WriteNamedCompareGroups(layer18_full_attention_groups);
    std::cout << ",";
    WriteNamedCompareGroups(layer18_oracle_input_strict_groups);
    std::cout << ",\\"l23_oracle_input_k_raw\\":{\\"gpu_vs_cpu\\":";
    WriteCompare(layer18_oracle_input_k_raw_gpu_vs_cpu);
    std::cout << "},";
    WriteNamedCompareGroups(layer18_oracle_input_full_groups);
    std::cout << ",\\"l2_k_raw\\":{\\"gpu_vs_cpu\\":";
''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << "\\"layer23_arc_device_selected\\":"
              << (layer18_arc_selected ? "true" : "false") << ",";
    std::cout << "\\"history_kv_state_payloads_present\\":"
''',
      '''    std::cout << "\\"layer23_arc_device_selected\\":"
              << (layer18_arc_selected ? "true" : "false") << ",";
    std::cout << "\\"layer23_oracle_input_strict_matches_oracle\\":"
              << (layer18_oracle_input_strict_ok ? "true" : "false") << ",";
    std::cout << "\\"layer23_oracle_input_full_attention_matches_oracle\\":"
              << (layer18_oracle_input_full_ok ? "true" : "false") << ",";
    std::cout << "\\"layer23_oracle_input_diagnostic_matches_oracle\\":"
              << (layer18_oracle_input_ok ? "true" : "false") << ",";
    std::cout << "\\"history_kv_state_payloads_present\\":"
''',
  )
  cpp = apply_layer23_ffn_lout_patch(cpp)
  cpp = apply_l22_z_source_patch(cpp)
  return cpp


def parse_args() -> Any:
  global INCLUDE_FFN_LOUT
  z_parser = argparse.ArgumentParser(add_help=False)
  z_parser.add_argument("--z-source", choices=Z_SOURCE_CHOICES, default=DEFAULT_Z_SOURCE)
  z_parser.add_argument("--include-ffn-lout", action="store_true")
  z_args, remaining = z_parser.parse_known_args()
  INCLUDE_FFN_LOUT = z_args.include_ffn_lout
  original_argv = sys.argv
  try:
    sys.argv = [original_argv[0], *remaining]
    args = BASE_PARSE_ARGS()
  finally:
    sys.argv = original_argv
  configure_z_source(z_args.z_source)
  args.z_source = z_args.z_source
  args.include_ffn_lout = INCLUDE_FFN_LOUT
  if args.out_dir is None:
    stamp = BASE.utc_stamp()
    slug = Z_SOURCE_OUTPUT_SLUGS[z_args.z_source]
    if INCLUDE_FFN_LOUT:
      slug += "-ffn-lout"
    args.out_dir = (
        BASE.ROOT
        / f"output/{slug}-{stamp}"
    )
  return args


def write_summary(path: Path, payload: dict[str, Any]) -> None:
  BASE_WRITE_SUMMARY(path, payload)
  if INCLUDE_FFN_LOUT:
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        "add, and post-attn RMSNorm. FFN/l_out remains the next gate. This is\n"
        "captured single-token handoff evidence, not decode throughput.",
        "add, post-attn RMSNorm, FFN, residual add, and layer output. This is\n"
        "captured single-token handoff evidence, not decode throughput.",
    )
    path.write_text(text, encoding="utf-8")
  probe = payload.get("probe") if isinstance(payload.get("probe"), dict) else {}
  timings = probe.get("timings", {}) if isinstance(probe, dict) else {}
  comparisons = probe.get("comparisons", {}) if isinstance(probe, dict) else {}
  lines = [
      "",
      "## Oracle-Input Diagnostic",
      "",
      f"- strict diagnostic passed: `{BASE.PRECONV.nested_bool(probe, 'checks', 'layer23_oracle_input_strict_matches_oracle')}`",
      f"- full-attention diagnostic passed: `{BASE.PRECONV.nested_bool(probe, 'checks', 'layer23_oracle_input_full_attention_matches_oracle')}`",
      f"- combined diagnostic passed: `{BASE.PRECONV.nested_bool(probe, 'checks', 'layer23_oracle_input_diagnostic_matches_oracle')}`",
      "",
      "| output | comparison | max abs | RMSE |",
      "|---|---|---:|---:|",
  ]
  for name in (
      "l23_oracle_input_attn_norm",
      "l23_oracle_input_q_full",
      "l23_oracle_input_q_rope",
      "l23_oracle_input_k_rope",
      "l23_oracle_input_v",
      "l23_oracle_input_attn_pregate",
      "l23_oracle_input_attn_output",
      "l23_oracle_input_attn_post_norm",
  ):
    group = comparisons.get(name, {}) if isinstance(comparisons, dict) else {}
    lane = group.get("gpu_vs_oracle", {}) if isinstance(group, dict) else {}
    lines.append(
        f"| {name} | gpu_vs_oracle | {lane.get('max_abs_diff')} | {lane.get('rmse')} |"
      )
  if INCLUDE_FFN_LOUT:
    lines += [
        "",
        "## Layer-23 FFN/l_out Diagnostic",
        "",
        f"- FFN/l_out diagnostic passed: `{BASE.PRECONV.nested_bool(probe, 'checks', 'layer23_ffn_lout_handoff_matches')}`",
        f"- FFN boundary: `{probe.get('layer23_ffn_boundary')}`",
        f"- l_out boundary: `{probe.get('layer23_lout_boundary')}`",
        f"- selected down type: `{probe.get('layer23_selected_down_tensor_type')}`",
        f"- shared down type: `{probe.get('layer23_shared_down_tensor_type')}`",
        "",
        "| output | comparison | max abs | RMSE |",
        "|---|---|---:|---:|",
    ]
    for name in (
        "l23_selected_down",
        "l23_shared_down",
        "l23_ffn_out",
        "l23_layer_output",
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
        f"| layer23_ffn_lout | {timings.get('resident_layer23_ffn_lout_kernel_sum_min_us') if isinstance(timings, dict) else None} |",
        f"| through_layer23_lout | {timings.get('resident_layer5_6_7_8_9_10_11_12_13_14_15_16_17_18_19_20_21_22_to_layer23_lout_kernel_sum_min_us') if isinstance(timings, dict) else None} |",
    ]
  if Z_SOURCE != DEFAULT_Z_SOURCE:
    z_flag = {
        "native": "layer22_native_z_correction_diagnostic",
        "gpu-f32": "layer22_gpu_f32_z_correction_diagnostic",
        "gpu-q4-cpu-order": "layer22_gpu_q4_cpu_order_z_correction_diagnostic",
    }[Z_SOURCE]
    z_label = {
        "native": "Native",
        "gpu-f32": "GPU F32",
        "gpu-q4-cpu-order": "GPU Q4 CPU-Order",
    }[Z_SOURCE]
    z_compare = {
        "native": "l22_z_native_correction_input",
        "gpu-f32": "l22_z_gpu_f32_correction_input",
        "gpu-q4-cpu-order": "l22_z_gpu_q4_cpu_order_correction_input",
    }[Z_SOURCE]
    lines += [
        "",
        f"## Layer-22 {z_label} Z Correction Diagnostic",
        "",
        f"- z source: `{Z_SOURCE}`",
        f"- correction flag: `{BASE.PRECONV.nested_bool(probe, 'checks', z_flag)}`",
        f"- non-bypassed layer23 full-attn passed: `{BASE.PRECONV.nested_bool(probe, 'checks', 'layer23_full_attn_core_output_matches_oracle')}`",
        "",
        "| output | comparison | max abs | RMSE |",
        "|---|---|---:|---:|",
    ]
    for name in (
        "l22_z",
        z_compare,
        "l22_final_output",
        "l22_linear_attn_out",
        "l22_attn_post_norm",
        "l22_layer_output",
        "l23_residual_input",
        "l23_attn_norm",
        "l23_attn_output",
        "l23_attn_post_norm",
    ):
      group = comparisons.get(name, {}) if isinstance(comparisons, dict) else {}
      lane = group.get("gpu_vs_oracle", {}) if isinstance(group, dict) else {}
      lines.append(
          f"| {name} | gpu_vs_oracle | {lane.get('max_abs_diff')} | {lane.get('rmse')} |"
      )
  path.write_text(path.read_text(encoding="utf-8") + "\n".join(lines) + "\n",
                  encoding="utf-8")


def main() -> int:
  configure_z_source(DEFAULT_Z_SOURCE)
  BASE.layer23_full_attn_probe_cpp = diagnostic_cpp
  BASE.parse_args = parse_args
  BASE.write_summary = write_summary
  BASE.L11_FULL.resolve_prefixed_full_attention_payloads = resolve_prefixed_full_attention_payloads
  return BASE.main()


if __name__ == "__main__":
  raise SystemExit(main())
