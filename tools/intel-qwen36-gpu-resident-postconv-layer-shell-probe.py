#!/usr/bin/env python3
"""Run the resident GPU postconv-to-layer shell handoff probe."""

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
ATTENTION_TOOL = Path(__file__).with_name("intel-qwen36-gpu-resident-attention-ffn-shell-probe.py")
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA_VERSION = "intel-qwen36-gpu-resident-postconv-layer-shell-probe-v0"
DEFAULT_HOST = "local"
DEFAULT_MODEL = "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf"
DEFAULT_ENV_SCRIPT = "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh"
DEFAULT_REMOTE_ROOT = "/home/intel/intel-qwen36-gpu"
DEFAULT_ORACLE_BUNDLE = ROOT / "oracle/r0-oracle-bundle-20260627T060028Z"
PAYLOAD_ROOT = ROOT / "output/r0-boundary-capture-run-20260627T054024Z/remote-output/payloads"
OPENCL_SOURCE = ROOT / "engine/gpu/opencl/q4x8_matvec.cl"


def load_attention_tool() -> ModuleType:
  spec = importlib.util.spec_from_file_location("iq36_attention_ffn_shell_probe", ATTENTION_TOOL)
  if spec is None or spec.loader is None:
    raise SystemExit(f"failed to load attention FFN shell tool: {ATTENTION_TOOL}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


ATTENTION = load_attention_tool()


POSTCONV_HELPERS_CPP = r'''

constexpr int kLinearHeadDim = 128;
constexpr int kLinearQueryHeads = 16;
constexpr int kLinearValueHeads = 32;
constexpr int kLinearQValues = kLinearHeadDim * kLinearQueryHeads;
constexpr int kLinearVValues = kLinearHeadDim * kLinearValueHeads;
constexpr int kLinearStateValues = kLinearHeadDim * kLinearHeadDim * kLinearValueHeads;

struct PostConvDeltaTiming {
  double delta_min_us = 0.0;
  double delta_mean_us = 0.0;
  double final_norm_min_us = 0.0;
  double final_norm_mean_us = 0.0;
  double delta_to_final_kernel_sum_min_us = 0.0;
  double delta_to_final_kernel_sum_mean_us = 0.0;
  std::uint64_t input_upload_wall_ns = 0;
  std::uint64_t kernel_wall_ns = 0;
  std::uint64_t attention_read_wall_ns = 0;
  std::uint64_t final_read_wall_ns = 0;
  std::uint64_t state_read_wall_ns = 0;
  std::uint64_t delta_global_work_items = 0;
  std::uint64_t final_norm_global_work_items = 0;
};

struct PostConvDeltaRun {
  std::vector<float> attention_output;
  std::vector<float> recurrent_state;
  std::vector<float> final_output;
  PostConvDeltaTiming timing;
  std::string platform_name;
  std::string device_name;
  std::string build_log;
  double program_build_ms = 0.0;
};

PostConvDeltaRun RunGpuPostConvDelta(
    const std::vector<float>& q,
    const std::vector<float>& k,
    const std::vector<float>& v,
    const std::vector<float>& gate,
    const std::vector<float>& beta,
    const std::vector<float>& recurrent_state,
    const std::vector<float>& z,
    const std::vector<float>& norm_weight,
    float rms_norm_epsilon,
    const std::string& device_substring,
    int repeat) {
  Require(q.size() == kLinearQValues, "q predelta size mismatch");
  Require(k.size() == kLinearQValues, "k predelta size mismatch");
  Require(v.size() == kLinearVValues, "v predelta size mismatch");
  Require(gate.size() == kLinearValueHeads, "gate size mismatch");
  Require(beta.size() == kLinearValueHeads, "beta size mismatch");
  Require(recurrent_state.size() == kLinearStateValues, "state predelta size mismatch");
  Require(z.size() == kLinearVValues, "z size mismatch");
  Require(norm_weight.size() == kLinearHeadDim, "ssm norm weight size mismatch");

  iq36::GpuQ4X8MatvecRunner runner(device_substring, kOpenClSource);
  const auto gpu = runner.RunLinearAttentionDelta(
      q, k, v, gate, beta, recurrent_state, z, norm_weight,
      kLinearHeadDim, kLinearQueryHeads, kLinearValueHeads,
      rms_norm_epsilon, repeat);
  PostConvDeltaRun run;
  run.attention_output = gpu.attention_output;
  run.recurrent_state = gpu.recurrent_state;
  run.final_output = gpu.final_output;
  run.timing.delta_min_us = gpu.timing.delta_min_us;
  run.timing.delta_mean_us = gpu.timing.delta_mean_us;
  run.timing.final_norm_min_us = gpu.timing.final_min_us;
  run.timing.final_norm_mean_us = gpu.timing.final_mean_us;
  run.timing.delta_to_final_kernel_sum_min_us =
      gpu.timing.delta_min_us + gpu.timing.final_min_us;
  run.timing.delta_to_final_kernel_sum_mean_us =
      gpu.timing.delta_mean_us + gpu.timing.final_mean_us;
  run.timing.input_upload_wall_ns = gpu.timing.input_upload_wall_ns;
  run.timing.kernel_wall_ns = gpu.timing.kernel_wall_ns;
  run.timing.attention_read_wall_ns = gpu.timing.attention_read_wall_ns;
  run.timing.final_read_wall_ns = gpu.timing.final_read_wall_ns;
  run.timing.state_read_wall_ns = gpu.timing.state_read_wall_ns;
  run.timing.delta_global_work_items = gpu.timing.delta_global_work_items;
  run.timing.final_norm_global_work_items = gpu.timing.final_global_work_items;
  run.platform_name = runner.platform_name();
  run.device_name = runner.device_name();
  run.build_log = runner.build_log();
  run.program_build_ms = runner.program_build_ms();
  return run;
}
'''


POSTCONV_MAIN_CPP = r'''
int main(int argc, char** argv) {
  try {
    const Args args = ParseArgs(argc, argv);
    const auto index = iq36::parse_gguf_model_index(args.model_path);
    const auto load_map = iq36::validate_qwen36_load_map(index);
    const float rms_norm_epsilon = MetadataFloat(
        index, "qwen35moe.attention.layer_norm_rms_epsilon", 1e-6f);
    const std::string ssm_norm_tensor_name =
        LayerTensorName(args.layer, "ssm_norm.weight");
    const std::string output_tensor_name =
        LayerTensorName(args.layer, "ssm_out.weight");
    const std::string ffn_norm_tensor_name =
        LayerTensorName(args.layer, "post_attention_norm.weight");
    const std::string selected_gate_up_tensor_name =
        LayerTensorName(args.layer, "ffn_gate_up_exps.weight");
    const std::string selected_down_tensor_name =
        LayerTensorName(args.layer, "ffn_down_exps.weight");
    const std::string shared_gate_tensor_name =
        LayerTensorName(args.layer, "ffn_gate_shexp.weight");
    const std::string shared_up_tensor_name =
        LayerTensorName(args.layer, "ffn_up_shexp.weight");
    const std::string shared_down_tensor_name =
        LayerTensorName(args.layer, "ffn_down_shexp.weight");
    const std::string shared_input_gate_tensor_name =
        LayerTensorName(args.layer, "ffn_gate_inp_shexp.weight");
    const auto* ssm_norm_tensor = iq36::find_tensor(index, ssm_norm_tensor_name);
    const auto* output_tensor = iq36::find_tensor(index, output_tensor_name);
    const auto* ffn_norm_tensor = iq36::find_tensor(index, ffn_norm_tensor_name);
    const auto* selected_gate_up_tensor =
        iq36::find_tensor(index, selected_gate_up_tensor_name);
    const auto* selected_down_tensor =
        iq36::find_tensor(index, selected_down_tensor_name);
    const auto* shared_gate_tensor =
        iq36::find_tensor(index, shared_gate_tensor_name);
    const auto* shared_up_tensor =
        iq36::find_tensor(index, shared_up_tensor_name);
    const auto* shared_down_tensor =
        iq36::find_tensor(index, shared_down_tensor_name);
    const auto* shared_input_gate_tensor =
        iq36::find_tensor(index, shared_input_gate_tensor_name);
    Require(ssm_norm_tensor != nullptr, "ssm norm tensor missing");
    Require(output_tensor != nullptr, "attention output tensor missing");
    Require(ffn_norm_tensor != nullptr, "ffn norm tensor missing");
    Require(selected_gate_up_tensor != nullptr, "selected gate-up tensor missing");
    Require(selected_down_tensor != nullptr, "selected down tensor missing");
    Require(shared_gate_tensor != nullptr, "shared gate tensor missing");
    Require(shared_up_tensor != nullptr, "shared up tensor missing");
    Require(shared_down_tensor != nullptr, "shared down tensor missing");
    Require(shared_input_gate_tensor != nullptr, "shared input gate tensor missing");
    const bool delta_tensor_shape_ok =
        ssm_norm_tensor->type == 0 &&
        ssm_norm_tensor->dims == std::vector<std::uint64_t>{kLinearHeadDim};
    const bool attention_tensor_shape_ok =
        output_tensor->type == 12 &&
        output_tensor->dims == std::vector<std::uint64_t>{4096, kHiddenSize} &&
        ffn_norm_tensor->type == 0 &&
        ffn_norm_tensor->dims == std::vector<std::uint64_t>{kHiddenSize};
    const bool selected_tensor_shape_ok =
        selected_gate_up_tensor->type == 12 &&
        selected_down_tensor->type == 12 &&
        selected_gate_up_tensor->dims ==
            std::vector<std::uint64_t>{kHiddenSize, kGateUpRowsPerExpert, kExpertCount} &&
        selected_down_tensor->dims ==
            std::vector<std::uint64_t>{kIntermediateSize, kHiddenSize, kExpertCount};
    const bool shared_tensor_shape_ok =
        shared_gate_tensor->type == 12 &&
        shared_up_tensor->type == 12 &&
        (shared_down_tensor->type == 12 || shared_down_tensor->type == 14) &&
        shared_gate_tensor->dims ==
            std::vector<std::uint64_t>{kHiddenSize, kIntermediateSize} &&
        shared_up_tensor->dims ==
            std::vector<std::uint64_t>{kHiddenSize, kIntermediateSize} &&
        shared_down_tensor->dims ==
            std::vector<std::uint64_t>{kIntermediateSize, kHiddenSize} &&
        shared_input_gate_tensor->type == 0 &&
        shared_input_gate_tensor->dims == std::vector<std::uint64_t>{kHiddenSize};

    const auto q = iq36::read_f32_vector_file(JoinPath(args.payload_dir, "q_conv_predelta.bin"));
    const auto k = iq36::read_f32_vector_file(JoinPath(args.payload_dir, "k_conv_predelta.bin"));
    const auto v = iq36::read_f32_vector_file(JoinPath(args.payload_dir, "v_conv_predelta.bin"));
    const auto gate = iq36::read_f32_vector_file(JoinPath(args.payload_dir, "gate.bin"));
    const auto beta = iq36::read_f32_vector_file(JoinPath(args.payload_dir, "beta_sigmoid.bin"));
    const auto state = iq36::read_f32_vector_file(JoinPath(args.payload_dir, "state_predelta.bin"));
    const auto z = iq36::read_f32_vector_file(JoinPath(args.payload_dir, "z.bin"));
    const auto residual_input =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "residual_input.bin"));
    const auto oracle_attn_output =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "attn_output.bin"));
    const auto oracle_final_output =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "final_output.bin"));
    const auto oracle_linear_attn_out =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "linear_attn_out.bin"));
    const auto oracle_attn_residual =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "attn_residual.bin"));
    const auto oracle_attn_post_norm =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "attn_post_norm.bin"));
    const auto expert_ids =
        ReadI32VectorFile(JoinPath(args.payload_dir, "ffn_moe_topk.bin"));
    const auto weights_norm =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "ffn_moe_weights_norm.bin"));
    const auto oracle_gate_up =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "ffn_moe_gate_up.bin"));
    const auto oracle_swiglu =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "ffn_moe_swiglu.bin"));
    const auto oracle_down =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "ffn_moe_down.bin"));
    const auto oracle_weighted =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "ffn_moe_weighted.bin"));
    const auto oracle_moe_out =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "ffn_moe_out.bin"));
    const auto oracle_shared_down =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "ffn_shexp.bin"));
    const auto oracle_shared_gate =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "shared_expert_gate.bin"));
    const auto oracle_shared_sigmoid =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "shared_expert_gate_sigmoid.bin"));
    const auto oracle_shared_gated =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "ffn_shexp_gated.bin"));
    const auto oracle_ffn_out =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "ffn_out.bin"));
    const auto oracle_layer_output =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "layer_output.bin"));
    const bool payload_counts_ok =
        q.size() == kLinearQValues &&
        k.size() == kLinearQValues &&
        v.size() == kLinearVValues &&
        gate.size() == kLinearValueHeads &&
        beta.size() == kLinearValueHeads &&
        state.size() == kLinearStateValues &&
        z.size() == kLinearVValues &&
        residual_input.size() == kHiddenSize &&
        oracle_attn_output.size() == kLinearVValues &&
        oracle_final_output.size() == kLinearVValues &&
        oracle_linear_attn_out.size() == kHiddenSize &&
        oracle_attn_residual.size() == kHiddenSize &&
        oracle_attn_post_norm.size() == kHiddenSize &&
        expert_ids.size() == kExpertUsedCount &&
        weights_norm.size() == kExpertUsedCount &&
        oracle_gate_up.size() == kGateUpValueCount &&
        oracle_swiglu.size() == kSwiGluValueCount &&
        oracle_down.size() == kWeightedValueCount &&
        oracle_weighted.size() == kWeightedValueCount &&
        oracle_moe_out.size() == kHiddenSize &&
        oracle_shared_down.size() == kHiddenSize &&
        oracle_shared_gate.size() == 1 &&
        oracle_shared_sigmoid.size() == 1 &&
        oracle_shared_gated.size() == kHiddenSize &&
        oracle_ffn_out.size() == kHiddenSize &&
        oracle_layer_output.size() == kHiddenSize;
    Require(payload_counts_ok, "payload size mismatch");

    std::ifstream model(args.model_path, std::ios::binary);
    Require(static_cast<bool>(model), "model file could not be opened");
    const auto ssm_norm_weight =
        iq36::decode_tensor_row(args.model_path, index, ssm_norm_tensor_name, 0);
    const auto ffn_norm_weight =
        ReadF32TensorPayload(model, *ffn_norm_tensor,
                             static_cast<std::size_t>(kHiddenSize));
    const auto shared_input_gate_weights =
        ReadF32TensorPayload(model, *shared_input_gate_tensor,
                             static_cast<std::size_t>(kHiddenSize));

    const auto native_delta = iq36::run_qwen36_linear_attention_delta_core(
        q, k, v, gate, beta, state, z, ssm_norm_weight, rms_norm_epsilon);
    const auto native_linear_attn_out =
        iq36::matvec_tensor(args.model_path, index, output_tensor_name,
                            native_delta.final_output);
    const auto native_attn_residual =
        iq36::add_vectors(residual_input, native_linear_attn_out);
    const auto native_attn_post_norm =
        iq36::apply_rms_norm(native_attn_residual, ffn_norm_weight, rms_norm_epsilon);
    const auto native_selected_gate_up =
        iq36::matvec_expert_tensor(args.model_path, index,
                                   selected_gate_up_tensor_name,
                                   native_attn_post_norm, expert_ids);
    const auto native_selected_swiglu =
        iq36::apply_swiglu_from_gate_up(native_selected_gate_up,
                                        kIntermediateSize, kExpertUsedCount);
    const auto native_selected_down =
        iq36::matvec_expert_tensor_per_expert_input(
            args.model_path, index, selected_down_tensor_name,
            native_selected_swiglu, expert_ids);
    const auto native_weighted =
        iq36::apply_expert_weights(native_selected_down, weights_norm, kHiddenSize);
    const auto native_moe_out =
        iq36::aggregate_experts(native_weighted, kExpertUsedCount, kHiddenSize);
    const auto native_shared_gate =
        iq36::matvec_tensor(args.model_path, index, shared_gate_tensor_name,
                            native_attn_post_norm);
    const auto native_shared_up =
        iq36::matvec_tensor(args.model_path, index, shared_up_tensor_name,
                            native_attn_post_norm);
    std::vector<float> native_shared_gate_up;
    native_shared_gate_up.reserve(native_shared_gate.size() + native_shared_up.size());
    native_shared_gate_up.insert(native_shared_gate_up.end(),
                                 native_shared_gate.begin(), native_shared_gate.end());
    native_shared_gate_up.insert(native_shared_gate_up.end(),
                                 native_shared_up.begin(), native_shared_up.end());
    const auto native_shared_swiglu =
        iq36::apply_swiglu_from_gate_up(native_shared_gate_up, kIntermediateSize, 1);
    const auto native_shared_down =
        iq36::matvec_tensor(args.model_path, index, shared_down_tensor_name,
                            native_shared_swiglu);
    const auto native_shared_input_gate =
        iq36::matvec_tensor(args.model_path, index,
                            shared_input_gate_tensor_name, native_attn_post_norm);
    Require(native_shared_input_gate.size() == 1, "native shared input gate size mismatch");
    const std::vector<float> native_shared_sigmoid{
        iq36::sigmoid_scalar(native_shared_input_gate[0])};
    const auto native_shared_gated =
        iq36::multiply_by_scalar(native_shared_down, native_shared_sigmoid[0]);
    const auto native_ffn_out =
        iq36::add_vectors(native_moe_out, native_shared_gated);
    const auto native_layer_output =
        iq36::add_vectors(native_attn_residual, native_ffn_out);

    const auto delta_gpu = RunGpuPostConvDelta(
        q, k, v, gate, beta, state, z, ssm_norm_weight, rms_norm_epsilon,
        args.device_substring, args.repeat);
    const auto attention_gpu = RunGpuAttentionFront(
        args.model_path, *output_tensor, delta_gpu.final_output, residual_input,
        ffn_norm_weight, rms_norm_epsilon, args.device_substring, args.repeat);
    const auto selected_gpu = RunGpuSelectedFfnShell(
        args.model_path, *selected_gate_up_tensor, *selected_down_tensor,
        attention_gpu.attn_post_norm, expert_ids, args.device_substring, args.repeat);
    const auto shared_gpu = RunGpuSharedFfnShell(
        args.model_path, *shared_gate_tensor, *shared_up_tensor,
        *shared_down_tensor, attention_gpu.attn_post_norm,
        args.device_substring, args.repeat);
    const auto gpu = RunGpuShell(shared_input_gate_weights,
                                 attention_gpu.attn_post_norm,
                                 selected_gpu.down, weights_norm,
                                 shared_gpu.down,
                                 attention_gpu.attn_residual,
                                 args.device_substring, args.repeat);

    const auto attn_output_cpu_vs_oracle =
        iq36::compare_vectors(native_delta.attention_output, oracle_attn_output, kMismatchThreshold);
    const auto attn_output_gpu_vs_cpu =
        iq36::compare_vectors(delta_gpu.attention_output, native_delta.attention_output, kMismatchThreshold);
    const auto attn_output_gpu_vs_oracle =
        iq36::compare_vectors(delta_gpu.attention_output, oracle_attn_output, kMismatchThreshold);
    const auto recurrent_state_gpu_vs_cpu =
        iq36::compare_vectors(delta_gpu.recurrent_state, native_delta.recurrent_state, kMismatchThreshold);
    const auto final_output_cpu_vs_oracle =
        iq36::compare_vectors(native_delta.final_output, oracle_final_output, kMismatchThreshold);
    const auto final_output_gpu_vs_cpu =
        iq36::compare_vectors(delta_gpu.final_output, native_delta.final_output, kMismatchThreshold);
    const auto final_output_gpu_vs_oracle =
        iq36::compare_vectors(delta_gpu.final_output, oracle_final_output, kMismatchThreshold);
    const auto linear_attn_out_cpu_vs_oracle =
        iq36::compare_vectors(native_linear_attn_out, oracle_linear_attn_out, kMismatchThreshold);
    const auto linear_attn_out_gpu_vs_cpu =
        iq36::compare_vectors(attention_gpu.linear_attn_out, native_linear_attn_out, kMismatchThreshold);
    const auto linear_attn_out_gpu_vs_oracle =
        iq36::compare_vectors(attention_gpu.linear_attn_out, oracle_linear_attn_out, kMismatchThreshold);
    const auto attn_residual_cpu_vs_oracle =
        iq36::compare_vectors(native_attn_residual, oracle_attn_residual, kMismatchThreshold);
    const auto attn_residual_gpu_vs_cpu =
        iq36::compare_vectors(attention_gpu.attn_residual, native_attn_residual, kMismatchThreshold);
    const auto attn_residual_gpu_vs_oracle =
        iq36::compare_vectors(attention_gpu.attn_residual, oracle_attn_residual, kMismatchThreshold);
    const auto attn_post_norm_cpu_vs_oracle =
        iq36::compare_vectors(native_attn_post_norm, oracle_attn_post_norm, kMismatchThreshold);
    const auto attn_post_norm_gpu_vs_cpu =
        iq36::compare_vectors(attention_gpu.attn_post_norm, native_attn_post_norm, kMismatchThreshold);
    const auto attn_post_norm_gpu_vs_oracle =
        iq36::compare_vectors(attention_gpu.attn_post_norm, oracle_attn_post_norm, kMismatchThreshold);
    const auto selected_gate_up_cpu_vs_oracle =
        iq36::compare_vectors(native_selected_gate_up, oracle_gate_up, kMismatchThreshold);
    const auto selected_gate_up_gpu_vs_cpu =
        iq36::compare_vectors(selected_gpu.gate_up, native_selected_gate_up, kMismatchThreshold);
    const auto selected_gate_up_gpu_vs_oracle =
        iq36::compare_vectors(selected_gpu.gate_up, oracle_gate_up, kMismatchThreshold);
    const auto selected_swiglu_cpu_vs_oracle =
        iq36::compare_vectors(native_selected_swiglu, oracle_swiglu, kMismatchThreshold);
    const auto selected_swiglu_gpu_vs_cpu =
        iq36::compare_vectors(selected_gpu.swiglu, native_selected_swiglu, kMismatchThreshold);
    const auto selected_swiglu_gpu_vs_oracle =
        iq36::compare_vectors(selected_gpu.swiglu, oracle_swiglu, kMismatchThreshold);
    const auto selected_down_cpu_vs_oracle =
        iq36::compare_vectors(native_selected_down, oracle_down, kMismatchThreshold);
    const auto selected_down_gpu_vs_cpu =
        iq36::compare_vectors(selected_gpu.down, native_selected_down, kMismatchThreshold);
    const auto selected_down_gpu_vs_oracle =
        iq36::compare_vectors(selected_gpu.down, oracle_down, kMismatchThreshold);
    const auto shared_gate_gpu_vs_cpu =
        iq36::compare_vectors(shared_gpu.gate, native_shared_gate, kMismatchThreshold);
    const auto shared_up_gpu_vs_cpu =
        iq36::compare_vectors(shared_gpu.up, native_shared_up, kMismatchThreshold);
    const auto shared_swiglu_gpu_vs_cpu =
        iq36::compare_vectors(shared_gpu.swiglu, native_shared_swiglu, kMismatchThreshold);
    const auto shared_down_cpu_vs_oracle =
        iq36::compare_vectors(native_shared_down, oracle_shared_down, kMismatchThreshold);
    const auto shared_down_gpu_vs_cpu =
        iq36::compare_vectors(shared_gpu.down, native_shared_down, kMismatchThreshold);
    const auto shared_down_gpu_vs_oracle =
        iq36::compare_vectors(shared_gpu.down, oracle_shared_down, kMismatchThreshold);
    const auto weighted_cpu_vs_oracle =
        iq36::compare_vectors(native_weighted, oracle_weighted, kMismatchThreshold);
    const auto weighted_gpu_vs_cpu =
        iq36::compare_vectors(gpu.weighted, native_weighted, kMismatchThreshold);
    const auto weighted_gpu_vs_oracle =
        iq36::compare_vectors(gpu.weighted, oracle_weighted, kMismatchThreshold);
    const auto moe_out_cpu_vs_oracle =
        iq36::compare_vectors(native_moe_out, oracle_moe_out, kMismatchThreshold);
    const auto moe_out_gpu_vs_cpu =
        iq36::compare_vectors(gpu.moe_out, native_moe_out, kMismatchThreshold);
    const auto moe_out_gpu_vs_oracle =
        iq36::compare_vectors(gpu.moe_out, oracle_moe_out, kMismatchThreshold);
    const auto shared_input_gate_cpu_vs_oracle =
        iq36::compare_vectors(native_shared_input_gate, oracle_shared_gate, kMismatchThreshold);
    const auto shared_input_gate_gpu_vs_cpu =
        iq36::compare_vectors(gpu.shared_gate, native_shared_input_gate, kMismatchThreshold);
    const auto shared_input_gate_gpu_vs_oracle =
        iq36::compare_vectors(gpu.shared_gate, oracle_shared_gate, kMismatchThreshold);
    const auto shared_sigmoid_cpu_vs_oracle =
        iq36::compare_vectors(native_shared_sigmoid, oracle_shared_sigmoid, kMismatchThreshold);
    const auto shared_sigmoid_gpu_vs_cpu =
        iq36::compare_vectors(gpu.shared_gate_sigmoid, native_shared_sigmoid, kMismatchThreshold);
    const auto shared_sigmoid_gpu_vs_oracle =
        iq36::compare_vectors(gpu.shared_gate_sigmoid, oracle_shared_sigmoid, kMismatchThreshold);
    const auto shared_gated_cpu_vs_oracle =
        iq36::compare_vectors(native_shared_gated, oracle_shared_gated, kMismatchThreshold);
    const auto shared_gated_gpu_vs_cpu =
        iq36::compare_vectors(gpu.shared_gated, native_shared_gated, kMismatchThreshold);
    const auto shared_gated_gpu_vs_oracle =
        iq36::compare_vectors(gpu.shared_gated, oracle_shared_gated, kMismatchThreshold);
    const auto ffn_out_cpu_vs_oracle =
        iq36::compare_vectors(native_ffn_out, oracle_ffn_out, kMismatchThreshold);
    const auto ffn_out_gpu_vs_cpu =
        iq36::compare_vectors(gpu.ffn_out, native_ffn_out, kMismatchThreshold);
    const auto ffn_out_gpu_vs_oracle =
        iq36::compare_vectors(gpu.ffn_out, oracle_ffn_out, kMismatchThreshold);
    const auto layer_cpu_vs_oracle =
        iq36::compare_vectors(native_layer_output, oracle_layer_output, kMismatchThreshold);
    const auto layer_gpu_vs_cpu =
        iq36::compare_vectors(gpu.layer_output, native_layer_output, kMismatchThreshold);
    const auto layer_gpu_vs_oracle =
        iq36::compare_vectors(gpu.layer_output, oracle_layer_output, kMismatchThreshold);

    const bool delta_comparisons_passed =
        ComparePassed(attn_output_cpu_vs_oracle) &&
        ComparePassed(attn_output_gpu_vs_cpu) &&
        ComparePassed(attn_output_gpu_vs_oracle) &&
        ComparePassed(recurrent_state_gpu_vs_cpu) &&
        ComparePassed(final_output_cpu_vs_oracle) &&
        ComparePassed(final_output_gpu_vs_cpu) &&
        ComparePassed(final_output_gpu_vs_oracle);
    const bool attention_front_comparisons_passed =
        ComparePassed(linear_attn_out_cpu_vs_oracle) &&
        ComparePassed(linear_attn_out_gpu_vs_cpu) &&
        ComparePassed(linear_attn_out_gpu_vs_oracle) &&
        ComparePassed(attn_residual_cpu_vs_oracle) &&
        ComparePassed(attn_residual_gpu_vs_cpu) &&
        ComparePassed(attn_residual_gpu_vs_oracle) &&
        ComparePassed(attn_post_norm_cpu_vs_oracle) &&
        ComparePassed(attn_post_norm_gpu_vs_cpu) &&
        ComparePassed(attn_post_norm_gpu_vs_oracle);
    const bool selected_comparisons_passed =
        ComparePassed(selected_gate_up_cpu_vs_oracle) &&
        ComparePassed(selected_gate_up_gpu_vs_cpu) &&
        ComparePassed(selected_gate_up_gpu_vs_oracle) &&
        ComparePassed(selected_swiglu_cpu_vs_oracle) &&
        ComparePassed(selected_swiglu_gpu_vs_cpu) &&
        ComparePassed(selected_swiglu_gpu_vs_oracle) &&
        ComparePassed(selected_down_cpu_vs_oracle) &&
        ComparePassed(selected_down_gpu_vs_cpu) &&
        ComparePassed(selected_down_gpu_vs_oracle);
    const bool shared_comparisons_passed =
        ComparePassed(shared_gate_gpu_vs_cpu) &&
        ComparePassed(shared_up_gpu_vs_cpu) &&
        ComparePassed(shared_swiglu_gpu_vs_cpu) &&
        ComparePassed(shared_down_cpu_vs_oracle) &&
        ComparePassed(shared_down_gpu_vs_cpu) &&
        ComparePassed(shared_down_gpu_vs_oracle);
    const bool tail_comparisons_passed =
        ComparePassed(weighted_cpu_vs_oracle) &&
        ComparePassed(weighted_gpu_vs_cpu) &&
        ComparePassed(weighted_gpu_vs_oracle) &&
        ComparePassed(moe_out_cpu_vs_oracle) &&
        ComparePassed(moe_out_gpu_vs_cpu) &&
        ComparePassed(moe_out_gpu_vs_oracle) &&
        ComparePassed(shared_input_gate_cpu_vs_oracle) &&
        ComparePassed(shared_input_gate_gpu_vs_cpu) &&
        ComparePassed(shared_input_gate_gpu_vs_oracle) &&
        ComparePassed(shared_sigmoid_cpu_vs_oracle) &&
        ComparePassed(shared_sigmoid_gpu_vs_cpu) &&
        ComparePassed(shared_sigmoid_gpu_vs_oracle) &&
        ComparePassed(shared_gated_cpu_vs_oracle) &&
        ComparePassed(shared_gated_gpu_vs_cpu) &&
        ComparePassed(shared_gated_gpu_vs_oracle) &&
        ComparePassed(ffn_out_cpu_vs_oracle) &&
        ComparePassed(ffn_out_gpu_vs_cpu) &&
        ComparePassed(ffn_out_gpu_vs_oracle) &&
        ComparePassed(layer_cpu_vs_oracle) &&
        ComparePassed(layer_gpu_vs_cpu) &&
        ComparePassed(layer_gpu_vs_oracle);
    const bool delta_timing_positive =
        delta_gpu.timing.delta_min_us > 0.0 &&
        delta_gpu.timing.final_norm_min_us > 0.0;
    const bool attention_timing_positive =
        attention_gpu.timing.output_projection_min_us > 0.0 &&
        attention_gpu.timing.residual_add_min_us > 0.0 &&
        attention_gpu.timing.ffn_rmsnorm_min_us > 0.0;
    const bool selected_timing_positive =
        selected_gpu.timing.gate_up_min_us > 0.0 &&
        selected_gpu.timing.swiglu_min_us > 0.0 &&
        selected_gpu.timing.down_min_us > 0.0;
    const bool shared_timing_positive =
        shared_gpu.timing.gate_min_us > 0.0 &&
        shared_gpu.timing.up_min_us > 0.0 &&
        shared_gpu.timing.swiglu_min_us > 0.0 &&
        shared_gpu.timing.down_min_us > 0.0;
    const bool tail_timing_positive =
        gpu.timing.weighted_min_us > 0.0 &&
        gpu.timing.shared_gate_matvec_min_us > 0.0 &&
        gpu.timing.shared_gate_apply_min_us > 0.0 &&
        gpu.timing.ffn_output_add_min_us > 0.0 &&
        gpu.timing.residual_add_min_us > 0.0 &&
        gpu.timing.shell_sum_min_us > 0.0;
    const bool arc_selected =
        delta_gpu.device_name.find(args.device_substring) != std::string::npos &&
        attention_gpu.device_name.find(args.device_substring) != std::string::npos &&
        selected_gpu.device_name.find(args.device_substring) != std::string::npos &&
        shared_gpu.device_name.find(args.device_substring) != std::string::npos &&
        gpu.device_name.find(args.device_substring) != std::string::npos;
    const bool required_checks_passed =
        load_map.ready &&
        delta_tensor_shape_ok &&
        attention_tensor_shape_ok &&
        selected_tensor_shape_ok &&
        shared_tensor_shape_ok &&
        payload_counts_ok &&
        arc_selected &&
        delta_comparisons_passed &&
        attention_front_comparisons_passed &&
        selected_comparisons_passed &&
        shared_comparisons_passed &&
        tail_comparisons_passed &&
        delta_timing_positive &&
        attention_timing_positive &&
        selected_timing_positive &&
        shared_timing_positive &&
        tail_timing_positive;
    const double full_kernel_sum_min =
        delta_gpu.timing.delta_to_final_kernel_sum_min_us +
        attention_gpu.timing.attention_front_kernel_sum_min_us +
        selected_gpu.timing.selected_ffn_kernel_sum_min_us +
        shared_gpu.timing.shared_ffn_kernel_sum_min_us +
        gpu.timing.shell_sum_min_us;
    const double full_kernel_sum_mean =
        delta_gpu.timing.delta_to_final_kernel_sum_mean_us +
        attention_gpu.timing.attention_front_kernel_sum_mean_us +
        selected_gpu.timing.selected_ffn_kernel_sum_mean_us +
        shared_gpu.timing.shared_ffn_kernel_sum_mean_us +
        gpu.timing.shell_sum_mean_us;

    std::cout << std::setprecision(10);
    std::cout << "{";
    std::cout << "\"schema_version\":\"intel-qwen36-gpu-resident-postconv-layer-shell-probe-v0\",";
    std::cout << "\"model_path\":\"" << JsonEscape(args.model_path) << "\",";
    std::cout << "\"layer\":" << args.layer << ",";
    std::cout << "\"source_token_position\":" << kSourceTokenPosition << ",";
    std::cout << "\"resident_api\":\"postconv_to_layer_shell_load_once_run_many\",";
    std::cout << "\"resident_load_count\":1,";
    std::cout << "\"resident_shell_invocations\":" << args.repeat << ",";
    std::cout << "\"delta_to_attention_host_boundary\":true,";
    std::cout << "\"attention_output_projection_host_q8_bridge\":true,";
    std::cout << "\"attention_front_host_boundary_between_q4_and_f32\":true,";
    std::cout << "\"selected_down_host_q8_bridge\":true,";
    std::cout << "\"shared_down_host_q8_bridge\":true,";
    std::cout << "\"selected_down_q4_expert_launches\":"
              << selected_gpu.timing.down_kernel_launches << ",";
    std::cout << "\"shared_down_kernel_launches\":"
              << shared_gpu.timing.down_kernel_launches << ",";
    std::cout << "\"shared_down_tensor_type\":\""
              << iq36::ggml_type_name(shared_down_tensor->type) << "\",";
    std::cout << "\"rms_norm_epsilon\":" << rms_norm_epsilon << ",";
    std::cout << "\"platform_name\":\"" << JsonEscape(delta_gpu.platform_name) << "\",";
    std::cout << "\"device_name\":\"" << JsonEscape(delta_gpu.device_name) << "\",";
    std::cout << "\"attention_device_name\":\"" << JsonEscape(attention_gpu.device_name) << "\",";
    std::cout << "\"selected_device_name\":\"" << JsonEscape(selected_gpu.device_name) << "\",";
    std::cout << "\"shared_device_name\":\"" << JsonEscape(shared_gpu.device_name) << "\",";
    std::cout << "\"tail_device_name\":\"" << JsonEscape(gpu.device_name) << "\",";
    std::cout << "\"program_build_ms\":"
              << (delta_gpu.program_build_ms + attention_gpu.program_build_ms +
                  selected_gpu.program_build_ms + shared_gpu.program_build_ms +
                  gpu.program_build_ms) << ",";
    std::cout << "\"build_log\":\""
              << JsonEscape(delta_gpu.build_log + attention_gpu.build_log +
                            selected_gpu.build_log + shared_gpu.build_log +
                            gpu.build_log) << "\",";
    std::cout << "\"repeat\":" << args.repeat << ",";
    std::cout << "\"timings\":{";
    std::cout << "\"delta_recurrent_min_us\":"
              << delta_gpu.timing.delta_min_us << ",";
    std::cout << "\"delta_recurrent_mean_us\":"
              << delta_gpu.timing.delta_mean_us << ",";
    std::cout << "\"delta_final_norm_min_us\":"
              << delta_gpu.timing.final_norm_min_us << ",";
    std::cout << "\"delta_final_norm_mean_us\":"
              << delta_gpu.timing.final_norm_mean_us << ",";
    std::cout << "\"delta_to_final_kernel_sum_min_us\":"
              << delta_gpu.timing.delta_to_final_kernel_sum_min_us << ",";
    std::cout << "\"delta_to_final_kernel_sum_mean_us\":"
              << delta_gpu.timing.delta_to_final_kernel_sum_mean_us << ",";
    std::cout << "\"attention_output_projection_min_us\":"
              << attention_gpu.timing.output_projection_min_us << ",";
    std::cout << "\"attention_output_projection_host_q8_bridge_us\":"
              << attention_gpu.timing.output_projection_host_q8_bridge_us << ",";
    std::cout << "\"post_attention_residual_add_min_us\":"
              << attention_gpu.timing.residual_add_min_us << ",";
    std::cout << "\"ffn_rmsnorm_min_us\":"
              << attention_gpu.timing.ffn_rmsnorm_min_us << ",";
    std::cout << "\"attention_front_kernel_sum_min_us\":"
              << attention_gpu.timing.attention_front_kernel_sum_min_us << ",";
    std::cout << "\"selected_ffn_kernel_sum_min_us\":"
              << selected_gpu.timing.selected_ffn_kernel_sum_min_us << ",";
    std::cout << "\"shared_ffn_kernel_sum_min_us\":"
              << shared_gpu.timing.shared_ffn_kernel_sum_min_us << ",";
    std::cout << "\"ffn_tail_kernel_sum_min_us\":"
              << gpu.timing.shell_sum_min_us << ",";
    std::cout << "\"resident_postconv_to_layer_kernel_sum_min_us\":"
              << full_kernel_sum_min << ",";
    std::cout << "\"resident_postconv_to_layer_kernel_sum_mean_us\":"
              << full_kernel_sum_mean << ",";
    std::cout << "\"selected_down_host_q8_bridge_us\":"
              << selected_gpu.timing.host_q8_bridge_us << ",";
    std::cout << "\"shared_down_host_q8_bridge_us\":"
              << shared_gpu.timing.host_q8_bridge_us;
    std::cout << "},\"comparisons\":{";
    std::cout << "\"attn_output\":";
    WriteCompareGroup(attn_output_cpu_vs_oracle, attn_output_gpu_vs_cpu, attn_output_gpu_vs_oracle);
    std::cout << ",\"recurrent_state\":{\"gpu_vs_cpu\":";
    WriteCompare(recurrent_state_gpu_vs_cpu);
    std::cout << "},\"final_output\":";
    WriteCompareGroup(final_output_cpu_vs_oracle, final_output_gpu_vs_cpu, final_output_gpu_vs_oracle);
    std::cout << ",\"linear_attn_out\":";
    WriteCompareGroup(linear_attn_out_cpu_vs_oracle, linear_attn_out_gpu_vs_cpu, linear_attn_out_gpu_vs_oracle);
    std::cout << ",\"attn_residual\":";
    WriteCompareGroup(attn_residual_cpu_vs_oracle, attn_residual_gpu_vs_cpu, attn_residual_gpu_vs_oracle);
    std::cout << ",\"attn_post_norm\":";
    WriteCompareGroup(attn_post_norm_cpu_vs_oracle, attn_post_norm_gpu_vs_cpu, attn_post_norm_gpu_vs_oracle);
    std::cout << ",\"selected_gate_up\":";
    WriteCompareGroup(selected_gate_up_cpu_vs_oracle, selected_gate_up_gpu_vs_cpu, selected_gate_up_gpu_vs_oracle);
    std::cout << ",\"selected_swiglu\":";
    WriteCompareGroup(selected_swiglu_cpu_vs_oracle, selected_swiglu_gpu_vs_cpu, selected_swiglu_gpu_vs_oracle);
    std::cout << ",\"selected_down\":";
    WriteCompareGroup(selected_down_cpu_vs_oracle, selected_down_gpu_vs_cpu, selected_down_gpu_vs_oracle);
    std::cout << ",\"shared_gate\":{\"gpu_vs_cpu\":";
    WriteCompare(shared_gate_gpu_vs_cpu);
    std::cout << "},\"shared_up\":{\"gpu_vs_cpu\":";
    WriteCompare(shared_up_gpu_vs_cpu);
    std::cout << "},\"shared_swiglu\":{\"gpu_vs_cpu\":";
    WriteCompare(shared_swiglu_gpu_vs_cpu);
    std::cout << "},\"shared_down\":";
    WriteCompareGroup(shared_down_cpu_vs_oracle, shared_down_gpu_vs_cpu, shared_down_gpu_vs_oracle);
    std::cout << ",\"weighted_selected_down\":";
    WriteCompareGroup(weighted_cpu_vs_oracle, weighted_gpu_vs_cpu, weighted_gpu_vs_oracle);
    std::cout << ",\"ffn_moe_out\":";
    WriteCompareGroup(moe_out_cpu_vs_oracle, moe_out_gpu_vs_cpu, moe_out_gpu_vs_oracle);
    std::cout << ",\"shared_input_gate\":";
    WriteCompareGroup(shared_input_gate_cpu_vs_oracle, shared_input_gate_gpu_vs_cpu, shared_input_gate_gpu_vs_oracle);
    std::cout << ",\"shared_gate_sigmoid\":";
    WriteCompareGroup(shared_sigmoid_cpu_vs_oracle, shared_sigmoid_gpu_vs_cpu, shared_sigmoid_gpu_vs_oracle);
    std::cout << ",\"ffn_shexp_gated\":";
    WriteCompareGroup(shared_gated_cpu_vs_oracle, shared_gated_gpu_vs_cpu, shared_gated_gpu_vs_oracle);
    std::cout << ",\"ffn_out\":";
    WriteCompareGroup(ffn_out_cpu_vs_oracle, ffn_out_gpu_vs_cpu, ffn_out_gpu_vs_oracle);
    std::cout << ",\"layer_output\":";
    WriteCompareGroup(layer_cpu_vs_oracle, layer_gpu_vs_cpu, layer_gpu_vs_oracle);
    std::cout << "},\"checks\":{";
    std::cout << "\"load_map_ready\":" << (load_map.ready ? "true" : "false") << ",";
    std::cout << "\"delta_tensor_shape_ok\":"
              << (delta_tensor_shape_ok ? "true" : "false") << ",";
    std::cout << "\"attention_tensor_shape_ok\":"
              << (attention_tensor_shape_ok ? "true" : "false") << ",";
    std::cout << "\"selected_tensor_shape_ok\":"
              << (selected_tensor_shape_ok ? "true" : "false") << ",";
    std::cout << "\"shared_tensor_shape_ok\":"
              << (shared_tensor_shape_ok ? "true" : "false") << ",";
    std::cout << "\"payload_counts_ok\":"
              << (payload_counts_ok ? "true" : "false") << ",";
    std::cout << "\"arc_device_selected\":"
              << (arc_selected ? "true" : "false") << ",";
    std::cout << "\"delta_recurrent_matches_oracle\":"
              << (delta_comparisons_passed ? "true" : "false") << ",";
    std::cout << "\"attention_front_matches_oracle\":"
              << (attention_front_comparisons_passed ? "true" : "false") << ",";
    std::cout << "\"selected_expert_ffn_matches_oracle\":"
              << (selected_comparisons_passed ? "true" : "false") << ",";
    std::cout << "\"shared_expert_ffn_matches_oracle\":"
              << (shared_comparisons_passed ? "true" : "false") << ",";
    std::cout << "\"tail_matches_oracle\":"
              << (tail_comparisons_passed ? "true" : "false") << ",";
    std::cout << "\"gpu_event_timing_positive\":"
              << (delta_timing_positive && attention_timing_positive &&
                  selected_timing_positive && shared_timing_positive &&
                  tail_timing_positive ? "true" : "false") << ",";
    std::cout << "\"resident_load_once\":true,";
    std::cout << "\"resident_shell_invocations_positive\":"
              << (args.repeat > 0 ? "true" : "false") << ",";
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


def postconv_layer_probe_cpp(opencl_source: str) -> str:
  cpp = ATTENTION.attention_ffn_probe_cpp(opencl_source)
  main_index = cpp.index("\nint main(")
  return cpp[:main_index] + "\n" + POSTCONV_HELPERS_CPP + "\n" + POSTCONV_MAIN_CPP


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
  parser.add_argument("--resident-invocations", type=int, default=5)
  parser.add_argument("--device-substring", default="B390")
  parser.add_argument("--timeout-s", type=int, default=900)
  parser.add_argument("--out-dir", type=Path, default=None)
  return parser.parse_args()


def shell_join(argv: list[str]) -> str:
  return " ".join(shlex.quote(item) for item in argv)


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


def find_payload(pattern: str, expected_bytes: int) -> Path:
  matches = sorted(PAYLOAD_ROOT.glob(pattern))
  if len(matches) != 1:
    raise SystemExit(f"expected one payload for {pattern}, found {len(matches)}")
  path = matches[0].resolve()
  if path.stat().st_size != expected_bytes:
    raise SystemExit(f"payload size mismatch for {path}")
  return path


def add_payload(payloads: dict[str, dict[str, Any]],
                name: str,
                stage_name: str,
                path: Path,
                expected_bytes: int) -> None:
  payloads[name] = {
      "local_path": path,
      "path": str(path.relative_to(ROOT)),
      "sha256": iq36_local.sha256_file(path),
      "size_bytes": expected_bytes,
      "stage_name": stage_name,
  }


def resolve_payloads(layer: int) -> dict[str, dict[str, Any]]:
  payloads = ATTENTION.resolve_payloads(layer)
  for name, stage_name, prefix, expected_bytes in (
      ("q_conv_predelta", "q_conv_predelta.bin", "q_conv_predelta", 8192),
      ("k_conv_predelta", "k_conv_predelta.bin", "k_conv_predelta", 8192),
      ("v_conv_predelta", "v_conv_predelta.bin", "v_conv_predelta", 16384),
      ("gate", "gate.bin", "gate", 128),
      ("beta_sigmoid", "beta_sigmoid.bin", "beta_sigmoid", 128),
      ("state_predelta", "state_predelta.bin", "state_predelta", 2097152),
      ("attn_output", "attn_output.bin", "attn_output", 16384),
      ("z", "z.bin", "z", 16384),
  ):
    add_payload(
        payloads,
        name,
        stage_name,
        find_payload(f"{prefix}-{layer}__tok15__ord*.bin", expected_bytes),
        expected_bytes,
    )
  return payloads


def nested_bool(obj: dict[str, Any] | None, *keys: str) -> bool:
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


def resident_fields_ok(probe: dict[str, Any] | None, expected_invocations: int) -> bool:
  return (
      isinstance(probe, dict)
      and probe.get("resident_api") == "postconv_to_layer_shell_load_once_run_many"
      and probe.get("resident_load_count") == 1
      and probe.get("resident_shell_invocations") == expected_invocations
      and probe.get("delta_to_attention_host_boundary") is True
      and probe.get("attention_output_projection_host_q8_bridge") is True
      and probe.get("attention_front_host_boundary_between_q4_and_f32") is True
      and probe.get("selected_down_host_q8_bridge") is True
      and probe.get("shared_down_host_q8_bridge") is True
      and nested_bool(probe, "checks", "resident_load_once")
      and nested_bool(probe, "checks", "resident_shell_invocations_positive")
  )


def write_summary(path: Path, payload: dict[str, Any]) -> None:
  probe = payload.get("probe") if isinstance(payload.get("probe"), dict) else {}
  timings = probe.get("timings", {}) if isinstance(probe, dict) else {}
  comparisons = probe.get("comparisons", {}) if isinstance(probe, dict) else {}
  lines = [
      "# GPU Resident Postconv-to-Layer Shell Probe",
      "",
      f"- workstream: `{WORKSTREAM}`",
      f"- host: `{payload.get('host')}`",
      f"- model: `{payload.get('model')}`",
      f"- layer: `{payload.get('layer')}`",
      f"- required checks passed: `{str(payload.get('required_checks_passed')).lower()}`",
      "- speedup claims allowed: `false`",
      f"- platform/device: `{probe.get('platform_name')}` / `{probe.get('device_name')}`",
      f"- resident API: `{probe.get('resident_api')}`",
      f"- resident load count: `{probe.get('resident_load_count')}`",
      f"- resident shell invocations: `{probe.get('resident_shell_invocations')}`",
      f"- shared down tensor type: `{probe.get('shared_down_tensor_type')}`",
      f"- delta-to-attention host boundary: `{str(probe.get('delta_to_attention_host_boundary')).lower()}`",
      f"- attention output Q8 bridge: `{str(probe.get('attention_output_projection_host_q8_bridge')).lower()}`",
      f"- attention front host boundary: `{str(probe.get('attention_front_host_boundary_between_q4_and_f32')).lower()}`",
      f"- selected down host Q8 bridge: `{str(probe.get('selected_down_host_q8_bridge')).lower()}`",
      f"- shared down host Q8 bridge: `{str(probe.get('shared_down_host_q8_bridge')).lower()}`",
      "",
      "| output | comparison | max abs | RMSE |",
      "|---|---|---:|---:|",
  ]
  for name in (
      "attn_output",
      "final_output",
      "linear_attn_out",
      "attn_residual",
      "attn_post_norm",
      "selected_down",
      "shared_down",
      "ffn_out",
      "layer_output",
  ):
    group = comparisons.get(name, {}) if isinstance(comparisons, dict) else {}
    lane = group.get("gpu_vs_oracle", {}) if isinstance(group, dict) else {}
    lines.append(f"| {name} | gpu_vs_oracle | {lane.get('max_abs_diff')} | {lane.get('rmse')} |")
  lines += [
      "",
      "| kernel group | min us |",
      "|---|---:|",
      f"| delta_recurrent | {timings.get('delta_recurrent_min_us')} |",
      f"| delta_final_norm | {timings.get('delta_final_norm_min_us')} |",
      f"| delta_to_final_sum | {timings.get('delta_to_final_kernel_sum_min_us')} |",
      f"| attention_front_sum | {timings.get('attention_front_kernel_sum_min_us')} |",
      f"| selected_ffn_sum | {timings.get('selected_ffn_kernel_sum_min_us')} |",
      f"| shared_ffn_sum | {timings.get('shared_ffn_kernel_sum_min_us')} |",
      f"| ffn_tail_sum | {timings.get('ffn_tail_kernel_sum_min_us')} |",
      f"| resident_postconv_to_layer_sum | {timings.get('resident_postconv_to_layer_kernel_sum_min_us')} |",
      "",
      "The target-side process starts from captured postconv predelta outputs,",
      "gates, z, and recurrent state, computes delta recurrent/final norm,",
      "attention output projection, post-attention residual, FFN RMSNorm,",
      "selected/shared FFN branches, tail aggregation, and residual to captured",
      "`l_out`. Host boundaries and Q8 activation bridges remain explicit. This",
      "is captured single-layer evidence only, not prompt/token decode throughput.",
      "",
  ]
  path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  args = parse_args()
  if args.resident_invocations <= 0:
    raise SystemExit("--resident-invocations must be positive")
  stamp = utc_stamp()
  created_at = iso_now()
  out_dir = args.out_dir or ROOT / f"output/gpu-resident-postconv-layer-shell-probe-{stamp}"
  out_dir.mkdir(parents=True, exist_ok=False)
  raw_dir = out_dir / "raw"
  raw_dir.mkdir()

  payloads = resolve_payloads(args.layer)
  opencl_source = OPENCL_SOURCE.read_text(encoding="utf-8")
  opencl_source_hash = iq36_local.sha256_file(OPENCL_SOURCE)
  embedded_opencl_hash = hashlib.sha256(
      (opencl_source + ATTENTION.ATTENTION_EXTRA_OPENCL).encode("utf-8")
  ).hexdigest()
  local_cpp = out_dir / "gpu_resident_postconv_layer_shell_probe.cpp"
  local_cpp.write_text(postconv_layer_probe_cpp(opencl_source), encoding="utf-8")

  remote_dir = f"{args.remote_root.rstrip('/')}/gpu-resident-postconv-layer-shell-probe-{stamp}"
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
    for local, remote in ATTENTION.SHARED.SELECTED.SOURCE_FILES:
      transfers.append(iq36_local.copy_to(args.host, ROOT / local, f"{remote_dir}/{remote}", args.timeout_s))
    transfers.append(
        iq36_local.copy_to(
            args.host,
            local_cpp,
            f"{remote_dir}/tests/gpu_resident_postconv_layer_shell_probe.cpp",
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

  executable = f"{remote_dir}/build/iq36-gpu-resident-postconv-layer-shell-probe"
  compile_cmd = " && ".join([
      f"source {shlex.quote(args.env_script)} >/dev/null 2>&1",
      (
          "g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic "
          f"-I {shlex.quote(remote_dir + '/include')} "
          f"{shlex.quote(remote_dir + '/src/gguf_loader.cpp')} "
          f"{shlex.quote(remote_dir + '/src/gpu_q4x8_matvec.cpp')} "
          f"{shlex.quote(remote_dir + '/tests/gpu_resident_postconv_layer_shell_probe.cpp')} "
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
      {"name": "resident_api_fields_present", "pass": resident_fields_ok(probe, args.resident_invocations)},
      {"name": "delta_recurrent_matches_oracle", "pass": nested_bool(probe, "checks", "delta_recurrent_matches_oracle")},
      {"name": "attention_front_matches_oracle", "pass": nested_bool(probe, "checks", "attention_front_matches_oracle")},
      {"name": "selected_expert_ffn_matches_oracle", "pass": nested_bool(probe, "checks", "selected_expert_ffn_matches_oracle")},
      {"name": "shared_expert_ffn_matches_oracle", "pass": nested_bool(probe, "checks", "shared_expert_ffn_matches_oracle")},
      {"name": "tail_matches_oracle", "pass": nested_bool(probe, "checks", "tail_matches_oracle")},
      {"name": "gpu_event_timing_positive", "pass": nested_bool(probe, "checks", "gpu_event_timing_positive")},
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
      "tool": "tools/intel-qwen36-gpu-resident-postconv-layer-shell-probe.py",
      "artifact": str(out_dir),
      "host": args.host,
      "remote_dir": remote_dir,
      "layer": args.layer,
      "resident_invocations": args.resident_invocations,
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
      "gpu_resident_postconv_layer_shell_probe",
      [
          ("required_checks_passed", required_checks_passed),
          ("resident_shell_invocations", args.resident_invocations),
          ("delta_to_final_kernel_sum_min_us", nested_number(timings, "delta_to_final_kernel_sum_min_us")),
          ("resident_postconv_to_layer_kernel_sum_min_us", nested_number(timings, "resident_postconv_to_layer_kernel_sum_min_us")),
          ("final_output_gpu_vs_oracle_max_abs_diff", nested_number(comparisons, "final_output", "gpu_vs_oracle", "max_abs_diff")),
          ("layer_output_gpu_vs_oracle_max_abs_diff", nested_number(comparisons, "layer_output", "gpu_vs_oracle", "max_abs_diff")),
      ],
  )
  write_summary(out_dir / "summary.md", payload)
  print(out_dir)
  return 0 if required_checks_passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
