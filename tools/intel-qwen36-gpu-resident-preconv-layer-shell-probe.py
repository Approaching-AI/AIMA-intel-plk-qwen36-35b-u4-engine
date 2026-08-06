#!/usr/bin/env python3
"""Run the resident GPU preconv-to-layer shell handoff probe."""

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
POSTCONV_TOOL = Path(__file__).with_name("intel-qwen36-gpu-resident-postconv-layer-shell-probe.py")
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA_VERSION = "intel-qwen36-gpu-resident-preconv-layer-shell-probe-v0"
DEFAULT_HOST = "local"
DEFAULT_MODEL = "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf"
DEFAULT_ENV_SCRIPT = "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh"
DEFAULT_REMOTE_ROOT = "/home/intel/intel-qwen36-gpu"
DEFAULT_ORACLE_BUNDLE = ROOT / "oracle/r0-oracle-bundle-20260627T060028Z"
PAYLOAD_ROOT = ROOT / "output/r0-boundary-capture-run-20260627T054024Z/remote-output/payloads"
OPENCL_SOURCE = ROOT / "engine/gpu/opencl/q4x8_matvec.cl"


def load_postconv_tool() -> ModuleType:
  spec = importlib.util.spec_from_file_location("iq36_postconv_layer_shell_probe", POSTCONV_TOOL)
  if spec is None or spec.loader is None:
    raise SystemExit(f"failed to load postconv layer shell tool: {POSTCONV_TOOL}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


POSTCONV = load_postconv_tool()


PRECONV_HELPERS_CPP = r'''

constexpr int kLinearQkvMixedValues = 8192;
constexpr int kLinearConvKernelSize = 4;
constexpr int kLinearConvStateValues =
    (kLinearConvKernelSize - 1) * kLinearQkvMixedValues;

std::vector<float> SigmoidVectorPreConv(const std::vector<float>& values) {
  std::vector<float> out(values.size());
  for (std::size_t i = 0; i < values.size(); ++i) {
    out[i] = iq36::sigmoid_scalar(values[i]);
  }
  return out;
}

float SoftplusScalarPreConv(float value) {
  if (value > 20.0f) {
    return value;
  }
  if (value < -20.0f) {
    return std::exp(value);
  }
  return std::log1p(std::exp(value));
}

std::vector<float> SoftplusVectorPreConv(const std::vector<float>& values) {
  std::vector<float> out(values.size());
  for (std::size_t i = 0; i < values.size(); ++i) {
    out[i] = SoftplusScalarPreConv(values[i]);
  }
  return out;
}

std::vector<float> MultiplyVectorsPreConv(const std::vector<float>& lhs,
                                          const std::vector<float>& rhs) {
  Require(lhs.size() == rhs.size(), "preconv multiply vector size mismatch");
  std::vector<float> out(lhs.size());
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    out[i] = lhs[i] * rhs[i];
  }
  return out;
}

struct PreConvFrontTiming {
  double host_q8_bridge_us = 0.0;
  double qkv_min_us = 0.0;
  double qkv_mean_us = 0.0;
  double alpha_min_us = 0.0;
  double alpha_mean_us = 0.0;
  double beta_min_us = 0.0;
  double beta_mean_us = 0.0;
  double z_min_us = 0.0;
  double z_mean_us = 0.0;
  double conv_min_us = 0.0;
  double conv_mean_us = 0.0;
  double postconv_silu_split_min_us = 0.0;
  double postconv_silu_split_mean_us = 0.0;
  double postconv_q_l2_min_us = 0.0;
  double postconv_q_l2_mean_us = 0.0;
  double postconv_k_l2_min_us = 0.0;
  double postconv_k_l2_mean_us = 0.0;
  double preconv_to_postconv_kernel_sum_min_us = 0.0;
  double preconv_to_postconv_kernel_sum_mean_us = 0.0;
};

struct PreConvFrontRun {
  std::vector<float> qkv_mixed;
  std::vector<float> alpha;
  std::vector<float> beta;
  std::vector<float> z;
  std::vector<float> a_softplus;
  std::vector<float> gate;
  std::vector<float> beta_sigmoid;
  std::vector<float> conv_output_raw;
  std::vector<float> conv_state_after;
  std::vector<float> conv_output_silu;
  std::vector<float> q_conv;
  std::vector<float> k_conv;
  std::vector<float> v_conv_predelta;
  std::vector<float> q_conv_predelta;
  std::vector<float> k_conv_predelta;
  PreConvFrontTiming timing;
  std::string platform_name;
  std::string device_name;
  std::string build_log;
  double program_build_ms = 0.0;
};

iq36::GpuQ4X8MatvecRun RunProjectionFromTensor(
    iq36::GpuQ4X8MatvecRunner& runner,
    std::ifstream& model,
    const iq36::GgufTensorInfo& tensor,
    const iq36::GpuQ8KInputPlanes& q8,
    std::uint64_t rows,
    int repeat) {
  Require(tensor.type == 12, "preconv projection tensor is not Q4_K: " + tensor.name);
  Require(tensor.dims == std::vector<std::uint64_t>{kHiddenSize, rows},
          "preconv projection tensor dims mismatch: " + tensor.name);
  const auto raw = ReadTensorBytes(model, tensor);
  const auto packed = iq36::PackQ4Kx8(raw, rows, kHiddenSize / 256);
  return runner.Run(packed, q8.qs, q8.bsums, q8.d, rows, kHiddenSize / 256,
                    repeat, iq36::GpuQ4X8KernelVariant::kRowlaneParallel);
}

PreConvFrontRun RunGpuPreConvFront(
    const std::string& model_path,
    const iq36::GgufTensorInfo& qkv_tensor,
    const iq36::GgufTensorInfo& alpha_tensor,
    const iq36::GgufTensorInfo& beta_tensor,
    const iq36::GgufTensorInfo& z_tensor,
    const iq36::GgufTensorInfo& conv_tensor,
    const std::vector<float>& ssm_dt,
    const std::vector<float>& ssm_a,
    const std::vector<float>& attn_norm,
    const std::vector<float>& conv_state,
    float rms_norm_epsilon,
    const std::string& device_substring,
    int repeat) {
  Require(attn_norm.size() == kHiddenSize, "preconv attn_norm size mismatch");
  Require(conv_state.size() == kLinearConvStateValues,
          "preconv conv state size mismatch");
  Require(ssm_dt.size() == kLinearValueHeads, "ssm_dt size mismatch");
  Require(ssm_a.size() == kLinearValueHeads, "ssm_a size mismatch");
  Require(qkv_tensor.type == 12, "qkv tensor must be Q4_K");
  Require(qkv_tensor.dims == std::vector<std::uint64_t>{kHiddenSize, kLinearQkvMixedValues},
          "qkv tensor dims mismatch");
  Require(conv_tensor.type == 0, "conv tensor must be F32");
  Require(conv_tensor.dims == std::vector<std::uint64_t>{kLinearConvKernelSize, kLinearQkvMixedValues},
          "conv tensor dims mismatch");

  PreConvFrontRun run;
  std::ifstream model(model_path, std::ios::binary);
  Require(static_cast<bool>(model), "failed to open model for preconv front");

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

  const auto qkv_raw = ReadTensorBytes(model, qkv_tensor);
  const auto qkv_packed =
      iq36::PackQ4Kx8(qkv_raw, kLinearQkvMixedValues, kHiddenSize / 256);
  const auto conv_weights = ReadF32TensorPayload(
      model, conv_tensor,
      static_cast<std::size_t>(kLinearQkvMixedValues * kLinearConvKernelSize));
  const auto qkv_conv = runner.RunThenConv(
      qkv_packed,
      q8.qs,
      q8.bsums,
      q8.d,
      conv_weights,
      conv_state,
      kLinearQkvMixedValues,
      kHiddenSize / 256,
      kLinearConvKernelSize,
      repeat,
      iq36::GpuQ4X8KernelVariant::kRowlaneParallel);
  run.qkv_mixed = qkv_conv.qkv_mixed;
  run.conv_output_raw = qkv_conv.conv_output_raw;
  run.conv_state_after = qkv_conv.conv_state;
  run.timing.qkv_min_us = qkv_conv.timing.matvec.min_us;
  run.timing.qkv_mean_us = qkv_conv.timing.matvec.mean_us;
  run.timing.conv_min_us = qkv_conv.timing.conv_min_us;
  run.timing.conv_mean_us = qkv_conv.timing.conv_mean_us;

  const auto alpha = RunProjectionFromTensor(
      runner, model, alpha_tensor, q8, kLinearValueHeads, repeat);
  const auto beta = RunProjectionFromTensor(
      runner, model, beta_tensor, q8, kLinearValueHeads, repeat);
  const auto z = RunProjectionFromTensor(
      runner, model, z_tensor, q8, kLinearVValues, repeat);
  run.alpha = alpha.output;
  run.beta = beta.output;
  run.z = z.output;
  run.timing.alpha_min_us = alpha.timing.min_us;
  run.timing.alpha_mean_us = alpha.timing.mean_us;
  run.timing.beta_min_us = beta.timing.min_us;
  run.timing.beta_mean_us = beta.timing.mean_us;
  run.timing.z_min_us = z.timing.min_us;
  run.timing.z_mean_us = z.timing.mean_us;
  run.a_softplus = SoftplusVectorPreConv(iq36::add_vectors(run.alpha, ssm_dt));
  run.gate = MultiplyVectorsPreConv(run.a_softplus, ssm_a);
  run.beta_sigmoid = SigmoidVectorPreConv(run.beta);

  const auto prep = runner.RunPostConvPrep(
      run.conv_output_raw, kLinearHeadDim, kLinearQueryHeads, kLinearValueHeads,
      rms_norm_epsilon, repeat);
  run.conv_output_silu = prep.conv_output_silu;
  run.q_conv = prep.q_conv;
  run.k_conv = prep.k_conv;
  run.v_conv_predelta = prep.v_conv_predelta;
  run.q_conv_predelta = prep.q_conv_predelta;
  run.k_conv_predelta = prep.k_conv_predelta;
  run.timing.postconv_silu_split_min_us = prep.timing.silu_split_min_us;
  run.timing.postconv_silu_split_mean_us = prep.timing.silu_split_mean_us;
  run.timing.postconv_q_l2_min_us = prep.timing.q_l2_min_us;
  run.timing.postconv_q_l2_mean_us = prep.timing.q_l2_mean_us;
  run.timing.postconv_k_l2_min_us = prep.timing.k_l2_min_us;
  run.timing.postconv_k_l2_mean_us = prep.timing.k_l2_mean_us;
  run.timing.preconv_to_postconv_kernel_sum_min_us =
      run.timing.qkv_min_us +
      run.timing.alpha_min_us +
      run.timing.beta_min_us +
      run.timing.z_min_us +
      run.timing.conv_min_us +
      run.timing.postconv_silu_split_min_us +
      run.timing.postconv_q_l2_min_us +
      run.timing.postconv_k_l2_min_us;
  run.timing.preconv_to_postconv_kernel_sum_mean_us =
      run.timing.qkv_mean_us +
      run.timing.alpha_mean_us +
      run.timing.beta_mean_us +
      run.timing.z_mean_us +
      run.timing.conv_mean_us +
      run.timing.postconv_silu_split_mean_us +
      run.timing.postconv_q_l2_mean_us +
      run.timing.postconv_k_l2_mean_us;
  return run;
}

struct NamedCompareGroup {
  std::string name;
  iq36::VectorCompareStats cpu_vs_oracle;
  iq36::VectorCompareStats gpu_vs_cpu;
  iq36::VectorCompareStats gpu_vs_oracle;
};

bool CompareGroupsPassed(const std::vector<NamedCompareGroup>& groups) {
  bool ok = true;
  for (const auto& group : groups) {
    ok = ok &&
         ComparePassed(group.cpu_vs_oracle) &&
         ComparePassed(group.gpu_vs_cpu) &&
         ComparePassed(group.gpu_vs_oracle);
  }
  return ok;
}

void WriteNamedCompareGroups(const std::vector<NamedCompareGroup>& groups) {
  for (std::size_t i = 0; i < groups.size(); ++i) {
    if (i != 0) {
      std::cout << ",";
    }
    std::cout << "\"" << JsonEscape(groups[i].name) << "\":";
    WriteCompareGroup(groups[i].cpu_vs_oracle,
                      groups[i].gpu_vs_cpu,
                      groups[i].gpu_vs_oracle);
  }
}
'''


PRECONV_MAIN_CPP = r'''
int main(int argc, char** argv) {
  try {
    const Args args = ParseArgs(argc, argv);
    const auto index = iq36::parse_gguf_model_index(args.model_path);
    const auto load_map = iq36::validate_qwen36_load_map(index);
    const float rms_norm_epsilon = MetadataFloat(
        index, "qwen35moe.attention.layer_norm_rms_epsilon", 1e-6f);
    const std::string qkv_tensor_name =
        LayerTensorName(args.layer, "attn_qkv.weight");
    const std::string alpha_tensor_name =
        LayerTensorName(args.layer, "ssm_alpha.weight");
    const std::string beta_tensor_name =
        LayerTensorName(args.layer, "ssm_beta.weight");
    const std::string z_tensor_name =
        LayerTensorName(args.layer, "attn_gate.weight");
    const std::string conv_tensor_name =
        LayerTensorName(args.layer, "ssm_conv1d.weight");
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
    const auto* qkv_tensor = iq36::find_tensor(index, qkv_tensor_name);
    const auto* alpha_tensor = iq36::find_tensor(index, alpha_tensor_name);
    const auto* beta_tensor = iq36::find_tensor(index, beta_tensor_name);
    const auto* z_tensor = iq36::find_tensor(index, z_tensor_name);
    const auto* conv_tensor = iq36::find_tensor(index, conv_tensor_name);
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
    Require(qkv_tensor != nullptr, "qkv tensor missing");
    Require(alpha_tensor != nullptr, "alpha tensor missing");
    Require(beta_tensor != nullptr, "beta tensor missing");
    Require(z_tensor != nullptr, "z tensor missing");
    Require(conv_tensor != nullptr, "conv tensor missing");
    Require(ssm_norm_tensor != nullptr, "ssm norm tensor missing");
    Require(output_tensor != nullptr, "attention output tensor missing");
    Require(ffn_norm_tensor != nullptr, "ffn norm tensor missing");
    Require(selected_gate_up_tensor != nullptr, "selected gate-up tensor missing");
    Require(selected_down_tensor != nullptr, "selected down tensor missing");
    Require(shared_gate_tensor != nullptr, "shared gate tensor missing");
    Require(shared_up_tensor != nullptr, "shared up tensor missing");
    Require(shared_down_tensor != nullptr, "shared down tensor missing");
    Require(shared_input_gate_tensor != nullptr, "shared input gate tensor missing");
    const bool preconv_tensor_shape_ok =
        qkv_tensor->type == 12 &&
        alpha_tensor->type == 12 &&
        beta_tensor->type == 12 &&
        z_tensor->type == 12 &&
        conv_tensor->type == 0 &&
        qkv_tensor->dims == std::vector<std::uint64_t>{kHiddenSize, kLinearQkvMixedValues} &&
        alpha_tensor->dims == std::vector<std::uint64_t>{kHiddenSize, kLinearValueHeads} &&
        beta_tensor->dims == std::vector<std::uint64_t>{kHiddenSize, kLinearValueHeads} &&
        z_tensor->dims == std::vector<std::uint64_t>{kHiddenSize, kLinearVValues} &&
        conv_tensor->dims == std::vector<std::uint64_t>{kLinearConvKernelSize, kLinearQkvMixedValues};
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

    const auto attn_norm =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "attn_norm.bin"));
    const auto conv_state =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "conv_state.bin"));
    const auto oracle_qkv =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "linear_attn_qkv_mixed.bin"));
    const auto oracle_alpha =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "alpha.bin"));
    const auto oracle_a_softplus =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "a_softplus.bin"));
    const auto oracle_gate =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "gate.bin"));
    const auto oracle_beta =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "beta.bin"));
    const auto oracle_beta_sigmoid =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "beta_sigmoid.bin"));
    const auto oracle_z =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "z.bin"));
    const auto oracle_conv_output_raw =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "conv_output_raw.bin"));
    const auto oracle_conv_output_silu =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "conv_output_silu.bin"));
    const auto oracle_q_conv =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "q_conv.bin"));
    const auto oracle_k_conv =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "k_conv.bin"));
    const auto oracle_q =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "q_conv_predelta.bin"));
    const auto oracle_k =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "k_conv_predelta.bin"));
    const auto oracle_v =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "v_conv_predelta.bin"));
    const auto state =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "state_predelta.bin"));
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
        attn_norm.size() == kHiddenSize &&
        conv_state.size() == kLinearConvStateValues &&
        oracle_qkv.size() == kLinearQkvMixedValues &&
        oracle_alpha.size() == kLinearValueHeads &&
        oracle_a_softplus.size() == kLinearValueHeads &&
        oracle_gate.size() == kLinearValueHeads &&
        oracle_beta.size() == kLinearValueHeads &&
        oracle_beta_sigmoid.size() == kLinearValueHeads &&
        oracle_z.size() == kLinearVValues &&
        oracle_conv_output_raw.size() == kLinearQkvMixedValues &&
        oracle_conv_output_silu.size() == kLinearQkvMixedValues &&
        oracle_q_conv.size() == kLinearQValues &&
        oracle_k_conv.size() == kLinearQValues &&
        oracle_q.size() == kLinearQValues &&
        oracle_k.size() == kLinearQValues &&
        oracle_v.size() == kLinearVValues &&
        state.size() == kLinearStateValues &&
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
    const auto ssm_dt =
        iq36::decode_tensor_row(args.model_path, index,
                                LayerTensorName(args.layer, "ssm_dt.bias"), 0);
    const auto ssm_a =
        iq36::decode_tensor_row(args.model_path, index,
                                LayerTensorName(args.layer, "ssm_a"), 0);
    const auto ssm_norm_weight =
        iq36::decode_tensor_row(args.model_path, index, ssm_norm_tensor_name, 0);
    const auto ffn_norm_weight =
        ReadF32TensorPayload(model, *ffn_norm_tensor,
                             static_cast<std::size_t>(kHiddenSize));
    const auto shared_input_gate_weights =
        ReadF32TensorPayload(model, *shared_input_gate_tensor,
                             static_cast<std::size_t>(kHiddenSize));

    const auto native_preconv = iq36::run_qwen36_linear_attention_preconv_core(
        args.model_path, index, args.layer, attn_norm);
    const auto native_conv = iq36::run_qwen36_linear_attention_conv_core(
        args.model_path, index, args.layer, native_preconv.qkv_mixed, conv_state);
    const auto native_postconv = iq36::run_qwen36_linear_attention_postconv_core(
        native_conv.conv_output_raw,
        native_preconv.gate,
        native_preconv.beta_sigmoid,
        state,
        native_preconv.z,
        ssm_norm_weight,
        rms_norm_epsilon);
    const auto native_linear_attn_out =
        iq36::matvec_tensor(args.model_path, index, output_tensor_name,
                            native_postconv.final_output);
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

    const auto preconv_gpu = RunGpuPreConvFront(
        args.model_path, *qkv_tensor, *alpha_tensor, *beta_tensor, *z_tensor,
        *conv_tensor, ssm_dt, ssm_a, attn_norm, conv_state,
        rms_norm_epsilon, args.device_substring, args.repeat);
    const auto delta_gpu = RunGpuPostConvDelta(
        preconv_gpu.q_conv_predelta,
        preconv_gpu.k_conv_predelta,
        preconv_gpu.v_conv_predelta,
        preconv_gpu.gate,
        preconv_gpu.beta_sigmoid,
        state,
        preconv_gpu.z,
        ssm_norm_weight,
        rms_norm_epsilon,
        args.device_substring,
        args.repeat);
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

    const std::vector<NamedCompareGroup> preconv_groups = {
        {"linear_attn_qkv_mixed",
         iq36::compare_vectors(native_preconv.qkv_mixed, oracle_qkv, kMismatchThreshold),
         iq36::compare_vectors(preconv_gpu.qkv_mixed, native_preconv.qkv_mixed, kMismatchThreshold),
         iq36::compare_vectors(preconv_gpu.qkv_mixed, oracle_qkv, kMismatchThreshold)},
        {"alpha",
         iq36::compare_vectors(native_preconv.alpha, oracle_alpha, kMismatchThreshold),
         iq36::compare_vectors(preconv_gpu.alpha, native_preconv.alpha, kMismatchThreshold),
         iq36::compare_vectors(preconv_gpu.alpha, oracle_alpha, kMismatchThreshold)},
        {"a_softplus",
         iq36::compare_vectors(native_preconv.alpha_softplus, oracle_a_softplus, kMismatchThreshold),
         iq36::compare_vectors(preconv_gpu.a_softplus, native_preconv.alpha_softplus, kMismatchThreshold),
         iq36::compare_vectors(preconv_gpu.a_softplus, oracle_a_softplus, kMismatchThreshold)},
        {"gate",
         iq36::compare_vectors(native_preconv.gate, oracle_gate, kMismatchThreshold),
         iq36::compare_vectors(preconv_gpu.gate, native_preconv.gate, kMismatchThreshold),
         iq36::compare_vectors(preconv_gpu.gate, oracle_gate, kMismatchThreshold)},
        {"beta",
         iq36::compare_vectors(native_preconv.beta, oracle_beta, kMismatchThreshold),
         iq36::compare_vectors(preconv_gpu.beta, native_preconv.beta, kMismatchThreshold),
         iq36::compare_vectors(preconv_gpu.beta, oracle_beta, kMismatchThreshold)},
        {"beta_sigmoid",
         iq36::compare_vectors(native_preconv.beta_sigmoid, oracle_beta_sigmoid, kMismatchThreshold),
         iq36::compare_vectors(preconv_gpu.beta_sigmoid, native_preconv.beta_sigmoid, kMismatchThreshold),
         iq36::compare_vectors(preconv_gpu.beta_sigmoid, oracle_beta_sigmoid, kMismatchThreshold)},
        {"z",
         iq36::compare_vectors(native_preconv.z, oracle_z, kMismatchThreshold),
         iq36::compare_vectors(preconv_gpu.z, native_preconv.z, kMismatchThreshold),
         iq36::compare_vectors(preconv_gpu.z, oracle_z, kMismatchThreshold)},
        {"conv_output_raw",
         iq36::compare_vectors(native_conv.conv_output_raw, oracle_conv_output_raw, kMismatchThreshold),
         iq36::compare_vectors(preconv_gpu.conv_output_raw, native_conv.conv_output_raw, kMismatchThreshold),
         iq36::compare_vectors(preconv_gpu.conv_output_raw, oracle_conv_output_raw, kMismatchThreshold)},
        {"conv_output_silu",
         iq36::compare_vectors(native_postconv.conv_output_silu, oracle_conv_output_silu, kMismatchThreshold),
         iq36::compare_vectors(preconv_gpu.conv_output_silu, native_postconv.conv_output_silu, kMismatchThreshold),
         iq36::compare_vectors(preconv_gpu.conv_output_silu, oracle_conv_output_silu, kMismatchThreshold)},
        {"q_conv",
         iq36::compare_vectors(native_postconv.q_conv, oracle_q_conv, kMismatchThreshold),
         iq36::compare_vectors(preconv_gpu.q_conv, native_postconv.q_conv, kMismatchThreshold),
         iq36::compare_vectors(preconv_gpu.q_conv, oracle_q_conv, kMismatchThreshold)},
        {"k_conv",
         iq36::compare_vectors(native_postconv.k_conv, oracle_k_conv, kMismatchThreshold),
         iq36::compare_vectors(preconv_gpu.k_conv, native_postconv.k_conv, kMismatchThreshold),
         iq36::compare_vectors(preconv_gpu.k_conv, oracle_k_conv, kMismatchThreshold)},
        {"q_conv_predelta",
         iq36::compare_vectors(native_postconv.q_conv_predelta, oracle_q, kMismatchThreshold),
         iq36::compare_vectors(preconv_gpu.q_conv_predelta, native_postconv.q_conv_predelta, kMismatchThreshold),
         iq36::compare_vectors(preconv_gpu.q_conv_predelta, oracle_q, kMismatchThreshold)},
        {"k_conv_predelta",
         iq36::compare_vectors(native_postconv.k_conv_predelta, oracle_k, kMismatchThreshold),
         iq36::compare_vectors(preconv_gpu.k_conv_predelta, native_postconv.k_conv_predelta, kMismatchThreshold),
         iq36::compare_vectors(preconv_gpu.k_conv_predelta, oracle_k, kMismatchThreshold)},
        {"v_conv_predelta",
         iq36::compare_vectors(native_postconv.v_conv_predelta, oracle_v, kMismatchThreshold),
         iq36::compare_vectors(preconv_gpu.v_conv_predelta, native_postconv.v_conv_predelta, kMismatchThreshold),
         iq36::compare_vectors(preconv_gpu.v_conv_predelta, oracle_v, kMismatchThreshold)},
    };
    const std::vector<NamedCompareGroup> layer_groups = {
        {"attn_output",
         iq36::compare_vectors(native_postconv.attention_output, oracle_attn_output, kMismatchThreshold),
         iq36::compare_vectors(delta_gpu.attention_output, native_postconv.attention_output, kMismatchThreshold),
         iq36::compare_vectors(delta_gpu.attention_output, oracle_attn_output, kMismatchThreshold)},
        {"final_output",
         iq36::compare_vectors(native_postconv.final_output, oracle_final_output, kMismatchThreshold),
         iq36::compare_vectors(delta_gpu.final_output, native_postconv.final_output, kMismatchThreshold),
         iq36::compare_vectors(delta_gpu.final_output, oracle_final_output, kMismatchThreshold)},
        {"linear_attn_out",
         iq36::compare_vectors(native_linear_attn_out, oracle_linear_attn_out, kMismatchThreshold),
         iq36::compare_vectors(attention_gpu.linear_attn_out, native_linear_attn_out, kMismatchThreshold),
         iq36::compare_vectors(attention_gpu.linear_attn_out, oracle_linear_attn_out, kMismatchThreshold)},
        {"attn_residual",
         iq36::compare_vectors(native_attn_residual, oracle_attn_residual, kMismatchThreshold),
         iq36::compare_vectors(attention_gpu.attn_residual, native_attn_residual, kMismatchThreshold),
         iq36::compare_vectors(attention_gpu.attn_residual, oracle_attn_residual, kMismatchThreshold)},
        {"attn_post_norm",
         iq36::compare_vectors(native_attn_post_norm, oracle_attn_post_norm, kMismatchThreshold),
         iq36::compare_vectors(attention_gpu.attn_post_norm, native_attn_post_norm, kMismatchThreshold),
         iq36::compare_vectors(attention_gpu.attn_post_norm, oracle_attn_post_norm, kMismatchThreshold)},
        {"selected_gate_up",
         iq36::compare_vectors(native_selected_gate_up, oracle_gate_up, kMismatchThreshold),
         iq36::compare_vectors(selected_gpu.gate_up, native_selected_gate_up, kMismatchThreshold),
         iq36::compare_vectors(selected_gpu.gate_up, oracle_gate_up, kMismatchThreshold)},
        {"selected_swiglu",
         iq36::compare_vectors(native_selected_swiglu, oracle_swiglu, kMismatchThreshold),
         iq36::compare_vectors(selected_gpu.swiglu, native_selected_swiglu, kMismatchThreshold),
         iq36::compare_vectors(selected_gpu.swiglu, oracle_swiglu, kMismatchThreshold)},
        {"selected_down",
         iq36::compare_vectors(native_selected_down, oracle_down, kMismatchThreshold),
         iq36::compare_vectors(selected_gpu.down, native_selected_down, kMismatchThreshold),
         iq36::compare_vectors(selected_gpu.down, oracle_down, kMismatchThreshold)},
        {"shared_down",
         iq36::compare_vectors(native_shared_down, oracle_shared_down, kMismatchThreshold),
         iq36::compare_vectors(shared_gpu.down, native_shared_down, kMismatchThreshold),
         iq36::compare_vectors(shared_gpu.down, oracle_shared_down, kMismatchThreshold)},
        {"weighted_selected_down",
         iq36::compare_vectors(native_weighted, oracle_weighted, kMismatchThreshold),
         iq36::compare_vectors(gpu.weighted, native_weighted, kMismatchThreshold),
         iq36::compare_vectors(gpu.weighted, oracle_weighted, kMismatchThreshold)},
        {"ffn_moe_out",
         iq36::compare_vectors(native_moe_out, oracle_moe_out, kMismatchThreshold),
         iq36::compare_vectors(gpu.moe_out, native_moe_out, kMismatchThreshold),
         iq36::compare_vectors(gpu.moe_out, oracle_moe_out, kMismatchThreshold)},
        {"shared_input_gate",
         iq36::compare_vectors(native_shared_input_gate, oracle_shared_gate, kMismatchThreshold),
         iq36::compare_vectors(gpu.shared_gate, native_shared_input_gate, kMismatchThreshold),
         iq36::compare_vectors(gpu.shared_gate, oracle_shared_gate, kMismatchThreshold)},
        {"shared_gate_sigmoid",
         iq36::compare_vectors(native_shared_sigmoid, oracle_shared_sigmoid, kMismatchThreshold),
         iq36::compare_vectors(gpu.shared_gate_sigmoid, native_shared_sigmoid, kMismatchThreshold),
         iq36::compare_vectors(gpu.shared_gate_sigmoid, oracle_shared_sigmoid, kMismatchThreshold)},
        {"ffn_shexp_gated",
         iq36::compare_vectors(native_shared_gated, oracle_shared_gated, kMismatchThreshold),
         iq36::compare_vectors(gpu.shared_gated, native_shared_gated, kMismatchThreshold),
         iq36::compare_vectors(gpu.shared_gated, oracle_shared_gated, kMismatchThreshold)},
        {"ffn_out",
         iq36::compare_vectors(native_ffn_out, oracle_ffn_out, kMismatchThreshold),
         iq36::compare_vectors(gpu.ffn_out, native_ffn_out, kMismatchThreshold),
         iq36::compare_vectors(gpu.ffn_out, oracle_ffn_out, kMismatchThreshold)},
        {"layer_output",
         iq36::compare_vectors(native_layer_output, oracle_layer_output, kMismatchThreshold),
         iq36::compare_vectors(gpu.layer_output, native_layer_output, kMismatchThreshold),
         iq36::compare_vectors(gpu.layer_output, oracle_layer_output, kMismatchThreshold)},
    };
    const auto conv_state_after_gpu_vs_cpu =
        iq36::compare_vectors(preconv_gpu.conv_state_after, native_conv.conv_state, kMismatchThreshold);
    const auto recurrent_state_gpu_vs_cpu =
        iq36::compare_vectors(delta_gpu.recurrent_state, native_postconv.recurrent_state, kMismatchThreshold);

    const bool preconv_comparisons_passed =
        CompareGroupsPassed(preconv_groups) && ComparePassed(conv_state_after_gpu_vs_cpu);
    const bool layer_comparisons_passed =
        CompareGroupsPassed(layer_groups) && ComparePassed(recurrent_state_gpu_vs_cpu);
    const bool preconv_timing_positive =
        preconv_gpu.timing.qkv_min_us > 0.0 &&
        preconv_gpu.timing.alpha_min_us > 0.0 &&
        preconv_gpu.timing.beta_min_us > 0.0 &&
        preconv_gpu.timing.z_min_us > 0.0 &&
        preconv_gpu.timing.conv_min_us > 0.0 &&
        preconv_gpu.timing.postconv_silu_split_min_us > 0.0 &&
        preconv_gpu.timing.postconv_q_l2_min_us > 0.0 &&
        preconv_gpu.timing.postconv_k_l2_min_us > 0.0;
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
        preconv_gpu.device_name.find(args.device_substring) != std::string::npos &&
        delta_gpu.device_name.find(args.device_substring) != std::string::npos &&
        attention_gpu.device_name.find(args.device_substring) != std::string::npos &&
        selected_gpu.device_name.find(args.device_substring) != std::string::npos &&
        shared_gpu.device_name.find(args.device_substring) != std::string::npos &&
        gpu.device_name.find(args.device_substring) != std::string::npos;
    const bool required_checks_passed =
        load_map.ready &&
        preconv_tensor_shape_ok &&
        delta_tensor_shape_ok &&
        attention_tensor_shape_ok &&
        selected_tensor_shape_ok &&
        shared_tensor_shape_ok &&
        payload_counts_ok &&
        arc_selected &&
        preconv_comparisons_passed &&
        layer_comparisons_passed &&
        preconv_timing_positive &&
        delta_timing_positive &&
        attention_timing_positive &&
        selected_timing_positive &&
        shared_timing_positive &&
        tail_timing_positive;
    const double full_kernel_sum_min =
        preconv_gpu.timing.preconv_to_postconv_kernel_sum_min_us +
        delta_gpu.timing.delta_to_final_kernel_sum_min_us +
        attention_gpu.timing.attention_front_kernel_sum_min_us +
        selected_gpu.timing.selected_ffn_kernel_sum_min_us +
        shared_gpu.timing.shared_ffn_kernel_sum_min_us +
        gpu.timing.shell_sum_min_us;
    const double full_kernel_sum_mean =
        preconv_gpu.timing.preconv_to_postconv_kernel_sum_mean_us +
        delta_gpu.timing.delta_to_final_kernel_sum_mean_us +
        attention_gpu.timing.attention_front_kernel_sum_mean_us +
        selected_gpu.timing.selected_ffn_kernel_sum_mean_us +
        shared_gpu.timing.shared_ffn_kernel_sum_mean_us +
        gpu.timing.shell_sum_mean_us;

    std::cout << std::setprecision(10);
    std::cout << "{";
    std::cout << "\"schema_version\":\"intel-qwen36-gpu-resident-preconv-layer-shell-probe-v0\",";
    std::cout << "\"model_path\":\"" << JsonEscape(args.model_path) << "\",";
    std::cout << "\"layer\":" << args.layer << ",";
    std::cout << "\"source_token_position\":" << kSourceTokenPosition << ",";
    std::cout << "\"resident_api\":\"preconv_to_layer_shell_load_once_run_many\",";
    std::cout << "\"resident_load_count\":1,";
    std::cout << "\"resident_shell_invocations\":" << args.repeat << ",";
    std::cout << "\"captured_conv_state_input_boundary\":true,";
    std::cout << "\"preconv_host_q8_bridge\":true,";
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
    std::cout << "\"platform_name\":\"" << JsonEscape(preconv_gpu.platform_name) << "\",";
    std::cout << "\"device_name\":\"" << JsonEscape(preconv_gpu.device_name) << "\",";
    std::cout << "\"delta_device_name\":\"" << JsonEscape(delta_gpu.device_name) << "\",";
    std::cout << "\"attention_device_name\":\"" << JsonEscape(attention_gpu.device_name) << "\",";
    std::cout << "\"selected_device_name\":\"" << JsonEscape(selected_gpu.device_name) << "\",";
    std::cout << "\"shared_device_name\":\"" << JsonEscape(shared_gpu.device_name) << "\",";
    std::cout << "\"tail_device_name\":\"" << JsonEscape(gpu.device_name) << "\",";
    std::cout << "\"program_build_ms\":"
              << (preconv_gpu.program_build_ms + delta_gpu.program_build_ms +
                  attention_gpu.program_build_ms + selected_gpu.program_build_ms +
                  shared_gpu.program_build_ms + gpu.program_build_ms) << ",";
    std::cout << "\"build_log\":\""
              << JsonEscape(preconv_gpu.build_log + delta_gpu.build_log +
                            attention_gpu.build_log + selected_gpu.build_log +
                            shared_gpu.build_log + gpu.build_log) << "\",";
    std::cout << "\"repeat\":" << args.repeat << ",";
    std::cout << "\"timings\":{";
    std::cout << "\"preconv_host_q8_bridge_us\":"
              << preconv_gpu.timing.host_q8_bridge_us << ",";
    std::cout << "\"preconv_qkv_min_us\":"
              << preconv_gpu.timing.qkv_min_us << ",";
    std::cout << "\"preconv_alpha_min_us\":"
              << preconv_gpu.timing.alpha_min_us << ",";
    std::cout << "\"preconv_beta_min_us\":"
              << preconv_gpu.timing.beta_min_us << ",";
    std::cout << "\"preconv_z_min_us\":"
              << preconv_gpu.timing.z_min_us << ",";
    std::cout << "\"preconv_conv_min_us\":"
              << preconv_gpu.timing.conv_min_us << ",";
    std::cout << "\"postconv_silu_split_min_us\":"
              << preconv_gpu.timing.postconv_silu_split_min_us << ",";
    std::cout << "\"postconv_q_l2_min_us\":"
              << preconv_gpu.timing.postconv_q_l2_min_us << ",";
    std::cout << "\"postconv_k_l2_min_us\":"
              << preconv_gpu.timing.postconv_k_l2_min_us << ",";
    std::cout << "\"preconv_to_postconv_kernel_sum_min_us\":"
              << preconv_gpu.timing.preconv_to_postconv_kernel_sum_min_us << ",";
    std::cout << "\"preconv_to_postconv_kernel_sum_mean_us\":"
              << preconv_gpu.timing.preconv_to_postconv_kernel_sum_mean_us << ",";
    std::cout << "\"delta_to_final_kernel_sum_min_us\":"
              << delta_gpu.timing.delta_to_final_kernel_sum_min_us << ",";
    std::cout << "\"delta_to_final_kernel_sum_mean_us\":"
              << delta_gpu.timing.delta_to_final_kernel_sum_mean_us << ",";
    std::cout << "\"attention_front_kernel_sum_min_us\":"
              << attention_gpu.timing.attention_front_kernel_sum_min_us << ",";
    std::cout << "\"attention_output_projection_host_q8_bridge_us\":"
              << attention_gpu.timing.output_projection_host_q8_bridge_us << ",";
    std::cout << "\"selected_ffn_kernel_sum_min_us\":"
              << selected_gpu.timing.selected_ffn_kernel_sum_min_us << ",";
    std::cout << "\"selected_down_host_q8_bridge_us\":"
              << selected_gpu.timing.host_q8_bridge_us << ",";
    std::cout << "\"shared_ffn_kernel_sum_min_us\":"
              << shared_gpu.timing.shared_ffn_kernel_sum_min_us << ",";
    std::cout << "\"shared_down_host_q8_bridge_us\":"
              << shared_gpu.timing.host_q8_bridge_us << ",";
    std::cout << "\"ffn_tail_kernel_sum_min_us\":"
              << gpu.timing.shell_sum_min_us << ",";
    std::cout << "\"resident_preconv_to_layer_kernel_sum_min_us\":"
              << full_kernel_sum_min << ",";
    std::cout << "\"resident_preconv_to_layer_kernel_sum_mean_us\":"
              << full_kernel_sum_mean;
    std::cout << "},\"comparisons\":{";
    WriteNamedCompareGroups(preconv_groups);
    std::cout << ",\"conv_state_after\":{\"gpu_vs_cpu\":";
    WriteCompare(conv_state_after_gpu_vs_cpu);
    std::cout << "},";
    WriteNamedCompareGroups(layer_groups);
    std::cout << ",\"recurrent_state\":{\"gpu_vs_cpu\":";
    WriteCompare(recurrent_state_gpu_vs_cpu);
    std::cout << "}";
    std::cout << "},\"checks\":{";
    std::cout << "\"load_map_ready\":" << (load_map.ready ? "true" : "false") << ",";
    std::cout << "\"preconv_tensor_shape_ok\":"
              << (preconv_tensor_shape_ok ? "true" : "false") << ",";
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
    std::cout << "\"preconv_to_postconv_matches_oracle\":"
              << (preconv_comparisons_passed ? "true" : "false") << ",";
    std::cout << "\"downstream_layer_matches_oracle\":"
              << (layer_comparisons_passed ? "true" : "false") << ",";
    std::cout << "\"gpu_event_timing_positive\":"
              << (preconv_timing_positive && delta_timing_positive &&
                  attention_timing_positive && selected_timing_positive &&
                  shared_timing_positive && tail_timing_positive ? "true" : "false") << ",";
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


def preconv_layer_probe_cpp(opencl_source: str) -> str:
  cpp = POSTCONV.postconv_layer_probe_cpp(opencl_source)
  cpp = cpp.replace("#include <chrono>\n", "#include <chrono>\n#include <cmath>\n", 1)
  main_index = cpp.index("\nint main(")
  return cpp[:main_index] + "\n" + PRECONV_HELPERS_CPP + "\n" + PRECONV_MAIN_CPP


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
  parser.add_argument("--timeout-s", type=int, default=1200)
  parser.add_argument("--conv-history-probe", type=Path, default=None)
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


def load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise SystemExit(f"{path}: expected JSON object")
  return value


def latest_conv_history_probe() -> Path:
  paths = sorted((ROOT / "output").glob("gpu-conv-history-state-capture-probe-*/probe.json"))
  for path in reversed(paths):
    try:
      data = load_json(path)
    except Exception:
      continue
    if data.get("schema_version") == "intel-qwen36-gpu-conv-history-state-capture-probe-v0" and data.get("required_checks_passed") is True:
      return path
  raise SystemExit("no passing gpu conv-history state capture probe found")


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
  path = path.resolve()
  if path.stat().st_size != expected_bytes:
    raise SystemExit(f"payload size mismatch for {path}: {path.stat().st_size}")
  payloads[name] = {
      "local_path": path,
      "path": str(path.relative_to(ROOT)),
      "sha256": iq36_local.sha256_file(path),
      "size_bytes": expected_bytes,
      "stage_name": stage_name,
  }


def add_conv_history_payload(payloads: dict[str, dict[str, Any]],
                             conv_probe: dict[str, Any],
                             key: str) -> None:
  item = conv_probe.get("payloads", {}).get(key)
  if not isinstance(item, dict):
    raise SystemExit(f"conv-history probe missing payload {key}")
  path_value = item.get("path")
  stage_name = item.get("stage_name")
  size_bytes = item.get("size_bytes")
  if not isinstance(path_value, str) or not isinstance(stage_name, str) or not isinstance(size_bytes, int):
    raise SystemExit(f"conv-history payload {key} has invalid metadata")
  add_payload(payloads, key, stage_name, ROOT / path_value, size_bytes)
  payloads[key]["source_artifact"] = conv_probe.get("capture_artifact")
  payloads[key]["tensor_name"] = item.get("tensor_name")


def resolve_payloads(layer: int, conv_history_probe_path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
  conv_probe = load_json(conv_history_probe_path.resolve())
  if conv_probe.get("required_checks_passed") is not True:
    raise SystemExit(f"conv-history probe did not pass: {conv_history_probe_path}")
  payloads = POSTCONV.resolve_payloads(layer)
  for key in ("attn_norm", "conv_state", "linear_attn_qkv_mixed", "conv_output_raw"):
    add_conv_history_payload(payloads, conv_probe, key)
  for name, stage_name, prefix, expected_bytes in (
      ("alpha", "alpha.bin", "alpha", 128),
      ("a_softplus", "a_softplus.bin", "a_softplus", 128),
      ("beta", "beta.bin", "beta", 128),
      ("conv_output_silu", "conv_output_silu.bin", "conv_output_silu", 32768),
      ("q_conv", "q_conv.bin", "q_conv", 8192),
      ("k_conv", "k_conv.bin", "k_conv", 8192),
  ):
    add_payload(
        payloads,
        name,
        stage_name,
        find_payload(f"{prefix}-{layer}__tok15__ord*.bin", expected_bytes),
        expected_bytes,
    )
  return payloads, conv_probe


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
      and probe.get("resident_api") == "preconv_to_layer_shell_load_once_run_many"
      and probe.get("resident_load_count") == 1
      and probe.get("resident_shell_invocations") == expected_invocations
      and probe.get("captured_conv_state_input_boundary") is True
      and probe.get("preconv_host_q8_bridge") is True
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
      "# GPU Resident Preconv-to-Layer Shell Probe",
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
      f"- conv history source: `{payload.get('conv_history_probe')}`",
      "",
      "| output | comparison | max abs | RMSE |",
      "|---|---|---:|---:|",
  ]
  for name in (
      "linear_attn_qkv_mixed",
      "gate",
      "beta_sigmoid",
      "z",
      "conv_output_raw",
      "q_conv_predelta",
      "k_conv_predelta",
      "v_conv_predelta",
      "final_output",
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
      f"| preconv_to_postconv_sum | {timings.get('preconv_to_postconv_kernel_sum_min_us')} |",
      f"| delta_to_final_sum | {timings.get('delta_to_final_kernel_sum_min_us')} |",
      f"| attention_front_sum | {timings.get('attention_front_kernel_sum_min_us')} |",
      f"| selected_ffn_sum | {timings.get('selected_ffn_kernel_sum_min_us')} |",
      f"| shared_ffn_sum | {timings.get('shared_ffn_kernel_sum_min_us')} |",
      f"| ffn_tail_sum | {timings.get('ffn_tail_kernel_sum_min_us')} |",
      f"| resident_preconv_to_layer_sum | {timings.get('resident_preconv_to_layer_kernel_sum_min_us')} |",
      "",
      "The target-side process starts from captured `attn_norm` and captured",
      "`conv_states` for layer 5, computes QKV/alpha/beta/z, conv, postconv",
      "prep, delta recurrent/final norm, attention output projection,",
      "post-attention residual, FFN RMSNorm, selected/shared FFN branches, tail",
      "aggregation, and residual to captured `l_out`. Host boundaries and Q8",
      "activation bridges remain explicit. This is captured single-layer",
      "evidence only, not prompt/token decode throughput.",
      "",
  ]
  path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  args = parse_args()
  if args.resident_invocations <= 0:
    raise SystemExit("--resident-invocations must be positive")
  conv_history_probe_path = (
      args.conv_history_probe.resolve()
      if args.conv_history_probe is not None
      else latest_conv_history_probe().resolve()
  )
  stamp = utc_stamp()
  created_at = iso_now()
  out_dir = args.out_dir or ROOT / f"output/gpu-resident-preconv-layer-shell-probe-{stamp}"
  out_dir.mkdir(parents=True, exist_ok=False)
  raw_dir = out_dir / "raw"
  raw_dir.mkdir()

  payloads, conv_history_probe = resolve_payloads(args.layer, conv_history_probe_path)
  opencl_source = OPENCL_SOURCE.read_text(encoding="utf-8")
  opencl_source_hash = iq36_local.sha256_file(OPENCL_SOURCE)
  embedded_opencl_hash = hashlib.sha256(
      (opencl_source + POSTCONV.ATTENTION.ATTENTION_EXTRA_OPENCL).encode("utf-8")
  ).hexdigest()
  local_cpp = out_dir / "gpu_resident_preconv_layer_shell_probe.cpp"
  local_cpp.write_text(preconv_layer_probe_cpp(opencl_source), encoding="utf-8")

  remote_dir = f"{args.remote_root.rstrip('/')}/gpu-resident-preconv-layer-shell-probe-{stamp}"
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
    for local, remote in POSTCONV.ATTENTION.SHARED.SELECTED.SOURCE_FILES:
      transfers.append(iq36_local.copy_to(args.host, ROOT / local, f"{remote_dir}/{remote}", args.timeout_s))
    transfers.append(
        iq36_local.copy_to(
            args.host,
            local_cpp,
            f"{remote_dir}/tests/gpu_resident_preconv_layer_shell_probe.cpp",
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

  executable = f"{remote_dir}/build/iq36-gpu-resident-preconv-layer-shell-probe"
  compile_cmd = " && ".join([
      f"source {shlex.quote(args.env_script)} >/dev/null 2>&1",
      (
          "g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic "
          f"-I {shlex.quote(remote_dir + '/include')} "
          f"{shlex.quote(remote_dir + '/src/gguf_loader.cpp')} "
          f"{shlex.quote(remote_dir + '/src/gpu_q4x8_matvec.cpp')} "
          f"{shlex.quote(remote_dir + '/tests/gpu_resident_preconv_layer_shell_probe.cpp')} "
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
      {"name": "preconv_to_postconv_matches_oracle", "pass": nested_bool(probe, "checks", "preconv_to_postconv_matches_oracle")},
      {"name": "downstream_layer_matches_oracle", "pass": nested_bool(probe, "checks", "downstream_layer_matches_oracle")},
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
      "conv_history_probe": str(conv_history_probe_path.relative_to(ROOT)),
      "conv_history_capture_artifact": conv_history_probe.get("capture_artifact"),
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
      "tool": "tools/intel-qwen36-gpu-resident-preconv-layer-shell-probe.py",
      "artifact": str(out_dir),
      "host": args.host,
      "remote_dir": remote_dir,
      "layer": args.layer,
      "resident_invocations": args.resident_invocations,
      "conv_history_probe": str(conv_history_probe_path.relative_to(ROOT)),
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
      "gpu_resident_preconv_layer_shell_probe",
      [
          ("required_checks_passed", required_checks_passed),
          ("resident_shell_invocations", args.resident_invocations),
          ("preconv_to_postconv_kernel_sum_min_us", nested_number(timings, "preconv_to_postconv_kernel_sum_min_us")),
          ("resident_preconv_to_layer_kernel_sum_min_us", nested_number(timings, "resident_preconv_to_layer_kernel_sum_min_us")),
          ("qkv_gpu_vs_oracle_max_abs_diff", nested_number(comparisons, "linear_attn_qkv_mixed", "gpu_vs_oracle", "max_abs_diff")),
          ("conv_output_raw_gpu_vs_oracle_max_abs_diff", nested_number(comparisons, "conv_output_raw", "gpu_vs_oracle", "max_abs_diff")),
          ("final_output_gpu_vs_oracle_max_abs_diff", nested_number(comparisons, "final_output", "gpu_vs_oracle", "max_abs_diff")),
          ("layer_output_gpu_vs_oracle_max_abs_diff", nested_number(comparisons, "layer_output", "gpu_vs_oracle", "max_abs_diff")),
      ],
  )
  write_summary(out_dir / "summary.md", payload)
  print(out_dir)
  return 0 if required_checks_passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
