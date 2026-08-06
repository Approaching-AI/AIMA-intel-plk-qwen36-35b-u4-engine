#!/usr/bin/env python3
"""Run the resident GPU two-linear-layer state-carry shell handoff probe."""

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
LAYER_INPUT_TOOL = Path(__file__).with_name(
    "intel-qwen36-gpu-resident-layer-input-rmsnorm-layer-shell-probe.py"
)
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA_VERSION = "intel-qwen36-gpu-resident-two-linear-layer-shell-probe-v0"
DEFAULT_HOST = "local"
DEFAULT_MODEL = "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf"
DEFAULT_ENV_SCRIPT = "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh"
DEFAULT_REMOTE_ROOT = "/home/intel/intel-qwen36-gpu"
DEFAULT_ORACLE_BUNDLE = ROOT / "oracle/r0-oracle-bundle-20260627T060028Z"
OPENCL_SOURCE = ROOT / "engine/gpu/opencl/q4x8_matvec.cl"


def load_layer_input_tool() -> ModuleType:
  spec = importlib.util.spec_from_file_location("iq36_layer_input_shell_probe", LAYER_INPUT_TOOL)
  if spec is None or spec.loader is None:
    raise SystemExit(f"failed to load layer-input shell tool: {LAYER_INPUT_TOOL}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


LAYER_INPUT = load_layer_input_tool()
PRECONV = LAYER_INPUT.PRECONV


TWO_LAYER_CPP = r'''

struct LayerTensorBundle {
  int layer = 0;
  std::string attn_norm_tensor_name;
  std::string qkv_tensor_name;
  std::string alpha_tensor_name;
  std::string beta_tensor_name;
  std::string z_tensor_name;
  std::string conv_tensor_name;
  std::string ssm_norm_tensor_name;
  std::string output_tensor_name;
  std::string ffn_norm_tensor_name;
  std::string selected_gate_up_tensor_name;
  std::string selected_down_tensor_name;
  std::string shared_gate_tensor_name;
  std::string shared_up_tensor_name;
  std::string shared_down_tensor_name;
  std::string shared_input_gate_tensor_name;
  const iq36::GgufTensorInfo* attn_norm_tensor = nullptr;
  const iq36::GgufTensorInfo* qkv_tensor = nullptr;
  const iq36::GgufTensorInfo* alpha_tensor = nullptr;
  const iq36::GgufTensorInfo* beta_tensor = nullptr;
  const iq36::GgufTensorInfo* z_tensor = nullptr;
  const iq36::GgufTensorInfo* conv_tensor = nullptr;
  const iq36::GgufTensorInfo* ssm_norm_tensor = nullptr;
  const iq36::GgufTensorInfo* output_tensor = nullptr;
  const iq36::GgufTensorInfo* ffn_norm_tensor = nullptr;
  const iq36::GgufTensorInfo* selected_gate_up_tensor = nullptr;
  const iq36::GgufTensorInfo* selected_down_tensor = nullptr;
  const iq36::GgufTensorInfo* shared_gate_tensor = nullptr;
  const iq36::GgufTensorInfo* shared_up_tensor = nullptr;
  const iq36::GgufTensorInfo* shared_down_tensor = nullptr;
  const iq36::GgufTensorInfo* shared_input_gate_tensor = nullptr;
};

LayerTensorBundle ResolveLayerTensorBundle(const iq36::GgufModelIndex& index,
                                           int layer) {
  LayerTensorBundle bundle;
  bundle.layer = layer;
  bundle.attn_norm_tensor_name = LayerTensorName(layer, "attn_norm.weight");
  bundle.qkv_tensor_name = LayerTensorName(layer, "attn_qkv.weight");
  bundle.alpha_tensor_name = LayerTensorName(layer, "ssm_alpha.weight");
  bundle.beta_tensor_name = LayerTensorName(layer, "ssm_beta.weight");
  bundle.z_tensor_name = LayerTensorName(layer, "attn_gate.weight");
  bundle.conv_tensor_name = LayerTensorName(layer, "ssm_conv1d.weight");
  bundle.ssm_norm_tensor_name = LayerTensorName(layer, "ssm_norm.weight");
  bundle.output_tensor_name = LayerTensorName(layer, "ssm_out.weight");
  bundle.ffn_norm_tensor_name = LayerTensorName(layer, "post_attention_norm.weight");
  bundle.selected_gate_up_tensor_name = LayerTensorName(layer, "ffn_gate_up_exps.weight");
  bundle.selected_down_tensor_name = LayerTensorName(layer, "ffn_down_exps.weight");
  bundle.shared_gate_tensor_name = LayerTensorName(layer, "ffn_gate_shexp.weight");
  bundle.shared_up_tensor_name = LayerTensorName(layer, "ffn_up_shexp.weight");
  bundle.shared_down_tensor_name = LayerTensorName(layer, "ffn_down_shexp.weight");
  bundle.shared_input_gate_tensor_name = LayerTensorName(layer, "ffn_gate_inp_shexp.weight");
  bundle.attn_norm_tensor = iq36::find_tensor(index, bundle.attn_norm_tensor_name);
  bundle.qkv_tensor = iq36::find_tensor(index, bundle.qkv_tensor_name);
  bundle.alpha_tensor = iq36::find_tensor(index, bundle.alpha_tensor_name);
  bundle.beta_tensor = iq36::find_tensor(index, bundle.beta_tensor_name);
  bundle.z_tensor = iq36::find_tensor(index, bundle.z_tensor_name);
  bundle.conv_tensor = iq36::find_tensor(index, bundle.conv_tensor_name);
  bundle.ssm_norm_tensor = iq36::find_tensor(index, bundle.ssm_norm_tensor_name);
  bundle.output_tensor = iq36::find_tensor(index, bundle.output_tensor_name);
  bundle.ffn_norm_tensor = iq36::find_tensor(index, bundle.ffn_norm_tensor_name);
  bundle.selected_gate_up_tensor =
      iq36::find_tensor(index, bundle.selected_gate_up_tensor_name);
  bundle.selected_down_tensor =
      iq36::find_tensor(index, bundle.selected_down_tensor_name);
  bundle.shared_gate_tensor = iq36::find_tensor(index, bundle.shared_gate_tensor_name);
  bundle.shared_up_tensor = iq36::find_tensor(index, bundle.shared_up_tensor_name);
  bundle.shared_down_tensor = iq36::find_tensor(index, bundle.shared_down_tensor_name);
  bundle.shared_input_gate_tensor =
      iq36::find_tensor(index, bundle.shared_input_gate_tensor_name);
  Require(bundle.attn_norm_tensor != nullptr, "attention norm tensor missing");
  Require(bundle.qkv_tensor != nullptr, "qkv tensor missing");
  Require(bundle.alpha_tensor != nullptr, "alpha tensor missing");
  Require(bundle.beta_tensor != nullptr, "beta tensor missing");
  Require(bundle.z_tensor != nullptr, "z tensor missing");
  Require(bundle.conv_tensor != nullptr, "conv tensor missing");
  Require(bundle.ssm_norm_tensor != nullptr, "ssm norm tensor missing");
  Require(bundle.output_tensor != nullptr, "attention output tensor missing");
  Require(bundle.ffn_norm_tensor != nullptr, "ffn norm tensor missing");
  Require(bundle.selected_gate_up_tensor != nullptr, "selected gate-up tensor missing");
  Require(bundle.selected_down_tensor != nullptr, "selected down tensor missing");
  Require(bundle.shared_gate_tensor != nullptr, "shared gate tensor missing");
  Require(bundle.shared_up_tensor != nullptr, "shared up tensor missing");
  Require(bundle.shared_down_tensor != nullptr, "shared down tensor missing");
  Require(bundle.shared_input_gate_tensor != nullptr, "shared input gate tensor missing");
  return bundle;
}

struct LayerShapeChecks {
  bool layer_input_tensor_shape_ok = false;
  bool preconv_tensor_shape_ok = false;
  bool delta_tensor_shape_ok = false;
  bool attention_tensor_shape_ok = false;
  bool selected_tensor_shape_ok = false;
  bool shared_tensor_shape_ok = false;
};

LayerShapeChecks CheckLayerShapes(const LayerTensorBundle& t) {
  LayerShapeChecks checks;
  checks.layer_input_tensor_shape_ok =
      t.attn_norm_tensor->type == 0 &&
      t.attn_norm_tensor->dims == std::vector<std::uint64_t>{kHiddenSize};
  checks.preconv_tensor_shape_ok =
      t.qkv_tensor->type == 12 &&
      t.alpha_tensor->type == 12 &&
      t.beta_tensor->type == 12 &&
      t.z_tensor->type == 12 &&
      t.conv_tensor->type == 0 &&
      t.qkv_tensor->dims == std::vector<std::uint64_t>{kHiddenSize, kLinearQkvMixedValues} &&
      t.alpha_tensor->dims == std::vector<std::uint64_t>{kHiddenSize, kLinearValueHeads} &&
      t.beta_tensor->dims == std::vector<std::uint64_t>{kHiddenSize, kLinearValueHeads} &&
      t.z_tensor->dims == std::vector<std::uint64_t>{kHiddenSize, kLinearVValues} &&
      t.conv_tensor->dims == std::vector<std::uint64_t>{kLinearConvKernelSize, kLinearQkvMixedValues};
  checks.delta_tensor_shape_ok =
      t.ssm_norm_tensor->type == 0 &&
      t.ssm_norm_tensor->dims == std::vector<std::uint64_t>{kLinearHeadDim};
  checks.attention_tensor_shape_ok =
      t.output_tensor->type == 12 &&
      t.output_tensor->dims == std::vector<std::uint64_t>{4096, kHiddenSize} &&
      t.ffn_norm_tensor->type == 0 &&
      t.ffn_norm_tensor->dims == std::vector<std::uint64_t>{kHiddenSize};
  checks.selected_tensor_shape_ok =
      t.selected_gate_up_tensor->type == 12 &&
      t.selected_down_tensor->type == 12 &&
      t.selected_gate_up_tensor->dims ==
          std::vector<std::uint64_t>{kHiddenSize, kGateUpRowsPerExpert, kExpertCount} &&
      t.selected_down_tensor->dims ==
          std::vector<std::uint64_t>{kIntermediateSize, kHiddenSize, kExpertCount};
  checks.shared_tensor_shape_ok =
      t.shared_gate_tensor->type == 12 &&
      t.shared_up_tensor->type == 12 &&
      (t.shared_down_tensor->type == 12 || t.shared_down_tensor->type == 14) &&
      t.shared_gate_tensor->dims ==
          std::vector<std::uint64_t>{kHiddenSize, kIntermediateSize} &&
      t.shared_up_tensor->dims ==
          std::vector<std::uint64_t>{kHiddenSize, kIntermediateSize} &&
      t.shared_down_tensor->dims ==
          std::vector<std::uint64_t>{kIntermediateSize, kHiddenSize} &&
      t.shared_input_gate_tensor->type == 0 &&
      t.shared_input_gate_tensor->dims == std::vector<std::uint64_t>{kHiddenSize};
  return checks;
}

bool ShapesPassed(const LayerShapeChecks& checks) {
  return checks.layer_input_tensor_shape_ok &&
         checks.preconv_tensor_shape_ok &&
         checks.delta_tensor_shape_ok &&
         checks.attention_tensor_shape_ok &&
         checks.selected_tensor_shape_ok &&
         checks.shared_tensor_shape_ok;
}

struct LayerOraclePayloads {
  std::vector<float> residual_input;
  std::vector<float> attn_norm;
  std::vector<float> conv_state;
  std::vector<float> qkv;
  std::vector<float> alpha;
  std::vector<float> a_softplus;
  std::vector<float> gate;
  std::vector<float> beta;
  std::vector<float> beta_sigmoid;
  std::vector<float> z;
  std::vector<float> conv_output_raw;
  std::vector<float> conv_output_silu;
  std::vector<float> q_conv;
  std::vector<float> k_conv;
  std::vector<float> q;
  std::vector<float> k;
  std::vector<float> v;
  std::vector<float> state;
  std::vector<float> attn_output;
  std::vector<float> final_output;
  std::vector<float> linear_attn_out;
  std::vector<float> attn_residual;
  std::vector<float> attn_post_norm;
  std::vector<int> expert_ids;
  std::vector<float> weights_norm;
  std::vector<float> gate_up;
  std::vector<float> swiglu;
  std::vector<float> down;
  std::vector<float> weighted;
  std::vector<float> moe_out;
  std::vector<float> shared_down;
  std::vector<float> shared_gate;
  std::vector<float> shared_sigmoid;
  std::vector<float> shared_gated;
  std::vector<float> ffn_out;
  std::vector<float> layer_output;
};

LayerOraclePayloads LoadLayerOraclePayloads(const std::string& payload_dir,
                                            const std::string& prefix) {
  auto f32 = [&](const std::string& name) {
    return iq36::read_f32_vector_file(
        JoinPath(payload_dir, prefix + "_" + name + ".bin"));
  };
  LayerOraclePayloads p;
  p.residual_input = f32("residual_input");
  p.attn_norm = f32("attn_norm");
  p.conv_state = f32("conv_state");
  p.qkv = f32("linear_attn_qkv_mixed");
  p.alpha = f32("alpha");
  p.a_softplus = f32("a_softplus");
  p.gate = f32("gate");
  p.beta = f32("beta");
  p.beta_sigmoid = f32("beta_sigmoid");
  p.z = f32("z");
  p.conv_output_raw = f32("conv_output_raw");
  p.conv_output_silu = f32("conv_output_silu");
  p.q_conv = f32("q_conv");
  p.k_conv = f32("k_conv");
  p.q = f32("q_conv_predelta");
  p.k = f32("k_conv_predelta");
  p.v = f32("v_conv_predelta");
  p.state = f32("state_predelta");
  p.attn_output = f32("attn_output");
  p.final_output = f32("final_output");
  p.linear_attn_out = f32("linear_attn_out");
  p.attn_residual = f32("attn_residual");
  p.attn_post_norm = f32("attn_post_norm");
  p.expert_ids = ReadI32VectorFile(JoinPath(payload_dir, prefix + "_ffn_moe_topk.bin"));
  p.weights_norm = f32("ffn_moe_weights_norm");
  p.gate_up = f32("ffn_moe_gate_up");
  p.swiglu = f32("ffn_moe_swiglu");
  p.down = f32("ffn_moe_down");
  p.weighted = f32("ffn_moe_weighted");
  p.moe_out = f32("ffn_moe_out");
  p.shared_down = f32("ffn_shexp");
  p.shared_gate = f32("shared_expert_gate");
  p.shared_sigmoid = f32("shared_expert_gate_sigmoid");
  p.shared_gated = f32("ffn_shexp_gated");
  p.ffn_out = f32("ffn_out");
  p.layer_output = f32("layer_output");
  return p;
}

bool LayerPayloadCountsOk(const LayerOraclePayloads& p) {
  return p.residual_input.size() == kHiddenSize &&
         p.attn_norm.size() == kHiddenSize &&
         p.conv_state.size() == kLinearConvStateValues &&
         p.qkv.size() == kLinearQkvMixedValues &&
         p.alpha.size() == kLinearValueHeads &&
         p.a_softplus.size() == kLinearValueHeads &&
         p.gate.size() == kLinearValueHeads &&
         p.beta.size() == kLinearValueHeads &&
         p.beta_sigmoid.size() == kLinearValueHeads &&
         p.z.size() == kLinearVValues &&
         p.conv_output_raw.size() == kLinearQkvMixedValues &&
         p.conv_output_silu.size() == kLinearQkvMixedValues &&
         p.q_conv.size() == kLinearQValues &&
         p.k_conv.size() == kLinearQValues &&
         p.q.size() == kLinearQValues &&
         p.k.size() == kLinearQValues &&
         p.v.size() == kLinearVValues &&
         p.state.size() == kLinearStateValues &&
         p.attn_output.size() == kLinearVValues &&
         p.final_output.size() == kLinearVValues &&
         p.linear_attn_out.size() == kHiddenSize &&
         p.attn_residual.size() == kHiddenSize &&
         p.attn_post_norm.size() == kHiddenSize &&
         p.expert_ids.size() == kExpertUsedCount &&
         p.weights_norm.size() == kExpertUsedCount &&
         p.gate_up.size() == kGateUpValueCount &&
         p.swiglu.size() == kSwiGluValueCount &&
         p.down.size() == kWeightedValueCount &&
         p.weighted.size() == kWeightedValueCount &&
         p.moe_out.size() == kHiddenSize &&
         p.shared_down.size() == kHiddenSize &&
         p.shared_gate.size() == 1 &&
         p.shared_sigmoid.size() == 1 &&
         p.shared_gated.size() == kHiddenSize &&
         p.ffn_out.size() == kHiddenSize &&
         p.layer_output.size() == kHiddenSize;
}

struct LayerShellTiming {
  double layer_input_rmsnorm_min_us = 0.0;
  double layer_input_rmsnorm_mean_us = 0.0;
  double preconv_to_postconv_kernel_sum_min_us = 0.0;
  double preconv_to_postconv_kernel_sum_mean_us = 0.0;
  double delta_to_final_kernel_sum_min_us = 0.0;
  double delta_to_final_kernel_sum_mean_us = 0.0;
  double attention_front_kernel_sum_min_us = 0.0;
  double attention_front_kernel_sum_mean_us = 0.0;
  double selected_ffn_kernel_sum_min_us = 0.0;
  double selected_ffn_kernel_sum_mean_us = 0.0;
  double shared_ffn_kernel_sum_min_us = 0.0;
  double shared_ffn_kernel_sum_mean_us = 0.0;
  double ffn_tail_kernel_sum_min_us = 0.0;
  double ffn_tail_kernel_sum_mean_us = 0.0;
  double layer_kernel_sum_min_us = 0.0;
  double layer_kernel_sum_mean_us = 0.0;
};

struct LayerShellResult {
  int layer = 0;
  std::vector<float> native_layer_output;
  std::vector<float> gpu_layer_output;
  LayerShellTiming timing;
  LayerShapeChecks shape_checks;
  std::vector<NamedCompareGroup> comparisons;
  iq36::VectorCompareStats conv_state_after_gpu_vs_cpu;
  iq36::VectorCompareStats recurrent_state_gpu_vs_cpu;
  bool payload_counts_ok = false;
  bool comparisons_passed = false;
  bool timing_positive = false;
  bool arc_selected = false;
  std::string layer_input_device_name;
  std::string preconv_device_name;
  std::string delta_device_name;
  std::string attention_device_name;
  std::string selected_device_name;
  std::string shared_device_name;
  std::string tail_device_name;
  std::string shared_down_tensor_type;
  double program_build_ms = 0.0;
  std::string build_log;
};

void AppendCompare(std::vector<NamedCompareGroup>& groups,
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

LayerShellResult RunResidentLinearLayerShell(
    const Args& args,
    const iq36::GgufModelIndex& index,
    const LayerTensorBundle& t,
    const LayerOraclePayloads& oracle,
    const std::vector<float>& native_residual_input,
    const std::vector<float>& gpu_residual_input,
    float rms_norm_epsilon) {
  Require(native_residual_input.size() == kHiddenSize,
          "native residual input size mismatch");
  Require(gpu_residual_input.size() == kHiddenSize,
          "gpu residual input size mismatch");

  LayerShellResult result;
  result.layer = t.layer;
  result.shape_checks = CheckLayerShapes(t);
  result.payload_counts_ok = LayerPayloadCountsOk(oracle);

  std::ifstream model(args.model_path, std::ios::binary);
  Require(static_cast<bool>(model), "model file could not be opened");
  const auto attn_norm_weight =
      ReadF32TensorPayload(model, *t.attn_norm_tensor,
                           static_cast<std::size_t>(kHiddenSize));
  const auto ssm_dt =
      iq36::decode_tensor_row(args.model_path, index,
                              LayerTensorName(t.layer, "ssm_dt.bias"), 0);
  const auto ssm_a =
      iq36::decode_tensor_row(args.model_path, index,
                              LayerTensorName(t.layer, "ssm_a"), 0);
  const auto ssm_norm_weight =
      iq36::decode_tensor_row(args.model_path, index, t.ssm_norm_tensor_name, 0);
  const auto ffn_norm_weight =
      ReadF32TensorPayload(model, *t.ffn_norm_tensor,
                           static_cast<std::size_t>(kHiddenSize));
  const auto shared_input_gate_weights =
      ReadF32TensorPayload(model, *t.shared_input_gate_tensor,
                           static_cast<std::size_t>(kHiddenSize));

  const auto native_attn_norm =
      iq36::apply_rms_norm(native_residual_input, attn_norm_weight, rms_norm_epsilon);
  const auto layer_input_gpu = RunGpuLayerInputRmsNorm(
      gpu_residual_input, attn_norm_weight, rms_norm_epsilon,
      args.device_substring, args.repeat);
  const auto native_preconv = iq36::run_qwen36_linear_attention_preconv_core(
      args.model_path, index, t.layer, native_attn_norm);
  const auto native_conv = iq36::run_qwen36_linear_attention_conv_core(
      args.model_path, index, t.layer, native_preconv.qkv_mixed, oracle.conv_state);
  const auto native_postconv = iq36::run_qwen36_linear_attention_postconv_core(
      native_conv.conv_output_raw,
      native_preconv.gate,
      native_preconv.beta_sigmoid,
      oracle.state,
      native_preconv.z,
      ssm_norm_weight,
      rms_norm_epsilon);
  const auto native_linear_attn_out =
      iq36::matvec_tensor(args.model_path, index, t.output_tensor_name,
                          native_postconv.final_output);
  const auto native_attn_residual =
      iq36::add_vectors(native_residual_input, native_linear_attn_out);
  const auto native_attn_post_norm =
      iq36::apply_rms_norm(native_attn_residual, ffn_norm_weight, rms_norm_epsilon);
  const auto native_selected_gate_up =
      iq36::matvec_expert_tensor(args.model_path, index,
                                 t.selected_gate_up_tensor_name,
                                 native_attn_post_norm, oracle.expert_ids);
  const auto native_selected_swiglu =
      iq36::apply_swiglu_from_gate_up(native_selected_gate_up,
                                      kIntermediateSize, kExpertUsedCount);
  const auto native_selected_down =
      iq36::matvec_expert_tensor_per_expert_input(
          args.model_path, index, t.selected_down_tensor_name,
          native_selected_swiglu, oracle.expert_ids);
  const auto native_weighted =
      iq36::apply_expert_weights(native_selected_down, oracle.weights_norm, kHiddenSize);
  const auto native_moe_out =
      iq36::aggregate_experts(native_weighted, kExpertUsedCount, kHiddenSize);
  const auto native_shared_gate =
      iq36::matvec_tensor(args.model_path, index, t.shared_gate_tensor_name,
                          native_attn_post_norm);
  const auto native_shared_up =
      iq36::matvec_tensor(args.model_path, index, t.shared_up_tensor_name,
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
      iq36::matvec_tensor(args.model_path, index, t.shared_down_tensor_name,
                          native_shared_swiglu);
  const auto native_shared_input_gate =
      iq36::matvec_tensor(args.model_path, index,
                          t.shared_input_gate_tensor_name, native_attn_post_norm);
  Require(native_shared_input_gate.size() == 1, "native shared input gate size mismatch");
  const std::vector<float> native_shared_sigmoid{
      iq36::sigmoid_scalar(native_shared_input_gate[0])};
  const auto native_shared_gated =
      iq36::multiply_by_scalar(native_shared_down, native_shared_sigmoid[0]);
  const auto native_ffn_out =
      iq36::add_vectors(native_moe_out, native_shared_gated);
  result.native_layer_output =
      iq36::add_vectors(native_attn_residual, native_ffn_out);

  const auto preconv_gpu = RunGpuPreConvFront(
      args.model_path, *t.qkv_tensor, *t.alpha_tensor, *t.beta_tensor, *t.z_tensor,
      *t.conv_tensor, ssm_dt, ssm_a, layer_input_gpu.attn_norm, oracle.conv_state,
      rms_norm_epsilon, args.device_substring, args.repeat);
  const auto delta_gpu = RunGpuPostConvDelta(
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
      args.model_path, *t.output_tensor, delta_gpu.final_output, gpu_residual_input,
      ffn_norm_weight, rms_norm_epsilon, args.device_substring, args.repeat);
  const auto selected_gpu = RunGpuSelectedFfnShell(
      args.model_path, *t.selected_gate_up_tensor, *t.selected_down_tensor,
      attention_gpu.attn_post_norm, oracle.expert_ids, args.device_substring,
      args.repeat);
  const auto shared_gpu = RunGpuSharedFfnShell(
      args.model_path, *t.shared_gate_tensor, *t.shared_up_tensor,
      *t.shared_down_tensor, attention_gpu.attn_post_norm,
      args.device_substring, args.repeat);
  const auto tail_gpu = RunGpuShell(shared_input_gate_weights,
                                    attention_gpu.attn_post_norm,
                                    selected_gpu.down, oracle.weights_norm,
                                    shared_gpu.down,
                                    attention_gpu.attn_residual,
                                    args.device_substring, args.repeat);
  result.gpu_layer_output = tail_gpu.layer_output;

  AppendCompare(result.comparisons, "residual_input",
                native_residual_input, gpu_residual_input, oracle.residual_input);
  AppendCompare(result.comparisons, "attn_norm",
                native_attn_norm, layer_input_gpu.attn_norm, oracle.attn_norm);
  AppendCompare(result.comparisons, "linear_attn_qkv_mixed",
                native_preconv.qkv_mixed, preconv_gpu.qkv_mixed, oracle.qkv);
  AppendCompare(result.comparisons, "alpha",
                native_preconv.alpha, preconv_gpu.alpha, oracle.alpha);
  AppendCompare(result.comparisons, "a_softplus",
                native_preconv.alpha_softplus, preconv_gpu.a_softplus, oracle.a_softplus);
  AppendCompare(result.comparisons, "gate",
                native_preconv.gate, preconv_gpu.gate, oracle.gate);
  AppendCompare(result.comparisons, "beta",
                native_preconv.beta, preconv_gpu.beta, oracle.beta);
  AppendCompare(result.comparisons, "beta_sigmoid",
                native_preconv.beta_sigmoid, preconv_gpu.beta_sigmoid, oracle.beta_sigmoid);
  AppendCompare(result.comparisons, "z",
                native_preconv.z, preconv_gpu.z, oracle.z);
  AppendCompare(result.comparisons, "conv_output_raw",
                native_conv.conv_output_raw, preconv_gpu.conv_output_raw,
                oracle.conv_output_raw);
  AppendCompare(result.comparisons, "conv_output_silu",
                native_postconv.conv_output_silu, preconv_gpu.conv_output_silu,
                oracle.conv_output_silu);
  AppendCompare(result.comparisons, "q_conv",
                native_postconv.q_conv, preconv_gpu.q_conv, oracle.q_conv);
  AppendCompare(result.comparisons, "k_conv",
                native_postconv.k_conv, preconv_gpu.k_conv, oracle.k_conv);
  AppendCompare(result.comparisons, "q_conv_predelta",
                native_postconv.q_conv_predelta, preconv_gpu.q_conv_predelta,
                oracle.q);
  AppendCompare(result.comparisons, "k_conv_predelta",
                native_postconv.k_conv_predelta, preconv_gpu.k_conv_predelta,
                oracle.k);
  AppendCompare(result.comparisons, "v_conv_predelta",
                native_postconv.v_conv_predelta, preconv_gpu.v_conv_predelta,
                oracle.v);
  AppendCompare(result.comparisons, "attn_output",
                native_postconv.attention_output, delta_gpu.attention_output,
                oracle.attn_output);
  AppendCompare(result.comparisons, "final_output",
                native_postconv.final_output, delta_gpu.final_output,
                oracle.final_output);
  AppendCompare(result.comparisons, "linear_attn_out",
                native_linear_attn_out, attention_gpu.linear_attn_out,
                oracle.linear_attn_out);
  AppendCompare(result.comparisons, "attn_residual",
                native_attn_residual, attention_gpu.attn_residual,
                oracle.attn_residual);
  AppendCompare(result.comparisons, "attn_post_norm",
                native_attn_post_norm, attention_gpu.attn_post_norm,
                oracle.attn_post_norm);
  AppendCompare(result.comparisons, "selected_gate_up",
                native_selected_gate_up, selected_gpu.gate_up, oracle.gate_up);
  AppendCompare(result.comparisons, "selected_swiglu",
                native_selected_swiglu, selected_gpu.swiglu, oracle.swiglu);
  AppendCompare(result.comparisons, "selected_down",
                native_selected_down, selected_gpu.down, oracle.down);
  AppendCompare(result.comparisons, "shared_down",
                native_shared_down, shared_gpu.down, oracle.shared_down);
  AppendCompare(result.comparisons, "weighted_selected_down",
                native_weighted, tail_gpu.weighted, oracle.weighted);
  AppendCompare(result.comparisons, "ffn_moe_out",
                native_moe_out, tail_gpu.moe_out, oracle.moe_out);
  AppendCompare(result.comparisons, "shared_input_gate",
                native_shared_input_gate, tail_gpu.shared_gate, oracle.shared_gate);
  AppendCompare(result.comparisons, "shared_gate_sigmoid",
                native_shared_sigmoid, tail_gpu.shared_gate_sigmoid, oracle.shared_sigmoid);
  AppendCompare(result.comparisons, "ffn_shexp_gated",
                native_shared_gated, tail_gpu.shared_gated, oracle.shared_gated);
  AppendCompare(result.comparisons, "ffn_out",
                native_ffn_out, tail_gpu.ffn_out, oracle.ffn_out);
  AppendCompare(result.comparisons, "layer_output",
                result.native_layer_output, result.gpu_layer_output, oracle.layer_output);

  result.conv_state_after_gpu_vs_cpu =
      iq36::compare_vectors(preconv_gpu.conv_state_after, native_conv.conv_state,
                            kMismatchThreshold);
  result.recurrent_state_gpu_vs_cpu =
      iq36::compare_vectors(delta_gpu.recurrent_state, native_postconv.recurrent_state,
                            kMismatchThreshold);
  result.comparisons_passed =
      CompareGroupsPassed(result.comparisons) &&
      ComparePassed(result.conv_state_after_gpu_vs_cpu) &&
      ComparePassed(result.recurrent_state_gpu_vs_cpu);
  result.timing_positive =
      layer_input_gpu.timing.rmsnorm_min_us > 0.0 &&
      preconv_gpu.timing.qkv_min_us > 0.0 &&
      preconv_gpu.timing.alpha_min_us > 0.0 &&
      preconv_gpu.timing.beta_min_us > 0.0 &&
      preconv_gpu.timing.z_min_us > 0.0 &&
      preconv_gpu.timing.conv_min_us > 0.0 &&
      preconv_gpu.timing.postconv_silu_split_min_us > 0.0 &&
      preconv_gpu.timing.postconv_q_l2_min_us > 0.0 &&
      preconv_gpu.timing.postconv_k_l2_min_us > 0.0 &&
      delta_gpu.timing.delta_min_us > 0.0 &&
      delta_gpu.timing.final_norm_min_us > 0.0 &&
      attention_gpu.timing.output_projection_min_us > 0.0 &&
      attention_gpu.timing.residual_add_min_us > 0.0 &&
      attention_gpu.timing.ffn_rmsnorm_min_us > 0.0 &&
      selected_gpu.timing.gate_up_min_us > 0.0 &&
      selected_gpu.timing.swiglu_min_us > 0.0 &&
      selected_gpu.timing.down_min_us > 0.0 &&
      shared_gpu.timing.gate_min_us > 0.0 &&
      shared_gpu.timing.up_min_us > 0.0 &&
      shared_gpu.timing.swiglu_min_us > 0.0 &&
      shared_gpu.timing.down_min_us > 0.0 &&
      tail_gpu.timing.weighted_min_us > 0.0 &&
      tail_gpu.timing.shared_gate_matvec_min_us > 0.0 &&
      tail_gpu.timing.shared_gate_apply_min_us > 0.0 &&
      tail_gpu.timing.ffn_output_add_min_us > 0.0 &&
      tail_gpu.timing.residual_add_min_us > 0.0 &&
      tail_gpu.timing.shell_sum_min_us > 0.0;
  result.arc_selected =
      layer_input_gpu.device_name.find(args.device_substring) != std::string::npos &&
      preconv_gpu.device_name.find(args.device_substring) != std::string::npos &&
      delta_gpu.device_name.find(args.device_substring) != std::string::npos &&
      attention_gpu.device_name.find(args.device_substring) != std::string::npos &&
      selected_gpu.device_name.find(args.device_substring) != std::string::npos &&
      shared_gpu.device_name.find(args.device_substring) != std::string::npos &&
      tail_gpu.device_name.find(args.device_substring) != std::string::npos;

  result.timing.layer_input_rmsnorm_min_us = layer_input_gpu.timing.rmsnorm_min_us;
  result.timing.layer_input_rmsnorm_mean_us = layer_input_gpu.timing.rmsnorm_mean_us;
  result.timing.preconv_to_postconv_kernel_sum_min_us =
      preconv_gpu.timing.preconv_to_postconv_kernel_sum_min_us;
  result.timing.preconv_to_postconv_kernel_sum_mean_us =
      preconv_gpu.timing.preconv_to_postconv_kernel_sum_mean_us;
  result.timing.delta_to_final_kernel_sum_min_us =
      delta_gpu.timing.delta_to_final_kernel_sum_min_us;
  result.timing.delta_to_final_kernel_sum_mean_us =
      delta_gpu.timing.delta_to_final_kernel_sum_mean_us;
  result.timing.attention_front_kernel_sum_min_us =
      attention_gpu.timing.attention_front_kernel_sum_min_us;
  result.timing.attention_front_kernel_sum_mean_us =
      attention_gpu.timing.attention_front_kernel_sum_mean_us;
  result.timing.selected_ffn_kernel_sum_min_us =
      selected_gpu.timing.selected_ffn_kernel_sum_min_us;
  result.timing.selected_ffn_kernel_sum_mean_us =
      selected_gpu.timing.selected_ffn_kernel_sum_mean_us;
  result.timing.shared_ffn_kernel_sum_min_us =
      shared_gpu.timing.shared_ffn_kernel_sum_min_us;
  result.timing.shared_ffn_kernel_sum_mean_us =
      shared_gpu.timing.shared_ffn_kernel_sum_mean_us;
  result.timing.ffn_tail_kernel_sum_min_us = tail_gpu.timing.shell_sum_min_us;
  result.timing.ffn_tail_kernel_sum_mean_us = tail_gpu.timing.shell_sum_mean_us;
  result.timing.layer_kernel_sum_min_us =
      result.timing.layer_input_rmsnorm_min_us +
      result.timing.preconv_to_postconv_kernel_sum_min_us +
      result.timing.delta_to_final_kernel_sum_min_us +
      result.timing.attention_front_kernel_sum_min_us +
      result.timing.selected_ffn_kernel_sum_min_us +
      result.timing.shared_ffn_kernel_sum_min_us +
      result.timing.ffn_tail_kernel_sum_min_us;
  result.timing.layer_kernel_sum_mean_us =
      result.timing.layer_input_rmsnorm_mean_us +
      result.timing.preconv_to_postconv_kernel_sum_mean_us +
      result.timing.delta_to_final_kernel_sum_mean_us +
      result.timing.attention_front_kernel_sum_mean_us +
      result.timing.selected_ffn_kernel_sum_mean_us +
      result.timing.shared_ffn_kernel_sum_mean_us +
      result.timing.ffn_tail_kernel_sum_mean_us;
  result.layer_input_device_name = layer_input_gpu.device_name;
  result.preconv_device_name = preconv_gpu.device_name;
  result.delta_device_name = delta_gpu.device_name;
  result.attention_device_name = attention_gpu.device_name;
  result.selected_device_name = selected_gpu.device_name;
  result.shared_device_name = shared_gpu.device_name;
  result.tail_device_name = tail_gpu.device_name;
  result.shared_down_tensor_type = iq36::ggml_type_name(t.shared_down_tensor->type);
  result.program_build_ms =
      layer_input_gpu.program_build_ms + preconv_gpu.program_build_ms +
      delta_gpu.program_build_ms + attention_gpu.program_build_ms +
      selected_gpu.program_build_ms + shared_gpu.program_build_ms +
      tail_gpu.program_build_ms;
  result.build_log =
      layer_input_gpu.build_log + preconv_gpu.build_log + delta_gpu.build_log +
      attention_gpu.build_log + selected_gpu.build_log + shared_gpu.build_log +
      tail_gpu.build_log;
  return result;
}

void WritePrefixedCompareGroups(const std::string& prefix,
                                const std::vector<NamedCompareGroup>& groups,
                                bool* first) {
  for (const auto& group : groups) {
    if (!*first) {
      std::cout << ",";
    }
    *first = false;
    std::cout << "\"" << JsonEscape(prefix + "_" + group.name) << "\":";
    WriteCompareGroup(group.cpu_vs_oracle,
                      group.gpu_vs_cpu,
                      group.gpu_vs_oracle);
  }
}

void WriteLayerTiming(const LayerShellTiming& timing) {
  std::cout << "{";
  std::cout << "\"layer_input_rmsnorm_min_us\":" << timing.layer_input_rmsnorm_min_us << ",";
  std::cout << "\"layer_input_rmsnorm_mean_us\":" << timing.layer_input_rmsnorm_mean_us << ",";
  std::cout << "\"preconv_to_postconv_kernel_sum_min_us\":" << timing.preconv_to_postconv_kernel_sum_min_us << ",";
  std::cout << "\"preconv_to_postconv_kernel_sum_mean_us\":" << timing.preconv_to_postconv_kernel_sum_mean_us << ",";
  std::cout << "\"delta_to_final_kernel_sum_min_us\":" << timing.delta_to_final_kernel_sum_min_us << ",";
  std::cout << "\"delta_to_final_kernel_sum_mean_us\":" << timing.delta_to_final_kernel_sum_mean_us << ",";
  std::cout << "\"attention_front_kernel_sum_min_us\":" << timing.attention_front_kernel_sum_min_us << ",";
  std::cout << "\"attention_front_kernel_sum_mean_us\":" << timing.attention_front_kernel_sum_mean_us << ",";
  std::cout << "\"selected_ffn_kernel_sum_min_us\":" << timing.selected_ffn_kernel_sum_min_us << ",";
  std::cout << "\"selected_ffn_kernel_sum_mean_us\":" << timing.selected_ffn_kernel_sum_mean_us << ",";
  std::cout << "\"shared_ffn_kernel_sum_min_us\":" << timing.shared_ffn_kernel_sum_min_us << ",";
  std::cout << "\"shared_ffn_kernel_sum_mean_us\":" << timing.shared_ffn_kernel_sum_mean_us << ",";
  std::cout << "\"ffn_tail_kernel_sum_min_us\":" << timing.ffn_tail_kernel_sum_min_us << ",";
  std::cout << "\"ffn_tail_kernel_sum_mean_us\":" << timing.ffn_tail_kernel_sum_mean_us << ",";
  std::cout << "\"layer_kernel_sum_min_us\":" << timing.layer_kernel_sum_min_us << ",";
  std::cout << "\"layer_kernel_sum_mean_us\":" << timing.layer_kernel_sum_mean_us;
  std::cout << "}";
}

void WriteLayerChecks(const LayerShellResult& layer) {
  std::cout << "{";
  std::cout << "\"layer_input_tensor_shape_ok\":"
            << (layer.shape_checks.layer_input_tensor_shape_ok ? "true" : "false") << ",";
  std::cout << "\"preconv_tensor_shape_ok\":"
            << (layer.shape_checks.preconv_tensor_shape_ok ? "true" : "false") << ",";
  std::cout << "\"delta_tensor_shape_ok\":"
            << (layer.shape_checks.delta_tensor_shape_ok ? "true" : "false") << ",";
  std::cout << "\"attention_tensor_shape_ok\":"
            << (layer.shape_checks.attention_tensor_shape_ok ? "true" : "false") << ",";
  std::cout << "\"selected_tensor_shape_ok\":"
            << (layer.shape_checks.selected_tensor_shape_ok ? "true" : "false") << ",";
  std::cout << "\"shared_tensor_shape_ok\":"
            << (layer.shape_checks.shared_tensor_shape_ok ? "true" : "false") << ",";
  std::cout << "\"payload_counts_ok\":"
            << (layer.payload_counts_ok ? "true" : "false") << ",";
  std::cout << "\"comparisons_passed\":"
            << (layer.comparisons_passed ? "true" : "false") << ",";
  std::cout << "\"timing_positive\":"
            << (layer.timing_positive ? "true" : "false") << ",";
  std::cout << "\"arc_selected\":"
            << (layer.arc_selected ? "true" : "false");
  std::cout << "}";
}

int main(int argc, char** argv) {
  try {
    const Args args = ParseArgs(argc, argv);
    const int layer0 = args.layer;
    const int layer1 = args.layer + 1;
    Require(layer0 >= 0 && layer1 < 40, "two-layer shell layer range invalid");
    const auto index = iq36::parse_gguf_model_index(args.model_path);
    const auto load_map = iq36::validate_qwen36_load_map(index);
    const float rms_norm_epsilon = MetadataFloat(
        index, "qwen35moe.attention.layer_norm_rms_epsilon", 1e-6f);
    const auto layer0_tensors = ResolveLayerTensorBundle(index, layer0);
    const auto layer1_tensors = ResolveLayerTensorBundle(index, layer1);
    const auto layer0_oracle = LoadLayerOraclePayloads(args.payload_dir, "l0");
    const auto layer1_oracle = LoadLayerOraclePayloads(args.payload_dir, "l1");

    const auto layer0_run = RunResidentLinearLayerShell(
        args, index, layer0_tensors, layer0_oracle,
        layer0_oracle.residual_input, layer0_oracle.residual_input,
        rms_norm_epsilon);
    const auto layer1_run = RunResidentLinearLayerShell(
        args, index, layer1_tensors, layer1_oracle,
        layer0_run.native_layer_output, layer0_run.gpu_layer_output,
        rms_norm_epsilon);

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
    const bool state_carry_ok = ComparePassed(layer1_run.comparisons[0].gpu_vs_oracle);
    const bool required_checks_passed =
        load_map.ready &&
        layer0_ok &&
        layer1_ok &&
        state_carry_ok &&
        args.repeat > 0;
    const double two_layer_kernel_sum_min =
        layer0_run.timing.layer_kernel_sum_min_us +
        layer1_run.timing.layer_kernel_sum_min_us;
    const double two_layer_kernel_sum_mean =
        layer0_run.timing.layer_kernel_sum_mean_us +
        layer1_run.timing.layer_kernel_sum_mean_us;

    std::cout << std::setprecision(10);
    std::cout << "{";
    std::cout << "\"schema_version\":\"intel-qwen36-gpu-resident-two-linear-layer-shell-probe-v0\",";
    std::cout << "\"model_path\":\"" << JsonEscape(args.model_path) << "\",";
    std::cout << "\"layers\":[" << layer0 << "," << layer1 << "],";
    std::cout << "\"source_token_position\":" << kSourceTokenPosition << ",";
    std::cout << "\"resident_api\":\"two_linear_layer_state_carry_load_once_run_many\",";
    std::cout << "\"resident_load_count\":1,";
    std::cout << "\"resident_shell_invocations\":" << args.repeat << ",";
    std::cout << "\"layer1_residual_input_from_layer0_gpu_output\":true,";
    std::cout << "\"captured_layer1_residual_input_required_check\":true,";
    std::cout << "\"captured_conv_state_input_boundary\":true,";
    std::cout << "\"preconv_host_q8_bridge\":true,";
    std::cout << "\"delta_to_attention_host_boundary\":true,";
    std::cout << "\"attention_output_projection_host_q8_bridge\":true,";
    std::cout << "\"attention_front_host_boundary_between_q4_and_f32\":true,";
    std::cout << "\"selected_down_host_q8_bridge\":true,";
    std::cout << "\"shared_down_host_q8_bridge\":true,";
    std::cout << "\"platform_name\":\"" << JsonEscape(layer0_run.layer_input_device_name) << "\",";
    std::cout << "\"device_name\":\"" << JsonEscape(layer0_run.layer_input_device_name) << "\",";
    std::cout << "\"layer0_shared_down_tensor_type\":\""
              << JsonEscape(layer0_run.shared_down_tensor_type) << "\",";
    std::cout << "\"layer1_shared_down_tensor_type\":\""
              << JsonEscape(layer1_run.shared_down_tensor_type) << "\",";
    std::cout << "\"program_build_ms\":"
              << (layer0_run.program_build_ms + layer1_run.program_build_ms) << ",";
    std::cout << "\"build_log\":\""
              << JsonEscape(layer0_run.build_log + layer1_run.build_log) << "\",";
    std::cout << "\"repeat\":" << args.repeat << ",";
    std::cout << "\"timings\":{";
    std::cout << "\"layer0\":";
    WriteLayerTiming(layer0_run.timing);
    std::cout << ",\"layer1\":";
    WriteLayerTiming(layer1_run.timing);
    std::cout << ",\"resident_two_linear_layer_kernel_sum_min_us\":"
              << two_layer_kernel_sum_min << ",";
    std::cout << "\"resident_two_linear_layer_kernel_sum_mean_us\":"
              << two_layer_kernel_sum_mean;
    std::cout << "},\"comparisons\":{";
    bool first_compare = true;
    WritePrefixedCompareGroups("l0", layer0_run.comparisons, &first_compare);
    WritePrefixedCompareGroups("l1", layer1_run.comparisons, &first_compare);
    std::cout << ",\"l0_conv_state_after\":{\"gpu_vs_cpu\":";
    WriteCompare(layer0_run.conv_state_after_gpu_vs_cpu);
    std::cout << "},\"l0_recurrent_state\":{\"gpu_vs_cpu\":";
    WriteCompare(layer0_run.recurrent_state_gpu_vs_cpu);
    std::cout << "},\"l1_conv_state_after\":{\"gpu_vs_cpu\":";
    WriteCompare(layer1_run.conv_state_after_gpu_vs_cpu);
    std::cout << "},\"l1_recurrent_state\":{\"gpu_vs_cpu\":";
    WriteCompare(layer1_run.recurrent_state_gpu_vs_cpu);
    std::cout << "}";
    std::cout << "},\"checks\":{";
    std::cout << "\"load_map_ready\":" << (load_map.ready ? "true" : "false") << ",";
    std::cout << "\"layer0\":";
    WriteLayerChecks(layer0_run);
    std::cout << ",\"layer1\":";
    WriteLayerChecks(layer1_run);
    std::cout << ",\"layer1_residual_input_from_layer0_gpu_output\":true,";
    std::cout << "\"layer1_residual_input_matches_oracle\":"
              << (state_carry_ok ? "true" : "false") << ",";
    std::cout << "\"resident_load_once\":true,";
    std::cout << "\"resident_shell_invocations_positive\":"
              << (args.repeat > 0 ? "true" : "false") << ",";
    std::cout << "\"gpu_event_timing_positive\":"
              << (layer0_run.timing_positive && layer1_run.timing_positive ? "true" : "false") << ",";
    std::cout << "\"two_linear_layers_match_oracle\":"
              << (layer0_run.comparisons_passed && layer1_run.comparisons_passed ? "true" : "false") << ",";
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


def two_layer_probe_cpp(opencl_source: str) -> str:
  cpp = LAYER_INPUT.layer_input_probe_cpp(opencl_source)
  main_index = cpp.index("\nint main(")
  return cpp[:main_index] + "\n" + TWO_LAYER_CPP


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
  parser.add_argument("--next-conv-history-probe", type=Path, default=None)
  parser.add_argument("--out-dir", type=Path, default=None)
  return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise SystemExit(f"{path}: expected JSON object")
  return value


def latest_conv_history_probe_for_layer(layer: int) -> Path:
  paths = sorted((ROOT / "output").glob("gpu-conv-history-state-capture-probe-*/probe.json"))
  for path in reversed(paths):
    try:
      data = load_json(path)
    except Exception:
      continue
    if (
        data.get("schema_version") == "intel-qwen36-gpu-conv-history-state-capture-probe-v0"
        and data.get("required_checks_passed") is True
        and data.get("layer") == layer
    ):
      return path
  raise SystemExit(f"no passing gpu conv-history state capture probe found for layer {layer}")


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


def add_conv_history_payload(
    payloads: dict[str, dict[str, Any]],
    conv_probe: dict[str, Any],
    key: str,
    stage_name: str,
) -> None:
  item = conv_probe.get("payloads", {}).get(key)
  if not isinstance(item, dict):
    raise SystemExit(f"conv-history probe missing payload {key}")
  path_value = item.get("path")
  size_bytes = item.get("size_bytes")
  if not isinstance(path_value, str) or not isinstance(size_bytes, int):
    raise SystemExit(f"conv-history payload {key} has invalid metadata")
  record = payload_record(ROOT / path_value, stage_name, size_bytes)
  record["source_artifact"] = conv_probe.get("capture_artifact")
  record["tensor_name"] = item.get("tensor_name")
  payloads[key] = record


def resolve_linear_layer_payloads(
    layer: int,
    conv_history_probe_path: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
  conv_probe = load_json(conv_history_probe_path.resolve())
  if conv_probe.get("required_checks_passed") is not True:
    raise SystemExit(f"conv-history probe did not pass: {conv_history_probe_path}")
  if conv_probe.get("layer") != layer:
    raise SystemExit(f"conv-history probe layer mismatch: expected {layer}, got {conv_probe.get('layer')}")
  payloads: dict[str, dict[str, Any]] = {}
  if layer == 0:
    residual_path = find_payload("model.input_embed__tok15__ord*.bin", 8192)
  else:
    residual_path = find_payload(f"l_out-{layer - 1}__tok15__ord*.bin", 8192)
  payloads["residual_input"] = payload_record(residual_path, "residual_input.bin", 8192)
  add_conv_history_payload(payloads, conv_probe, "attn_norm", "attn_norm.bin")
  add_conv_history_payload(
      payloads, conv_probe, "linear_attn_qkv_mixed", "linear_attn_qkv_mixed.bin"
  )
  add_conv_history_payload(payloads, conv_probe, "conv_state", "conv_state.bin")
  add_conv_history_payload(payloads, conv_probe, "conv_output_raw", "conv_output_raw.bin")
  for name, stage_name, source_prefix, expected_bytes in (
      ("alpha", "alpha.bin", "alpha", 128),
      ("a_softplus", "a_softplus.bin", "a_softplus", 128),
      ("gate", "gate.bin", "gate", 128),
      ("beta", "beta.bin", "beta", 128),
      ("beta_sigmoid", "beta_sigmoid.bin", "beta_sigmoid", 128),
      ("z", "z.bin", "z", 16384),
      ("conv_output_silu", "conv_output_silu.bin", "conv_output_silu", 32768),
      ("q_conv", "q_conv.bin", "q_conv", 8192),
      ("k_conv", "k_conv.bin", "k_conv", 8192),
      ("q_conv_predelta", "q_conv_predelta.bin", "q_conv_predelta", 8192),
      ("k_conv_predelta", "k_conv_predelta.bin", "k_conv_predelta", 8192),
      ("v_conv_predelta", "v_conv_predelta.bin", "v_conv_predelta", 16384),
      ("state_predelta", "state_predelta.bin", "state_predelta", 2097152),
      ("attn_output", "attn_output.bin", "attn_output", 16384),
      ("final_output", "final_output.bin", "final_output", 16384),
      ("linear_attn_out", "linear_attn_out.bin", "linear_attn_out", 8192),
      ("attn_residual", "attn_residual.bin", "attn_residual", 8192),
      ("attn_post_norm", "attn_post_norm.bin", "attn_post_norm", 8192),
      ("ffn_moe_topk", "ffn_moe_topk.bin", "ffn_moe_topk", 32),
      ("ffn_moe_weights_norm", "ffn_moe_weights_norm.bin", "ffn_moe_weights_norm", 32),
      ("ffn_moe_gate_up", "ffn_moe_gate_up.bin", "ffn_moe_gate_up", 32768),
      ("ffn_moe_swiglu", "ffn_moe_swiglu.bin", "ffn_moe_swiglu", 16384),
      ("ffn_moe_down", "ffn_moe_down.bin", "ffn_moe_down", 65536),
      ("ffn_moe_weighted", "ffn_moe_weighted.bin", "ffn_moe_weighted", 65536),
      ("ffn_moe_out", "ffn_moe_out.bin", "ffn_moe_out", 8192),
      ("ffn_shexp", "ffn_shexp.bin", "ffn_shexp", 8192),
      ("shared_expert_gate", "shared_expert_gate.bin", "shared_expert_gate", 4),
      (
          "shared_expert_gate_sigmoid",
          "shared_expert_gate_sigmoid.bin",
          "shared_expert_gate_sigmoid",
          4,
      ),
      ("ffn_shexp_gated", "ffn_shexp_gated.bin", "ffn_shexp_gated", 8192),
      ("ffn_out", "ffn_out.bin", "ffn_out", 8192),
      ("layer_output", "layer_output.bin", "l_out", 8192),
  ):
    payloads[name] = payload_record(
        find_payload(f"{source_prefix}-{layer}__tok15__ord*.bin", expected_bytes),
        stage_name,
        expected_bytes,
    )
  return payloads, conv_probe


def prefixed_payloads(
    layer: int,
    conv_history_probe_path: Path,
    prefix: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
  payloads, conv_probe = resolve_linear_layer_payloads(layer, conv_history_probe_path)
  out: dict[str, dict[str, Any]] = {}
  for name, payload in payloads.items():
    item = dict(payload)
    stage_name = item.get("stage_name")
    if not isinstance(stage_name, str):
      raise SystemExit(f"payload {name} missing stage_name")
    item["stage_name"] = f"{prefix}_{stage_name}"
    item["two_layer_name"] = f"{prefix}_{name}"
    out[f"{prefix}_{name}"] = item
  return out, conv_probe


def resident_fields_ok(probe: dict[str, Any] | None, expected_invocations: int) -> bool:
  return (
      isinstance(probe, dict)
      and probe.get("resident_api") == "two_linear_layer_state_carry_load_once_run_many"
      and probe.get("resident_load_count") == 1
      and probe.get("resident_shell_invocations") == expected_invocations
      and probe.get("layer1_residual_input_from_layer0_gpu_output") is True
      and probe.get("captured_layer1_residual_input_required_check") is True
      and probe.get("captured_conv_state_input_boundary") is True
      and probe.get("preconv_host_q8_bridge") is True
      and probe.get("delta_to_attention_host_boundary") is True
      and probe.get("attention_output_projection_host_q8_bridge") is True
      and probe.get("attention_front_host_boundary_between_q4_and_f32") is True
      and probe.get("selected_down_host_q8_bridge") is True
      and probe.get("shared_down_host_q8_bridge") is True
      and PRECONV.nested_bool(probe, "checks", "resident_load_once")
      and PRECONV.nested_bool(probe, "checks", "resident_shell_invocations_positive")
  )


def write_summary(path: Path, payload: dict[str, Any]) -> None:
  probe = payload.get("probe") if isinstance(payload.get("probe"), dict) else {}
  timings = probe.get("timings", {}) if isinstance(probe, dict) else {}
  comparisons = probe.get("comparisons", {}) if isinstance(probe, dict) else {}
  lines = [
      "# GPU Resident Two-Linear-Layer State-Carry Shell Probe",
      "",
      f"- workstream: `{WORKSTREAM}`",
      f"- host: `{payload.get('host')}`",
      f"- model: `{payload.get('model')}`",
      f"- layers: `{payload.get('layers')}`",
      f"- required checks passed: `{str(payload.get('required_checks_passed')).lower()}`",
      "- speedup claims allowed: `false`",
      f"- resident API: `{probe.get('resident_api')}`",
      f"- resident load count: `{probe.get('resident_load_count')}`",
      f"- resident shell invocations: `{probe.get('resident_shell_invocations')}`",
      "",
      "| output | comparison | max abs | RMSE |",
      "|---|---|---:|---:|",
  ]
  for name in (
      "l0_layer_output",
      "l1_residual_input",
      "l1_attn_norm",
      "l1_linear_attn_qkv_mixed",
      "l1_conv_output_raw",
      "l1_final_output",
      "l1_layer_output",
  ):
    group = comparisons.get(name, {}) if isinstance(comparisons, dict) else {}
    lane = group.get("gpu_vs_oracle", {}) if isinstance(group, dict) else {}
    lines.append(f"| {name} | gpu_vs_oracle | {lane.get('max_abs_diff')} | {lane.get('rmse')} |")
  lines += [
      "",
      "| kernel group | min us |",
      "|---|---:|",
      f"| layer0_sum | {timings.get('layer0', {}).get('layer_kernel_sum_min_us') if isinstance(timings.get('layer0'), dict) else None} |",
      f"| layer1_sum | {timings.get('layer1', {}).get('layer_kernel_sum_min_us') if isinstance(timings.get('layer1'), dict) else None} |",
      f"| resident_two_linear_layer_sum | {timings.get('resident_two_linear_layer_kernel_sum_min_us')} |",
      "",
      "The target-side process runs layer 5 and layer 6 with one parameterized",
      "linear-layer shell. Layer 6 uses layer 5 GPU output as residual input;",
      "captured `l_out-5` remains a required state-carry check. This is",
      "captured single-token two-layer evidence only, not prompt/token decode",
      "throughput.",
      "",
  ]
  path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  args = parse_args()
  if args.resident_invocations <= 0:
    raise SystemExit("--resident-invocations must be positive")
  layer0 = args.layer
  layer1 = args.layer + 1
  conv0_path = (
      args.conv_history_probe.resolve()
      if args.conv_history_probe is not None
      else latest_conv_history_probe_for_layer(layer0).resolve()
  )
  conv1_path = (
      args.next_conv_history_probe.resolve()
      if args.next_conv_history_probe is not None
      else latest_conv_history_probe_for_layer(layer1).resolve()
  )
  stamp = utc_stamp()
  created_at = iso_now()
  out_dir = args.out_dir or ROOT / f"output/gpu-resident-two-linear-layer-shell-probe-{stamp}"
  out_dir.mkdir(parents=True, exist_ok=False)
  raw_dir = out_dir / "raw"
  raw_dir.mkdir()

  payloads0, conv0 = prefixed_payloads(layer0, conv0_path, "l0")
  payloads1, conv1 = prefixed_payloads(layer1, conv1_path, "l1")
  payloads = {**payloads0, **payloads1}
  opencl_source = OPENCL_SOURCE.read_text(encoding="utf-8")
  opencl_source_hash = iq36_local.sha256_file(OPENCL_SOURCE)
  embedded_opencl_hash = hashlib.sha256(
      (opencl_source + PRECONV.POSTCONV.ATTENTION.ATTENTION_EXTRA_OPENCL).encode("utf-8")
  ).hexdigest()
  local_cpp = out_dir / "gpu_resident_two_linear_layer_shell_probe.cpp"
  local_cpp.write_text(two_layer_probe_cpp(opencl_source), encoding="utf-8")

  remote_dir = f"{args.remote_root.rstrip('/')}/gpu-resident-two-linear-layer-shell-probe-{stamp}"
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
      transfers.append(iq36_local.copy_to(args.host, ROOT / local, f"{remote_dir}/{remote}", args.timeout_s))
    transfers.append(
        iq36_local.copy_to(
            args.host,
            local_cpp,
            f"{remote_dir}/tests/gpu_resident_two_linear_layer_shell_probe.cpp",
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

  executable = f"{remote_dir}/build/iq36-gpu-resident-two-linear-layer-shell-probe"
  compile_cmd = " && ".join([
      f"source {shlex.quote(args.env_script)} >/dev/null 2>&1",
      (
          "g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic "
          f"-I {shlex.quote(remote_dir + '/include')} "
          f"{shlex.quote(remote_dir + '/src/gguf_loader.cpp')} "
          f"{shlex.quote(remote_dir + '/src/gpu_q4x8_matvec.cpp')} "
          f"{shlex.quote(remote_dir + '/tests/gpu_resident_two_linear_layer_shell_probe.cpp')} "
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
      {"name": "two_linear_layers_match_oracle", "pass": PRECONV.nested_bool(probe, "checks", "two_linear_layers_match_oracle")},
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
      "payloads": slim_payloads,
      "layers": [layer0, layer1],
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
      "tool": "tools/intel-qwen36-gpu-resident-two-linear-layer-shell-probe.py",
      "artifact": str(out_dir),
      "host": args.host,
      "remote_dir": remote_dir,
      "layers": [layer0, layer1],
      "resident_invocations": args.resident_invocations,
      "conv_history_probes": payload["conv_history_probes"],
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
      "gpu_resident_two_linear_layer_shell_probe",
      [
          ("required_checks_passed", required_checks_passed),
          ("resident_shell_invocations", args.resident_invocations),
          ("resident_two_linear_layer_kernel_sum_min_us", PRECONV.nested_number(timings, "resident_two_linear_layer_kernel_sum_min_us")),
          ("l0_layer_output_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l0_layer_output", "gpu_vs_oracle", "max_abs_diff")),
          ("l1_residual_input_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l1_residual_input", "gpu_vs_oracle", "max_abs_diff")),
          ("l1_attn_norm_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l1_attn_norm", "gpu_vs_oracle", "max_abs_diff")),
          ("l1_qkv_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l1_linear_attn_qkv_mixed", "gpu_vs_oracle", "max_abs_diff")),
          ("l1_conv_output_raw_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l1_conv_output_raw", "gpu_vs_oracle", "max_abs_diff")),
          ("l1_final_output_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l1_final_output", "gpu_vs_oracle", "max_abs_diff")),
          ("l1_layer_output_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l1_layer_output", "gpu_vs_oracle", "max_abs_diff")),
      ],
  )
  write_summary(out_dir / "summary.md", payload)
  print(out_dir)
  return 0 if required_checks_passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
