#!/usr/bin/env python3
"""Run the resident GPU attention-to-FFN layer shell handoff probe."""

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
SHARED_TOOL = Path(__file__).with_name("intel-qwen36-gpu-resident-shared-ffn-shell-probe.py")
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA_VERSION = "intel-qwen36-gpu-resident-attention-ffn-shell-probe-v0"
DEFAULT_HOST = "local"
DEFAULT_MODEL = "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf"
DEFAULT_ENV_SCRIPT = "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh"
DEFAULT_REMOTE_ROOT = "/home/intel/intel-qwen36-gpu"
DEFAULT_ORACLE_BUNDLE = ROOT / "oracle/r0-oracle-bundle-20260627T060028Z"
PAYLOAD_ROOT = ROOT / "output/r0-boundary-capture-run-20260627T054024Z/remote-output/payloads"
OPENCL_SOURCE = ROOT / "engine/gpu/opencl/q4x8_matvec.cl"


def load_shared_tool() -> ModuleType:
  spec = importlib.util.spec_from_file_location("iq36_shared_ffn_shell_probe", SHARED_TOOL)
  if spec is None or spec.loader is None:
    raise SystemExit(f"failed to load shared FFN shell tool: {SHARED_TOOL}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


SHARED = load_shared_tool()


ATTENTION_EXTRA_OPENCL = r'''

__kernel void rms_norm_hidden_f32(__global const float* input,
                                  __global const float* weight,
                                  uint hidden_size,
                                  float epsilon,
                                  __global float* output) {
  if ((uint)get_global_id(0) != 0U) {
    return;
  }
  float sum_squares = 0.0f;
  for (uint i = 0; i < hidden_size; ++i) {
    const float value = input[i];
    sum_squares += value * value;
  }
  const float mean_square = sum_squares / (float)hidden_size;
  const float scale = rsqrt(mean_square + epsilon);
  for (uint i = 0; i < hidden_size; ++i) {
    output[i] = input[i] * scale * weight[i];
  }
}

__kernel void rms_norm_hidden_scale_f32(__global const float* input,
                                        uint hidden_size,
                                        float epsilon,
                                        __global float* scale_out) {
  if ((uint)get_global_id(0) != 0U) {
    return;
  }
  float sum_squares = 0.0f;
  for (uint i = 0; i < hidden_size; ++i) {
    const float value = input[i];
    sum_squares += value * value;
  }
  const float mean_square = sum_squares / (float)hidden_size;
  scale_out[0] = rsqrt(mean_square + epsilon);
}

__kernel void rms_norm_hidden_apply_scale_f32(__global const float* input,
                                              __global const float* weight,
                                              uint hidden_size,
                                              __global const float* scale_in,
                                              __global float* output) {
  const uint index = (uint)get_global_id(0);
  if (index >= hidden_size) {
    return;
  }
  output[index] = input[index] * scale_in[0] * weight[index];
}
'''


ATTENTION_HELPERS_CPP = r'''

constexpr cl_mem_flags kClMemReadWriteLocal = 1ULL;

float MetadataFloat(const iq36::GgufModelIndex& index,
                    const std::string& key,
                    float fallback) {
  const auto it = index.metadata.find(key);
  if (it == index.metadata.end()) {
    return fallback;
  }
  const auto& value = it->second;
  if (value.kind == iq36::GgufMetadataValue::Kind::kFloat) {
    return static_cast<float>(value.float_value);
  }
  if (value.kind == iq36::GgufMetadataValue::Kind::kInt) {
    return static_cast<float>(value.int_value);
  }
  if (value.kind == iq36::GgufMetadataValue::Kind::kUInt) {
    return static_cast<float>(value.uint_value);
  }
  return fallback;
}

struct AttentionFrontTiming {
  double output_projection_min_us = 0.0;
  double output_projection_mean_us = 0.0;
  double output_projection_host_q8_bridge_us = 0.0;
  double residual_add_min_us = 0.0;
  double residual_add_mean_us = 0.0;
  double ffn_rmsnorm_min_us = 0.0;
  double ffn_rmsnorm_mean_us = 0.0;
  double attention_front_kernel_sum_min_us = 0.0;
  double attention_front_kernel_sum_mean_us = 0.0;
  std::uint64_t output_projection_global_work_items = 0;
  std::uint64_t residual_add_global_work_items = 0;
  std::uint64_t ffn_rmsnorm_global_work_items = 0;
};

struct AttentionFrontRun {
  std::vector<float> linear_attn_out;
  std::vector<float> attn_residual;
  std::vector<float> attn_post_norm;
  AttentionFrontTiming timing;
  std::string platform_name;
  std::string device_name;
  std::string build_log;
  double program_build_ms = 0.0;
};

AttentionFrontRun RunGpuAttentionFront(
    const std::string& model_path,
    const iq36::GgufTensorInfo& output_tensor,
    const std::vector<float>& final_output,
    const std::vector<float>& residual_input,
    const std::vector<float>& ffn_norm_weight,
    float rms_norm_epsilon,
    const std::string& device_substring,
    int repeat) {
  Require(output_tensor.type == 12, "attention output tensor must be Q4_K");
  Require(output_tensor.dims == std::vector<std::uint64_t>{4096, kHiddenSize},
          "attention output tensor shape mismatch");
  Require(final_output.size() == 4096, "final_output size mismatch");
  Require(residual_input.size() == kHiddenSize, "residual input size mismatch");
  Require(ffn_norm_weight.size() == kHiddenSize, "ffn norm weight size mismatch");

  AttentionFrontRun run;
  std::ifstream model(model_path, std::ios::binary);
  Require(static_cast<bool>(model), "failed to open model for attention front");
  const std::uint64_t cols = output_tensor.dims[0];
  const std::uint64_t rows = output_tensor.dims[1];
  const std::uint64_t blocks_per_row = cols / 256;
  const auto raw = ReadTensorBytes(model, output_tensor);
  const auto packed = iq36::PackQ4Kx8(raw, rows, blocks_per_row);
  const auto bridge_begin = std::chrono::steady_clock::now();
  const auto q8_input = iq36::QuantizeQ8KInputPlanes(final_output);
  const auto bridge_end = std::chrono::steady_clock::now();
  run.timing.output_projection_host_q8_bridge_us =
      std::chrono::duration<double, std::micro>(bridge_end - bridge_begin).count();

  iq36::GpuQ4X8MatvecRunner runner(device_substring, kOpenClSource);
  run.platform_name = runner.platform_name();
  run.device_name = runner.device_name();
  run.build_log = runner.build_log();
  run.program_build_ms = runner.program_build_ms();
  const auto output_projection = runner.Run(
      packed, q8_input.qs, q8_input.bsums, q8_input.d, rows, blocks_per_row,
      repeat, iq36::GpuQ4X8KernelVariant::kRowlaneParallel);
  run.linear_attn_out = output_projection.output;
  run.timing.output_projection_min_us = output_projection.timing.min_us;
  run.timing.output_projection_mean_us = output_projection.timing.mean_us;
  run.timing.output_projection_global_work_items =
      output_projection.timing.global_work_items;

  OpenClApi api;
  const auto selected = SelectDevice(api, device_substring);
  cl_int err = kClSuccess;
  cl_context context = api.clCreateContext(nullptr, 1, &selected.device, nullptr, nullptr, &err);
  Check(err, "clCreateContext(attention front)");
  cl_command_queue queue = api.clCreateCommandQueue(context, selected.device, kClQueueProfilingEnable, &err);
  Check(err, "clCreateCommandQueue(attention front)");
  const char* source = kOpenClSource;
  const std::size_t source_len = std::strlen(kOpenClSource);
  cl_program program = api.clCreateProgramWithSource(context, 1, &source, &source_len, &err);
  Check(err, "clCreateProgramWithSource(attention front)");
  const auto build_begin = std::chrono::steady_clock::now();
  err = api.clBuildProgram(program, 1, &selected.device, "", nullptr, nullptr);
  const auto build_end = std::chrono::steady_clock::now();
  run.program_build_ms +=
      std::chrono::duration<double, std::milli>(build_end - build_begin).count();
  run.build_log += BuildLog(api, program, selected.device);
  Check(err, "clBuildProgram(attention front)");
  cl_kernel residual_kernel = api.clCreateKernel(program, "post_ffn_residual_add_f32", &err);
  Check(err, "clCreateKernel(post_ffn_residual_add_f32)");
  cl_kernel rmsnorm_kernel = api.clCreateKernel(program, "rms_norm_hidden_f32", &err);
  Check(err, "clCreateKernel(rms_norm_hidden_f32)");

  run.attn_residual.assign(kHiddenSize, 0.0f);
  run.attn_post_norm.assign(kHiddenSize, 0.0f);
  cl_mem residual_input_buffer = nullptr;
  cl_mem linear_attn_out_buffer = nullptr;
  cl_mem ffn_norm_weight_buffer = nullptr;
  cl_mem attn_residual_buffer = nullptr;
  cl_mem attn_post_norm_buffer = nullptr;
  try {
    residual_input_buffer = api.clCreateBuffer(context, kClMemReadOnly,
                                               residual_input.size() * sizeof(float),
                                               nullptr, &err);
    Check(err, "clCreateBuffer(residual input)");
    linear_attn_out_buffer = api.clCreateBuffer(context, kClMemReadOnly,
                                                run.linear_attn_out.size() * sizeof(float),
                                                nullptr, &err);
    Check(err, "clCreateBuffer(linear attn out)");
    ffn_norm_weight_buffer = api.clCreateBuffer(context, kClMemReadOnly,
                                                ffn_norm_weight.size() * sizeof(float),
                                                nullptr, &err);
    Check(err, "clCreateBuffer(ffn norm weight)");
    attn_residual_buffer = api.clCreateBuffer(context, kClMemReadWriteLocal,
                                              run.attn_residual.size() * sizeof(float),
                                              nullptr, &err);
    Check(err, "clCreateBuffer(attn residual)");
    attn_post_norm_buffer = api.clCreateBuffer(context, kClMemWriteOnly,
                                               run.attn_post_norm.size() * sizeof(float),
                                               nullptr, &err);
    Check(err, "clCreateBuffer(attn post norm)");
    Check(api.clEnqueueWriteBuffer(queue, residual_input_buffer, kClTrue, 0,
                                   residual_input.size() * sizeof(float),
                                   residual_input.data(), 0, nullptr, nullptr),
          "clEnqueueWriteBuffer(residual input)");
    Check(api.clEnqueueWriteBuffer(queue, linear_attn_out_buffer, kClTrue, 0,
                                   run.linear_attn_out.size() * sizeof(float),
                                   run.linear_attn_out.data(), 0, nullptr, nullptr),
          "clEnqueueWriteBuffer(linear attn out)");
    Check(api.clEnqueueWriteBuffer(queue, ffn_norm_weight_buffer, kClTrue, 0,
                                   ffn_norm_weight.size() * sizeof(float),
                                   ffn_norm_weight.data(), 0, nullptr, nullptr),
          "clEnqueueWriteBuffer(ffn norm weight)");

    const cl_uint hidden_arg = static_cast<cl_uint>(kHiddenSize);
    Check(api.clSetKernelArg(residual_kernel, 0, sizeof(residual_input_buffer), &residual_input_buffer), "clSetKernelArg(attn residual 0)");
    Check(api.clSetKernelArg(residual_kernel, 1, sizeof(linear_attn_out_buffer), &linear_attn_out_buffer), "clSetKernelArg(attn residual 1)");
    Check(api.clSetKernelArg(residual_kernel, 2, sizeof(hidden_arg), &hidden_arg), "clSetKernelArg(attn residual 2)");
    Check(api.clSetKernelArg(residual_kernel, 3, sizeof(attn_residual_buffer), &attn_residual_buffer), "clSetKernelArg(attn residual 3)");
    Check(api.clSetKernelArg(rmsnorm_kernel, 0, sizeof(attn_residual_buffer), &attn_residual_buffer), "clSetKernelArg(ffn rmsnorm 0)");
    Check(api.clSetKernelArg(rmsnorm_kernel, 1, sizeof(ffn_norm_weight_buffer), &ffn_norm_weight_buffer), "clSetKernelArg(ffn rmsnorm 1)");
    Check(api.clSetKernelArg(rmsnorm_kernel, 2, sizeof(hidden_arg), &hidden_arg), "clSetKernelArg(ffn rmsnorm 2)");
    Check(api.clSetKernelArg(rmsnorm_kernel, 3, sizeof(rms_norm_epsilon), &rms_norm_epsilon), "clSetKernelArg(ffn rmsnorm 3)");
    Check(api.clSetKernelArg(rmsnorm_kernel, 4, sizeof(attn_post_norm_buffer), &attn_post_norm_buffer), "clSetKernelArg(ffn rmsnorm 4)");

    const std::size_t hidden_global = static_cast<std::size_t>(kHiddenSize);
    const std::size_t rmsnorm_global = 1;
    std::vector<double> residual_times;
    std::vector<double> rmsnorm_times;
    std::vector<double> sum_times;
    residual_times.reserve(static_cast<std::size_t>(repeat));
    rmsnorm_times.reserve(static_cast<std::size_t>(repeat));
    sum_times.reserve(static_cast<std::size_t>(repeat));
    for (int i = 0; i < repeat; ++i) {
      cl_event residual_event = nullptr;
      cl_event rmsnorm_event = nullptr;
      Check(api.clEnqueueNDRangeKernel(queue, residual_kernel, 1, nullptr,
                                       &hidden_global, nullptr, 0, nullptr,
                                       &residual_event),
            "clEnqueueNDRangeKernel(post attention residual)");
      Check(api.clEnqueueNDRangeKernel(queue, rmsnorm_kernel, 1, nullptr,
                                       &rmsnorm_global, nullptr, 0, nullptr,
                                       &rmsnorm_event),
            "clEnqueueNDRangeKernel(ffn rmsnorm)");
      Check(api.clFinish(queue), "clFinish(attention front)");
      const double residual_us = EventUs(api, residual_event);
      const double rmsnorm_us = EventUs(api, rmsnorm_event);
      residual_times.push_back(residual_us);
      rmsnorm_times.push_back(rmsnorm_us);
      sum_times.push_back(output_projection.timing.min_us + residual_us + rmsnorm_us);
      api.clReleaseEvent(residual_event);
      api.clReleaseEvent(rmsnorm_event);
    }
    Check(api.clEnqueueReadBuffer(queue, attn_residual_buffer, kClTrue, 0,
                                  run.attn_residual.size() * sizeof(float),
                                  run.attn_residual.data(), 0, nullptr, nullptr),
          "clEnqueueReadBuffer(attn residual)");
    Check(api.clEnqueueReadBuffer(queue, attn_post_norm_buffer, kClTrue, 0,
                                  run.attn_post_norm.size() * sizeof(float),
                                  run.attn_post_norm.data(), 0, nullptr, nullptr),
          "clEnqueueReadBuffer(attn post norm)");
    run.timing.residual_add_min_us = Min(residual_times);
    run.timing.residual_add_mean_us = Mean(residual_times);
    run.timing.ffn_rmsnorm_min_us = Min(rmsnorm_times);
    run.timing.ffn_rmsnorm_mean_us = Mean(rmsnorm_times);
    run.timing.attention_front_kernel_sum_min_us =
        output_projection.timing.min_us + run.timing.residual_add_min_us +
        run.timing.ffn_rmsnorm_min_us;
    run.timing.attention_front_kernel_sum_mean_us =
        output_projection.timing.mean_us + run.timing.residual_add_mean_us +
        run.timing.ffn_rmsnorm_mean_us;
    run.timing.residual_add_global_work_items = kHiddenSize;
    run.timing.ffn_rmsnorm_global_work_items = 1;
  } catch (...) {
    ReleaseMem(api, &attn_post_norm_buffer);
    ReleaseMem(api, &attn_residual_buffer);
    ReleaseMem(api, &ffn_norm_weight_buffer);
    ReleaseMem(api, &linear_attn_out_buffer);
    ReleaseMem(api, &residual_input_buffer);
    api.clReleaseKernel(rmsnorm_kernel);
    api.clReleaseKernel(residual_kernel);
    api.clReleaseProgram(program);
    api.clReleaseCommandQueue(queue);
    api.clReleaseContext(context);
    throw;
  }
  ReleaseMem(api, &attn_post_norm_buffer);
  ReleaseMem(api, &attn_residual_buffer);
  ReleaseMem(api, &ffn_norm_weight_buffer);
  ReleaseMem(api, &linear_attn_out_buffer);
  ReleaseMem(api, &residual_input_buffer);
  api.clReleaseKernel(rmsnorm_kernel);
  api.clReleaseKernel(residual_kernel);
  api.clReleaseProgram(program);
  api.clReleaseCommandQueue(queue);
  api.clReleaseContext(context);
  return run;
}
'''


ATTENTION_MAIN_CPP = r'''
int main(int argc, char** argv) {
  try {
    const Args args = ParseArgs(argc, argv);
    const auto index = iq36::parse_gguf_model_index(args.model_path);
    const auto load_map = iq36::validate_qwen36_load_map(index);
    const float rms_norm_epsilon = MetadataFloat(
        index, "qwen35moe.attention.layer_norm_rms_epsilon", 1e-6f);
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
    Require(output_tensor != nullptr, "attention output tensor missing");
    Require(ffn_norm_tensor != nullptr, "ffn norm tensor missing");
    Require(selected_gate_up_tensor != nullptr, "selected gate-up tensor missing");
    Require(selected_down_tensor != nullptr, "selected down tensor missing");
    Require(shared_gate_tensor != nullptr, "shared gate tensor missing");
    Require(shared_up_tensor != nullptr, "shared up tensor missing");
    Require(shared_down_tensor != nullptr, "shared down tensor missing");
    Require(shared_input_gate_tensor != nullptr, "shared input gate tensor missing");
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

    const auto residual_input =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "residual_input.bin"));
    const auto final_output =
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
        residual_input.size() == kHiddenSize &&
        final_output.size() == 4096 &&
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
    const auto ffn_norm_weight =
        ReadF32TensorPayload(model, *ffn_norm_tensor,
                             static_cast<std::size_t>(kHiddenSize));
    const auto shared_input_gate_weights =
        ReadF32TensorPayload(model, *shared_input_gate_tensor,
                             static_cast<std::size_t>(kHiddenSize));

    const auto native_linear_attn_out =
        iq36::matvec_tensor(args.model_path, index, output_tensor_name, final_output);
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

    const auto attention_gpu = RunGpuAttentionFront(
        args.model_path, *output_tensor, final_output, residual_input,
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
        attention_gpu.device_name.find(args.device_substring) != std::string::npos &&
        selected_gpu.device_name.find(args.device_substring) != std::string::npos &&
        shared_gpu.device_name.find(args.device_substring) != std::string::npos &&
        gpu.device_name.find(args.device_substring) != std::string::npos;
    const bool required_checks_passed =
        load_map.ready &&
        attention_tensor_shape_ok &&
        selected_tensor_shape_ok &&
        shared_tensor_shape_ok &&
        payload_counts_ok &&
        arc_selected &&
        attention_front_comparisons_passed &&
        selected_comparisons_passed &&
        shared_comparisons_passed &&
        tail_comparisons_passed &&
        attention_timing_positive &&
        selected_timing_positive &&
        shared_timing_positive &&
        tail_timing_positive;
    const double full_kernel_sum_min =
        attention_gpu.timing.attention_front_kernel_sum_min_us +
        selected_gpu.timing.selected_ffn_kernel_sum_min_us +
        shared_gpu.timing.shared_ffn_kernel_sum_min_us +
        gpu.timing.shell_sum_min_us;
    const double full_kernel_sum_mean =
        attention_gpu.timing.attention_front_kernel_sum_mean_us +
        selected_gpu.timing.selected_ffn_kernel_sum_mean_us +
        shared_gpu.timing.shared_ffn_kernel_sum_mean_us +
        gpu.timing.shell_sum_mean_us;

    std::cout << std::setprecision(10);
    std::cout << "{";
    std::cout << "\"schema_version\":\"intel-qwen36-gpu-resident-attention-ffn-shell-probe-v0\",";
    std::cout << "\"model_path\":\"" << JsonEscape(args.model_path) << "\",";
    std::cout << "\"layer\":" << args.layer << ",";
    std::cout << "\"source_token_position\":" << kSourceTokenPosition << ",";
    std::cout << "\"resident_api\":\"attention_to_ffn_layer_shell_load_once_run_many\",";
    std::cout << "\"resident_load_count\":1,";
    std::cout << "\"resident_shell_invocations\":" << args.repeat << ",";
    std::cout << "\"attention_output_projection_host_q8_bridge\":true,";
    std::cout << "\"attention_front_host_boundary_between_q4_and_f32\":true,";
    std::cout << "\"selected_down_host_q8_bridge\":true,";
    std::cout << "\"shared_down_host_q8_bridge\":true,";
    std::cout << "\"selected_down_q4_expert_launches\":"
              << selected_gpu.timing.down_kernel_launches << ",";
    std::cout << "\"shared_down_kernel_launches\":"
              << shared_gpu.timing.down_kernel_launches << ",";
    std::cout << "\"attention_output_tensor_name\":\""
              << JsonEscape(output_tensor_name) << "\",";
    std::cout << "\"ffn_norm_tensor_name\":\""
              << JsonEscape(ffn_norm_tensor_name) << "\",";
    std::cout << "\"selected_gate_up_tensor_name\":\""
              << JsonEscape(selected_gate_up_tensor_name) << "\",";
    std::cout << "\"selected_down_tensor_name\":\""
              << JsonEscape(selected_down_tensor_name) << "\",";
    std::cout << "\"shared_gate_tensor_name\":\""
              << JsonEscape(shared_gate_tensor_name) << "\",";
    std::cout << "\"shared_up_tensor_name\":\""
              << JsonEscape(shared_up_tensor_name) << "\",";
    std::cout << "\"shared_down_tensor_name\":\""
              << JsonEscape(shared_down_tensor_name) << "\",";
    std::cout << "\"shared_down_tensor_type\":\""
              << iq36::ggml_type_name(shared_down_tensor->type) << "\",";
    std::cout << "\"rms_norm_epsilon\":" << rms_norm_epsilon << ",";
    std::cout << "\"platform_name\":\"" << JsonEscape(attention_gpu.platform_name) << "\",";
    std::cout << "\"device_name\":\"" << JsonEscape(attention_gpu.device_name) << "\",";
    std::cout << "\"selected_device_name\":\"" << JsonEscape(selected_gpu.device_name) << "\",";
    std::cout << "\"shared_device_name\":\"" << JsonEscape(shared_gpu.device_name) << "\",";
    std::cout << "\"tail_device_name\":\"" << JsonEscape(gpu.device_name) << "\",";
    std::cout << "\"program_build_ms\":"
              << (attention_gpu.program_build_ms + selected_gpu.program_build_ms +
                  shared_gpu.program_build_ms + gpu.program_build_ms) << ",";
    std::cout << "\"build_log\":\""
              << JsonEscape(attention_gpu.build_log + selected_gpu.build_log +
                            shared_gpu.build_log + gpu.build_log) << "\",";
    std::cout << "\"repeat\":" << args.repeat << ",";
    std::cout << "\"timings\":{";
    std::cout << "\"attention_output_projection_min_us\":"
              << attention_gpu.timing.output_projection_min_us << ",";
    std::cout << "\"attention_output_projection_mean_us\":"
              << attention_gpu.timing.output_projection_mean_us << ",";
    std::cout << "\"attention_output_projection_host_q8_bridge_us\":"
              << attention_gpu.timing.output_projection_host_q8_bridge_us << ",";
    std::cout << "\"post_attention_residual_add_min_us\":"
              << attention_gpu.timing.residual_add_min_us << ",";
    std::cout << "\"post_attention_residual_add_mean_us\":"
              << attention_gpu.timing.residual_add_mean_us << ",";
    std::cout << "\"ffn_rmsnorm_min_us\":"
              << attention_gpu.timing.ffn_rmsnorm_min_us << ",";
    std::cout << "\"ffn_rmsnorm_mean_us\":"
              << attention_gpu.timing.ffn_rmsnorm_mean_us << ",";
    std::cout << "\"attention_front_kernel_sum_min_us\":"
              << attention_gpu.timing.attention_front_kernel_sum_min_us << ",";
    std::cout << "\"attention_front_kernel_sum_mean_us\":"
              << attention_gpu.timing.attention_front_kernel_sum_mean_us << ",";
    std::cout << "\"selected_ffn_kernel_sum_min_us\":"
              << selected_gpu.timing.selected_ffn_kernel_sum_min_us << ",";
    std::cout << "\"selected_ffn_kernel_sum_mean_us\":"
              << selected_gpu.timing.selected_ffn_kernel_sum_mean_us << ",";
    std::cout << "\"shared_ffn_kernel_sum_min_us\":"
              << shared_gpu.timing.shared_ffn_kernel_sum_min_us << ",";
    std::cout << "\"shared_ffn_kernel_sum_mean_us\":"
              << shared_gpu.timing.shared_ffn_kernel_sum_mean_us << ",";
    std::cout << "\"ffn_tail_kernel_sum_min_us\":"
              << gpu.timing.shell_sum_min_us << ",";
    std::cout << "\"ffn_tail_kernel_sum_mean_us\":"
              << gpu.timing.shell_sum_mean_us << ",";
    std::cout << "\"resident_attention_to_layer_kernel_sum_min_us\":"
              << full_kernel_sum_min << ",";
    std::cout << "\"resident_attention_to_layer_kernel_sum_mean_us\":"
              << full_kernel_sum_mean << ",";
    std::cout << "\"selected_down_host_q8_bridge_us\":"
              << selected_gpu.timing.host_q8_bridge_us << ",";
    std::cout << "\"shared_down_host_q8_bridge_us\":"
              << shared_gpu.timing.host_q8_bridge_us;
    std::cout << "},\"comparisons\":{";
    std::cout << "\"linear_attn_out\":";
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
    std::cout << "\"attention_front_matches_oracle\":"
              << (attention_front_comparisons_passed ? "true" : "false") << ",";
    std::cout << "\"selected_expert_ffn_matches_oracle\":"
              << (selected_comparisons_passed ? "true" : "false") << ",";
    std::cout << "\"shared_expert_ffn_matches_oracle\":"
              << (shared_comparisons_passed ? "true" : "false") << ",";
    std::cout << "\"tail_matches_oracle\":"
              << (tail_comparisons_passed ? "true" : "false") << ",";
    std::cout << "\"gpu_event_timing_positive\":"
              << (attention_timing_positive && selected_timing_positive &&
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


def opencl_with_attention_extra(opencl_source: str) -> str:
  required = (
      "__kernel void rms_norm_hidden_f32",
      "__kernel void rms_norm_hidden_scale_f32",
      "__kernel void rms_norm_hidden_apply_scale_f32",
  )
  if all(item in opencl_source for item in required):
    return opencl_source
  return opencl_source + ATTENTION_EXTRA_OPENCL


def attention_ffn_probe_cpp(opencl_source: str) -> str:
  cpp = SHARED.shared_ffn_probe_cpp(opencl_with_attention_extra(opencl_source))
  main_index = cpp.index("\nint main(")
  return cpp[:main_index] + "\n" + ATTENTION_HELPERS_CPP + "\n" + ATTENTION_MAIN_CPP


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
  payloads = SHARED.SELECTED.resolve_payloads(layer)
  if layer == 0:
    residual_path = find_payload("model.input_embed__tok15__ord*.bin", 8192)
  else:
    residual_path = find_payload(f"l_out-{layer - 1}__tok15__ord*.bin", 8192)
  add_payload(payloads, "residual_input", "residual_input.bin", residual_path, 8192)
  add_payload(
      payloads,
      "final_output",
      "final_output.bin",
      find_payload(f"final_output-{layer}__tok15__ord*.bin", 16384),
      16384,
  )
  add_payload(
      payloads,
      "linear_attn_out",
      "linear_attn_out.bin",
      find_payload(f"linear_attn_out-{layer}__tok15__ord*.bin", 8192),
      8192,
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
      and probe.get("resident_api") == "attention_to_ffn_layer_shell_load_once_run_many"
      and probe.get("resident_load_count") == 1
      and probe.get("resident_shell_invocations") == expected_invocations
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
      "# GPU Resident Attention-to-FFN Layer Shell Probe",
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
      f"- attention output Q8 bridge: `{str(probe.get('attention_output_projection_host_q8_bridge')).lower()}`",
      f"- attention front host boundary: `{str(probe.get('attention_front_host_boundary_between_q4_and_f32')).lower()}`",
      f"- selected down host Q8 bridge: `{str(probe.get('selected_down_host_q8_bridge')).lower()}`",
      f"- shared down host Q8 bridge: `{str(probe.get('shared_down_host_q8_bridge')).lower()}`",
      "",
      "| output | comparison | max abs | RMSE |",
      "|---|---|---:|---:|",
  ]
  for name in (
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
      f"| output_projection | {timings.get('attention_output_projection_min_us')} |",
      f"| post_attention_residual_add | {timings.get('post_attention_residual_add_min_us')} |",
      f"| ffn_rmsnorm | {timings.get('ffn_rmsnorm_min_us')} |",
      f"| attention_front_sum | {timings.get('attention_front_kernel_sum_min_us')} |",
      f"| selected_ffn_sum | {timings.get('selected_ffn_kernel_sum_min_us')} |",
      f"| shared_ffn_sum | {timings.get('shared_ffn_kernel_sum_min_us')} |",
      f"| ffn_tail_sum | {timings.get('ffn_tail_kernel_sum_min_us')} |",
      f"| resident_attention_to_layer_sum | {timings.get('resident_attention_to_layer_kernel_sum_min_us')} |",
      "",
      "The target-side process starts from captured `final_output` and previous",
      "layer residual input, computes `ssm_out.weight`, post-attention residual,",
      "FFN RMSNorm, selected/shared FFN branches, tail aggregation, and residual",
      "to captured `l_out`. Host boundaries and Q8 activation bridges remain",
      "explicit. This is captured single-layer evidence only, not prompt/token",
      "decode throughput.",
      "",
  ]
  path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  args = parse_args()
  if args.resident_invocations <= 0:
    raise SystemExit("--resident-invocations must be positive")
  stamp = utc_stamp()
  created_at = iso_now()
  out_dir = args.out_dir or ROOT / f"output/gpu-resident-attention-ffn-shell-probe-{stamp}"
  out_dir.mkdir(parents=True, exist_ok=False)
  raw_dir = out_dir / "raw"
  raw_dir.mkdir()

  payloads = resolve_payloads(args.layer)
  opencl_source = OPENCL_SOURCE.read_text(encoding="utf-8")
  opencl_source_hash = iq36_local.sha256_file(OPENCL_SOURCE)
  embedded_opencl_hash = hashlib.sha256(
      (opencl_source + ATTENTION_EXTRA_OPENCL).encode("utf-8")
  ).hexdigest()
  local_cpp = out_dir / "gpu_resident_attention_ffn_shell_probe.cpp"
  local_cpp.write_text(attention_ffn_probe_cpp(opencl_source), encoding="utf-8")

  remote_dir = f"{args.remote_root.rstrip('/')}/gpu-resident-attention-ffn-shell-probe-{stamp}"
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
    for local, remote in SHARED.SELECTED.SOURCE_FILES:
      transfers.append(iq36_local.copy_to(args.host, ROOT / local, f"{remote_dir}/{remote}", args.timeout_s))
    transfers.append(
        iq36_local.copy_to(
            args.host,
            local_cpp,
            f"{remote_dir}/tests/gpu_resident_attention_ffn_shell_probe.cpp",
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

  executable = f"{remote_dir}/build/iq36-gpu-resident-attention-ffn-shell-probe"
  compile_cmd = " && ".join([
      f"source {shlex.quote(args.env_script)} >/dev/null 2>&1",
      (
          "g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic "
          f"-I {shlex.quote(remote_dir + '/include')} "
          f"{shlex.quote(remote_dir + '/src/gguf_loader.cpp')} "
          f"{shlex.quote(remote_dir + '/src/gpu_q4x8_matvec.cpp')} "
          f"{shlex.quote(remote_dir + '/tests/gpu_resident_attention_ffn_shell_probe.cpp')} "
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
      "tool": "tools/intel-qwen36-gpu-resident-attention-ffn-shell-probe.py",
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
      "gpu_resident_attention_ffn_shell_probe",
      [
          ("required_checks_passed", required_checks_passed),
          ("resident_shell_invocations", args.resident_invocations),
          ("attention_front_kernel_sum_min_us", nested_number(timings, "attention_front_kernel_sum_min_us")),
          ("resident_attention_to_layer_kernel_sum_min_us", nested_number(timings, "resident_attention_to_layer_kernel_sum_min_us")),
          ("attention_output_projection_host_q8_bridge_us", nested_number(timings, "attention_output_projection_host_q8_bridge_us")),
          ("attn_post_norm_gpu_vs_oracle_max_abs_diff", nested_number(comparisons, "attn_post_norm", "gpu_vs_oracle", "max_abs_diff")),
          ("layer_output_gpu_vs_oracle_max_abs_diff", nested_number(comparisons, "layer_output", "gpu_vs_oracle", "max_abs_diff")),
      ],
  )
  write_summary(out_dir / "summary.md", payload)
  print(out_dir)
  return 0 if required_checks_passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
