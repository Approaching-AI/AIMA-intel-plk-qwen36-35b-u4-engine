#!/usr/bin/env python3
"""Run the resident GPU layer-10 Q6 state/input handoff probe."""

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
L9_TOOL = Path(__file__).with_name(
    "intel-qwen36-gpu-resident-layer9-state-input-handoff-probe.py"
)
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA_VERSION = "intel-qwen36-gpu-resident-layer10-state-input-handoff-probe-v0"
DEFAULT_HOST = "local"
DEFAULT_MODEL = "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf"
DEFAULT_ENV_SCRIPT = "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh"
DEFAULT_REMOTE_ROOT = "/home/intel/intel-qwen36-gpu"
DEFAULT_ORACLE_BUNDLE = ROOT / "oracle/r0-oracle-bundle-20260627T060028Z"
DEFAULT_ALL_HISTORY = ROOT / "output/r1-full-attn-all-history-capture-20260627T145615Z/history.json"
OPENCL_SOURCE = ROOT / "engine/gpu/opencl/q4x8_matvec.cl"


def load_l9_tool() -> ModuleType:
  spec = importlib.util.spec_from_file_location("iq36_layer9_state_input_probe", L9_TOOL)
  if spec is None or spec.loader is None:
    raise SystemExit(f"failed to load layer9 state/input tool: {L9_TOOL}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


L9 = load_l9_tool()
L8 = L9.L8
V_Q6 = L9.V_Q6
CORE = L9.CORE
L7_INPUT = L9.L7_INPUT
TWO = L9.TWO
PRECONV = L9.PRECONV


LAYER10_Q6_STATE_INPUT_CPP = r'''

struct Layer10Q6StateInputRun {
  std::vector<float> qkv_mixed;
  std::vector<float> conv_output_raw;
  std::vector<float> conv_state_after;
  double qkv_min_us = 0.0;
  double qkv_mean_us = 0.0;
  double qkv_effective_packed_gb_s = 0.0;
  double conv_min_us = 0.0;
  double conv_mean_us = 0.0;
  std::string platform_name;
  std::string device_name;
  std::string build_log;
  double program_build_ms = 0.0;
};

Layer10Q6StateInputRun RunGpuLayer10Q6StateInput(
    const std::string& model_path,
    const iq36::GgufTensorInfo& qkv_tensor,
    const iq36::GgufTensorInfo& conv_tensor,
    const std::vector<float>& attn_norm,
    const std::vector<float>& conv_state,
    int repeat,
    const std::string& device_substring) {
  Require(qkv_tensor.type == 14, "layer10 QKV tensor must be Q6_K");
  Require(qkv_tensor.dims == std::vector<std::uint64_t>{kHiddenSize, kLinearQkvMixedValues},
          "layer10 Q6 QKV tensor shape mismatch");
  Require(conv_tensor.type == 0, "layer10 conv tensor must be F32");
  Require(conv_tensor.dims == std::vector<std::uint64_t>{kLinearConvKernelSize, kLinearQkvMixedValues},
          "layer10 conv tensor shape mismatch");
  Require(attn_norm.size() == kHiddenSize, "layer10 Q6 attn_norm size mismatch");
  Require(conv_state.size() == kLinearConvStateValues, "layer10 Q6 conv state size mismatch");
  std::ifstream model(model_path, std::ios::binary);
  Require(static_cast<bool>(model), "failed to open model for layer10 Q6 state/input");
  const auto qkv_raw = ReadTensorBytes(model, qkv_tensor);
  const auto conv_weights =
      ReadF32TensorPayload(model, conv_tensor, kLinearQkvMixedValues * kLinearConvKernelSize);
  const auto q8 = iq36::QuantizeQ8KInputPlanes(attn_norm);
  Require(qkv_raw.size() == kLinearQkvMixedValues * (kHiddenSize / 256) * kQ6KBlockBytes,
          "layer10 Q6 raw byte size mismatch");

  Layer10Q6StateInputRun run;
  run.qkv_mixed.assign(kLinearQkvMixedValues, 0.0f);
  run.conv_output_raw.assign(kLinearQkvMixedValues, 0.0f);
  run.conv_state_after.assign(kLinearConvStateValues, 0.0f);

  OpenClApi api;
  const auto selected = SelectDevice(api, device_substring);
  run.platform_name = selected.platform_name;
  run.device_name = selected.device_name;
  cl_int err = kClSuccess;
  cl_context context =
      api.clCreateContext(nullptr, 1, &selected.device, nullptr, nullptr, &err);
  Check(err, "clCreateContext(layer10 q6 state/input)");
  cl_command_queue queue =
      api.clCreateCommandQueue(context, selected.device, kClQueueProfilingEnable, &err);
  Check(err, "clCreateCommandQueue(layer10 q6 state/input)");
  const char* source = kOpenClSource;
  const std::size_t source_len = std::strlen(kOpenClSource);
  cl_program program = api.clCreateProgramWithSource(context, 1, &source, &source_len, &err);
  Check(err, "clCreateProgramWithSource(layer10 q6 state/input)");
  const auto build_begin = std::chrono::steady_clock::now();
  err = api.clBuildProgram(program, 1, &selected.device, "", nullptr, nullptr);
  const auto build_end = std::chrono::steady_clock::now();
  run.program_build_ms =
      std::chrono::duration<double, std::milli>(build_end - build_begin).count();
  run.build_log = BuildLog(api, program, selected.device);
  Check(err, "clBuildProgram(layer10 q6 state/input)");
  cl_kernel q6_kernel = api.clCreateKernel(program, "q6k_selected_down_matvec_row", &err);
  Check(err, "clCreateKernel(layer10 q6k_selected_down_matvec_row)");
  cl_kernel conv_kernel = api.clCreateKernel(program, "linear_attn_conv_f32", &err);
  Check(err, "clCreateKernel(layer10 linear_attn_conv_f32)");

  cl_mem q6_buffer = nullptr;
  cl_mem q8_qs_buffer = nullptr;
  cl_mem q8_d_buffer = nullptr;
  cl_mem qkv_buffer = nullptr;
  cl_mem conv_weight_buffer = nullptr;
  cl_mem conv_state_buffer = nullptr;
  cl_mem conv_output_buffer = nullptr;
  cl_mem next_state_buffer = nullptr;
  try {
    q6_buffer = api.clCreateBuffer(context, kClMemReadOnly, qkv_raw.size(), nullptr, &err);
    Check(err, "clCreateBuffer(layer10 q6 raw)");
    q8_qs_buffer = api.clCreateBuffer(
        context, kClMemReadOnly, q8.qs.size() * sizeof(std::int8_t), nullptr, &err);
    Check(err, "clCreateBuffer(layer10 q8 qs)");
    q8_d_buffer = api.clCreateBuffer(
        context, kClMemReadOnly, q8.d.size() * sizeof(float), nullptr, &err);
    Check(err, "clCreateBuffer(layer10 q8 d)");
    qkv_buffer = api.clCreateBuffer(
        context, kClMemWriteOnly, run.qkv_mixed.size() * sizeof(float), nullptr, &err);
    Check(err, "clCreateBuffer(layer10 qkv output)");
    conv_weight_buffer = api.clCreateBuffer(
        context, kClMemReadOnly, conv_weights.size() * sizeof(float), nullptr, &err);
    Check(err, "clCreateBuffer(layer10 conv weights)");
    conv_state_buffer = api.clCreateBuffer(
        context, kClMemReadOnly, conv_state.size() * sizeof(float), nullptr, &err);
    Check(err, "clCreateBuffer(layer10 conv state)");
    conv_output_buffer = api.clCreateBuffer(
        context, kClMemWriteOnly, run.conv_output_raw.size() * sizeof(float), nullptr, &err);
    Check(err, "clCreateBuffer(layer10 conv output)");
    next_state_buffer = api.clCreateBuffer(
        context, kClMemWriteOnly, run.conv_state_after.size() * sizeof(float), nullptr, &err);
    Check(err, "clCreateBuffer(layer10 next state)");
    Check(api.clEnqueueWriteBuffer(queue, q6_buffer, kClTrue, 0,
                                   qkv_raw.size(), qkv_raw.data(), 0, nullptr, nullptr),
          "clEnqueueWriteBuffer(layer10 q6 raw)");
    Check(api.clEnqueueWriteBuffer(queue, q8_qs_buffer, kClTrue, 0,
                                   q8.qs.size() * sizeof(std::int8_t), q8.qs.data(),
                                   0, nullptr, nullptr),
          "clEnqueueWriteBuffer(layer10 q8 qs)");
    Check(api.clEnqueueWriteBuffer(queue, q8_d_buffer, kClTrue, 0,
                                   q8.d.size() * sizeof(float), q8.d.data(),
                                   0, nullptr, nullptr),
          "clEnqueueWriteBuffer(layer10 q8 d)");
    Check(api.clEnqueueWriteBuffer(queue, conv_weight_buffer, kClTrue, 0,
                                   conv_weights.size() * sizeof(float), conv_weights.data(),
                                   0, nullptr, nullptr),
          "clEnqueueWriteBuffer(layer10 conv weights)");
    Check(api.clEnqueueWriteBuffer(queue, conv_state_buffer, kClTrue, 0,
                                   conv_state.size() * sizeof(float), conv_state.data(),
                                   0, nullptr, nullptr),
          "clEnqueueWriteBuffer(layer10 conv state)");
    const cl_uint rows_arg = static_cast<cl_uint>(kLinearQkvMixedValues);
    const cl_uint blocks_arg = static_cast<cl_uint>(kHiddenSize / 256);
    const cl_uint conv_kernel_arg = static_cast<cl_uint>(kLinearConvKernelSize);
    Check(api.clSetKernelArg(q6_kernel, 0, sizeof(q6_buffer), &q6_buffer),
          "clSetKernelArg(layer10 q6 0)");
    Check(api.clSetKernelArg(q6_kernel, 1, sizeof(q8_qs_buffer), &q8_qs_buffer),
          "clSetKernelArg(layer10 q6 1)");
    Check(api.clSetKernelArg(q6_kernel, 2, sizeof(q8_d_buffer), &q8_d_buffer),
          "clSetKernelArg(layer10 q6 2)");
    Check(api.clSetKernelArg(q6_kernel, 3, sizeof(rows_arg), &rows_arg),
          "clSetKernelArg(layer10 q6 3)");
    Check(api.clSetKernelArg(q6_kernel, 4, sizeof(blocks_arg), &blocks_arg),
          "clSetKernelArg(layer10 q6 4)");
    Check(api.clSetKernelArg(q6_kernel, 5, sizeof(qkv_buffer), &qkv_buffer),
          "clSetKernelArg(layer10 q6 5)");
    Check(api.clSetKernelArg(conv_kernel, 0, sizeof(qkv_buffer), &qkv_buffer),
          "clSetKernelArg(layer10 conv 0)");
    Check(api.clSetKernelArg(conv_kernel, 1, sizeof(conv_state_buffer), &conv_state_buffer),
          "clSetKernelArg(layer10 conv 1)");
    Check(api.clSetKernelArg(conv_kernel, 2, sizeof(conv_weight_buffer), &conv_weight_buffer),
          "clSetKernelArg(layer10 conv 2)");
    Check(api.clSetKernelArg(conv_kernel, 3, sizeof(rows_arg), &rows_arg),
          "clSetKernelArg(layer10 conv 3)");
    Check(api.clSetKernelArg(conv_kernel, 4, sizeof(conv_kernel_arg), &conv_kernel_arg),
          "clSetKernelArg(layer10 conv 4)");
    Check(api.clSetKernelArg(conv_kernel, 5, sizeof(conv_output_buffer), &conv_output_buffer),
          "clSetKernelArg(layer10 conv 5)");
    Check(api.clSetKernelArg(conv_kernel, 6, sizeof(next_state_buffer), &next_state_buffer),
          "clSetKernelArg(layer10 conv 6)");
    const std::size_t global = static_cast<std::size_t>(kLinearQkvMixedValues);
    run.qkv_min_us = std::numeric_limits<double>::infinity();
    run.conv_min_us = std::numeric_limits<double>::infinity();
    for (int i = 0; i < repeat; ++i) {
      cl_event q6_event = nullptr;
      cl_event conv_event = nullptr;
      Check(api.clEnqueueNDRangeKernel(queue, q6_kernel, 1, nullptr, &global, nullptr,
                                       0, nullptr, &q6_event),
            "clEnqueueNDRangeKernel(layer10 q6)");
      Check(api.clEnqueueNDRangeKernel(queue, conv_kernel, 1, nullptr, &global, nullptr,
                                       0, nullptr, &conv_event),
            "clEnqueueNDRangeKernel(layer10 conv)");
      Check(api.clFinish(queue), "clFinish(layer10 q6 state/input)");
      const double q6_us = EventUs(api, q6_event);
      const double conv_us = EventUs(api, conv_event);
      run.qkv_min_us = std::min(run.qkv_min_us, q6_us);
      run.qkv_mean_us += q6_us;
      run.conv_min_us = std::min(run.conv_min_us, conv_us);
      run.conv_mean_us += conv_us;
      api.clReleaseEvent(q6_event);
      api.clReleaseEvent(conv_event);
    }
    run.qkv_mean_us /= static_cast<double>(repeat);
    run.conv_mean_us /= static_cast<double>(repeat);
    run.qkv_effective_packed_gb_s =
        static_cast<double>(qkv_raw.size()) / run.qkv_min_us / 1000.0;
    Check(api.clEnqueueReadBuffer(queue, qkv_buffer, kClTrue, 0,
                                  run.qkv_mixed.size() * sizeof(float),
                                  run.qkv_mixed.data(), 0, nullptr, nullptr),
          "clEnqueueReadBuffer(layer10 qkv)");
    Check(api.clEnqueueReadBuffer(queue, conv_output_buffer, kClTrue, 0,
                                  run.conv_output_raw.size() * sizeof(float),
                                  run.conv_output_raw.data(), 0, nullptr, nullptr),
          "clEnqueueReadBuffer(layer10 conv output)");
    Check(api.clEnqueueReadBuffer(queue, next_state_buffer, kClTrue, 0,
                                  run.conv_state_after.size() * sizeof(float),
                                  run.conv_state_after.data(), 0, nullptr, nullptr),
          "clEnqueueReadBuffer(layer10 conv state)");
  } catch (...) {
    ReleaseMem(api, &next_state_buffer);
    ReleaseMem(api, &conv_output_buffer);
    ReleaseMem(api, &conv_state_buffer);
    ReleaseMem(api, &conv_weight_buffer);
    ReleaseMem(api, &qkv_buffer);
    ReleaseMem(api, &q8_d_buffer);
    ReleaseMem(api, &q8_qs_buffer);
    ReleaseMem(api, &q6_buffer);
    api.clReleaseKernel(conv_kernel);
    api.clReleaseKernel(q6_kernel);
    api.clReleaseProgram(program);
    api.clReleaseCommandQueue(queue);
    api.clReleaseContext(context);
    throw;
  }
  ReleaseMem(api, &next_state_buffer);
  ReleaseMem(api, &conv_output_buffer);
  ReleaseMem(api, &conv_state_buffer);
  ReleaseMem(api, &conv_weight_buffer);
  ReleaseMem(api, &qkv_buffer);
  ReleaseMem(api, &q8_d_buffer);
  ReleaseMem(api, &q8_qs_buffer);
  ReleaseMem(api, &q6_buffer);
  api.clReleaseKernel(conv_kernel);
  api.clReleaseKernel(q6_kernel);
  api.clReleaseProgram(program);
  api.clReleaseCommandQueue(queue);
  api.clReleaseContext(context);
  return run;
}
'''


def layer10_state_input_probe_cpp(opencl_source: str) -> str:
  cpp = L9.layer9_state_input_probe_cpp(opencl_source)
  replace_once = L8.replace_once
  cpp = replace_once(cpp, L9.SCHEMA_VERSION, SCHEMA_VERSION)
  cpp = replace_once(
      cpp,
      "two_linear_layer_to_full_attention_v_q6_live_ffn_lout_layer9_state_input_load_once_run_many",
      "two_linear_layer_to_full_attention_v_q6_live_ffn_lout_layer10_q6_state_input_load_once_run_many",
  )
  cpp = replace_once(
      cpp,
      "layer9 state/input handoff probe expects --layer 5",
      "layer10 state/input handoff probe expects --layer 5",
  )
  cpp = replace_once(cpp, "\nint main(int argc, char** argv) {\n", LAYER10_Q6_STATE_INPUT_CPP + "\nint main(int argc, char** argv) {\n")
  cpp = replace_once(
      cpp,
      '''    const int layer4 = args.layer + 4;
    Require(layer0 == 5 && layer1 == 6 && layer2 == 7 && layer3 == 8 && layer4 == 9,
            "layer10 state/input handoff probe expects --layer 5");
''',
      '''    const int layer4 = args.layer + 4;
    const int layer5 = args.layer + 5;
    Require(layer0 == 5 && layer1 == 6 && layer2 == 7 && layer3 == 8 && layer4 == 9 && layer5 == 10,
            "layer10 state/input handoff probe expects --layer 5");
''',
  )
  cpp = replace_once(
      cpp,
      '''    const auto layer4_tensors = ResolveLayerTensorBundle(index, layer4);
    const auto layer0_oracle = LoadLayerOraclePayloads(args.payload_dir, "l0");
''',
      '''    const auto layer4_tensors = ResolveLayerTensorBundle(index, layer4);
    const auto layer5_tensors = ResolveLayerTensorBundle(index, layer5);
    const auto layer0_oracle = LoadLayerOraclePayloads(args.payload_dir, "l0");
''',
  )
  cpp = replace_once(
      cpp,
      '''    const auto layer4_oracle = LoadLayerOraclePayloads(args.payload_dir, "l4");
    const auto oracle_attn_residual =
''',
      '''    const auto layer4_oracle = LoadLayerOraclePayloads(args.payload_dir, "l4");
    const auto layer5_oracle = LoadLayerOraclePayloads(args.payload_dir, "l5");
    const auto oracle_attn_residual =
''',
  )
  cpp = replace_once(
      cpp,
      '''    const auto layer4_run = RunResidentLinearLayerShell(
        args, index, layer4_tensors, layer4_oracle,
        layer3_run.gpu_layer_output, layer3_run.gpu_layer_output, rms_norm_epsilon);

    std::vector<NamedCompareGroup> strict_groups;
''',
      '''    const auto layer4_run = RunResidentLinearLayerShell(
        args, index, layer4_tensors, layer4_oracle,
        layer3_run.gpu_layer_output, layer3_run.gpu_layer_output, rms_norm_epsilon);
    std::ifstream layer5_model(args.model_path, std::ios::binary);
    Require(static_cast<bool>(layer5_model), "layer10 state/input model open failed");
    const auto layer5_attn_norm_weight =
        ReadF32TensorPayload(layer5_model, *layer5_tensors.attn_norm_tensor, kHiddenSize);
    const auto layer5_native_attn_norm =
        iq36::apply_rms_norm(layer4_run.gpu_layer_output, layer5_attn_norm_weight,
                             rms_norm_epsilon);
    const auto layer5_rms_gpu = RunGpuLayerInputRmsNorm(
        layer4_run.gpu_layer_output, layer5_attn_norm_weight, rms_norm_epsilon,
        args.device_substring, args.repeat);
    const auto layer5_native_preconv = iq36::run_qwen36_linear_attention_preconv_core(
        args.model_path, index, layer5, layer5_native_attn_norm);
    const auto layer5_native_conv = iq36::run_qwen36_linear_attention_conv_core(
        args.model_path, index, layer5, layer5_native_preconv.qkv_mixed,
        layer5_oracle.conv_state);
    const auto layer5_q6_gpu = RunGpuLayer10Q6StateInput(
        args.model_path, *layer5_tensors.qkv_tensor, *layer5_tensors.conv_tensor,
        layer5_rms_gpu.attn_norm, layer5_oracle.conv_state, args.repeat,
        args.device_substring);

    std::vector<NamedCompareGroup> strict_groups;
''',
  )
  cpp = replace_once(
      cpp,
      '''    const bool layer4_ok =
        layer4_shapes_ok &&
        layer4_run.payload_counts_ok &&
        layer4_gpu_cpu_ok &&
        layer4_state_input_oracle_policy_ok &&
        layer4_run.timing_positive &&
        layer4_run.arc_selected;
    const bool layer2_timing_positive =
''',
      '''    const bool layer4_ok =
        layer4_shapes_ok &&
        layer4_run.payload_counts_ok &&
        layer4_gpu_cpu_ok &&
        layer4_state_input_oracle_policy_ok &&
        layer4_run.timing_positive &&
        layer4_run.arc_selected;
    std::vector<NamedCompareGroup> layer5_state_input_groups;
    AppendCpuGpuOracleCompare(layer5_state_input_groups, "l5_residual_input",
                              layer4_run.gpu_layer_output,
                              layer4_run.gpu_layer_output,
                              layer5_oracle.residual_input);
    AppendCpuGpuOracleCompare(layer5_state_input_groups, "l5_attn_norm",
                              layer5_native_attn_norm,
                              layer5_rms_gpu.attn_norm,
                              layer5_oracle.attn_norm);
    AppendCpuGpuOracleCompare(layer5_state_input_groups, "l5_linear_attn_qkv_mixed",
                              layer5_native_preconv.qkv_mixed,
                              layer5_q6_gpu.qkv_mixed,
                              layer5_oracle.qkv);
    AppendCpuGpuOracleCompare(layer5_state_input_groups, "l5_conv_output_raw",
                              layer5_native_conv.conv_output_raw,
                              layer5_q6_gpu.conv_output_raw,
                              layer5_oracle.conv_output_raw);
    bool layer5_gpu_cpu_ok = true;
    bool layer5_oracle_policy_ok = true;
    for (const auto& group : layer5_state_input_groups) {
      layer5_gpu_cpu_ok = layer5_gpu_cpu_ok && ComparePassed(group.gpu_vs_cpu);
      layer5_oracle_policy_ok =
          layer5_oracle_policy_ok &&
          ComparePassedFullAttentionComponent(group.gpu_vs_oracle);
    }
    const bool layer5_conv_state_ok =
        ComparePassed(iq36::compare_vectors(layer5_q6_gpu.conv_state_after,
                                            layer5_native_conv.conv_state,
                                            kMismatchThreshold));
    const bool layer5_shapes_ok =
        layer5_tensors.attn_norm_tensor->type == 0 &&
        layer5_tensors.qkv_tensor->type == 14 &&
        layer5_tensors.conv_tensor->type == 0 &&
        layer5_tensors.attn_norm_tensor->dims == std::vector<std::uint64_t>{kHiddenSize} &&
        layer5_tensors.qkv_tensor->dims == std::vector<std::uint64_t>{kHiddenSize, kLinearQkvMixedValues} &&
        layer5_tensors.conv_tensor->dims == std::vector<std::uint64_t>{kLinearConvKernelSize, kLinearQkvMixedValues};
    const bool layer5_ok =
        layer5_shapes_ok &&
        layer5_gpu_cpu_ok &&
        layer5_conv_state_ok &&
        layer5_oracle_policy_ok &&
        layer5_rms_gpu.timing.rmsnorm_min_us > 0.0 &&
        layer5_q6_gpu.qkv_min_us > 0.0 &&
        layer5_q6_gpu.conv_min_us > 0.0 &&
        layer5_rms_gpu.device_name.find(args.device_substring) != std::string::npos &&
        layer5_q6_gpu.device_name.find(args.device_substring) != std::string::npos;
    const bool layer2_timing_positive =
''',
  )
  cpp = replace_once(
      cpp,
      '''        layer3_ok &&
        layer4_ok &&
        layer2_timing_positive &&
''',
      '''        layer3_ok &&
        layer4_ok &&
        layer5_ok &&
        layer2_timing_positive &&
''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << "\\"layers\\":[" << layer0 << "," << layer1 << "," << layer2 << "," << layer3 << "," << layer4 << "],";
''',
      '''    std::cout << "\\"layers\\":[" << layer0 << "," << layer1 << "," << layer2 << "," << layer3 << "," << layer4 << "," << layer5 << "],";
''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << "\\"layer9_conv_state_boundary\\":\\"captured_conv_state\\",";''',
      '''    std::cout << "\\"layer9_conv_state_boundary\\":\\"captured_conv_state\\",";
    std::cout << "\\"layer10_residual_input_boundary\\":\\"live_gpu_l_out_9\\",";
    std::cout << "\\"layer10_qkv_tensor_type\\":\\"" << JsonEscape(iq36::ggml_type_name(layer5_tensors.qkv_tensor->type)) << "\\",";
    std::cout << "\\"layer10_conv_state_boundary\\":\\"captured_conv_state\\",";''',
  )
  cpp = replace_once(
      cpp,
      '''                  tail_gpu.program_build_ms + layer3_run.program_build_ms +
                  layer4_run.program_build_ms)
''',
      '''                  tail_gpu.program_build_ms + layer3_run.program_build_ms +
                  layer4_run.program_build_ms + layer5_rms_gpu.program_build_ms +
                  layer5_q6_gpu.program_build_ms)
''',
  )
  cpp = replace_once(
      cpp,
      '''                            tail_gpu.build_log + layer3_run.build_log +
                            layer4_run.build_log)
''',
      '''                            tail_gpu.build_log + layer3_run.build_log +
                            layer4_run.build_log + layer5_rms_gpu.build_log +
                            layer5_q6_gpu.build_log)
''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << "\\"resident_layer9_state_input_kernel_sum_min_us\\":"
              << layer4_state_input_sum_min << ",";
    std::cout << "\\"resident_layer5_6_7_8_to_layer9_state_input_kernel_sum_min_us\\":"
''',
      '''    const double layer5_state_input_sum_min =
        layer5_rms_gpu.timing.rmsnorm_min_us + layer5_q6_gpu.qkv_min_us +
        layer5_q6_gpu.conv_min_us;
    std::cout << "\\"resident_layer9_state_input_kernel_sum_min_us\\":"
              << layer4_state_input_sum_min << ",";
    std::cout << "\\"resident_layer10_state_input_kernel_sum_min_us\\":"
              << layer5_state_input_sum_min << ",";
    std::cout << "\\"resident_layer5_6_7_8_to_layer9_state_input_kernel_sum_min_us\\":"
''',
  )
  cpp = replace_once(
      cpp,
      '''    WritePrefixedCompareGroups("l4", layer4_run.comparisons, &first_compare);
    std::cout << ",\\"l2_k_raw\\":{\\"gpu_vs_cpu\\":";
''',
      '''    WritePrefixedCompareGroups("l4", layer4_run.comparisons, &first_compare);
    std::cout << ",";
    WriteNamedCompareGroups(layer5_state_input_groups);
    std::cout << ",\\"l2_k_raw\\":{\\"gpu_vs_cpu\\":";
''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << "},\\"l4_recurrent_state\\":{\\"gpu_vs_cpu\\":";
    WriteCompare(layer4_run.recurrent_state_gpu_vs_cpu);
    std::cout << "}";
    std::cout << "},\\"checks\\":{";
''',
      '''    std::cout << "},\\"l4_recurrent_state\\":{\\"gpu_vs_cpu\\":";
    WriteCompare(layer4_run.recurrent_state_gpu_vs_cpu);
    std::cout << "},\\"l5_conv_state_after\\":{\\"gpu_vs_cpu\\":";
    WriteCompare(iq36::compare_vectors(layer5_q6_gpu.conv_state_after,
                                       layer5_native_conv.conv_state,
                                       kMismatchThreshold));
    std::cout << "}";
    std::cout << "},\\"checks\\":{";
''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << "\\"layer9_state_input_handoff_matches\\":"
              << (layer4_ok ? "true" : "false") << ",";
    std::cout << "\\"history_kv_state_payloads_present\\":"
''',
      '''    std::cout << "\\"layer9_state_input_handoff_matches\\":"
              << (layer4_ok ? "true" : "false") << ",";
    std::cout << "\\"layer10_residual_input_from_layer9_live_gpu_lout\\":true,";
    std::cout << "\\"layer10_q6_qkv_boundary\\":"
              << (layer5_tensors.qkv_tensor->type == 14 ? "true" : "false") << ",";
    std::cout << "\\"layer10_conv_state_input_boundary\\":true,";
    std::cout << "\\"layer10_gpu_cpu_matches_native\\":"
              << (layer5_gpu_cpu_ok && layer5_conv_state_ok ? "true" : "false") << ",";
    std::cout << "\\"layer10_state_input_oracle_policy_matches\\":"
              << (layer5_oracle_policy_ok ? "true" : "false") << ",";
    std::cout << "\\"layer10_state_input_handoff_matches\\":"
              << (layer5_ok ? "true" : "false") << ",";
    std::cout << "\\"history_kv_state_payloads_present\\":"
''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << "\\"gpu_event_timing_positive\\":"
              << (layer0_run.timing_positive && layer1_run.timing_positive &&
                  layer2_timing_positive && layer3_run.timing_positive &&
                  layer4_run.timing_positive ? "true" : "false") << ",";
''',
      '''    std::cout << "\\"gpu_event_timing_positive\\":"
              << (layer0_run.timing_positive && layer1_run.timing_positive &&
                  layer2_timing_positive && layer3_run.timing_positive &&
                  layer4_run.timing_positive && layer5_ok ? "true" : "false") << ",";
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


def resident_fields_ok(probe: dict[str, Any] | None, expected_invocations: int) -> bool:
  return (
      isinstance(probe, dict)
      and probe.get("resident_api") == "two_linear_layer_to_full_attention_v_q6_live_ffn_lout_layer10_q6_state_input_load_once_run_many"
      and probe.get("resident_load_count") == 1
      and probe.get("resident_shell_invocations") == expected_invocations
      and probe.get("layer10_residual_input_boundary") == "live_gpu_l_out_9"
      and probe.get("layer10_qkv_tensor_type") == "Q6_K"
      and probe.get("layer10_conv_state_boundary") == "captured_conv_state"
      and PRECONV.nested_bool(probe, "checks", "resident_load_once")
      and PRECONV.nested_bool(probe, "checks", "resident_shell_invocations_positive")
  )


def write_summary(path: Path, payload: dict[str, Any]) -> None:
  probe = payload.get("probe") if isinstance(payload.get("probe"), dict) else {}
  timings = probe.get("timings", {}) if isinstance(probe, dict) else {}
  comparisons = probe.get("comparisons", {}) if isinstance(probe, dict) else {}
  lines = [
      "# GPU Resident Layer-10 Q6 State/Input Handoff Probe",
      "",
      f"- workstream: `{WORKSTREAM}`",
      f"- host: `{payload.get('host')}`",
      f"- model: `{payload.get('model')}`",
      f"- layers: `{payload.get('layers')}`",
      f"- required checks passed: `{str(payload.get('required_checks_passed')).lower()}`",
      "- speedup claims allowed: `false`",
      f"- resident API: `{probe.get('resident_api')}`",
      f"- resident shell invocations: `{probe.get('resident_shell_invocations')}`",
      f"- layer 10 residual input boundary: `{probe.get('layer10_residual_input_boundary')}`",
      f"- layer 10 qkv tensor type: `{probe.get('layer10_qkv_tensor_type')}`",
      f"- layer 10 conv state boundary: `{probe.get('layer10_conv_state_boundary')}`",
      "",
      "| output | comparison | max abs | RMSE |",
      "|---|---|---:|---:|",
  ]
  for name in (
      "l5_residual_input",
      "l5_attn_norm",
      "l5_linear_attn_qkv_mixed",
      "l5_conv_output_raw",
  ):
    group = comparisons.get(name, {}) if isinstance(comparisons, dict) else {}
    lane = group.get("gpu_vs_oracle", {}) if isinstance(group, dict) else {}
    lines.append(f"| {name} | gpu_vs_oracle | {lane.get('max_abs_diff')} | {lane.get('rmse')} |")
  lines += [
      "",
      "| kernel group | min us |",
      "|---|---:|",
      f"| layer10_state_input | {timings.get('resident_layer10_state_input_kernel_sum_min_us') if isinstance(timings, dict) else None} |",
      "",
      "This target-side process carries closed live GPU layer outputs through",
      "layer 9, then feeds live GPU `l_out-9` into layer 10 RMSNorm, Q6_K QKV,",
      "and F32 conv with captured layer-10 conv state. This is state/input",
      "evidence only; it is not a full layer-10 shell or throughput claim.",
      "",
  ]
  path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  args = parse_args()
  if args.resident_invocations <= 0:
    raise SystemExit("--resident-invocations must be positive")
  if args.layer != 5:
    raise SystemExit("--layer must be 5 for the layer-10 state/input handoff")

  layer0 = args.layer
  layer1 = args.layer + 1
  layer2 = args.layer + 2
  layer3 = args.layer + 3
  layer4 = args.layer + 4
  layer5 = args.layer + 5
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
  out_dir = args.out_dir or ROOT / f"output/gpu-resident-layer10-state-input-handoff-probe-{stamp}"
  out_dir.mkdir(parents=True, exist_ok=False)
  raw_dir = out_dir / "raw"
  raw_dir.mkdir()

  payloads0, conv0 = TWO.prefixed_payloads(layer0, conv0_path, "l0")
  payloads1, conv1 = TWO.prefixed_payloads(layer1, conv1_path, "l1")
  payloads2, conv2 = TWO.prefixed_payloads(layer3, conv2_path, "l3")
  payloads3, conv3 = TWO.prefixed_payloads(layer4, conv3_path, "l4")
  payloads4, conv4 = TWO.prefixed_payloads(layer5, conv4_path, "l5")
  full_payloads, all_history = L7_INPUT.resolve_full_attention_payloads(all_history_json, layer2)
  CORE.add_layer7_tail_payloads(full_payloads)
  L8.add_layer7_ffn_payloads(full_payloads)
  payloads = {**payloads0, **payloads1, **full_payloads, **payloads2, **payloads3, **payloads4}
  opencl_source = OPENCL_SOURCE.read_text(encoding="utf-8")
  opencl_source_hash = iq36_local.sha256_file(OPENCL_SOURCE)
  embedded_opencl_hash = hashlib.sha256(
      (
          opencl_source
          + PRECONV.POSTCONV.ATTENTION.ATTENTION_EXTRA_OPENCL
          + CORE.FULL_ATTN_CORE_EXTRA_OPENCL
      ).encode("utf-8")
  ).hexdigest()
  local_cpp = out_dir / "gpu_resident_layer10_state_input_handoff_probe.cpp"
  local_cpp.write_text(layer10_state_input_probe_cpp(opencl_source), encoding="utf-8")

  remote_dir = f"{args.remote_root.rstrip('/')}/gpu-resident-layer10-state-input-handoff-probe-{stamp}"
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
            f"{remote_dir}/tests/gpu_resident_layer10_state_input_handoff_probe.cpp",
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

  executable = f"{remote_dir}/build/iq36-gpu-resident-layer10-state-input-handoff-probe"
  compile_cmd = " && ".join([
      f"source {shlex.quote(args.env_script)} >/dev/null 2>&1",
      (
          "g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic "
          f"-I {shlex.quote(remote_dir + '/include')} "
          f"{shlex.quote(remote_dir + '/src/gguf_loader.cpp')} "
          f"{shlex.quote(remote_dir + '/src/gpu_q4x8_matvec.cpp')} "
          f"{shlex.quote(remote_dir + '/tests/gpu_resident_layer10_state_input_handoff_probe.cpp')} "
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
      {"name": "layer9_state_input_handoff_matches", "pass": PRECONV.nested_bool(probe, "checks", "layer9_state_input_handoff_matches")},
      {"name": "layer10_q6_qkv_boundary", "pass": PRECONV.nested_bool(probe, "checks", "layer10_q6_qkv_boundary")},
      {"name": "layer10_gpu_cpu_matches_native", "pass": PRECONV.nested_bool(probe, "checks", "layer10_gpu_cpu_matches_native")},
      {"name": "layer10_state_input_oracle_policy_matches", "pass": PRECONV.nested_bool(probe, "checks", "layer10_state_input_oracle_policy_matches")},
      {"name": "layer10_state_input_handoff_matches", "pass": PRECONV.nested_bool(probe, "checks", "layer10_state_input_handoff_matches")},
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
      "layer10_state_input_oracle_policy": CORE.FULL_ATTN_COMPARISON_THRESHOLDS,
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
      "all_history": all_history,
      "payloads": slim_payloads,
      "layers": [layer0, layer1, layer2, layer3, layer4, layer5],
      "resident_invocations": args.resident_invocations,
      "opencl_source": str(OPENCL_SOURCE.relative_to(ROOT)),
      "opencl_source_sha256": opencl_source_hash,
      "embedded_opencl_source_sha256": embedded_opencl_hash,
      "comparison_thresholds": comparison_thresholds,
      "probe": probe,
      "checks": checks,
      "required_checks_passed": required_checks_passed,
      "speedup_claims_allowed": False,
  }
  manifest = {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "created_at": created_at,
      "tool": "tools/intel-qwen36-gpu-resident-layer10-state-input-handoff-probe.py",
      "artifact": str(out_dir),
      "host": args.host,
      "remote_dir": remote_dir,
      "layers": payload["layers"],
      "resident_invocations": args.resident_invocations,
      "conv_history_probes": payload["conv_history_probes"],
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
      "gpu_resident_layer10_state_input_handoff_probe",
      [
          ("required_checks_passed", required_checks_passed),
          ("resident_shell_invocations", args.resident_invocations),
          ("resident_layer10_state_input_kernel_sum_min_us", PRECONV.nested_number(timings, "resident_layer10_state_input_kernel_sum_min_us")),
          ("layer10_residual_input_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l5_residual_input", "gpu_vs_oracle", "max_abs_diff")),
          ("layer10_attn_norm_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l5_attn_norm", "gpu_vs_oracle", "max_abs_diff")),
          ("layer10_qkv_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l5_linear_attn_qkv_mixed", "gpu_vs_oracle", "max_abs_diff")),
          ("layer10_conv_output_raw_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l5_conv_output_raw", "gpu_vs_oracle", "max_abs_diff")),
      ],
  )
  write_summary(out_dir / "summary.md", payload)
  print(out_dir)
  return 0 if required_checks_passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
