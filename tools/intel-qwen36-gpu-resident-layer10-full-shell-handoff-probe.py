#!/usr/bin/env python3
"""Run the resident GPU layer-10 Q6 full-shell handoff probe."""

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
SCHEMA_VERSION = "intel-qwen36-gpu-resident-layer10-full-shell-handoff-probe-v0"
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


Q6_QKV_PRECONV_CPP = r'''

struct Q6QkvConvPreconvRun {
  std::vector<float> qkv_mixed;
  std::vector<float> conv_output_raw;
  std::vector<float> conv_state_after;
  double qkv_min_us = 0.0;
  double qkv_mean_us = 0.0;
  double conv_min_us = 0.0;
  double conv_mean_us = 0.0;
  std::string platform_name;
  std::string device_name;
  std::string build_log;
  double program_build_ms = 0.0;
};

Q6QkvConvPreconvRun RunQ6QkvThenConvPreconv(
    const std::vector<std::uint8_t>& qkv_raw,
    const std::vector<std::int8_t>& q8_qs,
    const std::vector<float>& q8_d,
    const std::vector<float>& conv_weights,
    const std::vector<float>& conv_state,
    const std::string& device_substring,
    int repeat) {
  Require(qkv_raw.size() ==
              static_cast<std::size_t>(kLinearQkvMixedValues *
                                       (kHiddenSize / 256) * kQ6KBlockBytes),
          "Q6 QKV raw byte size mismatch");
  Require(q8_qs.size() == kHiddenSize, "Q6 QKV q8 qs size mismatch");
  Require(q8_d.size() == static_cast<std::size_t>(kHiddenSize / 256),
          "Q6 QKV q8 d size mismatch");
  Require(conv_weights.size() == kLinearQkvMixedValues * kLinearConvKernelSize,
          "Q6 QKV conv weight size mismatch");
  Require(conv_state.size() == kLinearConvStateValues,
          "Q6 QKV conv state size mismatch");

  Q6QkvConvPreconvRun run;
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
  Check(err, "clCreateContext(Q6 QKV preconv)");
  cl_command_queue queue =
      api.clCreateCommandQueue(context, selected.device, kClQueueProfilingEnable, &err);
  Check(err, "clCreateCommandQueue(Q6 QKV preconv)");
  const char* source = kOpenClSource;
  const std::size_t source_len = std::strlen(kOpenClSource);
  cl_program program = api.clCreateProgramWithSource(context, 1, &source, &source_len, &err);
  Check(err, "clCreateProgramWithSource(Q6 QKV preconv)");
  const auto build_begin = std::chrono::steady_clock::now();
  err = api.clBuildProgram(program, 1, &selected.device, "", nullptr, nullptr);
  const auto build_end = std::chrono::steady_clock::now();
  run.program_build_ms =
      std::chrono::duration<double, std::milli>(build_end - build_begin).count();
  run.build_log = BuildLog(api, program, selected.device);
  Check(err, "clBuildProgram(Q6 QKV preconv)");
  cl_kernel q6_kernel = api.clCreateKernel(program, "q6k_selected_down_matvec_row", &err);
  Check(err, "clCreateKernel(Q6 QKV q6k_selected_down_matvec_row)");
  cl_kernel conv_kernel = api.clCreateKernel(program, "linear_attn_conv_f32", &err);
  Check(err, "clCreateKernel(Q6 QKV linear_attn_conv_f32)");

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
    Check(err, "clCreateBuffer(Q6 QKV raw)");
    q8_qs_buffer = api.clCreateBuffer(
        context, kClMemReadOnly, q8_qs.size() * sizeof(std::int8_t), nullptr, &err);
    Check(err, "clCreateBuffer(Q6 QKV q8 qs)");
    q8_d_buffer = api.clCreateBuffer(
        context, kClMemReadOnly, q8_d.size() * sizeof(float), nullptr, &err);
    Check(err, "clCreateBuffer(Q6 QKV q8 d)");
    qkv_buffer = api.clCreateBuffer(
        context, kClMemReadWriteLocal, run.qkv_mixed.size() * sizeof(float), nullptr, &err);
    Check(err, "clCreateBuffer(Q6 QKV output)");
    conv_weight_buffer = api.clCreateBuffer(
        context, kClMemReadOnly, conv_weights.size() * sizeof(float), nullptr, &err);
    Check(err, "clCreateBuffer(Q6 QKV conv weights)");
    conv_state_buffer = api.clCreateBuffer(
        context, kClMemReadOnly, conv_state.size() * sizeof(float), nullptr, &err);
    Check(err, "clCreateBuffer(Q6 QKV conv state)");
    conv_output_buffer = api.clCreateBuffer(
        context, kClMemWriteOnly, run.conv_output_raw.size() * sizeof(float), nullptr, &err);
    Check(err, "clCreateBuffer(Q6 QKV conv output)");
    next_state_buffer = api.clCreateBuffer(
        context, kClMemWriteOnly, run.conv_state_after.size() * sizeof(float), nullptr, &err);
    Check(err, "clCreateBuffer(Q6 QKV next state)");
    Check(api.clEnqueueWriteBuffer(queue, q6_buffer, kClTrue, 0,
                                   qkv_raw.size(), qkv_raw.data(), 0, nullptr, nullptr),
          "clEnqueueWriteBuffer(Q6 QKV raw)");
    Check(api.clEnqueueWriteBuffer(queue, q8_qs_buffer, kClTrue, 0,
                                   q8_qs.size() * sizeof(std::int8_t), q8_qs.data(),
                                   0, nullptr, nullptr),
          "clEnqueueWriteBuffer(Q6 QKV q8 qs)");
    Check(api.clEnqueueWriteBuffer(queue, q8_d_buffer, kClTrue, 0,
                                   q8_d.size() * sizeof(float), q8_d.data(),
                                   0, nullptr, nullptr),
          "clEnqueueWriteBuffer(Q6 QKV q8 d)");
    Check(api.clEnqueueWriteBuffer(queue, conv_weight_buffer, kClTrue, 0,
                                   conv_weights.size() * sizeof(float), conv_weights.data(),
                                   0, nullptr, nullptr),
          "clEnqueueWriteBuffer(Q6 QKV conv weights)");
    Check(api.clEnqueueWriteBuffer(queue, conv_state_buffer, kClTrue, 0,
                                   conv_state.size() * sizeof(float), conv_state.data(),
                                   0, nullptr, nullptr),
          "clEnqueueWriteBuffer(Q6 QKV conv state)");
    const cl_uint rows_arg = static_cast<cl_uint>(kLinearQkvMixedValues);
    const cl_uint blocks_arg = static_cast<cl_uint>(kHiddenSize / 256);
    const cl_uint conv_kernel_arg = static_cast<cl_uint>(kLinearConvKernelSize);
    Check(api.clSetKernelArg(q6_kernel, 0, sizeof(q6_buffer), &q6_buffer),
          "clSetKernelArg(Q6 QKV 0)");
    Check(api.clSetKernelArg(q6_kernel, 1, sizeof(q8_qs_buffer), &q8_qs_buffer),
          "clSetKernelArg(Q6 QKV 1)");
    Check(api.clSetKernelArg(q6_kernel, 2, sizeof(q8_d_buffer), &q8_d_buffer),
          "clSetKernelArg(Q6 QKV 2)");
    Check(api.clSetKernelArg(q6_kernel, 3, sizeof(rows_arg), &rows_arg),
          "clSetKernelArg(Q6 QKV 3)");
    Check(api.clSetKernelArg(q6_kernel, 4, sizeof(blocks_arg), &blocks_arg),
          "clSetKernelArg(Q6 QKV 4)");
    Check(api.clSetKernelArg(q6_kernel, 5, sizeof(qkv_buffer), &qkv_buffer),
          "clSetKernelArg(Q6 QKV 5)");
    Check(api.clSetKernelArg(conv_kernel, 0, sizeof(qkv_buffer), &qkv_buffer),
          "clSetKernelArg(Q6 QKV conv 0)");
    Check(api.clSetKernelArg(conv_kernel, 1, sizeof(conv_state_buffer), &conv_state_buffer),
          "clSetKernelArg(Q6 QKV conv 1)");
    Check(api.clSetKernelArg(conv_kernel, 2, sizeof(conv_weight_buffer), &conv_weight_buffer),
          "clSetKernelArg(Q6 QKV conv 2)");
    Check(api.clSetKernelArg(conv_kernel, 3, sizeof(rows_arg), &rows_arg),
          "clSetKernelArg(Q6 QKV conv 3)");
    Check(api.clSetKernelArg(conv_kernel, 4, sizeof(conv_kernel_arg), &conv_kernel_arg),
          "clSetKernelArg(Q6 QKV conv 4)");
    Check(api.clSetKernelArg(conv_kernel, 5, sizeof(conv_output_buffer), &conv_output_buffer),
          "clSetKernelArg(Q6 QKV conv 5)");
    Check(api.clSetKernelArg(conv_kernel, 6, sizeof(next_state_buffer), &next_state_buffer),
          "clSetKernelArg(Q6 QKV conv 6)");
    const std::size_t global = static_cast<std::size_t>(kLinearQkvMixedValues);
    run.qkv_min_us = std::numeric_limits<double>::infinity();
    run.conv_min_us = std::numeric_limits<double>::infinity();
    for (int i = 0; i < repeat; ++i) {
      cl_event q6_event = nullptr;
      cl_event conv_event = nullptr;
      Check(api.clEnqueueNDRangeKernel(queue, q6_kernel, 1, nullptr, &global, nullptr,
                                       0, nullptr, &q6_event),
            "clEnqueueNDRangeKernel(Q6 QKV)");
      Check(api.clEnqueueNDRangeKernel(queue, conv_kernel, 1, nullptr, &global, nullptr,
                                       0, nullptr, &conv_event),
            "clEnqueueNDRangeKernel(Q6 QKV conv)");
      Check(api.clFinish(queue), "clFinish(Q6 QKV preconv)");
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
    Check(api.clEnqueueReadBuffer(queue, qkv_buffer, kClTrue, 0,
                                  run.qkv_mixed.size() * sizeof(float),
                                  run.qkv_mixed.data(), 0, nullptr, nullptr),
          "clEnqueueReadBuffer(Q6 QKV)");
    Check(api.clEnqueueReadBuffer(queue, conv_output_buffer, kClTrue, 0,
                                  run.conv_output_raw.size() * sizeof(float),
                                  run.conv_output_raw.data(), 0, nullptr, nullptr),
          "clEnqueueReadBuffer(Q6 QKV conv output)");
    Check(api.clEnqueueReadBuffer(queue, next_state_buffer, kClTrue, 0,
                                  run.conv_state_after.size() * sizeof(float),
                                  run.conv_state_after.data(), 0, nullptr, nullptr),
          "clEnqueueReadBuffer(Q6 QKV conv state)");
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


def layer10_full_shell_probe_cpp(opencl_source: str) -> str:
  cpp = L9.layer9_state_input_probe_cpp(opencl_source)
  replace_once = L8.replace_once
  cpp = replace_once(cpp, "#include <iomanip>\n", "#include <iomanip>\n#include <limits>\n")
  replacements = {
      L9.SCHEMA_VERSION: SCHEMA_VERSION,
      "two_linear_layer_to_full_attention_v_q6_live_ffn_lout_layer9_state_input_load_once_run_many":
          "two_linear_layer_to_full_attention_v_q6_live_ffn_lout_layer10_full_shell_load_once_run_many",
      "layer9 state/input handoff probe expects --layer 5":
          "layer10 full-shell handoff probe expects --layer 5",
      "qkv tensor must be Q4_K": "qkv tensor must be Q4_K or Q6_K",
  }
  for old, new in replacements.items():
    cpp = replace_once(cpp, old, new)
  cpp = replace_once(
      cpp,
      "  Require(qkv_tensor.type == 12, \"qkv tensor must be Q4_K or Q6_K\");",
      "  Require(qkv_tensor.type == 12 || qkv_tensor.type == 14,\n"
      "          \"qkv tensor must be Q4_K or Q6_K\");",
  )
  cpp = replace_once(
      cpp,
      "      t.qkv_tensor->type == 12 &&\n      t.alpha_tensor->type == 12 &&",
      "      (t.qkv_tensor->type == 12 || t.qkv_tensor->type == 14) &&\n"
      "      t.alpha_tensor->type == 12 &&",
  )
  cpp = replace_once(
      cpp,
      "      t.selected_gate_up_tensor->type == 12 &&\n"
      "      t.selected_down_tensor->type == 12 &&",
      "      t.selected_gate_up_tensor->type == 12 &&\n"
      "      (t.selected_down_tensor->type == 12 || t.selected_down_tensor->type == 14) &&",
  )
  cpp = replace_once(
      cpp,
      "\niq36::GpuQ4X8MatvecRun RunProjectionFromTensor(",
      Q6_QKV_PRECONV_CPP + "\n\niq36::GpuQ4X8MatvecRun RunProjectionFromTensor(",
  )
  cpp = replace_once(
      cpp,
      '''  const auto qkv_raw = ReadTensorBytes(model, qkv_tensor);
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
''',
      '''  const auto qkv_raw = ReadTensorBytes(model, qkv_tensor);
  const auto conv_weights = ReadF32TensorPayload(
      model, conv_tensor,
      static_cast<std::size_t>(kLinearQkvMixedValues * kLinearConvKernelSize));
  if (qkv_tensor.type == 12) {
    const auto qkv_packed =
        iq36::PackQ4Kx8(qkv_raw, kLinearQkvMixedValues, kHiddenSize / 256);
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
  } else {
    const auto q6_conv = RunQ6QkvThenConvPreconv(
        qkv_raw, q8.qs, q8.d, conv_weights, conv_state,
        device_substring, repeat);
    run.qkv_mixed = q6_conv.qkv_mixed;
    run.conv_output_raw = q6_conv.conv_output_raw;
    run.conv_state_after = q6_conv.conv_state_after;
    run.timing.qkv_min_us = q6_conv.qkv_min_us;
    run.timing.qkv_mean_us = q6_conv.qkv_mean_us;
    run.timing.conv_min_us = q6_conv.conv_min_us;
    run.timing.conv_mean_us = q6_conv.conv_mean_us;
    run.program_build_ms += q6_conv.program_build_ms;
    run.build_log += q6_conv.build_log;
    run.device_name = q6_conv.device_name;
  }
''',
  )
  cpp = replace_once(
      cpp,
      '''    const int layer4 = args.layer + 4;
    Require(layer0 == 5 && layer1 == 6 && layer2 == 7 && layer3 == 8 && layer4 == 9,
            "layer10 full-shell handoff probe expects --layer 5");
''',
      '''    const int layer4 = args.layer + 4;
    const int layer5 = args.layer + 5;
    Require(layer0 == 5 && layer1 == 6 && layer2 == 7 && layer3 == 8 && layer4 == 9 && layer5 == 10,
            "layer10 full-shell handoff probe expects --layer 5");
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
    const auto layer5_run = RunResidentLinearLayerShell(
        args, index, layer5_tensors, layer5_oracle,
        layer4_run.gpu_layer_output, layer4_run.gpu_layer_output, rms_norm_epsilon);

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
    const auto find_l5_group =
        [&](const std::string& name) -> const NamedCompareGroup& {
          const auto found = std::find_if(
              layer5_run.comparisons.begin(),
              layer5_run.comparisons.end(),
              [&](const NamedCompareGroup& group) {
                return group.name == name;
              });
          Require(found != layer5_run.comparisons.end(),
                  "layer10 comparison missing: " + name);
          return *found;
        };
    const auto& layer5_residual_input = find_l5_group("residual_input");
    const auto& layer5_attn_norm = find_l5_group("attn_norm");
    const auto& layer5_qkv = find_l5_group("linear_attn_qkv_mixed");
    const auto& layer5_conv_output_raw = find_l5_group("conv_output_raw");
    bool layer5_gpu_cpu_ok = true;
    for (const auto& group : layer5_run.comparisons) {
      layer5_gpu_cpu_ok = layer5_gpu_cpu_ok && ComparePassed(group.gpu_vs_cpu);
    }
    layer5_gpu_cpu_ok =
        layer5_gpu_cpu_ok &&
        ComparePassed(layer5_run.conv_state_after_gpu_vs_cpu) &&
        ComparePassed(layer5_run.recurrent_state_gpu_vs_cpu);
    const bool layer5_state_input_oracle_policy_ok =
        ComparePassedFullAttentionComponent(layer5_residual_input.gpu_vs_oracle) &&
        ComparePassedFullAttentionComponent(layer5_attn_norm.gpu_vs_oracle) &&
        ComparePassedFullAttentionComponent(layer5_qkv.gpu_vs_oracle) &&
        ComparePassedFullAttentionComponent(layer5_conv_output_raw.gpu_vs_oracle);
    const bool layer5_shapes_ok = ShapesPassed(layer5_run.shape_checks);
    const bool layer5_q6_qkv_boundary = layer5_tensors.qkv_tensor->type == 14;
    const bool layer5_q6_down_boundary =
        layer5_tensors.selected_down_tensor->type == 14 &&
        layer5_tensors.shared_down_tensor->type == 14;
    const bool layer5_ok =
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
      '''    const double layer4_state_input_sum_min =
        layer4_run.timing.layer_input_rmsnorm_min_us +
        layer4_run.timing.preconv_to_postconv_kernel_sum_min_us;
''',
      '''    const double layer4_state_input_sum_min =
        layer4_run.timing.layer_input_rmsnorm_min_us +
        layer4_run.timing.preconv_to_postconv_kernel_sum_min_us;
    const double layer5_state_input_sum_min =
        layer5_run.timing.layer_input_rmsnorm_min_us +
        layer5_run.timing.preconv_to_postconv_kernel_sum_min_us;
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
      '''    std::cout << "\\"layer9_conv_state_boundary\\":\\"captured_conv_state\\",";
    std::cout << "\\"full_attn_core_input_boundary\\":\\"captured_rope_kv_payloads\\",";''',
      '''    std::cout << "\\"layer9_conv_state_boundary\\":\\"captured_conv_state\\",";
    std::cout << "\\"layer10_residual_input_boundary\\":\\"live_gpu_l_out_9\\",";
    std::cout << "\\"layer10_qkv_tensor_type\\":\\""
              << JsonEscape(iq36::ggml_type_name(layer5_tensors.qkv_tensor->type)) << "\\",";
    std::cout << "\\"layer10_selected_down_tensor_type\\":\\""
              << JsonEscape(iq36::ggml_type_name(layer5_tensors.selected_down_tensor->type)) << "\\",";
    std::cout << "\\"layer10_shared_down_tensor_type\\":\\""
              << JsonEscape(iq36::ggml_type_name(layer5_tensors.shared_down_tensor->type)) << "\\",";
    std::cout << "\\"layer10_conv_state_boundary\\":\\"captured_conv_state\\",";
    std::cout << "\\"full_attn_core_input_boundary\\":\\"captured_rope_kv_payloads\\",";''',
  )
  cpp = replace_once(
      cpp,
      '''                  selected_gpu.program_build_ms + shared_gpu.program_build_ms +
                  tail_gpu.program_build_ms + layer3_run.program_build_ms +
                  layer4_run.program_build_ms)
''',
      '''                  selected_gpu.program_build_ms + shared_gpu.program_build_ms +
                  tail_gpu.program_build_ms + layer3_run.program_build_ms +
                  layer4_run.program_build_ms + layer5_run.program_build_ms)
''',
  )
  cpp = replace_once(
      cpp,
      '''                            selected_gpu.build_log + shared_gpu.build_log +
                            tail_gpu.build_log + layer3_run.build_log +
                            layer4_run.build_log)
''',
      '''                            selected_gpu.build_log + shared_gpu.build_log +
                            tail_gpu.build_log + layer3_run.build_log +
                            layer4_run.build_log + layer5_run.build_log)
''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << "\\"resident_layer9_state_input_kernel_sum_min_us\\":"
              << layer4_state_input_sum_min << ",";
    std::cout << "\\"resident_layer5_6_7_8_to_layer9_state_input_kernel_sum_min_us\\":"
              << (two_layer_kernel_sum_min + layer2_attention_sum_min +
                  layer2_ffn_sum_min + layer3_run.timing.layer_kernel_sum_min_us +
                  layer4_state_input_sum_min);
''',
      '''    std::cout << "\\"resident_layer9_state_input_kernel_sum_min_us\\":"
              << layer4_state_input_sum_min << ",";
    std::cout << "\\"resident_layer10_state_input_kernel_sum_min_us\\":"
              << layer5_state_input_sum_min << ",";
    std::cout << "\\"resident_layer10_full_shell_kernel_sum_min_us\\":"
              << layer5_run.timing.layer_kernel_sum_min_us << ",";
    std::cout << "\\"resident_layer5_6_7_8_9_10_full_shell_kernel_sum_min_us\\":"
              << (two_layer_kernel_sum_min + layer2_attention_sum_min +
                  layer2_ffn_sum_min + layer3_run.timing.layer_kernel_sum_min_us +
                  layer4_run.timing.layer_kernel_sum_min_us +
                  layer5_run.timing.layer_kernel_sum_min_us);
''',
  )
  cpp = replace_once(
      cpp,
      '''    WritePrefixedCompareGroups("l3", layer3_run.comparisons, &first_compare);
    WritePrefixedCompareGroups("l4", layer4_run.comparisons, &first_compare);
    std::cout << ",\\"l2_k_raw\\":{\\"gpu_vs_cpu\\":";
''',
      '''    WritePrefixedCompareGroups("l3", layer3_run.comparisons, &first_compare);
    WritePrefixedCompareGroups("l4", layer4_run.comparisons, &first_compare);
    WritePrefixedCompareGroups("l5", layer5_run.comparisons, &first_compare);
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
    WriteCompare(layer5_run.conv_state_after_gpu_vs_cpu);
    std::cout << "},\\"l5_recurrent_state\\":{\\"gpu_vs_cpu\\":";
    WriteCompare(layer5_run.recurrent_state_gpu_vs_cpu);
    std::cout << "}";
    std::cout << "},\\"checks\\":{";
''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << ",\\"layer4\\":";
    WriteLayerChecks(layer4_run);
    std::cout << ",\\"layer1_residual_input_matches_oracle\\":"
''',
      '''    std::cout << ",\\"layer4\\":";
    WriteLayerChecks(layer4_run);
    std::cout << ",\\"layer5\\":";
    WriteLayerChecks(layer5_run);
    std::cout << ",\\"layer1_residual_input_matches_oracle\\":"
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
              << (layer5_q6_qkv_boundary ? "true" : "false") << ",";
    std::cout << "\\"layer10_q6_down_boundary\\":"
              << (layer5_q6_down_boundary ? "true" : "false") << ",";
    std::cout << "\\"layer10_conv_state_input_boundary\\":true,";
    std::cout << "\\"layer10_gpu_cpu_matches_native\\":"
              << (layer5_gpu_cpu_ok ? "true" : "false") << ",";
    std::cout << "\\"layer10_state_input_oracle_policy_matches\\":"
              << (layer5_state_input_oracle_policy_ok ? "true" : "false") << ",";
    std::cout << "\\"layer10_full_shell_matches_oracle\\":"
              << (layer5_run.comparisons_passed ? "true" : "false") << ",";
    std::cout << "\\"layer10_full_shell_handoff_matches\\":"
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
                  layer4_run.timing_positive && layer5_run.timing_positive ? "true" : "false") << ",";
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
      and probe.get("resident_api")
      == "two_linear_layer_to_full_attention_v_q6_live_ffn_lout_layer10_full_shell_load_once_run_many"
      and probe.get("resident_load_count") == 1
      and probe.get("resident_shell_invocations") == expected_invocations
      and probe.get("layer10_residual_input_boundary") == "live_gpu_l_out_9"
      and probe.get("layer10_qkv_tensor_type") == "Q6_K"
      and probe.get("layer10_selected_down_tensor_type") == "Q6_K"
      and probe.get("layer10_shared_down_tensor_type") == "Q6_K"
      and probe.get("layer10_conv_state_boundary") == "captured_conv_state"
      and PRECONV.nested_bool(probe, "checks", "resident_load_once")
      and PRECONV.nested_bool(probe, "checks", "resident_shell_invocations_positive")
  )


def comparison_passed(probe: dict[str, Any] | None, name: str, lane: str) -> bool:
  return CORE.comparison_passed(probe, name, lane)


def write_summary(path: Path, payload: dict[str, Any]) -> None:
  probe = payload.get("probe") if isinstance(payload.get("probe"), dict) else {}
  timings = probe.get("timings", {}) if isinstance(probe, dict) else {}
  comparisons = probe.get("comparisons", {}) if isinstance(probe, dict) else {}
  lines = [
      "# GPU Resident Layer-10 Q6 Full-Shell Handoff Probe",
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
      f"- layer 10 selected/shared down tensor types: "
      f"`{probe.get('layer10_selected_down_tensor_type')}` / "
      f"`{probe.get('layer10_shared_down_tensor_type')}`",
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
      "l5_final_output",
      "l5_linear_attn_out",
      "l5_attn_residual",
      "l5_selected_down",
      "l5_shared_down",
      "l5_ffn_out",
      "l5_layer_output",
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
      f"| layer10_state_input | {timings.get('resident_layer10_state_input_kernel_sum_min_us') if isinstance(timings, dict) else None} |",
      f"| layer10_full_shell | {timings.get('resident_layer10_full_shell_kernel_sum_min_us') if isinstance(timings, dict) else None} |",
      "",
      "This target-side process carries closed live GPU layer outputs through",
      "layer 9, then runs the full layer-10 linear-attention shell from live",
      "GPU `l_out-9` with Q6_K QKV and Q6_K FFN down tensors. This is",
      "captured single-token shell evidence only; it is not a throughput claim.",
      "",
  ]
  path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  args = parse_args()
  if args.resident_invocations <= 0:
    raise SystemExit("--resident-invocations must be positive")
  if args.layer != 5:
    raise SystemExit("--layer must be 5 for the layer-10 full-shell handoff")

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
  out_dir = args.out_dir or ROOT / f"output/gpu-resident-layer10-full-shell-handoff-probe-{stamp}"
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
  local_cpp = out_dir / "gpu_resident_layer10_full_shell_handoff_probe.cpp"
  local_cpp.write_text(layer10_full_shell_probe_cpp(opencl_source), encoding="utf-8")

  remote_dir = f"{args.remote_root.rstrip('/')}/gpu-resident-layer10-full-shell-handoff-probe-{stamp}"
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
            f"{remote_dir}/tests/gpu_resident_layer10_full_shell_handoff_probe.cpp",
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

  executable = f"{remote_dir}/build/iq36-gpu-resident-layer10-full-shell-handoff-probe"
  compile_cmd = " && ".join([
      f"source {shlex.quote(args.env_script)} >/dev/null 2>&1",
      (
          "g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic "
          f"-I {shlex.quote(remote_dir + '/include')} "
          f"{shlex.quote(remote_dir + '/src/gguf_loader.cpp')} "
          f"{shlex.quote(remote_dir + '/src/gpu_q4x8_matvec.cpp')} "
          f"{shlex.quote(remote_dir + '/tests/gpu_resident_layer10_full_shell_handoff_probe.cpp')} "
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
      {"name": "layer10_q6_down_boundary", "pass": PRECONV.nested_bool(probe, "checks", "layer10_q6_down_boundary")},
      {"name": "layer10_gpu_cpu_matches_native", "pass": PRECONV.nested_bool(probe, "checks", "layer10_gpu_cpu_matches_native")},
      {"name": "layer10_state_input_oracle_policy_matches", "pass": PRECONV.nested_bool(probe, "checks", "layer10_state_input_oracle_policy_matches")},
      {"name": "layer10_full_shell_matches_oracle", "pass": PRECONV.nested_bool(probe, "checks", "layer10_full_shell_matches_oracle")},
      {"name": "layer10_full_shell_handoff_matches", "pass": PRECONV.nested_bool(probe, "checks", "layer10_full_shell_handoff_matches")},
      {"name": "layer10_layer_checks_comparisons_passed", "pass": PRECONV.nested_bool(probe, "checks", "layer5", "comparisons_passed")},
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
      "layer10_full_shell": CORE.STRICT_COMPARISON_THRESHOLDS,
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
      "tool": "tools/intel-qwen36-gpu-resident-layer10-full-shell-handoff-probe.py",
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
      "gpu_resident_layer10_full_shell_handoff_probe",
      [
          ("required_checks_passed", required_checks_passed),
          ("resident_shell_invocations", args.resident_invocations),
          ("resident_layer10_state_input_kernel_sum_min_us", PRECONV.nested_number(timings, "resident_layer10_state_input_kernel_sum_min_us")),
          ("resident_layer10_full_shell_kernel_sum_min_us", PRECONV.nested_number(timings, "resident_layer10_full_shell_kernel_sum_min_us")),
          ("layer10_qkv_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l5_linear_attn_qkv_mixed", "gpu_vs_oracle", "max_abs_diff")),
          ("layer10_final_output_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l5_final_output", "gpu_vs_oracle", "max_abs_diff")),
          ("layer10_selected_down_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l5_selected_down", "gpu_vs_oracle", "max_abs_diff")),
          ("layer10_shared_down_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l5_shared_down", "gpu_vs_oracle", "max_abs_diff")),
          ("layer10_layer_output_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l5_layer_output", "gpu_vs_oracle", "max_abs_diff")),
      ],
  )
  write_summary(out_dir / "summary.md", payload)
  print(out_dir)
  return 0 if required_checks_passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
