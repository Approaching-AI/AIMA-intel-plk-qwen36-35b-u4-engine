#!/usr/bin/env python3
"""Run the resident GPU shared-expert FFN shell handoff probe."""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import shlex
from pathlib import Path
from types import ModuleType
from typing import Any

import iq36_local


ROOT = Path(__file__).resolve().parents[1]
SELECTED_TOOL = Path(__file__).with_name("intel-qwen36-gpu-resident-selected-ffn-shell-probe.py")
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA_VERSION = "intel-qwen36-gpu-resident-shared-ffn-shell-probe-v0"
DEFAULT_HOST = "local"
DEFAULT_MODEL = "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf"
DEFAULT_ENV_SCRIPT = "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh"
DEFAULT_REMOTE_ROOT = "/home/intel/intel-qwen36-gpu"
DEFAULT_ORACLE_BUNDLE = ROOT / "oracle/r0-oracle-bundle-20260627T060028Z"
OPENCL_SOURCE = ROOT / "engine/gpu/opencl/q4x8_matvec.cl"


def load_selected_tool() -> ModuleType:
  spec = importlib.util.spec_from_file_location("iq36_selected_ffn_shell_probe", SELECTED_TOOL)
  if spec is None or spec.loader is None:
    raise SystemExit(f"failed to load selected FFN shell tool: {SELECTED_TOOL}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


SELECTED = load_selected_tool()


SHARED_HELPERS_CPP = r'''
constexpr int kSharedGateUpValueCount = kIntermediateSize * 2;
constexpr int kSharedSwiGluValueCount = kIntermediateSize;
constexpr int kQ6KBlockBytes = 210;

struct SharedFfnTiming {
  double gate_min_us = 0.0;
  double gate_mean_us = 0.0;
  double up_min_us = 0.0;
  double up_mean_us = 0.0;
  double swiglu_min_us = 0.0;
  double swiglu_mean_us = 0.0;
  double down_min_us = 0.0;
  double down_mean_us = 0.0;
  double host_q8_bridge_us = 0.0;
  double shared_ffn_kernel_sum_min_us = 0.0;
  double shared_ffn_kernel_sum_mean_us = 0.0;
  std::uint64_t gate_global_work_items = 0;
  std::uint64_t up_global_work_items = 0;
  std::uint64_t swiglu_global_work_items = 0;
  std::uint64_t down_global_work_items = 0;
  std::uint64_t down_kernel_launches = 0;
  std::uint64_t host_q8_bridge_count = 0;
};

struct SharedFfnRun {
  std::vector<float> gate;
  std::vector<float> up;
  std::vector<float> gate_up;
  std::vector<float> swiglu;
  std::vector<float> down;
  SharedFfnTiming timing;
  std::string platform_name;
  std::string device_name;
  std::string build_log;
  double program_build_ms = 0.0;
};

struct SharedSwiGluRun {
  std::vector<float> output;
  double min_us = 0.0;
  double mean_us = 0.0;
  std::uint64_t global_work_items = 0;
  std::string build_log;
  double program_build_ms = 0.0;
};

SharedSwiGluRun RunGpuSharedSwiGlu(const std::vector<float>& gate_up,
                                   const std::string& device_substring,
                                   int repeat) {
  Require(gate_up.size() == kSharedGateUpValueCount, "shared gate-up size mismatch");
  OpenClApi api;
  SharedSwiGluRun run;
  run.output.assign(kSharedSwiGluValueCount, 0.0f);
  const auto selected = SelectDevice(api, device_substring);
  cl_int err = kClSuccess;
  cl_context context = api.clCreateContext(nullptr, 1, &selected.device, nullptr, nullptr, &err);
  Check(err, "clCreateContext(shared swiglu)");
  cl_command_queue queue = api.clCreateCommandQueue(context, selected.device, kClQueueProfilingEnable, &err);
  Check(err, "clCreateCommandQueue(shared swiglu)");
  const char* source = kOpenClSource;
  const std::size_t source_len = std::strlen(kOpenClSource);
  cl_program program = api.clCreateProgramWithSource(context, 1, &source, &source_len, &err);
  Check(err, "clCreateProgramWithSource(shared swiglu)");
  const auto build_begin = std::chrono::steady_clock::now();
  err = api.clBuildProgram(program, 1, &selected.device, "", nullptr, nullptr);
  const auto build_end = std::chrono::steady_clock::now();
  run.program_build_ms = std::chrono::duration<double, std::milli>(build_end - build_begin).count();
  run.build_log = BuildLog(api, program, selected.device);
  Check(err, "clBuildProgram(shared swiglu)");
  cl_kernel kernel = api.clCreateKernel(program, "ffn_moe_swiglu_f32", &err);
  Check(err, "clCreateKernel(ffn_moe_swiglu_f32)");
  cl_mem gate_up_buffer = nullptr;
  cl_mem output_buffer = nullptr;
  try {
    gate_up_buffer = api.clCreateBuffer(context, kClMemReadOnly,
                                        gate_up.size() * sizeof(float), nullptr, &err);
    Check(err, "clCreateBuffer(shared swiglu gate_up)");
    output_buffer = api.clCreateBuffer(context, kClMemWriteOnly,
                                       run.output.size() * sizeof(float), nullptr, &err);
    Check(err, "clCreateBuffer(shared swiglu output)");
    Check(api.clEnqueueWriteBuffer(queue, gate_up_buffer, kClTrue, 0,
                                   gate_up.size() * sizeof(float), gate_up.data(),
                                   0, nullptr, nullptr),
          "clEnqueueWriteBuffer(shared swiglu gate_up)");
    const cl_uint intermediate_arg = static_cast<cl_uint>(kIntermediateSize);
    const cl_uint expert_arg = 1;
    Check(api.clSetKernelArg(kernel, 0, sizeof(gate_up_buffer), &gate_up_buffer), "clSetKernelArg(shared swiglu 0)");
    Check(api.clSetKernelArg(kernel, 1, sizeof(intermediate_arg), &intermediate_arg), "clSetKernelArg(shared swiglu 1)");
    Check(api.clSetKernelArg(kernel, 2, sizeof(expert_arg), &expert_arg), "clSetKernelArg(shared swiglu 2)");
    Check(api.clSetKernelArg(kernel, 3, sizeof(output_buffer), &output_buffer), "clSetKernelArg(shared swiglu 3)");
    const std::size_t global = static_cast<std::size_t>(kSharedSwiGluValueCount);
    std::vector<double> times;
    times.reserve(static_cast<std::size_t>(repeat));
    for (int i = 0; i < repeat; ++i) {
      cl_event event = nullptr;
      Check(api.clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global, nullptr, 0, nullptr, &event),
            "clEnqueueNDRangeKernel(ffn_moe_swiglu_f32 shared)");
      Check(api.clFinish(queue), "clFinish(shared swiglu)");
      times.push_back(EventUs(api, event));
      api.clReleaseEvent(event);
    }
    Check(api.clEnqueueReadBuffer(queue, output_buffer, kClTrue, 0,
                                  run.output.size() * sizeof(float),
                                  run.output.data(), 0, nullptr, nullptr),
          "clEnqueueReadBuffer(shared swiglu output)");
    run.min_us = Min(times);
    run.mean_us = Mean(times);
    run.global_work_items = kSharedSwiGluValueCount;
  } catch (...) {
    ReleaseMem(api, &output_buffer);
    ReleaseMem(api, &gate_up_buffer);
    api.clReleaseKernel(kernel);
    api.clReleaseProgram(program);
    api.clReleaseCommandQueue(queue);
    api.clReleaseContext(context);
    throw;
  }
  ReleaseMem(api, &output_buffer);
  ReleaseMem(api, &gate_up_buffer);
  api.clReleaseKernel(kernel);
  api.clReleaseProgram(program);
  api.clReleaseCommandQueue(queue);
  api.clReleaseContext(context);
  return run;
}

std::vector<float> RunGpuSharedDownQ6(
    const std::vector<std::uint8_t>& raw,
    const iq36::GpuQ8KInputPlanes& q8,
    std::uint64_t rows,
    std::uint64_t blocks_per_row,
    const std::string& device_substring,
    int repeat,
    SharedFfnTiming* timing,
    std::string* platform_name,
    std::string* device_name,
    std::string* build_log,
    double* program_build_ms) {
  Require(raw.size() == static_cast<std::size_t>(rows * blocks_per_row * kQ6KBlockBytes),
          "shared down Q6 raw byte size mismatch");
  OpenClApi api;
  std::vector<float> output(static_cast<std::size_t>(rows), 0.0f);
  const auto selected = SelectDevice(api, device_substring);
  *platform_name = selected.platform_name;
  *device_name = selected.device_name;
  cl_int err = kClSuccess;
  cl_context context = api.clCreateContext(nullptr, 1, &selected.device, nullptr, nullptr, &err);
  Check(err, "clCreateContext(shared down q6)");
  cl_command_queue queue = api.clCreateCommandQueue(context, selected.device, kClQueueProfilingEnable, &err);
  Check(err, "clCreateCommandQueue(shared down q6)");
  const char* source = kOpenClSource;
  const std::size_t source_len = std::strlen(kOpenClSource);
  cl_program program = api.clCreateProgramWithSource(context, 1, &source, &source_len, &err);
  Check(err, "clCreateProgramWithSource(shared down q6)");
  const auto build_begin = std::chrono::steady_clock::now();
  err = api.clBuildProgram(program, 1, &selected.device, "", nullptr, nullptr);
  const auto build_end = std::chrono::steady_clock::now();
  *program_build_ms += std::chrono::duration<double, std::milli>(build_end - build_begin).count();
  *build_log += BuildLog(api, program, selected.device);
  Check(err, "clBuildProgram(shared down q6)");
  cl_kernel kernel = api.clCreateKernel(program, "q6k_selected_down_matvec_row", &err);
  Check(err, "clCreateKernel(q6k_selected_down_matvec_row)");
  cl_mem raw_buffer = nullptr;
  cl_mem q8_qs_buffer = nullptr;
  cl_mem q8_d_buffer = nullptr;
  cl_mem output_buffer = nullptr;
  try {
    raw_buffer = api.clCreateBuffer(context, kClMemReadOnly, raw.size(), nullptr, &err);
    Check(err, "clCreateBuffer(shared down q6 raw)");
    q8_qs_buffer = api.clCreateBuffer(context, kClMemReadOnly,
                                      q8.qs.size() * sizeof(std::int8_t), nullptr, &err);
    Check(err, "clCreateBuffer(shared down q8 qs)");
    q8_d_buffer = api.clCreateBuffer(context, kClMemReadOnly,
                                     q8.d.size() * sizeof(float), nullptr, &err);
    Check(err, "clCreateBuffer(shared down q8 d)");
    output_buffer = api.clCreateBuffer(context, kClMemWriteOnly,
                                       output.size() * sizeof(float), nullptr, &err);
    Check(err, "clCreateBuffer(shared down output)");
    Check(api.clEnqueueWriteBuffer(queue, raw_buffer, kClTrue, 0,
                                   raw.size(), raw.data(), 0, nullptr, nullptr),
          "clEnqueueWriteBuffer(shared down q6 raw)");
    Check(api.clEnqueueWriteBuffer(queue, q8_qs_buffer, kClTrue, 0,
                                   q8.qs.size() * sizeof(std::int8_t),
                                   q8.qs.data(), 0, nullptr, nullptr),
          "clEnqueueWriteBuffer(shared down q8 qs)");
    Check(api.clEnqueueWriteBuffer(queue, q8_d_buffer, kClTrue, 0,
                                   q8.d.size() * sizeof(float), q8.d.data(),
                                   0, nullptr, nullptr),
          "clEnqueueWriteBuffer(shared down q8 d)");
    const cl_uint rows_per_expert_arg = static_cast<cl_uint>(rows);
    const cl_uint blocks_per_row_arg = static_cast<cl_uint>(blocks_per_row);
    Check(api.clSetKernelArg(kernel, 0, sizeof(raw_buffer), &raw_buffer), "clSetKernelArg(shared down q6 0)");
    Check(api.clSetKernelArg(kernel, 1, sizeof(q8_qs_buffer), &q8_qs_buffer), "clSetKernelArg(shared down q6 1)");
    Check(api.clSetKernelArg(kernel, 2, sizeof(q8_d_buffer), &q8_d_buffer), "clSetKernelArg(shared down q6 2)");
    Check(api.clSetKernelArg(kernel, 3, sizeof(rows_per_expert_arg), &rows_per_expert_arg), "clSetKernelArg(shared down q6 3)");
    Check(api.clSetKernelArg(kernel, 4, sizeof(blocks_per_row_arg), &blocks_per_row_arg), "clSetKernelArg(shared down q6 4)");
    Check(api.clSetKernelArg(kernel, 5, sizeof(output_buffer), &output_buffer), "clSetKernelArg(shared down q6 5)");
    const std::size_t global = output.size();
    std::vector<double> times;
    times.reserve(static_cast<std::size_t>(repeat));
    for (int i = 0; i < repeat; ++i) {
      cl_event event = nullptr;
      Check(api.clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global, nullptr, 0, nullptr, &event),
            "clEnqueueNDRangeKernel(shared down q6)");
      Check(api.clFinish(queue), "clFinish(shared down q6)");
      times.push_back(EventUs(api, event));
      api.clReleaseEvent(event);
    }
    Check(api.clEnqueueReadBuffer(queue, output_buffer, kClTrue, 0,
                                  output.size() * sizeof(float), output.data(),
                                  0, nullptr, nullptr),
          "clEnqueueReadBuffer(shared down q6 output)");
    timing->down_min_us = Min(times);
    timing->down_mean_us = Mean(times);
    timing->down_global_work_items = output.size();
    timing->down_kernel_launches = 1;
  } catch (...) {
    ReleaseMem(api, &output_buffer);
    ReleaseMem(api, &q8_d_buffer);
    ReleaseMem(api, &q8_qs_buffer);
    ReleaseMem(api, &raw_buffer);
    api.clReleaseKernel(kernel);
    api.clReleaseProgram(program);
    api.clReleaseCommandQueue(queue);
    api.clReleaseContext(context);
    throw;
  }
  ReleaseMem(api, &output_buffer);
  ReleaseMem(api, &q8_d_buffer);
  ReleaseMem(api, &q8_qs_buffer);
  ReleaseMem(api, &raw_buffer);
  api.clReleaseKernel(kernel);
  api.clReleaseProgram(program);
  api.clReleaseCommandQueue(queue);
  api.clReleaseContext(context);
  return output;
}

SharedFfnRun RunGpuSharedFfnShell(
    const std::string& model_path,
    const iq36::GgufTensorInfo& gate_tensor,
    const iq36::GgufTensorInfo& up_tensor,
    const iq36::GgufTensorInfo& down_tensor,
    const std::vector<float>& attn_post_norm,
    const std::string& device_substring,
    int repeat) {
  Require(attn_post_norm.size() == kHiddenSize, "shared input size mismatch");
  Require(gate_tensor.type == 12, "shared gate tensor must be Q4_K");
  Require(up_tensor.type == 12, "shared up tensor must be Q4_K");
  Require(down_tensor.type == 12 || down_tensor.type == 14,
          "shared down tensor must be Q4_K or Q6_K");
  Require(gate_tensor.dims == std::vector<std::uint64_t>{kHiddenSize, kIntermediateSize},
          "shared gate tensor shape mismatch");
  Require(up_tensor.dims == std::vector<std::uint64_t>{kHiddenSize, kIntermediateSize},
          "shared up tensor shape mismatch");
  Require(down_tensor.dims == std::vector<std::uint64_t>{kIntermediateSize, kHiddenSize},
          "shared down tensor shape mismatch");
  std::ifstream model(model_path, std::ios::binary);
  Require(static_cast<bool>(model), "failed to open model");
  const auto gate_raw = ReadTensorBytes(model, gate_tensor);
  const auto up_raw = ReadTensorBytes(model, up_tensor);
  const auto down_raw = ReadTensorBytes(model, down_tensor);
  const std::uint64_t gate_up_blocks_per_row = kHiddenSize / 256;
  const std::uint64_t down_blocks_per_row = kIntermediateSize / 256;
  const std::uint64_t gate_row_nbytes =
      iq36::ggml_tensor_nbytes(gate_tensor.type, std::vector<std::uint64_t>{kHiddenSize});
  const std::uint64_t up_row_nbytes =
      iq36::ggml_tensor_nbytes(up_tensor.type, std::vector<std::uint64_t>{kHiddenSize});
  const std::uint64_t down_row_nbytes =
      iq36::ggml_tensor_nbytes(down_tensor.type, std::vector<std::uint64_t>{kIntermediateSize});
  Require(gate_row_nbytes == gate_up_blocks_per_row * kQ4KBlockBytes,
          "shared gate Q4 row byte mismatch");
  Require(up_row_nbytes == gate_up_blocks_per_row * kQ4KBlockBytes,
          "shared up Q4 row byte mismatch");
  SharedFfnRun run;
  const auto q8_input = iq36::QuantizeQ8KInputPlanes(attn_post_norm);
  iq36::GpuQ4X8MatvecRunner runner(device_substring, kOpenClSource);
  run.platform_name = runner.platform_name();
  run.device_name = runner.device_name();
  run.build_log = runner.build_log();
  run.program_build_ms = runner.program_build_ms();
  const auto gate_packed = iq36::PackQ4Kx8(gate_raw, kIntermediateSize,
                                           gate_up_blocks_per_row);
  const auto up_packed = iq36::PackQ4Kx8(up_raw, kIntermediateSize,
                                         gate_up_blocks_per_row);
  const auto gate_run = runner.Run(gate_packed, q8_input.qs, q8_input.bsums,
                                   q8_input.d, kIntermediateSize,
                                   gate_up_blocks_per_row, repeat,
                                   iq36::GpuQ4X8KernelVariant::kRowlaneParallel);
  const auto up_run = runner.Run(up_packed, q8_input.qs, q8_input.bsums,
                                 q8_input.d, kIntermediateSize,
                                 gate_up_blocks_per_row, repeat,
                                 iq36::GpuQ4X8KernelVariant::kRowlaneParallel);
  run.gate = gate_run.output;
  run.up = up_run.output;
  run.gate_up.reserve(kSharedGateUpValueCount);
  run.gate_up.insert(run.gate_up.end(), run.gate.begin(), run.gate.end());
  run.gate_up.insert(run.gate_up.end(), run.up.begin(), run.up.end());
  run.timing.gate_min_us = gate_run.timing.min_us;
  run.timing.gate_mean_us = gate_run.timing.mean_us;
  run.timing.up_min_us = up_run.timing.min_us;
  run.timing.up_mean_us = up_run.timing.mean_us;
  run.timing.gate_global_work_items = gate_run.timing.global_work_items;
  run.timing.up_global_work_items = up_run.timing.global_work_items;
  const auto swiglu = RunGpuSharedSwiGlu(run.gate_up, device_substring, repeat);
  run.swiglu = swiglu.output;
  run.build_log += swiglu.build_log;
  run.program_build_ms += swiglu.program_build_ms;
  run.timing.swiglu_min_us = swiglu.min_us;
  run.timing.swiglu_mean_us = swiglu.mean_us;
  run.timing.swiglu_global_work_items = swiglu.global_work_items;

  const auto bridge_begin = std::chrono::steady_clock::now();
  const auto q8_down = iq36::QuantizeQ8KInputPlanes(run.swiglu);
  const auto bridge_end = std::chrono::steady_clock::now();
  run.timing.host_q8_bridge_us =
      std::chrono::duration<double, std::micro>(bridge_end - bridge_begin).count();
  run.timing.host_q8_bridge_count = 1;
  if (down_tensor.type == 12) {
    Require(down_row_nbytes == down_blocks_per_row * kQ4KBlockBytes,
            "shared down Q4 row byte mismatch");
    const auto down_packed =
        iq36::PackQ4Kx8(down_raw, kHiddenSize, down_blocks_per_row);
    const auto down_run = runner.Run(down_packed, q8_down.qs, q8_down.bsums,
                                     q8_down.d, kHiddenSize,
                                     down_blocks_per_row, repeat,
                                     iq36::GpuQ4X8KernelVariant::kRowlaneParallel);
    run.down = down_run.output;
    run.timing.down_min_us = down_run.timing.min_us;
    run.timing.down_mean_us = down_run.timing.mean_us;
    run.timing.down_global_work_items = down_run.timing.global_work_items;
    run.timing.down_kernel_launches = 1;
  } else {
    Require(down_row_nbytes == down_blocks_per_row * kQ6KBlockBytes,
            "shared down Q6 row byte mismatch");
    run.down = RunGpuSharedDownQ6(down_raw, q8_down, kHiddenSize,
                                  down_blocks_per_row, device_substring,
                                  repeat, &run.timing, &run.platform_name,
                                  &run.device_name, &run.build_log,
                                  &run.program_build_ms);
  }
  run.timing.shared_ffn_kernel_sum_min_us =
      run.timing.gate_min_us + run.timing.up_min_us +
      run.timing.swiglu_min_us + run.timing.down_min_us;
  run.timing.shared_ffn_kernel_sum_mean_us =
      run.timing.gate_mean_us + run.timing.up_mean_us +
      run.timing.swiglu_mean_us + run.timing.down_mean_us;
  return run;
}
'''


SHARED_MAIN_CPP = r'''
int main(int argc, char** argv) {
  try {
    const Args args = ParseArgs(argc, argv);
    const auto index = iq36::parse_gguf_model_index(args.model_path);
    const auto load_map = iq36::validate_qwen36_load_map(index);
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
    Require(selected_gate_up_tensor != nullptr, "selected gate-up tensor missing");
    Require(selected_down_tensor != nullptr, "selected down tensor missing");
    Require(shared_gate_tensor != nullptr, "shared gate tensor missing");
    Require(shared_up_tensor != nullptr, "shared up tensor missing");
    Require(shared_down_tensor != nullptr, "shared down tensor missing");
    Require(shared_input_gate_tensor != nullptr, "shared input gate tensor missing");
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

    const auto attn_residual =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "attn_residual.bin"));
    const auto attn_post_norm =
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
        attn_residual.size() == kHiddenSize &&
        attn_post_norm.size() == kHiddenSize &&
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
    const auto shared_input_gate_weights =
        ReadF32TensorPayload(model, *shared_input_gate_tensor,
                             static_cast<std::size_t>(kHiddenSize));

    const auto native_selected_gate_up =
        iq36::matvec_expert_tensor(args.model_path, index,
                                   selected_gate_up_tensor_name,
                                   attn_post_norm, expert_ids);
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
        iq36::matvec_tensor(args.model_path, index, shared_gate_tensor_name, attn_post_norm);
    const auto native_shared_up =
        iq36::matvec_tensor(args.model_path, index, shared_up_tensor_name, attn_post_norm);
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
                            shared_input_gate_tensor_name, attn_post_norm);
    Require(native_shared_input_gate.size() == 1, "native shared input gate size mismatch");
    const std::vector<float> native_shared_sigmoid{
        iq36::sigmoid_scalar(native_shared_input_gate[0])};
    const auto native_shared_gated =
        iq36::multiply_by_scalar(native_shared_down, native_shared_sigmoid[0]);
    const auto native_ffn_out =
        iq36::add_vectors(native_moe_out, native_shared_gated);
    const auto native_layer_output =
        iq36::add_vectors(attn_residual, native_ffn_out);

    const auto selected_gpu = RunGpuSelectedFfnShell(
        args.model_path, *selected_gate_up_tensor, *selected_down_tensor,
        attn_post_norm, expert_ids, args.device_substring, args.repeat);
    const auto shared_gpu = RunGpuSharedFfnShell(
        args.model_path, *shared_gate_tensor, *shared_up_tensor,
        *shared_down_tensor, attn_post_norm, args.device_substring, args.repeat);
    const auto gpu = RunGpuShell(shared_input_gate_weights, attn_post_norm,
                                 selected_gpu.down, weights_norm,
                                 shared_gpu.down, attn_residual,
                                 args.device_substring, args.repeat);

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
        selected_gpu.device_name.find(args.device_substring) != std::string::npos &&
        shared_gpu.device_name.find(args.device_substring) != std::string::npos &&
        gpu.device_name.find(args.device_substring) != std::string::npos;
    const bool required_checks_passed =
        load_map.ready &&
        selected_tensor_shape_ok &&
        shared_tensor_shape_ok &&
        payload_counts_ok &&
        arc_selected &&
        selected_comparisons_passed &&
        shared_comparisons_passed &&
        tail_comparisons_passed &&
        selected_timing_positive &&
        shared_timing_positive &&
        tail_timing_positive;

    const double ffn_kernel_sum_min =
        selected_gpu.timing.selected_ffn_kernel_sum_min_us +
        shared_gpu.timing.shared_ffn_kernel_sum_min_us +
        gpu.timing.shell_sum_min_us;
    const double ffn_kernel_sum_mean =
        selected_gpu.timing.selected_ffn_kernel_sum_mean_us +
        shared_gpu.timing.shared_ffn_kernel_sum_mean_us +
        gpu.timing.shell_sum_mean_us;
    std::cout << std::setprecision(10);
    std::cout << "{";
    std::cout << "\"schema_version\":\"intel-qwen36-gpu-resident-shared-ffn-shell-probe-v0\",";
    std::cout << "\"model_path\":\"" << JsonEscape(args.model_path) << "\",";
    std::cout << "\"layer\":" << args.layer << ",";
    std::cout << "\"source_token_position\":" << kSourceTokenPosition << ",";
    std::cout << "\"resident_api\":\"shared_expert_ffn_load_once_run_many\",";
    std::cout << "\"resident_load_count\":1,";
    std::cout << "\"resident_shell_invocations\":" << args.repeat << ",";
    std::cout << "\"selected_down_host_q8_bridge\":true,";
    std::cout << "\"shared_down_host_q8_bridge\":true,";
    std::cout << "\"selected_down_q4_expert_launches\":"
              << selected_gpu.timing.down_kernel_launches << ",";
    std::cout << "\"shared_down_kernel_launches\":"
              << shared_gpu.timing.down_kernel_launches << ",";
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
    std::cout << "\"platform_name\":\"" << JsonEscape(selected_gpu.platform_name) << "\",";
    std::cout << "\"device_name\":\"" << JsonEscape(selected_gpu.device_name) << "\",";
    std::cout << "\"shared_device_name\":\"" << JsonEscape(shared_gpu.device_name) << "\",";
    std::cout << "\"tail_device_name\":\"" << JsonEscape(gpu.device_name) << "\",";
    std::cout << "\"program_build_ms\":"
              << (selected_gpu.program_build_ms + shared_gpu.program_build_ms +
                  gpu.program_build_ms) << ",";
    std::cout << "\"build_log\":\""
              << JsonEscape(selected_gpu.build_log + shared_gpu.build_log + gpu.build_log) << "\",";
    std::cout << "\"repeat\":" << args.repeat << ",";
    std::cout << "\"timings\":{";
    std::cout << "\"selected_ffn_kernel_sum_min_us\":"
              << selected_gpu.timing.selected_ffn_kernel_sum_min_us << ",";
    std::cout << "\"selected_ffn_kernel_sum_mean_us\":"
              << selected_gpu.timing.selected_ffn_kernel_sum_mean_us << ",";
    std::cout << "\"shared_gate_min_us\":" << shared_gpu.timing.gate_min_us << ",";
    std::cout << "\"shared_gate_mean_us\":" << shared_gpu.timing.gate_mean_us << ",";
    std::cout << "\"shared_up_min_us\":" << shared_gpu.timing.up_min_us << ",";
    std::cout << "\"shared_up_mean_us\":" << shared_gpu.timing.up_mean_us << ",";
    std::cout << "\"shared_swiglu_min_us\":" << shared_gpu.timing.swiglu_min_us << ",";
    std::cout << "\"shared_swiglu_mean_us\":" << shared_gpu.timing.swiglu_mean_us << ",";
    std::cout << "\"shared_down_min_us\":" << shared_gpu.timing.down_min_us << ",";
    std::cout << "\"shared_down_mean_us\":" << shared_gpu.timing.down_mean_us << ",";
    std::cout << "\"shared_down_host_q8_bridge_us\":"
              << shared_gpu.timing.host_q8_bridge_us << ",";
    std::cout << "\"shared_ffn_kernel_sum_min_us\":"
              << shared_gpu.timing.shared_ffn_kernel_sum_min_us << ",";
    std::cout << "\"shared_ffn_kernel_sum_mean_us\":"
              << shared_gpu.timing.shared_ffn_kernel_sum_mean_us << ",";
    std::cout << "\"moe_weighted_aggregate_min_us\":" << gpu.timing.weighted_min_us << ",";
    std::cout << "\"shared_input_gate_matvec_min_us\":"
              << gpu.timing.shared_gate_matvec_min_us << ",";
    std::cout << "\"shared_gate_apply_min_us\":"
              << gpu.timing.shared_gate_apply_min_us << ",";
    std::cout << "\"ffn_output_add_min_us\":"
              << gpu.timing.ffn_output_add_min_us << ",";
    std::cout << "\"post_ffn_residual_add_min_us\":"
              << gpu.timing.residual_add_min_us << ",";
    std::cout << "\"resident_full_ffn_to_layer_kernel_sum_min_us\":"
              << ffn_kernel_sum_min << ",";
    std::cout << "\"resident_full_ffn_to_layer_kernel_sum_mean_us\":"
              << ffn_kernel_sum_mean;
    std::cout << "},";
    std::cout << "\"comparisons\":{";
    std::cout << "\"selected_gate_up\":";
    WriteCompareGroup(selected_gate_up_cpu_vs_oracle, selected_gate_up_gpu_vs_cpu, selected_gate_up_gpu_vs_oracle);
    std::cout << ",\"selected_swiglu\":";
    WriteCompareGroup(selected_swiglu_cpu_vs_oracle, selected_swiglu_gpu_vs_cpu, selected_swiglu_gpu_vs_oracle);
    std::cout << ",\"selected_down\":";
    WriteCompareGroup(selected_down_cpu_vs_oracle, selected_down_gpu_vs_cpu, selected_down_gpu_vs_oracle);
    std::cout << ",\"shared_gate_internal\":{\"gpu_vs_cpu\":";
    WriteCompare(shared_gate_gpu_vs_cpu);
    std::cout << "},\"shared_up_internal\":{\"gpu_vs_cpu\":";
    WriteCompare(shared_up_gpu_vs_cpu);
    std::cout << "},\"shared_swiglu_internal\":{\"gpu_vs_cpu\":";
    WriteCompare(shared_swiglu_gpu_vs_cpu);
    std::cout << "},\"shared_down\":";
    WriteCompareGroup(shared_down_cpu_vs_oracle, shared_down_gpu_vs_cpu, shared_down_gpu_vs_oracle);
    std::cout << ",\"ffn_moe_weighted\":";
    WriteCompareGroup(weighted_cpu_vs_oracle, weighted_gpu_vs_cpu, weighted_gpu_vs_oracle);
    std::cout << ",\"ffn_moe_out\":";
    WriteCompareGroup(moe_out_cpu_vs_oracle, moe_out_gpu_vs_cpu, moe_out_gpu_vs_oracle);
    std::cout << ",\"shared_gate\":";
    WriteCompareGroup(shared_input_gate_cpu_vs_oracle, shared_input_gate_gpu_vs_cpu, shared_input_gate_gpu_vs_oracle);
    std::cout << ",\"shared_gate_sigmoid\":";
    WriteCompareGroup(shared_sigmoid_cpu_vs_oracle, shared_sigmoid_gpu_vs_cpu, shared_sigmoid_gpu_vs_oracle);
    std::cout << ",\"ffn_shexp_gated\":";
    WriteCompareGroup(shared_gated_cpu_vs_oracle, shared_gated_gpu_vs_cpu, shared_gated_gpu_vs_oracle);
    std::cout << ",\"ffn_out\":";
    WriteCompareGroup(ffn_out_cpu_vs_oracle, ffn_out_gpu_vs_cpu, ffn_out_gpu_vs_oracle);
    std::cout << ",\"layer_output\":";
    WriteCompareGroup(layer_cpu_vs_oracle, layer_gpu_vs_cpu, layer_gpu_vs_oracle);
    std::cout << "},";
    std::cout << "\"checks\":{";
    std::cout << "\"load_map_ready\":" << (load_map.ready ? "true" : "false") << ",";
    std::cout << "\"selected_tensor_shape_ok\":" << (selected_tensor_shape_ok ? "true" : "false") << ",";
    std::cout << "\"shared_tensor_shape_ok\":" << (shared_tensor_shape_ok ? "true" : "false") << ",";
    std::cout << "\"payload_counts_ok\":" << (payload_counts_ok ? "true" : "false") << ",";
    std::cout << "\"arc_device_selected\":" << (arc_selected ? "true" : "false") << ",";
    std::cout << "\"selected_expert_ffn_matches_oracle\":"
              << (selected_comparisons_passed ? "true" : "false") << ",";
    std::cout << "\"shared_expert_ffn_matches_oracle\":"
              << (shared_comparisons_passed ? "true" : "false") << ",";
    std::cout << "\"tail_matches_oracle\":"
              << (tail_comparisons_passed ? "true" : "false") << ",";
    std::cout << "\"selected_down_host_q8_bridge\":true,";
    std::cout << "\"shared_down_host_q8_bridge\":true,";
    std::cout << "\"resident_load_once\":true,";
    std::cout << "\"resident_shell_invocations_positive\":"
              << (args.repeat > 0 ? "true" : "false") << ",";
    std::cout << "\"gpu_event_timing_positive\":"
              << ((selected_timing_positive && shared_timing_positive &&
                   tail_timing_positive) ? "true" : "false") << ",";
    std::cout << "\"speedup_claims_forbidden\":true";
    std::cout << "},";
    std::cout << "\"required_checks_passed\":"
              << (required_checks_passed ? "true" : "false") << ",";
    std::cout << "\"speedup_claims_allowed\":false";
    std::cout << "}\n";
    return required_checks_passed ? 0 : 1;
  } catch (const std::exception& ex) {
    std::cerr << "error: " << ex.what() << "\n";
    return 2;
  }
}
'''


def utc_stamp() -> str:
  return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def iso_now() -> str:
  return dt.datetime.now(dt.timezone.utc).isoformat()


def shell_join(argv: list[str]) -> str:
  return " ".join(shlex.quote(item) for item in argv)


def shared_ffn_probe_cpp(opencl_source: str) -> str:
  cpp = SELECTED.selected_ffn_probe_cpp(opencl_source)
  helper_marker = "\nvoid WriteCompare(const iq36::VectorCompareStats& stats) {\n"
  cpp = cpp.replace(helper_marker, "\n" + SHARED_HELPERS_CPP + helper_marker, 1)
  main_index = cpp.index("\nint main(")
  return cpp[:main_index] + "\n" + SHARED_MAIN_CPP


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--target", "--host", dest="host", metavar="TARGET", default=DEFAULT_HOST)
  parser.add_argument("--model", default=DEFAULT_MODEL)
  parser.add_argument("--env-script", default=DEFAULT_ENV_SCRIPT)
  parser.add_argument("--staging-root", "--remote-root", dest="remote_root", metavar="PATH", default=DEFAULT_REMOTE_ROOT)
  parser.add_argument("--oracle-bundle", type=Path, default=DEFAULT_ORACLE_BUNDLE)
  parser.add_argument("--layer", type=int, default=5)
  parser.add_argument("--resident-invocations", type=int, default=5)
  parser.add_argument("--device-substring", default="B390")
  parser.add_argument("--out-dir", type=Path)
  parser.add_argument("--timeout-s", type=int, default=300)
  return parser.parse_args()


def parse_probe_stdout(stdout: str) -> dict[str, Any] | None:
  for line in reversed(stdout.splitlines()):
    line = line.strip()
    if line.startswith("{") and line.endswith("}"):
      return json.loads(line)
  return None


def nested_bool(obj: dict[str, Any] | None, *keys: str) -> bool:
  current: Any = obj
  for key in keys:
    if not isinstance(current, dict):
      return False
    current = current.get(key)
  return bool(current)


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
      and probe.get("resident_api") == "shared_expert_ffn_load_once_run_many"
      and probe.get("resident_load_count") == 1
      and probe.get("resident_shell_invocations") == expected_invocations
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
      "# GPU Resident Shared-Expert FFN Shell Probe",
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
      f"- selected down host Q8 bridge: `{str(probe.get('selected_down_host_q8_bridge')).lower()}`",
      f"- shared down host Q8 bridge: `{str(probe.get('shared_down_host_q8_bridge')).lower()}`",
      "",
      "| output | comparison | max abs | RMSE |",
      "|---|---|---:|---:|",
  ]
  for name in (
      "selected_down",
      "shared_down",
      "ffn_moe_out",
      "ffn_shexp_gated",
      "ffn_out",
      "layer_output",
  ):
    group = comparisons.get(name, {}) if isinstance(comparisons, dict) else {}
    if not isinstance(group, dict):
      continue
    lane = group.get("gpu_vs_oracle", {}) if isinstance(group.get("gpu_vs_oracle"), dict) else {}
    lines.append(f"| {name} | gpu_vs_oracle | {lane.get('max_abs_diff')} | {lane.get('rmse')} |")
  lines += [
      "",
      "| kernel group | min us |",
      "|---|---:|",
      f"| selected_ffn_sum | {timings.get('selected_ffn_kernel_sum_min_us')} |",
      f"| shared_gate | {timings.get('shared_gate_min_us')} |",
      f"| shared_up | {timings.get('shared_up_min_us')} |",
      f"| shared_swiglu | {timings.get('shared_swiglu_min_us')} |",
      f"| shared_down | {timings.get('shared_down_min_us')} |",
      f"| shared_ffn_sum | {timings.get('shared_ffn_kernel_sum_min_us')} |",
      f"| resident_full_ffn_to_layer_sum | {timings.get('resident_full_ffn_to_layer_kernel_sum_min_us')} |",
      "",
      "The target-side process computes selected and shared FFN branches from",
      "captured `attn_post_norm`, then runs aggregation, shared input gate/apply,",
      "FFN add, and residual to captured `l_out`. Q8 activation bridges remain",
      "explicit for selected and shared down paths. This is captured single-layer",
      "evidence only, not prompt/token decode throughput.",
      "",
  ]
  path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  args = parse_args()
  if args.resident_invocations <= 0:
    raise SystemExit("--resident-invocations must be positive")
  stamp = utc_stamp()
  created_at = iso_now()
  out_dir = args.out_dir or ROOT / f"output/gpu-resident-shared-ffn-shell-probe-{stamp}"
  out_dir.mkdir(parents=True, exist_ok=False)
  raw_dir = out_dir / "raw"
  raw_dir.mkdir()

  payloads = SELECTED.resolve_payloads(args.layer)
  opencl_source = OPENCL_SOURCE.read_text(encoding="utf-8")
  opencl_source_hash = iq36_local.sha256_file(OPENCL_SOURCE)
  local_cpp = out_dir / "gpu_resident_shared_ffn_shell_probe.cpp"
  local_cpp.write_text(shared_ffn_probe_cpp(opencl_source), encoding="utf-8")

  remote_dir = f"{args.remote_root.rstrip('/')}/gpu-resident-shared-ffn-shell-probe-{stamp}"
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
    for local, remote in SELECTED.SOURCE_FILES:
      transfers.append(iq36_local.copy_to(args.host, ROOT / local, f"{remote_dir}/{remote}", args.timeout_s))
    transfers.append(
        iq36_local.copy_to(
            args.host,
            local_cpp,
            f"{remote_dir}/tests/gpu_resident_shared_ffn_shell_probe.cpp",
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

  executable = f"{remote_dir}/build/iq36-gpu-resident-shared-ffn-shell-probe"
  compile_cmd = " && ".join([
      f"source {shlex.quote(args.env_script)} >/dev/null 2>&1",
      (
          "g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic "
          f"-I {shlex.quote(remote_dir + '/include')} "
          f"{shlex.quote(remote_dir + '/src/gguf_loader.cpp')} "
          f"{shlex.quote(remote_dir + '/src/gpu_q4x8_matvec.cpp')} "
          f"{shlex.quote(remote_dir + '/tests/gpu_resident_shared_ffn_shell_probe.cpp')} "
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
      "probe": probe,
      "checks": checks,
      "required_checks_passed": required_checks_passed,
      "speedup_claims_allowed": False,
  }
  manifest = {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "created_at": created_at,
      "tool": "tools/intel-qwen36-gpu-resident-shared-ffn-shell-probe.py",
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
      "gpu_resident_shared_ffn_shell_probe",
      [
          ("required_checks_passed", required_checks_passed),
          ("resident_shell_invocations", args.resident_invocations),
          ("shared_ffn_kernel_sum_min_us", nested_number(timings, "shared_ffn_kernel_sum_min_us")),
          ("resident_full_ffn_to_layer_kernel_sum_min_us", nested_number(timings, "resident_full_ffn_to_layer_kernel_sum_min_us")),
          ("shared_down_host_q8_bridge_us", nested_number(timings, "shared_down_host_q8_bridge_us")),
          ("shared_down_gpu_vs_oracle_max_abs_diff", nested_number(comparisons, "shared_down", "gpu_vs_oracle", "max_abs_diff")),
          ("layer_output_gpu_vs_oracle_max_abs_diff", nested_number(comparisons, "layer_output", "gpu_vs_oracle", "max_abs_diff")),
      ],
  )
  write_summary(out_dir / "summary.md", payload)
  print(out_dir)
  return 0 if required_checks_passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
