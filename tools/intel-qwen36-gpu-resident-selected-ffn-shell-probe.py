#!/usr/bin/env python3
"""Run the resident GPU selected-expert FFN shell handoff probe."""

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
CAPTURED_TOOL = Path(__file__).with_name("intel-qwen36-gpu-captured-layer-shell-probe.py")
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA_VERSION = "intel-qwen36-gpu-resident-selected-ffn-shell-probe-v0"
DEFAULT_HOST = "local"
DEFAULT_MODEL = "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf"
DEFAULT_ENV_SCRIPT = "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh"
DEFAULT_REMOTE_ROOT = "/home/intel/intel-qwen36-gpu"
DEFAULT_ORACLE_BUNDLE = ROOT / "oracle/r0-oracle-bundle-20260627T060028Z"
PAYLOAD_ROOT = ROOT / "output/r0-boundary-capture-run-20260627T054024Z/remote-output/payloads"
OPENCL_SOURCE = ROOT / "engine/gpu/opencl/q4x8_matvec.cl"
SOURCE_FILES = [
    ("engine/include/intel_qwen36/gguf_loader.hpp", "include/intel_qwen36/gguf_loader.hpp"),
    ("engine/include/intel_qwen36/gpu_q4x8_matvec.hpp", "include/intel_qwen36/gpu_q4x8_matvec.hpp"),
    ("engine/src/gguf_loader.cpp", "src/gguf_loader.cpp"),
    ("engine/src/gpu_q4_cpu_order_matvec.cpp", "src/gpu_q4_cpu_order_matvec.cpp"),
    ("engine/src/gpu_q4x8_matvec.cpp", "src/gpu_q4x8_matvec.cpp"),
]
PAYLOAD_SPECS = {
    "attn_residual": ("attn_residual.bin", "attn_residual-{layer}__tok15__ord208.bin", 8192),
    "attn_post_norm": ("attn_post_norm.bin", "attn_post_norm-{layer}__tok15__ord209.bin", 8192),
    "ffn_moe_topk": ("ffn_moe_topk.bin", "ffn_moe_topk-{layer}__tok15__ord212.bin", 32),
    "ffn_moe_weights_norm": (
        "ffn_moe_weights_norm.bin",
        "ffn_moe_weights_norm-{layer}__tok15__ord214.bin",
        32,
    ),
    "ffn_moe_gate_up": ("ffn_moe_gate_up.bin", "ffn_moe_gate_up-{layer}__tok15__ord215.bin", 32768),
    "ffn_moe_swiglu": ("ffn_moe_swiglu.bin", "ffn_moe_swiglu-{layer}__tok15__ord218.bin", 16384),
    "ffn_moe_down": ("ffn_moe_down.bin", "ffn_moe_down-{layer}__tok15__ord219.bin", 65536),
    "ffn_moe_weighted": (
        "ffn_moe_weighted.bin",
        "ffn_moe_weighted-{layer}__tok15__ord220.bin",
        65536,
    ),
    "ffn_moe_out": ("ffn_moe_out.bin", "ffn_moe_out-{layer}__tok15__ord221.bin", 8192),
    "ffn_shexp": ("ffn_shexp.bin", "ffn_shexp-{layer}__tok15__ord222.bin", 8192),
    "shared_gate": (
        "shared_expert_gate.bin",
        "shared_expert_gate-{layer}__tok15__ord223.bin",
        4,
    ),
    "shared_gate_sigmoid": (
        "shared_expert_gate_sigmoid.bin",
        "shared_expert_gate_sigmoid-{layer}__tok15__ord224.bin",
        4,
    ),
    "ffn_shexp_gated": (
        "ffn_shexp_gated.bin",
        "ffn_shexp_gated-{layer}__tok15__ord225.bin",
        8192,
    ),
    "ffn_out": ("ffn_out.bin", "ffn_out-{layer}__tok15__ord226.bin", 8192),
    "layer_output": ("layer_output.bin", "l_out-{layer}__tok15__ord227.bin", 8192),
}


def load_captured_tool() -> ModuleType:
  spec = importlib.util.spec_from_file_location("iq36_captured_layer_shell_probe", CAPTURED_TOOL)
  if spec is None or spec.loader is None:
    raise SystemExit(f"failed to load captured shell tool: {CAPTURED_TOOL}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


CAPTURED = load_captured_tool()


SELECTED_CONSTANTS_CPP = r'''
constexpr int kExpertCount = 256;
constexpr int kIntermediateSize = 512;
constexpr int kGateUpRowsPerExpert = kIntermediateSize * 2;
constexpr int kGateUpValueCount = kGateUpRowsPerExpert * kExpertUsedCount;
constexpr int kSwiGluValueCount = kIntermediateSize * kExpertUsedCount;
constexpr int kQ4KBlockBytes = 144;
'''


SELECTED_HELPERS_CPP = r'''

struct SelectedFfnTiming {
  double gate_up_min_us = 0.0;
  double gate_up_mean_us = 0.0;
  double swiglu_min_us = 0.0;
  double swiglu_mean_us = 0.0;
  double down_min_us = 0.0;
  double down_mean_us = 0.0;
  double host_q8_bridge_us = 0.0;
  double selected_ffn_kernel_sum_min_us = 0.0;
  double selected_ffn_kernel_sum_mean_us = 0.0;
  std::uint64_t gate_up_global_work_items = 0;
  std::uint64_t swiglu_global_work_items = 0;
  std::uint64_t down_global_work_items = 0;
  std::uint64_t down_kernel_launches = 0;
  std::uint64_t host_q8_bridge_count = 0;
};

struct SelectedFfnRun {
  std::vector<float> gate_up;
  std::vector<float> swiglu;
  std::vector<float> down;
  SelectedFfnTiming timing;
  std::string platform_name;
  std::string device_name;
  std::string build_log;
  double program_build_ms = 0.0;
};

std::vector<std::int32_t> ReadI32VectorFile(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  Require(static_cast<bool>(input), "i32 vector file could not be opened");
  input.seekg(0, std::ios::end);
  const auto size = input.tellg();
  Require(size >= 0, "i32 vector file size failed");
  Require(static_cast<std::uint64_t>(size) % sizeof(std::int32_t) == 0,
          "i32 vector file size mismatch");
  input.seekg(0, std::ios::beg);
  std::vector<std::int32_t> values(
      static_cast<std::size_t>(size) / sizeof(std::int32_t), 0);
  input.read(reinterpret_cast<char*>(values.data()),
             static_cast<std::streamsize>(values.size() * sizeof(std::int32_t)));
  Require(static_cast<bool>(input), "i32 vector file read failed");
  return values;
}

std::vector<std::uint8_t> ReadSelectedExpertRaw(std::ifstream& model,
                                                const iq36::GgufTensorInfo& tensor,
                                                const std::vector<std::int32_t>& expert_ids,
                                                std::uint64_t rows_per_expert,
                                                std::uint64_t row_nbytes,
                                                const char* label) {
  std::vector<std::uint8_t> raw;
  raw.resize(static_cast<std::size_t>(expert_ids.size() * rows_per_expert * row_nbytes));
  for (std::size_t selected = 0; selected < expert_ids.size(); ++selected) {
    const auto expert_id = expert_ids[selected];
    Require(expert_id >= 0 && expert_id < kExpertCount, "selected expert id out of range");
    const std::uint64_t expert_row_base =
        static_cast<std::uint64_t>(expert_id) * rows_per_expert;
    const std::uint64_t source_offset = tensor.absolute_offset + expert_row_base * row_nbytes;
    const std::size_t target_offset =
        selected * static_cast<std::size_t>(rows_per_expert * row_nbytes);
    const std::size_t byte_count = static_cast<std::size_t>(rows_per_expert * row_nbytes);
    model.clear();
    model.seekg(static_cast<std::streamoff>(source_offset), std::ios::beg);
    Require(static_cast<bool>(model), std::string(label) + " selected expert slice seek failed");
    model.read(reinterpret_cast<char*>(raw.data() + target_offset),
               static_cast<std::streamsize>(byte_count));
    Require(model.gcount() == static_cast<std::streamsize>(byte_count),
            std::string(label) + " selected expert slice read failed");
  }
  return raw;
}

std::vector<std::uint8_t> SliceBytes(const std::vector<std::uint8_t>& bytes,
                                    std::size_t offset,
                                    std::size_t count) {
  Require(offset + count <= bytes.size(), "byte slice out of range");
  return std::vector<std::uint8_t>(bytes.begin() + static_cast<std::ptrdiff_t>(offset),
                                  bytes.begin() + static_cast<std::ptrdiff_t>(offset + count));
}

std::vector<float> SliceFloats(const std::vector<float>& values,
                               std::size_t offset,
                               std::size_t count) {
  Require(offset + count <= values.size(), "float slice out of range");
  return std::vector<float>(values.begin() + static_cast<std::ptrdiff_t>(offset),
                            values.begin() + static_cast<std::ptrdiff_t>(offset + count));
}

struct SwiGluRun {
  std::vector<float> output;
  double min_us = 0.0;
  double mean_us = 0.0;
  std::uint64_t global_work_items = 0;
  std::string build_log;
  double program_build_ms = 0.0;
};

SwiGluRun RunGpuSwiGluResident(const std::vector<float>& gate_up,
                               const std::string& device_substring,
                               int repeat) {
  Require(gate_up.size() == kGateUpValueCount, "selected gate-up size mismatch");
  OpenClApi api;
  SwiGluRun run;
  run.output.assign(kSwiGluValueCount, 0.0f);
  const auto selected = SelectDevice(api, device_substring);
  cl_int err = kClSuccess;
  cl_context context = api.clCreateContext(nullptr, 1, &selected.device, nullptr, nullptr, &err);
  Check(err, "clCreateContext(swiglu)");
  cl_command_queue queue = api.clCreateCommandQueue(context, selected.device, kClQueueProfilingEnable, &err);
  Check(err, "clCreateCommandQueue(swiglu)");
  const char* source = kOpenClSource;
  const std::size_t source_len = std::strlen(kOpenClSource);
  cl_program program = api.clCreateProgramWithSource(context, 1, &source, &source_len, &err);
  Check(err, "clCreateProgramWithSource(swiglu)");
  const auto build_begin = std::chrono::steady_clock::now();
  err = api.clBuildProgram(program, 1, &selected.device, "", nullptr, nullptr);
  const auto build_end = std::chrono::steady_clock::now();
  run.program_build_ms = std::chrono::duration<double, std::milli>(build_end - build_begin).count();
  run.build_log = BuildLog(api, program, selected.device);
  Check(err, "clBuildProgram(swiglu)");
  cl_kernel kernel = api.clCreateKernel(program, "ffn_moe_swiglu_f32", &err);
  Check(err, "clCreateKernel(ffn_moe_swiglu_f32)");
  cl_mem gate_up_buffer = nullptr;
  cl_mem output_buffer = nullptr;
  try {
    gate_up_buffer = api.clCreateBuffer(context, kClMemReadOnly,
                                        gate_up.size() * sizeof(float), nullptr, &err);
    Check(err, "clCreateBuffer(swiglu gate_up)");
    output_buffer = api.clCreateBuffer(context, kClMemWriteOnly,
                                       run.output.size() * sizeof(float), nullptr, &err);
    Check(err, "clCreateBuffer(swiglu output)");
    Check(api.clEnqueueWriteBuffer(queue, gate_up_buffer, kClTrue, 0,
                                   gate_up.size() * sizeof(float), gate_up.data(),
                                   0, nullptr, nullptr),
          "clEnqueueWriteBuffer(swiglu gate_up)");
    const cl_uint intermediate_arg = static_cast<cl_uint>(kIntermediateSize);
    const cl_uint expert_arg = static_cast<cl_uint>(kExpertUsedCount);
    Check(api.clSetKernelArg(kernel, 0, sizeof(gate_up_buffer), &gate_up_buffer), "clSetKernelArg(swiglu 0)");
    Check(api.clSetKernelArg(kernel, 1, sizeof(intermediate_arg), &intermediate_arg), "clSetKernelArg(swiglu 1)");
    Check(api.clSetKernelArg(kernel, 2, sizeof(expert_arg), &expert_arg), "clSetKernelArg(swiglu 2)");
    Check(api.clSetKernelArg(kernel, 3, sizeof(output_buffer), &output_buffer), "clSetKernelArg(swiglu 3)");
    const std::size_t global = static_cast<std::size_t>(kSwiGluValueCount);
    std::vector<double> times;
    times.reserve(static_cast<std::size_t>(repeat));
    for (int i = 0; i < repeat; ++i) {
      cl_event event = nullptr;
      Check(api.clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global, nullptr, 0, nullptr, &event),
            "clEnqueueNDRangeKernel(ffn_moe_swiglu_f32)");
      Check(api.clFinish(queue), "clFinish(swiglu)");
      times.push_back(EventUs(api, event));
      api.clReleaseEvent(event);
    }
    Check(api.clEnqueueReadBuffer(queue, output_buffer, kClTrue, 0,
                                  run.output.size() * sizeof(float),
                                  run.output.data(), 0, nullptr, nullptr),
          "clEnqueueReadBuffer(swiglu output)");
    run.min_us = Min(times);
    run.mean_us = Mean(times);
    run.global_work_items = kSwiGluValueCount;
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

SelectedFfnRun RunGpuSelectedFfnShell(
    const std::string& model_path,
    const iq36::GgufTensorInfo& gate_up_tensor,
    const iq36::GgufTensorInfo& down_tensor,
    const std::vector<float>& attn_post_norm,
    const std::vector<std::int32_t>& expert_ids,
    const std::string& device_substring,
    int repeat) {
  Require(attn_post_norm.size() == kHiddenSize, "attn_post_norm size mismatch");
  Require(expert_ids.size() == kExpertUsedCount, "selected expert count mismatch");
  Require(gate_up_tensor.type == 12, "selected gate-up tensor must be Q4_K");
  Require(down_tensor.type == 12, "selected down tensor must be Q4_K for this gate");
  Require(gate_up_tensor.dims == std::vector<std::uint64_t>{kHiddenSize, kGateUpRowsPerExpert, kExpertCount},
          "selected gate-up tensor shape mismatch");
  Require(down_tensor.dims == std::vector<std::uint64_t>{kIntermediateSize, kHiddenSize, kExpertCount},
          "selected down tensor shape mismatch");

  std::ifstream model(model_path, std::ios::binary);
  Require(static_cast<bool>(model), "failed to open model");
  const std::uint64_t gate_up_blocks_per_row = kHiddenSize / 256;
  const std::uint64_t down_blocks_per_row = kIntermediateSize / 256;
  const std::uint64_t gate_up_row_nbytes =
      iq36::ggml_tensor_nbytes(gate_up_tensor.type, std::vector<std::uint64_t>{kHiddenSize});
  const std::uint64_t down_row_nbytes =
      iq36::ggml_tensor_nbytes(down_tensor.type, std::vector<std::uint64_t>{kIntermediateSize});
  Require(gate_up_row_nbytes == gate_up_blocks_per_row * kQ4KBlockBytes,
          "selected gate-up Q4 row byte mismatch");
  Require(down_row_nbytes == down_blocks_per_row * kQ4KBlockBytes,
          "selected down Q4 row byte mismatch");

  const auto gate_up_raw =
      ReadSelectedExpertRaw(model, gate_up_tensor, expert_ids, kGateUpRowsPerExpert,
                            gate_up_row_nbytes, "gate-up");
  const auto down_raw =
      ReadSelectedExpertRaw(model, down_tensor, expert_ids, kHiddenSize,
                            down_row_nbytes, "down");
  const auto gate_up_packed =
      iq36::PackQ4Kx8(gate_up_raw, kGateUpRowsPerExpert * kExpertUsedCount,
                      gate_up_blocks_per_row);
  const auto q8_attn = iq36::QuantizeQ8KInputPlanes(attn_post_norm);
  iq36::GpuQ4X8MatvecRunner runner(device_substring, kOpenClSource);
  SelectedFfnRun run;
  run.platform_name = runner.platform_name();
  run.device_name = runner.device_name();
  run.build_log = runner.build_log();
  run.program_build_ms = runner.program_build_ms();
  const auto gate_up = runner.Run(gate_up_packed, q8_attn.qs, q8_attn.bsums,
                                  q8_attn.d,
                                  kGateUpRowsPerExpert * kExpertUsedCount,
                                  gate_up_blocks_per_row, repeat,
                                  iq36::GpuQ4X8KernelVariant::kRowlaneParallel);
  run.gate_up = gate_up.output;
  run.timing.gate_up_min_us = gate_up.timing.min_us;
  run.timing.gate_up_mean_us = gate_up.timing.mean_us;
  run.timing.gate_up_global_work_items = gate_up.timing.global_work_items;

  const auto swiglu = RunGpuSwiGluResident(run.gate_up, device_substring, repeat);
  run.swiglu = swiglu.output;
  run.build_log += swiglu.build_log;
  run.program_build_ms += swiglu.program_build_ms;
  run.timing.swiglu_min_us = swiglu.min_us;
  run.timing.swiglu_mean_us = swiglu.mean_us;
  run.timing.swiglu_global_work_items = swiglu.global_work_items;

  run.down.assign(kWeightedValueCount, 0.0f);
  for (std::uint64_t selected = 0; selected < kExpertUsedCount; ++selected) {
    const auto raw_offset =
        static_cast<std::size_t>(selected * kHiddenSize * down_blocks_per_row * kQ4KBlockBytes);
    const auto raw_count =
        static_cast<std::size_t>(kHiddenSize * down_blocks_per_row * kQ4KBlockBytes);
    const auto expert_raw = SliceBytes(down_raw, raw_offset, raw_count);
    const auto expert_packed =
        iq36::PackQ4Kx8(expert_raw, kHiddenSize, down_blocks_per_row);
    const auto input_offset = static_cast<std::size_t>(selected * kIntermediateSize);
    const auto expert_input = SliceFloats(run.swiglu, input_offset, kIntermediateSize);
    const auto bridge_begin = std::chrono::steady_clock::now();
    const auto q8_down = iq36::QuantizeQ8KInputPlanes(expert_input);
    const auto bridge_end = std::chrono::steady_clock::now();
    run.timing.host_q8_bridge_us +=
        std::chrono::duration<double, std::micro>(bridge_end - bridge_begin).count();
    run.timing.host_q8_bridge_count += 1;
    const auto expert_down =
        runner.Run(expert_packed, q8_down.qs, q8_down.bsums, q8_down.d,
                   kHiddenSize, down_blocks_per_row, repeat,
                   iq36::GpuQ4X8KernelVariant::kRowlaneParallel);
    std::copy(expert_down.output.begin(), expert_down.output.end(),
              run.down.begin() + static_cast<std::ptrdiff_t>(selected * kHiddenSize));
    run.timing.down_min_us += expert_down.timing.min_us;
    run.timing.down_mean_us += expert_down.timing.mean_us;
    run.timing.down_global_work_items += expert_down.timing.global_work_items;
    run.timing.down_kernel_launches += 1;
  }
  run.timing.selected_ffn_kernel_sum_min_us =
      run.timing.gate_up_min_us + run.timing.swiglu_min_us +
      run.timing.down_min_us;
  run.timing.selected_ffn_kernel_sum_mean_us =
      run.timing.gate_up_mean_us + run.timing.swiglu_mean_us +
      run.timing.down_mean_us;
  return run;
}
'''


NEW_MAIN_CPP = r'''
int main(int argc, char** argv) {
  try {
    const Args args = ParseArgs(argc, argv);
    const auto index = iq36::parse_gguf_model_index(args.model_path);
    const auto load_map = iq36::validate_qwen36_load_map(index);
    const std::string gate_up_tensor_name =
        LayerTensorName(args.layer, "ffn_gate_up_exps.weight");
    const std::string down_tensor_name =
        LayerTensorName(args.layer, "ffn_down_exps.weight");
    const std::string shared_gate_tensor_name =
        LayerTensorName(args.layer, "ffn_gate_inp_shexp.weight");
    const auto* gate_up_tensor = iq36::find_tensor(index, gate_up_tensor_name);
    const auto* down_tensor = iq36::find_tensor(index, down_tensor_name);
    const auto* shared_gate_tensor =
        iq36::find_tensor(index, shared_gate_tensor_name);
    Require(gate_up_tensor != nullptr, "selected expert gate-up tensor missing");
    Require(down_tensor != nullptr, "selected expert down tensor missing");
    Require(shared_gate_tensor != nullptr, "shared expert gate tensor missing");
    const bool gate_up_tensor_shape_ok =
        gate_up_tensor->type == 12 &&
        gate_up_tensor->dims ==
            std::vector<std::uint64_t>{kHiddenSize, kGateUpRowsPerExpert, kExpertCount};
    const bool down_tensor_shape_ok =
        down_tensor->type == 12 &&
        down_tensor->dims ==
            std::vector<std::uint64_t>{kIntermediateSize, kHiddenSize, kExpertCount};
    const bool shared_gate_tensor_shape_ok =
        shared_gate_tensor->type == 0 &&
        shared_gate_tensor->dims == std::vector<std::uint64_t>{kHiddenSize};

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
    const auto ffn_shexp =
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
        ffn_shexp.size() == kHiddenSize &&
        oracle_shared_gate.size() == 1 &&
        oracle_shared_sigmoid.size() == 1 &&
        oracle_shared_gated.size() == kHiddenSize &&
        oracle_ffn_out.size() == kHiddenSize &&
        oracle_layer_output.size() == kHiddenSize;
    Require(payload_counts_ok, "payload size mismatch");

    std::ifstream model(args.model_path, std::ios::binary);
    Require(static_cast<bool>(model), "model file could not be opened");
    const auto gate_weights =
        ReadF32TensorPayload(model, *shared_gate_tensor,
                             static_cast<std::size_t>(kHiddenSize));

    const auto native_gate_up =
        iq36::matvec_expert_tensor(args.model_path, index, gate_up_tensor_name,
                                   attn_post_norm, expert_ids);
    const auto native_swiglu =
        iq36::apply_swiglu_from_gate_up(native_gate_up, kIntermediateSize,
                                        kExpertUsedCount);
    const auto native_down =
        iq36::matvec_expert_tensor_per_expert_input(
            args.model_path, index, down_tensor_name, native_swiglu, expert_ids);
    const auto native_weighted =
        iq36::apply_expert_weights(native_down, weights_norm, kHiddenSize);
    const auto native_moe_out =
        iq36::aggregate_experts(native_weighted, kExpertUsedCount, kHiddenSize);
    const auto native_shared_gate =
        iq36::matvec_tensor(args.model_path, index, shared_gate_tensor_name, attn_post_norm);
    Require(native_shared_gate.size() == 1, "native shared gate size mismatch");
    const std::vector<float> native_shared_sigmoid{
        iq36::sigmoid_scalar(native_shared_gate[0])};
    const auto native_shared_gated =
        iq36::multiply_by_scalar(ffn_shexp, native_shared_sigmoid[0]);
    const auto native_ffn_out =
        iq36::add_vectors(native_moe_out, native_shared_gated);
    const auto native_layer_output =
        iq36::add_vectors(attn_residual, native_ffn_out);

    const auto selected_gpu = RunGpuSelectedFfnShell(
        args.model_path, *gate_up_tensor, *down_tensor, attn_post_norm,
        expert_ids, args.device_substring, args.repeat);
    const auto gpu = RunGpuShell(gate_weights, attn_post_norm, selected_gpu.down,
                                 weights_norm, ffn_shexp, attn_residual,
                                 args.device_substring, args.repeat);

    const auto gate_up_cpu_vs_oracle =
        iq36::compare_vectors(native_gate_up, oracle_gate_up, kMismatchThreshold);
    const auto gate_up_gpu_vs_cpu =
        iq36::compare_vectors(selected_gpu.gate_up, native_gate_up, kMismatchThreshold);
    const auto gate_up_gpu_vs_oracle =
        iq36::compare_vectors(selected_gpu.gate_up, oracle_gate_up, kMismatchThreshold);
    const auto swiglu_cpu_vs_oracle =
        iq36::compare_vectors(native_swiglu, oracle_swiglu, kMismatchThreshold);
    const auto swiglu_gpu_vs_cpu =
        iq36::compare_vectors(selected_gpu.swiglu, native_swiglu, kMismatchThreshold);
    const auto swiglu_gpu_vs_oracle =
        iq36::compare_vectors(selected_gpu.swiglu, oracle_swiglu, kMismatchThreshold);
    const auto down_cpu_vs_oracle =
        iq36::compare_vectors(native_down, oracle_down, kMismatchThreshold);
    const auto down_gpu_vs_cpu =
        iq36::compare_vectors(selected_gpu.down, native_down, kMismatchThreshold);
    const auto down_gpu_vs_oracle =
        iq36::compare_vectors(selected_gpu.down, oracle_down, kMismatchThreshold);
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
    const auto shared_gate_cpu_vs_oracle =
        iq36::compare_vectors(native_shared_gate, oracle_shared_gate, kMismatchThreshold);
    const auto shared_gate_gpu_vs_cpu =
        iq36::compare_vectors(gpu.shared_gate, native_shared_gate, kMismatchThreshold);
    const auto shared_gate_gpu_vs_oracle =
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
        ComparePassed(gate_up_cpu_vs_oracle) &&
        ComparePassed(gate_up_gpu_vs_cpu) &&
        ComparePassed(gate_up_gpu_vs_oracle) &&
        ComparePassed(swiglu_cpu_vs_oracle) &&
        ComparePassed(swiglu_gpu_vs_cpu) &&
        ComparePassed(swiglu_gpu_vs_oracle) &&
        ComparePassed(down_cpu_vs_oracle) &&
        ComparePassed(down_gpu_vs_cpu) &&
        ComparePassed(down_gpu_vs_oracle);
    const bool tail_comparisons_passed =
        ComparePassed(weighted_cpu_vs_oracle) &&
        ComparePassed(weighted_gpu_vs_cpu) &&
        ComparePassed(weighted_gpu_vs_oracle) &&
        ComparePassed(moe_out_cpu_vs_oracle) &&
        ComparePassed(moe_out_gpu_vs_cpu) &&
        ComparePassed(moe_out_gpu_vs_oracle) &&
        ComparePassed(shared_gate_cpu_vs_oracle) &&
        ComparePassed(shared_gate_gpu_vs_cpu) &&
        ComparePassed(shared_gate_gpu_vs_oracle) &&
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
        selected_gpu.timing.down_min_us > 0.0 &&
        selected_gpu.timing.selected_ffn_kernel_sum_min_us > 0.0;
    const bool tail_timing_positive =
        gpu.timing.weighted_min_us > 0.0 &&
        gpu.timing.shared_gate_matvec_min_us > 0.0 &&
        gpu.timing.shared_gate_apply_min_us > 0.0 &&
        gpu.timing.ffn_output_add_min_us > 0.0 &&
        gpu.timing.residual_add_min_us > 0.0 &&
        gpu.timing.shell_sum_min_us > 0.0;
    const bool arc_selected =
        selected_gpu.device_name.find(args.device_substring) != std::string::npos &&
        gpu.device_name.find(args.device_substring) != std::string::npos;
    const bool required_checks_passed =
        load_map.ready &&
        gate_up_tensor_shape_ok &&
        down_tensor_shape_ok &&
        shared_gate_tensor_shape_ok &&
        payload_counts_ok &&
        arc_selected &&
        selected_comparisons_passed &&
        tail_comparisons_passed &&
        selected_timing_positive &&
        tail_timing_positive;

    std::cout << std::setprecision(10);
    std::cout << "{";
    std::cout << "\"schema_version\":\"intel-qwen36-gpu-resident-selected-ffn-shell-probe-v0\",";
    std::cout << "\"model_path\":\"" << JsonEscape(args.model_path) << "\",";
    std::cout << "\"layer\":" << args.layer << ",";
    std::cout << "\"source_token_position\":" << kSourceTokenPosition << ",";
    std::cout << "\"resident_api\":\"selected_expert_ffn_load_once_run_many\",";
    std::cout << "\"resident_load_count\":1,";
    std::cout << "\"resident_shell_invocations\":" << args.repeat << ",";
    std::cout << "\"host_q8_bridge_for_selected_down\":true,";
    std::cout << "\"selected_down_q4_expert_launches\":"
              << selected_gpu.timing.down_kernel_launches << ",";
    std::cout << "\"gate_up_tensor_name\":\"" << JsonEscape(gate_up_tensor_name) << "\",";
    std::cout << "\"down_tensor_name\":\"" << JsonEscape(down_tensor_name) << "\",";
    std::cout << "\"shared_gate_tensor_name\":\"" << JsonEscape(shared_gate_tensor_name) << "\",";
    std::cout << "\"gate_up_tensor_type\":\"" << iq36::ggml_type_name(gate_up_tensor->type) << "\",";
    std::cout << "\"down_tensor_type\":\"" << iq36::ggml_type_name(down_tensor->type) << "\",";
    std::cout << "\"hidden_size\":" << kHiddenSize << ",";
    std::cout << "\"intermediate_size\":" << kIntermediateSize << ",";
    std::cout << "\"selected_expert_count\":" << kExpertUsedCount << ",";
    std::cout << "\"weighted_value_count\":" << kWeightedValueCount << ",";
    std::cout << "\"platform_name\":\"" << JsonEscape(selected_gpu.platform_name) << "\",";
    std::cout << "\"device_name\":\"" << JsonEscape(selected_gpu.device_name) << "\",";
    std::cout << "\"tail_device_name\":\"" << JsonEscape(gpu.device_name) << "\",";
    std::cout << "\"program_build_ms\":" << (selected_gpu.program_build_ms + gpu.program_build_ms) << ",";
    std::cout << "\"build_log\":\"" << JsonEscape(selected_gpu.build_log + gpu.build_log) << "\",";
    std::cout << "\"repeat\":" << args.repeat << ",";
    std::cout << "\"timings\":{";
    std::cout << "\"selected_gate_up_min_us\":" << selected_gpu.timing.gate_up_min_us << ",";
    std::cout << "\"selected_gate_up_mean_us\":" << selected_gpu.timing.gate_up_mean_us << ",";
    std::cout << "\"selected_swiglu_min_us\":" << selected_gpu.timing.swiglu_min_us << ",";
    std::cout << "\"selected_swiglu_mean_us\":" << selected_gpu.timing.swiglu_mean_us << ",";
    std::cout << "\"selected_down_min_us\":" << selected_gpu.timing.down_min_us << ",";
    std::cout << "\"selected_down_mean_us\":" << selected_gpu.timing.down_mean_us << ",";
    std::cout << "\"selected_down_host_q8_bridge_us\":" << selected_gpu.timing.host_q8_bridge_us << ",";
    std::cout << "\"selected_down_host_q8_bridge_count\":" << selected_gpu.timing.host_q8_bridge_count << ",";
    std::cout << "\"selected_down_q4_expert_launches\":" << selected_gpu.timing.down_kernel_launches << ",";
    std::cout << "\"selected_ffn_kernel_sum_min_us\":" << selected_gpu.timing.selected_ffn_kernel_sum_min_us << ",";
    std::cout << "\"selected_ffn_kernel_sum_mean_us\":" << selected_gpu.timing.selected_ffn_kernel_sum_mean_us << ",";
    std::cout << "\"moe_weighted_aggregate_min_us\":" << gpu.timing.weighted_min_us << ",";
    std::cout << "\"moe_weighted_aggregate_mean_us\":" << gpu.timing.weighted_mean_us << ",";
    std::cout << "\"shared_gate_matvec_min_us\":" << gpu.timing.shared_gate_matvec_min_us << ",";
    std::cout << "\"shared_gate_matvec_mean_us\":" << gpu.timing.shared_gate_matvec_mean_us << ",";
    std::cout << "\"shared_gate_apply_min_us\":" << gpu.timing.shared_gate_apply_min_us << ",";
    std::cout << "\"shared_gate_apply_mean_us\":" << gpu.timing.shared_gate_apply_mean_us << ",";
    std::cout << "\"ffn_output_add_min_us\":" << gpu.timing.ffn_output_add_min_us << ",";
    std::cout << "\"ffn_output_add_mean_us\":" << gpu.timing.ffn_output_add_mean_us << ",";
    std::cout << "\"post_ffn_residual_add_min_us\":" << gpu.timing.residual_add_min_us << ",";
    std::cout << "\"post_ffn_residual_add_mean_us\":" << gpu.timing.residual_add_mean_us << ",";
    std::cout << "\"resident_selected_ffn_to_layer_kernel_sum_min_us\":"
              << (selected_gpu.timing.selected_ffn_kernel_sum_min_us +
                  gpu.timing.shell_sum_min_us) << ",";
    std::cout << "\"resident_selected_ffn_to_layer_kernel_sum_mean_us\":"
              << (selected_gpu.timing.selected_ffn_kernel_sum_mean_us +
                  gpu.timing.shell_sum_mean_us) << ",";
    std::cout << "\"selected_gate_up_global_work_items\":"
              << selected_gpu.timing.gate_up_global_work_items << ",";
    std::cout << "\"selected_swiglu_global_work_items\":"
              << selected_gpu.timing.swiglu_global_work_items << ",";
    std::cout << "\"selected_down_global_work_items\":"
              << selected_gpu.timing.down_global_work_items;
    std::cout << "},";
    std::cout << "\"comparisons\":{";
    std::cout << "\"selected_gate_up\":";
    WriteCompareGroup(gate_up_cpu_vs_oracle, gate_up_gpu_vs_cpu, gate_up_gpu_vs_oracle);
    std::cout << ",\"selected_swiglu\":";
    WriteCompareGroup(swiglu_cpu_vs_oracle, swiglu_gpu_vs_cpu, swiglu_gpu_vs_oracle);
    std::cout << ",\"selected_down\":";
    WriteCompareGroup(down_cpu_vs_oracle, down_gpu_vs_cpu, down_gpu_vs_oracle);
    std::cout << ",\"ffn_moe_weighted\":";
    WriteCompareGroup(weighted_cpu_vs_oracle, weighted_gpu_vs_cpu, weighted_gpu_vs_oracle);
    std::cout << ",\"ffn_moe_out\":";
    WriteCompareGroup(moe_out_cpu_vs_oracle, moe_out_gpu_vs_cpu, moe_out_gpu_vs_oracle);
    std::cout << ",\"shared_gate\":";
    WriteCompareGroup(shared_gate_cpu_vs_oracle, shared_gate_gpu_vs_cpu, shared_gate_gpu_vs_oracle);
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
    std::cout << "\"gate_up_tensor_shape_ok\":" << (gate_up_tensor_shape_ok ? "true" : "false") << ",";
    std::cout << "\"down_tensor_shape_ok\":" << (down_tensor_shape_ok ? "true" : "false") << ",";
    std::cout << "\"shared_gate_tensor_shape_ok\":" << (shared_gate_tensor_shape_ok ? "true" : "false") << ",";
    std::cout << "\"payload_counts_ok\":" << (payload_counts_ok ? "true" : "false") << ",";
    std::cout << "\"arc_device_selected\":" << (arc_selected ? "true" : "false") << ",";
    std::cout << "\"selected_expert_ffn_matches_oracle\":"
              << (selected_comparisons_passed ? "true" : "false") << ",";
    std::cout << "\"tail_matches_oracle\":"
              << (tail_comparisons_passed ? "true" : "false") << ",";
    std::cout << "\"host_q8_bridge_for_selected_down\":true,";
    std::cout << "\"resident_load_once\":true,";
    std::cout << "\"resident_shell_invocations_positive\":"
              << (args.repeat > 0 ? "true" : "false") << ",";
    std::cout << "\"gpu_event_timing_positive\":"
              << ((selected_timing_positive && tail_timing_positive) ? "true" : "false") << ",";
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


def selected_ffn_probe_cpp(opencl_source: str) -> str:
  cpp = CAPTURED.PROBE_CPP
  cpp = cpp.replace(
      '#include "intel_qwen36/gguf_loader.hpp"\n',
      '#include "intel_qwen36/gguf_loader.hpp"\n#include "intel_qwen36/gpu_q4x8_matvec.hpp"\n',
      1,
  )
  marker = "constexpr int kWeightedValueCount = kHiddenSize * kExpertUsedCount;\n"
  cpp = cpp.replace(marker, marker + SELECTED_CONSTANTS_CPP + "\n", 1)
  helper_marker = "\nvoid WriteCompare(const iq36::VectorCompareStats& stats) {\n"
  cpp = cpp.replace(helper_marker, "\n" + SELECTED_HELPERS_CPP + helper_marker, 1)
  main_index = cpp.index("\nint main(")
  cpp = cpp[:main_index] + "\n" + NEW_MAIN_CPP
  return cpp.replace(
      "@@OPENCL_SOURCE_LITERAL@@",
      CAPTURED.cpp_raw_string_literal(opencl_source),
  )


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


def resolve_payloads(layer: int) -> dict[str, dict[str, Any]]:
  payloads: dict[str, dict[str, Any]] = {}
  for name, (stage_name, pattern, expected_bytes) in PAYLOAD_SPECS.items():
    path = PAYLOAD_ROOT / pattern.format(layer=layer)
    if not path.exists():
      raise SystemExit(f"missing payload for {name}: {path}")
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
      raise SystemExit(
          f"payload byte mismatch for {name}: expected {expected_bytes}, got {actual_bytes}: {path}"
      )
    payloads[name] = {
        "stage_name": stage_name,
        "local_path": path,
        "source_name": path.name,
        "bytes": actual_bytes,
        "sha256": iq36_local.sha256_file(path),
    }
  return payloads


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
      and probe.get("resident_api") == "selected_expert_ffn_load_once_run_many"
      and probe.get("resident_load_count") == 1
      and probe.get("resident_shell_invocations") == expected_invocations
      and probe.get("host_q8_bridge_for_selected_down") is True
      and probe.get("selected_down_q4_expert_launches") == 8
      and nested_bool(probe, "checks", "resident_load_once")
      and nested_bool(probe, "checks", "resident_shell_invocations_positive")
  )


def write_summary(path: Path, payload: dict[str, Any]) -> None:
  probe = payload.get("probe") if isinstance(payload.get("probe"), dict) else {}
  timings = probe.get("timings", {}) if isinstance(probe, dict) else {}
  comparisons = probe.get("comparisons", {}) if isinstance(probe, dict) else {}
  lines = [
      "# GPU Resident Selected-Expert FFN Shell Probe",
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
      f"- selected down Q4 expert launches: `{probe.get('selected_down_q4_expert_launches')}`",
      f"- host Q8 bridge for selected down: `{str(probe.get('host_q8_bridge_for_selected_down')).lower()}`",
      "",
      "| output | comparison | max abs | RMSE |",
      "|---|---|---:|---:|",
  ]
  for name in (
      "selected_gate_up",
      "selected_swiglu",
      "selected_down",
      "ffn_moe_weighted",
      "ffn_moe_out",
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
      "| kernel group | min us | mean us |",
      "|---|---:|---:|",
      f"| selected_gate_up | {timings.get('selected_gate_up_min_us')} | {timings.get('selected_gate_up_mean_us')} |",
      f"| selected_swiglu | {timings.get('selected_swiglu_min_us')} | {timings.get('selected_swiglu_mean_us')} |",
      f"| selected_down | {timings.get('selected_down_min_us')} | {timings.get('selected_down_mean_us')} |",
      f"| selected_ffn_kernel_sum | {timings.get('selected_ffn_kernel_sum_min_us')} | {timings.get('selected_ffn_kernel_sum_mean_us')} |",
      f"| resident_selected_ffn_to_layer_sum | {timings.get('resident_selected_ffn_to_layer_kernel_sum_min_us')} | {timings.get('resident_selected_ffn_to_layer_kernel_sum_mean_us')} |",
      "",
      "The target-side process loads selected expert slices once and runs the",
      "selected gate-up, selected SwiGLU, selected down, MoE aggregation, shared",
      "gate, and residual tail inside one process. The Q4 selected-down path still",
      "uses a host Q8 activation bridge per selected expert; this is captured",
      "single-layer evidence only, not prompt/token decode throughput.",
      "",
  ]
  path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  args = parse_args()
  if args.resident_invocations <= 0:
    raise SystemExit("--resident-invocations must be positive")
  stamp = utc_stamp()
  created_at = iso_now()
  out_dir = args.out_dir or ROOT / f"output/gpu-resident-selected-ffn-shell-probe-{stamp}"
  out_dir.mkdir(parents=True, exist_ok=False)
  raw_dir = out_dir / "raw"
  raw_dir.mkdir()

  payloads = resolve_payloads(args.layer)
  opencl_source = OPENCL_SOURCE.read_text(encoding="utf-8")
  opencl_source_hash = iq36_local.sha256_file(OPENCL_SOURCE)
  local_cpp = out_dir / "gpu_resident_selected_ffn_shell_probe.cpp"
  local_cpp.write_text(selected_ffn_probe_cpp(opencl_source), encoding="utf-8")

  remote_dir = f"{args.remote_root.rstrip('/')}/gpu-resident-selected-ffn-shell-probe-{stamp}"
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
    for local, remote in SOURCE_FILES:
      transfers.append(iq36_local.copy_to(args.host, ROOT / local, f"{remote_dir}/{remote}", args.timeout_s))
    transfers.append(
        iq36_local.copy_to(
            args.host,
            local_cpp,
            f"{remote_dir}/tests/gpu_resident_selected_ffn_shell_probe.cpp",
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

  executable = f"{remote_dir}/build/iq36-gpu-resident-selected-ffn-shell-probe"
  compile_cmd = " && ".join([
      f"source {shlex.quote(args.env_script)} >/dev/null 2>&1",
      (
          "g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic "
          f"-I {shlex.quote(remote_dir + '/include')} "
          f"{shlex.quote(remote_dir + '/src/gguf_loader.cpp')} "
          f"{shlex.quote(remote_dir + '/src/gpu_q4x8_matvec.cpp')} "
          f"{shlex.quote(remote_dir + '/tests/gpu_resident_selected_ffn_shell_probe.cpp')} "
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
      "tool": "tools/intel-qwen36-gpu-resident-selected-ffn-shell-probe.py",
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
      "gpu_resident_selected_ffn_shell_probe",
      [
          ("required_checks_passed", required_checks_passed),
          ("resident_shell_invocations", args.resident_invocations),
          ("selected_ffn_kernel_sum_min_us", nested_number(timings, "selected_ffn_kernel_sum_min_us")),
          ("resident_selected_ffn_to_layer_kernel_sum_min_us", nested_number(timings, "resident_selected_ffn_to_layer_kernel_sum_min_us")),
          ("selected_down_host_q8_bridge_us", nested_number(timings, "selected_down_host_q8_bridge_us")),
          ("selected_down_gpu_vs_oracle_max_abs_diff", nested_number(comparisons, "selected_down", "gpu_vs_oracle", "max_abs_diff")),
          ("layer_output_gpu_vs_oracle_max_abs_diff", nested_number(comparisons, "layer_output", "gpu_vs_oracle", "max_abs_diff")),
      ],
  )
  write_summary(out_dir / "summary.md", payload)
  print(out_dir)
  return 0 if required_checks_passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
