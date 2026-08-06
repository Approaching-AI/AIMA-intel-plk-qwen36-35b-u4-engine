#!/usr/bin/env python3
"""Run the resident GPU layer-8 state/input handoff probe."""

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
V_Q6_TOOL = Path(__file__).with_name(
    "intel-qwen36-gpu-resident-layer7-full-attn-v-q6-handoff-probe.py"
)
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA_VERSION = "intel-qwen36-gpu-resident-layer8-state-input-handoff-probe-v0"
DEFAULT_HOST = "local"
DEFAULT_MODEL = "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf"
DEFAULT_ENV_SCRIPT = "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh"
DEFAULT_REMOTE_ROOT = "/home/intel/intel-qwen36-gpu"
DEFAULT_ORACLE_BUNDLE = ROOT / "oracle/r0-oracle-bundle-20260627T060028Z"
DEFAULT_ALL_HISTORY = ROOT / "output/r1-full-attn-all-history-capture-20260627T145615Z/history.json"
OPENCL_SOURCE = ROOT / "engine/gpu/opencl/q4x8_matvec.cl"
FFN_PAYLOAD_ROOT = ROOT / "output/r0-boundary-capture-conv-state-filter-probe-20260630T000001Z/remote-output/payloads"


def load_v_q6_tool() -> ModuleType:
  spec = importlib.util.spec_from_file_location("iq36_layer7_v_q6_probe", V_Q6_TOOL)
  if spec is None or spec.loader is None:
    raise SystemExit(f"failed to load layer7 V Q6 tool: {V_Q6_TOOL}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


V_Q6 = load_v_q6_tool()
CORE = V_Q6.CORE
L7_INPUT = V_Q6.L7_INPUT
TWO = V_Q6.TWO
PRECONV = V_Q6.PRECONV


FFN_PAYLOAD_SPECS = {
    "l7_ffn_topk": ("l7_ffn_moe_topk.bin", "ffn_moe_topk-7__tok15__ord*.bin", 32),
    "l7_ffn_weights_norm": ("l7_ffn_moe_weights_norm.bin", "ffn_moe_weights_norm-7__tok15__ord*.bin", 32),
    "l7_ffn_gate_up": ("l7_ffn_moe_gate_up.bin", "ffn_moe_gate_up-7__tok15__ord*.bin", 32768),
    "l7_ffn_swiglu": ("l7_ffn_moe_swiglu.bin", "ffn_moe_swiglu-7__tok15__ord*.bin", 16384),
    "l7_ffn_down": ("l7_ffn_moe_down.bin", "ffn_moe_down-7__tok15__ord*.bin", 65536),
    "l7_ffn_weighted": ("l7_ffn_moe_weighted.bin", "ffn_moe_weighted-7__tok15__ord*.bin", 65536),
    "l7_ffn_moe_out": ("l7_ffn_moe_out.bin", "ffn_moe_out-7__tok15__ord*.bin", 8192),
    "l7_ffn_shexp": ("l7_ffn_shexp.bin", "ffn_shexp-7__tok15__ord*.bin", 8192),
    "l7_shared_gate": ("l7_shared_expert_gate.bin", "shared_expert_gate-7__tok15__ord*.bin", 4),
    "l7_shared_gate_sigmoid": (
        "l7_shared_expert_gate_sigmoid.bin",
        "shared_expert_gate_sigmoid-7__tok15__ord*.bin",
        4,
    ),
    "l7_ffn_shexp_gated": ("l7_ffn_shexp_gated.bin", "ffn_shexp_gated-7__tok15__ord*.bin", 8192),
    "l7_ffn_out": ("l7_ffn_out.bin", "ffn_out-7__tok15__ord*.bin", 8192),
}


SELECTED_Q6_EXTRA_CPP = r'''

constexpr int kQ6KBlockBytes = 210;

struct Layer7SelectedQ8Planes {
  std::vector<std::int8_t> qs;
  std::vector<float> d;
  std::uint64_t blocks_per_expert = 0;
};

int NearestIntLayer7Q6(float value) {
  float shifted = value + 12582912.0f;
  int bits = 0;
  std::memcpy(&bits, &shifted, sizeof(bits));
  return (bits & 0x007fffff) - 0x00400000;
}

Layer7SelectedQ8Planes QuantizePerExpertQ8KLayer7Q6(
    const std::vector<float>& input,
    std::uint64_t selected_count,
    std::uint64_t values_per_expert) {
  constexpr std::uint64_t kQ8BlockValuesLayer7 = 256;
  Require(values_per_expert % kQ8BlockValuesLayer7 == 0,
          "layer7 selected Q8_K input requires 256-aligned experts");
  Require(input.size() == selected_count * values_per_expert,
          "layer7 selected Q8_K input size mismatch");
  Layer7SelectedQ8Planes planes;
  planes.blocks_per_expert = values_per_expert / kQ8BlockValuesLayer7;
  planes.qs.assign(input.size(), 0);
  planes.d.assign(static_cast<std::size_t>(selected_count * planes.blocks_per_expert), 0.0f);
  for (std::uint64_t selected = 0; selected < selected_count; ++selected) {
    for (std::uint64_t block = 0; block < planes.blocks_per_expert; ++block) {
      const auto value_base =
          static_cast<std::size_t>(selected * values_per_expert +
                                   block * kQ8BlockValuesLayer7);
      float max = 0.0f;
      float amax = 0.0f;
      for (int i = 0; i < 256; ++i) {
        const float abs_value =
            std::abs(input[value_base + static_cast<std::size_t>(i)]);
        if (abs_value > amax) {
          amax = abs_value;
          max = input[value_base + static_cast<std::size_t>(i)];
        }
      }
      if (amax == 0.0f) {
        continue;
      }
      const float iscale = -127.0f / max;
      for (int i = 0; i < 256; ++i) {
        const int quantized =
            std::min(127, NearestIntLayer7Q6(
                              iscale * input[value_base + static_cast<std::size_t>(i)]));
        planes.qs[value_base + static_cast<std::size_t>(i)] =
            static_cast<std::int8_t>(quantized);
      }
      planes.d[static_cast<std::size_t>(selected * planes.blocks_per_expert + block)] =
          1.0f / iscale;
    }
  }
  return planes;
}

std::vector<float> RunGpuSelectedDownQ6Layer7(
    const std::vector<std::uint8_t>& selected_raw,
    const Layer7SelectedQ8Planes& q8,
    std::uint64_t rows_per_expert,
    std::uint64_t blocks_per_row,
    std::uint64_t selected_count,
    const std::string& device_substring,
    int repeat,
    SelectedFfnTiming* timing,
    std::string* platform_name,
    std::string* device_name,
    std::string* build_log,
    double* program_build_ms) {
  Require(selected_raw.size() ==
              static_cast<std::size_t>(selected_count * rows_per_expert *
                                       blocks_per_row * kQ6KBlockBytes),
          "layer7 selected-down Q6 raw byte size mismatch");
  Require(q8.blocks_per_expert == blocks_per_row,
          "layer7 selected-down Q8 block count mismatch");
  std::vector<float> output(
      static_cast<std::size_t>(selected_count * rows_per_expert), 0.0f);
  OpenClApi api;
  const auto selected = SelectDevice(api, device_substring);
  *platform_name = selected.platform_name;
  *device_name = selected.device_name;

  cl_int err = kClSuccess;
  cl_context context =
      api.clCreateContext(nullptr, 1, &selected.device, nullptr, nullptr, &err);
  Check(err, "clCreateContext(layer7 selected down q6)");
  cl_command_queue queue =
      api.clCreateCommandQueue(context, selected.device, kClQueueProfilingEnable, &err);
  Check(err, "clCreateCommandQueue(layer7 selected down q6)");
  const char* source = kOpenClSource;
  const std::size_t source_len = std::strlen(kOpenClSource);
  cl_program program =
      api.clCreateProgramWithSource(context, 1, &source, &source_len, &err);
  Check(err, "clCreateProgramWithSource(layer7 selected down q6)");
  const auto build_begin = std::chrono::steady_clock::now();
  err = api.clBuildProgram(program, 1, &selected.device, "", nullptr, nullptr);
  const auto build_end = std::chrono::steady_clock::now();
  *program_build_ms +=
      std::chrono::duration<double, std::milli>(build_end - build_begin).count();
  *build_log += BuildLog(api, program, selected.device);
  Check(err, "clBuildProgram(layer7 selected down q6)");
  cl_kernel kernel = api.clCreateKernel(program, "q6k_selected_down_matvec_row", &err);
  Check(err, "clCreateKernel(q6k_selected_down_matvec_row)");

  cl_mem raw_buffer = nullptr;
  cl_mem q8_qs_buffer = nullptr;
  cl_mem q8_d_buffer = nullptr;
  cl_mem output_buffer = nullptr;
  try {
    raw_buffer =
        api.clCreateBuffer(context, kClMemReadOnly, selected_raw.size(), nullptr, &err);
    Check(err, "clCreateBuffer(layer7 selected down q6 raw)");
    q8_qs_buffer =
        api.clCreateBuffer(context, kClMemReadOnly,
                           q8.qs.size() * sizeof(std::int8_t), nullptr, &err);
    Check(err, "clCreateBuffer(layer7 selected down q8 qs)");
    q8_d_buffer =
        api.clCreateBuffer(context, kClMemReadOnly,
                           q8.d.size() * sizeof(float), nullptr, &err);
    Check(err, "clCreateBuffer(layer7 selected down q8 d)");
    output_buffer =
        api.clCreateBuffer(context, kClMemWriteOnly,
                           output.size() * sizeof(float), nullptr, &err);
    Check(err, "clCreateBuffer(layer7 selected down output)");
    Check(api.clEnqueueWriteBuffer(queue, raw_buffer, kClTrue, 0,
                                   selected_raw.size(), selected_raw.data(),
                                   0, nullptr, nullptr),
          "clEnqueueWriteBuffer(layer7 selected down q6 raw)");
    Check(api.clEnqueueWriteBuffer(queue, q8_qs_buffer, kClTrue, 0,
                                   q8.qs.size() * sizeof(std::int8_t),
                                   q8.qs.data(), 0, nullptr, nullptr),
          "clEnqueueWriteBuffer(layer7 selected down q8 qs)");
    Check(api.clEnqueueWriteBuffer(queue, q8_d_buffer, kClTrue, 0,
                                   q8.d.size() * sizeof(float), q8.d.data(),
                                   0, nullptr, nullptr),
          "clEnqueueWriteBuffer(layer7 selected down q8 d)");
    const cl_uint rows_per_expert_arg = static_cast<cl_uint>(rows_per_expert);
    const cl_uint blocks_per_row_arg = static_cast<cl_uint>(blocks_per_row);
    Check(api.clSetKernelArg(kernel, 0, sizeof(raw_buffer), &raw_buffer),
          "clSetKernelArg(layer7 selected down q6 0)");
    Check(api.clSetKernelArg(kernel, 1, sizeof(q8_qs_buffer), &q8_qs_buffer),
          "clSetKernelArg(layer7 selected down q6 1)");
    Check(api.clSetKernelArg(kernel, 2, sizeof(q8_d_buffer), &q8_d_buffer),
          "clSetKernelArg(layer7 selected down q6 2)");
    Check(api.clSetKernelArg(kernel, 3, sizeof(rows_per_expert_arg), &rows_per_expert_arg),
          "clSetKernelArg(layer7 selected down q6 3)");
    Check(api.clSetKernelArg(kernel, 4, sizeof(blocks_per_row_arg), &blocks_per_row_arg),
          "clSetKernelArg(layer7 selected down q6 4)");
    Check(api.clSetKernelArg(kernel, 5, sizeof(output_buffer), &output_buffer),
          "clSetKernelArg(layer7 selected down q6 5)");
    const std::size_t global = output.size();
    std::vector<double> times;
    times.reserve(static_cast<std::size_t>(repeat));
    for (int i = 0; i < repeat; ++i) {
      cl_event event = nullptr;
      Check(api.clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global,
                                       nullptr, 0, nullptr, &event),
            "clEnqueueNDRangeKernel(layer7 selected down q6)");
      Check(api.clFinish(queue), "clFinish(layer7 selected down q6)");
      times.push_back(EventUs(api, event));
      api.clReleaseEvent(event);
    }
    Check(api.clEnqueueReadBuffer(queue, output_buffer, kClTrue, 0,
                                  output.size() * sizeof(float), output.data(),
                                  0, nullptr, nullptr),
          "clEnqueueReadBuffer(layer7 selected down q6 output)");
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
'''


def replace_once(text: str, old: str, new: str) -> str:
  count = text.count(old)
  if count != 1:
    raise SystemExit(f"expected exactly one source replacement for {old[:80]!r}, found {count}")
  return text.replace(old, new, 1)


def patch_selected_down_q6(cpp: str) -> str:
  cpp = replace_once(
      cpp,
      '  Require(down_tensor.type == 12, "selected down tensor must be Q4_K for this gate");',
      '  Require(down_tensor.type == 12 || down_tensor.type == 14,\n'
      '          "selected down tensor must be Q4_K or Q6_K");',
  )
  cpp = replace_once(
      cpp,
      '''  Require(down_row_nbytes == down_blocks_per_row * kQ4KBlockBytes,
          "selected down Q4 row byte mismatch");
''',
      '''  const std::uint64_t down_block_nbytes =
      down_tensor.type == 12 ? kQ4KBlockBytes : kQ6KBlockBytes;
  Require(down_row_nbytes == down_blocks_per_row * down_block_nbytes,
          "selected down row byte mismatch");
''',
  )
  cpp = replace_once(
      cpp,
      '''  run.down.assign(kWeightedValueCount, 0.0f);
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
''',
      '''  run.down.assign(kWeightedValueCount, 0.0f);
  if (down_tensor.type == 12) {
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
  } else {
    const auto bridge_begin = std::chrono::steady_clock::now();
    const auto q8_down =
        QuantizePerExpertQ8KLayer7Q6(run.swiglu, kExpertUsedCount, kIntermediateSize);
    const auto bridge_end = std::chrono::steady_clock::now();
    run.timing.host_q8_bridge_us +=
        std::chrono::duration<double, std::micro>(bridge_end - bridge_begin).count();
    run.timing.host_q8_bridge_count += 1;
    run.down = RunGpuSelectedDownQ6Layer7(
        down_raw, q8_down, kHiddenSize, down_blocks_per_row, kExpertUsedCount,
        device_substring, repeat, &run.timing, &run.platform_name,
        &run.device_name, &run.build_log, &run.program_build_ms);
  }
''',
  )
  return cpp


def layer8_state_input_probe_cpp(opencl_source: str) -> str:
  cpp = V_Q6.v_q6_probe_cpp(opencl_source)
  cpp = patch_selected_down_q6(cpp)
  selected_shell_index = cpp.index("\nSelectedFfnRun RunGpuSelectedFfnShell(")
  cpp = cpp[:selected_shell_index] + "\n" + SELECTED_Q6_EXTRA_CPP + cpp[selected_shell_index:]
  cpp = replace_once(
      cpp,
      "constexpr int kQ6KBlockBytes = 210;\n\nstruct SharedFfnTiming",
      "struct SharedFfnTiming",
  )
  replacements = {
      "layer7 V Q6 handoff probe expects --layer 5": "layer8 state/input handoff probe expects --layer 5",
      V_Q6.SCHEMA_VERSION: SCHEMA_VERSION,
      "two_linear_layer_to_full_attention_v_q6_core_output_load_once_run_many":
          "two_linear_layer_to_full_attention_v_q6_live_ffn_lout_layer8_state_input_load_once_run_many",
      '    std::cout << "\\"full_attn_ffn_boundary\\":\\"q6_down_reference_pending\\",";':
          '    std::cout << "\\"full_attn_ffn_boundary\\":\\"gpu_live_post_norm_to_q6_down\\",";',
  }
  for old, new in replacements.items():
    cpp = replace_once(cpp, old, new)
  cpp = replace_once(
      cpp,
      '''    const auto oracle_attn_post_norm =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "full_attn_post_norm.bin"));
''',
      '''    const auto oracle_attn_post_norm =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "full_attn_post_norm.bin"));
    const auto oracle_expert_ids =
        ReadI32VectorFile(JoinPath(args.payload_dir, "l7_ffn_moe_topk.bin"));
    const auto oracle_weights_norm =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "l7_ffn_moe_weights_norm.bin"));
    const auto oracle_selected_gate_up =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "l7_ffn_moe_gate_up.bin"));
    const auto oracle_selected_swiglu =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "l7_ffn_moe_swiglu.bin"));
    const auto oracle_selected_down =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "l7_ffn_moe_down.bin"));
    const auto oracle_weighted =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "l7_ffn_moe_weighted.bin"));
    const auto oracle_moe_out =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "l7_ffn_moe_out.bin"));
    const auto oracle_shared_down =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "l7_ffn_shexp.bin"));
    const auto oracle_shared_gate =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "l7_shared_expert_gate.bin"));
    const auto oracle_shared_sigmoid =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "l7_shared_expert_gate_sigmoid.bin"));
    const auto oracle_shared_gated =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "l7_ffn_shexp_gated.bin"));
    const auto oracle_ffn_out =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "l7_ffn_out.bin"));
    const auto oracle_layer_output =
        iq36::add_vectors(oracle_attn_residual, oracle_ffn_out);
''',
  )
  cpp = replace_once(
      cpp,
      '''        FullAttentionPayloadCountsOk(layer2_oracle) &&
        oracle_attn_residual.size() == kHiddenSize &&
        oracle_attn_post_norm.size() == kHiddenSize;
''',
      '''        FullAttentionPayloadCountsOk(layer2_oracle) &&
        oracle_attn_residual.size() == kHiddenSize &&
        oracle_attn_post_norm.size() == kHiddenSize &&
        oracle_expert_ids.size() == kExpertUsedCount &&
        oracle_weights_norm.size() == kExpertUsedCount &&
        oracle_selected_gate_up.size() == kGateUpValueCount &&
        oracle_selected_swiglu.size() == kSwiGluValueCount &&
        oracle_selected_down.size() == kWeightedValueCount &&
        oracle_weighted.size() == kWeightedValueCount &&
        oracle_moe_out.size() == kHiddenSize &&
        oracle_shared_down.size() == kHiddenSize &&
        oracle_shared_gate.size() == 1 &&
        oracle_shared_sigmoid.size() == 1 &&
        oracle_shared_gated.size() == kHiddenSize &&
        oracle_ffn_out.size() == kHiddenSize &&
        oracle_layer_output.size() == kHiddenSize;
''',
  )
  cpp = replace_once(
      cpp,
      '''    const auto ffn_norm_weight =
        iq36::decode_tensor_row(args.model_path, index,
                                LayerTensorName(layer2, "post_attention_norm.weight"), 0);
''',
      '''    const auto ffn_norm_weight =
        iq36::decode_tensor_row(args.model_path, index,
                                LayerTensorName(layer2, "post_attention_norm.weight"), 0);
    const std::string selected_gate_up_tensor_name =
        LayerTensorName(layer2, "ffn_gate_up_exps.weight");
    const std::string selected_down_tensor_name =
        LayerTensorName(layer2, "ffn_down_exps.weight");
    const std::string shared_gate_tensor_name =
        LayerTensorName(layer2, "ffn_gate_shexp.weight");
    const std::string shared_up_tensor_name =
        LayerTensorName(layer2, "ffn_up_shexp.weight");
    const std::string shared_down_tensor_name =
        LayerTensorName(layer2, "ffn_down_shexp.weight");
    const std::string shared_input_gate_tensor_name =
        LayerTensorName(layer2, "ffn_gate_inp_shexp.weight");
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
    Require(selected_gate_up_tensor != nullptr, "layer7 selected gate-up tensor missing");
    Require(selected_down_tensor != nullptr, "layer7 selected down tensor missing");
    Require(shared_gate_tensor != nullptr, "layer7 shared gate tensor missing");
    Require(shared_up_tensor != nullptr, "layer7 shared up tensor missing");
    Require(shared_down_tensor != nullptr, "layer7 shared down tensor missing");
    Require(shared_input_gate_tensor != nullptr, "layer7 shared input gate tensor missing");
    const auto shared_input_gate_weights =
        ReadF32TensorPayload(model, *shared_input_gate_tensor,
                             static_cast<std::size_t>(kHiddenSize));
    const bool layer2_ffn_tensor_shapes_ok =
        selected_gate_up_tensor->type == 12 &&
        selected_down_tensor->type == 14 &&
        shared_gate_tensor->type == 12 &&
        shared_up_tensor->type == 12 &&
        shared_down_tensor->type == 14 &&
        shared_input_gate_tensor->type == 0 &&
        selected_gate_up_tensor->dims ==
            std::vector<std::uint64_t>{kHiddenSize, kGateUpRowsPerExpert, kExpertCount} &&
        selected_down_tensor->dims ==
            std::vector<std::uint64_t>{kIntermediateSize, kHiddenSize, kExpertCount} &&
        shared_gate_tensor->dims ==
            std::vector<std::uint64_t>{kHiddenSize, kIntermediateSize} &&
        shared_up_tensor->dims ==
            std::vector<std::uint64_t>{kHiddenSize, kIntermediateSize} &&
        shared_down_tensor->dims ==
            std::vector<std::uint64_t>{kIntermediateSize, kHiddenSize} &&
        shared_input_gate_tensor->dims == std::vector<std::uint64_t>{kHiddenSize};
''',
  )
  cpp = replace_once(
      cpp,
      '''    const auto attention_gpu = RunGpuAttentionFront(
        args.model_path,
        *iq36::find_tensor(index, LayerTensorName(layer2, "attn_output.weight")),
        core_gate_gpu.attn_gated,
        layer1_run.gpu_layer_output,
        ffn_norm_weight,
        rms_norm_epsilon,
        args.device_substring,
        args.repeat);
''',
      '''    const auto attention_gpu = RunGpuAttentionFront(
        args.model_path,
        *iq36::find_tensor(index, LayerTensorName(layer2, "attn_output.weight")),
        core_gate_gpu.attn_gated,
        layer1_run.gpu_layer_output,
        ffn_norm_weight,
        rms_norm_epsilon,
        args.device_substring,
        args.repeat);

    const auto& ffn_input = attention_gpu.attn_post_norm;
    const auto native_selected_gate_up =
        iq36::matvec_expert_tensor(args.model_path, index,
                                   selected_gate_up_tensor_name,
                                   ffn_input, oracle_expert_ids);
    const auto native_selected_swiglu =
        iq36::apply_swiglu_from_gate_up(native_selected_gate_up,
                                        kIntermediateSize, kExpertUsedCount);
    const auto native_selected_down =
        iq36::matvec_expert_tensor_per_expert_input(
            args.model_path, index, selected_down_tensor_name,
            native_selected_swiglu, oracle_expert_ids);
    const auto native_weighted =
        iq36::apply_expert_weights(native_selected_down, oracle_weights_norm, kHiddenSize);
    const auto native_moe_out =
        iq36::aggregate_experts(native_weighted, kExpertUsedCount, kHiddenSize);
    const auto native_shared_gate =
        iq36::matvec_tensor(args.model_path, index, shared_gate_tensor_name, ffn_input);
    const auto native_shared_up =
        iq36::matvec_tensor(args.model_path, index, shared_up_tensor_name, ffn_input);
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
                            shared_input_gate_tensor_name, ffn_input);
    Require(native_shared_input_gate.size() == 1,
            "native layer7 shared input gate size mismatch");
    const std::vector<float> native_shared_sigmoid{
        iq36::sigmoid_scalar(native_shared_input_gate[0])};
    const auto native_shared_gated =
        iq36::multiply_by_scalar(native_shared_down, native_shared_sigmoid[0]);
    const auto native_ffn_out =
        iq36::add_vectors(native_moe_out, native_shared_gated);
    const auto native_layer_output =
        iq36::add_vectors(attention_gpu.attn_residual, native_ffn_out);

    const auto selected_gpu = RunGpuSelectedFfnShell(
        args.model_path, *selected_gate_up_tensor, *selected_down_tensor,
        ffn_input, oracle_expert_ids, args.device_substring, args.repeat);
    const auto shared_gpu = RunGpuSharedFfnShell(
        args.model_path, *shared_gate_tensor, *shared_up_tensor,
        *shared_down_tensor, ffn_input, args.device_substring, args.repeat);
    const auto tail_gpu = RunGpuShell(shared_input_gate_weights, ffn_input,
                                      selected_gpu.down, oracle_weights_norm,
                                      shared_gpu.down, attention_gpu.attn_residual,
                                      args.device_substring, args.repeat);
''',
  )
  cpp = replace_once(
      cpp,
      '''    AppendCpuGpuOracleCompare(strict_groups, "l2_v",
                              native_qkv.v,
                              layer2_v_gpu.v,
                              layer2_oracle.v);
''',
      '''    AppendCpuGpuOracleCompare(strict_groups, "l2_v",
                              native_qkv.v,
                              layer2_v_gpu.v,
                              layer2_oracle.v);
    std::vector<NamedCompareGroup> ffn_live_groups;
    AppendCpuGpuOracleCompare(ffn_live_groups, "l2_selected_gate_up",
                              native_selected_gate_up,
                              selected_gpu.gate_up,
                              oracle_selected_gate_up);
    AppendCpuGpuOracleCompare(ffn_live_groups, "l2_selected_swiglu",
                              native_selected_swiglu,
                              selected_gpu.swiglu,
                              oracle_selected_swiglu);
    AppendCpuGpuOracleCompare(ffn_live_groups, "l2_selected_down",
                              native_selected_down,
                              selected_gpu.down,
                              oracle_selected_down);
    AppendCpuGpuOracleCompare(ffn_live_groups, "l2_ffn_moe_weighted",
                              native_weighted,
                              tail_gpu.weighted,
                              oracle_weighted);
    AppendCpuGpuOracleCompare(ffn_live_groups, "l2_ffn_moe_out",
                              native_moe_out,
                              tail_gpu.moe_out,
                              oracle_moe_out);
    AppendCpuGpuOracleCompare(ffn_live_groups, "l2_shared_down",
                              native_shared_down,
                              shared_gpu.down,
                              oracle_shared_down);
    AppendCpuGpuOracleCompare(ffn_live_groups, "l2_shared_gate",
                              native_shared_input_gate,
                              tail_gpu.shared_gate,
                              oracle_shared_gate);
    AppendCpuGpuOracleCompare(ffn_live_groups, "l2_shared_gate_sigmoid",
                              native_shared_sigmoid,
                              tail_gpu.shared_gate_sigmoid,
                              oracle_shared_sigmoid);
    AppendCpuGpuOracleCompare(ffn_live_groups, "l2_ffn_shexp_gated",
                              native_shared_gated,
                              tail_gpu.shared_gated,
                              oracle_shared_gated);
    AppendCpuGpuOracleCompare(ffn_live_groups, "l2_ffn_out",
                              native_ffn_out,
                              tail_gpu.ffn_out,
                              oracle_ffn_out);
    AppendCpuGpuOracleCompare(ffn_live_groups, "l2_layer_output_derived",
                              native_layer_output,
                              tail_gpu.layer_output,
                              oracle_layer_output);
''',
  )
  cpp = replace_once(
      cpp,
      '''    const bool layer2_strict_input_ok =
        CompareGroupsPassed(strict_groups) &&
        ComparePassed(k_raw_gpu_vs_cpu);
    const bool layer2_full_component_ok =
        CompareGroupsPassedFullAttentionComponent(full_attention_groups);
    const bool layer2_comparisons_ok =
        layer2_strict_input_ok && layer2_full_component_ok;
''',
      '''    const bool layer2_strict_input_ok =
        CompareGroupsPassed(strict_groups) &&
        ComparePassed(k_raw_gpu_vs_cpu);
    const bool layer2_full_component_ok =
        CompareGroupsPassedFullAttentionComponent(full_attention_groups);
    const bool layer2_comparisons_ok =
        layer2_strict_input_ok && layer2_full_component_ok;
    const auto live_ffn_oracle_magnitude_passed =
        [](const iq36::VectorCompareStats& stats) {
          return stats.same_size &&
                 stats.finite &&
                 stats.mismatch_count == 0 &&
                 stats.max_abs_diff <= kMaxAbsDiffThreshold &&
                 stats.rmse <= kRmseThreshold;
        };
    bool layer2_live_ffn_gpu_cpu_ok = true;
    for (const auto& group : ffn_live_groups) {
      layer2_live_ffn_gpu_cpu_ok =
          layer2_live_ffn_gpu_cpu_ok && ComparePassed(group.gpu_vs_cpu);
    }
    const bool layer2_live_ffn_oracle_magnitude_ok =
        live_ffn_oracle_magnitude_passed(ffn_live_groups[2].gpu_vs_oracle) &&
        live_ffn_oracle_magnitude_passed(ffn_live_groups[5].gpu_vs_oracle) &&
        live_ffn_oracle_magnitude_passed(ffn_live_groups[9].gpu_vs_oracle) &&
        live_ffn_oracle_magnitude_passed(ffn_live_groups[10].gpu_vs_oracle);
    const bool layer2_live_layer_output_full_policy_ok =
        ComparePassedFullAttentionComponent(ffn_live_groups[10].gpu_vs_oracle);
    const bool layer2_live_ffn_lout_ok =
        layer2_live_ffn_gpu_cpu_ok &&
        layer2_live_ffn_oracle_magnitude_ok &&
        layer2_live_layer_output_full_policy_ok;
''',
  )
  cpp = replace_once(
      cpp,
      '''        layer2_comparisons_ok &&
        layer2_timing_positive &&
''',
      '''        layer2_comparisons_ok &&
        layer2_live_ffn_lout_ok &&
        layer2_timing_positive &&
''',
  )
  cpp = replace_once(
      cpp,
      '''        attention_gpu.timing.output_projection_min_us > 0.0 &&
        attention_gpu.timing.residual_add_min_us > 0.0 &&
        attention_gpu.timing.ffn_rmsnorm_min_us > 0.0;
''',
      '''        attention_gpu.timing.output_projection_min_us > 0.0 &&
        attention_gpu.timing.residual_add_min_us > 0.0 &&
        attention_gpu.timing.ffn_rmsnorm_min_us > 0.0 &&
        selected_gpu.timing.gate_up_min_us > 0.0 &&
        selected_gpu.timing.swiglu_min_us > 0.0 &&
        selected_gpu.timing.down_min_us > 0.0 &&
        shared_gpu.timing.gate_min_us > 0.0 &&
        shared_gpu.timing.up_min_us > 0.0 &&
        shared_gpu.timing.swiglu_min_us > 0.0 &&
        shared_gpu.timing.down_min_us > 0.0 &&
        tail_gpu.timing.shell_sum_min_us > 0.0;
''',
  )
  cpp = replace_once(
      cpp,
      '''        attention_gpu.device_name.find(args.device_substring) != std::string::npos;
''',
      '''        attention_gpu.device_name.find(args.device_substring) != std::string::npos &&
        selected_gpu.device_name.find(args.device_substring) != std::string::npos &&
        shared_gpu.device_name.find(args.device_substring) != std::string::npos &&
        tail_gpu.device_name.find(args.device_substring) != std::string::npos;
''',
  )
  cpp = replace_once(
      cpp,
      '''        ffn_q6_boundary &&
        args.repeat > 0;
''',
      '''        ffn_q6_boundary &&
        layer2_ffn_tensor_shapes_ok &&
        args.repeat > 0;
''',
  )
  cpp = replace_once(
      cpp,
      '''    const double two_layer_kernel_sum_min =
        layer0_run.timing.layer_kernel_sum_min_us +
        layer1_run.timing.layer_kernel_sum_min_us;
''',
      '''    const double two_layer_kernel_sum_min =
        layer0_run.timing.layer_kernel_sum_min_us +
        layer1_run.timing.layer_kernel_sum_min_us;
    const double layer2_ffn_sum_min =
        selected_gpu.timing.selected_ffn_kernel_sum_min_us +
        shared_gpu.timing.shared_ffn_kernel_sum_min_us +
        tail_gpu.timing.shell_sum_min_us;
''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << "\\"full_attn_ffn_boundary\\":\\"gpu_live_post_norm_to_q6_down\\",";''',
      '''    std::cout << "\\"full_attn_ffn_boundary\\":\\"gpu_live_post_norm_to_q6_down\\",";
    std::cout << "\\"ffn_input_boundary\\":\\"live_gpu_post_attention_norm\\",";
    std::cout << "\\"layer_output_residual_boundary\\":\\"live_gpu_attention_residual\\",";''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << "\\"layer7_v_tensor_type\\":\\""
              << JsonEscape(iq36::ggml_type_name(layer2_tensors.v_tensor->type)) << "\\",";''',
      '''    std::cout << "\\"layer7_v_tensor_type\\":\\""
              << JsonEscape(iq36::ggml_type_name(layer2_tensors.v_tensor->type)) << "\\",";
    std::cout << "\\"layer7_selected_down_tensor_type\\":\\""
              << JsonEscape(iq36::ggml_type_name(selected_down_tensor->type)) << "\\",";
    std::cout << "\\"layer7_shared_down_tensor_type\\":\\""
              << JsonEscape(iq36::ggml_type_name(shared_down_tensor->type)) << "\\",";''',
  )
  cpp = replace_once(
      cpp,
      '''                  layer2_v_gpu.program_build_ms +
                  core_gate_gpu.program_build_ms + attention_gpu.program_build_ms)
''',
      '''                  layer2_v_gpu.program_build_ms +
                  core_gate_gpu.program_build_ms + attention_gpu.program_build_ms +
                  selected_gpu.program_build_ms + shared_gpu.program_build_ms +
                  tail_gpu.program_build_ms)
''',
  )
  cpp = replace_once(
      cpp,
      '''                            layer2_v_gpu.build_log +
                            core_gate_gpu.build_log + attention_gpu.build_log)
''',
      '''                            layer2_v_gpu.build_log +
                            core_gate_gpu.build_log + attention_gpu.build_log +
                            selected_gpu.build_log + shared_gpu.build_log +
                            tail_gpu.build_log)
''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << "\\"resident_two_linear_plus_layer7_attention_kernel_sum_min_us\\":"
              << (two_layer_kernel_sum_min + layer2_attention_sum_min);
''',
      '''    std::cout << "\\"resident_two_linear_plus_layer7_attention_kernel_sum_min_us\\":"
              << (two_layer_kernel_sum_min + layer2_attention_sum_min) << ",";
    std::cout << "\\"resident_layer7_ffn_q6_down_kernel_sum_min_us\\":"
              << layer2_ffn_sum_min << ",";
    std::cout << "\\"resident_two_linear_plus_layer7_attention_ffn_kernel_sum_min_us\\":"
              << (two_layer_kernel_sum_min + layer2_attention_sum_min + layer2_ffn_sum_min);
''',
  )
  cpp = replace_once(
      cpp,
      '''    WriteNamedCompareGroups(strict_groups);
    std::cout << ",";
    WriteNamedCompareGroups(full_attention_groups);
    std::cout << ",\\"l2_k_raw\\":{\\"gpu_vs_cpu\\":";
''',
      '''    WriteNamedCompareGroups(strict_groups);
    std::cout << ",";
    WriteNamedCompareGroups(full_attention_groups);
    std::cout << ",";
    WriteNamedCompareGroups(ffn_live_groups);
    std::cout << ",\\"l2_k_raw\\":{\\"gpu_vs_cpu\\":";
''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << "\\"layer2_ffn_q6_boundary\\":"
              << (ffn_q6_boundary ? "true" : "false") << ",";''',
      '''    std::cout << "\\"layer2_live_ffn_gpu_cpu_matches_native\\":"
              << (layer2_live_ffn_gpu_cpu_ok ? "true" : "false") << ",";
    std::cout << "\\"layer2_live_ffn_oracle_magnitude_policy_matches\\":"
              << (layer2_live_ffn_oracle_magnitude_ok ? "true" : "false") << ",";
    std::cout << "\\"layer2_live_layer_output_full_policy_matches_oracle\\":"
              << (layer2_live_layer_output_full_policy_ok ? "true" : "false") << ",";
    std::cout << "\\"layer2_ffn_q6_boundary\\":"
              << (ffn_q6_boundary ? "true" : "false") << ",";
    std::cout << "\\"layer2_ffn_tensor_shapes_ok\\":"
              << (layer2_ffn_tensor_shapes_ok ? "true" : "false") << ",";''',
  )
  cpp = replace_once(
      cpp,
      '''    const int layer2 = args.layer + 2;
    Require(layer0 == 5 && layer1 == 6 && layer2 == 7,
            "layer8 state/input handoff probe expects --layer 5");
''',
      '''    const int layer2 = args.layer + 2;
    const int layer3 = args.layer + 3;
    Require(layer0 == 5 && layer1 == 6 && layer2 == 7 && layer3 == 8,
            "layer8 state/input handoff probe expects --layer 5");
''',
  )
  cpp = replace_once(
      cpp,
      '''    const auto layer2_tensors = ResolveFullAttentionTensorBundle(index, layer2);
    const auto layer0_oracle = LoadLayerOraclePayloads(args.payload_dir, "l0");
    const auto layer1_oracle = LoadLayerOraclePayloads(args.payload_dir, "l1");
    const auto layer2_oracle = LoadFullAttentionPayloads(args.payload_dir);
''',
      '''    const auto layer2_tensors = ResolveFullAttentionTensorBundle(index, layer2);
    const auto layer3_tensors = ResolveLayerTensorBundle(index, layer3);
    const auto layer0_oracle = LoadLayerOraclePayloads(args.payload_dir, "l0");
    const auto layer1_oracle = LoadLayerOraclePayloads(args.payload_dir, "l1");
    const auto layer2_oracle = LoadFullAttentionPayloads(args.payload_dir);
    const auto layer3_oracle = LoadLayerOraclePayloads(args.payload_dir, "l3");
''',
  )
  cpp = replace_once(
      cpp,
      '''    const auto tail_gpu = RunGpuShell(shared_input_gate_weights, ffn_input,
                                      selected_gpu.down, oracle_weights_norm,
                                      shared_gpu.down, attention_gpu.attn_residual,
                                      args.device_substring, args.repeat);
''',
      '''    const auto tail_gpu = RunGpuShell(shared_input_gate_weights, ffn_input,
                                      selected_gpu.down, oracle_weights_norm,
                                      shared_gpu.down, attention_gpu.attn_residual,
                                      args.device_substring, args.repeat);
    const auto layer3_run = RunResidentLinearLayerShell(
        args, index, layer3_tensors, layer3_oracle,
        native_layer_output, tail_gpu.layer_output, rms_norm_epsilon);
''',
  )
  cpp = replace_once(
      cpp,
      '''    const bool layer2_timing_positive =
        layer2_rms_gpu.timing.rmsnorm_min_us > 0.0 &&
''',
      '''    const auto find_l3_group =
        [&](const std::string& name) -> const NamedCompareGroup& {
          const auto found = std::find_if(
              layer3_run.comparisons.begin(),
              layer3_run.comparisons.end(),
              [&](const NamedCompareGroup& group) {
                return group.name == name;
              });
          Require(found != layer3_run.comparisons.end(),
                  "layer8 comparison missing: " + name);
          return *found;
        };
    const auto& layer3_residual_input = find_l3_group("residual_input");
    const auto& layer3_attn_norm = find_l3_group("attn_norm");
    const auto& layer3_qkv = find_l3_group("linear_attn_qkv_mixed");
    const auto& layer3_conv_output_raw = find_l3_group("conv_output_raw");
    bool layer3_gpu_cpu_ok = true;
    for (const auto& group : layer3_run.comparisons) {
      layer3_gpu_cpu_ok = layer3_gpu_cpu_ok && ComparePassed(group.gpu_vs_cpu);
    }
    layer3_gpu_cpu_ok =
        layer3_gpu_cpu_ok &&
        ComparePassed(layer3_run.conv_state_after_gpu_vs_cpu) &&
        ComparePassed(layer3_run.recurrent_state_gpu_vs_cpu);
    const bool layer3_state_input_oracle_policy_ok =
        ComparePassedFullAttentionComponent(layer3_residual_input.gpu_vs_oracle) &&
        ComparePassedFullAttentionComponent(layer3_attn_norm.gpu_vs_oracle) &&
        ComparePassedFullAttentionComponent(layer3_qkv.gpu_vs_oracle) &&
        ComparePassedFullAttentionComponent(layer3_conv_output_raw.gpu_vs_oracle);
    const bool layer3_shapes_ok = ShapesPassed(layer3_run.shape_checks);
    const bool layer3_ok =
        layer3_shapes_ok &&
        layer3_run.payload_counts_ok &&
        layer3_gpu_cpu_ok &&
        layer3_state_input_oracle_policy_ok &&
        layer3_run.timing_positive &&
        layer3_run.arc_selected;
    const bool layer2_timing_positive =
        layer2_rms_gpu.timing.rmsnorm_min_us > 0.0 &&
''',
  )
  cpp = replace_once(
      cpp,
      '''        layer2_live_ffn_lout_ok &&
        layer2_timing_positive &&
''',
      '''        layer2_live_ffn_lout_ok &&
        layer3_ok &&
        layer2_timing_positive &&
''',
  )
  cpp = replace_once(
      cpp,
      '''    const double layer2_ffn_sum_min =
        selected_gpu.timing.selected_ffn_kernel_sum_min_us +
        shared_gpu.timing.shared_ffn_kernel_sum_min_us +
        tail_gpu.timing.shell_sum_min_us;
''',
      '''    const double layer2_ffn_sum_min =
        selected_gpu.timing.selected_ffn_kernel_sum_min_us +
        shared_gpu.timing.shared_ffn_kernel_sum_min_us +
        tail_gpu.timing.shell_sum_min_us;
    const double layer3_state_input_sum_min =
        layer3_run.timing.layer_input_rmsnorm_min_us +
        layer3_run.timing.preconv_to_postconv_kernel_sum_min_us;
''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << "\\"layers\\":[" << layer0 << "," << layer1 << "," << layer2 << "],";
''',
      '''    std::cout << "\\"layers\\":[" << layer0 << "," << layer1 << "," << layer2 << "," << layer3 << "],";
''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << "\\"layer_output_residual_boundary\\":\\"live_gpu_attention_residual\\",";''',
      '''    std::cout << "\\"layer_output_residual_boundary\\":\\"live_gpu_attention_residual\\",";
    std::cout << "\\"layer8_residual_input_boundary\\":\\"live_gpu_l_out_7\\",";
    std::cout << "\\"layer8_conv_state_boundary\\":\\"captured_conv_state\\",";''',
  )
  cpp = replace_once(
      cpp,
      '''                  selected_gpu.program_build_ms + shared_gpu.program_build_ms +
                  tail_gpu.program_build_ms)
''',
      '''                  selected_gpu.program_build_ms + shared_gpu.program_build_ms +
                  tail_gpu.program_build_ms + layer3_run.program_build_ms)
''',
  )
  cpp = replace_once(
      cpp,
      '''                            selected_gpu.build_log + shared_gpu.build_log +
                            tail_gpu.build_log)
''',
      '''                            selected_gpu.build_log + shared_gpu.build_log +
                            tail_gpu.build_log + layer3_run.build_log)
''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << "\\"resident_layer7_ffn_q6_down_kernel_sum_min_us\\":"
              << layer2_ffn_sum_min << ",";
    std::cout << "\\"resident_two_linear_plus_layer7_attention_ffn_kernel_sum_min_us\\":"
              << (two_layer_kernel_sum_min + layer2_attention_sum_min + layer2_ffn_sum_min);
''',
      '''    std::cout << "\\"resident_layer7_ffn_q6_down_kernel_sum_min_us\\":"
              << layer2_ffn_sum_min << ",";
    std::cout << "\\"resident_two_linear_plus_layer7_attention_ffn_kernel_sum_min_us\\":"
              << (two_layer_kernel_sum_min + layer2_attention_sum_min + layer2_ffn_sum_min) << ",";
    std::cout << "\\"resident_layer8_state_input_kernel_sum_min_us\\":"
              << layer3_state_input_sum_min << ",";
    std::cout << "\\"resident_layer5_6_7_to_layer8_state_input_kernel_sum_min_us\\":"
              << (two_layer_kernel_sum_min + layer2_attention_sum_min +
                  layer2_ffn_sum_min + layer3_state_input_sum_min);
''',
  )
  cpp = replace_once(
      cpp,
      '''    WriteNamedCompareGroups(ffn_live_groups);
    std::cout << ",\\"l2_k_raw\\":{\\"gpu_vs_cpu\\":";
''',
      '''    WriteNamedCompareGroups(ffn_live_groups);
    WritePrefixedCompareGroups("l3", layer3_run.comparisons, &first_compare);
    std::cout << ",\\"l2_k_raw\\":{\\"gpu_vs_cpu\\":";
''',
  )
  cpp = replace_once(
      cpp,
      '''    WriteCompare(k_raw_gpu_vs_cpu);
    std::cout << "}";
    std::cout << "},\\"checks\\":{";
''',
      '''    WriteCompare(k_raw_gpu_vs_cpu);
    std::cout << "},\\"l3_conv_state_after\\":{\\"gpu_vs_cpu\\":";
    WriteCompare(layer3_run.conv_state_after_gpu_vs_cpu);
    std::cout << "},\\"l3_recurrent_state\\":{\\"gpu_vs_cpu\\":";
    WriteCompare(layer3_run.recurrent_state_gpu_vs_cpu);
    std::cout << "}";
    std::cout << "},\\"checks\\":{";
''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << ",\\"layer2_shapes\\":";
    WriteFullAttentionShapeChecks(full_shapes);
    std::cout << ",\\"layer1_residual_input_matches_oracle\\":"
''',
      '''    std::cout << ",\\"layer2_shapes\\":";
    WriteFullAttentionShapeChecks(full_shapes);
    std::cout << ",\\"layer3\\":";
    WriteLayerChecks(layer3_run);
    std::cout << ",\\"layer1_residual_input_matches_oracle\\":"
''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << "\\"layer2_ffn_tensor_shapes_ok\\":"
              << (layer2_ffn_tensor_shapes_ok ? "true" : "false") << ",";''',
      '''    std::cout << "\\"layer2_ffn_tensor_shapes_ok\\":"
              << (layer2_ffn_tensor_shapes_ok ? "true" : "false") << ",";
    std::cout << "\\"layer8_residual_input_from_layer7_live_gpu_lout\\":true,";
    std::cout << "\\"layer8_conv_state_input_boundary\\":true,";
    std::cout << "\\"layer8_gpu_cpu_matches_native\\":"
              << (layer3_gpu_cpu_ok ? "true" : "false") << ",";
    std::cout << "\\"layer8_state_input_oracle_policy_matches\\":"
              << (layer3_state_input_oracle_policy_ok ? "true" : "false") << ",";
    std::cout << "\\"layer8_state_input_handoff_matches\\":"
              << (layer3_ok ? "true" : "false") << ",";''',
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
  parser.add_argument("--out-dir", type=Path, default=None)
  return parser.parse_args()


def find_ffn_payload(pattern: str, expected_bytes: int) -> Path:
  matches = sorted(FFN_PAYLOAD_ROOT.glob(pattern))
  if len(matches) != 1:
    raise SystemExit(f"expected one layer7 FFN payload for {pattern}, found {len(matches)}")
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


def add_layer7_ffn_payloads(payloads: dict[str, dict[str, Any]]) -> None:
  for name, (stage_name, pattern, expected_bytes) in FFN_PAYLOAD_SPECS.items():
    payloads[name] = payload_record(
        find_ffn_payload(pattern, expected_bytes),
        stage_name,
        expected_bytes,
    )


def resident_fields_ok(probe: dict[str, Any] | None, expected_invocations: int) -> bool:
  return (
      isinstance(probe, dict)
      and probe.get("resident_api") == "two_linear_layer_to_full_attention_v_q6_live_ffn_lout_layer8_state_input_load_once_run_many"
      and probe.get("resident_load_count") == 1
      and probe.get("resident_shell_invocations") == expected_invocations
      and probe.get("full_attn_v_projection_boundary") == "gpu_q6_raw_matvec"
      and probe.get("full_attn_ffn_boundary") == "gpu_live_post_norm_to_q6_down"
      and probe.get("ffn_input_boundary") == "live_gpu_post_attention_norm"
      and probe.get("layer_output_residual_boundary") == "live_gpu_attention_residual"
      and probe.get("layer8_residual_input_boundary") == "live_gpu_l_out_7"
      and probe.get("layer8_conv_state_boundary") == "captured_conv_state"
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
      "# GPU Resident Layer-8 State/Input Handoff Probe",
      "",
      f"- workstream: `{WORKSTREAM}`",
      f"- host: `{payload.get('host')}`",
      f"- model: `{payload.get('model')}`",
      f"- layers: `{payload.get('layers')}`",
      f"- required checks passed: `{str(payload.get('required_checks_passed')).lower()}`",
      "- speedup claims allowed: `false`",
      f"- resident API: `{probe.get('resident_api')}`",
      f"- resident shell invocations: `{probe.get('resident_shell_invocations')}`",
      f"- V projection boundary: `{probe.get('full_attn_v_projection_boundary')}`",
      f"- FFN boundary: `{probe.get('full_attn_ffn_boundary')}`",
      f"- FFN input boundary: `{probe.get('ffn_input_boundary')}`",
      f"- layer output residual boundary: `{probe.get('layer_output_residual_boundary')}`",
      f"- layer 8 residual input boundary: `{probe.get('layer8_residual_input_boundary')}`",
      f"- layer 8 conv state boundary: `{probe.get('layer8_conv_state_boundary')}`",
      "",
      "| output | comparison | max abs | RMSE |",
      "|---|---|---:|---:|",
  ]
  for name in (
      "l3_residual_input",
      "l3_attn_norm",
      "l3_linear_attn_qkv_mixed",
      "l3_conv_output_raw",
  ):
    group = comparisons.get(name, {}) if isinstance(comparisons, dict) else {}
    lane = group.get("gpu_vs_oracle", {}) if isinstance(group, dict) else {}
    lines.append(f"| {name} | gpu_vs_oracle | {lane.get('max_abs_diff')} | {lane.get('rmse')} |")
  lines += [
      "",
      "| kernel group | min us |",
      "|---|---:|",
      f"| layer7_attention_total | {timings.get('resident_layer7_full_attention_kernel_sum_min_us') if isinstance(timings, dict) else None} |",
      f"| layer7_ffn_lout | {timings.get('resident_layer7_ffn_q6_down_kernel_sum_min_us') if isinstance(timings, dict) else None} |",
      f"| layer8_state_input | {timings.get('resident_layer8_state_input_kernel_sum_min_us') if isinstance(timings, dict) else None} |",
      f"| layer5_6_7_to_layer8_state_input | {timings.get('resident_layer5_6_7_to_layer8_state_input_kernel_sum_min_us') if isinstance(timings, dict) else None} |",
      "",
      "This target-side process carries layer 5 and layer 6 GPU outputs into",
      "layer 7, computes live GPU layer-7 l_out, then feeds that l_out into",
      "layer 8's linear-attention shell with captured layer-8 conv state. This",
      "is captured single-token state/input evidence, not decode throughput.",
      "",
  ]
  path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  args = parse_args()
  if args.resident_invocations <= 0:
    raise SystemExit("--resident-invocations must be positive")
  if args.layer != 5:
    raise SystemExit("--layer must be 5 for the layer-8 state/input handoff")

  layer0 = args.layer
  layer1 = args.layer + 1
  layer2 = args.layer + 2
  layer3 = args.layer + 3
  conv0_path = (
      args.conv_history_probe.resolve()
      if args.conv_history_probe is not None
      else TWO.latest_conv_history_probe_for_layer(layer0).resolve()
  )
  conv1_path = (
      args.next_conv_history_probe.resolve()
      if args.next_conv_history_probe is not None
      else TWO.latest_conv_history_probe_for_layer(layer1).resolve()
  )
  conv2_path = (
      args.layer8_conv_history_probe.resolve()
      if args.layer8_conv_history_probe is not None
      else TWO.latest_conv_history_probe_for_layer(layer3).resolve()
  )
  all_history_json = args.all_history_json.resolve()
  stamp = utc_stamp()
  created_at = iso_now()
  out_dir = args.out_dir or ROOT / f"output/gpu-resident-layer8-state-input-handoff-probe-{stamp}"
  out_dir.mkdir(parents=True, exist_ok=False)
  raw_dir = out_dir / "raw"
  raw_dir.mkdir()

  payloads0, conv0 = TWO.prefixed_payloads(layer0, conv0_path, "l0")
  payloads1, conv1 = TWO.prefixed_payloads(layer1, conv1_path, "l1")
  payloads2, conv2 = TWO.prefixed_payloads(layer3, conv2_path, "l3")
  full_payloads, all_history = L7_INPUT.resolve_full_attention_payloads(all_history_json, layer2)
  CORE.add_layer7_tail_payloads(full_payloads)
  add_layer7_ffn_payloads(full_payloads)
  payloads = {**payloads0, **payloads1, **full_payloads, **payloads2}
  opencl_source = OPENCL_SOURCE.read_text(encoding="utf-8")
  opencl_source_hash = iq36_local.sha256_file(OPENCL_SOURCE)
  embedded_opencl_hash = hashlib.sha256(
      (
          opencl_source
          + PRECONV.POSTCONV.ATTENTION.ATTENTION_EXTRA_OPENCL
          + CORE.FULL_ATTN_CORE_EXTRA_OPENCL
      ).encode("utf-8")
  ).hexdigest()
  local_cpp = out_dir / "gpu_resident_layer8_state_input_handoff_probe.cpp"
  local_cpp.write_text(layer8_state_input_probe_cpp(opencl_source), encoding="utf-8")

  remote_dir = f"{args.remote_root.rstrip('/')}/gpu-resident-layer8-state-input-handoff-probe-{stamp}"
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
            f"{remote_dir}/tests/gpu_resident_layer8_state_input_handoff_probe.cpp",
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

  executable = f"{remote_dir}/build/iq36-gpu-resident-layer8-state-input-handoff-probe"
  compile_cmd = " && ".join([
      f"source {shlex.quote(args.env_script)} >/dev/null 2>&1",
      (
          "g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic "
          f"-I {shlex.quote(remote_dir + '/include')} "
          f"{shlex.quote(remote_dir + '/src/gguf_loader.cpp')} "
          f"{shlex.quote(remote_dir + '/src/gpu_q4x8_matvec.cpp')} "
          f"{shlex.quote(remote_dir + '/tests/gpu_resident_layer8_state_input_handoff_probe.cpp')} "
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
      {"name": "layer2_ffn_tensor_shapes_ok", "pass": PRECONV.nested_bool(probe, "checks", "layer2_ffn_tensor_shapes_ok")},
      {"name": "layer2_core_output_matches_oracle", "pass": PRECONV.nested_bool(probe, "checks", "layer2_full_attn_core_output_matches_oracle")},
      {"name": "live_ffn_gpu_cpu_matches_native", "pass": PRECONV.nested_bool(probe, "checks", "layer2_live_ffn_gpu_cpu_matches_native")},
      {"name": "live_ffn_oracle_magnitude_policy_matches", "pass": PRECONV.nested_bool(probe, "checks", "layer2_live_ffn_oracle_magnitude_policy_matches")},
      {"name": "live_layer_output_full_policy_matches_oracle", "pass": PRECONV.nested_bool(probe, "checks", "layer2_live_layer_output_full_policy_matches_oracle")},
      {"name": "layer8_gpu_cpu_matches_native", "pass": PRECONV.nested_bool(probe, "checks", "layer8_gpu_cpu_matches_native")},
      {"name": "layer8_state_input_oracle_policy_matches", "pass": PRECONV.nested_bool(probe, "checks", "layer8_state_input_oracle_policy_matches")},
      {"name": "layer8_state_input_handoff_matches", "pass": PRECONV.nested_bool(probe, "checks", "layer8_state_input_handoff_matches")},
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
      "full_attn_component": CORE.FULL_ATTN_COMPARISON_THRESHOLDS,
      "live_ffn_oracle_magnitude": {
          "max_abs_diff": CORE.STRICT_COMPARISON_THRESHOLDS["max_abs_diff"],
          "rmse": CORE.STRICT_COMPARISON_THRESHOLDS["rmse"],
          "mismatch_abs_diff": CORE.STRICT_COMPARISON_THRESHOLDS["mismatch_abs_diff"],
      },
      "layer8_state_input_oracle_policy": CORE.FULL_ATTN_COMPARISON_THRESHOLDS,
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
      },
      "conv_history_capture_artifacts": {
          "layer0": conv0.get("capture_artifact"),
          "layer1": conv1.get("capture_artifact"),
          "layer3": conv2.get("capture_artifact"),
      },
      "all_history": all_history,
      "ffn_payload_root": str(FFN_PAYLOAD_ROOT.relative_to(ROOT)),
      "payloads": slim_payloads,
      "layers": [layer0, layer1, layer2, layer3],
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
      "tool": "tools/intel-qwen36-gpu-resident-layer8-state-input-handoff-probe.py",
      "artifact": str(out_dir),
      "host": args.host,
      "remote_dir": remote_dir,
      "layers": [layer0, layer1, layer2, layer3],
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
      "gpu_resident_layer8_state_input_handoff_probe",
      [
          ("required_checks_passed", required_checks_passed),
          ("resident_shell_invocations", args.resident_invocations),
          ("resident_layer8_state_input_kernel_sum_min_us", PRECONV.nested_number(timings, "resident_layer8_state_input_kernel_sum_min_us")),
          ("layer8_residual_input_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l3_residual_input", "gpu_vs_oracle", "max_abs_diff")),
          ("layer8_attn_norm_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l3_attn_norm", "gpu_vs_oracle", "max_abs_diff")),
          ("layer8_qkv_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l3_linear_attn_qkv_mixed", "gpu_vs_oracle", "max_abs_diff")),
          ("layer8_conv_output_raw_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l3_conv_output_raw", "gpu_vs_oracle", "max_abs_diff")),
      ],
  )
  write_summary(out_dir / "summary.md", payload)
  print(out_dir)
  return 0 if required_checks_passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
