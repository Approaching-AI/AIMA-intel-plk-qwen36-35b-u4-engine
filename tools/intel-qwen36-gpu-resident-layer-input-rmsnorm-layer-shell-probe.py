#!/usr/bin/env python3
"""Run the resident GPU layer-input RMSNorm-to-layer shell handoff probe."""

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
PRECONV_TOOL = Path(__file__).with_name("intel-qwen36-gpu-resident-preconv-layer-shell-probe.py")
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA_VERSION = "intel-qwen36-gpu-resident-layer-input-rmsnorm-layer-shell-probe-v0"
DEFAULT_HOST = "local"
DEFAULT_MODEL = "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf"
DEFAULT_ENV_SCRIPT = "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh"
DEFAULT_REMOTE_ROOT = "/home/intel/intel-qwen36-gpu"
DEFAULT_ORACLE_BUNDLE = ROOT / "oracle/r0-oracle-bundle-20260627T060028Z"
OPENCL_SOURCE = ROOT / "engine/gpu/opencl/q4x8_matvec.cl"


def load_preconv_tool() -> ModuleType:
  spec = importlib.util.spec_from_file_location("iq36_preconv_layer_shell_probe", PRECONV_TOOL)
  if spec is None or spec.loader is None:
    raise SystemExit(f"failed to load preconv layer shell tool: {PRECONV_TOOL}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


PRECONV = load_preconv_tool()


LAYER_INPUT_HELPERS_CPP = r'''

struct LayerInputRmsNormTiming {
  double rmsnorm_min_us = 0.0;
  double rmsnorm_mean_us = 0.0;
  std::uint64_t rmsnorm_global_work_items = 0;
};

struct LayerInputRmsNormRun {
  std::vector<float> attn_norm;
  LayerInputRmsNormTiming timing;
  std::string platform_name;
  std::string device_name;
  std::string build_log;
  double program_build_ms = 0.0;
};

LayerInputRmsNormRun RunGpuLayerInputRmsNorm(
    const std::vector<float>& residual_input,
    const std::vector<float>& attn_norm_weight,
    float rms_norm_epsilon,
    const std::string& device_substring,
    int repeat) {
  Require(residual_input.size() == kHiddenSize,
          "layer-input residual size mismatch");
  Require(attn_norm_weight.size() == kHiddenSize,
          "layer-input attention norm weight size mismatch");

  LayerInputRmsNormRun run;
  run.attn_norm.assign(kHiddenSize, 0.0f);

  OpenClApi api;
  const auto selected = SelectDevice(api, device_substring);
  run.platform_name = selected.platform_name;
  run.device_name = selected.device_name;
  cl_int err = kClSuccess;
  cl_context context = api.clCreateContext(nullptr, 1, &selected.device, nullptr, nullptr, &err);
  Check(err, "clCreateContext(layer input rmsnorm)");
  cl_command_queue queue = api.clCreateCommandQueue(context, selected.device, kClQueueProfilingEnable, &err);
  Check(err, "clCreateCommandQueue(layer input rmsnorm)");
  const char* source = kOpenClSource;
  const std::size_t source_len = std::strlen(kOpenClSource);
  cl_program program = api.clCreateProgramWithSource(context, 1, &source, &source_len, &err);
  Check(err, "clCreateProgramWithSource(layer input rmsnorm)");
  const auto build_begin = std::chrono::steady_clock::now();
  err = api.clBuildProgram(program, 1, &selected.device, "", nullptr, nullptr);
  const auto build_end = std::chrono::steady_clock::now();
  run.program_build_ms =
      std::chrono::duration<double, std::milli>(build_end - build_begin).count();
  run.build_log = BuildLog(api, program, selected.device);
  if (err != kClSuccess) {
    Die("clBuildProgram(layer input rmsnorm) failed with OpenCL error " +
        std::to_string(err) + "; build log: " + run.build_log);
  }
  Check(err, "clBuildProgram(layer input rmsnorm)");
  cl_kernel rmsnorm_kernel = api.clCreateKernel(program, "rms_norm_hidden_f32", &err);
  Check(err, "clCreateKernel(layer input rms_norm_hidden_f32)");

  cl_mem residual_input_buffer = nullptr;
  cl_mem attn_norm_weight_buffer = nullptr;
  cl_mem attn_norm_buffer = nullptr;
  try {
    residual_input_buffer = api.clCreateBuffer(context, kClMemReadOnly,
                                               residual_input.size() * sizeof(float),
                                               nullptr, &err);
    Check(err, "clCreateBuffer(layer input residual)");
    attn_norm_weight_buffer = api.clCreateBuffer(context, kClMemReadOnly,
                                                 attn_norm_weight.size() * sizeof(float),
                                                 nullptr, &err);
    Check(err, "clCreateBuffer(attn norm weight)");
    attn_norm_buffer = api.clCreateBuffer(context, kClMemWriteOnly,
                                          run.attn_norm.size() * sizeof(float),
                                          nullptr, &err);
    Check(err, "clCreateBuffer(layer input attn norm)");
    Check(api.clEnqueueWriteBuffer(queue, residual_input_buffer, kClTrue, 0,
                                   residual_input.size() * sizeof(float),
                                   residual_input.data(), 0, nullptr, nullptr),
          "clEnqueueWriteBuffer(layer input residual)");
    Check(api.clEnqueueWriteBuffer(queue, attn_norm_weight_buffer, kClTrue, 0,
                                   attn_norm_weight.size() * sizeof(float),
                                   attn_norm_weight.data(), 0, nullptr, nullptr),
          "clEnqueueWriteBuffer(attn norm weight)");

    const cl_uint hidden_arg = static_cast<cl_uint>(kHiddenSize);
    Check(api.clSetKernelArg(rmsnorm_kernel, 0, sizeof(residual_input_buffer), &residual_input_buffer), "clSetKernelArg(layer input rmsnorm 0)");
    Check(api.clSetKernelArg(rmsnorm_kernel, 1, sizeof(attn_norm_weight_buffer), &attn_norm_weight_buffer), "clSetKernelArg(layer input rmsnorm 1)");
    Check(api.clSetKernelArg(rmsnorm_kernel, 2, sizeof(hidden_arg), &hidden_arg), "clSetKernelArg(layer input rmsnorm 2)");
    Check(api.clSetKernelArg(rmsnorm_kernel, 3, sizeof(rms_norm_epsilon), &rms_norm_epsilon), "clSetKernelArg(layer input rmsnorm 3)");
    Check(api.clSetKernelArg(rmsnorm_kernel, 4, sizeof(attn_norm_buffer), &attn_norm_buffer), "clSetKernelArg(layer input rmsnorm 4)");

    const std::size_t rmsnorm_global = 1;
    std::vector<double> rmsnorm_times;
    rmsnorm_times.reserve(static_cast<std::size_t>(repeat));
    for (int i = 0; i < repeat; ++i) {
      cl_event event = nullptr;
      Check(api.clEnqueueNDRangeKernel(queue, rmsnorm_kernel, 1, nullptr,
                                       &rmsnorm_global, nullptr, 0, nullptr,
                                       &event),
            "clEnqueueNDRangeKernel(layer input rmsnorm)");
      Check(api.clFinish(queue), "clFinish(layer input rmsnorm)");
      rmsnorm_times.push_back(EventUs(api, event));
      api.clReleaseEvent(event);
    }
    Check(api.clEnqueueReadBuffer(queue, attn_norm_buffer, kClTrue, 0,
                                  run.attn_norm.size() * sizeof(float),
                                  run.attn_norm.data(), 0, nullptr, nullptr),
          "clEnqueueReadBuffer(layer input attn norm)");
    run.timing.rmsnorm_min_us = Min(rmsnorm_times);
    run.timing.rmsnorm_mean_us = Mean(rmsnorm_times);
    run.timing.rmsnorm_global_work_items = 1;
  } catch (...) {
    ReleaseMem(api, &attn_norm_buffer);
    ReleaseMem(api, &attn_norm_weight_buffer);
    ReleaseMem(api, &residual_input_buffer);
    api.clReleaseKernel(rmsnorm_kernel);
    api.clReleaseProgram(program);
    api.clReleaseCommandQueue(queue);
    api.clReleaseContext(context);
    throw;
  }
  ReleaseMem(api, &attn_norm_buffer);
  ReleaseMem(api, &attn_norm_weight_buffer);
  ReleaseMem(api, &residual_input_buffer);
  api.clReleaseKernel(rmsnorm_kernel);
  api.clReleaseProgram(program);
  api.clReleaseCommandQueue(queue);
  api.clReleaseContext(context);
  return run;
}
'''


def replace_once(text: str, old: str, new: str) -> str:
  count = text.count(old)
  if count != 1:
    raise RuntimeError(f"expected one replacement site, found {count}: {old[:80]!r}")
  return text.replace(old, new, 1)


def layer_input_main_cpp() -> str:
  cpp = PRECONV.PRECONV_MAIN_CPP
  cpp = replace_once(
      cpp,
      'const std::string qkv_tensor_name =\n        LayerTensorName(args.layer, "attn_qkv.weight");',
      'const std::string attn_norm_tensor_name =\n        LayerTensorName(args.layer, "attn_norm.weight");\n'
      '    const std::string qkv_tensor_name =\n        LayerTensorName(args.layer, "attn_qkv.weight");',
  )
  cpp = replace_once(
      cpp,
      'const auto* qkv_tensor = iq36::find_tensor(index, qkv_tensor_name);',
      'const auto* attn_norm_tensor = iq36::find_tensor(index, attn_norm_tensor_name);\n'
      '    const auto* qkv_tensor = iq36::find_tensor(index, qkv_tensor_name);',
  )
  cpp = replace_once(
      cpp,
      'Require(qkv_tensor != nullptr, "qkv tensor missing");',
      'Require(attn_norm_tensor != nullptr, "attention norm tensor missing");\n'
      '    Require(qkv_tensor != nullptr, "qkv tensor missing");',
  )
  cpp = replace_once(
      cpp,
      'const bool preconv_tensor_shape_ok =\n        qkv_tensor->type == 12 &&',
      'const bool layer_input_tensor_shape_ok =\n'
      '        attn_norm_tensor->type == 0 &&\n'
      '        attn_norm_tensor->dims == std::vector<std::uint64_t>{kHiddenSize};\n'
      '    const bool preconv_tensor_shape_ok =\n        qkv_tensor->type == 12 &&',
  )
  cpp = replace_once(
      cpp,
      'const auto attn_norm =\n        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "attn_norm.bin"));',
      'const auto oracle_attn_norm =\n        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "attn_norm.bin"));',
  )
  cpp = replace_once(
      cpp,
      'const bool payload_counts_ok =\n        attn_norm.size() == kHiddenSize &&',
      'const bool payload_counts_ok =\n        oracle_attn_norm.size() == kHiddenSize &&',
  )
  cpp = replace_once(
      cpp,
      'const auto ssm_dt =\n        iq36::decode_tensor_row(args.model_path, index,\n                                LayerTensorName(args.layer, "ssm_dt.bias"), 0);',
      'const auto attn_norm_weight =\n        ReadF32TensorPayload(model, *attn_norm_tensor,\n                             static_cast<std::size_t>(kHiddenSize));\n'
      '    const auto ssm_dt =\n        iq36::decode_tensor_row(args.model_path, index,\n                                LayerTensorName(args.layer, "ssm_dt.bias"), 0);',
  )
  cpp = replace_once(
      cpp,
      'const auto native_preconv = iq36::run_qwen36_linear_attention_preconv_core(\n        args.model_path, index, args.layer, attn_norm);',
      'const auto native_attn_norm =\n        iq36::apply_rms_norm(residual_input, attn_norm_weight, rms_norm_epsilon);\n'
      '    const auto layer_input_gpu = RunGpuLayerInputRmsNorm(\n'
      '        residual_input, attn_norm_weight, rms_norm_epsilon,\n'
      '        args.device_substring, args.repeat);\n'
      '    const auto native_preconv = iq36::run_qwen36_linear_attention_preconv_core(\n        args.model_path, index, args.layer, native_attn_norm);',
  )
  cpp = replace_once(
      cpp,
      'const auto preconv_gpu = RunGpuPreConvFront(\n        args.model_path, *qkv_tensor, *alpha_tensor, *beta_tensor, *z_tensor,\n        *conv_tensor, ssm_dt, ssm_a, attn_norm, conv_state,',
      'const auto preconv_gpu = RunGpuPreConvFront(\n        args.model_path, *qkv_tensor, *alpha_tensor, *beta_tensor, *z_tensor,\n        *conv_tensor, ssm_dt, ssm_a, layer_input_gpu.attn_norm, conv_state,',
  )
  cpp = replace_once(
      cpp,
      'const std::vector<NamedCompareGroup> preconv_groups = {',
      'const std::vector<NamedCompareGroup> layer_input_groups = {\n'
      '        {"attn_norm",\n'
      '         iq36::compare_vectors(native_attn_norm, oracle_attn_norm, kMismatchThreshold),\n'
      '         iq36::compare_vectors(layer_input_gpu.attn_norm, native_attn_norm, kMismatchThreshold),\n'
      '         iq36::compare_vectors(layer_input_gpu.attn_norm, oracle_attn_norm, kMismatchThreshold)},\n'
      '    };\n'
      '    const std::vector<NamedCompareGroup> preconv_groups = {',
  )
  cpp = replace_once(
      cpp,
      'const bool preconv_comparisons_passed =\n        CompareGroupsPassed(preconv_groups) && ComparePassed(conv_state_after_gpu_vs_cpu);',
      'const bool layer_input_comparisons_passed = CompareGroupsPassed(layer_input_groups);\n'
      '    const bool preconv_comparisons_passed =\n        CompareGroupsPassed(preconv_groups) && ComparePassed(conv_state_after_gpu_vs_cpu);',
  )
  cpp = replace_once(
      cpp,
      'const bool preconv_timing_positive =\n        preconv_gpu.timing.qkv_min_us > 0.0 &&',
      'const bool layer_input_timing_positive =\n'
      '        layer_input_gpu.timing.rmsnorm_min_us > 0.0;\n'
      '    const bool preconv_timing_positive =\n        preconv_gpu.timing.qkv_min_us > 0.0 &&',
  )
  cpp = replace_once(
      cpp,
      'const bool arc_selected =\n        preconv_gpu.device_name.find(args.device_substring) != std::string::npos &&',
      'const bool arc_selected =\n'
      '        layer_input_gpu.device_name.find(args.device_substring) != std::string::npos &&\n'
      '        preconv_gpu.device_name.find(args.device_substring) != std::string::npos &&',
  )
  cpp = replace_once(
      cpp,
      'const bool required_checks_passed =\n        load_map.ready &&\n        preconv_tensor_shape_ok &&',
      'const bool required_checks_passed =\n'
      '        load_map.ready &&\n'
      '        layer_input_tensor_shape_ok &&\n'
      '        preconv_tensor_shape_ok &&',
  )
  cpp = replace_once(
      cpp,
      'arc_selected &&\n        preconv_comparisons_passed &&',
      'arc_selected &&\n'
      '        layer_input_comparisons_passed &&\n'
      '        preconv_comparisons_passed &&',
  )
  cpp = replace_once(
      cpp,
      'preconv_timing_positive &&\n        delta_timing_positive &&',
      'layer_input_timing_positive &&\n'
      '        preconv_timing_positive &&\n'
      '        delta_timing_positive &&',
  )
  cpp = replace_once(
      cpp,
      'const double full_kernel_sum_min =\n        preconv_gpu.timing.preconv_to_postconv_kernel_sum_min_us +',
      'const double full_kernel_sum_min =\n'
      '        layer_input_gpu.timing.rmsnorm_min_us +\n'
      '        preconv_gpu.timing.preconv_to_postconv_kernel_sum_min_us +',
  )
  cpp = replace_once(
      cpp,
      'const double full_kernel_sum_mean =\n        preconv_gpu.timing.preconv_to_postconv_kernel_sum_mean_us +',
      'const double full_kernel_sum_mean =\n'
      '        layer_input_gpu.timing.rmsnorm_mean_us +\n'
      '        preconv_gpu.timing.preconv_to_postconv_kernel_sum_mean_us +',
  )
  cpp = replace_once(
      cpp,
      'std::cout << "\\"schema_version\\":\\"intel-qwen36-gpu-resident-preconv-layer-shell-probe-v0\\",";',
      'std::cout << "\\"schema_version\\":\\"intel-qwen36-gpu-resident-layer-input-rmsnorm-layer-shell-probe-v0\\",";',
  )
  cpp = replace_once(
      cpp,
      'std::cout << "\\"resident_api\\":\\"preconv_to_layer_shell_load_once_run_many\\",";',
      'std::cout << "\\"resident_api\\":\\"layer_input_rmsnorm_to_layer_shell_load_once_run_many\\",";',
  )
  cpp = replace_once(
      cpp,
      'std::cout << "\\"captured_conv_state_input_boundary\\":true,";',
      'std::cout << "\\"layer_input_rmsnorm_gpu_boundary\\":true,";\n'
      '    std::cout << "\\"captured_attn_norm_required_check\\":true,";\n'
      '    std::cout << "\\"captured_conv_state_input_boundary\\":true,";',
  )
  cpp = replace_once(
      cpp,
      'std::cout << "\\"platform_name\\":\\"" << JsonEscape(preconv_gpu.platform_name) << "\\",";',
      'std::cout << "\\"platform_name\\":\\"" << JsonEscape(layer_input_gpu.platform_name) << "\\",";',
  )
  cpp = replace_once(
      cpp,
      'std::cout << "\\"device_name\\":\\"" << JsonEscape(preconv_gpu.device_name) << "\\",";',
      'std::cout << "\\"device_name\\":\\"" << JsonEscape(layer_input_gpu.device_name) << "\\",";\n'
      '    std::cout << "\\"preconv_device_name\\":\\"" << JsonEscape(preconv_gpu.device_name) << "\\",";',
  )
  cpp = replace_once(
      cpp,
      '(preconv_gpu.program_build_ms + delta_gpu.program_build_ms +',
      '(layer_input_gpu.program_build_ms + preconv_gpu.program_build_ms + delta_gpu.program_build_ms +',
  )
  cpp = replace_once(
      cpp,
      'JsonEscape(preconv_gpu.build_log + delta_gpu.build_log +',
      'JsonEscape(layer_input_gpu.build_log + preconv_gpu.build_log + delta_gpu.build_log +',
  )
  cpp = replace_once(
      cpp,
      'std::cout << "\\"timings\\":{";',
      'std::cout << "\\"timings\\":{";\n'
      '    std::cout << "\\"layer_input_rmsnorm_min_us\\":"\n'
      '              << layer_input_gpu.timing.rmsnorm_min_us << ",";\n'
      '    std::cout << "\\"layer_input_rmsnorm_mean_us\\":"\n'
      '              << layer_input_gpu.timing.rmsnorm_mean_us << ",";',
  )
  cpp = replace_once(
      cpp,
      'std::cout << "\\"resident_preconv_to_layer_kernel_sum_min_us\\":"\n              << full_kernel_sum_min << ",";',
      'std::cout << "\\"resident_layer_input_rmsnorm_to_layer_kernel_sum_min_us\\":"\n              << full_kernel_sum_min << ",";',
  )
  cpp = replace_once(
      cpp,
      'std::cout << "\\"resident_preconv_to_layer_kernel_sum_mean_us\\":"\n              << full_kernel_sum_mean;',
      'std::cout << "\\"resident_layer_input_rmsnorm_to_layer_kernel_sum_mean_us\\":"\n              << full_kernel_sum_mean;',
  )
  cpp = replace_once(
      cpp,
      'WriteNamedCompareGroups(preconv_groups);\n    std::cout << ",\\"conv_state_after\\":{\\"gpu_vs_cpu\\":";',
      'WriteNamedCompareGroups(layer_input_groups);\n'
      '    std::cout << ",";\n'
      '    WriteNamedCompareGroups(preconv_groups);\n'
      '    std::cout << ",\\"conv_state_after\\":{\\"gpu_vs_cpu\\":";',
  )
  cpp = replace_once(
      cpp,
      'std::cout << "\\"preconv_tensor_shape_ok\\":"\n              << (preconv_tensor_shape_ok ? "true" : "false") << ",";',
      'std::cout << "\\"layer_input_tensor_shape_ok\\":"\n'
      '              << (layer_input_tensor_shape_ok ? "true" : "false") << ",";\n'
      '    std::cout << "\\"preconv_tensor_shape_ok\\":"\n              << (preconv_tensor_shape_ok ? "true" : "false") << ",";',
  )
  cpp = replace_once(
      cpp,
      'std::cout << "\\"preconv_to_postconv_matches_oracle\\":"\n              << (preconv_comparisons_passed ? "true" : "false") << ",";',
      'std::cout << "\\"layer_input_rmsnorm_matches_oracle\\":"\n'
      '              << (layer_input_comparisons_passed ? "true" : "false") << ",";\n'
      '    std::cout << "\\"preconv_to_postconv_matches_oracle\\":"\n              << (preconv_comparisons_passed ? "true" : "false") << ",";',
  )
  cpp = replace_once(
      cpp,
      'std::cout << "\\"gpu_event_timing_positive\\":"\n              << (preconv_timing_positive && delta_timing_positive &&',
      'std::cout << "\\"gpu_event_timing_positive\\":"\n'
      '              << (layer_input_timing_positive && preconv_timing_positive && delta_timing_positive &&',
  )
  return cpp


def layer_input_probe_cpp(opencl_source: str) -> str:
  cpp = PRECONV.preconv_layer_probe_cpp(opencl_source)
  main_index = cpp.index("\nint main(")
  return cpp[:main_index] + "\n" + LAYER_INPUT_HELPERS_CPP + "\n" + layer_input_main_cpp()


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


def resident_fields_ok(probe: dict[str, Any] | None, expected_invocations: int) -> bool:
  return (
      isinstance(probe, dict)
      and probe.get("resident_api") == "layer_input_rmsnorm_to_layer_shell_load_once_run_many"
      and probe.get("resident_load_count") == 1
      and probe.get("resident_shell_invocations") == expected_invocations
      and probe.get("layer_input_rmsnorm_gpu_boundary") is True
      and probe.get("captured_attn_norm_required_check") is True
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
      "# GPU Resident Layer-Input RMSNorm-to-Layer Shell Probe",
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
      "attn_norm",
      "linear_attn_qkv_mixed",
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
      f"| layer_input_rmsnorm | {timings.get('layer_input_rmsnorm_min_us')} |",
      f"| preconv_to_postconv_sum | {timings.get('preconv_to_postconv_kernel_sum_min_us')} |",
      f"| delta_to_final_sum | {timings.get('delta_to_final_kernel_sum_min_us')} |",
      f"| attention_front_sum | {timings.get('attention_front_kernel_sum_min_us')} |",
      f"| selected_ffn_sum | {timings.get('selected_ffn_kernel_sum_min_us')} |",
      f"| shared_ffn_sum | {timings.get('shared_ffn_kernel_sum_min_us')} |",
      f"| ffn_tail_sum | {timings.get('ffn_tail_kernel_sum_min_us')} |",
      f"| resident_layer_input_rmsnorm_to_layer_sum | {timings.get('resident_layer_input_rmsnorm_to_layer_kernel_sum_min_us')} |",
      "",
      "The target-side process starts from captured layer residual input and",
      "captured `conv_states` for layer 5. GPU computes attention RMSNorm, then",
      "runs the closed preconv-to-layer shell through `l_out`. Captured",
      "`attn_norm` remains a required oracle check. This is captured",
      "single-layer evidence only, not prompt/token decode throughput.",
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
      else PRECONV.latest_conv_history_probe().resolve()
  )
  stamp = utc_stamp()
  created_at = iso_now()
  out_dir = args.out_dir or ROOT / f"output/gpu-resident-layer-input-rmsnorm-layer-shell-probe-{stamp}"
  out_dir.mkdir(parents=True, exist_ok=False)
  raw_dir = out_dir / "raw"
  raw_dir.mkdir()

  payloads, conv_history_probe = PRECONV.resolve_payloads(args.layer, conv_history_probe_path)
  opencl_source = OPENCL_SOURCE.read_text(encoding="utf-8")
  opencl_source_hash = iq36_local.sha256_file(OPENCL_SOURCE)
  embedded_opencl_hash = hashlib.sha256(
      (opencl_source + PRECONV.POSTCONV.ATTENTION.ATTENTION_EXTRA_OPENCL).encode("utf-8")
  ).hexdigest()
  local_cpp = out_dir / "gpu_resident_layer_input_rmsnorm_layer_shell_probe.cpp"
  local_cpp.write_text(layer_input_probe_cpp(opencl_source), encoding="utf-8")

  remote_dir = f"{args.remote_root.rstrip('/')}/gpu-resident-layer-input-rmsnorm-layer-shell-probe-{stamp}"
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
            f"{remote_dir}/tests/gpu_resident_layer_input_rmsnorm_layer_shell_probe.cpp",
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

  executable = f"{remote_dir}/build/iq36-gpu-resident-layer-input-rmsnorm-layer-shell-probe"
  compile_cmd = " && ".join([
      f"source {shlex.quote(args.env_script)} >/dev/null 2>&1",
      (
          "g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic "
          f"-I {shlex.quote(remote_dir + '/include')} "
          f"{shlex.quote(remote_dir + '/src/gguf_loader.cpp')} "
          f"{shlex.quote(remote_dir + '/src/gpu_q4x8_matvec.cpp')} "
          f"{shlex.quote(remote_dir + '/tests/gpu_resident_layer_input_rmsnorm_layer_shell_probe.cpp')} "
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
      {"name": "layer_input_rmsnorm_matches_oracle", "pass": PRECONV.nested_bool(probe, "checks", "layer_input_rmsnorm_matches_oracle")},
      {"name": "preconv_to_postconv_matches_oracle", "pass": PRECONV.nested_bool(probe, "checks", "preconv_to_postconv_matches_oracle")},
      {"name": "downstream_layer_matches_oracle", "pass": PRECONV.nested_bool(probe, "checks", "downstream_layer_matches_oracle")},
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
      "tool": "tools/intel-qwen36-gpu-resident-layer-input-rmsnorm-layer-shell-probe.py",
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
      "gpu_resident_layer_input_rmsnorm_layer_shell_probe",
      [
          ("required_checks_passed", required_checks_passed),
          ("resident_shell_invocations", args.resident_invocations),
          ("layer_input_rmsnorm_min_us", PRECONV.nested_number(timings, "layer_input_rmsnorm_min_us")),
          ("preconv_to_postconv_kernel_sum_min_us", PRECONV.nested_number(timings, "preconv_to_postconv_kernel_sum_min_us")),
          ("resident_layer_input_rmsnorm_to_layer_kernel_sum_min_us", PRECONV.nested_number(timings, "resident_layer_input_rmsnorm_to_layer_kernel_sum_min_us")),
          ("attn_norm_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "attn_norm", "gpu_vs_oracle", "max_abs_diff")),
          ("qkv_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "linear_attn_qkv_mixed", "gpu_vs_oracle", "max_abs_diff")),
          ("conv_output_raw_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "conv_output_raw", "gpu_vs_oracle", "max_abs_diff")),
          ("final_output_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "final_output", "gpu_vs_oracle", "max_abs_diff")),
          ("layer_output_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "layer_output", "gpu_vs_oracle", "max_abs_diff")),
      ],
  )
  write_summary(out_dir / "summary.md", payload)
  print(out_dir)
  return 0 if required_checks_passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
