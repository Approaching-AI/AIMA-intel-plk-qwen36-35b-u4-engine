#!/usr/bin/env python3
"""Run the resident GPU layer-7 full-attention V Q6 handoff probe."""

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
CORE_TOOL = Path(__file__).with_name(
    "intel-qwen36-gpu-resident-layer7-full-attn-core-output-handoff-probe.py"
)
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA_VERSION = "intel-qwen36-gpu-resident-layer7-full-attn-v-q6-handoff-probe-v0"
DEFAULT_HOST = "local"
DEFAULT_MODEL = "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf"
DEFAULT_ENV_SCRIPT = "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh"
DEFAULT_REMOTE_ROOT = "/home/intel/intel-qwen36-gpu"
DEFAULT_ORACLE_BUNDLE = ROOT / "oracle/r0-oracle-bundle-20260627T060028Z"
DEFAULT_ALL_HISTORY = ROOT / "output/r1-full-attn-all-history-capture-20260627T145615Z/history.json"
OPENCL_SOURCE = ROOT / "engine/gpu/opencl/q4x8_matvec.cl"


def load_core_tool() -> ModuleType:
  spec = importlib.util.spec_from_file_location("iq36_layer7_core_output_probe", CORE_TOOL)
  if spec is None or spec.loader is None:
    raise SystemExit(f"failed to load layer7 core/output tool: {CORE_TOOL}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


CORE = load_core_tool()
L7_INPUT = CORE.L7_INPUT
TWO = CORE.TWO
PRECONV = CORE.PRECONV


V_Q6_HELPERS_CPP = r'''

constexpr int kFullAttentionQ6KBlockBytes = 210;
constexpr int kFullAttentionQ8BlockValues = 256;

struct FullAttentionVQ6Timing {
  double host_q8_bridge_us = 0.0;
  double v_projection_min_us = 0.0;
  double v_projection_mean_us = 0.0;
  double effective_raw_gb_s = 0.0;
  double effective_io_gb_s = 0.0;
  std::uint64_t global_work_items = 0;
  std::uint64_t kernel_launches = 0;
};

struct FullAttentionVQ6Run {
  std::vector<float> v;
  FullAttentionVQ6Timing timing;
  std::string platform_name;
  std::string device_name;
  std::string build_log;
  double program_build_ms = 0.0;
};

std::vector<std::uint8_t> ReadTensorRawPayload(
    const std::string& model_path,
    const iq36::GgufTensorInfo& tensor) {
  Require(tensor.nbytes <= static_cast<std::uint64_t>(
              std::numeric_limits<std::size_t>::max()),
          "tensor too large for probe raw read");
  std::ifstream model(model_path, std::ios::binary);
  Require(static_cast<bool>(model), "failed to open model for raw tensor read");
  std::vector<std::uint8_t> raw(static_cast<std::size_t>(tensor.nbytes));
  model.seekg(static_cast<std::streamoff>(tensor.absolute_offset), std::ios::beg);
  model.read(reinterpret_cast<char*>(raw.data()),
             static_cast<std::streamsize>(raw.size()));
  Require(model.gcount() == static_cast<std::streamsize>(raw.size()),
          "short raw tensor read");
  return raw;
}

FullAttentionVQ6Run RunGpuFullAttentionVQ6(
    const std::string& model_path,
    const iq36::GgufTensorInfo& v_tensor,
    const std::vector<float>& attn_norm,
    const std::string& device_substring,
    int repeat) {
  Require(v_tensor.type == 14, "full attention V tensor must be Q6_K");
  Require(v_tensor.dims == std::vector<std::uint64_t>{kHiddenSize, kFullKvValues},
          "full attention V tensor shape mismatch");
  Require(attn_norm.size() == kHiddenSize,
          "full attention V attn_norm size mismatch");
  const std::uint64_t cols = v_tensor.dims[0];
  const std::uint64_t rows = v_tensor.dims[1];
  const std::uint64_t blocks_per_row = cols / kFullAttentionQ8BlockValues;
  const std::uint64_t row_nbytes =
      iq36::ggml_tensor_nbytes(v_tensor.type, std::vector<std::uint64_t>{cols});
  Require(row_nbytes == blocks_per_row * kFullAttentionQ6KBlockBytes,
          "full attention V Q6 row byte mismatch");
  Require(v_tensor.nbytes == rows * row_nbytes,
          "full attention V Q6 tensor byte mismatch");

  FullAttentionVQ6Run run;
  run.v.assign(static_cast<std::size_t>(rows), 0.0f);
  const auto bridge_begin = std::chrono::steady_clock::now();
  const auto q8 = iq36::QuantizeQ8KInputPlanes(attn_norm);
  const auto bridge_end = std::chrono::steady_clock::now();
  run.timing.host_q8_bridge_us =
      std::chrono::duration<double, std::micro>(bridge_end - bridge_begin).count();
  const auto raw = ReadTensorRawPayload(model_path, v_tensor);

  OpenClApi api;
  const auto selected = SelectDevice(api, device_substring);
  run.platform_name = selected.platform_name;
  run.device_name = selected.device_name;
  cl_int err = kClSuccess;
  cl_context context =
      api.clCreateContext(nullptr, 1, &selected.device, nullptr, nullptr, &err);
  Check(err, "clCreateContext(full attention v q6)");
  cl_command_queue queue =
      api.clCreateCommandQueue(context, selected.device, kClQueueProfilingEnable, &err);
  Check(err, "clCreateCommandQueue(full attention v q6)");
  const char* source = kOpenClSource;
  const std::size_t source_len = std::strlen(kOpenClSource);
  cl_program program =
      api.clCreateProgramWithSource(context, 1, &source, &source_len, &err);
  Check(err, "clCreateProgramWithSource(full attention v q6)");
  const auto build_begin = std::chrono::steady_clock::now();
  err = api.clBuildProgram(program, 1, &selected.device, "", nullptr, nullptr);
  const auto build_end = std::chrono::steady_clock::now();
  run.program_build_ms =
      std::chrono::duration<double, std::milli>(build_end - build_begin).count();
  run.build_log = BuildLog(api, program, selected.device);
  Check(err, "clBuildProgram(full attention v q6)");
  cl_kernel kernel =
      api.clCreateKernel(program, "q6k_selected_down_matvec_row", &err);
  Check(err, "clCreateKernel(q6k_selected_down_matvec_row)");

  cl_mem raw_buffer = nullptr;
  cl_mem q8_qs_buffer = nullptr;
  cl_mem q8_d_buffer = nullptr;
  cl_mem output_buffer = nullptr;
  try {
    raw_buffer = api.clCreateBuffer(context, kClMemReadOnly, raw.size(), nullptr, &err);
    Check(err, "clCreateBuffer(full attention v raw)");
    q8_qs_buffer =
        api.clCreateBuffer(context, kClMemReadOnly,
                           q8.qs.size() * sizeof(std::int8_t), nullptr, &err);
    Check(err, "clCreateBuffer(full attention v q8 qs)");
    q8_d_buffer =
        api.clCreateBuffer(context, kClMemReadOnly,
                           q8.d.size() * sizeof(float), nullptr, &err);
    Check(err, "clCreateBuffer(full attention v q8 d)");
    output_buffer =
        api.clCreateBuffer(context, kClMemWriteOnly,
                           run.v.size() * sizeof(float), nullptr, &err);
    Check(err, "clCreateBuffer(full attention v output)");
    Check(api.clEnqueueWriteBuffer(queue, raw_buffer, kClTrue, 0, raw.size(),
                                   raw.data(), 0, nullptr, nullptr),
          "clEnqueueWriteBuffer(full attention v raw)");
    Check(api.clEnqueueWriteBuffer(queue, q8_qs_buffer, kClTrue, 0,
                                   q8.qs.size() * sizeof(std::int8_t),
                                   q8.qs.data(), 0, nullptr, nullptr),
          "clEnqueueWriteBuffer(full attention v q8 qs)");
    Check(api.clEnqueueWriteBuffer(queue, q8_d_buffer, kClTrue, 0,
                                   q8.d.size() * sizeof(float),
                                   q8.d.data(), 0, nullptr, nullptr),
          "clEnqueueWriteBuffer(full attention v q8 d)");
    const cl_uint rows_per_expert_arg = static_cast<cl_uint>(rows);
    const cl_uint blocks_per_row_arg = static_cast<cl_uint>(blocks_per_row);
    Check(api.clSetKernelArg(kernel, 0, sizeof(raw_buffer), &raw_buffer),
          "clSetKernelArg(full attention v q6 0)");
    Check(api.clSetKernelArg(kernel, 1, sizeof(q8_qs_buffer), &q8_qs_buffer),
          "clSetKernelArg(full attention v q6 1)");
    Check(api.clSetKernelArg(kernel, 2, sizeof(q8_d_buffer), &q8_d_buffer),
          "clSetKernelArg(full attention v q6 2)");
    Check(api.clSetKernelArg(kernel, 3, sizeof(rows_per_expert_arg), &rows_per_expert_arg),
          "clSetKernelArg(full attention v q6 3)");
    Check(api.clSetKernelArg(kernel, 4, sizeof(blocks_per_row_arg), &blocks_per_row_arg),
          "clSetKernelArg(full attention v q6 4)");
    Check(api.clSetKernelArg(kernel, 5, sizeof(output_buffer), &output_buffer),
          "clSetKernelArg(full attention v q6 5)");
    const std::size_t global = run.v.size();
    std::vector<double> times;
    times.reserve(static_cast<std::size_t>(repeat));
    for (int i = 0; i < repeat; ++i) {
      cl_event event = nullptr;
      Check(api.clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global,
                                       nullptr, 0, nullptr, &event),
            "clEnqueueNDRangeKernel(full attention v q6)");
      Check(api.clFinish(queue), "clFinish(full attention v q6)");
      times.push_back(EventUs(api, event));
      api.clReleaseEvent(event);
    }
    Check(api.clEnqueueReadBuffer(queue, output_buffer, kClTrue, 0,
                                  run.v.size() * sizeof(float), run.v.data(),
                                  0, nullptr, nullptr),
          "clEnqueueReadBuffer(full attention v q6 output)");
    run.timing.v_projection_min_us = *std::min_element(times.begin(), times.end());
    run.timing.v_projection_mean_us =
        std::accumulate(times.begin(), times.end(), 0.0) /
        static_cast<double>(times.size());
    const double raw_bytes = static_cast<double>(raw.size());
    const double io_bytes = raw_bytes +
        static_cast<double>(q8.qs.size() * sizeof(std::int8_t)) +
        static_cast<double>(q8.d.size() * sizeof(float)) +
        static_cast<double>(run.v.size() * sizeof(float));
    run.timing.effective_raw_gb_s =
        raw_bytes / (run.timing.v_projection_min_us / 1e6) / 1e9;
    run.timing.effective_io_gb_s =
        io_bytes / (run.timing.v_projection_min_us / 1e6) / 1e9;
    run.timing.global_work_items = run.v.size();
    run.timing.kernel_launches = 1;
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
  return run;
}

void WriteFullAttentionVQ6Timing(const FullAttentionVQ6Timing& timing) {
  std::cout << "{";
  std::cout << "\"host_q8_bridge_us\":" << timing.host_q8_bridge_us << ",";
  std::cout << "\"v_projection_min_us\":" << timing.v_projection_min_us << ",";
  std::cout << "\"v_projection_mean_us\":" << timing.v_projection_mean_us << ",";
  std::cout << "\"effective_raw_gb_s\":" << timing.effective_raw_gb_s << ",";
  std::cout << "\"effective_io_gb_s\":" << timing.effective_io_gb_s << ",";
  std::cout << "\"global_work_items\":" << timing.global_work_items << ",";
  std::cout << "\"kernel_launches\":" << timing.kernel_launches;
  std::cout << "}";
}
'''


def replace_once(text: str, old: str, new: str) -> str:
  count = text.count(old)
  if count != 1:
    raise SystemExit(f"expected exactly one source replacement for {old[:80]!r}, found {count}")
  return text.replace(old, new, 1)


def v_q6_probe_cpp(opencl_source: str) -> str:
  cpp = CORE.core_output_probe_cpp(opencl_source)
  main_index = cpp.index("\nint main(")
  cpp = cpp[:main_index] + "\n" + V_Q6_HELPERS_CPP + cpp[main_index:]
  cpp = replace_once(
      cpp,
      "layer7 core/output probe expects --layer 5",
      "layer7 V Q6 handoff probe expects --layer 5",
  )
  cpp = replace_once(
      cpp,
      "intel-qwen36-gpu-resident-layer7-full-attn-core-output-handoff-probe-v0",
      SCHEMA_VERSION,
  )
  cpp = replace_once(
      cpp,
      "two_linear_layer_to_full_attention_core_output_load_once_run_many",
      "two_linear_layer_to_full_attention_v_q6_core_output_load_once_run_many",
  )
  cpp = replace_once(
      cpp,
      """    const auto layer2_qk_gpu = RunGpuFullAttentionQkFront(
        args.model_path,
        *layer2_tensors.q_tensor,
        *layer2_tensors.k_tensor,
        layer2_rms_gpu.attn_norm,
        args.device_substring,
        args.repeat);
""",
      """    const auto layer2_qk_gpu = RunGpuFullAttentionQkFront(
        args.model_path,
        *layer2_tensors.q_tensor,
        *layer2_tensors.k_tensor,
        layer2_rms_gpu.attn_norm,
        args.device_substring,
        args.repeat);
    const auto layer2_v_gpu = RunGpuFullAttentionVQ6(
        args.model_path,
        *layer2_tensors.v_tensor,
        layer2_rms_gpu.attn_norm,
        args.device_substring,
        args.repeat);
""",
  )
  cpp = replace_once(cpp, "    gpu_v_history.push_back(layer2_oracle.v);", "    gpu_v_history.push_back(layer2_v_gpu.v);")
  cpp = replace_once(
      cpp,
      """    AppendCpuGpuOracleCompare(strict_groups, "l2_k_rope",
                              native_rope.k_rope,
                              gpu_rope.k_rope,
                              layer2_oracle.k_rope);
""",
      """    AppendCpuGpuOracleCompare(strict_groups, "l2_k_rope",
                              native_rope.k_rope,
                              gpu_rope.k_rope,
                              layer2_oracle.k_rope);
    AppendCpuGpuOracleCompare(strict_groups, "l2_v",
                              native_qkv.v,
                              layer2_v_gpu.v,
                              layer2_oracle.v);
""",
  )
  cpp = replace_once(
      cpp,
      """    const auto native_v_vs_oracle = iq36::compare_vectors(
        native_qkv.v, layer2_oracle.v, kMismatchThreshold);
""",
      "",
  )
  cpp = replace_once(
      cpp,
      """        CompareGroupsPassed(strict_groups) &&
        ComparePassed(k_raw_gpu_vs_cpu) &&
        ComparePassed(native_v_vs_oracle);
""",
      """        CompareGroupsPassed(strict_groups) &&
        ComparePassed(k_raw_gpu_vs_cpu);
""",
  )
  cpp = replace_once(
      cpp,
      """        layer2_qk_gpu.timing.k_projection_min_us > 0.0 &&
        core_gate_gpu.timing.core_min_us > 0.0 &&
""",
      """        layer2_qk_gpu.timing.k_projection_min_us > 0.0 &&
        layer2_v_gpu.timing.v_projection_min_us > 0.0 &&
        core_gate_gpu.timing.core_min_us > 0.0 &&
""",
  )
  cpp = replace_once(
      cpp,
      """        layer2_qk_gpu.device_name.find(args.device_substring) != std::string::npos &&
        core_gate_gpu.device_name.find(args.device_substring) != std::string::npos &&
""",
      """        layer2_qk_gpu.device_name.find(args.device_substring) != std::string::npos &&
        layer2_v_gpu.device_name.find(args.device_substring) != std::string::npos &&
        core_gate_gpu.device_name.find(args.device_substring) != std::string::npos &&
""",
  )
  cpp = replace_once(
      cpp,
      """    const bool v_q6_reference_boundary = layer2_tensors.v_tensor->type == 14;
""",
      """    const bool v_q6_gpu_boundary =
        layer2_tensors.v_tensor->type == 14 &&
        layer2_v_gpu.v.size() == kFullKvValues;
""",
  )
  cpp = replace_once(cpp, "        v_q6_reference_boundary &&", "        v_q6_gpu_boundary &&")
  cpp = replace_once(
      cpp,
      """        layer2_rms_gpu.timing.rmsnorm_min_us +
        layer2_qk_gpu.timing.qk_projection_kernel_sum_min_us;
""",
      """        layer2_rms_gpu.timing.rmsnorm_min_us +
        layer2_qk_gpu.timing.qk_projection_kernel_sum_min_us +
        layer2_v_gpu.timing.v_projection_min_us;
""",
  )
  cpp = replace_once(
      cpp,
      '    std::cout << "\\"full_attn_v_projection_gpu_supported\\":false,";\n',
      '    std::cout << "\\"full_attn_v_projection_gpu_supported\\":true,";\n',
  )
  cpp = replace_once(
      cpp,
      '    std::cout << "\\"full_attn_v_projection_boundary\\":\\"cpu_q6_reference\\",";\n',
      '    std::cout << "\\"full_attn_v_projection_boundary\\":\\"gpu_q6_raw_matvec\\",";\n',
  )
  cpp = replace_once(
      cpp,
      r'''    std::cout << "\"output_projection_device_name\":\"" << JsonEscape(attention_gpu.device_name) << "\",";
''',
      r'''    std::cout << "\"output_projection_device_name\":\"" << JsonEscape(attention_gpu.device_name) << "\",";
    std::cout << "\"v_projection_device_name\":\"" << JsonEscape(layer2_v_gpu.device_name) << "\",";
''',
  )
  cpp = replace_once(
      cpp,
      """                  layer2_rms_gpu.program_build_ms + layer2_qk_gpu.program_build_ms +
                  core_gate_gpu.program_build_ms + attention_gpu.program_build_ms)
""",
      """                  layer2_rms_gpu.program_build_ms + layer2_qk_gpu.program_build_ms +
                  layer2_v_gpu.program_build_ms +
                  core_gate_gpu.program_build_ms + attention_gpu.program_build_ms)
""",
  )
  cpp = replace_once(
      cpp,
      """                            layer2_rms_gpu.build_log + layer2_qk_gpu.build_log +
                            core_gate_gpu.build_log + attention_gpu.build_log)
""",
      """                            layer2_rms_gpu.build_log + layer2_qk_gpu.build_log +
                            layer2_v_gpu.build_log +
                            core_gate_gpu.build_log + attention_gpu.build_log)
""",
  )
  cpp = replace_once(
      cpp,
      r'''    std::cout << ",\"layer2_core_gate\":";
''',
      r'''    std::cout << ",\"layer2_v_q6\":";
    WriteFullAttentionVQ6Timing(layer2_v_gpu.timing);
    std::cout << ",\"layer2_core_gate\":";
''',
  )
  cpp = replace_once(
      cpp,
      r'''    std::cout << ",\"l2_k_raw\":{\"gpu_vs_cpu\":";
    WriteCompare(k_raw_gpu_vs_cpu);
    std::cout << "},\"l2_v\":{\"cpu_vs_oracle\":";
    WriteCompare(native_v_vs_oracle);
    std::cout << "}";
''',
      r'''    std::cout << ",\"l2_k_raw\":{\"gpu_vs_cpu\":";
    WriteCompare(k_raw_gpu_vs_cpu);
    std::cout << "}";
''',
  )
  cpp = replace_once(
      cpp,
      r'''    std::cout << "\"layer2_v_q6_reference_boundary\":"
              << (v_q6_reference_boundary ? "true" : "false") << ",";
''',
      r'''    std::cout << "\"layer2_v_q6_gpu_boundary\":"
              << (v_q6_gpu_boundary ? "true" : "false") << ",";
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
  parser.add_argument("--out-dir", type=Path, default=None)
  return parser.parse_args()


def resident_fields_ok(probe: dict[str, Any] | None, expected_invocations: int) -> bool:
  return (
      isinstance(probe, dict)
      and probe.get("resident_api") == "two_linear_layer_to_full_attention_v_q6_core_output_load_once_run_many"
      and probe.get("resident_load_count") == 1
      and probe.get("resident_shell_invocations") == expected_invocations
      and probe.get("layer7_residual_input_from_layer6_gpu_output") is True
      and probe.get("full_attn_v_projection_gpu_supported") is True
      and probe.get("full_attn_v_projection_boundary") == "gpu_q6_raw_matvec"
      and probe.get("full_attn_core_gpu_boundary") is True
      and probe.get("full_attn_gate_gpu_boundary") is True
      and probe.get("full_attn_output_projection_gpu_boundary") is True
      and probe.get("full_attn_post_norm_gpu_boundary") is True
      and probe.get("full_attn_ffn_boundary") == "q6_down_reference_pending"
      and PRECONV.nested_bool(probe, "checks", "resident_load_once")
      and PRECONV.nested_bool(probe, "checks", "resident_shell_invocations_positive")
  )


def write_summary(path: Path, payload: dict[str, Any]) -> None:
  probe = payload.get("probe") if isinstance(payload.get("probe"), dict) else {}
  timings = probe.get("timings", {}) if isinstance(probe, dict) else {}
  comparisons = probe.get("comparisons", {}) if isinstance(probe, dict) else {}
  lines = [
      "# GPU Resident Layer-7 Full-Attention V Q6 Handoff Probe",
      "",
      f"- workstream: `{WORKSTREAM}`",
      f"- host: `{payload.get('host')}`",
      f"- model: `{payload.get('model')}`",
      f"- layers: `{payload.get('layers')}`",
      f"- required checks passed: `{str(payload.get('required_checks_passed')).lower()}`",
      "- speedup claims allowed: `false`",
      f"- resident API: `{probe.get('resident_api')}`",
      f"- resident shell invocations: `{probe.get('resident_shell_invocations')}`",
      f"- all-history source: `{payload.get('all_history', {}).get('history_artifact')}`",
      f"- V projection boundary: `{probe.get('full_attn_v_projection_boundary')}`",
      f"- FFN boundary: `{probe.get('full_attn_ffn_boundary')}`",
      "",
      "| output | comparison | max abs | RMSE |",
      "|---|---|---:|---:|",
  ]
  for name in (
      "l2_v",
      "l2_attn_pregate",
      "l2_attn_gated",
      "l2_attn_output",
      "l2_attn_residual",
      "l2_attn_post_norm",
  ):
    group = comparisons.get(name, {}) if isinstance(comparisons, dict) else {}
    lane = group.get("gpu_vs_oracle", {}) if isinstance(group, dict) else {}
    lines.append(f"| {name} | gpu_vs_oracle | {lane.get('max_abs_diff')} | {lane.get('rmse')} |")
  v_timing = timings.get("layer2_v_q6", {}) if isinstance(timings, dict) else {}
  lines += [
      "",
      "| kernel group | min us |",
      "|---|---:|",
      f"| layer7_v_q6 | {v_timing.get('v_projection_min_us') if isinstance(v_timing, dict) else None} |",
      f"| layer7_full_attn_input | {timings.get('resident_layer7_full_attn_input_kernel_sum_min_us') if isinstance(timings, dict) else None} |",
      f"| layer7_core_output | {timings.get('resident_layer7_full_attn_core_output_kernel_sum_min_us') if isinstance(timings, dict) else None} |",
      f"| layer7_attention_total | {timings.get('resident_layer7_full_attention_kernel_sum_min_us') if isinstance(timings, dict) else None} |",
      f"| two_linear_plus_layer7_attention | {timings.get('resident_two_linear_plus_layer7_attention_kernel_sum_min_us') if isinstance(timings, dict) else None} |",
      "",
      "This target-side process carries layer 5 and layer 6 GPU outputs into",
      "layer 7. GPU computes layer-7 attention RMSNorm, Q/K Q4 projections,",
      "V Q6 projection, full-attention core, gate, output projection, residual",
      "add, and post-attention RMSNorm. Layer-7 FFN remains a Q6-down boundary.",
      "This is captured single-token evidence, not decode throughput.",
      "",
  ]
  path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  args = parse_args()
  if args.resident_invocations <= 0:
    raise SystemExit("--resident-invocations must be positive")
  if args.layer != 5:
    raise SystemExit("--layer must be 5 for the layer-7 V Q6 handoff")

  layer0 = args.layer
  layer1 = args.layer + 1
  layer2 = args.layer + 2
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
  all_history_json = args.all_history_json.resolve()
  stamp = utc_stamp()
  created_at = iso_now()
  out_dir = args.out_dir or ROOT / f"output/gpu-resident-layer7-full-attn-v-q6-handoff-probe-{stamp}"
  out_dir.mkdir(parents=True, exist_ok=False)
  raw_dir = out_dir / "raw"
  raw_dir.mkdir()

  payloads0, conv0 = TWO.prefixed_payloads(layer0, conv0_path, "l0")
  payloads1, conv1 = TWO.prefixed_payloads(layer1, conv1_path, "l1")
  full_payloads, all_history = L7_INPUT.resolve_full_attention_payloads(all_history_json, layer2)
  CORE.add_layer7_tail_payloads(full_payloads)
  payloads = {**payloads0, **payloads1, **full_payloads}
  opencl_source = OPENCL_SOURCE.read_text(encoding="utf-8")
  opencl_source_hash = iq36_local.sha256_file(OPENCL_SOURCE)
  embedded_opencl_hash = hashlib.sha256(
      (
          opencl_source
          + PRECONV.POSTCONV.ATTENTION.ATTENTION_EXTRA_OPENCL
          + CORE.FULL_ATTN_CORE_EXTRA_OPENCL
      ).encode("utf-8")
  ).hexdigest()
  local_cpp = out_dir / "gpu_resident_layer7_full_attn_v_q6_handoff_probe.cpp"
  local_cpp.write_text(v_q6_probe_cpp(opencl_source), encoding="utf-8")

  remote_dir = f"{args.remote_root.rstrip('/')}/gpu-resident-layer7-full-attn-v-q6-handoff-probe-{stamp}"
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
            f"{remote_dir}/tests/gpu_resident_layer7_full_attn_v_q6_handoff_probe.cpp",
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

  executable = f"{remote_dir}/build/iq36-gpu-resident-layer7-full-attn-v-q6-handoff-probe"
  compile_cmd = " && ".join([
      f"source {shlex.quote(args.env_script)} >/dev/null 2>&1",
      (
          "g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic "
          f"-I {shlex.quote(remote_dir + '/include')} "
          f"{shlex.quote(remote_dir + '/src/gguf_loader.cpp')} "
          f"{shlex.quote(remote_dir + '/src/gpu_q4x8_matvec.cpp')} "
          f"{shlex.quote(remote_dir + '/tests/gpu_resident_layer7_full_attn_v_q6_handoff_probe.cpp')} "
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
      {"name": "arc_b390_selected", "pass": bool(probe and "B390" in str(probe.get("device_name", "")) and "B390" in str(probe.get("v_projection_device_name", "")))},
      {"name": "resident_api_fields_present", "pass": resident_fields_ok(probe, args.resident_invocations)},
      {"name": "layer0_checks_passed", "pass": PRECONV.nested_bool(probe, "checks", "layer0", "comparisons_passed")},
      {"name": "layer1_checks_passed", "pass": PRECONV.nested_bool(probe, "checks", "layer1", "comparisons_passed")},
      {"name": "layer1_residual_input_matches_oracle", "pass": PRECONV.nested_bool(probe, "checks", "layer1_residual_input_matches_oracle")},
      {"name": "layer2_core_output_matches_oracle", "pass": PRECONV.nested_bool(probe, "checks", "layer2_full_attn_core_output_matches_oracle")},
      {"name": "layer2_v_q6_gpu_boundary", "pass": PRECONV.nested_bool(probe, "checks", "layer2_v_q6_gpu_boundary")},
      {"name": "layer2_ffn_q6_boundary", "pass": PRECONV.nested_bool(probe, "checks", "layer2_ffn_q6_boundary")},
      {"name": "history_kv_state_payloads_present", "pass": PRECONV.nested_bool(probe, "checks", "history_kv_state_payloads_present")},
      {"name": "l2_v_matches_oracle", "pass": CORE.comparison_passed(probe, "l2_v", "gpu_vs_oracle")},
      {"name": "l2_attn_pregate_matches_oracle", "pass": CORE.full_attention_comparison_passed(probe, "l2_attn_pregate", "gpu_vs_oracle")},
      {"name": "l2_attn_gated_matches_oracle", "pass": CORE.full_attention_comparison_passed(probe, "l2_attn_gated", "gpu_vs_oracle")},
      {"name": "l2_attn_output_matches_oracle", "pass": CORE.full_attention_comparison_passed(probe, "l2_attn_output", "gpu_vs_oracle")},
      {"name": "l2_attn_residual_matches_oracle", "pass": CORE.full_attention_comparison_passed(probe, "l2_attn_residual", "gpu_vs_oracle")},
      {"name": "l2_attn_post_norm_matches_oracle", "pass": CORE.full_attention_comparison_passed(probe, "l2_attn_post_norm", "gpu_vs_oracle")},
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
      "all_history": all_history,
      "payloads": slim_payloads,
      "layers": [layer0, layer1, layer2],
      "resident_invocations": args.resident_invocations,
      "opencl_source": str(OPENCL_SOURCE.relative_to(ROOT)),
      "opencl_source_sha256": opencl_source_hash,
      "embedded_opencl_source_sha256": embedded_opencl_hash,
      "comparison_thresholds": comparison_thresholds,
      "probe_extra_opencl": [
          "rms_norm_hidden_f32",
          "full_attn_core_f32",
          "full_attn_gate_f32",
          "q6k_selected_down_matvec_row",
      ],
      "probe": probe,
      "checks": checks,
      "required_checks_passed": required_checks_passed,
      "speedup_claims_allowed": False,
  }
  manifest = {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "created_at": created_at,
      "tool": "tools/intel-qwen36-gpu-resident-layer7-full-attn-v-q6-handoff-probe.py",
      "artifact": str(out_dir),
      "host": args.host,
      "remote_dir": remote_dir,
      "layers": [layer0, layer1, layer2],
      "resident_invocations": args.resident_invocations,
      "conv_history_probes": payload["conv_history_probes"],
      "all_history": all_history,
      "comparison_thresholds": comparison_thresholds,
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
      "gpu_resident_layer7_full_attn_v_q6_handoff_probe",
      [
          ("required_checks_passed", required_checks_passed),
          ("resident_shell_invocations", args.resident_invocations),
          ("layer7_v_q6_projection_min_us", PRECONV.nested_number(timings, "layer2_v_q6", "v_projection_min_us")),
          ("resident_layer7_full_attention_kernel_sum_min_us", PRECONV.nested_number(timings, "resident_layer7_full_attention_kernel_sum_min_us")),
          ("l2_v_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l2_v", "gpu_vs_oracle", "max_abs_diff")),
          ("l2_attn_pregate_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l2_attn_pregate", "gpu_vs_oracle", "max_abs_diff")),
          ("l2_attn_output_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l2_attn_output", "gpu_vs_oracle", "max_abs_diff")),
          ("l2_attn_post_norm_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l2_attn_post_norm", "gpu_vs_oracle", "max_abs_diff")),
      ],
  )
  write_summary(out_dir / "summary.md", payload)
  print(out_dir)
  return 0 if required_checks_passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
