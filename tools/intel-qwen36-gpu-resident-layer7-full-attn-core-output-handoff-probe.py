#!/usr/bin/env python3
"""Run the resident GPU layer-7 full-attention core/output handoff probe."""

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
INPUT_TOOL = Path(__file__).with_name(
    "intel-qwen36-gpu-resident-layer7-full-attn-input-handoff-probe.py"
)
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA_VERSION = "intel-qwen36-gpu-resident-layer7-full-attn-core-output-handoff-probe-v0"
DEFAULT_HOST = "local"
DEFAULT_MODEL = "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf"
DEFAULT_ENV_SCRIPT = "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh"
DEFAULT_REMOTE_ROOT = "/home/intel/intel-qwen36-gpu"
DEFAULT_ORACLE_BUNDLE = ROOT / "oracle/r0-oracle-bundle-20260627T060028Z"
DEFAULT_ALL_HISTORY = ROOT / "output/r1-full-attn-all-history-capture-20260627T145615Z/history.json"
OPENCL_SOURCE = ROOT / "engine/gpu/opencl/q4x8_matvec.cl"
STRICT_COMPARISON_THRESHOLDS = {
    "max_abs_diff": 5e-3,
    "min_cosine": 0.99999,
    "mismatch_abs_diff": 5e-3,
    "rmse": 1e-3,
}
FULL_ATTN_COMPARISON_THRESHOLDS = {
    "max_abs_diff": 1.25e-2,
    "min_cosine": 0.99998,
    "mismatch_abs_diff": 1.25e-2,
    "rmse": 1e-3,
}


def load_input_tool() -> ModuleType:
  spec = importlib.util.spec_from_file_location("iq36_layer7_input_probe", INPUT_TOOL)
  if spec is None or spec.loader is None:
    raise SystemExit(f"failed to load layer7 input handoff tool: {INPUT_TOOL}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


L7_INPUT = load_input_tool()
TWO = L7_INPUT.TWO
PRECONV = L7_INPUT.PRECONV


FULL_ATTN_CORE_EXTRA_OPENCL = r'''

inline uint iq36_float_to_half_bits(float value) {
  const uint bits = as_uint(value);
  const uint sign = (bits >> 16) & 0x8000u;
  const uint exponent_bits = (bits >> 23) & 0xffu;
  int exponent = (int)exponent_bits - 127 + 15;
  uint mantissa = bits & 0x7fffffu;
  if (exponent_bits == 0xffu) {
    return sign | (mantissa == 0u ? 0x7c00u : 0x7e00u);
  }
  if (exponent <= 0) {
    if (exponent < -10) {
      return sign;
    }
    mantissa |= 0x800000u;
    const uint shift = (uint)(14 - exponent);
    uint half_mantissa = mantissa >> shift;
    const uint remainder = mantissa & ((1u << shift) - 1u);
    const uint halfway = 1u << (shift - 1u);
    if (remainder > halfway ||
        (remainder == halfway && (half_mantissa & 1u) != 0u)) {
      ++half_mantissa;
    }
    return sign | half_mantissa;
  }
  if (exponent >= 31) {
    return sign | 0x7c00u;
  }
  uint half_bits = sign | ((uint)exponent << 10u) | (mantissa >> 13u);
  const uint remainder = mantissa & 0x1fffu;
  if (remainder > 0x1000u ||
      (remainder == 0x1000u && (half_bits & 1u) != 0u)) {
    ++half_bits;
  }
  return half_bits;
}

inline float iq36_half_bits_to_float(uint half_bits) {
  const uint sign = (half_bits & 0x8000u) << 16u;
  uint exponent = (half_bits >> 10u) & 0x1fu;
  uint mantissa = half_bits & 0x03ffu;
  uint bits = 0u;
  if (exponent == 0u) {
    if (mantissa == 0u) {
      bits = sign;
    } else {
      exponent = 1u;
      while ((mantissa & 0x0400u) == 0u) {
        mantissa <<= 1u;
        --exponent;
      }
      mantissa &= 0x03ffu;
      bits = sign | ((exponent + 127u - 15u) << 23u) | (mantissa << 13u);
    }
  } else if (exponent == 31u) {
    bits = sign | 0x7f800000u | (mantissa << 13u);
  } else {
    bits = sign | ((exponent + 127u - 15u) << 23u) | (mantissa << 13u);
  }
  return as_float(bits);
}

inline float iq36_fp16_round(float value) {
  return iq36_half_bits_to_float(iq36_float_to_half_bits(value));
}

inline float iq36_dot_fp16_avx_like(__global const float* q_rope,
                                    __global const float* k_history,
                                    uint q_base,
                                    uint hist_base,
                                    uint head_dim) {
  float lanes[32];
  for (uint lane = 0; lane < 32; ++lane) {
    lanes[lane] = 0.0f;
  }
  for (uint i = 0; i < head_dim; i += 32) {
    for (uint lane = 0; lane < 32; ++lane) {
      lanes[lane] = fma(iq36_fp16_round(q_rope[q_base + i + lane]),
                        iq36_fp16_round(k_history[hist_base + i + lane]),
                        lanes[lane]);
    }
  }
  float reduced[8];
  for (uint lane = 0; lane < 8; ++lane) {
    const float a = lanes[lane] + lanes[16 + lane];
    const float b = lanes[8 + lane] + lanes[24 + lane];
    reduced[lane] = a + b;
  }
  const float t0 = reduced[0] + reduced[4];
  const float t1 = reduced[1] + reduced[5];
  const float t2 = reduced[2] + reduced[6];
  const float t3 = reduced[3] + reduced[7];
  return (t0 + t1) + (t2 + t3);
}

__kernel void full_attn_core_f32(__global const float* q_rope,
                                 __global const float* k_history,
                                 __global const float* v_history,
                                 uint token_count,
                                 uint head_dim,
                                 uint q_head_count,
                                 uint kv_head_count,
                                 float attention_scale,
                                 __global float* attn_pregate) {
  const uint index = (uint)get_global_id(0);
  const uint q_size = q_head_count * head_dim;
  if (index >= q_size) {
    return;
  }
  const uint q_head = index / head_dim;
  const uint dim = index - q_head * head_dim;
  const uint gqa_group = q_head_count / kv_head_count;
  const uint kv_head = q_head / gqa_group;
  const uint q_base = q_head * head_dim;
  const uint kv_base = kv_head * head_dim;
  const uint kv_size = kv_head_count * head_dim;

  float sum = 0.0f;
  float max_score = -INFINITY;
  float out_acc = 0.0f;
  for (uint token = 0; token < token_count; ++token) {
    const uint hist_base = token * kv_size + kv_base;
    const float dot =
        iq36_dot_fp16_avx_like(q_rope, k_history, q_base, hist_base, head_dim);
    const float score = dot * attention_scale;
    const float old_max = max_score;
    float max_scale = 1.0f;
    float value_scale = 1.0f;
    if (score > max_score) {
      max_score = score;
      max_scale = exp(old_max - max_score);
      out_acc = iq36_fp16_round(out_acc * max_scale);
    } else {
      value_scale = exp(score - max_score);
    }
    out_acc = iq36_fp16_round(
        fma(iq36_fp16_round(v_history[token * kv_size + kv_base + dim]),
            value_scale,
            out_acc));
    sum = sum * max_scale + value_scale;
  }
  attn_pregate[index] = sum == 0.0f ? 0.0f : out_acc / sum;
}

__kernel void full_attn_gate_f32(__global const float* q_full,
                                 __global const float* attn_pregate,
                                 uint head_dim,
                                 uint q_head_count,
                                 __global float* attn_gated) {
  const uint index = (uint)get_global_id(0);
  const uint q_size = q_head_count * head_dim;
  if (index >= q_size) {
    return;
  }
  const uint q_head = index / head_dim;
  const uint dim = index - q_head * head_dim;
  const float gate = q_full[q_head * head_dim * 2U + head_dim + dim];
  const float sigmoid = 1.0f / (1.0f + exp(-gate));
  attn_gated[index] = attn_pregate[index] * sigmoid;
}
'''


FULL_ATTN_CORE_OUTPUT_CPP = r'''

struct FullAttentionCoreGateTiming {
  double core_min_us = 0.0;
  double core_mean_us = 0.0;
  double gate_min_us = 0.0;
  double gate_mean_us = 0.0;
  double core_gate_kernel_sum_min_us = 0.0;
  double core_gate_kernel_sum_mean_us = 0.0;
};

struct FullAttentionCoreGateRun {
  std::vector<float> attn_pregate;
  std::vector<float> attn_gated;
  FullAttentionCoreGateTiming timing;
  std::string platform_name;
  std::string device_name;
  std::string build_log;
  double program_build_ms = 0.0;
};

constexpr double kFullAttentionComponentMismatchThreshold = 1.25e-2;
constexpr double kFullAttentionComponentMaxAbsDiffThreshold = 1.25e-2;
constexpr double kFullAttentionComponentRmseThreshold = 1e-3;
constexpr double kFullAttentionComponentMinCosine = 0.99998;

bool ComparePassedFullAttentionComponent(const iq36::VectorCompareStats& stats) {
  return stats.same_size &&
         stats.finite &&
         stats.mismatch_count == 0 &&
         stats.max_abs_diff <= kFullAttentionComponentMaxAbsDiffThreshold &&
         stats.rmse <= kFullAttentionComponentRmseThreshold &&
         stats.cosine >= kFullAttentionComponentMinCosine;
}

bool CompareGroupsPassedFullAttentionComponent(
    const std::vector<NamedCompareGroup>& groups) {
  bool ok = true;
  for (const auto& group : groups) {
    ok = ok &&
         ComparePassedFullAttentionComponent(group.cpu_vs_oracle) &&
         ComparePassedFullAttentionComponent(group.gpu_vs_cpu) &&
         ComparePassedFullAttentionComponent(group.gpu_vs_oracle);
  }
  return ok;
}

void AppendFullAttentionComponentCompare(
    std::vector<NamedCompareGroup>& groups,
    const std::string& name,
    const std::vector<float>& cpu,
    const std::vector<float>& gpu,
    const std::vector<float>& oracle) {
  groups.push_back({
      name,
      iq36::compare_vectors(cpu, oracle, kFullAttentionComponentMismatchThreshold),
      iq36::compare_vectors(gpu, cpu, kFullAttentionComponentMismatchThreshold),
      iq36::compare_vectors(gpu, oracle, kFullAttentionComponentMismatchThreshold),
  });
}

std::vector<float> FlattenFullAttentionHistory(
    const std::vector<std::vector<float>>& history,
    const std::vector<float>& current) {
  std::vector<float> flat;
  flat.reserve((history.size() + 1) * kFullKvValues);
  for (const auto& item : history) {
    Require(item.size() == kFullKvValues, "full attention history item size mismatch");
    flat.insert(flat.end(), item.begin(), item.end());
  }
  Require(current.size() == kFullKvValues, "full attention current item size mismatch");
  flat.insert(flat.end(), current.begin(), current.end());
  return flat;
}

std::uint32_t FloatBitsFull(float value) {
  std::uint32_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value), "unexpected float size");
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

float BitsFloatFull(std::uint32_t bits) {
  float value = 0.0f;
  static_assert(sizeof(bits) == sizeof(value), "unexpected float size");
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

std::uint16_t FloatToHalfBitsFull(float value) {
  const std::uint32_t bits = FloatBitsFull(value);
  const std::uint16_t sign =
      static_cast<std::uint16_t>((bits >> 16) & 0x8000u);
  const std::uint32_t exponent = (bits >> 23) & 0xffu;
  std::uint32_t mantissa = bits & 0x7fffffu;
  if (exponent == 0xffu) {
    return static_cast<std::uint16_t>(
        sign | (mantissa == 0 ? 0x7c00u : 0x7e00u));
  }
  const int half_exponent = static_cast<int>(exponent) - 127 + 15;
  if (half_exponent >= 31) {
    return static_cast<std::uint16_t>(sign | 0x7c00u);
  }
  if (half_exponent <= 0) {
    if (half_exponent < -10) {
      return sign;
    }
    mantissa |= 0x800000u;
    const int shift = 14 - half_exponent;
    std::uint32_t half_mantissa = mantissa >> shift;
    const std::uint32_t round_bit = (mantissa >> (shift - 1)) & 1u;
    const std::uint32_t sticky_mask = (1u << (shift - 1)) - 1u;
    const std::uint32_t sticky = mantissa & sticky_mask;
    if (round_bit != 0 && (sticky != 0 || (half_mantissa & 1u) != 0)) {
      ++half_mantissa;
    }
    return static_cast<std::uint16_t>(sign | half_mantissa);
  }
  std::uint32_t half_mantissa = mantissa >> 13;
  std::uint32_t half_exp_bits =
      static_cast<std::uint32_t>(half_exponent) << 10;
  const std::uint32_t round_bits = mantissa & 0x1fffu;
  if (round_bits > 0x1000u ||
      (round_bits == 0x1000u && (half_mantissa & 1u) != 0)) {
    ++half_mantissa;
    if (half_mantissa == 0x400u) {
      half_mantissa = 0;
      half_exp_bits += 0x400u;
      if (half_exp_bits >= 0x7c00u) {
        return static_cast<std::uint16_t>(sign | 0x7c00u);
      }
    }
  }
  return static_cast<std::uint16_t>(sign | half_exp_bits | half_mantissa);
}

float HalfBitsToFloatFull(std::uint16_t half) {
  const std::uint32_t sign =
      static_cast<std::uint32_t>(half & 0x8000u) << 16;
  std::uint32_t exponent = (half >> 10) & 0x1fu;
  std::uint32_t mantissa = half & 0x03ffu;
  std::uint32_t bits = 0;
  if (exponent == 0) {
    if (mantissa == 0) {
      bits = sign;
    } else {
      int normalized_exponent = 1;
      while ((mantissa & 0x0400u) == 0) {
        mantissa <<= 1;
        --normalized_exponent;
      }
      mantissa &= 0x03ffu;
      const auto exponent_bits =
          static_cast<std::uint32_t>(normalized_exponent + 127 - 15);
      bits = sign | (exponent_bits << 23) | (mantissa << 13);
    }
  } else if (exponent == 31) {
    bits = sign | 0x7f800000u | (mantissa << 13);
  } else {
    exponent = exponent + 127 - 15;
    bits = sign | (exponent << 23) | (mantissa << 13);
  }
  return BitsFloatFull(bits);
}

float RoundToFp16Full(float value) {
  return HalfBitsToFloatFull(FloatToHalfBitsFull(value));
}

std::vector<float> RoundVectorToFp16Full(const std::vector<float>& values) {
  std::vector<float> rounded;
  rounded.reserve(values.size());
  for (const auto value : values) {
    rounded.push_back(RoundToFp16Full(value));
  }
  return rounded;
}

std::vector<std::vector<float>> RoundHistoryToFp16Full(
    const std::vector<std::vector<float>>& history) {
  std::vector<std::vector<float>> rounded;
  rounded.reserve(history.size());
  for (const auto& item : history) {
    rounded.push_back(RoundVectorToFp16Full(item));
  }
  return rounded;
}

FullAttentionCoreGateRun RunGpuFullAttentionCoreGate(
    const std::vector<float>& q_rope,
    const std::vector<float>& k_history_flat,
    const std::vector<float>& v_history_flat,
    const std::vector<float>& q_full,
    const std::string& device_substring,
    int repeat) {
  Require(q_rope.size() == kFullQValues, "full attention core q_rope size mismatch");
  Require(q_full.size() == kFullQFullValues, "full attention gate q_full size mismatch");
  Require(k_history_flat.size() == static_cast<std::size_t>(kFullUpdatedHistoryTokenCount * kFullKvValues),
          "full attention core k history size mismatch");
  Require(v_history_flat.size() == static_cast<std::size_t>(kFullUpdatedHistoryTokenCount * kFullKvValues),
          "full attention core v history size mismatch");

  FullAttentionCoreGateRun run;
  run.attn_pregate.assign(kFullQValues, 0.0f);
  run.attn_gated.assign(kFullQValues, 0.0f);

  OpenClApi api;
  const auto selected = SelectDevice(api, device_substring);
  run.platform_name = selected.platform_name;
  run.device_name = selected.device_name;
  cl_int err = kClSuccess;
  cl_context context = api.clCreateContext(nullptr, 1, &selected.device, nullptr, nullptr, &err);
  Check(err, "clCreateContext(full attention core)");
  cl_command_queue queue = api.clCreateCommandQueue(context, selected.device, kClQueueProfilingEnable, &err);
  Check(err, "clCreateCommandQueue(full attention core)");
  const char* source = kOpenClSource;
  const std::size_t source_len = std::strlen(kOpenClSource);
  cl_program program = api.clCreateProgramWithSource(context, 1, &source, &source_len, &err);
  Check(err, "clCreateProgramWithSource(full attention core)");
  const auto build_begin = std::chrono::steady_clock::now();
  err = api.clBuildProgram(program, 1, &selected.device, "", nullptr, nullptr);
  const auto build_end = std::chrono::steady_clock::now();
  run.program_build_ms =
      std::chrono::duration<double, std::milli>(build_end - build_begin).count();
  run.build_log = BuildLog(api, program, selected.device);
  Check(err, "clBuildProgram(full attention core)");
  cl_kernel core_kernel = api.clCreateKernel(program, "full_attn_core_f32", &err);
  Check(err, "clCreateKernel(full_attn_core_f32)");
  cl_kernel gate_kernel = api.clCreateKernel(program, "full_attn_gate_f32", &err);
  Check(err, "clCreateKernel(full_attn_gate_f32)");

  cl_mem q_rope_buffer = nullptr;
  cl_mem k_history_buffer = nullptr;
  cl_mem v_history_buffer = nullptr;
  cl_mem q_full_buffer = nullptr;
  cl_mem pregate_buffer = nullptr;
  cl_mem gated_buffer = nullptr;
  try {
    q_rope_buffer = api.clCreateBuffer(context, kClMemReadOnly,
                                       q_rope.size() * sizeof(float),
                                       nullptr, &err);
    Check(err, "clCreateBuffer(full attention q_rope)");
    k_history_buffer = api.clCreateBuffer(context, kClMemReadOnly,
                                          k_history_flat.size() * sizeof(float),
                                          nullptr, &err);
    Check(err, "clCreateBuffer(full attention k history)");
    v_history_buffer = api.clCreateBuffer(context, kClMemReadOnly,
                                          v_history_flat.size() * sizeof(float),
                                          nullptr, &err);
    Check(err, "clCreateBuffer(full attention v history)");
    q_full_buffer = api.clCreateBuffer(context, kClMemReadOnly,
                                       q_full.size() * sizeof(float),
                                       nullptr, &err);
    Check(err, "clCreateBuffer(full attention q_full)");
    pregate_buffer = api.clCreateBuffer(context, kClMemReadWriteLocal,
                                        run.attn_pregate.size() * sizeof(float),
                                        nullptr, &err);
    Check(err, "clCreateBuffer(full attention pregate)");
    gated_buffer = api.clCreateBuffer(context, kClMemWriteOnly,
                                      run.attn_gated.size() * sizeof(float),
                                      nullptr, &err);
    Check(err, "clCreateBuffer(full attention gated)");

    Check(api.clEnqueueWriteBuffer(queue, q_rope_buffer, kClTrue, 0,
                                   q_rope.size() * sizeof(float), q_rope.data(),
                                   0, nullptr, nullptr),
          "clEnqueueWriteBuffer(full attention q_rope)");
    Check(api.clEnqueueWriteBuffer(queue, k_history_buffer, kClTrue, 0,
                                   k_history_flat.size() * sizeof(float),
                                   k_history_flat.data(), 0, nullptr, nullptr),
          "clEnqueueWriteBuffer(full attention k history)");
    Check(api.clEnqueueWriteBuffer(queue, v_history_buffer, kClTrue, 0,
                                   v_history_flat.size() * sizeof(float),
                                   v_history_flat.data(), 0, nullptr, nullptr),
          "clEnqueueWriteBuffer(full attention v history)");
    Check(api.clEnqueueWriteBuffer(queue, q_full_buffer, kClTrue, 0,
                                   q_full.size() * sizeof(float), q_full.data(),
                                   0, nullptr, nullptr),
          "clEnqueueWriteBuffer(full attention q_full)");

    const cl_uint token_count_arg = static_cast<cl_uint>(kFullUpdatedHistoryTokenCount);
    const cl_uint head_dim_arg = static_cast<cl_uint>(kFullHeadDim);
    const cl_uint q_head_count_arg = static_cast<cl_uint>(kFullQHeadCount);
    const cl_uint kv_head_count_arg = static_cast<cl_uint>(kFullKvHeadCount);
    const float attention_scale_arg = 0.0625f;
    Check(api.clSetKernelArg(core_kernel, 0, sizeof(q_rope_buffer), &q_rope_buffer), "clSetKernelArg(full attention core 0)");
    Check(api.clSetKernelArg(core_kernel, 1, sizeof(k_history_buffer), &k_history_buffer), "clSetKernelArg(full attention core 1)");
    Check(api.clSetKernelArg(core_kernel, 2, sizeof(v_history_buffer), &v_history_buffer), "clSetKernelArg(full attention core 2)");
    Check(api.clSetKernelArg(core_kernel, 3, sizeof(token_count_arg), &token_count_arg), "clSetKernelArg(full attention core 3)");
    Check(api.clSetKernelArg(core_kernel, 4, sizeof(head_dim_arg), &head_dim_arg), "clSetKernelArg(full attention core 4)");
    Check(api.clSetKernelArg(core_kernel, 5, sizeof(q_head_count_arg), &q_head_count_arg), "clSetKernelArg(full attention core 5)");
    Check(api.clSetKernelArg(core_kernel, 6, sizeof(kv_head_count_arg), &kv_head_count_arg), "clSetKernelArg(full attention core 6)");
    Check(api.clSetKernelArg(core_kernel, 7, sizeof(attention_scale_arg), &attention_scale_arg), "clSetKernelArg(full attention core 7)");
    Check(api.clSetKernelArg(core_kernel, 8, sizeof(pregate_buffer), &pregate_buffer), "clSetKernelArg(full attention core 8)");
    Check(api.clSetKernelArg(gate_kernel, 0, sizeof(q_full_buffer), &q_full_buffer), "clSetKernelArg(full attention gate 0)");
    Check(api.clSetKernelArg(gate_kernel, 1, sizeof(pregate_buffer), &pregate_buffer), "clSetKernelArg(full attention gate 1)");
    Check(api.clSetKernelArg(gate_kernel, 2, sizeof(head_dim_arg), &head_dim_arg), "clSetKernelArg(full attention gate 2)");
    Check(api.clSetKernelArg(gate_kernel, 3, sizeof(q_head_count_arg), &q_head_count_arg), "clSetKernelArg(full attention gate 3)");
    Check(api.clSetKernelArg(gate_kernel, 4, sizeof(gated_buffer), &gated_buffer), "clSetKernelArg(full attention gate 4)");

    const std::size_t q_global = static_cast<std::size_t>(kFullQValues);
    std::vector<double> core_times;
    std::vector<double> gate_times;
    core_times.reserve(static_cast<std::size_t>(repeat));
    gate_times.reserve(static_cast<std::size_t>(repeat));
    for (int i = 0; i < repeat; ++i) {
      cl_event core_event = nullptr;
      cl_event gate_event = nullptr;
      Check(api.clEnqueueNDRangeKernel(queue, core_kernel, 1, nullptr,
                                       &q_global, nullptr, 0, nullptr,
                                       &core_event),
            "clEnqueueNDRangeKernel(full attention core)");
      Check(api.clEnqueueNDRangeKernel(queue, gate_kernel, 1, nullptr,
                                       &q_global, nullptr, 0, nullptr,
                                       &gate_event),
            "clEnqueueNDRangeKernel(full attention gate)");
      Check(api.clFinish(queue), "clFinish(full attention core/gate)");
      core_times.push_back(EventUs(api, core_event));
      gate_times.push_back(EventUs(api, gate_event));
      api.clReleaseEvent(core_event);
      api.clReleaseEvent(gate_event);
    }
    Check(api.clEnqueueReadBuffer(queue, pregate_buffer, kClTrue, 0,
                                  run.attn_pregate.size() * sizeof(float),
                                  run.attn_pregate.data(), 0, nullptr, nullptr),
          "clEnqueueReadBuffer(full attention pregate)");
    Check(api.clEnqueueReadBuffer(queue, gated_buffer, kClTrue, 0,
                                  run.attn_gated.size() * sizeof(float),
                                  run.attn_gated.data(), 0, nullptr, nullptr),
          "clEnqueueReadBuffer(full attention gated)");
    run.timing.core_min_us = Min(core_times);
    run.timing.core_mean_us = Mean(core_times);
    run.timing.gate_min_us = Min(gate_times);
    run.timing.gate_mean_us = Mean(gate_times);
    run.timing.core_gate_kernel_sum_min_us =
        run.timing.core_min_us + run.timing.gate_min_us;
    run.timing.core_gate_kernel_sum_mean_us =
        run.timing.core_mean_us + run.timing.gate_mean_us;
  } catch (...) {
    ReleaseMem(api, &gated_buffer);
    ReleaseMem(api, &pregate_buffer);
    ReleaseMem(api, &q_full_buffer);
    ReleaseMem(api, &v_history_buffer);
    ReleaseMem(api, &k_history_buffer);
    ReleaseMem(api, &q_rope_buffer);
    api.clReleaseKernel(gate_kernel);
    api.clReleaseKernel(core_kernel);
    api.clReleaseProgram(program);
    api.clReleaseCommandQueue(queue);
    api.clReleaseContext(context);
    throw;
  }
  ReleaseMem(api, &gated_buffer);
  ReleaseMem(api, &pregate_buffer);
  ReleaseMem(api, &q_full_buffer);
  ReleaseMem(api, &v_history_buffer);
  ReleaseMem(api, &k_history_buffer);
  ReleaseMem(api, &q_rope_buffer);
  api.clReleaseKernel(gate_kernel);
  api.clReleaseKernel(core_kernel);
  api.clReleaseProgram(program);
  api.clReleaseCommandQueue(queue);
  api.clReleaseContext(context);
  return run;
}

void WriteFullCoreGateTiming(const FullAttentionCoreGateTiming& timing) {
  std::cout << "{";
  std::cout << "\"core_min_us\":" << timing.core_min_us << ",";
  std::cout << "\"core_mean_us\":" << timing.core_mean_us << ",";
  std::cout << "\"gate_min_us\":" << timing.gate_min_us << ",";
  std::cout << "\"gate_mean_us\":" << timing.gate_mean_us << ",";
  std::cout << "\"core_gate_kernel_sum_min_us\":"
            << timing.core_gate_kernel_sum_min_us << ",";
  std::cout << "\"core_gate_kernel_sum_mean_us\":"
            << timing.core_gate_kernel_sum_mean_us;
  std::cout << "}";
}

void WriteFullAttentionOutputFrontTiming(const AttentionFrontTiming& timing) {
  std::cout << "{";
  std::cout << "\"output_projection_min_us\":"
            << timing.output_projection_min_us << ",";
  std::cout << "\"output_projection_mean_us\":"
            << timing.output_projection_mean_us << ",";
  std::cout << "\"output_projection_host_q8_bridge_us\":"
            << timing.output_projection_host_q8_bridge_us << ",";
  std::cout << "\"residual_add_min_us\":"
            << timing.residual_add_min_us << ",";
  std::cout << "\"residual_add_mean_us\":"
            << timing.residual_add_mean_us << ",";
  std::cout << "\"ffn_rmsnorm_min_us\":"
            << timing.ffn_rmsnorm_min_us << ",";
  std::cout << "\"ffn_rmsnorm_mean_us\":"
            << timing.ffn_rmsnorm_mean_us << ",";
  std::cout << "\"attention_front_kernel_sum_min_us\":"
            << timing.attention_front_kernel_sum_min_us << ",";
  std::cout << "\"attention_front_kernel_sum_mean_us\":"
            << timing.attention_front_kernel_sum_mean_us;
  std::cout << "}";
}

int main(int argc, char** argv) {
  try {
    const Args args = ParseArgs(argc, argv);
    const int layer0 = args.layer;
    const int layer1 = args.layer + 1;
    const int layer2 = args.layer + 2;
    Require(layer0 == 5 && layer1 == 6 && layer2 == 7,
            "layer7 core/output probe expects --layer 5");
    const auto index = iq36::parse_gguf_model_index(args.model_path);
    const auto load_map = iq36::validate_qwen36_load_map(index);
    const float rms_norm_epsilon = MetadataFloat(
        index, "qwen35moe.attention.layer_norm_rms_epsilon", 1e-6f);
    const auto head_dim = MetadataUIntFull(
        index, "qwen35moe.attention.key_length", kFullHeadDim);
    const auto value_length = MetadataUIntFull(
        index, "qwen35moe.attention.value_length", kFullHeadDim);
    const auto q_head_count = MetadataUIntFull(
        index, "qwen35moe.attention.head_count", kFullQHeadCount);
    const auto kv_head_count = MetadataUIntFull(
        index, "qwen35moe.attention.head_count_kv", kFullKvHeadCount);
    const auto full_attention_interval = MetadataUIntFull(
        index, "qwen35moe.full_attention_interval", 4);
    const auto rope_dimension_count = MetadataUIntFull(
        index, "qwen35moe.rope.dimension_count", 64);
    const auto rope_context_length = MetadataUIntFull(
        index, "qwen35moe.context_length", 262144);
    const auto rope_sections = MetadataIntArrayFull(
        index, "qwen35moe.rope.dimension_sections", {11, 11, 10, 0});
    const float rope_freq_base = MetadataFloat(
        index, "qwen35moe.rope.freq_base", 10000000.0f);
    constexpr float kRopeFreqScale = 1.0f;
    constexpr float kRopeExtFactor = 0.0f;
    constexpr float kRopeAttnFactor = 1.0f;
    constexpr float kRopeBetaFast = 32.0f;
    constexpr float kRopeBetaSlow = 1.0f;
    constexpr float kAttentionScale = 0.0625f;

    const auto layer0_tensors = ResolveLayerTensorBundle(index, layer0);
    const auto layer1_tensors = ResolveLayerTensorBundle(index, layer1);
    const auto layer2_tensors = ResolveFullAttentionTensorBundle(index, layer2);
    const auto layer0_oracle = LoadLayerOraclePayloads(args.payload_dir, "l0");
    const auto layer1_oracle = LoadLayerOraclePayloads(args.payload_dir, "l1");
    const auto layer2_oracle = LoadFullAttentionPayloads(args.payload_dir);
    const auto oracle_attn_residual =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "full_attn_residual.bin"));
    const auto oracle_attn_post_norm =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "full_attn_post_norm.bin"));

    const auto layer0_run = RunResidentLinearLayerShell(
        args, index, layer0_tensors, layer0_oracle,
        layer0_oracle.residual_input, layer0_oracle.residual_input,
        rms_norm_epsilon);
    const auto layer1_run = RunResidentLinearLayerShell(
        args, index, layer1_tensors, layer1_oracle,
        layer0_run.native_layer_output, layer0_run.gpu_layer_output,
        rms_norm_epsilon);

    const auto full_shapes = CheckFullAttentionShapes(layer2_tensors);
    const bool full_shapes_ok = FullAttentionShapesPassed(full_shapes);
    const bool layer2_payload_counts_ok =
        FullAttentionPayloadCountsOk(layer2_oracle) &&
        oracle_attn_residual.size() == kHiddenSize &&
        oracle_attn_post_norm.size() == kHiddenSize;
    const bool metadata_ok =
        full_attention_interval == 4 &&
        head_dim == kFullHeadDim &&
        value_length == kFullHeadDim &&
        q_head_count == kFullQHeadCount &&
        kv_head_count == kFullKvHeadCount &&
        rope_dimension_count == 64 &&
        rope_context_length == 262144 &&
        rope_sections == std::vector<std::int64_t>({11, 11, 10, 0}) &&
        rope_freq_base == 10000000.0f;

    std::ifstream model(args.model_path, std::ios::binary);
    Require(static_cast<bool>(model), "model file could not be opened");
    const auto attn_norm_weight =
        ReadF32TensorPayload(model, *layer2_tensors.attn_norm_tensor,
                             static_cast<std::size_t>(kHiddenSize));
    const auto q_norm_weight =
        iq36::decode_tensor_row(args.model_path, index,
                                layer2_tensors.q_norm_tensor_name, 0);
    const auto k_norm_weight =
        iq36::decode_tensor_row(args.model_path, index,
                                layer2_tensors.k_norm_tensor_name, 0);
    const auto ffn_norm_weight =
        iq36::decode_tensor_row(args.model_path, index,
                                LayerTensorName(layer2, "post_attention_norm.weight"), 0);

    const auto native_qkv = iq36::run_qwen36_full_attention_qkv_projection(
        args.model_path,
        index,
        layer2,
        layer1_run.native_layer_output,
        rms_norm_epsilon);
    const auto layer2_rms_gpu = RunGpuLayerInputRmsNorm(
        layer1_run.gpu_layer_output,
        attn_norm_weight,
        rms_norm_epsilon,
        args.device_substring,
        args.repeat);
    const auto layer2_qk_gpu = RunGpuFullAttentionQkFront(
        args.model_path,
        *layer2_tensors.q_tensor,
        *layer2_tensors.k_tensor,
        layer2_rms_gpu.attn_norm,
        args.device_substring,
        args.repeat);
    const auto gpu_q_split = SplitFullAttentionQ(layer2_qk_gpu.q_full);
    const auto gpu_q_normed = ApplyRepeatedRmsNormFull(
        gpu_q_split.q_raw, q_norm_weight, rms_norm_epsilon);
    const auto gpu_k_normed = ApplyRepeatedRmsNormFull(
        layer2_qk_gpu.k_raw, k_norm_weight, rms_norm_epsilon);
    const auto gpu_rope = iq36::run_qwen36_full_attention_rope(
        gpu_q_normed,
        gpu_k_normed,
        kFullSourceTokenPosition,
        head_dim,
        rope_dimension_count,
        rope_sections,
        rope_context_length,
        rope_freq_base,
        kRopeFreqScale,
        kRopeExtFactor,
        kRopeAttnFactor,
        kRopeBetaFast,
        kRopeBetaSlow);
    const auto native_rope = iq36::run_qwen36_full_attention_rope(
        native_qkv.q_normed,
        native_qkv.k_normed,
        kFullSourceTokenPosition,
        head_dim,
        rope_dimension_count,
        rope_sections,
        rope_context_length,
        rope_freq_base,
        kRopeFreqScale,
        kRopeExtFactor,
        kRopeAttnFactor,
        kRopeBetaFast,
        kRopeBetaSlow);

    auto native_k_history = layer2_oracle.k_history;
    auto native_v_history = layer2_oracle.v_history;
    native_k_history.push_back(layer2_oracle.k_rope);
    native_v_history.push_back(layer2_oracle.v);
    const auto native_core = iq36::run_qwen36_full_attention_core(
        layer2_oracle.q_rope,
        native_k_history,
        native_v_history,
        head_dim,
        q_head_count,
        kv_head_count,
        kAttentionScale);
    const auto native_gate = iq36::run_qwen36_full_attention_gate(
        layer2_oracle.q_full, native_core.attn_pregate, head_dim);
    const auto native_attn_output = iq36::matvec_tensor(
        args.model_path,
        index,
        LayerTensorName(layer2, "attn_output.weight"),
        native_gate.attn_gated);
    const auto native_attn_residual =
        iq36::add_vectors(layer2_oracle.residual_input, native_attn_output);
    const auto native_attn_post_norm =
        iq36::apply_rms_norm(native_attn_residual, ffn_norm_weight, rms_norm_epsilon);

    auto gpu_k_history = layer2_oracle.k_history;
    auto gpu_v_history = layer2_oracle.v_history;
    gpu_k_history.push_back(layer2_oracle.k_rope);
    gpu_v_history.push_back(layer2_oracle.v);
    std::vector<float> gpu_k_history_flat;
    std::vector<float> gpu_v_history_flat;
    for (const auto& item : gpu_k_history) {
      gpu_k_history_flat.insert(gpu_k_history_flat.end(), item.begin(), item.end());
    }
    for (const auto& item : gpu_v_history) {
      gpu_v_history_flat.insert(gpu_v_history_flat.end(), item.begin(), item.end());
    }
    const auto core_gate_gpu = RunGpuFullAttentionCoreGate(
        layer2_oracle.q_rope,
        gpu_k_history_flat,
        gpu_v_history_flat,
        layer2_qk_gpu.q_full,
        args.device_substring,
        args.repeat);
    const auto attention_gpu = RunGpuAttentionFront(
        args.model_path,
        *iq36::find_tensor(index, LayerTensorName(layer2, "attn_output.weight")),
        core_gate_gpu.attn_gated,
        layer1_run.gpu_layer_output,
        ffn_norm_weight,
        rms_norm_epsilon,
        args.device_substring,
        args.repeat);

    std::vector<NamedCompareGroup> strict_groups;
    AppendCpuGpuOracleCompare(strict_groups, "l2_residual_input",
                              layer1_run.native_layer_output,
                              layer1_run.gpu_layer_output,
                              layer2_oracle.residual_input);
    AppendCpuGpuOracleCompare(strict_groups, "l2_attn_norm",
                              native_qkv.attention_norm,
                              layer2_rms_gpu.attn_norm,
                              layer2_oracle.attn_norm);
    AppendCpuGpuOracleCompare(strict_groups, "l2_q_full",
                              native_qkv.q_full,
                              layer2_qk_gpu.q_full,
                              layer2_oracle.q_full);
    AppendCpuGpuOracleCompare(strict_groups, "l2_q_rope",
                              native_rope.q_rope,
                              gpu_rope.q_rope,
                              layer2_oracle.q_rope);
    AppendCpuGpuOracleCompare(strict_groups, "l2_k_rope",
                              native_rope.k_rope,
                              gpu_rope.k_rope,
                              layer2_oracle.k_rope);
    std::vector<NamedCompareGroup> full_attention_groups;
    AppendFullAttentionComponentCompare(full_attention_groups, "l2_attn_pregate",
                                        native_core.attn_pregate,
                                        core_gate_gpu.attn_pregate,
                                        layer2_oracle.attn_pregate);
    AppendFullAttentionComponentCompare(full_attention_groups, "l2_attn_gated",
                                        native_gate.attn_gated,
                                        core_gate_gpu.attn_gated,
                                        layer2_oracle.attn_gated);
    AppendFullAttentionComponentCompare(full_attention_groups, "l2_attn_output",
                                        native_attn_output,
                                        attention_gpu.linear_attn_out,
                                        layer2_oracle.attn_output);
    AppendFullAttentionComponentCompare(full_attention_groups, "l2_attn_residual",
                                        native_attn_residual,
                                        attention_gpu.attn_residual,
                                        oracle_attn_residual);
    AppendFullAttentionComponentCompare(full_attention_groups, "l2_attn_post_norm",
                                        native_attn_post_norm,
                                        attention_gpu.attn_post_norm,
                                        oracle_attn_post_norm);

    const auto k_raw_gpu_vs_cpu = iq36::compare_vectors(
        layer2_qk_gpu.k_raw, native_qkv.k_raw, kMismatchThreshold);
    const auto native_v_vs_oracle = iq36::compare_vectors(
        native_qkv.v, layer2_oracle.v, kMismatchThreshold);
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
    const bool layer2_strict_input_ok =
        CompareGroupsPassed(strict_groups) &&
        ComparePassed(k_raw_gpu_vs_cpu) &&
        ComparePassed(native_v_vs_oracle);
    const bool layer2_full_component_ok =
        CompareGroupsPassedFullAttentionComponent(full_attention_groups);
    const bool layer2_comparisons_ok =
        layer2_strict_input_ok && layer2_full_component_ok;
    const bool layer2_timing_positive =
        layer2_rms_gpu.timing.rmsnorm_min_us > 0.0 &&
        layer2_qk_gpu.timing.q_projection_min_us > 0.0 &&
        layer2_qk_gpu.timing.k_projection_min_us > 0.0 &&
        core_gate_gpu.timing.core_min_us > 0.0 &&
        core_gate_gpu.timing.gate_min_us > 0.0 &&
        attention_gpu.timing.output_projection_min_us > 0.0 &&
        attention_gpu.timing.residual_add_min_us > 0.0 &&
        attention_gpu.timing.ffn_rmsnorm_min_us > 0.0;
    const bool layer2_arc_selected =
        layer2_rms_gpu.device_name.find(args.device_substring) != std::string::npos &&
        layer2_qk_gpu.device_name.find(args.device_substring) != std::string::npos &&
        core_gate_gpu.device_name.find(args.device_substring) != std::string::npos &&
        attention_gpu.device_name.find(args.device_substring) != std::string::npos;
    const bool state_carry_ok = ComparePassed(layer1_run.comparisons[0].gpu_vs_oracle);
    const bool v_q6_reference_boundary = layer2_tensors.v_tensor->type == 14;
    const bool ffn_q6_boundary =
        iq36::find_tensor(index, LayerTensorName(layer2, "ffn_down_exps.weight"))->type == 14 &&
        iq36::find_tensor(index, LayerTensorName(layer2, "ffn_down_shexp.weight"))->type == 14;
    const bool required_checks_passed =
        load_map.ready &&
        layer0_ok &&
        layer1_ok &&
        state_carry_ok &&
        full_shapes_ok &&
        layer2_payload_counts_ok &&
        metadata_ok &&
        layer2_comparisons_ok &&
        layer2_timing_positive &&
        layer2_arc_selected &&
        v_q6_reference_boundary &&
        ffn_q6_boundary &&
        args.repeat > 0;

    const double layer2_input_sum_min =
        layer2_rms_gpu.timing.rmsnorm_min_us +
        layer2_qk_gpu.timing.qk_projection_kernel_sum_min_us;
    const double layer2_core_output_sum_min =
        core_gate_gpu.timing.core_gate_kernel_sum_min_us +
        attention_gpu.timing.attention_front_kernel_sum_min_us;
    const double layer2_attention_sum_min =
        layer2_input_sum_min + layer2_core_output_sum_min;
    const double two_layer_kernel_sum_min =
        layer0_run.timing.layer_kernel_sum_min_us +
        layer1_run.timing.layer_kernel_sum_min_us;

    std::cout << std::setprecision(10);
    std::cout << "{";
    std::cout << "\"schema_version\":\"intel-qwen36-gpu-resident-layer7-full-attn-core-output-handoff-probe-v0\",";
    std::cout << "\"model_path\":\"" << JsonEscape(args.model_path) << "\",";
    std::cout << "\"layers\":[" << layer0 << "," << layer1 << "," << layer2 << "],";
    std::cout << "\"source_token_position\":" << kFullSourceTokenPosition << ",";
    std::cout << "\"resident_api\":\"two_linear_layer_to_full_attention_core_output_load_once_run_many\",";
    std::cout << "\"resident_load_count\":1,";
    std::cout << "\"resident_shell_invocations\":" << args.repeat << ",";
    std::cout << "\"layer7_residual_input_from_layer6_gpu_output\":true,";
    std::cout << "\"full_attn_qk_projection_gpu_boundary\":true,";
    std::cout << "\"full_attn_core_gpu_boundary\":true,";
    std::cout << "\"full_attn_gate_gpu_boundary\":true,";
    std::cout << "\"full_attn_output_projection_gpu_boundary\":true,";
    std::cout << "\"full_attn_post_norm_gpu_boundary\":true,";
    std::cout << "\"full_attn_v_projection_gpu_supported\":false,";
    std::cout << "\"full_attn_v_projection_boundary\":\"cpu_q6_reference\",";
    std::cout << "\"full_attn_ffn_boundary\":\"q6_down_reference_pending\",";
    std::cout << "\"full_attn_core_input_boundary\":\"captured_rope_kv_payloads\",";
    std::cout << "\"kv_cache_precision_boundary\":\"captured_f32_payload\",";
    std::cout << "\"qk_norm_rope_host_boundary\":true,";
    std::cout << "\"history_kv_state_payload_boundary\":true,";
    std::cout << "\"platform_name\":\"" << JsonEscape(layer2_rms_gpu.platform_name) << "\",";
    std::cout << "\"device_name\":\"" << JsonEscape(layer2_rms_gpu.device_name) << "\",";
    std::cout << "\"core_gate_device_name\":\"" << JsonEscape(core_gate_gpu.device_name) << "\",";
    std::cout << "\"output_projection_device_name\":\"" << JsonEscape(attention_gpu.device_name) << "\",";
    std::cout << "\"layer7_v_tensor_type\":\""
              << JsonEscape(iq36::ggml_type_name(layer2_tensors.v_tensor->type)) << "\",";
    std::cout << "\"thresholds\":{";
    std::cout << "\"strict_component\":{";
    std::cout << "\"max_abs_diff\":0.005,";
    std::cout << "\"min_cosine\":0.99999,";
    std::cout << "\"mismatch_abs_diff\":0.005,";
    std::cout << "\"rmse\":0.001";
    std::cout << "},\"full_attn_component\":{";
    std::cout << "\"max_abs_diff\":" << kFullAttentionComponentMaxAbsDiffThreshold << ",";
    std::cout << "\"min_cosine\":" << kFullAttentionComponentMinCosine << ",";
    std::cout << "\"mismatch_abs_diff\":" << kFullAttentionComponentMismatchThreshold << ",";
    std::cout << "\"rmse\":" << kFullAttentionComponentRmseThreshold;
    std::cout << "}},";
    std::cout << "\"program_build_ms\":"
              << (layer0_run.program_build_ms + layer1_run.program_build_ms +
                  layer2_rms_gpu.program_build_ms + layer2_qk_gpu.program_build_ms +
                  core_gate_gpu.program_build_ms + attention_gpu.program_build_ms)
              << ",";
    std::cout << "\"build_log\":\""
              << JsonEscape(layer0_run.build_log + layer1_run.build_log +
                            layer2_rms_gpu.build_log + layer2_qk_gpu.build_log +
                            core_gate_gpu.build_log + attention_gpu.build_log)
              << "\",";
    std::cout << "\"repeat\":" << args.repeat << ",";
    std::cout << "\"attention_parameters\":{";
    std::cout << "\"attention_scale\":" << kAttentionScale << ",";
    std::cout << "\"full_attention_interval\":" << full_attention_interval << ",";
    std::cout << "\"gqa_group\":" << (q_head_count / kv_head_count) << ",";
    std::cout << "\"head_dim\":" << head_dim << ",";
    std::cout << "\"input_history_token_count\":" << kFullInputHistoryTokenCount << ",";
    std::cout << "\"kv_head_count\":" << kv_head_count << ",";
    std::cout << "\"q_head_count\":" << q_head_count << ",";
    std::cout << "\"updated_history_token_count\":" << kFullUpdatedHistoryTokenCount;
    std::cout << "},\"timings\":{";
    std::cout << "\"layer2_full_attn_input\":";
    WriteFullAttentionTiming(layer2_rms_gpu, layer2_qk_gpu);
    std::cout << ",\"layer2_core_gate\":";
    WriteFullCoreGateTiming(core_gate_gpu.timing);
    std::cout << ",\"layer2_output_front\":";
    WriteFullAttentionOutputFrontTiming(attention_gpu.timing);
    std::cout << ",\"resident_two_linear_layer_kernel_sum_min_us\":"
              << two_layer_kernel_sum_min << ",";
    std::cout << "\"resident_layer7_full_attn_input_kernel_sum_min_us\":"
              << layer2_input_sum_min << ",";
    std::cout << "\"resident_layer7_full_attn_core_output_kernel_sum_min_us\":"
              << layer2_core_output_sum_min << ",";
    std::cout << "\"resident_layer7_full_attention_kernel_sum_min_us\":"
              << layer2_attention_sum_min << ",";
    std::cout << "\"resident_two_linear_plus_layer7_attention_kernel_sum_min_us\":"
              << (two_layer_kernel_sum_min + layer2_attention_sum_min);
    std::cout << "},\"comparisons\":{";
    bool first_compare = true;
    WritePrefixedCompareGroups("l0", layer0_run.comparisons, &first_compare);
    WritePrefixedCompareGroups("l1", layer1_run.comparisons, &first_compare);
    if (!first_compare) {
      std::cout << ",";
    }
    first_compare = false;
    WriteNamedCompareGroups(strict_groups);
    std::cout << ",";
    WriteNamedCompareGroups(full_attention_groups);
    std::cout << ",\"l2_k_raw\":{\"gpu_vs_cpu\":";
    WriteCompare(k_raw_gpu_vs_cpu);
    std::cout << "},\"l2_v\":{\"cpu_vs_oracle\":";
    WriteCompare(native_v_vs_oracle);
    std::cout << "}";
    std::cout << "},\"checks\":{";
    std::cout << "\"load_map_ready\":" << (load_map.ready ? "true" : "false") << ",";
    std::cout << "\"layer0\":";
    WriteLayerChecks(layer0_run);
    std::cout << ",\"layer1\":";
    WriteLayerChecks(layer1_run);
    std::cout << ",\"layer2_shapes\":";
    WriteFullAttentionShapeChecks(full_shapes);
    std::cout << ",\"layer1_residual_input_matches_oracle\":"
              << (state_carry_ok ? "true" : "false") << ",";
    std::cout << "\"layer2_payload_counts_ok\":"
              << (layer2_payload_counts_ok ? "true" : "false") << ",";
    std::cout << "\"metadata_ok\":" << (metadata_ok ? "true" : "false") << ",";
    std::cout << "\"layer2_full_attn_input_matches_oracle\":"
              << (layer2_strict_input_ok ? "true" : "false") << ",";
    std::cout << "\"layer2_full_attn_component_policy_matches_oracle\":"
              << (layer2_full_component_ok ? "true" : "false") << ",";
    std::cout << "\"layer2_full_attn_core_output_matches_oracle\":"
              << (layer2_comparisons_ok ? "true" : "false") << ",";
    std::cout << "\"layer2_v_q6_reference_boundary\":"
              << (v_q6_reference_boundary ? "true" : "false") << ",";
    std::cout << "\"layer2_ffn_q6_boundary\":"
              << (ffn_q6_boundary ? "true" : "false") << ",";
    std::cout << "\"history_kv_state_payloads_present\":"
              << (layer2_payload_counts_ok ? "true" : "false") << ",";
    std::cout << "\"layer2_arc_device_selected\":"
              << (layer2_arc_selected ? "true" : "false") << ",";
    std::cout << "\"resident_load_once\":true,";
    std::cout << "\"resident_shell_invocations_positive\":"
              << (args.repeat > 0 ? "true" : "false") << ",";
    std::cout << "\"gpu_event_timing_positive\":"
              << (layer0_run.timing_positive && layer1_run.timing_positive &&
                  layer2_timing_positive ? "true" : "false") << ",";
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


def opencl_with_full_attn_core_extra(opencl_source: str) -> str:
  required = (
      "__kernel void full_attn_core_f32",
      "__kernel void full_attn_gate_f32",
  )
  if all(item in opencl_source for item in required):
    return opencl_source
  return opencl_source + FULL_ATTN_CORE_EXTRA_OPENCL


def core_output_probe_cpp(opencl_source: str) -> str:
  cpp = L7_INPUT.full_attn_handoff_cpp(
      opencl_with_full_attn_core_extra(opencl_source)
  )
  main_index = cpp.index("\nint main(")
  return cpp[:main_index] + "\n" + FULL_ATTN_CORE_OUTPUT_CPP


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


def add_layer7_tail_payloads(payloads: dict[str, dict[str, Any]]) -> None:
  for name, stage_name, pattern, expected_bytes in (
      ("full_attn_residual", "full_attn_residual.bin", "attn_residual-7__tok15__ord*.bin", 8192),
      ("full_attn_post_norm", "full_attn_post_norm.bin", "attn_post_norm-7__tok15__ord*.bin", 8192),
  ):
    payloads[name] = payload_record(
        L7_INPUT.find_payload(pattern, expected_bytes),
        stage_name,
        expected_bytes,
    )


def resident_fields_ok(probe: dict[str, Any] | None, expected_invocations: int) -> bool:
  return (
      isinstance(probe, dict)
      and probe.get("resident_api") == "two_linear_layer_to_full_attention_core_output_load_once_run_many"
      and probe.get("resident_load_count") == 1
      and probe.get("resident_shell_invocations") == expected_invocations
      and probe.get("layer7_residual_input_from_layer6_gpu_output") is True
      and probe.get("full_attn_core_gpu_boundary") is True
      and probe.get("full_attn_gate_gpu_boundary") is True
      and probe.get("full_attn_output_projection_gpu_boundary") is True
      and probe.get("full_attn_post_norm_gpu_boundary") is True
      and probe.get("full_attn_v_projection_boundary") == "cpu_q6_reference"
      and probe.get("full_attn_ffn_boundary") == "q6_down_reference_pending"
      and PRECONV.nested_bool(probe, "checks", "resident_load_once")
      and PRECONV.nested_bool(probe, "checks", "resident_shell_invocations_positive")
  )


def comparison_passed(
    probe: dict[str, Any] | None,
    name: str,
    lane: str,
    thresholds: dict[str, float] = STRICT_COMPARISON_THRESHOLDS,
) -> bool:
  if not isinstance(probe, dict):
    return False
  comparisons = probe.get("comparisons")
  if not isinstance(comparisons, dict):
    return False
  item = comparisons.get(name)
  if not isinstance(item, dict):
    return False
  stats = item.get(lane)
  return (
      isinstance(stats, dict)
      and stats.get("same_size") is True
      and stats.get("finite") is True
      and stats.get("mismatch_count") == 0
      and stats.get("max_abs_diff", 1.0) <= thresholds["max_abs_diff"]
      and stats.get("rmse", 1.0) <= thresholds["rmse"]
      and stats.get("cosine", 0.0) >= thresholds["min_cosine"]
  )


def full_attention_comparison_passed(
    probe: dict[str, Any] | None,
    name: str,
    lane: str,
) -> bool:
  return comparison_passed(probe, name, lane, FULL_ATTN_COMPARISON_THRESHOLDS)


def write_summary(path: Path, payload: dict[str, Any]) -> None:
  probe = payload.get("probe") if isinstance(payload.get("probe"), dict) else {}
  timings = probe.get("timings", {}) if isinstance(probe, dict) else {}
  comparisons = probe.get("comparisons", {}) if isinstance(probe, dict) else {}
  lines = [
      "# GPU Resident Layer-7 Full-Attention Core/Output Handoff Probe",
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
      "l2_attn_pregate",
      "l2_attn_gated",
      "l2_attn_output",
      "l2_attn_residual",
      "l2_attn_post_norm",
  ):
    group = comparisons.get(name, {}) if isinstance(comparisons, dict) else {}
    lane = group.get("gpu_vs_oracle", {}) if isinstance(group, dict) else {}
    lines.append(f"| {name} | gpu_vs_oracle | {lane.get('max_abs_diff')} | {lane.get('rmse')} |")
  lines += [
      "",
      "| kernel group | min us |",
      "|---|---:|",
      f"| layer7_full_attn_input | {timings.get('resident_layer7_full_attn_input_kernel_sum_min_us') if isinstance(timings, dict) else None} |",
      f"| layer7_core_output | {timings.get('resident_layer7_full_attn_core_output_kernel_sum_min_us') if isinstance(timings, dict) else None} |",
      f"| layer7_attention_total | {timings.get('resident_layer7_full_attention_kernel_sum_min_us') if isinstance(timings, dict) else None} |",
      f"| two_linear_plus_layer7_attention | {timings.get('resident_two_linear_plus_layer7_attention_kernel_sum_min_us') if isinstance(timings, dict) else None} |",
      "",
      "This target-side process carries layer 5 and layer 6 GPU outputs into",
      "layer 7. GPU computes layer-7 attention RMSNorm, Q/K Q4 projections,",
      "full-attention core, gate, output projection, residual add, and",
      "post-attention RMSNorm. V remains a CPU Q6 reference boundary; layer-7",
      "FFN remains a Q6-down boundary. This is captured single-token evidence,",
      "not decode throughput.",
      "",
  ]
  path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  args = parse_args()
  if args.resident_invocations <= 0:
    raise SystemExit("--resident-invocations must be positive")
  if args.layer != 5:
    raise SystemExit("--layer must be 5 for the layer-7 full-attention handoff")

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
  out_dir = args.out_dir or ROOT / f"output/gpu-resident-layer7-full-attn-core-output-handoff-probe-{stamp}"
  out_dir.mkdir(parents=True, exist_ok=False)
  raw_dir = out_dir / "raw"
  raw_dir.mkdir()

  payloads0, conv0 = TWO.prefixed_payloads(layer0, conv0_path, "l0")
  payloads1, conv1 = TWO.prefixed_payloads(layer1, conv1_path, "l1")
  full_payloads, all_history = L7_INPUT.resolve_full_attention_payloads(all_history_json, layer2)
  add_layer7_tail_payloads(full_payloads)
  payloads = {**payloads0, **payloads1, **full_payloads}
  opencl_source = OPENCL_SOURCE.read_text(encoding="utf-8")
  opencl_source_hash = iq36_local.sha256_file(OPENCL_SOURCE)
  embedded_opencl_hash = hashlib.sha256(
      (
          opencl_source
          + PRECONV.POSTCONV.ATTENTION.ATTENTION_EXTRA_OPENCL
          + FULL_ATTN_CORE_EXTRA_OPENCL
      ).encode("utf-8")
  ).hexdigest()
  local_cpp = out_dir / "gpu_resident_layer7_full_attn_core_output_handoff_probe.cpp"
  local_cpp.write_text(core_output_probe_cpp(opencl_source), encoding="utf-8")

  remote_dir = f"{args.remote_root.rstrip('/')}/gpu-resident-layer7-full-attn-core-output-handoff-probe-{stamp}"
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
            f"{remote_dir}/tests/gpu_resident_layer7_full_attn_core_output_handoff_probe.cpp",
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

  executable = f"{remote_dir}/build/iq36-gpu-resident-layer7-full-attn-core-output-handoff-probe"
  compile_cmd = " && ".join([
      f"source {shlex.quote(args.env_script)} >/dev/null 2>&1",
      (
          "g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic "
          f"-I {shlex.quote(remote_dir + '/include')} "
          f"{shlex.quote(remote_dir + '/src/gguf_loader.cpp')} "
          f"{shlex.quote(remote_dir + '/src/gpu_q4x8_matvec.cpp')} "
          f"{shlex.quote(remote_dir + '/tests/gpu_resident_layer7_full_attn_core_output_handoff_probe.cpp')} "
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
      {"name": "layer2_core_output_matches_oracle", "pass": PRECONV.nested_bool(probe, "checks", "layer2_full_attn_core_output_matches_oracle")},
      {"name": "layer2_v_q6_reference_boundary", "pass": PRECONV.nested_bool(probe, "checks", "layer2_v_q6_reference_boundary")},
      {"name": "layer2_ffn_q6_boundary", "pass": PRECONV.nested_bool(probe, "checks", "layer2_ffn_q6_boundary")},
      {"name": "history_kv_state_payloads_present", "pass": PRECONV.nested_bool(probe, "checks", "history_kv_state_payloads_present")},
      {"name": "l2_attn_pregate_matches_oracle", "pass": full_attention_comparison_passed(probe, "l2_attn_pregate", "gpu_vs_oracle")},
      {"name": "l2_attn_gated_matches_oracle", "pass": full_attention_comparison_passed(probe, "l2_attn_gated", "gpu_vs_oracle")},
      {"name": "l2_attn_output_matches_oracle", "pass": full_attention_comparison_passed(probe, "l2_attn_output", "gpu_vs_oracle")},
      {"name": "l2_attn_residual_matches_oracle", "pass": full_attention_comparison_passed(probe, "l2_attn_residual", "gpu_vs_oracle")},
      {"name": "l2_attn_post_norm_matches_oracle", "pass": full_attention_comparison_passed(probe, "l2_attn_post_norm", "gpu_vs_oracle")},
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
      "all_history": all_history,
      "payloads": slim_payloads,
      "layers": [layer0, layer1, layer2],
      "resident_invocations": args.resident_invocations,
      "opencl_source": str(OPENCL_SOURCE.relative_to(ROOT)),
      "opencl_source_sha256": opencl_source_hash,
      "embedded_opencl_source_sha256": embedded_opencl_hash,
      "comparison_thresholds": {
          "strict_component": STRICT_COMPARISON_THRESHOLDS,
          "full_attn_component": FULL_ATTN_COMPARISON_THRESHOLDS,
      },
      "probe_extra_opencl": [
          "rms_norm_hidden_f32",
          "full_attn_core_f32",
          "full_attn_gate_f32",
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
      "tool": "tools/intel-qwen36-gpu-resident-layer7-full-attn-core-output-handoff-probe.py",
      "artifact": str(out_dir),
      "host": args.host,
      "remote_dir": remote_dir,
      "layers": [layer0, layer1, layer2],
      "resident_invocations": args.resident_invocations,
      "conv_history_probes": payload["conv_history_probes"],
      "all_history": all_history,
      "comparison_thresholds": payload["comparison_thresholds"],
      "required_checks_passed": required_checks_passed,
      "speedup_claims_allowed": False,
  }
  correctness = {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "comparison_thresholds": payload["comparison_thresholds"],
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
      "gpu_resident_layer7_full_attn_core_output_handoff_probe",
      [
          ("required_checks_passed", required_checks_passed),
          ("resident_shell_invocations", args.resident_invocations),
          ("resident_layer7_full_attn_core_output_kernel_sum_min_us", PRECONV.nested_number(timings, "resident_layer7_full_attn_core_output_kernel_sum_min_us")),
          ("resident_layer7_full_attention_kernel_sum_min_us", PRECONV.nested_number(timings, "resident_layer7_full_attention_kernel_sum_min_us")),
          ("l2_attn_pregate_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l2_attn_pregate", "gpu_vs_oracle", "max_abs_diff")),
          ("l2_attn_gated_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l2_attn_gated", "gpu_vs_oracle", "max_abs_diff")),
          ("l2_attn_output_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l2_attn_output", "gpu_vs_oracle", "max_abs_diff")),
          ("l2_attn_residual_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l2_attn_residual", "gpu_vs_oracle", "max_abs_diff")),
          ("l2_attn_post_norm_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l2_attn_post_norm", "gpu_vs_oracle", "max_abs_diff")),
      ],
  )
  write_summary(out_dir / "summary.md", payload)
  print(out_dir)
  return 0 if required_checks_passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
