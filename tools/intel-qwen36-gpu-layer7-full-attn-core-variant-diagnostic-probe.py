#!/usr/bin/env python3
"""Diagnose layer-7 full-attention core numeric variants against oracle payloads."""

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
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA_VERSION = "intel-qwen36-gpu-layer7-full-attn-core-variant-diagnostic-probe-v0"
DEFAULT_HOST = "local"
DEFAULT_MODEL = "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf"
DEFAULT_ENV_SCRIPT = "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh"
DEFAULT_REMOTE_ROOT = "/home/intel/intel-qwen36-gpu"
DEFAULT_ALL_HISTORY = ROOT / "output/r1-full-attn-all-history-capture-20260627T145615Z/history.json"
L7_INPUT_TOOL = ROOT / "tools/intel-qwen36-gpu-resident-layer7-full-attn-input-handoff-probe.py"
SOURCE_FILES = [
    ("engine/include/intel_qwen36/gguf_loader.hpp", "include/intel_qwen36/gguf_loader.hpp"),
    ("engine/src/gguf_loader.cpp", "src/gguf_loader.cpp"),
]


def load_l7_input_tool() -> ModuleType:
  spec = importlib.util.spec_from_file_location("iq36_layer7_input_probe", L7_INPUT_TOOL)
  if spec is None or spec.loader is None:
    raise SystemExit(f"failed to load layer7 input handoff tool: {L7_INPUT_TOOL}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


L7_INPUT = load_l7_input_tool()
PRECONV = L7_INPUT.PRECONV


PROBE_CPP = r'''
#include "intel_qwen36/gguf_loader.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

constexpr int kHiddenSize = 2048;
constexpr int kFullQFullValues = 8192;
constexpr int kFullQValues = 4096;
constexpr int kFullKvValues = 512;
constexpr int kFullHeadDim = 256;
constexpr int kFullQHeadCount = 16;
constexpr int kFullKvHeadCount = 2;
constexpr int kFullInputHistoryTokenCount = 15;
constexpr int kFullUpdatedHistoryTokenCount = 16;
constexpr float kAttentionScale = 0.0625f;

struct Args {
  std::string model_path;
  std::string payload_dir;
  int layer = 7;
};

enum class RoundMode {
  kNone,
  kFp16,
  kFp16Even,
  kBf16,
};

struct CoreOptions {
  std::string name;
  RoundMode q_round = RoundMode::kNone;
  RoundMode k_round = RoundMode::kNone;
  RoundMode v_round = RoundMode::kNone;
  bool dot_double = true;
  bool softmax_double = true;
  bool fma_float = false;
  bool online_softmax = false;
  bool f16_accumulator = false;
  bool avx_f16_dot = false;
  bool fp16_even_accumulator = false;
};

struct FullAttentionPayloads {
  std::vector<float> residual_input;
  std::vector<float> q_full;
  std::vector<float> q_rope;
  std::vector<float> k_rope;
  std::vector<float> v;
  std::vector<float> attn_pregate;
  std::vector<float> attn_gated;
  std::vector<float> attn_output;
  std::vector<float> attn_residual;
  std::vector<float> attn_post_norm;
  std::vector<float> ffn_out;
  std::vector<std::vector<float>> k_history;
  std::vector<std::vector<float>> v_history;
};

struct VariantEval {
  CoreOptions options;
  iq36::VectorCompareStats attn_pregate;
  iq36::VectorCompareStats attn_gated;
  iq36::VectorCompareStats attn_output;
  iq36::VectorCompareStats attn_residual;
  iq36::VectorCompareStats attn_post_norm;
  iq36::VectorCompareStats ffn_out;
  iq36::VectorCompareStats layer_output;
  std::vector<std::int32_t> expert_ids;
};

[[noreturn]] void Die(const std::string& message) {
  throw std::runtime_error(message);
}

void Require(bool ok, const std::string& message) {
  if (!ok) {
    Die(message);
  }
}

std::string JoinPath(const std::string& dir, const std::string& name) {
  return (!dir.empty() && dir.back() == '/') ? dir + name : dir + "/" + name;
}

std::string LayerTensorName(int layer, const std::string& suffix) {
  return "blk." + std::to_string(layer) + "." + suffix;
}

Args ParseArgs(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    auto value = [&](const char* name) -> std::string {
      Require(i + 1 < argc, std::string("missing value for ") + name);
      return argv[++i];
    };
    if (key == "--model") args.model_path = value("--model");
    else if (key == "--payload-dir") args.payload_dir = value("--payload-dir");
    else if (key == "--layer") args.layer = std::stoi(value("--layer"));
    else Die("unknown argument: " + key);
  }
  Require(!args.model_path.empty(), "--model is required");
  Require(!args.payload_dir.empty(), "--payload-dir is required");
  Require(args.layer == 7, "--layer must be 7 for this diagnostic");
  return args;
}

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

std::uint32_t FloatBits(float value) {
  std::uint32_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value), "unexpected float size");
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

float BitsFloat(std::uint32_t bits) {
  float value = 0.0f;
  static_assert(sizeof(bits) == sizeof(value), "unexpected float size");
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

std::uint16_t FloatToHalfBits(float value) {
  const std::uint32_t bits = FloatBits(value);
  const std::uint32_t sign = (bits >> 16) & 0x8000U;
  std::int32_t exponent = static_cast<std::int32_t>((bits >> 23) & 0xffU) - 127 + 15;
  std::uint32_t mantissa = bits & 0x7fffffU;
  if (exponent <= 0) {
    if (exponent < -10) {
      return static_cast<std::uint16_t>(sign);
    }
    mantissa |= 0x800000U;
    const std::uint32_t shift = static_cast<std::uint32_t>(14 - exponent);
    std::uint32_t half_mantissa = mantissa >> shift;
    if ((mantissa >> (shift - 1U)) & 1U) {
      ++half_mantissa;
    }
    return static_cast<std::uint16_t>(sign | half_mantissa);
  }
  if (exponent >= 31) {
    return static_cast<std::uint16_t>(sign | 0x7c00U);
  }
  std::uint32_t half = sign | (static_cast<std::uint32_t>(exponent) << 10U) |
                       (mantissa >> 13U);
  if (mantissa & 0x1000U) {
    ++half;
  }
  return static_cast<std::uint16_t>(half);
}

std::uint16_t FloatToHalfBitsEven(float value) {
  const std::uint32_t bits = FloatBits(value);
  const std::uint32_t sign = (bits >> 16U) & 0x8000U;
  const std::uint32_t exponent = (bits >> 23U) & 0xffU;
  std::uint32_t mantissa = bits & 0x7fffffU;
  if (exponent == 0xffU) {
    return static_cast<std::uint16_t>(sign | (mantissa == 0 ? 0x7c00U : 0x7e00U));
  }
  const int half_exponent = static_cast<int>(exponent) - 127 + 15;
  if (half_exponent >= 31) {
    return static_cast<std::uint16_t>(sign | 0x7c00U);
  }
  if (half_exponent <= 0) {
    if (half_exponent < -10) {
      return static_cast<std::uint16_t>(sign);
    }
    mantissa |= 0x800000U;
    const int shift = 14 - half_exponent;
    std::uint32_t half_mantissa = mantissa >> shift;
    const std::uint32_t remainder = mantissa & ((1U << shift) - 1U);
    const std::uint32_t halfway = 1U << (shift - 1);
    if (remainder > halfway || (remainder == halfway && (half_mantissa & 1U) != 0)) {
      ++half_mantissa;
    }
    return static_cast<std::uint16_t>(sign | half_mantissa);
  }
  std::uint32_t half_mantissa = mantissa >> 13U;
  std::uint32_t half_exp_bits = static_cast<std::uint32_t>(half_exponent) << 10U;
  const std::uint32_t remainder = mantissa & 0x1fffU;
  if (remainder > 0x1000U || (remainder == 0x1000U && (half_mantissa & 1U) != 0)) {
    ++half_mantissa;
    if (half_mantissa == 0x400U) {
      half_mantissa = 0;
      half_exp_bits += 0x400U;
    }
  }
  return static_cast<std::uint16_t>(sign | half_exp_bits | half_mantissa);
}

float HalfBitsToFloat(std::uint16_t half) {
  const std::uint32_t sign = (static_cast<std::uint32_t>(half & 0x8000U)) << 16U;
  std::uint32_t exponent = (half >> 10U) & 0x1fU;
  std::uint32_t mantissa = half & 0x03ffU;
  std::uint32_t bits = 0;
  if (exponent == 0) {
    if (mantissa == 0) {
      bits = sign;
    } else {
      exponent = 1;
      while ((mantissa & 0x0400U) == 0) {
        mantissa <<= 1U;
        --exponent;
      }
      mantissa &= 0x03ffU;
      bits = sign | ((exponent + 127U - 15U) << 23U) | (mantissa << 13U);
    }
  } else if (exponent == 31) {
    bits = sign | 0x7f800000U | (mantissa << 13U);
  } else {
    bits = sign | ((exponent + 127U - 15U) << 23U) | (mantissa << 13U);
  }
  return BitsFloat(bits);
}

float RoundToFp16(float value) {
  if (!std::isfinite(value)) {
    return value;
  }
  return HalfBitsToFloat(FloatToHalfBits(value));
}

float RoundToFp16Even(float value) {
  if (!std::isfinite(value)) {
    return value;
  }
  return HalfBitsToFloat(FloatToHalfBitsEven(value));
}

float RoundToFp16WithMode(float value, bool even) {
  return even ? RoundToFp16Even(value) : RoundToFp16(value);
}

float RoundToBf16(float value) {
  if (!std::isfinite(value)) {
    return value;
  }
  std::uint32_t bits = FloatBits(value);
  const std::uint32_t lsb = (bits >> 16U) & 1U;
  bits += 0x7fffU + lsb;
  bits &= 0xffff0000U;
  return BitsFloat(bits);
}

float ApplyRound(float value, RoundMode mode) {
  switch (mode) {
    case RoundMode::kNone:
      return value;
    case RoundMode::kFp16:
      return RoundToFp16(value);
    case RoundMode::kFp16Even:
      return RoundToFp16Even(value);
    case RoundMode::kBf16:
      return RoundToBf16(value);
  }
  return value;
}

std::vector<float> RoundVector(const std::vector<float>& input, RoundMode mode) {
  if (mode == RoundMode::kNone) {
    return input;
  }
  std::vector<float> out;
  out.reserve(input.size());
  for (const float value : input) {
    out.push_back(ApplyRound(value, mode));
  }
  return out;
}

std::vector<std::vector<float>> RoundHistory(const std::vector<std::vector<float>>& input,
                                             RoundMode mode) {
  if (mode == RoundMode::kNone) {
    return input;
  }
  std::vector<std::vector<float>> out;
  out.reserve(input.size());
  for (const auto& item : input) {
    out.push_back(RoundVector(item, mode));
  }
  return out;
}

void ScaleF16Accumulator(std::vector<float>& accumulator, float scale, bool even) {
  for (float& value : accumulator) {
    value = RoundToFp16WithMode(value * scale, even);
  }
}

void MadF16Accumulator(std::vector<float>& accumulator,
                       const std::vector<float>& values,
                       int value_base,
                       float scale,
                       bool even) {
  for (int i = 0; i < kFullHeadDim; ++i) {
    accumulator[i] = RoundToFp16WithMode(
        std::fma(values[value_base + i], scale, accumulator[i]), even);
  }
}

float DotFp16AvxLike(const std::vector<float>& q,
                     int q_base,
                     const std::vector<float>& k,
                     int k_base) {
  float sum[4][8] = {};
  for (int i = 0; i < kFullHeadDim; i += 32) {
    for (int block = 0; block < 4; ++block) {
      for (int lane = 0; lane < 8; ++lane) {
        const int offset = i + block * 8 + lane;
        sum[block][lane] =
            std::fma(q[q_base + offset], k[k_base + offset], sum[block][lane]);
      }
    }
  }
  for (int lane = 0; lane < 8; ++lane) {
    sum[0][lane] += sum[2][lane];
    sum[1][lane] += sum[3][lane];
    sum[0][lane] += sum[1][lane];
  }
  const float t0_0 = sum[0][0] + sum[0][4];
  const float t0_1 = sum[0][1] + sum[0][5];
  const float t0_2 = sum[0][2] + sum[0][6];
  const float t0_3 = sum[0][3] + sum[0][7];
  const float t1_0 = t0_0 + t0_1;
  const float t1_1 = t0_2 + t0_3;
  return t1_0 + t1_1;
}

std::vector<float> RunOnlineCoreVariant(
    const std::vector<float>& q_rope,
    const std::vector<std::vector<float>>& k_history,
    const std::vector<std::vector<float>>& v_history,
    const CoreOptions& options) {
  Require(q_rope.size() == kFullQValues, "q_rope size mismatch");
  Require(k_history.size() == kFullUpdatedHistoryTokenCount, "K history token count mismatch");
  Require(v_history.size() == kFullUpdatedHistoryTokenCount, "V history token count mismatch");
  const auto q = RoundVector(q_rope, options.q_round);
  const auto k = RoundHistory(k_history, options.k_round);
  const auto v = RoundHistory(v_history, options.v_round);
  std::vector<float> out(kFullQValues, 0.0f);
  const int gqa_group = kFullQHeadCount / kFullKvHeadCount;

  for (int q_head = 0; q_head < kFullQHeadCount; ++q_head) {
    const int kv_head = q_head / gqa_group;
    const int q_base = q_head * kFullHeadDim;
    const int kv_base = kv_head * kFullHeadDim;
    float sum = 0.0f;
    float max_score = -std::numeric_limits<float>::infinity();
    std::vector<float> accumulator(kFullHeadDim, 0.0f);
    for (int token = 0; token < kFullUpdatedHistoryTokenCount; ++token) {
      Require(k[token].size() == kFullKvValues && v[token].size() == kFullKvValues,
              "K/V history item size mismatch");
      float dot = 0.0f;
      if (options.avx_f16_dot) {
        dot = DotFp16AvxLike(q, q_base, k[token], kv_base);
      } else {
        for (int i = 0; i < kFullHeadDim; ++i) {
          dot = std::fma(q[q_base + i], k[token][kv_base + i], dot);
        }
      }
      const float score = dot * kAttentionScale;
      const float old_max = max_score;
      float max_scale = 1.0f;
      float value_scale = 1.0f;
      if (score > max_score) {
        max_score = score;
        max_scale = std::exp(old_max - max_score);
        if (options.f16_accumulator) {
          ScaleF16Accumulator(accumulator, max_scale, options.fp16_even_accumulator);
        } else {
          for (float& value : accumulator) {
            value *= max_scale;
          }
        }
      } else {
        value_scale = std::exp(score - max_score);
      }
      if (options.f16_accumulator) {
        MadF16Accumulator(
            accumulator, v[token], kv_base, value_scale, options.fp16_even_accumulator);
      } else {
        for (int i = 0; i < kFullHeadDim; ++i) {
          accumulator[i] = std::fma(v[token][kv_base + i], value_scale, accumulator[i]);
        }
      }
      sum = sum * max_scale + value_scale;
    }
    const float inv_sum = sum == 0.0f ? 0.0f : 1.0f / sum;
    for (int i = 0; i < kFullHeadDim; ++i) {
      out[q_base + i] = accumulator[i] * inv_sum;
    }
  }
  return out;
}

std::vector<float> SoftmaxFloat(const std::vector<float>& logits) {
  Require(!logits.empty(), "softmax input is empty");
  const float max_value = *std::max_element(logits.begin(), logits.end());
  std::vector<float> weights(logits.size(), 0.0f);
  float denom = 0.0f;
  for (std::size_t i = 0; i < logits.size(); ++i) {
    const float weight = std::exp(logits[i] - max_value);
    weights[i] = weight;
    denom += weight;
  }
  Require(denom != 0.0f && std::isfinite(denom), "float softmax denominator is invalid");
  for (float& weight : weights) {
    weight /= denom;
  }
  return weights;
}

std::vector<float> SoftmaxDouble(const std::vector<float>& logits) {
  return iq36::softmax(logits);
}

std::vector<float> RunCoreVariant(const std::vector<float>& q_rope,
                                  const std::vector<std::vector<float>>& k_history,
                                  const std::vector<std::vector<float>>& v_history,
                                  const CoreOptions& options) {
  if (options.online_softmax) {
    return RunOnlineCoreVariant(q_rope, k_history, v_history, options);
  }
  Require(q_rope.size() == kFullQValues, "q_rope size mismatch");
  Require(k_history.size() == kFullUpdatedHistoryTokenCount, "K history token count mismatch");
  Require(v_history.size() == kFullUpdatedHistoryTokenCount, "V history token count mismatch");
  const auto q = RoundVector(q_rope, options.q_round);
  const auto k = RoundHistory(k_history, options.k_round);
  const auto v = RoundHistory(v_history, options.v_round);
  std::vector<float> out(kFullQValues, 0.0f);
  const int gqa_group = kFullQHeadCount / kFullKvHeadCount;
  std::vector<float> scores(kFullUpdatedHistoryTokenCount, 0.0f);

  for (int q_head = 0; q_head < kFullQHeadCount; ++q_head) {
    const int kv_head = q_head / gqa_group;
    const int q_base = q_head * kFullHeadDim;
    const int kv_base = kv_head * kFullHeadDim;
    for (int token = 0; token < kFullUpdatedHistoryTokenCount; ++token) {
      Require(k[token].size() == kFullKvValues && v[token].size() == kFullKvValues,
              "K/V history item size mismatch");
      if (options.dot_double) {
        double dot = 0.0;
        for (int i = 0; i < kFullHeadDim; ++i) {
          dot += static_cast<double>(q[q_base + i]) *
                 static_cast<double>(k[token][kv_base + i]);
        }
        scores[token] = static_cast<float>(dot * static_cast<double>(kAttentionScale));
      } else {
        float dot = 0.0f;
        for (int i = 0; i < kFullHeadDim; ++i) {
          if (options.fma_float) {
            dot = std::fma(q[q_base + i], k[token][kv_base + i], dot);
          } else {
            dot += q[q_base + i] * k[token][kv_base + i];
          }
        }
        scores[token] = dot * kAttentionScale;
      }
    }
    const auto weights =
        options.softmax_double ? SoftmaxDouble(scores) : SoftmaxFloat(scores);
    for (int token = 0; token < kFullUpdatedHistoryTokenCount; ++token) {
      const float weight = weights[token];
      for (int i = 0; i < kFullHeadDim; ++i) {
        float& dst = out[q_base + i];
        const float rhs = v[token][kv_base + i];
        if (options.fma_float) {
          dst = std::fma(weight, rhs, dst);
        } else {
          dst += weight * rhs;
        }
      }
    }
  }
  return out;
}

FullAttentionPayloads LoadPayloads(const std::string& payload_dir) {
  auto f32 = [&](const std::string& name) {
    return iq36::read_f32_vector_file(JoinPath(payload_dir, name));
  };
  FullAttentionPayloads p;
  p.residual_input = f32("full_residual_input.bin");
  p.q_full = f32("full_q_full.bin");
  p.q_rope = f32("full_q_rope.bin");
  p.k_rope = f32("full_k_rope.bin");
  p.v = f32("full_v.bin");
  p.attn_pregate = f32("full_attn_pregate.bin");
  p.attn_gated = f32("full_attn_gated.bin");
  p.attn_output = f32("full_attn_output.bin");
  p.attn_residual = f32("full_attn_residual.bin");
  p.attn_post_norm = f32("full_attn_post_norm.bin");
  p.ffn_out = f32("l7_ffn_out.bin");
  for (int token = 0; token < kFullInputHistoryTokenCount; ++token) {
    std::string prefix = "hist";
    if (token < 10) {
      prefix += "0";
    }
    prefix += std::to_string(token);
    p.k_history.push_back(f32(prefix + "_k_rope.bin"));
    p.v_history.push_back(f32(prefix + "_v.bin"));
  }
  return p;
}

bool PayloadCountsOk(const FullAttentionPayloads& p) {
  return p.residual_input.size() == kHiddenSize &&
         p.q_full.size() == kFullQFullValues &&
         p.q_rope.size() == kFullQValues &&
         p.k_rope.size() == kFullKvValues &&
         p.v.size() == kFullKvValues &&
         p.attn_pregate.size() == kFullQValues &&
         p.attn_gated.size() == kFullQValues &&
         p.attn_output.size() == kHiddenSize &&
         p.attn_residual.size() == kHiddenSize &&
         p.attn_post_norm.size() == kHiddenSize &&
         p.ffn_out.size() == kHiddenSize &&
         p.k_history.size() == kFullInputHistoryTokenCount &&
         p.v_history.size() == kFullInputHistoryTokenCount &&
         std::all_of(p.k_history.begin(), p.k_history.end(), [](const auto& item) {
           return item.size() == kFullKvValues;
         }) &&
         std::all_of(p.v_history.begin(), p.v_history.end(), [](const auto& item) {
           return item.size() == kFullKvValues;
         });
}

std::vector<std::vector<float>> WithCurrent(const std::vector<std::vector<float>>& history,
                                            const std::vector<float>& current) {
  std::vector<std::vector<float>> out = history;
  out.push_back(current);
  return out;
}

VariantEval EvalVariant(const CoreOptions& options,
                        const FullAttentionPayloads& payloads,
                        const iq36::GgufModelIndex& index,
                        const std::string& model_path,
                        int layer,
                        const std::vector<float>& ffn_norm_weight,
                        float rms_norm_epsilon,
                        const std::vector<float>& oracle_layer_output) {
  VariantEval eval;
  eval.options = options;
  const auto k_history = WithCurrent(payloads.k_history, payloads.k_rope);
  const auto v_history = WithCurrent(payloads.v_history, payloads.v);
  const auto pregate = RunCoreVariant(payloads.q_rope, k_history, v_history, options);
  const auto gate = iq36::run_qwen36_full_attention_gate(
      payloads.q_full, pregate, kFullHeadDim);
  const auto attn_output = iq36::matvec_tensor(
      model_path, index, LayerTensorName(layer, "attn_output.weight"), gate.attn_gated);
  const auto attn_residual = iq36::add_vectors(payloads.residual_input, attn_output);
  const auto attn_post_norm =
      iq36::apply_rms_norm(attn_residual, ffn_norm_weight, rms_norm_epsilon);
  const auto ffn = iq36::run_qwen36_moe_ffn_layer(
      model_path, index, layer, attn_residual, rms_norm_epsilon);
  const auto layer_output = iq36::add_vectors(attn_residual, ffn.ffn_out);

  constexpr double kMismatch = 5e-3;
  eval.attn_pregate = iq36::compare_vectors(pregate, payloads.attn_pregate, kMismatch);
  eval.attn_gated = iq36::compare_vectors(gate.attn_gated, payloads.attn_gated, kMismatch);
  eval.attn_output = iq36::compare_vectors(attn_output, payloads.attn_output, kMismatch);
  eval.attn_residual = iq36::compare_vectors(attn_residual, payloads.attn_residual, kMismatch);
  eval.attn_post_norm = iq36::compare_vectors(attn_post_norm, payloads.attn_post_norm, kMismatch);
  eval.ffn_out = iq36::compare_vectors(ffn.ffn_out, payloads.ffn_out, kMismatch);
  eval.layer_output = iq36::compare_vectors(layer_output, oracle_layer_output, kMismatch);
  eval.expert_ids = ffn.router.expert_ids;
  return eval;
}

const char* RoundName(RoundMode mode) {
  switch (mode) {
    case RoundMode::kNone:
      return "none";
    case RoundMode::kFp16:
      return "fp16";
    case RoundMode::kFp16Even:
      return "fp16_even";
    case RoundMode::kBf16:
      return "bf16";
  }
  return "unknown";
}

void WriteCompare(const iq36::VectorCompareStats& stats) {
  std::cout << "{";
  std::cout << "\"compared_value_count\":" << stats.compared_value_count << ",";
  std::cout << "\"cosine\":" << stats.cosine << ",";
  std::cout << "\"finite\":" << (stats.finite ? "true" : "false") << ",";
  std::cout << "\"finite_pair_count\":" << stats.finite_pair_count << ",";
  std::cout << "\"lhs_l2\":" << stats.lhs_l2 << ",";
  std::cout << "\"lhs_value_count\":" << stats.lhs_value_count << ",";
  std::cout << "\"max_abs_diff\":" << stats.max_abs_diff << ",";
  std::cout << "\"mean_abs_diff\":" << stats.mean_abs_diff << ",";
  std::cout << "\"mismatch_count\":" << stats.mismatch_count << ",";
  std::cout << "\"rhs_l2\":" << stats.rhs_l2 << ",";
  std::cout << "\"rhs_value_count\":" << stats.rhs_value_count << ",";
  std::cout << "\"rmse\":" << stats.rmse << ",";
  std::cout << "\"same_size\":" << (stats.same_size ? "true" : "false");
  std::cout << "}";
}

void WriteNamedCompare(const char* name,
                       const iq36::VectorCompareStats& stats,
                       bool& first) {
  if (!first) {
    std::cout << ",";
  }
  first = false;
  std::cout << "\"" << name << "\":";
  WriteCompare(stats);
}

void WriteIntVector(const std::vector<std::int32_t>& values) {
  std::cout << "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0) {
      std::cout << ",";
    }
    std::cout << values[i];
  }
  std::cout << "]";
}

void WriteVariant(const VariantEval& eval) {
  std::cout << "{";
  std::cout << "\"name\":\"" << eval.options.name << "\",";
  std::cout << "\"q_round\":\"" << RoundName(eval.options.q_round) << "\",";
  std::cout << "\"k_round\":\"" << RoundName(eval.options.k_round) << "\",";
  std::cout << "\"v_round\":\"" << RoundName(eval.options.v_round) << "\",";
  std::cout << "\"dot_double\":" << (eval.options.dot_double ? "true" : "false") << ",";
  std::cout << "\"softmax_double\":" << (eval.options.softmax_double ? "true" : "false") << ",";
  std::cout << "\"fma_float\":" << (eval.options.fma_float ? "true" : "false") << ",";
  std::cout << "\"online_softmax\":" << (eval.options.online_softmax ? "true" : "false") << ",";
  std::cout << "\"f16_accumulator\":" << (eval.options.f16_accumulator ? "true" : "false") << ",";
  std::cout << "\"avx_f16_dot\":" << (eval.options.avx_f16_dot ? "true" : "false") << ",";
  std::cout << "\"fp16_even_accumulator\":" << (eval.options.fp16_even_accumulator ? "true" : "false") << ",";
  std::cout << "\"ffn_expert_ids\":";
  WriteIntVector(eval.expert_ids);
  std::cout << ",\"comparisons\":{";
  bool first = true;
  WriteNamedCompare("attn_pregate", eval.attn_pregate, first);
  WriteNamedCompare("attn_gated", eval.attn_gated, first);
  WriteNamedCompare("attn_output", eval.attn_output, first);
  WriteNamedCompare("attn_residual", eval.attn_residual, first);
  WriteNamedCompare("attn_post_norm", eval.attn_post_norm, first);
  WriteNamedCompare("ffn_out", eval.ffn_out, first);
  WriteNamedCompare("layer_output", eval.layer_output, first);
  std::cout << "}}";
}

int main(int argc, char** argv) {
  try {
    const Args args = ParseArgs(argc, argv);
    const auto index = iq36::parse_gguf_model_index(args.model_path);
    const auto payloads = LoadPayloads(args.payload_dir);
    const bool payload_counts_ok = PayloadCountsOk(payloads);
    Require(payload_counts_ok, "payload counts do not match expected layer7 shapes");
    const float rms_norm_epsilon = MetadataFloat(
        index, "qwen35moe.attention.layer_norm_rms_epsilon", 1e-6f);
    const auto ffn_norm_weight = iq36::decode_tensor_row(
        args.model_path, index, LayerTensorName(args.layer, "post_attention_norm.weight"), 0);
    const auto oracle_layer_output = iq36::add_vectors(payloads.attn_residual, payloads.ffn_out);

    iq36::set_resident_tensor_cache_enabled(true);
    iq36::reset_resident_tensor_cache();

    const std::vector<CoreOptions> variants = {
        {"baseline_double_softmax", RoundMode::kNone, RoundMode::kNone, RoundMode::kNone, true, true, false},
        {"f32_fma_softmax_f32", RoundMode::kNone, RoundMode::kNone, RoundMode::kNone, false, false, true},
        {"f32_muladd_softmax_f32", RoundMode::kNone, RoundMode::kNone, RoundMode::kNone, false, false, false},
        {"f32_fma_softmax_double", RoundMode::kNone, RoundMode::kNone, RoundMode::kNone, false, true, true},
        {"fp16_kv_f32_fma_softmax_f32", RoundMode::kNone, RoundMode::kFp16, RoundMode::kFp16, false, false, true},
        {"fp16_kv_double_softmax", RoundMode::kNone, RoundMode::kFp16, RoundMode::kFp16, true, true, false},
        {"fp16_k_only_f32_fma_softmax_f32", RoundMode::kNone, RoundMode::kFp16, RoundMode::kNone, false, false, true},
        {"fp16_v_only_f32_fma_softmax_f32", RoundMode::kNone, RoundMode::kNone, RoundMode::kFp16, false, false, true},
        {"fp16_qkv_f32_fma_softmax_f32", RoundMode::kFp16, RoundMode::kFp16, RoundMode::kFp16, false, false, true},
        {"bf16_kv_f32_fma_softmax_f32", RoundMode::kNone, RoundMode::kBf16, RoundMode::kBf16, false, false, true},
        {"fp16_qkv_online_f32_accum", RoundMode::kFp16, RoundMode::kFp16, RoundMode::kFp16, false, false, true, true, false},
        {"flash_one_chunk_fp16_qkv_f16_accum", RoundMode::kFp16, RoundMode::kFp16, RoundMode::kFp16, false, false, true, true, true},
        {"flash_one_chunk_fp16_qkv_f16_accum_avx_dot", RoundMode::kFp16, RoundMode::kFp16, RoundMode::kFp16, false, false, true, true, true, true},
        {"flash_one_chunk_fp16_even_qkv_f16_accum", RoundMode::kFp16Even, RoundMode::kFp16Even, RoundMode::kFp16Even, false, false, true, true, true, false, true},
        {"flash_one_chunk_fp16_even_qkv_f16_accum_avx_dot", RoundMode::kFp16Even, RoundMode::kFp16Even, RoundMode::kFp16Even, false, false, true, true, true, true, true},
    };
    std::vector<VariantEval> results;
    results.reserve(variants.size());
    for (const auto& variant : variants) {
      results.push_back(EvalVariant(
          variant,
          payloads,
          index,
          args.model_path,
          args.layer,
          ffn_norm_weight,
          rms_norm_epsilon,
          oracle_layer_output));
    }

    const auto cache_stats = iq36::resident_tensor_cache_stats();
    std::cout << std::setprecision(10);
    std::cout << "{";
    std::cout << "\"schema_version\":\"target-layer7-full-attn-core-variant-diagnostic-v0\",";
    std::cout << "\"layer\":" << args.layer << ",";
    std::cout << "\"attention_scale\":" << kAttentionScale << ",";
    std::cout << "\"rms_norm_epsilon\":" << rms_norm_epsilon << ",";
    std::cout << "\"payload_counts_ok\":" << (payload_counts_ok ? "true" : "false") << ",";
    std::cout << "\"input_history_token_count\":" << kFullInputHistoryTokenCount << ",";
    std::cout << "\"updated_history_token_count\":" << kFullUpdatedHistoryTokenCount << ",";
    std::cout << "\"q_head_count\":" << kFullQHeadCount << ",";
    std::cout << "\"kv_head_count\":" << kFullKvHeadCount << ",";
    std::cout << "\"head_dim\":" << kFullHeadDim << ",";
    std::cout << "\"resident_tensor_cache_stats\":{";
    std::cout << "\"enabled\":" << (cache_stats.enabled ? "true" : "false") << ",";
    std::cout << "\"decoded_row_hits\":" << cache_stats.decoded_row_hits << ",";
    std::cout << "\"decoded_row_misses\":" << cache_stats.decoded_row_misses << ",";
    std::cout << "\"decoded_row_cached_values\":" << cache_stats.decoded_row_cached_values << ",";
    std::cout << "\"decoded_row_cached_bytes\":" << cache_stats.decoded_row_cached_bytes << ",";
    std::cout << "\"tensor_payload_hits\":" << cache_stats.tensor_payload_hits << ",";
    std::cout << "\"tensor_payload_misses\":" << cache_stats.tensor_payload_misses << ",";
    std::cout << "\"tensor_payload_cached_bytes\":" << cache_stats.tensor_payload_cached_bytes << ",";
    std::cout << "\"q4_plane_hits\":" << cache_stats.q4_plane_hits << ",";
    std::cout << "\"q4_plane_misses\":" << cache_stats.q4_plane_misses << ",";
    std::cout << "\"q4_plane_cached_bytes\":" << cache_stats.q4_plane_cached_bytes << ",";
    std::cout << "\"q4_plane_repack_ns\":" << cache_stats.q4_plane_repack_ns << ",";
    std::cout << "\"expert_slice_hits\":" << cache_stats.expert_slice_hits << ",";
    std::cout << "\"expert_slice_misses\":" << cache_stats.expert_slice_misses << ",";
    std::cout << "\"expert_slice_cached_bytes\":" << cache_stats.expert_slice_cached_bytes;
    std::cout << "},";
    std::cout << "\"variant_count\":" << results.size() << ",";
    std::cout << "\"variants\":[";
    for (std::size_t i = 0; i < results.size(); ++i) {
      if (i != 0) {
        std::cout << ",";
      }
      WriteVariant(results[i]);
    }
    std::cout << "],";
    std::cout << "\"speedup_claims_allowed\":false";
    std::cout << "}\n";
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "error: " << ex.what() << "\n";
    return 1;
  }
}
'''


def utc_stamp() -> str:
  return dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def iso_now() -> str:
  return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--target", "--host", dest="host", metavar="TARGET", default=DEFAULT_HOST)
  parser.add_argument("--model", default=DEFAULT_MODEL)
  parser.add_argument("--env-script", default=DEFAULT_ENV_SCRIPT)
  parser.add_argument("--staging-root", "--remote-root", dest="remote_root", metavar="PATH", default=DEFAULT_REMOTE_ROOT)
  parser.add_argument("--all-history-json", type=Path, default=DEFAULT_ALL_HISTORY)
  parser.add_argument("--layer", type=int, default=7)
  parser.add_argument("--timeout-s", type=int, default=1200)
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


def add_tail_payloads(payloads: dict[str, dict[str, Any]]) -> None:
  for name, stage_name, pattern, expected_bytes in (
      ("full_attn_residual", "full_attn_residual.bin", "attn_residual-7__tok15__ord*.bin", 8192),
      ("full_attn_post_norm", "full_attn_post_norm.bin", "attn_post_norm-7__tok15__ord*.bin", 8192),
      ("l7_ffn_out", "l7_ffn_out.bin", "ffn_out-7__tok15__ord*.bin", 8192),
  ):
    payloads[name] = payload_record(
        L7_INPUT.find_payload(pattern, expected_bytes),
        stage_name,
        expected_bytes,
    )


def parse_probe_stdout(stdout: str) -> dict[str, Any] | None:
  for line in reversed(stdout.splitlines()):
    line = line.strip()
    if not line.startswith("{") or not line.endswith("}"):
      continue
    value = json.loads(line)
    if isinstance(value, dict):
      return value
  return None


def nested_number(value: Any, *keys: str) -> float | int | None:
  current = value
  for key in keys:
    if not isinstance(current, dict):
      return None
    current = current.get(key)
  return current if isinstance(current, (int, float)) else None


def best_variant(probe: dict[str, Any] | None, comparison_name: str, metric: str = "rmse") -> dict[str, Any] | None:
  if not isinstance(probe, dict):
    return None
  variants = probe.get("variants")
  if not isinstance(variants, list):
    return None
  best: dict[str, Any] | None = None
  best_value: float | None = None
  for variant in variants:
    if not isinstance(variant, dict):
      continue
    value = nested_number(variant, "comparisons", comparison_name, metric)
    if value is None:
      continue
    if best_value is None or float(value) < best_value:
      best = variant
      best_value = float(value)
  if best is None or best_value is None:
    return None
  return {
      "name": best.get("name"),
      "comparison": comparison_name,
      "metric": metric,
      "value": best_value,
      "max_abs_diff": nested_number(best, "comparisons", comparison_name, "max_abs_diff"),
      "rmse": nested_number(best, "comparisons", comparison_name, "rmse"),
  }


def comparison_for(probe: dict[str, Any] | None, variant_name: str, comparison_name: str) -> dict[str, Any] | None:
  if not isinstance(probe, dict):
    return None
  variants = probe.get("variants")
  if not isinstance(variants, list):
    return None
  for variant in variants:
    if isinstance(variant, dict) and variant.get("name") == variant_name:
      comparisons = variant.get("comparisons")
      if isinstance(comparisons, dict) and isinstance(comparisons.get(comparison_name), dict):
        return comparisons[comparison_name]
  return None


def write_summary(path: Path, payload: dict[str, Any]) -> None:
  probe = payload.get("probe") if isinstance(payload.get("probe"), dict) else {}
  best = payload.get("best_variants") if isinstance(payload.get("best_variants"), dict) else {}
  baseline_layer = comparison_for(probe, "baseline_double_softmax", "layer_output")
  baseline_pregate = comparison_for(probe, "baseline_double_softmax", "attn_pregate")
  lines = [
      "# GPU Layer-7 Full-Attention Core Variant Diagnostic",
      "",
      f"- workstream: `{WORKSTREAM}`",
      f"- host: `{payload.get('host')}`",
      f"- model: `{payload.get('model')}`",
      f"- layer: `{payload.get('layer')}`",
      f"- required checks passed: `{str(payload.get('required_checks_passed')).lower()}`",
      "- speedup claims allowed: `false`",
      f"- variant count: `{probe.get('variant_count') if isinstance(probe, dict) else None}`",
      "",
      "| signal | baseline max abs | baseline RMSE | best variant | best max abs | best RMSE |",
      "|---|---:|---:|---|---:|---:|",
      (
          "| attn_pregate | "
          f"{baseline_pregate.get('max_abs_diff') if baseline_pregate else None} | "
          f"{baseline_pregate.get('rmse') if baseline_pregate else None} | "
          f"{best.get('attn_pregate_rmse', {}).get('name') if isinstance(best.get('attn_pregate_rmse'), dict) else None} | "
          f"{best.get('attn_pregate_rmse', {}).get('max_abs_diff') if isinstance(best.get('attn_pregate_rmse'), dict) else None} | "
          f"{best.get('attn_pregate_rmse', {}).get('rmse') if isinstance(best.get('attn_pregate_rmse'), dict) else None} |"
      ),
      (
          "| layer_output | "
          f"{baseline_layer.get('max_abs_diff') if baseline_layer else None} | "
          f"{baseline_layer.get('rmse') if baseline_layer else None} | "
          f"{best.get('layer_output_rmse', {}).get('name') if isinstance(best.get('layer_output_rmse'), dict) else None} | "
          f"{best.get('layer_output_rmse', {}).get('max_abs_diff') if isinstance(best.get('layer_output_rmse'), dict) else None} | "
          f"{best.get('layer_output_rmse', {}).get('rmse') if isinstance(best.get('layer_output_rmse'), dict) else None} |"
      ),
      "",
      "This diagnostic replays captured layer-7 full-attention payloads with",
      "host-side numeric variants. It does not close the layer-8 handoff gate",
      "and does not make a throughput claim.",
      "",
  ]
  path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  args = parse_args()
  if args.layer != 7:
    raise SystemExit("--layer must be 7")

  stamp = utc_stamp()
  created_at = iso_now()
  out_dir = args.out_dir or ROOT / f"output/gpu-layer7-full-attn-core-variant-diagnostic-probe-{stamp}"
  out_dir.mkdir(parents=True, exist_ok=False)
  raw_dir = out_dir / "raw"
  raw_dir.mkdir()

  full_payloads, all_history = L7_INPUT.resolve_full_attention_payloads(
      args.all_history_json.resolve(),
      args.layer,
  )
  add_tail_payloads(full_payloads)
  payloads = full_payloads

  local_cpp = out_dir / "gpu_layer7_full_attn_core_variant_diagnostic_probe.cpp"
  local_cpp.write_text(PROBE_CPP, encoding="utf-8")

  remote_dir = f"{args.remote_root.rstrip('/')}/gpu-layer7-full-attn-core-variant-diagnostic-probe-{stamp}"
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
      transfers.append(
          iq36_local.copy_to(args.host, ROOT / local, f"{remote_dir}/{remote}", args.timeout_s)
      )
    transfers.append(
        iq36_local.copy_to(
            args.host,
            local_cpp,
            f"{remote_dir}/tests/gpu_layer7_full_attn_core_variant_diagnostic_probe.cpp",
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

  executable = f"{remote_dir}/build/iq36-gpu-layer7-full-attn-core-variant-diagnostic-probe"
  stage_ok = (
      setup.get("returncode") == 0
      and transfers
      and all(item.get("returncode") == 0 for item in transfers)
      and all(item.get("returncode") == 0 for item in payload_transfers.values())
  )
  compile_cmd = " && ".join([
      f"source {shlex.quote(args.env_script)} >/dev/null 2>&1",
      (
          "g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic "
          f"-I {shlex.quote(remote_dir + '/include')} "
          f"{shlex.quote(remote_dir + '/src/gguf_loader.cpp')} "
          f"{shlex.quote(remote_dir + '/tests/gpu_layer7_full_attn_core_variant_diagnostic_probe.cpp')} "
          "-pthread "
          f"-o {shlex.quote(executable)}"
      ),
  ])
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
  probe = parse_probe_stdout(run_result.get("stdout", ""))
  best_variants = {
      "attn_pregate_rmse": best_variant(probe, "attn_pregate", "rmse"),
      "attn_output_rmse": best_variant(probe, "attn_output", "rmse"),
      "attn_post_norm_rmse": best_variant(probe, "attn_post_norm", "rmse"),
      "layer_output_rmse": best_variant(probe, "layer_output", "rmse"),
  }

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
      {"name": "payload_counts_ok", "pass": bool(probe and probe.get("payload_counts_ok") is True)},
      {"name": "variant_count_expected", "pass": bool(probe and probe.get("variant_count") == 15)},
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
      "layer": args.layer,
      "all_history": all_history,
      "payloads": slim_payloads,
      "probe": probe,
      "best_variants": best_variants,
      "checks": checks,
      "required_checks_passed": required_checks_passed,
      "speedup_claims_allowed": False,
  }
  manifest = {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "created_at": created_at,
      "tool": "tools/intel-qwen36-gpu-layer7-full-attn-core-variant-diagnostic-probe.py",
      "artifact": str(out_dir),
      "host": args.host,
      "remote_dir": remote_dir,
      "model": args.model,
      "layer": args.layer,
      "all_history": all_history,
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

  baseline_pregate = comparison_for(probe, "baseline_double_softmax", "attn_pregate")
  baseline_layer = comparison_for(probe, "baseline_double_softmax", "layer_output")
  iq36_local.write_metric(
      out_dir / "metrics.jsonl",
      "gpu_layer7_full_attn_core_variant_diagnostic_probe",
      [
          ("required_checks_passed", required_checks_passed),
          ("variant_count", nested_number(probe, "variant_count")),
          ("baseline_attn_pregate_max_abs_diff", baseline_pregate.get("max_abs_diff") if baseline_pregate else None),
          ("baseline_attn_pregate_rmse", baseline_pregate.get("rmse") if baseline_pregate else None),
          ("baseline_layer_output_max_abs_diff", baseline_layer.get("max_abs_diff") if baseline_layer else None),
          ("baseline_layer_output_rmse", baseline_layer.get("rmse") if baseline_layer else None),
          ("best_attn_pregate_rmse", best_variants["attn_pregate_rmse"].get("rmse") if best_variants["attn_pregate_rmse"] else None),
          ("best_layer_output_rmse", best_variants["layer_output_rmse"].get("rmse") if best_variants["layer_output_rmse"] else None),
      ],
  )
  write_summary(out_dir / "summary.md", payload)
  print(out_dir)
  return 0 if required_checks_passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
