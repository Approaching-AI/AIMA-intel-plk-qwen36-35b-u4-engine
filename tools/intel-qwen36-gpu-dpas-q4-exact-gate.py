#!/usr/bin/env python3
"""Run a synthetic Q4/Q8 DPAS integer-dot exactness gate on Arc B390.

This is a route gate for native prefill DPAS/XMX work. It checks that the Intel
OpenCL DPAS mapping used by the sibling kernel corpus compiles, matches scalar
Q4/Q8 integer dot products, and preserves a synthetic full-Q4_K scale/min
multi-token tile. It is not a model benchmark and does not allow speedup claims.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shlex
from pathlib import Path
from typing import Any

import iq36_local


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA_VERSION = "intel-qwen36-gpu-dpas-q4-exact-gate-v2"
DEFAULT_HOST = "local"
DEFAULT_MODEL = "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf"
DEFAULT_ENV_SCRIPT = "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh"
DEFAULT_REMOTE_ROOT = "/home/intel/intel-qwen36-gpu"
PAYLOAD_ROOT = ROOT / "output/r0-boundary-capture-run-20260627T054024Z/remote-output/payloads"
SOURCE_FILES = [
    ("engine/include/intel_qwen36/gguf_loader.hpp", "include/intel_qwen36/gguf_loader.hpp"),
    ("engine/src/gguf_loader.cpp", "src/gguf_loader.cpp"),
]
PAYLOAD_SPECS = {
    "attn_post_norm": ("attn_post_norm.bin", "attn_post_norm-{layer}__tok15__ord*.bin", 8192),
    "ffn_moe_topk": ("ffn_moe_topk.bin", "ffn_moe_topk-{layer}__tok15__ord*.bin", 32),
    "ffn_moe_gate_up": ("ffn_moe_gate_up.bin", "ffn_moe_gate_up-{layer}__tok15__ord*.bin", 32768),
    "ffn_moe_down": ("ffn_moe_down.bin", "ffn_moe_down-{layer}__tok15__ord*.bin", 65536),
    "ffn_shexp": ("ffn_shexp.bin", "ffn_shexp-{layer}__tok15__ord*.bin", 8192),
}


PROBE_CPP = r'''
#include "intel_qwen36/gguf_loader.hpp"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

const char* kOpenClSource = R"CLC(
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable

char q8_k_value(const __global uchar * q8_block, uint index) {
  return (char)q8_block[4U + index];
}

int q4_k_value_for_q8_dot(const __global uchar * qs, uint index) {
  uint segment = index / 64U;
  uint offset = index - segment * 64U;
  uchar packed = qs[segment * 32U + (offset & 31U)];
  return offset < 32U ? (int)(packed & 0x0FU) : (int)(packed >> 4);
}

uint pack_u8x4(uchar x0, uchar x1, uchar x2, uchar x3) {
  return (uint)x0 | ((uint)x1 << 8) | ((uint)x2 << 16) | ((uint)x3 << 24);
}

short pack_q8_pair_dpas(__global const uchar * q8_block, uint k_base) {
  const uchar lo = as_uchar(q8_k_value(q8_block, k_base));
  const uchar hi = as_uchar(q8_k_value(q8_block, k_base + 1U));
  return as_short((ushort)((ushort)lo | ((ushort)hi << 8)));
}

char q8_k_value_local(const __local uchar * q8_block, uint index) {
  return (char)q8_block[4U + index];
}

short q8_k_bsum_local(const __local uchar * q8_block, uint index) {
  ushort bits = (ushort)q8_block[260U + index * 2U] |
                ((ushort)q8_block[261U + index * 2U] << 8);
  return as_short(bits);
}

float load_f32_le_local(const __local uchar * p) {
  uint bits = (uint)p[0] | ((uint)p[1] << 8) |
              ((uint)p[2] << 16) | ((uint)p[3] << 24);
  return as_float(bits);
}

short pack_q8_pair_dpas_local(__local const uchar * q8_block, uint k_base) {
  const uchar lo = as_uchar(q8_k_value_local(q8_block, k_base));
  const uchar hi = as_uchar(q8_k_value_local(q8_block, k_base + 1U));
  return as_short((ushort)((ushort)lo | ((ushort)hi << 8)));
}

uint8 pack_q4_row32_dpas(__global const uchar * qs, uint k_base) {
  return (uint8)(
      pack_u8x4(
          (uchar)q4_k_value_for_q8_dot(qs, k_base + 0U),
          (uchar)q4_k_value_for_q8_dot(qs, k_base + 1U),
          (uchar)q4_k_value_for_q8_dot(qs, k_base + 2U),
          (uchar)q4_k_value_for_q8_dot(qs, k_base + 3U)),
      pack_u8x4(
          (uchar)q4_k_value_for_q8_dot(qs, k_base + 4U),
          (uchar)q4_k_value_for_q8_dot(qs, k_base + 5U),
          (uchar)q4_k_value_for_q8_dot(qs, k_base + 6U),
          (uchar)q4_k_value_for_q8_dot(qs, k_base + 7U)),
      pack_u8x4(
          (uchar)q4_k_value_for_q8_dot(qs, k_base + 8U),
          (uchar)q4_k_value_for_q8_dot(qs, k_base + 9U),
          (uchar)q4_k_value_for_q8_dot(qs, k_base + 10U),
          (uchar)q4_k_value_for_q8_dot(qs, k_base + 11U)),
      pack_u8x4(
          (uchar)q4_k_value_for_q8_dot(qs, k_base + 12U),
          (uchar)q4_k_value_for_q8_dot(qs, k_base + 13U),
          (uchar)q4_k_value_for_q8_dot(qs, k_base + 14U),
          (uchar)q4_k_value_for_q8_dot(qs, k_base + 15U)),
      pack_u8x4(
          (uchar)q4_k_value_for_q8_dot(qs, k_base + 16U),
          (uchar)q4_k_value_for_q8_dot(qs, k_base + 17U),
          (uchar)q4_k_value_for_q8_dot(qs, k_base + 18U),
          (uchar)q4_k_value_for_q8_dot(qs, k_base + 19U)),
      pack_u8x4(
          (uchar)q4_k_value_for_q8_dot(qs, k_base + 20U),
          (uchar)q4_k_value_for_q8_dot(qs, k_base + 21U),
          (uchar)q4_k_value_for_q8_dot(qs, k_base + 22U),
          (uchar)q4_k_value_for_q8_dot(qs, k_base + 23U)),
      pack_u8x4(
          (uchar)q4_k_value_for_q8_dot(qs, k_base + 24U),
          (uchar)q4_k_value_for_q8_dot(qs, k_base + 25U),
          (uchar)q4_k_value_for_q8_dot(qs, k_base + 26U),
          (uchar)q4_k_value_for_q8_dot(qs, k_base + 27U)),
      pack_u8x4(
          (uchar)q4_k_value_for_q8_dot(qs, k_base + 28U),
          (uchar)q4_k_value_for_q8_dot(qs, k_base + 29U),
          (uchar)q4_k_value_for_q8_dot(qs, k_base + 30U),
          (uchar)q4_k_value_for_q8_dot(qs, k_base + 31U)));
}

int q4_kq8_dot32_dpas(
    __global const uchar * qs,
    __global const uchar * q8_block,
    uint k_base) {
  const uint sg_lane = get_sub_group_local_id();
  const short a = pack_q8_pair_dpas(q8_block, k_base + sg_lane * 2U);
  const uint8 b = pack_q4_row32_dpas(qs, k_base);
  return intel_sub_group_i8_u8_matrix_mad_k32(a, b, 0);
}

int q4_kq8_dot32_dpas_prepacked(
    uint8 b,
    __global const uchar * q8_block,
    uint k_base) {
  const uint sg_lane = get_sub_group_local_id();
  const short a = pack_q8_pair_dpas(q8_block, k_base + sg_lane * 2U);
  return intel_sub_group_i8_u8_matrix_mad_k32(a, b, 0);
}

int q4_kq8_dot32_dpas_prepacked_local(
    uint8 b,
    __local const uchar * q8_block,
    uint k_base) {
  const uint sg_lane = get_sub_group_local_id();
  const short a = pack_q8_pair_dpas_local(q8_block, k_base + sg_lane * 2U);
  return intel_sub_group_i8_u8_matrix_mad_k32(a, b, 0);
}

int q4_kq8_dot32_scalar(
    __global const uchar * qs,
    __global const uchar * q8_block,
    uint k_base) {
  int acc = 0;
  for (uint i = 0; i < 32U; ++i) {
    acc += (int)q8_k_value(q8_block, k_base + i) *
           q4_k_value_for_q8_dot(qs, k_base + i);
  }
  return acc;
}

uint load_u16_le(const __global uchar * p) {
  return (uint)p[0] | ((uint)p[1] << 8);
}

short load_i16_le(const __global uchar * p) {
  ushort bits = (ushort)p[0] | ((ushort)p[1] << 8);
  return as_short(bits);
}

float load_f32_le(const __global uchar * p) {
  uint bits = (uint)p[0] | ((uint)p[1] << 8) |
              ((uint)p[2] << 16) | ((uint)p[3] << 24);
  return as_float(bits);
}

void store_u16_le(__global uchar * p, ushort value) {
  p[0] = (uchar)(value & 0xFFU);
  p[1] = (uchar)((value >> 8) & 0xFFU);
}

void store_i16_le(__global uchar * p, short value) {
  store_u16_le(p, as_ushort(value));
}

void store_f32_le(__global uchar * p, float value) {
  const uint bits = as_uint(value);
  p[0] = (uchar)(bits & 0xFFU);
  p[1] = (uchar)((bits >> 8) & 0xFFU);
  p[2] = (uchar)((bits >> 16) & 0xFFU);
  p[3] = (uchar)((bits >> 24) & 0xFFU);
}

float half_to_float(uint h) {
  uint sign = (h & 0x8000U) << 16;
  uint exp = (h >> 10) & 0x1FU;
  uint mantissa = h & 0x03FFU;
  uint out = 0;
  if (exp == 0U) {
    if (mantissa == 0U) {
      out = sign;
    } else {
      uint normalized = mantissa;
      uint shift = 0U;
      while ((normalized & 0x0400U) == 0U) {
        normalized <<= 1;
        shift += 1U;
      }
      normalized &= 0x03FFU;
      out = sign | ((127U - 14U - shift) << 23) | (normalized << 13);
    }
  } else if (exp == 0x1FU) {
    out = sign | 0x7F800000U | (mantissa << 13);
  } else {
    out = sign | ((exp + 112U) << 23) | (mantissa << 13);
  }
  return as_float(out);
}

short q8_k_bsum(const __global uchar * q8_block, uint index) {
  return load_i16_le(q8_block + 260U + index * 2U);
}

uchar get_scale_k4(int j, const __global uchar * q) {
  if (j < 4) {
    return q[j] & 63U;
  }
  return (q[j + 4] & 0x0FU) | ((q[j - 4] >> 6) << 4);
}

uchar get_min_k4(int j, const __global uchar * q) {
  if (j < 4) {
    return q[j + 4] & 63U;
  }
  return (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
}

float q4_kq8_dot_row_dpas(
    __global const uchar * row_raw,
    __global const uchar * q8_input,
    uint blocks_per_row) {
  float sumf = 0.0f;
  for (uint block = 0; block < blocks_per_row; ++block) {
    const __global uchar * q4_block = row_raw + block * 144U;
    const __global uchar * q8_block = q8_input + block * 292U;
    const __global uchar * scales = q4_block + 4U;
    const __global uchar * qs = q4_block + 16U;
    int aux_sum = 0;
    for (uint scale_index = 0; scale_index < 8U; ++scale_index) {
      const int scale = (int)get_scale_k4((int)scale_index, scales);
      aux_sum += scale * q4_kq8_dot32_dpas(qs, q8_block, scale_index * 32U);
    }
    int sumi = 0;
    for (uint bsum_index = 0; bsum_index < 16U; ++bsum_index) {
      const int minimum = (int)get_min_k4((int)(bsum_index >> 1), scales);
      sumi += (int)q8_k_bsum(q8_block, bsum_index) * minimum;
    }
    float q8_d = load_f32_le(q8_block);
    float d = half_to_float(load_u16_le(q4_block)) * q8_d;
    float dmin = half_to_float(load_u16_le(q4_block + 2U)) * q8_d;
    sumf += d * (float)aux_sum - dmin * (float)sumi;
  }
  return sumf;
}

float q4_kq8_dot_row_scalar(
    __global const uchar * row_raw,
    __global const uchar * q8_input,
    uint blocks_per_row) {
  float sumf = 0.0f;
  for (uint block = 0; block < blocks_per_row; ++block) {
    const __global uchar * q4_block = row_raw + block * 144U;
    const __global uchar * q8_block = q8_input + block * 292U;
    const __global uchar * scales = q4_block + 4U;
    const __global uchar * qs = q4_block + 16U;
    int aux_sum = 0;
    for (uint scale_index = 0; scale_index < 8U; ++scale_index) {
      const int scale = (int)get_scale_k4((int)scale_index, scales);
      for (uint i = 0; i < 32U; ++i) {
        const uint k = scale_index * 32U + i;
        aux_sum += scale * (int)q8_k_value(q8_block, k) *
                   q4_k_value_for_q8_dot(qs, k);
      }
    }
    int sumi = 0;
    for (uint bsum_index = 0; bsum_index < 16U; ++bsum_index) {
      const int minimum = (int)get_min_k4((int)(bsum_index >> 1), scales);
      sumi += (int)q8_k_bsum(q8_block, bsum_index) * minimum;
    }
    float q8_d = load_f32_le(q8_block);
    float d = half_to_float(load_u16_le(q4_block)) * q8_d;
    float dmin = half_to_float(load_u16_le(q4_block + 2U)) * q8_d;
    sumf += d * (float)aux_sum - dmin * (float)sumi;
  }
  return sumf;
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void dpas_q4_exact_probe(
    __global const uchar * rows,
    __global const uchar * q8_block,
    __global int * dpas_out,
    __global int * scalar_out) {
  const uint lane = get_sub_group_local_id();
  const __global uchar * qs = rows + lane * 128U;
  for (uint chunk = 0; chunk < 8U; ++chunk) {
    const uint out_index = lane * 8U + chunk;
    const uint k_base = chunk * 32U;
    dpas_out[out_index] = q4_kq8_dot32_dpas(qs, q8_block, k_base);
    scalar_out[out_index] = q4_kq8_dot32_scalar(qs, q8_block, k_base);
  }
}

int nearest_int_opencl(float value) {
  const float shifted = value + 12582912.0f;
  const int bits = as_int(shifted);
  return (bits & 0x007fffff) - 0x00400000;
}

float swiglu_value(float gate, float up) {
  const float sigmoid =
      gate >= 0.0f ? 1.0f / (1.0f + exp(-gate))
                   : exp(gate) / (1.0f + exp(gate));
  return gate * sigmoid * up;
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void dpas_q4k_full_gate(
    __global const uchar * q4_rows,
    __global const uchar * q8_tokens,
    __global float * dpas_out,
    __global float * scalar_out,
    uint blocks_per_row,
    uint row_count) {
  const uint lane = get_sub_group_local_id();
  const uint row_tiles = (row_count + 15U) / 16U;
  const uint group = get_group_id(0);
  const uint token = group / row_tiles;
  const uint row_tile = group - token * row_tiles;
  const uint row = row_tile * 16U + lane;
  if (row >= row_count) {
    return;
  }
  const uint row_bytes = blocks_per_row * 144U;
  const uint q8_token_bytes = blocks_per_row * 292U;
  const __global uchar * row_raw = q4_rows + row * row_bytes;
  const __global uchar * q8_input = q8_tokens + token * q8_token_bytes;
  const uint out_index = token * row_count + row;
  dpas_out[out_index] = q4_kq8_dot_row_dpas(row_raw, q8_input, blocks_per_row);
  scalar_out[out_index] = q4_kq8_dot_row_scalar(row_raw, q8_input, blocks_per_row);
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void dpas_q4k_full_gate_tokenreuse(
    __global const uchar * q4_rows,
    __global const uchar * q8_tokens,
    __global float * dpas_out,
    uint blocks_per_row,
    uint row_count,
    uint token_count,
    uint token_block,
    uint rows_per_q8_group) {
  const uint lane = get_sub_group_local_id();
  const uint row_tiles = (row_count + 15U) / 16U;
  const uint group = get_group_id(0);
  const uint token_tile = group / row_tiles;
  const uint row_tile = group - token_tile * row_tiles;
  const uint token_base = token_tile * token_block;
  const uint row = row_tile * 16U + lane;
  if (row >= row_count || token_block == 0U || token_block > 16U ||
      rows_per_q8_group == 0U) {
    return;
  }
  const uint row_bytes = blocks_per_row * 144U;
  const uint q8_token_bytes = blocks_per_row * 292U;
  const uint q8_group_count =
      (row_count + rows_per_q8_group - 1U) / rows_per_q8_group;
  const uint q8_group = row / rows_per_q8_group;
  const __global uchar * row_raw = q4_rows + row * row_bytes;
  float sumf[16];
  float sum_minf[16];
  for (uint t = 0; t < 16U; ++t) {
    sumf[t] = 0.0f;
    sum_minf[t] = 0.0f;
  }
  for (uint block = 0; block < blocks_per_row; ++block) {
    const __global uchar * q4_block = row_raw + block * 144U;
    const __global uchar * scales = q4_block + 4U;
    const __global uchar * qs = q4_block + 16U;
    const float q4_d = half_to_float(load_u16_le(q4_block));
    const float q4_dmin = half_to_float(load_u16_le(q4_block + 2U));
    for (uint scale_index = 0; scale_index < 8U; ++scale_index) {
      const int scale = (int)get_scale_k4((int)scale_index, scales);
      const uint k_base = scale_index * 32U;
      const uint8 q4_packed = pack_q4_row32_dpas(qs, k_base);
      for (uint t = 0; t < 16U; ++t) {
        const uint token = token_base + t;
        if (t >= token_block || token >= token_count) {
          continue;
        }
        const __global uchar * q8_input =
            q8_tokens + (token * q8_group_count + q8_group) * q8_token_bytes +
            block * 292U;
        const float d = q4_d * load_f32_le(q8_input);
        const int dot = q4_kq8_dot32_dpas_prepacked(q4_packed, q8_input, k_base);
        sumf[t] += d * (float)(scale * dot);
      }
    }
    for (uint bsum_index = 0; bsum_index < 16U; ++bsum_index) {
      const int minimum = (int)get_min_k4((int)(bsum_index >> 1), scales);
      for (uint t = 0; t < 16U; ++t) {
        const uint token = token_base + t;
        if (t >= token_block || token >= token_count) {
          continue;
        }
        const __global uchar * q8_input =
            q8_tokens + (token * q8_group_count + q8_group) * q8_token_bytes +
            block * 292U;
        const float dmin = q4_dmin * load_f32_le(q8_input);
        sum_minf[t] += dmin * (float)((int)q8_k_bsum(q8_input, bsum_index) * minimum);
      }
    }
  }
  for (uint t = 0; t < 16U; ++t) {
    const uint token = token_base + t;
    if (t < token_block && token < token_count) {
      dpas_out[token * row_count + row] = sumf[t] - sum_minf[t];
    }
  }
}

__kernel void q8k_from_gateup_pairs(
    __global const float * gateup,
    __global uchar * q8_tokens,
    uint token_count,
    uint group_count,
    uint intermediate_size) {
  const uint block_count = intermediate_size / 256U;
  if (intermediate_size == 0U || (intermediate_size % 256U) != 0U) {
    return;
  }
  const uint work_item = get_global_id(0);
  const uint block = work_item % block_count;
  const uint group_linear = work_item / block_count;
  const uint group = group_linear % group_count;
  const uint token = group_linear / group_count;
  if (token >= token_count || group >= group_count) {
    return;
  }

  const uint gateup_base =
      token * group_count * intermediate_size * 2U +
      group * intermediate_size * 2U + block * 256U;
  const uint q8_base =
      (token * group_count + group) * block_count * 292U + block * 292U;
  __global uchar * q8_block = q8_tokens + q8_base;

  float max_value = 0.0f;
  float amax = 0.0f;
  for (uint i = 0; i < 256U; ++i) {
    const float gate = gateup[gateup_base + i];
    const float up = gateup[gateup_base + intermediate_size + i];
    const float value = swiglu_value(gate, up);
    const float abs_value = fabs(value);
    if (abs_value > amax) {
      amax = abs_value;
      max_value = value;
    }
  }

  int bsum[16];
  for (uint i = 0; i < 16U; ++i) {
    bsum[i] = 0;
  }

  if (amax == 0.0f) {
    store_f32_le(q8_block, 0.0f);
    for (uint i = 0; i < 256U; ++i) {
      q8_block[4U + i] = 0U;
    }
    for (uint i = 0; i < 16U; ++i) {
      store_i16_le(q8_block + 260U + i * 2U, (short)0);
    }
    return;
  }

  const float iscale = -127.0f / max_value;
  store_f32_le(q8_block, 1.0f / iscale);
  for (uint i = 0; i < 256U; ++i) {
    const float gate = gateup[gateup_base + i];
    const float up = gateup[gateup_base + intermediate_size + i];
    const float value = swiglu_value(gate, up);
    const int rounded = nearest_int_opencl(iscale * value);
    const int quantized = rounded > 127 ? 127 : rounded;
    q8_block[4U + i] = (uchar)((char)quantized);
    bsum[i / 16U] += quantized;
  }
  for (uint i = 0; i < 16U; ++i) {
    store_i16_le(q8_block + 260U + i * 2U, (short)bsum[i]);
  }
}

__kernel void q8k_from_swiglu_values(
    __global const float * swiglu,
    __global uchar * q8_tokens,
    uint token_count,
    uint group_count,
    uint intermediate_size) {
  const uint block_count = intermediate_size / 256U;
  if (intermediate_size == 0U || (intermediate_size % 256U) != 0U) {
    return;
  }
  const uint work_item = get_global_id(0);
  const uint block = work_item % block_count;
  const uint group_linear = work_item / block_count;
  const uint group = group_linear % group_count;
  const uint token = group_linear / group_count;
  if (token >= token_count || group >= group_count) {
    return;
  }

  const uint value_base =
      token * group_count * intermediate_size +
      group * intermediate_size + block * 256U;
  const uint q8_base =
      (token * group_count + group) * block_count * 292U + block * 292U;
  __global uchar * q8_block = q8_tokens + q8_base;

  float max_value = 0.0f;
  float amax = 0.0f;
  for (uint i = 0; i < 256U; ++i) {
    const float value = swiglu[value_base + i];
    const float abs_value = fabs(value);
    if (abs_value > amax) {
      amax = abs_value;
      max_value = value;
    }
  }

  int bsum[16];
  for (uint i = 0; i < 16U; ++i) {
    bsum[i] = 0;
  }

  if (amax == 0.0f) {
    store_f32_le(q8_block, 0.0f);
    for (uint i = 0; i < 256U; ++i) {
      q8_block[4U + i] = 0U;
    }
    for (uint i = 0; i < 16U; ++i) {
      store_i16_le(q8_block + 260U + i * 2U, (short)0);
    }
    return;
  }

  const float iscale = -127.0f / max_value;
  store_f32_le(q8_block, 1.0f / iscale);
  for (uint i = 0; i < 256U; ++i) {
    const float value = swiglu[value_base + i];
    const int rounded = nearest_int_opencl(iscale * value);
    const int quantized = rounded > 127 ? 127 : rounded;
    q8_block[4U + i] = (uchar)((char)quantized);
    bsum[i / 16U] += quantized;
  }
  for (uint i = 0; i < 16U; ++i) {
    store_i16_le(q8_block + 260U + i * 2U, (short)bsum[i]);
  }
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void dpas_q4k_gateup_swiglu_tokenreuse(
    __global const uchar * q4_rows,
    __global const uchar * q8_tokens,
    __global float * swiglu_out,
    uint blocks_per_row,
    uint group_count,
    uint intermediate_size,
    uint token_count,
    uint token_block) {
  const uint lane = get_sub_group_local_id();
  const uint output_rows = group_count * intermediate_size;
  const uint row_tiles = (output_rows + 15U) / 16U;
  const uint group = get_group_id(0);
  const uint token_tile = group / row_tiles;
  const uint row_tile = group - token_tile * row_tiles;
  const uint token_base = token_tile * token_block;
  const uint out_row = row_tile * 16U + lane;
  if (out_row >= output_rows || token_block == 0U || token_block > 16U ||
      intermediate_size == 0U) {
    return;
  }

  const uint row_bytes = blocks_per_row * 144U;
  const uint q8_token_bytes = blocks_per_row * 292U;
  const uint expert = out_row / intermediate_size;
  const uint inner = out_row - expert * intermediate_size;
  const uint rows_per_group = intermediate_size * 2U;
  const uint gate_row = expert * rows_per_group + inner;
  const uint up_row = gate_row + intermediate_size;
  const __global uchar * gate_raw = q4_rows + gate_row * row_bytes;
  const __global uchar * up_raw = q4_rows + up_row * row_bytes;

  float gate_sum[16];
  float gate_min[16];
  float up_sum[16];
  float up_min[16];
  for (uint t = 0; t < 16U; ++t) {
    gate_sum[t] = 0.0f;
    gate_min[t] = 0.0f;
    up_sum[t] = 0.0f;
    up_min[t] = 0.0f;
  }

  for (uint block = 0; block < blocks_per_row; ++block) {
    const __global uchar * gate_block = gate_raw + block * 144U;
    const __global uchar * up_block = up_raw + block * 144U;
    const __global uchar * gate_scales = gate_block + 4U;
    const __global uchar * up_scales = up_block + 4U;
    const __global uchar * gate_qs = gate_block + 16U;
    const __global uchar * up_qs = up_block + 16U;
    const float gate_d = half_to_float(load_u16_le(gate_block));
    const float gate_dmin = half_to_float(load_u16_le(gate_block + 2U));
    const float up_d = half_to_float(load_u16_le(up_block));
    const float up_dmin = half_to_float(load_u16_le(up_block + 2U));

    for (uint scale_index = 0; scale_index < 8U; ++scale_index) {
      const int gate_scale = (int)get_scale_k4((int)scale_index, gate_scales);
      const int up_scale = (int)get_scale_k4((int)scale_index, up_scales);
      const uint k_base = scale_index * 32U;
      const uint8 gate_packed = pack_q4_row32_dpas(gate_qs, k_base);
      const uint8 up_packed = pack_q4_row32_dpas(up_qs, k_base);
      for (uint t = 0; t < 16U; ++t) {
        const uint token = token_base + t;
        if (t >= token_block || token >= token_count) {
          continue;
        }
        const __global uchar * q8_input =
            q8_tokens + token * q8_token_bytes + block * 292U;
        const float q8_d = load_f32_le(q8_input);
        const int gate_dot =
            q4_kq8_dot32_dpas_prepacked(gate_packed, q8_input, k_base);
        const int up_dot =
            q4_kq8_dot32_dpas_prepacked(up_packed, q8_input, k_base);
        gate_sum[t] += gate_d * q8_d * (float)(gate_scale * gate_dot);
        up_sum[t] += up_d * q8_d * (float)(up_scale * up_dot);
      }
    }

    for (uint bsum_index = 0; bsum_index < 16U; ++bsum_index) {
      const int gate_minimum =
          (int)get_min_k4((int)(bsum_index >> 1), gate_scales);
      const int up_minimum =
          (int)get_min_k4((int)(bsum_index >> 1), up_scales);
      for (uint t = 0; t < 16U; ++t) {
        const uint token = token_base + t;
        if (t >= token_block || token >= token_count) {
          continue;
        }
        const __global uchar * q8_input =
            q8_tokens + token * q8_token_bytes + block * 292U;
        const int bsum = (int)q8_k_bsum(q8_input, bsum_index);
        const float q8_d = load_f32_le(q8_input);
        gate_min[t] += gate_dmin * q8_d * (float)(bsum * gate_minimum);
        up_min[t] += up_dmin * q8_d * (float)(bsum * up_minimum);
      }
    }
  }

  for (uint t = 0; t < 16U; ++t) {
    const uint token = token_base + t;
    if (t < token_block && token < token_count) {
      const float gate = gate_sum[t] - gate_min[t];
      const float up = up_sum[t] - up_min[t];
      swiglu_out[token * output_rows + out_row] = swiglu_value(gate, up);
    }
  }
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void dpas_q4k_gateup_swiglu_tokenreuse_localq8(
    __global const uchar * q4_rows,
    __global const uchar * q8_tokens,
    __global float * swiglu_out,
    uint blocks_per_row,
    uint group_count,
    uint intermediate_size,
    uint token_count,
    uint token_block,
    __local uchar * q8_tile) {
  const uint lane = get_sub_group_local_id();
  const uint subgroup = get_sub_group_id();
  const uint subgroup_count = get_num_sub_groups();
  const uint local_id = get_local_id(0);
  const uint local_size = get_local_size(0);
  const uint output_rows = group_count * intermediate_size;
  const uint row_tiles = (output_rows + 15U) / 16U;
  const uint row_tile_groups = (row_tiles + subgroup_count - 1U) / subgroup_count;
  const uint workgroup = get_group_id(0);
  const uint token_tile = workgroup / row_tile_groups;
  const uint row_tile_group = workgroup - token_tile * row_tile_groups;
  const uint token_base = token_tile * token_block;
  const uint row_tile = row_tile_group * subgroup_count + subgroup;
  const uint out_row = row_tile * 16U + lane;
  const uint q8_block_bytes = 292U;
  const bool active =
      out_row < output_rows && token_block > 0U && token_block <= 16U &&
      intermediate_size > 0U && row_tile < row_tiles;

  const uint row_bytes = blocks_per_row * 144U;
  const uint q8_token_bytes = blocks_per_row * q8_block_bytes;
  const uint expert = active ? out_row / intermediate_size : 0U;
  const uint inner = active ? out_row - expert * intermediate_size : 0U;
  const uint rows_per_group = intermediate_size * 2U;
  const uint gate_row = expert * rows_per_group + inner;
  const uint up_row = gate_row + intermediate_size;
  const __global uchar * gate_raw = q4_rows + gate_row * row_bytes;
  const __global uchar * up_raw = q4_rows + up_row * row_bytes;

  float gate_sum[16];
  float gate_min[16];
  float up_sum[16];
  float up_min[16];
  for (uint t = 0; t < 16U; ++t) {
    gate_sum[t] = 0.0f;
    gate_min[t] = 0.0f;
    up_sum[t] = 0.0f;
    up_min[t] = 0.0f;
  }

  for (uint block = 0; block < blocks_per_row; ++block) {
    const uint q8_tile_bytes = token_block * q8_block_bytes;
    for (uint offset = local_id; offset < q8_tile_bytes; offset += local_size) {
      const uint t = offset / q8_block_bytes;
      const uint byte = offset - t * q8_block_bytes;
      const uint token = token_base + t;
      q8_tile[offset] =
          token < token_count
              ? q8_tokens[token * q8_token_bytes + block * q8_block_bytes + byte]
              : (uchar)0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (active) {
      const __global uchar * gate_block = gate_raw + block * 144U;
      const __global uchar * up_block = up_raw + block * 144U;
      const __global uchar * gate_scales = gate_block + 4U;
      const __global uchar * up_scales = up_block + 4U;
      const __global uchar * gate_qs = gate_block + 16U;
      const __global uchar * up_qs = up_block + 16U;
      const float gate_d = half_to_float(load_u16_le(gate_block));
      const float gate_dmin = half_to_float(load_u16_le(gate_block + 2U));
      const float up_d = half_to_float(load_u16_le(up_block));
      const float up_dmin = half_to_float(load_u16_le(up_block + 2U));

      for (uint scale_index = 0; scale_index < 8U; ++scale_index) {
        const int gate_scale = (int)get_scale_k4((int)scale_index, gate_scales);
        const int up_scale = (int)get_scale_k4((int)scale_index, up_scales);
        const uint k_base = scale_index * 32U;
        const uint8 gate_packed = pack_q4_row32_dpas(gate_qs, k_base);
        const uint8 up_packed = pack_q4_row32_dpas(up_qs, k_base);
        for (uint t = 0; t < 16U; ++t) {
          const uint token = token_base + t;
          if (t >= token_block || token >= token_count) {
            continue;
          }
          __local const uchar * q8_input = q8_tile + t * q8_block_bytes;
          const float q8_d = load_f32_le_local(q8_input);
          const int gate_dot =
              q4_kq8_dot32_dpas_prepacked_local(gate_packed, q8_input, k_base);
          const int up_dot =
              q4_kq8_dot32_dpas_prepacked_local(up_packed, q8_input, k_base);
          gate_sum[t] += gate_d * q8_d * (float)(gate_scale * gate_dot);
          up_sum[t] += up_d * q8_d * (float)(up_scale * up_dot);
        }
      }

      for (uint bsum_index = 0; bsum_index < 16U; ++bsum_index) {
        const int gate_minimum =
            (int)get_min_k4((int)(bsum_index >> 1), gate_scales);
        const int up_minimum =
            (int)get_min_k4((int)(bsum_index >> 1), up_scales);
        for (uint t = 0; t < 16U; ++t) {
          const uint token = token_base + t;
          if (t >= token_block || token >= token_count) {
            continue;
          }
          __local const uchar * q8_input = q8_tile + t * q8_block_bytes;
          const int bsum = (int)q8_k_bsum_local(q8_input, bsum_index);
          const float q8_d = load_f32_le_local(q8_input);
          gate_min[t] += gate_dmin * q8_d * (float)(bsum * gate_minimum);
          up_min[t] += up_dmin * q8_d * (float)(bsum * up_minimum);
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (active) {
    for (uint t = 0; t < 16U; ++t) {
      const uint token = token_base + t;
      if (t < token_block && token < token_count) {
        const float gate = gate_sum[t] - gate_min[t];
        const float up = up_sum[t] - up_min[t];
        swiglu_out[token * output_rows + out_row] = swiglu_value(gate, up);
      }
    }
  }
}
)CLC";

using cl_int = std::int32_t;
using cl_uint = std::uint32_t;
using cl_ulong = std::uint64_t;
using cl_bool = cl_uint;
using cl_bitfield = cl_ulong;
using cl_device_type = cl_bitfield;
using cl_platform_info = cl_uint;
using cl_device_info = cl_uint;
using cl_context_properties = intptr_t;
using cl_command_queue_properties = cl_bitfield;
using cl_mem_flags = cl_bitfield;
using cl_program_build_info = cl_uint;
using cl_profiling_info = cl_uint;
using cl_platform_id = struct _cl_platform_id*;
using cl_device_id = struct _cl_device_id*;
using cl_context = struct _cl_context*;
using cl_command_queue = struct _cl_command_queue*;
using cl_mem = struct _cl_mem*;
using cl_program = struct _cl_program*;
using cl_kernel = struct _cl_kernel*;
using cl_event = struct _cl_event*;

constexpr cl_int kClSuccess = 0;
constexpr cl_bool kClTrue = 1;
constexpr cl_device_type kClDeviceTypeGpu = 1ULL << 2;
constexpr cl_mem_flags kClMemReadOnly = 1ULL << 2;
constexpr cl_mem_flags kClMemReadWrite = 1ULL << 0;
constexpr cl_mem_flags kClMemWriteOnly = 1ULL << 1;
constexpr cl_command_queue_properties kClQueueProfilingEnable = 1ULL << 1;
constexpr cl_platform_info kClPlatformName = 0x0902;
constexpr cl_device_info kClDeviceName = 0x102B;
constexpr cl_program_build_info kClProgramBuildLog = 0x1183;
constexpr cl_profiling_info kClProfilingCommandStart = 0x1282;
constexpr cl_profiling_info kClProfilingCommandEnd = 0x1283;

constexpr std::size_t kRows = 16;
constexpr std::size_t kQ4BytesPerRow = 128;
constexpr std::size_t kQ8Bytes = 4 + 256 + 32;
constexpr std::size_t kChunks = 8;
constexpr std::size_t kOutputCount = kRows * kChunks;
constexpr std::size_t kQ4KRows = 16;
constexpr std::size_t kQ4KTokens = 4;
constexpr std::size_t kQ4KBlocksPerRow = 2;
constexpr std::size_t kQ4KBlockBytes = 144;
constexpr std::size_t kQ8KBlockBytes = 4 + 256 + 32;
constexpr std::size_t kQ4KRowBytes = kQ4KBlocksPerRow * kQ4KBlockBytes;
constexpr std::size_t kQ4KTokenBytes = kQ4KBlocksPerRow * kQ8KBlockBytes;
constexpr std::size_t kQ4KOutputCount = kQ4KRows * kQ4KTokens;
constexpr float kQ4KTolerance = 0.0f;
constexpr int kLayerCount = 40;
constexpr int kHiddenSize = 2048;
constexpr int kExpertCount = 256;
constexpr int kIntermediateSize = 512;
constexpr int kGateUpRowsPerExpert = kIntermediateSize * 2;
constexpr double kMismatchThreshold = 5e-3;
constexpr double kMaxAbsDiffThreshold = 5e-3;
constexpr double kRmseThreshold = 5e-4;
constexpr double kMinCosine = 0.999;

[[noreturn]] void Die(const std::string& message) {
  throw std::runtime_error(message);
}

void Require(bool ok, const std::string& message) {
  if (!ok) {
    Die(message);
  }
}

void Check(cl_int err, const std::string& where) {
  if (err != kClSuccess) {
    std::ostringstream oss;
    oss << where << " failed with OpenCL error " << err;
    Die(oss.str());
  }
}

std::string JsonEscape(const std::string& value) {
  std::string out;
  out.reserve(value.size() + 8);
  for (const char ch : value) {
    switch (ch) {
      case '\\': out += "\\\\"; break;
      case '"': out += "\\\""; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default: out += ch; break;
    }
  }
  return out;
}

template <typename Fn>
Fn LoadSym(void* lib, const char* name) {
  void* sym = dlsym(lib, name);
  if (!sym) {
    Die(std::string("missing OpenCL symbol: ") + name);
  }
  return reinterpret_cast<Fn>(sym);
}

struct OpenClApi {
  void* lib = nullptr;
  cl_int (*clGetPlatformIDs)(cl_uint, cl_platform_id*, cl_uint*) = nullptr;
  cl_int (*clGetPlatformInfo)(cl_platform_id, cl_platform_info, std::size_t, void*, std::size_t*) = nullptr;
  cl_int (*clGetDeviceIDs)(cl_platform_id, cl_device_type, cl_uint, cl_device_id*, cl_uint*) = nullptr;
  cl_int (*clGetDeviceInfo)(cl_device_id, cl_device_info, std::size_t, void*, std::size_t*) = nullptr;
  cl_context (*clCreateContext)(const cl_context_properties*, cl_uint, const cl_device_id*, void*, void*, cl_int*) = nullptr;
  cl_int (*clReleaseContext)(cl_context) = nullptr;
  cl_command_queue (*clCreateCommandQueue)(cl_context, cl_device_id, cl_command_queue_properties, cl_int*) = nullptr;
  cl_int (*clReleaseCommandQueue)(cl_command_queue) = nullptr;
  cl_mem (*clCreateBuffer)(cl_context, cl_mem_flags, std::size_t, void*, cl_int*) = nullptr;
  cl_int (*clReleaseMemObject)(cl_mem) = nullptr;
  cl_program (*clCreateProgramWithSource)(cl_context, cl_uint, const char**, const std::size_t*, cl_int*) = nullptr;
  cl_int (*clBuildProgram)(cl_program, cl_uint, const cl_device_id*, const char*, void*, void*) = nullptr;
  cl_int (*clGetProgramBuildInfo)(cl_program, cl_device_id, cl_program_build_info, std::size_t, void*, std::size_t*) = nullptr;
  cl_int (*clReleaseProgram)(cl_program) = nullptr;
  cl_kernel (*clCreateKernel)(cl_program, const char*, cl_int*) = nullptr;
  cl_int (*clSetKernelArg)(cl_kernel, cl_uint, std::size_t, const void*) = nullptr;
  cl_int (*clReleaseKernel)(cl_kernel) = nullptr;
  cl_int (*clEnqueueWriteBuffer)(cl_command_queue, cl_mem, cl_bool, std::size_t, std::size_t, const void*, cl_uint, const cl_event*, cl_event*) = nullptr;
  cl_int (*clEnqueueReadBuffer)(cl_command_queue, cl_mem, cl_bool, std::size_t, std::size_t, void*, cl_uint, const cl_event*, cl_event*) = nullptr;
  cl_int (*clEnqueueNDRangeKernel)(cl_command_queue, cl_kernel, cl_uint, const std::size_t*, const std::size_t*, const std::size_t*, cl_uint, const cl_event*, cl_event*) = nullptr;
  cl_int (*clFinish)(cl_command_queue) = nullptr;
  cl_int (*clGetEventProfilingInfo)(cl_event, cl_profiling_info, std::size_t, void*, std::size_t*) = nullptr;
  cl_int (*clReleaseEvent)(cl_event) = nullptr;

  OpenClApi() {
    lib = dlopen("libOpenCL.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!lib) {
      Die(std::string("dlopen libOpenCL.so.1 failed: ") + dlerror());
    }
    clGetPlatformIDs = LoadSym<decltype(clGetPlatformIDs)>(lib, "clGetPlatformIDs");
    clGetPlatformInfo = LoadSym<decltype(clGetPlatformInfo)>(lib, "clGetPlatformInfo");
    clGetDeviceIDs = LoadSym<decltype(clGetDeviceIDs)>(lib, "clGetDeviceIDs");
    clGetDeviceInfo = LoadSym<decltype(clGetDeviceInfo)>(lib, "clGetDeviceInfo");
    clCreateContext = LoadSym<decltype(clCreateContext)>(lib, "clCreateContext");
    clReleaseContext = LoadSym<decltype(clReleaseContext)>(lib, "clReleaseContext");
    clCreateCommandQueue = LoadSym<decltype(clCreateCommandQueue)>(lib, "clCreateCommandQueue");
    clReleaseCommandQueue = LoadSym<decltype(clReleaseCommandQueue)>(lib, "clReleaseCommandQueue");
    clCreateBuffer = LoadSym<decltype(clCreateBuffer)>(lib, "clCreateBuffer");
    clReleaseMemObject = LoadSym<decltype(clReleaseMemObject)>(lib, "clReleaseMemObject");
    clCreateProgramWithSource = LoadSym<decltype(clCreateProgramWithSource)>(lib, "clCreateProgramWithSource");
    clBuildProgram = LoadSym<decltype(clBuildProgram)>(lib, "clBuildProgram");
    clGetProgramBuildInfo = LoadSym<decltype(clGetProgramBuildInfo)>(lib, "clGetProgramBuildInfo");
    clReleaseProgram = LoadSym<decltype(clReleaseProgram)>(lib, "clReleaseProgram");
    clCreateKernel = LoadSym<decltype(clCreateKernel)>(lib, "clCreateKernel");
    clSetKernelArg = LoadSym<decltype(clSetKernelArg)>(lib, "clSetKernelArg");
    clReleaseKernel = LoadSym<decltype(clReleaseKernel)>(lib, "clReleaseKernel");
    clEnqueueWriteBuffer = LoadSym<decltype(clEnqueueWriteBuffer)>(lib, "clEnqueueWriteBuffer");
    clEnqueueReadBuffer = LoadSym<decltype(clEnqueueReadBuffer)>(lib, "clEnqueueReadBuffer");
    clEnqueueNDRangeKernel = LoadSym<decltype(clEnqueueNDRangeKernel)>(lib, "clEnqueueNDRangeKernel");
    clFinish = LoadSym<decltype(clFinish)>(lib, "clFinish");
    clGetEventProfilingInfo = LoadSym<decltype(clGetEventProfilingInfo)>(lib, "clGetEventProfilingInfo");
    clReleaseEvent = LoadSym<decltype(clReleaseEvent)>(lib, "clReleaseEvent");
  }

  ~OpenClApi() {
    if (lib) {
      dlclose(lib);
    }
  }
};

struct Args {
  std::string model_path;
  std::string payload_dir;
  int layer = 5;
  int real_tokens = 4;
  int repeat = 11;
  std::string device_substring = "B390";
};

struct DeviceSelection {
  cl_platform_id platform = nullptr;
  cl_device_id device = nullptr;
  std::string platform_name;
  std::string device_name;
};

std::string InfoString(OpenClApi& api, cl_platform_id platform, cl_platform_info name) {
  std::size_t size = 0;
  Check(api.clGetPlatformInfo(platform, name, 0, nullptr, &size), "clGetPlatformInfo size");
  std::string value(size, '\0');
  Check(api.clGetPlatformInfo(platform, name, size, value.data(), nullptr), "clGetPlatformInfo value");
  while (!value.empty() && value.back() == '\0') {
    value.pop_back();
  }
  return value;
}

std::string InfoString(OpenClApi& api, cl_device_id device, cl_device_info name) {
  std::size_t size = 0;
  Check(api.clGetDeviceInfo(device, name, 0, nullptr, &size), "clGetDeviceInfo size");
  std::string value(size, '\0');
  Check(api.clGetDeviceInfo(device, name, size, value.data(), nullptr), "clGetDeviceInfo value");
  while (!value.empty() && value.back() == '\0') {
    value.pop_back();
  }
  return value;
}

DeviceSelection SelectDevice(OpenClApi& api, const std::string& substring) {
  cl_uint platform_count = 0;
  Check(api.clGetPlatformIDs(0, nullptr, &platform_count), "clGetPlatformIDs count");
  std::vector<cl_platform_id> platforms(platform_count);
  Check(api.clGetPlatformIDs(platform_count, platforms.data(), nullptr), "clGetPlatformIDs");
  for (cl_platform_id platform : platforms) {
    cl_uint device_count = 0;
    const cl_int err = api.clGetDeviceIDs(platform, kClDeviceTypeGpu, 0, nullptr, &device_count);
    if (err != kClSuccess || device_count == 0) {
      continue;
    }
    std::vector<cl_device_id> devices(device_count);
    Check(api.clGetDeviceIDs(platform, kClDeviceTypeGpu, device_count, devices.data(), nullptr),
          "clGetDeviceIDs");
    for (cl_device_id device : devices) {
      const std::string device_name = InfoString(api, device, kClDeviceName);
      if (substring.empty() || device_name.find(substring) != std::string::npos) {
        return {platform, device, InfoString(api, platform, kClPlatformName), device_name};
      }
    }
  }
  Die("no matching OpenCL GPU for substring: " + substring);
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
    else if (key == "--real-tokens") args.real_tokens = std::stoi(value("--real-tokens"));
    else if (key == "--repeat") args.repeat = std::stoi(value("--repeat"));
    else if (key == "--device-substring") args.device_substring = value("--device-substring");
    else Die("unknown argument: " + key);
  }
  Require(!args.model_path.empty(), "--model is required");
  Require(!args.payload_dir.empty(), "--payload-dir is required");
  Require(args.layer >= 0 && args.layer < kLayerCount, "--layer is out of range");
  Require(args.real_tokens > 0 && args.real_tokens <= 64, "--real-tokens must be in [1, 64]");
  Require(args.repeat > 0, "--repeat must be positive");
  return args;
}

std::string JoinPath(const std::string& dir, const std::string& name) {
  return (!dir.empty() && dir.back() == '/') ? dir + name : dir + "/" + name;
}

std::string LayerTensorName(int layer, const std::string& suffix) {
  return "blk." + std::to_string(layer) + "." + suffix;
}

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

std::vector<std::uint8_t> ReadSelectedExpertRaw(
    std::ifstream& model,
    const iq36::GgufTensorInfo& tensor,
    const std::vector<std::int32_t>& expert_ids,
    std::uint64_t rows_per_expert,
    std::uint64_t row_nbytes) {
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
    Require(static_cast<bool>(model), "selected expert slice seek failed");
    model.read(reinterpret_cast<char*>(raw.data() + target_offset),
               static_cast<std::streamsize>(byte_count));
    Require(model.gcount() == static_cast<std::streamsize>(byte_count),
            "selected expert slice read failed");
  }
  return raw;
}

std::vector<std::uint8_t> ReadTensorRaw(std::ifstream& model,
                                        const iq36::GgufTensorInfo& tensor) {
  std::vector<std::uint8_t> raw(static_cast<std::size_t>(tensor.nbytes));
  model.clear();
  model.seekg(static_cast<std::streamoff>(tensor.absolute_offset), std::ios::beg);
  Require(static_cast<bool>(model), "tensor raw seek failed");
  model.read(reinterpret_cast<char*>(raw.data()),
             static_cast<std::streamsize>(raw.size()));
  Require(model.gcount() == static_cast<std::streamsize>(raw.size()),
          "tensor raw read failed");
  return raw;
}

bool ComparePassed(const iq36::VectorCompareStats& stats) {
  return stats.same_size &&
         stats.finite &&
         stats.mismatch_count == 0 &&
         stats.max_abs_diff <= kMaxAbsDiffThreshold &&
         stats.rmse <= kRmseThreshold &&
         stats.cosine >= kMinCosine;
}

void WriteCompare(const iq36::VectorCompareStats& stats) {
  std::cout << "{";
  std::cout << "\"compared_value_count\":" << stats.compared_value_count << ",";
  std::cout << "\"cosine\":" << stats.cosine << ",";
  std::cout << "\"finite\":" << (stats.finite ? "true" : "false") << ",";
  std::cout << "\"max_abs_diff\":" << stats.max_abs_diff << ",";
  std::cout << "\"mean_abs_diff\":" << stats.mean_abs_diff << ",";
  std::cout << "\"mismatch_count\":" << stats.mismatch_count << ",";
  std::cout << "\"rmse\":" << stats.rmse << ",";
  std::cout << "\"same_size\":" << (stats.same_size ? "true" : "false");
  std::cout << "}";
}

void WriteI32Vector(const std::vector<std::int32_t>& values) {
  std::cout << "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0) {
      std::cout << ",";
    }
    std::cout << values[i];
  }
  std::cout << "]";
}

std::vector<float> ConcatFloatVectors(const std::vector<float>& lhs,
                                      const std::vector<float>& rhs) {
  std::vector<float> out;
  out.reserve(lhs.size() + rhs.size());
  out.insert(out.end(), lhs.begin(), lhs.end());
  out.insert(out.end(), rhs.begin(), rhs.end());
  return out;
}

std::vector<std::uint8_t> ConcatRawVectors(const std::vector<std::uint8_t>& lhs,
                                           const std::vector<std::uint8_t>& rhs) {
  std::vector<std::uint8_t> out;
  out.reserve(lhs.size() + rhs.size());
  out.insert(out.end(), lhs.begin(), lhs.end());
  out.insert(out.end(), rhs.begin(), rhs.end());
  return out;
}

void SetQ4Value(std::vector<std::uint8_t>& row, std::size_t index, int value) {
  const std::size_t segment = index / 64;
  const std::size_t offset = index - segment * 64;
  const std::size_t byte_index = segment * 32 + (offset & 31);
  std::uint8_t& packed = row[byte_index];
  const std::uint8_t nibble = static_cast<std::uint8_t>(value & 0x0F);
  if (offset < 32) {
    packed = static_cast<std::uint8_t>((packed & 0xF0U) | nibble);
  } else {
    packed = static_cast<std::uint8_t>((packed & 0x0FU) | (nibble << 4));
  }
}

void SetQ4ValueAt(std::vector<std::uint8_t>& data,
                  std::size_t qs_base,
                  std::size_t index,
                  int value) {
  const std::size_t segment = index / 64;
  const std::size_t offset = index - segment * 64;
  const std::size_t byte_index = qs_base + segment * 32 + (offset & 31);
  std::uint8_t& packed = data[byte_index];
  const std::uint8_t nibble = static_cast<std::uint8_t>(value & 0x0F);
  if (offset < 32) {
    packed = static_cast<std::uint8_t>((packed & 0xF0U) | nibble);
  } else {
    packed = static_cast<std::uint8_t>((packed & 0x0FU) | (nibble << 4));
  }
}

int Q4Value(const std::vector<std::uint8_t>& row, std::size_t index) {
  const std::size_t segment = index / 64;
  const std::size_t offset = index - segment * 64;
  const std::uint8_t packed = row[segment * 32 + (offset & 31)];
  return offset < 32 ? static_cast<int>(packed & 0x0F) : static_cast<int>(packed >> 4);
}

int Q4ValueAt(const std::vector<std::uint8_t>& data, std::size_t qs_base, std::size_t index) {
  const std::size_t segment = index / 64;
  const std::size_t offset = index - segment * 64;
  const std::uint8_t packed = data[qs_base + segment * 32 + (offset & 31)];
  return offset < 32 ? static_cast<int>(packed & 0x0F) : static_cast<int>(packed >> 4);
}

int Q8Value(const std::vector<std::uint8_t>& q8, std::size_t index) {
  return static_cast<int>(static_cast<std::int8_t>(q8[4 + index]));
}

int Q8ValueAt(const std::vector<std::uint8_t>& q8, std::size_t base, std::size_t index) {
  return static_cast<int>(static_cast<std::int8_t>(q8[base + 4 + index]));
}

void StoreU16Le(std::vector<std::uint8_t>& data, std::size_t offset, std::uint16_t value) {
  data[offset] = static_cast<std::uint8_t>(value & 0xFFU);
  data[offset + 1] = static_cast<std::uint8_t>((value >> 8) & 0xFFU);
}

void StoreI16Le(std::vector<std::uint8_t>& data, std::size_t offset, std::int16_t value) {
  const std::uint16_t bits = static_cast<std::uint16_t>(value);
  StoreU16Le(data, offset, bits);
}

void StoreF32Le(std::vector<std::uint8_t>& data, std::size_t offset, float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  data[offset] = static_cast<std::uint8_t>(bits & 0xFFU);
  data[offset + 1] = static_cast<std::uint8_t>((bits >> 8) & 0xFFU);
  data[offset + 2] = static_cast<std::uint8_t>((bits >> 16) & 0xFFU);
  data[offset + 3] = static_cast<std::uint8_t>((bits >> 24) & 0xFFU);
}

std::uint16_t LoadU16Le(const std::vector<std::uint8_t>& data, std::size_t offset) {
  return static_cast<std::uint16_t>(data[offset]) |
         static_cast<std::uint16_t>(static_cast<std::uint16_t>(data[offset + 1]) << 8);
}

std::int16_t LoadI16Le(const std::vector<std::uint8_t>& data, std::size_t offset) {
  const std::uint16_t bits = LoadU16Le(data, offset);
  return static_cast<std::int16_t>(bits);
}

float LoadF32Le(const std::vector<std::uint8_t>& data, std::size_t offset) {
  const std::uint32_t bits =
      static_cast<std::uint32_t>(data[offset]) |
      (static_cast<std::uint32_t>(data[offset + 1]) << 8) |
      (static_cast<std::uint32_t>(data[offset + 2]) << 16) |
      (static_cast<std::uint32_t>(data[offset + 3]) << 24);
  float value = 0.0f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

float HalfToFloat(std::uint16_t h) {
  const std::uint32_t sign = (static_cast<std::uint32_t>(h) & 0x8000U) << 16;
  const std::uint32_t exp = (static_cast<std::uint32_t>(h) >> 10) & 0x1FU;
  std::uint32_t mantissa = static_cast<std::uint32_t>(h) & 0x03FFU;
  std::uint32_t out = 0;
  if (exp == 0U) {
    if (mantissa == 0U) {
      out = sign;
    } else {
      std::uint32_t shift = 0;
      while ((mantissa & 0x0400U) == 0U) {
        mantissa <<= 1;
        shift += 1;
      }
      mantissa &= 0x03FFU;
      out = sign | ((127U - 14U - shift) << 23) | (mantissa << 13);
    }
  } else if (exp == 0x1FU) {
    out = sign | 0x7F800000U | (mantissa << 13);
  } else {
    out = sign | ((exp + 112U) << 23) | (mantissa << 13);
  }
  float value = 0.0f;
  std::memcpy(&value, &out, sizeof(value));
  return value;
}

void EncodeQ4KScales(std::vector<std::uint8_t>& data,
                     std::size_t scale_base,
                     const int scales[8],
                     const int mins[8]) {
  for (int i = 0; i < 12; ++i) {
    data[scale_base + static_cast<std::size_t>(i)] = 0;
  }
  for (int j = 0; j < 4; ++j) {
    data[scale_base + static_cast<std::size_t>(j)] =
        static_cast<std::uint8_t>((scales[j] & 0x3F) | ((scales[j + 4] >> 4) << 6));
    data[scale_base + static_cast<std::size_t>(j + 4)] =
        static_cast<std::uint8_t>((mins[j] & 0x3F) | ((mins[j + 4] >> 4) << 6));
    data[scale_base + static_cast<std::size_t>(j + 8)] =
        static_cast<std::uint8_t>((scales[j + 4] & 0x0F) | ((mins[j + 4] & 0x0F) << 4));
  }
}

int GetScaleK4(const std::vector<std::uint8_t>& data, std::size_t scale_base, int j) {
  if (j < 4) {
    return static_cast<int>(data[scale_base + static_cast<std::size_t>(j)] & 63U);
  }
  return static_cast<int>(
      (data[scale_base + static_cast<std::size_t>(j + 4)] & 0x0FU) |
      ((data[scale_base + static_cast<std::size_t>(j - 4)] >> 6) << 4));
}

int GetMinK4(const std::vector<std::uint8_t>& data, std::size_t scale_base, int j) {
  if (j < 4) {
    return static_cast<int>(data[scale_base + static_cast<std::size_t>(j + 4)] & 63U);
  }
  return static_cast<int>(
      (data[scale_base + static_cast<std::size_t>(j + 4)] >> 4) |
      ((data[scale_base + static_cast<std::size_t>(j)] >> 6) << 4));
}

std::vector<std::uint8_t> BuildQ8() {
  std::vector<std::uint8_t> q8(kQ8Bytes, 0);
  for (std::size_t i = 0; i < 256; ++i) {
    const int value = static_cast<int>((i * 37 + 11) % 255) - 127;
    q8[4 + i] = static_cast<std::uint8_t>(static_cast<std::int8_t>(value));
  }
  return q8;
}

std::vector<std::uint8_t> BuildRows() {
  std::vector<std::uint8_t> rows(kRows * kQ4BytesPerRow, 0);
  for (std::size_t row_index = 0; row_index < kRows; ++row_index) {
    std::vector<std::uint8_t> row(kQ4BytesPerRow, 0);
    for (std::size_t i = 0; i < 256; ++i) {
      const int value = static_cast<int>((row_index * 7 + i * 5 + (i >> 3)) & 0x0F);
      SetQ4Value(row, i, value);
    }
    std::copy(row.begin(), row.end(), rows.begin() + row_index * kQ4BytesPerRow);
  }
  return rows;
}

std::vector<std::uint8_t> BuildQ4KRows() {
  std::vector<std::uint8_t> rows(kQ4KRows * kQ4KRowBytes, 0);
  for (std::size_t row = 0; row < kQ4KRows; ++row) {
    for (std::size_t block = 0; block < kQ4KBlocksPerRow; ++block) {
      const std::size_t base = row * kQ4KRowBytes + block * kQ4KBlockBytes;
      StoreU16Le(rows, base + 0, 0x3C00U);
      StoreU16Le(rows, base + 2, 0x3C00U);
      int scales[8];
      int mins[8];
      for (int i = 0; i < 8; ++i) {
        scales[i] = 1 + static_cast<int>((row * 5 + block * 7 + i * 9) % 63);
        mins[i] = 1 + static_cast<int>((row * 11 + block * 3 + i * 13) % 63);
      }
      EncodeQ4KScales(rows, base + 4, scales, mins);
      for (std::size_t i = 0; i < 256; ++i) {
        const int value = static_cast<int>((row * 7 + block * 17 + i * 5 + (i >> 3)) & 0x0F);
        SetQ4ValueAt(rows, base + 16, i, value);
      }
    }
  }
  return rows;
}

std::vector<std::uint8_t> BuildQ8KTokens() {
  std::vector<std::uint8_t> q8(kQ4KTokens * kQ4KTokenBytes, 0);
  for (std::size_t token = 0; token < kQ4KTokens; ++token) {
    for (std::size_t block = 0; block < kQ4KBlocksPerRow; ++block) {
      const std::size_t base = token * kQ4KTokenBytes + block * kQ8KBlockBytes;
      StoreF32Le(q8, base, 1.0f);
      int bsum[16];
      for (int i = 0; i < 16; ++i) {
        bsum[i] = 0;
      }
      for (std::size_t i = 0; i < 256; ++i) {
        const int value =
            static_cast<int>((token * 41 + block * 13 + i * 37 + 11) % 255) - 127;
        q8[base + 4 + i] = static_cast<std::uint8_t>(static_cast<std::int8_t>(value));
        bsum[i / 16] += value;
      }
      for (std::size_t i = 0; i < 16; ++i) {
        StoreI16Le(q8, base + 260 + i * 2, static_cast<std::int16_t>(bsum[i]));
      }
    }
  }
  return q8;
}

int NearestInt(float value) {
  float shifted = value + 12582912.0f;
  int bits = 0;
  std::memcpy(&bits, &shifted, sizeof(bits));
  return (bits & 0x007fffff) - 0x00400000;
}

std::vector<std::uint8_t> BuildQ8KTokenFromInput(const std::vector<float>& input) {
  Require(input.size() % 256 == 0, "real Q8_K input must be 256-aligned");
  const std::size_t blocks = input.size() / 256;
  std::vector<std::uint8_t> q8(blocks * kQ8KBlockBytes, 0);
  for (std::size_t block = 0; block < blocks; ++block) {
    const auto* block_input = input.data() + block * 256;
    float max = 0.0f;
    float amax = 0.0f;
    for (int i = 0; i < 256; ++i) {
      const float abs_value = std::fabs(block_input[i]);
      if (abs_value > amax) {
        amax = abs_value;
        max = block_input[i];
      }
    }
    const std::size_t base = block * kQ8KBlockBytes;
    if (amax == 0.0f) {
      continue;
    }
    const float iscale = -127.0f / max;
    StoreF32Le(q8, base, 1.0f / iscale);
    int bsum[16];
    for (int i = 0; i < 16; ++i) {
      bsum[i] = 0;
    }
    for (int i = 0; i < 256; ++i) {
      const int quantized = std::min(127, NearestInt(iscale * block_input[i]));
      q8[base + 4 + static_cast<std::size_t>(i)] =
          static_cast<std::uint8_t>(static_cast<std::int8_t>(quantized));
      bsum[i / 16] += quantized;
    }
    for (int i = 0; i < 16; ++i) {
      StoreI16Le(q8, base + 260 + static_cast<std::size_t>(i) * 2,
                 static_cast<std::int16_t>(bsum[i]));
    }
  }
  return q8;
}

std::vector<float> BuildRealTileInput(const std::vector<float>& base, int token) {
  if (token == 0) {
    return base;
  }
  std::vector<float> out(base.size(), 0.0f);
  const std::size_t n = base.size();
  const float scale = 1.0f + 0.0025f * static_cast<float>(token);
  const float mix = 0.001f * static_cast<float>(token);
  const std::size_t offset = static_cast<std::size_t>(token * 37) % n;
  for (std::size_t i = 0; i < n; ++i) {
    const float neighbor = base[(i + offset) % n];
    out[i] = base[i] * scale + neighbor * mix;
  }
  return out;
}

std::vector<std::vector<float>> BuildRealTileInputs(const std::vector<float>& base,
                                                    int token_count) {
  std::vector<std::vector<float>> inputs;
  inputs.reserve(static_cast<std::size_t>(token_count));
  for (int token = 0; token < token_count; ++token) {
    inputs.push_back(BuildRealTileInput(base, token));
  }
  return inputs;
}

std::vector<std::uint8_t> BuildQ8KTokensFromInputs(
    const std::vector<std::vector<float>>& inputs) {
  std::vector<std::uint8_t> q8_tokens;
  for (const auto& input : inputs) {
    const auto q8 = BuildQ8KTokenFromInput(input);
    q8_tokens.insert(q8_tokens.end(), q8.begin(), q8.end());
  }
  return q8_tokens;
}

std::vector<float> SliceFloatVector(const std::vector<float>& values,
                                    std::size_t offset,
                                    std::size_t count) {
  Require(offset <= values.size() && count <= values.size() - offset,
          "float slice out of range");
  return std::vector<float>(values.begin() + static_cast<std::ptrdiff_t>(offset),
                            values.begin() + static_cast<std::ptrdiff_t>(offset + count));
}

std::vector<int> CpuReference(const std::vector<std::uint8_t>& rows,
                              const std::vector<std::uint8_t>& q8) {
  std::vector<int> ref(kOutputCount, 0);
  for (std::size_t row_index = 0; row_index < kRows; ++row_index) {
    std::vector<std::uint8_t> row(
        rows.begin() + row_index * kQ4BytesPerRow,
        rows.begin() + (row_index + 1) * kQ4BytesPerRow);
    for (std::size_t chunk = 0; chunk < kChunks; ++chunk) {
      int acc = 0;
      for (std::size_t i = 0; i < 32; ++i) {
        const std::size_t k = chunk * 32 + i;
        acc += Q8Value(q8, k) * Q4Value(row, k);
      }
      ref[row_index * kChunks + chunk] = acc;
    }
  }
  return ref;
}

std::vector<float> CpuQ4KReference(const std::vector<std::uint8_t>& rows,
                                   const std::vector<std::uint8_t>& q8_tokens) {
  std::vector<float> ref(kQ4KOutputCount, 0.0f);
  for (std::size_t token = 0; token < kQ4KTokens; ++token) {
    for (std::size_t row = 0; row < kQ4KRows; ++row) {
      float sumf = 0.0f;
      for (std::size_t block = 0; block < kQ4KBlocksPerRow; ++block) {
        const std::size_t q4_base = row * kQ4KRowBytes + block * kQ4KBlockBytes;
        const std::size_t q8_base = token * kQ4KTokenBytes + block * kQ8KBlockBytes;
        int aux_sum = 0;
        for (std::size_t scale_index = 0; scale_index < 8; ++scale_index) {
          const int scale = GetScaleK4(rows, q4_base + 4, static_cast<int>(scale_index));
          for (std::size_t i = 0; i < 32; ++i) {
            const std::size_t k = scale_index * 32 + i;
            aux_sum += scale * Q8ValueAt(q8_tokens, q8_base, k) *
                       Q4ValueAt(rows, q4_base + 16, k);
          }
        }
        int sumi = 0;
        for (std::size_t bsum_index = 0; bsum_index < 16; ++bsum_index) {
          const int minimum = GetMinK4(rows, q4_base + 4, static_cast<int>(bsum_index >> 1));
          sumi += static_cast<int>(LoadI16Le(q8_tokens, q8_base + 260 + bsum_index * 2)) *
                  minimum;
        }
        const float q8_d = LoadF32Le(q8_tokens, q8_base);
        const float d = HalfToFloat(LoadU16Le(rows, q4_base)) * q8_d;
        const float dmin = HalfToFloat(LoadU16Le(rows, q4_base + 2)) * q8_d;
        sumf += d * static_cast<float>(aux_sum) - dmin * static_cast<float>(sumi);
      }
      ref[token * kQ4KRows + row] = sumf;
    }
  }
  return ref;
}

std::string BuildLog(OpenClApi& api, cl_program program, cl_device_id device) {
  std::size_t size = 0;
  api.clGetProgramBuildInfo(program, device, kClProgramBuildLog, 0, nullptr, &size);
  std::string value(size, '\0');
  if (size > 0) {
    api.clGetProgramBuildInfo(program, device, kClProgramBuildLog, size, value.data(), nullptr);
  }
  while (!value.empty() && value.back() == '\0') {
    value.pop_back();
  }
  return value;
}

double EventUs(OpenClApi& api, cl_event event) {
  cl_ulong start = 0;
  cl_ulong end = 0;
  Check(api.clGetEventProfilingInfo(event, kClProfilingCommandStart, sizeof(start), &start, nullptr),
        "clGetEventProfilingInfo start");
  Check(api.clGetEventProfilingInfo(event, kClProfilingCommandEnd, sizeof(end), &end, nullptr),
        "clGetEventProfilingInfo end");
  return static_cast<double>(end - start) / 1000.0;
}

struct TokenReuseLaneResult {
  int token_block = 0;
  double kernel_min_us = 0.0;
  double kernel_mean_us = 0.0;
  std::size_t global_work_items = 0;
  iq36::VectorCompareStats dpas_vs_cpu;
  bool dpas_matches_cpu = false;
};

TokenReuseLaneResult RunTokenReuseLane(
    OpenClApi& api,
    cl_command_queue queue,
    cl_kernel kernel,
    cl_mem rows_mem,
    cl_mem q8_mem,
    cl_mem out_mem,
    const std::vector<float>& cpu_ref,
    cl_uint blocks_per_row,
    cl_uint row_count,
    cl_uint token_count,
    cl_uint token_block,
    cl_uint rows_per_q8_group,
    int repeat) {
  Check(api.clSetKernelArg(kernel, 0, sizeof(rows_mem), &rows_mem),
        "clSetKernelArg tokenreuse rows");
  Check(api.clSetKernelArg(kernel, 1, sizeof(q8_mem), &q8_mem),
        "clSetKernelArg tokenreuse q8");
  Check(api.clSetKernelArg(kernel, 2, sizeof(out_mem), &out_mem),
        "clSetKernelArg tokenreuse dpas");
  Check(api.clSetKernelArg(kernel, 3, sizeof(blocks_per_row), &blocks_per_row),
        "clSetKernelArg tokenreuse blocks");
  Check(api.clSetKernelArg(kernel, 4, sizeof(row_count), &row_count),
        "clSetKernelArg tokenreuse row count");
  Check(api.clSetKernelArg(kernel, 5, sizeof(token_count), &token_count),
        "clSetKernelArg tokenreuse token count");
  Check(api.clSetKernelArg(kernel, 6, sizeof(token_block), &token_block),
        "clSetKernelArg tokenreuse token block");
  Check(api.clSetKernelArg(kernel, 7, sizeof(rows_per_q8_group), &rows_per_q8_group),
        "clSetKernelArg tokenreuse rows per q8 group");
  const std::size_t row_tiles = (static_cast<std::size_t>(row_count) + 15) / 16;
  const std::size_t token_tiles =
      (static_cast<std::size_t>(token_count) + static_cast<std::size_t>(token_block) - 1) /
      static_cast<std::size_t>(token_block);
  const std::size_t global = row_tiles * token_tiles * 16;
  const std::size_t local = 16;
  std::vector<double> kernel_us;
  kernel_us.reserve(static_cast<std::size_t>(repeat));
  for (int i = 0; i < repeat; ++i) {
    cl_event event = nullptr;
    Check(api.clEnqueueNDRangeKernel(
              queue, kernel, 1, nullptr, &global, &local, 0, nullptr, &event),
          "clEnqueueNDRangeKernel tokenreuse");
    Check(api.clFinish(queue), "clFinish tokenreuse");
    kernel_us.push_back(EventUs(api, event));
    api.clReleaseEvent(event);
  }
  std::vector<float> dpas(cpu_ref.size(), 0.0f);
  Check(api.clEnqueueReadBuffer(
            queue, out_mem, kClTrue, 0, dpas.size() * sizeof(float), dpas.data(), 0, nullptr, nullptr),
        "clEnqueueReadBuffer tokenreuse dpas");
  TokenReuseLaneResult result;
  result.token_block = static_cast<int>(token_block);
  result.kernel_min_us = *std::min_element(kernel_us.begin(), kernel_us.end());
  double sum_us = 0.0;
  for (const double value : kernel_us) {
    sum_us += value;
  }
  result.kernel_mean_us = sum_us / static_cast<double>(kernel_us.size());
  result.global_work_items = global;
  result.dpas_vs_cpu = iq36::compare_vectors(dpas, cpu_ref, kMismatchThreshold);
  result.dpas_matches_cpu = ComparePassed(result.dpas_vs_cpu);
  return result;
}

struct SwiGluFusionLaneResult {
  int token_block = 0;
  double kernel_min_us = 0.0;
  double kernel_mean_us = 0.0;
  std::size_t global_work_items = 0;
  std::size_t local_work_items = 0;
  iq36::VectorCompareStats output_vs_cpu;
  bool output_matches_cpu = false;
};

SwiGluFusionLaneResult RunSwiGluFusionLane(
    OpenClApi& api,
    cl_command_queue queue,
    cl_kernel kernel,
    cl_mem rows_mem,
    cl_mem q8_mem,
    cl_mem out_mem,
    const std::vector<float>& cpu_ref,
    cl_uint blocks_per_row,
    cl_uint group_count,
    cl_uint intermediate_size,
    cl_uint token_count,
    cl_uint token_block,
    int repeat) {
  Check(api.clSetKernelArg(kernel, 0, sizeof(rows_mem), &rows_mem),
        "clSetKernelArg swiglu fusion rows");
  Check(api.clSetKernelArg(kernel, 1, sizeof(q8_mem), &q8_mem),
        "clSetKernelArg swiglu fusion q8");
  Check(api.clSetKernelArg(kernel, 2, sizeof(out_mem), &out_mem),
        "clSetKernelArg swiglu fusion out");
  Check(api.clSetKernelArg(kernel, 3, sizeof(blocks_per_row), &blocks_per_row),
        "clSetKernelArg swiglu fusion blocks");
  Check(api.clSetKernelArg(kernel, 4, sizeof(group_count), &group_count),
        "clSetKernelArg swiglu fusion group count");
  Check(api.clSetKernelArg(kernel, 5, sizeof(intermediate_size), &intermediate_size),
        "clSetKernelArg swiglu fusion intermediate size");
  Check(api.clSetKernelArg(kernel, 6, sizeof(token_count), &token_count),
        "clSetKernelArg swiglu fusion token count");
  Check(api.clSetKernelArg(kernel, 7, sizeof(token_block), &token_block),
        "clSetKernelArg swiglu fusion token block");
  const std::size_t output_rows =
      static_cast<std::size_t>(group_count) * static_cast<std::size_t>(intermediate_size);
  const std::size_t row_tiles = (output_rows + 15) / 16;
  const std::size_t token_tiles =
      (static_cast<std::size_t>(token_count) + static_cast<std::size_t>(token_block) - 1) /
      static_cast<std::size_t>(token_block);
  const std::size_t global = row_tiles * token_tiles * 16;
  const std::size_t local = 16;
  std::vector<double> kernel_us;
  kernel_us.reserve(static_cast<std::size_t>(repeat));
  for (int i = 0; i < repeat; ++i) {
    cl_event event = nullptr;
    Check(api.clEnqueueNDRangeKernel(
              queue, kernel, 1, nullptr, &global, &local, 0, nullptr, &event),
          "clEnqueueNDRangeKernel swiglu fusion");
    Check(api.clFinish(queue), "clFinish swiglu fusion");
    kernel_us.push_back(EventUs(api, event));
    api.clReleaseEvent(event);
  }
  std::vector<float> actual(cpu_ref.size(), 0.0f);
  Check(api.clEnqueueReadBuffer(
            queue, out_mem, kClTrue, 0, actual.size() * sizeof(float),
            actual.data(), 0, nullptr, nullptr),
        "clEnqueueReadBuffer swiglu fusion");
  SwiGluFusionLaneResult result;
  result.token_block = static_cast<int>(token_block);
  result.kernel_min_us = *std::min_element(kernel_us.begin(), kernel_us.end());
  double sum_us = 0.0;
  for (const double value : kernel_us) {
    sum_us += value;
  }
  result.kernel_mean_us = sum_us / static_cast<double>(kernel_us.size());
  result.global_work_items = global;
  result.local_work_items = local;
  result.output_vs_cpu = iq36::compare_vectors(actual, cpu_ref, kMismatchThreshold);
  result.output_matches_cpu = ComparePassed(result.output_vs_cpu);
  return result;
}

SwiGluFusionLaneResult RunSwiGluFusionLocalQ8Lane(
    OpenClApi& api,
    cl_command_queue queue,
    cl_kernel kernel,
    cl_mem rows_mem,
    cl_mem q8_mem,
    cl_mem out_mem,
    const std::vector<float>& cpu_ref,
    cl_uint blocks_per_row,
    cl_uint group_count,
    cl_uint intermediate_size,
    cl_uint token_count,
    cl_uint token_block,
    std::size_t local,
    int repeat) {
  Check(api.clSetKernelArg(kernel, 0, sizeof(rows_mem), &rows_mem),
        "clSetKernelArg localq8 swiglu fusion rows");
  Check(api.clSetKernelArg(kernel, 1, sizeof(q8_mem), &q8_mem),
        "clSetKernelArg localq8 swiglu fusion q8");
  Check(api.clSetKernelArg(kernel, 2, sizeof(out_mem), &out_mem),
        "clSetKernelArg localq8 swiglu fusion out");
  Check(api.clSetKernelArg(kernel, 3, sizeof(blocks_per_row), &blocks_per_row),
        "clSetKernelArg localq8 swiglu fusion blocks");
  Check(api.clSetKernelArg(kernel, 4, sizeof(group_count), &group_count),
        "clSetKernelArg localq8 swiglu fusion group count");
  Check(api.clSetKernelArg(kernel, 5, sizeof(intermediate_size), &intermediate_size),
        "clSetKernelArg localq8 swiglu fusion intermediate size");
  Check(api.clSetKernelArg(kernel, 6, sizeof(token_count), &token_count),
        "clSetKernelArg localq8 swiglu fusion token count");
  Check(api.clSetKernelArg(kernel, 7, sizeof(token_block), &token_block),
        "clSetKernelArg localq8 swiglu fusion token block");
  const std::size_t local_q8_bytes =
      static_cast<std::size_t>(token_block) * kQ8KBlockBytes;
  Check(api.clSetKernelArg(kernel, 8, local_q8_bytes, nullptr),
        "clSetKernelArg localq8 swiglu fusion local q8 tile");
  Require(local >= 16 && (local % 16) == 0,
          "localq8 swiglu fusion local size must be a positive subgroup multiple");
  const std::size_t output_rows =
      static_cast<std::size_t>(group_count) * static_cast<std::size_t>(intermediate_size);
  const std::size_t row_tiles = (output_rows + 15) / 16;
  const std::size_t subgroup_count = local / 16;
  const std::size_t row_tile_groups =
      (row_tiles + subgroup_count - 1) / subgroup_count;
  const std::size_t token_tiles =
      (static_cast<std::size_t>(token_count) + static_cast<std::size_t>(token_block) - 1) /
      static_cast<std::size_t>(token_block);
  const std::size_t global = row_tile_groups * token_tiles * local;
  std::vector<double> kernel_us;
  kernel_us.reserve(static_cast<std::size_t>(repeat));
  for (int i = 0; i < repeat; ++i) {
    cl_event event = nullptr;
    Check(api.clEnqueueNDRangeKernel(
              queue, kernel, 1, nullptr, &global, &local, 0, nullptr, &event),
          "clEnqueueNDRangeKernel localq8 swiglu fusion");
    Check(api.clFinish(queue), "clFinish localq8 swiglu fusion");
    kernel_us.push_back(EventUs(api, event));
    api.clReleaseEvent(event);
  }
  std::vector<float> actual(cpu_ref.size(), 0.0f);
  Check(api.clEnqueueReadBuffer(
            queue, out_mem, kClTrue, 0, actual.size() * sizeof(float),
            actual.data(), 0, nullptr, nullptr),
        "clEnqueueReadBuffer localq8 swiglu fusion");
  SwiGluFusionLaneResult result;
  result.token_block = static_cast<int>(token_block);
  result.kernel_min_us = *std::min_element(kernel_us.begin(), kernel_us.end());
  double sum_us = 0.0;
  for (const double value : kernel_us) {
    sum_us += value;
  }
  result.kernel_mean_us = sum_us / static_cast<double>(kernel_us.size());
  result.global_work_items = global;
  result.local_work_items = local;
  result.output_vs_cpu = iq36::compare_vectors(actual, cpu_ref, kMismatchThreshold);
  result.output_matches_cpu = ComparePassed(result.output_vs_cpu);
  return result;
}

struct Q8PrepLaneResult {
  double kernel_min_us = 0.0;
  double kernel_mean_us = 0.0;
  std::size_t global_work_items = 0;
  bool q8_matches_cpu = false;
  std::size_t q8_byte_mismatch_count = 0;
  std::int64_t q8_first_mismatch_index = -1;
};

Q8PrepLaneResult RunQ8PrepLane(
    OpenClApi& api,
    cl_command_queue queue,
    cl_kernel kernel,
    cl_mem gateup_mem,
    cl_mem q8_mem,
    const std::vector<std::uint8_t>& expected_q8,
    cl_uint token_count,
    cl_uint group_count,
    cl_uint intermediate_size,
    int repeat) {
  Check(api.clSetKernelArg(kernel, 0, sizeof(gateup_mem), &gateup_mem),
        "clSetKernelArg q8 prep gateup");
  Check(api.clSetKernelArg(kernel, 1, sizeof(q8_mem), &q8_mem),
        "clSetKernelArg q8 prep q8");
  Check(api.clSetKernelArg(kernel, 2, sizeof(token_count), &token_count),
        "clSetKernelArg q8 prep token count");
  Check(api.clSetKernelArg(kernel, 3, sizeof(group_count), &group_count),
        "clSetKernelArg q8 prep group count");
  Check(api.clSetKernelArg(kernel, 4, sizeof(intermediate_size), &intermediate_size),
        "clSetKernelArg q8 prep intermediate size");
  Require(intermediate_size % 256U == 0U, "q8 prep intermediate size must be 256-aligned");
  const std::size_t global =
      static_cast<std::size_t>(token_count) *
      static_cast<std::size_t>(group_count) *
      static_cast<std::size_t>(intermediate_size / 256U);
  Require(global > 0, "q8 prep global size must be positive");
  const std::size_t local = (global % 64U == 0U) ? 64U : 1U;
  std::vector<double> kernel_us;
  kernel_us.reserve(static_cast<std::size_t>(repeat));
  for (int i = 0; i < repeat; ++i) {
    cl_event event = nullptr;
    Check(api.clEnqueueNDRangeKernel(
              queue, kernel, 1, nullptr, &global, &local, 0, nullptr, &event),
          "clEnqueueNDRangeKernel q8 prep");
    Check(api.clFinish(queue), "clFinish q8 prep");
    kernel_us.push_back(EventUs(api, event));
    api.clReleaseEvent(event);
  }
  std::vector<std::uint8_t> actual_q8(expected_q8.size(), 0);
  Check(api.clEnqueueReadBuffer(
            queue, q8_mem, kClTrue, 0, actual_q8.size(), actual_q8.data(),
            0, nullptr, nullptr),
        "clEnqueueReadBuffer q8 prep");

  Q8PrepLaneResult result;
  result.kernel_min_us = *std::min_element(kernel_us.begin(), kernel_us.end());
  double sum_us = 0.0;
  for (const double value : kernel_us) {
    sum_us += value;
  }
  result.kernel_mean_us = sum_us / static_cast<double>(kernel_us.size());
  result.global_work_items = global;
  for (std::size_t i = 0; i < actual_q8.size(); ++i) {
    if (actual_q8[i] != expected_q8[i]) {
      result.q8_byte_mismatch_count += 1;
      if (result.q8_first_mismatch_index < 0) {
        result.q8_first_mismatch_index = static_cast<std::int64_t>(i);
      }
    }
  }
  result.q8_matches_cpu = result.q8_byte_mismatch_count == 0;
  return result;
}

void WriteTokenReuseResults(const std::vector<TokenReuseLaneResult>& results) {
  std::cout << "[";
  for (std::size_t i = 0; i < results.size(); ++i) {
    if (i != 0) {
      std::cout << ",";
    }
    const auto& result = results[i];
    std::cout << "{";
    std::cout << "\"token_block\":" << result.token_block << ",";
    std::cout << "\"global_work_items\":" << result.global_work_items << ",";
    std::cout << "\"kernel_min_us\":" << result.kernel_min_us << ",";
    std::cout << "\"kernel_mean_us\":" << result.kernel_mean_us << ",";
    std::cout << "\"dpas_matches_cpu\":"
              << (result.dpas_matches_cpu ? "true" : "false") << ",";
    std::cout << "\"dpas_vs_cpu\":";
    WriteCompare(result.dpas_vs_cpu);
    std::cout << "}";
  }
  std::cout << "]";
}

void WriteSwiGluFusionResults(const std::vector<SwiGluFusionLaneResult>& results) {
  std::cout << "[";
  for (std::size_t i = 0; i < results.size(); ++i) {
    if (i != 0) {
      std::cout << ",";
    }
    const auto& result = results[i];
    std::cout << "{";
    std::cout << "\"token_block\":" << result.token_block << ",";
    std::cout << "\"global_work_items\":" << result.global_work_items << ",";
    std::cout << "\"local_work_items\":" << result.local_work_items << ",";
    std::cout << "\"kernel_min_us\":" << result.kernel_min_us << ",";
    std::cout << "\"kernel_mean_us\":" << result.kernel_mean_us << ",";
    std::cout << "\"output_matches_cpu\":"
              << (result.output_matches_cpu ? "true" : "false") << ",";
    std::cout << "\"output_vs_cpu\":";
    WriteCompare(result.output_vs_cpu);
    std::cout << "}";
  }
  std::cout << "]";
}

void WriteQ8PrepResult(const Q8PrepLaneResult& result) {
  std::cout << "{";
  std::cout << "\"global_work_items\":" << result.global_work_items << ",";
  std::cout << "\"kernel_min_us\":" << result.kernel_min_us << ",";
  std::cout << "\"kernel_mean_us\":" << result.kernel_mean_us << ",";
  std::cout << "\"q8_matches_cpu\":"
            << (result.q8_matches_cpu ? "true" : "false") << ",";
  std::cout << "\"q8_byte_mismatch_count\":"
            << result.q8_byte_mismatch_count << ",";
  std::cout << "\"q8_first_mismatch_index\":"
            << result.q8_first_mismatch_index;
  std::cout << "}";
}

int main(int argc, char** argv) {
  try {
    const Args args = ParseArgs(argc, argv);
    OpenClApi api;
    const DeviceSelection selected = SelectDevice(api, args.device_substring);
    const std::vector<std::uint8_t> rows = BuildRows();
    const std::vector<std::uint8_t> q8 = BuildQ8();
    const std::vector<int> cpu_ref = CpuReference(rows, q8);
    const std::vector<std::uint8_t> q4k_rows = BuildQ4KRows();
    const std::vector<std::uint8_t> q8k_tokens = BuildQ8KTokens();
    const std::vector<float> q4k_cpu_ref = CpuQ4KReference(q4k_rows, q8k_tokens);
    const auto index = iq36::parse_gguf_model_index(args.model_path);
    const auto load_map = iq36::validate_qwen36_load_map(index);
    const std::string real_tensor_name = LayerTensorName(args.layer, "ffn_gate_up_exps.weight");
    const auto* real_tensor = iq36::find_tensor(index, real_tensor_name);
    Require(real_tensor != nullptr, "real selected gate-up tensor missing");
    const std::string shared_gate_tensor_name = LayerTensorName(args.layer, "ffn_gate_shexp.weight");
    const std::string shared_up_tensor_name = LayerTensorName(args.layer, "ffn_up_shexp.weight");
    const auto* shared_gate_tensor = iq36::find_tensor(index, shared_gate_tensor_name);
    const auto* shared_up_tensor = iq36::find_tensor(index, shared_up_tensor_name);
    Require(shared_gate_tensor != nullptr, "shared gate tensor missing");
    Require(shared_up_tensor != nullptr, "shared up tensor missing");
    const std::string selected_down_tensor_name =
        LayerTensorName(args.layer, "ffn_down_exps.weight");
    const std::string shared_down_tensor_name =
        LayerTensorName(args.layer, "ffn_down_shexp.weight");
    const auto* selected_down_tensor = iq36::find_tensor(index, selected_down_tensor_name);
    const auto* shared_down_tensor = iq36::find_tensor(index, shared_down_tensor_name);
    const bool real_tensor_shape_ok =
        real_tensor->type == 12 &&
        real_tensor->dims == std::vector<std::uint64_t>{
            kHiddenSize, kGateUpRowsPerExpert, kExpertCount};
    const bool shared_tensor_shape_ok =
        shared_gate_tensor->type == 12 &&
        shared_up_tensor->type == 12 &&
        shared_gate_tensor->dims == std::vector<std::uint64_t>{kHiddenSize, kIntermediateSize} &&
        shared_up_tensor->dims == std::vector<std::uint64_t>{kHiddenSize, kIntermediateSize};
    const bool q4_down_available =
        selected_down_tensor != nullptr &&
        shared_down_tensor != nullptr &&
        selected_down_tensor->type == 12 &&
        shared_down_tensor->type == 12 &&
        selected_down_tensor->dims == std::vector<std::uint64_t>{
            kIntermediateSize, kHiddenSize, kExpertCount} &&
        shared_down_tensor->dims == std::vector<std::uint64_t>{
            kIntermediateSize, kHiddenSize};
    const auto real_input =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "attn_post_norm.bin"));
    const auto real_expert_ids = ReadI32VectorFile(JoinPath(args.payload_dir, "ffn_moe_topk.bin"));
    const auto real_oracle =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "ffn_moe_gate_up.bin"));
    const auto selected_down_oracle =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "ffn_moe_down.bin"));
    const auto shared_down_oracle =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "ffn_shexp.bin"));
    const std::uint64_t real_cols = real_tensor->dims[0];
    const std::uint64_t real_rows_per_expert = real_tensor->dims[1];
    const std::uint64_t real_selected_rows = real_rows_per_expert * real_expert_ids.size();
    const std::uint64_t real_blocks_per_row = real_cols / 256;
    const std::uint64_t real_row_nbytes =
        real_tensor->nbytes / (real_tensor->dims[1] * real_tensor->dims[2]);
    const std::uint64_t shared_rows =
        shared_gate_tensor->dims[1] + shared_up_tensor->dims[1];
    const std::uint64_t shared_blocks_per_row = shared_gate_tensor->dims[0] / 256;
    const std::uint64_t shared_gate_row_nbytes =
        shared_gate_tensor->nbytes / shared_gate_tensor->dims[1];
    const std::uint64_t shared_up_row_nbytes =
        shared_up_tensor->nbytes / shared_up_tensor->dims[1];
    const std::uint64_t selected_down_cols =
        q4_down_available ? selected_down_tensor->dims[0] : 0;
    const std::uint64_t selected_down_rows_per_expert =
        q4_down_available ? selected_down_tensor->dims[1] : 0;
    const std::uint64_t selected_down_rows =
        selected_down_rows_per_expert * real_expert_ids.size();
    const std::uint64_t selected_down_blocks_per_row =
        q4_down_available ? selected_down_cols / 256 : 0;
    const std::uint64_t selected_down_row_nbytes =
        q4_down_available
            ? selected_down_tensor->nbytes /
                  (selected_down_tensor->dims[1] * selected_down_tensor->dims[2])
            : 0;
    const std::uint64_t shared_down_rows =
        q4_down_available ? shared_down_tensor->dims[1] : 0;
    const std::uint64_t shared_down_blocks_per_row =
        q4_down_available ? shared_down_tensor->dims[0] / 256 : 0;
    const std::uint64_t shared_down_row_nbytes =
        q4_down_available ? shared_down_tensor->nbytes / shared_down_tensor->dims[1] : 0;
    Require(load_map.ready, "Qwen3.6 load map is not ready");
    Require(real_tensor_shape_ok, "real selected gate-up tensor shape mismatch");
    Require(shared_tensor_shape_ok, "shared gate/up tensor shape mismatch");
    Require(real_input.size() == static_cast<std::size_t>(kHiddenSize),
            "real input size mismatch");
    Require(real_expert_ids.size() == 8, "real top-k expert count mismatch");
    Require(real_oracle.size() == real_selected_rows, "real oracle size mismatch");
    Require(real_blocks_per_row == real_input.size() / 256, "real block count mismatch");
    Require(real_row_nbytes == real_blocks_per_row * kQ4KBlockBytes,
            "real Q4_K row byte mismatch");
    Require(shared_rows == static_cast<std::uint64_t>(kGateUpRowsPerExpert),
            "shared gate/up row count mismatch");
    Require(shared_blocks_per_row == real_blocks_per_row, "shared block count mismatch");
    Require(shared_gate_row_nbytes == real_row_nbytes &&
                shared_up_row_nbytes == real_row_nbytes,
            "shared Q4_K row byte mismatch");
    if (q4_down_available) {
      Require(selected_down_oracle.size() == selected_down_rows,
              "selected down oracle size mismatch");
      Require(shared_down_oracle.size() == static_cast<std::size_t>(kHiddenSize),
              "shared down oracle size mismatch");
      Require(selected_down_cols == static_cast<std::uint64_t>(kIntermediateSize),
              "selected down column count mismatch");
      Require(selected_down_rows_per_expert == static_cast<std::uint64_t>(kHiddenSize),
              "selected down rows per expert mismatch");
      Require(selected_down_blocks_per_row == static_cast<std::uint64_t>(kIntermediateSize / 256),
              "selected down block count mismatch");
      Require(selected_down_row_nbytes == selected_down_blocks_per_row * kQ4KBlockBytes,
              "selected down Q4_K row byte mismatch");
      Require(shared_down_rows == static_cast<std::uint64_t>(kHiddenSize),
              "shared down row count mismatch");
      Require(shared_down_blocks_per_row == selected_down_blocks_per_row,
              "shared down block count mismatch");
      Require(shared_down_row_nbytes == selected_down_row_nbytes,
              "shared down Q4_K row byte mismatch");
    }
    const auto real_inputs = BuildRealTileInputs(real_input, args.real_tokens);
    std::vector<float> real_cpu_tile;
    real_cpu_tile.reserve(static_cast<std::size_t>(real_selected_rows) *
                          static_cast<std::size_t>(args.real_tokens));
    std::vector<float> shared_cpu_tile;
    shared_cpu_tile.reserve(static_cast<std::size_t>(shared_rows) *
                            static_cast<std::size_t>(args.real_tokens));
    std::vector<float> selected_swiglu_cpu_tile;
    selected_swiglu_cpu_tile.reserve(
        static_cast<std::size_t>(real_expert_ids.size()) *
        static_cast<std::size_t>(kIntermediateSize) *
        static_cast<std::size_t>(args.real_tokens));
    std::vector<float> shared_swiglu_cpu_tile;
    shared_swiglu_cpu_tile.reserve(static_cast<std::size_t>(kIntermediateSize) *
                                   static_cast<std::size_t>(args.real_tokens));
    std::vector<float> selected_down_cpu_tile;
    std::vector<float> shared_down_cpu_tile;
    std::vector<std::uint8_t> selected_down_q8;
    std::vector<std::uint8_t> shared_down_q8;
    if (q4_down_available) {
      selected_down_cpu_tile.reserve(static_cast<std::size_t>(selected_down_rows) *
                                     static_cast<std::size_t>(args.real_tokens));
      shared_down_cpu_tile.reserve(static_cast<std::size_t>(shared_down_rows) *
                                   static_cast<std::size_t>(args.real_tokens));
      selected_down_q8.reserve(static_cast<std::size_t>(args.real_tokens) *
                               real_expert_ids.size() *
                               selected_down_blocks_per_row * kQ8KBlockBytes);
      shared_down_q8.reserve(static_cast<std::size_t>(args.real_tokens) *
                             shared_down_blocks_per_row * kQ8KBlockBytes);
    }
    for (const auto& input : real_inputs) {
      const auto token_cpu =
          iq36::matvec_expert_tensor(args.model_path, index, real_tensor_name,
                                     input, real_expert_ids);
      Require(token_cpu.size() == real_selected_rows, "real CPU token output size mismatch");
      real_cpu_tile.insert(real_cpu_tile.end(), token_cpu.begin(), token_cpu.end());
      const auto shared_gate_cpu =
          iq36::matvec_tensor(args.model_path, index, shared_gate_tensor_name, input);
      const auto shared_up_cpu =
          iq36::matvec_tensor(args.model_path, index, shared_up_tensor_name, input);
      const auto shared_cpu = ConcatFloatVectors(shared_gate_cpu, shared_up_cpu);
      Require(shared_cpu.size() == shared_rows, "shared CPU token output size mismatch");
      shared_cpu_tile.insert(shared_cpu_tile.end(), shared_cpu.begin(), shared_cpu.end());
      if (q4_down_available) {
        const auto selected_swiglu =
            iq36::apply_swiglu_from_gate_up(
                token_cpu, kIntermediateSize, real_expert_ids.size());
        Require(selected_swiglu.size() ==
                    static_cast<std::size_t>(kIntermediateSize) * real_expert_ids.size(),
                "selected down SwiGLU size mismatch");
        selected_swiglu_cpu_tile.insert(
            selected_swiglu_cpu_tile.end(),
            selected_swiglu.begin(),
            selected_swiglu.end());
        const auto selected_down_cpu =
            iq36::matvec_expert_tensor_per_expert_input(
                args.model_path, index, selected_down_tensor_name,
                selected_swiglu, real_expert_ids);
        Require(selected_down_cpu.size() == selected_down_rows,
                "selected down CPU token output size mismatch");
        selected_down_cpu_tile.insert(
            selected_down_cpu_tile.end(),
            selected_down_cpu.begin(),
            selected_down_cpu.end());
        for (std::size_t selected_index = 0;
             selected_index < real_expert_ids.size();
             ++selected_index) {
          const auto expert_input = SliceFloatVector(
              selected_swiglu,
              selected_index * static_cast<std::size_t>(kIntermediateSize),
              static_cast<std::size_t>(kIntermediateSize));
          const auto q8 = BuildQ8KTokenFromInput(expert_input);
          selected_down_q8.insert(selected_down_q8.end(), q8.begin(), q8.end());
        }
        const auto shared_swiglu =
            iq36::apply_swiglu_pair(shared_gate_cpu, shared_up_cpu);
        Require(shared_swiglu.size() == static_cast<std::size_t>(kIntermediateSize),
                "shared down SwiGLU size mismatch");
        shared_swiglu_cpu_tile.insert(
            shared_swiglu_cpu_tile.end(),
            shared_swiglu.begin(),
            shared_swiglu.end());
        const auto shared_down_cpu =
            iq36::matvec_tensor(
                args.model_path, index, shared_down_tensor_name, shared_swiglu);
        Require(shared_down_cpu.size() == shared_down_rows,
                "shared down CPU token output size mismatch");
        shared_down_cpu_tile.insert(
            shared_down_cpu_tile.end(),
            shared_down_cpu.begin(),
            shared_down_cpu.end());
        const auto shared_q8 = BuildQ8KTokenFromInput(shared_swiglu);
        shared_down_q8.insert(shared_down_q8.end(), shared_q8.begin(), shared_q8.end());
      }
    }
    const auto real_cpu = SliceFloatVector(real_cpu_tile, 0, static_cast<std::size_t>(real_selected_rows));
    std::ifstream model(args.model_path, std::ios::binary);
    Require(static_cast<bool>(model), "failed to open model");
    const auto real_rows =
        ReadSelectedExpertRaw(model, *real_tensor, real_expert_ids,
                              real_rows_per_expert, real_row_nbytes);
    const auto shared_gate_raw = ReadTensorRaw(model, *shared_gate_tensor);
    const auto shared_up_raw = ReadTensorRaw(model, *shared_up_tensor);
    const auto shared_rows_raw = ConcatRawVectors(shared_gate_raw, shared_up_raw);
    const auto selected_down_rows_raw =
        q4_down_available
            ? ReadSelectedExpertRaw(model, *selected_down_tensor, real_expert_ids,
                                    selected_down_rows_per_expert,
                                    selected_down_row_nbytes)
            : std::vector<std::uint8_t>{};
    const auto shared_down_rows_raw =
        q4_down_available ? ReadTensorRaw(model, *shared_down_tensor)
                          : std::vector<std::uint8_t>{};
    const auto real_q8 = BuildQ8KTokensFromInputs(real_inputs);

    cl_int err = kClSuccess;
    cl_context context = api.clCreateContext(nullptr, 1, &selected.device, nullptr, nullptr, &err);
    Check(err, "clCreateContext");
    cl_command_queue queue =
        api.clCreateCommandQueue(context, selected.device, kClQueueProfilingEnable, &err);
    Check(err, "clCreateCommandQueue");
    const char* source = kOpenClSource;
    const std::size_t source_len = std::strlen(kOpenClSource);
    cl_program program = api.clCreateProgramWithSource(context, 1, &source, &source_len, &err);
    Check(err, "clCreateProgramWithSource");
    const auto build_start = std::chrono::steady_clock::now();
    const cl_int build_err =
        api.clBuildProgram(program, 1, &selected.device, "-cl-std=CL2.0", nullptr, nullptr);
    const auto build_end = std::chrono::steady_clock::now();
    const std::string build_log = BuildLog(api, program, selected.device);
    if (build_err != kClSuccess) {
      Die("clBuildProgram failed: " + build_log);
    }
    const double build_ms =
        std::chrono::duration<double, std::milli>(build_end - build_start).count();
    cl_kernel kernel = api.clCreateKernel(program, "dpas_q4_exact_probe", &err);
    Check(err, "clCreateKernel");
    cl_kernel q4k_kernel = api.clCreateKernel(program, "dpas_q4k_full_gate", &err);
    Check(err, "clCreateKernel q4k");
    cl_kernel q4k_tokenreuse_kernel =
        api.clCreateKernel(program, "dpas_q4k_full_gate_tokenreuse", &err);
    Check(err, "clCreateKernel q4k tokenreuse");
    cl_kernel swiglu_fusion_kernel =
        api.clCreateKernel(program, "dpas_q4k_gateup_swiglu_tokenreuse", &err);
    Check(err, "clCreateKernel swiglu fusion");
    cl_kernel swiglu_fusion_localq8_kernel =
        api.clCreateKernel(program, "dpas_q4k_gateup_swiglu_tokenreuse_localq8", &err);
    Check(err, "clCreateKernel swiglu fusion localq8");
    cl_kernel q8_prep_kernel =
        api.clCreateKernel(program, "q8k_from_gateup_pairs", &err);
    Check(err, "clCreateKernel q8 prep");
    cl_kernel q8_values_prep_kernel =
        api.clCreateKernel(program, "q8k_from_swiglu_values", &err);
    Check(err, "clCreateKernel q8 values prep");

    cl_mem rows_mem = api.clCreateBuffer(
        context, kClMemReadOnly, rows.size(), nullptr, &err);
    Check(err, "clCreateBuffer rows");
    cl_mem q8_mem = api.clCreateBuffer(
        context, kClMemReadOnly, q8.size(), nullptr, &err);
    Check(err, "clCreateBuffer q8");
    cl_mem dpas_mem = api.clCreateBuffer(
        context, kClMemWriteOnly, cpu_ref.size() * sizeof(int), nullptr, &err);
    Check(err, "clCreateBuffer dpas");
    cl_mem scalar_mem = api.clCreateBuffer(
        context, kClMemWriteOnly, cpu_ref.size() * sizeof(int), nullptr, &err);
    Check(err, "clCreateBuffer scalar");
    cl_mem q4k_rows_mem = api.clCreateBuffer(
        context, kClMemReadOnly, q4k_rows.size(), nullptr, &err);
    Check(err, "clCreateBuffer q4k rows");
    cl_mem q8k_mem = api.clCreateBuffer(
        context, kClMemReadOnly, q8k_tokens.size(), nullptr, &err);
    Check(err, "clCreateBuffer q8k");
    cl_mem q4k_dpas_mem = api.clCreateBuffer(
        context, kClMemWriteOnly, q4k_cpu_ref.size() * sizeof(float), nullptr, &err);
    Check(err, "clCreateBuffer q4k dpas");
    cl_mem q4k_scalar_mem = api.clCreateBuffer(
        context, kClMemWriteOnly, q4k_cpu_ref.size() * sizeof(float), nullptr, &err);
    Check(err, "clCreateBuffer q4k scalar");
    cl_mem real_rows_mem = api.clCreateBuffer(
        context, kClMemReadOnly, real_rows.size(), nullptr, &err);
    Check(err, "clCreateBuffer real rows");
    cl_mem real_q8_mem = api.clCreateBuffer(
        context, kClMemReadOnly, real_q8.size(), nullptr, &err);
    Check(err, "clCreateBuffer real q8");
    cl_mem real_dpas_mem = api.clCreateBuffer(
        context, kClMemWriteOnly, real_cpu_tile.size() * sizeof(float), nullptr, &err);
    Check(err, "clCreateBuffer real dpas");
    cl_mem real_scalar_mem = api.clCreateBuffer(
        context, kClMemWriteOnly, real_cpu_tile.size() * sizeof(float), nullptr, &err);
    Check(err, "clCreateBuffer real scalar");
    cl_mem shared_rows_mem = api.clCreateBuffer(
        context, kClMemReadOnly, shared_rows_raw.size(), nullptr, &err);
    Check(err, "clCreateBuffer shared rows");
    cl_mem shared_dpas_mem = api.clCreateBuffer(
        context, kClMemWriteOnly, shared_cpu_tile.size() * sizeof(float), nullptr, &err);
    Check(err, "clCreateBuffer shared dpas");
    cl_mem shared_scalar_mem = api.clCreateBuffer(
        context, kClMemWriteOnly, shared_cpu_tile.size() * sizeof(float), nullptr, &err);
    Check(err, "clCreateBuffer shared scalar");
    cl_mem real_tokenreuse_mem = api.clCreateBuffer(
        context, kClMemReadWrite, real_cpu_tile.size() * sizeof(float), nullptr, &err);
    Check(err, "clCreateBuffer real tokenreuse");
    cl_mem shared_tokenreuse_mem = api.clCreateBuffer(
        context, kClMemReadWrite, shared_cpu_tile.size() * sizeof(float), nullptr, &err);
    Check(err, "clCreateBuffer shared tokenreuse");
    cl_mem selected_down_rows_mem = nullptr;
    cl_mem selected_down_q8_mem = nullptr;
    cl_mem selected_down_tokenreuse_mem = nullptr;
    cl_mem selected_down_q8_from_gpu_mem = nullptr;
    cl_mem selected_down_device_q8_tokenreuse_mem = nullptr;
    cl_mem selected_swiglu_fusion_mem = nullptr;
    cl_mem selected_down_q8_from_fused_mem = nullptr;
    cl_mem selected_down_fused_device_q8_tokenreuse_mem = nullptr;
    cl_mem shared_down_rows_mem = nullptr;
    cl_mem shared_down_q8_mem = nullptr;
    cl_mem shared_down_tokenreuse_mem = nullptr;
    cl_mem shared_down_q8_from_gpu_mem = nullptr;
    cl_mem shared_down_device_q8_tokenreuse_mem = nullptr;
    cl_mem shared_swiglu_fusion_mem = nullptr;
    cl_mem shared_down_q8_from_fused_mem = nullptr;
    cl_mem shared_down_fused_device_q8_tokenreuse_mem = nullptr;
    if (q4_down_available) {
      selected_down_rows_mem = api.clCreateBuffer(
          context, kClMemReadOnly, selected_down_rows_raw.size(), nullptr, &err);
      Check(err, "clCreateBuffer selected down rows");
      selected_down_q8_mem = api.clCreateBuffer(
          context, kClMemReadOnly, selected_down_q8.size(), nullptr, &err);
      Check(err, "clCreateBuffer selected down q8");
      selected_down_tokenreuse_mem = api.clCreateBuffer(
          context, kClMemWriteOnly, selected_down_cpu_tile.size() * sizeof(float),
          nullptr, &err);
      Check(err, "clCreateBuffer selected down tokenreuse");
      selected_down_q8_from_gpu_mem = api.clCreateBuffer(
          context, kClMemReadWrite, selected_down_q8.size(), nullptr, &err);
      Check(err, "clCreateBuffer selected down q8 from gpu");
      selected_down_device_q8_tokenreuse_mem = api.clCreateBuffer(
          context, kClMemWriteOnly, selected_down_cpu_tile.size() * sizeof(float),
          nullptr, &err);
      Check(err, "clCreateBuffer selected down device q8 tokenreuse");
      selected_swiglu_fusion_mem = api.clCreateBuffer(
          context, kClMemReadWrite, selected_swiglu_cpu_tile.size() * sizeof(float),
          nullptr, &err);
      Check(err, "clCreateBuffer selected swiglu fusion");
      selected_down_q8_from_fused_mem = api.clCreateBuffer(
          context, kClMemReadWrite, selected_down_q8.size(), nullptr, &err);
      Check(err, "clCreateBuffer selected down q8 from fused swiglu");
      selected_down_fused_device_q8_tokenreuse_mem = api.clCreateBuffer(
          context, kClMemWriteOnly, selected_down_cpu_tile.size() * sizeof(float),
          nullptr, &err);
      Check(err, "clCreateBuffer selected down fused device q8 tokenreuse");
      shared_down_rows_mem = api.clCreateBuffer(
          context, kClMemReadOnly, shared_down_rows_raw.size(), nullptr, &err);
      Check(err, "clCreateBuffer shared down rows");
      shared_down_q8_mem = api.clCreateBuffer(
          context, kClMemReadOnly, shared_down_q8.size(), nullptr, &err);
      Check(err, "clCreateBuffer shared down q8");
      shared_down_tokenreuse_mem = api.clCreateBuffer(
          context, kClMemWriteOnly, shared_down_cpu_tile.size() * sizeof(float),
          nullptr, &err);
      Check(err, "clCreateBuffer shared down tokenreuse");
      shared_down_q8_from_gpu_mem = api.clCreateBuffer(
          context, kClMemReadWrite, shared_down_q8.size(), nullptr, &err);
      Check(err, "clCreateBuffer shared down q8 from gpu");
      shared_down_device_q8_tokenreuse_mem = api.clCreateBuffer(
          context, kClMemWriteOnly, shared_down_cpu_tile.size() * sizeof(float),
          nullptr, &err);
      Check(err, "clCreateBuffer shared down device q8 tokenreuse");
      shared_swiglu_fusion_mem = api.clCreateBuffer(
          context, kClMemReadWrite, shared_swiglu_cpu_tile.size() * sizeof(float),
          nullptr, &err);
      Check(err, "clCreateBuffer shared swiglu fusion");
      shared_down_q8_from_fused_mem = api.clCreateBuffer(
          context, kClMemReadWrite, shared_down_q8.size(), nullptr, &err);
      Check(err, "clCreateBuffer shared down q8 from fused swiglu");
      shared_down_fused_device_q8_tokenreuse_mem = api.clCreateBuffer(
          context, kClMemWriteOnly, shared_down_cpu_tile.size() * sizeof(float),
          nullptr, &err);
      Check(err, "clCreateBuffer shared down fused device q8 tokenreuse");
    }
    Check(api.clEnqueueWriteBuffer(
              queue, rows_mem, kClTrue, 0, rows.size(), rows.data(), 0, nullptr, nullptr),
          "clEnqueueWriteBuffer rows");
    Check(api.clEnqueueWriteBuffer(
              queue, q8_mem, kClTrue, 0, q8.size(), q8.data(), 0, nullptr, nullptr),
          "clEnqueueWriteBuffer q8");
    Check(api.clEnqueueWriteBuffer(
              queue, q4k_rows_mem, kClTrue, 0, q4k_rows.size(), q4k_rows.data(), 0, nullptr, nullptr),
          "clEnqueueWriteBuffer q4k rows");
    Check(api.clEnqueueWriteBuffer(
              queue, q8k_mem, kClTrue, 0, q8k_tokens.size(), q8k_tokens.data(), 0, nullptr, nullptr),
          "clEnqueueWriteBuffer q8k");
    Check(api.clEnqueueWriteBuffer(
              queue, real_rows_mem, kClTrue, 0, real_rows.size(), real_rows.data(), 0, nullptr, nullptr),
          "clEnqueueWriteBuffer real rows");
    Check(api.clEnqueueWriteBuffer(
              queue, real_q8_mem, kClTrue, 0, real_q8.size(), real_q8.data(), 0, nullptr, nullptr),
          "clEnqueueWriteBuffer real q8");
    Check(api.clEnqueueWriteBuffer(
              queue, shared_rows_mem, kClTrue, 0, shared_rows_raw.size(), shared_rows_raw.data(), 0, nullptr, nullptr),
          "clEnqueueWriteBuffer shared rows");
    if (q4_down_available) {
      Check(api.clEnqueueWriteBuffer(
                queue, selected_down_rows_mem, kClTrue, 0,
                selected_down_rows_raw.size(), selected_down_rows_raw.data(),
                0, nullptr, nullptr),
            "clEnqueueWriteBuffer selected down rows");
      Check(api.clEnqueueWriteBuffer(
                queue, selected_down_q8_mem, kClTrue, 0,
                selected_down_q8.size(), selected_down_q8.data(),
                0, nullptr, nullptr),
            "clEnqueueWriteBuffer selected down q8");
      Check(api.clEnqueueWriteBuffer(
                queue, shared_down_rows_mem, kClTrue, 0,
                shared_down_rows_raw.size(), shared_down_rows_raw.data(),
                0, nullptr, nullptr),
            "clEnqueueWriteBuffer shared down rows");
      Check(api.clEnqueueWriteBuffer(
                queue, shared_down_q8_mem, kClTrue, 0,
                shared_down_q8.size(), shared_down_q8.data(),
                0, nullptr, nullptr),
            "clEnqueueWriteBuffer shared down q8");
    }
    Check(api.clSetKernelArg(kernel, 0, sizeof(rows_mem), &rows_mem), "clSetKernelArg rows");
    Check(api.clSetKernelArg(kernel, 1, sizeof(q8_mem), &q8_mem), "clSetKernelArg q8");
    Check(api.clSetKernelArg(kernel, 2, sizeof(dpas_mem), &dpas_mem), "clSetKernelArg dpas");
    Check(api.clSetKernelArg(kernel, 3, sizeof(scalar_mem), &scalar_mem), "clSetKernelArg scalar");
    const cl_uint q4k_blocks_per_row = static_cast<cl_uint>(kQ4KBlocksPerRow);
    Check(api.clSetKernelArg(q4k_kernel, 0, sizeof(q4k_rows_mem), &q4k_rows_mem),
          "clSetKernelArg q4k rows");
    Check(api.clSetKernelArg(q4k_kernel, 1, sizeof(q8k_mem), &q8k_mem),
          "clSetKernelArg q8k");
    Check(api.clSetKernelArg(q4k_kernel, 2, sizeof(q4k_dpas_mem), &q4k_dpas_mem),
          "clSetKernelArg q4k dpas");
    Check(api.clSetKernelArg(q4k_kernel, 3, sizeof(q4k_scalar_mem), &q4k_scalar_mem),
          "clSetKernelArg q4k scalar");
    Check(api.clSetKernelArg(q4k_kernel, 4, sizeof(q4k_blocks_per_row), &q4k_blocks_per_row),
          "clSetKernelArg q4k blocks");
    const cl_uint q4k_row_count = static_cast<cl_uint>(kQ4KRows);
    Check(api.clSetKernelArg(q4k_kernel, 5, sizeof(q4k_row_count), &q4k_row_count),
          "clSetKernelArg q4k row count");

    std::vector<double> kernel_us;
    kernel_us.reserve(static_cast<std::size_t>(args.repeat));
    const std::size_t global = 16;
    const std::size_t local = 16;
    for (int i = 0; i < args.repeat; ++i) {
      cl_event event = nullptr;
      Check(api.clEnqueueNDRangeKernel(
                queue, kernel, 1, nullptr, &global, &local, 0, nullptr, &event),
            "clEnqueueNDRangeKernel");
      Check(api.clFinish(queue), "clFinish");
      kernel_us.push_back(EventUs(api, event));
      api.clReleaseEvent(event);
    }
    std::vector<double> q4k_kernel_us;
    q4k_kernel_us.reserve(static_cast<std::size_t>(args.repeat));
    const std::size_t q4k_global = ((kQ4KRows + 15) / 16) * kQ4KTokens * 16;
    for (int i = 0; i < args.repeat; ++i) {
      cl_event event = nullptr;
      Check(api.clEnqueueNDRangeKernel(
                queue, q4k_kernel, 1, nullptr, &q4k_global, &local, 0, nullptr, &event),
            "clEnqueueNDRangeKernel q4k");
      Check(api.clFinish(queue), "clFinish q4k");
      q4k_kernel_us.push_back(EventUs(api, event));
      api.clReleaseEvent(event);
    }
    const cl_uint real_blocks_arg = static_cast<cl_uint>(real_blocks_per_row);
    const cl_uint real_rows_arg = static_cast<cl_uint>(real_selected_rows);
    Check(api.clSetKernelArg(q4k_kernel, 0, sizeof(real_rows_mem), &real_rows_mem),
          "clSetKernelArg real rows");
    Check(api.clSetKernelArg(q4k_kernel, 1, sizeof(real_q8_mem), &real_q8_mem),
          "clSetKernelArg real q8");
    Check(api.clSetKernelArg(q4k_kernel, 2, sizeof(real_dpas_mem), &real_dpas_mem),
          "clSetKernelArg real dpas");
    Check(api.clSetKernelArg(q4k_kernel, 3, sizeof(real_scalar_mem), &real_scalar_mem),
          "clSetKernelArg real scalar");
    Check(api.clSetKernelArg(q4k_kernel, 4, sizeof(real_blocks_arg), &real_blocks_arg),
          "clSetKernelArg real blocks");
    Check(api.clSetKernelArg(q4k_kernel, 5, sizeof(real_rows_arg), &real_rows_arg),
          "clSetKernelArg real row count");
    std::vector<double> real_kernel_us;
    real_kernel_us.reserve(static_cast<std::size_t>(args.repeat));
    const std::size_t real_row_tiles =
        (static_cast<std::size_t>(real_selected_rows) + 15) / 16;
    const std::size_t real_global =
        real_row_tiles * static_cast<std::size_t>(args.real_tokens) * 16;
    for (int i = 0; i < args.repeat; ++i) {
      cl_event event = nullptr;
      Check(api.clEnqueueNDRangeKernel(
                queue, q4k_kernel, 1, nullptr, &real_global, &local, 0, nullptr, &event),
            "clEnqueueNDRangeKernel real q4k");
      Check(api.clFinish(queue), "clFinish real q4k");
      real_kernel_us.push_back(EventUs(api, event));
      api.clReleaseEvent(event);
    }
    const cl_uint shared_blocks_arg = static_cast<cl_uint>(shared_blocks_per_row);
    const cl_uint shared_rows_arg = static_cast<cl_uint>(shared_rows);
    Check(api.clSetKernelArg(q4k_kernel, 0, sizeof(shared_rows_mem), &shared_rows_mem),
          "clSetKernelArg shared rows");
    Check(api.clSetKernelArg(q4k_kernel, 1, sizeof(real_q8_mem), &real_q8_mem),
          "clSetKernelArg shared q8");
    Check(api.clSetKernelArg(q4k_kernel, 2, sizeof(shared_dpas_mem), &shared_dpas_mem),
          "clSetKernelArg shared dpas");
    Check(api.clSetKernelArg(q4k_kernel, 3, sizeof(shared_scalar_mem), &shared_scalar_mem),
          "clSetKernelArg shared scalar");
    Check(api.clSetKernelArg(q4k_kernel, 4, sizeof(shared_blocks_arg), &shared_blocks_arg),
          "clSetKernelArg shared blocks");
    Check(api.clSetKernelArg(q4k_kernel, 5, sizeof(shared_rows_arg), &shared_rows_arg),
          "clSetKernelArg shared row count");
    std::vector<double> shared_kernel_us;
    shared_kernel_us.reserve(static_cast<std::size_t>(args.repeat));
    const std::size_t shared_row_tiles =
        (static_cast<std::size_t>(shared_rows) + 15) / 16;
    const std::size_t shared_global =
        shared_row_tiles * static_cast<std::size_t>(args.real_tokens) * 16;
    for (int i = 0; i < args.repeat; ++i) {
      cl_event event = nullptr;
      Check(api.clEnqueueNDRangeKernel(
                queue, q4k_kernel, 1, nullptr, &shared_global, &local, 0, nullptr, &event),
            "clEnqueueNDRangeKernel shared q4k");
      Check(api.clFinish(queue), "clFinish shared q4k");
      shared_kernel_us.push_back(EventUs(api, event));
      api.clReleaseEvent(event);
    }
    const std::vector<cl_uint> token_blocks = {4U, 8U, 16U};
    std::vector<TokenReuseLaneResult> real_tokenreuse_results;
    std::vector<TokenReuseLaneResult> shared_tokenreuse_results;
    real_tokenreuse_results.reserve(token_blocks.size());
    shared_tokenreuse_results.reserve(token_blocks.size());
    for (const cl_uint token_block : token_blocks) {
      real_tokenreuse_results.push_back(RunTokenReuseLane(
          api, queue, q4k_tokenreuse_kernel, real_rows_mem, real_q8_mem,
          real_tokenreuse_mem, real_cpu_tile, real_blocks_arg, real_rows_arg,
          static_cast<cl_uint>(args.real_tokens), token_block, real_rows_arg,
          args.repeat));
      shared_tokenreuse_results.push_back(RunTokenReuseLane(
          api, queue, q4k_tokenreuse_kernel, shared_rows_mem, real_q8_mem,
          shared_tokenreuse_mem, shared_cpu_tile, shared_blocks_arg, shared_rows_arg,
          static_cast<cl_uint>(args.real_tokens), token_block, shared_rows_arg,
          args.repeat));
    }
    std::vector<TokenReuseLaneResult> selected_down_tokenreuse_results;
    std::vector<TokenReuseLaneResult> shared_down_tokenreuse_results;
    Q8PrepLaneResult selected_down_q8_prep_result;
    Q8PrepLaneResult shared_down_q8_prep_result;
    std::vector<TokenReuseLaneResult> selected_down_device_q8_tokenreuse_results;
    std::vector<TokenReuseLaneResult> shared_down_device_q8_tokenreuse_results;
    std::vector<SwiGluFusionLaneResult> selected_swiglu_fusion_results;
    std::vector<SwiGluFusionLaneResult> shared_swiglu_fusion_results;
    std::vector<SwiGluFusionLaneResult> selected_swiglu_fusion_localq8_results;
    std::vector<SwiGluFusionLaneResult> shared_swiglu_fusion_localq8_results;
    Q8PrepLaneResult selected_down_fused_q8_prep_result;
    Q8PrepLaneResult shared_down_fused_q8_prep_result;
    std::vector<TokenReuseLaneResult> selected_down_fused_device_q8_tokenreuse_results;
    std::vector<TokenReuseLaneResult> shared_down_fused_device_q8_tokenreuse_results;
    if (q4_down_available) {
      selected_down_tokenreuse_results.reserve(token_blocks.size());
      shared_down_tokenreuse_results.reserve(token_blocks.size());
      selected_down_device_q8_tokenreuse_results.reserve(token_blocks.size());
      shared_down_device_q8_tokenreuse_results.reserve(token_blocks.size());
      selected_swiglu_fusion_results.reserve(token_blocks.size());
      shared_swiglu_fusion_results.reserve(token_blocks.size());
      selected_swiglu_fusion_localq8_results.reserve(token_blocks.size() * 2);
      shared_swiglu_fusion_localq8_results.reserve(token_blocks.size() * 2);
      selected_down_fused_device_q8_tokenreuse_results.reserve(token_blocks.size());
      shared_down_fused_device_q8_tokenreuse_results.reserve(token_blocks.size());
      const cl_uint selected_down_blocks_arg =
          static_cast<cl_uint>(selected_down_blocks_per_row);
      const cl_uint selected_down_rows_arg =
          static_cast<cl_uint>(selected_down_rows);
      const cl_uint shared_down_blocks_arg =
          static_cast<cl_uint>(shared_down_blocks_per_row);
      const cl_uint shared_down_rows_arg =
          static_cast<cl_uint>(shared_down_rows);
      const cl_uint down_rows_per_q8_group = static_cast<cl_uint>(kHiddenSize);
      for (const cl_uint token_block : token_blocks) {
        selected_down_tokenreuse_results.push_back(RunTokenReuseLane(
            api, queue, q4k_tokenreuse_kernel, selected_down_rows_mem,
            selected_down_q8_mem, selected_down_tokenreuse_mem,
            selected_down_cpu_tile, selected_down_blocks_arg,
            selected_down_rows_arg, static_cast<cl_uint>(args.real_tokens),
            token_block, down_rows_per_q8_group, args.repeat));
        shared_down_tokenreuse_results.push_back(RunTokenReuseLane(
            api, queue, q4k_tokenreuse_kernel, shared_down_rows_mem,
            shared_down_q8_mem, shared_down_tokenreuse_mem,
            shared_down_cpu_tile, shared_down_blocks_arg, shared_down_rows_arg,
            static_cast<cl_uint>(args.real_tokens), token_block,
            down_rows_per_q8_group, args.repeat));
      }
      selected_down_q8_prep_result = RunQ8PrepLane(
          api, queue, q8_prep_kernel, real_tokenreuse_mem,
          selected_down_q8_from_gpu_mem, selected_down_q8,
          static_cast<cl_uint>(args.real_tokens),
          static_cast<cl_uint>(real_expert_ids.size()),
          static_cast<cl_uint>(kIntermediateSize), args.repeat);
      shared_down_q8_prep_result = RunQ8PrepLane(
          api, queue, q8_prep_kernel, shared_tokenreuse_mem,
          shared_down_q8_from_gpu_mem, shared_down_q8,
          static_cast<cl_uint>(args.real_tokens), 1U,
          static_cast<cl_uint>(kIntermediateSize), args.repeat);
      for (const cl_uint token_block : token_blocks) {
        selected_swiglu_fusion_results.push_back(RunSwiGluFusionLane(
            api, queue, swiglu_fusion_kernel, real_rows_mem, real_q8_mem,
            selected_swiglu_fusion_mem, selected_swiglu_cpu_tile,
            real_blocks_arg, static_cast<cl_uint>(real_expert_ids.size()),
            static_cast<cl_uint>(kIntermediateSize),
            static_cast<cl_uint>(args.real_tokens), token_block,
            args.repeat));
        shared_swiglu_fusion_results.push_back(RunSwiGluFusionLane(
            api, queue, swiglu_fusion_kernel, shared_rows_mem, real_q8_mem,
            shared_swiglu_fusion_mem, shared_swiglu_cpu_tile,
            shared_blocks_arg, 1U, static_cast<cl_uint>(kIntermediateSize),
            static_cast<cl_uint>(args.real_tokens), token_block,
            args.repeat));
        for (const std::size_t local_size : {64U, 128U}) {
          selected_swiglu_fusion_localq8_results.push_back(
              RunSwiGluFusionLocalQ8Lane(
                  api, queue, swiglu_fusion_localq8_kernel, real_rows_mem,
                  real_q8_mem, selected_swiglu_fusion_mem,
                  selected_swiglu_cpu_tile, real_blocks_arg,
                  static_cast<cl_uint>(real_expert_ids.size()),
                  static_cast<cl_uint>(kIntermediateSize),
                  static_cast<cl_uint>(args.real_tokens), token_block,
                  local_size, args.repeat));
          shared_swiglu_fusion_localq8_results.push_back(
              RunSwiGluFusionLocalQ8Lane(
                  api, queue, swiglu_fusion_localq8_kernel, shared_rows_mem,
                  real_q8_mem, shared_swiglu_fusion_mem,
                  shared_swiglu_cpu_tile, shared_blocks_arg, 1U,
                  static_cast<cl_uint>(kIntermediateSize),
                  static_cast<cl_uint>(args.real_tokens), token_block,
                  local_size, args.repeat));
        }
      }
      selected_down_fused_q8_prep_result = RunQ8PrepLane(
          api, queue, q8_values_prep_kernel, selected_swiglu_fusion_mem,
          selected_down_q8_from_fused_mem, selected_down_q8,
          static_cast<cl_uint>(args.real_tokens),
          static_cast<cl_uint>(real_expert_ids.size()),
          static_cast<cl_uint>(kIntermediateSize), args.repeat);
      shared_down_fused_q8_prep_result = RunQ8PrepLane(
          api, queue, q8_values_prep_kernel, shared_swiglu_fusion_mem,
          shared_down_q8_from_fused_mem, shared_down_q8,
          static_cast<cl_uint>(args.real_tokens), 1U,
          static_cast<cl_uint>(kIntermediateSize), args.repeat);
      for (const cl_uint token_block : token_blocks) {
        selected_down_device_q8_tokenreuse_results.push_back(RunTokenReuseLane(
            api, queue, q4k_tokenreuse_kernel, selected_down_rows_mem,
            selected_down_q8_from_gpu_mem, selected_down_device_q8_tokenreuse_mem,
            selected_down_cpu_tile, selected_down_blocks_arg,
            selected_down_rows_arg, static_cast<cl_uint>(args.real_tokens),
            token_block, down_rows_per_q8_group, args.repeat));
        shared_down_device_q8_tokenreuse_results.push_back(RunTokenReuseLane(
            api, queue, q4k_tokenreuse_kernel, shared_down_rows_mem,
            shared_down_q8_from_gpu_mem, shared_down_device_q8_tokenreuse_mem,
            shared_down_cpu_tile, shared_down_blocks_arg, shared_down_rows_arg,
            static_cast<cl_uint>(args.real_tokens), token_block,
            down_rows_per_q8_group, args.repeat));
        selected_down_fused_device_q8_tokenreuse_results.push_back(RunTokenReuseLane(
            api, queue, q4k_tokenreuse_kernel, selected_down_rows_mem,
            selected_down_q8_from_fused_mem,
            selected_down_fused_device_q8_tokenreuse_mem,
            selected_down_cpu_tile, selected_down_blocks_arg,
            selected_down_rows_arg, static_cast<cl_uint>(args.real_tokens),
            token_block, down_rows_per_q8_group, args.repeat));
        shared_down_fused_device_q8_tokenreuse_results.push_back(RunTokenReuseLane(
            api, queue, q4k_tokenreuse_kernel, shared_down_rows_mem,
            shared_down_q8_from_fused_mem,
            shared_down_fused_device_q8_tokenreuse_mem,
            shared_down_cpu_tile, shared_down_blocks_arg, shared_down_rows_arg,
            static_cast<cl_uint>(args.real_tokens), token_block,
            down_rows_per_q8_group, args.repeat));
      }
    }

    std::vector<int> dpas(cpu_ref.size(), 0);
    std::vector<int> scalar(cpu_ref.size(), 0);
    Check(api.clEnqueueReadBuffer(
              queue, dpas_mem, kClTrue, 0, dpas.size() * sizeof(int), dpas.data(), 0, nullptr, nullptr),
          "clEnqueueReadBuffer dpas");
    Check(api.clEnqueueReadBuffer(
              queue, scalar_mem, kClTrue, 0, scalar.size() * sizeof(int), scalar.data(), 0, nullptr, nullptr),
          "clEnqueueReadBuffer scalar");
    std::vector<float> q4k_dpas(q4k_cpu_ref.size(), 0.0f);
    std::vector<float> q4k_scalar(q4k_cpu_ref.size(), 0.0f);
    Check(api.clEnqueueReadBuffer(
              queue, q4k_dpas_mem, kClTrue, 0, q4k_dpas.size() * sizeof(float), q4k_dpas.data(), 0, nullptr, nullptr),
          "clEnqueueReadBuffer q4k dpas");
    Check(api.clEnqueueReadBuffer(
              queue, q4k_scalar_mem, kClTrue, 0, q4k_scalar.size() * sizeof(float), q4k_scalar.data(), 0, nullptr, nullptr),
          "clEnqueueReadBuffer q4k scalar");
    std::vector<float> real_dpas_tile(real_cpu_tile.size(), 0.0f);
    std::vector<float> real_scalar_tile(real_cpu_tile.size(), 0.0f);
    Check(api.clEnqueueReadBuffer(
              queue, real_dpas_mem, kClTrue, 0, real_dpas_tile.size() * sizeof(float), real_dpas_tile.data(), 0, nullptr, nullptr),
          "clEnqueueReadBuffer real dpas");
    Check(api.clEnqueueReadBuffer(
              queue, real_scalar_mem, kClTrue, 0, real_scalar_tile.size() * sizeof(float), real_scalar_tile.data(), 0, nullptr, nullptr),
          "clEnqueueReadBuffer real scalar");
    const auto real_dpas =
        SliceFloatVector(real_dpas_tile, 0, static_cast<std::size_t>(real_selected_rows));
    const auto real_scalar =
        SliceFloatVector(real_scalar_tile, 0, static_cast<std::size_t>(real_selected_rows));
    std::vector<float> shared_dpas_tile(shared_cpu_tile.size(), 0.0f);
    std::vector<float> shared_scalar_tile(shared_cpu_tile.size(), 0.0f);
    Check(api.clEnqueueReadBuffer(
              queue, shared_dpas_mem, kClTrue, 0, shared_dpas_tile.size() * sizeof(float), shared_dpas_tile.data(), 0, nullptr, nullptr),
          "clEnqueueReadBuffer shared dpas");
    Check(api.clEnqueueReadBuffer(
              queue, shared_scalar_mem, kClTrue, 0, shared_scalar_tile.size() * sizeof(float), shared_scalar_tile.data(), 0, nullptr, nullptr),
          "clEnqueueReadBuffer shared scalar");
    const auto selected_down_cpu =
        q4_down_available
            ? SliceFloatVector(
                  selected_down_cpu_tile, 0,
                  static_cast<std::size_t>(selected_down_rows))
            : std::vector<float>{};
    const auto shared_down_cpu =
        q4_down_available
            ? SliceFloatVector(
                  shared_down_cpu_tile, 0,
                  static_cast<std::size_t>(shared_down_rows))
            : std::vector<float>{};

    int dpas_cpu_mismatches = 0;
    int dpas_scalar_mismatches = 0;
    int max_abs_dpas_cpu = 0;
    int max_abs_dpas_scalar = 0;
    int first_mismatch = -1;
    for (std::size_t i = 0; i < cpu_ref.size(); ++i) {
      const int diff_cpu = std::abs(dpas[i] - cpu_ref[i]);
      const int diff_scalar = std::abs(dpas[i] - scalar[i]);
      max_abs_dpas_cpu = std::max(max_abs_dpas_cpu, diff_cpu);
      max_abs_dpas_scalar = std::max(max_abs_dpas_scalar, diff_scalar);
      if (diff_cpu != 0) {
        dpas_cpu_mismatches += 1;
        if (first_mismatch < 0) {
          first_mismatch = static_cast<int>(i);
        }
      }
      if (diff_scalar != 0) {
        dpas_scalar_mismatches += 1;
      }
    }
    int q4k_dpas_cpu_mismatches = 0;
    int q4k_dpas_scalar_mismatches = 0;
    float q4k_max_abs_dpas_cpu = 0.0f;
    float q4k_max_abs_dpas_scalar = 0.0f;
    int q4k_first_mismatch = -1;
    for (std::size_t i = 0; i < q4k_cpu_ref.size(); ++i) {
      const float diff_cpu = std::fabs(q4k_dpas[i] - q4k_cpu_ref[i]);
      const float diff_scalar = std::fabs(q4k_dpas[i] - q4k_scalar[i]);
      q4k_max_abs_dpas_cpu = std::max(q4k_max_abs_dpas_cpu, diff_cpu);
      q4k_max_abs_dpas_scalar = std::max(q4k_max_abs_dpas_scalar, diff_scalar);
      if (diff_cpu > kQ4KTolerance) {
        q4k_dpas_cpu_mismatches += 1;
        if (q4k_first_mismatch < 0) {
          q4k_first_mismatch = static_cast<int>(i);
        }
      }
      if (diff_scalar > kQ4KTolerance) {
        q4k_dpas_scalar_mismatches += 1;
      }
    }
    const auto real_cpu_vs_oracle =
        iq36::compare_vectors(real_cpu, real_oracle, kMismatchThreshold);
    const auto real_dpas_vs_cpu =
        iq36::compare_vectors(real_dpas, real_cpu, kMismatchThreshold);
    const auto real_dpas_vs_oracle =
        iq36::compare_vectors(real_dpas, real_oracle, kMismatchThreshold);
    const auto real_scalar_vs_cpu =
        iq36::compare_vectors(real_scalar, real_cpu, kMismatchThreshold);
    const auto real_tile_dpas_vs_cpu =
        iq36::compare_vectors(real_dpas_tile, real_cpu_tile, kMismatchThreshold);
    const auto real_tile_scalar_vs_cpu =
        iq36::compare_vectors(real_scalar_tile, real_cpu_tile, kMismatchThreshold);
    const auto shared_tile_dpas_vs_cpu =
        iq36::compare_vectors(shared_dpas_tile, shared_cpu_tile, kMismatchThreshold);
    const auto shared_tile_scalar_vs_cpu =
        iq36::compare_vectors(shared_scalar_tile, shared_cpu_tile, kMismatchThreshold);
    const auto selected_down_cpu_vs_oracle =
        q4_down_available
            ? iq36::compare_vectors(
                  selected_down_cpu, selected_down_oracle, kMismatchThreshold)
            : iq36::VectorCompareStats{};
    const auto shared_down_cpu_vs_oracle =
        q4_down_available
            ? iq36::compare_vectors(
                  shared_down_cpu, shared_down_oracle, kMismatchThreshold)
            : iq36::VectorCompareStats{};
    const bool real_cpu_matches_oracle = ComparePassed(real_cpu_vs_oracle);
    const bool real_dpas_matches_cpu = ComparePassed(real_dpas_vs_cpu);
    const bool real_dpas_matches_oracle = ComparePassed(real_dpas_vs_oracle);
    const bool real_scalar_matches_cpu = ComparePassed(real_scalar_vs_cpu);
    const bool real_tile_dpas_matches_cpu = ComparePassed(real_tile_dpas_vs_cpu);
    const bool real_tile_scalar_matches_cpu = ComparePassed(real_tile_scalar_vs_cpu);
    const bool shared_tile_dpas_matches_cpu = ComparePassed(shared_tile_dpas_vs_cpu);
    const bool shared_tile_scalar_matches_cpu = ComparePassed(shared_tile_scalar_vs_cpu);
    const bool selected_down_cpu_matches_oracle =
        q4_down_available && ComparePassed(selected_down_cpu_vs_oracle);
    const bool shared_down_cpu_matches_oracle =
        q4_down_available && ComparePassed(shared_down_cpu_vs_oracle);
    bool real_tokenreuse_all_match = true;
    double real_tokenreuse_best_us = std::numeric_limits<double>::infinity();
    int real_tokenreuse_best_block = 0;
    for (const auto& result : real_tokenreuse_results) {
      real_tokenreuse_all_match = real_tokenreuse_all_match && result.dpas_matches_cpu;
      if (result.kernel_min_us < real_tokenreuse_best_us) {
        real_tokenreuse_best_us = result.kernel_min_us;
        real_tokenreuse_best_block = result.token_block;
      }
    }
    bool shared_tokenreuse_all_match = true;
    double shared_tokenreuse_best_us = std::numeric_limits<double>::infinity();
    int shared_tokenreuse_best_block = 0;
    for (const auto& result : shared_tokenreuse_results) {
      shared_tokenreuse_all_match = shared_tokenreuse_all_match && result.dpas_matches_cpu;
      if (result.kernel_min_us < shared_tokenreuse_best_us) {
        shared_tokenreuse_best_us = result.kernel_min_us;
        shared_tokenreuse_best_block = result.token_block;
      }
    }
    bool selected_down_tokenreuse_all_match = q4_down_available;
    double selected_down_tokenreuse_best_us = std::numeric_limits<double>::infinity();
    int selected_down_tokenreuse_best_block = 0;
    for (const auto& result : selected_down_tokenreuse_results) {
      selected_down_tokenreuse_all_match =
          selected_down_tokenreuse_all_match && result.dpas_matches_cpu;
      if (result.kernel_min_us < selected_down_tokenreuse_best_us) {
        selected_down_tokenreuse_best_us = result.kernel_min_us;
        selected_down_tokenreuse_best_block = result.token_block;
      }
    }
    bool shared_down_tokenreuse_all_match = q4_down_available;
    double shared_down_tokenreuse_best_us = std::numeric_limits<double>::infinity();
    int shared_down_tokenreuse_best_block = 0;
    for (const auto& result : shared_down_tokenreuse_results) {
      shared_down_tokenreuse_all_match =
          shared_down_tokenreuse_all_match && result.dpas_matches_cpu;
      if (result.kernel_min_us < shared_down_tokenreuse_best_us) {
        shared_down_tokenreuse_best_us = result.kernel_min_us;
        shared_down_tokenreuse_best_block = result.token_block;
      }
    }
    bool selected_down_device_q8_tokenreuse_all_match = q4_down_available;
    double selected_down_device_q8_tokenreuse_best_us =
        std::numeric_limits<double>::infinity();
    int selected_down_device_q8_tokenreuse_best_block = 0;
    for (const auto& result : selected_down_device_q8_tokenreuse_results) {
      selected_down_device_q8_tokenreuse_all_match =
          selected_down_device_q8_tokenreuse_all_match && result.dpas_matches_cpu;
      if (result.kernel_min_us < selected_down_device_q8_tokenreuse_best_us) {
        selected_down_device_q8_tokenreuse_best_us = result.kernel_min_us;
        selected_down_device_q8_tokenreuse_best_block = result.token_block;
      }
    }
    bool shared_down_device_q8_tokenreuse_all_match = q4_down_available;
    double shared_down_device_q8_tokenreuse_best_us =
        std::numeric_limits<double>::infinity();
    int shared_down_device_q8_tokenreuse_best_block = 0;
    for (const auto& result : shared_down_device_q8_tokenreuse_results) {
      shared_down_device_q8_tokenreuse_all_match =
          shared_down_device_q8_tokenreuse_all_match && result.dpas_matches_cpu;
      if (result.kernel_min_us < shared_down_device_q8_tokenreuse_best_us) {
        shared_down_device_q8_tokenreuse_best_us = result.kernel_min_us;
        shared_down_device_q8_tokenreuse_best_block = result.token_block;
      }
    }
    bool selected_swiglu_fusion_all_match = q4_down_available;
    double selected_swiglu_fusion_best_us =
        std::numeric_limits<double>::infinity();
    int selected_swiglu_fusion_best_block = 0;
    for (const auto& result : selected_swiglu_fusion_results) {
      selected_swiglu_fusion_all_match =
          selected_swiglu_fusion_all_match && result.output_matches_cpu;
      if (result.kernel_min_us < selected_swiglu_fusion_best_us) {
        selected_swiglu_fusion_best_us = result.kernel_min_us;
        selected_swiglu_fusion_best_block = result.token_block;
      }
    }
    bool shared_swiglu_fusion_all_match = q4_down_available;
    double shared_swiglu_fusion_best_us =
        std::numeric_limits<double>::infinity();
    int shared_swiglu_fusion_best_block = 0;
    for (const auto& result : shared_swiglu_fusion_results) {
      shared_swiglu_fusion_all_match =
          shared_swiglu_fusion_all_match && result.output_matches_cpu;
      if (result.kernel_min_us < shared_swiglu_fusion_best_us) {
        shared_swiglu_fusion_best_us = result.kernel_min_us;
        shared_swiglu_fusion_best_block = result.token_block;
      }
    }
    bool selected_swiglu_fusion_localq8_all_match = q4_down_available;
    double selected_swiglu_fusion_localq8_best_us =
        std::numeric_limits<double>::infinity();
    int selected_swiglu_fusion_localq8_best_block = 0;
    std::size_t selected_swiglu_fusion_localq8_best_local = 0;
    for (const auto& result : selected_swiglu_fusion_localq8_results) {
      selected_swiglu_fusion_localq8_all_match =
          selected_swiglu_fusion_localq8_all_match && result.output_matches_cpu;
      if (result.kernel_min_us < selected_swiglu_fusion_localq8_best_us) {
        selected_swiglu_fusion_localq8_best_us = result.kernel_min_us;
        selected_swiglu_fusion_localq8_best_block = result.token_block;
        selected_swiglu_fusion_localq8_best_local = result.local_work_items;
      }
    }
    bool shared_swiglu_fusion_localq8_all_match = q4_down_available;
    double shared_swiglu_fusion_localq8_best_us =
        std::numeric_limits<double>::infinity();
    int shared_swiglu_fusion_localq8_best_block = 0;
    std::size_t shared_swiglu_fusion_localq8_best_local = 0;
    for (const auto& result : shared_swiglu_fusion_localq8_results) {
      shared_swiglu_fusion_localq8_all_match =
          shared_swiglu_fusion_localq8_all_match && result.output_matches_cpu;
      if (result.kernel_min_us < shared_swiglu_fusion_localq8_best_us) {
        shared_swiglu_fusion_localq8_best_us = result.kernel_min_us;
        shared_swiglu_fusion_localq8_best_block = result.token_block;
        shared_swiglu_fusion_localq8_best_local = result.local_work_items;
      }
    }
    bool selected_down_fused_device_q8_tokenreuse_all_match = q4_down_available;
    double selected_down_fused_device_q8_tokenreuse_best_us =
        std::numeric_limits<double>::infinity();
    int selected_down_fused_device_q8_tokenreuse_best_block = 0;
    for (const auto& result : selected_down_fused_device_q8_tokenreuse_results) {
      selected_down_fused_device_q8_tokenreuse_all_match =
          selected_down_fused_device_q8_tokenreuse_all_match && result.dpas_matches_cpu;
      if (result.kernel_min_us < selected_down_fused_device_q8_tokenreuse_best_us) {
        selected_down_fused_device_q8_tokenreuse_best_us = result.kernel_min_us;
        selected_down_fused_device_q8_tokenreuse_best_block = result.token_block;
      }
    }
    bool shared_down_fused_device_q8_tokenreuse_all_match = q4_down_available;
    double shared_down_fused_device_q8_tokenreuse_best_us =
        std::numeric_limits<double>::infinity();
    int shared_down_fused_device_q8_tokenreuse_best_block = 0;
    for (const auto& result : shared_down_fused_device_q8_tokenreuse_results) {
      shared_down_fused_device_q8_tokenreuse_all_match =
          shared_down_fused_device_q8_tokenreuse_all_match && result.dpas_matches_cpu;
      if (result.kernel_min_us < shared_down_fused_device_q8_tokenreuse_best_us) {
        shared_down_fused_device_q8_tokenreuse_best_us = result.kernel_min_us;
        shared_down_fused_device_q8_tokenreuse_best_block = result.token_block;
      }
    }
    if (!q4_down_available) {
      selected_down_tokenreuse_best_us = 0.0;
      shared_down_tokenreuse_best_us = 0.0;
      selected_down_device_q8_tokenreuse_best_us = 0.0;
      shared_down_device_q8_tokenreuse_best_us = 0.0;
      selected_swiglu_fusion_best_us = 0.0;
      shared_swiglu_fusion_best_us = 0.0;
      selected_swiglu_fusion_localq8_best_us = 0.0;
      shared_swiglu_fusion_localq8_best_us = 0.0;
      selected_down_fused_device_q8_tokenreuse_best_us = 0.0;
      shared_down_fused_device_q8_tokenreuse_best_us = 0.0;
    }

    const double min_us = *std::min_element(kernel_us.begin(), kernel_us.end());
    double sum_us = 0.0;
    for (const double value : kernel_us) {
      sum_us += value;
    }
    const double mean_us = sum_us / static_cast<double>(kernel_us.size());
    const double integer_ops = static_cast<double>(kOutputCount * 32);
    const double min_gops = integer_ops / (min_us * 1000.0);
    const double q4k_min_us = *std::min_element(q4k_kernel_us.begin(), q4k_kernel_us.end());
    double q4k_sum_us = 0.0;
    for (const double value : q4k_kernel_us) {
      q4k_sum_us += value;
    }
    const double q4k_mean_us = q4k_sum_us / static_cast<double>(q4k_kernel_us.size());
    const double real_min_us = *std::min_element(real_kernel_us.begin(), real_kernel_us.end());
    double real_sum_us = 0.0;
    for (const double value : real_kernel_us) {
      real_sum_us += value;
    }
    const double real_mean_us = real_sum_us / static_cast<double>(real_kernel_us.size());
    const double shared_min_us = *std::min_element(shared_kernel_us.begin(), shared_kernel_us.end());
    double shared_sum_us = 0.0;
    for (const double value : shared_kernel_us) {
      shared_sum_us += value;
    }
    const double shared_mean_us = shared_sum_us / static_cast<double>(shared_kernel_us.size());

    std::cout << std::setprecision(12);
    std::cout << "{";
    std::cout << "\"schema_version\":\"intel-qwen36-gpu-dpas-q4-exact-gate-v2\",";
    std::cout << "\"platform_name\":\"" << JsonEscape(selected.platform_name) << "\",";
    std::cout << "\"device_name\":\"" << JsonEscape(selected.device_name) << "\",";
    std::cout << "\"rows\":" << kRows << ",";
    std::cout << "\"chunks_per_row\":" << kChunks << ",";
    std::cout << "\"dot_product_count\":" << kOutputCount << ",";
    std::cout << "\"dot_width\":32,";
    std::cout << "\"repeat\":" << args.repeat << ",";
    std::cout << "\"dpas_matches_cpu\":" << (dpas_cpu_mismatches == 0 ? "true" : "false") << ",";
    std::cout << "\"dpas_matches_scalar\":" << (dpas_scalar_mismatches == 0 ? "true" : "false") << ",";
    std::cout << "\"exact_match\":"
              << (dpas_cpu_mismatches == 0 && dpas_scalar_mismatches == 0 ? "true" : "false") << ",";
    std::cout << "\"dpas_cpu_mismatch_count\":" << dpas_cpu_mismatches << ",";
    std::cout << "\"dpas_scalar_mismatch_count\":" << dpas_scalar_mismatches << ",";
    std::cout << "\"max_abs_dpas_cpu\":" << max_abs_dpas_cpu << ",";
    std::cout << "\"max_abs_dpas_scalar\":" << max_abs_dpas_scalar << ",";
    std::cout << "\"first_mismatch_index\":" << first_mismatch << ",";
    std::cout << "\"kernel_min_us\":" << min_us << ",";
    std::cout << "\"kernel_mean_us\":" << mean_us << ",";
    std::cout << "\"synthetic_integer_gops_at_min\":" << min_gops << ",";
    std::cout << "\"q4k_full_rows\":" << kQ4KRows << ",";
    std::cout << "\"q4k_full_tokens\":" << kQ4KTokens << ",";
    std::cout << "\"q4k_full_blocks_per_row\":" << kQ4KBlocksPerRow << ",";
    std::cout << "\"q4k_full_output_count\":" << kQ4KOutputCount << ",";
    std::cout << "\"q4k_full_dpas_matches_cpu\":"
              << (q4k_dpas_cpu_mismatches == 0 ? "true" : "false") << ",";
    std::cout << "\"q4k_full_dpas_matches_scalar\":"
              << (q4k_dpas_scalar_mismatches == 0 ? "true" : "false") << ",";
    std::cout << "\"q4k_full_exact_match\":"
              << (q4k_dpas_cpu_mismatches == 0 && q4k_dpas_scalar_mismatches == 0 ? "true" : "false") << ",";
    std::cout << "\"q4k_full_dpas_cpu_mismatch_count\":" << q4k_dpas_cpu_mismatches << ",";
    std::cout << "\"q4k_full_dpas_scalar_mismatch_count\":" << q4k_dpas_scalar_mismatches << ",";
    std::cout << "\"q4k_full_max_abs_dpas_cpu\":" << q4k_max_abs_dpas_cpu << ",";
    std::cout << "\"q4k_full_max_abs_dpas_scalar\":" << q4k_max_abs_dpas_scalar << ",";
    std::cout << "\"q4k_full_first_mismatch_index\":" << q4k_first_mismatch << ",";
    std::cout << "\"q4k_full_kernel_min_us\":" << q4k_min_us << ",";
    std::cout << "\"q4k_full_kernel_mean_us\":" << q4k_mean_us << ",";
    std::cout << "\"q4k_full_claim_class\":\"component_full_q4k_scaled_tile_exactness\",";
    std::cout << "\"real_model_path\":\"" << JsonEscape(args.model_path) << "\",";
    std::cout << "\"real_layer\":" << args.layer << ",";
    std::cout << "\"real_source_token_position\":15,";
    std::cout << "\"real_tensor_name\":\"" << JsonEscape(real_tensor->name) << "\",";
    std::cout << "\"real_tensor_type\":\"" << iq36::ggml_type_name(real_tensor->type) << "\",";
    std::cout << "\"real_tensor_shape_ok\":" << (real_tensor_shape_ok ? "true" : "false") << ",";
    std::cout << "\"real_cols\":" << real_cols << ",";
    std::cout << "\"real_rows_per_expert\":" << real_rows_per_expert << ",";
    std::cout << "\"real_selected_rows\":" << real_selected_rows << ",";
    std::cout << "\"real_blocks_per_row\":" << real_blocks_per_row << ",";
    std::cout << "\"real_row_nbytes\":" << real_row_nbytes << ",";
    std::cout << "\"real_selected_raw_bytes\":" << real_rows.size() << ",";
    std::cout << "\"real_q8_token_bytes\":" << (real_q8.size() / static_cast<std::size_t>(args.real_tokens)) << ",";
    std::cout << "\"real_q8_bytes\":" << real_q8.size() << ",";
    std::cout << "\"real_output_count\":" << real_cpu.size() << ",";
    std::cout << "\"real_tile_tokens\":" << args.real_tokens << ",";
    std::cout << "\"real_tile_activation_source\":\"captured_token15_plus_deterministic_variants\",";
    std::cout << "\"real_tile_q8_bytes\":" << real_q8.size() << ",";
    std::cout << "\"real_tile_output_count\":" << real_cpu_tile.size() << ",";
    std::cout << "\"real_expert_ids\":";
    WriteI32Vector(real_expert_ids);
    std::cout << ",";
    std::cout << "\"real_cpu_matches_oracle\":"
              << (real_cpu_matches_oracle ? "true" : "false") << ",";
    std::cout << "\"real_dpas_matches_cpu\":"
              << (real_dpas_matches_cpu ? "true" : "false") << ",";
    std::cout << "\"real_dpas_matches_oracle\":"
              << (real_dpas_matches_oracle ? "true" : "false") << ",";
    std::cout << "\"real_scalar_matches_cpu\":"
              << (real_scalar_matches_cpu ? "true" : "false") << ",";
    std::cout << "\"real_tile_dpas_matches_cpu\":"
              << (real_tile_dpas_matches_cpu ? "true" : "false") << ",";
    std::cout << "\"real_tile_scalar_matches_cpu\":"
              << (real_tile_scalar_matches_cpu ? "true" : "false") << ",";
    std::cout << "\"real_kernel_min_us\":" << real_min_us << ",";
    std::cout << "\"real_kernel_mean_us\":" << real_mean_us << ",";
    std::cout << "\"real_comparisons\":{";
    std::cout << "\"cpu_vs_oracle\":";
    WriteCompare(real_cpu_vs_oracle);
    std::cout << ",\"dpas_vs_cpu\":";
    WriteCompare(real_dpas_vs_cpu);
    std::cout << ",\"dpas_vs_oracle\":";
    WriteCompare(real_dpas_vs_oracle);
    std::cout << ",\"scalar_vs_cpu\":";
    WriteCompare(real_scalar_vs_cpu);
    std::cout << ",\"tile_dpas_vs_cpu\":";
    WriteCompare(real_tile_dpas_vs_cpu);
    std::cout << ",\"tile_scalar_vs_cpu\":";
    WriteCompare(real_tile_scalar_vs_cpu);
    std::cout << "},";
    std::cout << "\"real_claim_class\":\"component_real_gguf_selected_gateup_multitoken_cpu_tile_plus_oracle_vector\",";
    std::cout << "\"shared_gate_tensor_name\":\"" << JsonEscape(shared_gate_tensor->name) << "\",";
    std::cout << "\"shared_up_tensor_name\":\"" << JsonEscape(shared_up_tensor->name) << "\",";
    std::cout << "\"shared_tensor_shape_ok\":" << (shared_tensor_shape_ok ? "true" : "false") << ",";
    std::cout << "\"shared_rows\":" << shared_rows << ",";
    std::cout << "\"shared_blocks_per_row\":" << shared_blocks_per_row << ",";
    std::cout << "\"shared_raw_bytes\":" << shared_rows_raw.size() << ",";
    std::cout << "\"shared_tile_tokens\":" << args.real_tokens << ",";
    std::cout << "\"shared_tile_output_count\":" << shared_cpu_tile.size() << ",";
    std::cout << "\"shared_tile_dpas_matches_cpu\":"
              << (shared_tile_dpas_matches_cpu ? "true" : "false") << ",";
    std::cout << "\"shared_tile_scalar_matches_cpu\":"
              << (shared_tile_scalar_matches_cpu ? "true" : "false") << ",";
    std::cout << "\"shared_kernel_min_us\":" << shared_min_us << ",";
    std::cout << "\"shared_kernel_mean_us\":" << shared_mean_us << ",";
    std::cout << "\"real_tokenreuse_all_match_cpu\":"
              << (real_tokenreuse_all_match ? "true" : "false") << ",";
    std::cout << "\"real_tokenreuse_best_kernel_min_us\":" << real_tokenreuse_best_us << ",";
    std::cout << "\"real_tokenreuse_best_token_block\":" << real_tokenreuse_best_block << ",";
    std::cout << "\"real_tokenreuse_results\":";
    WriteTokenReuseResults(real_tokenreuse_results);
    std::cout << ",";
    std::cout << "\"shared_tokenreuse_all_match_cpu\":"
              << (shared_tokenreuse_all_match ? "true" : "false") << ",";
    std::cout << "\"shared_tokenreuse_best_kernel_min_us\":" << shared_tokenreuse_best_us << ",";
    std::cout << "\"shared_tokenreuse_best_token_block\":" << shared_tokenreuse_best_block << ",";
    std::cout << "\"shared_tokenreuse_results\":";
    WriteTokenReuseResults(shared_tokenreuse_results);
    std::cout << ",";
    std::cout << "\"shared_comparisons\":{";
    std::cout << "\"tile_dpas_vs_cpu\":";
    WriteCompare(shared_tile_dpas_vs_cpu);
    std::cout << ",\"tile_scalar_vs_cpu\":";
    WriteCompare(shared_tile_scalar_vs_cpu);
    std::cout << "},";
    std::cout << "\"q4_down_available\":"
              << (q4_down_available ? "true" : "false") << ",";
    std::cout << "\"selected_down_tensor_name\":\""
              << JsonEscape(selected_down_tensor != nullptr ? selected_down_tensor->name : "")
              << "\",";
    std::cout << "\"selected_down_tensor_type\":\""
              << JsonEscape(selected_down_tensor != nullptr
                                ? iq36::ggml_type_name(selected_down_tensor->type)
                                : "")
              << "\",";
    std::cout << "\"selected_down_rows_per_expert\":" << selected_down_rows_per_expert << ",";
    std::cout << "\"selected_down_rows\":" << selected_down_rows << ",";
    std::cout << "\"selected_down_blocks_per_row\":" << selected_down_blocks_per_row << ",";
    std::cout << "\"selected_down_row_nbytes\":" << selected_down_row_nbytes << ",";
    std::cout << "\"selected_down_raw_bytes\":" << selected_down_rows_raw.size() << ",";
    std::cout << "\"selected_down_q8_bytes\":" << selected_down_q8.size() << ",";
    std::cout << "\"selected_down_tile_output_count\":"
              << selected_down_cpu_tile.size() << ",";
    std::cout << "\"selected_down_cpu_matches_oracle\":"
              << (selected_down_cpu_matches_oracle ? "true" : "false") << ",";
    std::cout << "\"selected_down_tokenreuse_all_match_cpu\":"
              << (selected_down_tokenreuse_all_match ? "true" : "false") << ",";
    std::cout << "\"selected_down_tokenreuse_best_kernel_min_us\":"
              << selected_down_tokenreuse_best_us << ",";
    std::cout << "\"selected_down_tokenreuse_best_token_block\":"
              << selected_down_tokenreuse_best_block << ",";
    std::cout << "\"selected_down_tokenreuse_results\":";
    WriteTokenReuseResults(selected_down_tokenreuse_results);
    std::cout << ",";
    std::cout << "\"selected_down_q8_prep\":";
    WriteQ8PrepResult(selected_down_q8_prep_result);
    std::cout << ",";
    std::cout << "\"selected_down_device_q8_tokenreuse_all_match_cpu\":"
              << (selected_down_device_q8_tokenreuse_all_match ? "true" : "false") << ",";
    std::cout << "\"selected_down_device_q8_tokenreuse_best_kernel_min_us\":"
              << selected_down_device_q8_tokenreuse_best_us << ",";
    std::cout << "\"selected_down_device_q8_tokenreuse_best_token_block\":"
              << selected_down_device_q8_tokenreuse_best_block << ",";
    std::cout << "\"selected_down_device_q8_tokenreuse_results\":";
    WriteTokenReuseResults(selected_down_device_q8_tokenreuse_results);
    std::cout << ",";
    std::cout << "\"selected_swiglu_fusion_all_match_cpu\":"
              << (selected_swiglu_fusion_all_match ? "true" : "false") << ",";
    std::cout << "\"selected_swiglu_fusion_best_kernel_min_us\":"
              << selected_swiglu_fusion_best_us << ",";
    std::cout << "\"selected_swiglu_fusion_best_token_block\":"
              << selected_swiglu_fusion_best_block << ",";
    std::cout << "\"selected_swiglu_fusion_results\":";
    WriteSwiGluFusionResults(selected_swiglu_fusion_results);
    std::cout << ",";
    std::cout << "\"selected_swiglu_fusion_localq8_all_match_cpu\":"
              << (selected_swiglu_fusion_localq8_all_match ? "true" : "false") << ",";
    std::cout << "\"selected_swiglu_fusion_localq8_best_kernel_min_us\":"
              << selected_swiglu_fusion_localq8_best_us << ",";
    std::cout << "\"selected_swiglu_fusion_localq8_best_token_block\":"
              << selected_swiglu_fusion_localq8_best_block << ",";
    std::cout << "\"selected_swiglu_fusion_localq8_best_local_work_items\":"
              << selected_swiglu_fusion_localq8_best_local << ",";
    std::cout << "\"selected_swiglu_fusion_localq8_results\":";
    WriteSwiGluFusionResults(selected_swiglu_fusion_localq8_results);
    std::cout << ",";
    std::cout << "\"selected_down_fused_q8_prep\":";
    WriteQ8PrepResult(selected_down_fused_q8_prep_result);
    std::cout << ",";
    std::cout << "\"selected_down_fused_device_q8_tokenreuse_all_match_cpu\":"
              << (selected_down_fused_device_q8_tokenreuse_all_match ? "true" : "false") << ",";
    std::cout << "\"selected_down_fused_device_q8_tokenreuse_best_kernel_min_us\":"
              << selected_down_fused_device_q8_tokenreuse_best_us << ",";
    std::cout << "\"selected_down_fused_device_q8_tokenreuse_best_token_block\":"
              << selected_down_fused_device_q8_tokenreuse_best_block << ",";
    std::cout << "\"selected_down_fused_device_q8_tokenreuse_results\":";
    WriteTokenReuseResults(selected_down_fused_device_q8_tokenreuse_results);
    std::cout << ",";
    std::cout << "\"selected_down_comparisons\":{";
    std::cout << "\"cpu_vs_oracle\":";
    WriteCompare(selected_down_cpu_vs_oracle);
    std::cout << "},";
    std::cout << "\"shared_down_tensor_name\":\""
              << JsonEscape(shared_down_tensor != nullptr ? shared_down_tensor->name : "")
              << "\",";
    std::cout << "\"shared_down_tensor_type\":\""
              << JsonEscape(shared_down_tensor != nullptr
                                ? iq36::ggml_type_name(shared_down_tensor->type)
                                : "")
              << "\",";
    std::cout << "\"shared_down_rows\":" << shared_down_rows << ",";
    std::cout << "\"shared_down_blocks_per_row\":" << shared_down_blocks_per_row << ",";
    std::cout << "\"shared_down_row_nbytes\":" << shared_down_row_nbytes << ",";
    std::cout << "\"shared_down_raw_bytes\":" << shared_down_rows_raw.size() << ",";
    std::cout << "\"shared_down_q8_bytes\":" << shared_down_q8.size() << ",";
    std::cout << "\"shared_down_tile_output_count\":"
              << shared_down_cpu_tile.size() << ",";
    std::cout << "\"shared_down_cpu_matches_oracle\":"
              << (shared_down_cpu_matches_oracle ? "true" : "false") << ",";
    std::cout << "\"shared_down_tokenreuse_all_match_cpu\":"
              << (shared_down_tokenreuse_all_match ? "true" : "false") << ",";
    std::cout << "\"shared_down_tokenreuse_best_kernel_min_us\":"
              << shared_down_tokenreuse_best_us << ",";
    std::cout << "\"shared_down_tokenreuse_best_token_block\":"
              << shared_down_tokenreuse_best_block << ",";
    std::cout << "\"shared_down_tokenreuse_results\":";
    WriteTokenReuseResults(shared_down_tokenreuse_results);
    std::cout << ",";
    std::cout << "\"shared_down_q8_prep\":";
    WriteQ8PrepResult(shared_down_q8_prep_result);
    std::cout << ",";
    std::cout << "\"shared_down_device_q8_tokenreuse_all_match_cpu\":"
              << (shared_down_device_q8_tokenreuse_all_match ? "true" : "false") << ",";
    std::cout << "\"shared_down_device_q8_tokenreuse_best_kernel_min_us\":"
              << shared_down_device_q8_tokenreuse_best_us << ",";
    std::cout << "\"shared_down_device_q8_tokenreuse_best_token_block\":"
              << shared_down_device_q8_tokenreuse_best_block << ",";
    std::cout << "\"shared_down_device_q8_tokenreuse_results\":";
    WriteTokenReuseResults(shared_down_device_q8_tokenreuse_results);
    std::cout << ",";
    std::cout << "\"shared_swiglu_fusion_all_match_cpu\":"
              << (shared_swiglu_fusion_all_match ? "true" : "false") << ",";
    std::cout << "\"shared_swiglu_fusion_best_kernel_min_us\":"
              << shared_swiglu_fusion_best_us << ",";
    std::cout << "\"shared_swiglu_fusion_best_token_block\":"
              << shared_swiglu_fusion_best_block << ",";
    std::cout << "\"shared_swiglu_fusion_results\":";
    WriteSwiGluFusionResults(shared_swiglu_fusion_results);
    std::cout << ",";
    std::cout << "\"shared_swiglu_fusion_localq8_all_match_cpu\":"
              << (shared_swiglu_fusion_localq8_all_match ? "true" : "false") << ",";
    std::cout << "\"shared_swiglu_fusion_localq8_best_kernel_min_us\":"
              << shared_swiglu_fusion_localq8_best_us << ",";
    std::cout << "\"shared_swiglu_fusion_localq8_best_token_block\":"
              << shared_swiglu_fusion_localq8_best_block << ",";
    std::cout << "\"shared_swiglu_fusion_localq8_best_local_work_items\":"
              << shared_swiglu_fusion_localq8_best_local << ",";
    std::cout << "\"shared_swiglu_fusion_localq8_results\":";
    WriteSwiGluFusionResults(shared_swiglu_fusion_localq8_results);
    std::cout << ",";
    std::cout << "\"shared_down_fused_q8_prep\":";
    WriteQ8PrepResult(shared_down_fused_q8_prep_result);
    std::cout << ",";
    std::cout << "\"shared_down_fused_device_q8_tokenreuse_all_match_cpu\":"
              << (shared_down_fused_device_q8_tokenreuse_all_match ? "true" : "false") << ",";
    std::cout << "\"shared_down_fused_device_q8_tokenreuse_best_kernel_min_us\":"
              << shared_down_fused_device_q8_tokenreuse_best_us << ",";
    std::cout << "\"shared_down_fused_device_q8_tokenreuse_best_token_block\":"
              << shared_down_fused_device_q8_tokenreuse_best_block << ",";
    std::cout << "\"shared_down_fused_device_q8_tokenreuse_results\":";
    WriteTokenReuseResults(shared_down_fused_device_q8_tokenreuse_results);
    std::cout << ",";
    std::cout << "\"shared_down_comparisons\":{";
    std::cout << "\"cpu_vs_oracle\":";
    WriteCompare(shared_down_cpu_vs_oracle);
    std::cout << "},";
    std::cout << "\"shared_claim_class\":\"component_real_gguf_shared_gateup_multitoken_cpu_tile\",";
    std::cout << "\"program_build_ms\":" << build_ms << ",";
    std::cout << "\"build_log\":\"" << JsonEscape(build_log) << "\",";
    std::cout << "\"claim_class\":\"component_integer_dot_full_q4k_real_gguf_gateup_q4_down_and_device_q8_pipeline_exactness\",";
    std::cout << "\"speedup_claims_allowed\":false";
    std::cout << "}" << std::endl;

    if (q4_down_available) {
      api.clReleaseMemObject(shared_down_fused_device_q8_tokenreuse_mem);
      api.clReleaseMemObject(shared_down_q8_from_fused_mem);
      api.clReleaseMemObject(shared_swiglu_fusion_mem);
      api.clReleaseMemObject(shared_down_device_q8_tokenreuse_mem);
      api.clReleaseMemObject(shared_down_q8_from_gpu_mem);
      api.clReleaseMemObject(shared_down_tokenreuse_mem);
      api.clReleaseMemObject(shared_down_q8_mem);
      api.clReleaseMemObject(shared_down_rows_mem);
      api.clReleaseMemObject(selected_down_fused_device_q8_tokenreuse_mem);
      api.clReleaseMemObject(selected_down_q8_from_fused_mem);
      api.clReleaseMemObject(selected_swiglu_fusion_mem);
      api.clReleaseMemObject(selected_down_device_q8_tokenreuse_mem);
      api.clReleaseMemObject(selected_down_q8_from_gpu_mem);
      api.clReleaseMemObject(selected_down_tokenreuse_mem);
      api.clReleaseMemObject(selected_down_q8_mem);
      api.clReleaseMemObject(selected_down_rows_mem);
    }
    api.clReleaseMemObject(shared_tokenreuse_mem);
    api.clReleaseMemObject(real_tokenreuse_mem);
    api.clReleaseMemObject(shared_scalar_mem);
    api.clReleaseMemObject(shared_dpas_mem);
    api.clReleaseMemObject(shared_rows_mem);
    api.clReleaseMemObject(real_scalar_mem);
    api.clReleaseMemObject(real_dpas_mem);
    api.clReleaseMemObject(real_q8_mem);
    api.clReleaseMemObject(real_rows_mem);
    api.clReleaseMemObject(q4k_scalar_mem);
    api.clReleaseMemObject(q4k_dpas_mem);
    api.clReleaseMemObject(q8k_mem);
    api.clReleaseMemObject(q4k_rows_mem);
    api.clReleaseMemObject(scalar_mem);
    api.clReleaseMemObject(dpas_mem);
    api.clReleaseMemObject(q8_mem);
    api.clReleaseMemObject(rows_mem);
    api.clReleaseKernel(q8_values_prep_kernel);
    api.clReleaseKernel(q8_prep_kernel);
    api.clReleaseKernel(swiglu_fusion_localq8_kernel);
    api.clReleaseKernel(swiglu_fusion_kernel);
    api.clReleaseKernel(q4k_tokenreuse_kernel);
    api.clReleaseKernel(q4k_kernel);
    api.clReleaseKernel(kernel);
    api.clReleaseProgram(program);
    api.clReleaseCommandQueue(queue);
    api.clReleaseContext(context);
    return (dpas_cpu_mismatches == 0 &&
            dpas_scalar_mismatches == 0 &&
            q4k_dpas_cpu_mismatches == 0 &&
            q4k_dpas_scalar_mismatches == 0 &&
            real_cpu_matches_oracle &&
            real_dpas_matches_cpu &&
            real_dpas_matches_oracle &&
            real_scalar_matches_cpu &&
            real_tile_dpas_matches_cpu &&
            real_tile_scalar_matches_cpu &&
            shared_tile_dpas_matches_cpu &&
            shared_tile_scalar_matches_cpu &&
            real_tokenreuse_all_match &&
            shared_tokenreuse_all_match &&
            (!q4_down_available ||
             (selected_down_cpu_matches_oracle &&
              shared_down_cpu_matches_oracle &&
              selected_down_tokenreuse_all_match &&
              shared_down_tokenreuse_all_match &&
              selected_down_q8_prep_result.kernel_min_us > 0.0 &&
              shared_down_q8_prep_result.kernel_min_us > 0.0 &&
              selected_down_device_q8_tokenreuse_all_match &&
              shared_down_device_q8_tokenreuse_all_match &&
              selected_swiglu_fusion_all_match &&
              shared_swiglu_fusion_all_match &&
              selected_swiglu_fusion_localq8_all_match &&
              shared_swiglu_fusion_localq8_all_match &&
              selected_down_fused_q8_prep_result.kernel_min_us > 0.0 &&
              shared_down_fused_q8_prep_result.kernel_min_us > 0.0 &&
              selected_down_fused_device_q8_tokenreuse_all_match &&
              shared_down_fused_device_q8_tokenreuse_all_match))) ? 0 : 1;
  } catch (const std::exception& exc) {
    std::cerr << "error: " << exc.what() << std::endl;
    return 2;
  }
}
'''


def utc_stamp() -> str:
  return dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def iso_now() -> str:
  return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def shell_join(argv: list[str]) -> str:
  return " ".join(shlex.quote(item) for item in argv)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--target", "--host", dest="host", metavar="TARGET", default=DEFAULT_HOST)
  parser.add_argument("--model", default=DEFAULT_MODEL)
  parser.add_argument("--env-script", default=DEFAULT_ENV_SCRIPT)
  parser.add_argument("--staging-root", "--remote-root", dest="remote_root", metavar="PATH", default=DEFAULT_REMOTE_ROOT)
  parser.add_argument("--layer", type=int, default=5)
  parser.add_argument("--real-tokens", type=int, default=4)
  parser.add_argument("--repeat", type=int, default=11)
  parser.add_argument("--device-substring", default="B390")
  parser.add_argument("--timeout-s", type=int, default=240)
  parser.add_argument("--out-dir", type=Path)
  return parser.parse_args()


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


def resolve_payloads(layer: int) -> dict[str, dict[str, Any]]:
  payloads: dict[str, dict[str, Any]] = {}
  for name, (stage_name, pattern, size_bytes) in PAYLOAD_SPECS.items():
    matches = sorted(PAYLOAD_ROOT.glob(pattern.format(layer=layer)))
    if not matches:
      raise SystemExit(
          f"DPAS real-GGUF payload missing: {PAYLOAD_ROOT / pattern.format(layer=layer)}"
      )
    if len(matches) != 1:
      candidates = ", ".join(str(path.relative_to(ROOT)) for path in matches)
      raise SystemExit(
          f"DPAS real-GGUF payload pattern was not unique for {name}: {candidates}"
      )
    path = matches[0].resolve()
    if path.stat().st_size != size_bytes:
      raise SystemExit(f"DPAS real-GGUF payload size mismatch: {path}")
    payloads[name] = {
        "local_path": path,
        "path": str(path.relative_to(ROOT)),
        "sha256": iq36_local.sha256_file(path),
        "size_bytes": size_bytes,
        "stage_name": stage_name,
    }
  return payloads


def nested_bool(obj: dict[str, Any] | None, *keys: str) -> bool:
  current: Any = obj
  for key in keys:
    if not isinstance(current, dict):
      return False
    current = current.get(key)
  return current is True


def nested_number(obj: dict[str, Any] | None, *keys: str) -> float | None:
  current: Any = obj
  for key in keys:
    if not isinstance(current, dict):
      return None
    current = current.get(key)
  return float(current) if isinstance(current, (int, float)) else None


def write_summary(path: Path, payload: dict[str, Any]) -> None:
  probe = payload.get("probe") if isinstance(payload.get("probe"), dict) else {}
  real_comparisons = (
      probe.get("real_comparisons", {}) if isinstance(probe.get("real_comparisons"), dict) else {}
  )
  shared_comparisons = (
      probe.get("shared_comparisons", {}) if isinstance(probe.get("shared_comparisons"), dict) else {}
  )
  selected_down_comparisons = (
      probe.get("selected_down_comparisons", {})
      if isinstance(probe.get("selected_down_comparisons"), dict) else {}
  )
  shared_down_comparisons = (
      probe.get("shared_down_comparisons", {})
      if isinstance(probe.get("shared_down_comparisons"), dict) else {}
  )
  real_tokenreuse = (
      probe.get("real_tokenreuse_results", [])
      if isinstance(probe.get("real_tokenreuse_results"), list) else []
  )
  shared_tokenreuse = (
      probe.get("shared_tokenreuse_results", [])
      if isinstance(probe.get("shared_tokenreuse_results"), list) else []
  )
  selected_down_tokenreuse = (
      probe.get("selected_down_tokenreuse_results", [])
      if isinstance(probe.get("selected_down_tokenreuse_results"), list) else []
  )
  shared_down_tokenreuse = (
      probe.get("shared_down_tokenreuse_results", [])
      if isinstance(probe.get("shared_down_tokenreuse_results"), list) else []
  )
  selected_down_device_q8_tokenreuse = (
      probe.get("selected_down_device_q8_tokenreuse_results", [])
      if isinstance(probe.get("selected_down_device_q8_tokenreuse_results"), list) else []
  )
  shared_down_device_q8_tokenreuse = (
      probe.get("shared_down_device_q8_tokenreuse_results", [])
      if isinstance(probe.get("shared_down_device_q8_tokenreuse_results"), list) else []
  )
  selected_swiglu_fusion = (
      probe.get("selected_swiglu_fusion_results", [])
      if isinstance(probe.get("selected_swiglu_fusion_results"), list) else []
  )
  selected_swiglu_fusion_localq8 = (
      probe.get("selected_swiglu_fusion_localq8_results", [])
      if isinstance(probe.get("selected_swiglu_fusion_localq8_results"), list) else []
  )
  shared_swiglu_fusion = (
      probe.get("shared_swiglu_fusion_results", [])
      if isinstance(probe.get("shared_swiglu_fusion_results"), list) else []
  )
  shared_swiglu_fusion_localq8 = (
      probe.get("shared_swiglu_fusion_localq8_results", [])
      if isinstance(probe.get("shared_swiglu_fusion_localq8_results"), list) else []
  )
  selected_down_fused_device_q8_tokenreuse = (
      probe.get("selected_down_fused_device_q8_tokenreuse_results", [])
      if isinstance(probe.get("selected_down_fused_device_q8_tokenreuse_results"), list) else []
  )
  shared_down_fused_device_q8_tokenreuse = (
      probe.get("shared_down_fused_device_q8_tokenreuse_results", [])
      if isinstance(probe.get("shared_down_fused_device_q8_tokenreuse_results"), list) else []
  )
  selected_down_q8_prep = (
      probe.get("selected_down_q8_prep", {})
      if isinstance(probe.get("selected_down_q8_prep"), dict) else {}
  )
  shared_down_q8_prep = (
      probe.get("shared_down_q8_prep", {})
      if isinstance(probe.get("shared_down_q8_prep"), dict) else {}
  )
  selected_down_fused_q8_prep = (
      probe.get("selected_down_fused_q8_prep", {})
      if isinstance(probe.get("selected_down_fused_q8_prep"), dict) else {}
  )
  shared_down_fused_q8_prep = (
      probe.get("shared_down_fused_q8_prep", {})
      if isinstance(probe.get("shared_down_fused_q8_prep"), dict) else {}
  )
  lines = [
      "# GPU DPAS Q4 Exact Gate",
      "",
      f"- workstream: `{WORKSTREAM}`",
      f"- host: `{payload.get('host')}`",
      f"- model: `{payload.get('model')}`",
      f"- layer: `{payload.get('layer')}`",
      f"- required checks passed: `{str(payload.get('required_checks_passed')).lower()}`",
      "- speedup claims allowed: `false`",
      f"- platform/device: `{probe.get('platform_name')}` / `{probe.get('device_name')}`",
      f"- dot products: `{probe.get('dot_product_count')}` x width `{probe.get('dot_width')}`",
      f"- DPAS matches CPU: `{str(probe.get('dpas_matches_cpu')).lower()}`",
      f"- DPAS matches scalar OpenCL: `{str(probe.get('dpas_matches_scalar')).lower()}`",
      f"- full Q4_K tile: `{probe.get('q4k_full_output_count')}` outputs "
      f"({probe.get('q4k_full_rows')} rows x {probe.get('q4k_full_tokens')} tokens)",
      f"- full Q4_K DPAS matches CPU: `{str(probe.get('q4k_full_dpas_matches_cpu')).lower()}`",
      f"- full Q4_K max abs: `{probe.get('q4k_full_max_abs_dpas_cpu')}`",
      f"- real GGUF tensor: `{probe.get('real_tensor_name')}` selected rows `{probe.get('real_selected_rows')}`",
      f"- real GGUF tile: `{probe.get('real_tile_output_count')}` outputs "
      f"({probe.get('real_tile_tokens')} tokens)",
      f"- real expert ids: `{probe.get('real_expert_ids')}`",
      f"- real DPAS matches oracle: `{str(probe.get('real_dpas_matches_oracle')).lower()}`",
      f"- real tile DPAS matches CPU: `{str(probe.get('real_tile_dpas_matches_cpu')).lower()}`",
      f"- shared GGUF tensors: `{probe.get('shared_gate_tensor_name')}` + `{probe.get('shared_up_tensor_name')}`",
      f"- shared GGUF tile: `{probe.get('shared_tile_output_count')}` outputs "
      f"({probe.get('shared_tile_tokens')} tokens)",
      f"- shared tile DPAS matches CPU: `{str(probe.get('shared_tile_dpas_matches_cpu')).lower()}`",
      f"- real token-reuse all match CPU: `{str(probe.get('real_tokenreuse_all_match_cpu')).lower()}`",
      f"- real token-reuse best: `{probe.get('real_tokenreuse_best_kernel_min_us')}` us "
      f"(block `{probe.get('real_tokenreuse_best_token_block')}`)",
      f"- shared token-reuse all match CPU: `{str(probe.get('shared_tokenreuse_all_match_cpu')).lower()}`",
      f"- shared token-reuse best: `{probe.get('shared_tokenreuse_best_kernel_min_us')}` us "
      f"(block `{probe.get('shared_tokenreuse_best_token_block')}`)",
      f"- Q4 down available: `{str(probe.get('q4_down_available')).lower()}`",
      f"- selected down tensor: `{probe.get('selected_down_tensor_name')}` "
      f"rows `{probe.get('selected_down_rows')}`",
      f"- selected down CPU matches oracle: `{str(probe.get('selected_down_cpu_matches_oracle')).lower()}`",
      f"- selected down token-reuse all match CPU: "
      f"`{str(probe.get('selected_down_tokenreuse_all_match_cpu')).lower()}`",
      f"- selected down token-reuse best: "
      f"`{probe.get('selected_down_tokenreuse_best_kernel_min_us')}` us "
      f"(block `{probe.get('selected_down_tokenreuse_best_token_block')}`)",
      f"- selected down GPU Q8 prep: `{selected_down_q8_prep.get('kernel_min_us')}` us, "
      f"byte-exact `{str(selected_down_q8_prep.get('q8_matches_cpu')).lower()}`, "
      f"mismatches `{selected_down_q8_prep.get('q8_byte_mismatch_count')}`",
      f"- selected down device-Q8 token-reuse all match CPU: "
      f"`{str(probe.get('selected_down_device_q8_tokenreuse_all_match_cpu')).lower()}`",
      f"- selected down device-Q8 token-reuse best: "
      f"`{probe.get('selected_down_device_q8_tokenreuse_best_kernel_min_us')}` us "
      f"(block `{probe.get('selected_down_device_q8_tokenreuse_best_token_block')}`)",
      f"- selected fused gate/up->SwiGLU all match CPU: "
      f"`{str(probe.get('selected_swiglu_fusion_all_match_cpu')).lower()}`",
      f"- selected fused gate/up->SwiGLU best: "
      f"`{probe.get('selected_swiglu_fusion_best_kernel_min_us')}` us "
      f"(block `{probe.get('selected_swiglu_fusion_best_token_block')}`)",
      f"- selected fused local-Q8 gate/up->SwiGLU all match CPU: "
      f"`{str(probe.get('selected_swiglu_fusion_localq8_all_match_cpu')).lower()}`",
      f"- selected fused local-Q8 gate/up->SwiGLU best: "
      f"`{probe.get('selected_swiglu_fusion_localq8_best_kernel_min_us')}` us "
      f"(block `{probe.get('selected_swiglu_fusion_localq8_best_token_block')}`, "
      f"local `{probe.get('selected_swiglu_fusion_localq8_best_local_work_items')}`)",
      f"- selected fused SwiGLU Q8 prep: `{selected_down_fused_q8_prep.get('kernel_min_us')}` us, "
      f"byte-exact `{str(selected_down_fused_q8_prep.get('q8_matches_cpu')).lower()}`, "
      f"mismatches `{selected_down_fused_q8_prep.get('q8_byte_mismatch_count')}`",
      f"- selected fused device-Q8 down all match CPU: "
      f"`{str(probe.get('selected_down_fused_device_q8_tokenreuse_all_match_cpu')).lower()}`",
      f"- selected fused device-Q8 down best: "
      f"`{probe.get('selected_down_fused_device_q8_tokenreuse_best_kernel_min_us')}` us "
      f"(block `{probe.get('selected_down_fused_device_q8_tokenreuse_best_token_block')}`)",
      f"- shared down tensor: `{probe.get('shared_down_tensor_name')}` "
      f"rows `{probe.get('shared_down_rows')}`",
      f"- shared down CPU matches oracle: `{str(probe.get('shared_down_cpu_matches_oracle')).lower()}`",
      f"- shared down token-reuse all match CPU: "
      f"`{str(probe.get('shared_down_tokenreuse_all_match_cpu')).lower()}`",
      f"- shared down token-reuse best: "
      f"`{probe.get('shared_down_tokenreuse_best_kernel_min_us')}` us "
      f"(block `{probe.get('shared_down_tokenreuse_best_token_block')}`)",
      f"- shared down GPU Q8 prep: `{shared_down_q8_prep.get('kernel_min_us')}` us, "
      f"byte-exact `{str(shared_down_q8_prep.get('q8_matches_cpu')).lower()}`, "
      f"mismatches `{shared_down_q8_prep.get('q8_byte_mismatch_count')}`",
      f"- shared down device-Q8 token-reuse all match CPU: "
      f"`{str(probe.get('shared_down_device_q8_tokenreuse_all_match_cpu')).lower()}`",
      f"- shared down device-Q8 token-reuse best: "
      f"`{probe.get('shared_down_device_q8_tokenreuse_best_kernel_min_us')}` us "
      f"(block `{probe.get('shared_down_device_q8_tokenreuse_best_token_block')}`)",
      f"- shared fused gate/up->SwiGLU all match CPU: "
      f"`{str(probe.get('shared_swiglu_fusion_all_match_cpu')).lower()}`",
      f"- shared fused gate/up->SwiGLU best: "
      f"`{probe.get('shared_swiglu_fusion_best_kernel_min_us')}` us "
      f"(block `{probe.get('shared_swiglu_fusion_best_token_block')}`)",
      f"- shared fused local-Q8 gate/up->SwiGLU all match CPU: "
      f"`{str(probe.get('shared_swiglu_fusion_localq8_all_match_cpu')).lower()}`",
      f"- shared fused local-Q8 gate/up->SwiGLU best: "
      f"`{probe.get('shared_swiglu_fusion_localq8_best_kernel_min_us')}` us "
      f"(block `{probe.get('shared_swiglu_fusion_localq8_best_token_block')}`, "
      f"local `{probe.get('shared_swiglu_fusion_localq8_best_local_work_items')}`)",
      f"- shared fused SwiGLU Q8 prep: `{shared_down_fused_q8_prep.get('kernel_min_us')}` us, "
      f"byte-exact `{str(shared_down_fused_q8_prep.get('q8_matches_cpu')).lower()}`, "
      f"mismatches `{shared_down_fused_q8_prep.get('q8_byte_mismatch_count')}`",
      f"- shared fused device-Q8 down all match CPU: "
      f"`{str(probe.get('shared_down_fused_device_q8_tokenreuse_all_match_cpu')).lower()}`",
      f"- shared fused device-Q8 down best: "
      f"`{probe.get('shared_down_fused_device_q8_tokenreuse_best_kernel_min_us')}` us "
      f"(block `{probe.get('shared_down_fused_device_q8_tokenreuse_best_token_block')}`)",
      f"- kernel min us: `{probe.get('kernel_min_us')}`",
      f"- full Q4_K kernel min us: `{probe.get('q4k_full_kernel_min_us')}`",
      f"- real GGUF kernel min us: `{probe.get('real_kernel_min_us')}`",
      f"- shared GGUF kernel min us: `{probe.get('shared_kernel_min_us')}`",
      "",
      "| real comparison | max abs | RMSE |",
      "|---|---:|---:|",
  ]
  for lane, cmp in [
      ("cpu_vs_oracle", real_comparisons.get("cpu_vs_oracle", {})),
      ("dpas_vs_cpu", real_comparisons.get("dpas_vs_cpu", {})),
      ("dpas_vs_oracle", real_comparisons.get("dpas_vs_oracle", {})),
      ("scalar_vs_cpu", real_comparisons.get("scalar_vs_cpu", {})),
      ("tile_dpas_vs_cpu", real_comparisons.get("tile_dpas_vs_cpu", {})),
      ("tile_scalar_vs_cpu", real_comparisons.get("tile_scalar_vs_cpu", {})),
  ]:
    cmp = cmp if isinstance(cmp, dict) else {}
    lines.append(f"| {lane} | {cmp.get('max_abs_diff')} | {cmp.get('rmse')} |")
  lines += [
      "",
      "| shared comparison | max abs | RMSE |",
      "|---|---:|---:|",
  ]
  for lane, cmp in [
      ("tile_dpas_vs_cpu", shared_comparisons.get("tile_dpas_vs_cpu", {})),
      ("tile_scalar_vs_cpu", shared_comparisons.get("tile_scalar_vs_cpu", {})),
  ]:
    cmp = cmp if isinstance(cmp, dict) else {}
    lines.append(f"| {lane} | {cmp.get('max_abs_diff')} | {cmp.get('rmse')} |")
  lines += [
      "",
      "| down comparison | max abs | RMSE |",
      "|---|---:|---:|",
  ]
  for lane, cmp in [
      ("selected_cpu_vs_oracle", selected_down_comparisons.get("cpu_vs_oracle", {})),
      ("shared_cpu_vs_oracle", shared_down_comparisons.get("cpu_vs_oracle", {})),
  ]:
    cmp = cmp if isinstance(cmp, dict) else {}
    lines.append(f"| {lane} | {cmp.get('max_abs_diff')} | {cmp.get('rmse')} |")
  lines += [
      "",
      "| token-reuse lane | block | kernel min us | matches CPU | max abs | RMSE |",
      "|---|---:|---:|---|---:|---:|",
  ]
  for lane_name, results in [
      ("real", real_tokenreuse),
      ("shared", shared_tokenreuse),
      ("selected_down", selected_down_tokenreuse),
      ("shared_down", shared_down_tokenreuse),
      ("selected_down_device_q8", selected_down_device_q8_tokenreuse),
      ("shared_down_device_q8", shared_down_device_q8_tokenreuse),
      ("selected_down_fused_device_q8", selected_down_fused_device_q8_tokenreuse),
      ("shared_down_fused_device_q8", shared_down_fused_device_q8_tokenreuse),
  ]:
    for result in results:
      result = result if isinstance(result, dict) else {}
      cmp = result.get("dpas_vs_cpu", {})
      cmp = cmp if isinstance(cmp, dict) else {}
      lines.append(
          f"| {lane_name} | {result.get('token_block')} | "
          f"{result.get('kernel_min_us')} | "
          f"{str(result.get('dpas_matches_cpu')).lower()} | "
          f"{cmp.get('max_abs_diff')} | {cmp.get('rmse')} |"
      )
  lines += [
      "",
      "| fused SwiGLU lane | block | local | kernel min us | matches CPU | max abs | RMSE |",
      "|---|---:|---:|---:|---|---:|---:|",
  ]
  for lane_name, results in [
      ("selected_swiglu_fusion", selected_swiglu_fusion),
      ("selected_swiglu_fusion_localq8", selected_swiglu_fusion_localq8),
      ("shared_swiglu_fusion", shared_swiglu_fusion),
      ("shared_swiglu_fusion_localq8", shared_swiglu_fusion_localq8),
  ]:
    for result in results:
      result = result if isinstance(result, dict) else {}
      cmp = result.get("output_vs_cpu", {})
      cmp = cmp if isinstance(cmp, dict) else {}
      lines.append(
          f"| {lane_name} | {result.get('token_block')} | "
          f"{result.get('local_work_items')} | "
          f"{result.get('kernel_min_us')} | "
          f"{str(result.get('output_matches_cpu')).lower()} | "
          f"{cmp.get('max_abs_diff')} | {cmp.get('rmse')} |"
      )
  lines += [
      "",
      "This component gate checks the exact integer-dot mapping, a synthetic",
      "full-Q4_K scale/min tile, one real GGUF selected gate/up oracle vector,",
      "a selected multi-token real-GGUF CPU-reference tile, and a shared gate/up",
      "multi-token real-GGUF CPU-reference tile. When the selected layer has Q4_K",
      "down tensors, it also checks selected and shared down token-reuse lanes",
      "against CPU and token-15 oracles. The token-reuse lane reuses a",
      "row tile across a small token block to test a non-rowlane DPAS tiling",
      "direction. It does not prove full prefill throughput or model speed.",
      "The device-Q8 rows add a bounded GPU SwiGLU plus Q8-prep pipeline check",
      "from gate/up DPAS outputs into down DPAS token reuse.",
      "The fused rows compute gate/up DPAS pairs directly into SwiGLU values,",
      "then run Q8 prep and down DPAS from that fused intermediate.",
      "The local-Q8 fused rows put multiple subgroups in one workgroup and",
      "stage each token/block Q8_K activation in local memory so adjacent row",
      "tiles share the activation load; this is the storage/work-distribution",
      "probe for the selected gate/up-to-SwiGLU DPAS branch.",
      "",
  ]
  path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  args = parse_args()
  stamp = utc_stamp()
  created_at = iso_now()
  out_dir = args.out_dir or ROOT / f"output/gpu-dpas-q4-exact-gate-{stamp}"
  if not out_dir.is_absolute():
    out_dir = ROOT / out_dir
  out_dir.mkdir(parents=True, exist_ok=False)
  raw_dir = out_dir / "raw"
  raw_dir.mkdir()
  payloads = resolve_payloads(args.layer)
  local_cpp = out_dir / "gpu_dpas_q4_exact_gate.cpp"
  local_cpp.write_text(PROBE_CPP, encoding="utf-8")

  remote_dir = f"{args.remote_root.rstrip('/')}/gpu-dpas-q4-exact-gate-{stamp}"
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
    transfers.append(iq36_local.copy_to(args.host, local_cpp, f"{remote_dir}/tests/gpu_dpas_q4_exact_gate.cpp", args.timeout_s))
    for name, payload_spec in payloads.items():
      payload_transfers[name] = iq36_local.copy_to(
          args.host,
          payload_spec["local_path"],
          f"{remote_payload_dir}/{payload_spec['stage_name']}",
          args.timeout_s,
      )
  executable = f"{remote_dir}/build/iq36-gpu-dpas-q4-exact-gate"
  compile_cmd = " && ".join([
      f"source {shlex.quote(args.env_script)} >/dev/null 2>&1",
      (
          "g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic "
          f"-I {shlex.quote(remote_dir + '/include')} "
          f"{shlex.quote(remote_dir + '/src/gguf_loader.cpp')} "
          f"{shlex.quote(remote_dir + '/tests/gpu_dpas_q4_exact_gate.cpp')} "
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
      "--real-tokens", str(args.real_tokens),
      "--repeat", str(args.repeat),
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
      {"name": "dpas_matches_cpu", "pass": nested_bool(probe, "dpas_matches_cpu")},
      {"name": "dpas_matches_scalar_opencl", "pass": nested_bool(probe, "dpas_matches_scalar")},
      {"name": "q4k_full_dpas_matches_cpu", "pass": nested_bool(probe, "q4k_full_dpas_matches_cpu")},
      {"name": "q4k_full_dpas_matches_scalar_opencl", "pass": nested_bool(probe, "q4k_full_dpas_matches_scalar")},
      {"name": "real_cpu_matches_oracle", "pass": nested_bool(probe, "real_cpu_matches_oracle")},
      {"name": "real_dpas_matches_cpu", "pass": nested_bool(probe, "real_dpas_matches_cpu")},
      {"name": "real_dpas_matches_oracle", "pass": nested_bool(probe, "real_dpas_matches_oracle")},
      {"name": "real_scalar_matches_cpu", "pass": nested_bool(probe, "real_scalar_matches_cpu")},
      {"name": "real_tile_dpas_matches_cpu", "pass": nested_bool(probe, "real_tile_dpas_matches_cpu")},
      {"name": "real_tile_scalar_matches_cpu", "pass": nested_bool(probe, "real_tile_scalar_matches_cpu")},
      {"name": "shared_tile_dpas_matches_cpu", "pass": nested_bool(probe, "shared_tile_dpas_matches_cpu")},
      {"name": "shared_tile_scalar_matches_cpu", "pass": nested_bool(probe, "shared_tile_scalar_matches_cpu")},
      {"name": "real_tokenreuse_all_match_cpu", "pass": nested_bool(probe, "real_tokenreuse_all_match_cpu")},
      {"name": "shared_tokenreuse_all_match_cpu", "pass": nested_bool(probe, "shared_tokenreuse_all_match_cpu")},
      {"name": "selected_down_cpu_matches_oracle_if_q4", "pass": (not nested_bool(probe, "q4_down_available")) or nested_bool(probe, "selected_down_cpu_matches_oracle")},
      {"name": "shared_down_cpu_matches_oracle_if_q4", "pass": (not nested_bool(probe, "q4_down_available")) or nested_bool(probe, "shared_down_cpu_matches_oracle")},
      {"name": "selected_down_tokenreuse_all_match_cpu_if_q4", "pass": (not nested_bool(probe, "q4_down_available")) or nested_bool(probe, "selected_down_tokenreuse_all_match_cpu")},
      {"name": "shared_down_tokenreuse_all_match_cpu_if_q4", "pass": (not nested_bool(probe, "q4_down_available")) or nested_bool(probe, "shared_down_tokenreuse_all_match_cpu")},
      {"name": "selected_down_device_q8_tokenreuse_all_match_cpu_if_q4", "pass": (not nested_bool(probe, "q4_down_available")) or nested_bool(probe, "selected_down_device_q8_tokenreuse_all_match_cpu")},
      {"name": "shared_down_device_q8_tokenreuse_all_match_cpu_if_q4", "pass": (not nested_bool(probe, "q4_down_available")) or nested_bool(probe, "shared_down_device_q8_tokenreuse_all_match_cpu")},
      {"name": "selected_swiglu_fusion_all_match_cpu_if_q4", "pass": (not nested_bool(probe, "q4_down_available")) or nested_bool(probe, "selected_swiglu_fusion_all_match_cpu")},
      {"name": "shared_swiglu_fusion_all_match_cpu_if_q4", "pass": (not nested_bool(probe, "q4_down_available")) or nested_bool(probe, "shared_swiglu_fusion_all_match_cpu")},
      {"name": "selected_swiglu_fusion_localq8_all_match_cpu_if_q4", "pass": (not nested_bool(probe, "q4_down_available")) or nested_bool(probe, "selected_swiglu_fusion_localq8_all_match_cpu")},
      {"name": "shared_swiglu_fusion_localq8_all_match_cpu_if_q4", "pass": (not nested_bool(probe, "q4_down_available")) or nested_bool(probe, "shared_swiglu_fusion_localq8_all_match_cpu")},
      {"name": "selected_down_fused_device_q8_tokenreuse_all_match_cpu_if_q4", "pass": (not nested_bool(probe, "q4_down_available")) or nested_bool(probe, "selected_down_fused_device_q8_tokenreuse_all_match_cpu")},
      {"name": "shared_down_fused_device_q8_tokenreuse_all_match_cpu_if_q4", "pass": (not nested_bool(probe, "q4_down_available")) or nested_bool(probe, "shared_down_fused_device_q8_tokenreuse_all_match_cpu")},
      {"name": "gpu_event_timing_positive", "pass": bool(nested_number(probe, "kernel_min_us") and nested_number(probe, "kernel_min_us") > 0.0)},
      {"name": "q4k_full_gpu_event_timing_positive", "pass": bool(nested_number(probe, "q4k_full_kernel_min_us") and nested_number(probe, "q4k_full_kernel_min_us") > 0.0)},
      {"name": "real_gpu_event_timing_positive", "pass": bool(nested_number(probe, "real_kernel_min_us") and nested_number(probe, "real_kernel_min_us") > 0.0)},
      {"name": "shared_gpu_event_timing_positive", "pass": bool(nested_number(probe, "shared_kernel_min_us") and nested_number(probe, "shared_kernel_min_us") > 0.0)},
      {"name": "real_tokenreuse_gpu_event_timing_positive", "pass": bool(nested_number(probe, "real_tokenreuse_best_kernel_min_us") and nested_number(probe, "real_tokenreuse_best_kernel_min_us") > 0.0)},
      {"name": "shared_tokenreuse_gpu_event_timing_positive", "pass": bool(nested_number(probe, "shared_tokenreuse_best_kernel_min_us") and nested_number(probe, "shared_tokenreuse_best_kernel_min_us") > 0.0)},
      {"name": "selected_down_tokenreuse_gpu_event_timing_positive_if_q4", "pass": (not nested_bool(probe, "q4_down_available")) or bool(nested_number(probe, "selected_down_tokenreuse_best_kernel_min_us") and nested_number(probe, "selected_down_tokenreuse_best_kernel_min_us") > 0.0)},
      {"name": "shared_down_tokenreuse_gpu_event_timing_positive_if_q4", "pass": (not nested_bool(probe, "q4_down_available")) or bool(nested_number(probe, "shared_down_tokenreuse_best_kernel_min_us") and nested_number(probe, "shared_down_tokenreuse_best_kernel_min_us") > 0.0)},
      {"name": "selected_down_q8_prep_gpu_event_timing_positive_if_q4", "pass": (not nested_bool(probe, "q4_down_available")) or bool(nested_number(probe, "selected_down_q8_prep", "kernel_min_us") and nested_number(probe, "selected_down_q8_prep", "kernel_min_us") > 0.0)},
      {"name": "shared_down_q8_prep_gpu_event_timing_positive_if_q4", "pass": (not nested_bool(probe, "q4_down_available")) or bool(nested_number(probe, "shared_down_q8_prep", "kernel_min_us") and nested_number(probe, "shared_down_q8_prep", "kernel_min_us") > 0.0)},
      {"name": "selected_down_device_q8_tokenreuse_gpu_event_timing_positive_if_q4", "pass": (not nested_bool(probe, "q4_down_available")) or bool(nested_number(probe, "selected_down_device_q8_tokenreuse_best_kernel_min_us") and nested_number(probe, "selected_down_device_q8_tokenreuse_best_kernel_min_us") > 0.0)},
      {"name": "shared_down_device_q8_tokenreuse_gpu_event_timing_positive_if_q4", "pass": (not nested_bool(probe, "q4_down_available")) or bool(nested_number(probe, "shared_down_device_q8_tokenreuse_best_kernel_min_us") and nested_number(probe, "shared_down_device_q8_tokenreuse_best_kernel_min_us") > 0.0)},
      {"name": "selected_swiglu_fusion_gpu_event_timing_positive_if_q4", "pass": (not nested_bool(probe, "q4_down_available")) or bool(nested_number(probe, "selected_swiglu_fusion_best_kernel_min_us") and nested_number(probe, "selected_swiglu_fusion_best_kernel_min_us") > 0.0)},
      {"name": "shared_swiglu_fusion_gpu_event_timing_positive_if_q4", "pass": (not nested_bool(probe, "q4_down_available")) or bool(nested_number(probe, "shared_swiglu_fusion_best_kernel_min_us") and nested_number(probe, "shared_swiglu_fusion_best_kernel_min_us") > 0.0)},
      {"name": "selected_swiglu_fusion_localq8_gpu_event_timing_positive_if_q4", "pass": (not nested_bool(probe, "q4_down_available")) or bool(nested_number(probe, "selected_swiglu_fusion_localq8_best_kernel_min_us") and nested_number(probe, "selected_swiglu_fusion_localq8_best_kernel_min_us") > 0.0)},
      {"name": "shared_swiglu_fusion_localq8_gpu_event_timing_positive_if_q4", "pass": (not nested_bool(probe, "q4_down_available")) or bool(nested_number(probe, "shared_swiglu_fusion_localq8_best_kernel_min_us") and nested_number(probe, "shared_swiglu_fusion_localq8_best_kernel_min_us") > 0.0)},
      {"name": "selected_down_fused_q8_prep_gpu_event_timing_positive_if_q4", "pass": (not nested_bool(probe, "q4_down_available")) or bool(nested_number(probe, "selected_down_fused_q8_prep", "kernel_min_us") and nested_number(probe, "selected_down_fused_q8_prep", "kernel_min_us") > 0.0)},
      {"name": "shared_down_fused_q8_prep_gpu_event_timing_positive_if_q4", "pass": (not nested_bool(probe, "q4_down_available")) or bool(nested_number(probe, "shared_down_fused_q8_prep", "kernel_min_us") and nested_number(probe, "shared_down_fused_q8_prep", "kernel_min_us") > 0.0)},
      {"name": "selected_down_fused_device_q8_tokenreuse_gpu_event_timing_positive_if_q4", "pass": (not nested_bool(probe, "q4_down_available")) or bool(nested_number(probe, "selected_down_fused_device_q8_tokenreuse_best_kernel_min_us") and nested_number(probe, "selected_down_fused_device_q8_tokenreuse_best_kernel_min_us") > 0.0)},
      {"name": "shared_down_fused_device_q8_tokenreuse_gpu_event_timing_positive_if_q4", "pass": (not nested_bool(probe, "q4_down_available")) or bool(nested_number(probe, "shared_down_fused_device_q8_tokenreuse_best_kernel_min_us") and nested_number(probe, "shared_down_fused_device_q8_tokenreuse_best_kernel_min_us") > 0.0)},
      {"name": "speedup_claims_forbidden", "pass": True},
  ]
  required_checks_passed = all(item["pass"] for item in checks)
  slim_payloads = {
      name: {key: value for key, value in payload_spec.items() if key != "local_path"}
      for name, payload_spec in payloads.items()
  }
  payload = {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "created_at": created_at,
      "host": args.host,
      "remote_dir": remote_dir,
      "model": args.model,
      "payloads": slim_payloads,
      "layer": args.layer,
      "real_tokens": args.real_tokens,
      "repeat": args.repeat,
      "probe": probe,
      "checks": checks,
      "required_checks_passed": required_checks_passed,
      "speedup_claims_allowed": False,
  }
  manifest = {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "created_at": created_at,
      "tool": "tools/intel-qwen36-gpu-dpas-q4-exact-gate.py",
      "artifact": str(out_dir.relative_to(ROOT)),
      "host": args.host,
      "remote_dir": remote_dir,
      "model": args.model,
      "payloads": slim_payloads,
      "layer": args.layer,
      "real_tokens": args.real_tokens,
      "required_checks_passed": required_checks_passed,
      "speedup_claims_allowed": False,
  }
  correctness = {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "checks": checks,
      "required_checks_passed": required_checks_passed,
      "claim_class": "component_integer_dot_full_q4k_real_gguf_gateup_q4_down_and_device_q8_pipeline_exactness",
      "speedup_claims_allowed": False,
  }
  iq36_local.write_json(out_dir / "probe.json", payload)
  iq36_local.write_json(out_dir / "manifest.json", manifest)
  iq36_local.write_json(out_dir / "correctness.json", correctness)
  iq36_local.write_metric(
      out_dir / "metrics.jsonl",
      "gpu_dpas_q4_exact_gate",
      [
          ("required_checks_passed", required_checks_passed),
          ("dpas_matches_cpu", nested_bool(probe, "dpas_matches_cpu")),
          ("dpas_matches_scalar", nested_bool(probe, "dpas_matches_scalar")),
          ("q4k_full_dpas_matches_cpu", nested_bool(probe, "q4k_full_dpas_matches_cpu")),
          ("q4k_full_dpas_matches_scalar", nested_bool(probe, "q4k_full_dpas_matches_scalar")),
          ("real_cpu_matches_oracle", nested_bool(probe, "real_cpu_matches_oracle")),
          ("real_dpas_matches_cpu", nested_bool(probe, "real_dpas_matches_cpu")),
          ("real_dpas_matches_oracle", nested_bool(probe, "real_dpas_matches_oracle")),
          ("real_scalar_matches_cpu", nested_bool(probe, "real_scalar_matches_cpu")),
          ("real_tile_dpas_matches_cpu", nested_bool(probe, "real_tile_dpas_matches_cpu")),
          ("real_tile_scalar_matches_cpu", nested_bool(probe, "real_tile_scalar_matches_cpu")),
          ("shared_tile_dpas_matches_cpu", nested_bool(probe, "shared_tile_dpas_matches_cpu")),
          ("shared_tile_scalar_matches_cpu", nested_bool(probe, "shared_tile_scalar_matches_cpu")),
          ("real_tokenreuse_all_match_cpu", nested_bool(probe, "real_tokenreuse_all_match_cpu")),
          ("shared_tokenreuse_all_match_cpu", nested_bool(probe, "shared_tokenreuse_all_match_cpu")),
          ("q4_down_available", nested_bool(probe, "q4_down_available")),
          ("selected_down_cpu_matches_oracle", nested_bool(probe, "selected_down_cpu_matches_oracle")),
          ("shared_down_cpu_matches_oracle", nested_bool(probe, "shared_down_cpu_matches_oracle")),
          ("selected_down_tokenreuse_all_match_cpu", nested_bool(probe, "selected_down_tokenreuse_all_match_cpu")),
          ("shared_down_tokenreuse_all_match_cpu", nested_bool(probe, "shared_down_tokenreuse_all_match_cpu")),
          ("kernel_min_us", nested_number(probe, "kernel_min_us")),
          ("q4k_full_kernel_min_us", nested_number(probe, "q4k_full_kernel_min_us")),
          ("real_kernel_min_us", nested_number(probe, "real_kernel_min_us")),
          ("shared_kernel_min_us", nested_number(probe, "shared_kernel_min_us")),
          ("real_tokenreuse_best_kernel_min_us", nested_number(probe, "real_tokenreuse_best_kernel_min_us")),
          ("real_tokenreuse_best_token_block", nested_number(probe, "real_tokenreuse_best_token_block")),
          ("shared_tokenreuse_best_kernel_min_us", nested_number(probe, "shared_tokenreuse_best_kernel_min_us")),
          ("shared_tokenreuse_best_token_block", nested_number(probe, "shared_tokenreuse_best_token_block")),
          ("selected_down_tokenreuse_best_kernel_min_us", nested_number(probe, "selected_down_tokenreuse_best_kernel_min_us")),
          ("selected_down_tokenreuse_best_token_block", nested_number(probe, "selected_down_tokenreuse_best_token_block")),
          ("shared_down_tokenreuse_best_kernel_min_us", nested_number(probe, "shared_down_tokenreuse_best_kernel_min_us")),
          ("shared_down_tokenreuse_best_token_block", nested_number(probe, "shared_down_tokenreuse_best_token_block")),
          ("selected_down_q8_prep_kernel_min_us", nested_number(probe, "selected_down_q8_prep", "kernel_min_us")),
          ("selected_down_q8_prep_q8_matches_cpu", nested_bool(probe, "selected_down_q8_prep", "q8_matches_cpu")),
          ("selected_down_q8_prep_byte_mismatch_count", nested_number(probe, "selected_down_q8_prep", "q8_byte_mismatch_count")),
          ("shared_down_q8_prep_kernel_min_us", nested_number(probe, "shared_down_q8_prep", "kernel_min_us")),
          ("shared_down_q8_prep_q8_matches_cpu", nested_bool(probe, "shared_down_q8_prep", "q8_matches_cpu")),
          ("shared_down_q8_prep_byte_mismatch_count", nested_number(probe, "shared_down_q8_prep", "q8_byte_mismatch_count")),
          ("selected_down_device_q8_tokenreuse_all_match_cpu", nested_bool(probe, "selected_down_device_q8_tokenreuse_all_match_cpu")),
          ("selected_down_device_q8_tokenreuse_best_kernel_min_us", nested_number(probe, "selected_down_device_q8_tokenreuse_best_kernel_min_us")),
          ("selected_down_device_q8_tokenreuse_best_token_block", nested_number(probe, "selected_down_device_q8_tokenreuse_best_token_block")),
          ("shared_down_device_q8_tokenreuse_all_match_cpu", nested_bool(probe, "shared_down_device_q8_tokenreuse_all_match_cpu")),
          ("shared_down_device_q8_tokenreuse_best_kernel_min_us", nested_number(probe, "shared_down_device_q8_tokenreuse_best_kernel_min_us")),
          ("shared_down_device_q8_tokenreuse_best_token_block", nested_number(probe, "shared_down_device_q8_tokenreuse_best_token_block")),
          ("selected_swiglu_fusion_all_match_cpu", nested_bool(probe, "selected_swiglu_fusion_all_match_cpu")),
          ("selected_swiglu_fusion_best_kernel_min_us", nested_number(probe, "selected_swiglu_fusion_best_kernel_min_us")),
          ("selected_swiglu_fusion_best_token_block", nested_number(probe, "selected_swiglu_fusion_best_token_block")),
          ("selected_swiglu_fusion_localq8_all_match_cpu", nested_bool(probe, "selected_swiglu_fusion_localq8_all_match_cpu")),
          ("selected_swiglu_fusion_localq8_best_kernel_min_us", nested_number(probe, "selected_swiglu_fusion_localq8_best_kernel_min_us")),
          ("selected_swiglu_fusion_localq8_best_token_block", nested_number(probe, "selected_swiglu_fusion_localq8_best_token_block")),
          ("selected_swiglu_fusion_localq8_best_local_work_items", nested_number(probe, "selected_swiglu_fusion_localq8_best_local_work_items")),
          ("shared_swiglu_fusion_all_match_cpu", nested_bool(probe, "shared_swiglu_fusion_all_match_cpu")),
          ("shared_swiglu_fusion_best_kernel_min_us", nested_number(probe, "shared_swiglu_fusion_best_kernel_min_us")),
          ("shared_swiglu_fusion_best_token_block", nested_number(probe, "shared_swiglu_fusion_best_token_block")),
          ("shared_swiglu_fusion_localq8_all_match_cpu", nested_bool(probe, "shared_swiglu_fusion_localq8_all_match_cpu")),
          ("shared_swiglu_fusion_localq8_best_kernel_min_us", nested_number(probe, "shared_swiglu_fusion_localq8_best_kernel_min_us")),
          ("shared_swiglu_fusion_localq8_best_token_block", nested_number(probe, "shared_swiglu_fusion_localq8_best_token_block")),
          ("shared_swiglu_fusion_localq8_best_local_work_items", nested_number(probe, "shared_swiglu_fusion_localq8_best_local_work_items")),
          ("selected_down_fused_q8_prep_kernel_min_us", nested_number(probe, "selected_down_fused_q8_prep", "kernel_min_us")),
          ("selected_down_fused_q8_prep_q8_matches_cpu", nested_bool(probe, "selected_down_fused_q8_prep", "q8_matches_cpu")),
          ("selected_down_fused_q8_prep_byte_mismatch_count", nested_number(probe, "selected_down_fused_q8_prep", "q8_byte_mismatch_count")),
          ("shared_down_fused_q8_prep_kernel_min_us", nested_number(probe, "shared_down_fused_q8_prep", "kernel_min_us")),
          ("shared_down_fused_q8_prep_q8_matches_cpu", nested_bool(probe, "shared_down_fused_q8_prep", "q8_matches_cpu")),
          ("shared_down_fused_q8_prep_byte_mismatch_count", nested_number(probe, "shared_down_fused_q8_prep", "q8_byte_mismatch_count")),
          ("selected_down_fused_device_q8_tokenreuse_all_match_cpu", nested_bool(probe, "selected_down_fused_device_q8_tokenreuse_all_match_cpu")),
          ("selected_down_fused_device_q8_tokenreuse_best_kernel_min_us", nested_number(probe, "selected_down_fused_device_q8_tokenreuse_best_kernel_min_us")),
          ("selected_down_fused_device_q8_tokenreuse_best_token_block", nested_number(probe, "selected_down_fused_device_q8_tokenreuse_best_token_block")),
          ("shared_down_fused_device_q8_tokenreuse_all_match_cpu", nested_bool(probe, "shared_down_fused_device_q8_tokenreuse_all_match_cpu")),
          ("shared_down_fused_device_q8_tokenreuse_best_kernel_min_us", nested_number(probe, "shared_down_fused_device_q8_tokenreuse_best_kernel_min_us")),
          ("shared_down_fused_device_q8_tokenreuse_best_token_block", nested_number(probe, "shared_down_fused_device_q8_tokenreuse_best_token_block")),
          ("q4k_full_max_abs_dpas_cpu", nested_number(probe, "q4k_full_max_abs_dpas_cpu")),
          ("real_tile_output_count", nested_number(probe, "real_tile_output_count")),
          ("shared_tile_output_count", nested_number(probe, "shared_tile_output_count")),
          ("selected_down_tile_output_count", nested_number(probe, "selected_down_tile_output_count")),
          ("shared_down_tile_output_count", nested_number(probe, "shared_down_tile_output_count")),
          ("real_dpas_vs_oracle_max_abs_diff", nested_number(probe, "real_comparisons", "dpas_vs_oracle", "max_abs_diff")),
          ("real_dpas_vs_oracle_rmse", nested_number(probe, "real_comparisons", "dpas_vs_oracle", "rmse")),
          ("real_dpas_vs_cpu_max_abs_diff", nested_number(probe, "real_comparisons", "dpas_vs_cpu", "max_abs_diff")),
          ("real_cpu_vs_oracle_max_abs_diff", nested_number(probe, "real_comparisons", "cpu_vs_oracle", "max_abs_diff")),
          ("real_tile_dpas_vs_cpu_max_abs_diff", nested_number(probe, "real_comparisons", "tile_dpas_vs_cpu", "max_abs_diff")),
          ("real_tile_dpas_vs_cpu_rmse", nested_number(probe, "real_comparisons", "tile_dpas_vs_cpu", "rmse")),
          ("shared_tile_dpas_vs_cpu_max_abs_diff", nested_number(probe, "shared_comparisons", "tile_dpas_vs_cpu", "max_abs_diff")),
          ("shared_tile_dpas_vs_cpu_rmse", nested_number(probe, "shared_comparisons", "tile_dpas_vs_cpu", "rmse")),
          ("selected_down_cpu_vs_oracle_max_abs_diff", nested_number(probe, "selected_down_comparisons", "cpu_vs_oracle", "max_abs_diff")),
          ("selected_down_cpu_vs_oracle_rmse", nested_number(probe, "selected_down_comparisons", "cpu_vs_oracle", "rmse")),
          ("shared_down_cpu_vs_oracle_max_abs_diff", nested_number(probe, "shared_down_comparisons", "cpu_vs_oracle", "max_abs_diff")),
          ("shared_down_cpu_vs_oracle_rmse", nested_number(probe, "shared_down_comparisons", "cpu_vs_oracle", "rmse")),
          ("synthetic_integer_gops_at_min", nested_number(probe, "synthetic_integer_gops_at_min")),
      ],
  )
  write_summary(out_dir / "summary.md", payload)
  print(out_dir)
  return 0 if required_checks_passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
