#!/usr/bin/env python3
"""Run the GPU selected-expert down handoff gate."""

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
SCHEMA_VERSION = "intel-qwen36-gpu-selected-down-probe-v9"
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
    ("engine/src/gpu_q4x8_matvec.cpp", "src/gpu_q4x8_matvec.cpp"),
]
PAYLOAD_SPECS = {
    "attn_post_norm": ("attn_post_norm.bin", "attn_post_norm-{layer}__tok15__ord209.bin", 8192),
    "ffn_moe_topk": ("ffn_moe_topk.bin", "ffn_moe_topk-{layer}__tok15__ord212.bin", 32),
    "ffn_moe_weights_norm": ("ffn_moe_weights_norm.bin", "ffn_moe_weights_norm-{layer}__tok15__ord214.bin", 32),
    "ffn_moe_gate_up": ("ffn_moe_gate_up.bin", "ffn_moe_gate_up-{layer}__tok15__ord215.bin", 32768),
    "ffn_moe_swiglu": ("ffn_moe_swiglu.bin", "ffn_moe_swiglu-{layer}__tok15__ord218.bin", 16384),
    "ffn_moe_down": ("ffn_moe_down.bin", "ffn_moe_down-{layer}__tok15__ord219.bin", 65536),
    "ffn_moe_weighted": ("ffn_moe_weighted.bin", "ffn_moe_weighted-{layer}__tok15__ord220.bin", 65536),
    "ffn_shexp": ("ffn_shexp.bin", "ffn_shexp-{layer}__tok15__ord222.bin", 8192),
    "attn_residual": ("attn_residual.bin", "attn_residual-{layer}__tok15__ord208.bin", 8192),
    "ffn_out": ("ffn_out.bin", "ffn_out-{layer}__tok15__ord226.bin", 8192),
}

SELECTED_Q4_PAIR2_OPENCL = r'''

__kernel void q4k_x8_matvec_rowlane_expert8_multiq8_pair2(
    __global const uchar* packed0,
    __global const uchar* packed1,
    __global const uchar* packed2,
    __global const uchar* packed3,
    __global const uchar* packed4,
    __global const uchar* packed5,
    __global const uchar* packed6,
    __global const uchar* packed7,
    __global const char* q8_qs,
    __global const short* q8_bsums,
    __global const float* q8_d,
    uint blocks_per_row,
    uint row_groups_per_expert,
    __global float* out) {
  const uint flat = (uint)get_global_id(0);
  const uint rows_per_expert = row_groups_per_expert * 8U;
  const uint pair = flat / rows_per_expert;
  const uint local_row = flat - pair * rows_per_expert;
  if (pair >= 4U || local_row >= rows_per_expert) {
    return;
  }
  const uint expert0 = pair * 2U;
  const uint expert1 = expert0 + 1U;
  const uint group = local_row >> 3;
  const uint j = local_row & 7U;
  __global const uchar* packed_a = packed0;
  __global const uchar* packed_b = packed1;
  if (expert0 == 2U) {
    packed_a = packed2;
    packed_b = packed3;
  } else if (expert0 == 4U) {
    packed_a = packed4;
    packed_b = packed5;
  } else if (expert0 == 6U) {
    packed_a = packed6;
    packed_b = packed7;
  }

  float sumf_a = 0.0f;
  float sum_minf_a = 0.0f;
  float sumf_b = 0.0f;
  float sum_minf_b = 0.0f;
  for (uint block_index = 0; block_index < blocks_per_row; ++block_index) {
    __global const uchar* block_a =
        packed_a + ((ulong)group * (ulong)blocks_per_row +
                    (ulong)block_index) * 1152UL;
    __global const uchar* block_b =
        packed_b + ((ulong)group * (ulong)blocks_per_row +
                    (ulong)block_index) * 1152UL;
    const ulong q8_block_a =
        (ulong)expert0 * (ulong)blocks_per_row + (ulong)block_index;
    const ulong q8_block_b =
        (ulong)expert1 * (ulong)blocks_per_row + (ulong)block_index;
    __global const char* q8_a = q8_qs + q8_block_a * 256UL;
    __global const char* q8_b = q8_qs + q8_block_b * 256UL;
    __global const short* bsums_a = q8_bsums + q8_block_a * 16UL;
    __global const short* bsums_b = q8_bsums + q8_block_b * 16UL;
    const float q8_scale_a = q8_d[q8_block_a];
    const float q8_scale_b = q8_d[q8_block_b];
    const float d_a = half_to_float(load_le16(block_a + j * 2)) * q8_scale_a;
    const float d_b = half_to_float(load_le16(block_b + j * 2)) * q8_scale_b;
    for (int k = 0; k < 16; ++k) {
      const int scale_pair = (k >> 2) * 2;
      const int q8_base = (k >> 2) * 64 + (k & 3) * 8;
      uchar scale0_a = 0;
      uchar min0_a = 0;
      uchar scale1_a = 0;
      uchar min1_a = 0;
      uchar scale0_b = 0;
      uchar min0_b = 0;
      uchar scale1_b = 0;
      uchar min1_b = 0;
      get_scale_min_k4((int)j, block_a + 32 + scale_pair * 12,
                       &scale0_a, &min0_a);
      get_scale_min_k4((int)j, block_a + 32 + (scale_pair + 1) * 12,
                       &scale1_a, &min1_a);
      get_scale_min_k4((int)j, block_b + 32 + scale_pair * 12,
                       &scale0_b, &min0_b);
      get_scale_min_k4((int)j, block_b + 32 + (scale_pair + 1) * 12,
                       &scale1_b, &min1_b);
      int sumi_a = 0;
      int sumi_b = 0;
      for (int i = 0; i < 8; ++i) {
        const uchar qa = block_a[128 + k * 64 + (int)j * 8 + i];
        const uchar qb = block_b[128 + k * 64 + (int)j * 8 + i];
        const int q8_low_a = (int)q8_a[q8_base + i];
        const int q8_high_a = (int)q8_a[q8_base + i + 32];
        const int q8_low_b = (int)q8_b[q8_base + i];
        const int q8_high_b = (int)q8_b[q8_base + i + 32];
        sumi_a += (int)(qa & (uchar)15) * q8_low_a * (int)scale0_a;
        sumi_a += (int)(qa >> 4) * q8_high_a * (int)scale1_a;
        sumi_b += (int)(qb & (uchar)15) * q8_low_b * (int)scale0_b;
        sumi_b += (int)(qb >> 4) * q8_high_b * (int)scale1_b;
      }
      sumf_a += (float)sumi_a * d_a;
      sumf_b += (float)sumi_b * d_b;
    }
    const float dmin_a =
        half_to_float(load_le16(block_a + 16 + j * 2)) * q8_scale_a;
    const float dmin_b =
        half_to_float(load_le16(block_b + 16 + j * 2)) * q8_scale_b;
    for (int sb = 0; sb < 8; ++sb) {
      const int bsum_pair_a = (int)bsums_a[sb * 2] + (int)bsums_a[sb * 2 + 1];
      const int bsum_pair_b = (int)bsums_b[sb * 2] + (int)bsums_b[sb * 2 + 1];
      uchar scale_a = 0;
      uchar minimum_a = 0;
      uchar scale_b = 0;
      uchar minimum_b = 0;
      get_scale_min_k4((int)j, block_a + 32 + sb * 12, &scale_a, &minimum_a);
      get_scale_min_k4((int)j, block_b + 32 + sb * 12, &scale_b, &minimum_b);
      sum_minf_a += (float)((int)minimum_a * bsum_pair_a) * dmin_a;
      sum_minf_b += (float)((int)minimum_b * bsum_pair_b) * dmin_b;
    }
  }
  out[expert0 * rows_per_expert + local_row] = sumf_a - sum_minf_a;
  out[expert1 * rows_per_expert + local_row] = sumf_b - sum_minf_b;
}

__kernel void q4k_x8_matvec_rowlane_expert8_multiq8_weighted(
    __global const uchar* packed0,
    __global const uchar* packed1,
    __global const uchar* packed2,
    __global const uchar* packed3,
    __global const uchar* packed4,
    __global const uchar* packed5,
    __global const uchar* packed6,
    __global const uchar* packed7,
    __global const char* q8_qs,
    __global const short* q8_bsums,
    __global const float* q8_d,
    uint blocks_per_row,
    uint row_groups_per_expert,
    __global const float* weights,
    __global float* out) {
  const uint row = (uint)get_global_id(0);
  const uint rows_per_expert = row_groups_per_expert * 8U;
  const uint total_rows = rows_per_expert * 8U;
  if (row >= total_rows) {
    return;
  }
  const uint expert = row / rows_per_expert;
  const uint local_row = row - expert * rows_per_expert;
  const uint group = local_row >> 3;
  const uint j = local_row & 7U;
  __global const uchar* packed = packed0;
  if (expert == 1U) {
    packed = packed1;
  } else if (expert == 2U) {
    packed = packed2;
  } else if (expert == 3U) {
    packed = packed3;
  } else if (expert == 4U) {
    packed = packed4;
  } else if (expert == 5U) {
    packed = packed5;
  } else if (expert == 6U) {
    packed = packed6;
  } else if (expert == 7U) {
    packed = packed7;
  }

  float sumf = 0.0f;
  float sum_minf = 0.0f;
  for (uint block_index = 0; block_index < blocks_per_row; ++block_index) {
    __global const uchar* block =
        packed + ((ulong)group * (ulong)blocks_per_row +
                  (ulong)block_index) * 1152UL;
    const ulong q8_block =
        (ulong)expert * (ulong)blocks_per_row + (ulong)block_index;
    __global const char* q8 = q8_qs + q8_block * 256UL;
    __global const short* bsums = q8_bsums + q8_block * 16UL;
    const float q8_scale = q8_d[q8_block];
    const float d = half_to_float(load_le16(block + j * 2)) * q8_scale;
    for (int k = 0; k < 16; ++k) {
      const int scale_pair = (k >> 2) * 2;
      const int q8_base = (k >> 2) * 64 + (k & 3) * 8;
      uchar scale0 = 0;
      uchar min0 = 0;
      uchar scale1 = 0;
      uchar min1 = 0;
      get_scale_min_k4((int)j, block + 32 + scale_pair * 12,
                       &scale0, &min0);
      get_scale_min_k4((int)j, block + 32 + (scale_pair + 1) * 12,
                       &scale1, &min1);
      int sumi = 0;
      for (int i = 0; i < 8; ++i) {
        const uchar q = block[128 + k * 64 + (int)j * 8 + i];
        const int v0 = (int)(q & (uchar)15);
        const int v1 = (int)(q >> 4);
        const int q8_low = (int)q8[q8_base + i];
        const int q8_high = (int)q8[q8_base + i + 32];
        sumi += v0 * q8_low * (int)scale0;
        sumi += v1 * q8_high * (int)scale1;
      }
      sumf += (float)sumi * d;
    }
    const float dmin =
        half_to_float(load_le16(block + 16 + j * 2)) * q8_scale;
    for (int sb = 0; sb < 8; ++sb) {
      const int bsum_pair = (int)bsums[sb * 2] + (int)bsums[sb * 2 + 1];
      uchar scale = 0;
      uchar minimum = 0;
      get_scale_min_k4((int)j, block + 32 + sb * 12, &scale, &minimum);
      sum_minf += (float)((int)minimum * bsum_pair) * dmin;
    }
  }
  out[row] = (sumf - sum_minf) * weights[expert];
}

__kernel void q4k_x8_matvec_rowlane_expert8_multiq8_weighted_sum(
    __global const uchar* packed0,
    __global const uchar* packed1,
    __global const uchar* packed2,
    __global const uchar* packed3,
    __global const uchar* packed4,
    __global const uchar* packed5,
    __global const uchar* packed6,
    __global const uchar* packed7,
    __global const char* q8_qs,
    __global const short* q8_bsums,
    __global const float* q8_d,
    uint blocks_per_row,
    uint row_groups_per_expert,
    __global const float* weights,
    __global float* out) {
  const uint local_row = (uint)get_global_id(0);
  const uint rows_per_expert = row_groups_per_expert * 8U;
  if (local_row >= rows_per_expert) {
    return;
  }
  const uint group = local_row >> 3;
  const uint j = local_row & 7U;

  float weighted_sum = 0.0f;
  for (uint expert = 0; expert < 8U; ++expert) {
    __global const uchar* packed = packed0;
    if (expert == 1U) {
      packed = packed1;
    } else if (expert == 2U) {
      packed = packed2;
    } else if (expert == 3U) {
      packed = packed3;
    } else if (expert == 4U) {
      packed = packed4;
    } else if (expert == 5U) {
      packed = packed5;
    } else if (expert == 6U) {
      packed = packed6;
    } else if (expert == 7U) {
      packed = packed7;
    }

    float sumf = 0.0f;
    float sum_minf = 0.0f;
    for (uint block_index = 0; block_index < blocks_per_row; ++block_index) {
      __global const uchar* block =
          packed + ((ulong)group * (ulong)blocks_per_row +
                    (ulong)block_index) * 1152UL;
      const ulong q8_block =
          (ulong)expert * (ulong)blocks_per_row + (ulong)block_index;
      __global const char* q8 = q8_qs + q8_block * 256UL;
      __global const short* bsums = q8_bsums + q8_block * 16UL;
      const float q8_scale = q8_d[q8_block];
      const float d = half_to_float(load_le16(block + j * 2)) * q8_scale;
      for (int k = 0; k < 16; ++k) {
        const int scale_pair = (k >> 2) * 2;
        const int q8_base = (k >> 2) * 64 + (k & 3) * 8;
        uchar scale0 = 0;
        uchar min0 = 0;
        uchar scale1 = 0;
        uchar min1 = 0;
        get_scale_min_k4((int)j, block + 32 + scale_pair * 12,
                         &scale0, &min0);
        get_scale_min_k4((int)j, block + 32 + (scale_pair + 1) * 12,
                         &scale1, &min1);
        int sumi = 0;
        for (int i = 0; i < 8; ++i) {
          const uchar q = block[128 + k * 64 + (int)j * 8 + i];
          const int v0 = (int)(q & (uchar)15);
          const int v1 = (int)(q >> 4);
          const int q8_low = (int)q8[q8_base + i];
          const int q8_high = (int)q8[q8_base + i + 32];
          sumi += v0 * q8_low * (int)scale0;
          sumi += v1 * q8_high * (int)scale1;
        }
        sumf += (float)sumi * d;
      }
      const float dmin =
          half_to_float(load_le16(block + 16 + j * 2)) * q8_scale;
      for (int sb = 0; sb < 8; ++sb) {
        const int bsum_pair = (int)bsums[sb * 2] + (int)bsums[sb * 2 + 1];
        uchar scale = 0;
        uchar minimum = 0;
        get_scale_min_k4((int)j, block + 32 + sb * 12, &scale, &minimum);
        sum_minf += (float)((int)minimum * bsum_pair) * dmin;
      }
    }
    weighted_sum += (sumf - sum_minf) * weights[expert];
  }
  out[local_row] = weighted_sum;
}

static inline float selected_q4_down_dot_probe(
    __global const uchar* packed,
    __global const char* q8_qs,
    __global const short* q8_bsums,
    __global const float* q8_d,
    uint blocks_per_row,
    uint expert,
    uint group,
    uint j) {
  float sumf = 0.0f;
  float sum_minf = 0.0f;
  for (uint block_index = 0; block_index < blocks_per_row; ++block_index) {
    __global const uchar* block =
        packed + ((ulong)group * (ulong)blocks_per_row +
                  (ulong)block_index) * 1152UL;
    const ulong q8_block =
        (ulong)expert * (ulong)blocks_per_row + (ulong)block_index;
    __global const char* q8 = q8_qs + q8_block * 256UL;
    __global const short* bsums = q8_bsums + q8_block * 16UL;
    const float q8_scale = q8_d[q8_block];
    const float d = half_to_float(load_le16(block + j * 2)) * q8_scale;
    for (int k = 0; k < 16; ++k) {
      const int scale_pair = (k >> 2) * 2;
      const int q8_base = (k >> 2) * 64 + (k & 3) * 8;
      uchar scale0 = 0;
      uchar min0 = 0;
      uchar scale1 = 0;
      uchar min1 = 0;
      get_scale_min_k4((int)j, block + 32 + scale_pair * 12,
                       &scale0, &min0);
      get_scale_min_k4((int)j, block + 32 + (scale_pair + 1) * 12,
                       &scale1, &min1);
      int sumi = 0;
      for (int i = 0; i < 8; ++i) {
        const uchar q = block[128 + k * 64 + (int)j * 8 + i];
        const int v0 = (int)(q & (uchar)15);
        const int v1 = (int)(q >> 4);
        const int q8_low = (int)q8[q8_base + i];
        const int q8_high = (int)q8[q8_base + i + 32];
        sumi += v0 * q8_low * (int)scale0;
        sumi += v1 * q8_high * (int)scale1;
      }
      sumf += (float)sumi * d;
    }
    const float dmin =
        half_to_float(load_le16(block + 16 + j * 2)) * q8_scale;
    for (int sb = 0; sb < 8; ++sb) {
      const int bsum_pair = (int)bsums[sb * 2] + (int)bsums[sb * 2 + 1];
      uchar scale = 0;
      uchar minimum = 0;
      get_scale_min_k4((int)j, block + 32 + sb * 12, &scale, &minimum);
      sum_minf += (float)((int)minimum * bsum_pair) * dmin;
    }
  }
  return sumf - sum_minf;
}

__kernel void q4k_x8_matvec_rowlane_expert8_multiq8_group8_weighted_sum(
    __global const uchar* packed0,
    __global const uchar* packed1,
    __global const uchar* packed2,
    __global const uchar* packed3,
    __global const uchar* packed4,
    __global const uchar* packed5,
    __global const uchar* packed6,
    __global const uchar* packed7,
    __global const char* q8_qs,
    __global const short* q8_bsums,
    __global const float* q8_d,
    uint blocks_per_row,
    uint row_groups_per_expert,
    __global const float* weights,
    __global float* out,
    __local float* partial) {
  const uint lid = (uint)get_local_id(0);
  const uint rows_per_tile = (uint)get_local_size(0) / 8U;
  const uint tile_row = lid / 8U;
  const uint expert = lid - tile_row * 8U;
  const uint rows_per_expert = row_groups_per_expert * 8U;
  const uint local_row = (uint)get_group_id(0) * rows_per_tile + tile_row;

  float contribution = 0.0f;
  if (expert < 8U && local_row < rows_per_expert) {
    const uint group = local_row >> 3;
    const uint j = local_row & 7U;
    __global const uchar* packed = packed0;
    if (expert == 1U) {
      packed = packed1;
    } else if (expert == 2U) {
      packed = packed2;
    } else if (expert == 3U) {
      packed = packed3;
    } else if (expert == 4U) {
      packed = packed4;
    } else if (expert == 5U) {
      packed = packed5;
    } else if (expert == 6U) {
      packed = packed6;
    } else if (expert == 7U) {
      packed = packed7;
    }
    contribution = selected_q4_down_dot_probe(
        packed, q8_qs, q8_bsums, q8_d, blocks_per_row, expert, group, j) *
        weights[expert];
  }
  partial[lid] = contribution;
  barrier(CLK_LOCAL_MEM_FENCE);

  if (expert == 0U && local_row < rows_per_expert) {
    float acc = 0.0f;
    const uint base = tile_row * 8U;
    for (uint e = 0; e < 8U; ++e) {
      acc += partial[base + e];
    }
    out[local_row] = acc;
  }
}

__kernel void q4k_x8_matvec_rowlane_expert8_multiq8_occupancy4(
    __global const uchar* packed0,
    __global const uchar* packed1,
    __global const uchar* packed2,
    __global const uchar* packed3,
    __global const uchar* packed4,
    __global const uchar* packed5,
    __global const uchar* packed6,
    __global const uchar* packed7,
    __global const char* q8_qs,
    __global const short* q8_bsums,
    __global const float* q8_d,
    uint blocks_per_row,
    uint row_groups_per_expert,
    __global float* out) {
  const uint flat = (uint)get_global_id(0);
  const uint rows_per_expert = row_groups_per_expert * 8U;
  const uint rows_per_group = rows_per_expert * 8U;
  const uint group_repeat = flat / rows_per_group;
  const uint group_row = flat - group_repeat * rows_per_group;
  if (group_repeat >= 4U || group_row >= rows_per_group) {
    return;
  }
  const uint expert = group_row / rows_per_expert;
  const uint local_row = group_row - expert * rows_per_expert;
  const uint group = local_row >> 3;
  const uint j = local_row & 7U;
  __global const uchar* packed = packed0;
  if (expert == 1U) {
    packed = packed1;
  } else if (expert == 2U) {
    packed = packed2;
  } else if (expert == 3U) {
    packed = packed3;
  } else if (expert == 4U) {
    packed = packed4;
  } else if (expert == 5U) {
    packed = packed5;
  } else if (expert == 6U) {
    packed = packed6;
  } else if (expert == 7U) {
    packed = packed7;
  }

  float sumf = 0.0f;
  float sum_minf = 0.0f;
  for (uint block_index = 0; block_index < blocks_per_row; ++block_index) {
    __global const uchar* block =
        packed + ((ulong)group * (ulong)blocks_per_row +
                  (ulong)block_index) * 1152UL;
    const ulong q8_block =
        (ulong)expert * (ulong)blocks_per_row + (ulong)block_index;
    __global const char* q8 = q8_qs + q8_block * 256UL;
    __global const short* bsums = q8_bsums + q8_block * 16UL;
    const float q8_scale = q8_d[q8_block];
    const float d = half_to_float(load_le16(block + j * 2)) * q8_scale;
    for (int k = 0; k < 16; ++k) {
      const int scale_pair = (k >> 2) * 2;
      const int q8_base = (k >> 2) * 64 + (k & 3) * 8;
      uchar scale0 = 0;
      uchar min0 = 0;
      uchar scale1 = 0;
      uchar min1 = 0;
      get_scale_min_k4((int)j, block + 32 + scale_pair * 12,
                       &scale0, &min0);
      get_scale_min_k4((int)j, block + 32 + (scale_pair + 1) * 12,
                       &scale1, &min1);
      int sumi = 0;
      for (int i = 0; i < 8; ++i) {
        const uchar q = block[128 + k * 64 + (int)j * 8 + i];
        const int v0 = (int)(q & (uchar)15);
        const int v1 = (int)(q >> 4);
        const int q8_low = (int)q8[q8_base + i];
        const int q8_high = (int)q8[q8_base + i + 32];
        sumi += v0 * q8_low * (int)scale0;
        sumi += v1 * q8_high * (int)scale1;
      }
      sumf += (float)sumi * d;
    }
    const float dmin =
        half_to_float(load_le16(block + 16 + j * 2)) * q8_scale;
    for (int sb = 0; sb < 8; ++sb) {
      const int bsum_pair = (int)bsums[sb * 2] + (int)bsums[sb * 2 + 1];
      uchar scale = 0;
      uchar minimum = 0;
      get_scale_min_k4((int)j, block + 32 + sb * 12, &scale, &minimum);
      sum_minf += (float)((int)minimum * bsum_pair) * dmin;
    }
  }
  out[flat] = sumf - sum_minf;
}

__kernel void q6k_selected_down_matvec_rowstripe_expert8_plus_shared_tail_contrib_raw(
    __global const uchar* raw0,
    __global const uchar* raw1,
    __global const uchar* raw2,
    __global const uchar* raw3,
    __global const uchar* raw4,
    __global const uchar* raw5,
    __global const uchar* raw6,
    __global const uchar* raw7,
    __global const uchar* shared_raw,
    __global const char* selected_q8_qs,
    __global const float* selected_q8_d,
    __global const char* shared_q8_qs,
    __global const float* shared_q8_d,
    __global const float* weights,
    __global const float* gate,
    uint rows_per_expert,
    uint blocks_per_row,
    uint rows_per_tile,
    __global float* contrib) {
  const uint flat_row = (uint)get_global_id(0);
  const uint selected_total_rows = rows_per_expert * 8U;
  if (flat_row < selected_total_rows) {
    const uint selected = flat_row / rows_per_expert;
    const uint local_row = flat_row - selected * rows_per_expert;
    __global const uchar* raw = raw0;
    if (selected == 1U) {
      raw = raw1;
    } else if (selected == 2U) {
      raw = raw2;
    } else if (selected == 3U) {
      raw = raw3;
    } else if (selected == 4U) {
      raw = raw4;
    } else if (selected == 5U) {
      raw = raw5;
    } else if (selected == 6U) {
      raw = raw6;
    } else if (selected == 7U) {
      raw = raw7;
    }
    const uint row_tile = local_row / rows_per_tile;
    const uint row_lane = local_row - row_tile * rows_per_tile;
    const uint tile_block_bytes = rows_per_tile * 210U;
    __global const char* expert_q8 =
        selected_q8_qs + (ulong)selected * (ulong)blocks_per_row * 256UL;
    __global const float* expert_q8_d =
        selected_q8_d + (ulong)selected * (ulong)blocks_per_row;
    float sum = 0.0f;
    for (uint block_index = 0; block_index < blocks_per_row; ++block_index) {
      __global const uchar* tile =
          raw + ((ulong)row_tile * (ulong)blocks_per_row +
                 (ulong)block_index) *
                    (ulong)tile_block_bytes;
      __global const uchar* ql0 = tile;
      __global const uchar* qh0 = ql0 + (ulong)rows_per_tile * 64UL;
      __global const uchar* ql1 = qh0 + (ulong)rows_per_tile * 32UL;
      __global const uchar* qh1 = ql1 + (ulong)rows_per_tile * 64UL;
      __global const char* scales =
          (__global const char*)(qh1 + (ulong)rows_per_tile * 32UL);
      __global const uchar* d_base =
          (__global const uchar*)scales + (ulong)rows_per_tile * 16UL;
      __global const char* q8 = expert_q8 + (ulong)block_index * 256UL;
      const float combined_scale =
          half_to_float(load_le16(d_base + (ulong)row_lane * 2UL)) *
          expert_q8_d[block_index];
      int lane_sums[8];
      for (int lane = 0; lane < 8; ++lane) {
        lane_sums[lane] = 0;
      }
      for (int half_index = 0; half_index < 2; ++half_index) {
        __global const uchar* ql =
            (half_index == 0 ? ql0 : ql1) + (ulong)row_lane * 64UL;
        __global const uchar* qh =
            (half_index == 0 ? qh0 : qh1) + (ulong)row_lane * 32UL;
        __global const char* half_scales =
            scales + (ulong)row_lane * 16UL + (ulong)half_index * 8UL;
        const int base = half_index * 128;
        for (int scale_group = 0; scale_group < 2; ++scale_group) {
          const int lane_begin = scale_group * 16;
          const int scale0 = (int)half_scales[scale_group];
          const int scale1 = (int)half_scales[scale_group + 2];
          const int scale2 = (int)half_scales[scale_group + 4];
          const int scale3 = (int)half_scales[scale_group + 6];
          for (int lane = lane_begin; lane < lane_begin + 16; ++lane) {
            const int high = (int)qh[lane];
            const int lane_index = lane & 7;
            lane_sums[lane_index] +=
                scale0 * (int)q8[base + lane] *
                (((int)(ql[lane] & (uchar)15) | (((high >> 0) & 3) << 4)) - 32);
            lane_sums[lane_index] +=
                scale1 * (int)q8[base + 32 + lane] *
                (((int)(ql[32 + lane] & (uchar)15) | (((high >> 2) & 3) << 4)) - 32);
            lane_sums[lane_index] +=
                scale2 * (int)q8[base + 64 + lane] *
                (((int)(ql[lane] >> 4) | (((high >> 4) & 3) << 4)) - 32);
            lane_sums[lane_index] +=
                scale3 * (int)q8[base + 96 + lane] *
                (((int)(ql[32 + lane] >> 4) | (((high >> 6) & 3) << 4)) - 32);
          }
        }
      }
      for (int lane = 0; lane < 8; ++lane) {
        sum += combined_scale * (float)lane_sums[lane];
      }
    }
    contrib[flat_row] = sum * weights[selected];
    return;
  }

  const uint row = flat_row - selected_total_rows;
  __global const uchar* row_raw =
      shared_raw + (ulong)row * (ulong)blocks_per_row * 210UL;
  float sum = 0.0f;
  for (uint block_index = 0; block_index < blocks_per_row; ++block_index) {
    __global const uchar* block = row_raw + (ulong)block_index * 210UL;
    __global const char* q8 = shared_q8_qs + (ulong)block_index * 256UL;
    __global const char* scales = (__global const char*)(block + 192);
    const float combined_scale =
        half_to_float(load_le16(block + 208)) * shared_q8_d[block_index];
    int lane_sums[8];
    for (int lane = 0; lane < 8; ++lane) {
      lane_sums[lane] = 0;
    }
    for (int half_index = 0; half_index < 2; ++half_index) {
      __global const uchar* ql = block + half_index * 64;
      __global const uchar* qh = block + 128 + half_index * 32;
      __global const char* half_scales = scales + half_index * 8;
      const int base = half_index * 128;
      for (int scale_group = 0; scale_group < 2; ++scale_group) {
        const int lane_begin = scale_group * 16;
        const int scale0 = (int)half_scales[scale_group];
        const int scale1 = (int)half_scales[scale_group + 2];
        const int scale2 = (int)half_scales[scale_group + 4];
        const int scale3 = (int)half_scales[scale_group + 6];
        for (int lane = lane_begin; lane < lane_begin + 16; ++lane) {
          const int high = (int)qh[lane];
          const int lane_index = lane & 7;
          lane_sums[lane_index] +=
              scale0 * (int)q8[base + lane] *
              (((int)(ql[lane] & (uchar)15) | (((high >> 0) & 3) << 4)) - 32);
          lane_sums[lane_index] +=
              scale1 * (int)q8[base + 32 + lane] *
              (((int)(ql[32 + lane] & (uchar)15) | (((high >> 2) & 3) << 4)) - 32);
          lane_sums[lane_index] +=
              scale2 * (int)q8[base + 64 + lane] *
              (((int)(ql[lane] >> 4) | (((high >> 4) & 3) << 4)) - 32);
          lane_sums[lane_index] +=
              scale3 * (int)q8[base + 96 + lane] *
              (((int)(ql[32 + lane] >> 4) | (((high >> 6) & 3) << 4)) - 32);
        }
      }
    }
    for (int lane = 0; lane < 8; ++lane) {
      sum += combined_scale * (float)lane_sums[lane];
    }
  }
  contrib[selected_total_rows + row] = sum * sigmoid_f32(gate[0]);
}

__kernel void ffn_tail_reduce9_contrib_f32(
    __global const float* contrib,
    __global const float* attn_residual,
    uint hidden_size,
    __global float* layer_output) {
  const uint row = (uint)get_global_id(0);
  if (row >= hidden_size) {
    return;
  }
  float acc = attn_residual[row];
  for (uint contributor = 0; contributor < 9U; ++contributor) {
    acc += contrib[contributor * hidden_size + row];
  }
  layer_output[row] = acc;
}

'''


PROBE_CPP = r'''
#include "intel_qwen36/gguf_loader.hpp"
#include "intel_qwen36/gpu_q4x8_matvec.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

const char* kOpenClSource = @@OPENCL_SOURCE_LITERAL@@;

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
constexpr cl_mem_flags kClMemWriteOnly = 1ULL << 1;
constexpr cl_mem_flags kClMemReadWrite = 1ULL << 0;
constexpr cl_command_queue_properties kClQueueProfilingEnable = 1ULL << 1;
constexpr cl_platform_info kClPlatformName = 0x0902;
constexpr cl_device_info kClDeviceName = 0x102B;
constexpr cl_program_build_info kClProgramBuildLog = 0x1183;
constexpr cl_profiling_info kClProfilingCommandStart = 0x1282;
constexpr cl_profiling_info kClProfilingCommandEnd = 0x1283;

constexpr int kLayerCount = 40;
constexpr int kIntermediateSize = 512;
constexpr int kHiddenSize = 2048;
constexpr int kExpertCount = 256;
constexpr int kExpertUsedCount = 8;
constexpr int kInputValueCount = kIntermediateSize * kExpertUsedCount;
constexpr int kOutputValueCount = kHiddenSize * kExpertUsedCount;
constexpr int kSourceTokenPosition = 15;
constexpr int kQ4KBlockBytes = 144;
constexpr int kQ6KBlockBytes = 210;
constexpr int kQ8BlockValues = 256;
constexpr double kMismatchThreshold = 5e-3;
constexpr double kMaxAbsDiffThreshold = 5e-3;
constexpr double kRmseThreshold = 5e-4;
constexpr double kMinCosine = 0.999;
constexpr double kQ8DMaxAbsDiffThreshold = 1e-8;
constexpr double kQ6RowgroupDownTailTargetUs = 106.59839375;
constexpr std::uint64_t kQ6RowgroupDownTailLocal = 16;

struct Args {
  std::string model_path;
  std::string payload_dir;
  int layer = 5;
  int repeat = 7;
  std::string device_substring = "B390";
};

struct SelectedDevice {
  cl_platform_id platform = nullptr;
  cl_device_id device = nullptr;
  std::string platform_name;
  std::string device_name;
};

struct Q8Planes {
  std::vector<std::int8_t> qs;
  std::vector<std::int16_t> bsums;
  std::vector<float> d;
  std::uint64_t blocks_per_expert = 0;
};

struct DownTiming {
  double min_us = 0.0;
  double mean_us = 0.0;
  double effective_raw_gb_s = 0.0;
  double effective_io_gb_s = 0.0;
  std::uint64_t global_work_items = 0;
  std::uint64_t kernel_launches = 0;
};

struct DownRun {
  std::vector<float> output;
  DownTiming timing;
  std::string platform_name;
  std::string device_name;
  std::string build_log;
  double program_build_ms = 0.0;
};

struct SelectedSharedQ6DownRun {
  std::vector<float> selected_separate_output;
  std::vector<float> shared_separate_output;
  std::vector<float> selected_combined_output;
  std::vector<float> shared_combined_output;
  DownTiming selected_timing;
  DownTiming shared_timing;
  DownTiming combined_timing;
  std::string platform_name;
  std::string device_name;
  std::string build_log;
  double program_build_ms = 0.0;
};

struct DownTailNonAtomicRun {
  std::vector<float> contributions;
  std::vector<float> layer_output;
  DownTiming contribution_timing;
  DownTiming reduce_timing;
  std::string platform_name;
  std::string device_name;
  std::string build_log;
  double program_build_ms = 0.0;
  double shell_sum_min_us = 0.0;
  double shell_sum_mean_us = 0.0;
};

struct DownTailRowgroupRun {
  std::vector<float> layer_output;
  DownTiming timing;
  std::string platform_name;
  std::string device_name;
  std::string build_log;
  double program_build_ms = 0.0;
  std::uint64_t local_work_items = kQ6RowgroupDownTailLocal;
};

struct DeviceQ8DownRun {
  std::vector<float> output;
  Q8Planes q8;
  DownTiming q8_quantize_timing;
  DownTiming down_timing;
  std::string platform_name;
  std::string device_name;
  std::string build_log;
  double program_build_ms = 0.0;
  double shell_sum_min_us = 0.0;
  double shell_sum_mean_us = 0.0;
};

struct DeviceSwiGluQ8DownRun {
  std::vector<float> swiglu;
  std::vector<float> output;
  Q8Planes q8;
  DownTiming swiglu_timing;
  DownTiming q8_quantize_timing;
  DownTiming down_timing;
  std::string platform_name;
  std::string device_name;
  std::string build_log;
  double program_build_ms = 0.0;
  double shell_sum_min_us = 0.0;
  double shell_sum_mean_us = 0.0;
};

struct DeviceSwiGluF32InputDownRun {
  std::vector<float> swiglu;
  std::vector<float> output;
  DownTiming swiglu_timing;
  DownTiming down_timing;
  std::string platform_name;
  std::string device_name;
  std::string build_log;
  double program_build_ms = 0.0;
  double shell_sum_min_us = 0.0;
  double shell_sum_mean_us = 0.0;
};

struct Q8PlaneCompareStats {
  bool qs_same_size = false;
  bool bsums_same_size = false;
  bool d_same_size = false;
  bool qs_exact = false;
  bool bsums_exact = false;
  bool d_finite = false;
  bool d_within_threshold = false;
  bool passed = false;
  std::uint64_t qs_value_count = 0;
  std::uint64_t bsums_value_count = 0;
  std::uint64_t d_value_count = 0;
  std::uint64_t qs_mismatch_count = 0;
  std::uint64_t bsums_mismatch_count = 0;
  std::uint64_t d_mismatch_count = 0;
  double d_max_abs_diff = 0.0;
  double d_rmse = 0.0;
};

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

std::string JoinPath(const std::string& dir, const std::string& name) {
  return (!dir.empty() && dir.back() == '/') ? dir + name : dir + "/" + name;
}

std::string LayerTensorName(int layer, const std::string& suffix) {
  return "blk." + std::to_string(layer) + "." + suffix;
}

std::string PlatformString(OpenClApi& api, cl_platform_id platform, cl_platform_info info) {
  std::size_t size = 0;
  Check(api.clGetPlatformInfo(platform, info, 0, nullptr, &size), "clGetPlatformInfo(size)");
  std::string out(size, '\0');
  Check(api.clGetPlatformInfo(platform, info, size, out.data(), nullptr), "clGetPlatformInfo(value)");
  if (!out.empty() && out.back() == '\0') out.pop_back();
  return out;
}

std::string DeviceString(OpenClApi& api, cl_device_id device, cl_device_info info) {
  std::size_t size = 0;
  Check(api.clGetDeviceInfo(device, info, 0, nullptr, &size), "clGetDeviceInfo(size)");
  std::string out(size, '\0');
  Check(api.clGetDeviceInfo(device, info, size, out.data(), nullptr), "clGetDeviceInfo(value)");
  if (!out.empty() && out.back() == '\0') out.pop_back();
  return out;
}

SelectedDevice SelectDevice(OpenClApi& api, const std::string& device_substring) {
  cl_uint platform_count = 0;
  Check(api.clGetPlatformIDs(0, nullptr, &platform_count), "clGetPlatformIDs(count)");
  std::vector<cl_platform_id> platforms(platform_count);
  Check(api.clGetPlatformIDs(platform_count, platforms.data(), nullptr), "clGetPlatformIDs(list)");
  for (cl_platform_id platform : platforms) {
    cl_uint device_count = 0;
    if (api.clGetDeviceIDs(platform, kClDeviceTypeGpu, 0, nullptr, &device_count) != kClSuccess ||
        device_count == 0) {
      continue;
    }
    std::vector<cl_device_id> devices(device_count);
    Check(api.clGetDeviceIDs(platform, kClDeviceTypeGpu, device_count, devices.data(), nullptr),
          "clGetDeviceIDs(list)");
    for (cl_device_id device : devices) {
      const std::string name = DeviceString(api, device, kClDeviceName);
      if (device_substring.empty() || name.find(device_substring) != std::string::npos) {
        return {platform, device, PlatformString(api, platform, kClPlatformName), name};
      }
    }
  }
  Die("no matching OpenCL GPU for substring: " + device_substring);
}

std::string BuildLog(OpenClApi& api, cl_program program, cl_device_id device) {
  std::size_t size = 0;
  const cl_int err = api.clGetProgramBuildInfo(program, device, kClProgramBuildLog, 0, nullptr, &size);
  if (err != kClSuccess || size == 0) {
    return "";
  }
  std::string out(size, '\0');
  Check(api.clGetProgramBuildInfo(program, device, kClProgramBuildLog, size, out.data(), nullptr),
        "clGetProgramBuildInfo(log)");
  if (!out.empty() && out.back() == '\0') out.pop_back();
  return out;
}

double EventUs(OpenClApi& api, cl_event event) {
  cl_ulong start = 0;
  cl_ulong end = 0;
  Check(api.clGetEventProfilingInfo(event, kClProfilingCommandStart, sizeof(start), &start, nullptr),
        "clGetEventProfilingInfo(start)");
  Check(api.clGetEventProfilingInfo(event, kClProfilingCommandEnd, sizeof(end), &end, nullptr),
        "clGetEventProfilingInfo(end)");
  return static_cast<double>(end - start) / 1000.0;
}

void ReleaseMem(OpenClApi& api, cl_mem* mem) {
  if (*mem) {
    api.clReleaseMemObject(*mem);
    *mem = nullptr;
  }
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

int NearestInt(float value) {
  float shifted = value + 12582912.0f;
  int bits = 0;
  std::memcpy(&bits, &shifted, sizeof(bits));
  return (bits & 0x007fffff) - 0x00400000;
}

Q8Planes QuantizePerExpertQ8K(const std::vector<float>& input,
                              std::uint64_t selected_count,
                              std::uint64_t values_per_expert) {
  Require(values_per_expert % kQ8BlockValues == 0,
          "Q8_K selected-down input requires 256-aligned experts");
  Require(input.size() == selected_count * values_per_expert,
          "selected-down input size mismatch");
  Q8Planes planes;
  planes.blocks_per_expert = values_per_expert / kQ8BlockValues;
  planes.qs.assign(input.size(), 0);
  planes.bsums.assign(static_cast<std::size_t>(selected_count * planes.blocks_per_expert * 16), 0);
  planes.d.assign(static_cast<std::size_t>(selected_count * planes.blocks_per_expert), 0.0f);
  for (std::uint64_t selected = 0; selected < selected_count; ++selected) {
    for (std::uint64_t block = 0; block < planes.blocks_per_expert; ++block) {
      const auto value_base =
          static_cast<std::size_t>(selected * values_per_expert + block * kQ8BlockValues);
      float max = 0.0f;
      float amax = 0.0f;
      for (int i = 0; i < kQ8BlockValues; ++i) {
        const float abs_value = std::abs(input[value_base + static_cast<std::size_t>(i)]);
        if (abs_value > amax) {
          amax = abs_value;
          max = input[value_base + static_cast<std::size_t>(i)];
        }
      }
      if (amax == 0.0f) {
        continue;
      }
      const float iscale = -127.0f / max;
      for (int i = 0; i < kQ8BlockValues; ++i) {
        const int quantized =
            std::min(127, NearestInt(iscale * input[value_base + static_cast<std::size_t>(i)]));
        planes.qs[value_base + static_cast<std::size_t>(i)] =
            static_cast<std::int8_t>(quantized);
      }
      const auto bsum_base =
          static_cast<std::size_t>((selected * planes.blocks_per_expert + block) * 16);
      for (int group = 0; group < 16; ++group) {
        int sum = 0;
        for (int i = 0; i < 16; ++i) {
          sum += planes.qs[value_base + static_cast<std::size_t>(group * 16 + i)];
        }
        planes.bsums[bsum_base + static_cast<std::size_t>(group)] =
            static_cast<std::int16_t>(sum);
      }
      planes.d[static_cast<std::size_t>(selected * planes.blocks_per_expert + block)] =
          1.0f / iscale;
    }
  }
  return planes;
}

std::vector<std::uint8_t> ReadSelectedExpertRaw(std::ifstream& model,
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
    Require(static_cast<bool>(model), "selected expert down slice seek failed");
    model.read(reinterpret_cast<char*>(raw.data() + target_offset),
               static_cast<std::streamsize>(byte_count));
    Require(model.gcount() == static_cast<std::streamsize>(byte_count),
            "selected expert down slice read failed");
  }
  return raw;
}

std::vector<std::uint8_t> ReadTensorBytes(std::ifstream& model,
                                          const iq36::GgufTensorInfo& tensor) {
  const std::uint64_t bytes = iq36::ggml_tensor_nbytes(tensor.type, tensor.dims);
  std::vector<std::uint8_t> raw(static_cast<std::size_t>(bytes), 0);
  model.clear();
  model.seekg(static_cast<std::streamoff>(tensor.absolute_offset), std::ios::beg);
  Require(static_cast<bool>(model), "tensor byte seek failed");
  model.read(reinterpret_cast<char*>(raw.data()),
             static_cast<std::streamsize>(raw.size()));
  Require(model.gcount() == static_cast<std::streamsize>(raw.size()),
          "tensor byte read failed");
  return raw;
}

std::vector<std::uint8_t> BuildQ6Rowstripe(const std::vector<std::uint8_t>& raw,
                                           std::uint64_t rows,
                                           std::uint64_t blocks_per_row,
                                           std::uint64_t rows_per_tile) {
  Require(rows > 0, "Q6 rowstripe rows must be nonzero");
  Require(blocks_per_row > 0, "Q6 rowstripe blocks must be nonzero");
  Require(rows_per_tile > 0, "Q6 rowstripe tile size must be nonzero");
  Require(raw.size() == static_cast<std::size_t>(rows * blocks_per_row * kQ6KBlockBytes),
          "Q6 rowstripe raw byte size mismatch");
  struct Segment {
    std::uint64_t offset;
    std::uint64_t bytes;
  };
  constexpr Segment kSegments[] = {
      {0ULL, 64ULL},
      {128ULL, 32ULL},
      {64ULL, 64ULL},
      {160ULL, 32ULL},
      {192ULL, 16ULL},
      {208ULL, 2ULL},
  };
  const std::uint64_t row_tile_count =
      (rows + rows_per_tile - 1ULL) / rows_per_tile;
  std::vector<std::uint8_t> striped;
  striped.reserve(static_cast<std::size_t>(
      row_tile_count * blocks_per_row * rows_per_tile * kQ6KBlockBytes));
  for (std::uint64_t row_tile = 0; row_tile < row_tile_count; ++row_tile) {
    for (std::uint64_t block = 0; block < blocks_per_row; ++block) {
      for (const auto& segment : kSegments) {
        for (std::uint64_t lane = 0; lane < rows_per_tile; ++lane) {
          const std::uint64_t row = row_tile * rows_per_tile + lane;
          if (row >= rows) {
            striped.insert(striped.end(),
                           static_cast<std::size_t>(segment.bytes), 0U);
            continue;
          }
          const std::uint8_t* src =
              raw.data() + (row * blocks_per_row + block) * kQ6KBlockBytes +
              segment.offset;
          striped.insert(striped.end(), src,
                         src + static_cast<std::ptrdiff_t>(segment.bytes));
        }
      }
    }
  }
  Require(striped.size() == static_cast<std::size_t>(
                                row_tile_count * blocks_per_row *
                                rows_per_tile * kQ6KBlockBytes),
          "Q6 rowstripe byte size mismatch");
  return striped;
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
    else if (key == "--repeat") args.repeat = std::stoi(value("--repeat"));
    else if (key == "--device-substring") args.device_substring = value("--device-substring");
    else Die("unknown argument: " + key);
  }
  Require(!args.model_path.empty(), "--model is required");
  Require(!args.payload_dir.empty(), "--payload-dir is required");
  Require(args.layer >= 0 && args.layer < kLayerCount, "--layer is out of range");
  Require(args.repeat > 0, "--repeat must be positive");
  return args;
}

bool ComparePassed(const iq36::VectorCompareStats& stats) {
  return stats.same_size &&
         stats.finite &&
         stats.mismatch_count == 0 &&
         stats.max_abs_diff <= kMaxAbsDiffThreshold &&
         stats.rmse <= kRmseThreshold &&
         stats.cosine >= kMinCosine;
}

bool CompareRecorded(const iq36::VectorCompareStats& stats) {
  return stats.same_size && stats.finite && stats.compared_value_count > 0;
}

Q8PlaneCompareStats CompareQ8Planes(const Q8Planes& actual,
                                    const Q8Planes& expected) {
  Q8PlaneCompareStats stats;
  stats.qs_same_size = actual.qs.size() == expected.qs.size();
  stats.bsums_same_size = actual.bsums.size() == expected.bsums.size();
  stats.d_same_size = actual.d.size() == expected.d.size();
  stats.qs_value_count =
      static_cast<std::uint64_t>(std::min(actual.qs.size(), expected.qs.size()));
  stats.bsums_value_count =
      static_cast<std::uint64_t>(std::min(actual.bsums.size(), expected.bsums.size()));
  stats.d_value_count =
      static_cast<std::uint64_t>(std::min(actual.d.size(), expected.d.size()));
  for (std::size_t i = 0; i < static_cast<std::size_t>(stats.qs_value_count); ++i) {
    if (actual.qs[i] != expected.qs[i]) {
      ++stats.qs_mismatch_count;
    }
  }
  for (std::size_t i = 0; i < static_cast<std::size_t>(stats.bsums_value_count); ++i) {
    if (actual.bsums[i] != expected.bsums[i]) {
      ++stats.bsums_mismatch_count;
    }
  }
  double d_sum_sq = 0.0;
  stats.d_finite = true;
  for (std::size_t i = 0; i < static_cast<std::size_t>(stats.d_value_count); ++i) {
    const double diff = static_cast<double>(actual.d[i]) -
                        static_cast<double>(expected.d[i]);
    const double abs_diff = std::abs(diff);
    if (!std::isfinite(actual.d[i]) || !std::isfinite(expected.d[i])) {
      stats.d_finite = false;
    }
    if (abs_diff > kQ8DMaxAbsDiffThreshold) {
      ++stats.d_mismatch_count;
    }
    stats.d_max_abs_diff = std::max(stats.d_max_abs_diff, abs_diff);
    d_sum_sq += diff * diff;
  }
  stats.d_rmse =
      stats.d_value_count == 0
          ? 0.0
          : std::sqrt(d_sum_sq / static_cast<double>(stats.d_value_count));
  stats.qs_exact = stats.qs_same_size && stats.qs_mismatch_count == 0;
  stats.bsums_exact = stats.bsums_same_size && stats.bsums_mismatch_count == 0;
  stats.d_within_threshold =
      stats.d_same_size && stats.d_finite &&
      stats.d_mismatch_count == 0 &&
      stats.d_max_abs_diff <= kQ8DMaxAbsDiffThreshold;
  stats.passed =
      stats.qs_exact && stats.bsums_exact && stats.d_within_threshold;
  return stats;
}

DownRun RunGpuSelectedDownQ4(const std::vector<std::uint8_t>& selected_raw,
                             const std::vector<float>& input,
                             std::uint64_t rows_per_expert,
                             std::uint64_t blocks_per_row,
                             std::uint64_t selected_count,
                             const std::string& device_substring,
                             int repeat) {
  Require(selected_raw.size() ==
              static_cast<std::size_t>(selected_count * rows_per_expert *
                                       blocks_per_row * kQ4KBlockBytes),
          "selected-down Q4 raw byte size mismatch");
  Require(input.size() == selected_count * kIntermediateSize,
          "selected-down Q4 input size mismatch");
  DownRun run;
  run.output.assign(static_cast<std::size_t>(selected_count * rows_per_expert), 0.0f);
  iq36::GpuQ4X8MatvecRunner runner(device_substring, kOpenClSource);
  run.platform_name = runner.platform_name();
  run.device_name = runner.device_name();
  run.build_log = runner.build_log();
  run.program_build_ms = runner.program_build_ms();
  for (std::uint64_t selected = 0; selected < selected_count; ++selected) {
    const auto raw_begin = selected_raw.begin() +
        static_cast<std::ptrdiff_t>(selected * rows_per_expert * blocks_per_row * kQ4KBlockBytes);
    const auto raw_end = raw_begin +
        static_cast<std::ptrdiff_t>(rows_per_expert * blocks_per_row * kQ4KBlockBytes);
    const std::vector<std::uint8_t> expert_raw(raw_begin, raw_end);
    const auto input_begin = input.begin() +
        static_cast<std::ptrdiff_t>(selected * kIntermediateSize);
    const auto input_end = input_begin + static_cast<std::ptrdiff_t>(kIntermediateSize);
    const std::vector<float> expert_input(input_begin, input_end);
    const auto packed = iq36::PackQ4Kx8(expert_raw, rows_per_expert, blocks_per_row);
    const auto q8 = iq36::QuantizeQ8KInputPlanes(expert_input);
    const auto expert_run =
        runner.Run(packed, q8.qs, q8.bsums, q8.d, rows_per_expert,
                   blocks_per_row, repeat,
                   iq36::GpuQ4X8KernelVariant::kRowlaneParallel);
    std::copy(expert_run.output.begin(), expert_run.output.end(),
              run.output.begin() +
                  static_cast<std::ptrdiff_t>(selected * rows_per_expert));
    run.timing.min_us += expert_run.timing.min_us;
    run.timing.mean_us += expert_run.timing.mean_us;
    run.timing.global_work_items += expert_run.timing.global_work_items;
    run.timing.kernel_launches += 1;
  }
  const double raw_bytes = static_cast<double>(selected_raw.size());
  const double io_bytes = raw_bytes +
      static_cast<double>(input.size() * sizeof(float)) +
      static_cast<double>(run.output.size() * sizeof(float));
  run.timing.effective_raw_gb_s = raw_bytes / (run.timing.min_us / 1e6) / 1e9;
  run.timing.effective_io_gb_s = io_bytes / (run.timing.min_us / 1e6) / 1e9;
  return run;
}

DownRun RunGpuSelectedDownQ4Expert8Kernel(
    const std::vector<std::uint8_t>& selected_raw,
    const std::vector<float>& input,
    const std::vector<float>* weights,
    std::uint64_t rows_per_expert,
    std::uint64_t blocks_per_row,
    std::uint64_t selected_count,
    const std::string& device_substring,
    int repeat,
    const char* kernel_name,
    bool pair2_kernel,
    bool weighted_sum_kernel,
    bool group_reduce_kernel,
    std::uint64_t output_groups = 1) {
  Require(selected_count == 8,
          "expert8 selected Q4 component probe requires eight experts");
  Require(!weighted_sum_kernel || weights != nullptr,
          "expert8 weighted-sum kernel requires router weights");
  Require(!group_reduce_kernel || weights != nullptr,
          "expert8 group-reduce kernel requires router weights");
  Require(!weighted_sum_kernel || !pair2_kernel,
          "expert8 weighted-sum kernel is incompatible with pair2 mode");
  Require(!group_reduce_kernel || (!pair2_kernel && !weighted_sum_kernel),
          "expert8 group-reduce kernel is incompatible with pair2/serial sum mode");
  Require(output_groups > 0, "expert8 selected Q4 output group count must be positive");
  Require(output_groups == 1 ||
              (!pair2_kernel && !weighted_sum_kernel &&
               !group_reduce_kernel && weights == nullptr),
          "expert8 output groups are only supported for unweighted kernels");
  Require(rows_per_expert % 8 == 0,
          "expert8 selected Q4 component probe requires x8 row packing");
  Require(selected_raw.size() ==
              static_cast<std::size_t>(selected_count * rows_per_expert *
                                       blocks_per_row * kQ4KBlockBytes),
          "expert8 selected-down Q4 raw byte size mismatch");
  Require(input.size() == selected_count * kIntermediateSize,
          "expert8 selected-down Q4 input size mismatch");
  if (weights != nullptr) {
    Require(weights->size() == selected_count,
            "expert8 selected-down Q4 weight count mismatch");
  }
  const auto q8 = QuantizePerExpertQ8K(input, selected_count, kIntermediateSize);
  std::vector<std::vector<std::uint8_t>> packed_by_expert;
  packed_by_expert.reserve(static_cast<std::size_t>(selected_count));
  for (std::uint64_t selected = 0; selected < selected_count; ++selected) {
    const auto raw_begin = selected_raw.begin() +
        static_cast<std::ptrdiff_t>(selected * rows_per_expert *
                                    blocks_per_row * kQ4KBlockBytes);
    const auto raw_end = raw_begin +
        static_cast<std::ptrdiff_t>(rows_per_expert * blocks_per_row *
                                    kQ4KBlockBytes);
    packed_by_expert.push_back(
        iq36::PackQ4Kx8(
            std::vector<std::uint8_t>(raw_begin, raw_end),
            rows_per_expert, blocks_per_row));
  }

  DownRun run;
  const std::size_t output_values = static_cast<std::size_t>(
      (weighted_sum_kernel || group_reduce_kernel)
          ? rows_per_expert
          : selected_count * rows_per_expert * output_groups);
  run.output.assign(output_values, 0.0f);
  OpenClApi api;
  const auto selected = SelectDevice(api, device_substring);
  run.platform_name = selected.platform_name;
  run.device_name = selected.device_name;

  cl_int err = kClSuccess;
  cl_context context = api.clCreateContext(nullptr, 1, &selected.device, nullptr, nullptr, &err);
  Check(err, "clCreateContext");
  cl_command_queue queue = api.clCreateCommandQueue(context, selected.device, kClQueueProfilingEnable, &err);
  Check(err, "clCreateCommandQueue");
  const char* source = kOpenClSource;
  const std::size_t source_len = std::strlen(kOpenClSource);
  cl_program program = api.clCreateProgramWithSource(context, 1, &source, &source_len, &err);
  Check(err, "clCreateProgramWithSource");
  const auto build_begin = std::chrono::steady_clock::now();
  err = api.clBuildProgram(program, 1, &selected.device, "", nullptr, nullptr);
  const auto build_end = std::chrono::steady_clock::now();
  run.program_build_ms =
      std::chrono::duration<double, std::milli>(build_end - build_begin).count();
  run.build_log = BuildLog(api, program, selected.device);
  if (err != kClSuccess) {
    throw std::runtime_error(
        std::string("clBuildProgram failed: ") + run.build_log);
  }
  Check(err, "clBuildProgram");
  cl_kernel kernel = api.clCreateKernel(program, kernel_name, &err);
  Check(err, std::string("clCreateKernel(") + kernel_name + ")");

  std::array<cl_mem, 8> packed_buffers{};
  cl_mem q8_qs_buffer = nullptr, q8_bsums_buffer = nullptr, q8_d_buffer = nullptr;
  cl_mem weights_buffer = nullptr;
  cl_mem output_buffer = nullptr;
  try {
    for (std::size_t i = 0; i < packed_buffers.size(); ++i) {
      packed_buffers[i] = api.clCreateBuffer(
          context, kClMemReadOnly, packed_by_expert[i].size(), nullptr, &err);
      Check(err, "clCreateBuffer(expert8 packed Q4)");
      Check(api.clEnqueueWriteBuffer(
                queue, packed_buffers[i], kClTrue, 0,
                packed_by_expert[i].size(), packed_by_expert[i].data(),
                0, nullptr, nullptr),
            "clEnqueueWriteBuffer(expert8 packed Q4)");
    }
    q8_qs_buffer = api.clCreateBuffer(
        context, kClMemReadOnly, q8.qs.size() * sizeof(std::int8_t),
        nullptr, &err);
    Check(err, "clCreateBuffer(expert8 q8 qs)");
    q8_bsums_buffer = api.clCreateBuffer(
        context, kClMemReadOnly, q8.bsums.size() * sizeof(std::int16_t),
        nullptr, &err);
    Check(err, "clCreateBuffer(expert8 q8 bsums)");
    q8_d_buffer = api.clCreateBuffer(
        context, kClMemReadOnly, q8.d.size() * sizeof(float), nullptr, &err);
    Check(err, "clCreateBuffer(expert8 q8 d)");
    output_buffer = api.clCreateBuffer(
        context, kClMemWriteOnly, run.output.size() * sizeof(float),
        nullptr, &err);
    Check(err, "clCreateBuffer(expert8 output)");
    Check(api.clEnqueueWriteBuffer(queue, q8_qs_buffer, kClTrue, 0,
                                   q8.qs.size() * sizeof(std::int8_t),
                                   q8.qs.data(), 0, nullptr, nullptr),
          "clEnqueueWriteBuffer(expert8 q8 qs)");
    Check(api.clEnqueueWriteBuffer(
              queue, q8_bsums_buffer, kClTrue, 0,
              q8.bsums.size() * sizeof(std::int16_t), q8.bsums.data(),
              0, nullptr, nullptr),
          "clEnqueueWriteBuffer(expert8 q8 bsums)");
    Check(api.clEnqueueWriteBuffer(queue, q8_d_buffer, kClTrue, 0,
                                   q8.d.size() * sizeof(float), q8.d.data(),
                                   0, nullptr, nullptr),
          "clEnqueueWriteBuffer(expert8 q8 d)");
    const cl_uint blocks_arg = static_cast<cl_uint>(blocks_per_row);
    const cl_uint row_groups_arg = static_cast<cl_uint>(rows_per_expert / 8);
    for (std::size_t i = 0; i < packed_buffers.size(); ++i) {
      Check(api.clSetKernelArg(kernel, static_cast<cl_uint>(i),
                               sizeof(packed_buffers[i]), &packed_buffers[i]),
            "clSetKernelArg(expert8 packed)");
    }
    Check(api.clSetKernelArg(kernel, 8, sizeof(q8_qs_buffer), &q8_qs_buffer),
          "clSetKernelArg(expert8 q8 qs)");
    Check(api.clSetKernelArg(kernel, 9, sizeof(q8_bsums_buffer), &q8_bsums_buffer),
          "clSetKernelArg(expert8 q8 bsums)");
    Check(api.clSetKernelArg(kernel, 10, sizeof(q8_d_buffer), &q8_d_buffer),
          "clSetKernelArg(expert8 q8 d)");
    Check(api.clSetKernelArg(kernel, 11, sizeof(blocks_arg), &blocks_arg),
          "clSetKernelArg(expert8 blocks)");
    Check(api.clSetKernelArg(kernel, 12, sizeof(row_groups_arg), &row_groups_arg),
          "clSetKernelArg(expert8 row groups)");
    if (weights != nullptr) {
      weights_buffer = api.clCreateBuffer(
          context, kClMemReadOnly, weights->size() * sizeof(float),
          nullptr, &err);
      Check(err, "clCreateBuffer(expert8 weights)");
      Check(api.clEnqueueWriteBuffer(queue, weights_buffer, kClTrue, 0,
                                     weights->size() * sizeof(float),
                                     weights->data(), 0, nullptr, nullptr),
            "clEnqueueWriteBuffer(expert8 weights)");
      Check(api.clSetKernelArg(kernel, 13, sizeof(weights_buffer), &weights_buffer),
            "clSetKernelArg(expert8 weights)");
      Check(api.clSetKernelArg(kernel, 14, sizeof(output_buffer), &output_buffer),
            "clSetKernelArg(expert8 weighted output)");
      if (group_reduce_kernel) {
        constexpr std::size_t kGroupReduceLocalItems = 32;
        Check(api.clSetKernelArg(kernel, 15,
                                 kGroupReduceLocalItems * sizeof(float),
                                 nullptr),
              "clSetKernelArg(expert8 group-reduce local)");
      }
    } else {
      Check(api.clSetKernelArg(kernel, 13, sizeof(output_buffer), &output_buffer),
            "clSetKernelArg(expert8 output)");
    }

    constexpr std::size_t kGroupReduceLocalItems = 32;
    const std::size_t global =
        static_cast<std::size_t>(
            group_reduce_kernel
                ? selected_count * rows_per_expert
                : (weighted_sum_kernel ? rows_per_expert
                                       : (pair2_kernel ? (selected_count / 2) * rows_per_expert
                                                       : selected_count * rows_per_expert * output_groups)));
    const std::size_t* local_ptr = nullptr;
    std::size_t local = 0;
    if (group_reduce_kernel) {
      Require(global % kGroupReduceLocalItems == 0,
              "expert8 group-reduce global/local mismatch");
      local = kGroupReduceLocalItems;
      local_ptr = &local;
    }
    std::vector<double> times;
    times.reserve(static_cast<std::size_t>(repeat));
    for (int i = 0; i < repeat; ++i) {
      cl_event event = nullptr;
      Check(api.clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global,
                                       local_ptr, 0, nullptr, &event),
            std::string("clEnqueueNDRangeKernel(") + kernel_name + ")");
      Check(api.clFinish(queue), "clFinish(expert8 selected Q4 component)");
      times.push_back(EventUs(api, event));
      api.clReleaseEvent(event);
    }
    Check(api.clEnqueueReadBuffer(queue, output_buffer, kClTrue, 0,
                                  run.output.size() * sizeof(float),
                                  run.output.data(), 0, nullptr, nullptr),
          "clEnqueueReadBuffer(expert8 selected Q4 component output)");
    run.timing.min_us = *std::min_element(times.begin(), times.end());
    run.timing.mean_us =
        std::accumulate(times.begin(), times.end(), 0.0) /
        static_cast<double>(times.size());
    const double logical_group_count = static_cast<double>(output_groups);
    const double raw_bytes =
        static_cast<double>(selected_raw.size()) * logical_group_count;
    const double io_bytes = raw_bytes +
        static_cast<double>(q8.qs.size() * sizeof(std::int8_t)) * logical_group_count +
        static_cast<double>(q8.bsums.size() * sizeof(std::int16_t)) * logical_group_count +
        static_cast<double>(q8.d.size() * sizeof(float)) * logical_group_count +
        (weights != nullptr
             ? static_cast<double>(weights->size() * sizeof(float))
             : 0.0) +
        static_cast<double>(run.output.size() * sizeof(float));
    run.timing.effective_raw_gb_s = raw_bytes / (run.timing.min_us / 1e6) / 1e9;
    run.timing.effective_io_gb_s = io_bytes / (run.timing.min_us / 1e6) / 1e9;
    run.timing.global_work_items = global;
    run.timing.kernel_launches = 1;
  } catch (...) {
    ReleaseMem(api, &output_buffer);
    ReleaseMem(api, &weights_buffer);
    ReleaseMem(api, &q8_d_buffer);
    ReleaseMem(api, &q8_bsums_buffer);
    ReleaseMem(api, &q8_qs_buffer);
    for (auto& packed_buffer : packed_buffers) {
      ReleaseMem(api, &packed_buffer);
    }
    api.clReleaseKernel(kernel);
    api.clReleaseProgram(program);
    api.clReleaseCommandQueue(queue);
    api.clReleaseContext(context);
    throw;
  }
  ReleaseMem(api, &output_buffer);
  ReleaseMem(api, &weights_buffer);
  ReleaseMem(api, &q8_d_buffer);
  ReleaseMem(api, &q8_bsums_buffer);
  ReleaseMem(api, &q8_qs_buffer);
  for (auto& packed_buffer : packed_buffers) {
    ReleaseMem(api, &packed_buffer);
  }
  api.clReleaseKernel(kernel);
  api.clReleaseProgram(program);
  api.clReleaseCommandQueue(queue);
  api.clReleaseContext(context);
  return run;
}

DeviceQ8DownRun RunGpuSelectedDownQ4DeviceQ8Component(
    const std::vector<std::uint8_t>& selected_raw,
    const std::vector<float>& input,
    std::uint64_t rows_per_expert,
    std::uint64_t blocks_per_row,
    std::uint64_t selected_count,
    const std::string& device_substring,
    int repeat) {
  Require(selected_count == 8,
          "device-Q8 selected Q4 component probe requires eight experts");
  Require(rows_per_expert % 8 == 0,
          "device-Q8 selected Q4 component probe requires x8 row packing");
  Require(input.size() == selected_count * blocks_per_row * kQ8BlockValues,
          "device-Q8 selected-down input size mismatch");
  Require(selected_raw.size() ==
              static_cast<std::size_t>(selected_count * rows_per_expert *
                                       blocks_per_row * kQ4KBlockBytes),
          "device-Q8 selected-down Q4 raw byte size mismatch");
  std::vector<std::vector<std::uint8_t>> packed_by_expert;
  packed_by_expert.reserve(static_cast<std::size_t>(selected_count));
  for (std::uint64_t selected = 0; selected < selected_count; ++selected) {
    const auto raw_begin = selected_raw.begin() +
        static_cast<std::ptrdiff_t>(selected * rows_per_expert *
                                    blocks_per_row * kQ4KBlockBytes);
    const auto raw_end = raw_begin +
        static_cast<std::ptrdiff_t>(rows_per_expert * blocks_per_row *
                                    kQ4KBlockBytes);
    packed_by_expert.push_back(
        iq36::PackQ4Kx8(
            std::vector<std::uint8_t>(raw_begin, raw_end),
            rows_per_expert, blocks_per_row));
  }

  DeviceQ8DownRun run;
  run.output.assign(static_cast<std::size_t>(selected_count * rows_per_expert), 0.0f);
  run.q8.blocks_per_expert = blocks_per_row;
  run.q8.qs.assign(input.size(), 0);
  run.q8.bsums.assign(
      static_cast<std::size_t>(selected_count * blocks_per_row * 16), 0);
  run.q8.d.assign(static_cast<std::size_t>(selected_count * blocks_per_row), 0.0f);

  OpenClApi api;
  const auto selected_device = SelectDevice(api, device_substring);
  run.platform_name = selected_device.platform_name;
  run.device_name = selected_device.device_name;

  cl_int err = kClSuccess;
  cl_context context =
      api.clCreateContext(nullptr, 1, &selected_device.device, nullptr, nullptr, &err);
  Check(err, "clCreateContext");
  cl_command_queue queue =
      api.clCreateCommandQueue(context, selected_device.device,
                               kClQueueProfilingEnable, &err);
  Check(err, "clCreateCommandQueue");
  const char* source = kOpenClSource;
  const std::size_t source_len = std::strlen(kOpenClSource);
  cl_program program =
      api.clCreateProgramWithSource(context, 1, &source, &source_len, &err);
  Check(err, "clCreateProgramWithSource");
  const auto build_begin = std::chrono::steady_clock::now();
  err = api.clBuildProgram(program, 1, &selected_device.device, "", nullptr, nullptr);
  const auto build_end = std::chrono::steady_clock::now();
  run.program_build_ms =
      std::chrono::duration<double, std::milli>(build_end - build_begin).count();
  run.build_log = BuildLog(api, program, selected_device.device);
  if (err != kClSuccess) {
    throw std::runtime_error(
        std::string("clBuildProgram failed: ") + run.build_log);
  }
  Check(err, "clBuildProgram");
  cl_kernel quant_kernel =
      api.clCreateKernel(program, "q8k_quantize_f32_blocks_with_bsums", &err);
  Check(err, "clCreateKernel(q8k_quantize_f32_blocks_with_bsums)");
  cl_kernel down_kernel =
      api.clCreateKernel(program, "q4k_x8_matvec_rowlane_expert8_multiq8", &err);
  Check(err, "clCreateKernel(q4k_x8_matvec_rowlane_expert8_multiq8)");

  std::array<cl_mem, 8> packed_buffers{};
  cl_mem input_buffer = nullptr;
  cl_mem q8_qs_buffer = nullptr;
  cl_mem q8_bsums_buffer = nullptr;
  cl_mem q8_d_buffer = nullptr;
  cl_mem output_buffer = nullptr;
  try {
    for (std::size_t i = 0; i < packed_buffers.size(); ++i) {
      packed_buffers[i] = api.clCreateBuffer(
          context, kClMemReadOnly, packed_by_expert[i].size(), nullptr, &err);
      Check(err, "clCreateBuffer(device-Q8 packed Q4)");
      Check(api.clEnqueueWriteBuffer(
                queue, packed_buffers[i], kClTrue, 0,
                packed_by_expert[i].size(), packed_by_expert[i].data(),
                0, nullptr, nullptr),
            "clEnqueueWriteBuffer(device-Q8 packed Q4)");
    }
    input_buffer = api.clCreateBuffer(
        context, kClMemReadOnly, input.size() * sizeof(float), nullptr, &err);
    Check(err, "clCreateBuffer(device-Q8 input)");
    q8_qs_buffer = api.clCreateBuffer(
        context, kClMemReadWrite, run.q8.qs.size() * sizeof(std::int8_t),
        nullptr, &err);
    Check(err, "clCreateBuffer(device-Q8 q8 qs)");
    q8_bsums_buffer = api.clCreateBuffer(
        context, kClMemReadWrite, run.q8.bsums.size() * sizeof(std::int16_t),
        nullptr, &err);
    Check(err, "clCreateBuffer(device-Q8 q8 bsums)");
    q8_d_buffer = api.clCreateBuffer(
        context, kClMemReadWrite, run.q8.d.size() * sizeof(float), nullptr, &err);
    Check(err, "clCreateBuffer(device-Q8 q8 d)");
    output_buffer = api.clCreateBuffer(
        context, kClMemWriteOnly, run.output.size() * sizeof(float),
        nullptr, &err);
    Check(err, "clCreateBuffer(device-Q8 output)");
    Check(api.clEnqueueWriteBuffer(queue, input_buffer, kClTrue, 0,
                                   input.size() * sizeof(float),
                                   input.data(), 0, nullptr, nullptr),
          "clEnqueueWriteBuffer(device-Q8 input)");

    const cl_uint block_count_arg =
        static_cast<cl_uint>(selected_count * blocks_per_row);
    Check(api.clSetKernelArg(quant_kernel, 0, sizeof(input_buffer), &input_buffer),
          "clSetKernelArg(device-Q8 quant input)");
    Check(api.clSetKernelArg(quant_kernel, 1, sizeof(block_count_arg), &block_count_arg),
          "clSetKernelArg(device-Q8 quant block count)");
    Check(api.clSetKernelArg(quant_kernel, 2, sizeof(q8_qs_buffer), &q8_qs_buffer),
          "clSetKernelArg(device-Q8 quant qs)");
    Check(api.clSetKernelArg(quant_kernel, 3, sizeof(q8_bsums_buffer), &q8_bsums_buffer),
          "clSetKernelArg(device-Q8 quant bsums)");
    Check(api.clSetKernelArg(quant_kernel, 4, sizeof(q8_d_buffer), &q8_d_buffer),
          "clSetKernelArg(device-Q8 quant d)");

    const cl_uint blocks_arg = static_cast<cl_uint>(blocks_per_row);
    const cl_uint row_groups_arg = static_cast<cl_uint>(rows_per_expert / 8);
    for (std::size_t i = 0; i < packed_buffers.size(); ++i) {
      Check(api.clSetKernelArg(down_kernel, static_cast<cl_uint>(i),
                               sizeof(packed_buffers[i]), &packed_buffers[i]),
            "clSetKernelArg(device-Q8 packed)");
    }
    Check(api.clSetKernelArg(down_kernel, 8, sizeof(q8_qs_buffer), &q8_qs_buffer),
          "clSetKernelArg(device-Q8 q8 qs)");
    Check(api.clSetKernelArg(down_kernel, 9, sizeof(q8_bsums_buffer), &q8_bsums_buffer),
          "clSetKernelArg(device-Q8 q8 bsums)");
    Check(api.clSetKernelArg(down_kernel, 10, sizeof(q8_d_buffer), &q8_d_buffer),
          "clSetKernelArg(device-Q8 q8 d)");
    Check(api.clSetKernelArg(down_kernel, 11, sizeof(blocks_arg), &blocks_arg),
          "clSetKernelArg(device-Q8 blocks)");
    Check(api.clSetKernelArg(down_kernel, 12, sizeof(row_groups_arg), &row_groups_arg),
          "clSetKernelArg(device-Q8 row groups)");
    Check(api.clSetKernelArg(down_kernel, 13, sizeof(output_buffer), &output_buffer),
          "clSetKernelArg(device-Q8 output)");

    const std::size_t q8_global =
        static_cast<std::size_t>(selected_count * blocks_per_row);
    const std::size_t down_global =
        static_cast<std::size_t>(selected_count * rows_per_expert);
    std::vector<double> quant_times;
    std::vector<double> down_times;
    std::vector<double> shell_times;
    quant_times.reserve(static_cast<std::size_t>(repeat));
    down_times.reserve(static_cast<std::size_t>(repeat));
    shell_times.reserve(static_cast<std::size_t>(repeat));
    for (int i = 0; i < repeat; ++i) {
      cl_event quant_event = nullptr;
      cl_event down_event = nullptr;
      Check(api.clEnqueueNDRangeKernel(queue, quant_kernel, 1, nullptr,
                                       &q8_global, nullptr, 0, nullptr,
                                       &quant_event),
            "clEnqueueNDRangeKernel(device-Q8 quantize)");
      Check(api.clEnqueueNDRangeKernel(queue, down_kernel, 1, nullptr,
                                       &down_global, nullptr, 0, nullptr,
                                       &down_event),
            "clEnqueueNDRangeKernel(device-Q8 selected Q4 down)");
      Check(api.clFinish(queue), "clFinish(device-Q8 selected Q4 component)");
      const double quant_us = EventUs(api, quant_event);
      const double down_us = EventUs(api, down_event);
      quant_times.push_back(quant_us);
      down_times.push_back(down_us);
      shell_times.push_back(quant_us + down_us);
      api.clReleaseEvent(quant_event);
      api.clReleaseEvent(down_event);
    }
    Check(api.clEnqueueReadBuffer(queue, output_buffer, kClTrue, 0,
                                  run.output.size() * sizeof(float),
                                  run.output.data(), 0, nullptr, nullptr),
          "clEnqueueReadBuffer(device-Q8 selected Q4 output)");
    Check(api.clEnqueueReadBuffer(queue, q8_qs_buffer, kClTrue, 0,
                                  run.q8.qs.size() * sizeof(std::int8_t),
                                  run.q8.qs.data(), 0, nullptr, nullptr),
          "clEnqueueReadBuffer(device-Q8 q8 qs)");
    Check(api.clEnqueueReadBuffer(
              queue, q8_bsums_buffer, kClTrue, 0,
              run.q8.bsums.size() * sizeof(std::int16_t),
              run.q8.bsums.data(), 0, nullptr, nullptr),
          "clEnqueueReadBuffer(device-Q8 q8 bsums)");
    Check(api.clEnqueueReadBuffer(queue, q8_d_buffer, kClTrue, 0,
                                  run.q8.d.size() * sizeof(float),
                                  run.q8.d.data(), 0, nullptr, nullptr),
          "clEnqueueReadBuffer(device-Q8 q8 d)");

    run.q8_quantize_timing.min_us =
        *std::min_element(quant_times.begin(), quant_times.end());
    run.q8_quantize_timing.mean_us =
        std::accumulate(quant_times.begin(), quant_times.end(), 0.0) /
        static_cast<double>(quant_times.size());
    run.down_timing.min_us =
        *std::min_element(down_times.begin(), down_times.end());
    run.down_timing.mean_us =
        std::accumulate(down_times.begin(), down_times.end(), 0.0) /
        static_cast<double>(down_times.size());
    run.shell_sum_min_us =
        *std::min_element(shell_times.begin(), shell_times.end());
    run.shell_sum_mean_us =
        std::accumulate(shell_times.begin(), shell_times.end(), 0.0) /
        static_cast<double>(shell_times.size());
    const double q8_input_bytes =
        static_cast<double>(input.size() * sizeof(float));
    const double q8_output_bytes =
        static_cast<double>(run.q8.qs.size() * sizeof(std::int8_t)) +
        static_cast<double>(run.q8.bsums.size() * sizeof(std::int16_t)) +
        static_cast<double>(run.q8.d.size() * sizeof(float));
    run.q8_quantize_timing.effective_raw_gb_s =
        q8_input_bytes / (run.q8_quantize_timing.min_us / 1e6) / 1e9;
    run.q8_quantize_timing.effective_io_gb_s =
        (q8_input_bytes + q8_output_bytes) /
        (run.q8_quantize_timing.min_us / 1e6) / 1e9;
    run.q8_quantize_timing.global_work_items = q8_global;
    run.q8_quantize_timing.kernel_launches = 1;

    const double raw_bytes = static_cast<double>(selected_raw.size());
    const double down_io_bytes = raw_bytes + q8_output_bytes +
        static_cast<double>(run.output.size() * sizeof(float));
    run.down_timing.effective_raw_gb_s =
        raw_bytes / (run.down_timing.min_us / 1e6) / 1e9;
    run.down_timing.effective_io_gb_s =
        down_io_bytes / (run.down_timing.min_us / 1e6) / 1e9;
    run.down_timing.global_work_items = down_global;
    run.down_timing.kernel_launches = 1;
  } catch (...) {
    ReleaseMem(api, &output_buffer);
    ReleaseMem(api, &q8_d_buffer);
    ReleaseMem(api, &q8_bsums_buffer);
    ReleaseMem(api, &q8_qs_buffer);
    ReleaseMem(api, &input_buffer);
    for (auto& packed_buffer : packed_buffers) {
      ReleaseMem(api, &packed_buffer);
    }
    api.clReleaseKernel(down_kernel);
    api.clReleaseKernel(quant_kernel);
    api.clReleaseProgram(program);
    api.clReleaseCommandQueue(queue);
    api.clReleaseContext(context);
    throw;
  }
  ReleaseMem(api, &output_buffer);
  ReleaseMem(api, &q8_d_buffer);
  ReleaseMem(api, &q8_bsums_buffer);
  ReleaseMem(api, &q8_qs_buffer);
  ReleaseMem(api, &input_buffer);
  for (auto& packed_buffer : packed_buffers) {
    ReleaseMem(api, &packed_buffer);
  }
  api.clReleaseKernel(down_kernel);
  api.clReleaseKernel(quant_kernel);
  api.clReleaseProgram(program);
  api.clReleaseCommandQueue(queue);
  api.clReleaseContext(context);
  return run;
}

DeviceSwiGluQ8DownRun RunGpuSelectedDownQ4SwiGluDeviceQ8Component(
    const std::vector<std::uint8_t>& selected_raw,
    const std::vector<float>& gate_up,
    std::uint64_t rows_per_expert,
    std::uint64_t blocks_per_row,
    std::uint64_t selected_count,
    const std::string& device_substring,
    int repeat) {
  Require(selected_count == 8,
          "SwiGLU device-Q8 selected Q4 component probe requires eight experts");
  Require(rows_per_expert % 8 == 0,
          "SwiGLU device-Q8 selected Q4 component probe requires x8 row packing");
  Require(gate_up.size() == selected_count * kIntermediateSize * 2,
          "SwiGLU device-Q8 gate/up input size mismatch");
  Require(selected_raw.size() ==
              static_cast<std::size_t>(selected_count * rows_per_expert *
                                       blocks_per_row * kQ4KBlockBytes),
          "SwiGLU device-Q8 selected-down Q4 raw byte size mismatch");
  std::vector<std::vector<std::uint8_t>> packed_by_expert;
  packed_by_expert.reserve(static_cast<std::size_t>(selected_count));
  for (std::uint64_t selected = 0; selected < selected_count; ++selected) {
    const auto raw_begin = selected_raw.begin() +
        static_cast<std::ptrdiff_t>(selected * rows_per_expert *
                                    blocks_per_row * kQ4KBlockBytes);
    const auto raw_end = raw_begin +
        static_cast<std::ptrdiff_t>(rows_per_expert * blocks_per_row *
                                    kQ4KBlockBytes);
    packed_by_expert.push_back(
        iq36::PackQ4Kx8(
            std::vector<std::uint8_t>(raw_begin, raw_end),
            rows_per_expert, blocks_per_row));
  }

  DeviceSwiGluQ8DownRun run;
  run.swiglu.assign(
      static_cast<std::size_t>(selected_count * kIntermediateSize), 0.0f);
  run.output.assign(
      static_cast<std::size_t>(selected_count * rows_per_expert), 0.0f);
  run.q8.blocks_per_expert = blocks_per_row;
  run.q8.qs.assign(run.swiglu.size(), 0);
  run.q8.bsums.assign(
      static_cast<std::size_t>(selected_count * blocks_per_row * 16), 0);
  run.q8.d.assign(static_cast<std::size_t>(selected_count * blocks_per_row), 0.0f);

  OpenClApi api;
  const auto selected_device = SelectDevice(api, device_substring);
  run.platform_name = selected_device.platform_name;
  run.device_name = selected_device.device_name;

  cl_int err = kClSuccess;
  cl_context context =
      api.clCreateContext(nullptr, 1, &selected_device.device, nullptr, nullptr, &err);
  Check(err, "clCreateContext");
  cl_command_queue queue =
      api.clCreateCommandQueue(context, selected_device.device,
                               kClQueueProfilingEnable, &err);
  Check(err, "clCreateCommandQueue");
  const char* source = kOpenClSource;
  const std::size_t source_len = std::strlen(kOpenClSource);
  cl_program program =
      api.clCreateProgramWithSource(context, 1, &source, &source_len, &err);
  Check(err, "clCreateProgramWithSource");
  const auto build_begin = std::chrono::steady_clock::now();
  err = api.clBuildProgram(program, 1, &selected_device.device, "", nullptr, nullptr);
  const auto build_end = std::chrono::steady_clock::now();
  run.program_build_ms =
      std::chrono::duration<double, std::milli>(build_end - build_begin).count();
  run.build_log = BuildLog(api, program, selected_device.device);
  if (err != kClSuccess) {
    throw std::runtime_error(
        std::string("clBuildProgram failed: ") + run.build_log);
  }
  Check(err, "clBuildProgram");
  cl_kernel swiglu_kernel =
      api.clCreateKernel(program, "ffn_moe_swiglu_f32", &err);
  Check(err, "clCreateKernel(ffn_moe_swiglu_f32)");
  cl_kernel quant_kernel =
      api.clCreateKernel(program, "q8k_quantize_f32_blocks_with_bsums", &err);
  Check(err, "clCreateKernel(q8k_quantize_f32_blocks_with_bsums)");
  cl_kernel down_kernel =
      api.clCreateKernel(program, "q4k_x8_matvec_rowlane_expert8_multiq8", &err);
  Check(err, "clCreateKernel(q4k_x8_matvec_rowlane_expert8_multiq8)");

  std::array<cl_mem, 8> packed_buffers{};
  cl_mem gate_up_buffer = nullptr;
  cl_mem swiglu_buffer = nullptr;
  cl_mem q8_qs_buffer = nullptr;
  cl_mem q8_bsums_buffer = nullptr;
  cl_mem q8_d_buffer = nullptr;
  cl_mem output_buffer = nullptr;
  try {
    for (std::size_t i = 0; i < packed_buffers.size(); ++i) {
      packed_buffers[i] = api.clCreateBuffer(
          context, kClMemReadOnly, packed_by_expert[i].size(), nullptr, &err);
      Check(err, "clCreateBuffer(SwiGLU device-Q8 packed Q4)");
      Check(api.clEnqueueWriteBuffer(
                queue, packed_buffers[i], kClTrue, 0,
                packed_by_expert[i].size(), packed_by_expert[i].data(),
                0, nullptr, nullptr),
            "clEnqueueWriteBuffer(SwiGLU device-Q8 packed Q4)");
    }
    gate_up_buffer = api.clCreateBuffer(
        context, kClMemReadOnly, gate_up.size() * sizeof(float), nullptr, &err);
    Check(err, "clCreateBuffer(SwiGLU device-Q8 gate/up)");
    swiglu_buffer = api.clCreateBuffer(
        context, kClMemReadWrite, run.swiglu.size() * sizeof(float), nullptr, &err);
    Check(err, "clCreateBuffer(SwiGLU device-Q8 swiglu)");
    q8_qs_buffer = api.clCreateBuffer(
        context, kClMemReadWrite, run.q8.qs.size() * sizeof(std::int8_t),
        nullptr, &err);
    Check(err, "clCreateBuffer(SwiGLU device-Q8 q8 qs)");
    q8_bsums_buffer = api.clCreateBuffer(
        context, kClMemReadWrite, run.q8.bsums.size() * sizeof(std::int16_t),
        nullptr, &err);
    Check(err, "clCreateBuffer(SwiGLU device-Q8 q8 bsums)");
    q8_d_buffer = api.clCreateBuffer(
        context, kClMemReadWrite, run.q8.d.size() * sizeof(float), nullptr, &err);
    Check(err, "clCreateBuffer(SwiGLU device-Q8 q8 d)");
    output_buffer = api.clCreateBuffer(
        context, kClMemWriteOnly, run.output.size() * sizeof(float),
        nullptr, &err);
    Check(err, "clCreateBuffer(SwiGLU device-Q8 output)");
    Check(api.clEnqueueWriteBuffer(queue, gate_up_buffer, kClTrue, 0,
                                   gate_up.size() * sizeof(float),
                                   gate_up.data(), 0, nullptr, nullptr),
          "clEnqueueWriteBuffer(SwiGLU device-Q8 gate/up)");

    const cl_uint intermediate_arg = static_cast<cl_uint>(kIntermediateSize);
    const cl_uint selected_count_arg = static_cast<cl_uint>(selected_count);
    Check(api.clSetKernelArg(swiglu_kernel, 0,
                             sizeof(gate_up_buffer), &gate_up_buffer),
          "clSetKernelArg(SwiGLU device-Q8 swiglu input)");
    Check(api.clSetKernelArg(swiglu_kernel, 1,
                             sizeof(intermediate_arg), &intermediate_arg),
          "clSetKernelArg(SwiGLU device-Q8 intermediate)");
    Check(api.clSetKernelArg(swiglu_kernel, 2,
                             sizeof(selected_count_arg), &selected_count_arg),
          "clSetKernelArg(SwiGLU device-Q8 selected count)");
    Check(api.clSetKernelArg(swiglu_kernel, 3,
                             sizeof(swiglu_buffer), &swiglu_buffer),
          "clSetKernelArg(SwiGLU device-Q8 swiglu output)");

    const cl_uint block_count_arg =
        static_cast<cl_uint>(selected_count * blocks_per_row);
    Check(api.clSetKernelArg(quant_kernel, 0, sizeof(swiglu_buffer), &swiglu_buffer),
          "clSetKernelArg(SwiGLU device-Q8 quant input)");
    Check(api.clSetKernelArg(quant_kernel, 1, sizeof(block_count_arg), &block_count_arg),
          "clSetKernelArg(SwiGLU device-Q8 quant block count)");
    Check(api.clSetKernelArg(quant_kernel, 2, sizeof(q8_qs_buffer), &q8_qs_buffer),
          "clSetKernelArg(SwiGLU device-Q8 quant qs)");
    Check(api.clSetKernelArg(quant_kernel, 3, sizeof(q8_bsums_buffer), &q8_bsums_buffer),
          "clSetKernelArg(SwiGLU device-Q8 quant bsums)");
    Check(api.clSetKernelArg(quant_kernel, 4, sizeof(q8_d_buffer), &q8_d_buffer),
          "clSetKernelArg(SwiGLU device-Q8 quant d)");

    const cl_uint blocks_arg = static_cast<cl_uint>(blocks_per_row);
    const cl_uint row_groups_arg = static_cast<cl_uint>(rows_per_expert / 8);
    for (std::size_t i = 0; i < packed_buffers.size(); ++i) {
      Check(api.clSetKernelArg(down_kernel, static_cast<cl_uint>(i),
                               sizeof(packed_buffers[i]), &packed_buffers[i]),
            "clSetKernelArg(SwiGLU device-Q8 packed)");
    }
    Check(api.clSetKernelArg(down_kernel, 8, sizeof(q8_qs_buffer), &q8_qs_buffer),
          "clSetKernelArg(SwiGLU device-Q8 q8 qs)");
    Check(api.clSetKernelArg(down_kernel, 9, sizeof(q8_bsums_buffer), &q8_bsums_buffer),
          "clSetKernelArg(SwiGLU device-Q8 q8 bsums)");
    Check(api.clSetKernelArg(down_kernel, 10, sizeof(q8_d_buffer), &q8_d_buffer),
          "clSetKernelArg(SwiGLU device-Q8 q8 d)");
    Check(api.clSetKernelArg(down_kernel, 11, sizeof(blocks_arg), &blocks_arg),
          "clSetKernelArg(SwiGLU device-Q8 blocks)");
    Check(api.clSetKernelArg(down_kernel, 12, sizeof(row_groups_arg), &row_groups_arg),
          "clSetKernelArg(SwiGLU device-Q8 row groups)");
    Check(api.clSetKernelArg(down_kernel, 13, sizeof(output_buffer), &output_buffer),
          "clSetKernelArg(SwiGLU device-Q8 output)");

    const std::size_t swiglu_global = run.swiglu.size();
    const std::size_t q8_global =
        static_cast<std::size_t>(selected_count * blocks_per_row);
    const std::size_t down_global =
        static_cast<std::size_t>(selected_count * rows_per_expert);
    std::vector<double> swiglu_times;
    std::vector<double> quant_times;
    std::vector<double> down_times;
    std::vector<double> shell_times;
    swiglu_times.reserve(static_cast<std::size_t>(repeat));
    quant_times.reserve(static_cast<std::size_t>(repeat));
    down_times.reserve(static_cast<std::size_t>(repeat));
    shell_times.reserve(static_cast<std::size_t>(repeat));
    for (int i = 0; i < repeat; ++i) {
      cl_event swiglu_event = nullptr;
      cl_event quant_event = nullptr;
      cl_event down_event = nullptr;
      Check(api.clEnqueueNDRangeKernel(queue, swiglu_kernel, 1, nullptr,
                                       &swiglu_global, nullptr, 0, nullptr,
                                       &swiglu_event),
            "clEnqueueNDRangeKernel(SwiGLU device-Q8 swiglu)");
      Check(api.clEnqueueNDRangeKernel(queue, quant_kernel, 1, nullptr,
                                       &q8_global, nullptr, 0, nullptr,
                                       &quant_event),
            "clEnqueueNDRangeKernel(SwiGLU device-Q8 quantize)");
      Check(api.clEnqueueNDRangeKernel(queue, down_kernel, 1, nullptr,
                                       &down_global, nullptr, 0, nullptr,
                                       &down_event),
            "clEnqueueNDRangeKernel(SwiGLU device-Q8 selected Q4 down)");
      Check(api.clFinish(queue), "clFinish(SwiGLU device-Q8 selected Q4 component)");
      const double swiglu_us = EventUs(api, swiglu_event);
      const double quant_us = EventUs(api, quant_event);
      const double down_us = EventUs(api, down_event);
      swiglu_times.push_back(swiglu_us);
      quant_times.push_back(quant_us);
      down_times.push_back(down_us);
      shell_times.push_back(swiglu_us + quant_us + down_us);
      api.clReleaseEvent(swiglu_event);
      api.clReleaseEvent(quant_event);
      api.clReleaseEvent(down_event);
    }
    Check(api.clEnqueueReadBuffer(queue, swiglu_buffer, kClTrue, 0,
                                  run.swiglu.size() * sizeof(float),
                                  run.swiglu.data(), 0, nullptr, nullptr),
          "clEnqueueReadBuffer(SwiGLU device-Q8 swiglu)");
    Check(api.clEnqueueReadBuffer(queue, output_buffer, kClTrue, 0,
                                  run.output.size() * sizeof(float),
                                  run.output.data(), 0, nullptr, nullptr),
          "clEnqueueReadBuffer(SwiGLU device-Q8 selected Q4 output)");
    Check(api.clEnqueueReadBuffer(queue, q8_qs_buffer, kClTrue, 0,
                                  run.q8.qs.size() * sizeof(std::int8_t),
                                  run.q8.qs.data(), 0, nullptr, nullptr),
          "clEnqueueReadBuffer(SwiGLU device-Q8 q8 qs)");
    Check(api.clEnqueueReadBuffer(
              queue, q8_bsums_buffer, kClTrue, 0,
              run.q8.bsums.size() * sizeof(std::int16_t),
              run.q8.bsums.data(), 0, nullptr, nullptr),
          "clEnqueueReadBuffer(SwiGLU device-Q8 q8 bsums)");
    Check(api.clEnqueueReadBuffer(queue, q8_d_buffer, kClTrue, 0,
                                  run.q8.d.size() * sizeof(float),
                                  run.q8.d.data(), 0, nullptr, nullptr),
          "clEnqueueReadBuffer(SwiGLU device-Q8 q8 d)");

    run.swiglu_timing.min_us =
        *std::min_element(swiglu_times.begin(), swiglu_times.end());
    run.swiglu_timing.mean_us =
        std::accumulate(swiglu_times.begin(), swiglu_times.end(), 0.0) /
        static_cast<double>(swiglu_times.size());
    run.q8_quantize_timing.min_us =
        *std::min_element(quant_times.begin(), quant_times.end());
    run.q8_quantize_timing.mean_us =
        std::accumulate(quant_times.begin(), quant_times.end(), 0.0) /
        static_cast<double>(quant_times.size());
    run.down_timing.min_us =
        *std::min_element(down_times.begin(), down_times.end());
    run.down_timing.mean_us =
        std::accumulate(down_times.begin(), down_times.end(), 0.0) /
        static_cast<double>(down_times.size());
    run.shell_sum_min_us =
        *std::min_element(shell_times.begin(), shell_times.end());
    run.shell_sum_mean_us =
        std::accumulate(shell_times.begin(), shell_times.end(), 0.0) /
        static_cast<double>(shell_times.size());

    const double gate_up_bytes =
        static_cast<double>(gate_up.size() * sizeof(float));
    const double swiglu_bytes =
        static_cast<double>(run.swiglu.size() * sizeof(float));
    const double q8_output_bytes =
        static_cast<double>(run.q8.qs.size() * sizeof(std::int8_t)) +
        static_cast<double>(run.q8.bsums.size() * sizeof(std::int16_t)) +
        static_cast<double>(run.q8.d.size() * sizeof(float));
    run.swiglu_timing.effective_raw_gb_s =
        gate_up_bytes / (run.swiglu_timing.min_us / 1e6) / 1e9;
    run.swiglu_timing.effective_io_gb_s =
        (gate_up_bytes + swiglu_bytes) /
        (run.swiglu_timing.min_us / 1e6) / 1e9;
    run.swiglu_timing.global_work_items = swiglu_global;
    run.swiglu_timing.kernel_launches = 1;
    run.q8_quantize_timing.effective_raw_gb_s =
        swiglu_bytes / (run.q8_quantize_timing.min_us / 1e6) / 1e9;
    run.q8_quantize_timing.effective_io_gb_s =
        (swiglu_bytes + q8_output_bytes) /
        (run.q8_quantize_timing.min_us / 1e6) / 1e9;
    run.q8_quantize_timing.global_work_items = q8_global;
    run.q8_quantize_timing.kernel_launches = 1;

    const double raw_bytes = static_cast<double>(selected_raw.size());
    const double down_io_bytes = raw_bytes + q8_output_bytes +
        static_cast<double>(run.output.size() * sizeof(float));
    run.down_timing.effective_raw_gb_s =
        raw_bytes / (run.down_timing.min_us / 1e6) / 1e9;
    run.down_timing.effective_io_gb_s =
        down_io_bytes / (run.down_timing.min_us / 1e6) / 1e9;
    run.down_timing.global_work_items = down_global;
    run.down_timing.kernel_launches = 1;
  } catch (...) {
    ReleaseMem(api, &output_buffer);
    ReleaseMem(api, &q8_d_buffer);
    ReleaseMem(api, &q8_bsums_buffer);
    ReleaseMem(api, &q8_qs_buffer);
    ReleaseMem(api, &swiglu_buffer);
    ReleaseMem(api, &gate_up_buffer);
    for (auto& packed_buffer : packed_buffers) {
      ReleaseMem(api, &packed_buffer);
    }
    api.clReleaseKernel(down_kernel);
    api.clReleaseKernel(quant_kernel);
    api.clReleaseKernel(swiglu_kernel);
    api.clReleaseProgram(program);
    api.clReleaseCommandQueue(queue);
    api.clReleaseContext(context);
    throw;
  }
  ReleaseMem(api, &output_buffer);
  ReleaseMem(api, &q8_d_buffer);
  ReleaseMem(api, &q8_bsums_buffer);
  ReleaseMem(api, &q8_qs_buffer);
  ReleaseMem(api, &swiglu_buffer);
  ReleaseMem(api, &gate_up_buffer);
  for (auto& packed_buffer : packed_buffers) {
    ReleaseMem(api, &packed_buffer);
  }
  api.clReleaseKernel(down_kernel);
  api.clReleaseKernel(quant_kernel);
  api.clReleaseKernel(swiglu_kernel);
  api.clReleaseProgram(program);
  api.clReleaseCommandQueue(queue);
  api.clReleaseContext(context);
  return run;
}

DeviceSwiGluF32InputDownRun RunGpuSelectedDownQ4SwiGluF32InputComponent(
    const std::vector<std::uint8_t>& selected_raw,
    const std::vector<float>& gate_up,
    std::uint64_t rows_per_expert,
    std::uint64_t blocks_per_row,
    std::uint64_t selected_count,
    const std::string& device_substring,
    int repeat) {
  Require(selected_count == 8,
          "SwiGLU f32-input selected Q4 component probe requires eight experts");
  Require(rows_per_expert % 8 == 0,
          "SwiGLU f32-input selected Q4 component probe requires x8 row packing");
  Require(gate_up.size() == selected_count * kIntermediateSize * 2,
          "SwiGLU f32-input gate/up input size mismatch");
  Require(selected_raw.size() ==
              static_cast<std::size_t>(selected_count * rows_per_expert *
                                       blocks_per_row * kQ4KBlockBytes),
          "SwiGLU f32-input selected-down Q4 raw byte size mismatch");
  std::vector<std::vector<std::uint8_t>> packed_by_expert;
  packed_by_expert.reserve(static_cast<std::size_t>(selected_count));
  for (std::uint64_t selected = 0; selected < selected_count; ++selected) {
    const auto raw_begin = selected_raw.begin() +
        static_cast<std::ptrdiff_t>(selected * rows_per_expert *
                                    blocks_per_row * kQ4KBlockBytes);
    const auto raw_end = raw_begin +
        static_cast<std::ptrdiff_t>(rows_per_expert * blocks_per_row *
                                    kQ4KBlockBytes);
    packed_by_expert.push_back(
        iq36::PackQ4Kx8(
            std::vector<std::uint8_t>(raw_begin, raw_end),
            rows_per_expert, blocks_per_row));
  }

  DeviceSwiGluF32InputDownRun run;
  run.swiglu.assign(
      static_cast<std::size_t>(selected_count * kIntermediateSize), 0.0f);
  run.output.assign(
      static_cast<std::size_t>(selected_count * rows_per_expert), 0.0f);

  OpenClApi api;
  const auto selected_device = SelectDevice(api, device_substring);
  run.platform_name = selected_device.platform_name;
  run.device_name = selected_device.device_name;

  cl_int err = kClSuccess;
  cl_context context =
      api.clCreateContext(nullptr, 1, &selected_device.device, nullptr, nullptr, &err);
  Check(err, "clCreateContext");
  cl_command_queue queue =
      api.clCreateCommandQueue(context, selected_device.device,
                               kClQueueProfilingEnable, &err);
  Check(err, "clCreateCommandQueue");
  const char* source = kOpenClSource;
  const std::size_t source_len = std::strlen(kOpenClSource);
  cl_program program =
      api.clCreateProgramWithSource(context, 1, &source, &source_len, &err);
  Check(err, "clCreateProgramWithSource");
  const auto build_begin = std::chrono::steady_clock::now();
  err = api.clBuildProgram(program, 1, &selected_device.device, "", nullptr, nullptr);
  const auto build_end = std::chrono::steady_clock::now();
  run.program_build_ms =
      std::chrono::duration<double, std::milli>(build_end - build_begin).count();
  run.build_log = BuildLog(api, program, selected_device.device);
  if (err != kClSuccess) {
    throw std::runtime_error(
        std::string("clBuildProgram failed: ") + run.build_log);
  }
  Check(err, "clBuildProgram");
  cl_kernel swiglu_kernel =
      api.clCreateKernel(program, "ffn_moe_swiglu_f32", &err);
  Check(err, "clCreateKernel(ffn_moe_swiglu_f32)");
  cl_kernel down_kernel =
      api.clCreateKernel(program, "q4k_x8_matvec_rowlane_expert8_f32input", &err);
  Check(err, "clCreateKernel(q4k_x8_matvec_rowlane_expert8_f32input)");

  std::array<cl_mem, 8> packed_buffers{};
  cl_mem gate_up_buffer = nullptr;
  cl_mem swiglu_buffer = nullptr;
  cl_mem output_buffer = nullptr;
  try {
    for (std::size_t i = 0; i < packed_buffers.size(); ++i) {
      packed_buffers[i] = api.clCreateBuffer(
          context, kClMemReadOnly, packed_by_expert[i].size(), nullptr, &err);
      Check(err, "clCreateBuffer(SwiGLU f32-input packed Q4)");
      Check(api.clEnqueueWriteBuffer(
                queue, packed_buffers[i], kClTrue, 0,
                packed_by_expert[i].size(), packed_by_expert[i].data(),
                0, nullptr, nullptr),
            "clEnqueueWriteBuffer(SwiGLU f32-input packed Q4)");
    }
    gate_up_buffer = api.clCreateBuffer(
        context, kClMemReadOnly, gate_up.size() * sizeof(float), nullptr, &err);
    Check(err, "clCreateBuffer(SwiGLU f32-input gate/up)");
    swiglu_buffer = api.clCreateBuffer(
        context, kClMemReadWrite, run.swiglu.size() * sizeof(float), nullptr, &err);
    Check(err, "clCreateBuffer(SwiGLU f32-input swiglu)");
    output_buffer = api.clCreateBuffer(
        context, kClMemWriteOnly, run.output.size() * sizeof(float),
        nullptr, &err);
    Check(err, "clCreateBuffer(SwiGLU f32-input output)");
    Check(api.clEnqueueWriteBuffer(queue, gate_up_buffer, kClTrue, 0,
                                   gate_up.size() * sizeof(float),
                                   gate_up.data(), 0, nullptr, nullptr),
          "clEnqueueWriteBuffer(SwiGLU f32-input gate/up)");

    const cl_uint intermediate_arg = static_cast<cl_uint>(kIntermediateSize);
    const cl_uint selected_count_arg = static_cast<cl_uint>(selected_count);
    Check(api.clSetKernelArg(swiglu_kernel, 0,
                             sizeof(gate_up_buffer), &gate_up_buffer),
          "clSetKernelArg(SwiGLU f32-input swiglu input)");
    Check(api.clSetKernelArg(swiglu_kernel, 1,
                             sizeof(intermediate_arg), &intermediate_arg),
          "clSetKernelArg(SwiGLU f32-input intermediate)");
    Check(api.clSetKernelArg(swiglu_kernel, 2,
                             sizeof(selected_count_arg), &selected_count_arg),
          "clSetKernelArg(SwiGLU f32-input selected count)");
    Check(api.clSetKernelArg(swiglu_kernel, 3,
                             sizeof(swiglu_buffer), &swiglu_buffer),
          "clSetKernelArg(SwiGLU f32-input swiglu output)");

    const cl_uint blocks_arg = static_cast<cl_uint>(blocks_per_row);
    const cl_uint row_groups_arg = static_cast<cl_uint>(rows_per_expert / 8);
    for (std::size_t i = 0; i < packed_buffers.size(); ++i) {
      Check(api.clSetKernelArg(down_kernel, static_cast<cl_uint>(i),
                               sizeof(packed_buffers[i]), &packed_buffers[i]),
            "clSetKernelArg(SwiGLU f32-input packed)");
    }
    Check(api.clSetKernelArg(down_kernel, 8, sizeof(swiglu_buffer), &swiglu_buffer),
          "clSetKernelArg(SwiGLU f32-input swiglu)");
    Check(api.clSetKernelArg(down_kernel, 9, sizeof(blocks_arg), &blocks_arg),
          "clSetKernelArg(SwiGLU f32-input blocks)");
    Check(api.clSetKernelArg(down_kernel, 10, sizeof(row_groups_arg), &row_groups_arg),
          "clSetKernelArg(SwiGLU f32-input row groups)");
    Check(api.clSetKernelArg(down_kernel, 11, sizeof(output_buffer), &output_buffer),
          "clSetKernelArg(SwiGLU f32-input output)");

    const std::size_t swiglu_global = run.swiglu.size();
    const std::size_t down_global =
        static_cast<std::size_t>(selected_count * rows_per_expert);
    std::vector<double> swiglu_times;
    std::vector<double> down_times;
    std::vector<double> shell_times;
    swiglu_times.reserve(static_cast<std::size_t>(repeat));
    down_times.reserve(static_cast<std::size_t>(repeat));
    shell_times.reserve(static_cast<std::size_t>(repeat));
    for (int i = 0; i < repeat; ++i) {
      cl_event swiglu_event = nullptr;
      cl_event down_event = nullptr;
      Check(api.clEnqueueNDRangeKernel(queue, swiglu_kernel, 1, nullptr,
                                       &swiglu_global, nullptr, 0, nullptr,
                                       &swiglu_event),
            "clEnqueueNDRangeKernel(SwiGLU f32-input swiglu)");
      Check(api.clEnqueueNDRangeKernel(queue, down_kernel, 1, nullptr,
                                       &down_global, nullptr, 0, nullptr,
                                       &down_event),
            "clEnqueueNDRangeKernel(SwiGLU f32-input selected Q4 down)");
      Check(api.clFinish(queue), "clFinish(SwiGLU f32-input selected Q4 component)");
      const double swiglu_us = EventUs(api, swiglu_event);
      const double down_us = EventUs(api, down_event);
      swiglu_times.push_back(swiglu_us);
      down_times.push_back(down_us);
      shell_times.push_back(swiglu_us + down_us);
      api.clReleaseEvent(swiglu_event);
      api.clReleaseEvent(down_event);
    }
    Check(api.clEnqueueReadBuffer(queue, swiglu_buffer, kClTrue, 0,
                                  run.swiglu.size() * sizeof(float),
                                  run.swiglu.data(), 0, nullptr, nullptr),
          "clEnqueueReadBuffer(SwiGLU f32-input swiglu)");
    Check(api.clEnqueueReadBuffer(queue, output_buffer, kClTrue, 0,
                                  run.output.size() * sizeof(float),
                                  run.output.data(), 0, nullptr, nullptr),
          "clEnqueueReadBuffer(SwiGLU f32-input selected Q4 output)");

    run.swiglu_timing.min_us =
        *std::min_element(swiglu_times.begin(), swiglu_times.end());
    run.swiglu_timing.mean_us =
        std::accumulate(swiglu_times.begin(), swiglu_times.end(), 0.0) /
        static_cast<double>(swiglu_times.size());
    run.down_timing.min_us =
        *std::min_element(down_times.begin(), down_times.end());
    run.down_timing.mean_us =
        std::accumulate(down_times.begin(), down_times.end(), 0.0) /
        static_cast<double>(down_times.size());
    run.shell_sum_min_us =
        *std::min_element(shell_times.begin(), shell_times.end());
    run.shell_sum_mean_us =
        std::accumulate(shell_times.begin(), shell_times.end(), 0.0) /
        static_cast<double>(shell_times.size());

    const double gate_up_bytes =
        static_cast<double>(gate_up.size() * sizeof(float));
    const double swiglu_bytes =
        static_cast<double>(run.swiglu.size() * sizeof(float));
    run.swiglu_timing.effective_raw_gb_s =
        gate_up_bytes / (run.swiglu_timing.min_us / 1e6) / 1e9;
    run.swiglu_timing.effective_io_gb_s =
        (gate_up_bytes + swiglu_bytes) /
        (run.swiglu_timing.min_us / 1e6) / 1e9;
    run.swiglu_timing.global_work_items = swiglu_global;
    run.swiglu_timing.kernel_launches = 1;

    const double raw_bytes = static_cast<double>(selected_raw.size());
    const double down_io_bytes = raw_bytes + swiglu_bytes +
        static_cast<double>(run.output.size() * sizeof(float));
    run.down_timing.effective_raw_gb_s =
        raw_bytes / (run.down_timing.min_us / 1e6) / 1e9;
    run.down_timing.effective_io_gb_s =
        down_io_bytes / (run.down_timing.min_us / 1e6) / 1e9;
    run.down_timing.global_work_items = down_global;
    run.down_timing.kernel_launches = 1;
  } catch (...) {
    ReleaseMem(api, &output_buffer);
    ReleaseMem(api, &swiglu_buffer);
    ReleaseMem(api, &gate_up_buffer);
    for (auto& packed_buffer : packed_buffers) {
      ReleaseMem(api, &packed_buffer);
    }
    api.clReleaseKernel(down_kernel);
    api.clReleaseKernel(swiglu_kernel);
    api.clReleaseProgram(program);
    api.clReleaseCommandQueue(queue);
    api.clReleaseContext(context);
    throw;
  }
  ReleaseMem(api, &output_buffer);
  ReleaseMem(api, &swiglu_buffer);
  ReleaseMem(api, &gate_up_buffer);
  for (auto& packed_buffer : packed_buffers) {
    ReleaseMem(api, &packed_buffer);
  }
  api.clReleaseKernel(down_kernel);
  api.clReleaseKernel(swiglu_kernel);
  api.clReleaseProgram(program);
  api.clReleaseCommandQueue(queue);
  api.clReleaseContext(context);
  return run;
}

DownRun RunGpuSelectedDownQ6(const std::vector<std::uint8_t>& selected_raw,
                             const Q8Planes& q8,
                             std::uint64_t rows_per_expert,
                             std::uint64_t blocks_per_row,
                             std::uint64_t selected_count,
                             const std::string& device_substring,
                             int repeat) {
  Require(selected_raw.size() ==
              static_cast<std::size_t>(selected_count * rows_per_expert *
                                       blocks_per_row * kQ6KBlockBytes),
          "selected-down raw byte size mismatch");
  Require(q8.blocks_per_expert == blocks_per_row, "selected-down q8 block count mismatch");
  DownRun run;
  run.output.assign(static_cast<std::size_t>(selected_count * rows_per_expert), 0.0f);
  OpenClApi api;
  const auto selected = SelectDevice(api, device_substring);
  run.platform_name = selected.platform_name;
  run.device_name = selected.device_name;

  cl_int err = kClSuccess;
  cl_context context = api.clCreateContext(nullptr, 1, &selected.device, nullptr, nullptr, &err);
  Check(err, "clCreateContext");
  cl_command_queue queue = api.clCreateCommandQueue(context, selected.device, kClQueueProfilingEnable, &err);
  Check(err, "clCreateCommandQueue");
  const char* source = kOpenClSource;
  const std::size_t source_len = std::strlen(kOpenClSource);
  cl_program program = api.clCreateProgramWithSource(context, 1, &source, &source_len, &err);
  Check(err, "clCreateProgramWithSource");
  const auto build_begin = std::chrono::steady_clock::now();
  err = api.clBuildProgram(program, 1, &selected.device, "", nullptr, nullptr);
  const auto build_end = std::chrono::steady_clock::now();
  run.program_build_ms = std::chrono::duration<double, std::milli>(build_end - build_begin).count();
  run.build_log = BuildLog(api, program, selected.device);
  Check(err, "clBuildProgram");
  cl_kernel kernel = api.clCreateKernel(program, "q6k_selected_down_matvec_row", &err);
  Check(err, "clCreateKernel(q6k_selected_down_matvec_row)");

  cl_mem raw_buffer = nullptr, q8_qs_buffer = nullptr, q8_d_buffer = nullptr;
  cl_mem output_buffer = nullptr;
  try {
    raw_buffer = api.clCreateBuffer(context, kClMemReadOnly,
                                    selected_raw.size(), nullptr, &err);
    Check(err, "clCreateBuffer(selected raw)");
    q8_qs_buffer = api.clCreateBuffer(context, kClMemReadOnly,
                                      q8.qs.size() * sizeof(std::int8_t),
                                      nullptr, &err);
    Check(err, "clCreateBuffer(q8 qs)");
    q8_d_buffer = api.clCreateBuffer(context, kClMemReadOnly,
                                     q8.d.size() * sizeof(float), nullptr, &err);
    Check(err, "clCreateBuffer(q8 d)");
    output_buffer = api.clCreateBuffer(context, kClMemWriteOnly,
                                       run.output.size() * sizeof(float),
                                       nullptr, &err);
    Check(err, "clCreateBuffer(output)");
    Check(api.clEnqueueWriteBuffer(queue, raw_buffer, kClTrue, 0,
                                   selected_raw.size(), selected_raw.data(),
                                   0, nullptr, nullptr),
          "clEnqueueWriteBuffer(selected raw)");
    Check(api.clEnqueueWriteBuffer(queue, q8_qs_buffer, kClTrue, 0,
                                   q8.qs.size() * sizeof(std::int8_t),
                                   q8.qs.data(), 0, nullptr, nullptr),
          "clEnqueueWriteBuffer(q8 qs)");
    Check(api.clEnqueueWriteBuffer(queue, q8_d_buffer, kClTrue, 0,
                                   q8.d.size() * sizeof(float), q8.d.data(),
                                   0, nullptr, nullptr),
          "clEnqueueWriteBuffer(q8 d)");
    const cl_uint rows_per_expert_arg = static_cast<cl_uint>(rows_per_expert);
    const cl_uint blocks_per_row_arg = static_cast<cl_uint>(blocks_per_row);
    Check(api.clSetKernelArg(kernel, 0, sizeof(raw_buffer), &raw_buffer), "clSetKernelArg(down 0)");
    Check(api.clSetKernelArg(kernel, 1, sizeof(q8_qs_buffer), &q8_qs_buffer), "clSetKernelArg(down 1)");
    Check(api.clSetKernelArg(kernel, 2, sizeof(q8_d_buffer), &q8_d_buffer), "clSetKernelArg(down 2)");
    Check(api.clSetKernelArg(kernel, 3, sizeof(rows_per_expert_arg), &rows_per_expert_arg), "clSetKernelArg(down 3)");
    Check(api.clSetKernelArg(kernel, 4, sizeof(blocks_per_row_arg), &blocks_per_row_arg), "clSetKernelArg(down 4)");
    Check(api.clSetKernelArg(kernel, 5, sizeof(output_buffer), &output_buffer), "clSetKernelArg(down 5)");

    const std::size_t global = run.output.size();
    std::vector<double> times;
    times.reserve(static_cast<std::size_t>(repeat));
    for (int i = 0; i < repeat; ++i) {
      cl_event event = nullptr;
      Check(api.clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global, nullptr, 0, nullptr, &event),
            "clEnqueueNDRangeKernel(q6k_selected_down_matvec_row)");
      Check(api.clFinish(queue), "clFinish(selected down)");
      times.push_back(EventUs(api, event));
      api.clReleaseEvent(event);
    }
    Check(api.clEnqueueReadBuffer(queue, output_buffer, kClTrue, 0,
                                  run.output.size() * sizeof(float),
                                  run.output.data(), 0, nullptr, nullptr),
          "clEnqueueReadBuffer(selected down output)");
    run.timing.min_us = *std::min_element(times.begin(), times.end());
    run.timing.mean_us =
        std::accumulate(times.begin(), times.end(), 0.0) /
        static_cast<double>(times.size());
    const double raw_bytes = static_cast<double>(selected_raw.size());
    const double io_bytes = raw_bytes +
        static_cast<double>(q8.qs.size() * sizeof(std::int8_t)) +
        static_cast<double>(q8.d.size() * sizeof(float)) +
        static_cast<double>(run.output.size() * sizeof(float));
    run.timing.effective_raw_gb_s = raw_bytes / (run.timing.min_us / 1e6) / 1e9;
    run.timing.effective_io_gb_s = io_bytes / (run.timing.min_us / 1e6) / 1e9;
    run.timing.global_work_items = run.output.size();
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

SelectedSharedQ6DownRun RunGpuSelectedSharedDownQ6Rowstripe(
    const std::vector<std::uint8_t>& selected_raw,
    const std::vector<std::uint8_t>& shared_raw,
    const Q8Planes& selected_q8,
    const Q8Planes& shared_q8,
    std::uint64_t rows_per_expert,
    std::uint64_t blocks_per_row,
    std::uint64_t selected_count,
    const std::string& device_substring,
    int repeat) {
  Require(selected_count == kExpertUsedCount,
          "selected+shared Q6 probe requires 8 selected experts");
  Require(selected_raw.size() ==
              static_cast<std::size_t>(selected_count * rows_per_expert *
                                       blocks_per_row * kQ6KBlockBytes),
          "selected+shared Q6 selected raw byte size mismatch");
  Require(shared_raw.size() ==
              static_cast<std::size_t>(rows_per_expert * blocks_per_row *
                                       kQ6KBlockBytes),
          "selected+shared Q6 shared raw byte size mismatch");
  Require(selected_q8.blocks_per_expert == blocks_per_row,
          "selected+shared Q6 selected q8 block count mismatch");
  Require(shared_q8.blocks_per_expert == blocks_per_row,
          "selected+shared Q6 shared q8 block count mismatch");
  constexpr std::uint64_t kRowsPerTile = 16;
  SelectedSharedQ6DownRun run;
  run.selected_separate_output.assign(
      static_cast<std::size_t>(selected_count * rows_per_expert), 0.0f);
  run.shared_separate_output.assign(static_cast<std::size_t>(rows_per_expert),
                                    0.0f);
  run.selected_combined_output.assign(run.selected_separate_output.size(),
                                      0.0f);
  run.shared_combined_output.assign(run.shared_separate_output.size(), 0.0f);
  OpenClApi api;
  const auto selected = SelectDevice(api, device_substring);
  run.platform_name = selected.platform_name;
  run.device_name = selected.device_name;

  cl_int err = kClSuccess;
  cl_context context =
      api.clCreateContext(nullptr, 1, &selected.device, nullptr, nullptr, &err);
  Check(err, "clCreateContext");
  cl_command_queue queue =
      api.clCreateCommandQueue(context, selected.device, kClQueueProfilingEnable, &err);
  Check(err, "clCreateCommandQueue");
  const char* source = kOpenClSource;
  const std::size_t source_len = std::strlen(kOpenClSource);
  cl_program program =
      api.clCreateProgramWithSource(context, 1, &source, &source_len, &err);
  Check(err, "clCreateProgramWithSource");
  const auto build_begin = std::chrono::steady_clock::now();
  err = api.clBuildProgram(program, 1, &selected.device, "", nullptr, nullptr);
  const auto build_end = std::chrono::steady_clock::now();
  run.program_build_ms =
      std::chrono::duration<double, std::milli>(build_end - build_begin).count();
  run.build_log = BuildLog(api, program, selected.device);
  Check(err, "clBuildProgram");
  cl_kernel selected_kernel =
      api.clCreateKernel(program, "q6k_selected_down_matvec_rowstripe_expert8", &err);
  Check(err, "clCreateKernel(q6k_selected_down_matvec_rowstripe_expert8)");
  cl_kernel shared_kernel =
      api.clCreateKernel(program, "q6k_selected_down_matvec_row", &err);
  Check(err, "clCreateKernel(q6k_selected_down_matvec_row)");
  cl_kernel combined_kernel =
      api.clCreateKernel(
          program,
          "q6k_selected_down_matvec_rowstripe_expert8_plus_shared_raw",
          &err);
  Check(err,
        "clCreateKernel(q6k_selected_down_matvec_rowstripe_expert8_plus_shared_raw)");
  std::array<cl_mem, kExpertUsedCount> rowstripe_buffers{};
  cl_mem shared_raw_buffer = nullptr;
  cl_mem selected_q8_qs_buffer = nullptr;
  cl_mem selected_q8_d_buffer = nullptr;
  cl_mem shared_q8_qs_buffer = nullptr;
  cl_mem shared_q8_d_buffer = nullptr;
  cl_mem selected_separate_output_buffer = nullptr;
  cl_mem shared_separate_output_buffer = nullptr;
  cl_mem selected_combined_output_buffer = nullptr;
  cl_mem shared_combined_output_buffer = nullptr;
  auto release_all = [&]() {
    ReleaseMem(api, &shared_combined_output_buffer);
    ReleaseMem(api, &selected_combined_output_buffer);
    ReleaseMem(api, &shared_separate_output_buffer);
    ReleaseMem(api, &selected_separate_output_buffer);
    ReleaseMem(api, &shared_q8_d_buffer);
    ReleaseMem(api, &shared_q8_qs_buffer);
    ReleaseMem(api, &selected_q8_d_buffer);
    ReleaseMem(api, &selected_q8_qs_buffer);
    ReleaseMem(api, &shared_raw_buffer);
    for (auto& buffer : rowstripe_buffers) {
      ReleaseMem(api, &buffer);
    }
  };
  try {
    const std::size_t per_expert_raw_bytes =
        static_cast<std::size_t>(rows_per_expert * blocks_per_row *
                                 kQ6KBlockBytes);
    for (std::size_t expert = 0; expert < rowstripe_buffers.size(); ++expert) {
      const auto begin = selected_raw.begin() +
          static_cast<std::ptrdiff_t>(expert * per_expert_raw_bytes);
      const auto end = begin + static_cast<std::ptrdiff_t>(per_expert_raw_bytes);
      const std::vector<std::uint8_t> raw_slice(begin, end);
      const auto striped =
          BuildQ6Rowstripe(raw_slice, rows_per_expert, blocks_per_row,
                           kRowsPerTile);
      rowstripe_buffers[expert] =
          api.clCreateBuffer(context, kClMemReadOnly, striped.size(), nullptr, &err);
      Check(err, "clCreateBuffer(selected Q6 rowstripe)");
      Check(api.clEnqueueWriteBuffer(queue, rowstripe_buffers[expert], kClTrue, 0,
                                     striped.size(), striped.data(), 0, nullptr,
                                     nullptr),
            "clEnqueueWriteBuffer(selected Q6 rowstripe)");
    }
    shared_raw_buffer =
        api.clCreateBuffer(context, kClMemReadOnly, shared_raw.size(), nullptr, &err);
    Check(err, "clCreateBuffer(shared Q6 raw)");
    selected_q8_qs_buffer =
        api.clCreateBuffer(context, kClMemReadOnly,
                           selected_q8.qs.size() * sizeof(std::int8_t),
                           nullptr, &err);
    Check(err, "clCreateBuffer(selected Q8 qs)");
    selected_q8_d_buffer =
        api.clCreateBuffer(context, kClMemReadOnly,
                           selected_q8.d.size() * sizeof(float), nullptr, &err);
    Check(err, "clCreateBuffer(selected Q8 d)");
    shared_q8_qs_buffer =
        api.clCreateBuffer(context, kClMemReadOnly,
                           shared_q8.qs.size() * sizeof(std::int8_t),
                           nullptr, &err);
    Check(err, "clCreateBuffer(shared Q8 qs)");
    shared_q8_d_buffer =
        api.clCreateBuffer(context, kClMemReadOnly,
                           shared_q8.d.size() * sizeof(float), nullptr, &err);
    Check(err, "clCreateBuffer(shared Q8 d)");
    selected_separate_output_buffer =
        api.clCreateBuffer(context, kClMemWriteOnly,
                           run.selected_separate_output.size() * sizeof(float),
                           nullptr, &err);
    Check(err, "clCreateBuffer(selected separate output)");
    shared_separate_output_buffer =
        api.clCreateBuffer(context, kClMemWriteOnly,
                           run.shared_separate_output.size() * sizeof(float),
                           nullptr, &err);
    Check(err, "clCreateBuffer(shared separate output)");
    selected_combined_output_buffer =
        api.clCreateBuffer(context, kClMemWriteOnly,
                           run.selected_combined_output.size() * sizeof(float),
                           nullptr, &err);
	    Check(err, "clCreateBuffer(selected combined output)");
		    shared_combined_output_buffer =
		        api.clCreateBuffer(context, kClMemWriteOnly,
		                           run.shared_combined_output.size() * sizeof(float),
		                           nullptr, &err);
		    Check(err, "clCreateBuffer(shared combined output)");
	    Check(api.clEnqueueWriteBuffer(queue, shared_raw_buffer, kClTrue, 0,
                                   shared_raw.size(), shared_raw.data(), 0,
                                   nullptr, nullptr),
          "clEnqueueWriteBuffer(shared Q6 raw)");
    Check(api.clEnqueueWriteBuffer(queue, selected_q8_qs_buffer, kClTrue, 0,
                                   selected_q8.qs.size() * sizeof(std::int8_t),
                                   selected_q8.qs.data(), 0, nullptr, nullptr),
          "clEnqueueWriteBuffer(selected Q8 qs)");
    Check(api.clEnqueueWriteBuffer(queue, selected_q8_d_buffer, kClTrue, 0,
                                   selected_q8.d.size() * sizeof(float),
                                   selected_q8.d.data(), 0, nullptr, nullptr),
          "clEnqueueWriteBuffer(selected Q8 d)");
    Check(api.clEnqueueWriteBuffer(queue, shared_q8_qs_buffer, kClTrue, 0,
                                   shared_q8.qs.size() * sizeof(std::int8_t),
                                   shared_q8.qs.data(), 0, nullptr, nullptr),
          "clEnqueueWriteBuffer(shared Q8 qs)");
    Check(api.clEnqueueWriteBuffer(queue, shared_q8_d_buffer, kClTrue, 0,
                                   shared_q8.d.size() * sizeof(float),
                                   shared_q8.d.data(), 0, nullptr, nullptr),
          "clEnqueueWriteBuffer(shared Q8 d)");

    const cl_uint rows_arg = static_cast<cl_uint>(rows_per_expert);
    const cl_uint blocks_arg = static_cast<cl_uint>(blocks_per_row);
    const cl_uint rows_per_tile_arg = static_cast<cl_uint>(kRowsPerTile);
    for (std::size_t i = 0; i < rowstripe_buffers.size(); ++i) {
      Check(api.clSetKernelArg(selected_kernel, static_cast<cl_uint>(i),
                               sizeof(rowstripe_buffers[i]),
                               &rowstripe_buffers[i]),
            "clSetKernelArg(selected Q6 raw)");
	      Check(api.clSetKernelArg(combined_kernel, static_cast<cl_uint>(i),
		                               sizeof(rowstripe_buffers[i]),
		                               &rowstripe_buffers[i]),
		            "clSetKernelArg(combined Q6 raw)");
		    }
    Check(api.clSetKernelArg(selected_kernel, 8, sizeof(selected_q8_qs_buffer),
                             &selected_q8_qs_buffer),
          "clSetKernelArg(selected Q6 q8_qs)");
    Check(api.clSetKernelArg(selected_kernel, 9, sizeof(selected_q8_d_buffer),
                             &selected_q8_d_buffer),
          "clSetKernelArg(selected Q6 q8_d)");
    Check(api.clSetKernelArg(selected_kernel, 10, sizeof(rows_arg), &rows_arg),
          "clSetKernelArg(selected Q6 rows)");
    Check(api.clSetKernelArg(selected_kernel, 11, sizeof(blocks_arg), &blocks_arg),
          "clSetKernelArg(selected Q6 blocks)");
    Check(api.clSetKernelArg(selected_kernel, 12, sizeof(rows_per_tile_arg),
                             &rows_per_tile_arg),
          "clSetKernelArg(selected Q6 rows_per_tile)");
    Check(api.clSetKernelArg(selected_kernel, 13,
                             sizeof(selected_separate_output_buffer),
                             &selected_separate_output_buffer),
          "clSetKernelArg(selected Q6 output)");

    Check(api.clSetKernelArg(shared_kernel, 0, sizeof(shared_raw_buffer),
                             &shared_raw_buffer),
          "clSetKernelArg(shared Q6 raw)");
    Check(api.clSetKernelArg(shared_kernel, 1, sizeof(shared_q8_qs_buffer),
                             &shared_q8_qs_buffer),
          "clSetKernelArg(shared Q6 q8_qs)");
    Check(api.clSetKernelArg(shared_kernel, 2, sizeof(shared_q8_d_buffer),
                             &shared_q8_d_buffer),
          "clSetKernelArg(shared Q6 q8_d)");
    Check(api.clSetKernelArg(shared_kernel, 3, sizeof(rows_arg), &rows_arg),
          "clSetKernelArg(shared Q6 rows)");
    Check(api.clSetKernelArg(shared_kernel, 4, sizeof(blocks_arg), &blocks_arg),
          "clSetKernelArg(shared Q6 blocks)");
    Check(api.clSetKernelArg(shared_kernel, 5,
                             sizeof(shared_separate_output_buffer),
                             &shared_separate_output_buffer),
          "clSetKernelArg(shared Q6 output)");

    Check(api.clSetKernelArg(combined_kernel, 8, sizeof(shared_raw_buffer),
                             &shared_raw_buffer),
          "clSetKernelArg(combined Q6 shared raw)");
    Check(api.clSetKernelArg(combined_kernel, 9, sizeof(selected_q8_qs_buffer),
                             &selected_q8_qs_buffer),
          "clSetKernelArg(combined Q6 selected q8_qs)");
    Check(api.clSetKernelArg(combined_kernel, 10, sizeof(selected_q8_d_buffer),
                             &selected_q8_d_buffer),
          "clSetKernelArg(combined Q6 selected q8_d)");
    Check(api.clSetKernelArg(combined_kernel, 11, sizeof(shared_q8_qs_buffer),
                             &shared_q8_qs_buffer),
          "clSetKernelArg(combined Q6 shared q8_qs)");
    Check(api.clSetKernelArg(combined_kernel, 12, sizeof(shared_q8_d_buffer),
                             &shared_q8_d_buffer),
          "clSetKernelArg(combined Q6 shared q8_d)");
    Check(api.clSetKernelArg(combined_kernel, 13, sizeof(rows_arg), &rows_arg),
          "clSetKernelArg(combined Q6 rows)");
    Check(api.clSetKernelArg(combined_kernel, 14, sizeof(blocks_arg), &blocks_arg),
          "clSetKernelArg(combined Q6 blocks)");
    Check(api.clSetKernelArg(combined_kernel, 15, sizeof(rows_per_tile_arg),
                             &rows_per_tile_arg),
          "clSetKernelArg(combined Q6 rows_per_tile)");
    Check(api.clSetKernelArg(combined_kernel, 16,
                             sizeof(selected_combined_output_buffer),
                             &selected_combined_output_buffer),
          "clSetKernelArg(combined Q6 selected output)");
		    Check(api.clSetKernelArg(combined_kernel, 17,
		                             sizeof(shared_combined_output_buffer),
		                             &shared_combined_output_buffer),
		          "clSetKernelArg(combined Q6 shared output)");

	    const std::size_t selected_global =
	        static_cast<std::size_t>(selected_count * rows_per_expert);
		    const std::size_t shared_global = static_cast<std::size_t>(rows_per_expert);
		    const std::size_t combined_global = selected_global + shared_global;
		    constexpr std::size_t kLocal = 64;
    const std::size_t* selected_local =
        selected_global % kLocal == 0 ? &kLocal : nullptr;
    const std::size_t* shared_local =
        shared_global % kLocal == 0 ? &kLocal : nullptr;
		    const std::size_t* combined_local =
		        combined_global % kLocal == 0 ? &kLocal : nullptr;
		    std::vector<double> selected_times;
		    std::vector<double> shared_times;
		    std::vector<double> combined_times;
		    selected_times.reserve(static_cast<std::size_t>(repeat));
		    shared_times.reserve(static_cast<std::size_t>(repeat));
		    combined_times.reserve(static_cast<std::size_t>(repeat));
		    for (int i = 0; i < repeat; ++i) {
      cl_event selected_event = nullptr;
      Check(api.clEnqueueNDRangeKernel(queue, selected_kernel, 1, nullptr,
                                       &selected_global, selected_local, 0,
                                       nullptr, &selected_event),
            "clEnqueueNDRangeKernel(selected Q6 rowstripe expert8)");
      Check(api.clFinish(queue), "clFinish(selected Q6 rowstripe expert8)");
      selected_times.push_back(EventUs(api, selected_event));
      api.clReleaseEvent(selected_event);

      cl_event shared_event = nullptr;
      Check(api.clEnqueueNDRangeKernel(queue, shared_kernel, 1, nullptr,
                                       &shared_global, shared_local, 0, nullptr,
                                       &shared_event),
            "clEnqueueNDRangeKernel(shared Q6 raw)");
      Check(api.clFinish(queue), "clFinish(shared Q6 raw)");
      shared_times.push_back(EventUs(api, shared_event));
      api.clReleaseEvent(shared_event);

      cl_event combined_event = nullptr;
	      Check(api.clEnqueueNDRangeKernel(queue, combined_kernel, 1, nullptr,
	                                       &combined_global, combined_local, 0,
	                                       nullptr, &combined_event),
	            "clEnqueueNDRangeKernel(selected+shared Q6 combined)");
		      Check(api.clFinish(queue), "clFinish(selected+shared Q6 combined)");
		      combined_times.push_back(EventUs(api, combined_event));
		      api.clReleaseEvent(combined_event);
		    }
    Check(api.clEnqueueReadBuffer(queue, selected_separate_output_buffer, kClTrue,
                                  0,
                                  run.selected_separate_output.size() * sizeof(float),
                                  run.selected_separate_output.data(), 0,
                                  nullptr, nullptr),
          "clEnqueueReadBuffer(selected separate output)");
    Check(api.clEnqueueReadBuffer(queue, shared_separate_output_buffer, kClTrue,
                                  0,
                                  run.shared_separate_output.size() * sizeof(float),
                                  run.shared_separate_output.data(), 0, nullptr,
                                  nullptr),
          "clEnqueueReadBuffer(shared separate output)");
    Check(api.clEnqueueReadBuffer(queue, selected_combined_output_buffer, kClTrue,
                                  0,
                                  run.selected_combined_output.size() * sizeof(float),
                                  run.selected_combined_output.data(), 0,
                                  nullptr, nullptr),
          "clEnqueueReadBuffer(selected combined output)");
		    Check(api.clEnqueueReadBuffer(queue, shared_combined_output_buffer, kClTrue,
		                                  0,
		                                  run.shared_combined_output.size() * sizeof(float),
		                                  run.shared_combined_output.data(), 0, nullptr,
		                                  nullptr),
		          "clEnqueueReadBuffer(shared combined output)");

    const auto fill_timing = [](const std::vector<double>& times,
                                std::uint64_t global,
                                double raw_bytes,
                                double io_bytes) {
      DownTiming timing;
      timing.min_us = *std::min_element(times.begin(), times.end());
      timing.mean_us =
          std::accumulate(times.begin(), times.end(), 0.0) /
          static_cast<double>(times.size());
      timing.effective_raw_gb_s = raw_bytes / (timing.min_us / 1e6) / 1e9;
      timing.effective_io_gb_s = io_bytes / (timing.min_us / 1e6) / 1e9;
      timing.global_work_items = global;
      timing.kernel_launches = 1;
      return timing;
    };
    const double selected_raw_bytes = static_cast<double>(selected_raw.size());
    const double shared_raw_bytes = static_cast<double>(shared_raw.size());
    const double selected_io_bytes =
        selected_raw_bytes +
        static_cast<double>(selected_q8.qs.size() * sizeof(std::int8_t)) +
        static_cast<double>(selected_q8.d.size() * sizeof(float)) +
        static_cast<double>(run.selected_separate_output.size() * sizeof(float));
    const double shared_io_bytes =
        shared_raw_bytes +
        static_cast<double>(shared_q8.qs.size() * sizeof(std::int8_t)) +
        static_cast<double>(shared_q8.d.size() * sizeof(float)) +
        static_cast<double>(run.shared_separate_output.size() * sizeof(float));
    run.selected_timing =
        fill_timing(selected_times, selected_global, selected_raw_bytes,
                    selected_io_bytes);
    run.shared_timing =
        fill_timing(shared_times, shared_global, shared_raw_bytes,
                    shared_io_bytes);
		    run.combined_timing =
		        fill_timing(combined_times, combined_global,
		                    selected_raw_bytes + shared_raw_bytes,
		                    selected_io_bytes + shared_io_bytes);
		  } catch (...) {
		    release_all();
		    api.clReleaseKernel(combined_kernel);
    api.clReleaseKernel(shared_kernel);
    api.clReleaseKernel(selected_kernel);
    api.clReleaseProgram(program);
    api.clReleaseCommandQueue(queue);
    api.clReleaseContext(context);
    throw;
		  }
		  release_all();
		  api.clReleaseKernel(combined_kernel);
  api.clReleaseKernel(shared_kernel);
  api.clReleaseKernel(selected_kernel);
  api.clReleaseProgram(program);
  api.clReleaseCommandQueue(queue);
  api.clReleaseContext(context);
  return run;
}

DownTailNonAtomicRun RunGpuSelectedSharedDownTailQ6NonAtomic(
    const std::vector<std::uint8_t>& selected_raw,
    const std::vector<std::uint8_t>& shared_raw,
    const Q8Planes& selected_q8,
    const Q8Planes& shared_q8,
    const std::vector<float>& weights,
    const std::vector<float>& shared_tail_gate,
    const std::vector<float>& attn_residual,
    std::uint64_t rows_per_expert,
    std::uint64_t blocks_per_row,
    std::uint64_t selected_count,
    const std::string& device_substring,
    int repeat) {
  Require(selected_count == kExpertUsedCount,
          "non-atomic down-tail Q6 proof requires 8 selected experts");
  Require(weights.size() == selected_count,
          "non-atomic down-tail weight size mismatch");
  Require(shared_tail_gate.size() == 1,
          "non-atomic down-tail shared input gate size mismatch");
  Require(attn_residual.size() == static_cast<std::size_t>(rows_per_expert),
          "non-atomic down-tail residual size mismatch");
  Require(selected_raw.size() ==
              static_cast<std::size_t>(selected_count * rows_per_expert *
                                       blocks_per_row * kQ6KBlockBytes),
          "non-atomic down-tail selected raw byte size mismatch");
  Require(shared_raw.size() ==
              static_cast<std::size_t>(rows_per_expert * blocks_per_row *
                                       kQ6KBlockBytes),
          "non-atomic down-tail shared raw byte size mismatch");
  Require(selected_q8.blocks_per_expert == blocks_per_row,
          "non-atomic down-tail selected q8 block count mismatch");
  Require(shared_q8.blocks_per_expert == blocks_per_row,
          "non-atomic down-tail shared q8 block count mismatch");
  constexpr std::uint64_t kRowsPerTile = 16;
  DownTailNonAtomicRun run;
  const std::size_t contribution_values =
      static_cast<std::size_t>(rows_per_expert * 9);
  run.contributions.assign(contribution_values, 0.0f);
  run.layer_output.assign(static_cast<std::size_t>(rows_per_expert), 0.0f);

  OpenClApi api;
  const auto selected = SelectDevice(api, device_substring);
  run.platform_name = selected.platform_name;
  run.device_name = selected.device_name;

  cl_int err = kClSuccess;
  cl_context context =
      api.clCreateContext(nullptr, 1, &selected.device, nullptr, nullptr, &err);
  Check(err, "clCreateContext");
  cl_command_queue queue =
      api.clCreateCommandQueue(context, selected.device, kClQueueProfilingEnable, &err);
  Check(err, "clCreateCommandQueue");
  const char* source = kOpenClSource;
  const std::size_t source_len = std::strlen(kOpenClSource);
  cl_program program =
      api.clCreateProgramWithSource(context, 1, &source, &source_len, &err);
  Check(err, "clCreateProgramWithSource");
  const auto build_begin = std::chrono::steady_clock::now();
  err = api.clBuildProgram(program, 1, &selected.device, "", nullptr, nullptr);
  const auto build_end = std::chrono::steady_clock::now();
  run.program_build_ms =
      std::chrono::duration<double, std::milli>(build_end - build_begin).count();
  run.build_log = BuildLog(api, program, selected.device);
  Check(err, "clBuildProgram");
  cl_kernel contrib_kernel = api.clCreateKernel(
      program,
      "q6k_selected_down_matvec_rowstripe_expert8_plus_shared_tail_contrib_raw",
      &err);
  Check(err, "clCreateKernel(non-atomic down-tail contrib)");
  cl_kernel reduce_kernel =
      api.clCreateKernel(program, "ffn_tail_reduce9_contrib_f32", &err);
  Check(err, "clCreateKernel(non-atomic down-tail reduce)");

  std::array<cl_mem, kExpertUsedCount> rowstripe_buffers{};
  cl_mem shared_raw_buffer = nullptr;
  cl_mem selected_q8_qs_buffer = nullptr;
  cl_mem selected_q8_d_buffer = nullptr;
  cl_mem shared_q8_qs_buffer = nullptr;
  cl_mem shared_q8_d_buffer = nullptr;
  cl_mem weights_buffer = nullptr;
  cl_mem shared_gate_buffer = nullptr;
  cl_mem attn_residual_buffer = nullptr;
  cl_mem contribution_buffer = nullptr;
  cl_mem layer_output_buffer = nullptr;
  auto release_all = [&]() {
    ReleaseMem(api, &layer_output_buffer);
    ReleaseMem(api, &contribution_buffer);
    ReleaseMem(api, &attn_residual_buffer);
    ReleaseMem(api, &shared_gate_buffer);
    ReleaseMem(api, &weights_buffer);
    ReleaseMem(api, &shared_q8_d_buffer);
    ReleaseMem(api, &shared_q8_qs_buffer);
    ReleaseMem(api, &selected_q8_d_buffer);
    ReleaseMem(api, &selected_q8_qs_buffer);
    ReleaseMem(api, &shared_raw_buffer);
    for (auto& buffer : rowstripe_buffers) {
      ReleaseMem(api, &buffer);
    }
  };
  try {
    const std::size_t per_expert_raw_bytes =
        static_cast<std::size_t>(rows_per_expert * blocks_per_row *
                                 kQ6KBlockBytes);
    for (std::size_t expert = 0; expert < rowstripe_buffers.size(); ++expert) {
      const auto begin = selected_raw.begin() +
          static_cast<std::ptrdiff_t>(expert * per_expert_raw_bytes);
      const auto end = begin + static_cast<std::ptrdiff_t>(per_expert_raw_bytes);
      const std::vector<std::uint8_t> raw_slice(begin, end);
      const auto striped =
          BuildQ6Rowstripe(raw_slice, rows_per_expert, blocks_per_row,
                           kRowsPerTile);
      rowstripe_buffers[expert] =
          api.clCreateBuffer(context, kClMemReadOnly, striped.size(), nullptr, &err);
      Check(err, "clCreateBuffer(non-atomic selected Q6 rowstripe)");
      Check(api.clEnqueueWriteBuffer(queue, rowstripe_buffers[expert], kClTrue,
                                     0, striped.size(), striped.data(), 0,
                                     nullptr, nullptr),
            "clEnqueueWriteBuffer(non-atomic selected Q6 rowstripe)");
    }
    shared_raw_buffer =
        api.clCreateBuffer(context, kClMemReadOnly, shared_raw.size(), nullptr, &err);
    Check(err, "clCreateBuffer(non-atomic shared Q6 raw)");
    selected_q8_qs_buffer = api.clCreateBuffer(
        context, kClMemReadOnly, selected_q8.qs.size() * sizeof(std::int8_t),
        nullptr, &err);
    Check(err, "clCreateBuffer(non-atomic selected q8 qs)");
    selected_q8_d_buffer = api.clCreateBuffer(
        context, kClMemReadOnly, selected_q8.d.size() * sizeof(float),
        nullptr, &err);
    Check(err, "clCreateBuffer(non-atomic selected q8 d)");
    shared_q8_qs_buffer = api.clCreateBuffer(
        context, kClMemReadOnly, shared_q8.qs.size() * sizeof(std::int8_t),
        nullptr, &err);
    Check(err, "clCreateBuffer(non-atomic shared q8 qs)");
    shared_q8_d_buffer = api.clCreateBuffer(
        context, kClMemReadOnly, shared_q8.d.size() * sizeof(float), nullptr,
        &err);
    Check(err, "clCreateBuffer(non-atomic shared q8 d)");
    weights_buffer = api.clCreateBuffer(
        context, kClMemReadOnly, weights.size() * sizeof(float), nullptr, &err);
    Check(err, "clCreateBuffer(non-atomic weights)");
    shared_gate_buffer = api.clCreateBuffer(
        context, kClMemReadOnly, sizeof(float), nullptr, &err);
    Check(err, "clCreateBuffer(non-atomic shared gate)");
    attn_residual_buffer = api.clCreateBuffer(
        context, kClMemReadOnly, attn_residual.size() * sizeof(float), nullptr,
        &err);
    Check(err, "clCreateBuffer(non-atomic residual)");
    contribution_buffer = api.clCreateBuffer(
        context, kClMemReadWrite, run.contributions.size() * sizeof(float),
        nullptr, &err);
    Check(err, "clCreateBuffer(non-atomic contributions)");
    layer_output_buffer = api.clCreateBuffer(
        context, kClMemWriteOnly, run.layer_output.size() * sizeof(float),
        nullptr, &err);
    Check(err, "clCreateBuffer(non-atomic layer output)");

    Check(api.clEnqueueWriteBuffer(queue, shared_raw_buffer, kClTrue, 0,
                                   shared_raw.size(), shared_raw.data(), 0,
                                   nullptr, nullptr),
          "clEnqueueWriteBuffer(non-atomic shared raw)");
    Check(api.clEnqueueWriteBuffer(queue, selected_q8_qs_buffer, kClTrue, 0,
                                   selected_q8.qs.size() * sizeof(std::int8_t),
                                   selected_q8.qs.data(), 0, nullptr, nullptr),
          "clEnqueueWriteBuffer(non-atomic selected q8 qs)");
    Check(api.clEnqueueWriteBuffer(queue, selected_q8_d_buffer, kClTrue, 0,
                                   selected_q8.d.size() * sizeof(float),
                                   selected_q8.d.data(), 0, nullptr, nullptr),
          "clEnqueueWriteBuffer(non-atomic selected q8 d)");
    Check(api.clEnqueueWriteBuffer(queue, shared_q8_qs_buffer, kClTrue, 0,
                                   shared_q8.qs.size() * sizeof(std::int8_t),
                                   shared_q8.qs.data(), 0, nullptr, nullptr),
          "clEnqueueWriteBuffer(non-atomic shared q8 qs)");
    Check(api.clEnqueueWriteBuffer(queue, shared_q8_d_buffer, kClTrue, 0,
                                   shared_q8.d.size() * sizeof(float),
                                   shared_q8.d.data(), 0, nullptr, nullptr),
          "clEnqueueWriteBuffer(non-atomic shared q8 d)");
    Check(api.clEnqueueWriteBuffer(queue, weights_buffer, kClTrue, 0,
                                   weights.size() * sizeof(float),
                                   weights.data(), 0, nullptr, nullptr),
          "clEnqueueWriteBuffer(non-atomic weights)");
    Check(api.clEnqueueWriteBuffer(queue, shared_gate_buffer, kClTrue, 0,
                                   sizeof(float), shared_tail_gate.data(), 0,
                                   nullptr, nullptr),
          "clEnqueueWriteBuffer(non-atomic shared gate)");
    Check(api.clEnqueueWriteBuffer(queue, attn_residual_buffer, kClTrue, 0,
                                   attn_residual.size() * sizeof(float),
                                   attn_residual.data(), 0, nullptr, nullptr),
          "clEnqueueWriteBuffer(non-atomic residual)");

    const cl_uint rows_arg = static_cast<cl_uint>(rows_per_expert);
    const cl_uint blocks_arg = static_cast<cl_uint>(blocks_per_row);
    const cl_uint rows_per_tile_arg = static_cast<cl_uint>(kRowsPerTile);
    for (std::size_t i = 0; i < rowstripe_buffers.size(); ++i) {
      Check(api.clSetKernelArg(contrib_kernel, static_cast<cl_uint>(i),
                               sizeof(rowstripe_buffers[i]),
                               &rowstripe_buffers[i]),
            "clSetKernelArg(non-atomic contrib selected raw)");
    }
    Check(api.clSetKernelArg(contrib_kernel, 8, sizeof(shared_raw_buffer),
                             &shared_raw_buffer),
          "clSetKernelArg(non-atomic contrib shared raw)");
    Check(api.clSetKernelArg(contrib_kernel, 9, sizeof(selected_q8_qs_buffer),
                             &selected_q8_qs_buffer),
          "clSetKernelArg(non-atomic contrib selected q8 qs)");
    Check(api.clSetKernelArg(contrib_kernel, 10, sizeof(selected_q8_d_buffer),
                             &selected_q8_d_buffer),
          "clSetKernelArg(non-atomic contrib selected q8 d)");
    Check(api.clSetKernelArg(contrib_kernel, 11, sizeof(shared_q8_qs_buffer),
                             &shared_q8_qs_buffer),
          "clSetKernelArg(non-atomic contrib shared q8 qs)");
    Check(api.clSetKernelArg(contrib_kernel, 12, sizeof(shared_q8_d_buffer),
                             &shared_q8_d_buffer),
          "clSetKernelArg(non-atomic contrib shared q8 d)");
    Check(api.clSetKernelArg(contrib_kernel, 13, sizeof(weights_buffer),
                             &weights_buffer),
          "clSetKernelArg(non-atomic contrib weights)");
    Check(api.clSetKernelArg(contrib_kernel, 14, sizeof(shared_gate_buffer),
                             &shared_gate_buffer),
          "clSetKernelArg(non-atomic contrib shared gate)");
    Check(api.clSetKernelArg(contrib_kernel, 15, sizeof(rows_arg), &rows_arg),
          "clSetKernelArg(non-atomic contrib rows)");
    Check(api.clSetKernelArg(contrib_kernel, 16, sizeof(blocks_arg),
                             &blocks_arg),
          "clSetKernelArg(non-atomic contrib blocks)");
    Check(api.clSetKernelArg(contrib_kernel, 17, sizeof(rows_per_tile_arg),
                             &rows_per_tile_arg),
          "clSetKernelArg(non-atomic contrib rows_per_tile)");
    Check(api.clSetKernelArg(contrib_kernel, 18, sizeof(contribution_buffer),
                             &contribution_buffer),
          "clSetKernelArg(non-atomic contrib out)");

    Check(api.clSetKernelArg(reduce_kernel, 0, sizeof(contribution_buffer),
                             &contribution_buffer),
          "clSetKernelArg(non-atomic reduce contrib)");
    Check(api.clSetKernelArg(reduce_kernel, 1, sizeof(attn_residual_buffer),
                             &attn_residual_buffer),
          "clSetKernelArg(non-atomic reduce residual)");
    Check(api.clSetKernelArg(reduce_kernel, 2, sizeof(rows_arg), &rows_arg),
          "clSetKernelArg(non-atomic reduce hidden)");
    Check(api.clSetKernelArg(reduce_kernel, 3, sizeof(layer_output_buffer),
                             &layer_output_buffer),
          "clSetKernelArg(non-atomic reduce output)");

    const std::size_t contribution_global =
        static_cast<std::size_t>(rows_per_expert * 9);
    const std::size_t reduce_global = static_cast<std::size_t>(rows_per_expert);
    constexpr std::size_t kLocal = 64;
    const std::size_t* contribution_local =
        contribution_global % kLocal == 0 ? &kLocal : nullptr;
    const std::size_t* reduce_local =
        reduce_global % kLocal == 0 ? &kLocal : nullptr;
    std::vector<double> contribution_times;
    std::vector<double> reduce_times;
    std::vector<double> shell_times;
    contribution_times.reserve(static_cast<std::size_t>(repeat));
    reduce_times.reserve(static_cast<std::size_t>(repeat));
    shell_times.reserve(static_cast<std::size_t>(repeat));
    for (int i = 0; i < repeat; ++i) {
      cl_event contribution_event = nullptr;
      cl_event reduce_event = nullptr;
      Check(api.clEnqueueNDRangeKernel(queue, contrib_kernel, 1, nullptr,
                                       &contribution_global,
                                       contribution_local, 0, nullptr,
                                       &contribution_event),
            "clEnqueueNDRangeKernel(non-atomic down-tail contribution)");
      Check(api.clEnqueueNDRangeKernel(queue, reduce_kernel, 1, nullptr,
                                       &reduce_global, reduce_local, 0,
                                       nullptr, &reduce_event),
            "clEnqueueNDRangeKernel(non-atomic down-tail reduce)");
      Check(api.clFinish(queue), "clFinish(non-atomic down-tail)");
      const double contribution_us = EventUs(api, contribution_event);
      const double reduce_us = EventUs(api, reduce_event);
      contribution_times.push_back(contribution_us);
      reduce_times.push_back(reduce_us);
      shell_times.push_back(contribution_us + reduce_us);
      api.clReleaseEvent(contribution_event);
      api.clReleaseEvent(reduce_event);
    }

    Check(api.clEnqueueReadBuffer(queue, contribution_buffer, kClTrue, 0,
                                  run.contributions.size() * sizeof(float),
                                  run.contributions.data(), 0, nullptr,
                                  nullptr),
          "clEnqueueReadBuffer(non-atomic contributions)");
    Check(api.clEnqueueReadBuffer(queue, layer_output_buffer, kClTrue, 0,
                                  run.layer_output.size() * sizeof(float),
                                  run.layer_output.data(), 0, nullptr,
                                  nullptr),
          "clEnqueueReadBuffer(non-atomic layer output)");

    const auto fill_timing = [](const std::vector<double>& times,
                                std::uint64_t global,
                                double raw_bytes,
                                double io_bytes) {
      DownTiming timing;
      timing.min_us = *std::min_element(times.begin(), times.end());
      timing.mean_us =
          std::accumulate(times.begin(), times.end(), 0.0) /
          static_cast<double>(times.size());
      timing.effective_raw_gb_s = raw_bytes / (timing.min_us / 1e6) / 1e9;
      timing.effective_io_gb_s = io_bytes / (timing.min_us / 1e6) / 1e9;
      timing.global_work_items = global;
      timing.kernel_launches = 1;
      return timing;
    };
    const double selected_raw_bytes = static_cast<double>(selected_raw.size());
    const double shared_raw_bytes = static_cast<double>(shared_raw.size());
    const double contribution_io_bytes =
        selected_raw_bytes + shared_raw_bytes +
        static_cast<double>((selected_q8.qs.size() + shared_q8.qs.size()) *
                            sizeof(std::int8_t)) +
        static_cast<double>((selected_q8.d.size() + shared_q8.d.size()) *
                            sizeof(float)) +
        static_cast<double>(weights.size() * sizeof(float)) +
        static_cast<double>(sizeof(float)) +
        static_cast<double>(run.contributions.size() * sizeof(float));
    const double reduce_io_bytes =
        static_cast<double>((run.contributions.size() + attn_residual.size() +
                            run.layer_output.size()) * sizeof(float));
    run.contribution_timing =
        fill_timing(contribution_times, contribution_global,
                    selected_raw_bytes + shared_raw_bytes,
                    contribution_io_bytes);
    run.reduce_timing =
        fill_timing(reduce_times, reduce_global,
                    static_cast<double>(run.contributions.size() * sizeof(float)),
                    reduce_io_bytes);
    run.shell_sum_min_us =
        *std::min_element(shell_times.begin(), shell_times.end());
    run.shell_sum_mean_us =
        std::accumulate(shell_times.begin(), shell_times.end(), 0.0) /
        static_cast<double>(shell_times.size());
  } catch (...) {
    release_all();
    api.clReleaseKernel(reduce_kernel);
    api.clReleaseKernel(contrib_kernel);
    api.clReleaseProgram(program);
    api.clReleaseCommandQueue(queue);
    api.clReleaseContext(context);
    throw;
  }
  release_all();
  api.clReleaseKernel(reduce_kernel);
  api.clReleaseKernel(contrib_kernel);
  api.clReleaseProgram(program);
  api.clReleaseCommandQueue(queue);
  api.clReleaseContext(context);
  return run;
}

DownTailRowgroupRun RunGpuSelectedSharedDownTailQ6RowgroupReduce(
    const std::vector<std::uint8_t>& selected_raw,
    const std::vector<std::uint8_t>& shared_raw,
    const Q8Planes& selected_q8,
    const Q8Planes& shared_q8,
    const std::vector<float>& weights,
    const std::vector<float>& shared_tail_gate,
    const std::vector<float>& attn_residual,
    std::uint64_t rows_per_expert,
    std::uint64_t blocks_per_row,
    std::uint64_t selected_count,
    const std::string& device_substring,
    int repeat) {
  Require(selected_count == kExpertUsedCount,
          "rowgroup down-tail Q6 proof requires 8 selected experts");
  Require(weights.size() == selected_count,
          "rowgroup down-tail weight size mismatch");
  Require(shared_tail_gate.size() == 1,
          "rowgroup down-tail shared input gate size mismatch");
  Require(attn_residual.size() == static_cast<std::size_t>(rows_per_expert),
          "rowgroup down-tail residual size mismatch");
  Require(selected_raw.size() ==
              static_cast<std::size_t>(selected_count * rows_per_expert *
                                       blocks_per_row * kQ6KBlockBytes),
          "rowgroup down-tail selected raw byte size mismatch");
  Require(shared_raw.size() ==
              static_cast<std::size_t>(rows_per_expert * blocks_per_row *
                                       kQ6KBlockBytes),
          "rowgroup down-tail shared raw byte size mismatch");
  Require(selected_q8.blocks_per_expert == blocks_per_row,
          "rowgroup down-tail selected q8 block count mismatch");
  Require(shared_q8.blocks_per_expert == blocks_per_row,
          "rowgroup down-tail shared q8 block count mismatch");
  constexpr std::uint64_t kRowsPerTile = 16;
  DownTailRowgroupRun run;
  run.layer_output.assign(static_cast<std::size_t>(rows_per_expert), 0.0f);

  OpenClApi api;
  const auto selected = SelectDevice(api, device_substring);
  run.platform_name = selected.platform_name;
  run.device_name = selected.device_name;

  cl_int err = kClSuccess;
  cl_context context =
      api.clCreateContext(nullptr, 1, &selected.device, nullptr, nullptr, &err);
  Check(err, "clCreateContext");
  cl_command_queue queue =
      api.clCreateCommandQueue(context, selected.device, kClQueueProfilingEnable, &err);
  Check(err, "clCreateCommandQueue");
  const char* source = kOpenClSource;
  const std::size_t source_len = std::strlen(kOpenClSource);
  cl_program program =
      api.clCreateProgramWithSource(context, 1, &source, &source_len, &err);
  Check(err, "clCreateProgramWithSource");
  const auto build_begin = std::chrono::steady_clock::now();
  err = api.clBuildProgram(program, 1, &selected.device, "", nullptr, nullptr);
  const auto build_end = std::chrono::steady_clock::now();
  run.program_build_ms =
      std::chrono::duration<double, std::milli>(build_end - build_begin).count();
  run.build_log = BuildLog(api, program, selected.device);
  Check(err, "clBuildProgram");
  cl_kernel rowgroup_kernel = api.clCreateKernel(
      program,
      "q6k_selected_down_matvec_rowstripe_expert8_plus_shared_tail_rowgroup_reduce_raw",
      &err);
  Check(err, "clCreateKernel(rowgroup down-tail reduce)");

  std::array<cl_mem, kExpertUsedCount> rowstripe_buffers{};
  cl_mem shared_raw_buffer = nullptr;
  cl_mem selected_q8_qs_buffer = nullptr;
  cl_mem selected_q8_d_buffer = nullptr;
  cl_mem shared_q8_qs_buffer = nullptr;
  cl_mem shared_q8_d_buffer = nullptr;
  cl_mem weights_buffer = nullptr;
  cl_mem shared_gate_buffer = nullptr;
  cl_mem attn_residual_buffer = nullptr;
  cl_mem layer_output_buffer = nullptr;
  auto release_all = [&]() {
    ReleaseMem(api, &layer_output_buffer);
    ReleaseMem(api, &attn_residual_buffer);
    ReleaseMem(api, &shared_gate_buffer);
    ReleaseMem(api, &weights_buffer);
    ReleaseMem(api, &shared_q8_d_buffer);
    ReleaseMem(api, &shared_q8_qs_buffer);
    ReleaseMem(api, &selected_q8_d_buffer);
    ReleaseMem(api, &selected_q8_qs_buffer);
    ReleaseMem(api, &shared_raw_buffer);
    for (auto& buffer : rowstripe_buffers) {
      ReleaseMem(api, &buffer);
    }
  };
  try {
    const std::size_t per_expert_raw_bytes =
        static_cast<std::size_t>(rows_per_expert * blocks_per_row *
                                 kQ6KBlockBytes);
    for (std::size_t expert = 0; expert < rowstripe_buffers.size(); ++expert) {
      const auto begin = selected_raw.begin() +
          static_cast<std::ptrdiff_t>(expert * per_expert_raw_bytes);
      const auto end = begin + static_cast<std::ptrdiff_t>(per_expert_raw_bytes);
      const std::vector<std::uint8_t> raw_slice(begin, end);
      const auto striped =
          BuildQ6Rowstripe(raw_slice, rows_per_expert, blocks_per_row,
                           kRowsPerTile);
      rowstripe_buffers[expert] =
          api.clCreateBuffer(context, kClMemReadOnly, striped.size(), nullptr, &err);
      Check(err, "clCreateBuffer(rowgroup selected Q6 rowstripe)");
      Check(api.clEnqueueWriteBuffer(queue, rowstripe_buffers[expert], kClTrue,
                                     0, striped.size(), striped.data(), 0,
                                     nullptr, nullptr),
            "clEnqueueWriteBuffer(rowgroup selected Q6 rowstripe)");
    }
    shared_raw_buffer =
        api.clCreateBuffer(context, kClMemReadOnly, shared_raw.size(), nullptr, &err);
    Check(err, "clCreateBuffer(rowgroup shared Q6 raw)");
    selected_q8_qs_buffer = api.clCreateBuffer(
        context, kClMemReadOnly, selected_q8.qs.size() * sizeof(std::int8_t),
        nullptr, &err);
    Check(err, "clCreateBuffer(rowgroup selected q8 qs)");
    selected_q8_d_buffer = api.clCreateBuffer(
        context, kClMemReadOnly, selected_q8.d.size() * sizeof(float),
        nullptr, &err);
    Check(err, "clCreateBuffer(rowgroup selected q8 d)");
    shared_q8_qs_buffer = api.clCreateBuffer(
        context, kClMemReadOnly, shared_q8.qs.size() * sizeof(std::int8_t),
        nullptr, &err);
    Check(err, "clCreateBuffer(rowgroup shared q8 qs)");
    shared_q8_d_buffer = api.clCreateBuffer(
        context, kClMemReadOnly, shared_q8.d.size() * sizeof(float), nullptr,
        &err);
    Check(err, "clCreateBuffer(rowgroup shared q8 d)");
    weights_buffer = api.clCreateBuffer(
        context, kClMemReadOnly, weights.size() * sizeof(float), nullptr, &err);
    Check(err, "clCreateBuffer(rowgroup weights)");
    shared_gate_buffer = api.clCreateBuffer(
        context, kClMemReadOnly, sizeof(float), nullptr, &err);
    Check(err, "clCreateBuffer(rowgroup shared gate)");
    attn_residual_buffer = api.clCreateBuffer(
        context, kClMemReadOnly, attn_residual.size() * sizeof(float), nullptr,
        &err);
    Check(err, "clCreateBuffer(rowgroup residual)");
    layer_output_buffer = api.clCreateBuffer(
        context, kClMemWriteOnly, run.layer_output.size() * sizeof(float),
        nullptr, &err);
    Check(err, "clCreateBuffer(rowgroup layer output)");

    Check(api.clEnqueueWriteBuffer(queue, shared_raw_buffer, kClTrue, 0,
                                   shared_raw.size(), shared_raw.data(), 0,
                                   nullptr, nullptr),
          "clEnqueueWriteBuffer(rowgroup shared raw)");
    Check(api.clEnqueueWriteBuffer(queue, selected_q8_qs_buffer, kClTrue, 0,
                                   selected_q8.qs.size() * sizeof(std::int8_t),
                                   selected_q8.qs.data(), 0, nullptr, nullptr),
          "clEnqueueWriteBuffer(rowgroup selected q8 qs)");
    Check(api.clEnqueueWriteBuffer(queue, selected_q8_d_buffer, kClTrue, 0,
                                   selected_q8.d.size() * sizeof(float),
                                   selected_q8.d.data(), 0, nullptr, nullptr),
          "clEnqueueWriteBuffer(rowgroup selected q8 d)");
    Check(api.clEnqueueWriteBuffer(queue, shared_q8_qs_buffer, kClTrue, 0,
                                   shared_q8.qs.size() * sizeof(std::int8_t),
                                   shared_q8.qs.data(), 0, nullptr, nullptr),
          "clEnqueueWriteBuffer(rowgroup shared q8 qs)");
    Check(api.clEnqueueWriteBuffer(queue, shared_q8_d_buffer, kClTrue, 0,
                                   shared_q8.d.size() * sizeof(float),
                                   shared_q8.d.data(), 0, nullptr, nullptr),
          "clEnqueueWriteBuffer(rowgroup shared q8 d)");
    Check(api.clEnqueueWriteBuffer(queue, weights_buffer, kClTrue, 0,
                                   weights.size() * sizeof(float),
                                   weights.data(), 0, nullptr, nullptr),
          "clEnqueueWriteBuffer(rowgroup weights)");
    Check(api.clEnqueueWriteBuffer(queue, shared_gate_buffer, kClTrue, 0,
                                   sizeof(float), shared_tail_gate.data(), 0,
                                   nullptr, nullptr),
          "clEnqueueWriteBuffer(rowgroup shared gate)");
    Check(api.clEnqueueWriteBuffer(queue, attn_residual_buffer, kClTrue, 0,
                                   attn_residual.size() * sizeof(float),
                                   attn_residual.data(), 0, nullptr, nullptr),
          "clEnqueueWriteBuffer(rowgroup residual)");

    const cl_uint rows_arg = static_cast<cl_uint>(rows_per_expert);
    const cl_uint blocks_arg = static_cast<cl_uint>(blocks_per_row);
    const cl_uint rows_per_tile_arg = static_cast<cl_uint>(kRowsPerTile);
    for (std::size_t i = 0; i < rowstripe_buffers.size(); ++i) {
      Check(api.clSetKernelArg(rowgroup_kernel, static_cast<cl_uint>(i),
                               sizeof(rowstripe_buffers[i]),
                               &rowstripe_buffers[i]),
            "clSetKernelArg(rowgroup selected raw)");
    }
    Check(api.clSetKernelArg(rowgroup_kernel, 8, sizeof(shared_raw_buffer),
                             &shared_raw_buffer),
          "clSetKernelArg(rowgroup shared raw)");
    Check(api.clSetKernelArg(rowgroup_kernel, 9, sizeof(selected_q8_qs_buffer),
                             &selected_q8_qs_buffer),
          "clSetKernelArg(rowgroup selected q8 qs)");
    Check(api.clSetKernelArg(rowgroup_kernel, 10, sizeof(selected_q8_d_buffer),
                             &selected_q8_d_buffer),
          "clSetKernelArg(rowgroup selected q8 d)");
    Check(api.clSetKernelArg(rowgroup_kernel, 11, sizeof(shared_q8_qs_buffer),
                             &shared_q8_qs_buffer),
          "clSetKernelArg(rowgroup shared q8 qs)");
    Check(api.clSetKernelArg(rowgroup_kernel, 12, sizeof(shared_q8_d_buffer),
                             &shared_q8_d_buffer),
          "clSetKernelArg(rowgroup shared q8 d)");
    Check(api.clSetKernelArg(rowgroup_kernel, 13, sizeof(weights_buffer),
                             &weights_buffer),
          "clSetKernelArg(rowgroup weights)");
    Check(api.clSetKernelArg(rowgroup_kernel, 14, sizeof(shared_gate_buffer),
                             &shared_gate_buffer),
          "clSetKernelArg(rowgroup shared gate)");
    Check(api.clSetKernelArg(rowgroup_kernel, 15, sizeof(attn_residual_buffer),
                             &attn_residual_buffer),
          "clSetKernelArg(rowgroup residual)");
    Check(api.clSetKernelArg(rowgroup_kernel, 16, sizeof(rows_arg), &rows_arg),
          "clSetKernelArg(rowgroup rows)");
    Check(api.clSetKernelArg(rowgroup_kernel, 17, sizeof(blocks_arg),
                             &blocks_arg),
          "clSetKernelArg(rowgroup blocks)");
    Check(api.clSetKernelArg(rowgroup_kernel, 18, sizeof(rows_per_tile_arg),
                             &rows_per_tile_arg),
          "clSetKernelArg(rowgroup rows_per_tile)");
    Check(api.clSetKernelArg(rowgroup_kernel, 19, sizeof(layer_output_buffer),
                             &layer_output_buffer),
          "clSetKernelArg(rowgroup output)");

    const std::size_t rowgroup_global =
        static_cast<std::size_t>(rows_per_expert * kQ6RowgroupDownTailLocal);
    const std::size_t rowgroup_local =
        static_cast<std::size_t>(kQ6RowgroupDownTailLocal);
    std::vector<double> rowgroup_times;
    rowgroup_times.reserve(static_cast<std::size_t>(repeat));
    for (int i = 0; i < repeat; ++i) {
      cl_event rowgroup_event = nullptr;
      Check(api.clEnqueueNDRangeKernel(queue, rowgroup_kernel, 1, nullptr,
                                       &rowgroup_global, &rowgroup_local, 0,
                                       nullptr, &rowgroup_event),
            "clEnqueueNDRangeKernel(rowgroup down-tail)");
      Check(api.clFinish(queue), "clFinish(rowgroup down-tail)");
      rowgroup_times.push_back(EventUs(api, rowgroup_event));
      api.clReleaseEvent(rowgroup_event);
    }

    Check(api.clEnqueueReadBuffer(queue, layer_output_buffer, kClTrue, 0,
                                  run.layer_output.size() * sizeof(float),
                                  run.layer_output.data(), 0, nullptr,
                                  nullptr),
          "clEnqueueReadBuffer(rowgroup layer output)");

    const double selected_raw_bytes = static_cast<double>(selected_raw.size());
    const double shared_raw_bytes = static_cast<double>(shared_raw.size());
    const double rowgroup_io_bytes =
        selected_raw_bytes + shared_raw_bytes +
        static_cast<double>((selected_q8.qs.size() + shared_q8.qs.size()) *
                            sizeof(std::int8_t)) +
        static_cast<double>((selected_q8.d.size() + shared_q8.d.size()) *
                            sizeof(float)) +
        static_cast<double>(weights.size() * sizeof(float)) +
        static_cast<double>(sizeof(float)) +
        static_cast<double>((attn_residual.size() + run.layer_output.size()) *
                            sizeof(float));
    run.timing.min_us = *std::min_element(rowgroup_times.begin(),
                                          rowgroup_times.end());
    run.timing.mean_us =
        std::accumulate(rowgroup_times.begin(), rowgroup_times.end(), 0.0) /
        static_cast<double>(rowgroup_times.size());
    run.timing.effective_raw_gb_s =
        (selected_raw_bytes + shared_raw_bytes) /
        (run.timing.min_us / 1e6) / 1e9;
    run.timing.effective_io_gb_s =
        rowgroup_io_bytes / (run.timing.min_us / 1e6) / 1e9;
    run.timing.global_work_items = rowgroup_global;
    run.timing.kernel_launches = 1;
  } catch (...) {
    release_all();
    api.clReleaseKernel(rowgroup_kernel);
    api.clReleaseProgram(program);
    api.clReleaseCommandQueue(queue);
    api.clReleaseContext(context);
    throw;
  }
  release_all();
  api.clReleaseKernel(rowgroup_kernel);
  api.clReleaseProgram(program);
  api.clReleaseCommandQueue(queue);
  api.clReleaseContext(context);
  return run;
}

SelectedSharedQ6DownRun RunGpuSelectedSharedDownQ4Q4(
    const std::vector<std::uint8_t>& selected_raw,
    const std::vector<std::uint8_t>& shared_raw,
    const Q8Planes& selected_q8,
    const Q8Planes& shared_q8,
    const DownRun& selected_separate,
    std::uint64_t rows_per_expert,
    std::uint64_t blocks_per_row,
    std::uint64_t selected_count,
    const std::string& device_substring,
    int repeat) {
  Require(selected_count == kExpertUsedCount,
          "selected+shared Q4/Q4 probe requires 8 selected experts");
  Require(selected_raw.size() ==
              static_cast<std::size_t>(selected_count * rows_per_expert *
                                       blocks_per_row * kQ4KBlockBytes),
          "selected+shared Q4/Q4 selected raw byte size mismatch");
  Require(shared_raw.size() ==
              static_cast<std::size_t>(rows_per_expert * blocks_per_row *
                                       kQ4KBlockBytes),
          "selected+shared Q4/Q4 shared raw byte size mismatch");
  Require(selected_q8.blocks_per_expert == blocks_per_row,
          "selected+shared Q4/Q4 selected q8 block count mismatch");
  Require(shared_q8.blocks_per_expert == blocks_per_row,
          "selected+shared Q4/Q4 shared q8 block count mismatch");

  SelectedSharedQ6DownRun run;
  run.selected_separate_output = selected_separate.output;
  run.selected_timing = selected_separate.timing;
  run.shared_separate_output.assign(static_cast<std::size_t>(rows_per_expert),
                                    0.0f);
  run.selected_combined_output.assign(run.selected_separate_output.size(),
                                      0.0f);
  run.shared_combined_output.assign(run.shared_separate_output.size(), 0.0f);

  std::vector<std::vector<std::uint8_t>> packed_by_expert;
  packed_by_expert.reserve(static_cast<std::size_t>(selected_count));
  const std::size_t selected_raw_bytes_per_expert =
      static_cast<std::size_t>(rows_per_expert * blocks_per_row *
                               kQ4KBlockBytes);
  for (std::uint64_t expert = 0; expert < selected_count; ++expert) {
    const auto begin = selected_raw.begin() +
        static_cast<std::ptrdiff_t>(expert * selected_raw_bytes_per_expert);
    const auto end = begin +
        static_cast<std::ptrdiff_t>(selected_raw_bytes_per_expert);
    packed_by_expert.push_back(iq36::PackQ4Kx8(
        std::vector<std::uint8_t>(begin, end), rows_per_expert,
        blocks_per_row));
  }
  const auto shared_packed =
      iq36::PackQ4Kx8(shared_raw, rows_per_expert, blocks_per_row);

  OpenClApi api;
  const auto selected = SelectDevice(api, device_substring);
  run.platform_name = selected.platform_name;
  run.device_name = selected.device_name;

  cl_int err = kClSuccess;
  cl_context context =
      api.clCreateContext(nullptr, 1, &selected.device, nullptr, nullptr, &err);
  Check(err, "clCreateContext");
  cl_command_queue queue =
      api.clCreateCommandQueue(context, selected.device,
                               kClQueueProfilingEnable, &err);
  Check(err, "clCreateCommandQueue");
  const char* source = kOpenClSource;
  const std::size_t source_len = std::strlen(kOpenClSource);
  cl_program program =
      api.clCreateProgramWithSource(context, 1, &source, &source_len, &err);
  Check(err, "clCreateProgramWithSource");
  const auto build_begin = std::chrono::steady_clock::now();
  err = api.clBuildProgram(program, 1, &selected.device, "", nullptr, nullptr);
  const auto build_end = std::chrono::steady_clock::now();
  run.program_build_ms =
      std::chrono::duration<double, std::milli>(build_end - build_begin).count();
  run.build_log = BuildLog(api, program, selected.device);
  if (err != kClSuccess) {
    throw std::runtime_error(
        std::string("clBuildProgram failed: ") + run.build_log);
  }
  Check(err, "clBuildProgram");
  cl_kernel shared_kernel =
      api.clCreateKernel(program, "q4k_x8_matvec_rowlane", &err);
  Check(err, "clCreateKernel(q4k_x8_matvec_rowlane)");
  cl_kernel combined_kernel = api.clCreateKernel(
      program, "q4k_x8_selected_down_expert8_plus_shared_q4", &err);
  Check(err,
        "clCreateKernel(q4k_x8_selected_down_expert8_plus_shared_q4)");

  std::array<cl_mem, kExpertUsedCount> packed_buffers{};
  cl_mem shared_raw_buffer = nullptr;
  cl_mem selected_q8_qs_buffer = nullptr;
  cl_mem selected_q8_bsums_buffer = nullptr;
  cl_mem selected_q8_d_buffer = nullptr;
  cl_mem shared_q8_qs_buffer = nullptr;
  cl_mem shared_q8_bsums_buffer = nullptr;
  cl_mem shared_q8_d_buffer = nullptr;
  cl_mem shared_separate_output_buffer = nullptr;
  cl_mem selected_combined_output_buffer = nullptr;
  cl_mem shared_combined_output_buffer = nullptr;
  auto release_all = [&]() {
    ReleaseMem(api, &shared_combined_output_buffer);
    ReleaseMem(api, &selected_combined_output_buffer);
    ReleaseMem(api, &shared_separate_output_buffer);
    ReleaseMem(api, &shared_q8_d_buffer);
    ReleaseMem(api, &shared_q8_bsums_buffer);
    ReleaseMem(api, &shared_q8_qs_buffer);
    ReleaseMem(api, &selected_q8_d_buffer);
    ReleaseMem(api, &selected_q8_bsums_buffer);
    ReleaseMem(api, &selected_q8_qs_buffer);
    ReleaseMem(api, &shared_raw_buffer);
    for (auto& buffer : packed_buffers) {
      ReleaseMem(api, &buffer);
    }
  };

  try {
    for (std::size_t i = 0; i < packed_buffers.size(); ++i) {
      packed_buffers[i] = api.clCreateBuffer(
          context, kClMemReadOnly, packed_by_expert[i].size(), nullptr, &err);
      Check(err, "clCreateBuffer(selected Q4 packed)");
      Check(api.clEnqueueWriteBuffer(queue, packed_buffers[i], kClTrue, 0,
                                     packed_by_expert[i].size(),
                                     packed_by_expert[i].data(), 0, nullptr,
                                     nullptr),
            "clEnqueueWriteBuffer(selected Q4 packed)");
    }
    shared_raw_buffer =
        api.clCreateBuffer(context, kClMemReadOnly, shared_packed.size(), nullptr,
                           &err);
    Check(err, "clCreateBuffer(shared Q4 packed)");
    selected_q8_qs_buffer = api.clCreateBuffer(
        context, kClMemReadOnly,
        selected_q8.qs.size() * sizeof(std::int8_t), nullptr, &err);
    Check(err, "clCreateBuffer(selected q8 qs)");
    selected_q8_bsums_buffer = api.clCreateBuffer(
        context, kClMemReadOnly,
        selected_q8.bsums.size() * sizeof(std::int16_t), nullptr, &err);
    Check(err, "clCreateBuffer(selected q8 bsums)");
    selected_q8_d_buffer = api.clCreateBuffer(
        context, kClMemReadOnly,
        selected_q8.d.size() * sizeof(float), nullptr, &err);
    Check(err, "clCreateBuffer(selected q8 d)");
    shared_q8_qs_buffer = api.clCreateBuffer(
        context, kClMemReadOnly,
        shared_q8.qs.size() * sizeof(std::int8_t), nullptr, &err);
    Check(err, "clCreateBuffer(shared q8 qs)");
    shared_q8_bsums_buffer = api.clCreateBuffer(
        context, kClMemReadOnly,
        shared_q8.bsums.size() * sizeof(std::int16_t), nullptr, &err);
    Check(err, "clCreateBuffer(shared q8 bsums)");
    shared_q8_d_buffer = api.clCreateBuffer(
        context, kClMemReadOnly, shared_q8.d.size() * sizeof(float), nullptr,
        &err);
    Check(err, "clCreateBuffer(shared q8 d)");
    shared_separate_output_buffer = api.clCreateBuffer(
        context, kClMemWriteOnly,
        run.shared_separate_output.size() * sizeof(float), nullptr, &err);
    Check(err, "clCreateBuffer(shared separate output)");
    selected_combined_output_buffer = api.clCreateBuffer(
        context, kClMemWriteOnly,
        run.selected_combined_output.size() * sizeof(float), nullptr, &err);
    Check(err, "clCreateBuffer(selected combined output)");
    shared_combined_output_buffer = api.clCreateBuffer(
        context, kClMemWriteOnly,
        run.shared_combined_output.size() * sizeof(float), nullptr, &err);
    Check(err, "clCreateBuffer(shared combined output)");

    Check(api.clEnqueueWriteBuffer(queue, shared_raw_buffer, kClTrue, 0,
                                   shared_packed.size(), shared_packed.data(), 0,
                                   nullptr, nullptr),
          "clEnqueueWriteBuffer(shared Q4 packed)");
    Check(api.clEnqueueWriteBuffer(
              queue, selected_q8_qs_buffer, kClTrue, 0,
              selected_q8.qs.size() * sizeof(std::int8_t),
              selected_q8.qs.data(), 0, nullptr, nullptr),
          "clEnqueueWriteBuffer(selected q8 qs)");
    Check(api.clEnqueueWriteBuffer(
              queue, selected_q8_bsums_buffer, kClTrue, 0,
              selected_q8.bsums.size() * sizeof(std::int16_t),
              selected_q8.bsums.data(), 0, nullptr, nullptr),
          "clEnqueueWriteBuffer(selected q8 bsums)");
    Check(api.clEnqueueWriteBuffer(queue, selected_q8_d_buffer, kClTrue, 0,
                                   selected_q8.d.size() * sizeof(float),
                                   selected_q8.d.data(), 0, nullptr, nullptr),
          "clEnqueueWriteBuffer(selected q8 d)");
    Check(api.clEnqueueWriteBuffer(
              queue, shared_q8_qs_buffer, kClTrue, 0,
              shared_q8.qs.size() * sizeof(std::int8_t),
              shared_q8.qs.data(), 0, nullptr, nullptr),
          "clEnqueueWriteBuffer(shared q8 qs)");
    Check(api.clEnqueueWriteBuffer(
              queue, shared_q8_bsums_buffer, kClTrue, 0,
              shared_q8.bsums.size() * sizeof(std::int16_t),
              shared_q8.bsums.data(), 0, nullptr, nullptr),
          "clEnqueueWriteBuffer(shared q8 bsums)");
    Check(api.clEnqueueWriteBuffer(queue, shared_q8_d_buffer, kClTrue, 0,
                                   shared_q8.d.size() * sizeof(float),
                                   shared_q8.d.data(), 0, nullptr, nullptr),
          "clEnqueueWriteBuffer(shared q8 d)");

    const cl_uint blocks_arg = static_cast<cl_uint>(blocks_per_row);
    const cl_uint row_groups_arg =
        static_cast<cl_uint>(rows_per_expert / 8U);
    Check(api.clSetKernelArg(shared_kernel, 0, sizeof(shared_raw_buffer),
                             &shared_raw_buffer),
          "clSetKernelArg(shared raw)");
    Check(api.clSetKernelArg(shared_kernel, 1, sizeof(shared_q8_qs_buffer),
                             &shared_q8_qs_buffer),
          "clSetKernelArg(shared q8 qs)");
    Check(api.clSetKernelArg(shared_kernel, 2, sizeof(shared_q8_bsums_buffer),
                             &shared_q8_bsums_buffer),
          "clSetKernelArg(shared q8 bsums)");
    Check(api.clSetKernelArg(shared_kernel, 3, sizeof(shared_q8_d_buffer),
                             &shared_q8_d_buffer),
          "clSetKernelArg(shared q8 d)");
    Check(api.clSetKernelArg(shared_kernel, 4, sizeof(blocks_arg), &blocks_arg),
          "clSetKernelArg(shared blocks)");
    Check(api.clSetKernelArg(shared_kernel, 5, sizeof(row_groups_arg),
                             &row_groups_arg),
          "clSetKernelArg(shared row groups)");
    Check(api.clSetKernelArg(shared_kernel, 6,
                             sizeof(shared_separate_output_buffer),
                             &shared_separate_output_buffer),
          "clSetKernelArg(shared output)");

    for (std::size_t i = 0; i < packed_buffers.size(); ++i) {
      Check(api.clSetKernelArg(combined_kernel, static_cast<cl_uint>(i),
                               sizeof(packed_buffers[i]), &packed_buffers[i]),
            "clSetKernelArg(combined selected packed)");
    }
    Check(api.clSetKernelArg(combined_kernel, 8, sizeof(shared_raw_buffer),
                             &shared_raw_buffer),
          "clSetKernelArg(combined shared raw)");
    Check(api.clSetKernelArg(combined_kernel, 9, sizeof(selected_q8_qs_buffer),
                             &selected_q8_qs_buffer),
          "clSetKernelArg(combined selected q8 qs)");
    Check(api.clSetKernelArg(combined_kernel, 10,
                             sizeof(selected_q8_bsums_buffer),
                             &selected_q8_bsums_buffer),
          "clSetKernelArg(combined selected q8 bsums)");
    Check(api.clSetKernelArg(combined_kernel, 11, sizeof(selected_q8_d_buffer),
                             &selected_q8_d_buffer),
          "clSetKernelArg(combined selected q8 d)");
    Check(api.clSetKernelArg(combined_kernel, 12, sizeof(shared_q8_qs_buffer),
                             &shared_q8_qs_buffer),
          "clSetKernelArg(combined shared q8 qs)");
    Check(api.clSetKernelArg(combined_kernel, 13,
                             sizeof(shared_q8_bsums_buffer),
                             &shared_q8_bsums_buffer),
          "clSetKernelArg(combined shared q8 bsums)");
    Check(api.clSetKernelArg(combined_kernel, 14, sizeof(shared_q8_d_buffer),
                             &shared_q8_d_buffer),
          "clSetKernelArg(combined shared q8 d)");
    Check(api.clSetKernelArg(combined_kernel, 15, sizeof(blocks_arg),
                             &blocks_arg),
          "clSetKernelArg(combined blocks)");
    Check(api.clSetKernelArg(combined_kernel, 16, sizeof(row_groups_arg),
                             &row_groups_arg),
          "clSetKernelArg(combined row groups)");
    Check(api.clSetKernelArg(combined_kernel, 17,
                             sizeof(selected_combined_output_buffer),
                             &selected_combined_output_buffer),
          "clSetKernelArg(combined selected output)");
    Check(api.clSetKernelArg(combined_kernel, 18,
                             sizeof(shared_combined_output_buffer),
                             &shared_combined_output_buffer),
          "clSetKernelArg(combined shared output)");

    const std::size_t shared_global = static_cast<std::size_t>(rows_per_expert);
    const std::size_t selected_global =
        static_cast<std::size_t>(selected_count * rows_per_expert);
    const std::size_t combined_global = selected_global + shared_global;
    constexpr std::size_t kLocal = 64;
    const std::size_t* shared_local =
        shared_global % kLocal == 0 ? &kLocal : nullptr;
    const std::size_t* combined_local =
        combined_global % kLocal == 0 ? &kLocal : nullptr;
    std::vector<double> shared_times;
    std::vector<double> combined_times;
    shared_times.reserve(static_cast<std::size_t>(repeat));
    combined_times.reserve(static_cast<std::size_t>(repeat));
    for (int i = 0; i < repeat; ++i) {
      cl_event shared_event = nullptr;
      Check(api.clEnqueueNDRangeKernel(queue, shared_kernel, 1, nullptr,
                                       &shared_global, shared_local, 0,
                                       nullptr, &shared_event),
            "clEnqueueNDRangeKernel(shared Q4 rowlane)");
      Check(api.clFinish(queue), "clFinish(shared Q4 rowlane)");
      shared_times.push_back(EventUs(api, shared_event));
      api.clReleaseEvent(shared_event);

      cl_event combined_event = nullptr;
      Check(api.clEnqueueNDRangeKernel(queue, combined_kernel, 1, nullptr,
                                       &combined_global, combined_local, 0,
                                       nullptr, &combined_event),
            "clEnqueueNDRangeKernel(selected Q4 + shared Q4 combined)");
      Check(api.clFinish(queue), "clFinish(selected Q4 + shared Q4 combined)");
      combined_times.push_back(EventUs(api, combined_event));
      api.clReleaseEvent(combined_event);
    }

    Check(api.clEnqueueReadBuffer(
              queue, shared_separate_output_buffer, kClTrue, 0,
              run.shared_separate_output.size() * sizeof(float),
              run.shared_separate_output.data(), 0, nullptr, nullptr),
          "clEnqueueReadBuffer(shared separate output)");
    Check(api.clEnqueueReadBuffer(
              queue, selected_combined_output_buffer, kClTrue, 0,
              run.selected_combined_output.size() * sizeof(float),
              run.selected_combined_output.data(), 0, nullptr, nullptr),
          "clEnqueueReadBuffer(selected combined output)");
    Check(api.clEnqueueReadBuffer(
              queue, shared_combined_output_buffer, kClTrue, 0,
              run.shared_combined_output.size() * sizeof(float),
              run.shared_combined_output.data(), 0, nullptr, nullptr),
          "clEnqueueReadBuffer(shared combined output)");

    const auto fill_timing = [](const std::vector<double>& times,
                                std::uint64_t global,
                                double raw_bytes,
                                double io_bytes) {
      DownTiming timing;
      timing.min_us = *std::min_element(times.begin(), times.end());
      timing.mean_us =
          std::accumulate(times.begin(), times.end(), 0.0) /
          static_cast<double>(times.size());
      timing.effective_raw_gb_s = raw_bytes / (timing.min_us / 1e6) / 1e9;
      timing.effective_io_gb_s = io_bytes / (timing.min_us / 1e6) / 1e9;
      timing.global_work_items = global;
      timing.kernel_launches = 1;
      return timing;
    };
    const double selected_raw_bytes = static_cast<double>(selected_raw.size());
    const double shared_raw_bytes = static_cast<double>(shared_raw.size());
    const double selected_io_bytes =
        selected_raw_bytes +
        static_cast<double>(selected_q8.qs.size() * sizeof(std::int8_t)) +
        static_cast<double>(selected_q8.bsums.size() * sizeof(std::int16_t)) +
        static_cast<double>(selected_q8.d.size() * sizeof(float)) +
        static_cast<double>(run.selected_combined_output.size() * sizeof(float));
    const double shared_io_bytes =
        shared_raw_bytes +
        static_cast<double>(shared_q8.qs.size() * sizeof(std::int8_t)) +
        static_cast<double>(shared_q8.bsums.size() * sizeof(std::int16_t)) +
        static_cast<double>(shared_q8.d.size() * sizeof(float)) +
        static_cast<double>(run.shared_combined_output.size() * sizeof(float));
    run.shared_timing =
        fill_timing(shared_times, shared_global, shared_raw_bytes,
                    shared_io_bytes);
    run.combined_timing =
        fill_timing(combined_times, combined_global,
                    selected_raw_bytes + shared_raw_bytes,
                    selected_io_bytes + shared_io_bytes);
  } catch (...) {
    release_all();
    api.clReleaseKernel(combined_kernel);
    api.clReleaseKernel(shared_kernel);
    api.clReleaseProgram(program);
    api.clReleaseCommandQueue(queue);
    api.clReleaseContext(context);
    throw;
  }
  release_all();
  api.clReleaseKernel(combined_kernel);
  api.clReleaseKernel(shared_kernel);
  api.clReleaseProgram(program);
  api.clReleaseCommandQueue(queue);
  api.clReleaseContext(context);
  return run;
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

void WriteQ8PlaneCompare(const Q8PlaneCompareStats& stats) {
  std::cout << "{";
  std::cout << "\"qs_same_size\":" << (stats.qs_same_size ? "true" : "false") << ",";
  std::cout << "\"bsums_same_size\":" << (stats.bsums_same_size ? "true" : "false") << ",";
  std::cout << "\"d_same_size\":" << (stats.d_same_size ? "true" : "false") << ",";
  std::cout << "\"qs_exact\":" << (stats.qs_exact ? "true" : "false") << ",";
  std::cout << "\"bsums_exact\":" << (stats.bsums_exact ? "true" : "false") << ",";
  std::cout << "\"d_finite\":" << (stats.d_finite ? "true" : "false") << ",";
  std::cout << "\"d_within_threshold\":" << (stats.d_within_threshold ? "true" : "false") << ",";
  std::cout << "\"qs_value_count\":" << stats.qs_value_count << ",";
  std::cout << "\"bsums_value_count\":" << stats.bsums_value_count << ",";
  std::cout << "\"d_value_count\":" << stats.d_value_count << ",";
  std::cout << "\"qs_mismatch_count\":" << stats.qs_mismatch_count << ",";
  std::cout << "\"bsums_mismatch_count\":" << stats.bsums_mismatch_count << ",";
  std::cout << "\"d_mismatch_count\":" << stats.d_mismatch_count << ",";
  std::cout << "\"d_max_abs_diff\":" << stats.d_max_abs_diff << ",";
  std::cout << "\"d_rmse\":" << stats.d_rmse << ",";
  std::cout << "\"passed\":" << (stats.passed ? "true" : "false");
  std::cout << "}";
}

void WriteI32Vector(const std::vector<std::int32_t>& values) {
  std::cout << "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0) std::cout << ",";
    std::cout << values[i];
  }
  std::cout << "]";
}

std::vector<float> ApplySelectedWeights(const std::vector<float>& down,
                                        const std::vector<float>& weights,
                                        std::uint64_t rows_per_expert) {
  Require(weights.size() == kExpertUsedCount,
          "selected weight count mismatch");
  Require(down.size() == weights.size() * rows_per_expert,
          "selected weighted down input size mismatch");
  std::vector<float> weighted(down.size(), 0.0f);
  for (std::size_t expert = 0; expert < weights.size(); ++expert) {
    const auto base = expert * static_cast<std::size_t>(rows_per_expert);
    for (std::uint64_t row = 0; row < rows_per_expert; ++row) {
      weighted[base + static_cast<std::size_t>(row)] =
          down[base + static_cast<std::size_t>(row)] * weights[expert];
    }
  }
  return weighted;
}

std::vector<float> SumSelectedExpertRows(const std::vector<float>& values,
                                         std::uint64_t rows_per_expert) {
  Require(values.size() == static_cast<std::size_t>(kExpertUsedCount) *
                               static_cast<std::size_t>(rows_per_expert),
          "selected expert sum input size mismatch");
  std::vector<float> summed(static_cast<std::size_t>(rows_per_expert), 0.0f);
  for (int expert = 0; expert < kExpertUsedCount; ++expert) {
    const auto base = static_cast<std::size_t>(expert) *
                      static_cast<std::size_t>(rows_per_expert);
    for (std::uint64_t row = 0; row < rows_per_expert; ++row) {
      summed[static_cast<std::size_t>(row)] +=
          values[base + static_cast<std::size_t>(row)];
    }
  }
  return summed;
}

float SigmoidF32(float value) {
  return 1.0f / (1.0f + std::exp(-value));
}

std::vector<float> BuildFfnOut(const std::vector<float>& selected_weighted_sum,
                               const std::vector<float>& shared_down,
                               float shared_gate_value) {
  Require(selected_weighted_sum.size() == shared_down.size(),
          "FFN out component size mismatch");
  const float sigmoid = SigmoidF32(shared_gate_value);
  std::vector<float> out(selected_weighted_sum.size(), 0.0f);
  for (std::size_t i = 0; i < out.size(); ++i) {
    out[i] = selected_weighted_sum[i] + shared_down[i] * sigmoid;
  }
  return out;
}

std::vector<float> AddVectorsChecked(const std::vector<float>& a,
                                     const std::vector<float>& b) {
  Require(a.size() == b.size(), "vector add size mismatch");
  std::vector<float> out(a.size(), 0.0f);
  for (std::size_t i = 0; i < out.size(); ++i) {
    out[i] = a[i] + b[i];
  }
  return out;
}

std::vector<float> BuildDownTailContributions(
    const std::vector<float>& selected_weighted,
    const std::vector<float>& shared_down,
    float shared_gate_value,
    std::uint64_t rows_per_expert) {
  Require(selected_weighted.size() == static_cast<std::size_t>(kExpertUsedCount) *
                                      static_cast<std::size_t>(rows_per_expert),
          "down-tail selected contribution size mismatch");
  Require(shared_down.size() == static_cast<std::size_t>(rows_per_expert),
          "down-tail shared contribution size mismatch");
  std::vector<float> contributions(
      static_cast<std::size_t>(rows_per_expert * 9), 0.0f);
  std::copy(selected_weighted.begin(), selected_weighted.end(),
            contributions.begin());
  const float sigmoid = SigmoidF32(shared_gate_value);
  const auto shared_base =
      static_cast<std::size_t>(kExpertUsedCount * rows_per_expert);
  for (std::uint64_t row = 0; row < rows_per_expert; ++row) {
    contributions[shared_base + static_cast<std::size_t>(row)] =
        shared_down[static_cast<std::size_t>(row)] * sigmoid;
  }
  return contributions;
}

std::vector<float> RepeatVector(const std::vector<float>& values,
                                std::uint64_t groups) {
  Require(groups > 0, "repeat group count must be positive");
  std::vector<float> repeated;
  repeated.reserve(values.size() * static_cast<std::size_t>(groups));
  for (std::uint64_t group = 0; group < groups; ++group) {
    repeated.insert(repeated.end(), values.begin(), values.end());
  }
  return repeated;
}

int main(int argc, char** argv) {
  try {
    const Args args = ParseArgs(argc, argv);
    const auto index = iq36::parse_gguf_model_index(args.model_path);
    const auto load_map = iq36::validate_qwen36_load_map(index);
    const std::string tensor_name = LayerTensorName(args.layer, "ffn_down_exps.weight");
    const std::string shared_gate_tensor_name =
        LayerTensorName(args.layer, "ffn_gate_shexp.weight");
    const std::string shared_up_tensor_name =
        LayerTensorName(args.layer, "ffn_up_shexp.weight");
    const std::string shared_input_gate_tensor_name =
        LayerTensorName(args.layer, "ffn_gate_inp_shexp.weight");
    const std::string shared_down_tensor_name =
        LayerTensorName(args.layer, "ffn_down_shexp.weight");
    const auto* tensor = iq36::find_tensor(index, tensor_name);
    const auto* shared_gate_tensor = iq36::find_tensor(index, shared_gate_tensor_name);
    const auto* shared_up_tensor = iq36::find_tensor(index, shared_up_tensor_name);
    const auto* shared_input_gate_tensor =
        iq36::find_tensor(index, shared_input_gate_tensor_name);
    const auto* shared_down_tensor = iq36::find_tensor(index, shared_down_tensor_name);
    Require(tensor != nullptr, "selected expert down tensor missing");
    Require(shared_gate_tensor != nullptr, "shared gate tensor missing");
    Require(shared_up_tensor != nullptr, "shared up tensor missing");
    Require(shared_input_gate_tensor != nullptr, "shared input gate tensor missing");
    Require(shared_down_tensor != nullptr, "shared down tensor missing");
    const bool tensor_shape_ok =
        (tensor->type == 12 || tensor->type == 14) &&
        tensor->dims == std::vector<std::uint64_t>{kIntermediateSize, kHiddenSize, kExpertCount};
    const bool shared_down_tensor_shape_ok =
        (shared_down_tensor->type == 12 || shared_down_tensor->type == 14) &&
        shared_down_tensor->dims == std::vector<std::uint64_t>{kIntermediateSize, kHiddenSize};
    const bool shared_input_gate_tensor_shape_ok =
        shared_input_gate_tensor->type == 0 &&
        shared_input_gate_tensor->dims == std::vector<std::uint64_t>{kHiddenSize};

    const auto attn_post_norm =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "attn_post_norm.bin"));
    const auto input = iq36::read_f32_vector_file(JoinPath(args.payload_dir, "ffn_moe_swiglu.bin"));
    const auto gate_up = iq36::read_f32_vector_file(JoinPath(args.payload_dir, "ffn_moe_gate_up.bin"));
    const auto weights =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "ffn_moe_weights_norm.bin"));
    const auto expert_ids = ReadI32VectorFile(JoinPath(args.payload_dir, "ffn_moe_topk.bin"));
    const auto oracle = iq36::read_f32_vector_file(JoinPath(args.payload_dir, "ffn_moe_down.bin"));
    const auto weighted_oracle =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "ffn_moe_weighted.bin"));
    const auto shared_oracle =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "ffn_shexp.bin"));
    const auto attn_residual =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "attn_residual.bin"));
    const auto ffn_out_oracle =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "ffn_out.bin"));
    const auto shared_tail_gate = iq36::matvec_tensor(
        args.model_path, index, shared_input_gate_tensor_name, attn_post_norm);
    Require(shared_tail_gate.size() == 1, "shared input gate output size mismatch");
    const auto cpu = iq36::matvec_expert_tensor_per_expert_input(
        args.model_path, index, tensor_name, input, expert_ids);

    const std::uint64_t cols = tensor->dims[0];
    const std::uint64_t rows_per_expert = tensor->dims[1];
    const std::uint64_t selected_rows = rows_per_expert * expert_ids.size();
    const std::uint64_t blocks_per_row = cols / kQ8BlockValues;
    const std::uint64_t block_nbytes =
        tensor->type == 12 ? kQ4KBlockBytes : kQ6KBlockBytes;
    const std::uint64_t row_nbytes =
        iq36::ggml_tensor_nbytes(tensor->type, std::vector<std::uint64_t>{cols});
    Require(row_nbytes == blocks_per_row * block_nbytes,
            "selected expert down row byte mismatch");
    std::ifstream model(args.model_path, std::ios::binary);
    Require(static_cast<bool>(model), "failed to open model");
    const auto selected_raw =
        ReadSelectedExpertRaw(model, *tensor, expert_ids, rows_per_expert, row_nbytes);
    const auto q8 = QuantizePerExpertQ8K(input, expert_ids.size(), cols);
    DownRun gpu;
    DownRun q4_pair2;
    DownRun q4_weighted;
    DownRun q4_weighted_sum;
    DownRun q4_group8_sum;
    DownRun q4_occupancy4;
    DeviceQ8DownRun q4_device_q8;
    DeviceSwiGluQ8DownRun q4_swiglu_device_q8;
    DeviceSwiGluF32InputDownRun q4_swiglu_f32_input;
    SelectedSharedQ6DownRun q4_selected_shared;
    SelectedSharedQ6DownRun q6_selected_shared;
    DownTailNonAtomicRun q6_nonatomic_tail;
    DownTailRowgroupRun q6_rowgroup_tail;
    std::vector<float> shared_gate;
    std::vector<float> shared_up;
    std::vector<float> shared_swiglu;
    std::vector<float> shared_cpu;
    bool q4_pair2_measured = false;
    bool q4_weighted_measured = false;
    bool q4_weighted_sum_measured = false;
    bool q4_group8_sum_measured = false;
    bool q4_occupancy4_measured = false;
    bool q4_device_q8_measured = false;
    bool q4_swiglu_device_q8_measured = false;
    bool q4_swiglu_f32_input_measured = false;
    bool q4_selected_shared_measured = false;
    bool q6_selected_shared_measured = false;
    bool q6_nonatomic_tail_measured = false;
    bool q6_rowgroup_tail_measured = false;
    constexpr std::uint64_t kOccupancyGroupCount = 4;
    if (tensor->type == 12) {
      gpu = RunGpuSelectedDownQ4Expert8Kernel(
          selected_raw, input, nullptr, rows_per_expert, blocks_per_row,
          expert_ids.size(), args.device_substring, args.repeat,
          "q4k_x8_matvec_rowlane_expert8_multiq8", false, false, false);
      q4_pair2 = RunGpuSelectedDownQ4Expert8Kernel(
          selected_raw, input, nullptr, rows_per_expert, blocks_per_row,
          expert_ids.size(), args.device_substring, args.repeat,
          "q4k_x8_matvec_rowlane_expert8_multiq8_pair2", true, false, false);
      q4_weighted = RunGpuSelectedDownQ4Expert8Kernel(
          selected_raw, input, &weights, rows_per_expert, blocks_per_row,
          expert_ids.size(), args.device_substring, args.repeat,
          "q4k_x8_matvec_rowlane_expert8_multiq8_weighted", false, false, false);
      q4_weighted_sum = RunGpuSelectedDownQ4Expert8Kernel(
          selected_raw, input, &weights, rows_per_expert, blocks_per_row,
          expert_ids.size(), args.device_substring, args.repeat,
          "q4k_x8_matvec_rowlane_expert8_multiq8_weighted_sum", false, true, false);
      q4_group8_sum = RunGpuSelectedDownQ4Expert8Kernel(
          selected_raw, input, &weights, rows_per_expert, blocks_per_row,
          expert_ids.size(), args.device_substring, args.repeat,
          "q4k_x8_matvec_rowlane_expert8_multiq8_group8_weighted_sum",
          false, false, true);
      q4_occupancy4 = RunGpuSelectedDownQ4Expert8Kernel(
          selected_raw, input, nullptr, rows_per_expert, blocks_per_row,
          expert_ids.size(), args.device_substring, args.repeat,
          "q4k_x8_matvec_rowlane_expert8_multiq8_occupancy4", false, false, false,
          kOccupancyGroupCount);
      q4_device_q8 = RunGpuSelectedDownQ4DeviceQ8Component(
          selected_raw, input, rows_per_expert, blocks_per_row,
          expert_ids.size(), args.device_substring, args.repeat);
      q4_swiglu_device_q8 = RunGpuSelectedDownQ4SwiGluDeviceQ8Component(
          selected_raw, gate_up, rows_per_expert, blocks_per_row,
          expert_ids.size(), args.device_substring, args.repeat);
      q4_swiglu_f32_input = RunGpuSelectedDownQ4SwiGluF32InputComponent(
          selected_raw, gate_up, rows_per_expert, blocks_per_row,
          expert_ids.size(), args.device_substring, args.repeat);
      if (shared_down_tensor->type == 12) {
        shared_gate = iq36::matvec_tensor(
            args.model_path, index, shared_gate_tensor_name, attn_post_norm);
        shared_up = iq36::matvec_tensor(
            args.model_path, index, shared_up_tensor_name, attn_post_norm);
        shared_swiglu = iq36::apply_swiglu_pair(shared_gate, shared_up);
        shared_cpu = iq36::matvec_tensor(
            args.model_path, index, shared_down_tensor_name, shared_swiglu);
        const auto shared_q8 = QuantizePerExpertQ8K(shared_swiglu, 1, cols);
        const auto shared_raw = ReadTensorBytes(model, *shared_down_tensor);
        q4_selected_shared = RunGpuSelectedSharedDownQ4Q4(
            selected_raw, shared_raw, q8, shared_q8, gpu, rows_per_expert,
            blocks_per_row, expert_ids.size(), args.device_substring,
            args.repeat);
        q4_selected_shared_measured = true;
      }
      q4_pair2_measured = true;
      q4_weighted_measured = true;
      q4_weighted_sum_measured = true;
      q4_group8_sum_measured = true;
      q4_occupancy4_measured = true;
      q4_device_q8_measured = true;
      q4_swiglu_device_q8_measured = true;
      q4_swiglu_f32_input_measured = true;
    } else {
      gpu = RunGpuSelectedDownQ6(selected_raw, q8, rows_per_expert,
                                blocks_per_row, expert_ids.size(),
                                args.device_substring, args.repeat);
      Require(shared_down_tensor->type == 14,
              "selected+shared Q6 probe requires shared Q6 down tensor");
      shared_gate = iq36::matvec_tensor(
          args.model_path, index, shared_gate_tensor_name, attn_post_norm);
      shared_up = iq36::matvec_tensor(
          args.model_path, index, shared_up_tensor_name, attn_post_norm);
      shared_swiglu = iq36::apply_swiglu_pair(shared_gate, shared_up);
      shared_cpu = iq36::matvec_tensor(
          args.model_path, index, shared_down_tensor_name, shared_swiglu);
      const auto shared_q8 = QuantizePerExpertQ8K(shared_swiglu, 1, cols);
      const auto shared_raw = ReadTensorBytes(model, *shared_down_tensor);
      q6_selected_shared = RunGpuSelectedSharedDownQ6Rowstripe(
          selected_raw, shared_raw, q8, shared_q8, rows_per_expert,
          blocks_per_row, expert_ids.size(), args.device_substring,
          args.repeat);
      q6_nonatomic_tail = RunGpuSelectedSharedDownTailQ6NonAtomic(
          selected_raw, shared_raw, q8, shared_q8, weights, shared_tail_gate,
          attn_residual, rows_per_expert, blocks_per_row, expert_ids.size(),
          args.device_substring, args.repeat);
      q6_rowgroup_tail = RunGpuSelectedSharedDownTailQ6RowgroupReduce(
          selected_raw, shared_raw, q8, shared_q8, weights, shared_tail_gate,
          attn_residual, rows_per_expert, blocks_per_row, expert_ids.size(),
          args.device_substring, args.repeat);
      q6_selected_shared_measured = true;
      q6_nonatomic_tail_measured = true;
      q6_rowgroup_tail_measured = true;
    }

    const auto cpu_vs_oracle = iq36::compare_vectors(cpu, oracle, kMismatchThreshold);
    const auto gpu_vs_cpu = iq36::compare_vectors(gpu.output, cpu, kMismatchThreshold);
    const auto gpu_vs_oracle = iq36::compare_vectors(gpu.output, oracle, kMismatchThreshold);
    const auto cpu_weighted = ApplySelectedWeights(cpu, weights, rows_per_expert);
    const auto gpu_weighted =
        ApplySelectedWeights(gpu.output, weights, rows_per_expert);
    const auto cpu_weighted_sum = SumSelectedExpertRows(cpu_weighted, rows_per_expert);
    const auto oracle_weighted_sum =
        SumSelectedExpertRows(weighted_oracle, rows_per_expert);
    const auto gpu_weighted_sum =
        SumSelectedExpertRows(gpu_weighted, rows_per_expert);
    const auto weighted_kernel_sum =
        q4_weighted_measured
            ? SumSelectedExpertRows(q4_weighted.output, rows_per_expert)
            : std::vector<float>{};
    const auto repeated_cpu = RepeatVector(cpu, kOccupancyGroupCount);
    const auto repeated_oracle = RepeatVector(oracle, kOccupancyGroupCount);
    const auto weighted_cpu_vs_oracle =
        iq36::compare_vectors(cpu_weighted, weighted_oracle, kMismatchThreshold);
    const auto weighted_sum_cpu_vs_oracle =
        iq36::compare_vectors(cpu_weighted_sum, oracle_weighted_sum, kMismatchThreshold);
    const auto pair2_vs_cpu =
        q4_pair2_measured
            ? iq36::compare_vectors(q4_pair2.output, cpu, kMismatchThreshold)
            : iq36::VectorCompareStats{};
    const auto pair2_vs_oracle =
        q4_pair2_measured
            ? iq36::compare_vectors(q4_pair2.output, oracle, kMismatchThreshold)
            : iq36::VectorCompareStats{};
    const auto pair2_vs_current =
        q4_pair2_measured
            ? iq36::compare_vectors(q4_pair2.output, gpu.output, kMismatchThreshold)
            : iq36::VectorCompareStats{};
    const auto weighted_vs_cpu =
        q4_weighted_measured
            ? iq36::compare_vectors(q4_weighted.output, cpu_weighted, kMismatchThreshold)
            : iq36::VectorCompareStats{};
    const auto weighted_vs_oracle =
        q4_weighted_measured
            ? iq36::compare_vectors(q4_weighted.output, weighted_oracle, kMismatchThreshold)
            : iq36::VectorCompareStats{};
    const auto weighted_vs_current_weighted =
        q4_weighted_measured
            ? iq36::compare_vectors(q4_weighted.output, gpu_weighted, kMismatchThreshold)
            : iq36::VectorCompareStats{};
    const auto weighted_sum_vs_cpu =
        q4_weighted_sum_measured
            ? iq36::compare_vectors(q4_weighted_sum.output, cpu_weighted_sum, kMismatchThreshold)
            : iq36::VectorCompareStats{};
    const auto weighted_sum_vs_oracle =
        q4_weighted_sum_measured
            ? iq36::compare_vectors(q4_weighted_sum.output, oracle_weighted_sum, kMismatchThreshold)
            : iq36::VectorCompareStats{};
    const auto weighted_sum_vs_current =
        q4_weighted_sum_measured
            ? iq36::compare_vectors(q4_weighted_sum.output, gpu_weighted_sum, kMismatchThreshold)
            : iq36::VectorCompareStats{};
    const auto weighted_sum_vs_weighted_kernel =
        q4_weighted_sum_measured && q4_weighted_measured
            ? iq36::compare_vectors(q4_weighted_sum.output, weighted_kernel_sum, kMismatchThreshold)
            : iq36::VectorCompareStats{};
    const auto group8_sum_vs_cpu =
        q4_group8_sum_measured
            ? iq36::compare_vectors(q4_group8_sum.output, cpu_weighted_sum, kMismatchThreshold)
            : iq36::VectorCompareStats{};
    const auto group8_sum_vs_oracle =
        q4_group8_sum_measured
            ? iq36::compare_vectors(q4_group8_sum.output, oracle_weighted_sum, kMismatchThreshold)
            : iq36::VectorCompareStats{};
    const auto group8_sum_vs_current =
        q4_group8_sum_measured
            ? iq36::compare_vectors(q4_group8_sum.output, gpu_weighted_sum, kMismatchThreshold)
            : iq36::VectorCompareStats{};
    const auto group8_sum_vs_weighted_kernel =
        q4_group8_sum_measured && q4_weighted_measured
            ? iq36::compare_vectors(q4_group8_sum.output, weighted_kernel_sum, kMismatchThreshold)
            : iq36::VectorCompareStats{};
    const auto occupancy4_vs_cpu =
        q4_occupancy4_measured
            ? iq36::compare_vectors(q4_occupancy4.output, repeated_cpu, kMismatchThreshold)
            : iq36::VectorCompareStats{};
    const auto occupancy4_vs_oracle =
        q4_occupancy4_measured
            ? iq36::compare_vectors(q4_occupancy4.output, repeated_oracle, kMismatchThreshold)
            : iq36::VectorCompareStats{};
    const auto device_q8_vs_cpu =
        q4_device_q8_measured
            ? iq36::compare_vectors(q4_device_q8.output, cpu, kMismatchThreshold)
            : iq36::VectorCompareStats{};
    const auto device_q8_vs_oracle =
        q4_device_q8_measured
            ? iq36::compare_vectors(q4_device_q8.output, oracle, kMismatchThreshold)
            : iq36::VectorCompareStats{};
    const auto device_q8_vs_current =
        q4_device_q8_measured
            ? iq36::compare_vectors(q4_device_q8.output, gpu.output, kMismatchThreshold)
            : iq36::VectorCompareStats{};
    const auto device_q8_planes_vs_cpu =
        q4_device_q8_measured ? CompareQ8Planes(q4_device_q8.q8, q8)
                              : Q8PlaneCompareStats{};
    const auto swiglu_device_q8_swiglu_vs_cpu =
        q4_swiglu_device_q8_measured
            ? iq36::compare_vectors(q4_swiglu_device_q8.swiglu, input, kMismatchThreshold)
            : iq36::VectorCompareStats{};
    const auto swiglu_device_q8_vs_cpu =
        q4_swiglu_device_q8_measured
            ? iq36::compare_vectors(q4_swiglu_device_q8.output, cpu, kMismatchThreshold)
            : iq36::VectorCompareStats{};
    const auto swiglu_device_q8_vs_oracle =
        q4_swiglu_device_q8_measured
            ? iq36::compare_vectors(q4_swiglu_device_q8.output, oracle, kMismatchThreshold)
            : iq36::VectorCompareStats{};
    const auto swiglu_device_q8_vs_current =
        q4_swiglu_device_q8_measured
            ? iq36::compare_vectors(q4_swiglu_device_q8.output, gpu.output, kMismatchThreshold)
            : iq36::VectorCompareStats{};
    const auto swiglu_device_q8_planes_vs_cpu =
        q4_swiglu_device_q8_measured ? CompareQ8Planes(q4_swiglu_device_q8.q8, q8)
                                     : Q8PlaneCompareStats{};
    const auto swiglu_f32_input_swiglu_vs_cpu =
        q4_swiglu_f32_input_measured
            ? iq36::compare_vectors(q4_swiglu_f32_input.swiglu, input, kMismatchThreshold)
            : iq36::VectorCompareStats{};
    const auto swiglu_f32_input_vs_cpu =
        q4_swiglu_f32_input_measured
            ? iq36::compare_vectors(q4_swiglu_f32_input.output, cpu, kMismatchThreshold)
            : iq36::VectorCompareStats{};
    const auto swiglu_f32_input_vs_oracle =
        q4_swiglu_f32_input_measured
            ? iq36::compare_vectors(q4_swiglu_f32_input.output, oracle, kMismatchThreshold)
            : iq36::VectorCompareStats{};
    const auto swiglu_f32_input_vs_current =
        q4_swiglu_f32_input_measured
            ? iq36::compare_vectors(q4_swiglu_f32_input.output, gpu.output, kMismatchThreshold)
            : iq36::VectorCompareStats{};
    const auto q4_selected_combined_vs_separate =
        q4_selected_shared_measured
            ? iq36::compare_vectors(q4_selected_shared.selected_combined_output,
                                    q4_selected_shared.selected_separate_output,
                                    kMismatchThreshold)
            : iq36::VectorCompareStats{};
    const auto q4_selected_combined_vs_oracle =
        q4_selected_shared_measured
            ? iq36::compare_vectors(q4_selected_shared.selected_combined_output,
                                    oracle, kMismatchThreshold)
            : iq36::VectorCompareStats{};
    const auto q4_shared_separate_vs_cpu =
        q4_selected_shared_measured
            ? iq36::compare_vectors(q4_selected_shared.shared_separate_output,
                                    shared_cpu, kMismatchThreshold)
            : iq36::VectorCompareStats{};
    const auto q4_shared_combined_vs_separate =
        q4_selected_shared_measured
            ? iq36::compare_vectors(q4_selected_shared.shared_combined_output,
                                    q4_selected_shared.shared_separate_output,
                                    kMismatchThreshold)
            : iq36::VectorCompareStats{};
    const auto q4_shared_combined_vs_oracle =
        q4_selected_shared_measured
            ? iq36::compare_vectors(q4_selected_shared.shared_combined_output,
                                    shared_oracle, kMismatchThreshold)
            : iq36::VectorCompareStats{};
    const auto shared_cpu_vs_oracle =
        (q4_selected_shared_measured || q6_selected_shared_measured)
            ? iq36::compare_vectors(shared_cpu, shared_oracle, kMismatchThreshold)
            : iq36::VectorCompareStats{};
    const auto q6_selected_separate_vs_cpu =
        q6_selected_shared_measured
            ? iq36::compare_vectors(q6_selected_shared.selected_separate_output,
                                    cpu, kMismatchThreshold)
            : iq36::VectorCompareStats{};
    const auto q6_selected_combined_vs_separate =
        q6_selected_shared_measured
            ? iq36::compare_vectors(q6_selected_shared.selected_combined_output,
                                    q6_selected_shared.selected_separate_output,
                                    kMismatchThreshold)
            : iq36::VectorCompareStats{};
    const auto q6_selected_combined_vs_oracle =
        q6_selected_shared_measured
            ? iq36::compare_vectors(q6_selected_shared.selected_combined_output,
                                    oracle, kMismatchThreshold)
            : iq36::VectorCompareStats{};
    const auto q6_shared_separate_vs_cpu =
        q6_selected_shared_measured
            ? iq36::compare_vectors(q6_selected_shared.shared_separate_output,
                                    shared_cpu, kMismatchThreshold)
            : iq36::VectorCompareStats{};
    const auto q6_shared_combined_vs_separate =
        q6_selected_shared_measured
            ? iq36::compare_vectors(q6_selected_shared.shared_combined_output,
                                    q6_selected_shared.shared_separate_output,
                                    kMismatchThreshold)
            : iq36::VectorCompareStats{};
    const auto q6_shared_combined_vs_oracle =
        q6_selected_shared_measured
            ? iq36::compare_vectors(q6_selected_shared.shared_combined_output,
                                    shared_oracle, kMismatchThreshold)
            : iq36::VectorCompareStats{};
    const auto cpu_ffn_out = BuildFfnOut(
        cpu_weighted_sum, shared_cpu.empty() ? shared_oracle : shared_cpu,
        shared_tail_gate[0]);
    const auto cpu_layer_output = AddVectorsChecked(attn_residual, cpu_ffn_out);
    const auto oracle_layer_output = AddVectorsChecked(attn_residual, ffn_out_oracle);
    const auto expected_tail_contrib = BuildDownTailContributions(
        cpu_weighted, shared_cpu.empty() ? shared_oracle : shared_cpu,
        shared_tail_gate[0], rows_per_expert);
    const auto q6_tail_contrib_vs_cpu =
        q6_nonatomic_tail_measured
            ? iq36::compare_vectors(q6_nonatomic_tail.contributions,
                                    expected_tail_contrib, kMismatchThreshold)
            : iq36::VectorCompareStats{};
    const auto q6_tail_layer_vs_cpu =
        q6_nonatomic_tail_measured
            ? iq36::compare_vectors(q6_nonatomic_tail.layer_output,
                                    cpu_layer_output, kMismatchThreshold)
            : iq36::VectorCompareStats{};
    const auto q6_tail_layer_vs_oracle =
        q6_nonatomic_tail_measured
            ? iq36::compare_vectors(q6_nonatomic_tail.layer_output,
                                    oracle_layer_output, kMismatchThreshold)
            : iq36::VectorCompareStats{};
    const auto q6_rowgroup_tail_layer_vs_cpu =
        q6_rowgroup_tail_measured
            ? iq36::compare_vectors(q6_rowgroup_tail.layer_output,
                                    cpu_layer_output, kMismatchThreshold)
            : iq36::VectorCompareStats{};
    const auto q6_rowgroup_tail_layer_vs_oracle =
        q6_rowgroup_tail_measured
            ? iq36::compare_vectors(q6_rowgroup_tail.layer_output,
                                    oracle_layer_output, kMismatchThreshold)
            : iq36::VectorCompareStats{};
    const auto q6_rowgroup_tail_layer_vs_nonatomic =
        q6_rowgroup_tail_measured && q6_nonatomic_tail_measured
            ? iq36::compare_vectors(q6_rowgroup_tail.layer_output,
                                    q6_nonatomic_tail.layer_output,
                                    kMismatchThreshold)
            : iq36::VectorCompareStats{};
    const auto cpu_ffn_out_vs_oracle =
        iq36::compare_vectors(cpu_ffn_out, ffn_out_oracle, kMismatchThreshold);
    const auto cpu_layer_vs_oracle =
        iq36::compare_vectors(cpu_layer_output, oracle_layer_output,
                              kMismatchThreshold);
    const bool comparisons_passed =
        ComparePassed(cpu_vs_oracle) && ComparePassed(gpu_vs_cpu) &&
        ComparePassed(gpu_vs_oracle);
    const bool pair2_comparisons_passed =
        !q4_pair2_measured ||
        (ComparePassed(pair2_vs_cpu) && ComparePassed(pair2_vs_oracle) &&
         ComparePassed(pair2_vs_current));
    const bool weighted_comparisons_passed =
        !q4_weighted_measured ||
        (ComparePassed(weighted_cpu_vs_oracle) &&
         ComparePassed(weighted_vs_cpu) &&
         ComparePassed(weighted_vs_oracle) &&
         ComparePassed(weighted_vs_current_weighted));
    const bool weighted_sum_comparisons_passed =
        !q4_weighted_sum_measured ||
        (ComparePassed(weighted_sum_cpu_vs_oracle) &&
         ComparePassed(weighted_sum_vs_cpu) &&
         ComparePassed(weighted_sum_vs_oracle) &&
         ComparePassed(weighted_sum_vs_current) &&
         ComparePassed(weighted_sum_vs_weighted_kernel));
    const bool group8_sum_comparisons_passed =
        !q4_group8_sum_measured ||
        (ComparePassed(weighted_sum_cpu_vs_oracle) &&
         ComparePassed(group8_sum_vs_cpu) &&
         ComparePassed(group8_sum_vs_oracle) &&
         ComparePassed(group8_sum_vs_current) &&
         ComparePassed(group8_sum_vs_weighted_kernel));
    const bool occupancy4_comparisons_passed =
        !q4_occupancy4_measured ||
        (ComparePassed(occupancy4_vs_cpu) &&
         ComparePassed(occupancy4_vs_oracle));
    const bool device_q8_comparisons_passed =
        !q4_device_q8_measured ||
        (ComparePassed(device_q8_vs_cpu) &&
         ComparePassed(device_q8_vs_oracle) &&
         ComparePassed(device_q8_vs_current) &&
         device_q8_planes_vs_cpu.passed);
    const bool swiglu_device_q8_comparisons_passed =
        !q4_swiglu_device_q8_measured ||
        (ComparePassed(swiglu_device_q8_swiglu_vs_cpu) &&
         ComparePassed(swiglu_device_q8_vs_cpu) &&
         ComparePassed(swiglu_device_q8_vs_oracle) &&
         ComparePassed(swiglu_device_q8_vs_current) &&
         swiglu_device_q8_planes_vs_cpu.passed);
    const bool swiglu_f32_input_output_oracle_compatible =
        q4_swiglu_f32_input_measured &&
        ComparePassed(swiglu_f32_input_swiglu_vs_cpu) &&
        ComparePassed(swiglu_f32_input_vs_cpu) &&
        ComparePassed(swiglu_f32_input_vs_oracle) &&
        ComparePassed(swiglu_f32_input_vs_current);
    const bool swiglu_f32_input_comparison_recorded =
        !q4_swiglu_f32_input_measured ||
        (ComparePassed(swiglu_f32_input_swiglu_vs_cpu) &&
         CompareRecorded(swiglu_f32_input_vs_cpu) &&
         CompareRecorded(swiglu_f32_input_vs_oracle) &&
         CompareRecorded(swiglu_f32_input_vs_current));
    const bool q6_selected_shared_comparisons_passed =
        !q6_selected_shared_measured ||
        (ComparePassed(shared_cpu_vs_oracle) &&
         ComparePassed(q6_selected_separate_vs_cpu) &&
         ComparePassed(q6_selected_combined_vs_separate) &&
         ComparePassed(q6_selected_combined_vs_oracle) &&
         ComparePassed(q6_shared_separate_vs_cpu) &&
         ComparePassed(q6_shared_combined_vs_separate) &&
         ComparePassed(q6_shared_combined_vs_oracle));
    const bool q4_selected_shared_comparisons_passed =
        !q4_selected_shared_measured ||
        (ComparePassed(shared_cpu_vs_oracle) &&
         ComparePassed(q4_selected_combined_vs_separate) &&
         ComparePassed(q4_selected_combined_vs_oracle) &&
         ComparePassed(q4_shared_separate_vs_cpu) &&
         ComparePassed(q4_shared_combined_vs_separate) &&
         ComparePassed(q4_shared_combined_vs_oracle));
    const bool q6_nonatomic_tail_comparisons_passed =
        !q6_nonatomic_tail_measured ||
        (ComparePassed(cpu_ffn_out_vs_oracle) &&
         ComparePassed(cpu_layer_vs_oracle) &&
         ComparePassed(q6_tail_contrib_vs_cpu) &&
         ComparePassed(q6_tail_layer_vs_cpu) &&
         ComparePassed(q6_tail_layer_vs_oracle));
    const bool q6_rowgroup_tail_comparisons_passed =
        !q6_rowgroup_tail_measured ||
        (ComparePassed(cpu_ffn_out_vs_oracle) &&
         ComparePassed(cpu_layer_vs_oracle) &&
         ComparePassed(q6_rowgroup_tail_layer_vs_cpu) &&
         ComparePassed(q6_rowgroup_tail_layer_vs_oracle) &&
         ComparePassed(q6_rowgroup_tail_layer_vs_nonatomic));
    const bool counts_ok =
        attn_post_norm.size() == static_cast<std::size_t>(kHiddenSize) &&
        input.size() == kInputValueCount &&
        gate_up.size() == kInputValueCount * 2 &&
        weights.size() == kExpertUsedCount &&
        expert_ids.size() == kExpertUsedCount &&
        oracle.size() == kOutputValueCount &&
        weighted_oracle.size() == kOutputValueCount &&
        shared_oracle.size() == static_cast<std::size_t>(kHiddenSize) &&
        shared_tail_gate.size() == 1 &&
        attn_residual.size() == static_cast<std::size_t>(kHiddenSize) &&
        ffn_out_oracle.size() == static_cast<std::size_t>(kHiddenSize) &&
        cpu.size() == kOutputValueCount &&
        gpu.output.size() == kOutputValueCount &&
        (!q4_pair2_measured || q4_pair2.output.size() == kOutputValueCount) &&
        (!q4_weighted_measured || q4_weighted.output.size() == kOutputValueCount) &&
        (!q4_weighted_sum_measured ||
         q4_weighted_sum.output.size() == static_cast<std::size_t>(rows_per_expert)) &&
        (!q4_group8_sum_measured ||
         q4_group8_sum.output.size() == static_cast<std::size_t>(rows_per_expert)) &&
        (!q4_occupancy4_measured ||
         q4_occupancy4.output.size() ==
             static_cast<std::size_t>(kOutputValueCount * kOccupancyGroupCount)) &&
        (!q4_device_q8_measured ||
         (q4_device_q8.output.size() == kOutputValueCount &&
          q4_device_q8.q8.qs.size() == q8.qs.size() &&
          q4_device_q8.q8.bsums.size() == q8.bsums.size() &&
          q4_device_q8.q8.d.size() == q8.d.size())) &&
        (!q4_swiglu_device_q8_measured ||
         (q4_swiglu_device_q8.swiglu.size() == kInputValueCount &&
          q4_swiglu_device_q8.output.size() == kOutputValueCount &&
          q4_swiglu_device_q8.q8.qs.size() == q8.qs.size() &&
          q4_swiglu_device_q8.q8.bsums.size() == q8.bsums.size() &&
          q4_swiglu_device_q8.q8.d.size() == q8.d.size())) &&
        (!q4_swiglu_f32_input_measured ||
         (q4_swiglu_f32_input.swiglu.size() == kInputValueCount &&
          q4_swiglu_f32_input.output.size() == kOutputValueCount)) &&
        (!q4_selected_shared_measured ||
         (shared_gate.size() == static_cast<std::size_t>(kIntermediateSize) &&
          shared_up.size() == static_cast<std::size_t>(kIntermediateSize) &&
          shared_swiglu.size() == static_cast<std::size_t>(kIntermediateSize) &&
          shared_cpu.size() == static_cast<std::size_t>(kHiddenSize) &&
          q4_selected_shared.selected_separate_output.size() == kOutputValueCount &&
          q4_selected_shared.selected_combined_output.size() == kOutputValueCount &&
          q4_selected_shared.shared_separate_output.size() ==
              static_cast<std::size_t>(kHiddenSize) &&
          q4_selected_shared.shared_combined_output.size() ==
              static_cast<std::size_t>(kHiddenSize))) &&
        (!q6_selected_shared_measured ||
         (shared_gate.size() == static_cast<std::size_t>(kIntermediateSize) &&
          shared_up.size() == static_cast<std::size_t>(kIntermediateSize) &&
          shared_swiglu.size() == static_cast<std::size_t>(kIntermediateSize) &&
          shared_cpu.size() == static_cast<std::size_t>(kHiddenSize) &&
          q6_selected_shared.selected_separate_output.size() == kOutputValueCount &&
          q6_selected_shared.selected_combined_output.size() == kOutputValueCount &&
          q6_selected_shared.shared_separate_output.size() ==
              static_cast<std::size_t>(kHiddenSize) &&
          q6_selected_shared.shared_combined_output.size() ==
              static_cast<std::size_t>(kHiddenSize))) &&
        (!q6_nonatomic_tail_measured ||
         (q6_nonatomic_tail.contributions.size() ==
              static_cast<std::size_t>(kHiddenSize * 9) &&
          q6_nonatomic_tail.layer_output.size() ==
              static_cast<std::size_t>(kHiddenSize))) &&
        (!q6_rowgroup_tail_measured ||
         q6_rowgroup_tail.layer_output.size() ==
             static_cast<std::size_t>(kHiddenSize)) &&
        repeated_cpu.size() == static_cast<std::size_t>(kOutputValueCount * kOccupancyGroupCount) &&
        repeated_oracle.size() == static_cast<std::size_t>(kOutputValueCount * kOccupancyGroupCount) &&
        cpu_weighted_sum.size() == static_cast<std::size_t>(rows_per_expert) &&
        oracle_weighted_sum.size() == static_cast<std::size_t>(rows_per_expert) &&
        gpu_weighted_sum.size() == static_cast<std::size_t>(rows_per_expert) &&
        selected_rows == kOutputValueCount &&
        blocks_per_row == 2;
    const bool timings_positive = gpu.timing.min_us > 0.0;
    const bool pair2_timing_positive =
        !q4_pair2_measured || q4_pair2.timing.min_us > 0.0;
    const bool weighted_timing_positive =
        !q4_weighted_measured || q4_weighted.timing.min_us > 0.0;
    const bool weighted_sum_timing_positive =
        !q4_weighted_sum_measured || q4_weighted_sum.timing.min_us > 0.0;
    const bool group8_sum_timing_positive =
        !q4_group8_sum_measured || q4_group8_sum.timing.min_us > 0.0;
    const bool occupancy4_timing_positive =
        !q4_occupancy4_measured || q4_occupancy4.timing.min_us > 0.0;
    const bool device_q8_timing_positive =
        !q4_device_q8_measured ||
        (q4_device_q8.q8_quantize_timing.min_us > 0.0 &&
         q4_device_q8.down_timing.min_us > 0.0 &&
         q4_device_q8.shell_sum_min_us > 0.0);
    const bool swiglu_device_q8_timing_positive =
        !q4_swiglu_device_q8_measured ||
        (q4_swiglu_device_q8.swiglu_timing.min_us > 0.0 &&
         q4_swiglu_device_q8.q8_quantize_timing.min_us > 0.0 &&
         q4_swiglu_device_q8.down_timing.min_us > 0.0 &&
         q4_swiglu_device_q8.shell_sum_min_us > 0.0);
    const bool swiglu_f32_input_timing_positive =
        !q4_swiglu_f32_input_measured ||
        (q4_swiglu_f32_input.swiglu_timing.min_us > 0.0 &&
         q4_swiglu_f32_input.down_timing.min_us > 0.0 &&
         q4_swiglu_f32_input.shell_sum_min_us > 0.0);
    const bool q4_selected_shared_timing_positive =
        !q4_selected_shared_measured ||
        (q4_selected_shared.selected_timing.min_us > 0.0 &&
         q4_selected_shared.shared_timing.min_us > 0.0 &&
         q4_selected_shared.combined_timing.min_us > 0.0);
    const bool q6_selected_shared_timing_positive =
        !q6_selected_shared_measured ||
        (q6_selected_shared.selected_timing.min_us > 0.0 &&
         q6_selected_shared.shared_timing.min_us > 0.0 &&
         q6_selected_shared.combined_timing.min_us > 0.0);
    const bool q6_nonatomic_tail_timing_positive =
        !q6_nonatomic_tail_measured ||
        (q6_nonatomic_tail.contribution_timing.min_us > 0.0 &&
         q6_nonatomic_tail.reduce_timing.min_us > 0.0 &&
         q6_nonatomic_tail.shell_sum_min_us > 0.0 &&
         q6_nonatomic_tail.contribution_timing.global_work_items ==
             static_cast<std::uint64_t>(kHiddenSize * 9) &&
         q6_nonatomic_tail.reduce_timing.global_work_items ==
             static_cast<std::uint64_t>(kHiddenSize));
    const bool q6_rowgroup_tail_timing_positive =
        !q6_rowgroup_tail_measured ||
        (q6_rowgroup_tail.timing.min_us > 0.0 &&
         q6_rowgroup_tail.timing.global_work_items ==
             static_cast<std::uint64_t>(kHiddenSize *
                                        kQ6RowgroupDownTailLocal) &&
         q6_rowgroup_tail.local_work_items == kQ6RowgroupDownTailLocal);
    const double pair2_speedup_vs_current =
        q4_pair2_measured && q4_pair2.timing.min_us > 0.0
            ? gpu.timing.min_us / q4_pair2.timing.min_us
            : 0.0;
    const double weighted_speedup_vs_current =
        q4_weighted_measured && q4_weighted.timing.min_us > 0.0
            ? gpu.timing.min_us / q4_weighted.timing.min_us
            : 0.0;
    const double weighted_sum_speedup_vs_current =
        q4_weighted_sum_measured && q4_weighted_sum.timing.min_us > 0.0
            ? gpu.timing.min_us / q4_weighted_sum.timing.min_us
            : 0.0;
    const double weighted_sum_speedup_vs_weighted =
        q4_weighted_sum_measured && q4_weighted_measured &&
                q4_weighted_sum.timing.min_us > 0.0
            ? q4_weighted.timing.min_us / q4_weighted_sum.timing.min_us
            : 0.0;
    const double group8_sum_speedup_vs_current =
        q4_group8_sum_measured && q4_group8_sum.timing.min_us > 0.0
            ? gpu.timing.min_us / q4_group8_sum.timing.min_us
            : 0.0;
    const double group8_sum_speedup_vs_weighted =
        q4_group8_sum_measured && q4_weighted_measured &&
                q4_group8_sum.timing.min_us > 0.0
            ? q4_weighted.timing.min_us / q4_group8_sum.timing.min_us
            : 0.0;
    const double group8_sum_speedup_vs_serial_sum =
        q4_group8_sum_measured && q4_weighted_sum_measured &&
                q4_group8_sum.timing.min_us > 0.0
            ? q4_weighted_sum.timing.min_us / q4_group8_sum.timing.min_us
            : 0.0;
    const double occupancy4_scaled_speedup_vs_current =
        q4_occupancy4_measured && q4_occupancy4.timing.min_us > 0.0
            ? (gpu.timing.min_us * static_cast<double>(kOccupancyGroupCount)) /
                  q4_occupancy4.timing.min_us
            : 0.0;
    const double occupancy4_effective_single_group_us =
        q4_occupancy4_measured
            ? q4_occupancy4.timing.min_us /
                  static_cast<double>(kOccupancyGroupCount)
            : 0.0;
    const double device_q8_down_speedup_vs_current =
        q4_device_q8_measured && q4_device_q8.down_timing.min_us > 0.0
            ? gpu.timing.min_us / q4_device_q8.down_timing.min_us
            : 0.0;
    const double device_q8_shell_ratio_vs_current =
        q4_device_q8_measured && gpu.timing.min_us > 0.0
            ? q4_device_q8.shell_sum_min_us / gpu.timing.min_us
            : 0.0;
    const double swiglu_device_q8_down_speedup_vs_current =
        q4_swiglu_device_q8_measured && q4_swiglu_device_q8.down_timing.min_us > 0.0
            ? gpu.timing.min_us / q4_swiglu_device_q8.down_timing.min_us
            : 0.0;
    const double swiglu_device_q8_shell_ratio_vs_current =
        q4_swiglu_device_q8_measured && gpu.timing.min_us > 0.0
            ? q4_swiglu_device_q8.shell_sum_min_us / gpu.timing.min_us
            : 0.0;
    const double swiglu_f32_input_down_speedup_vs_current =
        q4_swiglu_f32_input_measured && q4_swiglu_f32_input.down_timing.min_us > 0.0
            ? gpu.timing.min_us / q4_swiglu_f32_input.down_timing.min_us
            : 0.0;
    const double swiglu_f32_input_shell_ratio_vs_current =
        q4_swiglu_f32_input_measured && gpu.timing.min_us > 0.0
            ? q4_swiglu_f32_input.shell_sum_min_us / gpu.timing.min_us
            : 0.0;
    const double q6_selected_shared_separate_sum_us =
        q6_selected_shared_measured
            ? q6_selected_shared.selected_timing.min_us +
                  q6_selected_shared.shared_timing.min_us
            : 0.0;
    const double q4_selected_shared_separate_sum_us =
        q4_selected_shared_measured
            ? q4_selected_shared.selected_timing.min_us +
                  q4_selected_shared.shared_timing.min_us
            : 0.0;
    const double q6_selected_shared_combined_speedup_vs_separate =
        q6_selected_shared_measured &&
                q6_selected_shared.combined_timing.min_us > 0.0
            ? q6_selected_shared_separate_sum_us /
                  q6_selected_shared.combined_timing.min_us
            : 0.0;
    const double q6_nonatomic_tail_ratio_vs_combined_down =
        q6_nonatomic_tail_measured &&
                q6_selected_shared_measured &&
                q6_selected_shared.combined_timing.min_us > 0.0
            ? q6_nonatomic_tail.shell_sum_min_us /
                  q6_selected_shared.combined_timing.min_us
            : 0.0;
    const double q6_rowgroup_tail_ratio_vs_combined_down =
        q6_rowgroup_tail_measured &&
                q6_selected_shared_measured &&
                q6_selected_shared.combined_timing.min_us > 0.0
            ? q6_rowgroup_tail.timing.min_us /
                  q6_selected_shared.combined_timing.min_us
            : 0.0;
    const double q6_rowgroup_tail_ratio_vs_nonatomic =
        q6_rowgroup_tail_measured &&
                q6_nonatomic_tail_measured &&
                q6_nonatomic_tail.shell_sum_min_us > 0.0
            ? q6_rowgroup_tail.timing.min_us /
                  q6_nonatomic_tail.shell_sum_min_us
            : 0.0;
    const double q4_selected_shared_combined_speedup_vs_separate =
        q4_selected_shared_measured &&
                q4_selected_shared.combined_timing.min_us > 0.0
            ? q4_selected_shared_separate_sum_us /
                  q4_selected_shared.combined_timing.min_us
            : 0.0;
    const bool pair2_material_component_speedup =
        q4_pair2_measured && pair2_speedup_vs_current >= 1.10;
    const bool weighted_boundary_reopen_candidate =
        q4_weighted_measured && weighted_speedup_vs_current >= 0.95;
    const bool weighted_sum_boundary_reopen_candidate =
        q4_weighted_sum_measured && weighted_sum_comparisons_passed &&
        weighted_sum_speedup_vs_current >= 0.95;
    const bool group8_sum_boundary_reopen_candidate =
        q4_group8_sum_measured && group8_sum_comparisons_passed &&
        group8_sum_speedup_vs_current >= 0.95;
    const bool selected_shape_occupancy_bound =
        q4_occupancy4_measured && occupancy4_comparisons_passed &&
        occupancy4_scaled_speedup_vs_current >= 1.20;
    const bool device_q8_boundary_reopen_candidate =
        q4_device_q8_measured && device_q8_comparisons_passed &&
        device_q8_down_speedup_vs_current >= 0.95 &&
        device_q8_shell_ratio_vs_current <= 1.20;
    const bool swiglu_device_q8_boundary_reopen_candidate =
        q4_swiglu_device_q8_measured && swiglu_device_q8_comparisons_passed &&
        swiglu_device_q8_down_speedup_vs_current >= 0.95 &&
        swiglu_device_q8_shell_ratio_vs_current <= 1.20;
    const bool swiglu_f32_input_boundary_reopen_candidate =
        q4_swiglu_f32_input_measured &&
        swiglu_f32_input_output_oracle_compatible &&
        swiglu_f32_input_shell_ratio_vs_current <= 1.00;
    const bool q6_selected_shared_boundary_reopen_candidate =
        q6_selected_shared_measured &&
        q6_selected_shared_comparisons_passed &&
        q6_selected_shared_combined_speedup_vs_separate >= 1.10;
    const bool q6_nonatomic_tail_boundary_reopen_candidate =
        q6_nonatomic_tail_measured &&
        q6_nonatomic_tail_comparisons_passed &&
        q6_nonatomic_tail_timing_positive &&
        q6_nonatomic_tail_ratio_vs_combined_down <= 1.15;
    const bool q6_rowgroup_tail_boundary_reopen_candidate =
        q6_rowgroup_tail_measured &&
        q6_rowgroup_tail_comparisons_passed &&
        q6_rowgroup_tail_timing_positive &&
        q6_rowgroup_tail.timing.min_us <= kQ6RowgroupDownTailTargetUs;
    const bool q4_selected_shared_boundary_reopen_candidate =
        q4_selected_shared_measured &&
        q4_selected_shared_comparisons_passed &&
        q4_selected_shared_combined_speedup_vs_separate >= 1.10;
    const bool checks_passed =
        load_map.ready &&
        tensor_shape_ok &&
        shared_down_tensor_shape_ok &&
        shared_input_gate_tensor_shape_ok &&
        counts_ok &&
        gpu.device_name.find(args.device_substring) != std::string::npos &&
        comparisons_passed &&
        pair2_comparisons_passed &&
        weighted_comparisons_passed &&
        weighted_sum_comparisons_passed &&
        group8_sum_comparisons_passed &&
        occupancy4_comparisons_passed &&
        device_q8_comparisons_passed &&
        swiglu_device_q8_comparisons_passed &&
        swiglu_f32_input_comparison_recorded &&
        q4_selected_shared_comparisons_passed &&
        q6_selected_shared_comparisons_passed &&
        q6_nonatomic_tail_comparisons_passed &&
        q6_rowgroup_tail_comparisons_passed &&
        timings_positive &&
        pair2_timing_positive &&
        weighted_timing_positive &&
        weighted_sum_timing_positive &&
        group8_sum_timing_positive &&
        occupancy4_timing_positive &&
        device_q8_timing_positive &&
        swiglu_device_q8_timing_positive &&
        swiglu_f32_input_timing_positive &&
        q4_selected_shared_timing_positive &&
        q6_selected_shared_timing_positive &&
        q6_nonatomic_tail_timing_positive &&
        q6_rowgroup_tail_timing_positive;

    std::cout << std::setprecision(10);
    std::cout << "{";
    std::cout << "\"schema_version\":\"intel-qwen36-gpu-selected-down-probe-v9\",";
    std::cout << "\"model_path\":\"" << JsonEscape(args.model_path) << "\",";
    std::cout << "\"layer\":" << args.layer << ",";
    std::cout << "\"source_token_position\":" << kSourceTokenPosition << ",";
    std::cout << "\"tensor_name\":\"" << JsonEscape(tensor->name) << "\",";
    std::cout << "\"tensor_type\":\"" << iq36::ggml_type_name(tensor->type) << "\",";
    std::cout << "\"tensor_shape_ok\":" << (tensor_shape_ok ? "true" : "false") << ",";
    std::cout << "\"cols\":" << cols << ",";
    std::cout << "\"rows_per_expert\":" << rows_per_expert << ",";
    std::cout << "\"selected_rows\":" << selected_rows << ",";
    std::cout << "\"blocks_per_row\":" << blocks_per_row << ",";
    std::cout << "\"row_nbytes\":" << row_nbytes << ",";
    std::cout << "\"tensor_nbytes\":" << tensor->nbytes << ",";
    std::cout << "\"selected_raw_bytes\":" << selected_raw.size() << ",";
    std::cout << "\"q8_qs_bytes\":" << q8.qs.size() * sizeof(std::int8_t) << ",";
    std::cout << "\"q8_bsums_bytes\":" << q8.bsums.size() * sizeof(std::int16_t) << ",";
    std::cout << "\"q8_d_bytes\":" << q8.d.size() * sizeof(float) << ",";
    std::cout << "\"router_weight_bytes\":" << weights.size() * sizeof(float) << ",";
    std::cout << "\"expert_ids\":";
    WriteI32Vector(expert_ids);
    std::cout << ",";
    std::cout << "\"platform_name\":\"" << JsonEscape(gpu.platform_name) << "\",";
    std::cout << "\"device_name\":\"" << JsonEscape(gpu.device_name) << "\",";
    std::cout << "\"program_build_ms\":" << gpu.program_build_ms << ",";
    std::cout << "\"build_log\":\"" << JsonEscape(gpu.build_log) << "\",";
    std::cout << "\"repeat\":" << args.repeat << ",";
    std::cout << "\"current_kernel\":\""
              << (tensor->type == 12
                      ? "q4k_x8_matvec_rowlane_expert8_multiq8"
                      : "q6k_selected_down_matvec_row")
              << "\",";
    std::cout << "\"candidate_pair2_kernel\":\""
              << (q4_pair2_measured
                      ? "q4k_x8_matvec_rowlane_expert8_multiq8_pair2"
                      : "")
              << "\",";
    std::cout << "\"candidate_pair2_measured\":"
              << (q4_pair2_measured ? "true" : "false") << ",";
    std::cout << "\"candidate_pair2_speedup_vs_current\":"
              << pair2_speedup_vs_current << ",";
    std::cout << "\"candidate_pair2_material_component_speedup\":"
              << (pair2_material_component_speedup ? "true" : "false") << ",";
    std::cout << "\"candidate_weighted_kernel\":\""
              << (q4_weighted_measured
                      ? "q4k_x8_matvec_rowlane_expert8_multiq8_weighted"
                      : "")
              << "\",";
    std::cout << "\"candidate_weighted_measured\":"
              << (q4_weighted_measured ? "true" : "false") << ",";
    std::cout << "\"candidate_weighted_speedup_vs_current\":"
              << weighted_speedup_vs_current << ",";
    std::cout << "\"candidate_weighted_boundary_reopen_candidate\":"
              << (weighted_boundary_reopen_candidate ? "true" : "false") << ",";
    std::cout << "\"candidate_weighted_sum_kernel\":\""
              << (q4_weighted_sum_measured
                      ? "q4k_x8_matvec_rowlane_expert8_multiq8_weighted_sum"
                      : "")
              << "\",";
    std::cout << "\"candidate_weighted_sum_measured\":"
              << (q4_weighted_sum_measured ? "true" : "false") << ",";
    std::cout << "\"candidate_weighted_sum_speedup_vs_current\":"
              << weighted_sum_speedup_vs_current << ",";
    std::cout << "\"candidate_weighted_sum_speedup_vs_weighted\":"
              << weighted_sum_speedup_vs_weighted << ",";
    std::cout << "\"candidate_weighted_sum_boundary_reopen_candidate\":"
              << (weighted_sum_boundary_reopen_candidate ? "true" : "false") << ",";
    std::cout << "\"candidate_group8_sum_kernel\":\""
              << (q4_group8_sum_measured
                      ? "q4k_x8_matvec_rowlane_expert8_multiq8_group8_weighted_sum"
                      : "")
              << "\",";
    std::cout << "\"candidate_group8_sum_measured\":"
              << (q4_group8_sum_measured ? "true" : "false") << ",";
    std::cout << "\"candidate_group8_sum_local_items\":"
              << (q4_group8_sum_measured ? 32 : 0) << ",";
    std::cout << "\"candidate_group8_sum_speedup_vs_current\":"
              << group8_sum_speedup_vs_current << ",";
    std::cout << "\"candidate_group8_sum_speedup_vs_weighted\":"
              << group8_sum_speedup_vs_weighted << ",";
    std::cout << "\"candidate_group8_sum_speedup_vs_serial_sum\":"
              << group8_sum_speedup_vs_serial_sum << ",";
    std::cout << "\"candidate_group8_sum_boundary_reopen_candidate\":"
              << (group8_sum_boundary_reopen_candidate ? "true" : "false") << ",";
    std::cout << "\"candidate_occupancy_kernel\":\""
              << (q4_occupancy4_measured
                      ? "q4k_x8_matvec_rowlane_expert8_multiq8_occupancy4"
                      : "")
              << "\",";
    std::cout << "\"candidate_occupancy_measured\":"
              << (q4_occupancy4_measured ? "true" : "false") << ",";
    std::cout << "\"candidate_occupancy_group_count\":"
              << (q4_occupancy4_measured ? kOccupancyGroupCount : 0) << ",";
    std::cout << "\"candidate_occupancy_scaled_speedup_vs_current\":"
              << occupancy4_scaled_speedup_vs_current << ",";
    std::cout << "\"candidate_occupancy_effective_single_group_us\":"
              << occupancy4_effective_single_group_us << ",";
    std::cout << "\"selected_shape_occupancy_bound\":"
              << (selected_shape_occupancy_bound ? "true" : "false") << ",";
    std::cout << "\"candidate_device_q8_kernel\":\""
              << (q4_device_q8_measured
                      ? "q8k_quantize_f32_blocks_with_bsums+q4k_x8_matvec_rowlane_expert8_multiq8"
                      : "")
              << "\",";
    std::cout << "\"candidate_device_q8_measured\":"
              << (q4_device_q8_measured ? "true" : "false") << ",";
    std::cout << "\"candidate_device_q8_down_speedup_vs_current\":"
              << device_q8_down_speedup_vs_current << ",";
    std::cout << "\"candidate_device_q8_shell_ratio_vs_current\":"
              << device_q8_shell_ratio_vs_current << ",";
    std::cout << "\"candidate_device_q8_boundary_reopen_candidate\":"
              << (device_q8_boundary_reopen_candidate ? "true" : "false") << ",";
    std::cout << "\"candidate_swiglu_device_q8_kernel\":\""
              << (q4_swiglu_device_q8_measured
                      ? "ffn_moe_swiglu_f32+q8k_quantize_f32_blocks_with_bsums+q4k_x8_matvec_rowlane_expert8_multiq8"
                      : "")
              << "\",";
    std::cout << "\"candidate_swiglu_device_q8_measured\":"
              << (q4_swiglu_device_q8_measured ? "true" : "false") << ",";
    std::cout << "\"candidate_swiglu_device_q8_down_speedup_vs_current\":"
              << swiglu_device_q8_down_speedup_vs_current << ",";
    std::cout << "\"candidate_swiglu_device_q8_shell_ratio_vs_current\":"
              << swiglu_device_q8_shell_ratio_vs_current << ",";
    std::cout << "\"candidate_swiglu_device_q8_boundary_reopen_candidate\":"
              << (swiglu_device_q8_boundary_reopen_candidate ? "true" : "false") << ",";
    std::cout << "\"candidate_swiglu_f32_input_kernel\":\""
              << (q4_swiglu_f32_input_measured
                      ? "ffn_moe_swiglu_f32+q4k_x8_matvec_rowlane_expert8_f32input"
                      : "")
              << "\",";
    std::cout << "\"candidate_swiglu_f32_input_measured\":"
              << (q4_swiglu_f32_input_measured ? "true" : "false") << ",";
    std::cout << "\"candidate_swiglu_f32_input_down_speedup_vs_current\":"
              << swiglu_f32_input_down_speedup_vs_current << ",";
    std::cout << "\"candidate_swiglu_f32_input_shell_ratio_vs_current\":"
              << swiglu_f32_input_shell_ratio_vs_current << ",";
    std::cout << "\"candidate_swiglu_f32_input_output_oracle_compatible\":"
              << (swiglu_f32_input_output_oracle_compatible ? "true" : "false") << ",";
    std::cout << "\"candidate_swiglu_f32_input_boundary_reopen_candidate\":"
              << (swiglu_f32_input_boundary_reopen_candidate ? "true" : "false") << ",";
    std::cout << "\"candidate_q4_selected_shared_kernel\":\""
              << (q4_selected_shared_measured
                      ? "q4k_x8_selected_down_expert8_plus_shared_q4"
                      : "")
              << "\",";
    std::cout << "\"candidate_q4_selected_shared_measured\":"
              << (q4_selected_shared_measured ? "true" : "false") << ",";
    std::cout << "\"candidate_q4_selected_shared_separate_sum_us\":"
              << q4_selected_shared_separate_sum_us << ",";
    std::cout << "\"candidate_q4_selected_shared_combined_speedup_vs_separate\":"
              << q4_selected_shared_combined_speedup_vs_separate << ",";
    std::cout << "\"candidate_q4_selected_shared_boundary_reopen_candidate\":"
              << (q4_selected_shared_boundary_reopen_candidate ? "true" : "false") << ",";
    std::cout << "\"candidate_q6_selected_shared_kernel\":\""
              << (q6_selected_shared_measured
                      ? "q6k_selected_down_matvec_rowstripe_expert8_plus_shared_raw"
                      : "")
              << "\",";
    std::cout << "\"candidate_q6_selected_shared_measured\":"
              << (q6_selected_shared_measured ? "true" : "false") << ",";
    std::cout << "\"candidate_q6_selected_shared_separate_sum_us\":"
              << q6_selected_shared_separate_sum_us << ",";
    std::cout << "\"candidate_q6_selected_shared_combined_speedup_vs_separate\":"
              << q6_selected_shared_combined_speedup_vs_separate << ",";
    std::cout << "\"candidate_q6_selected_shared_boundary_reopen_candidate\":"
              << (q6_selected_shared_boundary_reopen_candidate ? "true" : "false") << ",";
    std::cout << "\"candidate_q6_nonatomic_down_tail_kernel\":\""
              << (q6_nonatomic_tail_measured
                      ? "q6k_selected_down_matvec_rowstripe_expert8_plus_shared_tail_contrib_raw+ffn_tail_reduce9_contrib_f32"
                      : "")
              << "\",";
    std::cout << "\"candidate_q6_nonatomic_down_tail_measured\":"
              << (q6_nonatomic_tail_measured ? "true" : "false") << ",";
    std::cout << "\"candidate_q6_nonatomic_down_tail_ratio_vs_combined_down\":"
              << q6_nonatomic_tail_ratio_vs_combined_down << ",";
    std::cout << "\"candidate_q6_nonatomic_down_tail_preserves_contributor_parallelism\":"
              << ((q6_nonatomic_tail_measured && q6_nonatomic_tail_timing_positive) ? "true" : "false") << ",";
    std::cout << "\"candidate_q6_nonatomic_down_tail_boundary_reopen_candidate\":"
              << (q6_nonatomic_tail_boundary_reopen_candidate ? "true" : "false") << ",";
    std::cout << "\"candidate_q6_rowgroup_down_tail_kernel\":\""
              << (q6_rowgroup_tail_measured
                      ? "q6k_selected_down_matvec_rowstripe_expert8_plus_shared_tail_rowgroup_reduce_raw"
                      : "")
              << "\",";
    std::cout << "\"candidate_q6_rowgroup_down_tail_measured\":"
              << (q6_rowgroup_tail_measured ? "true" : "false") << ",";
    std::cout << "\"candidate_q6_rowgroup_down_tail_ratio_vs_combined_down\":"
              << q6_rowgroup_tail_ratio_vs_combined_down << ",";
    std::cout << "\"candidate_q6_rowgroup_down_tail_ratio_vs_nonatomic\":"
              << q6_rowgroup_tail_ratio_vs_nonatomic << ",";
    std::cout << "\"candidate_q6_rowgroup_down_tail_component_shell_target_us\":"
              << kQ6RowgroupDownTailTargetUs << ",";
    std::cout << "\"candidate_q6_rowgroup_down_tail_preserves_contributor_parallelism\":"
              << ((q6_rowgroup_tail_measured && q6_rowgroup_tail_timing_positive) ? "true" : "false") << ",";
    std::cout << "\"candidate_q6_rowgroup_down_tail_boundary_reopen_candidate\":"
              << (q6_rowgroup_tail_boundary_reopen_candidate ? "true" : "false") << ",";
    std::cout << "\"timings\":{";
    std::cout << "\"selected_down_gpu_kernel_min_us\":" << gpu.timing.min_us << ",";
    std::cout << "\"selected_down_gpu_kernel_mean_us\":" << gpu.timing.mean_us << ",";
    std::cout << "\"selected_down_gpu_effective_raw_gb_s\":" << gpu.timing.effective_raw_gb_s << ",";
    std::cout << "\"selected_down_gpu_effective_io_gb_s\":" << gpu.timing.effective_io_gb_s << ",";
    std::cout << "\"global_work_items\":" << gpu.timing.global_work_items << ",";
    std::cout << "\"kernel_launches\":" << gpu.timing.kernel_launches << ",";
    std::cout << "\"candidate_pair2_kernel_min_us\":"
              << (q4_pair2_measured ? q4_pair2.timing.min_us : 0.0) << ",";
    std::cout << "\"candidate_pair2_kernel_mean_us\":"
              << (q4_pair2_measured ? q4_pair2.timing.mean_us : 0.0) << ",";
    std::cout << "\"candidate_pair2_effective_raw_gb_s\":"
              << (q4_pair2_measured ? q4_pair2.timing.effective_raw_gb_s : 0.0) << ",";
    std::cout << "\"candidate_pair2_effective_io_gb_s\":"
              << (q4_pair2_measured ? q4_pair2.timing.effective_io_gb_s : 0.0) << ",";
    std::cout << "\"candidate_pair2_global_work_items\":"
              << (q4_pair2_measured ? q4_pair2.timing.global_work_items : 0) << ",";
    std::cout << "\"candidate_pair2_kernel_launches\":"
              << (q4_pair2_measured ? q4_pair2.timing.kernel_launches : 0) << ",";
    std::cout << "\"candidate_weighted_kernel_min_us\":"
              << (q4_weighted_measured ? q4_weighted.timing.min_us : 0.0) << ",";
    std::cout << "\"candidate_weighted_kernel_mean_us\":"
              << (q4_weighted_measured ? q4_weighted.timing.mean_us : 0.0) << ",";
    std::cout << "\"candidate_weighted_effective_raw_gb_s\":"
              << (q4_weighted_measured ? q4_weighted.timing.effective_raw_gb_s : 0.0) << ",";
    std::cout << "\"candidate_weighted_effective_io_gb_s\":"
              << (q4_weighted_measured ? q4_weighted.timing.effective_io_gb_s : 0.0) << ",";
    std::cout << "\"candidate_weighted_global_work_items\":"
              << (q4_weighted_measured ? q4_weighted.timing.global_work_items : 0) << ",";
    std::cout << "\"candidate_weighted_kernel_launches\":"
              << (q4_weighted_measured ? q4_weighted.timing.kernel_launches : 0) << ",";
    std::cout << "\"candidate_weighted_sum_kernel_min_us\":"
              << (q4_weighted_sum_measured ? q4_weighted_sum.timing.min_us : 0.0) << ",";
    std::cout << "\"candidate_weighted_sum_kernel_mean_us\":"
              << (q4_weighted_sum_measured ? q4_weighted_sum.timing.mean_us : 0.0) << ",";
    std::cout << "\"candidate_weighted_sum_effective_raw_gb_s\":"
              << (q4_weighted_sum_measured ? q4_weighted_sum.timing.effective_raw_gb_s : 0.0) << ",";
    std::cout << "\"candidate_weighted_sum_effective_io_gb_s\":"
              << (q4_weighted_sum_measured ? q4_weighted_sum.timing.effective_io_gb_s : 0.0) << ",";
    std::cout << "\"candidate_weighted_sum_global_work_items\":"
              << (q4_weighted_sum_measured ? q4_weighted_sum.timing.global_work_items : 0) << ",";
    std::cout << "\"candidate_weighted_sum_kernel_launches\":"
              << (q4_weighted_sum_measured ? q4_weighted_sum.timing.kernel_launches : 0) << ",";
    std::cout << "\"candidate_group8_sum_kernel_min_us\":"
              << (q4_group8_sum_measured ? q4_group8_sum.timing.min_us : 0.0) << ",";
    std::cout << "\"candidate_group8_sum_kernel_mean_us\":"
              << (q4_group8_sum_measured ? q4_group8_sum.timing.mean_us : 0.0) << ",";
    std::cout << "\"candidate_group8_sum_effective_raw_gb_s\":"
              << (q4_group8_sum_measured ? q4_group8_sum.timing.effective_raw_gb_s : 0.0) << ",";
    std::cout << "\"candidate_group8_sum_effective_io_gb_s\":"
              << (q4_group8_sum_measured ? q4_group8_sum.timing.effective_io_gb_s : 0.0) << ",";
    std::cout << "\"candidate_group8_sum_global_work_items\":"
              << (q4_group8_sum_measured ? q4_group8_sum.timing.global_work_items : 0) << ",";
    std::cout << "\"candidate_group8_sum_kernel_launches\":"
              << (q4_group8_sum_measured ? q4_group8_sum.timing.kernel_launches : 0) << ",";
    std::cout << "\"candidate_occupancy_kernel_min_us\":"
              << (q4_occupancy4_measured ? q4_occupancy4.timing.min_us : 0.0) << ",";
    std::cout << "\"candidate_occupancy_kernel_mean_us\":"
              << (q4_occupancy4_measured ? q4_occupancy4.timing.mean_us : 0.0) << ",";
    std::cout << "\"candidate_occupancy_effective_raw_gb_s\":"
              << (q4_occupancy4_measured ? q4_occupancy4.timing.effective_raw_gb_s : 0.0) << ",";
    std::cout << "\"candidate_occupancy_effective_io_gb_s\":"
              << (q4_occupancy4_measured ? q4_occupancy4.timing.effective_io_gb_s : 0.0) << ",";
    std::cout << "\"candidate_occupancy_global_work_items\":"
              << (q4_occupancy4_measured ? q4_occupancy4.timing.global_work_items : 0) << ",";
    std::cout << "\"candidate_occupancy_kernel_launches\":"
              << (q4_occupancy4_measured ? q4_occupancy4.timing.kernel_launches : 0) << ",";
    std::cout << "\"candidate_device_q8_quantize_min_us\":"
              << (q4_device_q8_measured ? q4_device_q8.q8_quantize_timing.min_us : 0.0) << ",";
    std::cout << "\"candidate_device_q8_quantize_mean_us\":"
              << (q4_device_q8_measured ? q4_device_q8.q8_quantize_timing.mean_us : 0.0) << ",";
    std::cout << "\"candidate_device_q8_quantize_effective_raw_gb_s\":"
              << (q4_device_q8_measured ? q4_device_q8.q8_quantize_timing.effective_raw_gb_s : 0.0) << ",";
    std::cout << "\"candidate_device_q8_quantize_effective_io_gb_s\":"
              << (q4_device_q8_measured ? q4_device_q8.q8_quantize_timing.effective_io_gb_s : 0.0) << ",";
    std::cout << "\"candidate_device_q8_quantize_global_work_items\":"
              << (q4_device_q8_measured ? q4_device_q8.q8_quantize_timing.global_work_items : 0) << ",";
    std::cout << "\"candidate_device_q8_down_kernel_min_us\":"
              << (q4_device_q8_measured ? q4_device_q8.down_timing.min_us : 0.0) << ",";
    std::cout << "\"candidate_device_q8_down_kernel_mean_us\":"
              << (q4_device_q8_measured ? q4_device_q8.down_timing.mean_us : 0.0) << ",";
    std::cout << "\"candidate_device_q8_down_effective_raw_gb_s\":"
              << (q4_device_q8_measured ? q4_device_q8.down_timing.effective_raw_gb_s : 0.0) << ",";
    std::cout << "\"candidate_device_q8_down_effective_io_gb_s\":"
              << (q4_device_q8_measured ? q4_device_q8.down_timing.effective_io_gb_s : 0.0) << ",";
    std::cout << "\"candidate_device_q8_down_global_work_items\":"
              << (q4_device_q8_measured ? q4_device_q8.down_timing.global_work_items : 0) << ",";
    std::cout << "\"candidate_device_q8_shell_sum_min_us\":"
              << (q4_device_q8_measured ? q4_device_q8.shell_sum_min_us : 0.0) << ",";
    std::cout << "\"candidate_device_q8_shell_sum_mean_us\":"
              << (q4_device_q8_measured ? q4_device_q8.shell_sum_mean_us : 0.0) << ",";
    std::cout << "\"candidate_swiglu_device_q8_swiglu_min_us\":"
              << (q4_swiglu_device_q8_measured ? q4_swiglu_device_q8.swiglu_timing.min_us : 0.0) << ",";
    std::cout << "\"candidate_swiglu_device_q8_swiglu_mean_us\":"
              << (q4_swiglu_device_q8_measured ? q4_swiglu_device_q8.swiglu_timing.mean_us : 0.0) << ",";
    std::cout << "\"candidate_swiglu_device_q8_swiglu_effective_raw_gb_s\":"
              << (q4_swiglu_device_q8_measured ? q4_swiglu_device_q8.swiglu_timing.effective_raw_gb_s : 0.0) << ",";
    std::cout << "\"candidate_swiglu_device_q8_swiglu_effective_io_gb_s\":"
              << (q4_swiglu_device_q8_measured ? q4_swiglu_device_q8.swiglu_timing.effective_io_gb_s : 0.0) << ",";
    std::cout << "\"candidate_swiglu_device_q8_quantize_min_us\":"
              << (q4_swiglu_device_q8_measured ? q4_swiglu_device_q8.q8_quantize_timing.min_us : 0.0) << ",";
    std::cout << "\"candidate_swiglu_device_q8_quantize_mean_us\":"
              << (q4_swiglu_device_q8_measured ? q4_swiglu_device_q8.q8_quantize_timing.mean_us : 0.0) << ",";
    std::cout << "\"candidate_swiglu_device_q8_quantize_effective_raw_gb_s\":"
              << (q4_swiglu_device_q8_measured ? q4_swiglu_device_q8.q8_quantize_timing.effective_raw_gb_s : 0.0) << ",";
    std::cout << "\"candidate_swiglu_device_q8_quantize_effective_io_gb_s\":"
              << (q4_swiglu_device_q8_measured ? q4_swiglu_device_q8.q8_quantize_timing.effective_io_gb_s : 0.0) << ",";
    std::cout << "\"candidate_swiglu_device_q8_down_kernel_min_us\":"
              << (q4_swiglu_device_q8_measured ? q4_swiglu_device_q8.down_timing.min_us : 0.0) << ",";
    std::cout << "\"candidate_swiglu_device_q8_down_kernel_mean_us\":"
              << (q4_swiglu_device_q8_measured ? q4_swiglu_device_q8.down_timing.mean_us : 0.0) << ",";
    std::cout << "\"candidate_swiglu_device_q8_down_effective_raw_gb_s\":"
              << (q4_swiglu_device_q8_measured ? q4_swiglu_device_q8.down_timing.effective_raw_gb_s : 0.0) << ",";
    std::cout << "\"candidate_swiglu_device_q8_down_effective_io_gb_s\":"
              << (q4_swiglu_device_q8_measured ? q4_swiglu_device_q8.down_timing.effective_io_gb_s : 0.0) << ",";
    std::cout << "\"candidate_swiglu_device_q8_shell_sum_min_us\":"
              << (q4_swiglu_device_q8_measured ? q4_swiglu_device_q8.shell_sum_min_us : 0.0) << ",";
    std::cout << "\"candidate_swiglu_device_q8_shell_sum_mean_us\":"
              << (q4_swiglu_device_q8_measured ? q4_swiglu_device_q8.shell_sum_mean_us : 0.0) << ",";
    std::cout << "\"candidate_swiglu_f32_input_swiglu_min_us\":"
              << (q4_swiglu_f32_input_measured ? q4_swiglu_f32_input.swiglu_timing.min_us : 0.0) << ",";
    std::cout << "\"candidate_swiglu_f32_input_swiglu_mean_us\":"
              << (q4_swiglu_f32_input_measured ? q4_swiglu_f32_input.swiglu_timing.mean_us : 0.0) << ",";
    std::cout << "\"candidate_swiglu_f32_input_swiglu_effective_raw_gb_s\":"
              << (q4_swiglu_f32_input_measured ? q4_swiglu_f32_input.swiglu_timing.effective_raw_gb_s : 0.0) << ",";
    std::cout << "\"candidate_swiglu_f32_input_swiglu_effective_io_gb_s\":"
              << (q4_swiglu_f32_input_measured ? q4_swiglu_f32_input.swiglu_timing.effective_io_gb_s : 0.0) << ",";
    std::cout << "\"candidate_swiglu_f32_input_down_kernel_min_us\":"
              << (q4_swiglu_f32_input_measured ? q4_swiglu_f32_input.down_timing.min_us : 0.0) << ",";
    std::cout << "\"candidate_swiglu_f32_input_down_kernel_mean_us\":"
              << (q4_swiglu_f32_input_measured ? q4_swiglu_f32_input.down_timing.mean_us : 0.0) << ",";
    std::cout << "\"candidate_swiglu_f32_input_down_effective_raw_gb_s\":"
              << (q4_swiglu_f32_input_measured ? q4_swiglu_f32_input.down_timing.effective_raw_gb_s : 0.0) << ",";
    std::cout << "\"candidate_swiglu_f32_input_down_effective_io_gb_s\":"
              << (q4_swiglu_f32_input_measured ? q4_swiglu_f32_input.down_timing.effective_io_gb_s : 0.0) << ",";
    std::cout << "\"candidate_swiglu_f32_input_shell_sum_min_us\":"
              << (q4_swiglu_f32_input_measured ? q4_swiglu_f32_input.shell_sum_min_us : 0.0) << ",";
    std::cout << "\"candidate_swiglu_f32_input_shell_sum_mean_us\":"
              << (q4_swiglu_f32_input_measured ? q4_swiglu_f32_input.shell_sum_mean_us : 0.0) << ",";
    std::cout << "\"candidate_q4_selected_shared_selected_kernel_min_us\":"
              << (q4_selected_shared_measured ? q4_selected_shared.selected_timing.min_us : 0.0) << ",";
    std::cout << "\"candidate_q4_selected_shared_selected_kernel_mean_us\":"
              << (q4_selected_shared_measured ? q4_selected_shared.selected_timing.mean_us : 0.0) << ",";
    std::cout << "\"candidate_q4_selected_shared_shared_kernel_min_us\":"
              << (q4_selected_shared_measured ? q4_selected_shared.shared_timing.min_us : 0.0) << ",";
    std::cout << "\"candidate_q4_selected_shared_shared_kernel_mean_us\":"
              << (q4_selected_shared_measured ? q4_selected_shared.shared_timing.mean_us : 0.0) << ",";
    std::cout << "\"candidate_q4_selected_shared_combined_kernel_min_us\":"
              << (q4_selected_shared_measured ? q4_selected_shared.combined_timing.min_us : 0.0) << ",";
    std::cout << "\"candidate_q4_selected_shared_combined_kernel_mean_us\":"
              << (q4_selected_shared_measured ? q4_selected_shared.combined_timing.mean_us : 0.0) << ",";
    std::cout << "\"candidate_q4_selected_shared_combined_effective_raw_gb_s\":"
              << (q4_selected_shared_measured ? q4_selected_shared.combined_timing.effective_raw_gb_s : 0.0) << ",";
    std::cout << "\"candidate_q4_selected_shared_combined_effective_io_gb_s\":"
              << (q4_selected_shared_measured ? q4_selected_shared.combined_timing.effective_io_gb_s : 0.0) << ",";
    std::cout << "\"candidate_q4_selected_shared_combined_global_work_items\":"
              << (q4_selected_shared_measured ? q4_selected_shared.combined_timing.global_work_items : 0) << ",";
    std::cout << "\"candidate_q4_selected_shared_combined_kernel_launches\":"
              << (q4_selected_shared_measured ? q4_selected_shared.combined_timing.kernel_launches : 0) << ",";
    std::cout << "\"candidate_q6_selected_rowstripe_kernel_min_us\":"
              << (q6_selected_shared_measured ? q6_selected_shared.selected_timing.min_us : 0.0) << ",";
    std::cout << "\"candidate_q6_selected_rowstripe_kernel_mean_us\":"
              << (q6_selected_shared_measured ? q6_selected_shared.selected_timing.mean_us : 0.0) << ",";
    std::cout << "\"candidate_q6_shared_raw_kernel_min_us\":"
              << (q6_selected_shared_measured ? q6_selected_shared.shared_timing.min_us : 0.0) << ",";
    std::cout << "\"candidate_q6_shared_raw_kernel_mean_us\":"
              << (q6_selected_shared_measured ? q6_selected_shared.shared_timing.mean_us : 0.0) << ",";
    std::cout << "\"candidate_q6_selected_shared_combined_kernel_min_us\":"
              << (q6_selected_shared_measured ? q6_selected_shared.combined_timing.min_us : 0.0) << ",";
    std::cout << "\"candidate_q6_selected_shared_combined_kernel_mean_us\":"
              << (q6_selected_shared_measured ? q6_selected_shared.combined_timing.mean_us : 0.0) << ",";
    std::cout << "\"candidate_q6_selected_shared_combined_effective_raw_gb_s\":"
              << (q6_selected_shared_measured ? q6_selected_shared.combined_timing.effective_raw_gb_s : 0.0) << ",";
    std::cout << "\"candidate_q6_selected_shared_combined_effective_io_gb_s\":"
              << (q6_selected_shared_measured ? q6_selected_shared.combined_timing.effective_io_gb_s : 0.0) << ",";
    std::cout << "\"candidate_q6_selected_shared_combined_global_work_items\":"
              << (q6_selected_shared_measured ? q6_selected_shared.combined_timing.global_work_items : 0) << ",";
    std::cout << "\"candidate_q6_selected_shared_combined_kernel_launches\":"
              << (q6_selected_shared_measured ? q6_selected_shared.combined_timing.kernel_launches : 0) << ",";
    std::cout << "\"candidate_q6_nonatomic_down_tail_contribution_kernel_min_us\":"
              << (q6_nonatomic_tail_measured ? q6_nonatomic_tail.contribution_timing.min_us : 0.0) << ",";
    std::cout << "\"candidate_q6_nonatomic_down_tail_contribution_kernel_mean_us\":"
              << (q6_nonatomic_tail_measured ? q6_nonatomic_tail.contribution_timing.mean_us : 0.0) << ",";
    std::cout << "\"candidate_q6_nonatomic_down_tail_reduce_kernel_min_us\":"
              << (q6_nonatomic_tail_measured ? q6_nonatomic_tail.reduce_timing.min_us : 0.0) << ",";
    std::cout << "\"candidate_q6_nonatomic_down_tail_reduce_kernel_mean_us\":"
              << (q6_nonatomic_tail_measured ? q6_nonatomic_tail.reduce_timing.mean_us : 0.0) << ",";
    std::cout << "\"candidate_q6_nonatomic_down_tail_shell_sum_min_us\":"
              << (q6_nonatomic_tail_measured ? q6_nonatomic_tail.shell_sum_min_us : 0.0) << ",";
    std::cout << "\"candidate_q6_nonatomic_down_tail_shell_sum_mean_us\":"
              << (q6_nonatomic_tail_measured ? q6_nonatomic_tail.shell_sum_mean_us : 0.0) << ",";
    std::cout << "\"candidate_q6_nonatomic_down_tail_contribution_global_work_items\":"
              << (q6_nonatomic_tail_measured ? q6_nonatomic_tail.contribution_timing.global_work_items : 0) << ",";
    std::cout << "\"candidate_q6_nonatomic_down_tail_reduce_global_work_items\":"
              << (q6_nonatomic_tail_measured ? q6_nonatomic_tail.reduce_timing.global_work_items : 0) << ",";
    std::cout << "\"candidate_q6_nonatomic_down_tail_kernel_launches\":"
              << (q6_nonatomic_tail_measured
                      ? q6_nonatomic_tail.contribution_timing.kernel_launches +
                            q6_nonatomic_tail.reduce_timing.kernel_launches
                      : 0) << ",";
    std::cout << "\"candidate_q6_rowgroup_down_tail_kernel_min_us\":"
              << (q6_rowgroup_tail_measured ? q6_rowgroup_tail.timing.min_us : 0.0) << ",";
    std::cout << "\"candidate_q6_rowgroup_down_tail_kernel_mean_us\":"
              << (q6_rowgroup_tail_measured ? q6_rowgroup_tail.timing.mean_us : 0.0) << ",";
    std::cout << "\"candidate_q6_rowgroup_down_tail_effective_raw_gb_s\":"
              << (q6_rowgroup_tail_measured ? q6_rowgroup_tail.timing.effective_raw_gb_s : 0.0) << ",";
    std::cout << "\"candidate_q6_rowgroup_down_tail_effective_io_gb_s\":"
              << (q6_rowgroup_tail_measured ? q6_rowgroup_tail.timing.effective_io_gb_s : 0.0) << ",";
    std::cout << "\"candidate_q6_rowgroup_down_tail_global_work_items\":"
              << (q6_rowgroup_tail_measured ? q6_rowgroup_tail.timing.global_work_items : 0) << ",";
    std::cout << "\"candidate_q6_rowgroup_down_tail_local_work_items\":"
              << (q6_rowgroup_tail_measured ? q6_rowgroup_tail.local_work_items : 0) << ",";
    std::cout << "\"candidate_q6_rowgroup_down_tail_kernel_launches\":"
              << (q6_rowgroup_tail_measured ? q6_rowgroup_tail.timing.kernel_launches : 0);
    std::cout << "},\"comparisons\":{";
    std::cout << "\"selected_down\":{";
    std::cout << "\"cpu_vs_oracle\":";
    WriteCompare(cpu_vs_oracle);
    std::cout << ",\"gpu_vs_cpu\":";
    WriteCompare(gpu_vs_cpu);
    std::cout << ",\"gpu_vs_oracle\":";
    WriteCompare(gpu_vs_oracle);
    std::cout << "},\"candidate_pair2\":{";
    std::cout << "\"gpu_vs_cpu\":";
    WriteCompare(pair2_vs_cpu);
    std::cout << ",\"gpu_vs_oracle\":";
    WriteCompare(pair2_vs_oracle);
    std::cout << ",\"gpu_vs_current\":";
    WriteCompare(pair2_vs_current);
    std::cout << "},\"weighted_selected_down\":{";
    std::cout << "\"cpu_weighted_vs_oracle\":";
    WriteCompare(weighted_cpu_vs_oracle);
    std::cout << ",\"gpu_vs_cpu_weighted\":";
    WriteCompare(weighted_vs_cpu);
    std::cout << ",\"gpu_vs_oracle\":";
    WriteCompare(weighted_vs_oracle);
    std::cout << ",\"gpu_vs_current_weighted\":";
    WriteCompare(weighted_vs_current_weighted);
    std::cout << "},\"weighted_sum_selected_down\":{";
    std::cout << "\"cpu_weighted_sum_vs_oracle\":";
    WriteCompare(weighted_sum_cpu_vs_oracle);
    std::cout << ",\"gpu_vs_cpu_weighted_sum\":";
    WriteCompare(weighted_sum_vs_cpu);
    std::cout << ",\"gpu_vs_oracle_weighted_sum\":";
    WriteCompare(weighted_sum_vs_oracle);
    std::cout << ",\"gpu_vs_current_weighted_sum\":";
    WriteCompare(weighted_sum_vs_current);
    std::cout << ",\"gpu_vs_weighted_kernel_sum\":";
    WriteCompare(weighted_sum_vs_weighted_kernel);
    std::cout << "},\"group8_sum_selected_down\":{";
    std::cout << "\"gpu_vs_cpu_weighted_sum\":";
    WriteCompare(group8_sum_vs_cpu);
    std::cout << ",\"gpu_vs_oracle_weighted_sum\":";
    WriteCompare(group8_sum_vs_oracle);
    std::cout << ",\"gpu_vs_current_weighted_sum\":";
    WriteCompare(group8_sum_vs_current);
    std::cout << ",\"gpu_vs_weighted_kernel_sum\":";
    WriteCompare(group8_sum_vs_weighted_kernel);
    std::cout << "},\"selected_shape_occupancy\":{";
    std::cout << "\"occupancy4_vs_cpu\":";
    WriteCompare(occupancy4_vs_cpu);
    std::cout << ",\"occupancy4_vs_oracle\":";
    WriteCompare(occupancy4_vs_oracle);
    std::cout << "},\"device_q8_selected_down\":{";
    std::cout << "\"gpu_vs_cpu\":";
    WriteCompare(device_q8_vs_cpu);
    std::cout << ",\"gpu_vs_oracle\":";
    WriteCompare(device_q8_vs_oracle);
    std::cout << ",\"gpu_vs_current\":";
    WriteCompare(device_q8_vs_current);
    std::cout << ",\"q8_planes_vs_cpu\":";
    WriteQ8PlaneCompare(device_q8_planes_vs_cpu);
    std::cout << "},\"swiglu_device_q8_selected_down\":{";
    std::cout << "\"swiglu_vs_cpu\":";
    WriteCompare(swiglu_device_q8_swiglu_vs_cpu);
    std::cout << ",\"gpu_vs_cpu\":";
    WriteCompare(swiglu_device_q8_vs_cpu);
    std::cout << ",\"gpu_vs_oracle\":";
    WriteCompare(swiglu_device_q8_vs_oracle);
    std::cout << ",\"gpu_vs_current\":";
    WriteCompare(swiglu_device_q8_vs_current);
    std::cout << ",\"q8_planes_vs_cpu\":";
    WriteQ8PlaneCompare(swiglu_device_q8_planes_vs_cpu);
    std::cout << "},\"swiglu_f32_input_selected_down\":{";
    std::cout << "\"swiglu_vs_cpu\":";
    WriteCompare(swiglu_f32_input_swiglu_vs_cpu);
    std::cout << ",\"gpu_vs_cpu\":";
    WriteCompare(swiglu_f32_input_vs_cpu);
    std::cout << ",\"gpu_vs_oracle\":";
    WriteCompare(swiglu_f32_input_vs_oracle);
    std::cout << ",\"gpu_vs_current\":";
    WriteCompare(swiglu_f32_input_vs_current);
    std::cout << "},\"q4_selected_shared_down\":{";
    std::cout << "\"shared_cpu_vs_oracle\":";
    WriteCompare(shared_cpu_vs_oracle);
    std::cout << ",\"selected_combined_vs_separate\":";
    WriteCompare(q4_selected_combined_vs_separate);
    std::cout << ",\"selected_combined_vs_oracle\":";
    WriteCompare(q4_selected_combined_vs_oracle);
    std::cout << ",\"shared_separate_vs_cpu\":";
    WriteCompare(q4_shared_separate_vs_cpu);
    std::cout << ",\"shared_combined_vs_separate\":";
    WriteCompare(q4_shared_combined_vs_separate);
    std::cout << ",\"shared_combined_vs_oracle\":";
    WriteCompare(q4_shared_combined_vs_oracle);
    std::cout << "},\"q6_selected_shared_down\":{";
    std::cout << "\"shared_cpu_vs_oracle\":";
    WriteCompare(shared_cpu_vs_oracle);
    std::cout << ",\"selected_separate_vs_cpu\":";
    WriteCompare(q6_selected_separate_vs_cpu);
    std::cout << ",\"selected_combined_vs_separate\":";
    WriteCompare(q6_selected_combined_vs_separate);
    std::cout << ",\"selected_combined_vs_oracle\":";
    WriteCompare(q6_selected_combined_vs_oracle);
    std::cout << ",\"shared_separate_vs_cpu\":";
    WriteCompare(q6_shared_separate_vs_cpu);
    std::cout << ",\"shared_combined_vs_separate\":";
    WriteCompare(q6_shared_combined_vs_separate);
    std::cout << ",\"shared_combined_vs_oracle\":";
    WriteCompare(q6_shared_combined_vs_oracle);
    std::cout << "},\"q6_nonatomic_down_tail\":{";
    std::cout << "\"cpu_ffn_out_vs_oracle\":";
    WriteCompare(cpu_ffn_out_vs_oracle);
    std::cout << ",\"cpu_layer_vs_oracle\":";
    WriteCompare(cpu_layer_vs_oracle);
    std::cout << ",\"contrib_vs_cpu\":";
    WriteCompare(q6_tail_contrib_vs_cpu);
    std::cout << ",\"layer_vs_cpu\":";
    WriteCompare(q6_tail_layer_vs_cpu);
    std::cout << ",\"layer_vs_oracle\":";
    WriteCompare(q6_tail_layer_vs_oracle);
    std::cout << "},\"q6_rowgroup_down_tail\":{";
    std::cout << "\"cpu_ffn_out_vs_oracle\":";
    WriteCompare(cpu_ffn_out_vs_oracle);
    std::cout << ",\"cpu_layer_vs_oracle\":";
    WriteCompare(cpu_layer_vs_oracle);
    std::cout << ",\"layer_vs_cpu\":";
    WriteCompare(q6_rowgroup_tail_layer_vs_cpu);
    std::cout << ",\"layer_vs_oracle\":";
    WriteCompare(q6_rowgroup_tail_layer_vs_oracle);
    std::cout << ",\"layer_vs_nonatomic\":";
    WriteCompare(q6_rowgroup_tail_layer_vs_nonatomic);
    std::cout << "}";
    std::cout << "},\"checks\":{";
    std::cout << "\"load_map_ready\":" << (load_map.ready ? "true" : "false") << ",";
    std::cout << "\"tensor_shape_ok\":" << (tensor_shape_ok ? "true" : "false") << ",";
    std::cout << "\"shared_down_tensor_shape_ok\":" << (shared_down_tensor_shape_ok ? "true" : "false") << ",";
    std::cout << "\"shared_input_gate_tensor_shape_ok\":"
              << (shared_input_gate_tensor_shape_ok ? "true" : "false") << ",";
    std::cout << "\"counts_ok\":" << (counts_ok ? "true" : "false") << ",";
    std::cout << "\"arc_device_selected\":" << (gpu.device_name.find(args.device_substring) != std::string::npos ? "true" : "false") << ",";
    std::cout << "\"selected_down_matches_oracle\":" << (comparisons_passed ? "true" : "false") << ",";
    std::cout << "\"candidate_pair2_matches_current_and_oracle\":"
              << (pair2_comparisons_passed ? "true" : "false") << ",";
    std::cout << "\"candidate_weighted_matches_current_and_oracle\":"
              << (weighted_comparisons_passed ? "true" : "false") << ",";
    std::cout << "\"candidate_weighted_sum_matches_current_and_oracle\":"
              << (weighted_sum_comparisons_passed ? "true" : "false") << ",";
    std::cout << "\"candidate_group8_sum_matches_current_and_oracle\":"
              << (group8_sum_comparisons_passed ? "true" : "false") << ",";
    std::cout << "\"candidate_occupancy_matches_current_and_oracle\":"
              << (occupancy4_comparisons_passed ? "true" : "false") << ",";
    std::cout << "\"candidate_device_q8_matches_current_and_oracle\":"
              << (device_q8_comparisons_passed ? "true" : "false") << ",";
    std::cout << "\"candidate_device_q8_planes_match_cpu\":"
              << ((!q4_device_q8_measured || device_q8_planes_vs_cpu.passed) ? "true" : "false") << ",";
    std::cout << "\"candidate_swiglu_device_q8_matches_current_and_oracle\":"
              << (swiglu_device_q8_comparisons_passed ? "true" : "false") << ",";
    std::cout << "\"candidate_swiglu_device_q8_planes_match_cpu\":"
              << ((!q4_swiglu_device_q8_measured || swiglu_device_q8_planes_vs_cpu.passed) ? "true" : "false") << ",";
    std::cout << "\"candidate_swiglu_f32_input_comparison_recorded\":"
              << (swiglu_f32_input_comparison_recorded ? "true" : "false") << ",";
    std::cout << "\"candidate_swiglu_f32_input_matches_current_and_oracle\":"
              << (swiglu_f32_input_output_oracle_compatible ? "true" : "false") << ",";
    std::cout << "\"candidate_q4_selected_shared_matches_current_and_oracle\":"
              << (q4_selected_shared_comparisons_passed ? "true" : "false") << ",";
    std::cout << "\"candidate_q6_selected_shared_matches_current_and_oracle\":"
              << (q6_selected_shared_comparisons_passed ? "true" : "false") << ",";
    std::cout << "\"candidate_q6_nonatomic_down_tail_matches_current_and_oracle\":"
              << (q6_nonatomic_tail_comparisons_passed ? "true" : "false") << ",";
    std::cout << "\"candidate_q6_rowgroup_down_tail_matches_current_and_oracle\":"
              << (q6_rowgroup_tail_comparisons_passed ? "true" : "false") << ",";
    std::cout << "\"gpu_event_timing_positive\":" << (timings_positive ? "true" : "false") << ",";
    std::cout << "\"candidate_pair2_event_timing_positive\":"
              << (pair2_timing_positive ? "true" : "false") << ",";
    std::cout << "\"candidate_weighted_event_timing_positive\":"
              << (weighted_timing_positive ? "true" : "false") << ",";
    std::cout << "\"candidate_weighted_sum_event_timing_positive\":"
              << (weighted_sum_timing_positive ? "true" : "false") << ",";
    std::cout << "\"candidate_group8_sum_event_timing_positive\":"
              << (group8_sum_timing_positive ? "true" : "false") << ",";
    std::cout << "\"candidate_occupancy_event_timing_positive\":"
              << (occupancy4_timing_positive ? "true" : "false") << ",";
    std::cout << "\"candidate_device_q8_event_timing_positive\":"
              << (device_q8_timing_positive ? "true" : "false") << ",";
    std::cout << "\"candidate_swiglu_device_q8_event_timing_positive\":"
              << (swiglu_device_q8_timing_positive ? "true" : "false") << ",";
    std::cout << "\"candidate_swiglu_f32_input_event_timing_positive\":"
              << (swiglu_f32_input_timing_positive ? "true" : "false") << ",";
    std::cout << "\"candidate_q4_selected_shared_event_timing_positive\":"
              << (q4_selected_shared_timing_positive ? "true" : "false") << ",";
    std::cout << "\"candidate_q6_selected_shared_event_timing_positive\":"
              << (q6_selected_shared_timing_positive ? "true" : "false") << ",";
    std::cout << "\"candidate_q6_nonatomic_down_tail_event_timing_positive\":"
              << (q6_nonatomic_tail_timing_positive ? "true" : "false") << ",";
    std::cout << "\"candidate_q6_rowgroup_down_tail_event_timing_positive\":"
              << (q6_rowgroup_tail_timing_positive ? "true" : "false") << ",";
    std::cout << "\"speedup_claims_allowed\":false";
    std::cout << "},\"required_checks_passed\":" << (checks_passed ? "true" : "false");
    std::cout << "}\n";
    return checks_passed ? 0 : 3;
  } catch (const std::exception& exc) {
    std::cout << "{\"ok\":false,\"error\":\"" << JsonEscape(exc.what()) << "\"}\n";
    return 2;
  }
}
'''


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
  parser.add_argument("--repeat", type=int, default=7)
  parser.add_argument("--device-substring", default="B390")
  parser.add_argument("--timeout-s", type=int, default=900)
  parser.add_argument("--out-dir", type=Path, default=None)
  return parser.parse_args()


def shell_join(argv: list[str]) -> str:
  return " ".join(shlex.quote(item) for item in argv)


def cpp_raw_string_literal(value: str) -> str:
  delimiter = "IQ36CL"
  if f"){delimiter}\"" in value:
    raise ValueError(f"OpenCL source contains raw-string delimiter {delimiter}")
  return f'R"{delimiter}({value}){delimiter}"'


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
    path = (PAYLOAD_ROOT / pattern.format(layer=layer)).resolve()
    if not path.exists():
      glob_pattern = pattern.format(layer=layer)
      marker = "__ord"
      if marker in glob_pattern:
        prefix, suffix = glob_pattern.split(marker, 1)
        if ".bin" in suffix:
          glob_pattern = prefix + marker + "*.bin"
      matches = sorted(PAYLOAD_ROOT.glob(glob_pattern))
      if len(matches) != 1:
        raise SystemExit(
            f"selected down payload missing: {path}; glob {glob_pattern} matched {len(matches)} files"
        )
      path = matches[0].resolve()
    if path.stat().st_size != size_bytes:
      raise SystemExit(f"selected down payload size mismatch: {path}")
    payloads[name] = {
        "local_path": path,
        "path": str(path.relative_to(ROOT)),
        "sha256": iq36_local.sha256_file(path),
        "size_bytes": size_bytes,
        "stage_name": stage_name,
    }
  return payloads


def nested_bool(obj: dict[str, Any], *keys: str) -> bool:
  current: Any = obj
  for key in keys:
    if not isinstance(current, dict):
      return False
    current = current.get(key)
  return current is True


def nested_number(obj: dict[str, Any], *keys: str) -> float | None:
  current: Any = obj
  for key in keys:
    if not isinstance(current, dict):
      return None
    current = current.get(key)
  return float(current) if isinstance(current, (int, float)) else None


def write_summary(path: Path, payload: dict[str, Any]) -> None:
  probe = payload.get("probe") if isinstance(payload.get("probe"), dict) else {}
  comparison = (
      probe.get("comparisons", {}).get("selected_down", {})
      if isinstance(probe, dict)
      else {}
  )
  timings = probe.get("timings", {}) if isinstance(probe, dict) else {}
  speedup = probe.get("candidate_pair2_speedup_vs_current") if isinstance(probe, dict) else None
  weighted_speedup = (
      probe.get("candidate_weighted_speedup_vs_current")
      if isinstance(probe, dict)
      else None
  )
  weighted_sum_speedup = (
      probe.get("candidate_weighted_sum_speedup_vs_current")
      if isinstance(probe, dict)
      else None
  )
  weighted_sum_vs_weighted = (
      probe.get("candidate_weighted_sum_speedup_vs_weighted")
      if isinstance(probe, dict)
      else None
  )
  group8_sum_speedup = (
      probe.get("candidate_group8_sum_speedup_vs_current")
      if isinstance(probe, dict)
      else None
  )
  group8_sum_vs_weighted = (
      probe.get("candidate_group8_sum_speedup_vs_weighted")
      if isinstance(probe, dict)
      else None
  )
  group8_sum_vs_serial = (
      probe.get("candidate_group8_sum_speedup_vs_serial_sum")
      if isinstance(probe, dict)
      else None
  )
  occupancy_speedup = (
      probe.get("candidate_occupancy_scaled_speedup_vs_current")
      if isinstance(probe, dict)
      else None
  )
  occupancy_single_group = (
      probe.get("candidate_occupancy_effective_single_group_us")
      if isinstance(probe, dict)
      else None
  )
  device_q8_down_speedup = (
      probe.get("candidate_device_q8_down_speedup_vs_current")
      if isinstance(probe, dict)
      else None
  )
  device_q8_shell_ratio = (
      probe.get("candidate_device_q8_shell_ratio_vs_current")
      if isinstance(probe, dict)
      else None
  )
  swiglu_device_q8_down_speedup = (
      probe.get("candidate_swiglu_device_q8_down_speedup_vs_current")
      if isinstance(probe, dict)
      else None
  )
  swiglu_device_q8_shell_ratio = (
      probe.get("candidate_swiglu_device_q8_shell_ratio_vs_current")
      if isinstance(probe, dict)
      else None
  )
  swiglu_f32_input_down_speedup = (
      probe.get("candidate_swiglu_f32_input_down_speedup_vs_current")
      if isinstance(probe, dict)
      else None
  )
  swiglu_f32_input_shell_ratio = (
      probe.get("candidate_swiglu_f32_input_shell_ratio_vs_current")
      if isinstance(probe, dict)
      else None
  )
  q4_selected_shared_speedup = (
      probe.get("candidate_q4_selected_shared_combined_speedup_vs_separate")
      if isinstance(probe, dict)
      else None
  )
  q6_selected_shared_speedup = (
      probe.get("candidate_q6_selected_shared_combined_speedup_vs_separate")
      if isinstance(probe, dict)
      else None
  )
  q6_nonatomic_tail_ratio = (
      probe.get("candidate_q6_nonatomic_down_tail_ratio_vs_combined_down")
      if isinstance(probe, dict)
      else None
  )
  q6_rowgroup_tail_ratio = (
      probe.get("candidate_q6_rowgroup_down_tail_ratio_vs_combined_down")
      if isinstance(probe, dict)
      else None
  )
  q6_rowgroup_tail_ratio_vs_nonatomic = (
      probe.get("candidate_q6_rowgroup_down_tail_ratio_vs_nonatomic")
      if isinstance(probe, dict)
      else None
  )
  lines = [
      "# GPU Selected Down Probe",
      "",
      f"- workstream: `{WORKSTREAM}`",
      f"- host: `{payload.get('host')}`",
      f"- model: `{payload.get('model')}`",
      f"- layer: `{payload.get('layer')}`",
      f"- required checks passed: `{str(payload.get('required_checks_passed')).lower()}`",
      "- speedup claims allowed: `false`",
      f"- platform/device: `{probe.get('platform_name')}` / `{probe.get('device_name')}`",
      f"- tensor: `{probe.get('tensor_name')}` `{probe.get('tensor_type')}`",
      f"- expert ids: `{probe.get('expert_ids')}`",
      f"- current kernel: `{probe.get('current_kernel')}`",
      f"- pair2 candidate: `{probe.get('candidate_pair2_kernel')}`",
      f"- pair2/current component ratio: `{speedup}`",
      f"- pair2 material component speedup: `{str(probe.get('candidate_pair2_material_component_speedup')).lower()}`",
      f"- weighted candidate: `{probe.get('candidate_weighted_kernel')}`",
      f"- weighted/current component ratio: `{weighted_speedup}`",
      f"- weighted boundary reopen candidate: `{str(probe.get('candidate_weighted_boundary_reopen_candidate')).lower()}`",
      f"- weighted-sum candidate: `{probe.get('candidate_weighted_sum_kernel')}`",
      f"- weighted-sum/current component ratio: `{weighted_sum_speedup}`",
      f"- weighted-sum/weighted component ratio: `{weighted_sum_vs_weighted}`",
      f"- weighted-sum boundary reopen candidate: `{str(probe.get('candidate_weighted_sum_boundary_reopen_candidate')).lower()}`",
      f"- group8-sum candidate: `{probe.get('candidate_group8_sum_kernel')}`",
      f"- group8-sum local items: `{probe.get('candidate_group8_sum_local_items')}`",
      f"- group8-sum/current component ratio: `{group8_sum_speedup}`",
      f"- group8-sum/weighted component ratio: `{group8_sum_vs_weighted}`",
      f"- group8-sum/serial-sum component ratio: `{group8_sum_vs_serial}`",
      f"- group8-sum boundary reopen candidate: `{str(probe.get('candidate_group8_sum_boundary_reopen_candidate')).lower()}`",
      f"- occupancy candidate: `{probe.get('candidate_occupancy_kernel')}`",
      f"- occupancy groups: `{probe.get('candidate_occupancy_group_count')}`",
      f"- occupancy scaled ratio vs current: `{occupancy_speedup}`",
      f"- occupancy effective single-group us: `{occupancy_single_group}`",
      f"- selected-shape occupancy bound: `{str(probe.get('selected_shape_occupancy_bound')).lower()}`",
      f"- device-Q8 candidate: `{probe.get('candidate_device_q8_kernel')}`",
      f"- device-Q8 down/current component ratio: `{device_q8_down_speedup}`",
      f"- device-Q8 shell/current component ratio: `{device_q8_shell_ratio}`",
      f"- device-Q8 boundary reopen candidate: `{str(probe.get('candidate_device_q8_boundary_reopen_candidate')).lower()}`",
      f"- SwiGLU device-Q8 candidate: `{probe.get('candidate_swiglu_device_q8_kernel')}`",
      f"- SwiGLU device-Q8 down/current component ratio: `{swiglu_device_q8_down_speedup}`",
      f"- SwiGLU device-Q8 shell/current component ratio: `{swiglu_device_q8_shell_ratio}`",
      f"- SwiGLU device-Q8 boundary reopen candidate: `{str(probe.get('candidate_swiglu_device_q8_boundary_reopen_candidate')).lower()}`",
      f"- SwiGLU f32-input candidate: `{probe.get('candidate_swiglu_f32_input_kernel')}`",
      f"- SwiGLU f32-input down/current component ratio: `{swiglu_f32_input_down_speedup}`",
      f"- SwiGLU f32-input shell/current component ratio: `{swiglu_f32_input_shell_ratio}`",
      f"- SwiGLU f32-input output oracle compatible: `{str(probe.get('candidate_swiglu_f32_input_output_oracle_compatible')).lower()}`",
      f"- SwiGLU f32-input boundary reopen candidate: `{str(probe.get('candidate_swiglu_f32_input_boundary_reopen_candidate')).lower()}`",
      f"- Q4 selected+shared candidate: `{probe.get('candidate_q4_selected_shared_kernel')}`",
      f"- Q4 selected+shared combined/separate component ratio: `{q4_selected_shared_speedup}`",
      f"- Q4 selected+shared boundary reopen candidate: `{str(probe.get('candidate_q4_selected_shared_boundary_reopen_candidate')).lower()}`",
      f"- Q6 selected+shared candidate: `{probe.get('candidate_q6_selected_shared_kernel')}`",
      f"- Q6 selected+shared combined/separate component ratio: `{q6_selected_shared_speedup}`",
      f"- Q6 non-atomic down-tail candidate: `{probe.get('candidate_q6_nonatomic_down_tail_kernel')}`",
      f"- Q6 non-atomic down-tail shell/combined-down ratio: `{q6_nonatomic_tail_ratio}`",
      f"- Q6 non-atomic down-tail preserves contributor parallelism: `{str(probe.get('candidate_q6_nonatomic_down_tail_preserves_contributor_parallelism')).lower()}`",
      f"- Q6 non-atomic down-tail boundary reopen candidate: `{str(probe.get('candidate_q6_nonatomic_down_tail_boundary_reopen_candidate')).lower()}`",
      f"- Q6 rowgroup down-tail candidate: `{probe.get('candidate_q6_rowgroup_down_tail_kernel')}`",
      f"- Q6 rowgroup down-tail shell/combined-down ratio: `{q6_rowgroup_tail_ratio}`",
      f"- Q6 rowgroup down-tail shell/non-atomic ratio: `{q6_rowgroup_tail_ratio_vs_nonatomic}`",
      f"- Q6 rowgroup down-tail target us: `{probe.get('candidate_q6_rowgroup_down_tail_component_shell_target_us')}`",
      f"- Q6 rowgroup down-tail preserves contributor parallelism: `{str(probe.get('candidate_q6_rowgroup_down_tail_preserves_contributor_parallelism')).lower()}`",
      f"- Q6 rowgroup down-tail boundary reopen candidate: `{str(probe.get('candidate_q6_rowgroup_down_tail_boundary_reopen_candidate')).lower()}`",
      "",
      "| comparison | max abs | RMSE |",
      "|---|---:|---:|",
  ]
  for lane in ("cpu_vs_oracle", "gpu_vs_cpu", "gpu_vs_oracle"):
    cmp = comparison.get(lane, {}) if isinstance(comparison, dict) else {}
    lines.append(f"| {lane} | {cmp.get('max_abs_diff')} | {cmp.get('rmse')} |")
  pair2 = (
      probe.get("comparisons", {}).get("candidate_pair2", {})
      if isinstance(probe, dict)
      else {}
  )
  for lane in ("gpu_vs_cpu", "gpu_vs_oracle", "gpu_vs_current"):
    cmp = pair2.get(lane, {}) if isinstance(pair2, dict) else {}
    lines.append(f"| pair2 {lane} | {cmp.get('max_abs_diff')} | {cmp.get('rmse')} |")
  weighted = (
      probe.get("comparisons", {}).get("weighted_selected_down", {})
      if isinstance(probe, dict)
      else {}
  )
  for lane in ("cpu_weighted_vs_oracle", "gpu_vs_cpu_weighted", "gpu_vs_oracle", "gpu_vs_current_weighted"):
    cmp = weighted.get(lane, {}) if isinstance(weighted, dict) else {}
    lines.append(f"| weighted {lane} | {cmp.get('max_abs_diff')} | {cmp.get('rmse')} |")
  weighted_sum = (
      probe.get("comparisons", {}).get("weighted_sum_selected_down", {})
      if isinstance(probe, dict)
      else {}
  )
  for lane in (
      "cpu_weighted_sum_vs_oracle",
      "gpu_vs_cpu_weighted_sum",
      "gpu_vs_oracle_weighted_sum",
      "gpu_vs_current_weighted_sum",
      "gpu_vs_weighted_kernel_sum",
  ):
    cmp = weighted_sum.get(lane, {}) if isinstance(weighted_sum, dict) else {}
    lines.append(f"| weighted-sum {lane} | {cmp.get('max_abs_diff')} | {cmp.get('rmse')} |")
  group8_sum = (
      probe.get("comparisons", {}).get("group8_sum_selected_down", {})
      if isinstance(probe, dict)
      else {}
  )
  for lane in (
      "gpu_vs_cpu_weighted_sum",
      "gpu_vs_oracle_weighted_sum",
      "gpu_vs_current_weighted_sum",
      "gpu_vs_weighted_kernel_sum",
  ):
    cmp = group8_sum.get(lane, {}) if isinstance(group8_sum, dict) else {}
    lines.append(f"| group8-sum {lane} | {cmp.get('max_abs_diff')} | {cmp.get('rmse')} |")
  occupancy = (
      probe.get("comparisons", {}).get("selected_shape_occupancy", {})
      if isinstance(probe, dict)
      else {}
  )
  for lane in ("occupancy4_vs_cpu", "occupancy4_vs_oracle"):
    cmp = occupancy.get(lane, {}) if isinstance(occupancy, dict) else {}
    lines.append(f"| occupancy {lane} | {cmp.get('max_abs_diff')} | {cmp.get('rmse')} |")
  device_q8 = (
      probe.get("comparisons", {}).get("device_q8_selected_down", {})
      if isinstance(probe, dict)
      else {}
  )
  for lane in ("gpu_vs_cpu", "gpu_vs_oracle", "gpu_vs_current"):
    cmp = device_q8.get(lane, {}) if isinstance(device_q8, dict) else {}
    lines.append(f"| device-Q8 {lane} | {cmp.get('max_abs_diff')} | {cmp.get('rmse')} |")
  q8_cmp = device_q8.get("q8_planes_vs_cpu", {}) if isinstance(device_q8, dict) else {}
  lines.append(
      "| device-Q8 q8_planes_vs_cpu | "
      f"{q8_cmp.get('d_max_abs_diff')} | {q8_cmp.get('d_rmse')} |"
  )
  swiglu_device_q8 = (
      probe.get("comparisons", {}).get("swiglu_device_q8_selected_down", {})
      if isinstance(probe, dict)
      else {}
  )
  for lane in ("swiglu_vs_cpu", "gpu_vs_cpu", "gpu_vs_oracle", "gpu_vs_current"):
    cmp = swiglu_device_q8.get(lane, {}) if isinstance(swiglu_device_q8, dict) else {}
    lines.append(f"| SwiGLU device-Q8 {lane} | {cmp.get('max_abs_diff')} | {cmp.get('rmse')} |")
  swiglu_q8_cmp = (
      swiglu_device_q8.get("q8_planes_vs_cpu", {})
      if isinstance(swiglu_device_q8, dict)
      else {}
  )
  lines.append(
      "| SwiGLU device-Q8 q8_planes_vs_cpu | "
      f"{swiglu_q8_cmp.get('d_max_abs_diff')} | {swiglu_q8_cmp.get('d_rmse')} |"
  )
  swiglu_f32_input = (
      probe.get("comparisons", {}).get("swiglu_f32_input_selected_down", {})
      if isinstance(probe, dict)
      else {}
  )
  for lane in ("swiglu_vs_cpu", "gpu_vs_cpu", "gpu_vs_oracle", "gpu_vs_current"):
    cmp = swiglu_f32_input.get(lane, {}) if isinstance(swiglu_f32_input, dict) else {}
    lines.append(f"| SwiGLU f32-input {lane} | {cmp.get('max_abs_diff')} | {cmp.get('rmse')} |")
  q4_selected_shared = (
      probe.get("comparisons", {}).get("q4_selected_shared_down", {})
      if isinstance(probe, dict)
      else {}
  )
  for lane in (
      "shared_cpu_vs_oracle",
      "selected_combined_vs_separate",
      "selected_combined_vs_oracle",
      "shared_separate_vs_cpu",
      "shared_combined_vs_separate",
      "shared_combined_vs_oracle",
  ):
    cmp = q4_selected_shared.get(lane, {}) if isinstance(q4_selected_shared, dict) else {}
    lines.append(f"| Q4 selected+shared {lane} | {cmp.get('max_abs_diff')} | {cmp.get('rmse')} |")
  q6_selected_shared = (
      probe.get("comparisons", {}).get("q6_selected_shared_down", {})
      if isinstance(probe, dict)
      else {}
  )
  for lane in (
      "shared_cpu_vs_oracle",
      "selected_separate_vs_cpu",
      "selected_combined_vs_separate",
      "selected_combined_vs_oracle",
      "shared_separate_vs_cpu",
      "shared_combined_vs_separate",
      "shared_combined_vs_oracle",
  ):
    cmp = q6_selected_shared.get(lane, {}) if isinstance(q6_selected_shared, dict) else {}
    lines.append(f"| Q6 selected+shared {lane} | {cmp.get('max_abs_diff')} | {cmp.get('rmse')} |")
  q6_nonatomic_tail = (
      probe.get("comparisons", {}).get("q6_nonatomic_down_tail", {})
      if isinstance(probe, dict)
      else {}
  )
  for lane in (
      "cpu_ffn_out_vs_oracle",
      "cpu_layer_vs_oracle",
      "contrib_vs_cpu",
      "layer_vs_cpu",
      "layer_vs_oracle",
  ):
    cmp = q6_nonatomic_tail.get(lane, {}) if isinstance(q6_nonatomic_tail, dict) else {}
    lines.append(f"| Q6 non-atomic down-tail {lane} | {cmp.get('max_abs_diff')} | {cmp.get('rmse')} |")
  q6_rowgroup_tail = (
      probe.get("comparisons", {}).get("q6_rowgroup_down_tail", {})
      if isinstance(probe, dict)
      else {}
  )
  for lane in (
      "cpu_ffn_out_vs_oracle",
      "cpu_layer_vs_oracle",
      "layer_vs_cpu",
      "layer_vs_oracle",
      "layer_vs_nonatomic",
  ):
    cmp = q6_rowgroup_tail.get(lane, {}) if isinstance(q6_rowgroup_tail, dict) else {}
    lines.append(f"| Q6 rowgroup down-tail {lane} | {cmp.get('max_abs_diff')} | {cmp.get('rmse')} |")
  lines += [
      "",
      "| kernel | min us | mean us | raw GB/s | IO GB/s |",
      "|---|---:|---:|---:|---:|",
      "| current selected_down | "
      f"{timings.get('selected_down_gpu_kernel_min_us')} | "
      f"{timings.get('selected_down_gpu_kernel_mean_us')} | "
      f"{timings.get('selected_down_gpu_effective_raw_gb_s')} | "
      f"{timings.get('selected_down_gpu_effective_io_gb_s')} |",
      "| pair2 candidate | "
      f"{timings.get('candidate_pair2_kernel_min_us')} | "
      f"{timings.get('candidate_pair2_kernel_mean_us')} | "
      f"{timings.get('candidate_pair2_effective_raw_gb_s')} | "
      f"{timings.get('candidate_pair2_effective_io_gb_s')} |",
      "| weighted candidate | "
      f"{timings.get('candidate_weighted_kernel_min_us')} | "
      f"{timings.get('candidate_weighted_kernel_mean_us')} | "
      f"{timings.get('candidate_weighted_effective_raw_gb_s')} | "
      f"{timings.get('candidate_weighted_effective_io_gb_s')} |",
      "| weighted-sum candidate | "
      f"{timings.get('candidate_weighted_sum_kernel_min_us')} | "
      f"{timings.get('candidate_weighted_sum_kernel_mean_us')} | "
      f"{timings.get('candidate_weighted_sum_effective_raw_gb_s')} | "
      f"{timings.get('candidate_weighted_sum_effective_io_gb_s')} |",
      "| group8-sum candidate | "
      f"{timings.get('candidate_group8_sum_kernel_min_us')} | "
      f"{timings.get('candidate_group8_sum_kernel_mean_us')} | "
      f"{timings.get('candidate_group8_sum_effective_raw_gb_s')} | "
      f"{timings.get('candidate_group8_sum_effective_io_gb_s')} |",
      "| occupancy4 candidate | "
      f"{timings.get('candidate_occupancy_kernel_min_us')} | "
      f"{timings.get('candidate_occupancy_kernel_mean_us')} | "
      f"{timings.get('candidate_occupancy_effective_raw_gb_s')} | "
      f"{timings.get('candidate_occupancy_effective_io_gb_s')} |",
      "| device-Q8 quantize | "
      f"{timings.get('candidate_device_q8_quantize_min_us')} | "
      f"{timings.get('candidate_device_q8_quantize_mean_us')} | "
      f"{timings.get('candidate_device_q8_quantize_effective_raw_gb_s')} | "
      f"{timings.get('candidate_device_q8_quantize_effective_io_gb_s')} |",
      "| device-Q8 down | "
      f"{timings.get('candidate_device_q8_down_kernel_min_us')} | "
      f"{timings.get('candidate_device_q8_down_kernel_mean_us')} | "
      f"{timings.get('candidate_device_q8_down_effective_raw_gb_s')} | "
      f"{timings.get('candidate_device_q8_down_effective_io_gb_s')} |",
      "| device-Q8 shell sum | "
      f"{timings.get('candidate_device_q8_shell_sum_min_us')} | "
      f"{timings.get('candidate_device_q8_shell_sum_mean_us')} | "
      "None | None |",
      "| SwiGLU device-Q8 swiglu | "
      f"{timings.get('candidate_swiglu_device_q8_swiglu_min_us')} | "
      f"{timings.get('candidate_swiglu_device_q8_swiglu_mean_us')} | "
      f"{timings.get('candidate_swiglu_device_q8_swiglu_effective_raw_gb_s')} | "
      f"{timings.get('candidate_swiglu_device_q8_swiglu_effective_io_gb_s')} |",
      "| SwiGLU device-Q8 quantize | "
      f"{timings.get('candidate_swiglu_device_q8_quantize_min_us')} | "
      f"{timings.get('candidate_swiglu_device_q8_quantize_mean_us')} | "
      f"{timings.get('candidate_swiglu_device_q8_quantize_effective_raw_gb_s')} | "
      f"{timings.get('candidate_swiglu_device_q8_quantize_effective_io_gb_s')} |",
      "| SwiGLU device-Q8 down | "
      f"{timings.get('candidate_swiglu_device_q8_down_kernel_min_us')} | "
      f"{timings.get('candidate_swiglu_device_q8_down_kernel_mean_us')} | "
      f"{timings.get('candidate_swiglu_device_q8_down_effective_raw_gb_s')} | "
      f"{timings.get('candidate_swiglu_device_q8_down_effective_io_gb_s')} |",
      "| SwiGLU device-Q8 shell sum | "
      f"{timings.get('candidate_swiglu_device_q8_shell_sum_min_us')} | "
      f"{timings.get('candidate_swiglu_device_q8_shell_sum_mean_us')} | "
      "None | None |",
      "| SwiGLU f32-input swiglu | "
      f"{timings.get('candidate_swiglu_f32_input_swiglu_min_us')} | "
      f"{timings.get('candidate_swiglu_f32_input_swiglu_mean_us')} | "
      f"{timings.get('candidate_swiglu_f32_input_swiglu_effective_raw_gb_s')} | "
      f"{timings.get('candidate_swiglu_f32_input_swiglu_effective_io_gb_s')} |",
      "| SwiGLU f32-input down | "
      f"{timings.get('candidate_swiglu_f32_input_down_kernel_min_us')} | "
      f"{timings.get('candidate_swiglu_f32_input_down_kernel_mean_us')} | "
      f"{timings.get('candidate_swiglu_f32_input_down_effective_raw_gb_s')} | "
      f"{timings.get('candidate_swiglu_f32_input_down_effective_io_gb_s')} |",
      "| SwiGLU f32-input shell sum | "
      f"{timings.get('candidate_swiglu_f32_input_shell_sum_min_us')} | "
      f"{timings.get('candidate_swiglu_f32_input_shell_sum_mean_us')} | "
      "None | None |",
      "| Q4 selected+shared selected separate | "
      f"{timings.get('candidate_q4_selected_shared_selected_kernel_min_us')} | "
      f"{timings.get('candidate_q4_selected_shared_selected_kernel_mean_us')} | "
      "None | None |",
      "| Q4 selected+shared shared separate | "
      f"{timings.get('candidate_q4_selected_shared_shared_kernel_min_us')} | "
      f"{timings.get('candidate_q4_selected_shared_shared_kernel_mean_us')} | "
      "None | None |",
      "| Q4 selected+shared combined | "
      f"{timings.get('candidate_q4_selected_shared_combined_kernel_min_us')} | "
      f"{timings.get('candidate_q4_selected_shared_combined_kernel_mean_us')} | "
      f"{timings.get('candidate_q4_selected_shared_combined_effective_raw_gb_s')} | "
      f"{timings.get('candidate_q4_selected_shared_combined_effective_io_gb_s')} |",
      "| Q6 selected+shared selected separate | "
      f"{timings.get('candidate_q6_selected_rowstripe_kernel_min_us')} | "
      f"{timings.get('candidate_q6_selected_rowstripe_kernel_mean_us')} | "
      "None | None |",
      "| Q6 selected+shared shared separate | "
      f"{timings.get('candidate_q6_shared_raw_kernel_min_us')} | "
      f"{timings.get('candidate_q6_shared_raw_kernel_mean_us')} | "
      "None | None |",
      "| Q6 selected+shared combined | "
      f"{timings.get('candidate_q6_selected_shared_combined_kernel_min_us')} | "
      f"{timings.get('candidate_q6_selected_shared_combined_kernel_mean_us')} | "
      f"{timings.get('candidate_q6_selected_shared_combined_effective_raw_gb_s')} | "
      f"{timings.get('candidate_q6_selected_shared_combined_effective_io_gb_s')} |",
      "| Q6 non-atomic down-tail contribution | "
      f"{timings.get('candidate_q6_nonatomic_down_tail_contribution_kernel_min_us')} | "
      f"{timings.get('candidate_q6_nonatomic_down_tail_contribution_kernel_mean_us')} | "
      "None | None |",
      "| Q6 non-atomic down-tail reduce | "
      f"{timings.get('candidate_q6_nonatomic_down_tail_reduce_kernel_min_us')} | "
      f"{timings.get('candidate_q6_nonatomic_down_tail_reduce_kernel_mean_us')} | "
      "None | None |",
      "| Q6 non-atomic down-tail shell sum | "
      f"{timings.get('candidate_q6_nonatomic_down_tail_shell_sum_min_us')} | "
      f"{timings.get('candidate_q6_nonatomic_down_tail_shell_sum_mean_us')} | "
      "None | None |",
      "| Q6 rowgroup down-tail | "
      f"{timings.get('candidate_q6_rowgroup_down_tail_kernel_min_us')} | "
      f"{timings.get('candidate_q6_rowgroup_down_tail_kernel_mean_us')} | "
      f"{timings.get('candidate_q6_rowgroup_down_tail_effective_raw_gb_s')} | "
      f"{timings.get('candidate_q6_rowgroup_down_tail_effective_io_gb_s')} |",
      "",
      "The probe reads only the top-k expert slices and uses a separate Q8_K",
      "activation for each selected expert. The pair2, weighted,",
      "weighted-sum, group8-sum, occupancy, device-Q8, SwiGLU device-Q8,",
      "SwiGLU f32-input, Q4 selected+shared, Q6 selected+shared, Q6 non-atomic down-tail, and Q6 rowgroup down-tail checks are appended only inside this probe.",
      "This is component evidence only; it",
      "does not prove decode or model throughput.",
      "",
  ]
  path.write_text("\n".join(lines), encoding="utf-8")


def strip_opencl_kernel(source: str, kernel_name: str) -> str:
  marker = f"__kernel void {kernel_name}("
  start = source.find(marker)
  if start < 0:
    return source
  brace = source.find("{", start)
  if brace < 0:
    return source
  depth = 0
  end = -1
  for index in range(brace, len(source)):
    char = source[index]
    if char == "{":
      depth += 1
    elif char == "}":
      depth -= 1
      if depth == 0:
        end = index + 1
        break
  if end < 0:
    return source
  return source[:start].rstrip() + "\n\n" + source[end:].lstrip()


def merged_opencl_source() -> str:
  engine_source = OPENCL_SOURCE.read_text(encoding="utf-8")
  extra_source = SELECTED_Q4_PAIR2_OPENCL
  for kernel_name in (
      "q6k_selected_down_matvec_rowstripe_expert8_plus_shared_tail_contrib_raw",
      "q6k_selected_down_matvec_rowstripe_expert8_plus_shared_tail_rowgroup_reduce_raw",
      "ffn_tail_reduce9_contrib_f32",
  ):
    if f"__kernel void {kernel_name}(" in engine_source:
      extra_source = strip_opencl_kernel(extra_source, kernel_name)
  return engine_source + extra_source


def main() -> int:
  args = parse_args()
  stamp = utc_stamp()
  created_at = iso_now()
  out_dir = args.out_dir or ROOT / f"output/gpu-selected-down-probe-{stamp}"
  out_dir.mkdir(parents=True, exist_ok=False)
  raw_dir = out_dir / "raw"
  raw_dir.mkdir()
  payloads = resolve_payloads(args.layer)
  opencl_source = merged_opencl_source()
  opencl_source_hash = iq36_local.sha256_file(OPENCL_SOURCE)
  local_cpp = out_dir / "gpu_selected_down_probe.cpp"
  local_cpp.write_text(
      PROBE_CPP.replace("@@OPENCL_SOURCE_LITERAL@@", cpp_raw_string_literal(opencl_source)),
      encoding="utf-8",
  )

  remote_dir = f"{args.remote_root.rstrip('/')}/gpu-selected-down-probe-{stamp}"
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
    transfers.append(iq36_local.copy_to(args.host, local_cpp, f"{remote_dir}/tests/gpu_selected_down_probe.cpp", args.timeout_s))
    for name, payload in payloads.items():
      payload_transfers[name] = iq36_local.copy_to(
          args.host,
          payload["local_path"],
          f"{remote_payload_dir}/{payload['stage_name']}",
          args.timeout_s,
      )

  compile_cmd = " && ".join([
      f"source {shlex.quote(args.env_script)} >/dev/null 2>&1",
      (
          "g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic "
          f"-I {shlex.quote(remote_dir + '/include')} "
          f"{shlex.quote(remote_dir + '/src/gguf_loader.cpp')} "
          f"{shlex.quote(remote_dir + '/src/gpu_q4x8_matvec.cpp')} "
          f"{shlex.quote(remote_dir + '/tests/gpu_selected_down_probe.cpp')} "
          "-ldl -pthread "
          f"-o {shlex.quote(remote_dir + '/build/iq36-gpu-selected-down-probe')}"
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
      f"{remote_dir}/build/iq36-gpu-selected-down-probe",
      "--model", args.model,
      "--payload-dir", remote_payload_dir,
      "--layer", str(args.layer),
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
      {"name": "selected_down_matches_oracle", "pass": bool(probe and nested_bool(probe, "checks", "selected_down_matches_oracle"))},
      {"name": "candidate_pair2_matches_current_and_oracle", "pass": bool(probe and nested_bool(probe, "checks", "candidate_pair2_matches_current_and_oracle"))},
      {"name": "candidate_weighted_matches_current_and_oracle", "pass": bool(probe and nested_bool(probe, "checks", "candidate_weighted_matches_current_and_oracle"))},
      {"name": "candidate_weighted_sum_matches_current_and_oracle", "pass": bool(probe and nested_bool(probe, "checks", "candidate_weighted_sum_matches_current_and_oracle"))},
      {"name": "candidate_group8_sum_matches_current_and_oracle", "pass": bool(probe and nested_bool(probe, "checks", "candidate_group8_sum_matches_current_and_oracle"))},
      {"name": "candidate_occupancy_matches_current_and_oracle", "pass": bool(probe and nested_bool(probe, "checks", "candidate_occupancy_matches_current_and_oracle"))},
      {"name": "candidate_device_q8_matches_current_and_oracle", "pass": bool(probe and nested_bool(probe, "checks", "candidate_device_q8_matches_current_and_oracle"))},
      {"name": "candidate_device_q8_planes_match_cpu", "pass": bool(probe and nested_bool(probe, "checks", "candidate_device_q8_planes_match_cpu"))},
      {"name": "candidate_swiglu_device_q8_matches_current_and_oracle", "pass": bool(probe and nested_bool(probe, "checks", "candidate_swiglu_device_q8_matches_current_and_oracle"))},
      {"name": "candidate_swiglu_device_q8_planes_match_cpu", "pass": bool(probe and nested_bool(probe, "checks", "candidate_swiglu_device_q8_planes_match_cpu"))},
      {"name": "candidate_swiglu_f32_input_comparison_recorded", "pass": bool(probe and nested_bool(probe, "checks", "candidate_swiglu_f32_input_comparison_recorded"))},
      {"name": "candidate_q4_selected_shared_matches_current_and_oracle", "pass": bool(probe and nested_bool(probe, "checks", "candidate_q4_selected_shared_matches_current_and_oracle"))},
      {"name": "candidate_q6_selected_shared_matches_current_and_oracle", "pass": bool(probe and nested_bool(probe, "checks", "candidate_q6_selected_shared_matches_current_and_oracle"))},
      {"name": "candidate_q6_nonatomic_down_tail_matches_current_and_oracle", "pass": bool(probe and nested_bool(probe, "checks", "candidate_q6_nonatomic_down_tail_matches_current_and_oracle"))},
      {"name": "candidate_q6_rowgroup_down_tail_matches_current_and_oracle", "pass": bool(probe and nested_bool(probe, "checks", "candidate_q6_rowgroup_down_tail_matches_current_and_oracle"))},
      {"name": "gpu_event_timing_positive", "pass": bool(probe and nested_bool(probe, "checks", "gpu_event_timing_positive"))},
      {"name": "candidate_pair2_event_timing_positive", "pass": bool(probe and nested_bool(probe, "checks", "candidate_pair2_event_timing_positive"))},
      {"name": "candidate_weighted_event_timing_positive", "pass": bool(probe and nested_bool(probe, "checks", "candidate_weighted_event_timing_positive"))},
      {"name": "candidate_weighted_sum_event_timing_positive", "pass": bool(probe and nested_bool(probe, "checks", "candidate_weighted_sum_event_timing_positive"))},
      {"name": "candidate_group8_sum_event_timing_positive", "pass": bool(probe and nested_bool(probe, "checks", "candidate_group8_sum_event_timing_positive"))},
      {"name": "candidate_occupancy_event_timing_positive", "pass": bool(probe and nested_bool(probe, "checks", "candidate_occupancy_event_timing_positive"))},
      {"name": "candidate_device_q8_event_timing_positive", "pass": bool(probe and nested_bool(probe, "checks", "candidate_device_q8_event_timing_positive"))},
      {"name": "candidate_swiglu_device_q8_event_timing_positive", "pass": bool(probe and nested_bool(probe, "checks", "candidate_swiglu_device_q8_event_timing_positive"))},
      {"name": "candidate_swiglu_f32_input_event_timing_positive", "pass": bool(probe and nested_bool(probe, "checks", "candidate_swiglu_f32_input_event_timing_positive"))},
      {"name": "candidate_q4_selected_shared_event_timing_positive", "pass": bool(probe and nested_bool(probe, "checks", "candidate_q4_selected_shared_event_timing_positive"))},
      {"name": "candidate_q6_selected_shared_event_timing_positive", "pass": bool(probe and nested_bool(probe, "checks", "candidate_q6_selected_shared_event_timing_positive"))},
      {"name": "candidate_q6_nonatomic_down_tail_event_timing_positive", "pass": bool(probe and nested_bool(probe, "checks", "candidate_q6_nonatomic_down_tail_event_timing_positive"))},
      {"name": "candidate_q6_rowgroup_down_tail_event_timing_positive", "pass": bool(probe and nested_bool(probe, "checks", "candidate_q6_rowgroup_down_tail_event_timing_positive"))},
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
      "repeat": args.repeat,
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
      "tool": "tools/intel-qwen36-gpu-selected-down-probe.py",
      "artifact": str(out_dir),
      "host": args.host,
      "remote_dir": remote_dir,
      "layer": args.layer,
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
      "gpu_selected_down_probe",
      [
          ("required_checks_passed", required_checks_passed),
          ("selected_down_kernel_min_us", nested_number(timings, "selected_down_gpu_kernel_min_us")),
          ("selected_down_effective_raw_gb_s", nested_number(timings, "selected_down_gpu_effective_raw_gb_s")),
          ("candidate_pair2_kernel_min_us", nested_number(timings, "candidate_pair2_kernel_min_us")),
          ("candidate_pair2_effective_raw_gb_s", nested_number(timings, "candidate_pair2_effective_raw_gb_s")),
          ("candidate_pair2_speedup_vs_current", nested_number(aggregate, "candidate_pair2_speedup_vs_current")),
          ("candidate_pair2_material_component_speedup", aggregate.get("candidate_pair2_material_component_speedup")),
          ("candidate_weighted_kernel_min_us", nested_number(timings, "candidate_weighted_kernel_min_us")),
          ("candidate_weighted_effective_raw_gb_s", nested_number(timings, "candidate_weighted_effective_raw_gb_s")),
          ("candidate_weighted_speedup_vs_current", nested_number(aggregate, "candidate_weighted_speedup_vs_current")),
          ("candidate_weighted_boundary_reopen_candidate", aggregate.get("candidate_weighted_boundary_reopen_candidate")),
          ("candidate_weighted_sum_kernel_min_us", nested_number(timings, "candidate_weighted_sum_kernel_min_us")),
          ("candidate_weighted_sum_effective_raw_gb_s", nested_number(timings, "candidate_weighted_sum_effective_raw_gb_s")),
          ("candidate_weighted_sum_speedup_vs_current", nested_number(aggregate, "candidate_weighted_sum_speedup_vs_current")),
          ("candidate_weighted_sum_speedup_vs_weighted", nested_number(aggregate, "candidate_weighted_sum_speedup_vs_weighted")),
          ("candidate_weighted_sum_boundary_reopen_candidate", aggregate.get("candidate_weighted_sum_boundary_reopen_candidate")),
          ("candidate_group8_sum_kernel_min_us", nested_number(timings, "candidate_group8_sum_kernel_min_us")),
          ("candidate_group8_sum_effective_raw_gb_s", nested_number(timings, "candidate_group8_sum_effective_raw_gb_s")),
          ("candidate_group8_sum_speedup_vs_current", nested_number(aggregate, "candidate_group8_sum_speedup_vs_current")),
          ("candidate_group8_sum_speedup_vs_weighted", nested_number(aggregate, "candidate_group8_sum_speedup_vs_weighted")),
          ("candidate_group8_sum_speedup_vs_serial_sum", nested_number(aggregate, "candidate_group8_sum_speedup_vs_serial_sum")),
          ("candidate_group8_sum_boundary_reopen_candidate", aggregate.get("candidate_group8_sum_boundary_reopen_candidate")),
          ("candidate_occupancy_kernel_min_us", nested_number(timings, "candidate_occupancy_kernel_min_us")),
          ("candidate_occupancy_effective_raw_gb_s", nested_number(timings, "candidate_occupancy_effective_raw_gb_s")),
          ("candidate_occupancy_scaled_speedup_vs_current", nested_number(aggregate, "candidate_occupancy_scaled_speedup_vs_current")),
          ("candidate_occupancy_effective_single_group_us", nested_number(aggregate, "candidate_occupancy_effective_single_group_us")),
          ("selected_shape_occupancy_bound", aggregate.get("selected_shape_occupancy_bound")),
          ("candidate_device_q8_quantize_min_us", nested_number(timings, "candidate_device_q8_quantize_min_us")),
          ("candidate_device_q8_down_kernel_min_us", nested_number(timings, "candidate_device_q8_down_kernel_min_us")),
          ("candidate_device_q8_shell_sum_min_us", nested_number(timings, "candidate_device_q8_shell_sum_min_us")),
          ("candidate_device_q8_down_speedup_vs_current", nested_number(aggregate, "candidate_device_q8_down_speedup_vs_current")),
          ("candidate_device_q8_shell_ratio_vs_current", nested_number(aggregate, "candidate_device_q8_shell_ratio_vs_current")),
          ("candidate_device_q8_boundary_reopen_candidate", aggregate.get("candidate_device_q8_boundary_reopen_candidate")),
          ("candidate_swiglu_device_q8_swiglu_min_us", nested_number(timings, "candidate_swiglu_device_q8_swiglu_min_us")),
          ("candidate_swiglu_device_q8_quantize_min_us", nested_number(timings, "candidate_swiglu_device_q8_quantize_min_us")),
          ("candidate_swiglu_device_q8_down_kernel_min_us", nested_number(timings, "candidate_swiglu_device_q8_down_kernel_min_us")),
          ("candidate_swiglu_device_q8_shell_sum_min_us", nested_number(timings, "candidate_swiglu_device_q8_shell_sum_min_us")),
          ("candidate_swiglu_device_q8_down_speedup_vs_current", nested_number(aggregate, "candidate_swiglu_device_q8_down_speedup_vs_current")),
          ("candidate_swiglu_device_q8_shell_ratio_vs_current", nested_number(aggregate, "candidate_swiglu_device_q8_shell_ratio_vs_current")),
          ("candidate_swiglu_device_q8_boundary_reopen_candidate", aggregate.get("candidate_swiglu_device_q8_boundary_reopen_candidate")),
          ("candidate_swiglu_f32_input_swiglu_min_us", nested_number(timings, "candidate_swiglu_f32_input_swiglu_min_us")),
          ("candidate_swiglu_f32_input_down_kernel_min_us", nested_number(timings, "candidate_swiglu_f32_input_down_kernel_min_us")),
          ("candidate_swiglu_f32_input_shell_sum_min_us", nested_number(timings, "candidate_swiglu_f32_input_shell_sum_min_us")),
          ("candidate_swiglu_f32_input_down_speedup_vs_current", nested_number(aggregate, "candidate_swiglu_f32_input_down_speedup_vs_current")),
          ("candidate_swiglu_f32_input_shell_ratio_vs_current", nested_number(aggregate, "candidate_swiglu_f32_input_shell_ratio_vs_current")),
          ("candidate_swiglu_f32_input_output_oracle_compatible", aggregate.get("candidate_swiglu_f32_input_output_oracle_compatible")),
          ("candidate_swiglu_f32_input_boundary_reopen_candidate", aggregate.get("candidate_swiglu_f32_input_boundary_reopen_candidate")),
          ("candidate_q4_selected_shared_selected_kernel_min_us", nested_number(timings, "candidate_q4_selected_shared_selected_kernel_min_us")),
          ("candidate_q4_selected_shared_shared_kernel_min_us", nested_number(timings, "candidate_q4_selected_shared_shared_kernel_min_us")),
          ("candidate_q4_selected_shared_combined_kernel_min_us", nested_number(timings, "candidate_q4_selected_shared_combined_kernel_min_us")),
          ("candidate_q4_selected_shared_combined_speedup_vs_separate", nested_number(aggregate, "candidate_q4_selected_shared_combined_speedup_vs_separate")),
          ("candidate_q4_selected_shared_boundary_reopen_candidate", aggregate.get("candidate_q4_selected_shared_boundary_reopen_candidate")),
          ("candidate_q6_selected_rowstripe_kernel_min_us", nested_number(timings, "candidate_q6_selected_rowstripe_kernel_min_us")),
          ("candidate_q6_shared_raw_kernel_min_us", nested_number(timings, "candidate_q6_shared_raw_kernel_min_us")),
          ("candidate_q6_selected_shared_combined_kernel_min_us", nested_number(timings, "candidate_q6_selected_shared_combined_kernel_min_us")),
          ("candidate_q6_selected_shared_combined_speedup_vs_separate", nested_number(aggregate, "candidate_q6_selected_shared_combined_speedup_vs_separate")),
          ("candidate_q6_selected_shared_boundary_reopen_candidate", aggregate.get("candidate_q6_selected_shared_boundary_reopen_candidate")),
          ("candidate_q6_nonatomic_down_tail_contribution_kernel_min_us", nested_number(timings, "candidate_q6_nonatomic_down_tail_contribution_kernel_min_us")),
          ("candidate_q6_nonatomic_down_tail_reduce_kernel_min_us", nested_number(timings, "candidate_q6_nonatomic_down_tail_reduce_kernel_min_us")),
          ("candidate_q6_nonatomic_down_tail_shell_sum_min_us", nested_number(timings, "candidate_q6_nonatomic_down_tail_shell_sum_min_us")),
          ("candidate_q6_nonatomic_down_tail_ratio_vs_combined_down", nested_number(aggregate, "candidate_q6_nonatomic_down_tail_ratio_vs_combined_down")),
          ("candidate_q6_nonatomic_down_tail_preserves_contributor_parallelism", aggregate.get("candidate_q6_nonatomic_down_tail_preserves_contributor_parallelism")),
          ("candidate_q6_nonatomic_down_tail_boundary_reopen_candidate", aggregate.get("candidate_q6_nonatomic_down_tail_boundary_reopen_candidate")),
          ("candidate_q6_nonatomic_down_tail_contribution_global_work_items", nested_number(timings, "candidate_q6_nonatomic_down_tail_contribution_global_work_items")),
          ("candidate_q6_nonatomic_down_tail_reduce_global_work_items", nested_number(timings, "candidate_q6_nonatomic_down_tail_reduce_global_work_items")),
          ("candidate_q6_rowgroup_down_tail_kernel_min_us", nested_number(timings, "candidate_q6_rowgroup_down_tail_kernel_min_us")),
          ("candidate_q6_rowgroup_down_tail_kernel_mean_us", nested_number(timings, "candidate_q6_rowgroup_down_tail_kernel_mean_us")),
          ("candidate_q6_rowgroup_down_tail_ratio_vs_combined_down", nested_number(aggregate, "candidate_q6_rowgroup_down_tail_ratio_vs_combined_down")),
          ("candidate_q6_rowgroup_down_tail_ratio_vs_nonatomic", nested_number(aggregate, "candidate_q6_rowgroup_down_tail_ratio_vs_nonatomic")),
          ("candidate_q6_rowgroup_down_tail_component_shell_target_us", nested_number(aggregate, "candidate_q6_rowgroup_down_tail_component_shell_target_us")),
          ("candidate_q6_rowgroup_down_tail_preserves_contributor_parallelism", aggregate.get("candidate_q6_rowgroup_down_tail_preserves_contributor_parallelism")),
          ("candidate_q6_rowgroup_down_tail_boundary_reopen_candidate", aggregate.get("candidate_q6_rowgroup_down_tail_boundary_reopen_candidate")),
          ("candidate_q6_rowgroup_down_tail_global_work_items", nested_number(timings, "candidate_q6_rowgroup_down_tail_global_work_items")),
          ("candidate_q6_rowgroup_down_tail_local_work_items", nested_number(timings, "candidate_q6_rowgroup_down_tail_local_work_items")),
          ("selected_down_gpu_vs_oracle_max_abs_diff", nested_number(comparisons, "selected_down", "gpu_vs_oracle", "max_abs_diff")),
          ("selected_down_gpu_vs_oracle_rmse", nested_number(comparisons, "selected_down", "gpu_vs_oracle", "rmse")),
          ("candidate_pair2_gpu_vs_oracle_max_abs_diff", nested_number(comparisons, "candidate_pair2", "gpu_vs_oracle", "max_abs_diff")),
          ("candidate_pair2_gpu_vs_oracle_rmse", nested_number(comparisons, "candidate_pair2", "gpu_vs_oracle", "rmse")),
          ("candidate_weighted_gpu_vs_oracle_max_abs_diff", nested_number(comparisons, "weighted_selected_down", "gpu_vs_oracle", "max_abs_diff")),
          ("candidate_weighted_gpu_vs_oracle_rmse", nested_number(comparisons, "weighted_selected_down", "gpu_vs_oracle", "rmse")),
          ("candidate_weighted_sum_gpu_vs_oracle_max_abs_diff", nested_number(comparisons, "weighted_sum_selected_down", "gpu_vs_oracle_weighted_sum", "max_abs_diff")),
          ("candidate_weighted_sum_gpu_vs_oracle_rmse", nested_number(comparisons, "weighted_sum_selected_down", "gpu_vs_oracle_weighted_sum", "rmse")),
          ("candidate_group8_sum_gpu_vs_oracle_max_abs_diff", nested_number(comparisons, "group8_sum_selected_down", "gpu_vs_oracle_weighted_sum", "max_abs_diff")),
          ("candidate_group8_sum_gpu_vs_oracle_rmse", nested_number(comparisons, "group8_sum_selected_down", "gpu_vs_oracle_weighted_sum", "rmse")),
          ("candidate_occupancy_gpu_vs_oracle_max_abs_diff", nested_number(comparisons, "selected_shape_occupancy", "occupancy4_vs_oracle", "max_abs_diff")),
          ("candidate_occupancy_gpu_vs_oracle_rmse", nested_number(comparisons, "selected_shape_occupancy", "occupancy4_vs_oracle", "rmse")),
          ("candidate_device_q8_gpu_vs_oracle_max_abs_diff", nested_number(comparisons, "device_q8_selected_down", "gpu_vs_oracle", "max_abs_diff")),
          ("candidate_device_q8_gpu_vs_oracle_rmse", nested_number(comparisons, "device_q8_selected_down", "gpu_vs_oracle", "rmse")),
          ("candidate_device_q8_qs_mismatch_count", nested_number(comparisons, "device_q8_selected_down", "q8_planes_vs_cpu", "qs_mismatch_count")),
          ("candidate_device_q8_bsums_mismatch_count", nested_number(comparisons, "device_q8_selected_down", "q8_planes_vs_cpu", "bsums_mismatch_count")),
          ("candidate_device_q8_d_max_abs_diff", nested_number(comparisons, "device_q8_selected_down", "q8_planes_vs_cpu", "d_max_abs_diff")),
          ("candidate_swiglu_device_q8_swiglu_max_abs_diff", nested_number(comparisons, "swiglu_device_q8_selected_down", "swiglu_vs_cpu", "max_abs_diff")),
          ("candidate_swiglu_device_q8_gpu_vs_oracle_max_abs_diff", nested_number(comparisons, "swiglu_device_q8_selected_down", "gpu_vs_oracle", "max_abs_diff")),
          ("candidate_swiglu_device_q8_gpu_vs_oracle_rmse", nested_number(comparisons, "swiglu_device_q8_selected_down", "gpu_vs_oracle", "rmse")),
          ("candidate_swiglu_device_q8_qs_mismatch_count", nested_number(comparisons, "swiglu_device_q8_selected_down", "q8_planes_vs_cpu", "qs_mismatch_count")),
          ("candidate_swiglu_device_q8_bsums_mismatch_count", nested_number(comparisons, "swiglu_device_q8_selected_down", "q8_planes_vs_cpu", "bsums_mismatch_count")),
          ("candidate_swiglu_device_q8_d_max_abs_diff", nested_number(comparisons, "swiglu_device_q8_selected_down", "q8_planes_vs_cpu", "d_max_abs_diff")),
          ("candidate_swiglu_f32_input_swiglu_max_abs_diff", nested_number(comparisons, "swiglu_f32_input_selected_down", "swiglu_vs_cpu", "max_abs_diff")),
          ("candidate_swiglu_f32_input_gpu_vs_oracle_max_abs_diff", nested_number(comparisons, "swiglu_f32_input_selected_down", "gpu_vs_oracle", "max_abs_diff")),
          ("candidate_swiglu_f32_input_gpu_vs_oracle_rmse", nested_number(comparisons, "swiglu_f32_input_selected_down", "gpu_vs_oracle", "rmse")),
          ("candidate_swiglu_f32_input_gpu_vs_current_max_abs_diff", nested_number(comparisons, "swiglu_f32_input_selected_down", "gpu_vs_current", "max_abs_diff")),
          ("candidate_q4_selected_combined_vs_oracle_max_abs_diff", nested_number(comparisons, "q4_selected_shared_down", "selected_combined_vs_oracle", "max_abs_diff")),
          ("candidate_q4_selected_combined_vs_oracle_rmse", nested_number(comparisons, "q4_selected_shared_down", "selected_combined_vs_oracle", "rmse")),
          ("candidate_q4_shared_combined_vs_oracle_max_abs_diff", nested_number(comparisons, "q4_selected_shared_down", "shared_combined_vs_oracle", "max_abs_diff")),
          ("candidate_q4_shared_combined_vs_oracle_rmse", nested_number(comparisons, "q4_selected_shared_down", "shared_combined_vs_oracle", "rmse")),
          ("candidate_q6_selected_combined_vs_oracle_max_abs_diff", nested_number(comparisons, "q6_selected_shared_down", "selected_combined_vs_oracle", "max_abs_diff")),
          ("candidate_q6_selected_combined_vs_oracle_rmse", nested_number(comparisons, "q6_selected_shared_down", "selected_combined_vs_oracle", "rmse")),
          ("candidate_q6_shared_combined_vs_oracle_max_abs_diff", nested_number(comparisons, "q6_selected_shared_down", "shared_combined_vs_oracle", "max_abs_diff")),
          ("candidate_q6_shared_combined_vs_oracle_rmse", nested_number(comparisons, "q6_selected_shared_down", "shared_combined_vs_oracle", "rmse")),
          ("candidate_q6_nonatomic_down_tail_contrib_vs_cpu_max_abs_diff", nested_number(comparisons, "q6_nonatomic_down_tail", "contrib_vs_cpu", "max_abs_diff")),
          ("candidate_q6_nonatomic_down_tail_contrib_vs_cpu_rmse", nested_number(comparisons, "q6_nonatomic_down_tail", "contrib_vs_cpu", "rmse")),
          ("candidate_q6_nonatomic_down_tail_layer_vs_oracle_max_abs_diff", nested_number(comparisons, "q6_nonatomic_down_tail", "layer_vs_oracle", "max_abs_diff")),
          ("candidate_q6_nonatomic_down_tail_layer_vs_oracle_rmse", nested_number(comparisons, "q6_nonatomic_down_tail", "layer_vs_oracle", "rmse")),
          ("candidate_q6_rowgroup_down_tail_layer_vs_oracle_max_abs_diff", nested_number(comparisons, "q6_rowgroup_down_tail", "layer_vs_oracle", "max_abs_diff")),
          ("candidate_q6_rowgroup_down_tail_layer_vs_oracle_rmse", nested_number(comparisons, "q6_rowgroup_down_tail", "layer_vs_oracle", "rmse")),
          ("candidate_q6_rowgroup_down_tail_layer_vs_nonatomic_max_abs_diff", nested_number(comparisons, "q6_rowgroup_down_tail", "layer_vs_nonatomic", "max_abs_diff")),
          ("candidate_q6_rowgroup_down_tail_layer_vs_nonatomic_rmse", nested_number(comparisons, "q6_rowgroup_down_tail", "layer_vs_nonatomic", "rmse")),
      ],
  )
  write_summary(out_dir / "summary.md", payload)
  print(out_dir)
  return 0 if required_checks_passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
