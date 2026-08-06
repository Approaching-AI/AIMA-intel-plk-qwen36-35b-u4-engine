#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable

float half_to_float(ushort h) {
  return convert_float(as_half(h));
}

ushort load_le16(__global const uchar* p) {
  return (ushort)p[0] | ((ushort)p[1] << 8);
}

uint load_le32(__global const uchar* p) {
  return (uint)p[0] | ((uint)p[1] << 8) |
         ((uint)p[2] << 16) | ((uint)p[3] << 24);
}

uchar byte_from_word(uint value, uint index) {
  return (uchar)((value >> (index * 8U)) & 0xFFU);
}

void get_scale_min_k4(int index, __global const uchar* scales, uchar* scale, uchar* minimum) {
  if (index < 4) {
    *scale = scales[index] & (uchar)63;
    *minimum = scales[index + 4] & (uchar)63;
  } else {
    *scale = (scales[index + 4] & (uchar)15) |
             (uchar)((scales[index - 4] >> 6) << 4);
    *minimum = (scales[index + 4] >> 4) |
               (uchar)((scales[index] >> 6) << 4);
  }
}

void get_scale_min_k4_private(int index, const uchar* scales,
                              uchar* scale, uchar* minimum) {
  if (index < 4) {
    *scale = scales[index] & (uchar)63;
    *minimum = scales[index + 4] & (uchar)63;
  } else {
    *scale = (scales[index + 4] & (uchar)15) |
             (uchar)((scales[index - 4] >> 6) << 4);
    *minimum = (scales[index + 4] >> 4) |
               (uchar)((scales[index] >> 6) << 4);
  }
}

inline int q4_q8_dot8_global(__global const uchar* packed,
                             __global const char* q8,
                             uint nibble_shift) {
#if defined(IQ36_USE_INTEGER_DOT)
  const uchar4 packed0 = vload4(0, packed);
  const uchar4 packed1 = vload4(0, packed + 4);
  const uchar4 mask = (uchar4)(15);
  const char4 q40 = convert_char4((packed0 >> nibble_shift) & mask);
  const char4 q41 = convert_char4((packed1 >> nibble_shift) & mask);
  return dot(q40, vload4(0, q8)) + dot(q41, vload4(0, q8 + 4));
#else
  int sum = 0;
  for (uint index = 0; index < 8U; ++index) {
    sum += (int)((packed[index] >> nibble_shift) & (uchar)15) *
           (int)q8[index];
  }
  return sum;
#endif
}

inline int q4_q8_dot8_local(__global const uchar* packed,
                            __local const char* q8,
                            uint nibble_shift) {
#if defined(IQ36_USE_INTEGER_DOT)
  const uchar4 packed0 = vload4(0, packed);
  const uchar4 packed1 = vload4(0, packed + 4);
  const uchar4 mask = (uchar4)(15);
  const char4 q40 = convert_char4((packed0 >> nibble_shift) & mask);
  const char4 q41 = convert_char4((packed1 >> nibble_shift) & mask);
  return dot(q40, vload4(0, q8)) + dot(q41, vload4(0, q8 + 4));
#else
  int sum = 0;
  for (uint index = 0; index < 8U; ++index) {
    sum += (int)((packed[index] >> nibble_shift) & (uchar)15) *
           (int)q8[index];
  }
  return sum;
#endif
}

inline int i8_i8_dot4_global(__global const char* lhs,
                             __global const char* rhs) {
#if defined(IQ36_USE_INTEGER_DOT)
  return dot(vload4(0, lhs), vload4(0, rhs));
#else
  int sum = 0;
  for (uint index = 0; index < 4U; ++index) {
    sum += (int)lhs[index] * (int)rhs[index];
  }
  return sum;
#endif
}

inline float q4s_group128_rowstripe8_dot_f32(
    __global const uchar* packed,
    __global const char* q8_qs,
    __global const float* q8_d,
    uint group_count,
    ulong row) {
  const ulong row_group = row >> 3U;
  const uint row_lane = (uint)(row & 7UL);
  float sum = 0.0f;
  for (uint group = 0U; group < group_count; ++group) {
    __global const uchar* block = packed +
        (row_group * (ulong)group_count + group) * 528UL;
    __global const char* input = q8_qs + group * 128U;
    int dot_sum = 0;
    for (uint chunk = 0U; chunk < 16U; ++chunk) {
      const uchar4 codes = vload4(
          0, block + chunk * 32U + row_lane * 4U);
      const char4 low = convert_char4(codes & (uchar4)(15)) - (char4)(8);
      const char4 high = convert_char4(codes >> 4U) - (char4)(8);
#if defined(IQ36_USE_INTEGER_DOT)
      dot_sum += dot(low, vload4(0, input + chunk * 4U));
      dot_sum += dot(high, vload4(0, input + 64U + chunk * 4U));
#else
      for (uint element = 0U; element < 4U; ++element) {
        dot_sum += (int)low[element] * (int)input[chunk * 4U + element];
        dot_sum += (int)high[element] *
            (int)input[64U + chunk * 4U + element];
      }
#endif
    }
    const float weight_scale = vload_half(
        0, (__global const half*)(block + 512U + row_lane * 2U));
    sum += (float)dot_sum * weight_scale * q8_d[group >> 1U];
  }
  return sum;
}

inline float q4s_group64_rowstripe8_dot_f32(
    __global const uchar* packed,
    __global const char* q8_qs,
    __global const float* q8_d,
    uint group_count,
    ulong row) {
  const ulong row_group = row >> 3U;
  const uint row_lane = (uint)(row & 7UL);
  float sum = 0.0f;
  for (uint group = 0U; group < group_count; ++group) {
    __global const uchar* block = packed +
        (row_group * (ulong)group_count + group) * 288UL;
    __global const char* input = q8_qs + group * 64U;
    int dot_sum = 0;
    for (uint chunk = 0U; chunk < 8U; ++chunk) {
      const uchar4 codes = vload4(
          0, block + chunk * 32U + row_lane * 4U);
      const char4 low = convert_char4(codes & (uchar4)(15)) - (char4)(8);
      const char4 high = convert_char4(codes >> 4U) - (char4)(8);
#if defined(IQ36_USE_INTEGER_DOT)
      dot_sum += dot(low, vload4(0, input + chunk * 4U));
      dot_sum += dot(high, vload4(0, input + 32U + chunk * 4U));
#else
      for (uint element = 0U; element < 4U; ++element) {
        dot_sum += (int)low[element] * (int)input[chunk * 4U + element];
        dot_sum += (int)high[element] *
            (int)input[32U + chunk * 4U + element];
      }
#endif
    }
    const float weight_scale =
        *((__global const float*)(block + 256U + row_lane * 4U));
    sum += (float)dot_sum * weight_scale * q8_d[group >> 2U];
  }
  return sum;
}

inline float q4s_group32_rowstripe8_dot_f32(
    __global const uchar* packed,
    __global const char* q8_qs,
    __global const float* q8_d,
    uint group_count,
    ulong row) {
  const ulong row_group = row >> 3U;
  const uint row_lane = (uint)(row & 7UL);
  float sum = 0.0f;
  for (uint group = 0U; group < group_count; ++group) {
    __global const uchar* block = packed +
        (row_group * (ulong)group_count + group) * 160UL;
    __global const char* input = q8_qs + group * 32U;
    int dot_sum = 0;
    for (uint chunk = 0U; chunk < 4U; ++chunk) {
      const uchar4 codes = vload4(
          0, block + chunk * 32U + row_lane * 4U);
      const char4 low = convert_char4(codes & (uchar4)(15)) - (char4)(8);
      const char4 high = convert_char4(codes >> 4U) - (char4)(8);
#if defined(IQ36_USE_INTEGER_DOT)
      dot_sum += dot(low, vload4(0, input + chunk * 4U));
      dot_sum += dot(high, vload4(0, input + 16U + chunk * 4U));
#else
      for (uint element = 0U; element < 4U; ++element) {
        dot_sum += (int)low[element] * (int)input[chunk * 4U + element];
        dot_sum += (int)high[element] *
            (int)input[16U + chunk * 4U + element];
      }
#endif
    }
    const float weight_scale =
        *((__global const float*)(block + 128U + row_lane * 4U));
    sum += (float)dot_sum * weight_scale * q8_d[group >> 3U];
  }
  return sum;
}

// Locked OpenVINO LM-head decode codec.  Eight adjacent output rows share one
// stripe.  Each stripe contains sixteen coalesced 128-column Q4 code planes,
// followed by eight original per-row F16 scales and 16 bytes of alignment
// padding.  The signed Q4 code is round-to-even(i8 / 16), clamped to [-8, 7].
// Activation codes/scales match the provider's half group-256 symmetric DQ.
#define IQ36_LM_HEAD_I8Q4_ROWS 248320U
#define IQ36_LM_HEAD_I8Q4_COLS 2048U
#define IQ36_LM_HEAD_I8Q4_GROUPS 16U
#define IQ36_LM_HEAD_I8Q4_CODE_BYTES_PER_GROUP8 512U
#define IQ36_LM_HEAD_I8Q4_CODE_BYTES_PER_STRIPE 8192U
#define IQ36_LM_HEAD_I8Q4_STRIPE_BYTES 8224U

__attribute__((reqd_work_group_size(64, 1, 1)))
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void iq36_lm_head_q8_group256_f16(
    __global const float* input,
    __global char* q8_qs,
    __global float* q8_d) {
  const uint group = (uint)get_group_id(0);
  const uint lane = (uint)get_local_id(0);
  const uint base = group * 256U + lane * 4U;
  const half4 values = convert_half4_rte(vload4(0, input + base));
  half maximum = fmax(fmax(fabs(values[0]), fabs(values[1])),
                      fmax(fabs(values[2]), fabs(values[3])));
  __local half maxima[64];
  __local half quantize_scale;
  maxima[lane] = maximum;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lane == 0U) {
    half group_maximum = (half)0.0h;
    for (uint index = 0U; index < 64U; ++index) {
      group_maximum = fmax(group_maximum, maxima[index]);
    }
    group_maximum = fmax(group_maximum, (half)0.00006103515625h);
    quantize_scale = (half)127.0h / group_maximum;
    q8_d[group] = convert_float((half)1.0h / quantize_scale);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  vstore4(convert_char4_rte(values * (half4)quantize_scale),
          0, q8_qs + base);
}

__attribute__((reqd_work_group_size(64, 1, 1)))
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void iq36_lm_head_q8_group256_f16_sums(
    __global const float* input,
    __global char* q8_qs,
    __global float* q8_d,
    __global int* q8_sums) {
  const uint group = (uint)get_group_id(0);
  const uint lane = (uint)get_local_id(0);
  const uint base = group * 256U + lane * 4U;
  const half4 values = convert_half4_rte(vload4(0, input + base));
  __local half maxima[64];
  __local half quantize_scale;
  __local int partial_sums[64];
  maxima[lane] = fmax(fmax(fabs(values[0]), fabs(values[1])),
                      fmax(fabs(values[2]), fabs(values[3])));
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lane == 0U) {
    half maximum = (half)0.00006103515625h;
    for (uint index = 0U; index < 64U; ++index)
      maximum = fmax(maximum, maxima[index]);
    quantize_scale = (half)127.0h / maximum;
    q8_d[group] = convert_float((half)1.0h / quantize_scale);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  const char4 quantized = convert_char4_rte(
      values * (half4)quantize_scale);
  vstore4(quantized, 0, q8_qs + base);
  partial_sums[lane] = (int)quantized.s0 + (int)quantized.s1 +
      (int)quantized.s2 + (int)quantized.s3;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lane == 0U) {
    int total = 0;
    for (uint index = 0U; index < 64U; ++index)
      total += partial_sums[index];
    q8_sums[group] = total;
  }
}

// Diagnostic binary two-centroid LM-head codec. Each eight-row stripe stores
// 2048 one-bit weights per row, two F32 centroids per row, and the original
// F16 row scale. The Q8 group sum turns {low,high} reconstruction into one
// bit-dot plus one scalar term per 256-column activation group.
#define IQ36_LM_HEAD_I8Q1_CODE_BYTES_PER_STRIPE 2048U
#define IQ36_LM_HEAD_I8Q1_STRIPE_BYTES 2128U

inline float iq36_lm_head_i8q1_rowstripe8_dot(
    __global const uchar* packed,
    __global const char* q8_qs,
    __global const float* q8_d,
    __global const int* q8_sums,
    ulong row) {
  const ulong row_group = row >> 3U;
  const uint row_lane = (uint)(row & 7UL);
  __global const uchar* stripe =
      packed + row_group * (ulong)IQ36_LM_HEAD_I8Q1_STRIPE_BYTES;
  const float low = *(__global const float*)(
      stripe + IQ36_LM_HEAD_I8Q1_CODE_BYTES_PER_STRIPE + row_lane * 4U);
  const float high = *(__global const float*)(
      stripe + IQ36_LM_HEAD_I8Q1_CODE_BYTES_PER_STRIPE + 32U +
      row_lane * 4U);
  float scaled_sum = 0.0f;
  for (uint group = 0U; group < 8U; ++group) {
    int high_sum = 0;
    for (uint chunk = 0U; chunk < 8U; ++chunk) {
      const uchar4 codes = vload4(
          0, stripe + (group * 8U + chunk) * 32U + row_lane * 4U);
      __global const char* input = q8_qs + group * 256U + chunk * 32U;
      for (uint byte = 0U; byte < 4U; ++byte) {
        const uchar bits = codes[byte];
        const char4 first = (char4)(
            (char)((bits >> 0U) & 1U), (char)((bits >> 1U) & 1U),
            (char)((bits >> 2U) & 1U), (char)((bits >> 3U) & 1U));
        const char4 second = (char4)(
            (char)((bits >> 4U) & 1U), (char)((bits >> 5U) & 1U),
            (char)((bits >> 6U) & 1U), (char)((bits >> 7U) & 1U));
        high_sum += dot(first, vload4(0, input + byte * 8U));
        high_sum += dot(second, vload4(0, input + byte * 8U + 4U));
      }
    }
    scaled_sum += (low * (float)q8_sums[group] +
                   (high - low) * (float)high_sum) * q8_d[group];
  }
  const float original_weight_scale = vload_half(
      0, (__global const half*)(
          stripe + IQ36_LM_HEAD_I8Q1_CODE_BYTES_PER_STRIPE + 64U +
          row_lane * 2U));
  return scaled_sum * original_weight_scale;
}

__attribute__((reqd_work_group_size(256, 1, 1)))
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void iq36_lm_head_i8q1_rowstripe8_matvec_f32(
    __global const uchar* packed,
    __global const char* q8_qs,
    __global const float* q8_d,
    __global const int* q8_sums,
    uint rows,
    __global float* output) {
  const uint row = (uint)get_global_id(0);
  if (row >= rows) return;
  output[row] = iq36_lm_head_i8q1_rowstripe8_dot(
      packed, q8_qs, q8_d, q8_sums, (ulong)row);
}

__attribute__((reqd_work_group_size(256, 1, 1)))
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void iq36_lm_head_i8q1_rowstripe8_matvec_local_top12_f32(
    __global const uchar* packed,
    __global const char* q8_qs,
    __global const float* q8_d,
    __global const int* q8_sums,
    uint rows,
    __global float* output,
    __global int* partial_top_ids) {
  const uint row = (uint)get_global_id(0);
  const uint lane = (uint)get_local_id(0);
  const uint block = (uint)get_group_id(0);
  const float value = row < rows
      ? iq36_lm_head_i8q1_rowstripe8_dot(
            packed, q8_qs, q8_d, q8_sums, (ulong)row)
      : -INFINITY;
  if (row < rows) output[row] = value;
  __local float values[256];
  values[lane] = value;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lane != 0U) return;
  float best_values[12];
  int best_ids[12];
  for (uint index = 0U; index < 12U; ++index) {
    best_values[index] = -INFINITY;
    best_ids[index] = -1;
  }
  const uint begin = block * 256U;
  const uint count = min(256U, rows - begin);
  for (uint index = 0U; index < count; ++index) {
    const float candidate = values[index];
    uint position = 0U;
    while (position < 12U &&
           (best_values[position] > candidate ||
            (best_values[position] == candidate &&
             best_ids[position] < (int)(begin + index)))) {
      ++position;
    }
    if (position < 12U) {
      for (uint destination = 11U; destination > position; --destination) {
        best_values[destination] = best_values[destination - 1U];
        best_ids[destination] = best_ids[destination - 1U];
      }
      best_values[position] = candidate;
      best_ids[position] = (int)(begin + index);
    }
  }
  for (uint index = 0U; index < 12U; ++index)
    partial_top_ids[block * 12U + index] = best_ids[index];
}

// Token-only decode does not need the distribution-quality local-top12
// correction set.  Keep the same full-row approximation for a conservative
// component comparison, but select only the two rows per 256-row block needed
// by the captured-product greedy-token gate.
__attribute__((reqd_work_group_size(256, 1, 1)))
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void iq36_lm_head_i8q1_rowstripe8_matvec_local_top2_f32(
    __global const uchar* packed,
    __global const char* q8_qs,
    __global const float* q8_d,
    __global const int* q8_sums,
    uint rows,
    __global float* output,
    __global int* partial_top_ids) {
  const uint row = (uint)get_global_id(0);
  const uint lane = (uint)get_local_id(0);
  const uint block = (uint)get_group_id(0);
  const float value = row < rows
      ? iq36_lm_head_i8q1_rowstripe8_dot(
            packed, q8_qs, q8_d, q8_sums, (ulong)row)
      : -INFINITY;
  if (row < rows) output[row] = value;
  __local float values[256];
  values[lane] = value;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lane != 0U) return;
  float best_values[2] = {-INFINITY, -INFINITY};
  int best_ids[2] = {-1, -1};
  const uint begin = block * 256U;
  const uint count = min(256U, rows - begin);
  for (uint index = 0U; index < count; ++index) {
    const float candidate = values[index];
    uint position = 0U;
    while (position < 2U &&
           (best_values[position] > candidate ||
            (best_values[position] == candidate &&
             best_ids[position] < (int)(begin + index)))) {
      ++position;
    }
    if (position < 2U) {
      if (position == 0U) {
        best_values[1] = best_values[0];
        best_ids[1] = best_ids[0];
      }
      best_values[position] = candidate;
      best_ids[position] = (int)(begin + index);
    }
  }
  partial_top_ids[block * 2U + 0U] = best_ids[0];
  partial_top_ids[block * 2U + 1U] = best_ids[1];
}

// Exact token-equivalent compact surface for the local-top2 path.  The first
// two rows in each block are recomputed from the original I8 weights.  The
// third approximate row is the fallback winner when exact correction lowers
// both selected values.  Keeping all three is therefore equivalent to
// overwriting local top-2 in the complete approximate vocabulary and taking
// argmax, without materializing that vocabulary.
__attribute__((reqd_work_group_size(256, 1, 1)))
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void iq36_lm_head_i8q1_rowstripe8_matvec_local_top3_compact_f32(
    __global const uchar* packed,
    __global const char* q8_qs,
    __global const float* q8_d,
    __global const int* q8_sums,
    uint rows,
    __global int* partial_top_ids,
    __global float* partial_top_values) {
  const uint row = (uint)get_global_id(0);
  const uint lane = (uint)get_local_id(0);
  const uint block = (uint)get_group_id(0);
  const float value = row < rows
      ? iq36_lm_head_i8q1_rowstripe8_dot(
            packed, q8_qs, q8_d, q8_sums, (ulong)row)
      : -INFINITY;
  __local float values[256];
  values[lane] = value;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lane != 0U) return;
  float best_values[3] = {-INFINITY, -INFINITY, -INFINITY};
  int best_ids[3] = {-1, -1, -1};
  const uint begin = block * 256U;
  const uint count = min(256U, rows - begin);
  for (uint index = 0U; index < count; ++index) {
    const float candidate = values[index];
    uint position = 0U;
    while (position < 3U &&
           (best_values[position] > candidate ||
            (best_values[position] == candidate &&
             best_ids[position] < (int)(begin + index)))) {
      ++position;
    }
    if (position < 3U) {
      for (uint destination = 2U; destination > position; --destination) {
        best_values[destination] = best_values[destination - 1U];
        best_ids[destination] = best_ids[destination - 1U];
      }
      best_values[position] = candidate;
      best_ids[position] = (int)(begin + index);
    }
  }
  for (uint index = 0U; index < 3U; ++index) {
    partial_top_ids[block * 3U + index] = best_ids[index];
    partial_top_values[block * 3U + index] = best_values[index];
  }
}

__attribute__((reqd_work_group_size(64, 1, 1)))
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void iq36_lm_head_i8_exact_local_top2_compact_correction_f32(
    __global const char* weights,
    __global const half* weight_scales,
    __global const char* q8_qs,
    __global const float* q8_d,
    __global const int* selected_ids,
    __global float* selected_values) {
  const uint selected = (uint)get_group_id(0);
  const uint lane = (uint)get_local_id(0);
  if (selected >= 1940U) return;
  const uint slot = (selected >> 1U) * 3U + (selected & 1U);
  const int signed_row = selected_ids[slot];
  if (signed_row < 0 || signed_row >= (int)IQ36_LM_HEAD_I8Q4_ROWS) return;
  const uint row = (uint)signed_row;
  const ulong row_base = (ulong)row * 2048UL;
  const uint lane_base = lane * 32U;
  int dot_sum = 0;
  for (uint chunk = 0U; chunk < 8U; ++chunk) {
    const uint element = lane_base + chunk * 4U;
    dot_sum += dot(vload4(0, weights + row_base + element),
                   vload4(0, q8_qs + element));
  }
  __local int partial[64];
  __local float group_sums[8];
  partial[lane] = dot_sum;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lane < 8U) {
    int group_sum = 0;
    for (uint contributor = 0U; contributor < 8U; ++contributor)
      group_sum += partial[lane * 8U + contributor];
    group_sums[lane] = (float)group_sum * q8_d[lane];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lane == 0U) {
    float value = 0.0f;
    for (uint group = 0U; group < 8U; ++group) value += group_sums[group];
    selected_values[slot] = value * convert_float(weight_scales[row]);
  }
}

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void iq36_lm_head_compact_top3_merge_top8_f32(
    __global const int* selected_ids,
    __global const float* selected_values,
    __global int* top_ids,
    __global float* top_values) {
  const uint lane = (uint)get_local_id(0);
  float best_values[8];
  int best_ids[8];
  for (uint index = 0U; index < 8U; ++index) {
    best_values[index] = -INFINITY;
    best_ids[index] = -1;
  }
  for (uint selected = lane; selected < 2910U; selected += 256U) {
    const float value = selected_values[selected];
    const int id = selected_ids[selected];
    uint position = 0U;
    while (position < 8U &&
           (best_values[position] > value ||
            (best_values[position] == value &&
             best_ids[position] < id))) {
      ++position;
    }
    if (position < 8U) {
      for (uint destination = 7U; destination > position; --destination) {
        best_values[destination] = best_values[destination - 1U];
        best_ids[destination] = best_ids[destination - 1U];
      }
      best_values[position] = value;
      best_ids[position] = id;
    }
  }
  uint head = 0U;
  for (uint output = 0U; output < 8U; ++output) {
    const float lane_value = head < 8U ? best_values[head] : -INFINITY;
    const int lane_id = head < 8U ? best_ids[head] : 0x7fffffff;
    const float maximum = work_group_reduce_max(lane_value);
    const int winner = work_group_reduce_min(
        lane_value == maximum ? lane_id : 0x7fffffff);
    if (lane_id == winner) ++head;
    if (lane == 0U) {
      top_ids[output] = winner;
      top_values[output] = maximum;
    }
  }
}

__attribute__((reqd_work_group_size(64, 1, 1)))
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void iq36_lm_head_i8_direct_compact_top8_correction_f16(
    __global const char* weights,
    __global const half* weight_scales,
    __global const float* input,
    __global const int* top_ids,
    __global float* top_values) {
  const uint selected = (uint)get_group_id(0);
  const uint lane = (uint)get_local_id(0);
  if (selected >= 8U) return;
  const int signed_row = top_ids[selected];
  if (signed_row < 0 || signed_row >= (int)IQ36_LM_HEAD_I8Q4_ROWS) return;
  const uint row = (uint)signed_row;
  const ulong row_base = (ulong)row * 2048UL;
  const uint lane_base = lane * 32U;
  float dot_sum = 0.0f;
  for (uint chunk = 0U; chunk < 8U; ++chunk) {
    const uint element = lane_base + chunk * 4U;
    const float4 hidden = convert_float4(
        convert_half4_rte(vload4(0, input + element)));
    dot_sum += dot(
        convert_float4(vload4(0, weights + row_base + element)), hidden);
  }
  __local float partial[64];
  partial[lane] = dot_sum;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lane == 0U) {
    float value = 0.0f;
    for (uint contributor = 0U; contributor < 64U; ++contributor)
      value += partial[contributor];
    top_values[selected] = convert_float(convert_half_rte(
        value * convert_float(weight_scales[row])));
  }
}

inline ulong iq36_lm_head_ordered_top1_key(float value, uint token_id) {
  const uint bits = as_uint(value);
  const uint ordered =
      (bits & 0x80000000U) ? ~bits : (bits ^ 0x80000000U);
  return (((ulong)ordered) << 32) | ((ulong)(~token_id));
}

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void iq36_lm_head_f32_greedy_partials(
    __global const float* logits,
    __global ulong* partials) {
  const uint lane = (uint)get_local_id(0);
  const uint group = (uint)get_group_id(0);
  const uint global_size = (uint)get_global_size(0);
  __local ulong keys[256];
  ulong best = 0UL;
  for (uint token = group * 256U + lane;
       token < IQ36_LM_HEAD_I8Q4_ROWS; token += global_size) {
    best = max(best, iq36_lm_head_ordered_top1_key(logits[token], token));
  }
  keys[lane] = best;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint stride = 128U; stride != 0U; stride >>= 1U) {
    if (lane < stride) keys[lane] = max(keys[lane], keys[lane + stride]);
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lane == 0U) partials[group] = keys[0];
}

__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void iq36_lm_head_greedy_merge64(
    __global const ulong* partials,
    __global int* token) {
  const uint lane = (uint)get_local_id(0);
  __local ulong keys[64];
  keys[lane] = partials[lane];
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint stride = 32U; stride != 0U; stride >>= 1U) {
    if (lane < stride) keys[lane] = max(keys[lane], keys[lane + stride]);
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lane == 0U) token[0] = (int)(~(uint)keys[0]);
}

__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void iq36_lm_head_top8_greedy(
    __global const int* top_ids,
    __global const float* top_values,
    __global int* token) {
  const uint lane = (uint)get_local_id(0);
  __local ulong keys[64];
  ulong key = 0UL;
  if (lane < 8U && top_ids[lane] >= 0) {
    key = iq36_lm_head_ordered_top1_key(
        top_values[lane], (uint)top_ids[lane]);
  }
  keys[lane] = key;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint stride = 32U; stride != 0U; stride >>= 1U) {
    if (lane < stride) keys[lane] = max(keys[lane], keys[lane + stride]);
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lane == 0U) token[0] = (int)(~(uint)keys[0]);
}

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void iq36_lm_head_compact_top3_greedy(
    __global const int* selected_ids,
    __global const float* selected_values,
    __global int* token) {
  const uint lane = (uint)get_local_id(0);
  __local ulong keys[256];
  ulong best = 0UL;
  for (uint selected = lane; selected < 2910U; selected += 256U) {
    const int signed_id = selected_ids[selected];
    if (signed_id >= 0) {
      best = max(best, iq36_lm_head_ordered_top1_key(
          selected_values[selected], (uint)signed_id));
    }
  }
  keys[lane] = best;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint stride = 128U; stride != 0U; stride >>= 1U) {
    if (lane < stride) keys[lane] = max(keys[lane], keys[lane + stride]);
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lane == 0U) token[0] = (int)(~(uint)keys[0]);
}

inline float iq36_lm_head_i8q4_rowstripe8_dot(
    __global const uchar* packed,
    __global const char* q8_qs,
    __global const float* q8_d,
    ulong row) {
  const ulong row_group = row >> 3U;
  const uint row_lane = (uint)(row & 7UL);
  __global const uchar* stripe =
      packed + row_group * (ulong)IQ36_LM_HEAD_I8Q4_STRIPE_BYTES;
  float scaled_sum = 0.0f;
  for (uint group = 0U; group < IQ36_LM_HEAD_I8Q4_GROUPS; ++group) {
    __global const uchar* block =
        stripe + group * IQ36_LM_HEAD_I8Q4_CODE_BYTES_PER_GROUP8;
    __global const char* input = q8_qs + group * 128U;
    int dot_sum = 0;
    for (uint chunk = 0U; chunk < 16U; ++chunk) {
      const uchar4 codes = vload4(
          0, block + chunk * 32U + row_lane * 4U);
      const char4 low = convert_char4(codes & (uchar4)(15)) - (char4)(8);
      const char4 high = convert_char4(codes >> 4U) - (char4)(8);
#if defined(IQ36_USE_INTEGER_DOT)
      dot_sum += dot(low, vload4(0, input + chunk * 4U));
      dot_sum += dot(high, vload4(0, input + 64U + chunk * 4U));
#else
      for (uint element = 0U; element < 4U; ++element) {
        dot_sum += (int)low[element] * (int)input[chunk * 4U + element];
        dot_sum += (int)high[element] *
            (int)input[64U + chunk * 4U + element];
      }
#endif
    }
    scaled_sum += (float)dot_sum * q8_d[group >> 1U];
  }
  const float original_weight_scale = vload_half(
      0, (__global const half*)(
          stripe + IQ36_LM_HEAD_I8Q4_CODE_BYTES_PER_STRIPE +
          row_lane * 2U));
  return scaled_sum * original_weight_scale * 16.0f;
}

__attribute__((reqd_work_group_size(256, 1, 1)))
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void iq36_lm_head_i8q4_rowstripe8_matvec_f32(
    __global const uchar* packed,
    __global const char* q8_qs,
    __global const float* q8_d,
    uint rows,
    __global float* output) {
  const uint row = (uint)get_global_id(0);
  if (row >= rows) return;
  output[row] = iq36_lm_head_i8q4_rowstripe8_dot(
      packed, q8_qs, q8_d, (ulong)row);
}

__attribute__((reqd_work_group_size(256, 1, 1)))
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void iq36_lm_head_i8q4_rowstripe8_matvec_topk8_f32(
    __global const uchar* packed,
    __global const char* q8_qs,
    __global const float* q8_d,
    uint rows,
    __global float* output,
    __global int* partial_top_ids,
    __global float* partial_top_values) {
  const uint row = (uint)get_global_id(0);
  const uint lane = (uint)get_local_id(0);
  const uint block = (uint)get_group_id(0);
  const float value = row < rows
      ? iq36_lm_head_i8q4_rowstripe8_dot(
            packed, q8_qs, q8_d, (ulong)row)
      : -INFINITY;
  if (row < rows) output[row] = value;
  __local float values[256];
  values[lane] = value;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lane != 0U) return;
  float best_values[8];
  int best_ids[8];
  for (uint index = 0U; index < 8U; ++index) {
    best_values[index] = -INFINITY;
    best_ids[index] = -1;
  }
  const uint begin = block * 256U;
  const uint count = min(256U, rows - begin);
  for (uint index = 0U; index < count; ++index) {
    const float candidate = values[index];
    uint position = 0U;
    while (position < 8U &&
           (best_values[position] > candidate ||
            (best_values[position] == candidate &&
             best_ids[position] < (int)(begin + index)))) {
      ++position;
    }
    if (position < 8U) {
      for (uint destination = 7U; destination > position; --destination) {
        best_values[destination] = best_values[destination - 1U];
        best_ids[destination] = best_ids[destination - 1U];
      }
      best_values[position] = candidate;
      best_ids[position] = (int)(begin + index);
    }
  }
  for (uint index = 0U; index < 8U; ++index) {
    partial_top_ids[block * 8U + index] = best_ids[index];
    partial_top_values[block * 8U + index] = best_values[index];
  }
}

__attribute__((reqd_work_group_size(64, 1, 1)))
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void iq36_lm_head_i8_exact_topk8_correction_f32(
    __global const char* weights,
    __global const half* weight_scales,
    __global const char* q8_qs,
    __global const float* q8_d,
    __global const int* top_ids,
    uint topk,
    __global float* output) {
  const uint selected = (uint)get_group_id(0);
  const uint lane = (uint)get_local_id(0);
  if (selected >= topk) return;
  const int signed_row = top_ids[selected];
  if (signed_row < 0 || signed_row >= (int)IQ36_LM_HEAD_I8Q4_ROWS) return;
  const uint row = (uint)signed_row;
  const ulong row_base = (ulong)row * 2048UL;
  const uint lane_base = lane * 32U;
  int dot_sum = 0;
  for (uint chunk = 0U; chunk < 8U; ++chunk) {
    const uint element = lane_base + chunk * 4U;
    dot_sum += dot(vload4(0, weights + row_base + element),
                   vload4(0, q8_qs + element));
  }
  __local int partial[64];
  __local float group_sums[8];
  partial[lane] = dot_sum;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lane < 8U) {
    int group_sum = 0;
    for (uint contributor = 0U; contributor < 8U; ++contributor) {
      group_sum += partial[lane * 8U + contributor];
    }
    group_sums[lane] = (float)group_sum * q8_d[lane];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lane == 0U) {
    float value = 0.0f;
    for (uint group = 0U; group < 8U; ++group) value += group_sums[group];
    output[row] = value * convert_float(weight_scales[row]);
  }
}

__kernel void q4s_group32_matvec_f32(
    __global const uchar* packed,
    __global const char* q8_qs,
    __global const float* q8_d,
    uint rows,
    uint cols,
    __global float* out) {
  const uint row = (uint)get_global_id(0);
  if (row >= rows) return;
  const uint group_count = cols >> 5U;
  out[row] = q4s_group32_rowstripe8_dot_f32(
      packed, q8_qs, q8_d, group_count, row);
}

__kernel void q4s_group64_matvec_f32(
    __global const uchar* packed,
    __global const char* q8_qs,
    __global const float* q8_d,
    uint rows,
    uint cols,
    __global float* out) {
  const uint row = (uint)get_global_id(0);
  if (row >= rows) return;
  const uint group_count = cols >> 6U;
  out[row] = q4s_group64_rowstripe8_dot_f32(
      packed, q8_qs, q8_d, group_count, row);
}

__kernel void q4s_group128_matvec_f32(
    __global const uchar* packed,
    __global const char* q8_qs,
    __global const float* q8_d,
    uint rows,
    uint cols,
    __global float* out) {
  const uint row = (uint)get_global_id(0);
  if (row >= rows) return;
  const uint group_count = cols >> 7U;
  out[row] = q4s_group128_rowstripe8_dot_f32(
      packed, q8_qs, q8_d, group_count, row);
}

__kernel void q4s_group128_matvec_topk8_f32(
    __global const uchar* packed,
    __global const char* q8_qs,
    __global const float* q8_d,
    uint rows,
    uint cols,
    __global float* out,
    __global int* partial_top_ids,
    __global float* partial_top_values) {
  const uint row = (uint)get_global_id(0);
  const uint lid = (uint)get_local_id(0);
  const uint block = (uint)get_group_id(0);
  const uint group_count = cols >> 7U;
  const float value = row < rows
      ? q4s_group128_rowstripe8_dot_f32(
            packed, q8_qs, q8_d, group_count, row)
      : -INFINITY;
  if (row < rows) out[row] = value;
  __local float values[256];
  values[lid] = value;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lid != 0U) return;
  float best_values[8];
  int best_ids[8];
  for (uint index = 0U; index < 8U; ++index) {
    best_values[index] = -INFINITY;
    best_ids[index] = -1;
  }
  const uint begin = block * 256U;
  const uint count = min(256U, rows - begin);
  for (uint index = 0U; index < count; ++index) {
    const float candidate = values[index];
    uint position = 0U;
    while (position < 8U && best_values[position] >= candidate) ++position;
    if (position < 8U) {
      for (uint destination = 7U; destination > position; --destination) {
        best_values[destination] = best_values[destination - 1U];
        best_ids[destination] = best_ids[destination - 1U];
      }
      best_values[position] = candidate;
      best_ids[position] = (int)(begin + index);
    }
  }
  for (uint index = 0U; index < 8U; ++index) {
    partial_top_ids[block * 8U + index] = best_ids[index];
    partial_top_values[block * 8U + index] = best_values[index];
  }
}

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void q4s_group128_lm_head_topk8_f32(
    __global const uchar* packed,
    __global const char* q8_qs,
    __global const float* q8_d,
    uint rows,
    uint cols,
    __global float* out,
    __global int* partial_top_ids,
    __global float* partial_top_values) {
  const uint row = (uint)get_global_id(0);
  const uint lid = (uint)get_local_id(0);
  const uint block = (uint)get_group_id(0);
  const float value = row < rows && cols == 2048U
      ? q4s_group128_rowstripe8_dot_f32(
            packed, q8_qs, q8_d, 16U, row)
      : -INFINITY;
  if (row < rows) out[row] = value;
  __local float values[256];
  values[lid] = value;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lid != 0U) return;
  float best_values[8];
  int best_ids[8];
  for (uint index = 0U; index < 8U; ++index) {
    best_values[index] = -INFINITY;
    best_ids[index] = -1;
  }
  const uint begin = block * 256U;
  const uint count = min(256U, rows - begin);
  for (uint index = 0U; index < count; ++index) {
    const float candidate = values[index];
    uint position = 0U;
    while (position < 8U && best_values[position] >= candidate) ++position;
    if (position < 8U) {
      for (uint destination = 7U; destination > position; --destination) {
        best_values[destination] = best_values[destination - 1U];
        best_ids[destination] = best_ids[destination - 1U];
      }
      best_values[position] = candidate;
      best_ids[position] = (int)(begin + index);
    }
  }
  for (uint index = 0U; index < 8U; ++index) {
    partial_top_ids[block * 8U + index] = best_ids[index];
    partial_top_values[block * 8U + index] = best_values[index];
  }
}

__kernel void q4s_group128_all_expert_down_topk8_plus_shared(
    __global const uchar* all_expert_packed,
    __global const uchar* shared_packed,
    __global const uint* selected_positions,
    __global const char* selected_q8_qs,
    __global const float* selected_q8_d,
    __global const char* shared_q8_qs,
    __global const float* shared_q8_d,
    __global float* selected_out,
    __global float* shared_out) {
  const uint flat = (uint)get_global_id(0);
  const uint group_count = 4U;
  if (flat < 16384U) {
    const uint selected = flat >> 11U;
    const uint local_row = flat & 2047U;
    const uint material_expert = selected_positions[selected];
    if (material_expert >= 256U) return;
    const ulong material_row =
        (ulong)material_expert * 2048UL + (ulong)local_row;
    selected_out[flat] = q4s_group128_rowstripe8_dot_f32(
        all_expert_packed,
        selected_q8_qs + (ulong)selected * 512UL,
        selected_q8_d + selected * 2U, group_count, material_row);
    return;
  }
  const uint shared_row = flat - 16384U;
  if (shared_row >= 2048U) return;
  shared_out[shared_row] = q4s_group128_rowstripe8_dot_f32(
      shared_packed, shared_q8_qs, shared_q8_d, group_count, shared_row);
}

__kernel void q4s_group32_all_expert_down_topk8_plus_shared(
    __global const uchar* all_expert_packed,
    __global const uchar* shared_packed,
    __global const uint* selected_positions,
    __global const char* selected_q8_qs,
    __global const float* selected_q8_d,
    __global const char* shared_q8_qs,
    __global const float* shared_q8_d,
    __global float* selected_out,
    __global float* shared_out) {
  const uint flat = (uint)get_global_id(0);
  const uint group_count = 16U;
  if (flat < 16384U) {
    const uint selected = flat >> 11U;
    const uint local_row = flat & 2047U;
    const uint material_expert = selected_positions[selected];
    if (material_expert >= 256U) return;
    const ulong material_row =
        (ulong)material_expert * 2048UL + (ulong)local_row;
    selected_out[flat] = q4s_group32_rowstripe8_dot_f32(
        all_expert_packed,
        selected_q8_qs + (ulong)selected * 512UL,
        selected_q8_d + selected * 2U, group_count, material_row);
    return;
  }
  const uint shared_row = flat - 16384U;
  if (shared_row >= 2048U) return;
  shared_out[shared_row] = q4s_group32_rowstripe8_dot_f32(
      shared_packed, shared_q8_qs, shared_q8_d, group_count, shared_row);
}

__kernel void q4k_cpu_order_matvec(__global const uchar* raw,
                                   __global const char* q8_qs,
                                   __global const short* q8_bsums,
                                   __global const float* q8_d,
                                   uint blocks_per_row,
                                   uint rows,
                                   __global float* out) {
  const uint row = (uint)get_global_id(0);
  if (row >= rows) {
    return;
  }

  float sums[8];
  for (int lane = 0; lane < 8; ++lane) {
    sums[lane] = 0.0f;
  }
  float min_sum = 0.0f;

  for (uint block_index = 0; block_index < blocks_per_row; ++block_index) {
    __global const uchar* block =
        raw + ((ulong)row * (ulong)blocks_per_row + (ulong)block_index) * 144UL;

    uchar aux[256];
    uint aux_pos = 0U;
    __global const uchar* q4 = block + 16;
    for (int group = 0; group < 4; ++group) {
      for (int lane = 0; lane < 32; ++lane) {
        aux[aux_pos + (uint)lane] = q4[lane] & 0x0FU;
      }
      aux_pos += 32U;
      for (int lane = 0; lane < 32; ++lane) {
        aux[aux_pos + (uint)lane] = q4[lane] >> 4;
      }
      aux_pos += 32U;
      q4 += 32;
    }

    const uint kMask1 = 0x3f3f3f3fU;
    const uint kMask2 = 0x0f0f0f0fU;
    const uint kMask3 = 0x03030303U;
    uint u0 = load_le32(block + 4);
    uint u1 = load_le32(block + 8);
    uint u2 = load_le32(block + 12);
    uint u3 = ((u2 >> 4) & kMask2) | (((u1 >> 6) & kMask3) << 4);
    const uint aux_scales = u1 & kMask1;
    u1 = (u2 & kMask2) | (((u0 >> 6) & kMask3) << 4);
    u2 = aux_scales;
    u0 &= kMask1;

    uchar scales[8];
    scales[0] = byte_from_word(u0, 0U);
    scales[1] = byte_from_word(u0, 1U);
    scales[2] = byte_from_word(u0, 2U);
    scales[3] = byte_from_word(u0, 3U);
    scales[4] = byte_from_word(u1, 0U);
    scales[5] = byte_from_word(u1, 1U);
    scales[6] = byte_from_word(u1, 2U);
    scales[7] = byte_from_word(u1, 3U);
    uchar mins[8];
    mins[0] = byte_from_word(u2, 0U);
    mins[1] = byte_from_word(u2, 1U);
    mins[2] = byte_from_word(u2, 2U);
    mins[3] = byte_from_word(u2, 3U);
    mins[4] = byte_from_word(u3, 0U);
    mins[5] = byte_from_word(u3, 1U);
    mins[6] = byte_from_word(u3, 2U);
    mins[7] = byte_from_word(u3, 3U);

    int grouped_min_sum = 0;
    __global const short* block_bsums = q8_bsums + (ulong)block_index * 16UL;
    for (int group = 0; group < 16; ++group) {
      grouped_min_sum += (int)block_bsums[group] * (int)mins[group / 2];
    }

    int lane_sums[8];
    for (int lane = 0; lane < 8; ++lane) {
      lane_sums[lane] = 0;
    }
    __global const char* block_q8 = q8_qs + (ulong)block_index * 256UL;
    uint q8_pos = 0U;
    aux_pos = 0U;
    int scale_index = 0;
    for (int group = 0; group < 8; ++group) {
      const int scale = (int)scales[scale_index++];
      for (int repeat = 0; repeat < 4; ++repeat) {
        for (int lane = 0; lane < 8; ++lane) {
          lane_sums[lane] +=
              scale * ((int)block_q8[q8_pos + (uint)lane] *
                       (int)aux[aux_pos + (uint)lane]);
        }
        q8_pos += 8U;
        aux_pos += 8U;
      }
    }

    const float d = half_to_float(load_le16(block)) * q8_d[block_index];
    for (int lane = 0; lane < 8; ++lane) {
      sums[lane] += d * (float)lane_sums[lane];
    }
    const float dmin = half_to_float(load_le16(block + 2)) * q8_d[block_index];
    min_sum -= dmin * (float)grouped_min_sum;
  }

  float sum = min_sum;
  for (int lane = 0; lane < 8; ++lane) {
    sum += sums[lane];
  }
  out[row] = sum;
}

__kernel void q4k_embedding_row_decode_f32(
    __global const uchar* raw,
    __global const uint* token_id,
    uint row_count,
    __global float* output) {
  const uint column = (uint)get_global_id(0);
  if (column >= 2048U) return;
  const uint row = token_id[0];
  if (row >= row_count) return;
  const uint block_index = column >> 8;
  const uint within_block = column & 255U;
  __global const uchar* block =
      raw + ((ulong)row * 8UL + block_index) * 144UL;
  const uint scale_index = within_block >> 5;
  uchar scale = 0;
  uchar minimum = 0;
  get_scale_min_k4((int)scale_index, block + 4, &scale, &minimum);
  const uint chunk = within_block >> 6;
  const uint within_chunk = within_block & 63U;
  const uchar packed = block[16U + chunk * 32U + (within_chunk & 31U)];
  const uint quant = within_chunk < 32U ? (uint)(packed & (uchar)15)
                                        : (uint)(packed >> 4);
  const float d = half_to_float(load_le16(block));
  const float dmin = half_to_float(load_le16(block + 2));
  output[column] = d * (float)((uint)scale * quant) -
                   dmin * (float)minimum;
}

__kernel void q4k_x8_matvec_group8(__global const uchar* packed,
                                   __global const char* q8_qs,
                                   __global const short* q8_bsums,
                                   __global const float* q8_d,
                                   uint blocks_per_row,
                                   uint row_groups,
                                   __global float* out) {
  const uint group = (uint)get_global_id(0);
  if (group >= row_groups) {
    return;
  }
  float sumf[8];
  float sum_minf[8];
  for (int j = 0; j < 8; ++j) {
    sumf[j] = 0.0f;
    sum_minf[j] = 0.0f;
  }
  for (uint block_index = 0; block_index < blocks_per_row; ++block_index) {
    __global const uchar* block =
        packed + ((ulong)group * (ulong)blocks_per_row + (ulong)block_index) * 1152UL;
    __global const char* q8 = q8_qs + (ulong)block_index * 256UL;
    __global const short* bsums = q8_bsums + (ulong)block_index * 16UL;
    const float q8_scale = q8_d[block_index];
    for (int k = 0; k < 16; ++k) {
      const int scale_pair = (k >> 2) * 2;
      const int q8_base = (k >> 2) * 64 + (k & 3) * 8;
      for (int j = 0; j < 8; ++j) {
        uchar scale0 = 0;
        uchar min0 = 0;
        uchar scale1 = 0;
        uchar min1 = 0;
        get_scale_min_k4(j, block + 32 + scale_pair * 12, &scale0, &min0);
        get_scale_min_k4(j, block + 32 + (scale_pair + 1) * 12, &scale1, &min1);
        int sumi = 0;
        for (int i = 0; i < 8; ++i) {
          const uchar q = block[128 + k * 64 + j * 8 + i];
          const int v0 = (int)(q & (uchar)15);
          const int v1 = (int)(q >> 4);
          const int q8_low = (int)q8[q8_base + i];
          const int q8_high = (int)q8[q8_base + i + 32];
          sumi += v0 * q8_low * (int)scale0;
          sumi += v1 * q8_high * (int)scale1;
        }
        sumf[j] += (float)sumi * half_to_float(load_le16(block + j * 2)) * q8_scale;
      }
    }
    for (int sb = 0; sb < 8; ++sb) {
      const int bsum_pair = (int)bsums[sb * 2] + (int)bsums[sb * 2 + 1];
      for (int j = 0; j < 8; ++j) {
        uchar scale = 0;
        uchar minimum = 0;
        get_scale_min_k4(j, block + 32 + sb * 12, &scale, &minimum);
        sum_minf[j] +=
            (float)((int)minimum * bsum_pair) *
            half_to_float(load_le16(block + 16 + j * 2)) *
            q8_scale;
      }
    }
  }
  for (int j = 0; j < 8; ++j) {
    out[group * 8 + j] = sumf[j] - sum_minf[j];
  }
}

__kernel void q4k_x8_matvec_rowlane(__global const uchar* packed,
                                    __global const char* q8_qs,
                                    __global const short* q8_bsums,
                                    __global const float* q8_d,
                                    uint blocks_per_row,
                                    uint row_groups,
                                    __global float* out) {
  const uint row = (uint)get_global_id(0);
  const uint total_rows = row_groups * 8U;
  if (row >= total_rows) {
    return;
  }
  const uint group = row >> 3;
  const uint j = row & 7U;
  float sumf = 0.0f;
  float sum_minf = 0.0f;
  for (uint block_index = 0; block_index < blocks_per_row; ++block_index) {
    __global const uchar* block =
        packed + ((ulong)group * (ulong)blocks_per_row + (ulong)block_index) * 1152UL;
    __global const char* q8 = q8_qs + (ulong)block_index * 256UL;
    __global const short* bsums = q8_bsums + (ulong)block_index * 16UL;
    const float q8_scale = q8_d[block_index];
    const float d = half_to_float(load_le16(block + j * 2)) * q8_scale;
    uchar scales[8];
    uchar minimums[8];
    for (int sb = 0; sb < 8; ++sb) {
      get_scale_min_k4((int)j, block + 32 + sb * 12,
                       &scales[sb], &minimums[sb]);
    }
    for (int k = 0; k < 16; ++k) {
      const int scale_pair = (k >> 2) * 2;
      const int q8_base = (k >> 2) * 64 + (k & 3) * 8;
      const uchar scale0 = scales[scale_pair];
      const uchar scale1 = scales[scale_pair + 1];
      __global const uchar* packed_q4 =
          block + 128 + k * 64 + (int)j * 8;
      const int sumi =
          q4_q8_dot8_global(packed_q4, q8 + q8_base, 0U) * (int)scale0 +
          q4_q8_dot8_global(packed_q4, q8 + q8_base + 32, 4U) * (int)scale1;
      sumf += (float)sumi * d;
    }
    const float dmin = half_to_float(load_le16(block + 16 + j * 2)) * q8_scale;
    for (int sb = 0; sb < 8; ++sb) {
      const int bsum_pair = (int)bsums[sb * 2] + (int)bsums[sb * 2 + 1];
      sum_minf += (float)((int)minimums[sb] * bsum_pair) * dmin;
    }
  }
  out[row] = sumf - sum_minf;
}

__kernel void q4k_x8_matvec_rowblock16_rowlane_finalize(
    __global const uchar* packed,
    __global const char* q8_qs,
    __global const short* q8_bsums,
    __global const float* q8_d,
    uint blocks_per_row,
    uint row_groups,
    __global float* out) {
  if (blocks_per_row != 16U) {
    return;
  }
  const uint row = (uint)get_group_id(0);
  const uint lid = (uint)get_local_id(0);
  const uint total_rows = row_groups * 8U;
  if (row >= total_rows) {
    return;
  }

  const uint group = row >> 3;
  const uint j = row & 7U;
  __global const uchar* block =
      packed + ((ulong)group * 16UL + (ulong)lid) * 1152UL;
  __global const char* q8 = q8_qs + (ulong)lid * 256UL;
  __global const short* bsums = q8_bsums + (ulong)lid * 16UL;
  __local int ordered_sumi[16 * 16];
  __local int ordered_min[16 * 8];

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
    __global const uchar* packed_q4 =
        block + 128 + k * 64 + (int)j * 8;
    ordered_sumi[lid * 16U + (uint)k] =
        q4_q8_dot8_global(packed_q4, q8 + q8_base, 0U) * (int)scale0 +
        q4_q8_dot8_global(packed_q4, q8 + q8_base + 32, 4U) * (int)scale1;
  }
  for (int sb = 0; sb < 8; ++sb) {
    const int bsum_pair = (int)bsums[sb * 2] + (int)bsums[sb * 2 + 1];
    uchar scale = 0;
    uchar minimum = 0;
    get_scale_min_k4((int)j, block + 32 + sb * 12, &scale, &minimum);
    ordered_min[lid * 8U + (uint)sb] = (int)minimum * bsum_pair;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  if (lid == 0U) {
    float sumf = 0.0f;
    float sum_minf = 0.0f;
    for (uint block_index = 0; block_index < 16U; ++block_index) {
      __global const uchar* ordered =
          packed + ((ulong)group * 16UL + (ulong)block_index) * 1152UL;
      const float q8_scale = q8_d[block_index];
      const float d = half_to_float(load_le16(ordered + j * 2U)) * q8_scale;
      for (uint k = 0; k < 16U; ++k) {
        sumf += (float)ordered_sumi[block_index * 16U + k] * d;
      }
      const float dmin =
          half_to_float(load_le16(ordered + 16U + j * 2U)) * q8_scale;
      for (uint sb = 0; sb < 8U; ++sb) {
        sum_minf +=
            (float)ordered_min[block_index * 8U + sb] * dmin;
      }
    }
    out[row] = sumf - sum_minf;
  }
}

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(128, 1, 1)))
__kernel void q4k_blockstripe16_matvec_group_subgroups(
    __global const uchar* packed,
    __global const char* q8_qs,
    __global const short* q8_bsums,
    __global const float* q8_d,
    uint blocks_per_row,
    uint row_groups,
    __global float* out) {
  if (blocks_per_row != 16U) return;
  const uint packed_group = (uint)get_group_id(0);
  const uint row_lane = (uint)get_sub_group_id();
  const uint block_index = (uint)get_sub_group_local_id();
  if (packed_group >= row_groups || row_lane >= 8U) return;
  __global const uint* rowstripe = (__global const uint*)(packed +
      ((ulong)packed_group * 8UL + row_lane) * 144UL * 16UL);
  const uint meta = rowstripe[block_index];
  const uchar4 scale0 = as_uchar4(rowstripe[16U + block_index]);
  const uchar4 scale1 = as_uchar4(rowstripe[32U + block_index]);
  const uchar4 scale2 = as_uchar4(rowstripe[48U + block_index]);
  const uchar scales[12] = {
      scale0.s0, scale0.s1, scale0.s2, scale0.s3,
      scale1.s0, scale1.s1, scale1.s2, scale1.s3,
      scale2.s0, scale2.s1, scale2.s2, scale2.s3};
  const ushort d_bits = (ushort)(meta & 0xffffU);
  const ushort dmin_bits = (ushort)(meta >> 16U);
  const float source_scale = q8_d[block_index];
  const float d = half_to_float(d_bits) * source_scale;
  const float dmin = half_to_float(dmin_bits) * source_scale;
  float block_sum = 0.0f;
  float block_min = 0.0f;
  for (uint group = 0U; group < 8U; ++group) {
    uchar scale = 0;
    uchar minimum = 0;
    get_scale_min_k4_private((int)group, scales, &scale, &minimum);
    int dot_sum = 0;
    const uint segment = group >> 1U;
    const uint shift = (group & 1U) * 4U;
    for (uint chunk = 0U; chunk < 8U; ++chunk) {
      const uint raw_word = 4U + segment * 8U + chunk;
      const uchar4 codes =
          as_uchar4(rowstripe[raw_word * 16U + block_index]);
      const char4 values = convert_char4(
          (codes >> shift) & (uchar4)(15));
      const char4 source = vload4(
          0, q8_qs + block_index * 256U + group * 32U + chunk * 4U);
#if defined(IQ36_USE_INTEGER_DOT)
      dot_sum += dot(values, source);
#else
      dot_sum += (int)values.s0 * (int)source.s0;
      dot_sum += (int)values.s1 * (int)source.s1;
      dot_sum += (int)values.s2 * (int)source.s2;
      dot_sum += (int)values.s3 * (int)source.s3;
#endif
    }
    block_sum += (float)(dot_sum * (int)scale) * d;
    const int source_sum =
        (int)q8_bsums[block_index * 16U + group * 2U] +
        (int)q8_bsums[block_index * 16U + group * 2U + 1U];
    block_min += (float)(source_sum * (int)minimum) * dmin;
  }
  float sum = 0.0f;
  float minimum_sum = 0.0f;
  for (uint ordered_block = 0U; ordered_block < 16U; ++ordered_block) {
    sum += sub_group_broadcast(block_sum, ordered_block);
    minimum_sum += sub_group_broadcast(block_min, ordered_block);
  }
  if (block_index == 0U) {
    out[packed_group * 8U + row_lane] = sum - minimum_sum;
  }
}

__kernel void q4k_x8_matvec_rowblock16_reduce(
    __global const uchar* packed,
    __global const char* q8_qs,
    __global const short* q8_bsums,
    __global const float* q8_d,
    uint blocks_per_row,
    uint row_groups,
    __global float* out) {
  if (blocks_per_row != 16U) {
    return;
  }
  const uint row = (uint)get_group_id(0);
  const uint lid = (uint)get_local_id(0);
  const uint total_rows = row_groups * 8U;
  if (row >= total_rows) {
    return;
  }

  const uint group = row >> 3;
  const uint j = row & 7U;
  const uint block_index = lid;
  __local float partial[16];

  __global const uchar* block =
      packed + ((ulong)group * 16UL + (ulong)block_index) * 1152UL;
  __global const char* q8 = q8_qs + (ulong)block_index * 256UL;
  __global const short* bsums = q8_bsums + (ulong)block_index * 16UL;
  const float q8_scale = q8_d[block_index];
  const float d = half_to_float(load_le16(block + j * 2)) * q8_scale;
  float sumf = 0.0f;
  float sum_minf = 0.0f;

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

  const float dmin = half_to_float(load_le16(block + 16 + j * 2)) * q8_scale;
  for (int sb = 0; sb < 8; ++sb) {
    const int bsum_pair = (int)bsums[sb * 2] + (int)bsums[sb * 2 + 1];
    uchar scale = 0;
    uchar minimum = 0;
    get_scale_min_k4((int)j, block + 32 + sb * 12, &scale, &minimum);
    sum_minf += (float)((int)minimum * bsum_pair) * dmin;
  }

  partial[lid] = sumf - sum_minf;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint stride = 8U; stride > 0U; stride >>= 1U) {
    if (lid < stride) {
      partial[lid] += partial[lid + stride];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lid == 0U) {
    out[row] = partial[0];
  }
}

#pragma OPENCL FP_CONTRACT OFF
__kernel void q4k_x8_matvec_rowblock16_cpuorder_finalize(
    __global const uchar* packed,
    __global const char* q8_qs,
    __global const short* q8_bsums,
    __global const float* q8_d,
    uint blocks_per_row,
    uint row_groups,
    __global float* out) {
  if (blocks_per_row != 16U) {
    return;
  }
  const uint row = (uint)get_group_id(0);
  const uint lid = (uint)get_local_id(0);
  const uint total_rows = row_groups * 8U;
  if (row >= total_rows) {
    return;
  }

  const uint group = row >> 3;
  const uint j = row & 7U;
  const uint block_index = lid;
  __local int local_lane_sums[16 * 8];
  __local int local_grouped_min_sums[16];

  __global const uchar* block =
      packed + ((ulong)group * 16UL + (ulong)block_index) * 1152UL;
  __global const char* q8 = q8_qs + (ulong)block_index * 256UL;
  __global const short* bsums = q8_bsums + (ulong)block_index * 16UL;
  uchar scales[8];
  uchar minimums[8];
  for (int scale_index = 0; scale_index < 8; ++scale_index) {
    get_scale_min_k4((int)j, block + 32 + scale_index * 12,
                     &scales[scale_index], &minimums[scale_index]);
  }

  int grouped_min_sum = 0;
  for (int min_index = 0; min_index < 8; ++min_index) {
    const int bsum_pair =
        (int)bsums[min_index * 2] + (int)bsums[min_index * 2 + 1];
    grouped_min_sum += bsum_pair * (int)minimums[min_index];
  }

  int lane_sums[8];
  for (int lane = 0; lane < 8; ++lane) {
    lane_sums[lane] = 0;
  }
  for (int q4_group = 0; q4_group < 4; ++q4_group) {
    const int q8_base = q4_group * 64;
    const int low_scale = (int)scales[q4_group * 2];
    const int high_scale = (int)scales[q4_group * 2 + 1];
    for (int q4_lane = 0; q4_lane < 32; ++q4_lane) {
      const int q4_index = q4_group * 32 + q4_lane;
      const uchar q4 = block[
          128 + (q4_index >> 3) * 64 + (int)j * 8 + (q4_index & 7)];
      const int lane = q4_lane & 7;
      lane_sums[lane] +=
          low_scale * ((int)q8[q8_base + q4_lane] * (int)(q4 & 15));
      lane_sums[lane] +=
          high_scale * ((int)q8[q8_base + 32 + q4_lane] * (int)(q4 >> 4));
    }
  }

  for (int lane = 0; lane < 8; ++lane) {
    local_lane_sums[lid * 8U + (uint)lane] = lane_sums[lane];
  }
  local_grouped_min_sums[lid] = grouped_min_sum;
  barrier(CLK_LOCAL_MEM_FENCE);

  if (lid == 0U) {
    float sums[8];
    for (int lane = 0; lane < 8; ++lane) {
      sums[lane] = 0.0f;
    }
    float min_sum = 0.0f;
    for (uint ordered_block = 0U; ordered_block < 16U; ++ordered_block) {
      __global const uchar* ordered =
          packed + ((ulong)group * 16UL + (ulong)ordered_block) * 1152UL;
      const float q8_scale = q8_d[ordered_block];
      const float d = half_to_float(load_le16(ordered + j * 2U)) * q8_scale;
      for (int lane = 0; lane < 8; ++lane) {
        sums[lane] +=
            d * (float)local_lane_sums[ordered_block * 8U + (uint)lane];
      }
      const float dmin =
          half_to_float(load_le16(ordered + 16U + j * 2U)) * q8_scale;
      min_sum -=
          dmin * (float)local_grouped_min_sums[ordered_block];
    }
    float sum = min_sum;
    for (int lane = 0; lane < 8; ++lane) {
      sum += sums[lane];
    }
    out[row] = sum;
  }
}
#pragma OPENCL FP_CONTRACT ON

__kernel void q4k_x8_matvec_rowlane_localq8(
    __global const uchar* packed,
    __global const char* q8_qs,
    __global const short* q8_bsums,
    __global const float* q8_d,
    uint blocks_per_row,
    uint row_groups,
    __global float* out) {
  const uint row = (uint)get_global_id(0);
  const uint total_rows = row_groups * 8U;
  const uint lid = (uint)get_local_id(0);
  const uint lsize = (uint)get_local_size(0);
  const uint local_blocks = blocks_per_row <= 8U ? blocks_per_row : 8U;
  __local char local_q8[2048];
  __local short local_bsums[128];
  __local float local_d[8];
  for (uint i = lid; i < local_blocks * 256U; i += lsize) {
    local_q8[i] = q8_qs[i];
  }
  for (uint i = lid; i < local_blocks * 16U; i += lsize) {
    local_bsums[i] = q8_bsums[i];
  }
  for (uint i = lid; i < local_blocks; i += lsize) {
    local_d[i] = q8_d[i];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (row >= total_rows || blocks_per_row > 8U) {
    return;
  }
  const uint group = row >> 3;
  const uint j = row & 7U;
  float sumf = 0.0f;
  float sum_minf = 0.0f;
  for (uint block_index = 0; block_index < blocks_per_row; ++block_index) {
    __global const uchar* block =
        packed + ((ulong)group * (ulong)blocks_per_row + (ulong)block_index) * 1152UL;
    const float q8_scale = local_d[block_index];
    const float d = half_to_float(load_le16(block + j * 2)) * q8_scale;
    uchar scales[8];
    uchar minimums[8];
    for (int sb = 0; sb < 8; ++sb) {
      get_scale_min_k4((int)j, block + 32 + sb * 12, &scales[sb], &minimums[sb]);
    }
    for (int k = 0; k < 16; ++k) {
      const int scale_pair = (k >> 2) * 2;
      const int q8_base = (int)block_index * 256 + (k >> 2) * 64 + (k & 3) * 8;
      const uchar scale0 = scales[scale_pair];
      const uchar scale1 = scales[scale_pair + 1];
      __global const uchar* packed_q4 =
          block + 128 + k * 64 + (int)j * 8;
      const int sumi =
          q4_q8_dot8_local(packed_q4, local_q8 + q8_base, 0U) *
              (int)scale0 +
          q4_q8_dot8_local(packed_q4, local_q8 + q8_base + 32, 4U) *
              (int)scale1;
      sumf += (float)sumi * d;
    }
    const float dmin = half_to_float(load_le16(block + 16 + j * 2)) * q8_scale;
    const int bsum_base = (int)block_index * 16;
    for (int sb = 0; sb < 8; ++sb) {
      const int bsum_pair =
          (int)local_bsums[bsum_base + sb * 2] +
          (int)local_bsums[bsum_base + sb * 2 + 1];
      sum_minf += (float)((int)minimums[sb] * bsum_pair) * dmin;
    }
  }
  out[row] = sumf - sum_minf;
}

__kernel void q4k_x8_matvec_rowlane_expert8(
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
        packed + ((ulong)group * (ulong)blocks_per_row + (ulong)block_index) * 1152UL;
    __global const char* q8 = q8_qs + (ulong)block_index * 256UL;
    __global const short* bsums = q8_bsums + (ulong)block_index * 16UL;
    const float q8_scale = q8_d[block_index];
    const float d = half_to_float(load_le16(block + j * 2)) * q8_scale;
    for (int k = 0; k < 16; ++k) {
      const int scale_pair = (k >> 2) * 2;
      const int q8_base = (k >> 2) * 64 + (k & 3) * 8;
      uchar scale0 = 0;
      uchar min0 = 0;
      uchar scale1 = 0;
      uchar min1 = 0;
      get_scale_min_k4((int)j, block + 32 + scale_pair * 12, &scale0, &min0);
      get_scale_min_k4((int)j, block + 32 + (scale_pair + 1) * 12, &scale1, &min1);
      __global const uchar* packed_q4 =
          block + 128 + k * 64 + (int)j * 8;
      const int sumi =
          q4_q8_dot8_global(packed_q4, q8 + q8_base, 0U) * (int)scale0 +
          q4_q8_dot8_global(packed_q4, q8 + q8_base + 32, 4U) * (int)scale1;
      sumf += (float)sumi * d;
    }
    const float dmin = half_to_float(load_le16(block + 16 + j * 2)) * q8_scale;
    for (int sb = 0; sb < 8; ++sb) {
      const int bsum_pair = (int)bsums[sb * 2] + (int)bsums[sb * 2 + 1];
      uchar scale = 0;
      uchar minimum = 0;
      get_scale_min_k4((int)j, block + 32 + sb * 12, &scale, &minimum);
      sum_minf += (float)((int)minimum * bsum_pair) * dmin;
    }
  }
  out[row] = sumf - sum_minf;
}

__kernel void q4k_x8_matvec_rowlane_expert8_localq8(
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
  const uint row = (uint)get_global_id(0);
  const uint rows_per_expert = row_groups_per_expert * 8U;
  const uint total_rows = rows_per_expert * 8U;
  const uint lid = (uint)get_local_id(0);
  const uint lsize = (uint)get_local_size(0);
  const uint local_blocks = blocks_per_row <= 8U ? blocks_per_row : 8U;
  __local char local_q8[2048];
  __local short local_bsums[128];
  __local float local_d[8];
  for (uint i = lid; i < local_blocks * 256U; i += lsize) {
    local_q8[i] = q8_qs[i];
  }
  for (uint i = lid; i < local_blocks * 16U; i += lsize) {
    local_bsums[i] = q8_bsums[i];
  }
  for (uint i = lid; i < local_blocks; i += lsize) {
    local_d[i] = q8_d[i];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (row >= total_rows || blocks_per_row > 8U) {
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
        packed + ((ulong)group * (ulong)blocks_per_row + (ulong)block_index) * 1152UL;
    const float q8_scale = local_d[block_index];
    const float d = half_to_float(load_le16(block + j * 2)) * q8_scale;
    uchar scales[8];
    uchar minimums[8];
    for (int sb = 0; sb < 8; ++sb) {
      get_scale_min_k4((int)j, block + 32 + sb * 12, &scales[sb], &minimums[sb]);
    }
    for (int k = 0; k < 16; ++k) {
      const int scale_pair = (k >> 2) * 2;
      const int q8_base = (int)block_index * 256 + (k >> 2) * 64 + (k & 3) * 8;
      const uchar scale0 = scales[scale_pair];
      const uchar scale1 = scales[scale_pair + 1];
      int sumi = 0;
      for (int i = 0; i < 8; ++i) {
        const uchar q = block[128 + k * 64 + (int)j * 8 + i];
        const int v0 = (int)(q & (uchar)15);
        const int v1 = (int)(q >> 4);
        const int q8_low = (int)local_q8[q8_base + i];
        const int q8_high = (int)local_q8[q8_base + i + 32];
        sumi += v0 * q8_low * (int)scale0;
        sumi += v1 * q8_high * (int)scale1;
      }
      sumf += (float)sumi * d;
    }
    const float dmin = half_to_float(load_le16(block + 16 + j * 2)) * q8_scale;
    const int bsum_base = (int)block_index * 16;
    for (int sb = 0; sb < 8; ++sb) {
      const int bsum_pair =
          (int)local_bsums[bsum_base + sb * 2] +
          (int)local_bsums[bsum_base + sb * 2 + 1];
      sum_minf += (float)((int)minimums[sb] * bsum_pair) * dmin;
    }
  }
  out[row] = sumf - sum_minf;
}

__kernel void q4k_x8_matvec_rowlane_expert8_plus_shared_localq8(
    __global const uchar* packed0,
    __global const uchar* packed1,
    __global const uchar* packed2,
    __global const uchar* packed3,
    __global const uchar* packed4,
    __global const uchar* packed5,
    __global const uchar* packed6,
    __global const uchar* packed7,
    __global const uchar* packed_shared,
    __global const char* q8_qs,
    __global const short* q8_bsums,
    __global const float* q8_d,
    uint blocks_per_row,
    uint row_groups_per_expert,
    __global float* out) {
  const uint row = (uint)get_global_id(0);
  const uint rows_per_expert = row_groups_per_expert * 8U;
  const uint total_rows = rows_per_expert * 9U;
  const uint lid = (uint)get_local_id(0);
  const uint lsize = (uint)get_local_size(0);
  const uint local_blocks = blocks_per_row <= 8U ? blocks_per_row : 8U;
  __local char local_q8[2048];
  __local short local_bsums[128];
  __local float local_d[8];
  for (uint i = lid; i < local_blocks * 256U; i += lsize) {
    local_q8[i] = q8_qs[i];
  }
  for (uint i = lid; i < local_blocks * 16U; i += lsize) {
    local_bsums[i] = q8_bsums[i];
  }
  for (uint i = lid; i < local_blocks; i += lsize) {
    local_d[i] = q8_d[i];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (row >= total_rows || blocks_per_row > 8U) {
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
  } else if (expert == 8U) {
    packed = packed_shared;
  }

  float sumf = 0.0f;
  float sum_minf = 0.0f;
  for (uint block_index = 0; block_index < blocks_per_row; ++block_index) {
    __global const uchar* block =
        packed + ((ulong)group * (ulong)blocks_per_row + (ulong)block_index) * 1152UL;
    const float q8_scale = local_d[block_index];
    const float d = half_to_float(load_le16(block + j * 2)) * q8_scale;
    uchar scales[8];
    uchar minimums[8];
    for (int sb = 0; sb < 8; ++sb) {
      get_scale_min_k4((int)j, block + 32 + sb * 12, &scales[sb], &minimums[sb]);
    }
    for (int k = 0; k < 16; ++k) {
      const int scale_pair = (k >> 2) * 2;
      const int q8_base = (int)block_index * 256 + (k >> 2) * 64 + (k & 3) * 8;
      const uchar scale0 = scales[scale_pair];
      const uchar scale1 = scales[scale_pair + 1];
      int sumi = 0;
      for (int i = 0; i < 8; ++i) {
        const uchar q = block[128 + k * 64 + (int)j * 8 + i];
        const int v0 = (int)(q & (uchar)15);
        const int v1 = (int)(q >> 4);
        const int q8_low = (int)local_q8[q8_base + i];
        const int q8_high = (int)local_q8[q8_base + i + 32];
        sumi += v0 * q8_low * (int)scale0;
        sumi += v1 * q8_high * (int)scale1;
      }
      sumf += (float)sumi * d;
    }
    const float dmin = half_to_float(load_le16(block + 16 + j * 2)) * q8_scale;
    const int bsum_base = (int)block_index * 16;
    for (int sb = 0; sb < 8; ++sb) {
      const int bsum_pair =
          (int)local_bsums[bsum_base + sb * 2] +
          (int)local_bsums[bsum_base + sb * 2 + 1];
      sum_minf += (float)((int)minimums[sb] * bsum_pair) * dmin;
    }
  }
  out[row] = sumf - sum_minf;
}

__attribute__((reqd_work_group_size(1024, 1, 1)))
__kernel void q4k_x8_matvec_topk_indexed_expert8_plus_shared_localq8(
    __global const uchar* selected_gate_up_topk_indexed,
    __global const uchar* packed_shared,
    __global const uint* selected_positions,
    __global const char* q8_qs,
    __global const short* q8_bsums,
    __global const float* q8_d,
    uint blocks_per_row,
    uint row_groups_per_expert,
    uint material_expert_count,
    __global float* out) {
  const uint row = (uint)get_global_id(0);
  const uint rows_per_expert = row_groups_per_expert * 8U;
  const uint total_rows = rows_per_expert * 9U;
  const uint lid = (uint)get_local_id(0);
  const uint lsize = (uint)get_local_size(0);
  const uint local_blocks = blocks_per_row <= 8U ? blocks_per_row : 8U;
  __local char local_q8[2048];
  __local short local_bsums[128];
  __local float local_d[8];
  for (uint i = lid; i < local_blocks * 256U; i += lsize) {
    local_q8[i] = q8_qs[i];
  }
  for (uint i = lid; i < local_blocks * 16U; i += lsize) {
    local_bsums[i] = q8_bsums[i];
  }
  for (uint i = lid; i < local_blocks; i += lsize) {
    local_d[i] = q8_d[i];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (row >= total_rows || blocks_per_row > 8U) {
    return;
  }
  const uint expert = row / rows_per_expert;
  const uint local_row = row - expert * rows_per_expert;
  const uint group = local_row >> 3;
  const uint j = local_row & 7U;
  __global const uchar* packed = packed_shared;
  uint material_expert = 0U;
  if (expert < 8U) {
    material_expert = selected_positions[expert];
    if (material_expert >= material_expert_count) {
      return;
    }
    packed = selected_gate_up_topk_indexed;
  }

  float sumf = 0.0f;
  float sum_minf = 0.0f;
  for (uint block_index = 0; block_index < blocks_per_row; ++block_index) {
    const ulong selected_group =
        (ulong)material_expert * (ulong)row_groups_per_expert +
        (ulong)group;
    const ulong packed_group =
        expert < 8U ? selected_group : (ulong)group;
    __global const uchar* block =
        packed + (packed_group * (ulong)blocks_per_row +
                  (ulong)block_index) * 1152UL;
    const float q8_scale = local_d[block_index];
    const float d = half_to_float(load_le16(block + j * 2)) * q8_scale;
    const float dmin = half_to_float(load_le16(block + 16 + j * 2)) * q8_scale;
    const int bsum_base = (int)block_index * 16;
    for (int scale_group = 0; scale_group < 4; ++scale_group) {
      const int sb0 = scale_group * 2;
      const int sb1 = sb0 + 1;
      uchar scale0 = 0;
      uchar minimum0 = 0;
      uchar scale1 = 0;
      uchar minimum1 = 0;
      get_scale_min_k4((int)j, block + 32 + sb0 * 12,
                       &scale0, &minimum0);
      get_scale_min_k4((int)j, block + 32 + sb1 * 12,
                       &scale1, &minimum1);
      for (int lane = 0; lane < 4; ++lane) {
        const int k = scale_group * 4 + lane;
        const int q8_base =
            (int)block_index * 256 + scale_group * 64 + lane * 8;
        __global const uchar* packed_q4 =
            block + 128 + k * 64 + (int)j * 8;
        const int sumi =
            q4_q8_dot8_local(packed_q4, local_q8 + q8_base, 0U) *
                (int)scale0 +
            q4_q8_dot8_local(packed_q4, local_q8 + q8_base + 32, 4U) *
                (int)scale1;
        sumf += (float)sumi * d;
      }
      const int bsum_pair0 =
          (int)local_bsums[bsum_base + sb0 * 2] +
          (int)local_bsums[bsum_base + sb0 * 2 + 1];
      const int bsum_pair1 =
          (int)local_bsums[bsum_base + sb1 * 2] +
          (int)local_bsums[bsum_base + sb1 * 2 + 1];
      sum_minf += (float)((int)minimum0 * bsum_pair0) * dmin;
      sum_minf += (float)((int)minimum1 * bsum_pair1) * dmin;
    }
  }
  out[row] = sumf - sum_minf;
}

__kernel void q4k_x8_matvec_rowlane_expert8_multiq8(
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
        packed + ((ulong)group * (ulong)blocks_per_row + (ulong)block_index) * 1152UL;
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
      get_scale_min_k4((int)j, block + 32 + scale_pair * 12, &scale0, &min0);
      get_scale_min_k4((int)j, block + 32 + (scale_pair + 1) * 12, &scale1, &min1);
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
    const float dmin = half_to_float(load_le16(block + 16 + j * 2)) * q8_scale;
    for (int sb = 0; sb < 8; ++sb) {
      const int bsum_pair = (int)bsums[sb * 2] + (int)bsums[sb * 2 + 1];
      uchar scale = 0;
      uchar minimum = 0;
      get_scale_min_k4((int)j, block + 32 + sb * 12, &scale, &minimum);
      sum_minf += (float)((int)minimum * bsum_pair) * dmin;
    }
  }
  out[row] = sumf - sum_minf;
}

float q4k_x8_selected_down_dot_multiq8(
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
    uchar scales[8];
    uchar minimums[8];
    for (int sb = 0; sb < 8; ++sb) {
      get_scale_min_k4((int)j, block + 32 + sb * 12,
                       &scales[sb], &minimums[sb]);
    }
    for (int k = 0; k < 16; ++k) {
      const int scale_pair = (k >> 2) * 2;
      const int q8_base = (k >> 2) * 64 + (k & 3) * 8;
      const uchar scale0 = scales[scale_pair];
      const uchar scale1 = scales[scale_pair + 1];
      __global const uchar* packed_q4 =
          block + 128 + k * 64 + (int)j * 8;
      const int sumi =
          q4_q8_dot8_global(packed_q4, q8 + q8_base, 0U) * (int)scale0 +
          q4_q8_dot8_global(packed_q4, q8 + q8_base + 32, 4U) * (int)scale1;
      sumf += (float)sumi * d;
    }
    const float dmin =
        half_to_float(load_le16(block + 16 + j * 2)) * q8_scale;
    for (int sb = 0; sb < 8; ++sb) {
      const int bsum_pair = (int)bsums[sb * 2] + (int)bsums[sb * 2 + 1];
      sum_minf += (float)((int)minimums[sb] * bsum_pair) * dmin;
    }
  }
  return sumf - sum_minf;
}

__kernel void q4k_x8_selected_down_expert8_plus_shared_q4(
    __global const uchar* packed0,
    __global const uchar* packed1,
    __global const uchar* packed2,
    __global const uchar* packed3,
    __global const uchar* packed4,
    __global const uchar* packed5,
    __global const uchar* packed6,
    __global const uchar* packed7,
    __global const uchar* packed_shared,
    __global const char* selected_q8_qs,
    __global const short* selected_q8_bsums,
    __global const float* selected_q8_d,
    __global const char* shared_q8_qs,
    __global const short* shared_q8_bsums,
    __global const float* shared_q8_d,
    uint blocks_per_row,
    uint row_groups_per_expert,
    __global float* selected_out,
    __global float* shared_out) {
  const uint flat = (uint)get_global_id(0);
  const uint rows_per_expert = row_groups_per_expert * 8U;
  const uint selected_total_rows = rows_per_expert * 8U;
  if (flat < selected_total_rows) {
    const uint expert = flat / rows_per_expert;
    const uint local_row = flat - expert * rows_per_expert;
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
    selected_out[flat] = q4k_x8_selected_down_dot_multiq8(
        packed, selected_q8_qs, selected_q8_bsums, selected_q8_d,
        blocks_per_row, expert, group, j);
    return;
  }

  const uint shared_row = flat - selected_total_rows;
  if (shared_row >= rows_per_expert) {
    return;
  }
  const uint group = shared_row >> 3;
  const uint j = shared_row & 7U;
  shared_out[shared_row] = q4k_x8_selected_down_dot_multiq8(
      packed_shared, shared_q8_qs, shared_q8_bsums, shared_q8_d,
      blocks_per_row, 0U, group, j);
}

__kernel void q4k_x8_all_expert_down_topk8_plus_shared(
    __global const uchar* all_expert_packed,
    __global const uchar* shared_packed,
    __global const uint* selected_positions,
    __global const char* selected_q8_qs,
    __global const short* selected_q8_bsums,
    __global const float* selected_q8_d,
    __global const char* shared_q8_qs,
    __global const short* shared_q8_bsums,
    __global const float* shared_q8_d,
    __global float* selected_out,
    __global float* shared_out) {
  const uint flat = (uint)get_global_id(0);
  if (flat < 16384U) {
    const uint selected = flat >> 11;
    const uint local_row = flat & 2047U;
    const uint material_expert = selected_positions[selected];
    if (material_expert >= 256U) return;
    __global const uchar* packed =
        all_expert_packed + (ulong)material_expert * 589824UL;
    selected_out[flat] = q4k_x8_selected_down_dot_multiq8(
        packed, selected_q8_qs, selected_q8_bsums, selected_q8_d,
        2U, selected, local_row >> 3, local_row & 7U);
    return;
  }
  const uint shared_row = flat - 16384U;
  if (shared_row >= 2048U) return;
  shared_out[shared_row] = q4k_x8_selected_down_dot_multiq8(
      shared_packed, shared_q8_qs, shared_q8_bsums, shared_q8_d,
      2U, 0U, shared_row >> 3, shared_row & 7U);
}

__kernel void q4k_x8_matvec_rowlane_expert8_f32input(
    __global const uchar* packed0,
    __global const uchar* packed1,
    __global const uchar* packed2,
    __global const uchar* packed3,
    __global const uchar* packed4,
    __global const uchar* packed5,
    __global const uchar* packed6,
    __global const uchar* packed7,
    __global const float* input,
    uint blocks_per_row,
    uint row_groups_per_expert,
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
    __global const float* x =
        input + ((ulong)expert * (ulong)blocks_per_row +
                 (ulong)block_index) * 256UL;
    const float d = half_to_float(load_le16(block + j * 2));
    for (int k = 0; k < 16; ++k) {
      const int scale_pair = (k >> 2) * 2;
      const int input_base = (k >> 2) * 64 + (k & 3) * 8;
      uchar scale0 = 0;
      uchar min0 = 0;
      uchar scale1 = 0;
      uchar min1 = 0;
      get_scale_min_k4((int)j, block + 32 + scale_pair * 12,
                       &scale0, &min0);
      get_scale_min_k4((int)j, block + 32 + (scale_pair + 1) * 12,
                       &scale1, &min1);
      float sumi = 0.0f;
      for (int i = 0; i < 8; ++i) {
        const uchar q = block[128 + k * 64 + (int)j * 8 + i];
        const float low = x[input_base + i];
        const float high = x[input_base + i + 32];
        sumi += (float)((int)(q & (uchar)15) * (int)scale0) * low;
        sumi += (float)((int)(q >> 4) * (int)scale1) * high;
      }
      sumf += sumi * d;
    }
    const float dmin = half_to_float(load_le16(block + 16 + j * 2));
    for (int sb = 0; sb < 8; ++sb) {
      uchar scale = 0;
      uchar minimum = 0;
      get_scale_min_k4((int)j, block + 32 + sb * 12, &scale, &minimum);
      float xsum = 0.0f;
      const int input_base = sb * 32;
      for (int i = 0; i < 32; ++i) {
        xsum += x[input_base + i];
      }
      sum_minf += (float)((int)minimum) * xsum * dmin;
    }
  }
  out[row] = sumf - sum_minf;
}

__kernel void f32_matvec_row_f32(__global const float* weights,
                                __global const float* input,
                                uint cols,
                                uint rows,
                                __global float* out) {
  const uint row = (uint)get_global_id(0);
  if (row >= rows) {
    return;
  }
  const uint base = row * cols;
  float sum = 0.0f;
  uint col = 0;
  for (; col + 15U < cols; col += 16U) {
    const float4 w0 = vload4(0, weights + base + col);
    const float4 x0 = vload4(0, input + col);
    const float4 w1 = vload4(0, weights + base + col + 4U);
    const float4 x1 = vload4(0, input + col + 4U);
    const float4 w2 = vload4(0, weights + base + col + 8U);
    const float4 x2 = vload4(0, input + col + 8U);
    const float4 w3 = vload4(0, weights + base + col + 12U);
    const float4 x3 = vload4(0, input + col + 12U);
    sum += w0.s0 * x0.s0;
    sum += w0.s1 * x0.s1;
    sum += w0.s2 * x0.s2;
    sum += w0.s3 * x0.s3;
    sum += w1.s0 * x1.s0;
    sum += w1.s1 * x1.s1;
    sum += w1.s2 * x1.s2;
    sum += w1.s3 * x1.s3;
    sum += w2.s0 * x2.s0;
    sum += w2.s1 * x2.s1;
    sum += w2.s2 * x2.s2;
    sum += w2.s3 * x2.s3;
    sum += w3.s0 * x3.s0;
    sum += w3.s1 * x3.s1;
    sum += w3.s2 * x3.s2;
    sum += w3.s3 * x3.s3;
  }
  for (; col < cols; ++col) {
    sum += weights[base + col] * input[col];
  }
  out[row] = sum;
}

__kernel void q8_router_matvec_rows_parallel_f32(
    __global const char* weights,
    __global const float* weight_scales,
    __global const char* input,
    __global const float* input_scales,
    uint cols,
    __global float* logits) {
  const uint row = (uint)get_group_id(0);
  const uint lid = (uint)get_local_id(0);
  const uint group_count = cols >> 5U;
  __local float contributions[64];
  float contribution = 0.0f;
  if (row < 257U && lid < group_count) {
    const uint col = lid << 5U;
    const uint base = row * cols + col;
    int dot_sum = 0;
    dot_sum += i8_i8_dot4_global(weights + base, input + col);
    dot_sum += i8_i8_dot4_global(weights + base + 4U, input + col + 4U);
    dot_sum += i8_i8_dot4_global(weights + base + 8U, input + col + 8U);
    dot_sum += i8_i8_dot4_global(weights + base + 12U, input + col + 12U);
    dot_sum += i8_i8_dot4_global(weights + base + 16U, input + col + 16U);
    dot_sum += i8_i8_dot4_global(weights + base + 20U, input + col + 20U);
    dot_sum += i8_i8_dot4_global(weights + base + 24U, input + col + 24U);
    dot_sum += i8_i8_dot4_global(weights + base + 28U, input + col + 28U);
    contribution = (float)dot_sum *
        weight_scales[row * group_count + lid] * input_scales[lid >> 3U];
  }
  contributions[lid] = contribution;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lid == 0U && row < 257U) {
    float sum = 0.0f;
    for (uint index = 0U; index < group_count; ++index) {
      sum += contributions[index];
    }
    logits[row] = sum;
  }
}

__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void q8_router_matvec_group32_rows_parallel_f32_input(
    __global const char* weights,
    __global const float* weight_scales,
    __global const float* input,
    uint cols,
    uint group_size,
    __global float* logits) {
  const uint row = (uint)get_group_id(0);
  const uint lid = (uint)get_local_id(0);
  if (cols != 2048U || group_size != 32U) return;
  const uint col = lid * 32U;
  const uint base = row * 2048U + col;
  float dot_sum = 0.0f;
#pragma unroll
  for (uint offset = 0U; offset < 32U; offset += 4U) {
    dot_sum += dot(convert_float4(vload4(0, weights + base + offset)),
                   vload4(0, input + col + offset));
  }
  __local float contributions[64];
  contributions[lid] = dot_sum * weight_scales[row * 64U + lid];
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lid == 0U && row < 257U) {
    float sum = 0.0f;
#pragma unroll
    for (uint index = 0U; index < 64U; ++index) {
      sum += contributions[index];
    }
    logits[row] = sum;
  }
}

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void router_logits_topk8_f32(
    __global const float* logits,
    __global float* shared_gate,
    __global uint* selected_positions,
    __global float* normalized_weights) {
  const uint lid = (uint)get_local_id(0);
  float remaining = logits[lid];
  float best_values[8];
  uint best_ids[8];
  for (uint index = 0U; index < 8U; ++index) {
    const float maximum = work_group_reduce_max(remaining);
    const uint winner = work_group_reduce_min(
        remaining == maximum ? lid : 0xffffffffU);
    best_values[index] = maximum;
    best_ids[index] = winner;
    if (lid == winner) remaining = -INFINITY;
  }
  if (lid != 0U) return;
  shared_gate[0] = logits[256];
  const float maximum = best_values[0];
  float denominator = 0.0f;
  float exponentials[8];
  for (uint index = 0U; index < 8U; ++index) {
    exponentials[index] = exp(best_values[index] - maximum);
    denominator += exponentials[index];
  }
  denominator = fmax(denominator, 0.001f);
  for (uint index = 0U; index < 8U; ++index) {
    selected_positions[index] = best_ids[index];
    normalized_weights[index] = exponentials[index] / denominator;
  }
}

__kernel void linear_attn_conv_f32(__global const float* qkv_mixed,
                                   __global const float* conv_state,
                                   __global const float* weights,
                                   uint channel_count,
                                   uint kernel_size,
                                   __global float* conv_output_raw,
                                   __global float* next_conv_state) {
  const uint channel = (uint)get_global_id(0);
  if (channel >= channel_count) {
    return;
  }
  const uint history = kernel_size - 1U;
  const uint state_base = channel * history;
  const uint weight_base = channel * kernel_size;
  float sum = 0.0f;
  for (uint k = 0; k < history; ++k) {
    sum += conv_state[state_base + k] * weights[weight_base + k];
  }
  sum += qkv_mixed[channel] * weights[weight_base + history];
  conv_output_raw[channel] = sum;
  for (uint k = 0; k + 1U < history; ++k) {
    next_conv_state[state_base + k] = conv_state[state_base + k + 1U];
  }
  if (history > 0U) {
    next_conv_state[state_base + history - 1U] = qkv_mixed[channel];
  }
}

__kernel void linear_preconv_alpha_beta_f32(
    __global const float* alpha,
    __global const float* beta,
    __global const float* dt_bias,
    __global const float* ssm_a,
    __global float* gate,
    __global float* beta_sigmoid) {
  const uint index = (uint)get_global_id(0);
  if (index >= 32U) return;
  const float shifted = alpha[index] + dt_bias[index];
  const float softplus = shifted > 20.0f
      ? shifted
      : log1p(exp(shifted));
  gate[index] = softplus * ssm_a[index];
  const float beta_value = beta[index];
  beta_sigmoid[index] = beta_value >= 0.0f
      ? 1.0f / (1.0f + exp(-beta_value))
      : exp(beta_value) / (1.0f + exp(beta_value));
}

__kernel void linear_preconv_alpha_beta_conv_f32(
    __global const float* alpha,
    __global const float* beta,
    __global const float* dt_bias,
    __global const float* ssm_a,
    __global float* gate,
    __global float* beta_sigmoid,
    __global const float* qkv_mixed,
    __global const float* conv_state,
    __global const float* conv_weights,
    uint channel_count,
    uint kernel_size,
    __global float* conv_output_raw,
    __global float* next_conv_state) {
  const uint index = (uint)get_global_id(0);
  if (index < 32U) {
    const float shifted = alpha[index] + dt_bias[index];
    const float softplus = shifted > 20.0f
        ? shifted
        : log1p(exp(shifted));
    gate[index] = softplus * ssm_a[index];
    const float beta_value = beta[index];
    beta_sigmoid[index] = beta_value >= 0.0f
        ? 1.0f / (1.0f + exp(-beta_value))
        : exp(beta_value) / (1.0f + exp(beta_value));
  }
  if (index >= channel_count) return;
  const uint history = kernel_size - 1U;
  const uint state_base = index * history;
  const uint weight_base = index * kernel_size;
  float sum = 0.0f;
  for (uint k = 0; k < history; ++k) {
    sum += conv_state[state_base + k] * conv_weights[weight_base + k];
  }
  sum += qkv_mixed[index] * conv_weights[weight_base + history];
  conv_output_raw[index] = sum;
  for (uint k = 0; k + 1U < history; ++k) {
    next_conv_state[state_base + k] = conv_state[state_base + k + 1U];
  }
  if (history > 0U) {
    next_conv_state[state_base + history - 1U] = qkv_mixed[index];
  }
}

float sigmoid_f32(float x) {
  if (x >= 0.0f) {
    return 1.0f / (1.0f + exp(-x));
  }
  const float ex = exp(x);
  return ex / (1.0f + ex);
}

int nearest_int_f32(float value) {
  const float shifted = value + 12582912.0f;
  const int bits = as_int(shifted);
  return (bits & 0x007fffff) - 0x00400000;
}

__kernel void ffn_moe_swiglu_f32(__global const float* gate_up,
                                 uint intermediate_size,
                                 uint expert_count,
                                 __global float* output) {
  const uint index = (uint)get_global_id(0);
  const uint total_values = intermediate_size * expert_count;
  if (index >= total_values) {
    return;
  }
  const uint expert = index / intermediate_size;
  const uint row = index - expert * intermediate_size;
  const uint input_base = expert * intermediate_size * 2U;
  const float gate = gate_up[input_base + row];
  const float up = gate_up[input_base + intermediate_size + row];
  output[index] = gate * sigmoid_f32(gate) * up;
}

__kernel void ffn_moe_swiglu_reorder_f32(
    __global const float* gate_up,
    __global const uint* source_expert_by_output,
    uint intermediate_size,
    uint expert_count,
    __global float* output) {
  const uint index = (uint)get_global_id(0);
  const uint total_values = intermediate_size * expert_count;
  if (index >= total_values) {
    return;
  }
  const uint out_expert = index / intermediate_size;
  const uint row = index - out_expert * intermediate_size;
  const uint source_expert = source_expert_by_output[out_expert];
  const uint input_base = source_expert * intermediate_size * 2U;
  const float gate = gate_up[input_base + row];
  const float up = gate_up[input_base + intermediate_size + row];
  output[index] = gate * sigmoid_f32(gate) * up;
}

__kernel void q8k_quantize_f32_blocks(__global const float* input,
                                      uint block_count,
                                      __global char* q8_qs,
                                      __global float* q8_d) {
  const uint block = (uint)get_global_id(0);
  if (block >= block_count) {
    return;
  }
  const uint base = block * 256U;
  float max_value = 0.0f;
  float amax = 0.0f;
  for (uint i = 0; i < 256U; ++i) {
    const float value = input[base + i];
    const float abs_value = fabs(value);
    if (abs_value > amax) {
      amax = abs_value;
      max_value = value;
    }
  }
  if (amax == 0.0f) {
    q8_d[block] = 0.0f;
    for (uint i = 0; i < 256U; ++i) {
      q8_qs[base + i] = 0;
    }
    return;
  }
  const float iscale = -127.0f / max_value;
  for (uint i = 0; i < 256U; ++i) {
    const int quantized = min(127, nearest_int_f32(iscale * input[base + i]));
    q8_qs[base + i] = (char)quantized;
  }
  q8_d[block] = 1.0f / iscale;
}

__kernel void q8k_quantize_f32_blocks_with_bsums(
    __global const float* input,
    uint block_count,
    __global char* q8_qs,
    __global short* q8_bsums,
    __global float* q8_d) {
  const uint block = (uint)get_global_id(0);
  if (block >= block_count) {
    return;
  }
  const uint base = block * 256U;
  float max_value = 0.0f;
  float amax = 0.0f;
  for (uint i = 0; i < 256U; ++i) {
    const float value = input[base + i];
    const float abs_value = fabs(value);
    if (abs_value > amax) {
      amax = abs_value;
      max_value = value;
    }
  }
  if (amax == 0.0f) {
    q8_d[block] = 0.0f;
    for (uint i = 0; i < 256U; ++i) {
      q8_qs[base + i] = 0;
    }
    for (uint group = 0; group < 16U; ++group) {
      q8_bsums[block * 16U + group] = 0;
    }
    return;
  }
  const float iscale = -127.0f / max_value;
  for (uint group = 0; group < 16U; ++group) {
    int sum = 0;
    for (uint i = 0; i < 16U; ++i) {
      const uint index = base + group * 16U + i;
      const int quantized = min(127, nearest_int_f32(iscale * input[index]));
      q8_qs[index] = (char)quantized;
      sum += quantized;
    }
    q8_bsums[block * 16U + group] = (short)sum;
  }
  q8_d[block] = 1.0f / iscale;
}

__kernel void q8k_quantize_f32_blocks_with_bsums_parallel(
    __global const float* input,
    uint block_count,
    __global char* q8_qs,
    __global short* q8_bsums,
    __global float* q8_d) {
  const uint block = (uint)get_group_id(0);
  const uint lid = (uint)get_local_id(0);
  if (block >= block_count) return;
  const uint index = block * 256U + lid;
  __local float values[256];
  __local float iscale_shared;
  __local int quantized[256];
  values[lid] = input[index];
  barrier(CLK_LOCAL_MEM_FENCE);
  const float amax = work_group_reduce_max(fabs(values[lid]));
  const uint winner = work_group_reduce_min(
      fabs(values[lid]) == amax ? lid : 0xffffffffU);
  if (lid == 0U) {
    const float max_value = values[winner];
    iscale_shared = amax == 0.0f ? 0.0f : -127.0f / max_value;
    q8_d[block] = amax == 0.0f ? 0.0f : 1.0f / iscale_shared;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  const int q = iscale_shared == 0.0f
      ? 0
      : min(127, nearest_int_f32(iscale_shared * values[lid]));
  quantized[lid] = q;
  q8_qs[index] = (char)q;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lid < 16U) {
    int sum = 0;
    for (uint i = 0; i < 16U; ++i) {
      sum += quantized[lid * 16U + i];
    }
    q8_bsums[block * 16U + lid] = (short)sum;
  }
}

__kernel void ffn_swiglu_q8_blocks_with_bsums_parallel(
    __global const float* gate_up,
    __global char* q8_qs,
    __global short* q8_bsums,
    __global float* q8_d) {
  const uint block = (uint)get_group_id(0);
  const uint lid = (uint)get_local_id(0);
  if (block >= 18U) return;
  const uint expert = block >> 1;
  const uint block_in_expert = block & 1U;
  const uint gate_base = expert * 1024U + block_in_expert * 256U;
  const uint up_base = gate_base + 512U;
  const float gate = gate_up[gate_base + lid];
  __local float values[256];
  __local float iscale_shared;
  __local int quantized[256];
  values[lid] = gate * sigmoid_f32(gate) * gate_up[up_base + lid];
  barrier(CLK_LOCAL_MEM_FENCE);
  const float amax = work_group_reduce_max(fabs(values[lid]));
  const uint winner = work_group_reduce_min(
      fabs(values[lid]) == amax ? lid : 0xffffffffU);
  if (lid == 0U) {
    const float max_value = values[winner];
    iscale_shared = amax == 0.0f ? 0.0f : -127.0f / max_value;
    q8_d[block] = amax == 0.0f ? 0.0f : 1.0f / iscale_shared;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  const int q = iscale_shared == 0.0f
      ? 0
      : min(127, nearest_int_f32(iscale_shared * values[lid]));
  quantized[lid] = q;
  q8_qs[block * 256U + lid] = (char)q;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lid < 16U) {
    int sum = 0;
    for (uint i = 0; i < 16U; ++i) {
      sum += quantized[lid * 16U + i];
    }
    q8_bsums[block * 16U + lid] = (short)sum;
  }
}

__kernel void q6k_selected_down_matvec_row(
    __global const uchar* selected_raw,
    __global const char* q8_qs,
    __global const float* q8_d,
    uint rows_per_expert,
    uint blocks_per_row,
    __global float* out) {
  const uint flat_row = (uint)get_global_id(0);
  const uint selected = flat_row / rows_per_expert;
  const uint local_row = flat_row - selected * rows_per_expert;
  __global const uchar* row =
      selected_raw + ((ulong)selected * (ulong)rows_per_expert +
                      (ulong)local_row) * (ulong)blocks_per_row * 210UL;
  __global const char* expert_q8 =
      q8_qs + (ulong)selected * (ulong)blocks_per_row * 256UL;
  __global const float* expert_q8_d =
      q8_d + (ulong)selected * (ulong)blocks_per_row;
  float sum = 0.0f;
  for (uint block_index = 0; block_index < blocks_per_row; ++block_index) {
    __global const uchar* block = row + (ulong)block_index * 210UL;
    __global const char* q8 = expert_q8 + (ulong)block_index * 256UL;
    __global const char* scales = (__global const char*)(block + 192);
    const float combined_scale =
        half_to_float(load_le16(block + 208)) * expert_q8_d[block_index];
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
  out[flat_row] = sum;
}

__kernel void q6k_lm_head_sparse_exact_candidates(
    __global const uchar* packed,
    __global const char* q8_qs,
    __global const float* q8_d,
    __global const int* candidate_ids,
    uint block_count,
    uint candidates_per_block,
    uint blocks_per_row,
    uint rows_per_tile,
    __global float* logits,
    __global float* candidate_values,
    float min_secondary_logit) {
  const uint candidate = (uint)get_global_id(0);
  const uint candidate_count = block_count * candidates_per_block;
  if (candidate >= candidate_count) return;
  const uint block = candidate / candidates_per_block;
  const uint rank = candidate - block * candidates_per_block;
  const uint candidate_slot = block * 8U + rank;
  if (rank > 0U && candidate_values[candidate_slot] < min_secondary_logit) {
    return;
  }
  const int signed_row = candidate_ids[candidate_slot];
  if (signed_row < 0) return;
  const uint row = (uint)signed_row;
  const uint row_tile = row / rows_per_tile;
  const uint row_lane = row - row_tile * rows_per_tile;
  const uint tile_block_bytes = rows_per_tile * 210U;
  float sum = 0.0f;
  for (uint block_index = 0; block_index < blocks_per_row; ++block_index) {
    __global const uchar* tile =
        packed + ((ulong)row_tile * (ulong)blocks_per_row +
                  (ulong)block_index) * (ulong)tile_block_bytes;
    __global const uchar* ql0 = tile;
    __global const uchar* qh0 = ql0 + (ulong)rows_per_tile * 64UL;
    __global const uchar* ql1 = qh0 + (ulong)rows_per_tile * 32UL;
    __global const uchar* qh1 = ql1 + (ulong)rows_per_tile * 64UL;
    __global const char* scales =
        (__global const char*)(qh1 + (ulong)rows_per_tile * 32UL);
    __global const uchar* d_base =
        (__global const uchar*)scales + (ulong)rows_per_tile * 16UL;
    __global const char* q8 = q8_qs + (ulong)block_index * 256UL;
    const float combined_scale =
        half_to_float(load_le16(d_base + (ulong)row_lane * 2UL)) *
        q8_d[block_index];
    int block_dot = 0;
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
        for (int lane = lane_begin; lane < lane_begin + 16; lane += 4) {
          const uchar4 high = vload4(0, qh + lane);
          const uchar4 low0 = vload4(0, ql + lane);
          const uchar4 low1 = vload4(0, ql + 32 + lane);
          const char4 value0 = convert_char4(
              (low0 & (uchar4)(15)) | ((high & (uchar4)(3)) << 4)) -
              (char4)(32);
          const char4 value1 = convert_char4(
              (low1 & (uchar4)(15)) |
              (((high >> 2) & (uchar4)(3)) << 4)) - (char4)(32);
          const char4 value2 = convert_char4(
              (low0 >> 4) | (((high >> 4) & (uchar4)(3)) << 4)) -
              (char4)(32);
          const char4 value3 = convert_char4(
              (low1 >> 4) | (((high >> 6) & (uchar4)(3)) << 4)) -
              (char4)(32);
#if defined(IQ36_USE_INTEGER_DOT)
          block_dot += scale0 * dot(value0, vload4(0, q8 + base + lane));
          block_dot += scale1 * dot(
              value1, vload4(0, q8 + base + 32 + lane));
          block_dot += scale2 * dot(
              value2, vload4(0, q8 + base + 64 + lane));
          block_dot += scale3 * dot(
              value3, vload4(0, q8 + base + 96 + lane));
#else
          for (int element = 0; element < 4; ++element) {
            block_dot += scale0 * (int)value0[element] *
                (int)q8[base + lane + element];
            block_dot += scale1 * (int)value1[element] *
                (int)q8[base + 32 + lane + element];
            block_dot += scale2 * (int)value2[element] *
                (int)q8[base + 64 + lane + element];
            block_dot += scale3 * (int)value3[element] *
                (int)q8[base + 96 + lane + element];
          }
#endif
        }
      }
    }
    sum += combined_scale * (float)block_dot;
  }
  logits[row] = sum;
  candidate_values[candidate_slot] = sum;
}

__kernel void sort_partial_top8_blocks_in_place_f32(
    __global int* ids,
    __global float* values,
    uint block_count) {
  const uint block = (uint)get_global_id(0);
  if (block >= block_count) return;
  const uint base = block * 8U;
  for (uint source = 1U; source < 8U; ++source) {
    const float value = values[base + source];
    const int id = ids[base + source];
    uint destination = source;
    while (destination > 0U &&
           (values[base + destination - 1U] < value ||
            (values[base + destination - 1U] == value &&
             ids[base + destination - 1U] > id))) {
      values[base + destination] = values[base + destination - 1U];
      ids[base + destination] = ids[base + destination - 1U];
      --destination;
    }
    values[base + destination] = value;
    ids[base + destination] = id;
  }
}

__kernel void q6k_selected_down_matvec_rowstripe(
    __global const uchar* selected_scratch,
    __global const char* q8_qs,
    __global const float* q8_d,
    uint rows_per_expert,
    uint blocks_per_row,
    uint rows_per_tile,
    __global float* out) {
  const uint flat_row = (uint)get_global_id(0);
  const uint selected = flat_row / rows_per_expert;
  const uint local_row = flat_row - selected * rows_per_expert;
  const uint row_tile_count =
      (rows_per_expert + rows_per_tile - 1U) / rows_per_tile;
  const uint row_tile = local_row / rows_per_tile;
  const uint row_lane = local_row - row_tile * rows_per_tile;
  const uint tile_block_bytes = rows_per_tile * 210U;
  __global const char* expert_q8 =
      q8_qs + (ulong)selected * (ulong)blocks_per_row * 256UL;
  __global const float* expert_q8_d =
      q8_d + (ulong)selected * (ulong)blocks_per_row;
  float sum = 0.0f;
  for (uint block_index = 0; block_index < blocks_per_row; ++block_index) {
    __global const uchar* tile =
        selected_scratch +
        (((ulong)selected * (ulong)row_tile_count + (ulong)row_tile) *
             (ulong)blocks_per_row +
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
    int block_dot = 0;
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
        for (int lane = lane_begin; lane < lane_begin + 16; lane += 4) {
          const uchar4 high = vload4(0, qh + lane);
          const uchar4 low0 = vload4(0, ql + lane);
          const uchar4 low1 = vload4(0, ql + 32 + lane);
          const char4 value0 = convert_char4(
              (low0 & (uchar4)(15)) | ((high & (uchar4)(3)) << 4)) -
              (char4)(32);
          const char4 value1 = convert_char4(
              (low1 & (uchar4)(15)) |
              (((high >> 2) & (uchar4)(3)) << 4)) - (char4)(32);
          const char4 value2 = convert_char4(
              (low0 >> 4) | (((high >> 4) & (uchar4)(3)) << 4)) -
              (char4)(32);
          const char4 value3 = convert_char4(
              (low1 >> 4) | (((high >> 6) & (uchar4)(3)) << 4)) -
              (char4)(32);
#if defined(IQ36_USE_INTEGER_DOT)
          block_dot += scale0 * dot(value0, vload4(0, q8 + base + lane));
          block_dot += scale1 * dot(
              value1, vload4(0, q8 + base + 32 + lane));
          block_dot += scale2 * dot(
              value2, vload4(0, q8 + base + 64 + lane));
          block_dot += scale3 * dot(
              value3, vload4(0, q8 + base + 96 + lane));
#else
          for (int element = 0; element < 4; ++element) {
            block_dot += scale0 * (int)value0[element] *
                (int)q8[base + lane + element];
            block_dot += scale1 * (int)value1[element] *
                (int)q8[base + 32 + lane + element];
            block_dot += scale2 * (int)value2[element] *
                (int)q8[base + 64 + lane + element];
            block_dot += scale3 * (int)value3[element] *
                (int)q8[base + 96 + lane + element];
          }
#endif
        }
      }
    }
    sum += combined_scale * (float)block_dot;
  }
  out[flat_row] = sum;
}

__kernel void q6k_selected_down_matvec_rowstripe_localq8(
    __global const uchar* selected_scratch,
    __global const char* q8_qs,
    __global const float* q8_d,
    uint rows_per_expert,
    uint blocks_per_row,
    uint rows_per_tile,
    __global float* out) {
  const uint flat_row = (uint)get_global_id(0);
  const uint selected = flat_row / rows_per_expert;
  const uint local_row = flat_row - selected * rows_per_expert;
  const uint row_tile_count =
      (rows_per_expert + rows_per_tile - 1U) / rows_per_tile;
  const uint row_tile = local_row / rows_per_tile;
  const uint row_lane = local_row - row_tile * rows_per_tile;
  const uint tile_block_bytes = rows_per_tile * 210U;
  __global const char* expert_q8 =
      q8_qs + (ulong)selected * (ulong)blocks_per_row * 256UL;
  __global const float* expert_q8_d =
      q8_d + (ulong)selected * (ulong)blocks_per_row;
  const uint lid = (uint)get_local_id(0);
  const uint lsize = (uint)get_local_size(0);
  if (blocks_per_row > 8U) return;
  __local char local_q8[2048];
  __local float local_q8_d[8];
  for (uint q8_i = lid; q8_i < blocks_per_row * 256U; q8_i += lsize) {
    local_q8[q8_i] = expert_q8[q8_i];
  }
  for (uint block_index = lid; block_index < blocks_per_row;
       block_index += lsize) {
    local_q8_d[block_index] = expert_q8_d[block_index];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float sum = 0.0f;
  for (uint block_index = 0; block_index < blocks_per_row; ++block_index) {
    __global const uchar* tile =
        selected_scratch +
        (((ulong)selected * (ulong)row_tile_count + (ulong)row_tile) *
             (ulong)blocks_per_row +
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
    const float combined_scale =
        half_to_float(load_le16(d_base + (ulong)row_lane * 2UL)) *
        local_q8_d[block_index];
    int block_dot = 0;
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
        for (int lane = lane_begin; lane < lane_begin + 16; lane += 4) {
          const uchar4 high = vload4(0, qh + lane);
          const uchar4 low0 = vload4(0, ql + lane);
          const uchar4 low1 = vload4(0, ql + 32 + lane);
          const char4 value0 = convert_char4(
              (low0 & (uchar4)(15)) | ((high & (uchar4)(3)) << 4)) -
              (char4)(32);
          const char4 value1 = convert_char4(
              (low1 & (uchar4)(15)) |
              (((high >> 2) & (uchar4)(3)) << 4)) - (char4)(32);
          const char4 value2 = convert_char4(
              (low0 >> 4) | (((high >> 4) & (uchar4)(3)) << 4)) -
              (char4)(32);
          const char4 value3 = convert_char4(
              (low1 >> 4) | (((high >> 6) & (uchar4)(3)) << 4)) -
              (char4)(32);
#if defined(IQ36_USE_INTEGER_DOT)
          block_dot += scale0 * dot(
              value0, vload4(0, local_q8 + block_index * 256U + base + lane));
          block_dot += scale1 * dot(
              value1,
              vload4(0, local_q8 + block_index * 256U + base + 32 + lane));
          block_dot += scale2 * dot(
              value2,
              vload4(0, local_q8 + block_index * 256U + base + 64 + lane));
          block_dot += scale3 * dot(
              value3,
              vload4(0, local_q8 + block_index * 256U + base + 96 + lane));
#else
          for (int element = 0; element < 4; ++element) {
            block_dot += scale0 * (int)value0[element] *
                (int)local_q8[block_index * 256U + base + lane + element];
            block_dot += scale1 * (int)value1[element] *
                (int)local_q8[
                    block_index * 256U + base + 32 + lane + element];
            block_dot += scale2 * (int)value2[element] *
                (int)local_q8[
                    block_index * 256U + base + 64 + lane + element];
            block_dot += scale3 * (int)value3[element] *
                (int)local_q8[
                    block_index * 256U + base + 96 + lane + element];
          }
#endif
        }
      }
    }
    sum += combined_scale * (float)block_dot;
  }
  out[flat_row] = sum;
}

__kernel void q6k_selected_down_matvec_row_expert8(
    __global const uchar* raw0,
    __global const uchar* raw1,
    __global const uchar* raw2,
    __global const uchar* raw3,
    __global const uchar* raw4,
    __global const uchar* raw5,
    __global const uchar* raw6,
    __global const uchar* raw7,
    __global const char* q8_qs,
    __global const float* q8_d,
    uint rows_per_expert,
    uint blocks_per_row,
    __global float* out) {
  const uint flat_row = (uint)get_global_id(0);
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
  __global const uchar* row =
      raw + (ulong)local_row * (ulong)blocks_per_row * 210UL;
  __global const char* expert_q8 =
      q8_qs + (ulong)selected * (ulong)blocks_per_row * 256UL;
  __global const float* expert_q8_d =
      q8_d + (ulong)selected * (ulong)blocks_per_row;
  float sum = 0.0f;
  for (uint block_index = 0; block_index < blocks_per_row; ++block_index) {
    __global const uchar* block = row + (ulong)block_index * 210UL;
    __global const char* q8 = expert_q8 + (ulong)block_index * 256UL;
    __global const char* scales = (__global const char*)(block + 192);
    const float combined_scale =
        half_to_float(load_le16(block + 208)) * expert_q8_d[block_index];
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
  out[flat_row] = sum;
}

__kernel void q6k_selected_down_matvec_rowstripe_expert8(
    __global const uchar* raw0,
    __global const uchar* raw1,
    __global const uchar* raw2,
    __global const uchar* raw3,
    __global const uchar* raw4,
    __global const uchar* raw5,
    __global const uchar* raw6,
    __global const uchar* raw7,
    __global const char* q8_qs,
    __global const float* q8_d,
    uint rows_per_expert,
    uint blocks_per_row,
    uint rows_per_tile,
    __global float* out) {
  const uint flat_row = (uint)get_global_id(0);
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
      q8_qs + (ulong)selected * (ulong)blocks_per_row * 256UL;
  __global const float* expert_q8_d =
      q8_d + (ulong)selected * (ulong)blocks_per_row;
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
  out[flat_row] = sum;
}

__kernel void q6k_selected_down_matvec_rowstripe_expert8_plus_shared_raw(
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
    uint rows_per_expert,
    uint blocks_per_row,
    uint rows_per_tile,
    __global float* selected_out,
    __global float* shared_out) {
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
    selected_out[flat_row] = sum;
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
  shared_out[row] = sum;
}

inline float q6k_qwen36_down_coalesced_dot(
    __global const uchar* rowstripe,
    uint local_row,
    __global const char* q8,
    __global const float* q8_d) {
  const uint row_tile = local_row >> 5;
  const uint row_lane = local_row & 31U;
  const uint tile_block_bytes = 32U * 210U;
  float sum = 0.0f;
  #pragma unroll
  for (uint block_index = 0; block_index < 2U; ++block_index) {
    __global const char* block_q8 =
        q8 + ((ulong)block_index << 8);
    __global const uchar* tile =
        rowstripe + ((ulong)row_tile * 2UL +
                     (ulong)block_index) * (ulong)tile_block_bytes;
    __global const uchar* ql0 = tile;
    __global const uchar* qh0 = ql0 + 32UL * 64UL;
    __global const uchar* ql1 = qh0 + 32UL * 32UL;
    __global const uchar* qh1 = ql1 + 32UL * 64UL;
    __global const char* scales =
        (__global const char*)(qh1 + 32UL * 32UL);
    __global const uchar* d_base =
        (__global const uchar*)scales + 32UL * 16UL;
    const ushort d_bits =
        (ushort)d_base[row_lane] |
        ((ushort)d_base[32U + row_lane] << 8);
    const float combined_scale =
        half_to_float(d_bits) * q8_d[block_index];
    int block_dot = 0;
    for (int half_index = 0; half_index < 2; ++half_index) {
      __global const uchar* ql = half_index == 0 ? ql0 : ql1;
      __global const uchar* qh = half_index == 0 ? qh0 : qh1;
      const uint scale_base = (uint)half_index * 8U;
      const int base = half_index * 128;
      for (int scale_group = 0; scale_group < 2; ++scale_group) {
        const int lane_begin = scale_group * 16;
        const int scale0 = (int)scales[
            ((ulong)(scale_base + (uint)scale_group) << 5) + row_lane];
        const int scale1 = (int)scales[
            ((ulong)(scale_base + (uint)scale_group + 2U) << 5) +
            row_lane];
        const int scale2 = (int)scales[
            ((ulong)(scale_base + (uint)scale_group + 4U) << 5) +
            row_lane];
        const int scale3 = (int)scales[
            ((ulong)(scale_base + (uint)scale_group + 6U) << 5) +
            row_lane];
        for (int lane = lane_begin; lane < lane_begin + 16; lane += 4) {
          const ulong qh_index = ((ulong)(uint)lane << 5) + row_lane;
          const ulong ql_low_index = ((ulong)(uint)lane << 5) + row_lane;
          const ulong ql_high_index =
              ((ulong)(uint)(32 + lane) << 5) + row_lane;
          const uchar4 high = (uchar4)(
              qh[qh_index], qh[qh_index + 32UL],
              qh[qh_index + 64UL], qh[qh_index + 96UL]);
          const uchar4 low0 = (uchar4)(
              ql[ql_low_index], ql[ql_low_index + 32UL],
              ql[ql_low_index + 64UL], ql[ql_low_index + 96UL]);
          const uchar4 low1 = (uchar4)(
              ql[ql_high_index], ql[ql_high_index + 32UL],
              ql[ql_high_index + 64UL], ql[ql_high_index + 96UL]);
          const char4 value0 = convert_char4(
              (low0 & (uchar4)(15)) | ((high & (uchar4)(3)) << 4)) -
              (char4)(32);
          const char4 value1 = convert_char4(
              (low1 & (uchar4)(15)) |
              (((high >> 2) & (uchar4)(3)) << 4)) - (char4)(32);
          const char4 value2 = convert_char4(
              (low0 >> 4) | (((high >> 4) & (uchar4)(3)) << 4)) -
              (char4)(32);
          const char4 value3 = convert_char4(
              (low1 >> 4) | (((high >> 6) & (uchar4)(3)) << 4)) -
              (char4)(32);
#if defined(IQ36_USE_INTEGER_DOT)
          block_dot += scale0 * dot(
              value0, vload4(0, block_q8 + base + lane));
          block_dot += scale1 * dot(
              value1, vload4(0, block_q8 + base + 32 + lane));
          block_dot += scale2 * dot(
              value2, vload4(0, block_q8 + base + 64 + lane));
          block_dot += scale3 * dot(
              value3, vload4(0, block_q8 + base + 96 + lane));
#else
          for (int element = 0; element < 4; ++element) {
            block_dot += scale0 * (int)value0[element] *
                (int)block_q8[base + lane + element];
            block_dot += scale1 * (int)value1[element] *
                (int)block_q8[base + 32 + lane + element];
            block_dot += scale2 * (int)value2[element] *
                (int)block_q8[base + 64 + lane + element];
            block_dot += scale3 * (int)value3[element] *
                (int)block_q8[base + 96 + lane + element];
          }
#endif
        }
      }
    }
    sum += combined_scale * (float)block_dot;
  }
  return sum;
}

__kernel void q6k_all_expert_rowstripe_coalesced_topk8_plus_shared(
    __global const uchar* all_expert_rowstripe,
    __global const uchar* shared_rowstripe,
    __global const uint* selected_positions,
    __global const char* selected_q8_qs,
    __global const float* selected_q8_d,
    __global const char* shared_q8_qs,
    __global const float* shared_q8_d,
    __global float* selected_out,
    __global float* shared_out) {
  const uint flat_row = (uint)get_global_id(0);
  if (flat_row < 16384U) {
    const uint selected = flat_row >> 11;
    const uint local_row = flat_row & 2047U;
    const uint material_expert = selected_positions[selected];
    __global const uchar* rowstripe =
        all_expert_rowstripe + (ulong)material_expert * 860160UL;
    __global const char* q8 =
        selected_q8_qs + ((ulong)selected << 9);
    __global const float* d =
        selected_q8_d + ((ulong)selected << 1);
    selected_out[flat_row] = q6k_qwen36_down_coalesced_dot(
        rowstripe, local_row, q8, d);
    return;
  }
  const uint shared_row = flat_row - 16384U;
  shared_out[shared_row] = q6k_qwen36_down_coalesced_dot(
      shared_rowstripe, shared_row, shared_q8_qs, shared_q8_d);
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

__kernel void q6k_selected_down_matvec_rowstripe_expert8_plus_shared_tail_rowgroup_reduce_raw(
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
    __global const float* attn_residual,
    uint rows_per_expert,
    uint blocks_per_row,
    uint rows_per_tile,
    __global float* layer_output) {
  const uint row = (uint)get_group_id(0);
  const uint contributor = (uint)get_local_id(0);
  __local float partial[16];

  float value = 0.0f;
  if ((uint)get_local_size(0) == 16U && row < rows_per_expert) {
    if (contributor < 8U) {
      __global const uchar* raw = raw0;
      if (contributor == 1U) {
        raw = raw1;
      } else if (contributor == 2U) {
        raw = raw2;
      } else if (contributor == 3U) {
        raw = raw3;
      } else if (contributor == 4U) {
        raw = raw4;
      } else if (contributor == 5U) {
        raw = raw5;
      } else if (contributor == 6U) {
        raw = raw6;
      } else if (contributor == 7U) {
        raw = raw7;
      }
      const uint row_tile = row / rows_per_tile;
      const uint row_lane = row - row_tile * rows_per_tile;
      const uint tile_block_bytes = rows_per_tile * 210U;
      __global const char* expert_q8 =
          selected_q8_qs +
          (ulong)contributor * (ulong)blocks_per_row * 256UL;
      __global const float* expert_q8_d =
          selected_q8_d + (ulong)contributor * (ulong)blocks_per_row;
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
                  (((int)(ql[lane] & (uchar)15) |
                    (((high >> 0) & 3) << 4)) -
                   32);
              lane_sums[lane_index] +=
                  scale1 * (int)q8[base + 32 + lane] *
                  (((int)(ql[32 + lane] & (uchar)15) |
                    (((high >> 2) & 3) << 4)) -
                   32);
              lane_sums[lane_index] +=
                  scale2 * (int)q8[base + 64 + lane] *
                  (((int)(ql[lane] >> 4) | (((high >> 4) & 3) << 4)) -
                   32);
              lane_sums[lane_index] +=
                  scale3 * (int)q8[base + 96 + lane] *
                  (((int)(ql[32 + lane] >> 4) |
                    (((high >> 6) & 3) << 4)) -
                   32);
            }
          }
        }
        for (int lane = 0; lane < 8; ++lane) {
          sum += combined_scale * (float)lane_sums[lane];
        }
      }
      value = sum * weights[contributor];
    } else if (contributor == 8U) {
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
                  (((int)(ql[lane] & (uchar)15) |
                    (((high >> 0) & 3) << 4)) -
                   32);
              lane_sums[lane_index] +=
                  scale1 * (int)q8[base + 32 + lane] *
                  (((int)(ql[32 + lane] & (uchar)15) |
                    (((high >> 2) & 3) << 4)) -
                   32);
              lane_sums[lane_index] +=
                  scale2 * (int)q8[base + 64 + lane] *
                  (((int)(ql[lane] >> 4) | (((high >> 4) & 3) << 4)) -
                   32);
              lane_sums[lane_index] +=
                  scale3 * (int)q8[base + 96 + lane] *
                  (((int)(ql[32 + lane] >> 4) |
                    (((high >> 6) & 3) << 4)) -
                   32);
            }
          }
        }
        for (int lane = 0; lane < 8; ++lane) {
          sum += combined_scale * (float)lane_sums[lane];
        }
      }
      value = sum * sigmoid_f32(gate[0]);
    }
  }
  partial[contributor] = value;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (contributor == 0U && row < rows_per_expert) {
    float acc = attn_residual[row];
    for (uint i = 0; i < 9U; ++i) {
      acc += partial[i];
    }
    layer_output[row] = acc;
  }
}

__kernel void f32_topk8_blocks(__global const float* values,
                               uint value_count,
                               uint block_size,
                               __global int* top_ids,
                               __global float* top_values) {
  const uint block = (uint)get_global_id(0);
  const uint begin = block * block_size;
  const uint end = min(begin + block_size, value_count);
  float best_values[8];
  int best_ids[8];
  for (int i = 0; i < 8; ++i) {
    best_values[i] = -INFINITY;
    best_ids[i] = -1;
  }
  for (uint row = begin; row < end; ++row) {
    const float value = values[row];
    int pos = 0;
    while (pos < 8 && best_values[pos] >= value) {
      ++pos;
    }
    if (pos < 8) {
      for (int dst = 7; dst > pos; --dst) {
        best_values[dst] = best_values[dst - 1];
        best_ids[dst] = best_ids[dst - 1];
      }
      best_values[pos] = value;
      best_ids[pos] = (int)row;
    }
  }
  const uint out_base = block * 8U;
  for (uint i = 0; i < 8U; ++i) {
    top_ids[out_base + i] = best_ids[i];
    top_values[out_base + i] = best_values[i];
  }
}

__kernel void ffn_router_topk8_normalize_parallel8(
    __global const float* logits,
    __global uint* selected_positions,
    __global float* normalized_weights) {
  const uint lid = (uint)get_local_id(0);
  __local float lane_values[8 * 8];
  __local uint lane_ids[8 * 8];
  float best_values[8];
  uint best_ids[8];
  for (uint i = 0; i < 8U; ++i) {
    best_values[i] = -INFINITY;
    best_ids[i] = 0U;
  }
  const uint begin = lid * 32U;
  for (uint row = begin; row < begin + 32U; ++row) {
    const float value = logits[row];
    uint pos = 0U;
    while (pos < 8U &&
           (best_values[pos] > value ||
            (best_values[pos] == value && best_ids[pos] < row))) {
      ++pos;
    }
    if (pos < 8U) {
      for (uint dst = 7U; dst > pos; --dst) {
        best_values[dst] = best_values[dst - 1U];
        best_ids[dst] = best_ids[dst - 1U];
      }
      best_values[pos] = value;
      best_ids[pos] = row;
    }
  }
  for (uint i = 0; i < 8U; ++i) {
    lane_values[lid * 8U + i] = best_values[i];
    lane_ids[lid * 8U + i] = best_ids[i];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lid == 0U) {
    for (uint i = 0; i < 8U; ++i) {
      best_values[i] = -INFINITY;
      best_ids[i] = 0U;
    }
    for (uint candidate = 0; candidate < 64U; ++candidate) {
      const float value = lane_values[candidate];
      const uint id = lane_ids[candidate];
      uint pos = 0U;
      while (pos < 8U &&
             (best_values[pos] > value ||
              (best_values[pos] == value && best_ids[pos] < id))) {
        ++pos;
      }
      if (pos < 8U) {
        for (uint dst = 7U; dst > pos; --dst) {
          best_values[dst] = best_values[dst - 1U];
          best_ids[dst] = best_ids[dst - 1U];
        }
        best_values[pos] = value;
        best_ids[pos] = id;
      }
    }
    const float maximum = best_values[0];
    float denominator = 0.0f;
    float exponentials[8];
    for (uint i = 0; i < 8U; ++i) {
      exponentials[i] = exp(best_values[i] - maximum);
      denominator += exponentials[i];
    }
    denominator = fmax(denominator, 0.001f);
    for (uint i = 0; i < 8U; ++i) {
      selected_positions[i] = best_ids[i];
      normalized_weights[i] = exponentials[i] / denominator;
    }
  }
}

__kernel void f32_topk8_merge_blocks_parallel(
    __global const int* partial_ids,
    __global const float* partial_values,
    uint block_count,
    uint output_count,
    __global int* top_ids,
    __global float* top_values) {
  const uint lid = (uint)get_local_id(0);
  float best_values[8];
  int best_ids[8];
  for (uint i = 0; i < 8U; ++i) {
    best_values[i] = -INFINITY;
    best_ids[i] = -1;
  }
  const uint candidate_count = block_count * 8U;
  for (uint candidate = lid; candidate < candidate_count;
       candidate += (uint)get_local_size(0)) {
    const float value = partial_values[candidate];
    const int id = partial_ids[candidate];
    uint pos = 0U;
    while (pos < 8U &&
           (best_values[pos] > value ||
            (best_values[pos] == value && best_ids[pos] < id))) {
      ++pos;
    }
    if (pos < 8U) {
      for (uint dst = 7U; dst > pos; --dst) {
        best_values[dst] = best_values[dst - 1U];
        best_ids[dst] = best_ids[dst - 1U];
      }
      best_values[pos] = value;
      best_ids[pos] = id;
    }
  }
  uint head = 0U;
  for (uint output = 0U; output < min(output_count, 8U); ++output) {
    const float lane_value = head < 8U ? best_values[head] : -INFINITY;
    const int lane_id = head < 8U ? best_ids[head] : 0x7fffffff;
    const float maximum = work_group_reduce_max(lane_value);
    const int winner = work_group_reduce_min(
        lane_value == maximum ? lane_id : 0x7fffffff);
    if (lane_id == winner) ++head;
    if (lid == 0U) {
      top_ids[output] = winner;
      top_values[output] = maximum;
    }
  }
}

__kernel void f32_topk16_blocks(__global const float* values,
                                uint value_count,
                                uint block_size,
                                __global int* top_ids,
                                __global float* top_values) {
  const uint block = (uint)get_global_id(0);
  const uint begin = block * block_size;
  const uint end = min(begin + block_size, value_count);
  float best_values[16];
  int best_ids[16];
  for (int i = 0; i < 16; ++i) {
    best_values[i] = -INFINITY;
    best_ids[i] = -1;
  }
  for (uint row = begin; row < end; ++row) {
    const float value = values[row];
    int pos = 0;
    while (pos < 16 && best_values[pos] >= value) {
      ++pos;
    }
    if (pos < 16) {
      for (int dst = 15; dst > pos; --dst) {
        best_values[dst] = best_values[dst - 1];
        best_ids[dst] = best_ids[dst - 1];
      }
      best_values[pos] = value;
      best_ids[pos] = (int)row;
    }
  }
  const uint out_base = block * 16U;
  for (uint i = 0; i < 16U; ++i) {
    top_ids[out_base + i] = best_ids[i];
    top_values[out_base + i] = best_values[i];
  }
}

__kernel void qkv_delta_sparse_overlay_f32(
    __global const float* source,
    __global const int* selected_indices,
    uint selected_count,
    __global float* output) {
  const uint row = (uint)get_global_id(0);
  if (row >= selected_count) {
    return;
  }
  const int index = selected_indices[row];
  if (index < 0) {
    return;
  }
  output[index] = source[index];
}

__kernel void qkv_delta_blockq16_overlay_f32(
    __global const float* base,
    __global const int* selected_indices,
    __global const short* selected_q_delta,
    __global const float* block_scales,
    uint selected_count,
    __global float* output) {
  const uint row = (uint)get_global_id(0);
  if (row >= selected_count) {
    return;
  }
  const int index = selected_indices[row];
  if (index < 0) {
    return;
  }
  const uint block = ((uint)index) >> 6;
  output[index] = base[index] + ((float)selected_q_delta[row]) *
      block_scales[block];
}

__kernel void ffn_moe_weighted_aggregate_f32(
    __global const float* selected_down,
    __global const float* weights,
    uint hidden_size,
    uint expert_count,
    __global float* weighted,
    __global float* moe_out) {
  const uint row = (uint)get_global_id(0);
  if (row >= hidden_size) {
    return;
  }
  float acc = 0.0f;
  for (uint expert = 0; expert < expert_count; ++expert) {
    const uint index = expert * hidden_size + row;
    const float value = selected_down[index] * weights[expert];
    weighted[index] = value;
    acc += value;
  }
  moe_out[row] = acc;
}

__kernel void shared_expert_gate_apply_f32(
    __global const float* shared_down,
    __global const float* gate,
    uint hidden_size,
    __global float* gate_sigmoid,
    __global float* shared_gated) {
  const uint index = (uint)get_global_id(0);
  const float sigmoid = sigmoid_f32(gate[0]);
  if (index == 0U) {
    gate_sigmoid[0] = sigmoid;
  }
  if (index >= hidden_size) {
    return;
  }
  shared_gated[index] = shared_down[index] * sigmoid;
}

__kernel void ffn_output_add_f32(
    __global const float* moe_out,
    __global const float* shared_gated,
    uint hidden_size,
    __global float* ffn_out) {
  const uint index = (uint)get_global_id(0);
  if (index >= hidden_size) {
    return;
  }
  ffn_out[index] = moe_out[index] + shared_gated[index];
}

__kernel void post_ffn_residual_add_f32(
    __global const float* attn_residual,
    __global const float* ffn_out,
    uint hidden_size,
    __global float* layer_output) {
  const uint index = (uint)get_global_id(0);
  if (index >= hidden_size) {
    return;
  }
  layer_output[index] = attn_residual[index] + ffn_out[index];
}

inline void atomic_add_f32_bits(volatile __global uint* ptr, float value) {
  uint old_bits;
  uint new_bits;
  do {
    old_bits = *ptr;
    new_bits = as_uint(as_float(old_bits) + value);
  } while (atomic_cmpxchg(ptr, old_bits, new_bits) != old_bits);
}

__kernel void ffn_tail_init_residual_bits_f32(
    __global const float* attn_residual,
    uint hidden_size,
    __global uint* layer_output_bits) {
  const uint row = (uint)get_global_id(0);
  if (row >= hidden_size) {
    return;
  }
  layer_output_bits[row] = as_uint(attn_residual[row]);
}

__kernel void ffn_tail_reduce_down_atomic_f32(
    __global const float* selected_down,
    __global const float* weights,
    __global const float* shared_down,
    __global const float* gate,
    uint hidden_size,
    uint expert_count,
    __global uint* layer_output_bits) {
  const uint gid = (uint)get_global_id(0);
  const uint selected_total = hidden_size * expert_count;
  if (gid < selected_total) {
    const uint expert = gid / hidden_size;
    const uint row = gid - expert * hidden_size;
    atomic_add_f32_bits(
        (volatile __global uint*)(layer_output_bits + row),
        selected_down[gid] * weights[expert]);
    return;
  }
  const uint shared_gid = gid - selected_total;
  if (shared_gid >= hidden_size) {
    return;
  }
  const float shared = shared_down[shared_gid] * sigmoid_f32(gate[0]);
  atomic_add_f32_bits(
      (volatile __global uint*)(layer_output_bits + shared_gid), shared);
}

__kernel void q6k_selected_down_matvec_rowstripe_expert8_plus_shared_tail_atomic_raw(
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
    __global uint* layer_output_bits) {
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
    atomic_add_f32_bits(
        (volatile __global uint*)(layer_output_bits + local_row),
        sum * weights[selected]);
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
  const float shared = sum * sigmoid_f32(gate[0]);
  atomic_add_f32_bits(
      (volatile __global uint*)(layer_output_bits + row), shared);
}

__kernel void ffn_tail_fused_output_f32(
    __global const float* selected_down,
    __global const float* weights,
    __global const float* shared_down,
    __global const float* gate,
    __global const float* attn_residual,
    uint hidden_size,
    uint expert_count,
    __global float* layer_output) {
  const uint row = (uint)get_global_id(0);
  if (row >= hidden_size) return;
  float acc = 0.0f;
  for (uint expert = 0; expert < expert_count; ++expert) {
    acc += selected_down[expert * hidden_size + row] * weights[expert];
  }
  const float sigmoid = sigmoid_f32(gate[0]);
  const float shared = shared_down[row] * sigmoid;
  const float ffn_out = acc + shared;
  layer_output[row] = attn_residual[row] + ffn_out;
}

__kernel void linear_attn_postconv_silu_split_f32(
    __global const float* conv_output_raw,
    uint q_values,
    uint total_values,
    __global float* conv_output_silu,
    __global float* q_conv,
    __global float* k_conv,
    __global float* v_conv_predelta) {
  const uint index = (uint)get_global_id(0);
  if (index >= total_values) {
    return;
  }
  const float value = conv_output_raw[index];
  const float silu = value * sigmoid_f32(value);
  conv_output_silu[index] = silu;
  if (index < q_values) {
    q_conv[index] = silu;
  } else if (index < 2U * q_values) {
    k_conv[index - q_values] = silu;
  } else {
    v_conv_predelta[index - 2U * q_values] = silu;
  }
}

__kernel void linear_attn_l2_norm_heads_f32(__global const float* input,
                                            uint head_dim,
                                            uint head_count,
                                            float norm_epsilon,
                                            __global float* output) {
  const uint head = (uint)get_global_id(0);
  if (head >= head_count) {
    return;
  }
  const uint base = head * head_dim;
  float sum = 0.0f;
  for (uint i = 0; i < head_dim; ++i) {
    const float value = input[base + i];
    sum += value * value;
  }
  const float scale = 1.0f / fmax(sqrt(sum), norm_epsilon);
  for (uint i = 0; i < head_dim; ++i) {
    output[base + i] = input[base + i] * scale;
  }
}

__kernel void linear_attn_l2_norm_qk_heads_f32(
    __global const float* q_input,
    __global const float* k_input,
    uint head_dim,
    uint head_count,
    float norm_epsilon,
    __global float* q_output,
    __global float* k_output) {
  const uint index = (uint)get_global_id(0);
  if (index >= 2U * head_count) {
    return;
  }
  const uint head = index % head_count;
  const uint base = head * head_dim;
  __global const float* input = index < head_count ? q_input : k_input;
  __global float* output = index < head_count ? q_output : k_output;
  float sum = 0.0f;
  for (uint i = 0; i < head_dim; ++i) {
    const float value = input[base + i];
    sum += value * value;
  }
  const float scale = 1.0f / fmax(sqrt(sum), norm_epsilon);
  for (uint i = 0; i < head_dim; ++i) {
    output[base + i] = input[base + i] * scale;
  }
}

__kernel void linear_attn_postconv_fused_qk_l2_f32(
    __global const float* conv_output_raw,
    uint head_dim,
    uint query_heads,
    uint value_heads,
    float norm_epsilon,
    __global float* conv_output_silu,
    __global float* q_conv,
    __global float* k_conv,
    __global float* v_conv_predelta,
    __global float* q_conv_predelta,
    __global float* k_conv_predelta) {
  const uint index = (uint)get_global_id(0);
  const uint q_values = head_dim * query_heads;
  const uint qk_heads = query_heads * 2U;
  const uint v_values = head_dim * value_heads;
  if (index < qk_heads) {
    const int is_k = index >= query_heads;
    const uint head = is_k ? index - query_heads : index;
    const uint base = head * head_dim;
    const uint raw_base = (is_k ? q_values : 0U) + base;
    float sum = 0.0f;
    for (uint i = 0; i < head_dim; ++i) {
      const float value = conv_output_raw[raw_base + i];
      const float silu = value * sigmoid_f32(value);
      sum += silu * silu;
    }
    const float scale = 1.0f / fmax(sqrt(sum), norm_epsilon);
    for (uint i = 0; i < head_dim; ++i) {
      const float value = conv_output_raw[raw_base + i];
      const float silu = value * sigmoid_f32(value);
      conv_output_silu[raw_base + i] = silu;
      if (is_k) {
        k_conv[base + i] = silu;
        k_conv_predelta[base + i] = silu * scale;
      } else {
        q_conv[base + i] = silu;
        q_conv_predelta[base + i] = silu * scale;
      }
    }
    return;
  }

  const uint v_index = index - qk_heads;
  if (v_index >= v_values) {
    return;
  }
  const uint raw_index = 2U * q_values + v_index;
  const float value = conv_output_raw[raw_index];
  const float silu = value * sigmoid_f32(value);
  conv_output_silu[raw_index] = silu;
  v_conv_predelta[v_index] = silu;
}

__attribute__((reqd_work_group_size(128, 1, 1)))
__kernel void linear_attn_postconv_fused_qk_l2_parallel_f32(
    __global const float* conv_output_raw,
    uint head_dim,
    uint query_heads,
    uint value_heads,
    float norm_epsilon,
    __global float* conv_output_silu,
    __global float* q_conv,
    __global float* k_conv,
    __global float* v_conv_predelta,
    __global float* q_conv_predelta,
    __global float* k_conv_predelta) {
  const uint head = (uint)get_group_id(0);
  const uint lane = (uint)get_local_id(0);
  const uint q_values = head_dim * query_heads;
  __local float local_silu[128];
  __local float local_scale[1];
  if (head < 2U * query_heads) {
    const int is_k = head >= query_heads;
    const uint qk_head = is_k ? head - query_heads : head;
    const uint base = qk_head * head_dim;
    const uint raw_index = (is_k ? q_values : 0U) + base + lane;
    const float value = conv_output_raw[raw_index];
    const float silu = value * sigmoid_f32(value);
    local_silu[lane] = silu;
    conv_output_silu[raw_index] = silu;
    if (is_k) {
      k_conv[base + lane] = silu;
    } else {
      q_conv[base + lane] = silu;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lane == 0U) {
      float sum = 0.0f;
      for (uint index = 0U; index < 128U; ++index) {
        sum += local_silu[index] * local_silu[index];
      }
      local_scale[0] = 1.0f / fmax(sqrt(sum), norm_epsilon);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (is_k) {
      k_conv_predelta[base + lane] = silu * local_scale[0];
    } else {
      q_conv_predelta[base + lane] = silu * local_scale[0];
    }
    return;
  }
  const uint value_head = head - 2U * query_heads;
  if (value_head >= value_heads) return;
  const uint v_index = value_head * head_dim + lane;
  const uint raw_index = 2U * q_values + v_index;
  const float value = conv_output_raw[raw_index];
  const float silu = value * sigmoid_f32(value);
  conv_output_silu[raw_index] = silu;
  v_conv_predelta[v_index] = silu;
}

__kernel void linear_attn_delta_recurrent_f32(__global const float* q,
                                              __global const float* k,
                                              __global const float* v,
                                              __global const float* gate,
                                              __global const float* beta,
                                              __global const float* state_in,
                                              uint head_dim,
                                              uint query_heads,
                                              uint value_heads,
                                              __global float* attention_output,
                                              __global float* state_out) {
  const uint index = (uint)get_global_id(0);
  const uint total_rows = head_dim * value_heads;
  if (index >= total_rows) {
    return;
  }
  const uint value_head = index / head_dim;
  const uint row = index - value_head * head_dim;
  const uint query_head = value_head % query_heads;
  const uint q_base = query_head * head_dim;
  const uint v_base = value_head * head_dim;
  const uint state_base = value_head * head_dim * head_dim + row * head_dim;
  const float decay = exp(gate[value_head]);

  float sum_k = 0.0f;
  for (uint col = 0; col < head_dim; ++col) {
    const float decayed = state_in[state_base + col] * decay;
    sum_k += decayed * k[q_base + col];
  }
  const float delta = (v[v_base + row] - sum_k) * beta[value_head];

  float sum_q = 0.0f;
  for (uint col = 0; col < head_dim; ++col) {
    const float updated = state_in[state_base + col] * decay +
                          k[q_base + col] * delta;
    state_out[state_base + col] = updated;
    sum_q += updated * q[q_base + col];
  }
  attention_output[v_base + row] = sum_q * rsqrt((float)head_dim);
}

__kernel void linear_attn_delta_recurrent_qk_local_f32(
    __global const float* q,
    __global const float* k,
    __global const float* v,
    __global const float* gate,
    __global const float* beta,
    __global const float* state_in,
    uint head_dim,
    uint query_heads,
    uint value_heads,
    __global float* attention_output,
    __global float* state_out) {
  const uint index = (uint)get_global_id(0);
  const uint total_rows = head_dim * value_heads;
  if (index >= total_rows) {
    return;
  }
  const uint value_head = index / head_dim;
  const uint row = index - value_head * head_dim;
  const uint query_head = value_head % query_heads;
  const uint q_base = query_head * head_dim;
  const uint v_base = value_head * head_dim;
  const uint state_base = value_head * head_dim * head_dim + row * head_dim;
  const uint lid = (uint)get_local_id(0);
  __local float local_q[128];
  __local float local_k[128];
  local_q[lid] = q[q_base + lid];
  local_k[lid] = k[q_base + lid];
  barrier(CLK_LOCAL_MEM_FENCE);

  const float decay = exp(gate[value_head]);
  float sum_k = 0.0f;
  for (uint col = 0; col < 128U; ++col) {
    const float decayed = state_in[state_base + col] * decay;
    sum_k += decayed * local_k[col];
  }
  const float delta = (v[v_base + row] - sum_k) * beta[value_head];

  float sum_q = 0.0f;
  for (uint col = 0; col < 128U; ++col) {
    const float updated = state_in[state_base + col] * decay +
                          local_k[col] * delta;
    state_out[state_base + col] = updated;
    sum_q += updated * local_q[col];
  }
  attention_output[v_base + row] = sum_q * rsqrt(128.0f);
}

__kernel void linear_attn_delta_recurrent_final_qk_local_f32(
    __global const float* q,
    __global const float* k,
    __global const float* v,
    __global const float* gate,
    __global const float* beta,
    __global const float* state_in,
    __global const float* z,
    __global const float* norm_weight,
    uint head_dim,
    uint query_heads,
    uint value_heads,
    float norm_epsilon,
    __global float* attention_output,
    __global float* state_out,
    __global float* final_output) {
  const uint index = (uint)get_global_id(0);
  const uint total_rows = head_dim * value_heads;
  if (index >= total_rows) {
    return;
  }
  const uint value_head = index / head_dim;
  const uint row = index - value_head * head_dim;
  const uint query_head = value_head % query_heads;
  const uint q_base = query_head * head_dim;
  const uint v_base = value_head * head_dim;
  const uint state_base = value_head * head_dim * head_dim + row * head_dim;
  const uint lid = (uint)get_local_id(0);
  __local float local_q[128];
  __local float local_k[128];
  __local float local_attention[128];
  __local float local_norm_scale[1];
  local_q[lid] = q[q_base + lid];
  local_k[lid] = k[q_base + lid];
  barrier(CLK_LOCAL_MEM_FENCE);

  const float decay = exp(gate[value_head]);
  float sum_k = 0.0f;
  for (uint col = 0; col < 128U; ++col) {
    const float decayed = state_in[state_base + col] * decay;
    sum_k += decayed * local_k[col];
  }
  const float delta = (v[v_base + row] - sum_k) * beta[value_head];

  float sum_q = 0.0f;
  for (uint col = 0; col < 128U; ++col) {
    const float updated = state_in[state_base + col] * decay +
                          local_k[col] * delta;
    state_out[state_base + col] = updated;
    sum_q += updated * local_q[col];
  }
  const float attention = sum_q * rsqrt(128.0f);
  local_attention[row] = attention;
  attention_output[v_base + row] = attention;
  barrier(CLK_LOCAL_MEM_FENCE);

  if (lid == 0U) {
    float sum_squares = 0.0f;
    for (uint i = 0; i < 128U; ++i) {
      const float value = local_attention[i];
      sum_squares += value * value;
    }
    local_norm_scale[0] = rsqrt(sum_squares / 128.0f + norm_epsilon);
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  const float z_value = z[v_base + row];
  final_output[v_base + row] =
      attention * local_norm_scale[0] * norm_weight[row] *
      (z_value * sigmoid_f32(z_value));
}

__attribute__((reqd_work_group_size(128, 1, 1)))
__kernel void linear_attn_delta_recurrent_final_cpu_shape_qk_local_f32(
    __global const float* q,
    __global const float* k,
    __global const float* v,
    __global const float* gate,
    __global const float* beta,
    __global const float* state_in,
    __global const float* z,
    __global const float* norm_weight,
    uint head_dim,
    uint query_heads,
    uint value_heads,
    float norm_epsilon,
    __global float* attention_output,
    __global float* state_out,
    __global float* final_output) {
  const uint index = (uint)get_global_id(0);
  const uint total_rows = head_dim * value_heads;
  if (index >= total_rows) {
    return;
  }
  const uint value_head = index / head_dim;
  const uint row = index - value_head * head_dim;
  const uint query_head = value_head % query_heads;
  const uint q_base = query_head * head_dim;
  const uint v_base = value_head * head_dim;
  const uint state_base = value_head * head_dim * head_dim + row * head_dim;
  const uint lid = (uint)get_local_id(0);
  __local float local_q[128];
  __local float local_k[128];
  __local float local_attention[128];
  __local float local_norm_scale[1];
  local_q[lid] = q[q_base + lid];
  local_k[lid] = k[q_base + lid];
  barrier(CLK_LOCAL_MEM_FENCE);

  const float decay = exp(gate[value_head]);
  float sum_k = 0.0f;
  for (uint col = 0; col < 128U; ++col) {
    const float decayed = state_in[state_base + col] * decay;
    sum_k += decayed * local_k[col];
  }
  const float delta = (v[v_base + row] - sum_k) * beta[value_head];

  float sum_q = 0.0f;
  for (uint col = 0; col < 128U; ++col) {
    const float updated = state_in[state_base + col] * decay +
                          local_k[col] * delta;
    state_out[state_base + col] = updated;
    sum_q += updated * local_q[col];
  }
  const float attention = sum_q * rsqrt(128.0f);
  local_attention[row] = attention;
  attention_output[v_base + row] = attention;
  barrier(CLK_LOCAL_MEM_FENCE);

  if (lid == 0U) {
    float sum_squares = 0.0f;
    for (uint i = 0; i < 128U; ++i) {
      const float value = local_attention[i];
      sum_squares += value * value;
    }
    const float mean_square = sum_squares / 128.0f;
    local_norm_scale[0] =
        (float)(1.0 / sqrt((double)mean_square + (double)norm_epsilon));
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  const float z_value = z[v_base + row];
  const double z_double = (double)z_value;
  const double sigmoid_double =
      z_double >= 0.0 ? 1.0 / (1.0 + exp(-z_double))
                      : exp(z_double) / (1.0 + exp(z_double));
  const float z_silu = (float)(z_double * sigmoid_double);
  final_output[v_base + row] =
      attention * local_norm_scale[0] * norm_weight[row] * z_silu;
}

__kernel void linear_attn_final_norm_f32(__global const float* attention_output,
                                         __global const float* z,
                                         __global const float* norm_weight,
                                         uint head_dim,
                                         uint value_heads,
                                         float norm_epsilon,
                                         __global float* final_output) {
  const uint value_head = (uint)get_global_id(0);
  if (value_head >= value_heads) {
    return;
  }
  const uint base = value_head * head_dim;
  float sum_squares = 0.0f;
  for (uint i = 0; i < head_dim; ++i) {
    const float value = attention_output[base + i];
    sum_squares += value * value;
  }
  const float mean_square = sum_squares / (float)head_dim;
  const float norm_scale = rsqrt(mean_square + norm_epsilon);
  for (uint i = 0; i < head_dim; ++i) {
    const float z_value = z[base + i];
    final_output[base + i] =
        attention_output[base + i] * norm_scale * norm_weight[i] *
        (z_value * sigmoid_f32(z_value));
  }
}

__kernel void linear_attn_final_norm_cpu_shape_f32(
    __global const float* attention_output,
    __global const float* z,
    __global const float* norm_weight,
    uint head_dim,
    uint value_heads,
    float norm_epsilon,
    __global float* final_output) {
  const uint value_head = (uint)get_global_id(0);
  if (value_head >= value_heads) {
    return;
  }
  const uint base = value_head * head_dim;
  float sum_squares = 0.0f;
  for (uint i = 0; i < head_dim; ++i) {
    const float value = attention_output[base + i];
    sum_squares += value * value;
  }
  const float mean_square = sum_squares / (float)head_dim;
  const float norm_scale =
      (float)(1.0 / sqrt((double)mean_square + (double)norm_epsilon));
  for (uint i = 0; i < head_dim; ++i) {
    const float z_value = z[base + i];
    const double z_double = (double)z_value;
    const double sigmoid_double =
        z_double >= 0.0 ? 1.0 / (1.0 + exp(-z_double))
                        : exp(z_double) / (1.0 + exp(z_double));
    const float z_silu = (float)(z_double * sigmoid_double);
    final_output[base + i] =
        attention_output[base + i] * norm_scale * norm_weight[i] *
        z_silu;
  }
}

#pragma OPENCL FP_CONTRACT OFF

__kernel void q6k_linear_qkv_cpuorder_nofma(
    __global const uchar* raw,
    __global const char* q8_qs,
    __global const float* q8_d,
    uint rows_per_expert,
    uint blocks_per_row,
    __global float* out) {
  const uint row_index = (uint)get_global_id(0);
  const uint selected = row_index / rows_per_expert;
  const uint local_row = row_index - selected * rows_per_expert;
  __global const uchar* row =
      raw + ((ulong)selected * (ulong)rows_per_expert + (ulong)local_row) *
                (ulong)blocks_per_row * 210UL;
  __global const char* expert_q8 =
      q8_qs + (ulong)selected * (ulong)blocks_per_row * 256UL;
  __global const float* expert_q8_d =
      q8_d + (ulong)selected * (ulong)blocks_per_row;
  float sums[8];
  for (int lane = 0; lane < 8; ++lane) {
    sums[lane] = 0.0f;
  }
  for (uint block_index = 0; block_index < blocks_per_row; ++block_index) {
    __global const uchar* block = row + (ulong)block_index * 210UL;
    __global const char* q8 = expert_q8 + (ulong)block_index * 256UL;
    __global const char* scales = (__global const char*)(block + 192);
    const float combined_scale =
        half_to_float(load_le16(block + 208)) * expert_q8_d[block_index];
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
              (((int)(ql[lane] & (uchar)15) |
                (((high >> 0) & 3) << 4)) - 32);
          lane_sums[lane_index] +=
              scale1 * (int)q8[base + 32 + lane] *
              (((int)(ql[32 + lane] & (uchar)15) |
                (((high >> 2) & 3) << 4)) - 32);
          lane_sums[lane_index] +=
              scale2 * (int)q8[base + 64 + lane] *
              (((int)(ql[lane] >> 4) |
                (((high >> 4) & 3) << 4)) - 32);
          lane_sums[lane_index] +=
              scale3 * (int)q8[base + 96 + lane] *
              (((int)(ql[32 + lane] >> 4) |
                (((high >> 6) & 3) << 4)) - 32);
        }
      }
    }
    for (int lane = 0; lane < 8; ++lane) {
      const float block_lane = combined_scale * (float)lane_sums[lane];
      sums[lane] = sums[lane] + block_lane;
    }
  }
  float sum = 0.0f;
  for (int lane = 0; lane < 8; ++lane) {
    sum = sum + sums[lane];
  }
  out[row_index] = sum;
}

__kernel void linear_attn_conv_cpuorder_nofma_f32(
    __global const float* qkv_mixed,
    __global const float* conv_state,
    __global const float* weights,
    uint channel_count,
    uint kernel_size,
    __global float* conv_output_raw,
    __global float* next_conv_state) {
  const uint channel = (uint)get_global_id(0);
  if (channel >= channel_count) {
    return;
  }
  const uint history = kernel_size - 1U;
  const uint state_base = channel * history;
  const uint weight_base = channel * kernel_size;
  float sum = 0.0f;
  for (uint k = 0; k < history; ++k) {
    const float product =
        conv_state[state_base + k] * weights[weight_base + k];
    sum = sum + product;
  }
  const float newest =
      qkv_mixed[channel] * weights[weight_base + history];
  sum = sum + newest;
  conv_output_raw[channel] = sum;
  for (uint k = 0; k + 1U < history; ++k) {
    next_conv_state[state_base + k] = conv_state[state_base + k + 1U];
  }
  if (history > 0U) {
    next_conv_state[state_base + history - 1U] = qkv_mixed[channel];
  }
}

__kernel void linear_attn_postconv_silu_split_cpuorder_f32(
    __global const float* conv_output_raw,
    uint q_values,
    uint total_values,
    __global float* conv_output_silu,
    __global float* q_conv,
    __global float* k_conv,
    __global float* v_conv_predelta) {
  const uint index = (uint)get_global_id(0);
  if (index >= total_values) {
    return;
  }
  const float value = conv_output_raw[index];
  const double x = (double)value;
  const double sigmoid_double =
      x >= 0.0 ? 1.0 / (1.0 + exp(-x))
               : exp(x) / (1.0 + exp(x));
  const float sigmoid = (float)sigmoid_double;
  const float silu = value * sigmoid;
  conv_output_silu[index] = silu;
  if (index < q_values) {
    q_conv[index] = silu;
  } else if (index < 2U * q_values) {
    k_conv[index - q_values] = silu;
  } else {
    v_conv_predelta[index - 2U * q_values] = silu;
  }
}

__kernel void linear_attn_postconv_qk_l2_cpuorder_f32(
    __global const float* q_input,
    __global const float* k_input,
    uint head_dim,
    uint head_count,
    float norm_epsilon,
    __global float* q_output,
    __global float* k_output) {
  const uint index = (uint)get_global_id(0);
  if (index >= 2U * head_count) {
    return;
  }
  const uint head = index % head_count;
  const uint base = head * head_dim;
  __global const float* input = index < head_count ? q_input : k_input;
  __global float* output = index < head_count ? q_output : k_output;
  double sum = 0.0;
  for (uint i = 0; i < head_dim; ++i) {
    const float value = input[base + i];
    sum = sum + (double)value * (double)value;
  }
  const float sum_f32 = (float)sum;
  const float scale = 1.0f / fmax(sqrt(sum_f32), norm_epsilon);
  for (uint i = 0; i < head_dim; ++i) {
    output[base + i] = input[base + i] * scale;
  }
}

__kernel void linear_attn_delta_recurrent_final_cpuorder_nofma_f32(
    __global const float* q,
    __global const float* k,
    __global const float* v,
    __global const float* decay,
    __global const float* beta,
    __global const float* state_in,
    __global const float* z_silu,
    __global const float* norm_weight,
    uint head_dim,
    uint query_heads,
    uint value_heads,
    float norm_epsilon,
    __global float* attention_output,
    __global float* state_out,
    __global float* final_output) {
  const uint index = (uint)get_global_id(0);
  const uint total_rows = head_dim * value_heads;
  if (index >= total_rows) {
    return;
  }
  const uint value_head = index / head_dim;
  const uint row = index - value_head * head_dim;
  const uint query_head = value_head % query_heads;
  const uint q_base = query_head * head_dim;
  const uint v_base = value_head * head_dim;
  const uint state_base = value_head * head_dim * head_dim + row * head_dim;
  const uint lid = (uint)get_local_id(0);
  __local float local_q[128];
  __local float local_k[128];
  __local float local_attention[128];
  __local float local_norm_scale[1];
  local_q[lid] = q[q_base + lid];
  local_k[lid] = k[q_base + lid];
  barrier(CLK_LOCAL_MEM_FENCE);

  float sum_k = 0.0f;
  for (uint col = 0; col < 128U; ++col) {
    const float decayed = state_in[state_base + col] * decay[value_head];
    state_out[state_base + col] = decayed;
    const float product = decayed * local_k[col];
    sum_k = sum_k + product;
  }
  const float difference = v[v_base + row] - sum_k;
  const float delta = difference * beta[value_head];

  float sum_q = 0.0f;
  for (uint col = 0; col < 128U; ++col) {
    const float update = local_k[col] * delta;
    const float updated = state_out[state_base + col] + update;
    state_out[state_base + col] = updated;
    const float product = updated * local_q[col];
    sum_q = sum_q + product;
  }
  const float attention_scale = 1.0f / sqrt((float)head_dim);
  const float attention = sum_q * attention_scale;
  local_attention[row] = attention;
  attention_output[v_base + row] = attention;
  barrier(CLK_LOCAL_MEM_FENCE);

  if (lid == 0U) {
    float sum_squares = 0.0f;
    for (uint i = 0; i < 128U; ++i) {
      const float value = local_attention[i];
      const float square = value * value;
      sum_squares = sum_squares + square;
    }
    const float mean_square = sum_squares / (float)head_dim;
    local_norm_scale[0] = 1.0f / sqrt(mean_square + norm_epsilon);
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  const float normalized = attention * local_norm_scale[0];
  const float weighted = normalized * norm_weight[row];
  final_output[v_base + row] = weighted * z_silu[v_base + row];
}

#pragma OPENCL FP_CONTRACT ON

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
      lanes[lane] = fma(q_rope[q_base + i + lane],
                        k_history[hist_base + i + lane],
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
      out_acc *= max_scale;
    } else {
      value_scale = exp(score - max_score);
    }
    out_acc = fma(v_history[token * kv_size + kv_base + dim],
                  value_scale,
                  out_acc);
    sum = sum * max_scale + value_scale;
  }
  attn_pregate[index] = sum == 0.0f ? 0.0f : out_acc / sum;
}

__kernel void full_attn_core_control_f32(
    __global const float* q_rope,
    __global const float* k_history,
    __global const float* v_history,
    __global const ulong* token_control,
    uint capacity_tokens,
    uint head_dim,
    uint q_head_count,
    uint kv_head_count,
    float attention_scale,
    __global float* attn_pregate) {
  const uint index = (uint)get_global_id(0);
  const uint q_size = q_head_count * head_dim;
  if (index >= q_size) return;
  const uint token_count = (uint)min(
      token_control[1] + 1UL, (ulong)capacity_tokens);
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
    const float dot = iq36_dot_fp16_avx_like(
        q_rope, k_history, q_base, hist_base, head_dim);
    const float score = dot * attention_scale;
    const float old_max = max_score;
    float max_scale = 1.0f;
    float value_scale = 1.0f;
    if (score > max_score) {
      max_score = score;
      max_scale = exp(old_max - max_score);
      out_acc *= max_scale;
    } else {
      value_scale = exp(score - max_score);
    }
    out_acc = fma(v_history[token * kv_size + kv_base + dim],
                  value_scale, out_acc);
    sum = sum * max_scale + value_scale;
  }
  attn_pregate[index] = sum == 0.0f ? 0.0f : out_acc / sum;
}

__kernel void full_attn_score_f32(__global const float* q_rope,
                                  __global const float* k_history,
                                  uint token_count,
                                  uint head_dim,
                                  uint q_head_count,
                                  uint kv_head_count,
                                  float attention_scale,
                                  __global float* attn_scores) {
  const uint index = (uint)get_global_id(0);
  const uint total = q_head_count * token_count;
  if (index >= total) {
    return;
  }
  const uint q_head = index / token_count;
  const uint token = index - q_head * token_count;
  const uint gqa_group = q_head_count / kv_head_count;
  const uint kv_head = q_head / gqa_group;
  const uint q_base = q_head * head_dim;
  const uint kv_base = kv_head * head_dim;
  const uint kv_size = kv_head_count * head_dim;
  const uint hist_base = token * kv_size + kv_base;
  const float dot =
      iq36_dot_fp16_avx_like(q_rope, k_history, q_base, hist_base, head_dim);
  attn_scores[index] = dot * attention_scale;
}

__kernel void full_attn_score_control_f32(
    __global const float* q_rope,
    __global const float* k_history,
    __global const ulong* token_control,
    uint capacity_tokens,
    uint head_dim,
    uint q_head_count,
    uint kv_head_count,
    float attention_scale,
    __global float* attn_scores) {
  const uint index = (uint)get_global_id(0);
  const uint total = q_head_count * capacity_tokens;
  if (index >= total) return;
  const uint q_head = index / capacity_tokens;
  const uint token = index - q_head * capacity_tokens;
  const uint token_count = (uint)min(
      token_control[1] + 1UL, (ulong)capacity_tokens);
  if (token >= token_count) return;
  const uint gqa_group = q_head_count / kv_head_count;
  const uint kv_head = q_head / gqa_group;
  const uint q_base = q_head * head_dim;
  const uint kv_base = kv_head * head_dim;
  const uint kv_size = kv_head_count * head_dim;
  const uint hist_base = token * kv_size + kv_base;
  const float dot = iq36_dot_fp16_avx_like(
      q_rope, k_history, q_base, hist_base, head_dim);
  attn_scores[q_head * capacity_tokens + token] = dot * attention_scale;
}

__kernel void full_attn_apply_score_gate_f32(
    __global const float* attn_scores,
    __global const float* v_history,
    __global const float* q_full,
    uint token_count,
    uint head_dim,
    uint q_head_count,
    uint kv_head_count,
    __global float* attn_pregate,
    __global float* attn_gated) {
  const uint index = (uint)get_global_id(0);
  const uint q_size = q_head_count * head_dim;
  if (index >= q_size) {
    return;
  }
  const uint q_head = index / head_dim;
  const uint dim = index - q_head * head_dim;
  const uint gqa_group = q_head_count / kv_head_count;
  const uint kv_head = q_head / gqa_group;
  const uint kv_base = kv_head * head_dim;
  const uint kv_size = kv_head_count * head_dim;

  float sum = 0.0f;
  float max_score = -INFINITY;
  float out_acc = 0.0f;
  for (uint token = 0; token < token_count; ++token) {
    const float score = attn_scores[q_head * token_count + token];
    const float old_max = max_score;
    float max_scale = 1.0f;
    float value_scale = 1.0f;
    if (score > max_score) {
      max_score = score;
      max_scale = exp(old_max - max_score);
      out_acc *= max_scale;
    } else {
      value_scale = exp(score - max_score);
    }
    out_acc = fma(v_history[token * kv_size + kv_base + dim],
                  value_scale,
                  out_acc);
    sum = sum * max_scale + value_scale;
  }
  const float pregate = sum == 0.0f ? 0.0f : out_acc / sum;
  const float gate = q_full[q_head * head_dim * 2U + head_dim + dim];
  const float sigmoid = 1.0f / (1.0f + exp(-gate));
  attn_pregate[index] = pregate;
  attn_gated[index] = pregate * sigmoid;
}

__kernel void full_attn_apply_score_gate_control_f32(
    __global const float* attn_scores,
    __global const float* v_history,
    __global const float* q_full,
    __global const ulong* token_control,
    uint capacity_tokens,
    uint head_dim,
    uint q_head_count,
    uint kv_head_count,
    __global float* attn_pregate,
    __global float* attn_gated) {
  const uint index = (uint)get_global_id(0);
  const uint q_size = q_head_count * head_dim;
  if (index >= q_size) return;
  const uint token_count = (uint)min(
      token_control[1] + 1UL, (ulong)capacity_tokens);
  const uint q_head = index / head_dim;
  const uint dim = index - q_head * head_dim;
  const uint gqa_group = q_head_count / kv_head_count;
  const uint kv_head = q_head / gqa_group;
  const uint kv_base = kv_head * head_dim;
  const uint kv_size = kv_head_count * head_dim;
  float sum = 0.0f;
  float max_score = -INFINITY;
  float out_acc = 0.0f;
  for (uint token = 0; token < token_count; ++token) {
    const float score =
        attn_scores[q_head * capacity_tokens + token];
    const float old_max = max_score;
    float max_scale = 1.0f;
    float value_scale = 1.0f;
    if (score > max_score) {
      max_score = score;
      max_scale = exp(old_max - max_score);
      out_acc *= max_scale;
    } else {
      value_scale = exp(score - max_score);
    }
    out_acc = fma(v_history[token * kv_size + kv_base + dim],
                  value_scale, out_acc);
    sum = sum * max_scale + value_scale;
  }
  const float pregate = sum == 0.0f ? 0.0f : out_acc / sum;
  const float gate = q_full[q_head * head_dim * 2U + head_dim + dim];
  const float sigmoid = 1.0f / (1.0f + exp(-gate));
  attn_pregate[index] = pregate;
  attn_gated[index] = pregate * sigmoid;
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

__kernel void full_attn_kv_append_f32(
    __global const float* k_current,
    __global const float* v_current,
    __global const ulong* token_control,
    uint capacity_tokens,
    __global float* k_history,
    __global float* v_history) {
  const uint index = (uint)get_global_id(0);
  if (index >= 512U) return;
  const ulong position = token_control[1];
  if (position >= (ulong)capacity_tokens) return;
  const ulong offset = position * 512UL + (ulong)index;
  k_history[offset] = k_current[index];
  v_history[offset] = v_current[index];
}

#define IQ36_INT8_KV_HEAD_DIM 256U
#define IQ36_INT8_KV_Q_HEADS 16U
#define IQ36_INT8_KV_HEADS 2U
#define IQ36_INT8_KV_GQA_GROUP 8U
#define IQ36_INT8_KV_QUANT_GROUP 32U
#define IQ36_INT8_KV_SCALE_GROUPS 8U
#define IQ36_INT8_KV_CHUNK_TOKENS 256U
#define IQ36_INT8_KV_HOT_TOKENS 8192U
#define IQ36_INT8_KV_HOT_TILE_TOKENS 4U

__attribute__((reqd_work_group_size(32, 1, 1)))
__attribute__((intel_reqd_sub_group_size(32)))
__kernel void full_attn_kv_append_i8_block32_control(
    __global const float* current_k,
    __global const float* current_v,
    __global const ulong* token_control,
    uint capacity_tokens,
    __global char* k_history,
    __global char* v_history,
    __global half* k_scales,
    __global half* v_scales,
    __global float* hot_k_history,
    __global float* hot_v_history) {
  const uint lane = (uint)get_sub_group_local_id();
  const uint group = (uint)get_group_id(0);
  const uint tensor =
      group / (IQ36_INT8_KV_HEADS * IQ36_INT8_KV_SCALE_GROUPS);
  const uint within_tensor =
      group - tensor * IQ36_INT8_KV_HEADS * IQ36_INT8_KV_SCALE_GROUPS;
  const uint kv_head = within_tensor / IQ36_INT8_KV_SCALE_GROUPS;
  const uint scale_group =
      within_tensor - kv_head * IQ36_INT8_KV_SCALE_GROUPS;
  const uint dim = scale_group * IQ36_INT8_KV_QUANT_GROUP + lane;
  const ulong position = token_control[1];
  if (position >= (ulong)capacity_tokens) return;
  __global const float* input = tensor == 0U ? current_k : current_v;
  __global char* output = tensor == 0U ? k_history : v_history;
  __global half* scales = tensor == 0U ? k_scales : v_scales;
  const float value = input[kv_head * IQ36_INT8_KV_HEAD_DIM + dim];
  const float maximum = sub_group_reduce_max(fabs(value));
  const float scale = maximum == 0.0f ? 1.0f : maximum * (1.0f / 127.0f);
  const int quantized = clamp(convert_int_rte(value / scale), -127, 127);
  const ulong token_head = position * IQ36_INT8_KV_HEADS + kv_head;
  output[token_head * IQ36_INT8_KV_HEAD_DIM + dim] = (char)quantized;
  const ulong hot_token_head =
      (position % IQ36_INT8_KV_HOT_TOKENS) * IQ36_INT8_KV_HEADS + kv_head;
  if (tensor == 0U) {
    hot_k_history[hot_token_head * IQ36_INT8_KV_HEAD_DIM + dim] = value;
  } else {
    hot_v_history[hot_token_head * IQ36_INT8_KV_HEAD_DIM + dim] = value;
  }
  if (lane == 0U) {
    scales[token_head * IQ36_INT8_KV_SCALE_GROUPS + scale_group] =
        convert_half_rte(scale);
  }
}

__attribute__((reqd_work_group_size(256, 1, 1)))
__attribute__((intel_reqd_sub_group_size(32)))
__kernel void full_attn_i8_block32_gqa_partial_control(
    __global const float* q,
    __global const char* k_history,
    __global const char* v_history,
    __global const half* k_scales,
    __global const half* v_scales,
    __global const float* hot_k_history,
    __global const float* hot_v_history,
    __global const ulong* token_control,
    uint capacity_tokens,
    float attention_scale,
    __global float* partial_max,
    __global float* partial_sum,
    __global float* partial_output) {
  const uint lid = (uint)get_local_id(0);
  const uint subgroup = (uint)get_sub_group_id();
  const uint lane = (uint)get_sub_group_local_id();
  const uint capacity_chunks =
      (capacity_tokens + IQ36_INT8_KV_CHUNK_TOKENS - 1U) /
      IQ36_INT8_KV_CHUNK_TOKENS;
  const uint token_count = (uint)min(
      token_control[1] + 1UL, (ulong)capacity_tokens);
  const uint active_chunks =
      (token_count + IQ36_INT8_KV_CHUNK_TOKENS - 1U) /
      IQ36_INT8_KV_CHUNK_TOKENS;
  const uint group = (uint)get_group_id(0);
  const uint kv_head = group / capacity_chunks;
  const uint chunk = group - kv_head * capacity_chunks;
  if (kv_head >= IQ36_INT8_KV_HEADS || chunk >= active_chunks) return;
  const uint q_head = kv_head * IQ36_INT8_KV_GQA_GROUP + subgroup;
  const uint begin = chunk * IQ36_INT8_KV_CHUNK_TOKENS;
  const uint end = min(begin + IQ36_INT8_KV_CHUNK_TOKENS, token_count);
  __local float local_k[
      IQ36_INT8_KV_HOT_TILE_TOKENS * IQ36_INT8_KV_HEAD_DIM];
  __local float local_v[
      IQ36_INT8_KV_HOT_TILE_TOKENS * IQ36_INT8_KV_HEAD_DIM];

  float running_max = -INFINITY;
  float running_sum = 0.0f;
  float output_acc[8] = {
      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

#define IQ36_ACCUMULATE_LOADED_KV(tile_base)                             \
  do {                                                                   \
    float dot = 0.0f;                                                    \
    for (uint item = 0; item < 8U; ++item) {                             \
      const uint dim = lane + item * 32U;                                \
      dot = fma(q[q_head * IQ36_INT8_KV_HEAD_DIM + dim],                 \
                local_k[(tile_base) + dim], dot);                        \
    }                                                                    \
    const float score = sub_group_reduce_add(dot) * attention_scale;     \
    const float next_max = fmax(running_max, score);                     \
    const float previous_scale = native_exp(running_max - next_max);     \
    const float value_scale = native_exp(score - next_max);              \
    running_sum = running_sum * previous_scale + value_scale;            \
    for (uint item = 0; item < 8U; ++item) {                             \
      const uint dim = lane + item * 32U;                                \
      output_acc[item] = fma(                                            \
          local_v[(tile_base) + dim], value_scale,                       \
          output_acc[item] * previous_scale);                            \
    }                                                                    \
    running_max = next_max;                                              \
  } while (0)

  const uint hot_begin =
      token_count - min(token_count, IQ36_INT8_KV_HOT_TOKENS);
  const uint compressed_end = min(end, hot_begin);
  for (uint token = begin; token < compressed_end; ++token) {
    const ulong token_head = (ulong)token * IQ36_INT8_KV_HEADS + kv_head;
    const ulong history_base = token_head * IQ36_INT8_KV_HEAD_DIM;
    const ulong scale_base = token_head * IQ36_INT8_KV_SCALE_GROUPS;
    float k_scale = lane == 0U
        ? convert_float(k_scales[scale_base + subgroup]) : 0.0f;
    float v_scale = lane == 0U
        ? convert_float(v_scales[scale_base + subgroup]) : 0.0f;
    k_scale = sub_group_broadcast(k_scale, 0U);
    v_scale = sub_group_broadcast(v_scale, 0U);
    local_k[lid] = convert_float(k_history[history_base + lid]) * k_scale;
    local_v[lid] = convert_float(v_history[history_base + lid]) * v_scale;
    barrier(CLK_LOCAL_MEM_FENCE);
    IQ36_ACCUMULATE_LOADED_KV(0U);
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  uint hot_token = max(begin, hot_begin);
  for (; hot_token + IQ36_INT8_KV_HOT_TILE_TOKENS <= end;
       hot_token += IQ36_INT8_KV_HOT_TILE_TOKENS) {
#pragma unroll
    for (uint tile = 0U; tile < IQ36_INT8_KV_HOT_TILE_TOKENS; ++tile) {
      const ulong hot_token_head =
          ((ulong)(hot_token + tile) % IQ36_INT8_KV_HOT_TOKENS) *
              IQ36_INT8_KV_HEADS + kv_head;
      const ulong hot_base = hot_token_head * IQ36_INT8_KV_HEAD_DIM;
      const uint tile_base = tile * IQ36_INT8_KV_HEAD_DIM;
      local_k[tile_base + lid] = hot_k_history[hot_base + lid];
      local_v[tile_base + lid] = hot_v_history[hot_base + lid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
#pragma unroll
    for (uint tile = 0U; tile < IQ36_INT8_KV_HOT_TILE_TOKENS; ++tile) {
      IQ36_ACCUMULATE_LOADED_KV(tile * IQ36_INT8_KV_HEAD_DIM);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  for (; hot_token < end; ++hot_token) {
    const ulong hot_token_head =
        ((ulong)hot_token % IQ36_INT8_KV_HOT_TOKENS) *
            IQ36_INT8_KV_HEADS + kv_head;
    const ulong hot_base = hot_token_head * IQ36_INT8_KV_HEAD_DIM;
    local_k[lid] = hot_k_history[hot_base + lid];
    local_v[lid] = hot_v_history[hot_base + lid];
    barrier(CLK_LOCAL_MEM_FENCE);
    IQ36_ACCUMULATE_LOADED_KV(0U);
    barrier(CLK_LOCAL_MEM_FENCE);
  }
#undef IQ36_ACCUMULATE_LOADED_KV

  const uint meta = group * IQ36_INT8_KV_GQA_GROUP + subgroup;
  if (lane == 0U) {
    partial_max[meta] = running_max;
    partial_sum[meta] = running_sum;
  }
  for (uint item = 0; item < 8U; ++item) {
    const uint dim = lane + item * 32U;
    partial_output[(ulong)meta * IQ36_INT8_KV_HEAD_DIM + dim] =
        output_acc[item];
  }
}

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void full_attn_i8_block32_gqa_reduce_gate_control(
    __global const float* partial_max,
    __global const float* partial_sum,
    __global const float* partial_output,
    __global const float* q_full,
    __global const ulong* token_control,
    uint capacity_tokens,
    __global float* attn_pregate,
    __global float* attn_gated) {
  const uint q_head = (uint)get_group_id(0);
  const uint dim = (uint)get_local_id(0);
  const uint kv_head = q_head / IQ36_INT8_KV_GQA_GROUP;
  const uint gqa_head = q_head - kv_head * IQ36_INT8_KV_GQA_GROUP;
  const uint capacity_chunks =
      (capacity_tokens + IQ36_INT8_KV_CHUNK_TOKENS - 1U) /
      IQ36_INT8_KV_CHUNK_TOKENS;
  const uint token_count = (uint)min(
      token_control[1] + 1UL, (ulong)capacity_tokens);
  const uint active_chunks =
      (token_count + IQ36_INT8_KV_CHUNK_TOKENS - 1U) /
      IQ36_INT8_KV_CHUNK_TOKENS;
  float running_max = -INFINITY;
  float running_sum = 0.0f;
  float output_acc = 0.0f;
  for (uint chunk = 0; chunk < active_chunks; ++chunk) {
    const uint group = kv_head * capacity_chunks + chunk;
    const uint meta = group * IQ36_INT8_KV_GQA_GROUP + gqa_head;
    const float next_max = fmax(running_max, partial_max[meta]);
    const float previous_scale = native_exp(running_max - next_max);
    const float partial_scale = native_exp(partial_max[meta] - next_max);
    running_sum = running_sum * previous_scale +
        partial_sum[meta] * partial_scale;
    output_acc = output_acc * previous_scale +
        partial_output[(ulong)meta * IQ36_INT8_KV_HEAD_DIM + dim] *
            partial_scale;
    running_max = next_max;
  }
  const uint index = q_head * IQ36_INT8_KV_HEAD_DIM + dim;
  const float pregate =
      running_sum == 0.0f ? 0.0f : output_acc / running_sum;
  const float gate =
      q_full[q_head * IQ36_INT8_KV_HEAD_DIM * 2U +
             IQ36_INT8_KV_HEAD_DIM + dim];
  attn_pregate[index] = pregate;
  attn_gated[index] = pregate / (1.0f + native_exp(-gate));
}

__kernel void vector_add_f32(
    __global const float* lhs,
    __global const float* rhs,
    uint value_count,
    __global float* output) {
  const uint index = (uint)get_global_id(0);
  if (index < value_count) output[index] = lhs[index] + rhs[index];
}

__kernel void vector_add_rms_scale_f64_parallel(
    __global const float* lhs,
    __global const float* rhs,
    uint hidden_size,
    float epsilon,
    __global float* output,
    __global float* scale_out) {
  const uint lid = (uint)get_local_id(0);
  const uint local_size = (uint)get_local_size(0);
  __local double partial[256];
  double sum_squares = 0.0;
  for (uint index = lid; index < hidden_size; index += local_size) {
    const float value = lhs[index] + rhs[index];
    output[index] = value;
    const double wide_value = (double)value;
    sum_squares += wide_value * wide_value;
  }
  partial[lid] = sum_squares;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint step = local_size >> 1U; step > 0U; step >>= 1U) {
    if (lid < step) partial[lid] += partial[lid + step];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lid == 0U) {
    const float mean_square = (float)partial[0] / (float)hidden_size;
    scale_out[0] = rsqrt(mean_square + epsilon);
  }
}

__kernel void full_attn_qk_norm_rope_f32(
    __global const float* q_full,
    __global const float* k_raw,
    __global const float* q_norm_weight,
    __global const float* k_norm_weight,
    __global const float* rope_cache,
    uint head_dim,
    uint q_head_count,
    uint kv_head_count,
    uint rope_dimension_count,
    float norm_epsilon,
    __global float* q_rope,
    __global float* k_rope) {
  const uint index = (uint)get_global_id(0);
  const uint q_values = q_head_count * head_dim;
  const uint kv_values = kv_head_count * head_dim;
  if (index >= q_values + kv_values) {
    return;
  }

  const bool is_q = index < q_values;
  const uint local_index = is_q ? index : index - q_values;
  const uint head = local_index / head_dim;
  const uint dim = local_index - head * head_dim;
  const uint q_head_stride = head_dim * 2U;
  const uint base = is_q ? head * q_head_stride : head * head_dim;
  __global const float* input = is_q ? q_full : k_raw;
  __global const float* weight = is_q ? q_norm_weight : k_norm_weight;

  float sum_squares = 0.0f;
  for (uint i = 0; i < head_dim; ++i) {
    const uint input_index = is_q ? base + i : base + i;
    const float value = input[input_index];
    sum_squares += value * value;
  }
  const float scale = 1.0f / sqrt(sum_squares / (float)head_dim + norm_epsilon);

  const uint input_index = is_q ? base + dim : base + dim;
  float out = input[input_index] * scale * weight[dim];
  const uint rotated_half = rope_dimension_count / 2U;
  if (dim < rope_dimension_count && rotated_half > 0U) {
    const uint pair_dim = dim < rotated_half ? dim + rotated_half
                                             : dim - rotated_half;
    const uint pair_index = is_q ? base + pair_dim : base + pair_dim;
    const float x0 =
        dim < rotated_half ? out : input[pair_index] * scale * weight[pair_dim];
    const float x1 =
        dim < rotated_half ? input[pair_index] * scale * weight[pair_dim] : out;
    const uint cache_index =
        (dim < rotated_half ? dim : pair_dim) * 2U;
    const float cos_theta = rope_cache[cache_index];
    const float sin_theta = rope_cache[cache_index + 1U];
    out = dim < rotated_half ? x0 * cos_theta - x1 * sin_theta
                             : x0 * sin_theta + x1 * cos_theta;
  }

  if (is_q) {
    q_rope[index] = out;
  } else {
    k_rope[local_index] = out;
  }
}

__kernel void rms_norm_hidden_f32(__global const float* input,
                                  __global const float* weight,
                                  uint hidden_size,
                                  float epsilon,
                                  __global float* output) {
  if ((uint)get_global_id(0) != 0U) {
    return;
  }
  float sum_squares = 0.0f;
  for (uint i = 0; i < hidden_size; ++i) {
    const float value = input[i];
    sum_squares += value * value;
  }
  const float mean_square = sum_squares / (float)hidden_size;
  const float scale = rsqrt(mean_square + epsilon);
  for (uint i = 0; i < hidden_size; ++i) {
    output[i] = input[i] * scale * weight[i];
  }
}

__kernel void rms_norm_hidden_scale_f32(__global const float* input,
                                        uint hidden_size,
                                        float epsilon,
                                        __global float* scale_out,
                                        uint serial_reduction) {
  const uint lid = (uint)get_local_id(0);
  const uint local_size = (uint)get_local_size(0);
  __local float partial[256];
  float sum_squares = 0.0f;
  if (serial_reduction == 1U) {
    if (lid == 0U) {
      for (uint i = 0; i < hidden_size; ++i) {
        const float value = input[i];
        sum_squares += value * value;
      }
    }
  } else if (serial_reduction == 2U) {
    if (lid < 2U) {
      const uint chunk = (hidden_size + 1U) / 2U;
      const uint begin = lid * chunk;
      const uint end = min(begin + chunk, hidden_size);
      for (uint i = begin; i < end; ++i) {
        const float value = input[i];
        sum_squares += value * value;
      }
      partial[lid] = sum_squares;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid == 0U) {
      sum_squares = partial[0] + partial[1];
    }
  } else {
    const uint chunk = (hidden_size + local_size - 1U) / local_size;
    const uint begin = lid * chunk;
    const uint end = min(begin + chunk, hidden_size);
    for (uint i = begin; i < end; ++i) {
      const float value = input[i];
      sum_squares += value * value;
    }
    partial[lid] = sum_squares;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid == 0U) {
      float total = 0.0f;
      for (uint i = 0; i < local_size; ++i) {
        total += partial[i];
      }
      sum_squares = total;
    }
  }
  if (lid == 0U) {
    const float mean_square = sum_squares / (float)hidden_size;
    scale_out[0] = rsqrt(mean_square + epsilon);
  }
}

__kernel void rms_norm_hidden_scale_f64_parallel(
    __global const float* input,
    uint hidden_size,
    float epsilon,
    __global float* scale_out) {
  const uint lid = (uint)get_local_id(0);
  const uint local_size = (uint)get_local_size(0);
  __local double partial[256];
  double sum_squares = 0.0;
  for (uint index = lid; index < hidden_size; index += local_size) {
    const double value = (double)input[index];
    sum_squares += value * value;
  }
  partial[lid] = sum_squares;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint step = local_size >> 1U; step > 0U; step >>= 1U) {
    if (lid < step) {
      partial[lid] += partial[lid + step];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lid == 0U) {
    const float mean_square =
        (float)partial[0] / (float)hidden_size;
    scale_out[0] = rsqrt(mean_square + epsilon);
  }
}

__kernel void rms_norm_hidden_apply_scale_f32(__global const float* input,
                                              __global const float* weight,
                                              uint hidden_size,
                                              __global const float* scale_in,
                                              __global float* output) {
  const uint index = (uint)get_global_id(0);
  if (index >= hidden_size) {
    return;
  }
  output[index] = input[index] * scale_in[0] * weight[index];
}

__kernel void rms_norm_hidden_apply_q8_f32(
    __global const float* input,
    __global const float* weight,
    uint hidden_size,
    __global const float* scale_in,
    __global float* output,
    __global char* q8_qs,
    __global short* q8_bsums,
    __global float* q8_d) {
  const uint block = (uint)get_group_id(0);
  const uint lid = (uint)get_local_id(0);
  const uint index = block * 256U + lid;
  if (index >= hidden_size) return;
  __local float values[256];
  __local float iscale_shared;
  __local int quantized[256];
  const float value = input[index] * scale_in[0] * weight[index];
  output[index] = value;
  values[lid] = value;
  barrier(CLK_LOCAL_MEM_FENCE);
  const float amax = work_group_reduce_max(fabs(values[lid]));
  const uint winner = work_group_reduce_min(
      fabs(values[lid]) == amax ? lid : 0xffffffffU);
  if (lid == 0U) {
    const float max_value = values[winner];
    iscale_shared = amax == 0.0f ? 0.0f : -127.0f / max_value;
    q8_d[block] = amax == 0.0f ? 0.0f : 1.0f / iscale_shared;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  const int q = iscale_shared == 0.0f
      ? 0
      : min(127, nearest_int_f32(iscale_shared * value));
  quantized[lid] = q;
  q8_qs[index] = (char)q;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lid < 16U) {
    int sum = 0;
    for (uint i = 0; i < 16U; ++i) {
      sum += quantized[lid * 16U + i];
    }
    q8_bsums[block * 16U + lid] = (short)sum;
  }
}
