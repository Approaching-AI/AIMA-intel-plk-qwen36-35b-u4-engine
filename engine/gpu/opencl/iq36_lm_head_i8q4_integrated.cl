#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable

#define IQ36_ROWS 248320U
#define IQ36_GROUPS 16U
#define IQ36_CODE_BYTES_PER_GROUP8 512U
#define IQ36_CODE_BYTES_PER_STRIPE 8192U
#define IQ36_STRIPE_BYTES 8224U
#define IQ36_ADAPTIVE_CORRECTION_CAPACITY 4096U
#define IQ36_ADAPTIVE_CORRECTION_DELTA 8.0f

#if !defined(IQ36_QUANTIZE_KERNEL) && !defined(IQ36_TOPK_MATVEC_KERNEL) && \
    !defined(IQ36_OUTPUT_TOPK_KERNEL) && !defined(IQ36_TOPK_MERGE_KERNEL) && \
    !defined(IQ36_CORRECTION_KERNEL) && !defined(IQ36_DIRECT_CORRECTION_KERNEL) && \
    !defined(IQ36_ADAPTIVE_RESET_KERNEL) && \
    !defined(IQ36_ADAPTIVE_COLLECT_KERNEL) && \
    !defined(IQ36_ADAPTIVE_CORRECTION_KERNEL)
#define IQ36_QUANTIZE_KERNEL 1
#define IQ36_TOPK_MATVEC_KERNEL 1
#define IQ36_OUTPUT_TOPK_KERNEL 1
#define IQ36_TOPK_MERGE_KERNEL 1
#define IQ36_CORRECTION_KERNEL 1
#define IQ36_DIRECT_CORRECTION_KERNEL 1
#define IQ36_ADAPTIVE_RESET_KERNEL 1
#define IQ36_ADAPTIVE_COLLECT_KERNEL 1
#define IQ36_ADAPTIVE_CORRECTION_KERNEL 1
#endif

#if defined(IQ36_TOPK_MATVEC_KERNEL)
inline float iq36_lm_head_i8q4_rowstripe8_dot(
    __global const uchar* packed,
    __global const char* q8_qs,
    __global const float* q8_d,
    ulong row) {
  const ulong row_group = row >> 3U;
  const uint row_lane = (uint)(row & 7UL);
  __global const uchar* stripe =
      packed + row_group * (ulong)IQ36_STRIPE_BYTES;
  float scaled_sum = 0.0f;
  for (uint group = 0U; group < IQ36_GROUPS; ++group) {
    __global const uchar* block =
        stripe + group * IQ36_CODE_BYTES_PER_GROUP8;
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
  const float weight_scale = vload_half(
      0, (__global const half*)(
          stripe + IQ36_CODE_BYTES_PER_STRIPE + row_lane * 2U));
  return scaled_sum * weight_scale * 16.0f;
}
#endif

#if defined(IQ36_QUANTIZE_KERNEL)
__attribute__((reqd_work_group_size(64, 1, 1)))
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void iq36_lm_head_q8_group256_f16(
    __global const half* input,
    __global char* q8_qs,
    __global float* q8_d) {
  const uint group = (uint)get_group_id(0);
  const uint lane = (uint)get_local_id(0);
  const uint base = group * 256U + lane * 4U;
  const half4 values = vload4(0, input + base);
  __local half maxima[64];
  __local half quantize_scale;
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
  vstore4(convert_char4_rte(values * (half4)quantize_scale),
          0, q8_qs + base);
}
#endif

#if defined(IQ36_TOPK_MATVEC_KERNEL)
__attribute__((reqd_work_group_size(256, 1, 1)))
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void iq36_lm_head_i8q4_rowstripe8_matvec_topk8_f16(
    __global const uchar* packed,
    __global const char* q8_qs,
    __global const float* q8_d,
    __global half* output,
    __global int* partial_top_ids,
    __global float* partial_top_values) {
  const uint row = (uint)get_global_id(0);
  const uint lane = (uint)get_local_id(0);
  const uint block = (uint)get_group_id(0);
  const float full_value = row < IQ36_ROWS
      ? iq36_lm_head_i8q4_rowstripe8_dot(packed, q8_qs, q8_d, (ulong)row)
      : -INFINITY;
  const half rounded = convert_half_rte(full_value);
  const float value = row < IQ36_ROWS ? convert_float(rounded) : -INFINITY;
  if (row < IQ36_ROWS) output[row] = rounded;
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
  const uint count = min(256U, IQ36_ROWS - begin);
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
#endif

#if defined(IQ36_OUTPUT_TOPK_KERNEL)
__attribute__((reqd_work_group_size(256, 1, 1)))
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void iq36_lm_head_output_topk8_f16(
    __global const half* output,
    __global int* partial_top_ids,
    __global float* partial_top_values) {
  const uint row = (uint)get_global_id(0);
  const uint lane = (uint)get_local_id(0);
  const uint block = (uint)get_group_id(0);
  const float value = row < IQ36_ROWS
      ? convert_float(output[row])
      : -INFINITY;
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
  const uint count = min(256U, IQ36_ROWS - begin);
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
#endif

#if defined(IQ36_TOPK_MERGE_KERNEL)
__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void iq36_lm_head_topk8_merge_f32(
    __global const int* partial_ids,
    __global const float* partial_values,
    __global int* top_ids,
    __global float* top_values) {
  const uint lane = (uint)get_local_id(0);
  float best_values[8];
  int best_ids[8];
  for (uint index = 0U; index < 8U; ++index) {
    best_values[index] = -INFINITY;
    best_ids[index] = -1;
  }
  for (uint candidate = lane; candidate < 970U * 8U; candidate += 256U) {
    const float value = partial_values[candidate];
    const int id = partial_ids[candidate];
    uint position = 0U;
    while (position < 8U &&
           (best_values[position] > value ||
            (best_values[position] == value && best_ids[position] < id))) {
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
#endif

#if defined(IQ36_CORRECTION_KERNEL)
__attribute__((reqd_work_group_size(64, 1, 1)))
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void iq36_lm_head_i8_exact_topk8_correction_f16(
    __global const char* weights,
    __global const half* weight_scales,
    __global const char* q8_qs,
    __global const float* q8_d,
    __global const int* top_ids,
    __global half* output) {
  const uint selected = (uint)get_group_id(0);
  const uint lane = (uint)get_local_id(0);
  if (selected >= 8U) return;
  const int signed_row = top_ids[selected];
  if (signed_row < 0 || signed_row >= (int)IQ36_ROWS) return;
  const uint row = (uint)signed_row;
  const ulong row_base = (ulong)row * 2048UL;
  const uint lane_base = lane * 32U;
  int dot_sum = 0;
  for (uint chunk = 0U; chunk < 8U; ++chunk) {
    const uint element = lane_base + chunk * 4U;
#if defined(IQ36_USE_INTEGER_DOT)
    dot_sum += dot(vload4(0, weights + row_base + element),
                   vload4(0, q8_qs + element));
#else
    const char4 weight = vload4(0, weights + row_base + element);
    const char4 input = vload4(0, q8_qs + element);
    dot_sum += (int)weight.s0 * (int)input.s0;
    dot_sum += (int)weight.s1 * (int)input.s1;
    dot_sum += (int)weight.s2 * (int)input.s2;
    dot_sum += (int)weight.s3 * (int)input.s3;
#endif
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
    output[row] = convert_half_rte(value * convert_float(weight_scales[row]));
  }
}
#endif

#if defined(IQ36_DIRECT_CORRECTION_KERNEL)
__attribute__((reqd_work_group_size(64, 1, 1)))
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void iq36_lm_head_i8_direct_topk8_correction_f16(
    __global const char* weights,
    __global const half* weight_scales,
    __global const half* input,
    __global const int* top_ids,
    __global half* output) {
  const uint selected = (uint)get_group_id(0);
  const uint lane = (uint)get_local_id(0);
  if (selected >= 8U) return;
  const int signed_row = top_ids[selected];
  if (signed_row < 0 || signed_row >= (int)IQ36_ROWS) return;
  const uint row = (uint)signed_row;
  const ulong row_base = (ulong)row * 2048UL;
  const uint lane_base = lane * 32U;
  float dot_sum = 0.0f;
  for (uint chunk = 0U; chunk < 8U; ++chunk) {
    const uint element = lane_base + chunk * 4U;
    dot_sum += dot(
        convert_float4(vload4(0, weights + row_base + element)),
        convert_float4(vload4(0, input + element)));
  }
  __local float partial[64];
  partial[lane] = dot_sum;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lane == 0U) {
    float value = 0.0f;
    for (uint contributor = 0U; contributor < 64U; ++contributor)
      value += partial[contributor];
    output[row] = convert_half_rte(value * convert_float(weight_scales[row]));
  }
}
#endif

#if defined(IQ36_ADAPTIVE_RESET_KERNEL)
__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void iq36_lm_head_adaptive_correction_reset(
    __global uint* selected_count) {
  selected_count[0] = 0U;
}
#endif

#if defined(IQ36_ADAPTIVE_COLLECT_KERNEL)
__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void iq36_lm_head_adaptive_correction_collect_f16(
    __global const half* output,
    __global const float* top_values,
    volatile __global uint* selected_count,
    __global int* selected_ids) {
  const uint row = (uint)get_global_id(0);
  if (row >= IQ36_ROWS) return;
  const float cutoff = top_values[0] - IQ36_ADAPTIVE_CORRECTION_DELTA;
  if (convert_float(output[row]) < cutoff) return;
  const uint slot = atomic_inc(selected_count);
  if (slot < IQ36_ADAPTIVE_CORRECTION_CAPACITY)
    selected_ids[slot] = (int)row;
}
#endif

#if defined(IQ36_ADAPTIVE_CORRECTION_KERNEL)
__attribute__((reqd_work_group_size(64, 1, 1)))
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void iq36_lm_head_i8_adaptive_correction_f16(
    __global const char* weights,
    __global const half* weight_scales,
    __global const half* input,
    __global const uint* selected_count,
    __global const int* selected_ids,
    __global half* output) {
  const uint selected = (uint)get_group_id(0);
  const uint lane = (uint)get_local_id(0);
  const uint count = min(selected_count[0], IQ36_ADAPTIVE_CORRECTION_CAPACITY);
  if (selected >= count) return;
  const int signed_row = selected_ids[selected];
  if (signed_row < 0 || signed_row >= (int)IQ36_ROWS) return;
  const uint row = (uint)signed_row;
  const ulong row_base = (ulong)row * 2048UL;
  const uint lane_base = lane * 32U;
  float dot_sum = 0.0f;
  for (uint chunk = 0U; chunk < 8U; ++chunk) {
    const uint element = lane_base + chunk * 4U;
    dot_sum += dot(
        convert_float4(vload4(0, weights + row_base + element)),
        convert_float4(vload4(0, input + element)));
  }
  __local float partial[64];
  partial[lane] = dot_sum;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lane == 0U) {
    float value = 0.0f;
    for (uint contributor = 0U; contributor < 64U; ++contributor)
      value += partial[contributor];
    output[row] = convert_half_rte(value * convert_float(weight_scales[row]));
  }
}
#endif
