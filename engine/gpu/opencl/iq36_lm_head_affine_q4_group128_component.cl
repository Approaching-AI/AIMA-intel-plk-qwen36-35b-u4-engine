#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable

// Standalone component for the seq2289 affine-Q4/group128 certificate.
// The existing product Q8 activation is an input to both timed paths.  The
// candidate adds groupwise hidden norms, a conservative upper-bound scan,
// compact exact-I8 correction, and exact top-1 selection.  The full-I8 path is
// retained as the capacity-overflow fallback and timing denominator.
#define IQ36_ROWS 248320U
#define IQ36_COLUMNS 2048U
#define IQ36_Q8_GROUPS 8U
#define IQ36_GROUPS128 16U
#define IQ36_GROUP128 128U
#define IQ36_Q4_ROW_BYTES 1024U
#define IQ36_CAPACITY 16812U
#define IQ36_SCAN_WORKGROUPS 384U
#define IQ36_F32_U (0x1.0p-24f)
#define IQ36_DOT_GAMMA \
  ((4096.0f * IQ36_F32_U) / (1.0f - 4096.0f * IQ36_F32_U))

inline float iq36_f16_norm_upper(float square_sum) {
  // F32 square/reduction inflation is deliberately wider than the 128-term
  // gamma.  The final nextafter + RTP conversion mirrors the CPU certificate's
  // outward F32 then F16 rounding.
  const float norm = sqrt(square_sum * 1.00002f);
  return convert_float(convert_half_rtp(nextafter(norm, INFINITY)));
}

inline uint iq36_q4_code(
    __global const uchar* codes, ulong row_base, uint column) {
  const uchar packed = codes[row_base + (ulong)(column >> 1U)];
  return (column & 1U) == 0U
      ? (uint)(packed & (uchar)15)
      : (uint)(packed >> 4U);
}

__attribute__((reqd_work_group_size(64, 1, 1)))
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void iq36_affine_q4_q8_f16(
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

__attribute__((reqd_work_group_size(128, 1, 1)))
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void iq36_affine_q4_hidden_group_norms_f16(
    __global const half* input,
    __global const char* q8_qs,
    __global const float* q8_d,
    __global half* hidden_norms,
    __global half* hidden_delta_norms) {
  const uint q8_group = (uint)get_sub_group_id();
  const uint lane = (uint)get_sub_group_local_id();
  const uint begin = q8_group * 256U + lane * 8U;
  const half dequant_scale = convert_half_rte(q8_d[q8_group]);
  float hidden_square = 0.0f;
  float delta_square = 0.0f;
  for (uint offset = 0U; offset < 8U; ++offset) {
    const uint column = begin + offset;
    const float value = convert_float(input[column]);
    const half dequant = convert_half(q8_qs[column]) * dequant_scale;
    const float delta = value - convert_float(dequant);
    hidden_square = fma(value, value, hidden_square);
    delta_square = fma(delta, delta, delta_square);
  }
  const float hidden_sum = sub_group_reduce_add(hidden_square);
  const float delta_sum = sub_group_reduce_add(delta_square);
  if (lane == 0U) {
    const uint group128 = q8_group * 2U;
    hidden_norms[group128] =
        convert_half_rtp(nextafter(sqrt(hidden_sum * 1.00002f), INFINITY));
    hidden_delta_norms[group128] =
        convert_half_rtp(nextafter(sqrt(delta_sum * 1.00002f), INFINITY));
  }

  hidden_square = 0.0f;
  delta_square = 0.0f;
  const uint second = begin + 128U;
  for (uint offset = 0U; offset < 8U; ++offset) {
    const uint column = second + offset;
    const float value = convert_float(input[column]);
    const half dequant = convert_half(q8_qs[column]) * dequant_scale;
    const float delta = value - convert_float(dequant);
    hidden_square = fma(value, value, hidden_square);
    delta_square = fma(delta, delta, delta_square);
  }
  const float hidden_sum_second = sub_group_reduce_add(hidden_square);
  const float delta_sum_second = sub_group_reduce_add(delta_square);
  if (lane == 0U) {
    const uint group128 = q8_group * 2U + 1U;
    hidden_norms[group128] = convert_half_rtp(
        nextafter(sqrt(hidden_sum_second * 1.00002f), INFINITY));
    hidden_delta_norms[group128] = convert_half_rtp(
        nextafter(sqrt(delta_sum_second * 1.00002f), INFINITY));
  }
}

__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void iq36_affine_q4_reset(
    volatile __global uint* candidate_count) {
  candidate_count[0] = 0U;
}

__attribute__((reqd_work_group_size(256, 1, 1)))
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void iq36_affine_q4_bound_select_f16(
    __global const uchar* codes,
    __global const char* group_minmax,
    __global const half* residual_norms,
    __global const half* weight_scales,
    __global const char* q8_qs,
    __global const float* q8_d,
    __global const half* hidden_norms,
    __global const half* hidden_delta_norms,
    __global const half* seed_value,
    volatile __global uint* candidate_count,
    __global int* candidate_ids,
    __global half* upper_output) {
  const uint lane = (uint)get_sub_group_local_id();
  const uint subgroup =
      (uint)get_group_id(0) * 16U + (uint)get_sub_group_id();
  const uint subgroup_stride = (uint)get_num_groups(0) * 16U;
  for (uint row = subgroup; row < IQ36_ROWS; row += subgroup_stride) {
    const ulong code_row = (ulong)row * (ulong)IQ36_Q4_ROW_BYTES;
    const uint metadata_row = row * IQ36_GROUPS128 * 2U;
    float approximate = 0.0f;
    float residual_bound = 0.0f;
    float round_guard = 0.0f;
    const float row_scale = convert_float(weight_scales[row]);
    const float abs_scale = fabs(row_scale);
    for (uint group = 0U; group < IQ36_GROUPS128; ++group) {
      const uint metadata = metadata_row + group * 2U;
      const float minimum = convert_float(group_minmax[metadata]);
      const float maximum = convert_float(group_minmax[metadata + 1U]);
      const float step = (maximum - minimum) * (1.0f / 15.0f);
      const uint lane_base = group * IQ36_GROUP128 + lane * 8U;
      float lane_dot = 0.0f;
      float lane_codec_square = 0.0f;
      for (uint offset = 0U; offset < 8U; ++offset) {
        const uint column = lane_base + offset;
        const float codec = fma(
            convert_float(iq36_q4_code(codes, code_row, column)),
            step, minimum);
        lane_dot = fma(
            codec, convert_float(q8_qs[column]), lane_dot);
        lane_codec_square = fma(
            codec, codec, lane_codec_square);
      }
      const float dot_sum = sub_group_reduce_add(lane_dot);
      const float codec_square =
          sub_group_reduce_add(lane_codec_square);
      if (lane == 0U) {
        const float codec_norm = iq36_f16_norm_upper(codec_square);
        const float hidden_norm = convert_float(hidden_norms[group]);
        const float hidden_delta_norm =
            convert_float(hidden_delta_norms[group]);
        const float residual_norm = convert_float(
            residual_norms[row * IQ36_GROUPS128 + group]);
        approximate += dot_sum * q8_d[group >> 1U];
        residual_bound += abs_scale * (
            residual_norm * hidden_norm +
            codec_norm * hidden_delta_norm);
        const float magnitude =
            codec_norm * hidden_norm * abs_scale;
        round_guard += magnitude * IQ36_DOT_GAMMA +
            fmax(magnitude, 1.0f) * (2.0f * IQ36_F32_U);
      }
    }
    if (lane == 0U) {
      const float upper_real = fma(
          approximate, row_scale, residual_bound + round_guard);
      const half upper = convert_half_rtp(
          nextafter(upper_real, INFINITY));
      upper_output[row] = upper;
      if (upper >= seed_value[0]) {
        const uint slot = atomic_inc(candidate_count);
        if (slot < IQ36_CAPACITY) candidate_ids[slot] = (int)row;
      }
    }
  }
}

__attribute__((reqd_work_group_size(64, 1, 1)))
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void iq36_affine_q4_exact_candidates_f16(
    __global const char* weights,
    __global const half* weight_scales,
    __global const half* input,
    __global const uint* candidate_count,
    __global const int* candidate_ids,
    __global half* candidate_values) {
  const uint selected = (uint)get_group_id(0);
  const uint lane = (uint)get_local_id(0);
  const uint count = min(candidate_count[0], IQ36_CAPACITY);
  if (selected >= count) return;
  const int signed_row = candidate_ids[selected];
  if (signed_row < 0 || signed_row >= (int)IQ36_ROWS) return;
  const uint row = (uint)signed_row;
  const ulong row_base = (ulong)row * (ulong)IQ36_COLUMNS;
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
    candidate_values[selected] = convert_half_rte(
        value * convert_float(weight_scales[row]));
  }
}

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void iq36_affine_q4_candidate_top1_f16(
    __global const uint* candidate_count,
    __global const int* candidate_ids,
    __global const half* candidate_values,
    __global const int* seed_id,
    __global const half* seed_value,
    __global int* output_token) {
  const uint lane = (uint)get_local_id(0);
  const uint count = min(candidate_count[0], IQ36_CAPACITY);
  float best_value = lane == 0U
      ? convert_float(seed_value[0]) : -INFINITY;
  int best_id = lane == 0U ? seed_id[0] : 0x7fffffff;
  for (uint selected = lane; selected < count; selected += 256U) {
    const float value = convert_float(candidate_values[selected]);
    const int id = candidate_ids[selected];
    if (value > best_value || (value == best_value && id < best_id)) {
      best_value = value;
      best_id = id;
    }
  }
  const float maximum = work_group_reduce_max(best_value);
  const int winner = work_group_reduce_min(
      best_value == maximum ? best_id : 0x7fffffff);
  if (lane == 0U) output_token[0] = winner;
}

__attribute__((reqd_work_group_size(256, 1, 1)))
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void iq36_affine_q4_full_i8_q8_matvec_f16(
    __global const char* weights,
    __global const half* weight_scales,
    __global const char* q8_qs,
    __global const float* q8_d,
    __global half* output) {
  const uint subgroup_lane = (uint)get_sub_group_local_id();
  const uint subgroup =
      (uint)get_group_id(0) * 16U + (uint)get_sub_group_id();
  const uint subgroup_stride = (uint)get_num_groups(0) * 16U;
  for (uint row = subgroup; row < IQ36_ROWS; row += subgroup_stride) {
    const ulong row_base = (ulong)row * (ulong)IQ36_COLUMNS;
    float value = 0.0f;
    for (uint group = 0U; group < IQ36_Q8_GROUPS; ++group) {
      const uint lane_base = group * 256U + subgroup_lane * 4U;
      int lane_sum = 0;
      for (uint chunk = 0U; chunk < 4U; ++chunk) {
        const uint element = lane_base + chunk * 64U;
        lane_sum += dot(vload4(0, weights + row_base + element),
                        vload4(0, q8_qs + element));
      }
      const int group_sum = sub_group_reduce_add(lane_sum);
      if (subgroup_lane == 0U)
        value += convert_float(group_sum) * q8_d[group];
    }
    if (subgroup_lane == 0U) {
      output[row] = convert_half_rte(
          value * convert_float(weight_scales[row]));
    }
  }
}

__attribute__((reqd_work_group_size(256, 1, 1)))
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void iq36_affine_q4_reference_matvec_f16(
    __global const char* weights,
    __global const half* weight_scales,
    __global const half* input,
    __global half* output) {
  const uint subgroup_lane = (uint)get_sub_group_local_id();
  const uint subgroup =
      (uint)get_group_id(0) * 16U + (uint)get_sub_group_id();
  const uint subgroup_stride = (uint)get_num_groups(0) * 16U;
  for (uint row = subgroup; row < IQ36_ROWS; row += subgroup_stride) {
    const ulong row_base = (ulong)row * (ulong)IQ36_COLUMNS;
    float lane_sum = 0.0f;
    for (uint column = subgroup_lane * 4U;
         column < IQ36_COLUMNS; column += 64U) {
      lane_sum += dot(
          convert_float4(vload4(0, weights + row_base + column)),
          convert_float4(vload4(0, input + column)));
    }
    const float value = sub_group_reduce_add(lane_sum);
    if (subgroup_lane == 0U) {
      output[row] = convert_half_rte(
          value * convert_float(weight_scales[row]));
    }
  }
}

__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void iq36_affine_q4_violation_reset(
    volatile __global uint* violation_count) {
  violation_count[0] = 0U;
}

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void iq36_affine_q4_bound_violations_f16(
    __global const half* reference,
    __global const half* upper_output,
    volatile __global uint* violation_count) {
  const uint row = (uint)get_global_id(0);
  if (row < IQ36_ROWS && reference[row] > upper_output[row])
    atomic_inc(violation_count);
}

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void iq36_affine_q4_full_top1_f16(
    __global const half* output,
    __global int* output_token) {
  const uint lane = (uint)get_local_id(0);
  float best_value = -INFINITY;
  int best_id = 0x7fffffff;
  for (uint row = lane; row < IQ36_ROWS; row += 256U) {
    const float value = convert_float(output[row]);
    if (value > best_value || (value == best_value && (int)row < best_id)) {
      best_value = value;
      best_id = (int)row;
    }
  }
  const float maximum = work_group_reduce_max(best_value);
  const int winner = work_group_reduce_min(
      best_value == maximum ? best_id : 0x7fffffff);
  if (lane == 0U) output_token[0] = winner;
}
