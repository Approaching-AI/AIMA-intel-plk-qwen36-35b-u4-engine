#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable

// Standalone, forced-execution mirror of the accepted product fallback.  The
// component intentionally omits the gate state, reset, and collect stages so
// each timestamp covers exactly one fallback data-plane stage.
#define IQ36_ROWS 248320U
#define IQ36_COLUMNS 2048U
#define IQ36_Q8_GROUPS 8U
#define IQ36_BLOCK_ROWS 256U
#define IQ36_BLOCKS 970U
#define IQ36_TOPK 8U

__attribute__((reqd_work_group_size(64, 1, 1)))
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void iq36_lm_head_gated_exact_component_q8_f16(
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

__attribute__((reqd_work_group_size(256, 1, 1)))
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void iq36_lm_head_gated_exact_component_matvec_f16(
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
        value += (float)group_sum * q8_d[group];
    }
    if (subgroup_lane == 0U) {
      output[row] = convert_half_rte(
          value * convert_float(weight_scales[row]));
    }
  }
}

__attribute__((reqd_work_group_size(256, 1, 1)))
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void iq36_lm_head_gated_exact_component_block_topk8_f16(
    __global const half* output,
    __global int* partial_top_ids,
    __global float* partial_top_values) {
  const uint row = (uint)get_global_id(0);
  const uint lane = (uint)get_local_id(0);
  const uint block = (uint)get_group_id(0);
  const float value = row < IQ36_ROWS
      ? convert_float(output[row])
      : -INFINITY;
  __local float values[IQ36_BLOCK_ROWS];
  values[lane] = value;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lane != 0U) return;
  float best_values[IQ36_TOPK];
  int best_ids[IQ36_TOPK];
  for (uint index = 0U; index < IQ36_TOPK; ++index) {
    best_values[index] = -INFINITY;
    best_ids[index] = -1;
  }
  const uint begin = block * IQ36_BLOCK_ROWS;
  const uint count = min(IQ36_BLOCK_ROWS, IQ36_ROWS - begin);
  for (uint index = 0U; index < count; ++index) {
    const float candidate = values[index];
    uint position = 0U;
    while (position < IQ36_TOPK &&
           (best_values[position] > candidate ||
            (best_values[position] == candidate &&
             best_ids[position] < (int)(begin + index)))) {
      ++position;
    }
    if (position < IQ36_TOPK) {
      for (uint destination = IQ36_TOPK - 1U;
           destination > position; --destination) {
        best_values[destination] = best_values[destination - 1U];
        best_ids[destination] = best_ids[destination - 1U];
      }
      best_values[position] = candidate;
      best_ids[position] = (int)(begin + index);
    }
  }
  for (uint index = 0U; index < IQ36_TOPK; ++index) {
    partial_top_ids[block * IQ36_TOPK + index] = best_ids[index];
    partial_top_values[block * IQ36_TOPK + index] = best_values[index];
  }
}

// Arithmetic-equivalent candidate for the lane0-serial insertion scan above.
// Each lane owns exactly one F16 logit. Eight collective max/min rounds select
// the same value-descending, token-id-ascending order as the baseline.
__attribute__((reqd_work_group_size(256, 1, 1)))
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void iq36_lm_head_gated_exact_component_parallel_block_topk8_f16(
    __global const half* output,
    __global int* partial_top_ids,
    __global float* partial_top_values) {
  const uint lane = (uint)get_local_id(0);
  const uint block = (uint)get_group_id(0);
  const uint row = block * IQ36_BLOCK_ROWS + lane;
  float lane_value = convert_float(output[row]);
  for (uint selected = 0U; selected < IQ36_TOPK; ++selected) {
    const float maximum = work_group_reduce_max(lane_value);
    const int winner = work_group_reduce_min(
        lane_value == maximum ? (int)row : 0x7fffffff);
    if (lane == 0U) {
      partial_top_ids[block * IQ36_TOPK + selected] = winner;
      partial_top_values[block * IQ36_TOPK + selected] = maximum;
    }
    if ((int)row == winner) lane_value = -INFINITY;
  }
}

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void iq36_lm_head_gated_exact_component_topk8_merge_f32(
    __global const int* partial_ids,
    __global const float* partial_values,
    __global int* top_ids,
    __global float* top_values) {
  const uint lane = (uint)get_local_id(0);
  float best_values[IQ36_TOPK];
  int best_ids[IQ36_TOPK];
  for (uint index = 0U; index < IQ36_TOPK; ++index) {
    best_values[index] = -INFINITY;
    best_ids[index] = -1;
  }
  for (uint candidate = lane;
       candidate < IQ36_BLOCKS * IQ36_TOPK;
       candidate += 256U) {
    const float value = partial_values[candidate];
    const int id = partial_ids[candidate];
    uint position = 0U;
    while (position < IQ36_TOPK &&
           (best_values[position] > value ||
            (best_values[position] == value && best_ids[position] < id))) {
      ++position;
    }
    if (position < IQ36_TOPK) {
      for (uint destination = IQ36_TOPK - 1U;
           destination > position; --destination) {
        best_values[destination] = best_values[destination - 1U];
        best_ids[destination] = best_ids[destination - 1U];
      }
      best_values[position] = value;
      best_ids[position] = id;
    }
  }
  uint head = 0U;
  for (uint output = 0U; output < IQ36_TOPK; ++output) {
    const float lane_value =
        head < IQ36_TOPK ? best_values[head] : -INFINITY;
    const int lane_id =
        head < IQ36_TOPK ? best_ids[head] : 0x7fffffff;
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
__kernel void iq36_lm_head_gated_exact_component_correction_f16(
    __global const char* weights,
    __global const half* weight_scales,
    __global const half* input,
    __global const int* top_ids,
    __global half* output) {
  const uint selected = (uint)get_group_id(0);
  const uint lane = (uint)get_local_id(0);
  if (selected >= IQ36_TOPK) return;
  const int signed_row = top_ids[selected];
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
    output[row] = convert_half_rte(
        value * convert_float(weight_scales[row]));
  }
}
