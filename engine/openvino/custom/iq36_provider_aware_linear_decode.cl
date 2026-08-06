#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable

// Fixed decode-only Qwen3.6 linear-attention boundary. One operation consumes
// the U4 FC output plus both recurrent states, performs the already accepted
// width-four conv/state/SiLU boundary, and feeds its values directly into the
// exact GatedDeltaNet recurrence. The materialized 8192-element activated QKV
// tensor and the second provider launch disappear.

#define IQ36_FEATURES 8192
#define IQ36_CONV_WIDTH 4
#define IQ36_VALUE_HEADS 32
#define IQ36_KEY_HEADS 16
#define IQ36_HEAD_SIZE 128
#define IQ36_SUBGROUP_SIZE 16
#define IQ36_SUBGROUPS_PER_GROUP 16
#define IQ36_ATTENTION_ELEMENTS \
  (IQ36_VALUE_HEADS * IQ36_HEAD_SIZE)
#define IQ36_CONV_STATE_ELEMENTS \
  (IQ36_FEATURES * IQ36_CONV_WIDTH)
#define IQ36_GDN_STATE_OFFSET \
  (IQ36_ATTENTION_ELEMENTS + IQ36_CONV_STATE_ELEMENTS)

inline float iq36_sum8(float8 value) {
  return value.s0 + value.s1 + value.s2 + value.s3 +
         value.s4 + value.s5 + value.s6 + value.s7;
}

inline float iq36_l2_scale(float sum, float extra_scale) {
  sum = sub_group_reduce_add(sum);
  sum = sub_group_broadcast(sum, 0);
  return rsqrt(sum + 1.0e-6f) * extra_scale;
}

inline OUTPUT0_TYPE iq36_conv_swish(
    const __global INPUT0_TYPE* fc_output,
    const __global INPUT1_TYPE* previous_state,
    const __global INPUT2_TYPE* weights,
    int feature) {
  const int state_base = feature * IQ36_CONV_WIDTH;
  OUTPUT0_TYPE sum = (OUTPUT0_TYPE)0;
  sum = fma((OUTPUT0_TYPE)previous_state[state_base + 1],
            (OUTPUT0_TYPE)weights[state_base + 0], sum);
  sum = fma((OUTPUT0_TYPE)previous_state[state_base + 2],
            (OUTPUT0_TYPE)weights[state_base + 1], sum);
  sum = fma((OUTPUT0_TYPE)previous_state[state_base + 3],
            (OUTPUT0_TYPE)weights[state_base + 2], sum);
  sum = fma((OUTPUT0_TYPE)fc_output[feature],
            (OUTPUT0_TYPE)weights[state_base + 3], sum);
  // These volatile half boundaries preserve the two materialization rounds in
  // the accepted standalone conv kernel even after this function is inlined.
  const volatile OUTPUT0_TYPE convolved = sum;
  const volatile OUTPUT0_TYPE activated =
      convolved /
      ((OUTPUT0_TYPE)1 + exp(-(OUTPUT0_TYPE)convolved));
  return activated;
}

inline void iq36_write_conv_state(
    const __global INPUT0_TYPE* fc_output,
    const __global INPUT1_TYPE* previous_state,
    __global OUTPUT0_TYPE* output,
    int feature) {
  const int state_base = feature * IQ36_CONV_WIDTH;
  const int output_base = IQ36_ATTENTION_ELEMENTS + state_base;
  output[output_base + 0] =
      (OUTPUT0_TYPE)previous_state[state_base + 1];
  output[output_base + 1] =
      (OUTPUT0_TYPE)previous_state[state_base + 2];
  output[output_base + 2] =
      (OUTPUT0_TYPE)previous_state[state_base + 3];
  output[output_base + 3] = (OUTPUT0_TYPE)fc_output[feature];
}

__attribute__((intel_reqd_sub_group_size(IQ36_SUBGROUP_SIZE)))
__kernel void iq36_provider_aware_linear_decode(
    const __global INPUT0_TYPE* fc_output,
    const __global INPUT1_TYPE* previous_conv_state,
    const __global INPUT2_TYPE* conv_weights,
    const __global INPUT3_TYPE* initial_gdn_state,
    const __global INPUT4_TYPE* gate,
    const __global INPUT5_TYPE* beta,
    __global OUTPUT0_TYPE* output) {
  const int value_head = get_global_id(1);
  const int lane = get_sub_group_local_id();
  const int subgroup = get_sub_group_id();
  const int value_block =
      get_group_id(2) * IQ36_SUBGROUPS_PER_GROUP + subgroup;
  const int value_base = value_block * 4;
  const int key_head = value_head / 2;
  const int key_base = lane * 8;

  // A group owns half a value head. Its first subgroup computes the shared
  // normalized Q/K vectors once, replacing sixteen repeated conv evaluations.
  __local float local_query[IQ36_HEAD_SIZE];
  __local float local_key[IQ36_HEAD_SIZE];
  if (subgroup == 0) {
    float8 query = (float8)(0.0f);
    float8 key = (float8)(0.0f);
    float query_sum = 0.0f;
    float key_sum = 0.0f;
    for (int item = 0; item < 8; ++item) {
      const int key_index = key_base + item;
      const int query_feature =
          key_head * IQ36_HEAD_SIZE + key_index;
      const int key_feature =
          (IQ36_KEY_HEADS + key_head) * IQ36_HEAD_SIZE + key_index;
      query[item] = convert_float(iq36_conv_swish(
          fc_output, previous_conv_state, conv_weights, query_feature));
      key[item] = convert_float(iq36_conv_swish(
          fc_output, previous_conv_state, conv_weights, key_feature));
      query_sum += query[item] * query[item];
      key_sum += key[item] * key[item];
    }
    query *= (float8)(iq36_l2_scale(
        query_sum, 0.08838834764831845f));
    key *= (float8)(iq36_l2_scale(key_sum, 1.0f));
    for (int item = 0; item < 8; ++item) {
      const int key_index = key_base + item;
      local_query[key_index] = query[item];
      local_key[key_index] = key[item];
    }

    // Each paired value head shares one Q/K head. Publish its carried conv
    // state only once, from the first half and first subgroup.
    if (get_group_id(2) == 0 && (value_head & 1) == 0) {
      for (int item = 0; item < 8; ++item) {
        const int key_index = key_base + item;
        iq36_write_conv_state(
            fc_output, previous_conv_state, output,
            key_head * IQ36_HEAD_SIZE + key_index);
        iq36_write_conv_state(
            fc_output, previous_conv_state, output,
            (IQ36_KEY_HEADS + key_head) * IQ36_HEAD_SIZE + key_index);
      }
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  float8 query = (float8)(0.0f);
  float8 key = (float8)(0.0f);
  for (int item = 0; item < 8; ++item) {
    const int key_index = key_base + item;
    query[item] = local_query[key_index];
    key[item] = local_key[key_index];
  }

  float8 state[4];
  for (int value_lane = 0; value_lane < 4; ++value_lane) {
    const int value_index = value_base + value_lane;
    float8 column = (float8)(0.0f);
    for (int item = 0; item < 8; ++item) {
      const int key_index = key_base + item;
      const int state_index =
          (value_head * IQ36_HEAD_SIZE + key_index) *
              IQ36_HEAD_SIZE + value_index;
      column[item] = convert_float(initial_gdn_state[state_index]);
    }
    state[value_lane] = column;
  }

  float4 value = (float4)(0.0f);
  if (lane == 0) {
    for (int value_lane = 0; value_lane < 4; ++value_lane) {
      const int value_index = value_base + value_lane;
      const int feature =
          (2 * IQ36_KEY_HEADS + value_head) * IQ36_HEAD_SIZE + value_index;
      value[value_lane] = convert_float(iq36_conv_swish(
          fc_output, previous_conv_state, conv_weights, feature));
      iq36_write_conv_state(
          fc_output, previous_conv_state, output, feature);
    }
  }
  value = (float4)(
      sub_group_broadcast(value.s0, 0),
      sub_group_broadcast(value.s1, 0),
      sub_group_broadcast(value.s2, 0),
      sub_group_broadcast(value.s3, 0));

  const float decay = exp(convert_float(gate[value_head]));
  const float beta_value = convert_float(beta[value_head]);
  float4 key_part = (float4)(0.0f);
  for (int value_lane = 0; value_lane < 4; ++value_lane) {
    state[value_lane] *= (float8)(decay);
    key_part[value_lane] = iq36_sum8(fma(
        state[value_lane], key, (float8)(0.0f)));
  }
  const float4 state_key = (float4)(
      sub_group_reduce_add(key_part.s0),
      sub_group_reduce_add(key_part.s1),
      sub_group_reduce_add(key_part.s2),
      sub_group_reduce_add(key_part.s3));

  float4 query_part = (float4)(0.0f);
  for (int value_lane = 0; value_lane < 4; ++value_lane) {
    const float update =
        (value[value_lane] - state_key[value_lane]) * beta_value;
    state[value_lane] += key * (float8)(update);
    query_part[value_lane] = iq36_sum8(fma(
        state[value_lane], query, (float8)(0.0f)));
  }
  const float4 token_output = (float4)(
      sub_group_reduce_add(query_part.s0),
      sub_group_reduce_add(query_part.s1),
      sub_group_reduce_add(query_part.s2),
      sub_group_reduce_add(query_part.s3));

  if (lane == 0) {
    for (int value_lane = 0; value_lane < 4; ++value_lane) {
      const int value_index = value_base + value_lane;
      output[value_head * IQ36_HEAD_SIZE + value_index] =
          (OUTPUT0_TYPE)(token_output[value_lane]);
    }
  }

  for (int value_lane = 0; value_lane < 4; ++value_lane) {
    const int value_index = value_base + value_lane;
    for (int item = 0; item < 8; ++item) {
      const int key_index = key_base + item;
      const int state_index =
          (value_head * IQ36_HEAD_SIZE + key_index) *
              IQ36_HEAD_SIZE + value_index;
      output[IQ36_GDN_STATE_OFFSET + state_index] =
          (OUTPUT0_TYPE)(state[value_lane][item]);
    }
  }
}

#undef IQ36_SUBGROUPS_PER_GROUP
#undef IQ36_SUBGROUP_SIZE
#undef IQ36_GDN_STATE_OFFSET
#undef IQ36_CONV_STATE_ELEMENTS
#undef IQ36_ATTENTION_ELEMENTS
#undef IQ36_HEAD_SIZE
#undef IQ36_KEY_HEADS
#undef IQ36_VALUE_HEADS
#undef IQ36_CONV_WIDTH
#undef IQ36_FEATURES
