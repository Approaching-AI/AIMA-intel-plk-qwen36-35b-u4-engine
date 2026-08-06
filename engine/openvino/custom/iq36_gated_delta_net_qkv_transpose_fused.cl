#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable

// Fixed Qwen3.6-35B-A3B linear-attention boundary with the qkv Transpose
// absorbed into indexing:
//   input0 [batch, 8192, 1, token] = feature-major q16 | k16 | v32
//   input1 [batch, 32, 128, 128]   = recurrent state [head,key,value]
//   input2/3 [batch, token, 32, 1] = log gate / beta
//   output0 flat = [all token-major attention][all recurrent state]
// Every locked qkv Transpose is [0,2,1] and has exactly one consumer. Reading
// its Swish producer as feature-major input therefore deletes the materialized
// boundary without changing the GDN arithmetic or output order.

#define IQ36_TOKEN_COUNT 1024
#define IQ36_QKV_FEATURE_COUNT 8192
#define IQ36_VALUE_HEAD_COUNT 32
#define IQ36_HEAD_SIZE 128
#define IQ36_ATTENTION_ELEMENTS \
  (IQ36_TOKEN_COUNT * IQ36_VALUE_HEAD_COUNT * IQ36_HEAD_SIZE)

inline float iq36_sum8(float8 value) {
  return value.s0 + value.s1 + value.s2 + value.s3 +
         value.s4 + value.s5 + value.s6 + value.s7;
}

inline float iq36_l2_scale(float sum, float extra_scale) {
  sum = sub_group_reduce_add(sum);
  sum = sub_group_broadcast(sum, 0);
  return rsqrt(sum + 1.0e-6f) * extra_scale;
}

inline int iq36_feature_major_offset(
    int batch, int feature, int token) {
  return ((batch * IQ36_QKV_FEATURE_COUNT + feature) *
          IQ36_TOKEN_COUNT) + token;
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void iq36_gated_delta_net_qkv_transpose_fused(
    const __global INPUT0_TYPE* qkv,
    const __global INPUT1_TYPE* initial_state,
    const __global INPUT2_TYPE* gate,
    const __global INPUT3_TYPE* beta,
    __global OUTPUT0_TYPE* output) {
  const int batch = get_global_id(0);
  const int value_head = get_global_id(1);
  const int lane = get_sub_group_local_id();
  const int value_base = get_group_id(2) * 4;
  const int key_head = value_head / 2;
  const int key_base = lane * 8;

  float8 state[4];
  for (int value_lane = 0; value_lane < 4; ++value_lane) {
    const int value_index = value_base + value_lane;
    float8 column = (float8)(0.0f);
    for (int item = 0; item < 8; ++item) {
      const int key_index = key_base + item;
      const int state_index =
          ((batch * IQ36_VALUE_HEAD_COUNT + value_head) * IQ36_HEAD_SIZE +
           key_index) * IQ36_HEAD_SIZE + value_index;
      column[item] = convert_float(initial_state[state_index]);
    }
    state[value_lane] = column;
  }

  for (int token = 0; token < IQ36_TOKEN_COUNT; ++token) {
    const int gate_index =
        (batch * IQ36_TOKEN_COUNT + token) * IQ36_VALUE_HEAD_COUNT +
        value_head;
    const int beta_index = gate_index;
    const float decay = exp(convert_float(gate[gate_index]));
    const float beta_value = convert_float(beta[beta_index]);

    float8 query = (float8)(0.0f);
    float8 key = (float8)(0.0f);
    float query_sum = 0.0f;
    float key_sum = 0.0f;
    for (int item = 0; item < 8; ++item) {
      const int key_index = key_base + item;
      const int query_feature = key_head * IQ36_HEAD_SIZE + key_index;
      const int key_feature =
          (16 + key_head) * IQ36_HEAD_SIZE + key_index;
      query[item] = convert_float(qkv[iq36_feature_major_offset(
          batch, query_feature, token)]);
      key[item] = convert_float(qkv[iq36_feature_major_offset(
          batch, key_feature, token)]);
      query_sum += query[item] * query[item];
      key_sum += key[item] * key[item];
    }
    query *= (float8)(iq36_l2_scale(
        query_sum, 0.08838834764831845f));
    key *= (float8)(iq36_l2_scale(key_sum, 1.0f));

    float4 value = (float4)(0.0f);
    for (int value_lane = 0; value_lane < 4; ++value_lane) {
      const int value_index = value_base + value_lane;
      const int value_feature =
          (32 + value_head) * IQ36_HEAD_SIZE + value_index;
      value[value_lane] = convert_float(qkv[iq36_feature_major_offset(
          batch, value_feature, token)]);
    }

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
        const int output_index =
            ((batch * IQ36_TOKEN_COUNT + token) *
             IQ36_VALUE_HEAD_COUNT + value_head) * IQ36_HEAD_SIZE +
            value_index;
        output[output_index] = (OUTPUT0_TYPE)(token_output[value_lane]);
      }
    }
  }

  for (int value_lane = 0; value_lane < 4; ++value_lane) {
    const int value_index = value_base + value_lane;
    for (int item = 0; item < 8; ++item) {
      const int key_index = key_base + item;
      const int state_index =
          IQ36_ATTENTION_ELEMENTS +
          ((batch * IQ36_VALUE_HEAD_COUNT + value_head) * IQ36_HEAD_SIZE +
           key_index) * IQ36_HEAD_SIZE + value_index;
      output[state_index] = (OUTPUT0_TYPE)(state[value_lane][item]);
    }
  }
}

#undef IQ36_ATTENTION_ELEMENTS
#undef IQ36_HEAD_SIZE
#undef IQ36_VALUE_HEAD_COUNT
#undef IQ36_QKV_FEATURE_COUNT
#undef IQ36_TOKEN_COUNT
