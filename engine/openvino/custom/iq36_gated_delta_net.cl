#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable

// Fixed Qwen3.6-35B-A3B linear-attention boundary:
//   input0 [batch, token, 64, 128] = q16 | k16 | v32
//   input1 [batch, 32, 128, 128]   = recurrent state [head,key,value]
//   input2/3 [batch, token, 32, 1] = log gate / beta
//   output0 flat = [all token-major attention][all recurrent state]
// OpenVINO's SimpleGPU custom-operation path permits one output, so this fixed
// flat carrier matches the stock graph's post-Loop Reshape+Concat boundary.
// The graph decodes the two semantic outputs directly from this carrier and
// removes the original Loop Reshape+Concat+Slice unpack/repack chain.
// The workgroup owns four adjacent value columns for one value head.  Its
// arithmetic and operation order mirror OpenVINO's stock Xe3 reference kernel;
// the surrounding custom operation fixes layout and datatype explicitly.

#define IQ36_TOKEN_COUNT 1024
#define IQ36_QKV_HEAD_COUNT 64
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

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void iq36_gated_delta_net(
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
    const int qkv_token_base =
        (batch * IQ36_TOKEN_COUNT + token) *
        IQ36_QKV_HEAD_COUNT * IQ36_HEAD_SIZE;
    for (int item = 0; item < 8; ++item) {
      const int key_index = key_base + item;
      const int query_offset =
          qkv_token_base + key_head * IQ36_HEAD_SIZE + key_index;
      const int key_offset =
          qkv_token_base + (16 + key_head) * IQ36_HEAD_SIZE + key_index;
      query[item] = convert_float(qkv[query_offset]);
      key[item] = convert_float(qkv[key_offset]);
      query_sum += query[item] * query[item];
      key_sum += key[item] * key[item];
    }
    query *= (float8)(iq36_l2_scale(
        query_sum, 0.08838834764831845f));
    key *= (float8)(iq36_l2_scale(key_sum, 1.0f));

    float4 value = (float4)(0.0f);
    for (int value_lane = 0; value_lane < 4; ++value_lane) {
      const int value_index = value_base + value_lane;
      const int value_offset =
          qkv_token_base + (32 + value_head) * IQ36_HEAD_SIZE + value_index;
      value[value_lane] = convert_float(qkv[value_offset]);
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
#undef IQ36_QKV_HEAD_COUNT
#undef IQ36_TOKEN_COUNT
