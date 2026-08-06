#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#pragma OPENCL EXTENSION cl_khr_subgroup_shuffle : enable
#pragma OPENCL EXTENSION cl_khr_subgroup_non_uniform_arithmetic : enable
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Locked Qwen3.6-35B-A3B linear-attention shape.  One workgroup owns sixteen
// columns of one 128x128 recurrent-state head for the full prompt tile.  The
// state shard stays in registers across all tokens and is read/written once.
#define IQ36_LINEAR_HEAD_DIM 128U
#define IQ36_LINEAR_KEY_HEADS 16U
#define IQ36_LINEAR_VALUE_HEADS 32U
#define IQ36_LINEAR_TOKEN_COUNT 1024U
#define IQ36_LINEAR_LANES_PER_COLUMN 8U
#define IQ36_LINEAR_COLUMNS_PER_GROUP 16U
#define IQ36_LINEAR_COLUMN_GROUPS 8U
#define IQ36_LINEAR_F16_COLUMNS_PER_GROUP 8U
#define IQ36_LINEAR_F16_COLUMN_GROUPS 16U
#define IQ36_LINEAR_CHUNK_SIZE 64U
#define IQ36_LINEAR_CHUNK_COUNT 16U

inline float iq36_reduce_eight_lanes(float value) {
  value += sub_group_shuffle_xor(value, 4U);
  value += sub_group_shuffle_xor(value, 2U);
  value += sub_group_shuffle_xor(value, 1U);
  return value;
}

inline float iq36_reduce_sixteen_lanes(float value) {
  value += sub_group_shuffle_xor(value, 8U);
  value += sub_group_shuffle_xor(value, 4U);
  value += sub_group_shuffle_xor(value, 2U);
  value += sub_group_shuffle_xor(value, 1U);
  return value;
}

__attribute__((reqd_work_group_size(32, 1, 1)))
__attribute__((intel_reqd_sub_group_size(32)))
__kernel void iq36_linear_prefill_recurrent_f32(
    __global const float* q,
    __global const float* k,
    __global const float* v,
    __global const float* gate,
    __global const float* beta,
    __global const float* state_in,
    __global float* attention_out,
    __global float* state_out) {
  const uint physical_group = (uint)get_group_id(0);
  const uint value_head = physical_group / IQ36_LINEAR_COLUMN_GROUPS;
  const uint column_group =
      physical_group - value_head * IQ36_LINEAR_COLUMN_GROUPS;
  const uint lid = (uint)get_local_id(0);
  const uint lane = lid & (IQ36_LINEAR_LANES_PER_COLUMN - 1U);
  const uint lane_group = lid / IQ36_LINEAR_LANES_PER_COLUMN;
  const uint query_head = value_head % IQ36_LINEAR_KEY_HEADS;
  const uint state_head_base =
      value_head * IQ36_LINEAR_HEAD_DIM * IQ36_LINEAR_HEAD_DIM;
  const uint column_base = column_group * IQ36_LINEAR_COLUMNS_PER_GROUP;

  float state_shard[4][16];
  #pragma unroll
  for (uint column_slot = 0; column_slot < 4U; ++column_slot) {
    const uint column = column_base + column_slot * 4U + lane_group;
    #pragma unroll
    for (uint row_slot = 0; row_slot < 16U; ++row_slot) {
      const uint row = row_slot * IQ36_LINEAR_LANES_PER_COLUMN + lane;
      state_shard[column_slot][row_slot] =
          state_in[state_head_base + column * IQ36_LINEAR_HEAD_DIM + row];
    }
  }

  const float attention_scale = 1.0f / sqrt(128.0f);
  for (uint token = 0; token < IQ36_LINEAR_TOKEN_COUNT; ++token) {
    const uint qk_base =
        (token * IQ36_LINEAR_KEY_HEADS + query_head) *
        IQ36_LINEAR_HEAD_DIM;
    const uint v_base =
        (token * IQ36_LINEAR_VALUE_HEADS + value_head) *
        IQ36_LINEAR_HEAD_DIM;
    const uint control = token * IQ36_LINEAR_VALUE_HEADS + value_head;
    const float decay_value = exp(gate[control]);
    const float beta_value = beta[control];

    float q_registers[16];
    float k_registers[16];
    #pragma unroll
    for (uint row_slot = 0; row_slot < 16U; ++row_slot) {
      const uint row = row_slot * IQ36_LINEAR_LANES_PER_COLUMN + lane;
      q_registers[row_slot] = q[qk_base + row];
      k_registers[row_slot] = k[qk_base + row];
    }

    #pragma unroll
    for (uint column_slot = 0; column_slot < 4U; ++column_slot) {
      const uint column = column_base + column_slot * 4U + lane_group;
      float state_dot_k = 0.0f;
      #pragma unroll
      for (uint row_slot = 0; row_slot < 16U; ++row_slot) {
        state_dot_k = fma(state_shard[column_slot][row_slot],
                          k_registers[row_slot], state_dot_k);
      }
      state_dot_k = iq36_reduce_eight_lanes(state_dot_k) * decay_value;
      const float delta =
          (v[v_base + column] - state_dot_k) * beta_value;

      float state_dot_q = 0.0f;
      #pragma unroll
      for (uint row_slot = 0; row_slot < 16U; ++row_slot) {
        const float updated =
            fma(k_registers[row_slot], delta,
                state_shard[column_slot][row_slot] * decay_value);
        state_shard[column_slot][row_slot] = updated;
        state_dot_q = fma(updated, q_registers[row_slot], state_dot_q);
      }
      state_dot_q = iq36_reduce_eight_lanes(state_dot_q);
      if (lane == 0U) {
        attention_out[v_base + column] = state_dot_q * attention_scale;
      }
    }
  }

  #pragma unroll
  for (uint column_slot = 0; column_slot < 4U; ++column_slot) {
    const uint column = column_base + column_slot * 4U + lane_group;
    #pragma unroll
    for (uint row_slot = 0; row_slot < 16U; ++row_slot) {
      const uint row = row_slot * IQ36_LINEAR_LANES_PER_COLUMN + lane;
      state_out[state_head_base + column * IQ36_LINEAR_HEAD_DIM + row] =
          state_shard[column_slot][row_slot];
    }
  }
}

// One measured OpenVINO-informed resource-parity design: SIMD16, half storage,
// and eight columns per workgroup.  Each lane retains 64 half state values;
// there are no workgroup-shape variants to sweep.
__attribute__((reqd_work_group_size(16, 1, 1)))
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void iq36_linear_prefill_recurrent_f16(
    __global const half* q,
    __global const half* k,
    __global const half* v,
    __global const half* gate,
    __global const half* beta,
    __global const half* state_in,
    __global half* attention_out,
    __global half* state_out) {
  const uint physical_group = (uint)get_group_id(0);
  const uint value_head = physical_group / IQ36_LINEAR_F16_COLUMN_GROUPS;
  const uint column_group =
      physical_group - value_head * IQ36_LINEAR_F16_COLUMN_GROUPS;
  const uint lane = (uint)get_local_id(0);
  const uint query_head = value_head % IQ36_LINEAR_KEY_HEADS;
  const uint state_head_base =
      value_head * IQ36_LINEAR_HEAD_DIM * IQ36_LINEAR_HEAD_DIM;
  const uint column_base =
      column_group * IQ36_LINEAR_F16_COLUMNS_PER_GROUP;

  half state_shard[8][8];
  #pragma unroll
  for (uint column_slot = 0; column_slot < 8U; ++column_slot) {
    const uint column = column_base + column_slot;
    #pragma unroll
    for (uint row_slot = 0; row_slot < 8U; ++row_slot) {
      const uint row = row_slot * 16U + lane;
      state_shard[column_slot][row_slot] =
          state_in[state_head_base + column * IQ36_LINEAR_HEAD_DIM + row];
    }
  }

  const float attention_scale = 1.0f / sqrt(128.0f);
  for (uint token = 0; token < IQ36_LINEAR_TOKEN_COUNT; ++token) {
    const uint qk_base =
        (token * IQ36_LINEAR_KEY_HEADS + query_head) *
        IQ36_LINEAR_HEAD_DIM;
    const uint v_base =
        (token * IQ36_LINEAR_VALUE_HEADS + value_head) *
        IQ36_LINEAR_HEAD_DIM;
    const uint control = token * IQ36_LINEAR_VALUE_HEADS + value_head;
    const float decay_value = exp(convert_float(gate[control]));
    const float beta_value = convert_float(beta[control]);

    half q_registers[8];
    half k_registers[8];
    #pragma unroll
    for (uint row_slot = 0; row_slot < 8U; ++row_slot) {
      const uint row = row_slot * 16U + lane;
      q_registers[row_slot] = q[qk_base + row];
      k_registers[row_slot] = k[qk_base + row];
    }

    #pragma unroll
    for (uint column_slot = 0; column_slot < 8U; ++column_slot) {
      const uint column = column_base + column_slot;
      float state_dot_k = 0.0f;
      #pragma unroll
      for (uint row_slot = 0; row_slot < 8U; ++row_slot) {
        state_dot_k = fma(convert_float(state_shard[column_slot][row_slot]),
                          convert_float(k_registers[row_slot]), state_dot_k);
      }
      state_dot_k =
          iq36_reduce_sixteen_lanes(state_dot_k) * decay_value;
      const float delta =
          (convert_float(v[v_base + column]) - state_dot_k) * beta_value;

      float state_dot_q = 0.0f;
      #pragma unroll
      for (uint row_slot = 0; row_slot < 8U; ++row_slot) {
        const float updated =
            fma(convert_float(k_registers[row_slot]), delta,
                convert_float(state_shard[column_slot][row_slot]) *
                    decay_value);
        const half stored = convert_half_rte(updated);
        state_shard[column_slot][row_slot] = stored;
        state_dot_q = fma(convert_float(stored),
                          convert_float(q_registers[row_slot]), state_dot_q);
      }
      state_dot_q = iq36_reduce_sixteen_lanes(state_dot_q);
      if (lane == 0U) {
        attention_out[v_base + column] =
            convert_half_rte(state_dot_q * attention_scale);
      }
    }
  }

  #pragma unroll
  for (uint column_slot = 0; column_slot < 8U; ++column_slot) {
    const uint column = column_base + column_slot;
    #pragma unroll
    for (uint row_slot = 0; row_slot < 8U; ++row_slot) {
      const uint row = row_slot * 16U + lane;
      state_out[state_head_base + column * IQ36_LINEAR_HEAD_DIM + row] =
          state_shard[column_slot][row_slot];
    }
  }
}

// ADR 0043's single fixed chunk-64 WY design.  Cumulative gate, triangular
// inverse, W, and U are constructed entirely on device.  F32 is retained for
// the real-boundary accuracy gate; no chunk-size or precision variants exist.
__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void iq36_linear_prefill_chunk64_prepare_f32(
    __global const float* k,
    __global const float* v,
    __global const float* gate,
    __global const float* beta,
    __global float* cumulative_gate,
    __global float* w,
    __global float* u) {
  const uint chunk_head = (uint)get_group_id(0);
  const uint chunk = chunk_head / IQ36_LINEAR_VALUE_HEADS;
  const uint value_head =
      chunk_head - chunk * IQ36_LINEAR_VALUE_HEADS;
  const uint query_head = value_head % IQ36_LINEAR_KEY_HEADS;
  const uint lid = (uint)get_local_id(0);
  const uint token_base = chunk * IQ36_LINEAR_CHUNK_SIZE;
  __local float inverse[IQ36_LINEAR_CHUNK_SIZE * IQ36_LINEAR_CHUNK_SIZE];
  __local float saved_row[IQ36_LINEAR_CHUNK_SIZE];
  __local float local_gate[IQ36_LINEAR_CHUNK_SIZE];
  __local float local_exp_gate[IQ36_LINEAR_CHUNK_SIZE];

  if (lid == 0U) {
    float total = 0.0f;
    for (uint token_slot = 0; token_slot < IQ36_LINEAR_CHUNK_SIZE;
         ++token_slot) {
      const uint token = token_base + token_slot;
      total += gate[token * IQ36_LINEAR_VALUE_HEADS + value_head];
      local_gate[token_slot] = total;
      local_exp_gate[token_slot] = exp(total);
      cumulative_gate[token * IQ36_LINEAR_VALUE_HEADS + value_head] = total;
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

  const uint row_token = token_base + lid;
  const uint row_k_base =
      (row_token * IQ36_LINEAR_KEY_HEADS + query_head) *
      IQ36_LINEAR_HEAD_DIM;
  const float row_beta =
      beta[row_token * IQ36_LINEAR_VALUE_HEADS + value_head];
  for (uint column_token_slot = 0;
       column_token_slot < IQ36_LINEAR_CHUNK_SIZE; ++column_token_slot) {
    float value = 0.0f;
    if (lid > column_token_slot) {
      const uint column_token = token_base + column_token_slot;
      const uint column_k_base =
          (column_token * IQ36_LINEAR_KEY_HEADS + query_head) *
          IQ36_LINEAR_HEAD_DIM;
      float dot = 0.0f;
      #pragma unroll 4
      for (uint row = 0; row < IQ36_LINEAR_HEAD_DIM; ++row) {
        dot = fma(k[row_k_base + row], k[column_k_base + row], dot);
      }
      value = -row_beta * dot *
          exp(local_gate[lid] - local_gate[column_token_slot]);
    }
    inverse[lid * IQ36_LINEAR_CHUNK_SIZE + column_token_slot] = value;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  for (uint solve_row = 1U; solve_row < IQ36_LINEAR_CHUNK_SIZE;
       ++solve_row) {
    if (lid < solve_row) {
      saved_row[lid] =
          inverse[solve_row * IQ36_LINEAR_CHUNK_SIZE + lid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid < solve_row) {
      float solved = saved_row[lid];
      for (uint inner = 0U; inner < solve_row; ++inner) {
        solved = fma(saved_row[inner],
                     inverse[inner * IQ36_LINEAR_CHUNK_SIZE + lid], solved);
      }
      inverse[solve_row * IQ36_LINEAR_CHUNK_SIZE + lid] = solved;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  inverse[lid * IQ36_LINEAR_CHUNK_SIZE + lid] = 1.0f;
  barrier(CLK_LOCAL_MEM_FENCE);

  const uint output_token = token_base + lid;
  const uint output_k_base =
      (output_token * IQ36_LINEAR_VALUE_HEADS + value_head) *
      IQ36_LINEAR_HEAD_DIM;
  for (uint dimension = 0; dimension < IQ36_LINEAR_HEAD_DIM; ++dimension) {
    float w_value = 0.0f;
    float u_value = 0.0f;
    for (uint source_slot = 0; source_slot <= lid; ++source_slot) {
      const uint source_token = token_base + source_slot;
      const float coefficient =
          inverse[lid * IQ36_LINEAR_CHUNK_SIZE + source_slot];
      const float source_beta =
          beta[source_token * IQ36_LINEAR_VALUE_HEADS + value_head];
      const uint source_k_base =
          (source_token * IQ36_LINEAR_KEY_HEADS + query_head) *
          IQ36_LINEAR_HEAD_DIM;
      const uint source_v_base =
          (source_token * IQ36_LINEAR_VALUE_HEADS + value_head) *
          IQ36_LINEAR_HEAD_DIM;
      w_value = fma(coefficient,
                    k[source_k_base + dimension] * source_beta *
                        local_exp_gate[source_slot],
                    w_value);
      u_value = fma(coefficient,
                    v[source_v_base + dimension] * source_beta, u_value);
    }
    w[output_k_base + dimension] = w_value;
    u[output_k_base + dimension] = u_value;
  }
}

__attribute__((reqd_work_group_size(32, 1, 1)))
__attribute__((intel_reqd_sub_group_size(32)))
__kernel void iq36_linear_prefill_chunk64_scan_f32(
    __global const float* k,
    __global const float* cumulative_gate,
    __global const float* w,
    __global const float* u,
    __global const float* state_in,
    __global float* v_new,
    __global float* chunk_state,
    __global float* state_out) {
  const uint physical_group = (uint)get_group_id(0);
  const uint value_head = physical_group / IQ36_LINEAR_COLUMN_GROUPS;
  const uint column_group =
      physical_group - value_head * IQ36_LINEAR_COLUMN_GROUPS;
  const uint lid = (uint)get_local_id(0);
  const uint lane = lid & (IQ36_LINEAR_LANES_PER_COLUMN - 1U);
  const uint lane_group = lid / IQ36_LINEAR_LANES_PER_COLUMN;
  const uint query_head = value_head % IQ36_LINEAR_KEY_HEADS;
  const uint state_head_base =
      value_head * IQ36_LINEAR_HEAD_DIM * IQ36_LINEAR_HEAD_DIM;
  const uint column_base = column_group * IQ36_LINEAR_COLUMNS_PER_GROUP;

  float state_shard[4][16];
  #pragma unroll
  for (uint column_slot = 0; column_slot < 4U; ++column_slot) {
    const uint column = column_base + column_slot * 4U + lane_group;
    #pragma unroll
    for (uint row_slot = 0; row_slot < 16U; ++row_slot) {
      const uint row = row_slot * IQ36_LINEAR_LANES_PER_COLUMN + lane;
      state_shard[column_slot][row_slot] =
          state_in[state_head_base + column * IQ36_LINEAR_HEAD_DIM + row];
    }
  }

  for (uint chunk = 0; chunk < IQ36_LINEAR_CHUNK_COUNT; ++chunk) {
    const uint token_base = chunk * IQ36_LINEAR_CHUNK_SIZE;
    const uint chunk_head_base =
        (chunk * IQ36_LINEAR_VALUE_HEADS + value_head) *
        IQ36_LINEAR_HEAD_DIM * IQ36_LINEAR_HEAD_DIM;
    #pragma unroll
    for (uint column_slot = 0; column_slot < 4U; ++column_slot) {
      const uint column = column_base + column_slot * 4U + lane_group;
      #pragma unroll
      for (uint row_slot = 0; row_slot < 16U; ++row_slot) {
        const uint row = row_slot * IQ36_LINEAR_LANES_PER_COLUMN + lane;
        chunk_state[chunk_head_base +
                    column * IQ36_LINEAR_HEAD_DIM + row] =
            state_shard[column_slot][row_slot];
      }
    }

    for (uint token_slot = 0; token_slot < IQ36_LINEAR_CHUNK_SIZE;
         ++token_slot) {
      const uint token = token_base + token_slot;
      const uint wu_base =
          (token * IQ36_LINEAR_VALUE_HEADS + value_head) *
          IQ36_LINEAR_HEAD_DIM;
      float w_registers[16];
      #pragma unroll
      for (uint row_slot = 0; row_slot < 16U; ++row_slot) {
        const uint row = row_slot * IQ36_LINEAR_LANES_PER_COLUMN + lane;
        w_registers[row_slot] = w[wu_base + row];
      }
      #pragma unroll
      for (uint column_slot = 0; column_slot < 4U; ++column_slot) {
        const uint column = column_base + column_slot * 4U + lane_group;
        float projection = 0.0f;
        #pragma unroll
        for (uint row_slot = 0; row_slot < 16U; ++row_slot) {
          projection = fma(state_shard[column_slot][row_slot],
                           w_registers[row_slot], projection);
        }
        projection = iq36_reduce_eight_lanes(projection);
        if (lane == 0U) {
          v_new[wu_base + column] = u[wu_base + column] - projection;
        }
      }
    }
    barrier(CLK_GLOBAL_MEM_FENCE);

    const uint last_token = token_base + IQ36_LINEAR_CHUNK_SIZE - 1U;
    const float last_gate =
        cumulative_gate[last_token * IQ36_LINEAR_VALUE_HEADS + value_head];
    const float chunk_decay = exp(last_gate);
    #pragma unroll
    for (uint column_slot = 0; column_slot < 4U; ++column_slot) {
      #pragma unroll
      for (uint row_slot = 0; row_slot < 16U; ++row_slot) {
        state_shard[column_slot][row_slot] *= chunk_decay;
      }
    }
    for (uint token_slot = 0; token_slot < IQ36_LINEAR_CHUNK_SIZE;
         ++token_slot) {
      const uint token = token_base + token_slot;
      const uint k_base =
          (token * IQ36_LINEAR_KEY_HEADS + query_head) *
          IQ36_LINEAR_HEAD_DIM;
      const uint value_base =
          (token * IQ36_LINEAR_VALUE_HEADS + value_head) *
          IQ36_LINEAR_HEAD_DIM;
      const float gate_scale = exp(
          last_gate - cumulative_gate[
              token * IQ36_LINEAR_VALUE_HEADS + value_head]);
      float k_registers[16];
      #pragma unroll
      for (uint row_slot = 0; row_slot < 16U; ++row_slot) {
        const uint row = row_slot * IQ36_LINEAR_LANES_PER_COLUMN + lane;
        k_registers[row_slot] = k[k_base + row] * gate_scale;
      }
      #pragma unroll
      for (uint column_slot = 0; column_slot < 4U; ++column_slot) {
        const uint column = column_base + column_slot * 4U + lane_group;
        const float update = v_new[value_base + column];
        #pragma unroll
        for (uint row_slot = 0; row_slot < 16U; ++row_slot) {
          state_shard[column_slot][row_slot] =
              fma(k_registers[row_slot], update,
                  state_shard[column_slot][row_slot]);
        }
      }
    }
  }

  #pragma unroll
  for (uint column_slot = 0; column_slot < 4U; ++column_slot) {
    const uint column = column_base + column_slot * 4U + lane_group;
    #pragma unroll
    for (uint row_slot = 0; row_slot < 16U; ++row_slot) {
      const uint row = row_slot * IQ36_LINEAR_LANES_PER_COLUMN + lane;
      state_out[state_head_base + column * IQ36_LINEAR_HEAD_DIM + row] =
          state_shard[column_slot][row_slot];
    }
  }
}

__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void iq36_linear_prefill_chunk64_output_f32(
    __global const float* q,
    __global const float* k,
    __global const float* cumulative_gate,
    __global const float* v_new,
    __global const float* chunk_state,
    __global float* attention_out) {
  const uint chunk_head = (uint)get_group_id(0);
  const uint chunk = chunk_head / IQ36_LINEAR_VALUE_HEADS;
  const uint value_head =
      chunk_head - chunk * IQ36_LINEAR_VALUE_HEADS;
  const uint query_head = value_head % IQ36_LINEAR_KEY_HEADS;
  const uint lid = (uint)get_local_id(0);
  const uint token_base = chunk * IQ36_LINEAR_CHUNK_SIZE;
  const uint token = token_base + lid;
  const uint q_base =
      (token * IQ36_LINEAR_KEY_HEADS + query_head) *
      IQ36_LINEAR_HEAD_DIM;
  const float token_gate =
      cumulative_gate[token * IQ36_LINEAR_VALUE_HEADS + value_head];
  __local float score[IQ36_LINEAR_CHUNK_SIZE * IQ36_LINEAR_CHUNK_SIZE];

  for (uint source_slot = 0; source_slot < IQ36_LINEAR_CHUNK_SIZE;
       ++source_slot) {
    float value = 0.0f;
    if (source_slot <= lid) {
      const uint source_token = token_base + source_slot;
      const uint k_base =
          (source_token * IQ36_LINEAR_KEY_HEADS + query_head) *
          IQ36_LINEAR_HEAD_DIM;
      float dot = 0.0f;
      #pragma unroll 4
      for (uint row = 0; row < IQ36_LINEAR_HEAD_DIM; ++row) {
        dot = fma(q[q_base + row], k[k_base + row], dot);
      }
      const float source_gate = cumulative_gate[
          source_token * IQ36_LINEAR_VALUE_HEADS + value_head];
      value = dot * exp(token_gate - source_gate);
    }
    score[lid * IQ36_LINEAR_CHUNK_SIZE + source_slot] = value;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  const uint chunk_head_base =
      (chunk * IQ36_LINEAR_VALUE_HEADS + value_head) *
      IQ36_LINEAR_HEAD_DIM * IQ36_LINEAR_HEAD_DIM;
  const uint output_base =
      (token * IQ36_LINEAR_VALUE_HEADS + value_head) *
      IQ36_LINEAR_HEAD_DIM;
  const float initial_scale = exp(token_gate);
  const float attention_scale = 1.0f / sqrt(128.0f);
  for (uint column = 0; column < IQ36_LINEAR_HEAD_DIM; ++column) {
    float initial = 0.0f;
    const uint state_column =
        chunk_head_base + column * IQ36_LINEAR_HEAD_DIM;
    #pragma unroll 4
    for (uint row = 0; row < IQ36_LINEAR_HEAD_DIM; ++row) {
      initial = fma(q[q_base + row], chunk_state[state_column + row],
                    initial);
    }
    float local_contribution = 0.0f;
    for (uint source_slot = 0; source_slot <= lid; ++source_slot) {
      const uint source_token = token_base + source_slot;
      const uint source_value_base =
          (source_token * IQ36_LINEAR_VALUE_HEADS + value_head) *
          IQ36_LINEAR_HEAD_DIM;
      local_contribution = fma(
          score[lid * IQ36_LINEAR_CHUNK_SIZE + source_slot],
          v_new[source_value_base + column], local_contribution);
    }
    attention_out[output_base + column] =
        (initial * initial_scale + local_contribution) * attention_scale;
  }
}

__attribute__((reqd_work_group_size(32, 1, 1)))
__attribute__((intel_reqd_sub_group_size(32)))
__kernel void iq36_linear_prefill_norm_gate_f32(
    __global const float* attention,
    __global const float* z,
    __global const float* norm_weight,
    float epsilon,
    __global float* final_out) {
  const uint token_head = (uint)get_group_id(0);
  const uint lid = (uint)get_local_id(0);
  const uint base = token_head * IQ36_LINEAR_HEAD_DIM;
  float sum_squares = 0.0f;
  #pragma unroll
  for (uint slot = 0; slot < 4U; ++slot) {
    const float value = attention[base + lid + slot * 32U];
    sum_squares = fma(value, value, sum_squares);
  }
  sum_squares = sub_group_reduce_add(sum_squares);
  const float norm_scale =
      rsqrt(sum_squares / (float)IQ36_LINEAR_HEAD_DIM + epsilon);
  #pragma unroll
  for (uint slot = 0; slot < 4U; ++slot) {
    const uint row = lid + slot * 32U;
    const float z_value = z[base + row];
    const float z_silu = z_value / (1.0f + exp(-z_value));
    final_out[base + row] =
        attention[base + row] * norm_scale * norm_weight[row] * z_silu;
  }
}

__attribute__((reqd_work_group_size(16, 1, 1)))
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void iq36_linear_prefill_norm_gate_f16(
    __global const half* attention,
    __global const half* z,
    __global const half* norm_weight,
    float epsilon,
    __global float* final_out) {
  const uint token_head = (uint)get_group_id(0);
  const uint lid = (uint)get_local_id(0);
  const uint base = token_head * IQ36_LINEAR_HEAD_DIM;
  float sum_squares = 0.0f;
  #pragma unroll
  for (uint slot = 0; slot < 8U; ++slot) {
    const float value = convert_float(attention[base + lid + slot * 16U]);
    sum_squares = fma(value, value, sum_squares);
  }
  sum_squares = sub_group_reduce_add(sum_squares);
  const float norm_scale =
      rsqrt(sum_squares / (float)IQ36_LINEAR_HEAD_DIM + epsilon);
  #pragma unroll
  for (uint slot = 0; slot < 8U; ++slot) {
    const uint row = lid + slot * 16U;
    const float z_value = convert_float(z[base + row]);
    const float z_silu = z_value / (1.0f + exp(-z_value));
    final_out[base + row] =
        convert_float(attention[base + row]) * norm_scale *
        convert_float(norm_weight[row]) * z_silu;
  }
}
