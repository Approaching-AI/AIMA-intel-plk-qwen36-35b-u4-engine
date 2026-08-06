// Fused Q/K [B,Q,H,D] -> [B,H,Q,D] layout and rotate-half RoPE.
// Qwen3.6 rotates the first 64 of 256 dimensions and copies the tail.

inline half iq36_qk_rope_value(
    const half value, const half peer, const half cosine,
    const half sine, const bool first_half) {
  return first_half
      ? cosine * value - sine * peer
      : cosine * value + sine * peer;
}

__attribute__((reqd_work_group_size(16, 1, 1)))
__kernel void iq36_qk_rope_layout(
    const __global INPUT0_TYPE* query,
    const __global INPUT1_TYPE* key,
    const __global INPUT2_TYPE* cosine,
    const __global INPUT3_TYPE* sine,
    __global OUTPUT0_TYPE* query_output,
    __global OUTPUT1_TYPE* key_output) {
  const uint dimension = (uint)get_global_id(0);
  const uint token = (uint)get_global_id(1);
  const uint batch_head = (uint)get_global_id(2);
  const uint query_head = batch_head % (uint)OUTPUT0_DIMS[1];
  const uint batch = batch_head / (uint)OUTPUT0_DIMS[1];

  const ulong query_input_index = INPUT0_OFFSET +
      (ulong)batch * INPUT0_PITCHES[0] +
      (ulong)token * INPUT0_PITCHES[1] +
      (ulong)query_head * INPUT0_PITCHES[2] +
      (ulong)dimension * INPUT0_PITCHES[3];
  const ulong query_output_index = OUTPUT0_OFFSET +
      (ulong)batch * OUTPUT0_PITCHES[0] +
      (ulong)query_head * OUTPUT0_PITCHES[1] +
      (ulong)token * OUTPUT0_PITCHES[2] +
      (ulong)dimension * OUTPUT0_PITCHES[3];

  half query_value = convert_half_rte(query[query_input_index]);
  if (dimension < 64U) {
    const bool first_half = dimension < 32U;
    const uint peer_dimension = first_half
        ? dimension + 32U : dimension - 32U;
    const ulong query_peer_index = INPUT0_OFFSET +
        (ulong)batch * INPUT0_PITCHES[0] +
        (ulong)token * INPUT0_PITCHES[1] +
        (ulong)query_head * INPUT0_PITCHES[2] +
        (ulong)peer_dimension * INPUT0_PITCHES[3];
    const ulong table_index = INPUT2_OFFSET +
        (ulong)batch * INPUT2_PITCHES[0] +
        (ulong)token * INPUT2_PITCHES[2] +
        (ulong)dimension * INPUT2_PITCHES[3];
    const ulong sine_index = INPUT3_OFFSET +
        (ulong)batch * INPUT3_PITCHES[0] +
        (ulong)token * INPUT3_PITCHES[2] +
        (ulong)dimension * INPUT3_PITCHES[3];
    query_value = iq36_qk_rope_value(
        query_value, convert_half_rte(query[query_peer_index]),
        convert_half_rte(cosine[table_index]),
        convert_half_rte(sine[sine_index]),
        first_half);
  }
  query_output[query_output_index] = (OUTPUT0_TYPE)query_value;

  if (query_head < (uint)OUTPUT1_DIMS[1]) {
    const uint key_head = query_head;
    const ulong key_input_index = INPUT1_OFFSET +
        (ulong)batch * INPUT1_PITCHES[0] +
        (ulong)token * INPUT1_PITCHES[1] +
        (ulong)key_head * INPUT1_PITCHES[2] +
        (ulong)dimension * INPUT1_PITCHES[3];
    const ulong key_output_index = OUTPUT1_OFFSET +
        (ulong)batch * OUTPUT1_PITCHES[0] +
        (ulong)key_head * OUTPUT1_PITCHES[1] +
        (ulong)token * OUTPUT1_PITCHES[2] +
        (ulong)dimension * OUTPUT1_PITCHES[3];
    half key_value = convert_half_rte(key[key_input_index]);
    if (dimension < 64U) {
      const bool first_half = dimension < 32U;
      const uint peer_dimension = first_half
          ? dimension + 32U : dimension - 32U;
      const ulong key_peer_index = INPUT1_OFFSET +
          (ulong)batch * INPUT1_PITCHES[0] +
          (ulong)token * INPUT1_PITCHES[1] +
          (ulong)key_head * INPUT1_PITCHES[2] +
          (ulong)peer_dimension * INPUT1_PITCHES[3];
      const ulong table_index = INPUT2_OFFSET +
          (ulong)batch * INPUT2_PITCHES[0] +
          (ulong)token * INPUT2_PITCHES[2] +
          (ulong)dimension * INPUT2_PITCHES[3];
      const ulong sine_index = INPUT3_OFFSET +
          (ulong)batch * INPUT3_PITCHES[0] +
          (ulong)token * INPUT3_PITCHES[2] +
          (ulong)dimension * INPUT3_PITCHES[3];
      key_value = iq36_qk_rope_value(
          key_value, convert_half_rte(key[key_peer_index]),
          convert_half_rte(cosine[table_index]),
          convert_half_rte(sine[sine_index]), first_half);
    }
    key_output[key_output_index] = (OUTPUT1_TYPE)key_value;
  }
}
