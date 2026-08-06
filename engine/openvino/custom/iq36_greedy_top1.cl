// Candidate-only greedy feedback for the locked 248320-row language head.
//
// This first pass uses enough work-groups to stream the complete F32 logits
// tensor, then writes one packed (value, inverse-id) maximum per group.  A
// separate merge source reduces those 64 records to one token.  The inverse
// id makes equal logits select the lowest row, matching np.argmax.

inline ulong iq36_ordered_top1_key(float value, uint token_id) {
  const uint bits = as_uint(value);
  const uint ordered =
      (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
  return (((ulong)ordered) << 32) | ((ulong)(~token_id));
}

__kernel void iq36_greedy_top1_partials(
    const __global INPUT0_TYPE* logits,
    __global OUTPUT0_TYPE* partials) {
  const uint lane = get_local_id(0);
  const uint group = get_group_id(0);
  const uint local_size = get_local_size(0);
  const uint global_size = get_global_size(0);
  const uint vocabulary = INPUT0_DIMS[3];
  __local ulong local_keys[256];

  ulong best = 0ul;
  for (uint token = group * local_size + lane;
       token < vocabulary;
       token += global_size) {
    const uint input_index = INPUT0_OFFSET + token * INPUT0_PITCHES[3];
    const ulong key = iq36_ordered_top1_key(
        (float)logits[input_index], token);
    best = max(best, key);
  }
  local_keys[lane] = best;
  barrier(CLK_LOCAL_MEM_FENCE);

  for (uint stride = local_size >> 1; stride != 0; stride >>= 1) {
    if (lane < stride)
      local_keys[lane] = max(local_keys[lane], local_keys[lane + stride]);
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lane == 0) {
    const uint output_index = OUTPUT0_OFFSET + group * OUTPUT0_PITCHES[3];
    partials[output_index] = (OUTPUT0_TYPE)local_keys[0];
  }
}
