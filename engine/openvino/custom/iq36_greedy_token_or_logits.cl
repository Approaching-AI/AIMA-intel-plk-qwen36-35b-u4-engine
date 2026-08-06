// Candidate-only phase-safe token boundary for the locked LM head.
//
// Decode-local token providers place two exact base-2048 token digits in the
// first two logits and NaN markers in the next two.  Prefill keeps ordinary
// finite full logits.  One work-group decodes the compact carrier in the hot
// path or falls back to an exact full-vocabulary argmax for the single prefill
// call, so the graph needs only one custom result primitive in either phase.

inline ulong iq36_token_or_logits_key(float value, uint token_id) {
  const uint bits = as_uint(value);
  const uint ordered =
      (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
  return (((ulong)ordered) << 32) | ((ulong)(~token_id));
}

__kernel void iq36_greedy_token_or_logits(
    const __global INPUT0_TYPE* input,
    __global OUTPUT0_TYPE* output) {
  const uint lane = get_local_id(0);
  const uint pitch = INPUT0_PITCHES[3];
  const float marker0 = (float)input[INPUT0_OFFSET + 2u * pitch];
  const float marker1 = (float)input[INPUT0_OFFSET + 3u * pitch];
  const bool encoded = isnan(marker0) && isnan(marker1);
  if (encoded) {
    if (lane == 0u) {
      const uint low = convert_uint_rte(
          (float)input[INPUT0_OFFSET + 0u * pitch]);
      const uint high = convert_uint_rte(
          (float)input[INPUT0_OFFSET + 1u * pitch]);
      output[OUTPUT0_OFFSET] = (OUTPUT0_TYPE)(low + (high << 11u));
    }
    return;
  }

  const uint vocabulary = INPUT0_DIMS[3];
  const uint local_size = get_local_size(0);
  __local ulong keys[256];
  ulong best = 0ul;
  for (uint token = lane; token < vocabulary; token += local_size) {
    const float value = (float)input[INPUT0_OFFSET + token * pitch];
    best = max(best, iq36_token_or_logits_key(value, token));
  }
  keys[lane] = best;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint stride = local_size >> 1; stride != 0; stride >>= 1) {
    if (lane < stride) keys[lane] = max(keys[lane], keys[lane + stride]);
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lane == 0u) {
    output[OUTPUT0_OFFSET] =
        (OUTPUT0_TYPE)(~(uint)(keys[0] & 0xfffffffful));
  }
}
