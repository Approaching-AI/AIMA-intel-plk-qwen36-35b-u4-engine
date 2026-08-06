// Final reduction for the candidate-only greedy-feedback component.

__kernel void iq36_greedy_top1_merge(
    const __global INPUT0_TYPE* partials,
    __global OUTPUT0_TYPE* token) {
  const uint lane = get_local_id(0);
  __local ulong local_keys[64];
  const uint input_index = INPUT0_OFFSET + lane * INPUT0_PITCHES[3];
  local_keys[lane] = (ulong)partials[input_index];
  barrier(CLK_LOCAL_MEM_FENCE);

  for (uint stride = 32; stride != 0; stride >>= 1) {
    if (lane < stride)
      local_keys[lane] = max(local_keys[lane], local_keys[lane + stride]);
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lane == 0) {
    const uint inverse_id = (uint)(local_keys[0] & 0xfffffffful);
    token[OUTPUT0_OFFSET] = (OUTPUT0_TYPE)(~inverse_id);
  }
}
