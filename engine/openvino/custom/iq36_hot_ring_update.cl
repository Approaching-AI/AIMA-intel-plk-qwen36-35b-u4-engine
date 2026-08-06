// Stateful boundary probe for the fixed 8192-token full-attention hot ring.
//
// INPUT0 is a Variable-owned F32 ring.  The kernel intentionally updates that
// storage in place. OUTPUT0 echoes only the update payload, making the mutation
// an observable graph operation without materializing the full ring.

__kernel void iq36_hot_ring_update(
    __global INPUT0_TYPE* ring,
    const __global INPUT1_TYPE* updates,
    const __global INPUT2_TYPE* base_slot,
    __global OUTPUT0_TYPE* dependency) {
  const uint x = get_global_id(0);
  const uint update_y = get_global_id(1);
  const uint bf = get_global_id(2);
  const uint base = (uint)base_slot[INPUT2_OFFSET];
  const uint ring_y = (base + update_y) % (uint)INPUT0_DIMS[2];
  const uint ring_index = INPUT0_OFFSET +
      bf * INPUT0_PITCHES[1] + ring_y * INPUT0_PITCHES[2] +
      x * INPUT0_PITCHES[3];
  const uint update_index = INPUT1_OFFSET +
      bf * INPUT1_PITCHES[1] + update_y * INPUT1_PITCHES[2] +
      x * INPUT1_PITCHES[3];
  ring[ring_index] = (INPUT0_TYPE)updates[update_index];
  const uint output_index = OUTPUT0_OFFSET +
      bf * OUTPUT0_PITCHES[1] + update_y * OUTPUT0_PITCHES[2] +
      x * OUTPUT0_PITCHES[3];
  dependency[output_index] = (OUTPUT0_TYPE)updates[update_index];
}
