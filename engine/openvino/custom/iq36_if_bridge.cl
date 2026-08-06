// Explicit GPU copy boundary before a host-controlled OpenVINO If.
// The bridge output may be made lockable for the condition primitive without
// forcing the shared upstream producer (also consumed by custom decode) into
// host USM.

__kernel void iq36_if_bridge(
    const __global INPUT0_TYPE* input,
    __global OUTPUT0_TYPE* output) {
  const uint x = (uint)get_global_id(0);
  const uint y = (uint)get_global_id(1);
  const uint batch_feature = (uint)get_global_id(2);
  const uint feature = batch_feature % (uint)INPUT0_DIMS[1];
  const uint batch = batch_feature / (uint)INPUT0_DIMS[1];
  const ulong input_index = INPUT0_OFFSET +
      (ulong)batch * INPUT0_PITCHES[0] +
      (ulong)feature * INPUT0_PITCHES[1] +
      (ulong)y * INPUT0_PITCHES[2] +
      (ulong)x * INPUT0_PITCHES[3];
  const ulong output_index = OUTPUT0_OFFSET +
      (ulong)batch * OUTPUT0_PITCHES[0] +
      (ulong)feature * OUTPUT0_PITCHES[1] +
      (ulong)y * OUTPUT0_PITCHES[2] +
      (ulong)x * OUTPUT0_PITCHES[3];
  output[output_index] = (OUTPUT0_TYPE)input[input_index];
}
