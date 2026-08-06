#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Deliberately simple proof kernel for the candidate OpenVINO GPU path.  It
// preserves every value while proving that CONFIG_FILE selects repository
// OpenCL code for an OpenVINO extension operation.
__kernel void iq36_identity(
    const __global INPUT0_TYPE* input0,
    __global OUTPUT0_TYPE* output0) {
  const uint x = get_global_id(0);
  const uint y = get_global_id(1);
  const uint bf = get_global_id(2);
  const uint feature = bf % OUTPUT0_DIMS[1];
  const uint batch = bf / OUTPUT0_DIMS[1];
  const uint input_index =
      batch * INPUT0_PITCHES[0] + feature * INPUT0_PITCHES[1] +
      y * INPUT0_PITCHES[2] + x * INPUT0_PITCHES[3] + INPUT0_OFFSET;
  const uint output_index =
      batch * OUTPUT0_PITCHES[0] + feature * OUTPUT0_PITCHES[1] +
      y * OUTPUT0_PITCHES[2] + x * OUTPUT0_PITCHES[3] + OUTPUT0_OFFSET;
  output0[output_index] = input0[input_index];
}
