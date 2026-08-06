// Capability probe for the pinned OpenVINO GPU SimpleGPU path.  The first
// output copies the input and the second doubles it.  This source is not an
// inference-engine kernel; it locks whether one custom node can expose the
// attention result and persistent state as separate graph outputs.

__kernel void iq36_multi_output_probe(
    const __global INPUT0_TYPE* input,
    const __global INPUT1_TYPE* unused,
    __global OUTPUT0_TYPE* copied,
    __global OUTPUT1_TYPE* doubled) {
  const int index = get_global_id(0);
  const INPUT0_TYPE value = input[index];
  copied[index] = (OUTPUT0_TYPE)(value);
  doubled[index] = (OUTPUT1_TYPE)(value + value);
}
