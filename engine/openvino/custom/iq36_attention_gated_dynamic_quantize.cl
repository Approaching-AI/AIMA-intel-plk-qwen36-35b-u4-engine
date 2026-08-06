#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable

// Locked Qwen3.6 full-attention output consumer:
//   input0  [B,16,Q,256] head-major attention output
//   input1  [B,Q,4096]   sigmoid output gate
//   output0 [B,Q,4096]   graph-only shape carrier (intentionally unwritten)
//   output1 [B,Q,4096]   group-64 symmetric I8 activation
//   output2 [B,Q,64]     reciprocal F16 activation scales
//   output3 [B,Q,64]     I32 sums used by asymmetric U4 weights
//
// One subgroup owns one 64-element activation group.  Because 64 divides the
// 256-wide attention head, each subgroup reads one contiguous head fragment
// while writing the token-major compressed-FC layout.  Arithmetic mirrors
// dynamic_quantize_gpu_opt.cl MODE_SMALL_GS, including the 0.003h clamp and
// convert_char4_rte rounding contract.

#define IQ36_HEAD_DIM 256U
#define IQ36_GROUP_SIZE 64U
#define IQ36_VALUES_PER_LANE 4U
#define IQ36_ACT_MIN_VALUE 0.003h

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void iq36_attention_gated_dynamic_quantize(
    const __global INPUT0_TYPE* attention,
    const __global INPUT1_TYPE* gate,
    __global OUTPUT0_TYPE* shape_carrier,
    __global OUTPUT1_TYPE* quantized,
    __global OUTPUT2_TYPE* scale,
    __global OUTPUT3_TYPE* precomputed_reduction) {
  const uint lane = get_sub_group_local_id();
  const uint group = get_global_id(1);
  const uint bf = get_global_id(2);
  const uint query_count = OUTPUT0_DIMS[1];
  const uint batch = bf / query_count;
  const uint query = bf % query_count;
  const uint flat = group * IQ36_GROUP_SIZE +
      lane * IQ36_VALUES_PER_LANE;
  const uint head = flat / IQ36_HEAD_DIM;
  const uint dimension = flat % IQ36_HEAD_DIM;

  const ulong attention_offset = INPUT0_OFFSET +
      (ulong)batch * INPUT0_PITCHES[0] +
      (ulong)head * INPUT0_PITCHES[1] +
      (ulong)query * INPUT0_PITCHES[2] +
      (ulong)dimension * INPUT0_PITCHES[3];
  const ulong gate_offset = INPUT1_OFFSET +
      (ulong)batch * INPUT1_PITCHES[0] +
      (ulong)query * INPUT1_PITCHES[1] +
      (ulong)flat * INPUT1_PITCHES[2];

  const half4 attention_value = convert_half4(
      vload4(0, attention + attention_offset));
  const half4 gate_value = convert_half4(
      vload4(0, gate + gate_offset));
  const half4 gated = attention_value * gate_value;
  half max_value = fmax(
      fmax(fabs(gated.s0), fabs(gated.s1)),
      fmax(fabs(gated.s2), fabs(gated.s3)));
  max_value = fmax(
      (half)IQ36_ACT_MIN_VALUE, sub_group_reduce_max(max_value));
  const half quantize_scale = 127.0h / max_value;
  const char4 quantized_value = convert_char4_rte(
      gated * (half4)quantize_scale);

  const ulong quantized_offset = OUTPUT1_OFFSET +
      (ulong)batch * OUTPUT1_PITCHES[0] +
      (ulong)query * OUTPUT1_PITCHES[1] +
      (ulong)flat * OUTPUT1_PITCHES[2];
  vstore4(quantized_value, 0, quantized + quantized_offset);

  int reduction = quantized_value.s0 + quantized_value.s1 +
      quantized_value.s2 + quantized_value.s3;
  reduction = sub_group_reduce_add(reduction);
  if (lane == 0) {
    const ulong scale_offset = OUTPUT2_OFFSET +
        (ulong)batch * OUTPUT2_PITCHES[0] +
        (ulong)query * OUTPUT2_PITCHES[1] +
        (ulong)group * OUTPUT2_PITCHES[2];
    const ulong reduction_offset = OUTPUT3_OFFSET +
        (ulong)batch * OUTPUT3_PITCHES[0] +
        (ulong)query * OUTPUT3_PITCHES[1] +
        (ulong)group * OUTPUT3_PITCHES[2];
    scale[scale_offset] = (OUTPUT2_TYPE)(1.0h / quantize_scale);
    precomputed_reduction[reduction_offset] = (OUTPUT3_TYPE)reduction;
  }

  // The GPU transformation replaces the sole graph consumer of output zero
  // with outputs one through three before SimpleGPU lowering.  Keep the
  // argument in the ABI but do not recreate the eliminated F16/F32 carrier.
  (void)shape_carrier;
}
