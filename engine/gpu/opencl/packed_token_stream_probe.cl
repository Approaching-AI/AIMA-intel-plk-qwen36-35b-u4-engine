// Level Zero command-list carrier probe for the packed whole-token schedule.
// The kernel deliberately performs only a byte-stream reduction. Exact model
// math remains in the accepted Q4/Q6 carriers; this module measures whether the
// target can execute the complete logical command/byte census in one reusable
// native command list without host-stage drains.

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void iq36_packed_token_stream_stage(
    __global const ulong4* payload,
    ulong byte_count,
    __global const ulong* token_control,
    __global ulong* command_checksums,
    ulong checksum_offset,
    uint command_index) {
  const ulong gid = (ulong)get_global_id(0);
  const ulong global_size = (ulong)get_global_size(0);
  const uint lid = (uint)get_local_id(0);
  const ulong word_count = byte_count / 32UL;
  ulong checksum =
      ((ulong)command_index * 0x9e3779b97f4a7c15UL) ^ gid;
  if (lid == 0) {
    checksum ^= token_control[0] ^ (token_control[1] << 1);
  }
  for (ulong word = gid; word < word_count; word += global_size) {
    const ulong4 value = payload[word];
    checksum ^= value.s0 + rotate(value.s1, (ulong)13);
    checksum ^= value.s2 + rotate(value.s3, (ulong)29);
  }
  if (gid == 0UL) {
    __global const uchar* tail =
        (__global const uchar*)(payload + word_count);
    for (ulong index = 0; index < byte_count - word_count * 32UL; ++index) {
      checksum = (checksum << 5) ^ (checksum >> 2) ^ (ulong)tail[index];
    }
  }

  __local ulong reduced[256];
  reduced[lid] = checksum;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint stride = 128; stride > 0; stride >>= 1) {
    if (lid < stride) reduced[lid] ^= reduced[lid + stride];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lid == 0) {
    command_checksums[checksum_offset + (ulong)get_group_id(0)] = reduced[0];
  }
}
