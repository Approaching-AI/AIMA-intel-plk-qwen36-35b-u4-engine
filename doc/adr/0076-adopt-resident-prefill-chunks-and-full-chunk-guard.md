# ADR 0076: Adopt resident prefill chunks and a full-chunk hot guard

Date: 2026-07-14

## Status

Accepted

## Context

The all-ten carrier passed exact 2k/8k semantics, but the first direct 16k
correctness attempt sent the entire prompt through one language-model inference
call. Both the untouched stock graph and the all-ten candidate caused global
OOM on the 64-GiB target. The stock process was killed with about 7.8 GiB
anonymous RSS; the candidate was killed independently. The product OpenVINO
denominator already reaches longer contexts through resident prompt chunking,
so monolithic debug execution was not a valid product requirement.

The previous physical ring had only enough guard capacity for a one-token
decode update. A multi-token prefill must read the prior hot window while owner
work-groups publish the continuation. With no device-wide work-group barrier,
those writes must not overlap any prior row that another work-group can still
read.

Commit `dd55eef` fixes both issues. Each long prompt is processed in resident
8k chunks in one InferRequest with continuous positions and masks. The physical
ring becomes logical hot8192 plus an 8192-row write guard, with the exact sink
in slot zero. Prefill reads prior hot/cold state, writes the new chunk only into
non-overlapping capacity, and appends evicted rows to signed block32 I8 cold
state. Correctly rounded FP32 division fixes the last I8 tie-rounding ambiguity.

Clean seq864 crosses 16k with two chunks and one teacher-forced decode. Both KLD
rows pass at `0.001589733457/0.000308780631`; greedy tokens are exactly
`[271,248068]`; every full-attention layer advances cold length
`8192 -> 8193`; and all hot preservation, I8 payload, F16 scale-byte, sentinel,
and untouched-state checks pass. Clean seq865 reconfirms all-ten 2k/8k with all
five KLD rows at or below `0.002267511`.

Evidence:

- `output/openvino-hot-cold-attention-20260714Tseq864-allten-16k-transition-cleanZ/`
- `output/openvino-hot-cold-attention-20260714Tseq865-allten-2k8k-ring16k-cleanZ/`
- `output/openvino-hot-cold-attention-20260714Tseq857d-allten-16k-cleanZ/`
- `output/openvino-hot-cold-attention-20260714Tseq858-allten-16k-chunked-stock-dirtyZ/`

## Decision

1. Interpret every accepted context bucket as a complete logical prompt; do not
   require one monolithic inference call.
2. Permit a frozen resident prefill-chunk schedule in one request. Record chunk
   size/count and continuous positions/masks in correctness and performance
   manifests.
3. Use an 8192-token maximum continuation chunk for the active carrier. Keep
   one exact sink plus a 16384-row ring: logical hot8192 and an 8192-row
   non-overlap guard. The resulting V state has 16385 rows; packed K is
   `[1,2,1025,2048]`.
4. Preserve append-only signed block32-I8 cold state and exact F16 scale bytes.
   Any different chunk size, ring capacity, or codec reopens the full semantic
   ladder.
5. This supersedes ADR 0074 only for physical guard capacity. It does not alter
   the logical hot window, sink semantics, correctness thresholds, product
   buckets, or performance target, and it is not a speed claim.

## Consequences

- Direct monolithic 16k correctness workers are closed; use the resident
  chunked protocol.
- Long-context TTFT/prefill timing must include every chunk and exclude
  correctness-capture copies. Peak-memory/no-OOM evidence spans the entire
  request lifetime.
- Guard capacity is bounded and independent of total context. Older history
  still grows only in compressed append-only form.
- The remaining OV2 blocker is exact prefill/decode program separation and
  decode performance, not 16k state semantics.

Current gate state remains in
`doc/active/intel-qwen36-35b-a3b-gguf-q4km/STATUS.md`.
