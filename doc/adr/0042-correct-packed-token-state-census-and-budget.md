# ADR 0042: Correct the packed-token state census and backend budget

Date: 2026-07-12

## Status

Accepted; supersedes ADR 0040's `18.580 ms` kernel allocation and the seq732
strict-byte total. The product wall and throughput target do not change.

## Context

Seq732 counted active weights and 1k full-attention KV reads, but the real-stage
audit found that it did not count the 30 linear layers' recurrent-state and
convolution-state reads/writes. Each linear layer carries 524,288 recurrent
F32 values and 24,576 convolution-state F32 values. Across 30 layers this adds
`131,727,360` bytes/token. KV append writes add another `20,480` bytes.

Clean seq734 derives the complete state census:

- active weights: `1,975,676,544` bytes/token;
- resident state reads: `86,835,200` bytes/token;
- resident state writes: `65,884,160` bytes/token;
- corrected strict total: `2,128,395,904` bytes/token.

ADR 0040 conservatively reserved `1.500 ms` of the `20.080 ms` wall for host
submission. Seq733 measured the selected Level Zero mechanism directly: the
maximum paired wall-minus-device residual was `11.803 us`. A `0.100 ms` host
allocation retains an `8.47x` safety factor and moves the kernel allocation to
`19.980 ms` without changing the full wall.

With the corrected total, the wall-rate stream floor is `105.994 GB/s` and the
kernel-window floor is `106.525 GB/s`. Both remain below the real Q4/Q6 carrier
measurements (`110.522 / 107.579 GB/s`). The hardware-informed carrier model is
`18.748 ms`, leaving `1.233 ms` for fused math; the conservative uniform-rate
model in seq735 is `19.542 ms`, leaving only `0.439 ms`.

Clean seq735 recompiles the schedule with the corrected census. Clean seq736
then streams one disjoint range for every corrected command in a single reused
Level Zero list: repeat/confirm are `8.948 / 8.938 ms` device time at
`237.876 / 238.121 GB/s` proxy rate.

Evidence:

- `output/packed-token-state-budget-gate-20260712Tseq734cleanZ/`
- `output/packed-token-schedule-gate-20260712Tseq735-state-cleanZ/`
- `output/packed-token-level-zero-gate-20260712Tseq736-state-cleanZ/`

## Decision

1. Replace the strict 1k stream census with `2,128,395,904` bytes/token.
2. Set the coupled backend allocation to `19.980 ms` device plus `0.100 ms`
   host, still bounded by `20.080 ms` wall and `49.80 tok/s`.
3. Require `>=106.525 GB/s` over the kernel window. The lower
   `105.994 GB/s` number is only the complete-wall stream floor.
4. Treat the `0.439..1.233 ms` remainder as a fusion budget, not permission for
   separate state, normalization, router, activation, or readback passes.

## Consequences

- Seq732/733 remain useful source/mechanism history but their byte totals are
  superseded by seq735/736.
- The product target is still physically open, but only a real fused stage port
  can validate the narrow remaining math margin.
- No product speedup is claimed. Seq730 consensus, native prefill,
  sentinel/smoothness, remaining denominator rows, and the full product matrix
  remain open.

Current gate state remains in
`doc/active/intel-qwen36-35b-a3b-gguf-q4km/STATUS.md`.
