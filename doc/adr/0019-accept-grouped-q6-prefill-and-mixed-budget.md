# ADR 0019: Accept grouped Q6 prefill and the mixed-layer budget

Date: 2026-07-11

## Status

Superseded by ADR 0021. The mixed timing sum remains feasibility evidence, but
the centered-U8 Q6 representation misses the component relative-L2 contract.

## Context

ADR 0018 established the real down-tensor split as 20 Q4_K plus 20 Q6_K and
required a grouped Q6 prefill carrier whose complete mixed-layer sum clears the
fixed 8k target. The M=1 exact-Q6 decode carrier cannot substitute for this
variable-M operation.

Seq669 converts each Q6_K down tensor offline into centered U8 values with F16
scales per 32-value group. The accepted kernel uses M16 DPAS work and a
four-byte compact task coordinate. One parameterized implementation covers the
same 256-expert shapes as the accepted grouped-Q4 runtime; there are no
per-layer source variants.

On real layer 7, standalone primary/confirm kernel minimum is
`3944.479 / 3919.062 us`. All `16,777,216` weighted-down values pass with zero
values above `5e-3`, max absolute error `0.000276752`, and cosine
`0.999987209`. The resident boundary also passes all SwiGLU, weighted-down, and
routed-output values; complete primary/confirm is `9868.056 / 9855.699 us`.
The application maps neither oneDNN nor OpenVINO.

The decisive ruler is the complete real schedule split, not the representative
layer. The fixed cap is `40 * 9771.436 = 390857.440 us`; its 0.5% noise guard is
`1954.287 us`. Primary/confirm results are:

| row | Q4 20-layer sum | Q6 20-layer sum | mixed sum | headroom |
|---|---:|---:|---:|---:|
| primary | 181165.888 us | 203500.115 us | 384666.003 us | 6191.437 us |
| confirm | 180838.384 us | 203853.871 us | 384692.255 us | 6165.185 us |

Both rows clear the cap by more than three times the noise guard.

## Decision

Accept the centered-U8/F16-scale M16 compact grouped-Q6 carrier and the mixed
40-layer prefill feasibility budget. Reuse the accepted grouped-Q4 carrier for
the other 20 down tensors and all gate/up tensors.

Advance the sole route to generation and resident loading of all 40 real layer
payloads, followed by live preceding-layer transformer state. Only after that
boundary passes may the project run teacher-forced distribution and
deterministic-token gates.

## Consequences

- The grouped-Q6 implementation and complete mixed schedule budget are closed;
  do not reopen Q6 M-tile, scale-width, task-table, workgroup, or GRF sweeps
  without a new integration failure or profile.
- All 40 real payloads have not yet been loaded together, and capture inputs
  have not yet been replaced by live transformer state.
- The hard product target remains 1.10x same-run OpenVINO independently in
  prefill and decode at every accepted bucket. Seq669 is not product speed
  evidence.

Evidence:

- `output/grouped-s8-u8-q6-prefill-gate-20260711Tseq669cleanZ/`
- `tools/intel-qwen36-grouped-s8-u8-q6-prefill-gate.py`
- `engine/gpu/opencl/grouped_s8_u8_q6_surrogate_down.cl`
- `engine/tools/grouped_s8_u8_q6_prefill_schedule_envelope_smoke.cpp`

Current gate state remains in
`doc/active/intel-qwen36-35b-a3b-gguf-q4km/STATUS.md`.
