# ADR 0023: Close exact-Q6 codegen and select S8 per-32 requantization

Date: 2026-07-11

## Status

Accepted as a component-route rejection and one bounded successor selection.
No teacher-forced, token, context-ladder, or product promotion is implied.

## Context

ADR 0022 selected one offline-generated grouped exact-Q6/per-16 binary after
the correct handwritten mapping missed its fixed timing gate. Clean seq676
tests the two legal integer encodings against the worst captured Q6 layer 39.

The selected S8-by-U8 form stores `q + 32` and supplies a per-16 zero point of
32. The pinned oneDNN generator faults during primitive construction in both
primary and confirm processes (`SIGSEGV`, return code `-11`), before a binary
can be produced. The zero-point-free S8-by-S8 fallback does generate and its
raw core clears the `4316.404 us` gate at `3146.703 / 3121.195 us`, but it is
not numerically the requested operation. The generated K32 DPAS applies the
first per-16 weight scale to both halves of every pair. On a 64-value compiler
probe, GPU versus that effective per-32 calculation differs by only
`1.175e-7`, while GPU versus exact per-16 differs by `0.122409`. The full
`16,777,216`-value comparison consequently has cosine `0.955397` and relative
L2 `0.297616` in both runs.

Evidence:

- `output/onednn-grouped-q6-exact-preflight-gate-20260711Tseq676cleanZ/`
- `output/q6-exact-per16-prefill-gate-20260711Tseq675cleanZ/`

## Decision

Close the pinned-oneDNN grouped exact-Q6/per-16 codegen route on the current
PTL software stack. Do not sweep workgroups, integer signedness, zero-point
types, scale layouts, or JIT strategies around this failure.

Select exactly one materially different representation probe: offline
requantize each dequantized Q6_K weight K32 group to symmetric S8 with one F32
scale per 32 values. Keep the accepted F32 SwiGLU input and per-256 source
scales. On layer 39 it must compare all `16,777,216` outputs at finite, cosine
`>=0.999`, relative L2 `<=0.002`, while the generated raw core remains
`<=4316.404 us`.

This does not relax the component contract, authorize runtime oneDNN/OpenVINO
linkage, or make a product speed claim. It changes the offline weight carrier
to match the measured K32 scale capability; correctness remains the arbiter.

## Consequences

- Exact Q6_K per-16 remains the CPU/handwritten accuracy reference, not the
  generated runtime carrier.
- U8 zero-point and zero-point-free S8 exact-per-16 codegen are closed on this
  pinned compiler and device.
- The S8-per-32 probe is a single fixed representation. Do not vary group size,
  scale precision, requantization rule, workgroup, tile, or shape.
- A passing raw preflight still requires binary extraction and pure-OpenCL
  complete-boundary timing before all-40 integration.

## Follow-Up

- Run the fixed S8-per-32 layer-39 all-value/raw-core gate once.
- If it passes, extract the generated binary and enforce the same accuracy and
  timing gate in the pure OpenCL runtime. If it fails, record the terminal gap
  before selecting another representation family.

Current gate state remains in
`doc/active/intel-qwen36-35b-a3b-gguf-q4km/STATUS.md`.
