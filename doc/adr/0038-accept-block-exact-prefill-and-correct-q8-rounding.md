# ADR 0038: Accept block-exact prefill and correctly rounded Q8

Date: 2026-07-11

## Status

Accepted

## Context

The shared-prefix ladder showed that the earlier eight-token passes were false
confidence. Seq698-700 retained the expected top-1 IDs in several codec-isolated
rows while position-2 KLD was `2.87..6.60`. Prefix and complement probes
seq701-706 showed that even a one-layer routed-MoE delta could move that
position onto a different trajectory.

Reference-boundary probes isolated two arithmetic gaps. Replacing native down
with the captured reference made the continuation exact; block-exact Q6
accumulation then reduced the direct component to relative L2 `1.15e-7`.
Seq708-711 accepted its payload/residency and restored the eight expected IDs,
but position-2 KLD remained `4.919957`.

The Q4 successor preserves Q4_K scale/min codes and block `d/dmin`, accumulates
integer products per K256 block, and performs one FMA per block. Clean seq712
loads all 40 real layers in one native context using `21,726,494,720` resident
bytes. Clean seq713 passes all `167,772,160` SwiGLU, `671,088,640` weighted-down,
and `83,886,080` routed values. Seq714 still fails the teacher ladder.

The remaining avoidable gap was OpenCL division accuracy in Q8_K quantization.
The correctly-rounded divide option changes a reference-SwiGLU diagnostic from
69 Q8 code mismatches and 206,816 scale-bit mismatches to zero of each. Clean
seq720 records aggregate SwiGLU/routed relative L2
`2.4386e-7 / 8.3445e-5`; clean seq721 nevertheless follows a different
position-2 trajectory.

Evidence:

- `output/all-layer-exact-block-q4q6-prepack-load-20260711Tseq712cleanZ/`
- `output/all-layer-exact-block-q4q6-component-20260711Tseq713cleanZ/`
- `output/all-layer-teacher-forced-exact-block-q4q6-20260711Tseq714cleanZ/`
- `output/all-layer-exact-block-q4q6-crdiv-component-20260711Tseq720cleanZ/`
- `output/all-layer-teacher-forced-exact-q4q6-crdiv-20260711Tseq721cleanZ/`

## Decision

Accept block-exact Q4/Q6 payloads and correctly rounded Q8_K quantization as
the R2 numeric-correctness carrier. Do not promote it as a product or token
pass: the layer-27 exact-Q4 boundary is about `20 ms`, above the corrected
`9.526 ms` product cap, and the llama.cpp teacher ladder still diverges.

Close further scale-code, contribution-width, and approximate-codec sweeps for
this failure. The next decision must first establish whether the requested
multi-reference exact-token predicate is internally consistent.

## Consequences

- Block-exact payload generation and the one-context 40-layer runtime are
  accepted implementation capabilities.
- Seq720 is the numeric component prerequisite; seq714/721 are rejection
  evidence, not speed or token claims.
- Runtime linkage remains free of oneDNN/OpenVINO; oneDNN is offline codegen
  only.

## Follow-Up

- Measure llama.cpp and OpenVINO greedy IDs from the exact same token payload.
- Select a consensus-aware token policy before another arithmetic repair.

Current gate state remains in
`doc/active/intel-qwen36-35b-a3b-gguf-q4km/STATUS.md`.
