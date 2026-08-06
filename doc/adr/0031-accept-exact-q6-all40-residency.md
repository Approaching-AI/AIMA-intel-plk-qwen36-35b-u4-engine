# ADR 0031: Accept exact-Q6 all-40 residency

Date: 2026-07-11

## Status

Accepted for payload generation, capacity, and native runtime ownership only.
Component, live-state, token, context, and product-speed gates remain open.

## Context

ADR 0030 selected the correct slow R2 carriers after the fast S8 Q6 codec
family closed. The existing seq670 capacity proof held centered-U8 surrogate
payloads and therefore could not establish residency for the selected exact
representation.

Commit `374af88` adds an exact-Q6 resident down kind, an exact K16 source-sum
stage, and dispatch of the existing exact per-16 OpenCL kernel. Each Q6 layer
now owns all signed Q6 values recentered in U8 plus F32 per-16 scales. The Q4
side reuses the byte-identical clean seq670 gate/up and Q4 down payloads already
accepted by seq673.

Clean seq685 generates all 20 exact-Q6 down payloads from the locked GGUF and
records every file size and SHA-256. The resulting all-layer set contains:

- 20 Q4 layers at `541,065,216` bytes each;
- 20 exact-Q6 layers at `696,254,464` bytes each;
- `24,746,393,600` total resident payload bytes.

One native OpenCL context loads four programs and 40 sequential real-layer
handles. The process maps no oneDNN or OpenVINO runtime library.

Evidence:

- `output/all-layer-exact-q6-prepack-load-20260711Tseq685cleanZ/`
- `output/all-layer-mixed-prepack-load-20260711Tseq670cleanZ/`
- `output/q6-exact-per16-prefill-gate-20260711Tseq675cleanZ/`

## Decision

Accept capture-free exact-Q6 payload generation and single-context residency
for the selected R2 carrier. Advance directly to one live-capture all-40
component execution using seq673 Q4 and exact per-16/F32-input Q6.

The component gate must compare every per-layer and aggregate SwiGLU,
weighted-down, and routed-output boundary under finite, cosine `>=0.999`, and
relative L2 `<=0.002`. The runtime must retain one context, four programs, 40
real handles, `24,746,393,600` resident bytes, and native-only process maps.

## Consequences

- Exact payload size, distinctness, and all-40 resident capacity need not be
  reprobed unless the model or exact representation changes.
- This accepts no kernel timing. Seq675's exact-Q6 performance rejection
  remains in force.
- Live-state chaining and the teacher-forced/token ladder remain blocked on the
  all-40 component result.

## Follow-Up

- Run `tools/intel-qwen36-all-layer-mixed-component-gate.py` once from a clean
  commit against seq685 payloads.
- If every layer passes, integrate the same handles into the live transformer
  loop rather than revisiting payload or codec variants.

Current gate state remains in
`doc/active/intel-qwen36-35b-a3b-gguf-q4km/STATUS.md`.
