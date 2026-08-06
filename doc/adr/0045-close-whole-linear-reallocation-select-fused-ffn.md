# ADR 0045: Close whole-linear reallocation and select one fused FFN gate

Date: 2026-07-12

## Status

Superseded by ADR 0046 after the source gate and a complete-boundary audit. No
FFN component or product speed row was accepted here.

## Context

Seq758 closes the only ADR-0044 non-state carrier. Its projection-only subset
is stable at `2036.839 / 2027.162 us`, already above the `1840 us` residual
before convolution/control/final normalization. Exact Q4 boundaries pass; the
only fast K32 S8 representation available for the Q6 QKV projection reaches
relative L2 `0.002080`, narrowly above the `0.002` contract. Per-16 exact-Q6
codegen and representation refinements were already closed by ADRs 0023-0030.

The failure also exposes a product-level composition gap at the headline 8k
target. One 1024-token tile has
`1024 / 2510 * 1e6 = 407968.127 us`. Conservatively composing the measured
frontiers already exceeds it:

- seq749 routed MoE boundary: `8948.390 * 40 = 357935.600 us`;
- seq753 slower F32 state plus seq758 slower projection subset:
  `(2211.041 + 2036.839) * 30 = 127436.400 us`;
- subtotal before convolution, full attention, normalization, and scheduling:
  `485372.000 us`, or `77403.873 us` above the entire tile budget.

Using only the same profile's explicitly attributed linear-convolution and
full-attention categories, divided by `1.10`, reserves another
`12118/1.10 + (10986+10373)/1.10 = 30433.636 us`. The remaining FFN budget is
`(407968.127 - 127436.400 - 30433.636) / 40 = 6252.452 us/layer`; round down
to `6250 us`. This still omits other work, so it is a necessary, not
sufficient, gate.

The then-audited layer-27 routed gate/up plus down work is about `25.77B` MACs.
After charging router/control/SwiGLU/scatter, the routed design appeared to
require `5.4 TMAC/s`. This calculation did not include the model's shared
expert and therefore was not a valid complete-FFN kill number; ADR 0046 records
the correction.

## Decision

1. Close whole-linear budget reallocation. Do not vary Q6 representation,
   projection fusion, codegen API, storage precision, tile, or workgroup shape.
2. Admit one materially different complete-FFN design for the fixed real
   layer-27/top-8 histogram: a resident expert-major persistent work queue that
   fuses gate/up, SwiGLU, down, router weighting, and scatter, without a
   materialized F16 SwiGLU plane or host control/readback.
3. Before source, the design must account for every matrix MAC, byte, task,
   synchronization, and resident buffer and project matrix phases at
   `>=5.4 TMAC/s` and the complete boundary at `<=6250 us`.
4. A source gate, if admitted, must compare all seq749 real outputs at cosine
   `>=0.999` and relative L2 `<=0.002`; repeat and confirm must each be
   `<=6250 us` and within `0.5%`.
5. Failure at design or runtime returns to product architecture reflection.
   It does not authorize codec, expert-bucket, tile, subgroup, workgroup, or
   synthetic-assignment sweeps.

## Consequences

- Seq749 remains the correct zero-readback grouped control/runtime reference,
  but its `8.948 ms` FFN cannot compose with measured linear lower bounds.
- A fused-FFN pass frees product budget; it does not solve the closed linear
  projection/GDN boundary or promote native prefill.
- Runtime oneDNN/OpenVINO linkage remains forbidden. Offline code generation
  is allowed only if the resulting program executes in the native runtime.

Evidence:

- `output/grouped-prefill-device-schedule-gate-20260712Tseq749cleanZ/`
- `output/linear-prefill-whole-stage-boundary-20260712Tseq757cleanZ/`
- `output/linear-prefill-nonstate-feasibility-20260712Tseq758cleanZ/`

Current gate state remains in
`doc/active/intel-qwen36-35b-a3b-gguf-q4km/STATUS.md`.

## Follow-Up

- Commit `a5e7623` adds the evidence-only design gate. Clean seq759 confirms
  the real histogram has `12,896 / 10,224 / 9,120` padded rows at M32/M16/M8.
  M8 removes `29.3%` of padded work.
- Its fixed `28.689B` matrix MACs take `5312.785 us` at the registered
  `5.4 TMAC/s`; seq749 router/schedule plus scatter projects the complete
  boundary to `5939.351 us`, leaving `310.649 us`. The required rate uplift
  over the existing 256-GRF kernels is `10.21%`.
- Admit one compiler/rate preflight with M8, gate/up N32, down N64, 16 KiB SLM,
  96 persistent workgroups, 128 GRFs/eight EU threads, and no sweep flags.
- Clean seq761 compiles correctly at SIMD16/96 GRFs/ten EU threads and passes
  the routed numeric oracle, but reaches only `4.453 / 4.390 TMAC/s` with
  `1.428%` spread. Its `6442.333 / 6534.345 us` matrix rows activate the
  terminal no-sweep rejection.
- The audit also proves that seq749/759/761 end at `ffn_moe_out`. Restoring the
  1024-token shared expert adds `3.221B` MACs; the fixed-M8 design would need
  `5.675 TMAC/s` and projects to `6535.874 us` at the old floor, above the
  `6250 us` cap. ADR 0046 supersedes this routed-only admission.
