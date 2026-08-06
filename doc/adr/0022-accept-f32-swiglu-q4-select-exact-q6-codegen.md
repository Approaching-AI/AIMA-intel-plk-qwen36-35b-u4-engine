# ADR 0022: Accept F32 SwiGLU Q4 and select exact-Q6 codegen

Date: 2026-07-11

## Status

Accepted as the Q4 component correction and the next bounded Q6 route. No
teacher-forced, token, context-ladder, or product promotion is implied.

## Context

Seq671 showed every grouped Q4/Q6 down component missing the locked relative-L2
ceiling `0.002`. Seq675 isolates two independent causes. With exact Q6_K values
and F32 per-16 scales, the worst layer-39 down comparison reaches relative L2
`0.000206724`; rounding the same input through the prior F16 SwiGLU boundary
alone raises it to `0.00222462`. Thus F16 handoff is itself promotion-blocking.

Seq673 changes the native gate/up binary to emit an auxiliary F32 SwiGLU plane
before down quantization, while retaining F16 contributions and a runtime with
no oneDNN/OpenVINO mapping. On real Q4 layer 27:

| row | complete time | SwiGLU relL2 | weighted-down relL2 | routed relL2 |
|---|---:|---:|---:|---:|
| primary | `9302.607 us` | `0.000222649` | `0.000647684` | `0.000396234` |
| confirm | `9389.725 us` | `0.000222649` | `0.000647684` | `0.000396234` |

Both rows are below the fixed `9771.436 us` layer cap and all component contract
checks pass.

Exact Q6 per-16 does not satisfy the timing contract in the handwritten M16
mapping. Seq675 measures `6630.000 / 6629.895 us` on layer 39 and projects the
complete mixed window to `435070.930 / 434995.073 us`, exceeding the
`390857.440 us` cap by `44213.490 / 44137.633 us`. The prior centered-U8 route
is fast but inaccurate; exact per-16 is accurate with F32 input but too slow.

Evidence:

- `output/grouped-s8-u4-prefill-gate-20260711Tseq673cleanZ/`
- `output/q6-exact-per16-prefill-gate-20260711Tseq675cleanZ/`

## Decision

Accept the auxiliary F32 SwiGLU plane as the grouped-Q4 correctness carrier.
Use seq673's gate/up binary as the default for subsequent all-layer gates.

Reject the current handwritten exact-Q6 M16 mapping as a product timing carrier,
but preserve it as the exact accuracy reference. Select one materially different
successor: offline-generate a grouped S8-by-U8 exact-Q6/per-16 native binary with
oneDNN codegen, extract the binary, and execute it from the pure OpenCL runtime.
The fixed worst-layer down-only gate is `<=4316.404 us`, derived from the clean
mixed-window cap after retaining the `1954.287 us` noise guard. It must also pass
all `16,777,216` values at finite, cosine `>=0.999`, relative L2 `<=0.002`.

This does not authorize a oneDNN/OpenVINO runtime dependency, a threshold
relaxation, or a workgroup/precision/shape sweep.

## Consequences

- Q4 component accuracy is closed; rerun all 20 Q4 layers only as part of the
  next all-40 confirmation, not as isolated variants.
- F16 SwiGLU handoff and centered-U8 Q6 are rejected correctness routes.
- Handwritten exact-Q6 M16 is rejected on complete mixed timing. Reopen it only
  with a new source-derived mechanism capable of at least `1.536x` layer-39
  movement (`6630 / 4316.404`).
- Live preceding-state chaining remains blocked on the Q6 carrier.

## Follow-Up

- Run exactly one offline-generated grouped S8-by-U8 exact-Q6/per-16 native
  component under the fixed `4316.404 us` gate.
- If it passes, integrate it into the resident loop and rerun all 40 components;
  if it fails, record the terminal measured gap before selecting a different
  representation family.

Current gate state remains in
`doc/active/intel-qwen36-35b-a3b-gguf-q4km/STATUS.md`.
